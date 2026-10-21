"""Analyze the reference-only exp3.3 CSV/video, including control identities."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    run = json.loads((HERE / "zero_residual_latest.json").read_text())["run"]
    output = HERE / ("zero_residual_" + run)
    summary = json.loads((output / "server_summary.json").read_text())
    limits = {joint.get("name"): (float(joint.find("limit").get("lower")), float(joint.find("limit").get("upper")))
              for joint in ET.parse(ROOT / "resources/robots/x1/urdf/X1_29DOF_physically_mirrored.urdf").findall("joint")
              if joint.get("type") == "revolute"}
    metrics = {}
    for label, case in summary["cases"].items():
        if not case["success"]:
            continue
        frame = pd.read_csv(output / label / "isaac_diag.csv")
        names = [name.removeprefix("ref_pos_") for name in frame if name.startswith("ref_pos_")]
        values = lambda prefix: frame[[prefix + name for name in names]].to_numpy()
        reference, actual, target = values("ref_pos_"), values("pos_"), values("pd_target_effective_")
        delta, commanded = values("delta_q_"), values("pd_target_commanded_")
        error = actual - reference
        violation = sum(int(((target[:, i] < limits[name][0] - 1e-5)
                             | (target[:, i] > limits[name][1] + 1e-5)).sum()) for i, name in enumerate(names))
        stages = []
        for index, segment in frame.groupby("active_stage"):
            steady = segment.iloc[min(100, max(0, len(segment) - 1)):]
            stages.append(dict(stage=int(index), requested=float(segment.command_requested.iloc[0]),
                               applied_mean=float(segment.command_applied.mean()),
                               vx_mean=float(segment.base_vel_x.mean()), vx_steady_mean=float(steady.base_vel_x.mean()),
                               resets=int(segment.done.sum()),
                               roll_peak_deg=float(np.degrees(segment.base_euler_x.abs().max())),
                               pitch_peak_deg=float(np.degrees(segment.base_euler_y.abs().max()))))
        preterminal = frame[frame.done != 0]
        result = dict(rows=len(frame), columns=len(frame.columns), finite=bool(np.isfinite(frame.select_dtypes("number")).all().all()),
                      duration_s=float(frame.time_s.iloc[-1] + .01), resets=int(frame.done.sum()),
                      first_termination_s=float(preterminal.time_s.iloc[0]) if len(preterminal) else None,
                      delta_q_abs_max_rad=float(np.max(np.abs(delta))),
                      target_reference_abs_max_rad=float(np.max(np.abs(target - reference))),
                      commanded_identity_abs_max_rad=float(np.max(np.abs(commanded - reference - delta))),
                      observation_command_abs_error_max=float((frame.command_observed - frame.command_applied).abs().max()),
                      physical_target_violations=violation,
                      reference_joint_rms_rad=float(np.sqrt(np.square(error).mean())),
                      reference_joint_l2_mean_rad=float(np.linalg.norm(error, axis=1).mean()),
                      pd_target_clip_fraction=float(frame.pd_target_clip_joint_substeps.sum() / (len(frame) * 29 * 10)),
                      torque_clip_joint_substeps=int(frame.torque_clip_step.sum()),
                      roll_peak_deg=float(np.degrees(frame.base_euler_x.abs().max())),
                      pitch_peak_deg=float(np.degrees(frame.base_euler_y.abs().max())),
                      sole_height_min_m=[float(frame.sole_height_l.min()), float(frame.sole_height_r.min())],
                      stages=stages,
                      joint_stats=sorted([dict(joint=name, rms_error_rad=float(np.sqrt(np.square(error[:, i]).mean())),
                                                max_error_rad=float(np.abs(error[:, i]).max())) for i, name in enumerate(names)],
                                         key=lambda item: -item["rms_error_rad"]))
        cap = cv2.VideoCapture(str(output / label / "play_output.mp4"))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_count = round(result["duration_s"] * 50)
        if not cap.isOpened() or abs(fps - 50) > .01 or count != expected_count:
            raise RuntimeError("bad video timing: " + label)
        snapshots = []
        for i, second in enumerate(([0.2, 0.8, 1.4, 1.61] if label == "stand10" else [0, 6, 11, 16, 21, 26])):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, round(second * fps)))
            success, image = cap.read()
            if not success:
                raise RuntimeError("video decode failed: " + label)
            tile = cv2.resize(image, (640, 360))
            cv2.putText(tile, label + " t=" + str(second) + "s", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            snapshots.append(tile)
        cap.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
        if not cap.read()[0]:
            raise RuntimeError("last video frame cannot be decoded")
        cap.release()
        columns = 2 if label == "stand10" else 3
        montage = np.vstack([np.hstack(snapshots[i:i + columns]) for i in range(0, len(snapshots), columns)])
        cv2.imwrite(str(output / (label + "_montage.jpg")), montage)
        result["video"] = dict(fps=fps, frames=count, seconds=count / fps, decoded_first_middle_last=True)
        metrics[label] = result

        figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(frame.time_s, frame.command_requested, label="requested vx")
        axes[0].plot(frame.time_s, frame.command_applied, label="applied vx", linestyle="--")
        axes[0].plot(frame.time_s, frame.base_vel_x, label="actual vx")
        axes[0].set_ylabel("m/s")
        axes[1].plot(frame.time_s, np.degrees(frame.base_euler_x), label="roll")
        axes[1].plot(frame.time_s, np.degrees(frame.base_euler_y), label="pitch")
        axes[1].set_ylabel("deg")
        axes[2].plot(frame.time_s, frame.base_height, label="base height")
        axes[2].plot(frame.time_s, frame.sole_height_l, label="left sole")
        axes[2].plot(frame.time_s, frame.sole_height_r, label="right sole")
        axes[2].set_ylabel("m")
        axes[3].plot(frame.time_s, np.sqrt(np.square(error).mean(axis=1)), label="joint reference RMS")
        axes[3].set_ylabel("rad")
        axes[3].set_xlabel("simulation time (s)")
        for axis in axes:
            for t in preterminal.time_s:
                axis.axvline(t, color="red", alpha=.3)
            axis.grid(alpha=.2)
            axis.legend(loc="upper right", fontsize=8)
        figure.suptitle("exp3.3 zero-residual: " + label + "; red lines = physical terminations")
        figure.tight_layout()
        figure.savefig(output / (label + "_signals.png"), dpi=130)
        plt.close(figure)
        print(label, json.dumps({key: value for key, value in result.items() if key not in ("joint_stats", "stages")}))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# exp3.3 零残差回放结果", "", "日期：2026-10-21。平台：201.5 / RTX A6000 / F1 / Isaac Gym / Torch 1.12.1+cu113。",
             "", "源为当前工作区 exp3.3，逐文件 SHA256 见 manifest.json；不是基于旧提交归档的回放。无 checkpoint、无训练，delta_q 固定为零。",
             "", "## 结果", "", "| 检查 | 10秒站立 | 30秒六段速度 |", "| --- | ---: | ---: |"]
    for title, key, fmt in (("物理终止次数", "resets", "{}"), ("首次终止（秒）", "first_termination_s", "{}"),
                            ("关节参考 RMS（rad）", "reference_joint_rms_rad", "{:.5f}"),
                            ("最大绝对残差（rad）", "delta_q_abs_max_rad", "{:.1e}"),
                            ("目标与参考最大差（rad）", "target_reference_abs_max_rad", "{:.1e}"),
                            ("PD目标裁剪比例", "pd_target_clip_fraction", "{:.2%}"),
                            ("力矩限幅关节×子步数", "torque_clip_joint_substeps", "{}"),
                            ("俯仰绝对峰值（度）", "pitch_peak_deg", "{:.2f}")):
        values = [fmt.format(metrics[case][key]) if case in metrics else "待完成" for case in ("stand10", "profile30")]
        lines.append("| " + title + " | " + " | ".join(values) + " |")
    if "profile30" in metrics:
        lines += ["", "## 六段实际指令与速度", "", "| 段 | 请求 vx | 实际施加指令均值 | 实际 vx 均值 | 终止次数 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for stage in metrics["profile30"]["stages"]:
            lines.append("| {stage} | {requested:+.2f} | {applied_mean:+.3f} | {vx_mean:+.3f} | {resets} |".format(**stage))
        lines += ["", "重置后 play.py 会强制1.5秒站立缓冲；反复跌倒使部分行走请求不能持续施加。不能把本回放当作已完成稳定前后步态验收。"]
    lines += ["", "## 判断", "", "- 控制接线已验证：零残差、参考与PD目标相等、无目标限幅；CSV为终止前快照，数值有限。",
              "- 站立实际关节角大部分接近参考，但整机逐渐前倾。第一episode左/右踝pitch参考误差RMS约0.131/0.130 rad；全身RMS会稀释这一局部问题。",
              "- 离线几何检查的初始脚底法向倾角约0.75°，重心投影位于估算接触区域内，边界余量约61mm。该模型惯量与Isaac名义base COM覆盖可能有差异，不能作为精确接触平衡证明。",
              "- 这支持固定参考＋当前PD缺少自由基座的平衡修正；未发现本次失败由目标限位或力矩饱和造成。也不能据零残差跌倒判定v1.2轨迹不可用。",
              "- 先研究站立负载下踝部跟踪、重心/COP和PD参数，或进行有界残差的站立短训练，建立平衡后再评估完整步态；不直接启动6000轮长训。",
              "", "## 产物", "", "此处为零残差诊断，不生成训练模型PT。CSV/MP4保存在本分析目录，避免在训练三件套目录伪造模型文件。"]
    for case in metrics:
        lines += ["", "- " + case + ": [MP4](" + case + "/play_output.mp4)、[CSV](" + case + "/isaac_diag.csv)",
                  "- [信号图](" + case + "_signals.png)、[姿态截图](" + case + "_montage.jpg)"]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
