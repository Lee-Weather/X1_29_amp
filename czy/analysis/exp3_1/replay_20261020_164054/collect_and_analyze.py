"""Collect finished server replays, validate artifacts and summarize the CSV."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SERVER = "robot@10.12.201.5:"


def fetch(source, destination, recursive=False):
    command = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if recursive:
        command.append("-r")
    command += [SERVER + source, str(destination)]
    subprocess.run(command, check=True, capture_output=True)


def analyze(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) != 3000 or len(df.groupby("active_stage")) != 6:
        raise RuntimeError("incomplete six-stage replay")
    if not np.isfinite(df.select_dtypes(include="number").to_numpy()).all():
        raise RuntimeError("non-finite CSV values")
    dt = .01
    command_error = float((df.command_applied - df.command_observed).abs().max())
    if command_error > 1e-6:
        raise RuntimeError("actor observation and applied command disagree")
    stages = []
    for index, part in df.groupby("active_stage", sort=True):
        requested = float(part.command_requested.iloc[0])
        valid = part[(part.time_s >= float(part.time_s.iloc[0]) + 1) &
                     np.isclose(part.command_applied, requested) & (part.done == 0)]
        stage = dict(stage=int(index), requested=requested, rows=len(part),
            mean_vx_all=float(part.base_vel_x.mean()), resets=int(part.done.sum()),
            applied_fraction=float(np.isclose(part.command_applied, requested).mean()),
            steady_rows=len(valid), min_height=float(part.base_height.min()))
        if len(valid):
            stage.update(steady_vx=float(valid.base_vel_x.mean()), median_vx=float(valid.base_vel_x.median()),
                vx_rmse=float(np.sqrt(((valid.base_vel_x - requested) ** 2).mean())),
                mean_vy=float(valid.base_vel_y.mean()), vy_rms=float(np.sqrt((valid.base_vel_y ** 2).mean())),
                roll_rms_deg=float(np.degrees(np.sqrt((valid.base_euler_x ** 2).mean()))),
                pitch_rms_deg=float(np.degrees(np.sqrt((valid.base_euler_y ** 2).mean()))))
            if requested:
                stage["direction_fraction"] = float((valid.base_vel_x * requested > 0).mean())
            for column in ("amp_fused", "amp_style_contribution", "amp_task_contribution", "amp_direction_gate", "amp_disc_score"):
                if column in valid:
                    stage[column] = float(valid[column].mean())
        stages.append(stage)
    terminations = []
    for _, row in df[df.done != 0].iterrows():
        terminations.append(dict(time_s=float(row.time_s), stage=int(row.active_stage),
            requested=float(row.command_requested), applied=float(row.command_applied),
            height=float(row.base_height), roll_deg=float(np.degrees(row.base_euler_x)),
            pitch_deg=float(np.degrees(row.base_euler_y)), vx=float(row.base_vel_x),
            causes=[name.removeprefix("termination_") for name in (
                "termination_height", "termination_roll", "termination_pitch", "termination_contact") if row[name]],
            timeout=bool(row.timeout)))
    errors = {}
    for column in df.columns:
        if column.startswith("ref_pos_"):
            joint = column[len("ref_pos_"):]
            errors[joint] = float(np.sqrt(((df["pos_" + joint] - df[column]) ** 2).mean()))
    ref_columns = [c for c in df if c.startswith("ref_pos_")]
    qdiff = np.column_stack([df["pos_" + col[len("ref_pos_"):]] - df[col] for col in ref_columns])
    episodes = df.groupby("episode_id").size()
    result = dict(rows=len(df), duration_s=len(df) * dt, command_observed_error=command_error,
        resets=len(terminations), timeouts=int(df.timeout.sum()),
        first_failure_s=terminations[0]["time_s"] if terminations else None,
        longest_continuous_segment_s=int(episodes.max()) * dt,
        min_height=float(df.base_height.min()), max_roll_deg=float(np.degrees(df.base_euler_x.abs().max())),
        max_pitch_deg=float(np.degrees(df.base_euler_y.abs().max())),
        max_episode_yaw_drift_deg=float(np.degrees(df.yaw_from_episode_start.abs().max())),
        torque_clip_count=int(df.clip_count.iloc[-1]), action_clip_count=int(df.action_clip_step.sum()),
        ref_q_l2_mean=float(np.linalg.norm(qdiff, axis=1).mean()),
        largest_joint_tracking_rms=sorted(errors.items(), key=lambda pair: pair[1], reverse=True)[:8],
        terminations=terminations, stages=stages)
    return result


def main():
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    remote = manifest["server_root"]
    fetch(remote + "/replay_summary.json", HERE / "server_summary.json")
    server = json.loads((HERE / "server_summary.json").read_text())
    analyses = {}
    for name, case in server["cases"].items():
        if not case.get("success"):
            continue
        letter, mode = name.split("_", 1)
        label = "nominal" if mode == "nominal_no_push" else "randomized"
        destination = ROOT / ("czy/data/exp3.1" + letter) / ("replay_" + manifest["run"] + "_" + label)
        destination.mkdir(parents=True, exist_ok=True)
        fetch(case["directory"] + "/isaac_diag.csv", destination / "isaac_diag.csv")
        fetch(case["directory"] + "/play_output.mp4", destination / "play_output.mp4")
        fetch(remote + "/replay_logs/" + name + "_metadata.json", HERE / (name + "_metadata.json"))
        fetch(remote + "/replay_logs/" + name + ".log", HERE / (name + ".log"))
        model = manifest["experiments"][letter]
        shutil.copyfile(model["local_source"], destination / model["model"])
        if hashlib.sha256((destination / model["model"]).read_bytes()).hexdigest() != model["sha256"]:
            raise RuntimeError("archived model checksum mismatch")
        result = analyze(destination / "isaac_diag.csv")
        video = cv2.VideoCapture(str(destination / "play_output.mp4"))
        fps, frames = video.get(cv2.CAP_PROP_FPS), int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if not video.isOpened() or abs(fps - 50) > .01 or frames != 1500:
            raise RuntimeError("unexpected video timing")
        for frame in (0, frames - 1):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, _ = video.read()
            if not ok:
                raise RuntimeError("cannot decode video frame")
        video.release()
        result.update(directory=str(destination), checkpoint=model["model"], fps=fps, frames=frames,
                      video_duration_s=frames / fps, files={p.name: p.stat().st_size for p in destination.iterdir()})
        if len(result["files"]) != 3:
            raise RuntimeError("result directory must contain only model, MP4 and CSV")
        analyses[name] = result
        print("REPLAY_RESULT", name, "resets", result["resets"], "first_failure", result["first_failure_s"],
              "steady_vx", [round(s.get("steady_vx", 0.), 3) for s in result["stages"]], flush=True)
    (HERE / "metrics.json").write_text(json.dumps(analyses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["collected_cases"] = list(analyses)
    manifest["all_replays_finished"] = server["finished"]
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("COLLECTED", len(analyses), "cases; batch_finished", server["finished"], flush=True)


if __name__ == "__main__":
    main()
