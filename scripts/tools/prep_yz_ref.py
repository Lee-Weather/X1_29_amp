#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""exp1.6 预处理：yz CSV（真实地面直线行走）→ ref_lib.pt 第 4 段 walk_yz

与 prep_mocap_ref.py（gmr pkl → 3 段）的差异（czy/exp1/exp1.md §16）：
  ① 来源   yz CSV 30Hz 37 列（root quat xyzw；关节列名 = Isaac Gym 序；缺 wrist_roll×2；
           多 neck/head×2 丢弃）vs gmr pkl 120Hz 自带 dof_names
  ② 翻转   yz 数据已是 Isaac 序语义角，无需 FLIP_JOINTS 翻转（gmr 右侧 6 关节需取反）
  ③ 校验   无 body_positions 可 FK 对拍 → 替代校验：root z 稳定 / 无越限 / 帧间跳变
  ④ 重采样 30→50Hz 上采样（复用 prep_mocap_ref.resample 线性插值）
  ⑤ detrend yz 是直线行走非 loop：只对关节角做首尾线性 detrend（loop 缝），
           root_pos 保留真实前进位移不 detrend；loop 缝检查降级为 WARN 不 assert

用法（在 ref_lib.pt 已存在三段的基础上追加，不改旧段）：
  python scripts/tools/prep_yz_ref.py            # 追加 walk_yz
  python scripts/tools/prep_yz_ref.py --dry-run  # 只跑校验不入库
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from prep_mocap_ref import (  # noqa: E402
    ISAAC_GYM_DOF_ORDER, gait_period_autocorr, phase_anchor, resample,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
YZ_CSV = os.path.join(ROOT, "czy/diff/yz",
                      "07_03_walk_yup_recwalk_base_lowerbody_smooth_p8_120_180_groundfit_minima_safe.csv")
OUT = os.path.join(ROOT, "resources/motions/processed/ref_lib.pt")

SEG_NAME = "walk_yz"
FPS_SRC, FPS_DST = 30.0, 50.0
# 直线行走：掐头去尾留稳态（首尾各 ~1s 过渡）
T_START, MARGIN = 1.0, 1.0
# wrist_roll yz 未采集（x1 无该传感器）→ 填 0（default 位姿；回放样本里 wrist 本就近静止 ±0.02）
FILL_DEFAULT = {"left_wrist_roll_joint": 0.0, "right_wrist_roll_joint": 0.0}
DROP_JOINTS = ("neck_motor_base_pitch_joint", "head_face_bracket_pitch_joint")  # x1_29 无此执行器

# URDF 限位外裁剪容差（与 prep_mocap_ref clip 同义，无 mujoco 时用宽界校验）
SANITY_ABS_LIMIT = 3.2   # rad 超此值视为数据异常
FRAME_JUMP_LIMIT = 0.15  # rad 相邻帧跳变上限（预检实测最大 0.062）


def load_yz():
    rows = list(csv.reader(open(YZ_CSV)))
    hdr = rows[0]
    data = np.asarray([[float(x) for x in r] for r in rows[1:]], dtype=np.float64)
    ix = {n: i for i, n in enumerate(hdr)}
    joints = [c for c in hdr if c.endswith("_joint")]
    print(f"  ①CSV: {data.shape[0]}帧@{FPS_SRC}Hz {data.shape[1]}列, 关节列 {len(joints)}")

    # 剔除 env 无关执行器的关节
    drop = [j for j in joints if j in DROP_JOINTS]
    print(f"  ②丢弃 x1_29 无执行器关节: {drop}")

    # 重排到 Isaac Gym 序 + 补缺失关节
    q = np.zeros((data.shape[0], len(ISAAC_GYM_DOF_ORDER)))
    missing = []
    for k, name in enumerate(ISAAC_GYM_DOF_ORDER):
        if name in ix:
            q[:, k] = data[:, ix[name]]
        elif name in FILL_DEFAULT:
            q[:, k] = FILL_DEFAULT[name]
            missing.append(name)
        else:
            raise KeyError(f"yz 缺关节且无默认值规则: {name}")
    print(f"  ②重排→Isaac Gym 序；补默认值关节: {missing}")

    root_pos = data[:, [ix["root_pos_x"], ix["root_pos_y"], ix["root_pos_z"]]]
    root_quat_xyzw = data[:, [ix["root_quat_x"], ix["root_quat_y"],
                              ix["root_quat_z"], ix["root_quat_w"]]]
    root_rot_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]  # xyzw → wxyz
    # 归一化四元数（interp 后可能失模）
    root_rot_wxyz = root_rot_wxyz / np.linalg.norm(root_rot_wxyz, axis=1, keepdims=True)
    return q, root_pos, root_rot_wxyz


def sanity_check(q, root_pos):
    """替代 FK 对拍的三项轻量校验"""
    # ① 关节角绝对值界
    mx = np.abs(q).max()
    assert mx < SANITY_ABS_LIMIT, f"关节角异常大 {mx:.2f} rad"
    # ② 相邻帧跳变
    jmp = np.abs(np.diff(q, axis=0)).max()
    assert jmp < FRAME_JUMP_LIMIT, f"相邻帧跳变 {jmp:.3f} rad 超阈值"
    # ③ root z 稳定（行走段不塌不跳）
    z = root_pos[:, 2]
    assert 0.50 < z.min() and z.max() < 0.75, f"root z 异常 [{z.min():.3f},{z.max():.3f}]"
    print(f"  ③校验: |q|max={mx:.2f} 帧跳max={jmp:.3f}rad root_z=[{z.min():.3f},{z.max():.3f}] [OK]")


def yz_gait_period(x, fps):
    """④a yz 版自相关求步态周期：搜索窗 1.0~6.0s（gmr 版 0.5~2.0s 装不下 yz 的 ~4.8s 完整循环）。

    左髋 pitch 在左右腿交替期信号里左右步等幅混叠 → 自相关最近峰 = 完整步态循环
    （左右各一步，实测 ~4.8s）。与 ref_lib 三段 gait_period 语义一致（相位时钟
    一圈 = 左右各一步），phase sin>=0 左支撑的锚点约定不变。
    """
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    lo, hi = int(1.0 * fps), int(6.0 * fps)
    seg = ac[lo:hi]
    peaks = [i + lo for i in range(1, len(seg) - 1)
             if seg[i] > seg[i - 1] and seg[i] > seg[i + 1] and seg[i] > 0.3]
    assert peaks, "自相关无显著峰（1~6s 窗）"
    p = peaks[0]
    y0, y1, y2 = ac[p - 1], ac[p], ac[p + 1]
    dp = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return (p + dp) / fps, ac[p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只跑校验与统计，不写 ref_lib.pt")
    args = ap.parse_args()

    lib = torch.load(OUT, map_location="cpu") if not args.dry_run else {}
    if args.dry_run:
        print("[DRY-RUN] 不写库，仅校验")
    print(f"===== {SEG_NAME} ← yz CSV =====")
    q, root_pos, root_rot = load_yz()
    sanity_check(q, root_pos)

    # 周期（整段 30Hz 上自相关，左髋 pitch——与三段同法，但 yz 是慢速真实地面行走：
    # 完整步态循环（左右各一步）≈4.8s，半周期 ≈2.4s，gmr 版 0.5~2.0s 搜索窗装不下 → 本地放宽版）
    i_lhip = ISAAC_GYM_DOF_ORDER.index("left_hip_pitch_joint")
    i_rhip = ISAAC_GYM_DOF_ORDER.index("right_hip_pitch_joint")
    T_gait, ac_pk = yz_gait_period(q[:, i_lhip], FPS_SRC)
    print(f"  ④周期: T_gait={T_gait:.3f}s ({1/T_gait:.2f}Hz, r={ac_pk:.2f})"
          f" 半周期≈{T_gait/2:.2f}s——gait_period 语义 = 相位时钟完整周期(左右各一步)，"
          f"与 ref_lib 三段同义")

    # 切段：掐头去尾 + 整周期截断（30Hz 上切）
    dur = q.shape[0] / FPS_SRC
    n_cycles = int((dur - T_START - MARGIN) / T_gait)
    f0 = int(T_START * FPS_SRC)
    f1 = f0 + int(round(n_cycles * T_gait * FPS_SRC))
    qs, rps, rrs = q[f0:f1], root_pos[f0:f1], root_rot[f0:f1]
    print(f"  ⑤切段: [{T_START:.1f}s, {f1/FPS_SRC:.2f}s] {n_cycles}整周期 {qs.shape[0]}帧@30Hz")

    # 锚点
    tau, c_l, c_r = phase_anchor(qs, T_gait, FPS_SRC, i_lhip, i_rhip)
    print(f"  ④锚点: tau={tau:.3f}s corr(L,−sin)={c_l:.2f} corr(R,−sin)={c_r:.2f}"
          f" {'[OK]' if c_l > 0.5 and c_r > 0.5 else '[WARN 相位弱]'}")

    # detrend：只对关节角做首尾线性（治 loop 缝）；root_pos 保留真实前进位移
    lag = np.linspace(0.0, 1.0, qs.shape[0])[:, None]
    qs = qs - lag * (qs[-1:, :] - qs[0:1, :])

    # 重采样 30→50Hz（root quat 线性插值 + 重归一化，预检角速度 ≤0.38 rad/s 误差可忽略）
    q50, rp50, rr50, _ = resample(qs, rps, rrs, FPS_SRC, FPS_DST)
    rr50 = rr50 / np.linalg.norm(rr50, axis=1, keepdims=True)

    # loop 缝检查（直线行走放宽：WARN 不 assert）
    gap = np.abs(q50[0] - q50[-1]).max()
    print(f"  ⑤loop缝: max|q[0]-q[-1]|={gap:.4f}rad {'[OK]' if gap < 0.05 else '[WARN 直线段放宽]'}")

    # 速度统计（入库证据）
    d = float(np.hypot(rp50[-1, 0] - rp50[0, 0], rp50[-1, 1] - rp50[0, 1]))
    t = rp50.shape[0] / FPS_DST
    print(f"  ⑥速度: 位移 {d:.2f}m / {t:.1f}s = {d/t:.3f} m/s（补位：原三段均速 0.09/0.11/1.23）")

    anchor50 = int(round(tau * FPS_DST))
    entry = dict(
        dof_pos=torch.from_numpy(q50.astype(np.float32)),
        root_pos=torch.from_numpy(rp50.astype(np.float32)),
        root_rot_wxyz=torch.from_numpy(rr50.astype(np.float32)),
        gait_period=float(T_gait),
        phase_anchor_frame=anchor50,
        fps=FPS_DST,
        dof_names=list(ISAAC_GYM_DOF_ORDER),
        source="yz_csv",
        n_cycles=n_cycles,
    )
    if args.dry_run:
        print(f"[DRY-RUN] 将写入 {SEG_NAME}: dof_pos{tuple(q50.shape)}@50Hz "
              f"period={T_gait:.3f}s anchor={anchor50}")
        return

    lib[SEG_NAME] = entry
    torch.save(lib, OUT)
    total = sum(v["dof_pos"].shape[0] for v in lib.values())
    print(f"\n[SAVED] {OUT}  段数={len(lib)} 总帧={total} ({total/FPS_DST:.1f}s) 段列表={sorted(lib)}")


if __name__ == "__main__":
    main()
