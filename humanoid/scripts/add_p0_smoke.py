#!/usr/bin/env python3
"""P0 冒烟：ADD（差分判别器）数据管线可行性验证 —— 不训练、不改动任何生产代码。

背景
----
ADD 把判别器输入从「观测本身」改为「demo − agent 的差」，正样本恒为全零向量。
这要求 demo 观测与 agent **同相位对齐**，而现有 AMP 管线是刻意解耦的随机抽帧
（x1_dh_stand_env._sample_amp_demo）。所以在动手改生产代码之前，先用真实数据
回答三个问题：

  A. 相位→帧映射自洽吗？段内一个周期是否回到同一位形、环绕是否连续。
  B. 对齐后的 diff 有意义吗？用 exp2.1 真实回放轨迹算 diff，看量级、系统性
     偏移（后仰类问题会在这里现形）、demo/agent 相关性。
  C. 差值归一化后是否良态？DiffNormalizer（除以平均绝对幅值）是否把各通道
     拉到 O(1)，使判别器不会被大数量纲项主导。

本脚本是**只读诊断**：读 ref_lib.pt 与一份 isaac_diag.csv，不 import isaacgym，
不需要 GPU。

用法
----
  python humanoid/scripts/add_p0_smoke.py
  python humanoid/scripts/add_p0_smoke.py --csv czy/data/exp2.1/isaac_diag.csv
  python humanoid/scripts/add_p0_smoke.py --csv czy/data/exp1.12/isaac_diag.csv

判读要点见脚本末尾打印的「判读」小节。
"""

import argparse
import csv
import os

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 与 env 保持一致的映射常量（x1_dh_stand_env.py L769-780 / L382-384 / L161-181）
# ---------------------------------------------------------------------------
SEG_ORDER = ["walk_norm", "walk_slow", "walk_turn", "walk_yz"]  # = sorted(lib.keys())
TURN_WZ_THRESH = 0.15   # |wz| > 0.15 → walk_turn
NORM_VX_THRESH = 0.80   # |vx| >= 0.8 → walk_norm
SLOW_VX_THRESH = 0.15   # |vx| < 0.15 → walk_slow


# ---------------------------------------------------------------------------
# ref 库加载与 demo 特征预计算（逐行对齐 env._init_amp L841-876）
# ---------------------------------------------------------------------------
def load_ref_lib(path):
    lib = torch.load(path, map_location="cpu")
    names = sorted(lib.keys())
    if names != SEG_ORDER:
        raise SystemExit("ref_lib 段名 {} 与脚本假设 {} 不符，请核对".format(names, SEG_ORDER))
    return lib


def build_seg_meta(lib):
    """返回每段的帧数/帧率/周期帧数/锚点，并预计算 demo 特征（dof_pos/dof_vel/root_ang_vel）"""
    meta = {}
    for s in SEG_ORDER:
        d = lib[s]
        q = d["dof_pos"].float()                       # (T,29) 绝对角
        fps = float(d.get("fps", 50.0))
        T = q.shape[0]
        period_frames = int(round(float(d["gait_period"]) * fps))   # env L773
        anchor = int(d["phase_anchor_frame"])                        # env L774

        # dof_vel：前向差分 × fps，末帧沿用（env L850-852）
        dq = torch.zeros_like(q)
        dq[:-1] = (q[1:] - q[:-1]) * fps
        dq[-1] = dq[-2]

        # root_ang_vel（体轴）：q_rel = q_t^{-1} ⊗ q_{t+1}，精确 log 映射 ×fps（env L854-872）
        rot = d["root_rot_wxyz"].float()               # (T,4) wxyz
        quat = rot[:, [1, 2, 3, 0]]                    # → xyzw
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        conj = torch.cat([-quat[:-1, :3], quat[:-1, 3:4]], dim=-1)
        ax, ay, az, aw = conj[:, 0], conj[:, 1], conj[:, 2], conj[:, 3]
        bx, by, bz, bw = quat[1:, 0], quat[1:, 1], quat[1:, 2], quat[1:, 3]
        q_rel = torch.stack([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ], dim=-1)
        vec = q_rel[:, :3]
        n = vec.norm(dim=-1).clamp(min=1e-8)
        angle = 2.0 * torch.atan2(n, q_rel[:, 3])
        ang = torch.zeros(T, 3)
        ang[:-1] = vec / n.unsqueeze(-1) * (angle * fps).unsqueeze(-1)
        ang[-1] = ang[-2]

        meta[s] = dict(T=T, fps=fps, period_frames=period_frames, anchor=anchor,
                       seg_cycle=period_frames / fps,
                       dof_pos=q, dof_vel=dq, root_ang_vel=ang)
    return meta


def seg_id_of(vx, wz):
    """env._current_seg_id 的逐样本复刻（L161-181）"""
    turn = abs(wz) > TURN_WZ_THRESH
    if turn:
        return SEG_ORDER.index("walk_turn")
    if abs(vx) >= NORM_VX_THRESH:
        return SEG_ORDER.index("walk_norm")
    if abs(vx) < SLOW_VX_THRESH:
        return SEG_ORDER.index("walk_slow")
    return SEG_ORDER.index("walk_yz")


def ref_frame(phase, seg, meta):
    """相位 → 段内帧（env L382-384）：floor(φ·P + anchor) mod T"""
    m = meta[SEG_ORDER[seg]]
    f = int(np.floor(phase * m["period_frames"] + m["anchor"]))
    return f % m["T"]


# ---------------------------------------------------------------------------
# DiffNormalizer（MimicKit diff_normalizer.py L54-58）：除以平均绝对幅值
# ---------------------------------------------------------------------------
class DiffNormalizer:
    def __init__(self, dim, min_diff=1e-4):
        self.min_diff = min_diff
        self.mean_abs = np.ones(dim)
        self._sum = np.zeros(dim)
        self._count = 0

    def record(self, x):                     # x: (N, dim)
        self._sum += np.abs(x).sum(axis=0)
        self._count += x.shape[0]

    def update(self):
        if self._count > 0:
            self.mean_abs = self._sum / self._count

    def normalize(self, x):
        return x / np.clip(self.mean_abs, self.min_diff, None)


# ---------------------------------------------------------------------------
# Check A：相位→帧映射自洽性
# ---------------------------------------------------------------------------
def check_a(meta):
    print("=" * 78)
    print("A. 相位→帧映射自洽性（ref_lib 自检）")
    print("=" * 78)
    print("{:<11} {:>6} {:>6} {:>8} {:>8} {:>9}".format(
        "段", "T", "fps", "周期帧P", "锚点A", "段周期(s)"))
    for s in SEG_ORDER:
        m = meta[s]
        print("{:<11} {:>6} {:>6} {:>8} {:>8} {:>9.3f}".format(
            s, m["T"], int(m["fps"]), m["period_frames"], m["anchor"], m["seg_cycle"]))

    print("\n周期性检查：q[f] vs q[f+P]（应接近 0，说明 P 是真实步态周期）")
    print("{:<11} {:>12} {:>12} {:>12}".format("段", "mean|Δ|", "max|Δ|", "样本数"))
    for s in SEG_ORDER:
        m = meta[s]
        q, P, T = m["dof_pos"], m["period_frames"], m["T"]
        if T <= P:
            print("{:<11} {:>12} {:>12} {:>12}".format(s, "-", "-", "段长<=P"))
            continue
        d = (q[:T - P] - q[P:]).abs()
        print("{:<11} {:>12.4f} {:>12.4f} {:>12}".format(
            s, float(d.mean()), float(d.max()), T - P))

    print("\n环绕连续性：锚点处跨周期（φ=0 与 φ=1 应给出同一位形）")
    print("{:<11} {:>12} {:>12}".format("段", "mean|Δ|", "max|Δ|"))
    for s in SEG_ORDER:
        m = meta[s]
        q = m["dof_pos"]
        f0 = m["anchor"] % m["T"]
        f1 = (m["anchor"] + m["period_frames"]) % m["T"]
        d = (q[f0] - q[f1]).abs()
        print("{:<11} {:>12.4f} {:>12.4f}".format(s, float(d.mean()), float(d.max())))

    print("\ndemo 特征量纲（mean|·|，用于判断哪些通道会主导未归一化的 diff）")
    print("{:<11} {:>14} {:>14} {:>14}".format("段", "root_ang_vel", "dof_pos", "dof_vel"))
    for s in SEG_ORDER:
        m = meta[s]
        print("{:<11} {:>14.4f} {:>14.4f} {:>14.4f}".format(
            s, float(m["root_ang_vel"].abs().mean()),
            float(m["dof_pos"].abs().mean()),
            float(m["dof_vel"].abs().mean())))


# ---------------------------------------------------------------------------
# Check B：真实轨迹上的 diff 统计
# ---------------------------------------------------------------------------
def load_csv(path, dof_names):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("CSV 为空: {}".format(path))

    need = ["step", "time_s", "phase_sin", "phase_cos", "cycle_time", "cmd_linear_x", "cmd_angular_z"]
    for c in need:
        if c not in rows[0]:
            raise SystemExit("CSV 缺列 {}（该文件可能不是 play.py 诊断输出）".format(c))

    def col(name):
        return np.array([float(r[name]) for r in rows], dtype=np.float64)

    # play.py L323：CSV 列名用去掉 _joint 后缀的关节名（pos_left_hip_pitch_joint）
    short = [n[:-6] if n.endswith("_joint") else n for n in dof_names]

    data = {c: col(c) for c in need}
    data["agent_pos"] = np.stack([col("pos_{}_joint".format(j)) for j in short], axis=1)
    data["agent_vel"] = np.stack([col("vel_{}_joint".format(j)) for j in short], axis=1)
    data["agent_ang"] = np.stack([col("base_ang_vel_{}".format(a)) for a in "xyz"], axis=1)
    return data


def check_b(data, meta, dof_names, stand_thresh):
    print()
    print("=" * 78)
    print("B. 相位对齐 diff 统计（真实 agent 轨迹 vs 相位对齐参考）")
    print("=" * 78)

    N = len(data["phase_sin"])
    phase = (np.arctan2(data["phase_sin"], data["phase_cos"]) / (2 * np.pi)) % 1.0
    walk = np.hypot(data["cmd_linear_x"], data["cmd_angular_z"]) > stand_thresh

    seg = np.array([seg_id_of(vx, wz) for vx, wz in
                    zip(data["cmd_linear_x"], data["cmd_angular_z"])])
    frames = np.array([ref_frame(phase[i], seg[i], meta) for i in range(N)])

    demo_pos = np.zeros((N, len(dof_names)))
    demo_vel = np.zeros((N, len(dof_names)))
    demo_ang = np.zeros((N, 3))
    comp = np.ones(N)                     # 时间压缩因子 = 段周期 / 有效周期
    for i in range(N):
        m = meta[SEG_ORDER[seg[i]]]
        f = frames[i]
        demo_pos[i] = m["dof_pos"][f].numpy()
        demo_vel[i] = m["dof_vel"][f].numpy()
        demo_ang[i] = m["root_ang_vel"][f].numpy()
        ct = data["cycle_time"][i]
        if ct > 0:
            comp[i] = m["seg_cycle"] / ct

    idx = np.where(walk)[0]
    print("总步数 {}，行走步 {}（站立步被 stand_mask 屏蔽，不计入）".format(N, len(idx)))
    seg_cnt = {SEG_ORDER[s]: int((seg[idx] == s).sum()) for s in range(len(SEG_ORDER))}
    print("行走步段分布:", seg_cnt)
    print("时间压缩因子 comp = 段周期/有效周期: 中位 {:.3f}，范围 [{:.3f}, {:.3f}]".format(
        float(np.median(comp[idx])), float(comp[idx].min()), float(comp[idx].max())))

    diff_pos = demo_pos[idx] - data["agent_pos"][idx]
    diff_vel_raw = demo_vel[idx] - data["agent_vel"][idx]
    diff_vel_cmp = demo_vel[idx] * comp[idx][:, None] - data["agent_vel"][idx]
    diff_ang_raw = demo_ang[idx] - data["agent_ang"][idx]
    diff_ang_cmp = demo_ang[idx] * comp[idx][:, None] - data["agent_ang"][idx]

    def group_report(title, diff):
        mean = diff.mean(axis=0)
        std = diff.std(axis=0)
        mabs = np.abs(diff).mean(axis=0)
        bias = np.abs(mean) / np.maximum(std, 1e-9)
        print("\n  [{}] 通道数 {} | mean|Δ| {:.4f} | std 中位 {:.4f} | "
              "系统性偏移(|mean|/std>0.5) 通道数 {}/{}".format(
                  title, diff.shape[1], float(mabs.mean()), float(np.median(std)),
                  int((bias > 0.5).sum()), len(bias)))
        return mean, std, mabs, bias

    p_mean, p_std, p_mabs, p_bias = group_report("dof_pos（29 关节角，rad）", diff_pos)
    v_mean, v_std, v_mabs, v_bias = group_report("dof_vel 未补偿（rad/s）", diff_vel_raw)
    vc_mean, vc_std, vc_mabs, vc_bias = group_report("dof_vel 已补偿（rad/s）", diff_vel_cmp)
    a_mean, a_std, a_mabs, a_bias = group_report("root_ang_vel 未补偿（rad/s）", diff_ang_raw)
    ac_mean, ac_std, ac_mabs, ac_bias = group_report("root_ang_vel 已补偿（rad/s）", diff_ang_cmp)

    print("\n  速度通道补偿效果（mean|Δ| 越小说明对齐越好）:")
    print("    dof_vel      : 未补偿 {:.4f} → 已补偿 {:.4f}  ({:+.1f}%)".format(
        float(v_mabs.mean()), float(vc_mabs.mean()),
        100.0 * (float(vc_mabs.mean()) - float(v_mabs.mean())) / max(float(v_mabs.mean()), 1e-9)))
    print("    root_ang_vel : 未补偿 {:.4f} → 已补偿 {:.4f}  ({:+.1f}%)".format(
        float(a_mabs.mean()), float(ac_mabs.mean()),
        100.0 * (float(ac_mabs.mean()) - float(a_mabs.mean())) / max(float(a_mabs.mean()), 1e-9)))

    # demo/agent 相关性：判断相位对齐是否「有意义」（近 0 = 对齐无信息，ADD 只会加噪）
    corr = np.nan_to_num(np.array([
        np.corrcoef(demo_pos[idx][:, k], data["agent_pos"][idx][:, k])[0, 1]
        for k in range(len(dof_names))
    ]))

    # 身体分区（X1 29DOF 序：左腿0-5 / 腰6-8 / 左臂9-15 / 右臂16-22 / 右腿23-28）
    def grp(n):
        n = n.lower()
        if ("hip" in n) or ("knee" in n) or ("ankle" in n):
            return "腿"
        return "腰" if "waist" in n else "臂"
    groups = [grp(n) for n in dof_names]

    print("\n  按身体分区（dof_pos）：分区差异是本轮判读的关键——臂部噪声会掩盖腿/腰信号")
    print("    {:<4} {:>4} {:>10} {:>10} {:>12} {:>10}".format(
        "分区", "通道", "mean|Δ|", "std中位", "偏移通道数", "corr中位"))
    for g in ["腿", "腰", "臂"]:
        sel = np.array([i for i, x in enumerate(groups) if x == g], dtype=int)
        if len(sel) == 0:
            continue
        print("    {:<4} {:>4} {:>10.4f} {:>10.4f} {:>7}/{:<4} {:>10.3f}".format(
            g, len(sel), float(p_mabs[sel].mean()), float(np.median(p_std[sel])),
            int((p_bias[sel] > 0.5).sum()), len(sel), float(np.median(corr[sel]))))

    # 相位偏移扫描：若 agent 只是"整体滞后/超前"参考，则固定帧偏移能显著拉高相关性。
    # 这决定了 ADD 对齐时是否需要补偿一个常量相位偏置（否则 diff 里混入纯相位误差）。
    leg_sel = np.array([i for i, x in enumerate(groups) if x == "腿"], dtype=int)
    print("\n  相位偏移扫描（腿部 {} 个关节的平均 corr，看是否存在常量相位滞后）:".format(len(leg_sel)))
    best = (None, -2.0)
    for off in range(-12, 13, 2):
        dp = np.zeros((len(idx), len(dof_names)))
        for j, i in enumerate(idx):
            m = meta[SEG_ORDER[seg[i]]]
            dp[j] = m["dof_pos"][(frames[i] + off) % m["T"]].numpy()
        c = np.nan_to_num(np.array([
            np.corrcoef(dp[:, k], data["agent_pos"][idx][:, k])[0, 1] for k in leg_sel]))
        mc = float(c.mean())
        if mc > best[1]:
            best = (off, mc)
        print("    off {:>+3} 帧 ({:>+5.0f} ms): 腿 mean corr {:+.3f}".format(
            off, off * 1000.0 / 50.0, mc))
    print("    → 最佳偏移 {:>+d} 帧 (corr {:.3f})；若相对 off=0 提升有限，说明不是常量相位滞后".format(
        best[0], best[1]))

    print("\n  demo/agent 关节角相关性（全 29 通道）: 中位 {:.3f} | >0.5 的 {}/{} | <0 的 {}/{}".format(
        float(np.median(corr)), int((corr > 0.5).sum()), len(corr),
        int((corr < 0).sum()), len(corr)))
    print("    相关性最高（相位对齐最有效）的 6 个关节:")
    for k in np.argsort(-corr)[:6]:
        print("      {:<28} [{:<2}] corr {:+.3f} | mean|Δ| {:.4f}".format(
            dof_names[k], groups[k], corr[k], p_mabs[k]))
    print("    相关性最低（对齐几乎无信息）的 6 个关节:")
    for k in np.argsort(corr)[:6]:
        print("      {:<28} [{:<2}] corr {:+.3f} | mean|Δ| {:.4f}".format(
            dof_names[k], groups[k], corr[k], p_mabs[k]))

    print("\n  dof_pos 系统性偏移最严重的 8 个关节（|mean|/std 大 = 持续偏置而非抖动）:")
    for k in np.argsort(-p_bias)[:8]:
        print("    {:<28} [{:<2}] bias {:>6.2f} | mean {:+.4f} | std {:.4f}".format(
            dof_names[k], groups[k], p_bias[k], p_mean[k], p_std[k]))

    return np.concatenate([diff_pos, diff_vel_cmp, diff_ang_cmp], axis=1)


# ---------------------------------------------------------------------------
# Check C：DiffNormalizer 良态性
# ---------------------------------------------------------------------------
def check_c(diff_all):
    print()
    print("=" * 78)
    print("C. DiffNormalizer 良态性（除以平均绝对幅值）")
    print("=" * 78)
    dim = diff_all.shape[1]
    norm = DiffNormalizer(dim)
    norm.record(diff_all)
    norm.update()

    print("原始 diff : mean|·| {:.4f} | 通道 mean|·| 范围 [{:.4f}, {:.4f}]（跨 {:.0f}x）".format(
        float(np.abs(diff_all).mean()),
        float(norm.mean_abs.min()), float(norm.mean_abs.max()),
        float(norm.mean_abs.max() / max(norm.mean_abs.min(), 1e-9))))

    z = norm.normalize(diff_all)
    raw_std = diff_all.std(axis=0)
    z_std = z.std(axis=0)
    print("归一化后 : 全局 mean {:+.4f} | 全局 std {:.4f} | |z|>5 占比 {:.3f}%".format(
        float(z.mean()), float(z.std()), 100.0 * float((np.abs(z) > 5).mean())))
    print("归一化后 : 逐通道 std 范围 [{:.4f}, {:.4f}]（跨 {:.0f}x）".format(
        float(z_std.min()), float(z_std.max()),
        float(z_std.max() / max(z_std.min(), 1e-9))))

    # 注意：mean|·| 归一化后每通道 mean|·| 恒为 1（构造使然，不构成判据）。
    # 真正有信息量的是「归一化后的逐通道 std」——它衡量该通道的**变化量**是否与其他
    # 通道可比。std 很小的通道 = 近似常量，判别器可直接读它的符号作为平凡特征。
    near_const = int((z_std < 0.5).sum())
    print("\n近似常量通道（归一化后 std < 0.5）: {}/{}".format(near_const, dim))
    print("  → 这类通道的 |mean| 已接近其 mean|·|，归一化后被放大成近似恒定的 ±1，")
    print("     判别器可绕过运动质量、直接读符号完成分类（与 AMP 的「幅度平凡特征」同构）。")
    if near_const > 0:
        order = np.argsort(z_std)[:min(near_const, 8)]
        for k in order:
            print("      ch {:>3}: z_std {:.3f} | z_mean {:+.3f} | raw mean|Δ| {:.4f}".format(
                k, z_std[k], z.mean(axis=0)[k], np.abs(diff_all[:, k]).mean()))
    print("\n判据：near_const 通道数应远小于总通道数；若占比可观，需按身体分区加掩码或分组归一化。")


# ---------------------------------------------------------------------------
# Check D：agent 实际步态周期 vs 参考钟有效周期（相位对齐是否有物理基础）
# ---------------------------------------------------------------------------
def check_d(data, meta, dof_names, stand_thresh):
    """相位对齐成立的前提是：agent 的实际步态节奏与相位钟的节奏一致。

    相位钟的有效周期由 cycle_time（= _current_cycle_time()）给出；agent 的实际
    步态周期由髋 pitch 信号的自相关峰估计。两者不一致 → 相位必然滑移 → 相位对齐
    的参考与 agent 真实姿态逐周期错开 → ADD 的 diff 退化为「参考库帧匹配」。
    """
    print()
    print("=" * 78)
    print("D. agent 实际步态周期 vs 参考钟有效周期")
    print("=" * 78)

    k = dof_names.index("left_hip_pitch_joint") if "left_hip_pitch_joint" in dof_names else 0
    vx = data["cmd_linear_x"]
    walk = np.hypot(vx, data["cmd_angular_z"]) > stand_thresh

    # CSV 是控制步（100 Hz），不是库帧率（50 Hz）——周期比较必须在同一时间基准上
    t = data["time_s"]
    dt_csv = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    print("CSV 控制步 dt = {:.4f}s（{} Hz）".format(dt_csv, int(round(1.0 / max(dt_csv, 1e-9)))))

    print("{:<10} {:>8} {:>14} {:>14} {:>10} {:>10}".format(
        "指令vx", "步数", "agent周期(步)", "参考钟周期(步)", "周期比", "自相关峰"))
    for cmd in sorted({round(float(v), 2) for v in vx[walk]}):
        sel = np.where(walk & (np.abs(vx - cmd) < 0.01))[0]
        if len(sel) < 120:
            continue
        sig = data["agent_pos"][sel, k].astype(np.float64)
        # 高通去漂移（减移动平均）：否则低频漂移会同时压低自相关峰并污染周期估计
        w = min(100, max(10, len(sig) // 4))
        kernel = np.ones(w) / w
        sig = sig - np.convolve(sig, kernel, mode="same")
        sig = sig - sig.mean()
        ac = np.correlate(sig, sig, "full")[len(sig) - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        hi = min(300, len(ac) - 2)
        peaks = [(l, ac[l]) for l in range(15, hi) if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1]]
        if not peaks:
            print("{:<10} {:>8} {:>14} {:>14} {:>10} {:>10}".format(
                "{:.2f}".format(cmd), len(sel), "无显著峰", "-", "-", "-"))
            continue
        lag, val = max(peaks, key=lambda p: p[1])
        ref_steps = float(np.median(data["cycle_time"][sel])) / max(dt_csv, 1e-9)
        print("{:<10} {:>8} {:>14.1f} {:>14.1f} {:>10.2f} {:>10.3f}".format(
            "{:.2f}".format(cmd), len(sel), float(lag), ref_steps,
            float(lag) / max(ref_steps, 1e-9), float(val)))

    print("\n判据：周期比应接近 1.00。偏离越大 → 相位钟与 agent 实际步态越不同步，")
    print("      相位对齐的参考在若干步后就与 agent 真实姿态错开，ADD 的 diff 失去意义。")


# ---------------------------------------------------------------------------
def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    ap = argparse.ArgumentParser(description="ADD P0 冒烟：相位对齐 demo 采样 + 差值归一化")
    ap.add_argument("--ref-lib", default=os.path.join(root, "resources/motions/processed/ref_lib.pt"))
    ap.add_argument("--csv", default=os.path.join(root, "czy/data/exp2.1/isaac_diag.csv"))
    ap.add_argument("--stand-thresh", type=float, default=1e-6,
                    help="指令范数低于此值判为站立步（默认 1e-6，仅滤掉真零指令段）")
    args = ap.parse_args()

    print("ref_lib : {}".format(args.ref_lib))
    print("csv     : {}".format(args.csv))

    lib = load_ref_lib(args.ref_lib)
    dof_names = list(lib[SEG_ORDER[0]]["dof_names"])
    meta = build_seg_meta(lib)

    check_a(meta)

    data = load_csv(args.csv, dof_names)
    diff_all = check_b(data, meta, dof_names, args.stand_thresh)

    check_d(data, meta, dof_names, args.stand_thresh)

    check_c(diff_all)

    print()
    print("=" * 78)
    print("判读")
    print("=" * 78)
    print("A 周期性 max|Δ| 大     → period_frames 与库不自洽，查表会跳帧（参考本身有抖动）")
    print("B 腿 corr 低           → 相位对齐无信息：agent 步态与相位钟未锁相，ADD 的 diff 会")
    print("                         退化成「参考库帧匹配」而非「风格偏差」")
    print("B 偏移扫描峰在 0 帧    → 不是常量滞后（滞后可用固定偏移补偿；非滞后则不可）")
    print("B 补偿后 mean|Δ| 下降  → gait_speed_adaptive 生效中，速度通道必须做时间缩放补偿")
    print("B 臂 corr≈0 且 bias 高 → 臂部近乎静止：信息量近零、噪声与偏置占比高，建议掩码")
    print("C near_const 占比高    → 归一化把常量偏置放大成平凡特征，需分组归一化/掩码")


if __name__ == "__main__":
    main()
