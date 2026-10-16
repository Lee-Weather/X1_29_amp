#!/usr/bin/env python
"""sim2sim: Isaac Gym 策略（ONNX）在 MuJoCo 中闭环运行 — X1 29DOF。

复刻训练观测管线（num_single_obs=98 × frame_stack=66，与 ExportedDH 输入一致）：
  [ sin(2πφ), cos(2πφ),                       # 相位钟（exp1.7 自适应）
    vx·0.1, vy·0.1, wz·0.2,                   # commands_scale = [lin_vel, lin_vel, ang_vel]
    (q − q_default)·0.02 ×29,                  # obs_scales.dof_pos
    dq·1.5 ×29,                                # obs_scales.dof_vel
    prev_action ×29,                           # 上一步已执行动作
    ω_body·0.2 ×3, rpy·0.1 ×3 ]                # obs_scales.ang_vel / quat

相位钟（exp1.7 自适应步频，_post_physics_step_callback 语义）：
  行走: φ += dt_ctrl / (T_seg · clamp(v_demo/v_cmd, 0.5, 1.6))；站立: φ=0；
  从站立恢复: φ 起点取 0.5。段路由: |wz|>0.15→turn, |vx|≥0.8→norm, <0.15→slow, 其余→yz。
  段周期/速度表从 ref_lib.pt 按训练同款公式实测（跑步机段用 config 表值兜底）。

控制: PD 位置执行器按 config 注入 MJCF（kp/kd=stiffness/damping），
  q_des = default + 0.3·action；timestep 1ms × decimation 10 = 100Hz 控制。
关节序: MJCF 序 ≠ Isaac dof 序，按名字映射（Isaac 序取自 ref_lib.dof_names）。

用法（F1 环境: mujoco + onnxruntime）:
  python scripts/tools/sim2sim_mujoco.py --onnx czy/data/exp2.6/model_32997_policy.onnx \
      --out czy/data/exp2.6/sim2sim
"""
import argparse
import csv
import math
import os
import re

import numpy as np
import torch
import mujoco
import onnxruntime as ort

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_XML = os.path.join(REPO, "czy/mujoco/mjcf/x1_29dof_scene.xml")   # raw+地板（URDF 修复同步版）
DEFAULT_REF = os.path.join(REPO, "resources/motions/processed/ref_lib.pt")

# ---- config 直录（x1_dh_stand_config.py，e969023 血统）----
ACTION_SCALE = 0.3
DECIMATION = 10
SIM_DT = 0.001
CTRL_DT = SIM_DT * DECIMATION
FRAME_STACK = 66
S_Q, S_DQ, S_CMD_L, S_CMD_A, S_W, S_RPY = 0.02, 1.5, 0.1, 0.2, 0.2, 0.1
GAIT_START = 0.5          # 从站立恢复时的相位起点（训练 gait_start 随机 0/0.5，取 0.5）
STAND_TH = 0.05
CLAMP = (0.5, 1.6)
SEG_SPEED_TABLE = {"walk_norm": 1.23, "walk_slow": 0.10}   # 跑步机段兜底（实测≈0）

DEFAULT_ANGLES = {
    'left_hip_pitch_joint': 0.4, 'left_hip_roll_joint': 0.05, 'left_hip_yaw_joint': -0.31,
    'left_knee_pitch_joint': 0.49, 'left_ankle_pitch_joint': -0.21, 'left_ankle_roll_joint': 0.0,
    'right_hip_pitch_joint': -0.4, 'right_hip_roll_joint': -0.05, 'right_hip_yaw_joint': 0.31,
    'right_knee_pitch_joint': 0.49, 'right_ankle_pitch_joint': -0.21, 'right_ankle_roll_joint': 0.0,
    'lumbar_yaw_joint': 0.0, 'lumbar_roll_joint': 0.0, 'lumbar_pitch_joint': 0.03,
    'left_shoulder_pitch_joint': 0.03, 'right_shoulder_pitch_joint': -0.03,
    'left_shoulder_roll_joint': -0.06, 'right_shoulder_roll_joint': -0.06,
    'left_shoulder_yaw_joint': 0.18, 'right_shoulder_yaw_joint': -0.18,
    'left_elbow_pitch_joint': 0.34, 'right_elbow_pitch_joint': 0.34,
    'left_elbow_yaw_joint': 0.0, 'right_elbow_yaw_joint': 0.0,
    'left_wrist_pitch_joint': 0.0, 'right_wrist_pitch_joint': 0.0,
    'left_wrist_roll_joint': 0.0, 'right_wrist_roll_joint': 0.0,
}
STIFFNESS = {'hip_pitch': 30, 'hip_roll': 40, 'hip_yaw': 35, 'knee_pitch': 100,
             'ankle_pitch': 35, 'ankle_roll': 35, 'lumbar_yaw': 60, 'lumbar_roll': 60,
             'lumbar_pitch': 80, 'shoulder_pitch': 40, 'shoulder_roll': 40, 'shoulder_yaw': 40,
             'elbow_pitch': 30, 'elbow_yaw': 30, 'wrist_pitch': 8, 'wrist_roll': 8}
DAMPING = {'hip_pitch': 3, 'hip_roll': 3.0, 'hip_yaw': 4, 'knee_pitch': 8,
           'ankle_pitch': 1.5, 'ankle_roll': 1.5, 'lumbar_yaw': 4, 'lumbar_roll': 4,
           'lumbar_pitch': 5, 'shoulder_pitch': 2, 'shoulder_roll': 2, 'shoulder_yaw': 2,
           'elbow_pitch': 1.5, 'elbow_yaw': 1.5, 'wrist_pitch': 0.5, 'wrist_roll': 0.5}

# Isaac 侧逐关节 armature（训练随机化中心/辨识值，config L299-306）——
# MuJoCo 缺它则关节有效惯量 ~5e-4，PD 增益即成暴力激励（开环回灌也塌的根因）
ARMATURE = {"left_hip_pitch_joint": 0.196, "right_hip_pitch_joint": 0.196,
            "left_hip_roll_joint": 0.025, "right_hip_roll_joint": 0.025,
            "left_hip_yaw_joint": 0.0148, "right_hip_yaw_joint": 0.0060,
            "left_knee_pitch_joint": 0.25, "right_knee_pitch_joint": 0.25,
            "left_ankle_pitch_joint": 0.0215, "right_ankle_pitch_joint": 0.0215,
            "left_ankle_roll_joint": 0.0215, "right_ankle_roll_joint": 0.0215}
ARMATURE_DEFAULT = 0.01  # 上半身（训练统一范围 [0.0001, 0.05] 的中心量级）

VEL_PROFILE = [(5.0, 0.0), (5.0, 0.4), (5.0, 0.6), (5.0, 0.0)]   # 与 play.py 回放同阶梯




def pd_for(jname):
    base = jname.replace("left_", "").replace("right_", "")
    for key, val in STIFFNESS.items():
        if base.startswith(key):
            return val, DAMPING[key]
    raise KeyError(jname)


def load_seg_tables(ref_path):
    """段周期/速度表——按 env _init_amp 同款公式（路径长×fps/帧数，跑步机段表值兜底）。"""
    lib = torch.load(ref_path, map_location="cpu")
    segs = {}
    for name, d in lib.items():
        T = d["dof_pos"].shape[0]
        fps = float(d["fps"])
        period = float(d["gait_period"]) if "gait_period" in d else T / fps
        root_xy = d["root_pos"][:, :2]
        v_meas = float(np.linalg.norm(root_xy[1:] - root_xy[:-1], axis=1).sum()) * fps / T
        v_cfg = SEG_SPEED_TABLE.get(name, 0.0)
        v_demo = v_meas if (v_cfg <= 0 or v_meas > 0.5 * v_cfg) else v_cfg
        segs[name] = dict(T=period, v=v_demo, dof_names=list(d["dof_names"]))
        print(f"[SEG] {name}: T={period:.3f}s v_demo={v_demo:.3f}m/s")
    return segs


def route_seg(vx, wz):
    if abs(wz) > 0.15:
        return "walk_turn"
    if abs(vx) >= 0.8:
        return "walk_norm"
    if abs(vx) < 0.15:
        return "walk_slow"
    return "walk_yz"


def quat_to_rpy_R(qwxyz):
    w, x, y, z = qwxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw, R


def _strip_3x(xml):
    xml = re.sub(r'\s+content_type="[^"]*"', "", xml)   # mujoco 3.x 属性，2.3.6 不识别
    xml = re.sub(r'\s+actuatorfrcrange="[^"]*"', "", xml)
    return xml


def _inline_includes(xml, base_dir):
    """递归展开 <include file=...>（取被含文件 <mujoco> 的 body），逐个剥 3.x 属性。"""
    def repl(m):
        inc = m.group(1)
        sub = _strip_3x(open(os.path.join(base_dir, inc)).read())
        body = re.search(r"<mujoco[^>]*>(.*)</mujoco>", sub, re.S)
        return body.group(1) if body else sub
    prev = None
    while prev != xml:
        prev = xml
        xml = re.sub(r'<include\s+file\s*=\s*"([^"]+)"\s*/>', repl, xml)
    return xml


def build_model(xml_path):
    xml_path = os.path.abspath(xml_path)
    base_dir = os.path.dirname(xml_path)
    xml = _inline_includes(open(xml_path).read(), base_dir)
    if "autolimits" not in xml:
        xml = xml.replace("<compiler ", '<compiler autolimits="true" ', 1)
    # from_xml_string 无 meshdir 上下文：写同目录临时文件保住相对路径
    tmp = os.path.join(os.path.dirname(os.path.abspath(xml_path)), "_sim2sim_tmp.xml")
    with open(tmp, "w") as f:
        f.write(xml)
    mj = mujoco.MjModel.from_xml_path(tmp)             # 先解析拿关节表
    actuators = ["<actuator>"]
    for j in range(1, mj.njnt):                        # 跳过 freejoint
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
        kp, kv = pd_for(name)
        # 2.3.6 无 <position kv>：general 显式 PD（force = kp(ctrl−q) − kv·dq）
        actuators.append(
            f'<general name="act_{name}" joint="{name}" gaintype="fixed" '
            f'gainprm="{kp}" biastype="affine" biasprm="0 {-kp} {-kv}"/>')
    actuators.append("</actuator>")
    if "</mujoco>" not in xml:
        raise RuntimeError("xml 无 </mujoco> 收尾，无法注入执行器")
    xml = xml.replace("</mujoco>", "\n".join(actuators) + "\n</mujoco>")
    with open(tmp, "w") as f:
        f.write(xml)
    model = mujoco.MjModel.from_xml_path(tmp)
    os.remove(tmp)
    model.opt.timestep = SIM_DT
    # Euler+显式 PD 阻尼 1ms 下不稳（QACC NaN）：implicitfast 隐式处理执行器阻尼
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # 注入 Isaac 侧 armature（对角惯量加成，语义与 Isaac dof armature 一致）
    for j in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        model.dof_armature[model.jnt_dofadr[j]] = ARMATURE.get(name, ARMATURE_DEFAULT)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(REPO, "czy/data/exp2.6/model_32997_policy.onnx"))
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--out", default=os.path.join(REPO, "czy/data/exp2.6/sim2sim"))
    ap.add_argument("--replay_csv", default=None,
                    help="开环回灌：用 Isaac 回放 CSV 的 action 序列代替策略（动力学兼容性判定）")
    ap.add_argument("--noise", action="store_true", default=False,
                    help="obs 加训练同款噪声（noise_level=1.5）")
    ap.add_argument("--lag", type=int, default=20, help="统一滞后（物理步）：action 与 dof obs")
    ap.add_argument("--lag_act", type=int, default=None, help="单独覆盖 action 滞后")
    args = ap.parse_args()

    replay_rows = None
    if args.replay_csv:
        replay_rows = list(csv.DictReader(open(args.replay_csv)))
        print(f"[REPLAY] 开环回灌 {len(replay_rows)} 步 from {args.replay_csv}")

    segs = load_seg_tables(args.ref)
    isaac_dofs = segs["walk_norm"]["dof_names"]
    assert len(isaac_dofs) == 29

    model = build_model(args.xml)
    data = mujoco.MjData(model)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    # ---- 关节序映射 & 默认角 ----
    mj_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in range(1, model.njnt)]
    qadr = {n: model.jnt_qposadr[j] for j, n in enumerate(mj_names, start=1)}
    vadr = {n: model.jnt_dofadr[j] for j, n in enumerate(mj_names, start=1)}
    isaac2mj = [qadr[n] for n in isaac_dofs]            # isaac idx -> mj qpos 地址
    isaac2mjv = [vadr[n] for n in isaac_dofs]
    isaac2act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_" + n)
                 for n in isaac_dofs]
    q_default = np.array([DEFAULT_ANGLES[n] for n in isaac_dofs])

    # ---- 初始状态：默认位姿 + 基座贴地（FK 实算，消除落地冲击伪影）----
    data.qpos[:3] = [0, 0, 0.70]
    data.qpos[3:7] = [1, 0, 0, 0]
    for i, a in enumerate(isaac2mj):
        data.qpos[a] = q_default[i]
    mujoco.mj_forward(model, data)
    foot_z = [data.geom_xpos[g][2] - (model.geom_size[g][1] if model.geom_type[g] == 2 else 0)
              for g in range(model.ngeom) if model.geom_type[g] in (2, 6, 7)]
    data.qpos[2] -= min(foot_z) - 0.002
    mujoco.mj_forward(model, data)

    # ---- 观测栈（训练 reset 语义：66 帧全零）+ 物理步级滞后环（add_lag/add_dof_lag 域中心值）----
    hist = np.zeros((FRAME_STACK, 98), dtype=np.float32)
    prev_action = np.zeros(29, dtype=np.float32)
    phase = 0.0
    rows = []
    ACT_LAG = args.lag_act if args.lag_act is not None else args.lag
    DOF_LAG, RING = args.lag, max(args.lag, args.lag_act or 0) + 4
    des_ring = np.tile((q_default.copy()), (RING, 1))   # (RING,29) isaac 序
    q_ring = np.tile(data.qpos[isaac2mj], (RING, 1))
    dq_ring = np.tile(data.qvel[isaac2mjv], (RING, 1))
    ring_i = 0

    def ring_get(ring, lag):
        return ring[(ring_i - lag) % RING]

    total = int(sum(d for d, _ in VEL_PROFILE) / CTRL_DT)
    t_cmd, seg_i, cmd = 0.0, 0, VEL_PROFILE[0][1]
    for step in range(total):
        # 指令阶梯
        if t_cmd >= VEL_PROFILE[seg_i][0] and seg_i < len(VEL_PROFILE) - 1:
            seg_i += 1
            t_cmd = 0.0
            cmd = VEL_PROFILE[seg_i][1]
        t_cmd += CTRL_DT

        # 相位钟（站立清零；恢复行走起点 0.5）
        standing = math.hypot(cmd, 0.0) <= STAND_TH and abs(0.0) <= STAND_TH
        if standing:
            phase = 0.0
        else:
            if phase == 0.0:
                phase = GAIT_START
            seg = route_seg(cmd, 0.0)
            scale = min(max(segs[seg]["v"] / max(abs(cmd), 1e-3), CLAMP[0]), CLAMP[1])
            phase += CTRL_DT / (segs[seg]["T"] * scale)
        phase = phase % 1.0

        # 观测 98 维（q/dq 取滞后环值 = add_dof_lag 语义）
        qpos = ring_get(q_ring, DOF_LAG)
        qvel = ring_get(dq_ring, DOF_LAG)
        qwxyz = data.qpos[3:7]
        roll, pitch, yaw, R = quat_to_rpy_R(qwxyz)
        w_body = data.qvel[3:6]   # freejoint 角速度已是体坐标系（cvel 交叉验证）
        obs = np.concatenate([[
            math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase),
            cmd * S_CMD_L, 0.0, 0.0],
            (qpos - q_default) * S_Q, qvel * S_DQ, prev_action,
            w_body * S_W, [roll * S_RPY, pitch * S_RPY, yaw * S_RPY]]).astype(np.float32)
        if args.noise:   # 训练同款均匀噪声（noise_level=1.5：q±0.03 dq±2.25 w±0.3 rpy±0.15）
            obs[5:34] += np.random.uniform(-0.03, 0.03, 29)
            obs[34:63] += np.random.uniform(-2.25, 2.25, 29)
            obs[92:95] += np.random.uniform(-0.3, 0.3, 3)
            obs[95:98] += np.random.uniform(-0.15, 0.15, 3)
        hist[:-1] = hist[1:]
        hist[-1] = obs

        if replay_rows is not None and step < len(replay_rows):
            action = np.array([float(replay_rows[step][f'action_{n}'])
                               for n in isaac_dofs], dtype=np.float32)
        else:
            action = sess.run(None, {inp: hist.reshape(1, -1)})[0][0].astype(np.float32)
        if not np.isfinite(action).all():
            print(f"[NaN] step={step} t={step*CTRL_DT:.3f}s action 非有限 |obs|max={np.nanmax(np.abs(obs))} "
                  f"z={data.qpos[2]:.3f} |qvel|max={np.abs(data.qvel[isaac2mjv]).max():.1f} "
                  f"|qacc|max={np.abs(data.qacc).max():.3e}")
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, -100., 100.)          # 与训练 clip_actions 对齐
        prev_action = action

        q_des = q_default + ACTION_SCALE * action       # isaac 序（进滞后环，环出口才是执行目标）
        for _ in range(DECIMATION):
            ring_i = (ring_i + 1) % RING
            des_ring[ring_i] = q_des                    # Isaac：每个物理步把当步 scaled action 入环
            q_ring[ring_i] = data.qpos[isaac2mj]
            dq_ring[ring_i] = data.qvel[isaac2mjv]
            ctrl_now = ring_get(des_ring, ACT_LAG)
            for i, a in enumerate(isaac2act):
                data.ctrl[a] = ctrl_now[i]
            mujoco.mj_step(model, data)

        # 诊断记录
        v_world = data.qvel[0:3]
        v_body = R @ v_world
        rows.append([round(step * CTRL_DT, 4), cmd,
                     *data.qpos[0:3], roll, pitch, yaw,
                     v_body[0], v_body[1], data.qpos[2]])

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "sim2sim_diag.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "cmd_linear_x", "pos_x", "pos_y", "pos_z",
                    "roll", "pitch", "yaw", "vfwd_body", "vlat_body", "base_height"])
        w.writerows(rows)

    # ---- Summary（与 isaac 回放同口径）----
    print(f"\nSaved {csv_path}")
    lo = sum(1 for r in rows if r[10] < 0.45)
    yaw0, yaw1 = rows[0][8], rows[-1][8]
    dyaw = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))
    print(f"跌倒帧(h<0.45): {lo}/{len(rows)}  h_min={min(r[10] for r in rows):.3f}")
    print(f"航向净漂: {dyaw:+.1f}°  pos_x终={rows[-1][2]:.2f}m  pos_y终={rows[-1][3]:.2f}m")
    for cv in (0.4, 0.6):
        seg = [r for r in rows if abs(r[1] - cv) < 1e-6]
        if seg:
            print(f"cmd={cv}: 前进向均速 {sum(r[8] for r in seg)/len(seg):+.3f} m/s "
                  f"({(sum(r[8] for r in seg)/len(seg)-cv)/cv*100:+.0f}%)  pitch均 "
                  f"{sum(r[6] for r in seg)/len(seg):+.3f}")


if __name__ == "__main__":
    main()
