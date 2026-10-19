#!/usr/bin/env python
"""Build the X1 v1.2 straight walking reference and AMP libraries.

Input clips are already normalized to named, 50 Hz GMR records. B3 and B4
provide the two cyclic references. B1 and B2 are transitions and must not be
sampled as cyclic gait in this first training version.
"""

import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import torch

from prep_mocap_ref import ISAAC_GYM_DOF_ORDER


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "czy/gmr/slow_walk_amp_v1.2/normalized"
OUTPUT = ROOT / "resources/motions/processed"
MODEL = ROOT / "resources/robots/x1/mjcf/x1_29dof_scene.xml"
FPS = 50.0
CLIPS = {
    "walk_forward": ("F1_B3_continuous_walk", 20),
    "walk_backward": ("F1_B4_walk_backwards", 100),
}


def load_clip(name, joint_limits):
    with (INPUT / (name + ".pkl")).open("rb") as stream:
        data = pickle.load(stream)
    names = list(data["dof_names"])
    if float(data["fps"]) != FPS or set(names) != set(ISAAC_GYM_DOF_ORDER):
        raise ValueError("{}: fps or joint names do not match X1".format(name))
    if data.get("joint_signs_changed") is not False:
        raise ValueError("{}: expected unflipped v1.2 joint angles".format(name))
    order = [names.index(joint) for joint in ISAAC_GYM_DOF_ORDER]
    q = np.asarray(data["dof_pos"], dtype=np.float64)[:, order]
    root = np.asarray(data["root_pos"], dtype=np.float64)
    quat = np.asarray(data["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]
    bodies = np.asarray(data["body_positions"], dtype=np.float64)
    if q.shape[1] != 29 or root.shape != (len(q), 3) or quat.shape != (len(q), 4):
        raise ValueError("{}: unexpected trajectory shape".format(name))
    if not all(np.isfinite(x).all() for x in (q, root, quat, bodies)):
        raise ValueError("{}: nonfinite trajectory value".format(name))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    for frame in range(1, len(quat)):
        if np.dot(quat[frame - 1], quat[frame]) < 0:
            quat[frame] *= -1
    lower, upper = joint_limits
    excess = np.maximum(lower - q, 0) + np.maximum(q - upper, 0)
    limit_report = {
        "values_clipped": int(np.count_nonzero(excess)),
        "max_excess_rad": float(np.max(excess)),
    }
    q = np.clip(q, lower, upper)
    return data, q, root, quat, bodies, limit_report


def estimate_period(q):
    """Median first strong 0.5-2 s autocorrelation peak across both legs."""
    periods = []
    for joint in ("left_hip_pitch_joint", "right_hip_pitch_joint",
                  "left_knee_pitch_joint", "right_knee_pitch_joint"):
        signal = q[:, ISAAC_GYM_DOF_ORDER.index(joint)].copy()
        signal -= signal.mean()
        corr = np.correlate(signal, signal, mode="full")[len(signal) - 1:]
        corr /= max(corr[0], 1e-12)
        peaks = [lag for lag in range(25, min(100, len(corr) - 1))
                 if corr[lag] > 0.3 and corr[lag] > corr[lag - 1]
                 and corr[lag] >= corr[lag + 1]]
        if not peaks:
            raise ValueError("no gait period found for {}".format(joint))
        periods.append(peaks[0])
    return int(round(float(np.median(periods)))), periods


def select_cycle(q, bodies, body_names, period, minimum_start):
    """Choose an active cycle with the smallest pose and velocity loop seam."""
    left = body_names.index("leg_l_toe_a_link")
    right = body_names.index("leg_r_toe_a_link")
    candidates = []
    for length in range(period - 2, period + 3):
        for start in range(minimum_start, len(q) - length - 10):
            z = bodies[start:start + length, :, 2]
            if min(np.ptp(z[:, left]), np.ptp(z[:, right])) < 0.03:
                continue
            pose_gap = np.sqrt(np.mean((q[start + length] - q[start]) ** 2))
            vel_gap = np.sqrt(np.mean(((q[start + length] - q[start + length - 1])
                                       - (q[start + 1] - q[start])) ** 2))
            candidates.append((pose_gap + 0.5 * vel_gap, start, length,
                               pose_gap, vel_gap))
    if not candidates:
        raise ValueError("no active gait cycle with both feet swinging")
    return min(candidates)


def phase_anchor(bodies, body_names, start, length):
    """Align positive sin(phase) with right swing / left stance."""
    left = body_names.index("leg_l_toe_a_link")
    right = body_names.index("leg_r_toe_a_link")
    height_diff = (bodies[start:start + length, right, 2]
                   - bodies[start:start + length, left, 2])
    correlations = [np.corrcoef(
        height_diff,
        np.sin(2 * np.pi * (np.arange(length) - anchor) / length)
    )[0, 1] for anchor in range(length)]
    anchor = int(np.argmax(correlations))
    if correlations[anchor] < 0.5:
        raise ValueError("foot swing does not align with the phase clock")
    return anchor, float(correlations[anchor])


def pack(q, root, quat, source, period, anchor):
    return {
        "dof_pos": torch.from_numpy(q.astype(np.float32)),
        "root_pos": torch.from_numpy(root.astype(np.float32)),
        "root_rot_wxyz": torch.from_numpy(quat.astype(np.float32)),
        "dof_names": list(ISAAC_GYM_DOF_ORDER),
        "fps": FPS,
        "gait_period": period / FPS,
        "phase_anchor_frame": anchor,
        "source": source,
    }


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
              for name in ISAAC_GYM_DOF_ORDER]
    if any(index < 0 for index in joints):
        raise ValueError("MJCF joint names differ from Isaac Gym DOF list")
    limits = (model.jnt_range[joints, 0], model.jnt_range[joints, 1])
    phase_lib, amp_lib, report = {}, {}, {}
    for segment, (source, minimum_start) in CLIPS.items():
        data, q, root, quat, bodies, limit_report = load_clip(source, limits)
        period, joint_periods = estimate_period(q)
        _, start, length, pose_gap, velocity_gap = select_cycle(
            q, bodies, data["body_names"], period, minimum_start)
        anchor, phase_corr = phase_anchor(bodies, data["body_names"], start, length)

        # Close the pose loop with a small distributed correction. q[start+length]
        # is the equivalent phase of the next cycle, not an extra output frame.
        drift = q[start + length] - q[start]
        loop_q = q[start:start + length].copy()
        loop_q -= np.arange(length)[:, None] / length * drift[None, :]
        loop_q = np.clip(loop_q, limits[0], limits[1])
        loop_end_gap = float(np.max(np.abs(loop_q[-1] - loop_q[0])))
        native_step_max = float(np.max(np.abs(np.diff(loop_q, axis=0))))
        if loop_end_gap > 1.2 * native_step_max:
            raise ValueError("{}: loop jump {:.3f} rad exceeds native step {:.3f}".format(
                segment, loop_end_gap, native_step_max))
        phase_lib[segment] = pack(
            loop_q, root[start:start + length], quat[start:start + length],
            source, length, anchor)
        root_steps = np.linalg.norm(
            np.diff(root[start:start + length, :2], axis=0), axis=1)
        demo_speed = float(root_steps.sum() * FPS / length)

        end = len(q) - 10
        amp_lib[segment] = pack(q[minimum_start:end], root[minimum_start:end],
                                quat[minimum_start:end], source, length, anchor)
        report[segment] = {
            "source": source,
            "fps": FPS,
            "autocorr_period_frames": joint_periods,
            "selected_cycle": [start, start + length],
            "gait_period_s": length / FPS,
            "selected_cycle_speed_mps": demo_speed,
            "phase_anchor_frame": anchor,
            "foot_phase_correlation": phase_corr,
            "pose_seam_rms_rad_before_correction": float(pose_gap),
            "velocity_seam_rms_rad_per_frame": float(velocity_gap),
            "loop_max_jump_rad": loop_end_gap,
            "native_max_step_rad": native_step_max,
            "amp_interval": [minimum_start, end],
            **limit_report,
        }
        print("{}: cycle {}:{} ({} s), anchor {}, loop jump {:.4f} rad".format(
            segment, start, start + length, length / FPS, anchor, loop_end_gap))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(phase_lib, OUTPUT / "ref_lib_v1_2_phase.pt")
    torch.save(amp_lib, OUTPUT / "amp_v1_2.pt")
    (OUTPUT / "v1_2_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote v1.2 phase and AMP libraries to", OUTPUT)


if __name__ == "__main__":
    main()
