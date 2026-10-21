"""Write the reference/residual interface alongside an exported policy."""
import hashlib
import json
from pathlib import Path
import shutil

import torch

from humanoid import LEGGED_GYM_ROOT_DIR


def export_residual_control_spec(cfg, output_dir):
    if not getattr(cfg.env, "observe_reference_pose", False):
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(cfg.rewards.mocap_ref_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    library = torch.load(str(source), map_location="cpu")
    record = library[sorted(library)[0]]
    names = list(record["dof_names"])
    residual = [next(value for suffix, value in cfg.control.residual_limit_rad.items()
                     if name.endswith(suffix)) for name in names]
    bundled = output / source.name
    if source.resolve() != bundled.resolve():
        shutil.copy2(str(source), str(bundled))
    spec = dict(
        control="reference_plus_bounded_angle_residual", joint_names=names,
        policy_output="raw residual action, not joint angles or torques",
        residual_formula="delta_q = residual_limit_rad * clip(action, -clip_actions, clip_actions) / clip_actions",
        target_formula="q_target = clamp(delay(q_ref + delta_q) + motor_offset, target_lower, target_upper)",
        torque_formula="tau = clamp(Kp*(q_target-q) - Kd*dq - friction, -torque_limit, torque_limit)",
        residual_limit_rad=residual, clip_actions=cfg.normalization.clip_actions,
        reference_file=bundled.name, reference_sha256=hashlib.sha256(bundled.read_bytes()).hexdigest(),
        stand_dof_pos=[cfg.init_state.default_joint_angles[name] for name in names],
        stand_root_height_m=record["exp33_geometry"]["stand_root_height_m"],
        stiffness_by_joint_suffix=cfg.control.stiffness, damping_by_joint_suffix=cfg.control.damping,
        pd_target_limit_margin=cfg.control.pd_target_limit_margin,
        target_limits_include_reference_extrema=True,
        robot_asset=cfg.asset.file,
        control_dt=cfg.sim.dt * cfg.control.decimation, physics_dt=cfg.sim.dt,
        reference_interpolation="cyclic linear; frame = phase*period_frames + anchor",
        reference_blend_time=cfg.rewards.v12_ref_blend_time,
        gait_scale_clamp=cfg.rewards.gait_scale_clamp,
        stand_command_threshold=cfg.commands.stand_com_threshold,
        phase_reset="stand: zero; walking resumes from sampled 0 or 0.5 gait_start",
        observation=dict(single_frame=cfg.env.num_single_obs,
                         frame_stack=cfg.env.frame_stack, short_frame_stack=cfg.env.short_frame_stack,
                         layout=["sin_phase, cos_phase, vx_cmd, vy_cmd, yaw_rate_cmd (5)",
                                 "scaled q-default (29)", "scaled dq (29)", "previous clipped raw action (29)",
                                 "scaled base angular velocity (3)", "scaled base Euler angles (3)",
                                 "scaled q_ref-default (29; no observation noise)"],
                         dof_pos_scale=cfg.normalization.obs_scales.dof_pos,
                         dof_vel_scale=cfg.normalization.obs_scales.dof_vel,
                         lin_vel_scale=cfg.normalization.obs_scales.lin_vel,
                         ang_vel_scale=cfg.normalization.obs_scales.ang_vel,
                         quat_scale=cfg.normalization.obs_scales.quat),
        clip_observations=cfg.normalization.clip_observations,
        note="The JIT exports raw actions. Reference generation, target delay, joint limits and PD remain in the controller.")
    (output / "residual_control.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
