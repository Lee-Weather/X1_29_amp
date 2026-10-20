"""Capture one playback row, preferring pre-reset simulator snapshots."""
import math

import torch


def capture_playback_step(env, infos, rewards, dones, algorithm, requested, stage, clip_count, robot=0):
    saved = infos.get("diagnostics", {})
    scalar = lambda value: value[robot].item()
    position = saved.get("base_pos", env.root_states[:, :3])
    quaternion = saved.get("base_quat", env.root_states[:, 3:7])
    velocity = saved.get("base_lin_vel", env.base_lin_vel)
    angular = saved.get("base_ang_vel", env.base_ang_vel)
    command = saved.get("command", env.commands)
    qx, qy, qz, qw = quaternion[robot].tolist()
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx*qx + qy*qy))
    pitch = math.asin(max(-1., min(1., 2 * (qw*qy - qz*qx))))
    yaw = math.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
    phase = scalar(saved["phase"] if "phase" in saved else env._get_phase()) % 1.
    cycle = env.cfg.rewards.cycle_time
    if abs(scalar(command[:, 0])) > env.cfg.commands.stand_com_threshold:
        cycle = scalar(saved["cycle_time"] if "cycle_time" in saved else env._current_cycle_time())
    vx = scalar(velocity[:, 0])
    ema = getattr(env, "_play_speed_ema", 0.) * math.exp(-env.dt / .1) + vx * (1 - math.exp(-env.dt / .1))
    env._play_speed_ema = 0. if bool(dones[robot]) else ema
    episode = int(scalar(saved["episode_id"])) if "episode_id" in saved else getattr(env, "_play_episode_id", 0)
    if episode != getattr(env, "_play_previous_episode", None):
        env._play_yaw_origin = yaw
        env._play_yaw_last = yaw
        env._play_yaw_accum = 0.
    env._play_yaw_accum += math.atan2(math.sin(yaw - env._play_yaw_last), math.cos(yaw - env._play_yaw_last))
    env._play_yaw_last = yaw
    env._play_previous_episode = episode
    row = dict(
        phase_sin=math.sin(2*math.pi*phase), phase_cos=math.cos(2*math.pi*phase),
        cycle_time=cycle, smoothed_speed=scalar(saved["smoothed_speed"]) if "smoothed_speed" in saved else ema,
        active_stage=stage, cmd_linear_x=scalar(command[:, 0]), cmd_linear_y=scalar(command[:, 1]),
        cmd_angular_z=scalar(command[:, 2]), command_x=scalar(command[:, 0]),
        command_requested=float(requested),
        command_applied=scalar(saved.get("command_applied", command[:, 0])),
        command_observed=scalar(saved.get("command_observed", env.obs_buf[:, -env.cfg.env.num_single_obs+2] / env.commands_scale[0])),
        base_euler_x=roll, base_euler_y=pitch, base_euler_z=yaw,
        base_ang_vel_x=scalar(angular[:, 0]), base_ang_vel_y=scalar(angular[:, 1]), base_ang_vel_z=scalar(angular[:, 2]),
        base_vel_x=vx, base_vel_y=scalar(velocity[:, 1]), base_vel_z=scalar(velocity[:, 2]),
        base_height=scalar(position[:, 2]), base_pos_x=scalar(position[:, 0]), base_pos_y=scalar(position[:, 1]),
        base_yaw=yaw, yaw_from_episode_start=env._play_yaw_accum,
        episode_id=episode, done=int(bool(dones[robot])),
        timeout=int(bool(saved.get("timeout", infos.get("time_outs", torch.zeros_like(dones)))[robot])),
        terminal_snapshot=int("base_pos" in saved),
    )
    for name in ("termination_height", "termination_roll", "termination_pitch", "termination_contact"):
        row[name] = int(bool(saved[name][robot])) if name in saved else 0
    feet = saved.get("foot_pos", env.rigid_state[:, env.feet_indices, :3])
    force = saved.get("foot_force", env.contact_forces[:, env.feet_indices, 2])
    for index, side in enumerate(("l", "r")):
        for axis, suffix in enumerate(("x", "y", "z")):
            row["foot_{}_{}".format(suffix, side)] = scalar(feet[:, index, axis])
        row["foot_force_" + side] = scalar(force[:, index])
    actions = saved.get("actions", env.actions)
    torque = saved.get("torques", env.torques)
    delta = int(scalar(saved["torque_clip_step"])) if "torque_clip_step" in saved else int(
        (torque[robot].abs() >= env.torque_limits - 1e-6).sum().item())
    row.update(torque_clip_step=delta, clip_count=clip_count + delta,
               action_clip_step=int((actions[robot].abs() >= env.cfg.normalization.clip_actions-1e-6).sum().item()))
    pos = saved.get("dof_pos", env.dof_pos)
    vel = saved.get("dof_vel", env.dof_vel)
    desired = saved.get("pos_des", env.lagged_actions_scaled + env.default_dof_pos)
    for index, name in enumerate(env.dof_names):
        for prefix, tensor in (("action", actions), ("pos", pos), ("vel", vel), ("effort", torque), ("pos_des_raw", desired)):
            row[prefix + "_" + name] = scalar(tensor[:, index])
        if "ref_dof_pos" in saved:
            row["ref_pos_" + name] = scalar(saved["ref_dof_pos"][:, index])
    if "ref_roll_pitch" in saved:
        row.update(ref_roll=scalar(saved["ref_roll_pitch"][:, 0]), ref_pitch=scalar(saved["ref_roll_pitch"][:, 1]))
    for name in ("action_lag", "dof_lag"):
        if name in saved:
            row[name + "_substeps"] = scalar(saved[name])
    row.update(task_before_clip=scalar(saved.get("task_before_clip", rewards)),
               task_after_clip=scalar(saved.get("task_after_clip", rewards)), task_reward=scalar(rewards))
    for name, value in saved.get("components", {}).items():
        row["reward_" + name] = scalar(value)
    if "amp" in infos and getattr(algorithm, "amp_discriminator", None) is not None:
        reward = algorithm.compute_amp_reward(rewards, dones, infos["amp"])
        for name, value in reward.items():
            row["amp_" + name] = scalar(value)
    if bool(dones[robot]):
        env._play_episode_id = episode + 1
    return row
