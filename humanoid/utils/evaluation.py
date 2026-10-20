"""Explicit playback domains and command injection without adding history frames."""
import numbers


def _center_range(value):
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
            isinstance(x, numbers.Real) and not isinstance(x, bool) for x in value):
        center = (value[0] + value[1]) / 2
        return [center, center]
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, (list, tuple)) for x in value):
        return [_center_range(x) for x in value]
    return value


def configure_evaluation(cfg, mode, seed):
    """Point ranges retain calibrated dynamics instead of falling back to URDF defaults."""
    if mode not in ("train", "nominal_no_push", "randomized_no_push"):
        raise ValueError("unknown evaluation mode: {}".format(mode))
    cfg.seed = seed
    cfg.env.episode_length_s = 1000
    cfg.commands.heading_command = False
    cfg.commands.external_commands = True
    cfg.env.record_diagnostics = True
    if mode != "train":
        cfg.domain_rand.push_robots = False
        cfg.domain_rand.continuous_push = False
    if mode == "nominal_no_push":
        cfg.noise.add_noise = False
        cfg.terrain.mesh_type = "plane"
        cfg.terrain.curriculum = False
        domain = cfg.domain_rand
        for name in dir(domain):
            if name.endswith("_range"):
                setattr(domain, name, _center_range(getattr(domain, name)))
        for name in ("lag_timesteps_range", "dof_lag_timesteps_range", "dof_pos_lag_timesteps_range",
                     "dof_vel_lag_timesteps_range", "imu_lag_timesteps_range"):
            if hasattr(domain, name):
                point = int(round(getattr(domain, name)[0]))
                setattr(domain, name, [point, point])
    if mode != "train" and cfg.domain_rand.push_robots:
        raise RuntimeError("evaluation must have pushes disabled")


def inject_evaluation_command(env, speed):
    if hasattr(env, "set_evaluation_command"):
        return env.set_evaluation_command(speed)
    import torch
    env.commands.zero_()
    env.commands[:, 0] = speed
    phase = env._get_phase()
    command = torch.cat((torch.sin(2 * torch.pi * phase).unsqueeze(1),
                         torch.cos(2 * torch.pi * phase).unsqueeze(1),
                         env.commands[:, :3] * env.commands_scale), dim=1)
    width = env.cfg.env.num_single_obs
    env.obs_history[-1][:, :5] = command
    env.obs_buf[:, -width:-width + 5] = command
    return env.obs_buf
