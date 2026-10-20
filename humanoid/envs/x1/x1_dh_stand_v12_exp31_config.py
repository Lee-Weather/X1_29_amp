"""exp3.1: signed velocity control with a smaller, direction-gated AMP bonus."""
from .x1_dh_stand_v12_config import X1DHStandV12Cfg, X1DHStandV12CfgPPO


class X1DHStandV12Exp31Cfg(X1DHStandV12Cfg):
    class env(X1DHStandV12Cfg.env):
        record_diagnostics = False

    class commands(X1DHStandV12Cfg.commands):
        external_commands = False
        v12_forward_range = [0.35, 0.65]
        v12_backward_range = [-0.35, -0.15]

        class ranges(X1DHStandV12Cfg.commands.ranges):
            lin_vel_x = [-0.35, 0.65]

    class domain_rand(X1DHStandV12Cfg.domain_rand):
        push_robots = False
        continuous_push = False

    class rewards(X1DHStandV12Cfg.rewards):
        gait_scale_clamp = [0.8, 3.0]
        only_positive_rewards = False
        reference_pose_tracking = True
        v12_ref_blend_time = 0.3
        speed_filter_time = 0.1

        class scales(X1DHStandV12Cfg.rewards.scales):
            foot_place = 0.0
            low_speed = 0.0
            signed_speed_error = 4.0
            # Reward scales are multiplied by the control dt (0.01 s).
            termination = -100.0


class X1DHStandV12Exp31CfgPPO(X1DHStandV12CfgPPO):
    class algorithm(X1DHStandV12CfgPPO.algorithm):
        amp_style_reward_scale = 5.0
        amp_task_lerp = 0.5
        amp_direction_gate = True

    class runner(X1DHStandV12CfgPPO.runner):
        experiment_name = "x1_dh_stand_v12_exp31"
        run_name = "exp3_1_direction_task"
        resume = False
        max_iterations = 6000
