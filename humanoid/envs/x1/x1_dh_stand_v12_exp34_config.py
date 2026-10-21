"""exp3.4: transfer a mature default+policy controller to B3 forward motion."""
from .x1_dh_stand_config import X1DHStandCfg, X1DHStandCfgPPO


class X1DHStandExp34SourceCfg(X1DHStandCfg):
    """Source-policy playback in the exact nominal domain used by exp3.4."""
    class env(X1DHStandCfg.env):
        play_speed_profile = [0.0, 0.4, 0.6, 0.0, 0.4, 0.0]

    class domain_rand(X1DHStandCfg.domain_rand):
        initial_domain_nominal = True
        push_robots = False
        continuous_push = False
        add_lag = False
        add_dof_lag = False
        add_dof_pos_vel_lag = False
        add_imu_lag = False

    class noise(X1DHStandCfg.noise):
        add_noise = False

    class terrain(X1DHStandCfg.terrain):
        mesh_type = 'plane'
        curriculum = False

    class rewards(X1DHStandCfg.rewards):
        class scales(X1DHStandCfg.rewards.scales):
            # 00dee7a, the actual source-training configuration, predates this item.
            yaw_drift_angle = 0.0


class X1DHStandV12Exp34Cfg(X1DHStandExp34SourceCfg):
    class env(X1DHStandExp34SourceCfg.env):
        observe_reference_pose = False
        num_single_obs = 98
        num_observations = 66 * num_single_obs
        use_ref_actions = False
        record_diagnostics = False

    class commands(X1DHStandExp34SourceCfg.commands):
        forward_only = True
        curriculum = False
        external_commands = False
        heading_command = False
        gait = ['stand', 'walk_v12', 'stand']
        gait_time_range = {'stand': [2.0, 3.0], 'walk_v12': [6.0, 9.0]}
        v12_forward_range = [0.35, 0.65]
        v12_backward_probability = 0.0
        v12_walk_from_reset_probability = 0.0

        class ranges(X1DHStandExp34SourceCfg.commands.ranges):
            lin_vel_x = [0.35, 0.65]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class control(X1DHStandExp34SourceCfg.control):
        action_scale = 0.3
        arm_action_ema_alpha = 1.0

    class rewards(X1DHStandExp34SourceCfg.rewards):
        reference_mode = 'v12_forward'
        reference_pose_tracking = True
        mocap_ref_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/processed/ref_lib_v1_2_exp34_phase.pt'
        v12_ref_blend_time = 0.3
        speed_filter_time = 0.1
        gait_speed_adaptive = True
        gait_scale_clamp = [0.8, 3.0]
        seg_demo_speed_table = {}
        only_positive_rewards = True
        sole_height_tolerance = 0.025
        feet_rotation_sigma = 15.0

        class scales(X1DHStandExp34SourceCfg.rewards.scales):
            ref_joint_pos = 0.5

    class amp(X1DHStandExp34SourceCfg.amp):
        demo_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/processed/amp_v1_2_exp34.pt'
        demo_seg_weights = [1.0]


class X1DHStandV12Exp34CfgPPO(X1DHStandCfgPPO):
    class policy(X1DHStandCfgPPO.policy):
        zero_init_actor_output = False

    class algorithm(X1DHStandCfgPPO.algorithm):
        amp_style_reward_scale = 5.0
        amp_task_lerp = 0.5
        amp_direction_gate = False  # Preserve the source fusion interface; B3 is forward-only.

    class runner(X1DHStandCfgPPO.runner):
        experiment_name = 'x1_dh_stand_v12_exp34'
        run_name = 'exp3_4_b3_transfer'
        resume = False
        max_iterations = 12000  # Additional updates after the pinned source checkpoint.
        transfer_manifest = '{LEGGED_GYM_ROOT_DIR}/resources/motions/processed/v1_2_exp34_manifest.json'
