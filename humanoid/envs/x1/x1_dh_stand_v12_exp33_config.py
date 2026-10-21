"""exp3.3: reference joint targets with bounded policy angle residuals."""
from .x1_dh_stand_v12_exp32_config import X1DHStandV12Exp32Cfg, X1DHStandV12Exp32CfgPPO


class X1DHStandV12Exp33Cfg(X1DHStandV12Exp32Cfg):
    class env(X1DHStandV12Exp32Cfg.env):
        observe_reference_pose = True
        num_single_obs = 127  # Existing 98 + 29 reference angles relative to default.
        num_observations = X1DHStandV12Exp32Cfg.env.frame_stack * num_single_obs
        use_ref_actions = False  # No legacy ref_action addition or double scaling.

    class commands(X1DHStandV12Exp32Cfg.commands):
        # Reset in standing pose; enter walking through the scheduled blend.
        v12_walk_from_reset_probability = 0.0

    class control(X1DHStandV12Exp32Cfg.control):
        residual_enabled = True  # False selects a zero-residual tracking baseline.
        residual_limit_rad = {
            "hip_pitch_joint": 0.20, "hip_roll_joint": 0.10, "hip_yaw_joint": 0.10,
            "knee_pitch_joint": 0.20, "ankle_pitch_joint": 0.12, "ankle_roll_joint": 0.12,
            "lumbar_yaw_joint": 0.08, "lumbar_roll_joint": 0.08, "lumbar_pitch_joint": 0.08,
            "shoulder_pitch_joint": 0.10, "shoulder_roll_joint": 0.10,
            "shoulder_yaw_joint": 0.10, "elbow_pitch_joint": 0.10,
            "elbow_yaw_joint": 0.10, "wrist_pitch_joint": 0.10, "wrist_roll_joint": 0.10,
        }
        arm_action_ema_alpha = 1.0

    class domain_rand(X1DHStandV12Exp32Cfg.domain_rand):
        # Initial learning uses nominal calibrated dynamics. Restore these only
        # after the reference/residual controller has been replayed successfully.
        initial_domain_nominal = True  # Point ranges retain calibrated armature/friction.
        add_lag = False
        add_dof_lag = False
        add_dof_pos_vel_lag = False
        add_imu_lag = False

    class noise(X1DHStandV12Exp32Cfg.noise):
        add_noise = False

    class terrain(X1DHStandV12Exp32Cfg.terrain):
        mesh_type = "plane"
        curriculum = False

    class rewards(X1DHStandV12Exp32Cfg.rewards):
        mocap_ref_file = "{LEGGED_GYM_ROOT_DIR}/resources/motions/processed/ref_lib_v1_2_exp33_phase.pt"
        feet_rotation_sigma = 15.0

        class scales(X1DHStandV12Exp32Cfg.rewards.scales):
            # One dense grouped pose cost in both standing and walking.
            default_joint_pos = 0.0
            stand_still = 0.0
            action_smoothness = 0.0  # Replaced by normalized bounded-residual costs.
            residual_magnitude = -0.10
            residual_rate = -0.02

    class amp(X1DHStandV12Exp32Cfg.amp):
        demo_file = "{LEGGED_GYM_ROOT_DIR}/resources/motions/processed/amp_v1_2_exp33.pt"


class X1DHStandV12Exp33CfgPPO(X1DHStandV12Exp32CfgPPO):
    class policy(X1DHStandV12Exp32CfgPPO.policy):
        zero_init_actor_output = True
        init_noise_std = 0.3  # Leg residual std <= 0.20 * 0.3 / 3 = 0.02 rad initially.

    class runner(X1DHStandV12Exp32CfgPPO.runner):
        experiment_name = "x1_dh_stand_v12_exp33"
        run_name = "exp3_3_reference_residual"
        resume = False
        max_iterations = 6000
