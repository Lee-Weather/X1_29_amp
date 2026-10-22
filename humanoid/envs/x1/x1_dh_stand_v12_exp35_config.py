"""exp3.5: retain mature leg control while restoring posture supervision."""
from .x1_dh_stand_v12_exp34_config import X1DHStandV12Exp34Cfg, X1DHStandV12Exp34CfgPPO


class X1DHStandV12Exp35Cfg(X1DHStandV12Exp34Cfg):
    class control(X1DHStandV12Exp34Cfg.control):
        target_bounds_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/processed/v1_2_exp35_target_bounds.json'
        bounds_warmup_steps = 24000  # 1000 PPO updates, 24 control steps/update
        bounds_ramp_steps = 24000
        evaluation_bounds_strength = 1.0

    class rewards(X1DHStandV12Exp34Cfg.rewards):
        upper_pose_tolerance = 0.25
        waist_pose_tolerance = 0.15

        class scales(X1DHStandV12Exp34Cfg.rewards.scales):
            default_joint_pos = 1.0
            yaw_drift_angle = 0.5
            upper_pose = -0.5
            waist_pose = -0.5
            leg_target_overflow = -0.1


class X1DHStandV12Exp35CfgPPO(X1DHStandV12Exp34CfgPPO):
    class runner(X1DHStandV12Exp34CfgPPO.runner):
        experiment_name = 'x1_dh_stand_v12_exp35'
        run_name = 'exp3_5_posture_transfer'
        max_iterations = 6000
        # Same verified B3 assets and healthy source checkpoint as exp3.4.
