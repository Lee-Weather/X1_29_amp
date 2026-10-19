"""First v1.2 experiment: stand, straight forward walk, straight backward walk."""

from .x1_dh_stand_config import X1DHStandCfg, X1DHStandCfgPPO


class X1DHStandV12Cfg(X1DHStandCfg):
    class commands(X1DHStandCfg.commands):
        curriculum = False  # Keep the command range inside the two demonstrations.
        gait = ["stand", "walk_v12", "stand"]
        gait_time_range = {"stand": [3, 5], "walk_v12": [6, 9]}
        v12_forward_range = [0.45, 0.75]
        v12_backward_range = [-0.40, -0.30]
        v12_backward_probability = 0.5
        v12_walk_from_reset_probability = 0.5

        class ranges(X1DHStandCfg.commands.ranges):
            lin_vel_x = [-0.40, 0.75]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(X1DHStandCfg.rewards):
        reference_mode = "v12_sagittal"
        mocap_ref_file = "{LEGGED_GYM_ROOT_DIR}/resources/motions/processed/ref_lib_v1_2_phase.pt"
        gait_speed_adaptive = True
        gait_scale_clamp = [0.8, 1.5]
        seg_demo_speed_table = {}
        v12_ref_blend_time = 0.3
        foot_height_target = 0.05

    class amp(X1DHStandCfg.amp):
        demo_file = "{LEGGED_GYM_ROOT_DIR}/resources/motions/processed/amp_v1_2.pt"
        # Sorted segment names: walk_backward, walk_forward.
        demo_seg_weights = [1.0, 1.0]


class X1DHStandV12CfgPPO(X1DHStandCfgPPO):
    class runner(X1DHStandCfgPPO.runner):
        experiment_name = "x1_dh_stand_v12"
        run_name = "b3_b4_sagittal"
        resume = False
