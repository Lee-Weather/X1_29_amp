"""exp3.2: reachable PD targets and phase-aware stable sagittal walking."""
from .x1_dh_stand_v12_exp31_config import X1DHStandV12Exp31Cfg, X1DHStandV12Exp31CfgPPO


class X1DHStandV12Exp32Cfg(X1DHStandV12Exp31Cfg):
    class control(X1DHStandV12Exp31Cfg.control):
        pd_target_limit_margin = 0.02

    class rewards(X1DHStandV12Exp31Cfg.rewards):
        ref_group_tolerances = [0.2, 0.35, 0.4]
        ref_group_weights = [2.0, 1.0, 0.2]
        orientation_cost_deadband = 0.1
        orientation_cost_tolerance = 0.2
        stability_min_height = 0.55
        stability_height_tolerance = 0.1
        heading_cost_tolerance = 0.35
        amp_start_gate = 0.2
        amp_gate_wrong_speed = -0.08
        amp_gate_start_speed = -0.03

        class scales(X1DHStandV12Exp31Cfg.rewards.scales):
            signed_speed_error = 8.0
            foot_height = 0.25
            ref_joint_pos = 1.0
            orientation_stability = -2.0
            height_stability = -2.0
            heading_error = -0.5
            # dt=0.01: -30 task / -15 walking fused, once per physical failure.
            termination = -3000.0


class X1DHStandV12Exp32CfgPPO(X1DHStandV12Exp31CfgPPO):
    class runner(X1DHStandV12Exp31CfgPPO.runner):
        experiment_name = "x1_dh_stand_v12_exp32"
        run_name = "exp3_2_reachable_stable"
        resume = False
        max_iterations = 6000
