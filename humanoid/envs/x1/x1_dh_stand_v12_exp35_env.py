"""Explicit posture rewards and gradual upper-body PD target bounds."""
import json
import torch
from humanoid import LEGGED_GYM_ROOT_DIR
from .x1_dh_stand_env import X1DHStandEnv
from .x1_dh_stand_v12_exp31_env import X1DHStandV12Exp31Env
from .x1_dh_stand_v12_exp34_env import X1DHStandV12Exp34Env


class X1DHStandV12Exp35Env(X1DHStandV12Exp34Env):
    def _init_buffers(self):
        super()._init_buffers()
        self._upper_ids = torch.tensor([i for i, n in enumerate(self.dof_names)
                                       if any(p in n for p in ('shoulder', 'elbow', 'wrist'))], device=self.device)
        self._waist_ids = torch.tensor([i for i, n in enumerate(self.dof_names) if 'lumbar' in n], device=self.device)
        path = self.cfg.control.target_bounds_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        with open(path) as stream:
            bounds = json.load(stream)['bounds']
        names = [n for n in self.dof_names if n in bounds]
        if len(names) != 17:
            raise ValueError('exp3.5 expects 17 bounded upper-body joints')
        self._bound_ids = torch.tensor([self.dof_names.index(n) for n in names], device=self.device)
        self._bound_lower = torch.tensor([bounds[n][0] for n in names], device=self.device)
        self._bound_upper = torch.tensor([bounds[n][1] for n in names], device=self.device)
        if not torch.all(self._bound_lower < self._bound_upper):
            raise ValueError('invalid exp3.5 target bounds')
        self._target_raw = self.default_dof_pos.expand(self.num_envs, -1).clone()
        self._target_effective = self._target_raw.clone()
        self._bound_clip_counts = torch.zeros_like(self._target_raw, dtype=torch.long)
        self._bounds_evaluation = False

    def _bounds_strength(self):
        if self._bounds_evaluation:
            return self.cfg.control.evaluation_bounds_strength
        elapsed = self.common_step_counter - self.cfg.control.bounds_warmup_steps
        return min(1.0, max(0.0, elapsed / max(1, self.cfg.control.bounds_ramp_steps)))

    def _pd_position_target(self):
        raw = super()._pd_position_target()
        if not hasattr(self, '_bound_ids'):
            return raw
        target = raw.clone()
        selected = raw[:, self._bound_ids]
        clipped = torch.maximum(torch.minimum(selected, self._bound_upper), self._bound_lower)
        target[:, self._bound_ids] = torch.lerp(selected, clipped, self._bounds_strength())
        self._target_raw.copy_(raw)
        self._target_effective.copy_(target)
        self._bound_clip_counts += (target - raw).abs() > 1e-6
        return target

    def step(self, actions):
        self._bound_clip_counts.zero_()
        return super().step(actions)

    def set_evaluation_command(self, speed):
        self._bounds_evaluation = True
        return super().set_evaluation_command(speed)

    # Select each posture reward explicitly: exp3.1 gated the first to standing.
    def _reward_default_joint_pos(self):
        return X1DHStandEnv._reward_default_joint_pos(self)

    def _reward_ref_joint_pos(self):
        return X1DHStandV12Exp31Env._reward_ref_joint_pos(self)

    def _reward_hip_ref(self):
        return X1DHStandV12Exp31Env._reward_hip_ref(self)

    def _reward_stand_still(self):
        return X1DHStandV12Exp31Env._reward_stand_still(self)

    def _reward_orientation(self):
        return X1DHStandV12Exp31Env._reward_orientation(self)

    def _reward_yaw_drift_angle(self):
        # Reference is initialized at reset; repeated forward/stop stages do not refresh it.
        return X1DHStandEnv._reward_yaw_drift_angle(self)

    def _pose_cost(self, indices, tolerance):
        error = (self.dof_pos[:, indices] - self.ref_dof_pos[:, indices]).abs() / tolerance
        return torch.where(error < 1, 0.5 * error.square(), error - 0.5).mean(dim=1)

    def _reward_upper_pose(self):
        return self._pose_cost(self._upper_ids, self.cfg.rewards.upper_pose_tolerance)

    def _reward_waist_pose(self):
        return self._pose_cost(self._waist_ids, self.cfg.rewards.waist_pose_tolerance)

    def _reward_leg_target_overflow(self):
        # Use the live legacy command: legs are penalized, not abruptly clamped.
        raw = self.lagged_actions_scaled + self.default_dof_pos + self.motor_offsets
        ids = self.leg_dof_indices
        overflow = (self.dof_pos_limits[ids, 0] - raw[:, ids]).clamp(min=0)
        overflow += (raw[:, ids] - self.dof_pos_limits[ids, 1]).clamp(min=0)
        return overflow.mean(dim=1) / 0.2

    def compute_reward(self):
        super().compute_reward()
        if not hasattr(self, '_target_raw'):
            return
        diagnostic = self.extras['diagnostics']
        diagnostic.update(exp35_target_raw=self._target_raw.clone(),
                          exp35_target_effective=self._target_effective.clone(),
                          exp35_target_clip_counts=self._bound_clip_counts.clone(),
                          exp35_bounds_strength=torch.full((self.num_envs,), self._bounds_strength(), device=self.device),
                          exp35_upper_cost=self._reward_upper_pose().detach(),
                          exp35_waist_cost=self._reward_waist_pose().detach())
        # Parent diagnostics record the legacy target; expose actual applied target too.
        if self.cfg.env.record_diagnostics:
            diagnostic['pos_des'] = self._target_effective.clone()
