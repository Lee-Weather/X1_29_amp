"""Independent exp3.2 changes; exp3.1 remains available for reproduction."""
import torch

from .x1_dh_stand_v12_exp31_env import X1DHStandV12Exp31Env


class X1DHStandV12Exp32Env(X1DHStandV12Exp31Env):
    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            margin = self.cfg.control.pd_target_limit_margin
            lower = torch.as_tensor(props["lower"].copy(), device=self.device, dtype=torch.float)
            upper = torch.as_tensor(props["upper"].copy(), device=self.device, dtype=torch.float)
            self._pd_physical_lower = lower
            self._pd_physical_upper = upper
            self._pd_lower = lower + margin * (upper - lower)
            self._pd_upper = upper - margin * (upper - lower)
        return super()._process_dof_props(props, env_id)

    def _init_buffers(self):
        super()._init_buffers()
        # Some reference knees reach the physical lower bound. Preserve these
        # feasible targets instead of introducing an incompatible 2% margin.
        targets = torch.cat([self.mocap_q[:int(self.seg_len[i]), i]
                             for i in range(len(self.seg_names))] + [self.default_dof_pos], dim=0)
        minimum, maximum = targets.min(dim=0)[0], targets.max(dim=0)[0]
        if ((minimum < self._pd_physical_lower - 1e-5) |
                (maximum > self._pd_physical_upper + 1e-5)).any():
            raise ValueError("reference/default pose exceeds physical joint limits")
        self._pd_lower = torch.maximum(self._pd_physical_lower, torch.minimum(self._pd_lower, minimum))
        self._pd_upper = torch.minimum(self._pd_physical_upper, torch.maximum(self._pd_upper, maximum))
        self._pd_target_raw = self.default_dof_pos.expand(self.num_envs, -1).clone()
        self._pd_target_effective = self._pd_target_raw.clone()
        self._pd_target_clip_counts = torch.zeros_like(self._pd_target_raw, dtype=torch.long)
        lateral = self.leg_dof_indices[[1, 2, 5, 7, 8, 11]]
        sagittal = self.leg_dof_indices[[0, 3, 4, 6, 9, 10]]
        upper = [i for i in range(self.num_dof) if i not in self.leg_dof_indices.tolist()]
        self._ref_groups = (lateral, sagittal, torch.tensor(upper, device=self.device, dtype=torch.long))
        self._ref_group_costs = torch.zeros(self.num_envs, 3, device=self.device)
        self._pd_clip_episode = torch.zeros(self.num_envs, device=self.device)
        self._ref_cost_episode = torch.zeros(self.num_envs, 3, device=self.device)
        self._walk_steps_episode = torch.zeros(self.num_envs, device=self.device)
        print("[EXP3.2] PD targets constrained after lag and motor offset; episode heading fixed.")

    def step(self, actions):
        self._pd_target_clip_counts.zero_()
        return super().step(actions)

    def _pd_position_target(self):
        raw = super()._pd_position_target()
        effective = torch.maximum(self._pd_lower, torch.minimum(raw, self._pd_upper))
        self._pd_target_raw.copy_(raw)
        self._pd_target_effective.copy_(effective)
        self._pd_target_clip_counts.add_((raw - effective).abs() > 1e-6)
        return effective

    def _resample_commands(self):
        # The parent refreshes _yaw_ref at each stand/walk segment boundary.
        # Here straight forward/backward commands share the episode heading.
        heading = self._yaw_ref.clone() if hasattr(self, "_yaw_ref") else None
        super()._resample_commands()
        if heading is not None:
            self._yaw_ref.copy_(heading)

    def _sample_amp(self):
        super()._sample_amp()
        cfg = self.cfg.rewards
        command = self.commands[:, 0]
        signed_speed = command.sign() * self._speed_ema
        progress = (signed_speed / command.abs().clamp(min=0.1)).clamp(0, 1)
        taper = ((signed_speed - cfg.amp_gate_wrong_speed) /
                 (cfg.amp_gate_start_speed - cfg.amp_gate_wrong_speed)).clamp(0, 1)
        self.extras["amp"]["direction_gate"] = progress.clamp(min=cfg.amp_start_gate) * taper

    @staticmethod
    def _huber(value):
        magnitude = value.abs()
        return torch.where(magnitude <= 1, 0.5 * value.square(), magnitude - 0.5)

    def _reward_ref_joint_pos(self):
        cfg = self.cfg.rewards
        error = self.dof_pos - self.ref_dof_pos
        costs = [self._huber(error[:, group] / tolerance).mean(dim=1)
                 for group, tolerance in zip(self._ref_groups, cfg.ref_group_tolerances)]
        self._ref_group_costs = torch.stack(costs, dim=1)
        cost = sum(weight * value for weight, value in zip(cfg.ref_group_weights, costs))
        walk = torch.norm(self.commands[:, :3], dim=1) > self.cfg.commands.stand_com_threshold
        # Preserve exp3.1 standing contribution despite scale 0.5 -> 1.0.
        return torch.where(walk, -cost, 0.5 * super()._reward_ref_joint_pos())

    def _reward_orientation_stability(self):
        cfg = self.cfg.rewards
        error = self.base_euler_xyz[:, :2] - self._ref_rp_target
        error = torch.atan2(torch.sin(error), torch.cos(error))
        excess = (error.abs() - cfg.orientation_cost_deadband).clamp(min=0)
        return self._huber(excess / cfg.orientation_cost_tolerance).mean(dim=1)

    def _reward_height_stability(self):
        cfg = self.cfg.rewards
        deficit = (cfg.stability_min_height - self.root_states[:, 2]).clamp(min=0)
        return self._huber(deficit / cfg.stability_height_tolerance)

    def _reward_heading_error(self):
        diff = self.base_euler_xyz[:, 2] - self._yaw_ref
        error = torch.atan2(torch.sin(diff), torch.cos(diff))
        return self._huber(error / self.cfg.rewards.heading_cost_tolerance)

    def compute_reward(self):
        super().compute_reward()
        walk = self._command_for_action.abs() > self.cfg.commands.stand_com_threshold
        self._pd_clip_episode += self._pd_target_clip_counts.sum(dim=1)
        self._ref_cost_episode += self._ref_group_costs * walk.unsqueeze(1)
        self._walk_steps_episode += walk
        if not self.cfg.env.record_diagnostics:
            return
        saved = self.extras["diagnostics"]
        saved.update(pd_target_raw=self._pd_target_raw.clone(),
                     pd_target_effective=self._pd_target_effective.clone(),
                     pd_target_clip_counts=self._pd_target_clip_counts.clone(),
                     ref_group_costs=self._ref_group_costs.clone(),
                     heading_reference=self._yaw_ref.clone())

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return super().reset_idx(env_ids)
        steps = self.episode_length_buf[env_ids].clamp(min=1)
        summary = {
            "exp32_pd_target_clip_fraction": (self._pd_clip_episode[env_ids] /
                (steps * self.cfg.control.decimation * self.num_dof)).mean(),
            "exp32_walk_step_fraction": (self._walk_steps_episode[env_ids] / steps).mean(),
        }
        for index, name in enumerate(("lateral", "sagittal", "upper")):
            summary["exp32_ref_" + name + "_cost_walk"] = (
                self._ref_cost_episode[env_ids, index] /
                self._walk_steps_episode[env_ids].clamp(min=1)).mean()
        if hasattr(self, "_termination"):
            for name in ("height", "roll", "pitch", "contact"):
                summary["exp32_failure_" + name + "_fraction"] = self._termination[
                    "termination_" + name][env_ids].float().mean()
            failed = (~self.time_out_buf[env_ids]) & self.reset_buf[env_ids].bool()
            command = self._command_for_action[env_ids]
            summary["exp32_failure_forward_fraction"] = (failed & (command > .05)).float().mean()
            summary["exp32_failure_backward_fraction"] = (failed & (command < -.05)).float().mean()
        super().reset_idx(env_ids)
        self.extras.setdefault("episode", {}).update(summary)
        self._pd_clip_episode[env_ids] = 0
        self._ref_cost_episode[env_ids] = 0
        self._walk_steps_episode[env_ids] = 0
