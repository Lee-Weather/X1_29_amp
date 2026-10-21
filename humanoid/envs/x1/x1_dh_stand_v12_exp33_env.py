"""Reference-based PD targets; the policy supplies only bounded angle residuals."""
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply

from humanoid.utils.evaluation import _center_range
from .x1_dh_stand_v12_exp32_env import X1DHStandV12Exp32Env


class X1DHStandV12Exp33Env(X1DHStandV12Exp32Env):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        if cfg.env.use_ref_actions:
            raise ValueError("exp3.3 must not add the legacy ref_action")
        if cfg.control.arm_action_ema_alpha != 1.0:
            raise ValueError("exp3.3 expects unfiltered policy residual actions")
        if getattr(cfg.domain_rand, "initial_domain_nominal", False):
            for name in dir(cfg.domain_rand):
                if name.endswith("_range"):
                    value = _center_range(getattr(cfg.domain_rand, name))
                    if "timesteps" in name:
                        value = [int(round(item)) for item in value]
                    setattr(cfg.domain_rand, name, value)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # X1DHStandEnv.__init__ assigns zeros after _init_buffers; restore the
        # explicit standing reference before the first reset/observation.
        self.ref_dof_pos = self._ref_q_target

    def _init_mocap_lib(self):
        super()._init_mocap_lib()
        library = torch.load(self._resolve_motion_file(self.cfg.rewards.mocap_ref_file),
                             map_location=self.device)
        metadata = library[self.seg_names[0]]["exp33_geometry"]
        self._reference_geometry = metadata
        handles = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
                   for name in metadata["foot_names"]]
        order = [handles.index(int(index)) for index in self.feet_indices]
        self._sole_normals_local = metadata["sole_normals_local"][order].to(self.device)
        # These convex-hull vertices reproduce the minimum Z of the collision
        # mesh without transferring thousands of STL vertices every reward step.
        self._sole_vertices_local = [metadata["sole_vertices_local"][i].to(self.device) for i in order]

    def _init_buffers(self):
        super()._init_buffers()
        limits = []
        for name in self.dof_names:
            matches = [value for suffix, value in self.cfg.control.residual_limit_rad.items()
                       if name.endswith(suffix)]
            if len(matches) != 1 or matches[0] <= 0:
                raise ValueError("missing/ambiguous residual limit: " + name)
            limits.append(matches[0])
        self.residual_limits = torch.tensor(limits, dtype=torch.float, device=self.device)
        self._delta_q = torch.zeros_like(self._ref_q_target)
        self._previous_delta_q = torch.zeros_like(self._delta_q)
        self._reference_for_action = self._ref_q_target.clone()
        self._phase_for_action = torch.zeros(self.num_envs, device=self.device)
        self._cycle_for_action = torch.ones(self.num_envs, device=self.device)
        self._reward_reference_active = False
        self._residual_saturation_episode = torch.zeros(self.num_envs, device=self.device)
        self._control_steps_episode = torch.zeros(self.num_envs, device=self.device)
        self.lagged_actions_scaled = self._ref_q_target.clone()
        if not self.cfg.domain_rand.add_lag:
            self.lag_timestep = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if not self.cfg.domain_rand.add_dof_lag:
            self.dof_lag_timestep = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.base_init_state[2] = float(self._reference_geometry["stand_root_height_m"])
        expected = self._reference_geometry["stand_dof_pos"].to(self.device)
        if not torch.allclose(expected, self.default_dof_pos[0], atol=1e-6):
            raise ValueError("standing pose changed; rebuild exp3.3 reference geometry")
        self.ref_dof_pos = self._ref_q_target
        print("[EXP3.3] q_target = reference + bounded residual; actor frame = 127.")

    def _get_phase(self):
        if getattr(self, "_reward_reference_active", False):
            return self._phase_for_action
        return super()._get_phase()

    def _current_cycle_time(self):
        if getattr(self, "_reward_reference_active", False):
            return self._cycle_for_action
        return super()._current_cycle_time()

    def compute_ref_state(self):
        if getattr(self, "_reward_reference_active", False):
            return
        phase = self._get_phase()
        self.seg_id = self._current_seg_id()
        walk = torch.norm(self.commands[:, :3], dim=1) > self.cfg.commands.stand_com_threshold
        # Continuous interpolation avoids repeating a 50 Hz staircase at a
        # 100 Hz policy rate, including the last-to-first cyclic boundary.
        frame = torch.remainder(phase * self.seg_period_frames[self.seg_id]
                                + self.seg_anchor[self.seg_id], self.seg_len[self.seg_id])
        first = frame.floor().long()
        second = torch.remainder(first + 1, self.seg_len[self.seg_id])
        fraction = (frame - first).unsqueeze(1)
        desired_q = torch.lerp(self.mocap_q[first, self.seg_id],
                               self.mocap_q[second, self.seg_id], fraction)
        desired_q = torch.where(walk.unsqueeze(1), desired_q, self.default_dof_pos)
        rp0, rp1 = self.mocap_rp[first, self.seg_id], self.mocap_rp[second, self.seg_id]
        rp_delta = torch.atan2(torch.sin(rp1 - rp0), torch.cos(rp1 - rp0))
        desired_rp = torch.where(walk.unsqueeze(1), rp0 + fraction * rp_delta, torch.zeros_like(rp0))
        mode = torch.where(walk, self.seg_id, torch.full_like(self.seg_id, -1))
        changed = mode != self._ref_mode
        self._ref_from_q[changed] = self._ref_q_target[changed]
        self._ref_from_rp[changed] = self._ref_rp_target[changed]
        self._ref_elapsed[changed] = 0.0
        if self._ref_update_step != self.common_step_counter:
            self._ref_elapsed[~changed] += self.dt
            self._ref_update_step = self.common_step_counter
        self._ref_mode.copy_(mode)
        blend = (self._ref_elapsed / self.cfg.rewards.v12_ref_blend_time).clamp(0, 1).unsqueeze(1)
        self._ref_q_target.copy_(torch.lerp(self._ref_from_q, desired_q, blend))
        self._ref_rp_target.copy_(torch.lerp(self._ref_from_rp, desired_rp, blend))
        self.ref_dof_pos = self._ref_q_target

    def step(self, actions):
        # This is the same reference present in the newest actor observation.
        # The post-physics callback advances the phase only for the next action.
        self._reference_for_action.copy_(self.ref_dof_pos)
        self._phase_for_action.copy_(self._get_phase())
        self._cycle_for_action.copy_(self._current_cycle_time())
        self._previous_delta_q.copy_(self._delta_q)
        clip = self.cfg.normalization.clip_actions
        bounded = actions.clamp(-clip, clip) / clip
        if not self.cfg.control.residual_enabled:
            actions = torch.zeros_like(actions)
            bounded = torch.zeros_like(bounded)
        self._residual_saturation_episode += (bounded.abs() >= 1 - 1e-6).float().mean(dim=1)
        self._control_steps_episode += 1
        return super().step(actions)

    def _pd_command_from_actions(self, actions):
        self._delta_q.copy_(actions.clamp(-self.cfg.normalization.clip_actions,
                                         self.cfg.normalization.clip_actions)
                            / self.cfg.normalization.clip_actions * self.residual_limits)
        # The existing lag buffer now delays the complete angular command, so
        # reference and residual cannot acquire different execution delays.
        return self._reference_for_action + self._delta_q

    def _pd_position_target(self):
        raw = self.lagged_actions_scaled + self.motor_offsets
        effective = torch.maximum(self._pd_lower, torch.minimum(raw, self._pd_upper))
        self._pd_target_raw.copy_(raw)
        self._pd_target_effective.copy_(effective)
        self._pd_target_clip_counts.add_((raw - effective).abs() > 1e-6)
        return effective

    def compute_reward(self):
        # Rewards describe the reference that produced this physical step,
        # including terminal snapshots taken before reset/command resampling.
        self._reward_reference_active = True
        try:
            super().compute_reward()
        finally:
            self._reward_reference_active = False
        if self.cfg.env.record_diagnostics:
            self.extras["diagnostics"].update(
                pos_des=self._pd_target_raw.clone(),
                ref_dof_pos=self._reference_for_action.clone(),
                delta_q=self._delta_q.clone(),
                pd_target_commanded=(self._reference_for_action + self._delta_q).clone(),
                pd_target_delayed=self.lagged_actions_scaled.clone(),
                sole_height=self._sole_heights().clone())

    def _reward_ref_joint_pos(self):
        cfg = self.cfg.rewards
        error = self.dof_pos - self.ref_dof_pos
        self._ref_group_costs = torch.stack([
            self._huber(error[:, group] / tolerance).mean(dim=1)
            for group, tolerance in zip(self._ref_groups, cfg.ref_group_tolerances)], dim=1)
        return -sum(weight * self._ref_group_costs[:, i]
                    for i, weight in enumerate(cfg.ref_group_weights))

    def _reward_residual_magnitude(self):
        return (self._delta_q / self.residual_limits).square().mean(dim=1)

    def _reward_residual_rate(self):
        return ((self._delta_q - self._previous_delta_q) / self.residual_limits).square().mean(dim=1)

    def _sole_heights(self):
        heights = []
        for foot, vertices in zip(self.feet_indices, self._sole_vertices_local):
            state = self.rigid_state[:, foot]
            quat = state[:, 3:7]
            # Only the Z row of the rotation is needed for a mesh's lowest point.
            qx, qy, qz, qw = quat.unbind(dim=1)
            row_z = torch.stack((2 * (qx * qz - qw * qy),
                                 2 * (qy * qz + qw * qx),
                                 1 - 2 * (qx.square() + qy.square())), dim=1)
            heights.append((row_z @ vertices.T).min(dim=1)[0] + state[:, 2])
        return torch.stack(heights, dim=1)

    def _reward_feet_rotation(self):
        local = self._sole_normals_local.unsqueeze(0).expand(self.num_envs, -1, -1)
        normal = quat_apply(self.feet_quat.reshape(-1, 4), local.reshape(-1, 3)).view(self.num_envs, 2, 3)
        tilt = torch.acos(normal[:, :, 2].clamp(-1, 1))
        support = self._get_stance_mask().bool() | (self.contact_forces[:, self.feet_indices, 2] > 5)
        reward = torch.exp(-self.cfg.rewards.feet_rotation_sigma * tilt.square())
        return (reward * support).sum(dim=1) / support.sum(dim=1).clamp(min=1)

    def _reward_foot_height(self):
        sin = torch.sin(2 * torch.pi * self._get_phase())
        amplitude = self.cfg.rewards.foot_height_target
        target = torch.stack((torch.relu(-sin), torch.relu(sin)), dim=1) * amplitude
        return torch.exp(-(self._sole_heights() - target).abs() / (0.5 * amplitude)).sum(dim=1)

    def _reward_feet_clearance(self):
        height = self._sole_heights()
        inside = ((height > self.cfg.rewards.target_feet_height)
                  & (height < self.cfg.rewards.target_feet_height_max))
        return (inside * (1 - self._get_stance_mask())).sum(dim=1)

    def _reward_base_height(self):
        support = self._get_stance_mask()
        floor = (self._sole_heights() * support).sum(dim=1) / support.sum(dim=1).clamp(min=1)
        height = self.root_states[:, 2] - floor
        return torch.exp(-(height - self.cfg.rewards.base_height_target).abs() * 100)

    def randomize_lag_props(self, env_ids):
        super().randomize_lag_props(env_ids)
        # Absolute target buffers must never be cleared to zero joint angles.
        pose = self.default_dof_pos.expand(len(env_ids), -1)
        if self.cfg.domain_rand.add_lag:
            self.lag_buffer[env_ids] = pose.unsqueeze(-1)
        if self.cfg.domain_rand.add_dof_lag:
            self.dof_lag_buffer[env_ids, :self.num_actions] = self.dof_pos[env_ids].unsqueeze(-1)
        if self.cfg.domain_rand.add_dof_pos_vel_lag:
            self.dof_pos_lag_buffer[env_ids] = self.dof_pos[env_ids].unsqueeze(-1)

    def _reset_dofs(self, env_ids):
        # Reference and initial pose agree; random phase starts are deliberately
        # disabled until a phase-consistent dynamic initialization is available.
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0
        indices = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                             gymtorch.unwrap_tensor(indices), len(indices))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        saturation = (self._residual_saturation_episode[env_ids]
                      / self._control_steps_episode[env_ids].clamp(min=1)).mean()
        super().reset_idx(env_ids)
        episode = self.extras["episode"]
        for name in list(episode):
            if name.startswith("exp32_"):
                episode[name.replace("exp32_", "exp33_", 1)] = episode.pop(name)
        episode["exp33_residual_saturation_fraction"] = saturation
        self._residual_saturation_episode[env_ids] = 0
        self._control_steps_episode[env_ids] = 0
        self._delta_q[env_ids] = 0
        self._previous_delta_q[env_ids] = 0
        self._reference_for_action[env_ids] = self.default_dof_pos
        self.lagged_actions_scaled[env_ids] = self.default_dof_pos
        self._pd_target_raw[env_ids] = self.default_dof_pos
        self._pd_target_effective[env_ids] = self.default_dof_pos
        self.ref_dof_pos = self._ref_q_target

    def set_evaluation_command(self, speed):
        super().set_evaluation_command(speed)
        # Parent updates the command prefix only. Replace reference-dependent
        # features in the same newest history frame, without appending a frame.
        reference = (self.ref_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        self.obs_history[-1][:, -self.num_actions:] = reference
        self.obs_buf[:, -self.num_actions:] = reference
        start = self.cfg.env.num_commands + 3 * self.num_actions
        error = self.dof_pos - self.ref_dof_pos
        self.critic_history[-1][:, start:start + self.num_actions] = error
        width = self.critic_history[-1].shape[1]
        offset = self.privileged_obs_buf.shape[1] - width + start
        self.privileged_obs_buf[:, offset:offset + self.num_actions] = error
        return self.obs_buf
