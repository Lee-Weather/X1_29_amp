"""exp3.1 reference targets, reward timing and reset-safe diagnostics."""
import math

import torch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse

from .x1_dh_stand_env import X1DHStandEnv, get_euler_xyz_tensor


class X1DHStandV12Exp31Env(X1DHStandEnv):
    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self._joint_property_baseline = {
                name: props[name].copy() for name in ("friction", "damping", "armature")}
        return super()._process_dof_props(props, env_id)

    def _refresh_actor_dof_props(self, env_ids):
        # Sampling a multiplier must not multiply the previous episode's sample.
        domain = self.cfg.domain_rand
        fixed_armature = getattr(domain, "fixed_armature", {}) or {}
        fixed_damping = getattr(domain, "fixed_joint_damping", {}) or {}
        for env_id in env_ids:
            props = self.gym.get_actor_dof_properties(self.envs[env_id], 0)
            for index, name in enumerate(self.dof_names):
                for prop, samples in (("friction", "joint_friction_coeffs"), ("damping", "joint_damping_coeffs")):
                    value = self._joint_property_baseline[prop][index]
                    if getattr(domain, "randomize_joint_" + prop):
                        column = index if getattr(domain, "randomize_joint_" + prop + "_each_joint") else 0
                        value *= getattr(self, samples)[env_id, column].item()
                    props[prop][index] = fixed_damping.get(name, value) if prop == "damping" else value
                value = self._joint_property_baseline["armature"][index]
                if domain.randomize_joint_armature:
                    column = index if domain.randomize_joint_armature_each_joint else 0
                    value = self.joint_armatures[env_id, column].item()
                props["armature"][index] = fixed_armature.get(name, value)
            self.gym.set_actor_dof_properties(self.envs[env_id], 0, props)

    def _init_buffers(self):
        super()._init_buffers()
        self._defer_command_sampling = False
        self._speed_ema = torch.zeros(self.num_envs, device=self.device)
        self._episode_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._ref_q_target = self.default_dof_pos.expand(self.num_envs, -1).clone()
        self._ref_rp_target = torch.zeros(self.num_envs, 2, device=self.device)
        self._ref_from_q = self._ref_q_target.clone()
        self._ref_from_rp = self._ref_rp_target.clone()
        self._ref_mode = torch.full_like(self._episode_id, -1)
        self._ref_elapsed = torch.full_like(self._speed_ema, self.cfg.rewards.v12_ref_blend_time)
        self._ref_update_step = -1
        self._command_for_action = torch.zeros(self.num_envs, device=self.device)
        self._observed_command_for_action = self._command_for_action.clone()
        self._torque_clip_step = torch.zeros_like(self._episode_id)

    def _init_mocap_lib(self):
        super()._init_mocap_lib()
        lib = torch.load(self._resolve_motion_file(self.cfg.rewards.mocap_ref_file),
                         map_location=self.device)
        self.mocap_rp = torch.zeros(*self.mocap_q.shape[:2], 2, device=self.device)
        for index, name in enumerate(self.seg_names):
            quat = lib[name]["root_rot_wxyz"].float()[:, [1, 2, 3, 0]]
            quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.mocap_rp[:len(quat), index] = get_euler_xyz_tensor(quat)[:, :2]

    def compute_ref_state(self):
        phase = self._get_phase()
        walk = torch.norm(self.commands[:, :3], dim=1) > self.cfg.commands.stand_com_threshold
        self.seg_id = self._current_seg_id()
        desired_q = self.default_dof_pos.expand(self.num_envs, -1).clone()
        desired_rp = torch.zeros_like(self._ref_rp_target)
        frames = torch.remainder((phase * self.seg_period_frames[self.seg_id]
                                  + self.seg_anchor[self.seg_id]).long(), self.seg_len[self.seg_id])
        desired_q[walk] = self.mocap_q[frames[walk], self.seg_id[walk]]
        desired_rp[walk] = self.mocap_rp[frames[walk], self.seg_id[walk]]
        mode = torch.where(walk, self.seg_id, torch.full_like(self.seg_id, -1))
        changed = mode != self._ref_mode
        self._ref_from_q[changed] = self._ref_q_target[changed]
        self._ref_from_rp[changed] = self._ref_rp_target[changed]
        self._ref_elapsed[changed] = 0.0
        if self._ref_update_step != self.common_step_counter:
            self._ref_elapsed[~changed] += self.dt
            self._ref_update_step = self.common_step_counter
        self._ref_mode.copy_(mode)
        blend = (self._ref_elapsed / self.cfg.rewards.v12_ref_blend_time).clamp(0.0, 1.0).unsqueeze(1)
        self._ref_q_target = torch.lerp(self._ref_from_q, desired_q, blend)
        self._ref_rp_target = torch.lerp(self._ref_from_rp, desired_rp, blend)
        self.ref_dof_pos = self._ref_q_target
        self.ref_action = 2 * (self.ref_dof_pos - self.default_dof_pos)

    def _resample_commands(self):
        if self._defer_command_sampling or getattr(self.cfg.commands, "external_commands", False):
            return
        super()._resample_commands()

    def _resample_walk_v12_command(self, env_ids):
        if not getattr(self.cfg.commands, "external_commands", False):
            super()._resample_walk_v12_command(env_ids)

    def _post_physics_step_callback(self):
        # Keep the action's command through rewards. Sample the next command afterwards.
        self._defer_command_sampling = True
        try:
            super()._post_physics_step_callback()
        finally:
            self._defer_command_sampling = False

    def step(self, actions):
        self._torque_clip_step.zero_()
        self._command_for_action = self.commands[:, 0].clone()
        self._observed_command_for_action = (
            self.obs_buf[:, -self.cfg.env.num_single_obs + 2] / self.commands_scale[0]).clone()
        return super().step(actions)

    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        self._torque_clip_step += (torques.abs() >= self.torque_limits - 1e-6).sum(dim=1)
        return torques

    def _sample_amp(self):
        super()._sample_amp()
        velocity = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])[:, 0]
        alpha = math.exp(-self.dt / self.cfg.rewards.speed_filter_time)
        self._speed_ema.mul_(alpha).add_(velocity * (1 - alpha))
        command = self.commands[:, 0].clone()
        gate = (command.sign() * self._speed_ema / command.abs().clamp(min=0.1)).clamp(0.0, 1.0)
        amp = self.extras["amp"]
        amp.update(direction_gate=gate.clone(), command_x=command,
                   base_vx=velocity.clone(), smoothed_vx=self._speed_ema.clone(),
                   episode_length=self.episode_length_buf.clone())

    def check_termination(self):
        super().check_termination()
        self._termination = {
            "done": self.reset_buf.clone(), "timeout": self.time_out_buf.clone(),
            "termination_height": self.root_states[:, 2] < self.cfg.termination.base_height_cutoff,
            "termination_roll": self.base_euler_xyz[:, 0].abs() > self.cfg.termination.roll_pitch_cutoff,
            "termination_pitch": self.base_euler_xyz[:, 1].abs() > self.cfg.termination.roll_pitch_cutoff,
            "termination_contact": torch.any(torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1),
        }

    def compute_reward(self):
        self.compute_ref_state()
        self.rew_buf.zero_()
        components = {}
        for name, function in zip(self.reward_names, self.reward_functions):
            value = function() * self.reward_scales[name]
            self.rew_buf += value
            self.episode_sums[name] += value
            components[name] = value.detach().clone()
        before_clip = self.rew_buf.clone()
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf.clamp_(min=0.0)
        after_clip = self.rew_buf.clone()
        if "termination" in self.reward_scales:
            value = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += value
            self.episode_sums["termination"] += value
            components["termination"] = value.detach().clone()
        diagnostic = dict(self._termination)
        diagnostic.update(episode_id=self._episode_id.clone(), command_applied=self._command_for_action.clone(),
                          command_observed=self._observed_command_for_action.clone(),
                          task_before_clip=before_clip, task_after_clip=after_clip,
                          task_reward=self.rew_buf.clone(), components=components,
                          smoothed_speed=self._speed_ema.clone(),
                          torque_clip_step=self._torque_clip_step.clone())
        if self.cfg.env.record_diagnostics:
            phase = self._get_phase()
            diagnostic.update(
                phase=phase.clone(), cycle_time=self._current_cycle_time().clone(),
                command=self.commands.clone(), base_quat=self.base_quat.clone(),
                base_pos=self.root_states[:, :3].clone(), base_lin_vel=self.base_lin_vel.clone(),
                base_ang_vel=self.base_ang_vel.clone(), dof_pos=self.dof_pos.clone(),
                dof_vel=self.dof_vel.clone(), actions=self.actions.clone(), torques=self.torques.clone(),
                pos_des=(self.lagged_actions_scaled + self.default_dof_pos).clone(),
                foot_pos=self.rigid_state[:, self.feet_indices, :3].clone(),
                foot_force=self.contact_forces[:, self.feet_indices, 2].clone(),
                ref_dof_pos=self.ref_dof_pos.clone(), ref_roll_pitch=self._ref_rp_target.clone(),
                action_lag=self.lag_timestep.clone(), dof_lag=self.dof_lag_timestep.clone())
        self.extras["diagnostics"] = diagnostic
        self.extras["episode"] = {}
        self._resample_commands()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self._episode_id[env_ids] += 1
        self._speed_ema[env_ids] = self.base_lin_vel[env_ids, 0]
        self._ref_q_target[env_ids] = self.default_dof_pos
        self._ref_from_q[env_ids] = self.default_dof_pos
        self._ref_rp_target[env_ids] = 0.0
        self._ref_from_rp[env_ids] = 0.0
        self._ref_mode[env_ids] = -1
        self._ref_elapsed[env_ids] = self.cfg.rewards.v12_ref_blend_time
        if getattr(self.cfg.commands, "external_commands", False):
            self.commands[env_ids] = 0.0

    def set_evaluation_command(self, speed):
        self.commands.zero_()
        self.commands[:, 0] = speed
        self.compute_ref_state()
        phase = self._get_phase()
        self.command_input = torch.cat((torch.sin(2 * torch.pi * phase).unsqueeze(1),
                                        torch.cos(2 * torch.pi * phase).unsqueeze(1),
                                        self.commands[:, :3] * self.commands_scale), dim=1)
        # Replace only the newest frame; do not append an extra history sample.
        self.obs_history[-1][:, :5] = self.command_input
        self.obs_buf[:, -self.cfg.env.num_single_obs:-self.cfg.env.num_single_obs + 5] = self.command_input
        self.critic_history[-1][:, :5] = self.command_input
        width = self.critic_history[-1].shape[1]
        self.privileged_obs_buf[:, -width:-width + 5] = self.command_input
        return self.obs_buf

    def _reward_foot_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        speed = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        return (torch.sqrt(speed) * contact).sum(dim=1)

    def _reward_default_joint_pos(self):
        stand = torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold
        diff = self.dof_pos - self.ref_dof_pos
        left = torch.norm(diff[:, self.leg_dof_indices[[1, 2, 5]]], dim=1)
        right = torch.norm(diff[:, self.leg_dof_indices[[7, 8, 11]]], dim=1)
        return (torch.exp(-(left + right - .1).clamp(min=0) * 100)
                - .01 * torch.norm(diff, dim=1)) * stand

    def _reward_ref_joint_pos(self):
        distance = torch.norm(self.dof_pos - self.ref_dof_pos, dim=1)
        return torch.exp(-2 * distance) - .2 * distance.clamp(0, .5)

    def _reward_hip_ref(self):
        hips = self.leg_dof_indices[[0, 6]]
        distance = torch.norm(self.dof_pos[:, hips] - self.ref_dof_pos[:, hips], dim=1)
        return torch.exp(-2 * distance) - .2 * distance.clamp(0, .5)

    def _reward_stand_still(self):
        stand = torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold
        return torch.exp(-(self.dof_pos - self.ref_dof_pos).square().sum(dim=1)) * stand

    def _reward_orientation(self):
        diff = self.base_euler_xyz[:, :2] - self._ref_rp_target
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        rpy = torch.exp(-diff.abs().sum(dim=1) * 10)
        target_quat = quat_from_euler_xyz(self._ref_rp_target[:, 0], self._ref_rp_target[:, 1],
                                         torch.zeros(self.num_envs, device=self.device))
        gravity = quat_rotate_inverse(target_quat, self.gravity_vec)
        upright = torch.exp(-torch.norm(self.projected_gravity - gravity, dim=1) * 20)
        return (rpy + upright) / 2

    def _reward_signed_speed_error(self):
        walking = torch.norm(self.commands[:, :3], dim=1) > self.cfg.commands.stand_com_threshold
        return -(self.base_lin_vel[:, 0] - self.commands[:, 0]).abs() * walking
