"""Forward-only reference supervision without adding the reference to PD targets."""
import torch
from isaacgym.torch_utils import quat_apply
from humanoid.utils.evaluation import _center_range
from .x1_dh_stand_env import X1DHStandEnv
from .x1_dh_stand_v12_exp31_env import X1DHStandV12Exp31Env


def configure_nominal_domain(cfg):
    if getattr(cfg.domain_rand, 'initial_domain_nominal', False):
        for name in dir(cfg.domain_rand):
            if name.endswith('_range'):
                value = _center_range(getattr(cfg.domain_rand, name))
                if 'timesteps' in name:
                    value = [int(round(item)) for item in value]
                setattr(cfg.domain_rand, name, value)


class X1DHStandExp34SourceEnv(X1DHStandEnv):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        configure_nominal_domain(cfg)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)


class X1DHStandV12Exp34Env(X1DHStandV12Exp31Env):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        configure_nominal_domain(cfg)
        if cfg.env.use_ref_actions or cfg.env.num_single_obs != 98:
            raise ValueError('exp3.4 requires the legacy 98-feature default+policy interface')
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.ref_dof_pos = self._ref_q_target
        print('[EXP3.4] B3-only; target=default+0.3*action; real-second command schedule')

    def _current_seg_id(self):
        return torch.zeros_like(self.phase_length_buf)

    def _init_mocap_lib(self):
        super()._init_mocap_lib()
        library = torch.load(self._resolve_motion_file(self.cfg.rewards.mocap_ref_file), map_location=self.device)
        demo = torch.load(self._resolve_motion_file(self.cfg.amp.demo_file), map_location='cpu')
        if set(library) != {'walk_forward'} or set(demo) != {'walk_forward'}:
            raise ValueError('exp3.4 must contain B3 only in both phase and AMP libraries')
        if self.seg_len[0] != self.seg_period_frames[0]:
            raise ValueError('exp3.4 phase library must contain one complete cycle')
        record = library['walk_forward']
        geometry = record['exp33_geometry']
        handles = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
                   for name in geometry['foot_names']]
        order = [handles.index(int(index)) for index in self.feet_indices]
        self._sole_normals_local = geometry['sole_normals_local'][order].to(self.device)
        self._sole_vertices_local = [geometry['sole_vertices_local'][i].to(self.device) for i in order]
        self._reference_soles = record['sole_height'][:, order].to(self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self._schedule_end = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ref_sole_target = torch.zeros(self.num_envs, 2, device=self.device)
        self.lag_timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.dof_lag_timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Keep the source model's root initialization as well as its PD interface.

    def generate_gait_time(self, env_ids):
        # Real seconds, with no normalization to the maximum episode duration.
        durations = torch.stack([torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.commands.gait_time_range[name]) for name in self.cfg.commands.gait], dim=1)
        ends = torch.cumsum(torch.round(durations / self.dt).long().clamp(min=1), dim=1)
        self.gait_time[env_ids, 0] = 0
        self.gait_time[env_ids, 1:] = ends[:, :-1].to(self.gait_time.dtype)
        self._schedule_end[env_ids] = ends[:, -1]

    def _resample_commands(self):
        if self._defer_command_sampling or getattr(self.cfg.commands, 'external_commands', False):
            return
        # At the end of the stop interval, resume walking; repeat within the episode.
        repeat = (self.episode_length_buf >= self._schedule_end) & (self.episode_length_buf > 0)
        ids = repeat.nonzero(as_tuple=False).flatten()
        if len(ids):
            now = self.episode_length_buf[ids]
            walk = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.commands.gait_time_range['walk_v12'])
            stop = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.commands.gait_time_range['stand'])
            self.gait_time[ids, 0] = -1
            self.gait_time[ids, 1] = now.to(self.gait_time.dtype)
            self.gait_time[ids, 2] = (now + torch.round(walk/self.dt).long().clamp(min=1)).to(self.gait_time.dtype)
            self._schedule_end[ids] = self.gait_time[ids, 2].long() + torch.round(stop/self.dt).long().clamp(min=1)
        for stage, name in enumerate(self.cfg.commands.gait):
            ids = (self.episode_length_buf == self.gait_time[:, stage]).nonzero(as_tuple=False).flatten()
            if len(ids):
                getattr(self, '_resample_' + name + '_command')(ids)
                if stage == 0:
                    self._yaw_ref[ids] = self.base_euler_xyz[ids, 2]

    def _resample_walk_v12_command(self, env_ids):
        low, high = self.cfg.commands.v12_forward_range
        self.commands[env_ids] = 0
        self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(low, high)

    def compute_ref_state(self):
        phase = self._get_phase()
        frame = torch.remainder(phase * self.seg_period_frames[0] + self.seg_anchor[0], self.seg_len[0])
        first = frame.floor().long()
        second = torch.remainder(first + 1, self.seg_len[0])
        fraction = (frame-first).unsqueeze(1)
        walk = self.commands[:, 0] > self.cfg.commands.stand_com_threshold
        q = torch.lerp(self.mocap_q[first, 0], self.mocap_q[second, 0], fraction)
        q = torch.where(walk.unsqueeze(1), q, self.default_dof_pos)
        rp0, rp1 = self.mocap_rp[first, 0], self.mocap_rp[second, 0]
        rp = rp0 + fraction * torch.atan2(torch.sin(rp1-rp0), torch.cos(rp1-rp0))
        rp = torch.where(walk.unsqueeze(1), rp, torch.zeros_like(rp))
        mode = torch.where(walk, torch.zeros_like(self._ref_mode), torch.full_like(self._ref_mode, -1))
        changed = mode != self._ref_mode
        self._ref_from_q[changed] = self._ref_q_target[changed]
        self._ref_from_rp[changed] = self._ref_rp_target[changed]
        self._ref_elapsed[changed] = 0
        if self._ref_update_step != self.common_step_counter:
            self._ref_elapsed[~changed] += self.dt
            self._ref_update_step = self.common_step_counter
        self._ref_mode.copy_(mode)
        blend = (self._ref_elapsed/self.cfg.rewards.v12_ref_blend_time).clamp(0,1).unsqueeze(1)
        self._ref_q_target.copy_(torch.lerp(self._ref_from_q, q, blend))
        self._ref_rp_target.copy_(torch.lerp(self._ref_from_rp, rp, blend))
        self.ref_dof_pos = self._ref_q_target
        soles = torch.lerp(self._reference_soles[first], self._reference_soles[second], fraction)
        self._ref_sole_target.copy_(torch.where(walk.unsqueeze(1), soles*blend, torch.zeros_like(soles)))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids):
            self._ref_sole_target[env_ids] = 0

    def set_evaluation_command(self, speed):
        if speed < 0:
            raise ValueError('exp3.4 trains forward motion only; select a forward-only playback profile')
        return super().set_evaluation_command(speed)

    def _sole_heights(self):
        heights = []
        for foot, vertices in zip(self.feet_indices, self._sole_vertices_local):
            state = self.rigid_state[:, foot]
            qx,qy,qz,qw = state[:,3:7].unbind(dim=1)
            row = torch.stack((2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)),dim=1)
            heights.append((row @ vertices.T).min(dim=1)[0] + state[:,2])
        return torch.stack(heights,dim=1)

    def _reward_foot_height(self):
        reward = torch.exp(-(self._sole_heights()-self._ref_sole_target).abs()/self.cfg.rewards.sole_height_tolerance)
        stand = self.commands[:,0] <= self.cfg.commands.stand_com_threshold
        return torch.where(stand, torch.full_like(stand,2,dtype=torch.float), reward.sum(dim=1))

    def _reward_feet_clearance(self):
        swing = (1-self._get_stance_mask()) * (self._ref_sole_target > .005)
        reward = torch.exp(-(self._sole_heights()-self._ref_sole_target).abs()/self.cfg.rewards.sole_height_tolerance)
        return (reward*swing).sum(dim=1)

    def _reward_feet_rotation(self):
        local = self._sole_normals_local.unsqueeze(0).expand(self.num_envs,-1,-1)
        normal = quat_apply(self.feet_quat.reshape(-1,4),local.reshape(-1,3)).view(self.num_envs,2,3)
        tilt = torch.acos(normal[:,:,2].clamp(-1,1))
        support = self._get_stance_mask().bool() | (self.contact_forces[:,self.feet_indices,2]>5)
        return (torch.exp(-self.cfg.rewards.feet_rotation_sigma*tilt.square())*support).sum(dim=1)/support.sum(dim=1).clamp(min=1)

    def _reward_base_height(self):
        support = self._get_stance_mask()
        floor = (self._sole_heights()*support).sum(dim=1)/support.sum(dim=1).clamp(min=1)
        return torch.exp(-(self.root_states[:,2]-floor-self.cfg.rewards.base_height_target).abs()*100)
