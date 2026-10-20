"""CPU contracts for exp3.1; no Isaac Gym installation required.

Run: python scripts/tools/test_exp31.py
Simulator-independent source is loaded directly. Only the environment's pure
tensor methods are exercised here; this does not validate PhysX or training.
"""
import ast
from collections import deque
import importlib.util
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward = load_file("reward31", "humanoid/algo/amp/reward.py")
evaluation = load_file("evaluation31", "humanoid/utils/evaluation.py")
diagnostics = load_file("diagnostics31", "humanoid/utils/playback_diagnostics.py")


def load_classes(path, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(tree, str(path), "exec"), namespace)


config = {"inspect": inspect}
for path in ("humanoid/envs/base/base_config.py", "humanoid/envs/base/legged_robot_config.py",
             "humanoid/envs/x1/x1_dh_stand_config.py", "humanoid/envs/x1/x1_dh_stand_v12_config.py",
             "humanoid/envs/x1/x1_dh_stand_v12_exp31_config.py"):
    load_classes(path, config)


def quat_from_euler(roll, pitch, yaw):
    angles = torch.stack([roll, pitch, yaw], -1).numpy()
    return torch.tensor(Rotation.from_euler("xyz", angles).as_quat(), dtype=torch.float32)


def rotate_inverse(quat, vector):
    return torch.tensor(Rotation.from_quat(quat.numpy()).inv().apply(vector.numpy()), dtype=torch.float32)


class Parent:
    def _get_phase(self):
        return self.phase

    def _current_seg_id(self):
        return (self.commands[:, 0] >= 0).long()

    def _current_cycle_time(self):
        return torch.ones(self.num_envs)


namespace = dict(torch=torch, math=math, X1DHStandEnv=Parent,
                 quat_from_euler_xyz=quat_from_euler, quat_rotate_inverse=rotate_inverse)
load_classes("humanoid/envs/x1/x1_dh_stand_v12_exp31_env.py", namespace)
Environment = namespace["X1DHStandV12Exp31Env"]


def fixture():
    env = Environment()
    env.cfg = config["X1DHStandV12Exp31Cfg"]()
    env.num_envs, env.num_actions, env.device, env.dt = 1, 29, "cpu", .01
    env.common_step_counter = 0
    lib = torch.load(ROOT / "resources/motions/processed/ref_lib_v1_2_phase.pt",
                     map_location="cpu", weights_only=False)
    env.dof_names = lib["walk_forward"]["dof_names"]
    env.seg_names = sorted(lib)
    env.commands = torch.zeros(1, 4)
    env.commands_scale = torch.tensor([2., 2., 1.])
    env.phase = torch.tensor([.25])
    env.default_dof_pos = torch.tensor([[env.cfg.init_state.default_joint_angles[n] for n in env.dof_names]])
    env._ref_q_target = env.default_dof_pos.clone()
    env._ref_rp_target = torch.zeros(1, 2)
    env._ref_from_q = env._ref_q_target.clone()
    env._ref_from_rp = env._ref_rp_target.clone()
    env._ref_mode = torch.tensor([-1])
    env._ref_elapsed = torch.tensor([.3])
    env._ref_update_step = -1
    env.seg_period_frames = torch.tensor([len(lib[n]["dof_pos"]) for n in env.seg_names])
    env.seg_len = env.seg_period_frames.clone()
    env.seg_anchor = torch.tensor([lib[n]["phase_anchor_frame"] for n in env.seg_names])
    env.mocap_q = torch.zeros(73, 2, 29)
    env.mocap_rp = torch.zeros(73, 2, 2)
    for index, name in enumerate(env.seg_names):
        q = lib[name]["dof_pos"]
        env.mocap_q[:len(q), index] = q
        quat = lib[name]["root_rot_wxyz"].numpy()[:, [1, 2, 3, 0]]
        env.mocap_rp[:len(q), index] = torch.tensor(Rotation.from_quat(quat).as_euler("xyz")[:, :2])
    env.leg_dof_indices = torch.tensor([env.dof_names.index(n) for n in (
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_pitch_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_hip_yaw_joint", "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint")])
    env.obs_history = deque([torch.zeros(1, 98), torch.zeros(1, 98)], maxlen=2)
    env.obs_buf = torch.zeros(1, 196)
    env.critic_history = deque([torch.zeros(1, 141), torch.zeros(1, 141)], maxlen=2)
    env.privileged_obs_buf = torch.zeros(1, 282)
    env.dof_pos = env.default_dof_pos.clone()
    env.dof_vel = torch.zeros(1, 29)
    env.actions = torch.zeros(1, 29)
    env.torques = torch.zeros(1, 29)
    env.torque_limits = torch.full((29,), 100.)
    env.lagged_actions_scaled = torch.zeros(1, 29)
    env.root_states = torch.zeros(1, 13)
    env.root_states[:, 2] = .7
    env.root_states[:, 6] = 1.
    env.base_lin_vel = torch.zeros(1, 3)
    env.base_ang_vel = torch.zeros(1, 3)
    env.rigid_state = torch.zeros(1, 2, 13)
    env.feet_indices = torch.tensor([0, 1])
    env.contact_forces = torch.zeros(1, 2, 3)
    return env


class Contracts(unittest.TestCase):
    def test_reward_gates_and_signed_costs(self):
        task = torch.tensor([-.2, -.2, -.2, -.2, -.2])
        raw = torch.tensor([.8, .8, -.4, .8, .8])
        stand = torch.tensor([False, False, False, True, False])
        done = torch.tensor([False, False, False, False, True])
        result = reward.fuse_amp_rewards(task, raw, stand, done, .5, torch.tensor([0., .5, 0., 1., 1.]))
        torch.testing.assert_close(result["style_used"], torch.tensor([0., .4, -.4, 0., 0.]))
        torch.testing.assert_close(result["fused"], torch.tensor([-.1, .1, -.3, -.2, -.1]))
        old = reward.fuse_amp_rewards(task, raw, stand, done, .5)
        torch.testing.assert_close(old["fused"], torch.where(stand, task, .5 * task + .5 * raw))

    def test_config_isolation_and_nominal_domain(self):
        new, old = config["X1DHStandV12Exp31Cfg"](), config["X1DHStandV12Cfg"]()
        self.assertFalse(new.domain_rand.push_robots)
        self.assertTrue(old.domain_rand.push_robots)
        self.assertEqual(new.rewards.scales.foot_place, 0.)
        self.assertEqual(old.rewards.scales.foot_place, 1.)
        train = config["X1DHStandV12Exp31CfgPPO"]()
        self.assertEqual(train.algorithm.amp_style_reward_scale, 5.)
        self.assertEqual(train.runner.max_iterations, 6000)
        self.assertFalse(train.runner.resume)
        evaluation.configure_evaluation(new, "nominal_no_push", 123145)
        self.assertFalse(new.noise.add_noise)
        self.assertEqual(new.domain_rand.friction_range, [.75, .75])
        self.assertEqual(new.domain_rand.lag_timesteps_range, [22, 22])
        self.assertEqual(new.domain_rand.dof_lag_timesteps_range, [20, 20])
        self.assertEqual(new.domain_rand.joint_1_armature_range, [.16, .16])
        self.assertEqual(old.domain_rand.friction_range, [.2, 1.3])

    def test_reference_start_stop_continuity_and_idempotence(self):
        env = fixture()
        initial = env.default_dof_pos.clone()
        env.commands[:, 0] = .6
        env.compute_ref_state()
        torch.testing.assert_close(env.ref_dof_pos, initial)
        for tick in range(1, 16):
            env.common_step_counter = tick
            env.compute_ref_state()
        halfway = env.ref_dof_pos.clone()
        env.compute_ref_state()
        torch.testing.assert_close(env.ref_dof_pos, halfway)
        frame = int((.25 * env.seg_len[1] + env.seg_anchor[1]).long() % env.seg_len[1])
        torch.testing.assert_close(halfway, (initial + env.mocap_q[frame, 1]) / 2, atol=1e-6, rtol=1e-5)
        env.commands.zero_()
        env.compute_ref_state()
        torch.testing.assert_close(env.ref_dof_pos, halfway)
        for tick in range(16, 47):
            env.common_step_counter = tick
            env.compute_ref_state()
        torch.testing.assert_close(env.ref_dof_pos, initial)

    def test_velocity_direction_and_slip_fields(self):
        env = fixture()
        env.commands[:, 0] = .6
        values = []
        for speed in [-.2, 0., .2, .6]:
            env.base_lin_vel[:, 0] = speed
            values.append(float(env._reward_signed_speed_error()))
        self.assertEqual(values, sorted(values))
        env.contact_forces[:, :, 2] = 100
        env.rigid_state[:, :, 10:12] = 100
        self.assertEqual(float(env._reward_foot_slip()), 0.)
        env.rigid_state[:, 0, 7] = 1
        self.assertEqual(float(env._reward_foot_slip()), 1.)

    def test_command_injection_preserves_history(self):
        env = fixture()
        env.obs_history[0].fill_(7.)
        env.obs_buf[:, :98] = 7.
        env.set_evaluation_command(.6)
        self.assertEqual(len(env.obs_history), 2)
        torch.testing.assert_close(env.obs_buf[:, :98], torch.full((1, 98), 7.))
        self.assertAlmostEqual(float(env.obs_buf[0, -96]), 1.2, places=6)
        self.assertAlmostEqual(float(env.privileged_obs_buf[0, -139]), 1.2, places=6)

    def test_reward_command_is_sampled_after_task_reward(self):
        env = fixture()
        env.commands[:, 0] = .6
        env.base_lin_vel[:, 0] = .2
        env._command_for_action = torch.tensor([.6])
        env._observed_command_for_action = torch.tensor([.6])
        env._termination = {"done": torch.tensor([False])}
        env._episode_id = torch.tensor([0])
        env._speed_ema = torch.tensor([.2])
        env._torque_clip_step = torch.tensor([0])
        env.extras = {}
        env.rew_buf = torch.zeros(1)
        env.reward_names = ["signed_speed_error"]
        env.reward_functions = [env._reward_signed_speed_error]
        env.reward_scales = {"signed_speed_error": .04}
        env.episode_sums = {"signed_speed_error": torch.zeros(1)}
        env._resample_commands = lambda: env.commands[:, 0].fill_(-.2)
        env.compute_reward()
        self.assertAlmostEqual(float(env.rew_buf), -.016, places=6)
        self.assertAlmostEqual(float(env.commands[0, 0]), -.2, places=6)
        self.assertAlmostEqual(float(env.extras["diagnostics"]["command_applied"]), .6, places=6)

    def test_joint_properties_do_not_compound_on_reset(self):
        env = fixture()
        env.num_dof = 29
        props = np.ones(29, dtype=[("friction", "f4"), ("damping", "f4"), ("armature", "f4")])
        class Gym:
            def get_actor_dof_properties(self, *_):
                return props.copy()

            def set_actor_dof_properties(self, _env, _actor, value):
                props[:] = value
        env.gym = Gym()
        env.envs = [object()]
        env._joint_property_baseline = {name: np.ones(29) for name in props.dtype.names}
        env.joint_friction_coeffs = torch.full((1, 29), .5)
        env.joint_damping_coeffs = torch.full((1, 29), .9)
        env.joint_armatures = torch.full((1, 29), .16)
        env._refresh_actor_dof_props([0])
        first = props.copy()
        env._refresh_actor_dof_props([0])
        np.testing.assert_array_equal(first, props)
        self.assertAlmostEqual(float(props["damping"][0]), .9, places=6)

    def test_terminal_capture_uses_pre_reset_state(self):
        env = fixture()
        env.commands.zero_()  # Reset already happened in the simulator.
        saved = dict(base_pos=torch.tensor([[.8, .2, .43]]), base_lin_vel=torch.tensor([[1.6, .2, -.5]]),
                     command=torch.tensor([[.6, 0., 0., 0.]]), command_applied=torch.tensor([.6]),
                     command_observed=torch.tensor([.6]), episode_id=torch.tensor([4]),
                     timeout=torch.tensor([False]), torque_clip_step=torch.tensor([3]))
        row = diagnostics.capture_playback_step(env, {"diagnostics": saved}, torch.tensor([-.1]),
                    torch.tensor([True]), SimpleNamespace(amp_discriminator=None), .6, 2, 7)
        self.assertAlmostEqual(row["base_height"], .43, places=6)
        self.assertAlmostEqual(row["base_vel_x"], 1.6, places=6)
        self.assertEqual(row["episode_id"], 4)
        self.assertEqual(row["clip_count"], 10)
        self.assertEqual(row["done"], 1)
        self.assertAlmostEqual(row["command_applied"], .6, places=6)


if __name__ == "__main__":
    unittest.main()
