#!/usr/bin/env python
"""Short Isaac Gym smoke check for the v1.2 sagittal X1 task.

Run on a machine with Isaac Gym:
    python scripts/tools/smoke_gmr_v1_2_isaac.py --task=x1_dh_stand_v12 --num_envs=16 --headless
"""

from isaacgym import gymapi  # Isaac Gym must be imported before PyTorch.

import torch

import humanoid.envs  # noqa: F401 - registers the task before importing task_registry
from humanoid.utils import get_args, task_registry


def main():
    args = get_args()
    assert args.task == "x1_dh_stand_v12", "this smoke check needs the v1.2 task"
    env, _ = task_registry.make_env(name=args.task, args=args)
    assert set(env.seg_names) == {"walk_forward", "walk_backward"}
    assert env._amp_num_seg == 2
    demo = env._sample_amp_demo()
    assert demo.shape == (env.num_envs, 3, 61)
    assert torch.isfinite(demo).all()

    for mode, speed, segment in (
        ("stand", 0.0, None),
        ("forward", 0.6, "walk_forward"),
        ("backward", -0.35, "walk_backward"),
    ):
        env.commands.zero_()
        env.commands[:, 0] = speed
        env.phase_length_buf.fill_(100)
        env._gait_phase.fill_(0.25)
        env.compute_ref_state()

        if segment is None:
            expected = env.default_dof_pos.expand_as(env.ref_dof_pos)
        else:
            index = env.seg_names.index(segment)
            assert torch.all(env._current_seg_id() == index)
            phase = env._get_phase()
            frames = torch.remainder(
                (phase * env.seg_period_frames[index].float()
                 + env.seg_anchor[index].float()).long(), env.seg_len[index])
            expected = env.mocap_q[frames, index]
        error = (env.ref_dof_pos - expected).abs().max().item()
        assert error < 1e-5, "{} reference error: {}".format(mode, error)
        assert torch.isfinite(env._reward_foot_place()).all()

        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        observations, _, rewards, _, extras = env.step(actions)
        assert torch.isfinite(observations).all()
        assert torch.isfinite(rewards).all()
        amp = extras["amp"]
        assert amp["disc_obs"].shape == (env.num_envs, 3, 61)
        assert torch.isfinite(amp["disc_obs"]).all()
        assert torch.isfinite(amp["disc_demo_obs"]).all()
        print("SMOKE_{} OK: ref_error={:.2e}, reward_mean={:.3f}".format(
            mode.upper(), error, rewards.mean().item()), flush=True)

    print("SMOKE_V12_ALL_OK", flush=True)


if __name__ == "__main__":
    main()
