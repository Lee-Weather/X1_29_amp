"""CPU interface check for the new 127-feature actor and full DH JIT export."""
import isaacgym  # Isaac Gym must be imported before torch on the F1 image.
import json
from pathlib import Path
import torch

from humanoid.envs import task_registry
from humanoid.algo.ppo import ActorCriticDH
from humanoid.scripts.export_policy_dh import ExportedDH
from humanoid.utils.helpers import class_to_dict
from humanoid.utils.residual_control import export_residual_control_spec


def main():
    cfg, train = task_registry.get_cfgs("x1_dh_stand_v12_exp33")
    model = ActorCriticDH(cfg.env.short_frame_stack * cfg.env.num_single_obs,
                          cfg.env.num_single_obs, cfg.env.num_privileged_obs, cfg.env.num_actions,
                          **class_to_dict(train.policy)).eval()
    observations = torch.randn(2, cfg.env.num_observations)
    with torch.no_grad():
        actions = model.act_inference(observations)
        value = model.evaluate(torch.zeros(2, cfg.env.num_privileged_obs))
        exported = ExportedDH(model.actor, model.long_history, model.state_estimator,
                              model.num_short_obs, model.in_channels, model.num_proprio_obs).eval()
        scripted = torch.jit.script(exported)
        jit_actions = scripted(observations)
    assert actions.shape == (2, 29) and value.shape == (2, 1)
    assert actions.abs().max().item() == 0
    assert torch.isfinite(value).all()
    assert torch.allclose(actions, jit_actions)
    out = Path("interface_validation")
    export_residual_control_spec(cfg, out)
    spec = json.loads((out / "residual_control.json").read_text())
    assert len(spec["residual_limit_rad"]) == 29 and spec["observation"]["single_frame"] == 127
    result = dict(success=True, single_obs=cfg.env.num_single_obs, total_obs=cfg.env.num_observations,
                  actions_shape=list(actions.shape), critic_shape=list(value.shape),
                  initial_mean_abs_max=float(actions.abs().max()),
                  jit_abs_error_max=float((jit_actions - actions).abs().max()),
                  trained_checkpoint=False, torch_version=torch.__version__)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("INTERFACE_VALIDATION", json.dumps(result))


if __name__ == "__main__":
    main()
