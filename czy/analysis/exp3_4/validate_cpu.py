"""CPU checks of transfer compatibility and real-second scheduling, without Isaac Gym."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[3]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    sys.path.insert(0, str(ROOT))
    # Import config classes without executing humanoid.envs' simulator registrations.
    for name, relative in [('humanoid.envs','humanoid/envs'),('humanoid.envs.base','humanoid/envs/base'),
                           ('humanoid.envs.x1','humanoid/envs/x1')]:
        package = ModuleType(name); package.__path__ = [str(ROOT/relative)]; sys.modules[name] = package
    cfg_mod=module(ROOT/'humanoid/envs/x1/x1_dh_stand_v12_exp34_config.py',
                   'humanoid.envs.x1.x1_dh_stand_v12_exp34_config')
    cfg, train = cfg_mod.X1DHStandV12Exp34Cfg(), cfg_mod.X1DHStandV12Exp34CfgPPO()
    assert cfg.env.num_observations == 6468 and cfg.env.num_privileged_obs == 423
    assert not cfg.env.use_ref_actions and cfg.amp.demo_seg_weights == [1.0]
    assert train.algorithm.amp_style_reward_scale == 5 and train.algorithm.amp_task_lerp == .5
    assert cfg.rewards.scales.ref_joint_pos == .5 and cfg.rewards.scales.foot_place == 1
    assert cfg.rewards.scales.low_speed == 2 and cfg.rewards.scales.stand_still == 3.5
    assert cfg.rewards.scales.yaw_drift_angle == 0
    model_module=module(ROOT/'humanoid/algo/ppo/actor_critic_dh.py','exp34_actor_cpu')
    kwargs={name:getattr(train.policy,name) for name in dir(train.policy) if not name.startswith('_')}
    with contextlib.redirect_stdout(io.StringIO()):
        actor=model_module.ActorCriticDH(490,98,423,29,**kwargs)
    checkpoint=torch.load(ROOT/'czy/data/exp2.4a_cont2/model_26998.pt',map_location='cpu',weights_only=False)
    actor.load_state_dict(checkpoint['model_state_dict'],strict=True)
    with torch.no_grad():
        action=actor.act_inference(torch.randn(2,6468))
        value=actor.evaluate(torch.zeros(2,423))
    assert action.shape==(2,29) and torch.isfinite(action).all() and torch.isfinite(value).all()

    # Execute the real scheduler methods on CPU tensors; no simulator methods are replaced by results.
    tree=ast.parse((ROOT/'humanoid/envs/x1/x1_dh_stand_v12_exp34_env.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='X1DHStandV12Exp34Env')
    node.bases=[ast.Name(id='Parent',ctx=ast.Load())]
    class Parent:
        def _resample_stand_command(self,ids): self.commands[ids]=0
    namespace={'torch':torch,'Parent':Parent}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'exp34_cpu_scheduler','exec'),namespace)
    env=namespace[node.name].__new__(namespace[node.name])
    env.cfg=cfg;env.dt=.01;env.device='cpu';env.num_envs=16;env._defer_command_sampling=False
    env.commands=torch.zeros(16,4);env.episode_length_buf=torch.zeros(16,dtype=torch.long)
    env.gait_time=torch.zeros(16,3,dtype=torch.int);env._schedule_end=torch.zeros(16,dtype=torch.long)
    env._yaw_ref=torch.zeros(16);env.base_euler_xyz=torch.zeros(16,3)
    torch.manual_seed(5);env.generate_gait_time(torch.arange(16))
    initial=env.gait_time[:,1].clone()
    assert ((initial>=200)&(initial<=300)).all()
    transitions=[[] for _ in range(16)]; previous=torch.zeros(16,dtype=torch.bool)
    for step in range(4001):
        env.episode_length_buf.fill_(step);env._resample_commands()
        walking=env.commands[:,0]>.05
        for i in (walking!=previous).nonzero().flatten().tolist():transitions[i].append(step)
        assert (env.commands[:,0]>=0).all()
        if walking.any(): assert ((env.commands[walking,0]>=.35)&(env.commands[walking,0]<=.65)).all()
        previous=walking.clone()
    for times in transitions:
        assert len(times)>=6
        for i,gap in enumerate(b-a for a,b in zip(times,times[1:])):
            lower,upper=(600,900) if i%2==0 else (200,300)
            assert lower<=gap<=upper,(times,gap)
    cfg.commands.external_commands=True
    env.commands.fill_(.123);env._resample_commands();assert (env.commands==.123).all()

    preflight=module(ROOT/'humanoid/utils/transfer_preflight.py','exp34_preflight_cpu')
    args=SimpleNamespace(ckpt_path=str(ROOT/'czy/data/exp2.4a_cont2/model_26998.pt'),disc_fresh=False,max_iterations=None)
    preflight.prepare_initial_transfer(args,train,ROOT);assert args.disc_fresh
    with tempfile.TemporaryDirectory() as folder:
        bad=Path(folder)/'wrong.pt';bad.write_bytes(b'wrong checkpoint')
        args.ckpt_path=str(bad)
        try: preflight.prepare_initial_transfer(args,train,ROOT)
        except ValueError: pass
        else: raise AssertionError('wrong source was accepted')
    phase=torch.load(ROOT/'resources/motions/processed/ref_lib_v1_2_exp34_phase.pt',weights_only=False)
    demo=torch.load(ROOT/'resources/motions/processed/amp_v1_2_exp34.pt',weights_only=False)
    assert set(phase)==set(demo)=={'walk_forward'}
    assert len(phase['walk_forward']['dof_pos'])==63 and len(demo['walk_forward']['dof_pos'])==351
    record=phase['walk_forward']
    env.commands.zero_();env.commands[:,0]=.4
    env.seg_len=torch.tensor([63]);env.seg_period_frames=torch.tensor([63]);env.seg_anchor=torch.tensor([44])
    env.mocap_q=record['dof_pos'].unsqueeze(1);env.mocap_rp=torch.zeros(63,1,2)
    env.default_dof_pos=record['exp33_geometry']['stand_dof_pos'].unsqueeze(0)
    env._ref_q_target=env.default_dof_pos.repeat(16,1);env._ref_from_q=env._ref_q_target.clone()
    env._ref_rp_target=torch.zeros(16,2);env._ref_from_rp=env._ref_rp_target.clone()
    env._ref_mode=torch.zeros(16,dtype=torch.long);env._ref_elapsed=torch.full((16,),.3)
    env._ref_update_step=0;env.common_step_counter=0
    env._reference_soles=record['sole_height'];env._ref_sole_target=torch.zeros(16,2)
    env._get_phase=lambda:torch.full((16,),.25)
    env.compute_ref_state()
    expected=.25*record['dof_pos'][59]+.75*record['dof_pos'][60]
    assert torch.allclose(env.ref_dof_pos,expected.expand(16,-1),atol=1e-6)
    assert torch.isfinite(env._ref_sole_target).all()
    env.commands.zero_();env.compute_ref_state();env._ref_elapsed.fill_(.3);env.compute_ref_state()
    assert torch.allclose(env.ref_dof_pos,env.default_dof_pos.expand(16,-1))
    assert (env._ref_sole_target==0).all()
    result=dict(success=True,kind='CPU only, no Isaac Gym simulation',checkpoint_internal_iter=checkpoint['iter'],
        actor_shape=list(action.shape),critic_shape=list(value.shape),single_observation=98,history=6468,
        initial_stand_steps=[int(initial.min()),int(initial.max())],scheduler_environments=16,
        scheduler_steps=4001,transitions_first_env=transitions[0],fresh_discriminator_enforced=True,
        wrong_checkpoint_rejected=True,reference_interpolation_and_standing_checked=True,phase_frames=63,demo_frames=351)
    (Path(__file__).parent/'cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
