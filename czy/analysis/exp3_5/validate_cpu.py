import ast,json,types,torch
from pathlib import Path
root=Path.cwd();ns={'torch':torch}
# Execute actual reward implementations without importing Isaac Gym.
def cls(path,name,bases,scope):
 tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'));c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
 c.bases=[ast.Name(id=b,ctx=ast.Load()) for b in bases];m=ast.fix_missing_locations(ast.Module(body=[c],type_ignores=[]));exec(compile(m,path,'exec'),scope);return scope[name]
class Parent:
 def _pd_position_target(self):return self.raw
ns.update(X1DHStandV12Exp34Env=Parent,json=json,LEGGED_GYM_ROOT_DIR=str(root))
# Only methods needed in CPU test are compiled from ancestors.
for file,name in [('humanoid/envs/x1/x1_dh_stand_env.py','X1DHStandEnv'),('humanoid/envs/x1/x1_dh_stand_v12_exp31_env.py','X1DHStandV12Exp31Env')]:
 tree=ast.parse(Path(file).read_text(encoding="utf-8-sig"));c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
 methods=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_reward_')];c.body=methods;c.bases=[];exec(compile(ast.fix_missing_locations(ast.Module(body=[c],type_ignores=[])),file,'exec'),ns)
C=cls('humanoid/envs/x1/x1_dh_stand_v12_exp35_env.py','X1DHStandV12Exp35Env',['X1DHStandV12Exp34Env'],ns);e=C();e.cfg=types.SimpleNamespace(control=types.SimpleNamespace(bounds_warmup_steps=24000,bounds_ramp_steps=24000,evaluation_bounds_strength=1))
e.raw=torch.tensor([[2.,-.8,.7],[0.,.2,-.3]]);e._bound_ids=torch.tensor([0,1]);e._bound_lower=torch.tensor([-.4,-.4]);e._bound_upper=torch.tensor([.4,.4]);e._target_raw=torch.zeros_like(e.raw);e._target_effective=e._target_raw.clone();e._bound_clip_counts=torch.zeros_like(e.raw,dtype=torch.long);e._bounds_evaluation=False
for step,strength in [(0,0),(24000,0),(36000,.5),(48000,1)]:
 e.common_step_counter=step;t=e._pd_position_target();assert e._bounds_strength()==strength;assert torch.equal(t[:,2],e.raw[:,2]);assert torch.allclose(t[:,:2],torch.lerp(e.raw[:,:2],e.raw[:,:2].clamp(-.4,.4),strength))
e._bounds_evaluation=True;e.common_step_counter=0;assert e._bounds_strength()==1
# Restored gait reward must remain active for a walking command.
e.dof_pos=torch.zeros(2,12);e.default_joint_pd_target=torch.zeros_like(e.dof_pos);e.leg_dof_indices=torch.arange(12);e.commands=torch.tensor([[.4,0,0],[.6,0,0]])
assert torch.allclose(e._reward_default_joint_pos(),torch.ones(2));e.dof_pos[:,1]=.5;assert (e._reward_default_joint_pos()<0).all()
# Group Huber cost continues increasing outside the old exponential's saturation zone.
e.ref_dof_pos=torch.zeros_like(e.dof_pos);ids=torch.tensor([0,1]);e.dof_pos.fill_(1);c1=e._pose_cost(ids,.25);e.dof_pos.fill_(2);c2=e._pose_cost(ids,.25);assert (c2>c1).all()
result={'passed':True,'checks':['warmup identity','half/full target bounds','leg targets unchanged','evaluation full-strength','walking default reward active','Huber far-error cost increasing'],'scope':'CPU tensor checks on real methods; no Isaac Gym physics'}
Path('czy/analysis/exp3_5/cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
