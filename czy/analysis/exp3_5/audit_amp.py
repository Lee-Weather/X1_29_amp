import csv,json,numpy as np,torch
from pathlib import Path
R=Path.cwd();out=R/'czy/analysis/exp3_5';lib=torch.load(R/'resources/motions/processed/amp_v1_2_exp34.pt',map_location='cpu',weights_only=False)['walk_forward'];phase=torch.load(R/'resources/motions/processed/ref_lib_v1_2_exp34_phase.pt',map_location='cpu',weights_only=False)['walk_forward']
with (R/'czy/analysis/exp3_4/preflight_20261021_144449/b3_initial_isaac_diag.csv').open(newline='',encoding='utf-8-sig') as f:rows=[r for r in csv.DictReader(f) if float(r['command_x'])>.1]
names=[k[4:] for k in rows[0] if k.startswith('pos_') and not k.startswith('pos_des_')];assert names==lib['dof_names']==phase['dof_names']
q=lib['dof_pos'].numpy();fps=float(lib['fps']);dq=np.concatenate([np.diff(q,axis=0)*fps,np.diff(q,axis=0)[-1:]*fps])
metrics=[]
for i,n in enumerate(names):
 aq=np.array([float(r['pos_'+n]) for r in rows]);av=np.array([float(r['vel_'+n]) for r in rows]);den=max(float(q[:,i].std()),.05)
 metrics.append({'joint':n,'agent_q_mean':float(aq.mean()),'demo_q_mean':float(q[:,i].mean()),'position_offset_in_demo_std':float(abs(aq.mean()-q[:,i].mean())/den),'agent_dq_std':float(av.std()),'demo_dq_std':float(dq[:,i].std())})
result={'names_order_match':True,'fps':fps,'control_dt':.01,'agent_stride':2,'demo_stride':1,'window_samples':3,'agent_window_seconds':.04,'demo_window_seconds':.04,'demo_finite':bool(np.isfinite(q).all() and np.isfinite(dq).all()),'notes':['joint names/time span checked; names alone do not prove physical axis correctness','demo finite differences versus simulator instantaneous velocities differ in filtering','prior code audit uses root angular velocity in body frame on both sides; no changes to AMP hyperparameters in exp3.5'], 'joint_distributions':metrics}
h=phase['sole_height'].numpy();clock=np.arange(len(h))/len(h);sin=np.sin(2*np.pi*clock);idx=(np.arange(len(h))+int(phase['phase_anchor_frame']))%len(h)
result['phase_contact_geometry']={}
for foot,name in enumerate(phase['exp33_geometry']['foot_names']):
 support=((sin>=0) if foot==0 else (sin<0)) | (abs(sin)<.1)
 result['phase_contact_geometry'][name]={'support_sole_height_mean':float(h[idx,foot][support].mean()),'swing_sole_height_mean':float(h[idx,foot][~support].mean())}
(out/'amp_input_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('AMP_AUDIT',json.dumps({k:v for k,v in result.items() if k!='joint_distributions'}));print('LARGEST_Q_OFFSETS',json.dumps(sorted(metrics,key=lambda x:x['position_offset_in_demo_std'],reverse=True)[:5]))
