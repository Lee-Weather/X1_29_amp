"""Matched 60-second legacy-control replay, after the main smoke process."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
os.chdir(root)
while True:
    progress = root / 'preflight_summary.json'
    if progress.exists() and json.loads(progress.read_text()).get('finished'):
        break
    time.sleep(3)
env = os.environ.copy()
env.update(PYTHONPATH=str(root) + ':' + env.get('PYTHONPATH', ''), PLAY_RENDER='0',
           SDL_VIDEODRIVER='dummy', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
           LD_LIBRARY_PATH='/home/robot/Anaconda/envs/F1/lib:' + env.get('LD_LIBRARY_PATH', ''))
command = [sys.executable, '-u', 'humanoid/scripts/play.py', '--task=x1_dh_stand_v12_exp34',
           '--headless', '--num_envs=1', '--seed=123145', '--eval_mode=nominal_no_push',
           '--ckpt_path=' + str(root / 'model_26998.pt'), '--run_name=exp35_unbounded_control']
print('CONTROL_START', flush=True)
with (root / 'control.log').open('w') as stream:
    result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
summary = {'finished': True, 'returncode': result.returncode}
if result.returncode == 0:
    path = max((root / 'logs').rglob('*_exp35_unbounded_control_isaac_diag.csv'), key=lambda p: p.stat().st_mtime)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    meta = json.loads(path.with_name(path.name.replace('_isaac_diag.csv', '_metadata.json')).read_text())
    summary.update(csv=str(path), metadata=meta,
                   yaw_drift_end=float(rows[-1]['yaw_from_episode_start']),
                   stage_vx=[sum(float(r['base_vel_x']) for r in rows[i*1000:(i+1)*1000])/1000 for i in range(6)])
(root / 'control_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('CONTROL_END', json.dumps(summary), flush=True)
