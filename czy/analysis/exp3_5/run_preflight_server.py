"""Full-strength bounds replay followed by compressed-schedule smoke training."""
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    manifest = json.loads((ROOT / 'preflight_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('bundle checksum mismatch: ' + name)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT) + ':' + env.get('PYTHONPATH', ''), PLAY_RENDER='0',
               SDL_VIDEODRIVER='dummy', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               LD_LIBRARY_PATH='/home/robot/Anaconda/envs/F1/lib:' + env.get('LD_LIBRARY_PATH', ''))
    result = {'finished': False, 'cases': {}, 'started_at': datetime.now().isoformat()}
    destination = ROOT / 'preflight_summary.json'
    def save():
        destination.write_text(json.dumps(result, indent=2) + '\n')
    command = [sys.executable, '-u', 'humanoid/scripts/play.py', '--task=x1_dh_stand_v12_exp35',
               '--headless', '--num_envs=1', '--seed=123145', '--eval_mode=nominal_no_push',
               '--ckpt_path=' + str(ROOT / 'model_26998.pt'), '--run_name=exp35_bounded']
    print('CASE_START bounded: full-strength upper-body bounds, 60s, no video', flush=True)
    with (ROOT / 'bounded.log').open('w') as stream:
        replay = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    case = {'returncode': replay.returncode}
    if replay.returncode == 0:
        csv_path = max((ROOT / 'logs').rglob('*_exp35_bounded_isaac_diag.csv'), key=lambda p: p.stat().st_mtime)
        metadata = json.loads(csv_path.with_name(csv_path.name.replace('_isaac_diag.csv', '_metadata.json')).read_text())
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        speeds = [sum(float(r['base_vel_x']) for r in rows[i*1000:(i+1)*1000])/1000 for i in range(6)]
        strength = [float(r['exp35_bounds_strength']) for r in rows]
        case.update(csv=str(csv_path), metadata=metadata, stage_vx=speeds,
                    strength_min=min(strength), strength_max=max(strength),
                    full_bounds_stable=metadata['resets'] == 0 and speeds[1] > .2 and speeds[2] > .35)
    result['cases']['bounded'] = case
    save()
    print('CASE_END bounded', json.dumps(case), flush=True)
    if replay.returncode or not case.get('full_bounds_stable'):
        result.update(finished=True, ready_for_training=False, reason='full bounds need calibration')
        save()
        return
    # Exercise warmup, ramp and full-strength execution within 200 updates.
    # These overrides apply only to this smoke process; formal training is 24000/24000.
    entry = ROOT / 'smoke_entry.py'
    entry.write_text("from humanoid.envs import task_registry\n"
                     "from humanoid.scripts.train import train\n"
                     "from humanoid.utils import get_args\n"
                     "args=get_args()\n"
                     "cfg,_=task_registry.get_cfgs(name=args.task)\n"
                     "cfg.control.bounds_warmup_steps=480\n"
                     "cfg.control.bounds_ramp_steps=960\n"
                     "print('SMOKE_ONLY bounds schedule: 480 warmup + 960 ramp', flush=True)\n"
                     "train(args)\n")
    command = [sys.executable, '-u', str(entry), '--task=x1_dh_stand_v12_exp35', '--headless',
               '--num_envs=64', '--seed=5', '--max_iterations=200', '--ckpt_path=' + str(ROOT/'model_26998.pt'),
               '--disc_fresh', '--run_name=exp35_smoke200']
    print('CASE_START smoke200', flush=True)
    with (ROOT / 'smoke200.log').open('w') as stream:
        smoke = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    text = (ROOT / 'smoke200.log').read_text(errors='replace')
    errors = [line for line in text.splitlines() if any(k in line for k in ('Traceback', 'RuntimeError', 'ValueError', 'KeyError'))]
    models = list((ROOT/'logs').rglob('model_27198.pt'))
    result['cases']['smoke200'] = dict(returncode=smoke.returncode, errors=errors,
                                     final_checkpoint=str(models[0]) if models else None)
    result.update(finished=True, ready_for_training=smoke.returncode == 0 and not errors and bool(models),
                  finished_at=datetime.now().isoformat())
    save()
    print('PREFLIGHT_FINISHED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
