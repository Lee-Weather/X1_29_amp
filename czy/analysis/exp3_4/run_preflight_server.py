"""Run source playback, initial B3 playback and a short exp3.4 training check."""
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    manifest=json.loads((ROOT/'preflight_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('bundle checksum mismatch: '+name)
    env=os.environ.copy()
    env.update(PYTHONPATH=str(ROOT)+':'+env.get('PYTHONPATH',''),PLAY_RENDER='0',
        SDL_VIDEODRIVER='dummy',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        LD_LIBRARY_PATH='/home/robot/Anaconda/envs/F1/lib:'+env.get('LD_LIBRARY_PATH',''))
    result={'finished':False,'cases':{},'started_at':datetime.now().isoformat()}
    destination=ROOT/'preflight_summary.json'
    def save():destination.write_text(json.dumps(result,indent=2)+'\n')
    for label,task in [('source','x1_dh_stand_exp34_source'),('b3_initial','x1_dh_stand_v12_exp34')]:
        command=[sys.executable,'-u','humanoid/scripts/play.py','--task='+task,'--headless',
            '--num_envs=1','--seed=123145','--eval_mode=nominal_no_push',
            '--ckpt_path='+str(ROOT/'model_26998.pt'),'--run_name=exp34_'+label]
        log=ROOT/(label+'.log')
        print('CASE_START',label,flush=True)
        with log.open('w') as stream:r=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        case={'returncode':r.returncode,'log':str(log)}
        if not r.returncode:
            csv_path=max((ROOT/'logs').rglob('*_exp34_'+label+'_isaac_diag.csv'),key=lambda p:p.stat().st_mtime)
            metadata=json.loads(csv_path.with_name(csv_path.name.replace('_isaac_diag.csv','_metadata.json')).read_text())
            with csv_path.open() as stream:rows=list(csv.DictReader(stream))
            stage_means=[]
            for stage in range(6):
                points=rows[500*stage:500*(stage+1)]
                stage_means.append(sum(float(p['base_vel_x']) for p in points)/len(points))
            case.update(metadata=metadata,csv=str(csv_path),stage_vx=stage_means,
                source_ready=metadata['resets']==0 and stage_means[1]>.1 and stage_means[2]>.1)
        result['cases'][label]=case;save()
        print('CASE_END',label,json.dumps(case),flush=True)
        if r.returncode:raise RuntimeError(label+' playback failed; see its log')
    print('CASE_START','smoke200',flush=True)
    command=[sys.executable,'-u','humanoid/scripts/train.py','--task=x1_dh_stand_v12_exp34',
        '--headless','--num_envs=64','--seed=5','--max_iterations=200',
        '--ckpt_path='+str(ROOT/'model_26998.pt'),'--disc_fresh','--run_name=exp34_smoke200']
    log=ROOT/'smoke200.log'
    with log.open('w') as stream:r=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    text=log.read_text(errors='replace')
    errors=[line for line in text.splitlines() if any(k in line for k in ['Traceback','RuntimeError','ValueError','KeyError'])]
    checkpoints=list((ROOT/'logs').rglob('model_27198.pt'))
    case=dict(returncode=r.returncode,errors=errors,final_checkpoint=str(checkpoints[0]) if checkpoints else None,
        last_iterations=re.findall(r'Learning iteration (\d+)/(\d+)',text)[-3:])
    result['cases']['smoke200']=case
    result.update(finished=True,finished_at=datetime.now().isoformat(),
        ready_for_training=result['cases']['source'].get('source_ready',False) and r.returncode==0 and not errors and bool(checkpoints))
    save();print('PREFLIGHT_FINISHED',json.dumps(result),flush=True)


if __name__=='__main__':main()
