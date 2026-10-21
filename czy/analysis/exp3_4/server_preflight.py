"""Transfer a minimal working tree bundle to the authorized Isaac Gym server."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BIN=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/OpenSSH'
SSH=[str(BIN/'ssh.exe'),'-o','BatchMode=yes','-o','ConnectTimeout=10','robot@10.12.201.5']
SCP=[str(BIN/'scp.exe'),'-o','BatchMode=yes','-o','ConnectTimeout=10']
POINTER=HERE/'preflight_latest.json'


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','status','collect']);args=p.parse_args()
    if args.action=='prepare':
        run=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        POINTER.write_text(json.dumps({'run':run})+'\n')
    run=json.loads(POINTER.read_text())['run'];out=HERE/('preflight_'+run);out.mkdir(exist_ok=True)
    stage=ROOT/'logs'/('exp34_preflight_'+run);stage.mkdir(parents=True,exist_ok=True)
    remote='/home/robot/czy/exp1/exp_'+run+'/x1_exp34_preflight'
    if args.action=='prepare':
        files={p.relative_to(ROOT).as_posix():p for p in (ROOT/'humanoid').rglob('*.py')}
        urdf='resources/robots/x1/urdf/X1_29DOF_physically_mirrored.urdf'
        for name in ['setup.py',urdf,'resources/motions/processed/ref_lib.pt',
            'resources/motions/processed/ref_lib_v1_2_exp34_phase.pt',
            'resources/motions/processed/amp_v1_2_exp34.pt','resources/motions/processed/v1_2_exp34_manifest.json']:
            files[name]=ROOT/name
        for mesh in ET.parse(ROOT/urdf).iter('mesh'):
            path=((ROOT/urdf).parent/mesh.attrib['filename']).resolve();files[path.relative_to(ROOT).as_posix()]=path
        files['model_26998.pt']=ROOT/'czy/data/exp2.4a_cont2/model_26998.pt'
        hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
        manifest=dict(run=run,remote=remote,files=hashes,source='working tree for exp3.4 server preflight')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        with tarfile.open(stage/'bundle.tar.gz','w:gz') as tar:
            for name,path in files.items():tar.add(path,arcname=name)
            tar.add(out/'manifest.json',arcname='preflight_manifest.json')
            tar.add(HERE/'run_preflight_server.py',arcname='run_replays_server.py')
            tar.add(ROOT/'czy/analysis/exp3_1/replay_20261020_164054/launch_server.py',arcname='launch_server.py')
        print('PREPARED',run,'files',len(files),'bytes',(stage/'bundle.tar.gz').stat().st_size);return
    def call(command):subprocess.run(command,check=True)
    if args.action=='launch':
        call([*SSH,'mkdir -p '+shlex.quote(remote)])
        call([*SCP,str(stage/'bundle.tar.gz'),'robot@10.12.201.5:'+remote+'/'])
        call([*SSH,'source /home/robot/Anaconda/etc/profile.d/conda.sh && conda activate F1 && cd '+shlex.quote(remote)+
              ' && tar -xzf bundle.tar.gz && python launch_server.py']);return
    if args.action=='status':
        command=('cd '+shlex.quote(remote)+" && tail -n 10 replay_batch.log && for p in source.log b3_initial.log smoke200.log; do test ! -f $p || tail -n 6 $p; done")
        call([*SSH,command]);return
    call([*SCP,'robot@10.12.201.5:'+remote+'/preflight_summary.json',str(out/'summary.json')])
    summary=json.loads((out/'summary.json').read_text())
    for label,case in summary['cases'].items():
        call([*SCP,'robot@10.12.201.5:'+remote+'/'+label+'.log',str(out/(label+'.log'))])
        if 'csv' in case:call([*SCP,'robot@10.12.201.5:'+case['csv'],str(out/(label+'_isaac_diag.csv'))])
    print('COLLECTED',json.dumps(summary));


if __name__=='__main__':main()
