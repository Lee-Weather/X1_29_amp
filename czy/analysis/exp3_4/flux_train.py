"""Account-verified, idempotent launch of the authorized exp3.4 transfer."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'launch.json'
BRANCH='codex/exp3-4'
TASK='x1_dh_stand_v12_exp34'
EMAIL='limxmv4xnjtlno3gsm@maxxspace.com'
TZ=timezone(timedelta(hours=8))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','create','clear-storage','run','status','note'])
    args=parser.parse_args();os.chdir(ROOT)
    pool=json.loads((ROOT/'czy/skills/flux-cli/api_key.json').read_text(encoding='utf-8-sig'))['accounts']
    index=next(i for i,a in enumerate(pool) if a['email']==EMAIL);account=pool[index]
    environment=os.environ.copy()
    for key in list(environment):
        if key.lower() in ('http_proxy','https_proxy','all_proxy'):environment.pop(key)
    cli=shutil.which('flux.cmd') or shutil.which('flux')
    def redact(value):
        text=str(value)
        for entry in pool:text=text.replace(entry['api_key'],'[REDACTED]')
        return re.sub(r'https?://[^\s\"<>]+','[URL]',text)
    def call(*command,dry=False,raw=False):
        r=subprocess.run([cli,'--yes',*(['--dry-run'] if dry else []),*command],env=environment,
            capture_output=True,text=True,encoding='utf-8',errors='replace')
        if raw:
            if r.returncode:raise RuntimeError(redact(r.stderr))
            return redact(r.stdout)
        try:response=json.loads(r.stdout)
        except ValueError:raise RuntimeError(redact(r.stderr or r.stdout)) from None
        if dry and r.returncode==10:
            print('DRY_RUN_OK',command[:2],flush=True);return response.get('data')
        if r.returncode or not response.get('success'):raise RuntimeError(redact(response.get('error',{})))
        data=response.get('data')
        if isinstance(data,dict) and data.get('success') is False:raise RuntimeError(redact(data))
        return data
    def rows(data):return data.get('rows',[]) if isinstance(data,dict) else data
    def save():MANIFEST.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def submit(kind,action,payload):
        path=HERE/('.flux-'+kind+'-'+action+'.json');path.write_text(json.dumps(payload),encoding='utf-8')
        try:
            call(kind,action,'--file',path.relative_to(ROOT).as_posix(),dry=True)
            return call(kind,action,'--file',path.relative_to(ROOT).as_posix())
        finally:path.unlink(missing_ok=True)
    call('auth','login','--api-key',account['api_key']);call('auth','status');who=call('auth','whoami')
    if who.get('email')!=EMAIL or str(who['userId'])!='4565':raise RuntimeError('selected account identity mismatch')
    print('ACCOUNT_VERIFIED',index+1,'userId',who['userId'],flush=True)
    record=json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}
    if record.get('task_id') and record.get('account_index')!=index:raise RuntimeError('existing task belongs to another account')
    record.update(experiment='exp3.4',account_index=index,user_id=who['userId'],email=EMAIL,
        source_checkpoint='model_26998.pt',source_completed_iterations=26998,additional_iterations=12000,target_completed_iterations=38998)
    if args.action=='prepare':
        available=rows(call('task','resource','list','--goods-back-category','3','--page','1','--limit','100'))
        print('RESOURCES',json.dumps([{k:r.get(k) for k in ('goodsId','goodsBackId','goodsName','goodsPrice')} for r in available],ensure_ascii=False))
        resource=next((r for r in available if r['goodsId']=='ESKU000005'),None)
        if resource is None:resource=next((r for r in available if r['goodsId']=='ESKU000001'),None)
        if resource is None:raise RuntimeError('no validated resource available')
        call('task','image','official');versions=rows(call('task','image','versions','--image-id','BJX00000001'))
        if not any(v['id']=='V000412' for v in versions):raise RuntimeError('validated image not available')
        record.update(goods_id=resource['goodsId'],goods_back_id=resource['goodsBackId'],hourly_price=resource['goodsPrice'],
            image_id='BJX00000001',image_version='V000412');save();return
    if args.action=='create':
        if record.get('task_id'):print('ALREADY_CREATED',record['task_id']);return
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if branch!=BRANCH:raise RuntimeError('unexpected branch')
        remote=subprocess.check_output(['git','-c','http.proxy=socks5h://10.12.201.122:39009','ls-remote','--heads','origin',BRANCH],text=True).strip()
        if not remote or remote.split()[0]!=commit:raise RuntimeError('training version is not synchronized to origin')
        pin=json.loads((ROOT/'resources/motions/processed/v1_2_exp34_manifest.json').read_text())['transfer_source']
        if hashlib.sha256(subprocess.check_output(['git','show',commit+':model_26998.pt'])).hexdigest()!=pin['sha256']:
            raise RuntimeError('committed transfer checkpoint hash differs')
        project_name='X1 exp3.4 B3 mature-policy transfer'
        project=next((p for p in rows(call('project','list','--page','1','--limit','100')) if p.get('projectName')==project_name),None)
        if project is None:project=submit('project','create',{'projectName':project_name})
        project_id=project['projectId'] if isinstance(project,dict) else project
        record.update(project_id=project_id,commit=commit,branch=branch,source_sha256=pin['sha256']);save()
        name='exp3.4-B3-transfer-26998-plus12000'
        existing=next((t for t in rows(call('task','list','--project-id',project_id,'--limit','100')) if t.get('taskName')==name),None)
        if existing:record.update(task_id=existing['taskId'],status=existing['taskStatus']);save();return
        script=('gm-run X1_29_amp/humanoid/scripts/train.py --task='+TASK+
            ' --headless --num_envs=4096 --seed=5 --max_iterations=12000 --ckpt_path=X1_29_amp/model_26998.pt --disc_fresh --run_name=exp3_4')
        payload=dict(taskBaseInfo=dict(projectId=project_id,taskType='1',trainType='2',taskName=name,
            taskDescription='Transfer post-axis-fix model_26998 to B3 forward only; 98 features; default+policy PD; fresh AMP and optimizers; nominal domain; 12000 additional updates.',
            taskTag=['exp3.4','B3','transfer','12000'],goodsId=record['goods_id'],goodsBackId=record['goods_back_id'],
            gpuNum=1,imageId=record['image_id'],imageVersion=record['image_version'],personalDataPath=''),
            taskCodeInfo=dict(codeType='2',codeUrl=json.dumps([dict(codeUrl='https://github.com/Lee-Weather/X1_29_amp',versionType='1',versionName=BRANCH)]),
                mainCodeUri='X1_29_amp/humanoid/scripts/train.py',hparamsPath='X1_29_amp/humanoid/envs/x1/x1_dh_stand_v12_exp34_config.py',startScript=script,isOpen='1'),
            runtimeReminderConfig=dict(enableRuntimeReminder=False,reminderDurations=[]))
        created=submit('task','create',payload);record.update(task_id=created['taskId'] if isinstance(created,dict) else created,status='0');save()
        (HERE/'task_config.json').write_text(json.dumps(payload,indent=2)+'\n');print('CREATED',record['task_id'],commit,flush=True);return
    info=call('task','info','--task-id',record['task_id']);base,code=info['taskBaseInfo'],info['taskCodeInfo']
    if str(base['userId'])!=str(who['userId']):raise RuntimeError('task owner mismatch')
    if not all(s in code.get('startScript','') for s in ['--task='+TASK,'--max_iterations=12000','--num_envs=4096','--ckpt_path=X1_29_amp/model_26998.pt','--disc_fresh']):
        raise RuntimeError('task arguments mismatch')
    if args.action=='clear-storage':
        if str(base['taskStatus'])!='0':raise RuntimeError('storage edit requires unstarted task')
        submit('task','edit',dict(taskBaseInfo=dict(base,personalDataPath=''),taskCodeInfo=dict(code,taskId=record['task_id']),
            runtimeReminderConfig=info.get('runtimeReminderConfig')));print('STORAGE_CLEARED');return
    if args.action=='run':
        run=json.loads((HERE/'preflight_latest.json').read_text())['run'];folder=HERE/('preflight_'+run)
        check=json.loads((folder/'summary.json').read_text())
        if not check.get('finished') or not check.get('ready_for_training'):raise RuntimeError('server preflight not ready')
        hashes=json.loads((folder/'manifest.json').read_text())['files']
        for name,expected in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('code changed since preflight: '+name)
        if base.get('personalDataPath'):raise RuntimeError('unexpected personal storage mount')
        if str(base['taskStatus'])=='0':
            call('task','run','--task-id',record['task_id'],dry=True);call('task','run','--task-id',record['task_id'])
            base=call('task','info','--task-id',record['task_id'])['taskBaseInfo']
        record['server_preflight_run']=run
    elif args.action=='note':
        note=('<h3>exp3.4 B3 forward transfer</h3><p>Source: post-axis-fix exp2.4a-cont2 model_26998.pt. '
            'Actor/critic/estimator inherited; discriminator and all optimizers fresh. Additional 12000 updates, target completed 38998, 4096 env, seed5. '
            '98 features; q_target=default+0.3*clip(action,-3,3). B3 only, phase63 / AMP351 frames, weight1.0. '
            'AMP scale5, task/style0.5/0.5, ref pose0.5. Real-second stand2-3 / walk6-9 / stop2-3 loop. '
            'Nominal calibrated dynamics, no noise/push/delay. Server source replay and smoke200 required before run.</p>'
            '<p>Commit '+record['commit']+'; branch '+BRANCH+'; owner '+str(who['userId'])+'.</p>')
        call('task','note','update','--task-id',record['task_id'],'--note',note,dry=True)
        call('task','note','update','--task-id',record['task_id'],'--note',note)
    else:
        log=call('task','logs','--task-id',record['task_id'],'--raw','--no-request-log',raw=True)
        (HERE/'startup.log').write_text(log,encoding='utf-8');clean=re.sub(r'\x1b\[[0-9;]*m','',log)
        matches=list(re.finditer(r'Learning iteration (\d+)/(\d+)',clean))
        if matches:
            record.update(latest_iteration=int(matches[-1][1]),target_iteration=int(matches[-1][2]))
            tail=clean[matches[-1].start():]
            for key,label in [('mean_reward','Mean reward'),('mean_episode_length','Mean episode length'),('eta_seconds','ETA')]:
                m=re.search(re.escape(label)+r':\s*([-+\d.eE]+)',tail)
                if m:record[key]=float(m[1])
            print('\n'.join(l.strip() for l in tail.splitlines() if any(k in l for k in ['Learning iteration','Computation:','Mean reward:','Mean episode length:','ETA:','AMP raw/used'])))
        else:print('LOG_TAIL\n'+'\n'.join(clean.splitlines()[-16:]))
        record['startup_error_lines']=[l for l in clean.splitlines() if any(k in l for k in ['Traceback','RuntimeError','ValueError','KeyError'])]
        record['checkpoint_loaded']='Loading model from ckpt_path:' in clean
        record['transfer_preflight_logged']='[TRANSFER]' in clean
        record['exp34_configuration_logged']='[EXP3.4]' in clean
    record.update(status=base['taskStatus'],start_time=base.get('startTime'),end_time=base.get('endTime'),checked_at=datetime.now(TZ).isoformat(timespec='seconds'));save()
    print('TASK_STATE',{k:base.get(k) for k in ['taskId','taskStatus','userId','startTime','endTime']},flush=True)


if __name__=='__main__':main()
