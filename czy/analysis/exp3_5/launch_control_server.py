import os,sys,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
pid=root/'control_batch.pid'
if pid.exists():
 try:
  os.kill(int(pid.read_text()),0);print('CONTROL_ALREADY_RUNNING');raise SystemExit(0)
 except ProcessLookupError:pass
with (root/'control_batch.log').open('a') as f:
 p=subprocess.Popen([sys.executable,'-u',str(root/'run_control_server.py')],cwd=str(root),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
pid.write_text(str(p.pid));print('CONTROL_LAUNCHED',p.pid)
