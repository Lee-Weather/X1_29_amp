"""Launch once, returning promptly without keeping the SSH session open."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
pid_file = root / "replay_batch.pid"
if pid_file.exists():
    try:
        os.kill(int(pid_file.read_text()), 0)
        print("BATCH_ALREADY_RUNNING", pid_file.read_text().strip())
        raise SystemExit(0)
    except ProcessLookupError:
        pass
with (root / "replay_batch.log").open("a") as output:
    process = subprocess.Popen([sys.executable, "-u", str(root / "run_replays_server.py")], cwd=str(root),
                               stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                               start_new_session=True)
pid_file.write_text(str(process.pid) + "\n")
print("BATCH_STARTED", process.pid)
