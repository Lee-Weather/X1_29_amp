"""Bundle current exp3.3 sources, run reference-only checks on 201.5, collect artifacts."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
POINTER = HERE / "zero_residual_latest.json"
TZ = timezone(timedelta(hours=8))
SSH_EXE = str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/OpenSSH/ssh.exe")
SCP_EXE = str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/OpenSSH/scp.exe")
SSH = [SSH_EXE, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "robot@10.12.201.5"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "status", "collect", "validate"))
    args = parser.parse_args()
    if args.action == "prepare":
        run = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
        POINTER.write_text(json.dumps(dict(run=run)) + "\n")
    run = json.loads(POINTER.read_text())["run"]
    out = HERE / ("zero_residual_" + run)
    stage = ROOT / "logs" / ("exp33_zero_" + run)
    out.mkdir(exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    remote = "/home/robot/czy/exp1/exp_" + run + "/x1_exp33_zero_residual"
    if args.action == "prepare":
        files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "humanoid").rglob("*.py")}
        urdf = "resources/robots/x1/urdf/X1_29DOF_physically_mirrored.urdf"
        files.update(("setup.py", urdf,
                      "resources/motions/processed/ref_lib_v1_2_exp33_phase.pt",
                      "resources/motions/processed/amp_v1_2_exp33.pt",
                      "resources/motions/processed/v1_2_exp33_manifest.json"))
        for mesh in ET.parse(ROOT / urdf).iter("mesh"):
            files.add(((ROOT / urdf).parent / mesh.attrib["filename"]).resolve().relative_to(ROOT).as_posix())
        manifest = dict(run=run, source="current working tree, including uncommitted exp3.3 changes",
                        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        created_at=datetime.now(TZ).isoformat(), server_root=remote,
                        task="x1_dh_stand_v12_exp33", checkpoint=None, residual="zero",
                        seed=123145, files={name: digest(ROOT / name) for name in sorted(files)})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        archive = stage / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name in sorted(files):
                tar.add(ROOT / name, arcname=name)
            tar.add(manifest_path, arcname="replay_manifest.json")
            tar.add(HERE / "run_zero_residual_server.py", arcname="run_zero_residual_server.py")
            tar.add(ROOT / "czy/analysis/exp3_1/replay_20261020_164054/launch_server.py", arcname="launch_server.py")
        print("PREPARED", run, "files", len(files), "archive_bytes", archive.stat().st_size,
              "archive_sha256", digest(archive))
        return
    if args.action == "launch":
        subprocess.run([*SSH, "mkdir -p " + shlex.quote(remote)], check=True)
        subprocess.run([SCP_EXE, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        str(stage / "source.tar.gz"), "robot@10.12.201.5:" + remote + "/"], check=True)
        command = ("source /home/robot/Anaconda/etc/profile.d/conda.sh && conda activate F1 && cd "
                   + shlex.quote(remote) + " && tar -xzf source.tar.gz && mv run_zero_residual_server.py run_replays_server.py"
                   + " && python launch_server.py")
        subprocess.run([*SSH, command], check=True)
        return
    if args.action == "status":
        code = ("from pathlib import Path; p=Path(" + repr(remote) + "); "
                "s=p/'replay_summary.json'; print(s.read_text() if s.exists() else 'NO_SUMMARY'); "
                "p=p/'replay_batch.log'; print('\\n'.join(p.read_text().splitlines()[-45:]) if p.exists() else 'NO_LOG')")
        subprocess.run([*SSH, "python3 -c " + shlex.quote(code)], check=True)
        return
    if args.action == "validate":
        subprocess.run([SCP_EXE, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        str(HERE / "validate_interface_server.py"), "robot@10.12.201.5:" + remote + "/"], check=True)
        command = ("source /home/robot/Anaconda/etc/profile.d/conda.sh && conda activate F1 && cd "
                   + shlex.quote(remote) + " && PYTHONPATH=. python validate_interface_server.py")
        subprocess.run([*SSH, command], check=True)
        subprocess.run([SCP_EXE, "-o", "BatchMode=yes", "robot@10.12.201.5:" + remote
                        + "/interface_validation/result.json", str(out / "interface_validation.json")], check=True)
        return
    def fetch(source, destination):
        subprocess.run([SCP_EXE, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        "robot@10.12.201.5:" + source, str(destination)], check=True)
    fetch(remote + "/replay_summary.json", out / "server_summary.json")
    summary = json.loads((out / "server_summary.json").read_text())
    fetch(remote + "/replay_batch.log", out / "replay_batch.log")
    for name, case in summary["cases"].items():
        fetch(case["log"], out / (name + ".log"))
        if case.get("success"):
            destination = out / name
            destination.mkdir(exist_ok=True)
            for filename in ("isaac_diag.csv", "play_output.mp4"):
                fetch(case["directory"] + "/" + filename, destination / filename)
            fetch(remote + "/replay_logs/" + name + "_metadata.json", out / (name + "_metadata.json"))
            print("COLLECTED", name, str(destination))


if __name__ == "__main__":
    main()
