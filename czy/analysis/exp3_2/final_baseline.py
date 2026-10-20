"""Prepare/launch/collect final exp3.1 checkpoints under the existing replay protocol."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent / "final_baseline"
OLD = ROOT / "czy/analysis/exp3_1/replay_20261020_164054"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "status", "collect"))
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    manifest_path = HERE / "manifest.json"
    if args.action == "launch":
        snap = module(ROOT / "czy/analysis/exp3_1/download_replay_snapshots.py", "snap")
        run = (json.loads(manifest_path.read_text())["run"] if manifest_path.exists()
               else datetime.now().strftime("%Y%m%d_%H%M%S"))
        snap.RUN = run
        snap.STAGE = ROOT / "logs" / ("exp32_baseline_" + run)
        snap.OUT = HERE
        snap.main()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for letter, expected in (("a", 12000), ("b", 6000)):
            if manifest["experiments"][letter]["checkpoint"] != expected:
                raise RuntimeError("final checkpoint not yet available: " + letter)
        remote = manifest["server_root"]
        archive = snap.STAGE / "final_replay.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for letter, record in manifest["experiments"].items():
                tar.add(record["local_source"], arcname="models/" + letter + "/" + record["model"])
            tar.add(manifest_path, arcname="replay_manifest.json")
            for name in ("launch_server.py", "run_replays_server.py"):
                tar.add(OLD / name, arcname=name)
        # The existing source archive contains only code/setup, not credentials.
        source = ROOT / "logs/exp31_replay_20261020_164054/source.tar.gz"
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "robot@10.12.201.5",
                        "mkdir -p " + remote], check=True)
        for path in (archive, source):
            subprocess.run(["scp", "-o", "BatchMode=yes", str(path), "robot@10.12.201.5:" + remote + "/"], check=True)
        command = ("cd " + remote + " && tar -xzf source.tar.gz && tar -xzf final_replay.tar.gz"
                   " && ln -s /home/robot/czy/exp1/exp_20261020_164054/x1_exp31_replay/resources resources"
                   " && source /home/robot/Anaconda/etc/profile.d/conda.sh && conda activate F1"
                   " && python launch_server.py")
        subprocess.run(["ssh", "-o", "BatchMode=yes", "robot@10.12.201.5", command], check=True)
    elif args.action == "status":
        remote = json.loads(manifest_path.read_text())["server_root"]
        code = ("import json; from pathlib import Path; root=Path(" + repr(remote) + "); "
                "d=json.loads((root/'replay_summary.json').read_text()); "
                "print(json.dumps({'finished':d['finished'],'cases':{k:{'success':v['success'],'returncode':v['returncode']} for k,v in d['cases'].items()}})); "
                "print((root/'replay_batch.log').read_text().splitlines()[-1])")
        subprocess.run(["ssh", "-o", "BatchMode=yes", "robot@10.12.201.5",
                        "python3 -c " + shlex.quote(code)], check=True)
    else:
        collector = module(OLD / "collect_and_analyze.py", "collector")
        collector.ROOT, collector.HERE = ROOT, HERE
        collector.main()


if __name__ == "__main__":
    main()
