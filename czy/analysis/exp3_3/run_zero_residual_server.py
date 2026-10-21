"""Run the stand and speed-profile reference tracking checks sequentially."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))


def main():
    os.chdir(ROOT)
    manifest = json.loads((ROOT / "replay_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("source checksum mismatch: " + name)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT) + ":" + environment.get("PYTHONPATH", ""),
                       PLAY_RENDER="1", SDL_VIDEODRIVER="dummy", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       LD_LIBRARY_PATH="/home/robot/Anaconda/envs/F1/lib:" + environment.get("LD_LIBRARY_PATH", ""))
    summary_path = ROOT / "replay_summary.json"
    summary = dict(finished=False, cases={}, started_at=datetime.now(TZ).isoformat())
    (ROOT / "replay_logs").mkdir(exist_ok=True)
    for name, extra in (("stand10", ["--play_speed=0", "--play_duration=10"]), ("profile30", [])):
        run_name = "exp33_zero_" + name
        log = ROOT / "replay_logs" / (name + ".log")
        command = [sys.executable, "-u", "humanoid/scripts/play.py", "--task=x1_dh_stand_v12_exp33",
                   "--zero_residual", "--headless", "--num_envs=1", "--seed=123145",
                   "--eval_mode=nominal_no_push", "--run_name=" + run_name] + extra
        print("REPLAY_START", name, flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record = dict(returncode=result.returncode, success=False, log=str(log), command=command)
        if result.returncode == 0:
            source = ROOT / "logs/x1_dh_stand_v12_exp33/play_output"
            csv = next(source.glob("*_" + run_name + "_isaac_diag.csv"))
            video = next(source.glob("*_" + run_name + ".mp4"))
            metadata = csv.with_name(csv.name.replace("_isaac_diag.csv", "_metadata.json"))
            destination = ROOT / "replay_results" / name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(csv, destination / "isaac_diag.csv")
            shutil.copyfile(metadata, ROOT / "replay_logs" / (name + "_metadata.json"))
            encoded = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video),
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
                str(destination / "play_output.mp4")], capture_output=True, text=True)
            if encoded.returncode:
                raise RuntimeError("video encoding failed: " + encoded.stderr[-600:])
            record.update(success=True, directory=str(destination), metadata=json.loads(metadata.read_text()))
        summary["cases"][name] = record
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print("REPLAY_END", name, "returncode", result.returncode, flush=True)
        if result.returncode:
            print("\n".join(log.read_text(errors="replace").splitlines()[-35:]), flush=True)
            raise RuntimeError("playback failed: " + name)
    summary.update(finished=True, finished_at=datetime.now(TZ).isoformat())
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("ALL_REPLAYS_FINISHED", flush=True)


if __name__ == "__main__":
    main()
