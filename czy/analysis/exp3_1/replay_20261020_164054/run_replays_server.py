"""Run the authorized exp3.1 snapshot replays sequentially on the server."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    manifest = json.loads((ROOT / "replay_manifest.json").read_text())
    summary_file = ROOT / "replay_summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {"cases": {}, "finished": False}
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT) + ":" + environment.get("PYTHONPATH", ""),
                       PLAY_RENDER="1", SDL_VIDEODRIVER="dummy", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       LD_LIBRARY_PATH="/home/robot/Anaconda/envs/F1/lib:" + environment.get("LD_LIBRARY_PATH", ""))
    (ROOT / "replay_logs").mkdir(exist_ok=True)
    for mode in ("nominal_no_push", "randomized_no_push"):
        for letter in ("a", "b"):
            name = letter + "_" + mode
            if name in summary["cases"] and summary["cases"][name].get("success"):
                print("ALREADY_FINISHED", name, flush=True)
                continue
            model = manifest["experiments"][letter]
            checkpoint = ROOT / "models" / letter / model["model"]
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != model["sha256"]:
                raise RuntimeError("checkpoint checksum mismatch: " + name)
            run_name = "exp31{}_{}_{}_{}".format(letter, model["checkpoint"], mode, datetime.now().strftime("%H%M%S"))
            log = ROOT / "replay_logs" / (name + ".log")
            command = [sys.executable, "-u", "humanoid/scripts/play.py", "--task=x1_dh_stand_v12_exp31",
                       "--headless", "--num_envs=1", "--seed=" + str(manifest["seed"]),
                       "--eval_mode=" + mode, "--ckpt_path=" + str(checkpoint), "--run_name=" + run_name]
            print("REPLAY_START", name, model["model"], flush=True)
            with log.open("w") as output:
                result = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
            record = dict(model=model["model"], mode=mode, seed=manifest["seed"], returncode=result.returncode,
                          run_name=run_name, log=str(log), success=False)
            if result.returncode == 0:
                source = ROOT / "logs/x1_dh_stand_v12_exp31/play_output"
                csv = next(source.glob("*_" + run_name + "_isaac_diag.csv"))
                metadata = csv.with_name(csv.name.replace("_isaac_diag.csv", "_metadata.json"))
                video = next(source.glob("*_" + run_name + ".mp4"))
                destination = ROOT / "replay_results" / name
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(csv, destination / "isaac_diag.csv")
                shutil.copyfile(metadata, ROOT / "replay_logs" / (name + "_metadata.json"))
                # Browser-compatible MP4, keeping the 50 fps / 30 s timing.
                encoding = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
                    str(destination / "play_output.mp4")], capture_output=True, text=True)
                if encoding.returncode:
                    raise RuntimeError("video encoding failed: " + encoding.stderr[-600:])
                record.update(success=True, directory=str(destination), metadata=json.loads(metadata.read_text()))
            summary["cases"][name] = record
            summary_file.write_text(json.dumps(summary, indent=2) + "\n")
            print("REPLAY_END", name, "returncode", result.returncode, flush=True)
            if result.returncode:
                print("\n".join(log.read_text(errors="replace").splitlines()[-25:]), flush=True)
                raise RuntimeError("playback failed: " + name)
    summary["finished"] = True
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print("ALL_REPLAYS_FINISHED", flush=True)


if __name__ == "__main__":
    main()
