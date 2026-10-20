"""Download current checkpoints from the owned, running exp3.1 tasks."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
RUN = "20261020_164054"
STAGE = ROOT / "logs" / ("exp31_replay_" + RUN)
OUT = ROOT / "czy/analysis/exp3_1" / ("replay_" + RUN)


def main():
    os.chdir(ROOT)
    STAGE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "run": RUN, "source_commit": "e3033b3cd6f7ede4c70c9e1ef0e8730bf0bf696b",
        "seed": 123145, "profile": [0., .4, .6, 0., -.2, 0.], "stage_seconds": 5.,
        "server_root": "/home/robot/czy/exp1/exp_" + RUN + "/x1_exp31_replay", "experiments": {}}
    pool = json.loads((ROOT / "czy/skills/flux-cli/api_key.json").read_text(encoding="utf-8"))["accounts"]
    environment = os.environ.copy()
    for name in list(environment):
        if name.lower() in ("http_proxy", "https_proxy", "all_proxy"):
            environment.pop(name)
    cli = shutil.which("flux.cmd") or shutil.which("flux")

    def invoke(*args):
        result = subprocess.run([cli, "--yes", *args], env=environment, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        try:
            response = json.loads(result.stdout)
        except ValueError:
            raise RuntimeError("Flux returned no JSON; output omitted") from None
        if result.returncode or not response.get("success"):
            message = json.dumps(response.get("error"), ensure_ascii=False)
            for account in pool:
                message = message.replace(account["api_key"], "[REDACTED]")
            raise RuntimeError(re.sub(r"https?://[^\s\"]+", "[URL]", message))
        return response["data"]

    for letter, index, task, offset in (("a", 5, "TASK_20261020_072", 6000), ("b", 7, "TASK_20261020_073", 0)):
        invoke("auth", "login", "--api-key", pool[index]["api_key"])
        who = invoke("auth", "whoami")
        assert who["email"] == pool[index]["email"]
        base = invoke("task", "info", "--task-id", task)["taskBaseInfo"]
        assert str(base["userId"]) == str(who["userId"])
        models = invoke("task", "model", "list", "--task-id", task, "--page", "1", "--limit", "100")["rows"]
        model = max((m for m in models if re.fullmatch(r"model_\d+\.pt", m.get("fileName", ""))),
                    key=lambda m: int(m["checkpoint"]))
        # Pin the snapshot selected at the first successful download.
        if letter in manifest["experiments"]:
            print("SNAPSHOT_ALREADY_PINNED", letter, manifest["experiments"][letter]["model"], flush=True)
            continue
        destination = STAGE / "models" / letter / model["fileName"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        download = subprocess.run(["curl.exe", "--fail", "--location", "--ssl-no-revoke", "--silent", "--show-error",
                                   "--connect-timeout", "15", "--max-time", "120",
                                   "--proxy", "socks5h://10.12.201.122:39009",
                                   "--output", str(destination), model["policUrlDown"]],
                                  env=environment, capture_output=True, text=True)
        if download.returncode:
            error = re.sub(r"https?://[^\s\"]+", "[URL]", download.stderr)
            raise RuntimeError("checkpoint download failed ({}): {}".format(download.returncode, error))
        if destination.stat().st_size < 1000000:
            raise RuntimeError("downloaded checkpoint is unexpectedly small")
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        manifest["experiments"][letter] = dict(task_id=task, status_at_download=base["taskStatus"],
            account_index=index, user_id=who["userId"], model=model["fileName"], checkpoint=int(model["checkpoint"]),
            iteration_offset=offset, model_created_at=model["createTime"], downloaded_at=datetime.now().astimezone().isoformat(),
            local_source=str(destination), bytes=destination.stat().st_size, sha256=digest)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("DOWNLOADED", letter, model["fileName"], destination.stat().st_size, "bytes", "sha256", digest, flush=True)
    print("REPLAY_MANIFEST", manifest_path.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
