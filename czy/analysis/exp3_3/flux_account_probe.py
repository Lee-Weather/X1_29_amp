"""Verify selected pool identities and quota without emitting credentials."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-index", type=int, required=True)
    args = parser.parse_args()
    pool = json.loads((ROOT / "czy/skills/flux-cli/api_key.json").read_text(encoding="utf-8-sig"))["accounts"]
    account = pool[args.account_index]
    environment = os.environ.copy()
    for name in list(environment):
        if name.lower() in ("http_proxy", "https_proxy", "all_proxy"):
            environment.pop(name)
    cli = shutil.which("flux.cmd")
    def invoke(*command):
        result = subprocess.run([cli, "--yes", *command], env=environment, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        try:
            response = json.loads(result.stdout)
        except ValueError:
            raise RuntimeError("Flux failed; raw output withheld") from None
        if result.returncode or not response.get("success"):
            error = str(response.get("error"))
            for entry in pool:
                error = error.replace(entry["api_key"], "[REDACTED]")
            raise RuntimeError(re.sub(r"https?://[^\s\"]+", "[URL]", error))
        return response["data"]
    invoke("auth", "login", "--api-key", account["api_key"])
    invoke("auth", "status")
    who = invoke("auth", "whoami")
    if who.get("email") != account["email"]:
        raise RuntimeError("identity mismatch")
    numeric = {}
    def find_quota(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                path = prefix + key
                if re.search(r"balance|amount|wallet|quota|credit|used|remaining", key, re.I):
                    if isinstance(item, (int, float)) or (isinstance(item, str) and re.fullmatch(r"[-+]?\d+(\.\d+)?", item)):
                        numeric[path] = item
                if isinstance(item, dict):
                    find_quota(item, path + ".")
    find_quota(who)
    tasks = invoke("task", "list", "--page", "1", "--limit", "100")
    tasks = tasks.get("rows", []) if isinstance(tasks, dict) else tasks
    print(json.dumps(dict(index=args.account_index, user_id=who["userId"],
                          email_mask=account["email"][:3]+"..."+account["email"][-20:],
                          github_name=who.get("github_name", who.get("githubName")),
                          quota=numeric, identity_field_names=sorted(who),
                          running_tasks=[{key: item.get(key) for key in ("taskId", "taskName", "taskStatus")}
                                         for item in tasks if str(item.get("taskStatus")) == "3"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
