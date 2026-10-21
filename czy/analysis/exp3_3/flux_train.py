"""Account-checked, idempotent Flux launch for the authorized exp3.3 run."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "launch.json"
BRANCH = "codex/exp3-3"
TASK = "x1_dh_stand_v12_exp33"
TZ = timezone(timedelta(hours=8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "create", "run", "status", "clear-storage", "note"))
    parser.add_argument("--account-index", type=int, default=5)
    args = parser.parse_args()
    os.chdir(ROOT)
    pool = json.loads((ROOT / "czy/skills/flux-cli/api_key.json").read_text(encoding="utf-8-sig"))["accounts"]
    account = pool[args.account_index]
    if account.get("exhausted") and args.action != "status":
        raise RuntimeError("account is marked exhausted")
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in ("http_proxy", "https_proxy", "all_proxy"):
            env.pop(key)
    cli = shutil.which("flux.cmd") or shutil.which("flux")

    def redact(value):
        value = str(value)
        for item in pool:
            value = value.replace(item["api_key"], "[REDACTED]")
        return re.sub(r"https?://[^\s\"<>]+", "[URL]", value)

    def invoke(*command, dry=False, raw=False):
        result = subprocess.run([cli, "--yes", *(["--dry-run"] if dry else []), *command],
                                env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if raw:
            if result.returncode:
                raise RuntimeError(redact(result.stderr))
            return redact(result.stdout)
        try:
            response = json.loads(result.stdout)
        except ValueError:
            raise RuntimeError(redact(result.stderr or result.stdout)) from None
        if dry and result.returncode == 10:
            print("DRY_RUN_OK", command[:2], flush=True)
            return response.get("data")
        if result.returncode or not response.get("success"):
            raise RuntimeError(redact(response.get("error", {})))
        data = response.get("data")
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(redact(data))
        return data

    def rows(data):
        return data.get("rows", []) if isinstance(data, dict) else data

    def save():
        MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def submit(kind, action, payload):
        path = HERE / (".flux-" + kind + "-" + action + ".json")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            invoke(kind, action, "--file", path.relative_to(ROOT).as_posix(), dry=True)
            return invoke(kind, action, "--file", path.relative_to(ROOT).as_posix())
        finally:
            path.unlink(missing_ok=True)

    invoke("auth", "login", "--api-key", account["api_key"])
    invoke("auth", "status")
    who = invoke("auth", "whoami")
    if who.get("email") != account["email"]:
        raise RuntimeError("Flux account verification failed")
    print("ACCOUNT_VERIFIED", args.account_index + 1, "userId", who["userId"], flush=True)
    record = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    if record.get("task_id") and record.get("account_index") != args.account_index:
        raise RuntimeError("existing task belongs to a different recorded account")
    record.update(experiment="exp3.3", account_index=args.account_index, user_id=who["userId"],
                  email_mask=account["email"][:3] + "..." + account["email"][-20:])
    if args.action == "prepare":
        available = rows(invoke("task", "resource", "list", "--goods-back-category", "3", "--page", "1", "--limit", "100"))
        print("RESOURCES", json.dumps([{k: r.get(k) for k in ("goodsId", "goodsBackId", "goodsName", "goodsPrice")}
                                        for r in available], ensure_ascii=False), flush=True)
        resource = next((r for r in available if r["goodsId"] == "ESKU000005"), None)
        if resource is None:
            resource = next((r for r in available if r["goodsId"] == "ESKU000001"), None)
        if resource is None:
            raise RuntimeError("no validated resource available")
        invoke("task", "image", "official")
        versions = rows(invoke("task", "image", "versions", "--image-id", "BJX00000001"))
        if not any(v["id"] == "V000412" for v in versions):
            raise RuntimeError("validated image V000412 not returned")
        record.update(goods_id=resource["goodsId"], goods_back_id=resource["goodsBackId"],
                      hourly_price=resource["goodsPrice"], image_id="BJX00000001", image_version="V000412")
        tasks = rows(invoke("task", "list", "--page", "1", "--limit", "200"))
        print("ACCOUNT_RUNNING_TASKS", [{k: t.get(k) for k in ("taskId", "taskName", "taskStatus")}
                                        for t in tasks if str(t.get("taskStatus")) == "3"], flush=True)
        save()
        return

    if args.action == "create":
        if record.get("task_id"):
            print("ALREADY_CREATED", record["task_id"])
            return
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if branch != BRANCH:
            raise RuntimeError("unexpected training branch")
        remote = subprocess.check_output(["git", "-c", "http.proxy=socks5h://10.12.201.122:39009",
                                          "ls-remote", "--heads", "origin", BRANCH], text=True).strip()
        if not remote or remote.split()[0] != commit:
            raise RuntimeError("origin training branch does not match local commit")
        project_name = "X1 exp3.3 reference residual walking"
        projects = rows(invoke("project", "list", "--page", "1", "--limit", "100"))
        project = next((p for p in projects if p.get("projectName") == project_name), None)
        if project is None:
            project = submit("project", "create", {"projectName": project_name})
        project_id = project["projectId"] if isinstance(project, dict) else project
        record.update(project_id=project_id, commit=commit, branch=branch)
        save()
        existing = rows(invoke("task", "list", "--project-id", project_id, "--page", "1", "--limit", "100"))
        matching = next((item for item in existing if item.get("taskName") == "exp3.3-fresh-reference-residual-6000"), None)
        if matching is not None:
            record.update(task_id=matching["taskId"], status=matching["taskStatus"])
            save()
            print("RECOVERED_TASK", record["task_id"])
            return
        script = ("gm-run X1_29_amp/humanoid/scripts/train.py --task=" + TASK
                  + " --headless --num_envs=4096 --seed=5 --max_iterations=6000 --run_name=exp3_3")
        payload = dict(taskBaseInfo=dict(projectId=project_id, taskType="1", trainType="1",
            taskName="exp3.3-fresh-reference-residual-6000", taskDescription="Reference joint targets + bounded angle residuals; 127 actor features; nominal calibrated domain, no delays/noise/pushes; fresh all networks 6000 updates. Zero-residual standing failed; policy must learn balance.",
            taskTag=["exp3.3", "v1.2", "6000"], goodsId=record["goods_id"], goodsBackId=record["goods_back_id"],
            gpuNum=1, imageId=record["image_id"], imageVersion=record["image_version"], personalDataPath=""),
            taskCodeInfo=dict(codeType="2", codeUrl=json.dumps([dict(codeUrl="https://github.com/Lee-Weather/X1_29_amp",
                versionType="1", versionName=BRANCH)]), mainCodeUri="X1_29_amp/humanoid/scripts/train.py",
                hparamsPath="X1_29_amp/humanoid/envs/x1/x1_dh_stand_v12_exp33_config.py", startScript=script, isOpen="1"),
            runtimeReminderConfig=dict(enableRuntimeReminder=False, reminderDurations=[]))
        result = submit("task", "create", payload)
        record.update(task_id=result["taskId"] if isinstance(result, dict) else result, status="0")
        save()
        (HERE / "task_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("CREATED", record["task_id"], commit, flush=True)
        return

    info = invoke("task", "info", "--task-id", record["task_id"])
    base, code = info["taskBaseInfo"], info["taskCodeInfo"]
    if str(base["userId"]) != str(who["userId"]):
        raise RuntimeError("task owner mismatch")
    script = code.get("startScript", "")
    if not all(text in script for text in ("--task=" + TASK, "--max_iterations=6000", "--num_envs=4096")):
        raise RuntimeError("unexpected start arguments")
    if "--ckpt_path" in script or "--resume" in script or code.get("checkPointFilePath"):
        raise RuntimeError("fresh training unexpectedly restores a checkpoint")
    if args.action == "note":
        note = ("<h3>exp3.3: reference plus bounded angle residuals</h3>"
                "<p>Fresh actor/critic/estimator/AMP; 6000 updates, 4096 envs, seed 5. "
                "127 features per actor frame. q_target=q_ref+delta_q; residual limits: hip pitch/knee 0.20, "
                "ankles 0.12, hip roll/yaw and arms 0.10, lumbar 0.08 rad. "
                "Zero actor output initialization; initial raw std=0.3. "
                "B3/B4 demos 1:1; geometry-calibrated independent references; standing/walking grouped Huber. "
                "Nominal calibrated dynamics, no execution/sensor delay, observation noise or pushes. "
                "Zero-residual check: standing 10s/6 failures, profile 30s/18 failures; "
                "target-reference difference zero, no PD target or torque clipping. "
                "This task trains the residual policy to learn balance; gait success is not established.</p>"
                "<p>Training commit: " + record["commit"] + "; branch: " + record["branch"]
                + "; account userId: " + str(record["user_id"]) + ".</p>")
        invoke("task", "note", "update", "--task-id", record["task_id"], "--note", note, dry=True)
        invoke("task", "note", "update", "--task-id", record["task_id"], "--note", note)
        record["note_updated_at"] = datetime.now(TZ).isoformat(timespec="seconds")
        save()
        print("NOTE_UPDATED", record["task_id"])
        return
    if args.action == "clear-storage":
        if str(base["taskStatus"]) != "0":
            raise RuntimeError("storage repair requires an unstarted task")
        submit("task", "edit", dict(taskBaseInfo=dict(base, personalDataPath=""),
               taskCodeInfo=dict(code, taskId=record["task_id"]), runtimeReminderConfig=info.get("runtimeReminderConfig")))
        print("PERSONAL_STORAGE_DISABLED", record["task_id"])
        return
    if args.action == "run":
        if str(base["taskStatus"]) == "0":
            invoke("task", "run", "--task-id", record["task_id"], dry=True)
            invoke("task", "run", "--task-id", record["task_id"])
            base = invoke("task", "info", "--task-id", record["task_id"])["taskBaseInfo"]
    else:
        log = invoke("task", "logs", "--task-id", record["task_id"], "--raw", "--no-request-log", raw=True)
        (HERE / "startup.log").write_text(log, encoding="utf-8")
        clean = re.sub(r"\x1b\[[0-9;]*m", "", log)
        matches = list(re.finditer(r"Learning iteration (\d+)/(\d+)", clean))
        if matches:
            record.update(first_observed_iteration=min(record.get("first_observed_iteration", int(matches[0][1])), int(matches[0][1])),
                          latest_iteration=int(matches[-1][1]), target_iteration=int(matches[-1][2]))
            summary = clean[matches[-1].start():]
            for key, label in (("mean_reward", "Mean reward"), ("mean_episode_length", "Mean episode length"), ("eta_seconds", "ETA")):
                match = re.search(re.escape(label) + r":\s*([-+\d.eE]+)", summary)
                if match:
                    record[key] = float(match[1])
            record["latest_exp33_metrics"] = {name: float(value) for name, value in re.findall(
                r"Mean episode exp33_([a-z_]+):\s*([-+\d.eE]+)", summary)}
            labels = ("Learning iteration", "Computation:", "Mean reward:", "Mean episode length:", "AMP raw/used", "Task/style contribution:", "AMP direction gate:", "ETA:")
            print("\n".join(line.strip() for line in summary.splitlines() if any(label in line for label in labels)), flush=True)
        else:
            print("LOG_TAIL\n" + "\n".join(clean.splitlines()[-15:]), flush=True)
        errors = [line for line in clean.splitlines() if any(word in line for word in ("Traceback", "KeyError", "RuntimeError", "ValueError"))]
        record.update(startup_error_lines=errors, exp33_configuration_logged=record.get("exp33_configuration_logged", False) or "[EXP3.3]" in clean)
        if errors:
            print("ERRORS", errors, flush=True)
        models = rows(invoke("task", "model", "list", "--task-id", record["task_id"], "--page", "1", "--limit", "20"))
        checkpoints = [m for m in models if re.fullmatch(r"model_\d+\.pt", m.get("fileName", ""))]
        if checkpoints:
            latest = max(checkpoints, key=lambda item: int(item["checkpoint"]))
            record["latest_checkpoint"] = {k: latest.get(k) for k in ("fileName", "checkpoint", "createTime")}
            print("LATEST_CHECKPOINT", record["latest_checkpoint"], flush=True)
    record.update(status=base["taskStatus"], start_time=base.get("startTime"), end_time=base.get("endTime"),
                  checked_at=datetime.now(TZ).isoformat(timespec="seconds"))
    save()
    print("TASK_STATE", {k: base.get(k) for k in ("taskId", "taskStatus", "userId", "startTime", "endTime")}, flush=True)


if __name__ == "__main__":
    main()
