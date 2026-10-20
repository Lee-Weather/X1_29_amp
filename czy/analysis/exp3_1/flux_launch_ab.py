"""Explicit, account-checked Flux operations for the approved exp3.1 a/b run.

Credentials stay in the ignored account pool. Task requests use short-lived
relative JSON files, removed immediately after a successful create.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
POOL = ROOT / "czy/skills/flux-cli/api_key.json"
MANIFEST = HERE / "launch_ab.json"
BRANCH = "codex/exp3-1-ab"
TASK = "x1_dh_stand_v12_exp31"
ACCOUNTS = {"a": 5, "b": 7}  # Zero-based indices; 094's exhausted account is 6.


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "create", "retry-create", "run", "status", "clear-storage"))
    parser.add_argument("experiment", choices=tuple(ACCOUNTS))
    args = parser.parse_args()
    os.chdir(ROOT)
    pool = read_json(POOL)["accounts"]
    account_index = ACCOUNTS[args.experiment]
    account = pool[account_index]
    if account.get("exhausted") and args.action != "status":
        raise RuntimeError("selected account is marked exhausted")
    environment = os.environ.copy()
    for name in list(environment):
        if name.lower() in ("http_proxy", "https_proxy", "all_proxy"):
            environment.pop(name)
    cli = shutil.which("flux.cmd") or shutil.which("flux")

    def redact(value):
        text = str(value)
        for entry in pool:
            text = text.replace(entry["api_key"], "[REDACTED]")
        return re.sub(r"https?://[^\s\"<>]+", "[URL]", text)

    def invoke(*command, dry_run=False, raw=False):
        flags = ["--yes"] + (["--dry-run"] if dry_run else [])
        result = subprocess.run([cli, *flags, *command], env=environment,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if raw:
            if result.returncode:
                raise RuntimeError(redact(result.stderr))
            return redact(result.stdout)
        try:
            response = json.loads(result.stdout)
        except ValueError:
            raise RuntimeError(redact(result.stderr or result.stdout)) from None
        if dry_run and result.returncode == 10:
            print("DRY_RUN_OK", command[0], command[1])
            return response.get("data")
        if result.returncode or not response.get("success"):
            raise RuntimeError(redact(json.dumps(response.get("error", {}), ensure_ascii=False)))
        data = response.get("data")
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(redact(json.dumps(data, ensure_ascii=False)))
        return data

    def save_manifest():
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def rows(data):
        return data.get("rows", []) if isinstance(data, dict) else data

    def create_with_file(kind, payload):
        path = HERE / (".flux-create-{}-{}.json".format(kind, args.experiment))
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        invoke(kind, "create", "--file", relative, dry_run=True)
        result = invoke(kind, "create", "--file", relative)
        path.unlink()
        return result

    # Switch the real CLI identity first, then verify server ownership.
    invoke("auth", "login", "--api-key", account["api_key"])
    who = invoke("auth", "whoami")
    if who.get("email") != account["email"]:
        raise RuntimeError("Flux identity does not match selected pool entry")
    print("ACCOUNT_VERIFIED", account_index + 1, "userId", who["userId"])
    manifest = read_json(MANIFEST, {})
    record = manifest.setdefault(args.experiment, {})
    record.update(experiment="exp3.1" + args.experiment, account_index=account_index,
                  user_id=who["userId"], email_mask=account["email"][:3] + "..." + account["email"][-24:])
    if args.action == "retry-create":
        failed = invoke("task", "info", "--task-id", record["task_id"])["taskBaseInfo"]
        if str(failed["userId"]) != str(who["userId"]) or str(failed["taskStatus"]) != "6":
            raise RuntimeError("retry requires an owned, failed task")
        previous = {k: failed.get(k) for k in ("taskId", "taskStatus", "startTime", "endTime")}
        previous.update(commit=record["commit"], reason="episode logger KeyError on empty episode entries")
        record.setdefault("failed_attempts", []).append(previous)
        log = HERE / ("startup_" + args.experiment + ".log")
        if log.exists():
            shutil.copyfile(log, HERE / ("startup_{}_{}.log".format(args.experiment, failed["taskId"])))
        record.pop("task_id")
        record["attempt"] = record.get("attempt", 1) + 1
        save_manifest()
        args.action = "create"
    name = "exp3.1{}-{}-6000".format(args.experiment, "resume085" if args.experiment == "a" else "fresh")
    if record.get("attempt", 1) > 1:
        name += "-r" + str(record["attempt"])

    if args.action == "prepare":
        resource_rows = rows(invoke("task", "resource", "list", "--goods-back-category", "3",
                                    "--page", "1", "--limit", "100"))
        resources = [{key: item.get(key) for key in ("goodsId", "goodsBackId", "goodsName", "goodsPrice", "goodsStatus")}
                     for item in resource_rows]
        print("AVAILABLE_RESOURCES", json.dumps(resources, ensure_ascii=False))
        selected = next((r for r in resources if r["goodsId"] == "ESKU000001"), None)
        if selected is None:
            raise RuntimeError("the matched 4090D resource is unavailable")
        versions = invoke("task", "image", "versions", "--image-id", "BJX00000001")
        if not any(v["id"] == "V000412" for v in versions):
            raise RuntimeError("the matched Isaac Gym image version is unavailable")
        record.update(goods_id=selected["goodsId"], goods_back_id=selected["goodsBackId"],
                      hourly_price=selected["goodsPrice"], image_id="BJX00000001", image_version="V000412")
        tasks = rows(invoke("task", "list", "--page", "1", "--limit", "200"))
        print("ACCOUNT_TASK_COUNT", len(tasks))
        print("EXISTING_RUNNING_TASKS", json.dumps([
            {k: t.get(k) for k in ("taskId", "taskName", "taskStatus")} for t in tasks if str(t.get("taskStatus")) == "3"], ensure_ascii=False))
        if args.experiment == "a":
            source = invoke("task", "info", "--task-id", "TASK_20261019_085")["taskBaseInfo"]
            if str(source["userId"]) != str(who["userId"]):
                raise RuntimeError("085 owner mismatch")
            models = rows(invoke("task", "model", "list", "--task-id", source["taskId"],
                                 "--checkpoint", "6000", "--limit", "1"))
            model = next(m for m in models if str(m["checkpoint"]) == "6000")
            record.update(source_task=source["taskId"], source_name=source["taskName"], source_iteration=6000,
                          source_model=model["fileName"], source_object=model["policUrl"])
        save_manifest()
        print("PREPARED", record["experiment"])
        return

    if args.action == "create":
        expected = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        if branch != BRANCH:
            raise RuntimeError("unexpected training branch")
        record.update(commit=expected, branch=branch)
        if record.get("task_id"):
            print("ALREADY_CREATED", record["task_id"])
            return
        project_name = "X1 exp3.1{} {}".format(args.experiment, "085 continuation" if args.experiment == "a" else "fresh 6000")
        projects = rows(invoke("project", "list", "--page", "1", "--limit", "100"))
        matching = next((p for p in projects if p.get("projectName") == project_name), None)
        if matching is None:
            result = create_with_file("project", {"projectName": project_name})
            matching = result if isinstance(result, dict) else {"projectId": result}
        project_id = matching["projectId"]
        record["project_id"] = project_id
        # Recover an already successful create if a previous response was lost.
        existing = rows(invoke("task", "list", "--project-id", project_id, "--page", "1", "--limit", "100"))
        matching_task = next((t for t in existing if t.get("taskName") == name), None)
        if matching_task is not None:
            record["task_id"] = matching_task["taskId"]
            save_manifest()
            print("RECOVERED_TASK", record["task_id"])
            return
        description = ("exp3.1 corrections; 4096 envs; seed 5; 6000 additional iterations; "
                       + ("inherit 085 model_6000 actor/critic/estimator/AMP, reset optimizers; target cumulative 12000."
                          if args.experiment == "a" else "initialize all networks and optimizers from scratch; target 6000."))
        script = ("gm-run X1_29_amp/humanoid/scripts/train.py --task=" + TASK
                  + " --headless --num_envs=4096 --seed=5 --max_iterations=6000 --run_name=exp3_1" + args.experiment)
        code = dict(codeType="2", codeUrl=json.dumps([dict(codeUrl="https://github.com/Lee-Weather/X1_29_amp",
                    versionType="1", versionName=BRANCH)]), mainCodeUri="X1_29_amp/humanoid/scripts/train.py",
                    hparamsPath="X1_29_amp/humanoid/envs/x1/x1_dh_stand_v12_exp31_config.py", isOpen="1")
        if args.experiment == "a":
            code.update(checkPointFilePath=record["source_object"], checkPointMountPath="X1_29_amp/",
                        resumeFromTaskId=record["source_task"], resumeFromTaskName=record["source_name"],
                        resumeFromCheckPoint="6000")
            script += " --ckpt_path=X1_29_amp/model_6000.pt"
        code["startScript"] = script
        payload = dict(taskBaseInfo=dict(projectId=project_id, taskType="1", trainType="2" if args.experiment == "a" else "1",
            taskName=name, taskDescription=description, taskTag=["exp3.1" + args.experiment, "v1.2", "6000"],
            goodsId=record["goods_id"], goodsBackId=record["goods_back_id"], gpuNum=1,
            imageId=record["image_id"], imageVersion=record["image_version"], personalDataPath=""),
            taskCodeInfo=code, runtimeReminderConfig=dict(enableRuntimeReminder=False, reminderDurations=[]))
        result = create_with_file("task", payload)
        record["task_id"] = result["taskId"] if isinstance(result, dict) else result
        record.update(status="0", start_time=None)
        (HERE / ("task_config_" + args.experiment + ".json")).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        save_manifest()
        print("CREATED", record["experiment"], record["task_id"], "commit", expected)
        return

    info = invoke("task", "info", "--task-id", record["task_id"])
    base, code = info["taskBaseInfo"], info["taskCodeInfo"]
    if str(base["userId"]) != str(who["userId"]):
        raise RuntimeError("task ownership mismatch")
    script = code.get("startScript", "")
    if not ("--task=" + TASK in script and "--max_iterations=6000" in script and "--num_envs=4096" in script):
        raise RuntimeError("unexpected task start parameters")
    if args.experiment == "a" and ("--ckpt_path=" not in script or code.get("resumeFromTaskId") != "TASK_20261019_085"):
        raise RuntimeError("085 continuation metadata mismatch")
    if args.experiment == "b" and ("--ckpt_path=" in script or code.get("checkPointFilePath")):
        raise RuntimeError("fresh task unexpectedly has a checkpoint")
    if args.action == "clear-storage":
        if str(base["taskStatus"]) != "0":
            raise RuntimeError("only repair personal storage on an unstarted draft")
        # The edit API replaces complete records; preserve all fields and IDs.
        payload = dict(taskBaseInfo=dict(base, personalDataPath=""),
                       taskCodeInfo=dict(code, taskId=record["task_id"]),
                       runtimeReminderConfig=info.get("runtimeReminderConfig"))
        path = HERE / (".flux-edit-storage-" + args.experiment + ".json")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        invoke("task", "edit", "--file", relative, dry_run=True)
        invoke("task", "edit", "--file", relative)
        path.unlink()
        stored = HERE / ("task_config_" + args.experiment + ".json")
        config = read_json(stored)
        config["taskBaseInfo"]["personalDataPath"] = ""
        stored.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("PERSONAL_STORAGE_DISABLED", record["task_id"])
        return
    if args.action == "run":
        if str(base["taskStatus"]) == "0":
            invoke("task", "run", "--task-id", record["task_id"], dry_run=True)
            invoke("task", "run", "--task-id", record["task_id"])
            base = invoke("task", "info", "--task-id", record["task_id"])["taskBaseInfo"]
        print("RUN_RESULT", record["task_id"], "status", base["taskStatus"])
    else:
        log = invoke("task", "logs", "--task-id", record["task_id"], "--raw", "--no-request-log", raw=True)
        (HERE / ("startup_" + args.experiment + ".log")).write_text(log, encoding="utf-8")
        clean = re.sub(r"\x1b\[[0-9;]*m", "", log)
        iterations = list(re.finditer(r"Learning iteration (\d+)/(\d+)", clean))
        if iterations:
            last = iterations[-1]
            record.update(first_iteration=int(iterations[0][1]), latest_iteration=int(last[1]),
                          target_iteration=int(last[2]), checkpoint_loaded="Loading model from ckpt_path:" in clean)
            summary = clean[last.start():]
            names = {"mean_reward": "Mean reward", "mean_episode_length": "Mean episode length", "eta_seconds": "ETA"}
            for key, label in names.items():
                match = re.search(re.escape(label) + r":\s*([-+\d.eE]+)", summary)
                if match:
                    record[key] = float(match[1])
            wanted = ("Learning iteration", "Computation:", "Mean reward:", "Mean episode length:",
                      "AMP raw/used", "Task/style contribution:", "AMP direction gate:", "ETA:")
            print("TRAINING_LOG\n" + "\n".join(line.strip() for line in summary.splitlines()
                                                 if any(label in line for label in wanted)))
            print("INITIALIZATION", "checkpoint_loaded=" + str(record["checkpoint_loaded"]),
                  "first_iteration=" + str(record["first_iteration"]))
        else:
            print("LOG_TAIL\n" + "\n".join(line[:250] for line in clean.splitlines()[-10:]))
        errors = [line for line in clean.splitlines() if any(word in line for word in ("Traceback", "KeyError", "RuntimeError", "ValueError"))]
        record["startup_error_lines"] = errors
        record["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if errors:
            print("ERRORS", json.dumps(errors, ensure_ascii=False))
    record["status"] = base["taskStatus"]
    record["start_time"] = base.get("startTime")
    save_manifest()
    print("TASK_STATE", json.dumps({k: base.get(k) for k in (
        "taskId", "taskName", "taskStatus", "userId", "startTime", "endTime", "goodsId", "imageVersion")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
