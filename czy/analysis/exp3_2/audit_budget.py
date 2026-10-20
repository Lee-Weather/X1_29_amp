"""Re-score existing replay CSVs; this is offline analysis, not a new policy evaluation."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def huber(value):
    magnitude = np.abs(value)
    return np.where(magnitude <= 1, .5 * value ** 2, magnitude - .5)


def main():
    reference = torch.load(ROOT / "resources/motions/processed/ref_lib_v1_2_phase.pt", map_location="cpu", weights_only=False)
    limits = {j.attrib["name"]: (float(j.find("limit").attrib["lower"]), float(j.find("limit").attrib["upper"]))
              for j in ET.parse(ROOT / "resources/robots/x1/urdf/X1_29DOF_physically_mirrored.urdf").findall("joint")
              if j.find("limit") is not None and "lower" in j.find("limit").attrib}
    result = {"meaning": "Fixed old trajectories, exp3.2 reward re-score; not new-policy performance.",
              "walking_terminal_fused_reward": -15, "gamma": .994, "cases": {}, "reference": {}}
    for segment, record in reference.items():
        conflicts = []
        physical = []
        for index, name in enumerate(record["dof_names"]):
            lo, hi = limits[name]
            q = record["dof_pos"][:, index]
            if float(q.min()) < lo - 1e-5 or float(q.max()) > hi + 1e-5:
                physical.append(name)
            if float(q.min()) < lo + .02 * (hi - lo) - 1e-5 or float(q.max()) > hi - .02 * (hi - lo) + 1e-5:
                conflicts.append(name)
        result["reference"][segment] = dict(physical_limit_conflicts=physical, fixed_2percent_margin_conflicts=conflicts,
            root_height_min_m=float(record["root_pos"][:, 2].min()), root_height_max_m=float(record["root_pos"][:, 2].max()))
    for letter in ("a", "b"):
        for mode in ("nominal", "randomized"):
            path = ROOT / ("czy/data/exp3.1" + letter + "/replay_20261020_164054_" + mode + "/isaac_diag.csv")
            df = pd.read_csv(path)
            names = [c[len("pos_"):] for c in df if c.startswith("pos_") and not c.startswith("pos_des_")]
            groups = ([n for n in names if any(t in n for t in ("hip_roll", "hip_yaw", "ankle_roll"))],
                      [n for n in names if any(t in n for t in ("hip_pitch", "knee_pitch", "ankle_pitch"))],
                      [n for n in names if not any(t in n for t in ("hip_", "knee_", "ankle_"))])
            pose = np.zeros(len(df))
            for group, tolerance, weight in zip(groups, (.2, .35, .4), (2, 1, .2)):
                error = np.column_stack([(df["pos_" + n] - df["ref_pos_" + n]).to_numpy() for n in group])
                pose += weight * huber(error / tolerance).mean(axis=1)
            angles = np.column_stack([df["base_euler_" + axis] - df["ref_" + name]
                                      for axis, name in (("x", "roll"), ("y", "pitch"))])
            angles = np.arctan2(np.sin(angles), np.cos(angles))
            orientation = 2 * huber(np.maximum(np.abs(angles) - .1, 0) / .2).mean(axis=1)
            height = 2 * huber(np.maximum(.55 - df.base_height.to_numpy(), 0) / .1)
            heading = .5 * huber(np.arctan2(np.sin(df.yaw_from_episode_start), np.cos(df.yaw_from_episode_start)) / .35)
            walk = df.command_applied.abs() > .05
            rate = df.task_reward.to_numpy() / .01 + df.reward_signed_speed_error / .01 - .75 * df.reward_foot_height / .01
            rate -= orientation + height + heading
            rate[walk] -= df.loc[walk, "reward_ref_joint_pos"] / .01 + pose[walk]
            rate -= 2900 * df.done * (1 - df.timeout)
            alive = walk & (df.done == 0)
            negative_per_step = np.maximum(-.5 * rate[alive].to_numpy() * .01, 0)
            result["cases"][letter + "_" + mode] = dict(
                walk_task_rate_mean=float(rate[alive].mean()),
                p95_negative_fused_rate=float(np.quantile(negative_per_step, .95) / .01),
                p95_constant_loss_discounted_bound=float(np.quantile(negative_per_step, .95) / (1 - .994)),
                joint_group_sizes=[len(g) for g in groups], stages=[])
            for stage, part in df.groupby("active_stage"):
                mask = (df.active_stage == stage) & (df.time_s >= part.time_s.min() + 1) & (df.done == 0) & np.isclose(df.command_applied, part.command_requested.iloc[0])
                if mask.any():
                    result["cases"][letter + "_" + mode]["stages"].append(dict(
                        command=float(part.command_requested.iloc[0]), rows=int(mask.sum()),
                        task_rate_mean=float(rate[mask].mean()), pose_cost_rate=float(pose[mask].mean())))
    (OUT / "budget_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, case in result["cases"].items():
        print(name, "p95_discounted_negative_tail", round(case["p95_constant_loss_discounted_bound"], 3),
              "walking task rates", [(s["command"], round(s["task_rate_mean"], 3)) for s in case["stages"]])


if __name__ == "__main__":
    main()
