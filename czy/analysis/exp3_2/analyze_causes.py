"""Offline evidence for the exp3.2 proposal; does not run simulation or training."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def summarize(frame):
    limits = {}
    tree = ET.parse(ROOT / "resources/robots/x1/urdf/X1_29DOF_physically_mirrored.urdf")
    for joint in tree.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib:
            limits[joint.attrib["name"]] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    violations = []
    for name, (lower, upper) in limits.items():
        column = "pos_des_raw_" + name
        if column not in frame:
            continue
        outside = (lower - frame[column]).clip(lower=0) + (frame[column] - upper).clip(lower=0)
        if outside.max() > 1e-5:
            violations.append({"joint": name, "rows": int((outside > 1e-5).sum()),
                               "max_outside_rad": float(outside.max())})
    # Existing components already include control dt=0.01 and configured scales.
    counterfactual = (frame.task_reward + frame.reward_signed_speed_error
                      - 0.75 * frame.reward_foot_height)
    result = {
        "rows": len(frame), "time_start_s": float(frame.time_s.min()),
        "time_end_s": float(frame.time_s.max()),
        "vx_mean_mps": float(frame.base_vel_x.mean()),
        "task_rate_before_amp_fusion": float(frame.task_reward.mean() / .01),
        "reward_component_rates": {c: float(frame[c].mean() / .01)
                                   for c in frame if c.startswith("reward_")},
        "double_contact_fraction_force_gt_5N": float(
            ((frame.foot_force_l > 5) & (frame.foot_force_r > 5)).mean()),
        "torque_clip_joint_substep_events": int(frame.torque_clip_step.sum()),
        "raw_target_outside_urdf": sorted(violations, key=lambda x: -x["max_outside_rad"]),
        "fixed_trajectory_counterfactual_task_rate_speed8_footheight025":
            float(counterfactual.mean() / .01),
    }
    for c in ("amp_direction_gate", "amp_style_contribution", "amp_task_contribution"):
        if c in frame:
            result[c + "_mean"] = float(frame[c].mean())
    return result


def main():
    data = {}
    for label in ("a", "b"):
        data[label] = pd.read_csv(ROOT / (
            "czy/data/exp3.1" + label + "/replay_20261020_164054_nominal/isaac_diag.csv"))
    windows = {"a_forward04": ("a", 6, 9.99), "a_forward06": ("a", 11, 14.99),
               "b_first_startup": ("b", 5.05, 6.14)}
    result = {"source": "exp3.1 a=11000 / b=4400, nominal replay, seed=123145",
              "notes": ["Rates are task reward per second before the 0.5 walking fusion.",
                        "Counterfactual reweights the fixed recorded trajectory; it is not a policy result.",
                        "pos_des_raw is the lagged PD target before motor offset, not the measured angle.",
                        "Only the last physics-substep position target is saved per control step."]}
    for key, (label, start, end) in windows.items():
        result[key] = summarize(data[label][data[label].time_s.between(start, end)])
    b = data["b"]
    result["b_first_failure_approach"] = []
    for time in (4.99, 5.2, 5.5, 5.8, 6., 6.15):
        row = b.iloc[(b.time_s - time).abs().argmin()]
        result["b_first_failure_approach"].append({
            "time_s": float(row.time_s), "vx_mps": float(row.base_vel_x),
            "height_m": float(row.base_height), "pitch_deg": float(np.rad2deg(row.base_euler_y)),
            "torque_clip_joint_substep_events": int(row.torque_clip_step)})
    (OUT / "cause_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {k: value[k] for k in (
        "rows", "task_rate_before_amp_fusion", "double_contact_fraction_force_gt_5N",
        "fixed_trajectory_counterfactual_task_rate_speed8_footheight025")}
        for key, value in result.items() if key in windows}, indent=2))


if __name__ == "__main__":
    main()
