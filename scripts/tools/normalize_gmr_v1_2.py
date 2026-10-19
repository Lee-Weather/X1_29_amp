#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Normalize slow_walk_amp_v1.2 clips to the named GMR pkl schema.

The source clips are 50 Hz and contain 29 unnamed joint columns. They were
exported in the MJCF hinge order. This adapter adds explicit names and FK body
positions so existing GMR readers can inspect/remap them safely. It does not
resample or change joint signs.
"""
import argparse
import json
import os
import pickle

import mujoco
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SOURCE_DIR = os.path.join(ROOT, "czy/gmr/slow_walk_amp_v1.2")
MODEL_PATH = os.path.join(ROOT, "resources/robots/x1/mjcf/x1_29dof_scene.xml")
OUTPUT_DIR = os.path.join(SOURCE_DIR, "normalized")
TRANSITION_CLIPS = {"F1_B1_stand_to_walk", "F1_B2_walk_to_stand"}
FREEJOINT_NAME = "base_link_free_joint"


def model_joint_order(model):
    names = [model.joint(i).name for i in range(model.njnt)
             if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if len(names) != 29 or len(set(names)) != len(names):
        raise ValueError("expected 29 uniquely named hinge joints in the X1 MJCF")
    return names


def normalize_clip(path, model, joint_names, body_names):
    with open(path, "rb") as stream:
        source = pickle.load(stream)

    fps = float(source["fps"])
    root_pos = np.asarray(source["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(source["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(source["dof_pos"], dtype=np.float64)
    n = dof_pos.shape[0]
    if fps != 50.0:
        raise ValueError("{}: expected 50 Hz, got {}".format(path, fps))
    if dof_pos.shape != (n, len(joint_names)):
        raise ValueError("{}: unexpected dof_pos shape {}".format(path, dof_pos.shape))
    if root_pos.shape != (n, 3) or root_rot_xyzw.shape != (n, 4):
        raise ValueError("{}: root array shapes do not match dof_pos".format(path))
    if not (np.isfinite(dof_pos).all() and np.isfinite(root_pos).all()
            and np.isfinite(root_rot_xyzw).all()):
        raise ValueError("{}: contains non-finite values".format(path))

    # The source stores XYZW; MuJoCo free-joint qpos stores WXYZ.
    quat_norm = np.linalg.norm(root_rot_xyzw, axis=1, keepdims=True)
    if np.any(quat_norm < 1e-8):
        raise ValueError("{}: contains a zero quaternion".format(path))
    quat_xyzw = root_rot_xyzw / quat_norm
    # Preserve quaternion sign continuity for downstream angular-velocity extraction.
    for i in range(1, n):
        if np.dot(quat_xyzw[i - 1], quat_xyzw[i]) < 0:
            quat_xyzw[i] *= -1
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    qpos = np.zeros((n, model.nq), dtype=np.float64)
    qpos[:, :3] = root_pos
    qpos[:, 3:7] = quat_wxyz
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        qpos[:, model.jnt_qposadr[i]] = dof_pos[:, joint_names.index(model.joint(i).name)]

    data = mujoco.MjData(model)
    body_positions = np.empty((n, len(body_names), 3), dtype=np.float32)
    for frame in range(n):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        body_positions[frame] = np.asarray(
            [data.xpos[model.body(name).id] for name in body_names], dtype=np.float32)

    output = {
        "fps": fps,
        "root_pos": root_pos.astype(np.float32),
        "root_rot": quat_xyzw.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "dof_names": list(joint_names),
        "body_positions": body_positions,
        "body_names": list(body_names),
        "source": os.path.basename(path),
        "source_format": "slow_walk_amp_v1.2_mjcf_hinge_order",
        "joint_order_verified_by": os.path.relpath(MODEL_PATH, ROOT),
        "joint_signs_changed": False,
        "motion_role": "transition" if os.path.splitext(os.path.basename(path))[0]
        in TRANSITION_CLIPS else "continuous_gait",
        "periodic": os.path.splitext(os.path.basename(path))[0]
        not in TRANSITION_CLIPS,
    }
    # A quick FK sanity check: output body positions must be finite and include the base.
    if "base_link" not in body_names or not np.isfinite(body_positions).all():
        raise ValueError("{}: FK output sanity check failed".format(path))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    joint_names = model_joint_order(model)
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = {
        "format": "named GMR pkl compatible",
        "model": os.path.relpath(MODEL_PATH, ROOT),
        "source_dir": os.path.relpath(SOURCE_DIR, ROOT),
        "fps": 50.0,
        "source_joint_order": "MJCF hinge order (29 joints)",
        "source_quaternion_order": "xyzw",
        "normalized_quaternion_order": "xyzw",
        "root_rot_sign_continuity": True,
        "joint_signs_changed": False,
        "dof_names": joint_names,
        "body_names": body_names,
        "clips": {},
    }

    for filename in sorted(os.listdir(SOURCE_DIR)):
        if not (filename.startswith("F1_B") and filename.endswith(".pkl")):
            continue
        source_path = os.path.join(SOURCE_DIR, filename)
        motion = normalize_clip(source_path, model, joint_names, body_names)
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "wb") as stream:
            pickle.dump(motion, stream, protocol=pickle.HIGHEST_PROTOCOL)
        manifest["clips"][os.path.splitext(filename)[0]] = {
            "frames": int(motion["dof_pos"].shape[0]),
            "duration_s": round(motion["dof_pos"].shape[0] / motion["fps"], 3),
            "motion_role": motion["motion_role"],
            "periodic": motion["periodic"],
            "output": filename,
        }
        print("[OK] {}: {} frames @ {} Hz, {}, FK bodies {}".format(
            filename, motion["dof_pos"].shape[0], motion["fps"],
            motion["motion_role"], len(body_names)))

    manifest_path = os.path.join(args.output_dir, "format_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print("normalized data -> {}".format(args.output_dir))


if __name__ == "__main__":
    main()
