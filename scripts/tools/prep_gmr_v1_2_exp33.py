#!/usr/bin/env python
"""Create separate exp3.3 libraries with collision-sole floor calibration.

The v1.2 joint trajectories, rotations, phase anchors and horizontal motion
are preserved. Root Z is adjusted using both actual foot collision meshes.
This is kinematic preprocessing, not evidence of dynamic tracking stability.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "resources/motions/processed"
MODEL = ROOT / "resources/robots/x1/mjcf/x1_29dof_scene.xml"
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
CLEARANCE = 0.001
SMOOTHING_M = 0.001


def default_pose(names):
    tree = ast.parse((ROOT / "humanoid/envs/x1/x1_dh_stand_config.py").read_text(encoding="utf-8"))
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "X1DHStandCfg")
    init = next(node for node in cfg.body if isinstance(node, ast.ClassDef) and node.name == "init_state")
    angles = ast.literal_eval(next(node.value for node in init.body if isinstance(node, ast.Assign)
                                  and node.targets[0].id == "default_joint_angles"))
    return np.array([angles[name] for name in names])


def set_pose(model, data, addresses, q, root, quaternion):
    data.qpos[:] = 0
    data.qpos[:3] = root
    data.qpos[3:7] = quaternion
    data.qpos[addresses] = q
    mujoco.mj_forward(model, data)


def foot_geometry(model, data):
    """Express collision hulls and neutral-pose upward normals in foot frames."""
    vertices, normals = [], []
    for name in FEET:
        body = model.body(name).id
        rotation = data.xmat[body].reshape(3, 3)
        parts = []
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError("expected collision mesh on " + name)
            mesh = model.geom_dataid[geom]
            mesh_vertices = model.mesh_vert[model.mesh_vertadr[mesh]:
                                            model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            world = mesh_vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            parts.append((world - data.xpos[body]) @ rotation)
        if not parts:
            raise ValueError("missing collision geometry on " + name)
        points = np.unique(np.concatenate(parts), axis=0)
        vertices.append(points[ConvexHull(points).vertices])
        normals.append(rotation.T @ np.array([0., 0., 1.]))
    return vertices, np.array(normals)


def sole_heights(model, data, vertices):
    return np.array([float((points @ data.xmat[model.body(name).id].reshape(3, 3).T
                           + data.xpos[model.body(name).id])[:, 2].min())
                     for name, points in zip(FEET, vertices)])


def load_library(path):
    # These are repository-owned tensor libraries, not downloaded checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    phase_source = OUT / "ref_lib_v1_2_phase.pt"
    names = load_library(phase_source)["walk_forward"]["dof_names"]
    addresses = np.array([model.jnt_qposadr[model.joint(name).id] for name in names])
    set_pose(model, data, addresses, np.zeros(len(names)), [0, 0, 0], [1, 0, 0, 0])
    vertices, normals = foot_geometry(model, data)
    stand = default_pose(names)
    set_pose(model, data, addresses, stand, [0, 0, 0], [1, 0, 0, 0])
    # A small initial clearance avoids starting inside the contact geometry.
    stand_height = float(0.005 - sole_heights(model, data, vertices).min())
    geometry = dict(foot_names=FEET, sole_normals_local=torch.tensor(normals, dtype=torch.float32),
                    sole_vertices_local=[torch.tensor(points, dtype=torch.float32) for points in vertices],
                    stand_dof_pos=torch.tensor(stand, dtype=torch.float32),
                    stand_root_height_m=stand_height)
    report = dict(method="smooth collision-sole lower envelope; no joint/rotation/XY changes",
                  model="resources/robots/x1/mjcf/x1_29dof_scene.xml",
                  model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                  clearance_m=CLEARANCE, smoothing_m=SMOOTHING_M,
                  stand_root_height_m=stand_height,
                  sole_normals_local=normals.tolist(),
                  collision_hull_vertices=[len(points) for points in vertices], files={})
    for source_name, output_name, periodic in (
            ("ref_lib_v1_2_phase.pt", "ref_lib_v1_2_exp33_phase.pt", True),
            ("amp_v1_2.pt", "amp_v1_2_exp33.pt", False)):
        source = OUT / source_name
        library = load_library(source)
        segment_report = {}
        for name, record in library.items():
            if list(record["dof_names"]) != list(names):
                raise ValueError("inconsistent joint order in " + name)
            q, root, quat = (record[field].numpy().astype(np.float64)
                             for field in ("dof_pos", "root_pos", "root_rot_wxyz"))
            heights = []
            for frame in range(len(q)):
                set_pose(model, data, addresses, q[frame], root[frame], quat[frame])
                heights.append(sole_heights(model, data, vertices))
            heights = np.array(heights)
            # Smooth minimum across both soles accounts for changing support.
            # It is no higher than the true minimum, including unexpected
            # penetration of the nominal swing foot.
            lower = -SMOOTHING_M * np.logaddexp(-heights[:, 0] / SMOOTHING_M,
                                               -heights[:, 1] / SMOOTHING_M)
            required = CLEARANCE - lower
            filtered = gaussian_filter1d(required, sigma=1.0, mode="wrap" if periodic else "nearest")
            # A smooth upper envelope preserves clearance after smoothing.
            lift = SMOOTHING_M * np.logaddexp(required / SMOOTHING_M, filtered / SMOOTHING_M)
            root[:, 2] += lift
            velocity = np.gradient(root, 1 / float(record["fps"]), axis=0)
            if periodic:
                velocity[:, 2] = (np.roll(root[:, 2], -1) - np.roll(root[:, 2], 1)) * float(record["fps"]) / 2
            record["root_pos"] = torch.tensor(root, dtype=torch.float32)
            record["root_lin_vel_world"] = torch.tensor(velocity, dtype=torch.float32)
            record["root_z_correction"] = torch.tensor(lift, dtype=torch.float32)
            record["sole_height"] = torch.tensor(heights + lift[:, None], dtype=torch.float32)
            record["exp33_geometry"] = copy.deepcopy(geometry)
            segment_report[name] = dict(frames=len(q), original_sole_min_m=float(heights.min()),
                                        corrected_sole_min_m=float((heights + lift[:, None]).min()),
                                        root_z_correction_min_m=float(lift.min()),
                                        root_z_correction_max_m=float(lift.max()),
                                        root_z_correction_mean_m=float(lift.mean()))
        torch.save(library, OUT / output_name)
        report["files"][output_name] = dict(
            source=source_name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            sha256=hashlib.sha256((OUT / output_name).read_bytes()).hexdigest(), segments=segment_report)
        print("Wrote", output_name, json.dumps(segment_report))
    (OUT / "v1_2_exp33_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
