"""Offline geometry diagnosis of the standing target used in the zero-residual replay."""
import ast
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    run = json.loads((HERE / "zero_residual_latest.json").read_text())["run"]
    output = HERE / ("zero_residual_" + run)
    source = ROOT / "humanoid/envs/x1/x1_dh_stand_config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "X1DHStandCfg")
    init = next(node for node in cfg.body if isinstance(node, ast.ClassDef) and node.name == "init_state")
    defaults = ast.literal_eval(next(node.value for node in init.body if isinstance(node, ast.Assign)
                                    and node.targets[0].id == "default_joint_angles"))
    lib = torch.load(ROOT / "resources/motions/processed/ref_lib_v1_2_exp33_phase.pt", map_location="cpu", weights_only=False)
    record = lib["walk_forward"]
    geometry = record["exp33_geometry"]
    model = mujoco.MjModel.from_xml_path(str(ROOT / "resources/robots/x1/mjcf/x1_29dof_scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    data.qpos[2] = geometry["stand_root_height_m"]
    data.qpos[3] = 1
    for name in record["dof_names"]:
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = defaults[name]
    mujoco.mj_forward(model, data)
    com = data.subtree_com[model.body("base_link").id].copy()
    feet, contact = {}, []
    for i, name in enumerate(geometry["foot_names"]):
        index = model.body(name).id
        matrix = data.xmat[index].reshape(3, 3)
        vertices = geometry["sole_vertices_local"][i].numpy() @ matrix.T + data.xpos[index]
        normal = matrix @ geometry["sole_normals_local"][i].numpy()
        patch = vertices[vertices[:, 2] < vertices[:, 2].min() + .002]
        contact.append(patch[:, :2])
        feet[name] = dict(normal=normal.tolist(), tilt_deg=float(np.degrees(np.arccos(np.clip(normal[2], -1, 1)))),
                          sole_z_range_m=[float(vertices[:, 2].min()), float(vertices[:, 2].max())],
                          projected_contact_x_range_m=[float(patch[:, 0].min()), float(patch[:, 0].max())],
                          foot_origin=data.xpos[index].tolist())
    points = np.concatenate(contact)
    hull = ConvexHull(points)
    margin = -(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]).max()
    result = dict(method="MuJoCo offline FK with original model inertias; nominal Isaac base COM override may differ",
                  standing_com_world_m=com.tolist(), feet=feet,
                  approximate_contact_patch_height_tolerance_m=.002,
                  projected_com_inside_approximate_patch=bool(margin >= 0),
                  projected_com_margin_m=float(margin), contact_polygon_xy_m=points[hull.vertices].tolist())
    (output / "standing_geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
