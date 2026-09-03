#!/usr/bin/env python
"""Convert X1 URDF -> MJCF scene used by play_motion_mujoco.py.

Directly loads the workspace X1_29DOF_physically_mirrored.urdf with MuJoCo
(MjModel.from_xml_path auto-adds a free joint base_link_free_joint, nq=7+29=36),
then dumps the compiled body with mj_saveLastXML and wraps it into a scene that
adds a floor, lights, and the robot.

Outputs:
  resources/robots/x1/mjcf/x1_29dof_raw.xml    compiled bodies/geoms only
  resources/robots/x1/mjcf/x1_29dof_scene.xml  full scene (robot + floor + lights)

Usage:
  python scripts/tools/urdf2mjcf.py
"""
import os

import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF = os.path.join(ROOT, "resources/robots/x1/urdf/X1_29DOF_physically_mirrored.urdf")
MJCF_DIR = os.path.join(ROOT, "resources/robots/x1/mjcf")
RAW_XML = os.path.join(MJCF_DIR, "x1_29dof_raw.xml")
SCENE_XML = os.path.join(MJCF_DIR, "x1_29dof_scene.xml")


def main():
    model = mujoco.MjModel.from_xml_path(URDF)  # auto-injects base_link_free_joint

    n_hinge = sum(1 for i in range(model.njnt)
                  if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE)
    n_free = sum(1 for i in range(model.njnt)
                 if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    print(f"URDF load OK: nq={model.nq} nbody={model.nbody} nmesh={model.nmesh}, "
          f"hinges={n_hinge} free={n_free}")
    assert n_hinge == 29 and n_free == 1 and model.nq == 36

    os.makedirs(MJCF_DIR, exist_ok=True)
    mujoco.mj_saveLastXML(RAW_XML, model)

    scene = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="x1_29dof_scene">
  <include file="x1_29dof_raw.xml"/>
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.85 0.85 0.85" rgb2="0.75 0.75 0.75"/>
    <material name="gridmat" texrepeat="8 8" reflectance="0.1" texuniform="true"
              texture="grid"/>
  </asset>
  <worldbody>
    <light name="light0" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"
           pos="0 -3 4" dir="0 0 -1"/>
    <light name="light1" mode="targetbodycom" target="base_link"
           pos="-2 3 4" diffuse="0.7 0.7 0.7"/>
    <geom name="floor" type="plane" size="0 0 20" pos="0 0 0"
          material="gridmat" condim="3"/>
  </worldbody>
</mujoco>
"""
    with open(SCENE_XML, "w") as f:
        f.write(scene)

    m2 = mujoco.MjModel.from_xml_path(SCENE_XML)
    print(f"scene OK: nq={m2.nq} nbody={m2.nbody} ngeom={m2.ngeom} "
          f"nmesh={m2.nmesh} njnt={m2.njnt}")
    print(f"wrote:\n  {RAW_XML}\n  {SCENE_XML}")


if __name__ == "__main__":
    main()