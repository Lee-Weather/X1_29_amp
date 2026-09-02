# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class X1DHStandCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 66      #all histroy obs num
        short_frame_stack = 5   #short history step
        c_frame_stack = 3  #all histroy privileged obs num
        num_single_obs = 98    # 29DOF: 5(cmd) + 3*29(q/dq/action) + 6(ang_vel/euler)
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 141  # 29DOF: 5 + 4*29(q/dq/action/diff) + 20(其余特权量)
        single_linvel_index = 121  # 29DOF: 5 + 4*29（base_lin_vel 起始列）
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 29
        num_envs = 4096
        episode_length_s = 24 #episode length in seconds
        use_ref_actions = False
        num_commands = 5 # sin_pos cos_pos vx vy vz

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85


    class asset(LeggedRobotCfg.asset):
        # exp2：29DOF 全身 URDF（Isaac Gym dof 序：左腿0-5/腰6-8/左臂9-15/右臂16-22/右腿23-28，env 按名索引腿部）
        # 右踝 pitch 轴 (0 0 -1)@rpy(π,0,0)，与左踝世界轴反平行（原版 physically_mirrored 约定，exp0 验证）
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/urdf/X1_29DOF_physically_mirrored.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/mjcf/xyber_x1_flat.xml'

        name = "x1"
        foot_name = "ankle_roll"
        knee_name = "knee_pitch"

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
        platform = 3.
        terrain_dict = {"flat": 0.3, 
                        "rough flat": 0.2,
                        "slope up": 0.2,
                        "slope down": 0.2, 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0, 
                        "stairs up": 0., 
                        "stairs down": 0.,
                        "discrete": 0.1, 
                        "wave": 0.0,}
        terrain_proportions = list(terrain_dict.values())

        rough_flat_range = [0.005, 0.01]  # meter
        slope_range = [0, 0.1]   # rad
        rough_slope_range = [0.005, 0.02]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.01, 0.1]
        discrete_height_range = [0.0, 0.01]
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5    # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02
            dof_vel = 1.5 
            ang_vel = 0.2   
            lin_vel = 0.1   
            quat = 0.1
            gravity = 0.05
            height_measurements = 0.1


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.7]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # ---- 腿部（沿用 exp1.5；右踝 pitch 反号：29DOF PM 轴与左踝世界轴反平行）----
            'left_hip_pitch_joint': 0.4,
            'left_hip_roll_joint': 0.05,
            'left_hip_yaw_joint': -0.31,
            'left_knee_pitch_joint': 0.49,
            'left_ankle_pitch_joint': -0.21,
            'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.4,
            'right_hip_roll_joint': -0.05,
            'right_hip_yaw_joint': 0.31,
            'right_knee_pitch_joint': 0.49,
            'right_ankle_pitch_joint': 0.21,
            'right_ankle_roll_joint': 0.0,
            # ---- 上半身（amp CSV 均值左右对称化；符号约定回放目视校验）----
            'lumbar_yaw_joint': 0.0,
            'lumbar_roll_joint': 0.0,
            'lumbar_pitch_joint': 0.03,
            'left_shoulder_pitch_joint': 0.03,  'right_shoulder_pitch_joint': 0.03,
            'left_shoulder_roll_joint': -0.06,  'right_shoulder_roll_joint': -0.06,
            'left_shoulder_yaw_joint': 0.18,    'right_shoulder_yaw_joint': 0.18,
            'left_elbow_pitch_joint': 0.34,     'right_elbow_pitch_joint': 0.34,
            'left_elbow_yaw_joint': 0.0,        'right_elbow_yaw_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,      'right_wrist_pitch_joint': 0.0,
            'left_wrist_roll_joint': 0.0,       'right_wrist_roll_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {'hip_pitch_joint': 30, 'hip_roll_joint': 40,'hip_yaw_joint': 35,
                     'knee_pitch_joint': 100, 'ankle_pitch_joint': 35, 'ankle_roll_joint': 35,
                     # exp2 29DOF 上半身（无真机辨识，量级建议值；缺键=被动悬摆）
                     'lumbar_yaw_joint': 60, 'lumbar_roll_joint': 60, 'lumbar_pitch_joint': 80,
                     'shoulder_pitch_joint': 40, 'shoulder_roll_joint': 40, 'shoulder_yaw_joint': 40,
                     'elbow_pitch_joint': 30, 'elbow_yaw_joint': 30,
                     'wrist_pitch_joint': 8, 'wrist_roll_joint': 8}
        damping = {'hip_pitch_joint': 3, 'hip_roll_joint': 3.0,'hip_yaw_joint': 4,
                   'knee_pitch_joint': 8, 'ankle_pitch_joint': 1.5, 'ankle_roll_joint': 1.5,
                   # exp2 29DOF 上半身
                   'lumbar_yaw_joint': 4, 'lumbar_roll_joint': 4, 'lumbar_pitch_joint': 5,
                   'shoulder_pitch_joint': 2, 'shoulder_roll_joint': 2, 'shoulder_yaw_joint': 2,
                   'elbow_pitch_joint': 1.5, 'elbow_yaw_joint': 1.5,
                   'wrist_pitch_joint': 0.5, 'wrist_roll_joint': 0.5}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 200 Hz 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
     
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.3]
        restitution_range = [0.0, 0.4]

        # push
        push_robots = True
        push_interval_s = 4 # every this second, push robot
        update_step = 2000 * 24 # after this count, increase push_duration index
        push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.25] # increase push duration during training
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.2

        randomize_base_mass = True
        added_mass_range = [-3, 3] # base mass rand range, base mass is all fix link sum mass

        randomize_com = True
        com_displacement_range = [[-0.05, 0.05],
                                  [-0.05, 0.05],
                                  [-0.05, 0.05]]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]    # Factor

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        added_link_mass_range = [0.9, 1.1]

        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035] # Offset to add to the motor angles
        
        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]

        randomize_joint_armature = True
        randomize_joint_armature_each_joint = True  # 必须开启，否则逐关节范围不生效
        joint_armature_range = [0.0001, 0.05]     # 统一回退值（each_joint=False 时使用）
        # exp2（29DOF）：Isaac Gym 实际 dof 顺序（字母序 DFS，冒烟实测打印的 [DOF] 表）：
        # 左腿(0-5)→腰(6-8)→左臂(9-15)→右臂(16-22)→右腿(23-28)，joint_N 对应 dof N-1
        # 腿部辨识值沿用 exp1.5（29DOF 腿部与 12DOF physically_mirrored 同源）：
        # armature = 真机辨识J − M_ii：髋Pitch 0.196（左右对称化）、髋Yaw 0.0148/0.0060、膝 0.250/0.247
        joint_1_armature_range = [0.09, 0.23]     # dof0 L hip_pitch (id 0.196)：M_ii=0.271，左右对称化（exp1.1 教训）
        joint_2_armature_range = [0.0001, 0.05]   # dof1 L hip_roll (id unreliable)
        joint_3_armature_range = [0.003, 0.018]   # dof2 L hip_yaw (id 0.0148)：M_ii=0.0309
        joint_4_armature_range = [0.18, 0.32]     # dof3 L knee (id 0.250) CORE：M_ii=0.1127
        joint_5_armature_range = [0.003, 0.04]    # dof4 L ankle_pitch: exp1.4 覆盖随机化（无辨识数据，不猜中心）
        joint_6_armature_range = [0.003, 0.04]    # dof5 L ankle_roll: exp1.4 同上（真机抖动关节，覆盖最关键）
        joint_7_armature_range = [0.003, 0.04]    # dof6 lumbar_yaw（无辨识，覆盖随机化）
        joint_8_armature_range = [0.003, 0.04]    # dof7 lumbar_roll
        joint_9_armature_range = [0.003, 0.04]    # dof8 lumbar_pitch
        joint_10_armature_range = [0.003, 0.04]   # dof9 L shoulder_pitch
        joint_11_armature_range = [0.003, 0.04]   # dof10 L shoulder_roll
        joint_12_armature_range = [0.003, 0.04]   # dof11 L shoulder_yaw
        joint_13_armature_range = [0.003, 0.04]   # dof12 L elbow_pitch
        joint_14_armature_range = [0.003, 0.04]   # dof13 L elbow_yaw
        joint_15_armature_range = [0.003, 0.04]   # dof14 L wrist_pitch
        joint_16_armature_range = [0.003, 0.04]   # dof15 L wrist_roll
        joint_17_armature_range = [0.003, 0.04]   # dof16 R shoulder_pitch
        joint_18_armature_range = [0.003, 0.04]   # dof17 R shoulder_roll
        joint_19_armature_range = [0.003, 0.04]   # dof18 R shoulder_yaw
        joint_20_armature_range = [0.003, 0.04]   # dof19 R elbow_pitch
        joint_21_armature_range = [0.003, 0.04]   # dof20 R elbow_yaw
        joint_22_armature_range = [0.003, 0.04]   # dof21 R wrist_pitch
        joint_23_armature_range = [0.003, 0.04]   # dof22 R wrist_roll
        joint_24_armature_range = [0.09, 0.23]    # dof23 R hip_pitch (id 0.128, symmetrized)：与左侧一致
        joint_25_armature_range = [0.0001, 0.05]  # dof24 R hip_roll (id unreliable)
        joint_26_armature_range = [0.003, 0.018]  # dof25 R hip_yaw (id 0.0060)：与左侧一致
        joint_27_armature_range = [0.18, 0.32]    # dof26 R knee (id 0.246) CORE：与左侧一致
        joint_28_armature_range = [0.003, 0.04]   # dof27 R ankle_pitch: exp1.4 与左侧一致（覆盖随机化）
        joint_29_armature_range = [0.003, 0.04]   # dof28 R ankle_roll: exp1.4 与左侧一致（真机抖动关节）

        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 40]
        
        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 40]
        
        add_dof_pos_vel_lag = False
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False
        dof_pos_lag_timesteps_range = [7, 25]
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False
        dof_vel_lag_timesteps_range = [7, 25]
        
        add_imu_lag = False
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [1, 10]
        
        randomize_coulomb_friction = True
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.05, 0.1]
        
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 25.  # time before command are changed[s]
        gait = ["walk_omnidirectional","stand","walk_omnidirectional"] # gait type during training
        # proportion during whole life time
        gait_time_range = {"walk_sagittal": [2,6],
                           "walk_lateral": [2,6],
                           "rotate": [2,3],
                           "stand": [2,3],
                           "walk_omnidirectional": [4,6]}

        heading_command = False  # if true: compute ang vel command from heading error
        stand_com_threshold = 0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True # use stand_com_threshold or not

        class ranges:
            lin_vel_x = [-0.4, 1.2] # min max [m/s] 
            lin_vel_y = [-0.4, 0.4]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.61
        foot_min_dist = 0.2
        foot_max_dist = 1.0

        # final_swing_joint_pos = final_swing_joint_delta_pos + default_pos
        # 12 元素按腿部顺序（env 的 leg_dof_names 解析索引；上半身不参与步态摆动，保持默认位姿）
        # 索引 10 = right_ankle_pitch：29DOF PM 轴 (0 0 -1) 与左踝世界轴反平行，摆幅反号（exp0 验证）
        final_swing_joint_delta_pos = [0.25, 0.05, -0.11, 0.35, -0.16, 0.0, -0.25, -0.05, 0.11, 0.35, 0.16, 0.0]
        target_feet_height = 0.03 
        target_feet_height_max = 0.06
        feet_to_ankle_distance = 0.041
        cycle_time = 0.7
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(-error*sigma)
        tracking_sigma = 5
        max_contact_force = 700  # forces above this value are penalized
        
        class scales:
            ref_joint_pos = 2.2
            feet_clearance = 1.
            feet_contact_number = 2.0
            # gait
            feet_air_time = 1.2
            foot_slip = -0.1
            feet_distance = 0.2   # exp0.3: 0.3→0.2 回退（exp0.2 证实带来 vx 过冲副作用，收益不明显）
            knee_distance = 0.2
            feet_contact_number = 2.4  # exp1.3: 2.0→2.4 强化左右步节拍对称（治偏航离散累积）
            # lateral
            lat_vel = -2.0        # exp1.1: -1.2->-2.0 加压（exp1 净漂 -0.10/-0.12 未压住）
            yaw_drift = -0.8      # exp1.3: 新增偏航角速度线性惩罚（无转向指令时生效）
            # contact 
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 2.2  # exp1.3: 1.8→2.2 提升跟踪优先级
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 1.0
            orientation = 1.2     # exp1.1: 1.0->1.2 微调（-y 漂伴随左倾，roll 姿态保持协同纠偏）
            feet_rotation = 0.3
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.008  # exp1.4: -0.002→-0.008 压制真机右踝roll 5.5~6.5Hz 输出振荡（二阶差分对高频灵敏度∝f⁴）
            torques = -8e-9
            dof_vel = -2e-8
            dof_acc = -1e-7
            collision = -1.
            stand_still = 2.5
            # limits
            dof_vel_limits = -1
            dof_pos_limits = -10.
            dof_torque_limits = -0.1

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


class X1DHStandCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'DHOnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims=[256, 128, 64]
        
        #for long_history cnn only
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64   #long history output dim
        in_channels = X1DHStandCfg.env.frame_stack

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
        if X1DHStandCfg.terrain.measure_heights:
            lin_vel_idx = (X1DHStandCfg.env.single_num_privileged_obs + X1DHStandCfg.terrain.num_height) * (X1DHStandCfg.env.c_frame_stack - 1) + X1DHStandCfg.env.single_linvel_index
        else:
            lin_vel_idx = X1DHStandCfg.env.single_num_privileged_obs * (X1DHStandCfg.env.c_frame_stack - 1) + X1DHStandCfg.env.single_linvel_index

    class runner:
        policy_class_name = 'ActorCriticDH'
        algorithm_class_name = 'DHPPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 6000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'x1_dh_stand'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
