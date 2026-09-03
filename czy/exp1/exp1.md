# 实验记录（X1_29_amp · 29DOF 全身控制）

> 本文件为 29DOF 项目实验记录（lab-notebook 规范）。
> 12DOF 旧项目（X1_29_re0）完整历史见 [exp1_12dof_legacy.md](exp1_12dof_legacy.md)（exp0~exp1.4，其结论与教训作为本系列先验）。

## 实验索引

| 编号 | 日期 | 摘要 | 状态 | Task ID | GM账号 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| exp0 | 2026-09-02 | 29DOF 全身控制基线：env 腿部按名索引 + config 29 维（obs 98/action 29/priv 141）+ 29DOF PM URDF + 上半身默认位姿锁定，从零 L4 训练至 5800 轮额度耗尽；回放摔倒（min_h 0.092m）+ 严重过冲（0.4 段 286%）+ 停不住，站立段完美 | ❌未达标（已测试） | TASK_20260902_185(停)→186 | limxmtcm6wjlso8ce4@emalupe.com（账号3，已耗尽） | model_5800.pt |

---

## 实验 exp0：29DOF 全身控制基线（修改编号从此重置）

### 1. 上一实验结果与教训

> 本系列首个实验。先验来自 12DOF 旧项目（[exp1_12dof_legacy.md](exp1_12dof_legacy.md) exp0~exp1.4）+ 本轮改造验证：
> - 12DOF 遗产：armature 真机辨识配置（膝 3.2×/髋Pitch 1.7× 惯量缺口）、lat_vel/yaw_drift 线性惩罚经验、exp1.4 遗留偏航/侧漂回退未解
> - 12DOF exp1.4 回放基准（本机固定 armature）：0.4/0.6 稳态跟踪 70.6%/69.3%
> - **改造验证**（改 config 前后分步执行）：
>   - 12DOF 回归（重构后、config 改前）：exp1.4 ckpt17996 回放 0.4/0.6 稳态 65.0%/65.9%、漂移同号、不摔倒 → env 重构无副作用
>   - 29DOF 冒烟：网络维度 actor 557→29 / critic 423 / estimator 490 全对，2 iters 无报错
>   - FK 镜像：默认位姿位置残差 0.00mm、姿态 0.00°（右踝 pitch=+0.21 判别严格：错误符号显 24.06°）
>
> **核心教训**：
> - Isaac Gym dof 顺序为字母序 DFS（左腿0-5/腰6-8/左臂9-15/右臂16-22/右腿23-28），**非 URDF 文档顺序**——armature 逐关节编号曾按文档序写错，被启动打印的 [DOF] 表当场纠正
> - F1 环境 humanoid 包 editable 指向需 `pip show` 确认（曾指向旧仓库导致冒烟跑错代码）

### 2. 本轮修改目标

- 目标1：29DOF 全身从零训练收敛——不摔倒、Mean reward ≥ 120、ep_len ≥ 2100
- 目标2：行走质量不低于 12DOF 基线量级——0.4/0.6 稳态跟踪 ≥ 65%（对齐 exp1.4 本机回归值）
- 目标3：上半身稳定锁定默认位姿（无甩臂、无自碰撞 terminate）
- 验收标准：目标 1+3 必须满足；目标 2 作为量级参考（首训重点在收敛与稳定）

### 3. 修改内容

### 修改一：env 腿部 dof 索引按关节名解析（兼容 12/29DOF）

| 位置 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| `_init_buffers` | 无 | `leg_dof_names` 12 关节名 → `leg_dof_indices` 按名解析 | 12DOF 下解析结果即 0-11，行为等价 |
| `compute_ref_state` | 硬编码 `ref_dof_pos[:, 0..11]` | `ref_dof_pos[:, leg_dof_indices[:6]/[6:]]` 广播写入 | 上半身 dof 保持 0，+=default 后即默认位姿 |
| `_reward_default_joint_pos` | `joint_diff[:, [1,2,5]]/[7,8,11]]` | 经 `leg_dof_indices` 间接索引 | 髋 roll/yaw + 踝 roll 惩罚语义不变 |
| `_reward_ankle_torques` | `[4,5,10,11]` | 经 `leg_dof_indices` 间接索引 | 未启用项顺手修正 |
| 启动日志 | 无 | 打印 `[DOF] 索引表` | 供 config 逐关节参数人工核对（本次已实战纠错一次） |

### 修改二：config 12→29DOF

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| `num_single_obs` / `num_observations` | 47 / 3102 | **98 / 6468** | 5+3×29+6 |
| `single_num_privileged_obs` / `single_linvel_index` | 73 / 53 | **141 / 121** | 5+4×29+20 / 5+4×29 |
| `num_actions` | 12 | **29** | PPO `lin_vel_idx` 公式自动=403 |
| `asset.file` | X1_12DOF_physically_mirrored.urdf | **X1_29DOF_physically_mirrored.urdf** | 右踝轴 (0 0 -1)@rpy(π,0,0)，原版 PM 约定 |
| `right_ankle_pitch_joint` 默认角 | -0.21 | **+0.21** | 轴翻转，FK 验证 0.00° 镜像 |
| `final_swing_joint_delta_pos[10]` | -0.16 | **+0.16** | 同步摆幅反号 |
| `default_joint_angles` 上半身 17 项 | 无 | lumbar(0/0/0.03)、肩(0.03/-0.06/0.18)、肘(0.34/0)、腕(0) | amp CSV 均值左右对称化（legacy exp1.1 不对称教训） |
| `control.stiffness/damping` | 仅腿部 6 键 | **16 键**（腰 60-80/肩 40/肘 30/腕 8） | 子串匹配，缺键=被动悬摆 |
| armature joint_1~29 | 按文档序（错） | **按实测 dof 序重排**：腿辨识值原样保留（joint_1-6 左腿/24-29 右腿），上半身 17 项 [0.003,0.04] 覆盖随机化 | 真机辨识成果不破坏 |

**理由**：29DOF URDF 腿部与 12DOF PM 同源（同轴系/限位/质量 36.462kg），腿部动力学配置直接继承；上半身无辨识数据，按"不猜中心、宽覆盖"原则（legacy exp1.4 踝策略迁移）。

### 4. 修改文件

- `humanoid/envs/x1/x1_dh_stand_env.py`：修改一（4 处索引化 + [DOF] 日志）
- `humanoid/envs/x1/x1_dh_stand_config.py`：修改二（维度/资产/默认角/增益/armature）
- `humanoid/scripts/play.py`：set_camera headless guard、固定 armature/damping 补上半身 17 项
- `humanoid/algo/ppo/dh_on_policy_runner.py`：obs 日志 `range(47)`→`num_single_obs`
- 环境项：F1 的 humanoid editable 安装重指到本工作区（原指向 X1_29_re0）
- 仓库：git init + 首次提交 `c169bfa` → github.com/Lee-Weather/X1_29_amp.git（api_key.json 已验证被 .gitignore 排除）

### 5. 训练参数

| 参数 | 值 |
| --- | --- |
| 训练方式 | **从零**（Flux 云端，trainType=1） |
| GM账号 | limxmtcm6wjlso8ce4@emalupe.com（账号3） |
| max_iterations | 6000 |
| save_interval | 100 |
| num_envs | 4096（config 默认） |
| seed | 5（默认） |
| learning_rate | 1e-5（fixed） |
| 算力 | **ESKU000003（1×L4 24G，¥4.11/时）**；原 4090D 任务 TASK_20260902_185 排队期改 L4 重建 |
| 镜像 | BJX00000001 / V000124（isaac-gym-v19） |
| 代码仓库 | https://github.com/Lee-Weather/X1_29_amp.git @ main，commit `c169bfa` |
| 启动命令 | `gm-run X1_29_amp/humanoid/scripts/train.py --task=x1_dh_stand --run_name=exp2_29dof_baseline --headless --max_iterations=6000`（run_name 为旧编号时期历史名） |

**风险预案**：24G 显存——29DOF obs 2.1×，若 OOM 降 `--num_envs=2048` 重跑；obs 66×98 的 CNN/estimator 维度已冒烟验证。

### 6. 预期与验收

**目标指标**（训练日志，6000 轮）：

| 指标 | 12DOF 参考（legacy exp0 本机首轮） | 本轮目标 | 异常信号 |
| --- | --- | --- | --- |
| Mean reward | 147.8 | ≥ 120 | < 80 |
| Mean episode length | 2210/2400 | ≥ 2100 | < 1500 |
| 回放跟踪 0.4/0.6 | 70.6%/69.3%（legacy exp1.4） | ≥ 65% | < 55% |
| 上半身 | — | 锁定默认位姿、无自碰撞 | 甩臂/terminate 频发 |
| 不摔倒 | ✅ | ✅ | 中途摔倒 |

### 7. 实验结果

> 训练任务：TASK_20260902_186（L4 24G，2026-09-02 17:05~22:14，运行约 5.2h，**额度耗尽自动终止于 iter 5800/6000**）
> 最终 checkpoint：**model_5800.pt**（14.1MB，29DOF 维度校验通过：actor 29/critic 423/estimator 490）
> 回放：**已执行（2026-09-03，本机 201.5 A6000，固定 armature 基准，速度阶梯 0→0.4→0.6→0）**，两次回放（无渲染/录像）Summary 逐段一致，结果可复现
> 三件套归档（lab-notebook §8）：`czy/data/exp0/` = model_5800.pt + play_output.mp4（36.6MB，40s，1:1 速度）+ isaac_diag.csv（2000 行 × 15 列，与视频同 run）

#### 训练趋势（Flux 图表 accelerate 采样）

| iter | Mean reward | Mean episode length |
| --- | --- | --- |
| 50 | 2.0 | 169 |
| 500 | 72.5 | 2048 |
| 1500 | 108.9 | 2114 |
| 3000 | 105.5 | 2116 |
| 4500 | 98.0 | 1998 |
| 5500 | 99.7 | 1994 |
| 5800（末） | 113.1 | 2242 |

#### 回放结果（0→0.4→0.6→0，各 10s）

| 指标 | 目标 | 实测 | 判定 |
| --- | --- | --- | --- |
| 不摔倒（min_height） | ✅ | **0.092 m** | ❌ **摔倒**（0.4 段 vx 冲至 2.01 后跌倒） |
| 0.4 稳态跟踪 | 90%~105% | **286%**（1.145 m/s） | ❌ 严重过冲 |
| 0.6 稳态跟踪 | 90%~105% | **203%**（1.218 m/s） | ❌ 严重过冲 |
| 停止段 | 干净停止 | **0.371 m/s 残速** | ❌ 停不下来 |
| 偏航漂移（0.4 段） | ≤3° | **-25.5°** | ❌ |
| 站立段 | 稳定 | vx≈0、净漂 0.001、偏航 0.07° | ✅ 站立完美 |
| 左右力比 | 0.95~1.05 | 0.991 | ✅ |

**结论**：❌ 未达标——训练日志健康（不摔倒、reward 平台 113）但固定基准回放**摔倒+严重过冲+停不住**，训练-回放表现严重背离。站立段完美说明上半身默认位姿锁定机制本身工作正常。

**根因分析**：
- **训练奖励存在超速漏洞**：`low_speed` 对超速（>1.2×cmd）给 0 分不罚，`tracking_lin_vel` 的 exp 高斯对大误差梯度趋零——策略在指令随机采样下学到"向前冲"的省事解，训练时 4096 env 平均 ep_len 依然高（摔倒环境占比小）；固定指令回放暴露该捷径
- **训练-回放动力学差异**：训练 armature 全域随机（髋Pitch [0.09,0.23] 等），回放固定中心值——策略可能依赖了随机域内的特定动力学
- 上半身默认角符号错误假设**排除**（站立段姿态完美、力比 0.991）；上半身行走中行为仍待目视确认

**下一轮方向（exp0.1 候选）**：
1. 堵超速漏洞：`low_speed` 对 speed_too_high 加罚（如 -0.5）或收紧 `tracking_lin_vel`（sigma 5→8）
2. 回放诊断：FIX_COMMAND 小步长（0.1/0.2 m/s）扫描，定位过冲起始的指令幅值
3. 若仍摔倒：怀疑上半身行走抖动，冻结上半身动作（回放时 action 上半身置 0）对照验证
4. 续训需切换账号 4~8（账号3 已耗尽）
