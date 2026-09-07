# 实验记录（X1_29_amp · 29DOF 全身控制）

> 本文件为 29DOF 项目实验记录（lab-notebook 规范）。
> 12DOF 旧项目（X1_29_re0）完整历史见 [exp1_12dof_legacy.md](exp1_12dof_legacy.md)（exp0~exp1.4，其结论与教训作为本系列先验）。

## 实验索引

| 编号 | 日期 | 摘要 | 状态 | Task ID | GM账号 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| exp0 | 2026-09-02 | 29DOF 全身控制基线：env 腿部按名索引 + config 29 维（obs 98/action 29/priv 141）+ 29DOF PM URDF + 上半身默认位姿锁定，从零 L4 训练至 5800 轮额度耗尽；回放摔倒（min_h 0.092m）+ 严重过冲（0.4 段 286%）+ 停不住，站立段完美 | ❌未达标（已测试） | TASK_20260902_185(停)→186 | limxmtcm6wjlso8ce4@emalupe.com（账号3，已耗尽） | model_5800.pt |
| exp0.2 | 2026-09-03 | Phase 2b mocap 参考行走：ref_lib.pt 三段（0000/0002/0026，50Hz）全身查表（腿臂同源同拍）+ 逐段步频相位 + URDF 右臂限位镜像修复；本机先训（从零 ~1600 iter 形态健康）→切云端 L20+L4 双任务并行（被手动停）→换账号5 L4 重训；回放摔倒 ~4 次/40s + 指令跟随差（cmd=0 自走 0.5m/s） | ❌未达标（已测试） | TASK_20260904_006(L20,停)/007(L4,停)/008(L4·账号5)/073(回放) | limxmtjqbym1pg0fra@emalupe.com（账号5） | model_6000.pt |
| exp0.3 | 2026-09-04 | 根因导向微调（不用 AMP）：压动作幅度（action_scale 0.3+smoothness×2.5+clip 3）治 bang-bang 前扑 + gait 调度改出生/结尾站立治停不住 + ref_joint_pos 加压制参考架空；回放零摔倒+站得住+corr 0.91，但 0.4/0.6 原地踏步不平移 | ⚠️部分达标（已测试） | TASK_20260904_086(L4·账号6) | limxmtjqd2kli2rjom@emalupe.com（账号6） | model_6000.pt |
| exp1 | 2026-09-07 | AMP 判别器引入（robolab 移植）：DHPPOAMP + LSGAN style reward lerp 融合替代 ref_joint_pos 逐关节 L2 + task 锐化（σ20/low_speed 加重）治踏步；demo 库复用 ref_lib.pt 三段差分特征。**失败**：D 86 iter 饱和死锁（agent -0.995 钉死、style≈0），1267 iter 止损（§8） | ❌失败 | TASK_20260907_046(停) | limxmtjqe95pp63oab（当前CLI） | — |
| exp1.1 | 2026-09-07 | exp1 修复：demo 侧混静立窗 + disc_lr 减半。**失败**：it30 即钉死（比 exp1 更快），静立窗位形错配 + 平凡可分根因未除（§10）；083 转纯 task 锐化基线后亦被停 | ❌失败 | TASK_20260907_083(停) | 同上 | — |
| exp1.2 | 2026-09-07 | §11 label smoothing 方案推翻（换 label 不解平凡可分）→ **exp0.2 底模 resume + AMP**：新增 --ckpt_path 直连加载；静立窗 default_dof_pos；**发现量纲淹没隐藏根因**（style 上限 0.015 vs task O(6)，梯度弱 60 倍）scale 1.5→100。本地验证：加载✅ style×54✅ 无破坏✅，D 分离待云端长程检验（§12） | ▶️待云端 | — | 同上 | 底模 czy/data/exp0.2/model_6000.pt |

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
> ⚠️ 归档已丢失（2026-09-04 检查：`czy/data/` 目录不存在，全盘无 model_5800.pt/mp4 本地副本；mp4/csv 为本机回放产物未上云，不可恢复）。诊断数值以上方 Summary 为准，如需三件套须重跑 exp0 回放

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

---

## 实验 exp0.2：Phase 2b mocap 参考轨迹行走（2026-09-03 起）

> 方案与执行细节见 [plan.md](../plan/plan.md)（§0 总体设计、§4.1 Step2/3 执行结果、§4.2 决策更新、周期 2 倍速 bug 修复记录）。

### 1. 上一实验结果与教训

> exp0 ❌：训练日志健康但回放摔倒+过冲+停不住。教训：上半身锁定默认位姿虽站立完美，但行走时无摆臂参考、全身协调无从谈起；训练-回放背离需固定指令基准回放暴露。
> exp0.2 的直接动机：给行走一个**真实的全身步态参考**（人走路必摆臂）。

### 2. 本轮修改目标

- 目标1：2b 全身查表收敛——不摔倒、Mean reward ≥ 120、ep_len ≥ 2100（对齐 exp0 验收线）
- 目标2：摆臂自然——回放目视 + 上半身 dof 轨迹 vs mocap 参考相关系数（验收新增项）
- 目标3：步速张力可控——腿部 ref 跟踪误差与 tracking 速度达标并存（0.4 段过冲 ≤ exp0 的 286% 量级）
- 验收标准：目标1 必须满足；目标2/3 记录量级，决定是否启用 plan.md §4.2 后备（时间缩放/腿部降权）

### 3. 修改内容（相对 exp0）

| 类别 | 内容 |
|---|---|
| 新增 `scripts/tools/prep_mocap_ref.py` | GMR→ref_lib.pt：重排 Isaac 序/翻转6关节/120→50Hz/自相关周期/锚点/整周期切段/限位 clip |
| 新增产物 `resources/motions/processed/ref_lib.pt` | 3 段 2620 帧@50Hz：walk_norm←0000(T=972,P=57,A=19) / walk_slow←0002(T=965,P=74,A=48) / walk_turn←0026(T=683,P=62,A=19) |
| env `x1_dh_stand_env.py` | `_init_mocap_lib`/`_current_seg_id`（指令分档）/`_get_phase` 逐段周期/`compute_ref_state` 查表（mocap_full_body=True 全身）+ `ref_action=2*(ref-default)` 修正 |
| config | `use_mocap_ref=True`、`mocap_full_body=True`（直接 2b，跳过 2a）、`ref_joint_pos` 2.2→1.8、右臂 default 镜像取反 |
| URDF | 右肩 roll limit [−2,0]→[0,2]、右肘 pitch [0,2]→[−2,0]（§0.2，训练 clamp 隐患） |
| 单测 | `test_mocap_ref_lookup.py` 6 项（含周期语义防 2 倍速回归）全过 |

### 4. 风险与预案

- mocap 步速 ~1.2 m/s vs 低指令 0.4 m/s 步速张力 → 后备：腿部 ref 降权 / 时间缩放帧推进（plan §4.2）
- `feet_*` 相位判据 vs mocap 真实触地微错位（双支撑带 13% vs 实际 ~20%）→ 回放观察，必要时放宽 |sin|<0.1

### 5. 训练参数

| 项 | 值 |
|---|---|
| 训练方式 | **本机从零**（RTX A6000 48G，非云端） |
| max_iterations | 6000 |
| num_envs | 4096（config 默认） |
| run 目录 | `logs/x1_dh_stand/exported_data/2026-09-03_17-18-51exp0_2_mocap2b/` |
| 启动命令 | `source conda.sh && conda activate F1 && cd ~/czy/X1_29_amp && pip install -e . && xvfb-run -a python -u humanoid/scripts/train.py --task=x1_dh_stand --run_name=exp0_2_mocap2b --headless --max_iterations=6000` |
| 启动验证点 | ✅ `[MOCP] 段: walk_norm(T=972,P=57,A=19), walk_slow(T=965,P=74,A=48), walk_turn(T=683,P=62,A=19)`（P=周期 bug 修复后正确值，修复前为 114/148/124） |

> ⚠️ 部署注意（post-201-5 惯例）：本机/远程每次启动前必须在项目根 `conda activate F1 && pip install -e .`——humanoid editable 会被其他实验目录（如 `czy/exp1/exp_*/`）的重装覆盖，启动日志 traceback 的 import 路径可当场鉴别。

### 6. 结果

> **训练（TASK_20260904_008，账号5 L4，从零 6000 iter，2026-09-04 14:22 完成）**：
> 最终 reward≈103（未达 120 验收线）、ep_len≈2210（≥2100 ✅）、ref_joint_pos≈+0.178。本机预训（iter 1669 被外部 kill，ckpt 完好）与云端趋势一致。
>
> **回放（TASK_20260904_073，2026-09-04 15:00 完成）**：play.py 新增 `--checkpoint_url_b64` 运行时下载 + gm 模式产物打包（cc25a36）。
> 三件套已归档 `czy/data/exp0.2/`（lab-notebook §8：每实验独立子目录，仅三文件）：model_6000.pt（14.8MB）+ play_output.mp4（43.6MB，1000 帧 @25fps，40s 1:1）+ isaac_diag.csv（2000 行 × **174 列**）。
>
> **本机复跑（2026-09-04 15:28，A6000，增强诊断版 play.py）**：对齐真机 walk_diag 列（phase_sin/cos、cycle_time、cmd_linear_*、base_euler/ang_vel 全分量、逐关节 action/pos/vel/effort/pos_des_raw ×29、clip_count）。分段 avg 0.266/0.861/1.102/0.425——与云端定性一致（0.4 段过冲 215%、停不住、站立自走），但数值有差异（GPU/PhysX/初始态非确定）。摔倒 reset 5 次（t≈12.4/16.3/22.3/25.9/29.8s，roll 峰值 1.39rad、pitch 峰值 1.49rad）；cycle_time 实测三档 {0.7, 1.14, 1.24} 验证逐段相位机制正确（0.7=站立回退、1.14=walk_norm 57帧/50Hz、1.24=walk_slow 62帧/50Hz）；力矩限幅累计 848 次（58000 dof-step 的 1.5%）。
>
> 回放 Summary（速度阶梯 0→0.4→0.6→0 各 10s）：

| 段 | cmd | avg_real | 末10s均值 |
|---|---|---|---|
| 站立 | 0.0 | 0.502 | 0.929 |
| 中速 | 0.4 | 0.541 | 1.219 |
| 快速 | 0.6 | 1.006 | 0.911 |
| 停止 | 0.0 | 0.467 | 0.126 |

> **回放诊断**：40s 内摔倒重置 **~4 次**（base_height 跌至 0.09–0.15 后跳回 0.7 = env reset，t≈8.6/19.3/24/28/33s）；指令条件化弱——cmd=0 仍自走 ~0.5 m/s（mocap 跟踪策略优先复现参考步态，速度指令通道欠训练）；0.6 段过冲至 ~1.2 m/s；末段（1800–1999）趋于稳定（h≈0.611、vel 0.124）。
>
> **验收结论**：❌ 未达标。ep_len 达标但 reward 未达线；固定基准回放暴露：摔倒频发 + 指令跟随差。mocap 全身查表解决了"形态/摆臂来源"问题（ref_joint_pos 转正），但**鲁棒性与指令条件化**是下一阶段主要矛盾。

#### 根因分析（2026-09-04，基于 174 列增强诊断 CSV，本机回放）

**① 直接死因：前向加速失控 → 前扑（5/5 摔倒同一签名）**
- 每次摔倒前 ~1s：vx 0.5→1.5-2.3 m/s，pitch +0.1→+0.6-0.9 rad，末端双离地，前扑（pitch 峰值 1.49 rad）
- 存活间隔仅 3.6-6.0s（reset 后很快再次失控）；正常步并不差（vx 0.53/0.60 vs cmd 0.4/0.6，pitch +0.04~0.09，超 1.5×cmd 仅 13%/6%）→ **边缘稳定**，偶发扰动即发散
- 云端 L4 vs 本机 A6000 同 ckpt 分段 0.502/0.541/1.006/0.467 vs 0.266/0.861/1.102/0.425、摔倒次数/时刻均不同 → 对初始条件/数值噪声高度敏感，佐证边缘稳定

**② 核心病灶：动作幅度失控（bang-bang 化），参考跟踪被架空**
- 期望位置 des 半幅 vs mocap 参考：hip pitch L 1.07/0.46、R 1.55/0.42 rad（**3-3.7 倍**）；ankle pitch L 0.73/0.38、R 0.97/0.34；hip roll R 1.25/0.32
- 原始 action |a|：右 hip pitch 均值 1.40 峰值 3.56；失控瞬间 aR_ankle -5.1、aHipR +4.3（action_scale=0.5 → des 偏离 ±2.5 rad）
- corr(des, pos)：hip pitch -0.02/+0.01、right_ankle -0.02 → **期望与实动完全脱钩**；实动/mocap 幅度比膝 0.48-0.64、右踝 0.47（kp=30/35 太低跟不上 1.6Hz 大摆幅指令）
- 右侧系统性比左侧极端（des R/L = 1.55/1.07），肘部实动 3.5x/8.9x mocap（挥舞）
- 推论：策略学成"大幅甩腿侥幸保平衡"——低增益下输出 2-3 倍幅度换取部分实动，偶发过推即触发①

**③ 站立吸引子缺失（停不住）**
- S0/S3（cmd=0，phase_sin 全程=0、cycle=0.7）机器人仍行走 0.27/0.43 m/s，超速步占比 58%/92%
- 训练 gait 调度 `["walk_omni","stand","walk_omni"]`：站立段仅占 ~20%（2-3s/12.5s 周期）且**出生段必为行走**（generate_gait_time 的 gait_time[:,0]=0 → spawn 即采样行走指令）→ "出生+站立"组合零训练（恰是回放 S0 分布）；中段站立仅 2-3s，"滑行穿越站立段"代价低于"停-再起步"
- 奖励漏洞延续 exp0：tracking 高斯在 cmd=0、v=0.4 时仍得 53% 分（σ=0.25）；low_speed 超速分支得 0 分不罚；only_positive_rewards 进一步弱化负激励

**④ 相位时钟与实际迈步脱钩（次要）**
- 时钟周期 1.14s vs 实际触地节律 ~0.6-1.0s；站立段 phase=0 仍迈步 → 节律由策略自身反馈维持（obs 含历史 action），时钟未成为主导
- 手臂 17Nm 弱限幅频繁饱和（right_shoulder_pitch 260 dof-step，83% 在摔倒窗口）属摔倒后果非原因

**为什么训练日志看似健康**：4096 env 平均化 + 25s 指令重采样，单回合内失控概率被摊薄（ep_len 2210）；stumbling 步态仍能拿 tracking(2.2)+ref_joint_pos(1.8) 中等分；动作正则太弱（dof_acc=-1e-7 形同虚设）无法压制 bang-bang。

**下一轮方向（exp0.3 候选修改）**：
1. **压制动作幅度**（治②，优先）：action_scale 0.5→0.3；action_rate 惩罚×3-5；可加 |action| L1 罚或输出限幅
2. **治理停不住**（治③）：指令采样 30% 概率 cmd=0；low_speed 改对称罚（超 1.2×cmd 线性罚）
3. **锁相**（治④）：stance_mask vs 实际触地一致性奖励，或 feet_contact_number 权重再调
4. **中期**：参照 amp_architecture_notes.md §6 引入 AMP 判别器替代逐关节 ref 惩罚（治②的根治路径）

---

## 实验 exp0.3：根因导向微调——压幅度 + 治停不住（2026-09-04）

### 1. 上一实验结果与教训

> 数据：exp0.2 model_6000 本机增强诊断回放（174 列 CSV，40s 速度阶梯）
> - 摔倒 5 次/40s，全部同签名：vx 0.5→1.5-2.3 m/s 失控 → pitch +0.9 → 双离地前扑；存活间隔仅 3.6-6s
> - des 半幅 = mocap 参考 3-3.7 倍（右 hip_pitch 1.55/0.42 rad），corr(des,pos)≈0，实动仅参考一半
> - cmd=0 段仍走 0.27-0.43 m/s（超速步 58%/92%）；站立组合"出生+cmd=0"训练分布外
>
> **核心教训**：
> - 证明了：mocap 全身查表机制本身正确（cycle_time 三档实测无误），但**参考被 bang-bang 动作架空**——策略输出 2-3 倍幅度、低 kp(30/35) 实动塌缩、期望与实动脱钩
> - 否定了："reward 103 + ep_len 2210 = 接近达标"的乐观解读——4096 env 平均摊薄了边缘不稳定
> - 本轮要解决：① 动作幅度失控（直接死因）② 站立吸引子缺失（停不住）

### 2. 本轮修改目标

- 目标1：回放 40s **零摔倒**（exp0.2 为 5 次）
- 目标2：cmd=0 段 |vx| < 0.15 m/s（停得住）
- 目标3：0.4/0.6 稳态跟踪进入 80-120% 区间（exp0.2 过冲 135%/189%）
- 验收标准：训练 reward ≥ 105、ep_len ≥ 2300；回放 corr(des,pos) hip_pitch ≥ 0.5、clip_count < 200（exp0.2: 848）

### 3. 修改内容

#### 修改一：压制动作幅度（治②→①，核心）

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| control.action_scale | 0.5 | 0.3 | des 偏移幅度直接 -40%：hip des ±1.5→±0.9 rad，可跟踪性大增 |
| rewards.action_smoothness | -0.008 | -0.02 | 已含 Σ\|a\| L1 + 一阶/二阶差分，×2.5 直击 bang-bang |
| normalization.clip_actions | 100. | 3. | env.step 入口硬界原始 action（legged_robot L118），des 偏移上限 = 3×0.3 = ±0.9 rad |

**理由**：实动/mocap 幅度比仅 0.47-0.64、corr≈0 说明策略用"大幅甩"换部分实动；把输出幅度压到参考可实现范围，ref_joint_pos 与 tracking 才能形成有效梯度。clip=3 与 init_noise_std=1.0 兼容（±3σ 内），PPO log_prob 用截断前分布计算，梯度正常。

#### 修改二：gait 调度与站立奖励（治③）

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| commands.gait | ["walk_omni","stand","walk_omni"] | ["stand","walk_omni","stand"] | 出生段必为行走是"出生+站立"零训练的根源；改后覆盖回放 S0（出生站立）与 S3（走后停）两个分布 |
| commands.gait_time_range.stand | [2,3] | [3,5] | 站立段更长（"滑行穿越"代价升高） |
| commands.gait_time_range.walk_omnidirectional | [4,6] | [6,9] | 平衡占比：站立 ~20%→~26%，行走数据不稀释过度 |
| rewards.scales.stand_still | 2.5 | 3.5 | 加强站立吸引子；仅 stand_command 时非零，不伤行走段 |
| _reward_low_speed 超速分支 | 0. | -1.0 | 对称罚：低速 -1.0 / 超速 -1.0 / 达标 +1.2（掩码 cmd>0.05 不变） |

**理由**：根因③实测——S0/S3 全程 phase=0 语义生效但仍自走。出生+站立为分布外组合是 S0 失败的直接解释；站立段 2-3s 太短使"滑行穿越"成为省事解。

#### 修改三：强化 mocap 参考约束（治②"架空"）

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| rewards.scales.ref_joint_pos | 1.8 | 2.4 | 压幅度后参考可实现（des ≈ ±0.6-0.9 vs mocap ±0.45），升权让查表参考真正成为主导目标 |

**理由**：exp0.2 参考跟踪被架空时 ref_joint_pos 仅 +0.178；修改一落地后参考与输出同量级，此时加压才有意义（否则逼策略追不可实现目标）。

### 4. 修改文件

- `humanoid/envs/x1/x1_dh_stand_config.py`：control.action_scale、normalization.clip_actions、commands.gait、gait_time_range、rewards.scales.{action_smoothness, stand_still, ref_joint_pos}
- `humanoid/envs/x1/x1_dh_stand_env.py`：`_reward_low_speed` 超速分支 0→-1.0

### 5. 训练参数

| 参数 | 值 |
| --- | --- |
| 训练方式 | 从零 |
| GM账号 | limxmtjqd2kli2rjom@emalupe.com（账号6） |
| 任务 ID | **TASK_20260904_086**（2026-09-04 16:12 启动，L4） |
| max_iterations | 6000 |
| save_interval | 100 |
| num_envs | 4096 |
| seed | 5 |
| learning_rate | 3e-4（同 exp0.2，未动） |
| 算力 | L4（ESKU000003，¥4.11/h） |
| 镜像 | BJX00000001, V000124 |
| 代码仓库 | https://github.com/Lee-Weather/X1_29_amp.git, main @ 017463d |
| 启动命令 | `gm-run X1_29_amp/humanoid/scripts/train.py --task=x1_dh_stand --run_name=exp0_3_amplitude --headless --max_iterations=6000` |

### 6. 预期与验收

**目标指标**（训练日志，6000 轮）：

| 指标 | exp0.2 实测 | 本轮目标 | 异常信号 |
| --- | --- | --- | --- |
| Mean reward | 103 | ≥ 105 | < 90（幅度压制过狠） |
| Mean episode length | 2210 | ≥ 2300 | < 2000 |
| ref_joint_pos | +0.178 | ≥ +0.3 | < +0.1（参考仍架空） |
| tracking_lin_vel | — | ≥ 0.7 | < 0.5 |

**回放验收**（速度阶梯 0→0.4→0.6→0，增强诊断 CSV）：

| 指标 | exp0.2 实测 | 本轮目标 |
| --- | --- | --- |
| 摔倒次数/40s | 5 | **0** |
| cmd=0 段 \|vx\| | 0.27-0.43 | < 0.15 |
| 0.4/0.6 稳态跟踪 | 135%/189% | 80-120% |
| corr(des,pos) hip_pitch | ≈0 | ≥ 0.5 |
| clip_count | 848 | < 200 |
| 触地节律 vs cycle_time | 0.6-1.0s vs 1.14s | 1.14s ±20% |

### 7. 实验结果

> 训练任务：TASK_20260904_086（账号6 L4，2026-09-04 16:12 → 09-04 21:21 完成，从零 6000 iter）
> 最终 checkpoint：model_6000.pt（14.8MB）
> 三件套归档：`czy/data/exp0.3/`（model_6000.pt + play_output.mp4 29.7MB 1000帧 + isaac_diag.csv 2000行×174列）

#### 最终结果（iter 6000）

| 指标 | exp0.2 | 目标 | 实测 | 判定 |
| --- | --- | --- | --- | --- |
| Mean reward | 103 | ≥105* | 63.8 | —（惩罚加严+结构变化，与旧值不可直接比） |
| Mean episode length | 2210 | ≥2300 | 1125 | ❌ 腰斩 |
| rew_ref_joint_pos | +0.178 | ≥+0.3 | **+0.566** | ✅ |
| rew_tracking_lin_vel | — | ≥0.7 | 0.409 | ❌ |
| rew_stand_still | — | — | 0.530 | 站立吸引子生效 |

#### 回放验收（速度阶梯 0→0.4→0.6→0，本机 A6000，增强诊断 CSV）

| 验收线 | exp0.2 | 目标 | exp0.3 实测 | 判定 |
| --- | --- | --- | --- | --- |
| 摔倒次数/40s | 5 | 0 | **0**（h_min 0.547，无 reset） | ✅ |
| cmd=0 段 \|vx\| | 0.27-0.43 | <0.15 | **0.000-0.012** | ✅ |
| 0.4/0.6 稳态跟踪 | 135%/189% | 80-120% | **5%/-2%**（原地踏步） | ❌ |
| corr(des,pos) 左髋 | ≈0 | ≥0.5 | **0.908**（右髋 0.379） | ✅ |
| clip_count | 848 | <200 | **0** | ✅ |
| des 半幅 vs mocap | 3-3.7× | ≈1× | 0.9-1.7×（LHP 0.573/0.457） | ✅ |

**结论**：⚠️ 部分达标（有 trade-off）——**稳定性与听话问题全治好了，但走路能力丢了**。机器人全程零摔倒、cmd=0 完美站住、动作幅度回归参考量级且期望-实动高度相关；然而 0.4/0.6 指令下只做原地踏步（腿动、相位跑、触地交替，但 vx≈0）。

**根因分析**：
- **"原地踏步"是奖励结构下的局部最优**：踏步可全额拿到 ref_joint_pos(2.4)+feet_air_time(1.2)+feet_contact_number(2.4) 等步态形奖励；唯一驱动平移的 tracking_lin_vel 用 σ=5 的高斯，cmd=0.4/v=0 时仍得 exp(-0.16×5)=0.45——"不平移"几乎不受罚（low_speed 权重仅 0.2，too-slow 罚 -1×0.2=-0.2 vs stand_still 3.5 满额）。压幅度后走路的风险成本（smoothness/摔倒）上升而收益未变 → 策略选择最安全的踏步
- exp0.2 的过冲与 exp0.3 的踏步是同一枚硬币两面：tracking 梯度太平（σ=5），策略在"冲过去"和"不走"之间摆，没有精确跟踪的吸引子
- 训练 ep_len 1125 腰砍与轨迹中段摔倒+惩罚性终止一致（训练期曾尝试行走，回放则更保守）

**exp0.4 方向（微调，不动 exp0.3 已验证的稳定性项）**：
1. **锐化平移梯度**（核心）：tracking_sigma 5→20（cmd=0.4/v=0 → exp(-3.2)=0.04，踏步不再白拿 tracking 分）
2. **加重不走罚**：low_speed 权重 0.2→1.0 且 too-slow 罚 -1→-2（踏步净收益转负）
3. 保持 exp0.3 全部修改（action_scale 0.3/clip 3/smoothness -0.02/gait 出生站立/ref 2.4）——它们是本次成功的部分

---

## 实验 exp1：引入 AMP 判别器（相位时钟 + 风格判别混合）（2026-09-07 方案）

> 晋级依据（lab-notebook §1.3）：跨 ≥2 模块大改（新增 algo/amp 模块 + env 特征管线 + 奖励结构换血）——exp0.3 未达标也晋级，修改编号重置。

### 1. 上一实验结果与教训

> 数据：exp0.3 model_6000（TASK_20260904_086，reward 63.8 / ep_len 1125 / ref_joint_pos +0.566）
> - 回放：零摔倒（exp0.2 为 5 次）、cmd=0 段 vx≈0.00（完美站住）、corr(des,pos) 左髋 0.908、clip_count 0
> - 但 0.4/0.6 指令下原地踏步（vx 0.019/-0.01），稳态跟踪 5%/-2%
> - 根因：踏步是局部最优——步态形奖励（ref_joint_pos 2.4+air_time 1.2+contact_number 2.4）全额白拿，tracking σ=5 太平（cmd=0.4/v=0 仍得 45% 分），low_speed 罚仅 -0.2
>
> **核心教训**：
> - 证明了：exp0.3 的幅度压制/站立调度/参考可实现化三项机制全部有效（三大病灶治愈）
> - 否定了："加 ref_joint_pos 权重能让策略走起来"——逐关节 L2 只管形态不管平移，反而给踏步发奖
> - 本轮要解决：① 平移驱动（task 侧锐化）② 风格从"逐关节 L2"升级为"全局判别"（AMP，治摆臂自然度上限 + 替掉踏步白拿的 ref_joint_pos）

### 2. 本轮修改目标

- 目标1（继承 exp0.3）：回放零摔倒、cmd=0 停住（|vx|<0.15）、clip_count<200
- 目标2（治踏步，task 侧）：0.4/0.6 稳态跟踪 80-120%
- 目标3（AMP 侧）：风格自然度——视频目视摆臂协调 + disc demo/agent score 差值收敛（|Δ|<0.5 且不再单调增大）
- 目标4（工程）：`amp.enabled=False` 单开关退化为纯 task 基线（消融能力）
- 验收标准：回放 0.4/0.6 稳态 80-120% + 零摔倒 + 摆臂目视自然（对比 exp0.2 视频明显改善）

### 3. 修改内容

#### 修改一：新增 AMP 算法模块 `humanoid/algo/amp/` + `dh_ppo_amp.py`

**架构**（robolab ppo_amp.py 移植映射，见 amp_architecture_notes.md §6.3）：

```
DHPPOAMP(DHPPO)                          # dh_ppo_amp.py，不改父类
  ├─ AMPDiscriminator                    # amp/amp_discriminator.py
  │    MLP [1024,512]+ELU → Linear(1)；逐单步 EmpiricalNormalization
  │    输入 183 维 = 3 步窗 × 61 维/步
  │    61 = root_ang_vel(3, 体轴) + dof_pos(29, 绝对) + dof_vel(29)
  │    （v1 不含 key_body_pos——ref_lib 无 FK 数据，notes §6.4 允许）
  ├─ CircularBuffer ×2                   # amp/amp_buffers.py
  │    agent/demo 各一，容量 100 控制步 > rollout 窗，FIFO 滑窗不清空
  ├─ disc_optimizer                      # 独立 Adam lr=1e-4 恒定（KL 自适应不波及）
  │    trunk/linear 分组 weight decay 1e-3/1e-1；grad clip 1.0
  └─ 覆写两个方法（父类其余 199 行不动）：
       process_env_step()  # 奖励融合点
       update()            # disc loss 训练点
```

**a) 奖励融合**（覆写 `process_env_step`，对应 dh_ppo.py L110-120）：

```python
disc_obs      = infos["amp"]["disc_obs"]        # env 侧采好经 extras 传入
disc_demo_obs = infos["amp"]["disc_demo_obs"]
with torch.no_grad():                            # 旧参 eval
    d = self.disc(disc_obs)
    rew = torch.clamp(1 - (d - 1) ** 2 / 4, min=0)   # LSGAN 映射，值域[0,1]
style = dt * style_reward_scale * rew            # ×0.02×1.5，与控制频率解耦
rewards_fused = lerp * rewards + (1 - lerp) * style   # lerp=0.6
super().process_env_step(rewards_fused, dones, infos) # 融合值进 GAE
self.disc_obs_buffer.append(disc_obs); self.disc_demo_buffer.append(disc_demo_obs)
```

- only_positive_rewards=True 在 env 侧先 clip 后传出，style≥0 无冲突
- **站立掩码**：站立 env（‖cmd‖≤0.05）style 项置零（纯 task）——demo 库三段全是行走，不 mask 会与 stand_still(3.5) 打架，威胁 exp0.3 已验证的站立成果

**b) 判别器训练**（覆写 `update`，mini-batch 循环内 PPO backward 之后追加）：

```python
agent_batch = self.disc_obs_buffer.sample(mb_size)     # CircularBuffer 滑窗采样
demo_batch  = self.disc_demo_buffer.sample(mb_size)
d_agent, d_demo = self.disc(agent_batch), self.disc(demo_batch)
disc_loss = 0.5 * (MSE(d_agent, -1) + MSE(d_demo, +1))       # LSGAN
gp = 10 * grad_norm(demo_batch → d_demo).pow(2)              # 只罚 demo 侧（AMP 论文标准）
disc_optimizer.step(); PPO optimizer 照常独立 step
update_normalization()  # 事后更新（本批用旧统计量）
```

- rollout 时 reward 用旧 D 算（no_grad），训练时同批交替，无冻结
- `storage.clear()` 照旧，CircularBuffer 不清空（跨迭代混合防 stale）
- 返回值保持 3 元组兼容 runner；AMP 指标存 `self.amp_stats`（disc_score/disc_demo_score/disc_loss/grad_penalty/style_reward）

**c) 启动硬断言**（notes §4-4）：agent/demo 特征逐项维度断言 + ref dof_names 与 env dof_names 顺序一致性断言（沿用 ref_lib 加载惯例）

#### 修改二：env 侧 demo 特征管线（`x1_dh_stand_env.py`）

- **加载期预计算**（ref_lib.pt 三段，毫秒级）：`dof_vel = diff(dof_pos)×fps`、`root_ang_vel = 体轴差分(root_rot)×fps`；拼成每段 `(T, 61)` 特征张量 + 预取 3 步滑窗索引池
- **每步采样**（post_physics_step 中，step 之后、reset_idx **之前**——robolab reset 污染反面教材 notes §4-7）：
  - agent 侧：`[base_ang_vel(3), dof_pos(29), dof_vel(29)]` 维护 3 步环形历史
  - demo 侧：每步随机抽（段、起始帧）——random_fetch 风格，与相位解耦（纯风格匹配，无 tracking 张力）
- 经 `extras["amp"]` 传出（policy 观测保持纯净，notes §6.3 反面教材规避）
- 开关：`rewards.amp.enabled=False` 时 extras 不产出、DHPPOAMP 退化为纯 task（消融用）

#### 修改三：task 侧锐化（治 exp0.3 踏步，exp0.4 方案并入）

| 参数 | exp0.3 值 | 新值 | 说明 |
| --- | --- | --- | --- |
| rewards.tracking_sigma | 5 | 20 | cmd=0.4/v=0 → exp(-3.2)=0.04，踏步不再白拿 tracking 分 |
| rewards.scales.low_speed | 0.2 | 1.0 | 配合 σ 锐化 |
| _reward_low_speed too-slow 罚 | -1.0 | -2.0 | 踏步净收益转负 |
| rewards.scales.ref_joint_pos | 2.4 | **0.0** | **AMP style 接管形态**——保留则踏步白拿漏洞仍在且与 style 双重计分打架 |

**保持 exp0.3 全部修改不动**（action_scale 0.3 / clip_actions 3 / smoothness -0.02 / gait 出生结尾站立 / stand_still 3.5 / low_speed 对称罚）。

#### 修改四：配置注入（零侵入）

- `X1DHStandCfgPPO.algorithm.algorithm_class_name = 'DHPPO'` → `'DHPPOAMP'`（runner eval 字符串注入，L73-74，改动仅 1 行）
- PPO cfg 新增 `class amp`：disc_hidden `[1024,512]` / disc_lr 1e-4 / grad_penalty 10 / buffer_size 100 / disc_obs_steps 3 / style_reward_scale 1.5 / task_lerp 0.6
- env cfg 新增 `class amp`：enabled / 站立掩码开关
- runner `save/load` 扩展：disc + normalizer + disc_optimizer state（断电续训不丢判别器）；`log()` 增补 5 个 AMP 标量

### 4. 修改文件

- 新建 `humanoid/algo/amp/__init__.py`、`amp/amp_discriminator.py`、`amp/amp_buffers.py`
- 新建 `humanoid/algo/ppo/dh_ppo_amp.py`（DHPPOAMP，仅覆写 2 方法 + __init__ 扩展）
- 改 `humanoid/envs/x1/x1_dh_stand_env.py`：demo 特征预计算 + 每步采样 + extras 通道 + amp.enabled 开关
- 改 `humanoid/envs/x1/x1_dh_stand_config.py`：algorithm_class_name、amp 超参块、tracking_sigma/low_speed/ref_joint_pos
- 改 `humanoid/algo/ppo/dh_on_policy_runner.py`：save/load amp state、log amp stats（~15 行）
- 新建 `scripts/tools/test_amp_disc.py`：维度断言 / buffer FIFO / style reward 值域 [0,1] / LSGAN loss 收敛 sanity（离线小张量，无需 GPU 仿真）

### 5. 训练参数

| 参数 | 值 |
| --- | --- |
| 训练方式 | 从零（备选：exp0.3 model_6000 续训，见决策点②） |
| GM账号 | 账号6 limxmtjqd2kli2rjom（余额 ~29，一次 6000 iter L4 ≈ ¥21 够） |
| max_iterations | 6000 |
| num_envs / num_steps_per_env | 4096 / 24（rollout 窗 < buffer 100 ✅） |
| seed | 5 |
| PPO lr | 3e-4 adaptive（不动）；disc lr 1e-4 恒定 |
| 算力 | L4（ESKU000003） |
| 启动命令 | `gm-run .../train.py --task=x1_dh_stand --run_name=exp1_amp --headless --max_iterations=6000` |

### 6. 预期与验收

**训练监控**（新增 AMP 指标，止损线）：

| 指标 | 健康 | 异常处置 |
| --- | --- | --- |
| disc_demo_score → +1 / disc_score → -1 | 渐近但不清零（对抗平衡） | agent 分恒 0：判别器碾压 → disc_lr 1e-4→5e-5 或 grad_penalty 10→20 |
| style_reward 均值 | 中后期 > 0.3（满额 dt×1.5×1≈0.03/步，归一后看趋势） | 恒 0：policy 无风格梯度 |
| disc_loss | 平稳小幅震荡 | 单调 → 0：判别器必胜，同上处置 |
| Mean reward | 与 exp0.3(63.8) 结构不同不可直比，看 ep_len | ep_len < 800 → style 压垮 task，lerp 0.6→0.75 |

**回放验收**（速度阶梯，增强诊断 CSV）：

| 验收线 | exp0.2 | exp0.3 | exp1 目标 |
| --- | --- | --- | --- |
| 摔倒/40s | 5 | 0 | 0 |
| cmd=0 停住 | ✗ | 0.00 | <0.15 |
| 0.4/0.6 稳态 | 135%/189% | 5%/-2% | **80-120%** |
| 摆臂自然度（目视） | 挥舞 | 端平 | 接近 mocap 步行摆臂节律 |
| 触地节律 vs cycle_time | 0.6-1.0 vs 1.14 | — | 1.14±20% |

### 7. 实施与启动记录（2026-09-07）

**决策拍板**：① ref_joint_pos 归零 ② 从零训练 ③ 不做并行消融（失败时再补跑）。

**实施完成**（commit 8937629，9 文件 +1101 行）：

- 新建 `humanoid/algo/amp/`（AMPDiscriminator + EmpiricalNormalization + CircularBuffer 精简移植）与 `dh_ppo_amp.py`（覆写 `process_env_step`/`update`，`configure_amp` 由 runner 注入 obs 维/num_envs/dt）
- env：`_init_amp`（demo 特征预计算：dof_vel 前向差分、root_ang_vel 四元数差分精确 log 映射）+ `post_physics_step` pre-super 采样 + reset 整窗填充防污染 + `amp.enabled` 开关
- config：`DHPPOAMP` + FLAT amp 超参（class_to_dict 平铺键直传 kwargs）+ task 锐化四项
- runner：save/load 判别器+归一化统计+独立优化器、log 增 5 个 AMP 标量
- 离线单测 `scripts/tools/test_amp_disc.py` **28/28 通过**（含 inference_mode 采集→update 反向的推理张量污染专项）
- 本地 isaacgym 冒烟（64 env×2 iter）通过

**实施中发现并解决的问题**：

1. **dt 疑云落地**：env.dt 实测 0.01（100Hz），mocap 50Hz → 0.5 帧/步，整数 stride 会使 demo 窗时间尺度失真 2 倍。解法：agent 侧隔步采样（`agent_stride=2`，每 2 控制步滚动一次历史），两侧窗口跨度精确对齐 0.04s（打印确认）。
2. **云端任务 v1（TASK_20260907_065）启动即败**：`can't open file 'X1_29_amp/.../train.py'`——create JSON 缺 `goodsBackId`/`personalDataPath` 致代码卷未挂载。v2 补齐（SKUSL000003 + /personal）后正常。

**云端任务**：TASK_20260907_067（exp1_amp_v2），项目 PRO_20260907_019（X1_29_amp），账号 limxmtjqe95pp63oab（CLI 现登账号，今日登录），L4 ¥4.11/h，6000 iter ETA ~7h。

**早期监控（11 iter）**：disc loss 0.012、agent -0.960 / demo +0.996、style 0.0——判别器初期碾压属预期（随机策略 vs mocap 天然可分），关注 500+ iter 后 agent 分是否回升（止损处置见 §6 表）。

**中期监控（1267 iter）**：判别器死锁确认（详见 §8），2026-09-07 13:11 手动终止。

### 8. 实验结果（exp1 失败：AMP 通道死锁，1267/6000 iter 提前终止）

**终止**：TASK_20260907_067 手动停止（2026-09-07 13:11，运行 1.4h ≈ ¥5.7）。

**数据快照（终止时 / 全程曲线）**：

| 指标 | it86 | it1267 | 判读 |
| --- | --- | --- | --- |
| disc_score (agent) | -0.995 | -0.995 | **86 iter 起钉死，1180 iter 零变化** |
| disc_demo_score | +0.999 | +1.000 | 完美分类 |
| disc_loss / grad_penalty | — | 0.0005 / 0.0000 | D 饱和，梯度≈0，事实冻结 |
| style_reward | 0.00022（全程峰） | 0.00015 | 满额 0.03 的 0.5%，无爬升 |
| Mean reward | 7.9 | 36.35（持续爬升） | task 锐化独立起效 |
| Mean episode length | 347 | 799-858 波动 | 止损线 800 边缘徘徊 |

**根因分析（三层）**：

1. **现象层**：D 在 86 iter 内完成 agent/demo 完美分类并饱和——LSGAN 损失→0 → 梯度→0 → D 冻结；style = clamp(1-(D-1)²/4) 在 D≈-1 时恒 0。
2. **死锁闭环**：style=0 → policy 无风格梯度 → 只按 task 学（ref_joint_pos 已归零，无任何形态先验）→ 步态分布不向 mocap 移动 → D 保持满分 → 永不解锁。task 侧（reward 36↑）与 AMP 侧（死平）完全解耦，互证。
3. **结构根因（为何 D 能瞬间碾压——与 robolab 成功配置的关键差异）**：
   - **站立样本污染 agent buffer**（主因）：gait 调度 [stand,walk,stand] 使 ~26% 样本为静立窗，D 靠"运动幅度"单一平凡特征即可近乎完美分类，无需学细粒度步态风格；且静立样本持续给 D 提供"免费正确分类"，压低 loss、稳定死锁。
   - **demo 多样性低**（次因）：3 段 vs robolab 14 段带权，D 记忆 demo 流形过易。
   - 特征 61 维无 key_body_pos（v1 已知取舍），判别信息更集中，助攻碾压。

**教训**：

- AMP 对抗平衡对 **agent/demo 分布的可比性**极敏感：env 中存在 demo 库没有的行为模式（站立）时，D 必然走捷径——demo 侧须对称补齐该模式。
- 止损表 §6 第一行在 it~100 即触发，但"早期碾压属预期"的误判拖到 1267。**判据应加时间窗：it>300 仍碾压即为异常，立即处置**。

### 9. exp1.1 修改方案（2026-09-07，待审批）

> 目标：解开死锁——堵掉 D 的平凡分类特征，降 D 收敛速度给 policy 追赶窗口。exp1 基础上仅两处改动（<20 行）。

**修改一（核心）：demo 侧按站立占比混入静立窗**（`x1_dh_stand_env.py` `_sample_amp_demo`）

- 每步实时取 p = stand_mask.mean()（agent 侧当前站立占比，~26% 随 gait 调度波动）
- 常规随机行走窗采样后，每个 env 以概率 p 替换为静立窗：随机段随机帧 q_t 重复 S 次，特征 [ang=0, q_t, dq=0]
- 效果：两侧静立占比逐批精确匹配 → "运动幅度"不再是判别特征 → D 被迫学习行走动态细节（真正的风格）
- 融合不受影响：站立 env 的 style 仍被 stand_mask 屏蔽（纯 task），风格信号只作用于行走 env

**修改二（双保险）：amp_disc_lr 1e-4 → 5e-5**（config 1 行）

- 即便堵了平凡特征，[1024,512] 判别器对 61 维特征仍有容量优势；降速给 policy 分布靠近 demo 的时间窗

**监控线（更新判据）**：

| 指标 | exp1 实测 | exp1.1 解锁信号 | 仍死锁处置（it500 判定） |
| --- | --- | --- | --- |
| disc_score (agent) | -0.995 钉死 | it<300 回升 >-0.9 | 候选：LSGAN label smoothing（目标 ±0.7） |
| style_reward | 0.00015 | >0.001 且持续爬升 | 同上 |
| disc_loss | 0.0005（饱和） | 平衡于 0.1-0.5 区间 | 单调→0 同上 |

**训练参数**：同 exp1（从零 / 6000 iter / L4 / seed 5），run_name=exp1_1_amp。

### 10. exp1.1 结果（2026-09-07，修复失败，TASK_20260907_083 转任务基线）

- 三卡并行启动：083 L20（运行中）/ 084 L4 / 085 4090（草稿未起）；用户决定只保留 083
- **083 实测（it613，代码已验证为 exp1.1 commit 5e8511e）**：disc_score it30 即 -0.979、it613 -0.995 钉死，style 0.00013——**静立窗混合 + lr 减半均未防住死锁**（比 exp1 略快：86→30 iter 饱和）
- 084/085 已停/未启，损失合计 ~¥0.8

**exp1.1 失败的诚实归因**：

1. **静立窗内容错了**：混入的是"冻结的迈步中间帧"（mocap 随机帧 q_t），而 agent 站立是默认位姿（dof≈default）——两侧关节构型不同，D 仍一票分类。正确做法要么用 default_dof_pos 造静立窗，要么干脆移除站立样本。
2. **根因定位不完整（更重要）**：**早期随机策略本身就与 mocap 平凡可分**——乱动/摔倒/大幅抖动 vs 平滑周期步态，无需借助站立特征。死锁是"D 容量优势 + 早期分布天然分离"的结构性结果，堵任何一个单一特征都防不住。

**083 的意外价值**：D 死锁 → style≡0 → 083 实际是**纯 task 锐化基线**（= 从没买的消融：σ20 / ref_joint_pos 0 / low_speed 1.0 / too-slow -2，无任何形态先验）。it613 reward 23.5↑ / ep_len 665，健康跑完 6000 iter 后直接回答："task 锐化单独能否逃出原地踏步？"——决定 AMP 定位（锦上添花 vs 必需品）。

### 11. 实验 exp1.2：LSGAN label smoothing（2026-09-07 方案，待审批）

> 晋级依据：exp1/exp1.1 两连败同根因（D 饱和→梯度消失），非超参问题，需结构性修复。exp1.1 修复（数据侧）失败，本轮转攻损失函数侧——这是 GAN 反判别器饱和的标准解，也是 §9 备选清单第一项。

#### 1. 核心思路：不让 D 有"满分"可达

exp1/exp1.1 的死锁链：

```
D(agent)→-1, D(demo)→+1 完全可达 → MSE→0 → 梯度→0 → D 冻结
→ style = clamp(1-(D-1)²/4) 在 D=-1 时恒 0 → policy 无风格梯度 → 永不解锁
```

**label smoothing**：判别目标从 ±1 改为 ±τ（τ=0.7）。D 最优解变成 D(agent)=-0.7, D(demo)=+0.7——**loss 在此处有非零曲率**，梯度永不消失：

- D 停在 -0.7 附近小幅波动 → 偶尔越过 -0.7 的 agent 窗（更像 demo 的）被推向更接近 demo 一侧
- **风格排序信号始终存在**：style = clamp(1-(D-1)²/4)，D∈[-0.7,0.7] 时 style∈[0.20, 0.68]——policy 内部不同行为有持续的风格分差
- 这正是 AMP 论文社区对 LSGAN 碾压的标准处置（StyleGAN/Pix2Pix 同思路）

#### 2. 修改内容（2 文件，~15 行）

**修改一：判别器损失目标 ±1 → ±0.7**（`dh_ppo_amp.py` update() 内 2 处）

```python
# 现：disc_loss = 0.5 * (mse(disc_score, -1) + mse(disc_demo_score, +1))
amp_label_smooth = 0.7   # __init__ 传入，config 可调
disc_loss = 0.5 * (mse(disc_score, -amp_label_smooth * ones)
                   + mse(disc_demo_score, +amp_label_smooth * ones))
```

**修改二：style 奖励映射同步平移**（`amp_discriminator.py` predict_style_reward 1 处）

LSGAN 映射 `clamp(1-(D-1)²/4)` 的"满分点"是 D=+1（demo 目标），agent 目标 D=-1 时 style=0。平移后 demo 目标 D=+0.7、agent 目标 D=-0.7：

```python
# 现：rew = clamp(1 - 0.25*(D-1)²)
# 新：满分点对齐 demo 目标 τ=0.7
rew = clamp(1 - 0.25*((D - tau) / 1.0)² * 1.0)   # 化简：clamp(1 - (D-0.7)²/4·(1/0.7²)·...)
```

精确推导（保持 agent 目标处 style=0、demo 目标处 style=1 的锚点语义）：

```
rew(D) = clamp(1 - ((D - tau)²) / ((1 + tau)²) , min=0)
   D=tau=0.7 → rew=1（像 demo 得满分）
   D=-0.7    → rew = 1 - (1.4²/1.4²) = 0（agent 目标得 0 分）
   D=0（中性）→ rew = 1 - 0.49/1.96 = 0.75
```

值域仍 [0,1]、单调、单调方向不变——**GAE 尺度与 exp1 可比，无需调 lerp/scale**。

**修改三（辅助，保留 exp1.1 修复）**：静立窗内容修正——mocap 随机帧 → `default_dof_pos`（与 agent 站立位形一致），stand_ratio 混合逻辑保留。数据侧对齐仍有意义（即使非主因），且不与 label smoothing 冲突。

**不改**：disc_lr 保持 5e-5（label smoothing 已根治梯度消失，无需再压速度）；lerp 0.6 / 特征 61 维 / 缓冲 100 / 梯度惩罚 10 全部不动。

#### 3. 预期判据（it<500 定生死）

| 指标 | exp1/exp1.1 实测 | exp1.2 健康形态 |
| --- | --- | --- |
| disc_score (agent) | 30-86 iter 钉死 -0.995 | **稳定在 -0.7±0.15 波动，永不钉死** |
| disc_demo_score | +1.000 钉死 | +0.7±0.15 波动 |
| disc_loss | →0.0005 饱和 | **平衡于 0.05-0.3 区间不再下降** |
| style_reward | 0.0001 恒 0 | **it<300 出现 >0.01 且随训练爬升** |
| grad_penalty | →0 | 小正值持续 |

**升级止损**：it500 仍钉死 ±0.99 → label smoothing 失效 → exp1.3 方向：D 网络缩容（[1024,512]→[256,128]，削容量优势）或两阶段训练（前 1000 iter 冻结 D）。

#### 4. 修改文件

| 文件 | 改动 |
| --- | --- |
| `humanoid/algo/ppo/dh_ppo_amp.py` | disc loss 目标 ±τ（2 行）+ `amp_label_smooth` 参数 |
| `humanoid/algo/amp/amp_discriminator.py` | predict_style_reward 映射平移（3 行）+ tau 参数 |
| `humanoid/envs/x1/x1_dh_stand_env.py` | 静立窗 q_t → default_dof_pos（2 行） |
| `humanoid/envs/x1/x1_dh_stand_config.py` | `amp_label_smooth = 0.7`（1 行） |
| `scripts/tools/test_amp_disc.py` | 映射锚点断言 + 平滑后收敛测试更新（~15 行） |

#### 5. 训练参数

同 exp1.1（从零 / 6000 iter / seed 5 / commit 待推），run_name=exp1_2_amp。硬件建议：**等 083 基线出中期信号（it2500+，约 2h 后）再定**——若 task 锐化已能走，exp1.2 优先级可降（AMP=抛光）；若仍踏步，立即上 L20。

#### 6. 与 083 基线的归因关系

exp1.2 若成功（style>0 且步态自然），与 083（style≡0）的对比天然构成"AMP 有无贡献"的消融对——正好补上当初没买的消融，且同 task 锐化配置，归因干净。

### 12. 实验 exp1.2（实际执行）：§11 方案推翻 → exp0.2 底模 + AMP + 量纲修复（2026-09-07）

#### 1. §11 label smoothing 方案的推翻（未实施）

用户质疑"这样相当于强行让判别器迭代？"触发机制重审，自查发现两处硬伤：

- **公式算术错误**：声称的映射 `rew = clamp(1-(D-τ)²/(1+τ)²)`，D=-0.7 时实际 rew=0.32 而非 0（分母应为 (2τ)²）；锚点位置错误本身不致命，但暴露推导粗糙
- **机制漏洞（根本）**：τ=0.7 只是把 D 钉死点从 -0.995 挪到 -0.7——分布**平凡可分**时换 label 不改变"D 几步就找到完美分离面"的事实，style 梯度照样趋零。用户质疑成立：这是强行移动钉死点，不是解锁

#### 2. 方向转向：exp0.2 底模 + AMP（两阶段）

用户提出：exp0.2（有平移粗走）作底模再叠 AMP。三底模对比后确认：

| 底模 | 平移粗走 | AMP 适配性 | 结论 |
| --- | --- | --- | --- |
| exp0.2 | ✅ 有 | 手臂本就 mocap 塑形，agent 云贴近 demo 流形 | **选用** |
| exp0.3 | ❌ 原地踏步 | AMP 特征 61 维不含根线速度，D 区分不了踏步 vs 前行 | 弃 |
| 083 基线 | 未知（已停） | 纯 task 锐化产物，与 demo 距离同 exp1 问题 | 弃 |

分工假设：task 锐化治前扑超速（exp0.2 老毛病），AMP 治 bang-bang 平滑度（exp0.3 老毛病），各失败模式有主。

#### 3. 实施内容（4 文件）

1. **`--ckpt_path` 直连加载**（helpers.py + task_registry.py）：绕开 `--resume` 的 logs 目录扫描（云端挂载不适配）；`resolve_ckpt_path` 路径缺失时仓库内 glob `model_*.pt` 兜底（适配 checkPointMountPath 不确定性）；`load_optimizer=False` 微调语义；无 AMP 键旧 ckpt 自动跳过（exp0.2 的 DHPPO ckpt 兼容加载进 DHPPOAMP）
2. **静立窗位形修正**（x1_dh_stand_env.py）：mocap 随机帧 q_t → `default_dof_pos`（与 agent 站立位形一致，§10 归因①的正式修复）
3. **style 量纲修复**（x1_dh_stand_config.py）：`amp_style_reward_scale` 1.5 → **100**，见下

#### 4. 量纲淹没——三轮实验共同的隐藏根因（本轮最重要发现）

本地 resume 验证（64 env × 30 iter）发现 style reward 仅 0.001，追查公式量纲：

```
style = dt(0.01) × scale(1.5) × rew  →  上限 0.015/步
task（exp1 系列 FLAT config）         →  实测 O(6-15)/步
融合 = 0.6·task + 0.4·style          →  style 梯度比 task 弱 60 倍以上
```

**robolab 原版同样公式但 task 仅 O(0.8)/步（track 1.0 + 罚项），style:task 量级比比我们高 ~7 倍**——移植时公式抄对了，但量级背景没对齐。这解释了为何 exp1/exp1.1 中 policy 从未被 style 驱动：D 分离只是让 rew≈0.05，量纲淹没让即使 rew=1 也无感。前三轮一直在修 D 侧，真正致命的是奖励侧。

修复：scale=100（上限 1.0/步，d(style)/d(D)=0.5·(1-D)·1.0 与 task 梯度 O(1) 同量级；乘 dt 保留，控制频率解耦设计不变）。

#### 5. 本地验证（exp0.2 model_6000.pt 底模，64 env）

| 组 | iter | ep_len | mean_reward | style | agent/demo score |
| --- | --- | --- | --- | --- | --- |
| scale=1.5 | 30 | 280 | 6.06 | 0.001 | -0.94 / 0.97 |
| scale=100 | 60 | 543 | 15.1 | **0.054** | -0.98 / 0.98 |
| scale=1.5 对照 | 60 | 545 | 15.2 | 0.001 | -0.97 / 0.98 |

判读：

- **ckpt 加载 ✅**：iteration 6028/6029 确认载入，`--max_iterations` 增量语义正确
- **量纲修复生效 ✅**：style 0.001→0.054（×54，精确符合 rew≈0.055 预测）
- **无破坏性 ✅**：ep_len/reward 与对照完全一致（ep_len 翻倍是 task 锐化适应过程，与 style 无关——对照组同样翻倍）
- **D 分离依旧 ⚠️**：agent score 60 iter 内未上移。exp0.2 粗走（bang-bang）vs mocap（平滑）在 183 维空间仍可分——这是分布事实，scale 修复只保证梯度存在，牵引效果需云端长程观察

#### 6. 云端计划与判据

4096 env（梯度噪声比本地小 64 倍）× 数千 iter，观察：

- **健康形态**：agent score 缓慢上移（-0.98 → -0.8 …），style reward 爬升，ep_len 不崩
- **失败形态**：agent score 钉死 -0.98 且 style 停在 ~0.05 不动 → 分布不重叠假说成立 → exp1.3 方向：D 网络缩容 / 特征降维（去 dof_vel 只留 dof_pos）/ demo 侧混入 agent 噪声增强

### 附：决策点（已拍板 2026-09-07）

1. **ref_joint_pos 2.4→0（推荐）vs 0.5 过渡**：推荐 0——踏步白拿漏洞必须在源头堵死；若担心风格突变过大，可 0.5 但接受归因混杂
2. **从零（推荐）vs 续训 exp0.3**：续训收敛快（policy 已会站）但 action 分布已收敛、noise_std 低，AMP 新梯度注入效果存疑且归因混杂；从零干净
3. **是否并行消融任务**（账号7/8 各 ¥50 可用）：amp=True 主实验 + amp=False（纯 task 锐化基线）各一任务并行——多花一份钱，买"AMP 是否真有贡献"的干净归因；不并行则失败时再补跑消融
