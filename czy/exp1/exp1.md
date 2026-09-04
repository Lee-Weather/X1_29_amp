# 实验记录（X1_29_amp · 29DOF 全身控制）

> 本文件为 29DOF 项目实验记录（lab-notebook 规范）。
> 12DOF 旧项目（X1_29_re0）完整历史见 [exp1_12dof_legacy.md](exp1_12dof_legacy.md)（exp0~exp1.4，其结论与教训作为本系列先验）。

## 实验索引

| 编号 | 日期 | 摘要 | 状态 | Task ID | GM账号 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| exp0 | 2026-09-02 | 29DOF 全身控制基线：env 腿部按名索引 + config 29 维（obs 98/action 29/priv 141）+ 29DOF PM URDF + 上半身默认位姿锁定，从零 L4 训练至 5800 轮额度耗尽；回放摔倒（min_h 0.092m）+ 严重过冲（0.4 段 286%）+ 停不住，站立段完美 | ❌未达标（已测试） | TASK_20260902_185(停)→186 | limxmtcm6wjlso8ce4@emalupe.com（账号3，已耗尽） | model_5800.pt |
| exp0.2 | 2026-09-03 | Phase 2b mocap 参考行走：ref_lib.pt 三段（0000/0002/0026，50Hz）全身查表（腿臂同源同拍）+ 逐段步频相位 + URDF 右臂限位镜像修复；本机先训（从零 ~1600 iter 形态健康）→切云端 L20+L4 双任务并行（被手动停）→换账号5 L4 重训；回放摔倒 ~4 次/40s + 指令跟随差（cmd=0 自走 0.5m/s） | ❌未达标（已测试） | TASK_20260904_006(L20,停)/007(L4,停)/008(L4·账号5)/073(回放) | limxmtjqbym1pg0fra@emalupe.com（账号5） | model_6000.pt |
| exp0.3 | 2026-09-04 | 根因导向微调（不用 AMP）：压动作幅度（action_scale 0.3+smoothness×2.5+clip 3）治 bang-bang 前扑 + gait 调度改出生/结尾站立治停不住 + ref_joint_pos 加压制参考架空 | 训练中 | TASK_20260904_086(L4·账号6) | limxmtjqd2kli2rjom@emalupe.com（账号6） | — |

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

待训练完成后补充。
