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
| exp1.2 | 2026-09-07 | §11 label smoothing 方案推翻（换 label 不解平凡可分）→ **exp0.2 底模 resume + AMP**：新增 --ckpt_path 直连加载；静立窗 default_dof_pos；**发现量纲淹没隐藏根因**（style 上限 0.015 vs task O(6)，梯度弱 60 倍）scale 1.5→100。训练全程监控"健康"（score 收窄/style 爬升/reward 翻倍）但**回放判死：0.4/0.6 指令完全冻结，行走能力被拆**（对照底模同流程会走）——站立成为新奖励面+静立窗 style 的 net 最优，监控三绿是站立体化假阳性（§12.7） | ❌失败（新失败模式） | TASK_20260907_113(新账号) | limxmtjqfkh52btio6 | model_11999.pt |
| exp1.3 | 2026-09-08 | 拆三个"站立补贴"（stand_ratio→0 / low_speed 1.0→2.0 / ref_joint_pos 0→0.5）+ exp0.2 底模续训。**行走保住**（tracking 0.5+/reward ~104/回放确认在走，"防拆"目标达成，commit 5dd55da）但 **D 死锁回归**：agent score -0.994 钉死、style 0.026——良性死锁，AMP 通道退化为旁观者；分布指纹定位分离面主成分=手臂高频抖动（§14） | ⚠️部分达标（AMP 死锁） | TASK_20260908_241(跑着当 task 锐化基线) | limxmtrzffpgvnsh9w@uberip.com（账号池[1]） | — |
| exp1.4 | 2026-09-08 | 手臂/腰部 17 关节 action **EMA 低通滤波**（α=0.85，fc≈2.8Hz）：频率维度治本——物理消除手臂高频抖动（D 分离面主成分），真实移动 agent 分布（非缩 D 容量）。本地 64env×60iter 快测通过（含 pip install -e . 重装规范确立）。resume exp0.2 model_6000 + 6001 iter（终点 12001），**双线策略**：AMP 主线（本任务）+ 无 AMP 基模线（另一项目，好底模回流作 resume 源） | 🚧训练中 | TASK_20260908_319 | limxmtrzffpgvnsh9w@uberip.com（账号池[1]） | 目标 model_12001.pt |
| exp1.5 | 2026-09-08 | exp1.3 底模（model_12000，实质=纯 task 基模）+ AMP 破死锁三修改：**① agent buffer 三重门控**（主攻：行走&未终止&ep_len>50 剥掉站立/摔倒/复位送分样本——核对新发现 exp1.3 死锁头号嫌疑：gait 26% 站立段样本 vs 100% 行走 demo，"速度幅度"一维秒分，与 exp1 的 demo 静立窗互为镜像）**② D 从零初始化**（--disc_fresh 拆死锁先验）**③ style 负斜坡下界**（eps=0.05 防过冲死区，保险丝）。resume 快测验证：healthy 0.6 / D fresh score -0.90 / style 0.18（exp1.3 死锁 0.026）。云端任务已建：TASK_20260908_339（4090，exp1.3 model_12000 底模 + --disc_fresh，iter 12000→18000，commit 733beb6） | 🚧训练中 | TASK_20260908_339 | limxmtrzg1znyycbwp@uberip.com（账号池[2]） | 目标 model_18000.pt |

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

#### 7. 实验结果（2026-09-08，训练完成 + 回放判读：❌失败——新失败模式"行走能力被拆"）

训练：TASK_20260907_113（新账号 limxmtjqfkh52btio6，L4，底模 model_6000 → model_11999，增量 6000 iter 约 5.8h）。

**训练中监控（全程"健康"形态，事后证明为假阳性）**：

| iter | reward | ep_len | agent score | demo score | style | d_loss |
| --- | --- | --- | --- | --- | --- | --- |
| 6111 | 57.6 | 1686 | -0.971 | 0.968 | 0.107 | 0.020 |
| 7581 | 84.1 | 2210 | -0.935 | 0.934 | 0.194 | 0.031 |
| 8226 | 96.4 | 2190 | -0.908 | 0.904 | 0.241 | 0.051 |

score 差距 1.87→1.81 持续收窄、d_loss 0.02→0.05（D 越来越难分）——当时判读为"policy 分布靠近 demo"。**真相：靠近的是静立/微动分布，不是行走分布**。

**回放判读（model_11999，play 速度阶梯 0→0.4→0.6→0，174 列诊断 CSV）**：

| 段 | cmd | 实际vx | Δpos_x | 脚z幅 | action RMS | exp0.2 底模对照（9/4 同流程回放） |
| --- | --- | --- | --- | --- | --- | --- |
| 站立0 | 0 | 0.006 ✅ | 0.028 | 0.09 | 0.117 | 自走 0.266（停不住） |
| 前进0.4 | 0.4 | **0.000 ❌** | 0.000 | **0.000** | 0.065 | **0.861（超速215%，在走）** |
| 前进0.6 | 0.6 | **-0.001 ❌** | -0.003 | **0.000** | 0.048 | **1.102（超速184%，在走）** |
| 停止0 | 0 | 0.001 ✅ | 0.004 | 0.000 | — | 0.425（停不住） |

**核心发现：行走能力被完全拆掉**。0.4/0.6 指令下 Δpos=0、双脚 z 幅=0、膝 vel≈0.03——policy 输出冻结；且指令越大 action 幅度越小（0.117→0.065→0.048，"大指令→僵住"映射）。对照 exp0.2 底模同流程回放（超速但在走、明显迈步），确认行走能力是 resume 训练期间丢失的，非回放环境差异。

**根因链（推定，待云端回放复核）**：

1. **静立窗混入的合谋**：demo 侧按 stand_ratio 混 default_dof_pos 静立窗（exp1.1 修复保留）→ D 被教会"静立窗也算 demo"→ policy 站立/微动同样能吃 style 分。score 收窄有一部分是"静立匹配"假象
2. **站立成为新奖励面的 net 最优**：行走 = 能量罚（action_smoothness/torques）+ 真走暴露 bang-bang 被 D 打低分（style→0）+ 超速罚风险；站立 = style 静立分 + 省能量 + ep_len 长（bootstrapping 收益）。σ20 锐化下 tracking 4% 损失远小于上述收益
3. **resume 奖励面迁移**：exp0.2 的行走是旧奖励面（σ 旧值 + ref_joint_pos 2.4）训出的；resume 后奖励面大改（σ20 + low_speed -2 + ref_joint_pos 0 + AMP style），旧行为在新区间持续失血，6000 iter 足以把它拆干净

**教训（回答"好基模加强 AMP"的边界）**：好基模提供分布重叠区，但 **AMP 只放大"基模与 demo 已有的相似方向"，不保护基模能力**——当 style 梯度方向（静立匹配）与 task 需求（行走保持）冲突时，AMP 会主动拆掉行走。两阶段方案里底模与 AMP 的 demo 侧预处理（静立窗混入）必须联合设计。

**监控判据修订（下轮起用）**：训练监控必须加回放抽查——reward↑/score↑/style↑ 三绿不能证明行走保留；看 `rew_feet_air_time`、`feet_contact_number`（行走应 ~1.0-1.3 且摆动相存在）与 `rew_low_speed`（walk 段大额负值 = 没在走）。

三件套：`czy/data/exp1.2/{model_11999.pt, play_output.mp4, isaac_diag.csv}`。

### 13. 实验 exp1.3 方案：拆除三个"站立补贴"——AMP 奖励真空原则（2026-09-08，待审批）

#### 1. exp1.2 冻结根因的完整账本（why standing wins）

行走 env 中"站立不动"的每步收支（终期日志反推）：

| 收入项 | 站立所得 | 说明 |
| --- | --- | --- |
| tracking_ang_vel | ~+0.5 满分 | cmd yaw=0、站立 yaw=0 → exp(0)=1 |
| 姿态类（orientation/base_height/feet_dist/knee_dist/contact_number/vel_mismatch） | ~+2.9 | 站立全额领取 |
| AMP style | +0.4×0.141≈**+0.06** | **demo 混 26% 静立窗 → D 认可静立 = demo** |
| tracking_lin_vel（σ20, err=0.4） | +0.04 | 全或无，站立与超速同得 4% |
| low_speed（too_slow -2×1.0） | -2.0 | 唯一惩罚 |
| **净额** | **≈ +1.5/步** | only_positive_rewards=True 兜底 ≥0 |

对照 exp0.2 超速步态在新奖励面下的收支：tracking≈0.04（σ20 全或无）+ too_fast **-1**（比 too_slow 轻！）+ 能量罚 + style≈0.02（bang-bang 被 D 打低）——**旧行为收入崩塌至 ~1/步，低于站立 1.5/步**。排名翻转后 6000 iter 滑入冻结盆地，且冻结自锁：不摆腿 → feet_air_time(权重1.2) 永远为 0 → 无摆腿梯度。

**三个 robolab 没有的"站立补贴"**：① gait 调度 26% 站立段 + stand_still 3.5（练站立技能本身没错，但占了 1/4 训练时间在排练冻结）② demo 静立窗混合 → AMP 给冻结发 style 工资 ③ only_positive_rewards 吸收惩罚（-2 被正收益质量抵消）。robolab 的 demo 100% 行走、无站立段、无 stand_still、无静立混合——**AMP 在"运动奖励真空"里工作，没有第二个收入来源**。

#### 2. 修改内容（3 文件，~10 行）

| # | 修改 | 位置 | 旧→新 | 理由 |
| --- | --- | --- | --- | --- |
| 1 | **stand_ratio → 0**（demo 纯行走） | env `_sample_amp` 调用处 | 按站立占比混合 → 固定 0.0 | 拆 AMP 冻结工资：纯行走 demo 下站立 style≈0.05、真步态可达 1.0，AMP 恢复"只奖励走路" |
| 2 | **low_speed 权重 1.0→2.0** | config rewards | too_slow -2×1.0 → ×2.0=-4 | -4 > 站立正收益 3.4 → walk env 站立净额触 0 钳位，行走（~2.8-4.6）明确胜出；保持 only_positive_rewards 不动（稳定性） |
| 3 | **ref_joint_pos 0→0.5** | config rewards | 0 → 0.5 | exp1 以"AMP 替代它"为由删除，实测 AMP 标量太粗保不住姿态；恢复半值作逐关节密集锚，防迁移期步态崩解（exp0.2 原值 2.4） |

不改：lerp 0.6 / scale 100 / σ20 / gait 调度占比 / stand_still 门控（已正确按指令门控）/ feet_air_time 1.2（摆腿恢复后自然生效）。resume 源：**exp0.2 model_6000**（不是 exp1.2 的冻结 ckpt）。

#### 3. 预期判据（冻结检测器，it<300 定生死）

| 指标 | exp1.2 冻结形态 | exp1.3 健康形态 |
| --- | --- | --- |
| rew_feet_air_time | ≈0.0002（不摆腿） | **it<300 回到 >0.1 且爬升** |
| rew_low_speed | -0.8（吸收后残值） | >0 或小负（too_slow 消失） |
| AMP style (walk) | 0.14（静立匹配假象） | 先跌至 ~0.05 再随步态塑形爬升 |
| agent score | -0.91（静立匹配） | 与 style 同步的真实上移 |
| **中途回放**（it+500，本地 64env 快测） | — | 0.4 段 vx>0.2 即续跑 |

**止损**：it500 feet_air_time 仍 <0.01 → 冻结未解 → exp1.4 方向：行为级 style 门控（仅摆腿 env 吃 style）/ 站立段占比压 15% / σ 迁移课程。

#### 3b. 最终结果与回放验收（2026-09-08，TASK_20260908_241 完成后）

训练终点 model_12000（status=5）。本地回放（速度阶梯 0→0.4→0.6→0 各 500 步，`czy/data/exp1.3/` 三件套）：

| 段 | cmd | Δx(m) | 摆腿幅(m) | actRMS |
| --- | --- | --- | --- | --- |
| 静立 | 0.0 | 0.01 | 0.095 | 0.369 |
| 行走 | 0.4 | 0.76 | 0.349 | 1.420 |
| 行走 | 0.6 | 2.60 | 0.053 | 1.448 |
| 停止 | 0.0 | 0.36 | 0.017 | 0.659 |

**exp1.2 冻结三特征全部反转**（有位移/在摆腿/actRMS 随指令放大），"行走能力被拆"根除——但代价是 D 死锁（§14）。遗留：速度跟随偏慢（0.4 段 ~0.04 m/s、0.6 段 ~0.13 m/s 远低于指令，0.6 段摆腿塌 0.053），高速段步态未立起，归 task 侧后续锐化。

#### 3c. 对照 exp0.2 底模回放（同阶梯判读 + 逐关节组 vel std）

| 段 | cmd | exp0.2 Δx(m) | exp1.3 Δx(m) | exp0.2 摆腿幅 | exp1.3 摆腿幅 |
| --- | --- | --- | --- | --- | --- |
| 静立 | 0.0 | 1.42（漂移） | **0.01** | 0.094 | 0.095 |
| 行走 | 0.4 | **-0.52（倒退）** | **+0.76** | 0.273 | 0.349 |
| 行走 | 0.6 | **-0.91（倒退）** | **+2.60** | 0.291 | 0.053（塌） |
| 停止 | 0.0 | 2.21（滑行） | **0.36** | 0.064 | 0.017 |

行走段（0.4+0.6）逐关节组 vel std，exp1.3/exp0.2 倍率：lumbar 0.89x / **左臂 0.34x / 右臂 0.35x** / 左腿 0.47x / 右腿 0.41x。

**exp1.3 相对 exp0.2 的四项改进**：① 指令方向跟随由负转正（0.4/0.6 段从倒退 0.5~0.9m 变前进 0.8~2.6m）② 停得住（初始漂移 1.42→0.01m、停止滑行 2.21→0.36m）③ 全身动作更安静（vel std 降至 1/3~1/2，手臂最明显；注意训练期指纹 3.2~10.1x 是带探索噪声的分布，回放为确定性策略，两口径不同）④ 0.6 段实际速度 0.045→0.130 m/s。**退步一项**：0.6 段摆腿幅 0.291→0.053，高速段步态未立起（速度绝对值跟踪仍远低于指令）。

#### 4. 修改文件

| 文件 | 改动 |
| --- | --- |
| `humanoid/envs/x1/x1_dh_stand_env.py` | `_sample_amp` 调用 stand_ratio 固定 0.0（1 行） |
| `humanoid/envs/x1/x1_dh_stand_config.py` | low_speed 1.0→2.0、ref_joint_pos 0→0.5（2 行） |
| `czy/exp1/exp1.md` | 本节（方案存档） |

### 附：决策点（已拍板 2026-09-07）

1. **ref_joint_pos 2.4→0（推荐）vs 0.5 过渡**：推荐 0——踏步白拿漏洞必须在源头堵死；若担心风格突变过大，可 0.5 但接受归因混杂
2. **从零（推荐）vs 续训 exp0.3**：续训收敛快（policy 已会站）但 action 分布已收敛、noise_std 低，AMP 新梯度注入效果存疑且归因混杂；从零干净
3. **是否并行消融任务**（账号7/8 各 ¥50 可用）：amp=True 主实验 + amp=False（纯 task 锐化基线）各一任务并行——多花一份钱，买"AMP 是否真有贡献"的干净归因；不并行则失败时再补跑消融

### 14. 实验 exp1.4：手臂/腰部 action EMA 低通滤波——频率维度治本（2026-09-08，已实施未提交）

#### 1. 上一实验（exp1.3）结果：良性死锁

TASK_20260908_241（exp0.2 底模 + 三修改续训）监控判读：

| 侧面 | 指标 | 数值 | 判读 |
| --- | --- | --- | --- |
| task 侧 | tracking_lin/ang、reward、回放 | tracking 0.5+、reward ~104、**在走** | **exp1.3 核心目标"防行走被拆"达成**（对照 exp1.2 冻结） |
| AMP 侧 | agent score / style (walk) | **-0.994 钉死 / 0.026** | D 死锁回归，与 exp1/exp1.1 同形态 |

结论：行走能力保住了，但 style 通道死了——AMP 退化成旁观者，花 D 的算力买不到任何风格梯度。exp1.4 的任务：**在不牺牲 task 侧健康的前提下复活 AMP**。

#### 2. 根因定位：分布指纹逐关节解剖（决定性数据）

对 exp1.3 期间 agent 轨迹 vs mocap demo 逐关节算 `dof_vel` std 倍率（agent/demo）：

| 关节组 | dof_vel std 倍率 | 判读 |
| --- | --- | --- |
| 右臂（肩 roll/yaw、肘 pitch 最重） | **7.8x ~ 10.1x** | 抖动重灾区 |
| 左臂 | ~3x | 同样超速但轻一半 |
| 腿部均值 | 1.8x | 轻微 |
| 右膝 / 左膝 | **1.0x / 0.6x** | **完美匹配**——腿部动作本身就是 demo 风格 |

**D 的 183 维特征里（61×3 步窗），分离面主要由 10 个手臂关节的高频抖动支撑**。腿部已经"像 demo"，D 不需要靠腿区分两边；手臂一抖，D 一个线性组合就能满分分类。

机理：ref_joint_pos 从 2.4 归零（exp1.3 回 0.5 半值）后手臂没有任务老师 → 手臂成**噪声海绵**（PPO noise_std 全关节共享 ~0.21 + 腿部梯度经共享网络传播到手臂头）。右侧 10x vs 左侧 3x 的不对称叠加 URDF 右臂历史问题。

**为什么 exp1.3 的 ref_joint_pos 0.5 救不了**：L2 罚的是位置误差，高频小幅抖动的位置误差极小（抖 ±2° 误差贡献可忽略），但对 dof_vel 特征是 std 直接放大——**位置类奖励在频率维度是瞎的**。历史证据：exp0.2 时代 ref_joint_pos 2.4 全身工作时手臂照样抖。

#### 3. 方案：EMA 低通滤波（治本路线）

方向决策（用户拍板）：**绝不放弃 AMP；缩 D 容量是让 D 变笨，治标不治本**。治本 = 改变 agent 生成侧分布，让抖动这个分离面主成分物理消失：

- **滤波公式**：`filt = α·prev + (1-α)·raw`，作用于手臂+腰部 17 关节（lumbar×3 + shoulder/elbow/wrist×14）的 action
- **α=0.85**：fc = (1-α)/(2πα·dt)，dt=0.01（100Hz 控制）→ **fc≈2.8Hz**、群延迟 ≈0.057s。（更正：§13 讨论中口算的 1.6Hz 有误，按公式为 2.8Hz。）人类手臂摆动主频 <2Hz，2.8Hz 截止保信号滤噪声
- **PPO 一致性**：policy 采样 log_prob 在原始 action 上（滤波在 env.step 内、采样之后），obs 的 last_action 用滤波后值（滤波在 super().step() 的 clip 之前写入 actions）→ 观测与实际执行一致；EMA 凸组合保证输出不越 [-clip, clip]
- **底模 resume 风险**：exp0.2 底模手臂未经滤波训练，突然加滤波手臂行为会突变——手臂质量小不威胁平衡，且 ref_joint_pos 0.5 提供位置锚，可接受
- **判据逻辑**：D 的分离面主成分消失后，D 无法再用"手臂抖动"分类 → agent score 解冻 → style 梯度恢复 → AMP 重新参与步态塑形

#### 4. 修改内容（2 文件 4 处）

| # | 文件 | 改动 |
| --- | --- | --- |
| 1 | `x1_dh_stand_config.py` control 类 | 新增 `arm_action_ema_alpha = 0.85`（含 fc/延迟注释；α=1.0 关闭滤波） |
| 2 | `x1_dh_stand_env.py` `_init_buffers` | arm dof 索引按名解析（lumbar/shoulder/elbow/wrist，硬断言 ==17）+ `_arm_action_filt` 状态 buffer + [EMA] 启动打印（α/fc/索引） |
| 3 | `x1_dh_stand_env.py` `step()` | ref_action 叠加后、super() 前：17 关节 action EMA 滤波并写回 |
| 4 | `x1_dh_stand_env.py` `reset_idx` | `self._arm_action_filt[env_ids] = 0.`（与 actions 复位对齐） |

#### 5. 本地快测（2026-09-08，64 env × 60 iter）

- exit 0，60/60 iter，[EMA] 打印 `alpha=0.85, fc≈2.8Hz, dofs=[6..22]`（lumbar 6-8 + 左臂 9-15 + 右臂 16-22，共 17），AMP 管线每 iter 正常输出，无 NaN/断言
- 从零训练 60 iter 内 agent score -0.99 属预期（随机策略 D 必然秒分），EMA 效果只能在云端底模续训上判
- **工程坑（重要，规范确立）**：第一次 64×60 快测跑的是**错误代码**——`python humanoid/scripts/train.py` 的包解析依赖 egg-link，而 `python -c` 验证因 cwd 在 sys.path 总是命中本仓库、失真。重装前 egg-link 指向旧 checkout（12DOF、无 AMP），表现为无 [EMA]/[AMP] 打印、Actor out_features=12。**规范：每次本地测试前先 `pip install -e .`**（用户指示，2026-09-08）

#### 6. 预期判据（云端，exp0.2 底模续训）

| 指标 | exp1.3 死锁形态 | exp1.4 健康形态 |
| --- | --- | --- |
| AMP agent score | -0.994 钉死 | **解冻（>-0.9 且持续上移）** |
| AMP style (walk) | 0.026 | 随步态塑形爬升（>0.1 量级） |
| 手臂 dof_vel std 倍率（回放 CSV） | 3.2~10.1x | **<2x** |
| 行走指标（tracking/feet_air_time/reward） | 健康 | **不倒退**（exp1.3 水平之上） |

**止损**：若 D 仍死锁 → 分离面不止手臂，用回放 CSV 重算全身倍率定位新主成分；备选：加大 α（0.9 → fc≈1.8Hz）/ 扩展滤波范围。

#### 7. 风险与预案

| 风险 | 评估 | 预案 |
| --- | --- | --- |
| lumbar 滤波影响平衡补偿 | 腰部参与平衡，0.057s 延迟或拖慢响应 | 若平衡退化，把 lumbar 移出滤波（改索引断言 14） |
| 手臂群延迟 0.057s | 手臂质量小、无平衡职责 | 可接受 |
| EMA 滤掉有用的高频手臂动作（如摆臂配重） | mocap 手臂主频 <2Hz，2.8Hz 截止在其上 | 若 style 爬升但手臂"僵"，降 α 至 0.8（fc≈4Hz） |

### 15. 实验 exp1.5：exp1.3 底模 + AMP 破死锁三修改（2026-09-08，已实施未提交）

#### 1. 为什么可以用 exp1.3 作底模（决策依据）

- **exp1.3 实质=纯 task 基模**：D 死锁、style 恒 0.026，6000 iter 里 AMP 通道零有效贡献，全部进步来自 task 奖励面——用它的意义与"基模线回流好底模"完全同构（双线策略的现成实例）
- **奖励面一致性**：已在当前奖励面（low_speed 2.0 / ref_joint_pos 0.5 / stand_ratio 0）下收敛，无 exp0.2 底模的"奖励面突变"迁移期，EMA 与 D 互动观察窗口干净
- **底模更强**：回放四项改进（§13.3c），腿部已近 demo 风格（膝 1.0x）

#### 2. 死锁闭环结构与 EMA 的局限

```
D 找到稳定分离面 → disc loss→0 停更新
   ↓
D_agent≈-1 → style≈0.006（淹没级）→ policy 分布不变
   ↓
分离面继续有效 → 闭环自锁
```

EMA（exp1.4）只摧毁旧分离面（手臂抖动），**重置时钟而非拆环**——D 会找新分离面再钉死。exp1.5 必须拆环。

#### 3. 实施中的新发现：agent buffer 站立样本污染（exp1.3 死锁头号嫌疑）

核对 `dh_ppo_amp.process_env_step` 时发现：agent 侧每步**全部 env 样本无差别入 buffer**——exp1.3 的 stand_ratio=0 只改了 demo 侧，agent 侧 gait 调度 26% 站立段 + 复位静止窗照常入队，对阵 100% 行走 demo。**"速度幅度"一维秒分的 exp1 死锁根因以镜像形式回归**（exp1：demo 有静立窗 agent 没有；exp1.3：agent 有站立样本 demo 没有）。此前手臂抖动指纹分析在行走段回放做的（排除站立），只看到手臂——训练期 D 眼里的第一分离面很可能就是站立样本本身。

#### 4. 三修改（对闭环三环各补一刀）

| # | 修改 | 拆哪环 | 实现 | 定位 |
| --- | --- | --- | --- | --- |
| 1 | **agent buffer 三重门控** | 数据不对称（D 的分离面来源） | `healthy = (~stand) & (~done) & (ep_len>50)`；`CircularBuffer.append_masked`（不健康 env 槽位保留上一行旧样本）；demo 侧不门控 | **主攻** |
| 2 | **D 从零初始化** | D 先验（死锁权重=作弊记忆） | `--disc_fresh` → `load(skip_disc=True)` 跳过 D state_dict（归一化统计量随之重积累） | 必做 |
| 3 | **style 负斜坡下界** | policy 死区 | `rew = max(1-(D-1)²/4, 0.05·(D+1))`；D∈(-1,1) 与旧公式逐点一致（量纲零扰动），D<-1 给恒定 eps 梯度且越负越罚 | 保险丝 |

修改 1 的镜像修法优于 exp1.1：exp1.1 曾"demo 侧加静立窗"失败于位形错配；此番"agent 侧去站立"无需手造数据，站立样本本就不该参与行走风格判别。

**修改 3 的诚实定位修正**：细算发现 exp1.3 死锁值 -0.994 在梯度区（真死区是 D<-1，rew 对 D 梯度 0.997 非零），死锁本质是"信号淹没"（rew≈0.006 vs task O(1)）而非"梯度为零"——修改 3 防的是 D 过冲真空区 + 连续负反馈，不是主攻。

#### 5. 修改文件与参数

| 文件 | 改动 |
| --- | --- |
| `amp_discriminator.py` | `style_floor_eps` 参数 + 负斜坡公式 |
| `amp_buffers.py` | `append_masked(data, mask)` |
| `dh_ppo_amp.py` | `amp_style_floor_eps`/`amp_buffer_min_episode_len` 参数、configure_amp 传参、process_env_step 门控 + healthy 监控 |
| `dh_on_policy_runner.py` | `load(skip_disc)` + `AMP buffer healthy` 日志行 |
| `task_registry.py` / `helpers.py` | `--disc_fresh` CLI 透传 |
| `x1_dh_stand_env.py` | `extras["amp"]["episode_length"]` |
| `x1_dh_stand_config.py` | `amp_style_floor_eps=0.05`、`amp_buffer_min_episode_len=50`（-1=关闭） |

#### 6. 本地快测（2026-09-08，两轮，含重要分布差异发现）

**从零 64env×60iter**：healthy 恒 0 → 定位为**从零快测与底模续训的分布差异**而非 bug——gait 出生站立段 [3,5]s（exp0.3 设计）≈300+ 步，从零 episode 均长仅 91 步（<1s 摔完），**活不出出生站立段** → 全程站立指令 → 门控剥光。副作用验证了门控逻辑正确（该场景下确实没有行走样本）。

**exp1.3 底模 resume 64env×40iter + --disc_fresh**（真实场景验证，全过）：

| 指标 | exp1.3 死锁形态 | exp1.5 快测 | 判读 |
| --- | --- | --- | --- |
| AMP buffer healthy | —（无此监控） | **0.57~0.63** | 门控工作，落在预期 0.5~0.7 |
| agent / demo score | -0.994 / 0.998 双钉死 | **-0.90 / 0.92** | D 从零且被持续挑战（disc fresh 生效） |
| disc loss | 0.0013 | **0.035~0.04** | D 不再秒分 |
| style reward (walk) | 0.026 | **0.17~0.19** | style 通道复活 |
| episode 均长 / reward | — | 440→540 / 19→25 | 底模在走且变好 |

#### 7. 云端任务参数（待用户指令创建）

- 底模：`czy/data/exp1.3/model_12000.pt`（上传 OSS 后挂载）
- startScript：`gm-run X1_29_amp/humanoid/scripts/train.py --task=x1_dh_stand --run_name=exp1_5_break_lock --headless --seed=5 --ckpt_path=X1_29_amp/model_12000.pt --disc_fresh --max_iterations=6000`（iter 12000 → 终点 18000）
- 账号：账号池[2] limxmtrzg1znyycbwp@uberip.com（已切换登录）；算力 4090（ESKU000001）

#### 8. 监控判据（升级版，不可与 exp1.3 绝对值直接对比）

| 指标 | 死锁形态 | 健康形态 |
| --- | --- | --- |
| disc loss | →0.001 | **0.3~0.7 平台**（D 持续被挑战） |
| agent / demo score | 双钉死 ±0.99 | **双方 0.7~0.9 震荡** |
| style (walk) | 0.026 | 爬升 >0.3 且**跨 env 方差 >0** |
| buffer healthy | — | 0.5~0.7（趋 0 = 门控过严/episode 过短） |
| 行走指标 | exp1.3 水平 | 不倒退 |

**止损**：it 500 内 disc loss 仍 →0 + 双 score 钉死 = 新分离面存在 → 用回放 CSV 重算全身逐关节倍率定位（此时门控已剥站立样本，指纹将是纯行走段差异）；下一张牌：demo 侧噪声匹配（训练期 agent 探索噪声 vs 干净 demo 的结构性可分维）。

#### 9. 云端结果：完全未收敛 + 根因实锤（2026-09-09 诊断）

**339（exp1.5，exp1.3 底模 + 三修改 + EMA）**：status=5 跑完 6000 iter，tracking_lin_vel 从加载 iter 12000 起 ≈0 全程为 0；healthy 全程 0.000（episode ~53 < 门限 50 挡死）；reward 0.1；D 双钉死；style 0。回放 model_17999 实锤**原地高频震荡冻结**：四段 Δx 全 ≈0（0.06/-0.00/-0.04/-0.22），actRMS 2.95（exp1.3 对照 0.37~1.45）。

**319（exp1.4，exp0.2 model_6000 底模 + 仅 EMA，账号[1]，额度不足终止于 iter 8202）**：tracking 首 100 iter mean=0.0022（峰值 0.008@6032）→ 全程 0.0006；episode 193→~100；reward 4.75→0.085；style 恒 0.002~0.009。

**A/B 对照（同底模 exp0.2 model_6000，单变量=EMA）**：

| 指标 | 241（exp1.3，无 EMA） | 319（仅 EMA） |
| --- | --- | --- |
| tracking 首 100 iter / 全程 | 0.071 / **0.503** | 0.0022 / **0.0006** |
| episode 首 100 iter / 全程 | 677 / **2060** | 193 / **~100** |
| reward 全程 | **102** | 0.24→0.085 |

**根因结论：env 侧 EMA 滤波是 319/339 双失败的共同主因**，与底模（exp0.2/exp1.3）和三修改无关。机理两层：

1. **resume 冲击（执行层）**：底模在"原始手臂/腰部动作"动力学下训练收敛，EMA α=0.85 等效更换执行器——episode 开始 filt=0，首步执行动作 ≈0.15×raw（手臂/腰部被拽向零位），此后永久 0.057s 滞后。腰部参与平衡、手臂摆动参与配重，策略学到的耦合反馈环被打断 → 加载即摔（episode 1~2s）→ 短 episode 陷阱。
2. **PPO 一致性破坏（学习层）**：storage 记录的是采样原始动作的 log_prob，env 实际执行滤波后动作，而 obs.last_action 又是滤波值——(s,a)→s' 从学习器视角变成历史依赖的"随机"映射，advantage 估计持续偏置，6000/2200 iter 都无法恢复。

下游效应自洽：加载即摔 → episode <50 → 339 门控 healthy=0（agent buffer 只剩摔倒样本）→ fresh D 秒分"摔 vs 走" → 钉死 → style 0。D 钉死是冻结的结果而非原因。

**本地快测为何没暴露**：64env×40iter 的小批量更新（1536 样本/更新 vs 云端 ~10 万）几乎不移动策略，底模行走惯性保留 → "在走"的假象；云端大批量更新在破碎执行模型上快速摧毁策略。教训：**改变执行链路（滤波/延迟/clip 语义）的改动，本地小步快测无判别力**。

**exp1.6 方向（待拍板）**：EMA 撤出执行链路（`arm_action_ema_alpha=1.0`），三修改保留，exp1.3 model_12000 底模重跑 339 同款配置——给三修改一次干净的云端检验；手臂抖动分离面若回归，改用 reward 侧手段（手臂关节 action-rate/dof_acc 惩罚，PPO 一致性天然成立）而非 env 侧滤波。

### 16. 实验 exp1.6 预备：yz 真实地面行走轨迹接入（2026-09-08，已实施未提交）

#### 1. 动机：demo 库速度覆盖的结构性缺陷

原三段 ref 库（x1_gmr 跑步机 retarget）均速 0.09 / 0.11 / 1.23 m/s——**库中没有 0.12~1.2 m/s 区间的样本**，而回放 0.4 指令段实速仅 0.04、0.6 段 0.13 m/s（exp1.3 回放），demo 无法示范"中速行走"。yz 轨迹（`czy/diff/yz/07_03_walk_*.csv`，真实地面直线行走 retarget）体检：415 帧@30Hz、均速 0.255 m/s、左膝 std 0.416（walk_norm 的 1.9 倍——摆腿充分）、帧间跳变 ≤0.064 rad、root z [0.612,0.633]、无越限——**正好补速度空洞**。

#### 2. 实施中发现的数据特性（两次修正）

- **真实步态周期 4.78s 不是 1.2s**：左膝自相关最近峰在 4.87s（左髋 4.77s 一致）——左右腿交替期信号等幅混叠，自相关最近峰 = 完整步态循环（左右各一步 ~2.4s/步），此前粗算把半周期当全周期。gmr 版 `gait_period_autocorr` 的 0.5~2.0s 搜索窗装不下 → 本地版放宽到 1.0~6.0s
- **手部 4 关节全常数**：wrist_roll×2 为填充（yz 无该传感器，填 default 0）；wrist_pitch×2 **源数据本身全 0**（yz 手腕未动）——查表单测的"全身动态"断言按此适配（排除 wrist×4，其余 25 列须有动态）

#### 3. 改动内容

| # | 文件 | 改动 |
| --- | --- | --- |
| 1 | `scripts/tools/prep_yz_ref.py`（新建） | yz CSV → walk_yz 段入库：丢弃 neck/head×2、补 wrist_roll×2=0、重排 Isaac 序、三项替代校验（无 body_positions 可 FK 对拍）、**yz 版周期窗 1~6s**、detrend 只对关节角（root 保留真实前进位移）、30→50Hz 重采样。`--dry-run` 只校验不入库 |
| 2 | `resources/motions/processed/ref_lib.pt` | 3 段 → 4 段：walk_yz(477帧@50Hz, P=239=4.78s, A=103, n_cycles=2)，总帧 3097 |
| 3 | `x1_dh_stand_env.py` `_current_seg_id` | **前进全走 walk_yz**（用户拍板）：\|wz\|>0.15→walk_turn；其余行走→walk_yz。slow/norm 桶出路由但留库中（AMP 仍抽） |
| 4 | `x1_dh_stand_config.py` amp 类 + `_init_amp` + `_sample_amp_demo` | demo 段权重 `demo_seg_weights=[1.0, 0.5, 1.0, 2.0]`（sorted 段序），randint→multinomial；空列表=均匀旧行为。yz 加倍防"ref 教 yz、style 评跑步机慢步"通道分裂，slow 减半（0.1 m/s 极慢步原占比 39%） |
| 5 | `test_mocap_ref_lookup.py` | 路由断言同步（vx=0.4/0.1 均→walk_yz）+ wrist×4 常数适配 + 周期语义测试自动覆盖 walk_yz |

#### 4. 验证结果

**查表单测 6 项全过**——关键项：walk_yz P=239帧=4.78s 与 gait_period 自洽，**corr(查表左髋,−sin)=0.98**（周期语义端到端无 2 倍速）；20 步相邻跳变 0.0066 rad ≤ 源数据原生 p99.9。

**64env×40iter resume 快测**（exp1.3 model_12000 + --disc_fresh，即 exp1.5 同配置换库）：

| 指标 | 数值 | 判读 |
| --- | --- | --- |
| [MOCP]/[AMP] 启动 | 4 段 + 权重 [1.0,0.5,1.0,2.0] + demo_stride [1,1,1,1] | 两通道均识别新段，窗口跨度 0.04s 对齐 |
| rew_ref_joint_pos | 0.12~0.13 | ref 通道在查 yz 表 |
| AMP score / style / healthy | -0.87±/0.87 / 0.13~0.20 / 0.60~0.65 | 与 exp1.5 快测同形态（无回退） |
| reward / episode | 19.7→22.6 / 440→496 | 底模在走且变好 |

#### 5. 时机与提交计划

- **339（exp1.5）不受影响**：云端容器用自己的库副本，本地 ref_lib.pt 改动不触达运行中任务
- ~~exp1.6 = 339 产物 resume + 4 段库~~ **（已作废，2026-09-09）**：339 因 EMA 根因完全冻结（§15.9），产物不可用作底模；yz 库实验的底模回归 exp1.3 model_12000，且须与"EMA 撤除"解耦排期（单变量纪律）
- 已知局限：yz 是纯直线行走，横移/后退指令的 ref 近似为直线位形（原 walk_slow 同为跑步机直线，无恶化）；wrist×4 恒 0 意味着 style 通道里手腕维度 agent 有自由度而 demo 恒定——D 可能拿手腕做次级分离面，监控点

#### 6. exp1.6 正式配置与快测（2026-09-09，已实施未提交）

§15.9 根因实锤后用户拍板：**撤 EMA + yz 库随行**（yz 改动本就在工作树中、339 未包含，属 exp1.6 新变量之一）。

| 项 | 内容 |
| --- | --- |
| 改动 | `x1_dh_stand_config.py`：`arm_action_ema_alpha` 0.85→**1.0**（代码路径保留，含根因注释）；`x1_dh_stand_env.py`：[EMA] 打印加 α=1.0 关闭分支 |
| 配置 | exp1.3 model_12000 底模 + 三修改（门控/disc_fresh/style floor）+ **EMA off** + **yz 4 段库**（本节 §1~4） |

**本地快测**（F1 env，重装 egg-link 后；64env×40iter resume + --disc_fresh + seed=5）：

| 指标 | §15.6（EMA on，3段） | §16.4（EMA on，4段） | **本次（EMA off，4段）** |
| --- | --- | --- | --- |
| reward / episode | 19→25 / 440→540 | 19.7→22.6 / 440→496 | **0.73→35.6 / 33→672**（单调升） |
| buffer healthy | 0.57~0.63 | 0.60~0.65 | **0.83** |
| AMP score / disc loss | -0.90/0.92 / 0.035~0.04 | -0.87/0.87 / — | **-0.872/0.849 / 0.063**（D 被挑战未钉死） |
| style (walk) | 0.17~0.19 | 0.13~0.20 | 0.182 |
| tracking_lin_vel | — | — | 0.005→**0.146**（随 episode 变长爬升） |

判读：EMA 撤除后底模行走惯性完整保留且快速放大（episode 672 为历次快测最高）——与 §15.9"EMA 是 319/339 冻结主因"结论互证。yz 库 + demo 权重正常加载（[MOCP] 4 段、权重 [1.0,0.5,1.0,2.0]）。

工程注记：本地快测需 `PATH=$F1/bin:$PATH` 直调 F1 python（ninja 在 F1/bin，PATH 缺失时 gymtorch JIT 编译报 "Ninja is required"）；egg-link 用 `--no-deps` 重装（isaacgym 不在 PyPI）。

#### 7. 云端任务（2026-09-09，最终 TASK_20260909_061 运行中）

账号[2]额度耗尽 → 切账号[3] limxmtrzgos0jxztfp（新建项目 PRO_20260909_002）。曲折记录：

| 任务 | 结局 | 说明 |
| --- | --- | --- |
| 045 | 手动停 | 首建：OSS 挂载账号[2]上传的 model_12000 + commit b27aef5。**误判**：45min 零日志以为挂载卡死 |
| 056 | 手动停 | 我的止损重建（model_12000.pt 提交入库 d75b9b9、去挂载字段）——用户实停 |
| **061** | **运行中** | 用户复制 045 而来，start 10:36，iter 12000→18000 |

**平台教训（重要）**：该平台任务出日志要 1 小时以上，零日志 ≠ 卡死，不要据此停任务。model_12000.pt 已入库（d75b9b9，`git add -f`，*.pt 默认被 ignore）作为挂载失败的兜底通道。

监控判据沿用 §15.8（disc loss 0.3~0.7 平台 / 双 score 0.7~0.9 震荡 / style >0.3 / healthy 0.5~0.7 / 行走不倒退）；重点首验：tracking 不能像 339 那样从加载起 ≈0（EMA 已撤，底模行走惯性应保留）。

#### 8. 结果与回放验收（2026-09-09，TASK_20260909_061 完成后）

**训练曲线**（iter 12000→17997）：

| 指标 | 首 600 | 尾段 | exp1.3 对照 | 判读 |
| --- | --- | --- | --- | --- |
| tracking_lin_vel | 0.378 | 0.330 | 0.503 | 行走保住（非 339 的 0），略低于 exp1.3（绕圈所致，见下） |
| reward / episode | 83 / 1699 | **90.6 / 2076** | 102 / 2060 | 同量级 |
| disc loss / style | 0.015 / 0.053 | **0.0066 / 0.048** | 0.001 / 0.026 | **D 再锁**（换分离面） |
| score (agent/demo) | — | **-0.979 / 0.977**（日志尾部） | -0.994 / 0.998 | 双钉死回归 |
| buffer healthy | — | 0.489 | — | 门控工作（0.5 下沿） |

**三件套**：`czy/data/exp1.6/{model_17999.pt, play_output.mp4, isaac_diag.csv}`（本地回放，速度阶梯同 §13.3b）。

| 段 | cmd | 体轴 vx | Δx | Δy | Δyaw | actRMS |
| --- | --- | --- | --- | --- | --- | --- |
| 静立 | 0.0 | 0.006 | +0.03 | -0.01 | +3° | 0.380 |
| 行走 | 0.4 | **0.265** | -0.13 | +1.27 | **+141°** | 1.227 |
| 行走 | 0.6 | **0.318** | -0.59 | -1.38 | **-220°** | 1.297 |
| 停止 | 0.0 | 0.087 | +0.54 | -0.02 | +65° | 0.695 |

**两胜一败一新增**：
- ✅ **EMA 根因修复实锤**：行走从加载起保留（339 同位置已冻结），actRMS 1.2~1.3 健康（339 是 2.95 原地震荡）
- ✅ **yz 库速度质变**：体轴实速 0.265/0.318 m/s（exp1.3 仅 0.04/0.13，**6~8×**）——demo 库补速度空洞直接兑现
- ❌ **AMP 仍未复活**：D 换分离面再锁（score -0.979/0.977、style 0.044）。回放确定性口径下膝 vel std 1.2 vs yz demo 0.416（2.9×）、wrist×4 demo 恒 0（§16.5 预警点）——分离面候选：手臂残余抖动 / wrist 自由度 / yz 真实步态 vs 跑步机段的系统性差异
- ⚠️ **新问题：偏航漂移**：行走段 Δyaw ±141°/220°、位移大半在 y（绕圈斜走），净前进反而小；exp1.3 是直线前进。疑似 yz 纯直线 ref + 前进路由全指 walk_yz 后转向多样性下降 + 速度上来后 yaw_drift 惩罚权重不足；停止段滑行也放大（0.54 vs 0.36）

**exp1.7 候选方向**（待拍板）：① 偏航修复（yaw_drift 权重↑ / 指令重采样加 yaw 多样性 / walk_turn 权重调整）；② AMP 复活 reward 侧手段（手臂 action-rate/dof_acc 惩罚——PPO 一致性成立）；③ D 分离面定位（回放 CSV 重算全身逐关节倍率，此时门控已剥站立样本）。

**工程记录**：本地回放段错误 ×4 的根因不是 Vulkan ICD 而是 **DISPLAY 未设**（机器 16:18 重启后 shell 无 X 会话；`DISPLAY=:1` 后渲染恢复正常）——教训：headless 相机渲染仍依赖 X 会话存在。另：回放前必须根目录 `pip install -e .`（本次曾因 egg-link 落到 `/home/robot/F1_train/yanni/...` 旧 checkout，CSV 写错根目录且跑的是旧代码，数据作废重跑）。云端回放任务 148 已停（本地成功后止损）。

## 实验 exp1.7：交替步态重建——速度自适应步频（2026-09-09）

### 1. 问题定位：劈叉滑行的根因是步频-速度结构性失配（exp1.6 回放数据实锤）

exp1.6 回放（isaac_diag.csv，10s×4 段速度阶梯）量化证据——机器人没有形成左右交替迈步，而是一腿前伸一腿后蹬的劈叉滑行：

| 证据 | 数值 | 健康交替步态应有 |
| --- | --- | --- |
| 左右髋关节相关系数 | **+0.467（同相）** | <0（反相交替） |
| 髋活动度 std（左/右） | 0.194 / **0.054（近冻结）** | 两侧对称 0.1+ |
| 髋角速度自相关 | 无步态周期峰 | 4.78s（yz 周期）处峰 |
| 摆动事件（10s） | 每脚仅 **1 次** | ≥3 次 |
| feet_air_time（训练全程） | **0.0015** | ≥0.05 |
| 双脚触地占比 | 40% | <20% |
| 足底载荷（左/右） | 227N / 129N（不对称） | 对称 |

根因链：`_get_phase` 时间驱动固定周期（phase = phase_length_buf×dt/cycle_time，cycle=段周期不随指令缩放）。yz demo 0.255 m/s@4.78s=0.61 m/步；指令 0.4/0.6 m/s 时若要跟拍需要 0.95/1.43 m/步——**物理不可达**（腿长上限 ~0.55 m）→ 策略只能弃跟拍：相位时钟照走、腿跟不上，左右同相伸蹬滑行，feet_air_time 归零，foot_slip 惩罚小到忽略。佐证：ref_joint_pos 仅 0.20/0.5（exp1.3 为 0.9+）、low_speed 尾段 -0.38（不达标被罚）、feet_clearance 0.02。

（相位时钟本身正确：实测 4.78s 精确推进，非半速。）

### 2. 方案设计（两项修改，AMP 全链路零改动保归因）

**修改一：速度自适应步频 + 多段路由恢复（根治）**

- `cycle_eff = T_seg × clamp(v_demo/v_cmd, 0.5, 1.6)`——保持 demo 步幅几何（查表轨迹不变），仅按指令/参考速度比缩放节奏：v_cmd>v_demo 步频加快，反之慢放。yz 段 0.6 m/s 指令下周期 4.78→2.39s，步幅需求回到 0.61 m/步可达区。
- **段参考速度**：真实地面段（yz/turn）按 ref_lib root_pos 水平路径长实测（yz=0.255）；跑步机段（norm/slow）root 静止（皮带抵消位移），用体检表兜底 `{walk_norm:1.23, walk_slow:0.10}`。
- **多段路由恢复**：exp1.6 全指 yz 是速度失配诱因之一。恢复：`|wz|>0.15→turn`（不缩放，按原速）；`|vx|<0.15→slow`；`0.15≤|vx|<0.8→yz`；`≥0.8→norm`。
- **相位连续**：原公式 phase=帧数×dt/cycle 在段切换/缩放变化时相位跳变（换段时突降 4.2 倍）。改为持久积分量 `_gait_phase`，在 `_post_physics_step_callback` 每物理步 +dt/cycle_eff 推进（`_get_phase` 纯读保幂等，stand→walk 恢复时取 gait_start 随机半周期起点）。

**修改二：奖励三调（治标兜底）**

| 项 | 旧→新 | 理由 |
| --- | --- | --- |
| feet_air_time | 1.2→1.5 | 抬腿加压（0.0015 是滑行直接表现） |
| foot_slip | -0.1→-0.25 | 触地脚水平速度惩罚翻倍以上 |
| yaw_drift | -0.8→-1.2 | exp1.6 行走段 Δyaw ±141°/220° 绕圈；步频自适应后漂移源应减弱，此为兜底 |

### 3. 目标判据

- **主判据（回放 CSV）**：左右髋 corr<0（反相）；每脚摆动事件 ≥3 次/10s；feet_air_time ≥0.05；0.4 段 |Δyaw|<45°。
- **训练曲线**：feet_air_time 起量（it500 ≥0.05）；tracking ≥0.3 不倒退；ref_joint_pos 回升。
- **分级**：it500 曲线检查点；it2000 中途 ckpt 回放抽查。

### 4. 风险与止损

| 风险 | 缓解 |
| --- | --- |
| 周期切换相位跳变 | 积分制相位天然连续（只变速不变跳） |
| norm 段回归引手臂分离面（AMP D 再换面） | AMP 归因独立；若 style 再跌，exp1.8 定位分离面 |
| foot_slip 再升 → 滑步硬惩罚（脚粘地） | -0.25 封顶，不再升；异常时回退 -0.1 |
| clamp 边界步频极端（0.5≈2 倍速） | [0.5,1.6] 上下限钳制，物理可达范围内 |

### 5. 实施记录（2026-09-09）

代码四文件五处：

| 文件 | 修改 |
| --- | --- |
| x1_dh_stand_config.py | rewards：`gait_speed_adaptive=True`、`gait_scale_clamp=[0.5,1.6]`、`seg_demo_speed_table={walk_norm:1.23, walk_slow:0.10}`；scales 三调（feet_air_time 1.5 / foot_slip -0.25 / yaw_drift -1.2） |
| x1_dh_stand_env.py | `_init_mocap_lib`：seg_demo_speed 实测+表兜底、[GAIT] 启动打印；`_current_seg_id` 多段路由；新增 `_current_cycle_time`；`_get_phase` 自适应路径读 `_gait_phase`（幂等）；`_post_physics_step_callback` 相位积分推进；reset/init `_gait_phase` |
| play.py | 诊断列 cycle_time 改用 `_current_cycle_time()`（自适应后的有效周期） |

本地快测/云端任务记录见下节补。

**本地快测**（64env×40iter resume model_12000 + --disc_fresh + seed=5，2026-09-09）：

- 启动打印验证 ✅：`[GAIT] 自适应步频 ON, clamp=(0.5, 1.6), 段参考速度: {walk_norm: 1.23, walk_slow: 0.105, walk_turn: 1.228, walk_yz: 0.259}`——norm/slow 跑步机段表值兜底生效（实测 root 静止≈0），yz 实测 0.259 与体检一致，turn 实测 1.228（绕圈 root 位移真实，但不缩放故不使用）
- 曲线不崩 ✅：reward 0.71→30.5 / episode 33→572（exp1.6 快测 35.6/672 同量级）；healthy 0.891、style 0.162、score -0.893/0.857 与 exp1.6 快测相当
- 新奖励起效：foot_slip -0.105（权重 -0.25 下）、yaw_drift -0.027（权重 -1.2 下）
- feet_air_time 0.0002 仍低——已知局限：本地小批量（1536 样本/更新）几乎不移动步态行为，劈叉滑行修复的主验在云端（it500 判据 feet_air_time ≥0.05）
- 数学预检（各指令下 cycle_eff 理论值）：yz@0.4→3.09s、yz@0.6→2.39s（clamp 0.5 触发）、norm@1.2→1.17s、slow@0.05→2.37s（clamp 1.6 触发）——步幅需求全部回到 ≤0.61 m/步可达区

**云端任务**（2026-09-09）：commit 52a8eba；账号[4] limxmtrzhd234eie6t（[3] 额度耗尽标记，新项目 PRO_20260909_022——跨账号项目不共享，[4] 下看不到 PRO_20260909_002）；**TASK_20260909_181**，4090D（ESKU000001/SKUSL000002，¥5.4/时）+ 镜像 BJX00000001/V000124（isaac-gym-v19）；`gm-run X1_29_amp/humanoid/scripts/train.py --task=x1_dh_stand --run_name=exp1_7_gait --headless --seed=5 --ckpt_path=X1_29_amp/model_12000.pt --disc_fresh --max_iterations=6000`（12000→18000，底模仓库直载）。监控：日志 >1h 才出属正常；首验 it500 feet_air_time ≥0.05、tracking 不倒退、healthy 门控正常。
