# AMP（Adversarial Motion Prior）整体架构与可参考设计

> 参考代码库：`czy/diff/roboparty_train/robolab`（Isaac Lab 风格 + 魔改 rsl_rl）
> 分析日期：2026-09-04。目的：为 X1_29_amp 从"相位对齐参考追踪"（exp0.2 查表式）升级到真 AMP 做技术储备。

## 1. 整体架构

```
环境侧（robolab/tasks/manager_based/amp/）
  AmpEnv → AnimationEnv → ManagerBasedRLEnv
    ├─ MotionDataManager    # 动作库加载 / 连续时间查询 / 插值
    └─ AnimationManager     # 每 env 每步发参考状态窗口

算法侧（rsl_rl/ 魔改，"RPO"是机器人名非算法名）
  PPOAMP(PPO) + AMPRunner
    └─ AMPDiscriminator     # 判别器（LSGAN）
```

- 算法：`rsl_rl/algorithms/ppo_amp.py`、`runners/amp_runner.py`
- 判别器：`rsl_rl/modules/amp.py`
- 环境：`tasks/manager_based/amp/amp_env.py`、`animation_env.py`
- 数据：`tasks/manager_based/amp/managers/motion_data_manager.py`、`animation_manager.py`
- X1 配置：`x1_amp_env_cfg.py`、`agents/x1_amp_agent_cfg.py`

## 2. 核心机制

### 2.1 判别器输入（两侧逐项镜像，79 维/步 × 3 步窗 = 237 维）

| 特征 | agent 侧 | 参考侧 |
|---|---|---|
| root 角速度（体轴） | 3 | 3 |
| joint pos（**绝对值**） | 29 | 29 |
| joint vel | 29 | 29 |
| key_body_pos_b（6 终端体×3） | 18 | 18 |

**刻意排除**：velocity command、root 线速度、projected_gravity、last_action、相位——参考侧没有的，判别器也不许看；启动时维度硬断言（`amp.py:247`）拦截两侧错位。

- 网络：[1024, 512]+ELU → Linear(→1) 标量 logit；逐单步 EmpiricalNormalization
- policy 观测与 disc 观测刻意差异化：policy 用 rel 角+噪声，disc 用绝对角无噪声（防任务信号泄漏）

### 2.2 AMP reward（每 env step 实时计算）

```python
rew = clamp(1 - (D(s)-1)**2 / 4, min=0)      # LSGAN 映射：有界、免指数、数值稳
style_reward = dt * style_reward_scale * rew  # ×0.02×1.5 → 与控制频率解耦
total = lerp·task + (1-lerp)·style            # lerp=0.6（60% task + 40% style）
```

- 融合发生在 rollout 采集时（no_grad + eval 旧参），融合值直接进 GAE
- lerp 静态无调度（注释记录 0.75→0.6 回调史：style 压过验收线）

### 2.3 动作数据管理

- 每段一个 `.pkl`：`fps / loop_mode / root_pos(F,3) / root_rot(F,4) / dof_pos(F,29) / key_body_pos(F,6,3)`
- **速度类全部不存盘**，加载时前向差分现算（lin/ang/dof_vel）；key_body_pos 离线 FK 一次性算好存盘（训练零开销）
- **无相位列、无接触列**
- 查询：连续时间 `calc_phase → frame_idx0/1 + blend`，平量 lerp、四元数 slerp，WRAP 取模 / CLAMP 裁剪
- **采样与相位完全解耦**（非 tracking）：reset 抽 motion id（带权 multinomial）+ 随机时刻；`random_fetch=True` 时每步都重抽参考窗口——纯风格匹配
- 框架留有混合钩子：`random_fetch=False`（时间推进 = 相位对齐 tracking）+ ref root 线速度字段（`observations.py:239-257` 注释项）

### 2.4 判别器训练（LSGAN + 三件套稳定化）

- 损失：`0.5·[MSE(D(agent),−1) + MSE(D(demo),+1)]`
- **梯度惩罚只加 demo 侧**：`10·(‖∇_x D‖)²`（AMP 论文标准）
- **与 PPO 同频同批**：每个 mini-batch 一次 disc step（5 epochs×4 mb）；rollout 时 reward 用旧 D 算完，无冻结交替
- replay buffer 仅 **100 控制步**（> rollout 24 步）：每步即时 append，新旧样本混合防 stale
- 独立 Adam lr 1e-4；trunk/linear 差异化 weight decay（1e-3 / 1e-1）；grad clip 1.0
- KL 自适应 lr 只作用于 PPO，disc lr 恒定；ckpt 含判别器+normalizer+disc optimizer

### 2.5 与 PPO 耦合

同 mini-batch 顺序执行：PPO loss backward → disc loss backward → 各自 clip + step。style reward 权重静态靠人工调。

## 3. X1 关键配置（与 X1_29_amp 同为 29DOF）

| 项 | 值 |
|---|---|
| 控制 | 50Hz（sim 0.005×decimation 4），动作=关节位置目标 ×0.25 |
| key bodies | 双 ankle_roll / knee_pitch / elbow_yaw |
| 命令域 | vx(−0.5,2.5) vy±0.5 wz±1.5，heading 模式 10s 重采 |
| 动作库 | 14 段 pkl，带权采样（主段 4.0 / 跑步机 2.0 / 其余 1.0） |
| disc 超参 | buffer 100 / lr 1e-4 / grad penalty 10 / style_scale 1.5 / lerp 0.6 / LSGAN |
| PPO | 4096 env、24 steps/env、actor/critic [512,256,128]、lr 1e-4 adaptive、γ0.99 λ0.95 |
| symmetry | **关闭**（29DOF 镜像索引未建；RPO 版开了 mirror loss 0.2） |

## 4. 可直接借鉴的工程细节

1. **LSGAN + dt 缩放**：style reward 有界且换控制频率免重调 scale
2. **lerp 融合**：task/style 比例直读，比加权和好调
3. **判别器稳定三件套**：demo-only grad penalty + 差异化 weight decay + 小 buffer
4. **两侧特征硬断言**：启动拦截错位（与 ref_lib.pt 的 dof_names assert 同思路，可扩展到特征级）
5. **连续时间插值**：`calc_frame_blend` 亚帧插值，任意 dt 帧率解耦——ref_lib.pt 整数帧查表可升级
6. **key_body_pos 离线 FK**：prep 阶段算好存盘，训练零开销——可在 prep_mocap_ref.py 中照做
7. **reset 污染警示**：robolab 在 reset **后**才取 disc 样本（docstring 声称保留但实现被注释，`amp_env.py:51-72`）→ 自实现时必须在 reset 前抓帧，否则 buffer 注入初始姿态垃圾样本
8. **重力门控 tracking**：track 奖励乘 `clamp(−proj_gravity_z, 0, 0.7)/0.7`，倒地断奖励防躺平刷分
9. **symmetry 增强**：29DOF 镜像索引若建好（我们 URDF 是物理镜像，左右数值同相），可免费样本翻倍

## 5. 与 exp0.2（当前方案）的关系

| | exp0.2 查表式参考追踪 | robolab 真 AMP |
|---|---|---|
| 参考用法 | 相位对齐查表 → `ref_joint_pos` 单点奖励 | 判别器风格匹配（随机窗口，无对齐） |
| 步频 | 被 ref 段周期硬约束（可控） | 无约束（风格自由，步频随任务） |
| 风格表达 | 受限于逐关节 L2 奖励 | 全局风格判别（含速度+终端体位置） |
| 失配风险 | mocap 步速 vs 指令速度张力（plan §4.2） | 无对齐故无此张力 |

**互补结论**：exp0.2 的 ref_lib.pt 管线（Isaac 序、周期、锚点已验证）可直接喂给 AMP——
1. prep 补两件事：离线 FK 算 key_body_pos 存盘、加载时差分算速度
2. 开 `random_fetch=False` + 保留相位锚点 = **相位对齐 + 风格判别**的混合模式（它框架已留钩子）
3. 或者保守路线：先让 exp0.2 达标，AMP 作为 exp0.3+ 的风格升级项

## 6. `ppo_amp.py` 代码解剖与 DHPPO 移植映射

> 源文件：`czy/diff/roboparty_train/rsl_rl/rsl_rl/algorithms/ppo_amp.py`（432 行）
> 本质：**继承标准 PPO，只加三块 AMP 专属逻辑**——初始化区（判别器+独立优化器）、采集区重写（style reward 融合）、更新区重写（PPO loss + disc loss 同批交替）。

### 6.1 区域划分

**区域 A：类声明（L1-20）**——`PPOAMP(PPO)` 继承标准 PPO；policy 网络零改动（仍是标准 ActorCritic），AMP 不动策略结构。

**区域 B：初始化（L22-117）**
- B1 继承父类：全部 PPO 超参原样透传（L51-71）
- B2 判别器构建（L73-93）：损失类型 GAN/LSGAN/WGAN 三选一；`obs_groups` 直接取自 policy——保证判别器与 policy 用同一套观测分组定义（防特征错位）
- B3 独立优化器（L96-117）：disc trunk/linear **分组 weight decay（1e-3/1e-1）**，线性输出层重正则防 logit 漂移；独立 Adam，PPO 的 KL 自适应 lr 不会波及；两个 CircularBuffer 由 runner 创建后注入

**区域 C：`process_env_step` 重写（L119-132）**——AMP 的唯一奖励改造点，每 env step 执行：

```python
disc_obs      = get_disc_obs(obs)            # ① 从 policy 观测抽判别器特征组
disc_demo_obs = get_disc_demo_obs(obs)       # ② 参考侧窗口（env 随机采好放 obs）
style_rewards = predict_style_reward(disc_obs, dt)   # ③ 旧参 no_grad 算风格分
rewards_lerp  = lerp_reward(task, style)     # ④ 0.6·task + 0.4·style
disc_obs_buffer.append(disc_obs)             # ⑤ 策略样本即时入 buffer
disc_demo_obs_buffer.append(...)             # ⑥ 参考样本即时入 buffer
super().process_env_step(obs, rewards_lerp, ...)    # ⑦ 融合奖励走标准 PPO
```

传给父类的是融合值（L132）——**GAE/return 用的就是融合奖励**，PPO 无感知。

**区域 D：`update` 重写（L134-432）**，按功能 6 段：
- D1 生成器（L148-166）：policy 侧用 RolloutStorage；disc 侧用两个 CircularBuffer 的 generator；三路 zip 对齐（L166）
- D2 PPO 标准流程（L180-271）：symmetry 增强 / 重算 log_prob / KL 自适应 / surrogate+value+entropy——基本抄父类
- D3 附加 loss（L273-319）：symmetry mirror、RND——父类可选件，与 AMP 无关
- D4 disc loss（L321-356）：归一化 → D(agent)应≈−1、D(demo)应≈+1 → 三种损失二选一（LSGAN=`0.5·[MSE(D(agent),−1)+MSE(D(demo),+1)]`）→ **grad_penalty 只加 demo 侧**
- D5 梯度与更新（L358-382）：**同批三连，无冻结交替**——PPO backward → disc backward → 各自 clip+step → **事后** `update_normalization`（本批用旧统计量，避免同批既训练又定义归一化）
- D6 统计与返回（L384-432）：均值化；`storage.clear()`（**CircularBuffer 不清空**，滑窗跨迭代混合）；返回 dict 带 4 个 AMP 指标

### 6.2 数据流一图

```
env.step ──→ process_env_step ──→ style=旧D(obs) ─→ lerp融合 ─→ RolloutStorage（供GAE）
                    │
                    └─→ 原始disc样本 ─→ CircularBuffer(滑窗100步, 不清空)

update(): 每 mini-batch {
    PPO:   surrogate+value+entropy ← RolloutStorage 样本   ┐
    disc:  LSGAN(D(agent) vs −1, D(demo) vs +1)            ├→ 各自 backward+step
           + grad_penalty(仅demo)  ← CircularBuffer 样本    ┘
}
```

### 6.3 移植映射：robolab PPOAMP → 我们 DHPPO

**结构差异**（决定不能照搬文件，但设计模式 100% 可移植）：

| | robolab `PPOAMP(PPO)` | 我们 `DHPPO` |
|---|---|---|
| 基类 | 继承新版 rsl_rl PPO | 独立类（legged_gym 早期风格） |
| 策略网络 | 标准 ActorCritic | `ActorCriticDH`（自带 state_estimator） |
| rollout | RolloutStorage+父类管理 | 自带简化 RolloutStorage |
| 观测载体 | TensorDict（obs 分组） | 拼接张量+索引切片 |

**改造清单**（对应 DHPPO 三处）：
1. `process_env_step`（dh_ppo.py L110-120）：`rewards` 入 storage 前插 lerp 融合（一行）
2. `update`（dh_ppo.py L126）：循环内 PPO backward 旁加 disc backward + step
3. 新建 `humanoid/algo/amp/`：AMPDiscriminator + 两个 CircularBuffer；`DHOnPolicyRunner` 仿照 AMPRunner 注入 buffer

**三个照抄的工程细节**：
- **独立 disc optimizer + 分组 weight decay**：disc lr 恒定不被 KL 自适应误伤；我们 `state_estimator_optimizer`（dh_ppo.py L72-73）已证明多 optimizer 并存可行
- **每步即时入 buffer**（L129-130）：buffer 覆盖 100 步 > rollout 窗口 → 新旧混合防过拟合旧策略
- **事后更新 normalizer**（L382）：本批用旧统计量归一化

**监控指标直接可用**（L427-430）：`amp/disc_score、disc_demo_score、disc_loss、grad_penalty`——demo_score 与 agent_score 的差值即"风格差距"实时量化，可作 exp0.2 验收中"摆臂自然度"的客观替代指标。DHPPO 的 update 返回三元组（L201）扩成 dict 加这几个 key 即可。

**一个反面教材（别抄）**：它的 `get_disc_demo_obs(obs)` 把参考窗口混在 policy 观测里传算法——env 每步往 obs 塞参考数据。我们移植时更干净：env 把 disc 样本放 `infos`/extras（DHPPO `process_env_step` 已有 infos 通道，L110），policy 观测保持纯净。

### 6.4 实施路径建议

1. **现在**：等 exp0.2 云端训练出验收结论
2. **exp0.3 候选**：若摆臂/风格达标但想提自然度上限 → 按 6.3 清单给 DHPPO 加 `DHPPOAMP`，判别器特征先复用 ref_lib.pt（dof_pos+差分 vel，不含 key_body_pos 也能先跑）
3. **低成本预热**：prep_mocap_ref.py 补 key_body_pos 离线 FK（§4-6），为完整特征集做准备
