用户要求给出详细的修改步骤方案——实现 Phase 2 参考轨迹行走。这是一个方案设计请求，需要基于我对代码库的深入理解给出详细方案。让我先确认几个关键细节，然后给出方案。

关键设计问题：
1. 当前 `compute_ref_state` 是相位驱动的正弦步态（cycle_time 0.7s），Phase 2 要换成"查表式"参考轨迹
2. 参考轨迹是有限片段，需要与指令（速度）对齐——怎么对齐？
   - 最自然的方案：保留现有相位机制，用相位索引查表参考轨迹（按步态周期归一化）
   - 或者：时间驱动查表 + 指令选段
3. 跑步机段 vx≈0 但关节角是走路的——参考只给姿态，速度由 tracking 驱动（我上一轮说的）
4. 站立段怎么处理：保持现有锁默认
5. 奖励怎么改：ref_joint_pos 现在是全身 exp(-2*||diff||)，相位查表后同样适用

设计决策（详细方案）：

**核心思路：保留相位驱动框架，把"正弦摆幅"替换为"查表轨迹"**

当前架构的优势是相位系统已经在观测里（sin/cos phase 输入策略），站立检测、双支撑掩码都依赖相位。所以最小侵入的设计：
- 相位 0~1 对应参考轨迹的归一化步态周期
- 从 mocap 轨迹切出"标准步态周期"（或直接整段循环），按相位插值
- 上半身直接查表（这是新增价值）
- 腿部可以混合：初期全部查表（mocap 腿部质量高，前代已验证）

具体步骤：

### Step 1: 预处理脚本（新文件 `humanoid/utils/motion_prep.py` 或 scripts/tools）
- 读 resources/x1_gmr/*.pkl
- 按 dof_names 重排到 Isaac Gym 序
- 120→50Hz 重采样（线性插值）
- 右踝符号 FK 核查（对照 body_positions，错了翻转）
- 步态周期检测（脚步事件或相关周期）→ 切出整周期或保留整段+loop
- 关节角转 delta（相对默认位姿）？——不，直接存绝对角，env 里 ref = 查表值（现在 ref_dof_pos = delta*sin + default，新方案 ref = 查表绝对角）
- 输出 .pt 或 .npz 到 resources/motions/processed/

### Step 2: env 加载器（x1_dh_stand_env.py）
- `_init_buffers` 或 __init__ 加载参考库（GPU tensor）
- 指令→段名映射：站立→无（锁默认）；前进 0.2~0.6 → 对应速度段；转向 → circle_walk
- 相位→帧索引：frame = phase * cycle_frames（每个段有自己的 cycle_time，从数据检测步态周期）
- compute_ref_state 查表替换

### Step 3: config
- motion 文件路径、段映射、开关 use_mocap_ref
- cycle_time 改为从数据来的每段值

### Step 4: 奖励微调
- ref_joint_pos 不变（自动跟随查表 ref）
- feet_contact_number 依赖 stance_mask（相位驱动，保留）
- 双支撑掩码 |sin|<0.1 保留

### Step 5: 验证
- FK 验证（第一帧）
- 冒烟训练
- 训练对比

关键技术细节：

**步态周期检测**：用左踝 pitch 关节的自相关或过零，或者 hip pitch 的主周期。treadmill 段 24s 很多个周期，检测平均周期。

**相位对齐**：参考轨迹的相位 0 要对准"左腿支撑中点"（与现有 gait_start 语义一致），这需要对齐参考的左右腿相位与当前 stance_mask 的定义（sin>0 左支撑）。

**指令分档**：0.05~0.25 → slow；0.25~0.45 → norm；0.45~0.7 → 36_11 快段或 norm；yaw>0.05 → circle；侧向 → 暂用前进段（无侧移 mocap）或保持 12DOF 式处理

还有一个重要问题：**腿部用 mocap 还是保留正弦？** 
风险：mocap 是 1.8m 人缩放的，步长/步频与 X1 实际能力可能有偏差；当前指令速度 0~0.6 但 treadmill 是人的步速（~1.2-1.4 m/s）节拍。
方案：分两档实施
- 2a（保守）：上半身查表 + 腿部保留正弦——风险最小，直接解决"手臂锁死"
- 2b（完整）：全身查表——更接近 AMP 前置，需验证腿部可行性

我建议方案里两个子阶段都写清楚，先 2a 后 2b。

用户要"详细修改步骤方案，以及修改哪些内容"——输出应该是一份可执行的实施计划，文件级别的清单+代码级的关键改动点。不改代码（等确认）。

格式：结构化 markdown，含文件清单、每步的修改内容（伪代码/关键代码段）、验证清单、风险。长度适中偏详细，因为这是实施蓝图。
# Phase 2 参考轨迹行走——详细实施方案

## 0. 总体设计

**保留现有相位驱动框架，只换“参考来源”**：观测里的 sin/cos 相位、站立检测、双支撑掩码、feet_contact_number 全部不动；`compute_ref_state` 的“正弦摆幅查表”升级为“mocap 轨迹查表”。参考给**姿态**（含摆臂），tracking 奖励给**速度**——跑步机段（root 不前进）也因此可用。

分两个子阶段，先易后难：

| 子阶段 | 上半身 | 腿部 | 目的 |
|---|---|---|---|
| **2a（已跳过，保留为回退）** | mocap 查表 | 保留正弦 | 直接治“手臂锁死”，风险最小 |
| **2b（直接实施 ✅）** | mocap 查表 | mocap 查表 | 全身同轨迹腿臂同帧同拍，为 AMP 铺路 |

> **决策更新（2026-09-03，用户定）**：跳过 2a 直接 2b。理由：2a 上下半身参考异源（mocap 人 vs 正弦理想化）协调性打折；全身查同一条轨迹后，腿臂在同一帧推进下天然同拍，锚点只需对轨迹自身做一次，“纠结 cycle_time”退化为单一播放速率问题。0026_circle_walk（全库唯一真实持续走动段，1.18 m/s）在此方案下价值最大。

### 0.1 数据实测勘误（2026-09-03，root z 波动 + 净位移实测后修正）

| 类别 | 段 | 证据 |
|---|---|---|
| **平地（可用）** | 0000/0002/0003 跑步机、0005/0007/0008/0009 短走、0026 走圈 | z 波动 ≤0.12m |
| **台阶（Phase 2 移除）** | **36_11**、36_01、114_08、114_09、127_04、127_06 | z 波动 0.19~0.74m |
| 原地动作 | 0000/0002/0003 | 净位移 ≈0（跑步机原地踏步——姿态参考仍有效，速度由 tracking 驱动）|
| 真实持续行走 | **0026_circle_walk**（16.6s 走圈，路径速度 1.18 m/s） | 全库唯一长时持续走动段 |

**关键更正**：36_11 实为**台阶段**（z 波动 0.50m，回放视频确认为上台阶），原方案 `walk_free ← 36_11` 作废；36_01 同为台阶，一并移出备用。

**方案 A 切段（已定）**：
- **主参考 `walk_turn` ← 0026_circle_walk**（真实持续行走，肢体运动丰富）
- **主参考·稳定源 `walk_norm` ← 0000_treadmill_norm**（23.7s 原地踏步，关节运动干净规律）
- `walk_slow` ← 0002_treadmill_slow（辅助）
- 备用：0005 / 0007 / 0008（短平地段）

### 0.2 URDF 右臂限位修复（2026-09-03，Step1 限位检查发现，已执行）

**发现**：物理镜像 URDF 右臂多处"轴镜像但 limit 未配套"。FLIP 后（FK 实证 0.3mm 的正确值）右肩 roll ∈[+0.15,+0.39] 超 `[-2,0]` 上限、右肘 pitch ∈[-2.0,-1.75] 超 `[0,2]` 下限——**训练时会被 Isaac Gym clamp**（右臂 roll 卡死/右肘强制伸直），2a 摆臂目标失败。而 GMR 原始值虽在声明限位内但 FK 差 51.7mm，证明声明限位与轴约定不配套。

**修复（最小集）**：
- URDF：`right_shoulder_roll` [-2,0]→[0,2]；`right_elbow_pitch` [0,2]→[-2,0]
- config default 取反（同一物理姿态）：右肩 roll −0.06→+0.06；右肘 pitch +0.34→−0.34
- 重转 MJCF（注：mujoco 3.11 加载 URDF root 固定，urdf2mjcf.py 改为 saveLastXML 后文本注入 freejoint）
- 右踝 pitch FLIP 值域 [-0.35,+0.38] vs [-0.41,+0.35] 仅峰值微超 0.03rad → prep 内 clip，不动腿部限位
- ⚠ 影响：exp0 已训 ckpt 的右肩/右肘 default 语义变化（原同号 default 本就镜像不自然），Phase 2 从零重训无兼容问题

**镜像轴关键结论**：物理镜像 URDF 下左右关节角**数值同相**（左右髋 pitch corr≈+0.99），与 env 正弦参考约定自洽（右腿 swing_delta 全反号 → ref 与 −sin 同相）——mocap 数据 FLIP 后可直接按 phase 语义查表，无需半周期偏移。

## 1. 修改文件总览

| # | 文件 | 性质 | 内容 |
|---|---|---|---|
| 1 | `scripts/tools/prep_mocap_ref.py` | **新建** | 预处理：重排/重采样/符号核查/周期检测 → 产出参考库 |
| 2 | `resources/motions/processed/*.pt` | 新产物 | 处理后的参考数据（gitignore 可控） |
| 3 | `humanoid/envs/x1/x1_dh_stand_env.py` | 修改 | 加载器 + `compute_ref_state` 查表（+2a/2b 开关） |
| 4 | `humanoid/envs/x1/x1_dh_stand_config.py` | 修改 | motion 配置段 + 奖励微调 |
| 5 | `czy/exp1/exp1.md` | 记录 | exp0.2 建档（Phase 2a 起步） |

---

## 2. Step 1 — 预处理脚本（`scripts/tools/prep_mocap_ref.py`，新建）

**输入**：`resources/x1_gmr/*.pkl`（带 dof_names）｜**输出**：`resources/motions/processed/ref_lib.pt`

五个子步骤：

```
① 重排：按 dof_names 把 29 列从 gmr 序 → Isaac Gym 序
   （gmr: 腰0-2/左臂3-9/右臂10-16/左腿17-22/右腿23-28 → gym: 左腿0-5/腰6-8/左臂9-15/右臂16-22/右腿23-28）
   输出仍带 names，供 env 端 assert 校验

② 重采样：120Hz → 50Hz（线性插值，np.interp 逐列；控制周期 dt=0.02s）

③ 右踝符号 FK 核查（一次性的硬校验）：
   用当前工作区 URDF（X1_29DOF_physically_mirrored）对第 0/中间/最后帧做 FK
   对照 gmr 的 body_positions 同名 link
   - 全身位置误差 < 2cm → 符号约定一致，直接用
   - 右脚姿态差 ~24°(0.42rad) → 翻转 dof 27 列（right_ankle_pitch）符号后重验
   （复用之前 exp0 验证时写过的 FK 代码）

④ 步态周期检测：对每段用左髋 pitch（gym 列 0）自相关求主周期 T_gait
   - 校验合理域：0.5~1.0s（人走 ~0.7-1.0s，缩放后可能偏快，记录实际值）
   - 相位锚点对齐：找"左脚触地"帧（左踝 z 局部最小 或 左膝角速度峰）定义为 phase=0
     ——必须与现有 stance_mask 语义对齐（sin(2πφ)>0 = 左支撑），否则 feet_contact_number 惩罚会反向
   
⑤ 切段打包（方案 A，首期 3 段 + 备用 3 段）：
   walk_norm  ← 0000_treadmill_norm（主参考·稳定源：中间稳态 20s，整周期截断；原地踏步→只查关节角）
   walk_turn  ← 0026_circle_walk（主参考：稳态圆周段，整周期截断）
   walk_slow  ← 0002_treadmill_slow（辅助：慢速踏步）
   （walk_free ← 36_11 已移除：实测台阶段，z 波动 0.50m）
   备用：0005 / 0007 / 0008（短平地段）
   每段存：dof_pos(T,29)@50Hz + gait_period + phase_anchor_frame + root_pos(参考用) + names
```

**验证标准**：FK 核查已完成（36_11 mean err 0.3mm + 回放视频确认映射自洽，FLIP_JOINTS 6 关节规则直接集成进脚本）；处理后的 0026/0000 首帧 FK 误差 <2cm；每段首尾帧 dof 连续（loop 缝 <0.05rad）；周期检测值打印人工确认。

## 3. Step 2 — env 修改（`x1_dh_stand_env.py`）

### 3a. 加载器（`_init_buffers` 末尾追加）

```python
# Phase2: mocap 参考轨迹库
if getattr(self.cfg.rewards, 'use_mocap_ref', False):
    lib = torch.load(self.cfg.rewards.mocap_ref_file, map_location=self.device)
    self.mocat_lib = {}
    for name, seg in lib.items():
        q = seg['dof_pos']                       # (T,29) gym 序
        assert list(seg['dof_names']) == self.dof_names, f"{name} 关节序不匹配"
        self.mocap_lib[name] = dict(
            q=q,
            period=int(round(seg['gait_period'] / self.dt)),   # 周期帧数
            anchor=int(seg['phase_anchor_frame']),
            upper_idx=torch.tensor([i for i,n in enumerate(self.dof_names)
                                    if 'hip' not in n and 'knee' not in n
                                    and 'ankle' not in n]),      # 上半身 dof
        )
```

### 3b. 指令→段名映射（新方法）

```python
def _current_motion_seg(self):
    """按指令分档选参考段；站立返回 None（保持现有锁默认逻辑）"""
    vx, vy, wz = self.commands[:, 0], self.commands[:, 1], self.commands[:, 2]
    # 逐 env 向量化选择（返回 seg 名张量或列表，站立 env 用 None 分支）
    # |wz|>0.15 → walk_turn；|vx|<0.25 → walk_slow；else → walk_norm（walk_free 已随 36_11 移除）
    # vy 显著 → 暂用 walk_norm（无侧移 mocap，侧向靠 tracking 自由发挥）
```

### 3c. `compute_ref_state` 查表（核心改动）

保留方法骨架，替换摆幅来源：

```python
phase = self._get_phase()   # 不变：站立归零、gait_start 随机 0/0.5

if use_mocap_ref and 当前指令非站立:
    # 相位 → 段内帧索引（每 env 独立段与相位）
    for 段去重:   # 实际用向量化 gather，这里示意
        t = ((phase * period + anchor) % T).long()
        q_ref_seg = lib.q[t]                          # (N_seg,29) 绝对关节角
    # 2a 模式：只取上半身列写入，腿部仍走原正弦分支
    ref[:, upper_idx] = q_ref_seg[:, upper_idx]
    # 2b 模式：全身直接 q_ref_seg（腿正弦分支删除）
    双支撑段处理：保留 |sin|<0.1 → 插值向双支撑姿态过渡（或按段内实际双支撑帧，Phase2 简化为原逻辑）
    self.ref_action = 2 * (self.ref_dof_pos - self.default_dof_pos)  # 注意：改为相对默认
else:
    原正弦逻辑不变（站立 / use_mocap_ref=False 回退）
```

**关键差异点**：原 `ref_dof_pos` 是“default+delta”，mocap 是**绝对角**——所以 `ref_action` 公式要从 `2*ref_dof_pos` 改为 `2*(ref_dof_pos - default_dof_pos)`（仅影响 `use_ref_actions=True` 路径，默认关闭，低风险但必须改对）。

### 3d. 向量化实现要点

每 env 可能处于不同段+不同相位 → 用 `torch.gather`/高级索引一次完成：
```python
# frames: (N,) 各 env 的查表帧号；segs_q: (num_seg, T, 29)
# q_ref = segs_q[seg_id_per_env, frames]   # 高级索引 (N,29)
```
避免 python 循环（4096 env）。

## 4. Step 3 — config 修改

```python
class rewards:
    # ---- Phase2: mocap 参考轨迹 ----
    use_mocap_ref = True          # 2a：上半身查表；False 回退正弦
    mocap_full_body = False       # 2b 开关（True 时腿部也查表）
    mocap_ref_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/processed/ref_lib.pt'
    # cycle_time 含义变化：mocap 模式下每段用自身 gait_period（覆盖全局 0.7）
    # ——实现：_get_phase 的 cycle_time 改为逐 env 从段周期表取（张量）
```

奖励微调（2a 起步建议）：

| 项 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `ref_joint_pos` | 2.2 | **1.8** | 上半身从常数变成动态目标，先降压防止手臂跟踪压力过大压制步态 |
| `stand_still` | 2.5 | 不变 | 站立分支未动 |
| 新增 `upper_ref_w`（可选） | — | 0.4 | 上半身/腿部 ref 分离权重，便于单独调摆臂贴合度 |

`feet_contact_number`/`feet_clearance` 不动（相位语义未变）；`_get_phase` 的 cycle_time 向量化改造是唯一侵入现有逻辑的点。

## 4.1 Step 2/3 执行结果（2026-09-03 ✅）

**env 改造（x1_dh_stand_env.py）**
- `_init_mocap_lib`：ref_lib.pt 常驻 GPU；assert `dof_names == Isaac Gym 序`；upper_dof_indices（非 hip/knee/ankle 共 17 关节）
- `_current_seg_id`：`|wz|>0.15 → walk_turn`，`|vx|<0.25 → walk_slow`，其余 → walk_norm
- `_get_phase`：mocap 模式下 cycle_time 逐 env 从 `seg_period_frames[seg_id]` 取（seg_id 在入口即时计算，避免 _get_stance_mask 链先于 compute_ref_state 的时序问题）
- `compute_ref_state`：查表 `phase*P + A → remainder(T)`，2a 仅覆盖上半身列；`ref_action = 2*(ref_dof_pos - default)`（mocap 为绝对角，不再叠加 default）

**config（x1_dh_stand_config.py）**
- `use_mocap_ref=True`，`mocap_full_body=False`（2a），`mocap_ref_file` 指向 ref_lib.pt
- `scales.ref_joint_pos` 2.2 → **1.8**；右臂 default 镜像取反（shoulder_roll +0.06 / elbow_pitch −0.34，与 URDF §0.2 修复配套）

**验证**
| 项 | 结果 |
|---|---|
| 单测（test_mocap_ref_lookup.py，6 项） | 全部通过；段映射/查表动态/站立回默认/相位连续性（0.0782 rad ≤ 数据原生 p99.9 0.0990）/ref_action 公式 OK；新增⑥周期语义：P=57/74/62 帧@50Hz=1.14/1.48/1.24s，corr(查表左髋,−sin)=0.97~0.99 |
| 冒烟训练（2 iters × 64 env） | 修复前 EXIT=0（但 P 值带 bug：114/148/124）；周期单位修复后待复跑 |
| 0026_circle_walk.mp4 | 1995 帧 16.6s h264 720p@120fps，解码无错 |

**⚠ 已修复 bug：周期帧数单位错位（2 倍速）**
- 原实现 `seg_period_frames = round(gait_period / policy_dt)`（policy dt=0.01s → walk_norm P=114），但查表索引的 mocap_q 是 **50Hz 帧**，114 帧 = 2.286s = 2 个步态周期 → 手臂参考以 2 倍步频播放、仅偶数周期与锚点对齐
- 修复：按**库自带 fps** 换算 `round(gait_period × fps)`（P=57/74/62），`_get_phase` 相位周期 = `frames/fps` 秒
- 单测为何没拦住：mock 的 DT=0.02 恰好算出正确值 57，与 env 真实 dt=0.01 的 114 不一致。新增第⑥项端到端同相验收（扫一个相位单位的查表帧与 −sin 相关 >0.5，2 倍速时会跌到 ≈0）防回归

## 4.2 决策更新：跳过 2a，直接 2b 全身查表（2026-09-03 ✅）

**改动**：`mocap_full_body=True`（config 一行，env 2b 分支已实现）；单测测试②改为全身断言（行走 ref 29 列逐列等于 mocap 查表帧，std∈[0.015,0.261]），6 项全过。指令分档保留：turn→0026 段、norm→0000 段、slow→0002 段，每段全身自洽。

**已知权衡（与 tracking_lin_vel 的步速失配）**：mocap 段步速 ≈1.2 m/s（0026 路径 1.18 m/s、0000 跑步机带速 ~1.2 m/s），低指令（0.4 m/s）时腿部参考"大步快节拍"与速度跟踪存在固有张力。后备手段（按 exp0.2 指标启用）：
1. 降腿部列 ref 跟踪权重（ref_joint_pos 内分腿/上半身权重）
2. 时间缩放：帧推进速率 ∝ vx/v_mocap（步幅恒定、步频随指令，期望速度自然匹配）——需 `_get_phase` 周期乘缩放因子，改动仍局限在逐 env cycle_time
3. `feet_contact_number`/`feet_clearance`（相位驱动）与 mocap 真实触地时刻的微错位（双支撑占比 ~20% vs |sin|<0.1 带 ~13%）：验收时观察，必要时放宽双支撑带

## 5. Step 4 — 训练与验证流程

1. **预处理验证**（本机，~10min）：跑 prep 脚本，FK 误差/周期/loop 缝三项检查
2. **加载验证**：env 单测——固定指令 0.4，打印连续 100 步 `ref_dof_pos` 上半身列，确认随相位平滑变化、站立时回默认
3. **冒烟**：`train.py --max_iterations=2 --num_envs=64`（[DOF] 表后新增段加载日志）
4. **正式训练 exp0.2**：云端（账号4），从零 6000 iter，4090D/L4/L20 任选；**2b 全身查表**（mocap_full_body=True）；验收=exp0 指标全部 + **回放目视摆臂自然度 + 上半身 dof 轨迹 vs 参考的相关系数 + 腿部 ref 跟踪误差（观察步速失配张力）**
5. **失配后备**（exp0.2 达标但步速张力明显时）：时间缩放帧推进（§4.2 后备2）或腿部 ref 降权；2a 回退开关 `mocap_full_body=False` 始终可用

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| 右踝符号 FK 核查不通过 | 预处理内置翻转+复验（Step1③），不改 env |
| mocap 步频（人 ~1.4Hz）与 0.4 m/s 指令失配 → feet_contact 惩罚与参考打架 | 段周期检测后若 >1.2×config 周期，按 X1 指令域时间缩放重采样（预处理⑤加选项） |
| 相位锚点对不齐（左支撑定义反） | Step1④ 用 stance_mask 同款判据（sin>0=左支撑）反推锚点，并单测打印 |
| 上半身参考幅度超 X1 限位（人臂展>机臂） | 预处理 clip 到 dof_limits 的 soft 区间，超限帧数打印 |
| 4096 env 段混合查表性能 | gather 向量化；T≈1200 帧库常驻 GPU（<1MB）无压力 |

## 7. 实施顺序

```
✅ FK 核查（FLIP_JOINTS 6 关节，mean err 0.3mm）
✅ 36_11 端到端回放（mocap_36_11.mp4，映射自洽；身份更正为台阶段 → 移出参考）
✅ 0026_circle_walk.mp4（主参考视觉验收，1995 帧解码无错）
✅ Step 1 全部：prep_mocap_ref.py → ref_lib.pt（3 段 2620 帧，结果见 §2 表）
✅ §0.2 URDF 限位修复 + config default 镜像 + MJCF 重转
✅ Step 2+3：env 加载器+查表 + config + 单测/冒烟（见 §4.1，含周期 2 倍速 bug 修复）
✅ 决策更新：跳过 2a 直接 2b 全身查表（mocap_full_body=True，见 §4.2；单测 6 项含全身断言全过）
→ 冒烟复跑（沙箱放行后）→ exp0.2 建档+云端训练 → 回放验收（摆臂+腿部跟踪张力观察）
→ (步速张力明显时) §4.2 后备手段 → exp0.3
```

方案完整。确认后我从 Step 1 预处理脚本开始写（先 36_11 单段跑通 FK 核查与周期检测，再批量）。