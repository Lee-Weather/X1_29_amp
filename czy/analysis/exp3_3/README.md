# exp3.3：参考姿态＋有界关节残差

日期：2026-10-21。状态：代码、参考预处理和接口验证完成；201.5 零残差回放已完成，站立未通过；未提交训练任务。

用户已确认跳过进一步PD对照，直接提交当前配置从零6000轮云端训练。训练版本保持现有PD增益和残差范围，任务及启动状态见`launch.json`，成功启动后补充记录。

## 控制接口

任务：`x1_dh_stand_v12_exp33`，从头训练，配置上限 6000 轮。

```text
a = clip(policy_action, -3, 3)
delta_q[i] = residual_limit_rad[i] * a[i] / 3
command = q_ref + delta_q
raw_target = delay(command) + motor_offset
effective_target = clip(raw_target, joint_target_lower, joint_target_upper)
torque = clip(Kp * (effective_target - q) - Kd * dq - friction, ±torque_limit)
```

`action_scale=0.3` 不再用于 exp3.3 的残差换算；原始动作裁剪仍为 ±3。未开启旧 `use_ref_actions` 分支。零动作且无延迟/零点偏移时，有效目标等于参考。实际关节角度不受残差范围直接约束。

| 关节组 | 残差最大绝对值 |
| --- | ---: |
| 髋俯仰、膝 | 0.20 rad |
| 踝 | 0.12 rad |
| 髋侧摆、髋偏航 | 0.10 rad |
| 腰 | 0.08 rad |
| 上肢 | 0.10 rad |

策略输出层权重和偏置初始化为零，初始动作标准差 0.3，对应髋俯仰/膝残差标准差约 0.02 rad。保留高斯原始动作与 PPO log-prob 语义，环境仅裁剪执行动作。

## 参考和时序

- 前后行走继续使用 v1.2 的 B3/B4；未加入 B1/B2 周期样本。
- 站立使用现有 `default_joint_angles`，同时作为 reset 姿态；尚未证明该站姿在动力学中稳定。
- 周期 q 改为循环线性插值；段切换保留 0.3 s 混合。
- actor 每帧追加 `scaled(q_ref-default)` 29 维，98→127；历史66帧、短历史5帧。critic 仍为141维/帧，线速度索引不变。
- 每个动作保存参考、相位和周期快照，本步奖励与诊断使用该快照；下一步观测才使用推进后的参考。
- 评估指令注入同时替换最新帧的指令、参考后缀和 critic 参考误差，不额外添加历史帧。
- 执行延迟（重新开启时）施加于完整角度目标。reset 的目标延迟缓存填站姿，传感器 q 缓存填初始 q，不沿用上一 episode 或填全零角度。
- 不再随机从半个行走周期直接初始化；episode 从站立开始，随后按原指令时间表进入行走。

## 几何预处理

生成命令（已执行，是离线数据处理，不是动态回放）：

```bash
python scripts/tools/prep_gmr_v1_2_exp33.py
```

独立输出，不覆盖 exp3.0–3.2 的两个参考 PT：

- `resources/motions/processed/ref_lib_v1_2_exp33_phase.pt`
- `resources/motions/processed/amp_v1_2_exp33.pt`
- `resources/motions/processed/v1_2_exp33_manifest.json`

根据实际脚部碰撞网格，构造双脚最低高度的平滑下包络，逐帧修正 root Z；周期采用循环平滑。保留 q、root 旋转、XY 和相位锚点，重算修正后的 root 线速度。库中还保存脚底局部法向和凸包顶点，运行时无需 MuJoCo。

| 周期 | 原始最低脚底 Z | 校准后最低脚底 Z | 平均 root Z 修正 |
| --- | ---: | ---: | ---: |
| 前进 | −51.99 mm | +1.08 mm | +46.39 mm |
| 后退 | −31.22 mm | +1.23 mm | +30.12 mm |

站立 root 初始高度由站姿与碰撞脚底推算为 **0.61603 m**（最低脚底初始间隙5 mm），代替0.7 m悬空出生。此项不改变站立关节角度。

几何校准不证明动力学可行。AMP 特征不含 root 高度，修正 root Z 不会直接修复判别器分离问题；行走期间也没有新增 root Z 轨迹跟踪奖励。

## 奖励和初始训练域

- `ref_joint_pos` 在站立和行走均使用分组 Huber 误差；取消旧 `stand_still`、`default_joint_pos`，避免指数姿态项在大误差区失效。
- `residual_magnitude=-0.10`、`residual_rate=-0.02`，均为逐关节归一化残差的均方值；rate 为控制步间差值，不除 dt。替代旧原始动作 `action_smoothness`。
- 足部旋转评价脚底法向与世界竖直的夹角，仅约束期望支撑脚或实际接触脚，允许摆动脚转动。
- 足高、离地和基座相对地面高度使用碰撞脚底最低点，取消这些项的固定 ankle 高度近似。
- 初始配置将物理随机范围固定到原区间中点，保留标定的 armature、摩擦、阻尼处理；关闭执行/传感器延迟、观测噪声和推力，地形为平面。
- AMP 权重、前后采样1:1、方向门控、速度奖励、PD增益、力矩限位及 exp3.2 跌倒代价均未在本轮额外调参。
- 恢复随机域时先设置 `initial_domain_nominal=False`，再明确打开相应延迟/噪声开关；不能将当前初始域回放称为完整旧训练随机域。

## 运行入口（已执行零残差，尚未训练）

先检查零残差跟踪，不需要 checkpoint；输出 MP4、CSV 和元数据，不会生成训练模型：

```bash
python humanoid/scripts/play.py --task=x1_dh_stand_v12_exp33 --zero_residual --headless --num_envs=1 --eval_mode=nominal_no_push --seed=123145 --run_name=exp33_zero_residual
```

默认六段各5秒：`0 → +0.4 → +0.6 → 0 → −0.2 → 0 m/s`。若需要先单独看站立，追加 `--play_speed=0 --play_duration=10`。

零残差的自由行走跌倒不等同于参考不可用：应先看跟踪误差、接触和力矩是否说明控制/参考有问题，或只是缺少策略平衡修正。

之后才安排短训练和6000轮正式训练。代码默认从零，不接 exp3.2 checkpoint：

```bash
python humanoid/scripts/train.py --task=x1_dh_stand_v12_exp33 --headless --num_envs=4096 --seed=5
```

新观测维度和动作语义不兼容旧策略，不能使用旧 checkpoint 回放或续训本任务。

## 诊断与导出

CSV 保留原字段，新增逐关节 `delta_q_*`、`pd_target_commanded_*`（参考＋残差）、`pd_target_delayed_*`（延迟后、零点偏移前），以及左右 `sole_height_*`。`ref_pos_*` 与本步动作参考一致；`pos_des_raw_*` 为实际进入限位前的目标，而不是旧的 default＋action。

训练新增 `exp33_residual_saturation_fraction`，继承诊断前缀改为 `exp33_*`。

`export_policy_dh.py` 按127维观测导出完整 DH 网络，并附 `residual_control.json` 和参考库。JIT 输出仍是原始残差动作，参考生成、动作缩放、延迟、关节限位与 PD 需在控制器中执行；不能将 JIT 输出直接当关节角度。`play.py` 的可选导出入口同步采用完整网络。

已在201.5通过127维actor/critic与完整JIT接口检查。零残差站立10s终止6次、六段30s终止18次；两例delta_q和目标与参考差均为0，无目标裁剪/力矩饱和。结果详见[回放报告](zero_residual_20261021_094601/README.md)。固定参考与当前PD不能独立保持平衡；完整步态和学习后表现仍待验证。未启动短训练或长训。
