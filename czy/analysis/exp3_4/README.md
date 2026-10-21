# exp3.4：成熟策略迁移到 v1.2 B3

日期：2026-10-21。状态：201.5物理回放及200轮冒烟通过；GitHub/GitLab已同步30c4226；云端TASK_20261021_068已启动。

## 固定配置

| 项目 | 配置 |
| --- | --- |
| task | `x1_dh_stand_v12_exp34` |
| 底模 | 轴修复后的exp2.4a-cont2 `model_26998.pt`，内部iter26997 |
| 底模SHA256 | `f1838ebe07007d1f65380d67db7915a23c3793aa2bb2786d1006d7b1f03664a5` |
| 初始化 | 继承actor、critic、状态估计器及动作std；重新初始化AMP判别器和所有优化器 |
| 控制 | `a=clip(policy_action,-3,3)`；`q_target=q_default+0.3*a+motor_offset`，名义offset=0；保留成熟接口，不叠加q_ref，不使用exp3.3小残差或exp3.2目标裁剪 |
| 观测 | 单帧98，总历史6468，短历史490；critic423 |
| 指令 | 站立2–3秒→前进6–9秒→停止2–3秒→再次前进/停止，按真实控制步计时；episode上限24秒 |
| 速度 | 前进[0.35,0.65]m/s，横向/偏航为0；无后退 |
| 参考 | B3周期63帧，用于相位和软姿态目标；B3连续351帧，用于AMP示范；均50Hz |
| 数据权重 | 仅`walk_forward`，`demo_seg_weights=[1.0]`；B1/B2/B4未进入两个库 |
| AMP | scale5，task/style融合系数0.5/0.5；站立纯task，不新增方向门控 |
| 姿态奖励 | `ref_joint_pos=0.5`，继承成熟的有界软跟踪函数 |
| 其他奖励 | 以00dee7a时期底模配置为基准：保留foot_place1、low_speed2、stand_still3.5等；正奖励截断，不继承exp3.2/3.3额外稳定性/残差惩罚；撤销当前公共配置后来增加的yaw_drift_angle |
| 足部修正 | 按碰撞脚底计算实际高度与法向；脚高/clearance目标由当前B3周期插值得到，避免固定5cm目标与B3冲突；保留落脚点随命令速度缩放 |
| 动力学 | 训练范围中心值、plane，无推力/噪声/执行及传感器延迟；保留名义armature/friction/damping标定值；root初始化保持底模配置 |
| 预算 | 4096env、seed5、**新增12000轮**、每100轮保存，预计最终`model_38998.pt` |
| 回放 | 自动采用每段5秒`0 → +0.4 → +0.6 → 0 → +0.4 → 0`，总30秒；显式负速度参数拒绝 |

当前实现只固定初始迁移来源。训练入口在创建仿真前检查精确checkpoint路径、底模哈希和两个参考文件哈希，并强制fresh discriminator；旧任务没有transfer_manifest，继续使用原入口逻辑。exp3.4已训练checkpoint的后续恢复需明确来源后扩展恢复校验，不能伪装成初始底模。

## 数据生成

```powershell
python scripts/tools/prep_gmr_v1_2_exp34.py
```

独立生成`ref_lib_v1_2_exp34_phase.pt`、`amp_v1_2_exp34.pt`和`v1_2_exp34_manifest.json`，不覆盖旧库。B3来自exp3.3已经校准脚底几何的库，原关节角、相位锚点和连续帧保持不变，只删除后退段。它仍是运动学参考，尚未证明在新任务中可稳定实现。

## 服务器执行顺序（已执行）

先在同一名义域回放原底模。独立task`x1_dh_stand_exp34_source`使用旧参考/旧控制，关闭与新任务相同的随机扰动和延迟，便于判断底模是否因域改变失效：

```powershell
python humanoid/scripts/play.py --task=x1_dh_stand_exp34_source --headless --num_envs=1 --seed=123145 --eval_mode=nominal_no_push --ckpt_path=czy/data/exp2.4a_cont2/model_26998.pt --run_name=exp34_source_nominal
```

随后以相同底模在新task回放并做短训检查，确认引用、奖励、脚部目标、复位和判别器更新均正常。源模型名义回放若失败，应先分析域适应，不能把失败归因于B3。

正式命令准备如下，需上述物理检查后执行：

```powershell
python humanoid/scripts/train.py --task=x1_dh_stand_v12_exp34 --headless --num_envs=4096 --seed=5 --max_iterations=12000 --ckpt_path=czy/data/exp2.4a_cont2/model_26998.pt --disc_fresh --run_name=exp3_4_b3_transfer
```

云端提交时须显式打包manifest指定的源模型并给出精确路径，不能依赖仓库中模型的兜底扫描。API Key/skills等凭据内容不进入训练包。

## 已完成验证

`python czy/analysis/exp3_4/validate_cpu.py`通过：源模型strict加载，actor输出2×29、critic2×1且有限；16个CPU环境张量执行4001步调度，初次站立203–299步，后续前进/停止时长均在指定范围；外部评测命令不被调度覆盖；参考插值及站立切换正确；错误checkpoint被拒绝；两个库只含B3。结果见[cpu_validation.json](cpu_validation.json)。该检查用实际调度方法和CPU张量替代物理环境，**不代表Isaac Gym物理验收通过**。

## 验收

新增2000/6000/12000轮分别回放，保留所有结果。最终固定seed123145/123146/123147，30秒无终止；+0.4/+0.6各段去除最初1秒后平均跟速误差≤15%；两脚均持续离地/落地，无单侧长期承重；停止2秒内|vx|<0.05且保持。报告航向、关节限位、力矩、接触时序和AMP实际贡献，不能仅看平均reward。

## 修改文件

- `humanoid/envs/x1/x1_dh_stand_v12_exp34_config.py`、`x1_dh_stand_v12_exp34_env.py`：独立前进迁移任务及源模型名义域回放。
- `humanoid/envs/__init__.py`：注册两项任务。
- `humanoid/utils/transfer_preflight.py`、`humanoid/scripts/train.py`：初始迁移校验，仅对显式配置该功能的任务启用。
- `humanoid/scripts/play.py`：支持任务指定的速度序列；原任务继续沿用原六段序列。
- `scripts/tools/prep_gmr_v1_2_exp34.py`、`.gitignore`、两个新PT及manifest：B3独立数据及底模来源。

后退留待B3前进迁移验收后作为独立尝试；本次未注册后退训练。

## 云端启动记录（2026-10-21）

- 账号：limxmv4xnjtlno3gsm@maxxspace.com（userId4565）。
- Task：TASK_20261021_068；平台启动时间2026-10-21 14:59:56，状态3；实际PPO进度以launch.json为准。
- 资源：1×4090D24G，¥5.4/小时；4096env、seed5、新增12000轮，目标model_38998.pt。
- 训练代码：codex/exp3-4，30c4226ea834ac817da797af900beead8a1fcba0；GitHub和GitLab同步完成。用户明确授权发布算法、B3数据、实验记录和源模型权重。
- 云端源模型精确路径：X1_29_amp/model_26998.pt；继承成熟策略，fresh判别器及优化器。
- 201.5检查：preflight_20261021_144449/summary.json。源模型和B3初始30秒回放均0次终止；B3的+0.4/+0.6段均速0.320/0.493m/s，42次关节子步力矩裁剪事件，未做最终稳态验收。
- 64env×200新增轮短训退出码0，无异常，保存model_27198.pt。仅说明训练链路可运行，不代表长训收敛。两份CSV及日志已取回同目录；本次预检未录制MP4。

首轮运行确认：日志27015/38998，约完成新增18轮；Mean reward=8.67，Mean episode length=169.25步，约66356steps/s，启动期ETA17732.6秒。已确认源模型加载、迁移校验和exp3.4配置日志，无启动异常；这些是启动期指标，不用于收敛判定。
