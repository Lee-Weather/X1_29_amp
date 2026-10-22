# exp3.5：姿态约束和渐进目标边界

日期2026-10-22。状态：实现及服务器预检完成，云端长训TASK_20261022_065已启动。

## 1. 上一实验结果与教训

exp3.4最新model_38700在60秒回放无终止，但净偏航2.134rad；右肩yaw均值0.58rad、左腕roll均值-0.384rad，右踝目标88.4%越过硬限位。行走AMP贡献约0.29%。继承的default_joint_pos实现只在站立有效，削弱了腿部yaw/roll约束。

## 2. 本轮目标

保留model_26998的行走接口，恢复姿态与航向监督，减少窄范围关节越界。仍为站立/前进/停止；B3=1.0。训练前先验证完整边界不会破坏底模稳定性。

## 3. 修改内容

### 修改一：显式选择姿态奖励

exp3.5显式调用成熟X1DHStandEnv的default_joint_pos实现，行走也有效，权重1.0。ref_joint_pos、hip_ref、stand_still及orientation明确采用exp3.1实现，避免继承行为不明；B3脚底目标继续用exp3.4几何修正。

恢复yaw_drift_angle=0.5，保留yaw_drift=-1.2。航向以episode复位起点为准，连续前进/停止不更新锚点。

上肢14关节和腰3关节分别计算归一化Huber误差，组内取平均；容差0.25/0.15rad，权重各-0.5。行走跟踪B3，站立跟踪默认姿态，过渡沿用0.3秒混合。远场误差保持线性增长，不能以其大误差渐近饱和。原ref_joint_pos=0.5保留。

### 修改二：数据标定的分组目标边界

保留default+0.3*clip(action,±3)。17个上肢/腰关节的最终范围：健康model_26998初始B3回放PD目标0.5/99.5分位区间，与B3参考范围取并集，向外留0.05rad，再与URDF硬限位取交集。边界及来源摘要见资源JSON和bounds_provenance.json，可用prepare_bounds.py重建。

正式训练前24000控制步（1000更新）不裁剪；接着24000步线性收紧，之后保持完整边界。该时钟是并行环境共同控制步，不乘环境数、不因episode复位而重启。回放主动使用完整边界，确保检验最终状态。

腿部目标不裁剪、不缩小成熟控制空间；增加leg_target_overflow=-0.1，按环境soft joint limits计算越界平均量/0.2rad。其作用为弱引导，不能视为机械保护。上肢/腰裁剪真正作用于PD扭矩计算，CSV同时保存raw/effective目标、每关节裁剪子步数、边界强度和姿态组代价；actual角度可离线统计贴限位情况。

### 修改三：AMP检查与预算

AMP尺度5、task/style=0.5/0.5、D学习率保持继承值，不盲目放大风格回报。初步audit确认29关节顺序一致、3帧agent/demo窗口均0.04秒、数据有限；健康底模和B3的肘/腰/腕位置分布仍有明显差异。名称和时间间隔一致不能证明物理轴或动态接触完全正确，暂不宣称AMP饱和已解决。检查记录为amp_input_audit.json。

## 4. 修改文件

- humanoid/envs/x1/x1_dh_stand_v12_exp35_config.py、x1_dh_stand_v12_exp35_env.py：独立任务。
- humanoid/envs/__init__.py：注册x1_dh_stand_v12_exp35。
- humanoid/utils/playback_diagnostics.py：exp3.5目标和姿态代价字段。
- resources/motions/processed/v1_2_exp35_target_bounds.json：17关节边界。
- czy/analysis/exp3_5/：CPU检查、AMP审计、范围标定和服务器预检工具。

## 5. 训练参数

| 项目 | 值 |
| --- | --- |
| 来源 | exp2.4a-cont2 model_26998，SHA256 f1838ebe07007d1f65380d67db7915a23c3793aa2bb2786d1006d7b1f03664a5 |
| 初始化 | actor/critic/estimator/std继承，D和所有优化器重新初始化 |
| task | x1_dh_stand_v12_exp35 |
| 正式预算 | 4096env、seed5、新增6000轮，预计最终model_32998.pt |
| 数据 | 复用经过哈希校验的exp3.4 B3周期63帧及连续351帧，50Hz |
| 云端任务/账号 | TASK_20261022_065 / limxmv4xopmuth560d@maxxspace.com（userId4567） |
| 监控点 | 新增1000、3000、6000轮回放，决定是否延长 |

准备命令：

```bash
python humanoid/scripts/train.py --task=x1_dh_stand_v12_exp35 --headless --num_envs=4096 --seed=5 --max_iterations=6000 --ckpt_path=model_26998.pt --disc_fresh --run_name=exp3_5
```

## 6. 验收

统一play.py：0→0.4→0.6→0→0.4→0，各10秒、60秒。正式模型3个固定seed123145/123146/123147均无终止；稳态跟速误差≤15%；直行累计偏航、肩腕腰姿态偏差、贴限位与裁剪比例较exp3.4明显下降；左右持续交替离地，停下后恢复合理站姿。预检仅检查底模兼容和运行链路，不能替代最终验收。

## 7. 当前验证

validate_cpu.py通过真实方法的CPU张量检查：热身恒等、半/全强度裁剪、腿部目标不变、回放完整强度、行走默认姿态约束生效、Huber远场代价递增。语法检查通过。

服务器预检运行编号20261022_105603：健康底模在完整边界下运行60秒、不录制视频；通过后执行64env×200新增轮短训。为覆盖三个阶段，短训进程单独将热身/渐变压缩为480/960控制步，正式配置仍为24000/24000。结果已回收，详见下方预检完成；该短训用于检查接口和复位，不用于收敛判定。


### 预检完成（2026-10-22）

201.5同seed、相同60秒六段序列的底模对照：

| 条件 | 重置 | 力矩裁剪子步事件 | 前进三段均速(m/s) | 最终净偏航(rad) |
| --- | ---: | ---: | --- | ---: |
| 原目标，无新边界 | 0 | 20 | 0.357 / 0.496 / 0.278 | 4.685 |
| exp3.5完整上肢/腰边界 | 0 | 0 | 0.303 / 0.488 / 0.348 | 2.804 |

完整边界6000步全部强度为1；12个腿部目标raw/effective逐值相同。肩腕和腰姿态仍接近健康底模，腿部控制未被裁剪。结果支持边界与底模基本兼容；底模在长序列中本身已有严重净偏航，新约束还需通过训练学习，不能据此判为航向验收通过。

64env×200新增轮短训正常结束，无异常或NaN/Inf指标，保存model_27198.pt。最后Mean reward209.11、Mean episode length2299.33步（22.99秒），新upper_pose/waist_pose/leg_target_overflow/yaw_drift_angle均有日志贡献。值函数loss604.11、AMP agent约-0.994，说明短训末价值适应和风格学习仍需观察，不能将链路通过等同于收敛。正式长训从健康model_26998开始，现已启动TASK_20261022_065，详见下方启动记录。

本次不录视频；两份6000行CSV、日志和summary已回收至czy/analysis/exp3_5/preflight_20261022_105603/。验证摘要为validation_summary.json。


### 正式长训启动记录（2026-10-22）

- Task ID：TASK_20261022_065；账号limxmv4xopmuth560d@maxxspace.com，userId4567。
- 平台启动时间：2026-10-22 13:13:29（北京时间）；状态3。实际首轮更新以launch.json及启动日志为准。
- 资源：1×4090D24G，ESKU000001/SKUSL000002，¥5.4/小时；镜像BJX00000001/V000412。当前资源查询仅返回4090D。
- 训练提交：c345e078d320242339d431326bfbd35eb511f95e，codex/exp3-5，GitHub和GitLab已同步。用户明确授权发布本轮代码及启动记录。
- 精确底模路径：X1_29_amp/model_26998.pt；actor/critic/estimator/std继承，D和所有优化器重新初始化。4096env、seed5、新增6000轮，目标model_32998.pt。
- 正式边界调度24000步热身+24000步渐变；未使用冒烟测试压缩调度。
- 启动前确认任务挂载元数据personalDataPath为空；未删除云端存储文件。服务器预检和同条件对照均已回收；源模型及参考资源哈希核对通过，配置末尾空行清理与已测版本AST一致。

首轮运行确认：日志27002/32998；已确认精确底模加载、fresh discriminator及迁移校验，进入PPO更新，无启动异常。约63827steps/s，启动期ETA9592.3秒（约2小时40分钟），仅为估算；尚未形成足够episode统计。训练代码提交保持c345e07，后续提交仅补充启动记录。
