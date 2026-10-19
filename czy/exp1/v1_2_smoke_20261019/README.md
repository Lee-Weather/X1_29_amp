# v1.2 Isaac Gym 冒烟测试（2026-10-19）

服务器：`robot@10.12.201.5`，F1 环境，GPU PhysX，独立目录
`/home/robot/czy/exp1/exp_20261019_151449/x1_v12_smoke`。

## 结果

- 定向检查通过：站立、`vx=0.6` 前进、`vx=-0.35` 后退均完成物理步；前后参考查表最大误差均约 `1.49e-8 rad`，AMP agent/demo 窗口为 `3×61`，无非有限值。原始日志：[smoke_directions_3.log](smoke_directions_3.log)。
- `16 env × 50 iteration` 短训练退出码为 0。第 49 轮 AMP 行走奖励 `0.12746`、健康样本比例 `0.495`，说明行走样本已进入 AMP 更新。原始日志：[smoke_walk_resetmix.log](smoke_walk_resetmix.log)。
- 首次短训练的健康样本比例与行走奖励全为 0：随机初始策略平均约 146 个控制步即复位，早于原定数秒后的行走段。为此在 v1.2 配置中增加 50% 的复位即行走环境；旧结果见 [smoke_walk_50.log](smoke_walk_50.log)。

本次测试只证明任务能启动、参考/AMP 数据能加载、三种命令可执行、行走样本可用于更新。第 49 轮平均 episode 长度约 142 个控制步，尚不能据此判断学得的步态质量或长期稳定性。
