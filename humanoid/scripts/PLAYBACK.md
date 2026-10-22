# 策略回放

统一入口为 `humanoid/scripts/play.py`，本地和云端使用同一脚本。

固定速度：`0 → +0.4 → +0.6 → 0 → +0.4 → 0 m/s`。
每段10秒，总计60秒仿真时间，横向与偏航指令均为0。
不再提供单速度覆盖参数或独立速度扫描入口。

在已安装 Isaac Gym 的服务器上，从仓库根目录执行：

```bash
PLAY_RENDER=1 python humanoid/scripts/play.py \
  --task=x1_dh_stand_v12_exp35 \
  --headless --num_envs=1 --seed=123145 \
  --eval_mode=nominal_no_push \
  --ckpt_path=/absolute/path/model.pt \
  --run_name=exp35_replay
```

生成MP4、诊断CSV和metadata JSON。`PLAY_RENDER=0`只关闭视频生成。
源模型对照任务可将task改为`x1_dh_stand_exp34_source`。
历史实验的批处理脚本保留用于追溯，其调用的策略回放入口仍为`play.py`。
MuJoCo参考数据查看、sim2sim工具属于其他仿真用途。

2026-10-21以前保存的30秒回放与预检结果保留原口径。
每次回放的metadata记录实际task、模型路径、速度序列和控制步长。
