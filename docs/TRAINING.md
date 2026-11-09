# 训练与复现说明

`train.py`、`training_support.py`、`checkpoint_shards.py`、`adapters.py` 和
`v4_data.py` 保持原始字节，SHA256 与 checkpoint 中的记录一致。
`scripts/train_hamlet.sh` 只负责设置导入路径及选择单卡或 torchrun。
原始共享脚本还含其他模型分支，本次仅支持 HAMLET 路径。

## 安装

先按首页安装 PyTorch 和推理依赖，再安装训练依赖：

```bash
pip install -r requirements-training.txt
python scripts/download_assets.py
```

下载器生成 `reports/download_hamlet.json`，供原训练脚本记录底座版本。
启动脚本会将保存的 LeRobot 源码加入 `PYTHONPATH`；不要随意换成最新版 LeRobot。
原训练环境与当前本地验证环境的来源不同，后者的完整包版本已记录在
`provenance/environment.local.json`；不能将当前每个包的版本都视作原服务器训练版本。

## 数据集要求

使用本地 LeRobot **v3.0** 数据，包含 `meta/info.json`、`meta/tasks.parquet`、
`meta/episodes/**/*.parquet`、`data/**/*.parquet` 和与元数据一致的 `videos/`。
训练读取器检查以下约定：

- `fps=30`，`robot_type=fr3_teach_aligned_single_arm`。
- `observation.state` 和 `action` 均为七维，按 `fr3_joint1` 至 `fr3_joint7`
  排列，单位 rad，绝对关节位置，不含夹爪。
- `action` 已对齐到下一帧关节位置，不要再次平移标签。
- 三个视频字段是 `observation.images.overhead`、`.wrist`、`.third`。
- 每个回合内帧索引和时间戳连续。
- 使用验证划分时，`cleaning/provenance.json` 的 `episode_mapping` 需要提供
  `episode_index` 和 `source_group_id`；同一原始录制派生的回合不能跨训练、验证集合。

原始数据名称、帧数和划分保存在 `data_audit.json` 与 `training_records/`。
历史记录中的 `/root/franka-hockey/` 是原服务器路径；实际运行始终传自己的
`--dataset-root`。原审核划分为 `training_records/original_split_manifest.json`，
只适用于名称、内容和顺序一致的原始数据。新数据用 `--validation-fraction` 或新划分文件。

## 从发布权重继续微调

单卡、有效批次 16 的命令形式：

```bash
NUM_GPUS=1 bash scripts/train_hamlet.sh \
  --dataset-root /data/my_fr3_dataset \
  --init-from "$PWD" --output outputs/my_run \
  --steps 20000 --lr 2e-5 --warmup-steps 500 \
  --batch-size 1 --grad-accum 16 --num-workers 2 \
  --save-every 1000 --eval-every 1000 --eval-samples 1024 \
  --validation-fraction 0.1
```

四卡、与原训练批次安排一致的命令：

```bash
NUM_GPUS=4 bash scripts/train_hamlet.sh \
  --dataset-root /data/my_fr3_dataset \
  --init-from "$PWD" --output outputs/my_ddp_run \
  --steps 20000 --lr 2e-5 --warmup-steps 500 \
  --batch-size 1 --grad-accum 4 --num-workers 2 \
  --save-every 1000 --eval-every 1000 --eval-samples 1024 \
  --validation-fraction 0.1
```

原训练记录每卡峰值约 29.1 GB。单卡示例未分片 AdamW 状态，可能显著超过单张
32 GB 显卡容量；这是受支持的参数形式，不是单卡显存实测结论。

自己新训练的完整 checkpoint 可以用 `--resume outputs/my_ddp_run/checkpoint-NNNNNN`
恢复，要求所有优化器分片都存在，卡数、数据和划分一致；恢复时使用相同输出目录，
去掉 `--init-from`。**本次发布包不要使用 `--resume`**：四份优化器分片未包含。
`--init-from` 会从步数零启动新优化器，并保留已保存的归一化坐标系。

原 v6 训练还从更早的 30k 父 checkpoint 初始化，该父模型及原始数据未一并发布。
因此以上是继续微调示例，不宣称从零完整复现历史训练。此次整理已检查源码、导入和
命令入口，没有重新运行八小时的四卡训练任务。
