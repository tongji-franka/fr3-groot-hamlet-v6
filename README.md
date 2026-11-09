---
license: other
license_name: nvidia-research-license
license_link: https://huggingface.co/tongji-franka/fr3-groot-hamlet-v6/blob/main/LICENSE.model
library_name: pytorch
base_model: nvidia/GR00T-N1.6-3B
base_model_relation: finetune
language:
  - zh
tags:
  - robotics
  - vision-language-action
  - gr00t
  - hamlet
  - franka
  - fr3
  - hockey
  - safetensors
---

# FR3 GR00T + HAMLET v6：224×224、20,000 步

基于三路相机和历史观测的 Franka Research 3 冰球策略。
**根据团队真机测试，这是目前真机表现最好的版本。** 这是团队对实机表现的
结论，尚未整理成统一试验次数和成功率的公开量化评测；下文同时保留离线指标。

模型已发布在 [Hugging Face](https://huggingface.co/tongji-franka/fr3-groot-hamlet-v6)，
对应代码仓库为 [tongji-franka/fr3-groot-hamlet-v6](https://github.com/tongji-franka/fr3-groot-hamlet-v6)。
GitHub 保存训练、推理、部署源码和中文文档，权重由 Hugging Face 提供。模型页面公开，
**目前权重下载需要先申请并通过仓库的人工访问审核，再登录获准账号**。
版本标识为 `v6-224-020000`，对应 2026-11-06 至 11-07 的 224×224 训练结果，
与后续 448×252 实验区分。下载和后续发布说明见[上传说明](docs/UPLOAD.md)。

本项目基于 **[HAMLET-Isaac-GR00T](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T)**
及 **[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)**，增加 FR3 数据与动作
适配、微调参数、训练快照及部署集成。HAMLET 记忆机制、GR00T 架构和底座的贡献
归原作者。详见 [引用与许可证](THIRD_PARTY_NOTICES.md)、[参考文献](references.bib)。

## 环境与安装

使用 Linux x86_64、Python **3.12**。原训练记录为 PyTorch **2.8.0 + CUDA 12.8**、
四张 RTX 5090；本地推理验证环境的关键依赖也已固定。CPU 支持用于验证，但速度较慢。
GPU 推理可先按至少 16 GB 显存准备，实际峰值取决于输入和 token 数量，
此数值是容量建议，不是本次重新测量结果。

从 GitHub 下载代码后安装环境，再从 Hugging Face 下载权重：

```bash
git clone https://github.com/tongji-franka/fr3-groot-hamlet-v6.git
cd fr3-groot-hamlet-v6
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-torch.txt
pip install -r requirements-inference.txt
export NO_ALBUMENTATIONS_UPDATE=1
hf auth login
python scripts/download_assets.py --revision v6-224-020000
python -m unittest discover -s tests -v
```

下载器固定 NVIDIA 底座版本为 `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61`，
存入 `weights/hamlet/`，并校验微调权重 SHA256。若底座页面要求访问许可，先完成
相应流程，再用 `hf auth login` 登录。不要把令牌写入源码。
可以加 `--revision <模型仓库提交号>`，同时固定本仓库下载版本。
下载完成后，模型加载只访问本地文件。

不要直接执行 `pip install -e models/hamlet/upstream`：上游原始依赖组合与此发布环境
不同。本项目直接加载保存的本地源码，使用 PyTorch SDPA，不需要 FlashAttention 或
DeepSpeed。各类依赖分在推理、训练和机器人三个文件中；安装顺序均是先安装
`requirements-torch.txt`，再安装对应依赖文件。

## 无需数据集或机器人的推理示例

```python
import numpy as np
from policy import HamletPolicy

policy = HamletPolicy(device="cuda")
sample = np.load("observation.npz", allow_pickle=False)
actions = policy.predict(sample["state"], sample["images"], seed=42)
assert actions.shape == (50, 7)
np.save("actions.npy", actions)
```

输入 NPZ 的两个字段：

| 字段 | 形状与类型 | 含义 |
|---|---|---|
| `state` | `[7]` 浮点数组 | 当前实测 FR3 J1–J7 绝对角度，单位 rad |
| `images` | `[4,3,H,W,3]`、uint8 | RGB 图像，时间从旧到新，每时刻按 overhead、wrist、third 排列 |

30 Hz 下，历史偏移为 **[-48,-32,-16,0]** 帧，覆盖 1.6 秒。历史必须来自同一个
回合；仅在需要的历史早于回合起点时，重复该回合首帧。输入原始像素和关节角度，
不要提前归一化。一次输入中的图像分辨率必须一致，训练时原始相机分辨率为 1280×720。
加载器负责按训练方式缩放、居中补黑边到 224×224，并使用保存的状态、动作归一化统计。

命令行示例：

```bash
python examples/infer_npz.py --input observation.npz --output actions.npy --device cuda
```

默认任务文本是 `hit the red hockey puck until it crosses the center line of the table`，
即将红色冰球击过球桌中线。模型针对该任务微调，改变指令不代表已经具备新任务泛化能力。
输出为 **50 个 30 Hz 时刻的七关节绝对位置**，不是位置增量、速度、力矩或夹爪控制。
示例只保存预测，不连接或控制机器人。实机集成见 [部署说明](docs/DEPLOYMENT.md)。

## 文件分布

| 路径 | 用途 |
|---|---|
| `trainable.safetensors` | 493 个 FP32 张量、1,552,860,160 个微调参数，6,211,508,320 字节 |
| `config.json` | 原始训练参数和来源记录；不是 Transformers AutoModel 配置 |
| `data_audit.json` | 实际使用的状态、动作归一化统计，以及数据审计记录 |
| `training_contract.json`、`runtime/` | 训练时保存的推理约定和适配源码 |
| `scripts/train.py`、`training_support.py`、`checkpoint_shards.py` | 原始训练入口及辅助文件 |
| `models/hamlet/upstream/` | HAMLET/GR00T 本地源码，含 Eagle tokenizer 和图像处理配置 |
| `models/lerobot/src/` | 数据读取集成使用的 LeRobot 源码快照 |
| `policy.py`、`examples/` | 不依赖数据集的公共加载、推理入口 |
| `fr3_vla/`、`deployment_runtime/`、`robot/` | 原有部署、缓存、相机、执行器和原生控制桥源码 |
| `training_records/`、`provenance/` | 评估记录、版本来源、文件校验和发布验证结果 |

实例化模型共 **3,110,204,352** 个参数，冻结的视觉和语言参数由固定版本的
NVIDIA 底座提供。微调文件保存完整参数数值，不是 LoRA 矩阵或数值差分。
需要先构建 HAMLET、加载底座，再覆盖微调参数；本项目不支持直接对这个仓库调用
`AutoModel.from_pretrained()`。公共加载入口会检查参数名、形状和有限性。

## 训练、真机表现与离线指标

原训练为 20,000 步，学习率 `2e-5`，预热 500 步，每卡微批次 1、梯度累积 4、
四卡有效批次 16。视觉和语言骨干冻结，训练动作头、两层记忆模块和四个 moment tokens。
归一化继承自父 checkpoint，推理时必须使用保存值。

**团队真机测试结论：本版本为目前实机表现最好的版本。**
这个结论与下面的离线动作误差指标分别报告，避免混淆评价依据。

| 离线记录：1,024 个验证锚点 | 父模型基线 | 本轮 20,000 步 |
|---|---:|---:|
| 动作 MSE，rad² | 0.0031565339 | 0.0033362802 |
| 动作 MAE，rad | 0.0312511904 | 0.0321730874 |

本 checkpoint 是本轮新增 checkpoint 中最好的，但父模型基线 MSE 更低，本轮高约 5.69%。
父模型此前用过验证集中的旧数据，因此这不是严格的未见数据泛化基准，也不直接代表
真机成功率。历史离线一致性报告保留原始验证范围，不能直接用于批准新设备部署。

原始数据集和更早的父 checkpoint 未在此重新发布。完整训练入口及其关键辅助文件与
checkpoint 记录的源码 SHA256 一致。可按 [训练说明](docs/TRAINING.md) 用
`--init-from` 从发布权重继续微调；本包未包含优化器分片，不能精确恢复原训练状态。

## 引用与许可证

使用记忆方法时请引用 **HAMLET**，并注明 **NVIDIA GR00T、Eagle、LeRobot** 等
上游项目的贡献。参考 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、
[references.bib](references.bib)、[CITATION.cff](CITATION.cff) 和
[修改说明](provenance/MODIFICATIONS.md)。许可证保留原文，论文题名及作者采用正式原名。
模型与 HAMLET 源码受上游研究许可证约束，不应将整个项目理解为 MIT/Apache 或不受
商业用途限制。GitHub 与 Hugging Face 保持同一套模型版本和源码；大权重留在 Hugging Face，GitHub 不重复存储。
