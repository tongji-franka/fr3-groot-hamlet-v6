# 开源项目引用、来源与许可证

本项目是 FR3 冰球策略的微调和部署集成，不将 HAMLET 记忆方法、GR00T、Eagle、
LeRobot 或其组成模块称作本项目原创。分发时应保留本说明、上游许可证及源码内的
版权通知。论文引用不能替代许可证要求。

## 使用或包含的上游项目

| 项目及作者 | 实际用途与保存位置 | 版本依据 | 许可证与通知 |
|---|---|---|---|
| [HAMLET-Isaac-GR00T，Myungkyu Koo 等](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T) | 历史记忆、moment tokens、GR00T 集成；`models/hamlet/upstream/` | `63b40557503e32dd93baebb24923e8ebf42121ef`，有本地修改 | 原 `models/hamlet/upstream/LICENSE`，另存 `LICENSE.hamlet` |
| [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 经 HAMLET 继承的 VLA 架构及核心代码 | HAMLET 原项目注明基于 `n1.6.1-release`；本包不是另一个纯净 GR00T 检出 | 保留 HAMLET 快照内原始许可证与版权通知 |
| [NVIDIA GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) | 预训练参数、冻结视觉语言骨干 | `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61` | 模型原始许可证 `LICENSE.model` |
| [NVIDIA Eagle-Block2A-2B-v2](https://huggingface.co/nvidia/Eagle-Block2A-2B-v2) | VLM 配置、定制模型、图像处理和 tokenizer；位于 HAMLET 的 `gr00t/model/modules/nvidia/` | 配置记录 `_commit_hash=950beaa7db5649958a88c9565837d32e3e6a7543`；本地文件存在修改 | 保留各文件 NVIDIA MIT / Hugging Face Apache 声明，另见 `licenses/`；权重从 GR00T 底座取得，不单独下载 Eagle |
| [Hugging Face LeRobot](https://github.com/huggingface/lerobot) | 数据格式、读取器及相关支持代码；`models/lerobot/src/` | 记录 `d3793b7e36522321c1c455f0dc4826db3793dceb`，原服务器为工作区复制 | 完整原始 `models/lerobot/LICENSE`，包含其附带组件通知 |
| [YEKESONG/FrankaFR3_gello](https://github.com/YEKESONG/FrankaFR3_gello) | 本地相机与录制集成的来源，特别是 RealSense 采集辅助代码 | 本地记录 `fe3c86a`；详见 `robot/UPSTREAM.zh-CN.md` | 保留来源说明，不把其他 ROS 包许可证自动当作顶层工具的授权 |
| [GELLO，Philipp Wu 及贡献者](https://github.com/wuphilipp/gello_software) | 周边机器人软件来源 | 本地适配快照，另有逐文件哈希 | 保留原始 `robot/LICENSE.gello`（MIT）；不据此重新许可 NVIDIA 派生模型和源码 |

来源记录见 `provenance/upstream_sources.json`，实际分发文件的校验见
`provenance/files.sha256`。Eagle 配置中的提交号和 LeRobot 的记录版本是来源线索，
不代表本地每个文件都与上游该版本完全相同。原始英文许可证保留法律文本，
上游论文题名、作者与软件专名使用正式原名。

## 独立安装的运行与原生依赖

以下项目按自己的许可证使用，不将其贡献归入本项目：

- [PyTorch / torchvision](https://github.com/pytorch/pytorch)：张量、自动求导与图像处理。
- [Transformers](https://github.com/huggingface/transformers)、
  [Diffusers](https://github.com/huggingface/diffusers)、
  [PEFT](https://github.com/huggingface/peft)、
  [huggingface_hub](https://github.com/huggingface/huggingface_hub)、
  [safetensors](https://github.com/huggingface/safetensors)、
  [Datasets](https://github.com/huggingface/datasets)：模型、权重和数据工具。
- [timm](https://github.com/huggingface/pytorch-image-models)、
  [einops](https://github.com/arogozhnikov/einops)、
  [Albumentations](https://github.com/albumentations-team/albumentations)、
  [OpenCV](https://github.com/opencv/opencv)、[Pillow](https://github.com/python-pillow/Pillow)：视觉模块和图像变换。
- [NumPy](https://github.com/numpy/numpy)、[SciPy](https://github.com/scipy/scipy)、
  [pandas](https://github.com/pandas-dev/pandas)、[Apache Arrow](https://github.com/apache/arrow)、
  [PyAV](https://github.com/PyAV-Org/PyAV)：数值、元数据和视频读取。
- [libfranka](https://github.com/frankarobotics/libfranka)、
  [Ruckig](https://github.com/pantor/ruckig)、
  [librealsense](https://github.com/realsenseai/librealsense)：可选 FR3 原生控制和相机依赖。

当前本地验证环境的版本及包元数据中的许可证声明分别保存在
`provenance/environment.local.json` 与 `provenance/python_dependency_notices.json`。
它们是环境记录，不是所有依赖都属于原始训练环境的证明，也不是替代原始许可证的授权。
本包不包含这些依赖的二进制、CUDA 工具链或虚拟环境。无法从历史构建数据确认的
原生依赖版本在部署说明中明确留空，不编造版本。

## 原始共享适配器内其他模型路径

原训练适配文件保留了 X-VLA、FLOWER、RDT 和 pi0.5 的可选分支，分别注明
[2toinf/X-VLA](https://github.com/2toinf/X-VLA)、
[intuitive-robots/flower_vla_calvin](https://github.com/intuitive-robots/flower_vla_calvin)、
[thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer)
及 [Physical Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) 的来源。
这些模型的权重不在本次发布中，示例也不支持这些模型。

## 修改说明与论文引用

本地修改见 `provenance/MODIFICATIONS.md`。修改过的 HAMLET 文件另有对上游固定提交
逐字节比较的清单。为满足中文文档要求，本次移除了无关模型的说明页，将 README 与
三个 LeRobot 模板改为简体中文；这些文档改动与训练源码分开记录。

正式论文引用条目见 `references.bib`，GitHub 引用元数据见 `CITATION.cff`。
使用记忆方法时请引用 HAMLET 论文，并说明 GR00T、Eagle 和 LeRobot 的贡献；
只引用本微调 checkpoint 不足以体现上游方法来源。

模型许可证与 HAMLET 源码许可证原文不同，分别保存。两者均有限定用途，
不能把整个发布包笼统视作 MIT/Apache 或不受商业用途限制。没有独立许可声明的
本地集成文件不由本说明额外授予新的许可；本说明不编造授权，也不抹除原始权利归属。
