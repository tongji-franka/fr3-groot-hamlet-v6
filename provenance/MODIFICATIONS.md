# 来源、本地改动与训练源码保留

本包 HAMLET 源码来自 `myungkyuKoo/HAMLET-Isaac-GR00T` 的记录版本
`63b40557503e32dd93baebb24923e8ebf42121ef`。发布准备时将本地源码与该固定提交的
GitHub tarball 逐字节比较，原本有七个新增或修改文件：

- `gr00t/model/gr00t_n1d6/gr00t_n1d6.py`：本地初始化、动作推理和历史处理修正，
  以及单次采样内部的可选 K/V 复用。
- `gr00t/model/modules/dit.py`：动作头注意力与推理支持。
- `gr00t/model/modules/eagle_backbone.py`：PyTorch SDPA 配置、moment-token 本地集成修正。
- Eagle 目录的 `modeling_eagle3_vl.py`、`modeling_siglip2.py`：为本地环境保留的兼容调整。
- Eagle 目录的 `vocab.json`、`merges.txt`：离线 tokenizer 需要的附加资源。

准确状态及哈希见 `hamlet_changes_from_upstream.json`。为满足本次中文文档要求，
本包又将 `models/hamlet/upstream/README.md` 改为中文来源说明，原版完整说明仍可
从上述固定提交读取。各源码版权头与原始许可证保留原文，不能由修改说明推断原作者
已经审查或认可这些本地变更。

LeRobot 是记录提交的工作区复制，不称为纯净上游发行版。Python 源码保持复制时内容；
本次移除了与当前训练无关的 README，保留并简化翻译三个运行时卡片模板。
`provenance/lerobot_documentation_changes.json` 记录文档路径与处理方式。
模板改动不改变训练数据读取逻辑。完整原始许可证保存在 `models/lerobot/LICENSE`。

FR3 的数据约定、绝对关节角度、三路历史图像、训练、检查点导出及部署集成位于
`scripts/`、`fr3_vla/`、`deployment_runtime/`、`robot/`。相机工具的来源另见
`robot/UPSTREAM.zh-CN.md`。

本次整理没有修改 checkpoint 的 `runtime/` 文件，也没有修改原训练入口和关键辅助
文件。它们仍与 `config.json.source_sha256` 一致。新增的 `policy.py`、下载器和示例
属于发布使用入口，不冒充原训练代码。CPU 推理入口补上 BF16 自动混合精度上下文，
以衔接 BF16 骨干与 FP32 动作头；磁盘中的模型参数和保存的适配源码未改变。

本次完整验证使用真实本地历史帧，但没有连接机器人或发送运动命令。真机最好版本的
结论来自团队提供的实机测试结果，不由这次离线检查重新得出。
