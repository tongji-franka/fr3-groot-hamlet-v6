# HAMLET / Isaac-GR00T 本地源码快照

原项目：[myungkyuKoo/HAMLET-Isaac-GR00T](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T)。
作者为 Myungkyu Koo 及其合作者。源码基于 NVIDIA Isaac-GR00T，
通过 moment tokens 和历史记忆模块为动作预测提供时间信息。

本目录来自记录版本 `63b40557503e32dd93baebb24923e8ebf42121ef`，包含为本地
FR3 推理环境保留的修改。具体文件见仓库根目录的 `provenance/MODIFICATIONS.md`。
本中文说明替代上游基准使用指南，原文可从上述版本的 GitHub 页面查阅；
原始 `LICENSE` 与各源码版权声明保持原文。

本次实际使用的入口、依赖和训练配置以仓库根目录中文 README 为准，
不要直接套用本目录原始 pyproject 的依赖，也不要将上游 RoboMME/RMBench
基准脚本当作本次 FR3 微调命令。

论文题名为 **HAMLET: Switch Your Vision-Language-Action Model into a History-Aware Policy**，
发表于 ICLR 2026。正式作者、BibTeX 和其他上游项目引用见根目录 `references.bib`。
