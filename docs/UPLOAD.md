# 双仓库维护与版本对齐

## 仓库分工

- [GitHub](https://github.com/tongji-franka/fr3-groot-hamlet-v6)：训练、推理、部署源码，中文文档与上游引用。
- [Hugging Face](https://huggingface.co/tongji-franka/fr3-groot-hamlet-v6)：同一套模型版本、代码和模型权重。

此 GitHub 源码以 Hugging Face 实际发布提交
`ac36c918c3094b44433daba3840154e4f3b1a2a8` 为基础。除仓库链接、下载入口、
来源清单和 Git 属性说明外，训练与推理代码保持一致。原始发布文件清单保存在
`provenance/huggingface_files.sha256`；当前源码清单为 `provenance/files.sha256`。
源版本及权重标识见 `provenance/huggingface_sync.json`。

模型标签是 `v6-224-020000`，微调权重大小为 6,211,508,320 字节，SHA256 为
`69a6633f025eeab2b365522634ae43356f100ebc04c7bbecb372d64cc2660781`。
Hugging Face 模型页面公开，但当前下载需要人工访问审核；用户需先申请，再运行
`hf auth login`。GitHub 代码不携带访问令牌。

## 从 GitHub 获取模型

```bash
git clone https://github.com/tongji-franka/fr3-groot-hamlet-v6.git
cd fr3-groot-hamlet-v6
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-torch.txt
pip install -r requirements-inference.txt
hf auth login
python scripts/download_assets.py --revision v6-224-020000
```

底座固定为 NVIDIA GR00T 版本 `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61`。
下载器会核对微调权重哈希；底座若要求额外访问许可，同样需先在相应模型页面申请。

## 后续更新代码

```bash
python scripts/make_manifest.py
python -m unittest discover -s tests -v
git add .
git diff --cached --stat
git commit -m '更新训练部署代码与中文说明'
git push origin main
```

提交前检查 `.gitignore`，确保没有纳入权重、本机数据、虚拟环境、缓存或构建产物。
GitHub 使用普通文本 Git 属性，保留完整许可证正文，不把许可证转成 LFS 指针。
训练与推理源码改动应在两个仓库同步，模型参数或推理约定变化时使用新模型版本，
不要移动已经用于复现的旧模型标签。同步 Hugging Face 时只上传明确修改的源码和文档，
无需重传已存在的权重。
