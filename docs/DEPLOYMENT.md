# 部署与推理说明

## 先运行不连接机器人的推理

使用 `HamletPolicy` 或 `examples/infer_npz.py`。这两个入口不需要数据集、相机驱动
或原生控制桥。`policy.py` 只调整资源路径，调用未改动的 `runtime/adapters.py` 与
`runtime/v4_data.py`，使用保存的归一化统计。每次调用先清空动作头记忆，再输入完整
四帧历史。CPU 和 CUDA 都使用 BF16 自动混合精度，衔接 BF16 骨干与 FP32 动作头。
公共入口不启用真机专用加速，不宣称达到历史优化部署的延迟。

## 原有 FR3 部署源码

- `fr3_vla/cloud.py`：检查点加载、带时间戳的历史选帧和实时循环。
- `fr3_vla/hamlet_inference.py`：可选 K/V 复用、BF16 线性层与 CUDA Graph。
- `robot/ros2/src/franka_lerobot_recorder/`：相机时间戳、异步执行器、观测对齐和执行检查。
- `robot/tools/fr3_act/`：带运动约束与看门狗的 C++ 控制桥及模拟器。
- `robot/ros2/src/franka_lerobot_recorder/config/act_limits.json`：原始执行约束配置。

在 PyTorch 基础环境中安装 `requirements-robot.txt`。原生桥还需要 C++17 编译器、
CMake >=3.19、提供 CMake 配置的 Ruckig，以及与 FR3 固件匹配的 libfranka。
原安装来自 ROS 2 Humble 环境；历史构建元数据未记录准确的 Ruckig/libfranka 版本，
这里不编造版本号。它们独立于 Python 模型环境，发布包不包含编译二进制。

仅构建模拟器，不连接机器人：

```bash
cmake -S robot/tools/fr3_act -B robot/artifacts/build \
  -DCMAKE_BUILD_TYPE=Release -DFR3_ACT_WITH_ROBOT=OFF
cmake --build robot/artifacts/build --parallel 2
ctest --test-dir robot/artifacts/build --output-on-failure
```

需要在线桥时使用 `FR3_ACT_WITH_ROBOT=ON`，并设置正确的 `Franka_DIR`。
`robot/scripts/fr3/build_act.sh` 和 `check_native_build.py` 保留原始源码、协议与约束
身份检查，不能拿不匹配的旧二进制直接替代。

## 为自己的安装生成部署导出

原部署 CLI 除了模型，还会验证本地数据集契约，需要数据集包含
`recording/contract.json` 及匹配的相机、动作元数据。从本仓库根目录导出：

```bash
python scripts/export_cloud_checkpoint.py \
  --checkpoint "$PWD" --output exports/my_fr3 \
  --training-root "$PWD" --dataset-root /data/matching_fr3_dataset
```

导出器检查数据集内容与训练记录的哈希。不同数据应配套自己的新训练 checkpoint，
不要修改哈希绕过检查。不依赖原数据集的评估直接使用公共推理入口。

将 `configs/fr3_teach_v4.example.yaml` 复制为自己的配置，修改相机序列号、摆放与
数据集路径。原相机参数是示例，不代表其他安装已经完成标定。
模型输入顺序是 **overhead/wrist/third**，实时执行器顺序是 **wrist/overhead/third**，
适配代码会显式重排，保留这个映射。

只检查配置的命令：

```bash
python -m fr3_vla.deploy hamlet --checkpoint exports/my_fr3 \
  --training-root "$PWD" --dataset-root /data/matching_fr3_dataset \
  --fr3-root "$PWD/robot" --config configs/my_fr3.yaml --check-config
```

动作启用前，先运行 `python scripts/check_cloud_parity.py --help`，并完成新导出对应的
离线一致性检查，生成与本次清单匹配的报告。实时运动还需要显式
`--move --execute`、独立启动的原生桥及匹配的机器人配置，原有执行检查保持启用。
模型返回 50 步动作，历史部署默认每 4 个控制时刻重规划一次；历史选帧仍严格按
`[-48,-32,-16,0]`，不由重规划周期代替。

`training_records/` 中历史报告只说明原环境下的单样本检查。旧路径、旧哈希不构成
对新安装的验证。本次发布验证只读取本地样本并生成动作，没有发送机器人命令。
