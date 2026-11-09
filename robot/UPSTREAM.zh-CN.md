# 来源与改动

本地源文件来自 `/home/user1/下载/FrankaFR3_gello-main`，对应公开仓库
<https://github.com/YEKESONG/FrankaFR3_gello> 的 `fe3c86a` 快照。

- `record_right_lerobot.py`：录制交互、后台保存与收尾流程的参考。
  SHA256 `c36a1634fae0a1e28cb9254a956fc88ebaedcbb36c74d234d4311ad7ff90710c`
- `record_dual_lerobot.py`：提取 `KeyReader` 到 `common.py`。
  SHA256 `bfe052e5a31f7df3cea34b69cae7329ec1b94b75f948cf8c805c78d43d35322d`
- `realsense_camera.py`：保留 SDK 序列号定位、曝光预热和 RGB 后台采集；
  添加原子图像/时间读取、设备时间戳、帧号、内参和独立旋转。
  SHA256 `a30def3feaadbf829bc5d7c40994cd0686c04bd7bdba00afe0c4f465cde5a3f8`

其余桥接、数据契约、源数据检查及配置代码为本地适配。
上游 README 的许可证说明针对各 ROS 包；这里只移植顶层采集工具，
保留其来源信息，不将各 ROS 包许可证自动扩展为顶层工具的独立授权声明。
LeRobot 是独立安装的 Apache-2.0 第三方依赖。

适配前本地录制包的副本保存在
`/tmp/franka-recorder-before-adaptation-20261013/`，包括文件名中多余空格和
空配置文件 `single_fr3_d345.yaml`。
