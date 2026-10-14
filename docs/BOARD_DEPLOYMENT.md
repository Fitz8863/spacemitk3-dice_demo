# K3 客户部署

面向与当前开发板相同的 Bianbu 4 镜像，当前验证环境为 **Bianbu 4.0.6、riscv64、系统 Python 3.14**。本流程安装程序依赖，不替客户修改 IP、摄像头序列号、机械臂姿态或标定。NERO 七轴 + 右 Revo2、C920 和 D435i 的接线与现场设置由客户完成。

## 1. 安装

在板端普通登录用户的终端执行；用户需要 sudo 权限，首次安装需要能访问 Bianbu APT 源和 GitHub Release。

```bash
git clone https://github.com/Fitz8863/spacemitk3-dice_demo.git
cd spacemitk3-dice_demo
scripts/setup_board.sh --yes
```

不要 `sudo scripts/setup_board.sh`。普通用户拥有模型和运行目录，MediaMTX 与 PipeWire 使用该用户会话；脚本仅在装包、配置设备组和 RealSense 扩展时调用 sudo。

| 步骤 | 自动完成的内容 |
| --- | --- |
| `deps` | SpaceMIT 原生/Python ORT、TCM、NumPy、OpenCV/OpenCL/GStreamer 编译与运行依赖，PipeWire/WirePlumber、ALSA、V4L2、curl、iproute2 |
| `robot` | SciPy、Python OpenCV/ArUco、python-can、wrapt、packaging、typing-extensions；校验并安装仓内 riscv64/CPython 3.14 RealSense wheel；下载、校验并解压锁定识别库 |
| `permissions` | 当前用户加入 `video` 组；`input_board_enable=true` 时同时加入 `input` 组。仅新增缺失的成员关系 |
| `build` / `models` | 编译骰子和猜拳视觉引擎；下载默认 MOSS 的模型权重、Python 包、动态库、参考音色与资产；逐项核验模型及外部数据文件 |
| `mediamtx` / `can0` / `detect` / `check` | 安装并注册 MediaMTX 用户服务；提示 CAN/摄像头现场配置；运行无硬件依赖自检 |

锁定库放在 `dice_demo/runtime/`，不降级或替换系统共享库。完整依赖会复用，已有配置不会改写。MOSS 压缩包约 452 MB；编译和首次下载耗时取决于板端资源与网络。

## 2. 启动与现场设置

安装提示设备组发生变更时，先新开 SSH 会话；本机桌面终端需注销重登。旧会话不会自动获得 `input`/`video` 组权限。

```bash
# 插好 USB-CAN，驱动已创建 can0 后，由客户初始化通信接口：
sudo ip link set can0 up type can bitrate 1000000
scripts/start_web.sh
curl --noproxy '*' http://127.0.0.1:8080/api/health
```

没有机械臂时可用 `DICE_NO_ARM=1 scripts/start_web.sh` 启动网页与视觉服务；完整机械臂对局仍需硬件可用。

客户按现场修改 `backend/config.json` 的 WebRTC 地址、两款游戏 `adjudicator_config.json` 的采集节点，以及 `dice_demo` 的 D435i 序列号/标定。USB-CAN 驱动、机械臂固件与使能也需匹配，见 [机械臂环境说明](../dice_demo/docs/ENVIRONMENT.md)、[固件要求](../dice_demo/docs/firmware_validation.md) 和 [标定说明](../dice_demo/docs/CALIBRATION_WEB.md)。依赖自检通过不代表这些现场设置已经完成。

板端播报需要登录用户的 PipeWire 会话和可用的默认扬声器。自检只连接音频服务读取元数据，不自动播放测试音。

## 3. 可选组件和复查

```bash
# 额外准备语音识别和备用本地 TTS；不会自动开启功能或更改组件选择。
scripts/setup_board.sh --yes --with-asr --with-matcha
# 只复查，不安装、不启动游戏：
scripts/setup_board.sh check --with-asr --with-matcha
```

默认安装只保证默认 MOSS、两个视觉引擎、机械臂 SDK 与共享播报的依赖。ASR/Matcha 通过上述选项准备；Qwen3、外部 GPT-SoVITS 或自定义组件按各组件说明另行部署。启用 SenseVoice 前先加 `--with-asr` 安装，并由客户修改 `asr_enabled`。

`check` 导入 Python/SDK/RealSense、核验模型和动态库，并用合成 NV12 图像执行一次 OpenCL/YOLO 推理；不打开摄像头、麦克风或 CAN，不发送机械臂动作，不合成/播放语音。机械臂原有完整检查仍可用 `bash dice_demo/scripts/check_environment.sh` 执行；仅核对依赖使用 `bash scripts/robot_dependencies.sh check`，不读取客户标定。部署脚本由主项目维护，不改写 `dice_demo/` 上游快照。

默认机械臂运行目录为仓内 `dice_demo/`，与安装脚本一致，无需额外克隆兄弟目录。接入层使用随仓库的 `third_party/pyAgxArm` SDK，并在配置的 Python 中查找已安装的 RealSense；旧 `vendor-site/` 不再是启动或健康检查前提。显式设置外部 `demo_root` 时，该目录的依赖须另行准备。

## 4. 安装失败后继续

| 提示 | 处理 |
| --- | --- |
| 系统架构或 Python ABI 不匹配 | 使用与当前板一致的镜像；当前随仓库的扩展不兼容其他 CPython 版本 |
| 系统包缺失或下载失败 | 检查 Bianbu APT 源/网络，重跑 `scripts/setup_board.sh deps robot --yes` |
| MOSS 缺模型、外部数据、Python 包或音色 | 重跑 `scripts/setup_board.sh models`；目录已存在不会被当作安装完成 |
| 资产下载中断/校验失败 | 原命令可重跑，`.part` 续传；已完整且校验通过的临时文件直接复用，损坏缓存会清理。GitHub 发布资产必须有有效 SHA256 |
| 按键无权限、无声或没有 `can0` | 刷新登录会话，检查健康接口与默认音频输出；`can0` 缺失需先处理接线/驱动，依赖脚本不会安装或切换内核 |

单独重跑一个步骤，例如 `scripts/setup_board.sh robot --yes`，不会重启正在运行的 8080 游戏服务。修复依赖后运行 `scripts/setup_board.sh check`，再按现场安排启动或切换服务。
