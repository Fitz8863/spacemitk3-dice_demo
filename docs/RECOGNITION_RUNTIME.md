# 识别运行库锁定

K3 的杯子识别和 YOLO 裁决使用已验证的 SpaceMIT 2.0.6、TCM 3.0.0+5。版本、包和库的 SHA256 统一由机械臂仓库 `configs/k3_runtime.lock.json` 管理。

1. 在 `backend/components/robot_arm_nero/config.json` 的 `demo_root` 指定机械臂仓库，相对路径以本应用根目录为基准。
2. 在该机械臂仓库执行 `python3 scripts/k3_runtime.py install`。
3. 本应用 `backend/components/vision_yolov8_objdetect/config.json` 中 `runtime.use_robot_runtime_lock: true` 使原生视觉子进程使用同一锁定环境。普通启动脚本仍可直接使用。

Python 杯子识别通过机械臂仓库 `scripts/env.sh` 选择 Python 版 ORT；原生 YOLO 使用原生版 ORT。锁定库只加入视觉子进程的环境，不修改应用父进程、TTS 配置、系统 APT 包或共享库。不要在整个应用的启动环境中全局导出识别专用 `LD_PRELOAD`、`LD_LIBRARY_PATH`、`PYTHONPATH`。

缺少锁定库、校验失败或机械臂仓库路径错误时，视觉启动失败并保留错误，不会悄悄切回新版系统库。非 K3 主机及显式注入的测试/自定义视觉可执行程序维持原有行为；关闭该开关会恢复原生视觉进程原有环境，独立于机械臂仓库的依赖选择。

## 视频调试文字

两个游戏分别在 `backend/games/dice/adjudicator_config.json` 和
`backend/games/rps/adjudicator_config.json` 设置 `"show_debug_text": false`，
隐藏视频内的状态提示（如 `Ready`）、帧率、检测数量及 `ROI` 字样。
检测框、检测类别标签、区域边框、识别结果和后台性能日志不受影响。
设为 `true` 可恢复文字；省略此字段时保持旧版的显示行为。
修改后需重新启动对应视觉进程，网页刷新本身不会重载运行中的视觉配置。
