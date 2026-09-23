# test2 K3 部署

目标：`test2@10.0.91.132`，目录 `/home/test2/spacemitk3-dice_demo_wangjie`。
上层分支 `wangjie_dev`，基于 `main` 的 `3c3d737`。
机械臂接入 `/home/test2/dice_demo_hwj`（`hwj_dev`），Python 使用 `/home/test2/.venv-grasp/bin/python`。
底层 vendor-site 和 vendor-site-deps 通过软链接复用本板已有 SDK、RealSense 和 python-can。后续标定、放杯参数和同步归位修改位于独立的 dice_demo_hwj 仓库，不包含在本上层仓库提交中。

## 使用

```bash
cd ~/spacemitk3-dice_demo_wangjie
bash scripts/start_k3.sh
# 停止网页、TTS 和该部署的视觉/机械臂子进程
bash scripts/stop_web.sh
# 日志
tail -f .runtime/web-8080.log
```

浏览器访问 `http://10.0.91.132:8080`。待机页按任意键，选择摇骰子。
网页的“开始摇骰”会触发真实机械臂；首次联调需现场检查杯子位置和运动区域。
当前语音播报使用 MOSS；语音输入保持仓库默认关闭，使用按键或页面按钮。

MediaMTX：`systemctl --user status dice-wangjie-mediamtx.service`。
该用户服务已启用，启动脚本也会检查启动它。Web 服务由 start_k3.sh 手动启动。
如重启后 can0 未 UP，执行 `sudo ip link set can0 up type can bitrate 1000000`。

## 部署来源与依赖

仅从 `spacemit@10.0.90.160` 只读复制 TTS 模型、Python 包、native 库、参考音频及 MediaMTX 二进制/配置；未修改来源机器的代码、配置或服务。
视觉程序最终在本板编译，使用系统 OpenCV（`/usr/lib/riscv64-linux-gnu/cmake/opencv4`），不依赖额外 LD_LIBRARY_PATH。
GStreamer 运行组件和开发头文件已通过系统包管理器安装。
裁决相机使用 C920 by-id；视频地址已调整为 `http://10.0.91.132:8889`。

## 验证与边界

- K3 后端回归测试 317 项全部通过；`git diff --check` 和启动脚本语法检查通过。
- OpenCL + YOLO 模型自测通过；真实 30 帧采集推理约 16 FPS。
- HTTP 裁决成功：左右各 5 颗，左 18 点、右 20 点，YOLO-only。
- TTS HTTP 合成成功，输出保存在 `.runtime/deployment-tts.wav`。
- 机械臂常驻控制器完成 ready → close，退出码 0；未发送运动指令。
- 停止脚本成功回收 Web、TTS 和视觉进程。
- PowerVR OpenCL 驱动在独立视觉程序析构时有延迟；应用现有 stop() 带超时终止与 kill 回收，部署验收确认停止后无视觉进程残留。没有修改驱动或跳过清理。
- 远程 GPT-SoVITS、云 LLM 及未启用的可选引擎未作为本次本地摇骰子路径依赖；总健康检查可能显示这些可选组件不可用。
- 后续现场已运行抓杯、摇骰、放杯、回 HOME 流程；机械臂精度和标定质量以 dice_demo_hwj 的现场记录为准，上层测试通过不代表底层精度合格。

详细证据位于 `.runtime/adjudication-test.json`、`.runtime/arm-preflight-2/`、`.runtime/deployment-tests.log`、`.runtime/deployment-stop.log`。
