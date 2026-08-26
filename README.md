# SpaceMIT K3 Dice Arena Demo

这是 RV 峰会 / 开发者大会「机械臂骰子挑战」的第一版整体效果原型：

- `web/`：大屏 Web 前端，完成游戏列表、规则确认、同步倒计时、双方摇骰、同时开盖、视觉分析动画、胜负播报和再来一局。
- `vision/yolov8_objdetect/`：从现有 `/home/heweijie/spacemit-k3-dev/projects/dice-game/yolov8_objdetect` 迁移的 YOLOv8 K3 摄像头推理工程，保留 OpenCL 前处理、SpaceMIT ONNX Runtime EP、GStreamer 摄像头和骰子分区求和逻辑。

当前阶段**不接机械臂**，用人手和网页按钮代替机械臂的摇骰、停骰、开盖指令。前端默认使用演示数据完成一整局流程；开启浏览器摄像头后可预览实时画面，但还没有把浏览器视频流自动送入 K3 YOLOv8 推理程序。

## 在 K3 板端运行前端

当前目录是 K3 板端目录通过 SSHFS 挂载到开发机的路径，文件实际位于板端：

```text
开发机挂载路径：/home/heweijie/spacemit-k3-dev/projects/dice-game/main
K3 板端路径：   /home/spacemit/projects/dice-game/main
```

因此前端应直接在 K3 板端启动，而不是在开发机启动。登录 K3 后执行：

```bash
cd /home/spacemit/projects/dice-game/main
scripts/start_web.sh
```

默认监听 `0.0.0.0:8080`。如果显示屏和浏览器就在 K3 板端，打开：

```text
http://127.0.0.1:8080
```

如果从同一局域网另一台设备访问，先在板端查看 IP，再打开：

```text
http://<K3板端IP>:8080
```

停止服务：

```bash
cd /home/spacemit/projects/dice-game/main
scripts/stop_web.sh
```

也可以安装 systemd 服务（可选）：

```bash
sudo cp deploy/dice-arena-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dice-arena-web.service
systemctl status dice-arena-web.service
```

`web/` 是纯静态前端，使用 K3 系统自带的 `/usr/bin/python3` 提供 HTTP 服务，不需要 Node.js、npm 或开发机依赖。浏览器在板端通过 `127.0.0.1` 访问时，可以正常申请摄像头权限；如果从其他设备通过 HTTP IP 访问，浏览器可能因非安全上下文限制摄像头权限，正式部署建议使用 HTTPS 或让浏览器直接运行在 K3 板端。

页面交互流程：

1. 选择游戏列表中的「摇骰子」，点击「进入摇骰子」；
2. 点击「我明白了」进入准备状态；
3. 点击「开始摇骰」，网页执行 3、2、1 倒计时；
4. 当前由人手实际摇动骰盅，完成后点击「停止摇骰」；
5. 人手打开双方骰盅，点击「双方已开盖」；
6. 页面展示视觉识别动画和本局胜负（当前识别结果为演示数据）；
7. 点击「再来一局」回到准备状态。

键盘操作：`↑/↓` 选择游戏，`Enter` 确认，摇骰阶段按 `Q` 停止。

## 迁移的 YOLOv8 工程

源码、模型和 K3 配置位于：

```text
vision/yolov8_objdetect/
├── src/
├── models/best.q.onnx
├── config.json
└── CMakeLists.txt
```

在 SpaceMIT K3 板端编译：

```bash
cd /home/spacemit/projects/dice-game/main/vision/yolov8_objdetect
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build build -j4

# 先做模型 / OpenCL 自测
./build/yolov8_camera --model models/best.q.onnx --self-test --no-display

# 再做短时摄像头测试
./build/yolov8_camera --model models/best.q.onnx --camera 1 \
  --no-display --max-frames 30
```

当前迁移的模型是 YOLOv8 raw 输出模型，预期输出 `[1, 10, 8400]`，类别 ID `0..5` 对应骰子面 `1..6`。程序会在 CPU 侧执行 YOLOv8 解码、分区和 NMS，并要求左右两侧各识别到 5 颗骰子后才判定总和。

`vision/yolov8_objdetect/config.json` 已去掉源工程中的 API Key，不把凭据提交到仓库。若需要 LLM 复核，请在板端创建未纳入 Git 的 `config.local.json` 或使用原工程的本地配置；演示阶段可使用 `--no-llm`，直接展示稳定 YOLO 结果。

## 与后续 ROS2 / Agent / 机械臂的接口边界

前端已经按状态机拆分为以下可替换阶段：

```text
SELECT -> RULES -> READY -> COUNTDOWN -> SHAKING -> OPEN
       -> ANALYSIS -> RESULT -> READY / SELECT
```

后续接入时，可以把前端的三个“人手按钮”替换成 ROS2 / Agent 事件：

- `startShake`：下发摇骰语义指令；
- `stopShake`：下发停骰语义指令；
- `revealDice`：收到开盖完成事件后开始采集与识别；
- `ANALYSIS`：接收 K3 YOLOv8 输出的 10 颗骰子、置信度、两侧总和和判定结果。

建议下一阶段增加一个轻量 WebSocket 或 HTTP bridge，统一传递 `game_state`、`countdown`、`camera_frame`、`detections`、`scores` 和 `winner`，这样不会把 ROS2、视觉和网页 UI 互相耦合。
