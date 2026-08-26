# SpaceMIT K3 Dice Arena Demo

这是 RV 峰会 / 开发者大会「机械臂骰子挑战」的第一版整体效果原型：

- `web/`：大屏 Web 前端，完成游戏列表、规则确认、同步倒计时、双方摇骰、同时开盖、视觉分析动画、胜负播报和再来一局。
- `vision/yolov8_objdetect/`：从现有 `/home/heweijie/spacemit-k3-dev/projects/dice-game/yolov8_objdetect` 迁移的 YOLOv8 K3 摄像头推理工程，保留 OpenCL 前处理、SpaceMIT ONNX Runtime EP、GStreamer 摄像头和骰子分区求和逻辑。

当前阶段**不接机械臂**，用人手和网页按钮代替机械臂的摇骰、停骰、开盖指令。前端默认使用演示数据完成一整局流程；开启浏览器摄像头后可预览实时画面，但还没有把浏览器视频流自动送入 K3 YOLOv8 推理程序。

## 运行前端原型

在当前目录执行：

```bash
cd /home/heweijie/spacemit-k3-dev/projects/dice-game/main
python3 -m http.server 8080 --directory web
```

浏览器打开 `http://127.0.0.1:8080`，即可体验：

1. 选择「摇骰子」，点击「进入摇骰子」；
2. 点击「我明白了」进入准备状态；
3. 点击「开始摇骰」，跟随 3、2、1 倒计时；
4. 人手摇动骰盅，点击「停止摇骰」；
5. 倒计时结束后点击「双方已开盖」；
6. 页面展示 YOLOv8 视觉识别过程和最终胜负；
7. 点击「再来一局」重复演示。

页面支持键盘：`↑/↓` 选择游戏、`Enter` 确认、摇骰阶段按 `Q` 停止。

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
cd /path/to/spacemitk3-dice_demo/vision/yolov8_objdetect
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
