# yolov8_posehand

K3（SpacemiT）单路摄像头 YOLOv8-pose 人体骨架（COCO 17 点）C++ 推理工程。
框架与 [yolov8_segdetect](../yolov8_segdetect) 一致：GStreamer V4L2 采集（MJPEG 硬解 → NV12）→
OpenCL 前处理（letterbox 640 FP32）→ SpaceMIT EP（NPU）推理 → 骨架叠加 → RTSP 推流（MediaMTX）。

## 模型

`models/yolov8n-pose.q.onnx`：SpaceMIT PPQ 量化（QDQ），输入 `[1,3,640,640]`，
单输出 `[1,56,8400]` = 4 box + 1 类置信 + 17 关键点 ×3。
box 与关键点坐标已在图内解码为 640 域像素坐标，类别分与关键点置信度已在图内 sigmoid；
后处理只做阈值筛选 + NMS + letterbox 反映射。

## 构建（板端）

```bash
cd /home/spacemit/projects/dice-game/yolov8_posehand
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

## 运行

```bash
./build/yolov8_pose_camera --config config.json             # RTSP 推到 rtsp://<board>:8554/dice/pose
./build/yolov8_pose_camera --config config.json --self-test # 合成帧走一次全链路
./build/yolov8_pose_camera --config config.json --no-rtsp --no-display --max-frames 30
```

拉流：`ffplay rtsp://<board-ip>:8554/dice/pose`（WebRTC: `http://<board-ip>:8889/dice/pose/`）。

## 配置

参数统一走 `config.json`（cv::FileStorage JSON 后端），命令行同名参数可临时覆盖。
关键键：

| 键 | 默认 | 说明 |
|---|---|---|
| `model` | models/yolov8n-pose.q.onnx | 模型路径（相对 config.json 所在目录时按 CWD 解析） |
| `conf` | 0.25 | person 检测阈值（量化模型置信度偏保守，可实测后调高） |
| `kpt_conf` | 0.30 | 关键点绘制阈值，低于该值的点与相关骨架线不画 |
| `iou` | 0.45 | NMS IoU 阈值 |
| `intra_threads` / `ep_affinity` | 2 / "12;13" | SpaceMIT EP 线程数与绑核（数量必须一致） |
| `rtsp.path` | /dice/pose | MediaMTX 挂载路径 |
| `decoder` | auto | MJPEG 解码：auto/hw(spacemitdec)/sw |
| `class_names` | ["person"] | 必须与模型类别数一致（1 类） |

调试环境变量：`YOLO_POSE_DEBUG=1` 首次推理打印输出各段数值统计（确认分数域）；
`SPACEMIT_FORCE_SOFTWARE_DECODER=1` 强制软解。

## 线程模型（与 yolov8_segdetect 相同）

采集线程（GStreamer appsink）→ 前处理线程（OpenCL）→ 推理线程（ORT+EP）→ 主线程
（NV12→BGR、骨架绘制、imshow、RTSP publish）。队列全部 latest-only（`LatestQueue`），
消费慢时丢弃旧帧不阻塞采集。

## 退出码

2 配置错误 / 3 模型不存在 / 4 OpenCL 初始化失败 / 5 模型（EP）加载失败 / 6 self-test 失败 / 8 运行期 stage 错误。
