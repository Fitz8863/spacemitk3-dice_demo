# YOLOv8-seg 单路摄像头推理

这个项目面向 SpaceMIT K3 板端，使用单路 V4L2 MJPEG 摄像头完成：

```text
摄像头 → GStreamer/V4L2 → NV12
       ├─ latest-only 显示槽 → 主线程 BGR/OSD → OpenCV 窗口
       └─ latest-only 算法槽 → OpenCL 前处理 → latest-only 推理槽
                                      → SpaceMIT EP → 最近一次分割结果
```

当前实现将采集、前处理、推理、显示拆成独立阶段。每个跨线程槽位最多保留一个最新对象，生产者不会等待消费者；算法落后时丢弃旧帧，避免延迟持续累积。显示使用最新采集帧叠加最近一次可用推理结果，优先保证画面连续性。

当前实现基于两个已经验证过的板端项目：

- `/home/spacemit/projects/yolo-demo`：YOLOv8-seg 的 13 输出后处理和 mask contour 映射思路。
- `/home/spacemit/projects/dice-game/yolov8_objdetect`：GStreamer 摄像头、`spacemitdec` 能力检测、`jpegdec` 回退、OpenCL NV12 前处理和 SpaceMIT EP 绑核方式。

## 配置

默认配置在 `/home/spacemit/projects/dice-game/yolov8_segdetect/config.json`：

```json
{
  "model": "models/yolov8n-seg.q.onnx",
  "camera": "/dev/video1",
  "device": "",
  "width": 1280,
  "height": 720,
  "fps": 25,
  "intra_threads": 2,
  "ep_affinity": "12;13",
  "conf": 0.5,
  "iou": 0.45,
  "max_detections": 100,
  "queue_depth": 2,
  "display_enabled": true,
  "decoder": "auto",
  "focus": 0,
  "zoom": 160,
  "self_test": false,
  "no_display": false,
  "max_frames": 0,
  "dump_input": "",
  "yolov8_enabled": true,
  "rtsp": {
    "enabled": false,
    "host": "127.0.0.1",
    "port": 8554,
    "path": "/dice"
  }
}
```

`ep_affinity` 通过 SpaceMIT EP 选项 `SPACEMIT_EP_INTRA_THREAD_AFFINITY` 设置。配置中的核数量必须与 `intra_threads` 一致。当前默认是 `intra_threads=2`、`ep_affinity="12;13"`，只绑定 EP 推理线程；不会把整个进程的所有线程都绑定到 12、13。

`queue_depth` 目前保留用于兼容配置，但单路低延迟实现使用固定深度 1 的 latest-only 槽位，不会阻塞等待旧帧完成。

`camera` 可以配置为数字索引或设备路径字符串，例如 `"/dev/video1"`。`device` 非空时优先于 `camera`；命令行 `--camera` 也同时支持数字和路径。

`self_test`、`no_display`、`max_frames`、`dump_input`、`yolov8_enabled` 与兄弟项目保持兼容；当前分割程序要求 `yolov8_enabled=true`，`dump_input` 作为兼容字段保留。

板上 C920 当前应优先使用：

```text
/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_9395301F-video-index0
```

## 构建

在 K3 板端执行：

```bash
cd /home/spacemit/projects/dice-game/yolov8_segdetect
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

## 运行

模型文件已放在 `models/yolov8n-seg.q.onnx` 时：

```bash
./build/yolov8_seg_camera --config config.json
```

无窗口、有界摄像头验证：

```bash
./build/yolov8_seg_camera \
  --config config.json \
  --device /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_9395301F-video-index0 \
  --no-display --max-frames 30
```

模型/前处理自测：

```bash
./build/yolov8_seg_camera --config config.json --self-test
```

可用命令行参数会覆盖 `config.json`，例如：

```bash
./build/yolov8_seg_camera \
  --config config.json \
  --model models/yolov8n-seg.q.onnx \
  --camera /dev/video1 \
  --ep-affinity '12;13'
```

### RTSP 推流

开启配置：

```json
"rtsp": {
  "enabled": true,
  "host": "127.0.0.1",
  "port": 8554,
  "path": "/dice"
}
```

程序会把最新的已渲染 BGR 帧异步送入：

```text
appsrc → leaky queue → videoconvert → NV12 → spacemith264enc
       → h264parse → rtspclientsink → MediaMTX
```

推流线程使用 latest-only 策略，网络或编码变慢时只丢弃旧帧，不阻塞摄像头、前处理、推理和本地显示。板端需要先运行 MediaMTX，并确保 `rtspclientsink`、`spacemith264enc` 可用。

命令行也可以覆盖推流配置：

```bash
./build/yolov8_seg_camera --config config.json \
  --rtsp --rtsp-host 127.0.0.1 --rtsp-port 8554 --rtsp-path /dice
```

默认播放地址：

```text
rtsp://127.0.0.1:8554/dice
```

解码器策略：

- `decoder: "auto"`：检测兼容的 V4L2 M2M 后优先 `spacemitdec`，失败回退 `jpegdec`。
- `decoder: "hw"`：强制要求硬件 M2M，硬件不可用时失败。
- `decoder: "sw"`：强制 `jpegdec` 软件解码。

## 验证边界

启动时会打印：

- 实际模型输入和 13 个输出形状。
- OpenCL 设备。
- SpaceMIT EP affinity。
- 摄像头节点、协商分辨率/FPS 和实际 decoder。

验证时需要区分：

- OpenCL 前处理是否成功。
- ORT/SpaceMIT EP 是否成功运行。
- EP 线程是否使用请求的 CPU 核。
- 是否仍有 CPU fallback。

模型二进制被 `.gitignore` 忽略；Git 只保存源码、配置、README 和模型说明。
