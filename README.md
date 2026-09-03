# YOLOv8-seg 单路摄像头推理

这个项目面向 SpaceMIT K3 板端，使用单路 V4L2 MJPEG 摄像头完成：

```text
摄像头 → GStreamer/V4L2 → NV12 → OpenCL 前处理 → SpaceMIT EP
       → YOLOv8-seg 后处理 → mask/框叠加 → OpenCV 窗口显示
```

当前实现基于两个已经验证过的板端项目：

- `/home/spacemit/projects/yolo-demo`：YOLOv8-seg 的 13 输出后处理和 mask contour 映射思路。
- `/home/spacemit/projects/dice-game/yolov8_objdetect`：GStreamer 摄像头、`spacemitdec` 能力检测、`jpegdec` 回退、OpenCL NV12 前处理和 SpaceMIT EP 绑核方式。

## 配置

默认配置在 `/home/spacemit/projects/dice-game/yolov8_segdetect/config.json`：

```json
{
  "model": "models/yolov8n-seg.q.onnx",
  "camera": 1,
  "device": "",
  "width": 1280,
  "height": 720,
  "fps": 24,
  "intra_threads": 2,
  "ep_affinity": "12;13",
  "conf": 0.25,
  "iou": 0.45,
  "max_detections": 100,
  "queue_depth": 2,
  "display_enabled": true,
  "decoder": "auto",
  "focus": -1,
  "zoom": -1
}
```

`ep_affinity` 通过 SpaceMIT EP 选项 `SPACEMIT_EP_INTRA_THREAD_AFFINITY` 设置。配置中的核数量必须与 `intra_threads` 一致。当前默认是 `intra_threads=2`、`ep_affinity="12;13"`，只绑定 EP 推理线程；不会把整个进程的所有线程都绑定到 12、13。

`device` 非空时优先于 `camera`。板上 C920 当前应优先使用：

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
  --camera 1 \
  --ep-affinity '12;13'
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
