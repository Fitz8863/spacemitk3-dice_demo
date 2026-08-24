# SpaceMIT K3 YOLOv8 摄像头推理

这个目录实现了 YOLOv8 在 SpaceMIT K3 板端的实时摄像头推理，包含摄像头读取、MJPEG 解码、OpenCL GPU 前处理、队列和显示。

## 数据流

```text
USB 摄像头 V4L2 MJPEG 1280x720@25
  -> GStreamer v4l2src
  -> 优先 spacemitdec code-type=9（K3 VPU 硬件解码）
     无可用 V4L2 M2M 解码节点时自动回退 jpegdec + videoconvert
  -> appsink NV12（只保留最新帧）
  -> OpenCL GPU：Y/UV 上传 + NV12->RGB + resize + letterbox + CHW + FP32/255
  -> SpaceMIT ONNX Runtime EP
  -> YOLOv8 raw output [1, 4+nc, N]
  -> xywh 解码 + 置信度筛选 + class-aware NMS
  -> OpenCV HighGUI 显示
```

当前模型 `models/best.q.onnx` 的文件元数据表明它是 Ultralytics YOLOv8s-relu、6 类骰子模型，导出参数为 `nms=False`、`imgsz=[640,640]`、`end2end=False`。因此它不是 YOLO26 的 `[1,300,6]` end-to-end 输出，程序会在 CPU 侧执行 YOLOv8 的外部解码和 NMS。

类别 ID `0..5` 在显示时对应骰子面 `1..6`。

## 编译

请在 K3 板端 `~/projects/dice-demo` 执行：

```bash
cd ~/projects/dice-demo
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build build -j4
```

如果板端没有 `/opt/opencv-spacemit/lib/cmake/opencv4`，去掉 `-DOpenCV_DIR=...`，以 CMake 实际打印的 OpenCV include/library 路径为准。参考分支使用的是系统/板端 OpenCV 包；应在 K3 板端编译；本工作区的本机环境不作为 riscv64/SpaceMIT 运行时验证依据。

## 运行

### 模型和 OpenCL/SpaceMIT EP 自测（不访问摄像头）

```bash
cd ~/projects/dice-demo
./build/yolov8_camera \
  --model models/best.q.onnx \
  --self-test --no-display
```

### 摄像头显示

```bash
cd ~/projects/dice-demo
export DISPLAY=:0
export XDG_RUNTIME_DIR=/run/user/1000

./build/yolov8_camera \
  --model models/best.q.onnx \
  --camera 1 \
  --width 1280 --height 720 --fps 25 \
  --intra-threads 4 \
  --ep-affinity "8;9;10;11" \
  --queue-depth 3 \
  --conf 0.25
```

参考分支中的摄像头可能只接受 1280x720@24；程序会先尝试请求的 FPS，`--fps 25` 失败时自动回退到 24 FPS。

### 无显示端到端测试

```bash
./build/yolov8_camera \
  --model models/best.q.onnx \
  --camera 1 \
  --width 1280 --height 720 --fps 25 \
  --intra-threads 4 \
  --queue-depth 3 \
  --conf 0.25 \
  --no-display --max-frames 30
```

也可以显式指定设备：

```bash
./build/yolov8_camera --model models/best.q.onnx --device /dev/video1
```

## 双方骰子裁决

程序会自动检测画面中的长黑色分界线，并按检测框中心将骰子分到线的两侧。分界线接近竖直时显示 `LEFT/RIGHT`，接近水平时显示 `UPPER/LOWER`。类别 ID `0..5` 分别计为点数 `1..6`。

只有两侧都**恰好检测到 5 个骰子**时才计算点数和并显示胜方或平局；任一侧不是 5 个、检测框中心压在线上，或未找到黑色分界线时，画面会显示红色 `INVALID` 提示，终端会打印当前两侧数量，并且不会执行胜负判断。

## 黑色分界线检测原理与可靠性

当前实现不是通过训练模型识别黑线，而是使用 OpenCV 的固定规则进行检测：

1. 将当前 BGR 画面缩小到 1/4，降低 CPU 开销；
2. 转为灰度图，把灰度值 `0..75` 的像素判定为黑色；
3. 使用 `5x5` 闭运算连接黑线中的小断点；
4. 提取外部轮廓，只保留跨度至少覆盖缩小画面最长边约 `45%`、面积大于 `300` 的轮廓；
5. 按“面积 × 细长程度”选择得分最高的轮廓，并使用 `cv::fitLine` 拟合出分界线；
6. 计算每个 YOLO 检测框中心点到该直线的有符号距离，按距离正负分到两侧。距离直线不超过约 `4` 像素的检测框会被暂时忽略，不参与计数。

这种方法在**固定机位、白色背景、黑色胶带清晰且分界线是画面中最长的深色条带**时通常有效；当前附件图片属于这种较理想情况。它的优点是实现简单、无需额外模型、运行开销低，且检测不到分界线时会直接拒绝裁决。

但它目前仍是启发式方法，不能保证复杂环境下始终可靠，主要干扰包括：

- 光照变化、阴影或曝光不足导致白色背景出现大面积深色区域；
- 画面中存在更长或更粗的黑色物体、桌边、线缆或其他胶带；
- 黑线断裂严重、被骰子遮挡、反光，或摄像头移动后黑线位置和角度变化较大；
- 固定灰度阈值 `75` 和缩小到 1/4 可能造成漏检、误合并或细节丢失。

严格的“两侧必须都恰好 5 个骰子”规则只能阻止数量不满足时出结果，不能完全防止黑线被误识别后恰好形成 `5+5` 的错误裁决。因此当前方案适合受控场景和原型验证；若要用于长期无人值守，建议增加固定 ROI、亮度自适应阈值、Hough 直线几何约束，以及连续多帧一致性确认，并在画面中保留橙色拟合线供现场检查。

## 重要参数

```text
--model PATH       ONNX 模型
--camera N         使用 /dev/videoN，默认 1
--device PATH      显式指定 V4L2 节点，覆盖 --camera
--width N --height N --fps N
--conf FLOAT       置信度阈值，默认 0.25
--queue-depth N    每级队列深度，默认 3；满时丢旧帧降低延迟
--focus N          手动对焦，默认 0；-1 表示不改动
--zoom N           绝对变焦，默认 181；-1 表示不改动
--intra-threads N  SpaceMIT EP 线程数，默认 1
--ep-affinity LIST  EP 线程绑定的 AI 核，数量必须等于 --intra-threads
--no-display       不创建 HighGUI 窗口
--max-frames N     处理 N 帧后退出
--dump-input PATH  保存首帧 640x640 FP32 CHW 输入
--self-test        初始化 OpenCL GPU 和模型，并执行一次真实 NV12 OpenCL 前处理和推理
```

## 实现边界

- 摄像头阶段优先使用 `v4l2src ! image/jpeg ! spacemitdec code-type=9 ! video/x-raw,format=NV12 ! appsink`。如果没有检测到可用的 V4L2 M2M 节点，则跳过可能触发 MPP 段错误的 `spacemitdec`，自动使用 `jpegdec ! videoconvert ! video/x-raw,format=NV12` 软件解码。
- NV12 默认尽量走浅拷贝：`GstVideoFrame` 映射后，OpenCV `cv::Mat` 只创建 header，不复制像素；`GstreamerFrame::owner` 持有 `GstSample` 和映射状态，并随帧经过 OpenCL 前处理、推理、显示队列，直到最后一个消费者释放后才 unmap/unref。
- 零拷贝只在两个 plane 能表示为一个兼容的 NV12 视图时启用：Y/UV stride 满足要求，且 UV 紧接在 Y plane 后面；如果 VPU/GStreamer 给出分离 plane 或不兼容 padding，则自动逐行深拷贝，并打印 `safe copy fallback`。旧的 `read(cv::Mat&)` 兼容接口会主动 `clone()`。
- 队列深度必须保持有限（默认 3），因为零拷贝会短暂持有 VPU/GStreamer buffer；退出时先停止采集线程、等待工作线程退出并清空应用队列，再向 GStreamer pipeline 发送 EOS、等待 decoder drain，最后释放 pipeline，避免 VPU buffer 生命周期问题。
- 前处理使用 OpenCL GPU kernel 完成 Y/UV 图像采样、NV12 转 RGB、resize、114/128 letterbox、CHW 和 `/255`；主机侧仅负责将 NV12 的 Y/UV 数据上传到 OpenCL。
- 推理线程只访问一个 ORT session；显示在主线程执行，保持 HighGUI 事件循环安全。
- YOLOv8 解码当前支持 `[1,C,N]` 和 `[1,N,C]` 两种三维输出布局；对当前模型预期为 `[1,10,8400]`，即 4 个框通道加 6 个类别通道。
- YOLOv8 输出的框按 `cx,cy,w,h`、类别分数已在导出图中完成 DFL/激活，程序不会再次对类别分数做 sigmoid；随后撤销 letterbox 并做按类别 NMS。

## 当前验证状态

当前工程使用 **OpenCL GPU** 完成 NV12 图像预处理，包括 NV12 转 RGB、resize、letterbox、CHW 排布和归一化；SpaceMIT ONNX Runtime EP 负责模型推理。

已在 SpaceMIT K3 板端验证：

- CMake 编译成功，生成 `build/yolov8_camera`；
- `--self-test --no-display` 成功执行真实 OpenCL 前处理和模型推理；
- `/dev/video1` 可通过 `spacemitdec` 硬件解码 MJPEG；没有可用 V4L2 M2M 解码器时自动回退到 `jpegdec` 软件解码；
- 摄像头请求 25 FPS 时，当前设备可能协商为 24 FPS；
- 无显示端到端测试可使用 `--no-display --max-frames 30`，退出时允许驱动打印 `V4L2_EVENT_EOS event is not support yet` 提示，只要程序最终输出 `Done.` 即表示正常退出。

推荐先执行自测，再执行短时摄像头测试：

```bash
cd ~/projects/dice-demo
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build build -j4
./build/yolov8_camera --model models/best.q.onnx --self-test --no-display
./build/yolov8_camera --model models/best.q.onnx --camera 1 \
  --no-display --max-frames 30
```
