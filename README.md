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

程序默认从当前工作目录的 `config.json` 读取初始化参数，因此在 `~/projects/dice-demo` 中编译后可以直接运行：

```bash
cd ~/projects/dice-demo
./build/yolov8_camera
```

配置文件包含模型、摄像头、分辨率、帧率、SpaceMIT EP 线程/绑核、队列、置信度、对焦/变焦以及 LLM 请求参数。默认示例值为：

```json
{
  "model": "models/best.q.onnx",
  "camera": 1,
  "width": 1280,
  "height": 720,
  "fps": 25,
  "intra_threads": 2,
  "ep_affinity": "14;15",
  "queue_depth": 2,
  "stable_frames": 20,
  "rejudge_on_change": false,
  "conf": 0.50,
  "focus": 0,
  "zoom": 150
}
```

### `config.json` 参数说明

| 参数 | 作用 |
| --- | --- |
| `model` | YOLOv8 ONNX 模型路径；相对路径以程序启动目录为基准。 |
| `camera` | 摄像头编号 `N`，对应 `/dev/videoN`；命令行 `--device` 的优先级更高。 |
| `width` / `height` | 摄像头请求的图像宽度和高度，单位为像素。 |
| `fps` | 摄像头请求帧率；设备不支持时程序可能回退到可用帧率。 |
| `intra_threads` | SpaceMIT ONNX Runtime EP 推理线程数，必须大于等于 1。 |
| `ep_affinity` | EP 线程绑定的 CPU 核，使用分号分隔，例如 `14;15`；数量必须与 `intra_threads` 一致。 |
| `queue_depth` | 前处理和推理队列深度，程序限制为 1–8；数值越大，缓存、延迟和内存占用越高。 |
| `conf` | YOLO 检测置信度阈值，范围为 0.0–1.0。 |
| `focus` | 摄像头手动对焦值；`-1` 表示不修改当前设置。 |
| `zoom` | 摄像头绝对变焦值；`-1` 表示不修改当前设置。 |
| `llm.url` | OpenAI 兼容 API 基础地址，程序请求其 `/chat/completions` 接口。 |
| `llm.model` | 用于复核骰子点数和的模型名称。 |
| `llm.system_prompt` | 约束大模型只根据程序提供的整数点数和进行判断。 |
| `llm.user_prompt_template` | 请求模板，必须保留 `{left_name}`、`{right_name}`、`{left_sum}`、`{right_sum}`。 |
| `stable_frames` | 左右严格各有 5 个骰子且点数组成连续一致达到此帧数后，才调用一次大模型。 |
| `rejudge_on_change` | `false` 表示每次进程只复核一次；`true` 表示点数组成变化后重新稳定计数并再次复核。 |

API Key 不属于 JSON 配置项，只从环境变量 `DICE_LLM_API_KEY` 读取。

命令行参数仍然保留，并在 JSON 加载后覆盖同名配置。例如：

```bash
./build/yolov8_camera --model models/other.onnx --conf 0.60
./build/yolov8_camera --config config.test.json
```

`--self-test --no-display` 不访问摄像头，只验证真实 OpenCL 前处理和 SpaceMIT EP 推理：

```bash
./build/yolov8_camera --self-test --no-display --no-llm
```

摄像头请求 25 FPS 失败时，程序会自动回退到设备可协商的 24 FPS。无显示端到端测试：

```bash
./build/yolov8_camera --no-display --max-frames 30 --no-llm
```

也可以显式指定设备，`--device` 优先于 `--camera`：

```bash
./build/yolov8_camera --device /dev/video1 --no-llm
```

### YOLO + 大模型稳定复核

当 YOLO 连续稳定确认左右两侧各有 5 个骰子后，程序会把两侧点数和发送到 OpenAI 兼容的 `/chat/completions` 接口。每个稳定判定周期只调用一次大模型，并冻结该周期的左右骰子点数组成和点数和；只有大模型返回的 `LEFT`、`RIGHT` 或 `TIE` 与这个 YOLO 快照一致，程序才打印一次最终胜负。

只有 YOLO 连续得到相同的有效 5+5 结果达到 `stable_frames` 次后，程序才调用大模型。任何一帧未检测到分界线、左右数量不是 5 个，或左右骰子点数组成发生变化，连续计数都会清零并重新开始；因此未稳定前不会求胜负，也不会请求大模型。

`rejudge_on_change` 控制完成一次判定后的行为：

- `false`（默认）：保持一次性模式，本进程不再调用大模型；后续画面与已复核快照不同时只隐藏胜负。
- `true`：如果任意一侧的排序后骰子点数组成发生变化，立即把该变化帧作为新一轮稳定计数的第 1 帧。新结果必须再次连续稳定 `stable_frames` 帧且仍满足严格 5+5，才会再次调用一次大模型并输出新结果。短暂误检后恢复到上一次已复核快照时会取消本轮计数，不会重复请求相同结果。

LLM 地址、模型名、system prompt 和 user prompt 模板都在 `config.json` 的 `llm` 对象中配置。模板支持 `{left_name}`、`{right_name}`、`{left_sum}`、`{right_sum}` 四个占位符；程序发送请求前会替换为当前快照值。API Key 不写入代码、配置文件、README 或 Git，只从环境变量读取：

```bash
export DICE_LLM_API_KEY='替换为你的 API Key'
```

如需临时覆盖 JSON 中的地址、模型或稳定帧数，可使用 `--llm-url URL`、`--llm-model NAME`、`--stable-frames N`。`--rejudge-on-change` 临时开启变化后重新判定，`--no-rejudge-on-change` 临时关闭；`--no-llm` 关闭复核。未设置 API Key、请求失败或大模型与 YOLO 结果不一致时，程序不会打印胜负结论。

### 单核运行与 TCM 资源冲突

单核运行时用命令行覆盖 JSON 中的线程参数：

```bash
./build/yolov8_camera --intra-threads 1 --ep-affinity "14"
```

`--ep-affinity` 只设置 EP 工作线程的 CPU 亲和性，不能直接指定内部 TCM/A100 block。若其他 EP 进程仍占用 TCM，或上一次异常退出留下残留状态，推理可能报告 `tcm buffer acquire/release failed`。程序不会在同一个 ORT Session 上重试这类错误，因为 EP 内部锁/TCM 状态已经异常时继续复用 Session 不安全；最终失败会以非零状态退出，并提示排查命令。最终失败时请在板端执行：

```bash
spacemit-tcm-smi -i   # 查看 TCM/运行实例占用
# 确认没有 EP 进程后再执行：
spacemit-tcm-smi -c   # 清理残留 TCM 状态
```

然后确认没有其他 EP 进程占用对应的运行资源，再重新启动本程序。`tcm buffer release failed` 通常不是摄像头、黑线检测或 OpenCL 前处理错误，而是 EP/TCM 资源冲突，需要先处理占用关系。`spacemit-tcm-smi -i` 可查看 block 与 PID；只有确认没有其他推理进程后，才允许使用 `spacemit-tcm-smi -c` 清理残留 block。

## 双方骰子裁决

程序会在画面水平中心的候选区域内检测近似竖直的黑色分界线，并按检测框中心将骰子分到线的左右两侧。水平黑线不会作为分界线接受。类别 ID `0..5` 分别计为点数 `1..6`。

只有两侧都**恰好检测到 5 个骰子**时才计算点数和并显示胜方或平局；任一侧不是 5 个、检测框中心压在线上，或未找到黑色分界线时，画面会显示红色 `INVALID` 提示，终端会打印当前两侧数量，并且不会执行胜负判断。

## 黑色分界线检测原理与可靠性

当前实现不是通过训练模型识别黑线，而是使用 OpenCV 的固定规则进行检测。为避免画面下方的水平黑线或手部等深色区域被误认为分界线，检测规则严格利用分界线的已知几何位置：

1. 将当前 BGR 画面缩小到 1/4，降低 CPU 开销；
2. 转为灰度图，把灰度值 `0..45` 的像素判定为黑色；
3. 只保留画面水平中心约 `30%..70%`、垂直方向约 `5%..95%` 的候选区域；
4. 使用 `3x9` 竖直闭运算连接小断点，避免把水平线连接成候选轮廓；
5. 只接受高度至少覆盖候选区域约 `55%`、拟合方向垂直度不低于 `0.90`、且靠近画面水平中心的轮廓；
6. 用 `cv::fitLine` 拟合竖直分界线，再将骰子中心点投影到直线两侧，分别统计点数总和。距离直线不超过约 `4` 像素的检测框会被暂时忽略，不参与计数。

因此，水平黑线即使很长，也会因不在中心走廊、垂直跨度不足或方向不满足而被拒绝。若摄像头位置改变、分界线不在水平中心或被遮挡超过阈值，需要同步调整这些固定几何约束。该方案比“全画面按面积选最长黑轮廓”更适合当前固定机位，但仍属于启发式检测，不等同于模型识别；现场应通过画面的橙色拟合线检查结果。


## 重要参数

持久化运行参数优先写入 `config.json`；命令行同名选项覆盖 JSON。

```text
--config PATH      JSON 配置文件，默认当前目录 config.json
--model PATH       ONNX 模型
--camera N         使用 /dev/videoN
--device PATH      显式指定 V4L2 节点，覆盖 --camera
--width N --height N --fps N
--conf FLOAT       置信度阈值
--queue-depth N    每级队列深度
--stable-frames N  相同有效 5+5 YOLO 结果达到 N 帧后才调用 LLM，默认 20
--rejudge-on-change 检测结果变化后重新稳定计数并再次复核
--no-rejudge-on-change 保持一次性复核，覆盖 JSON 中的 true
--focus N          手动对焦；-1 表示不改动
--zoom N           绝对变焦；-1 表示不改动
--intra-threads N  SpaceMIT EP 线程数
--ep-affinity LIST EP 线程绑核，数量必须匹配线程数
--llm-url URL      覆盖 config.json 中的 LLM 地址
--llm-model NAME   覆盖 config.json 中的 LLM 模型
--no-llm           关闭 LLM 复核
--no-display       不创建 HighGUI 窗口
--max-frames N     处理 N 帧后退出
--dump-input PATH  保存首帧 640x640 FP32 CHW 输入
--self-test        执行一次真实 OpenCL 前处理和推理
```

## 实现边界

- 摄像头阶段优先使用 `v4l2src ! image/jpeg ! spacemitdec code-type=9 ! video/x-raw,format=NV12 ! appsink`。如果没有检测到可用的 V4L2 M2M 节点，则跳过可能触发 MPP 段错误的 `spacemitdec`，自动使用 `jpegdec ! videoconvert ! video/x-raw,format=NV12` 软件解码。
- NV12 默认尽量走浅拷贝：`GstVideoFrame` 映射后，OpenCV `cv::Mat` 只创建 header，不复制像素；`GstreamerFrame::owner` 持有 `GstSample` 和映射状态，并随帧经过 OpenCL 前处理、推理、显示队列，直到最后一个消费者释放后才 unmap/unref。
- 零拷贝只在两个 plane 能表示为一个兼容的 NV12 视图时启用：Y/UV stride 满足要求，且 UV 紧接在 Y plane 后面；如果 VPU/GStreamer 给出分离 plane 或不兼容 padding，则自动逐行深拷贝，并打印 `safe copy fallback`。旧的 `read(cv::Mat&)` 兼容接口会主动 `clone()`。
- 队列深度必须保持有限（默认 3），因为零拷贝会短暂持有 VPU/GStreamer buffer；退出时先停止采集线程、等待工作线程退出并清空应用队列，再向 GStreamer pipeline 发送 EOS、等待 decoder drain，最后释放 pipeline，避免 VPU buffer 生命周期问题。
- 前处理使用 OpenCL GPU kernel 完成 Y/UV 图像采样、NV12 转 RGB、resize、114/128 letterbox、CHW 和 `/255`；主机侧仅负责将 NV12 的 Y/UV 数据上传到 OpenCL。
- 推理线程只访问一个 ORT session；显示在主线程执行，保持 HighGUI 事件循环安全。
- 单路推理显式设置 ORT `ORT_SEQUENTIAL`、`InterOpNumThreads=1` 和 SpaceMIT EP `SPACEMIT_EP_INTER_THREAD_NUM=1`，避免单核配置继承多会话/多流设置；TCM acquire/release 错误不在原 Session 上重试，推理线程失败时以非零状态退出并打印 TCM 占用排查命令。
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
