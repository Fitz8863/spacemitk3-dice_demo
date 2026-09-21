# vision/yolov10_objdetect —— YOLOv10 目标检测功能包

主项目（dice-game）的第二个视觉功能包，服务于石头剪刀布（rps）等手势类游戏。
与 `vision/yolov8_objdetect` 同骨架（采集/前处理/推理/RTSP 线程 + latest-only 队列），
差异只在检测器与**运行时配置注入的词表**：

- 检测器是 **YOLOv10 end-to-end**（`[1,300,6]` 已解码 xyxy+conf+class_id，免 NMS）；
- 支持 **OpenCL 旋转**（相机横装：`rotate{enabled,direction,angle}`，kernel 内采样重映射）；
- 支持 **ROI 推理裁剪**（`roi{enabled,x,y,w,h}`，模型只看该区域，排除机械臂所在半幅）；
- **包内零内置词表**：`classes`（模型词表）与 `rps_map`（折叠表）全部来自游戏配置。

姊妹试验工程 `~/projects/dice-game/yolov10_objdetect`（板上独立 repo）是本包的前身，
保留了无协议层的裸推流形态，调参仍可在那边做。

## 职责边界（与游戏侧的契约）

本包只做：采集 → 旋转 → ROI 裁剪 → 前处理 → 推理 → 词表折叠 → **稳定检测观测输出**。
分组、规则、输赢、诊断全部在 Python（游戏侧）。

- 事件协议 **`jsonl-events-v2`**，`started.component = "vision_yolov10_objdetect"`；
  Python 适配器（`backend/components/vision_yolov10_objdetect/process.py`）在握手时
  校验 protocol，协议不符立即报错（fail fast）。
- **稳定语义（v2）**：`rps_map` 折叠**之后**的类别多重集连续 `stable_frames` 帧不变
  才发 `observation`（`stable: true`，每轮恰好一条）。折叠先于稳定计数是关键设计：
  palm↔stop（同为 Paper）之间的源类抖动不会清零稳定计数。置信度/框几何不参与签名。
- `rps_mode=false` 时按 `classes` 原始词表上报（id + label），事件行为与 v8 一致。
- 观测事件不带 `divider` 字段（本包不做分界线检测；rps 双方位置固定，不需要）。
- 控制协议 `vision-control-v1`：`START_ADJUDICATION` / `STOP_ADJUDICATION` /
  `CANCEL` / `FINAL_RESULT`（JSONL，control-fd 或 prewarm 时的 stdin）。
- `--prewarm` 常驻模式：只采集+推流，不进 OpenCL/推理，等 START 再激活。

## 数据流

```text
USB 摄像头 V4L2 MJPEG 1920x1080@25（captured）
  -> GStreamer v4l2src -> spacemitdec 硬解（失败自动回退 jpegdec 软解）-> NV12
  -> OpenCL GPU：NV12->RGB + 旋转重映射 + ROI 采样 + resize/letterbox + CHW + /255
  -> SpaceMIT EP（intra_threads/ep_affinity；PPQ 图固定 ORT_ENABLE_BASIC）
  -> [1,300,6] 已解码输出 -> conf 过滤 + no_gesture 过滤 + rps_map 折叠 + ROI 中心门
  -> 稳定判定（折叠后类别多重集）-> observation 事件（event-fd JSONL）
  -> RTSP 推流（spacemith264enc VPU 硬编 -> MediaMTX，画面带框/标签/ROI 白框）
```

旋转后推流帧为竖屏（1080x1920）；detection 坐标始终在**旋转后推流坐标系**。

## 编译（必须在 K3 板端）

```bash
cd vision/yolov10_objdetect
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4        # 零警告（-Wall -Wextra -Wpedantic）
```

## 运行

本包目录**不带 config.json**——按主项目 per-game 配置制，必须用 `--config` 指向
游戏的 `adjudicator_config.json`（rps 用 `backend/games/rps/adjudicator_config.json`）：

```bash
# 自测（不占摄像头：OpenCL + 模型 + 一次推理 + class_id 范围校验）
./build/yolov10_camera --config ../../backend/games/rps/adjudicator_config.json \
    --self-test --no-rtsp

# 手工短跑（60 帧）
./build/yolov10_camera --config <同上> --max-frames 60 --no-rtsp

# 拉流查看
ffplay -rtsp_transport tcp rtsp://<板端IP>:8554/rps/det
```

## 配置键（游戏 adjudicator_config.json 中本包消费的部分）

| 键 | 作用 |
| --- | --- |
| `model` | ONNX 模型路径，相对本配置文件所在目录解析 |
| `classes` | **模型词表**（字符串数组，与模型输出 id 一一对应）。`rps_mode` 或 `filter_no_gesture` 开启时必填——包内无内置词表 |
| `rps_mode` | `true` 启用词表折叠；`false` 按原始 classes 上报 |
| `rps_map` | 折叠表：游戏标签 → 源类名数组（多对一合法；未知类名/空标签/源类重复映射启动即报错）。折叠后 class_id = 标签的声明序（std::map 字母序，Paper=0/Rock=1/Scissors=2），label 字段同时携带标签文本 |
| `filter_no_gesture` | `true` 丢弃 `no_gesture` 类（HaGRID 兜底类，纯噪声） |
| `conf` | 置信度阈值（模型输出已是概率域） |
| `stable_frames` | 折叠后类别多重集需连续重复的帧数（1 = 第一帧即稳定） |
| `rotate` | `{enabled, direction: cw\|ccw, angle: 90\|180}`，OpenCL kernel 采样重映射，90 度模式推流为竖屏 |
| `roi` | `{enabled, x, y, w, h}`（占推流画面比例）：模型只看该区域（手部放大约 1.8×），且只保留框中心落在区域内的检测 |
| `camera`/`width`/`height`/`fps` | 采集规格（旋转前） |
| `intra_threads`/`ep_affinity` | SpaceMIT EP 线程数与绑核（数量必须一致） |
| `focus`/`zoom` | `-1` 表示不动 |
| `rtsp` | `{enabled, host, port}`；**path 不在此声明**——生产由 manifest `video.path` 经 `--rtsp-path` 恒压 |

命令行参数同名覆盖配置；框架注入的参数（`--control-fd/--event-fd/--view-id/
--snapshot-dir/--prewarm/--no-display`）由 Python 适配器（`process.py`）下发。

## 已知边界

- PPQ 量化图（INT8/UINT8 zero point 混用）必须 `ORT_ENABLE_BASIC` 优化级，
  EXTENDED/ALL 会在图优化阶段报错；BASIC 失败自动降级 DISABLE_ALL 重试。
- TCM 僵尸块（其他 EP 进程异常退出遗留）会让推理报 `tcm buffer acquire failed`：
  `spacemit-tcm-smi -i` 查占用、无主时 `spacemit-tcm-smi -c` 清理。
- 板子当前被系统级 cpuset 钉在 0-7 核运行（2026-09-20 记录）；EP 仍能跑满
  24fps/35ms，真正想用 8-15 核需先解除限制。

## 板端验证记录

- 2026-09-21（集成日）：主项目包形态（协议层 v2 + 配置注入词表）首次自测通过——
  `started` 事件握手、`Filtering class_id 33 (no_gesture)`、
  `RPS mode: Paper/Rock/Scissors<={...}`、`Rotation: 90 cw`、
  `Inference crop: 1080x960 at +0,+960 (stream 1080x1920, hand scale x1.78)`、
  OpenCL 25.2ms、模型 `[1,3,640,640]→[1,300,6]`、0 误检。
- 旋转/ROI/解码/推流的实机调参结论继承自前身 demo 工程（README 见那边，
  2026-09-20/21 板端实测：fps_infer 21~23、infer_ms ~35、旋转正确性相关性
  0.999+、`--no-ep` CPU 对照一致）。
