# YOLOv8 摄像头 Runtime（SpaceMIT K3）

本目录是 `vision_yolov8_adjudicator` 使用的私有硬件 runtime。它只负责摄像头采集、
OpenCL 预处理、YOLOv8 推理、稳定观测、稳定帧快照和 RTSP 发布，不负责游戏规则、
胜负语义或云端大模型请求。游戏差异由后端的
`backend/games/<game_id>/manifest.json` 的 `vision_profile` 节点描述，Python provider 负责读取 profile、
聚合多视角结果、调用无状态多模态 LLM 并生成最终裁决。

## Runtime 数据流

```text
K3 摄像头
  -> GStreamer V4L2/MJPEG 解码（优先硬件解码，失败时安全回退）
  -> NV12 最新帧队列
  -> OpenCL：NV12 -> RGB、resize、letterbox、CHW、归一化
  -> SpaceMIT ONNX Runtime EP
  -> 通用 detection 事件 + stable snapshot
  -> GStreamer H.264/VPU 编码 -> MediaMTX RTSP 发布 -> WebRTC
```

MediaMTX 由部署管理。runtime 配置中的 `video.webrtc_base_url` 提供部署基础地址，浏览器使用后端根据游戏 manifest 中的
profile path 合成的
WebRTC URL；runtime 自身产生的 RTSP/内部地址不能直接作为浏览器地址。

## 编译（在 K3 板端）

```bash
cd ~/projects/dice-game/main/vision/yolov8_adjudicator
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build build -j4
```

如果 OpenCV 安装位置不同，以 CMake 打印的 include/library 路径为准。本机编译不能
替代 riscv64、OpenCL、SpaceMIT EP 和摄像头验证。

## Provider 调度模式

生产环境由 `vision_yolov8_adjudicator` 启动 resident runtime。摄像头、预处理和视频
发布链路提前打开，进程在 `idle` 状态等待控制命令；这避免每局重复创建/销毁硬件资源。
只有收到 `START_ADJUDICATION` 后才开始稳定帧计数并发布 observation，收到最终结果后
可继续保持画面一段时间，之后回到 `idle`。取消或异常时才释放 runtime。

默认 `config.json` 使用 `"yolov8_enabled": false`，表示预热阶段不主动做推理；当进程
通过控制通道运行时，`START_ADJUDICATION` 会按本局 profile 启用 YOLO，
`STOP_ADJUDICATION` 会立即停止推理但保留摄像头和 RTSP/MediaMTX 画面。

启动命令由 provider 内部生成，示意如下（路径和参数来自组件配置，不应由网页拼接）：

```text
build/yolov8_camera --config config.json --no-display --prewarm \
  --control-fd <inherited-fd> --event-fd <inherited-fd> \
  --snapshot-dir <private-round-dir> --view-id <view-id>
```

`config.json` 只保存 runtime、硬件和部署视频基础地址默认值；模型、参与方、稳定帧阈值、规则、提示词、
LLM 超时和每个游戏的视频 path 均由游戏 manifest 的 `vision_profile` 管理。

## vision-control-v1 协议

控制通道和事件通道均为独立的 UTF-8 JSONL 文件描述符。stdout/stderr 仅用于诊断，
不能被当作业务事件解析。

Provider 向 `control-fd` 发送：

```json
{"command":"START_ADJUDICATION","request_id":"job-abc","profile_id":"game-id"}
{"command":"FINAL_RESULT","request_id":"job-abc","outcome":{"kind":"winner","value":"LEFT"}}
{"command":"STOP_ADJUDICATION","request_id":"job-abc"}
{"command":"CANCEL","request_id":"job-abc"}
```

Runtime 向 `event-fd` 发送：

```json
{"event":"started","component":"vision_yolov8_adjudicator","protocol":"jsonl-events-v1"}
{"event":"phase","phase":"idle"}
{"event":"ready","view_id":"front"}
{"event":"video","view_id":"front","url":"rtsp://127.0.0.1:8554/internal"}
{"event":"phase","phase":"detecting"}
{"event":"progress","stable_count":3,"stable_frames":5}
{"event":"diagnostic_snapshot","stable":false,"detections":[],"divider":{"found":false},"snapshot":{"path":"/tmp/private/latest-front.jpg"}}
{"event":"observation","stable":true,"detections":[],"divider":{"found":true},"snapshot":{"path":"/tmp/private/stable.jpg"}}
{"event":"phase","phase":"idle"}
```

`observation` 是通用检测证据，包含 detection 列表和稳定帧图片；runtime 不写入游戏
winner。多视角由 provider 并行启动多个 runtime，并以 `view_id` 区分。LLM 只由 provider
调用一次，将全部稳定帧作为无状态单轮多模态请求。最终结果优先级为：YOLO 与 LLM 一致
使用 `consensus`；LLM 成功但不一致使用 `llm_override`；LLM 超时使用
`yolo_timeout_fallback`；其他失败返回错误。

## 组件与游戏配置边界

组件级配置：

```text
backend/components/vision_yolov8_adjudicator/config.json
  runtime.binary / runtime.working_dir / runtime.config / runtime.mode
  runtime.prewarm_camera / runtime.terminate_grace_seconds
  llm.endpoint / llm.model / llm.api_key
  （不再重复保存摄像头、推理、RTSP 或 WebRTC 参数）
  events.protocol
```

游戏级 profile：

```text
backend/games/<game_id>/manifest.json -> vision_profile
  vision.model / class_map / participants / stable_frames
  rule（numeric_compare 或 categorical_relation）
  llm.system_prompt / user_prompt_template / allowed_outcomes
  multi_view.views[].camera / multi_view.views[].video.path
  video.path / lifecycle.post_result_hold_seconds
  llm.timeout_seconds
  timeouts.yolo_detection_seconds / adjudication_seconds
```

时间参数只保留四种语义：`yolo_detection_seconds` 限制等待稳定 YOLO 结果的时间，
`llm.timeout_seconds` 限制每次大模型请求（包括正常复核和失败原因诊断），
`adjudication_seconds` 限制从开始检测到产生最终裁决的总处理预算，
`post_result_hold_seconds` 控制裁决成功后继续播放实时画面的时间。最后一个保持时间
在已经产生结果后独立执行，不占用前面的裁决处理预算。

新增游戏不需要修改本 runtime：新增模型文件和 manifest 中的 `vision_profile` 即可。
profile 中的 path 只能是 URL 路径（例如 `/dice/`），不能包含主机、查询串或 `..`；
WebRTC 基础地址通过 `vision/yolov8_adjudicator/config.json` 的 `video.webrtc_base_url` 配置，游戏只配置自己的 `video.path`。LLM 的 endpoint/model/api_key 保存在组件配置 `backend/components/vision_yolov8_adjudicator/config.json` 的 `llm` 段（该文件被 Git 跟踪，仓库必须保持私有；环境变量覆盖层已于 2026-09-01 移除，JSON 是唯一配置来源）。组件与 runtime 配置的完整字段说明见 `backend/components/vision_yolov8_adjudicator/参数说明.md`。

## 诊断模式

命令行仅用于硬件自测和问题定位，不是网页游戏的调用接口。常用操作：

```bash
./build/yolov8_camera --help
./build/yolov8_camera --config config.json --self-test --no-display --no-rtsp
./build/yolov8_camera --config config.json --no-display --max-frames 30
./build/yolov8_camera --config config.json --no-yolov8 --max-frames 30
```

`--self-test` 验证真实 OpenCL 前处理和 SpaceMIT EP；`--no-yolov8` 只验证摄像头读取和
视频链路。生产调用不要使用诊断模式绕过 provider 的 profile、快照目录和控制协议。

## MediaMTX 验证

runtime 是 RTSP 发布端，MediaMTX 是接收和分发端。部署应先启动 MediaMTX，再由 provider
启动 runtime；各游戏的 RTSP 发布路径由 profile/部署映射决定。板端可检查：

```bash
gst-inspect-1.0 rtspclientsink
gst-inspect-1.0 spacemith264enc
curl -s http://127.0.0.1:9997/v3/paths/list | python3 -m json.tool
```

网页播放地址使用 `http://<mediamtx-host>:8889/<profile-video-path>/`，不使用 runtime
事件中的 RTSP 地址。若 MediaMTX 路径不可用，前端应提示视频不可用，但不能改变已经
完成的结构化裁决结果。

## 资源和生命周期约束

- resident runtime 在空闲时保持摄像头和视频链路，不做稳定帧计数、不调用 LLM。
- active runtime 以受控频率覆盖写入一张最新诊断帧；YOLO 稳定超时后 provider 使用该帧
  请求诊断 LLM。诊断 LLM 超时或失败时，根据最近的类别数量、目标数量和场景分界信息
  生成 `yolo_fallback` 原因，不伪造 LEFT/RIGHT/TIE 胜负。
- 单个裁决对象的多路视角并行运行；provider 负责超时、取消和结果后的保持时长。
- 稳定帧快照写入每局私有目录，LLM 消费后立即清理；禁止使用不受控的共享路径。
- 队列深度保持有限，避免摄像头缓冲反向阻塞推理或占满 K3 内存。
- 取消、超时和进程异常必须关闭文件描述符、停止 GStreamer pipeline，并释放相机资源。
- stdout/stderr 日志可以用于诊断，但不能作为 API 业务协议。

硬件验证时应在 K3 上记录 CMake、自测、摄像头协商帧率、OpenCL/EP 初始化和 MediaMTX
路径在线证据；不要仅凭本机编译或端口可访问就宣称板端推理链路可用。
