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

`progress` 的 `stable_count` **只统计游戏 profile 真正能裁决的帧**。一帧要进入连续计数
必须同时满足三条，否则把连续计数清零而不是累加：

1. 开启 `vision.divider_detection` 时，该帧定位到了分界线；
2. 按该帧**定位到的分界线**切分（定位不到时才回落到 `vision.divider.position` /
   `orientation`，缺省 `0.5` / `vertical`；与 provider 的 `normalize_observation` 使用
   同一个像素点）后，两侧各恰好 `vision.expected_count` 个目标；
3. 两侧的类别多重集与上一帧完全一致（任一侧点数变化即清零）。

因此 `stable_count` 不会超过 `stable_frames`（外部观察者不会看到 `48/30` 这类越界值），
`stable_frames` 门槛也一定由连续的有效帧满足，而不是拿分界线缺席或数量不达标期间的帧凑数。
profile 未声明 `expected_count` 时第 2 条不生效（runtime 以 `--expected-count 0` 运行），
只保留"检测非空 + 分界线已定位"的旧行为。runtime 不因此固化游戏规则：数量与分区位置
都是 profile 经 `process.py` 转发的参数（`main.cpp` 的 `region_layout_usable`），且
runtime 统计的是模型输出的**全部**类别——profile 的 `class_map` 若只覆盖部分模型类别，
需要额外引入类别过滤参数（当前没有这种 profile）。

### 分界线怎么找（第 1 条判据的实现）

`detect_scene_divider` 依次尝试两个信号，任一成功即算"分界线已定位"：

1. **红蓝地垫分区**（`detect_red_blue_divider`，首选）—— 在 1/4 缩图上逐像素算 `R − B`，
   先按行筛出"左半偏红、右半偏蓝"的有效行（左半中位 > +25 且右半中位 < −25），
   要求有效行 ≥ 全高的 30%；再对有效行逐列取中位得到剖面，在画面中央 25%~75% 走廊
   找 `R − B` 由正变负的过零点，取跨零落差最大的一处并做子像素细化。
2. **印刷深色线**（`detect_black_divider`，回退）—— 灰度 ≤ 45 出掩码，中央走廊内找
   竖直长条（详见该函数注释）。

**为什么首选红蓝**：实测同一帧上，红蓝的通道差在左右两侧相差 **~270 级**（左 +35~+108、
右 −198~−244），而深色线对蓝底的灰度差只有 **~13 级**（地垫条纹 94 vs 蓝区 107）。
光照会抬升亮度，却基本不改变色相，所以红蓝信号在反光下依然可用——这正是原来那条
黑线在反光环境里失效的原因。若换回没有红蓝配色的桌面，红蓝信号会失败并回退到深色线；
两者都没有时 `divider.found=false`，稳定帧计数会一直清零。

**这条线同时就是左右分组的依据**。`emit_observation` 把 `point` 一起发出去，
`normalize_observation` 用 `point / width`（`horizontal` 时是 `point / height`）作为切分
比例，runtime 的区域计数门控用同一个比值切分，所以门控判定的"两侧各 5 个"与 provider
最终分组的左右两侧永远是同一组目标。`vision.divider.position` 只在"这一帧没定位到分界线"
时兜底（例如反光极强、镜头被挡），不再是常规切分位置——把相机挪了之后不需要改 manifest，
接缝移动到哪里就切到哪里。比例落在 `(0, 1)` 之外、`found` 不是 `true`（runtime 定位失败时
仍会发占位 `point:[0,0]`）、或缺少帧尺寸时，provider 一律忽略该 `point` 并回落配置值。

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
  （不再重复保存摄像头、推理、RTSP 或 WebRTC 参数）
  events.protocol
  注：LLM 三件套已于 2026-09-04 迁到 llm 槽位指向的组件
  （backend/components/llm_openai_compat/config.json）；本文件不再有 llm 段
```

游戏级 profile：

```text
backend/games/<game_id>/manifest.json -> vision_profile
  vision.model / class_map / participants / stable_frames
  vision.expected_count / vision.divider.position / vision.divider.orientation
  rule（numeric_compare 或 categorical_relation）
  llm.enabled（判胜前复核；失败诊断已本地化，不读 llm 段）
  llm.reasoning_effort（none/low/high/max，热加载；缺省用组件 config 默认）
  llm.system_prompt / user_prompt_template / allowed_outcomes
  multi_view.views[].camera / multi_view.views[].video.path
  video.path / lifecycle.post_result_hold_seconds
  llm.timeout_seconds
  timeouts.yolo_detection_seconds / adjudication_seconds
```

时间参数只保留四种语义：`yolo_detection_seconds` 限制等待稳定 YOLO 结果的时间，
`llm.timeout_seconds` 限制每次复核请求（失败诊断不再调用 LLM，故不受它约束；
`reasoning_effort: none` 关掉模型思考后实测复核只需 1–4s，默认的思考模式要 5–10s，
预算紧张时优先关思考而不是压这个超时），`adjudication_seconds` 限制从开始检测到产生最终裁决的总处理预算，
`post_result_hold_seconds` 控制裁决成功后继续播放实时画面的时间。最后一个保持时间
在已经产生结果后独立执行，不占用前面的裁决处理预算。

> 热加载边界（2026-09-14 实测）：`vision.expected_count`、`stable_frames`、
> `divider_detection`、`divider.position/orientation`、`model` 都是**启动参数**
> （由 `process.py` 转发为 `--expected-count` 等），resident runtime 跨回合复用，
> 改 manifest 后 runtime 侧只有**重启后**才生效——Python 侧读 manifest 是立即生效的，
> 因此改动后可能出现"Python 按新值校验、runtime 仍按旧值出稳定帧"的短暂不一致
> （此时 provider 的数量校验会兜底成失败诊断）。改这些字段请一并重启 Web 服务。
>
> **"用全局默认"的唯一写法是整行不写**：`llm.reasoning_effort`（以及其它游戏级覆盖项）
> 留空字符串或 `null` 不算"未设置"，而是非法值 → 校验器拒载整个 manifest，服务保留
> 上一份可用配置（页面照常可用，很难察觉）。同理，组件 config 的部署默认值也在**启动时**
> 读入，改它必须重启，否则运行中的进程仍用旧值。

`yolo_detection_seconds` 必须容得下 `stable_frames` 个**有效**帧——要求分界线检测的
游戏里，分界线缺席的帧不计入；声明了 `expected_count` 的游戏里，数量不达标的帧同样
不计入（遮挡、叠放、漏检都会让计数停在低位而不是凑满）。窗口太短会让本来只需重新
摆放就能通过的画面直接落入失败诊断。调大 `stable_frames`、打开 `divider_detection`
或声明 `expected_count` 时要同步放宽这个预算。

新增游戏不需要修改本 runtime：新增模型文件和 manifest 中的 `vision_profile` 即可。
profile 中的 path 只能是 URL 路径（例如 `/dice/`），不能包含主机、查询串或 `..`；
WebRTC 基础地址通过 `vision/yolov8_adjudicator/config.json` 的 `video.webrtc_base_url` 配置，游戏只配置自己的 `video.path`。LLM 的 endpoint/model/api_key 保存在全局 `providers.llm` 槽位指向的组件（当前 `backend/components/llm_openai_compat/config.json` 的 `llm` 段），**不在**视觉组件配置里（该文件被 Git 跟踪，仓库必须保持私有；环境变量覆盖层已于 2026-09-01 移除，JSON 是唯一配置来源）。组件与 runtime 配置的完整字段说明见 `backend/components/vision_yolov8_adjudicator/参数说明.md`。

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
- active runtime 以受控频率覆盖写入一张最新诊断帧；等不到稳定观测、或稳定观测的每侧
  数量不符时，provider 用最近的类别数量、目标数量和场景分界信息生成**失败诊断**
  （本地证据规则，2026-09-14 起不再请求诊断 LLM），不伪造 LEFT/RIGHT/TIE 胜负。
- 单个裁决对象的多路视角并行运行；provider 负责超时、取消和结果后的保持时长。
- 稳定帧快照写入每局私有目录，LLM 消费后立即清理；禁止使用不受控的共享路径。
- 队列深度保持有限，避免摄像头缓冲反向阻塞推理或占满 K3 内存。
- 取消、超时和进程异常必须关闭文件描述符、停止 GStreamer pipeline，并释放相机资源。
- stdout/stderr 日志可以用于诊断，但不能作为 API 业务协议。

硬件验证时应在 K3 上记录 CMake、自测、摄像头协商帧率、OpenCL/EP 初始化和 MediaMTX
路径在线证据；不要仅凭本机编译或端口可访问就宣称板端推理链路可用。
