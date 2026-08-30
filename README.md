# SpaceMIT K3 Dice Arena Demo

这是 RV 峰会 / 开发者大会「机械臂骰子挑战」的第一版整体效果原型：

- `web/`：大屏 Web 前端，完成游戏列表、规则确认、同步倒计时、双方摇骰、同时开盖、视觉分析动画、胜负播报和再来一局；各游戏的 `manifest.json` 集中维护 TTS 文案与默认音色/语速。
- `backend/server.py`：K3 板端轻量 HTTP bridge；通过视觉裁决功能包调度 YOLOv8 runtime，并使用独立结构化事件通道和 SSE 将进度/结果推送给网页。
- `vision/yolov8_adjudicator/`：通用 YOLOv8 K3 摄像头 runtime，负责 OpenCL 前处理、SpaceMIT ONNX Runtime EP、GStreamer 摄像头、稳定检测、场景几何辅助和快照；游戏规则与 LLM 复核由 Python provider 调度。
- `tts/qwen3-tts/`：迁移的 Qwen3-TTS 0.6B + SpaceMIT `llama-server` 服务；网页通过后端代理获取 24 kHz 单声道 WAV。
- `tts/moss-tts-nano/`：迁移的 MOSS-TTS-Nano SpaceMIT EP runtime 源码与板端交付目录，布局与 `tts/qwen3-tts/` 一致；模型、riscv64 Python 包和 native 库按该目录 `.gitignore` 保留为板端运行时文件。
- `backend/components/tts_moss_nano/`：MOSS-TTS-Nano 组件适配器；调用仓库内 runtime，按文本 chunk 流式返回 WAV。

当前阶段**不接机械臂**，用人手和网页按钮代替机械臂的摇骰、停骰、开盖指令。胜负由 K3 板端摄像头上的 YOLOv8 检测和大模型复核产生，不由网页随机生成。浏览器摄像头只用于页面预览；实际识别直接读取 K3 摄像头设备。

## 在 K3 板端运行前端

当前目录是 K3 板端目录通过 SSHFS 挂载到开发机的路径，文件实际位于板端：

```text
开发机挂载路径：/home/heweijie/spacemit-k3-dev/projects/dice-game/main
K3 板端路径：   /home/spacemit/projects/dice-game/main
```

因此前端应直接在 K3 板端启动，而不是在开发机启动。登录 K3 后执行：

```bash
cd <repo-root>
scripts/start_web.sh  # 自动启动当前游戏选中的 TTS provider
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
cd <repo-root>
scripts/stop_web.sh
```

如需在没有网页的情况下单独调试 TTS，可直接运行对应功能包的交互脚本。脚本会
自动启动并预热 provider；每次输入一行文字并回车后，音频会通过板端播放器播放：

```bash
# Qwen3-TTS（默认音频播放器自动检测为 aplay）
backend/components/tts_qwen3/scripts/debug_tts.sh

# MOSS-TTS-Nano
backend/components/tts_moss_nano/scripts/debug_tts.sh

# 任意已注册 provider（适用于以后新增的本地或云端 TTS）
python3 backend/tts_debug.py <provider_id>
```

输入 `/quit` 或 `/exit` 退出。也可以指定播放器，例如
`--player ffplay`；设置 `DICE_TTS_PLAYER` 可固定默认播放器。调试脚本只会停止
本次会话自己启动的 TTS，不会停止已由网页或其他服务运行的 provider。

也可以安装 systemd 服务（可选）：

```bash
sudo cp deploy/dice-arena-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dice-arena-web.service
systemctl status dice-arena-web.service

```

`web/` 前端和 `backend/server.py` 都只使用 K3 系统自带的 `python3`，不需要 Node.js 或 npm。网页请求 `/api/adjudicate` 后，bridge 会通过 `vision_yolov8_adjudicator` 启动或复用 `vision/yolov8_adjudicator/build/yolov8_camera`；YOLO runtime 只输出稳定检测证据，LLM 请求由 Python provider 发起。浏览器在板端通过 `127.0.0.1` 访问时，可以正常申请摄像头权限；如果从其他设备通过 HTTP IP 访问，浏览器可能因非安全上下文限制摄像头权限，但实际识别仍使用 K3 板端摄像头。

首次接入大模型时，在 K3 项目根目录创建不纳入 Git 的凭据文件（不要把 key 写进网页或提交到仓库）：

```bash
cd <repo-root>
printf 'DICE_LLM_API_KEY=%s\n' '你的大模型API_KEY' > .dice-arena.env
chmod 600 .dice-arena.env
```

也可以在启动 `scripts/start_web.sh` 前直接导出 `DICE_LLM_API_KEY`。如果没有配置 key，`/api/health` 会显示 `llm_configured:false`，点击“双方已开盖”会明确提示未配置，而不是使用随机骰子或直接判定胜负。

页面交互流程：

1. 选择游戏列表中的「摇骰子」，点击「进入摇骰子」；
2. 点击「我明白了」进入准备状态；
3. 点击「开始摇骰」，网页执行 3、2、1 倒计时；
4. 当前由人手实际摇动骰盅，完成后点击「停止摇骰」；
5. 人手打开双方骰盅，点击「双方已开盖」；
6. 页面展示 YOLOv8 + 大模型复核后的真实识别结果；
7. 点击「再来一局」回到准备状态。

键盘操作：游戏列表用 `↑/↓` 选择、`Enter` 确认；规则页按 `Enter` 表示“我明白了”、按 `↓` 再听一次；摇骰阶段按 `Q` 停止。

## 配置游戏语音

每个游戏的页面状态语音集中在 `backend/games/<game_id>/manifest.json`。每条台词可选择实时 TTS 或已有 WAV；网页只提交状态键，具体播放策略由后端读取 manifest 决定：

```json
{
  "id": "dice",
  "voice": "default",
  "speed": 1.0,
  "texts": {
    "rules_intro": {
      "mode": "audio",
      "audio": "audio/rules_intro.wav",
      "text": "双方各摇五颗骰子，停止后同时开盖。"
    },
    "result_player_win": {
      "mode": "tts",
      "text": "恭喜你，玩家获胜。玩家点数 {player_score}，Agent 点数 {agent_score}。"
    }
  }
}
```

`mode=tts` 使用当前游戏选中的 TTS provider；`mode=audio` 从该游戏目录读取 WAV，例如上述文件应放在 `backend/games/dice/audio/rules_intro.wav`。第一版只接受 WAV，并拒绝绝对路径和 `..` 越界路径。`text` 在 audio 模式下是可选说明。旧的纯字符串条目仍兼容并视为 TTS。

当前状态键包括：`rules_intro`、`rules_confirmed`、`shake_started`、`shake_stopped`、`analysis_started`、`result_tie`、`result_player_win` 和 `result_agent_win`。TTS 胜负文案支持 `{player_score}`、`{agent_score}` 占位符；`voice` 和 `speed` 是游戏级 TTS 默认参数。修改 manifest 或添加 WAV 后需要重启后端，使 manifest 重新加载。

## K3 后端接口

```text
GET  /api/health                    查看 bridge、YOLOv8、LLM 和 TTS 状态
GET  /api/tts/health                检查当前选中的 TTS provider
POST /api/speech/stream              按游戏台词键选择 TTS 或已有 WAV
POST /api/tts/stream                 单次提交整段文本，按 WAV 帧持续返回
POST /api/tts/synthesize              手工调试：单段文本转一个 WAV
POST /api/adjudicate                   启动一轮视觉裁决，返回 job_id
GET  /api/adjudicate/<job_id>          兼容查询任务快照（旧客户端可轮询）
GET  /api/adjudicate/<job_id>/events   查询结构化裁决事件
GET  /api/adjudicate/<job_id>/stream   SSE 推送结构化进度和最终结果
POST /api/adjudicate/<job_id>/cancel   取消当前裁决任务
```

旧的 `/api/analyze...` 路由仍作为迁移别名保留。未来用于目标坐标/空间位置的视觉定位器应使用独立接口和路由，不复用裁决接口。

一次裁决只允许一个 YOLOv8 runtime 运行，避免多个会话同时争用 K3 的 TCM/算力资源。
runtime 默认常驻预热，空闲时保持摄像头和视频链路但不做推理；点击裁决后才计稳定帧，
结果按 profile 的规则生成。LLM 与 YOLO 一致时使用共识结果，LLM 成功但不一致时使用
LLM 覆盖结果，LLM 超时则受控回退到 YOLO；其他失败、数量不符或超时进入错误，不会用网页随机数据兜底。

## 迁移的 YOLOv8 工程

源码、模型和 K3 配置位于：

```text
vision/yolov8_adjudicator/
├── src/
├── models/best.q.onnx
├── config.json
└── CMakeLists.txt
```

在 SpaceMIT K3 板端编译：

```bash
cd <repo-root>/vision/yolov8_adjudicator
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build build -j4

# 先做模型 / OpenCL 自测
./build/yolov8_camera --model models/best.q.onnx --self-test --no-display

# 再做短时摄像头测试
./build/yolov8_camera --model models/best.q.onnx --camera 1 \
  --no-display --max-frames 30
```

当前迁移的模型是 YOLOv8 raw 输出模型，预期输出 `[1, 10, 8400]`。程序会在 CPU 侧执行 YOLOv8 解码和 NMS，并以模型无关的 detection 列表和稳定帧快照交给游戏 profile 解释；不会在 C++ 中固化骰子数量、分区、求和或胜负规则。

`vision/yolov8_adjudicator/config.json` 是 YOLO runtime 的硬件、推理、RTSP 和 WebRTC 基础地址唯一默认来源。`backend/components/vision_yolov8_adjudicator/config.json` 只保存 Provider 的 runtime 路径/生命周期与 LLM endpoint、model；其中 `runtime.config` 显式指向前述 runtime 配置，避免两份硬件配置分叉。真实 API key 仍只通过未纳入 Git 的 `.dice-arena.env` 或 `DICE_LLM_API_KEY` 提供。游戏 manifest 只声明自己的 `video.path`，部署环境可通过 `DICE_MEDIAMTX_WEBRTC_BASE_URL` 覆盖基础地址。


## 迁移的 Qwen3-TTS 服务

TTS 在 K3 上作为独立的 `llama-server` 进程运行：

```text
浏览器 Audio
  <- 单次 POST /api/tts/stream；连续接收长度前缀 WAV 帧
  <- backend/server.py（通过 TtsDispatcher 调用当前 provider）
  <- 多次 POST http://127.0.0.1:18080/v1/audio/speech（同一个后端请求内部）
  <- tts/qwen3-tts/runtime/bin/llama-server
```

从板端原项目同步资产：

```bash
cd <repo-root>
scripts/start_web.sh
curl -fsS http://127.0.0.1:18080/health
```

模型约 2 GB，因此 `*.onnx`、`*.gguf`、speaker `*.bin`、参考音频、生成 WAV、日志和 PID 均不提交 GitHub。迁移脚本默认只复制当前 `config.json` 指定的 speaker 文件，不复制 `voice_presets/source_audio` 或其他未配置的音色。浏览器只播放后端当前 TTS provider 返回的 WAV；provider 不可用时会明确报错，不使用浏览器 `speechSynthesis` 掩盖后端故障。

当前 `/v1/audio/speech` 仍然是“一次请求返回一个完整 WAV”，不是逐 PCM 帧接口。网页针对一整段规则只发起一次 `/api/tts/stream`，后端由 `TtsDispatcher` 选择游戏 manifest 声明的 provider；Qwen3 provider 在内部按自然标点切分并逐段生成，MOSS provider 直接转发 chunk 级 WAV 帧。浏览器收到第一帧后马上播放，同时继续读取后续帧。

手工测试后端代理：

```bash
curl -f http://127.0.0.1:8080/api/tts/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"骰子游戏开始，请双方准备。","voice":"default","speed":1.0}' \
  -o /tmp/dice-tts.wav
file /tmp/dice-tts.wav
```

当前模型配置以板端实际 `tts/qwen3-tts/qwen3-tts-0.6b/config.json` 为准；已迁移配置为 24 kHz、`frontend_threads=2`、`codec_threads=4`、`talker_threads=4`。TTS preferred cores 默认 `8,9,10,11,12,13`，YOLO EP affinity 仍为 `14;15`，两者不要混用。CPU/环境变量配置本身不等于 AI Core 利用率证明，验证时还要检查实际进程映射和运行日志。

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

当前 HTTP bridge 已通过 SSE 推送分析进度和结果。下一阶段接入机械臂时，应继续让后端作为权威状态源；只有需要双向机器人事件或高频画面时，再增加 WebSocket/视频通道，不让 ROS2、视觉和网页 UI 互相耦合。

## 可插拔组件与模型切换

调度层依赖的是 provider 的**职责接口**，而不是模型算法名称；新版协议下不读取 stdout/stderr 来判断业务进度或结果。每个 provider 都是一个独立目录功能包：启用/删除一个功能包只会影响注册表和声明引用它的游戏，不需要在 `server.py` 中增加硬编码 import。


后端组件采用目录功能包形式：每个组件放在 `backend/components/<component_id>/`，至少包含：

```text
manifest.json       # id/type/role/capabilities/enabled/entry
provider.py         # Component 子类，实现统一接口
```

当前组件：

```text
vision_yolov8_adjudicator -> vision/adjudicator，按游戏 profile 执行稳定帧、多视角投票、胜负规则和 LLM 复核
tts_qwen3     -> tts provider，代理 Qwen3-TTS
tts_moss_nano -> tts provider，代理仓库内 `tts/moss-tts-nano` 的 MOSS-TTS-Nano SpaceMIT EP runtime，支持 chunk 级 WAV 流式
```

`vision_yolov8_adjudicator` 是当前 YOLOv8 实现，`role=adjudicator` 表示它在系统里的职责。旧 ID `vision_yolo` 仅作为一次性 registry 迁移别名，不再作为独立组件注册。以后即使新增的空间定位模块也使用 YOLO，也必须注册为 `role=localizer` 并继承 `VisionLocalizerProvider`，不能接入裁决器插槽。

游戏通过 `manifest.json` 的 `providers` 选择具体实现：

```json
"providers": {
  "vision_adjudicator": "vision_yolov8_adjudicator",
  "tts": "tts_qwen3"
}
```

临时切换到板端 MOSS-TTS-Nano：

```bash
scripts/stop_web.sh
DICE_TTS_PROVIDER=tts_moss_nano scripts/start_web.sh
```

MOSS 组件默认使用仓库内路径：

```text
tts/moss-tts-nano
```

MOSS 组件直接调用板端 runtime 的 `on_pcm_chunk` 回调：每个文本 chunk 解码完成后立即作为一个 WAV 帧送入
`/api/tts/stream`，网页收到首帧就开始播放，后续 chunk 继续生成并播放。当前是 chunk 级流式，不是逐 codec 帧真流式。
如果使用其他 MOSS 交付目录，只需要在 `.dice-arena.env` 中调整
`DICE_MOSS_TTS_ROOT`、`DICE_MOSS_TTS_MODEL_DIR`、`DICE_MOSS_TTS_VOICE` 等配置；默认路径已经是仓库内的
`tts/moss-tts-nano`，模型、依赖包和生成音频按该目录 `.gitignore` 管理。

添加新的 TTS 时，继承 `backend/core/tts.py` 的 `TtsProvider`。最小实现只需要提供 `health()` 和 `synthesize()`；基类会把单个完整 WAV 自动包装成一帧 `/api/tts/stream`。如果新模型支持更低延迟的分段生成，再覆盖 `stream()`：

```python
class TtsNew(TtsProvider):
    id = "tts_new"

    def health(self) -> dict: ...
    def synthesize(self, payload: dict) -> tuple[bytes, dict[str, str]]: ...
    # 可选：def stream(self, payload: dict, write_frame) -> None: ...
```

然后将游戏配置改为（无需修改前端请求格式）：

```json
"providers": {
  "vision_adjudicator": "vision_yolov8_adjudicator",
  "tts": "tts_new"
}
```

也可以使用环境变量临时切换当前 Web 服务的默认 TTS：

```bash
scripts/stop_web.sh
DICE_TTS_PROVIDER=tts_new scripts/start_web.sh
```

组件包可在 `manifest.json.lifecycle` 声明自己的启动/停止命令；`scripts/start_web.sh` 和 systemd Web 服务会启动当前选中的 TTS provider，而不是硬编码启动 Qwen3。没有本地生命周期、由外部服务管理的 provider 可设置 `TTS_AUTOSTART=0`。新增/删除组件或修改游戏 `providers` 后需要重启后端，使注册表重新扫描。

正式 TTS 请求只传 `game`、`text`、`voice` 和 `speed`，由后端根据游戏配置选择 provider；请求体中的 `provider` 不会覆盖后端选择。查看组件和运行状态：

```bash
curl http://127.0.0.1:8080/api/components
curl http://127.0.0.1:8080/api/health
```

YOLO 结果不再依赖 stdout 日志。新版本 C++ 程序支持 `--event-fd FD`，通过继承的文件描述符输出 JSONL 事件；stdout/stderr 只保留诊断信息。后端把事件写入任务事件流，并通过 `/api/adjudicate/<job_id>/stream` 以 SSE 增量推送给前端。2026-08-27 已在 K3 完成重新编译、自测、摄像头测试和 YOLO + LLM 完整分析；尚未升级的旧二进制仍支持 `[RESULT]` 兼容解析，但这只是过渡路径。
