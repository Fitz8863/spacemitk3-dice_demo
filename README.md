# SpaceMIT K3 Dice Arena Demo

这是 RV 峰会 / 开发者大会「机械臂骰子挑战」的第一版整体效果原型：

- `web/`：大屏 Web 前端，完成游戏列表、规则确认、同步倒计时、双方摇骰、同时开盖、视觉分析动画、胜负播报和再来一局；各游戏的 `manifest.json` 集中维护 TTS 文案与默认音色/语速。
- `backend/server.py`：K3 板端轻量 HTTP bridge；开盖后按局启动 YOLOv8 C++ 进程，使用独立结构化事件通道和 SSE 将进度/结果推送给网页。
- `vision/yolov8_objdetect/`：迁移的 YOLOv8 K3 摄像头推理工程，保留 OpenCL 前处理、SpaceMIT ONNX Runtime EP、GStreamer 摄像头、骰子分区求和和 LLM 复核逻辑。
- `tts/qwen3-tts/`：从板端 `/home/spacemit/projects/qwen3-tts` 迁移的 Qwen3-TTS 0.6B + SpaceMIT `llama-server` 服务；网页通过后端代理获取 24 kHz 单声道 WAV。

当前阶段**不接机械臂**，用人手和网页按钮代替机械臂的摇骰、停骰、开盖指令。胜负由 K3 板端摄像头上的 YOLOv8 检测和大模型复核产生，不由网页随机生成。浏览器摄像头只用于页面预览；实际识别直接读取 K3 摄像头设备。

## 在 K3 板端运行前端

当前目录是 K3 板端目录通过 SSHFS 挂载到开发机的路径，文件实际位于板端：

```text
开发机挂载路径：/home/heweijie/spacemit-k3-dev/projects/dice-game/main
K3 板端路径：   /home/spacemit/projects/dice-game/main
```

因此前端应直接在 K3 板端启动，而不是在开发机启动。登录 K3 后执行：

```bash
cd /home/spacemit/projects/dice-game/main
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
cd /home/spacemit/projects/dice-game/main
scripts/stop_web.sh
/usr/bin/python3 backend/componentctl.py stop-selected tts --game dice
```

也可以安装 systemd 服务（可选）：

```bash
sudo cp deploy/dice-arena-tts.service deploy/dice-arena-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dice-arena-web.service
systemctl status dice-arena-web.service

# 仅当需要把 Qwen3 作为独立固定服务管理时，再启用可选服务：
# sudo systemctl enable --now dice-arena-tts.service
```

`web/` 前端和 `backend/server.py` 都只使用 K3 系统自带的 `/usr/bin/python3`，不需要 Node.js 或 npm。网页请求 `/api/adjudicate` 后，bridge 会在板端直接启动 `vision/yolov8_objdetect/build/yolov8_camera`，所以裁决阶段应该能看到 YOLOv8 进程、OpenCL GPU、SpaceMIT EP 和 LLM 请求日志。浏览器在板端通过 `127.0.0.1` 访问时，可以正常申请摄像头权限；如果从其他设备通过 HTTP IP 访问，浏览器可能因非安全上下文限制摄像头权限，但实际识别仍使用 K3 板端摄像头。

首次接入大模型时，在 K3 项目根目录创建不纳入 Git 的凭据文件（不要把 key 写进网页或提交到仓库）：

```bash
cd /home/spacemit/projects/dice-game/main
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

## 修改 TTS 播报文案

每个游戏的页面状态语音集中在 `backend/games/<game_id>/manifest.json` 的 `texts`、`voice`、`speed` 字段，网页通过 `/api/games` 加载，不需要修改 `web/app.js`。例如：

```json
{
  "id": "dice",
  "voice": "default",
  "speed": 1.0,
  "texts": {
    "rules_intro": "双方各摇五颗骰子，停止后同时开盖。",
    "result_player_win": "恭喜你，玩家获胜。玩家点数 {player_score}，Agent 点数 {agent_score}。"
  }
}
```

当前已接入的状态键包括：`rules_intro`、`rules_confirmed`、`shake_started`、`shake_stopped`、`analysis_started`、`result_tie`、`result_player_win` 和 `result_agent_win`。胜负文案支持 `{player_score}`、`{agent_score}` 占位符。`voice` 和 `speed` 会作为每次 TTS 请求的默认参数发送到 K3 后端，`speed` 范围为 `0.25` 到 `4.0`。修改板端挂载目录下的 JSON 后，刷新页面即可生效。

## K3 后端接口

```text
GET  /api/health                    查看 bridge、YOLOv8、LLM 和 TTS 状态
GET  /api/tts/health                检查当前选中的 TTS provider
POST /api/tts/stream                 单次提交整段文本，按 WAV 帧持续返回
POST /api/tts/synthesize              手工调试：单段文本转一个 WAV
POST /api/adjudicate                   启动一轮视觉裁决，返回 job_id
GET  /api/adjudicate/<job_id>          兼容查询任务快照（旧客户端可轮询）
GET  /api/adjudicate/<job_id>/events   查询结构化裁决事件
GET  /api/adjudicate/<job_id>/stream   SSE 推送结构化进度和最终结果
POST /api/adjudicate/<job_id>/cancel   取消当前裁决任务
```

旧的 `/api/analyze...` 路由仍作为迁移别名保留。未来用于目标坐标/空间位置的视觉定位器应使用独立接口和路由，不复用裁决接口。

一次裁决只允许一个 YOLOv8 进程运行，避免多个会话同时争用 K3 的 TCM/算力资源。只有 YOLOv8 识别稳定、且大模型复核结果与 YOLOv8 一致时，后端才返回 `verified:true` 的胜负结果；超时、数量不是 5+5、LLM 失败或结果不一致都会返回错误，不会用网页随机数据兜底。

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
cd /home/spacemit/projects/dice-game/main/vision/yolov8_objdetect
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

仓库提交版本的 `vision/yolov8_objdetect/config.json` 必须保持 `llm.api_key` 为空；运行时优先通过未纳入 Git 的 `.dice-arena.env` 或 `DICE_LLM_API_KEY` 提供密钥。如果板端工作副本的 `config.json` 已含本地密钥，应把它视为本地配置：不要打印、覆盖或直接提交；提交相关配置变更时先清空密钥，提交后再恢复本地值。`--no-llm` 只可用于独立诊断，不能作为网页游戏判胜路径。


## 迁移的 Qwen3-TTS 服务

TTS 在 K3 上作为独立的 `llama-server` 进程运行：

```text
浏览器 Audio
  <- 单次 POST /api/tts/stream；连续接收长度前缀 WAV 帧
  <- backend/server.py（按自然标点复用 qwen3_tts_interactive.py）
  <- 多次 POST http://127.0.0.1:18080/v1/audio/speech（同一个后端请求内部）
  <- tts/qwen3-tts/runtime/bin/llama-server
```

从板端原项目同步资产：

```bash
cd /home/spacemit/projects/dice-game/main
scripts/migrate_qwen3_tts_assets.sh
scripts/start_tts.sh
curl -fsS http://127.0.0.1:18080/health
```

模型约 2 GB，因此 `*.onnx`、`*.gguf`、speaker `*.bin`、参考音频、生成 WAV、日志和 PID 均不提交 GitHub。迁移脚本默认只复制当前 `config.json` 指定的 speaker 文件，不复制 `voice_presets/source_audio` 或其他未配置的音色。浏览器只播放后端当前 TTS provider 返回的 WAV；provider 不可用时会明确报错，不使用浏览器 `speechSynthesis` 掩盖后端故障。

当前 `/v1/audio/speech` 仍然是“一次请求返回一个完整 WAV”，不是逐 PCM 帧接口。网页现在不会切段后发起多个请求：针对一整段规则只发起一次 `/api/tts/stream`，后端内部复用 `qwen3_tts_interactive.py` 的自然标点切分和逐段生成，将每个已经完成的 WAV 以长度前缀帧立即写入同一个 HTTP 响应；浏览器收到第一帧后马上播放，同时继续读取后续帧。因此这是“单 HTTP 请求内的完整 WAV 分段流”，不是逐 PCM 帧流。

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
vision_yolo  -> vision/adjudicator，执行骰子点数、总和、胜负和 LLM 复核
tts_qwen3    -> tts provider，代理 Qwen3-TTS
```

`vision_yolo` 的 ID 表示当前实现，`role=adjudicator` 才表示它在系统里的职责。以后即使新增的空间定位模块也使用 YOLO，也必须注册为 `role=localizer` 并继承 `VisionLocalizerProvider`，不能接入裁决器插槽。

游戏通过 `manifest.json` 的 `providers` 选择具体实现：

```json
"providers": {
  "vision_adjudicator": "vision_yolo",
  "tts": "tts_qwen3"
}
```

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
  "vision_adjudicator": "vision_yolo",
  "tts": "tts_new"
}
```

也可以使用环境变量临时切换当前 Web 服务的默认 TTS：

```bash
scripts/stop_web.sh
DICE_TTS_PROVIDER=tts_new scripts/start_web.sh
```

组件包可在 `manifest.json.lifecycle` 声明自己的启动/停止命令；`scripts/start_web.sh` 和 systemd Web 服务会启动当前选中的 TTS provider，而不是硬编码启动 Qwen3。没有本地生命周期、由外部服务管理的 provider 可设置 `TTS_AUTOSTART=0`。新增/删除组件或修改游戏 `providers` 后需要重启后端，使注册表重新扫描。

TTS 请求也支持传递 `provider` 进行调试，但正式游戏请求只传 `game`，由后端根据游戏配置选择 provider。查看组件和运行状态：

```bash
curl http://127.0.0.1:8080/api/components
curl http://127.0.0.1:8080/api/health
```

YOLO 结果不再依赖 stdout 日志。新版本 C++ 程序支持 `--event-fd FD`，通过继承的文件描述符输出 JSONL 事件；stdout/stderr 只保留诊断信息。后端把事件写入任务事件流，并通过 `/api/adjudicate/<job_id>/stream` 以 SSE 增量推送给前端。2026-08-27 已在 K3 完成重新编译、自测、摄像头测试和 YOLO + LLM 完整分析；尚未升级的旧二进制仍支持 `[RESULT]` 兼容解析，但这只是过渡路径。
