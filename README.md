# SpaceMIT K3 Dice Arena Demo

这是 RV 峰会 / 开发者大会「机械臂骰子挑战」的第一版整体效果原型：

- `web/`：大屏 Web 前端，完成游戏列表、规则确认、同步倒计时、双方摇骰、同时开盖、视觉分析动画、胜负播报和再来一局；摇骰阶段为 10 秒，最后 3 秒使用红色高对比倒计时和浏览器提示音；各游戏的 `manifest.json` 集中维护 TTS 文案与默认音色/语速。
- `backend/server.py`：K3 板端轻量 HTTP bridge；通过视觉裁决功能包调度 YOLOv8 runtime，并使用独立结构化事件通道和 SSE 将进度/结果推送给网页。
- `vision/yolov8_adjudicator/`：通用 YOLOv8 K3 摄像头 runtime，负责 OpenCL 前处理、SpaceMIT ONNX Runtime EP、GStreamer 摄像头、稳定检测、场景几何辅助和快照；游戏规则与 LLM 复核由 Python provider 调度。
- `tts/qwen3-tts/`：迁移的 Qwen3-TTS 0.6B + SpaceMIT `llama-server` 服务；网页通过后端代理获取 24 kHz 单声道 WAV。
- `tts/moss-tts-nano/`：迁移的 MOSS-TTS-Nano SpaceMIT EP runtime 源码与板端交付目录，布局与 `tts/qwen3-tts/` 一致；模型、riscv64 Python 包和 native 库按该目录 `.gitignore` 保留为板端运行时文件。
- `backend/components/tts_moss_nano/`：MOSS-TTS-Nano 组件适配器；调用仓库内 runtime，按文本 chunk 流式返回 WAV。

文档索引见 [`docs/README.md`](docs/README.md)。想了解端到端请求如何调度，请阅读
[`FRAMEWORK_DISPATCH.md`](FRAMEWORK_DISPATCH.md)；想接手或修改代码，请先阅读
[`AI_PROJECT_CONTEXT.md`](AI_PROJECT_CONTEXT.md) 和 [`CLAUDE.md`](CLAUDE.md)。

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
# Qwen3-TTS（可选 provider；音频播放器自动检测为 aplay）
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

大模型的 endpoint、model 和 API key 统一配置在 `backend/components/vision_yolov8_adjudicator/config.json` 的 `llm` 段。该文件被 Git 跟踪，**仓库必须保持私有**；不要把 key 写进网页或日志。修改后重启 `scripts/start_web.sh` 生效。如果没有配置 key，`/api/health` 会显示 `llm_configured:false`，进入开盖后的视觉裁决阶段会明确提示未配置，而不是使用随机骰子或直接判定胜负。

页面交互流程：

1. 选择游戏列表中的「摇骰子」，点击「进入摇骰子」；
2. 点击「我明白了」进入准备状态；
3. 点击「开始摇骰」，网页执行 3、2、1 倒计时；
4. 当前由人手实际摇动骰盅，摇骰阶段持续 10 秒；剩余 3、2、1 秒时倒计时数字变红并播放提示音，完成后也可以提前点击「停止摇骰」；
5. 人手打开双方骰盅，开盖过场页面停留 2 秒后自动播放 `3、2、1` 倒计时；
6. 倒计时结束后页面自动进入 YOLOv8 + 大模型复核的真实识别流程；
7. 点击「再来一局」回到准备状态。

实体按键：绿色按键发送 `Enter`，用于确认、进入和开始；红色按键发送 `Escape`，用于返回或取消；蓝色按键发送 `ArrowDown`，用于向下选择或重听规则；黄色按键发送 `ArrowUp`，用于向上选择。摇骰进行中继续使用页面上的“停止摇骰”按钮。

## 配置游戏状态机与语音

> 切换 TTS provider、调整音色/语速/流式参数的完整步骤见 [`TTS配置与切换指南.md`](TTS配置与切换指南.md)。

配置分两层：**全局配置 `backend/config.json`**（部署级：引擎槽位、默认音色/语速、语音总闸，对所有游戏生效，字段见 [`backend/参数说明.md`](backend/参数说明.md)）与**游戏 manifest `backend/games/<game_id>/manifest.json`**（该游戏怎么玩：状态机、台词、词表、vision_profile；不写的槽位继承全局）。游戏流程由 manifest 的 `state_machine` 节点声明，**后端是唯一权威**：前端只提交意图（实体按键/页面按钮）并渲染事件流，不再自行推进状态或决定播报时机。台词内联在状态的 `speech` 动作里，每条动作可选择实时 TTS（本地/远程槽位）或已有 WAV：

```json
"state_machine": {
  "schema_version": 1,
  "initial": "rules",
  "states": {
    "rules": {
      "ui": {"view": "rules", "title": "游戏规则", "copy": "……"},
      "on_enter": [
        {"action": "speech", "mode": "tts_local", "text": "欢迎来到摇骰子游戏。……"}
      ],
      "on_intent": {
        "confirm": {"to": "ready"},
        "repeat": {"actions": [{"action": "speech", "mode": "tts_local", "text": "……"}]},
        "back": {"exit": true}
      }
    },
    "open_reveal": {
      "on_enter": [
        {"action": "speech", "mode": "audio", "audio": "audio/停.wav", "text": "停！", "await": true},
        {"action": "speech", "mode": "tts_local", "text": "准备好了没有？三，二，一,开盖！"}
      ],
      "duration": 4,
      "on_expire": {"to": "vision_countdown"}
    },
    "result": {
      "on_enter": [{
        "action": "speech",
        "select_by": "winner_role",
        "cases": {
          "PLAYER": {"mode": "tts_local", "text": "……{player_score}……{agent_score}……"},
          "AGENT": {"mode": "tts_local", "text": "……"},
          "TIE": {"mode": "tts_local", "text": "……"}
        }
      }],
      "on_intent": {"new_round": {"to": "ready"}, "back": {"exit": true}}
    }
  }
}
```

要点：

- 状态机是**显式命名的有向图**：`to` 按状态名引用，支持任意跳转、回跳（如 `analysis_failed --retry--> analysis`）与跳过；删除状态时改掉引用它的边即可，悬空引用在加载时报错。
- 触发器三类：`on_intent`（前端按键意图）、`duration` + `on_expire`（计时器，`tick_seconds` 可配，倒计时默认 0.9 秒还原舞台节奏）、`on_event`（后端内部事件，如 `adjudication.result`/`adjudication.diagnosis`）。
- `speech` 动作 `mode` 只有 `tts_local`/`tts_remote`/`audio` 三种；`await: true` 表示后端等待前端播放完成回执（`speech_done`）后才继续推进，保住「停 → 开盖词 → 4 秒过场」的节奏；`select_by: winner_role` 按裁决结果选台词，`{player_score}`/`{agent_score}` 占位符由引擎渲染。
- `audio` 模式从该游戏目录读取 WAV（如 `audio/停.wav`），拒绝绝对路径和 `..` 越界。游戏级 `voice`/`speed` 是 TTS 默认参数，单条动作可覆盖。
- 未来接机械臂时，在对应状态加一条新动作类型（如 `{"action": "robot", "command": "shake_dice"}`）并注册对应执行器与 `command` 类型功能包即可，无需改引擎和前端。
- manifest 支持热加载（mtime 检测，保存后下一局生效；坏配置保留最后可用版本）。正在跑的一局使用创建时的状态机快照。修改 manifest 结构后无需重启后端。

## 语音输入（ASR 语音确认）

除按键外，游戏可开启语音作为第二种意图输入。骰子游戏当前在 `rules` 状态支持：对着麦克风说「确认」等价于按绿色按钮（提交 `confirm` 意图），「重复/再来一遍」重播规则，「返回/退出」退出。

```jsonc
// backend/games/dice/manifest.json —— 游戏层：开关与触发词
"asr": {
  "enabled": true,                       // 游戏级开关，热加载（改后下一局生效）
  "phrases": {                           // 意图 → 触发词表，可自由增删
    "confirm": ["确认"],
    "repeat": ["重复", "再来一遍"],
    "back": ["返回", "退出"]
  }
}
// backend/config.json —— 全局层：识别引擎与总闸
"providers": { "asr": "asr_zipformer" },
"asr_enabled": true                      // 总闸：false 时所有游戏语音失效
```

工作方式：

- 识别由板端 `asr/zipformer-streaming`（真流式 Zipformer，SpaceMIT EP，RTF≈0.24）完成，`asr_zipformer` 功能包在**回合期间**按需拉起 `arecord | stream_asr --pcm --jsonl` 子进程对，回合结束自动停麦；麦克风跟随**系统默认输入设备**（桌面声音设置里换，无需改配置）。
- **播报闸**：语音输入只在台词播报结束后有效——TTS 播报期间说的触发词会被忽略（防止游戏自己的播报触发自己）。不想等播报就按实体按键，按键不受此限制。
- 触发词做子串匹配（识别文本去空格转小写后包含触发词即命中），所以「那我就确认了」也能确认。词表应选日常口语中不常出现的词，避免播报结束后旁人闲聊误触发。
- 引擎零侵入：语音意图与按键走同一条 `submit_intent` 路径，不适配当前状态的词会被静默忽略（如裁决阶段说「确认」）。
- 新游戏/新 provider：功能包继承 `AsrProvider`（`core/asr.py`）实现 `start_session`/`stop_session`，游戏 manifest 声明 `asr` 节即可；ASR 故障不影响按键流程。

## K3 后端接口

```text
GET  /api/health                       查看 bridge、YOLOv8、LLM 和 TTS 状态
GET  /api/tts/health                   检查当前选中的 TTS provider
POST /api/tts/stream                   单次提交整段文本，按 WAV 帧持续返回
POST /api/tts/synthesize               手工调试：单段文本转一个 WAV
POST /api/game/rounds                  创建一局权威状态机对局，返回 round_id
POST /api/game/rounds/<id>/intents     提交意图（按键动作与 speech_done 回执）
GET  /api/game/rounds/<id>             查询对局快照（状态、事件、结果）
GET  /api/game/rounds/<id>/stream      SSE 推送对局事件（state_changed/speech/tick/裁决透传）
POST /api/game/rounds/<id>/speech      按指令 id 拉取台词音频帧（audio 读 WAV / TTS 流式）
POST /api/game/rounds/<id>/cancel      取消对局（浏览器刷新即放弃，新对局自动取消旧对局）
POST /api/adjudicate                   调试入口：直接启动一轮视觉裁决，返回 job_id
GET  /api/adjudicate/<job_id>          兼容查询任务快照（旧客户端可轮询）
GET  /api/adjudicate/<job_id>/events   查询结构化裁决事件
GET  /api/adjudicate/<job_id>/stream   SSE 推送结构化进度和最终结果
POST /api/adjudicate/<job_id>/cancel   取消当前裁决任务
```

游戏对局的正常入口是 `/api/game/rounds` 系列：前端创建 round 后按状态机提交意图并渲染事件流；`/api/adjudicate` 保留为独立调试入口，走同一条 provider 管线。

旧的 `/api/analyze...` 路由仍作为迁移别名保留。未来用于目标坐标/空间位置的视觉定位器应使用独立接口和路由，不复用裁决接口。

一次裁决只允许一个 YOLOv8 runtime 运行，避免多个会话同时争用 K3 的 TCM/算力资源。
runtime 默认常驻预热，空闲时保持摄像头和视频链路但不做推理；点击裁决后才计稳定帧，
结果按 profile 的规则生成。LLM 与 YOLO 一致时使用共识结果，LLM 成功但不一致时使用
LLM 不一致时先复问一次：复问与 YOLO 一致则维持 YOLO，两次一致才允许覆盖（但点数相同的平局是算术事实，永不被覆盖成胜局），复问无定论回退 YOLO；LLM 超时也受控回退到 YOLO；其他失败、数量不符或超时进入错误，不会用网页随机数据兜底。

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

`vision/yolov8_adjudicator/config.json` 是 YOLO runtime 的硬件、推理、RTSP 和 WebRTC 基础地址唯一默认来源。`backend/components/vision_yolov8_adjudicator/config.json` 只保存 Provider 的 runtime 路径/生命周期与 LLM endpoint、model、API key（仓库须保持私有）。游戏 manifest 只声明自己的 `video.path`，完整播放地址由 runtime 配置的 `video.webrtc_base_url` 与 path 安全拼接。


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

后续接入时，可以把前端的两个“人手按钮”替换成 ROS2 / Agent 事件：

- `startShake`：下发摇骰语义指令；
- `stopShake`：下发停骰语义指令；
- 开盖过场：摇骰结束后自动等待 2 秒，再开始视觉倒计时与采集识别；
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
tts_gptsovits -> tts provider，HTTP 客户端调用 Tailscale 内另一台 GPU 主机上的 GPT-SoVITS v2ProPlus（9873 按音色名调用），真流式 PCM 实时包装为 WAV 帧
```

GPT-SoVITS 组件没有本地 lifecycle（服务由其所在主机的 `start.sh`/`stop.sh` 管理），
`backend/components/tts_gptsovits/config.json` 的 `runtime.base_url` 是服务地址的唯一配置点，
IP 变化只需改这一行；`voice.name` 是默认音色（需先在 9873 管理界面注册）。切换方法同上：
把游戏 manifest 的 `providers.tts` 改为 `tts_gptsovits` 后重启。健康检查：
`curl "http://127.0.0.1:8080/api/tts/health?provider=tts_gptsovits"`。

当前骰子游戏默认选择 `tts_moss_nano`；`tts_qwen3` 仍是可选的本地 provider，可通过游戏
manifest 的 `providers.tts` 切换。视觉裁决器的模型、类别映射、规则、LLM prompt、视频
path、超时、裁决前置等待（`lifecycle.pre_adjudication_wait_seconds`，检测开始前的静默等待，
不占用裁决超时预算）和结果保持时间统一放在游戏 `manifest.json` 的 `vision_profile` 中；不要再创建
同目录的外置 `vision_profile.json`。视觉 runtime 的摄像头、推理、RTSP 和 MediaMTX
WebRTC 基础地址只在 `vision/yolov8_adjudicator/config.json` 保存部署默认值。

`vision_yolov8_adjudicator` 是当前 YOLOv8 实现，`role=adjudicator` 表示它在系统里的职责。旧 ID `vision_yolo` 仅作为一次性 registry 迁移别名，不再作为独立组件注册。以后即使新增的空间定位模块也使用 YOLO，也必须注册为 `role=localizer` 并继承 `VisionLocalizerProvider`，不能接入裁决器插槽。

游戏通过 `manifest.json` 的 `providers` 选择具体实现：

```json
"providers": {
  "vision_adjudicator": "vision_yolov8_adjudicator",
  "tts": "tts_moss_nano"
}
```

`vision_adjudicator` 只返回物理侧 `LEFT`、`RIGHT` 或 `TIE`。玩家和 Agent 的身份由游戏
manifest 顶层 `participants` 映射，属于上层业务，不由视觉功能包解释。

临时切换到板端 MOSS-TTS-Nano：把 `backend/games/dice/manifest.json` 的 `providers.tts` 改为 `tts_moss_nano`，然后重启：

```bash
scripts/stop_web.sh
scripts/start_web.sh
```

MOSS 组件默认使用仓库内路径：

```text
tts/moss-tts-nano
```

MOSS 组件直接调用板端 runtime 的 `on_pcm_chunk` 回调：每个文本 chunk 解码完成后立即作为一个 WAV 帧送入
`/api/tts/stream`，网页收到首帧就开始播放，后续 chunk 继续生成并播放。当前是 chunk 级流式，不是逐 codec 帧真流式。
如果使用其他 MOSS 交付目录，只需要在 `backend/components/tts_moss_nano/config.json` 中调整
`runtime.root`、`runtime.model_dir`、`voice` 等配置；默认路径已经是仓库内的
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

切换当前 Web 服务的默认 TTS：把游戏 manifest 的 `providers.tts` 改为目标 provider id（如 `tts_new`），然后重启：

```bash
scripts/stop_web.sh
scripts/start_web.sh
```

组件包可在 `manifest.json.lifecycle` 声明自己的启动/停止命令；`scripts/start_web.sh` 和 systemd Web 服务会启动当前选中的 TTS provider，而不是硬编码启动 Qwen3。没有本地生命周期、由外部服务管理的 provider 可设置 `TTS_AUTOSTART=0`。新增/删除组件或修改游戏 `providers` 后需要重启后端，使注册表重新扫描。

正式 TTS 请求只传 `game`、`text`、`voice` 和 `speed`，由后端根据游戏配置选择 provider；请求体中的 `provider` 不会覆盖后端选择。查看组件和运行状态：

```bash
curl http://127.0.0.1:8080/api/components
curl http://127.0.0.1:8080/api/health
```

YOLO 结果不再依赖 stdout 日志。新版本 C++ 程序支持 `--event-fd FD`，通过继承的文件描述符输出 JSONL 事件；stdout/stderr 只保留诊断信息。后端把事件写入任务事件流，并通过 `/api/adjudicate/<job_id>/stream` 以 SSE 增量推送给前端。2026-08-27 已在 K3 完成重新编译、自测、摄像头测试和 YOLO + LLM 完整分析；尚未升级的旧二进制仍支持 `[RESULT]` 兼容解析，但这只是过渡路径。
