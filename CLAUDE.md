# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目是什么

SpaceMIT K3 板端的「机械臂骰子挑战」交互 Demo。玩家在网页上选「摇骰子」，人手代替（尚未接机械臂）完成摇骰/停骰/开盖，K3 板端用 YOLOv8 C++ 进程识别左右各 5 颗骰子，再由大模型复核，两者结论一致才判定胜负。浏览器摄像头仅作预览，实际识别读 K3 板端摄像头。

**关键文档**（接手先读，本文不重复其全部内容）：
- `README.md` — 运行与接口说明。
- `AI_PROJECT_CONTEXT.md` — 最完整的技术上下文（架构、状态机、安全约束、未来机械臂演进）。
- `vision/yolov8_objdetect/AGENTS.md` — YOLOv8 C++ 子工程的构建/测试/编码规范。
- `tts/qwen3-tts/AGENTS.md` — Qwen3-TTS 子工程的运行/验证/核心亲和性约束。

## 环境与路径（重要）

当前目录是通过 SSHFS 挂载到开发机的 K3 板端目录：

```text
开发机：/home/heweijie/spacemit-k3-dev/projects/dice-game/main
板端：  /home/spacemit/projects/dice-game/main
```

可以在开发机编辑文件，但**编译、跑摄像头、OpenCL/SpaceMIT EP、算力核验证必须在 K3 板端执行**。不要用开发机编译结果声称板端可用。登录板端：`ssh spacemit@<K3-IP>`。

## 当前组件调度实现（2026-08-28）

- `backend/components/<id>/manifest.json` + `provider.py` 是可插拔功能包；`backend/core/components.py` 动态扫描并注册 provider，并校验视觉/TTS 的正式接口。
- 游戏通过语义插槽选择 provider；当前骰子配置为 `providers.vision_adjudicator=vision_yolo` 与 `providers.tts=tts_qwen3`。
- 新 TTS 复制一个功能包并继承 `TtsProvider`，最小实现 `health()`、`synthesize()` 即可接入；需要分段低延迟时再覆盖 `stream()`；可用 `DICE_TTS_PROVIDER=<id>` 切换默认 provider，前端请求保持不变。Provider 可用 `manifest.lifecycle.start/stop` 声明本地模型进程管理命令，`componentctl.py`/`start_web.sh` 会按所选 provider 调度。
- 当前 YOLO 包是 `type=vision, role=adjudicator` 的视觉裁决器，继承 `VisionAdjudicatorProvider` 并实现 `adjudicate()`；以后用于目标坐标的 YOLO 包应使用 `role=localizer`、继承 `VisionLocalizerProvider`，不得混入裁决器插槽。算法名不是职责接口。
- 裁决器通过 `--event-fd` 输出结构化 JSONL 事件，后端从独立管道读取事件；stdout/stderr 只保存诊断日志。2026-08-27 已在 K3 编译并完成结构化事件/SSE/LLM 全链路验证；旧二进制仍兼容 `[RESULT]`。
- 裁决主接口为 `GET/POST /api/adjudicate...`；`/api/analyze...` 仅作为旧客户端迁移别名。
- 2026-08-28 已在 K3 通过 14 个后端测试，重启 Web/TTS，并验证裁决器注册、`/api/adjudicate` 结构化事件到 `verifying`、取消与子进程退出。

## 构建、运行与测试命令

### 后端 + Web（板端）
```bash
cd /home/spacemit/projects/dice-game/main
scripts/start_web.sh   # 启动当前游戏选择的 TTS provider，再启动 backend/server.py
# 停止
scripts/stop_web.sh

# 选择 TTS 后启动整体项目
DICE_TTS_PROVIDER=tts_qwen3 scripts/start_web.sh
DICE_TTS_PROVIDER=tts_moss_nano scripts/start_web.sh
```

健康检查：
```bash
curl http://127.0.0.1:8080/api/health     # 看 yolo_ready / llm_configured / tts_ready
curl http://127.0.0.1:8080/api/tts/health
```

后端与 Web 只依赖板端系统 `/usr/bin/python3`，无需 Node/npm。

### YOLOv8 C++（板端，riscv64 工具链）
```bash
cd /home/spacemit/projects/dice-game/main/vision/yolov8_objdetect
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build build -j4

# 无摄像头自测（模型/OpenCL 冒烟，作为必需验证，无 ctest）
./build/yolov8_camera --model models/best.q.onnx --self-test --no-display

# 短时有界摄像头测试
./build/yolov8_camera --model models/best.q.onnx --camera 1 --no-display --max-frames 30
```

### TTS 手工验证
```bash
curl -f http://127.0.0.1:8080/api/tts/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"骰子游戏开始，请双方准备。","voice":"default","speed":1.0}' \
  -o /tmp/dice-tts.wav
file /tmp/dice-tts.wav   # 期望 RIFF/WAVE, 24 kHz, 16-bit, mono
```

## 架构：三个进程 + 一个 HTTP 服务

代码职责分离，但 Web 与后端部署在同一 Python 服务里（同源，无 CORS；分析进度使用 SSE，不是 WebSocket）：

1. **`backend/server.py`** — 轻量 `ThreadingHTTPServer`，同时提供 `web/` 静态文件和 `/api/*`；负责路由、provider 选择和 `ComponentJob` 生命周期，不包含具体 YOLO/TTS 实现。
2. **`vision/.../build/yolov8_camera`** — **非常驻裁决器子进程**，只在 `/api/adjudicate` 触发后按局启动，得到 `verified:true` 结果即退出。空闲时 `pgrep yolov8_camera` 无结果属正常。
3. **`tts/qwen3-tts/runtime/bin/llama-server`** — 独立常驻进程，监听 `127.0.0.1:18080`，后端通过 `/v1/audio/speech` 代理。

数据流（详见 `AI_PROJECT_CONTEXT.md` 的 mermaid 图）：

```text
浏览器 <-> backend/server.py (HTTP/SSE，轮询仅作兼容回退)
             └─ subprocess -> yolov8_camera (摄像头 -> OpenCL 预处理 -> SpaceMIT EP 推理 -> 5+5 稳定帧 -> LLM 复核 -> JSON)
             └─ HTTP -> llama-server (Qwen3-TTS 24kHz mono WAV)
```

### 后端 API
```text
GET  /api/health                      整体状态
GET  /api/tts/health                  当前 TTS provider 是否可用
POST /api/speech/stream               按 manifest 台词键选择 TTS 或已有 WAV
POST /api/tts/stream                  直接提交文本，按长度前缀 WAV 帧持续返回
POST /api/tts/synthesize              单段文本转一个 WAV（手工调试）
POST /api/adjudicate                     启动一轮视觉裁决，返回 job_id（同时只允许一个任务）
GET  /api/adjudicate/<job_id>            兼容查询状态/阶段/日志/事件/结果
GET  /api/adjudicate/<job_id>/events     查询结构化事件
GET  /api/adjudicate/<job_id>/stream     SSE 推送任务变化
POST /api/adjudicate/<job_id>/cancel     取消
```

任务状态 `queued → running → success | error`；阶段 `queued → starting → detecting → verifying → complete | error`。

### TTS 流协议（`/api/tts/stream`）
长度前缀帧协议，前端 `web/app.js` 的 `readTtsFrames()` 解析：每帧 = 4 字节大端长度 + WAV 字节；结束帧长度 `0`，错误帧长度 `0xffffffff`（后跟 4 字节消息长度 + 消息）。普通 TTS provider 可由基类返回一个完整 WAV 帧；Qwen3 provider 在自己的功能包内按标点分段并逐帧返回。这是「单请求 + 完整 WAV 帧」，**不是**逐 PCM 帧流。

### Qwen3-TTS 启动与调用链

**启动**：`scripts/start_web.sh` → `backend/componentctl.py` → 当前选中的 TTS provider 生命周期脚本 → `tts/qwen3-tts/start_server.sh` 或 MOSS daemon，随后启动 Web backend。核心命令：

```bash
llama-server --media-backend smt --smt-config-dir qwen3-tts-0.6b \
             --host 127.0.0.1 --port 18080 --no-ui
```

`--media-backend smt` + `--smt-config-dir <模型目录>` 让 llama.cpp 走 SpaceMIT 媒体后端（内部加载 ONNX Runtime + SpaceMIT EP + Qwen3-TTS 模型）。`start_server.sh` 还负责：定位 ORT 库（`QWEN3_TTS_ORT_LIB_DIR` → 项目打包 → `/usr/lib` 系统安装）、设 `SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13`、写 `llama-server.pid`、最多 90 秒轮询 `/health` 直到就绪。`scripts/start_web.sh` 默认 `TTS_AUTOSTART=1`，会先拉起它。

**网页调用**（整段文本只发一次请求）：

```text
web/app.js speakState()
  → POST /api/tts/stream  {text, voice, speed}
  → backend/server.py stream_tts()
      ├─ TtsDispatcher 选择 manifest 声明的 provider
      ├─ split_text() 按自然标点切段
      └─ 逐段 POST http://127.0.0.1:18080/v1/audio/speech（TTS_REQUEST_LOCK 串行）
  ← 每段 WAV 以长度前缀帧写回，网页 readTtsFrames() 逐段播放
```

backend 不再动态加载底层 runtime 的交互脚本。Qwen3 的切分和 HTTP 请求位于
`backend/components/tts_qwen3/client.py`，MOSS 的 chunk 协议位于
`backend/core/tts_protocol.py`。底层 `tts/*` 目录只保留模型 runtime 和内部启停脚本，
不提供用户交互式 CLI；调度统一经 `backend/componentctl.py` 和 `TtsDispatcher`。

## 前端状态机（`web/app.js`）

```text
select → rules → ready → countdown → shaking → open → analysis → result → ready/select
```

人手操作按钮 = 未来机械臂 Action 的占位：`startShake` / `stopShake` / `revealDice`（见 `AI_PROJECT_CONTEXT.md` 第 4.3 节映射）。所有播报文案集中在 `backend/games/<game_id>/manifest.json`（状态键可选 `mode=tts` 或 `mode=audio`，TTS 支持 `{player_score}`/`{agent_score}` 占位符），`app.js` 通过 `/api/games` 加载，只引用键、不硬编码文案。

## 必须遵守的约束（非可选）

- **胜负只能由 K3 YOLOv8 + LLM 复核产生**，禁止网页随机骰子兜底。有效结果要求：左右各 5 颗、稳定帧达标、LLM 与 YOLO 结论一致、`verified:true`。
- **LLM key 不进仓库**：优先从 `.dice-arena.env`（`DICE_LLM_API_KEY=...`，chmod 600）或环境变量提供。若板端工作副本的 `vision/yolov8_objdetect/config.json` 已含本地 key，不要打印、覆盖或直接提交；提交时使用清空 key 的版本，完成后恢复用户本地值。不要把 key 写进 `web/`、API 响应或日志。
- **不要回滚/覆盖用户本地修改**：`git status` 里 `vision/yolov8_objdetect/config.json` 常处于未提交的本地修改状态（涉及 LLM 配置），操作前重新确认，提交时只提交本次任务相关文件。
- **CPU/EP 亲和性不要混用**：TTS 用 preferred cores `8,9,10,11,12,13`；YOLO EP affinity 是 `14;15`（`config.json` 的 `ep_affinity`）。`taskset`/环境变量只证明配置意图，不证明 AI Core 实际利用率。
- **不要声称未实现的功能**：当前没有机械臂、没有 ROS2、没有 WebSocket、TTS 不是逐 PCM 流式、底层模型支持 voice cloning 但接口未开放参考音频上传。
- **一次只允许一个 YOLOv8 分析任务**（`create_job()` 里 `active_job_id` 单任务锁），避免争用摄像头/算力。
- 不提交：`.dice-arena.env`、`*.log`、`*.pid`、`build/`、`*.onnx`、`*.gguf`、speaker `*.bin`（见根 `.gitignore`）。

## 子工程规范速查

- **YOLOv8 C++**：C++17，四空格缩进，类 PascalCase、局部/函数 snake_case，RAII 管理 GStreamer/OpenCL/ORT 资源。改动采集或预处理后，必须同时跑 `--self-test --no-display` 和短 `--max-frames` 测试。
- **Qwen3/MOSS TTS**：改动功能包后至少验证 `python3 -m py_compile backend/components/tts_*/*.py`、
  `python3 -m unittest tests.test_tts_a2 -v` 和已部署板端 `/health`；真实模型合成需在 K3 上执行。
