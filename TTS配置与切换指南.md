# TTS 配置与切换指南

> 更新日期：2026-09-02 | 适用版本：`asr-voice-input` 分支
> 本文回答两个问题：**怎么切换 TTS**、**每个参数在哪里改、是什么意思**。
> GPT-SoVITS 服务端自身的部署与接口细节见 [`TTS接口文档.md`](TTS接口文档.md)。

---

## 0. 一条核心规则

**所有配置都是 JSON 文件，没有环境变量覆盖层。** 生效规则分三类：

- **全局配置（`backend/config.json`）与游戏 manifest（`backend/games/<游戏>/manifest.json`）支持热加载**：改完保存即生效（后端检测 mtime 自动重载，下一回合用新值）。写错了不会让游戏消失——后端保留最后一份可用配置并在 Web 日志记录解析错误。**例外：本地 TTS 引擎选择在启动时钉死**，运行期改 `tts_local` 不会切换（只打日志），切换本地引擎必须重启。
- **组件 config.json / vision runtime config 仍需重启**（引擎进程启动时读取）：

```bash
cd /home/spacemit/projects/dice-game/main
scripts/stop_web.sh
scripts/start_web.sh
```

配置分四层，各管一件事（越靠近使用点优先级越高）：

| 层 | 文件 | 管什么 |
|---|---|---|
| **全局层** | `backend/config.json` | **部署级默认**：本地/远程 TTS 槽位、默认音色/语速（所有游戏共享；字段说明见 `backend/参数说明.md`） |
| **游戏层** | `backend/games/dice/manifest.json` | 每句播报文案（三种 mode，见第 3 节）；可覆盖全局槽位/音色/语速（不写即继承全局） |
| **组件层** | `backend/components/tts_<id>/config.json` | 该 provider 的引擎地址、音色、生成参数 |
| **请求层** | 浏览器/API 请求体 | 单次合成的临时覆盖（前端正常情况下不传，全部走配置） |

---

## 1. 三个 TTS Provider 总览

| | `tts_moss_nano`（当前默认） | `tts_qwen3` | `tts_gptsovits` |
|---|---|---|---|
| 引擎 | MOSS-TTS-Nano 100M | Qwen3-TTS 0.6B | GPT-SoVITS v2ProPlus |
| 运行位置 | **K3 板端**（SpaceMIT EP 加速） | **K3 板端**（llama-server） | **另一台 GPU 主机**（Tailscale 内） |
| K3 端口/地址 | 127.0.0.1:18082 | 127.0.0.1:18080 | http://100.95.19.17:9873 |
| 音色 | 克隆音色 `Junhao`（参考 `voice/warm.wav`） | 内置 `anke.spk.bin`，不支持换声 | 服务端注册表音色（当前 `demo_female_zh`） |
| 语速 | 不支持（固定 1.0） | 支持 | 支持 0.5~2.0 |
| 流式 | chunk 级 WAV 帧 | 按标点分段的完整 WAV 帧 | 真流式 PCM 实时包帧（首包最快） |
| 服务管理 | K3 随 Web 自启自停 | K3 随 Web 自启自停 | 对端 `start.sh`/`stop.sh`，K3 侧无生命周期 |
| 配置文件 | `backend/components/tts_moss_nano/config.json` | `backend/components/tts_qwen3/config.json` | `backend/components/tts_gptsovits/config.json` |

---

## 2. 切换默认 TTS（标准操作）

**唯一方法**：改全局配置 `backend/config.json` 的 `providers.tts_local`，然后重启（本地引擎运行期钉死，切换必须重启）。

### 2.1 步骤

1. 编辑 `backend/config.json` 的槽位：

```json
"providers": {
  "tts_local": "tts_moss_nano",
  "tts_remote": "tts_gptsovits"
}
```

- `tts_local`（本地槽）：板端引擎，合法值 `tts_moss_nano` / `tts_qwen3`
- `tts_remote`（远程槽）：云端/远程引擎，合法值 `tts_gptsovits` 或未来新增的远程 provider
- "默认 provider" 指 `tts_local` 槽位，`/api/health` 的 `tts_provider` 字段显示它
- **唯一性约束**：全局与各启用游戏若解析出多个不同的本地引擎，后端拒绝启动（一台演示机只跑一个本地引擎）
- 个别游戏想用不同引擎时，在该游戏 manifest 的 `providers` 里覆盖对应槽位（例如未来的猜拳游戏可用不同声音）

2. 重启：

```bash
scripts/stop_web.sh
scripts/start_web.sh
```

3. 验证生效：

```bash
curl -s http://127.0.0.1:8080/api/health | python3 -c "import json,sys; d=json.load(sys.stdin); print('本地:', d['tts_provider'], d['tts_ready'], '| 远程:', d['tts_remote_provider'], d['tts_remote']['ok'])"
```

### 2.2 注意事项

- **启动脚本会做一致性检查**：如果服务已在运行而配置又改了 provider，`start_web.sh` 会拒绝启动并提示先 stop——按提示先 `stop_web.sh` 再 `start_web.sh` 即可，这是防止新旧 provider 混跑的保护，不是故障。
- **换 provider 必换声音**：三个 provider 音色体系互不相通，`voice` 字段的含义随 provider 变（见第 4 节）。
- **切换前建议先单独验通**目标 provider（第 5 节的 health 检查），避免游戏进行中才发现服务不可达。
- **启动脚本会自动启动全局槽位与各游戏引用到的全部本地 provider**（含台词级 `provider` 钉死），远程 provider 无本地生命周期自动跳过；这样按句混用时切换零等待。
- **运行期改 `tts_local` 不会切换引擎**：后端启动时钉死本地引擎，改配置只会在日志打漂移提示；想切换必须重启。远程槽位无此限制（热加载生效）。

---

## 3. 游戏层参数（dice manifest）

位置：`backend/games/dice/manifest.json`（完整字段参考见同目录 `参数说明.md`）。

### 3.1 默认音色与语速

默认值在**全局配置** `backend/config.json`（`voice`/`speed`，所有游戏共享）。游戏 manifest 可覆盖（写在顶层），单条台词可再覆盖（写在 speech 动作里）——三层阶梯：台词 > 游戏 > 全局。骰子游戏当前不写，直接用全局默认。

- `voice` 的解析规则随 provider：
  - `tts_moss_nano`：只认 `default`（实际音色由组件 config 的 `voice.name` 决定）；传其他值会报错
  - `tts_qwen3`：`default` → 组件 config 的 `voice.default`（当前就是 "default"，引擎内置说话人）
  - `tts_gptsovits`：`default` → 组件 config 的 `voice.name`；也可直接写服务端注册表里的音色 id（当前已注册：`fll`、`leimu`、`violet`、`anke`、`demo_female_zh`、`zanni`）
- `speed`：只对 qwen3 和 gptsovits 生效（gptsovits 会被夹到 0.5~2.0）；moss 恒为 1.0，传其他值会明确报错而不是静默忽略。

### 3.2 每句播报（state_machine 的 speech 动作）—— 三种 mode 按句混用

台词内联在 manifest `state_machine.states.<状态>` 的 `speech` 动作里（2026-09-02 起后端
权威状态机驱动，旧 `texts` 台词表已移除）：

```json
"shake_countdown": {
  "on_enter": [
    {"action": "speech", "mode": "audio", "audio": "audio/warm_321开始.wav", "text": "三，二，一，开始"}
  ],
  "duration": 3, "tick_seconds": 0.9,
  "on_expire": {"to": "shaking"}
},
"analysis": {
  "on_enter": [
    {"action": "speech", "mode": "tts_local", "text": "正在调用视觉判断结果"},
    {"action": "adjudicate"}
  ]
}
```

`await: true` 的 speech 动作要等前端播完回执 `speech_done` 才继续推进（保住「停→开盖」
节奏）；`select_by: winner_role` + `cases` 按胜负选台词。完整 schema 见 README 的
「配置游戏状态机与语音」。

| mode | 走哪个引擎 | 典型用途 |
|---|---|---|
| `audio` | 不合成，直接播 WAV | 节奏/音质固定的开场词（如 321 开始） |
| `tts_local` | 本地槽 `providers.tts_local` | 板端稳定输出，无网络依赖 |
| `tts_remote` | 远程槽 `providers.tts_remote` | 远程 GPU 的更高音质/更快首包 |

- **按句指定具体引擎**（逃生舱）：任意合成 mode 可加 `"provider": "tts_qwen3"` 显式钉住一个 provider，优先于槽位。适合某一句必须固定音色的场景；常规切换请改槽位（一处生效全游戏）。
- `text` 支持 `{player_score}`/`{agent_score}` 占位符，由后端用真实比分渲染。
- 本地槽与远程槽**可以同时使用**：启动脚本会把 manifest 引用到的全部本地 provider 拉起，按句混用无冷启动等待。

---

## 4. 组件层参数（各 provider config.json 全参数）

> 以下均为当前实际值；改任意参数后重启 Web 服务生效。`runtime.kind` 决定校验规则：
> `local` 必须回环地址（防 SSRF），`cloud`/`external` 必须绝对 HTTP(S) 地址。

### 4.1 tts_moss_nano（板端 MOSS）

```json
{
  "runtime":    { "kind": "local", "root": "tts/moss-tts-nano",
                  "model_dir": "models/MOSS-TTS-Nano-100M-ONNX-xslim-dynq",
                  "host": "127.0.0.1", "port": 18082, "base_url": "http://127.0.0.1:18082" },
  "voice":      { "mode": "clone", "name": "Junhao", "reference_audio": "voice/warm.wav" },
  "generation": { "max_new_frames": 120, "voice_clone_max_text_tokens": 24,
                  "first_chunk_text_tokens": 16, "seed": 1234 },
  "startup":    { "warmup_text": "你好，这是 MOSS TTS Nano 在 K3 上的演示。",
                  "start_timeout_seconds": 300 },
  "limits":     { "request_timeout_seconds": 120 },
  "execution_provider": { "name": "spacemit", "intra_thread_num": 4, "inter_thread_num": 1,
                  "intra_thread_affinity": "8;9;10;11", "disable_op_type_filter": "" }
}
```

| 参数 | 作用 | 调整建议 |
|---|---|---|
| `voice.name` | 使用的内置音色名 | 换音色改这里；仅 `mode=clone` 时需要 `reference_audio` |
| `voice.reference_audio` | 克隆参考 WAV（相对 runtime root） | 换参考音频即换音色，3~10s 干净人声效果最好 |
| `generation.first_chunk_text_tokens` | 首个音频块的文本 token 预算 | 调大→首帧更完整但出声更晚；当前 16 是延迟/完整性折中 |
| `generation.voice_clone_max_text_tokens` | 每块最大文本 token | 一般不动；改大可能超 KV 容量（启动时会按模型容量自动收紧） |
| `generation.seed` | 采样种子 | 固定可复现同一声音表现 |
| `startup.start_timeout_seconds` | 启动就绪等待上限 | 板端偶发预热 >300s 时可调大（见第 7 节故障排查） |
| `execution_provider.intra_thread_affinity` | EP 算力核绑定 | **保持 `8;9;10;11`，不要与 YOLO 的 `14;15` 混用** |

### 4.2 tts_qwen3（板端 Qwen3）

```json
{
  "runtime":    { "kind": "local", "root": "tts/qwen3-tts", "model_dir": "qwen3-tts-0.6b",
                  "host": "127.0.0.1", "port": 18080, "base_url": "http://127.0.0.1:18080" },
  "voice":      { "default": "default", "speaker_file": "anke.spk.bin" },
  "generation": { "timeout_seconds": 120, "speed": 1.0, "chunk_chars": 24 }
}
```

| 参数 | 作用 | 调整建议 |
|---|---|---|
| `generation.speed` | manifest 未带 speed 时的默认语速 | 0.25~4.0 |
| `generation.chunk_chars` | 按标点切分的段长上限 | 调小→出声更早、句间停顿略多；调大反之 |
| `voice.speaker_file` | 说话人嵌入文件（模型目录内） | 换内置说话人时改；仓库不跟踪 `.bin`，需板端自备 |

### 4.3 tts_gptsovits（远程 GPT-SoVITS）

```json
{
  "runtime":  { "kind": "external", "base_url": "http://100.95.19.17:9873" },
  "request":  { "text_lang": "zh", "streaming_mode": 3,
                "text_split_method": "cut5", "timeout_seconds": 120,
                "top_k": 15, "top_p": 1.0, "temperature": 1.0,
                "repetition_penalty": 1.35, "seed": -1, "fragment_interval": 0.3 },
  "voice":    { "name": "demo_female_zh" },
  "audio":    { "sample_rate": 32000, "channels": 1 }
}
```

| 参数 | 作用 | 调整建议 |
|---|---|---|
| `runtime.base_url` | **服务地址唯一配置点** | **IP 变化只改这一行**。管理后台 9873（按音色名）；也可改 9880 直连（但本组件按名调用的请求体不适用于 9880，勿混用） |
| `voice.name` | 默认音色 id | 必须是服务端注册表已有音色；查列表：`curl http://<服务IP>:9873/voices` |
| `request.text_lang` | 合成语言模式 | `zh` 为中英混合；11 种模式见接口文档语言表 |
| `request.streaming_mode` | 3=首包最快 / 2=质量优先 / 0=整段 | 组件 stream 链路固定传 3；`synthesize` 固定 0，此值仅作默认声明 |
| `request.text_split_method` | 长文本切句 | 短句播报用 `cut0` 更可控，长段落用 `cut5` |
| `request.timeout_seconds` | 单次 HTTP 超时 | 服务端热切换模型时首请求较慢，可适当调大 |
| `request.repetition_penalty` | 重复惩罚 | **出现复读/卡字就调大**（1.0~3.0，默认 1.35） |
| `request.seed` | 随机种子 | 试音满意后固定数值可复现同一表现；-1 为随机 |
| `request.top_k` / `top_p` / `temperature` | 采样参数 | 越小越稳、越大越有情感；默认 15/1.0/1.0 一般不动 |
| `request.fragment_interval` | 多句之间的静音秒数 | 默认 0.3，嫌句间停顿长可调小 |
| `audio.sample_rate` / `channels` | PCM 包装参数 | **必须与引擎输出一致（32000/单声道）**，改错会变调，勿动 |

> 音质类参数（`top_k` 及以下）均为可选：从 config 里**删除该键**即回退服务端默认；
> `media_type` 与流式 `streaming_mode` 属于帧协议固定值，由代码决定，不作为配置暴露。

---

## 5. 单独验证某个 provider（不动游戏）

### 5.1 健康检查

```bash
# 查指定 provider（不切换）
curl -s "http://127.0.0.1:8080/api/tts/health?provider=tts_gptsovits" | python3 -m json.tool
# 全部组件一览
curl -s http://127.0.0.1:8080/api/components | python3 -c "import json,sys; [print(c['id'], '->', c['health'].get('ok')) for c in json.load(sys.stdin)['components']]"
```

### 5.2 命令行试听（用当前默认 provider）

```bash
curl -f http://127.0.0.1:8080/api/tts/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"骰子游戏测试，三二一，开始。","game":"dice"}' -o /tmp/tts-test.wav
file /tmp/tts-test.wav   # 期望 RIFF/WAVE
```

### 5.3 组件级交互调试（不经 HTTP，直接驱动 provider）

```bash
python3 backend/tts_debug.py tts_gptsovits   # 任意已注册 provider id
# 或板端各组件自带脚本：
backend/components/tts_moss_nano/scripts/debug_tts.sh
backend/components/tts_qwen3/scripts/debug_tts.sh
```

输入一行文字回车即板端播放；`/quit` 退出。调试脚本只停自己拉起的进程，不影响网页正在用的 provider。

### 5.4 GPT-SoVITS 服务端直测（跳过 K3）

```bash
curl -X POST http://100.95.19.17:9873/tts \
  -H "Content-Type: application/json" \
  -d '{"voice":"demo_female_zh","text":"直连服务端测试。"}' -o /tmp/direct.wav
```

---

## 6. 请求层参数（API 调用方参考）

游戏流程内前端不直接调 TTS 接口：状态机下发 speech 指令，前端经
`/api/game/rounds/<id>/speech` 按指令拉帧（provider 由后端按"台词钉死 > 游戏槽位 > 全局槽位"解析）。
手工调试直接调 `/api/tts/stream` 时可用：

| 字段 | 说明 |
|---|---|
| `text` | 必填，≤4000 字符 |
| `voice` | 省略/`default` 走组件默认；显式值按第 3.1 节规则解析 |
| `speed` | 省略走全局/游戏/组件默认；moss 只接受 1.0 |
| `game` | 决定用哪个游戏的 manifest（叠加全局默认）选择 provider |

请求体里的 `provider` 字段**不会**覆盖后端选择——provider 只认配置文件。

---

## 7. 故障排查

| 现象 | 先查什么 | 处理 |
|---|---|---|
| 改了配置没生效 | 是否重启？ | `stop_web.sh` + `start_web.sh`（配置仅启动时读取） |
| `start_web.sh` 报 provider 不一致 | 服务是否带旧 provider 在跑 | 先 stop 再 start（保护机制，见 2.2） |
| `tts_ready: false`（moss） | `tail .runtime/moss-tts-18082.log` | 启动预热偶发超 300s 会自愈；频繁出现可调大 `startup.start_timeout_seconds` |
| `tts_gptsovits` health `ok:false` | 对端服务是否在线 | `curl http://100.95.19.17:9873/voices`；不在线去 GPU 主机 `bash start.sh` |
| 播报中途无声/报 Broken pipe | 多为客户端（浏览器）中途断开 | 偶发可忽略；rule 页频繁出现再查前端 |
| 切 gptsovits 后报 `voice` 相关 400 | 音色 id 是否在注册表 | `curl http://100.95.19.17:9873/voices` 核对，或改 `config.json` 的 `voice.name` |
| 声音变调（gptsovits） | 采样率是否被改 | `audio.sample_rate` 必须保持 32000 |
| 语音整体延迟高 | 首包耗时在哪一段 | 分别测 5.2（全链路）与 5.4（纯服务端）定位是 K3 代理还是引擎 |

---

## 8. 参数速查卡

```text
切本地 provider   →  backend/config.json providers.tts_local（moss / qwen3）+ 重启
切远程 provider   →  backend/config.json providers.tts_remote（gptsovits / 未来新增云端）
某句换引擎        →  state_machine.states.<状态> 内该 speech 动作的 mode（或加 "provider": "<id>" 钉死）
换 moss 音色      →  backend/components/tts_moss_nano/config.json → voice.name / voice.reference_audio
换 gptsovits 音色 →  backend/components/tts_gptsovits/config.json → voice.name（或全局 voice）
改默认语速        →  backend/config.json 顶层 speed（moss 不支持）
改某句文案        →  游戏 manifest 对应状态 speech 动作的 text
改服务地址        →  对应组件 config.json 的 runtime.base_url
改全局/游戏配置后 →  保存即热加载（下一回合生效；唯独本地 TTS 切换需重启）
改组件 config 后  →  scripts/stop_web.sh && scripts/start_web.sh
```

### 新增一个云端 TTS 服务（零核心代码改动）

1. 复制 `backend/components/tts_gptsovits/` 为 `tts_<新id>/`，改三个文件：
   `manifest.json`（id/entry/name）、`config.json`（新服务的 base_url 与参数）、
   `provider.py`（继承 `TtsProvider`，实现 `health()`/`synthesize()`，流式再覆盖 `stream()`）；
2. 重启后端，注册表自动发现（无需改 server.py / 前端 / 调度核心）；
3. 想让某句/全部远程台词用新服务：把 `providers.tts_remote` 改成新 id，或在该句加 `"provider"`。
