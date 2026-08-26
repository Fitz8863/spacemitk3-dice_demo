# Codex Project Context: Qwen3-TTS on SpaceMIT K3

此文件供新的 Codex 会话快速接手。开始工作时先读本文档，再用实时进程、文件和日志验证状态。

## 当前用户需求（2026-08-26 更新）

当前交互程序的行为是：

```text
输入文字 -> 回车 -> 分段合成 -> 第一段完整返回后立即启动 aplay -> 后续分段边生成边播放
```

默认交互路径只播放，不保存 WAV。`--no-play` 仅用于不启动播放器的合成测速，也不保存 WAV。

当前实现是“完整分段级流式”，不是服务端逐 PCM 帧真流式：服务端仍要先完成一段 WAV，客户端解包为 PCM 后，通过一个长生命周期的 `aplay` raw PCM 进程播放。低延迟模式不等待覆盖整个 RTF 亏空，RTF 大于 1 时可能出现分段之间短暂停顿。


## 环境

```text
SSH 别名:       spacemit-k3
用户:           spacemit
主机:           spacemit-spacemitk3picoitx
架构:           riscv64
项目:           /home/spacemit/projects/dice-game/main/tts/qwen3-tts
服务:           http://127.0.0.1:18080
```

接手检查：

```bash
ssh spacemit-k3
cd /home/spacemit/projects/dice-game/main/tts/qwen3-tts
pwd
cat AGENTS.md
curl -fsS http://127.0.0.1:18080/health || true
pid="$(cat llama-server.pid 2>/dev/null || true)"
[[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" && ps -fp "$pid" || true
tail -n 30 llama-server.log
```

PID 会变化，不能把文档中的历史 PID 当成当前 PID。

## 关键文件

```text
/home/spacemit/projects/dice-game/main/tts/qwen3-tts/
├── AGENTS.md
├── README.md
├── start_server.sh
├── stop_server.sh
├── run_interactive.sh
├── qwen3_tts_interactive.py
├── qwen3-tts-0.6b/
├── runtime/bin/llama-server
├── .gitignore
└── wav-output/output.wav  # 运行时生成，不提交
```

不要破坏其他项目：

```text
/home/spacemit/projects/tts
/home/spacemit/projects/sm-sdk
```

## 使用方法

交互生成：

```bash
cd /home/spacemit/projects/dice-game/main/tts/qwen3-tts
./run_interactive.sh
```

单次生成：

```bash
./run_interactive.sh '你好，这是 Qwen3 TTS。'
```

无论从哪个目录启动，输出都固定在项目目录：

```bash
cd /tmp
/home/spacemit/projects/dice-game/main/tts/qwen3-tts/run_interactive.sh '仍保存到项目的 wav-output/output.wav。'
```

调整分段长度：

```bash
QWEN3_TTS_CHUNK_CHARS=24 ./run_interactive.sh
```

退出交互：输入 `quit`、`exit` 或 `退出`。

## 播放实现

代码位于 `qwen3_tts_interactive.py`：

- `split_text()`：按自然标点或长度分段；英文避免截断单词；缺少结束标点时自动补齐，以降低 `frame limit without EOS` 风险。
- `synthesize()`：请求 `POST /v1/audio/speech`，每段返回完整 WAV 字节。
- `decode_chunk()`：解析 WAV，只提取并校验 PCM frame。
- `StreamingPlayer`：启动一个长生命周期 `aplay` raw PCM 进程，后台线程按分段写入；队列有界，避免积压导致播放延迟不断增加。
- `stream_speech()`：首段完整 WAV 返回后立即播放，后续分段合成完成即继续送入播放器。

默认播放参数：

```text
24 kHz / 16-bit / mono / PCM
QWEN3_TTS_PLAYBACK_BUFFER_US=250000
QWEN3_TTS_PLAYBACK_PERIOD_US=30000
QWEN3_TTS_PLAYBACK_START_DELAY_US=0
QWEN3_TTS_PLAYBACK_QUEUE_SEGMENTS=2
```

当前默认不创建或覆盖 `wav-output/output.wav`，也不创建 `.part` 文件。


## 模型与 Runtime

模型目录：

```text
/home/spacemit/projects/dice-game/main/tts/qwen3-tts/qwen3-tts-0.6b
```

主要文件：

```text
text-embed-proj.onnx
codec-decoder-t50-q.onnx
talker-q8_0.gguf
code-predictor-q4_0.gguf
qwen3-tts-aux.gguf
tokenizer.gguf
default.spk.bin
config.json
```

默认优化 Runtime：

```text
/home/spacemit/projects/dice-game/main/tts/qwen3-tts/runtime/bin/llama-server
version: 3 (787e5fc)
```

`runtime/bin/` 只保留当前 Qwen3-TTS 服务实际加载的优化版 `llama-server` 及其 `libggml*`、`libllama*`、`libmtmd` 共享库。

SpaceMIT ORT 默认使用板端系统安装的 `spacemit-onnxruntime`：

```text
/usr/lib/libonnxruntime.so.1
/usr/lib/libspacemit_ep.so
```

如需使用其他 ORT，可设置 `QWEN3_TTS_ORT_LIB_DIR=/path/to/lib`；旧的项目内 ORT 目录已不再作为默认依赖。

确认实际加载：

```bash
pid="$(cat /home/spacemit/projects/dice-game/main/tts/qwen3-tts/llama-server.pid)"
grep -E 'libonnxruntime|libspacemit_ep' "/proc/$pid/maps" | awk '{print $6}' | sort -u
```

模型配置关键值：

```text
sample_rate:                    24000
frontend_threads:               2
codec_threads:                  4
talker_threads:                 4
SPACEMIT_EP_INTRA_THREAD_NUM:    4
SPACEMIT_EP_INTER_THREAD_NUM:    1
```


## 当前实时优化约束

- TTS 的 SpaceMIT preferred core 默认配置为 `8,9,10,11,12,13` 六个核。
- YOLO 继续使用 EP affinity `14;15`，不要修改或混用。
- realtime llama.cpp patch 的 `QWEN3_TTS_GEMV_CPU_MASK=4-7` 只绑定普通 CPU 侧 GEMV worker，不代表 AI Core。
- `QWEN3_TTS_THREADPOOL_POLL=50` 是当前 YOLO 并行运行时的实测默认值；需要使用实时 patch 构建的 runtime 才会生效。
- 当前板端实际 `qwen3-tts-0.6b/config.json` 使用 `frontend_threads=2`、`codec_threads=4`、`talker_threads=4`、`SPACEMIT_EP_INTRA_THREAD_NUM=4`。如果板端配置被用户修改，以实际文件为准。

## 音色能力边界

当前加载：

```text
qwen3-tts-0.6b/anke.spk.bin
```

Runtime 读取的 speaker 文件格式是 raw `float32[1024]`，即 4096 字节。模型底座属于支持 voice cloning 的 Qwen3-TTS Base 路线，但当前 K3 HTTP 服务没有实现直接上传 `ref_audio`/`ref_text` 并现场提取音色。不要把“底层模型支持克隆”错误表述为“当前接口已能直接上传录音克隆”。

## 流式能力边界

当前不是逐 PCM 真流式。服务端行为是：

```text
每段文本 -> 完整生成该段 WAV -> 返回完整 HTTP 响应
```

交互程序实现的是低延迟分段级流式播放：第一段完整返回后立即启动 `aplay`，主线程继续生成后续分段并将 PCM 送入同一播放器。由于当前板端实测短句 RTF 大约在 `1.12-1.45`，低延迟模式优先首播速度，不能保证长文本全程无缝；若需要更少停顿，可设 `QWEN3_TTS_STREAM_LOW_LATENCY=0` 使用保守 RTF 预缓冲模式。


## CPU、EP 线程和 AI Core

- `taskset` 只控制 Linux CPU 亲和性。
- `SPACEMIT_EP_INTRA_THREAD_NUM` 是 EP 线程配置。
- 这两者都不能直接证明使用了几个 A100/AI Core。

若要声明 AI Core 使用数量，需要设备级调度日志、profile 或计数器。`/proc/<pid>/task/*/status` 只能证明 CPU 亲和性。

## 服务管理

```bash
cd /home/spacemit/projects/dice-game/main/tts/qwen3-tts
./start_server.sh
curl -fsS http://127.0.0.1:18080/health
./stop_server.sh
tail -n 100 llama-server.log
```

默认只绑定 `127.0.0.1`。远程访问优先用 SSH 隧道，不要无必要暴露到局域网。

## 修改和验证规则

1. 先确认绝对路径和当前工作目录。
2. 修改前读实际代码、配置、服务日志和进程 maps。
3. 使用短文本和 `timeout` 做有界测试。
4. 只清理本项目准确 PID，不使用宽泛 `killall`。
5. 修改交互代码后至少验证：
   - `python3 -m py_compile qwen3_tts_interactive.py`
   - `/health` 正常
   - 单段中文直接播放成功
   - 多段文本生成期间播放器已启动并继续消费后续分段
   - 默认运行不修改 `wav-output/output.wav`
   - 默认运行不产生 `.output.wav.part`
   - 播放格式为 24 kHz、16-bit、mono、PCM
   - 播放结束后无残留 `aplay`
   - `--no-play` 模式能只合成并打印 RTF
6. 不要把“ORT 能运行”“加载 SpaceMIT EP”“全图使用 AI Core”混为一谈。
7. 向用户报告真实生成路径、文件大小、音频时长和耗时。

## 快速回归

短文本直接播放：

```bash
ssh spacemit-k3 "cd /home/spacemit/projects/dice-game/main/tts/qwen3-tts && timeout 40s ./run_interactive.sh '你好，这是低延迟播放测试。'"
```

多段流式播放：

```bash
ssh spacemit-k3 "cd /home/spacemit/projects/dice-game/main/tts/qwen3-tts && timeout 90s ./run_interactive.sh '这是一段较长的文本，用来确认第一段播放后，后续分段仍然可以继续生成并立即播放。'"
```

不播放测速：

```bash
ssh spacemit-k3 "cd /home/spacemit/projects/dice-game/main/tts/qwen3-tts && timeout 40s ./run_interactive.sh --no-play '低延迟模式测速。'"
```

检查播放器和临时文件：

```bash
pgrep -af '[a]play' || true
find /home/spacemit/projects/dice-game/main/tts/qwen3-tts/wav-output -maxdepth 1 -type f -name '.output.wav.part' -print
```
