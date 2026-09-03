# AGENTS.md — asr/zipformer-streaming

## 定位

骰子游戏 ASR 输入通道的板端运行时（真流式 Zipformer transducer，中英混合，
SpaceMIT EP 加速，RTF 0.27~0.35）。由 `backend/components/asr_zipformer` 功能包
spawn 调用；本目录是**正本**，`~/projects/asr/zipformer-streaming` 是移植前的
参考快照，不再维护。完整原理与调参见本目录 README.md。

## 结构

- `src/zipformer_streaming.{h,cc}` — 流式引擎（fbank + encoder/decoder/joiner
  三模型 + transducer 贪心解码 + VAD 支撑接口）。引擎与 CLI 解耦，可直接嵌入。
- `src/stream_asr_main.cc` — CLI：`--wav` / `--realtime` / `--pcm`（stdin 裸流）
  / `--jsonl`（机器可读输出）。
- `third_party/` — kaldi-native-fbank、kissfft，源码随仓库分发，无外部目录依赖。
- `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/` — 模型目录。
  `*.onnx` **不入库**（encoder 超 Git 单文件上限），从
  `https://archive.spacemit.com/spacemit-ai/model_zoo/asr/zipformer-streaming.tar.gz`
  下载（md5 见 README"模型与致谢"节）；tokens.txt / bpe.* / test_wavs 已入库。
- `build/` — 编译产物，不入库。

## 构建（只能在 K3 板上）

riscv64 的 spacemit-onnxruntime / libsndfile 只在板上，开发机不可验证：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

## 验证（改动后必跑）

1. wav 冒烟（4 个官方测试音频的参考转写 = 验收基准，见 README）：
   `./build/stream_asr --wav sherpa-*/test_wavs/0.wav --vad`
2. JSONL 契约（骰子游戏集成依赖，改动必须同步
   `backend/components/asr_zipformer/provider.py` 与其测试）：
   `ffmpeg -i sherpa-*/test_wavs/0.wav -f s16le -ar 16000 -ac 1 - | ./build/stream_asr --pcm --jsonl`

## JSONL 事件契约（stdout，每行一个 JSON 对象；模型日志走 stderr）

```
{"type":"partial","text":...}    增量识别文本（每 320ms，空文本不发）
{"type":"sentence","text":...}   VAD 停顿断句的整句（意图匹配用这个）
{"type":"final","text":...}      stdin EOF 后的尾部整句
{"type":"stats","audio_seconds":..,"infer_seconds":..,"rtf":..,"chunks":..,"tokens":..}
```

`--jsonl` 模式下 stdout **只允许** JSON 行；新增任何输出必须走 stderr。
音频输入格式：stdin s16le / 16kHz / mono 裸流（`arecord -D default -f S16_LE -r
16000 -c 1 -t raw` 直连，设备跟随系统设置）。

## 约束

- 编译与运行结论必须在 K3 板上得出，不得用开发机结果声称可用。
- EP 模式按官方推荐 `SPACEMIT_EP_PERFER_CORE_ARCH=0x5064` 运行，线程落在
  X100 通用核（cpu0-7），与骰子游戏后端进程同簇；需隔离时用 taskset。
- `q.onnx` 只能走 SpaceMIT EP（CPU 模式会因 DynamicQuantizeMatMul 报错，
  CPU 请换 int8.onnx）。
- 已知限制：贪心解码、无标点/大小写/ITN、无说话人分离；纯静音零输出，
  但旁人人声会被如实转写（由上层词表匹配 + 播报闸过滤）。
