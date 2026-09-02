# zipformer-streaming —— K3 板真流式 ASR（边说边出字）

基于 [archive.spacemit.com](https://archive.spacemit.com/spacemit-ai/model_zoo/asr/) 的
`zipformer-streaming.tar.gz` 模型包，在 SpacemiT K3（RISC-V）上实现**真流式**语音识别：
音频以 320ms 为粒度逐块送入 encoder，边说边出字，无需等说完。

**特性一览**

- 真流式：encoder 状态跨 chunk 传递，每 320ms 出一次增量结果
- 中英混合识别（sherpa-onnx 双语流式 zipformer transducer）
- SpaceMIT EP 加速，RTF 0.27~0.35，远快于实时
- 麦克风实时字幕 + 能量 VAD 自动断句（说完一句停顿即出整句）
- wav 文件 / stdin 裸流两种输入，8kHz 自动重采样
- 引擎与 CLI 分离，可直接嵌入自己的程序

## 目录

- [它是什么](#它是什么)
- [实测性能](#实测性能)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [麦克风实时识别](#麦克风实时识别)
- [断句（VAD）参数](#断句vad参数)
- [命令行参考](#命令行参考)
- [延迟构成](#延迟构成)
- [工作原理](#工作原理)
- [二次开发（引擎 API）](#二次开发引擎-api)
- [目录结构](#目录结构)
- [性能与绑核](#性能与绑核)
- [已知限制](#已知限制)
- [常见问题](#常见问题)
- [模型与致谢](#模型与致谢)

## 它是什么

模型包内容是 **sherpa-onnx 流式中英双语 Zipformer Transducer**
（`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`），
encoder / decoder / joiner 三件套 + tokens + 测试音频。其中
`encoder-*.q.onnx` 是进迭时空用 xslim 重新导出的 SpaceMIT EP 量化版
（包内 `zipformer.config` 就是官方推荐的 EP 线程配置）。

> 与 `model-zoo-asr`（ai-sdk）里 zipformer 后端的区别：ai-sdk 是"假流式"
> ——feedAudio 只缓存、Flush 时整段离线解码，无逐字中间结果。本项目自写
> 流式 transducer 解码器，encoder 的 35 个状态张量（cached_len/key/val/conv…）
> 跨 chunk 传递，真正边说边出字。

## 实测性能

K3 板（X100 性能核），SpaceMIT EP + q.onnx：

| 配置 | RTF（越小越快） | 说明 |
|------|------|------|
| q.onnx + SpaceMIT EP | **0.27 ~ 0.35** | 默认，推荐 |
| int8.onnx + CPU 4 线程 | 0.37 | 无 EP 时的选择 |
| int8.onnx + CPU 2 线程 | 0.43 | |

RTF < 1 即快于实时：1 秒音频的推理耗时不到 0.35 秒，麦克风长会话不会积压
（实测 12 秒连续输入，chunk 处理节奏与实时完全同步）。

4 个官方测试音频全部正确识别，其中 `0.wav` 断句后拼接与标准转写一字不差
（见下方[验收基准](#快速开始)）。

## 环境要求

- SpacemiT K3 板，Bianbu 系统（ssh `spacemit@spacemit-k3`）
- `spacemit-onnxruntime 2.0.6`（含 SpaceMIT EP，库在 `/usr/local/lib`）
- `libsndfile`、`arecord`（Bianbu 自带）
- **kaldi-native-fbank 预编译产物**：复用 `../model-zoo-asr/build/lib/`
  下的 `libkaldi-native-fbank-core.a`、`libkissfft-float.a`（编译过
  model-zoo-asr 就有；没有的话先编它）

## 快速开始

```bash
cd ~/projects/asr/zipformer-streaming

# 1. 编译（约 1 分钟）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8

# 2. 识别测试音频（q.onnx + EP）
./build/stream_asr --wav sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav
```

运行效果（`--vad` 开断句时的真实输出，字幕行为 `\r` 单行刷新，此处展开示意）：

```
音频: .../test_wavs/0.wav  10s  (VAD 断句开)
[1.3s] 昨天
[2.6s] 昨天是 MON          ← 单行字幕：滚动显示最近约 44 个字符
[2.8s] 昨天是 MONDAY       ← 停顿 0.6s，整句单独成行，下一句从头开始
...
---- 分句结果 (3 句) ----
1. 昨天是 MONDAY
2. TODAY IS礼拜二
3. THE DAY AFTER TOMORROW是星期三

[统计] 音频=10.05s 推理=3.10s RTF=0.308 chunks=32 tokens=11
```

**验收基准**：4 个官方测试音频的参考转写（识别结果应与此基本一致）：

| 文件 | 参考转写 |
|------|----------|
| `0.wav` | 昨天是 MONDAY TODAY IS礼拜二 THE DAY AFTER TOMORROW是星期三 |
| `1.wav` | 这是第一种第二种叫呃与 ALWAYS ALWAYS什么意思 |
| `2.wav` | 这个是频繁的啊不认识记下来 FREQUENTLY频繁的 |
| `3.wav` | 第一句是个什么时态加了 ES是一般现在时对吧后面还时态写上 |

## 麦克风实时识别

```bash
# 方式一：一键脚本（跟随系统默认输入设备，即桌面设置里选的麦克风）
./run_mic.sh

# 指定直连硬件（跳过 PipeWire，延迟更低）
./run_mic.sh c920      # C920 USB 摄像头麦克风
./run_mic.sh es8326    # 板载 ES8326 音频口

# 方式二：手动管道（default = 系统默认输入，走 PipeWire）
arecord -D default -f S16_LE -r 16000 -c 1 -t raw \
    | ./build/stream_asr --pcm
```

显示效果：底部单行字幕只滚动显示最近约 44 个字符（超长文本不会刷屏）；
说完一句停顿约 0.6 秒，整句单独成行打印并重新开始下一句。Ctrl-C 结束后
打印分句结果和统计。

查看录音设备：`arecord -l`；`default` 由 PipeWire 管理，跟随桌面设置。

## 断句（VAD）参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--vad` / `--no-vad` | pcm 开 / wav 关 | 强制开关 |
| `--vad-rms N` | 400 | 静音 RMS 门限（s16 量级）。环境噪声大调高，声音小调低 |
| `--vad-pause-ms N` | 600 | 停顿多久断句 |
| `--vad-max-ms N` | 8000 | 单句最长时长，到点强制断句（防长语音无停顿） |

断句只重置 token 假设，encoder 声学状态保持连续，句首词不失准；
断句前会先解码完缓冲尾帧，不丢字。

## 命令行参考

```
--model-dir DIR    模型目录（默认 ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20）
--wav FILE         识别 wav 文件（边识别边显示单行字幕，结束出完整结果）
--realtime         wav 按实时节奏喂入，模拟边说边出字
--pcm              从 stdin 读 s16le/16k/mono 裸流（麦克风管道）
--cpu              不用 SpaceMIT EP，encoder 纯 CPU（配合 int8.onnx）
--encoder FILE     指定 encoder onnx（默认 q.onnx；CPU 模式建议 int8.onnx）
--ep-disable-conv  SpaceMIT EP 禁用 Conv 算子（遇到 Conv 报错时加）
--threads N        CPU 会话线程数（默认 2）
--vad / --no-vad   静音断句开关（--pcm 默认开，--wav 默认关），详见上表
--vad-rms N        静音 RMS 门限（默认 400）
--vad-pause-ms N   停顿断句时长（默认 600）
--vad-max-ms N     单句最长时长（默认 8000）
-v                 调试输出
```

常用组合：

```bash
M=sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

# 纯 CPU 模式（int8 模型；q.onnx 在 CPU 下会因 DynamicQuantizeMatMul 报错）
./build/stream_asr --wav $M/test_wavs/0.wav --cpu --threads 4 \
    --encoder $M/encoder-epoch-99-avg-1.int8.onnx

# wav 文件也开断句（输出分句列表）
./build/stream_asr --wav $M/test_wavs/0.wav --vad

# 管道喂任意音频（ffmpeg 转 PCM，mp3/m4a 均可）
ffmpeg -loglevel error -i 任意音频.mp3 -f s16le -ar 16000 -ac 1 - \
    | ./build/stream_asr --pcm

# 环境嘈杂、总是误出字 → 调高静音门限
./run_mic.sh -- --vad-rms 800
```

## 延迟构成

说完一个词到它出现在字幕上的总延迟约 **0.4 ~ 0.5 秒**：

| 环节 | 耗时 |
|------|------|
| chunk 粒度（攒 32 帧 fbank） | 320 ms |
| 单 chunk 推理（EP） | ~73 ms |
| arecord/PipeWire 缓冲 | 数十 ms |

断句模式下整句行在停顿 0.6 秒后出现（这是配置的停顿阈值，不是处理慢）。

## 工作原理

```
麦克风/文件 → knf OnlineFbank (80 维 log-mel, 10ms 帧)
  → 每攒 32 帧(320ms) 组一个 x[1,39,80]（含 7 帧左上下文，首块补零）
  → encoder.q.onnx (SpaceMIT EP): 输出 encoder_out[1,8,512] + 35 个新状态
      （状态下一块回传，实现因果流式，不回看历史音频）
  → 对 8 个输出帧逐帧贪心 transducer 解码:
      decoder.onnx: 最近 2 个 token → decoder_out[1,512]（有缓存）
      joiner.onnx : encoder_out ⊕ decoder_out → logits[1,6254]
      argmax 非 blank 即发射 token，滑动 token 上下文继续
  → tokens.txt 映射 + ▁→空格 → 每 320ms 输出一次增量文本
```

- 模型 chunk 参数（T=39、decode_chunk_len=32）从 onnx metadata 自动读取，
  换模型不用改代码
- blank id 取 tokens.txt 中的 `<blk>`；状态张量动态维初始化为 1、全零起步
- VAD 断句 = 能量 RMS 检测停顿 → `FlushPartial()` 解码尾帧 → 取整句文本 →
  `ResetHypothesis()` 只清 token 假设（声学状态连续，句首不失准）

## 二次开发（引擎 API）

引擎与 CLI 完全解耦（`src/zipformer_streaming.h`），可直接嵌入语音助手等
下游程序：

```cpp
#include "zipformer_streaming.h"

zstream::Options opts;
opts.model_dir = ".../sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20";
// 默认 SpaceMIT EP；纯 CPU: opts.use_spacemit_ep = false;（需换 int8.onnx）
zstream::StreamingASR asr(opts);            // 构造即加载模型

// 循环喂入 16kHz 单声道 float 样本（任意长度，麦克风/文件皆可）
asr.AcceptWaveform(buf, n);

// 每次喂完调用：攒够 320ms 推进一次推理，返回当前累积文本
bool has_new = false;
std::string text = asr.PollPartial(&has_new);

// 断句场景（说完一句）：
asr.FlushPartial();              // 先解码缓冲尾帧（否则最多丢 320ms）
std::string sentence = asr.CurrentText();
asr.ResetHypothesis();           // 只清 token 假设，声学状态连续

// 输入结束：解码剩余尾帧，返回最终文本
std::string final_text = asr.InputFinished();

// 统计：audio_seconds() / infer_seconds() / chunk_count() / emitted_tokens()
```

编译时链接 `src/zipformer_streaming.cc` + onnxruntime + knf 静态库
（参考 `CMakeLists.txt` 中 `stream_asr` 目标的写法）。

## 目录结构

```
zipformer-streaming/
├── CMakeLists.txt
├── README.md
├── run_mic.sh                     # 麦克风一键启动（系统默认输入）
├── src/
│   ├── zipformer_streaming.{h,cc} # 流式引擎（fbank + 三模型 + transducer 解码 + VAD 支撑接口）
│   └── stream_asr_main.cc         # CLI（wav / realtime / pcm 三种模式 + VAD + 字幕显示）
├── tools/
│   ├── dump_onnx.cc               # 查看 onnx 输入输出/元数据
│   └── test_enc.cc                # encoder 最小化/多 chunk 压测工具（排障用）
├── build/                         # 编译产物（stream_asr, dump_onnx, test_enc）
└── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/   # 模型
    ├── encoder-epoch-99-avg-1.q.onnx      # EP 量化版（默认）
    ├── encoder-epoch-99-avg-1.int8.onnx   # CPU int8 版
    ├── decoder-epoch-99-avg-1.onnx
    ├── joiner-epoch-99-avg-1.onnx
    ├── tokens.txt
    └── test_wavs/                 # 官方测试音频
```

## 性能与绑核

EP 模式已按官方推荐配置（`zipformer.config` 的
`SPACEMIT_EP_PERFER_CORE_ARCH=0x5064`）运行，实测全部线程落在 **X100 性能核
（cpu0-7）**，无需额外操作。

自查命令：

```bash
ps -T -o spid,psr,comm -p $(pgrep -x stream_asr)
grep -H Cpus_allowed_list /proc/$(pgrep -x stream_asr)/task/*/status
```

如需把 ASR 与其他负载隔离（如同时跑 TTS/LLM），可用 taskset 划分核组：

```bash
taskset -c 0,3 ./run_mic.sh     # ASR 只用 X100 的 0、3 核
```

如需让出性能核、改用 A100 能效核跑 ASR（RTF 会变差但省性能核），需修改
`zipformer_streaming.cc` 中 `MakeEpSession` 的 EP 参数：去掉
`SPACEMIT_EP_PERFER_CORE_ARCH`，改传 `SPACEMIT_EP_INTRA_THREAD_AFFINITY="8;10"`。

## 已知限制

- **贪心解码**：无 beam search，长难句准确率略低于离线模型（SenseVoice）
- **无标点、无大小写、无 ITN**：输出即模型 BPE token 拼接（如 "MONDAY"）
- **8 秒强制断句**：连续说话不停顿时会在 8 秒处硬切，可能切在词中间
  （文本不丢，下一句接着出）
- **无说话人分离、无时间戳**：整句一个时间点
- **纯静音零输出**：安静环境下不出垃圾字；但旁边设备播放人声会被如实转写

## 常见问题

**Q: 长时间挂麦克风，输出一直刷屏/像不停按回车？**
旧版显示逻辑按 `\r` 重印全部累积文本，文本超过终端一行宽度后就会变成刷屏。
现已改为只滚动显示最近约 44 个字符的字幕窗，且 VAD 断句（默认开）让每句
独立成行、假设定期重置，长会话不会无限累积。

**Q: 没说话也一直出奇怪的文本？**
那是麦克风真的拾到了声音（比如旁边设备在放视频/直播），流式模型会如实转写；
持续纯噪声下 transducer 也可能复读机式幻觉。VAD 断句能把影响限制在单句内。
环境底噪大就把 `--vad-rms` 调高（如 800）。

**Q: 报 `MatmulInteger: b zero point is not valid`？**
q.onnx 用了 SpaceMIT 的 DynamicQuantizeMatMul 贡献算子，只能走 EP，不能
`--cpu`。CPU 模式请换 `--encoder .../encoder-epoch-99-avg-1.int8.onnx`。

**Q: EP 下遇到 Conv 算子报错？**
加 `--ep-disable-conv`（等价于 `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=Conv`）。
当前版本的 q.onnx 不需要，model-zoo-asr 的旧 zipformer 需要。

**Q: 编译报找不到 kaldi-native-fbank？**
CMake 从 `../model-zoo-asr/build/lib` 找静态库。先编译 model-zoo-asr，
或手动传 `-DKNF_LIB=... -DKISSFFT_LIB=... -DKNF_INC_DIR=... -DKNF_INC_ROOT=...`。

**Q: 重新下载/解压模型？**
```bash
wget https://archive.spacemit.com/spacemit-ai/model_zoo/asr/zipformer-streaming.tar.gz
tar -xzf zipformer-streaming.tar.gz
# md5 校验：cat zipformer-streaming.tar.gz.md5（2d83f932a89a585aa1889a1191c5b241）
```

## 模型与致谢

- 模型包：进迭时空（SpacemiT）model zoo，
  [zipformer-streaming.tar.gz](https://archive.spacemit.com/spacemit-ai/model_zoo/asr/zipformer-streaming.tar.gz)，
  Apache-2.0
- 原始模型：[pfluo/k2fsa-zipformer-chinese-english-mixed](https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed)
  （sherpa-onnx 格式导出）
- 训练代码：[k2-fsa/icefall](https://github.com/k2-fsa/icefall)
  pruned_transducer_stateless7_streaming
- 特征提取：[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
- 推理：onnxruntime（spacemit-onnxruntime 2.0.6，含 SpaceMIT EP）
