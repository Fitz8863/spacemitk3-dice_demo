# MOSS-TTS-Nano SpaceMIT EP Demo 1.0.7 Slim

K3 `riscv64` 精简交付包，支持固定采样、非流式语音合成、内置音色和自定义
WAV 声音克隆。SpaceMIT EP 使用 A100 CPU8-11，Python 和 CPU fallback 使用
X100 CPU0-7。

## 系统要求

- Python 3.14
- SpaceMIT `onnxruntime 1.24.2+spacemit.a1`
- NumPy 2.3.5、系统 `libsndfile`
- `spacemit-tcm-smi`、`flock`、`timeout`

当前默认分支使用动态 KV decode 图；固定 KV 的 320/512/1024 行优化图保存在本地 Git 分支中，
详见 [MODEL_VARIANTS.md](MODEL_VARIANTS.md)。

包内已包含 SciPy、`spacemit_ort`、`sentencepiece`、`soundfile`、EP 动态库、
头文件和模型。当前板端包使用动态 KV decode 图，最大位置容量按官方模型配置为
32768；不能替换为普通 PyPI ONNX Runtime。

```bash
sha256sum -c SHA256SUMS
```

## 一次性合成

```bash
./run_demo.sh \
  --text "你好，这是 MOSS TTS Nano 在 K3 上的演示。" \
  --output outputs/demo.wav
```

## 内置温柔女声克隆

```bash
./run_voice_clone.sh \
  --text "你好，这是温柔女声的声音克隆演示。" \
  --output outputs/warm-female.wav
```

## 预热后输入文本

内置音色：

```bash
./run_interactive.sh
```

温柔女声：

```bash
./run_interactive.sh \
  --reference-audio assets/warm-female-reference.wav \
  --output-dir outputs/interactive-warm-female
```

等待以下提示后再输入文本：

```text
runtime ready; enter text below (:quit to exit)
text>
```

交互模式默认使用 `aplay` 将每个已完成文本 chunk 的 PCM16 音频直接播放，不保存最终 WAV。
播放格式为 48 kHz、双声道、`S16_LE`；可用 `--audio-device hw:2,0` 指定 ALSA 设备，或用
`--no-pcm-playback` 关闭播放。需要保存副本时再加 `--save-wav`。这是 chunk 级准流式播放，
不是逐音频帧流式；当前 codec 仍然需要一个文本 chunk 的全部音频帧生成完后才能解码。

交互模式默认把首个文本 chunk 限制为 8 个文本 token，以更早开始播放；后续 chunk 仍使用
`--voice-clone-max-text-tokens`（默认 24）。首段被拆成多个连续子段时，子段之间不会插入额外
chunk 静音，避免连续语句被人为打断。可用 `--first-chunk-text-tokens 12` 调整首段大小，或
设为 `0` 关闭首段优化。

输入 `:quit`、`:q` 或 Ctrl-D 退出。默认不保存输出文件；加上 `--save-wav` 后才会依次保存为
`request-0001.wav`、`request-0002.wav`。

## 自定义参考音频

```bash
./run_interactive.sh \
  --reference-audio /path/to/reference.wav \
  --output-dir outputs/customer-voice
```

建议使用 48 kHz、单声道、PCM16 WAV：

```bash
ffmpeg -i reference.mp3 -ar 48000 -ac 1 -c:a pcm_s16le reference.wav
```

交互模式在显示 `text>` 前完成会话创建、参考音频编码和完整预热。每条输出的
`warm RTF` 不包含这些启动开销；同一进程内会复用 runtime 和 prompt codes。
一次性合成也会复用容量检查阶段生成的 prompt codes，并单独打印不计入 RTF 的
`voice preparation` 时间。

## 功能边界

- 仅支持 `fixed` 采样；交互模式默认按文本 chunk 播放 PCM，可选保存完整 WAV。
- 保留自定义参考音频编码器和完整音频解码器。
- 不包含 local cached/local decoder 回退模型和逐帧流式 codec step 模型。
- 长文本按完整句子和成对引号进行语义切分；固定 KV 容量不足时才按分句切开。
- 每段默认最多生成 120 帧；decode 图支持动态 KV，官方模型最大位置容量为 32768。
- 单段达到 120 帧仍未自然结束时会有界重切该段；重试仍失败则返回错误。
- 只有全文成功后才原子写入最终 WAV，不会留下本次生成的截断文件。
- 启动前检查竞争 AI 进程并清理 TCM；退出后确认 TCM 已释放。

详细诊断使用 `--verbose`；完整 JSON 使用 `--report-json <path>`。
