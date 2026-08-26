# 音色特征提取与 `.spk.bin` 转换

这个目录提供把 WAV、MP3 或 FLAC 参考音频转换成当前 SpaceMIT K3 Qwen3-TTS runtime 使用的音色文件的脚本。

## 输出格式

当前 K3 runtime 的 `speaker_file` 不是音频文件，而是：

```text
raw little-endian float32[1024]
4096 bytes
```

因此不能把 WAV/MP3 直接改名为 `.spk.bin`。必须使用 Qwen3-TTS **Base** 模型的 speaker encoder 从参考音频提取 1024 维向量。
脚本直接读取 WAV/FLAC；遇到 MP3 会回退到 `librosa/audioread`，因此 MP3 输入需要系统安装 `ffmpeg`。

脚本使用 `x_vector_only_mode=True`，只保存 speaker embedding；不会把参考音频的 codec token 或文字内容保存进 `.spk.bin`。这正好对应当前 K3 服务的固定 speaker 文件接口。

## 环境安装（推荐使用 PC/服务器）

不建议在 K3 riscv64 板上安装完整 PyTorch/Qwen Python 推理栈。可以在 x86_64 Linux、GPU 服务器或其他支持 Qwen3-TTS 的机器上执行：

```bash
cd /path/to/qwen3-tts
python3 -m venv .venv-qwen3-voice
. .venv-qwen3-voice/bin/activate
python -m pip install -U pip
python -m pip install qwen-tts soundfile librosa safetensors numpy
# MP3 输入还需要系统 ffmpeg；WAV/FLAC 通常只需要 soundfile。
sudo apt install -y ffmpeg
```

CPU-only PyTorch 环境可以按 PyTorch 官方 CPU wheel 方式先安装 CPU 版本，再安装 `qwen-tts`。GPU 环境可使用官方推荐的 CUDA/PyTorch 组合。

## 提取用户自己的声音

```bash
python voice_embeddings/extract_speaker_embedding.py \
  --input '/path/to/my_voice.wav' \
  --output voice_presets/embeddings/my_voice.spk.bin \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-text '参考音频对应的完整文字' \
  --overwrite
```

推荐使用本地模型目录。脚本会优先读取本地目录中的 `speaker_encoder.safetensors`；如果没有这个小文件，则从完整的 `model.safetensors` 中读取 `speaker_encoder.*` 张量，不需要加载 talker/codec。使用模型 ID 时会回退到 Qwen 官方高层 API，可能额外下载并加载 `speech_tokenizer`。

如果网络受限，先把模型下载到本地，再将 `--model` 改为本地目录。例如：

```bash
modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --local_dir /data/models/Qwen3-TTS-12Hz-0.6B-Base

python voice_embeddings/extract_speaker_embedding.py \
  --model /data/models/Qwen3-TTS-12Hz-0.6B-Base \
  --input my_voice.wav \
  --output my_voice.spk.bin \
  --overwrite
```

`--ref-text` 会写入旁边的 JSON 元数据，便于追溯；本目录当前使用的是纯 x-vector 模式，提取 1024 维音色向量时不依赖文字稿。若后续实现完整的 Base ICL 克隆接口，文字稿才会参与参考内容建模。

## 用户本次提供的安可音色

本项目已用以下音频作为输入：

```text
用户提供的本地 `1安可.wav`（原始路径不写入仓库）
```

对应文字：

```text
妈妈说过，只要相信，故事就会给人力量。所以每次想到这些故事的时候，安卡都觉得心里暖融融的。
```

注意：项目只提交生成后的 `.spk.bin` 和元数据，不提交原始录音；原始音频的版权、角色声音授权和使用范围需要由使用者自行确认。

## 检查输出

```bash
stat -c '%n: %s bytes' voice_presets/embeddings/my_voice.spk.bin
python - <<'PY'
import numpy as np
p = 'voice_presets/embeddings/my_voice.spk.bin'
x = np.fromfile(p, dtype='<f4')
print(x.shape, np.isfinite(x).all(), float(np.linalg.norm(x)))
PY
```

应当看到：

```text
(1024,) True ...
```

可以额外检查文件类型和哈希：

```bash
file voice_presets/embeddings/my_voice.spk.bin
sha256sum voice_presets/embeddings/my_voice.spk.bin
```

## 部署到 K3

把音色文件复制到板端模型目录，先备份原始音色：

```bash
scp voice_presets/embeddings/my_voice.spk.bin \
  spacemit-k3:/home/spacemit/projects/qwen3-tts/qwen3-tts-0.6b/

ssh spacemit-k3 'cd /home/spacemit/projects/qwen3-tts && \
  cp qwen3-tts-0.6b/default.spk.bin qwen3-tts-0.6b/default.spk.bin.backup && \
  sed -i "s/\\\"speaker_file\\\": \\"default.spk.bin\\\"/\\\"speaker_file\\\": \\"my_voice.spk.bin\\\"/" qwen3-tts-0.6b/config.json && \
  ./stop_server.sh && ./start_server.sh'
```

也可以只改 `config.json` 的 `speaker_file`，不改客户端请求中的 `voice: "default"`。当前服务启动时加载 speaker 文件，因此替换后必须重启。

## 参考音频建议

- 单人、安静环境、无背景音乐；
- 5～15 秒通常足够，尽量不要包含其他人的声音；
- 原始文字稿要保存好，尤其是将来要使用 Base 模型的 ICL 模式时；
- 不要未经授权克隆真实人物、配音演员或其他受保护声音。
