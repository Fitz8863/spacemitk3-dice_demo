# CosyVoice TTS 服务调用说明

本文档对应云端当前运行的 `Fun-CosyVoice3-0.5B-2512` 服务。服务由自定义 FastAPI 程序提供，支持 HTTP 流式返回 PCM 音频。

## 1. 服务信息

- 服务地址：`http://<云端IP或主机名>:50000`
- 健康检查：`GET /health`
- 音色列表：`GET /v1/voices`
- 流式接口：`POST /v1/tts/stream`
- OpenAI 风格接口：`POST /v1/audio/speech`
- 音频格式：单声道、24 kHz、16-bit little-endian PCM（`pcm_s16le`）
- 当前已注册音色：`anke`

将文档中的 `<云端IP或主机名>` 替换成实际可以访问服务器的地址。客户端和云端之间需要能够访问 TCP `50000` 端口。

## 2. 先检查服务是否正常

```bash
curl http://<云端IP或主机名>:50000/health
```

正常响应类似：

```json
{
  "status": "ok",
  "model": "/home/hwj/AI/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
  "sample_rate": 24000,
  "audio_format": "pcm_s16le",
  "channels": 1,
  "device": "cuda",
  "voices": ["anke"]
}
```

如果返回连接失败，先确认云端进程和端口；如果返回 `503`，通常表示服务仍在启动、模型尚未加载完成。

## 3. 使用已注册音色进行流式合成（推荐）

使用 `voice_id=anke` 时不需要每次上传参考音频：

```bash
curl -N --fail \
  -X POST "http://<云端IP或主机名>:50000/v1/tts/stream" \
  -F "text=你好，这是一次流式语音合成测试。" \
  -F "voice_id=anke" \
  -F "speed=1.0" \
  -o output.pcm
```

`-N` 会关闭 curl 的输出缓冲。这个命令把返回的裸 PCM 保存为 `output.pcm`，它不是 WAV 文件，播放时必须指定采样率、位深和声道数。

Linux/macOS 可以用 ffplay 播放：

```bash
ffplay -f s16le -ar 24000 -ac 1 output.pcm
```

也可以转换成 WAV：

```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

## 4. Python 流式接收

下面的代码会在收到每个音频块时立即处理，不等待完整响应结束：

```python
import requests

url = "http://<云端IP或主机名>:50000/v1/tts/stream"
data = {
    "text": "你好，这是一次流式语音合成测试。",
    "voice_id": "anke",
    "speed": "1.0",
}

with requests.post(url, data=data, stream=True, timeout=(10, 300)) as response:
    response.raise_for_status()
    sample_rate = response.headers.get("X-Audio-Sample-Rate", "24000")
    print("sample rate:", sample_rate)
    with open("output.pcm", "wb") as audio_file:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                audio_file.write(chunk)
                # 在这里把 chunk 送入播放器，就可以边收边播。
```

`timeout=(10, 300)` 表示连接最多等待 10 秒，连接建立后读取数据最多等待 300 秒。不要使用 `response.content`，因为它会等完整音频生成完才返回。

## 5. Python 边收边播示例

需要安装：

```bash
pip install requests sounddevice numpy
```

示例：

```python
import numpy as np
import requests
import sounddevice as sd

url = "http://<云端IP或主机名>:50000/v1/tts/stream"
data = {
    "text": "你好，音频生成后会逐块播放，而不是等待全文结束。",
    "voice_id": "anke",
    "speed": "1.0",
}

with requests.post(url, data=data, stream=True, timeout=(10, 300)) as response:
    response.raise_for_status()
    with sd.RawOutputStream(
        samplerate=24000,
        channels=1,
        dtype="int16",
        blocksize=0,
    ) as player:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                player.write(chunk)
```

如果播放设备不接受 24 kHz，可以在本地增加重采样，但建议先直接使用服务返回的 24 kHz 格式，减少额外延迟。

## 6. OpenAI 风格接口

服务也提供 `/v1/audio/speech`，字段名称更接近 OpenAI TTS：

```bash
curl -N --fail \
  -X POST "http://<云端IP或主机名>:50000/v1/audio/speech" \
  -F "input=请输出一段简短的测试语音。" \
  -F "voice=anke" \
  -F "speed=1.0" \
  -o output.pcm
```

这里的 `voice` 实际对应服务内部的 `voice_id`。返回格式仍然是裸 PCM，不是 MP3，也不是带 WAV 头的文件。

## 7. 不使用已注册音色，临时上传参考音频

如果不传 `voice_id`，必须同时传 `prompt_text` 和 `prompt_wav`：

```bash
curl -N --fail \
  -X POST "http://<云端IP或主机名>:50000/v1/tts/stream" \
  -F "text=请用参考音色朗读这句话。" \
  -F "prompt_text=希望你以后能够做的比我还好呦。" \
  -F "prompt_wav=@reference.wav" \
  -F "speed=1.0" \
  -o output.pcm
```

这种方式每次请求都要上传和处理参考音频，首包延迟通常高于直接使用 `voice_id`。设备端长期调用时优先使用已注册音色。

## 8. 请求参数

### `/v1/tts/stream`

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `text` | 是 | 要合成的文本，不能为空 |
| `voice_id` | 否 | 已注册音色 ID，例如 `anke`；不填时必须上传参考音频 |
| `prompt_text` | 否 | 参考音频对应的文字 |
| `prompt_wav` | 否 | 参考音频文件；不使用 `voice_id` 时必填 |
| `speed` | 否 | 语速，范围 `0.5` 到 `2.0`，默认 `1.0` |

### `/v1/audio/speech`

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `input` | 是 | 要合成的文本 |
| `voice` | 否 | 等同于 `voice_id` |
| `prompt_text` | 否 | 参考音频对应的文字 |
| `prompt_wav` | 否 | 参考音频文件 |
| `speed` | 否 | 语速，范围 `0.5` 到 `2.0` |

## 9. 查看可用音色

```bash
curl http://<云端IP或主机名>:50000/v1/voices
```

正常响应类似：

```json
{
  "voices": ["anke"],
  "default_prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
}
```

## 10. API Key 鉴权

当前运行进程未配置 `COSYVOICE_API_KEY` 时，不需要鉴权。如果以后在服务端设置了 API Key，所有 `/v1/*` 接口都需要携带：

```bash
curl -H "Authorization: Bearer <API_KEY>" \
  http://<云端IP或主机名>:50000/v1/voices
```

`/health` 不需要鉴权。不要把 API Key 写死在公开代码或日志中。

## 11. 实时性建议

1. 客户端必须使用流式读取，例如 Python 的 `stream=True`，不能一次性读取 `response.content`。
2. 文本较长时，按句号、逗号、问号等切成短句，上一句播放时尽早请求下一句。
3. 优先使用 `voice_id=anke`，避免每次上传参考音频。
4. 播放器收到第一个块后立即启动，不要等到 HTTP 响应结束。
5. `chunk_size` 可以从 `4096` 或 `8192` 字节开始测试；过大可能增加等待，过小会增加调用开销。
6. 当前服务为单 GPU 串行推理，多台设备同时请求时后续请求可能排队。
7. 服务端当前实测首包延迟约为数秒，网络和客户端缓冲还会增加额外延迟。

## 12. 常见错误

- `Connection refused`：服务未启动、端口不通，或云端防火墙未放行 TCP `50000`。
- `503 model is still loading`：模型正在加载，等待 `/health` 返回 `200`。
- `400 text must not be empty`：`text`/`input` 为空。
- `400 prompt_wav is required without voice_id`：没有音色 ID 时缺少参考音频。
- `400 unknown voice_id`：音色 ID 不存在，先调用 `/v1/voices` 查看列表。
- `401 invalid bearer token`：服务端已启用 API Key，但请求没有携带正确的 Bearer Token。
- 播放噪声或速度异常：把返回数据当成 WAV/MP3 播放了；应按 `s16le、24000 Hz、单声道` 播放。

## 13. 最小调用模板

```bash
curl -N --fail \
  -X POST "http://<云端IP或主机名>:50000/v1/tts/stream" \
  -F "text=这里替换成要合成的文字。" \
  -F "voice_id=anke" \
  -o output.pcm

ffplay -f s16le -ar 24000 -ac 1 output.pcm
```
