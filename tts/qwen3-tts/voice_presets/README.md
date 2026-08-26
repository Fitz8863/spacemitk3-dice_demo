# 音色预设与已下载样例

这个目录存放音色向量的元数据，以及用于复现/验证的官方参考音频来源信息。当前仓库只提交 README、manifest 和元数据 JSON；`.spk.bin` 与原始参考音频属于板端资产，不提交 GitHub。

## 当前文件

| 文件 | 用途 | 来源 |
| --- | --- | --- |
| `embeddings/anke.spk.bin` | 板端使用的用户音色向量（不在 Git） | 用户指定的本地 WAV |
| `embeddings/qwen_clone.spk.bin` | 板端可选的官方示例音色（不在 Git） | Qwen3-TTS 官方示例 `clone.wav` |
| `embeddings/qwen_clone_1.spk.bin` | 板端可选的官方示例音色（不在 Git） | Qwen3-TTS 官方示例 `clone_1.wav` |
| `embeddings/qwen_clone_2.spk.bin` | 板端可选的兼容别名（不在 Git） | 当前下载内容与 `clone.wav` 的 SHA-256 相同 |

如果板端已经准备对应文件，每个 `.spk.bin` 都应是：

```text
4096 bytes = little-endian float32[1024]
```

旁边的 `.json` 文件记录了输入音频哈希、模型、采样率和生成结果哈希。

## 官方参考音频下载来源

这些样例来自 Qwen3-TTS 官方仓库示例代码引用的 Qwen 资源地址：

```text
https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav
https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav
https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav
```

项目没有把官方音频作为运行时模型的一部分；音频只用于离线提取和复现，实际部署只需要对应的 `.spk.bin`。当前 `clone_2.wav` 与 `clone.wav` 内容相同，因此两者生成的向量也相同；它保留在 manifest 中作为官方 URL 的可追溯条目，不应当当作第三个独立音色。

## 使用某个预设（仅限已准备对应板端资产）

修改板端 `qwen3-tts-0.6b/config.json`：

```json
{
  "tts_model": {
    "speaker_file": "qwen_clone.spk.bin"
  }
}
```

并将对应的 `.spk.bin` 复制到 `qwen3-tts-0.6b/`，然后重启服务：

```bash
./stop_server.sh
./start_server.sh
```

当前 HTTP 接口仍使用 `voice: "default"` 这个兼容字段；它不会自动在多个文件之间切换。要做真正的请求级多音色选择，需要后续给服务端增加 `voice -> speaker_file` 映射。

## 许可与声音使用

Qwen3-TTS 代码和模型仓库的许可信息请以官方仓库/模型卡为准。模型代码许可证不等于对每一段参考录音或真人声音的单独授权。部署或分发前请分别确认：

1. 模型和 tokenizer 的许可证；
2. 官方样例音频的再分发条件；
3. 用户提供录音的授权范围；
4. 是否会造成对真实人物、配音演员或角色声音的误认。
