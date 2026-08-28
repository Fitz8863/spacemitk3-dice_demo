# Qwen3-TTS runtime

此目录只保存 SpaceMIT K3 上的 Qwen3-TTS 底层 runtime、模型目录和启动/停止脚本。
它不是 Dice Arena 的功能包，也不提供交互式 CLI、音频播放或浏览器入口。

## 在 Dice Arena 中使用

生产入口是 `backend/components/tts_qwen3/`：

```text
backend/components/tts_qwen3/
├── manifest.json      # provider 与生命周期声明
├── config.json        # 组件参数单一来源
├── settings.py        # 配置解析与校验
├── provider.py        # TtsProvider 适配器
├── client.py          # Qwen HTTP client
└── scripts/           # componentctl 使用的内部生命周期 hook
```

从项目根目录管理服务：

```bash
python3 backend/componentctl.py selected tts --game dice
python3 backend/componentctl.py start-selected tts --game dice
python3 backend/componentctl.py health tts_qwen3
python3 backend/componentctl.py stop-selected tts --game dice
```

组件 provider 访问本目录的 `start_server.sh`，runtime 再启动 `llama-server`。
浏览器只调用 backend 的 `/api/tts/stream` 或 `/api/tts/synthesize`，不会直接接触模型文件。

## 资产与路径

默认组件配置位于 `backend/components/tts_qwen3/config.json`：

- `runtime.root` 相对项目根目录；
- `runtime.model_dir` 相对本目录；
- `runtime.base_url` 是 provider 访问的 HTTP 地址；
- `voice.speaker_file` 相对模型目录。

模型权重、ONNX/GGUF 文件、speaker embedding、日志和 PID 文件由 `.gitignore` 排除，
需要在 K3 板端单独准备。修改音色或 runtime 路径后重启组件即可生效。

`start_server.sh` 和 `stop_server.sh` 是底层进程 hook，不应再向其中加入业务参数解析；
组件配置与环境覆盖由 `settings.py` 统一处理。
