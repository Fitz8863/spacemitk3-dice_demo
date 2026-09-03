# MOSS-TTS-Nano runtime

此目录只保存 SpaceMIT EP 的 MOSS-TTS-Nano 底层 runtime、模型和运行库。
它不是用户 CLI，也不提供 Demo、交互输入或 `aplay` 播放入口。

Dice Arena 的生产功能包位于 `backend/components/tts_moss_nano/`：

```text
backend/components/tts_moss_nano/
├── manifest.json      # provider 与生命周期声明
├── config.json        # 组件参数单一来源
├── settings.py        # 配置解析与校验
├── provider.py        # TtsProvider 适配器
├── daemon.py          # 内部 HTTP bridge
├── launcher.py        # start/stop 生命周期 hook
└── scripts/           # componentctl 使用的薄包装
```

管理命令：

```bash
python3 backend/componentctl.py selected tts --game dice
python3 backend/componentctl.py start tts_moss_nano
python3 backend/componentctl.py health tts_moss_nano
python3 backend/componentctl.py stop tts_moss_nano
```

## 配置

`backend/components/tts_moss_nano/config.json` 是运行参数的单一来源：

- `runtime.root` 相对项目根目录；
- `runtime.model_dir` 相对 MOSS runtime 根目录；
- `runtime.base_url` 是 provider 访问的本地 HTTP 地址；
- `voice.mode` 为 `builtin` 或 `clone`；clone 模式需要 `voice.reference_audio`；
- `generation`、`startup`、`limits` 和 `execution_provider` 控制生成及启动参数。

环境变量可以覆盖配置，provider 与 daemon 使用同一 `settings.py` 解析优先级。
daemon 是内部 HTTP 进程，不解析用户命令行参数。MOSS 当前只接受 `speed=1.0`，
网页请求中的 provider 选择也不会覆盖后端游戏 manifest 的选择。

模型、Python wheels、SpaceMIT 库、参考音频和输出文件属于板端资产，默认被 Git
忽略。runtime API 变更时只需调整该功能包和底层交付，不需要修改
`backend/server.py` 或 `web/app.js`。
