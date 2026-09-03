# Qwen3-TTS runtime maintenance

这里是底层 K3 runtime 交付目录。Dice Arena 的调用契约和配置位于
`backend/components/tts_qwen3/`，新增或修改 TTS 行为时优先修改 provider 包，
不要恢复交互式脚本、`aplay` 播放器或单次 CLI。

保持以下约束：

- `start_server.sh` 只负责检查 runtime/model/ORT 并启动 `llama-server`；
- `stop_server.sh` 只停止由本目录启动且端口匹配的服务；
- 模型和板端私有音色文件不提交；
- 修改 runtime 后运行 `python3 -m py_compile backend/components/tts_qwen3/*.py`，
  再运行 `python3 -m unittest tests.test_tts_a2 -v`；
- 真实音频和 `/health` 验证必须在已部署模型的 K3 上完成。

不要把本目录当作独立产品 CLI。生产生命周期由
`backend/components/tts_qwen3/launcher.py` 和 `backend/componentctl.py` 管理。
