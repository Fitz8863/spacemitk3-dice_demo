# MOSS-TTS-Nano runtime maintenance

这里是底层 SpaceMIT EP runtime 交付目录。生产 provider、配置和生命周期位于
`backend/components/tts_moss_nano/`。

保持以下约束：

- 不恢复 `run_demo.sh`、`run_interactive.sh`、`run_voice_clone.sh` 或其它用户 CLI；
- `src/` 只保留 runtime 库和必要的模型适配代码；
- HTTP bridge 通过长度前缀 WAV 协议输出 chunk，协议实现位于
  `backend/core/tts_protocol.py`；
- 模型、native library、参考音频和输出不提交；
- 修改后运行 `python3 -m py_compile backend/components/tts_moss_nano/*.py`，
  再运行 `python3 -m unittest tests.test_tts_a2 -v`；
- 真实 SpaceMIT EP、TCM 和音频验证必须在 K3 板端完成。
