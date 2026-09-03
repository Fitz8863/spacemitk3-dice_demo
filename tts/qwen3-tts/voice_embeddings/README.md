# Speaker embedding assets

Dice Arena 只读取 `backend/components/tts_qwen3/config.json` 中配置的
`voice.speaker_file`。embedding 是板端私有运行时资产，不随仓库发布，也不提供
在线提取或交互式播放脚本。

如需更换 embedding，请在 K3 上把文件放入模型目录（或使用绝对路径），修改组件
配置后重启 `tts_qwen3`。文件应符合底层 Qwen3 runtime 要求的 raw `float32[1024]`
格式；具体转换流程由模型交付方负责。
