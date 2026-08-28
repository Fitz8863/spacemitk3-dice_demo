# MOSS model variants

生产功能包只依赖 `backend/components/tts_moss_nano/config.json` 指定的
`runtime.model_dir`。如果板端需要切换动态或固定 KV 模型，只需把模型目录切换到
兼容 MOSS runtime 的交付物并重启 provider；不需要修改 Dice Arena 调度代码。

模型目录必须包含 `browser_poc_manifest.json`，并与当前 `OnnxTtsRuntime` API 和
SpaceMIT EP 版本匹配。切换前请在 K3 上运行组件 health 和一次短文本合成，确认
采样率、声道数与浏览器播放协议一致。
