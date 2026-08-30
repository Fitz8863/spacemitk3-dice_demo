# YOLOv8 Runtime 迁移说明

本目录由原工程 `/home/heweijie/spacemit-k3-dev/projects/dice-game/yolov8_objdetect` 迁移并重命名为通用 `yolov8_adjudicator` runtime。历史目录只作为迁移来源记录，不应在当前构建或部署中重新创建。

迁移时排除了 `.git/`、原有 `build/`、CMake 缓存和运行时 `.shaders/`，避免把旧工作区状态和板端生成物带进新仓库。runtime `config.json` 只保留硬件默认值，LLM transport 位于 Python 组件配置。

当前 C++ 程序作为 `vision_yolov8_adjudicator` 的私有 K3 runtime 使用。后端 provider
通过 `control-fd` / `event-fd` JSONL 通道发送 `START_ADJUDICATION`、
`FINAL_RESULT`、`STOP_ADJUDICATION` 和 `CANCEL`，runtime 输出通用 detection、stable
snapshot 与生命周期事件。游戏规则和 LLM prompt 不在此目录实现，而是在
`backend/games/<game_id>/manifest.json` 的 `vision_profile` 节点中声明；视频 WebRTC
基础地址和游戏 path 也由该节点管理。详细协议见同目录 `README.md`。
