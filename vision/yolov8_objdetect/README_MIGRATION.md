# YOLOv8 迁移说明

本目录内容由原工程 `/home/heweijie/spacemit-k3-dev/projects/dice-game/yolov8_objdetect` 迁移而来，未修改核心推理代码。

迁移时排除了 `.git/`、原有 `build/` 和运行时 `.shaders/`，避免把旧工作区状态和板端生成物带进新仓库。`config.json` 使用了不含 API Key 的安全副本。

前端原型当前通过演示数据模拟 `ANALYSIS` 阶段；这套 C++ 程序仍作为 K3 端视觉推理模块独立编译运行。下一步可新增一个 bridge 进程，将 `DiceJudgment` 序列化为前端约定的 JSON 事件。
