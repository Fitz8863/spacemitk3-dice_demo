# YOLOv8 Adjudicator Runtime 重命名与场景辅助设计

## 目标

将 C++ 视觉 runtime 从 `vision/yolov8_objdetect` 重命名为
`vision/yolov8_adjudicator`，删除生成物和已不参与功能包调用的骰子专用 LLM 源码，
并修复常驻模式下 `yolov8_enabled=false` 后收到 `START_ADJUDICATION` 不能恢复 YOLO
推理的问题。

## 边界

- Python 功能包 ID 仍为 `vision_yolov8_adjudicator`，视觉总接口仍为 `VisionProvider`。
- C++ 可执行文件仍命名为 `yolov8_camera`，它是 runtime 进程名，不是功能包 ID。
- 旧目录不保留兼容软链接或副本。
- `build/`、`.shaders/`、CMake 缓存和未被 generic control-fd 构建使用的
  `llm_dice_verifier.{h,cpp}` 不迁移。
- 用户已有的 `backend/games/dice/audio/fll.wav` 不修改、不删除、不纳入本次提交。

## 运行时行为

常驻模式启动时始终打开摄像头和 RTSP；`yolov8_enabled=false` 只表示初始处于 idle，
不进行预处理和推理。若存在 control-fd，runtime 仍初始化可复用的 OpenCL/YOLO 会话，
收到 `START_ADJUDICATION` 后将 `adjudication_active` 设为 true，开始推理；STOP/CANCEL
只停止当前轮推理并回到 idle。

## 黑线辅助信息

黑线检测是帧级场景几何辅助处理，不负责最终胜负。激活裁决后，对当前 BGR 帧执行
`detect_black_divider`，将检测到的分界线绘制到 RTSP/显示帧，并在 observation 中输出
安全的 `divider` 元数据（是否找到、方向、点和法向量）。稳定帧候选必须在启用该辅助
处理时包含有效分界线；未启用时保留通用 detection 稳定策略，避免位置检测或猜拳 profile
被骰子分区规则耦合。骰子 profile 通过 `vision.divider_detection=true` 启用该辅助处理，
其他游戏默认关闭。

## 配置与接口

- `backend/components/vision_yolov8_adjudicator/config.json` 的 runtime binary 和
  working_dir 指向新目录；不保存游戏规则或云端 prompt。
- `manifest.json.vision_profile.vision.divider_detection` 是游戏级开关。
- Python provider 继续消费通用 detections/snapshot；若 observation 提供 divider，
  `normalize_observation` 可使用其几何位置作为 `divider_regions` 的默认边界。

## 验证

- Python 单元测试覆盖：profile 开关、observation divider 字段、动态启动参数和旧目录
  引用不存在。
- K3 上使用 `cmake -S vision/yolov8_adjudicator -B ...` 构建，运行 `--self-test`，
  再用 resident control-fd 启动/停止一轮，确认日志显示 YOLO 推理和黑线检测路径。
