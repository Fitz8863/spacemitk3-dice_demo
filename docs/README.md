# 项目文档索引

运行行为以代码、游戏清单和组件配置为准；文档用于解释职责边界、部署方法和配置参考。

## 当前有效文档

| 文档 | 用途 |
| --- | --- |
| [`../README.md`](../README.md) | 使用者和部署者入口：启动、配置、API 和常用验证命令。 |
| [`../FRAMEWORK_DISPATCH.md`](../FRAMEWORK_DISPATCH.md) | 从浏览器请求到视觉/TTS provider 的端到端调度说明。 |
| [`../backend/components/README.md`](../backend/components/README.md) | 可插拔 provider 功能包的目录、manifest 和接口约定。 |
| [`../vision/yolov8_objdetect/README.md`](../vision/yolov8_objdetect/README.md) | YOLOv8 K3 runtime、控制协议、快照和 MediaMTX 播放边界。 |
| [`../backend/components/tts_qwen3/README.md`](../backend/components/tts_qwen3/README.md) | Qwen3-TTS provider 的配置与运行说明。 |
| [`../backend/components/tts_moss_nano/README.md`](../backend/components/tts_moss_nano/README.md) | MOSS-TTS-Nano provider 的配置与运行说明。 |
| [`../backend/参数说明.md`](../backend/参数说明.md) | 全局配置 backend/config.json 字段参考（引擎槽位/音色语速/语音总闸、优先级阶梯、本地 TTS 钉死规则）。 |
| [`../backend/games/dice/参数说明.md`](../backend/games/dice/参数说明.md) | 骰子游戏 manifest.json 全字段参考（JSON 无注释，这份就是注释），含生效方式速查。 |
| [`../backend/components/asr_zipformer/参数说明.md`](../backend/components/asr_zipformer/参数说明.md) | ASR 引擎组件配置（采集设备/VAD 断句/绑核）全字段参考。 |
| [`../backend/components/tts_gptsovits/参数说明.md`](../backend/components/tts_gptsovits/参数说明.md) | 远程 GPT-SoVITS 组件配置（服务地址/请求采样参数/音色）全字段参考。 |
| [`../backend/components/tts_moss_nano/参数说明.md`](../backend/components/tts_moss_nano/参数说明.md) | 本地 MOSS 组件配置（音色克隆/生成参数/EP 绑核）全字段参考，补充其 README 未覆盖的段落。 |
| [`../backend/components/tts_qwen3/参数说明.md`](../backend/components/tts_qwen3/参数说明.md) | 本地 Qwen3-TTS 组件配置全字段参考。 |
| [`../backend/components/vision_yolov8_objdetect/参数说明.md`](../backend/components/vision_yolov8_objdetect/参数说明.md) | 视觉裁决组件配置 + runtime 硬件配置双文件参考，含游戏 profile 覆盖优先级。 |
| [`../dice_demo/README.md`](../dice_demo/README.md) | 机械臂子系统（NERO 抓放/摇骰）上游快照的运行说明。 |
| [`../dice_demo/STRUCTURE.md`](../dice_demo/STRUCTURE.md) | 机械臂子系统目录结构与清理边界（上游快照，随 subtree 同步）。 |
| [`../dice_demo/docs/INTEGRATION.md`](../dice_demo/docs/INTEGRATION.md) | 机械臂常驻控制器与本仓 `robot_arm_nero` 组件的 JSONL 接入契约。 |
| [`adr/0001-dice-demo-upstream-snapshot.md`](adr/0001-dice-demo-upstream-snapshot.md) | 架构决策记录：dice_demo 以上游快照子目录融入 main 的决策与后果。 |
| [`video_startup.md`](video_startup.md) | 视频启动/复用/重连的运行行为说明（`vision_always_on` 语义）。 |

## 配置入口

- 游戏配置：`backend/games/<game_id>/manifest.json`。其中的 `providers` 选择语义职责，`vision_profile` 描述该游戏的模型、类别、规则、视频 path、超时和结果保持时间。
- 视觉 runtime 配置：各游戏 `backend/games/<game_id>/adjudicator_config.json`（manifest 的 `vision_profile.runtime_config` 必填指向）。这里保存该游戏的摄像头、推理、RTSP 和 MediaMTX WebRTC 基础地址。
- provider 配置：`backend/components/<provider_id>/config.json`。这里保存适配器的运行时路径、端口和生命周期设置（LLM 组件已随复核链移除，现存配置不含任何云服务凭证）。
- 机械臂子系统：`dice_demo/` 是上游快照子目录（`git subtree` 合入，内部文件不可在本仓直接修改）；`backend/components/robot_arm_nero/config.json` 的 `demo_root` 指向它，改动需重启生效。

新增游戏通常只需要添加一个游戏目录和 manifest；新增 TTS 或视觉能力只需要添加对应 provider 功能包并在游戏 manifest 中选择。空间定位类视觉必须使用独立的 `role=localizer` 插槽，不能接入 `vision_adjudicator`。
