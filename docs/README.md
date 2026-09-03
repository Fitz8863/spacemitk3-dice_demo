# 项目文档索引

这里按“当前有效说明”和“历史设计记录”区分文档。运行行为以代码、游戏清单和组件配置为准；文档用于解释职责边界、部署方法和迁移背景。

## 当前有效文档

| 文档 | 用途 |
| --- | --- |
| [`../README.md`](../README.md) | 使用者和部署者入口：启动、配置、API 和常用验证命令。 |
| [`../TTS配置与切换指南.md`](../TTS配置与切换指南.md) | TTS provider 切换步骤与三层参数（游戏/组件/请求）详解。 |
| [`../AI_PROJECT_CONTEXT.md`](../AI_PROJECT_CONTEXT.md) | AI 或新开发者接手时的上下文、目录职责、安全约束和当前状态。 |
| [`../CLAUDE.md`](../CLAUDE.md) | 编程代理修改本仓库时必须遵守的工程约束。 |
| [`../FRAMEWORK_DISPATCH.md`](../FRAMEWORK_DISPATCH.md) | 从浏览器请求到视觉/TTS provider 的端到端调度说明。 |
| [`../backend/components/README.md`](../backend/components/README.md) | 可插拔 provider 功能包的目录、manifest 和接口约定。 |
| [`../vision/yolov8_adjudicator/README.md`](../vision/yolov8_adjudicator/README.md) | YOLOv8 K3 runtime、控制协议、快照和 MediaMTX 播放边界。 |
| [`../backend/components/tts_qwen3/README.md`](../backend/components/tts_qwen3/README.md) | Qwen3-TTS provider 的配置与运行说明。 |
| [`../backend/components/tts_moss_nano/README.md`](../backend/components/tts_moss_nano/README.md) | MOSS-TTS-Nano provider 的配置与运行说明。 |
| [`../backend/参数说明.md`](../backend/参数说明.md) | 全局配置 backend/config.json 字段参考（引擎槽位/音色语速/语音总闸、优先级阶梯、本地 TTS 钉死规则）。 |
| [`../backend/games/dice/参数说明.md`](../backend/games/dice/参数说明.md) | 骰子游戏 manifest.json 全字段参考（JSON 无注释，这份就是注释），含生效方式速查。 |
| [`../backend/components/asr_zipformer/参数说明.md`](../backend/components/asr_zipformer/参数说明.md) | ASR 引擎组件配置（采集设备/VAD 断句/绑核）全字段参考。 |
| [`../backend/components/tts_gptsovits/参数说明.md`](../backend/components/tts_gptsovits/参数说明.md) | 远程 GPT-SoVITS 组件配置（服务地址/请求采样参数/音色）全字段参考。 |
| [`../backend/components/tts_moss_nano/参数说明.md`](../backend/components/tts_moss_nano/参数说明.md) | 本地 MOSS 组件配置（音色克隆/生成参数/EP 绑核）全字段参考，补充其 README 未覆盖的段落。 |
| [`../backend/components/tts_qwen3/参数说明.md`](../backend/components/tts_qwen3/参数说明.md) | 本地 Qwen3-TTS 组件配置全字段参考。 |
| [`../backend/components/vision_yolov8_adjudicator/参数说明.md`](../backend/components/vision_yolov8_adjudicator/参数说明.md) | 视觉裁决组件配置 + runtime 硬件配置双文件参考，含游戏 profile 覆盖优先级。 |

## 配置入口

- 游戏配置：`backend/games/<game_id>/manifest.json`。其中的 `providers` 选择语义职责，`vision_profile` 描述该游戏的模型、类别、规则、LLM prompt、视频 path、超时和结果保持时间。
- 视觉 runtime 配置：`vision/yolov8_adjudicator/config.json`。这里保存摄像头、推理、RTSP 和 MediaMTX WebRTC 基础地址等部署默认值。
- provider 配置：`backend/components/<provider_id>/config.json`。这里保存适配器的运行时路径、端口、endpoint 和生命周期设置；LLM 密钥也直接保存在该文件的 `llm` 段（仓库须保持私有）。

新增游戏通常只需要添加一个游戏目录和 manifest；新增 TTS 或视觉能力只需要添加对应 provider 功能包并在游戏 manifest 中选择。空间定位类视觉必须使用独立的 `role=localizer` 插槽，不能接入 `vision_adjudicator`。

## 历史资料

- [`archive/legacy/CosyVoice-TTS-调用说明.md`](archive/legacy/CosyVoice-TTS-调用说明.md)：旧版 CosyVoice 云端调用手册，保留作追溯，不是当前 TTS 入口。
- [`archive/vision/README_MIGRATION.md`](archive/vision/README_MIGRATION.md)：YOLOv8 runtime 从旧目录迁移并重命名的记录，当前运行说明以 `vision/yolov8_adjudicator/README.md` 为准。
- [`superpowers/plans/`](superpowers/plans/) 和 [`superpowers/specs/`](superpowers/specs/)：历次架构设计和实现计划，记录当时的决策，不作为当前配置示例的唯一来源。

历史文档中的旧路径（例如 `vision/yolov8_objdetect`）只用于解释迁移过程。新代码、配置和部署命令统一使用 `vision/yolov8_adjudicator`。
