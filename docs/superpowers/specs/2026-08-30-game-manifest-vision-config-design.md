# 游戏 Manifest 统一视觉配置设计

## 目标

让 `backend/games/<game_id>/manifest.json` 成为单个游戏的完整配置入口。游戏涉及视觉裁决时，在 manifest 的 `vision_profile` 节点声明模型、规则、LLM 提示词、视频地址和裁决时限；视觉组件包只保存部署级运行参数。

## 角色边界

`VisionProvider` 是视觉能力的总接口，不等于裁决器。当前 YOLOv8 实现注册为 `type=vision`、`role=adjudicator`，只实现 `VisionAdjudicatorProvider.adjudicate()`。未来位置检测器注册为 `role=localizer`，实现 `VisionLocalizerProvider.locate()`，使用独立的游戏配置节点和 provider slot，不读取裁决规则或胜负语义。

## 配置分层

游戏 manifest 内嵌：

- `vision_profile.vision`：模型路径、类别映射、参与方、稳定帧；
- `vision_profile.multi_view`：视角、摄像头和多数投票；
- `vision_profile.rule`：游戏裁决规则；
- `vision_profile.llm`：无状态 prompt、允许结果和单次 LLM 超时；
- `vision_profile.video`：WebRTC 基础地址、MediaMTX path 和前端播放策略；
- `vision_profile.lifecycle`：最终结果后的画面保持时间；
- `vision_profile.timeouts.adjudication_seconds`：从开始裁决到 complete 的整轮总预算。

视觉组件 `backend/components/vision_yolov8_adjudicator/config.json` 保留：

- YOLO 二进制和工作目录；
- resident/prewarm 运行模式；
- RTSP 发布参数；
- 云端 LLM endpoint、模型默认值和 API key；
- 事件协议。

MediaMTX 的基础地址不再从组件 config 读取。`DICE_MEDIAMTX_WEBRTC_BASE_URL` 仅作为临时运维覆盖，正式来源是游戏 profile 的 `video.webrtc_base_url`。

## 运行和兼容

游戏加载器优先校验并使用 manifest 内嵌的 `vision_profile`。迁移期仍支持同目录 `vision_profile.json`，仅当 manifest 没有内嵌 profile 时读取；内嵌和外部 profile 同时存在时以内嵌为准。所有 profile 都必须匹配外层游戏 `id`。

provider 解析游戏 profile 的 `timeouts.adjudication_seconds`，没有该字段时回退到 `DICE_JOB_TIMEOUT_SECONDS`（默认 120 秒）。LLM 请求仍由 `llm.timeout_seconds` 单独限制；结果保持时间由 `lifecycle.post_result_hold_seconds` 限制，但不得突破整轮总预算。

`video.webrtc_base_url` 必须是无路径、无 query、无 fragment、无凭据的 HTTP(S) 地址；`video.path` 仍只允许安全的 URL path。多视角只覆盖各自 path，默认继承 profile 级基础地址。

## 错误处理

- profile 校验失败时跳过该游戏，并记录明确的配置错误；
- 缺少稳定帧在整轮预算耗尽后返回 YOLO timeout；
- LLM 在自己的预算内超时，按既有策略回退 YOLO；
- 已获得结果但整轮预算不足以完成保持时间时，缩短保持时间并发送 complete，不阻塞后端关闭；
- 浏览器公开的 games/health 数据只返回视频地址和能力元数据，不返回 prompt、模型绝对路径、摄像头设备路径或 API key。

## 测试要求

- 内嵌 profile 能被加载并校验；
- 外部 profile 兼容回退仍可用；
- 视频 URL 使用 profile 的 WebRTC 基础地址和游戏 path；
- 游戏级裁决总超时优先于全局环境默认值；
- LLM 超时和结果保持仍遵守各自参数；
- 视觉角色契约继续拒绝把 localizer 接入 adjudicator 插槽；
- 全量本地测试、K3 启动/健康/裁决/停止流程通过。
