# YOLO 检测超时与可解释失败诊断设计

> 设计日期：2026-08-30  
> 适用组件：`vision_yolov8_adjudicator` 及骰子游戏 `vision_profile`

## 1. 问题

视觉裁决不能只等待 `adjudication_seconds` 的总预算。骰子叠放、遮挡、光线不足
或摄像头画面异常时，YOLO 可能永远无法得到双方完整且稳定的检测结果。原流程只
保存稳定帧，也只接受 `stable=true` 的 observation，因此超时后没有图片和检测证据
可以解释失败原因，前端只能显示笼统的超时。

## 2. 目标

- 增加独立的 `timeouts.yolo_detection_seconds`，只限制等待稳定 YOLO 结果的阶段。
- YOLO 超时后立即停止本轮推理，保留最新摄像头/MediaMTX 画面，并进入诊断阶段。
- runtime 在 active 阶段以受控频率覆盖写入每个视角的一张 `latest-<view>.jpg`，通过
  `diagnostic_snapshot` 事件携带图片、类别检测框和分界线辅助信息。
- 诊断 LLM 使用当前最新帧、检测摘要和游戏专属诊断 prompt，单轮无历史调用；诊断
  只允许返回原因，不允许返回 LEFT/RIGHT/TIE 胜负。
- 诊断 LLM 超时、网络失败、响应非法或未配置时，根据 YOLO 最近证据生成本地诊断。
- 最终返回 `diagnosed=true`、`retry_required=true` 的可解释失败，不伪造胜负结果，
  前端提示原因并邀请玩家重新开始。
- 正常稳定结果路径保持现有 YOLO 初判、LLM 复核、结果保持时长和多视角多数投票。

## 3. 配置

游戏的 `manifest.json -> vision_profile` 负责超时和诊断 prompt：

```json
{
  "timeouts": {
    "yolo_detection_seconds": 8,
    "adjudication_seconds": 120
  },
  "llm": {
    "timeout_seconds": 3,
    "diagnosis_system_prompt": "Do not declare a winner. Diagnose only.",
    "diagnosis_user_prompt_template": "Detector summary: {detector_summary}",
    "diagnosis_allowed_reason_codes": [
      "INCOMPLETE_OBJECTS", "OVERLAPPING_OBJECTS", "LOW_LIGHT", "OCCLUDED",
      "NO_OBJECTS_DETECTED", "UNSTABLE_DETECTION", "SCENE_GEOMETRY_UNCLEAR", "UNKNOWN"
    ]
  }
}
```

`adjudication_seconds` 是从开始检测到产生最终裁决的总处理预算，不包含裁决成功后的
画面保持时间；`yolo_detection_seconds` 默认回退到该总预算。正常裁决复核和失败原因
诊断统一使用 `llm.timeout_seconds` 作为单次大模型请求上限，不再设置独立的诊断超时。
`lifecycle.post_result_hold_seconds` 在结果产生后单独控制实时画面保持时长。每个游戏
可以独立调整这些值。

## 4. 调度流程

```text
START_ADJUDICATION
  -> active YOLO + diagnostic_snapshot
  -> 稳定 observation ?
       是 -> 规则计算 -> 一次 LLM 胜负复核 -> FINAL_RESULT -> STOP -> holding -> complete
       否 -> yolo_detection_seconds 到期
              -> STOP_ADJUDICATION
              -> 最新帧 + detector summary -> 一次诊断 LLM
                   成功 -> LLM 原因
                   超时/失败 -> YOLO 证据本地原因
              -> diagnosis event -> complete
              -> job error + retry_required
```

常驻 runtime 只停止本轮 YOLO 推理，不关闭摄像头和 RTSP/MediaMTX 链路；按局 runtime
在诊断完成后释放资源。诊断帧和稳定帧均由 provider 在消费后清理。

## 5. 结果合同

成功结果继续使用 `adjudicated=true` 和 `outcome.kind=winner`。失败诊断使用：

```json
{
  "adjudicated": false,
  "diagnosed": true,
  "retry_required": true,
  "diagnosis": {
    "reason_code": "INCOMPLETE_OBJECTS",
    "message": "当前目标数量不完整（LEFT=4、RIGHT=5，每侧应为 5 个），可能存在目标叠放、遮挡或漏检，请重新摆放后再试。",
    "source": "yolo_fallback",
    "llm_status": "timeout",
    "detected_counts": {"LEFT": 4, "RIGHT": 5},
    "retry": true
  }
}
```

`ComponentJob` 将这种结果标记为 `status=error`，同时保留结构化 `result`，SSE 和轮询
客户端可以显示具体原因。骰子 pipeline 不把诊断结果送入 winner/score 角色投影层。

## 6. 本地诊断优先级

1. 配置了 `expected_count` 且任一参与方数量不符：`INCOMPLETE_OBJECTS`；
2. 配置了 `expected_count` 且双方均为零：`NO_OBJECTS_DETECTED`；
3. 有目标但分界线明确不可用：`SCENE_GEOMETRY_UNCLEAR`；
4. 有目标但连续类别签名未稳定：`UNSTABLE_DETECTION`。

消息只描述检测证据和可能原因，不宣判左右胜负，也不推断玩家/Agent 身份。身份继续
由上层游戏 manifest 负责。

## 7. 验证

- Python：profile 校验、诊断规则、LLM JSON/超时、provider 超时分支、pipeline 和 job
  错误合同、前端诊断提示。
- C++：K3 上重新构建 runtime；执行 `--self-test --yolov8 --no-display --no-rtsp`；
  执行有限帧摄像头测试；确认 active 阶段输出 `diagnostic_snapshot`，STOP 后推理回到 idle。
- 全量门禁：`python3 -m pytest -q tests`、`python3 -m compileall -q backend`、
  `node --check web/app.js`、`node --check web/games/dice.js`、`git diff --check`。
