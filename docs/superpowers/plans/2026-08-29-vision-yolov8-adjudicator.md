# vision_yolov8_adjudicator 架构重构实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将当前骰子专用视觉实现重构为可按游戏 profile 复用的 YOLOv8 视觉裁决功能包，支持多视角多数投票、稳定帧图片 LLM 复核、MediaMTX WebRTC 地址配置、摄像头预热和结果后的画面保持。

**架构：** 后端通过 `VisionAdjudicationRequest` 调用 `vision_yolov8_adjudicator`，游戏只提供经过校验的 `vision_profile.json`。YOLOv8 runtime 输出统一 detection / stable snapshot 事件；provider 在 Python 侧完成 profile 规则聚合、单轮多模态 LLM 请求和最终结果优先级。常驻 runtime 提前打开摄像头和视频链路，用户点击后通过私有控制通道启动本局推理。

**技术栈：** Python 3 标准库、现有 HTTP/SSE bridge、OpenAI-compatible HTTP API、C++17、OpenCV、GStreamer、OpenCL、SpaceMIT ONNX Runtime EP、MediaMTX WebRTC iframe 播放。

---

## 文件清单与职责

### 新增

- `backend/components/vision_yolov8_adjudicator/manifest.json`：新组件声明、能力和入口。
- `backend/components/vision_yolov8_adjudicator/config.json`：通用 runtime、摄像头、控制通道和 MediaMTX 基础地址。
- `backend/components/vision_yolov8_adjudicator/provider.py`：对外 `VisionAdjudicatorProvider` adapter，编排单路/多路运行时、规则和 LLM。
- `backend/components/vision_yolov8_adjudicator/process.py`：YOLOv8 子进程、event FD、Unix 控制通道、超时和资源回收。
- `backend/components/vision_yolov8_adjudicator/profile.py`：profile/config 加载、字段校验、路径安全和 WebRTC URL 合成。
- `backend/components/vision_yolov8_adjudicator/rules.py`：多数投票、`numeric_compare`、`categorical_relation` 和统一结果投影。
- `backend/components/vision_yolov8_adjudicator/llm.py`：无状态单轮多模态 OpenAI-compatible 请求，图片转 base64，结果解析。
- `backend/components/vision_yolov8_adjudicator/README.md`：功能包配置、控制协议和接入新游戏说明。
- `backend/games/dice/vision_profile.json`：骰子模型、规则、prompt、WebRTC path 和单路配置。
- `tests/test_vision_adjudicator.py`：profile、规则、LLM、fake runtime 和 provider 生命周期测试。

### 修改

- `backend/core/vision.py`：增加 `VisionAdjudicationRequest`，将 adjudicate 接口改为请求对象。
- `backend/core/games.py`：加载和校验游戏 vision profile，向 pipeline 提供受信任 profile 快照。
- `backend/core/jobs.py`：支持 `holding` 阶段，verified/adjudicated 结果不再隐式完成任务。
- `backend/server.py`：创建任务时传递 game/profile，health 暴露 profile 和视频能力。
- `backend/games/dice/pipeline.py`：只负责选择 provider 和构造请求，不包含骰子规则。
- `web/index.html`：删除固定 MediaMTX URL。
- `web/games/dice.js`：消费 `video`、`result`、`holding` 事件，保持视频直到 complete。
- `vision/yolov8_objdetect/src/main.cpp`：通用 observation/snapshot、预热和运行时控制通道、多轮 resident 生命周期。
- `vision/yolov8_objdetect/src/llm_dice_verifier.cpp`、`.h`：从业务裁决路径移除骰子专属 LLM；保留或替换为通用 runtime 代码。
- `vision/yolov8_objdetect/CMakeLists.txt`：加入控制/快照所需源文件（如拆分后）。
- `tests/test_components_and_jobs.py`、`tests/test_server_api.py`：更新新 provider ID、请求接口和 holding 语义。
- `README.md`、`backend/components/README.md`、`AI_PROJECT_CONTEXT.md`、`FRAMEWORK_DISPATCH.md`：同步新组件、profile、MediaMTX 和预热说明。

### 删除或迁移

- `backend/components/vision_yolo/`：新 provider 完成并通过兼容别名验证后删除；删除前保留 `vision_yolo → vision_yolov8_adjudicator` registry 迁移映射。
- C++ 中 `DiceResultSnapshot`、`LlmDiceVerifier`、骰子专属 prompt 和固定 5 + 5 业务分支：迁移为 profile/rules.py 处理后删除。

---

## 任务 1：建立统一请求接口和 profile 加载器

**文件：**

- 修改：`backend/core/vision.py`
- 修改：`backend/core/games.py`
- 创建：`backend/components/vision_yolov8_adjudicator/profile.py`
- 创建：`backend/components/vision_yolov8_adjudicator/config.json`
- 创建：`backend/components/vision_yolov8_adjudicator/manifest.json`
- 创建：`backend/games/dice/vision_profile.json`
- 测试：`tests/test_vision_adjudicator.py`

- [ ] **步骤 1：编写失败测试，锁定 profile 和 URL 规则**

```python
def test_profile_loads_dice_and_composes_mediamtx_url(tmp_path):
    profile = load_profile(ROOT / "backend/games/dice/vision_profile.json")
    config = load_component_config(ROOT / "backend/components/vision_yolov8_adjudicator")
    assert profile["game_id"] == "dice"
    assert profile["llm"]["context_mode"] == "single_turn_no_history"
    assert profile["video"]["path"] == "/dice/"
    assert compose_video_url(config["mediamtx"]["webrtc_base_url"], profile["video"]["path"]) == (
        "http://100.118.229.28:8889/dice/"
    )

def test_profile_rejects_full_url_in_game_path(tmp_path):
    bad = {"schema_version": 1, "game_id": "bad", "video": {"path": "https://x/"}}
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ProfileError, match="video.path"):
        load_profile(path)
```

- [ ] **步骤 2：运行测试确认接口尚未实现**

运行：`python3 -m pytest tests/test_vision_adjudicator.py -q`

预期：FAIL，报错来自缺少 `profile` 模块或 `load_profile`。

- [ ] **步骤 3：实现 profile/config 校验和安全 URL 合成**

实现以下函数并保持纯函数边界：

```python
def load_profile(path: Path) -> dict[str, Any]: ...
def load_component_config(package_dir: Path) -> dict[str, Any]: ...
def compose_video_url(base_url: str, path: str) -> str: ...
def resolve_project_path(value: str, project_root: Path) -> Path: ...
```

校验 `schema_version=1`、`game_id`、`vision.model`、`llm.system_prompt`、`llm.user_prompt_template`、`llm.allowed_outcomes`、`video.path`、`multi_view`、`lifecycle.post_result_hold_seconds`。基础 URL 只允许无 path/query/fragment/credentials 的 `http` 或 `https`；path 只能是以 `/` 开头的 URL path，拒绝 `..`、query、fragment、控制字符和完整 URL。

- [ ] **步骤 4：加入新 manifest/config/profile**

组件 manifest 使用：

```json
{
  "id": "vision_yolov8_adjudicator",
  "type": "vision",
  "role": "adjudicator",
  "name": "YOLOv8 Vision Adjudicator",
  "version": "2.0",
  "enabled": true,
  "entry": "provider.py:VisionYolov8Adjudicator",
  "capabilities": ["object_detection", "stable_frame", "multiview", "llm_verification"],
  "config": "config.json"
}
```

组件 config 的 `runtime.prewarm_camera=true`、`runtime.mode=resident`、`mediamtx.webrtc_base_url=http://100.118.229.28:8889`；骰子 profile 使用 `video.path=/dice/`，并使用无状态图片裁决 prompt，不再向 prompt 注入 `{left_sum}` / `{right_sum}`。

- [ ] **步骤 5：扩展 games registry 和 dice pipeline 的 profile 传递**

`load_games()` 加载同目录 `vision_profile.json`，校验失败时跳过该游戏并记录明确错误。`run_game()` 将 profile 快照传入 pipeline；pipeline 构造：

```python
request = VisionAdjudicationRequest(
    game_id=GAME_ID,
    profile=manifest["vision_profile"],
    request_id=uuid.uuid4().hex,
    timeout_seconds=timeout_seconds,
)
return adjudicate(request, on_log=on_log, on_event=on_event, is_cancelled=is_cancelled)
```

- [ ] **步骤 6：运行测试并提交**

运行：`python3 -m pytest tests/test_vision_adjudicator.py tests/test_components_and_jobs.py -q`

预期：新 profile/config 测试和原有组件测试通过；旧 provider ID 测试暂时按迁移兼容映射通过。

提交：`git add backend/core/vision.py backend/core/games.py backend/components/vision_yolov8_adjudicator backend/games/dice/vision_profile.json tests/test_vision_adjudicator.py && git commit -m "feat: add vision adjudicator request and profiles"`

---

## 任务 2：实现通用规则、最终结果策略和多视角融合

**文件：**

- 创建：`backend/components/vision_yolov8_adjudicator/rules.py`
- 修改：`backend/components/vision_yolov8_adjudicator/profile.py`
- 测试：`tests/test_vision_adjudicator.py`

- [ ] **步骤 1：编写失败测试，固定结果策略**

```python
def test_majority_vote_requires_strict_majority():
    assert fuse_yolo_outcomes(["LEFT", "LEFT", "RIGHT"]) == "LEFT"
    assert fuse_yolo_outcomes(["LEFT", "RIGHT"]) is None

def test_llm_success_overrides_yolo_mismatch():
    result = finalize_outcome(yolo_outcome="LEFT", llm_outcome="RIGHT", llm_status="success")
    assert result["outcome"]["value"] == "RIGHT"
    assert result["decision_source"] == "llm_override"

def test_llm_timeout_falls_back_to_yolo():
    result = finalize_outcome(yolo_outcome="LEFT", llm_outcome=None, llm_status="timeout")
    assert result["outcome"]["value"] == "LEFT"
    assert result["decision_source"] == "yolo_timeout_fallback"
```

- [ ] **步骤 2：运行测试确认失败**

运行：`python3 -m pytest tests/test_vision_adjudicator.py -q`

预期：FAIL，报错来自缺少 `fuse_yolo_outcomes` 或 `finalize_outcome`。

- [ ] **步骤 3：实现规则函数**

实现：

```python
def fuse_yolo_outcomes(outcomes: Sequence[str]) -> str | None: ...
def evaluate_rule(rule: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> str: ...
def finalize_outcome(*, yolo_outcome: str | None, llm_outcome: str | None, llm_status: str) -> dict[str, Any]: ...
def project_result(profile: Mapping[str, Any], decision: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]: ...
```

`majority_vote` 只接受严格多数；并列返回 `None` 和 `multi_view_no_majority`。LLM 合法成功结果总是最终结果；仅 `timeout` 允许 YOLO fallback；LLM 其他失败返回错误。结果包含 `adjudicated`、`outcome`、`decision_source`、`verification`、`evidence`、`profile_id` 和 `provider_id`，骰子兼容字段由 profile 投影层填充。

- [ ] **步骤 4：加入 numeric/categorical profile 规则测试**

使用固定 observation fixtures 验证骰子求和、猜拳关系、数量校验、未知 outcome 拒绝和多路严格多数。所有测试不得导入 `games.dice.pipeline` 或骰子专属 C++ 类型。

- [ ] **步骤 5：运行测试并提交**

运行：`python3 -m pytest tests/test_vision_adjudicator.py -q`

预期：所有 profile、规则和最终结果策略测试通过。

提交：`git add backend/components/vision_yolov8_adjudicator/rules.py backend/components/vision_yolov8_adjudicator/profile.py tests/test_vision_adjudicator.py && git commit -m "feat: add profile rules and multiview fusion"`

---

## 任务 3：实现无状态多模态 LLM verifier 和稳定帧 snapshot

**文件：**

- 创建：`backend/components/vision_yolov8_adjudicator/llm.py`
- 修改：`backend/components/vision_yolov8_adjudicator/process.py`
- 测试：`tests/test_vision_adjudicator.py`

- [ ] **步骤 1：编写失败测试，锁定请求体**

```python
def test_llm_request_is_single_turn_with_image(monkeypatch, tmp_path):
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg-bytes")
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"winner":"LEFT"}'}}]}

    verifier = OpenAICompatibleVisionVerifier(post=fake_post)
    assert verifier.verify(
        image_path=image,
        system_prompt="Judge the image.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT", "RIGHT", "TIE"],
        timeout_seconds=3,
    ).outcome == "LEFT"
    assert len(captured["messages"]) == 2
    assert captured["messages"][1]["content"][0]["type"] == "image_url"
```

- [ ] **步骤 2：运行测试确认失败**

运行：`python3 -m pytest tests/test_vision_adjudicator.py -q`

预期：FAIL，报错来自缺少 `OpenAICompatibleVisionVerifier`。

- [ ] **步骤 3：实现 verifier**

使用标准库 `urllib.request`，构造：

```json
{
  "model": "<profile model>",
  "messages": [
    {"role":"system","content":"<profile system prompt>"},
    {"role":"user","content":[
      {"type":"text","text":"<profile user prompt>"},
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
    ]}
  ]
}
```

不发送历史 messages、job history、YOLO 数字或其他游戏 profile。解析严格 JSON winner/outcome；响应合法值才算 `success`。连接和读取超过 timeout 返回 `timeout`；HTTP 错误、无效 JSON 和未知 outcome 返回 `failure`。

- [ ] **步骤 4：实现 snapshot 生命周期**

`process.py` 读取 observation 的 `snapshot.path`，验证路径位于任务临时目录、格式为 JPEG/PNG，调用 verifier 前读取，调用结束后在 `finally` 删除文件。事件引用不得是 URL、绝对路径之外的任意客户端输入，也不得跨任务复用。

- [ ] **步骤 5：运行测试并提交**

运行：`python3 -m pytest tests/test_vision_adjudicator.py -q`

预期：请求体、timeout、无效响应和 snapshot 清理测试全部通过。

提交：`git add backend/components/vision_yolov8_adjudicator/llm.py backend/components/vision_yolov8_adjudicator/process.py tests/test_vision_adjudicator.py && git commit -m "feat: add stateless multimodal llm verifier"`

---

## 任务 4：实现 Python provider 的按局、预热和多视角调度

**文件：**

- 修改：`backend/components/vision_yolov8_adjudicator/provider.py`
- 修改：`backend/components/vision_yolov8_adjudicator/process.py`
- 修改：`backend/core/vision.py`
- 测试：`tests/test_vision_adjudicator.py`

- [ ] **步骤 1：编写 fake runtime 测试**

fake runtime 事件顺序：

```python
[
    {"event": "started", "phase": "starting"},
    {"event": "ready", "phase": "idle"},
    {"event": "video", "url": "http://100.118.229.28:8889/dice/"},
    {"event": "observation", "stable": True, "yolo_outcome": "LEFT", "snapshot": {"path": "..."}},
]
```

测试 provider 必须在调用 adjudicate 时发送 `START_ADJUDICATION`，收到 observation 后只调用一次 LLM，发送 `FINAL_RESULT`，然后收到 result/complete；LLM timeout 时结果 source 为 `yolo_timeout_fallback`。

- [ ] **步骤 2：实现 process adapter**

`YoloRuntimeProcess` 暴露：

```python
start(profile, view_id, prewarm=True) -> None
send(command: dict[str, Any]) -> None
events() -> Iterator[dict[str, Any]]
stop() -> None
```

使用独立 event FD 读取 JSONL，Unix socket/pipe 发送 JSON 控制命令，stdout/stderr 只进入 `on_log`。每个 process 绑定一个 view；多路 process 并行创建，使用统一截止时间和 `ThreadPoolExecutor`，禁止串行启动或等待。

- [ ] **步骤 3：实现 provider 状态机**

状态：`prewarming`、`idle`、`detecting`、`verifying`、`holding`、`complete`、`error`。`video` 事件在 `ready` 后发送，结果事件只发送一次。多路 observation 达到 `min_views` 后先多数融合，再把所有稳定 snapshot 以固定 view 顺序交给一次 verifier。

- [ ] **步骤 4：实现结果后保持**

收到最终结果后立即发送：

```json
{"event":"result","adjudicated":true,"decision_source":"consensus",...}
{"event":"phase","phase":"holding","remaining_ms":2500}
```

保持期间每 250 ms 发送剩余时间，禁止重新检测和重复 LLM；保持结束发送 `complete` 并按 mode 发送 `STOP_ADJUDICATION` 或返回 idle。`post_result_hold_seconds=0` 直接发送 complete。

- [ ] **步骤 5：运行 provider 测试并提交**

运行：`python3 -m pytest tests/test_vision_adjudicator.py -q`

预期：fake runtime 的按局、预热、多视角、多数、LLM timeout、LLM mismatch、holding、cancel、timeout 和进程回收测试全部通过。

提交：`git add backend/components/vision_yolov8_adjudicator backend/core/vision.py tests/test_vision_adjudicator.py && git commit -m "feat: implement vision provider lifecycle"`

---

## 任务 5：接入 `ComponentJob` holding 状态和 HTTP profile

**文件：**

- 修改：`backend/core/jobs.py`
- 修改：`backend/server.py`
- 修改：`backend/core/games.py`
- 修改：`backend/games/dice/pipeline.py`
- 测试：`tests/test_components_and_jobs.py`
- 测试：`tests/test_server_api.py`

- [ ] **步骤 1：编写 holding 生命周期测试**

```python
def test_verified_result_does_not_complete_job_before_holding_finishes():
    release = threading.Event()
    def run(_log, _cancelled, on_event):
        on_event({"event": "result", "adjudicated": True, "outcome": {"value": "LEFT"}})
        on_event({"event": "phase", "phase": "holding"})
        release.wait(timeout=1)
        on_event({"event": "complete", "phase": "complete"})
        return {"adjudicated": True}
    job = ComponentJob(run)
    job.start()
    time.sleep(0.05)
    assert job.snapshot()["status"] == "running"
    assert job.snapshot()["phase"] == "holding"
    release.set()
    job.thread.join(timeout=2)
    assert job.snapshot()["status"] == "success"
```

- [ ] **步骤 2：运行测试确认旧隐式 complete 规则失败**

运行：`python3 -m pytest tests/test_components_and_jobs.py::test_verified_result_does_not_complete_job_before_holding_finishes -q`

预期：FAIL，当前 `_append_event_locked()` 会把 verified 事件直接设置为 complete。

- [ ] **步骤 3：修改 job 事件处理**

只由显式 `phase` 和 `complete` 事件更新阶段；兼容旧结果时使用 `adjudicated` 或旧 `verified` 生成结果事件，但不设置 complete。`_succeed()` 只有 provider 返回且任务尚未 terminal 时才将 status 置为 success；holding 期间 cancel 原子地转 error。

- [ ] **步骤 4：把 profile 传递到 HTTP/SSE**

`create_adjudication_job()` 调用 `run_game()` 时传递 profile；`/api/health` 返回选中的新 provider、profile 校验状态、MediaMTX base URL 和预热状态。请求体只读取 `game`，忽略或拒绝 `model`、`prompt`、`rule`、`video` 和 `post_result_hold_seconds`。

- [ ] **步骤 5：运行回归测试并提交**

运行：`python3 -m pytest tests/test_components_and_jobs.py tests/test_server_api.py -q`

预期：holding SSE、cancel、旧 analyze 别名、provider 选择和 TTS 相关测试全部通过。

提交：`git add backend/core/jobs.py backend/server.py backend/core/games.py backend/games/dice/pipeline.py tests/test_components_and_jobs.py tests/test_server_api.py && git commit -m "feat: keep adjudication jobs alive during holding"`

---

## 任务 6：修改 C++ runtime，支持通用 observation、图片快照和预热控制

**文件：**

- 修改：`vision/yolov8_objdetect/src/main.cpp`
- 修改：`vision/yolov8_objdetect/src/llm_dice_verifier.cpp`
- 修改：`vision/yolov8_objdetect/src/llm_dice_verifier.h`
- 修改：`vision/yolov8_objdetect/CMakeLists.txt`
- 测试：`vision/yolov8_objdetect/build/yolov8_camera --help`
- 测试：K3 板端 CMake build、self-test 和 fake control session

- [ ] **步骤 1：先加入 CLI/control 协议失败检查**

在 `main.cpp` 增加 `--control-fd FD`、`--snapshot-dir PATH`、`--prewarm` 和 `--no-llm` 的 help 文本；本地运行 `./build/yolov8_camera --help | rg 'control-fd|snapshot-dir|prewarm'`，在实现解析前预期找不到新增选项。

- [ ] **步骤 2：实现 control reader 和预热状态**

使用继承 FD 读取 `vision-control-v1` JSONL：`START_ADJUDICATION`、`STOP_ADJUDICATION`、`FINAL_RESULT`、`CANCEL`。进程启动后先打开摄像头和 RTSP 发布，发送 `started`、`ready`、`video`；`--prewarm` 下不计稳定帧、不调用 detector、不调用 LLM。收到 START 后在同一进程启用 YOLO pipeline。

- [ ] **步骤 3：抽取可重复的 inference round**

将当前摄像头/预处理/推理/显示线程移入 `run_inference_round(...)`，round 结束时停止并 join worker，但不关闭 camera、RTSP streamer 或 control reader。外层循环在 `idle` 等待下一次 START；同一 process 可连续运行两局。所有视角由 provider 启动独立 runtime process，单个 process 内只负责一个 view。

- [ ] **步骤 4：输出通用 observation 和 snapshot**

稳定帧时使用 `cv::imwrite` 保存到任务 snapshot 目录，并通过 event FD 输出：

```json
{
  "event":"observation",
  "view_id":"front",
  "frame_id":1820,
  "stable":true,
  "yolo_outcome":"LEFT",
  "snapshot":{"format":"image/jpeg","path":"/tmp/vision-job/stable-front-1820.jpg"},
  "detections":[{"class_id":0,"label":"1","confidence":0.96,"bbox":[10,20,80,90]}]
}
```

不再在 C++ 内调用骰子 LLM；Python provider 收到 observation 后发送 `FINAL_RESULT`，C++ 只把结果叠加到视频、发送 result/complete 事件并进入 holding/idle。

- [ ] **步骤 5：移除 C++ dice-only LLM 依赖**

删除 `LlmDiceVerifier` 对 `left_sum/right_sum` prompt 的调用和 `DiceResultSnapshot` 作为外部契约的用法；C++ 保留硬件推理、稳定计数和检测框输出。CMake 删除不再使用的源文件或替换为通用 transport 文件。

- [ ] **步骤 6：在开发机做静态/帮助检查，在 K3 编译验证**

开发机运行：`cmake --build vision/yolov8_objdetect/build -j2`（SDK 可用时）；K3 运行：

```bash
cmake -S vision/yolov8_objdetect -B vision/yolov8_objdetect/build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build vision/yolov8_objdetect/build -j4
vision/yolov8_objdetect/build/yolov8_camera --help
vision/yolov8_objdetect/build/yolov8_camera --model vision/yolov8_objdetect/models/best.q.onnx --self-test --no-display
```

预期：build exit 0，help 显示 control/prewarm 选项，self-test exit 0。硬件不可用时记录明确阻塞，不把静态检查当作实机通过。

- [ ] **步骤 7：提交 C++ 迁移**

提交：`git add vision/yolov8_objdetect && git commit -m "feat: add prewarm control and generic vision snapshots"`

---

## 任务 7：前端动态 MediaMTX iframe 和 holding 展示

**文件：**

- 修改：`web/index.html`
- 修改：`web/games/dice.js`
- 测试：`tests/test_server_api.py`
- 测试：`tests/test_web_contract.py`

- [ ] **步骤 1：编写前端契约测试**

```python
def test_frontend_has_no_hardcoded_mediamtx_host():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    assert "100.118.229.28:8889" not in html
    assert "data-stream-url" not in html
    assert "event.event === 'video'" in js
```

- [ ] **步骤 2：实现 video event 驱动播放**

删除 iframe `data-stream-url`。`dice.js` 保存最近 `video` 事件，按事件 URL 设置 iframe，并继续附加 `autoplay=1`、`muted=1`、`controls=0`、`playsinline=1`。`video` 事件的 URL 只能来自 SSE snapshot/event。

- [ ] **步骤 3：实现 holding UI**

`applyAnalysisSnapshot()` 在 `phase=holding` 时保留分析 view、结果和 iframe，显示剩余毫秒；只在 `status=success` / `phase=complete` 后调用 `showResult()`，而不是收到 result 时关闭视频。错误和取消清理 iframe。

- [ ] **步骤 4：运行测试并提交**

运行：`python3 -m pytest tests/test_web_contract.py tests/test_server_api.py -q`

预期：无固定 host、video event、holding 和 SSE complete 测试通过。

提交：`git add web/index.html web/games/dice.js tests/test_web_contract.py tests/test_server_api.py && git commit -m "feat: drive vision video from profile events"`

---

## 任务 8：多摄像头配置、profile 示例和 API/health 完整接线

**文件：**

- 修改：`backend/games/dice/vision_profile.json`
- 修改：`backend/components/vision_yolov8_adjudicator/config.json`
- 修改：`backend/server.py`
- 修改：`backend/core/games.py`
- 创建：`backend/games/rps/vision_profile.json`
- 测试：`tests/test_vision_adjudicator.py`
- 测试：`tests/test_server_api.py`

- [ ] **步骤 1：增加多视角 profile fixture**

骰子默认保持单路兼容：`multi_view.enabled=false`、`min_views=1`。测试 fixture 设置两路：

```json
{
  "multi_view": {
    "enabled": true,
    "min_views": 2,
    "yolo_fusion": "majority_vote",
    "llm_images": "all_stable_views",
    "views": [
      {"id":"front", "camera":"/dev/video1", "video":{"path":"/dice-front/"}},
      {"id":"side", "camera":"/dev/video2", "video":{"path":"/dice-side/"}}
    ]
  }
}
```

- [ ] **步骤 2：接线多路 WebRTC 事件**

每个 view 的 `video.path` 与同一个 `mediamtx.webrtc_base_url` 合成 URL；SSE 可以发送多个 `video` 事件并携带 `view_id`。前端按 view_id 保存 iframe/画面槽位；单路 profile 仍使用现有单 iframe。

- [ ] **步骤 3：补充 health 和 games API**

`/api/health` 返回 `vision_yolov8_adjudicator`、`mode`、`prewarm`、`profiles`、`mediamtx_base_url` 和 `multi_view` 能力；`/api/games` 返回公开的 profile video enabled/path，不返回 LLM key、prompt 全文或本地临时路径。

- [ ] **步骤 4：运行完整 Python 测试并提交**

运行：`python3 -m pytest -q`

预期：所有 Python 测试通过，旧 TTS 测试不受影响；失败时先修复新视觉代码或更新明确过时的断言，不回退用户 TTS 修改。

提交：`git add backend/games backend/components/vision_yolov8_adjudicator backend/server.py backend/core/games.py tests && git commit -m "feat: wire multiview profiles and health"`

---

## 任务 9：清理旧组件、更新文档并做迁移检查

**文件：**

- 删除：`backend/components/vision_yolo/`
- 修改：`backend/core/components.py`
- 修改：`README.md`
- 修改：`backend/components/README.md`
- 修改：`AI_PROJECT_CONTEXT.md`
- 修改：`FRAMEWORK_DISPATCH.md`
- 测试：`tests/test_components_and_jobs.py`

- [ ] **步骤 1：保留并测试旧 ID 兼容别名**

在 registry 解析时将 `vision_yolo` 映射到 `vision_yolov8_adjudicator`，只输出一次迁移日志；新 manifest、pipeline、health 和文档全部使用新 ID。测试旧 manifest 能解析到新 provider。

- [ ] **步骤 2：删除旧 package 和骰子专用调用路径**

确认 `rg -n "vision_yolo|DiceYoloAdjudicator|llm_dice|left_sum|right_sum" backend web vision` 只剩迁移映射、历史说明或 C++ 兼容注释；删除旧 Python package。保留 C++ 构建、自测和硬件诊断选项，不保留面向网页的 CLI。

- [ ] **步骤 3：同步文档**

文档必须准确写明：profile 路径、组件 base URL、游戏 video path、MediaMTX 外部接管 RTSP、iframe 播放、无状态图片 prompt、多数投票、LLM 超时回退、holding、prewarm/control-fd 和 K3 验收命令。删除固定骰子 IP、旧组件职责和“C++ 直接做骰子胜负”的描述。

- [ ] **步骤 4：运行迁移检查并提交**

运行：

```bash
python3 -m pytest -q
git diff --check
rg -n "TODO|待定|未完成|后续实现" docs/superpowers/specs backend/components/vision_yolov8_adjudicator backend/games/dice/vision_profile.json
```

预期：pytest 全部通过、diff check 无输出、占位符搜索无输出。提交：`git add -A backend README.md backend/components/README.md AI_PROJECT_CONTEXT.md FRAMEWORK_DISPATCH.md tests && git commit -m "refactor: migrate vision adjudicator package"`

---

## 任务 10：K3 实机验收和最终验证

**文件：**

- 测试：K3 `/home/spacemit/projects/dice-game/main`
- 记录：`docs/superpowers/specs/2026-08-29-vision-yolov8-adjudicator-k3-validation.md`

- [ ] **步骤 1：检查板端工作区和凭据边界**

运行：`ssh spacemit@spacemit-k3 'cd /home/spacemit/projects/dice-game/main && git status --short && test -f .dice-arena.env'`

不打印 `.dice-arena.env` 内容，不覆盖板端 TTS 或 LLM 配置。

- [ ] **步骤 2：编译并做 runtime help/self-test**

运行：

```bash
ssh spacemit@spacemit-k3 'cd /home/spacemit/projects/dice-game/main/vision/yolov8_objdetect && cmake --build build -j4 && ./build/yolov8_camera --help && ./build/yolov8_camera --self-test --no-display'
```

预期：build 和 self-test exit 0；记录实际模型、OpenCL、SpaceMIT EP 日志。

- [ ] **步骤 3：运行后端测试和受控裁决**

运行：

```bash
ssh spacemit@spacemit-k3 'cd /home/spacemit/projects/dice-game/main && python3 -m pytest -q'
ssh spacemit@spacemit-k3 'curl -fsS http://127.0.0.1:8080/api/health'
```

在已配置 LLM key、摄像头和 MediaMTX 的条件下，执行一次按局裁决和一次 `post_result_hold_seconds=3`；记录点击到结果、结果到 complete 的耗时、SSE video/result/holding/complete 事件、进程数和摄像头释放。

- [ ] **步骤 4：验证多视角和取消**

启用测试 profile 的两路 camera，在 K3 观察多数投票、单次多图 LLM、LLM timeout fallback、holding 取消和下一局复用；确认无孤儿进程、无摄像头抢占和无上一局 evidence 泄漏。

- [ ] **步骤 5：运行最终本地验证并提交验收记录**

运行：`python3 -m pytest -q && git diff --check && git status --short`

验收记录只写命令结果、耗时和错误，不写 LLM key。提交：`git add docs/superpowers/specs/2026-08-29-vision-yolov8-adjudicator-k3-validation.md && git commit -m "test: record vision adjudicator k3 validation"`

---

## 计划自检

- 规格中的 MediaMTX 约束由任务 1、7、8、9 覆盖：base URL 在组件配置，game path 在 profile，前端只接收合成的 WebRTC URL，不管理 RTSP。
- 规格中的无状态图片 LLM、LLM mismatch/timeout 策略由任务 2、3、4、6 覆盖；每局只调用一次。
- 规格中的 `holding`、SSE 和前端实时画面由任务 4、5、7 覆盖。
- 规格中的按局/常驻、预热和控制通道由任务 4、6、10 覆盖。
- 规格中的多摄像头并行、多数投票、多图单次 LLM 由任务 2、4、8、10 覆盖。
- 规格中的 C++ dice 解耦、统一 observation 和 snapshot 由任务 6、9 覆盖。
- 没有使用 TODO、待定、未完成或“补充细节”作为实现占位；每项都给出文件、接口、测试命令和提交点。
