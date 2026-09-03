# 游戏 Manifest 统一视觉配置实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将游戏视觉裁决 profile、WebRTC 基础地址和裁决总超时迁移到游戏 `manifest.json`，同时保持视觉 provider 按 role 解耦。

**架构：** 游戏 manifest 内嵌 `vision_profile`，裁决器只读取该节点；`VisionProvider` 仍是总接口，adjudicator 与未来 localizer 使用不同 role 和配置槽位。组件 `config.json` 只保存部署级 runtime、RTSP 和 LLM 传输配置。

**技术栈：** Python 3、JSON 配置、pytest、现有 `ComponentRegistry`/`GameRegistry` 和 YOLOv8 control-fd runtime。

---

### 任务 1：锁定配置迁移契约

**文件：**
- 修改：`tests/test_vision_adjudicator.py`
- 修改：`tests/test_components_and_jobs.py`

- [ ] **步骤 1：编写失败测试**

添加测试，断言 profile 必须校验 `video.webrtc_base_url` 和 `timeouts.adjudication_seconds`；添加测试，断言 `load_games()` 在 manifest 内嵌 `vision_profile` 时优先使用它；添加 provider 视频事件测试，断言 URL 来自 profile 而不是组件 config。

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest tests/test_vision_adjudicator.py tests/test_components_and_jobs.py -q
```

预期：新增断言失败，原因是当前 profile 仍从组件 config 获取基础地址，且游戏加载器只读取外部 `vision_profile.json`。

### 任务 2：迁移游戏加载器和骰子 manifest

**文件：**
- 修改：`backend/core/games.py`
- 修改：`backend/games/dice/manifest.json`
- 修改：`backend/components/vision_yolov8_adjudicator/profile.py`
- 删除：`backend/games/dice/vision_profile.json`

- [ ] **步骤 1：实现内嵌 profile 优先和外部兼容回退**

在 `load_games()` 中先读取并校验 `manifest["vision_profile"]`；缺少内嵌对象时再读取同目录 `vision_profile.json`；校验 profile 的 `game_id` 与 manifest `id` 一致，并将规范化 profile 写回 `manifest["vision_profile"]`。

- [ ] **步骤 2：扩展 profile 校验**

在 `validate_profile()` 中校验 `video.webrtc_base_url` 为无路径 HTTP(S) URL，并校验 `timeouts.adjudication_seconds` 为有限正数；保留 `video.path`、stable frames、多视角和角色无关的通用规则校验。

- [ ] **步骤 3：迁移骰子配置**

将现有 `vision_profile.json` 内容作为 `manifest.json.vision_profile` 写入，并把视频基础地址设置为 `video.webrtc_base_url`；将总超时设置为 `timeouts.adjudication_seconds`。

- [ ] **步骤 4：运行配置测试**

运行：

```bash
python3 -m pytest tests/test_vision_adjudicator.py tests/test_components_and_jobs.py -q
```

预期：配置加载、校验和 manifest 优先级测试通过。

### 任务 3：切换 provider、健康检查和运行时超时来源

**文件：**
- 修改：`backend/components/vision_yolov8_adjudicator/provider.py`
- 修改：`backend/server.py`
- 修改：`backend/components/vision_yolov8_adjudicator/config.json`
- 修改：`tests/test_server_api.py`

- [ ] **步骤 1：实现 profile 视频地址解析**

让 `_video_event()` 从 `profile["video"]["webrtc_base_url"]` 读取基础地址；保留 `DICE_MEDIAMTX_WEBRTC_BASE_URL` 作为显式运维覆盖；多视角只覆盖 path。

- [ ] **步骤 2：实现游戏级总超时**

在 provider 开始裁决时读取 `profile["timeouts"]["adjudication_seconds"]`，没有该值时回退到传入的 `timeout_seconds`。用一个整轮 deadline 约束稳定帧等待和 LLM 请求剩余时间；裁决成功后的 `lifecycle.post_result_hold_seconds` 独立执行，不占用该处理预算。所有 LLM 请求仍不能超过统一的 `llm.timeout_seconds`。

- [ ] **步骤 3：清理组件 MediaMTX 配置**

从视觉组件 `config.json` 删除 `mediamtx` 块；保留 runtime、RTSP、LLM 和 events。健康检查改从 profile 生成 `video_url`，兼容返回字段名 `mediamtx_base_url` 但来源改为 profile。

- [ ] **步骤 4：运行 provider/server 测试**

运行：

```bash
python3 -m pytest tests/test_vision_adjudicator.py tests/test_server_api.py -q
```

预期：视频 URL、总超时、LLM 超时、holding 和 health 测试全部通过。

### 任务 4：更新文档和迁移检查

**文件：**
- 修改：`README.md`
- 修改：`FRAMEWORK_DISPATCH.md`
- 修改：`vision/yolov8_objdetect/README.md`
- 修改：`tests/test_yolov8_runtime_docs.py`

- [ ] **步骤 1：更新配置说明**

明确游戏 manifest 的 `vision_profile.video.webrtc_base_url`、`video.path` 和 `timeouts.adjudication_seconds`，并说明组件 config 不再保存 MediaMTX 基础地址。

- [ ] **步骤 2：运行文档一致性检查**

运行：

```bash
python3 -m pytest tests/test_yolov8_runtime_docs.py -q
git diff --check
```

预期：2 个文档测试通过且无空白错误。

### 任务 5：本地与 K3 验收

**文件：**
- 不新增文件

- [ ] **步骤 1：运行本地全量测试**

运行：`python3 -m pytest tests -q`；预期全部通过。

- [ ] **步骤 2：提交迁移变更**

运行：

```bash
git add backend/core/games.py backend/games/dice/manifest.json backend/components/vision_yolov8_adjudicator/profile.py backend/components/vision_yolov8_adjudicator/provider.py backend/components/vision_yolov8_adjudicator/config.json backend/server.py tests README.md FRAMEWORK_DISPATCH.md vision/yolov8_objdetect/README.md docs/superpowers/specs/2026-08-30-game-manifest-vision-config-design.md docs/superpowers/plans/2026-08-30-game-manifest-vision-config.md
git commit -m "refactor: embed vision adjudication in game manifest"
```

- [ ] **步骤 3：在 K3 验收**

运行 `scripts/start_web.sh`、`curl http://127.0.0.1:8080/api/health`、一次 `/api/adjudicate`、查询事件直到结果或超时，再运行 `scripts/stop_web.sh`。确认健康检查中的视频 URL 来自 manifest，任务使用 profile 总超时，停止后只保留 MediaMTX。
