# 视觉 Runtime 配置归属收敛实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 YOLOv8 视觉裁决器的硬件、推理和 RTSP 运行参数收敛到 `vision/yolov8_adjudicator/config.json`，让 backend 组件配置只描述 Provider 部署与 LLM，并保持游戏 manifest 只覆盖游戏规则和 WebRTC 路径。

**架构：** C++ runtime 配置是硬件/推理/RTSP 的唯一默认来源，Provider 启动时显式传递 `--config`。backend 组件配置通过 `runtime.config` 定位该文件，并保留旧 `rtsp`/`video` 字段的读取回退以兼容未迁移部署；游戏配置中的 `video.path` 仍是每个游戏或视角的唯一流路径，MediaMTX 基础地址由 runtime 配置统一提供并可由环境变量覆盖。

**技术栈：** Python 3 标准库、JSON 配置、现有 unittest/pytest 测试、C++ OpenCV runtime、MediaMTX WebRTC。

---

### 任务 1：锁定配置归属与兼容读取行为

**文件：**
- 修改：`tests/test_vision_adjudicator.py`
- 修改：`backend/components/vision_yolov8_adjudicator/profile.py`

- [ ] **步骤 1：编写失败测试**

增加以下行为断言：组件配置的 `runtime.config` 能解析到仓库内 runtime 配置；加载后的 runtime 配置包含硬件/RTSP 默认值；新的组件配置不再声明 `rtsp` 或 `video`；旧组件配置仍可在显式提供旧字段时被读取。

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest -q tests/test_vision_adjudicator.py -k 'runtime_config or component_config'`

预期：FAIL，报错来自尚未提供的 runtime 配置加载函数或新配置字段。

- [ ] **步骤 3：编写最少实现代码**

在 `profile.py` 增加 `resolve_runtime_config_path()` 和 `load_runtime_config()`；校验 `runtime.config` 是仓库内相对路径并读取 JSON 对象；`load_component_config()` 校验该路径，同时只对旧的 `video` 字段做兼容校验。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest -q tests/test_vision_adjudicator.py -k 'runtime_config or component_config'`

预期：新增断言和现有组件配置校验测试全部 PASS。

- [ ] **步骤 5：提交**

```bash
git add tests/test_vision_adjudicator.py backend/components/vision_yolov8_adjudicator/profile.py
git commit -m "test: define consolidated vision runtime config"
```

### 任务 2：让 runtime 启动显式使用唯一配置

**文件：**
- 修改：`tests/test_vision_adjudicator.py`
- 修改：`backend/components/vision_yolov8_adjudicator/process.py`
- 修改：`vision/yolov8_adjudicator/config.json`
- 修改：`backend/components/vision_yolov8_adjudicator/config.json`

- [ ] **步骤 1：编写失败测试**

增加启动命令测试，断言 `YoloRuntimeProcess` 传入的命令包含 `--config <仓库>/vision/yolov8_adjudicator/config.json`；增加 RTSP 测试，断言默认值来自 runtime 配置而不是 component 配置。

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest -q tests/test_vision_adjudicator.py -k 'runtime_command or rtsp'`

预期：FAIL，当前命令没有 `--config` 且从组件配置读取 RTSP。

- [ ] **步骤 3：编写最少实现代码**

在 `process.py` 启动时解析并加载 `runtime.config`，把绝对路径加入 C++ 命令；从 runtime 配置读取 RTSP，profile 的视角 path 只覆盖 mount path；旧 component `rtsp` 仅作为兼容回退。保留游戏 profile 的模型、稳定帧、置信度和 divider CLI 覆盖。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest -q tests/test_vision_adjudicator.py -k 'runtime_command or rtsp'`，随后运行完整 `pytest -q tests/test_vision_adjudicator.py`。

预期：全部 PASS，且现有注入 fake runtime 测试不受影响。

- [ ] **步骤 5：提交**

```bash
git add tests/test_vision_adjudicator.py backend/components/vision_yolov8_adjudicator/process.py vision/yolov8_adjudicator/config.json backend/components/vision_yolov8_adjudicator/config.json
git commit -m "refactor: centralize yolov8 runtime defaults"
```

### 任务 3：统一 WebRTC 基础地址读取并同步文档

**文件：**
- 修改：`tests/test_vision_adjudicator.py`
- 修改：`backend/components/vision_yolov8_adjudicator/profile.py`
- 修改：`backend/components/vision_yolov8_adjudicator/provider.py`
- 修改：`backend/server.py`
- 修改：`vision/yolov8_adjudicator/README.md`
- 修改：`README.md`
- 修改：`docs/superpowers/specs/2026-08-29-vision-yolov8-adjudicator-design.md`

- [ ] **步骤 1：编写失败测试**

断言没有 `video.webrtc_base_url` 的游戏 profile 仍能由 runtime 配置生成 URL，环境变量仍具有最高优先级，旧 component `video.webrtc_base_url` 仍能作为迁移回退。

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest -q tests/test_vision_adjudicator.py -k 'video_event or compose_video'`

预期：至少一项因 provider/server 仍只读取组件或 profile 的旧位置而 FAIL。

- [ ] **步骤 3：编写最少实现代码**

新增统一的 runtime video 配置读取辅助函数；Provider 和 server 采用 `DICE_MEDIAMTX_WEBRTC_BASE_URL > game video.webrtc_base_url (legacy) > runtime video.webrtc_base_url > component video.webrtc_base_url (legacy)` 优先级。更新文档说明配置边界、迁移方式和环境变量覆盖。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest -q tests/test_vision_adjudicator.py tests/test_server_api.py`、`node --check web/app.js`、`node --check web/games/dice.js`、`git diff --check`。

- [ ] **步骤 5：提交**

```bash
git add tests/test_vision_adjudicator.py backend/components/vision_yolov8_adjudicator/profile.py backend/components/vision_yolov8_adjudicator/provider.py backend/server.py vision/yolov8_adjudicator/README.md README.md docs/superpowers/specs/2026-08-29-vision-yolov8-adjudicator-design.md
git commit -m "docs: document consolidated vision video configuration"
```

### 任务 4：全量验证与 K3 烟囱测试

**文件：** 无新增文件；只检查前述变更。

- [ ] **步骤 1：本地全量验证**

运行：`pytest -q tests`、`python3 -m compileall -q backend`、`node --check web/app.js`、`node --check web/games/dice.js`、`git diff --check`。

- [ ] **步骤 2：检查 Git 边界**

确认工作区只剩用户原有的 `backend/components/vision_yolov8_adjudicator/config.json` 和 `backend/games/dice/audio/fll.wav` 未提交内容，且不输出其中的密钥或音频内容。

- [ ] **步骤 3：K3 验证**

使用 `SSHPASS='bianbu' sshpass -e ssh spacemit@spacemit-k3` 在 `/home/spacemit/projects/dice-game/main` 检查服务单实例、`/api/health`、`/api/games`、Provider 启动命令包含显式 runtime config、WebRTC URL 使用 `/dice/` path；不在设备上启动第二个 8080 服务，不进行真实云端调用或修改用户运行数据。

- [ ] **步骤 4：最终提交**

完成验证后提交剩余实现和文档变更，并报告本地与 K3 的实际输出。
