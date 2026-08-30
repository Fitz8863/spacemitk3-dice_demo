# YOLOv8 Adjudicator Runtime 重命名与黑线辅助实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 C++ runtime 迁移到 `vision/yolov8_adjudicator`，清理无用文件，并让常驻 runtime 在开始裁决时动态启用 YOLO 与黑线辅助检测。

**架构：** 摄像头/RTSP 生命周期与 YOLO 推理生命周期分离；control-fd 负责在同一进程内切换 idle/active。黑线检测作为可配置的帧级几何辅助信息执行，不承担游戏胜负规则。

**技术栈：** C++17、OpenCV、OpenCL、SpaceMIT ONNX Runtime、GStreamer、CMake、Python pytest。

---

### 任务 1：建立黑线辅助行为的失败测试

**文件：**
- 修改：`tests/test_vision_adjudicator.py`
- 修改：`tests/test_yolov8_generic_build_boundary.py`

- [ ] **步骤 1：添加测试**

添加静态源码测试，断言 generic control-fd 路径包含 `detect_black_divider` 调用、
observation 输出 `divider`，并断言 `yolov8_enabled=false` 且存在 control-fd 时不会提前
进入 camera-only return。添加配置测试，断言 dice profile 的
`vision.divider_detection` 为 true。

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
python3 -m pytest tests/test_vision_adjudicator.py tests/test_yolov8_generic_build_boundary.py -q
```

预期：新增断言失败，因为当前 generic 分支跳过 `judge_dice`，且 false 配置会提前返回。

---

### 任务 2：迁移 runtime 目录并清理文件

**文件：**
- 重命名目录：`vision/yolov8_objdetect/` -> `vision/yolov8_adjudicator/`
- 删除：新目录下 `build/`、`.shaders/`、`src/llm_dice_verifier.{h,cpp}`
- 修改：`backend/components/vision_yolov8_adjudicator/config.json`
- 修改：`backend/components/vision_yolov8_adjudicator/process.py`

- [ ] **步骤 1：执行目录迁移**

使用 `git mv vision/yolov8_objdetect vision/yolov8_adjudicator`，再删除生成目录和 generic
构建不需要的骰子 LLM 源码；保留 CMake、模型、摄像头、OpenCL、检测器和 RTSP 源码。

- [ ] **步骤 2：更新 runtime 路径**

将组件配置和 process 默认解析路径改为 `vision/yolov8_adjudicator`，保持可执行文件名
`build/yolov8_camera` 不变。

- [ ] **步骤 3：运行路径检查**

运行：

```bash
rg -n "vision/yolov8_objdetect" backend vision tests README.md FRAMEWORK_DISPATCH.md AI_PROJECT_CONTEXT.md CLAUDE.md
```

预期：无正式运行时引用；历史设计文档中的旧路径需标注为历史迁移内容或同步更新。

---

### 任务 3：修复动态启用 YOLO 和黑线检测

**文件：**
- 修改：`vision/yolov8_adjudicator/src/main.cpp`
- 修改：`vision/yolov8_adjudicator/config.json`
- 修改：`backend/components/vision_yolov8_adjudicator/process.py`
- 修改：`backend/components/vision_yolov8_adjudicator/provider.py`

- [ ] **步骤 1：扩展配置开关**

增加 `Args::divider_detection_enabled` 和 JSON `vision.divider_detection` 的读取；
provider 从游戏 profile 的 `vision.divider_detection` 传入 `--divider-detection` 或
`--no-divider-detection`。

- [ ] **步骤 2：保留 control-fd 的 YOLO 会话**

把 detector/preprocessor 初始化条件改为 `a.yolov8_enabled || a.control_fd >= 0`；
camera-only early return 只允许在没有 control-fd 时执行。active 状态仍由 START/STOP
控制，idle 不执行 preprocess/infer。

- [ ] **步骤 3：接入黑线辅助**

generic active 帧调用 `detect_black_divider`，绘制 `draw_divider_and_judgment` 所需的
分界线，稳定候选在开关开启时要求 `divider.valid`，并扩展 `emit_observation` 输出 divider
点、方向和法向量；不调用骰子求和或 C++ LLM。

- [ ] **步骤 4：运行 targeted 测试和 C++ 静态检查**

运行：

```bash
python3 -m pytest tests/test_vision_adjudicator.py tests/test_yolov8_generic_build_boundary.py -q
git diff --check
```

---

### 任务 4：更新游戏配置、后端引用和文档

**文件：**
- 修改：`backend/games/dice/manifest.json`
- 修改：`backend/games/rps/manifest.json`
- 修改：`README.md`
- 修改：`FRAMEWORK_DISPATCH.md`
- 修改：`AI_PROJECT_CONTEXT.md`
- 修改：`CLAUDE.md`
- 修改：`vision/yolov8_adjudicator/README.md`
- 修改：`vision/yolov8_adjudicator/README_MIGRATION.md`
- 修改：`tests/test_yolov8_runtime_docs.py`

- [ ] **步骤 1：配置骰子 profile**

在 dice 的 `vision_profile.vision` 增加 `divider_detection: true`；RPS 明确设置为 false
或省略并由 runtime 默认关闭。

- [ ] **步骤 2：同步所有正式路径**

将构建、运行、模型、配置和 README 示例全部改为 `vision/yolov8_adjudicator`；
不得修改 Python provider ID 或 `yolov8_camera` 可执行文件名。

- [ ] **步骤 3：文档一致性测试**

运行：

```bash
python3 -m pytest tests/test_yolov8_runtime_docs.py -q
```

---

### 任务 5：完整验证、Git 提交和 K3 验收

**文件：**
- 不新增文件

- [ ] **步骤 1：本地回归**

运行 `python3 -m pytest tests -q`、`python3 -m compileall -q backend tests` 和
`git diff --check`；预期所有测试通过。

- [ ] **步骤 2：提交前审查并提交**

确认不包含 `backend/games/dice/audio/fll.wav`，再提交：

```bash
git add -u
git commit -m "refactor: rename yolov8 adjudicator runtime"
```

- [ ] **步骤 3：K3 构建与 runtime 验收**

在 `/home/spacemit/projects/dice-game/main` 执行：

```bash
cmake -S vision/yolov8_adjudicator -B vision/yolov8_adjudicator/build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build vision/yolov8_adjudicator/build -j4
vision/yolov8_adjudicator/build/yolov8_camera --config vision/yolov8_adjudicator/config.json --self-test --no-display --no-rtsp
```

再通过 `scripts/start_web.sh` 和 `/api/adjudicate` 验证 control-fd 进入 detecting、发送
WebRTC 视频事件、输出 YOLO 推理日志，最后 `scripts/stop_web.sh` 确认只保留 MediaMTX。
