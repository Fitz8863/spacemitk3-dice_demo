# 项目目录与文档整理实施计划

> **面向 AI 代理的工作者：** 本计划用于当前项目的目录、文档和 Git 交付整理。

**目标：** 让当前仓库的有效入口、组件目录、视觉 runtime 和历史资料边界清晰，并安全推送到当前功能分支。

**架构：** 保留代码与运行时目录的现有职责；把已完成迁移说明和旧 TTS 调用手册归档到 `docs/archive/`，用 `docs/README.md` 统一索引。通过根 `.gitignore` 约束构建物、缓存和板端生成文件，避免把用户本地密钥配置或音频带入提交。

**技术栈：** Python HTTP bridge、YOLOv8 C++ runtime、TTS provider packages、Markdown、Git。

---

### 任务 1：盘点并保护用户本地文件

**文件：**
- 检查：`.gitignore`、Git 状态、组件和游戏目录
- 不修改：`backend/components/vision_yolov8_adjudicator/config.json`、`backend/games/dice/audio/fll.wav`

- [x] 确认当前分支和远程地址。
- [x] 确认用户配置和音频只保留在工作区，不进入索引。

### 任务 2：整理历史资料与忽略规则

**文件：**
- 移动：`CosyVoice-TTS-调用说明.md` → `docs/archive/legacy/CosyVoice-TTS-调用说明.md`
- 移动：`vision/yolov8_adjudicator/README_MIGRATION.md` → `docs/archive/vision/README_MIGRATION.md`
- 修改：`.gitignore`
- 创建：`docs/README.md`

- [x] 统一 `yolov8_adjudicator` 路径和生成目录忽略规则。
- [x] 为当前文档、历史计划和归档资料建立索引。

### 任务 3：更新当前说明

**文件：**
- 修改：`README.md`
- 修改：`AI_PROJECT_CONTEXT.md`
- 修改：`CLAUDE.md`
- 重写：`FRAMEWORK_DISPATCH.md`
- 修改：`backend/components/README.md`
- 修改：`vision/yolov8_adjudicator/README.md`

- [x] 移除旧目录、外置 `vision_profile.json` 和错误的默认 provider 描述。
- [x] 说明 manifest 内嵌 `vision_profile`、MediaMTX 地址归属、resident 生命周期和未来 localizer 边界。

### 任务 4：验证、提交与推送

**命令：**

```bash
python3 -m pytest -q tests
python3 -m compileall -q backend
node --check web/app.js
node --check web/games/dice.js
git diff --check
git status --short
git push -u origin codex/vision-yolov8-adjudicator
```

- [x] 验证测试和语法检查退出码为 0；K3 只读检查因 SSH 凭据和 HTTP 502 阻断。
- [x] 只暂存本计划列出的整理文件，不使用 `git add .`。
- [ ] 推送后用 `git ls-remote` 核对远程分支提交。
