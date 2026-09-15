# 多游戏视觉裁决架构改造实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpower:subagent-driven-development`（推荐）或
> `superpower:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**存档点：** `pre-multigame-vision`（动手前已打 tag 并推送）。回退：`git reset --hard pre-multigame-vision`。

**目标：** 让同一套视觉裁决器服务多个游戏，每个游戏独立配置自己的**裁决参数**（class_map / 规则 /
稳定帧 / 置信度 / 分组 / RTSP 路径 / prompt —— 这些已可由 manifest 表达），并修掉会**跨游戏串味**
的运行时缺陷。本次**不构建**石头剪刀布（rps）游戏本体。

**架构：** 视觉裁决器的「硬件/部署」与「游戏裁决参数」两条边界保持不变：硬件项留在
`vision/yolov8_adjudicator/config.json`，游戏项留在各游戏 manifest 的 `vision_profile`。
本次改造只补三处缺口：① resident runtime 的**重启判据（签名）不完整**，导致参数不同的两个游戏
可能复用同一个进程；② 裁决流程与结果投影**硬编码在 dice 里**，第二个游戏无法复用；
③ `/api/health` **硬编码 `"dice"`**。

**技术栈：** Python 3 标准库、JSON 配置、现有 pytest 套件、C++ OpenCV runtime（本次**不改 C++**）、
MediaMTX / RTSP。

---

## 背景：为什么需要这次改造

现状核实（均有代码证据）：

| 事实 | 位置 |
|---|---|
| rps **已有完整 `vision_profile`**，但目录下只有 `manifest.json`，**没有 `pipeline.py`** → `importlib.import_module("games.rps.pipeline")` 会 ModuleNotFoundError | `backend/core/games.py:317` |
| dice pipeline **硬编码 `GAME_ID = "dice"`** 并拒绝其他 profile | `backend/games/dice/pipeline.py` |
| `project_participant_result` **硬编码 1–6 骰子整数校验** → 猜拳的字符串类别必然 ValueError | `backend/games/dice/result.py` |
| **★ 核心缺陷**：`_runtime_cache` 键是 `view_id`（两游戏都是 `"default"`），而 `_runtime_signature` 只含 `model`/`stable_frames`/`confidence`/`camera`/`runtime`，**缺 `game_id`、`divider_detection`、`expected_count`、`grouping`、`video.path`** | `backend/components/vision_yolov8_adjudicator/provider.py` |
| `/api/health` 四处硬编码 `"dice"` | `backend/server.py:1089-1105` |
| 视觉槽位**没有**像 ASR / 本地 TTS 那样的唯一性封口 | `backend/server.py:277,1470-1485` |
| rps 状态机只有 1 态、前端 `rps.js` 是占位 → 本次不动 | `backend/games/rps/manifest.json`、`web/games/rps.js` |

**结论：** 不需要 per-game 的 runtime config 文件；缺的是**签名完整性 + 通用 pipeline + 通用结果投影**。

## 关键设计决定

1. **不改 `_runtime_cache` 的键**。继续以 `view_id` 为键、签名变了就拆旧的建新的。
   理由：两个游戏**共用同一个摄像头设备**（`/dev/video1`），若允许两个 runtime 共存会**抢同一个
   V4L2 设备**而失败。「一个 view_id 一个 runtime」是共用摄像头场景下唯一正确的选择。
   **代价（写进文档）**：切换游戏必然重建 runtime（付一次摄像头打开 + 模型加载），这是固有成本。
2. **每个要跑的游戏自带一个薄壳 pipeline**，与 `run_game` 的 `games.<id>.pipeline` 约定一致。
3. **dice 的对外行为必须逐字不变**——`run()` 签名、`project_participant_result()` 签名与返回、
   以及 `tests/test_participants.py` 的全部断言一行不改即为验收标准。

---

### 任务 0：存档（**动手前第一件事**）

**文件：**
- 新增：`docs/superpowers/plans/2026-09-15-multi-game-vision-architecture.md`（本文件）

- [ ] **步骤 1：确认工作区干净**

运行：`git status --short`（应为空）、`git rev-list --count origin/main..HEAD`（应为 9）。

- [ ] **步骤 2：提交本计划书**

```bash
git add docs/superpowers/plans/2026-09-15-multi-game-vision-architecture.md
git commit -m "docs(plan): 多游戏视觉裁决架构改造计划书（动手前存档点）"
```

- [ ] **步骤 3：打注释 tag**

```bash
git tag -a pre-multigame-vision -m "多游戏视觉架构改造前的存档点（回退: git reset --hard pre-multigame-vision）"
```

- [ ] **步骤 4：推送**

```bash
git push origin main
git push origin pre-multigame-vision
```

预期：`origin/main` 与本地一致，tag 在远端可见。

---

### 任务 1：复现跨游戏串味（失败的回归测试）

**文件：**
- 新增：`tests/test_multi_game_vision.py`

- [ ] **步骤 1：编写失败测试**

构造**两个游戏**，`model`/`stable_frames`/`confidence`/`camera` **完全相同**，仅
`game_id`、`divider_detection`、`expected_count`、`video.path` 不同。先按游戏 A 裁决，再按游戏 B 裁决。

断言：
- B 的 runtime **是新建的**（`runtime_factory` 被调用 2 次，A 的 runtime 收到 `stop()`）；
- B 启动的命令行带 **B 自己的** `--divider-detection` / `--expected-count` / `--rtsp-path`。

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest -q tests/test_multi_game_vision.py`

预期：FAIL —— 今天 B 会**复用 A 的 runtime**（签名相同），因此 B 的 `runtime_factory` 只被调用一次、
命令行仍是 A 的参数。**这条失败就是本阶段的核心证据。**

---

### 任务 2：补全 `_runtime_signature`

**文件：**
- 修改：`backend/components/vision_yolov8_adjudicator/provider.py`
- 修改：`tests/test_multi_game_vision.py`

- [ ] **步骤 1：补齐签名 payload**

在 `_runtime_signature(profile, view_id)` 的 payload 中增加下列键（保留原有 5 项）：

| 新增 | 取值 | 为何影响进程行为 |
|---|---|---|
| `game_id` | `profile.get("game_id")` | 游戏身份；`validate_profile` 强制存在且 `load_games` 校验与 manifest id 一致 |
| `divider_detection` | `vision.get("divider_detection")` | 转发成 `--divider-detection`，改变门控 |
| `expected_count` | `vision.get("expected_count")` | 转发成 `--expected-count` |
| `region_position` / `region_orientation` | 复用 `process.region_split(vision)` 的解析结果 | 转发成 `--region-position` / `--region-orientation` |
| `video_path` | `profile.get("video", {}).get("path")` | 转发成 `--rtsp-path`，决定推流挂载点 |

注意：`region_split` 在 `process.py`，`provider.py` 引入它时避免循环导入（`process` 已 import
`profile`，`provider` 已 import `process` 的类型；若不便，则在 provider 内联同样的解析逻辑并加注释）。

- [ ] **步骤 2：运行测试验证通过**

运行：`pytest -q tests/test_multi_game_vision.py tests/test_vision_adjudicator.py`

预期：新的串味用例 PASS，且既有 97 项 vision 测试不退化。

- [ ] **步骤 3：提交**

```bash
git add backend/components/vision_yolov8_adjudicator/provider.py tests/test_multi_game_vision.py
git commit -m "fix(vision): 补全 runtime 签名，消除跨游戏裁决参数串味"
```

---

### 任务 3：把 runtime config 文件内容纳入签名

**文件：**
- 修改：`backend/components/vision_yolov8_adjudicator/provider.py`
- 修改：`tests/test_multi_game_vision.py`

- [ ] **步骤 1：编写失败测试**

断言：修改 runtime config 文件（touch 或改内容）后，下一次 `_ensure_runtime` **重建** runtime。

- [ ] **步骤 2：实现**

把 `resolve_runtime_config_path()` 解析出的路径的 `stat()`（`st_mtime_ns` + `st_size`）纳入签名
payload；**取不到时降级为 `None`**，绝不因读不到文件而拒绝启动。

- [ ] **步骤 3：运行测试**

运行：`pytest -q tests/test_multi_game_vision.py`

- [ ] **步骤 4：提交**

```bash
git add backend/components/vision_yolov8_adjudicator/provider.py tests/test_multi_game_vision.py
git commit -m "fix(vision): runtime 配置文件变化即重建 resident runtime"
```

> **行为变化提醒（必须写进提交信息与文档）**：此前热改 `vision/yolov8_adjudicator/config.json`
> 的 `conf`/`zoom`/`focus` **不生效、需重启服务**；此后**下一回合自动生效**，与
> `vision_always_on` 每回合现读的语义一致。

---

### 任务 4：抽出通用视觉 pipeline

**文件：**
- 新增：`backend/core/vision_pipeline.py`
- 修改：`backend/games/dice/pipeline.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/test_multi_game_vision.py` 增加：一个**非 dice** 的游戏（合成 manifest + 假 provider）
走 `run_game`，断言能正常裁决并投影，且**不需要 dice 的 pipeline**。

- [ ] **步骤 2：实现 `core/vision_pipeline.py`**

```python
def run_vision_game(on_log, is_cancelled, timeout_seconds, *, components, manifest,
                    on_event, game_id: str, projector) -> dict:
    """Resolve the adjudicator slot, run one bounded adjudication, project it."""
```

职责：解析 `vision_adjudicator` 槽位 → 校验 `vision_profile` 是对象且
`profile["game_id"] == game_id` → 解析 `llm` 槽位（缺失/坏槽位**软降级**为 YOLO-only，保持现有语义）
→ 构造 `VisionAdjudicationRequest` → 调 `adjudicate()`（**保留现有 `except TypeError` 的旧接口
兼容分支**）→ 若 `diagnosed` 则直通 → 否则交 `projector` 投影。

- [ ] **步骤 3：`games/dice/pipeline.py` 变薄壳**

```python
GAME_ID = "dice"

def run(on_log, is_cancelled, timeout_seconds, *, components, manifest, on_event):
    return run_vision_game(..., game_id=GAME_ID, projector=project_participant_result)
```

**必须保持 `run()` 的签名与模块路径不变**——`tests/test_components_and_jobs.py` 与
`tests/test_vision_adjudicator.py` 都 `import games.dice.pipeline as dice_pipeline` 并直接调用。
`core/games.py` 的 `importlib.import_module(f"games.{game_id}.pipeline")` **不动**。

- [ ] **步骤 4：运行测试**

运行：`pytest -q tests/test_vision_adjudicator.py tests/test_components_and_jobs.py tests/test_multi_game_vision.py`

- [ ] **步骤 5：提交**

```bash
git add backend/core/vision_pipeline.py backend/games/dice/pipeline.py tests/test_multi_game_vision.py
git commit -m "refactor(vision): 抽出游戏无关的裁决 pipeline，dice 退化为薄壳"
```

---

### 任务 5：结果投影通用化（角色投影 + 分类投影）

**文件：**
- 修改：`backend/core/participants.py`
- 修改：`backend/games/dice/result.py`
- 修改：`tests/test_multi_game_vision.py`

- [ ] **步骤 1：在 `core/participants.py` 增加 `project_roles`**

只做物理侧 → 角色：`winner_role`（复用 `role_for_winner`）、`player_side`、`agent_side`。
**不含任何游戏语义**。

- [ ] **步骤 2：`dice/result.py` 改为复用（对外行为逐字不变）**

`project_participant_result(result, participants)` 的**签名与返回完全不变**，内部改为
`project_roles()` + 保留骰子专有的 1–6 校验与 `left_sum`/`first_dice` 等兼容字段校验。

- [ ] **步骤 3：新增 `project_categorical_result`**

```python
def project_categorical_result(result, participants, *, choice_field="choice") -> dict
```

把左右两侧的**字符串类别**（如 `rock`/`scissors`/`paper`）投影为
`player_choice`/`agent_choice` + 角色字段；**不做 1–6 数字校验**。

- [ ] **步骤 4：运行测试**

运行：`pytest -q tests/test_participants.py tests/test_multi_game_vision.py`
并在其中新增分类投影用例（两种摆位 + TIE）。

预期：`tests/test_participants.py` **一行不改且全绿**（这是 dice 行为不变的验收标准）。

- [ ] **步骤 5：提交**

```bash
git add backend/core/participants.py backend/games/dice/result.py tests/test_multi_game_vision.py
git commit -m "refactor(vision): 角色投影抽出到 core，新增分类投影以支持非数值游戏"
```

---

### 任务 6：`/api/health` 不再假定 dice

**文件：**
- 修改：`backend/server.py`
- 修改：`tests/test_server_api.py`

- [ ] **步骤 1：编写失败测试**

断言：当 dice 停用而有另一个启用的视觉游戏时，健康元数据取自那个游戏而不是 dice。

- [ ] **步骤 2：实现 `_primary_vision_game_id()`**

优先级：① 第一个**启用且带 `vision_profile`** 的游戏 → ② 回落 `"dice"`（兼容现状）→
③ 各字段留空（`_provider_health` / `_vision_profile_metadata` 已有 try/except 兜底，不抛）。

把 `server.py:1089-1105` 的四处 `"dice"` 换成该函数的结果。**API 形状不变**（字段名与结构一致），
因此 `tests/test_server_api.py` 既有断言（含 `len(profiles) == 1` 的合成单游戏用例）不受影响。

- [ ] **步骤 3：运行测试**

运行：`pytest -q tests/test_server_api.py`

- [ ] **步骤 4：提交**

```bash
git add backend/server.py tests/test_server_api.py
git commit -m "fix(server): 健康元数据不再硬编码 dice"
```

---

### 任务 7：全量验证

- [ ] **步骤 1：板端全量测试**

```bash
cd /home/spacemit/projects/dice-game/main
PYTHONPATH=/home/spacemit/dice-test-deps python3 -m pytest tests/ -q
```

预期：**539 passed / 1 skipped 起步，加上新增用例数**，无失败。
（必须在板端跑：开发机无 pytest、Python 3.12。**必须写 `tests/`**，根目录直接 pytest 会收集
vendored 目录炸掉。）

- [ ] **步骤 2：语法检查**

运行：`python3 -m py_compile` 覆盖全部改动文件；`node --check web/app.js`（本次应无前端改动）。

- [ ] **步骤 3：板端冒烟（写操作，先与用户确认）**

1. `scripts/stop_web.sh && scripts/start_web.sh`
2. dice 跑一局完整对局 → `/api/health` 的 `yolo_ready` / 槽位 / profile 元数据与改动前一致
3. `[vision] stream up ... (…, vision_always_on=…)` 与裁决结果正常
4. 若完成任务 3：热改 `vision/yolov8_adjudicator/config.json` 的 `conf`，**不重启**进下一局 →
   确认 runtime 自动重建

---

### 任务 8：文档同步

**文件：**
- 修改：`README.md`、`CLAUDE.md`、`AI_PROJECT_CONTEXT.md`、`FRAMEWORK_DISPATCH.md`
- 修改：`backend/games/dice/参数说明.md`

- [ ] **步骤 1：补「多游戏视觉架构」小节**

要点：签名契约（哪些 `vision_profile` 字段会触发 runtime 重建）；**共用摄像头下「切换游戏必重建」
是设计取舍不是缺陷**；新游戏接入清单（薄壳 pipeline + 投影 + manifest）；任务 3 的
「热改 runtime config 即生效」行为变化。
`backend/games/dice/参数说明.md` 说明 `vision_profile` 是**每游戏各写一份**的裁决参数入口。

- [ ] **步骤 2：`AI_PROJECT_CONTEXT.md` 补带日期的变更记录**

根因、签名补全内容、为什么缓存键坚持 `view_id`。按仓库惯例**不改写历史段落**，只追加。

- [ ] **步骤 3：提交**

```bash
git add README.md CLAUDE.md AI_PROJECT_CONTEXT.md FRAMEWORK_DISPATCH.md backend/games/dice/参数说明.md
git commit -m "docs(vision): 多游戏视觉架构与签名契约"
```

---

## 边界：明确不做

- **不构建 rps 游戏本体**：不动 rps 状态机（只有 1 态）、不写 rps 前端、不启用 rps、不接识别模型。
- **不做 per-game runtime config 文件**；不把摄像头/焦距/分辨率/EP 绑核做成按游戏可配。
- **不改缓存键为 `(game_id, view_id)`**：共用摄像头下不允许双 runtime 常驻。
- **不加视觉槽位唯一性封口**（dice/rps 用同一 provider，加了也无用）——列入"将来做"。
- **不改 C++**、不动 `web/`。

## 风险与处置

| 风险 | 处置 |
|---|---|
| 签名补全后部署第一局会重建 runtime | 一次性成本；已写入文档 |
| 任务 3 变更了"热改 runtime config 是否生效"的行为 | 显式写进提交信息与文档；**若用户不希望此行为变化，可跳过任务 3**（开工前再次确认） |
| `dice/pipeline.py` 薄壳化可能坏测试 | `run()` 签名与模块路径保持不变；两个 import 它的测试须在最终全量里绿 |
| 改 `dice/result.py` 可能坏 `test_participants.py` | 对外签名与返回不变，**测试文件一行不改即为验收标准** |
| 文档面较大（5 文件） | 文档独立提交，可单独回退 |

## 假设

- 两个游戏**共用同一个摄像头与 EP 绑定**（当前硬件即如此）。
- `profile["game_id"]` 可信（`validate_profile` 强制 + `load_games` 校验与 manifest id 一致）。
- 板端 `/home/spacemit/dice-test-deps` 提供 pytest。
