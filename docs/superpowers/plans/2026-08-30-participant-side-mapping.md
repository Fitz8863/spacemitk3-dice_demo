# 游戏参与者左右位置映射实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让每个游戏通过 `manifest.json` 配置玩家和 Agent 的物理侧，并让结果、页面布局和 TTS 使用统一的角色化字段。

**架构：** 游戏清单模块验证并安全公开角色到物理侧的映射；通用视觉裁决器继续只输出 `LEFT/RIGHT/TIE`。骰子游戏的结果投影模块在 pipeline seam 将物理结果转换为 `PLAYER/AGENT/TIE` 和角色化数值，前端仅消费角色化字段并按清单调整 CSS Grid 列位置。

**技术栈：** Python 3、标准库 JSON/HTTP、pytest/unittest、原生 JavaScript ES modules、CSS Grid、SpaceMIT K3 运行环境。

---

## 文件结构

- 创建 `backend/core/participants.py`：验证通用的双角色左右位置配置，并将物理胜者转换为角色胜者。
- 创建 `backend/games/dice/result.py`：把视觉裁决器的左右骰子结果投影为玩家/Agent 字段。
- 修改 `backend/core/games.py`：加载时验证 `participants`，并通过安全游戏清单公开它。
- 修改 `backend/games/dice/pipeline.py`：在视觉裁决器返回后调用角色结果投影。
- 修改 `backend/games/dice/manifest.json`：声明玩家在左、Agent 在右。
- 修改 `backend/games/rps/manifest.json`：为未来猜拳接入声明相同角色配置。
- 修改 `web/app.js`：进入游戏时把对应安全清单传给游戏模块。
- 修改 `web/games/dice.js`：验证位置一致性，使用角色化结果并调整玩家/Agent 卡片列位置。
- 修改 `web/index.html`：给两张角色卡片增加稳定 DOM id。
- 修改 `web/styles.css`：固定 `VS` 的中间网格位置。
- 创建 `tests/test_participants.py`：覆盖清单配置与骰子结果投影。
- 修改 `tests/test_components_and_jobs.py`：覆盖清单加载和 dice pipeline 的角色化结果。
- 修改 `tests/test_server_api.py`：验证 `/api/games` 安全公开参与者映射。
- 修改 `tests/test_web_contract.py`：约束前端不再把 LEFT/first 字段写死为玩家。

### 任务 1：建立参与者配置 seam

**文件：**
- 创建：`backend/core/participants.py`
- 修改：`backend/core/games.py`
- 修改：`backend/games/dice/manifest.json`
- 修改：`backend/games/rps/manifest.json`
- 创建：`tests/test_participants.py`
- 修改：`tests/test_components_and_jobs.py`
- 修改：`tests/test_server_api.py`

- [ ] **步骤 1：编写失败的配置验证测试**

在 `tests/test_participants.py` 中加入：

```python
from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.participants import normalize_participants, role_for_winner


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"player": "LEFT", "agent": "RIGHT"}, {"player": "LEFT", "agent": "RIGHT"}),
        ({"player": "RIGHT", "agent": "LEFT"}, {"player": "RIGHT", "agent": "LEFT"}),
    ],
)
def test_normalize_participants_accepts_both_physical_layouts(raw, expected):
    assert normalize_participants(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"player": "LEFT"},
        {"player": "LEFT", "agent": "LEFT"},
        {"player": "CENTER", "agent": "RIGHT"},
    ],
)
def test_normalize_participants_rejects_ambiguous_layouts(raw):
    with pytest.raises(ValueError, match="participants"):
        normalize_participants(raw)


def test_role_for_winner_preserves_tie_and_maps_both_sides():
    sides = {"player": "RIGHT", "agent": "LEFT"}
    assert role_for_winner("RIGHT", sides) == "PLAYER"
    assert role_for_winner("LEFT", sides) == "AGENT"
    assert role_for_winner("TIE", sides) == "TIE"
    with pytest.raises(ValueError, match="winner"):
        role_for_winner("UNKNOWN", sides)
```

- [ ] **步骤 2：运行测试并确认正确失败**

运行：

```bash
pytest -q tests/test_participants.py
```

预期：测试收集失败，提示 `No module named 'core.participants'`。

- [ ] **步骤 3：实现最小参与者配置模块**

创建 `backend/core/participants.py`：

```python
"""Two-role game participant placement and physical winner mapping."""
from __future__ import annotations

from typing import Any, Mapping


SIDES = {"LEFT", "RIGHT"}
ROLES = ("player", "agent")


def normalize_participants(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("participants must map player and agent to physical sides")
    normalized: dict[str, str] = {}
    for role in ROLES:
        side = value.get(role)
        if side not in SIDES:
            raise ValueError(f"participants.{role} must be LEFT or RIGHT")
        normalized[role] = str(side)
    if normalized["player"] == normalized["agent"]:
        raise ValueError("participants.player and participants.agent must use different sides")
    return normalized


def role_for_winner(winner: str, participants: Mapping[str, Any]) -> str:
    sides = normalize_participants(participants)
    if winner == "TIE":
        return "TIE"
    if winner == sides["player"]:
        return "PLAYER"
    if winner == sides["agent"]:
        return "AGENT"
    raise ValueError("winner must be LEFT, RIGHT, or TIE")
```

- [ ] **步骤 4：让游戏清单验证并公开参与者配置**

在 `backend/core/games.py` 导入 `normalize_participants`，在 `load_games()` 验证基础字段后加入：

```python
manifest["participants"] = normalize_participants(manifest.get("participants"))
```

在 `public_game_manifest()` 的安全顶层字段白名单中加入 `participants`。更新两个游戏清单：

```json
"participants": {
  "player": "LEFT",
  "agent": "RIGHT"
}
```

在 `tests/test_components_and_jobs.py::test_game_loader_prefers_inline_vision_profile` 的临时清单中加入相同配置，并断言加载后的 `participants`。

在 `tests/test_server_api.py::test_games_api_exposes_only_safe_vision_video_metadata` 的测试清单中加入交换配置并断言：

```python
self.assertEqual(payload["games"][0]["participants"], {"player": "RIGHT", "agent": "LEFT"})
```

- [ ] **步骤 5：运行配置相关测试**

运行：

```bash
pytest -q tests/test_participants.py tests/test_components_and_jobs.py::ComponentTests::test_game_loader_prefers_inline_vision_profile tests/test_server_api.py::ServerApiTests::test_games_api_exposes_only_safe_vision_video_metadata
```

预期：全部 PASS。若实际 unittest 类名与计划不同，先用 `pytest --collect-only -q tests/test_server_api.py` 取得精确 node id，再运行同一个测试方法。

- [ ] **步骤 6：提交配置 seam**

```bash
git add backend/core/participants.py backend/core/games.py \
  backend/games/dice/manifest.json backend/games/rps/manifest.json \
  tests/test_participants.py tests/test_components_and_jobs.py tests/test_server_api.py
git commit -m "feat: configure game participant sides"
```

### 任务 2：在骰子 pipeline 投影角色结果

**文件：**
- 创建：`backend/games/dice/result.py`
- 修改：`backend/games/dice/pipeline.py`
- 修改：`tests/test_participants.py`
- 修改：`tests/test_components_and_jobs.py`

- [ ] **步骤 1：编写默认、交换和平局的失败测试**

在 `tests/test_participants.py` 加入：

```python
from games.dice.result import project_participant_result


def physical_result(winner="RIGHT"):
    return {
        "winner": winner,
        "outcome": {"kind": "winner", "value": winner},
        "left_values": [4, 4, 1, 1, 1],
        "right_values": [5, 4, 6, 2, 2],
        "left_sum": 11,
        "right_sum": 19,
        "first_dice": [4, 4, 1, 1, 1],
        "second_dice": [5, 4, 6, 2, 2],
    }


def test_project_participant_result_default_layout():
    result = project_participant_result(
        physical_result(), {"player": "LEFT", "agent": "RIGHT"}
    )
    assert result["winner"] == "RIGHT"
    assert result["winner_role"] == "AGENT"
    assert result["player_values"] == [4, 4, 1, 1, 1]
    assert result["agent_values"] == [5, 4, 6, 2, 2]
    assert result["player_score"] == 11
    assert result["agent_score"] == 19


def test_project_participant_result_swapped_layout_preserves_physical_fields():
    original = physical_result()
    result = project_participant_result(original, {"player": "RIGHT", "agent": "LEFT"})
    assert result["winner"] == "RIGHT"
    assert result["winner_role"] == "PLAYER"
    assert result["player_values"] == original["right_values"]
    assert result["agent_values"] == original["left_values"]
    assert result["player_score"] == 19
    assert result["agent_score"] == 11
    assert result["first_dice"] == original["first_dice"]
    assert result["second_dice"] == original["second_dice"]


def test_project_participant_result_maps_tie():
    result = project_participant_result(
        physical_result("TIE"), {"player": "LEFT", "agent": "RIGHT"}
    )
    assert result["winner_role"] == "TIE"
```

- [ ] **步骤 2：运行结果投影测试并确认失败**

运行：

```bash
pytest -q tests/test_participants.py -k project_participant_result
```

预期：测试收集失败，提示 `No module named 'games.dice.result'`。

- [ ] **步骤 3：实现骰子角色结果投影**

创建 `backend/games/dice/result.py`：

```python
"""Project physical dice adjudication into player and Agent roles."""
from __future__ import annotations

from numbers import Real
from typing import Any, Mapping

from core.participants import normalize_participants, role_for_winner


def _side_values(result: Mapping[str, Any], side: str) -> list[Any]:
    value = result.get(f"{side.lower()}_values")
    if not isinstance(value, list):
        raise ValueError(f"dice result is missing {side.lower()}_values")
    return list(value)


def _side_score(result: Mapping[str, Any], side: str) -> Real:
    value = result.get(f"{side.lower()}_sum")
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"dice result is missing numeric {side.lower()}_sum")
    return value


def project_participant_result(
    result: Mapping[str, Any], participants: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("dice result must be an object")
    sides = normalize_participants(participants)
    winner = result.get("winner")
    if not isinstance(winner, str):
        raise ValueError("dice result winner must be LEFT, RIGHT, or TIE")
    projected = dict(result)
    projected.update({
        "winner_role": role_for_winner(winner, sides),
        "player_side": sides["player"],
        "agent_side": sides["agent"],
        "player_values": _side_values(result, sides["player"]),
        "agent_values": _side_values(result, sides["agent"]),
        "player_score": _side_score(result, sides["player"]),
        "agent_score": _side_score(result, sides["agent"]),
    })
    return projected
```

- [ ] **步骤 4：让 dice pipeline 只在返回 seam 投影结果**

在 `backend/games/dice/pipeline.py` 导入 `project_participant_result`。把两条 adjudicator 调用路径的返回值先保存为 `physical_result`，最后统一：

```python
return project_participant_result(physical_result, manifest["participants"])
```

不要修改或包装视觉裁决器的 `on_event`；任务完成时的规范 `snapshot.result` 使用 pipeline 返回的角色化结果，视觉功能包仍保持通用物理事件。

更新 `tests/test_components_and_jobs.py::test_dice_pipeline_invokes_adjudicator_interface`，让 dummy 结果包含左右数组和分数，并断言 `winner_role`、`player_score`、`agent_score`。

- [ ] **步骤 5：运行 pipeline 和投影测试**

运行：

```bash
pytest -q tests/test_participants.py tests/test_components_and_jobs.py -k "participant or dice_pipeline"
```

预期：全部选中测试 PASS。

- [ ] **步骤 6：提交角色结果投影**

```bash
git add backend/games/dice/result.py backend/games/dice/pipeline.py \
  tests/test_participants.py tests/test_components_and_jobs.py
git commit -m "feat: project dice results by participant role"
```

### 任务 3：让前端按配置布局并消费角色结果

**文件：**
- 修改：`web/app.js`
- 修改：`web/games/dice.js`
- 修改：`web/index.html`
- 修改：`web/styles.css`
- 修改：`tests/test_web_contract.py`

- [ ] **步骤 1：编写失败的前端契约测试**

在 `tests/test_web_contract.py` 加入：

```python
def test_frontend_uses_manifest_participant_layout_and_role_result():
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    dice = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")

    assert "module.enter(manifest)" in app
    assert "playerScoreSide" in html
    assert "agentScoreSide" in html
    assert "result.player_values" in dice
    assert "result.agent_values" in dice
    assert "result.winner_role" in dice
    assert "result.player_side" in dice
    assert "result.agent_side" in dice
    assert "winner === 'LEFT'" not in dice
    assert "result.first_dice" not in dice
    assert "result.second_dice" not in dice
```

- [ ] **步骤 2：运行前端契约测试并确认失败**

运行：

```bash
pytest -q tests/test_web_contract.py::test_frontend_uses_manifest_participant_layout_and_role_result
```

预期：FAIL，首先缺少 `module.enter(manifest)`。

- [ ] **步骤 3：把安全游戏清单传入游戏模块**

在 `web/app.js::enterSelectedGame()` 查找当前 manifest，并调用：

```javascript
const manifest = games.find((game) => game.id === state.selectedGame);
if (!manifest) {
  toast(`缺少游戏配置：${state.selectedGame}`);
  return;
}
activeGame = module;
module.enter(manifest);
```

- [ ] **步骤 4：给角色卡片稳定 id 并固定中间列**

在 `web/index.html` 增加：

```html
<div class="score-side player-side" id="playerScoreSide">
...
<div class="score-side agent-side" id="agentScoreSide">
```

在 `web/styles.css` 增加：

```css
.score-side { grid-row: 1; }
.versus { grid-column: 2; grid-row: 1; }
```

- [ ] **步骤 5：实现骰子前端角色配置和一致性检查**

在 `web/games/dice.js` 增加模块状态和辅助函数：

```javascript
let participantSides = null;

function configureParticipants(manifest) {
  const participants = manifest && manifest.participants;
  const player = participants && participants.player;
  const agent = participants && participants.agent;
  if (!['LEFT', 'RIGHT'].includes(player)
      || !['LEFT', 'RIGHT'].includes(agent)
      || player === agent) {
    throw new Error('游戏参与者左右位置配置无效');
  }
  participantSides = { player, agent };
  $('playerScoreSide').style.gridColumn = player === 'LEFT' ? '1' : '3';
  $('agentScoreSide').style.gridColumn = agent === 'LEFT' ? '1' : '3';
}

function assertResultParticipants(result) {
  if (!participantSides
      || result.player_side !== participantSides.player
      || result.agent_side !== participantSides.agent) {
    throw new Error('裁决结果与游戏参与者位置配置不一致');
  }
}
```

把 `enter()` 改为 `enter(manifest)` 并首先调用 `configureParticipants(manifest)`。

- [ ] **步骤 6：让结果页和 TTS 只消费角色化字段**

在 `showResult(result)` 首先调用 `assertResultParticipants(result)`，随后改为：

```javascript
playerDice = Array.isArray(result.player_values) ? result.player_values : [];
agentDice = Array.isArray(result.agent_values) ? result.agent_values : [];
const player = Number(result.player_score);
const agent = Number(result.agent_score);
const winnerRole = result.winner_role;
const tie = winnerRole === 'TIE';
const playerWins = winnerRole === 'PLAYER';
if (!tie && !playerWins && winnerRole !== 'AGENT') {
  throw new Error('裁决结果缺少有效 winner_role');
}
```

结果副标题改为明确角色顺序：

```javascript
$('resultSubtitle').textContent = `YOLOv8：玩家 ${player} : Agent ${agent}；大模型${verificationText}`;
```

TTS 仍调用：

```javascript
speakState(resultTtsKey, { player_score: player, agent_score: agent });
```

- [ ] **步骤 7：运行前端测试和语法检查**

运行：

```bash
pytest -q tests/test_web_contract.py
node --check web/app.js
node --check web/games/dice.js
```

预期：全部 PASS，两个 `node --check` 退出码均为 0。

- [ ] **步骤 8：提交前端配置化布局**

```bash
git add web/app.js web/games/dice.js web/index.html web/styles.css tests/test_web_contract.py
git commit -m "feat: render dice participants by configured side"
```

### 任务 4：完整回归与 K3 验证

**文件：**
- 验证：`backend/games/dice/manifest.json`
- 验证：`backend/components/vision_yolov8_adjudicator/config.json`（只检查状态，不读取或提交内容）
- 验证：`backend/games/dice/audio/fll.wav`（只确认未提交）

- [ ] **步骤 1：运行完整本地测试**

```bash
pytest -q tests
node --check web/app.js
node --check web/games/dice.js
git diff --check
```

预期：全部测试 PASS，JavaScript 语法检查与 Git 空白检查退出码为 0。

- [ ] **步骤 2：确认 Git 修改边界**

```bash
git status --short --branch
git log -6 --oneline --decorate
```

预期：只剩用户已有的未提交项：

```text
 M backend/components/vision_yolov8_adjudicator/config.json
?? backend/games/dice/audio/fll.wav
```

- [ ] **步骤 3：验证 K3 默认位置配置**

通过 SSH 在 `/home/spacemit/projects/dice-game/main`：

```bash
curl -fsS http://127.0.0.1:8080/api/games
curl -fsS http://127.0.0.1:8080/api/health
```

重启 Web 服务以重新加载 manifest，完成一次真实骰子裁决。断言：

- `/api/games` 返回 `participants.player=LEFT` 和 `participants.agent=RIGHT`。
- 最终任务同时返回物理 `winner` 和角色 `winner_role`。
- `player_score` 对应左侧，`agent_score` 对应右侧。
- 页面左列显示玩家、右列显示 Agent。
- holding 仍完整显示 3、2、1 秒，完成后 YOLO 推理停止。

- [ ] **步骤 4：受控验证交换位置并恢复**

只对已跟踪的 `backend/games/dice/manifest.json` 做临时交换：玩家 `RIGHT`、Agent `LEFT`，重启 Web 服务并完成页面/API 验证：

- `/api/games` 返回交换后的配置。
- 玩家卡片位于右列、Agent 卡片位于左列。
- `winner` 仍代表物理侧。
- `winner_role`、玩家/Agent 数值与分数随配置正确交换。

验证后立即恢复清单为玩家 `LEFT`、Agent `RIGHT`，重启服务并用 `git diff -- backend/games/dice/manifest.json` 确认没有临时改动残留。

- [ ] **步骤 5：最终健康和服务状态核对**

```bash
curl -fsS http://127.0.0.1:8080/api/health
pgrep -af "[b]ackend/server.py --host 0.0.0.0 --port 8080"
git status --short --branch
```

预期：健康接口 `ok=true`，Web 服务运行，Git 仅保留用户原有未提交配置和音频文件。
