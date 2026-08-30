# 游戏参与者左右位置映射设计

日期：2026-08-30

## 背景

Dice Arena 的游戏通常包含两个业务角色：人类玩家 `PLAYER` 和智能体玩家 `AGENT`。智能体未来由机械臂执行。两个角色分别占据画面的物理左侧和右侧，但实际部署中可能交换位置。

视觉裁决器只应理解画面中的物理方位 `LEFT`、`RIGHT` 和 `TIE`。玩家、Agent、页面文案、TTS 和机械臂属于游戏业务语义，不应进入 YOLO 或云端视觉复核功能包。

当前骰子页面把左侧结果直接当作玩家、右侧结果直接当作 Agent。该写死关系会在物理位置交换后导致分数、胜负标题、TTS 和页面位置全部错误。

## 目标

- 每个游戏在自己的 `manifest.json` 中声明玩家和 Agent 所在的物理侧。
- 只修改配置即可交换玩家和 Agent 的左右位置。
- 视觉裁决器继续输出通用的物理方位结果。
- 游戏 pipeline 集中完成物理方位到业务角色的转换。
- 前端、TTS 和未来机械臂调度使用统一的角色化结果。
- 保留现有物理方位与骰子兼容字段，避免破坏现有调用方。
- 对无效或含糊的位置配置尽早报错。

## 非目标

- 本次不支持两个以上的游戏参与者。
- 本次不支持自定义角色类型、角色名称或页面配色。
- 本次不修改 YOLO 分类、稳定帧、分区或 LLM 提示词语义。
- 本次不实现机械臂控制，只为其提供稳定的角色结果字段。
- 本次不删除现有骰子兼容字段。

## 方案选择

采用游戏顶层的角色到物理位置映射：

```json
"participants": {
  "player": "LEFT",
  "agent": "RIGHT"
}
```

交换位置时只交换两个值：

```json
"participants": {
  "player": "RIGHT",
  "agent": "LEFT"
}
```

未采用以下方案：

- `positions.LEFT = PLAYER`：查询业务角色位置时需要反向查找，不如角色到位置映射直接。
- 放入 `vision_profile`：页面、TTS 和机械臂都会被迫依赖视觉配置，扩大视觉裁决器的接口并产生不必要耦合。

## 模块与接口

### 游戏清单模块

游戏清单模块负责验证和公开 `participants`。它的接口保证：

- `player` 和 `agent` 均存在。
- 两个值只能是 `LEFT` 或 `RIGHT`。
- 两个角色不能占据同一侧。
- 浏览器安全清单 `/api/games` 包含规范化后的 `participants`。
- 模型路径、提示词、摄像头和凭据仍不会通过 `/api/games` 暴露。

仓库内的骰子和猜拳清单都应声明该字段。无效清单按照现有加载策略拒绝注册并输出明确配置错误。

### 视觉裁决器模块

视觉裁决器的接口保持不变：

- 参与者证据继续按 `LEFT` 和 `RIGHT` 分组。
- `outcome.value` 和兼容字段 `winner` 继续输出 `LEFT`、`RIGHT` 或 `TIE`。
- YOLO 和 LLM 不理解 `PLAYER` 或 `AGENT`。
- 通用视觉功能包不读取游戏顶层 `participants`。

### 骰子游戏 pipeline

骰子 pipeline 是物理方位到业务角色转换的 seam。它在视觉裁决器返回后读取 `manifest.participants`，生成角色化结果，再把结果返回任务系统。

输入：

- 已验证的游戏清单。
- 视觉裁决器的物理方位结果和左右证据。

输出新增字段：

```json
{
  "winner": "RIGHT",
  "winner_role": "AGENT",
  "player_side": "LEFT",
  "agent_side": "RIGHT",
  "player_values": [4, 4, 1, 1, 1],
  "agent_values": [5, 4, 6, 2, 2],
  "player_score": 11,
  "agent_score": 19
}
```

转换规则：

- `winner == player_side` 时，`winner_role = "PLAYER"`。
- `winner == agent_side` 时，`winner_role = "AGENT"`。
- `winner == "TIE"` 时，`winner_role = "TIE"`。
- `player_values` 取自玩家配置侧的物体值。
- `agent_values` 取自 Agent 配置侧的物体值。
- 数值数组分别求和生成 `player_score` 和 `agent_score`。
- 视觉结果缺少合法物理胜者或所需左右证据时，pipeline 报错，不猜测角色。

### 前端游戏模块

前端引擎在进入游戏时把该游戏的安全清单传给游戏模块。骰子模块读取 `participants`，不再假设玩家固定在左侧。

结果页面继续保留玩家卡片和 Agent 卡片各自的样式及 DOM 身份。页面通过 CSS Grid 列位置放置卡片：

- 配置为玩家 `LEFT` 时，玩家卡片在左列，Agent 卡片在右列。
- 配置为玩家 `RIGHT` 时，Agent 卡片在左列，玩家卡片在右列。
- 中间的 `VS` 始终位于中间列。

前端使用：

- `player_values` / `agent_values` 渲染骰子。
- `player_score` / `agent_score` 渲染分数。
- `winner_role` 生成“玩家获胜”“Agent 获胜”或“平局”标题。
- `player_side` / `agent_side` 与公开清单互相校验并确定页面列位置。

前端不再通过 `winner === "LEFT"` 判断玩家获胜，也不再把 `first_dice` 和 `second_dice` 当作玩家与 Agent。

### TTS 与未来机械臂

TTS 模板保持 `player_score` 和 `agent_score` 占位符。页面根据角色化结果选择：

- `result_player_win`
- `result_agent_win`
- `result_tie`

因此交换物理位置不会改变 TTS 模板或胜负播报语义。

未来机械臂调度可直接使用 `winner_role` 判断业务胜者，并使用 `player_side`、`agent_side` 定位物理侧，无需解析视觉证据或复制映射逻辑。

## 兼容策略

以下字段继续保留原有物理方位语义：

- `outcome.value`
- `winner`
- `left_values` / `right_values`
- `left_sum` / `right_sum`
- `first_dice` / `second_dice`
- `first_sum` / `second_sum`

骰子兼容约定保持：

- `first_*` 始终代表 `LEFT`。
- `second_*` 始终代表 `RIGHT`。

新代码必须优先使用角色化字段。兼容字段不参与玩家/Agent 判断。

## 数据流

```text
摄像头画面
  -> YOLO 稳定帧和 LEFT/RIGHT 证据
  -> LLM 复核 LEFT/RIGHT/TIE
  -> 视觉裁决器生成物理 winner
  -> 骰子 pipeline 读取 manifest.participants
  -> 生成 winner_role 和角色化数值/分数
  -> 任务结果
  -> 前端布局、结果文案和 TTS
  -> 未来机械臂调度
```

视觉裁决器和游戏角色映射之间只有物理方位结果这一条接口。交换角色位置不需要修改视觉功能包。

## 错误处理

- 缺少 `participants`：游戏清单加载失败。
- 缺少 `player` 或 `agent`：游戏清单加载失败。
- 位置不是 `LEFT` 或 `RIGHT`：游戏清单加载失败。
- 两个角色位置相同：游戏清单加载失败。
- 视觉胜者不在 `LEFT`、`RIGHT`、`TIE` 中：pipeline 拒绝生成角色结果。
- 结果缺少配置侧的数值：pipeline 返回明确错误，不使用另一侧数据补齐。
- 前端清单与任务结果的位置字段冲突：页面报告配置不一致，不宣判错误角色。

## 测试策略

### 单元测试

- 清单验证接受玩家左、Agent 右。
- 清单验证接受玩家右、Agent 左。
- 清单验证拒绝缺少角色、非法位置和重复位置。
- 公共清单包含 `participants`，且继续过滤视觉私有配置。
- 角色投影在默认位置下正确生成所有新字段。
- 角色投影在交换位置下正确交换数值、分数和 `winner_role`。
- 平局映射为 `winner_role = "TIE"`。
- 原有物理方位和兼容字段保持不变。

### 前端契约测试

- 游戏模块从安全清单读取参与者位置。
- 胜负标题只依赖 `winner_role`。
- 角色卡片按配置进入正确网格列。
- TTS 使用 `player_score` 和 `agent_score`。
- 前端不再将 `LEFT` 或 `first_*` 写死为玩家。

### 完整验证

- 运行本地完整 `tests` 测试集和 JavaScript 语法检查。
- 在 K3 上确认清单和健康接口正常。
- 使用默认配置完成一次真实骰子裁决，确认左侧玩家、右侧 Agent。
- 用交换位置配置做受控验证，确认页面左右卡片、分数、胜负标题和 TTS 参数同步交换；验证结束后恢复用户指定配置。
- 确认视觉裁决器仍只输出物理方位，YOLO 停止与结果 holding 生命周期不受影响。

## 修改边界

预计只修改：

- 游戏清单验证与安全公开逻辑。
- 骰子和猜拳的 `manifest.json`。
- 骰子 pipeline 的结果角色投影。
- 前端引擎进入游戏的清单传递。
- 骰子结果页布局和结果字段消费。
- 对应测试与架构文档。

不修改视觉功能包的检测、稳定帧、LLM、控制协议和部署凭据配置，也不修改 TTS 功能包内部实现。
