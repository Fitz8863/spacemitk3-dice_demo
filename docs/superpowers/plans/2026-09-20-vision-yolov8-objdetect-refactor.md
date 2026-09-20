# 视觉功能包解耦改造：yolov8_adjudicator → yolov8_objdetect

> **面向 AI 代理的工作者：** 使用 `superpower:executing-plans` 逐任务实现此计划。
> 步骤使用复选框（`- [ ]`）语法跟踪进度。每个阶段验证通过后独立提交。

**存档点：** `pre-objdetect-refactor`（动手前在 `cca9568` 打的本地 tag）。
回退：`git reset --hard pre-objdetect-refactor`。

**日期：** 2026-09-20　**用户决策访谈：** 已完成（6 项决策全部确认）

## 目标

把 `vision/yolov8_adjudicator` 从"视觉裁决器"重命名为纯目标检测功能包
`vision/yolov8_objdetect`，并将残留的 dice 游戏语义（区域门控）从 C++ 拿掉。
解耦后视觉包只负责：**采集 → 预处理 → 推理 → 稳定检测观测输出**；
分组、规则、输赢、诊断全部由游戏侧（Python）规划。

## 边界（本次改造后的职责划分）

```
C++ vision/yolov8_objdetect（纯检测包）          Python 游戏侧
├─ 摄像头采集/解码（GStreamer）                 ├─ class_map/participants 分组（provider）
├─ OpenCL NV12 预处理                           ├─ expected_count 数量校验（已有，复核）
├─ YOLOv8 推理 + NMS                            ├─ 规则评估 numeric_compare 等（rules.py）
├─ 稳定判定：类别多重集连续 stable_frames 帧不变  ├─ 诊断重试 INCOMPLETE_OBJECTS 等
├─ 分界线几何定位（可选，开关控制，仅提供坐标）    ├─ 结果投影到角色（games/*/result.py）
└─ JSONL 事件输出（jsonl-events-v2）             └─ LLM 复核（可配置，当前关）
```

## 六项已确认的设计决策

1. **stable_frames 留 C++**：帧内防抖是检测语义而非游戏语义；数值由各游戏
   `adjudicator_config.json` 自定义（dice=30，rps=15，现状已如此）。
2. **区域门控删除、上移 Python**：`expected_count`/`region_position`/
   `region_layout_usable()` 整套从 C++ 删除；"每侧必须 5 个"由 Python 在拿到
   stable 观测后校验（`provider.py:935` 的 `_has_incomplete_expected_counts`
   已存在，本次复核加固而非新写）。
3. **分界线检测留 C++、纯开关控制**：它是场景几何服务（黑线/红蓝线定位），
   在已解码帧上计算成本≈0；`divider_detection=false`（rps）时不跑、事件不带
   `divider` 字段。代码留在视觉包，但语义中性、默认可关。
4. **目录+组件全改名**：C++ 目录、Python 组件目录、组件 id、事件 component
   标识、入口类名、文档一次改齐。**二进制名 `yolov8_camera` 保留**（名字中性，
   减少波及面）；**后端槽位名 `vision_adjudicator` 保留**（它命名的是后端能力
   槽位而非组件，游戏裁决流程仍在 Python）。
5. **事件协议升 `jsonl-events-v2`**：stable 语义从"每侧数量达标且稳定"收紧为
   "类别多重集稳定"；`started` 事件的 `component` 改为 `vision_yolov8_objdetect`。
   Python 适配器在握手时校验 `started.protocol`（当前无任何代码校验，本次补上），
   旧二进制混跑立即报错而不是静默误读。
6. **验证标准：全链路**——Python 全量测试（开发机）+ C++ 板端重编/self-test +
   dice、rps 各一次有相机实测。

## 背景事实（代码证据，2026-09-20 核实）

| 事实 | 位置 |
|---|---|
| 输赢裁决/求和**已不在 C++**：无 participants/sum/winner 计算，规则引擎在 Python 且游戏无关 | `vision/yolov8_adjudicator/src/main.cpp` 全文；`backend/components/vision_yolov8_adjudicator/rules.py` |
| 左右分组**已在 Python**（class_map + divider_regions/x_midpoint） | `provider.py:107 normalize_observation()` |
| C++ 残留 dice 门控：expected_count/region_position 参数 + `region_layout_usable()` + 主循环 region_gate/region_signature | `main.cpp:192-193,531-533,732-760,1413-1437` |
| Python 已有数量补偿：stable 观测每侧 ≠ expected_count → STOP_ADJUDICATION + 诊断重试 | `provider.py:225,935` |
| 分界线检测黑线+红蓝两实现，事件带 divider 坐标，rps 配置已关 | `main.cpp:823,923,1023`；`games/rps/adjudicator_config.json` `divider_detection:false` |
| 组件 id 接线：manifest `id`、`backend/config.json` 槽位映射、componentctl fallback、games.py×2、server.py×2、vision_pipeline、components 别名 | 见下"改名清单" |
| `jsonl-events-v1` 无任何 Python 校验（声明性字段） | 全仓 grep 仅 config.json/文档 |
| 测试引用密度：test_vision_adjudicator 21 处、test_multi_game_vision 17 处等 9 个文件 | `tests/` |

## 改名清单（精确到文件）

**C++ 侧**（`git mv vision/yolov8_adjudicator vision/yolov8_objdetect`）
- `src/main.cpp`：started 事件 `component`/`protocol` 两字符串；删区域门控
  （Args 三字段、validate_args 两段、parse 三分支、usage 三行、
  `region_layout_usable()`、主循环 region_gate/region_signature/`divider_ready`
  中引用 expected_count 的分支）；分界线相关注释措辞中性化。
- `README.md`、`AGENTS.md`：目录名、协议版本、命令示例。

**Python 侧**（`git mv backend/components/vision_yolov8_adjudicator backend/components/vision_yolov8_objdetect`）
- 组件 `manifest.json`：`id`→`vision_yolov8_objdetect`、`name`、
  `entry`→`provider.py:VisionYolov8Objdetect`、capabilities 措辞。
- `config.json`：`runtime.binary`/`runtime.working_dir` 路径、
  `events.protocol`→`jsonl-events-v2`。
- `provider.py`：类名 `VisionYolov8Adjudicator`→`VisionYolov8Objdetect`、
  内部 import 路径×4。
- `process.py`：import 路径；新增 started 事件协议握手校验。
- `profile.py`/`rules.py`：import/文档串。
- `参数说明.md`：全文。
- 外部接线：`backend/config.json` 槽位值、`componentctl.py:69`、
  `core/components.py:43`、`core/games.py:235,239`、`core/vision_pipeline.py:49`、
  `server.py:70,1136`、`scripts/bundle_install_dice.sh`、`scripts/make_bundle.sh`。
- 测试：9 个文件全部引用更新 + 门控删除导致的语义断言更新。

**文档同步**（改名+协议+职责描述）：`AI_PROJECT_CONTEXT.md`、`CLAUDE.md`、
`README.md`、`FRAMEWORK_DISPATCH.md`、`docs/README.md`、`backend/README.md`、
`backend/参数说明.md`、`backend/components/README.md`、
`backend/games/dice/参数说明.md`。
**不动**：`docs/superpowers/`（历史规划/规格存档）、`docs/archive/`。

## 行为变化（需知悉）

1. dice 场景"每侧数量不对"时：原来 C++ 不计稳定帧 → 走 8s 检测超时诊断；
   现在快速出 stable 观测 → Python 立即 INCOMPLETE_OBJECTS 诊断重试。
   **诊断更快，路径不同，结果等价。**
2. 协议 v2 是破坏性标记：C++/Python 必须同仓同换（本就同仓，一次提交完成）。
3. 板端 `build/` 目录因改名含旧绝对路径缓存，需删掉重新 configure。

## 任务分解

### 阶段 A：C++ 功能包化（提交点 A：5793556 ✅）
- [x] A1 `git mv vision/yolov8_adjudicator vision/yolov8_objdetect`
- [x] A2 main.cpp：started 事件 protocol→v2、component→vision_yolov8_objdetect
- [x] A3 main.cpp：删区域门控全套（Args/validate/parse/usage/函数/主循环）
- [x] A4 main.cpp：stable 判定收敛为纯 `detection_signature` 比较；usage/注释中性化
- [x] A5 README.md/AGENTS.md 同步
- [x] A6 板端：rm -rf build → cmake 重新 configure → build → self-test 通过
- [x] A7 提交

### 阶段 B：Python 组件化改造（提交点 B：fbcbb7d ✅）
- [x] B1 `git mv backend/components/vision_yolov8_adjudicator backend/components/vision_yolov8_objdetect`
- [x] B2 manifest/config/provider/process/profile/rules/参数说明 全部更新
- [x] B3 外部接线 8 处更新（config/componentctl/components/games/vision_pipeline/server/scripts×2）
- [x] B4 process.py：started 协议握手校验（≠jsonl-events-v2 即报错）
- [x] B5 provider.py：复核 `_has_incomplete_expected_counts` 调用链
      ——结论：`_diagnose_failure` 内部自行 normalize，传原始观测正确，无缺陷
- [x] B6 测试 9 文件更新；新增：协议握手拒绝 v1、expected_count 不重建 runtime
- [x] B7 开发机 pytest 全量通过：595 passed（含真实进程测试）
- [x] B8 提交

### 阶段 C：板端全链路验证（部分完成）
- [ ] C1 dice 有相机实测：5+5 骰子 → 稳定 → 裁决正确；故意摆 4+5 → 立即诊断重试
      ——**被占用阻塞**：/dev/video1 由用户运行的 yolov8_seg_camera 持有
- [ ] C2 rps 有相机实测：divider_detection=false 生效、事件无 divider 字段
      ——同上阻塞；rps 配置 self-test 已过（模型临时借用回收站文件），
      "无 divider 字段"已有源码证据（emit_observation 关闭态传 nullptr + if(divider_assist)）
- [x] C3 文档收尾同步（A/B 阶段一并完成，无偏差）
- [x] 板端 web 后端已用新代码重启（旧代码在内存里找旧目录路径，重启前新对局会失败）
      并通过 /api/health（新组件名注册、无旧名残留）

## 验证标准（用户已确认"全链路"）

| 层 | 验证 | 通过标准 |
|---|---|---|
| Python | 开发机 `python3 -m pytest tests/ -x -q` | 全绿 |
| C++ 编译 | 板端 cmake build | 零警告（-Wall -Wextra -Wpedantic） |
| C++ 自检 | `--self-test --no-display`（dice/rps 配置各一次） | passed |
| dice 实测 | 相机 + 骰子 | 5+5 正确裁决；4+5 快速诊断 |
| rps 实测 | 相机 + 手势 | 无分界线检测、检测框正常、无 divider 事件字段 |

## 风险与固有代价

- **协议 v2 破坏性**：旧板端二进制 + 新 Python 混跑会在握手时报错——这是设计
  行为（fail fast），部署时同仓替换即可。
- **诊断路径变化**：dice 不完整场景从超时诊断变为立即诊断；若有前端文案依赖
  超时路径需回归（诊断码相同：INCOMPLETE_OBJECTS）。
- **与独立工程同名**：`projects/dice-game/yolov8_objdetect`（旧单进程试验工程）
  与 `main/vision/yolov8_objdetect`（本组件）目录名相同但父目录不同，无冲突；
  未来如合并需注意。
