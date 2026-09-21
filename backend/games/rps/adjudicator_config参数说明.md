# rps 视觉 runtime 配置（adjudicator_config.json）参数说明

`backend/games/rps/adjudicator_config.json` 是猜拳游戏**唯一的 C++ runtime 配置**，由
`vision/yolov10_objdetect/build/yolov10_camera` 消费（游戏 manifest 的
`vision_profile.runtime_config` 指向本文件）。JSON 不支持注释，本文档就是它的注释。

**三份文档的分工**：本文件管"rps 的视觉怎么调"；`参数说明.md`（同目录）管游戏层
（manifest 状态机/词表/槽位）；`backend/components/vision_yolov10_objdetect/参数说明.md`
管 v10 组件包视角（组件 config、协议、observe 接口）。键的完整语义表在那边，本文
给**每个键的当前值、实测依据与调参落点**。

## 0. 这个文件怎么生效（先读）

| 规则 | 说明 |
|---|---|
| **整份替换** | 没写的键回落到 **C++ 编译期默认值**（不是回落别的文件）：`conf 0.25`、`stable_frames 20`、`focus 0`、`rtsp.path /rps/det` 等。加字段请整份补齐 |
| **下一回合生效** | 文件 mtime+size 已进 runtime 签名：保存后**下一个回合自动重建 resident runtime**，无需重启后端；正在跑的一局用旧参数 |
| **启动即校验** | 词表/折叠表写错（未知类名、空标签、源类重复映射、`rps_mode` 无词表）在 runtime 启动时**报错退出**——那一局的裁决直接失败（页面显示"识别未完成"，日志有具体原因）。改完词表先跑一次 self-test（第 7 节） |
| **路径解析** | `model` 与模型文件按**本文件所在目录**解析（`models/yolov10n_gestures.q.onnx` 即 `backend/games/rps/models/`） |
| **命令行恒压** | 生产由 provider 下发 `--config/--rtsp-path/--conf` 等参数压过本文件（`rtsp.path` 因此不在此声明，见第 6 节）；手工跑 CLI 时本文件说了算 |
| `_note` | 两个解析器都忽略未知键——这个键是给人看的语义说明，保留 |

## 1. 词表与折叠（本文件的核心，rps 特有）

| 键 | 当前值 | 说明 |
|---|---|---|
| `model` | `models/yolov10n_gestures.q.onnx` | 34 类 HaGRID 手势模型（YOLOv10n end-to-end，`[1,300,6]` 已解码免 NMS，PPQ INT8）。**换模型必须同步换 `classes`**——词表与模型成对，错了是标签错位不是配置报错（self-test 的 class_id 范围检查是唯一防线） |
| `classes` | 34 个类名 | **模型词表**，按模型输出 id 排序（`no_gesture` 是 id 33）。包内零内置词表，`rps_mode`/`filter_no_gesture` 开启时必填 |
| `rps_mode` | `true` | 启用折叠；`false` 按原始 34 类上报（id-only 形态，调试用） |
| `rps_map` | Rock←{fist,grabbing,grip}、Paper←{palm,stop,stop_inverted,four}、Scissors←{peace,peace_inverted,two_up,two_up_inverted} | **折叠表**：游戏标签 ← 源手势。多对一合法；**折叠发生在 C++ 稳定计数之前**——palm↔stop（同为布）的源类抖动不清零稳定计数。折叠后 class_id = 标签声明序（字母序：**Paper=0、Rock=1、Scissors=2**），观测事件同时带 `label` 文本（pipeline 按 label 取手势，不依赖 id） |
| `filter_no_gesture` | `true` | 丢弃 HaGRID 的 `no_gesture` 兜底类（画出来全是噪声） |

**改折叠表的注意**：加源手势（如把 `three` 也算剪刀）只改 `rps_map`；**换词表顺序 = 换 id 映射**，
`classes` 动了要回头核对 `rps_map` 的类名都还在。启动日志会打印解析结果
`RPS mode: Paper<={...} Rock<={...} ...` 供核对。

## 2. 检测参数

| 键 | 当前值 | 说明 |
|---|---|---|
| `conf` | `0.25` | 置信度阈值（模型输出已是概率域）。实测空场景 best_conf≈0.008，真人手势远高于此；**识别不出手时可降到 0.15 试**，误检变多再回调 |
| `stable_frames` | `15` | **折叠后**类别多重集需连续重复的帧数（≈0.7s @24fps 推理）。手势切换的过渡帧会清零计数——玩家要**保持手势不动**到出结果；识别太慢可降到 10，误判提前可升到 20 |
| `yolov10_enabled` | `true` | `false` = 只推流不推理（透视对位用） |

## 3. 画面几何（旋转 + ROI）

| 键 | 当前值 | 说明 |
|---|---|---|
| `rotate` | `{enabled, cw, 90}` | **相机横装**：OpenCL kernel 采样重映射（CPU 零开销）。推流/识别画面为**竖屏 1080x1920**；detection 坐标恒在旋转后坐标系。改 `direction: ccw` 或 `angle: 180` 前先确认相机物理朝向，别用配置"纠正"画面方向 |
| `roi` | `{x:0, y:0.5, w:1, h:0.5}` | **识别区域 = 竖屏画面下半**（y 0.5~1.0）：①模型只看该区域（手部目标放大约 **1.8 倍**，小目标更稳）；②框中心不在区域内的检测丢弃——**上半幅（机械臂位置）被确定性排除**。推流画面画绿框白边供对位（实测绿框在 y=960 像素行）。玩家出手要在**下半画面**内，手放太靠上会恒"未识别" |

**ROI 是"玩家手在画面下半"这一摆位的固化**：如果摄像头/桌子摆位变了，先对位（看推流画面的
ROI 框），再改 x/y/w/h（占竖屏画面比例，越界启动报错）。

## 4. 采集与推理硬件

| 键 | 当前值 | 说明 |
|---|---|---|
| `camera` / `device` | `/dev/video1` / `""` | C920 采集节点（`device` 显式覆盖 `camera`）。**相机换 USB 口后节点号可能变**——改前 `v4l2-ctl --list-devices` 核对；UVC 掉线重连（journalctl -k 看 uvc）是"识别未完成"的常见根因 |
| `width`/`height`/`fps` | 1920/1080/**25** | 采集规格（**旋转前**的横帧）。C920 的 1080p 实际只有 30/24fps 档：25 请求**首次协商失败自动回退 24**（日志一条 `GStreamer-Video-CRITICAL` 是回退痕迹，不是错误）。推理实测 ~24fps、35ms/帧（EP 2 线程） |
| `intra_threads`/`ep_affinity` | 2 / `14;15` | SpaceMIT EP 线程数与绑核，**数量必须一致**（不一致启动报错）。注意板子当前被系统钉在 0-7 核跑（2026-09-20 记录），EP 仍能跑满帧率 |
| `focus` / `zoom` | `-1` / `100` | **`-1` 才是"不动焦距"**（`0` 是真的写 V4L2 对焦值——写失败是硬失败，整台相机起不来）；zoom 100 是该镜头最广端。**只在 runtime 启动（camera open）时写一次** |
| `queue_depth` | `2` | 各级队列深度（latest-only 丢旧保新，防延迟累积） |

## 5. 推流（video / rtsp）

| 键 | 当前值 | 说明 |
|---|---|---|
| `video.webrtc_base_url` | `http://127.0.0.1:8889` | 浏览器画面的 WebRTC 基址（mediamtx）；游戏 manifest 的 `video.enabled` 决定页面是否显示（当前 `false`：页面不显示，推流照常，ffplay 可拉流看） |
| `rtsp.enabled/host/port` | true/127.0.0.1/8554 | runtime 向 mediamtx 推 H.264（VPU 硬编，竖屏 1080x1920） |
| `rtsp.path` | **不在此声明** | 生产由 manifest 的 `vision_profile.video.path: /rps/det` 经 `--rtsp-path` 恒压；写进来是死键（ Dice 侧已清理过同款纪律）。mediamtx 的 `all_others` 规则覆盖 `/rps/det`，无需改服务配置 |

## 6. 常见调参速查

| 想改什么 | 改哪里 | 生效 |
|---|---|---|
| 手势识别不出 / 误检 | `conf`（先降后调） | 下一回合 |
| 识别太慢 / 太快出结果 | `stable_frames`（10~20 之间） | 下一回合 |
| 手总识别不到 | 看推流绿框：手要在**竖屏下半**；不在就改 `roi` 或调相机 | 下一回合 |
| 加/改折叠手势 | `rps_map`（源类名必须在 `classes` 里） | 下一回合（先 self-test） |
| 换模型 | `model` + `classes` 成对换（文件放 `models/`） | 下一回合（必须 self-test） |
| 相机换口后打不开 | `v4l2-ctl --list-devices` 核对节点 → 改 `camera` | 下一回合 |
| 焦距/变焦 | `focus`/`zoom`（`-1`=不动；C270 定焦必须 -1） | 下一回合（runtime 重建时写） |
| 画面方向不对 | 先查相机物理朝向，再改 `rotate` | 下一回合 |

## 7. 改完怎么验证（板上）

```bash
# 1. self-test：词表/折叠/模型/OpenCL 一次全验（不占摄像头）
cd ~/projects/dice-game/main/vision/yolov10_objdetect
./build/yolov10_camera --config ../../backend/games/rps/adjudicator_config.json \
    --self-test --no-rtsp
# 期望：started 事件 + Filtering class_id 33 + RPS mode: Paper/Rock/Scissors<={...}
#       + Rotation: 90 cw + Inference crop: 1080x960 at +0,+960 + Self-test passed

# 2. 真机短跑 60 帧（占摄像头，看检测统计）
./build/yolov10_camera --config ../../backend/games/rps/adjudicator_config.json \
    --max-frames 60 --no-rtsp
# 期望：GStreamer camera opened ... @24 + 竖屏帧 + 你的手势出现在检测里

# 3. 推流目检（绿框/手部框/标签）
ffplay -rtsp_transport tcp rtsp://<板端IP>:8554/rps/det
```

改坏时**下一局裁决失败但游戏不挂**（诊断路径兜底）；词表类错误在 runtime 启动日志里
有具体原因（`rps_map error: ...` / `config classes must be ...`）。
