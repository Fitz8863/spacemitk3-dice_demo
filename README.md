# SpaceMIT K3 Dice Arena

基于 SpacemiT K3（RISC-V）开发板的机械臂对战游戏演示项目，最初为 RV 峰会「机械臂骰子挑战」打造。大屏网页主持对局，NERO 机械臂执行动作，K3 板端摄像头 + YOLO 视觉裁决胜负，本地 TTS 全程语音播报，支持实体按键与语音两种输入。

内置两个游戏：

- **摇骰子**（`dice`）：台面双方各一盅骰子，摇骰开盖后由视觉判定两侧点数总和的胜负；
- **猜拳**（`rps`）：人与机械臂石头剪刀布，视觉识别双方手势。

胜负始终由板端 YOLO 检测产生（纯视觉单出口），网页不生成任何随机结果。

## 系统组成

```text
浏览器（web/，纯静态页面 + 原生 JS，无 Node/npm）
   │  HTTP + SSE —— 后端是唯一权威状态源，前端只提交意图、渲染事件
backend/server.py（K3 板端轻量 HTTP 服务）
   ├─ vision_yolov8_objdetect（dice）/ vision_yolov10_objdetect（rps）  视觉裁决
   ├─ tts_moss_nano / tts_qwen3 / tts_matcha / tts_gptsovits            语音合成
   ├─ asr_sensevoice / asr_zipformer                                    语音识别（可选）
   └─ robot_arm_nero                                                    NERO 机械臂（消费仓内 dice_demo/ 快照）
```

组件都是 `backend/components/` 下的独立功能包，按「职责插槽」被游戏选用（约定见 [`backend/components/README.md`](backend/components/README.md)）。

## 环境要求

| 项 | 说明 |
| --- | --- |
| 硬件 | SpacemiT K3 板（riscv64，内存 ≥ 8G，建议 16G）；USB 摄像头（俯拍台面，接了多台时可用 `scripts/detect.sh` 确认各自的采集节点）；NERO 机械臂（完整对局的动作执行方，**没有也能起服务玩视觉裁决**，见下文免臂模式）；麦克风 / 扬声器（语音功能用，可选） |
| 系统 | Bianbu（自带 python3、SpaceMIT onnxruntime、OpenCL） |
| 系统包 | 由 `scripts/setup_board.sh` 自动安装（OpenCV 运行库与开发包、GStreamer 插件组、v4l-utils、python3-numpy 等；GStreamer 是**硬依赖**——缺了视觉链路起不来） |
| mediamtx | 网页实时画面需要（WebRTC :8889 / RTSP :8554），`setup_board.sh` 默认自动安装并注册开机自启，`--no-mediamtx` 可跳过 |

前端与后端只使用 K3 系统自带的 `python3`，不需要 Node.js 或 npm。

## 快速开始

在 K3 板端执行（下文 `<repo-root>` 指仓库根目录）：

```bash
git clone <本仓库地址>   # 克隆到板端任意目录
cd <repo-root>

scripts/setup_board.sh   # 一键：装依赖 → 编译视觉 runtime → 下载模型资产 → 装 mediamtx → 自检
scripts/start_web.sh     # 启动；无机械臂：DICE_NO_ARM=1 scripts/start_web.sh
```

浏览器打开 `http://<板端IP>:8080`（板端本机即 `http://127.0.0.1:8080`），可用 `curl http://127.0.0.1:8080/api/health` 验证服务状态；停止用 `scripts/stop_web.sh`。

说明：

- `setup_board.sh` 分步幂等（`deps` / `build` / `models` / `mediamtx` / `can0` / `detect` / `check`，可单跑、可失败后重跑续传），模型资产从本仓库 GitHub Release 下载（本地 TTS 约 700M）；语音识别与备用 TTS 引擎是可选项（`--with-asr` / `--with-matcha`）。
- **免臂模式**：没有 NERO 机械臂时用 `DICE_NO_ARM=1 scripts/start_web.sh` 起服务——网页、语音播报、视觉裁决（`/api/adjudicate`）均可用；完整一局需要机械臂（摇骰 / 出拳环节由它执行），对局推进到机械臂步骤时会走失败页。
- 从局域网其他设备经 HTTP 访问时，浏览器可能因非安全上下文限制「页面预览摄像头」权限——实际识别用的始终是 K3 板端摄像头，不受影响。
- 不想用一键脚本、逐项手动部署：编译 `vision/yolov8_objdetect` 与 `vision/yolov10_objdetect`（板上手动装过 `/opt/opencv-spacemit` SDK 的需加 `-DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4`，apt 安装 `libopencv-dev` 的无需）、按各组件 README 放置 TTS/ASR 模型资产、自行部署 mediamtx，最后 `scripts/start_web.sh`。
- 单独调试 TTS：`python3 backend/tts_debug.py <provider_id>`。

## 玩法与按键

以摇骰子为例：游戏列表选择 → 「我明白了」确认规则 → 「开始摇骰」倒计时 → 机械臂摇骰（抓杯、摇、放杯、张手、归位一条链）→ 语音喊「停！」后进入视觉识别 → 胜负播报 → 「再来一局」。

实体按键与网页按钮等价：**绿色 = 确认 / 开始**（Enter），**红色 = 取消 / 返回**（Esc），**蓝色 = 向下选择 / 重听规则**（ArrowDown）。语音输入由全局 `asr_enabled` 开启，开启后说「确认」等触发词等价于按绿色键。

## 配置

配置分三层，均不含任何云服务凭证：

| 层 | 文件 | 生效方式 | 字段文档 |
| --- | --- | --- | --- |
| 全局 | `backend/config.json` | 保存即热加载 | [`backend/参数说明.md`](backend/参数说明.md) |
| 游戏 | `backend/games/<id>/manifest.json`（状态机、台词、触发词、视觉规则） | 热加载，下一局生效 | [`backend/games/dice/参数说明.md`](backend/games/dice/参数说明.md)、[`backend/games/rps/参数说明.md`](backend/games/rps/参数说明.md) |
| 组件 | `backend/components/<id>/config.json` | 需重启后端 | 各组件目录下的 `参数说明.md` |

视觉硬件参数（摄像头 / EP 绑核 / 焦距 / 推流）**每游戏一份**：manifest 的 `vision_profile.runtime_config` 指向该游戏的 `adjudicator_config.json`，且为**整份替换、不做字段级合并**——要做差异就整份复制一份再改。

常用调整：

- 切换 TTS：游戏 manifest 的 `providers.tts` 改为目标组件 id，然后重启（`scripts/stop_web.sh && scripts/start_web.sh`）；
- 语音输入总闸：全局 `asr_enabled`（游戏 manifest 只声明触发词表）；
- 识别画面推流是否常驻：全局 `vision_always_on`。

同一时刻只允许一个视觉 runtime（共用摄像头设备）；切换游戏时会按新参数自动重建，无需手动干预。

## HTTP 接口

```text
GET  /api/health                      后端与各组件状态
GET  /api/components                  组件注册表
POST /api/tts/stream                  文本 → WAV 流式合成
POST /api/game/rounds                 创建对局（正常入口）
POST /api/game/rounds/<id>/intents    提交意图（按键动作 / speech_done 回执）
GET  /api/game/rounds/<id>/stream     SSE 对局事件流（状态/台词/倒计时/裁决）
POST /api/game/rounds/<id>/cancel     取消对局（刷新或关闭页面即自动取消）
POST /api/adjudicate                  调试入口：直接发起一轮视觉裁决
GET  /api/adjudicate/<job_id>/stream  裁决进度与结果（SSE）
```

请求从浏览器到各 provider 的完整调度说明见 [`FRAMEWORK_DISPATCH.md`](FRAMEWORK_DISPATCH.md)。

## 机械臂子系统（dice_demo/）

`dice_demo/` 是上游协作仓的快照子目录（`git subtree --squash` 合入），**内部文件不可在本仓直接修改**，改动一律在上游仓提交后再同步。手眼标定数据绑定具体物理机器，更换机械臂或相机安装后需重新标定。

接入契约见 [`dice_demo/docs/INTEGRATION.md`](dice_demo/docs/INTEGRATION.md)；本仓侧组件说明见 [`backend/components/robot_arm_nero/参数说明.md`](backend/components/robot_arm_nero/参数说明.md)。

## 文档索引

- [`docs/README.md`](docs/README.md) —— 全部文档索引（含各组件 `参数说明.md` 清单）
- [`FRAMEWORK_DISPATCH.md`](FRAMEWORK_DISPATCH.md) —— 端到端请求调度
- [`GLOSSARY.md`](GLOSSARY.md) —— 项目术语表
- 想新增一个视觉游戏：新增 `backend/games/<id>/` 目录（manifest + 运行配置 + `pipeline.py` 薄壳 + 结果投影）和 `web/games/<id>.js` 即可，`backend/core/` 不需要改。
