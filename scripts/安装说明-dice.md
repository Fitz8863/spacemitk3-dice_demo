# Dice Arena 安装说明（主包）

摇骰子对战 Demo：板端网页 + 语音播报（本地 TTS）+ 可选语音控制 + YOLOv8 骰子识别裁决。
本包面向 **SpacemiT K3 (riscv64) + Bianbu 系统**，解压即用，**不需要 git、不需要联网**。

## 前置条件

1. 同规格 K3 板（riscv64，内存 ≥ 8G 建议 16G）；
2. Bianbu 系统（自带 SpaceMIT onnxruntime、OpenCL、python3）；
3. **先安装 mediamtx 小包**（骰子识别实时画面依赖它）：解压 `mediamtx-bundle-*.tar` 后运行其中的 `install.sh`；
4. USB 摄像头已接好（识别用，画面正对骰盘）；
5. 麦克风/扬声器按需连接（语音功能用；在系统桌面声音设置里选好默认输入/输出设备）。

## 安装（3 步）

```bash
tar -xf dice-arena-bundle-*.tar
cd dice-arena
./install.sh
```

install.sh 会自动完成：系统依赖自检 → 包完整性自检 → mediamtx 探测 → 摄像头设备提示 → 启动服务 → 打印各组件健康状态。

启动成功后，板载浏览器（或本机任意浏览器）打开：

```
http://127.0.0.1:8080
```

## 日常操作

```bash
# 停止 / 再启动（在 dice-arena/ 目录）
scripts/stop_web.sh
scripts/start_web.sh

# 日志
tail -f .runtime/web-8080.log
```

## 常见问题

**语音控制没反应？**
语音总闸默认随包发布（`backend/config.json` 的 `asr_enabled`，`true` 开 / `false` 关）。
确认它是 `true`，改后刷新页面即生效。游戏内可用语句见游戏页面提示。

**网页里没有骰子识别画面？**
mediamtx 没跑。`systemctl --user status mediamtx` 检查，没装就先装 mediamtx 小包，然后 `scripts/stop_web.sh && scripts/start_web.sh` 重启本服务。

**摄像头识别不到？**
`ls /dev/video*` 与 `v4l2-ctl --list-devices` 找到 USB 摄像头的设备号，修改
`vision/yolov8_adjudicator/config.json` 的 `camera` 字段（默认 `/dev/video1`），重启服务。

**摄像头型号与源部署机不同时（首块新板实测经验）**，还需按相机能力调整
`vision/yolov8_adjudicator/config.json` 以下字段：
- `fps`：引擎对 720p@25 有 24fps 特判（源机 C920 通告 24fps）；相机只通告
  25/30fps 时改 `30`；
- `focus` / `zoom`：固定焦距相机（如 C270）没有这两个控制，改 `-1` 跳过设置
  （源机 C920 的值是 `0` / `160`，原值会让相机打开直接失败）。

**网页没有骰子识别实时画面？**
该画面依赖 RTSP 推流（VPU 硬编码）。若本机 VPU 编码设备与源部署机不一致
（引擎日志出现 MPP-ERROR 刷屏后崩溃），可把 `vision/yolov8_adjudicator/config.json`
的 `rtsp.enabled` 改为 `false` —— 裁决识别不受影响，仅网页画面缺失。

**健康检查里 tts_gptsovits 显示异常？**
预期现象。那是远程 TTS 引擎槽位，分发环境没有对应远程服务；摇骰子台词全部使用本地 MOSS 引擎，不受影响。

**LLM 大模型复核？**
分发版默认关闭（YOLO 识别结果直接生效）。如需启用，在 `backend/games/dice/manifest.json` 的 `llm` 节配置你自己的 OpenAI 兼容服务与 key。

## 硬件摆位

识别模型对骰盘摆放方式有要求：左右两侧骰子区域对称、光照均匀、摄像头俯视覆盖整个骰盘。
换桌面/换摆位后识别质量下降属正常现象，需要按新场景重新适配。
