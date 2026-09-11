#!/usr/bin/env bash
# Dice Arena 主包安装脚本 —— 在解压出来的 dice-arena/ 目录里运行。
# 依赖自检 → mediamtx 探测 → 原地启动 → 健康验收。
#
# 用法: ./install.sh            (自检 + 启动 + 验收)
#       DICE_NO_START=1 ./install.sh   (只自检不启动, 用于检查环境)
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8080}"

say()  { echo "install: $*"; }
warn() { echo "install: [警告] $*" >&2; }
die()  { echo "install: [错误] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. 系统依赖自检
# ---------------------------------------------------------------------------
[[ "$(uname -m)" == "riscv64" ]] || die "本包面向 SpacemiT K3 (riscv64), 当前架构 $(uname -m)"
command -v python3 >/dev/null || die "缺少 python3 (Bianbu 应自带)"

# 系统包清单分两组:
#   BASE_PKGS  —— 推理与音频/工具链 (缺了组件起不来)
#   GST_PKGS   —— 视觉链路的 GStreamer 插件。整条链是硬依赖: rtsp_streamer
#                 启动失败时 yolov8_camera 直接 return 8 退出, 于是裁决和
#                 网页画面一起失效 (不只是"没有实时画面")。
#                 x264enc 属 plugins-ugly, 只在这块板没有 VPU 时才会被用到,
#                 但缺了它软件回退路径就断, 所以同样列为必需。
BASE_PKGS=(
    spacemit-onnxruntime libsndfile1 alsa-utils curl
    libopencv-core410 libopencv-imgproc410 libopencv-imgcodecs410 libopencv-highgui410
    v4l-utils
)
GST_PKGS=(
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-rtsp
)

MISSING_PKGS=()
for pkg in "${BASE_PKGS[@]}" "${GST_PKGS[@]}"; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING_PKGS+=("$pkg")
done
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    echo "install: 缺少系统包: ${MISSING_PKGS[*]}"
    echo "  安装命令: sudo apt-get update && sudo apt-get install -y ${MISSING_PKGS[*]}"
    # 全新环境的 apt 包列表可能为空 (首块新板实测), 安装前必须刷新索引。
    # 终端交互环境里提供自动安装 (sudo 可能提示密码);
    # 非交互环境 (管道/CI) 退回手动模式。
    if [[ -t 0 && -t 1 ]]; then
        reply=""
        read -r -p "install: 现在自动安装这些系统包? [Y/n] " reply || true
        if [[ ! "$reply" =~ ^[Nn] ]]; then
            sudo apt-get update \
                || echo "install: [警告] apt-get update 失败, 继续尝试用现有索引安装" >&2
            sudo apt-get install -y "${MISSING_PKGS[@]}" \
                || die "自动安装失败, 请手动执行上面的安装命令后重跑 install.sh"
            say "系统包安装完成"
        else
            die "请手动安装后重跑 install.sh"
        fi
    else
        die "非交互环境, 请手动执行上面的安装命令后重跑 install.sh"
    fi
fi
say "系统依赖自检通过 (riscv64 / python3 / onnxruntime-EP / sndfile / alsa / curl / OpenCV / v4l2-ctl)"

# ---------------------------------------------------------------------------
# 1b. GStreamer 元素级探测
# ---------------------------------------------------------------------------
# 查元素比查包名更贴近真实可用性: 发行版可能把插件打进别的包里, 包在而元素
# 不在的情况也真实存在 (模块被裁剪)。这里只探测, 不再触发安装 —— 上面那步
# 已把插件包装齐, 走到这里还缺就说明是环境异常, 直接报出来比默默启动好。
if ! command -v gst-inspect-1.0 >/dev/null 2>&1; then
    die "缺少 gst-inspect-1.0, 无法校验视觉链路 (应随 gstreamer1.0-tools 安装)"
fi

# 格式: 元素|用途|提供包
GST_REQUIRED=(
    "v4l2src|摄像头采集|gstreamer1.0-plugins-good"
    "jpegdec|MJPEG 解码|gstreamer1.0-plugins-good"
    "videoconvert|色彩转换|gstreamer1.0-plugins-base"
    "appsink|取帧|gstreamer1.0-plugins-base"
    "appsrc|送帧|gstreamer1.0-plugins-base"
    "queue|推流缓冲|gstreamer1.0-plugins-base"
    "h264parse|H.264 解析|gstreamer1.0-plugins-bad"
    "x264enc|软件编码回退|gstreamer1.0-plugins-ugly"
    "rtspclientsink|RTSP 推流|gstreamer1.0-rtsp"
)
MISSING_ELEMENTS=()
for entry in "${GST_REQUIRED[@]}"; do
    element="${entry%%|*}"
    rest="${entry#*|}"
    gst-inspect-1.0 "$element" >/dev/null 2>&1 || MISSING_ELEMENTS+=("$element (${rest%%|*}) ← 需要 ${rest##*|}")
done
if [[ ${#MISSING_ELEMENTS[@]} -gt 0 ]]; then
    echo "install: [错误] 视觉链路缺少 GStreamer 元素:" >&2
    for item in "${MISSING_ELEMENTS[@]}"; do echo "    - $item" >&2; done
    echo "  装齐上面的插件包后重跑 install.sh; 排查: gst-inspect-1.0 <元素名>" >&2
    die "yolov8_camera 会因推流链路启动失败而直接退出 (裁决与画面一起失效)"
fi

# VPU 硬解/硬编只在有对应硬件的板子上存在, 缺失时引擎会自动回退到软件路径
# (jpegdec / x264enc), 因此只提示不阻塞。
GST_OPTIONAL=(spacemitdec spacemith264enc)
missing_optional=()
for element in "${GST_OPTIONAL[@]}"; do
    gst-inspect-1.0 "$element" >/dev/null 2>&1 || missing_optional+=("$element")
done
if [[ ${#missing_optional[@]} -gt 0 ]]; then
    warn "未发现 VPU 编解码元素: ${missing_optional[*]} —— 将使用软件路径 (jpegdec/x264enc), 功能不受影响。"
fi
say "视觉链路自检通过 (采集/解码/转换/编码/RTSP 推流元素齐全)"

# ---------------------------------------------------------------------------
# 2. 包内资产自检 (确保拿到的包是完整的)
# ---------------------------------------------------------------------------
KEY_FILES=(
    asr/sensevoice/build/bin/asr_pipe_demo
    vision/yolov8_adjudicator/build/yolov8_camera
    backend/components/tts_moss_nano/launcher.py
)
for f in "${KEY_FILES[@]}"; do
    [[ -e "$BUNDLE_DIR/$f" ]] || die "包不完整, 缺少: $f"
done
[[ -x "$BUNDLE_DIR/${KEY_FILES[0]}" ]] || die "引擎二进制不可执行: ${KEY_FILES[0]}"
[[ -x "$BUNDLE_DIR/${KEY_FILES[1]}" ]] || die "引擎二进制不可执行: ${KEY_FILES[1]}"
ls "$BUNDLE_DIR"/asr/sensevoice/model/*.onnx >/dev/null 2>&1 \
    || die "包不完整, 缺少 SenseVoice 模型"
ls "$BUNDLE_DIR"/tts/moss-tts-nano/models/*/ >/dev/null 2>&1 \
    || die "包不完整, 缺少 MOSS 模型"
# MOSS 除模型外还要 runtime 依赖树: python/ 是 riscv64 依赖, lib/ 是 native
# 库, voice/ 是内置音色参考音频, assets/ 是 manifest 引用的小资产。
# 四者缺一 daemon 都起不来, 而它们都不在 git 里, 靠打包时的 rsync 带进来,
# 是最容易在换源板时漏掉的部分, 所以逐个校验。
for d in python lib voice assets; do
    [[ -d "$BUNDLE_DIR/tts/moss-tts-nano/$d" ]] \
        || die "包不完整, 缺少 MOSS runtime 目录: tts/moss-tts-nano/$d"
done
say "包完整性自检通过 (ASR/YOLO 引擎与模型、MOSS 模型与 runtime 依赖树均在位)"

# ---------------------------------------------------------------------------
# 3. mediamtx 探测 (画面推流依赖它; 缺失时裁决仍可跑, 但网页没有实时画面)
# ---------------------------------------------------------------------------
if curl -fsS --max-time 2 "http://127.0.0.1:8889" >/dev/null 2>&1 \
    || ss -ltn "sport = :8889" 2>/dev/null | grep -q LISTEN; then
    say "mediamtx 已在运行 (127.0.0.1:8889)"
else
    warn "mediamtx 未在运行 —— 网页将没有骰子识别实时画面。"
    warn "请先安装 mediamtx 小包: tar -xf mediamtx-bundle-*.tar && mediamtx-bundle/install.sh"
    warn "继续启动 (YOLO 裁决不受影响, 但建议装好 mediamtx 后重启本服务)。"
fi

# 摄像头提示 (设备号因外设而异, 提前给出排查入口)
CAM="$(python3 -c "import json; print(json.load(open('$BUNDLE_DIR/vision/yolov8_adjudicator/config.json'))['camera'])")"
if [[ ! -e "$CAM" ]]; then
    warn "配置的摄像头设备 $CAM 当前不存在。"
    warn "用 'ls /dev/video*' 或 'v4l2-ctl --list-devices' 找到 USB 摄像头后,"
    warn "修改 vision/yolov8_adjudicator/config.json 的 camera 字段再启动。"
fi

[[ "${DICE_NO_START:-0}" == "1" ]] && { say "DICE_NO_START=1, 自检到此结束, 未启动服务。"; exit 0; }

# ---------------------------------------------------------------------------
# 4. 原地启动 (本项目无绝对路径绑定, 解压在哪就在哪跑)
# ---------------------------------------------------------------------------
say "启动 Dice Arena (板载浏览器打开 http://127.0.0.1:${PORT}) ..."
bash "$BUNDLE_DIR/scripts/start_web.sh"

# ---------------------------------------------------------------------------
# 5. 健康验收
# ---------------------------------------------------------------------------
say "服务健康状态:"
python3 - "http://127.0.0.1:${PORT}" "$BUNDLE_DIR" <<'PY'
import json, sys, urllib.request
h = json.load(urllib.request.urlopen(sys.argv[1] + "/api/health", timeout=8))
# 只报告选中的槽位与关键组件 —— 未选槽的备用引擎不跑属正常, 逐条列
# "异常"只会误导。
selected = {h.get("tts_provider"), h.get("tts_remote_provider"),
            "asr_sensevoice", "vision_yolov8_adjudicator", "llm_openai_compat"}
idle, shown = [], []
try:
    arena = json.load(open(f"{sys.argv[2]}/backend/config.json"))
except OSError:
    arena = {}
for c in h.get("components", []):
    if c["id"] not in selected:
        idle.append(c["id"])
        continue
    health = c.get("health", {})
    if "running" in health:
        ok = health["running"]
    else:
        ok = health.get("ok", health.get("configured", False))
    mark = "OK " if ok else "异常"
    # ASR 引擎不跑且全局语音总闸是关的 → 出厂状态, 不是故障。
    if c["id"] == "asr_sensevoice" and not ok and not arena.get("asr_enabled", True):
        mark = "关闭"
    shown.append(f"    [{mark}] {c['id']}")
print("\n".join(shown))
if idle:
    print(f"    (未启用引擎, 属正常: {', '.join(sorted(idle))})")
if not arena.get("asr_enabled", True):
    print("    注: 语音控制出厂关闭 (backend/config.json 的 asr_enabled=false),")
    print("        需要语音时改回 true 并重启服务。")
print(f"    tts_ready={h.get('tts_ready')}  yolo={h.get('vision', {}).get('ok')}")
if h.get("tts_remote_provider"):
    print("    注: 远程 TTS 槽位的引擎在分发环境不可达属预期 (无对应远程服务),")
    print("        摇骰子台词全部使用本地 MOSS 引擎, 不受影响。")
PY

say "安装完成。常用操作:"
say "  停止:   $(dirname "$BUNDLE_DIR")/dice-arena/scripts/stop_web.sh"
say "  再启动: $(dirname "$BUNDLE_DIR")/dice-arena/scripts/start_web.sh"
say "  日志:   $BUNDLE_DIR/.runtime/web-${PORT}.log"
say "  语音开关 (默认随包): backend/config.json 的 asr_enabled"
