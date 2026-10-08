#!/usr/bin/env bash
# Dice Arena 源码一键部署 —— git clone 后在 K3 板端运行本脚本，
# 从裸系统到可启动：装依赖 → 编译视觉 runtime → 下载模型资产 →
# 安装 mediamtx → 摄像头/CAN 提示 → 自检。
#
# 用法:
#   scripts/setup_board.sh                     # 全部步骤顺序执行
#   scripts/setup_board.sh check               # 只跑指定步骤（可多个: deps build models mediamtx can0 detect check）
#   scripts/setup_board.sh --with-asr          # 附带语音识别引擎（编译 + 模型，默认不带）
#   scripts/setup_board.sh --with-matcha       # 附带 matcha 备用本地 TTS 资产（默认不带）
#   scripts/setup_board.sh --no-mediamtx       # 跳过 mediamtx 安装（无实时画面小窗，识别不受影响）
#
# 可用环境变量覆盖下载地址（默认指向本仓库与官方模型站的 Release）:
#   DICE_ASSETS_BASE_URL   模型资产 Release 基址
#   DICE_MEDIAMTX_URL      mediamtx riscv64 tar 包地址
#   MEDIAMTX_DIR           mediamtx 安装目录（默认 ~/projects/mediamtx）
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="${DICE_DOWNLOAD_DIR:-$ROOT_DIR/.runtime/downloads}"

ASSETS_BASE_URL="${DICE_ASSETS_BASE_URL:-"https://github.com/Fitz8863/spacemitk3-dice_demo/releases/download/board-assets-v1"}"
MOSS_ASSET="moss-tts-nano-board-assets.tar.gz"
MATCHA_ASSET="matcha-tts-board-assets.tar.gz"
MEDIAMTX_URL="${DICE_MEDIAMTX_URL:-"https://github.com/Fitz8863/spacemit-mediamtx/releases/download/v1.20.1-riscv64/mediamtx_v1.20.1_linux_riscv64.tar.gz"}"
SENSEVOICE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/asr/sensevoice.tar.gz"
MEDIAMTX_DIR="${MEDIAMTX_DIR:-$HOME/projects/mediamtx}"

say()  { echo "setup: $*"; }
warn() { echo "setup: [警告] $*" >&2; }
die()  { echo "setup: [错误] $*" >&2; exit 1; }

WITH_ASR=0
WITH_MATCHA=0
NO_MEDIAMTX=0
STEPS=()

for arg in "$@"; do
    case "$arg" in
        --with-asr)    WITH_ASR=1 ;;
        --with-matcha) WITH_MATCHA=1 ;;
        --no-mediamtx) NO_MEDIAMTX=1 ;;
        -h|--help)     awk 'NR>1 && /^set -euo/ {exit} NR>1 {print}' "${BASH_SOURCE[0]}"; exit 0 ;;
        deps|build|models|mediamtx|can0|detect|check) STEPS+=("$arg") ;;
        *) die "未知参数: $arg（--help 查看用法）" ;;
    esac
done
# 未指定步骤 = 全部顺序执行
if [[ ${#STEPS[@]} -eq 0 ]]; then
    STEPS=(deps build models mediamtx can0 detect check)
fi
has_step() { [[ " ${STEPS[*]} " == *" $1 "* ]]; }

# ---------------------------------------------------------------------------
# deps: 系统包（apt）
# ---------------------------------------------------------------------------
step_deps() {
    [[ "$(uname -m)" == "riscv64" ]] || die "本项目面向 SpacemiT K3 (riscv64)，当前架构 $(uname -m)"
    command -v python3 >/dev/null || die "缺少 python3（Bianbu 应自带）"

    # 分三组:
    #   运行库组 —— 组件运行必需（推理 EP / 音频 / 摄像头工具 / numpy 是后端唯一第三方 Python 依赖）
    #   编译组   —— 从源码编译视觉 runtime 必需（OpenCV 开发包同时提供 410 运行库）
    #   GST 组   —— 视觉链路的 GStreamer 插件。整条链是硬依赖: rtsp_streamer
    #               启动失败时 yolov8_camera 直接退出，裁决和网页画面一起失效。
    RUN_PKGS=(
        spacemit-onnxruntime libsndfile1 alsa-utils curl v4l-utils python3-numpy
    )
    BUILD_PKGS=(
        cmake build-essential pkg-config libopencv-dev
        ocl-icd-opencl-dev opencl-clhpp-headers
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
    )
    GST_PKGS=(
        gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-rtsp
    )
    local pkgs=("${RUN_PKGS[@]}" "${BUILD_PKGS[@]}" "${GST_PKGS[@]}")
    if [[ $WITH_ASR == 1 ]]; then
        pkgs+=(libsndfile1-dev libfftw3-dev libcurl4-openssl-dev)
    fi

    local missing=()
    for pkg in "${pkgs[@]}"; do
        dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "setup: 缺少系统包: ${missing[*]}"
        echo "  安装命令: sudo apt-get update && sudo apt-get install -y ${missing[*]}"
        # 全新环境 apt 索引可能为空，安装前刷新；交互终端提供自动安装，非交互退回手动。
        if [[ -t 0 && -t 1 ]]; then
            local reply=""
            read -r -p "setup: 现在自动安装这些系统包? [Y/n] " reply || true
            if [[ ! "$reply" =~ ^[Nn] ]]; then
                sudo apt-get update \
                    || warn "apt-get update 失败，继续尝试用现有索引安装"
                sudo apt-get install -y "${missing[@]}" \
                    || die "自动安装失败，请手动执行上面的安装命令后重跑本脚本"
            else
                die "请手动安装后重跑本脚本"
            fi
        else
            die "非交互环境，请手动执行上面的安装命令后重跑本脚本"
        fi
    fi
    say "deps 通过（运行库 / 编译工具链 / GStreamer 插件均在位）"

    # 元素级探测：查元素比查包名更贴近真实可用性（发行版可能裁剪模块）。
    # 包刚装齐还缺元素属环境异常，直接报出来比默默编译好再在运行期炸掉好。
    command -v gst-inspect-1.0 >/dev/null 2>&1 \
        || die "缺少 gst-inspect-1.0（应随 gstreamer1.0-tools 安装）"
    local -a gst_required=(
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
    local -a missing_elements=()
    for entry in "${gst_required[@]}"; do
        element="${entry%%|*}"
        gst-inspect-1.0 "$element" >/dev/null 2>&1 \
            || missing_elements+=("$element")
    done
    if [[ ${#missing_elements[@]} -gt 0 ]]; then
        echo "setup: [错误] 视觉链路缺少 GStreamer 元素: ${missing_elements[*]}" >&2
        die "yolov8_camera 会因推流链路失败直接退出（裁决与画面一起失效）；排查: gst-inspect-1.0 <元素名>"
    fi
    # VPU 硬解硬编只在部分板子存在，缺失时引擎自动走软件路径，只提示不阻塞。
    local -a missing_optional=()
    for element in spacemitdec spacemith264enc; do
        gst-inspect-1.0 "$element" >/dev/null 2>&1 || missing_optional+=("$element")
    done
    [[ ${#missing_optional[@]} -eq 0 ]] \
        || warn "未发现 VPU 编解码元素: ${missing_optional[*]} —— 将用软件路径，功能不受影响"
}

# ---------------------------------------------------------------------------
# build: 编译视觉 runtime（+ 可选 ASR 引擎）
# ---------------------------------------------------------------------------
step_build() {
    # 源板上手动装的 OpenCV SDK 优先于 apt 版（保持源板行为一致）；
    # 常规新板走 apt libopencv-dev 的标准搜索路径。
    local opencv_args=()
    if [[ -d /opt/opencv-spacemit/lib/cmake/opencv4 ]]; then
        opencv_args=(-DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4)
    fi
    local jobs
    jobs="$(nproc)"

    for pkg_dir in vision/yolov8_objdetect vision/yolov10_objdetect; do
        say "编译 $pkg_dir ..."
        cmake -S "$pkg_dir" -B "$pkg_dir/build" -DCMAKE_BUILD_TYPE=Release "${opencv_args[@]}"
        cmake --build "$pkg_dir/build" -j"$jobs"
    done
    if [[ $WITH_ASR == 1 ]]; then
        say "编译 asr/sensevoice（--with-asr）..."
        cmake -S asr/sensevoice -B asr/sensevoice/build -DCMAKE_BUILD_TYPE=Release
        cmake --build asr/sensevoice/build -j"$jobs"
    fi
    local built="vision/yolov8_objdetect、vision/yolov10_objdetect"
    if [[ $WITH_ASR == 1 ]]; then built="$built、asr/sensevoice"; fi
    say "build 完成（$built）"
}

# ---------------------------------------------------------------------------
# models: 下载模型资产（git 不带的部分）
# ---------------------------------------------------------------------------
fetch_verified() {  # fetch_verified <文件名> <URL>
    local name="$1" url="$2"
    mkdir -p "$DOWNLOAD_DIR"
    if [[ ! -f "$DOWNLOAD_DIR/$name" ]]; then
        say "下载 $name ..."
        curl -fL --retry 3 -C - -o "$DOWNLOAD_DIR/$name" "$url" \
            || die "下载失败: $url（可稍后重跑本脚本续传；或用 DICE_ASSETS_BASE_URL 覆盖地址）"
    fi
    # .sha256 同源发布，随下随验；校验失败删掉重下，避免半截文件续传掩盖损坏。
    # 侧车里记录的可能是原始文件名（如 mediamtx 官方包名）而非本地保存名，
    # 所以只取哈希值自拼校验行，不直接 `sha256sum -c` 整个侧车。
    if [[ ! -f "$DOWNLOAD_DIR/$name.sha256" ]]; then
        curl -fL --retry 3 -o "$DOWNLOAD_DIR/$name.sha256" "$url.sha256" || true
    fi
    if [[ -f "$DOWNLOAD_DIR/$name.sha256" ]]; then
        local expected
        expected="$(awk '{print $1}' "$DOWNLOAD_DIR/$name.sha256")"
        if [[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]]; then
            ( cd "$DOWNLOAD_DIR" && echo "$expected  $name" | sha256sum -c - ) \
                || { rm -f "$DOWNLOAD_DIR/$name"; die "$name 校验失败，已删除请重跑（断点续传损坏或发布物更新）"; }
        else
            warn "$name.sha256 内容不是 64 位哈希，跳过校验"
        fi
    else
        warn "未找到 $name 的 .sha256，跳过校验"
    fi
}

step_models() {
    # MOSS（默认本地 TTS，必需）: 模型 + riscv64 依赖树，解压到 tts/moss-tts-nano/
    if [[ -d "$ROOT_DIR/tts/moss-tts-nano/models/MOSS-TTS-Nano-100M-ONNX-xslim-dynq" ]]; then
        say "MOSS 资产已存在，跳过（强制重装: 删除 tts/moss-tts-nano/{models,python,lib,voice,assets} 后重跑）"
    else
        fetch_verified "$MOSS_ASSET" "$ASSETS_BASE_URL/$MOSS_ASSET"
        say "解压 MOSS 资产到 tts/moss-tts-nano/ ..."
        tar -xzf "$DOWNLOAD_DIR/$MOSS_ASSET" -C "$ROOT_DIR/tts/moss-tts-nano"
        for d in models python lib voice assets; do
            [[ -d "$ROOT_DIR/tts/moss-tts-nano/$d" ]] || die "MOSS 资产包不完整，缺少目录: $d"
        done
    fi

    # matcha（备用本地 TTS，--with-matcha）: 解压到 tts/matcha-tts/
    if [[ $WITH_MATCHA == 1 ]]; then
        if [[ -f "$ROOT_DIR/tts/matcha-tts/matcha-model/model-steps-3.q.onnx" ]]; then
            say "matcha 资产已存在，跳过"
        else
            fetch_verified "$MATCHA_ASSET" "$ASSETS_BASE_URL/$MATCHA_ASSET"
            say "解压 matcha 资产到 tts/matcha-tts/ ..."
            tar -xzf "$DOWNLOAD_DIR/$MATCHA_ASSET" -C "$ROOT_DIR/tts/matcha-tts"
        fi
    fi

    # SenseVoice（语音识别，--with-asr）: 官方模型站，放到组件约定的仓内目录
    if [[ $WITH_ASR == 1 ]]; then
        if [[ -f "$ROOT_DIR/asr/sensevoice/model/model_quant_optimized.onnx" ]]; then
            say "SenseVoice 模型已存在，跳过"
        else
            fetch_verified "sensevoice.tar.gz" "$SENSEVOICE_URL"
            say "解压 SenseVoice 模型到 asr/sensevoice/model/ ..."
            local tmp="$DOWNLOAD_DIR/sensevoice-extract"
            rm -rf "$tmp" && mkdir -p "$tmp"
            tar -xzf "$DOWNLOAD_DIR/sensevoice.tar.gz" -C "$tmp"
            # 官方包内层目录结构不定，以 model_quant_optimized.onnx 所在目录为准整个搬入
            local model_src
            model_src="$(dirname "$(find "$tmp" -name model_quant_optimized.onnx -print -quit)")"
            [[ -n "$model_src" ]] || die "sensevoice.tar.gz 内未找到 model_quant_optimized.onnx"
            mkdir -p "$ROOT_DIR/asr/sensevoice/model"
            cp -a "$model_src"/. "$ROOT_DIR/asr/sensevoice/model/"
            rm -rf "$tmp"
        fi
    fi
    say "models 完成（语音总闸 asr_enabled 默认 false，需要语音时改 backend/config.json 并重启）"
}

# ---------------------------------------------------------------------------
# mediamtx: 实时画面推流服务（网页小窗）
# ---------------------------------------------------------------------------
step_mediamtx() {
    if [[ $NO_MEDIAMTX == 1 ]]; then
        say "按 --no-mediamtx 跳过（识别与裁决不受影响，仅网页无实时画面小窗）"
        return 0
    fi
    if curl -fsS --max-time 2 "http://127.0.0.1:8889" >/dev/null 2>&1 \
        || ss -ltn "sport = :8889" 2>/dev/null | grep -q LISTEN; then
        say "mediamtx 已在运行（127.0.0.1:8889），跳过"
        return 0
    fi

    fetch_verified "mediamtx_riscv64.tar.gz" "$MEDIAMTX_URL"
    say "安装 mediamtx 到 $MEDIAMTX_DIR ..."
    if [[ -e "$MEDIAMTX_DIR" ]] && [[ ! -x "$MEDIAMTX_DIR/bin/mediamtx" ]]; then
        die "$MEDIAMTX_DIR 已存在但不是 mediamtx 安装目录，请手动处理或用 MEDIAMTX_DIR 指定别处"
    fi
    mkdir -p "$MEDIAMTX_DIR/bin" "$MEDIAMTX_DIR/config"
    local tmp="$DOWNLOAD_DIR/mediamtx-extract"
    rm -rf "$tmp" && mkdir -p "$tmp"
    tar -xzf "$DOWNLOAD_DIR/mediamtx_riscv64.tar.gz" -C "$tmp"
    # 官方包带一层版本目录（mediamtx_v1.20.1_linux_riscv64/），按名字定位不猜路径
    local mtx_bin mtx_yml
    mtx_bin="$(find "$tmp" -type f -name mediamtx -print -quit)"
    mtx_yml="$(find "$tmp" -type f -name mediamtx.yml -print -quit)"
    [[ -n "$mtx_bin" ]] || die "mediamtx 包内未找到 mediamtx 可执行文件"
    cp -a "$mtx_bin" "$MEDIAMTX_DIR/bin/mediamtx"
    chmod +x "$MEDIAMTX_DIR/bin/mediamtx"
    # 已有配置不覆盖（现场可能改过端口/路径）
    if [[ ! -f "$MEDIAMTX_DIR/config/mediamtx.yml" ]]; then
        [[ -n "$mtx_yml" ]] && cp -a "$mtx_yml" "$MEDIAMTX_DIR/config/mediamtx.yml"
    fi
    rm -rf "$tmp"

    say "注册 systemd --user 服务 ..."
    mkdir -p "$HOME/.config/systemd/user"
    cat > "$HOME/.config/systemd/user/mediamtx.service" <<UNIT
[Unit]
Description=MediaMTX RTSP and WebRTC server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$MEDIAMTX_DIR
ExecStart=$MEDIAMTX_DIR/bin/mediamtx $MEDIAMTX_DIR/config/mediamtx.yml
Restart=on-failure
RestartSec=2
LimitNOFILE=65536

[Install]
WantedBy=default.target
UNIT
    systemctl --user daemon-reload
    systemctl --user enable --now mediamtx
    sleep 1
    systemctl --user is-active --quiet mediamtx \
        || die "mediamtx 服务未正常运行，查看: systemctl --user status mediamtx"
    say "mediamtx 已运行并开机自启（WebRTC :8889 / RTSP :8554）"
}

# ---------------------------------------------------------------------------
# can0: 机械臂 CAN 提示（不阻塞）
# ---------------------------------------------------------------------------
step_can0() {
    if [[ -d /sys/class/net/can0 ]]; then
        if ip link show can0 2>/dev/null | grep -q 'state UP'; then
            say "can0 已就绪（机械臂可用）"
        else
            warn "can0 存在但未启用。插了机械臂请执行: sudo ip link set can0 up type can bitrate 1000000"
        fi
    else
        say "未检测到 can0 接口。"
        say "  · 没有机械臂: 用免臂模式启动 —— DICE_NO_ARM=1 scripts/start_web.sh"
        say "  · 有机械臂:   插好 USB-CAN 后重跑本步骤，启动前拉起 can0"
    fi
}

# ---------------------------------------------------------------------------
# detect: 摄像头采集节点提示
# ---------------------------------------------------------------------------
step_detect() {
    bash "$ROOT_DIR/scripts/detect.sh" || warn "摄像头探测未通过，接好 USB 摄像头后重跑: scripts/setup_board.sh detect"
    say "请核对 backend/games/{dice,rps}/adjudicator_config.json 的 camera 值与上面列出的采集节点一致"
}

# ---------------------------------------------------------------------------
# check: 引擎自测 + 资产核验 + 启动指引
# ---------------------------------------------------------------------------
step_check() {
    local y8="$ROOT_DIR/vision/yolov8_objdetect/build/yolov8_camera"
    local y10="$ROOT_DIR/vision/yolov10_objdetect/build/yolov10_camera"
    [[ -x "$y8" ]] || die "缺少 yolov8_camera，请先跑 build 步骤"
    [[ -x "$y10" ]] || die "缺少 yolov10_camera，请先跑 build 步骤"

    say "yolov8 自测（OpenCL + 模型 + 一次推理）..."
    "$y8" --config "$ROOT_DIR/backend/games/dice/adjudicator_config.json" \
        --model "$ROOT_DIR/backend/games/dice/models/best.q.onnx" \
        --yolov8 --self-test --no-display \
        || die "yolov8 自测失败（模型/EP/OpenCL 任一异常都会在这里暴露）"
    say "yolov10 自测..."
    "$y10" --config "$ROOT_DIR/backend/games/rps/adjudicator_config.json" \
        --self-test --no-rtsp \
        || die "yolov10 自测失败"

    for d in python lib voice assets; do
        [[ -d "$ROOT_DIR/tts/moss-tts-nano/$d" ]] || die "MOSS runtime 目录缺失: tts/moss-tts-nano/$d（重跑 models 步骤）"
    done
    ls "$ROOT_DIR"/tts/moss-tts-nano/models/*/ >/dev/null 2>&1 || die "MOSS 模型缺失（重跑 models 步骤）"

    if curl -fsS --max-time 2 "http://127.0.0.1:8889" >/dev/null 2>&1; then
        say "mediamtx 在线"
    else
        warn "mediamtx 未运行 —— 网页将没有实时画面（识别不受影响）"
    fi

    echo
    say "自检全部通过。启动服务:"
    say "  有机械臂:   scripts/start_web.sh"
    say "  无机械臂:   DICE_NO_ARM=1 scripts/start_web.sh"
    say "  然后浏览器打开 http://<板端IP>:8080"
}

for step in "${STEPS[@]}"; do
    "step_$step"
done
