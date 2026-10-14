#!/usr/bin/env bash
# Dice Arena 源码一键部署 —— git clone 后在 K3 板端运行本脚本，
# 从裸系统到可启动：装依赖 → 编译视觉 runtime → 下载模型资产 →
# 安装机械臂依赖/按键权限/mediamtx → 摄像头/CAN 提示 → 无硬件自检。
#
# 用法:
#   scripts/setup_board.sh                     # 全部步骤顺序执行
#   scripts/setup_board.sh check               # 只跑指定步骤（可多个: deps robot permissions build models mediamtx can0 detect check）
#   scripts/setup_board.sh --yes               # 自动确认安装依赖（sudo 可能要求密码）
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
ASSUME_YES=0
NEEDS_LOGIN=0
STEPS=()

for arg in "$@"; do
    case "$arg" in
        --yes|-y)     ASSUME_YES=1 ;;
        --with-asr)    WITH_ASR=1 ;;
        --with-matcha) WITH_MATCHA=1 ;;
        --no-mediamtx) NO_MEDIAMTX=1 ;;
        -h|--help)     awk 'NR>1 && /^set -euo/ {exit} NR>1 {print}' "${BASH_SOURCE[0]}"; exit 0 ;;
        deps|robot|permissions|build|models|mediamtx|can0|detect|check) STEPS+=("$arg") ;;
        *) die "未知参数: $arg（--help 查看用法）" ;;
    esac
done
# 未指定步骤 = 全部顺序执行
if [[ ${#STEPS[@]} -eq 0 ]]; then
    STEPS=(deps robot permissions build models mediamtx can0 detect check)
fi
has_step() { [[ " ${STEPS[*]} " == *" $1 "* ]]; }

# Run the top-level installer as the customer, so asset ownership, PipeWire,
# and systemd --user all belong to the same account as the game service.
if [[ $EUID == 0 ]] && { has_step permissions || has_step mediamtx; }; then
    die "请用普通用户运行 scripts/setup_board.sh --yes；脚本会按需调用 sudo（不要 sudo 整个脚本）"
fi

confirm_install() {
    [[ $ASSUME_YES == 1 || $EUID == 0 ]] && return 0
    [[ -t 0 && -t 1 ]] || die "非交互环境请加 --yes，或先手动安装依赖后重跑"
    local reply=""
    read -r -p "setup: $1 [Y/n] " reply || true
    [[ ! "$reply" =~ ^[Nn] ]] || die "已取消安装；安装依赖后可重跑本脚本"
}

as_root() {
    if [[ $EUID == 0 ]]; then "$@"; else sudo "$@"; fi
}

# ---------------------------------------------------------------------------
# deps: 系统包（apt）
# ---------------------------------------------------------------------------
step_deps() {
    [[ "$(uname -m)" == "riscv64" ]] || die "本项目面向 SpacemiT K3 (riscv64)，当前架构 $(uname -m)"
    command -v python3 >/dev/null || die "缺少 python3（Bianbu 应自带）"
    /usr/bin/python3 -c 'import sys; assert sys.implementation.cache_tag == "cpython-314", "随仓库 RealSense 和 MOSS 依赖要求系统 Python 3.14，请使用与当前板一致的 Bianbu 4 镜像"' \
        || die "系统 Python ABI 不匹配；不能直接使用其他版本的 Bianbu 镜像"

    # 分三组:
    #   运行库组 —— 组件运行必需（推理 EP/Python 绑定、PipeWire 音频、摄像头工具；机械臂 Python 依赖由 robot 步骤安装）
    #   编译组   —— 从源码编译视觉 runtime 必需（OpenCV 开发包同时提供 410 运行库）
    #   GST 组   —— 视觉链路的 GStreamer 插件。整条链是硬依赖: rtsp_streamer
    #               启动失败时 yolov8_camera 直接退出，裁决和网页画面一起失效。
    RUN_PKGS=(
        spacemit-onnxruntime python3-spacemit-ort spacemit-tcm
        libsndfile1 alsa-utils pipewire-bin pipewire wireplumber
        curl v4l-utils python3-numpy iproute2
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
        if [[ "$(dpkg-query -W -f='${db:Status-Status}' "$pkg" 2>/dev/null)" == installed ]]; then
            continue
        fi
        # 命令型工具按二进制兜底：Debian 系 pkg-config 是过渡包名，
        # 实际二进制由 pkgconf 提供（板上真实踩过：dpkg 查无此包但命令在）。
        case "$pkg" in
            pkg-config) command -v pkg-config >/dev/null 2>&1 || missing+=("$pkg") ;;
            *) missing+=("$pkg") ;;
        esac
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "setup: 缺少系统包: ${missing[*]}"
        echo "  安装命令: sudo apt-get update && sudo apt-get install -y ${missing[*]}"
        confirm_install "现在安装这些系统包?"
        as_root apt-get update || warn "apt-get update 失败，继续尝试用现有索引安装"
        as_root apt-get install -y "${missing[@]}" \
            || die "系统包安装失败；请检查 Bianbu APT 源后重跑 deps --yes"
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
# robot: 系统 Python / RealSense wheel / 项目内锁定的 ORT（不连接设备）
# ---------------------------------------------------------------------------
step_robot() {
    if PYTHONNOUSERSITE=1 DICE_PYTHON=/usr/bin/python3 \
        bash "$ROOT_DIR/scripts/robot_dependencies.sh" check >/dev/null 2>&1; then
        say "robot 依赖已通过（Python / RealSense / SDK / 锁定运行库），跳过安装"
        return
    fi
    confirm_install "现在安装机械臂 Python、RealSense 和锁定识别运行库?"
    as_root bash "$ROOT_DIR/scripts/robot_dependencies.sh" install \
        || die "机械臂依赖安装失败；修复错误后重跑 robot --yes"
    PYTHONNOUSERSITE=1 DICE_PYTHON=/usr/bin/python3 \
        bash "$ROOT_DIR/scripts/robot_dependencies.sh" check \
        || die "机械臂依赖仍不可用；请根据上面的模块/运行库错误修复后重跑 robot"
    say "robot 通过（没有连接摄像头或 CAN）"
}

# ---------------------------------------------------------------------------
# permissions: 普通用户的设备读取权限（不更改游戏配置或打开设备）
# ---------------------------------------------------------------------------
step_permissions() {
    local account group board_enabled groups=(video)
    account="$(id -un)"
    board_enabled="$(python3 - "$ROOT_DIR/backend/config.json" <<'PYCONFIG'
import json, sys
print('true' if json.load(open(sys.argv[1])).get('input_board_enable', False) else 'false')
PYCONFIG
    )" || die "backend/config.json 无法读取，未修改设备权限"
    [[ "$board_enabled" == true ]] && groups+=(input)
    for group in "${groups[@]}"; do
        if ! id -nG "$account" | tr ' ' '\n' | grep -qx "$group"; then
            confirm_install "将 $account 加入 $group 组以读取设备?"
            getent group "$group" >/dev/null || as_root groupadd --system "$group"
            as_root usermod -aG "$group" "$account"
        fi
        if ! id -nG | tr ' ' '\n' | grep -qx "$group"; then NEEDS_LOGIN=1; fi
    done
    if [[ $NEEDS_LOGIN == 1 ]]; then
        warn "设备组已配置；当前会话仍是旧权限。安装完成后新开 SSH，桌面请注销重登，再启动服务"
    else
        say "permissions 通过（当前会话已具备设备组权限）"
    fi
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
        cmake -S asr/sensevoice -B asr/sensevoice/build -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_STREAM_DEMO=OFF -DBUILD_PYTHON_BINDINGS=OFF -DASR_MODEL_FETCH_OFF=ON
        cmake --build asr/sensevoice/build -j"$jobs"
    fi
    local built="vision/yolov8_objdetect、vision/yolov10_objdetect"
    if [[ $WITH_ASR == 1 ]]; then built="$built、asr/sensevoice"; fi
    say "build 完成（$built）"
}

# ---------------------------------------------------------------------------
# models: 下载模型资产（git 不带的部分）
# ---------------------------------------------------------------------------
fetch_verified() {  # fetch_verified <文件名> <URL> [required|optional]
    local name="$1" url="$2" policy="${3:-required}" expected=""
    mkdir -p "$DOWNLOAD_DIR"
    if [[ ! -s "$DOWNLOAD_DIR/$name.sha256" ]]; then
        curl -fL --retry 3 -o "$DOWNLOAD_DIR/$name.sha256.part" "$url.sha256" \
            && mv "$DOWNLOAD_DIR/$name.sha256.part" "$DOWNLOAD_DIR/$name.sha256" \
            || { rm -f "$DOWNLOAD_DIR/$name.sha256.part"; [[ "$policy" == optional ]] || die "无法获取 $name 校验和；请检查网络后重跑"; }
    fi
    if [[ -s "$DOWNLOAD_DIR/$name.sha256" ]]; then
        expected="$(awk 'NR==1 {print $1}' "$DOWNLOAD_DIR/$name.sha256")"
        [[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] || die "$name.sha256 不是有效的 SHA256，清理该校验文件后重跑"
    else
        warn "$name 的上游未发布 .sha256，下载后检查资产文件"
    fi
    if [[ -f "$DOWNLOAD_DIR/$name" ]]; then
        if [[ -n "$expected" ]]; then
            (cd "$DOWNLOAD_DIR" && echo "$expected  $name" | sha256sum -c -) \
                || { rm -f "$DOWNLOAD_DIR/$name"; die "$name 校验失败，已清理损坏缓存，请重跑"; }
        fi
        return
    fi
    if [[ -n "$expected" && -f "$DOWNLOAD_DIR/$name.part" ]] &&
        (cd "$DOWNLOAD_DIR" && echo "$expected  $name.part" | sha256sum -c - >/dev/null 2>&1); then
        # A previous run may have finished downloading and been interrupted
        # before rename; promoting it avoids an HTTP 416 resume failure.
        mv "$DOWNLOAD_DIR/$name.part" "$DOWNLOAD_DIR/$name"
        return
    fi
    say "下载 $name ..."
    # Only complete downloads get the final name. Interrupted .part files
    # remain resumable, so a rerun cannot mistake them for an installed asset.
    curl -fL --retry 3 -C - -o "$DOWNLOAD_DIR/$name.part" "$url" \
        || die "下载失败，已保留断点：$name.part；检查网络后重跑"
    if [[ -n "$expected" ]]; then
        (cd "$DOWNLOAD_DIR" && echo "$expected  $name.part" | sha256sum -c -) \
            || { rm -f "$DOWNLOAD_DIR/$name.part"; die "$name 校验失败，已清理损坏下载，请重跑"; }
    fi
    mv "$DOWNLOAD_DIR/$name.part" "$DOWNLOAD_DIR/$name"
}

step_models() {
    # Verify files, not a tracked model directory that exists after clone.
    local moss_root
    moss_root="$(python3 "$ROOT_DIR/scripts/board_assets.py" moss --path)" \
        || die "MOSS 配置不可用，请检查组件 config.json"
    if python3 "$ROOT_DIR/scripts/board_assets.py" moss --quiet; then
        say "MOSS 模型、Python、动态库和音色完整，跳过下载"
    else
        fetch_verified "$MOSS_ASSET" "$ASSETS_BASE_URL/$MOSS_ASSET"
        mkdir -p "$moss_root"
        tar -xzf "$DOWNLOAD_DIR/$MOSS_ASSET" -C "$moss_root"
        python3 "$ROOT_DIR/scripts/board_assets.py" moss \
            || die "MOSS 资产不完整，请检查上面的缺失文件及资产发布包"
    fi

    if [[ $WITH_MATCHA == 1 ]]; then
        local matcha_root
        matcha_root="$(python3 "$ROOT_DIR/scripts/board_assets.py" matcha --path)"
        if python3 "$ROOT_DIR/scripts/board_assets.py" matcha --quiet; then
            say "matcha 资产完整，跳过下载"
        else
            fetch_verified "$MATCHA_ASSET" "$ASSETS_BASE_URL/$MATCHA_ASSET"
            mkdir -p "$matcha_root"
            tar -xzf "$DOWNLOAD_DIR/$MATCHA_ASSET" -C "$matcha_root"
            python3 "$ROOT_DIR/scripts/board_assets.py" matcha || die "matcha 资产不完整"
        fi
    fi

    # SenseVoice（语音识别，--with-asr）: 官方模型站，放到组件约定的仓内目录
    if [[ $WITH_ASR == 1 ]]; then
        if python3 "$ROOT_DIR/scripts/board_assets.py" sensevoice --quiet; then
            say "SenseVoice 模型已存在，跳过"
        else
            fetch_verified "sensevoice.tar.gz" "$SENSEVOICE_URL" optional
            say "解压 SenseVoice 模型到 asr/sensevoice/model/ ..."
            local tmp="$DOWNLOAD_DIR/sensevoice-extract"
            rm -rf "$tmp" && mkdir -p "$tmp"
            tar -xzf "$DOWNLOAD_DIR/sensevoice.tar.gz" -C "$tmp"
            # 官方包内层目录结构不定，以 model_quant_optimized.onnx 所在目录为准整个搬入
            local model_src
            model_src="$(find "$tmp" -name model_quant_optimized.onnx -type f -print -quit)"
            [[ -n "$model_src" ]] || die "sensevoice.tar.gz 内未找到 model_quant_optimized.onnx"
            model_src="$(dirname "$model_src")"
            mkdir -p "$ROOT_DIR/asr/sensevoice/model"
            cp -a "$model_src"/. "$ROOT_DIR/asr/sensevoice/model/"
            python3 "$ROOT_DIR/scripts/board_assets.py" sensevoice || die "SenseVoice 模型资产不完整"
            rm -rf "$tmp"
        fi
    fi
    say "models 完成（语音总闸 asr_enabled 默认 false，需要语音时改 backend/config.json 并重启）"
}

# ---------------------------------------------------------------------------
# mediamtx: 实时画面推流服务（网页小窗）
# ---------------------------------------------------------------------------
mediamtx_probe() {  # 本地服务探活：绕过代理环境变量，curl 失败再以 ss 兜底
    curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:8889" >/dev/null 2>&1 \
        || ss -ltn "sport = :8889" 2>/dev/null | grep -q LISTEN
}

step_mediamtx() {
    if [[ $NO_MEDIAMTX == 1 ]]; then
        say "按 --no-mediamtx 跳过（识别与裁决不受影响，仅网页无实时画面小窗）"
        return 0
    fi
    if mediamtx_probe; then
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
    local options=()
    [[ $WITH_ASR == 1 ]] && options+=(--with-asr)
    [[ $WITH_MATCHA == 1 ]] && options+=(--with-matcha)
    bash "$ROOT_DIR/scripts/check_board_environment.sh" "${options[@]}" \
        || die "依赖自检失败；按上面的错误补齐依赖后重跑 check"

    if mediamtx_probe; then
        say "mediamtx 在线"
    else
        warn "mediamtx 未运行 —— 网页将没有实时画面（识别不受影响）"
    fi

    echo
    say "程序依赖和无硬件自检通过（摄像头/CAN/音频输出/标定需现场确认）。启动服务:"
    if [[ $NEEDS_LOGIN == 1 ]]; then say "  先新开 SSH 或注销桌面重登，刷新设备组权限"; fi
    say "  有机械臂:   scripts/start_web.sh"
    say "  无机械臂:   DICE_NO_ARM=1 scripts/start_web.sh"
    say "  然后浏览器打开 http://<板端IP>:8080"
}

for step in "${STEPS[@]}"; do
    "step_$step"
done
