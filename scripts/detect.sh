#!/usr/bin/env bash
# 列出 USB 摄像头的型号与 /dev/videoN 节点对应关系 (两摄像头部署用)。
#
# 一台 UVC 相机会占用多个 /dev/video 节点: 一个视频采集节点 (视觉程序
# 该用的) + 一个或多个 metadata 节点 (不能用于采集)。本脚本按物理设备
# 分组列出每台相机的采集节点, 并给出 dice 视觉配置的填写提示。
#
# 用法: ./detect.sh   (在 dice-arena 目录或任意位置均可)
set -euo pipefail

SAY() { echo "detect: $*"; }

# v4l2-ctl 依赖检查: 缺失时交互式自动安装 (全新环境 apt 索引可能为空,
# 先 update), 非交互环境打印手动命令退出 —— 与 install.sh 的模式一致。
if ! command -v v4l2-ctl >/dev/null 2>&1; then
    echo "detect: 缺少 v4l2-ctl (v4l-utils)"
    echo "  安装命令: sudo apt-get update && sudo apt-get install -y v4l-utils"
    if [[ -t 0 && -t 1 ]]; then
        reply=""
        read -r -p "detect: 现在自动安装 v4l-utils? [Y/n] " reply || true
        if [[ ! "$reply" =~ ^[Nn] ]]; then
            sudo apt-get update \
                || echo "detect: [警告] apt-get update 失败, 继续尝试用现有索引安装" >&2
            sudo apt-get install -y v4l-utils \
                || { echo "detect: [错误] 安装失败, 请手动执行上面的安装命令" >&2; exit 1; }
        else
            echo "detect: 请手动安装后重跑 detect.sh" >&2
            exit 1
        fi
    else
        echo "detect: [错误] 非交互环境, 请手动执行上面的安装命令后重跑" >&2
        exit 1
    fi
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
v4l2-ctl --list-devices >"$tmp" 2>/dev/null

# 判断一个节点是否为视频采集节点 (有可枚举的像素格式; metadata 节点没有)。
# 注意用 grep -c 而非 -q: -q 找到即退出会让上游 v4l2-ctl 收到 SIGPIPE,
# 在本脚本的 pipefail 下污染函数返回值。
is_capture_node() {
    v4l2-ctl -d "$1" --list-formats-ext 2>/dev/null | grep -c '^\s*\[[0-9]' >/dev/null
}

SAY "USB 摄像头探测:"
echo

found=0
while IFS= read -r header; do
    # 分组头形如 "C270 HD WEBCAM (usb-xhci-hcd.15.auto-1.4):"
    # 只关心 USB 相机, 跳过平台设备 (如 VPU 的 mvx)。
    [[ "$header" == *"("*usb-*"):"* ]] || continue
    found=$((found + 1))
    full="${header%:}"
    model="$(printf '%s' "$full" | sed 's/ *(.*//')"
    usb="$(printf '%s' "$full" | sed 's/.*(\(.*\))/\1/')"

    # 该分组头之后、下一分组头之前的缩进行即节点列表
    nodes="$(awk -v h="$header" \
        'index($0, h) == 1 {matched = 1; next}
         matched && /^[ \t]/ {print $1}
         matched && /^[^ \t]/ {exit}' "$tmp")"

    capture=""
    others=""
    for node in $nodes; do
        [[ "$node" == /dev/video* ]] || continue
        if [[ -z "$capture" ]] && is_capture_node "$node"; then
            capture="$node"
        else
            others="${others:+$others }$node"
        fi
    done

    echo "[$found] $model  (USB: $usb)"
    if [[ -n "$capture" ]]; then
        formats="$(v4l2-ctl -d "$capture" --list-formats-ext 2>/dev/null \
            | grep -oE "'[A-Z0-9]+'" | tr -d "'" | sort -u | tr '\n' ' ')"
        echo "    采集节点 : $capture   ← 视觉程序使用这个节点"
        echo "    支持格式 : ${formats:-未知}"
    else
        echo "    [警告] 未找到采集节点 (设备被占用或驱动异常)"
    fi
    [[ -n "$others" ]] && echo "    其他节点 : $others (metadata, 不可用于采集)"
    echo
done < <(grep ':$' "$tmp")

if [[ "$found" -eq 0 ]]; then
    echo "未发现 USB 摄像头。检查: lsusb / 线缆 / ls /dev/video*"
    exit 1
fi

SAY "dice 视觉: 把要用的采集节点填入 vision/yolov8_adjudicator/config.json 的 \"camera\" 字段,"
SAY "然后 scripts/stop_web.sh && scripts/start_web.sh 重启生效。"
