#!/usr/bin/env bash
# mediamtx 小包安装脚本 —— 在解压出来的 mediamtx-bundle/ 目录里运行。
# 落位 ~/projects/mediamtx 并注册 systemd --user 服务 (无需 root)。
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_DIR="${MEDIA_DIR:-$HOME/projects/mediamtx}"
UNIT_DST="$HOME/.config/systemd/user/mediamtx.service"

say() { echo "install: $*"; }
die() { echo "install: [错误] $*" >&2; exit 1; }

[[ -f "$BUNDLE_DIR/mediamtx" ]] || die "包不完整, 缺少 mediamtx 二进制"
[[ -f "$BUNDLE_DIR/mediamtx.yml" ]] || die "包不完整, 缺少 mediamtx.yml"
[[ -f "$BUNDLE_DIR/mediamtx.service.template" ]] || die "包不完整, 缺少 unit 模板"

if [[ -e "$MEDIA_DIR" ]]; then
    die "目标目录已存在: $MEDIA_DIR —— 如需重装请先手动删除它 (systemctl --user stop mediamtx && rm -rf $MEDIA_DIR)"
fi

say "安装 mediamtx 到 $MEDIA_DIR"
mkdir -p "$MEDIA_DIR/bin" "$MEDIA_DIR/config"
cp "$BUNDLE_DIR/mediamtx" "$MEDIA_DIR/bin/"
chmod +x "$MEDIA_DIR/bin/mediamtx"
cp "$BUNDLE_DIR/mediamtx.yml" "$MEDIA_DIR/config/"

say "注册 systemd --user 服务"
mkdir -p "$HOME/.config/systemd/user"
# 模板里的绝对路径按本机实际位置改写。
sed "s|/home/spacemit/projects/mediamtx|$MEDIA_DIR|g" \
    "$BUNDLE_DIR/mediamtx.service.template" > "$UNIT_DST"
systemctl --user daemon-reload
systemctl --user enable --now mediamtx

sleep 1
systemctl --user is-active --quiet mediamtx \
    || die "mediamtx 服务未正常运行, 查看: systemctl --user status mediamtx"

say "完成。mediamtx 已作为用户服务运行 (开机自启):"
say "  WebRTC/HTTP: http://127.0.0.1:8889"
say "  RTSP:        rtsp://127.0.0.1:8554"
say "  管理:        systemctl --user status|stop|restart mediamtx"
say "下一步: 安装 dice-arena 主包 (解压后运行 ./install.sh)。"
