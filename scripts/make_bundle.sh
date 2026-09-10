#!/usr/bin/env bash
# 产出两套可分发安装包（在源板或有完整资产的检出上运行）：
#
#   dice-arena-bundle-<date>.tar    主包 ~1.05G：完整成品树（代码 + 全部运行资产），
#                                   解压即用，无需 git、无需网络
#   mediamtx-bundle-<date>.tar      流媒体小包 ~34M：独立安装的 mediamtx 服务
#
# 用法: scripts/make_bundle.sh [输出目录，默认 $HOME]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d)"
OUT_ROOT="${1:-$HOME}"
WORK_DIR="${OUT_ROOT%/}/.bundle-work-${STAMP}"
DICE_DIR="$WORK_DIR/dice-arena"
MTX_DIR="$WORK_DIR/mediamtx-bundle"
MEDIAMTX_DIR="${MEDIAMTX_DIR:-$HOME/projects/mediamtx}"

die() { echo "make_bundle: $*" >&2; exit 1; }

[[ -d "$ROOT_DIR/.git" ]] || die "必须在仓库检出内运行 (当前: $ROOT_DIR)"

# git archive 只导出已提交内容 —— 工作区若有未提交改动 (除板上 qwen3-tts
# symlink 与 HEAD 实体文件的已知 typechange 共存外), 打出来的包会缺改动。
if git -C "$ROOT_DIR" status --porcelain | grep -qvE '^\s*T '; then
    echo "工作区存在未提交改动, 先提交再打包:" >&2
    git -C "$ROOT_DIR" status --porcelain | grep -vE '^\s*T ' >&2
    exit 1
fi

REQUIRED_DIRS=(
    asr/sensevoice/model
    asr/sensevoice/build
    tts/moss-tts-nano/models
    tts/moss-tts-nano/voice
    tts/moss-tts-nano/python
    tts/moss-tts-nano/lib
    tts/moss-tts-nano/assets
    vision/yolov8_adjudicator/build
)
for d in "${REQUIRED_DIRS[@]}"; do
    [[ -d "$ROOT_DIR/$d" ]] || die "缺少板端资产目录: $d (应在源板上运行)"
done
[[ -x "$MEDIAMTX_DIR/bin/mediamtx" ]] || die "找不到 mediamtx: $MEDIAMTX_DIR/bin/mediamtx"
[[ -f "$HOME/.config/systemd/user/mediamtx.service" ]] || die "找不到 mediamtx systemd unit"

echo "==> 打包内容取自已提交状态 (HEAD: $(git -C "$ROOT_DIR" rev-parse --short HEAD))"
echo "==> 关键配置预览 (打包前请确认这些值是分发想要的):"
python3 - "$ROOT_DIR" <<'PY'
import json, sys
root = sys.argv[1]
cfg = json.load(open(f"{root}/backend/config.json"))
manifest = json.load(open(f"{root}/backend/games/dice/manifest.json"))
llm = manifest.get("vision_profile", manifest.get("vision", {})).get("llm", {})
print(f"    asr_enabled (全局语音总闸) = {cfg.get('asr_enabled')}")
print(f"    providers = {json.dumps(cfg.get('providers', {}))}")
print(f"    dice LLM 复核 (vision_profile.llm.enabled) = {llm.get('enabled', '(未设置)')}")
PY

[[ -e "$WORK_DIR" ]] && die "工作目录已存在: $WORK_DIR (上次打包未清理?)"
mkdir -p "$DICE_DIR" "$MTX_DIR"

echo "==> 1/6 导出代码树 (git archive, 无 .git)"
git -C "$ROOT_DIR" archive --format=tar main | tar -x -C "$DICE_DIR"

echo "==> 2/6 合入板端资产 (排除 __pycache__ / *.pyc)"
# rsync 语义: 源/目标都带尾斜杠 = 拷内容。无尾斜杠目录对不存在的目标
# 会被当容器, 造成 bin/bin 嵌套 (板上实测踩过)。
for d in "${REQUIRED_DIRS[@]}"; do
    mkdir -p "$DICE_DIR/$d"
    rsync -a --exclude='__pycache__/' --exclude='*.pyc' \
        "$ROOT_DIR/$d/" "$DICE_DIR/$d/"
done

echo "==> 3/6 安装脚本与说明入主包"
cp "$ROOT_DIR/scripts/bundle_install_dice.sh" "$DICE_DIR/install.sh"
chmod +x "$DICE_DIR/install.sh"
cp "$ROOT_DIR/scripts/安装说明-dice.md" "$DICE_DIR/安装说明.md"

echo "==> 4/6 组装 mediamtx 小包"
cp "$MEDIAMTX_DIR/bin/mediamtx" "$MTX_DIR/"
cp "$MEDIAMTX_DIR/config/mediamtx.yml" "$MTX_DIR/"
# unit 里的绝对路径由 install.sh 按目标板改写, 这里原样带走当模板。
cp "$HOME/.config/systemd/user/mediamtx.service" "$MTX_DIR/mediamtx.service.template"
cp "$ROOT_DIR/scripts/bundle_install_mediamtx.sh" "$MTX_DIR/install.sh"
chmod +x "$MTX_DIR/install.sh"
cp "$ROOT_DIR/scripts/安装说明-mediamtx.md" "$MTX_DIR/安装说明.md"

echo "==> 5/6 打 tar (模型已压缩, 不再 gzip)"
tar -cf "${OUT_ROOT%/}/dice-arena-bundle-${STAMP}.tar" -C "$WORK_DIR" dice-arena
tar -cf "${OUT_ROOT%/}/mediamtx-bundle-${STAMP}.tar" -C "$WORK_DIR" mediamtx-bundle

echo "==> 6/6 清理临时目录"
rm -rf "$WORK_DIR"

echo
echo "完成。分发物 (拷 U 盘或内网传输):"
du -sh "${OUT_ROOT%/}/dice-arena-bundle-${STAMP}.tar" \
       "${OUT_ROOT%/}/mediamtx-bundle-${STAMP}.tar" | sed 's|^|  |'
echo
echo "新板安装顺序: 先 mediamtx 后 dice ——"
echo "  tar -xf mediamtx-bundle-${STAMP}.tar && mediamtx-bundle/install.sh"
echo "  tar -xf dice-arena-bundle-${STAMP}.tar && dice-arena/install.sh"
