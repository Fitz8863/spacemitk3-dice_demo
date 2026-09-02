#!/bin/bash
# zipformer 流式 ASR 麦克风实时识别启动脚本
#
# 用法:
#   ./run_mic.sh                # 跟随系统默认输入设备（桌面设置里选的麦克风）
#   ./run_mic.sh c920           # 直连 C920 USB 摄像头麦克风（跳过 PipeWire，延迟更低）
#   ./run_mic.sh es8326         # 直连板载 ES8326 音频口
#   ./run_mic.sh --cpu          # 不用 SpaceMIT EP（纯 CPU int8）
#   ./run_mic.sh wav 文件.wav   # 识别 wav 文件
set -e
cd "$(dirname "$0")"

MODEL_DIR="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
BIN=build/stream_asr

if [ ! -x "$BIN" ]; then
    echo "先编译: cmake -S . -B build && cmake --build build -j8"
    exit 1
fi

# 剩余参数透传
ARGS=()
MIC="default"   # ALSA default → PipeWire → 桌面设置里选择的默认输入
MODE=""

while [ $# -gt 0 ]; do
    case "$1" in
        c920)    MIC="plughw:CARD=C920,DEV=0"; shift ;;
        es8326)  MIC="plughw:CARD=sndes8326,DEV=0"; shift ;;
        wav)     MODE="wav"; shift ;;
        --)      shift ;;            # 分隔符，后面的参数原样透传
        *)       ARGS+=("$1"); shift ;;
    esac
done

if [ "$MODE" = "wav" ]; then
    exec "$BIN" "${ARGS[@]}"
else
    echo "麦克风: $MIC （说话即出字，Ctrl-C 结束）"
    exec arecord -D "$MIC" -f S16_LE -r 16000 -c 1 -t raw 2>/dev/null \
        | "$BIN" --pcm "${ARGS[@]}"
fi
