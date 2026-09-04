#!/bin/bash
# 麦克风实时语音转文字 (SpacemiT K3, 16核: 0-7=X100性能核 2.15GHz / 8-15=A100能效核 1.8GHz)
#
# 采集与识别已整合为单进程程序 asr_live_demo
#   (audio 采集组件 + model-zoo-asr 识别组件, 不再依赖 arecord 管道)
#
# 用法:
#   ./run_mic_asr.sh                        # 模式1(默认): 官方 asr_stream_demo
#                                           #   PortAudio 采集, SenseVoice, 中文, 自动标点
#   ./run_mic_asr.sh --auto                 # 模式2: 多语言自动检测 (定时断句)
#   ./run_mic_asr.sh --captions             # 模式3: 实时字幕 (VAD 断句, 说完一句立即出字)
#   ./run_mic_asr.sh --wake 小美,小美同学    # 模式4: 语音唤醒 (说到唤醒词打印 [🔔 唤醒])
#   ./run_mic_asr.sh --wake 小美 --wake-exit # 唤醒后退出, 供脚本衔接 LLM/TTS
#   ./run_mic_asr.sh --zipformer            # 模式5: Zipformer (无标点)
#
# 绑核:
#   ./run_mic_asr.sh --cpu 8,9              # 指定 A100 能效核 (8-15, 必须恰好2个, EP推理线程精确绑定)
#   ./run_mic_asr.sh --cpu 0,1              # 指定 X100 性能核 (0-7, 全进程绑定, 性能最佳)
#   ./run_mic_asr.sh --cpu 4-6              # 范围写法
#   不指定时: 引擎默认 core_arch=x100, EP 推理线程绑 X100 通用核 (0-7)
#
# 音频输入: 默认设备跟随 PipeWire (板子设置界面里的输入设备), -i N 可指定
#
# 停止: Ctrl+C
# 其余参数透传给底层 demo, 例如 --pause 0.3 / --vad-thresh 600 / --enable-emotion

DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

MODE=default
CPU_LIST=""
WAKE_WORDS=""
PASS_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --auto)      MODE=auto;      shift ;;
        --captions)  MODE=captions;  shift ;;
        --zipformer) MODE=zipformer; shift ;;
        --wake)      MODE=wake; WAKE_WORDS="$2"; shift 2 ;;
        --cpu)       CPU_LIST="$2";  shift 2 ;;
        *) PASS_ARGS+=("$1"); shift ;;  # 其余参数透传给底层 demo
    esac
done

# 展开 CPU 列表: 支持 "8,9" "4-6" "0,2-3" 混合写法
expand_cpu() {
    local out=""
    local part a b i
    for part in ${1//,/ }; do
        if [[ "$part" == *-* ]]; then
            a=${part%-*}; b=${part#*-}
            for ((i=a; i<=b; i++)); do out="$out $i"; done
        else
            out="$out $part"
        fi
    done
    echo $out
}

# 根据 CPU 列表设置绑定方式:
#   A100 核 (8-15): EP 的 SPACEMIT_EP_INTRA_THREAD_AFFINITY (分号分隔, 每线程一个核,
#                   EP 固定 2 个推理线程, 因此必须恰好 2 个核; taskset 无法绑到 8-15);
#                   同时传 --core-arch a100 把引擎从默认 x100 切回 A100
#   X100 核 (0-7):  taskset 全进程绑定; 引擎内 core_arch=x100 已让 EP 推理线程
#                   绑定 X100, taskset 只是把非推理线程也限制在指定核上
setup_cpu() {
    local cores c
    cores=$(expand_cpu "$CPU_LIST")
    [ -z "$cores" ] && return 0

    local n_a100=0 n_x100=0
    for c in $cores; do
        if [ "$c" -ge 8 ] 2>/dev/null && [ "$c" -le 15 ] 2>/dev/null; then
            n_a100=$((n_a100+1))
        elif [ "$c" -ge 0 ] 2>/dev/null && [ "$c" -le 7 ] 2>/dev/null; then
            n_x100=$((n_x100+1))
        else
            echo "错误: 无效核号 '$c' (有效范围 0-15)" >&2; exit 1
        fi
    done

    if [ "$n_a100" -gt 0 ] && [ "$n_x100" -gt 0 ]; then
        echo "错误: --cpu 不能混选 X100(0-7) 和 A100(8-15) 核" >&2; exit 1
    fi

    if [ "$n_a100" -gt 0 ]; then
        if [ "$n_a100" -ne 2 ]; then
            echo "错误: A100 核必须恰好选 2 个 (EP 固定 2 个推理线程), 例如 --cpu 8,9 或 --cpu 8,10" >&2
            exit 1
        fi
        local first second
        set -- $cores; first=$1; second=$2
        export SPACEMIT_EP_INTRA_THREAD_AFFINITY="${first};${second}"
        PASS_ARGS+=(--core-arch a100)   # 引擎默认 x100, 选 A100 时需显式切回
        echo ">>> 绑核: A100 能效核 ${first} 和 ${second} (EP 推理线程)"
    else
        TASKSET="taskset -c ${cores// /,}"
        echo ">>> 绑核: X100 通用核 ${cores// /,} (全进程 taskset)"
    fi
}

TASKSET=""
[ -n "$CPU_LIST" ] && setup_cpu

case "$MODE" in
    default)
        exec $TASKSET "$DIR/build/bin/asr_stream_demo" -i -1 -f "${FLUSH_SEC:-3}" "${PASS_ARGS[@]}"
        ;;
    auto)
        exec $TASKSET "$DIR/build/bin/asr_live_demo" -i -1 "${PASS_ARGS[@]}"
        ;;
    captions)
        # VAD 断句 + 单句上限 4 秒 (连续说话时的最长等待; 可用 --max-utt 覆盖)
        exec $TASKSET "$DIR/build/bin/asr_live_demo" -i -1 --vad --max-utt 4 "${PASS_ARGS[@]}"
        ;;
    wake)
        # 语音唤醒: VAD 断句 + 唤醒词文本匹配; --wake-exit 供衔接下一级流程
        exec $TASKSET "$DIR/build/bin/asr_live_demo" -i -1 --vad --wake "$WAKE_WORDS" --max-utt 4 "${PASS_ARGS[@]}"
        ;;
    zipformer)
        # K3 + SpaceMIT EP 下 zipformer 需临时禁用 Conv 算子
        export SPACEMIT_EP_DISABLE_OP_TYPE_FILTER="Conv"
        exec $TASKSET "$DIR/build/bin/asr_live_demo" -i -1 --engine zipformer "${PASS_ARGS[@]}"
        ;;
esac
