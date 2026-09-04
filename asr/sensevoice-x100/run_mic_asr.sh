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
# 绑核 (专用核模式, 资源利用最优):
#   ./run_mic_asr.sh                        # 默认: ASR 独占 X100 核 6,7 (taskset),
#                                           #   其余 6 个 X100 + 8 个 A100 全留给应用
#   ./run_mic_asr.sh --cpu 0,1              # 换其他 X100 核 (全进程 taskset)
#   ./run_mic_asr.sh --cpu 8,9              # A100 算力核 (恰好 2 个, EP 线程精确绑定, 推理最快)
#   ./run_mic_asr.sh --cpu 10               # A100 单核 (自动 --threads 1, 最省资源)
#   ./run_mic_asr.sh --cpu none             # 不绑核 (推理线程在 X100 0-7 浮动)
#
# 音频输入: 默认设备跟随 PipeWire (板子设置界面里的输入设备), -i N 可指定
#
# 停止: Ctrl+C
# 其余参数透传给底层 demo, 例如 --pause 0.3 / --vad-thresh 600 / --enable-emotion

DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

MODE=default
CPU_LIST="${ASR_CPUS:-6,7}"   # 默认专用核 6,7; 环境变量 ASR_CPUS 可改默认值
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

# --cpu none: 显式关闭绑核
[ "$CPU_LIST" = "none" ] && CPU_LIST=""

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
#   A100 核 (8-15): EP 的 SPACEMIT_EP_INTRA_THREAD_AFFINITY (每线程绑一个核,
#                   核数必须等于线程数, 故只允许 1 或 2 个核, 并自动透传 --threads N;
#                   taskset 无法绑到 8-15)
#   X100 核 (0-7):  taskset 全进程绑定; 引擎内 core_arch=x100 已让 EP 推理线程
#                   绑定 X100, taskset 把整个进程限制在指定核上
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
        if [ "$n_a100" -ne 1 ] && [ "$n_a100" -ne 2 ]; then
            echo "错误: A100 核必须选 1 或 2 个 (EP 线程数=核数), 例如 --cpu 8,9 或 --cpu 10" >&2
            exit 1
        fi
        export SPACEMIT_EP_INTRA_THREAD_AFFINITY="${cores// /;}"
        PASS_ARGS+=(--core-arch a100 --threads "$n_a100")   # 切回 A100; 线程数=核数
        echo ">>> 绑核: A100 算力核 ${cores// /,} (EP 线程精确绑定, threads=$n_a100)"
    else
        TASKSET="taskset -c ${cores// /,}"
        # X100 上 EP 线程数不影响 RTF (实测 1/2 线程均 ~0.41), 默认单线程少占线程位
        if ! printf "%s\n" "${PASS_ARGS[@]}" | grep -qx -- --threads; then
            PASS_ARGS+=(--threads 1)
        fi
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
