#!/bin/bash
# 探测 EP 推理线程落在哪些核
# 用法: ./probe_ep.sh [NAME=VALUE 环境变量...] [--taskset=核列表]
# 示例: ./probe_ep.sh SPACEMIT_EP_PERFER_CORE_ARCH=x100 --taskset=0,1
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
A="$HOME/.cache/models/assets/audio/001_zh_daily_weather.wav"

ENVS=()
TASKSET_CMD=()
for arg in "$@"; do
    case "$arg" in
        --taskset=*) TASKSET_CMD=(taskset -c "${arg#--taskset=}") ;;
        *) ENVS+=("$arg") ;;
    esac
done

(
    env "${ENVS[@]}" "${TASKSET_CMD[@]}" ./build/bin/asr_file_demo "$A" --rounds 6 >/tmp/asr_out.txt 2>&1 &
    sleep 3
    PID=$(pgrep -x asr_file_demo | head -1)
    if [ -n "$PID" ]; then
        for t in /proc/$PID/task/*; do
            grep Cpus_allowed_list $t/status | sed 's|.*0x[0-9a-f]*||'
        done | sort | uniq -c | tr '\n' ' '
    else
        echo -n "FAIL: "
        grep -iE "affinity|invalid" /tmp/asr_out.txt | head -1
    fi
    wait
)
echo " <= env=[${ENVS[*]}] taskset=[${TASKSET_CMD[*]}]"
