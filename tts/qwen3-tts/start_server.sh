#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUNTIME="$ROOT/runtime"
[[ -x "$DEFAULT_RUNTIME/bin/llama-server" ]] || DEFAULT_RUNTIME="$ROOT/llama.cpp-optimized/build-k3"
RUNTIME="${QWEN3_TTS_RUNTIME:-$DEFAULT_RUNTIME}"
MODEL="${QWEN3_TTS_MODEL:-$ROOT/qwen3-tts-0.6b}"
HOST="${QWEN3_TTS_HOST:-127.0.0.1}"
PORT="${QWEN3_TTS_PORT:-18080}"
PID_FILE="$ROOT/llama-server.pid"
LOG_FILE="$ROOT/llama-server.log"

[[ -x "$RUNTIME/bin/llama-server" ]] || { echo "缺少 runtime: $RUNTIME/bin/llama-server" >&2; exit 1; }
SERVER_BIN="$RUNTIME/bin/llama-server"
[[ -f "$MODEL/config.json" ]] || { echo "缺少模型配置: $MODEL/config.json" >&2; exit 1; }

# Prefer an explicit ORT directory, then the bundled package for backward
# compatibility, and finally the board's system-installed SpaceMIT ORT.
if [[ -n "${QWEN3_TTS_ORT_LIB_DIR:-}" ]]; then
    ORT_LIB_DIR="$QWEN3_TTS_ORT_LIB_DIR"
elif [[ -n "${QWEN3_TTS_ORT:-}" ]]; then
    ORT_LIB_DIR="$QWEN3_TTS_ORT/lib"
elif [[ -f "$ROOT/spacemit-ort.riscv64.2.0.6/lib/libspacemit_ep.so" ]]; then
    ORT_LIB_DIR="$ROOT/spacemit-ort.riscv64.2.0.6/lib"
elif [[ -f /usr/lib/libspacemit_ep.so && -f /usr/lib/libonnxruntime.so.1 ]]; then
    ORT_LIB_DIR=/usr/lib
elif [[ -f /usr/local/lib/libspacemit_ep.so && -f /usr/local/lib/libonnxruntime.so.1 ]]; then
    ORT_LIB_DIR=/usr/local/lib
else
    echo "缺少 SpaceMIT ORT（请安装 spacemit-onnxruntime 或设置 QWEN3_TTS_ORT_LIB_DIR）" >&2
    exit 1
fi
[[ -f "$ORT_LIB_DIR/libspacemit_ep.so" ]] || { echo "缺少 SpaceMIT EP: $ORT_LIB_DIR/libspacemit_ep.so" >&2; exit 1; }
[[ -f "$ORT_LIB_DIR/libonnxruntime.so.1" ]] || { echo "缺少 ONNX Runtime: $ORT_LIB_DIR/libonnxruntime.so.1" >&2; exit 1; }
if [[ -d "$RUNTIME/lib" ]]; then
    RUNTIME_LIB="$RUNTIME/lib"
else
    RUNTIME_LIB="$RUNTIME/bin"
fi

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        old_exe="$(readlink -f "/proc/$old_pid/exe" 2>/dev/null || true)"
        if curl -fsS --max-time 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
            if [[ "$old_exe" == "$SERVER_BIN" ]]; then
                echo "Qwen3-TTS 服务已经运行: pid=$old_pid http://$HOST:$PORT"
                exit 0
            fi

            # A runtime switch must not silently reuse a healthy server started
            # with a different binary. Only replace a server whose executable
            # is inside this project and whose command line identifies this
            # project's SMT TTS endpoint.
            old_cmdline="$(tr '\0' ' ' < "/proc/$old_pid/cmdline" 2>/dev/null || true)"
            if [[ "$old_exe" == "$ROOT"/*/bin/llama-server \
                  && "$old_cmdline" == *"--media-backend smt"* \
                  && "$old_cmdline" == *"--port $PORT"* ]]; then
                echo "检测到不同 runtime 的 Qwen3-TTS 服务，切换 pid=$old_pid"
                kill "$old_pid" 2>/dev/null || true
                for _ in $(seq 1 40); do
                    kill -0 "$old_pid" 2>/dev/null || break
                    sleep 0.25
                done
                if kill -0 "$old_pid" 2>/dev/null; then
                    echo "旧 Qwen3-TTS 进程未能退出: pid=$old_pid" >&2
                    exit 1
                fi
            else
                echo "端口 $PORT 已被不同进程占用，拒绝覆盖: pid=$old_pid exe=$old_exe" >&2
                exit 1
            fi
        else
            echo "停止无响应的旧进程 pid=$old_pid"
            kill "$old_pid" 2>/dev/null || true
            sleep 1
        fi
    fi
fi

# Keep TTS on the six requested SpaceMIT preferred cores while YOLO owns 14-15.
# The stable runtime reads SPACEMIT_PERFER_CORE_ID. The two QWEN3_TTS_* tuning
# variables are only consumed by an explicitly selected realtime test runtime.
export SPACEMIT_PERFER_CORE_ID="${SPACEMIT_PERFER_CORE_ID:-8,9,10,11,12,13}"
export QWEN3_TTS_GEMV_CPU_MASK="${QWEN3_TTS_GEMV_CPU_MASK:-4-7}"
export QWEN3_TTS_THREADPOOL_POLL="${QWEN3_TTS_THREADPOOL_POLL:-50}"
export LD_LIBRARY_PATH="$ORT_LIB_DIR:$RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
: > "$LOG_FILE"
setsid "$RUNTIME/bin/llama-server" \
    --media-backend smt \
    --smt-config-dir "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --no-ui \
    > "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "正在启动 Qwen3-TTS: pid=$pid"
echo "runtime: $RUNTIME"
echo "ORT lib dir: $ORT_LIB_DIR"

for _ in $(seq 1 90); do
    if curl -fsS --max-time 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        echo "Qwen3-TTS 已就绪: http://$HOST:$PORT"
        exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Qwen3-TTS 启动失败，日志如下：" >&2
        tail -200 "$LOG_FILE" >&2
        exit 1
    fi
    sleep 1
done

echo "Qwen3-TTS 启动超时，日志如下：" >&2
tail -200 "$LOG_FILE" >&2
exit 1
