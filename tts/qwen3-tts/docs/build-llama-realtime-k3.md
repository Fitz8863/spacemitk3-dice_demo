# 构建 Qwen3-TTS 实时优化版 llama.cpp runtime

本项目的 `runtime/bin/` 是板端 `riscv64` 预编译 runtime。实时优化没有把完整的
`llama.cpp` 源码树提交进来，而是提交了一个最小 patch：

- `patches/llama.cpp-realtime.patch`

该 patch 基于 SpaceMIT Qwen3-TTS 对应的 `llama.cpp` commit：

```text
787e5fc qwen3-tts : attach a shared threadpool to avoid per-graph pool churn
```

## Patch 内容

patch 修改两个位置：

- `tools/smt-mtmd/smt/qwen3-tts/qwen3_tts_talker.cpp`
  - 可选地把 4 个 f16 GEMV worker 一一绑定到普通 CPU；
  - 可选地设置 Qwen3-TTS ggml threadpool 的 polling 参数。
- `tools/smt-mtmd/media-worker.cpp`
  - 可选地设置 media worker 的普通 CPU affinity。

这些 CPU affinity 只约束 Linux CPU 线程，不能用来证明 AI Core 使用情况。
SpaceMIT preferred AI core 通过 `SPACEMIT_PERFER_CORE_ID` 固定为 `8,9,10,11`。
当前 YOLO 继续由它自己的配置使用 `14;15`，不要把两者混用。

## 在 K3 板端构建

先准备与当前 runtime 对应的 SpaceMIT fork，并确认源码处于基础 commit：

```bash
cd /home/spacemit/projects/qwen3-tts
# 示例：源码目录由部署者自行准备
cd /home/spacemit/projects/qwen3-tts/llama.cpp-realtime
git checkout 787e5fc
patch -p1 < /home/spacemit/projects/qwen3-tts/patches/llama.cpp-realtime.patch
```

使用 K3 构建参数：

```bash
cmake -S . -B build-k3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON \
  -DGGML_CPU_RISCV64_SPACEMIT=ON \
  -DGGML_RV_ZBA=ON \
  -DLLAMA_SERVER_SMT_MTMD=ON \
  -DSPACEMIT_ORT_DIR=/home/spacemit/projects/qwen3-tts/spacemit-ort-build \
  -DCMAKE_INSTALL_RPATH='$ORIGIN'
cmake --build build-k3 --target llama-server -j4
file build-k3/bin/llama-server
```

`-DGGML_RV_ZBA=ON` 是 K3 构建的必要选项；缺少它会在编译时触发
`riscv zba extension not enabled`。构建结果必须是 `riscv64` ELF。

## 安装到临时 runtime 并验证

不要一开始覆盖正式 runtime。先复制到独立目录：

```bash
cd /home/spacemit/projects/qwen3-tts
rm -rf runtime-realtime-test
mkdir -p runtime-realtime-test/bin
cp /path/to/build-k3/bin/llama-server runtime-realtime-test/bin/
cp /path/to/build-k3/bin/lib*.so* runtime-realtime-test/bin/ 2>/dev/null || true
```

若构建系统把共享库放在 `build-k3/lib/`，也复制到 `runtime-realtime-test/lib/`，或按
`start_server.sh` 的 runtime 布局放置。使用项目脚本启动：

```bash
QWEN3_TTS_RUNTIME="$PWD/runtime-realtime-test" \
QWEN3_TTS_ORT_LIB_DIR=/usr/lib \
./start_server.sh
```

启动日志应包含：

```text
preferred core ids: 8,9,10,11
[qwen3-tts] gemv worker 0 bound to CPU 4
[qwen3-tts] gemv worker 1 bound to CPU 5
[qwen3-tts] gemv worker 2 bound to CPU 6
[qwen3-tts] gemv worker 3 bound to CPU 7
[qwen3-tts] ggml threadpool poll=50
```

通过 Dice Arena 功能包进行 3 次相同文本测速，避免旧交互式播放器影响 RTF：

```bash
cd /home/spacemit/projects/dice-game/main
QWEN3_TTS_RUNTIME="$PWD/tts/qwen3-tts/runtime-realtime-test" \
python3 backend/componentctl.py start tts_qwen3
for _ in 1 2 3; do
  curl -fsS http://127.0.0.1:18080/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-tts","input":"实时性基准测试，当前YOLO检查模型同时运行。","response_format":"wav"}' \
    -o /tmp/qwen3-benchmark.wav
done
```

确认：

```bash
pid="$(cat llama-server.pid)"
curl -fsS http://127.0.0.1:18080/health
for status in /proc/"$pid"/task/*/status; do
  tid="${status%/status}"; tid="${tid##*/}"
  awk -v tid="$tid" '/^Cpus_allowed_list:/{print "TID=" tid " " $0}' "$status"
done
```

`Cpus_allowed_list` 只能验证 Linux CPU affinity；SpaceMIT EP 是否加载可用：

```bash
grep -E 'libonnxruntime|libspacemit_ep' "/proc/$pid/maps" \
  | awk '{print $6}' | sort -u
```

## 默认运行时环境变量

`start_server.sh` 已提供适合本项目当前板端的默认值：

```text
SPACEMIT_PERFER_CORE_ID=8,9,10,11
QWEN3_TTS_GEMV_CPU_MASK=4-7
QWEN3_TTS_THREADPOOL_POLL=50
```

每个变量都可以在启动前覆盖。默认值来自在 YOLO 持续运行时的短文本 A/B：

- `poll=50`：约 `1.66-1.68`；
- `poll=0`：约 `1.71-1.77`；
- `poll=10`：约 `1.75-1.80`。

这些是板端实测区间，不是所有负载下的理论保证；切换模型、YOLO 负载或系统调度后应重新测量。
