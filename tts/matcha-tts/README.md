# Matcha-TTS（Dice Arena 板端本地 TTS 引擎）

Sherpa-ONNX C API 实现的 Matcha-TTS（`matcha-icefall-zh-en`，单说话人 sid=0，
16 kHz mono 16-bit），运行在 SpacemiT K3 的 EP 加速核上。本目录是引擎源码与
板端资产的落位；Dice Arena 侧的适配组件在
`backend/components/tts_matcha/`。

## 目录结构

```text
tts/matcha-tts/
├── cpp/
│   ├── CMakeLists.txt
│   ├── matcha_tts_capi.cpp        # 单发文件合成（冒烟工具）
│   ├── matcha_tts_interactive.cpp # 交互式调试（stdin→aplay 板端播放）
│   └── matcha_tts_service.cpp     # 常驻服务模式（Dice Arena 使用）
├── matcha-model/                  # 板端资产（.gitignore）：声学/声码器 ONNX、
│                                  # tokens/lexicon、date-zh/number-zh.fst、espeak-ng-data
├── runtime/sherpa_onnx/           # 板端资产（.gitignore）：riscv64 wheel 提取的
│                                  # include/ + lib/（libsherpa-onnx-c-api.so 等）
├── build-cpp/                     # 编译产物（.gitignore）
├── run_cpp.sh                     # 单发合成入口（写 WAV 文件）
└── run_interactive.sh             # 交互调试入口（板端扬声器播放）
```

## 板端资产准备

模型与 sherpa 库不入 Git，首次部署时从验证项目 `~/projects/matcha-tts`
复制（或按 SHA 重新分发）：

```bash
cd ~/projects/dice-game/main/tts/matcha-tts
cp -a ~/projects/matcha-tts/matcha-model .
mkdir -p runtime
cp -a ~/projects/matcha-tts/py313/lib/python3.13/site-packages/sherpa_onnx \
       runtime/sherpa_onnx
```

## 编译（板上）

```bash
cd ~/projects/dice-game/main/tts/matcha-tts
cmake -S cpp -B build-cpp \
  -DSHERPA_ROOT="$PWD/runtime/sherpa_onnx" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp -j2
```

CMake 已把 sherpa 库目录写进 ELF 的 `RUNPATH`；run 脚本还会显式设置
`LD_LIBRARY_PATH` 双保险。

## 三个入口

```bash
# 1) 单发文件合成（EP 路径冒烟）
./run_cpp.sh /tmp/out.wav "你好，这是一次测试。" 1.0

# 2) 交互式调试（句子级流式 + aplay 播放）
./run_interactive.sh

# 3) 常驻服务模式（Dice Arena 后端专用）
./build-cpp/matcha_tts_service --model-dir "$PWD/matcha-model" \
    --ep-threads 2 --ep-affinity "8;9"
```

## 服务模式协议

- stdin 每行一个请求：`<id>\t<speed>\t<text>`（text 为行内剩余内容，
  可含 tab；`#` 开头与空行忽略）。
- stdout 每行一个 JSON 事件（逐事件 flush）：
  `ready`（模型加载+预热完成后才发，含 sample_rate/voice/EP 参数）、
  `sentence`（每句合成前）、`audio`（每句一帧完整 WAV 的 base64）、
  `done`（请求汇总）、`error`（请求级失败，进程存活）。
- 预热失败 = 启动失败：进程发 `error` 后以非 0 退出码结束，
  后端据此拒绝启动（本地 TTS 启动钉死 + 预热保证）。
- stderr 为人类可读诊断日志。
- 孤儿保护：内置 getppid() 看门狗线程，父进程（后端）消失后 2 秒内
  自行退出；stdin EOF 正常退出。

## 资源占用（K3 实测，2026-09-03）

- 常驻 RSS ≈ 330–360 MB（对照 MOSS-TTS-Nano daemon ≈ 4 GB）。
- EP 2 线程绑 8;9 核，预热后 RTF ≈ 0.16（5.4 s 音频 0.85 s 合成）。
- 冷加载（进程启动到模型可用）≈ 2.6–2.9 s，页缓存不加速——
  因此必须常驻 + 预热，禁止按请求拉起。
