# vision_yolov8_adjudicator K3 验收记录

> 验收日期：2026-08-29  
> 工作区：`codex/vision-yolov8-adjudicator`  
> 说明：本记录只保存命令结果和阻塞信息，不包含任何 API Key 或环境文件内容。

## 本机 Python 验证

专项测试已执行：

```bash
python3 -m pytest \
  tests/test_vision_adjudicator.py \
  tests/test_components_and_jobs.py \
  tests/test_server_api.py \
  tests/test_web_contract.py \
  tests/test_yolov8_runtime_docs.py \
  tests/test_yolov8_generic_build_boundary.py -q
```

结果：`77 passed`。

配置检查结果：

- profile：`backend/games/dice/vision_profile.json` 可加载；
- runtime：`resident`，`prewarm_camera=true`；
- 模型：`vision/yolov8_objdetect/models/best.q.onnx` 存在；
- WebRTC：`http://100.118.229.28:8889/dice/`；
- 结果后保持：`3` 秒。

全量测试已执行：

```bash
python3 -m pytest -q
```

结果：测试收集阶段被仓库内已有的 vendored SciPy 阻塞，缺少
`scipy._lib._ccallback_c` 扩展；该错误发生在 TTS 测试导入阶段，不能归因于视觉组件。

## 本机 C++ 构建检查

执行：

```bash
cmake --build vision/yolov8_objdetect/build -j2
```

结果：失败。现有 `CMakeCache.txt` 指向板端路径
`/home/spacemit/projects/dice-game/main/vision/yolov8_objdetect`，不是当前开发机路径。
现有二进制经 `file` 检查为 `RISC-V 64-bit`，开发机不能直接运行。

## K3 SSH 验证

执行：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 spacemit@spacemit-k3 'echo connected && hostname && id'
```

结果：`Permission denied (publickey,password)`。当前凭据未通过 SSH 认证，因此尚未执行
K3 编译、`--help`、`--self-test`、摄像头读取、MediaMTX WebRTC、多局 resident 复用或
多摄像头实测；不能将这些项目标记为已通过。

SSH 权限恢复后，在板端执行：

```bash
cd /home/spacemit/projects/dice-game/main/vision/yolov8_objdetect
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build build -j4
./build/yolov8_camera --help
./build/yolov8_camera --self-test --no-display
```

随后由后端 provider 验证 resident 启动、`START_ADJUDICATION` / `STOP_ADJUDICATION`、
结果后的 holding、取消回 idle、连续两局复用、快照清理、无孤儿进程，以及启用两路 profile
后的并行检测、严格多数投票和单次多图 LLM 复核。

