# Repository Guidelines

## Project Structure & Module Organization

This package is the second C++17 SpaceMIT K3 vision runtime of the dice-game
project (the first is `../yolov8_objdetect`), serving gesture games such as
rock-paper-scissors:

- `src/main.cpp` owns CLI parsing, frame queues, the vision-control-v1
  protocol layer (resident prewarm, START/STOP/CANCEL/FINAL_RESULT commands,
  jsonl-events-v2 event emission), stability gating, and pipeline
  orchestration.
- `src/gstreamer_camera.{h,cpp}` handles V4L2/GStreamer capture, MJPEG
  decoding, and software-decoder fallback (shared shape with the v8 package).
- `src/opencl_preprocess.{h,cpp}` implements NV12-to-YOLO preprocessing on
  OpenCL, including in-kernel rotation remapping and the ROI inference crop.
- `src/yolov10_detector.{h,cpp}` initializes SpaceMIT ONNX Runtime (graph
  optimization pinned to BASIC for the PPQ quantized graph) and parses the
  end-to-end `[1,300,6]` output (decoded xyxy + conf + class_id, no NMS).
- `src/rps_mapper.h` folds the config-declared model vocabulary onto game
  labels. The package has **no built-in class table**: `classes` and
  `rps_map` always come from the game's `adjudicator_config.json`.
- `src/control_protocol.h` is the newline-delimited control command reader.
- Models are not stored here: the game's model lives under
  `../../backend/games/<id>/models/` and reaches this binary through the
  game profile's runtime config (`--config`, model path resolved relative to
  that file).
- `CMakeLists.txt` defines the `yolov10_camera` executable; `build/` is
  generated output and should not be edited manually.

## Build, Test, and Development Commands

Build on the SpaceMIT K3 board, where the riscv64 OpenCV, OpenCL, GStreamer,
and SpaceMIT ONNX Runtime libraries are available:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

If the SDK is outside system paths, add `-DSPACEMIT_ORT_ROOT=/path/to/sdk`.

Run the model and preprocessing smoke test without a camera:

```bash
./build/yolov10_camera \
  --config ../../backend/games/rps/adjudicator_config.json \
  --self-test --no-rtsp
```

The upstream sandbox for parameter tuning is the standalone sibling repo
`~/projects/dice-game/yolov10_objdetect` on the board; production behavior
is this package plus the per-game config.
