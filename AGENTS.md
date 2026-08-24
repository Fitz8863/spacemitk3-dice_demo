# Repository Guidelines

## Project Structure & Module Organization

This repository is a small C++17 SpaceMIT K3 camera-inference application:

- `src/main.cpp` owns CLI parsing, frame queues, display, and pipeline orchestration.
- `src/gstreamer_camera.{h,cpp}` handles V4L2/GStreamer capture, MJPEG decoding, and software-decoder fallback.
- `src/opencl_preprocess.{h,cpp}` implements NV12-to-YOLO preprocessing on OpenCL.
- `src/yolov8_detector.{h,cpp}` initializes SpaceMIT ONNX Runtime and performs YOLOv8 decode/NMS.
- `models/best.q.onnx` is the checked-in inference model.
- `CMakeLists.txt` defines the `yolov8_camera` executable; `build/` is generated output and should not be edited manually.

There is currently no dedicated test directory or test framework.

## Build, Test, and Development Commands

Build on the SpaceMIT K3 board, where the riscv64 OpenCV, OpenCL, GStreamer, and SpaceMIT ONNX Runtime libraries are available:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build build -j4
```

If the SDK is outside system paths, add `-DSPACEMIT_ORT_ROOT=/path/to/sdk`.

Run the model and preprocessing smoke test without a camera:

```bash
./build/yolov8_camera --model models/best.q.onnx --self-test --no-display
```

Run a bounded headless camera test with `--device /dev/videoN` or `--camera N`:

```bash
./build/yolov8_camera --model models/best.q.onnx --camera 1 \
  --no-display --max-frames 30
```

No automated `ctest` target is configured; treat the self-test and bounded camera run as required validation.

## Coding Style & Naming Conventions

Use C++17, four-space indentation, braces on the same line, and descriptive `snake_case` for local variables/functions. Use `PascalCase` for classes and `camelCase` only when matching external APIs. Keep headers self-contained, use RAII for GStreamer/OpenCL/ORT resources, and preserve compiler warnings (`-Wall -Wextra -Wpedantic`).

## Testing Guidelines

For changes affecting capture or preprocessing, validate both `--self-test --no-display` and a short `--max-frames` run on K3. Confirm the selected V4L2 node, decoder path, model output shape, and clean shutdown. This project has no coverage threshold or test naming convention yet.

## Commit & Pull Request Guidelines

No Git history is present in this checkout, so existing commit conventions cannot be verified. Use concise imperative messages, preferably with a scope, for example: `camera: add explicit V4L2 device option`. Pull requests should describe the hardware/runtime environment, commands executed, model and camera nodes used, decoder/EP behavior, and any performance or fallback changes; include logs or screenshots for visible pipeline changes.

## Configuration and Hardware Notes

Do not claim host builds prove K3 runtime support. Keep model paths and device paths configurable, avoid committing credentials or private calibration data, and document any SpaceMIT EP CPU fallback or `spacemitdec`/`jpegdec` selection observed during testing.
