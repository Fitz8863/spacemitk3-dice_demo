# Models

Model binaries are intentionally ignored by Git. Place the board-compatible
model files in this directory before running the application.

## SpaceMIT 13-output models

```text
models/yolov8n-seg.q.onnx
models/yolov8s-seg.q.onnx
```

The SpaceMIT `*.q.onnx` segmentation models use the 13-output layout
(auto-detected by the detector alongside the standard 2-output layout):

- three DFL box branches;
- three class-score branches;
- three score-sum branches;
- three mask-coefficient branches;
- one `[1,32,160,160]` prototype output.

The current default configuration uses `yolov8s-seg.fp32.q.onnx`.

The `yolov8s-seg.q.onnx` file supplied from the SpaceMIT model archive has:

```text
size: 12095249 bytes
sha256: 294b21d44dfc85fd06b46966d69492c764a1387a4356d3dddea0fc458d3ee42d
```

## SpaceMIT-quantized standard 2-output model

```text
models/yolov8s-seg.fp32.q.onnx
```

This is the official Ultralytics YOLOv8s-seg graph quantized for the SpaceMIT
EP while keeping the standard 2-output layout:

```text
input:   [1,3,640,640]
output0: [1,116,8400]  (4 box + 80 class logits + 32 mask coefficients)
output1: [1,32,160,160] (prototype)
```

The detector auto-detects this layout by output count. As with the SpaceMIT
13-output exports, class scores are already activated probabilities inside the
graph and are used directly by the postprocessor (verified 2026-09-09; do not
apply a second sigmoid).

The local artifact has:

```text
size: 12265685 bytes
sha256: fc3ffd7f18f0c135c0dbc3582840e9bbe8233ba8a05fbe57987853fa094b98e5
```

## Official Ultralytics FP32 model

The official Ultralytics release artifact can also be kept here as:

```text
models/yolov8s-seg.fp32.onnx
```

It is a standard FP32 ONNX export with:

```text
input:  [1,3,640,640]
output: [1,116,8400] and [1,32,160,160]
```

This 2-output layout loads and runs, but the official export keeps **raw class
logits** while the current standard postprocessor expects already activated
probabilities (SpaceMIT export convention) — confidence values from this file
will be misinterpreted. Kept for reference/对比; do not use it as a daily
driver unless a sigmoid variant of the postprocessor is added.

The locally downloaded official FP32 artifact has:

```text
size: 47498649 bytes
sha256: b3a62e190cab4f7ec46251dc3cf9826ba066d95e3b2b4f3c6f0e4b6fd5a31f76
```
