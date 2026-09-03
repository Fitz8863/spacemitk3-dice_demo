# Models

Model binaries are intentionally ignored by Git. Place the board-compatible
model files in this directory before running the application.

## SpaceMIT model used by this application

```text
models/yolov8n-seg.q.onnx
models/yolov8s-seg.q.onnx
```

The SpaceMIT `*.q.onnx` segmentation models use the 13-output layout expected
by the current detector:

- three DFL box branches;
- three class-score branches;
- three score-sum branches;
- three mask-coefficient branches;
- one `[1,32,160,160]` prototype output.

The current default configuration uses `yolov8s-seg.q.onnx`.

The `yolov8s-seg.q.onnx` file supplied from the SpaceMIT model archive has:

```text
size: 12095249 bytes
sha256: 294b21d44dfc85fd06b46966d69492c764a1387a4356d3dddea0fc458d3ee42d
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

This is a 2-output layout and is **not currently loadable by this application**,
which expects the SpaceMIT 13-output layout. Do not switch `config.json` to this
file until a standard Ultralytics 2-output postprocessor is added.

The locally downloaded official FP32 artifact has:

```text
size: 47498649 bytes
sha256: b3a62e190cab4f7ec46251dc3cf9826ba066d95e3b2b4f3c6f0e4b6fd5a31f76
```
