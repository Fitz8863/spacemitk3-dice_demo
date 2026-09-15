# Models

本目录存放板端可加载的 YOLOv8-seg ONNX 模型。除 `best.q.onnx` 外，
`models/*.onnx` 被 `.gitignore` 忽略，需要自行放置。

## 当前默认模型：`best.q.onnx`

自训练的 **盖子（骰盅盖）分割模型**，用于机械臂抓取定位。随仓库分发。

```text
input:   [1,3,640,640]
output0: [1,38,8400]     (4 box + 2 class + 32 mask 系数)
output1: [1,32,160,160]  (prototype)
```

- 输出布局：**标准 Ultralytics 2 输出**，由 `Yolov8SegDetector::init()` 按输出数量自动识别。
- 类别：**2 类**，`config.json` 的 `class_names` 必须与之逐项同序：
  ```json
  ["cap", "ground"]
  ```
- 类别分数是**图内已激活的概率**，后处理直接与 `conf` 比较，**不要再叠加 sigmoid**；
  这一约定与 SpaceMIT 导出格式一致，见 `src/yolov8_postprocess.cpp` 的 `decode_standard2()`。

本地制品：

```text
size:   12164138 bytes
sha256: ed93624195eb1869d7132f474b81b8212871e9df45298ab98a6468fbd7bd18a3
```

### 训练与标注约定

- 数据集任务类型为 `segment`，标签为**归一化多边形**（`class x1 y1 x2 y2 ...`）。
- 标注平台：Ultralytics Platform（数据集 `fitz/datasets/gai`）。
- ⚠️ **导出训练集时必须剔除未标注图片**：Ultralytics 会把没有 label 文件的图片
  当作背景负样本参与训练，导致模型在"明明有盖子"的图上学习"这里什么都没有"。
- 类别索引必须与 `config.json` 的 `class_names` **严格同序**。程序仅在类别**数量**
  不一致时告警（`src/yolov8_seg_detector.cpp`），数量相同但顺序错是**静默出错**。

## 其他兼容格式（不在本仓库内）

程序支持两种输出布局，按输出数量自动识别：

| 布局 | 输出 | 说明 |
|---|---|---|
| SpaceMIT 13 输出 | 3×(DFL box, class, score-sum) + 3×mask 系数 + proto | 走 DFL 解码，类别分已是概率 |
| 标准 2 输出 | `[1, 4+C+32, anchors]` + `[1,32,160,160]` | 通道 0-3 为 cx,cy,w,h |

两者共用同一套 NMS、mask 组装与坐标映射；类别数分别从各自输出形状推导
（`classes = channels - 4 - 32`），因此换类别数的模型无需改代码。

⚠️ **官方 Ultralytics 原始导出（FP32，未量化）保留的是原始 class logit**，
与当前后处理期望的"已激活概率"不符，直接用会导致置信度被误读；
如需使用，必须改用 SpaceMIT 的导出/量化约定，或给后处理增加 sigmoid 变体。
