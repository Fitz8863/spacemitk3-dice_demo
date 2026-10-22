# 桌布颜色

骰子视觉程序支持两种桌布。在 `backend/games/dice/adjudicator_config.json` 设置：

```json
"divider_colors": "green_gray"
```

`green_gray` 适用于进迭绿 `#B0E237` 与灰色 `#666666`。程序按相机图像的色相、饱和度和亮度寻找两侧分界，支持横向、纵向及左右互换；不按设计色号逐像素精确匹配。省略字段或设置 `red_blue` 保持原红蓝桌布算法。修改后重启视觉进程生效。

该参数只寻找骰子区域的分界，不改变玩家／机械臂归属、点数、胜负或网页旋转；显示方向继续使用全局 `display.player_side`。猜拳的人手检测仍使用原来的玩家 ROI，无需因桌布颜色调整手势模型。

机械臂抓杯和标定登记桌面属于独立的 dice_demo 程序，另在其 `configs/green_cup.json` 的 `green_cup.perception` 中设置 `workspace: {"color": "green"}`，并删除旧的 `red_workspace`。详见 dice_demo 的 `docs/CONFIGURATION.md`。两项分别对应 USB 摄像头的对战分区和深度相机的机械臂工作区，不能互相替代。换桌布后应重新登记桌面平面。
