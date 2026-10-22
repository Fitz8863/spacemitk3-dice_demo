# 桌布颜色

骰子视觉程序支持两种桌布。在 `backend/games/dice/adjudicator_config.json` 设置：

```json
"divider_colors": "green_gray"
```

`green_gray` 适用于进迭绿 `#B0E237` 与灰色 `#666666`。程序按相机图像的色相、饱和度和亮度寻找两侧分界，支持横向、纵向及左右互换；不按设计色号逐像素精确匹配。省略字段或设置 `red_blue` 保持原红蓝桌布算法。修改后重启视觉进程生效。

该参数只寻找骰子区域的分界，不改变玩家／机械臂归属、点数、胜负或网页旋转；显示方向继续使用全局 `display.player_side`。猜拳的人手检测仍使用原来的玩家 ROI，无需因桌布颜色调整手势模型。

机械臂抓杯和标定登记桌面属于独立的 dice_demo 程序，另在其 `configs/green_cup.json` 的 `green_cup.perception` 中设置 `workspace: {"color": "green"}`，并删除旧的 `red_workspace`。详见 dice_demo 的 `docs/CONFIGURATION.md`。两项分别对应 USB 摄像头的对战分区和深度相机的机械臂工作区，不能互相替代。换桌布后应重新登记桌面平面。

先核对 `backend/components/robot_arm_nero/config.json` 的 `demo_root`，再编辑它指向目录中的 `configs/green_cup.json`。`dice_demo` 指仓内快照，`../dice_demo` 指主仓旁的独立开发目录；修改另一份配置不会影响当前抓杯程序。仓内快照通过上游同步维护，不能把旧快照参数误当作正在运行的独立仓库参数。

杯沿的绿色阈值描述杯子本身，与桌布工作区的阈值分开配置。工作区已选中杯子但仍报边缘支撑不足时，应检查双目红外图像、遮挡与曝光；不能通过扩大桌布区域或降低几何质量门槛代替杯沿验证。
