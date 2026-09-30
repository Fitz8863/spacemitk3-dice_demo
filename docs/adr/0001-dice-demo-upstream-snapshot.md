# dice_demo 以上游快照子目录融入 main

机械臂子系统（NERO 抓放/摇骰）的开发主仓是外部协作仓 wangannyi/dice_demo（hwj_dev 分支）。2026-09-30 决定用 `git subtree --squash` 把它以快照方式合入 main 的 `dice_demo/` 子目录：上游仍是机械臂代码唯一的开发地（协作同事工作流不变），main 克隆即得完整系统、发布产物与板端运行永远同源。

## Considered Options

- **git submodule**：克隆不自带内容、需要二次拉取和 network 往返，违背「整体性、给人读」的目标。
- **收口合并**（demo 仓冻结、开发收进 main）：协作方工作流被迫迁移，且上游仍需继续演进。
- **完整历史并入**：main 历史会多出数百个外部提交，阅读体验差；上游历史在原仓永远可追溯，无需复制。
- **上游快照 subtree（选定）**。

## Consequences

- `dice_demo/` 内部文件**不可在 main 侧修改**（下次 subtree pull 会冲突）；机械臂侧改动一律在上游仓提交，然后在 main 执行 `git subtree pull --prefix=dice_demo <上游路径> hwj_dev --squash` 同步并重启板端服务。
- 板上双位分工：上游仓检出 = 开发位（改码/pytest/标定），main 检出内 `dice_demo/` = 运行位（`robot_arm_nero` 的 `demo_root` 指向它，datasets 运行数据也写在这里）。
