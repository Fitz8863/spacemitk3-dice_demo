# 游戏界面主题

界面固定使用「石墨灰绿」(`dark-arena`)，已移除主题选择器和萌趣主题。
`web/theme.js` 在 CSS 加载前应用石墨灰绿，主页面和实时画面加载页保持一致。
旧浏览器保存的 `spacemit.arena.theme.v1` 会被清理；即使浏览器禁止存储，也使用同一外观。
迁移只更新呈现，不创建对局或重新连接视频。

`web/styles.css` 继续提供共享组件结构，`web/dark-arena.css` 提供竞技界面外观。
宽屏布局将画面放在左侧，状态、进度和操作放在右侧；小屏使用纵向布局。
在 1920×1080 屏幕上按信息层级设置字号：大厅标题约 56px、局内标题约 40px、游戏名称约 32px、按钮文字约 24px、规则和判定说明约 20px、按键提示及辅助说明约 16–18px、导航和页脚约 14px。

大厅和规则页隐藏重复的彩色按键说明，统一保留左下角按键提示、规则内容及操作按钮。
对局开始、倒数、摇骰/出拳以及正在识别时，视频区域放大；结果锁定、判定失败或进入结果页后恢复普通布局。
`web/video-focus.js` 只读取页面阶段和判定标记，保留同一个 iframe 和视频连接；缩放采用 180ms transform 过渡，减少动态效果设置下直接切换布局。

按钮 ID、绿色确认/开始、蓝色重听/重试、红色返回、黄色向上选择以及游戏意图映射
继续沿用当前版本。UI 不新增后端游戏阶段，也不修改机械臂动作、ASR、TTS、裁决
配置或实际胜负数据。

验证：

```sh
python3 -m pytest tests/test_web_contract.py tests/test_web_keyboard_intents.py tests/test_ui_theme.py tests/test_display_layout.py tests/test_rps_lifecycle.py tests/test_video_focus.py -q
node tests/js/theme_regressions.cjs web/theme.js
node tests/js/rps_round_regressions.mjs .
node --test tests/test_stream_player.cjs
```

Node 测试需要 Node.js；没有 Node 的板端用实际页面验证旧主题迁移、游戏选择与刷新行为。

## 大会入口

石墨灰绿主题的大厅和待机页以「骰王挑战赛」为主。宽屏大厅用约 3:1 的主副卡片
展示摇骰子和猜拳；窄屏上下排列，骰子卡片保留更大的图形和高度。猜拳仍直接可选，
不增加二级入口。主副卡片大小不受当前选中游戏影响，选中态和确认按钮遵循实际选择。
待机页展示大会标题与本地骰子 SVG，不请求摄像头或创建对局；原待机唤醒通道不变。

首次加载、退出任意游戏或从待机唤醒返回大厅时优先选中启用的骰子游戏；骰子关闭或缺失时回退到
第一个启用游戏。只更改默认选择，不自动开局；确认、上下选择、点击和双击操作继续沿用。
本次不调整局内规则、视频布局、机械臂动作或裁决逻辑。

入口行为回归可单独运行：

```sh
node tests/js/hall_regressions.cjs web/app.js
```
