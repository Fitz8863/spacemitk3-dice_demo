# 游戏界面主题

默认主题为「石墨灰绿」(`dark-arena`)。页面右上角「界面主题」可以切换到原有
「萌趣游乐舱」(`storybook`)，切换仅影响呈现，不重建对局或重新连接视频。

主题选择以 `spacemit.arena.theme.v1` 保存在当前浏览器的 localStorage 中。
新浏览器、未保存选择或保存值无效时均使用石墨灰绿竞技舱；明确选过萌趣主题的浏览器
刷新后继续使用萌趣主题。存储不可用时主题仍可在当前页面切换。

`web/theme.js` 在 CSS 加载前设置根元素主题，避免启动时先闪现浅色。
`web/dark-arena.css` 的竞技舱样式均限定在对应主题下；原来的 `web/styles.css`
保留原样。实时画面的加载页也使用同一主题。宽屏布局将画面放在左侧，状态、进度和
操作放在右侧；小屏使用纵向布局。石墨灰绿版本在 1920×1080 屏幕上按信息层级设置字号：大厅标题约 56px、局内标题约 40px、游戏名称约 32px、按钮文字约 24px、规则和判定说明约 20px、按键提示及辅助说明约 16–18px、导航和页脚约 14px。辅助按键提示由页脚统一承载，保留游戏规则和操作按钮。

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

Node 测试需要 Node.js；没有 Node 的板端用实际页面验证主题、选择与刷新行为。
