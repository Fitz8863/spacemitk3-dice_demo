# 视频启动与复用

板端 `backend/config.json` 启用 `vision_always_on: true`。进入游戏时启动视频；返回菜单、刷新后重新进入同一游戏时复用仍存活且配置一致的进程。切换摇骰子与猜拳时先停止旧进程、释放共享摄像头，再启动对应模型，避免两个进程抢占设备。应用退出时释放所有视频资源。

如需游戏退出后立即释放摄像头，可改回 `false`，下一次进入会重新初始化。此开关不会在应用启动时提前打开摄像头，也不会跳过识别。

两款游戏的 `backend/games/<game>/adjudicator_config.json` 均设置 `fps: 24`，使用现场 C920 已验证的帧率。保留各自分辨率、识别区域和显示方向；不再先尝试不兼容的 1080p@25fps。

网页 `web/stream-player.js` 在连接失败后关闭旧 Reader（取消其内置 2 秒重试），前 20 次按 250ms 间隔重连，持续失败后退避到 2 秒。解码到真实视频帧后恢复快速重连预算，并隐藏宠物加载动画。旧连接回调和页面退出后的回调不能恢复播放器。摄像头断开时继续显示加载场景，不伪造视频就绪。

验证：板端运行 `python3 -m pytest -q tests/test_vision_stream.py tests/test_multi_game_vision.py tests/test_vision_control_protocol.py tests/test_vision_yolov10.py`；有 Node 的机器运行 `node --test tests/test_stream_player.cjs`。板端另测两种视频冷启动、退出后同游戏复用和跨游戏摄像头交接。首次进入及切换模型仍有冷启动成本；当前仍先初始化模型再打开摄像头。
