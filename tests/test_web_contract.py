import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def dice_manifest():
    return json.loads(
        (ROOT / "backend/games/dice/manifest.json").read_text(encoding="utf-8")
    )


def rps_manifest():
    return json.loads(
        (ROOT / "backend/games/rps/manifest.json").read_text(encoding="utf-8")
    )


def dice_state(name):
    return dice_manifest()["state_machine"]["states"][name]


def test_frontend_has_no_hardcoded_mediamtx_host():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "127.0.0.1:8889" not in html
    assert "data-stream-url" not in html
    assert "event.event === 'video'" in js


def test_frontend_keeps_video_until_terminal_complete():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")

    assert "phase === 'holding'" in js
    assert "'round_complete'" in app
    assert "startVisionStream(event)" in js
    assert "stopVisionStream()" in js


def test_frontend_preserves_holding_countdown_from_structured_event():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    holding = js.split("event.phase === 'holding'", 1)[1].split("}", 1)[0]

    assert "remaining_ms" in holding
    assert "实时画面将继续播放" in js
    assert "Math.ceil(remaining / 1000)" in holding


def test_frontend_open_phase_prompts_readiness_without_duplicate_banner():
    """The open phase shows one lowered prompt; the old bottom banner is gone."""
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    css = (ROOT / "web/styles.css").read_text(encoding="utf-8")

    open_section = html.split('data-view="open"', 1)[1].split("</section>", 1)[0]
    assert "请同时开盖" not in open_section
    assert "请同时开盖" not in html
    assert "你准备好了吗？听语音倒计时同时开盖" in js
    assert "document.body.dataset.phase = phase" in app
    assert 'body[data-phase="open"] .stage-head' in css


def test_frontend_schedules_streamed_tts_frames_back_to_back():
    """Streamed TTS frames play on one WebAudio timeline, not per-frame Audio.

    Per-frame ``Audio`` elements left an audible gap between every ~1s frame
    of remote streaming TTS; the scheduler decodes each frame and lines the
    buffers up back-to-back instead. The Audio-element path stays as the
    no-WebAudio fallback.
    """
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")

    assert "function createSpeechScheduler" in app
    assert "decodeAudioData" in app
    assert "createBufferSource" in app
    assert "player.waitDrained" in app
    assert "await scheduler.schedule(blob)" in app
    assert "await playSpeechBlob(blob, requestId)" in app  # fallback kept


def test_frontend_result_copy_has_no_doubled_llm_prefix():
    """The result subtitle must not render 大模型 twice for the yolo_only case."""
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "当前未启用大模型" in js
    assert "大模型未启用大模型" not in js


def test_frontend_buttons_match_controller_key_colors():
    """Page buttons mirror the physical controller color semantics.

    绿=Enter 确认/开始，蓝=ArrowDown 向下/重听/重试，红=Escape 停止/返回；
    短文案用圆形，长文案用胶囊。旧调色板类必须移除以免颜色语义分叉。
    """
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    css = (ROOT / "web/styles.css").read_text(encoding="utf-8")
    lines = html.splitlines()

    def class_of(button_id):
        line = next(line for line in lines if f'id="{button_id}"' in line)
        return line.split("class=\"", 1)[1].split("\"", 1)[0]

    assert class_of("startGame") == "btn-circle btn-green"
    assert class_of("confirmRules") == "btn-circle btn-green"
    assert class_of("startShake") == "btn-circle btn-green"
    assert class_of("analysisNewRound") == "btn-circle btn-green"
    assert class_of("newRound") == "btn-circle btn-green"
    assert class_of("repeatRules") == "btn-circle btn-blue"
    assert class_of("analysisRetry") == "btn-circle btn-blue"
    # 停止摇骰保留给人工模式（manual_shaking）：默认隐藏，进入人工模式才显示。
    assert class_of("stopShake") == "btn-circle btn-red hidden"
    # 机械臂失败页三键：红=退出、黄=人工模式（新色，须与实体按键提示同源）、蓝=重试。
    assert class_of("armBack") == "btn-pill btn-red"
    assert class_of("armManual") == "btn-circle btn-yellow"
    assert class_of("armRetry") == "btn-circle btn-blue"
    assert class_of("backFromRules") == "btn-circle btn-red"
    assert class_of("readyBack") == "btn-circle btn-red"
    assert class_of("analysisBackToGames") == "btn-pill btn-red"
    assert class_of("backToGames") == "btn-pill btn-red"

    assert "primary-button" not in html
    assert "secondary-button" not in html
    assert "stop-button" not in html
    for token in (".btn-circle", ".btn-pill", ".btn-green", ".btn-red", ".btn-blue"):
        assert token in css
    # 按钮色必须与 controller-key 提示圆点同源
    assert css.count("background: #16a34a") == 2
    assert css.count("background: #dc2626") == 2
    assert css.count("background: #2563eb") == 2


def test_frontend_shouts_stop_before_reveal_ready():
    """The 停 → reveal rhythm is declared by the backend state machine."""
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")

    open_reveal = dice_state("open_reveal")
    stop_entry = open_reveal["on_enter"][0]
    assert stop_entry["mode"] == "audio"
    assert stop_entry["audio"] == "audio/停.wav"
    assert stop_entry["text"].strip() == "停！"
    assert stop_entry["await"] is True
    # The reveal clip follows in the same on_enter sequence and is not awaited,
    # so the hold window below runs in parallel with the voice instead of
    # queueing up behind it.
    assert open_reveal["on_enter"][1]["text"].startswith("准备好了没有")
    assert open_reveal["on_enter"][1].get("await") is not True
    assert open_reveal["duration"] == 1.6

    # The frontend must not re-implement the chain: awaiting is an engine
    # concern, playback acknowledgement happens through the round client.
    assert "function stopShake" not in js
    assert "speakState" not in js
    assert "startRevealTransition" not in js
    assert "submitIntent('speech_done'" in app


def test_frontend_distinguishes_llm_override_from_consensus():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    assert "llm_override" in js


def test_frontend_renders_structured_diagnosis_and_retry_prompt():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    assert "showDiagnosis" in js
    assert "本次裁决未完成" in js
    assert "diagnosis.message" in js
    assert "analysisFailureActions" in js


def test_frontend_blue_button_retries_adjudication_after_diagnosis():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert 'aria-keyshortcuts="ArrowDown"' in html
    # 蓝键（↓）在 analysis 失败页仍是重试；arm_failed 失败页是机械臂重试。
    # 两条都从同一个按键分支出发，与屏幕按钮走同一个 handler。
    assert "else if (state.phase === 'analysis' && analysisFailureVisible()) submitIntent('retry');" in js
    assert "else if (state.phase === 'arm_failed') handlers.armRetry();" in js
    # Retrying re-enters adjudication through the backend state machine.
    assert "submitIntent('retry')" in js
    # The retry hint itself is spoken by the backend on entering the failed state.
    retry_entry = dice_state("analysis_failed")["on_enter"][0]
    assert retry_entry["mode"] == "tts_local"
    assert "重新识别" in retry_entry["text"]


def test_frontend_diagnosis_marks_detection_failed_and_shows_evidence():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    css = (ROOT / "web/styles.css").read_text(encoding="utf-8")
    diagnosis = js.split("function showDiagnosis", 1)[1].split("function showResult", 1)[0]

    assert "markAnalysisFailure()" in diagnosis
    assert "diagnosisDetails(diagnosis)" in diagnosis
    assert "reason_code" in js
    assert "detected_counts" in js
    assert "textContent = '✕'" in js
    assert ".analysis-step.failed" in css


def test_frontend_failure_state_offers_retry_new_round_or_game_list():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert 'id="analysisRetry"' in html
    assert 'id="analysisNewRound"' in html
    assert 'id="analysisBackToGames"' in html
    assert "重新识别" in html
    assert "再来一局" in html
    assert "退出游戏，返回列表" in html
    assert "analysisRetry: () => submitIntent('retry')" in js
    assert "analysisNewRound: () => submitIntent('new_round')" in js
    assert "analysisBackToGames: () => submitIntent('back')" in js
    # Rendering a diagnosis must not relaunch adjudication by itself; only
    # submitting the retry intent may re-enter the analysis state.
    diagnosis = js.split("function showDiagnosis", 1)[1].split("function handleRoundEvent", 1)[0]
    assert "submitIntent" not in diagnosis


def test_frontend_does_not_mark_yolo_complete_while_still_detecting():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    detecting = js.split("if (event.phase === 'detecting')", 1)[1].split(
        "} else if (event.phase === 'verifying')", 1
    )[0]
    assert "querySelector('span').textContent = '…'" in detecting
    assert "querySelector('span').textContent = '✓'" not in detecting
    assert "以大模型为准" in js


def test_frontend_uses_manifest_participant_layout_and_role_result():
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    dice = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")

    assert "module.enter(manifest)" in app
    assert "playerScoreSide" in html
    assert "agentScoreSide" in html
    assert "result.player_values" in dice
    assert "result.agent_values" in dice
    assert "result.player_score" in dice
    assert "result.agent_score" in dice
    assert "result.winner_role" in dice
    assert "result.player_side" in dice
    assert "result.agent_side" in dice
    assert "winner === 'LEFT'" not in dice
    assert "result.first_dice" not in dice
    assert "result.second_dice" not in dice
    assert "result.llm_winner" not in dice
    assert re.search(r"result\.winner(?!_role)", dice) is None


def test_frontend_enters_open_transition_and_starts_countdown_automatically():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")

    # Timers, transitions, and the reveal chain all live in the backend
    # state machine; the game module renders events and submits intents only.
    assert "setTimeout" not in js
    assert "setInterval" not in js
    assert "setPhase('open'" not in js
    assert 'id="revealDice"' not in html
    open_reveal = dice_state("open_reveal")
    assert open_reveal["on_enter"][0].get("await") is True
    assert open_reveal["on_expire"]["to"] == "vision_countdown"
    reveal_entry = open_reveal["on_enter"][1]
    # The reveal countdown is a pre-recorded clip now, and it is deliberately
    # NOT awaited: the hold timer must run *with* the voice so the screen
    # countdown starts on the voice's "三".
    assert reveal_entry["mode"] == "audio"
    assert reveal_entry["audio"].endswith(".wav")
    assert reveal_entry.get("await") is not True
    assert reveal_entry["text"]


def test_frontend_counts_down_after_open_transition_before_adjudication():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    # The vision countdown is a backend state; the frontend only renders ticks.
    # 0.8s per number matches the reveal voice's 三二一 pacing.
    vision_countdown = dice_state("vision_countdown")
    assert vision_countdown["duration"] == 2.4
    assert vision_countdown["tick_seconds"] == 0.8
    assert vision_countdown["on_expire"]["to"] == "analysis"
    assert vision_countdown["ui"]["view"] == "countdown"
    assert "请保持骰子和骰盅位置不动" in vision_countdown["ui"]["copy"]
    assert "countdownNumber" in js
    assert "revealDice" not in js


def test_reveal_voice_and_screen_countdown_stay_in_step():
    """The two halves of the reveal rhythm must be edited together.

    The voice clip counts 三/二/一 and the *screen* numbers come from
    ``vision_countdown``; they only line up because ``open_reveal`` hands over
    at the moment the voice reaches 三.  That handover is a single number:
    ``open_reveal.duration``, which starts once the awaited 停 clip is
    acknowledged (the engine only starts a state's timer after its on_enter
    sequence has been issued) and must therefore cover the reveal clip's
    lead-in up to its first number (~1.5s) — not a "transition length" chosen
    for feel.  Changing any one of these four facts desynchronises the demo,
    so pin them together.
    """
    open_reveal = dice_state("open_reveal")
    stop_entry = open_reveal["on_enter"][0]
    reveal_entry = open_reveal["on_enter"][1]
    vision_countdown = dice_state("vision_countdown")

    # 停 must still finish first, and only 停 may block the sequence.
    assert stop_entry["await"] is True
    assert reveal_entry.get("await") is not True

    # Handover at the voice's 三 (~1.5s into the clip + browser start latency).
    assert open_reveal["duration"] == 1.6
    assert open_reveal["on_expire"]["to"] == "vision_countdown"

    # The screen side: three numbers, 0.8s apart, ending as the clip does.
    assert vision_countdown["duration"] == 2.4
    assert vision_countdown["tick_seconds"] == 0.8
    assert vision_countdown["on_expire"]["to"] == "analysis"


def test_frontend_uses_vision_specific_copy_during_post_open_countdown():
    copy = dice_state("vision_countdown")["ui"]["copy"]
    assert "倒计时结束后开始视觉裁决" in copy


def test_robot_shake_is_event_driven_with_manual_fallback():
    """机械臂摇骰契约（2026-09-23 集成拍板 + 过场态改造）：

    - game_start 过场（用户拍板 2026-09-23 晚）：入口即刻下发抓取（advance until
      GRIP，识别+抓取不摇），大字"游戏开始！"动画 + 臂进度行；duration 就是过场
      秒数的唯一旋钮（热加载可调，默认 3s——抓取实测 3.5-4s，过场+2.4s 倒计时
      后臂已持杯待命）；只声明失败路由，抓取提前完成不打断过场
    - shake_countdown 只剩三二一口令（抓取已前移）；保留 grasp 失败路由——
      抓取偶发偏慢时失败事件落在倒计时态，仍能正确进兜底页
    - shaking 无定时无停止按钮：入场即下发摇，完成/失败双路由
    - arm_failed 兜底：蓝=重试回 game_start（重新过场+重新抓取）、黄=人工、
      红=退出；manual_shaking 30s 兜底 + 停止按钮
    """
    machine = dice_manifest()["state_machine"]

    intro = machine["states"]["game_start"]
    assert intro["duration"] == 3.0
    assert intro["on_enter"] == [
        {"action": "robot", "command": "grasp_cup", "timeout_seconds": 30}
    ]
    assert intro["on_expire"]["to"] == "shake_countdown"
    assert intro["on_event"] == {"robot.grasp_cup.failed": {"to": "arm_failed"}}
    assert machine["states"]["ready"]["on_intent"]["start_shake"]["to"] == "game_start"

    countdown = machine["states"]["shake_countdown"]
    assert countdown["on_enter"][0]["mode"] == "audio"
    assert countdown["on_enter"][0]["audio"] == "audio/warm_321开始.wav"
    assert len(countdown["on_enter"]) == 1
    assert countdown["duration"] == 2.4
    assert countdown["on_expire"]["to"] == "shaking"
    assert countdown["on_event"] == {"robot.grasp_cup.failed": {"to": "arm_failed"}}

    shaking = machine["states"]["shaking"]
    assert "duration" not in shaking
    assert "on_intent" not in shaking
    assert "on_expire" not in shaking
    shake = shaking["on_enter"][1]
    assert shake == {"action": "robot", "command": "shake_dice", "timeout_seconds": 90}
    assert shaking["on_event"] == {
        "robot.shake_dice.completed": {"to": "vision_countdown"},
        "robot.shake_dice.failed": {"to": "arm_failed"},
    }

    arm_failed = machine["states"]["arm_failed"]
    intents = arm_failed["on_intent"]
    assert intents["retry"]["to"] == "game_start"
    assert intents["manual"]["to"] == "manual_shaking"
    assert intents["back"]["exit"] is True

    manual = machine["states"]["manual_shaking"]
    assert manual["duration"] == 30
    assert manual["on_intent"]["stop_shake"]["to"] == "vision_countdown"
    assert manual["on_expire"]["to"] == "vision_countdown"

    # The manual intent must stay voice-reachable.
    assert dice_manifest()["asr"]["phrases"]["manual"]

    # 前端布局：过场页大字动画 + 臂进度行；倒计时页与摇骰页各一条进度行；
    # 人工模式的 30s 计时与停止按钮默认隐藏。
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    css = (ROOT / "web/styles.css").read_text(encoding="utf-8")
    assert 'data-view="game_start"' in html
    assert "游戏开始！" in html
    assert 'id="armGameStartRow"' in html
    assert ".game-start-text" in css
    assert 'id="armCountdownRow"' in html
    assert 'id="armShakingRow"' in html
    assert 'id="shakeTimerRow"' in html
    assert 'id="shakeSeconds">30<' in html
    assert "const urgent = seconds <= 3;" in js
    assert "updateArmProgress(event)" in js


def test_frontend_uses_user_gesture_audio_for_countdown_cues():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "window.AudioContext || window.webkitAudioContext" in js
    assert "prepareCountdownAudio()" in js
    assert "countdownAudioContext.resume()" in js
    assert "oscillator.connect(gain)" in js
    assert "frequency.setValueAtTime" in js
    assert "seconds === 1" in js
    assert "if (!state.sound" in js


def test_frontend_uses_light_theme_and_high_contrast_urgent_styles():
    css = (ROOT / "web/styles.css").read_text(encoding="utf-8")
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")

    assert "--bg: #f7f9fc" in css
    assert "--surface: #ffffff" in css
    assert ".shake-timer strong.is-urgent" in css
    assert "color: var(--loss)" in css
    assert "@keyframes urgentPulse" in css
    assert 'content="#f7f9fc"' in html


def test_frontend_uses_louder_tense_warning_tone_for_urgent_countdown():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "oscillator.type = 'triangle';" in js
    assert "const secondaryOscillator = context.createOscillator();" in js
    assert "const volume = seconds === 1 ? 0.28 : 0.22;" in js
    assert "secondaryOscillator.frequency.setValueAtTime" in js
    assert "secondaryOscillator.start(secondaryStart)" in js
    assert "oscillator.stop(now + 0.28)" in js


def test_frontend_plays_shake_started_with_get_ready_countdown_not_ten_second_timer():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    # The start cue is the shake_countdown state's on_enter audio, and the
    # frontend no longer owns any shake timer or countdown orchestration.
    countdown_state = dice_state("shake_countdown")
    entry = countdown_state["on_enter"][0]
    assert entry["mode"] == "audio"
    assert entry["audio"] == "audio/warm_321开始.wav"
    # 0.8s per number, 2.4s total: the countdown ends as the clip says 开始.
    assert countdown_state["duration"] == 2.4
    assert countdown_state["tick_seconds"] == 0.8
    assert "function beginShake" not in js
    assert "SHAKE_DURATION_SECONDS" not in js
    assert "start_shake" in js


def test_frontend_removes_decorative_status_labels_and_keyboard_hints():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    css = (ROOT / "web/styles.css").read_text(encoding="utf-8")
    js = (ROOT / "web/app.js").read_text(encoding="utf-8")

    assert 'id="phaseKicker"' not in html
    assert 'id="progressDots"' not in html
    assert 'class="app-footer"' not in html
    assert 'class="hint"' not in html
    assert "$('phaseKicker')" not in js
    assert "$('stageFooterText')" not in js
    assert ".kicker" not in css
    assert ".keycap" not in css


def test_frontend_maps_controller_colors_to_navigation_keys():
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    dice = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "event.key === 'Enter'" in app
    assert "event.key === 'Escape'" in app
    assert "event.key === 'ArrowDown'" in app
    assert "event.key === 'ArrowUp'" in app
    assert "event.key === 'Enter'" in dice
    assert "event.key === 'Escape'" in dice
    assert "event.key === 'ArrowDown'" in dice
    assert "event.key === 'ArrowUp'" in dice
    assert "event.key.toLowerCase() === 'q'" not in dice


def test_frontend_hides_unused_round_indicator():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    css = (ROOT / "web/styles.css").read_text(encoding="utf-8")
    dice = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert 'class="round-badge"' not in html
    assert 'id="roundNumber"' not in html
    assert '.round-badge' not in css
    assert 'roundNumber' not in dice


def test_frontend_uses_color_controller_hints_for_navigation_copy():
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    css = (ROOT / "web/styles.css").read_text(encoding="utf-8")

    assert 'controller-key-green' in app
    assert 'controller-key-blue' in app
    assert 'controller-key-red' in app
    assert 'controller-key-yellow' in app
    assert 'controller-hint' in app
    assert 'aria-label="黄色按钮"' in app
    assert 'aria-label="蓝色按钮"' in app
    assert 'aria-label="绿色按钮"' in app
    assert 'aria-label="红色按钮"' in app
    assert '>↑<' not in app
    assert '>↓<' not in app
    assert '>✓<' not in app
    assert '>↻<' not in app
    assert '>↩<' not in app
    assert 'renderPhaseCopy(phase, resolved[1])' in app
    assert '.controller-key-green' in css
    assert '.controller-key-blue' in css
    assert '.controller-key-red' in css
    assert '.controller-key-yellow' in css
    assert 'border-radius: 50%' in css
    assert '.controller-hint' in css
    assert '听完规则后按 Enter 确认' not in app


def test_frontend_hides_result_phase_copy_when_empty():
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    dice = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "node.classList.add('hidden')" in app
    assert "result: ['本局结果', '']" in dice
    assert '点数已经锁定，看看谁赢下了这一局。' not in dice


def test_frontend_never_shows_stable_count_above_threshold():
    """The analysis copy must not print a streak above stable_frames.

    The runtime no longer counts frames it cannot adjudicate, but an older
    resident binary could still report "48/30", so the page clamps as well.
    """
    dice = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    detecting = dice.split("if (event.phase === 'detecting')", 1)[1].split(
        "} else if (event.phase === 'verifying')", 1
    )[0]
    assert "Math.min(count, required)" in detecting
    assert "${shownCount}/${required}" in detecting
    assert "${count}/${required}" not in detecting


def test_frontend_round_client_drives_the_game():
    """The game module submits intents and renders; it never advances state."""
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "createRoundClient" in app
    assert "/api/game/rounds" in app
    assert "`/api/game/rounds/${round.roundId}/speech`" in app
    assert "EventSource(`/api/game/rounds/${roundId}/stream`)" in app
    assert "submitIntent" in js
    assert "createRoundClient" in js
    # Intent outcomes that conflict with the current state are normal play.
    assert "ROUND_INTENT_REJECTED" in app
    assert "error.silent" in app


def test_frontend_acks_awaited_directives_and_mutes_cleanly():
    """所有指令播完都回执（await 唤醒等待者、非 await 释放播报闸）；静音立即回执。"""
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")

    play = app.split("async function playDirective", 1)[1].split(
        "// ---- 权威对局客户端", 1
    )[0]
    # Acknowledgement is unconditional: every directive (awaited or not)
    # releases the engine's speech-gate registration when playback ends.
    assert "submitIntent('speech_done'" in play
    assert "directive.await &&" not in play
    # All exit paths acknowledge exactly once via the shared finally block
    # (the last one in the function — the producer closure has its own).
    body = play.rsplit("} finally {", 1)[1]
    assert "acknowledge()" in body
    # Muted playback must still acknowledge immediately.
    muted = play.split("if (!state.sound)", 1)[1].split("}", 1)[0]
    assert "acknowledge()" in muted


def test_reveal_hold_delays_the_acknowledgement_not_the_playback():
    """开盖转场的 2 秒停顿＝压后回执，不是压后播放。

    后端对 await 的台词是「收到 speech_done 才发下一句、才启动该状态的
    duration 计时」，所以把回执压后 N 秒会让下游整条序列一起后移，语音与
    屏幕倒计时的相对对齐不受影响。反过来，只把音频延后播放的话，屏幕倒计时
    仍按原时刻出现——两者会错开约 2 秒，正是要避免的回归。
    """
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    stop_audio = dice_state("open_reveal")["on_enter"][0]["audio"]

    # The policy is hardcoded in the game module and names the stop clip, so a
    # manifest rename fails here instead of silently dropping the pause.
    assert "REVEAL_HOLD_SECONDS = 2" in js
    assert f"REVEAL_STOP_AUDIO = '{stop_audio}'" in js
    assert "ackHoldSeconds: hold" in js
    # dice.js stays timer-free: the wait lives in the engine layer.
    assert "setTimeout" not in js

    # The engine holds the acknowledgement, and only after a normal finish so a
    # cancelled or superseded line still acknowledges at once.
    play = app.split("async function playDirective", 1)[1].split(
        "// ---- 权威对局客户端", 1
    )[0]
    assert "options.ackHoldSeconds" in play
    assert "playbackComplete && ackHoldSeconds > 0" in play
    assert "waitSeconds(ackHoldSeconds)" in play


def test_frontend_surfaces_asr_recognition_feedback():
    """每句 ASR 识别结果都要有可见反馈：生效/播报闸暂缓/不支持/未匹配。"""
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    css = (ROOT / "web/styles.css").read_text(encoding="utf-8")

    # Feedback lives in the engine layer (any game benefits) and taps the
    # round event stream — no separate transport, no game-module awareness.
    assert "event.event === 'asr'" in app
    assert "showAsrFeedback(event)" in app
    assert "asrFeedback" in html
    assert "aria-live=\"polite\"" in html.split('id="asrFeedback"', 1)[1].split(">", 1)[0]
    # All four outcomes are distinguishable, and the speech-gate one points
    # the player at the physical buttons instead of just going silent.
    assert "status === 'submitted'" in app
    assert "status === 'suppressed'" in app
    assert "status === 'rejected'" in app
    assert "未匹配语音指令" in app
    assert "不想等可按绿色按钮" in app
    assert ".asr-feedback" in css
    assert ".asr-feedback.level-ok" in css
    assert ".asr-feedback.level-wait" in css
    assert ".asr-feedback.level-info" in css


def test_manifest_voice_phrases_cover_every_state_intent():
    """每个可按键触发的意图都必须有语音触发词（speech_done 除外）。

    回归守护：2026-09-03 实测 ready 状态说"确定"无效——confirm 只在
    rules 声明，start_shake/retry/new_round 等完全没有词表。新状态/
    新意图接入时若漏配语音，这里直接报出缺口位置。
    """
    manifest = dice_manifest()
    phrases = manifest["asr"]["phrases"]
    missing = []
    for state_name, state in manifest["state_machine"]["states"].items():
        for intent in (state.get("on_intent") or {}):
            if intent == "speech_done":
                continue
            if intent not in phrases:
                missing.append(f"{state_name}.{intent}")
    assert not missing, f"语音不可达的意图（asr.phrases 缺条目）: {missing}"
    # Dual-purpose words are the mechanism that lets one word follow the
    # game: 确定 confirms in rules and starts the shake in ready.
    assert "确定" in phrases["confirm"]
    assert "确定" in phrases["start_shake"]


def test_ready_start_button_can_interrupt_the_opening_announcement():
    """ready 的绿键不必等开场播报念完（2026-09-21 恢复旧行为）。

    回归守护：9249ee9（2026-09-18）曾给 start_shake 声明 after_speech 闸
    （播报未完按绿键 409 静默拒绝），2026-09-21 按用户要求撤销——播报中途
    按绿键/Enter 直接打断进入过场（2026-09-23 晚起目标态为 game_start），
    残留播报由后续状态的入场语音顶替（前端顶替路径停止旧音频并立即回执
    speech_done）。这条声明被误加回来时这里先红。
    """
    ready = dice_state("ready")
    start = ready["on_intent"]["start_shake"]
    assert start["to"] == "game_start"
    assert "after_speech" not in start
    # back 同样不设闸：播报中途仍可返回规则页。
    assert "after_speech" not in ready["on_intent"]["back"]


def test_rps_vision_contract_pins_the_confirmed_flow_decisions():
    """rps 视觉契约（2026-09-21 视觉接入后与用户确认的决策，防误删/误加）：

    - providers 覆盖 vision_adjudicator → vision_yolov10_objdetect（独立
      视觉包，不走全局默认的 v8 槽位）
    - vision_profile 声明 runtime_config 指向 rps 自己的 adjudicator_config；
      video 关闭但 path 保留 /rps/det（用户 2026-09-21 晚拍板：浏览器不显示
      画面，识别/推流照常；将来要开画面只翻 enabled，路径不动）
    - confirm 不设 after_speech（老玩家可跳过规则宣读直接开始）
    - play 口令 await（念到「布」亮手势，念完即裁决），analysis 双路由齐全
    - 再来一局直接回 play（不重读规则）
    - asr 短语覆盖全部可按键意图（镜像 dice 那条契约）
    """
    manifest = rps_manifest()
    assert manifest["enabled"] is True
    assert manifest["providers"]["vision_adjudicator"] == "vision_yolov10_objdetect"
    profile = manifest["vision_profile"]
    assert profile["game_id"] == "rps"
    assert profile["runtime_config"] == "backend/games/rps/adjudicator_config.json"
    assert profile["video"] == {"enabled": False, "path": "/rps/det"}
    assert profile["llm"]["enabled"] is False  # 纯视觉裁决，无 LLM 复核
    states = manifest["state_machine"]["states"]
    rules = states["rules"]
    assert "after_speech" not in rules["on_intent"]["confirm"]
    assert rules["on_intent"]["confirm"]["to"] == "play"
    chant = states["play"]["on_enter"][0]
    assert chant["await"] is True
    analysis = states["analysis"]
    assert analysis["on_event"]["adjudication.result"]["to"] == "result"
    assert analysis["on_event"]["adjudication.diagnosis"]["to"] == "analysis_failed"
    assert states["result"]["on_intent"]["new_round"]["to"] == "play"
    phrases = manifest["asr"]["phrases"]
    missing = []
    for state_name, state in states.items():
        for intent in (state.get("on_intent") or {}):
            if intent == "speech_done":
                continue
            if intent not in phrases:
                missing.append(f"{state_name}.{intent}")
    assert not missing, f"语音不可达的意图（asr.phrases 缺条目）: {missing}"


def test_frontend_cancels_the_round_when_the_page_is_hidden():
    """pagehide 保险（方案A）：关标签/刷新即 sendBeacon 取消对局。"""
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    dice = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    # The beacon fires on pagehide with the live round id; /cancel carries no
    # body (an unread body would poison keep-alive connections).
    assert "addEventListener('pagehide'" in app
    assert "navigator.sendBeacon?.(`/api/game/rounds/${roundId}/cancel`)" in app
    # The engine tracks the live round client; dice registers on start and
    # clears on teardown (rps has no rounds and never touches it).
    assert "setActiveRound(client) { activeRound = client || null; }" in app
    assert "setActiveRound(round);" in dice
    assert "setActiveRound(null);" in dice


def test_frontend_standby_screen_engine_level():
    """待机页：select 空闲超时进入（阈值来自全局配置），任意输入唤醒且不穿透。"""
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    css = (ROOT / "web/styles.css").read_text(encoding="utf-8")

    # The standby view sits before the game list and is engine-owned (no
    # game symbols), so every future game inherits it.
    assert 'data-view="standby"' in html
    assert html.find('data-view="standby"') < html.find('data-view="select"')
    # Idle entry from the select phase only; the threshold comes from the
    # arena config (backend/config.json standby.idle_seconds), with the
    # builtin constant as offline fallback.
    assert "IDLE_ENTER_SECONDS = 120" in app
    assert "standbySettings.idle_seconds" in app
    assert "state.phase !== 'select'" in app
    # boot_standby=true: the page loads straight into standby.
    assert "if (standbySettings.boot_standby) enterStandby();" in app
    # Wake words: the standby screen asks the server to listen, polls the
    # wake events, and stops listening on wake / round start / teardown.
    assert "startStandbyListening()" in app
    assert "'/api/asr/standby'" in app
    assert "'/api/asr/standby/events'" in app
    assert "stopStandbyListening()" in app
    assert "standbyWakeWords" in html
    # Wake-up swallows the first input at capture time: it must never reach
    # the game list (no accidental game start while "asleep").  Channels are
    # voice (server-side wake words) + physical keys only — deliberately NO
    # click, so a stray mouse tap cannot wake the screen.  keyup covers
    # devices that only report release events.
    assert "['keydown', 'keyup']" in app
    assert "event.stopPropagation();" in app
    # Visuals: brand-neutral, animated, and degradable under reduced motion.
    assert "standby-core" in html and "standby-zzz" in html
    assert "standbyRipple" in css and "standbyBreath" in css
    assert "standby" in css.split("@media (prefers-reduced-motion: reduce)", 1)[1]


def test_manifest_state_machine_declares_the_full_graph():
    manifest = dice_manifest()
    machine = manifest["state_machine"]
    assert machine["initial"] == "rules"
    names = set(machine["states"])
    assert {
        "rules", "ready", "shake_countdown", "shaking", "arm_failed",
        "manual_shaking", "open_reveal",
        "vision_countdown", "analysis", "analysis_failed", "result",
    } <= names
    # Analysis routes both provider outcomes.
    analysis = machine["states"]["analysis"]["on_event"]
    assert analysis["adjudication.result"]["to"] == "result"
    assert analysis["adjudication.diagnosis"]["to"] == "analysis_failed"
    # Retry re-enters adjudication without resetting the round.
    failed = machine["states"]["analysis_failed"]["on_intent"]
    assert failed["retry"]["to"] == "analysis"
    assert failed["new_round"]["to"] == "ready"
    # The result announcement picks its line from the adjudicated role.
    result_entry = machine["states"]["result"]["on_enter"][0]
    assert result_entry["select_by"] == "winner_role"
    assert set(result_entry["cases"]) == {"PLAYER", "AGENT", "TIE"}
    # Robot wiring: ready 刻意不再挂任何 robot 动作（2026-09-23 拍板）——
    # 归位本来多余（每轮 RETURN_HOME 已归位、终态 watcher 兜底），且归位
    # 手势会占住臂锁 1.8s+，玩家快点开始时抓取被迫排队、整个抓取动作
    # 挪到 shaking 里才执行。抓取链第一步就是 HOME，姿态恢复免费。
    ready_actions = machine["states"]["ready"]["on_enter"]
    assert all(a.get("action") != "robot" for a in ready_actions)
    feedback = machine["states"]["result"]["on_enter"][1]
    assert feedback["command"] == "feedback"
    assert feedback["select_by"] == "winner_role"
    assert feedback["cases"] == {"PLAYER": "lose", "AGENT": "win", "TIE": "draw"}


def test_frontend_ignores_stale_events_after_round_ends():
    """A terminal round must never resurrect its finished view."""
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")
    dice = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    ingest = app.split("function ingest", 1)[1].split("function subscribe", 1)[0]
    assert "if (closed) return;" in ingest
    # The terminal snapshot closes the client without syncing its stale
    # top-level state, so returnToSelect() stays the last navigation.
    assert "snapshot.status !== 'running'" in ingest
    assert "teardownStream()" in ingest
    # Intent responses are ingested too, so exit intents navigate even when
    # the SSE stream is broken; sequence dedup keeps this idempotent.
    client = app.split("async function submitIntent", 1)[1].split("async function cancel", 1)[0]
    assert "ingest(snapshot)" in client
    assert "closed = true" in app
    # dice.js must not re-render states once its round is gone.
    sync = dice.split("onSyncState:", 1)[1].split("},", 1)[0]
    assert "if (!round || !round.roundId) return;" in sync


def test_frontend_renders_countdown_top_value_with_ceil():
    """Countdown numbers step once per tick_seconds to match the voice pace."""
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    tick = js.split("function renderTick", 1)[1].split("function resetAnalysisSteps", 1)[0]
    # One number per tick_seconds (from the tick event), not per whole second.
    assert "Number(event.tick_seconds || 1) * 1000" in tick
    assert "Math.ceil(remaining / perNumber)" in tick
    # The shake timer keeps literal seconds-to-go.
    assert "Math.ceil(remaining / 1000)" in tick
    assert "Math.floor" not in tick


def test_frontend_renders_reask_and_tie_upheld_sources():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    assert "yolo_reask_confirmed" in js
    assert "大模型复问后与 YOLOv8 一致" in js
    assert "yolo_reask_fallback" in js
    assert "tie_upheld" in js
    assert "双方点数相同，判定平局" in js


def test_frontend_verifying_copy_follows_the_llm_flag():
    """步骤文案必须跟随事件里的 llm 标志，而不是无条件说"调用大模型"。

    纯 YOLO 的一局（当前 dice 就是）里，页面曾写着「正在调用大模型复核…」，
    与事实不符。
    """
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    verifying = js.split("} else if (event.phase === 'verifying')", 1)[1].split(
        "} else if (event.phase === 'holding')", 1
    )[0]
    assert "event.llm" in verifying, "必须按 llm 标志分支"
    # 关闭复核时不得出现点名大模型的措辞。
    assert "大模型" not in verifying.split("event.llm", 1)[0]


def test_static_analysis_copy_never_promises_a_disabled_llm():
    """静态文案（manifest 与前端兜底）要与 llm.enabled 一致。"""
    manifest = json.loads(
        (ROOT / "backend/games/dice/manifest.json").read_text(encoding="utf-8")
    )
    enabled = bool((manifest.get("vision_profile") or {}).get("llm", {}).get("enabled", True))
    copy = manifest["state_machine"]["states"]["analysis"]["ui"]["copy"]
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    fallback = js.split("analysis: [", 1)[1].split("]", 1)[0]
    if not enabled:
        assert "大模型" not in copy, f"复核已关闭，静态文案不能承诺大模型：{copy!r}"
        assert "大模型" not in fallback, f"前端兜底文案不能承诺大模型：{fallback!r}"
