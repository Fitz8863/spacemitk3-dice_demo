import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def dice_manifest():
    return json.loads(
        (ROOT / "backend/games/dice/manifest.json").read_text(encoding="utf-8")
    )


def dice_state(name):
    return dice_manifest()["state_machine"]["states"][name]


def test_frontend_has_no_hardcoded_mediamtx_host():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "100.118.229.28:8889" not in html
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
    assert class_of("stopShake") == "btn-circle btn-red"
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
    # The reveal line follows in the same on_enter sequence, and the hold
    # window starts only after the awaited clip finishes.
    assert open_reveal["on_enter"][1]["text"].startswith("准备好了没有")
    assert open_reveal["duration"] == 4

    # The frontend must not re-implement the chain: awaiting is an engine
    # concern, playback acknowledgement happens through the round client.
    assert "function stopShake" not in js
    assert "speakState" not in js
    assert "startRevealTransition" not in js
    assert "submitIntent('speech_done'" in app
    assert "directive.await" in app


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
    assert "state.phase === 'analysis' && event.key === 'ArrowDown'" in js
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
    assert reveal_entry["mode"] == "tts_local"
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


def test_frontend_uses_vision_specific_copy_during_post_open_countdown():
    copy = dice_state("vision_countdown")["ui"]["copy"]
    assert "倒计时结束后开始视觉裁决" in copy


def test_frontend_uses_ten_second_shake_and_urgent_last_three_seconds():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")

    # The ten-second budget is owned by the backend state machine.
    assert dice_state("shaking")["duration"] == 10
    assert 'id="shakeSeconds">10<' in html
    assert "const urgent = seconds <= 3;" in js
    assert "shakeSeconds.classList.toggle('is-urgent', urgent)" in js
    assert "if (urgent) playCountdownCue(seconds);" in js


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
    """await 指令播完回执；静音模式立即回执，避免状态机等待兜底。"""
    app = (ROOT / "web/app.js").read_text(encoding="utf-8")

    play = app.split("async function playDirective", 1)[1].split(
        "// ---- 权威对局客户端", 1
    )[0]
    assert "directive.await" in play
    assert "submitIntent('speech_done'" in play
    # Muted playback must still acknowledge awaited directives.
    muted = play.split("if (!state.sound)", 1)[1].split("}", 1)[0]
    assert "acknowledge()" in muted


def test_manifest_state_machine_declares_the_full_graph():
    manifest = dice_manifest()
    machine = manifest["state_machine"]
    assert machine["initial"] == "rules"
    names = set(machine["states"])
    assert {
        "rules", "ready", "shake_countdown", "shaking", "open_reveal",
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
