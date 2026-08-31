import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_has_no_hardcoded_mediamtx_host():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "100.118.229.28:8889" not in html
    assert "data-stream-url" not in html
    assert "event.event === 'video'" in js


def test_frontend_keeps_video_until_terminal_complete():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "phase === 'holding'" in js
    assert "status === 'success'" in js
    assert "phase === 'complete'" in js
    assert "startVisionStream(event)" in js


def test_frontend_preserves_holding_countdown_from_structured_event():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    apply_snapshot = js.split("function applyAnalysisSnapshot", 1)[1].split(
        "function streamAnalysis", 1
    )[0]

    assert "latestHoldingEvent" in js
    assert "updateAnalysisProgress(latestHoldingEvent || snapshot)" in js
    assert "updateAnalysisProgress(snapshot);" not in apply_snapshot


def test_frontend_distinguishes_llm_override_from_consensus():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    assert "llm_override" in js


def test_frontend_renders_structured_diagnosis_and_retry_prompt():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    assert "showDiagnosis" in js
    assert "retry_required" in js
    assert "本次裁决未完成" in js
    assert "diagnosis.message" in js


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


def test_frontend_failure_state_offers_new_round_or_game_list():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert 'id="analysisNewRound"' in html
    assert 'id="analysisBackToGames"' in html
    assert "再来一局" in html
    assert "退出游戏，返回列表" in html
    assert 'id="analysisRetry"' not in html
    assert "analysisNewRound" in js
    assert "analysisBackToGames" in js
    assert "analysisNewRound: () => resetRound()" in js
    assert "analysisBackToGames: () => returnToSelect()" in js
    assert "analysisRetry" not in js
    assert "reveal()" not in js.split("function showDiagnosis", 1)[1].split("function showResult", 1)[0]


def test_frontend_does_not_mark_yolo_complete_while_still_detecting():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    detecting = js.split("if (job.phase === 'detecting')", 1)[1].split(
        "} else if (job.phase === 'verifying')", 1
    )[0]
    assert "querySelector('span').textContent = '…'" in detecting
    assert "querySelector('span').textContent = '✓'" not in detecting
    assert "以大模型结果为准" in js


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
    manifest = json.loads(
        (ROOT / "backend/games/dice/manifest.json").read_text(encoding="utf-8")
    )
    stop_shake = js.split("function stopShake", 1)[1].split(
        "function beginRevealCountdown", 1
    )[0]

    assert "setPhase('open')" in stop_shake
    assert "countdown(" not in stop_shake
    assert "speakState('reveal_ready')" in stop_shake
    assert "revealTransitionTimer = setTimeout" in stop_shake
    assert "beginRevealCountdown();" in stop_shake
    assert "}, 2000);" in stop_shake
    assert "clearTimeout(revealTransitionTimer)" in js
    assert "开盖过场中" in js
    assert 'id="revealDice"' not in html
    assert "shake_stopped" not in manifest["texts"]
    assert manifest["texts"]["reveal_ready"] == {
        "mode": "tts",
        "text": "准备好了没有？来,3,2,1,开盖！",
    }


def test_frontend_counts_down_after_open_transition_before_adjudication():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "function beginRevealCountdown" in js
    confirm_open = js.split("function beginRevealCountdown", 1)[1].split(
        "function resetAnalysisSteps", 1
    )[0]
    assert re.search(r"countdown\(\s*reveal,", confirm_open)
    assert "VISION COUNTDOWN" in confirm_open
    assert "请保持骰子和骰盅位置不动" in confirm_open
    assert "revealDice" not in js


def test_frontend_uses_vision_specific_copy_during_post_open_countdown():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")

    assert "visionCountdownMeta" in js
    assert "倒计时结束后开始视觉裁决" in js
    confirm_open = js.split("function beginRevealCountdown", 1)[1].split(
        "function resetAnalysisSteps", 1
    )[0]
    assert "visionCountdownMeta" in confirm_open


def test_frontend_uses_ten_second_shake_and_urgent_last_three_seconds():
    js = (ROOT / "web/games/dice.js").read_text(encoding="utf-8")
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")

    assert "const SHAKE_DURATION_SECONDS = 10;" in js
    assert 'id="shakeSeconds">10<' in html
    assert "const urgent = seconds <= 3 && seconds > 0;" in js
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
    begin_shake = js.split("function beginShake", 1)[1].split(
        "function stopShake", 1
    )[0]
    start_handler = js.split("startShake: () => {", 1)[1].split(
        "stopShake: () => stopShake()", 1
    )[0]

    assert "speakState('shake_started')" not in begin_shake
    assert "countdown(beginShake, 'GET READY', '和 Agent 同步');" in start_handler
    assert "speakState('shake_started');" in start_handler
    assert start_handler.index("speakState('shake_started')") < start_handler.index(
        "countdown(beginShake"
    )
