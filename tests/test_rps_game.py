"""RPS tests: outcome logic, vision pipeline, full round flow.

The player gesture comes from the vision provider's observe() (the v10
runtime folds classes in C++ and emits game labels), the agent gesture stays
a program stub — the robot-arm hook.  These tests drive the pipeline with a
fake provider and pin the event shape the analysis page renders plus the
public result fields every consumer (frontend, speech placeholders) needs.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.state_machine import GameRound  # noqa: E402
from games.rps import pipeline as rps_pipeline  # noqa: E402
from games.rps.result import BEATS, GESTURES, decide, project  # noqa: E402


def rps_manifest():
    return json.loads(
        (ROOT / "backend/games/rps/manifest.json").read_text(encoding="utf-8")
    )


def wait_for(predicate, timeout=20.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class ResultLogicTests(unittest.TestCase):
    def test_decide_covers_all_outcomes(self):
        self.assertEqual(decide("石头", "石头"), "TIE")
        for gesture in GESTURES:
            self.assertEqual(decide(gesture, gesture), "TIE")
            self.assertEqual(decide(gesture, BEATS[gesture]), "PLAYER")
            self.assertEqual(decide(BEATS[gesture], gesture), "AGENT")

    def test_project_yields_public_result_fields(self):
        participants = {"player": "RIGHT", "agent": "LEFT"}
        result = project("石头", "剪刀", participants, source="stub")
        # 物理侧：玩家在 RIGHT，石头赢剪刀 -> RIGHT 胜。
        self.assertEqual(result["winner"], "RIGHT")
        self.assertEqual(result["winner_role"], "PLAYER")
        self.assertEqual(result["player_side"], "RIGHT")
        self.assertEqual(result["agent_side"], "LEFT")
        self.assertEqual(result["player_choice"], "石头")
        self.assertEqual(result["agent_choice"], "剪刀")
        self.assertEqual(result["left_choice"], "剪刀")
        self.assertEqual(result["right_choice"], "石头")
        self.assertEqual(result["source"], "stub")
        self.assertNotIn("diagnosed", result)

    def test_project_reports_tie(self):
        participants = {"player": "LEFT", "agent": "RIGHT"}
        result = project("布", "布", participants, source="stub")
        self.assertEqual(result["winner"], "TIE")
        self.assertEqual(result["winner_role"], "TIE")
        self.assertEqual(result["player_choice"], "布")
        self.assertEqual(result["agent_choice"], "布")


def _observation(label: str, confidence: float = 0.9) -> dict:
    return {
        "event": "observation",
        "view_id": "default",
        "stable": True,
        "width": 1080,
        "height": 1920,
        "detections": [
            {"class_id": 1, "label": label, "confidence": confidence,
             "bbox": [10, 20, 30, 40]}
        ],
    }


class _FakeV10Provider:
    """Test double for the v10 adjudicator: observe() with scripted evidence."""

    def __init__(self, observation):
        self.observation = observation
        self.observe_calls = 0

    def observe(self, request, *, on_log, on_event, is_cancelled, timeout_seconds=None):
        self.observe_calls += 1
        on_event({"event": "phase", "phase": "detecting"})
        if isinstance(self.observation, dict) and self.observation.get("diagnosed"):
            return self.observation
        return {"observations": {"default": self.observation}}


class _FakeComponents:
    def __init__(self, provider):
        self.provider = provider
        self.requested = []

    def require(self, provider_id, expected_type=None, expected_role=None):
        self.requested.append((provider_id, expected_type, expected_role))
        if provider_id != "vision_yolov10_objdetect":
            raise RuntimeError(f"unexpected provider id {provider_id!r}")
        return self.provider


class VisionPipelineTests(unittest.TestCase):
    def _run(self, provider, is_cancelled=lambda: False):
        events = []
        manifest = rps_manifest()
        manifest["participants"] = {"player": "RIGHT", "agent": "LEFT"}
        components = _FakeComponents(provider)
        # monotonic 序列让 hold 循环恰好发一次 holding 就结束（sleep 被
        # mock 后真实单调钟不会推进，会无限狂发事件）；尾值无限重复，
        # 前置的 is_cancelled 检查多耗掉一个也不受影响。
        clock = itertools.chain([0, 1, 10], itertools.repeat(10))
        with mock.patch("games.rps.pipeline.time.sleep"), mock.patch(
            "games.rps.pipeline.time.monotonic", side_effect=lambda: next(clock)
        ):
            result = rps_pipeline.run(
                lambda line: None,
                is_cancelled,
                30.0,
                components=components,
                manifest=manifest,
                on_event=events.append,
            )
        return result, events, components

    def test_run_maps_folded_label_to_gesture_and_projects_result(self):
        # 玩家手势来自观测的折叠 label（Rock → 石头），agent 手势是程序随机。
        provider = _FakeV10Provider(_observation("Rock"))
        with mock.patch("games.rps.pipeline.random.choice", return_value="剪刀"):
            result, events, components = self._run(provider)
        self.assertEqual(components.requested[0][0], "vision_yolov10_objdetect")
        self.assertEqual(components.requested[0][1], "vision")
        self.assertEqual(result["source"], "yolo_only")
        self.assertEqual(result["player_choice"], "石头")
        self.assertEqual(result["agent_choice"], "剪刀")
        self.assertEqual(result["winner_role"], "PLAYER")
        self.assertNotIn("diagnosed", result)
        # 事件形状：detecting 来自 provider（observe 内部），verifying/
        # result/holding 由 pipeline 编排——与前端分析页渲染的 phase 集一致。
        phases = [e.get("phase") for e in events if e.get("event") == "phase"]
        self.assertEqual(phases, ["detecting", "verifying", "holding"])
        self.assertTrue(any(e.get("event") == "result" for e in events))

    def test_run_highest_confidence_detection_wins(self):
        observation = _observation("Paper", 0.3)
        observation["detections"].append(
            {"class_id": 2, "label": "Scissors", "confidence": 0.8,
             "bbox": [50, 60, 80, 90]}
        )
        provider = _FakeV10Provider(observation)
        with mock.patch("games.rps.pipeline.random.choice", return_value="石头"):
            result, _, _ = self._run(provider)
        self.assertEqual(result["player_choice"], "剪刀")
        self.assertEqual(result["winner_role"], "AGENT")

    def test_run_returns_provider_diagnosis_unchanged(self):
        diagnosis = {
            "diagnosed": True,
            "retry_required": True,
            "diagnosis": {"reason_code": "NO_OBJECTS_DETECTED", "message": "画面中没有手"},
        }
        provider = _FakeV10Provider(diagnosis)
        result, events, _ = self._run(provider)
        self.assertEqual(result, diagnosis)

    def test_run_rejects_labels_outside_the_fold_table(self):
        provider = _FakeV10Provider(_observation("call"))
        with self.assertRaises(RuntimeError):
            self._run(provider)

    def test_run_requires_observe_on_the_provider(self):
        class _NoObserve:
            pass

        manifest = rps_manifest()
        with self.assertRaises(RuntimeError):
            rps_pipeline.run(
                lambda line: None, lambda: False, 30.0,
                components=_FakeComponents(_NoObserve()),
                manifest=manifest,
                on_event=lambda e: None,
            )


class RoundFlowTests(unittest.TestCase):
    def make_round(self, player_label="Rock"):
        manifest = rps_manifest()
        manifest["participants"] = {"player": "RIGHT", "agent": "LEFT"}
        provider = _FakeV10Provider(_observation(player_label))
        components = _FakeComponents(provider)

        def adjudicate(manifest, on_event, is_cancelled, log):
            return rps_pipeline.run(
                log,
                is_cancelled,
                30.0,
                components=components,
                manifest=manifest,
                on_event=on_event,
            )

        round_ = GameRound(
            game_id="rps",
            manifest=manifest,
            adjudicate_fn=adjudicate,
            log=lambda line: None,
        )
        round_.start()
        return round_

    def drive_to_result(self, agent_gesture, player_gesture):
        """Drive one round with scripted gestures: rules → play → analysis → result.

        玩家手势来自注入的观测 label（fake provider），agent 手势是
        random.choice（机械臂指令位）。
        """
        label = {"石头": "Rock", "剪刀": "Scissors", "布": "Paper"}[player_gesture]
        with mock.patch(
            "games.rps.pipeline.random.choice",
            return_value=agent_gesture,
        ), mock.patch("games.rps.pipeline.time.sleep"):
            round_ = self.make_round(label)
            self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "rules"))
            # confirm 无 after_speech 闸：不等规则宣读的 speech_done 直接按绿键。
            round_.submit_intent("confirm")
            self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "play"))
            # play 的口令是 await 台词：无头驱动必须自己回执，否则吃 30s 兜底。
            self.assertTrue(
                wait_for(
                    lambda: [
                        e for e in round_.snapshot()["events"]
                        if e.get("event") == "speech" and e.get("await")
                    ]
                )
            )
            # 覆盖 worker 在「事件已发、_awaiting_directive 未置位」之间的窗口。
            time.sleep(0.2)
            chant = next(
                e for e in round_.snapshot()["events"]
                if e.get("event") == "speech" and e.get("await")
            )
            round_.submit_intent("speech_done", {"directive_id": chant["directive_id"]})
            self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "result"))
            return round_

    def tearDown(self):
        # 各用例的 round_ 由驱动辅助函数持有；统一在事件流里找残留并取消，
        # daemon 线程不阻塞进程退出，但显式收尾更干净。
        pass

    def result_speeches(self, round_, keyword):
        """Wait for the result-state announcement (worker emits it after
        state_changed, so an immediate read races the on_enter thread)."""
        self.assertTrue(
            wait_for(
                lambda: [
                    e for e in round_.snapshot()["events"]
                    if e.get("event") == "speech" and keyword in e.get("text", "")
                ]
            )
        )
        return [
            e for e in round_.snapshot()["events"]
            if e.get("event") == "speech" and keyword in e.get("text", "")
        ]

    def test_flow_player_win(self):
        round_ = self.drive_to_result("剪刀", "石头")
        try:
            snapshot = round_.snapshot()
            self.assertEqual(snapshot["result"]["winner_role"], "PLAYER")
            self.assertEqual(snapshot["result"]["player_choice"], "石头")
            self.assertEqual(snapshot["result"]["agent_choice"], "剪刀")
            speeches = self.result_speeches(round_, "恭喜你赢了")
            self.assertIn("石头", speeches[0]["text"])
            self.assertIn("剪刀", speeches[0]["text"])
        finally:
            round_.cancel()

    def test_flow_agent_win(self):
        round_ = self.drive_to_result("石头", "剪刀")
        try:
            snapshot = round_.snapshot()
            self.assertEqual(snapshot["result"]["winner_role"], "AGENT")
            self.result_speeches(round_, "很遗憾你输了")
        finally:
            round_.cancel()

    def test_flow_tie(self):
        round_ = self.drive_to_result("布", "布")
        try:
            snapshot = round_.snapshot()
            self.assertEqual(snapshot["result"]["winner_role"], "TIE")
            self.assertEqual(snapshot["result"]["player_choice"], "布")
            speeches = self.result_speeches(round_, "平局")
            self.assertIn("布", speeches[0]["text"])
        finally:
            round_.cancel()

    def test_new_round_returns_to_play_and_skips_rules(self):
        round_ = self.drive_to_result("布", "布")
        try:
            round_.submit_intent("new_round")
            # 再来一局直接回出拳口令，不重读规则。
            self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "play"))
        finally:
            round_.cancel()


if __name__ == "__main__":
    unittest.main()
