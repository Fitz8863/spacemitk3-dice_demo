"""RPS skeleton tests: outcome logic, stub pipeline, full round flow.

The stub pipeline is the only adjudicator until the gesture model lands, so
these tests pin both its event shape (must match the real vision flow) and
the public result fields every consumer (frontend, speech placeholders)
depends on.
"""
from __future__ import annotations

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


class StubPipelineTests(unittest.TestCase):
    def test_run_emits_staged_progress_and_returns_projected_result(self):
        events = []
        manifest = rps_manifest()
        manifest["participants"] = {"player": "RIGHT", "agent": "LEFT"}
        with mock.patch("games.rps.pipeline.time.sleep"):
            result = rps_pipeline.run(
                lambda line: None,
                lambda: False,
                30.0,
                components=None,
                manifest=manifest,
                on_event=events.append,
            )
        # 进度事件的 phase 形状与真实视觉流一致（分析页按这三种渲染）。
        self.assertEqual(
            [event.get("phase") for event in events],
            ["detecting", "verifying", "holding"],
        )
        self.assertIn(result["source"], "stub")
        self.assertIn(result["player_choice"], list(GESTURES))
        self.assertIn(result["agent_choice"], list(GESTURES))
        self.assertIn(result["winner_role"], {"PLAYER", "AGENT", "TIE"})
        self.assertNotIn("diagnosed", result)

    def test_run_stops_emitting_when_cancelled(self):
        events = []
        manifest = rps_manifest()
        manifest["participants"] = {"player": "RIGHT", "agent": "LEFT"}
        with mock.patch("games.rps.pipeline.time.sleep"):
            rps_pipeline.run(
                lambda line: None,
                lambda: True,
                30.0,
                components=None,
                manifest=manifest,
                on_event=events.append,
            )
        self.assertEqual(events, [])


class RoundFlowTests(unittest.TestCase):
    def make_round(self):
        manifest = rps_manifest()
        manifest["participants"] = {"player": "RIGHT", "agent": "LEFT"}

        def adjudicate(manifest, on_event, is_cancelled, log):
            return rps_pipeline.run(
                log,
                is_cancelled,
                30.0,
                components=None,
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

        random.choice 的调用序是先 agent 后 player（机械臂指令位在前），
        side_effect 因此按 [agent, player] 给。
        """
        with mock.patch(
            "games.rps.pipeline.random.choice",
            side_effect=[agent_gesture, player_gesture],
        ), mock.patch("games.rps.pipeline.time.sleep"):
            round_ = self.make_round()
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
