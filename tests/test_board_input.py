"""板端键盘直读（core/board_input.py）的解析、映射与防抖单测。

字节流全部用与实现相同的 struct 合成——不需要真实设备，也不起线程；
读取线程的 select 循环由板端真机 E2E 覆盖。DispatchTests 直接驱动
server._board_dispatch（fake 回合 + monkeypatch 依赖），不起真引擎。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.board_input import (  # noqa: E402
    EV_KEY,
    EV_SYN,
    KEY_ACTIONS,
    KEY_DOWN,
    KEY_ENTER,
    KEY_ESC,
    KEY_UP,
    BoardKeyReader,
    _INPUT_EVENT,
    parse_key_events,
    resolve_intent,
)
import server  # noqa: E402

MOUSE_LEFT = 0x110


def event(type_: int, code: int, value: int) -> bytes:
    return _INPUT_EVENT.pack(0, 0, type_, code, value)


class ParseTests(unittest.TestCase):
    def test_press_release_and_repeat_are_all_returned(self):
        stream = (
            event(EV_SYN, 0, 0)
            + event(EV_KEY, KEY_ENTER, 1)
            + event(EV_KEY, KEY_ENTER, 2)
            + event(EV_KEY, KEY_ENTER, 0)
        )
        self.assertEqual(
            parse_key_events(stream),
            [(KEY_ENTER, 1), (KEY_ENTER, 2), (KEY_ENTER, 0)],
        )

    def test_partial_trailing_bytes_are_ignored(self):
        stream = event(EV_KEY, KEY_ESC, 1) + b"\x01\x02"
        self.assertEqual(parse_key_events(stream), [(KEY_ESC, 1)])

    def test_empty_stream(self):
        self.assertEqual(parse_key_events(b""), [])


class MappingTests(unittest.TestCase):
    def test_key_actions_cover_the_four_convention_keys(self):
        self.assertEqual(
            KEY_ACTIONS,
            {KEY_ENTER: "confirm", KEY_ESC: "back", KEY_UP: "up", KEY_DOWN: "down"},
        )
        self.assertNotIn(MOUSE_LEFT, KEY_ACTIONS)  # 鼠标键天然不在映射里

    def test_intent_table_matches_frontend_onkey(self):
        """与 dice.js / rps.js 的 onKey 逐字对齐的锚点断言（改前端要同步）。"""
        self.assertEqual(
            resolve_intent("dice", "rules", "confirm"), "confirm"
        )
        self.assertEqual(resolve_intent("dice", "rules", "down"), "repeat")
        self.assertEqual(resolve_intent("dice", "ready", "confirm"), "start_shake")
        self.assertEqual(resolve_intent("dice", "result", "confirm"), "new_round")
        self.assertEqual(resolve_intent("dice", "arm_failed", "down"), "retry")
        self.assertEqual(resolve_intent("dice", "analysis_failed", "confirm"), "new_round")
        self.assertEqual(resolve_intent("rps", "rules", "down"), "repeat")
        self.assertEqual(resolve_intent("rps", "result", "back"), "back")
        # 摇骰中刻意无键：机械臂摇物理上不应被打断（2026-09-24 起人工摇骰
        # 模式移除，Esc 不再停止摇骰）。
        self.assertIsNone(resolve_intent("dice", "shaking", "back"))
        self.assertIsNone(resolve_intent("dice", "stop_call", "back"))
        # 纯判定中（非失败页）与过场状态无键。
        self.assertIsNone(resolve_intent("dice", "analysis", "confirm"))
        self.assertIsNone(resolve_intent("rps", "play", "confirm"))
        self.assertIsNone(resolve_intent("rps", "preparing", "back"))
        # 未知游戏不炸。
        self.assertIsNone(resolve_intent("unknown", "rules", "confirm"))


class ReaderDispatchTests(unittest.TestCase):
    def _reader(self, actions: list[str], debounce: float = 0.03):
        return BoardKeyReader(
            actions.append, paths=[], debounce_seconds=debounce, log=lambda *_: None
        )

    def test_press_edge_dispatches_release_and_repeat_do_not(self):
        actions: list[str] = []
        reader = self._reader(actions)
        reader._handle_buffer(
            event(EV_KEY, KEY_ENTER, 1)
            + event(EV_KEY, KEY_ENTER, 2)
            + event(EV_KEY, KEY_ENTER, 0),
            now=1.0,
        )
        self.assertEqual(actions, ["confirm"])

    def test_unmapped_mouse_button_is_ignored(self):
        actions: list[str] = []
        reader = self._reader(actions)
        reader._handle_buffer(event(EV_KEY, MOUSE_LEFT, 1), now=1.0)
        self.assertEqual(actions, [])

    def test_debounce_drops_same_key_inside_window(self):
        actions: list[str] = []
        reader = self._reader(actions, debounce=0.03)
        reader._handle_buffer(event(EV_KEY, KEY_UP, 1), now=1.0)
        reader._handle_buffer(event(EV_KEY, KEY_UP, 1), now=1.01)  # 10ms < 30ms
        self.assertEqual(actions, ["up"])
        reader._handle_buffer(event(EV_KEY, KEY_UP, 1), now=1.05)  # 50ms 后放行
        self.assertEqual(actions, ["up", "up"])

    def test_status_reports_last_action_and_devices(self):
        reader = self._reader([])
        reader._handle_buffer(event(EV_KEY, KEY_ESC, 1), now=2.0)
        status = reader.status()
        self.assertEqual(status["last_action"], "back")
        self.assertEqual(status["devices"], [])
        self.assertFalse(status["permission_denied"])

    def test_callback_exception_does_not_break_dispatch(self):
        def boom(action: str) -> None:
            raise RuntimeError("boom")

        reader = BoardKeyReader(boom, paths=[], log=lambda *_: None)
        reader._handle_buffer(event(EV_KEY, KEY_ENTER, 1), now=1.0)
        self.assertEqual(reader.last_action, "confirm")  # 状态仍更新，线程不死


class _FakeRound:
    def __init__(self, game_id: str, state: str, round_id: str = "r-test"):
        self.game_id = game_id
        self.state = state
        self.id = round_id
        self.status = "running"
        self.submitted: list[tuple[str, dict]] = []

    def submit_intent(self, name: str, payload: dict | None = None):
        self.submitted.append((name, payload or {}))
        return {"id": self.id}


class DispatchTests(unittest.TestCase):
    """server._board_dispatch 的分相位派发与去重（fake 依赖，不起真引擎）。"""

    def setUp(self):
        self.round_ = _FakeRound("dice", "rules")
        # 每个测试用干净的注册表/光标/去重窗。
        self._patches = [
            mock.patch.object(server, "rounds", {"r-test": self.round_}),
            mock.patch.object(server, "rounds_lock", __import__("threading").Lock()),
            mock.patch.object(server, "get_games", lambda: [
                {"id": "dice", "enabled": True}, {"id": "rps", "enabled": True},
            ]),
            mock.patch.object(server, "get_arena_config",
                              lambda: {"input_board_enable": True}),
            mock.patch.object(server, "_board_cursor", {"game_id": None}),
            mock.patch.object(server, "_board_dedup", {}),
        ]
        for patch in self._patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_in_round_intent_uses_the_table(self):
        server._board_dispatch("confirm")
        self.assertEqual(self.round_.submitted, [("confirm", {"source": "board"})])

    def test_shaking_has_no_keys(self):
        self.round_.state = "shaking"
        server._board_dispatch("back")
        self.assertEqual(self.round_.submitted, [])

    def test_disabled_config_swallows_everything(self):
        with mock.patch.object(server, "get_arena_config",
                               lambda: {"input_board_enable": False}):
            server._board_dispatch("confirm")
        self.assertEqual(self.round_.submitted, [])

    def test_duplicate_intent_inside_window_is_dropped(self):
        clock = iter([100.0, 100.05])
        with mock.patch.object(server.time, "monotonic", lambda: next(clock)):
            server._board_dispatch("confirm")
            server._board_dispatch("confirm")  # 50ms < 250ms：吞掉
        self.assertEqual(len(self.round_.submitted), 1)

    def test_no_round_navigate_moves_cursor_cyclically(self):
        with mock.patch.object(server, "rounds", {}), \
                mock.patch.object(server, "_SELECT_BUS") as select_bus:
            server._board_dispatch("down")   # dice -> rps
            server._board_dispatch("down")   # rps -> dice（环形）
        pushes = [c.args[0] for c in select_bus.push.call_args_list]
        self.assertEqual([p["game_id"] for p in pushes], ["rps", "dice"])
        self.assertTrue(all(p["status"] == "board_navigate" for p in pushes))

    def test_no_round_confirm_wakes_and_selects(self):
        with mock.patch.object(server, "rounds", {}), \
                mock.patch.object(server, "_STANDBY_BUS") as standby_bus, \
                mock.patch.object(server, "_SELECT_BUS") as select_bus:
            server._board_dispatch("down")   # 光标 -> rps
            server._board_dispatch("confirm")
        standby_bus.push.assert_called_once_with(
            {"event": "asr", "status": "wake", "text": "board"})
        selected = select_bus.push.call_args_list[-1].args[0]
        self.assertEqual(selected["status"], "selected")
        self.assertEqual(selected["game_id"], "rps")


if __name__ == "__main__":
    unittest.main()
