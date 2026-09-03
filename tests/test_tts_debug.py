from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from tts_debug import AudioPlayer, is_quit_command  # noqa: E402


class TtsDebugTests(unittest.TestCase):
    def test_quit_commands_are_recognized_without_matching_text(self):
        self.assertTrue(is_quit_command("/quit"))
        self.assertTrue(is_quit_command(" /exit "))
        self.assertFalse(is_quit_command("quit"))
        self.assertFalse(is_quit_command("请退出游戏"))

    def test_audio_player_builds_stdin_command_for_aplay(self):
        player = AudioPlayer("aplay")
        self.assertEqual(player.command(), ["aplay", "-q", "-"])

    def test_audio_player_builds_stdin_command_for_ffplay(self):
        player = AudioPlayer("ffplay")
        self.assertEqual(
            player.command(),
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", "-"],
        )

    def test_each_local_tts_package_exposes_the_shared_debug_console(self):
        expected = {
            "tts_qwen3": 'backend/tts_debug.py" tts_qwen3',
            "tts_moss_nano": 'backend/tts_debug.py" tts_moss_nano',
        }
        for provider_id, invocation in expected.items():
            script = (
                ROOT
                / "backend"
                / "components"
                / provider_id
                / "scripts"
                / "debug_tts.sh"
            )
            self.assertTrue(script.is_file(), script)
            self.assertIn(invocation, script.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
