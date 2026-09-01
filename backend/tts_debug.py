#!/usr/bin/env python3
"""Interactive developer console for any packaged TTS provider.

The console deliberately uses the same provider contract as the web backend:
one input line becomes one ``TtsProvider.stream`` call, and each returned WAV
frame is sent to the selected local audio player through stdin.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.components import build_registry  # noqa: E402
from core.games import load_games  # noqa: E402
from core.tts_dispatch import TtsDispatcher  # noqa: E402


PLAYER_CANDIDATES = ("aplay", "paplay", "ffplay", "mpv")


def is_quit_command(text: str) -> bool:
    return text.strip().lower() in {"/quit", "/exit"}


class AudioPlayer:
    """Play one complete WAV frame using a command that accepts stdin."""

    def __init__(self, executable: str) -> None:
        self.executable = executable

    def command(self) -> list[str]:
        name = Path(self.executable).name.lower()
        if name == "aplay":
            return [self.executable, "-q", "-"]
        if name == "paplay":
            return [self.executable]
        if name == "ffplay":
            return [
                self.executable,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                "-",
            ]
        if name == "mpv":
            return [self.executable, "--no-video", "--really-quiet", "-"]
        return [self.executable]

    def play(self, audio: bytes) -> None:
        subprocess.run(self.command(), input=audio, check=True)


def resolve_player(requested: str | None) -> AudioPlayer:
    value = requested or os.environ.get("DICE_TTS_PLAYER", "").strip()
    if value and value.lower() not in {"auto", "default"}:
        executable = shutil.which(value) or (value if Path(value).is_file() else None)
        if executable is None:
            raise RuntimeError(f"audio player not found: {value}")
        return AudioPlayer(executable)

    for candidate in PLAYER_CANDIDATES:
        executable = shutil.which(candidate)
        if executable:
            return AudioPlayer(executable)
    candidates = ", ".join(PLAYER_CANDIDATES)
    raise RuntimeError(f"no audio player found; install one of: {candidates}")


def _health_ok(provider: Any) -> bool:
    try:
        health = provider.health()
    except Exception:
        return False
    return isinstance(health, dict) and bool(health.get("ok"))


def _run_lifecycle(provider_id: str, action: str) -> None:
    command = [
        sys.executable,
        str(BACKEND_ROOT / "componentctl.py"),
        action,
        provider_id,
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"provider {provider_id} {action} failed with exit code {completed.returncode}")


def _ensure_ready(provider: Any, provider_id: str) -> bool:
    """Start an unavailable provider and return whether this process owns it."""
    if _health_ok(provider):
        return False
    _run_lifecycle(provider_id, "start")
    if not _health_ok(provider):
        raise RuntimeError(f"provider {provider_id} did not become ready")
    return True


def _stop_if_owned(provider_id: str, owned: bool) -> None:
    if owned:
        try:
            _run_lifecycle(provider_id, "stop")
        except RuntimeError as exc:
            print(f"Warning: {exc}", file=sys.stderr)


def interactive(provider_id: str | None, game_id: str, player: AudioPlayer) -> int:
    components = build_registry(log=False)
    games = load_games()
    dispatcher = TtsDispatcher(components, games)
    selected_id = provider_id or dispatcher.provider_id(game_id)
    provider = components.require(selected_id, expected_type="tts")
    owned = _ensure_ready(provider, selected_id)
    try:
        print(f"TTS ready: {selected_id}")
        print("输入文字后回车播放，输入 /quit 或 /exit 退出。")
        while True:
            try:
                text = input("> ")
            except EOFError:
                print()
                break
            if is_quit_command(text):
                break
            if not text.strip():
                continue
            print("生成并播放中...", flush=True)
            provider.stream({"game": game_id, "text": text}, player.play)
    finally:
        _stop_if_owned(selected_id, owned)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive console for packaged TTS providers")
    parser.add_argument("provider", nargs="?", help="provider id; default comes from the game manifest")
    parser.add_argument("--game", default="dice", help="game manifest id used for provider selection")
    parser.add_argument("--player", help="audio player executable; default auto-detects ALSA/PulseAudio players")
    args = parser.parse_args(argv)
    try:
        audio_player = resolve_player(args.player)
        return interactive(args.provider, args.game, audio_player)
    except KeyboardInterrupt:
        print("\n已退出。")
        return 130
    except Exception as exc:
        print(f"TTS debug failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
