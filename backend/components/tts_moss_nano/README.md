# MOSS-TTS-Nano provider

This is a Dice Arena adapter for the board-local project:

```text
/home/spacemit/projects/moss-tts-nano-spacemit-ep-demo-1.0.7-slim-riscv64
```

The adapter deliberately does **not** copy or modify that project. It imports
its packaged `OnnxTtsRuntime` directly through a small local HTTP bridge, so
model files, bundled Python packages, and board-specific runtime changes stay
in the MOSS checkout.

## Streaming behavior

The bridge uses the current K3 delivery's callback path:

```text
text -> MOSS generates one text chunk -> decodes that chunk -> HTTP sends a WAV frame
```

The browser starts playing the first WAV frame as soon as it arrives while the
bridge continues generating later text chunks. This is **chunk-level streaming**
(the current MOSS codec does not emit browser-playable PCM for every individual
codec frame), not a claim of frame-level codec streaming.

The bridge no longer launches the interactive CLI or parses its diagnostic log.
That avoids coupling request completion to log wording and preserves the MOSS
runtime's `on_pcm_chunk` callback. The MOSS runtime is initialized and warmed
once per provider process, then requests are serialized because the packaged
SpaceMIT runtime is single-session.

## Configuration

The defaults are intended for the current K3 board:

```bash
DICE_MOSS_TTS_ROOT=/home/spacemit/projects/moss-tts-nano-spacemit-ep-demo-1.0.7-slim-riscv64
DICE_MOSS_TTS_MODEL_DIR=/home/spacemit/projects/moss-tts-nano-spacemit-ep-demo-1.0.7-slim-riscv64/models/MOSS-TTS-Nano-100M-ONNX-xslim-dynq
DICE_MOSS_TTS_HOST=127.0.0.1
DICE_MOSS_TTS_PORT=18082
DICE_MOSS_TTS_VOICE=Junhao
# Optional voice-clone reference WAV. It is loaded when the bridge starts.
DICE_MOSS_TTS_REFERENCE_AUDIO=/absolute/path/reference.wav
DICE_MOSS_TTS_MAX_NEW_FRAMES=120
DICE_MOSS_TTS_VOICE_CLONE_MAX_TEXT_TOKENS=24
DICE_MOSS_TTS_FIRST_CHUNK_TEXT_TOKENS=16
DICE_MOSS_TTS_TIMEOUT_SECONDS=120
DICE_MOSS_TTS_START_TIMEOUT_SECONDS=300
```

MOSS currently has no generic speed control, so this provider accepts only
`speed=1.0`. The configured voice is prepared when the bridge starts; changing
voice or enabling voice cloning requires restarting the provider.

## Select it

Keep Qwen3 as the default and select MOSS temporarily:

```bash
cd /home/spacemit/projects/dice-game/main
scripts/stop_web.sh
DICE_TTS_PROVIDER=tts_moss_nano scripts/start_web.sh
```

Or set the dice game manifest's `providers.tts` to `tts_moss_nano`. The web
backend must be restarted after changing provider selection.

## Component checks

```bash
/usr/bin/python3 backend/componentctl.py list
/usr/bin/python3 backend/componentctl.py selected tts --game dice
/usr/bin/python3 backend/componentctl.py health tts_moss_nano
/usr/bin/python3 backend/componentctl.py start tts_moss_nano
/usr/bin/python3 backend/componentctl.py stop tts_moss_nano
```

Changing the external MOSS project later only requires updating that project,
`DICE_MOSS_TTS_ROOT`, or `DICE_MOSS_TTS_MODEL_DIR`; the Dice Arena adapter
contract remains unchanged unless the external runtime API changes.
