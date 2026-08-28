# MOSS-TTS-Nano provider

This is a Dice Arena adapter for the migrated MOSS-TTS-Nano delivery in:

```text
tts/moss-tts-nano/
```

The complete runtime source is kept alongside `tts/qwen3-tts`. The adapter
imports the packaged `OnnxTtsRuntime` directly through a small local HTTP
bridge. Large board artifacts (models, bundled Python packages, native
libraries, reference audio, and generated output) stay ignored in that
directory, so a checkout can either provision them there or override the
location with `DICE_MOSS_TTS_ROOT` during development.

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

The defaults are derived from the repository root, so the checkout can be moved without changing code:

```bash
# Paths are resolved from the repository root by default.
DICE_MOSS_TTS_ROOT=tts/moss-tts-nano
DICE_MOSS_TTS_MODEL_DIR=tts/moss-tts-nano/models/MOSS-TTS-Nano-100M-ONNX-xslim-dynq
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
cd <repo-root>
scripts/stop_web.sh
DICE_TTS_PROVIDER=tts_moss_nano scripts/start_web.sh
```

Or set the dice game manifest's `providers.tts` to `tts_moss_nano`. The web
backend must be restarted after changing provider selection.

## Component checks

```bash
python3 backend/componentctl.py list
python3 backend/componentctl.py selected tts --game dice
python3 backend/componentctl.py health tts_moss_nano
python3 backend/componentctl.py start tts_moss_nano
python3 backend/componentctl.py stop tts_moss_nano
```

The migrated MOSS source can be updated in `tts/moss-tts-nano` without
changing Dice Arena core scheduling. If a separate delivery is used, set
`DICE_MOSS_TTS_ROOT` and optionally `DICE_MOSS_TTS_MODEL_DIR`; the adapter
contract remains unchanged unless the runtime API changes.

## Component-local configuration

This adapter reads `backend/components/tts_moss_nano/config.json` on startup.
The file is deliberately kept next to the adapter rather than mixed into the
runtime delivery. Paths in the checked-in config are relative:

- `runtime.root` is relative to the repository root;
- `runtime.model_dir` and `voice.reference_audio` are relative to the MOSS
  runtime root;
- `voice.name` selects the built-in voice, while `voice.reference_audio`
  enables voice cloning from a WAV reference.

The precedence is **environment variable > component config > code default**.
This means existing `DICE_MOSS_TTS_*` deployments continue to work, while
changing a voice or reference audio normally only requires editing this
component's config and restarting the TTS provider.

Example voice-clone switch:

```json
{
  "voice": {
    "mode": "clone",
    "name": "Junhao",
    "reference_audio": "voice/reference.wav"
  }
}
```

`mode: "builtin"` keeps the built-in voice path and ignores the reference
file; `mode: "clone"` loads the reference WAV during startup and reuses its
prompt audio codes for requests. Restart the component after changing this
section.
