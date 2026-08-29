# Qwen3-TTS provider

Qwen3 settings used by Dice Arena live in
`backend/components/tts_qwen3/config.json`. Keep the paths relative:

- `runtime.root` is relative to the repository root;
- `runtime.model_dir` and `voice.speaker_file` are relative to the Qwen3
  runtime/model directory.

The provider preserves the existing environment overrides. The precedence is
**environment variable > component config > code default**. The Qwen runtime
loads one speaker embedding at server startup; changing `speaker_file` requires
restarting the provider. Unlike MOSS-TTS-Nano, this adapter does not accept a
reference WAV for live voice cloning.

A custom embedding can be selected by changing `voice.speaker_file` and
running the component's internal `scripts/stop_tts.sh` followed by its internal
`scripts/start_tts.sh`. The component
launcher creates an ignored runtime model overlay, so the checked-in model
`config.json` is not modified.
