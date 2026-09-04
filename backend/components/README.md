# Provider packages

`backend/components/` is the shared provider registry for all games. A package
owns its adapter and deployment details; a game owns the semantic parameters
that describe how that adapter is used. Adding a package does not require
editing `backend/server.py` or a game pipeline.

Each runtime adapter is one directory:

```text
backend/components/<provider_id>/
├── manifest.json
├── config.json        # component-local runtime settings (required for TTS)
├── provider.py
└── scripts/            # optional; local runtime lifecycle only
```

The backend scans these packages at process startup. Adding or deleting a
package therefore requires a backend restart, but does not require editing
`backend/server.py`, `web/app.js`, or a game pipeline. A local package may
declare `lifecycle.start/stop`; a cloud or externally managed package omits
those hooks and is still a valid provider.

Minimum TTS manifest:

```json
{
  "id": "tts_new",
  "type": "tts",
  "name": "New TTS",
  "version": "1.0",
  "enabled": true,
  "entry": "provider.py:TtsNew",
  "config": "config.json"
}
```

TTS providers should keep their model-specific settings in a component-local
`config.json` and parse them through a package-local `settings.py` (voice,
model/runtime path, endpoint, generation and startup tuning).
Use repository-relative paths in checked-in defaults; environment variables may
override them for board-local deployments.

A provider that owns a local model process may also declare lifecycle commands:

```json
{
  "lifecycle": {
    "start": ["scripts/start_new_tts.sh"],
    "stop": ["scripts/stop_new_tts.sh"]
  }
}
```

Commands run from the repository root. Inspect/manage them with:

```bash
python3 backend/componentctl.py list
python3 backend/componentctl.py health tts_new
python3 backend/componentctl.py start tts_new
python3 backend/componentctl.py stop tts_new
python3 backend/componentctl.py start-selected tts --game dice
python3 backend/componentctl.py selected vision_adjudicator --game dice
```

### Package templates

Cloud/external TTS (no process lifecycle):

```text
backend/components/tts_cloud_xxx/
├── manifest.json
├── config.json          # runtime.kind=cloud, absolute runtime.base_url
├── settings.py
└── provider.py
```

Local TTS (optional process lifecycle):

```text
backend/components/tts_local_xxx/
├── manifest.json
├── config.json          # runtime.kind=local
├── settings.py
├── provider.py
└── scripts/start_tts.sh # optional provider-internal lifecycle hook
```

## TTS interface

Inherit `core.tts.TtsProvider`. The smallest adapter implements `health()` and
`synthesize(payload)`. The base class automatically exposes it through the
browser's framed stream protocol as one WAV frame. Override `stream()` only if
the model can produce lower-latency ordered segments.

## Visual roles

Visual packages are selected by responsibility, not by whether their
implementation happens to use YOLO.

### Visual adjudicator

Use `type=vision`, `role=adjudicator`, inherit
`core.vision.VisionAdjudicatorProvider`, and implement:

```python
def adjudicate(*, on_log, on_event, is_cancelled, timeout_seconds) -> dict:
    ...
```

An adjudicator owns the business decision contract: for the dice game this
includes pip detection, side scoring, winner calculation, and verification.
Business progress/results go to `on_event({...})`; `on_log(...)` is diagnostic
text only. A verified result event uses
`{"event":"result","verified":true,...}`.

The adjudicator returns physical outcomes (`LEFT`, `RIGHT`, or `TIE`). It does
not decide which side is the human player or the agent. That mapping belongs
to `backend/games/<game_id>/manifest.json` and is projected by the game
pipeline.

LLM verification/diagnosis engines are no longer configured inside the vision
package: the game pipeline resolves the global `llm` provider slot per round
and hands the engine to the adjudicator on the request object
(`VisionAdjudicationRequest.llm_provider`). `None` means "verification
disabled" — the detector-only result stands and the round never fails because
of a missing LLM.

Current package manifest:

```json
{
  "id": "vision_yolov8_adjudicator",
  "type": "vision",
  "role": "adjudicator",
  "entry": "provider.py:VisionYolov8Adjudicator",
  "capabilities": [
    "stable_frame_detection",
    "multiview_majority_vote",
    "stateless_multimodal_llm"
  ]
}
```

## LLM interface

LLM packages use `type=llm`, inherit `core.llm.LlmProvider`, and implement
the two bounded structured multimodal requests `verify(...)` and
`diagnose(...)`. Prompts, allowed outcomes and timeouts arrive with each
call from the game's vision profile — a provider is a pure transport adapter
(cloud API, vLLM/llama.cpp server, or a local resident engine) and knows
nothing about any game's rules.

The vision adjudicator is currently the only consumer; the pipeline resolves
the engine from the `llm` slot (game manifest override > arena default,
hot-reloaded per round). A deployment without an `llm` slot simply runs
detector-only rounds.

### Visual localizer

A future target-coordinate/spatial-perception package must use
`type=vision`, `role=localizer` and inherit
`core.vision.VisionLocalizerProvider`. It implements `locate(...)` and returns
coordinate data; it must not calculate or return a game winner. The object and
coordinate-frame schema should be finalized with the first actual localizer.

Games select adapters through semantic slots:

```json
"providers": {
  "vision_adjudicator": "vision_yolov8_adjudicator",
  "tts": "tts_new"
}
```

`DICE_VISION_ADJUDICATOR_PROVIDER=<id>` temporarily overrides the adjudicator.
The old `providers.vision` key and `DICE_VISION_PROVIDER` variable remain only
as migration aliases.

## Vision configuration ownership

Game-specific vision settings are embedded in
`backend/games/<game_id>/manifest.json` under `vision_profile`. They include
the model, class map, rule, prompts, per-game video path, timeout and
post-result hold. Do not add a second `vision_profile.json` beside the
manifest.

The YOLO runtime's deployment defaults live in
`vision/yolov8_adjudicator/config.json`: camera, inference/EP settings, RTSP
and the MediaMTX WebRTC base URL. The vision component config only contains
provider lifecycle/runtime paths and LLM endpoint/model/key settings.
Games still provide only a safe path such as
`/dice/`; the deployment WebRTC base URL comes from the runtime config.
