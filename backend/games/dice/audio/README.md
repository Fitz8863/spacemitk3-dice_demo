# Dice speech assets

Place pre-recorded WAV announcements in this directory. Reference a file from
`../manifest.json` relative to the game directory:

```json
"rules_intro": {
  "mode": "audio",
  "audio": "audio/rules_intro.wav",
  "text": "Optional transcript for operators"
}
```

The first version accepts WAV only. Paths must remain inside
`backend/games/dice/`; absolute paths and `..` segments are rejected.
