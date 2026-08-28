# Local decode-cache variants

This delivery keeps the current dynamic-KV graph on `main` and stores fixed-KV
alternatives in Git branches. The fixed graphs are derived from the original
optimized/pruned 320-row graph; only the cache dimension and `fixed_kv_indices`
initializer differ.

```bash
# Keep the dynamic graph (default)
git switch main

# Fixed optimized graphs for K3 A/B tests
git switch kv-fixed-512
git switch kv-fixed-1024

# Restore the original fixed baseline
git switch kv-fixed-320
```

Run `git status` before switching. Do not switch with uncommitted changes to the
tracked model, metadata, README, or checksum files. Each branch keeps the active
filename `models/MOSS-TTS-Nano-100M-ONNX-xslim-dynq/moss_tts_decode_step.onnx`,
so the existing launch commands do not change.

The graph builder can reproduce the larger fixed variants from a fixed source:

```bash
python3 scripts/build_fixed_kv_variants.py \
  --source /path/to/moss_tts_decode_step.inplace-fixed320.pruned.onnx \
  --output-dir /tmp/moss-tts-fixed-variants \
  --capacity 512 --capacity 1024
```

The host can inspect the graphs, but final SpaceMIT EP/TCM and RTF validation
must be performed on the K3 board.
