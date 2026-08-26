#!/usr/bin/env python3
"""Extract a Qwen3-TTS Base speaker embedding for the K3 runtime.

The K3 runtime expects a little-endian raw float32 vector with 1024 values
(4096 bytes).  This script intentionally uses Qwen's official Base-model
speaker encoder rather than trying to derive a speaker vector from the audio
samples directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Reference WAV/MP3/FLAC path")
    p.add_argument("--output", required=True, type=Path, help="Output raw .spk.bin path")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Local model directory or Hugging Face/ModelScope-compatible model id",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Inference device, for example cpu or cuda:0",
    )
    p.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="Model dtype; defaults to float32 on CPU and bfloat16 on CUDA",
    )
    p.add_argument(
        "--ref-text",
        default=None,
        help="Optional transcript, recorded in the sidecar metadata; x-vector extraction does not use it",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional JSON sidecar path (default: output path with .json suffix)",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    return p.parse_args()


def dtype_for(device: str, requested: str | None):
    import torch
    if requested:
        return getattr(torch, requested)
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audio(path: Path):
    """Read WAV/FLAC directly and fall back to librosa/audioread for MP3."""
    import soundfile as sf

    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        channels = int(audio.shape[1]) if getattr(audio, "ndim", 0) == 2 else 1
        return audio, int(sample_rate), channels
    except RuntimeError as soundfile_error:
        try:
            import librosa
        except ImportError as exc:
            raise SystemExit(
                f"soundfile cannot decode {path}; install librosa and ffmpeg for MP3 input"
            ) from exc
        try:
            audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
        except Exception as decode_error:
            raise SystemExit(
                f"could not decode {path}; install ffmpeg for MP3 input: {decode_error}"
            ) from soundfile_error
        if getattr(audio, "ndim", 0) == 2:
            channels = int(audio.shape[0])
            audio = audio.T
        else:
            channels = 1
        return audio.astype("float32", copy=False), int(sample_rate), channels


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input does not exist: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists (use --overwrite): {args.output}")

    # Import after argument validation so --help works without the ML stack.
    import numpy as np
    import torch
    audio, sample_rate, input_channels = load_audio(args.input)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise SystemExit("input must contain a non-empty mono or multi-channel audio stream")
    if not np.isfinite(audio).all():
        raise SystemExit("input contains NaN or infinity samples")

    device = args.device
    dtype = dtype_for(device, args.dtype)
    print(f"loading model: {args.model}", file=sys.stderr)
    print(f"device={device}, dtype={dtype}, input_sr={sample_rate}", file=sys.stderr)

    # A local Base checkpoint can be reduced to the speaker encoder only.
    # If speaker_encoder.safetensors is present, only that small subset is
    # loaded; otherwise the script reads speaker_encoder.* tensors from the
    # full model.safetensors. This avoids the talker, codec decoder, and speech
    # tokenizer just to create the 1024-D x-vector.
    local_model = Path(args.model)
    speaker_checkpoint = local_model / "speaker_encoder.safetensors"
    if not speaker_checkpoint.is_file():
        speaker_checkpoint = local_model / "model.safetensors"
    if local_model.is_dir() and speaker_checkpoint.is_file():
        import librosa
        from safetensors import safe_open
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSSpeakerEncoder,
            mel_spectrogram,
        )

        with (local_model / "config.json").open(encoding="utf-8") as f:
            model_config = json.load(f)
        speaker_config = Qwen3TTSSpeakerEncoderConfig(**model_config.get("speaker_encoder_config", {}))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(f"requested {device}, but CUDA is not available")
        speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_config).to(device=device, dtype=dtype)
        state = {}
        with safe_open(str(speaker_checkpoint), framework="pt", device="cpu") as sfh:
            for key in sfh.keys():
                if key.startswith("speaker_encoder."):
                    state[key[len("speaker_encoder."):]] = sfh.get_tensor(key)
                elif speaker_checkpoint.name == "speaker_encoder.safetensors":
                    state[key] = sfh.get_tensor(key)
        if not state:
            raise SystemExit("model.safetensors contains no speaker_encoder.* tensors")
        missing, unexpected = speaker_encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"speaker encoder weights mismatch: missing={missing}, unexpected={unexpected}")
        speaker_encoder.eval()

        if sample_rate != speaker_config.sample_rate:
            audio_24k = librosa.resample(
                y=audio.astype(np.float32),
                orig_sr=int(sample_rate),
                target_sr=int(speaker_config.sample_rate),
            )
        else:
            audio_24k = audio.astype(np.float32)
        mels = mel_spectrogram(
            torch.from_numpy(audio_24k).unsqueeze(0),
            n_fft=1024,
            num_mels=speaker_config.mel_dim,
            sampling_rate=speaker_config.sample_rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        embedding = speaker_encoder(mels.to(device=device, dtype=dtype))[0].detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        # For a Hub/model-id argument retain the official high-level API.
        # That path may download and load speech_tokenizer automatically.
        from qwen_tts import Qwen3TTSModel
        tts = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map=device,
            dtype=dtype,
        )
        prompt = tts.create_voice_clone_prompt(
            ref_audio=(audio, int(sample_rate)),
            ref_text=args.ref_text,
            x_vector_only_mode=True,
        )[0]
        embedding = prompt.ref_spk_embedding.detach().to(device="cpu", dtype=torch.float32).numpy()
    embedding = np.asarray(embedding, dtype="<f4").reshape(-1)

    if embedding.size != 1024:
        raise SystemExit(
            f"unexpected speaker embedding dimension: {embedding.size}; "
            "the current K3 runtime requires 1024"
        )
    if not np.isfinite(embedding).all():
        raise SystemExit("speaker embedding contains NaN or infinity")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".part")
    embedding.tofile(tmp)
    if tmp.stat().st_size != 4096:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"unexpected output size: {tmp.stat().st_size} bytes")
    tmp.replace(args.output)

    metadata_path = args.metadata or args.output.with_suffix(args.output.suffix + ".json")
    metadata = {
        "format": "qwen3-tts-k3-speaker-embedding",
        "dtype": "float32",
        "endianness": "little",
        "dimensions": int(embedding.size),
        "bytes": int(args.output.stat().st_size),
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "input_sample_rate": int(sample_rate),
        "input_channels": input_channels,
        "model": args.model,
        "mode": "x_vector_only",
        "reference_text": args.ref_text,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "output_sha256": sha256(args.output),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
