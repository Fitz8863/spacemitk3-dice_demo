from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_DEMO_TEXT = "你好，这是 MOSS TTS Nano 在 K3 上的演示。"
DEFAULT_MAX_NEW_FRAMES = 120
DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS = 24
DEFAULT_FIRST_CHUNK_TEXT_TOKENS = 16


def parse_cpu_set(raw_value: str) -> set[int]:
    result: set[int] = set()
    for item in raw_value.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid CPU range: {token}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    if not result:
        raise ValueError("CPU set must not be empty")
    return result


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the packaged MOSS-TTS-Nano SpaceMIT EP demo.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", default=DEFAULT_DEMO_TEXT)
    parser.add_argument(
        "--voice",
        default="Junhao",
        help="Built-in voice used when no reference audio is provided.",
    )
    parser.add_argument(
        "--prompt-audio",
        "--prompt-audio-path",
        "--reference-audio",
        "--reference-audio-path",
        dest="prompt_audio_path",
        default=None,
        help="Reference WAV used for voice cloning. When provided, it overrides --voice.",
    )
    parser.add_argument("--output", default="outputs/demo.wav")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Warm the runtime first, then read synthesis text from the terminal.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/interactive",
        help="Output directory used by --interactive.",
    )
    parser.add_argument(
        "--warmup-text",
        default=DEFAULT_DEMO_TEXT,
        help="Text used for the full synthesis warmup before the interactive prompt appears.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_MAX_NEW_FRAMES,
        help="Maximum frames per text chunk. Generation normally stops earlier at model EOS.",
    )
    parser.add_argument(
        "--voice-clone-max-text-tokens",
        type=int,
        default=DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS,
        help="Upper bound before the fixed-KV capacity guard applies a smaller chunk budget.",
    )
    parser.add_argument(
        "--first-chunk-text-tokens",
        type=int,
        default=DEFAULT_FIRST_CHUNK_TEXT_TOKENS,
        help=(
            "Use a smaller token budget for only the first interactive voice-clone chunk; "
            "set 0 to disable the optimization."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--main-cpus", default="0-7")
    parser.add_argument("--max-rtf", type=float)
    parser.add_argument("--require-stable-output", action="store_true")
    parser.add_argument(
        "--report-json",
        help="Write the full machine-readable report to this path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full diagnostic JSON report and enable ORT warnings.",
    )
    parser.add_argument(
        "--pcm-playback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In interactive mode, play each completed text chunk as raw PCM (default: enabled).",
    )
    parser.add_argument(
        "--audio-device",
        default="default",
        help="ALSA device passed to aplay for interactive PCM playback (default: default).",
    )
    parser.add_argument(
        "--save-wav",
        action="store_true",
        help="Also save a WAV copy in interactive mode; playback does not require this.",
    )
    parser.add_argument(
        "--strict-frames",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if platform.machine() != "riscv64":
        raise RuntimeError(f"this demo requires riscv64, got {platform.machine()}")
    if args.frames <= 0 or args.runs <= 0 or args.warmup < 0:
        raise ValueError("frames/runs must be positive and warmup must be non-negative")
    if args.voice_clone_max_text_tokens <= 0:
        raise ValueError("voice-clone-max-text-tokens must be positive")
    if args.first_chunk_text_tokens < 0:
        raise ValueError("first-chunk-text-tokens must be non-negative")
    first_chunk_token_budget = (
        args.first_chunk_text_tokens
        if args.interactive and args.pcm_playback and args.first_chunk_text_tokens > 0
        else None
    )
    if args.verbose:
        os.environ.setdefault("MOSS_TTS_ORT_LOG_SEVERITY", "2")

    requested_main_cpus = parse_cpu_set(args.main_cpus)
    available_cpus = set(os.sched_getaffinity(0))
    effective_main_cpus = requested_main_cpus & available_cpus
    if not effective_main_cpus:
        raise RuntimeError(
            f"requested X100 CPU set {sorted(requested_main_cpus)} is unavailable; "
            f"allowed CPUs are {sorted(available_cpus)}"
        )
    os.sched_setaffinity(0, effective_main_cpus)

    import onnxruntime as ort
    import spacemit_ort
    from onnx_tts_runtime import OnnxTtsRuntime

    if "+spacemit" not in ort.__version__:
        raise RuntimeError(f"vendor ONNX Runtime is required, got {ort.__version__}")

    ep_library = Path(spacemit_ort.EPLibPath).resolve()
    expected_ep_dir_text = os.environ.get("SPACEMIT_DEMO_EXPECT_EP_DIR", "").strip()
    if expected_ep_dir_text and ep_library.parent != Path(expected_ep_dir_text).resolve():
        raise RuntimeError(
            f"loaded EP is outside the delivery package: {ep_library}; "
            f"expected directory {Path(expected_ep_dir_text).resolve()}"
        )

    model_dir = Path(args.model_dir).resolve()
    output_path = Path(args.output).resolve()
    runtime_output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.interactive
        else output_path.parent
    )
    prompt_audio_path = Path(args.prompt_audio_path).expanduser().resolve() if args.prompt_audio_path else None
    if prompt_audio_path is not None and not prompt_audio_path.is_file():
        raise FileNotFoundError(f"reference audio not found: {prompt_audio_path}")
    session_start = time.perf_counter()
    runtime = OnnxTtsRuntime(
        model_dir=model_dir,
        thread_count=4,
        max_new_frames=args.frames,
        do_sample=True,
        sample_mode="fixed",
        execution_provider="spacemit",
        output_dir=runtime_output_dir,
    )
    session_seconds = time.perf_counter() - session_start

    if args.interactive:
        from moss_spacemit_interactive import run_interactive_session

        run_interactive_session(
            args=args,
            runtime=runtime,
            session_seconds=session_seconds,
            ep_library=ep_library,
            prompt_audio_path=prompt_audio_path,
        )
        return

    prepared = runtime.prepare_synthesis_text(
        text=args.text,
        voice=args.voice,
        enable_wetext=False,
        enable_normalize_tts_text=False,
    )
    voice_prepare_start = time.perf_counter()
    prompt_audio_codes = runtime.resolve_prompt_audio_codes(
        voice=args.voice,
        prompt_audio_path=prompt_audio_path,
    )
    voice_prepare_seconds = time.perf_counter() - voice_prepare_start
    decode_input = next(value for value in runtime.sessions["decode"].get_inputs() if value.name == "past_key_0")
    raw_capacity = decode_input.shape[1] if len(decode_input.shape) > 1 else None
    fixed_kv_capacity = int(raw_capacity) if isinstance(raw_capacity, int) else None
    fixed_prompt_rows = len(runtime.build_voice_clone_request_rows(prompt_audio_codes, [])["inputIds"])
    chunk_token_budget = int(args.voice_clone_max_text_tokens)
    if fixed_kv_capacity is not None:
        capacity_token_budget = fixed_kv_capacity - fixed_prompt_rows - args.frames
        if capacity_token_budget <= 0:
            raise RuntimeError(
                f"fixed KV capacity {fixed_kv_capacity} cannot reserve {args.frames} generation frames "
                f"for the selected reference; the reference prompt already occupies {fixed_prompt_rows} rows"
            )
        chunk_token_budget = min(chunk_token_budget, capacity_token_budget)
    initial_chunk_plans = runtime.plan_voice_clone_text(
        str(prepared["text"]),
        max_tokens=chunk_token_budget,
    )
    if not initial_chunk_plans:
        raise RuntimeError("input text produced no synthesis chunks")
    text_chunks = [str(plan["synthesis_text"]) for plan in initial_chunk_plans]
    original_text_chunks = [str(plan["original_text"]) for plan in initial_chunk_plans]
    text_chunk_token_counts = [int(plan["text_token_count"]) for plan in initial_chunk_plans]
    input_lengths = [
        len(runtime.build_voice_clone_request_rows(prompt_audio_codes, runtime.encode_text(chunk))["inputIds"])
        for chunk in text_chunks
    ]
    maximum_safe_frames = (
        min(fixed_kv_capacity - input_length for input_length in input_lengths)
        if fixed_kv_capacity is not None
        else args.frames
    )
    if maximum_safe_frames < args.frames:
        raise RuntimeError(
            f"capacity-aware chunking failed to reserve {args.frames} generation frames: "
            f"fixed KV capacity={fixed_kv_capacity}, input lengths={input_lengths}, "
            f"chunk token budget={chunk_token_budget}"
        )
    effective_frames = args.frames
    runtime.manifest["generation_defaults"]["max_new_frames"] = effective_frames

    runtime.warmup()
    warmup_output_path = output_path.parent / f".{output_path.name}.warmup-{os.getpid()}.wav"
    try:
        for _ in range(args.warmup):
            runtime.synthesize(
                text=args.text,
                voice=args.voice,
                prompt_audio_codes=prompt_audio_codes,
                output_audio_path=warmup_output_path,
                sample_mode="fixed",
                do_sample=True,
                streaming=False,
                max_new_frames=effective_frames,
                voice_clone_max_text_tokens=chunk_token_budget,
                first_chunk_text_tokens=first_chunk_token_budget,
                enable_wetext=False,
                enable_normalize_tts_text=False,
                seed=args.seed,
            )
    finally:
        warmup_output_path.unlink(missing_ok=True)

    rows: list[dict[str, object]] = []
    for run_index in range(args.runs):
        start = time.perf_counter()
        result = runtime.synthesize(
            text=args.text,
            voice=args.voice,
            prompt_audio_codes=prompt_audio_codes,
            output_audio_path=output_path,
            sample_mode="fixed",
            do_sample=True,
            streaming=False,
            max_new_frames=effective_frames,
            voice_clone_max_text_tokens=chunk_token_budget,
            first_chunk_text_tokens=first_chunk_token_budget,
            enable_wetext=False,
            enable_normalize_tts_text=False,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - start
        waveform = np.asarray(result["waveform"], dtype=np.float32)
        audio_tokens = np.asarray(result["audio_token_ids"], dtype=np.int32)
        chunk_generated_frames = [
            len(chunk_result["generated_frames"])
            for chunk_result in result["chunk_results"]
        ]
        chunk_stop_reasons = [
            str(chunk_result["stop_reason"])
            for chunk_result in result["chunk_results"]
        ]
        frame_limit_hit = any(reason != "audio_end" for reason in chunk_stop_reasons)
        audio_seconds = waveform.shape[0] / float(result["sample_rate"])
        rows.append(
            {
                "run": run_index + 1,
                "elapsed_seconds": elapsed,
                "audio_seconds": audio_seconds,
                "rtf": elapsed / audio_seconds,
                "generated_frames": int(audio_tokens.shape[0]),
                "chunk_generated_frames": chunk_generated_frames,
                "chunk_stop_reasons": chunk_stop_reasons,
                "frame_limit_hit": frame_limit_hit,
                "frame_limit_retry_count": len(result["retry_events"]),
                "retry_events": result["retry_events"],
                "synthesis_chunks": result["text_chunks"],
                "original_chunks": [
                    str(chunk_result["original_text"])
                    for chunk_result in result["chunk_results"]
                ],
                "audio_tokens_sha256": sha256_array(audio_tokens),
                "waveform_sha256": sha256_array(waveform),
                "waveform_shape": list(waveform.shape),
                "waveform_finite": bool(np.isfinite(waveform).all()),
                "waveform_peak": float(np.max(np.abs(waveform))) if waveform.size else 0.0,
            }
        )

    median_elapsed_seconds = statistics.median(float(row["elapsed_seconds"]) for row in rows)
    median_audio_seconds = statistics.median(float(row["audio_seconds"]) for row in rows)
    median_rtf = statistics.median(float(row["rtf"]) for row in rows)
    token_hashes = {str(row["audio_tokens_sha256"]) for row in rows}
    waveform_hashes = {str(row["waveform_sha256"]) for row in rows}
    acceptance_failures: list[str] = []
    if args.max_rtf is not None and median_rtf > args.max_rtf:
        acceptance_failures.append(f"median RTF {median_rtf:.6f} exceeds {args.max_rtf:.6f}")
    if args.require_stable_output and (len(token_hashes) != 1 or len(waveform_hashes) != 1):
        acceptance_failures.append("measured runs produced different token or waveform hashes")
    if not all(bool(row["waveform_finite"]) for row in rows):
        acceptance_failures.append("at least one waveform contains a non-finite value")
    if any(bool(row["frame_limit_hit"]) for row in rows):
        acceptance_failures.append(
            "at least one text chunk reached the generation frame limit; refusing a possibly truncated output"
        )

    report = {
        "demo": "MOSS-TTS-Nano SpaceMIT EP",
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "torch_loaded": "torch" in sys.modules or "torchaudio" in sys.modules,
        "model_dir": str(model_dir),
        "output": str(output_path),
        "reference_audio": (
            {
                "path": str(prompt_audio_path),
                "sha256": sha256_file(prompt_audio_path),
                "encoded_frames": len(prompt_audio_codes),
            }
            if prompt_audio_path is not None
            else None
        ),
        "voice": None if prompt_audio_path is not None else args.voice,
        "execution_provider": runtime.execution_provider,
        "actual_session_providers": {
            name: session.get_providers() for name, session in runtime.sessions.items()
        },
        "ep_library": str(ep_library),
        "ep_library_sha256": sha256_file(ep_library),
        "ep_intra_threads": os.environ.get("SPACEMIT_EP_INTRA_THREAD_NUM"),
        "ep_affinity": os.environ.get("SPACEMIT_EP_INTRA_THREAD_AFFINITY"),
        "process_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "session_creation_seconds": session_seconds,
        "voice_preparation_seconds": voice_prepare_seconds,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "requested_frames": args.frames,
        "effective_frames": effective_frames,
        "fixed_kv_capacity": fixed_kv_capacity,
        "fixed_prompt_rows": fixed_prompt_rows,
        "voice_clone_max_text_tokens": args.voice_clone_max_text_tokens,
        "effective_chunk_token_budget": chunk_token_budget,
        "original_text_chunks": original_text_chunks,
        "text_chunks": text_chunks,
        "text_chunk_token_counts": text_chunk_token_counts,
        "initial_chunk_plans": initial_chunk_plans,
        "input_lengths": input_lengths,
        "maximum_safe_frames": maximum_safe_frames,
        "median_elapsed_seconds": median_elapsed_seconds,
        "median_audio_seconds": median_audio_seconds,
        "median_rtf": median_rtf,
        "acceptance": {
            "max_rtf": args.max_rtf,
            "require_stable_output": args.require_stable_output,
            "passed": not acceptance_failures,
            "failures": acceptance_failures,
        },
        "runs": rows,
    }
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_json + "\n", encoding="utf-8")

    if args.verbose:
        print(report_json)
    else:
        mode = (
            f"voice clone ({prompt_audio_path})"
            if prompt_audio_path is not None
            else f"built-in voice {args.voice}"
        )
        print("MOSS-TTS-Nano / SpaceMIT EP")
        print(f"mode: {mode}")
        print(f"output: {output_path}")
        print(f"session creation: {session_seconds:.3f} s (excluded from RTF)")
        print(f"voice preparation: {voice_prepare_seconds:.3f} s (excluded from RTF)")
        print(f"warmup: {args.warmup} run(s) (excluded from RTF)")
        print(f"measured: {args.runs} run(s)")
        print(f"median synthesis: {median_elapsed_seconds:.3f} s")
        print(f"median audio: {median_audio_seconds:.3f} s")
        print(f"warm RTF: {median_rtf:.4f}")
        print(
            "frame-limit retries: "
            f"{sum(int(row['frame_limit_retry_count']) for row in rows)}"
        )
        print(
            "EP workers: "
            f"{os.environ.get('SPACEMIT_EP_INTRA_THREAD_NUM', 'unknown')} on "
            f"CPU {os.environ.get('SPACEMIT_EP_INTRA_THREAD_AFFINITY', 'unknown')}"
        )
        print(f"status: {'PASS' if not acceptance_failures else 'FAIL'}")
        if args.report_json:
            print(f"report: {report_path}")
    if acceptance_failures:
        raise SystemExit("acceptance failed: " + "; ".join(acceptance_failures))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        if "--verbose" in sys.argv:
            raise
        raise SystemExit(f"ERROR: {exc}") from None
