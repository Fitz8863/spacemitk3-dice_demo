from __future__ import annotations

import copy
import logging
import math
import os
import shutil
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import sentencepiece as spm
import soundfile as sf
from scipy.signal import resample_poly

from moss_tts_nano.defaults import DEFAULT_OUTPUT_DIR
from text_normalization_pipeline import WeTextProcessingManager, prepare_tts_request_texts

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR
from ort_cpu_runtime import (
    OrtCpuRuntime,
    _normalize_sample_mode,
    EXECUTION_PROVIDER_CPU,
    SAMPLE_MODE_FIXED,
    SAMPLE_MODE_GREEDY,
)

DEFAULT_BROWSER_ONNX_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_BROWSER_ONNX_TTS_DIR = DEFAULT_BROWSER_ONNX_MODEL_DIR / "MOSS-TTS-Nano-100M-ONNX"
DEFAULT_BROWSER_ONNX_CODEC_DIR = DEFAULT_BROWSER_ONNX_MODEL_DIR / "MOSS-Audio-Tokenizer-Nano-ONNX"
DEFAULT_BROWSER_ONNX_TTS_REPO_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
DEFAULT_BROWSER_ONNX_CODEC_REPO_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
DEFAULT_BROWSER_ONNX_TTS_REPO_URL = f"https://huggingface.co/{DEFAULT_BROWSER_ONNX_TTS_REPO_ID}"
DEFAULT_BROWSER_ONNX_CODEC_REPO_URL = f"https://huggingface.co/{DEFAULT_BROWSER_ONNX_CODEC_REPO_ID}"
DEFAULT_BROWSER_ONNX_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "infer_onnx_output.wav"
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS = 0.40
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS = 0.24
SENTENCE_END_PUNCTUATION = set(".!?。！？；;")
CLAUSE_SPLIT_PUNCTUATION = set(",，、；;：:")
CLOSING_PUNCTUATION = set("\"'”’)]}）】》」』")
CONTINUATION_END_PUNCTUATION = set(",，、：:")
OUTER_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "《": "》",
}
DEFAULT_FRAME_LIMIT_RETRY_DEPTH = 2
DEFAULT_FIXED_KV_TEXT_TOKEN_BUDGET = 24


MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS = (
    "browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
    "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
)


def _resolve_model_dir_path(model_dir: str | Path | None) -> Path:
    if model_dir is None:
        return DEFAULT_BROWSER_ONNX_MODEL_DIR.expanduser().resolve()
    return Path(model_dir).expanduser().resolve()


def _default_model_dir_requested(model_dir: str | Path | None) -> bool:
    if model_dir is None:
        return True
    return _resolve_model_dir_path(model_dir) == DEFAULT_BROWSER_ONNX_MODEL_DIR.expanduser().resolve()


def _find_manifest_path(model_dir: Path) -> Path | None:
    for relative_path in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS:
        candidate = (model_dir / relative_path).resolve()
        if candidate.is_file():
            return candidate
    return None


def _directory_contains_all(parent: Path, required_names: Sequence[str]) -> bool:
    return all((parent / name).exists() for name in required_names)


def _find_directory_with_required_names(root_dir: Path, required_names: Sequence[str]) -> Path | None:
    if not root_dir.exists():
        return None
    if _directory_contains_all(root_dir, required_names):
        return root_dir
    sentinel_name = str(required_names[0])
    for candidate in root_dir.rglob(sentinel_name):
        parent = candidate.parent
        if _directory_contains_all(parent, required_names):
            return parent
    return None


def _promote_directory_contents(source_dir: Path, target_dir: Path) -> None:
    if source_dir.resolve() == target_dir.resolve():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if destination.exists():
            continue
        shutil.move(str(child), str(destination))


def _normalize_download_layout(target_dir: Path, required_names: Sequence[str]) -> None:
    candidate_dir = _find_directory_with_required_names(target_dir, required_names)
    if candidate_dir is None:
        return
    _promote_directory_contents(candidate_dir, target_dir)


def _snapshot_download_repo(
    *,
    repo_id: str,
    local_dir: Path,
    allow_patterns: Sequence[str],
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required to auto-download ONNX assets. Install it with `pip install huggingface_hub`."
        ) from exc
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns),
    )


def _download_default_browser_onnx_assets(model_dir: Path) -> None:
    logging.info("browser_onnx assets missing under %s; downloading from Hugging Face.", model_dir)
    logging.info("browser_onnx TTS repo: %s", DEFAULT_BROWSER_ONNX_TTS_REPO_URL)
    logging.info("browser_onnx codec repo: %s", DEFAULT_BROWSER_ONNX_CODEC_REPO_URL)
    tts_dir = model_dir / DEFAULT_BROWSER_ONNX_TTS_DIR.name
    codec_dir = model_dir / DEFAULT_BROWSER_ONNX_CODEC_DIR.name
    _snapshot_download_repo(
        repo_id=DEFAULT_BROWSER_ONNX_TTS_REPO_ID,
        local_dir=tts_dir,
        allow_patterns=("*.onnx", "*.data", "*.json", "tokenizer.model"),
    )
    _snapshot_download_repo(
        repo_id=DEFAULT_BROWSER_ONNX_CODEC_REPO_ID,
        local_dir=codec_dir,
        allow_patterns=("*.onnx", "*.data", "*.json"),
    )
    _normalize_download_layout(
        tts_dir,
        required_names=("browser_poc_manifest.json", "tts_browser_onnx_meta.json", "tokenizer.model"),
    )
    _normalize_download_layout(
        codec_dir,
        required_names=("codec_browser_onnx_meta.json",),
    )


def ensure_browser_onnx_model_dir(model_dir: str | Path | None = None) -> Path:
    resolved_model_dir = _resolve_model_dir_path(model_dir)
    manifest_path = _find_manifest_path(resolved_model_dir)
    if manifest_path is not None:
        return resolved_model_dir
    if not _default_model_dir_requested(model_dir):
        tried_paths = [str((resolved_model_dir / item).resolve()) for item in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS]
        raise FileNotFoundError(
            "browser_onnx model assets not found under the provided --model-dir. tried: " + ", ".join(tried_paths)
        )
    _download_default_browser_onnx_assets(resolved_model_dir)
    manifest_path = _find_manifest_path(resolved_model_dir)
    if manifest_path is None:
        tried_paths = [str((resolved_model_dir / item).resolve()) for item in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS]
        raise FileNotFoundError(
            "browser_onnx assets were downloaded but browser_poc_manifest.json is still missing. "
            + "tried: "
            + ", ".join(tried_paths)
        )
    return resolved_model_dir


def _contains_cjk(text: str) -> bool:
    for character in str(text or ""):
        if (
            "\u4e00" <= character <= "\u9fff"
            or "\u3400" <= character <= "\u4dbf"
            or "\u3040" <= character <= "\u30ff"
            or "\uac00" <= character <= "\ud7af"
        ):
            return True
    return False


def _prepare_text_for_sentence_chunking(text: str) -> str:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise ValueError("Text prompt cannot be empty.")
    normalized_text = normalized_text.replace("\r", " ").replace("\n", " ")
    while "  " in normalized_text:
        normalized_text = normalized_text.replace("  ", " ")
    text_without_closers = normalized_text.rstrip()
    while text_without_closers and text_without_closers[-1] in CLOSING_PUNCTUATION:
        text_without_closers = text_without_closers[:-1].rstrip()
    if _contains_cjk(normalized_text):
        if not text_without_closers or text_without_closers[-1] not in SENTENCE_END_PUNCTUATION:
            normalized_text += "。"
        return normalized_text
    if normalized_text[:1].islower():
        normalized_text = normalized_text[:1].upper() + normalized_text[1:]
    if normalized_text[-1].isalnum():
        normalized_text += "."
    if len([item for item in normalized_text.split() if item]) < 5:
        normalized_text = f"        {normalized_text}"
    return normalized_text


def _split_text_by_punctuation(text: str, punctuation: set[str]) -> list[str]:
    sentences: list[str] = []
    current_chars: list[str] = []
    index = 0
    normalized_text = str(text or "")
    while index < len(normalized_text):
        character = normalized_text[index]
        current_chars.append(character)
        decimal_point = (
            character == "."
            and index > 0
            and index + 1 < len(normalized_text)
            and normalized_text[index - 1].isdigit()
            and normalized_text[index + 1].isdigit()
        )
        if character in punctuation and not decimal_point:
            lookahead = index + 1
            while lookahead < len(normalized_text) and normalized_text[lookahead] in CLOSING_PUNCTUATION:
                current_chars.append(normalized_text[lookahead])
                lookahead += 1
            sentence = "".join(current_chars).strip()
            if sentence:
                sentences.append(sentence)
            current_chars.clear()
            while lookahead < len(normalized_text) and normalized_text[lookahead].isspace():
                lookahead += 1
            index = lookahead
            continue
        index += 1
    tail = "".join(current_chars).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _join_sentence_parts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if _contains_cjk(left) or _contains_cjk(right):
        return left + right
    return f"{left} {right}"


def _repair_synthesis_chunk_boundary(text: str) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return candidate
    expected_closer = OUTER_QUOTE_PAIRS.get(candidate[0])
    if expected_closer is not None and len(candidate) > 1:
        candidate = candidate[1:-1].strip() if candidate.endswith(expected_closer) else candidate[1:].strip()
    for opening_quote, closing_quote in OUTER_QUOTE_PAIRS.items():
        if opening_quote == closing_quote:
            continue
        while candidate.count(opening_quote) > candidate.count(closing_quote):
            candidate = candidate.replace(opening_quote, "", 1)
        while candidate.count(closing_quote) > candidate.count(opening_quote):
            closing_index = candidate.rfind(closing_quote)
            candidate = candidate[:closing_index] + candidate[closing_index + 1 :]
    trailing_closers = ""
    while candidate and candidate[-1] in CLOSING_PUNCTUATION:
        trailing_closers = candidate[-1] + trailing_closers
        candidate = candidate[:-1].rstrip()
    if not candidate:
        return str(text or "").strip()
    if candidate[-1] in CONTINUATION_END_PUNCTUATION:
        candidate = candidate[:-1].rstrip() + ("。" if _contains_cjk(candidate) else ".")
    elif candidate[-1] not in SENTENCE_END_PUNCTUATION:
        candidate += "。" if _contains_cjk(candidate) else "."
    return candidate + trailing_closers


def _merge_audio_channels(channel_arrays: list[np.ndarray]) -> np.ndarray:
    if not channel_arrays:
        return np.zeros((0, 1), dtype=np.float32)
    if len(channel_arrays) == 1:
        return np.asarray(channel_arrays[0], dtype=np.float32).reshape(-1, 1)
    min_length = min(int(channel.shape[0]) for channel in channel_arrays)
    trimmed = [np.asarray(channel[:min_length], dtype=np.float32) for channel in channel_arrays]
    return np.stack(trimmed, axis=1)


def _concat_waveforms(waveforms: list[np.ndarray]) -> np.ndarray:
    if not waveforms:
        return np.zeros((0, 1), dtype=np.float32)
    non_empty = [waveform for waveform in waveforms if waveform.size > 0]
    if not non_empty:
        channel_count = int(waveforms[0].shape[1]) if waveforms[0].ndim == 2 and waveforms[0].shape[1] > 0 else 1
        return np.zeros((0, channel_count), dtype=np.float32)
    return np.concatenate(non_empty, axis=0)


def _write_waveform_to_wav(path: str | Path, waveform: np.ndarray, sample_rate: int) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype(np.int16)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        with wave.open(str(temporary_path), "wb") as wav_file:
            wav_file.setnchannels(int(pcm16.shape[1]))
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm16.tobytes())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


class OnnxTtsRuntime(OrtCpuRuntime):
    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        thread_count: int = 4,
        max_new_frames: int | None = None,
        do_sample: bool | None = None,
        sample_mode: str | None = None,
        execution_provider: str = EXECUTION_PROVIDER_CPU,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        resolved_model_dir = ensure_browser_onnx_model_dir(model_dir)
        super().__init__(
            model_dir=resolved_model_dir,
            thread_count=thread_count,
            max_new_frames=max_new_frames,
            do_sample=do_sample,
            sample_mode=sample_mode,
            execution_provider=execution_provider,
        )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_relative_path = str(self.manifest["model_files"].get("tokenizer_model", "tokenizer.model"))
        tokenizer_path = self.resolve_manifest_relative_path(tokenizer_relative_path)
        self.sp_model = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self._text_normalizer_manager: WeTextProcessingManager | None = None

    def _ensure_text_normalizer(self, enable_wetext: bool) -> WeTextProcessingManager | None:
        if not enable_wetext:
            return None
        if self._text_normalizer_manager is None:
            self._text_normalizer_manager = WeTextProcessingManager()
        snapshot = self._text_normalizer_manager.ensure_ready()
        if not snapshot.ready:
            raise RuntimeError(snapshot.error or snapshot.message)
        return self._text_normalizer_manager

    def encode_text(self, text: str) -> list[int]:
        return [int(token_id) for token_id in self.sp_model.encode(str(text or ""), out_type=int)]

    def count_text_tokens(self, text: str) -> int:
        return len(self.encode_text(text))

    def prepare_synthesis_text(
        self,
        *,
        text: str,
        voice: str = "",
        prompt_text: str = "",
        enable_wetext: bool = True,
        enable_normalize_tts_text: bool = True,
    ) -> dict[str, object]:
        text_normalizer_manager = self._ensure_text_normalizer(enable_wetext)
        return prepare_tts_request_texts(
            text=text,
            prompt_text=prompt_text,
            voice=voice,
            enable_wetext=bool(enable_wetext),
            enable_normalize_tts_text=bool(enable_normalize_tts_text),
            text_normalizer_manager=text_normalizer_manager,
        )

    def split_text_by_token_budget(self, text: str, max_tokens: int) -> list[str]:
        remaining_text = str(text or "").strip()
        if not remaining_text:
            return []
        pieces: list[str] = []
        preferred_boundary_chars = set(CLAUSE_SPLIT_PUNCTUATION) | set(SENTENCE_END_PUNCTUATION) | {" "}
        while remaining_text:
            if self.count_text_tokens(remaining_text) <= max_tokens:
                pieces.append(remaining_text)
                break
            low = 1
            high = len(remaining_text)
            best_prefix_length = 1
            while low <= high:
                middle = (low + high) // 2
                candidate = remaining_text[:middle].strip()
                if not candidate:
                    low = middle + 1
                    continue
                if self.count_text_tokens(candidate) <= max_tokens:
                    best_prefix_length = middle
                    low = middle + 1
                else:
                    high = middle - 1
            cut_index = best_prefix_length
            prefix = remaining_text[:best_prefix_length]
            preferred_index = -1
            scan_min = max(-1, len(prefix) - 25)
            for scan_index in range(len(prefix) - 1, scan_min, -1):
                if prefix[scan_index] in preferred_boundary_chars:
                    preferred_index = scan_index + 1
                    break
            if preferred_index > 0:
                cut_index = preferred_index
            piece = remaining_text[:cut_index].strip()
            if not piece:
                piece = remaining_text[:best_prefix_length].strip()
                cut_index = best_prefix_length
            pieces.append(piece)
            remaining_text = remaining_text[cut_index:].strip()
        return pieces

    def split_text_balanced(self, text: str, max_tokens: int) -> list[str]:
        safe_max_tokens = max(1, int(max_tokens))
        pending = [str(text or "").strip()]
        pieces: list[str] = []
        preferred_boundaries = set(CLAUSE_SPLIT_PUNCTUATION) | set(SENTENCE_END_PUNCTUATION) | {" "}
        while pending:
            current = pending.pop(0).strip()
            if not current:
                continue
            if self.count_text_tokens(current) <= safe_max_tokens or len(current) <= 1:
                pieces.append(current)
                continue
            candidates: list[tuple[tuple[int, int, int, int, int], int]] = []
            midpoint = len(current) // 2
            for cut_index in range(1, len(current)):
                left = current[:cut_index].strip()
                right = current[cut_index:].strip()
                if not left or not right:
                    continue
                left_tokens = self.count_text_tokens(left)
                right_tokens = self.count_text_tokens(right)
                orphan_penalty = int(len(left) < 2 or len(right) < 2)
                boundary_penalty = int(current[cut_index - 1] not in preferred_boundaries)
                capacity_penalty = int(
                    left_tokens > safe_max_tokens or right_tokens > safe_max_tokens
                )
                score = (
                    orphan_penalty,
                    boundary_penalty,
                    capacity_penalty,
                    max(left_tokens, right_tokens),
                    abs(cut_index - midpoint),
                )
                candidates.append((score, cut_index))
            if not candidates:
                pieces.append(current)
                continue
            _score, best_cut_index = min(candidates, key=lambda item: item[0])
            left = current[:best_cut_index].strip()
            right = current[best_cut_index:].strip()
            if not left or not right or left == current or right == current:
                pieces.append(current)
                continue
            pending = [left, right] + pending
        return pieces

    def _pack_text_slices(self, text_slices: list[str], max_tokens: int) -> list[str]:
        chunks: list[str] = []
        current_chunk = ""
        for text_slice in text_slices:
            normalized_slice = str(text_slice or "").strip()
            if not normalized_slice:
                continue
            candidate = _join_sentence_parts(current_chunk, normalized_slice)
            if current_chunk and self.count_text_tokens(candidate) > max_tokens:
                chunks.append(current_chunk)
                current_chunk = normalized_slice
            else:
                current_chunk = candidate
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _plans_from_raw_chunks(
        self,
        raw_chunks: list[str],
        *,
        max_tokens: int,
        retry_depth: int,
        parent_text: str | None,
    ) -> list[dict[str, Any]]:
        pending = [str(item or "").strip() for item in raw_chunks if str(item or "").strip()]
        plans: list[dict[str, Any]] = []
        while pending:
            original_text = pending.pop(0)
            synthesis_text = _repair_synthesis_chunk_boundary(original_text)
            token_count = self.count_text_tokens(synthesis_text)
            if token_count > max_tokens and len(original_text) > 1:
                reduced_budget = max(1, max_tokens - 1)
                pieces = self.split_text_balanced(original_text, reduced_budget)
                if len(pieces) == 1 and pieces[0].strip() == original_text:
                    midpoint = max(1, len(original_text) // 2)
                    pieces = [original_text[:midpoint], original_text[midpoint:]]
                pending = [piece.strip() for piece in pieces if piece.strip()] + pending
                continue
            plans.append(
                {
                    "original_text": original_text,
                    "synthesis_text": synthesis_text,
                    "text_token_count": token_count,
                    "boundary_repaired": synthesis_text != original_text,
                    "retry_depth": int(retry_depth),
                    "parent_text": parent_text,
                }
            )
        return plans

    def plan_voice_clone_text(
        self,
        text: str,
        max_tokens: int = 75,
        *,
        retry_depth: int = 0,
        parent_text: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []
        safe_max_tokens = max(1, int(max_tokens))
        prepared_text = _prepare_text_for_sentence_chunking(normalized_text)
        sentence_candidates = _split_text_by_punctuation(prepared_text, SENTENCE_END_PUNCTUATION) or [prepared_text.strip()]
        sentence_slices: list[str] = []
        for sentence_text in sentence_candidates:
            normalized_sentence = sentence_text.strip()
            if not normalized_sentence:
                continue
            sentence_token_count = self.count_text_tokens(normalized_sentence)
            if sentence_token_count <= safe_max_tokens:
                sentence_slices.append(normalized_sentence)
                continue
            clause_candidates = _split_text_by_punctuation(normalized_sentence, CLAUSE_SPLIT_PUNCTUATION)
            if len(clause_candidates) <= 1:
                clause_candidates = [normalized_sentence]
            for clause_text in clause_candidates:
                normalized_clause = clause_text.strip()
                if not normalized_clause:
                    continue
                clause_token_count = self.count_text_tokens(normalized_clause)
                if clause_token_count <= safe_max_tokens:
                    sentence_slices.append(normalized_clause)
                    continue
                for piece in self.split_text_balanced(normalized_clause, safe_max_tokens):
                    normalized_piece = piece.strip()
                    if normalized_piece:
                        sentence_slices.append(normalized_piece)
        raw_chunks = self._pack_text_slices(sentence_slices, safe_max_tokens)
        if len(raw_chunks) == 1 and self.count_text_tokens(normalized_text) <= safe_max_tokens:
            raw_chunks = [normalized_text]
        return self._plans_from_raw_chunks(
            raw_chunks,
            max_tokens=safe_max_tokens,
            retry_depth=retry_depth,
            parent_text=parent_text,
        )

    def split_voice_clone_text(self, text: str, max_tokens: int = 75) -> list[str]:
        return [
            str(plan["synthesis_text"])
            for plan in self.plan_voice_clone_text(text, max_tokens=max_tokens)
        ]

    def estimate_voice_clone_inter_chunk_pause_seconds(self, text_chunk: str) -> float:
        word_count = len([item for item in str(text_chunk or "").strip().split() if item])
        return (
            DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS
            if word_count <= 4
            else DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS
        )

    def _load_reference_audio(self, reference_audio_path: str | Path) -> np.ndarray:
        audio_path = Path(reference_audio_path).expanduser().resolve()
        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = np.asarray(waveform.T, dtype=np.float32)
        target_sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        target_channels = int(self.codec_meta["codec_config"]["channels"])
        if sample_rate != target_sample_rate:
            rate_gcd = math.gcd(int(sample_rate), target_sample_rate)
            waveform = resample_poly(
                waveform,
                target_sample_rate // rate_gcd,
                int(sample_rate) // rate_gcd,
                axis=-1,
            ).astype(np.float32, copy=False)
        current_channels = int(waveform.shape[0])
        if current_channels == target_channels:
            pass
        elif current_channels == 1 and target_channels > 1:
            waveform = np.repeat(waveform, target_channels, axis=0)
        elif current_channels > 1 and target_channels == 1:
            waveform = waveform.mean(axis=0, keepdims=True, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported reference audio channel conversion: {current_channels} -> {target_channels}")
        return waveform[np.newaxis, ...].astype(np.float32, copy=False)

    def encode_reference_audio(self, reference_audio_path: str | Path) -> list[list[int]]:
        waveform = self._load_reference_audio(reference_audio_path)
        waveform_length = int(waveform.shape[-1])
        outputs = self.sessions["codec_encode"].run(
            None,
            {
                "waveform": waveform,
                "input_lengths": np.asarray([waveform_length], dtype=np.int32),
            },
        )
        output_names = [output.name for output in self.sessions["codec_encode"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_codes = np.asarray(named_outputs["audio_codes"], dtype=np.int32)
        audio_code_lengths = np.asarray(named_outputs["audio_code_lengths"], dtype=np.int32)
        code_length = int(audio_code_lengths.reshape(-1)[0])
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        prompt_audio_codes: list[list[int]] = []
        for frame_index in range(code_length):
            prompt_audio_codes.append(
                [int(audio_codes[0, frame_index, quantizer_index]) for quantizer_index in range(num_quantizers)]
            )
        return prompt_audio_codes

    def resolve_prompt_audio_codes(
        self,
        *,
        voice: str | None,
        prompt_audio_path: str | Path | None,
    ) -> list[list[int]]:
        if prompt_audio_path:
            return self.encode_reference_audio(prompt_audio_path)
        resolved_voice = str(voice or self.list_builtin_voices()[0]["voice"])
        voice_row = next((item for item in self.list_builtin_voices() if item["voice"] == resolved_voice), None)
        if voice_row is None:
            raise ValueError(f"Built-in voice not found: {resolved_voice}")
        return list(voice_row["prompt_audio_codes"])

    def decode_full_audio_safe(self, generated_frames: list[list[int]]) -> np.ndarray:
        channel_arrays, _audio_length = self.decode_full_audio(generated_frames)
        return _merge_audio_channels(channel_arrays)

    def synthesize_single_chunk(
        self,
        *,
        text: str,
        prompt_audio_codes: list[list[int]],
        streaming: bool,
    ) -> dict[str, Any]:
        # ``streaming`` means chunk-level PCM delivery here. The bundled codec
        # is still a full-chunk decoder, so generation remains non-streaming at
        # the individual audio-frame level.
        text_token_ids = self.encode_text(text)
        request_rows = self.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
        generation_result = self.generate_audio_frames(request_rows)
        generated_frames = list(generation_result["generated_frames"])
        stop_reason = str(generation_result["stop_reason"])
        waveform = (
            self.decode_full_audio_safe(generated_frames)
            if stop_reason == "audio_end"
            else np.zeros((0, int(self.codec_meta["codec_config"]["channels"])), dtype=np.float32)
        )
        return {
            "text": text,
            "text_token_ids": text_token_ids,
            "generated_frames": generated_frames,
            "stop_reason": stop_reason,
            "audio_end_reached": bool(generation_result["audio_end_reached"]),
            "waveform": waveform,
        }

    def _retry_plans_for_chunk(
        self,
        plan: dict[str, Any],
        *,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        token_count = max(1, int(plan["text_token_count"]))
        retry_budget_upper_bound = max(1, int(max_tokens) - 1)
        retry_budget = max(
            1,
            min(retry_budget_upper_bound, max(2, (token_count * 3 + 3) // 4)),
        )
        original_text = str(plan["original_text"])
        prepared_text = _prepare_text_for_sentence_chunking(original_text)
        clause_candidates = _split_text_by_punctuation(
            prepared_text,
            CLAUSE_SPLIT_PUNCTUATION,
        ) or [prepared_text]
        raw_retry_chunks: list[str] = []
        for clause_candidate in clause_candidates:
            raw_retry_chunks.extend(
                self.split_text_balanced(clause_candidate, retry_budget)
            )
        retry_plans = self._plans_from_raw_chunks(
            raw_retry_chunks,
            retry_depth=int(plan["retry_depth"]) + 1,
            max_tokens=retry_budget,
            parent_text=original_text,
        )
        if len(retry_plans) == 1 and retry_plans[0]["synthesis_text"] == plan["synthesis_text"]:
            midpoint = max(1, len(original_text) // 2)
            retry_plans = self._plans_from_raw_chunks(
                [original_text[:midpoint], original_text[midpoint:]],
                max_tokens=retry_budget,
                retry_depth=int(plan["retry_depth"]) + 1,
                parent_text=original_text,
            )
        return retry_plans

    def _synthesize_planned_chunk(
        self,
        *,
        plan: dict[str, Any],
        prompt_audio_codes: list[list[int]],
        streaming: bool,
        max_tokens: int,
        max_retry_depth: int,
        retry_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        result = self.synthesize_single_chunk(
            text=str(plan["synthesis_text"]),
            prompt_audio_codes=prompt_audio_codes,
            streaming=streaming,
        )
        result.update(plan)
        if result["stop_reason"] == "audio_end":
            return [result]
        retry_events.append(
            {
                "original_text": plan["original_text"],
                "synthesis_text": plan["synthesis_text"],
                "text_token_count": plan["text_token_count"],
                "generated_frames": len(result["generated_frames"]),
                "stop_reason": result["stop_reason"],
                "retry_depth": plan["retry_depth"],
            }
        )
        if int(plan["retry_depth"]) >= max_retry_depth:
            raise RuntimeError(
                f"text chunk did not reach audio_end after {max_retry_depth} retries: "
                f"{plan['original_text']!r}"
            )
        self.rng.bit_generator.state = rng_state
        retry_plans = self._retry_plans_for_chunk(plan, max_tokens=max_tokens)
        if len(retry_plans) < 2:
            raise RuntimeError(f"cannot safely split frame-limited text chunk: {plan['original_text']!r}")
        if bool(plan.get("suppress_leading_pause", False)):
            for retry_plan in retry_plans:
                retry_plan["suppress_leading_pause"] = True
        recovered_results: list[dict[str, Any]] = []
        for retry_plan in retry_plans:
            recovered_results.extend(
                self._synthesize_planned_chunk(
                    plan=retry_plan,
                    prompt_audio_codes=prompt_audio_codes,
                    streaming=streaming,
                    max_tokens=max_tokens,
                    max_retry_depth=max_retry_depth,
                    retry_events=retry_events,
                )
            )
        return recovered_results

    def synthesize(
        self,
        *,
        text: str,
        voice: str | None = None,
        prompt_audio_path: str | Path | None = None,
        prompt_audio_codes: list[list[int]] | None = None,
        output_audio_path: str | Path | None = None,
        sample_mode: str | None = None,
        do_sample: bool = True,
        streaming: bool = False,
        max_new_frames: int | None = None,
        voice_clone_max_text_tokens: int = DEFAULT_FIXED_KV_TEXT_TOKEN_BUDGET,
        first_chunk_text_tokens: int | None = None,
        enable_wetext: bool = True,
        enable_normalize_tts_text: bool = True,
        seed: int | None = None,
        max_frame_limit_retry_depth: int = DEFAULT_FRAME_LIMIT_RETRY_DEPTH,
        on_pcm_chunk: Callable[[np.ndarray, int, dict[str, Any]], None] | None = None,
        write_wav: bool = True,
    ) -> dict[str, Any]:
        if max_new_frames is not None:
            self.manifest["generation_defaults"]["max_new_frames"] = int(max_new_frames)
        normalized_sample_mode = _normalize_sample_mode(sample_mode, do_sample)
        self.manifest["generation_defaults"]["sample_mode"] = normalized_sample_mode
        self.manifest["generation_defaults"]["do_sample"] = normalized_sample_mode != SAMPLE_MODE_GREEDY
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        prepared_texts = self.prepare_synthesis_text(
            text=text,
            voice=str(voice or ""),
            enable_wetext=enable_wetext,
            enable_normalize_tts_text=enable_normalize_tts_text,
        )
        prepared_text = str(prepared_texts["text"])
        resolved_prompt_audio_codes = (
            [list(frame) for frame in prompt_audio_codes]
            if prompt_audio_codes is not None
            else self.resolve_prompt_audio_codes(voice=voice, prompt_audio_path=prompt_audio_path)
        )
        chunk_token_budget = max(1, int(voice_clone_max_text_tokens))
        if first_chunk_text_tokens is not None and int(first_chunk_text_tokens) <= 0:
            raise ValueError("first_chunk_text_tokens must be positive when provided")
        initial_chunk_plans = self.plan_voice_clone_text(
            prepared_text,
            max_tokens=chunk_token_budget,
        )
        first_chunk_token_budget = (
            min(chunk_token_budget, int(first_chunk_text_tokens))
            if first_chunk_text_tokens is not None
            else None
        )
        if first_chunk_token_budget is not None and initial_chunk_plans:
            first_plan = initial_chunk_plans[0]
            if int(first_plan["text_token_count"]) > first_chunk_token_budget:
                first_chunk_plans = self.plan_voice_clone_text(
                    str(first_plan["original_text"]),
                    max_tokens=first_chunk_token_budget,
                )
                if first_chunk_plans:
                    for plan_index, plan in enumerate(first_chunk_plans):
                        if plan_index > 0:
                            plan["suppress_leading_pause"] = True
                    initial_chunk_plans = first_chunk_plans + initial_chunk_plans[1:]
        all_waveforms: list[np.ndarray] = []
        all_generated_frames: list[list[int]] = []
        sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        channels = int(self.codec_meta["codec_config"]["channels"])
        chunk_results: list[dict[str, Any]] = []
        retry_events: list[dict[str, Any]] = []
        for chunk_plan in initial_chunk_plans:
            recovered_results = self._synthesize_planned_chunk(
                plan=chunk_plan,
                prompt_audio_codes=resolved_prompt_audio_codes,
                streaming=bool(streaming or on_pcm_chunk is not None),
                max_tokens=chunk_token_budget,
                max_retry_depth=max(0, int(max_frame_limit_retry_depth)),
                retry_events=retry_events,
            )
            for chunk_result in recovered_results:
                if chunk_results and not bool(chunk_result.get("suppress_leading_pause", False)):
                    pause_seconds = self.estimate_voice_clone_inter_chunk_pause_seconds(
                        str(chunk_results[-1]["synthesis_text"])
                    )
                    pause_samples = max(0, int(round(sample_rate * pause_seconds)))
                    if pause_samples > 0:
                        pause_waveform = np.zeros((pause_samples, channels), dtype=np.float32)
                        all_waveforms.append(pause_waveform)
                        if on_pcm_chunk is not None:
                            on_pcm_chunk(
                                pause_waveform,
                                sample_rate,
                                {
                                    "kind": "pause",
                                    "chunk_index": len(chunk_results),
                                    "duration_seconds": pause_samples / float(sample_rate),
                                },
                            )
                chunk_results.append(chunk_result)
                chunk_waveform = np.asarray(chunk_result["waveform"], dtype=np.float32)
                all_waveforms.append(chunk_waveform)
                all_generated_frames.extend(chunk_result["generated_frames"])
                if on_pcm_chunk is not None:
                    on_pcm_chunk(
                        chunk_waveform,
                        sample_rate,
                        {
                            "kind": "audio",
                            "chunk_index": len(chunk_results) - 1,
                            "text": str(chunk_result["synthesis_text"]),
                            "frames": len(chunk_result["generated_frames"]),
                            "duration_seconds": (
                                chunk_waveform.shape[0] / float(sample_rate)
                                if chunk_waveform.ndim == 2
                                else 0.0
                            ),
                        },
                    )
        if not chunk_results:
            raise RuntimeError("input text produced no synthesis chunks")
        waveform = _concat_waveforms(all_waveforms)
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise RuntimeError("synthesis produced an invalid or empty waveform")
        audio_path: Path | None = None
        if write_wav:
            resolved_output_audio_path = (
                Path(output_audio_path).expanduser().resolve()
                if output_audio_path
                else (self.output_dir / DEFAULT_BROWSER_ONNX_OUTPUT_PATH.name).resolve()
            )
            audio_path = _write_waveform_to_wav(resolved_output_audio_path, waveform, sample_rate)
        return {
            "audio_path": str(audio_path) if audio_path is not None else None,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "audio_token_ids": np.asarray(all_generated_frames, dtype=np.int32),
            "text_chunks": [str(result["synthesis_text"]) for result in chunk_results],
            "initial_chunk_plans": initial_chunk_plans,
            "voice_clone_max_text_tokens": chunk_token_budget,
            "first_chunk_text_tokens": first_chunk_token_budget,
            "retry_events": retry_events,
            "prepared_texts": prepared_texts,
            "sample_mode": normalized_sample_mode,
            "do_sample": normalized_sample_mode != SAMPLE_MODE_GREEDY,
            "streaming": bool(streaming),
            "chunk_results": chunk_results,
        }
