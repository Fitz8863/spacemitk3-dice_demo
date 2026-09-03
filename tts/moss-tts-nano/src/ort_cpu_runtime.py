from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort

SAMPLE_MODE_GREEDY = "greedy"
SAMPLE_MODE_FIXED = "fixed"
SAMPLE_MODE_FULL = "full"
EXECUTION_PROVIDER_CPU = "cpu"
EXECUTION_PROVIDER_CUDA = "cuda"
EXECUTION_PROVIDER_SPACEMIT = "spacemit"

# SpaceMIT EP 2.0.6 appends affinity values from the environment after parsing
# provider options. Keep affinity environment-only to avoid duplicating IDs.
SPACEMIT_PROVIDER_OPTION_KEYS = (
    "SPACEMIT_EP_INTRA_THREAD_NUM",
    "SPACEMIT_EP_INTER_THREAD_NUM",
    "SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD",
    "SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE",
    "SPACEMIT_EP_DENSE_ACCURACY_LEVEL",
    "SPACEMIT_EP_ENABLE_BLOCKLAYOUT",
    "SPACEMIT_EP_ENABLE_DMA",
    "SPACEMIT_EP_DISABLE_TLS_RELEASE",
    "SPACEMIT_EP_DISABLE_OP_TYPE_FILTER",
    "SPACEMIT_EP_DISABLE_OP_NAME_FILTER",
    "SPACEMIT_EP_DISABLE_PASSES_FILTER",
    "SPACEMIT_EP_DUMP_SUBGRAPHS",
    "SPACEMIT_EP_DUMP_TENSORS",
    "SPACEMIT_EP_DEBUG_PROFILE",
    "SPERT_BACKEND",
)

MANIFEST_CANDIDATE_RELATIVE_PATHS = (
    "browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
    "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
)
MODEL_DIR_ALIAS_MAP = {
    "MOSS-TTS-Nano-ONNX-CPU": "MOSS-TTS-Nano-100M-ONNX",
    "MOSS-Audio-Tokenizer-Nano-ONNX-CPU": "MOSS-Audio-Tokenizer-Nano-ONNX",
}


def _normalize_execution_provider(raw_execution_provider: str | None) -> str:
    normalized = str(raw_execution_provider or EXECUTION_PROVIDER_CPU).strip().lower()
    if normalized in {EXECUTION_PROVIDER_CPU, "CPUExecutionProvider".lower()}:
        return EXECUTION_PROVIDER_CPU
    if normalized in {EXECUTION_PROVIDER_CUDA, "gpu", "CUDAExecutionProvider".lower()}:
        return EXECUTION_PROVIDER_CUDA
    if normalized in {
        EXECUTION_PROVIDER_SPACEMIT,
        "smt",
        "SpaceMITExecutionProvider".lower(),
    }:
        return EXECUTION_PROVIDER_SPACEMIT
    raise ValueError("execution_provider must be one of: cpu, cuda, spacemit")


def _resolve_ort_providers(execution_provider: str) -> list[Any]:
    normalized = _normalize_execution_provider(execution_provider)
    if normalized == EXECUTION_PROVIDER_CPU:
        return ["CPUExecutionProvider"]
    if normalized == EXECUTION_PROVIDER_SPACEMIT:
        try:
            import spacemit_ort  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SpaceMITExecutionProvider was requested, but spacemit_ort is not installed."
            ) from exc
        available_providers = set(ort.get_available_providers())
        if "SpaceMITExecutionProvider" not in available_providers:
            available = ", ".join(ort.get_available_providers()) or "none"
            raise RuntimeError(
                "spacemit_ort was imported, but SpaceMITExecutionProvider is unavailable. "
                f"Available providers: {available}"
            )
        provider_options = {
            option_name: os.environ[option_name]
            for option_name in SPACEMIT_PROVIDER_OPTION_KEYS
            if option_name in os.environ
        }
        return [("SpaceMITExecutionProvider", provider_options), "CPUExecutionProvider"]
    available_providers = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available_providers:
        available = ", ".join(ort.get_available_providers()) or "none"
        raise RuntimeError(
            "CUDAExecutionProvider was requested, but this onnxruntime build does not expose it. "
            "Install onnxruntime-gpu that matches your CUDA/cuDNN runtime. "
            f"Available providers: {available}"
        )
    preload_dlls = getattr(ort, "preload_dlls", None)
    if callable(preload_dlls):
        preload_dlls()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _flatten3d_int32(nested: list[list[list[int]]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    dim2 = len(nested[0][0])
    data = np.zeros((dim0 * dim1 * dim2,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            for k in range(dim2):
                data[offset] = int(nested[i][j][k])
                offset += 1
    return data, [dim0, dim1, dim2]


def _flatten2d_int32(nested: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    data = np.zeros((dim0 * dim1,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            data[offset] = int(nested[i][j])
            offset += 1
    return data, [dim0, dim1]


def _slice_channel_major_audio(audio: np.ndarray, start_sample: int = 0, end_sample: int | None = None) -> list[np.ndarray]:
    if audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError(f"Unexpected audio tensor shape: {audio.shape}")
    channels = int(audio.shape[1])
    total_samples = int(audio.shape[2])
    start = max(0, int(start_sample))
    end = total_samples if end_sample is None else max(start, min(int(end_sample), total_samples))
    return [audio[0, channel_index, start:end].astype(np.float32, copy=False) for channel_index in range(channels)]


def _extract_last_hidden(hidden_states: np.ndarray) -> np.ndarray:
    if hidden_states.ndim == 2:
        return hidden_states.astype(np.float32, copy=False)
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"Unexpected global_hidden shape: {hidden_states.shape}")
    return hidden_states[:, -1, :].astype(np.float32, copy=False)


def _normalize_sample_mode(raw_sample_mode: str | None, raw_do_sample: bool = True) -> str:
    normalized = str(raw_sample_mode or "").strip()
    if normalized in {SAMPLE_MODE_GREEDY, SAMPLE_MODE_FIXED, SAMPLE_MODE_FULL}:
        return normalized
    if normalized == "mixed3":
        return SAMPLE_MODE_FIXED if raw_do_sample else SAMPLE_MODE_GREEDY
    return SAMPLE_MODE_GREEDY if not raw_do_sample else SAMPLE_MODE_FIXED


@dataclass
class _DecodeIoBindingStep:
    binding: Any
    input_ids: np.ndarray
    global_hidden: np.ndarray
    ort_values: list[Any]


class _GrowingKvDecodeIoBindings:
    """Pre-bind every decode length to one reusable set of KV-cache buffers."""

    def __init__(
        self,
        *,
        session: ort.InferenceSession,
        initial_past_by_name: dict[str, np.ndarray],
        decode_output_names: list[str],
        max_steps: int,
        row_width: int,
        hidden_size: int,
    ) -> None:
        self.session = session
        self.past_names = list(initial_past_by_name)
        if not self.past_names:
            raise ValueError("decode KV cache is empty")
        self.initial_length = int(initial_past_by_name[self.past_names[0]].shape[1])
        self.max_steps = max(1, int(max_steps))
        self.present_names = [name for name in decode_output_names if name != "global_hidden"]
        expected_present_names = [name.replace("past_", "present_", 1) for name in self.past_names]
        if self.present_names != expected_present_names:
            raise ValueError(
                "decode cache input/output ordering mismatch: "
                f"expected {expected_present_names}, got {self.present_names}"
            )

        maximum_length = self.initial_length + self.max_steps
        self.backings: dict[str, np.ndarray] = {}
        for name, initial_value in initial_past_by_name.items():
            if int(initial_value.shape[1]) != self.initial_length:
                raise ValueError(f"inconsistent decode cache length for {name}: {initial_value.shape}")
            shape = (int(initial_value.shape[0]), maximum_length, *initial_value.shape[2:])
            self.backings[name] = np.empty(shape, dtype=initial_value.dtype)

        self.steps = [
            self._build_step(
                past_length=self.initial_length + step_index,
                row_width=row_width,
                hidden_size=hidden_size,
            )
            for step_index in range(self.max_steps)
        ]
        self.reset(initial_past_by_name)

    def _build_step(self, *, past_length: int, row_width: int, hidden_size: int) -> _DecodeIoBindingStep:
        binding = self.session.io_binding()
        ort_values: list[Any] = []

        input_ids = np.empty((1, 1, row_width), dtype=np.int32)
        past_valid_lengths = np.asarray([past_length], dtype=np.int32)
        for name, value in (("input_ids", input_ids), ("past_valid_lengths", past_valid_lengths)):
            ort_value = ort.OrtValue.ortvalue_from_numpy(value)
            ort_values.append(ort_value)
            binding.bind_ortvalue_input(name, ort_value)

        global_hidden = np.empty((1, 1, hidden_size), dtype=np.float32)
        global_ort_value = ort.OrtValue.ortvalue_from_numpy(global_hidden)
        ort_values.append(global_ort_value)
        binding.bind_ortvalue_output("global_hidden", global_ort_value)

        for past_name, present_name in zip(self.past_names, self.present_names, strict=True):
            backing = self.backings[past_name]
            input_value = backing[:, :past_length, ...]
            output_value = backing[:, : past_length + 1, ...]
            input_ort_value = ort.OrtValue.ortvalue_from_numpy(input_value)
            output_ort_value = ort.OrtValue.ortvalue_from_numpy(output_value)
            ort_values.extend((input_ort_value, output_ort_value))
            binding.bind_ortvalue_input(past_name, input_ort_value)
            binding.bind_ortvalue_output(present_name, output_ort_value)

        return _DecodeIoBindingStep(
            binding=binding,
            input_ids=input_ids,
            global_hidden=global_hidden,
            ort_values=ort_values,
        )

    def reset(self, initial_past_by_name: dict[str, np.ndarray]) -> None:
        for name in self.past_names:
            initial_value = initial_past_by_name[name]
            if int(initial_value.shape[1]) != self.initial_length:
                raise ValueError(
                    f"decode cache length changed for {name}: "
                    f"expected {self.initial_length}, got {initial_value.shape[1]}"
                )
            np.copyto(self.backings[name][:, : self.initial_length, ...], initial_value)
        self.step_index = 0

    def run(self, input_ids: np.ndarray) -> np.ndarray:
        if self.step_index >= len(self.steps):
            raise RuntimeError("decode I/O binding cache exhausted")
        step = self.steps[self.step_index]
        np.copyto(step.input_ids, input_ids)
        self.session.run_with_iobinding(step.binding)
        self.step_index += 1
        return _extract_last_hidden(step.global_hidden)


class _FixedKvDecodeIoBindings:
    """Bind fixed-size KV inputs and outputs to the same reusable buffers."""

    def __init__(
        self,
        *,
        session: ort.InferenceSession,
        initial_past_by_name: dict[str, np.ndarray],
        decode_output_names: list[str],
        max_steps: int,
        capacity: int,
        row_width: int,
        hidden_size: int,
    ) -> None:
        self.session = session
        self.past_names = list(initial_past_by_name)
        if not self.past_names:
            raise ValueError("decode KV cache is empty")
        self.initial_length = int(initial_past_by_name[self.past_names[0]].shape[1])
        self.max_steps = max(1, int(max_steps))
        self.capacity = int(capacity)
        if self.initial_length + self.max_steps > self.capacity:
            raise ValueError(
                "fixed decode KV capacity is too small: "
                f"initial={self.initial_length}, steps={self.max_steps}, capacity={self.capacity}"
            )
        self.present_names = [name for name in decode_output_names if name != "global_hidden"]
        expected_present_names = [name.replace("past_", "present_", 1) for name in self.past_names]
        if self.present_names != expected_present_names:
            raise ValueError(
                "decode cache input/output ordering mismatch: "
                f"expected {expected_present_names}, got {self.present_names}"
            )

        self.binding = session.io_binding()
        self.ort_values: list[Any] = []
        self.input_ids = np.empty((1, 1, row_width), dtype=np.int32)
        self.past_valid_lengths = np.asarray([self.initial_length], dtype=np.int32)
        for name, value in (
            ("input_ids", self.input_ids),
            ("past_valid_lengths", self.past_valid_lengths),
        ):
            ort_value = ort.OrtValue.ortvalue_from_numpy(value)
            self.ort_values.append(ort_value)
            self.binding.bind_ortvalue_input(name, ort_value)

        self.global_hidden = np.empty((1, 1, hidden_size), dtype=np.float32)
        global_ort_value = ort.OrtValue.ortvalue_from_numpy(self.global_hidden)
        self.ort_values.append(global_ort_value)
        self.binding.bind_ortvalue_output("global_hidden", global_ort_value)

        self.backings: dict[str, np.ndarray] = {}
        for past_name, present_name in zip(self.past_names, self.present_names, strict=True):
            initial_value = initial_past_by_name[past_name]
            shape = (int(initial_value.shape[0]), self.capacity, *initial_value.shape[2:])
            backing = np.zeros(shape, dtype=initial_value.dtype)
            ort_value = ort.OrtValue.ortvalue_from_numpy(backing)
            self.backings[past_name] = backing
            self.ort_values.append(ort_value)
            self.binding.bind_ortvalue_input(past_name, ort_value)
            self.binding.bind_ortvalue_output(present_name, ort_value)
        self.reset(initial_past_by_name)

    def reset(self, initial_past_by_name: dict[str, np.ndarray]) -> None:
        for name in self.past_names:
            initial_value = initial_past_by_name[name]
            if int(initial_value.shape[1]) != self.initial_length:
                raise ValueError(
                    f"decode cache length changed for {name}: "
                    f"expected {self.initial_length}, got {initial_value.shape[1]}"
                )
            backing = self.backings[name]
            np.copyto(backing[:, : self.initial_length, ...], initial_value)
            backing[:, self.initial_length :, ...].fill(0)
        self.step_index = 0

    def run(self, input_ids: np.ndarray) -> np.ndarray:
        if self.step_index >= self.max_steps:
            raise RuntimeError("fixed decode I/O binding cache exhausted")
        np.copyto(self.input_ids, input_ids)
        self.past_valid_lengths[0] = self.initial_length + self.step_index
        self.session.run_with_iobinding(self.binding)
        self.step_index += 1
        return _extract_last_hidden(self.global_hidden)


class OrtCpuRuntime:
    def __init__(
        self,
        model_dir: str | Path,
        thread_count: int = 4,
        max_new_frames: int | None = None,
        do_sample: bool | None = None,
        sample_mode: str | None = None,
        execution_provider: str = EXECUTION_PROVIDER_CPU,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.thread_count = max(1, int(thread_count))
        self.execution_provider = _normalize_execution_provider(execution_provider)
        self.ort_providers = _resolve_ort_providers(self.execution_provider)
        self.manifest_path = self._resolve_manifest_path(self.model_dir)
        self.manifest_dir = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest = manifest
        if max_new_frames is not None:
            self.manifest["generation_defaults"]["max_new_frames"] = int(max_new_frames)
        if do_sample is not None:
            self.manifest["generation_defaults"]["do_sample"] = bool(do_sample)
        self.manifest["generation_defaults"]["sample_mode"] = _normalize_sample_mode(
            sample_mode if sample_mode is not None else self.manifest["generation_defaults"].get("sample_mode"),
            bool(self.manifest["generation_defaults"]["do_sample"]),
        )
        self.manifest["generation_defaults"]["do_sample"] = (
            self.manifest["generation_defaults"]["sample_mode"] != SAMPLE_MODE_GREEDY
        )
        self.tts_meta_path = self.resolve_manifest_relative_path(manifest["model_files"]["tts_meta"])
        self.codec_meta_path = self.resolve_manifest_relative_path(manifest["model_files"]["codec_meta"])
        self.tts_meta = json.loads(self.tts_meta_path.read_text(encoding="utf-8"))
        self.codec_meta = json.loads(self.codec_meta_path.read_text(encoding="utf-8"))
        self.rng = np.random.default_rng(1234)
        self.sessions = self._create_sessions()
        self._decode_io_binding_caches: dict[
            tuple[int, int, int], _GrowingKvDecodeIoBindings | _FixedKvDecodeIoBindings
        ] = {}

    @staticmethod
    def _resolve_manifest_path(model_dir: Path) -> Path:
        tried_paths: list[Path] = []
        for relative_path in MANIFEST_CANDIDATE_RELATIVE_PATHS:
            candidate = (model_dir / relative_path).resolve()
            tried_paths.append(candidate)
            if candidate.is_file():
                return candidate
        joined = ", ".join(str(path_value) for path_value in tried_paths)
        raise FileNotFoundError(f"browser_poc_manifest.json not found. tried: {joined}")

    def resolve_manifest_relative_path(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        resolved = (self.manifest_dir / relative).resolve()
        if resolved.exists():
            return resolved
        relative_text = str(relative).replace("\\", "/")
        for legacy_name, canonical_name in MODEL_DIR_ALIAS_MAP.items():
            legacy_fragment = f"/{legacy_name}/"
            if legacy_fragment not in f"/{relative_text}/":
                continue
            rewritten_text = relative_text.replace(legacy_name, canonical_name)
            rewritten = (self.manifest_dir / Path(rewritten_text)).resolve()
            if rewritten.exists():
                return rewritten
        return resolved

    def _session(self, path_value: Path) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self.thread_count
        options.inter_op_num_threads = 1
        log_severity = int(os.environ.get("MOSS_TTS_ORT_LOG_SEVERITY", "3"))
        if not 0 <= log_severity <= 4:
            raise ValueError("MOSS_TTS_ORT_LOG_SEVERITY must be between 0 and 4")
        options.log_severity_level = log_severity
        session = ort.InferenceSession(str(path_value), sess_options=options, providers=self.ort_providers)
        required_provider = {
            EXECUTION_PROVIDER_CUDA: "CUDAExecutionProvider",
            EXECUTION_PROVIDER_SPACEMIT: "SpaceMITExecutionProvider",
        }.get(self.execution_provider)
        if required_provider is not None and required_provider not in session.get_providers():
            raise RuntimeError(
                f"{required_provider} was requested, but ONNX Runtime created a session without it "
                f"for {path_value}. Actual providers: {session.get_providers()}"
            )
        return session

    def _create_sessions(self) -> dict[str, ort.InferenceSession]:
        tts_dir = self.tts_meta_path.parent
        codec_dir = self.codec_meta_path.parent
        return {
            "prefill": self._session(tts_dir / self.tts_meta["files"]["prefill"]),
            "decode": self._session(tts_dir / self.tts_meta["files"]["decode_step"]),
            "local_fixed_sampled_frame": self._session(
                tts_dir / self.tts_meta["files"]["local_fixed_sampled_frame"]
            ),
            "codec_encode": self._session(codec_dir / self.codec_meta["files"]["encode"]),
            "codec_decode": self._session(codec_dir / self.codec_meta["files"]["decode_full"]),
        }

    def list_builtin_voices(self) -> list[dict[str, Any]]:
        return list(self.manifest["builtin_voices"])

    def list_text_samples(self) -> list[dict[str, Any]]:
        return list(self.manifest["text_samples"])

    def warmup(self) -> None:
        voice = self.list_builtin_voices()[0]
        text_sample = self.list_text_samples()[0]
        request_rows = self.build_voice_clone_request_rows(voice["prompt_audio_codes"], text_sample["text_token_ids"])
        prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = _flatten2d_int32(request_rows["attentionMask"])
        outputs = self.sessions["prefill"].run(
            None,
            {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        output_names = [output.name for output in self.sessions["prefill"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        global_hidden = _extract_last_hidden(named_outputs["global_hidden"])
        if self.manifest["generation_defaults"]["sample_mode"] != SAMPLE_MODE_FIXED:
            raise RuntimeError("the slim delivery supports fixed sampling only")
        self.run_local_fixed_sampled_frame(
            global_hidden,
            previous_token_sets_by_channel=[set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))],
        )
        empty_frames = [([0] * int(self.manifest["tts_config"]["n_vq"]))]
        self.decode_full_audio(empty_frames)

    def build_text_rows(self, token_ids: list[int]) -> list[list[int]]:
        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        for token_id in token_ids:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = int(token_id)
            rows.append(row)
        return rows

    def build_audio_prefix_rows(self, prompt_audio_codes: list[list[int]], slot_token_id: int | None = None) -> list[list[int]]:
        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        resolved_slot_token_id = int(
            self.manifest["tts_config"]["audio_user_slot_token_id"] if slot_token_id is None else slot_token_id
        )
        for code_row in prompt_audio_codes:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = resolved_slot_token_id
            for index in range(min(len(code_row), int(self.manifest["tts_config"]["n_vq"]))):
                row[index + 1] = int(code_row[index])
            rows.append(row)
        return rows

    def build_voice_clone_request_rows(self, prompt_audio_codes: list[list[int]], text_token_ids: list[int]) -> dict[str, list[list[int]]]:
        prefix_text_token_ids = [
            int(token_id)
            for token_id in self.manifest["prompt_templates"]["user_prompt_prefix_token_ids"]
        ]
        im_start_token_id = int(self.manifest["tts_config"]["im_start_token_id"])
        # Released ONNX manifests can omit the leading user-turn marker even though
        # the native Torch prompt builder always emits it. Repair that legacy asset
        # contract here while avoiding a duplicate once the manifest is corrected.
        if not prefix_text_token_ids or prefix_text_token_ids[0] != im_start_token_id:
            prefix_text_token_ids.insert(0, im_start_token_id)
        prefix_text_token_ids.append(int(self.manifest["tts_config"]["audio_start_token_id"]))
        suffix_text_token_ids = [
            int(self.manifest["tts_config"]["audio_end_token_id"]),
            *self.manifest["prompt_templates"]["user_prompt_after_reference_token_ids"],
            *text_token_ids,
            *self.manifest["prompt_templates"]["assistant_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        rows = [
            *self.build_text_rows(prefix_text_token_ids),
            *self.build_audio_prefix_rows(prompt_audio_codes),
            *self.build_text_rows(suffix_text_token_ids),
        ]
        return {
            "inputIds": rows,
            "attentionMask": [[1 for _ in rows]],
        }

    def run_local_fixed_sampled_frame(
        self,
        global_hidden: np.ndarray,
        *,
        previous_token_sets_by_channel: list[set[int]],
    ) -> tuple[bool, list[int]]:
        audio_codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        repetition_seen_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= token_id < audio_codebook_size:
                    repetition_seen_mask[0, channel_index, token_id] = 1
        assistant_random_u = np.asarray([min(0.99999994, max(0.0, float(self.rng.random())))], dtype=np.float32)
        audio_random_u = np.asarray(
            [[min(0.99999994, max(0.0, float(self.rng.random()))) for _ in range(n_vq)]],
            dtype=np.float32,
        )
        outputs = self.sessions["local_fixed_sampled_frame"].run(
            None,
            {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "assistant_random_u": assistant_random_u,
                "audio_random_u": audio_random_u,
            },
        )
        output_names = [output.name for output in self.sessions["local_fixed_sampled_frame"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        frame_token_ids = np.asarray(named_outputs["frame_token_ids"]).reshape(-1).astype(np.int32, copy=False).tolist()
        should_continue = bool(int(np.asarray(named_outputs["should_continue"]).reshape(-1)[0]))
        return should_continue, [int(item) for item in frame_token_ids]

    def decode_full_audio(self, generated_frames: list[list[int]]) -> tuple[list[np.ndarray], int]:
        if not generated_frames:
            return [], 0
        audio_codes, dims = _flatten3d_int32([generated_frames])
        outputs = self.sessions["codec_decode"].run(
            None,
            {
                "audio_codes": audio_codes.reshape(dims),
                "audio_code_lengths": np.asarray([len(generated_frames)], dtype=np.int32),
            },
        )
        output_names = [output.name for output in self.sessions["codec_decode"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_length = int(named_outputs["audio_lengths"].reshape(-1)[0])
        return _slice_channel_major_audio(named_outputs["audio"], 0, audio_length), audio_length

    def generate_audio_frames(
        self,
        request_rows: dict[str, list[list[int]]],
        on_frame: Callable[[list[list[int]], int, list[int]], None] | None = None,
    ) -> dict[str, Any]:
        generation_defaults = self.manifest["generation_defaults"]
        if generation_defaults["sample_mode"] != SAMPLE_MODE_FIXED:
            raise RuntimeError("the slim delivery supports fixed sampling only")
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = _flatten2d_int32(request_rows["attentionMask"])
        outputs = self.sessions["prefill"].run(
            None,
            {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        output_names = [output.name for output in self.sessions["prefill"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        global_hidden = _extract_last_hidden(named_outputs["global_hidden"])
        past_valid_length = sum(int(item) for item in request_rows["attentionMask"][0])
        past_by_name = {
            output_name.replace("present_", "past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["prefill_output_names"][1:]
        }
        decode_io_bindings: _GrowingKvDecodeIoBindings | _FixedKvDecodeIoBindings | None = None
        decode_io_binding_disabled = os.environ.get(
            "SPACEMIT_MOSS_DISABLE_DECODE_IO_BINDING", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self.execution_provider == EXECUTION_PROVIDER_SPACEMIT and not decode_io_binding_disabled:
            initial_cache_length = int(next(iter(past_by_name.values())).shape[1])
            max_steps = int(generation_defaults["max_new_frames"])
            cache_key = (initial_cache_length, max_steps, row_width)
            decode_io_bindings = self._decode_io_binding_caches.get(cache_key)
            if decode_io_bindings is None:
                decode_session = self.sessions["decode"]
                decode_input = next(
                    value for value in decode_session.get_inputs() if value.name == "past_key_0"
                )
                decode_output = next(
                    value for value in decode_session.get_outputs() if value.name == "present_key_0"
                )
                input_capacity = decode_input.shape[1] if len(decode_input.shape) > 1 else None
                output_capacity = decode_output.shape[1] if len(decode_output.shape) > 1 else None
                fixed_capacity = (
                    int(input_capacity)
                    if isinstance(input_capacity, int)
                    and isinstance(output_capacity, int)
                    and int(input_capacity) == int(output_capacity)
                    else None
                )
                if fixed_capacity is not None:
                    decode_io_bindings = _FixedKvDecodeIoBindings(
                        session=decode_session,
                        initial_past_by_name=past_by_name,
                        decode_output_names=list(self.tts_meta["onnx"]["decode_output_names"]),
                        max_steps=max_steps,
                        capacity=fixed_capacity,
                        row_width=row_width,
                        hidden_size=int(global_hidden.shape[-1]),
                    )
                else:
                    decode_io_bindings = _GrowingKvDecodeIoBindings(
                        session=decode_session,
                        initial_past_by_name=past_by_name,
                        decode_output_names=list(self.tts_meta["onnx"]["decode_output_names"]),
                        max_steps=max_steps,
                        row_width=row_width,
                        hidden_size=int(global_hidden.shape[-1]),
                    )
                self._decode_io_binding_caches[cache_key] = decode_io_bindings
            else:
                decode_io_bindings.reset(past_by_name)
        generated_frames: list[list[int]] = []
        previous_token_sets_by_channel = [set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))]
        stop_reason = "frame_limit"

        for step_index in range(int(generation_defaults["max_new_frames"])):
            should_continue, frame = self.run_local_fixed_sampled_frame(
                global_hidden,
                previous_token_sets_by_channel=previous_token_sets_by_channel,
            )
            if not should_continue:
                stop_reason = "audio_end"
                break
            for channel_index, sampled_token in enumerate(frame):
                previous_token_sets_by_channel[channel_index].add(sampled_token)
            generated_frames.append(frame)

            next_row = np.full((1, 1, row_width), int(self.manifest["tts_config"]["audio_pad_token_id"]), dtype=np.int32)
            next_row[0, 0, 0] = int(self.manifest["tts_config"]["audio_assistant_slot_token_id"])
            for index, token in enumerate(frame):
                next_row[0, 0, index + 1] = int(token)
            if decode_io_bindings is not None:
                global_hidden = decode_io_bindings.run(next_row)
            else:
                decode_feeds: dict[str, np.ndarray] = {
                    "input_ids": next_row,
                    "past_valid_lengths": np.asarray([past_valid_length], dtype=np.int32),
                }
                for input_name in self.tts_meta["onnx"]["decode_input_names"][2:]:
                    decode_feeds[input_name] = past_by_name[input_name]
                decode_outputs = self.sessions["decode"].run(None, decode_feeds)
                decode_output_names = [output.name for output in self.sessions["decode"].get_outputs()]
                named_decode_outputs = dict(zip(decode_output_names, decode_outputs, strict=True))
                global_hidden = _extract_last_hidden(named_decode_outputs["global_hidden"])
                past_by_name = {
                    output_name.replace("present_", "past_"): named_decode_outputs[output_name]
                    for output_name in self.tts_meta["onnx"]["decode_output_names"][1:]
                }
            past_valid_length += 1
            if on_frame is not None:
                on_frame(generated_frames, step_index, frame)
        return {
            "generated_frames": generated_frames,
            "stop_reason": stop_reason,
            "audio_end_reached": stop_reason == "audio_end",
        }

__all__ = [
    "EXECUTION_PROVIDER_CPU",
    "EXECUTION_PROVIDER_CUDA",
    "EXECUTION_PROVIDER_SPACEMIT",
    "OrtCpuRuntime",
    "SAMPLE_MODE_FIXED",
    "SAMPLE_MODE_FULL",
    "SAMPLE_MODE_GREEDY",
    "_normalize_execution_provider",
    "_normalize_sample_mode",
]
