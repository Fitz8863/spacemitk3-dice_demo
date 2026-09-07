#!/usr/bin/env bash
set -euo pipefail

module_dir="components/model_zoo/asr"

python3 -m py_compile \
  "${module_dir}/python/spacemit_asr/engine.py" \
  "${module_dir}/python/spacemit_asr/__init__.py" \
  "${module_dir}/python/examples/asr_file_demo.py"

python3 - <<'PY'
import enum
import sys
import types
from pathlib import Path

module_dir = Path("components/model_zoo/asr")
source_root = module_dir / "python"

bindings = (module_dir / "python/asr_bindings.cpp").read_text()
assert '.value("FUNASR"' in bindings
assert '.value("QWEN3_ASR"' in bindings
assert '.value("GEMMA4_ASR"' in bindings
assert 'RecognitionTask' in bindings
assert '.def_static("zipformer"' in bindings
assert '.def_static("funasr"' in bindings
assert '.def_static("funasr_cloud"' in bindings
assert '.def_static("qwen3_asr"' in bindings
assert '.def_static("gemma4_asr"' in bindings


class Language(enum.Enum):
    AUTO = 0
    ZH = 1
    EN = 2
    JA = 3
    KO = 4
    YUE = 5


class BackendType(enum.Enum):
    SENSEVOICE = 0
    FUNASR = 1
    WHISPER = 2
    PARAFORMER = 3
    QWEN3_ASR = 4
    ZIPFORMER = 5
    CUSTOM = 6
    GEMMA4_ASR = 7


class RecognitionTask(enum.Enum):
    TRANSCRIBE = 0
    TRANSLATE = 1


class NativeConfig:
    def __init__(self):
        self.backend = BackendType.SENSEVOICE
        self.language = Language.AUTO
        self.sample_rate = 16000
        self.punctuation_enabled = True
        self.enable_emotion = False
        self.task = RecognitionTask.TRANSCRIBE
        self.extra_params = {}

    @staticmethod
    def sensevoice(model_dir):
        config = NativeConfig()
        config.backend = BackendType.SENSEVOICE
        config.model_path = f"{model_dir}/model_quant_optimized.onnx"
        return config

    @staticmethod
    def zipformer(model_dir):
        config = NativeConfig()
        config.backend = BackendType.ZIPFORMER
        config.model_path = f"{model_dir}/ctc-epoch-20-avg-1-chunk-16-left-128.q.onnx"
        return config

    @staticmethod
    def qwen3_asr(endpoint, model, timeout):
        config = NativeConfig()
        config.backend = BackendType.QWEN3_ASR
        config.extra_params = {
            "endpoint": endpoint,
            "model": model,
            "timeout": str(timeout),
        }
        return config

    @staticmethod
    def funasr(endpoint, model, timeout):
        config = NativeConfig()
        config.backend = BackendType.FUNASR
        config.extra_params = {
            "endpoint": endpoint,
            "model": model,
            "timeout": str(timeout),
        }
        return config

    @staticmethod
    def gemma4_asr(endpoint, model, timeout, task):
        config = NativeConfig()
        config.backend = BackendType.GEMMA4_ASR
        config.task = task
        config.extra_params = {
            "endpoint": endpoint,
            "model": model,
            "timeout": str(timeout),
        }
        return config


class NativeEngine:
    @staticmethod
    def get_available_backends():
        return [
            BackendType.SENSEVOICE,
            BackendType.FUNASR,
            BackendType.QWEN3_ASR,
            BackendType.ZIPFORMER,
            BackendType.GEMMA4_ASR,
        ]

    @staticmethod
    def get_version():
        return "contract-test"


fake_asr = types.SimpleNamespace(
    Language=Language,
    RecognitionTask=RecognitionTask,
    BackendType=BackendType,
    ASRConfig=NativeConfig,
    ASREngine=NativeEngine,
)

sys.modules["spacemit_asr._spacemit_asr"] = fake_asr
sys.path.insert(0, str(source_root.resolve()))

import spacemit_asr

assert spacemit_asr.BackendType.QWEN3_ASR.name == "QWEN3_ASR"
assert spacemit_asr.Config().backend == spacemit_asr.BackendType.SENSEVOICE
assert spacemit_asr.Config.zipformer().backend == spacemit_asr.BackendType.ZIPFORMER

funasr = spacemit_asr.Config.funasr(timeout=5)
assert funasr.backend == spacemit_asr.BackendType.FUNASR
assert funasr._config.extra_params["endpoint"].endswith("/v1/audio/transcriptions")
assert funasr._config.extra_params["model"] == "funasr"
assert funasr._config.extra_params["timeout"] == "5"

qwen = spacemit_asr.Config.qwen3_asr(
    endpoint="http://127.0.0.1:8063/v1/chat/completions",
    model="qwen3-asr",
    timeout=7,
)
assert qwen.backend == spacemit_asr.BackendType.QWEN3_ASR
assert qwen._config.extra_params["timeout"] == "7"

gemma = spacemit_asr.Config.gemma4_asr(task="translate", timeout=9)
assert gemma.backend == spacemit_asr.BackendType.GEMMA4_ASR
assert gemma.task == spacemit_asr.RecognitionTask.TRANSLATE
assert gemma._config.extra_params["endpoint"].endswith("/v1/audio/transcriptions")
assert gemma._config.extra_params["model"] == "gemma4-asr"
assert gemma._config.extra_params["timeout"] == "9"

try:
    spacemit_asr.Config.funasr(task="translate")
except TypeError:
    pass
else:
    raise AssertionError("Fun-ASR factory must not accept a translation task")

try:
    spacemit_asr.Config(backend="funasr", task="translate")
except ValueError as error:
    assert "only supported by the Gemma4" in str(error)
else:
    raise AssertionError("non-Gemma backend must reject translation")

backends = spacemit_asr.Engine.get_available_backends()
assert backends == [
    spacemit_asr.BackendType.SENSEVOICE,
    spacemit_asr.BackendType.FUNASR,
    spacemit_asr.BackendType.QWEN3_ASR,
    spacemit_asr.BackendType.ZIPFORMER,
    spacemit_asr.BackendType.GEMMA4_ASR,
]

print("PASS --python-api-contract")
PY
