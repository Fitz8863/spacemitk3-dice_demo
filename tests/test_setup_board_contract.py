"""Contract for the public source-deployment script ``scripts/setup_board.sh``.

The script is the open-source front door: ``git clone`` + one command should
take a bare K3 board to a running system.  These tests pin the parts whose
silent regression would only surface on a fresh board, far away from help:

- the apt package list (a missing build dep fails at cmake; a missing runtime
  dep fails at start; the old bundle installer notably omitted python3-numpy,
  the backend's only third-party import);
- the GStreamer element gate (engine exits when the RTSP pipeline cannot be
  built, so a missing element kills adjudication, not just the preview);
- the asset download sources (default URLs must stay reachable-in-principle:
  same-repo Release for board assets, official model zoo for SenseVoice);
- the arm-less bypass in ``start_web.sh`` (DICE_NO_ARM) — open-source users
  without the NERO arm must still be able to start the service.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup_board.sh"
START_WEB = ROOT / "scripts" / "start_web.sh"


def _script():
    return SETUP.read_text(encoding="utf-8")


def test_setup_installs_every_runtime_package():
    script = _script()
    for pkg in (
        "spacemit-onnxruntime", "libsndfile1", "alsa-utils", "curl",
        "v4l-utils",
        "python3-numpy",  # backend/components/tts_moss_nano/daemon.py 唯一第三方 import
    ):
        assert pkg in script, f"setup must install {pkg}"


def test_setup_installs_build_toolchain_and_dev_headers():
    script = _script()
    for pkg in (
        "cmake", "build-essential", "pkg-config",
        "libopencv-dev", "ocl-icd-opencl-dev", "opencl-clhpp-headers",
        "libgstreamer1.0-dev", "libgstreamer-plugins-base1.0-dev",
    ):
        assert pkg in script, f"source build needs {pkg}"
    # pkg-config 是 Debian 过渡包名（二进制实际由 pkgconf 提供），必须按命令兜底，
    # 否则环境齐全的板会被误报缺包而拒绝继续（板上实测踩过）。
    assert "command -v pkg-config" in script


def test_setup_keeps_the_gstreamer_element_gate():
    script = _script()
    for element in (
        "v4l2src", "jpegdec", "videoconvert", "appsink", "appsrc", "queue",
        "h264parse", "x264enc", "rtspclientsink",
    ):
        assert element in script, f"setup must probe gst element {element}"
    assert "gst-inspect-1.0" in script
    # VPU elements warn-only; they must not gate (engine falls back to software).
    assert re.search(r"spacemitdec spacemith264enc", script)


def test_setup_builds_both_vision_runtimes():
    script = _script()
    for d in ("vision/yolov8_objdetect", "vision/yolov10_objdetect"):
        assert d in script, f"setup must build {d}"


def test_setup_downloads_assets_from_same_repo_release():
    """Board assets have no public mirror — the default URL must point at this
    repository's own Release so a clone is self-contained in provenance."""
    script = _script()
    match = re.search(
        r'DICE_ASSETS_BASE_URL:-"(?P<url>https://github\.com/[^"]+)"', script
    )
    assert match, "DICE_ASSETS_BASE_URL default must be a github.com URL"
    assert "spacemitk3-dice_demo" in match.group("url")
    # 断点续传与校验必须保留：公开网络下大文件半截传输是常态。
    assert "-C -" in script  # curl resume
    assert "sha256sum -c" in script


def test_setup_fetches_sensevoice_from_official_model_zoo():
    script = _script()
    assert (
        "https://archive.spacemit.com/spacemit-ai/model_zoo/asr/sensevoice.tar.gz" in script
    ), "SenseVoice has an official download source; do not re-host it"


def test_setup_mediamtx_points_at_the_riscv64_build():
    script = _script()
    match = re.search(r'DICE_MEDIAMTX_URL:-"(?P<url>[^"]+)"', script)
    assert match, "DICE_MEDIAMTX_URL default must exist"
    url = match.group("url")
    assert "spacemit-mediamtx" in url and "riscv64" in url


def test_start_web_has_armless_bypass():
    script = START_WEB.read_text(encoding="utf-8")
    assert 'DICE_NO_ARM:-0' in script, "arm-less users need DICE_NO_ARM=1 to start"
    # 有臂部署的 can0 三关（存在/UP/波特率）必须保留——免臂旁路不许偷走原语义。
    assert "can0 不存在" in script
    assert "bitrate 1000000" in script
