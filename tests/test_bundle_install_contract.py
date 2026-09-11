"""Contract for the distribution bundle scripts.

The engine's whole video path is GStreamer-based, and ``yolov8_camera`` exits
with an error when the RTSP publish pipeline cannot be built — so a board
missing a plugin loses YOLO adjudication, not merely the preview. The
installer must therefore gate on the real elements, not just on package names.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "scripts" / "bundle_install_dice.sh"
MAKE = ROOT / "scripts" / "make_bundle.sh"


def test_installer_checks_the_gstreamer_plugin_packages():
    script = INSTALL.read_text(encoding="utf-8")
    for pkg in (
        "gstreamer1.0-tools",
        "gstreamer1.0-plugins-base",
        "gstreamer1.0-plugins-good",
        "gstreamer1.0-plugins-bad",
        "gstreamer1.0-plugins-ugly",  # provides x264enc, the software fallback
        "gstreamer1.0-rtsp",          # provides rtspclientsink, the publish sink
    ):
        assert pkg in script, f"installer must verify {pkg}"


def test_installer_probes_every_required_gstreamer_element():
    script = INSTALL.read_text(encoding="utf-8")
    for element in (
        "v4l2src", "jpegdec", "videoconvert", "appsink", "appsrc", "queue",
        "h264parse", "x264enc", "rtspclientsink",
    ):
        assert element in script, f"installer must probe gst element {element}"
    # Probing must be element-level (gst-inspect), because a distro can ship a
    # plugin package without the element the engine needs.
    assert "gst-inspect-1.0" in script


def test_vpu_elements_are_optional_because_the_engine_falls_back():
    script = INSTALL.read_text(encoding="utf-8")
    assert "GST_OPTIONAL=(spacemitdec spacemith264enc)" in script
    # Optional misses warn; they must not be collected into the fatal list.
    optional_at = script.index("GST_OPTIONAL=(")
    fatal_at = script.index("MISSING_ELEMENTS+=(")
    assert fatal_at < optional_at, "VPU elements must not gate startup"


def test_installer_verifies_the_whole_moss_runtime_tree():
    """models/ alone is not enough: python/lib/voice/assets are gitignored and
    arrive only via the packer's rsync, so they are the easiest thing to drop."""
    script = INSTALL.read_text(encoding="utf-8")
    assert 'for d in python lib voice assets; do' in script


def test_packer_and_installer_agree_on_the_moss_runtime_dirs():
    make = MAKE.read_text(encoding="utf-8")
    install = INSTALL.read_text(encoding="utf-8")
    for d in ("python", "lib", "voice", "assets"):
        assert f"tts/moss-tts-nano/{d}" in make
        assert d in install
