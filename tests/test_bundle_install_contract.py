"""Contract for the distribution bundle scripts.

The engine's whole video path is GStreamer-based, and ``yolov8_camera`` exits
with an error when the RTSP publish pipeline cannot be built — so a board
missing a plugin loses YOLO adjudication, not merely the preview. The
installer must therefore gate on the real elements, not just on package names.

The other half is the opposite failure: assets that are not in git (models,
engine binaries, Sherpa-ONNX libraries) travel only through the packer's
rsync whitelist, so a local TTS engine can be silently left out of the package
and only surface on a new board when someone switches slots to it. The rules
below derive that whitelist from the components' own configs instead of
listing engines by hand.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "scripts" / "bundle_install_dice.sh"
MAKE = ROOT / "scripts" / "make_bundle.sh"
COMPONENTS = ROOT / "backend" / "components"

# 组件 config 没声明、但运行期确实会读的目录（tts_moss_nano 的 launcher
# 自己拼 root/python、root/lib 等，见 backend/components/tts_moss_nano/launcher.py）。
EXTRA_LOCAL_ASSET_DIRS = {
    "tts_moss_nano": ("python", "lib", "voice", "assets"),
}

# 有意**不**进分发包的本地 TTS 引擎：必须写明理由，否则白名单会变成随手加的地方，
# 而"新板上切过去才发现缺资产"这类问题正是从"随手漏掉"来的。
LOCAL_ENGINES_NOT_SHIPPED = {
    "tts_qwen3": "模型 qwen3-tts-0.6b 共 2.0G，分发主包不带；要用需自行准备并拷到板端",
}


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


def _whitelist_entries(make_text):
    """REQUIRED_DIRS=( ... ) 里的条目（去掉注释与空行）。"""
    block = make_text.split("REQUIRED_DIRS=(", 1)[1].split("\n)", 1)[0]
    return [
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _declared_local_tts_assets():
    """每个 kind=local 的 TTS 组件声明的资产目录（项目相对路径）。"""
    declared = {}
    for cfg_path in sorted(COMPONENTS.glob("tts_*/config.json")):
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        runtime = config.get("runtime") or {}
        if runtime.get("kind") != "local" or not runtime.get("root"):
            continue
        root = str(runtime["root"]).rstrip("/")
        paths = {
            f"{root}/{extra}"
            for extra in EXTRA_LOCAL_ASSET_DIRS.get(cfg_path.parent.name, ())
        }
        for field in ("model_dir", "sherpa_lib_dir"):
            value = runtime.get(field)
            if value:
                paths.add(f"{root}/{value}")
        binary = runtime.get("binary")
        if binary:
            parent = str(Path(binary).parent)
            paths.add(root if parent == "." else f"{root}/{parent}")
        declared[cfg_path.parent.name] = paths
    return declared


def test_packer_carries_every_local_tts_engine_asset():
    """本地 TTS 的板端资产全在 .gitignore 里，只能靠打包白名单带出去。

    matcha 当初就是漏在这里：包能装、默认也能跑，直到在新板上把
    providers.tts_local 切过去，才发现没有它的模型与 Sherpa-ONNX 运行库。
    规则：每个 kind=local 的 TTS 引擎，要么资产进包，要么在
    LOCAL_ENGINES_NOT_SHIPPED 里写明为什么不带 —— 不允许"悄悄没有"。
    """
    entries = _whitelist_entries(MAKE.read_text(encoding="utf-8"))
    declared = _declared_local_tts_assets()
    # 解析失效（例如 REQUIRED_DIRS 写法变了）必须让测试响，而不是静默通过。
    assert declared, "没解析到任何 kind=local 的 TTS 组件，测试本身失效了"
    missing = [
        f"{component}: {path}"
        for component, paths in sorted(declared.items())
        if component not in LOCAL_ENGINES_NOT_SHIPPED
        for path in sorted(paths)
        if not any(path == entry or path.startswith(f"{entry}/") for entry in entries)
    ]
    assert not missing, f"打包白名单缺少本地 TTS 资产目录: {missing}"


def test_the_exclusion_list_has_no_stale_entries():
    """排除清单必须对应真实存在的本地引擎，理由也要非空 —— 免得它过期后
    变成一句骗人的注释。"""
    shipped = _declared_local_tts_assets()
    for component, reason in LOCAL_ENGINES_NOT_SHIPPED.items():
        assert component in shipped, f"{component} 已不存在，排除清单该删了"
        assert reason.strip(), f"{component} 的排除理由不能为空"


def test_installer_verifies_the_matcha_engine_assets():
    """matcha 的三块资产（服务二进制 / 模型 / Sherpa-ONNX 库）缺一即装不起来,
    而它们只在切换槽位时才被用到, 所以要在安装自检里提前拦截。"""
    script = INSTALL.read_text(encoding="utf-8")
    assert "tts/matcha-tts/build-cpp/matcha_tts_service" in script
    for name in ("model-steps-3.q.onnx", "vocos-16khz-univ.q.onnx", "lexicon.txt", "tokens.txt"):
        assert name in script, f"installer must verify matcha model file {name}"
    for lib in ("libsherpa-onnx-c-api.so", "libonnxruntime.so.1"):
        assert lib in script, f"installer must verify matcha runtime library {lib}"
