"""板端物理键盘直读（``input_board_enable=true`` 时的附加输入源，2026-09-29）。

零依赖原生 ``/dev/input`` 读取：板上没有 python-evdev、pypi 也不通，
``input_event`` 的二进制结构用 struct 直接解。只关心 EV_KEY 的按下沿
（value==1），键码 < 0x100 天然排除鼠标/游戏杆按钮（BTN_* 从 0x110 起）。

读取线程与派发解耦：本模块只产出动作名（confirm/back/up/down），
选关/待机/局内的分相位派发在 server.py。权限或设备缺失只告警不退出
——键盘可以热插拔，服务不该因为一个附加输入源挂掉。
"""
from __future__ import annotations

import glob
import os
import select
import struct
import threading
import time
from typing import Callable, Iterable

EV_KEY = 0x01
EV_SYN = 0x00

# 与前端通则一一对应（绿=Enter、红=Esc、上下=导航；dice 规则页"重听"是 ↓）。
KEY_ENTER = 28
KEY_ESC = 1
KEY_UP = 103
KEY_DOWN = 108
KEY_ACTIONS: dict[int, str] = {
    KEY_ENTER: "confirm",
    KEY_ESC: "back",
    KEY_UP: "up",
    KEY_DOWN: "down",
}

# 64 位内核的 struct input_event：timeval(sec,usec) + type/code/value。
# riscv64/x86_64 均为 24 字节；32 位平台不同，但本项目只跑 64 位板子。
_INPUT_EVENT = struct.Struct("qqHHi")

# 局内意图表：{game_id: {state: {动作名: intent}}}，与 dice.js / rps.js 的
# onKey 逐字对齐（改前端键位必须同步这里，契约测试会钉基本形状）。
# analysis_failed 是失败页（页面相位仍渲染 analysis 视图）；shaking/stop_call
# 刻意无键——机械臂摇物理上不应被打断；game_start/shake_countdown/rehome/
# preparing/play 同样无键。
INTENT_TABLE: dict[str, dict[str, dict[str, str]]] = {
    "dice": {
        "rules": {"confirm": "confirm", "down": "repeat", "back": "back"},
        "ready": {"confirm": "start_shake", "back": "back"},
        "arm_failed": {"back": "back", "down": "retry"},
        "analysis_failed": {"confirm": "new_round", "down": "retry", "back": "back"},
        "result": {"confirm": "new_round", "back": "back"},
    },
    "rps": {
        "rules": {"confirm": "confirm", "down": "repeat", "back": "back"},
        "analysis_failed": {"confirm": "new_round", "back": "back"},
        "result": {"confirm": "new_round", "back": "back"},
    },
}


def resolve_intent(game_id: str, state: str, action: str) -> str | None:
    """局内动作 → 意图；无键位的状态返回 None（按键被忽略）。"""
    return INTENT_TABLE.get(game_id, {}).get(state, {}).get(action)


def parse_key_events(buf: bytes) -> list[tuple[int, int]]:
    """解一段设备字节流，返回其中 EV_KEY 条目的 (code, value)。

    value 语义：1=按下沿、0=释放沿、2=长按自动重复。过滤交给调用方，
    这里保持纯解析便于单测。
    """
    events: list[tuple[int, int]] = []
    size = _INPUT_EVENT.size
    for offset in range(0, len(buf) - size + 1, size):
        _, _, type_, code, value = _INPUT_EVENT.unpack_from(buf, offset)
        if type_ == EV_KEY:
            events.append((code, value))
    return events


class BoardKeyReader:
    """监听一组 /dev/input/event* 设备，把映射键的按下沿回调出去。

    ``on_action(action)`` 在读取线程里被调，必须轻量/不抛错；设备热插拔
    靠周期重扫（rescan_seconds），无权限只记状态不打断。
    """

    def __init__(
        self,
        on_action: Callable[[str], None],
        *,
        paths: Iterable[str] | None = None,
        device_glob: str = "/dev/input/event*",
        rescan_seconds: float = 5.0,
        debounce_seconds: float = 0.03,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._on_action = on_action
        self._paths = list(paths) if paths is not None else None
        self._device_glob = device_glob
        self._rescan_seconds = rescan_seconds
        self._debounce_seconds = debounce_seconds
        self._log = log or (lambda line: print(f"[board-input] {line}", flush=True))
        self._devices: dict[int, tuple[int, str]] = {}  # fd -> (fd 原始句柄, path)
        self._permission_denied = False
        self._last_emit: dict[int, float] = {}
        self.last_action: str | None = None
        self.last_key_monotonic: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- 生命周期 ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="board-input", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_devices()

    def status(self) -> dict:
        return {
            "running": self._thread is not None,
            "devices": [path for _, path in self._devices.values()],
            "permission_denied": self._permission_denied,
            "last_action": self.last_action,
            "last_key_age_seconds": (
                round(time.monotonic() - self.last_key_monotonic, 1)
                if self.last_key_monotonic is not None
                else None
            ),
        }

    # ---- 内部 ----------------------------------------------------------

    def _candidate_paths(self) -> list[str]:
        if self._paths is not None:
            return list(self._paths)
        return sorted(glob.glob(self._device_glob))

    def _scan_devices(self) -> None:
        alive_paths = {path for _, path in self._devices.values()}
        for path in self._candidate_paths():
            if path in alive_paths:
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except PermissionError:
                if not self._permission_denied:
                    self._log(f"无权限读取 {path}（需要 input 组），板端键盘输入不工作")
                self._permission_denied = True
                continue
            except OSError:
                continue  # 设备消失/被占用，下轮重扫再试
            self._devices[fd] = (fd, path)
        self._permission_denied = self._permission_denied and not self._devices

    def _close_devices(self) -> None:
        for fd, _ in self._devices.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._devices.clear()

    def _drop_device(self, fd: int) -> None:
        entry = self._devices.pop(fd, None)
        if entry is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _handle_buffer(self, buf: bytes, now: float) -> None:
        """解析一段读取到的字节并把按下沿派发出去（单测直接喂合成流）。"""
        for code, value in parse_key_events(buf):
            if value != 1:
                continue  # 0=释放 2=长按重复，都不要
            action = KEY_ACTIONS.get(code)
            if action is None:
                continue  # 鼠标键等非映射键
            last = self._last_emit.get(code)
            if last is not None and now - last < self._debounce_seconds:
                continue
            self._last_emit[code] = now
            self.last_action = action
            self.last_key_monotonic = now
            try:
                self._on_action(action)
            except Exception as exc:  # 回调绝不能带死读取线程
                self._log(f"on_action 回调异常: {exc}")

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._devices:
                self._scan_devices()
                if not self._devices:
                    self._stop.wait(self._rescan_seconds)
                    continue
            fds = list(self._devices)
            try:
                readable, _, _ = select.select(fds, [], [], self._rescan_seconds)
            except (OSError, ValueError):
                self._close_devices()
                continue
            if self._stop.is_set():
                break
            if not readable:
                self._scan_devices()  # 周期重扫：补热插拔、清死句柄
                continue
            for fd in readable:
                try:
                    buf = os.read(fd, _INPUT_EVENT.size * 16)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    self._drop_device(fd)
                    continue
                if not buf:
                    self._drop_device(fd)
                    continue
                self._handle_buffer(buf, time.monotonic())
