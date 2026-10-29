"""Which emulator this project is allowed to talk to.

verify() runs before anything sends a tap, and refuses any device that is not
running Clash Royale. On the author's PC two MuMu instances ran: one Clash
Royale, one a Clash of Clans bot, and driving the wrong one by accident means
taps landing in somebody's village - so getting this wrong is not a crash, it
is damage. The check stays on every machine.

Identity cannot come from the hardware. Instances report the same spoofed
model (`SM_G9980`), the same 1080x1920, and the same density, because that is
what MuMu presents. It has to come from the instance itself:

    MuMuManager.exe info -v all      lists instances by index and name
    MuMuManager.exe adb -v <index>   gives that instance's adb port

Port = 16384 + 32 * index, so a fresh single-instance install (index 0) is on
127.0.0.1:16384. The adb path, serial and instance name come from
src/hastycr/machine.py: HASTYCR_ADB / HASTYCR_SERIAL / HASTYCR_INSTANCE, then
machine.json, then the defaults (`python -m hastycr.machine` shows them).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)                    # hastycr, also without `pip install -e .`

from hastycr import machine  # noqa: E402

_MACHINE = machine.get()
DEFAULT_ADB = _MACHINE.adb
DEFAULT_SERIAL = _MACHINE.serial
DEFAULT_INSTANCE = _MACHINE.instance

REQUIRED_PACKAGE = "com.supercell.clashroyale"
FORBIDDEN_PACKAGE = "com.supercell.clashofclans"


def packages(serial: str = DEFAULT_SERIAL, adb: str | Path = DEFAULT_ADB) -> set[str]:
    """Every package installed on `serial`, or an empty set if unreachable."""
    try:
        out = subprocess.run([str(adb), "-s", serial, "shell", "pm", "list", "packages"],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=20)
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip().replace("package:", "")
            for line in out.stdout.splitlines() if line.strip()}


def verify(serial: str = DEFAULT_SERIAL, adb: str | Path = DEFAULT_ADB) -> None:
    """Raise unless `serial` is the Clash Royale instance.

    Called before anything sends a tap. A wrong serial used to be silent: the
    bot would connect, fail to recognise the screen, and keep tapping - into
    Clash of Clans. Better to refuse to start than to play someone else's
    game.
    """
    found = packages(serial, adb)
    if not found:
        raise SystemExit(
            f"cannot reach {serial}. Start the emulator, or run:\n"
            f'  "{adb}" connect {serial}\n'
            f"Ports come from: MuMuManager.exe adb -v <instance index>")
    if FORBIDDEN_PACKAGE in found and REQUIRED_PACKAGE not in found:
        raise SystemExit(
            f"refusing to run: {serial} is the CLASH OF CLANS instance.\n"
            f"That device runs another bot and must not be driven from here.\n"
            f"The Clash Royale instance is {DEFAULT_SERIAL} "
            f'(MuMu instance "{DEFAULT_INSTANCE}").')
    if REQUIRED_PACKAGE not in found:
        raise SystemExit(
            f"refusing to run: {REQUIRED_PACKAGE} is not installed on {serial}.\n"
            f"Expected the Clash Royale instance at {DEFAULT_SERIAL} "
            f'("{DEFAULT_INSTANCE}").')


if __name__ == "__main__":                      # a quick check from the shell
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL
    verify(target)
    print(f"{target} is the Clash Royale instance ({DEFAULT_INSTANCE})")
