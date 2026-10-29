"""Capture the emulator screen as a small JPEG.

PowerShell's `>` mangles binary streams, so every agent that wants to look at
the game needs this instead of piping `adb exec-out screencap` to a file.  The
default size is deliberately small: these images are read by billed models, and
a third-scale JPEG is enough to see which cards are in hand, what is on the
field, and what the tower bars say.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
# adb and serial defaults: HASTYCR_ADB / HASTYCR_SERIAL, machine.json, then hastycr.machine's.
from emulator import DEFAULT_ADB, DEFAULT_SERIAL  # noqa: E402


def capture(adb: Path, serial: str) -> Image.Image:
    raw = subprocess.run(
        [str(adb), "-s", serial, "exec-out", "screencap", "-p"],
        capture_output=True, check=True, timeout=60,
    ).stdout
    return Image.open(io.BytesIO(raw)).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Screenshot the emulator, downscaled")
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--out", type=Path, default=Path("tmp/live/snap.jpg"))
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--quality", type=int, default=60)
    args = parser.parse_args()

    image = capture(args.adb, args.serial)
    small = image.resize((int(image.width / args.scale), int(image.height / args.scale)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    small.save(args.out, quality=args.quality)
    print(f"{args.out} {small.width}x{small.height} "
          f"{args.out.stat().st_size // 1024}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
