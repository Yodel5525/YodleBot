"""Between-battle screen handling, ported from the old bot (scripts/cr_bot.py).

FRIENDLY BATTLES ONLY (hastycr.machine.FRIENDLY_ONLY). A person starts every
battle, so this never presses the lobby Battle button and never taps the end
screen's button: Navigator(queue=True) raises FriendlyOnlyError, and there is
no option that would queue. What is left is recovery on unrecognised screens:

  lobby          BuildABot's screen hash says "lobby" -> nothing (wait)
  end_of_game /  BuildABot's screen hash -> nothing; the person dismisses it.
  bypass_end     (Its click point is BuildABot's ladder end screen, unverified
                 on a friendly's.)
  unknown        reward_screen (empty top band = chest / card reveal) -> tap
                 to advance; connection_lost -> RELOAD; after 15 s a popup's
                 close button (popup_guard); after 75 s relaunch Clash Royale

Each of those tests is the old bot's, measured there; none is re-guessed
here. The colour-only checks this replaces mistook the gold coin bag on the
win screen for the Battle button (2026-10-10).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import connection_lost  # noqa: E402
import emulator  # noqa: E402
import popup_guard  # noqa: E402
import reward_screen  # noqa: E402
from hastycr import machine  # noqa: E402

ADB = str(emulator.DEFAULT_ADB)                 # HASTYCR_ADB / machine.json / PATH / MuMu
SCREEN_DIR = ROOT / "vendor" / "ClashRoyaleBuildABot" / "clashroyalebuildabot" / "images" / "screen"
# BuildABot Screens (720x1280 regions and click points), copied as data. The click
# points are kept as data only: nothing here taps them (friendly battles only).
SCREENS = {
    "in_game": ((148, 1254, 163, 1274), None),
    "lobby": ((424, 126, 506, 181), (360, 1000)),
    "end_of_game": ((379, 1122, 560, 1201), (243, 1158)),
    "bypass_end_of_game": ((269, 1123, 451, 1201), (360, 1161)),
}
UNKNOWN_POPUP_SECONDS = 15                      # cr_bot.py
UNKNOWN_RECOVER_SECONDS = 75
UNKNOWN_RECOVER_COOLDOWN = 150


def _hash(img):
    return np.asarray(img.resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32).flatten()


_REF: dict | None = None


def _references():
    """BuildABot's reference crops, loaded on first use (vendor/ is installed separately)."""
    global _REF
    if _REF is None:
        _REF = {name: _hash(Image.open(SCREEN_DIR / f"{name}.jpg")) for name in SCREENS}
    return _REF


def classify(small):
    """BuildABot's ScreenDetector.run: nearest reference crop under threshold 30."""
    refs = _references()
    best, best_diff = "unknown", 30.0
    w, h = small.size
    for name, (ltrb, _) in SCREENS.items():
        box = (int(ltrb[0] * w / 720), int(ltrb[1] * h / 1280),
               int(ltrb[2] * w / 720), int(ltrb[3] * h / 1280))
        diff = float(np.mean(np.abs(_hash(small.crop(box)) - refs[name])))
        if diff < best_diff:
            best, best_diff = name, diff
    return best


def capture(serial):
    """Raw screencap (header, then RGBA). The PNG form came back truncated
    over exec-out on 2026-10-10 ("image file is truncated")."""
    for _ in range(3):
        raw = subprocess.run([ADB, "-s", serial, "exec-out", "screencap"],
                             capture_output=True, timeout=15).stdout
        if len(raw) < 16:
            continue
        w, h = (int(v) for v in np.frombuffer(raw[:8], "<u4"))
        hdr = len(raw) - w * h * 4
        if 0 <= hdr <= 16:
            break
    else:
        raise OSError("screencap failed 3 times")
    rgba = np.frombuffer(raw[hdr:], np.uint8).reshape(h, w, 4)
    full = Image.fromarray(rgba[:, :, :3].copy(), "RGB")
    if full.size != (1080, 1920):
        full = full.resize((1080, 1920))
    return full, full.resize((368, 652))


def home_guard(full):
    """mumu_overnight_bot.home_guard: the orange Battle button fills its region.
    Detection only: it tells the lobby apart, nothing taps the button."""
    roi = np.asarray(full)[1380:1600, 350:730].astype(np.int16)
    r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    orange = (r > 180) & (g > 80) & (g < 210) & (b < 100) & (r > g * 1.25)
    return float(orange.mean()) > 0.35


def tap(serial, xy):
    subprocess.run([ADB, "-s", serial, "shell", "input", "tap", str(xy[0]), str(xy[1])],
                   capture_output=True, timeout=10)


class Navigator:
    """Call step() whenever no battle is being read. Returns a log line or None.

    Friendly battles only: queue=True raises FriendlyOnlyError, the lobby and the
    end screen are never tapped. `queue` stays a parameter so an old caller that
    asks for ladder fails loudly instead of silently getting something else.
    """

    def __init__(self, serial, queue=False):
        if queue:
            machine.refuse_queue("Navigator(queue=True)")
        self.serial, self.queue = serial, False
        now = time.monotonic()
        self.last_nav = self.last_reward = self.last_reload = 0.0
        self.last_popup = self.last_recover = 0.0
        self.unknown_since = now
        self.reward_taps = 0
        self.screen = None

    def step(self):
        full, small = capture(self.serial)
        screen = classify(small)
        now = time.monotonic()
        note = None
        if screen != self.screen:
            note = f"    screen {self.screen} -> {screen}"
            self.screen = screen
        if screen == "in_game":
            self.unknown_since = now
            return note
        if screen == "lobby":                   # a person starts the next friendly
            self.unknown_since, self.reward_taps = now, 0
            return note
        if screen in ("end_of_game", "bypass_end_of_game"):
            self.unknown_since = now            # a person dismisses the end screen
            return note
        # unknown
        if now - self.last_reward > 1.5 and reward_screen.is_reward_screen(full):
            self.last_reward = now
            self.reward_taps += 1
            spot = reward_screen.advance_point(self.reward_taps)
            tap(self.serial, spot)
            self.unknown_since = now
            return f"    reward screen: tap #{self.reward_taps} at {spot}"
        spot = connection_lost.find_reload(full)
        if spot is not None and now - self.last_reload > 8:
            self.last_reload = now
            tap(self.serial, spot)
            self.unknown_since = now
            return f"    connection lost: tapped RELOAD at {spot}"
        if now - self.unknown_since > UNKNOWN_POPUP_SECONDS and now - self.last_popup > 25:
            self.last_popup = now
            spot = popup_guard.find_close_button(full)
            if spot is not None:
                tap(self.serial, spot)
                return f"    popup: tapped close at {spot}"
            return "    popup: no close button found"
        if (now - self.unknown_since > UNKNOWN_RECOVER_SECONDS
                and now - self.last_recover > UNKNOWN_RECOVER_COOLDOWN):
            self.last_recover = now
            subprocess.run([ADB, "-s", self.serial, "shell", "monkey", "-p",
                            "com.supercell.clashroyale", "-c",
                            "android.intent.category.LAUNCHER", "1"],
                           capture_output=True, timeout=20)
            return "    stuck 75 s on an unknown screen: relaunched Clash Royale"
        return note
