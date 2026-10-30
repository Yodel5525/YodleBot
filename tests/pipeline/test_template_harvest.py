"""V07 helpers: clock-sweep, harvest and harvest-label (scripts/vision/harvest.py).

Synthetic recordings only, no device, no OpenCV or Pillow needed:
* seven-segment digits drawn with numpy (as in test_hud.py), white on the clock
  box (dark in regulation, red in overtime), tinted for the towers;
* a clock counting down at 57 fps with an image every 5th frame (t = i / 57),
  rendered on demand by a loader, so no image touches the disk except in the
  open_recording and CLI tests (.npy images there);
* card art from Walsh-Hadamard rows (exactly orthogonal, so different cards
  score NCC 0), in colour and greyed out.
The clock changes at t = k - 0.37 s, between images, never on one.
"""
from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import harvest as HV  # noqa: E402
from scripts.vision import hud as H  # noqa: E402
from scripts.vision import record as R  # noqa: E402
from scripts.vision import templates as TP  # noqa: E402
from scripts.vision import types as VT  # noqa: E402

SIZE = (360, 640)                       # a 9:16 test frame (width, height)
BG = (20, 90, 40)
WHITE = (255, 255, 255)
BOX = (20, 30, 25)                      # the regulation clock box (dark)
RED_BOX = (200, 60, 60)                 # the overtime clock box (red)
FLOOR = (250, 245, 210)                 # arena floor by the enemy towers (owner's frames)
PINKISH = (255, 225, 235)               # the opponent's tower hit points
STONE = (130, 129, 127)                 # our tower behind our hit points
LIGHT_BLUE = (190, 225, 245)            # our tower hit points
FPS, EVERY = 57.0, 5
PHASE = 0.37                            # the clock changes at t = k - PHASE

HAND_PX = [(40 + 72 * i, 520, 104 + 72 * i, 584) for i in range(4)]
NEXT_PX = (4, 590, 36, 622)
TOWER_PX = {"tower_enemy_left": (20, 100, 120, 130), "tower_enemy_right": (240, 100, 340, 130),
            "tower_enemy_king": (130, 60, 230, 90), "tower_own_left": (20, 400, 120, 430),
            "tower_own_right": (240, 400, 340, 430), "tower_own_king": (130, 470, 230, 500)}
PX = {"clock": (250, 20, 350, 60), "elixir_digit": (60, 590, 110, 636), "next": NEXT_PX,
      **{f"hand_{i}": HAND_PX[i] for i in range(4)}, **TOWER_PX}
CLOCK_AT = (260, 25)                    # where the M:SS text is pasted: ones digit at x 293-303, rows 27-45
CARD_IDS = {f"c{i}": 100 + i for i in range(8)}
DECK = tuple(CARD_IDS)


def regions() -> H.HudRegions:
    return H.HudRegions({k: H.Rect.from_px(v, SIZE) for k, v in PX.items()}, hand_centres=[(0.5, 0.9)] * 4)


def blank(size=SIZE) -> np.ndarray:
    f = np.empty((size[1], size[0], 3), dtype=np.uint8)
    f[:] = BG
    return f


def paste(frame: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    frame[y:y + img.shape[0], x:x + img.shape[1]] = img


# -- seven-segment digits (test_hud.py's) ---------------------------------------
_SEG = {"a": (slice(0, 1), slice(0, 5)), "b": (slice(0, 5), slice(4, 5)), "c": (slice(4, 9), slice(4, 5)),
        "d": (slice(8, 9), slice(0, 5)), "e": (slice(4, 9), slice(0, 1)), "f": (slice(0, 5), slice(0, 1)),
        "g": (slice(4, 5), slice(0, 5))}
_DIGIT_SEGS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd",
               "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg"}


def seg_glyph(ch: str, cell: int = 2) -> np.ndarray:
    """A 9x5-cell seven-segment digit, or ':' as two dots, as a bool mask (18 px tall)."""
    if ch == ":":
        g = np.zeros((9, 1), dtype=bool)
        g[2, 0] = g[6, 0] = True
    else:
        g = np.zeros((9, 5), dtype=bool)
        for s in _DIGIT_SEGS[ch]:
            g[_SEG[s]] = True
    return np.kron(g, np.ones((cell, cell), dtype=bool)).astype(bool)


def tight(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.asarray(mask)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def compose(text: str, fg=WHITE, bg=BG, gap: int = 3, pad: int = 2) -> np.ndarray:
    """Glyphs in `fg` on `bg`, top-aligned, `gap` columns apart, `pad` around."""
    gs = [seg_glyph(c) for c in text]
    h = max(g.shape[0] for g in gs)
    w = sum(g.shape[1] for g in gs) + gap * (len(gs) - 1) + 2 * pad
    mask = np.zeros((h + 2 * pad, w), dtype=bool)
    x = pad
    for g in gs:
        mask[pad:pad + g.shape[0], x:x + g.shape[1]] |= g
        x += g.shape[1] + gap
    out = np.empty(mask.shape + (3,), dtype=np.uint8)
    out[:] = bg
    out[mask] = fg
    return out


def which_char(crop: np.ndarray, style: TP.GlyphStyle = TP.DEFAULT_GLYPH_STYLE) -> str | None:
    """The seven-segment character a glyph crop shows, by exact mask equality."""
    m = tight(style.mask(crop))
    for ch in "0123456789:":
        if np.array_equal(m, tight(seg_glyph(ch))):
            return ch
    return None


def mmss(s: int) -> str:
    return f"{s // 60}:{s % 60:02d}"


def draw_clock(f: np.ndarray, text: str | None, red: bool = False, overlay: bool = False) -> None:
    x0, y0, x1, y1 = PX["clock"]
    box = RED_BOX if red else BOX
    f[y0:y1, x0:x1] = box
    if text is not None:
        paste(f, compose(text, bg=box), *CLOCK_AT)
    if overlay:
        f[27:45, 293:303] = WHITE                       # something passing over the ones digit


class Sim:
    """A recording rendered on demand: frame i at t = i / 57 s, an image every
    5th frame. draw(frame, t) paints the HUD; `bad` maps image numbers to
    'raise' (unreadable), 'size' (not 9:16) or 'grey' (a 2-D array)."""

    def __init__(self, seconds: float, draw, size=SIZE):
        self.n = int(round(seconds * FPS))
        self.draw = draw
        self.size = size
        self.bad: dict[int, str] = {}

    def frame(self, i: int) -> np.ndarray:
        f = blank(self.size)
        self.draw(f, i / FPS)
        return f

    def load(self, path) -> np.ndarray:
        i = int(Path(path).stem)
        kind = self.bad.get(i)
        if kind == "raise":
            raise OSError("truncated jpeg")
        if kind == "size":
            return np.zeros((400, 400, 3), dtype=np.uint8)
        if kind == "grey":
            return np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        return self.frame(i)

    def recording(self) -> HV.Recording:
        return HV.Recording(Path("sim"), [HV.RecImage(i, i / FPS, Path(f"sim/images/{i:08d}.npy"))
                                          for i in range(0, self.n, EVERY)])


def countdown(start: int, shown=None, red: bool = False):
    """draw(): a clock from `start` seconds down, one value per second;
    shown(v, t) may change the value (None: blank) or return (text, overlay)."""
    def draw(f, t):
        v = start - math.floor(t + PHASE)
        out = shown(v, t) if shown is not None else v
        overlay = False
        if isinstance(out, tuple):
            out, overlay = out
        draw_clock(f, None if out is None else mmss(out), red, overlay)
    return draw


def truth(sim: Sim, rec: HV.Recording, seg: HV.ClockSegment, start: int, shown=None) -> int:
    t = rec.images[seg.mid].t
    v = start - math.floor(t + PHASE)
    out = shown(v, t) if shown is not None else v
    return out[0] if isinstance(out, tuple) else out


def sweep(sim: Sim, anchors, store=None, **kw):
    lines: list[str] = []
    res = HV.clock_sweep(sim.recording(), regions(), store if store is not None else TP.TemplateStore.memory(),
                         anchors, load=sim.load, log=lines.append, **kw)
    return res, lines


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def test_glyph_change_display_change_and_clock_text():
    g8, g9 = tight(seg_glyph("8")), tight(seg_glyph("9"))
    assert HV.glyph_change(g8, g8) == 0.0
    assert HV.glyph_change(g8, np.pad(g8, ((1, 0), (1, 0)))) == 0.0      # a pixel off: the same glyph
    assert HV.glyph_change(g8, g9) == pytest.approx(0.125)               # one segment of 7: still a change
    assert HV.glyph_change(g8, g9) >= HV.CHANGE_FRAC
    a = HV.clock_obs(compose("1:48", bg=BOX))
    assert a is not None and len(a.glyphs) == 4 and not a.red
    assert HV.display_change(a, HV.clock_obs(compose("1:48", bg=BOX))) == 0.0
    assert HV.display_change(a, HV.clock_obs(compose("1:49", bg=BOX))) >= HV.CHANGE_FRAC
    assert HV.display_change(a, HV.clock_obs(compose("148", bg=BOX))) == math.inf
    red = HV.clock_obs(compose("1:48", bg=RED_BOX))
    assert red.red and HV.display_change(a, red) == math.inf
    assert HV.clock_obs(np.full((20, 40, 3), 30, np.uint8)) is None       # no glyph
    assert [HV.clock_text(s, n) for s, n in [(108, 4), (9, 4), (9, 2), (9, 1), (45, 2), (120, 4)]] == \
        ["1:48", "0:09", "09", "9", "45", "2:00"]
    assert HV.clock_text(108, 2) is None and HV.clock_text(-1, 4) is None


def test_split_wide_glyphs_cuts_touching_digits_at_the_waist():
    zero = seg_glyph("0")                                   # 18 x 10
    mask = np.zeros((18, 21), dtype=bool)
    mask[:, :10] = zero
    mask[:, 11:] = zero
    mask[8, 10] = True                                      # the two zeros touch: one column run
    boxes = TP.segment_glyphs(mask)
    assert boxes == [(0, 0, 21, 18)]
    assert HV.split_wide_glyphs(mask, boxes) == [(0, 0, 10, 18), (11, 0, 21, 18)]
    eight = np.zeros((18, 10), dtype=bool) | seg_glyph("8")
    assert HV.split_wide_glyphs(eight, [(0, 0, 10, 18)]) == [(0, 0, 10, 18)]          # not wider than tall
    bar = np.ones((10, 30), dtype=bool)
    assert HV.split_wide_glyphs(bar, [(0, 0, 30, 10)]) == [(0, 0, 30, 10)]            # wide, but no waist


def _seg(t_prev, t0, t1, linked):
    return HV.ClockSegment([0], HV.ClockObs((), ()), linked, t_prev, t0, t1)


def test_seconds_between_counts_changes_and_gaps():
    a = _seg(0.0, 0.1, 0.9, True)
    assert HV.seconds_between([a, _seg(0.9, 1.0, 1.9, True)], 1) == 1          # a change between images
    # after a gap: the value appeared in [2.8, 3.3], a's in [0.0, 0.1] -> 2.7..3.3 s -> 3 values on
    assert HV.seconds_between([a, _seg(3.2, 3.3, 3.9, False)], 1) == 3
    # a short segment after a gap: 1.85..3.0 s fits both 2 and 3 -> the count is lost
    assert HV.seconds_between([a, _seg(2.9, 3.0, 3.05, False)], 1) is None
    # consecutive images 2.15 s apart (a capture stall): not one change but the times' count,
    # the value appeared in [3.9 - 1.1, 3.05] -> 2.7..3.05 s after a's -> 3
    assert HV.seconds_between([a, _seg(0.9, 3.05, 3.9, True)], 1) == 3


def _ones_seg(ch: str) -> HV.ClockSegment:
    g = tight(seg_glyph(ch))
    return HV.ClockSegment([0], HV.ClockObs(((0, 0, g.shape[1], g.shape[0]),), (g,)), True, 0.0, 0.0, 0.0)


def test_count_check_needs_the_same_last_digit_ten_seconds_away():
    # segment 0 is labelled 0:38; the ones digits 10, 9 and 11 s nearer the anchor
    segs = [_ones_seg("8"), _ones_seg("2"), _ones_seg("1"), _ones_seg("7"), _ones_seg("8")]
    labels = {0: 38, 1: 48, 2: 47, 3: 49}
    # '2' is the closest of the three (0.29 vs 2.0 and 1.08) but not the same glyph: a 2 s slip
    assert HV.glyph_change(segs[0].ref.glyphs[-1], segs[1].ref.glyphs[-1]) == pytest.approx(24 / 84)
    assert HV._ones_check(segs, labels, 0, +1) is False
    assert HV._ones_check(segs, {0: 38, 4: 48, 2: 47, 3: 49}, 0, +1) is True       # an '8' 10 s away
    assert HV._ones_check(segs, {0: 38, 1: 48, 2: 47}, 0, +1) is None               # 0:49 not labelled


def test_open_recording_reads_times_and_skips_bad_lines(tmp_path):
    root = tmp_path / "rec"
    with R.VisionRecorder(root, "s1", image_every=5, image_format="npy", wall_clock=lambda: 1000.0) as rec:
        for f in range(12):
            rgb = np.zeros((384, 216, 3), dtype=np.uint8) if f % 5 == 0 else None
            rec.write(VT.VisionFrame(t_capture=10.0 + f / 57.0), rgb)
    session = root / "s1"
    (session / "images" / "00000005.npy").unlink()                       # pruned by the disk budget
    with open(session / "frames.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"kind": "frame", "i": 99, "ima')                      # cut short by a crash
    r = HV.open_recording(session)
    assert [im.i for im in r.images] == [0, 10]
    assert r.images[1].t == pytest.approx(10.0 + 10 / 57.0, abs=1e-4)
    assert any("missing" in n for n in r.notes) and any("unreadable line" in n for n in r.notes)
    assert r.find("00000010.npy") == 1 and r.find(str(session / "images" / "00000010.npy")) == 1
    assert r.find("10") == 1 and r.find("7") is None
    assert HV.open_recording(root).session == session                    # a root: its newest session
    assert HV.open_recording(session / "frames.jsonl").session == session
    (session / "frames.jsonl").unlink()                                  # images only: times at --fps
    r = HV.open_recording(session, fps=50.0)
    assert [(im.i, im.t) for im in r.images] == [(0, 0.0), (10, 0.2)]
    with pytest.raises(FileNotFoundError):
        HV.open_recording(tmp_path / "nothing")


# ---------------------------------------------------------------------------
# clock-sweep
# ---------------------------------------------------------------------------
def _store_with(chars: str) -> TP.TemplateStore:
    store = TP.TemplateStore.memory()
    TP.add_digit_templates(store, "clock", compose(chars), chars)
    return store


def test_clock_sweep_labels_every_second_and_stores_the_missing_digits():
    # 1:16 down to 0:56 over 20 s; the owner has 1-7 and ':' already (list: missing 0 8 9)
    sim = Sim(20.0, countdown(76))
    rec = sim.recording()
    assert rec.images[99].i == 495 and mmss(76 - math.floor(495 / FPS + PHASE)) == "1:07"
    store = _store_with("1:234567")
    dry, _ = sweep(sim, [("00000495.npy", "1:07")], store, dry_run=True)
    assert dry.ok and len(dry.stored) == 5 and store.labels(TP.digit_kind("clock")) == list("1234567:")
    res, lines = sweep(sim, [("495", "1:07")], store)
    assert res.ok and len(res.segments) == 21 and res.skipped == []
    for seg in res.segments:
        assert seg.label == truth(sim, rec, seg, 76) and seg.phase == "regulation"
    assert sorted((c, p) for c, p, _, _ in res.stored) == [
        ("0", "minutes"), ("0", "ones"), ("0", "tens"), ("8", "ones"), ("9", "ones")]
    # nearest the anchor: 0:59 (minutes), 1:10 (ones), the anchor 1:07 itself (tens), 1:08, 1:09
    texts = {(c, p): img for c, p, img, _ in res.stored}
    assert mmss(76 - math.floor(int(texts[("0", "tens")][:8]) / FPS + PHASE)) == "1:07"
    assert mmss(76 - math.floor(int(texts[("0", "minutes")][:8]) / FPS + PHASE)) == "0:59"
    assert store.labels(TP.digit_kind("clock")) == list("0123456789:")
    ocr = TP.DigitOCR(store, "clock")
    for text in ("0:58", "1:09", "1:00", "0:08", "2:59"):
        assert ocr.read(compose(text)).text == text
    assert any("labelled 1:16 down to 0:56 (21 segments)" in line for line in lines)
    again, _ = sweep(sim, [("495", "1:07")], store)                  # nothing missing any more
    assert again.ok and again.stored == [] and again.not_found == []


def test_clock_sweep_survives_interruptions_gaps_and_bad_frames():
    def shown(v, t):
        if 2.95 <= t < 3.1:
            return v, True              # images 170, 175: something over the ones digit
        if 5.5 <= t < 5.9:
            return None                 # images 315-335: no digits (the change at 5.63 s is hidden)
        return v
    sim = Sim(20.0, countdown(76, shown))
    sim.bad = {600: "raise", 800: "size", 1000: "grey"}
    rec = sim.recording()
    res, lines = sweep(sim, [("495", "1:07")])
    assert res.ok and (res.unreadable, res.interruptions, res.no_clock) == (3, 2, 5)
    assert len(res.segments) == 21 and res.skipped == []
    for seg in res.segments:
        assert seg.label == truth(sim, rec, seg, 76, shown)
    assert any("3 unreadable or odd-sized images skipped" in line for line in lines)
    ocr = TP.DigitOCR(TP.TemplateStore.memory(), "clock")
    assert not ocr.ready
    store = TP.TemplateStore.memory()
    HV.clock_sweep(rec, regions(), store, [("495", "1:07")], all_chars=True, load=sim.load, log=lambda s: None)
    ocr = TP.DigitOCR(store, "clock")
    for text in ("1:16", "0:57", "1:43", "0:20"):
        assert ocr.read(compose(text)).text == text


def test_clock_sweep_counts_through_a_capture_stall_and_a_glitch_at_a_change():
    def shown(v, t):
        if 10.65 <= t < 10.75:
            return v, True              # image 610, the first of 1:05, caught mid-update
        return v
    sim = Sim(20.0, countdown(76, shown))
    rec = sim.recording()
    # the capture stalls 2.2 s: images 265-380 never written, 1:11 and 1:10 unseen,
    # 1:12 (image 260) is followed directly by 1:09 (image 385)
    rec.images = [im for im in rec.images if not 4.6 < im.t < 6.7]
    store = TP.TemplateStore.memory()
    lines: list[str] = []
    res = HV.clock_sweep(rec, regions(), store, [("00000120.npy", "1:14")], load=sim.load, log=lines.append)
    assert res.ok and res.interruptions == 1 and res.skipped == []      # the glitch image, dropped
    assert len(res.segments) == 19                                       # 1:16 .. 0:56 without 1:11, 1:10
    for seg in res.segments:
        assert seg.label == truth(sim, rec, seg, 76, shown)              # 1:09 after 1:12, 1:05 after 1:06
    assert res.runs[0].stops == {+1: "the recording ends", -1: "the recording starts"}
    for e in store.entries(TP.digit_kind("clock")):
        assert which_char(e.image) == e.label


def test_clock_sweep_stops_at_a_skipped_second_and_stores_nothing_wrong():
    # 1:35 down; when 1:11 is due the display jumps to 1:10 (a lag catch-up), then counts on
    def shown(v, t):
        return v if v >= 72 else v - 1
    sim = Sim(30.0, countdown(95, shown))
    rec = sim.recording()
    store = TP.TemplateStore.memory()
    res, lines = sweep(sim, [("00000570.npy", "1:25")], store, all_chars=True)
    assert res.ok
    labelled = [s for s in res.segments if s.label is not None]
    assert len(labelled) == 24                                           # 1:35 .. 1:12
    for seg in res.segments:
        real = truth(sim, rec, seg, 95, shown)
        assert seg.label == (real if real >= 72 else None)
    run = res.runs[0]
    assert run.stops[+1].startswith("count check failed at 1:11")
    assert len(res.skipped) == 7 and all("count check failed" in why for _, why in res.skipped)
    assert {(c, p) for c, p, _, _ in res.stored} == {
        ("1", "minutes"), (":", "colon"), ("1", "tens"), ("2", "tens"), ("3", "tens"),
        *((d, "ones") for d in "0123456789")}
    for e in store.entries(TP.digit_kind("clock")):                      # every template is what it says
        assert which_char(e.image) == e.label
    assert res.not_found == []


def overtime_draw(f, t):
    if t < 6.63:
        draw_clock(f, mmss(6 - math.floor(t + PHASE)))                  # 0:06 .. 0:00 (regulation)
    elif t < 8.0:
        draw_clock(f, None)                                              # the overtime banner
    else:
        draw_clock(f, mmss(120 - math.floor(t - 8.0)), red=True)         # 2:00 .. 1:53 (overtime)


def overtime_truth(t: float) -> tuple[int, str]:
    if t < 6.63:
        return 6 - math.floor(t + PHASE), "regulation"
    return 120 - math.floor(t - 8.0), "overtime"


def test_clock_sweep_stops_at_the_regulation_end_unless_given_an_overtime_anchor():
    sim = Sim(16.0, overtime_draw)
    rec = sim.recording()
    reg, ot = ("00000120.npy", "0:04"), ("00000760.npy", "1:55")
    res, lines = sweep(sim, [reg])
    assert len(res.segments) == 15
    for seg in res.segments:
        sec, phase = overtime_truth(rec.images[seg.mid].t)
        assert (seg.label, seg.phase) == ((sec, phase) if phase == "regulation" else (None, ""))
    assert res.runs[0].stops[+1] == "the clock box changes colour (regulation / overtime)"
    res, lines = sweep(sim, [reg], until_overtime=True)
    assert any("--until-overtime needs a second --anchor" in line for line in lines)
    assert sum(s.label is not None for s in res.segments) == 7
    res, lines = sweep(sim, [reg, ot])                                   # no flag: the red anchor is skipped
    assert any("is an overtime frame" in line for line in lines) and len(res.runs) == 1
    store = TP.TemplateStore.memory()
    res, lines = sweep(sim, [reg, ot], store, until_overtime=True)
    assert res.ok and len(res.runs) == 2 and res.skipped == []
    for seg in res.segments:
        assert (seg.label, seg.phase) == overtime_truth(rec.images[seg.mid].t)
    assert res.runs[1].stops[-1] == "the clock box changes colour (regulation / overtime)"
    assert res.not_found == []
    ocr = TP.DigitOCR(store, "clock")
    for text in ("1:53", "2:00", "0:05", "1:59"):
        assert ocr.read(compose(text)).text == text


def test_clock_sweep_refuses_bad_anchors_without_raising():
    sim = Sim(16.0, overtime_draw)
    for anchor, text, why in [("00009999.npy", "0:04", "is not an image"),
                              ("00000400.npy", "0:00", "shows no digits"),
                              ("00000120.npy", "4", "cannot show '4'"),
                              ("00000120.npy", "0:4x", "is not a clock reading")]:
        res, lines = sweep(sim, [(anchor, text)])
        assert not res.ok and any(why in line for line in lines), (anchor, text, lines)
    sim.bad = {120: "raise"}
    res, lines = sweep(sim, [("00000120.npy", "0:04")])
    assert not res.ok and any("unreadable" in line for line in lines)


# ---------------------------------------------------------------------------
# harvest and harvest-label
# ---------------------------------------------------------------------------
TOWER_HP = [  # (from t, {region: hit points}); kings show a number only once damaged
    (0.0, {"tower_enemy_left": "3052", "tower_enemy_right": "3052", "tower_own_left": "3052",
           "tower_own_right": "3052"}),
    (1.0, {"tower_enemy_left": "2101", "tower_enemy_right": "1315", "tower_own_left": "2731",
           "tower_own_right": "544", "tower_enemy_king": "4824"}),
    (2.0, {"tower_enemy_left": "1876", "tower_enemy_right": "960", "tower_own_left": "2731",
           "tower_own_right": "544", "tower_enemy_king": "4321", "tower_own_king": "4790"}),
]


def draw_towers(f, t):
    shown = [hp for start, hp in TOWER_HP if t >= start][-1]
    for name, (x0, y0, x1, y1) in TOWER_PX.items():
        enemy = "enemy" in name
        bg, fg = (FLOOR, PINKISH) if enemy else (STONE, LIGHT_BLUE)
        f[y0:y1, x0:x1] = bg
        if name in shown:
            paste(f, compose(shown[name], fg=fg, bg=bg), x0 + 10, y0 + 4)


def _identify(sheet_json: Path, what) -> dict[int, str]:
    doc = json.loads(sheet_json.read_text(encoding="utf-8"))
    out = {}
    with np.load(sheet_json.with_suffix(".npz")) as z:
        for c in doc["clusters"]:
            out[c["id"]] = what(z[f"c{c['id']}_r0"])
    return out


def test_harvest_tower_digits_cluster_by_digit_and_label_into_the_store(tmp_path):
    sim = Sim(3.0, draw_towers)
    sim.bad = {50: "raise", 100: "size"}
    res = HV.harvest(sim.recording(), "tower", regions(), min_ncc=0.95, load=sim.load)
    assert res.kind == "digits/tower" and res.unreadable == 2 and res.read == res.images - 2
    assert len(res.clusters) == 10 and res.left_out == (0, 0)            # one cluster per digit
    assert sum(c.size for c in res.clusters) == res.samples
    style = TP.TOWER_GLYPH_STYLE
    for c in res.clusters:                                               # pure: every member is one digit
        chars = {which_char(m.crop, style) for m in c.members}
        assert len(chars) == 1 and None not in chars and c.scores[0] == pytest.approx(1.0)
    sheet, js, npz = HV.write_harvest(res, tmp_path / "out" / "harvest_tower.png")
    assert sheet.exists() and js.exists() and npz.exists() and sheet.suffix in (".png", ".npy")
    doc = json.loads(js.read_text(encoding="utf-8"))
    assert doc["what"] == "tower" and [c["id"] for c in doc["clusters"]] == list(range(10))
    assert [c["size"] for c in doc["clusters"]] == sorted((c["size"] for c in doc["clusters"]), reverse=True)
    digits = _identify(js, lambda crop: which_char(crop, style))
    assert sorted(digits.values()) == list("0123456789")
    store = TP.TemplateStore(tmp_path / "tpl")
    mapping = ",".join(f"{cid}={d}" for cid, d in sorted(digits.items()) if d != "6")    # leave one out
    out = HV.harvest_label(js, mapping, store, log=lambda s: None)
    assert len(out) == 18                                                # 9 clusters x 2 crops
    assert store.labels(TP.digit_kind("tower")) == list("012345789")
    again = HV.harvest_label(js, mapping, store, log=lambda s: None)     # stored once only
    assert again == [] and len(store.entries(TP.digit_kind("tower"))) == 18
    HV.harvest_label(js, {cid: "6" for cid, d in digits.items() if d == "6"}, store, log=lambda s: None)
    ocr = TP.DigitOCR(store, "tower")
    assert ocr.read(compose("1876", fg=PINKISH, bg=FLOOR)).text == "1876"
    assert ocr.read(compose("4790", fg=LIGHT_BLUE, bg=STONE)).text == "4790"


def test_harvest_elixir_reads_ten_as_two_glyphs(tmp_path):
    def draw(f, t):
        x0, y0, x1, y1 = PX["elixir_digit"]
        text = "10" if t < 1.0 else "7" if t < 2.0 else "0"
        paste(f, compose(text), x0 + 10, y0 + 10)
    sim = Sim(3.0, draw)
    res = HV.harvest(sim.recording(), "elixir", regions(), min_ncc=0.95, load=sim.load)
    found = {which_char(c.members[0].crop): c.size for c in res.clusters}
    n10 = sum(1 for im in sim.recording().images if im.t < 1.0)
    n0 = sum(1 for im in sim.recording().images if im.t >= 2.0)
    assert found == {"1": n10, "0": n10 + n0, "7": len(sim.recording().images) - n10 - n0}


def card_art(i: int, greyed: bool = False, scale: int = 2) -> np.ndarray:
    """Card i: a Walsh-Hadamard row (208 / 48) in colour, or greyed out (its
    luminance x 0.6, so NCC alone cannot tell the two apart)."""
    k = np.arange(1024)
    parity = np.zeros(1024, dtype=np.int64)
    for b in range(10):
        parity ^= ((k & (33 * (i + 1))) >> b) & 1
    a = (128 + 80 * (1 - 2 * parity)).reshape(32, 32)
    rgb = np.stack([a, 255 - a, np.full_like(a, 128)], axis=2).astype(np.float64)
    if greyed:
        rgb = np.repeat((rgb @ np.array([0.299, 0.587, 0.114]) * 0.6)[:, :, None], 3, axis=2)
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return np.kron(rgb, np.ones((scale, scale, 1), dtype=np.uint8))


HAND_SCHEDULE = [  # (from t, 4 slots as (card, greyed) or None for an empty slot, next card)
    (0.0, [(0, False), (1, False), (2, False), (3, False)], 4),
    (1.0, [(0, True), (1, False), (2, True), (3, False)], 4),
    (2.0, [(5, False), (1, True), (2, False), None], 6),
]


def draw_hand(f, t):
    slots, nxt = [(s, n) for start, s, n in HAND_SCHEDULE if t >= start][-1]
    for (x0, y0, _, _), card in zip(HAND_PX, slots):
        if card is not None:
            paste(f, card_art(*card), x0, y0)
    paste(f, card_art(nxt, scale=1), NEXT_PX[0], NEXT_PX[1])


def _which_card(crop: np.ndarray, scale: int = 2) -> str | None:
    for i in range(8):
        for greyed in (False, True):
            if np.array_equal(crop, card_art(i, greyed, scale)):
                return f"greyed:c{i}" if greyed else f"c{i}"
    return None


def test_harvest_hand_keeps_greyed_apart_and_labels_variants(tmp_path):
    sim = Sim(3.0, draw_hand)
    res = HV.harvest(sim.recording(), "hand", regions(), load=sim.load)
    assert res.kind == TP.KIND_HAND and res.min_ncc == HV.CARD_MIN_NCC
    n_empty = sum(1 for im in sim.recording().images if im.t >= 2.0)
    assert res.flat == n_empty                                           # the empty slot: no cluster
    got = {_which_card(c.members[0].crop): c.greyed for c in res.clusters}
    assert got == {"c0": False, "c1": False, "c2": False, "c3": False, "c5": False,
                   "greyed:c0": True, "greyed:c1": True, "greyed:c2": True}
    _, js, _ = HV.write_harvest(res, tmp_path / "harvest_hand.png")
    cards = _identify(js, _which_card)
    store = TP.TemplateStore(tmp_path / "tpl")
    grey0 = next(cid for cid, v in cards.items() if v == "greyed:c0")
    with pytest.raises(TP.TemplateError, match="looks greyed"):
        HV.harvest_label(js, {grey0: "c0"}, store, deck=DECK, log=lambda s: None)
    assert store.kinds() == []                                           # a bad map stores nothing
    with pytest.raises(TP.TemplateError, match="not in the deck"):
        HV.harvest_label(js, {grey0: "greyed:c0"}, store, log=lambda s: None)    # default deck: Hog 2.6
    mapping = ",".join(f"{cid}={v}" for cid, v in sorted(cards.items()))
    out = HV.harvest_label(js, mapping, store, deck=DECK, per_cluster=1, log=lambda s: None)
    assert len(out) == 8
    assert store.labels(TP.KIND_HAND) == ["c0", "c1", "c2", "c3", "c5"]
    assert {e.variant for e in store.entries(TP.KIND_HAND, "c0")} == {"base", "greyed"}
    reader = H.HandReader(store, CARD_IDS, regions())
    frame = blank()
    for (x0, y0, _, _), card in zip(HAND_PX, [(5, False), (1, False), (2, True), (3, False)]):
        paste(frame, card_art(*card), x0, y0)
    assert [s.card for s in reader.read(frame).slots] == [105, 101, 102, 103]
    nxt = HV.harvest(sim.recording(), "next", regions(), load=sim.load)
    assert sorted(_which_card(c.members[0].crop, 1) for c in nxt.clusters) == ["c4", "c6"]


def test_harvest_never_raises_on_bad_frames():
    sim = Sim(1.0, draw_hand)
    sim.bad = {0: "raise", 5: "size", 10: "grey"}
    res = HV.harvest(sim.recording(), "hand", regions(), load=sim.load)
    assert res.unreadable == 3 and res.read == res.images - 3 and len(res.clusters) == 4
    assert any("truncated jpeg" in p for p in res.problems)
    with pytest.raises(TP.TemplateError):
        HV.harvest(sim.recording(), "banner", regions(), load=sim.load)
    empty = HV.harvest(HV.Recording(Path("none"), []), "tower", regions())
    assert empty.clusters == [] and empty.images == 0


def test_label_map_and_label_values():
    assert HV.parse_label_map("0=7, 1=3,2=greyed:Cannon,#4=5") == {0: "7", 1: "3", 2: "greyed:Cannon", 4: "5"}
    for bad in ("x=7", "1=", "1=2,1=3", "", "5"):
        with pytest.raises(TP.TemplateError):
            HV.parse_label_map(bad)
    assert HV.resolve_label("tower", "3") == ("digits/tower", "3", "base")
    assert HV.resolve_label("clock", ":") == ("digits/clock", ":", "base")
    assert HV.resolve_label("hand", "greyed:Cannon") == ("hand", "Cannon", "greyed")
    assert HV.resolve_label("next", "Log") == ("next", "Log", "base")
    assert HV.resolve_label("hand", "empty") == ("empty", "hand", "base")
    for what, value, why in [("tower", ":", "not one tower glyph"), ("elixir", "10", "two clusters"),
                             ("hand", "Hogrider", "not in the deck"), ("hand", "shiny:Log", "variant")]:
        with pytest.raises(TP.TemplateError, match=why):
            HV.resolve_label(what, value)


def test_contact_sheet_layout():
    m = HV.Member(np.zeros(4, np.float32), np.full((18, 10, 3), 255, np.uint8), "00000000.jpg", 0, "clock",
                  (0, 0, 10, 18))
    clusters = [HV.Cluster(i, 10 - i, i == 1, [m, m, m], [1.0, 0.9, 0.8], {"clock": 10 - i}) for i in range(9)]
    sheet = HV.contact_sheet(clusters, (48, 40))
    tile_w, tile_h = 4 + 3 * (40 + 4), 5 * 2 + 2 * 4 + 48 + 4
    cols = 1280 // tile_w
    assert sheet.shape == (-(-9 // cols) * tile_h, cols * tile_w, 3) and sheet.dtype == np.uint8
    assert (sheet == 255).all(axis=2).any()                               # the labels and crops are drawn
    assert HV.text_mask("#1 n=2").shape == (10, 6 * 4 * 2)


# ---------------------------------------------------------------------------
# The CLI, on a recording on disk
# ---------------------------------------------------------------------------
SMALL = (216, 384)
SMALL_PX = {"clock": (150, 8, 214, 40), "elixir_digit": (20, 330, 60, 370)}


def test_cli_clock_sweep_harvest_and_label(tmp_path, capsys):
    regions_path = H.HudRegions({k: H.Rect.from_px(v, SMALL) for k, v in SMALL_PX.items()},
                                hand_centres=[(0.5, 0.9)] * 4).save(tmp_path / "regions.json", SMALL)
    root = tmp_path / "recordings"
    with R.VisionRecorder(root, "capture_1", image_every=EVERY, image_format="npy") as rec:
        for f in range(int(2.6 * FPS)):
            t = f / FPS
            rgb = None
            if f % EVERY == 0:
                rgb = blank(SMALL)
                x0, y0, x1, y1 = SMALL_PX["clock"]
                rgb[y0:y1, x0:x1] = BOX
                paste(rgb, compose(mmss(12 - math.floor(t + PHASE)), bg=BOX), 160, 12)
                paste(rgb, compose("7" if t < 1.3 else "10"), 30, 338)
            rec.write(VT.VisionFrame(t_capture=t), rgb)
    session = root / "capture_1"
    tpl = tmp_path / "tpl"
    common = ["--recording", str(root), "--regions", str(regions_path), "--root", str(tpl)]
    anchor = str(session / "images" / "00000060.npy")                    # 0:11
    assert TP.main(["clock-sweep", *common, "--anchor", anchor, "--text", "0:11"]) == 0
    out = capsys.readouterr().out
    assert "using the newest session capture_1" in out and "labelled 0:12 down to 0:10 (3 segments)" in out
    store = TP.TemplateStore(tpl)
    assert store.labels(TP.digit_kind("clock")) == ["0", "1", "2", ":"]
    n = len(store.entries(TP.digit_kind("clock")))
    assert n == 6                                                        # 0 x2, ':', 1 x2, 2
    assert TP.main(["clock-sweep", *common, "--anchor", anchor, "--text", "0:11", "--all"]) == 0
    assert "already stored" in capsys.readouterr().out
    store.reload()
    assert len(store.entries(TP.digit_kind("clock"))) == n
    assert TP.main(["clock-sweep", *common, "--anchor", anchor, "--text", "0:11", "--until-overtime",
                    "--anchor", anchor]) == 1                            # one --text for two anchors
    sheet = tmp_path / "sheets" / "harvest_elixir.png"
    assert TP.main(["harvest", *common, "--what", "elixir", "--out", str(sheet), "--min-ncc", "0.95"]) == 0
    js = sheet.with_suffix(".json")
    assert "wrote" in capsys.readouterr().out and js.exists()
    digits = _identify(js, which_char)
    assert sorted(digits.values()) == ["0", "1", "7"]
    mapping = ",".join(f"{cid}={d}" for cid, d in digits.items())
    assert TP.main(["harvest-label", "--sheet", str(sheet), "--map", "0=10", "--root", str(tpl)]) == 1
    assert "two clusters" in capsys.readouterr().out
    assert TP.main(["harvest-label", "--sheet", str(js), "--map", mapping, "--root", str(tpl)]) == 0
    capsys.readouterr()
    assert TP.main(["list", "--root", str(tpl)]) == 0
    out = capsys.readouterr().out
    assert "digits/elixir: missing 2 3 4 5 6 8 9" in out and "digits/clock: missing 3 4 5 6 7 8 9" in out


def test_harvest_module_imports_light_and_never_reads_memory():
    tree = ast.parse((ROOT / "scripts" / "vision" / "harvest.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        assert not any("memread" in m or "gamestate" in m or "hcr_reader" in m for m in mods), mods
    code = ("import sys\nimport scripts.vision.harvest\n"
            "heavy = ('torch', 'cv2', 'PIL', 'ultralytics', 'scripts.gamestate.memread', 'royalesim')\n"
            "print(','.join(m for m in heavy if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""
    r = subprocess.run([sys.executable, "-m", "scripts.vision.templates", "harvest", "--help"], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "--max-clusters" in r.stdout, r.stderr
