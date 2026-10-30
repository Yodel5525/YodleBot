"""V07: HUD readers (scripts/vision/hud.py) and the template store (scripts/vision/templates.py).

Synthetic frames only, no device, no network, no OpenCV:
* seven-segment digits drawn with numpy (always run), and digits rendered by
  Pillow's default font when Pillow is installed (skipped otherwise);
* card art made of Walsh-Hadamard rows, which are exactly orthogonal, so every
  NCC score is known in closed form (a 2:3 blend of cards E and F scores
  0.4/sqrt(0.52) = 0.5547 against E and 0.6/sqrt(0.52) = 0.8321 against F);
* hand-made elixir and tower bars.
Times are multiples of 1/8 s so every float in the expected values is exact.
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

from scripts.vision import geometry as G  # noqa: E402
from scripts.vision import hud as H  # noqa: E402
from scripts.vision import record as R  # noqa: E402
from scripts.vision import templates as TP  # noqa: E402
from scripts.vision import types as VT  # noqa: E402

SIZE = (360, 640)                       # a 9:16 test frame (width, height)
BG = (20, 90, 40)                       # no glyph, no bar colour, no elixir colour
WHITE = (255, 255, 255)
PURPLE = (200, 60, 220)                 # whole elixir (owner's frames: (204, 35, 212))
CHARGE = (50, 80, 160)                  # the elixir still charging (owner's frames: (51, 78, 157))
TRACK = (5, 55, 125)                    # the empty elixir track (owner's frames: (5, 57, 125))
DIVIDER = (90, 30, 110)                 # neither fill, charge nor track
REG, OT = VT.PHASE_REGULATION, VT.PHASE_OVERTIME

HAND_PX = [(40 + 72 * i, 520, 104 + 72 * i, 584) for i in range(4)]
NEXT_PX = (4, 590, 36, 622)
PX = {"clock": (250, 20, 350, 60), "clock_label": (260, 0, 324, 16),
      "elixir_digit": (70, 596, 100, 636), "elixir_bar": (100, 610, 350, 620), "next": NEXT_PX,
      **{f"hand_{i}": HAND_PX[i] for i in range(4)}}
CARD_IDS = {f"c{i}": 100 + i for i in range(8)}
DECK = list(range(100, 108))


def regions(extra: dict | None = None) -> H.HudRegions:
    px = dict(PX, **(extra or {}))
    return H.HudRegions({k: H.Rect.from_px(v, SIZE) for k, v in px.items()}, hand_centres=[(0.5, 0.9)] * 4)


def blank() -> np.ndarray:
    f = np.empty((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    f[:] = BG
    return f


def paste(frame: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    frame[y:y + img.shape[0], x:x + img.shape[1]] = img


def rgb3(a: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(a, dtype=np.uint8)[:, :, None], 3, axis=2)


# -- seven-segment digits -----------------------------------------------------
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


def bold_glyph(ch: str) -> np.ndarray:
    """A seven-segment digit with 3 px strokes, 27 px tall: the elixir font. Its
    digits are bold (7-8 px strokes, 27-28 rows on the owner's frames), and
    ELIXIR_GLYPH_STYLE drops fill thinner than 3 px, so seg_glyph's 2 px
    strokes are not elixir digits."""
    return seg_glyph(ch, cell=3)


def font_glyph(font: str):
    return bold_glyph if font == "elixir" else seg_glyph


def compose(text: str, glyph=seg_glyph, gap: int = 3, pad: int = 2) -> np.ndarray:
    """White glyphs on BG, top-aligned, `gap` columns apart, `pad` around."""
    gs = [glyph(c) for c in text]
    h = max(g.shape[0] for g in gs)
    w = sum(g.shape[1] for g in gs) + gap * (len(gs) - 1) + 2 * pad
    mask = np.zeros((h + 2 * pad, w), dtype=bool)
    x = pad
    for g in gs:
        mask[pad:pad + g.shape[0], x:x + g.shape[1]] |= g
        x += g.shape[1] + gap
    out = np.empty(mask.shape + (3,), dtype=np.uint8)
    out[:] = BG
    out[mask] = WHITE
    return out


def digit_store(fonts=("clock", "elixir", "tower"), glyph=None) -> TP.TemplateStore:
    """Templates of 0-9 for each font, from `glyph` or the font's (font_glyph)."""
    store = TP.TemplateStore.memory()
    for font in fonts:
        TP.add_digit_templates(store, font, compose("0123456789", glyph or font_glyph(font)), "0123456789")
    return store


# -- Walsh-Hadamard card art ----------------------------------------------------
ROWS = (33, 66, 99, 132, 165, 198, 231, 264)


def hadamard_art(row: int) -> np.ndarray:
    """Row `row` of the 1024-point Sylvester-Hadamard matrix as 32x32 art: 208 / 48."""
    k = np.arange(1024)
    bits = k & row
    parity = np.zeros(1024, dtype=np.int64)
    for b in range(10):
        parity ^= (bits >> b) & 1
    return (128 + 80 * (1 - 2 * parity)).reshape(32, 32).astype(np.uint8)


def slot_art(i: int) -> np.ndarray:
    """Card i as a 64x64 hand slot (each art pixel doubled)."""
    return rgb3(np.kron(hadamard_art(ROWS[i]), np.ones((2, 2), dtype=np.uint8)))


def next_art(i: int) -> np.ndarray:
    return rgb3(hadamard_art(ROWS[i]))


def blend_art(i: int, j: int) -> np.ndarray:
    """(2*card i + 3*card j) / 5, exact in integers: 0.4 e_i + 0.6 e_j around 128."""
    a, b = hadamard_art(ROWS[i]).astype(int), hadamard_art(ROWS[j]).astype(int)
    v = (2 * a + 3 * b) // 5
    return rgb3(np.kron(v, np.ones((2, 2), dtype=int)))


def card_store() -> TP.TemplateStore:
    store = TP.TemplateStore.memory()
    for i in range(8):
        store.add(TP.KIND_HAND, f"c{i}", slot_art(i))
    return store


def hand_frame(slots, nxt=None) -> np.ndarray:
    f = blank()
    for (x0, y0, _, _), art in zip(HAND_PX, slots):
        if art is not None:
            paste(f, art, x0, y0)
    if nxt is not None:
        paste(f, nxt, NEXT_PX[0], NEXT_PX[1])
    return f


FLAT = np.full((64, 64, 3), 100, dtype=np.uint8)
E_SCORE, F_SCORE = 0.4 / math.sqrt(0.52), 0.6 / math.sqrt(0.52)


# ---------------------------------------------------------------------------
# NCC primitives and the store
# ---------------------------------------------------------------------------
def test_area_resize_is_exact_box_averaging():
    a = np.arange(12, dtype=float).reshape(3, 4)
    assert np.array_equal(TP.area_resize(np.kron(a, np.ones((2, 2))), 4, 3), a)
    assert np.allclose(TP.area_resize(a, 4, 3), a)
    assert np.allclose(TP.area_resize(a, 2, 1), [[27 / 6, 39 / 6]])      # (0+1+4+5+8+9)/6, ...
    assert TP.area_resize(np.zeros((0, 5)), 3, 2).shape == (2, 3)


def test_signature_ncc_is_brightness_invariant_orthogonal_and_flat_is_zero():
    art = hadamard_art(ROWS[0]).astype(float)
    s = TP.signature(art)
    assert float(s @ s) == pytest.approx(1.0)
    assert float(s @ TP.signature(0.5 * art + 20)) == pytest.approx(1.0)      # greyed: linear change
    assert float(s @ TP.signature(hadamard_art(ROWS[1]))) == pytest.approx(0.0, abs=1e-12)
    assert not TP.signature(np.full((10, 10), 7.0)).any()
    m = TP.CardMatcher(card_store().entries(TP.KIND_HAND)).match(FLAT)
    assert m.label is None and m.score == pytest.approx(0.0, abs=1e-12)


def test_labels_and_variants():
    assert TP.label_dir(":") == "colon" and TP.label_dir("HogRider") == "HogRider"
    for bad in ("", "../x", "a/b", "a\\b", ".hidden"):
        with pytest.raises(TP.TemplateError):
            TP.label_dir(bad)
    assert TP.variant_of("evo_greyed_shot1") == "evo_greyed"
    assert TP.variant_of("hero_x") == "hero"
    assert TP.variant_of("evo_x") == "evo"
    assert TP.variant_of("greyed") == "greyed"
    assert TP.variant_of("evolved_x") == "base"
    assert TP.variant_of("shot1") == "base"
    with pytest.raises(TP.TemplateError):
        TP.TemplateStore.memory().add(TP.KIND_HAND, "Log", slot_art(0), "shiny")
    with pytest.raises(TP.TemplateError):
        TP.TemplateStore.memory().add("Hand/../x", "Log", slot_art(0))


def test_store_memory_and_disk_round_trip(tmp_path):
    art = slot_art(0)
    mem = TP.TemplateStore.memory()
    assert mem.add(TP.KIND_HAND, "HogRider", art, "evo") is None
    (e,) = mem.entries(TP.KIND_HAND)
    assert (e.label, e.variant, e.form, e.greyed) == ("HogRider", "evo", VT.FORM_EVO, False)

    disk = TP.TemplateStore(tmp_path)
    p = disk.add(TP.KIND_HAND, "HogRider", art, "hero_greyed", name="shot 1")
    assert p.parent == tmp_path / "hand" / "HogRider" and p.stem == "hero_greyed_shot-1"
    p2 = disk.add(TP.digit_kind("clock"), ":", art[:10, :6], name="x")
    assert p2.parent.name == "colon"
    assert disk.add(TP.KIND_HAND, "HogRider", art, "hero_greyed", name="shot 1").stem == "hero_greyed_shot-1-2"

    fresh = TP.TemplateStore(tmp_path)
    entries = fresh.entries(TP.KIND_HAND)
    assert len(entries) == 2
    e = entries[0]
    assert (e.label, e.variant, e.form, e.greyed) == ("HogRider", "hero_greyed", VT.FORM_HERO, True)
    assert np.array_equal(e.image, art)
    assert fresh.labels(TP.digit_kind("clock")) == [":"]
    assert fresh.summary() == {"digits/clock": {":": 1}, "hand": {"HogRider": 2}}
    assert fresh.errors == []


def test_store_skips_unreadable_files_and_a_missing_root(tmp_path):
    d = tmp_path / "hand" / "Log"
    d.mkdir(parents=True)
    (d / "base_bad.npy").write_bytes(b"not an array")
    (d / "notes.txt").write_text("ignored", encoding="utf-8")
    store = TP.TemplateStore(tmp_path)
    assert store.entries(TP.KIND_HAND) == []
    assert len(store.errors) == 1 and "base_bad.npy" in store.errors[0]
    empty = TP.TemplateStore(tmp_path / "nowhere")
    assert empty.entries(TP.KIND_HAND) == [] and empty.summary() == {} and empty.errors == []


# ---------------------------------------------------------------------------
# Digit OCR
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["2:59", "0:07", "1:10", "3052", "10", "0", "9876543210"])
def test_digit_ocr_reads_seven_segment_digits_exactly(text):
    r = TP.DigitOCR(digit_store(("clock",)), "clock").read(compose(text))
    assert r.text == text
    assert r.conf == pytest.approx(1.0)
    assert [g.char for g in r.glyphs] == list(text)
    assert [g.small for g in r.glyphs] == [c == ":" for c in text]


def test_digit_ocr_without_templates_or_with_ambiguous_ones_reads_nothing():
    r = TP.DigitOCR(TP.TemplateStore.memory(), "clock").read(compose("2:59"))
    assert r.text is None and "no clock digit templates" in r.reason
    assert TP.DigitOCR(digit_store(("clock",)), "tower").read(compose("12")).text is None
    assert TP.DigitOCR(digit_store(("clock",)), "clock").read(np.zeros((20, 20, 3), np.uint8)).reason == "no glyphs"
    store = digit_store(("clock",))
    zero = TP.segment_glyphs(seg_glyph("0"))
    assert zero == [(0, 0, 10, 18)]
    store.add(TP.digit_kind("clock"), "8", rgb3(seg_glyph("0").astype(np.uint8) * 255))   # a mislabelled 0
    r = TP.DigitOCR(store, "clock").read(compose("0"))
    assert r.text is None and "unclear" in r.reason


def test_add_digit_templates_checks_the_glyph_count():
    with pytest.raises(TP.TemplateError, match="found 3 glyphs"):
        TP.add_digit_templates(TP.TemplateStore.memory(), "clock", compose("259"), "2:59")
    with pytest.raises(TP.TemplateError):
        TP.add_digit_templates(TP.TemplateStore.memory(), "clock", compose("25"), "2a")
    with pytest.raises(TP.TemplateError):
        TP.add_digit_templates(TP.TemplateStore.memory(), "comic", compose("25"), "25")


FLOOR = (250, 245, 210)                 # the light arena floor by the enemy towers (owner's frames)
PINKISH = (255, 225, 235)               # the opponent's tower hit points
LIGHT_BLUE = (190, 225, 245)            # ours
STONE = (130, 129, 127)                 # our tower behind them


def tinted(text: str, colour, bg) -> np.ndarray:
    img = compose(text)
    out = np.empty_like(img)
    out[:] = bg
    out[np.all(img == WHITE, axis=2)] = colour
    return out


def test_tower_digits_read_in_their_tints_over_the_floor():
    store = TP.TemplateStore.memory()
    TP.add_digit_templates(store, "tower", tinted("0123456789", PINKISH, FLOOR), "0123456789")
    ocr = TP.DigitOCR(store, "tower")                          # TOWER_GLYPH_STYLE by default
    assert ocr.read(tinted("3052", PINKISH, FLOOR)).text == "3052"
    assert ocr.read(tinted("2731", LIGHT_BLUE, STONE)).text == "2731"
    # the white test alone takes the floor as well: one blob, no reading
    assert TP.DigitOCR(store, "tower", TP.DEFAULT_GLYPH_STYLE).read(tinted("3052", PINKISH, FLOOR)).text is None
    d = json.loads(json.dumps(TP.TOWER_GLYPH_STYLE.to_dict()))
    assert TP.GlyphStyle.from_dict(d) == TP.TOWER_GLYPH_STYLE
    assert TP.GlyphStyle.from_dict(None) == TP.DEFAULT_GLYPH_STYLE


def test_digit_specks_are_ignored():
    img = np.concatenate([compose("3052"), np.empty((22, 8, 3), dtype=np.uint8)], axis=1)
    img[:, 53:] = BG
    img[19, 56:59] = WHITE                      # 3 px in one row, clear of the glyphs
    assert len(TP.segment_glyphs(TP.DEFAULT_GLYPH_STYLE.mask(img))) == 5
    assert TP.drop_specks(TP.segment_glyphs(TP.DEFAULT_GLYPH_STYLE.mask(img))) == \
        TP.segment_glyphs(TP.DEFAULT_GLYPH_STYLE.mask(compose("3052")))
    store = TP.TemplateStore.memory()
    assert len(TP.add_digit_templates(store, "clock", img, "3052")) == 4
    assert TP.DigitOCR(digit_store(("clock",)), "clock").read(img).text == "3052"


# -- the elixir bar's top line through the elixir count -------------------------------
ELX = (66, 37)                          # the owner's elixir_digit box at 747x1328 (w, h)
ELX_AT = (16, 6)                        # where the count starts in it (x, top row)
LINE_ROW = 15                           # the elixir bar's 1 px white top line in it


def elixir_box(text: str, *, line_px: int = 1, touch: bool = False, speck: bool = True,
               thick: bool = False, line_row: int = LINE_ROW) -> tuple[np.ndarray, np.ndarray, list]:
    """A synthetic elixir_digit box shaped like the owner's frame 00000665 (elixir 9):
    `text` in bold digits from ELX_AT, 3 px apart, and the bar's white top line,
    `line_px` tall from `line_row`, from box edge to box edge on both sides of the
    digits and between them, stopping 2 px short of each glyph (its dark
    outline) or, with touch, at it. speck: a 1 px speck in the line's columns,
    8 px right of the last digit, on the digits' top row. thick: the line 3 px
    tall (around line_row) in the 3 columns right of the last digit, touching it.
    Returns the image, the digits alone (the clean image) and their boxes."""
    w, h = ELX
    digits = np.zeros((h, w), dtype=bool)
    boxes, x = [], ELX_AT[0]
    for ch in text:
        g = bold_glyph(ch)
        digits[ELX_AT[1]:ELX_AT[1] + g.shape[0], x:x + g.shape[1]] = g
        ys, xs = np.nonzero(g)
        boxes.append((x + int(xs.min()), ELX_AT[1] + int(ys.min()), x + int(xs.max()) + 1,
                      ELX_AT[1] + int(ys.max()) + 1))
        x += g.shape[1] + 3
    line = np.ones(w, dtype=bool)
    clear = 0 if touch else 2
    for x0, _, x1, _ in boxes:
        line[max(0, x0 - clear):x1 + clear] = False
    extra = np.zeros((h, w), dtype=bool)
    extra[line_row:line_row + line_px, line] = True
    end = boxes[-1][2]
    if thick:
        extra[line_row - 1:line_row + 2, end:end + 3] = True
    if speck:
        extra[ELX_AT[1], end + 8] = True
    clean = np.empty((h, w, 3), dtype=np.uint8)
    clean[:] = BG
    clean[digits] = WHITE
    img = clean.copy()
    img[extra & ~digits] = WHITE
    return img, clean, boxes


def test_thin_fill_and_bold_digits_survive_cleaning():
    m = np.zeros((9, 4), dtype=bool)
    m[0, 0] = True                              # runs of 1, 2 and 3 px in column 0
    m[2:4, 0] = True
    m[5:8, 0] = True
    m[:, 1] = True                              # a stroke
    m[4, 2:4] = True                            # a 1 px line
    want = np.zeros_like(m)
    want[0, 0] = want[2, 0] = want[3, 0] = True
    want[4, 2:4] = True
    assert np.array_equal(TP.thin_fill(m, 3), want)
    assert not TP.thin_fill(m, 1).any() and int(TP.thin_fill(m, 10).sum()) == int(m.sum())
    assert TP.thin_fill(np.zeros((0, 3), dtype=bool), 3).shape == (0, 3)
    assert np.array_equal(TP.clean_glyph_mask(m, 3), m & ~want)
    assert np.array_equal(TP.clean_glyph_mask(m, 0), m)
    for ch in TP.DIGITS:                        # 3 px strokes are kept whole
        assert np.array_equal(TP.clean_glyph_mask(bold_glyph(ch), 3), bold_glyph(ch)), ch
    sheet = compose("0123456789", bold_glyph)
    assert TP.drop_specks(TP.segment_glyphs(TP.ELIXIR_GLYPH_STYLE.mask(sheet))) == \
        TP.segment_glyphs(TP.DEFAULT_GLYPH_STYLE.mask(sheet))


def test_elixir_count_ignores_the_bar_line_and_a_speck():
    # frame 00000665 (elixir 9): the line right of the '9' and a speck above it, in its
    # columns, are one column run 10 rows tall, which the clock's style kept as a glyph
    img, clean, boxes = elixir_box("9")
    assert boxes == [(16, 6, 31, 33)]
    assert TP.drop_specks(TP.segment_glyphs(TP.DEFAULT_GLYPH_STYLE.mask(img))) == \
        [(16, 6, 31, 33), (33, 6, 66, 16)]
    with pytest.raises(TP.TemplateError, match="found 2 glyphs"):
        TP.add_digit_templates(TP.TemplateStore.memory(), "elixir", img, "9", style=TP.DEFAULT_GLYPH_STYLE)
    style = TP.font_style("elixir")
    assert style is TP.ELIXIR_GLYPH_STYLE and style.min_stroke == 3
    assert np.array_equal(style.mask(img), TP.DEFAULT_GLYPH_STYLE.mask(clean))
    store = TP.TemplateStore.memory()
    assert TP.add_digit_templates(store, "elixir", img, "9") == [None]            # `digits --text 9`
    (e,) = store.entries(TP.digit_kind("elixir"))
    assert e.label == "9" and np.array_equal(e.image, img[6:33, 16:31])
    r = TP.DigitOCR(digit_store(("elixir",)), "elixir").read(img)                 # the live reader
    assert r.text == "9" and r.conf == pytest.approx(1.0) and [g.box for g in r.glyphs] == boxes


@pytest.mark.parametrize("text, kw", [
    ("9", {"touch": True}), ("9", {"line_px": 2}), ("9", {"line_px": 2, "touch": True}),
    ("9", {"thick": True}), ("0", {"thick": True, "touch": True}),
    ("10", {"touch": True}), ("10", {"line_px": 2, "touch": True, "speck": False})])
def test_elixir_bar_line_never_joins_or_widens_a_glyph(text, kw):
    img, clean, boxes = elixir_box(text, **kw)
    # the clock's style: the line joins the glyphs into one, widens one, or is a glyph itself
    assert TP.drop_specks(TP.segment_glyphs(TP.DEFAULT_GLYPH_STYLE.mask(img))) != boxes
    mask = TP.ELIXIR_GLYPH_STYLE.mask(img)
    assert np.array_equal(mask, TP.DEFAULT_GLYPH_STYLE.mask(clean))
    assert TP.drop_specks(TP.segment_glyphs(mask)) == boxes
    for x0, y0, x1, y1 in boxes:                # a glyph's crop cleans to its part of the box
        assert np.array_equal(TP.ELIXIR_GLYPH_STYLE.mask(img[y0:y1, x0:x1]), mask[y0:y1, x0:x1])
    store = TP.TemplateStore.memory()
    assert len(TP.add_digit_templates(store, "elixir", img, text)) == len(text)
    assert [e.image.shape[:2] for e in store.entries(TP.digit_kind("elixir"))] == \
        [(y1 - y0, x1 - x0) for x0, y0, x1, y1 in boxes]
    r = TP.DigitOCR(digit_store(("elixir",)), "elixir").read(img)
    assert r.text == text and r.conf == pytest.approx(1.0)


@pytest.mark.parametrize("line_row, line_px, touch", [
    (6, 1, False), (6, 1, True), (7, 1, True), (5, 2, True), (6, 2, False)])
def test_elixir_bar_line_on_a_digits_end_stroke_keeps_the_stroke(line_row, line_px, touch):
    # the line on the rows of the '7''s 3 px top bar: the bar's 12 columns left of
    # its stem lie on the line's band, but they are more than a piece of the line
    # touching the glyph (at most min_stroke columns), and 2 px clear of the
    # line they do not touch it at all; trimming them read the '7' as a '1'
    img, clean, boxes = elixir_box("7", line_row=line_row, line_px=line_px, touch=touch)
    assert boxes == [(16, 6, 31, 33)]
    mask = TP.ELIXIR_GLYPH_STYLE.mask(img)
    assert np.array_equal(mask, TP.DEFAULT_GLYPH_STYLE.mask(clean))
    assert TP.drop_specks(TP.segment_glyphs(mask)) == boxes
    r = TP.DigitOCR(digit_store(("elixir",)), "elixir").read(img)
    assert r.text == "7" and r.conf == pytest.approx(1.0)


def test_clean_glyph_mask_keeps_a_glyph_end_the_line_does_not_touch():
    # a stroke with a 3x3 flag on its left, on the line's rows; the line stops 2 px
    # short of the glyph on both sides (its outline, as on the owner's frames)
    glyph = np.zeros((27, 30), dtype=bool)
    glyph[:, 10:13] = True
    glyph[8:11, 7:10] = True
    m = glyph.copy()
    m[9, 0:5] = m[9, 15:30] = True
    m[0, 20] = True                             # a speck
    assert np.array_equal(TP.clean_glyph_mask(m, 3), glyph)
    # a piece of the line 3 px tall on its own is dropped, wherever it is
    m[8:11, 22:27] = True
    assert np.array_equal(TP.clean_glyph_mask(m, 3), glyph)


def test_elixir_templates_stored_before_the_fix_still_match():
    # with the line touching the '9', the old splitter stored the whole box width as
    # the '9'; loading cleans the line out of it, so it matches like a clean '9'
    img, _, _ = elixir_box("9", touch=True)
    store = TP.TemplateStore.memory()
    TP.add_digit_templates(store, "elixir", img, "9", style=TP.DEFAULT_GLYPH_STYLE)
    (e,) = store.entries(TP.digit_kind("elixir"))
    assert e.image.shape[:2] == (27, 66)
    ocr = TP.DigitOCR(store, "elixir")
    assert np.allclose(ocr.refs["9"][0], TP.glyph_signature(bold_glyph("9")))
    assert ocr.read(elixir_box("9")[0]).text == "9"


def test_glyph_style_dicts_and_which_fonts_clean():
    for st in (TP.DEFAULT_GLYPH_STYLE, TP.ELIXIR_GLYPH_STYLE, TP.TOWER_GLYPH_STYLE):
        assert TP.GlyphStyle.from_dict(json.loads(json.dumps(st.to_dict()))) == st
    assert TP.ELIXIR_GLYPH_STYLE.to_dict()["min_stroke"] == 3
    before = {"min_value": 190, "max_chroma": 45, "tints": []}             # a dict from before min_stroke
    assert TP.GlyphStyle.from_dict(before) == TP.DEFAULT_GLYPH_STYLE
    with pytest.raises(ValueError):
        TP.GlyphStyle.from_dict({"min_stroke": -1})
    with pytest.raises(ValueError):
        TP.GlyphStyle(min_stroke=-1)
    # the clock and tower styles do not clean: their masks, so their readings, are as before
    assert TP.font_style("clock") is TP.DEFAULT_GLYPH_STYLE and TP.font_style("tower") is TP.TOWER_GLYPH_STYLE
    img, _, _ = elixir_box("9")
    white = np.all(img == WHITE, axis=2)
    assert np.array_equal(TP.DEFAULT_GLYPH_STYLE.mask(img), white)
    assert np.array_equal(TP.TOWER_GLYPH_STYLE.mask(img), white)             # pure white passes its first tint


def _pil_glyph_fn():
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont
    try:
        font, scale = ImageFont.load_default(size=28), 1
    except Exception:                          # Pillow < 10.1 or no FreeType: the bitmap font
        font, scale = ImageFont.load_default(), 4

    def glyph(ch: str) -> np.ndarray:
        if ch == ":":
            return seg_glyph(":")
        img = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(img).text((16, 16), ch, fill=255, font=font)
        a = np.asarray(img) >= 128
        ys, xs = np.nonzero(a)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return np.kron(a, np.ones((scale, scale), dtype=bool)).astype(bool)

    for ch in "0123456789":
        g = glyph(ch)
        if len(TP.segment_glyphs(g)) != 1 or g.shape[0] < 14:
            pytest.skip(f"this Pillow default font renders {ch!r} in pieces or too small")
    return glyph


@pytest.mark.parametrize("text", ["2:59", "1:05", "3052", "10", "0123456789"])
def test_digit_ocr_reads_pillow_rendered_digits_exactly(text):
    glyph = _pil_glyph_fn()
    store = digit_store(("tower",), glyph=glyph)
    r = TP.DigitOCR(store, "tower").read(compose(text, glyph=glyph))
    assert r.text == text


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
def test_clock_tick_formula():
    assert H.clock_tick(180) == 0
    assert H.clock_tick(179) == 20
    assert H.clock_tick(0) == 3600
    assert H.clock_tick(120, OT) == 3600
    assert H.clock_tick(119, OT) == 3620
    assert H.clock_tick(0, OT) == 6000
    assert H.clock_tick(180, REG, offset_ticks=90) == 90


@pytest.mark.parametrize("text, seconds", [("2:59", 179), ("3:00", 180), ("0:05", 5), ("259", 179),
                                           ("59", 59), ("9", 9), ("2:75", None), ("4:00", None),
                                           ("60", None), ("12:34", None), ("1:2:3", None), ("", None),
                                           (None, None), ("2:5a", None)])
def test_parse_clock_text(text, seconds):
    assert H.parse_clock_text(text) == seconds


def test_clock_sub_second_from_digit_change_timestamps():
    f = H.ClockFilter()
    reads = [(179, 0.0), (179, 0.25), (178, 0.5), (178, 0.75), (178, 1.0), (178, 1.25), (177, 1.5)]
    states = [f.update(s, t, REG) for s, t in reads]
    # first read: base 20 + 10; then continuity 35; 2:58 seen between 0.25 and 0.5 -> anchored at
    # 0.375: 40 + round(2.5) = 43, 48, 53, 58; 2:57 anchored at 1.375 -> 60 + round(2.5) = 63
    assert [s.tick for s in states] == [30, 35, 43, 48, 53, 58, 63]
    assert [s.anchored for s in states] == [False, False, True, True, True, True, True]
    assert all(s.phase == REG for s in states)
    assert states[0].source == "read"


def test_clock_rejects_a_misread_and_stays_monotonic():
    f = H.ClockFilter()
    assert f.update(179, 0.0).tick == 30
    s = f.update(159, 0.25)                    # '5' misread as '3': 20 s off the estimate
    assert (s.tick, s.source) == (35, "rejected")
    s = f.update(178, 0.5)                     # back on track; frames 0.5 s apart: no anchor
    assert (s.tick, s.anchored) == (40, False)
    assert f.update(178, 0.75).tick == 45
    s = f.update(177, 1.0)                     # anchored at 0.875: 60 + round(2.5)
    assert (s.tick, s.anchored) == (63, True)
    assert f.rejected == 1


def test_clock_resyncs_after_three_consistent_readings():
    f = H.ClockFilter()
    assert f.update(179, 0.0).tick == 30
    assert [f.update(90, t).tick for t in (0.25, 0.5)] == [35, 40]
    s = f.update(90, 0.75)
    assert (s.tick, s.phase, s.source) == (1810, REG, "resync")
    assert f.update(90, 1.0).tick == 1815


def test_clock_enters_overtime_without_a_label():
    f = H.ClockFilter()
    assert (f.update(1, 0.0).tick, f.phase) == (3590, REG)
    assert (f.update(0, 0.5).tick, f.phase) == (3600, REG)
    s = f.update(120, 1.0)                     # 2:00 fits overtime (3600), not regulation (1200)
    assert (s.tick, s.phase) == (3610, OT)
    assert (f.update(119, 1.5).tick, f.phase) == (3620, OT)


def test_clock_overtime_hint_end_and_coasting():
    f = H.ClockFilter()
    assert f.update(90, 0.0, OT).tick == 4210  # 3600 + 30 * 20 + 10
    g = H.ClockFilter()
    assert g.update(1, 0.0, OT).tick == 5990
    s = g.update(0, 1.0, OT)
    assert (s.tick, s.phase) == (6000, VT.PHASE_ENDED)
    c = H.ClockFilter()
    assert c.update(None, 0.0).tick is None
    assert c.update(179, 0.0).tick == 30
    s = c.update(None, 1.0)
    assert (s.tick, s.source) == (50, "unread")
    s = c.update(None, 2.5)                    # 2.5 s since the last reading
    assert (s.tick, s.source) == (None, "lost")


def test_clock_reader_uses_label_templates_for_the_phase():
    def label_art(kind: str) -> np.ndarray:
        y, x = np.mgrid[0:16, 0:64]
        if kind == "regulation":
            v = np.where(((y // 4) + (x // 4)) % 2 == 0, 200, 60)
        else:
            v = np.where((y // 2) % 2 == 0, 200, 60)
        return rgb3(v)

    store = digit_store(("clock",))
    store.add(TP.KIND_CLOCK_LABEL, "regulation", label_art("regulation"))
    store.add(TP.KIND_CLOCK_LABEL, "overtime", label_art("overtime"))
    reader = H.ClockReader(store, regions())
    f = blank()
    paste(f, compose("1:30"), 260, 25)
    paste(f, label_art("overtime"), 260, 0)
    r = reader.read(f)
    assert (r.seconds_left, r.phase_hint, r.text) == (90, OT, "1:30")
    paste(f, label_art("regulation"), 260, 0)
    assert reader.read(f).phase_hint == REG
    assert H.ClockReader(digit_store(("clock",)), regions()).read(f).phase_hint is None


# ---------------------------------------------------------------------------
# Elixir
# ---------------------------------------------------------------------------
def test_regen_rate_by_tick():
    assert [H.regen_ms(t) for t in (None, 0, 2399, 2400, 4799, 4800, 6000)] == \
        [2800, 2800, 2800, 1400, 1400, 930, 930]


def bar_crop(fill_px: int, charge_px: int = 0, w: int = 250, dividers=(), divider_px: int = 2) -> np.ndarray:
    c = np.empty((10, w, 3), dtype=np.uint8)
    c[:] = TRACK
    c[:, :fill_px] = PURPLE
    c[:, fill_px:fill_px + charge_px] = CHARGE
    for x in dividers:
        c[:, x:x + divider_px] = DIVIDER
    return c


def test_elixir_bar_fraction():
    # whole elixir + the charging segment, dividers inside: (160 + 20) / 250
    assert H.bar_fraction(bar_crop(160, 20, dividers=range(25, 160, 25))) == pytest.approx(0.72)
    assert H.bar_fraction(bar_crop(160)) == pytest.approx(0.64)
    over = bar_crop(160, 20)
    over[:7, 10:30] = WHITE                     # the elixir count drawn over the bar's left end
    assert H.bar_fraction(over) == pytest.approx(0.72)
    hidden = bar_crop(20)
    hidden[:7, 10:40] = WHITE                   # the fill ends under the count: unknown
    assert H.bar_fraction(hidden) is None
    assert H.bar_fraction(bar_crop(0)) == 0.0
    assert H.bar_fraction(bar_crop(250, dividers=range(25, 250, 25))) == 1.0
    off = bar_crop(0)
    off[:, 100:150] = PURPLE
    assert H.bar_fraction(off) is None          # fill not anchored at the left: something else
    loose = bar_crop(160)
    loose[:, 160:] = BG                         # no track after the fill: not the bar
    assert H.bar_fraction(loose) is None
    assert H.bar_fraction(np.zeros((10, 250, 3), dtype=np.uint8)) is None


def test_elixir_masks_take_the_owners_colours():
    def px(c):
        return np.array([[c]], dtype=np.uint8)
    for c in ((204, 35, 212), (243, 136, 242), (195, 33, 207)):          # whole elixir
        assert H.elixir_fill_mask(px(c))[0, 0] and not H.elixir_track_mask(px(c))[0, 0], c
    for c in ((51, 78, 157), (46, 84, 161), (50, 83, 160)):              # charging
        assert H.elixir_charge_mask(px(c))[0, 0] and not H.elixir_fill_mask(px(c))[0, 0], c
        assert H.elixir_lit_mask(px(c))[0, 0] and not H.elixir_track_mask(px(c))[0, 0], c
    for c in ((5, 57, 125), (3, 52, 121), (0, 19, 84)):                  # empty track
        assert H.elixir_track_mask(px(c))[0, 0] and not H.elixir_lit_mask(px(c))[0, 0], c
    for c in ((7, 71, 143), (255, 255, 255), (53, 0, 104)):              # HUD panel, a digit, the top border
        assert not (H.elixir_lit_mask(px(c))[0, 0] or H.elixir_track_mask(px(c))[0, 0]), c


def test_elixir_milli_from_digit_change_regen_spend_and_bar():
    f = H.ElixirFilter()
    out = [(s.milli, s.source) for s in (
        f.update(5, 0.0),                      # fraction unknown: conservative
        f.update(5, 0.25),
        f.update(6, 0.5),                      # 6 seen between 0.25 and 0.5: 6000 + 0.125 s * 1e6/2800
        f.update(6, 1.5),                      # + 1 s of regen
        f.update(2, 2.0),                      # a 4-cost play: 6580.36 -> 2580, the fraction kept
        f.update(2, 2.25, bar_frac=0.27),      # the bar agrees with the digit: take the bar
        f.update(2, 3.25),                     # 3057 predicted but 2 shown: ran past by 57 -> 2999
        f.update(None, 3.5),                   # nothing read: regen for <= 1.5 s
        f.update(None, 5.0))]
    assert out == [(5000, "digit"), (5000, "digit"), (6045, "digit-change"), (6402, "digit+regen"),
                   (2580, "digit+regen"), (2700, "digit+bar"), (2999, "digit+regen"), (3088, "regen"),
                   (None, "unread")]


def test_elixir_other_sources_and_double_rate():
    assert H.ElixirFilter().update(None, 0.0, bar_frac=0.64).milli == 6400
    assert H.ElixirFilter().update(10, 0.0).milli == 10000
    g = H.ElixirFilter()
    s = g.update(3, 0.0, bar_frac=0.64)         # the bar disagrees with the digit: digit wins
    assert (s.milli, s.source) == (3000, "digit")
    d = H.ElixirFilter()
    d.update(5, 0.0, tick=2400)
    assert d.update(6, 0.25, tick=2405).milli == 6089      # 6000 + 0.125 s * 1e6/1400


def test_elixir_rejects_a_one_frame_jump():
    f = H.ElixirFilter()
    out = [(s.milli, s.source) for s in (
        f.update(5, 0.0),
        f.update(8, 0.125),                    # +3 in 1/8 s: a misread; 5000 + 0.125 s of regen
        f.update(8, 0.25))]                    # shown again: believed
    assert out == [(5000, "digit"), (5045, "rejected"), (8000, "digit")]
    g = H.ElixirFilter()
    g.update(5, 0.0)
    assert g.update(8, 0.125, bar_frac=0.81).source == "digit+bar"      # the bar confirms it


def test_clock_misread_slightly_behind_does_not_fake_a_digit_change():
    f = H.ClockFilter()
    assert f.update(178, 0.0).tick == 50       # base 40 + 10
    assert f.update(179, 0.25).tick == 50      # window 20-39 is 16 behind 55: accepted, the tick holds
    s = f.update(178, 0.5)                     # 2:58 again: no change happened, so no anchor
    assert (s.tick, s.anchored) == (55, False)


# ---------------------------------------------------------------------------
# Tower HP
# ---------------------------------------------------------------------------
def test_tower_hp_filter_rules():
    f = H.TowerHpFilter()
    k = "enemy_left"
    assert f.update(k, 0.0, 0.0) == 1.0          # zero from a healthy tower: occlusion
    assert f.update(k, 0.5, 0.125) == 1.0        # a fall must hold 0.6 s
    assert f.update(k, 0.5, 0.5) == 1.0
    assert f.update(k, 0.5, 0.75) == 0.5         # 0.625 s and 3 readings
    assert f.update(k, 0.9, 1.0) == 0.5          # never rises
    assert f.update(k, 0.0, 1.25) == 0.5         # a zero from <= 0.75 starts its run
    assert f.update(k, 0.0, 3.5) == 0.5
    assert f.update(k, 0.0, 3.75) == 0.0         # 2.5 s
    assert f.update(k, 0.8, 4.0) == 0.0          # destroyed stays destroyed


def test_tower_hp_filter_matches_the_authors_filter():
    pytest.importorskip("PIL")                   # scripts/tower_hp.py imports Pillow at the top
    from scripts import tower_hp
    theirs, ours = tower_hp.TowerHpFilter(), H.TowerHpFilter()
    seq = [(0.0, 1.0), (0.25, 0.0), (0.5, 0.93), (0.625, 0.94), (1.0, 0.95), (1.25, 0.94), (1.5, 0.7),
           (1.75, 0.69), (2.0, 0.0), (2.25, 0.71), (2.5, 0.7), (3.0, 0.7), (3.25, 0.0), (4.0, 0.0),
           (6.0, 0.0), (6.25, 0.5)]
    for key_theirs, key_ours in (("enemy_left", "enemy_left"), ("ally_right", "own_right")):
        for t, raw in seq:
            assert ours.update(key_ours, raw, t) == theirs.update(key_theirs, raw, t), (key_ours, t)


def test_tower_tracker_max_from_first_frames_then_filter():
    tr = H.TowerHpTracker(confirm_max_reads=2)                   # the default is 5 now
    k = "enemy_left"
    seq = [(0.0, 3052), (0.5, 3052), (1.0, 2500), (1.25, 2500), (1.75, 2480), (2.0, 3052),
           (2.25, 2000), (2.5, 2000), (3.0, 2000), (3.25, None), (4.5, None), (5.75, None), (6.0, 2000)]
    got = [tr.update({k: hp}, t)[k] for t, hp in seq]
    assert got[0] == VT.TowerHp(3052, None)                      # max not confirmed yet
    assert [(h.hp, h.max) for h in got[1:]] == [
        (3052, 3052), (3052, 3052), (3052, 3052), (2500, 3052),   # the fall holds 0.75 s: median 2500
        (2500, 3052),                                             # a rise is a misread
        (2500, 3052), (2500, 3052), (2000, 3052),
        (2000, 3052), (2000, 3052), (0, 3052),                    # unread for 2.5 s from <= 0.75: gone
        (0, 3052)]
    assert tr.crowns() == (1, 0)
    assert tr.update({}, 7.0)["own_king"] == VT.TowerHp(None, None)
    known = H.TowerHpTracker({"own_king": 4824})
    assert known.update({}, 0.0)["own_king"] == VT.TowerHp(4824, 4824)     # no number until damaged: full
    assert known.update({"own_king": 4824}, 0.25)["own_king"] == VT.TowerHp(4824, 4824)
    quiet = H.TowerHpTracker({"own_king": 4824}, kings_full_until_read=False)
    assert quiet.update({}, 0.0)["own_king"] == VT.TowerHp(None, 4824)


def test_tower_tracker_prior_max_is_raised_not_lowered():
    k = "enemy_left"
    tr = H.TowerHpTracker(prior_max_hp=H.LEVEL11_TOWER_MAX_HP)
    assert tr.update({}, 0.0, {k: 0.5})[k] == VT.TowerHp(3052, 3052)     # bars only: a fall must hold
    assert tr.update({}, 0.75, {k: 0.5})[k] == VT.TowerHp(1526, 3052)    # 0.5 of the prior max
    assert tr.update({k: 3300}, 1.0)[k].max == 3052                      # one higher reading: not yet
    assert tr.update({k: 3300}, 1.25)[k] == VT.TowerHp(1526, 3052)       # after damage: a misread, never a max
    # (a raise, before damage, to a legal value: test_tower_tracker.py)
    low = H.TowerHpTracker(prior_max_hp=H.LEVEL11_TOWER_MAX_HP)
    assert [low.update({k: 2500}, t)[k].max for t in (0.0, 0.25)] == [3052, 3052]


def test_tower_reader_reads_digits_in_its_region():
    regs = regions({"tower_enemy_left": (200, 100, 300, 140)})
    f = blank()
    paste(f, compose("3052"), 205, 105)
    reader = H.TowerReader(digit_store(("tower",)), regs)
    reads = reader.read_digits(f)
    assert reads["enemy_left"].hp == 3052
    assert reads["own_king"].hp is None and reads["own_king"].reason == "no region"
    assert reader.read_bars(f) == {"own_left": None, "own_right": None, "enemy_left": None, "enemy_right": None}
    none = H.TowerReader(TP.TemplateStore.memory(), regs).read_digits(f)
    assert none["enemy_left"].hp is None and none["enemy_left"].reason == "no tower templates"


# ---------------------------------------------------------------------------
# Hand
# ---------------------------------------------------------------------------
def test_hand_reader_reads_slots_next_and_empty():
    reader = H.HandReader(card_store(), CARD_IDS, regions())
    r = reader.read(hand_frame([slot_art(0), FLAT, slot_art(2), slot_art(3)], next_art(4)))
    assert [s.card for s in r.slots] == [100, VT.EMPTY_CARD, 102, 103]
    assert r.slots[1].empty and not r.slots[0].empty
    assert r.slots[0].score == pytest.approx(1.0) and r.slots[0].margin == pytest.approx(1.0)
    assert r.next.card == 104                   # next box: the hand art, no next templates needed
    no_art = H.HandReader(TP.TemplateStore.memory(), CARD_IDS, regions()).read(
        hand_frame([slot_art(0)] * 4, next_art(4)))
    assert [s.card for s in no_art.slots] == [None] * 4


def slot(card, score=1.0, scores=None) -> H.SlotRead:
    if scores is None:
        scores = {card: score} if card is not None and card >= 0 else {}
    return H.SlotRead(card, score, 1.0, "base", scores)


def reading(cards, nxt) -> H.HandReading:
    return H.HandReading(tuple(slot(c) for c in cards), slot(nxt))


def test_hand_fifo_prior_corrects_one_misread():
    reader, tr = H.HandReader(card_store(), CARD_IDS, regions()), H.HandTracker(deck=DECK)
    hand = [slot_art(i) for i in range(4)]
    st = tr.update(reader.read(hand_frame(hand, next_art(4))), 0.0)
    assert (st.hand, st.next, st.cycle) == ((100, 101, 102, 103), 104, (104, None, None, None))

    hand[1] = FLAT                             # card 1 played: its slot is empty
    st = tr.update(reader.read(hand_frame(hand, next_art(4))), 1.0)
    assert (st.hand, st.next, st.cycle) == ((100, VT.EMPTY_CARD, 102, 103), 104, (104, None, None, None, 101))
    assert tr.plays == [(1.0, 1, 101)] and st.known == (True, True, True, True)

    hand[1] = blend_art(4, 5)                  # the refill (card 4) seen as 0.4 card 4 + 0.6 card 5
    raw = reader.read(hand_frame(hand, next_art(5)))
    assert raw.slots[1].card == 105            # alone, the art says card 5 ...
    assert raw.slots[1].scores[105] == pytest.approx(F_SCORE)
    assert raw.slots[1].scores[104] == pytest.approx(E_SCORE)
    st = tr.update(raw, 2.0)
    assert st.hand == (100, 104, 102, 103)     # ... the FIFO prior says the old next card, 4
    assert (st.next, st.cycle, st.corrections) == (105, (105, None, None, 101), 1)


def test_hand_duplicates_resolved_by_the_weaker_slot():
    strong = H.SlotRead(100, 0.95, 0.5, "base", {100: 0.95, 101: 0.2})
    rest = (H.SlotRead(102, 0.9, 0.9, "base", {102: 0.9}), H.SlotRead(103, 0.9, 0.9, "base", {103: 0.9}))
    nxt = slot(104)
    dup = H.SlotRead(100, 0.70, 0.3, "base", {100: 0.70, 105: 0.40, 106: 0.10})
    st = H.HandTracker(deck=DECK).update(H.HandReading((strong, dup) + rest, nxt), 0.0)
    assert st.hand == (100, 105, 102, 103)
    weak = H.SlotRead(100, 0.70, 0.3, "base", {100: 0.70, 105: 0.25})
    st = H.HandTracker(deck=DECK).update(H.HandReading((strong, weak) + rest, nxt), 0.0)
    assert st.hand == (100, VT.EMPTY_CARD, 102, 103) and st.known == (True, False, True, True)
    assert st.cycle == (104, None, None, None, None)


def test_hand_change_without_an_empty_frame():
    tr = H.HandTracker(deck=DECK)
    tr.update(reading([100, 101, 102, 103], 104), 0.0)
    st = tr.update(reading([100, 101, 104, 103], 105), 0.5)      # slot 2 already shows the next card
    assert (st.hand, st.next, st.cycle) == ((100, 101, 104, 103), 105, (105, None, None, 102))
    assert tr.plays == [(0.5, 2, 102)]
    assert tr.update(reading([106, 101, 104, 103], 105), 0.75).hand[0] == 100    # flicker: held
    tr.update(reading([100, 101, 104, 103], 105), 1.0)
    got = [tr.update(reading([106, 101, 104, 103], 105), t).hand[0] for t in (1.25, 1.5, 1.75)]
    assert got == [100, 100, 106]              # three agreeing frames: accepted as a play
    assert tr.state().cycle == (105, 107, 102, 100)      # the one unseen deck card is inferred


def test_hand_tracker_waits_for_a_card_and_holds_unsure_slots():
    tr = H.HandTracker(deck=DECK)
    st = tr.update(reading([None, VT.EMPTY_CARD, None, None], None), 0.0)
    assert not tr.started and st.hand == (VT.EMPTY_CARD,) * 4 and st.next == VT.EMPTY_CARD
    tr.update(reading([100, 101, 102, 103], 104), 0.25)
    st = tr.update(reading([None, None, 102, None], None), 0.5)
    assert (st.hand, st.next) == ((100, 101, 102, 103), 104)


def test_hand_refills_follow_the_play_order():
    """The owner's friendly (frames 1035-1066): slot 1 was played, then slot 0, and slot 1 got
    the next card. The prior expects the refills in play order, so two ambiguous refills come
    out right (in slot-index order each was forced into the other's card)."""
    tr = H.HandTracker(deck=DECK)
    tr.update(reading([100, 101, 102, 103], 104), 0.0)
    tr.cycle = [104, 105, 106, 107]                    # the order the cards come back in
    tr.update(reading([100, VT.EMPTY_CARD, 102, 103], None), 1.0)      # slot 1 played first
    st = tr.update(reading([VT.EMPTY_CARD, VT.EMPTY_CARD, 102, 103], None), 1.5)
    assert st.hand == (VT.EMPTY_CARD, VT.EMPTY_CARD, 102, 103) and tr.left == {1: 101, 0: 100}
    both = H.HandReading((slot(105, 0.6, {105: 0.6, 104: 0.5}), slot(104, 0.6, {104: 0.6, 105: 0.5}),
                          slot(102), slot(103)), slot(None))
    st = tr.update(both, 2.0)
    assert st.hand == (105, 104, 102, 103) and st.corrections == 0
    assert st.cycle == (106, 107, 101, 100) and tr.left == {}
    assert [p[1:] for p in tr.plays] == [(1, 101), (0, 100)]


def test_a_misread_refill_is_fixed_in_place_not_played():
    """The prior forced the expected card into a refill whose art was the next one; three
    frames of the right card within a second of the refill correct it in place: no phantom
    play, and the forced card goes back to the front of the cycle."""
    tr = H.HandTracker(deck=DECK)
    tr.update(reading([100, 101, 102, 103], 104), 0.0)
    tr.cycle = [104, 105, 106, 107]
    tr.update(reading([100, 101, VT.EMPTY_CARD, 103], None), 1.0)
    unsure = H.HandReading((slot(100), slot(101), slot(105, 0.6, {105: 0.6, 104: 0.5}), slot(103)),
                           slot(None))
    st = tr.update(unsure, 2.0)
    assert st.hand[2] == 104 and st.corrections == 1          # the prior took the expected card
    got = [tr.update(reading([100, 101, 105, 103], None), t).hand[2] for t in (2.1, 2.2, 2.3)]
    assert got == [104, 104, 105]           # by votes (not at once, though 105 is the next card)
    st = tr.state()
    assert (st.fixes, st.cycle, st.next) == (1, (104, 106, 107, 102), 104)
    assert tr.plays == [(1.0, 2, 102)]                         # no phantom play of 104


def test_hand_slots_empty_at_the_start_are_dealt_without_the_prior():
    """The owner's friendly dealt the hand over a second after the clock started: a slot empty
    at the start is unknown (not a play waiting for the next card) and takes the card it reads;
    reading empty again keeps it unknown. A frame with no card on which two or more cards
    would leave at once (a deal, a covered panel) is held."""
    tr = H.HandTracker(deck=DECK)
    st = tr.update(reading([100, VT.EMPTY_CARD, VT.EMPTY_CARD, VT.EMPTY_CARD], 104), 0.0)
    assert st.hand == (100, VT.EMPTY_CARD, VT.EMPTY_CARD, VT.EMPTY_CARD)
    assert st.known == (True, False, False, False) and tr.pending == {}
    dealt = H.HandReading((slot(100), slot(101, 0.6, {101: 0.6, 104: 0.5}), slot(VT.EMPTY_CARD),
                           slot(VT.EMPTY_CARD)), slot(104))
    st = tr.update(dealt, 0.25)
    assert st.hand[:2] == (100, 101) and st.corrections == 0 and tr.plays == []
    st = tr.update(reading([100, 101, 102, 103], 104), 0.5)
    assert st.hand == (100, 101, 102, 103) and st.known == (True,) * 4
    st = tr.update(reading([VT.EMPTY_CARD] * 4, None), 0.75)        # no card at all: held
    assert st.hand == (100, 101, 102, 103) and tr.plays == []
    tr.update(reading([VT.EMPTY_CARD, 101, 102, 103], 104), 1.0)     # one card leaves: a play
    assert tr.plays == [(1.0, 0, 100)]


def test_hand_slots_read_the_wrong_way_round_are_swapped():
    tr = H.HandTracker(deck=DECK)
    tr.update(reading([100, 101, 102, 103], 104), 0.0)
    got = [tr.update(reading([101, 100, 102, 103], 104), t).hand[:2] for t in (0.25, 0.5, 0.75)]
    assert got == [(100, 101), (100, 101), (101, 100)]
    assert tr.state().swaps == 1 and tr.plays == []
    assert tr.update(reading([101, 100, 102, 103], 104), 1.0).hand[:2] == (101, 100)


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
CENTRES = [(x / 1080, y / 1920) for x, y in G.DEFAULT_CARD_SLOTS_PX]


def test_default_regions_file_matches_the_builtin_table():
    loaded = H.HudRegions.load(H.DEFAULT_REGIONS_PATH, hand_centres=CENTRES)
    builtin = H.HudRegions.builtin(hand_centres=CENTRES)
    assert loaded.rects == builtin.rects
    assert loaded.sources == builtin.sources
    assert loaded.bar_margin == builtin.bar_margin
    need = {"clock", "clock_label", "elixir_digit", "elixir_bar", "next"} | \
        {f"tower_{k}" for k in VT.TOWER_KEYS} | {f"bar_{k}" for k in VT.TOWER_KEYS if "king" not in k}
    assert need <= set(builtin.rects)
    # the princess bar windows are the author's anchors +-8 px, 119 px of fill (tower_hp.py:42-49)
    assert builtin.px("bar_enemy_left", H.REF_SIZE) == (216 - 8, 275, 216 + 119 + 8, 325)
    assert builtin.px("bar_own_right", H.REF_SIZE) == (782 - 8, 1145, 782 + 119 + 8, 1215)
    # the boxes measured on the owner's 747x1328 frames come back there
    owner = (747, 1328)
    assert builtin.px("clock", owner) == (636, 31, 736, 70)
    assert builtin.px("elixir_digit", owner) == (196, 1264, 262, 1301)
    assert builtin.px("elixir_bar", owner) == (205, 1283, 721, 1310)
    assert builtin.px("tower_enemy_left", owner) == (147, 171, 236, 197)
    assert builtin.px("tower_own_left", owner) == (147, 820, 236, 848)


def test_regions_scale_and_round_trip(tmp_path):
    r = H.Rect.from_px((216, 270, 344, 330), H.REF_SIZE)
    assert r.px((1080, 1920)) == (216, 270, 344, 330)
    assert r.px((540, 960)) == (108, 135, 172, 165)
    with pytest.raises(ValueError):
        H.Rect(0.5, 0.1, 0.4, 0.2)
    builtin = H.HudRegions.builtin(hand_centres=CENTRES)
    p = builtin.save(tmp_path / "r.json")
    again = H.HudRegions.load(p, hand_centres=CENTRES)
    assert again.rects == builtin.rects and again.sources == builtin.sources
    small = regions()
    p2 = small.save(tmp_path / "s.json", SIZE)
    assert json.loads(p2.read_text(encoding="utf-8"))["regions"]["hand_1"]["px"] == [112, 520, 176, 584]
    assert H.HudRegions.load(p2, hand_centres=CENTRES).px("hand_1", SIZE) == (112, 520, 176, 584)
    with pytest.raises(FileNotFoundError):
        H.load_regions(tmp_path / "missing.json")
    with pytest.raises(ValueError):
        H.HudRegions.from_dict({"schema": "something/else", "regions": {}}, hand_centres=CENTRES)
    with pytest.raises(G.FrameSizeError):
        H.HudRegions.from_dict({"frame_size": [1080, 2400], "regions": {}}, hand_centres=CENTRES)


def test_hand_boxes_from_the_geometry_match_the_authors_card_boxes():
    regs = H.HudRegions.builtin(hand_centres=CENTRES)
    sx, sy = 1080 / 368, 1920 / 652
    for s in range(4):
        x0 = 84 + 69 * s                       # scripts/brain/cards.py:41-61 hand_boxes(1080, 1920)
        expected = (round(x0 * sx), round(543 * sy), round((x0 + 61) * sx), round((543 + 73) * sy))
        got = regs.px(f"hand_{s}", (1080, 1920))
        assert all(abs(a - b) <= 6 for a, b in zip(got, expected)), (s, got, expected)


# ---------------------------------------------------------------------------
# HudReader end to end, and never raising
# ---------------------------------------------------------------------------
def hud_frame(clock="2:59", elixir="6", fill_px=160) -> np.ndarray:
    f = hand_frame([slot_art(i) for i in range(4)], next_art(4))
    paste(f, compose(clock), 260, 25)
    paste(f, compose(elixir, bold_glyph), 71, 600)
    f[610:620, 100:100 + fill_px] = PURPLE
    f[610:620, 100 + fill_px:350] = TRACK
    return f


def full_store() -> TP.TemplateStore:
    store = digit_store(("clock", "elixir"))
    for i in range(8):
        store.add(TP.KIND_HAND, f"c{i}", slot_art(i))
    return store


def test_hud_reader_end_to_end():
    # towers_clock_reads=1: the towers count from this first clock reading (the default waits for 3)
    reader = H.HudReader(full_store(), CARD_IDS, regions=regions(), towers_clock_reads=1)
    hud = reader.read(hud_frame(), 0.0)
    assert isinstance(hud, VT.Hud)
    assert (hud.clock_tick, hud.phase) == (30, REG)
    assert hud.elixir_milli == 6400            # digit 6 and bar 0.64 agree
    assert (hud.hand, hud.next) == ([100, 101, 102, 103], 104)
    # no tower regions here: the level-11 prior maxima, and kings full until they show a number
    assert hud.towers == {k: VT.TowerHp(4824, 4824) if k.endswith("king") else VT.TowerHp(None, 3052)
                          for k in VT.TOWER_KEYS}
    assert hud.crowns == (0, 0)
    assert reader.report.errors == []
    assert set(reader.report.ms) == {"clock", "elixir", "towers", "hand"}
    hud = reader.read(hud_frame(), 0.25)
    assert (hud.clock_tick, hud.elixir_milli) == (35, 6400)
    assert reader.report.summary()["clock"]["text"] == "2:59"
    VT.Hud.from_dict(json.loads(json.dumps(hud.to_dict())))      # the contract still round-trips


def test_hud_reader_restarts_the_hand_once_the_clock_runs():
    """The intro shows cards that are not the hand (and the owner's friendly then showed four
    empty slots): on the third accepted clock reading the hand tracker and the elixir filter
    start again, once per reset(), so an intro "play" is forgotten."""
    reader = H.HudReader(full_store(), CARD_IDS, regions=regions())
    intro = hud_frame()
    paste(intro, FLAT, HAND_PX[1][0], HAND_PX[1][1])
    reader.read(hud_frame(), 0.0)
    assert not reader.report.hand_restarted
    reader.read(intro, 0.25)
    assert reader.hand.plays == [(0.25, 1, 101)] and not reader.report.hand_restarted
    hud = reader.read(hud_frame(), 0.5)                   # the third accepted clock reading
    assert reader.report.hand_restarted and reader.report.summary()["hand_restarted"] is True
    assert hud.hand == [100, 101, 102, 103] and reader.hand.plays == [] and 101 not in reader.hand.cycle
    assert hud.elixir_milli == 6400
    reader.read(hud_frame(), 0.75)
    assert not reader.report.hand_restarted and "hand_restarted" not in reader.report.summary()
    reader.reset()
    reader.read(hud_frame(), 1.0)
    assert not reader.report.hand_restarted                 # counting again from the reset
    off = H.HudReader(full_store(), CARD_IDS, regions=regions(), hand_restarts_at_clock=False)
    for i, f in enumerate((hud_frame(), intro, hud_frame())):
        off.read(f, 0.25 * i)
    assert off.hand.plays == [(0.25, 1, 101)]


def test_hud_reader_reads_the_elixir_count_through_the_bar_line():
    # the elixir bar's 1 px top line across the elixir_digit box (x 70-100), 2 px
    # clear of the '6' (x 73-87), and a speck above it in its columns, as on the
    # owner's frame 00000665; the elixir font's style cleans both away, the
    # clock's style, passed for the elixir font, reads them as a second glyph
    f = hud_frame()
    f[612, 70] = WHITE
    f[612, 90:100] = WHITE
    f[602, 95] = WHITE
    reader = H.HudReader(full_store(), CARD_IDS, regions=regions())
    hud = reader.read(f, 0.0)
    assert (reader.report.elixir_read.digit, reader.report.elixir_read.text) == (6, "6")
    assert hud.elixir_milli == 6400 and reader.report.errors == []
    plain = H.HudReader(full_store(), CARD_IDS, regions=regions(),
                        glyph_styles={"elixir": TP.DEFAULT_GLYPH_STYLE})
    plain.read(f, 0.0)
    assert (plain.report.elixir_read.digit, plain.report.elixir_read.text) == (None, "6:")


def test_hud_reader_without_templates_reads_unknowns_without_raising():
    reader = H.HudReader(TP.TemplateStore.memory(), {}, regions=regions())
    hud = reader.read(np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8), 0.0)
    assert hud.clock_tick is None and hud.phase == VT.PHASE_UNKNOWN
    assert hud.elixir_milli is None and reader.report.elixir.source == "unread"   # black is no bar
    assert hud.hand == [VT.EMPTY_CARD] * 4 and hud.next == VT.EMPTY_CARD
    assert reader.report.errors == []


def test_hud_reader_never_raises():
    reader = H.HudReader(full_store(), CARD_IDS, regions=regions())
    hud = reader.read(np.zeros((400, 400, 3), dtype=np.uint8), 0.0)        # not 9:16
    assert hud == VT.Hud() and reader.report.errors[0].startswith("frame: FrameSizeError")
    assert reader.read(np.zeros((640, 360), dtype=np.uint8), 0.0) == VT.Hud()

    def boom(_rgb):
        raise RuntimeError("bad crop")
    reader.clock_reader.read = boom
    hud = reader.read(hud_frame(), 0.0)
    assert reader.report.errors == ["clock: RuntimeError: bad crop"]
    assert hud.clock_tick is None and hud.hand == [100, 101, 102, 103]      # the rest still read


def test_hud_reader_with_default_regions_and_store():
    reader = H.HudReader()                     # data/vision/templates may be empty or missing
    hud = reader.read(np.zeros((1920, 1080, 3), dtype=np.uint8), 0.0)
    assert isinstance(hud, VT.Hud) and reader.report.errors == []
    assert set(hud.towers) == set(VT.TOWER_KEYS)


# ---------------------------------------------------------------------------
# CLIs
# ---------------------------------------------------------------------------
def test_templates_cli_captures_digits_and_hand(tmp_path, capsys):
    regions_path = regions().save(tmp_path / "regions.json", SIZE)
    shot = hud_frame()
    image = tmp_path / "shot.npy"
    R.save_image(image, shot)
    root = tmp_path / "tpl"
    common = ["--image", str(image), "--regions", str(regions_path), "--root", str(root)]
    assert TP.main(["digits", "--font", "clock", "--text", "2:59", *common]) == 0
    assert TP.main(["hand", "--cards", "c0,c1,-,c3", "--next", "c4", *common]) == 0
    assert TP.main(["card", "--slot", "2", "--label", "c2", "--variant", "evo", *common]) == 0
    assert TP.main(["digits", "--font", "clock", "--text", "2:5", *common]) == 1     # 4 glyphs, 3 chars
    store = TP.TemplateStore(root)
    assert store.labels(TP.digit_kind("clock")) == ["2", "5", "9", ":"]
    assert store.labels(TP.KIND_HAND) == ["c0", "c1", "c2", "c3"] and store.labels(TP.KIND_NEXT) == ["c4"]
    assert np.array_equal(store.entries(TP.KIND_HAND, "c1")[0].image, slot_art(1))
    assert store.entries(TP.KIND_HAND, "c2")[0].variant == "evo"
    capsys.readouterr()
    assert TP.main(["list", "--root", str(root), "--deck", "c0,c9"]) == 0
    out = capsys.readouterr().out
    assert "digits/clock: missing 0 1 3 4 6 7 8" in out
    assert "hand/c0: base" in out and "hand/c9: MISSING" in out
    # the HUD CLI reads the same screenshot with those templates
    assert H.main(["read", "--image", str(image), "--root", str(root), "--deck", "c0,c1,c2,c3",
                   "--regions", str(regions_path)]) == 0
    assert "tick 30" in capsys.readouterr().out
    out_regions = tmp_path / "pc.json"
    assert H.main(["regions", "--regions", str(regions_path), "--out", str(out_regions), "--size", "360x640"]) == 0
    assert H.HudRegions.load(out_regions, hand_centres=CENTRES).px("next", SIZE) == NEXT_PX


def test_new_vision_modules_never_import_the_memory_reader():
    for name in ("hud.py", "hpbars.py", "templates.py"):
        tree = ast.parse((ROOT / "scripts" / "vision" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            assert not any("memread" in m or "gamestate" in m or "hcr_reader" in m for m in mods), (name, mods)


def test_hud_modules_import_light_and_run_as_files():
    code = ("import sys\n"
            "import scripts.vision.hud, scripts.vision.hpbars, scripts.vision.templates\n"
            "heavy = ('torch', 'cv2', 'PIL', 'ultralytics', 'scripts.gamestate.memread', 'royalesim')\n"
            "print(','.join(m for m in heavy if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""
    for module in ("hud", "templates"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "vision" / f"{module}.py"), "--help"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        assert "usage" in r.stdout.lower()
