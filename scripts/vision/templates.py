"""Template store, numpy NCC and digit OCR for the HUD readers, plus the capture CLI (V07).

Templates are small crops of the game's own HUD, captured from YOUR client on the
PC (the user's build is newer than the author's, and the vendored BuildABot card
art differs from the live client: vision-15). Nothing here ships templates; the
readers in scripts/vision/hud.py work without them and report "unread".

Layout under the root (default data/vision/templates/, or --root / TemplateStore(root)):

    digits/clock/<char>/<variant>_<name>.png    char 0-9, "colon" for ':' (optional)
    digits/elixir/<char>/...                    the number in the elixir drop
    digits/tower/<char>/...                     tower hit points (princess and king)
    hand/<Card>/<variant>_<name>.png            Card = RoyaleSim catalogue name (HogRider)
    next/<Card>/<variant>_<name>.png            optional: the hand art is used without it
    label/clock/<regulation|overtime>/...       the text above the clock digits (optional)
    empty/<hand|next>/<variant>_<name>.png      an empty slot between a play and its refill

`variant` is one of VARIANTS (base, evo, hero, greyed, evo_greyed, hero_greyed); a
file whose name does not start with one counts as "base". Files are .png/.jpg
(OpenCV or Pillow reads them) or .npy (numpy only). Several files per label are
fine: a label scores the best of its files. They are small UI crops, kept on the
PC; whether to commit them is up to you.

Capture them in one friendly (screenshots: `adb exec-out screencap -p > shot.png`,
or a session recorded with `python -m scripts.vision.capture --record ...`):

    python -m scripts.vision.hud overlay --image shot.png --out check.png   (boxes right?)
    python -m scripts.vision.templates digits --image shot.png --font clock --text 2:47
    python -m scripts.vision.templates digits --image shot.png --font elixir --text 7
    python -m scripts.vision.templates digits --image shot.png --font tower \
        --region tower_enemy_left --text 3052
    python -m scripts.vision.templates hand --image shot.png \
        --cards HogRider,Musketeer,Cannon,Log --next Fireball [--variants base,evo,base,base]
    python -m scripts.vision.templates empty --image just_played.png --slot 2
    python -m scripts.vision.templates label --image shot.png --label regulation
    python -m scripts.vision.templates list --deck HogRider,Musketeer,Cannon,IceSpirits,\
        Skeletons,IceGolemite,Log,Fireball

Repeat `digits` on different screenshots until `list` shows 0-9 for each font. For
greyed art take a shot while the card is unaffordable (--variants greyed,...); for
evolved or hero art, while that form is in hand (--variants evo / hero).

Faster, from a whole recording (scripts/vision/harvest.py): `clock-sweep` labels
every second of the clock from one frame you read; `harvest` clusters the elixir,
tower, hand or next-card crops into a contact sheet and `harvest-label` stores the
clusters you name:

    python -m scripts.vision.templates clock-sweep --recording REC --anchor REC/images/00001200.jpg --text 1:48
    python -m scripts.vision.templates harvest --recording REC --what tower --out tmp/harvest_tower.png
    python -m scripts.vision.templates harvest-label --sheet tmp/harvest_tower.json --map "0=3,1=0,2=5"

Matching (numpy only; OpenCV is never needed):
* cards and labels: zero-mean, unit-norm greyscale signatures compared by dot
  product = normalised cross-correlation, the method of scripts/brain/cards.py
  (32x32 NCC, cards.py:46-72; MIN_SCORE 0.30 / MIN_MARGIN 0.04, cards.py:47-48,
  89-103). NCC is invariant to the linear brightness/contrast change of greyed
  cards (cards.py:19-22). Resizing is exact area averaging (`area_resize`), so a
  crop and its template give identical signatures in every environment.
* digits: the digit fill is thresholded (GlyphStyle: white for the clock and
  the elixir count, TOWER_GLYPH_STYLE's pinkish-white / light-blue tints for
  tower hit points, measured on the owner's frames), split into glyphs by
  column runs, each glyph letterboxed to 20x20 (aspect kept, so '1' stays thin)
  and matched by NCC against the font's templates. A glyph much shorter than
  the tallest is punctuation (the clock's ':') unless ':' templates exist;
  one under SPECK_FRAC of the tallest is noise and ignored.
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__" and not __package__:
    # Run as a file, sys.path[0] is scripts/vision, whose types.py would shadow
    # the standard library's `types`: put the repo root there instead and
    # import as part of the package (PEP 366), so the relative imports work.
    _here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.path and os.path.normcase(os.path.abspath(sys.path[0] or os.curdir)) == _here:
        sys.path[0] = _root
    else:
        sys.path.insert(0, _root)
    __import__("scripts.vision")
    __package__ = "scripts.vision"

import argparse  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from .record import image_codec_available, load_image, save_image  # noqa: E402
from .types import FORM_BASE, FORM_EVO, FORM_HERO  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data" / "vision" / "templates"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".npy")

FONTS = ("clock", "elixir", "tower")
DIGITS = "0123456789"
CHAR_DIRS = {":": "colon"}                      # ':' cannot be a folder name on Windows
DIR_CHARS = {v: k for k, v in CHAR_DIRS.items()}
VARIANTS = ("base", "evo", "hero", "greyed", "evo_greyed", "hero_greyed")
VARIANT_FORM = {"base": FORM_BASE, "greyed": FORM_BASE, "evo": FORM_EVO, "evo_greyed": FORM_EVO,
                "hero": FORM_HERO, "hero_greyed": FORM_HERO}
KIND_HAND = "hand"
KIND_NEXT = "next"
KIND_EMPTY = "empty"                            # labels "hand" and "next"
KIND_CLOCK_LABEL = "label/clock"                # labels "regulation" and "overtime"
CLOCK_LABELS = ("regulation", "overtime")

# cards.py:46-48 (the author's hand classifier): 32x32 NCC, a floor and a margin.
PATCH = (32, 32)
MIN_SCORE = 0.30
MIN_MARGIN = 0.04
LABEL_PATCH = (64, 16)
GLYPH_W = GLYPH_H = 20

_KIND_RE = re.compile(r"^[a-z]+(/[a-z]+)?$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float64)


class TemplateError(ValueError):
    """A template that cannot be stored (bad kind/label/variant, glyph count mismatch)."""


def digit_kind(font: str) -> str:
    if font not in FONTS:
        raise TemplateError(f"font must be one of {FONTS}, not {font!r}")
    return f"digits/{font}"


# ---------------------------------------------------------------------------
# Image primitives (numpy only)
# ---------------------------------------------------------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    """HxW float64 grey from an HxW or HxWxC (RGB first) array."""
    a = np.asarray(img)
    if a.ndim == 2:
        return a.astype(np.float64)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., :3].astype(np.float64) @ _GRAY
    raise ValueError(f"expected an HxW or HxWx3 image, got shape {a.shape}")


@lru_cache(maxsize=512)
def _area_weights(n: int, m: int) -> np.ndarray:
    """(m, n) matrix averaging n input cells into m output cells by overlap."""
    scale = n / m
    edges = np.arange(m + 1, dtype=np.float64) * scale
    k = np.arange(n, dtype=np.float64)
    lo = np.maximum(edges[:-1, None], k[None, :])
    hi = np.minimum(edges[1:, None], k[None, :] + 1.0)
    w = np.clip(hi - lo, 0.0, None) / scale
    w.setflags(write=False)
    return w


def area_resize(a: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a 2-D array to (height, width) by exact area averaging.

    Each output cell is the overlap-weighted mean of the input cells it covers
    (a box filter when enlarging). Shrinking a 2x-upscaled image by 2 returns
    the original exactly. Empty input gives zeros.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"area_resize needs a 2-D array, got {a.shape}")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"target size must be positive, got {width}x{height}")
    if a.shape[0] == 0 or a.shape[1] == 0:
        return np.zeros((height, width), dtype=np.float64)
    wy = _area_weights(a.shape[0], height)
    wx = _area_weights(a.shape[1], width)
    return wy @ a @ wx.T


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros_like(v)


def signature(img: np.ndarray, size: Sequence[int] = PATCH) -> np.ndarray:
    """Zero-mean, unit-norm greyscale vector at `size` (w, h): the NCC-ready form
    (cards.py:64-72). A flat crop gives the zero vector, which scores 0 against
    everything."""
    return _normalise(area_resize(to_gray(img), int(size[0]), int(size[1])))


def crop_std(img: np.ndarray) -> float:
    """Standard deviation of the grey levels (0 for a flat crop)."""
    g = to_gray(img)
    return float(g.std()) if g.size else 0.0


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TemplateEntry:
    kind: str
    label: str
    variant: str
    image: np.ndarray = field(repr=False)
    path: Path | None = None

    @property
    def form(self) -> int:
        return VARIANT_FORM.get(self.variant, FORM_BASE)

    @property
    def greyed(self) -> bool:
        return self.variant.endswith("greyed")


def _check_kind(kind: str) -> str:
    if not _KIND_RE.match(str(kind)):
        raise TemplateError(f"bad template kind {kind!r} (e.g. 'hand', 'digits/clock')")
    return str(kind)


def label_dir(label: str) -> str:
    """Folder name for a label (':' -> 'colon'); refuses path-like labels."""
    name = CHAR_DIRS.get(str(label), str(label))
    if not _LABEL_RE.match(name):
        raise TemplateError(f"bad template label {label!r}: letters, digits, _ . + - only")
    return name


def variant_of(stem: str) -> str:
    """The variant a file name starts with ('evo_greyed_shot1' -> 'evo_greyed'), else 'base'."""
    for v in sorted(VARIANTS, key=len, reverse=True):
        if stem == v or stem.startswith(v + "_"):
            return v
    return "base"


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-.")
    return s or time.strftime("%Y%m%d-%H%M%S")


class TemplateStore:
    """Labelled template images on disk under `root`, plus in-memory ones.

    root=None keeps everything in memory (tests, experiments). Disk folders are
    scanned once per kind and cached; `reload()` rescans. Unreadable files are
    skipped and listed in `errors`; a missing root is simply empty. Nothing
    here raises while reading.
    """

    def __init__(self, root: str | os.PathLike | None = DEFAULT_ROOT):
        self.root = None if root is None else Path(root)
        self._mem: dict[str, list[TemplateEntry]] = {}
        self._disk: dict[str, list[TemplateEntry]] = {}
        self.errors: list[str] = []

    @classmethod
    def memory(cls) -> "TemplateStore":
        return cls(None)

    def __repr__(self) -> str:
        return f"TemplateStore({str(self.root) if self.root else 'memory'!r})"

    # -- reading -------------------------------------------------------------
    def _scan(self, kind: str) -> list[TemplateEntry]:
        if kind in self._disk:
            return self._disk[kind]
        out: list[TemplateEntry] = []
        base = None if self.root is None else self.root.joinpath(*kind.split("/"))
        if base is not None and base.is_dir():
            for sub in sorted(p for p in base.iterdir() if p.is_dir()):
                label = DIR_CHARS.get(sub.name, sub.name)
                for f in sorted(sub.iterdir()):
                    if f.suffix.lower() not in IMAGE_SUFFIXES or not f.is_file():
                        continue
                    try:
                        img = load_image(f)
                    except Exception as exc:          # a bad file must not stop the readers
                        self.errors.append(f"{f}: {type(exc).__name__}: {exc}")
                        continue
                    if img.size == 0:
                        self.errors.append(f"{f}: empty image")
                        continue
                    out.append(TemplateEntry(kind, label, variant_of(f.stem), img, f))
        self._disk[kind] = out
        return out

    def entries(self, kind: str, label: str | None = None) -> list[TemplateEntry]:
        kind = _check_kind(kind)
        items = self._scan(kind) + self._mem.get(kind, [])
        return [e for e in items if label is None or e.label == label]

    def labels(self, kind: str) -> list[str]:
        return sorted({e.label for e in self.entries(kind)})

    def reload(self) -> None:
        self._disk.clear()
        self.errors.clear()

    def kinds(self) -> list[str]:
        """Kinds present on disk or in memory."""
        found = set(self._mem)
        if self.root is not None and self.root.is_dir():
            for top in sorted(p for p in self.root.iterdir() if p.is_dir()):
                subs = [p for p in top.iterdir() if p.is_dir()]
                if top.name in ("digits", "label"):
                    found |= {f"{top.name}/{s.name}" for s in subs}
                elif subs:
                    found.add(top.name)
        return sorted(found)

    def summary(self) -> dict[str, dict[str, int]]:
        """kind -> label -> number of templates."""
        out: dict[str, dict[str, int]] = {}
        for kind in self.kinds():
            counts: dict[str, int] = {}
            for e in self.entries(kind):
                counts[e.label] = counts.get(e.label, 0) + 1
            out[kind] = dict(sorted(counts.items()))
        return out

    # -- writing ---------------------------------------------------------------
    def add(self, kind: str, label: str, image: np.ndarray, variant: str = "base",
            name: str | None = None) -> Path | None:
        """Store one template; returns the file written (None for a memory store).

        Writes PNG when OpenCV or Pillow is installed, else .npy.
        """
        kind = _check_kind(kind)
        folder = label_dir(label)
        if variant not in VARIANTS:
            raise TemplateError(f"variant must be one of {VARIANTS}, not {variant!r}")
        img = np.ascontiguousarray(np.asarray(image))
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        if img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
            raise TemplateError(f"template image must be a non-empty HxWx3 array, got {img.shape}")
        img = img.astype(np.uint8, copy=False)
        if self.root is None:
            self._mem.setdefault(kind, []).append(TemplateEntry(kind, str(label), variant, img.copy()))
            return None
        d = self.root.joinpath(*kind.split("/"), folder)
        d.mkdir(parents=True, exist_ok=True)
        ext = "png" if image_codec_available("png") else "npy"
        stem = f"{variant}_{_safe_name(name or time.strftime('%Y%m%d-%H%M%S'))}"
        path, i = d / f"{stem}.{ext}", 2
        while path.exists():
            path, i = d / f"{stem}-{i}.{ext}", i + 1
        save_image(path, img, ext)
        self._disk.pop(kind, None)
        return path


# ---------------------------------------------------------------------------
# Card / label matching
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Match:
    """Best label for a crop: label None when below the floor or ambiguous."""
    label: str | None
    score: float
    margin: float
    variant: str = "base"
    scores: Mapping[str, float] = field(default_factory=dict)   # label -> best score
    variants: Mapping[str, str] = field(default_factory=dict)   # label -> best variant


class CardMatcher:
    """NCC of a crop against labelled templates (cards.py's method, numpy only)."""

    def __init__(self, entries: Iterable[TemplateEntry], size: Sequence[int] = PATCH, *,
                 labels: Iterable[str] | None = None, min_score: float = MIN_SCORE,
                 min_margin: float = MIN_MARGIN):
        keep = None if labels is None else set(labels)
        self.size = (int(size[0]), int(size[1]))
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self._labels: list[str] = []
        self._variants: list[str] = []
        sigs = []
        for e in entries:
            if keep is not None and e.label not in keep:
                continue
            s = signature(e.image, self.size)
            if not s.any():
                continue                              # a flat template matches nothing
            self._labels.append(e.label)
            self._variants.append(e.variant)
            sigs.append(s)
        self._sigs = np.stack(sigs) if sigs else np.zeros((0, self.size[0] * self.size[1]))

    @property
    def ready(self) -> bool:
        return len(self._labels) > 0

    @property
    def labels(self) -> list[str]:
        return sorted(set(self._labels))

    def scores(self, crop: np.ndarray) -> tuple[dict[str, float], dict[str, str]]:
        """label -> best NCC over its templates, and label -> that template's variant."""
        best: dict[str, float] = {}
        var: dict[str, str] = {}
        if not self.ready or np.asarray(crop).size == 0:
            return best, var
        dots = self._sigs @ signature(crop, self.size)
        for lab, v, d in zip(self._labels, self._variants, dots.tolist()):
            if lab not in best or d > best[lab]:
                best[lab], var[lab] = float(d), v
        return best, var

    def match(self, crop: np.ndarray) -> Match:
        best, var = self.scores(crop)
        if not best:
            return Match(None, 0.0, 0.0)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        lab, score = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = score - runner
        ok = score >= self.min_score and margin >= self.min_margin
        return Match(lab if ok else None, score, margin, var[lab], best, var)


# ---------------------------------------------------------------------------
# Digit OCR
# ---------------------------------------------------------------------------
Rule = tuple[int, int, int, int]                 # (cr, cg, cb, t): cr*R + cg*G + cb*B >= t


@dataclass(frozen=True)
class GlyphStyle:
    """Which pixels are digit fill.

    White test: bright (min channel >= min_value) and unsaturated (max - min <=
    max_chroma); min_value > 255 switches it off. `tints` adds tinted fills (OR):
    each tint is a tuple of linear rules (cr, cg, cb, t), all of which a pixel
    must meet (cr*R + cg*G + cb*B >= t). The clock and elixir digits are white
    with a dark outline; the tower digits are tinted (TOWER_GLYPH_STYLE).
    """
    min_value: int = 190
    max_chroma: int = 45
    tints: tuple[tuple[Rule, ...], ...] = ()

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        a = np.asarray(rgb)
        if a.ndim == 2:
            return a >= self.min_value
        a = a[..., :3].astype(np.int32)
        lo, hi = a.min(axis=2), a.max(axis=2)
        m = (lo >= self.min_value) & (hi - lo <= self.max_chroma)
        for rules in self.tints:
            t = np.ones(m.shape, dtype=bool)
            for cr, cg, cb, th in rules:
                t &= (cr * a[..., 0] + cg * a[..., 1] + cb * a[..., 2]) >= th
            m |= t
        return m

    def to_dict(self) -> dict[str, Any]:
        return {"min_value": self.min_value, "max_chroma": self.max_chroma,
                "tints": [[list(r) for r in rules] for rules in self.tints]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "GlyphStyle":
        d = d or {}
        tints = tuple(tuple(tuple(int(c) for c in r) for r in rules) for rules in d.get("tints") or ())
        for rules in tints:
            if any(len(r) != 4 for r in rules):
                raise ValueError("a tint rule is (cr, cg, cb, threshold)")
        return cls(int(d.get("min_value", 190)), int(d.get("max_chroma", 45)), tints)


DEFAULT_GLYPH_STYLE = GlyphStyle()
#: Tower hit points on the owner's client (frames of 2026-10-29, build 160635001):
#: the opponent's digits are pinkish white (255, 220-240, 230-245) over the light,
#: yellowish arena floor (250, 245, 210), which the white test would take too,
#: so blue >= green tells them apart; ours are light blue (180-200, 210-230,
#: 225-250) over the grey tower. The bars (deep pink, mid blue) and the floor
#: fail both tints. Pure white passes the first tint.
TOWER_GLYPH_STYLE = GlyphStyle(
    min_value=256, max_chroma=0,
    tints=(((1, 0, 0, 225), (0, 1, 0, 185), (0, 0, 1, 195), (0, -1, 1, -2)),     # enemy: pinkish white
           ((1, 0, 0, 160), (0, 1, 0, 195), (0, 0, 1, 215), (-1, 0, 1, 15))))    # own: light blue
#: The style each font's templates and readings use unless one is passed.
FONT_STYLES = {"clock": DEFAULT_GLYPH_STYLE, "elixir": DEFAULT_GLYPH_STYLE, "tower": TOWER_GLYPH_STYLE}
#: A glyph box less than this fraction of the tallest glyph's height is a speck
#: (JPEG noise, a bar highlight), not a character; ':' is about 0.55-0.7.
SPECK_FRAC = 0.3


def font_style(font: str) -> GlyphStyle:
    return FONT_STYLES.get(font, DEFAULT_GLYPH_STYLE)


def segment_glyphs(mask: np.ndarray, min_pixels: int = 3) -> list[tuple[int, int, int, int]]:
    """Glyph boxes (x0, y0, x1, y1), left to right: runs of columns holding any
    fill pixel, cut to their own rows; runs with fewer than `min_pixels` pixels
    (specks) are dropped."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.size == 0:
        return []
    cols = np.concatenate([[False], m.any(axis=0), [False]]).astype(np.int8)
    d = np.diff(cols)
    boxes = []
    for x0, x1 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        rows = np.nonzero(m[:, x0:x1].any(axis=1))[0]
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        if int(m[y0:y1, x0:x1].sum()) < min_pixels:
            continue
        boxes.append((int(x0), y0, int(x1), y1))
    return boxes


def drop_specks(boxes: Sequence[tuple[int, int, int, int]],
                frac: float = SPECK_FRAC) -> list[tuple[int, int, int, int]]:
    """Boxes at least `frac` of the tallest box's height (specks removed)."""
    if not boxes:
        return []
    tallest = max(b[3] - b[1] for b in boxes)
    return [b for b in boxes if (b[3] - b[1]) >= frac * tallest]


def glyph_signature(mask: np.ndarray) -> np.ndarray:
    """A glyph mask letterboxed into GLYPH_W x GLYPH_H (height filled, aspect kept,
    centred), zero-mean and unit-norm."""
    m = np.asarray(mask, dtype=np.float64)
    h, w = m.shape
    if h == 0 or w == 0:
        return np.zeros(GLYPH_W * GLYPH_H)
    tw = min(GLYPH_W, max(1, int(math.floor(w * GLYPH_H / h + 0.5))))
    canvas = np.zeros((GLYPH_H, GLYPH_W), dtype=np.float64)
    x0 = (GLYPH_W - tw) // 2
    canvas[:, x0:x0 + tw] = area_resize(m, tw, GLYPH_H)
    return _normalise(canvas)


def _tight(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask[:0, :0]
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


@dataclass(frozen=True)
class Glyph:
    box: tuple[int, int, int, int]     # in the region's pixels
    char: str
    score: float
    margin: float
    small: bool = False


@dataclass(frozen=True)
class DigitRead:
    """text is None when nothing (or not everything) could be read; `reason` says why."""
    text: str | None
    glyphs: tuple[Glyph, ...] = ()
    conf: float = 0.0
    reason: str = ""


class DigitOCR:
    """Template OCR for one HUD font (digits/<font> in the store). The glyph style
    defaults to the font's (FONT_STYLES); glyph boxes under SPECK_FRAC of the
    tallest are ignored."""

    def __init__(self, store: TemplateStore | None, font: str, style: GlyphStyle | None = None, *,
                 min_score: float = 0.6, min_margin: float = 0.02, small_frac: float = 0.75,
                 min_pixels: int = 3):
        self.font = font
        self.style = style or font_style(font)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.small_frac = float(small_frac)
        self.min_pixels = int(min_pixels)
        self.refs: dict[str, np.ndarray] = {}
        if store is None:
            return
        per: dict[str, list[np.ndarray]] = {}
        for e in store.entries(digit_kind(font)):
            g = _tight(self.style.mask(e.image))
            if g.size == 0:
                continue
            per.setdefault(e.label, []).append(glyph_signature(g))
        self.refs = {ch: np.stack(v) for ch, v in sorted(per.items())}

    @property
    def ready(self) -> bool:
        return bool(self.refs)

    def read(self, rgb: np.ndarray) -> DigitRead:
        if not self.refs:
            return DigitRead(None, (), 0.0, f"no {self.font} digit templates")
        if np.asarray(rgb).size == 0:
            return DigitRead(None, (), 0.0, "empty region")
        mask = self.style.mask(rgb)
        boxes = drop_specks(segment_glyphs(mask, self.min_pixels))
        if not boxes:
            return DigitRead(None, (), 0.0, "no glyphs")
        tallest = max(b[3] - b[1] for b in boxes)
        glyphs: list[Glyph] = []
        conf = 1.0
        for box in boxes:
            x0, y0, x1, y1 = box
            small = (y1 - y0) < self.small_frac * tallest
            if small and ":" not in self.refs:
                glyphs.append(Glyph(box, ":", 1.0, 1.0, True))
                continue
            sig = glyph_signature(mask[y0:y1, x0:x1])
            ranked = sorted(((float((refs @ sig).max()), ch) for ch, refs in self.refs.items()),
                            key=lambda sc: (-sc[0], sc[1]))
            best, ch = ranked[0]
            runner = ranked[1][0] if len(ranked) > 1 else -1.0
            glyphs.append(Glyph(box, ch, best, best - runner, small))
            if best < self.min_score or best - runner < self.min_margin:
                return DigitRead(None, tuple(glyphs), best,
                                 f"glyph {len(glyphs) - 1} unclear ({ch!r} {best:.2f}, margin {best - runner:.2f})")
            conf = min(conf, best)
        return DigitRead("".join(g.char for g in glyphs), tuple(glyphs), conf)


def add_digit_templates(store: TemplateStore, font: str, region_rgb: np.ndarray, text: str, *,
                        style: GlyphStyle | None = None, name: str | None = None) -> list[Path | None]:
    """Segment a HUD region showing `text` and store one template per glyph.

    `text` lists the characters left to right as the screen shows them, ':'
    included (e.g. '2:47'); spaces are ignored. Uses the font's glyph style
    unless `style` is given, and ignores specks (drop_specks). Raises
    TemplateError when the number of glyphs found differs from the number of
    characters.
    """
    kind = digit_kind(font)
    chars = [c for c in str(text) if not c.isspace()]
    bad = [c for c in chars if c not in DIGITS and c != ":"]
    if bad:
        raise TemplateError(f"digit templates take 0-9 and ':', not {''.join(bad)!r}")
    mask = (style or font_style(font)).mask(region_rgb)
    boxes = drop_specks(segment_glyphs(mask))
    if len(boxes) != len(chars):
        where = ", ".join(f"x {b[0]}-{b[2]}" for b in boxes) or "none"
        raise TemplateError(f"found {len(boxes)} glyphs ({where}) but the text has {len(chars)} "
                            f"characters; check the region with `hud overlay` or pass --box")
    out = []
    for i, ((x0, y0, x1, y1), ch) in enumerate(zip(boxes, chars)):
        crop = np.asarray(region_rgb)[y0:y1, x0:x1]
        out.append(store.add(kind, ch, crop, "base", f"{name or 'glyph'}-{i}"))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_box(text: str) -> tuple[int, int, int, int]:
    try:
        x0, y0, x1, y1 = (int(v) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"box must be x0,y0,x1,y1 pixels, not {text!r}") from exc
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError(f"box {text!r} is empty")
    return x0, y0, x1, y1


def _load_shot(path: str) -> np.ndarray:
    from .geometry import check_frame_size
    rgb = load_image(path)
    check_frame_size(rgb.shape[1], rgb.shape[0])
    return rgb


def _regions(path: str | None):
    from .hud import load_regions                  # lazy: hud imports this module
    return load_regions(path)


def _crop(rgb: np.ndarray, args, region: str | None) -> np.ndarray:
    if getattr(args, "box", None):
        x0, y0, x1, y1 = args.box
        return rgb[y0:y1, x0:x1]
    if not region:
        raise TemplateError("name a --region (see `python -m scripts.vision.hud regions`) or pass --box")
    regions = _regions(args.regions)
    crop = regions.crop(rgb, region)
    if crop is None or crop.size == 0:
        raise TemplateError(f"region {region!r} is not defined or empty")
    return crop


def _store(args) -> TemplateStore:
    return TemplateStore(args.root or DEFAULT_ROOT)


def _stem(args) -> str:
    return args.name or Path(args.image).stem


def _cmd_list(args) -> int:
    store = _store(args)
    summary = store.summary()
    print(f"templates in {store.root}")
    if not summary:
        print("   (none yet: see `python -m scripts.vision.templates --help`)")
    for kind, labels in summary.items():
        print(f"{kind}: " + ", ".join(f"{k} x{n}" for k, n in labels.items()))
    for font in FONTS:
        have = set(summary.get(digit_kind(font), {}))
        missing = [c for c in DIGITS if c not in have]
        print(f"digits/{font}: " + ("complete" if not missing else "missing " + " ".join(missing)))
    if args.deck:
        hand = {e.label: set() for e in store.entries(KIND_HAND)}
        for e in store.entries(KIND_HAND):
            hand[e.label].add(e.variant)
        for card in [c.strip() for c in args.deck.split(",") if c.strip()]:
            v = sorted(hand.get(card, ()))
            print(f"hand/{card}: " + (", ".join(v) if v else "MISSING"))
    for err in store.errors:
        print(f"unreadable: {err}")
    return 0


_FONT_REGION = {"clock": "clock", "elixir": "elixir_digit"}


def _cmd_digits(args) -> int:
    rgb = _load_shot(args.image)
    region = args.region or _FONT_REGION.get(args.font)
    crop = _crop(rgb, args, region)
    paths = add_digit_templates(_store(args), args.font, crop, args.text, name=_stem(args))
    for ch, p in zip([c for c in args.text if not c.isspace()], paths):
        print(f"'{ch}' -> {p}")
    return 0


def _slot_region(slot: str) -> tuple[str, str]:
    if slot == "next":
        return "next", KIND_NEXT
    if slot in ("0", "1", "2", "3"):
        return f"hand_{slot}", KIND_HAND
    raise TemplateError(f"slot must be 0-3 or next, not {slot!r}")


def _cmd_card(args) -> int:
    rgb = _load_shot(args.image)
    region, kind = _slot_region(args.slot)
    print(_store(args).add(kind, args.label, _crop(rgb, args, region), args.variant, _stem(args)))
    return 0


def _cmd_hand(args) -> int:
    rgb = _load_shot(args.image)
    regions = _regions(args.regions)
    store = _store(args)
    cards = [c.strip() for c in args.cards.split(",")]
    if len(cards) != 4:
        raise TemplateError(f"--cards needs 4 names (use - to skip a slot), got {len(cards)}")
    variants = [v.strip() for v in (args.variants or "base,base,base,base").split(",")]
    if len(variants) != 4:
        raise TemplateError("--variants needs 4 values")
    for slot, (card, variant) in enumerate(zip(cards, variants)):
        if card in ("", "-"):
            continue
        p = store.add(KIND_HAND, card, regions.crop(rgb, f"hand_{slot}"), variant, _stem(args))
        print(f"slot {slot} {card} ({variant}) -> {p}")
    if args.next:
        p = store.add(KIND_NEXT, args.next, regions.crop(rgb, "next"), args.next_variant, _stem(args))
        print(f"next {args.next} ({args.next_variant}) -> {p}")
    return 0


def _cmd_empty(args) -> int:
    rgb = _load_shot(args.image)
    region, kind = _slot_region(args.slot)
    print(_store(args).add(KIND_EMPTY, kind, _crop(rgb, args, region), "base", _stem(args)))
    return 0


def _cmd_label(args) -> int:
    rgb = _load_shot(args.image)
    print(_store(args).add(KIND_CLOCK_LABEL, args.label, _crop(rgb, args, "clock_label"),
                           "base", _stem(args)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", help=f"template folder (default {DEFAULT_ROOT})")
    shot = argparse.ArgumentParser(add_help=False, parents=[common])
    shot.add_argument("--image", required=True, help="screenshot of a friendly battle (9:16 portrait)")
    shot.add_argument("--regions", help="HUD regions JSON (default: hud.load_regions())")
    shot.add_argument("--box", type=_parse_box, help="crop x0,y0,x1,y1 pixels instead of a named region")
    shot.add_argument("--name", help="file name part (default: the screenshot's stem)")

    ap = argparse.ArgumentParser(prog="python -m scripts.vision.templates",
                                 description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", parents=[common], help="what templates exist, and which digits are missing")
    p.add_argument("--deck", help="comma-separated card names to check for hand art")
    p.set_defaults(func=_cmd_list)
    p = sub.add_parser("digits", parents=[shot], help="store the glyphs of a number the screenshot shows")
    p.add_argument("--font", required=True, choices=FONTS)
    p.add_argument("--text", required=True, help="what the region shows, e.g. 2:47, 7 or 3052")
    p.add_argument("--region", help="HUD region (default: clock / elixir_digit; tower_* for towers)")
    p.set_defaults(func=_cmd_digits)
    p = sub.add_parser("card", parents=[shot], help="store one hand slot's (or the next box's) art")
    p.add_argument("--slot", required=True, help="0-3 (left to right) or next")
    p.add_argument("--label", required=True, help="RoyaleSim card name, e.g. HogRider")
    p.add_argument("--variant", default="base", choices=VARIANTS)
    p.set_defaults(func=_cmd_card)
    p = sub.add_parser("hand", parents=[shot], help="store all 4 hand slots (and the next card) at once")
    p.add_argument("--cards", required=True, help="4 card names left to right; - skips a slot")
    p.add_argument("--variants", help="4 variants, e.g. base,evo,base,greyed (default all base)")
    p.add_argument("--next", help="card in the next box")
    p.add_argument("--next-variant", default="base", choices=VARIANTS)
    p.set_defaults(func=_cmd_hand)
    p = sub.add_parser("empty", parents=[shot], help="store an empty slot (a frame right after a play)")
    p.add_argument("--slot", required=True, help="0-3 or next")
    p.set_defaults(func=_cmd_empty)
    p = sub.add_parser("label", parents=[shot], help="store the clock's label text")
    p.add_argument("--label", required=True, choices=CLOCK_LABELS,
                   help="regulation ('Time left') or overtime")
    p.set_defaults(func=_cmd_label)
    from .harvest import add_parsers           # whole recordings: clock-sweep, harvest, harvest-label
    add_parsers(sub, common)
    args = ap.parse_args(argv)
    try:
        return int(args.func(args))
    except (TemplateError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
