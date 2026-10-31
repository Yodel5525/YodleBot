"""Is this battle classic 1v1? The game-mode check of the live runner's tap gate (skin audit P1-12).

The bot's timings, its elixir model and the token model's training are all classic 1v1: a 3:00
clock and one elixir every 2.8 s for the first two minutes (hud.py's REGEN_MS_1X). The owner says
the arena changes with the game mode, and a friendly can be started in another mode. So in the
first WINDOW_S (10) s of each battle the runner checks two things, and taps in AUTO only when both
say classic (scripts/vision/tap_gate.py; the six towers calibrated at their positions is the
gate's third check, P0-4):

* THE CLOCK. The battle's first clock reading (vision_play's BattleState.first_tick, the tick
  hud.ClockFilter gave it) shows 2:5x-3:00 in regulation: first_tick <= FIRST_TICK_MAX (219: 2:50
  shows from tick 200 to 219). A clock that starts at 2:00 or 1:00 (tick 1200, 2400), or a battle
  the runner joined in the middle, is not classic.
* THE ELIXIR. Our elixir regenerates at REGEN_S_1X (2.8) s per elixir +-15% while no card is
  played, which double (1.4 s) and triple elixir (0.93 s) do not. Each battle frame gives an
  interval the elixir lies in (elixir_interval): the count digit d says [d, d + 1) (the digit is
  the whole elixir), the bar's fill f says 10 f +- BAR_TOL (0.3) elixir; both together their
  intersection. When they disagree (no intersection) one of them is misread and the frame gives
  no reading at all: a misread digit kept would cut the readings or ask for a rate no mode has.
  A reading clearly below the one before it (its top under the other's bottom, or the bar's own
  value CUT_DROP (0.5) elixir or more below the one before) is a play: the readings are cut there
  into segments, and within a segment the elixir rises at one rate,
  e(t) = c + t / P. A period P is consistent with a segment when one c puts every reading of it
  inside its interval, all but its outliers (misreads): OUTLIER_SHARE (5%) of its readings, and
  at least OUTLIER_MIN (1) once it has OUTLIER_FROM (5) readings. regen_fit() tries every P from
  0.5 to 6 s in 0.02 s steps on every segment; the consistent ones are the battle's possible
  regeneration periods [lo, hi].
  Classic: some of them lie in 2.38-3.22 s, none is 1.61 s or less (2x elixir's 1.4 s + 15%),
  and the elixir was seen to rise (a reading of a segment whose bottom is at or above an earlier
  reading's top: a stuck or unread count proves nothing). Not classic: none lies in 2.38-3.22
  s (at least MIN_READINGS readings and a rise). Sparse frames (a replay with an image every 2 s)
  give a wider range, still enough to tell 1x from 2x: the elixir staying on one count for more
  than 1.61 s already rules out double elixir.

The verdict (ModeCheck.update, once per battle; it latches): "classic", or "unknown" with why:
the clock ("first clock 2:00"), the elixir ("regen 1.4 s", "elixir readings do not fit one
regeneration rate", "too few elixir readings", "the elixir did not rise", "regen unclear
(1.2-3.0 s)"). The clock decides at once; the elixir as soon as its readings do (about 3 s into
a battle at the live frame rate: the first count change after 1.6 s on one count), at the
latest when the window ends. An elixir "not classic" must hold on every fit for NOT_CLASSIC_HOLD_S
(1.5) s first (or until the window ends): a misread the fit cannot drop yet is no game mode, and
the gate stays shut (pending) meanwhile.

Standard library and numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

TICKS_PER_S = 20
REGULATION_S = 180
#: per elixir, s (hud.REGEN_MS_1X / _2X / _3X)
REGEN_S_1X, REGEN_S_2X, REGEN_S_3X = 2.8, 1.4, 0.93
REGEN_TOL = 0.15
CLASSIC_REGEN_S = (REGEN_S_1X * (1.0 - REGEN_TOL), REGEN_S_1X * (1.0 + REGEN_TOL))    # 2.38, 3.22
FAST_REGEN_MAX_S = REGEN_S_2X * (1.0 + REGEN_TOL)                                        # 1.61
#: The first clock reading of a classic battle shows 2:5x-3:00: ClockFilter's tick for 2:50 is
#: 200-219 (its first reading of a battle is the display's base + 10).
FIRST_TICK_MAX = 219
WINDOW_S = 10.0
#: elixir: the bar's fill against the count (measured: 0.2-0.25). A reading outside it is one of the
#: misreads the outlier allowance and the hold below absorb; a wider tolerance (0.4) was more robust
#: to heavy bar noise but needed ~1.7 s more readings on the sparse 10-31 dry run (classic at 9.6 s
#: of the 10-s window, bar only), so it stays.
BAR_TOL = 0.3
MIN_READINGS = 3
OUTLIER_SHARE = 0.05
OUTLIER_MIN, OUTLIER_FROM = 1, 5   # at least 1 outlier allowed in a segment of 5 readings or more
NOT_CLASSIC_HOLD_S = 1.5           # an elixir "not classic" decides once it held this long
CUT_DROP = 0.5                     # the bar's value falling this much (elixir) between readings: a play
SAMPLE_DT_S = 0.05             # at most one reading per 0.05 s is kept (the live loop runs ~50 fps)
EVAL_EVERY_S = 0.2             # the readings are fitted at most every 0.2 s
PERIODS_S = np.round(np.arange(0.5, 6.0 + 1e-9, 0.02), 2)

CLASSIC, UNKNOWN = "classic", "unknown"


def clock_text(tick: int | None) -> str:
    """The regulation clock shown at `tick` ("2:59"; "?" for None, "OT" past regulation)."""
    if tick is None:
        return "?"
    s = REGULATION_S - int(tick) // TICKS_PER_S
    if s < 0:
        return "OT"
    return f"{s // 60}:{s % 60:02d}"


def clock_ok(first_tick: int | None, *, first_tick_max: int = FIRST_TICK_MAX) -> tuple[bool | None, str]:
    """(classic?, why) of the battle's first clock reading: None when there is none."""
    if first_tick is None:
        return None, "no clock reading"
    if 0 <= int(first_tick) <= int(first_tick_max):
        return True, f"first clock {clock_text(first_tick)}"
    return False, f"first clock {clock_text(first_tick)} (tick {int(first_tick)}), not 2:5x-3:00"


def elixir_interval(digit: int | None, bar_frac: float | None, *,
                    bar_tol: float = BAR_TOL) -> tuple[float, float] | None:
    """[lo, hi) the elixir lies in from one frame's count digit and bar fill (module doc), or
    None with neither. A full elixir (10, or a full bar) has no top (hi = inf)."""
    lo, hi = -math.inf, math.inf
    if digit is not None and 0 <= int(digit) <= 10:
        d = int(digit)
        lo, hi = float(d), (float(d + 1) if d < 10 else math.inf)
    if bar_frac is not None and math.isfinite(float(bar_frac)):
        e = 10.0 * float(bar_frac)
        blo, bhi = e - bar_tol, (math.inf if float(bar_frac) >= 0.999 else e + bar_tol)
        nlo, nhi = max(lo, blo), min(hi, bhi)
        if nlo >= nhi:
            return None                                  # they disagree: one is misread
        lo, hi = nlo, nhi
    if lo == -math.inf and hi == math.inf:
        return None
    return lo, hi


def _cut(prev: Sequence[float], cur: Sequence[float]) -> bool:
    """A card was played between two readings: `cur`'s top is under `prev`'s bottom, or the bar's
    own value (a reading's 4th element) fell by CUT_DROP or more (a 1-elixir play the wide bar
    tolerance would otherwise hide)."""
    if cur[2] <= prev[1]:
        return True
    p0 = prev[3] if len(prev) > 3 else None
    p1 = cur[3] if len(cur) > 3 else None
    return p0 is not None and p1 is not None and float(p0) - float(p1) >= CUT_DROP


def segments(samples: Sequence[tuple[float, ...]]) -> list[list[tuple[float, ...]]]:
    """(t, lo, hi[, bar point]) readings in time order, cut where one lies clearly below the one
    before it (_cut: a card was played)."""
    out: list[list[tuple[float, ...]]] = []
    for s in samples:
        if out and out[-1] and _cut(out[-1][-1], s):
            out.append([s])
        elif out:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def _rose(seg: Sequence[tuple[float, ...]]) -> bool:
    """Some reading's bottom is at or above an earlier reading's top: the elixir rose."""
    low_top = math.inf
    for r in seg:
        lo, hi = r[1], r[2]
        if lo >= low_top:
            return True
        low_top = min(low_top, hi)
    return False


def _consistent(seg: Sequence[tuple[float, ...]], periods: np.ndarray, allowed: int) -> np.ndarray:
    """Per period, whether one c fits c + t / P into every reading's [lo, hi) but at most
    `allowed` of them (dropping the readings with the highest bottoms or the lowest tops: a
    sufficient test)."""
    t = np.array([s[0] for s in seg], dtype=float)
    t -= t[0]
    lo = np.array([s[1] for s in seg], dtype=float)
    hi = np.array([s[2] for s in seg], dtype=float)
    r = 1.0 / periods[:, None]
    a = lo[None, :] - r * t[None, :]                 # c >= a
    b = hi[None, :] - r * t[None, :]                 # c < b (inf: no top)
    n = len(seg)
    if n <= allowed + 1:
        return np.ones(len(periods), dtype=bool)
    a_desc = -np.sort(-a, axis=1)
    b_asc = np.sort(b, axis=1)
    ok = np.zeros(len(periods), dtype=bool)
    for i in range(allowed + 1):
        for j in range(allowed + 1 - i):
            ok |= a_desc[:, i] < b_asc[:, j]
    return ok


@dataclass(frozen=True)
class RegenFit:
    """What the elixir readings say (regen_fit): n readings in `segments` segments, whether the
    elixir rose, the range of regeneration periods consistent with them (s; None when none is,
    or with fewer than 2 readings in every segment), and a point estimate inside it (estimate_s:
    the least-squares slope of the readings' midpoints, one intercept per segment; None without
    a rising slope)."""
    n: int
    segments: int
    rose: bool
    lo_s: float | None
    hi_s: float | None
    constrained: bool                 # a segment had 2 readings or more
    estimate_s: float | None = None

    @property
    def fits(self) -> bool:
        return self.lo_s is not None

    def regen_text(self) -> str:
        """The estimate ('2.8 s'), else the range ('2.5-3.4 s', '>= 2.1 s' when open above)."""
        if self.lo_s is None:
            return "?"
        if self.estimate_s is not None:
            return f"{self.estimate_s:.1f} s"
        if self.hi_s >= float(PERIODS_S[-1]) - 1e-9:
            return f">= {self.lo_s:.1f} s"
        return f"{self.lo_s:.1f}-{self.hi_s:.1f} s"

    def to_dict(self) -> dict[str, Any]:
        return {"readings": self.n, "segments": self.segments, "rose": self.rose,
                "regen_s": None if self.lo_s is None else [round(self.lo_s, 2), round(self.hi_s, 2)],
                "estimate_s": None if self.estimate_s is None else round(self.estimate_s, 2)}


def _point(r: Sequence[float]) -> float | None:
    """A reading's best single value: the bar's own (its 4th element), else its interval's middle."""
    if len(r) > 3 and r[3] is not None:
        return float(r[3])
    return (r[1] + r[2]) / 2.0 if math.isfinite(r[1]) and math.isfinite(r[2]) else None


def _slope_estimate(segs: Sequence[Sequence[Sequence[float]]]) -> float | None:
    """The period (s) of the pooled least-squares slope of the readings' points (the bar's value,
    else the interval's middle; a full elixir left out), one intercept per segment; None when
    there is no rising slope."""
    num = den = 0.0
    for seg in segs:
        pts = [(r[0], _point(r)) for r in seg if math.isfinite(r[2]) and _point(r) is not None]
        if len(pts) < 2:
            continue
        tm = sum(p[0] for p in pts) / len(pts)
        mm = sum(p[1] for p in pts) / len(pts)
        num += sum((p[0] - tm) * (p[1] - mm) for p in pts)
        den += sum((p[0] - tm) ** 2 for p in pts)
    if den <= 0.0 or num <= 0.0:
        return None
    return den / num


def outliers_allowed(n: int, share: float = OUTLIER_SHARE) -> int:
    """The misreads a segment of n readings may hold (module doc)."""
    allowed = int(n * float(share))
    return max(OUTLIER_MIN, allowed) if n >= OUTLIER_FROM else allowed


def regen_fit(samples: Iterable[tuple[float, ...]], *, periods: np.ndarray = PERIODS_S,
              outlier_share: float = OUTLIER_SHARE) -> RegenFit:
    """The regeneration periods consistent with (t, lo, hi) elixir readings (module doc)."""
    samples = sorted(samples, key=lambda s: s[0])
    segs = segments(samples)
    ok = np.ones(len(periods), dtype=bool)
    constrained = False
    for seg in segs:
        if len(seg) < 2:
            continue
        constrained = True
        ok &= _consistent(seg, periods, outliers_allowed(len(seg), outlier_share))
    rose = any(_rose(seg) for seg in segs)
    if not constrained or not ok.any():
        return RegenFit(len(samples), len(segs), rose, None, None, constrained)
    good = periods[ok]
    lo, hi = float(good.min()), float(good.max())
    est = _slope_estimate(segs)
    if est is not None:
        est = min(max(est, lo), hi)
    return RegenFit(len(samples), len(segs), rose, lo, hi, constrained, est)


def regen_classic(fit: RegenFit, *, classic_s: Sequence[float] = CLASSIC_REGEN_S,
                  fast_max_s: float = FAST_REGEN_MAX_S, min_readings: int = MIN_READINGS
                  ) -> tuple[bool | None, str]:
    """(classic?, why) of a RegenFit: True / False when the readings decide (module doc), None
    while they do not (yet)."""
    if fit.n < min_readings or not fit.constrained:
        return None, "too few elixir readings"
    if not fit.rose:
        return None, "the elixir did not rise"
    if not fit.fits:
        return None, "elixir readings do not fit one regeneration rate"
    in_classic = fit.lo_s <= classic_s[1] and fit.hi_s >= classic_s[0]
    if not in_classic:
        return False, f"regen {fit.regen_text()}"
    if fit.lo_s > fast_max_s:
        return True, f"elixir every {fit.regen_text()}"
    return None, f"regen unclear ({fit.regen_text()})"


@dataclass(frozen=True)
class ModeVerdict:
    """The battle's game mode: CLASSIC or UNKNOWN, why (the clock's and the elixir's words), when
    it was decided (frame time and tick), the first clock reading and the elixir fit."""
    mode: str
    why: str
    t: float
    tick: int | None
    first_tick: int | None
    fit: RegenFit | None

    @property
    def classic(self) -> bool:
        return self.mode == CLASSIC

    def text(self) -> str:
        """'mode classic (first clock 2:59, elixir every 2.8 s)' / 'mode unknown (regen 1.4 s)'."""
        return f"mode {self.mode} ({self.why})"

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "why": self.why, "t": round(self.t, 3), "tick": self.tick,
                "first_tick": self.first_tick, "first_clock": clock_text(self.first_tick),
                **({"elixir": self.fit.to_dict()} if self.fit is not None else {})}


class ModeCheck:
    """One battle's game-mode check (module doc). start() at the battle's start with its first
    clock reading's tick and the frame time; note() every battle frame with the HUD's raw elixir
    reading (hud.HudReport.elixir_read's digit and bar fill: never the filtered elixir, which
    extrapolates at the classic rate); update() decides (once, then the verdict stays)."""

    def __init__(self, *, window_s: float = WINDOW_S, first_tick_max: int = FIRST_TICK_MAX,
                 bar_tol: float = BAR_TOL, min_readings: int = MIN_READINGS,
                 sample_dt_s: float = SAMPLE_DT_S, eval_every_s: float = EVAL_EVERY_S,
                 hold_s: float = NOT_CLASSIC_HOLD_S):
        self.window_s = float(window_s)
        self.hold_s = float(hold_s)
        self.first_tick_max = int(first_tick_max)
        self.bar_tol = float(bar_tol)
        self.min_readings = max(2, int(min_readings))
        self.sample_dt_s = float(sample_dt_s)
        self.eval_every_s = float(eval_every_s)
        self.reset()

    def reset(self) -> None:
        self.t0: float | None = None
        self.first_tick: int | None = None
        self.samples: list[tuple[float, ...]] = []
        self.verdict: ModeVerdict | None = None
        self.fit: RegenFit | None = None
        self._fit_t: float | None = None
        self._fit_n = -1
        self._against_t: float | None = None      # since when the elixir says "not classic"

    def start(self, first_tick: int | None, t: float) -> None:
        self.reset()
        self.first_tick = None if first_tick is None else int(first_tick)
        self.t0 = float(t)

    @property
    def started(self) -> bool:
        return self.t0 is not None

    def note(self, t: float, digit: int | None, bar_frac: float | None) -> None:
        """One frame's raw elixir reading (inside the window; ignored once decided)."""
        if self.verdict is not None or self.t0 is None or float(t) - self.t0 > self.window_s:
            return
        iv = elixir_interval(digit, bar_frac, bar_tol=self.bar_tol)
        if iv is None:
            return
        if self.samples and float(t) - self.samples[-1][0] < self.sample_dt_s:
            last = self.samples[-1]
            if iv[1] > last[1] and iv[0] < last[2]:      # the same elixir: one reading is enough
                return
        point = None
        if bar_frac is not None and math.isfinite(float(bar_frac)):
            point = min(max(10.0 * float(bar_frac), iv[0]), iv[1] if math.isfinite(iv[1]) else math.inf)
        self.samples.append((float(t), iv[0], iv[1], point))

    def update(self, t: float, tick: int | None = None) -> ModeVerdict | None:
        """The verdict, deciding it when the clock or the readings allow (or the window ended);
        None while undecided."""
        if self.verdict is not None or self.t0 is None:
            return self.verdict
        t = float(t)
        ok, clock_why = clock_ok(self.first_tick, first_tick_max=self.first_tick_max)
        if ok is False:
            return self._decide(UNKNOWN, clock_why, t, tick)
        ended = t - self.t0 >= self.window_s
        due = (self._fit_t is None or t - self._fit_t >= self.eval_every_s or ended)
        if due and len(self.samples) != self._fit_n:
            self.fit = regen_fit(self.samples)
            self._fit_t, self._fit_n = t, len(self.samples)
        elif self.fit is None:
            return None
        regen, regen_why = regen_classic(self.fit, min_readings=self.min_readings)
        if regen is False:                             # it must hold (module doc)
            if self._against_t is None:
                self._against_t = t
            if ended or t - self._against_t >= self.hold_s:
                return self._decide(UNKNOWN, regen_why, t, tick)
            return None
        self._against_t = None
        if ok is None:
            if ended:
                return self._decide(UNKNOWN, clock_why, t, tick)
            return None
        if regen is True:
            return self._decide(CLASSIC, f"{clock_why}, {regen_why}", t, tick)
        if ended:
            return self._decide(UNKNOWN, regen_why, t, tick)
        return None

    def _decide(self, mode: str, why: str, t: float, tick: int | None) -> ModeVerdict:
        self.verdict = ModeVerdict(mode, why, t, None if tick is None else int(tick), self.first_tick, self.fit)
        return self.verdict

    def summary(self) -> dict[str, Any]:
        """The battle_end record's view: the verdict, or what is known so far."""
        if self.verdict is not None:
            return self.verdict.to_dict()
        out: dict[str, Any] = {"mode": None, "first_tick": self.first_tick,
                               "first_clock": clock_text(self.first_tick), "readings": len(self.samples)}
        if self.fit is not None:
            out["elixir"] = self.fit.to_dict()
        return out
