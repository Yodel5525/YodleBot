"""The game-mode check of the live runner's tap gate (scripts/vision/mode.py, skin audit P1-12).

The bot's timings, its elixir model and its training are classic 1v1: a 3:00 clock and one elixir
every 2.8 s. So each battle's first clock reading must show 2:5x-3:00, and our elixir must fill at
1x (2.8 s +-15%) while no card is played; double (1.4 s) and triple elixir (0.93 s) are refused.
The elixir readings are made up here: the count digit (floor of the elixir) and the bar's fill
(the elixir / 10), at the live frame rate and at a sparse replay's one image every 2 s.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import hud as H  # noqa: E402
from scripts.vision import mode as M  # noqa: E402

READINGS = {"digit+bar": (True, True), "digit": (True, False), "bar": (False, True)}
RATES = {"1x": M.REGEN_S_1X, "2x": M.REGEN_S_2X, "3x": M.REGEN_S_3X}


def first_reading_tick(seconds_left: int) -> int:
    """The tick hud.ClockFilter gives a battle's first clock reading (its base + 10)."""
    f = H.ClockFilter()
    return f.update(seconds_left, 0.0).tick


def battle(period: float, *, fps: float = 10.0, digit: bool = True, bar: bool = True, e0: float = 5.0,
           start: float = 1.0, first_tick: int = 30, seconds: float = 12.0, plays=(), misread=None,
           stop_at_verdict: bool = True):
    """Runs a ModeCheck over a battle whose elixir starts at e0 at t=0 and fills one per `period`
    s; the check starts at `start` (the runner's battle start). plays: (t, cost) spent; misread:
    {frame index: digit} shown instead of the count. Returns (check, verdict, s from start)."""
    mc = M.ModeCheck()
    mc.start(first_tick, start)
    spent, i, t = 0.0, 0, start
    verdict = None
    while t <= start + seconds + 1e-9:
        spent = sum(c for tp, c in plays if tp <= t)
        e = min(10.0, e0 + t / period - spent)
        d = int(math.floor(e)) if digit else None
        if misread and i in misread:
            d = misread[i]
        mc.note(t, d, e / 10.0 if bar else None)
        verdict = mc.update(t, int(round(t * 20)))
        if verdict is not None and stop_at_verdict:
            return mc, verdict, round(t - start, 3)
        i += 1
        t = round(start + i / fps, 6)
    return mc, verdict, None if verdict is None else round(verdict.t - start, 3)


# ---------------------------------------------------------------------------
# the elixir
# ---------------------------------------------------------------------------
def test_an_elixir_reading_is_the_interval_the_digit_and_the_bar_allow():
    assert M.elixir_interval(6, None) == (6.0, 7.0)
    assert M.elixir_interval(10, None) == (10.0, math.inf)
    lo, hi = M.elixir_interval(6, 0.64)
    assert (round(lo, 3), round(hi, 3)) == (6.1, 6.7)
    lo, hi = M.elixir_interval(6, 0.61)
    assert (round(lo, 3), round(hi, 3)) == (6.0, 6.4)
    assert M.elixir_interval(None, 0.5) == pytest.approx((4.7, 5.3))
    # They disagree: one of them is misread, the frame gives no reading (the review's misread
    # digit kept against its bar cut the readings and closed the gate in a classic battle).
    assert M.elixir_interval(6, 0.9) is None and M.elixir_interval(7, 0.55) is None
    assert M.elixir_interval(None, 1.0)[1] == math.inf
    assert M.elixir_interval(None, None) is None and M.elixir_interval(11, None) is None


@pytest.mark.parametrize("fps", [10.0, 2.0, 0.5])
@pytest.mark.parametrize("reading", sorted(READINGS))
@pytest.mark.parametrize("rate", sorted(RATES))
def test_only_1x_regeneration_is_classic(rate, reading, fps):
    """Elixir readings at 1x, 2x and 3x regeneration, at 10 frames a second, 2, and one image every
    2 s: 1x is classic, the others are not, within the 10-s window (live: classic within 2.5 s;
    not classic within 3 s plus NOT_CLASSIC_HOLD_S, the time a "not classic" must hold)."""
    digit, bar = READINGS[reading]
    mc, verdict, after = battle(RATES[rate], fps=fps, digit=digit, bar=bar)
    assert verdict is not None
    assert verdict.classic is (rate == "1x"), verdict.text()
    assert after <= M.WINDOW_S
    if fps >= 10.0:
        assert after <= (2.5 if rate == "1x" else 3.0 + M.NOT_CLASSIC_HOLD_S)
    if rate == "1x":
        assert verdict.why.startswith("first clock 2:59, elixir every ")
    else:
        assert verdict.why.startswith("regen ")
        est = mc.fit.estimate_s
        assert est is not None and est <= M.FAST_REGEN_MAX_S, verdict.why


def test_the_regeneration_estimate_names_the_rate():
    """The verdict says how fast the elixir filled: '1.4 s' for double elixir, '0.9 s' for triple."""
    _, two, _ = battle(M.REGEN_S_2X)
    _, three, _ = battle(M.REGEN_S_3X, digit=False)
    assert two.text() == "mode unknown (regen 1.4 s)"
    assert three.text() == "mode unknown (regen 0.9 s)"
    mc, one, _ = battle(M.REGEN_S_1X, digit=False, stop_at_verdict=False)
    assert one.classic and 2.6 <= mc.fit.estimate_s <= 3.0
    assert one.to_dict()["elixir"]["estimate_s"] == pytest.approx(mc.fit.estimate_s, abs=0.01)


@pytest.mark.parametrize("rate", ["1x", "2x"])
def test_plays_cut_the_readings_and_do_not_change_the_verdict(rate):
    """Cards played in the window (a Hog Rider, Ice Spirits, the Log): the elixir falls, the readings
    are cut there, and each stretch still fills at the battle's rate."""
    plays = [(1.6, 4.0), (2.4, 1.0), (3.1, 2.0)]
    mc, verdict, _ = battle(RATES[rate], e0=8.0, plays=plays, fps=10.0)
    assert verdict is not None and verdict.classic is (rate == "1x"), verdict.text()
    assert mc.fit.segments >= 2


def test_a_misread_digit_is_tolerated():
    """One count misread (6 shown as 8) among the live frames: still classic."""
    mc, verdict, _ = battle(M.REGEN_S_1X, fps=10.0, bar=False, misread={25: 8}, seconds=8.0)
    assert verdict is not None and verdict.classic, verdict.text()


@pytest.mark.parametrize("bar", [True, False], ids=["digit+bar", "digit"])
@pytest.mark.parametrize("frame", [2, 5, 8, 12, 20])
@pytest.mark.parametrize("shift", [1, -1])
def test_one_misread_count_in_the_first_seconds_is_tolerated(shift, frame, bar):
    """The review's case: ONE count digit read 1 off on one of the first frames (before the old
    5% outlier share allowed any). With the bar, the frame disagrees with it and gives no reading;
    digit only, the segment it ends allows one outlier from 5 readings on, and a "not classic"
    fit must hold 1.5 s. Classic either way."""
    def misread(i, e0=5.0, fps=10.0, start=1.0):
        return int(math.floor(min(10.0, e0 + (start + i / fps) / M.REGEN_S_1X))) + shift
    mc, verdict, _ = battle(M.REGEN_S_1X, fps=10.0, bar=bar, misread={frame: misread(frame)})
    assert verdict is not None and verdict.classic, verdict.text()


@pytest.mark.parametrize("at", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("err", [-0.6, -0.4, -0.35, 0.35, 0.6])
def test_one_bar_reading_off_is_tolerated(err, at):
    """ONE bar reading 0.35-0.6 elixir off (the digit right), 8 frames a second: classic (it was
    "regen 3.3-3.9 s" for a reading 0.35-0.4 low at 0.5-1.5 s)."""
    mc = M.ModeCheck()
    mc.start(10, 0.0)
    verdict, t = None, 0.0
    while t <= 10.5 and verdict is None:
        e = 5.0 + t / M.REGEN_S_1X
        mc.note(t, int(math.floor(e)), (e + (err if abs(t - at) < 0.06 else 0.0)) / 10.0)
        verdict = mc.update(t, int(t * 20))
        t += 0.125
    assert verdict is not None and verdict.classic, verdict.text()


@pytest.mark.parametrize("seed", range(10))
def test_a_bar_reading_a_quarter_low_with_noise_is_classic(seed):
    """A bar 0.25 elixir low with +-0.1 of noise (its measured error at worst, plus noise): a
    quarter of its readings miss the true elixir by up to 0.05; the outliers allowed and the hold
    keep 1x classic (it was "regen 3.3-3.9 s" on 7 seeds in 10)."""
    import random
    rnd = random.Random(seed)
    mc = M.ModeCheck()
    mc.start(10, 0.0)
    verdict, t = None, 0.0
    while t <= 10.5 and verdict is None:
        e = 5.0 + t / M.REGEN_S_1X
        mc.note(t, int(math.floor(e)), min(1.0, max(0.0, (e - 0.25 + rnd.uniform(-0.1, 0.1)) / 10.0)))
        verdict = mc.update(t, int(t * 20))
        t += 0.125
    assert verdict is not None and verdict.classic, verdict.text()


@pytest.mark.parametrize("rate", ["2x", "3x"])
def test_not_classic_must_hold_before_it_decides(rate):
    """A fit that says "not classic" decides only once it held NOT_CLASSIC_HOLD_S (fail-closed:
    the gate stays pending meanwhile); double and triple elixir are still refused well within
    the window."""
    mc = M.ModeCheck()
    mc.start(30, 0.0)
    first_against, verdict, t = None, None, 0.0
    while verdict is None and t <= M.WINDOW_S:
        e = 5.0 + t / RATES[rate]
        mc.note(t, int(math.floor(e)), e / 10.0)
        verdict = mc.update(t, int(t * 20))
        if first_against is None and mc.fit is not None and M.regen_classic(mc.fit)[0] is False:
            first_against = t
        t = round(t + 0.1, 6)
    assert verdict is not None and not verdict.classic and verdict.why.startswith("regen ")
    assert verdict.t - first_against >= M.NOT_CLASSIC_HOLD_S - 1e-6
    assert verdict.t <= 3.0 + M.NOT_CLASSIC_HOLD_S


def test_a_bar_falling_half_an_elixir_cuts_the_readings():
    """A 1-elixir play seen through the bar as a fall of 0.55 (its intervals still overlap): a cut,
    so the stretch before the play does not have to fit the one after."""
    a = (0.0, *M.elixir_interval(None, 0.56), 5.6)
    b = (0.8, *M.elixir_interval(None, 0.505), 5.05)
    c = (1.6, *M.elixir_interval(None, 0.535), 5.35)
    assert b[2] > a[1]                                                   # no cut by the intervals alone
    assert [len(x) for x in M.segments([a, b, c])] == [1, 2]
    assert [len(x) for x in M.segments([a[:3], b[:3], c[:3]])] == [3]      # no bar values: no cut


def test_a_segment_allows_one_outlier_from_five_readings():
    assert [M.outliers_allowed(n) for n in (2, 4, 5, 19, 20, 40)] == [0, 0, 1, 1, 1, 2]


def test_a_stuck_or_missing_elixir_is_not_classic_when_the_window_ends():
    """A count that never changes proves nothing about the rate, nor do no readings at all: at the
    window's end the mode is unknown, with why."""
    mc = M.ModeCheck()
    mc.start(30, 0.0)
    for i in range(101):
        mc.note(i * 0.1, 7, None)
        v = mc.update(i * 0.1)
    assert v is not None and v.text() == "mode unknown (the elixir did not rise)"
    mc = M.ModeCheck()
    mc.start(30, 0.0)
    for i in range(101):
        mc.note(i * 0.1, None, None)
        v = mc.update(i * 0.1)
    assert v is not None and v.text() == "mode unknown (too few elixir readings)"
    _, full, _ = battle(M.REGEN_S_1X, e0=10.0)
    assert full is not None and not full.classic


def test_nothing_is_decided_before_the_readings_allow():
    """Two frames 0.1 s apart say nothing about the rate yet."""
    mc = M.ModeCheck()
    mc.start(30, 0.0)
    mc.note(0.0, 5, 0.5)
    mc.note(0.1, 5, 0.503)
    assert mc.update(0.1) is None and mc.verdict is None
    assert mc.summary()["mode"] is None and mc.summary()["first_clock"] == "2:59"


# ---------------------------------------------------------------------------
# the clock
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shown, classic", [(180, True), (179, True), (175, True), (170, True),
                                            (169, False), (120, False), (60, False)])
def test_the_first_clock_reading_must_show_2_5x_to_3_00(shown, classic):
    """A clock that starts at 2:00 or 1:00 (or a battle joined late) is not classic, whatever the
    elixir does; 3:00-2:50 is."""
    tick = first_reading_tick(shown)
    ok, why = M.clock_ok(tick)
    assert ok is classic, why
    mc, verdict, after = battle(M.REGEN_S_1X, first_tick=tick)
    assert verdict.classic is classic
    if not classic:
        assert after == 0.0                                       # decided on the first frame
        assert verdict.why.startswith(f"first clock {shown // 60}:{shown % 60:02d} (tick {tick})")
        assert verdict.why.endswith("not 2:5x-3:00")


def test_clock_text():
    assert M.clock_text(0) == "3:00" and M.clock_text(30) == "2:59" and M.clock_text(1210) == "2:00"
    assert M.clock_text(None) == "?" and M.clock_text(3700) == "OT"


def test_the_verdict_latches_and_reads_as_a_record():
    mc, verdict, _ = battle(M.REGEN_S_1X)
    assert mc.update(99.0) is verdict
    rec = verdict.to_dict()
    assert rec["mode"] == "classic" and rec["first_tick"] == 30 and rec["first_clock"] == "2:59"
    assert rec["elixir"]["rose"] is True and rec["elixir"]["readings"] >= 3
    assert mc.summary() == rec
    mc.start(1210, 0.0)                                           # a new battle starts again
    assert mc.verdict is None and mc.samples == []
