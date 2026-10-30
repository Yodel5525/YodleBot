"""TowerHpTracker on synthetic reading sequences (scripts/vision/hud.py, vision-09).

The sequences reproduce what the owner's 2026-10-29 friendly showed through
tools/vision_replay.py: battle-screen frames before the clock was read (the
intro) read the opponent's full right tower (3052) as 855 and then 577, which
the old tracker kept for 2.5 minutes; later the digits read 1667 as 3667 on two
frames in a row, which the old tracker took as the new maximum. Frames are
DT = 0.125 s apart (exact in binary, so TowerHpFilter's 0.6 s hold ends on the
6th frame of a run).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import hud as H  # noqa: E402
from scripts.vision import templates as TP  # noqa: E402
from scripts.vision import types as VT  # noqa: E402

K = "enemy_right"
DT = 0.125
PRIOR = H.LEVEL11_TOWER_MAX_HP


def _at(v, i):
    return v[i] if isinstance(v, (list, tuple)) else v


def feed(tr: H.TowerHpTracker, reads, t0: float = 0.0, *, key: str = K, clock=None, bars=None,
         confs=None) -> list[VT.TowerHp]:
    """One update per reading (an HP or None), DT apart from t0; `clock`, `bars` and
    `confs` are one value for every frame or a list. Returns `key`'s TowerHp per frame."""
    out = []
    for i, hp in enumerate(reads):
        bar, conf = _at(bars, i), _at(confs, i)
        out.append(tr.update({key: hp}, t0 + i * DT, None if bar is None else {key: bar},
                             hp_conf=None if conf is None else {key: conf},
                             clock_running=_at(clock, i))[key])
    return out


def tracker(**kw) -> H.TowerHpTracker:
    return H.TowerHpTracker(prior_max_hp=PRIOR, **kw)


def all_are(out: list, want: VT.TowerHp) -> bool:
    """Every TowerHp in `out` equals `want` (TowerHp is not hashable: no sets)."""
    return len(out) > 0 and out == [want] * len(out)


# ---------------------------------------------------------------------------
# The legal maxima
# ---------------------------------------------------------------------------
def test_tower_hp_table_is_royalesims_tower_ladder():
    def pct(level: int, rate: int) -> int:          # calibration combat.TOWER_HITPOINT_LADDER
        p = 100
        for lv in range(2, level + 1):
            p = p * (100 + (rate if lv <= 9 else 10)) // 100
        return p
    assert H.TOWER_HP_BY_LEVEL == {lv: (1400 * pct(lv, 8) // 100, 2400 * pct(lv, 7) // 100)
                                   for lv in range(1, 17)}
    # the measured rows (king 3312 / 4824, princess 2030 / 3052 at 6 / 11) and RoyaleSim's
    # tests/levels.rs king table
    assert H.TOWER_HP_BY_LEVEL[6] == (2030, 3312) and H.TOWER_HP_BY_LEVEL[11] == (3052, 4824)
    assert [H.TOWER_HP_BY_LEVEL[lv][1] for lv in range(1, 16)] == [
        2400, 2568, 2736, 2904, 3096, 3312, 3528, 3768, 4008, 4392, 4824, 5304, 5832, 6408, 7032]
    assert PRIOR == {k: H.TOWER_HP_BY_LEVEL[11][1 if k.endswith("king") else 0] for k in VT.TOWER_KEYS}
    legal = H.LEGAL_TOWER_MAX_HP
    assert set(legal) == set(VT.TOWER_KEYS)
    assert 3052 in legal[K] and 3346 in legal[K] and 3668 in legal[K]
    assert 3667 not in legal[K] and 4824 not in legal[K]          # the misread; a king's value
    assert 4824 in legal["own_king"] and 3052 not in legal["own_king"]


# ---------------------------------------------------------------------------
# Item 1: the intro
# ---------------------------------------------------------------------------
def test_intro_misreads_before_the_clock_runs_do_not_count():
    tr = tracker()
    out = feed(tr, [855] * 8 + [577] * 8, clock=False)          # the intro: the clock is not read yet
    assert all_are(out, VT.TowerHp(3052, 3052))                 # full at the prior max
    assert tr.frac[K] is None and not tr.started
    assert tr.update({}, 2.0, clock_running=False) == {k: VT.TowerHp(m, m) for k, m in PRIOR.items()}
    out = feed(tr, [3052] * 8, 2.125, clock=True)               # the clock runs: the true 3052
    assert all_are(out, VT.TowerHp(3052, 3052)) and tr.started and tr.frac[K] == 1.0
    # it latches: a fall while the clock is unread for a moment is still followed, on time
    out = feed(tr, [2101] * 6, 3.125, clock=False)
    assert out[4] == VT.TowerHp(3052, 3052) and out[5] == VT.TowerHp(2101, 3052)
    tr.reset()                                                  # a new battle waits for its clock again
    assert not tr.started and feed(tr, [855] * 8, clock=False)[-1] == VT.TowerHp(3052, 3052)
    nothing = H.TowerHpTracker()                                # without a prior: unknown, not full
    assert nothing.update({K: 855}, 0.0, clock_running=False)[K] == VT.TowerHp(None, None)


def test_a_value_misread_too_low_is_recovered():
    # a caller that does not follow the clock (clock_running None): the intro misreads count...
    tr = tracker()
    out = feed(tr, [855] * 6 + [577] * 6)
    assert out[5] == VT.TowerHp(855, 3052) and out[11] == VT.TowerHp(577, 3052)
    # ...until 5 confident readings agree on a higher value <= max: the filtered HP is reset
    out = feed(tr, [3052] * 5, 1.5)
    assert [h.hp for h in out] == [577, 577, 577, 577, 3052] and tr.recoveries == 1
    assert out[-1] == VT.TowerHp(3052, 3052) and tr.frac[K] == 1.0
    out = feed(tr, [2101] * 6, 2.125)                           # and a real fall is followed after it
    assert out[4].hp == 3052 and out[5] == VT.TowerHp(2101, 3052)


def low_tower(**kw) -> H.TowerHpTracker:
    """A tracker whose enemy_right was accepted at a misread 577 (frames 0..0.625 s)."""
    tr = tracker(**kw)
    assert feed(tr, [577] * 6)[-1] == VT.TowerHp(577, 3052)
    return tr


def test_the_recovery_needs_consecutive_confident_agreeing_readings():
    T0 = 0.75
    tr = low_tower()                         # a reading at or below the filtered HP breaks the run
    assert {h.hp for h in feed(tr, [3052] * 4 + [577] + [3052] * 4, T0)} == {577}
    assert feed(tr, [3052], T0 + 9 * DT)[0] == VT.TowerHp(3052, 3052)
    tr = low_tower()                         # readings under the confidence floor never reset it
    assert {h.hp for h in feed(tr, [3052] * 10, T0, confs=0.7)} == {577} and tr.recoveries == 0
    tr = low_tower()                         # a frame without digits leaves the run as it is
    out = feed(tr, [3052] * 3 + [None] * 2 + [3052] * 2, T0)
    assert [h.hp for h in out] == [577] * 6 + [3052]
    tr = low_tower()                         # readings above the max are misreads: no reset
    assert all_are(feed(tr, [3667] * 10, T0), VT.TowerHp(577, 3052))
    assert tr.max[K] == 3052 and tr.recoveries == 0
    tr = low_tower()                         # the run keeps its newest readings that agree
    out = feed(tr, [3052] * 3 + [2700] * 5, T0)
    assert [h.hp for h in out] == [577] * 7 + [2700]


def test_the_recovery_must_agree_with_the_bar():
    T0 = 0.75
    tr = low_tower()                         # the bar says 0.19 too: the digits are the misread
    assert {h.hp for h in feed(tr, [3052] * 10, T0, bars=0.19)} == {577} and tr.recoveries == 0
    tr = low_tower()                         # within bar_disagree (0.25) but not recover_bar_tol (0.1)
    assert {h.hp for h in feed(tr, [3052] * 10, T0, bars=0.8)} == {577} and tr.recoveries == 0
    tr = low_tower()                         # the bar agrees: reset on the 5th reading
    assert [h.hp for h in feed(tr, [3052] * 5, T0, bars=1.0)] == [577] * 4 + [3052]
    tr = low_tower()                         # measure_tower_bar's 0.0 is no bar seen: the digits alone
    assert [h.hp for h in feed(tr, [3052] * 5, T0, bars=0.0)] == [577] * 4 + [3052]


# ---------------------------------------------------------------------------
# Item 2: 1667 read as 3667
# ---------------------------------------------------------------------------
def test_a_higher_reading_after_damage_is_never_a_new_max():
    tr = tracker()
    feed(tr, [3052] * 4, clock=True)
    seq = [2101] * 6 + [1884] * 6 + [1667, 3667, 3667] + [1667] * 6
    out = feed(tr, seq, 0.5, clock=True)
    assert {h.max for h in out} == {3052} and tr.max[K] == 3052
    assert out[5] == VT.TowerHp(2101, 3052) and out[11] == VT.TowerHp(1884, 3052)
    assert out[13] == out[14] == VT.TowerHp(1884, 3052)        # the misreads change nothing...
    assert out[17] == VT.TowerHp(1667, 3052)                   # ...nor delay the fall (from its first 1667)
    assert abs(tr.frac[K] - 1667 / 3052) < 1e-9                # 0.546, not 1667 / 3667 = 0.455
    # however often, and even at a legal value (a level-13 princess): never a max after damage
    out = feed(tr, [3668] * 12, 3.25, clock=True)
    assert all_are(out, VT.TowerHp(1667, 3052)) and tr.recoveries == 0


def test_the_max_rises_only_before_damage_to_a_legal_value():
    tr = tracker()                                              # a level-12 princess, prior 3052
    out = feed(tr, [3346] * 5)
    assert [h.max for h in out] == [3052] * 4 + [3346] and out[-1] == VT.TowerHp(3346, 3346)
    assert out[0] == VT.TowerHp(None, 3052)                     # above the max: not trusted yet
    tr = tracker()                                              # not a tower's HP at any level
    assert {h.max for h in feed(tr, [3300] * 10)} == {3052}
    tr = tracker()                                              # under the confidence floor
    assert {h.max for h in feed(tr, [3346] * 10, confs=0.7)} == {3052}
    tr = tracker()                                              # another confident value breaks the run
    assert {h.max for h in feed(tr, [3346] * 4 + [3345] + [3346] * 4)} == {3052}
    assert feed(tr, [3346], 9 * DT)[0] == VT.TowerHp(3346, 3346)
    tr = tracker()                                              # after damage: never
    feed(tr, [2500] * 6)
    assert all_are(feed(tr, [3346] * 10, 0.75), VT.TowerHp(2500, 3052))
    fixed = H.TowerHpTracker({K: 3052})                         # a given max_hp is fixed
    assert {h.max for h in feed(fixed, [3346] * 10)} == {3052}
    free = tracker(legal_max_hp=None)                           # None: any value may become the max
    assert feed(free, [3300] * 5)[-1] == VT.TowerHp(3300, 3300)
    none = H.TowerHpTracker()                                   # no prior: the first legal value, 5 times
    out = feed(none, [3052] * 5)
    assert out[:4] == [VT.TowerHp(3052, None)] * 4 and out[4] == VT.TowerHp(3052, 3052)


# ---------------------------------------------------------------------------
# Real damage, the bar, destruction
# ---------------------------------------------------------------------------
def _bar(hp: int) -> float:
    f = round(hp / 3052, 3)
    return 1.0 if f >= 0.95 else f                              # measure_tower_bar snaps >= 0.95 to full


def test_real_damage_is_followed_promptly():
    # a Hog Rider's hits (263 each) 13 frames apart, then a Fireball and a hit 2 frames apart
    truth = [3052] * 8 + [2789] * 13 + [2526] * 13 + [2263] * 13 + [1900] * 2 + [1637] * 11
    starts = [i for i in range(1, len(truth)) if truth[i] != truth[i - 1]]
    for bars in (None, [_bar(hp) for hp in truth]):
        tr = tracker()
        out = feed(tr, truth, clock=True, bars=bars)
        want, shown = [], 3052
        for i in range(len(truth)):
            for s in starts:
                if i == s + 5:                                  # the 6th frame of a run: 0.625 s >= 0.6 s
                    shown = truth[s] if s != starts[-2] else truth[starts[-1]]
            want.append(VT.TowerHp(shown, 3052))
        assert out == want, bars is not None
        assert tr.recoveries == 0 and tr.max[K] == 3052
    # the run that held 1900 for 2 frames and 1637 after it is taken at its upper median, 1637,
    # on the 6th frame after 1900 first showed
    assert want[starts[-2] + 5] == VT.TowerHp(1637, 3052) and want[starts[-2] + 4] == VT.TowerHp(2263, 3052)


def test_digits_the_bar_contradicts_are_not_trusted():
    tr = tracker()
    out = feed(tr, [1000] * 10, bars=1.0)                       # 0.33 against a full bar: the bar counts
    assert all_are(out, VT.TowerHp(3052, 3052))
    out = feed(tr, [2101] * 6, 1.25, bars=0.683)                # the owner's frames: 0.683 for 2101
    assert out[4] == VT.TowerHp(3052, 3052) and out[5] == VT.TowerHp(2101, 3052)
    out = feed(tr, [1000] * 10, 2.0, bars=0.683)
    assert all_are(out, VT.TowerHp(2101, 3052))
    out = feed(tr, [1000] * 6, 3.25, bars=0.0)                  # no bar seen: the digits are all there is
    assert out[4] == VT.TowerHp(2101, 3052) and out[5] == VT.TowerHp(1000, 3052)


def test_a_destroyed_tower_goes_to_zero_and_stays():
    tr = tracker()
    feed(tr, [3052] * 4, clock=True)
    assert feed(tr, [400] * 6, 0.5, clock=True)[-1] == VT.TowerHp(400, 3052)
    # a reading above the max is something visible, not the occlusion zero: 3 s of it change nothing
    assert all_are(feed(tr, [5000] * 24, 1.25, clock=True), VT.TowerHp(400, 3052))
    # destroyed: no number and no bar (measure_tower_bar's 0.0) for 2.5 s, from <= 0.75
    out = feed(tr, [None] * 21, 4.25, clock=True, bars=0.0)
    assert out[19] == VT.TowerHp(400, 3052) and out[20] == VT.TowerHp(0, 3052)
    assert tr.crowns() == (1, 0)
    # it stays destroyed, whatever the rubble reads as
    assert all_are(feed(tr, [3052] * 10 + [400] * 10, 7.0, clock=True), VT.TowerHp(0, 3052))
    assert tr.recoveries == 0 and tr.crowns() == (1, 0)


def test_old_callers_keep_their_api():
    tr = H.TowerHpTracker(prior_max_hp=PRIOR)                   # update(hp_reads, now, bar_fracs)
    got = tr.update({K: 3052, "own_left": None}, 0.0, {"own_left": 0.9})
    assert set(got) == set(VT.TOWER_KEYS) and got[K] == VT.TowerHp(3052, 3052)
    assert got["own_king"] == VT.TowerHp(4824, 4824) and got["enemy_left"] == VT.TowerHp(None, 3052)
    f = H.TowerHpFilter()
    assert f.update(K, 0.5, 0.0) == 1.0 and f.force(K, 0.4) == 0.4 and f.accepted(K) == 0.4
    assert f.update(K, 0.6, 0.125) == 0.4                       # never rises after a force either


# ---------------------------------------------------------------------------
# HudReader: the clock gate end to end
# ---------------------------------------------------------------------------
class StubClock:
    ready = True

    def __init__(self, seconds):
        self.seconds = list(seconds)

    def read(self, rgb):
        s = self.seconds.pop(0)
        if s is None:
            return H.ClockRead(None, None, 0.0, None, "no glyphs")
        return H.ClockRead(s, None, 0.95, f"{s // 60}:{s % 60:02d}")


class StubTowers:
    def __init__(self, reads):
        self.reads = list(reads)

    def read_digits(self, rgb):
        hp = self.reads.pop(0)
        if hp is None:
            return {K: H.TowerRead(None, 0.0, None, "no glyphs")}
        return {K: H.TowerRead(hp, 0.95, str(hp))}

    def read_bars(self, rgb):
        return {}


def reader_with(seconds, reads, **kw) -> H.HudReader:
    r = H.HudReader(TP.TemplateStore.memory(), {}, regions=H.HudRegions.builtin(hand_centres=[(0.5, 0.9)] * 4),
                    **kw)
    if seconds is not None:
        r.clock_reader = StubClock(seconds)
    r.tower_reader = StubTowers(reads)
    return r


FRAME = np.zeros((640, 360, 3), dtype=np.uint8)


def test_hud_reader_towers_wait_for_the_battle_clock():
    seconds = [None, None, None] + [176] * 7
    reads = [855, 855, 577] + [3052] + [2101] * 6
    r = reader_with(seconds, reads)
    huds = [r.read(FRAME, i * DT) for i in range(3)]
    assert all_are([h.towers[K] for h in huds], VT.TowerHp(3052, 3052))
    assert r.report.towers_live is False and r.report.tower_frac[K] is None and r.report.errors == []
    assert r.report.summary()["tower_reads"] == {K: [577, 0.95]}
    hud = r.read(FRAME, 3 * DT)                                 # 2:56: the clock runs
    assert hud.clock_tick is not None and hud.towers[K] == VT.TowerHp(3052, 3052)
    assert r.report.towers_live is True and r.towers.started
    huds = [r.read(FRAME, i * DT) for i in range(4, 10)]
    assert huds[4].towers[K] == VT.TowerHp(3052, 3052) and huds[5].towers[K] == VT.TowerHp(2101, 3052)
    assert r.report.summary()["towers_live"] is True and r.report.errors == []
    r.reset()
    assert not r.towers.started


def test_hud_reader_without_clock_templates_counts_from_the_first_frame():
    for kw, seconds in (({}, None), ({"towers_wait_for_clock": False}, [None] * 6)):
        r = reader_with(seconds, [855] * 6, **kw)
        assert r.clock_running(None) is None
        huds = [r.read(FRAME, i * DT) for i in range(6)]
        assert huds[4].towers[K] == VT.TowerHp(3052, 3052) and huds[5].towers[K] == VT.TowerHp(855, 3052)
        assert r.report.towers_live is True and r.report.errors == []


def test_hud_reader_clock_running_rules():
    r = reader_with([None], [None])
    st = H.ClockState
    assert r.clock_running(None) is False
    assert r.clock_running(st(None, VT.PHASE_UNKNOWN, 0.0, False, "unread")) is False
    assert r.clock_running(st(90, VT.PHASE_REGULATION, 0.8, False, "read")) is True
    assert r.clock_running(st(3700, VT.PHASE_OVERTIME, 1.0, True, "read")) is True
    assert r.clock_running(st(95, VT.PHASE_REGULATION, 0.5, False, "rejected")) is False   # coasting
    assert r.clock_running(st(6000, VT.PHASE_ENDED, 0.8, False, "read")) is False
