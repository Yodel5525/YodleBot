"""The live elixir cap after our own taps (tools/rs_decide.ElixirCap; scripts/vision_play.py).

Owner's first --tap friendly (2026-11-01): a Cannon tapped at tick 610 with 4.26 elixir left
its slot at 614, but the HUD read 3.99, then 3.00 on 615-619 and the true 1.43 only from
621. prune_pending had dropped the Cannon at 614, so at 619 the bot spent the 3.00 on a Log
the game refused. Once a tap of ours is paid, the runner now spends at most

    min(HUD, spendable at the tap - cost + the most regen possible since the tap)

until the HUD reads within 0.2 elixir of that cap or 30 ticks pass. Everything here is
synthetic: the HUD readings below copy the shape of that sequence, no owner record is used.
The loop tests run scripts/vision_play.VisionPlay on test_vision_play's scripted frames.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import vision_play as VP  # noqa: E402
from scripts.vision import control as CT  # noqa: E402
from scripts.vision import tap_refusal as TR  # noqa: E402
from tests.pipeline import test_vision_play as TVP  # noqa: E402  (its scripted loop)

import rs_elixir  # noqa: E402  (tools/ is on sys.path once vision_play is imported)

RD = VP.RD
LAW = rs_elixir.ElixirLaw.constants()
COST = TVP.COST                      # conftest's catalogue: 0 Cannon 3, 1 Fireball 4, 2 Hog 4,
CANNON, FIREBALL, HOG, GOLEMITE, SPIRITS, MUSKETEER, SKELETONS, LOG = range(8)   # ... 7 Log 2
EMPTY = VP.EMPTY_CARD
PER_TICK_1X = 1000 / 56              # milli a tick at 1x (2.8 s an elixir)


def up(ticks: int, per_tick: float = PER_TICK_1X) -> int:
    """The cap's regen over `ticks` ticks at one rate: rounded up."""
    return math.ceil(ticks * per_tick - 1e-9)


# ---------------------------------------------------------------------------
# helpers: a policy that plays a slot at chosen ticks, the loop with or without the cap
# ---------------------------------------------------------------------------

class PlanPolicy:
    """rs_decide's policy protocol: plan(tick) -> the slot it wants (all of the gate on it,
    wait -1e4: never a drawn wait; the other slots below rs_decide.UNPLAYABLE, so a masked
    wanted slot is a no_card wait, never another card) or None (a sure wait). Tile logits put
    TVP.PLAY_TILE first."""

    def __init__(self, plan):
        self.plan = plan

    def batch(self, row):
        return row

    def gate(self, b):
        g = np.full(5, -5e8, dtype=np.float32)
        slot = self.plan(int(b["tick"]))
        if slot is None:
            g[0] = 50.0
        else:
            g[0] = -1e4
            g[1 + slot] = 50.0
        return g

    def tiles(self, b, slot):
        t = np.zeros(RD.TILES, dtype=np.float32)
        t[TVP.PLAY_TILE[1] * RD.GRID_W + TVP.PLAY_TILE[0]] = 10.0
        return t


def make_decider(policy, *, cap=True, refusals=None):
    """test_vision_play's minimal Decider (state from the HUD, accept-all legality), with the
    elixir cap on the constants' law (cap=True) or off (cap=False: elixir_cap=None)."""
    def state_fn(vf, ctx):
        hand = list(vf.hud.hand)
        me = {"hand": hand, "hand_costs": [COST.get(c, 0) for c in hand],
              "elixir_milli": vf.hud.elixir_milli or 0, "next_card": vf.hud.next}
        them = {"hand": [], "hand_costs": [], "elixir_milli": 0}
        return {"tick": vf.tick, "entities": [], "players": [them, me]}

    legality = lambda slot, card, x, y: True  # noqa: E731
    if refusals is not None:
        legality = VP.RefusalLegality(legality, refusals)
    return VP.Decider(policy=policy, legality=legality, config=RD.DecideConfig.live(0.0),
                      state_fn=state_fn, row_fn=lambda state, view, tick: {"tick": tick},
                      rng=np.random.default_rng(0),
                      elixir_cap=(lambda: RD.ElixirCap(LAW)) if cap else None)


def script(ticks, hand_at, elixir_at):
    """Two menu frames, a battle frame per tick in `ticks` (0.1 s apart), the end screen."""
    items = [TVP.menu_vf(TVP.t_of(0)), TVP.menu_vf(TVP.t_of(1))]
    for tick in ticks:
        items.append(TVP.battle_vf(TVP.t_of(len(items)), tick, hand=hand_at(tick), elixir=elixir_at(tick)))
    items += [TVP.end_vf(TVP.t_of(len(items)), tick=ticks[-1] + 1) for _ in range(5)]
    items += [TVP.menu_vf(TVP.t_of(len(items))) for _ in range(3)]
    return [(TVP.t_of(i), item) for i, item in enumerate(items)]


def frames_by_tick(res):
    """{tick: the frame line's extra} of the decision frames."""
    return {e["decision"]["tick"]: e for e in (f["extra"] for f in res.records["frame"])
            if "decision" in e}


def tap_ticks(res):
    return [(t["tick"], t["card"]) for t in res.records.get("tap", [])]


# The 610/619 sequence (synthetic, the shape of the friendly's): hand Cannon, Ice Golemite,
# Fireball, Log; the Cannon tapped at 610 with 4.264; its slot reads empty ("?") from 614; the
# HUD reads 3.992 at 614, 3.004 / 2.984 on 615-619 (the counter animating), the true 1.434 at
# 621, then fills at 1x. After the Cannon the policy wants the Log (2) every tick.
HAND_610 = [CANNON, GOLEMITE, FIREBALL, LOG]
TICKS_610 = list(range(70, 601, 2)) + [610, 612, 614, 615, 616, 617, 618, 619] + list(range(621, 671))
HUD_610 = {614: 3992, 615: 3004, 616: 2984, 617: 3004, 618: 3004, 619: 3004}


def hand_610(tick):
    return list(HAND_610) if tick < 614 else [EMPTY, GOLEMITE, FIREBALL, LOG]


def elixir_610(tick):
    if tick <= 612:
        return 4264
    if tick in HUD_610:
        return HUD_610[tick]
    return 1434 + (tick - 621) * 1000 // 56          # the true bar, read right


def plan_610(tick):
    return CANNON if tick == 610 else (3 if tick > 610 else None)   # slot 0, then the Log's slot 3


def run_610(tmp_path, *, cap=True, **kw):
    return TVP.run_script(tmp_path, script(TICKS_610, hand_610, elixir_610),
                          decider=make_decider(PlanPolicy(plan_610), cap=cap), **kw)


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------

def test_regen_is_the_most_the_bar_can_gain_1x_2x_and_overtime_2x_3x():
    """1x 2.8 s, 2x 1.4 s, 3x 0.93 s an elixir by the tick (2x from 2400, overtime from 3600 at
    2x, 3x from 4800), rounded up, and dated 20 ticks later: across a switch the faster rate
    already counts (the clock reads whole seconds)."""
    cap = RD.ElixirCap(LAW)
    assert cap.regen_milli(600, 609) == 161 == up(9)                    # 1x: 160.7
    assert cap.regen_milli(3000, 3010) == 358 == up(10, 1000 / 28)      # regulation 2x: 357.1
    assert cap.regen_milli(4000, 4010) == 358                           # overtime 2x
    assert cap.regen_milli(5000, 5010) == 538 == up(10, 1505 / 28)      # overtime 3x: 537.5
    # 20 ticks of slack: 2375..2385 is regen'd as 2395..2405, half of it at 2x
    assert cap.regen_milli(2375, 2385) == 268 > LAW.regen(2375, 2385) * 1000 / LAW.scale
    assert cap.regen_milli(4775, 4785) == 448                           # 5 ticks 2x + 5 ticks 3x
    assert cap.regen_milli(700, 700) == 0 and cap.regen_milli(700, 690) == 0
    for t0, n in ((0, 30), (2350, 30), (2399, 5), (3590, 30), (4790, 30), (5500, 30)):
        assert cap.regen_milli(t0, t0 + n) * LAW.scale >= LAW.regen(t0, t0 + n) * 1000
    assert cap.config() == {"tol_milli": 200, "timeout_ticks": 30, "slack_ticks": 20, "max_age": 60}


def test_the_610_619_sequence_in_the_ledger_and_in_decide():
    """The Cannon's cap goes on at 614 (its card gone), 1.264 + the regen since 610; it bites
    while the HUD reads 3.99 / 3.00; it ends at 621, the HUD within 0.2 of it. decide spends
    min(HUD, cap): the Log (2) is masked at 619, where the HUD alone let it in."""
    cap = RD.ElixirCap(LAW)
    cannon = RD.Pending(CANNON, 3, 0, 610)
    cap.sent(cannon, 4264)
    assert cap.observe(612, hand_610(612), 4264) is None                # not paid yet
    want = {t: 1264 + up(t - 610) for t in (614, 615, 616, 617, 618, 619, 621)}
    assert want[614] == 1336 and want[619] == 1425 and want[621] == 1461
    for t in (614, 615, 616, 617, 618, 619):
        assert cap.observe(t, hand_610(t), elixir_610(t)) == want[t] == cap.value
    assert cap.observe(621, hand_610(621), 1434) == 1461                # its last frame
    assert cap.observe(622, hand_610(622), elixir_610(622)) is None and cap.plays == []
    assert cap.summary() == {"plays": 1, "paid": 1, "caught_up": 1, "timed_out": 0, "lost": 0,
                             "frames_on": 7}

    policy = PlanPolicy(lambda t: 3)
    state = {"tick": 619, "entities": [], "players": [
        {"hand": [], "hand_costs": [], "elixir_milli": 0},
        {"hand": hand_610(619), "hand_costs": [COST.get(c, 0) for c in hand_610(619)],
         "elixir_milli": 3004}]}

    def decide(**kw):
        return RD.decide(policy, {"tick": 619}, state, 1, lambda *a: True, RD.DecideConfig.live(0.0),
                         rng=np.random.default_rng(0), pending=[], tick=619, **kw)
    old, new = decide(), decide(elixir_cap=1425)
    assert old.is_play and old.card == LOG and old.spendable_milli == 3004
    assert new.kind == RD.WAIT and new.reason == RD.R_NO_CARD and new.spendable_milli == 1425
    assert old.mask[1 + 1] and old.mask[1 + 3] and not new.mask[1 + 1] and not new.mask[1 + 3]


def test_plays_in_a_row_stack():
    """A Hog then Skeletons two ticks later, both paid before the HUD drops: the Skeletons'
    base is its decision's spendable (HUD - the Hog in flight), so its cap holds both costs."""
    cap = RD.ElixirCap(LAW)
    hog = RD.Pending(HOG, 4, 1, 300)
    cap.sent(hog, 7000)
    assert cap.observe(302, [LOG, HOG, SKELETONS, FIREBALL], 7035) is None
    skel = RD.Pending(SKELETONS, 1, 2, 302)
    cap.sent(skel, 7035 - 4000)                                         # decide's spendable at 302
    assert cap.observe(303, [LOG, EMPTY, SKELETONS, FIREBALL], 7053) == 3000 + up(3)   # the Hog's
    assert cap.observe(305, [LOG, EMPTY, EMPTY, FIREBALL], 7089) == 2035 + up(3)       # both
    assert [c.paid_tick for c in cap.plays] == [303, 305]
    # the HUD drops to the true bar: both caps end on that frame (within 0.2 of each)
    true = 2000 + up(10)
    assert cap.observe(310, [LOG, EMPTY, EMPTY, FIREBALL], true) == 2035 + up(8)
    assert cap.plays == [] and cap.summary()["caught_up"] == 2


def test_a_cap_times_out_30_ticks_after_payment_and_an_unpaid_tap_is_forgotten():
    """The card left its slot but the HUD never fell (a misread): the cap is on 30 ticks from
    the payment, then off. A tap whose card never leaves (refused) puts no cap on, and is
    forgotten at 60 ticks; a press is not followed."""
    cap = RD.ElixirCap(LAW)
    cap.sent(RD.Pending(HOG, 4, 1, 100), 7000)
    gone = [LOG, EMPTY, SKELETONS, SPIRITS]
    on = {t: cap.observe(t, gone, 7000 + up(t - 100)) for t in range(103, 140)}
    assert all(on[t] == 3000 + up(t - 100) for t in range(103, 133))     # 103..132: 30 ticks
    assert all(on[t] is None for t in range(133, 140))
    assert cap.summary()["timed_out"] == 1 and cap.summary()["caught_up"] == 0

    cap = RD.ElixirCap(LAW)
    cap.sent(RD.Pending(HOG, 4, 1, 100), 7000)
    cap.sent(RD.Pending(-1, 3, tick=100, button=0), 7000)              # a press: not followed
    held = [LOG, HOG, SKELETONS, SPIRITS]
    assert all(cap.observe(t, held, 7000) is None for t in range(101, 165))
    assert cap.plays == [] and cap.summary() == {"plays": 1, "paid": 0, "caught_up": 0,
                                                 "timed_out": 0, "lost": 1, "frames_on": 0}


def test_overtime_caps_regen_at_2x_and_3x():
    """Overtime (the tick past 3600): 2x until 4800, then 3x; the HUD's phase is not read."""
    for tap, per_tick in ((4000, 1000 / 28), (5000, 1505 / 28)):
        cap = RD.ElixirCap(LAW)
        cap.sent(RD.Pending(CANNON, 3, 0, tap), 5000)
        assert cap.observe(tap + 3, [EMPTY, LOG, HOG, SPIRITS], 5100) == 2000 + up(3, per_tick)
        assert cap.observe(tap + 10, [EMPTY, LOG, HOG, SPIRITS], 5100) == 2000 + up(10, per_tick)
    assert 2000 + up(10, 1000 / 28) == 2358 and 2000 + up(10, 1505 / 28) == 2538


# ---------------------------------------------------------------------------
# the loop (scripts/vision_play.VisionPlay, AUTO with taps)
# ---------------------------------------------------------------------------

def test_loop_holds_the_log_until_the_real_elixir_pays_for_it(tmp_path):
    """The 610/619 sequence with taps: the Cannon at 610, then no tap while the HUD's 3.00 is
    not ours; the Log at 653, the first frame the bar reads 2.0. Without the cap the Log goes
    at 614 on the 3.99 (on the friendly the bot drew waits until 619 and tapped it on the
    3.00; the game refused it). The frame and decision records carry elixir_cap while it
    bites; battle_end counts it."""
    res = run_610(tmp_path / "cap")
    assert tap_ticks(res) == [(610, CANNON), (653, LOG)]
    assert elixir_610(652) < 2000 <= elixir_610(653)
    frames = frames_by_tick(res)
    for t in (614, 615, 616, 617, 618, 619):
        e = frames[t]
        assert e["elixir_cap"] == 1264 + up(t - 610) == e["decision"]["spendable_milli"]
        assert e["decision"]["kind"] == RD.WAIT and e["decision"]["reason"] == RD.R_NO_CARD
    assert "elixir_cap" not in frames[612] and frames[612]["decision"]["spendable_milli"] == 1264
    assert "elixir_cap" not in frames[621] and frames[621]["decision"]["spendable_milli"] == 1434
    assert all("elixir_cap" not in frames[t] for t in range(622, 671))
    [log_tap] = [d for d in TVP.decisions(res.records, CT.ACTION_TAP) if d["card"] == LOG]
    assert "elixir_cap" not in log_tap and log_tap["elixir"] == 2005
    [end] = res.records["battle_end"]
    assert end["elixir_cap"] == {"plays": 2, "paid": 1, "caught_up": 1, "timed_out": 0, "lost": 0,
                                 "frames_on": 7}

    without = run_610(tmp_path / "off", cap=False)
    assert tap_ticks(without) == [(610, CANNON), (614, LOG)]
    assert frames_by_tick(without)[614]["decision"]["spendable_milli"] == 3992
    assert "elixir_cap" not in without.records["battle_end"][0]


def test_loop_stacks_a_double_play(tmp_path):
    """Hog at 300 (7.0), Skeletons at 302 (spendable 7.035 - 4 in flight); their slots empty at
    303 and 305 while the HUD still shows the 7 and up; it drops to the true bar at 310. The
    policy wants the Fireball (4) from 303 on: held until the bar reads 4.0 (412). Without the
    cap it is tapped at 303 on the lagging 7.05 (minus the Skeletons in flight)."""
    hand0 = [LOG, HOG, SKELETONS, FIREBALL]

    def hand(t):
        return [LOG, EMPTY if t >= 303 else HOG, EMPTY if t >= 305 else SKELETONS, FIREBALL]

    def elixir(t):
        if t < 300:
            return 7000
        lag = 5000 if t < 310 else 0                    # both paid, the HUD has not shown it
        return 2000 + lag + (t - 300) * 1000 // 56

    def plan(t):
        return {300: 1, 302: 2}.get(t, 3 if t > 302 else None)

    ticks = list(range(70, 299, 2)) + list(range(300, 421))
    assert hand(300) == hand0
    res = TVP.run_script(tmp_path / "cap", script(ticks, hand, elixir),
                         decider=make_decider(PlanPolicy(plan)))
    assert tap_ticks(res) == [(300, HOG), (302, SKELETONS), (412, FIREBALL)]
    assert elixir(411) < 4000 <= elixir(412)
    frames = frames_by_tick(res)
    hog_cap = {t: 3000 + up(t - 300) for t in range(303, 311)}
    skel_cap = {t: (7035 - 4000) - 1000 + up(t - 302) for t in range(305, 311)}
    assert frames[302]["decision"]["spendable_milli"] == 7035 - 4000      # the Skeletons' base
    assert frames[303]["elixir_cap"] == hog_cap[303]
    assert frames[303]["decision"]["spendable_milli"] == hog_cap[303] - 1000   # Skeletons in flight
    assert frames[304]["decision"]["spendable_milli"] == hog_cap[304] - 1000
    for t in range(305, 310):
        assert frames[t]["elixir_cap"] == skel_cap[t] == frames[t]["decision"]["spendable_milli"]
        assert skel_cap[t] < hog_cap[t] < 4000
    assert all(frames[t]["decision"]["kind"] == RD.WAIT for t in range(303, 412))
    assert res.records["battle_end"][0]["elixir_cap"]["caught_up"] == 2

    without = TVP.run_script(tmp_path / "off", script(ticks, hand, elixir),
                             decider=make_decider(PlanPolicy(plan), cap=False))
    assert tap_ticks(without)[:3] == [(300, HOG), (302, SKELETONS), (303, FIREBALL)]


def test_the_refused_tap_guard_gets_the_capped_spendable(tmp_path):
    """A 1-elixir Ice Spirits tapped at 616 while the Cannon's cap bites (HUD 2.984, cap 1.372)
    and never leaving its slot: tap_refused reports elixir_at_tap 2984 (the HUD) and
    spendable_at_tap 1372 (what the decision had), as the decision record does."""
    g = TR.RefusedTapGuard(enforce=True)

    def hand(t):
        return [CANNON if t < 614 else EMPTY, SPIRITS, FIREBALL, LOG]

    res = TVP.run_script(tmp_path, script(TICKS_610, hand, elixir_610),
                         decider=make_decider(PlanPolicy({610: 0, 616: 1}.get), refusals=g), refusals=g)
    assert tap_ticks(res) == [(610, CANNON), (616, SPIRITS)]
    [refused] = res.records["tap_refused"]
    assert (refused["tap_tick"], refused["card"]) == (616, SPIRITS)
    assert refused["elixir_at_tap"] == 2984 and refused["spendable_at_tap"] == 1264 + up(6) == 1372
    [d] = [d for d in TVP.decisions(res.records, CT.ACTION_TAP) if d["card"] == SPIRITS]
    assert d["elixir"] == 2984 and d["elixir_cap"] == 1372


# ---------------------------------------------------------------------------
# dry runs and the simulator paths: unchanged
# ---------------------------------------------------------------------------

def _decided(res):
    return [(e["decision"], e.get("action"), e.get("would")) for e in frames_by_tick(res).values()]


@pytest.mark.parametrize("mode,taps", [(CT.AUTO, False), (CT.MANUAL, True)])
def test_a_dry_run_decides_exactly_as_without_the_cap(tmp_path, mode, taps):
    """No tap is sent (a dry run, or MANUAL): no cap ever goes on, so every decision, would-play
    and record is what it was without the cap; nothing says elixir_cap."""
    on = run_610(tmp_path / "on", mode=mode, taps=taps)
    off = run_610(tmp_path / "off", mode=mode, taps=taps, cap=False)
    assert on.tapper.calls == off.tapper.calls == []
    assert _decided(on) == _decided(off) and len(_decided(on)) > 300
    assert [d for d in TVP.decisions(on.records)] == [d for d in TVP.decisions(off.records)]
    assert TVP.decisions(on.records)                   # the Log was suggested (dry / manual)
    for rec in (on.records, off.records):
        assert not any("elixir_cap" in f.get("extra", {}) for f in rec["frame"])
        assert not any("elixir_cap" in d for d in rec.get("decision", []))
        assert "elixir_cap" not in rec["battle_end"][0]


def test_decide_without_a_cap_is_unchanged_and_a_cap_only_lowers_the_spendable():
    """elixir_cap=None is the call without it, in every preset, draw for draw; a cap at or above
    the HUD changes nothing; one below it only lowers spendable (and so the masks)."""
    hand = [CANNON, GOLEMITE, FIREBALL, LOG]
    state = {"tick": 619, "entities": [], "players": [
        {"hand": [], "hand_costs": [], "elixir_milli": 0},
        {"hand": hand, "hand_costs": [COST[c] for c in hand], "elixir_milli": 5200, "pending_cost": 1}]}
    policy = TVP.FakePolicy((619,))
    for cfg in (RD.DecideConfig.legacy(0.0), RD.DecideConfig.live(0.0), RD.DecideConfig.sampling(0.0)):
        for pending in (None, [RD.Pending(SKELETONS, 1, -1, 610)]):
            def go(**kw):
                rng = np.random.default_rng(7)
                d = RD.decide(policy, {"tick": 619}, state, 1, lambda *a: True, cfg, rng=rng,
                              pending=pending, tick=619, **kw)
                return d.to_record(), rng.random()
            base = go()
            assert go(elixir_cap=None) == base == go(elixir_cap=5200) == go(elixir_cap=10 ** 6)
            low, _ = go(elixir_cap=3500)
            assert low["spendable_milli"] == base[0]["spendable_milli"] - 1700 == 3500 - 1000
            assert low["mask"][1 + 2] is False and base[0]["mask"][1 + 2] is True     # the Fireball


def test_a_stale_full_bar_no_longer_forces_a_play():
    """The cap rule (live: no drawn wait at >= 9.5 spendable) read the HUD's stale bar: on the
    friendly a Hog paid from 9.48 at tick 3066 still read 9.54, so the Log tapped there was
    forced. With the Hog's cap (5.549) the rule does not fire and the wait is drawn."""
    hand = [FIREBALL, CANNON, EMPTY, LOG]
    state = {"tick": 3066, "entities": [], "players": [
        {"hand": [], "hand_costs": [], "elixir_milli": 0},
        {"hand": hand, "hand_costs": [COST.get(c, 0) for c in hand], "elixir_milli": 9535}]}
    policy = TVP.FakePolicy(())                     # p_wait 1: every drawn wait is a wait

    def decide(**kw):
        return RD.decide(policy, {"tick": 3066}, state, 1, lambda *a: True, RD.DecideConfig.live(0.0),
                         rng=np.random.default_rng(0), pending=[], tick=3066, **kw)
    old, new = decide(), decide(elixir_cap=5549)
    assert old.capped and old.is_play and old.spendable_milli == 9535
    assert not new.capped and new.kind == RD.WAIT and new.reason == RD.R_WAIT
    assert new.spendable_milli == 5549


def test_only_the_live_runner_uses_the_cap():
    """rs_eval, rs_match, RL, search and the rows never pass elixir_cap: their decisions are
    as before (their own tests replay them); only rs_decide and vision_play name it."""
    users = set()
    for folder in ("tools", "scripts", "sim", "src"):
        for path in (ROOT / folder).rglob("*.py"):
            if re.search(r"elixir_cap|ElixirCap", path.read_text(encoding="utf-8", errors="replace")):
                users.add(path.relative_to(ROOT).as_posix())
    assert users == {"tools/rs_decide.py", "scripts/vision_play.py"}
