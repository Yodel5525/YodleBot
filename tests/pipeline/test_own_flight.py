"""Our own plays in flight, live (v2 rows): own_pending, the board and our elixir while a tap is
on its way (tools/public_state.OwnTaps / LiveObserver, scripts/vision_play.make_state_fn /
row_state), the deriver's drop-in rule (scripts/vision/events.py rule 4b) and latency_probe's D.

The decision-time rows list a play in own_pending while tap < t <= exec (24 +- 3 ticks), with the
card in hand, the elixir bar UNPAID and no unit on the board. Live (owner's first --tap friendly,
2026-11-01) the play runs a median 25 ticks after the decided frame; the client shows a placement
preview (the "ghost") from the tap + 3 that the detector boxes as our unit, pays the HUD bar from
+3..10, and drops the real unit in at the run, 2-3 tiles above the aim. Everything here is
SYNTHETIC: the sequences copy the shapes measured on that friendly, no owner record is used.

Catalogue ids (test_live_v2's): 0 Cannon 3, 1 Fireball 4, 2 Hog Rider 4, 3 Ice Golem 2, 4 Ice
Spirits 1, 5 Musketeer 4, 6 Skeletons 1, 7 Log 2; delay 24 (v2_hog's --obs-delay).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import vision_play as VP  # noqa: E402
from scripts.vision import control as CT  # noqa: E402
from scripts.vision import events as EV  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from scripts.vision.types import Events, OwnLanding  # noqa: E402
from tests.pipeline import test_live_v2 as TLV  # noqa: E402  (catalogue, builder, rr)
from tests.pipeline import test_vision_play as TVP  # noqa: E402  (the scripted loop)
from tools import public_state as PS  # noqa: E402

RD = VP.RD
rr = TLV.rr                                    # the rs_rows fixture
LAW = TLV.LAW
CANNON, FIREBALL, HOG, GOLEM, SPIRITS, MUSKETEER, SKELETONS, LOG = range(8)
COST = {CANNON: 3, FIREBALL: 4, HOG: 4, GOLEM: 2, SPIRITS: 1, MUSKETEER: 4, SKELETONS: 1, LOG: 2}
EMPTY = VT.EMPTY_CARD
OWN, ENEMY = VT.OWN_TEAM, VT.ENEMY_TEAM
DELAY = 24
#: The labels a play of each card shows (events.EventTables.classes_of) and its unit count.
INFO = {CANNON: (("Cannon",), False, 1), FIREBALL: (("Fireball", "FireballSpell"), True, 1),
        HOG: (("HogRider",), False, 1), GOLEM: (("IceGolemite",), False, 1),
        SPIRITS: (("IceSpirits",), False, 1), MUSKETEER: (("Musketeer",), False, 1),
        SKELETONS: (("Skeleton",), False, 3), LOG: (("Log", "LogProjectile"), True, 1)}


def observer(delay=DELAY):
    return PS.LiveObserver(TLV.CATALOGUE, delay=delay, law=LAW, start_tick=40)


def send(obs, card, tick, slot, aim):
    labels, spell, units = INFO[card]
    obs.sent(card, tick, slot=slot, cost=COST[card], aim=aim, labels=labels, spell=spell, units=units)


def tr(uid, cls, x, y, first, last, team=OWN):
    return VT.Track(uid=uid, cls=cls, team=team, conf=0.9, x_mt=float(x), y_mt=float(y),
                    first_seen_tick=first, last_seen_tick=last)


def pending(obs, t):
    return obs.inputs(t)["own_pending"]


# ---------------------------------------------------------------------------
# own_pending: listed from the tick after the tap until the play runs
# ---------------------------------------------------------------------------

def test_a_ghost_detected_play_is_pending_until_its_next_card_changes():
    """An Ice Golem tapped at 200 (slot 0, aim (9500, 31500)): its card leaves the slot at 203 and
    its preview is boxed there from 203; the hand's next card changes at 225 (the play runs) and the
    slot refills at 227. own_pending lists it 201..224 (24 - age to land), as the rows do, not
    until the preview's sighting (HEAD: 3 ticks); the deriver's landing of the preview ends
    nothing."""
    obs = observer()
    aim = (9500.0, 31500.0)
    hand = [GOLEM, HOG, FIREBALL, LOG]
    gone = [EMPTY, HOG, FIREBALL, LOG]
    obs.watch(200, hand, SKELETONS, [])
    send(obs, GOLEM, 200, 0, aim)
    shown = {}
    for t in range(201, 232):
        h = hand if t < 203 else gone if t < 227 else [SKELETONS, HOG, FIREBALL, LOG]
        tracks = [tr(1, "IceGolemite", 9600, 31700, 203, t)] if t >= 203 else []
        obs.events(Events([], [OwnLanding(GOLEM, 0, 200, 203, x_mt=9600, y_mt=31700, uid=1)] if t == 205 else []))
        obs.watch(t, h, SKELETONS if t < 225 else MUSKETEER, tracks)
        shown[t] = pending(obs, t)
    assert all(shown[t] == [[GOLEM + 2, max(0, 224 - t)]] for t in range(201, 226))  # 225: read once
    assert all(shown[t] == [] for t in range(226, 232))
    [f] = obs.own.flights
    assert (f.left_tick, f.run_how, f.run_tick, f.ghosts) == (203, "next", 225, {1: 203})
    assert obs.own.summary() == {"ghost_tracks": 1, "next": 1, "sent": 1}


def test_a_hog_play_shows_no_ghost_and_runs_when_its_unit_is_seen():
    """Hog Rider shows no preview: its real unit is first seen at +25, 1.5 tiles above the aim
    (it drops in): that sighting is the run (the next card changes later, as on a quick pair).
    The deriver's landing of a unit 20+ ticks on is the run too."""
    obs = observer()
    send(obs, HOG, 300, 1, (2500.0, 17500.0))
    hand = [LOG, EMPTY, FIREBALL, SKELETONS]
    for t in range(301, 330):
        tracks = [tr(5, "HogRider", 2600, 16000, 325, t)] if t >= 325 else []
        obs.watch(t, [LOG, HOG, FIREBALL, SKELETONS] if t < 302 else hand, CANNON, tracks)
        assert pending(obs, t) == ([[HOG + 2, max(0, 324 - t)]] if t < 325 else []), t
    [f] = obs.own.flights
    assert (f.run_how, f.run_tick, f.ghosts, f.drops) == ("unit", 325, {}, {5: 325})
    tracks = [tr(5, "HogRider", 2600, 16000, 325, 329)]
    assert obs.board(tracks, 329) is tracks                     # its own unit: shown as it is

    obs = observer()
    send(obs, HOG, 300, 1, (2500.0, 17500.0))
    obs.watch(303, [LOG, EMPTY, FIREBALL, SKELETONS], CANNON, [])
    assert obs.own.landed([OwnLanding(HOG, 1, 300, 310)]) == 0          # the 10-tick sighting: no run
    assert obs.own.landed([OwnLanding(HOG, 1, 300, 326)]) == 1
    assert pending(obs, 327) == [] and obs.own.flights[0].run_how == "landing"


def test_a_fireball_runs_when_its_projectile_leaves_the_king():
    """A Fireball aimed at the enemy tower (3500, 9500): no preview track; the projectile is first
    seen 24 ticks on next to our king, 16 tiles from the aim. A spell's sighting may be anywhere,
    so that is its run; the projectile is a spell in flight on the board as it is."""
    obs = observer()
    send(obs, FIREBALL, 400, 2, (3500.0, 9500.0))
    hand = [LOG, HOG, EMPTY, SKELETONS]
    for t in range(401, 430):
        tracks = [tr(8, "Fireball", 8800, 25800, 424, t)] if t >= 424 else []
        obs.watch(t, [LOG, HOG, FIREBALL, SKELETONS] if t < 402 else hand, CANNON, tracks)
        assert pending(obs, t) == ([[FIREBALL + 2, max(0, 424 - t)]] if t < 424 else []), t
    [f] = obs.own.flights
    assert (f.run_how, f.run_tick, f.drops) == ("unit", 424, {8: 424})
    # a Log's preview IS boxed at its aim (a spell label): hidden until the Log runs
    obs = observer()
    send(obs, LOG, 500, 3, (3500.0, 24500.0))
    log = [tr(9, "Log", 3800, 24300, 504, 510)]
    obs.watch(510, [LOG, HOG, FIREBALL, EMPTY], CANNON, log)
    assert obs.board(log, 510) == [] and obs.own.flights[0].ghosts == {9: 504}


def test_a_refused_tap_leaves_own_pending_and_never_adds_elixir_back():
    """A Log tapped at 619 that the client refused: its card never leaves the slot. From 8 ticks
    on that says the client did not take it: not listed (the rows only list plays that ran);
    tap_refused (30 ticks on) ends it. Nothing is ever added back to the row's elixir."""
    obs = observer()
    send(obs, LOG, 619, 3, (3500.0, 17500.0))
    hand = [EMPTY, GOLEM, FIREBALL, LOG]
    seen = {}
    for t in range(620, 650):
        obs.watch(t, hand, HOG, [])
        seen[t] = (pending(obs, t), obs.row_elixir(3000, None, t))
    assert all(seen[t] == ([[LOG + 2, 643 - t]], None) for t in range(620, 627))
    assert all(seen[t] == ([], None) for t in range(627, 650))
    assert obs.refused(LOG, 619, 3) and obs.own.flights[0].run_how == "refused"
    assert obs.own.summary() == {"refused": 1, "sent": 1}             # it timed out first: re-counted
    assert not obs.refused(HOG, 619, 1)                                 # not one of ours


def test_a_quick_double_play_gives_each_tap_its_own_run():
    """An Ice Golem at 600 and a Hog at 602 (a quick pair), Skeletons at 620. The Golem's next card
    changes at 625; the Hog's change only shows at 640 (the HUD's lag on a pair) but its unit
    drops in at 628: that is its run. The late change at 640 is still the Hog's (the HUD shows
    the changes in tap order), so the Skeletons' run is their own change at 645, not 640."""
    obs = observer()
    send(obs, GOLEM, 600, 0, (2500.0, 17500.0))
    send(obs, HOG, 602, 1, (2500.0, 17500.0))
    nxt = SPIRITS
    runs = {}
    for t in range(601, 660):
        hand = [GOLEM if t < 603 else EMPTY, HOG if t < 605 else EMPTY, SKELETONS if t < 622 else EMPTY, LOG]
        if t == 620:
            send(obs, SKELETONS, 620, 2, (3500.0, 22500.0))
        nxt = {625: MUSKETEER, 640: CANNON, 645: FIREBALL}.get(t, nxt)
        tracks = [tr(5, "HogRider", 2600, 15500, 628, t)] if t >= 628 else []
        obs.watch(t, hand, nxt, tracks)
        for f in obs.own.flights:
            runs.setdefault(f.pending.card, None)
            if f.run_tick is not None and runs[f.pending.card] is None:
                runs[f.pending.card] = (f.run_how, f.run_tick)
    assert runs == {GOLEM: ("next", 625), HOG: ("unit", 628), SKELETONS: ("next", 645)}
    assert all(f.next_seen for f in obs.own.flights)


def test_a_play_never_seen_to_run_is_dropped_at_the_delay_plus_3():
    """A Cannon tapped at 700 whose run nothing shows (no next-card change, no refill, no unit):
    listed until 727 (due from 724), then dropped (HEAD kept it 60 ticks); its preview, hidden
    until then, shows from 728 dated from 724 (the run it should have had)."""
    obs = observer()
    send(obs, CANNON, 700, 0, (8500.0, 22500.0))
    ghost = lambda t: [tr(9, "Cannon", 8500, 23500, 702, t)]  # noqa: E731
    for t in range(701, 735):
        obs.watch(t, [EMPTY if t >= 703 else CANNON, HOG, FIREBALL, LOG], SPIRITS, ghost(t) if t >= 702 else [])
        assert pending(obs, t) == ([[CANNON + 2, max(0, 724 - t)]] if t <= 727 else []), t
        board = obs.board(ghost(t) if t >= 702 else [], t)
        if t <= 727:
            assert board == [], t
        else:
            assert [(b.uid, b.first_seen_tick) for b in board] == [(9, 724)]
    assert obs.own.flights[0].run_how == "timeout" and obs.own.summary()["timeout"] == 1


# ---------------------------------------------------------------------------
# the board: the ghost out of the row until the play runs, then dated from the run
# ---------------------------------------------------------------------------

def test_ghost_tracks_are_left_out_until_the_run_then_the_drop_in_is_the_same_unit():
    """Skeletons (3 units) at 800, aim (3500, 17500): three preview tracks from 803 (up to 1.7
    tiles apart: the formation), one more first seen at 815 (a fragment of the preview), the run at
    825 (next card), two drop-in tracks at 828 / 830 falling from 1.5 tiles above. Before the run
    all four previews are left out; from the run they show (at most 3) dated 825, the drop-ins
    while they fall are the same units again; once landed and seen they come first. An own Hog
    first seen near there in that window is another card's: it stays. So does an enemy Skeleton."""
    obs = observer()
    send(obs, SKELETONS, 800, 0, (3500.0, 17500.0))
    ghosts = [tr(11, "Skeleton", 4100, 18000, 803, 0), tr(12, "Skeleton", 3600, 16900, 803, 0),
              tr(13, "Skeleton", 2000, 18000, 803, 0), tr(14, "Skeleton", 3900, 17700, 815, 0)]
    other = [tr(20, "HogRider", 3400, 17500, 810, 0), tr(21, "Skeleton", 3500, 17400, 806, 0, team=ENEMY)]
    drops = [tr(15, "Skeleton", 2800, 16200, 828, 0), tr(16, "Skeleton", 4200, 16000, 830, 0)]

    def at(t, items):
        return [dataclasses.replace(x, last_seen_tick=t) for x in items if x.first_seen_tick <= t]
    for t in range(801, 846):
        tracks = at(t, ghosts + other + drops)
        obs.watch(t, [EMPTY if t >= 803 else SKELETONS, HOG, FIREBALL, LOG], CANNON if t < 825 else MUSKETEER, tracks)
        board = obs.board(tracks, t)
        uids = sorted(int(b.uid) for b in board)
        if t <= 825:                                                # the run is read twice: 825, 826
            assert uids == [u for u in (20, 21) if u in {int(x.uid) for x in tracks}], t
        elif t <= 838:                                              # the drop-ins still falling
            assert uids == [11, 12, 13, 20, 21], t                  # 3 previews, dated from the run
            assert all(b.first_seen_tick == 825 for b in board if int(b.uid) in (11, 12, 13))
        elif t <= 840:
            assert uids == [11, 12, 15, 20, 21], t                  # one drop-in landed: it first
        else:
            assert uids == [11, 15, 16, 20, 21], t                  # both landed
    f = obs.own.flights[0]
    assert set(f.ghosts) == {11, 12, 13, 14} and set(f.drops) == {15, 16}


def test_the_row_leaves_the_ghost_out_and_dates_its_deploy_from_the_run(rr):
    """Through vision_play's state_fn and row_fn (the real builder and obs_row): a Skeletons
    preview at the aim 10 ticks after the tap is not on the row; at the run (825) it is, deploying
    for its full deploy time; HEAD showed it from 803, deploying until ~823 and active after."""
    plan = VP.row_plan({"schema": TLV.V2, "obs_delay": {"ticks": DELAY}})
    obs = observer()
    send(obs, SKELETONS, 800, 2, (3500.0, 17500.0))
    ctx = VP.BattleContext(1, 0.0, 40)
    ctx.observer = obs
    state_fn = VP.make_state_fn(TLV.builder(spell_fields=rr.SF), SimpleNamespace(track_cards=lambda t: {}), plan)
    row_fn = VP.make_row_fn(rr.obs_row, TLV.SUB, plan, live_compat=False, units="types", row_inputs=PS.row_inputs)
    mem = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}

    def row_at(t):
        tracks = [TLV.track(11, "Skeleton", OWN, 3600, 17600, first=803, last=t)]
        vf = TLV.frame(tracks, tick=t, hand=[7, 2, EMPTY, 4], next=0)
        obs.watch(t, [7, 2, EMPTY, 4], 0 if t < 825 else 5, tracks)
        st = state_fn(vf, ctx)
        return st, row_fn(st, mem, t)
    seen = {t: row_at(t) for t in range(803, 827)}
    st, row = seen[813]
    assert st["entities"] == [] and row["unit_ids"].tolist() == [0] * 48
    assert row["own_pending"].tolist() == [SKELETONS + 2, 0, 0, 0]
    assert all(seen[t][0]["entities"] == [] for t in range(803, 825))
    st, row = seen[826]
    [e] = st["entities"]
    skel = TLV.CLASSES.by_name["Skeleton"]
    full = int(round(skel.form(skel.base_form).deploy_time_ms / 50))
    assert full > 0 and e[TLV.FIELD["deploy_ticks"]] == full - 1             # dated from 825
    assert row["unit_ids"][0] != 0 and row["own_pending"].tolist() == [0, 0, 0, 0]
    # HEAD's view: the same frame without the observer's board
    plain = TLV.builder(spell_fields=rr.SF).state(TLV.frame([TLV.track(11, "Skeleton", OWN, 3600, 17600, first=803,
                                                                        last=826)], tick=826))
    assert plain["entities"][0][TLV.FIELD["deploy_ticks"]] == max(0, full - 23)


def _shown(obs, tracks, t):
    return sorted((int(b.uid), b.cls, int(b.first_seen_tick)) for b in obs.board(tracks, t))


def test_a_preview_relabelled_to_the_next_taps_card_is_that_taps_preview():
    """The Ice Golem + Hog bridge push, both on one tile (aim (2500, 17500)): the Golem at 1000, the
    Hog at 1006. The Golem's preview track (1) reads IceGolemite from 1002 and HogRider from 1016
    (the tracker relabels it: the Hog's preview there). The Golem runs at 1025 (its next card),
    its real unit (2) dropping in; the Hog's unit (3) drops in at 1032 (its run). While the Hog is
    in flight the board shows the Golem and no Hog (own_pending lists it); after its run, one Golem
    and one Hog. Before this fix track 1 stayed the Golem's: a HogRider on the board next to the
    pending Hog, then two HogRiders and no Golem."""
    obs = observer()
    aim = (2500.0, 17500.0)
    send(obs, GOLEM, 1000, 0, aim)
    send(obs, HOG, 1006, 1, aim)
    shown, pend = {}, {}
    for t in range(1001, 1080):
        hand = [GOLEM if t < 1002 else EMPTY, HOG if t < 1008 else EMPTY, FIREBALL, LOG]
        tracks = []
        if 1002 <= t <= 1045:                                                  # gone at the run + 13
            tracks.append(tr(1, "IceGolemite" if t < 1016 else "HogRider", 2600, 17700, 1002, t))
        if t >= 1025:
            tracks.append(tr(2, "IceGolemite", 2700, max(17600, 15600 + 200 * (t - 1025)), 1025, t))
        if t >= 1032:
            tracks.append(tr(3, "HogRider", 2800, max(17400, 15800 + 200 * (t - 1032)), 1032, t))
        obs.watch(t, hand, SKELETONS if t < 1025 else CANNON if t < 1040 else MUSKETEER, tracks)
        shown[t], pend[t] = _shown(obs, tracks, t), pending(obs, t)
    golem, hog = obs.own.flights
    assert (golem.ghosts, golem.drops) == ({}, {2: 1025})
    assert (hog.ghosts, hog.drops, hog.run_how, hog.run_tick) == ({1: 1016}, {3: 1032}, "unit", 1032)
    assert all(shown[t] == [] for t in range(1002, 1025))                     # both previews hidden
    for t in range(1025, 1032):                                                # the Hog still in flight
        assert shown[t] == [(2, "IceGolemite", 1025)] and pend[t] == [[HOG + 2, max(0, 1030 - t)]], t
    for t in range(1032, 1080):                                                # one Golem, one Hog
        cls = [c for _, c, _ in shown[t]]
        assert sorted(cls) == ["HogRider", "IceGolemite"], t
        assert all(first >= 1025 for _, _, first in shown[t]), t              # dated from the runs
    assert obs.own.summary()["released"] == 1


def test_a_track_that_carries_another_unit_later_is_shown_and_the_cap_ends():
    """The Golem at 2000 (preview track 1 from 2002, run at 2025, its real unit 2 dropping in), the
    Hog at 2028 on the same tile (no preview; it runs at 2053). Track 1 lingers at the aim, picks
    up our Hog as it runs (still read IceGolemite) and is relabelled HogRider at 2071, 43 ticks
    after the Hog's tap: it is the Hog's drop-in and shows. Until the Golem's run + DUP_TICKS
    (2060) the Golem shows one unit; after that both tracks show as they are. Before this fix track
    1 stayed the Golem's extra track, hidden for BOARD_TICKS: our Hog off the row."""
    obs = observer()
    aim = (2500.0, 17500.0)
    send(obs, GOLEM, 2000, 0, aim)
    shown = {}
    for t in range(2001, 2140):
        if t == 2028:
            send(obs, HOG, 2028, 1, aim)
        hand = [GOLEM if t < 2002 else EMPTY, HOG if t < 2030 else EMPTY, FIREBALL, LOG]
        y1 = 17700 - max(0, t - 2055) * 40                                     # off to the bridge
        tracks = [tr(1, "IceGolemite" if t < 2071 else "HogRider", 2600, y1, 2002, t)]
        if t >= 2025:
            tracks.append(tr(2, "IceGolemite", 2700, max(17600, 15600 + 200 * (t - 2025)) - max(0, t - 2046) * 20,
                             2025, t))
        nxt = SKELETONS if t < 2025 else CANNON if t < 2053 else MUSKETEER
        obs.watch(t, hand, nxt, tracks)
        shown[t] = _shown(obs, tracks, t)
    golem, hog = obs.own.flights
    # its real unit at 2025 is the run (the next-card change is only read twice at 2026)
    assert (golem.run_how, golem.run_tick, golem.ghosts, golem.drops) == ("unit", 2025, {}, {2: 2025})
    assert (hog.run_how, hog.run_tick, hog.drops) == ("next", 2053, {1: 2071})
    for t in range(2036, 2061):                                                # the cap: one Golem
        assert [c for _, c, _ in shown[t]] == ["IceGolemite"], t
    for t in range(2061, 2071):                                                # cap over: as they are
        assert [u for u, _, _ in shown[t]] == [1, 2], t
    for t in range(2071, 2140):                                                # our Hog, and the Golem
        assert [(u, c) for u, c, _ in shown[t]] == [(1, "HogRider"), (2, "IceGolemite")], t


def test_a_claimed_track_the_enemy_takes_over_is_never_hidden_or_redated():
    """Our Musketeer at 200 (aim (4000, 26000)): its preview track 10 from 203; at 214 the tracker
    gives that track to the enemy's Musketeer walking by (team ENEMY). From then the track is the
    enemy's: shown as it is while our Musketeer is still in flight, its first sighting kept."""
    obs = observer()
    send(obs, MUSKETEER, 200, 1, (4000.0, 26000.0))
    for t in range(201, 260):
        team = OWN if t < 214 else ENEMY
        tracks = [tr(10, "Musketeer", 4000, 26000 - max(0, t - 214) * 30, 203, t, team=team)] if t >= 203 else []
        obs.watch(t, [HOG, MUSKETEER if t < 203 else EMPTY, LOG, SKELETONS], CANNON if t < 225 else GOLEM, tracks)
        board = obs.board(tracks, t)
        if 203 <= t < 214:
            assert board == [], t
        elif t >= 214:
            assert [(b.uid, b.team, b.first_seen_tick) for b in board] == [(10, ENEMY, 203)], t
    assert obs.own.flights[0].ghosts == {}


# ---------------------------------------------------------------------------
# the row's own elixir: unpaid until the play runs, as the rows' engine bar
# ---------------------------------------------------------------------------

def test_the_row_adds_back_what_the_hud_paid_for_a_play_not_run_yet():
    """An Ice Golem (2) tapped at 200 on 8.0: its card leaves at 203, the HUD pays from 205 (and
    reads high while it animates); the run at 225. The decision spends min(HUD, cap) as before;
    the row shows min(HUD, cap) + 2 while the tap is listed and paid: the unpaid engine bar.
    Before the card leaves (201-202) the HUD has not paid: nothing is added (no double count).
    The elixir cap's own base is the decision's (rs_decide.ElixirCap)."""
    obs, cap = observer(), RD.ElixirCap(LAW)
    hand = [GOLEM, HOG, FIREBALL, LOG]
    gone = [EMPTY, HOG, FIREBALL, LOG]
    send(obs, GOLEM, 200, 0, (9500.0, 31500.0))
    cap.sent(RD.Pending(GOLEM, 2, 0, 200), 8000)
    up = lambda n: -(-n * 1000 // 56)  # noqa: E731
    hud = {t: 8000 + up(t - 200) for t in range(200, 205)}
    hud.update({205: 7100, 206: 6400, 207: 6150})                 # the counter animating down
    hud.update({t: 6000 + up(t - 200) for t in range(208, 240)})
    rows = {}
    for t in range(201, 235):
        h = hand if t < 203 else gone if t < 227 else [SKELETONS, HOG, FIREBALL, LOG]
        obs.watch(t, h, SKELETONS if t < 225 else MUSKETEER, [])
        c = cap.observe(t, h, hud[t])
        rows[t] = (c, obs.row_elixir(hud[t], c, t))
    assert rows[201] == (None, None) and rows[202] == (None, None)
    for t in range(203, 226):                                     # listed and paid: + 2 (225: run read once)
        c, row = rows[t]
        assert row == (hud[t] if c is None else min(hud[t], c)) + 2000, t
        assert abs(row - (8000 + up(t - 200))) <= 400, t          # the engine's unpaid bar (cap slack)
    assert rows[203][1] == min(8054, 6000 + cap.regen_milli(200, 203)) + 2000
    for t in range(226, 235):                                     # run: nothing to add
        c, row = rows[t]
        assert row == (None if c is None or c >= hud[t] else c), t
    fresh = observer()
    send(fresh, GOLEM, 200, 0, (9500.0, 31500.0))
    fresh.watch(203, gone, SKELETONS, [])
    assert fresh.row_elixir(None, None, 210) is None              # the HUD unread: as before
    assert fresh.row_elixir(9500, None, 210) == 10_000            # at most 10 elixir
    assert fresh.row_elixir(9500, 6000, 210) == 8000              # min(HUD, cap) + 2


def test_row_state_reads_the_row_elixir_and_decide_never_does():
    st = {"tick": 5, "players": [{"elixir_milli": 0, "hand": []},
                                 {"elixir_milli": 6000, "hand": [1, 2, 3, 4], VP.ROW_ELIXIR_KEY: 8000}]}
    row = VP.row_state(st)
    assert row["players"][1]["elixir_milli"] == 8000 and st["players"][1]["elixir_milli"] == 6000
    assert row["players"][1]["hand"] == [1, 2, 3, 4]
    assert VP.row_state({"players": [{}, {"elixir_milli": 1}]})["players"][1]["elixir_milli"] == 1


# ---------------------------------------------------------------------------
# the deriver: the drop-in is the tap's play, not a human's
# ---------------------------------------------------------------------------

def _deriver(config):
    tables = EV.EventTables(compositions={GOLEM: {"IceGolemite": 1}, HOG: {"HogRider": 1},
                                          SKELETONS: {"Skeleton": 3}, SPIRITS: {"IceSpirits": 1}},
                            spells={LOG: {"Log"}})
    return EV.EventDeriver(tables, config, own_deck=list(range(8)))


def _drop_in_battle(config, drop_first=225, drop_cls="IceGolemite", drop_xy=(9600.0, 29000.0)):
    """An Ice Golem tapped at 200 at (9500, 31500): its preview from 203 (the landing), the real
    unit dropping in at `drop_first` 2.5 tiles above the aim. Returns the deriver and every own
    landing it reported."""
    d = _deriver(config)
    d.note_own_tap(GOLEM, 200, slot=0, aim_x_mt=9500.0, aim_y_mt=31500.0)
    out = []
    for t in range(201, 290):
        tracks = [tr(1, "IceGolemite", 9600, 31700, 203, t)] if t >= 203 else []
        if t >= drop_first:
            tracks.append(tr(2, drop_cls, drop_xy[0], drop_xy[1], drop_first, t))
        out += d.update(tracks, t).own_landings
    return d, out


def test_the_drop_in_joins_its_taps_play_and_is_no_human_play():
    d, landings = _drop_in_battle(EV.EventConfig.live())
    assert [(x.slot, x.tap_tick, x.land_tick, x.uid) for x in landings] == [(0, 200, 203, 1)]
    assert d.how_of(2) == "drop_in" and d.card_of(2) == GOLEM and d.landing_stats.drop_ins == 1
    att = CT.PlayAttributor()
    att.note_bot_tap(GOLEM, 200, slot=0)
    assert [att.classify(CT.as_own_play(x), mode="auto", lookup=lambda t: None).who for x in landings] == [CT.BOT]
    # the rule off (EventConfig(), the default): the drop-in is an own play no tap explains
    d, landings = _drop_in_battle(EV.EventConfig())
    assert [(x.slot, x.uid) for x in landings] == [(0, 1), (-1, 2)] and d.landing_stats.drop_ins == 0
    att = CT.PlayAttributor()
    att.note_bot_tap(GOLEM, 200, slot=0)
    assert [att.classify(CT.as_own_play(x), mode="auto", lookup=lambda t: None).who for x in landings] == [
        CT.BOT, CT.HUMAN]


def test_the_drop_in_rule_has_limits():
    """Within 2 tiles another unit label still joins (the detector misread the card) and, live, takes
    the tap's card (events PHANTOMS, reguess_ticks: on the 2026-11-01 friendly our Hog dropping in,
    misread as Ice Spirits, kept the Ice Spirits card for its whole life); with the rule off it keeps
    its own label's. A unit of another card 3 tiles away, or one first seen after the horizon (45
    ticks), is a play of its own: a human's."""
    d, landings = _drop_in_battle(EV.EventConfig.live(), drop_cls="IceSpirits", drop_xy=(9600.0, 30000.0))
    assert d.how_of(2) == "drop_in" and d.card_of(2) == GOLEM and len(landings) == 1
    d, landings = _drop_in_battle(EV.EventConfig.live(reguess_ticks=0), drop_cls="IceSpirits",
                                  drop_xy=(9600.0, 30000.0))
    assert d.how_of(2) == "drop_in" and d.card_of(2) == SPIRITS and len(landings) == 1
    d, landings = _drop_in_battle(EV.EventConfig.live(), drop_cls="IceSpirits", drop_xy=(9600.0, 28500.0))
    assert d.how_of(2) == "play" and [x.slot for x in landings] == [0, -1]
    d, landings = _drop_in_battle(EV.EventConfig.live(), drop_first=246)
    assert d.how_of(2) == "play" and [x.slot for x in landings] == [0, -1]


# ---------------------------------------------------------------------------
# the loop: a dry run changes nothing; real taps feed the observer
# ---------------------------------------------------------------------------

class PlainObserver(PS.LiveObserver):
    """The observer as HEAD had it: no HUD / track feed, no board, no row elixir, no refusals."""
    watch = board = row_elixir = refused = None


class GhostPerception(TVP.FakePerception):
    """test_vision_play's scripted perception with the deriver's card info (LivePerception.tap_info)."""

    def tap_info(self, card):
        labels, spell, units = INFO[int(card)]
        return {"labels": labels, "spell": spell, "units": units}


def _loop(tmp_path, rr, observer_cls, *, mode=CT.AUTO, taps=True, leaves=True, refusals=None):
    plan = VP.row_plan({"schema": TLV.V2, "obs_delay": {"ticks": DELAY, "jitter": 3}})
    make = VP.make_observer(TLV.CATALOGUE, plan, law=LAW, observer_cls=observer_cls)
    made = []

    def kept(ctx):
        made.append(make(ctx))
        return made[-1]
    policy = TLV.RowPolicy()
    aim = RD.to_world(TVP.PLAY_TILE[0] * 1000 + 500, TVP.PLAY_TILE[1] * 1000 + 500, OWN)
    script = []
    items = [TVP.menu_vf(TVP.t_of(0)), TVP.menu_vf(TVP.t_of(1))]
    for k in range(40):
        tick = 70 + 2 * k
        hand = list(TVP.HAND) if tick < 103 or not leaves else [7, EMPTY, 6, 4]
        tracks = [VT.Track(uid=1, cls="HogRider", team=OWN, conf=0.9, x_mt=float(aim[0]), y_mt=float(aim[1]) + 300,
                           first_seen_tick=103, last_seen_tick=tick)] if tick >= 104 and taps and leaves else []
        vf = TVP.battle_vf(TVP.t_of(len(items)), tick, hand=hand)
        vf.tracks = tracks
        items.append(vf)
    items += [TVP.end_vf(TVP.t_of(len(items)), tick=150) for _ in range(5)]
    items += [TVP.menu_vf(TVP.t_of(len(items))) for _ in range(3)]
    script = [(TVP.t_of(i), item) for i, item in enumerate(items)]
    decider = VP.Decider(policy=policy, legality=lambda slot, card, x, y: True, config=RD.DecideConfig.live(0.0),
                         state_fn=VP.make_state_fn(TLV.builder(spell_fields=rr.SF),
                                                   SimpleNamespace(track_cards=lambda tracks: {}), plan),
                         row_fn=VP.make_row_fn(rr.obs_row, TLV.SUB, plan, live_compat=False, units="types",
                                               row_inputs=PS.row_inputs),
                         rng=np.random.default_rng(0), observer=kept)
    if refusals is not None:
        decider.legality = VP.RefusalLegality(decider.legality, refusals)
    res = TVP.run_script(tmp_path, script, decider=decider, mode=mode, taps=taps,
                         perception=GhostPerception(script), refusals=refusals)
    return res, policy.rows, made


@pytest.mark.parametrize("mode,taps", [(CT.AUTO, False), (CT.MANUAL, True)])
def test_a_dry_run_builds_every_row_as_before(tmp_path, rr, mode, taps):
    """No tap sent (a dry run, or MANUAL): the new observer changes no row and no decision; the
    observer without the new feeds (HEAD's) gives the same rows byte for byte."""
    new, rows_new, made = _loop(tmp_path / "new", rr, PS.LiveObserver, mode=mode, taps=taps)
    old, rows_old, _ = _loop(tmp_path / "old", rr, PlainObserver, mode=mode, taps=taps)
    assert new.tapper.calls == old.tapper.calls == [] and sorted(rows_new) == sorted(rows_old)
    for t in rows_new:
        for k in rows_new[t]:
            assert np.array_equal(np.asarray(rows_new[t][k]), np.asarray(rows_old[t][k])), (t, k)
    assert TVP.decisions(new.records) == TVP.decisions(old.records)
    assert made[0].own.flights == [] and "own_flight" not in new.records["battle_end"][0]
    assert not any("row_elixir" in d for d in new.records.get("decision", []))


def test_a_real_tap_feeds_the_observer_its_aim_labels_and_the_run(tmp_path, rr):
    """AUTO with taps: the Hog tapped at 100 reaches the observer with its aim and labels (the
    perception's tap_info); its card leaves the slot at 103 (HUD) and its own track there is first
    seen 3 ticks on: the preview, left out of the rows; own_pending lists it until 127 (nothing
    shows it run in this script); the rows' elixir adds its 4 back from 104; battle_end counts it."""
    res, rows, made = _loop(tmp_path, rr, PS.LiveObserver)
    assert len(res.tapper.calls) == 1
    [obs] = made
    [f] = obs.own.flights
    assert f.pending == RD.Pending(HOG, 4, TVP.PLAY_SLOT, 100) and f.labels == frozenset({"HogRider"})
    assert f.aim == tuple(float(v) for v in RD.to_world(TVP.PLAY_TILE[0] * 1000 + 500, TVP.PLAY_TILE[1] * 1000 + 500, OWN))
    assert (f.left_tick, f.ghosts, f.run_how) == (104, {1: 103}, "timeout")
    for t in range(102, 128, 2):
        assert rows[t]["own_pending"].tolist()[0] == HOG + 2, t
        assert rows[t]["unit_ids"].tolist()[0] == 0, t                       # the preview: not shown
    assert rows[130]["own_pending"].tolist()[0] == 0 and rows[130]["unit_ids"].tolist()[0] != 0
    # the row's own bar: the HUD's 7.0 until the card leaves (102); from 104 min(HUD, the cap) + 4,
    # the cap 7.0 - 4 + regen (this script's HUD never pays): the unpaid 7.0 + regen, not 10
    unpaid = 7000 + RD.ElixirCap(LAW).regen_milli(100, 104)
    assert rows[102]["scalars"][0] == np.float32(0.7)
    assert rows[104]["scalars"][0] == np.float32(unpaid / 10_000) and unpaid == 7072
    end = res.records["battle_end"][0]
    assert end["own_flight"] == {"ghost_tracks": 1, "sent": 1, "timeout": 1}
    [d] = TVP.decisions(res.records, CT.ACTION_TAP)
    frames = {e["decision"]["tick"]: e for e in (f["extra"] for f in res.records["frame"]) if "decision" in e}
    assert "row_elixir" not in d and "row_elixir" not in frames[102]
    assert frames[104]["row_elixir"] == 7072 and frames[104]["decision"]["spendable_milli"] < 7000
    # ended at 128: nothing to add; the cap still bites (this HUD never pays): the row shows it
    assert frames[130]["row_elixir"] == frames[130]["elixir_cap"] < 7000


# ---------------------------------------------------------------------------
# latency_probe: D is the decided frame -> the play running
# ---------------------------------------------------------------------------

def test_latency_probe_d_is_the_run_and_the_preview_is_reported_apart(tmp_path):
    """A session (VisionRecorder) with three taps: an Ice Golem whose preview is seen 3 ticks on and
    whose next card changes 25 ticks on, a Hog seen 24 ticks on (no preview) whose change lags to
    38, and a Log refused (its card stays in its slot; no change is its). D = 25: the Golem's
    change and the Hog's unit; the first sightings (3 and 24) are reported apart."""
    from scripts.vision.record import VisionRecorder
    from tools import latency_probe as LP
    clock = TVP.FakeClock()
    rec = VisionRecorder(tmp_path, session="vision_play_20261101_120000", image_every=0, clock=clock,
                         wall_clock=lambda: 0.0)
    taps = {200: (GOLEM, 0, (9, 31)), 202: (HOG, 1, (2, 17)), 260: (LOG, 3, (3, 17))}
    nxt = SKELETONS
    for tick in range(190, 300):
        t = 10.0 + tick / 20.0
        nxt = {225: MUSKETEER, 240: CANNON}.get(tick, nxt)
        hand = [GOLEM if tick < 203 else EMPTY, HOG if tick < 204 else EMPTY, FIREBALL, LOG]
        tracks = []
        if tick >= 203:
            tracks.append(VT.Track(1, "IceGolemite", OWN, 0.9, 9600.0, 31700.0, first_seen_tick=203, last_seen_tick=tick))
        if tick >= 226:
            tracks.append(VT.Track(2, "HogRider", OWN, 0.9, 2600.0, 16200.0, first_seen_tick=226, last_seen_tick=tick))
        tapped = taps.get(tick)
        if tapped is not None:
            card, slot, tile = tapped
            clock.t = t + 0.04
            rec.log("tap", battle=1, tick=tick, card=card, name=TVP.NAMES[card], slot=slot,
                    model_tile=[17 - tile[0], tile[1]], screen_tile=list(tile), ms={"capture_to_tap": 40.0})
        vf = VT.VisionFrame(t_capture=t, tick=tick, tracks=tracks, events=Events(),
                            hud=VT.Hud(hand=hand, next=nxt, elixir_milli=5000, clock_tick=tick),
                            screen=VT.SCREEN_BATTLE, frame_size=(747, 1328))
        rec.write(vf, None, battle=1, state="battle", ms={"stale": 1.0, "perceive": 1.0})
    rec.close()
    classes = LP.CardClasses({"IceGolemite": {"IceGolemite"}, "HogRider": {"HogRider"}, "Log": {"Log"}},
                             {"IceGolemite": "troop", "HogRider": "troop", "Log": "spell"})
    rep = LP.analyse([LP.load_session(rec.session_dir)], classes)
    plays = {p["tick"]: p for p in rep["plays"]}
    assert (plays[200]["run_lag_ticks"], plays[200]["next_lag_ticks"], plays[200]["ghost_lag_ticks"]) == (25, 25, 3)
    assert plays[200]["lag_ticks"] == 3                                     # the first sighting: the preview
    assert (plays[202]["run_lag_ticks"], plays[202]["unit_lag_ticks"], plays[202]["next_lag_ticks"]) == (24, 24, 38)
    assert plays[260]["accepted"] is False and plays[260]["run_lag_ticks"] is None
    assert (rep["D"]["ticks"], rep["D"]["source"], rep["D"]["n"]) == (25, "run", 2)
    assert rep["run_ticks"]["ghost_taps"] == 1 and rep["run_ticks"]["not_accepted"] == 1
    text = "\n".join(LP.format_report(rep))
    assert "D = 25 ticks" in text and "--delay 25" in text and "first sighting (not D" in text


def test_a_refused_tap_in_the_loop_ends_its_flight(tmp_path, rr):
    """The Hog tapped at 100 never leaves its slot: from 108 it is not in own_pending (its card
    stayed), and the refused-tap guard's tap_refused (30 ticks on) ends its flight."""
    from scripts.vision import tap_refusal as TR
    guard = TR.RefusedTapGuard(enforce=True)
    res, rows, made = _loop(tmp_path, rr, PS.LiveObserver, leaves=False, refusals=guard)
    assert len(res.tapper.calls) == 1 and [r["tap_tick"] for r in res.records["tap_refused"]] == [100]
    [f] = made[0].own.flights
    assert f.run_how == "refused" and f.stayed and f.left_tick is None
    assert [t for t in sorted(rows) if rows[t]["own_pending"].tolist()[0]] == [102, 104, 106]
    assert res.records["battle_end"][0]["own_flight"] == {"refused": 1, "sent": 1}


def test_live_perception_gives_the_observer_the_cards_labels():
    perception = VP.LivePerception(hud_reader=None, tracker=None, deriver=_deriver(EV.EventConfig.live()),
                                   geometry=None)
    assert perception.tap_info(SKELETONS) == {"labels": ("Skeleton",), "spell": False, "units": 3}
    assert perception.tap_info(LOG) == {"labels": ("Log",), "spell": True, "units": 1}
    assert perception.tap_info(MUSKETEER) == {"labels": (), "spell": False, "units": 1}   # not in these tables
    assert VP.LivePerception(hud_reader=None, tracker=None, deriver=None, geometry=None).tap_info(HOG) == {}


# ---------------------------------------------------------------------------
# one-frame HUD misreads are not run signs; a withdrawn refusal runs after all
# ---------------------------------------------------------------------------

def test_a_one_frame_next_card_flicker_is_no_run_and_a_change_read_twice_is():
    """The HUD's next card flickers back and forth right after a run (on the 11-01 friendly:
    30 -> 38 -> 30 -> 38 at 2143/2145/2148). An Ice Golem tapped at 200: the next card reads
    Musketeer on one frame (222) and Skeletons again at 223: no run, still listed at 224. Read
    twice in a row (226, 227): the play ran, dated the first of the two frames (226)."""
    obs = observer()
    aim = (9500.0, 31500.0)
    hand = [GOLEM, HOG, FIREBALL, LOG]
    gone = [EMPTY, HOG, FIREBALL, LOG]
    obs.watch(200, hand, SKELETONS, [])
    send(obs, GOLEM, 200, 0, aim)
    nxt = {222: MUSKETEER, 226: MUSKETEER, 227: MUSKETEER}
    for t in range(201, 228):
        obs.watch(t, hand if t < 203 else gone, nxt.get(t, SKELETONS), [])
        if t == 224:
            assert pending(obs, t) == [[GOLEM + 2, 0]]
    [f] = obs.own.flights
    assert (f.run_how, f.run_tick) == ("next", 226)


def test_a_one_frame_slot_misread_is_no_refill():
    """The tap's slot misreads another card on one frame (221) and reads empty again: no run.
    The slot refilled for real reads the new card on 226 and 227: the run, dated 226."""
    obs = observer()
    hand = [GOLEM, HOG, FIREBALL, LOG]
    gone = [EMPTY, HOG, FIREBALL, LOG]
    refill = [SKELETONS, HOG, FIREBALL, LOG]
    obs.watch(200, hand, SKELETONS, [])
    send(obs, GOLEM, 200, 0, (9500.0, 31500.0))
    for t in range(201, 228):
        h = hand if t < 203 else refill if t in (221, 226, 227) else gone
        obs.watch(t, h, SKELETONS, [])
        if t == 224:
            assert pending(obs, t) == [[GOLEM + 2, 0]]
    [f] = obs.own.flights
    assert (f.run_how, f.run_tick) == ("refill", 226)


def test_a_withdrawn_refusal_ends_its_flight_as_run():
    """tap_refusal_retracted: a Log refused at +30 whose card then left after all. The flight
    leaves "refused" and ends "retracted" at the guard's evidence tick, so its later run signs
    are not handed to a younger tap."""
    obs = observer()
    send(obs, LOG, 619, 3, (3500.0, 17500.0))
    obs.watch(620, [EMPTY, GOLEM, FIREBALL, LOG], HOG, [])
    assert obs.refused(LOG, 619, 3)
    assert obs.retracted(LOG, 619, 3, 655)
    [f] = obs.own.flights
    assert (f.run_how, f.run_tick) == ("retracted", 655)
    assert not obs.retracted(LOG, 619, 3, 656)                 # once only
    assert obs.own.counts["refused"] == 0 and obs.own.counts["retracted"] == 1
