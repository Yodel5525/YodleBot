"""Lane C2 (E14 live): the live runner builds v2 observations from the screen.

tools/public_state.py: OpponentElixir (the opponent's bar counted from the plays perception
reports, rs_rows.SeenElixir's count with lag 0), OwnTaps (our sent taps until they land),
LiveObserver (both, per battle), state(..., observer=) and row_inputs. scripts/vision_play.py:
row_plan picks legacy or v2 rows from the checkpoint meta, make_state_fn / make_row_fn /
make_observer wire them, and the loop feeds the observer every frame's events and each tap.
A legacy checkpoint's rows are built by exactly the calls they always were.

No engine, model or device: a small synthetic catalogue (elixir in column 2, as royalesim's
CATALOGUE_FIELDS put it; Hog 2.6 is 0-7 and Giant 10, as tests/pipeline/test_vision_play.py's
names), the shipped data/vision/classes.json, rs_elixir's constant law, and the scripted loop of
test_vision_play (battle from tick 90, decisions every 2 ticks to 120, a Hog tapped at 100).

Elixir arithmetic (rs_elixir, the constants): one elixir is 28000 fine units, a tick at 1x adds
500, the bar starts at 6 elixir (168000) at tick 0 and caps at 280000 (full from tick 224);
milli = fine * 1000 // 28000. A Giant costs 5 (140000), Arrows 3 (84000), Knight 3.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import FIELD

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import vision_play as VP  # noqa: E402
from scripts.vision import control as CT  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from scripts.vision.types import Events, OppPlay, OwnLanding  # noqa: E402
from tests.pipeline import test_vision_play as TVP  # noqa: E402  (its scripted loop)
from tools import public_state as PS  # noqa: E402

import rs_schema  # noqa: E402  (tools/ is on sys.path once public_state is imported)

RD = VP.RD
SUB = 18
LAW = PS.rs_elixir.ElixirLaw.constants()
CLASSES = PS.load_classes()

#: (name, elixir, card_kind) in catalogue id order: test_vision_play's NAMES, the first 12.
CARD_ROWS = (("Cannon", 3, "BUILDING"), ("Fireball", 4, "SPELL"), ("HogRider", 4, "TROOP"),
             ("IceGolemite", 2, "TROOP"), ("IceSpirits", 1, "TROOP"), ("Musketeer", 4, "TROOP"),
             ("Skeletons", 1, "TROOP"), ("Log", 2, "SPELL"), ("Knight", 3, "TROOP"),
             ("Archer", 3, "TROOP"), ("Giant", 5, "TROOP"), ("Arrows", 3, "SPELL"))
#: catalogue_json rows: name, placement, elixir, count, radius, flying, hitpoints, footprint, kind.
CATALOGUE = [[n, 0, cost, 1, 0, False, 0, 0, kind] for n, cost, kind in CARD_ROWS]
CARDS = [row[0] for row in CATALOGUE]
CID = {n: i for i, n in enumerate(CARDS)}
HOG = list(range(8))
GIANT, ARROWS, KNIGHT, SKELETONS = CID["Giant"], CID["Arrows"], CID["Knight"], CID["Skeletons"]
UNITS = ["Giant", "HogRider", "Knight", "Skeleton", "Skeleton_EV1"]
UID = {n: i for i, n in enumerate(UNITS)}
PRIMARY = {"Giant": "Giant", "HogRider": "HogRider", "Knight": "Knight", "Skeletons": "Skeleton"}
HP11 = {"Giant": 3968, "HogRider": 1697, "Knight": 1766, "Skeletons": 81}
SPAWN_TABLE = {
    "format": 1, "card_names": list(CARDS), "unit_types": list(UNITS),
    "primary": {str(CID[c]): UID[u] for c, u in PRIMARY.items()},
    "all": {str(CID[c]): [UID[u]] for c, u in PRIMARY.items()},
    "hp": {str(CID[c]): {"11": [["own", PRIMARY[c], hp]]} for c, hp in HP11.items()},
}
V2 = rs_schema.SCHEMA_V2


def f16(xs):
    return np.asarray(xs, dtype=np.float16).tolist()


def milli(fine):
    """state_json's elixir_milli of a bar of `fine` units."""
    return fine * 1000 // 28000


def play(card, tick, exec_tick=None):
    return OppPlay(card=card, tick=tick, exec_tick=tick if exec_tick is None else exec_tick)


def builder(**kw):
    args = dict(deck=HOG, catalogue=CATALOGUE, fields=FIELD, sub=SUB)
    args.update(kw)
    return PS.PublicStateBuilder(CLASSES, SPAWN_TABLE, **args)


def frame(tracks=(), tick=300, **hud):
    hud = {"hand": [7, 2, 6, 4], "next": 0, "elixir_milli": 7000, "clock_tick": tick, **hud}
    return VT.VisionFrame(t_capture=1.0, tick=tick, tracks=list(tracks), hud=VT.Hud(**hud),
                          screen=VT.SCREEN_BATTLE)


def track(uid, cls, team, x, y, *, first, form=0, last=300):
    return VT.Track(uid=uid, cls=cls, team=team, conf=0.9, x_mt=float(x), y_mt=float(y),
                    form=form, first_seen_tick=first, last_seen_tick=last)


@pytest.fixture
def rr(monkeypatch):
    """tools/rs_rows as the tools import it, set up as _setup would set it for this catalogue
    and these unit types (conftest's FIELD, the v2 spell / form tables, 2 ability buttons),
    every other setting at its default; restored after the test."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    for name, value in {
            "F": dict(FIELD), "LIVE": False, "UNITS": rr.UNITS_TYPES, "VC": None, "VCT": None,
            "VISIBILITY": rr.VIS_LEGACY, "PRESSES": "off", "BUTTONS": 2, "HAND_SIZE": 4,
            "ROW_SCHEMA": rs_schema.SCHEMA_LEGACY, "DECISION": rr.DecisionRule(),
            "CARD_FLY": {i: False for i in range(len(CATALOGUE))},
            "CARD_BUILDING": {i: row[8] == "BUILDING" for i, row in enumerate(CATALOGUE)},
            "SF": rr.spell_index(), "UNIT_FORMS": rr.unit_form_table(UNITS),
            "SPELL_CARDS": rr.spell_cards(CATALOGUE)}.items():
        monkeypatch.setattr(rr, name, value)
    assert rr.SPELL_CARDS == frozenset({CID["Fireball"], CID["Log"], ARROWS})
    assert rr.UNIT_FORMS == (0.0, 0.0, 0.0, 0.0, 0.5)
    return rr


# ---------------------------------------------------------------------------
# the opponent's elixir
# ---------------------------------------------------------------------------

def test_opponent_elixir_pays_each_seen_play_at_its_execution_tick():
    oe = PS.OpponentElixir(CATALOGUE, law=LAW)
    assert oe.view(100) == (milli(168000 + 50000), True) == (7785, True)
    assert oe.record([play(GIANT, 300)]) == 1
    # full (280000) from tick 224; the Giant's 140000 is gone at 300, and 300 regenerates 500
    assert oe.view(300) == (5000, True)
    assert oe.view(301) == (milli(140500), True) == (5017, True)
    oe.record([play(ARROWS, 500)])
    assert oe.view(500) == (milli(140500 + 199 * 500 - 84000), True) == (5571, True)
    assert oe.view(520) == (milli(156500 + 19 * 500), True) == (5928, True)
    assert oe.recorded == 2 and oe.unpriced == 0


def test_a_play_reported_late_or_early_is_paid_at_its_own_tick():
    oe = PS.OpponentElixir(CATALOGUE, law=LAW)
    assert oe.view(320) == (10_000, True)                  # nothing seen yet: a full bar
    oe.record([OppPlay(card=GIANT, tick=330, exec_tick=300)])   # first seen 30 ticks after it ran
    assert oe.view(330) == (milli(140500 + 29 * 500), True) == (5535, True)
    # a play dated after the asked tick (a clock read early) waits until its tick
    oe.record([play(KNIGHT, 400)])
    assert oe.view(399) == (milli(140500 + 98 * 500), True) == (6767, True)
    assert oe.view(400) == (milli(140500 + 99 * 500 - 84000), True) == (3785, True)
    # no exec_tick: the first sighting is the tick
    nt = PS.OpponentElixir(CATALOGUE, law=LAW)
    nt.record([OppPlay(card=GIANT, tick=300)])
    assert nt.view(301) == (5017, True)


def test_the_live_count_is_rs_rows_seen_elixir_with_lag_zero(rr):
    """The rule the v2 rows were built with (rs_rows.SeenElixir, lag 0 as rs_eval's seats count),
    frame by frame: what is reported on each frame, then the view at that frame's tick."""
    frames = [(100, []), (300, [play(GIANT, 300)]), (301, []), (500, []),
              (510, [OppPlay(card=ARROWS, tick=510, exec_tick=500)]), (520, []),
              (700, [play(KNIGHT, 700), play(SKELETONS, 700)]), (2500, []),
              (2600, [play(GIANT, 2600)]), (4900, [])]
    ours = PS.OpponentElixir(CATALOGUE, law=LAW)
    rows = rr.SeenElixir(CATALOGUE, law=LAW, lag=0)
    got, want = [], []
    for tick, plays in frames:
        ours.record(plays)
        for p in plays:
            rows.record(0, p.exec_tick, p.card)
        got.append(ours.view(tick))
        want.append(rows.view(1, tick))
    assert got == want
    # the Arrows seen at 510 were paid at 500: 240000 - 84000 + 500, then 9 ticks
    assert got[3] == (milli(240000), True) == (8571, True) and got[4] == (5750, True)


def test_the_count_is_exact_only_when_every_play_can_have_been_seen():
    # an unaffordable play means one was missed: 88000 at 120 cannot pay a second Giant
    oe = PS.OpponentElixir(CATALOGUE, law=LAW)
    oe.record([play(GIANT, 100), play(GIANT, 120)])
    assert oe.view(120) == (0, False)
    # no detector, or watching from after the deploy lockout (tick 90) and one more displayed
    # second (LATE_MARGIN_TICKS 20: a first clock reading of 2:55, tick 110, is in time): never
    # exact
    assert PS.DEPLOY_LOCKOUT_TICKS + PS.LATE_MARGIN_TICKS == 110
    assert PS.OpponentElixir(CATALOGUE, law=LAW, blind=True).view(100) == (7785, False)
    assert PS.OpponentElixir(CATALOGUE, law=LAW, start_tick=111).view(100) == (7785, False)
    assert PS.OpponentElixir(CATALOGUE, law=LAW, start_tick=110).view(100) == (7785, True)
    assert PS.OpponentElixir(CATALOGUE, law=LAW, start_tick=90).view(100) == (7785, True)
    assert PS.OpponentElixir(CATALOGUE, law=LAW, start_tick=None).view(100) == (7785, True)
    # a card the catalogue does not price is skipped, and the count is not exact
    un = PS.OpponentElixir(CATALOGUE, law=LAW)
    assert un.record([play(99, 150)]) == 0 and un.unpriced == 1
    assert un.view(200) == (milli(168000 + 100000), False) == (9571, False)


# ---------------------------------------------------------------------------
# our own taps in flight
# ---------------------------------------------------------------------------

def test_own_taps_are_pending_until_they_run():
    """OwnTaps (delay 24): a sent tap is in own_pending from the tick after it until it is seen
    to RUN, refused, or 24 + 3 ticks old (the rows list a play while tap < t <= exec, exec 24 +- 3
    ticks after the tap). The deriver's first sighting soon after the tap is the client's
    placement preview and ends nothing; a sighting 20 or more ticks on is the play running."""
    taps = PS.OwnTaps(delay=24)
    taps.sent(2, 100, slot=1, cost=4)                     # Hog Rider
    taps.sent(7, 104, slot=0, cost=2)                     # Log
    assert taps.taps == [RD.Pending(2, 4, 1, 100), RD.Pending(7, 2, 0, 104)]
    # a tap at the asked tick is that decision's own; one after it is not sent yet
    assert taps.inputs(100) == [] and taps.inputs(104) == [(4, 20)]
    # (card + 2, tap + delay - now), soonest first; landing-time rows (delay 0) show none
    assert taps.inputs(106) == [(4, 18), (9, 22)]
    assert taps.inputs(106, 0) == [] and PS.OwnTaps().inputs(106) == []
    # the Hog's first sighting 10 ticks on (the preview) ends nothing ...
    assert taps.landed([OwnLanding(card=2, slot=1, tap_tick=100, land_tick=110)]) == 0
    assert taps.inputs(112) == [(4, 12), (9, 16)]
    # ... nor a human's play (slot -1), another tap's landing or a miss; the Hog's unit 25 on does
    assert taps.landed([OwnLanding(card=7, slot=-1, tap_tick=104, land_tick=125),
                        OwnLanding(card=7, slot=0, tap_tick=99, land_tick=125),
                        OwnLanding(card=7, slot=0, tap_tick=104, land_tick=None)]) == 0
    assert taps.landed([OwnLanding(card=2, slot=1, tap_tick=100, land_tick=125)]) == 1
    assert taps.inputs(126) == [(9, 2)] and taps.taps == [RD.Pending(7, 2, 0, 104)]
    # one never seen to run: due (0 ticks to land) until its tap + 24 + 3, then gone
    assert taps.inputs(128) == [(9, 0)] and taps.inputs(131) == [(9, 0)]
    assert taps.inputs(132) == [] and taps.taps == []
    assert taps.summary() == {"landing": 1, "sent": 2, "timeout": 1}
    # the rows' delay caps the window, and TAP_MAX_AGE (60) still bounds it
    long = PS.OwnTaps(delay=58)
    long.sent(0, 200, slot=2, cost=3)
    assert long.inputs(260) == [(2, 0)] and long.inputs(261) == []


# ---------------------------------------------------------------------------
# the state and the row
# ---------------------------------------------------------------------------

def test_the_state_carries_the_observer_inputs_and_row_inputs_reads_them():
    obs = PS.LiveObserver(CATALOGUE, delay=24, law=LAW, start_tick=40)
    obs.events(Events([play(GIANT, 290)], []))
    obs.events(None)                                      # a frame without events
    obs.sent(2, 290, slot=1, cost=4)
    b = builder()
    st = b.state(frame(), observer=obs)
    # 280000 at 290 less the Giant, + 500 at 290, + 9 ticks to 300: 145000
    assert st["observer"] == {"opp_elixir": [milli(145000), True], "own_pending": [[4, 14]],
                              "delay": 24}
    assert milli(145000) == 5178
    json.dumps(st)                                        # plain JSON values
    assert PS.row_inputs(st) == {"schema": V2, "opp_elixir": (5178, True), "pending": [(4, 14)]}
    assert PS.row_inputs(st, buttons=[]) == {"schema": V2, "opp_elixir": (5178, True),
                                             "pending": [(4, 14)], "buttons": []}
    # precomputed inputs, and the one-call form
    given = b.state(frame(), observer={"opp_elixir": (6000, False), "own_pending": [(5, 3)]})
    assert given["observer"] == {"opp_elixir": [6000, False], "own_pending": [[5, 3]], "delay": 0}
    one = PS.rs_state_from_vision(frame(), HOG, CLASSES, SPAWN_TABLE, catalogue=CATALOGUE,
                                  fields=FIELD, observer={"opp_elixir": [7000, True]})
    assert one["observer"] == {"opp_elixir": [7000, True], "own_pending": [], "delay": 0}
    # legacy: no observer, no key, and no v2 keywords from it
    legacy = b.state(frame())
    assert "observer" not in legacy
    assert {k: v for k, v in st.items() if k != "observer"} == legacy
    with pytest.raises(ValueError, match="no observer inputs"):
        PS.row_inputs(legacy)


def test_a_v2_row_from_the_screen(rr):
    """An opponent's evolved Skeleton, their Fireball in flight and our rolling Log at tick 300;
    the opponent played a Giant at 290 and we tapped the Hog at 290 (delay 24)."""
    obs = PS.LiveObserver(CATALOGUE, delay=24, law=LAW, start_tick=40)
    obs.events(Events([play(GIANT, 290)], []))
    obs.sent(2, 290, slot=1, cost=4)
    tracks = [track(11, "Skeleton", VT.ENEMY_TEAM, 9000, 12000, first=200, form=1),
              track(21, "Fireball", VT.ENEMY_TEAM, 9000, 20000, first=290),
              track(22, "the-log", VT.OWN_TEAM, 3500, 12000, first=295)]
    st = builder(spell_fields=rr.SF).state(frame(tracks), track_cards={11: SKELETONS},
                                           opp_deck=[GIANT, SKELETONS], observer=obs)
    assert [e[FIELD["unit_type"]] for e in st["entities"]] == [UID["Skeleton_EV1"]]
    mem = {"seen": {0: {GIANT + 2}, 1: set()}, "last": {0: [(GIANT + 2, 290)], 1: []}}
    row = rr.obs_row(st, VT.OWN_TEAM, SUB, mem, 300, **PS.row_inputs(st))
    assert rs_schema.V2.check_row(row, buttons=2) == []
    # the evolved unit: own frame x 18000 - 9000, y 12000; full bar; form 0.5, no burrow
    assert row["unit_ids"].tolist() == [UID["Skeleton_EV1"] + 2] + [0] * 47
    assert row["unit_feats"][0].tolist() == f16([0, 0.5, 0.375, 1, 0, 0, 0, 0.5, 0])
    # elixir 7000, 15 s, no 2x / overtime / 3x, the counted bar, exact
    assert np.array_equal(row["scalars"], np.asarray([0.7, 15000 / 300000, 0, 0, 0, 0.5178, 1],
                                                     dtype=np.float32))
    assert row["own_pending"].tolist() == [4, 0, 0, 0]
    assert row["own_pending_land"].tolist() == f16([14 / 40, 0, 0, 0])
    # spells nearest our king (9000, 29000) first: their Fireball (flown 10), our Log (5)
    assert row["spell_ids"].tolist() == [CID["Fireball"] + 2, CID["Log"] + 2] + [0] * 6
    assert row["spell_feats"][:2].tolist() == [
        f16([0, 0.5, 0.625, 0.5, 0.625, 10 / 40]),
        f16([1, 14500 / 18000, 0.375, 14500 / 18000, 0.375, 5 / 40])]
    # our hand (Log, Hog, Skeletons, Ice Spirits) and next (Cannon); no evolution, no button
    assert row["hand"].tolist() == [9, 4, 8, 6] and row["next"] == 2
    assert not row["hand_feats"].any()
    assert row["ability_card"].tolist() == [0, 0] and not row["ability_state"].any()
    assert row["opp_last"].tolist() == [GIANT + 2, 0, 0, 0]
    assert row["opp_last_age"].tolist() == f16([10 / 400, 1, 1, 1])


# ---------------------------------------------------------------------------
# vision_play: which rows, and the calls that build them
# ---------------------------------------------------------------------------

def test_row_plan_picks_the_rows_by_the_checkpoint_schema():
    legacy = VP.RowPlan(rs_schema.SCHEMA_LEGACY, False, 0)
    assert VP.row_plan({}) == VP.row_plan(None) == legacy
    assert VP.row_plan({"schema": "rs_rows/legacy", "obs_delay": {"ticks": 24}}) == legacy
    assert VP.row_plan({"schema": V2, "obs_delay": {"ticks": 24, "jitter": 3}}) == \
        VP.RowPlan(V2, True, 24)
    assert VP.row_plan({"schema": "v2"}) == VP.RowPlan(V2, True, 0)
    assert VP.RowPlan(V2, True, 24).to_json() == {"schema": V2, "v2": True, "obs_delay": 24}
    with pytest.raises(ValueError, match="'rs_rows/v3'"):
        VP.row_plan({"schema": "rs_rows/v3"})


def test_legacy_rows_are_built_by_the_calls_they_always_were():
    plan = VP.row_plan({"live_compat": True})
    calls = []

    def obs_row(*args, **kw):
        calls.append((args, kw))
        return {"tick": args[4]}

    row_fn = VP.make_row_fn(obs_row, SUB, plan, live_compat=True, units="types")
    state, view = {"tick": 100}, {"seen": {}, "last": {}}
    assert row_fn(state, view, 100) == {"tick": 100}
    assert calls == [((state, VT.OWN_TEAM, SUB, view, 100), {"live": True, "units": None})]
    calls.clear()
    VP.make_row_fn(obs_row, SUB, plan, live_compat=False, units="vision")(state, view, 7)
    assert calls == [((state, VT.OWN_TEAM, SUB, view, 7), {"live": False, "units": "vision"})]
    got = []
    fake = SimpleNamespace(state=lambda vf, **kw: got.append(kw) or {})
    perception = SimpleNamespace(track_cards=lambda tracks: {5: 2})
    ctx = VP.BattleContext(1, 0.0, 40, opp_cards=[10], pending=[RD.Pending(2, 4, 1, 90)])
    ctx.observer = "never read for legacy rows"
    VP.make_state_fn(fake, perception, plan)(frame(), ctx)
    assert got == [{"opp_deck": [10], "track_cards": {5: 2}, "pending_cost": 4}]
    assert VP.make_observer(CATALOGUE, plan) is None
    decider = VP.Decider(policy=TVP.FakePolicy(), legality=lambda *a: True,
                         config=RD.DecideConfig.live(0.0), state_fn=lambda vf, c: {},
                         row_fn=row_fn)
    assert decider.observer_for(ctx) is None


def test_v2_rows_add_the_observer_inputs():
    plan = VP.row_plan({"schema": V2, "obs_delay": {"ticks": 24}})
    calls = []
    row_fn = VP.make_row_fn(lambda *a, **kw: calls.append((a, kw)) or {}, SUB, plan,
                            live_compat=True, units="types", row_inputs=PS.row_inputs)
    state = {"observer": {"opp_elixir": [6000, True], "own_pending": [[4, 14]]}}
    row_fn(state, {}, 300)
    assert calls == [((state, VT.OWN_TEAM, SUB, {}, 300),
                      {"live": True, "units": None, "schema": V2, "opp_elixir": (6000, True),
                       "pending": [(4, 14)]})]
    got = []
    fake = SimpleNamespace(state=lambda vf, **kw: got.append(kw) or {})
    ctx = VP.BattleContext(1, 0.0, 120)
    ctx.observer = object()
    VP.make_state_fn(fake, SimpleNamespace(track_cards=lambda tracks: {}), plan)(frame(), ctx)
    assert got == [{"opp_deck": [], "track_cards": {}, "pending_cost": 0,
                    "observer": ctx.observer}]
    # one observer per battle: the plan's delay, watching from the battle's start tick
    make = VP.make_observer(CATALOGUE, plan, law=LAW, observer_cls=PS.LiveObserver)
    obs = make(ctx)
    assert isinstance(obs, PS.LiveObserver) and obs.delay == 24
    assert obs.opp.start_tick == 120 and obs.opp.late          # started after the lockout
    # a first clock reading of 2:55 (tick 110: 2:56 was missed) is still in time
    assert not make(VP.BattleContext(4, 0.0, 110)).opp.late and make(VP.BattleContext(5, 0.0, 111)).opp.late
    assert obs.inputs(300)["opp_elixir"] == [10_000, False]    # full from tick 224
    blind = VP.make_observer(CATALOGUE, plan, law=LAW, blind=True,
                             observer_cls=PS.LiveObserver)(VP.BattleContext(2, 0.0, 30))
    assert blind.opp.blind and not blind.opp.late and blind.inputs(100)["opp_elixir"][1] is False
    # the runner's own import of tools/public_state (top level, as main() imports it)
    default = VP.make_observer(CATALOGUE, plan, law=LAW)(VP.BattleContext(3, 0.0, 30))
    assert type(default).__name__ == "LiveObserver" and default.delay == 24


# ---------------------------------------------------------------------------
# the loop (test_vision_play's scripted battle)
# ---------------------------------------------------------------------------

class RowPolicy(TVP.FakePolicy):
    """test_vision_play's policy (the Hog in slot 1 at tick 100, waits otherwise), keeping the
    row of each decision by its tick."""

    def __init__(self, play_ticks=(100,)):
        super().__init__(play_ticks)
        self.rows = {}

    def batch(self, row):
        self.rows[int(row["tick"])] = row
        return row


def v2_decider(rr, policy, *, observer=None, made=None):
    """A v2 Decider made by vision_play's helpers (delay 24) on the real builder and obs_row;
    each observer it makes is appended to `made`, unless `observer` replaces the factory."""
    plan = VP.row_plan({"schema": V2, "obs_delay": {"ticks": 24, "jitter": 3}})
    if observer is None:
        make = VP.make_observer(CATALOGUE, plan, law=LAW, observer_cls=PS.LiveObserver)

        def kept(ctx):
            made.append(make(ctx))
            return made[-1]
        observer = kept
    return VP.Decider(policy=policy, legality=lambda slot, card, x, y: True,
                      config=RD.DecideConfig.live(0.0),
                      state_fn=VP.make_state_fn(builder(spell_fields=rr.SF),
                                                SimpleNamespace(track_cards=lambda tracks: {}),
                                                plan),
                      row_fn=VP.make_row_fn(rr.obs_row, SUB, plan, live_compat=False,
                                            units="types", row_inputs=PS.row_inputs),
                      rng=np.random.default_rng(0), observer=observer)


def test_the_loop_feeds_the_observer_and_the_rows_show_it(tmp_path, rr):
    """The opponent's Giant is reported at 104; our Hog, tapped at 100, is seen landing at 110.
    The opponent's bar: 168000 + 500 t before 104; at 104 220000 less the Giant, 80000, then
    500 a tick. Our tap shows as pending (100 + 24 - t ticks to land) from 102 to 106: this
    script's HUD hand never changes, so the Hog still reads in its slot 8 ticks after the tap
    (the client did not take it: a refusal on its way) and from 108 it is not listed; the
    landing 10 ticks on (the client's preview) ends nothing (tests/pipeline/test_own_flight.py
    has the real sequence)."""
    giant = OppPlay(card=GIANT, tick=104, exec_tick=104, x_mt=4000.0, y_mt=8000.0)
    landing = OwnLanding(card=2, slot=TVP.PLAY_SLOT, tap_tick=100, land_tick=110)
    policy, made = RowPolicy(), []
    res = TVP.run_script(tmp_path, TVP.standard_script(opp_at={104: [giant]},
                                                       own_at={110: [landing]}),
                         decider=v2_decider(rr, policy, made=made))
    assert len(res.tapper.calls) == 1 and res.runner.taps == 1
    assert dict(res.runner.errors) == {}
    rows = policy.rows
    assert sorted(rows) == list(range(90, 121, 2))
    want = {90: 7607, 100: 7785, 102: 7821, 104: 2857, 106: 2892, 120: 3142}
    for t, m in want.items():
        bar = 168000 + 500 * t if t < 104 else 80000 + 500 * (t - 104)
        assert m == milli(bar), t
        assert rows[t]["scalars"][5] == np.float32(m / 10_000) and rows[t]["scalars"][6] == 1.0
    for t in rows:
        due = 102 <= t <= 106
        assert rows[t]["own_pending"].tolist() == ([4, 0, 0, 0] if due else [0, 0, 0, 0]), t
        land = [(100 + 24 - t) / 40, 0, 0, 0] if due else [0, 0, 0, 0]
        assert rows[t]["own_pending_land"].tolist() == f16(land), t
        assert rs_schema.V2.check_row(rows[t], buttons=2) == []
    assert rows[102]["own_pending_land"][0] == np.float16(0.55)
    # one observer for the battle, watching from the battle's first clock reading (tick 70; it
    # started at 90): the Giant counted; the Hog not taken (its card stayed), not ended yet
    [obs] = made
    assert obs.opp.start_tick == 70 and not obs.opp.late and obs.opp.recorded == 1
    [hog] = obs.own.flights
    assert hog.stayed and hog.left_tick is None and hog.run_how is None
    assert hog.aim == tuple(RD.to_world(TVP.PLAY_TILE[0] * 1000 + 500, TVP.PLAY_TILE[1] * 1000 + 500, VT.OWN_TEAM))
    # the tap's decision record says what the row was given
    [d] = TVP.decisions(res.records, CT.ACTION_TAP)
    assert d["tick"] == 100 and d["opp_elixir"] == [7785, True] and d["own_pending"] == []


def test_legacy_rows_in_the_loop_get_no_observer(tmp_path):
    """A legacy decider made by the same helpers: obs_row is called as always, the state has no
    observer block and the decision records carry nothing new."""
    plan = VP.row_plan({"live_compat": True})
    calls = []

    def obs_row(state, team, sub, view, tick, **kw):
        calls.append((team, sub, tick, kw, "observer" in state))
        return {"tick": tick}

    decider = VP.Decider(policy=TVP.FakePolicy((100,)), legality=lambda slot, card, x, y: True,
                         config=RD.DecideConfig.live(0.0),
                         state_fn=VP.make_state_fn(builder(), SimpleNamespace(
                             track_cards=lambda tracks: {}), plan),
                         row_fn=VP.make_row_fn(obs_row, SUB, plan, live_compat=True,
                                               units="types"),
                         rng=np.random.default_rng(0), observer=VP.make_observer(CATALOGUE, plan))
    giant = OppPlay(card=GIANT, tick=104, exec_tick=104)
    res = TVP.run_script(tmp_path, TVP.standard_script(opp_at={104: [giant]}), decider=decider)
    assert [c[2] for c in calls] == list(range(90, 121, 2))
    assert all(c[:2] == (VT.OWN_TEAM, SUB) and c[3] == {"live": True, "units": None}
               and not c[4] for c in calls)
    assert len(res.tapper.calls) == 1
    [d] = TVP.decisions(res.records, CT.ACTION_TAP)
    assert "opp_elixir" not in d and "own_pending" not in d
    assert "observer_error" not in res.runner.errors


def test_a_v2_battle_without_its_observer_never_taps(tmp_path, rr):
    """The observer cannot be made: logged once; every v2 row then refuses (no observer
    inputs in the state), so no decision is made and nothing is tapped."""
    def broken(ctx):
        raise RuntimeError("no counter")

    res = TVP.run_script(tmp_path, TVP.standard_script(),
                         decider=v2_decider(rr, TVP.FakePolicy((100,)), observer=broken))
    assert res.tapper.calls == [] and res.runner.taps == 0
    assert res.runner.errors["observer_error"] == 1
    assert res.runner.errors["decide_error"] == 16          # ticks 90, 92, ..., 120
    first = res.records["decide_error"][0]
    assert first["error"].startswith("ValueError: this state has no observer inputs")
