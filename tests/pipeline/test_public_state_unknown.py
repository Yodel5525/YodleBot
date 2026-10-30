"""N02 in tools/public_state.py (docs/NEW_CARDS_PLAN.md 2.2): unknown enemy units on the board
and UNKNOWN plays in the opponent's elixir count, and how both reach the token model's rows.

* PublicStateBuilder(unknown_units=True), OFF by default: an ENEMY track of a label the class
  table cannot place stays on the board as card_id -1, unit_type -1 (row token 1 in
  --live-compat, unit-type and vision rows) instead of being dropped. Off, the state is exactly
  the old one.
* OpponentElixir pays an OppPlay's cost when it has one (a Spirit Empress form; an UNKNOWN play's
  expected cost) and calls a count with an UNKNOWN play not exact.
* An UNKNOWN play (card -1) is token 1 in opp_seen / opp_last of legacy and v2 rows, and its cost
  comes off the v2 row's opponent elixir, whose exact flag drops to 0. With the deriver's switch
  off the v2 row is test_live_v2's, number for number.

The fixtures are tests/pipeline/test_public_state.py's (legacy rows) and test_live_v2.py's (v2
rows): synthetic catalogues and spawn tables, the shipped classes.json, rs_elixir's constant law.
No engine, model or device.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import events as EV  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from scripts.vision.types import Events, OppPlay  # noqa: E402
from tests.pipeline import test_live_v2 as TLV  # noqa: E402
from tests.pipeline import test_public_state as TPS  # noqa: E402
from tests.pipeline.conftest import FIELD  # noqa: E402
from tools import public_state as PS  # noqa: E402

import rs_memory  # noqa: E402  (tools/ is on sys.path once public_state is imported)
import rs_schema  # noqa: E402

SUB = 18
E, O, U = VT.ENEMY_TEAM, VT.OWN_TEAM, VT.UNKNOWN_TEAM
ODD = "zzz-not-a-class"                 # a label the class table cannot place

TRACKS = [TPS.track(201, "Knight", E, 3500, 18000), TPS.track(101, "HogRider", O, 3500, 14000),
          TPS.track(301, ODD, E, 9000, 9000, first=1490), TPS.track(302, ODD, O, 9000, 25000)]


@pytest.fixture
def rr(monkeypatch):
    """tools/rs_rows for legacy rows, as test_public_state's fixture sets it."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    monkeypatch.setattr(rr, "F", dict(FIELD))
    monkeypatch.setattr(rr, "LIVE", False)
    monkeypatch.setattr(rr, "CARD_FLY", {i: bool(row[5]) for i, row in enumerate(TPS.CATALOGUE)})
    monkeypatch.setattr(rr, "CARD_BUILDING", {i: row[8] == "BUILDING"
                                              for i, row in enumerate(TPS.CATALOGUE)})
    monkeypatch.setattr(rr, "VC", TPS.CLASSES.lookup(TPS.UNITS))
    return rr


@pytest.fixture
def rr2(monkeypatch):
    """tools/rs_rows for v2 rows, as test_live_v2's fixture sets it."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    for name, value in {
            "F": dict(FIELD), "LIVE": False, "UNITS": rr.UNITS_TYPES, "VC": None, "VCT": None,
            "VISIBILITY": rr.VIS_LEGACY, "PRESSES": "off", "BUTTONS": 2, "HAND_SIZE": 4,
            "ROW_SCHEMA": rs_schema.SCHEMA_LEGACY, "DECISION": rr.DecisionRule(),
            "CARD_FLY": {i: False for i in range(len(TLV.CATALOGUE))},
            "CARD_BUILDING": {i: row[8] == "BUILDING" for i, row in enumerate(TLV.CATALOGUE)},
            "SF": rr.spell_index(), "UNIT_FORMS": rr.unit_form_table(TLV.UNITS),
            "SPELL_CARDS": rr.spell_cards(TLV.CATALOGUE)}.items():
        monkeypatch.setattr(rr, name, value)
    return rr


# ---------------------------------------------------------------------------
# the board
# ---------------------------------------------------------------------------
def test_unknown_labels_are_dropped_by_default_and_kept_as_unknown_units_with_the_option():
    vf = TPS.frame(TRACKS)
    off = TPS.builder().state(vf, opp_deck=["Knight"])
    assert [e[FIELD["uid"]] for e in off["entities"]] == [201, 101]
    assert off["perception"]["dropped"] == {"unknown": 2} and "unknown_units" not in off["perception"]
    assert TPS.builder(unknown_units=False).state(vf, opp_deck=["Knight"]) == off

    on = TPS.builder(unknown_units=True).state(vf, opp_deck=["Knight"])
    assert on["entities"][:2] == off["entities"]
    assert [e[FIELD["uid"]] for e in on["entities"]] == [201, 101, 301]
    row = on["entities"][2]
    assert {name: row[FIELD[name]] for name in PS.REQUIRED_FIELDS} == {
        "uid": 301, "team": E, "kind": 0, "card_id": -1, "tower_slot": -1,
        "x": 9000 * SUB, "y": 9000 * SUB, "hp": PS.UNKNOWN_MAX_HP, "max_hp": PS.UNKNOWN_MAX_HP,
        "flying": False, "deploy_ticks": 10, "status_flags": 0, "level": 11, "unit_type": -1}
    assert all(row[i] == PS.ENTITY_DEFAULTS.get(name, 0) for name, i in FIELD.items()
               if name not in PS.REQUIRED_FIELDS)
    # our own unknown label is dropped as before; the count of what was kept is new
    assert on["perception"]["dropped"] == {"unknown": 1} and on["perception"]["unknown_units"] == 1
    assert on["perception"]["units"] == 3
    assert ({k: v for k, v in on.items() if k not in ("entities", "perception")}
            == {k: v for k, v in off.items() if k not in ("entities", "perception")})
    one = PS.rs_state_from_vision(vf, TPS.HOG, TPS.CLASSES, TPS.SPAWN_TABLE, catalogue=TPS.CATALOGUE,
                                  fields=FIELD, opp_deck=["Knight"], unknown_units=True)
    assert one == on


def test_an_unknown_unit_takes_its_given_card_its_team_rule_and_a_finite_position():
    odd = TPS.track(301, ODD, E, 9000, 9000, first=1490)
    st = TPS.builder(unknown_units=True).state(TPS.frame([odd]), track_cards={301: TPS.CID["Giant"]})
    (row,) = st["entities"]
    assert (row[FIELD["card_id"]], row[FIELD["unit_type"]]) == (TPS.CID["Giant"], -1)
    assert row[FIELD["max_hp"]] == row[FIELD["hp"]] == 3968        # the Giant's own hit points
    st = TPS.builder(unknown_units=True).state(TPS.frame([odd]), track_cards={301: -1})
    assert st["entities"][0][FIELD["card_id"]] == -1
    half = [TPS.track(1, ODD, U, 9000, 9000, hp=0.25), TPS.track(2, ODD, U, 9000, 25000),
            TPS.track(3, ODD, E, float("nan"), 9000)]
    st = TPS.builder(unknown_units=True).state(TPS.frame(half))
    assert [(e[FIELD["uid"]], e[FIELD["team"]], e[FIELD["hp"]]) for e in st["entities"]] == [(1, E, 250)]
    assert st["perception"]["dropped"] == {"position": 1, "unknown": 1}
    st = TPS.builder(unknown_units=True, unknown_team="drop").state(TPS.frame(half))
    assert st["entities"] == [] and st["perception"]["dropped"] == {"position": 1, "team": 2}


def test_an_unknown_unit_is_token_1_in_every_legacy_unit_schema(rr):
    vf = TPS.frame(TRACKS)
    off = TPS.builder().state(vf, opp_deck=["Knight"])
    on = TPS.builder(unknown_units=True).state(vf, opp_deck=["Knight"])
    # the opponent's memory, as perception records it: a Giant, then an UNKNOWN play (card -1)
    mem = rs_memory.PublicMemory()
    EV.record_plays(mem, [OppPlay(card=TPS.CID["Giant"], tick=1200, exec_tick=1200),
                          OppPlay(card=VT.UNKNOWN_CARD, tick=1400, exec_tick=1400, cost=4)])
    view = mem.view(1500, 0)
    giant = TPS.CID["Giant"] + 2
    for kw in ({"live": True}, {"live": False}, {"live": False, "units": "vision"}):
        a = rr.obs_row(off, O, SUB, view, 1500, **kw)
        b = rr.obs_row(on, O, SUB, view, 1500, **kw)
        # nearest our king first: the Knight, our Hog, then the unknown unit at (9000, 9000)
        assert b["unit_ids"][:2].tolist() == a["unit_ids"][:2].tolist() and 1 not in a["unit_ids"].tolist()
        assert (int(a["unit_ids"][2]), int(b["unit_ids"][2])) == (0, 1), kw
        assert np.array_equal(a["unit_feats"][:2], b["unit_feats"][:2]), kw
        if not kw["live"]:          # theirs, own frame (18000 - 9000, 9000), full, ground, deploying
            assert b["unit_feats"][2].tolist() == TLV.f16([0, 0.5, 0.28125, 1, 0, 0, 1]), kw
        # the UNKNOWN play: token 1 sorts first in opp_seen and keeps its place in opp_last
        assert b["opp_seen"].tolist() == [1, giant, 0, 0, 0, 0, 0, 0]
        assert b["opp_last"].tolist() == [giant, 1, 0, 0]
        assert b["opp_last_age"].tolist() == TLV.f16([0.75, 0.25, 1, 1])
    assert "unit_form" in rr.obs_row(on, O, SUB, view, 1500, live=False, units="vision")


def test_the_rebuild_sanitiser_drops_an_unknown_unit():
    on = TPS.builder(unknown_units=True).state(TPS.frame(TRACKS), opp_deck=["Knight"])
    clean = PS.sanitize_for_rebuild(on, TPS.SPAWN_TABLE, fields=FIELD, sub=SUB)
    assert [e[FIELD["uid"]] for e in clean["entities"]] == [201, 101]
    assert clean["sanitized"]["dropped"] == {"no spawn card": 1}


# ---------------------------------------------------------------------------
# the opponent's elixir
# ---------------------------------------------------------------------------
def test_an_unknown_play_is_paid_its_expected_cost_and_the_count_is_not_exact():
    milli = TLV.milli
    oe = PS.OpponentElixir(TLV.CATALOGUE, law=TLV.LAW)
    unk = OppPlay(card=VT.UNKNOWN_CARD, tick=300, exec_tick=300, cost=4)
    assert oe.record([unk]) == 1 and (oe.unknown, oe.unpriced, oe.recorded) == (1, 0, 1)
    # full (280000) from tick 224; the 4 elixir (112000) are gone at 300
    assert oe.view(300) == (milli(168000), False) == (6000, False)
    assert oe.view(301) == (milli(168500), False) == (6017, False)
    # an UNKNOWN play that carries no cost (not one the deriver makes) is unpriced, as before
    old = PS.OpponentElixir(TLV.CATALOGUE, law=TLV.LAW)
    assert old.record([OppPlay(card=VT.UNKNOWN_CARD, tick=300)]) == 0 and old.unpriced == 1
    assert old.view(300) == (10_000, False)
    # a known card with a cost of its own (a Spirit Empress form) pays that, and stays exact
    form = PS.OpponentElixir(TLV.CATALOGUE, law=TLV.LAW)
    form.record([OppPlay(card=TLV.GIANT, tick=300, exec_tick=300, cost=3)])
    assert form.view(300) == (milli(280000 - 84000), True) == (7000, True)
    plain = PS.OpponentElixir(TLV.CATALOGUE, law=TLV.LAW)
    plain.record([TLV.play(TLV.GIANT, 300)])
    assert plain.view(300) == (5000, True) and plain.unknown == 0       # test_live_v2's numbers
    obs = PS.LiveObserver(TLV.CATALOGUE, law=TLV.LAW, start_tick=40)
    obs.events(Events([unk], []))
    assert obs.inputs(300)["opp_elixir"] == [6000, False]


def test_a_v2_row_reads_an_unknown_play_as_token_1_and_its_count_as_not_exact(rr2):
    """test_live_v2's Giant at 290, then at 295 a Knight the tables do not know: with the
    deriver's switch on it is an UNKNOWN play, token 1 in the memory and 4 elixir off the count."""
    tables = EV.EventTables({TLV.GIANT: {"Giant": 1}})
    giant = TLV.track(11, "Giant", E, 9000, 8000, first=290, last=290)
    knight = TLV.track(12, "Knight", E, 3500, 9000, first=295, last=295)
    frames = [(290, [giant]), (295, [TLV.track(11, "Giant", E, 9000, 8500, first=290, last=295), knight])]

    def row(unknown):
        d = EV.EventDeriver(tables, EV.EventConfig(unknown_plays=unknown))
        d.update([], 0)
        obs = PS.LiveObserver(TLV.CATALOGUE, delay=24, law=TLV.LAW, start_tick=40)
        mem = rs_memory.PublicMemory()
        for tick, tracks in frames:
            ev = d.update(tracks, tick)
            EV.record_plays(mem, ev.opp_plays)
            obs.events(ev)
        tracks = frames[-1][1]
        st = TLV.builder(spell_fields=rr2.SF).state(
            TLV.frame(tracks), track_cards=d.track_cards(tracks), opp_deck=[TLV.GIANT], observer=obs)
        r = rr2.obs_row(st, O, SUB, mem.view(300, 0), 300, **PS.row_inputs(st))
        assert rs_schema.V2.check_row(r, buttons=2) == []
        return r, st

    a, sa = row(False)
    b, sb = row(True)
    g = TLV.GIANT + 2
    # off: test_live_v2's count (5178, exact) and memory
    assert sa["observer"]["opp_elixir"] == [5178, True]
    assert np.array_equal(a["scalars"], np.asarray([0.7, 0.05, 0, 0, 0, 0.5178, 1], dtype=np.float32))
    assert a["opp_seen"].tolist() == [g, 0, 0, 0, 0, 0, 0, 0] and a["opp_last"].tolist() == [g, 0, 0, 0]
    # on: the Giant's bar at 295 (142500) less 4 elixir (112000), + 5 ticks: 33000
    assert sb["observer"]["opp_elixir"] == [TLV.milli(33000), False] == [1178, False]
    assert np.array_equal(b["scalars"], np.asarray([0.7, 0.05, 0, 0, 0, 0.1178, 0], dtype=np.float32))
    assert b["opp_seen"].tolist() == [1, g, 0, 0, 0, 0, 0, 0] and b["opp_last"].tolist() == [g, 1, 0, 0]
    assert b["opp_last_age"].tolist() == TLV.f16([10 / 400, 5 / 400, 1, 1])
    # the board is the same either way: the Knight is named by the class table
    for key in a:
        if key not in ("scalars", "opp_seen", "opp_last", "opp_last_age"):
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    assert sa["entities"] == sb["entities"]
    assert [e[FIELD["card_id"]] for e in sb["entities"]] == [TLV.GIANT, TLV.KNIGHT]
