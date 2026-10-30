"""E12: the public visibility filter, tools/rs_rows.py --visibility legacy|public, and the
rebuilt / obs paths of tools/rs_eval.py.

Golden: under legacy (the default) obs_row is a frozen copy of obs_row at 53b7875 (before
E12) for the types, live and vision unit tokens, on states with every status bit set and with
status_flags -1 (legacy never reads them); rows_for, build and vocab_record give what they
gave; RSAgent.rebuilt spawns and aim_spell aims as before. Public, worked by hand on synthetic
entities, one per flag: an enemy with STATUS_INVISIBLE (2) or STATUS_HIDDEN (4) is left out, a
tunnelling one (STATUS_UNDERGROUND, 1) stays where it is with unit_burrow 1, own units are all
kept. Everything runs on conftest's fake royalesim (no engine, no corpus, no torch).
"""
from __future__ import annotations

import ast
import copy
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    FIELD,
    SUBTILE_PER_MILLITILE,
    FakeBattle,
    entity_row,
    il_payload,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = SUBTILE_PER_MILLITILE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UNDER, INVIS, HIDDEN, EVOLVED = 1, 2, 4, 8
MEM = {"seen": {0: {4, 7}, 1: {10, 13}}, "last": {0: [(4, 300), (7, 700)], 1: [(10, 650)]}}
EMPTY_MEM = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}


@pytest.fixture
def tools(monkeypatch):
    """rs_rows and rs_eval as the tools import them, with their module settings at the
    defaults (restored after the test) and no engine config from the PC."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    ev = importlib.import_module("rs_eval")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "OUT": rr.OUT, "CARD_FLY": {},
                        "CARD_BUILDING": {}}.items():
        monkeypatch.setattr(rr, name, value)
    monkeypatch.setattr(ev.RSAgent, "_CAT", None)
    return types.SimpleNamespace(rows=rr, eval=ev, engine=eng)


@pytest.fixture
def fields(tools, monkeypatch):
    """rs_rows.F and the catalogue flags, as _setup fills them on the fake engine."""
    rr = tools.rows
    monkeypatch.setattr(rr, "F", dict(FIELD))
    for i, row in enumerate(DEFAULT_CATALOGUE):
        rr.CARD_FLY[i] = bool(row[5])
        rr.CARD_BUILDING[i] = row[8] == "BUILDING"
    return rr


def cfg(tools):
    return tools.engine.EngineConfig.defaults()


def same_row(a, b):
    assert list(a) == list(b)
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        assert x.dtype == y.dtype and np.array_equal(x, y), key


class StubLookup:
    """vision_classes.UnitLookup's surface obs_row reads (token, flags, form_feature)."""

    def token(self, unit):
        return 1 if unit < 0 else unit + 3

    def flags(self, unit):
        return (1.0 if unit % 2 else 0.0, 1.0 if unit == 2 else 0.0)

    def form_feature(self, unit):
        return 0.5 if unit == 3 else 0.0


# ---------------------------------------------------------------------------
# golden: legacy obs_row is the obs_row before E12
# ---------------------------------------------------------------------------

def frozen_obs_row(rr, state, team, sub, mem, tick, live, units):
    """tools/rs_rows.obs_row at 53b7875 (before E12), frozen; module names through `rr`,
    MAX_UNITS 48 and UF 7 the values it had."""
    vision = not live and units == "vision"
    forms = np.zeros(48, dtype=np.float16) if vision else None
    F = rr.F
    ents = [e for e in state["entities"] if e[F["tower_slot"]] < 0 and e[F["hp"]] > 0]
    king = next((e for e in state["entities"]
                 if e[F["team"]] == team and e[F["tower_slot"]] == 0), None)
    kx, ky = ((king[F["x"]], king[F["y"]]) if king is not None
              else (9000 * 18, (3000 if team == 0 else 29000) * 18))
    ents.sort(key=lambda e: (e[F["x"]] - kx) ** 2 + (e[F["y"]] - ky) ** 2)
    ids = np.zeros(48, dtype=np.int16)
    feats = np.zeros((48, 7), dtype=np.float16)
    for k, e in enumerate(ents[:48]):
        ox, oy = rr.own(e[F["x"]], e[F["y"]], team, sub)
        if live:
            card = e[F["card_id"]]
            ids[k] = card + 2
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]),
                        1.0 if rr.CARD_FLY.get(card) else 0.0,
                        1.0 if rr.CARD_BUILDING.get(card) else 0.0,
                        float(np.log1p(max(0, e[F["max_hp"]]))) / 10.0)
        elif vision:
            unit = e[F["unit_type"]]
            ids[k] = rr.VC.token(unit)
            fly, building = rr.VC.flags(unit)
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]), fly, building,
                        1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
            forms[k] = rr.VC.form_feature(unit)
        else:
            ids[k] = e[F["unit_type"]] + 2
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]), 1.0 if e[F["flying"]] else 0.0,
                        1.0 if e[F["kind"]] == 1 else 0.0,
                        1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
    p = state["players"][team]
    hand = [c + 2 for c in p["hand"][:4]] + [0] * (4 - len(p["hand"][:4]))
    opp = 1 - team
    seen = sorted(mem["seen"][opp])[:8]
    last = mem["last"][opp][-4:]
    t_ms = tick * 50
    row = {
        "tick": tick, "side": 1 if team == 0 else -1, "hand_exact": True,
        "unit_ids": ids, "unit_feats": feats, "towers": rr.towers_vec(state, team),
        "scalars": np.asarray([p["elixir_milli"] / 10_000, min(1.0, t_ms / 300_000),
                               1.0 if t_ms >= 120_000 else 0.0,
                               1.0 if t_ms >= 180_000 else 0.0], dtype=np.float32),
        "hand": np.asarray(hand, dtype=np.int16),
        "next": (p["next_card"] + 2) if p.get("next_card", -1) >= 0 else 0,
        "opp_seen": np.asarray(seen + [0] * (8 - len(seen)), dtype=np.int16),
        "opp_last": np.asarray([c for c, _ in last] + [0] * (4 - len(last)), dtype=np.int16),
        "opp_last_age": np.asarray([min(1.0, (tick - t) / 400) for _, t in last]
                                   + [1.0] * (4 - len(last)), dtype=np.float16),
    }
    if vision:
        row["unit_form"] = forms
    return row


STATUSES = (0, UNDER, INVIS, UNDER | INVIS, HIDDEN, INVIS | HIDDEN, EVOLVED, 16, 512, -1)


def busy_state(tick, king_alive=True, statuses=STATUSES):
    """60 units of both teams, more than MAX_UNITS, every status in `statuses` (the engine's
    -1, "not reported", too), buildings, fliers, deploying and dead ones."""
    ents = [entity_row(uid=100 + k, team=k % 2, kind=1 if k % 3 == 0 else 0, card_id=k % 16,
                       x=(500 + 331 * k) % 18_000 * SUB, y=(700 + 613 * k) % 32_000 * SUB,
                       hp=0 if k % 11 == 5 else 20 + 7 * k, max_hp=40 + 9 * k,
                       flying=k % 4 == 1, deploy_ticks=k % 3, unit_type=k % 14,
                       status_flags=statuses[k % len(statuses)])
            for k in range(60)]
    players = [player_dict(0, elixir_milli=6100, hand=[2, 5, -1, 6], next_card=1,
                           tower_hp=(4824 if king_alive else 0, 3052, 1200)),
               player_dict(1, hand=[8, 9], next_card=-1, tower_hp=(4000, 0, 3052))]
    return state_dict(tick, players=players, entities=ents)


def test_legacy_obs_row_is_the_obs_row_before_e12(fields, monkeypatch):
    rr = fields
    monkeypatch.setattr(rr, "VC", StubLookup())
    n = 0
    for tick, king in ((1234, True), (3700, False)):
        state = busy_state(tick, king)
        for team in (0, 1):
            for live, units in ((False, "types"), (True, "types"), (False, "vision")):
                want = frozen_obs_row(rr, state, team, SUB, MEM, tick, live, units)
                same_row(rr.obs_row(state, team, SUB, MEM, tick, live=live, units=units), want)
                same_row(rr.obs_row(state, team, SUB, MEM, tick, live=live, units=units,
                                    visibility="legacy"), want)
                n += 1
    assert n == 12
    assert "unit_burrow" not in rr.obs_row(busy_state(1234), 0, SUB, MEM, 1234)


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def test_visibility_values(tools):
    rr = tools.rows
    assert rr.VISIBILITY == rr.VIS_LEGACY == "legacy" and rr.VIS_PUBLIC == "public"
    assert rr.VISIBILITY_MODES == ("legacy", "public")
    assert rr.visibility_mode("public") == "public" and rr.visibility_mode("legacy") == "legacy"
    for bad in ("Public", "", None, 1, "screen"):
        with pytest.raises(ValueError, match="--visibility"):
            rr.visibility_mode(bad)
    # royalegym protocol.py:579-583, royalesim.STATUS_BITS names 0-2
    assert (rr.STATUS_UNDERGROUND, rr.STATUS_INVISIBLE, rr.STATUS_HIDDEN) == (1, 2, 4)
    assert rr.UNSEEN_STATUS == 6 and rr.STATUS_NAMES == ("underground", "invisible", "hidden")
    assert rr.VISIBILITY_FIELDS == ("unit_burrow",)
    assert issubclass(rr.VisibilityError, ValueError)
    assert rr.describe_visibility("legacy").startswith("legacy")
    assert rr.describe_visibility("public").startswith("public")


def _main(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _argument(main, flag):
    hits = [n for n in ast.walk(main) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "add_argument"
            and n.args and getattr(n.args[0], "value", None) == flag]
    assert len(hits) == 1, flag
    return {k.arg: k.value for k in hits[0].keywords}


def test_both_command_lines_take_visibility_with_the_old_default():
    rows = _argument(_main(TOOLS / "rs_rows.py"), "--visibility")
    assert isinstance(rows["default"], ast.Name) and rows["default"].id == "VIS_LEGACY"
    ev = _argument(_main(TOOLS / "rs_eval.py"), "--visibility")
    assert isinstance(ev["default"], ast.Constant) and ev["default"].value is None
    called = {getattr(n.func, "attr", getattr(n.func, "id", ""))
              for n in ast.walk(_main(TOOLS / "rs_rows.py")) if isinstance(n, ast.Call)}
    assert {"visibility_vocab", "describe_visibility"} <= called


def test_every_obs_row_call_of_rs_eval_passes_the_seat_visibility():
    tree = ast.parse((TOOLS / "rs_eval.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "obs_row"]
    assert len(calls) == 4                      # RSAgent.act, RSSearchAgent.act (3)
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "visibility" in kw, call.lineno
        assert ast.unparse(kw["visibility"]) == "self.visibility", call.lineno


def test_status_bits_are_checked_when_public(tools, monkeypatch):
    rr = tools.rows
    rr.check_status_bits(types.SimpleNamespace(
        STATUS_BITS=["underground", "invisible", "hidden", "evolved", "hero"]))
    rr.check_status_bits(types.SimpleNamespace())          # an engine that does not list them
    with pytest.raises(rr.rs_engine.EngineError, match="STATUS_BITS"):
        rr.check_status_bits(types.SimpleNamespace(STATUS_BITS=["invisible", "underground",
                                                                 "hidden"]))
    mod = make_fake_royalesim()
    mod.STATUS_BITS = ["evolved", "hero", "clone"]
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    rr._setup(cfg(tools))                                  # legacy: the bits are not read
    monkeypatch.setattr(rr, "VISIBILITY", "public")
    with pytest.raises(rr.rs_engine.EngineError, match="STATUS_BITS"):
        rr._setup(cfg(tools))
    mod.STATUS_BITS = list(rr.STATUS_NAMES) + ["evolved"]
    assert rr._setup(cfg(tools))[0] is mod


# ---------------------------------------------------------------------------
# obs_row --visibility public, by hand
# ---------------------------------------------------------------------------

#: (team, status, kind, unit_type) of entity k = 0..8, uid 10 + k, card k, at x 9000 and
#: y 4000 + 1000 k millitiles: one per flag, for both teams.
FLAGGED = (
    (1, 0, 0, 8),                   # 0 enemy of blue, plain
    (1, INVIS, 0, 9),               # 1 a Royal Ghost
    (1, HIDDEN, 1, 2),              # 2 a Tesla in the ground
    (1, UNDER, 0, 10),              # 3 a Miner on its way
    (0, INVIS, 0, 3),               # 4 blue's own ghost
    (0, UNDER, 0, 4),               # 5 blue's own Miner
    (0, HIDDEN, 1, 2),              # 6 blue's own hidden Tesla
    (1, EVOLVED, 0, 11),            # 7 another bit: shown
    (1, UNDER | INVIS, 0, 12),      # 8 both: left out (invisible wins)
)


def flagged_entities():
    return [entity_row(uid=10 + k, team=team, kind=kind, card_id=k, x=9000 * SUB,
                       y=(4000 + 1000 * k) * SUB, hp=100 + 10 * k, max_hp=300,
                       unit_type=unit, deploy_ticks=k % 2, status_flags=status)
            for k, (team, status, kind, unit) in enumerate(FLAGGED)]


def flagged_state(tick=900):
    return state_dict(tick, entities=flagged_entities())


#: team -> (indexes into the legacy row of the units public keeps, in order; their burrow).
#: Blue sorts from its king at y 3000 (k ascending) and drops enemies 1, 2 and 8; red sorts
#: from y 29000 (k descending: legacy index i is k = 8 - i) and drops blue's 4 and 6.
KEPT = {0: ([0, 3, 4, 5, 6, 7], [0, 1, 0, 1, 0, 0]),
        1: ([0, 1, 3, 5, 6, 7, 8], [1, 0, 1, 1, 0, 0, 0])}
#: The unit-type tokens (unit_type + 2) of those units.
KEPT_IDS = {0: [10, 12, 5, 6, 4, 13], 1: [14, 13, 6, 12, 4, 11, 10]}


def test_public_rows_leave_out_unseen_enemies_and_flag_burrowers(fields, monkeypatch):
    rr = fields
    monkeypatch.setattr(rr, "VC", StubLookup())
    state = flagged_state()
    for team in (0, 1):
        keep, burrow = KEPT[team]
        n = len(keep)
        for live, units in ((False, "types"), (True, "types"), (False, "vision")):
            old = rr.obs_row(state, team, SUB, MEM, 900, live=live, units=units)
            new = rr.obs_row(state, team, SUB, MEM, 900, live=live, units=units,
                             visibility="public")
            assert list(new) == list(old) + ["unit_burrow"]
            assert np.count_nonzero(old["unit_ids"]) == 9
            assert np.array_equal(new["unit_ids"][:n], old["unit_ids"][keep])
            assert not new["unit_ids"][n:].any()
            assert np.array_equal(new["unit_feats"][:n], old["unit_feats"][keep])
            assert not new["unit_feats"][n:].any()
            if units == "vision":
                assert np.array_equal(new["unit_form"][:n], old["unit_form"][keep])
                assert not new["unit_form"][n:].any()
            b = new["unit_burrow"]
            assert b.dtype == np.float16 and b.shape == (48,)
            assert b.tolist() == burrow + [0.0] * (48 - n)
            same_row({k: v for k, v in old.items() if not k.startswith("unit_")},
                     {k: v for k, v in new.items() if not k.startswith("unit_")})
        types_row = rr.obs_row(state, team, SUB, MEM, 900, visibility="public")
        assert types_row["unit_ids"][:n].tolist() == KEPT_IDS[team]
        live_row = rr.obs_row(state, team, SUB, MEM, 900, live=True, visibility="public")
        cards = [k if team == 0 else 8 - k for k in keep]       # card k, legacy index -> k
        assert live_row["unit_ids"][:n].tolist() == [c + 2 for c in cards]
    # blue's first unit shown, the plain enemy at y 4000: own frame (9000, 28000), hp 100 / 300
    first = rr.obs_row(state, 0, SUB, MEM, 900, visibility="public")["unit_feats"][0]
    want = np.asarray((0.0, 0.5, 28_000 / 32_000, 100 / 300, 0.0, 0.0, 0.0), dtype=np.float16)
    assert np.array_equal(first, want)


def test_the_module_setting_is_the_default(fields, monkeypatch):
    rr = fields
    state = flagged_state()
    public = rr.obs_row(state, 0, SUB, MEM, 900, visibility="public")
    monkeypatch.setattr(rr, "VISIBILITY", "public")
    same_row(rr.obs_row(state, 0, SUB, MEM, 900), public)
    legacy = rr.obs_row(state, 0, SUB, MEM, 900, visibility="legacy")
    assert "unit_burrow" not in legacy and np.count_nonzero(legacy["unit_ids"]) == 9
    with pytest.raises(ValueError, match="--visibility"):
        rr.obs_row(state, 0, SUB, MEM, 900, visibility="screen")


def test_public_rows_fill_the_places_of_the_units_left_out(fields):
    """Six invisible enemy ghosts nearest blue's king, then 50 own Skeletons: legacy shows the
    ghosts and the 42 nearest Skeletons, public the 48 nearest Skeletons."""
    rr = fields
    ghosts = [entity_row(uid=200 + i, team=1, card_id=9, x=9000 * SUB, y=(3200 + 50 * i) * SUB,
                         hp=450, unit_type=9, status_flags=INVIS) for i in range(6)]
    skeletons = [entity_row(uid=300 + j, team=0, card_id=6, x=9000 * SUB,
                            y=(4000 + 100 * j) * SUB, hp=32, unit_type=7) for j in range(50)]
    state = state_dict(800, entities=ghosts + skeletons)
    legacy = rr.obs_row(state, 0, SUB, EMPTY_MEM, 800)
    public = rr.obs_row(state, 0, SUB, EMPTY_MEM, 800, visibility="public")
    assert legacy["unit_ids"].tolist() == [11] * 6 + [9] * 42
    assert public["unit_ids"].tolist() == [9] * 48
    assert not public["unit_burrow"].any()
    # the 48th Skeleton (j = 47) is the last one shown: own y 32000 - 8700
    assert public["unit_feats"][47][2] == np.float16(23_300 / 32_000)
    # red sees its own ghosts: nothing is left out, public is legacy plus unit_burrow
    red = rr.obs_row(state, 1, SUB, EMPTY_MEM, 800, visibility="public")
    same_row({k: v for k, v in red.items() if k != "unit_burrow"},
             rr.obs_row(state, 1, SUB, EMPTY_MEM, 800))


def test_public_row_is_the_legacy_row_of_the_screen(fields):
    """On the busy states (every bit but -1): public = legacy on the state without the enemy
    units that have an UNSEEN bit, plus unit_burrow = the UNDERGROUND bit of each unit shown."""
    rr = fields
    statuses = tuple(s for s in STATUSES if s >= 0)
    for tick in (1234, 3700):
        state = busy_state(tick, statuses=statuses)
        for team in (0, 1):
            screen = copy.deepcopy(state)
            screen["entities"] = [e for e in state["entities"]
                                  if e[FIELD["team"]] == team or e[FIELD["tower_slot"]] >= 0
                                  or not e[FIELD["status_flags"]] & 6]
            assert len(screen["entities"]) < len(state["entities"])
            for live in (False, True):
                public = rr.obs_row(state, team, SUB, MEM, tick, live=live, visibility="public")
                burrow = public.pop("unit_burrow")
                same_row(public, rr.obs_row(screen, team, SUB, MEM, tick, live=live))
                king = next(e for e in screen["entities"]
                            if e[FIELD["team"]] == team and e[FIELD["tower_slot"]] == 0)
                units = [e for e in screen["entities"]
                         if e[FIELD["tower_slot"]] < 0 and e[FIELD["hp"]] > 0]
                units.sort(key=lambda e: (e[FIELD["x"]] - king[FIELD["x"]]) ** 2
                           + (e[FIELD["y"]] - king[FIELD["y"]]) ** 2)
                want = [float(e[FIELD["status_flags"]] & 1) for e in units[:48]]
                assert burrow.tolist() == want + [0.0] * (48 - len(want))
                assert 0 < sum(want) < len(want)


def test_public_refuses_units_without_status_flags(fields):
    rr = fields
    plain, ghost = flagged_entities()[:2]           # red's units uid 10 and 11
    for ent in (plain, ghost):
        ent[FIELD["status_flags"]] = -1             # the engine's "not reported"
    # blue cannot judge red's unit; red cannot tell whether its own unit burrows
    for team in (0, 1):
        state = state_dict(900, entities=[plain])
        with pytest.raises(rr.VisibilityError, match="uid 10 has no status_flags"):
            rr.obs_row(state, team, SUB, MEM, 900, visibility="public")
        rr.obs_row(state, team, SUB, MEM, 900)                     # legacy does not read them
    # dead units and crown towers are not read
    dead = entity_row(uid=50, team=1, hp=0, max_hp=100, unit_type=8, status_flags=-1)
    state = state_dict(900, entities=[dead])
    for e in state["entities"]:
        e[FIELD["status_flags"]] = -1
    row = rr.obs_row(state, 0, SUB, MEM, 900, visibility="public")
    assert not row["unit_ids"].any() and not row["unit_burrow"].any()
    with pytest.raises(rr.VisibilityError):
        rr.seen_by(ghost, 0)
    assert rr.seen_by(ghost, 1) is True                            # own: status not read
    with pytest.raises(rr.VisibilityError):
        rr.burrowing(ghost)


# ---------------------------------------------------------------------------
# rows_for and build
# ---------------------------------------------------------------------------

def rows_state(tick):
    """A ghost of red's (A, invisible), an invisible unit of blue's (B) and red's Miner (C)."""
    ents = [entity_row(uid=60, team=1, card_id=9, x=9000 * SUB, y=8000 * SUB, hp=300,
                       unit_type=9, status_flags=INVIS),
            entity_row(uid=61, team=0, card_id=5, x=9000 * SUB, y=10_000 * SUB, hp=200,
                       unit_type=6, status_flags=INVIS),
            entity_row(uid=62, team=1, card_id=12, x=9000 * SUB, y=14_000 * SUB, hp=500,
                       unit_type=11, status_flags=UNDER)]
    return state_dict(tick, entities=ents)


@pytest.fixture
def ghost_royalesim(monkeypatch):
    mod = make_fake_royalesim(states=lambda battle: rows_state(battle.tick))
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    return mod


#: Blue sorts A, B, C (5000, 7000, 11000 mt from its king) and keeps B, C; red sorts C, B, A
#: (15000, 19000, 21000) and keeps its own C and A. side -> (legacy indexes kept, burrow).
ROWS_KEPT = {1: ([1, 2], [0.0, 1.0]), -1: ([0, 2], [1.0, 0.0])}
#: Conftest's plays: 31 ticks (20 .. 620) x 2 seats, each a play row or a wait row.
N_ROWS = 62


def test_rows_for_public_by_hand(ghost_royalesim, tools, monkeypatch):
    rr = tools.rows
    ctx = rr._setup(cfg(tools))
    legacy, why = rr.rows_for(il_payload(tag="vis-1"), ctx, cfg(tools))
    assert why == "ok" and len(legacy) == N_ROWS
    monkeypatch.setattr(rr, "VISIBILITY", "public")
    public, why = rr.rows_for(il_payload(tag="vis-1"), ctx, cfg(tools))
    assert why == "ok" and len(public) == N_ROWS
    for old, new in zip(legacy, public):
        assert set(new) == set(old) | {"unit_burrow"} and "unit_burrow" not in old
        keep, burrow = ROWS_KEPT[old["side"]]
        assert np.count_nonzero(old["unit_ids"]) == 3
        assert np.array_equal(new["unit_ids"][:2], old["unit_ids"][keep])
        assert np.array_equal(new["unit_feats"][:2], old["unit_feats"][keep])
        assert not new["unit_ids"][2:].any() and not new["unit_feats"][2:].any()
        assert new["unit_burrow"].tolist() == burrow + [0.0] * 46
        for key in old:
            if key not in ("unit_ids", "unit_feats"):
                assert np.array_equal(np.asarray(old[key]), np.asarray(new[key])), key
    blue = next(r for r in public if r["side"] == 1)
    red = next(r for r in public if r["side"] == -1)
    assert blue["unit_ids"][:2].tolist() == [8, 13] and red["unit_ids"][:2].tolist() == [13, 11]


def test_build_writes_unit_burrow_only_when_public(ghost_royalesim, tools, tmp_path):
    rr = tools.rows
    text = json.dumps(il_payload(tag="vis-2"))
    eng = cfg(tools).to_json()
    act = ("act_card", "act_x", "act_y", "match", "split")

    def shard(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    assert rr.build(1, [text], False, str(tmp_path), eng) == (1, N_ROWS, {})
    old = shard(tmp_path / "shard-0001.npz")
    assert sorted(old) == sorted(rr.FIELDS + act) and rr.VISIBILITY == "legacy"
    assert rr.build(2, [text], False, str(tmp_path), eng, visibility="public")[1] == N_ROWS
    assert rr.VISIBILITY == "public"
    new = shard(tmp_path / "shard-0002.npz")
    assert sorted(new) == sorted(rr.FIELDS + act + ("unit_burrow",))
    assert new["unit_burrow"].dtype == np.float16 and new["unit_burrow"].shape == (N_ROWS, 48)
    assert new["unit_burrow"].sum() == N_ROWS                      # one Miner in every row
    for k in old:
        if k not in ("unit_ids", "unit_feats"):
            assert np.array_equal(old[k], new[k]), k
    assert (np.count_nonzero(new["unit_ids"], axis=1) == 2).all()
    assert rr.build(3, [text], False, str(tmp_path), eng)[1] == N_ROWS    # None: keep public
    again = shard(tmp_path / "shard-0003.npz")
    assert all(np.array_equal(again[k], new[k]) for k in new)
    assert rr.build(4, [text], False, str(tmp_path), eng, visibility="legacy")[1] == N_ROWS
    back = shard(tmp_path / "shard-0004.npz")
    assert sorted(back) == sorted(old) and all(np.array_equal(back[k], old[k]) for k in old)
    with pytest.raises(ValueError, match="--visibility"):
        rr.build(5, [text], False, str(tmp_path), eng, visibility="screen")
    assert rr.VISIBILITY == "legacy" and not (tmp_path / "shard-0005.npz").exists()


# ---------------------------------------------------------------------------
# vocab.json
# ---------------------------------------------------------------------------

class StubClasses(list):
    """The surface of a vision_classes.ClassTable vocab_record reads."""
    digest = "c1a55e5"


def _vocab_inputs(tools):
    cat = [list(r) for r in DEFAULT_CATALOGUE]
    fp = {"version": "0.0.0+fake", "card_table": None, "id": "f00d"}
    return cat, list(DEFAULT_UNIT_TYPES), cfg(tools), fp


RULE = {"drop_enemy": ["invisible", "hidden"], "unit_burrow": "underground",
        "bits": {"underground": 1, "invisible": 2, "hidden": 4}}


def test_vocab_records_public_visibility_only(tools):
    rr = tools.rows
    cat, units, eng, fp = _vocab_inputs(tools)
    base = rr.vocab_record(cat, units, eng, fp, {}, False)
    assert rr.visibility_vocab("legacy") == {}
    assert rr.visibility_vocab("public") == {"visibility": "public"}
    for same in (rr.vocab_record(cat, units, eng, fp, {}, False, visibility="legacy"),
                 rr.vocab_record(cat, units, eng, fp, {}, False, **rr.visibility_vocab("legacy"))):
        assert json.dumps(same) == json.dumps(base)              # the vocab.json of before
    assert "visibility" not in base and "extra_columns" not in base
    pub = rr.vocab_record(cat, units, eng, fp, {}, False, **rr.visibility_vocab("public"),
                          **rr.memory_vocab(1))
    assert list(pub) == list(base) + ["opp_memory", "visibility", "visibility_rule",
                                      "extra_columns"]
    assert pub["visibility"] == "public" and pub["visibility_rule"] == RULE
    assert pub["extra_columns"] == ["unit_burrow"]
    assert rr.visibility_rule() == RULE and rr.visibility_rule() is not rr.visibility_rule()
    classes = StubClasses([types.SimpleNamespace(name="Knight", id=0),
                           types.SimpleNamespace(name="Archer", id=1)])
    vis = rr.vocab_record(cat, units, eng, fp, {}, False, classes=classes)
    assert list(vis) == list(base) + ["units_mode", "vision_classes_digest", "extra_columns"]
    assert vis["extra_columns"] == ["unit_form"]                  # as before E12
    both = rr.vocab_record(cat, units, eng, fp, {}, False, classes=classes, visibility="public")
    assert list(both) == list(base) + ["units_mode", "vision_classes_digest", "visibility",
                                       "visibility_rule", "extra_columns"]
    assert both["extra_columns"] == ["unit_form", "unit_burrow"]
    with pytest.raises(ValueError, match="--visibility"):
        rr.vocab_record(cat, units, eng, fp, {}, False, visibility="screen")
    # a checkpoint's meta is all of vocab.json plus the model sizes (train_tokens.checkpoint_meta)
    assert rr.visibility_of(dict(pub, d=192, layers=4, heads=6)) == "public"
    assert rr.visibility_of(base) == rr.visibility_of({}) == rr.visibility_of(None) == "legacy"
    with pytest.raises(ValueError, match="--visibility"):
        rr.visibility_of({"visibility": "screen"})


# ---------------------------------------------------------------------------
# rs_eval: the seat's visibility, rebuilt, aim_spell, act, play
# ---------------------------------------------------------------------------

def test_seat_visibility(tools, monkeypatch):
    ev = tools.eval
    assert ev.RSAgent.visibility is None
    assert ev.seat_visibility(types.SimpleNamespace()) == "legacy"
    assert ev.seat_visibility(types.SimpleNamespace(visibility=None)) == "legacy"
    assert ev.seat_visibility(types.SimpleNamespace(visibility="public")) == "public"
    monkeypatch.setattr(tools.rows, "VISIBILITY", "public")
    assert ev.seat_visibility(types.SimpleNamespace()) == "public"
    assert ev.seat_visibility(types.SimpleNamespace(visibility="legacy")) == "legacy"
    with pytest.raises(ValueError, match="--visibility"):
        ev.seat_visibility(types.SimpleNamespace(visibility="screen"))


def rebuild_state():
    players = [player_dict(0, deck=list(range(8))), player_dict(1, deck=list(range(8, 16)))]
    return state_dict(1500, players=players, entities=flagged_entities())


def _spawns(ev, agent, team):
    fake = make_fake_royalesim()
    battle = ev.RSAgent.rebuilt(agent, rebuild_state(), team, fake)
    assert battle is fake.battles[-1], getattr(agent, "rebuild_error", None)
    return battle.reset_kwargs["spawns"]


def test_rebuilt_spawns_only_what_the_screen_shows(fields, tools, monkeypatch):
    ev = tools.eval
    # every card k makes its own unit, so each unit is spawned as its card
    prim = {k: unit for k, (_, _, _, unit) in enumerate(FLAGGED)}
    monkeypatch.setattr(ev, "_spawn_tables", lambda: (prim, {}))
    every = [(team, k, 9000 * SUB, (4000 + 1000 * k) * SUB, 100 + 10 * k)
             for k, (team, _, _, _) in enumerate(FLAGGED)]
    legacy = types.SimpleNamespace(engine=tools.engine.EngineConfig.legacy())
    public = types.SimpleNamespace(engine=tools.engine.EngineConfig.legacy(), visibility="public")
    for team in (0, 1):
        assert _spawns(ev, legacy, team) == every                  # as before E12
    assert _spawns(ev, public, 0) == [every[k] for k in (0, 3, 4, 5, 6, 7)]
    assert _spawns(ev, public, 1) == [every[k] for k in (0, 1, 2, 3, 5, 7, 8)]
    monkeypatch.setattr(tools.rows, "VISIBILITY", "public")       # a seat without its own
    assert _spawns(ev, legacy, 0) == [every[k] for k in (0, 3, 4, 5, 6, 7)]


#: Fireball (card 1, radius 2.5 tiles) for blue over tiles A = (4, 10) and B = (13, 10): their
#: centres are world (4.5, 21.5) and (13.5, 21.5) tiles.
TILE_A, TILE_B = 10 * 18 + 4, 10 * 18 + 13


def aim_state():
    ents = [entity_row(uid=70, team=1, card_id=8, x=4500 * SUB, y=21_500 * SUB, hp=100,
                       unit_type=8),                                    # seen, under A
            entity_row(uid=71, team=1, card_id=9, x=13_500 * SUB, y=21_500 * SUB, hp=1000,
                       unit_type=9, status_flags=INVIS),                # a ghost under B
            entity_row(uid=72, team=0, card_id=5, x=4500 * SUB, y=21_000 * SUB, hp=2000,
                       unit_type=6, status_flags=INVIS)]                # own: never counted
    return state_dict(500, entities=ents, towers=False)


def test_aim_spell_counts_only_enemies_on_the_screen(fields, tools, monkeypatch):
    ev = tools.eval
    monkeypatch.setattr(ev.RSAgent, "_CAT", [list(r) for r in DEFAULT_CATALOGUE])
    order = np.asarray([TILE_A, TILE_B])
    legacy = ev.RSAgent.aim_spell(types.SimpleNamespace(), order, aim_state(), 0, 1, SUB)
    assert legacy.tolist() == [TILE_B, TILE_A]                  # the ghost's 1000 hp, as before
    public = ev.RSAgent.aim_spell(types.SimpleNamespace(visibility="public"), order,
                                  aim_state(), 0, 1, SUB)
    assert public.tolist() == [TILE_A, TILE_B]
    state = aim_state()
    state["entities"][0][FIELD["status_flags"]] = -1
    with pytest.raises(tools.rows.VisibilityError):
        ev.RSAgent.aim_spell(types.SimpleNamespace(visibility="public"), order, state, 0, 1, SUB)


def test_act_builds_the_row_of_the_seat_visibility(fields, tools, monkeypatch):
    ev, rr = tools.eval, tools.rows
    rows = []

    def decide(policy, row, *args, **kwargs):
        rows.append(row)
        return types.SimpleNamespace(command=None)

    monkeypatch.setattr(ev.rs_decide, "decide", decide)
    agent = object.__new__(ev.RSAgent)              # what __init__ sets, by hand; no torch
    agent.model, agent.torch, agent.live = object(), object(), False
    agent.rng, agent.wait_bias = None, 1.0
    for vis, module in ((None, "legacy"), ("public", "legacy"), (None, "public"),
                        ("legacy", "public")):
        monkeypatch.setattr(rr, "VISIBILITY", module)
        agent.visibility = vis
        assert agent.act(FakeBattle(), rows_state(500), 0, EMPTY_MEM, 500, SUB, None) is None
    legacy, public, follows, explicit = rows
    assert "unit_burrow" not in legacy and np.count_nonzero(legacy["unit_ids"]) == 3
    assert public["unit_burrow"].tolist()[:2] == [0.0, 1.0]
    assert np.count_nonzero(public["unit_ids"]) == 2
    same_row(follows, public)
    same_row(explicit, legacy)


class StubAgent:
    """AGENT_CLASS for play(): no checkpoint; `meta_visibility` stands for its meta's."""
    meta_visibility = "legacy"
    made: list = []

    def __init__(self, ckpt, seed=0):
        self.visibility = StubAgent.meta_visibility
        self.acted = set()
        StubAgent.made.append(self)

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        self.acted.add(self.visibility)
        return None


def _play_agent(tools, visibility):
    """play() in the agent seat: (payload, ckpt, mode, seed, opp_ckpt, delay, lead, hold,
    shadow, spell_aim, engine, mem_lag, decide, visibility)."""
    return tools.eval.play((json.dumps(il_payload(tag="vis-3")), "stub.pt", "agent", 0, "", 0,
                            0, False, False, False, cfg(tools).to_json(), None, None,
                            visibility))


def test_play_sets_the_visibility_over_the_checkpoints(fake_royalesim, tools, monkeypatch):
    ev = tools.eval
    monkeypatch.setattr(ev, "AGENT_CLASS", StubAgent)
    monkeypatch.setattr(StubAgent, "made", [])
    for meta, given, want in (("legacy", None, "legacy"), ("public", None, "public"),
                              ("public", "legacy", "legacy"), ("legacy", "public", "public")):
        monkeypatch.setattr(StubAgent, "meta_visibility", meta)
        out = _play_agent(tools, given)
        assert "skipped" not in out and out["hog_team"] == 0
        assert StubAgent.made[-1].acted == {want}, (meta, given)
    with pytest.raises(ValueError, match="--visibility"):
        _play_agent(tools, "screen")
    # a public seat checks the engine's status bits; a legacy one does not read them
    fake_royalesim.STATUS_BITS = ["evolved", "hero", "clone"]
    monkeypatch.setattr(StubAgent, "meta_visibility", "legacy")
    _play_agent(tools, None)
    with pytest.raises(tools.engine.EngineError, match="STATUS_BITS"):
        _play_agent(tools, "public")
