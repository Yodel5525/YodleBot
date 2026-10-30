"""E10: ability presses replayed and labelled (tools/rs_presses.py, rs_rows.py and
royalesim_redrive.py --presses off|on).

Pure tests of rs_presses, worked by hand: the button order (heroes, form 2, in dealt-deck order,
then champions), champions by catalogue column or by name, the engine's own order, which card
an event names, button_of's ok / unknown / ambiguous, the command slot HAND_SIZE + button, the
ability-state features and the counts.

rows_for, build and royalesim_redrive.one on conftest's fake royalesim (no engine, no corpus).
The fixture match is conftest's payload (plays at 100, 160, 240, 300, 420; the team at low y
plays blue) with a hero Musketeer and a hero Ice Golem in the Hog 2.6 deck and a Golden Knight
(a champion, catalogue id 16 in a catalogue that adds him) in the opponent's, which presses at
240 (blue, Musketeer), 330 (red, Golden Knight) and 360 (blue, Ice Golem). The team's deck is
dealt as Cannon, Fireball, HogRider, Musketeer, IceGolemite, Skeletons, IceSpirits, Log (the
first order royalesim_redrive.deal_order finds for Musketeer, HogRider, Skeletons), so the
Musketeer is button 0 (slot 4) and the Ice Golem button 1 (slot 5), although the Ice Golem
comes first in the recorded deck.

Golden: with --presses off (the default) the rows, info, every call to the engine and the
re-drive's result are those of the same match without its activate_ability events.

The `engine`-marked tests need the real royalesim: a press before the hero stands is refused
NO_HERO, one after it is taken, and the press row shows the button available, then spent.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    DEPLOY_REASONS,
    SUBTILE_PER_MILLITILE,
    catalogue_row,
    il_payload,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = SUBTILE_PER_MILLITILE

#: conftest's catalogue and a champion after it.
GK = 16
CAT = [list(r) for r in DEFAULT_CATALOGUE] + [catalogue_row("GoldenKnight", 4, "TROOP")]
OPP_DECK = ("knight", "archers", "giant", "arrows", "golden-knight", "mini-pekka", "baby-dragon",
            "zap")
TEAM_FORMS = {"skeletons": 1, "musketeer": 2, "ice-golem": 2}
PRESS_EVENTS = ((240, "team", "musketeer"), (330, "opponent", "golden-knight"),
                (360, "team", "ice-golem"))
#: Every card a side plays in the payload is in its hand (as in test_trust).
HOG_HAND, HOG_NEXT = [5, 2, 6, 0], 1          # Musketeer, HogRider, Skeletons, Cannon
OPP_HAND, OPP_NEXT = [8, 11, 9, 10], 12       # Knight, Arrows, Archer, Giant
#: Blue's abilities rows as an engine that names its cards might send them, Ice Golem first:
#: [available, spent, cost, card_id, cooldown_ticks].
ENGINE_ROWS = [[1, 0, 2, 3, 0], [0, 1, 3, 5, 0]]
NO_HERO = DEPLOY_REASONS.index("NO_HERO")
#: play tick -> (side, action) of the payload's play rows (test_orient LEGACY_PLAY_ROWS).
PLAYS = {100: (1, (7, 3, 24)), 160: (-1, (10, 3, 24)), 240: (1, (4, 3, 17)),
         300: (-1, (13, 14, 8)), 420: (1, (8, 9, 27))}
ACT = ("act_card", "act_x", "act_y", "match", "split")
WAIT = (0, 0, 0)
#: The commands rs_rows / the re-drive send without presses, (tick, commands) of each step
#: that carried any: hand slots of HOG_HAND / OPP_HAND, coordinates x 18.
OFF_SENT = [
    (100, [(0, 0, 3500 * SUB, 7500 * SUB)]),      # Musketeer, blue slot 0
    (160, [(1, 0, 14500 * SUB, 24500 * SUB)]),    # Knight, red slot 0
    (240, [(0, 1, 3500 * SUB, 14500 * SUB)]),     # HogRider, blue slot 1
    (300, [(1, 1, 3500 * SUB, 8000 * SUB)]),      # Arrows, red slot 1
    (420, [(0, 2, 9000 * SUB, 5000 * SUB)]),      # Skeletons, blue slot 2
]
#: ... and with them: Musketeer's button (slot 4) in the step of the HogRider at 240, the
#: Golden Knight's (red slot 4) at 330, the Ice Golem's (slot 5) at 360.
ON_SENT = [
    OFF_SENT[0], OFF_SENT[1],
    (240, [(0, 1, 3500 * SUB, 14500 * SUB), (0, 4, 0, 0)]),
    OFF_SENT[3],
    (330, [(1, 4, 0, 0)]),
    (360, [(0, 5, 0, 0)]),
    OFF_SENT[4],
]
#: ... and under ENGINE_ROWS, whose order puts the Ice Golem on button 0.
NAMED_SENT = [ON_SENT[0], ON_SENT[1], (240, [ON_SENT[2][1][0], (0, 5, 0, 0)]), ON_SENT[3],
              ON_SENT[4], (360, [(0, 4, 0, 0)]), ON_SENT[6]]
ALL_OK = {"events": 3, "sent": 3, "ok": 3, "refused": {}, "skipped": {}, "order_mismatch": 0}


def press_payload(**kwargs):
    kwargs.setdefault("abilities", PRESS_EVENTS)
    return il_payload(opponent_deck=OPP_DECK, team_forms=TEAM_FORMS, **kwargs)


def on_order():
    """(tick, side, action, press) of rows_for's rows on press_payload() under --presses on:
    the legacy order (a play row, then the other seat's wait; two waits, blue first), with
    blue's press row after its play at 240, red's wait at 320 gone (it presses at 330, within
    10 ticks), red's press row at 330, and blue's press row at 360 in place of its wait."""
    out = []
    for t in range(20, 621, 20):
        if t in PLAYS:
            s, act = PLAYS[t]
            out.append((t, s, act, 0))
            if t == 240:
                out.append((240, 1, WAIT, 1))
            out.append((t, -s, WAIT, 0))
        elif t == 320:
            out += [(320, 1, WAIT, 0), (330, -1, WAIT, 1)]
        elif t == 360:
            out += [(360, 1, WAIT, 2), (360, -1, WAIT, 0)]
        else:
            out += [(t, 1, WAIT, 0), (t, -1, WAIT, 0)]
    return out


def states(abilities=None):
    """FakeBattle states: hands holding every played card, and `abilities` ({team: rows})."""
    abilities = abilities or {}

    def make(battle):
        return state_dict(battle.tick, players=[
            player_dict(0, hand=HOG_HAND, next_card=HOG_NEXT, abilities=abilities.get(0, [])),
            player_dict(1, hand=OPP_HAND, next_card=OPP_NEXT, abilities=abilities.get(1, []))])
    return make


def refuse_presses(battle, cmds, n):
    """Step results: every press (slot >= 4) refused NO_HERO, every play taken."""
    return [[c[0], NO_HERO if c[1] >= 4 else 0] for c in cmds]


def sent(battle):
    """(tick, commands) of every step of a FakeBattle that carried commands."""
    out, tick = [], int((battle.reset_kwargs or {}).get("start_tick") or 0)
    for call in battle.calls:
        if call.name == "step":
            cmds, n = call.args
            if cmds:
                out.append((tick, list(cmds)))
            tick += int(n)
    return out


def shard(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def f16(rows):
    return np.asarray(rows, dtype=np.float16).tolist()


@pytest.fixture
def rr(monkeypatch):
    """tools/rs_rows.py as the tools import it, its module settings at the defaults (restored
    after the test) and no engine config from the PC."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "TRUST": rr.TrustRule(), "OUT": rr.OUT,
                        "CARD_FLY": {}, "CARD_BUILDING": {}, "PRESSES": "off",
                        "BUTTONS": 2, "HAND_SIZE": 4}.items():
        monkeypatch.setattr(rr, name, value)
    return rr


@pytest.fixture
def rp(rr):
    """tools/rs_presses.py, the module rs_rows uses."""
    return rr.rs_presses


@pytest.fixture
def redrive(rr, monkeypatch):
    mod = importlib.import_module("royalesim_redrive")
    assert mod.rs_presses is rr.rs_presses
    monkeypatch.setattr(mod, "_ENGINE", {})
    return mod


@pytest.fixture
def engine(monkeypatch):
    """install(**battle_kwargs): a fake royalesim in sys.modules for this test."""
    def install(**kwargs):
        mod = make_fake_royalesim(**kwargs)
        monkeypatch.setitem(sys.modules, "royalesim", mod)
        return mod
    return install


def cfg(rr):
    return rr.rs_engine.EngineConfig.defaults()


def run(rr, payload=None, **kwargs):
    """rows_for on `payload` (press_payload()): (rows, info)."""
    info = {}
    rows, why = rr.rows_for(press_payload() if payload is None else payload,
                            rr._setup(cfg(rr)), cfg(rr), info=info, **kwargs)
    assert why == "ok"
    return rows, info


def same(a, b, keys):
    for k in keys:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


# ---------------------------------------------------------------------------
# rs_presses, by hand
# ---------------------------------------------------------------------------

def test_modes_constants_and_engine_numbers(rp):
    assert rp.PRESS_MODES == ("off", "on") and rp.PRESS_KIND == "activate_ability"
    assert (rp.FALLBACK_HAND_SIZE, rp.FALLBACK_BUTTONS, rp.FORM_HERO) == (4, 2, 2)
    assert rp.PRESS_REASONS == ("NO_HERO", "ABILITY_NOT_READY", "ABILITY_SPENT")
    assert rp.ABILITY_FEATS == ("reported", "available", "spent", "cost", "cooldown")
    assert rp.press_mode("off") == "off" and rp.press_mode("on") == "on"
    for bad in ("yes", True, None, 1):
        with pytest.raises(ValueError, match="--presses: one of off, on"):
            rp.press_mode(bad)
    assert rp.describe_presses("off").startswith("off (")
    assert rp.describe_presses("on").startswith("on (")
    ns = types.SimpleNamespace
    assert rp.button_count(ns(ABILITY_BUTTONS=3)) == 3 and rp.button_count(ns()) == 2
    assert rp.button_count(ns(ABILITY_BUTTONS=0)) == 0
    assert rp.button_count(ns(ABILITY_BUTTONS="3")) == 3
    for bad in (True, -1, "x", None, 2.5j):
        assert rp.button_count(ns(ABILITY_BUTTONS=bad)) == 2, bad
    assert rp.hand_size(ns(HAND_SIZE=5)) == 5 and rp.hand_size(ns()) == 4
    assert rp.command(1, 0) == (1, 4, 0, 0) and rp.command(0, 2, hand=5) == (0, 7, 0, 0)


def test_split_key_is_the_redrive_rule(rp, redrive):
    for key in ("musketeer-hero", "skeletons-ev1", "knight", "ice-golem-hero", "hero", "-ev1",
                "x-ev1-hero"):
        assert rp.split_key(key) == redrive.base_key(key), key
    assert rp.split_key("ice-golem-hero") == ("ice-golem", 2)


def test_champion_ids_by_column_else_by_name(rp):
    cat = [catalogue_row("Knight", 3, "TROOP"), catalogue_row("GoldenKnight", 4, "TROOP"),
           catalogue_row("ArcherQueen", 5, "TROOP"), catalogue_row("Monk", 5, "TROOP")]
    assert rp.champion_ids(cat) == {1, 2, 3}
    fields = ["name", "placement", "elixir", "count", "radius", "flying", "hitpoints",
              "footprint_tiles", "card_kind", "champion"]
    flagged = [row + [flag] for row, flag in zip(cat, (False, True, False, 1))]
    assert rp.champion_ids(flagged, fields) == {1, 3}             # the column decides
    short = flagged[:2] + [cat[2], flagged[3]]                    # a row without the column
    assert rp.champion_ids(short, fields) == {1, 2, 3}            # ... falls back to its name
    assert rp.champion_ids(flagged, fields[:9]) == {1, 2, 3}      # no "champion" field: names
    assert rp.champion_ids([]) == frozenset()


def test_side_buttons_heroes_in_dealt_order_then_champions(rp):
    dealt, forms = [0, 1, 2, 5, 3, 6, 4, 7], [0, 0, 0, 2, 2, 1, 0, 0]
    assert rp.side_buttons(dealt, forms, frozenset()) == [5, 3]   # Musketeer, then Ice Golem
    assert rp.side_buttons(list(range(8)), [0, 0, 0, 2, 0, 2, 1, 0], ()) == [3, 5]  # recorded order
    assert rp.side_buttons([16, 5, 3, 8], [0, 2, 0, 0], {16}) == [5, 16]   # heroes first
    assert rp.side_buttons([16, 5, 3, 8], None, {16, 8}) == [16, 8]
    assert rp.side_buttons([16, 5], [2, 2], {16}) == [16, 5]       # a hero entry counts once
    assert rp.side_buttons([5, 3], [2], ()) == [5]                 # forms short: the rest base
    assert rp.side_buttons([], [], {16}) == []


def test_engine_order_when_the_engine_names_its_cards(rp):
    named = {"abilities": ENGINE_ROWS}
    assert rp.engine_order(named, [5, 3]) == ([3, 5], True)
    assert rp.engine_order(named, (3, 5)) == ([3, 5], False)
    anon = {"abilities": [[1, 0, 2, -1, -1], [0, 0, 3, 5, 0]]}
    assert rp.engine_order(anon, [5, 3]) == ([5, 3], False)
    assert rp.engine_order({"abilities": [[1, 0, 2]]}, [5]) == ([5], False)   # no card column
    assert rp.engine_order({"abilities": []}, [5, 3]) == ([5, 3], False)
    assert rp.engine_order({}, (5,)) == ([5], False)


def test_event_keys_and_parse(rp):
    assert rp.event_keys({"card_key": "musketeer-hero",
                          "ability_source_candidates": ["x", "y"]}) == ("musketeer-hero",)
    assert rp.event_keys({"card_key": None,
                          "ability_source_candidates": ["a", "b", "a", "", None, 3]}) == ("a", "b")
    assert rp.event_keys({"card_key": ""}) == () and rp.event_keys({}) == ()
    names = {"musketeer": "Musketeer", "ice-golem": "IceGolemite", "golden-knight": "GoldenKnight",
             "void": "DarkMagic"}
    ids = {"Musketeer": 5, "IceGolemite": 3, "GoldenKnight": GK}
    cands = ["musketeer-hero", "ice-golem-hero", "musketeer"]
    events = [
        {"kind": "play_card", "side": "team", "card_key": "hog-rider", "replay_tick_20hz": 10,
         "source_index": 0},
        {"kind": "activate_ability", "side": "opponent", "card_key": "golden-knight",
         "replay_tick_20hz": 300, "source_index": 5},
        {"kind": "activate_ability", "side": "team", "card_key": None,
         "ability_source_candidates": cands, "replay_tick_20hz": 250, "source_index": 4},
        {"kind": "activate_ability", "side": "team", "card_key": "void", "replay_tick_20hz": 250,
         "source_index": 3},
        {"kind": "activate_ability", "side": "team", "card_key": "rune-giant",
         "replay_tick_20hz": None},
    ]
    P = rp.Press
    assert rp.parse(events, names, ids) == [
        P(0, 0, (), ("rune-giant",), 0),                  # no tick: 0; not in the card map
        P(250, 0, (), ("void",), 3),                       # mapped, but not in the catalogue
        P(250, 0, (5, 3), tuple(cands), 4),                # Musketeer twice counts once
        P(300, 1, (GK,), ("golden-knight",), 5),
    ]


def test_button_of(rp):
    assert rp.button_of((5,), [5, 3]) == (0, "ok")
    assert rp.button_of((3,), [5, 3]) == (1, "ok")
    assert rp.button_of((5, 3), [5, 3]) == (None, "ambiguous")
    assert rp.button_of((2, 3), [5, 3]) == (1, "ok")       # HogRider is no button
    assert rp.button_of((2,), [5, 3]) == (None, "unknown")
    assert rp.button_of((), [5, 3]) == (None, "unknown")
    assert rp.button_of((GK,), []) == (None, "unknown")


def test_ability_features(rp):
    player = {"abilities": [[1, 0, 3, 5, 0], [0, 1, 2, 3, 250, 99]]}
    cards, feats = rp.ability_features(player, 3, [5, 3])
    assert cards == [7, 5, 0]
    assert feats == [[1.0, 1.0, 0.0, 0.3, 0.0], [1.0, 0.0, 1.0, 0.2, 0.625], [0.0] * 5]
    # cooldown clipped at COOLDOWN_TICKS; the row's own card wins over the caller's
    assert rp.ability_features({"abilities": [[1, 0, 1, GK, 900]]}, 1, [5]) == (
        [GK + 2], [[1.0, 1.0, 0.0, 0.1, 1.0]])
    # a row without card_id / cooldown: the caller's card, no cooldown
    assert rp.ability_features({"abilities": [[1, 0, 2]]}, 1, [9]) == ([11], [[1.0, 1.0, 0.0,
                                                                                0.2, 0.0]])
    # a reported button nobody names reads 1; no rows: the deck's cards and zeros
    assert rp.ability_features({"abilities": [[0, 0, 3, -1, -1]]}, 1) == (
        [1], [[1.0, 0.0, 0.0, 0.3, 0.0]])
    assert rp.ability_features({"abilities": []}, 2, [5]) == ([7, 0], [[0.0] * 5, [0.0] * 5])
    assert rp.ability_features({}, 0) == ([], [])


def test_press_log_vocab_entry_and_describe(rp):
    log = rp.PressLog(events=4)
    log.skip("ambiguous")
    log.skip("unknown")
    log.skip("ambiguous")
    log.sent = 2
    assert log.result("OK") is True and log.result("NO_HERO") is False
    got = log.to_json()
    assert got == {"events": 4, "sent": 2, "ok": 1, "refused": {"NO_HERO": 1},
                   "skipped": {"ambiguous": 2, "unknown": 1}, "order_mismatch": 0}
    assert rp.describe_log(got) == ("2 of 4 presses sent, 1 accepted, refused NO_HERO 1, "
                                    "skipped ambiguous 2, unknown 1")
    assert rp.describe_log(dict(got, order_mismatch=2)).endswith(
        ", engine button order differed on 2 sides")
    assert rp.describe_log({}) == "0 of 0 presses sent, 0 accepted"
    entry = rp.vocab_entry(3, 4)
    assert {k: entry[k] for k in ("mode", "buttons", "slot_base", "act_press", "ability_feats",
                                  "cost_scale", "cooldown_ticks")} == {
        "mode": "on", "buttons": 3, "slot_base": 4, "act_press": "0 none, k + 1 button k",
        "ability_feats": ["reported", "available", "spent", "cost", "cooldown"],
        "cost_scale": 10, "cooldown_ticks": 400}
    assert json.loads(json.dumps(entry)) == entry


def test_rs_presses_is_pure():
    text = (TOOLS / "rs_presses.py").read_text(encoding="utf-8")
    for bad in ("import numpy", "import royalesim", "import torch", "gamestate", "memread"):
        assert bad not in text, bad


# ---------------------------------------------------------------------------
# rs_rows: obs_row, rows_for, build
# ---------------------------------------------------------------------------

def test_obs_row_press_columns(rr, engine, monkeypatch):
    engine(catalogue=CAT)
    rr._setup(cfg(rr))
    assert (rr.BUTTONS, rr.HAND_SIZE) == (2, 4)               # the fake's ABILITY_BUTTONS
    state = state_dict(100, players=[player_dict(0, abilities=[[1, 0, 3, 5, 0]]),
                                     player_dict(1)])
    view = rr.rs_memory.empty_view()
    old = rr.obs_row(state, 0, SUB, view, 100)
    assert "ability_card" not in old and "ability_state" not in old
    new = rr.obs_row(state, 0, SUB, view, 100, presses="on", buttons=[5, 3])
    assert list(new) == list(old) + ["ability_card", "ability_state"]
    same(old, new, old)
    assert new["ability_card"].dtype == np.int16 and new["ability_card"].tolist() == [7, 5]
    assert new["ability_state"].dtype == np.float16 and new["ability_state"].shape == (2, 5)
    assert new["ability_state"].tolist() == f16([[1, 1, 0, 0.3, 0], [0, 0, 0, 0, 0]])
    red = rr.obs_row(state, 1, SUB, view, 100, presses="on")
    assert red["ability_card"].tolist() == [0, 0] and not red["ability_state"].any()
    monkeypatch.setattr(rr, "PRESSES", "on")                  # the module setting
    assert rr.obs_row(state, 1, SUB, view, 100)["ability_state"].shape == (2, 5)
    monkeypatch.setattr(rr, "BUTTONS", 3)
    assert rr.obs_row(state, 0, SUB, view, 100)["ability_card"].tolist() == [7, 0, 0]
    with pytest.raises(ValueError, match="--presses"):
        rr.obs_row(state, 0, SUB, view, 100, presses="maybe")


def test_presses_off_is_the_rows_before(rr, engine):
    """Golden: the default and --presses off give the rows, info and engine calls of the same
    match without its activate_ability events."""
    mod = engine(catalogue=CAT, states=states())
    runs = {}
    for name, payload, kwargs in (("default", press_payload(), {}),
                                  ("off", press_payload(), {"presses": "off"}),
                                  ("none", press_payload(abilities=()), {})):
        rows, info = run(rr, payload, **kwargs)
        runs[name] = rows, info, mod.battles[-1]
    base, base_info, base_battle = runs["none"]
    assert len(base) == 62 and base_info == {"trust": {"cut": None, "why": "none"}}
    assert sent(base_battle) == OFF_SENT
    assert [(r["tick"], r["side"], tuple(r["action"])) for r in base if r["action"][0]] == \
        [(t, s, a) for t, (s, a) in PLAYS.items()]
    for name in ("default", "off"):
        rows, info, battle = runs[name]
        assert info == base_info and battle.calls == base_battle.calls, name
        assert len(rows) == len(base)
        for a, b in zip(rows, base):
            assert list(a) == list(b)
            same(a, b, a)
    # build: the columns and arrays of before
    assert all("press" not in r and "ability_card" not in r for r in runs["default"][0])


def test_presses_on_rows_labels_and_commands(rr, engine):
    mod = engine(catalogue=CAT, states=states())
    rows, info = run(rr, presses="on")
    battle = mod.battles[-1]
    assert [(r["tick"], r["side"], tuple(r["action"]), r["press"]) for r in rows] == on_order()
    assert len(rows) == 63
    assert sent(battle) == ON_SENT
    assert battle.reset_kwargs["decks"] == [[0, 1, 2, 5, 3, 6, 4, 7],
                                            [8, 9, 10, 11, GK, 13, 14, 15]]
    assert battle.reset_kwargs["forms"] == [[0, 0, 0, 2, 2, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]
    assert info == {"trust": {"cut": None, "why": "none"}, "presses": ALL_OK}
    for r in rows:
        assert r["ability_card"].tolist() == ([7, 5] if r["side"] == 1 else [GK + 2, 0])
        assert r["ability_state"].shape == (2, 5) and not r["ability_state"].any()
        assert r["trusted"] and type(r["press"]) is int
    # Every other column is the legacy row's: a press row's is the row of its side and tick
    # (blue's play row at 240, its wait row at 360) but for the action; red's at 330 is new.
    legacy, _ = run(rr)
    old = {(r["tick"], r["side"]): r for r in legacy}
    matched = 0
    for r in rows:
        key = (r["tick"], r["side"])
        if key not in old:
            assert key == (330, -1)
            continue
        o = old[key]
        assert set(r) == set(o) | {"ability_card", "ability_state", "press"}
        same(r, o, [k for k in o if k != "action"] if r["press"] else o)
        matched += 1
    assert matched == 62 and (320, -1) in old


def test_presses_on_follow_the_module_setting_and_check_the_mode(rr, engine, monkeypatch):
    engine(catalogue=CAT, states=states())
    monkeypatch.setattr(rr, "PRESSES", "on")
    rows, info = run(rr)
    assert len(rows) == 63 and info["presses"] == ALL_OK
    rows, info = run(rr, presses="off")
    assert len(rows) == 62 and "presses" not in info
    with pytest.raises(ValueError, match="--presses"):
        run(rr, presses="sometimes")


def test_the_engines_button_order_wins(rr, engine):
    mod = engine(catalogue=CAT, states=states({0: ENGINE_ROWS}))
    rows, info = run(rr, presses="on")
    assert [(r["tick"], r["side"], r["press"]) for r in rows if r["press"]] == \
        [(240, 1, 2), (330, -1, 1), (360, 1, 1)]
    assert sent(mod.battles[-1]) == NAMED_SENT
    assert info["presses"] == dict(ALL_OK, order_mismatch=1)
    blue = [r for r in rows if r["side"] == 1]
    assert all(r["ability_card"].tolist() == [5, 7] for r in blue)
    assert all(r["ability_state"].tolist() == f16([[1, 1, 0, 0.2, 0], [1, 0, 1, 0.3, 0]])
               for r in blue)
    red = [r for r in rows if r["side"] == -1]
    assert all(r["ability_card"].tolist() == [GK + 2, 0] and not r["ability_state"].any()
               for r in red)


def test_a_refused_press_cuts_trust(rr, engine):
    engine(catalogue=CAT, states=states(), step_results=refuse_presses)
    rows, info = run(rr, presses="on", trust="v2")
    assert len(rows) == 63
    assert info["trust"] == {"cut": 240, "why": "press", "cut_v2": 240, "why_v2": "press",
                             "votes": [None, None]}
    assert info["presses"] == dict(ALL_OK, ok=0, refused={"NO_HERO": 3})
    assert [(r["tick"], r["side"]) for r in rows if r["trusted"]] == \
        [(20, 1), (20, -1), (40, 1), (40, -1)]                     # tick + 200 <= 240
    assert [r["trusted_v2"] for r in rows] == [r["trusted"] for r in rows]


def test_unplaceable_presses_are_skipped_and_cut_trust(rr, engine):
    mod = engine(catalogue=CAT, states=states())
    payload = press_payload()
    musk, gk, golem = [e for e in payload["events"] if e["kind"] == "activate_ability"]
    assert (musk["replay_tick_20hz"], gk["replay_tick_20hz"], golem["replay_tick_20hz"]) == \
        (240, 330, 360)
    musk["card_key"] = None                                       # both heroes: ambiguous
    musk["ability_source_candidates"] = ["musketeer-hero", "ice-golem-hero"]
    golem["card_key"] = "hog-rider"                               # no button: unknown
    del golem["ability_source_candidates"]
    rows, info = run(rr, payload, presses="on")
    assert len(rows) == 61
    assert [(r["tick"], r["side"], r["press"]) for r in rows if r["press"]] == [(330, -1, 1)]
    assert [(r["side"], tuple(r["action"])) for r in rows if r["tick"] == 240] == \
        [(1, PLAYS[240][1]), (-1, WAIT)]
    assert [r["side"] for r in rows if r["tick"] == 360] == [-1]   # blue acted: no wait row
    assert info["trust"] == {"cut": 240, "why": "press"}
    assert info["presses"] == {"events": 3, "sent": 1, "ok": 1, "refused": {},
                               "skipped": {"ambiguous": 1, "unknown": 1}, "order_mismatch": 0}
    assert sent(mod.battles[-1]) == [ON_SENT[0], ON_SENT[1], OFF_SENT[2], ON_SENT[3],
                                     ON_SENT[4], ON_SENT[6]]
    assert sum(r["trusted"] for r in rows) == 4


def test_a_press_after_the_last_play_extends_the_run(rr, engine):
    """conftest's own payload: the hero Musketeer presses at 500, after the last play (420).
    The run ends at 500 + HORIZON = 700 (70 rows) and the wait tail of trusted_v2 at 520."""
    mod = engine(states=states())
    rows, info = run(rr, il_payload(), presses="on", trust="v2")
    assert len(rows) == 70 and rows[-1]["tick"] == 700
    assert [(r["tick"], r["side"], r["press"]) for r in rows if r["press"]] == [(500, 1, 1)]
    assert sent(mod.battles[-1])[-1] == (500, [(0, 4, 0, 0)])
    assert info["presses"] == {"events": 1, "sent": 1, "ok": 1, "refused": {}, "skipped": {},
                               "order_mismatch": 0}
    assert sum(r["trusted_v2"] for r in rows) == 52               # ticks 20-520, two rows each
    legacy, _ = run(rr, il_payload(), trust="v2")
    assert len(legacy) == 62 and sum(r["trusted_v2"] for r in legacy) == 44


def test_build_writes_press_columns_only_under_on(rr, engine, tmp_path):
    engine(catalogue=CAT, states=states())
    text = json.dumps(press_payload(tag="press-1"))
    eng = cfg(rr).to_json()
    assert rr.build(1, [text], False, str(tmp_path), eng) == (1, 62, {})
    old = shard(tmp_path / "shard-0001.npz")
    assert sorted(old) == sorted(rr.FIELDS + ACT) and rr.PRESSES == "off"
    batch, n, skipped, st = rr.build(2, [text], False, str(tmp_path), eng, presses="on",
                                     stats=True)
    assert (batch, n, skipped) == (2, 63, {}) and rr.PRESSES == "on"
    new = shard(tmp_path / "shard-0002.npz")
    assert rr.PRESS_COLUMNS == ("act_press", "ability_card", "ability_state")
    assert sorted(new) == sorted(rr.FIELDS + ACT + rr.PRESS_COLUMNS)
    order = on_order()
    assert new["act_press"].dtype == np.int16
    assert new["act_press"].tolist() == [p for *_, p in order]
    assert new["act_card"].tolist() == [a[0] for _, _, a, _ in order]
    assert new["ability_card"].dtype == np.int16 and new["ability_card"].shape == (63, 2)
    assert new["ability_card"].tolist() == [[7, 5] if s == 1 else [GK + 2, 0]
                                            for _, s, _, _ in order]
    assert new["ability_state"].dtype == np.float16 and new["ability_state"].shape == (63, 2, 5)
    assert st["presses"] == dict(ALL_OK, press_rows=3)
    assert (st["rows"], st["play_rows"], st["trusted"], st["hog_rows"]) == (63, 5, 63, 32)
    assert rr.build(3, [text], False, str(tmp_path), eng)[1] == 63    # None: keep on
    back = rr.build(4, [text], False, str(tmp_path), eng, presses="off", stats=True)
    assert back[:3] == (4, 62, {}) and "presses" not in back[3]
    again = shard(tmp_path / "shard-0004.npz")
    assert sorted(again) == sorted(old) and all(np.array_equal(again[k], old[k]) for k in old)
    with pytest.raises(ValueError, match="--presses"):
        rr.build(5, [text], False, str(tmp_path), eng, presses="maybe")
    assert rr.PRESSES == "off" and not (tmp_path / "shard-0005.npz").exists()


# ---------------------------------------------------------------------------
# rs_rows: stats, manifest, vocab.json, main
# ---------------------------------------------------------------------------

def test_trust_stats_and_manifest_sum_press_counts(rr):
    rows = [{"tick": 20, "action": (0, 0, 0), "trusted": True, "hog": True, "press": 1},
            {"tick": 20, "action": (7, 1, 1), "trusted": True, "hog": False, "press": 0}]
    infos = [{"trust": {"cut": None, "why": "none"}, "presses": dict(ALL_OK, events=2, sent=1,
                                                                     ok=1)}]
    st = rr.trust_stats(rows, infos)
    assert st["presses"] == dict(ALL_OK, events=2, sent=1, ok=1, press_rows=1)
    assert "presses" not in rr.trust_stats(rows, [{"trust": {"cut": None, "why": "none"}}])
    a = {"rows": 63, "play_rows": 5, "trusted": 63, "hog_rows": 32, "hog_trusted": 32,
         "matches": 1, "cuts": {"none": 1}, "presses": dict(ALL_OK, press_rows=3)}
    b = {"rows": 61, "play_rows": 5, "trusted": 57, "hog_rows": 30, "hog_trusted": 30,
         "matches": 1, "cuts": {"press": 1},
         "presses": {"press_rows": 1, "events": 3, "sent": 1, "ok": 0, "refused": {"NO_HERO": 1},
                     "skipped": {"ambiguous": 1, "unknown": 1}, "order_mismatch": 1}}
    totals = rr.merge_stats(rr.merge_stats({}, a), b)
    assert totals["presses"] == {"press_rows": 4, "events": 6, "sent": 4, "ok": 3,
                                 "refused": {"NO_HERO": 1},
                                 "skipped": {"ambiguous": 1, "unknown": 1}, "order_mismatch": 1}
    man = rr.manifest(totals, [], {})
    assert man["presses"] == totals["presses"] and man["cuts"] == {"none": 1, "press": 1}
    assert rr.describe_manifest(man) == (
        "trusted 120 of 124 rows (96.8%; Hog 2.6 100.0%); 4 of 6 presses sent, 3 accepted, "
        "refused NO_HERO 1, skipped ambiguous 1, unknown 1, engine button order differed on 1 "
        "sides, 4 press rows")
    legacy = rr.manifest({k: v for k, v in totals.items() if k != "presses"}, [], {})
    assert "presses" not in legacy
    assert rr.describe_manifest(legacy) == "trusted 120 of 124 rows (96.8%; Hog 2.6 100.0%)"


def test_vocab_records_presses_only_when_on(rr, rp):
    cat = [list(r) for r in CAT]
    units, eng = list(DEFAULT_UNIT_TYPES), cfg(rr)
    fp = {"version": "0.0.0+fake", "card_table": None, "id": "f00d"}
    base = rr.vocab_record(cat, units, eng, fp, {}, False)
    assert rr.press_vocab("off") == {}
    for same_vocab in (rr.vocab_record(cat, units, eng, fp, {}, False, presses="off"),
                       rr.vocab_record(cat, units, eng, fp, {}, False, presses={"mode": "off"}),
                       rr.vocab_record(cat, units, eng, fp, {}, False, **rr.press_vocab("off"))):
        assert json.dumps(same_vocab) == json.dumps(base)       # the vocab.json of before
    assert "presses" not in base and "extra_columns" not in base
    on = rr.vocab_record(cat, units, eng, fp, {}, False, **rr.press_vocab("on"))
    assert list(on) == list(base) + ["presses", "extra_columns"]
    assert on["presses"] == rp.vocab_entry(2, 4) and on["presses"]["buttons"] == 2
    assert on["extra_columns"] == ["act_press", "ability_card", "ability_state"]
    assert rr.vocab_record(cat, units, eng, fp, {}, False, presses="on") == on
    assert rr.vocab_record(cat, units, eng, fp, {}, False, presses=on["presses"]) == on
    both = rr.vocab_record(cat, units, eng, fp, {}, False, trust="v2", presses="on")
    assert list(both) == list(base) + ["trust", "presses", "extra_columns"]
    assert both["extra_columns"] == ["trusted_v2", "act_press", "ability_card", "ability_state"]
    for bad in ("maybe", {"mode": "sometimes"}, {}):
        with pytest.raises(ValueError, match="--presses"):
            rr.vocab_record(cat, units, eng, fp, {}, False, presses=bad)
    with pytest.raises(ValueError, match="--presses"):
        rr.press_vocab("maybe")


def test_rows_main_wires_presses_into_workers_vocab_and_console():
    text = (TOOLS / "rs_rows.py").read_text(encoding="utf-8")
    main = text[text.index("def main()"):]
    for needle in ('"--presses"', "default=rs_presses.PRESS_OFF", "PRESSES = args.presses",
                   "presses=PRESSES, stats=True", "**press_vocab(PRESSES)",
                   "describe_presses(PRESSES)"):
        assert needle in main, needle


# ---------------------------------------------------------------------------
# royalesim_redrive
# ---------------------------------------------------------------------------

def test_redrive_presses_off_is_the_redrive_before(rr, redrive, engine):
    mod = engine(catalogue=CAT, states=states())
    outs, calls = {}, {}
    for name, payload, kwargs in (("default", press_payload(), {}),
                                  ("off", press_payload(), {"presses": "off"}),
                                  ("none", press_payload(abilities=()), {})):
        outs[name] = redrive.one(json.dumps(payload), engine=cfg(rr), **kwargs)
        calls[name] = mod.battles[-1].calls
    assert outs["default"] == outs["off"] == outs["none"]
    assert calls["default"] == calls["off"] == calls["none"]
    out = outs["default"]
    assert out["ok"] and out["forced"] == {} and out["plays"] == 5 and "presses" not in out
    assert sent(mod.battles[-1]) == OFF_SENT


@pytest.mark.parametrize("named", [False, True])
def test_redrive_sends_presses_in_the_step_of_their_tick(rr, redrive, engine, named):
    mod = engine(catalogue=CAT, states=states({0: ENGINE_ROWS} if named else None))
    out = redrive.one(json.dumps(press_payload()), engine=cfg(rr), presses="on")
    assert out["ok"] and out["error"] == "" and out["forced"] == {} and out["plays"] == 5
    assert sent(mod.battles[-1]) == (NAMED_SENT if named else ON_SENT)
    assert out["presses"] == dict(ALL_OK, order_mismatch=int(named))
    assert redrive._ENGINE["champions"] == {GK}


def test_redrive_counts_refused_and_skipped_presses_apart(rr, redrive, engine):
    engine(catalogue=CAT, states=states(), step_results=refuse_presses)
    out = redrive.one(json.dumps(press_payload()), engine=cfg(rr), presses="on")
    assert out["ok"] and out["forced"] == {}                      # not a play's refusal
    assert out["presses"] == dict(ALL_OK, ok=0, refused={"NO_HERO": 3})
    payload = press_payload()
    musk = next(e for e in payload["events"] if e["kind"] == "activate_ability")
    musk["card_key"] = "hog-rider"
    out = redrive.one(json.dumps(payload), engine=cfg(rr), presses="on")
    assert out["presses"] == dict(ALL_OK, sent=2, ok=0, refused={"NO_HERO": 2},
                                  skipped={"unknown": 1})
    with pytest.raises(ValueError, match="--presses"):
        redrive.one(json.dumps(payload), engine=cfg(rr), presses="maybe")


def test_redrive_output_path_and_totals(redrive):
    reports = ROOT / "reports" / "redrive"
    assert redrive.output_path("b2000") == reports / "b2000_royalesim.jsonl"
    assert redrive.output_path("b2000", "off") == reports / "b2000_royalesim.jsonl"
    assert redrive.output_path("b2000", "on") == reports / "b2000_royalesim_presses.jsonl"
    rows = [{"presses": {"events": 2, "sent": 1, "ok": 1, "refused": {},
                         "skipped": {"unknown": 1}, "order_mismatch": 0}},
            {"presses": {"events": 1, "sent": 1, "ok": 0, "refused": {"NO_HERO": 1},
                         "skipped": {}, "order_mismatch": 1}},
            {"ok": True}]
    assert redrive.press_totals(rows) == {"events": 3, "sent": 2, "ok": 1,
                                          "refused": {"NO_HERO": 1}, "skipped": {"unknown": 1},
                                          "order_mismatch": 1}
    assert redrive.press_totals([{"ok": True}]) == {}


def test_redrive_main_wires_presses():
    text = (TOOLS / "royalesim_redrive.py").read_text(encoding="utf-8")
    main = text[text.index("def main()"):]
    for needle in ('"--presses"', "default=rs_presses.PRESS_OFF",
                   "partial(one, presses=args.presses)", "output_path(args.pair, args.presses)",
                   "press_totals(ok)"):
        assert needle in main, needle


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

#: A hero Musketeer and nothing else with a button; plain opponent at level 11.
REAL = {"team_forms": {"musketeer": 2}, "opponent_forms": {}, "opponent_levels": 11}


@pytest.mark.engine
def test_real_engine_takes_a_press_once_the_hero_stands(rr, redrive):
    """The Musketeer goes down at 100: a press at 95 has no hero behind it, one at 200 does
    (deployed by 120, about 5.6 elixir by then, 3 for the press)."""
    cfg_real = rr.rs_engine.EngineConfig.load()
    payload = il_payload(abilities=((95, "team", "musketeer"), (200, "team", "musketeer")),
                         **REAL)
    out = redrive.one(json.dumps(payload), engine=cfg_real, presses="on")
    assert out["ok"], out
    assert out["presses"] == {"events": 2, "sent": 2, "ok": 1, "refused": {"NO_HERO": 1},
                              "skipped": {}, "order_mismatch": 0}, out


@pytest.mark.engine
def test_real_engine_press_row_shows_the_button_then_spends_it(rr):
    import royalesim
    cfg_real = rr.rs_engine.EngineConfig.load()
    ctx = rr._setup(cfg_real)
    assert rr.BUTTONS == rr.rs_presses.button_count(royalesim)
    info = {}
    rows, why = rr.rows_for(il_payload(abilities=((200, "team", "musketeer"),), **REAL), ctx,
                            cfg_real, presses="on", info=info)
    assert why == "ok" and info["presses"]["ok"] == 1, info
    ids = {row[0]: i for i, row in enumerate(ctx[3])}
    press = [r for r in rows if r["press"]]
    assert [(r["tick"], r["side"], r["press"]) for r in press] == [(200, 1, 1)]
    row = press[0]
    assert row["ability_card"][0] == ids["Musketeer"] + 2
    assert row["ability_state"].shape == (rr.BUTTONS, 5)
    st = row["ability_state"].astype(float)
    assert st[0, :3].tolist() == [1.0, 1.0, 0.0]                 # reported, available, not spent
    assert st[0, 3] == pytest.approx(0.3, abs=1e-3)               # 3 elixir
    after = next(r for r in rows if r["tick"] == 220 and r["side"] == 1)
    assert float(after["ability_state"][0, 2]) == 1.0             # the hero's one charge is used
