"""Wiring of E08 (opponent memory dated at execution), E16 (--units vision rows) and L01 (the
league Wilson interval) into tools/rs_rows.py and tools/rs_eval.py.

Golden: obs_row in the types and live modes is a frozen copy of the obs_row before this batch;
rows_for --mem-lag legacy hands obs_row, row by row, exactly the dict the old loop built (each
play appended right after its row); legacy and default rows differ in exactly the four rows a
same-tick play touches, worked by hand; rs_eval's legacy memory is the old dict under delay 0
and 21; rs_eval.wilson is tools.league.stats.wilson, which equals the old formula clipped to
[0, 1]. Everything runs on conftest's fake royalesim (no engine, no corpus, no torch except
the one RSAgent test).
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import math
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    FIELD,
    SUBTILE_PER_MILLITILE,
    entity_row,
    il_payload,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
CLASSES = ROOT / "data" / "vision" / "classes.json"
SUB = SUBTILE_PER_MILLITILE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
                        "OUT": rr.OUT, "CARD_FLY": {}, "CARD_BUILDING": {}}.items():
        monkeypatch.setattr(rr, name, value)
    return types.SimpleNamespace(rows=rr, eval=ev, engine=eng)


def cfg(tools):
    return tools.engine.EngineConfig.defaults()


def legacy_mem(plays):
    """The dict rs_rows / rs_eval filled before E08: each (team, token, tick) appended in order."""
    mem = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}
    for team, card, tick in plays:
        mem["seen"][team].add(card)
        mem["last"][team].append((card, tick))
    return mem


def same_row(a, b):
    assert list(a) == list(b)
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        assert x.dtype == y.dtype and np.array_equal(x, y), key


# ---------------------------------------------------------------------------
# settings: --mem-lag, --units
# ---------------------------------------------------------------------------

def test_mem_lag_values_and_flag(tools):
    rr = tools.rows
    assert rr.MEM_LAG == rr.rs_memory.BOARD_LAG == 1 and rr.MEM_LEGACY == "legacy"
    assert rr.mem_lag_value("legacy") == rr.mem_lag_value(" Legacy ") == "legacy"
    assert rr.mem_lag_value("21") == 21 and rr.mem_lag_value(0) == 0
    lag = rr.mem_lag_value(np.int64(3))
    assert lag == 3 and type(lag) is int
    for bad in (-1, "-1", True, 1.5, "1.5", "", "x", None):
        with pytest.raises(ValueError, match="mem-lag"):
            rr.mem_lag_value(bad)
    ap = argparse.ArgumentParser()
    rr.add_mem_lag_arg(ap)
    assert ap.parse_args([]).mem_lag == 1
    assert ap.parse_args(["--mem-lag", "legacy"]).mem_lag == "legacy"
    assert ap.parse_args(["--mem-lag", "0"]).mem_lag == 0
    assert ap.parse_args(["--mem-lag", "22"]).mem_lag == 22
    for bad in ("-1", "x", "1.5"):
        with pytest.raises(SystemExit):
            ap.parse_args(["--mem-lag", bad])
    assert rr.describe_memory("legacy").startswith("legacy")
    assert rr.describe_memory(0).endswith("t - 1)") and rr.describe_memory(21).endswith("t - 21)")


def test_unit_modes(tools):
    rr = tools.rows
    assert rr.UNITS == rr.UNITS_TYPES == "types" and rr.UNIT_MODES == ("types", "vision")
    assert rr.unit_mode("types", True) == "types" and rr.unit_mode("vision") == "vision"
    with pytest.raises(ValueError, match="live-compat"):
        rr.unit_mode("vision", True)
    with pytest.raises(ValueError, match="--units"):
        rr.unit_mode("cards")


def _calls(path: Path, func: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func)
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def test_both_command_lines_take_mem_lag_and_rows_take_units():
    assert "add_mem_lag_arg" in _calls(TOOLS / "rs_rows.py", "main")
    assert "add_mem_lag_arg" in _calls(TOOLS / "rs_eval.py", "main")
    assert {"memory_vocab", "vocab_record", "unit_mode"} <= _calls(TOOLS / "rs_rows.py", "main")
    tree = ast.parse((TOOLS / "rs_rows.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    units = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "add_argument"
             and n.args and getattr(n.args[0], "value", None) == "--units"]
    assert len(units) == 1
    default = next(k.value for k in units[0].keywords if k.arg == "default")
    assert isinstance(default, ast.Name) and default.id == "UNITS_TYPES"


# ---------------------------------------------------------------------------
# obs_row golden: types and live as before this batch
# ---------------------------------------------------------------------------

def frozen_obs_row(rr, state, team, sub, mem, tick, live):
    """tools/rs_rows.obs_row before E08 / E16 (commit 1195892), frozen. Module names are read
    through `rr`; MAX_UNITS 48 and UF 7 are the values it had."""
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
    return {
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


def busy_state(tick: int, king_alive: bool = True) -> dict:
    """Both teams' units, more than MAX_UNITS of them: buildings, fliers, deploying, dead."""
    ents = [
        entity_row(uid=40, team=0, kind=1, card_id=0, x=9000 * SUB, y=7000 * SUB, hp=200,
                   max_hp=322, unit_type=2, deploy_ticks=4),
        entity_row(uid=41, team=1, kind=0, card_id=14, x=4000 * SUB, y=12000 * SUB, hp=450,
                   flying=True, unit_type=13),
        entity_row(uid=42, team=1, kind=0, card_id=8, x=14000 * SUB, y=20000 * SUB, hp=0,
                   max_hp=690, unit_type=8),                                   # dead
        entity_row(uid=43, team=0, kind=0, card_id=5, x=3000 * SUB, y=26000 * SUB, hp=100,
                   max_hp=282, unit_type=6),
    ]
    ents += [entity_row(uid=100 + k, team=k % 2, kind=0, card_id=6, x=(1000 + 300 * k) * SUB,
                        y=(2000 + 577 * k) % 30000 * SUB, hp=32, unit_type=7,
                        deploy_ticks=k % 3)
             for k in range(52)]
    players = [player_dict(0, elixir_milli=7300, hand=[2, 5, -1, 6], next_card=1,
                           tower_hp=(4824 if king_alive else 0, 3052, 1200)),
               player_dict(1, hand=[8, 9], next_card=-1, tower_hp=(4000, 0, 3052))]
    return state_dict(tick, players=players, entities=ents)


BUSY_MEM = {"seen": {0: {2, 3, 4, 5, 6, 7, 8, 9, 10}, 1: {10, 11, 12}},
            "last": {0: [(2, 100), (3, 200), (4, 300), (5, 400), (6, 1200)], 1: [(10, 50)]}}


def test_obs_row_types_and_live_are_the_frozen_obs_row(tools, monkeypatch):
    rr = tools.rows
    rr.F = dict(FIELD)
    for i, row in enumerate(DEFAULT_CATALOGUE):         # what _setup fills
        rr.CARD_FLY[i] = bool(row[5])
        rr.CARD_BUILDING[i] = row[8] == "BUILDING"
    n = 0
    for tick, king in ((1234, True), (3700, True), (2500, False)):
        state = busy_state(tick, king)
        for team in (0, 1):
            for live in (False, True):
                want = frozen_obs_row(rr, state, team, SUB, BUSY_MEM, tick, live)
                same_row(rr.obs_row(state, team, SUB, BUSY_MEM, tick, live=live), want)
                same_row(rr.obs_row(state, team, SUB, BUSY_MEM, tick, live=live,
                                    units="types"), want)
                n += 1
    assert n == 12
    # live wins over --units vision, which needs no class table then
    monkeypatch.setattr(rr, "UNITS", rr.UNITS_VISION)
    state = busy_state(1234)
    same_row(rr.obs_row(state, 0, SUB, BUSY_MEM, 1234, live=True),
             frozen_obs_row(rr, state, 0, SUB, BUSY_MEM, 1234, True))
    with pytest.raises(RuntimeError, match="_setup"):
        rr.obs_row(state, 0, SUB, BUSY_MEM, 1234, live=False)
    with pytest.raises(ValueError, match="--units"):
        rr.obs_row(state, 0, SUB, BUSY_MEM, 1234, live=False, units="cards")


# ---------------------------------------------------------------------------
# rows_for: opponent memory
# ---------------------------------------------------------------------------

#: Both seats play at 100 (blue's Musketeer first: the queue sorts blue before red), then
#: HogRider 240, Arrows 300 and Skeletons 420, each alone. Tokens: Musketeer 7, Knight 10,
#: HogRider 4, Arrows 13, Skeletons 8 (conftest catalogue id + 2).
SAME_TICK_PLAYS = ((100, "team", "musketeer", 3500, 7500), (100, "opponent", "knight", 14500, 24500),
                   (240, "team", "hog-rider", 3500, 14500), (300, "opponent", "arrows", 3500, 8000),
                   (420, "team", "skeletons", 9000, 5000))
#: (team, token, tick) of the five play rows, in row order.
PLAYED = [(0, 7, 100), (1, 10, 100), (0, 4, 240), (1, 13, 300), (0, 8, 420)]
#: Wait rows every 20 ticks from 20 to 620 for each seat not playing (no play within 10 ticks
#: follows any of them here): 31 ticks x 2 - 5 plays + 5 play rows.
N_ROWS = 62


def same_tick_payload() -> dict:
    return il_payload(tag="same-tick-1", plays=SAME_TICK_PLAYS)


def spy_rows(tools, monkeypatch, payload, **kwargs):
    """rows_for(payload), and (tick, team, a copy of the memory) of every obs_row call."""
    rr = tools.rows
    calls = []
    real = rr.obs_row

    def spy(state, team, sub, mem, tick, live=None, units=None):
        calls.append((tick, team, copy.deepcopy(mem)))
        return real(state, team, sub, mem, tick, live=live, units=units)

    monkeypatch.setattr(rr, "obs_row", spy)
    try:
        rows, why = rr.rows_for(payload, rr._setup(cfg(tools)), cfg(tools), **kwargs)
    finally:
        monkeypatch.setattr(rr, "obs_row", real)
    assert why == "ok"
    return rows, calls


@pytest.mark.parametrize("lag", ["legacy", 0, 1, 21])
def test_each_row_reads_the_old_dict_in_legacy_and_only_plays_that_ran_otherwise(
        fake_royalesim, tools, monkeypatch, lag):
    rows, calls = spy_rows(tools, monkeypatch, same_tick_payload(), mem_lag=lag)
    assert len(rows) == len(calls) == N_ROWS
    done = []           # the plays of the rows so far: before E08 each was appended right after
    for row, (tick, team, mem) in zip(rows, calls):
        assert (row["tick"], row["side"]) == (tick, 1 if team == 0 else -1)
        if lag == "legacy":
            assert mem == legacy_mem(done), (tick, team)
        else:            # ran by tick - max(1, lag): never a play of the row's own tick
            assert mem == legacy_mem([p for p in done if p[2] <= tick - max(1, lag)]), (tick, team)
        if row["action"][0]:
            done.append((team, int(row["action"][0]), tick))
    assert done == PLAYED


#: (tick, side, action card) -> ((opp_seen, opp_last, opp_last_age) legacy, the same at
#: BOARD_LAG), for the only rows where the two differ: red's play row at 100 and the wait
#: rows of the seat not playing at 240, 300 and 420. Ages (row tick - play tick) / 400.
DIFFERING = {
    (100, -1, 10): (([7] + [0] * 7, [7, 0, 0, 0], [0.0, 1.0, 1.0, 1.0]),
                    ([0] * 8, [0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0])),
    (240, -1, 0): (([4, 7] + [0] * 6, [7, 4, 0, 0], [0.35, 0.0, 1.0, 1.0]),
                   ([7] + [0] * 7, [7, 0, 0, 0], [0.35, 1.0, 1.0, 1.0])),
    (300, 1, 0): (([10, 13] + [0] * 6, [10, 13, 0, 0], [0.5, 0.0, 1.0, 1.0]),
                  ([10] + [0] * 7, [10, 0, 0, 0], [0.5, 1.0, 1.0, 1.0])),
    (420, -1, 0): (([4, 7, 8] + [0] * 5, [7, 4, 8, 0], [0.8, 0.45, 0.0, 1.0]),
                   ([4, 7] + [0] * 6, [7, 4, 0, 0], [0.8, 0.45, 1.0, 1.0])),
}


def _memory_columns(row):
    return (row["opp_seen"].tolist(), row["opp_last"].tolist(), row["opp_last_age"])


def test_same_tick_plays_by_hand(fake_royalesim, tools):
    rr = tools.rows
    ctx = rr._setup(cfg(tools))
    legacy, why = rr.rows_for(same_tick_payload(), ctx, cfg(tools), mem_lag="legacy")
    assert why == "ok"
    new, _ = rr.rows_for(same_tick_payload(), ctx, cfg(tools))          # MEM_LAG = BOARD_LAG
    assert len(legacy) == len(new) == N_ROWS
    keys = [(r["tick"], r["side"], r["action"][0]) for r in legacy]
    assert keys == [(r["tick"], r["side"], r["action"][0]) for r in new]
    assert sorted({k[0] for k in keys}) == list(range(20, 621, 20))
    seen = set()
    for old, row in zip(legacy, new):
        key = (old["tick"], old["side"], old["action"][0])
        if key in DIFFERING:
            seen.add(key)
            for got, (s, last, age) in zip((old, row), DIFFERING[key]):
                g_seen, g_last, g_age = _memory_columns(got)
                assert (g_seen, g_last) == (s, last), key
                assert np.array_equal(g_age, np.asarray(age, dtype=np.float16)), key
            same_row({k: v for k, v in old.items() if not k.startswith("opp_")},
                     {k: v for k, v in row.items() if not k.startswith("opp_")})
        else:
            same_row(old, row)
    assert seen == set(DIFFERING)
    # lag 0 builds the same rows as 1 (the view is taken before the tick's plays are recorded)
    for a, b in zip(new, rr.rows_for(same_tick_payload(), ctx, cfg(tools), mem_lag=0)[0]):
        same_row(a, b)
    # perception lag 21: blue's wait row at 120 does not know the Knight yet, at 140 it does
    late = {(r["tick"], r["side"], r["action"][0]): r
            for r in rr.rows_for(same_tick_payload(), ctx, cfg(tools), mem_lag=21)[0]}
    assert late[(120, 1, 0)]["opp_last"].tolist() == [0, 0, 0, 0]
    assert late[(140, 1, 0)]["opp_last"].tolist() == [10, 0, 0, 0]
    assert late[(320, -1, 0)]["opp_last"].tolist() == [7, 4, 0, 0]     # 240 + 21 <= 320
    with pytest.raises(ValueError, match="mem-lag"):
        rr.rows_for(same_tick_payload(), ctx, cfg(tools), mem_lag=-1)


def test_rows_follow_the_module_setting(fake_royalesim, tools, monkeypatch):
    rr = tools.rows
    ctx = rr._setup(cfg(tools))
    legacy, _ = rr.rows_for(same_tick_payload(), ctx, cfg(tools), mem_lag="legacy")
    monkeypatch.setattr(rr, "MEM_LAG", "legacy")
    for a, b in zip(legacy, rr.rows_for(same_tick_payload(), ctx, cfg(tools))[0]):
        same_row(a, b)


def test_build_writes_the_old_columns_and_keeps_the_settings(fake_royalesim, tools, tmp_path):
    rr = tools.rows
    text = json.dumps(same_tick_payload())
    eng = cfg(tools).to_json()
    columns = sorted(rr.FIELDS + ("act_card", "act_x", "act_y", "match", "split"))

    def shard(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    assert rr.build(1, [text], False, str(tmp_path), eng, mem_lag="legacy") == (1, N_ROWS, {})
    assert (rr.MEM_LAG, rr.UNITS, rr.LIVE) == ("legacy", "types", False)
    old = shard(tmp_path / "shard-0001.npz")
    assert sorted(old) == columns
    rows, _ = rr.rows_for(json.loads(text), rr._setup(cfg(tools)), cfg(tools), mem_lag="legacy")
    for k in rr.FIELDS:
        assert np.array_equal(old[k], np.stack([np.asarray(r[k]) for r in rows])), k
    assert old["opp_seen"].dtype == np.int16 and old["opp_last_age"].dtype == np.float16

    assert rr.build(2, [text], False, str(tmp_path), eng)[1] == N_ROWS     # None: keep legacy
    again = shard(tmp_path / "shard-0002.npz")
    assert all(np.array_equal(again[k], old[k]) for k in columns)

    assert rr.build(3, [text], False, str(tmp_path), eng, mem_lag=1)[1] == N_ROWS
    new = shard(tmp_path / "shard-0003.npz")
    assert rr.MEM_LAG == 1 and sorted(new) == columns
    differ = np.flatnonzero((new["opp_seen"] != old["opp_seen"]).any(1))
    assert [(int(old["tick"][i]), int(old["side"][i]), int(old["act_card"][i]))
            for i in differ] == sorted(DIFFERING)

    with pytest.raises(ValueError, match="live-compat"):
        rr.build(4, [text], True, str(tmp_path), eng, units="vision")
    assert (rr.LIVE, rr.UNITS, rr.MEM_LAG) == (False, "types", 1)
    assert not (tmp_path / "shard-0004.npz").exists()


# ---------------------------------------------------------------------------
# vocab.json
# ---------------------------------------------------------------------------

def _vocab_inputs(tools):
    cat = [list(r) for r in DEFAULT_CATALOGUE]
    fp = {"version": "0.0.0+fake", "card_table": None, "id": "f00d"}
    return cat, list(DEFAULT_UNIT_TYPES), cfg(tools), fp


def test_vocab_records_the_memory_only_when_it_is_not_legacy(tools):
    rr = tools.rows
    cat, units, eng, fp = _vocab_inputs(tools)
    base = rr.vocab_record(cat, units, eng, fp, {}, False)
    assert rr.memory_vocab("legacy") == {}
    legacy = rr.vocab_record(cat, units, eng, fp, {}, False, **rr.memory_vocab("legacy"))
    assert json.dumps(legacy) == json.dumps(base)              # the vocab.json of before
    assert base["obs_delay"] == {"ticks": 0, "jitter": 0, "opp_lag": 0}
    assert "opp_memory" not in base and "units_mode" not in base
    assert rr.memory_vocab(1) == rr.memory_vocab(0) == rr.memory_vocab("1") == \
        {"opp_lag": 1, "opp_memory": "exec"}
    assert rr.memory_vocab(21) == {"opp_lag": 21, "opp_memory": "exec"}
    new = rr.vocab_record(cat, units, eng, fp, {}, False, **rr.memory_vocab(1))
    assert list(new) == list(base) + ["opp_memory"]
    assert new["obs_delay"] == {"ticks": 0, "jitter": 0, "opp_lag": 1}
    assert {k: v for k, v in new.items() if k not in ("obs_delay", "opp_memory")} == \
        {k: v for k, v in base.items() if k != "obs_delay"}


def test_vocab_lists_the_vision_classes(tools):
    rr = tools.rows
    cat, units, eng, fp = _vocab_inputs(tools)
    doc = json.loads(CLASSES.read_text(encoding="utf-8"))
    names = [c["name"] for c in doc["classes"]]
    table = rr.vision_classes.load()
    vis = rr.vocab_record(cat, units, eng, fp, {}, False, classes=table, **rr.memory_vocab(1))
    base = rr.vocab_record(cat, units, eng, fp, {}, False, **rr.memory_vocab(1))
    assert vis["n_units"] == len(names) + 2 and vis["n_cards"] == len(cat) + 2
    assert list(vis["units"]) == names
    assert list(vis["units"].values()) == list(range(2, len(names) + 2))
    digest = hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()
    assert vis["vision_classes_digest"] == digest == doc["digest"]
    assert (vis["units_mode"], vis["extra_columns"]) == ("vision", ["unit_form"])
    assert list(vis) == list(base) + ["units_mode", "vision_classes_digest", "extra_columns"]
    assert {k: v for k, v in vis.items() if k not in ("n_units", "units", "units_mode",
                                                      "vision_classes_digest", "extra_columns")} \
        == {k: v for k, v in base.items() if k not in ("n_units", "units")}
    with pytest.raises(ValueError, match="live-compat"):
        rr.vocab_record(cat, units, eng, fp, {}, True, classes=table)


# ---------------------------------------------------------------------------
# --units vision
# ---------------------------------------------------------------------------

#: The fake engine's unit types for the vision tests (unit_type = index).
VISION_UNITS = ("KingTower", "PrincessTower", "Cannon", "BabyDragon", "Skeleton_EV1",
                "MusketeerHero", "NotAUnitOfAnyTable", "HogRider")


def vision_state(tick: int = 0) -> dict:
    """Six living units straight up the middle from blue's king (9000, 3000) and a dead one.
    The engine-only flags disagree with the class table on purpose: the Cannon says troop,
    the Baby Dragon says not flying."""
    ents = [
        entity_row(uid=20, team=0, kind=0, card_id=0, x=9000 * SUB, y=5000 * SUB, hp=300,
                   max_hp=322, unit_type=2, deploy_ticks=5),
        entity_row(uid=21, team=1, kind=0, card_id=14, x=9000 * SUB, y=8000 * SUB, hp=450,
                   flying=False, unit_type=3),
        entity_row(uid=22, team=0, kind=0, card_id=6, x=9000 * SUB, y=11000 * SUB, hp=32,
                   unit_type=4),
        entity_row(uid=23, team=0, kind=0, card_id=5, x=9000 * SUB, y=14000 * SUB, hp=200,
                   max_hp=282, unit_type=5),
        entity_row(uid=24, team=1, kind=0, card_id=9, x=9000 * SUB, y=17000 * SUB, hp=100,
                   unit_type=6),
        entity_row(uid=25, team=1, kind=0, card_id=8, x=9000 * SUB, y=20000 * SUB, hp=100,
                   unit_type=-1),
        entity_row(uid=26, team=0, kind=0, card_id=2, x=9000 * SUB, y=23000 * SUB, hp=0,
                   max_hp=663, unit_type=7),
    ]
    return state_dict(tick, entities=ents)


def class_expectations():
    """unit_type -> (token, flying, building, form feature), read from classes.json itself."""
    doc = json.loads(CLASSES.read_text(encoding="utf-8"))
    out = {-1: (1, 0.0, 0.0, 0.0)}
    for u, name in enumerate(VISION_UNITS):
        hit = doc["units"].get(name)
        if hit is None:
            out[u] = (1, 0.0, 0.0, 0.0)
            continue
        cid, form = hit
        info = doc["classes"][cid]["forms"][str(form)]
        out[u] = (cid + 2, float(info["flying"]), float(info["building"]), form / 2)
    # what the table must say for this test to mean anything
    assert out[2][2] == 1.0 and out[3][1] == 1.0                     # Cannon builds, BD flies
    assert out[4][3] == 0.5 and out[5][3] == 1.0                     # evo, hero
    assert out[4][0] == doc["units"]["Skeleton_EV1"][0] + 2 and out[6][0] == 1
    return out


@pytest.fixture
def vision_royalesim(monkeypatch):
    mod = make_fake_royalesim(states=lambda battle: vision_state(battle.tick),
                              unit_types=list(VISION_UNITS))
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    return mod


def test_vision_obs_row_tokens_flags_and_forms(vision_royalesim, tools, monkeypatch):
    rr = tools.rows
    want = class_expectations()
    monkeypatch.setattr(rr, "UNITS", rr.UNITS_VISION)
    rr._setup(cfg(tools))
    assert rr.VC.unit_types == VISION_UNITS and rr.VC.digest == rr.VCT.digest
    assert rr.VC.unknown == ["NotAUnitOfAnyTable"]
    state = vision_state(300)
    order = [2, 3, 4, 5, 6, -1]                 # the unit types, nearest blue's king first
    vis = rr.obs_row(state, 0, SUB, BUSY_MEM, 300)
    typ = rr.obs_row(state, 0, SUB, BUSY_MEM, 300, units="types")
    assert list(vis) == list(typ) + ["unit_form"]
    assert vis["unit_ids"].tolist() == [want[u][0] for u in order] + [0] * 42
    assert typ["unit_ids"].tolist() == [u + 2 for u in order] + [0] * 42
    assert vis["unit_feats"][:6, 4].tolist() == [want[u][1] for u in order]
    assert vis["unit_feats"][:6, 5].tolist() == [want[u][2] for u in order]
    assert typ["unit_feats"][:6, 4].tolist() == [0.0] * 6                # the engine's flags
    assert typ["unit_feats"][:6, 5].tolist() == [0.0] * 6
    assert vis["unit_form"].dtype == np.float16 and vis["unit_form"].shape == (48,)
    assert vis["unit_form"].tolist() == [want[u][3] for u in order] + [0.0] * 42
    keep = [0, 1, 2, 3, 6]                       # own, x, y, hp share, deploying: unchanged
    assert np.array_equal(vis["unit_feats"][:, keep], typ["unit_feats"][:, keep])
    assert vis["unit_feats"][0, 6] == 1.0 and vis["unit_feats"][1, 6] == 0.0
    same_row({k: v for k, v in vis.items() if k not in ("unit_ids", "unit_feats", "unit_form")},
             {k: v for k, v in typ.items() if k not in ("unit_ids", "unit_feats")})
    red = rr.obs_row(state, 1, SUB, BUSY_MEM, 300)                       # nearest red's king first
    assert red["unit_ids"][:6].tolist() == [want[u][0] for u in reversed(order)]


def test_vision_rows_through_rows_for_and_build(vision_royalesim, tools, monkeypatch, tmp_path):
    rr = tools.rows
    want = class_expectations()
    token_of = {u + 2: want[u][0] for u in want} | {0: 0}
    form_of = {u + 2: want[u][3] for u in want} | {0: 0.0}
    payload = same_tick_payload()
    ctx = rr._setup(cfg(tools))
    typed, why = rr.rows_for(payload, ctx, cfg(tools))
    assert why == "ok" and rr.VC is None
    monkeypatch.setattr(rr, "UNITS", rr.UNITS_VISION)
    vis, why = rr.rows_for(payload, rr._setup(cfg(tools)), cfg(tools))
    assert why == "ok" and len(vis) == len(typed) == N_ROWS
    for a, b in zip(typed, vis):
        assert [k for k in b if k != "unit_form"] == list(a)
        assert b["unit_ids"].tolist() == [token_of[int(t)] for t in a["unit_ids"]]
        assert b["unit_form"].tolist() == [form_of[int(t)] for t in a["unit_ids"]]
        same_row({k: v for k, v in a.items() if k not in ("unit_ids", "unit_feats")},
                 {k: v for k, v in b.items() if k not in ("unit_ids", "unit_feats", "unit_form")})
    batch, n, skipped = rr.build(5, [json.dumps(payload)], False, str(tmp_path),
                                 cfg(tools).to_json(), units="vision")
    assert (batch, n, skipped) == (5, N_ROWS, {})
    with np.load(tmp_path / "shard-0005.npz") as z:
        assert sorted(z.files) == sorted(rr.FIELDS + rr.VISION_FIELDS
                                         + ("act_card", "act_x", "act_y", "match", "split"))
        assert z["unit_form"].shape == (N_ROWS, 48) and z["unit_form"].dtype == np.float16
        assert np.array_equal(z["unit_ids"], np.stack([r["unit_ids"] for r in vis]))


# ---------------------------------------------------------------------------
# rs_eval: the league Wilson interval
# ---------------------------------------------------------------------------

def frozen_wilson(k, n, z=1.96):
    """tools/rs_eval.wilson before L01's rewire (commit 1195892), frozen."""
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def test_eval_wilson_is_the_league_one_with_the_old_numbers(tools):
    from tools.league import stats as S
    assert tools.eval.wilson is S.wilson
    tree = ast.parse((TOOLS / "rs_eval.py").read_text(encoding="utf-8"))
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "wilson" for n in tree.body)
    assert any(isinstance(n, ast.ImportFrom) and n.module == "tools.league.stats"
               and [a.name for a in n.names] == ["wilson"] for n in tree.body)
    for n in (0, 1, 7, 100, 300, 655, 1000):
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n} if n else {0}):
            lo, hi = frozen_wilson(k, n)
            assert tools.eval.wilson(k, n) == (max(0.0, lo), min(1.0, hi)), (k, n)
            assert tools.eval.wilson(k, n) == (pytest.approx(lo, abs=1e-12),
                                               pytest.approx(hi, abs=1e-12))


# ---------------------------------------------------------------------------
# rs_eval: memory in the match loop
# ---------------------------------------------------------------------------

#: Ticks at which the stub agents keep the memory they are handed.
WATCH = (140, 160, 170, 200, 210, 230, 280, 300, 310, 330)


def stub_class(play_at=(), watch=WATCH):
    """An agent class for rs_eval.play that keeps (a copy of) the memory each act() is
    handed and plays hand slot 0 at the ticks in `play_at`. play() returns an agent's
    `records`: here that dict, tick -> memory."""
    class Stub:
        made: list = []

        def __init__(self, ckpt, seed=0, **_):
            self.records = {}
            Stub.made.append(self)

        def act(self, battle, state, team, mem, tick, sub, royalesim):
            if tick in watch:
                self.records[tick] = copy.deepcopy(mem)
            if tick in play_at:
                return (team, 0, 9000 * sub, (10_000 if team == 0 else 22_000) * sub)
            return None
    return Stub


def V(last0=(), last1=()):
    """A memory view from its "last" lists (every play also counts in "seen")."""
    return {"seen": {0: {c for c, _ in last0}, 1: {c for c, _ in last1}},
            "last": {0: list(last0), 1: list(last1)}}


#: The ghost (red) replays the Knight (token 10, it ran at 160) and the Arrows (13, ran at 300);
#: the stub agent (blue) plays slot 0, the Cannon (token 2), at 200 and 300.
KNIGHT, ARROWS = [(10, 160)], [(10, 160), (13, 300)]
EXPECTED = {
    # delay 0, before E08: the agent at 300 already sees the Arrows sent that tick
    (0, "legacy"): {140: V(), 160: V((), KNIGHT), 170: V((), KNIGHT), 200: V((), KNIGHT),
                    210: V([(2, 200)], KNIGHT), 230: V([(2, 200)], KNIGHT),
                    280: V([(2, 200)], KNIGHT), 300: V([(2, 200)], ARROWS),
                    310: V([(2, 200), (2, 300)], ARROWS), 330: V([(2, 200), (2, 300)], ARROWS)},
    # delay 0, BOARD_LAG: a play of tick t shows from t + 1
    (0, 1): {140: V(), 160: V(), 170: V((), KNIGHT), 200: V((), KNIGHT),
             210: V([(2, 200)], KNIGHT), 230: V([(2, 200)], KNIGHT), 280: V([(2, 200)], KNIGHT),
             300: V([(2, 200)], KNIGHT), 310: V([(2, 200), (2, 300)], ARROWS),
             330: V([(2, 200), (2, 300)], ARROWS)},
    # delay 21, before E08: the ghost's plays are sent at 139 and 279 and dated then
    (21, "legacy"): {140: V((), [(10, 139)]), 160: V((), [(10, 139)]), 170: V((), [(10, 139)]),
                     200: V((), [(10, 139)]), 210: V([(2, 200)], [(10, 139)]),
                     230: V([(2, 200)], [(10, 139)]),
                     280: V([(2, 200)], [(10, 139), (13, 279)]),
                     300: V([(2, 200)], [(10, 139), (13, 279)]),
                     310: V([(2, 200), (2, 300)], [(10, 139), (13, 279)]),
                     330: V([(2, 200), (2, 300)], [(10, 139), (13, 279)])},
    # delay 21, BOARD_LAG: dated when they run (ghost 160 / 300, agent 221 / 321)
    (21, 1): {140: V(), 160: V(), 170: V((), KNIGHT), 200: V((), KNIGHT), 210: V((), KNIGHT),
              230: V([(2, 221)], KNIGHT), 280: V([(2, 221)], KNIGHT), 300: V([(2, 221)], KNIGHT),
              310: V([(2, 221)], ARROWS), 330: V([(2, 221), (2, 321)], ARROWS)},
}


def sent(battle) -> list[tuple[int, list[tuple]]]:
    """(tick, commands) of every step of a FakeBattle that carried commands."""
    out, tick = [], int((battle.reset_kwargs or {}).get("start_tick") or 0)
    for call in battle.calls:
        if call.name == "step":
            cmds, n = call.args
            if cmds:
                out.append((tick, list(cmds)))
            tick += int(n)
    return out


def _play(tools, delay, *extra, opp=""):
    return tools.eval.play((json.dumps(il_payload()), "", "agent", 0, opp, delay, 0, False,
                            False, False, cfg(tools).to_json(), *extra))


@pytest.mark.parametrize("delay", [0, 21])
def test_eval_memory_legacy_and_dated_at_execution(fake_royalesim, tools, monkeypatch, delay):
    commands = {}
    for lag in ("legacy", 1):
        monkeypatch.setattr(tools.eval, "AGENT_CLASS", stub_class(play_at=(200, 300)))
        out = _play(tools, delay, lag)
        assert out["hog_team"] == 0 and out["agent_plays"] == 2
        assert out["records"] == EXPECTED[(delay, lag)], (delay, lag)
        commands[lag] = sent(fake_royalesim.battles[-1])
    # the memory changes what a seat knows, never what the ghost sends
    assert commands["legacy"] == commands[1]
    assert [t for t, _ in commands[1]] == ([160, 200, 300] if delay == 0 else [139, 200, 279, 300])
    # no mem_lag in the task (rs_selfplay's tuples): BOARD_LAG
    monkeypatch.setattr(tools.eval, "AGENT_CLASS", stub_class(play_at=(200, 300)))
    assert _play(tools, delay)["records"] == EXPECTED[(delay, 1)]
    monkeypatch.setattr(tools.eval, "AGENT_CLASS", stub_class(play_at=(200, 300)))
    assert _play(tools, delay, None)["records"] == EXPECTED[(delay, 1)]


def test_eval_perception_lag(fake_royalesim, tools, monkeypatch):
    monkeypatch.setattr(tools.eval, "AGENT_CLASS", stub_class(play_at=(200, 300)))
    seen = _play(tools, 0, 21)["records"]
    assert seen[170] == V() and seen[200] == V((), KNIGHT)      # 160 + 21 > 170, <= 200
    assert seen[310] == V([(2, 200)], KNIGHT) and seen[330] == V([(2, 200), (2, 300)], ARROWS)


def test_eval_reactive_opponent_no_longer_sees_the_agents_same_tick_play(fake_royalesim, tools,
                                                                         monkeypatch):
    """Both seats play slot 0 at 200 (blue the Cannon, token 2; red the Knight, 10)."""
    views = {}
    for lag in ("legacy", 1):
        agent, opponent = stub_class(play_at=(200,)), stub_class(play_at=(200,))
        monkeypatch.setattr(tools.eval, "AGENT_CLASS", agent)
        monkeypatch.setattr(tools.eval, "RSAgent", opponent)
        out = _play(tools, 0, lag, opp="opp.pt")
        assert out["agent_plays"] == 1 and len(opponent.made) == 1
        views[lag] = out["records"], opponent.made[0].records
        assert sent(fake_royalesim.battles[-1])[0][0] == 200        # no ghost: the first send
    both = V([(2, 200)], [(10, 200)])
    (a_old, o_old), (a_new, o_new) = views["legacy"], views[1]
    assert (a_old[200], a_old[210]) == (V(), both)
    assert (o_old[200], o_old[210]) == (V([(2, 200)]), both)          # the leak, as before
    assert (a_new[200], a_new[210]) == (V(), both)
    assert (o_new[200], o_new[210]) == (V(), both)


def test_rsagent_refuses_a_vision_units_checkpoint(tools, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    rc = importlib.import_module("rs_checkpoint")
    meta = {}

    def load_policy(ckpt, engine=None, allow_drift=None):
        return object(), dict(meta), []

    monkeypatch.setattr(rc, "load_policy", load_policy)
    ckpt = str(tmp_path / "best.pt")
    try:
        meta.update(units_mode="vision", live_compat=False)
        with pytest.raises(ValueError, match="--units vision checkpoint"):
            tools.eval.RSAgent(ckpt, engine=cfg(tools))
        meta.clear()
        meta.update(live_compat=True)
        agent = tools.eval.RSAgent(ckpt, engine=cfg(tools))
        assert agent.live is True and agent.meta == {"live_compat": True}
        meta.update(units_mode="types")
        assert tools.eval.RSAgent(ckpt, engine=cfg(tools)).live is True
    finally:
        torch.set_num_threads(threads)
