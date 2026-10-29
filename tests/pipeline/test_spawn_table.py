"""E06: tools/rs_spawn_table.py (the card -> unit-type table and unit hitpoints) and the
regenerate-on-mismatch hook in tools/rs_eval.py _spawn_tables.

Pure tests run a spawning FakeBattle; `engine`-marked ones build the real table under
data/engine.json's engine. Unit-type counts and ids of the real engine are never
hard-coded: they differ between card tables and installs.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    Call,
    FakeBattle,
    catalogue_row,
    entity_row,
    make_fake_royalesim,
    state_dict,
)

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = 18

#: The conftest catalogue (ids 0-15) plus a troop that puts nothing down and names no unit
#: type (16), and a building the engine refuses to spawn (17).
CATALOGUE = [list(r) for r in DEFAULT_CATALOGUE] + [catalogue_row("Mystery", 3, "TROOP"),
                                                    catalogue_row("Broken", 3, "BUILDING")]
UNITS = list(DEFAULT_UNIT_TYPES)
#: What the fake puts on the board for a spawn of each card (unit names, one per entity).
UNIT_OF = {
    "Cannon": ["Cannon"], "HogRider": ["HogRider"], "IceGolemite": ["IceGolemite"],
    "IceSpirits": ["IceSpirit"], "Musketeer": ["Musketeer"], "Skeletons": ["Skeleton"] * 3,
    "Knight": ["Knight"], "Archer": ["Archer", "Archer"], "Giant": ["Giant", "Skeleton", "Skeleton"],
    "Valkyrie": ["Valkyrie"], "MiniPekka": ["MiniPekka"], "BabyDragon": [], "Mystery": [],
}
#: The unit each card's unit_hitpoints "own" row names (default: the card's name).
OWN = {"IceSpirits": "IceSpirit", "Skeletons": "Skeleton", "Archer": "ArcherPair",
       "MiniPekka": "Knight", "Mystery": "MysteryUnit"}
#: Cards whose units appear only one tick after the reset.
DELAYED = {"Valkyrie"}
#: (lowest, highest) level of a card's ladder (default 1-16).
LADDER = {"Musketeer": (3, 16), "BabyDragon": (6, 16)}
HOG26_IDS = [2, 5, 0, 3, 4, 6, 1, 7]          # HogRider Musketeer Cannon IceGolemite ...

#: card id -> unit type, what the fake must give.
EXPECTED_PRIMARY = {"0": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "8": 8, "9": 9, "10": 10,
                    "12": 11, "13": 12, "14": 13}
EXPECTED_ALL = {"0": [2], "2": [3], "3": [4], "4": [5], "5": [6], "6": [7], "8": [8], "9": [9],
                "10": [7, 10], "12": [11], "13": [12], "14": [], "16": []}


class SpawningBattle(FakeBattle):
    """A FakeBattle whose reset puts each spawn's units (UNIT_OF) on the board, with
    unit_hitpoints rows; a red Knight is always on the board too."""

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        for spawn in kwargs.get("spawns") or []:
            if self.catalogue[spawn[1]][0] == "Broken":
                raise ValueError("spawn 0: Broken cannot be spawned here")

    def state(self):
        ents = [entity_row(uid=99, team=1, card_id=8, x=9000 * SUB, y=22000 * SUB, unit_type=8)]
        uid = 100
        for team, cid, x, y, _hp in (self.reset_kwargs or {}).get("spawns") or []:
            name = self.catalogue[cid][0]
            if name in DELAYED and self.steps == 0:
                continue
            for unit in UNIT_OF.get(name, []):
                ents.append(entity_row(uid=uid, team=team, card_id=cid, x=x, y=y,
                                       unit_type=self.unit_types.index(unit)))
                uid += 1
        return state_dict(self.tick, entities=ents)

    def unit_hitpoints(self, card_id, level):
        name = self.catalogue[card_id][0]
        if name == "Broken":
            raise RuntimeError("no card data")
        low, high = LADDER.get(name, (1, 16))
        if not low <= level <= high:
            raise ValueError(f"{name} has no level {level}")
        if self.catalogue[card_id][8] == "SPELL":
            return [("own", name, 0)]
        rows = [("own", OWN.get(name, name), 100 * level)]
        if name == "Giant":
            rows.append(("death_spawn", "Skeleton", 10 * level))
        return rows


def spawning_royalesim() -> types.ModuleType:
    mod = make_fake_royalesim(catalogue=CATALOGUE, unit_types=UNITS)

    def Battle(*args, **kwargs):
        battle = SpawningBattle(*args, **{**mod.battle_kwargs, **kwargs})
        mod.battles.append(battle)
        return battle

    mod.Battle = Battle
    return mod


@pytest.fixture
def st(monkeypatch):
    """tools/rs_spawn_table.py, imported as the tools import each other."""
    monkeypatch.syspath_prepend(str(TOOLS))
    module = importlib.import_module("rs_spawn_table")
    rs_engine = module.rs_engine
    for var in (rs_engine.CONFIG_ENV, *rs_engine.ENV.values()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rs_engine, "_ACTIVE", None)
    return module


@pytest.fixture
def fake_rs(monkeypatch):
    mod = spawning_royalesim()
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    return mod


def defaults(st):
    return st.rs_engine.EngineConfig.defaults()


# ---------------------------------------------------------------------------
# pure rules
# ---------------------------------------------------------------------------

def test_battles_come_only_from_rs_engine():
    import ast
    tree = ast.parse((TOOLS / "rs_spawn_table.py").read_text(encoding="utf-8"))
    attrs = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "Battle" not in attrs and "set_command_delay_ticks" not in attrs
    assert "make_battle" in attrs


def test_by_unit_takes_the_lowest_card_for_a_shared_unit(st):
    prim = {9: 1, 3: 1, 5: 2, 1: 2, 7: 0}
    old = {}                                    # rs_eval._spawn_tables before E06
    for card, unit in sorted(prim.items()):
        old.setdefault(unit, card)
    assert st.by_unit(prim) == old == {1: 3, 2: 1, 0: 7}
    assert st.by_unit({"6": 7, "2": 7, "10": 3}) == {7: 2, 3: 10}       # file keys are text


@pytest.mark.parametrize("observed, own, expected", [
    ([7, 7, 7], 7, 7),
    ([10, 7, 7], 10, 10),           # the own row's unit wins when it was put down
    ([10, 7, 7], None, 7),          # else the most common
    ([9, 7], None, 7),              # a tie goes to the lowest id
    ([12], 8, 12),                  # an own row naming a unit not put down
    ([], 13, 13),                   # nothing on the board: the own row
    ([], None, None),
])
def test_pick_primary(st, observed, own, expected):
    assert st.pick_primary(observed, own) == expected


def test_card_kinds_and_filler_deck(st):
    assert st.spawn_cards(CATALOGUE) == [0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17]
    assert st.card_kind(["Old", 3, 0, 0, 0, False, 0, 0]) is None        # no card_kind column
    assert st.filler_deck(CATALOGUE) == HOG26_IDS
    no_log = [r for r in CATALOGUE if r[0] != "Log"]
    assert st.filler_deck(no_log) == list(range(8))
    with pytest.raises(ValueError):
        st.filler_deck(CATALOGUE[:5])
    assert st.spawn_point(SUB) == (9000 * SUB, 10000 * SUB)


def test_own_unit_name_prefers_the_level_asked(st):
    hp = {"3": [["own", "A", 1]], "11": [["death_spawn", "B", 2], ["own", "C", 3]]}
    assert st.own_unit_name(hp, 11) == "C"
    assert st.own_unit_name(hp, 4) == "A"
    assert st.own_unit_name({}, 11) is None


# ---------------------------------------------------------------------------
# building the table on the fake engine
# ---------------------------------------------------------------------------

def test_build_on_the_fake_engine(st, fake_rs):
    cfg = defaults(st)
    table = st.build(cfg)
    assert table["format"] == st.FORMAT
    assert table["primary"] == EXPECTED_PRIMARY
    assert table["all"] == EXPECTED_ALL
    for spell in ("1", "7", "11", "15"):                  # Fireball, Log, Arrows, Zap
        assert spell not in table["primary"] and spell not in table["all"]
        assert spell not in table["unresolved"]
    assert set(table["unresolved"]) == {"16", "17"}
    assert table["unresolved"]["16"] == ("the spawn put no unit down and unit_hitpoints names "
                                         "'MysteryUnit', not a unit type")
    assert table["unresolved"]["17"].startswith("spawn refused: ValueError: spawn 0: Broken")
    notes = table["notes"]
    assert any(n.startswith("MiniPekka (13): the own row names Knight") for n in notes)
    assert any(n.startswith("BabyDragon (14): the spawn put no unit down") for n in notes)
    assert any(n.startswith("Broken (17): unit_hitpoints failed: RuntimeError") for n in notes)
    assert len(notes) == 3

    hp = table["hp"]
    assert set(hp) == {str(i) for i in range(17)}         # every card but Broken, spells too
    assert set(hp["2"]) == {str(lv) for lv in range(1, 17)}
    assert set(hp["5"]) == {str(lv) for lv in range(3, 17)}      # Musketeer's ladder: 3-16
    assert set(hp["14"]) == {str(lv) for lv in range(6, 17)}
    assert hp["5"]["11"] == [["own", "Musketeer", 1100]]
    assert hp["10"]["2"] == [["own", "Giant", 200], ["death_spawn", "Skeleton", 20]]
    assert hp["1"]["11"] == [["own", "Fireball", 0]]

    assert table["card_names"] == [r[0] for r in CATALOGUE] and table["unit_types"] == UNITS
    assert table["engine_config"] == cfg.undelayed().to_json()
    assert table["spawn"] == {"team": 0, "point_milli": [9000, 10000], "deck": HOG26_IDS,
                              "settle_ticks": 1, "levels": [1, 16], "own_row_level": 11}
    assert table["fingerprint"]["id"] == st.rs_engine.fingerprint(
        fake_rs.battles[0], fake_rs, cfg.undelayed())["id"]
    assert json.loads(json.dumps(table)) == table         # JSON as it is

    # one battle, undelayed, reset once per troop / building; a clean one after the refusal
    assert len(fake_rs.battles) == 2
    probe = fake_rs.battles[0]
    assert probe.init_kwargs == cfg.constructor_kwargs()
    assert probe.names()[0] == "set_command_delay_ticks" and probe.calls[0].args == (0, 0)
    resets = probe.calls_named("reset")
    assert len(resets) == 14
    assert resets[0].kwargs == {"seed": 0, "decks": [HOG26_IDS, HOG26_IDS], "shuffle": 0,
                                "start_tick": 0, "elixir_milli": [5000, 5000], "tower_hp": None,
                                "spawns": [(0, 0, 9000 * SUB, 10000 * SUB, -1)]}
    assert [c.kwargs["spawns"][0][1] for c in resets] == [0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13,
                                                          14, 16, 17]
    # one settle tick each for Valkyrie (appears then), BabyDragon and Mystery (never)
    assert probe.calls_named("step") == [Call("step", ([], 1), {})] * 3


def test_build_limited_to_some_cards(st, fake_rs):
    table = st.build(defaults(st), cards=[2, 1])          # HogRider; Fireball is no troop
    assert table["primary"] == {"2": 3} and table["all"] == {"2": [3]}
    assert table["unresolved"] == {}
    assert len(table["hp"]) == 17                         # hitpoints for every card anyway
    assert len(fake_rs.battles[0].calls_named("reset")) == 1


def test_build_uses_the_active_config(st, fake_rs):
    rs_engine = st.rs_engine
    rs_engine.set_active(rs_engine.EngineConfig(card_table="client15535", delay=(21, 21)))
    table = st.build()
    assert table["engine_config"]["card_table"] == "client15535"
    assert table["engine_config"]["delay"] == [0, 0]       # never a command delay
    assert fake_rs.battles[0].calls[0].args == (0, 0)


def test_build_without_unit_hitpoints(st, monkeypatch):
    class Plain(SpawningBattle):
        unit_hitpoints = None
    mod = make_fake_royalesim(catalogue=CATALOGUE, unit_types=UNITS)
    mod.Battle = lambda *a, **k: Plain(*a, **{**mod.battle_kwargs, **k})
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    table = st.build(defaults(st))
    assert table["hp"] == {}
    assert any("has no Battle.unit_hitpoints" in n for n in table["notes"])
    # from the spawns alone: Giant's Skeletons outnumber it, BabyDragon resolves to nothing
    assert table["primary"]["10"] == 7
    assert "14" in table["unresolved"] and table["primary"]["13"] == 12


# ---------------------------------------------------------------------------
# the file: ensure, write, read
# ---------------------------------------------------------------------------

def test_ensure_writes_a_missing_file(st, fake_rs, tmp_path):
    path = tmp_path / "rs_rows" / "spawn_primary.json"
    logs = []
    table = st.ensure(path, defaults(st), log=logs.append)
    assert table["primary"] == EXPECTED_PRIMARY
    assert json.loads(path.read_text(encoding="utf-8")) == table
    assert len(logs) == 1 and "was missing; regenerated" in logs[0]
    assert list(path.parent.glob("*.tmp")) == []


def test_ensure_keeps_a_current_file(st, fake_rs, tmp_path):
    path = tmp_path / "spawn_primary.json"
    first = st.ensure(path, defaults(st), log=lambda m: None)
    logs = []
    again = st.ensure(path, defaults(st), log=logs.append)
    assert again == first and logs == []
    assert fake_rs.battles[-1].calls_named("reset") == []      # only the fingerprint battle


def test_ensure_regenerates_on_a_fingerprint_mismatch(st, fake_rs, tmp_path):
    path = tmp_path / "spawn_primary.json"
    table = st.ensure(path, defaults(st), log=lambda m: None)
    stale = dict(table, primary={"2": 9},
                 fingerprint=dict(table["fingerprint"], id="0000000000000000"))
    st.write(stale, path)
    logs = []
    fresh = st.ensure(path, defaults(st), log=logs.append)
    assert fresh["primary"]["2"] == 3 and fresh["fingerprint"]["id"] == table["fingerprint"]["id"]
    assert st.read(path) == fresh
    assert "made under engine 0000000000000000; regenerated" in logs[0]

    # another card table is another engine: regenerated for it
    other = st.rs_engine.EngineConfig(card_table="client15535", level=11, tower_level=11)
    swapped = st.ensure(path, other, log=logs.append)
    assert swapped["fingerprint"]["id"] != fresh["fingerprint"]["id"]
    assert st.read(path)["engine_config"]["card_table"] == "client15535"
    assert f"made under engine {fresh['fingerprint']['id']}" in logs[-1]


@pytest.mark.parametrize("content, why", [
    ("not json", "was unreadable"),
    (json.dumps({"format": 0, "primary": {"2": 3}}), "was format 0, not 1"),
    (json.dumps({"format": 1, "primary": {"2": 3}}), "made under engine unknown (no fingerprint)"),
    (json.dumps([1, 2]), "was unreadable"),
])
def test_ensure_regenerates_an_unusable_file(st, fake_rs, tmp_path, content, why):
    path = tmp_path / "spawn_primary.json"
    path.write_text(content, encoding="utf-8")
    logs = []
    table = st.ensure(path, defaults(st), log=logs.append)
    assert table["primary"] == EXPECTED_PRIMARY and st.read(path) == table
    assert why in logs[0]


def test_ensure_without_royalesim(st, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "royalesim", None)       # import royalesim fails
    path = tmp_path / "spawn_primary.json"
    with pytest.raises(st.rs_engine.EngineError, match="cannot be generated"):
        st.ensure(path, defaults(st), log=lambda m: None)
    old = {"primary": {"2": 3}}                    # e.g. a file made before E06
    path.write_text(json.dumps(old), encoding="utf-8")
    logs = []
    assert st.ensure(path, defaults(st), log=logs.append) == old
    assert "without checking its engine" in logs[0]


def test_ensure_keeps_the_table_when_it_cannot_write(st, fake_rs, tmp_path, monkeypatch):
    def refuse(table, path, attempts=5):
        raise PermissionError("in use")
    monkeypatch.setattr(st, "write", refuse)
    logs = []
    table = st.ensure(tmp_path / "spawn_primary.json", defaults(st), log=logs.append)
    assert table["primary"] == EXPECTED_PRIMARY
    assert "not written (PermissionError" in logs[0]


def test_write_retries_a_refused_replace(st, tmp_path, monkeypatch):
    real_replace = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError("another process has it open")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(st.time, "sleep", lambda s: None)
    path = tmp_path / "t.json"
    st.write({"format": 1, "primary": {}}, path)
    assert len(calls) == 3 and st.read(path) == {"format": 1, "primary": {}}

    def never(src, dst):
        raise PermissionError("still open")

    monkeypatch.setattr(os, "replace", never)
    with pytest.raises(PermissionError):
        st.write({"format": 1}, tmp_path / "u.json", attempts=2)
    assert list(tmp_path.glob("*.tmp")) == [] and not (tmp_path / "u.json").exists()


# ---------------------------------------------------------------------------
# rs_eval._spawn_tables
# ---------------------------------------------------------------------------

@pytest.fixture
def rs_eval(st, monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    module = importlib.import_module("rs_eval")
    monkeypatch.setattr(module, "SPAWN_TABLE", tmp_path / "spawn_primary.json")
    monkeypatch.setattr(module, "_SPAWN", {})
    return module


def test_rs_eval_spawn_tables_generate_a_missing_file(st, fake_rs, rs_eval, capsys):
    st.rs_engine.set_active(defaults(st))
    prim, by_unit = rs_eval._spawn_tables()
    assert prim == {int(k): v for k, v in EXPECTED_PRIMARY.items()}
    assert by_unit == st.by_unit(prim) and by_unit[7] == 6          # Skeleton -> Skeletons
    assert st.read(rs_eval.SPAWN_TABLE)["primary"] == EXPECTED_PRIMARY
    assert "spawn table:" in capsys.readouterr().err
    assert rs_eval._spawn_tables()[0] is prim                        # cached for the process


def test_rs_eval_spawn_tables_read_a_current_file_as_is(st, fake_rs, rs_eval, monkeypatch):
    cfg = defaults(st)
    st.rs_engine.set_active(cfg)
    table = st.ensure(rs_eval.SPAWN_TABLE, cfg, log=lambda m: None)
    st.write(dict(table, primary={"2": 9, "6": 7, "10": 7}), rs_eval.SPAWN_TABLE)
    prim, by_unit = rs_eval._spawn_tables()
    assert prim == {2: 9, 6: 7, 10: 7} and by_unit == {9: 2, 7: 6}

    # the same file under another engine is regenerated, not trusted
    monkeypatch.setattr(rs_eval, "_SPAWN", {})
    st.rs_engine.set_active(st.rs_engine.EngineConfig(card_table="client15535"))
    prim, _ = rs_eval._spawn_tables()
    assert prim[2] == 3 and prim == {int(k): v for k, v in EXPECTED_PRIMARY.items()}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def test_cli_writes_and_checks(st, fake_rs, tmp_path, capsys):
    out = tmp_path / "spawn_primary.json"
    assert st.main(["--out", str(out), "--check"]) == 1
    assert "missing" in capsys.readouterr().out
    assert st.main(["--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "12 of 14 troop / building cards resolved (2 unresolved)" in text
    assert "unresolved Mystery (16)" in text
    assert st.read(out)["engine_config"]["card_table"] == st.rs_engine.TARGET_TABLE
    assert st.main(["--out", str(out), "--check"]) == 0
    assert st.main(["--out", str(out), "--check", "--card-table", "client15535"]) == 1
    assert st.main(["--out", str(out), "--engine-config", str(tmp_path / "none.json")]) == 2


def test_cli_without_royalesim(st, monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "royalesim", None)
    assert st.main(["--out", str(tmp_path / "x.json")]) == 1
    assert "does not import" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

HOG26_SPAWNED = ("Cannon", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons")


def _import_tool(name: str):
    """tools/<name>.py as the tools import it, without leaving tools/ on sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    saved = list(sys.path)
    sys.path.insert(0, str(TOOLS))
    try:
        return importlib.import_module(name)
    finally:
        sys.path[:] = saved


@pytest.fixture(scope="module")
def real_table():
    """(rs_spawn_table, data/engine.json's config, the table built under it)."""
    royalesim = pytest.importorskip("royalesim")
    st = _import_tool("rs_spawn_table")
    cfg = st.rs_engine.EngineConfig.load(env={})
    try:
        return st, cfg, st.build(cfg, royalesim)
    except st.rs_engine.EngineError as exc:         # royalesim before card tables
        pytest.skip(f"the engine refused data/engine.json's config: {exc}")


@pytest.mark.engine
def test_engine_hog26_troops_and_buildings_resolve(real_table):
    st, cfg, table = real_table
    ids = {name: i for i, name in enumerate(table["card_names"])}
    for name in HOG26_SPAWNED:
        key = str(ids[name])
        assert key in table["primary"], (name, table["unresolved"].get(key))
        assert table["all"][key], f"{name}: a spawn put nothing on the board"
    hog = table["primary"][str(ids["HogRider"])]
    assert table["unit_types"][hog] == "HogRider"
    for spell in ("Fireball", "Log"):
        key = str(ids[spell])
        assert key not in table["primary"] and key not in table["all"]
        assert key not in table["unresolved"]


@pytest.mark.engine
def test_engine_table_is_consistent(real_table):
    st, cfg, table = real_table
    units, names = table["unit_types"], table["card_names"]
    assert units and names
    for key, unit in table["primary"].items():
        assert 0 <= unit < len(units)
        assert not table["all"][key] or unit in table["all"][key], names[int(key)]
    import royalesim
    battle = st.rs_engine.make_battle(cfg.undelayed(), royalesim)
    cat = json.loads(battle.catalogue_json())
    assert [r[0] for r in cat] == names
    assert {int(k) for k in table["primary"]} | {int(k) for k in table["unresolved"]} == \
        set(st.spawn_cards(cat))
    assert table["fingerprint"]["id"] == st.rs_engine.fingerprint(battle, royalesim, cfg.undelayed())["id"]


@pytest.mark.engine
def test_engine_hitpoints_match_the_catalogue(real_table):
    st, cfg, table = real_table
    if not table["hp"]:
        pytest.skip("this royalesim has no Battle.unit_hitpoints")
    import royalesim
    cat = json.loads(st.rs_engine.make_battle(cfg.undelayed(), royalesim).catalogue_json())
    hog = table["card_names"].index("HogRider")
    levels = table["hp"][str(hog)]
    assert "11" in levels and set(levels) <= {str(lv) for lv in range(1, 17)}
    role, unit, hp = levels["11"][0]
    assert role == "own" and unit == "HogRider"
    assert hp == cat[hog][6]               # the catalogue's hitpoints at level 11


@pytest.mark.engine
def test_engine_stale_file_is_regenerated(real_table, tmp_path):
    st, cfg, table = real_table
    path = tmp_path / "spawn_primary.json"
    st.write(dict(table, fingerprint=dict(table["fingerprint"], id="0" * 16)), path)
    logs = []
    fresh = st.ensure(path, cfg, log=logs.append)
    assert fresh["fingerprint"]["id"] == table["fingerprint"]["id"]
    assert st.read(path)["fingerprint"]["id"] == table["fingerprint"]["id"]
    assert fresh["primary"] == table["primary"]
    assert f"made under engine {'0' * 16}" in logs[0]
    assert st.ensure(path, cfg, log=logs.append) == fresh and len(logs) == 1
