"""E01: tools/rs_engine.py, the one RoyaleSim battle factory, and the tools routed through it.

Pure tests use the fake royalesim from conftest, with royaleimitate made unimportable
(the tools read the card map from data/cards through tools/rs_cards.py, E02);
`engine`-marked ones need the real royalesim.
Unit-type counts of the real engine are never hard-coded: they differ between card
tables and installs (237 on the default table on the PC).
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import pickle
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    ENTITY_FIELDS,
    FIELD,
    FakeBattle,
    il_payload,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def _load_rs_engine():
    """tools/rs_engine.py as the module `rs_engine`, the name the tools import it by,
    without putting tools/ on sys.path for the whole session."""
    if "rs_engine" in sys.modules:
        return sys.modules["rs_engine"]
    spec = importlib.util.spec_from_file_location("rs_engine", TOOLS / "rs_engine.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rs_engine"] = module
    spec.loader.exec_module(module)
    return module


rs_engine = _load_rs_engine()
EngineConfig = rs_engine.EngineConfig
TARGET = "client160402017_20261019"
LEGACY_KWARGS = {"card_names": None, "slot_of_k": [[0, 1, 2], [0, 1, 2]]}
TARGET_OVERRIDE = {"cards.CARD_TABLE": '"client160402017_20261019"'}

#: IL_Replay key -> RoyaleSim name for the conftest payload's two decks.
NAMES = {"cannon": "Cannon", "fireball": "Fireball", "hog-rider": "HogRider",
         "ice-golem": "IceGolemite", "ice-spirit": "IceSpirits", "musketeer": "Musketeer",
         "skeletons": "Skeletons", "the-log": "Log", "knight": "Knight", "archers": "Archer",
         "giant": "Giant", "arrows": "Arrows", "valkyrie": "Valkyrie",
         "mini-pekka": "MiniPekka", "baby-dragon": "BabyDragon", "zap": "Zap"}


# ---------------------------------------------------------------------------
# helpers and fixtures of this module
# ---------------------------------------------------------------------------

class ReportingBattle(FakeBattle):
    """A FakeBattle with royalesim 0.1.42's read-outs, answered from how it was built."""

    default_arm = "client15535"

    def _arm(self):
        over = self.init_kwargs.get("calibration_overrides") or {}
        key = rs_engine.CARD_TABLE_KEY
        return json.loads(over[key]) if key in over else self.default_arm

    def card_table(self):
        return rs_engine.KNOWN_TABLES.get(self._arm(), "cards-unknown.1")

    def card_table_fnv1a64(self):
        return rs_engine.fnv1a64(self.card_table().encode("utf-8"))

    def unit_types_digest(self):
        return rs_engine.fnv1a64(self.unit_types_json().encode("utf-8"))

    def calibration_overrides(self):
        return dict(self.init_kwargs.get("calibration_overrides") or {})

    def card_level(self):
        return self.init_kwargs.get("level") or 11

    def tower_level(self):
        return self.init_kwargs.get("tower_level") or self.card_level()

    def command_delay_ticks(self):
        return self.delay


class IgnoringBattle(ReportingBattle):
    """Takes the override and runs the default table anyway."""

    def card_table(self):
        return "cards-15535.1"


class NewRevisionBattle(ReportingBattle):
    def card_table(self):
        return super().card_table().replace(".1", ".2")


class IntHashBattle(ReportingBattle):
    def card_table_fnv1a64(self):
        return 0x1234


def royalesim_with(battle_cls=ReportingBattle, **battle_kwargs) -> types.ModuleType:
    """A conftest fake royalesim whose Battle builds `battle_cls`, with provenance()."""
    mod = make_fake_royalesim(**battle_kwargs)

    def Battle(*args, **kwargs):
        battle = battle_cls(*args, **{**mod.battle_kwargs, **kwargs})
        mod.battles.append(battle)
        return battle

    mod.Battle = Battle
    mod.provenance = lambda: ("abc1234", "clean")
    return mod


def refusing_royalesim(always: bool = False) -> types.ModuleType:
    """A fake royalesim whose Battle refuses calibration_overrides (as 0.1.22 refuses
    cards.CARD_TABLE), or everything."""
    mod = make_fake_royalesim()

    def Battle(*args, **kwargs):
        if always or "calibration_overrides" in kwargs:
            raise ValueError("cards.CARD_TABLE is not a key in the ledger, and an override "
                             "cannot add one")
        battle = FakeBattle(*args, **kwargs)
        mod.battles.append(battle)
        return battle

    mod.Battle = Battle
    return mod


@pytest.fixture(autouse=True)
def clean_engine(monkeypatch):
    """No HASTYCR_* engine variable from the PC, and no active config left from a test."""
    for var in (rs_engine.CONFIG_ENV, *rs_engine.ENV.values()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rs_engine, "_ACTIVE", None)


@pytest.fixture
def fake_imitate(monkeypatch):
    """No royaleimitate: `import royaleimitate` fails. The tools read the card map from
    data/cards (tools/rs_cards.py, E02), which maps NAMES' keys to NAMES' names."""
    monkeypatch.setitem(sys.modules, "royaleimitate", None)


@pytest.fixture
def tools(monkeypatch):
    """The routed tools, imported as they import each other, with their caches empty."""
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(TOOLS))
    rs_rows = importlib.import_module("rs_rows")
    redrive = importlib.import_module("royalesim_redrive")
    rs_eval = importlib.import_module("rs_eval")
    rs_shadow = importlib.import_module("rs_shadow")
    assert rs_rows.rs_engine is rs_engine and rs_eval.rs_engine is rs_engine
    monkeypatch.setattr(rs_rows, "F", None)
    monkeypatch.setattr(redrive, "_ENGINE", {})
    monkeypatch.setattr(rs_eval, "_UNITS", [])
    monkeypatch.setattr(rs_eval, "_SPAWN", {})
    monkeypatch.setattr(rs_eval.RSAgent, "_CAT", None)
    return types.SimpleNamespace(rs_rows=rs_rows, redrive=redrive, rs_eval=rs_eval,
                                 rs_shadow=rs_shadow)


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def test_shipped_engine_json_targets_the_16_402_22_table():
    cfg = EngineConfig.load(env={})
    assert cfg.card_table == TARGET == rs_engine.TARGET_TABLE
    assert (cfg.level, cfg.tower_level, cfg.delay, dict(cfg.calibration_overrides)) == \
        (11, 11, (0, 0), {})
    assert Path(cfg.sources["card_table"]).name == "engine.json"
    # data/engine.json and the built-in fallback say the same thing
    assert EngineConfig.from_json(rs_engine.read_config_file(rs_engine.CONFIG_PATH)) == \
        EngineConfig.defaults()


def test_legacy_config_is_the_old_constructor_call():
    legacy = EngineConfig.legacy()
    assert legacy == EngineConfig()
    assert legacy.is_legacy and not EngineConfig.defaults().is_legacy
    assert legacy.constructor_kwargs() == LEGACY_KWARGS
    assert legacy.encoded_overrides() == {}


def test_default_constructor_kwargs_carry_table_and_levels():
    assert EngineConfig.defaults().constructor_kwargs() == {
        **LEGACY_KWARGS, "calibration_overrides": TARGET_OVERRIDE, "level": 11, "tower_level": 11}
    # the override value is the arm as JSON text, as the ledger documents it
    assert TARGET_OVERRIDE["cards.CARD_TABLE"] == json.dumps(TARGET)


@pytest.mark.parametrize("given, arm", [
    ("client15535", "client15535"),
    (TARGET, TARGET),
    ("16.402.22", TARGET),
    ("16.402.19", "client160402017_20261006"),
    ("15.535.29", "client15535"),
    ("cards-160402017-20261019.1", TARGET),
    ("cards-15535.1", "client15535"),
    ('"client15535"', "client15535"),
    ("  client15535 ", "client15535"),
    ("client170000000_20270101", "client170000000_20270101"),   # unknown: passed through
    ("default", None), ("Default", None), ("none", None), ("null", None), ("legacy", None),
    ("", None), (None, None),
])
def test_table_names(given, arm):
    assert rs_engine.table_name(given) == arm
    assert EngineConfig(card_table=given).card_table == arm


def test_table_version_stems_match_the_known_tables():
    for arm, version in rs_engine.KNOWN_TABLES.items():
        assert version.rpartition(".")[0] == rs_engine.table_version_stem(arm)
        assert rs_engine.table_name(version) == arm
    assert rs_engine.table_version_stem("weird") is None
    assert rs_engine.table_version_stem("client") is None


@pytest.mark.parametrize("given, delay", [
    (None, None), ("default", None), (21, (21, 21)), ("21", (21, 21)), ("21,22", (21, 22)),
    (" 21 , 22 ", (21, 22)), ("21/22", (21, 22)), ([21, 22], (21, 22)), ((0, 0), (0, 0)),
    ([5], (5, 5)), (0, (0, 0)),
])
def test_parse_delay(given, delay):
    assert rs_engine.parse_delay(given) == delay


@pytest.mark.parametrize("bad", [-1, "-1", "a", [1, 2, 3], True, {"blue": 1}])
def test_parse_delay_refuses(bad):
    with pytest.raises(ValueError, match="delay"):
        rs_engine.parse_delay(bad)


@pytest.mark.parametrize("given, level", [(11, 11), ("11", 11), (16, 16), (1, 1), (11.0, 11),
                                          ("default", None), (None, None)])
def test_parse_level(given, level):
    assert rs_engine.parse_level(given) == level


@pytest.mark.parametrize("bad", [0, 17, "x", True, 11.5, "11.5"])
def test_parse_level_refuses(bad):
    with pytest.raises(ValueError, match="level"):
        rs_engine.parse_level(bad)


def test_overrides_are_sorted_and_checked():
    got = rs_engine.parse_overrides('{"z.B": 1, "a.A": "x", "_comment": "ignored"}')
    assert list(got.items()) == [("a.A", "x"), ("z.B", 1)]
    assert rs_engine.parse_overrides(None) == {} and rs_engine.parse_overrides(" ") == {}
    with pytest.raises(ValueError, match="card_table"):
        rs_engine.parse_overrides({"cards.CARD_TABLE": "client15535"})
    with pytest.raises(ValueError, match="section.KEY"):
        rs_engine.parse_overrides({"nodot": 1})
    with pytest.raises(ValueError, match="JSON"):
        rs_engine.parse_overrides("{not json")
    with pytest.raises(ValueError, match="object"):
        rs_engine.parse_overrides([["a.A", 1]])
    cfg = EngineConfig(calibration_overrides={"placement.TAP_SNAP": "none"}, card_table="15.535")
    assert cfg.encoded_overrides() == {"cards.CARD_TABLE": '"client15535"',
                                       "placement.TAP_SNAP": '"none"'}


def _write(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_precedence(tmp_path):
    file = _write(tmp_path / "engine.json", {"_note": "x", "card_table": "client15535",
                                             "level": 12, "delay": 21})
    env = {"HASTYCR_ENGINE_CONFIG": str(file), "HASTYCR_CARD_LEVEL": "13",
           "HASTYCR_COMMAND_DELAY": "21,22"}
    cfg = EngineConfig.load(env=env, tower_level="14")
    assert (cfg.card_table, cfg.level, cfg.tower_level, cfg.delay) == \
        ("client15535", 13, 14, (21, 22))
    assert cfg.sources == {"card_table": str(file.resolve()), "level": "HASTYCR_CARD_LEVEL",
                           "tower_level": "caller", "delay": "HASTYCR_COMMAND_DELAY",
                           "calibration_overrides": "built-in"}
    # an explicit None is "not given"; "default" is the engine's own
    assert EngineConfig.load(env=env, level=None).level == 13
    assert EngineConfig.load(env=env, level="default").level is None
    # the path argument beats HASTYCR_ENGINE_CONFIG
    other = _write(tmp_path / "other.json", {"card_table": "default"})
    assert EngineConfig.load(other, env=env).card_table is None


def test_load_env_overrides_and_blank_values(tmp_path):
    assert EngineConfig.load(env={"HASTYCR_CARD_TABLE": "default"}).card_table is None
    assert EngineConfig.load(env={"HASTYCR_CARD_TABLE": "  "}).card_table == TARGET
    cfg = EngineConfig.load(env={"HASTYCR_CALIBRATION_OVERRIDES": '{"placement.TAP_SNAP": "none"}',
                                 "HASTYCR_TOWER_LEVEL": "default"})
    assert dict(cfg.calibration_overrides) == {"placement.TAP_SNAP": "none"}
    assert cfg.tower_level is None and cfg.level == 11


def test_load_named_files_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="--engine-config"):
        EngineConfig.load(tmp_path / "nope.json", env={})
    with pytest.raises(FileNotFoundError, match="HASTYCR_ENGINE_CONFIG"):
        EngineConfig.load(env={"HASTYCR_ENGINE_CONFIG": str(tmp_path / "nope.json")})
    # relative to the repo root when it comes from the environment
    cfg = EngineConfig.load(env={"HASTYCR_ENGINE_CONFIG": "data/engine.json"})
    assert cfg.card_table == TARGET and Path(cfg.sources["card_table"]).name == "engine.json"


def test_missing_shipped_file_falls_back_to_builtin(tmp_path, monkeypatch):
    monkeypatch.setattr(rs_engine, "CONFIG_PATH", tmp_path / "absent.json")
    cfg = EngineConfig.load(env={})
    assert cfg == EngineConfig.defaults()
    assert set(cfg.sources.values()) == {"built-in"}


def test_load_legacy_name():
    assert EngineConfig.load("legacy", env={}) == EngineConfig.legacy()
    cfg = EngineConfig.load(env={"HASTYCR_ENGINE_CONFIG": "Legacy",
                                 "HASTYCR_CARD_TABLE": "client15535"})
    assert (cfg.card_table, cfg.level, cfg.tower_level, cfg.delay) == \
        ("client15535", None, None, None)
    assert cfg.sources["level"] == "legacy" and cfg.sources["card_table"] == "HASTYCR_CARD_TABLE"


def test_load_errors_say_where_the_value_came_from(tmp_path):
    bad = _write(tmp_path / "bad.json", {"level": 0})
    with pytest.raises(ValueError, match=r"level.*bad\.json"):
        EngineConfig.load(bad, env={})
    typo = _write(tmp_path / "typo.json", {"card_tabel": "client15535"})
    with pytest.raises(ValueError, match="unknown setting"):
        EngineConfig.load(typo, env={})
    with pytest.raises(ValueError, match="HASTYCR_COMMAND_DELAY"):
        EngineConfig.load(env={"HASTYCR_COMMAND_DELAY": "-3"})
    with pytest.raises(TypeError, match="unknown setting"):
        EngineConfig.load(env={}, card_tabel="x")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not JSON"):
        EngineConfig.load(tmp_path / "broken.json", env={})


def test_file_with_bom_reads(tmp_path):
    file = tmp_path / "bom.json"
    file.write_bytes(b"\xef\xbb\xbf" + json.dumps({"card_table": "client15535"}).encode("utf-8"))
    assert EngineConfig.load(file, env={}).card_table == "client15535"


def test_round_trips_json_pickle_and_hash():
    cfg = EngineConfig(card_table="16.402.22", level=11, tower_level=12, delay=21,
                       calibration_overrides={"placement.TAP_SNAP": "none"})
    assert cfg.delay == (21, 21) and cfg.card_table == TARGET
    assert EngineConfig.from_json(json.loads(json.dumps(cfg.to_json()))) == cfg
    assert pickle.loads(pickle.dumps(cfg)) == cfg
    assert pickle.loads(pickle.dumps(rs_engine.set_active)) is rs_engine.set_active
    twin = EngineConfig.from_json(cfg.to_json())
    assert twin.key() == cfg.key() and hash(twin) == hash(cfg)
    assert EngineConfig.from_json(EngineConfig.legacy().to_json()) == EngineConfig.legacy()
    with pytest.raises(ValueError, match="unknown"):
        EngineConfig.from_json({"cardtable": 1})


def test_coerce_resolve_and_active(monkeypatch):
    cfg = EngineConfig(card_table="client15535")
    assert EngineConfig.coerce(cfg) is cfg
    assert EngineConfig.coerce(cfg.to_json()) == cfg
    foreign = types.SimpleNamespace(card_table="client15535", to_json=cfg.to_json)
    assert EngineConfig.coerce(foreign) == cfg          # a second import of rs_engine
    with pytest.raises(TypeError):
        EngineConfig.coerce(42)
    assert rs_engine.resolve(cfg) is cfg
    # active(): loaded once (environment included), then cached until set_active
    monkeypatch.setenv("HASTYCR_CARD_TABLE", "client15535")
    first = rs_engine.active()
    assert first.card_table == "client15535" and rs_engine.resolve() is first
    monkeypatch.setenv("HASTYCR_CARD_TABLE", "default")
    assert rs_engine.active() is first
    assert rs_engine.set_active(EngineConfig.legacy().to_json()) == EngineConfig.legacy()
    assert rs_engine.active() == EngineConfig.legacy()
    assert rs_engine.set_active(None) is None
    assert rs_engine.active().card_table is None        # loaded again: HASTYCR_CARD_TABLE=default


def test_delayed_undelayed_and_replace():
    legacy = EngineConfig.legacy()
    assert legacy.delayed(0) is legacy and legacy.undelayed().delay is None
    assert legacy.delayed(21).delay == (21, 21)
    assert EngineConfig.defaults().delayed(0).delay == (0, 0)
    assert EngineConfig(delay=(21, 22)).undelayed().delay == (0, 0)
    changed = EngineConfig.defaults().replace(card_table="15.535")
    assert changed.card_table == "client15535" and changed.sources["card_table"] == "caller"
    assert changed.sources["level"] == "built-in"
    with pytest.raises(TypeError):
        EngineConfig.defaults().replace(sources={})
    text = EngineConfig.defaults().describe()
    assert TARGET in text and "[built-in]" in text and "delay 0/0" in text
    assert "no override" in legacy.describe() and "not set" in legacy.describe()


# ---------------------------------------------------------------------------
# make_battle on the fake engine
# ---------------------------------------------------------------------------

def test_make_battle_legacy_is_exactly_the_old_battle(fake_royalesim):
    battle = rs_engine.make_battle(EngineConfig.legacy())
    assert fake_royalesim.battles == [battle]
    assert battle.init_args == ()
    assert battle.init_kwargs == LEGACY_KWARGS
    assert battle.calls == []                       # no delay call


def test_make_battle_sets_the_delay_before_the_first_reset(fake_royalesim):
    cfg = EngineConfig(card_table=TARGET, level=11, tower_level=11, delay=(21, 22))
    battle = rs_engine.make_battle(cfg, fake_royalesim)
    battle.reset(seed=0, decks=[list(range(8)), list(range(8, 16))], shuffle=0, start_tick=0,
                 elixir_milli=[6000, 6000], tower_hp=None, spawns=[])
    assert battle.names() == ["set_command_delay_ticks", "reset"]
    assert battle.calls[0].args == (21, 22) and battle.delay == (21, 22)
    assert battle.init_kwargs == {**LEGACY_KWARGS, "calibration_overrides": TARGET_OVERRIDE,
                                  "level": 11, "tower_level": 11}
    assert json.loads(battle.init_kwargs["calibration_overrides"]["cards.CARD_TABLE"]) == TARGET


def test_make_battle_sets_an_explicit_zero_delay(fake_royalesim):
    battle = rs_engine.make_battle(EngineConfig.defaults())
    assert [(c.name, c.args) for c in battle.calls] == [("set_command_delay_ticks", (0, 0))]


def test_make_battle_uses_the_active_config(fake_royalesim):
    rs_engine.set_active(EngineConfig(card_table="client15535"))
    battle = rs_engine.make_battle()
    assert battle.init_kwargs == {**LEGACY_KWARGS,
                                  "calibration_overrides": {"cards.CARD_TABLE": '"client15535"'}}
    assert battle.calls == []


def test_make_battle_extra_overrides_only(fake_royalesim):
    cfg = EngineConfig(calibration_overrides={"placement.TAP_SNAP": "none"})
    battle = rs_engine.make_battle(cfg)
    assert battle.init_kwargs == {**LEGACY_KWARGS,
                                  "calibration_overrides": {"placement.TAP_SNAP": '"none"'}}


def test_make_battle_names_a_refused_table():
    mod = refusing_royalesim()
    with pytest.raises(rs_engine.EngineError, match="0.1.42") as info:
        rs_engine.make_battle(EngineConfig(card_table=TARGET), mod)
    assert isinstance(info.value.__cause__, ValueError)
    assert rs_engine.make_battle(EngineConfig.legacy(), mod).init_kwargs == LEGACY_KWARGS
    # a legacy Battle that fails is the engine's own error, unwrapped
    with pytest.raises(ValueError):
        rs_engine.make_battle(EngineConfig.legacy(), refusing_royalesim(always=True))


def test_make_battle_checks_the_table_the_battle_reports():
    ok = royalesim_with(ReportingBattle)
    assert rs_engine.make_battle(EngineConfig(card_table=TARGET), ok).card_table() == \
        "cards-160402017-20261019.1"
    assert rs_engine.check_card_table(ok.battles[-1], EngineConfig(card_table=TARGET)) == \
        "cards-160402017-20261019.1"
    with pytest.raises(rs_engine.EngineError, match="cards-15535.1"):
        rs_engine.make_battle(EngineConfig(card_table=TARGET), royalesim_with(IgnoringBattle))
    # a new revision of the same table is the same table
    rs_engine.make_battle(EngineConfig(card_table=TARGET), royalesim_with(NewRevisionBattle))
    # no table named: nothing to check
    assert rs_engine.check_card_table(ok.battles[-1], EngineConfig.legacy()) is None
    # an engine without card_table() (0.1.22, the fake) is not checked
    assert rs_engine.check_card_table(FakeBattle(), EngineConfig(card_table=TARGET)) is None


def test_catalogue_and_unit_types_helpers(fake_royalesim):
    assert rs_engine.catalogue(EngineConfig.legacy()) == [list(r) for r in DEFAULT_CATALOGUE]
    assert rs_engine.unit_types(EngineConfig.legacy()) == list(DEFAULT_UNIT_TYPES)


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------

def test_fnv1a64_reference_vectors():
    assert rs_engine.fnv1a64(b"") == "cbf29ce484222325"
    assert rs_engine.fnv1a64(b"a") == "af63dc4c8601ec8c"
    assert rs_engine.fnv1a64(b"foobar") == "85944171f73967e8"


def test_fingerprint_of_a_fake_battle(fake_royalesim):
    cfg = EngineConfig.defaults()
    fp = rs_engine.fingerprint(rs_engine.make_battle(cfg), cfg=cfg)
    names = [row[0] for row in DEFAULT_CATALOGUE]
    assert fp == {
        "engine": "royalesim", "version": "0.0.0+fake", "provenance": None,
        "card_table": None, "card_table_fnv1a64": None,
        "unit_types": len(DEFAULT_UNIT_TYPES),
        "unit_types_digest": rs_engine.fnv1a64(json.dumps(list(DEFAULT_UNIT_TYPES)).encode()),
        "catalogue": len(DEFAULT_CATALOGUE),
        "catalogue_sha1": hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest(),
        "calibration_overrides": TARGET_OVERRIDE,     # the fake does not report: cfg's
        "card_level": None, "tower_level": None, "command_delay_ticks": None,
        "entity_fields": 28, "id": fp["id"],
    }
    assert fp["id"] == rs_engine.fingerprint_id(fp) and len(fp["id"]) == 16
    assert rs_engine.fingerprint(rs_engine.make_battle(cfg))["calibration_overrides"] == {}
    assert fp["id"] in rs_engine.summary(fp)


def test_fingerprint_reads_what_the_engine_reports():
    mod = royalesim_with(ReportingBattle)
    cfg = EngineConfig.defaults()
    fp = rs_engine.fingerprint(rs_engine.make_battle(cfg, mod), mod, cfg)
    assert fp["card_table"] == "cards-160402017-20261019.1"
    assert fp["card_table_fnv1a64"] == rs_engine.fnv1a64(b"cards-160402017-20261019.1")
    assert fp["calibration_overrides"] == TARGET_OVERRIDE
    assert (fp["card_level"], fp["tower_level"], fp["command_delay_ticks"]) == (11, 11, [0, 0])
    assert fp["provenance"] == ["abc1234", "clean"]
    ints = royalesim_with(IntHashBattle)
    assert rs_engine.fingerprint(rs_engine.make_battle(cfg, ints), ints)["card_table_fnv1a64"] \
        == "0000000000001234"


def test_fingerprints_differ_by_table_vocabulary_catalogue_and_version():
    mod = royalesim_with(ReportingBattle)

    def fp_of(cfg, module=mod):
        return rs_engine.fingerprint(rs_engine.make_battle(cfg, module), module)

    old, new = fp_of(EngineConfig(card_table="client15535")), fp_of(EngineConfig(card_table=TARGET))
    assert old["id"] != new["id"] and old["card_table"] != new["card_table"]
    # levels and delay are settings, not identity
    other_level = fp_of(EngineConfig(card_table=TARGET, level=12, delay=21))
    assert other_level["id"] == new["id"] and other_level["card_level"] == 12

    plain = fp_of(EngineConfig.legacy(), make_fake_royalesim())
    more_units = fp_of(EngineConfig.legacy(),
                       make_fake_royalesim(unit_types=[*DEFAULT_UNIT_TYPES, "ElectroGiant"]))
    assert more_units["unit_types"] == plain["unit_types"] + 1
    assert more_units["unit_types_digest"] != plain["unit_types_digest"]
    assert more_units["id"] != plain["id"]
    fewer_cards = fp_of(EngineConfig.legacy(),
                        make_fake_royalesim(catalogue=list(DEFAULT_CATALOGUE[:-1])))
    assert fewer_cards["catalogue_sha1"] != plain["catalogue_sha1"]
    assert fewer_cards["id"] != plain["id"]
    newer = make_fake_royalesim()
    newer.__version__ = "0.1.43"
    assert fp_of(EngineConfig.legacy(), newer)["id"] != plain["id"]
    assert fp_of(EngineConfig.legacy(), make_fake_royalesim())["id"] == plain["id"]


# ---------------------------------------------------------------------------
# entity layout
# ---------------------------------------------------------------------------

def test_expected_entity_fields_are_the_conftest_layout():
    assert rs_engine.ENTITY_FIELDS_EXPECTED == ENTITY_FIELDS
    assert len(rs_engine.ENTITY_FIELDS_EXPECTED) == 28


def test_field_index_of_the_fake(fake_royalesim):
    assert rs_engine.field_index() == FIELD
    assert rs_engine.check_entity_fields(fake_royalesim) == ENTITY_FIELDS


def test_field_index_refuses_another_layout():
    swapped = make_fake_royalesim()
    fields = list(ENTITY_FIELDS)
    fields[15], fields[16] = fields[16], fields[15]
    swapped.ENTITY_FIELDS = fields
    with pytest.raises(rs_engine.EngineError, match=r"\[15\] is 'attack_phase'"):
        rs_engine.field_index(swapped)
    longer = make_fake_royalesim()
    longer.ENTITY_FIELDS = [*ENTITY_FIELDS, "new_field"]
    with pytest.raises(rs_engine.EngineError, match="29 fields"):
        rs_engine.field_index(longer)
    none = make_fake_royalesim()
    del none.ENTITY_FIELDS
    with pytest.raises(rs_engine.EngineError, match="no ENTITY_FIELDS"):
        rs_engine.field_index(none)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def test_cli_prints_the_settings_without_the_engine(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "royalesim", None)      # not needed without --probe
    assert rs_engine.main([]) == 0
    out = capsys.readouterr().out
    assert TARGET in out and "data/engine.json" in out
    assert rs_engine.main(["--json", "--card-table", "default"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["config"]["card_table"] is None and data["sources"]["card_table"] == "caller"
    assert data["config"]["level"] == 11


def test_cli_probe_on_the_fake(fake_royalesim, capsys):
    assert rs_engine.main(["--probe", "--json", "--card-table", "client15535"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["config"]["card_table"] == "client15535"
    assert data["fingerprint"]["calibration_overrides"] == {"cards.CARD_TABLE": '"client15535"'}
    assert data["entity_fields_ok"] is True
    battle = fake_royalesim.battles[-1]
    assert battle.init_kwargs["level"] == 11 and battle.names() == ["set_command_delay_ticks"]
    assert rs_engine.main(["--probe", "--delay", "21,22", "--level", "default"]) == 0
    out = capsys.readouterr().out
    assert "as expected" in out and "id " in out
    assert fake_royalesim.battles[-1].calls[0].args == (21, 22)
    assert "level" not in fake_royalesim.battles[-1].init_kwargs


def test_cli_probe_text_with_engine_readouts(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "royalesim", royalesim_with(ReportingBattle))
    assert rs_engine.main(["--probe"]) == 0
    out = capsys.readouterr().out
    assert "cards-160402017-20261019.1" in out and "command delay 0/0" in out
    assert "provenance abc1234 clean" in out


def test_cli_exit_codes(monkeypatch, capsys):
    assert rs_engine.main(["--level", "0"]) == 2
    assert "level" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "royalesim", None)
    assert rs_engine.main(["--probe"]) == 1
    assert "does not import" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "royalesim", refusing_royalesim())
    assert rs_engine.main(["--probe"]) == 1
    assert "engine:" in capsys.readouterr().err
    swapped = make_fake_royalesim()
    swapped.ENTITY_FIELDS = list(reversed(ENTITY_FIELDS))
    monkeypatch.setitem(sys.modules, "royalesim", swapped)
    assert rs_engine.main(["--probe"]) == 1
    assert "DIFFERENT" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the routed tools (fake engine)
# ---------------------------------------------------------------------------

ROUTED = ("rs_rows.py", "rs_eval.py", "royalesim_redrive.py", "rs_shadow.py")


def _attribute_calls(path: Path, attr: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr]


@pytest.mark.parametrize("name", ROUTED)
def test_tools_build_battles_only_through_rs_engine(name):
    path = TOOLS / name
    assert _attribute_calls(path, "Battle") == [], f"{name} builds a Battle itself"
    assert _attribute_calls(path, "set_command_delay_ticks") == []
    assert "rs_engine.make_battle(" in path.read_text(encoding="utf-8")


def test_rs_engine_is_the_one_battle_constructor():
    assert len(_attribute_calls(TOOLS / "rs_engine.py", "Battle")) == 1


def test_rs_rows_setup_and_rows_follow_the_config(fake_royalesim, fake_imitate, tools):
    RR = tools.rs_rows
    cfg = EngineConfig.defaults()
    ctx = RR._setup(cfg)
    rs, names, ids, cat = ctx
    assert rs is fake_royalesim and NAMES.items() <= names.items() and ids["HogRider"] == 2
    assert cat == [list(r) for r in DEFAULT_CATALOGUE]
    assert RR.F == FIELD
    assert fake_royalesim.battles[-1].init_kwargs == cfg.constructor_kwargs()

    rows, why = RR.rows_for(il_payload(), ctx, cfg)
    assert why == "ok" and rows
    battle = fake_royalesim.battles[-1]
    assert battle.init_kwargs == cfg.constructor_kwargs()
    assert battle.names()[:2] == ["set_command_delay_ticks", "reset"]
    assert battle.calls[0].args == (0, 0)

    legacy = EngineConfig.legacy()
    legacy_rows, why = RR.rows_for(il_payload(), RR._setup(legacy), legacy)
    old = fake_royalesim.battles[-1]
    assert why == "ok"
    assert old.init_kwargs == LEGACY_KWARGS
    assert old.names()[0] == "reset" and "set_command_delay_ticks" not in old.names()
    # the same match, the same commands, the same rows: only the Battle's settings differ
    assert old.calls_named("reset")[0].kwargs == battle.calls_named("reset")[0].kwargs
    assert old.calls_named("step") == battle.calls_named("step")
    assert [(r["tick"], r["side"], r["action"], r["trusted"]) for r in legacy_rows] == \
        [(r["tick"], r["side"], r["action"], r["trusted"]) for r in rows]

    # a configured command delay never reaches a re-drive
    RR.rows_for(il_payload(), ctx, EngineConfig(card_table=TARGET, delay=(21, 21)))
    assert fake_royalesim.battles[-1].calls[0].args == (0, 0)


def test_rs_rows_build_passes_the_config_to_workers(fake_royalesim, fake_imitate, tools,
                                                    tmp_path, monkeypatch):
    RR = tools.rs_rows
    monkeypatch.setattr(RR, "OUT", RR.OUT)
    monkeypatch.setattr(RR, "LIVE", RR.LIVE)
    cfg = EngineConfig(card_table="client15535", level=11)
    batch, n_rows, skipped = RR.build(3, [json.dumps(il_payload())], False, str(tmp_path),
                                      cfg.to_json())
    assert (batch, skipped) == (3, {}) and n_rows > 0
    assert (tmp_path / "shard-0003.npz").is_file()
    assert {json.dumps(b.init_kwargs, sort_keys=True) for b in fake_royalesim.battles} == \
        {json.dumps(cfg.constructor_kwargs(), sort_keys=True)}


def test_redrive_one_follows_the_config(fake_royalesim, fake_imitate, tools):
    RD = tools.redrive
    payload = json.dumps(il_payload())
    cfg = EngineConfig(card_table=TARGET, level=11, tower_level=11, delay=(21, 21))
    out = RD.one(payload, engine=cfg)
    assert out["ok"], out
    assert out["plays"] == 5 and out["forced"] == {"hand": 2}
    assert len(fake_royalesim.battles) == 2                 # the card map's, then the match
    match = fake_royalesim.battles[-1]
    assert match.init_kwargs == cfg.constructor_kwargs()
    assert match.names()[:2] == ["set_command_delay_ticks", "reset"]
    assert match.calls[0].args == (0, 0)                    # re-drives run undelayed
    # cached per config; a new config rebuilds the card map
    RD.one(payload, engine=cfg)
    assert len(fake_royalesim.battles) == 3
    legacy_out = RD.one(payload, engine=EngineConfig.legacy())
    assert len(fake_royalesim.battles) == 5
    old = fake_royalesim.battles[-1]
    assert old.init_kwargs == LEGACY_KWARGS and old.names()[0] == "reset"
    assert old.calls_named("step") == match.calls_named("step")
    assert legacy_out == out
    # with no engine given, the process's active config
    rs_engine.set_active(EngineConfig(card_table="client15535"))
    RD.one(payload)
    assert fake_royalesim.battles[-1].init_kwargs["calibration_overrides"] == \
        {"cards.CARD_TABLE": '"client15535"'}


def _play(rs_eval, delay, engine):
    payload = json.dumps(il_payload())
    return rs_eval.play((payload, "", "pro", 0, "", delay, 0, False, False, False, engine))


def test_rs_eval_play_sets_the_delay_before_reset(fake_royalesim, fake_imitate, tools):
    RE = tools.rs_eval
    cfg = EngineConfig.defaults()
    out = _play(RE, 21, cfg.to_json())
    assert out["mode"] == "pro" and out["hog_team"] == 0
    assert rs_engine.active() == cfg                       # args[10] became the worker's config
    match = fake_royalesim.battles[-1]
    assert match.init_kwargs == cfg.constructor_kwargs()
    assert match.names()[:2] == ["set_command_delay_ticks", "reset"]
    assert match.calls[0].args == (21, 21)

    _play(RE, 0, cfg.to_json())
    assert fake_royalesim.battles[-1].calls[0].args == (0, 0)

    # legacy: no delay call at delay 0, the old call before reset at delay 21
    _play(RE, 0, EngineConfig.legacy().to_json())
    old = fake_royalesim.battles[-1]
    assert old.init_kwargs == LEGACY_KWARGS and old.names()[0] == "reset"
    assert "set_command_delay_ticks" not in old.names()
    _play(RE, 21, EngineConfig.legacy().to_json())
    old21 = fake_royalesim.battles[-1]
    assert old21.names()[:2] == ["set_command_delay_ticks", "reset"]
    assert old21.calls[0].args == (21, 21)
    assert old21.calls_named("step") == match.calls_named("step")


def test_rs_eval_rebuilt_runs_undelayed(fake_royalesim, tools, monkeypatch):
    RE = tools.rs_eval
    monkeypatch.setattr(RE, "_spawn_tables", lambda: ({}, {}))
    monkeypatch.setattr(RE.RR, "F", rs_engine.field_index(fake_royalesim))
    agent = types.SimpleNamespace(engine=EngineConfig(card_table=TARGET, level=11, delay=(21, 21)))
    state = state_dict(300, players=[player_dict(0, deck=list(range(8))),
                                     player_dict(1, deck=list(range(8, 16)))])
    battle = RE.RSAgent.rebuilt(agent, state, 0, fake_royalesim)
    assert battle is fake_royalesim.battles[-1]
    assert battle.init_kwargs == {**LEGACY_KWARGS, "calibration_overrides": TARGET_OVERRIDE,
                                  "level": 11}
    assert battle.names() == ["set_command_delay_ticks", "reset"]
    assert battle.calls[0].args == (0, 0)
    assert battle.reset_kwargs["start_tick"] == 300
    assert RE.RSAgent.engine is None                        # class default: active()


def test_rs_eval_helpers_use_the_active_config(fake_royalesim, tools):
    RE = tools.rs_eval
    rs_engine.set_active(EngineConfig(card_table="client15535"))
    assert RE._unit_names() == list(DEFAULT_UNIT_TYPES)
    assert fake_royalesim.battles[-1].init_kwargs["calibration_overrides"] == \
        {"cards.CARD_TABLE": '"client15535"'}
    assert RE.RSAgent.cat_kind(types.SimpleNamespace(engine=None), 1) == "SPELL"   # Fireball
    assert RE.RSAgent._CAT == [list(r) for r in DEFAULT_CATALOGUE]


def test_shadow_battles_follow_the_config_undelayed(fake_royalesim, tools):
    fake_royalesim.SPELL_FIELDS = ["uid", "team", "card_id", "aim_x", "aim_y", "ticks_flown"]
    cfg = EngineConfig(card_table=TARGET, level=11, tower_level=11, delay=(21, 22))
    cat = [list(r) for r in DEFAULT_CATALOGUE]
    shadow = tools.rs_shadow.Shadow(fake_royalesim, cat, 0, list(range(8)), [11] * 8, 11, {},
                                    engine=cfg)
    assert shadow.F == FIELD and shadow.unit_names == list(DEFAULT_UNIT_TYPES)
    assert shadow.engine == cfg.undelayed()
    battle = shadow.battle(100)
    assert battle is fake_royalesim.battles[-1]
    assert battle.init_kwargs == cfg.constructor_kwargs()
    assert battle.names() == ["set_command_delay_ticks", "reset", "step"]
    assert battle.calls[0].args == (0, 0)


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

def _real_with_tables():
    import royalesim
    probe = rs_engine.make_battle(EngineConfig.legacy(), royalesim)
    if not hasattr(probe, "card_table"):
        pytest.skip(f"royalesim {royalesim.__version__} has no card tables (0.1.42 adds them)")
    return royalesim


@pytest.mark.engine
def test_engine_tables_build_and_their_fingerprints_differ():
    rs = _real_with_tables()
    fps = {}
    for arm in ("client15535", TARGET):
        battle = rs_engine.make_battle(EngineConfig(card_table=arm, level=11, tower_level=11), rs)
        assert battle.card_table().rpartition(".")[0] == rs_engine.table_version_stem(arm)
        fp = rs_engine.fingerprint(battle, rs)
        assert fp["unit_types"] > 0 and fp["catalogue"] > 0
        assert fp["calibration_overrides"]["cards.CARD_TABLE"] in (json.dumps(arm), arm)
        assert (fp["card_level"], fp["tower_level"]) == (11, 11)
        fps[arm] = fp
    old, new = fps["client15535"], fps[TARGET]
    assert old["id"] != new["id"]
    assert old["card_table"] != new["card_table"]
    assert old["card_table_fnv1a64"] != new["card_table_fnv1a64"]


@pytest.mark.engine
def test_engine_legacy_runs_the_ledgers_default_table():
    rs = _real_with_tables()
    ledger = json.loads((Path(rs.data_dir()) / "calibration.json").read_text(encoding="utf-8"))
    default_arm = ledger["cards"]["CARD_TABLE"]["value"]
    legacy = rs_engine.fingerprint(rs_engine.make_battle(EngineConfig.legacy(), rs), rs)
    named = rs_engine.fingerprint(rs_engine.make_battle(EngineConfig(card_table=default_arm), rs), rs)
    assert legacy["calibration_overrides"] == {}
    assert legacy["card_table"] == named["card_table"]
    assert legacy["card_table_fnv1a64"] == named["card_table_fnv1a64"]
    assert legacy["unit_types_digest"] == named["unit_types_digest"]


@pytest.mark.engine
def test_engine_delay_is_set_on_the_new_battle():
    import royalesim
    battle = rs_engine.make_battle(EngineConfig(delay=(21, 22)), royalesim)
    if not hasattr(battle, "command_delay_ticks"):
        pytest.skip("this royalesim has no command_delay_ticks()")
    assert tuple(battle.command_delay_ticks()) == (21, 22)


@pytest.mark.engine
def test_engine_unit_types_digest_is_fnv1a64_of_the_json():
    import royalesim
    battle = rs_engine.make_battle(EngineConfig.legacy(), royalesim)
    if not hasattr(battle, "unit_types_digest"):
        pytest.skip("this royalesim has no unit_types_digest()")
    assert rs_engine.fnv1a64(battle.unit_types_json().encode("utf-8")) == battle.unit_types_digest()


@pytest.mark.engine
def test_engine_entity_fields_are_the_expected_28():
    import royalesim
    assert rs_engine.field_index(royalesim) == FIELD


@pytest.mark.engine
def test_engine_default_config_builds_at_level_11():
    rs = _real_with_tables()
    battle = rs_engine.make_battle(EngineConfig.load(env={}), rs)
    assert battle.card_table().startswith(rs_engine.table_version_stem(TARGET))
    assert (battle.card_level(), battle.tower_level()) == (11, 11)


@pytest.mark.engine
def test_engine_probe_cli(capsys):
    _real_with_tables()
    assert rs_engine.main(["--probe", "--json", "--card-table", TARGET]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["entity_fields_ok"] is True
    assert data["fingerprint"]["card_table"].startswith(rs_engine.table_version_stem(TARGET))
