"""E02: tools/rs_cards.py, the vendored IL_Replay -> RoyaleSim card map in data/cards/.

Pure tests read the shipped files, and copies of them in tmp_path. Those that drive
tools/rs_rows.py and royalesim_redrive.py use conftest's fake royalesim, with royaleimitate
made unimportable. The `engine`-marked tests need the real royalesim, and the `il_replay`
one needs the corpus.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    OPPONENT_KEYS,
    catalogue_row,
    il_payload,
    il_replay_dir,
    make_fake_royalesim,
)

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def _load(name: str):
    """tools/<name>.py as the module `name`, the name the tools import it by, without
    putting tools/ on sys.path for the whole session."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rs_cards = _load("rs_cards")
rs_engine = _load("rs_engine")

SHA256 = "aeca8fdf8eebafadbe3b1f4dba6eebb6132715e935bfe1db63f5968670c039ee"
EXTRAS = {"void": "DarkMagic", "rune-giant": "GiantBuffer", "spirit-empress": "MergeMaiden"}
#: The conftest opponent deck with its unplayed Zap swapped for Void.
VOID_DECK = tuple("void" if key == "zap" else key for key in OPPONENT_KEYS)
#: An opponent deck with all three extras, in slots the default plays do not use.
EXTRA_DECK = ("knight", "archers", "rune-giant", "arrows", "spirit-empress", "mini-pekka",
              "baby-dragon", "void")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No HASTYCR_CARD_EXTRAS and no HASTYCR_* engine setting from the PC."""
    monkeypatch.delenv(rs_cards.EXTRAS_ENV, raising=False)
    for var in (rs_engine.CONFIG_ENV, *rs_engine.ENV.values()):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def tools(monkeypatch):
    """rs_rows and royalesim_redrive as they import each other, caches empty, no royaleimitate."""
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rs_rows = importlib.import_module("rs_rows")
    redrive = importlib.import_module("royalesim_redrive")
    assert rs_rows.rs_cards is rs_cards and redrive.rs_cards is rs_cards
    monkeypatch.setattr(rs_rows, "F", None)
    monkeypatch.setattr(redrive, "_ENGINE", {})
    return rs_rows, redrive


@pytest.fixture
def void_royalesim(monkeypatch):
    """conftest's fake royalesim whose catalogue also has DarkMagic (id 16)."""
    mod = make_fake_royalesim(catalogue=[*DEFAULT_CATALOGUE, catalogue_row("DarkMagic", 3, "SPELL")])
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    return mod


# ---------------------------------------------------------------------------
# the shipped files
# ---------------------------------------------------------------------------

def test_vendored_file_is_royaleimitate_0_2_15():
    assert rs_cards.VENDORED_SHA256 == SHA256
    assert rs_cards.file_sha256(rs_cards.VENDORED_PATH) == SHA256
    vendored = rs_cards.read_vendored()
    assert len(vendored) == rs_cards.VENDORED_ENTRIES == 119
    assert list(vendored) == sorted(vendored)
    assert {k: vendored[k] for k in ("hog-rider", "the-log", "ice-golem", "ice-spirit",
                                     "elixir-collector", "zappies", "bandit")} == {
        "hog-rider": "HogRider", "the-log": "Log", "ice-golem": "IceGolemite",
        "ice-spirit": "IceSpirits", "elixir-collector": "Elixir Collector",
        "zappies": "MiniSparkys", "bandit": "Assassin"}
    assert not set(EXTRAS) & set(vendored)


def test_mit_notice_ships_with_the_file():
    notice = rs_cards.LICENSE_PATH.read_text(encoding="utf-8")
    assert notice.startswith("MIT License")
    assert "Copyright (c) 2026 RoyaleGym contributors" in notice
    assert "The above copyright notice and this permission notice shall be included" in notice
    readme = (rs_cards.CARDS_DIR / "README.md").read_text(encoding="utf-8")
    assert notice.strip() in readme
    assert SHA256 in readme


def test_load_merges_the_three_extras_into_122_keys():
    vendored = rs_cards.read_vendored()
    names = rs_cards.load(env={})
    assert len(names) == 122
    assert {k: names[k] for k in EXTRAS} == EXTRAS
    assert set(names) - set(vendored) == set(EXTRAS)
    assert vendored.items() <= names.items()
    assert rs_cards.load(extras=True) == names == rs_cards.load()
    assert rs_cards.load() is not rs_cards.load()          # a new dict per call


def test_no_key_maps_to_two_names_and_no_name_has_two_keys():
    vendored, extras = rs_cards.read_vendored(), rs_cards.read_extras()
    assert not set(extras) & set(vendored)
    names = rs_cards.load(extras=True)
    assert [name for name, n in Counter(names.values()).items() if n > 1] == []


def test_extras_file_entries():
    extras = rs_cards.read_extras()
    assert {k: e.name for k, e in extras.items()} == EXTRAS
    for key, extra in extras.items():
        assert extra.key == key
        assert isinstance(extra.verified, bool)
        assert extra.evidence


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_an_altered_vendored_file_is_refused(tmp_path):
    data = json.loads(rs_cards.VENDORED_PATH.read_text(encoding="utf-8"))
    data["hog-rider"] = "Giant"
    altered = tmp_path / "replay_cards.json"
    altered.write_text(json.dumps(data, indent=1), encoding="utf-8")
    with pytest.raises(rs_cards.CardMapError, match="SHA-256"):
        rs_cards.read_vendored(altered)
    with pytest.raises(rs_cards.CardMapError, match="SHA-256"):
        rs_cards.load(vendored_path=altered)
    assert rs_cards.read_vendored(altered, sha256=None)["hog-rider"] == "Giant"
    with pytest.raises(rs_cards.CardMapError, match="missing"):
        rs_cards.read_vendored(tmp_path / "nope.json")


def test_hash_reads_crlf_as_lf(tmp_path):
    lf = rs_cards.VENDORED_PATH.read_bytes().replace(b"\r\n", b"\n")
    crlf = tmp_path / "crlf.json"
    crlf.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert crlf.read_bytes() != lf
    assert rs_cards.file_sha256(crlf) == SHA256
    assert rs_cards.read_vendored(crlf) == rs_cards.read_vendored()


def test_a_key_given_twice_is_refused(tmp_path):
    extras = tmp_path / "extras.json"
    extras.write_text('{"cards": {"void": {"name": "DarkMagic"}, "void": {"name": "Zap"}}}',
                      encoding="utf-8")
    with pytest.raises(rs_cards.CardMapError, match="'void' appears twice"):
        rs_cards.read_extras(extras)
    vendored = tmp_path / "vendored.json"
    vendored.write_text('{"zap": "Zap", "zap": "Log"}', encoding="utf-8")
    with pytest.raises(rs_cards.CardMapError, match="'zap' appears twice"):
        rs_cards.read_vendored(vendored, sha256=None)


def test_an_extra_may_not_rename_a_mapped_key(tmp_path):
    vendored = {"zap": "Zap", "the-log": "Log"}
    with pytest.raises(rs_cards.CardMapError, match="maps to one name"):
        rs_cards.merge(vendored, {"zap": rs_cards.Extra("zap", "Lightning")})
    assert rs_cards.merge(vendored, {"zap": "Zap", "void": "DarkMagic"}) == \
        {"zap": "Zap", "the-log": "Log", "void": "DarkMagic"}
    assert vendored == {"zap": "Zap", "the-log": "Log"}
    clash = tmp_path / "clash.json"
    clash.write_text(json.dumps({"cards": {"hog-rider": {"name": "Giant"}}}), encoding="utf-8")
    with pytest.raises(rs_cards.CardMapError, match="'hog-rider'"):
        rs_cards.load(extra_path=clash)
    same = tmp_path / "same.json"
    same.write_text(json.dumps({"cards": {"hog-rider": {"name": "HogRider"}}}), encoding="utf-8")
    assert rs_cards.load(extra_path=same) == rs_cards.read_vendored()


@pytest.mark.parametrize("text, match", [
    ("[]", "expected"),
    ('{"cards": []}', "expected"),
    ('{"cards": {"void": "DarkMagic"}}', "must be an object"),
    ('{"cards": {"void": {}}}', "name must be"),
    ('{"cards": {"void": {"name": ""}}}', "name must be"),
    ('{"cards": {"void": {"name": "DarkMagic", "verified": "no"}}}', '"verified"'),
    ('{"cards": {"void": {"name": "DarkMagic", "evidence": 3}}}', '"evidence"'),
    ('{"cards": {" void": {"name": "DarkMagic"}}}', "not a card key"),
    ('{"cards": {', "not valid JSON"),
])
def test_malformed_extras_are_refused(tmp_path, text, match):
    path = tmp_path / "extras.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(rs_cards.CardMapError, match=match):
        rs_cards.read_extras(path)


def test_empty_extras_add_nothing(tmp_path):
    path = tmp_path / "extras.json"
    path.write_text('{"_comment": "none yet", "cards": {}}', encoding="utf-8")
    assert rs_cards.read_extras(path) == {}
    assert rs_cards.load(extra_path=path) == rs_cards.read_vendored()


# ---------------------------------------------------------------------------
# HASTYCR_CARD_EXTRAS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, on", [("", True), ("1", True), ("yes", True), (" ON ", True),
                                       ("true", True), ("0", False), ("off", False),
                                       ("False", False), ("no", False)])
def test_extras_switch(value, on):
    env = {rs_cards.EXTRAS_ENV: value}
    assert rs_cards.extras_enabled(env) is on
    assert len(rs_cards.load(env=env)) == (122 if on else 119)
    # an explicit argument wins over the variable
    assert len(rs_cards.load(extras=True, env=env)) == 122
    assert len(rs_cards.load(extras=False, env=env)) == 119


def test_extras_switch_refuses_other_values():
    with pytest.raises(rs_cards.CardMapError, match=rs_cards.EXTRAS_ENV):
        rs_cards.extras_enabled({rs_cards.EXTRAS_ENV: "maybe"})


def test_extras_switch_reads_the_process_environment(monkeypatch):
    monkeypatch.setenv(rs_cards.EXTRAS_ENV, "0")
    assert rs_cards.load() == rs_cards.read_vendored()
    monkeypatch.delenv(rs_cards.EXTRAS_ENV)
    assert len(rs_cards.load()) == 122


# ---------------------------------------------------------------------------
# describe / summary / catalogue / coverage
# ---------------------------------------------------------------------------

def test_describe_records_the_map():
    names, vendored = rs_cards.load(extras=True), rs_cards.read_vendored()
    d = rs_cards.describe(names)
    assert d["entries"] == 122 and d["extras"] == ["rune-giant", "spirit-empress", "void"]
    assert d["vendored_sha256"] == SHA256 and "0.2.15" in d["vendored"]
    assert len(d["sha256"]) == 64
    assert d["sha256"] == rs_cards.map_sha256(dict(reversed(list(names.items()))))
    plain = rs_cards.describe(vendored, vendored)
    assert plain["entries"] == 119 and plain["extras"] == [] and plain["sha256"] != d["sha256"]
    assert "122 keys (extras: rune-giant, spirit-empress, void)" in rs_cards.summary(names)
    assert "extras: none, HASTYCR_CARD_EXTRAS=0" in rs_cards.summary(vendored)


def test_missing_from_catalogue():
    names = {"zap": "Zap", "void": "DarkMagic", "the-log": "Log"}
    catalogue = [catalogue_row("Zap", 2, "SPELL"), catalogue_row("Log", 2, "SPELL")]
    assert rs_cards.missing_from_catalogue(names, catalogue) == {"void": "DarkMagic"}
    assert rs_cards.missing_from_catalogue(names, ["Zap", "Log", "DarkMagic"]) == {}


def test_coverage_counts_unmapped_keys_and_extras(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))           # base_key comes from royalesim_redrive
    plain, void = il_payload(), il_payload(tag="void-1", opponent_deck=VOID_DECK)
    two_v_two = il_payload(tag="2v2")
    team = two_v_two["battle"]["team"]["players"]
    team.append(dict(team[0]))
    payloads = [plain, void, two_v_two]
    watch = rs_cards.read_extras()
    cov = rs_cards.coverage(payloads, rs_cards.load(extras=True), watch=watch)
    assert cov == {"matches": 2, "mapped": 2, "not_1v1": 1, "blocked": {},
                   "watched": {"rune-giant": 0, "spirit-empress": 0, "void": 1}, "keys": 17}
    cov = rs_cards.coverage(payloads, rs_cards.load(extras=False), watch=watch)
    assert (cov["mapped"], cov["blocked"]) == (1, {"void": 1})
    assert cov["watched"]["void"] == 1
    # an explicit key rule is handed the raw card keys, form suffixes included
    seen = []
    rs_cards.coverage([plain], {}, key_of=lambda k: (seen.append(k) or k, 0))
    assert "knight-ev1" in seen and "musketeer-hero" in seen


def test_corpus_parts(tmp_path):
    assert rs_cards.corpus_parts(tmp_path) == []
    (tmp_path / "replays").mkdir()
    for k in (2, 0, 1):
        (tmp_path / "replays" / f"part-{k:05d}.parquet").write_bytes(b"")
    (tmp_path / "replays" / "other.parquet").write_bytes(b"")
    assert [p.name for p in rs_cards.corpus_parts(tmp_path)] == \
        ["part-00000.parquet", "part-00001.parquet", "part-00002.parquet"]
    assert [p.name for p in rs_cards.corpus_parts(tmp_path, parts=2)] == \
        ["part-00000.parquet", "part-00001.parquet"]


# ---------------------------------------------------------------------------
# the tools read the map through rs_cards
# ---------------------------------------------------------------------------

def _imported_modules(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_no_tool_imports_royaleimitate():
    offenders = sorted(p.relative_to(ROOT).as_posix() for p in TOOLS.rglob("*.py")
                       if any(m.split(".")[0] == "royaleimitate" for m in _imported_modules(p)))
    assert offenders == []
    for name in ("rs_rows.py", "royalesim_redrive.py"):
        assert "rs_cards" in set(_imported_modules(TOOLS / name))
        assert "rs_cards.load()" in (TOOLS / name).read_text(encoding="utf-8")


def _row_view(rows):
    return [(r["tick"], r["side"], r["action"], r["trusted"], r["outcome"], r["hand"].tolist(),
             r["unit_ids"].tolist(), r["opp_seen"].tolist()) for r in rows]


def test_rows_and_redrive_use_the_vendored_map(fake_royalesim, tools, monkeypatch):
    rs_rows, redrive = tools
    cfg = rs_engine.EngineConfig.defaults()
    _, names, ids, _ = rs_rows._setup(cfg)
    assert names == rs_cards.load()
    assert redrive._engine(cfg)["names"] == rs_cards.load()
    rows, why = rs_rows.rows_for(il_payload(), rs_rows._setup(cfg), cfg)
    assert why == "ok" and rows
    out = redrive.one(json.dumps(il_payload()), engine=cfg)
    assert out["ok"]

    # golden: a match without the extra cards gives the same rows and re-drive as the
    # 119-key map the tools read from royaleimitate before
    monkeypatch.setenv(rs_cards.EXTRAS_ENV, "0")
    monkeypatch.setattr(redrive, "_ENGINE", {})
    old_rows, old_why = rs_rows.rows_for(il_payload(), rs_rows._setup(cfg), cfg)
    assert old_why == "ok" and _row_view(old_rows) == _row_view(rows)
    assert redrive.one(json.dumps(il_payload()), engine=cfg) == out
    assert len(redrive._ENGINE["names"]) == 119


def test_a_void_match_is_kept_with_the_extras_and_skipped_without(void_royalesim, tools,
                                                                  monkeypatch):
    rs_rows, redrive = tools
    cfg = rs_engine.EngineConfig.defaults()
    payload = il_payload(tag="void-1", opponent_deck=VOID_DECK)
    rows, why = rs_rows.rows_for(payload, rs_rows._setup(cfg), cfg)
    assert why == "ok" and rows
    out = redrive.one(json.dumps(payload), engine=cfg)
    assert out["ok"] and out["error"] == ""
    match = void_royalesim.battles[-1]
    assert 16 in match.calls_named("reset")[0].kwargs["decks"][1]     # DarkMagic dealt to red

    monkeypatch.setenv(rs_cards.EXTRAS_ENV, "0")      # the 119-key map: skipped, as before
    monkeypatch.setattr(redrive, "_ENGINE", {})
    assert rs_rows.rows_for(payload, rs_rows._setup(cfg), cfg) == (None, "card")
    out = redrive.one(json.dumps(payload), engine=cfg)
    assert not out["ok"] and out["error"] == "card not in RoyaleSim: void"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_checks_and_lists(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(TOOLS))
    assert rs_cards.main([]) == 0
    out = capsys.readouterr().out
    assert "119 keys, sha256 ok" in out and "card map 122 keys" in out
    for key, name in EXTRAS.items():
        assert f"{key} -> {name}" in out


def test_cli_no_extras_json(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(TOOLS))
    assert rs_cards.main(["--no-extras", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["map"]["entries"] == 119 and report["map"]["extras"] == []
    assert {k: v["merged"] for k, v in report["extras"].items()} == {k: False for k in EXTRAS}


def test_cli_coverage_without_a_corpus(monkeypatch, tmp_path, capsys):
    monkeypatch.syspath_prepend(str(TOOLS))
    assert rs_cards.main(["--coverage", "--corpus", str(tmp_path)]) == 1
    assert "no part-*.parquet" in capsys.readouterr().err


def test_cli_catalogue_with_a_fake_engine(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(TOOLS))
    every = make_fake_royalesim(catalogue=[catalogue_row(n, 3, "TROOP")
                                           for n in sorted(set(rs_cards.load().values()))])
    monkeypatch.setitem(sys.modules, "royalesim", every)
    assert rs_cards.main(["--catalogue"]) == 0
    assert "every name is in it" in capsys.readouterr().out
    assert every.battles[-1].init_kwargs == rs_engine.EngineConfig.defaults().constructor_kwargs()

    monkeypatch.setitem(sys.modules, "royalesim", make_fake_royalesim())   # 16 cards only
    assert rs_cards.main(["--catalogue", "--card-table", "default", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["catalogue"]["cards"] == len(DEFAULT_CATALOGUE)
    assert report["catalogue"]["missing"]["void"] == "DarkMagic"
    assert "hog-rider" not in report["catalogue"]["missing"]


# ---------------------------------------------------------------------------
# the real engine and corpus
# ---------------------------------------------------------------------------

def _configs():
    return {"configured": rs_engine.EngineConfig.load(), "legacy": rs_engine.EngineConfig.legacy()}


@pytest.mark.engine
@pytest.mark.parametrize("which", ["configured", "legacy"])
def test_every_name_is_a_catalogue_name_under_the_card_table(which):
    import royalesim
    cfg = _configs()[which]
    catalogue = rs_engine.catalogue(cfg, royalesim)
    missing = rs_cards.missing_from_catalogue(rs_cards.load(extras=True), catalogue)
    assert missing == {}, f"names not in the catalogue under {cfg.describe()}: {missing}"


@pytest.mark.engine
@pytest.mark.parametrize("which", ["configured", "legacy"])
def test_the_engine_plays_a_match_with_the_three_extras(which, monkeypatch):
    import royalesim
    monkeypatch.syspath_prepend(str(TOOLS))
    redrive = importlib.import_module("royalesim_redrive")
    monkeypatch.setattr(redrive, "_ENGINE", {})
    cfg = _configs()[which]
    payload = il_payload(tag="extras-1", opponent_deck=EXTRA_DECK, team_forms={},
                         opponent_forms={}, opponent_levels=11)
    out = redrive.one(json.dumps(payload), engine=cfg)
    assert out["ok"], out
    assert redrive._ENGINE["royalesim"] is royalesim
    assert {EXTRAS[k] for k in EXTRAS} <= set(redrive._ENGINE["ids"])


@pytest.mark.il_replay
def test_coverage_reads_the_corpus(monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.syspath_prepend(str(TOOLS))
    files = rs_cards.corpus_parts(il_replay_dir(), parts=1)
    assert files
    cov = rs_cards.coverage(rs_cards.iter_payloads(files), rs_cards.load(extras=True),
                            watch=rs_cards.read_extras())
    assert cov["matches"] > 0 and 0 < cov["mapped"] <= cov["matches"]
    assert set(cov["watched"]) == set(EXTRAS)
