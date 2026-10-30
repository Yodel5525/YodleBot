"""E04: vocab.json's engine fields (tools/rs_rows.vocab_record), the load-time vocabulary
check (tools/rs_checkpoint.py) and RSAgent.__init__ using it (tools/rs_eval.py).

Pure tests compare fake checkpoint metas with fake engine lists; the RSAgent / load_policy
ones also need torch, the rs_rows ones numpy. The `engine`-marked test checks a real
engine's vocab.json against itself and against another card table.

U01 (docs/SEASON_UPDATES.md 6.1): another card table is a note, not a refusal, when every
card name (and unit type name, unless --live-compat) matches in id order; it stays an error
next to any name difference. The royalesim version and, on the same table, card_table_fnv1a64
are new warn-only notes. A refusal whose engine only added cards or unit types names
tools/rs_vocab.py grow.
"""
from __future__ import annotations

import ast
import importlib
import json
import math
import warnings
from pathlib import Path

import pytest

from tests.pipeline.conftest import DEFAULT_CATALOGUE, DEFAULT_UNIT_TYPES, FakeBattle

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
NAMES = tuple(str(r[0]) for r in DEFAULT_CATALOGUE)
UNITS = tuple(DEFAULT_UNIT_TYPES)
TARGET_REPORTED = "cards-160402017-20261019.1"       # what Battle.card_table() says for it


def meta_for(cards=NAMES, units=UNITS, **extra) -> dict:
    """A checkpoint meta as sim/train_tokens writes it: vocab.json plus the sizes."""
    meta = {"n_cards": len(cards) + 2, "n_units": len(units) + 2, "live_compat": False,
            "cards": {c: i + 2 for i, c in enumerate(cards)},
            "units": {u: i + 2 for i, u in enumerate(units)},
            "card_table": TARGET_REPORTED, "d": 16, "layers": 1, "heads": 2}
    meta.update(extra)
    return meta


def swapped(names, i, j):
    out = list(names)
    out[i], out[j] = out[j], out[i]
    return tuple(out)


@pytest.fixture
def rc(monkeypatch):
    """tools/rs_checkpoint.py, imported as the tools import each other, with no engine or
    drift setting from the PC and no cached engine vocabulary."""
    monkeypatch.syspath_prepend(str(TOOLS))
    module = importlib.import_module("rs_checkpoint")
    rs_engine = module.rs_engine
    for var in (rs_engine.CONFIG_ENV, *rs_engine.ENV.values(), module.ALLOW_DRIFT_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rs_engine, "_ACTIVE", None)
    monkeypatch.setattr(module, "_VOCABS", {})
    return module


def engine(rc, cards=NAMES, units=UNITS, table=TARGET_REPORTED):
    return rc.EngineVocab(tuple(cards), tuple(units), table)


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------

def test_equal_vocabularies_load_silently(rc):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert rc.check_meta(meta_for(), engine(rc)) == []
        assert rc.vocab_differences(meta_for(), engine(rc)) == ([], [])


def test_reordered_cards_raise_naming_the_index(rc):
    vocab = engine(rc, cards=swapped(NAMES, 3, 4))
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta_for(), vocab, allow_drift=False, source="best.pt")
    first = "card 3 (token 5): the checkpoint has 'IceGolemite', the engine 'IceSpirits'"
    assert exc.value.differences == [first]
    text = str(exc.value)
    assert text.startswith(f"best.pt does not fit this engine: {first}.")
    assert "(and" not in text and "--allow-vocab-drift" in text
    assert isinstance(exc.value, rc.rs_engine.EngineError)


def test_reordered_units_raise_naming_the_index(rc):
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta_for(), engine(rc, units=swapped(UNITS, 5, 6)))
    assert exc.value.differences == [
        "unit type 5 (token 7): the checkpoint has 'IceSpirit', the engine 'Musketeer'"]


def test_a_longer_unit_list_is_a_difference(rc):
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta_for(), engine(rc, units=UNITS + ("Witch",)))
    assert exc.value.differences == [
        "unit types: the checkpoint has 14, the engine 15; unit type 14 (token 16) is only "
        "in the engine: 'Witch'"]
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta_for(cards=NAMES + ("Zap2",)), engine(rc))
    assert exc.value.differences == [
        "cards: the checkpoint has 17, the engine 16; card 16 (token 18) is only in the "
        "checkpoint: 'Zap2'"]


def test_every_difference_is_listed_cards_first(rc):
    vocab = engine(rc, cards=swapped(NAMES, 0, 1), units=swapped(UNITS, 2, 3),
                   table="cards-15535.1")
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta_for(), vocab)
    diffs = exc.value.differences
    assert [d.split(" ")[0] for d in diffs] == ["card", "unit", "card"]
    assert diffs[2] == ("card table: the checkpoint's rows were built under "
                        f"{TARGET_REPORTED}, the engine runs cards-15535.1")
    assert "(and 2 more)" in str(exc.value)


@pytest.mark.parametrize("meta_table, engine_table, differs", [
    (TARGET_REPORTED, "cards-15535.1", True),
    (TARGET_REPORTED, "client160402017_20261019", False),       # arm and report: one table
    ("16.402.22", TARGET_REPORTED, False),                       # content version alias
    ("cards-15535.1", "cards-15535.2", False),                   # a revision is the table
    (None, "cards-15535.1", False),                              # the checkpoint does not say
    (TARGET_REPORTED, None, False),                              # the engine does not say
])
def test_card_table_comparison(rc, meta_table, engine_table, differs):
    """The names match here, so another table is a note (U01), never an error; with a name
    differing too it is an error (test_every_difference_is_listed_cards_first)."""
    meta = meta_for(card_table=meta_table)
    errors, notes = rc.vocab_differences(meta, engine(rc, table=engine_table))
    assert errors == []
    assert any(n.startswith("card table: ") for n in notes) == differs
    assert rc.other_arm(meta, engine(rc, table=engine_table)) == differs


def test_card_table_from_the_fingerprint_or_config(rc):
    meta = meta_for()
    del meta["card_table"]
    meta["engine_fingerprint"] = {"card_table": "cards-15535.1"}
    assert rc.meta_card_table(meta) == "cards-15535.1"
    errors, notes = rc.vocab_differences(meta, engine(rc))
    assert errors == [] and notes[0].startswith("card table: the checkpoint's rows were built "
                                                "under cards-15535.1, the engine runs ")
    assert rc.vocab_differences(meta, engine(rc, cards=swapped(NAMES, 0, 1)))[0][-1] == (
        f"card table: the checkpoint's rows were built under cards-15535.1, the engine runs "
        f"{TARGET_REPORTED}")
    del meta["engine_fingerprint"]
    meta["engine_config"] = {"card_table": "client15535"}
    assert rc.meta_card_table(meta) == "client15535"
    meta["engine_config"] = {"card_table": None}            # engine default: not recorded
    assert rc.meta_card_table(meta) is None
    assert rc.vocab_differences(meta, engine(rc)) == ([], [])


def test_allow_drift_warns_and_returns_the_differences(rc):
    vocab = engine(rc, cards=swapped(NAMES, 3, 4))
    with pytest.warns(rc.VocabDriftWarning, match=r"card 3 \(token 5\)"):
        drift = rc.check_meta(meta_for(), vocab, allow_drift=True)
    assert drift == ["card 3 (token 5): the checkpoint has 'IceGolemite', the engine 'IceSpirits'"]


def test_the_environment_allows_drift_unless_refused(rc, monkeypatch):
    vocab = engine(rc, cards=swapped(NAMES, 3, 4))
    monkeypatch.setenv(rc.ALLOW_DRIFT_ENV, "1")
    with pytest.warns(rc.VocabDriftWarning):
        assert rc.check_meta(meta_for(), vocab)                 # allow_drift=None: the env
    with pytest.raises(rc.VocabMismatch):
        rc.check_meta(meta_for(), vocab, allow_drift=False)
    monkeypatch.setenv(rc.ALLOW_DRIFT_ENV, "0")
    with pytest.raises(rc.VocabMismatch):
        rc.check_meta(meta_for(), vocab)


def test_set_allow_drift_reaches_the_environment(rc, monkeypatch):
    import os
    monkeypatch.setenv(rc.ALLOW_DRIFT_ENV, "0")      # restored after the test, whatever happens
    rc.set_allow_drift(True)
    assert os.environ[rc.ALLOW_DRIFT_ENV] == "1" and rc.resolve_allow_drift(None)
    rc.set_allow_drift(False)
    assert rc.ALLOW_DRIFT_ENV not in os.environ and not rc.resolve_allow_drift(None)
    assert rc.resolve_allow_drift(True) and not rc.allow_drift_from_env({})


def test_a_meta_without_vocabulary_is_refused(rc):
    meta = {"n_cards": 18, "n_units": 16}
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta, engine(rc))
    assert len(exc.value.differences) == 2
    assert exc.value.differences[0].startswith("the checkpoint's meta has no card vocabulary")
    assert exc.value.differences[1].startswith("the checkpoint's meta has no unit type vocabulary")
    with pytest.warns(rc.VocabDriftWarning):
        assert len(rc.check_meta(meta, engine(rc), allow_drift=True)) == 2


def test_tokens_must_be_2_to_n_plus_1(rc):
    meta = meta_for()
    meta["cards"]["Fireball"] = 99
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta, engine(rc))
    assert exc.value.differences == [
        "the checkpoint's card tokens are not 2..17: 'HogRider' has token 4"]


def test_live_compat_units_are_only_warned_about(rc):
    meta = meta_for(units=UNITS[:-1], live_compat=True)
    with pytest.warns(rc.VocabDriftWarning, match="live-compat"):
        assert rc.check_meta(meta, engine(rc)) == []
    with pytest.raises(rc.VocabMismatch):                     # its cards still count
        rc.check_meta(meta_for(cards=swapped(NAMES, 0, 1), live_compat=True), engine(rc))


# ---------------------------------------------------------------------------
# U01: another card table with the same names, and the drift notes
# ---------------------------------------------------------------------------

OTHER = "cards-15535.1"


def test_same_names_on_another_table_load_with_a_note(rc):
    """docs/SEASON_UPDATES.md 6.1: the ids mean the same cards and units, so a balance update
    no longer refuses the checkpoint; it says what it learned under."""
    vocab = engine(rc, table=OTHER)
    errors, notes = rc.vocab_differences(meta_for(), vocab)
    assert errors == [] and notes == [
        f"card table: the checkpoint's rows were built under {TARGET_REPORTED}, the engine runs "
        f"{OTHER}; its card and unit type names are the engine's in id order, so it loads, with "
        "what it learned under the other table's values"]
    with pytest.warns(rc.VocabDriftWarning, match=r"^best\.pt: card table: .* so it loads"):
        assert rc.check_meta(meta_for(), vocab, allow_drift=False, source="best.pt") == []


def test_an_appended_card_still_refuses_and_names_grow(rc):
    vocab = engine(rc, cards=NAMES + ("Witch",), table=OTHER)
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta_for(), vocab, allow_drift=False, source="runs/a/best.pt")
    assert exc.value.differences == [
        "cards: the checkpoint has 16, the engine 17; card 16 (token 18) is only in the "
        "engine: 'Witch'",
        f"card table: the checkpoint's rows were built under {TARGET_REPORTED}, the engine "
        f"runs {OTHER}"]
    text = str(exc.value)
    assert ("Every card and unit type it knows is still in the engine, which has 1 card "
            "appended (Witch): tools/rs_vocab.py grow runs/a/best.pt --out DIR carries it "
            "across") in text
    assert "Or rebuild the rows (tools/rs_rows.py)" in text and "--allow-vocab-drift" in text
    # a blob has no path to name; the same engine table changes nothing about the refusal
    with pytest.raises(rc.VocabMismatch, match="rs_vocab.py grow CKPT --out DIR"):
        rc.check_meta(meta_for(), engine(rc, cards=NAMES + ("Witch",)))


def test_a_removed_card_gets_no_grow_hint(rc):
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta_for(cards=NAMES + ("Zap2",)), engine(rc, table=OTHER))
    assert "rs_vocab" not in str(exc.value)
    assert exc.value.differences[-1].startswith("card table: ")          # still an error
    assert rc.grow_blockers(meta_for(cards=NAMES + ("Zap2",)), engine(rc)) == [
        "1 card(s) it knows are not in the engine: 'Zap2'"]
    # an unset units_mode is a types checkpoint (as vision_play reads it); vision is blocked
    assert rc.grow_blockers(meta_for(units_mode=None), engine(rc)) == []
    assert rc.grow_blockers(meta_for(units_mode="vision"), engine(rc))[0].startswith(
        "a --units vision checkpoint")


def test_a_moved_card_is_refused_and_names_grow_with_a_warning(rc):
    """grow carries cards at other ids by name (rs_vocab diff says GROW), so the refusal names
    it, as for an appended card, and warns: RoyaleSim only appends cards."""
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta_for(), engine(rc, cards=swapped(NAMES, 3, 4), table=OTHER))
    text = str(exc.value)
    assert ("Every card and unit type it knows is still in the engine, which has 2 cards moved: "
            "tools/rs_vocab.py grow CKPT --out DIR carries it across") in text
    assert "only appends cards and never renumbers one" in text
    assert exc.value.differences[-1].startswith("card table: ")          # still an error
    # an appended card gets no such warning; a pure unit move is named but not warned about
    assert "never renumbers" not in rc.grow_hint(meta_for(), engine(rc, cards=NAMES + ("W",)))
    hint = rc.grow_hint(meta_for(), engine(rc, units=swapped(UNITS, 5, 6)))
    assert "which has 2 unit types moved:" in hint and "never renumbers" not in hint
    assert rc.grow_hint(meta_for(), engine(rc, table=OTHER)) is None    # nothing to grow


def test_a_types_checkpoint_with_changed_units_refuses_on_another_table(rc):
    vocab = engine(rc, units=swapped(UNITS, 5, 6), table=OTHER)
    with pytest.raises(rc.VocabMismatch) as exc:
        rc.check_meta(meta_for(), vocab)
    assert [d.split(" ")[0] for d in exc.value.differences] == ["unit", "card"]
    # a unit type added (a new form's unit, sorted in by name): refused, grow named
    added = engine(rc, units=UNITS[:3] + ("Giant_EV1",) + UNITS[3:], table=OTHER)
    with pytest.raises(rc.VocabMismatch, match=r"1 unit type added \(Giant_EV1\)"):
        rc.check_meta(meta_for(), added)


def test_a_live_compat_checkpoint_on_another_table_only_notes(rc):
    meta = meta_for(units=UNITS[:-1], live_compat=True)
    vocab = engine(rc, units=UNITS + ("Witch",), table=OTHER)
    errors, notes = rc.vocab_differences(meta, vocab)
    assert errors == [] and len(notes) == 2
    assert "live-compat" in notes[0]
    assert notes[1].endswith("; its card names are the engine's in id order, so it loads, with "
                             "what it learned under the other table's values")
    with pytest.warns(rc.VocabDriftWarning):
        assert rc.check_meta(meta, vocab) == []
    with pytest.raises(rc.VocabMismatch):                     # its cards still count
        rc.check_meta(meta_for(cards=swapped(NAMES, 0, 1), live_compat=True), vocab)


def test_royalesim_version_drift_is_a_note(rc):
    meta = meta_for(engine_fingerprint={"version": "0.1.42", "card_table": TARGET_REPORTED},
                    version="0.1.42")
    newer = rc.EngineVocab(NAMES, UNITS, TARGET_REPORTED, {"version": "0.1.43"})
    errors, notes = rc.vocab_differences(meta, newer)
    assert errors == [] and notes == [
        "royalesim: the checkpoint's rows were built under royalesim 0.1.42, the engine is "
        "0.1.43; the ids are checked, the simulation may differ"]
    same = rc.EngineVocab(NAMES, UNITS, TARGET_REPORTED, {"version": "0.1.42"})
    assert rc.vocab_differences(meta, same) == ([], [])
    assert rc.vocab_differences(meta, engine(rc)) == ([], [])          # the engine does not say
    unknown = meta_for(version="?")                                      # vocab_record's "unknown"
    assert rc.vocab_differences(unknown, newer) == ([], [])
    assert rc.meta_version(meta_for(version="0.1.40")) == "0.1.40"     # no fingerprint
    with pytest.warns(rc.VocabDriftWarning, match="royalesim 0.1.42, the engine is 0.1.43"):
        assert rc.check_meta(meta, newer) == []


def test_card_table_fnv_drift_is_a_note_on_the_same_table(rc):
    old, new = "15bb4e60e69d05f3", "87cca9def6207621"
    meta = meta_for(engine_fingerprint={"card_table": TARGET_REPORTED, "card_table_fnv1a64": old})
    assert rc.meta_fnv(meta) == old
    moved = rc.EngineVocab(NAMES, UNITS, TARGET_REPORTED, card_table_fnv1a64=new)
    errors, notes = rc.vocab_differences(meta, moved)
    assert errors == [] and notes == [
        f"card table bytes: the checkpoint's rows were built under {TARGET_REPORTED} with "
        f"FNV-1a 64 {old}, the engine's has {new}: the same table, other values"]
    # the same bytes, an FNV recorded as an int or in upper case, or one side not saying
    for fnv in (old, int(old, 16), old.upper()):
        assert rc.vocab_differences(meta, rc.EngineVocab(NAMES, UNITS, TARGET_REPORTED,
                                                         card_table_fnv1a64=fnv)) == ([], [])
    assert rc.vocab_differences(meta, engine(rc)) == ([], [])
    # from the engine's fingerprint when the Battle did not report it
    fp = rc.EngineVocab(NAMES, UNITS, TARGET_REPORTED, {"card_table_fnv1a64": new})
    assert rc.vocab_differences(meta, fp)[1] == notes
    # another table: the table note says it; no second note on its bytes
    other = rc.EngineVocab(NAMES, UNITS, OTHER, card_table_fnv1a64=new)
    errors, notes = rc.vocab_differences(meta, other)
    assert errors == [] and [n.split(":")[0] for n in notes] == ["card table"]
    with pytest.warns(rc.VocabDriftWarning, match="card table bytes"):
        assert rc.check_meta(meta, moved) == []


class FnvBattle(FakeBattle):
    def card_table(self):
        return TARGET_REPORTED

    def card_table_fnv1a64(self):
        return "87CCA9DEF6207621"


def test_engine_vocab_reads_the_battles_fnv(rc):
    vocab = rc.engine_vocab(FnvBattle())
    assert (vocab.card_table_fnv1a64, vocab.fnv) == ("87cca9def6207621", "87cca9def6207621")
    assert "(fnv1a64 87cca9def6207621)" in vocab.describe()
    assert rc.engine_vocab(ReportingBattle()).fnv is None
    rows = meta_for(engine_fingerprint={"card_table_fnv1a64": "15bb4e60e69d05f3"})
    assert rc.EngineVocab.from_vocab(rows).fnv == "15bb4e60e69d05f3"
    assert rc.vocab_differences(rows, vocab)[1][0].startswith("card table bytes: ")


def test_name_changes(rc):
    same = rc.name_changes(NAMES, NAMES)
    assert same.same and same.describe("card") == "the same"
    grown = rc.name_changes(NAMES, NAMES + ("Witch", "Bats"))
    assert (grown.added, grown.removed, grown.moved, grown.appended) == (("Witch", "Bats"), (), (),
                                                                         True)
    assert grown.describe("card") == "2 cards appended (Witch, Bats)"
    inserted = rc.name_changes(UNITS, UNITS[:3] + ("Giant_EV1",) + UNITS[3:])
    assert inserted.added == ("Giant_EV1",) and not inserted.appended
    assert len(inserted.moved) == len(UNITS) - 3 and inserted.moved[0] == (UNITS[3], 3, 4)
    assert inserted.describe("unit type") == f"1 unit type added (Giant_EV1), {len(UNITS) - 3} moved"
    gone = rc.name_changes(NAMES, NAMES[1:])
    assert gone.removed == (NAMES[0],) and len(gone.moved) == len(NAMES) - 1
    many = rc.name_changes((), NAMES[:5])
    assert many.describe("card") == ("5 cards appended (Cannon, Fireball, HogRider, ... 2 more)")


# ---------------------------------------------------------------------------
# the engine's vocabulary
# ---------------------------------------------------------------------------

class ReportingBattle(FakeBattle):
    def card_table(self):
        return "cards-15535.1"


def test_engine_vocab_from_a_config_is_built_once(rc, fake_royalesim):
    cfg = rc.rs_engine.EngineConfig.defaults()
    vocab = rc.engine_vocab(cfg)
    assert (vocab.cards, vocab.units) == (NAMES, UNITS)
    assert vocab.card_table == "client160402017_20261019"        # the fake reports none
    assert vocab.fingerprint["version"] == "0.0.0+fake"
    built = len(fake_royalesim.battles)
    assert built == 1 and fake_royalesim.battles[0].calls[0].args == (0, 0)   # undelayed
    assert rc.engine_vocab(cfg.delayed(21)) is vocab and len(fake_royalesim.battles) == built
    rc.rs_engine.set_active(rc.rs_engine.EngineConfig(card_table="client15535"))
    assert rc.engine_vocab(None).card_table == "client15535"     # None: the active config
    assert "16 cards, 14 unit types" in vocab.describe()


def test_engine_vocab_from_a_battle_or_a_vocab(rc):
    battle = ReportingBattle()
    vocab = rc.engine_vocab(battle)
    assert (vocab.cards, vocab.units, vocab.card_table) == (NAMES, UNITS, "cards-15535.1")
    assert rc.engine_vocab(vocab) is vocab
    from_rows = rc.engine_vocab(meta_for())                      # a vocab.json-like mapping
    assert (from_rows.cards, from_rows.units, from_rows.card_table) == \
        (NAMES, UNITS, TARGET_REPORTED)
    # rows of one table, engine of another: the same names load with a note (U01) ...
    with pytest.warns(rc.VocabDriftWarning, match="card table"):
        assert rc.check_meta(meta_for(), battle) == []
    with pytest.raises(rc.VocabMismatch, match="card table"):    # ... other names do not
        rc.check_meta(meta_for(cards=swapped(NAMES, 0, 1)), battle)


# ---------------------------------------------------------------------------
# vocab.json (tools/rs_rows.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def rr(rc):
    pytest.importorskip("numpy")
    return importlib.import_module("rs_rows")


def _rows_inputs(rc):
    cat = [list(r) for r in DEFAULT_CATALOGUE]
    units = list(DEFAULT_UNIT_TYPES)
    cfg = rc.rs_engine.EngineConfig.defaults()
    fp = {"version": "0.0.0+fake", "card_table": TARGET_REPORTED, "id": "f00d"}
    return cat, units, cfg, fp


def test_vocab_record_keeps_the_legacy_keys_first(rc, rr):
    cat, units, cfg, fp = _rows_inputs(rc)
    card_map = {"hog-rider": "HogRider"}
    old = {                                 # rs_rows.main's vocab.json before E04
        "n_cards": len(cat) + 2,
        "n_units": len(units) + 2,
        "live_compat": False,
        "cards": {row[0]: i + 2 for i, row in enumerate(cat)},
        "units": {u: i + 2 for i, u in enumerate(units)},
        "engine": "royalesim", "version": "0.1.42",
        "engine_config": cfg.to_json(), "engine_fingerprint": fp, "card_map": card_map}
    new = rr.vocab_record(cat, units, cfg, fp, card_map, False, version="0.1.42")
    assert list(new)[:len(old)] == list(old)
    assert {k: new[k] for k in old} == old
    assert json.dumps(new).startswith(json.dumps(old)[:-1] + ", ")      # the same bytes first
    assert {k: new[k] for k in new if k not in old} == {
        "card_table": TARGET_REPORTED, "schema": "rs_rows/legacy",
        "obs_delay": {"ticks": 0, "jitter": 0, "opp_lag": 0},
        "wait_keep_at_build": None, "wait_every": rr.WAIT_EVERY}
    live = rr.vocab_record(cat, units, cfg, fp, {}, True)
    assert live["n_units"] == len(cat) + 2 and live["live_compat"] is True
    assert live["version"] == "0.0.0+fake"                         # from the fingerprint
    later = rr.vocab_record(cat, units, cfg, fp, {}, False, schema="rs_rows/v2", obs_delay=21,
                            obs_delay_jitter=2, opp_lag=4, wait_keep_at_build=0.25)
    assert later["obs_delay"] == {"ticks": 21, "jitter": 2, "opp_lag": 4}
    assert (later["schema"], later["wait_keep_at_build"]) == ("rs_rows/v2", 0.25)


def test_rs_rows_main_writes_vocab_record():
    tree = ast.parse((TOOLS / "rs_rows.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {n.func.id for n in ast.walk(main)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "vocab_record" in called


def test_the_rows_vocabulary_survives_into_a_checkpoint(rc, rr, fake_royalesim):
    pytest.importorskip("torch")
    tt = importlib.import_module("sim.train_tokens")
    cat, units, cfg, fp = _rows_inputs(rc)
    meta = tt.checkpoint_meta(rr.vocab_record(cat, units, cfg, fp, {}, False), 16, 1, 2)
    assert rc.check_meta(meta, cfg) == []
    other = rc.rs_engine.EngineConfig(card_table="client15535")
    with pytest.warns(rc.VocabDriftWarning, match="card table"):    # the fake's names: a note
        assert rc.check_meta(meta, other) == []
    meta["cards"] = {c: i + 2 for i, c in enumerate(swapped(NAMES, 0, 1))}
    with pytest.raises(rc.VocabMismatch, match="card table"):
        rc.check_meta(meta, other)


# ---------------------------------------------------------------------------
# load_policy and RSAgent (torch)
# ---------------------------------------------------------------------------

@pytest.fixture
def torch(rc):
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    yield torch
    torch.set_num_threads(threads)          # RSAgent sets 1 for the process


def save_checkpoint(torch, path: Path, meta: dict):
    from sim.model_tokens import TokenPolicy
    torch.manual_seed(0)
    model = TokenPolicy(meta["n_cards"], meta["n_units"], d=meta["d"], layers=meta["layers"],
                        heads=meta["heads"])
    torch.save({"model": model.state_dict(), "meta": meta, "epoch": 0}, path)
    return model


def test_load_policy_builds_the_checked_model(rc, torch, tmp_path):
    path = tmp_path / "best.pt"
    saved = save_checkpoint(torch, path, meta_for())
    model, meta, drift = rc.load_policy(path, engine(rc))
    assert (meta, drift) == (meta_for(), []) and not model.training
    state = model.state_dict()
    assert all(torch.equal(v, state[k]) for k, v in saved.state_dict().items())
    blob = {"model": saved.state_dict(), "meta": meta_for()}
    assert rc.load_policy(blob, engine(rc)).meta == meta_for()      # an already loaded blob
    assert rc.check_file(path, engine(rc)) == []
    with pytest.raises(rc.VocabMismatch, match=r"card 3 \(token 5\)"):
        rc.load_policy(path, engine(rc, cards=swapped(NAMES, 3, 4)))


@pytest.fixture
def rs_eval(rc):
    pytest.importorskip("numpy")
    return importlib.import_module("rs_eval")


def test_rsagent_loads_through_the_check(rc, torch, rs_eval, fake_royalesim, tmp_path):
    rs_engine = rc.rs_engine
    rs_engine.set_active(rs_engine.EngineConfig.defaults())
    good = tmp_path / "good" / "best.pt"
    good.parent.mkdir()
    save_checkpoint(torch, good, meta_for())
    agent = rs_eval.RSAgent(str(good), seed=0)
    assert agent.meta == meta_for() and agent.vocab_drift == [] and agent.live is False
    assert agent.wait_bias == pytest.approx(math.log(1 / 0.35))     # no args.json
    assert not agent.model.training

    bad = tmp_path / "bad" / "best.pt"
    bad.parent.mkdir()
    save_checkpoint(torch, bad, meta_for(cards=swapped(NAMES, 3, 4)))
    with pytest.raises(rc.VocabMismatch, match=r"card 3 \(token 5\)"):
        rs_eval.RSAgent(str(bad), seed=0)
    with pytest.warns(rc.VocabDriftWarning):
        drifted = rs_eval.RSAgent(str(bad), seed=0, allow_drift=True)
    assert len(drifted.vocab_drift) == 1

    # the agent's own engine, when given, is the one checked: another table with the same
    # names loads with a note (U01); with other names it refuses and names the table
    other = rs_engine.EngineConfig(card_table="client15535")
    with pytest.warns(rc.VocabDriftWarning, match="card table"):
        assert rs_eval.RSAgent(str(good), engine=other).vocab_drift == []
    with pytest.raises(rc.VocabMismatch, match="card table"):
        rs_eval.RSAgent(str(bad), engine=other)


def test_rs_checkpoint_builds_battles_only_through_rs_engine():
    tree = ast.parse((TOOLS / "rs_checkpoint.py").read_text(encoding="utf-8"))
    attrs = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "Battle" not in attrs and "make_battle" in attrs


def test_rsagent_no_longer_loads_checkpoints_itself():
    tree = ast.parse((TOOLS / "rs_eval.py").read_text(encoding="utf-8"))
    agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RSAgent")
    init = next(n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    attrs = {n.func.attr for n in ast.walk(init)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "load_policy" in attrs and "load" not in attrs and "load_state_dict" not in attrs


def test_cli_checks_checkpoints(rc, torch, fake_royalesim, tmp_path, capsys):
    good, bad = tmp_path / "good.pt", tmp_path / "bad.pt"
    save_checkpoint(torch, good, meta_for())
    save_checkpoint(torch, bad, meta_for(units=swapped(UNITS, 5, 6)))
    assert rc.main([str(good)]) == 0
    assert f"ok       {good}" in capsys.readouterr().out
    assert rc.main([str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert f"MISMATCH {bad} does not fit this engine: unit type 5 (token 7)" in out
    assert rc.main([str(good), "--engine-config", str(tmp_path / "none.json")]) == 2
    # U01: another table with the same names is ok, its note printed under it
    noted = tmp_path / "noted.pt"
    save_checkpoint(torch, noted, meta_for(card_table=OTHER))
    capsys.readouterr()
    assert rc.main([str(noted)]) == 0
    out = capsys.readouterr().out
    assert f"ok       {noted} (1 note)\n" in out
    assert (f"  note   card table: the checkpoint's rows were built under {OTHER}, the engine "
            "runs client160402017_20261019; its card and unit type names") in out


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

@pytest.mark.engine
def test_engine_rows_vocabulary_fits_its_engine_only(rc, rr):
    import royalesim
    rs_engine = rc.rs_engine
    cfg = rs_engine.EngineConfig.load(env={}).undelayed()
    try:
        battle = rs_engine.make_battle(cfg, royalesim)
    except rs_engine.EngineError as exc:
        pytest.skip(f"the engine refused data/engine.json's config: {exc}")
    if not hasattr(battle, "card_table"):
        pytest.skip(f"royalesim {royalesim.__version__} has no card tables (0.1.42 adds them)")
    fp = rs_engine.fingerprint(battle, royalesim, cfg)
    vocab = rr.vocab_record(json.loads(battle.catalogue_json()),
                            json.loads(battle.unit_types_json()), cfg, fp, {}, False,
                            version=royalesim.__version__)
    assert vocab["card_table"] == battle.card_table()
    assert rc.check_meta(vocab, cfg) == []
    other = rs_engine.EngineConfig(card_table="client15535", level=11, tower_level=11)
    # The tables' unit lists differ (237 unit types on 15535), so this refuses and names the
    # table; were every name the same, the other table would be a note (U01).
    errors, notes = rc.vocab_differences(vocab, rc.engine_vocab(other))
    if errors:
        with pytest.raises(rc.VocabMismatch) as exc:
            rc.check_meta(vocab, other)
        assert any(d.startswith("card table: ") for d in exc.value.differences)
    else:
        assert any(n.startswith("card table: ") for n in notes)
    # the engine's last card appended after a checkpoint's: refused, grow named
    short = dict(vocab, cards={c: t for c, t in vocab["cards"].items()
                               if t < len(vocab["cards"]) + 1})
    with pytest.raises(rc.VocabMismatch, match="rs_vocab.py grow"):
        rc.check_meta(short, cfg)
