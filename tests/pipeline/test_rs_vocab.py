"""U14: tools/rs_vocab.py (docs/SEASON_UPDATES.md 6.1, 6.2): diff, grow, reset-cold and check.

The diff and check verdicts are pure (fake metas against fake engine lists). grow and
reset-cold run on tiny TokenPolicy checkpoints (d 16, one layer; torch importorskip): an appended
card, a unit type sorted in by name, a --live-compat checkpoint, a v2 model with a critic, the
refusals (a removed card, weights that do not match the meta, an unknown tensor), function
preservation, the args.json carried, and a strict load of what was written. The CLI runs
against the fake royalesim with one card added to its catalogue. The `engine`-marked test grows
a checkpoint missing the real engine's last card.
"""
from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path

import pytest

from tests.pipeline.conftest import DEFAULT_CATALOGUE, DEFAULT_UNIT_TYPES, catalogue_row

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
NAMES = tuple(str(r[0]) for r in DEFAULT_CATALOGUE)
UNITS = tuple(DEFAULT_UNIT_TYPES)
TARGET_REPORTED = "cards-160402017-20261019.1"
OTHER = "cards-15535.1"
CARD_EMB, UNIT_EMB = "card_emb.weight", "unit_emb.weight"
WITH_EVO = UNITS[:3] + ("Giant_EV1",) + UNITS[3:]        # a new form's unit, sorted in by name


def meta_for(cards=NAMES, units=UNITS, *, live=False, **extra) -> dict:
    """A checkpoint meta as sim/train_tokens writes it (vocab.json plus the sizes)."""
    meta = {"n_cards": len(cards) + 2, "n_units": (len(cards) if live else len(units)) + 2,
            "live_compat": live,
            "cards": {c: i + 2 for i, c in enumerate(cards)},
            "units": {u: i + 2 for i, u in enumerate(units)},
            "card_table": TARGET_REPORTED, "d": 16, "layers": 1, "heads": 2}
    meta.update(extra)
    return meta


@pytest.fixture
def rv(monkeypatch):
    """tools/rs_vocab.py as the tools import each other, with no engine or drift setting from
    the PC and no cached engine vocabulary."""
    monkeypatch.syspath_prepend(str(TOOLS))
    module = importlib.import_module("rs_vocab")
    rs_engine = module.rs_engine
    for var in (rs_engine.CONFIG_ENV, *rs_engine.ENV.values(), module.RC.ALLOW_DRIFT_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rs_engine, "_ACTIVE", None)
    monkeypatch.setattr(module.RC, "_VOCABS", {})
    return module


def engine(rv, cards=NAMES, units=UNITS, table=TARGET_REPORTED, **kw):
    return rv.RC.EngineVocab(tuple(cards), tuple(units), table, **kw)


# ---------------------------------------------------------------------------
# diff (pure)
# ---------------------------------------------------------------------------

def test_token_map(rv):
    assert rv.token_map(["A", "B"], ["B", "A", "C"]) == [0, 1, 3, 2, None]
    assert rv.token_map([], []) == [0, 1]


@pytest.mark.parametrize("case, meta_kw, engine_kw, verdict", [
    ("same", {}, {}, "SAME"),
    ("another table", {}, {"table": OTHER}, "LOADS"),
    ("another royalesim", {"engine_fingerprint": {"version": "0.1.42"}},
     {"fingerprint": {"version": "0.1.43"}}, "LOADS"),
    ("a card appended", {}, {"cards": NAMES + ("Witch",)}, "GROW"),
    ("a unit type sorted in", {}, {"units": WITH_EVO}, "GROW"),
    ("live-compat, other units", {"live": True}, {"units": WITH_EVO}, "LOADS"),
    ("a card removed", {}, {"cards": NAMES[:-1] + ("Witch",)}, "BREAKING"),
    ("a unit type removed", {}, {"units": UNITS[:-1]}, "BREAKING"),
    ("vision units", {"units_mode": "vision"}, {}, "BREAKING"),
    ("no vocabulary", {"cards": {}}, {}, "BREAKING"),
])
def test_diff_verdicts(rv, case, meta_kw, engine_kw, verdict):
    diff = rv.vocab_diff(meta_for(**meta_kw), engine(rv, **engine_kw))
    assert diff.verdict == verdict, case
    assert bool(diff.blockers) == (verdict == "BREAKING")
    assert diff.drift == (case in ("another table", "another royalesim"))


def test_diff_lines_and_json(rv):
    diff = rv.vocab_diff(meta_for(), engine(rv, cards=NAMES + ("Witch",), units=WITH_EVO,
                                             table=OTHER))
    lines = diff.lines()
    assert lines[0] == "cards       1 card appended (Witch)"
    assert lines[1] == f"unit types  1 unit type added (Giant_EV1), {len(UNITS) - 3} moved"
    assert lines[2] == f"card table  {TARGET_REPORTED} -> {OTHER}"
    assert lines[3] == "fnv1a64     n/a -> n/a (not compared)"
    assert lines[-1] == "verdict     GROW"
    doc = diff.to_json()
    assert doc["verdict"] == "GROW" and doc["cards"] == {"added": ["Witch"], "removed": [],
                                                         "moved": 0, "appended": True}
    assert doc["other_table"] and json.loads(json.dumps(doc)) == doc
    same = rv.vocab_diff(meta_for(), engine(rv)).lines()
    assert same[2] == f"card table  {TARGET_REPORTED} (the same)"
    gone = rv.vocab_diff(meta_for(), engine(rv, cards=NAMES[1:])).lines()
    assert gone[-2:] == ["verdict     BREAKING",
                         "  blocked: 1 card(s) it knows are not in the engine: 'Cannon'"]


# ---------------------------------------------------------------------------
# check (pure)
# ---------------------------------------------------------------------------

def test_check_verdicts(rv):
    ok = rv.check({"meta": meta_for()}, engine(rv), source="best.pt")
    assert (ok.verdict, ok.errors, ok.notes, ok.fix, ok.cold) == ("OK", [], [], None, [])
    drift = rv.check({"meta": meta_for()}, engine(rv, table=OTHER))
    assert drift.verdict == "ENGINE_DRIFT" and len(drift.notes) == 1 and drift.fix is None
    grow = rv.check({"meta": meta_for()}, engine(rv, cards=NAMES + ("Witch",)), source="best.pt")
    assert grow.verdict == "REFUSED" and grow.errors
    assert grow.fix.startswith("python tools/rs_vocab.py grow best.pt --out DIR")
    gone = rv.check({"meta": meta_for()}, engine(rv, cards=NAMES[:-1]))
    assert gone.verdict == "REFUSED" and gone.fix.startswith("rebuild the rows")
    assert "unexpected" not in grow.fix
    # cards at other ids: grow carries them by name, as rs_checkpoint's refusal says, with a
    # warning (RoyaleSim only appends cards)
    moved = list(NAMES)
    moved[3], moved[4] = moved[4], moved[3]
    swap = rv.check({"meta": meta_for()}, engine(rv, cards=moved), source="best.pt")
    assert swap.verdict == "REFUSED" and swap.diff.verdict == "GROW"
    assert swap.fix.startswith("python tools/rs_vocab.py grow best.pt --out DIR")
    assert "cards at other ids are unexpected" in swap.fix
    assert "tools/rs_vocab.py grow" in rv.RC.grow_hint(meta_for(), engine(rv, cards=moved))


def test_check_cold_cards_in_the_pool(rv):
    meta = meta_for(cold_cards=["Zap", "NotACard"])
    cold = rv.check({"meta": meta}, engine(rv), matters={"Zap", "HogRider"})
    assert (cold.verdict, cold.cold, cold.cold_in_pool) == ("COLD", ["Zap"], ["Zap"])
    assert "reset-cold" in cold.fix
    elsewhere = rv.check({"meta": meta}, engine(rv), matters={"HogRider"})
    assert (elsewhere.verdict, elsewhere.cold) == ("OK", ["Zap"])
    given = rv.check({"meta": meta_for()}, engine(rv), cold=["Giant"], matters={"Giant"})
    assert given.verdict == "COLD"
    refused = rv.check({"meta": meta}, engine(rv, cards=NAMES[:-1]), matters={"Zap"})
    assert refused.verdict == "REFUSED"                    # refused beats cold
    lines = cold.lines()
    assert lines[0] == "COLD         checkpoint" and "  cold:    Zap (in the pool: Zap)" in lines
    assert cold.to_json()["diff"]["verdict"] == "SAME"


# ---------------------------------------------------------------------------
# grow and reset-cold (torch)
# ---------------------------------------------------------------------------

@pytest.fixture
def torch(rv):
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    yield torch
    torch.set_num_threads(threads)


def make_blob(torch, meta: dict, seed: int = 0) -> dict:
    from sim.model_tokens import policy_from_meta
    torch.manual_seed(seed)
    model = policy_from_meta(meta)
    return {"model": model.state_dict(), "meta": meta, "epoch": 0}


def test_grow_an_appended_card(rv, torch):
    blob = make_blob(torch, meta_for())
    vocab = engine(rv, cards=NAMES + ("Witch",))
    with pytest.raises(rv.RC.VocabMismatch, match="rs_vocab.py grow"):
        rv.RC.check_meta(blob["meta"], vocab)
    carried = rv.grow(blob, vocab, source="best.pt")
    old, new = blob["model"], carried.blob["model"]
    assert new[CARD_EMB].shape == (19, 16)
    assert torch.equal(new[CARD_EMB][:18], old[CARD_EMB])          # every row by name
    assert torch.equal(new[CARD_EMB][18], old[CARD_EMB][1])        # Witch: the unknown row
    assert torch.equal(new[UNIT_EMB], old[UNIT_EMB])
    assert all(new[k] is old[k] for k in old if k not in (CARD_EMB, UNIT_EMB))
    assert carried.max_delta < 1e-6 and carried.notes == []
    meta = carried.blob["meta"]
    assert (meta["n_cards"], meta["n_units"], meta["cards"]["Witch"]) == (19, 16, 18)
    assert meta["card_table"] == TARGET_REPORTED                    # what it learned under
    assert meta["cold_cards"] == ["Witch"]
    record = meta["vocab_carry"][-1]
    assert (record["op"], record["from"], record["cards_added"], record["units_added"]) == \
        ("grow", "best.pt", ["Witch"], [])
    assert record["max_delta"] == carried.max_delta and carried.blob["epoch"] == 0
    assert blob["meta"] == meta_for()                               # the source is untouched
    with warnings.catch_warnings():         # (torch warns about norm_first on its own)
        warnings.simplefilter("error", rv.RC.VocabDriftWarning)
        assert rv.RC.load_policy(carried.blob, vocab, allow_drift=False).drift == []


def test_grow_onto_another_table_keeps_the_drift_note(rv, torch):
    carried = rv.grow(make_blob(torch, meta_for()), engine(rv, cards=NAMES + ("Witch",),
                                                            table=OTHER))
    assert len(carried.notes) == 1 and "card table: " in carried.notes[0]
    with pytest.warns(rv.RC.VocabDriftWarning, match="so it loads"):
        rv.RC.load_policy(carried.blob, engine(rv, cards=NAMES + ("Witch",), table=OTHER))


def test_grow_carries_unit_types_by_name(rv, torch):
    blob = make_blob(torch, meta_for())
    carried = rv.grow(blob, engine(rv, units=WITH_EVO))
    old, new = blob["model"][UNIT_EMB], carried.blob["model"][UNIT_EMB]
    assert new.shape == (len(WITH_EVO) + 2, 16)
    for j, name in enumerate(WITH_EVO):
        want = old[1] if name == "Giant_EV1" else old[UNITS.index(name) + 2]
        assert torch.equal(new[j + 2], want), name
    assert torch.equal(new[:2], old[:2])
    assert torch.equal(carried.blob["model"][CARD_EMB], blob["model"][CARD_EMB])
    assert carried.max_delta < 1e-6 and carried.diff.units.added == ("Giant_EV1",)
    assert carried.blob["meta"]["units"]["HogRider"] == 6 and carried.blob["meta"]["n_units"] == 17
    assert carried.blob["meta"]["cold_cards"] == []


def test_grow_a_live_compat_checkpoint_maps_units_with_the_cards(rv, torch):
    blob = make_blob(torch, meta_for(live=True))
    vocab = engine(rv, cards=NAMES + ("Witch",), units=UNITS + ("Witch",))
    carried = rv.grow(blob, vocab)
    old, new = blob["model"], carried.blob["model"]
    for key in (CARD_EMB, UNIT_EMB):
        assert new[key].shape == (19, 16)
        assert torch.equal(new[key][:18], old[key]) and torch.equal(new[key][18], old[key][1])
    meta = carried.blob["meta"]
    assert (meta["n_cards"], meta["n_units"], meta["live_compat"]) == (19, 19, True)
    assert meta["units"] == {u: i + 2 for i, u in enumerate(UNITS + ("Witch",))}
    assert carried.max_delta < 1e-6 and carried.record["units_added"] == []


def test_grow_a_v2_model_with_its_critic(rv, torch):
    blob = make_blob(torch, meta_for(schema="rs_rows/v2"))
    assert any(k.startswith("value2.") for k in blob["model"])
    carried = rv.grow(blob, engine(rv, cards=NAMES + ("Witch", "Bats"), units=WITH_EVO))
    assert carried.max_delta < 1e-6
    model = rv.RC.load_policy(carried.blob, engine(rv, cards=NAMES + ("Witch", "Bats"),
                                                   units=WITH_EVO)).model
    assert model.v2 and model.has_critic and model.card_emb.weight.shape[0] == 20


def test_grow_refuses_what_it_cannot_carry(rv, torch):
    blob = make_blob(torch, meta_for())
    with pytest.raises(rv.VocabError, match=r"BREAKING.*'Zap'"):
        rv.grow(blob, engine(rv, cards=NAMES[:-1] + ("Witch",)))
    with pytest.raises(rv.VocabError, match="BREAKING"):
        rv.grow(make_blob(torch, meta_for(units_mode="vision")), engine(rv))
    # weights of another vocabulary than the meta says
    wrong = dict(make_blob(torch, meta_for(cards=NAMES + ("X", "Y"))), meta=meta_for())
    with pytest.raises(rv.VocabError, match=r"card_emb.weight is \(20, 16\), its meta's "
                                            r"vocabulary says 18 rows"):
        rv.grow(wrong, engine(rv, cards=NAMES + ("Witch",)))
    # a tensor the model does not have (a vocabulary-sized head grow would not carry)
    extra = dict(blob, model={**blob["model"], "card_head.weight": torch.zeros(18, 16)})
    with pytest.raises(rv.VocabError, match="does not load as the TokenPolicy"):
        rv.grow(extra, engine(rv, cards=NAMES + ("Witch",)))
    state = dict(blob["model"], **{CARD_EMB: torch.zeros(19, 16)})
    with pytest.raises(rv.VocabError, match=r"shape differs: \['card_emb.weight'\].*grows only"):
        rv._check_shapes(meta_for(), state)


def test_grow_refuses_a_carried_model_that_is_not_the_same_function(rv, torch, monkeypatch):
    blob = make_blob(torch, meta_for())
    take = rv.take_rows

    def shuffled(table, mapping, unknown=1):
        out = take(table, mapping, unknown)
        out[2:] = out[2:].flip(0)                          # rows of the wrong names
        return out

    monkeypatch.setattr(rv, "take_rows", shuffled)
    with pytest.raises(rv.VocabError, match="moved by .* nothing written"):
        rv.grow(blob, engine(rv, cards=NAMES + ("Witch",)))


def test_grow_refuses_rows_that_are_not_the_old_rows_by_name(rv, torch, monkeypatch):
    """A wrong token map moves the rows and the probe's tokens alike, so the outputs do not
    change; the by-name check reads the two metas and sees it."""
    blob = make_blob(torch, meta_for())
    vocab = engine(rv, cards=NAMES + ("Witch",))
    good = rv.grow(blob, vocab)
    assert rv.rows_by_name(blob["model"], good.blob["model"], blob["meta"],
                           good.blob["meta"]) == []
    token_map = rv.token_map

    def crossed(old, new):
        out = token_map(old, new)
        out[2], out[3] = out[3], out[2]                    # Cannon and Fireball crossed
        return out

    monkeypatch.setattr(rv, "token_map", crossed)
    with pytest.raises(rv.VocabError, match=r"not the old rows by name \(card_emb.weight "
                                            r"'Cannon', card_emb.weight 'Fireball', "
                                            r"unit_emb.weight '[A-Za-z]+', "):
        rv.grow(blob, vocab)
    # the probe alone passes it: the same function on crossed tokens
    old = rv._strict_model(blob["meta"], blob["model"], "old")
    card_map = crossed(NAMES, vocab.cards)
    state = dict(blob["model"], **{CARD_EMB: rv.take_rows(blob["model"][CARD_EMB], card_map)})
    new = rv._strict_model(meta_for(cards=NAMES + ("Witch",)), state, "new")
    assert rv.preservation_delta(old, new, card_map, list(range(16))) < 1e-6


def test_preservation_delta_sees_one_changed_row(rv, torch):
    import copy
    from sim.model_tokens import policy_from_meta
    meta = meta_for()
    torch.manual_seed(0)
    old = policy_from_meta(meta)
    new = copy.deepcopy(old)
    with torch.no_grad():                   # not a constant shift, which LayerNorm would hide
        new.card_emb.weight[5] += torch.linspace(-1.0, 1.0, 16)
    identity, units = list(range(18)), list(range(16))
    assert rv.preservation_delta(old, copy.deepcopy(old), identity, units) == 0.0
    assert rv.preservation_delta(old, new, identity, units) > 1e-3
    assert rv.preservation_delta(old, new, identity, units, skip_cards=[5]) == 0.0


def test_probe_batch_covers_every_token(rv, torch):
    from sim.model_tokens import TokenPolicy
    spec = TokenPolicy(18, 16, d=16, layers=1, heads=2, schema="rs_rows/v2").schema
    batch = rv.probe_batch(spec, list(range(1, 18)), list(range(1, 16)), rows=8)
    assert set(batch) == set(spec.model_inputs)
    assert set(batch["hand"].flatten().tolist()) == set(range(1, 18))   # 32 slots, no padding
    assert set(batch["unit_ids"].flatten().tolist()) <= set(range(16))
    assert set(range(1, 16)) <= set(batch["unit_ids"].flatten().tolist())
    assert batch["ability_card"].shape == (8, 2) and batch["ability_state"].shape == (8, 2, 5)
    assert batch["unit_feats"].dtype == torch.float32 and batch["next"].shape == (8,)
    # one place per token is never padded: a pool as large as the column fills it exactly
    n = batch["unit_ids"][0].numel()
    one = rv.probe_batch(spec, [2], list(range(1, n + 1)), rows=1)
    assert sorted(one["unit_ids"].flatten().tolist()) == list(range(1, n + 1))


def test_reset_cold(rv, torch):
    blob = make_blob(torch, meta_for())
    carried = rv.reset_cold(blob, ["MiniPekka"], source="best.pt")
    t = NAMES.index("MiniPekka") + 2
    old, new = blob["model"][CARD_EMB], carried.blob["model"][CARD_EMB]
    assert torch.equal(new[t], old[1]) and not torch.equal(old[t], old[1])
    keep = [i for i in range(18) if i != t]
    assert torch.equal(new[keep], old[keep])
    assert torch.equal(carried.blob["model"][UNIT_EMB], blob["model"][UNIT_EMB])
    assert carried.max_delta == 0.0
    meta = carried.blob["meta"]
    assert meta["cold_cards"] == ["MiniPekka"] and meta["vocab_carry"][-1]["op"] == "reset-cold"
    assert meta["cards"] == blob["meta"]["cards"] and old is not new
    # check finds the row without the meta saying so
    quiet = dict(carried.blob, meta={k: v for k, v in meta.items() if k != "cold_cards"})
    assert rv.check(quiet, engine(rv), matters={"MiniPekka"}).cold == ["MiniPekka"]
    # a unit type too (--units types), and a live-compat checkpoint's unit row with its card
    typed = rv.reset_cold(blob, ["MiniPekka"], units=["MiniPekka"])
    u = UNITS.index("MiniPekka") + 2
    assert torch.equal(typed.blob["model"][UNIT_EMB][u], blob["model"][UNIT_EMB][1])
    live = make_blob(torch, meta_for(live=True))
    lived = rv.reset_cold(live, ["MiniPekka"])
    assert torch.equal(lived.blob["model"][UNIT_EMB][t], live["model"][UNIT_EMB][1])
    # what each row was: its own, marked cold, or already the unknown row
    assert carried.record["was"] == {"MiniPekka": "its own row"}
    marked = make_blob(torch, meta_for(cold_cards=["Zap"]))
    assert rv.reset_cold(marked, ["Zap", "Knight"]).record["was"] == {
        "Knight": "its own row", "Zap": "marked cold"}
    assert rv.reset_cold(carried.blob, ["MiniPekka"]).record["was"] == {
        "MiniPekka": "the unknown row"}


def test_reset_cold_refuses_the_deck_unless_allowed(rv, torch):
    pytest.importorskip("numpy")                      # Hog 2.6 from tools/rs_rows.py
    blob = make_blob(torch, meta_for())
    assert rv.deck_cards() == frozenset({"Cannon", "Fireball", "HogRider", "IceGolemite",
                                         "IceSpirits", "Musketeer", "Skeletons", "Log"})
    with pytest.raises(rv.VocabError, match=r"HogRider is in Hog 2\.6, which every row trains.*"
                                            r"--allow-deck"):
        rv.reset_cold(blob, ["Zap", "HogRider"])
    with pytest.raises(rv.VocabError, match="Cannon, HogRider are in Hog 2.6"):
        rv.reset_cold(blob, ["HogRider", "Cannon"])
    hog = NAMES.index("HogRider") + 2
    allowed = rv.reset_cold(blob, ["HogRider"], allow_protected=True)
    assert torch.equal(allowed.blob["model"][CARD_EMB][hog], blob["model"][CARD_EMB][1])
    # another protected set
    with pytest.raises(rv.VocabError, match="Zap is in the protected cards, so the model"):
        rv.reset_cold(blob, ["Zap"], protected={"Zap"})
    assert rv.reset_cold(blob, ["HogRider"], protected=()).record["cards"] == ["HogRider"]


def test_reset_cold_refusals(rv, torch):
    blob = make_blob(torch, meta_for())
    with pytest.raises(rv.VocabError, match=r"not in its card vocabulary: \['Nope'\]"):
        rv.reset_cold(blob, ["Nope"])
    with pytest.raises(rv.VocabError, match="at least one card"):
        rv.reset_cold(blob, [])
    with pytest.raises(rv.VocabError, match=r"not in its unit vocabulary: \['Nope'\]"):
        rv.reset_cold(blob, ["Zap"], units=["Nope"])
    with pytest.raises(rv.VocabError, match="--units"):
        rv.reset_cold(make_blob(torch, meta_for(live=True)), ["Zap"], units=["Knight"])


def test_write_carries_args_json_and_loads_strictly(rv, torch, tmp_path):
    rs_decide = importlib.import_module("rs_decide")
    vocab = engine(rv, cards=NAMES + ("Witch",))
    src = tmp_path / "src" / "best.pt"
    src.parent.mkdir()
    torch.save(make_blob(torch, meta_for()), src)
    (src.parent / "args.json").write_text(json.dumps({"wait_keep": "0.25", "lr": "0.001"}),
                                          encoding="utf-8")
    carried = rv.grow(torch.load(src, map_location="cpu"), vocab, source=src)
    assert carried.record["from_sha256"] and len(carried.record["from_sha256"]) == 64
    best, args_json = rv.write_carried(carried, tmp_path / "out", src)
    doc = json.loads(args_json.read_text(encoding="utf-8"))
    assert (doc["wait_keep"], doc["lr"]) == ("0.25", "0.001")
    assert doc["vocab_carry"][-1]["op"] == "grow"
    assert rs_decide.wait_keep_for(str(best)) == rs_decide.wait_keep_for(str(src)) == 0.25
    loaded = rv.RC.load_policy(best, vocab, allow_drift=False)        # torch.load of the file
    assert loaded.meta["cards"]["Witch"] == 18 and not loaded.model.training
    # refusals: the source's own folder, an existing best.pt unless forced
    with pytest.raises(rv.VocabError, match="own folder"):
        rv.write_carried(carried, src.parent, src)
    with pytest.raises(rv.VocabError, match="exists"):
        rv.write_carried(carried, tmp_path / "out", src)
    rv.write_carried(carried, tmp_path / "out", src, force=True)
    # no args.json next to the source: the default it ran with, written down
    bare = tmp_path / "bare" / "best.pt"
    bare.parent.mkdir()
    torch.save(make_blob(torch, meta_for()), bare)
    best2, args2 = rv.write_carried(rv.grow(torch.load(bare), vocab, source=bare),
                                    tmp_path / "out2", bare)
    assert json.loads(args2.read_text(encoding="utf-8"))["wait_keep"] == \
        str(rs_decide.DEFAULT_WAIT_KEEP)
    assert rs_decide.wait_keep_for(str(best2)) == rs_decide.wait_keep_for(str(bare))
    # another run's args.json alone in --out is not replaced unless forced
    lone = tmp_path / "lone"
    lone.mkdir()
    (lone / "args.json").write_text("{}", encoding="utf-8")
    with pytest.raises(rv.VocabError, match=r"args\.json exists \(--force"):
        rv.write_carried(carried, lone, src)
    assert not (lone / "best.pt").exists()
    rv.write_carried(carried, lone, src, force=True)
    assert json.loads((lone / "args.json").read_text(encoding="utf-8"))["wait_keep"] == "0.25"


def test_write_puts_args_json_down_before_best_pt(rv, torch, tmp_path, monkeypatch):
    """A failure between the two leaves no best.pt without its args.json, and no temp file."""
    vocab = engine(rv, cards=NAMES + ("Witch",))
    carried = rv.grow(make_blob(torch, meta_for()), vocab)
    real = rv.os.replace
    done = []

    def replace(src, dst):
        if Path(dst).name == "best.pt":
            raise OSError("disk full")
        done.append(Path(dst).name)
        return real(src, dst)

    monkeypatch.setattr(rv.os, "replace", replace)
    out = tmp_path / "out"
    with pytest.raises(OSError, match="disk full"):
        rv.write_carried(carried, out)
    assert done == ["args.json"] and sorted(p.name for p in out.iterdir()) == ["args.json"]


# ---------------------------------------------------------------------------
# the command line (fake royalesim, one card added)
# ---------------------------------------------------------------------------

def test_cli(rv, torch, fake_royalesim, tmp_path, capsys):
    pytest.importorskip("numpy")                      # check reads Hog 2.6 from tools/rs_rows.py
    fake_royalesim.battle_kwargs["catalogue"] = DEFAULT_CATALOGUE + (
        catalogue_row("Witch", 5, "TROOP"),)
    src = tmp_path / "src" / "best.pt"
    src.parent.mkdir()
    torch.save(make_blob(torch, meta_for()), src)
    out = tmp_path / "carried"

    assert rv.main(["diff", str(src)]) == rv.EXIT_GROW
    text = capsys.readouterr().out
    assert "cards       1 card appended (Witch)" in text and "verdict     GROW" in text
    assert rv.main(["check", str(src)]) == rv.EXIT_FAILED
    text = capsys.readouterr().out
    assert f"REFUSED      {src}" in text
    assert f"  fix:     python tools/rs_vocab.py grow {src} --out DIR" in text

    assert rv.main(["grow", str(src), "--out", str(out)]) == rv.EXIT_OK
    assert f"wrote {out / 'best.pt'}" in capsys.readouterr().out
    assert rv.main(["grow", str(src), "--out", str(out)]) == rv.EXIT_FAILED      # exists
    assert "--force" in capsys.readouterr().err

    grown = str(out / "best.pt")
    assert rv.main(["check", grown]) == rv.EXIT_OK          # Witch is cold, not in the pool
    assert f"OK           {grown}" in capsys.readouterr().out
    assert rv.main(["check", grown, "--pool", "Witch"]) == rv.EXIT_COLD
    assert "  cold:    Witch (in the pool: Witch)" in capsys.readouterr().out
    assert rv.main(["check", grown, str(src), "--json"]) == rv.EXIT_FAILED     # the worst
    doc = json.loads(capsys.readouterr().out)
    assert doc["verdict"] == "REFUSED"
    assert [c["verdict"] for c in doc["checkpoints"]] == ["OK", "REFUSED"]
    assert rv.main(["diff", str(src), grown]) == rv.EXIT_GROW
    assert rv.main(["diff", grown]) == rv.EXIT_OK
    capsys.readouterr()
    assert rv.main(["diff", grown, "--json"]) == rv.EXIT_OK
    assert json.loads(capsys.readouterr().out)["verdict"] == "SAME"

    assert rv.main(["reset-cold", grown, "--cards", "Nope", "--out", str(tmp_path / "r")]) == \
        rv.EXIT_FAILED
    assert "not in its card vocabulary" in capsys.readouterr().err
    assert rv.main(["reset-cold", grown, "--cards", "HogRider", "--out", str(tmp_path / "r")]) \
        == rv.EXIT_FAILED
    assert "--allow-deck" in capsys.readouterr().err and not (tmp_path / "r").exists()
    assert rv.main(["reset-cold", grown, "--cards", "Zap", "--out", str(tmp_path / "r")]) == \
        rv.EXIT_OK
    assert "  note: Zap had its own row and no cold mark" in capsys.readouterr().out
    assert rv.main(["check", str(tmp_path / "r" / "best.pt"), "--pool", "Zap"]) == rv.EXIT_COLD
    assert rv.main(["reset-cold", grown, "--cards", "HogRider", "--allow-deck", "--out",
                    str(tmp_path / "r2")]) == rv.EXIT_OK
    assert rv.main(["diff", str(tmp_path / "missing.json")]) == rv.EXIT_ARGS


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

@pytest.mark.engine
def test_grow_on_the_real_engine(rv, torch):
    """A checkpoint whose rows lack the engine's last card (as one from before a new card
    shipped would): refused with grow named, then carried, the new row the unknown row."""
    import royalesim
    rs_engine = rv.rs_engine
    cfg = rs_engine.EngineConfig.load(env={})
    try:
        vocab = rv.RC.engine_vocab(cfg, royalesim)
    except rs_engine.EngineError as exc:
        pytest.skip(f"the engine refused data/engine.json's config: {exc}")
    cards, units = vocab.cards, vocab.units
    meta = {"n_cards": len(cards) + 1, "n_units": len(units) + 2, "live_compat": False,
            "cards": {c: i + 2 for i, c in enumerate(cards[:-1])},
            "units": {u: i + 2 for i, u in enumerate(units)},
            "card_table": vocab.card_table, "engine_fingerprint": dict(vocab.fingerprint or {}),
            "d": 16, "layers": 1, "heads": 2}
    blob = make_blob(torch, meta)
    with pytest.raises(rv.RC.VocabMismatch, match="rs_vocab.py grow"):
        rv.RC.check_meta(meta, vocab)
    carried = rv.grow(blob, vocab, source="engine-test")
    assert carried.diff.cards.added == (cards[-1],) and carried.max_delta < 1e-4
    new = carried.blob["model"][CARD_EMB]
    assert new.shape[0] == len(cards) + 2 and torch.equal(new[-1], blob["model"][CARD_EMB][1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", rv.RC.VocabDriftWarning)        # its own engine: no note
        rv.RC.load_policy(carried.blob, vocab, allow_drift=False)
    assert rv.check(carried.blob, vocab).verdict == "OK"
