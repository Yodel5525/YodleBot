"""S03: tools/opp_model.py, the deck prior from the IL_Replay train split
(docs/LIVE_SEARCH_PLAN.md section 5, S03).

No marker: a tiny hand-made prior (letters for card names) checks the plan's weight
sum over decks d holding c of exp(beta |d & known|) against a brute-force sum, the sequential draw,
the co-occurrence rest (exact up to one known card), the never-together rule, binding to catalogue
ids, filling a CardCycle (rs_public_view, S02), the file and its refusals, the builder refusing every
pair outside train, the build from synthetic IL_Replay rows through rs_match.read_pairs (numpy), the
command line and the import weight. Under `il_replay`: a prior built from one corpus part holds
exactly the train seats of that part and no dev or test pairing. A committed prior (once the owner
has built it) must be train and anonymous.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import (HOG26_KEYS, OPPONENT_KEYS, il_payload, il_replay_dir,
                                     il_replay_row)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import opp_model as OM  # noqa: E402

PV = OM.PV                       # the rs_public_view module opp_model uses (one CardCycle class)

D1, D2, D3, D4 = "ABCDEFGH", "ABCDIJKL", "MNOPQRST", "ABCDEFGX"
DECKS = {tuple(D1): 10, tuple(D2): 5, tuple(D3): 20, tuple(D4): 1}
#: A catalogue without X: X is in the prior but not in the engine.
CATALOGUE = list("ZYABCDEFGHIJKLMNOPQRST")
ID = {n: i for i, n in enumerate(CATALOGUE)}

#: IL_Replay key -> RoyaleSim name (tests/pipeline/test_rs_match.py's NAMES) plus a card only the
#: held-out rows play.
NAMES = {"cannon": "Cannon", "fireball": "Fireball", "hog-rider": "HogRider",
         "ice-golem": "IceGolemite", "ice-spirit": "IceSpirits", "musketeer": "Musketeer",
         "skeletons": "Skeletons", "the-log": "Log", "knight": "Knight", "archers": "Archer",
         "giant": "Giant", "arrows": "Arrows", "valkyrie": "Valkyrie", "mini-pekka": "MiniPekka",
         "baby-dragon": "BabyDragon", "zap": "Zap", "pekka": "Pekka"}
HELD_OUT_KEYS = tuple("pekka" if k == "zap" else k for k in OPPONENT_KEYS)


def brute(known, decks=DECKS, beta=1.0):
    """The plan's weight, summed deck by deck."""
    out: dict = {}
    for deck, n in decks.items():
        ov = len(set(deck) & set(known))
        for c in deck:
            if c not in known:
                out[c] = out.get(c, 0.0) + n * math.exp(beta * ov)
    return out


def close(got: dict, want: dict) -> bool:
    return set(got) == set(want) and all(abs(got[c] - want[c]) <= 1e-9 * max(1.0, want[c])
                                         for c in want)


@pytest.fixture
def prior():
    return OM.DeckPrior.from_counts(DECKS)


def ids(cards):
    return [ID[c] for c in cards]


def letters(cards):
    return [CATALOGUE[i] for i in cards]


def tag_of(split: str, skip: int = 0) -> str:
    tags = (f"t{i}" for i in range(10_000))
    return [t for t in tags if OM.deck_stats.split_of(t) == split][skip]


# ---------------------------------------------------------------------------
# the draw
# ---------------------------------------------------------------------------

def test_weights_follow_the_plans_formula(prior):
    assert prior.cards[:8] == tuple("MNOPQRST")               # most played first, then names
    assert prior.cards[8:] == tuple("ABCDEFGHIJKLX")
    assert prior.seats == 36 and len(prior.decks) == 4         # every deck kept (min_count 1)
    for known in ((), "A", "AB", "AM", "ABCD", "EIM", "ABCDEFGH"):
        assert close(prior.weights(known), brute(known)), known
    w = prior.weights("A")
    assert w["E"] == pytest.approx(11 * math.e) and w["M"] == pytest.approx(20.0)
    assert prior.weights("A", exclude="BE").keys() == brute("A").keys() - {"B", "E"}
    beta2 = OM.DeckPrior.from_counts(DECKS, beta=2.0)
    assert close(beta2.weights("AB"), brute("AB", beta=2.0))
    flat = OM.DeckPrior.from_counts(DECKS, beta=0.0)          # beta 0: the cards' seat counts
    assert close(flat.weights(), {c: float(prior._cs[i]) for i, c in enumerate(prior.cards)})


def test_top_completes_the_most_played_deck(prior):
    assert prior.top("A", 1) == ["B"]                         # 16e beats M's 20
    assert prior.top("ABCD", 4) == list("EFGH")               # ties E, F, G by name
    assert prior.top("ABCDI", 3) == list("JKL")               # one card turns it to D2
    assert prior.top("ABCD", 4) == list("EFGH")               # the cache gives copies
    got = prior.top("ABCD", 4)
    got.append("junk")
    assert prior.top("ABCD", 4) == list("EFGH")
    assert prior.draw("ABCD", 4) == list("EFGH")              # no rng: top
    assert prior.top("ABCD", 0) == []
    with pytest.raises(ValueError):
        prior.top("A", -1)
    assert prior.top(list("ABCDEFGHIJKLMNOPQRST"), 3) == ["X"]   # only X left
    assert prior.top([*D1, "Unknown", None, -1], 1) == ["X"]   # unknown labels are ignored


def test_sample_is_distinct_seeded_and_follows_the_probabilities(prior):
    rng = np.random.default_rng(0)
    for _ in range(50):
        got = prior.sample("A", 7, rng, exclude="C")
        assert len(got) == len(set(got)) == 7 and not set(got) & {"A", "C"}
    assert prior.sample("A", 5, 7) == prior.sample("A", 5, 7)                # a seed
    assert prior.sample("A", 5, random.Random(3)) == prior.sample("A", 5, random.Random(3))
    assert prior.draw("A", 5, 7) == prior.sample("A", 5, 7)
    with pytest.raises(ValueError):
        prior.sample("A", 2, None)
    with pytest.raises(TypeError):
        prior.sample("A", 2, "seed")
    assert prior.sample(list("ABCDEFGHIJKLMNOPQRST"), 5, 0) == ["X"]         # fewer when empty
    p = prior.probabilities("A")
    assert sum(p.values()) == pytest.approx(1.0)
    w = brute("A")
    assert p["B"] == pytest.approx(w["B"] / sum(w.values()))
    rng, n = np.random.default_rng(1), 6000
    seen = Counter(prior.sample("A", 1, rng)[0] for _ in range(n))
    assert max(abs(seen[c] / n - p[c]) for c in p) < 0.025
    # The first card joins the known set. Two equal decks and beta 3: after ABCD the first card
    # is any of E..L (1/8 each); after I the rest of D2 holds 3e^15 / (3e^15 + 4e^12) = 0.94 of
    # the weight (independent draws would give it 3/7).
    seq = OM.DeckPrior.from_counts({tuple(D1): 10, tuple(D2): 10}, beta=3.0)
    assert seq.probabilities("ABCD") == pytest.approx({c: 1 / 8 for c in "EFGHIJKL"})
    assert sum(seq.probabilities("ABCDI")[c] for c in "JKL") == pytest.approx(
        3 * math.e ** 3 / (3 * math.e ** 3 + 4))
    rng = np.random.default_rng(2)
    follow = Counter()
    for _ in range(400):
        first, second = seq.sample("ABCD", 2, rng)
        if first == "I":
            follow[second] += 1
    assert sum(follow.values()) >= 20
    assert sum(follow[c] for c in "JKL") > 0.8 * sum(follow.values())


def test_the_rest_is_exact_up_to_one_known_card():
    for max_decks, min_count in ((0, 100), (1, 1), (2, 1), (0, 2)):
        part = OM.DeckPrior.from_counts(DECKS, max_decks=max_decks, min_count=min_count)
        assert len(part.decks) == {(0, 100): 0, (1, 1): 1, (2, 1): 2, (0, 2): 3}[max_decks,
                                                                                  min_count]
        for known in ((), "A", "M", "X", "I"):
            assert close(part.weights(known), brute(known)), (max_decks, min_count, known)
        two = part.weights("AI")                                  # an approximation from 2 on
        assert set(two) == set(brute("AI")) and all(v > 0 for v in two.values())
    rest = OM.DeckPrior.from_counts(DECKS, min_count=100)
    assert rest.kept["decks"] == 0 and rest.kept["share"] == 0.0
    assert rest.top("ABCD", 1)[0] in "EFGHIJKLX"


def test_never_together_keeps_two_champions_apart():
    decks = {tuple("Pabcdefg"): 50, tuple("Qabcdefh"): 50, tuple("Rijklmno"): 100}
    prior = OM.DeckPrior.from_counts(decks, min_expected=10)  # P, Q: 50 x 50 / 200 = 12.5 apart
    assert prior.never_together("P", "Q") and prior.never_together("Q", "P")
    assert not prior.never_together("P", "a") and not prior.never_together("P", "nope")
    assert prior.weights("P")["Q"] == pytest.approx(50.0)     # the formula still weighs Q
    assert "Q" not in prior.probabilities("P")                # but Q is never drawn beside P
    rng = np.random.default_rng(0)
    assert all("Q" not in prior.sample("P", 7, rng) for _ in range(100))
    # Every card left is apart from a known one: the draw falls back to the formula (Q and h
    # 50e^6 each, R and i..o 100 each; Q before h: as many seats, "Q" < "h").
    assert all(prior.never_together(c, "P") or prior.never_together(c, "g") for c in "Qhijklmno")
    assert prior.probabilities("Pabcdefg") == pytest.approx(
        {c: w / sum(prior.weights("Pabcdefg").values())
         for c, w in prior.weights("Pabcdefg").items()})
    assert prior.top("Pabcdefg", 1) == ["Q"]
    assert len(prior.sample("Pabcdefg", 1, 0)) == 1
    off = OM.DeckPrior.from_counts(decks, min_expected=math.inf)
    assert "Q" in off.probabilities("P") and not off.never_together("P", "Q")
    assert "Q" in OM.DeckPrior.from_counts(decks).probabilities("P")      # 12.5 < 25


def test_bind_draws_catalogue_ids_and_skips_cards_the_engine_lacks(prior):
    bound = prior.bind(CATALOGUE)
    assert bound.bound and not prior.bound
    assert prior.top("A", 1) == ["B"]                         # the original keeps its names
    assert bound.top(ids("A"), 3) == ids("BCD")
    assert bound.top(ids("ABCDEFG"), 1) == ids("H")
    assert bound.top(ids("ABCDEFGHIJKLMNOPQRST"), 3) == []   # X is not in the catalogue
    rng = np.random.default_rng(4)
    assert all(isinstance(c, int) for c in bound.sample(ids("A"), 7, rng))
    assert np.int64(ID["A"]) in bound and ID["Z"] not in bound and "A" not in bound
    assert bound.top([np.int64(ID["A"])], 1) == ids("B")      # numpy ints read as ids
    assert bound.digest() == prior.digest()                   # the same draws, other labels


# ---------------------------------------------------------------------------
# cycles
# ---------------------------------------------------------------------------

def test_fill_cards_world0_and_shuffled_worlds(prior):
    bound = prior.bind(CATALOGUE)
    cyc = PV.CardCycle.from_plays(ids("AB"))
    assert cyc.queue == (None, None, ID["A"], ID["B"]) and cyc.hand == (None,) * 4
    assert letters(bound.fill_cards(cyc)) == list("CDEFGH")
    assert bound.fill_cards(cyc.to_json()) == bound.fill_cards(cyc)
    full = bound.fill_cycle(cyc)
    assert letters(full.hand) == list("CDEF") and letters(full.queue) == list("GHAB")
    assert full.exact and full.filled == 6
    # A recast card takes an open slot: one card fewer, never drawn.
    recast = bound.fill_cards(cyc, known=ids("I"))
    assert len(recast) == 5 and ID["I"] not in recast and ID["J"] in recast
    with_recast = bound.fill_cycle(cyc, known=ids("I"))
    assert ID["I"] in with_recast.hand + with_recast.queue and None not in with_recast.hand
    # Worlds: a sample, shuffled, so a drawn card sits in the hand or at the queue's head.
    where = Counter()
    for seed in range(200):
        world = bound.fill_cycle(cyc, rng=seed)
        cards = world.hand + world.queue
        assert None not in cards and len(set(cards)) == 8
        assert world.queue[2:] == (ID["A"], ID["B"])
        where["hand" if ID["C"] in world.hand else "queue" if ID["C"] in world.queue else "out"] += 1
        assert bound.fill_cards(cyc, rng=seed) == bound.fill_cards(cyc, rng=seed)
    assert where["hand"] and where["queue"]
    # No open slot: nothing to draw.
    done = PV.CardCycle.from_known(ids("CDEF"), ids("GHAB"))
    assert bound.fill_cards(done) == [] and bound.fill_cards(done, rng=1) == []
    with pytest.raises(OM.DeckPriorError, match="bind"):
        prior.fill_cards(cyc)


def test_fill_cycle_keeps_omit_cards_out_of_the_hand(prior):
    bound = prior.bind(CATALOGUE)
    cyc = PV.CardCycle.from_plays(ids("AB"))
    filled = bound.fill_cycle(cyc, omit=ids("C"))             # C as a Mirror: a queue slot
    assert ID["C"] in filled.queue and ID["C"] not in filled.hand
    exclude = bound.fill_cards(cyc, exclude=ids("CD"))
    assert not set(exclude) & set(ids("CD")) and len(exclude) == 6


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------

def test_the_file_round_trips_and_refuses_another_split(prior, tmp_path):
    doc = prior.to_dict()
    assert tuple(doc) == OM.FILE_KEYS and doc["split"] == "train"
    text = OM.render(doc)
    assert json.loads(text) == doc and text.endswith("}\n")
    assert '\n  [20,0,1,2,3,4,5,6,7],\n  [10,8,9,10,11,12,13,14,15],\n' in text   # a row a line
    path = prior.save(tmp_path / "opp_prior" / "deck_prior.json")
    back = OM.DeckPrior.load(path)
    assert back.digest() == prior.digest() and back.weights("AB") == prior.weights("AB")
    assert back.decks == prior.decks and back.cards == prior.cards
    assert OM.DeckPrior.load(path, card_names=CATALOGUE).top(ids("A"), 1) == ids("B")
    assert OM.DeckPrior.load(path, beta=2.0).digest() != prior.digest()
    assert OM.load_prior(path).digest() == prior.digest()
    assert OM.load_prior(tmp_path / "missing.json") is None
    with pytest.raises(FileNotFoundError, match="build-prior"):
        OM.load_prior(tmp_path / "missing.json", required=True)
    for key, value, match in (("split", "val", "train split"), ("split", "test", "train split"),
                              ("schema", 2, "schema"), ("kind", "deck_stats", "schema")):
        bad = dict(doc, **{key: value})
        with pytest.raises(OM.DeckPriorError, match=match):
            OM.DeckPrior.from_dict(bad)
    inflated = json.loads(text)
    inflated["decks"][0][0] += 1000                         # more seats than the card counts
    with pytest.raises(OM.DeckPriorError, match="do not add up"):
        OM.DeckPrior.from_dict(inflated)
    short = json.loads(text)
    short["pairs"] = short["pairs"][:-1]
    with pytest.raises(OM.DeckPriorError, match="rows expected"):
        OM.DeckPrior.from_dict(short)
    (tmp_path / "junk.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(OM.DeckPriorError, match="not JSON"):
        OM.DeckPrior.load(tmp_path / "junk.json")
    with pytest.raises(OM.DeckPriorError):
        OM.DeckPrior.from_counts({tuple("ABCDEFG"): 1})      # 7 cards
    with pytest.raises(OM.DeckPriorError):
        OM.DeckPrior.from_counts(DECKS, beta=-1)


# ---------------------------------------------------------------------------
# building from the train split only
# ---------------------------------------------------------------------------

def pair(tag, split="train", decks=(tuple(D1), tuple(D3))):
    return SimpleNamespace(tag=tag, split=split, decks=decks, forms=((0,) * 8, (0,) * 8))


def test_the_builder_refuses_every_pair_outside_train():
    b = OM.PriorBuilder()
    train = [tag_of("train", k) for k in range(3)]
    for t in train:
        assert b.add(pair(t))
    assert not b.add(pair(train[0]))                          # a tag counts once
    assert b.pairs == 3 and b.skipped == {"duplicate": 1}
    for split in ("val", "test"):
        held = tag_of(split)
        with pytest.raises(OM.DeckPriorError, match="train split only"):
            b.add(pair(held))                                 # says train, tag says otherwise
        with pytest.raises(OM.DeckPriorError, match="train split only"):
            b.add(pair(held, split=split))
        with pytest.raises(OM.DeckPriorError, match="train split only"):
            b.add_set(SimpleNamespace(split=split, pairs=(), meta={}))
    with pytest.raises(OM.DeckPriorError, match="train split only"):
        b.add(pair(train[1], split="val"))                    # a train tag in a dev pair set
    assert b.pairs == 3
    body = json.dumps([[t, [list(D1), list(D3)], [[0] * 8, [0] * 8]] for t in train],
                      separators=(",", ":"))
    assert b.digest() == hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]
    assert OM.PriorBuilder().digest() == hashlib.sha1(b"[]").hexdigest()[:16]
    p = b.prior(max_decks=0, min_count=1)
    assert p.seats == 6 and p.source["pairs"] == 3 and p.source["digest"] == b.digest()
    assert {frozenset(p.cards[i] for i in idx): n for n, idx in p.decks} == {
        frozenset(D1): 3, frozenset(D3): 3}
    with pytest.raises(OM.DeckPriorError, match="no train pairs"):
        OM.PriorBuilder().prior()


@pytest.fixture
def rs_match(monkeypatch):
    """tools/rs_match as the tools import it (test_rs_match.py's setup)."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    return importlib.import_module("rs_match")


def synthetic_rows(n=60):
    """IL_Replay rows synthetic-0000 .. n-1: train rows play Hog 2.6 against OPPONENT_KEYS (the
    first one a Hog mirror); dev and test rows play a deck with Pekka no train row has."""
    rows, first = [], True
    for i in range(n):
        tag = f"synthetic-{i:04d}"
        if OM.deck_stats.split_of(tag) != "train":
            p = il_payload(tag=tag, opponent_deck=HELD_OUT_KEYS, plays=(), abilities=())
        elif first:
            p = il_payload(tag=tag, opponent_deck=HOG26_KEYS, opponent_forms={}, plays=(),
                           abilities=())
            first = False
        else:
            p = il_payload(tag=tag, plays=(), abilities=())
        rows.append(il_replay_row(p))
    return rows


def test_a_build_from_rows_counts_the_train_rows_only(rs_match):
    rows = synthetic_rows()
    splits = Counter(OM.deck_stats.split_of(r["replay_tag"]) for r in rows)
    assert splits["train"] and splits["val"] and splits["test"]
    b = OM.build_from_rows(rows, NAMES)
    prior = b.prior(max_decks=0, min_count=1)
    assert b.pairs == splits["train"] and b.other_splits == splits["val"] + splits["test"]
    assert "Pekka" not in prior and "Pekka" not in prior.cards     # only held-out rows play it
    want = rs_match.read_pairs(rows, NAMES, "train", exclude_mirror=False)
    assert prior.source["digest"] == want.digest() and prior.source["pairs"] == len(want)
    assert prior.seats == 2 * splits["train"]
    hog = frozenset(rs_match.HOG26)
    got = {frozenset(prior.cards[i] for i in idx): n for n, idx in prior.decks}
    assert got[hog] == splits["train"] + 1                     # the mirror counts both seats
    assert sum(got.values()) == prior.seats
    assert prior.source["exclude_mirror"] is False and prior.source["split_rule"]


def test_build_prior_writes_the_file_and_refuses_dev(monkeypatch, tmp_path, capsys):
    b = OM.PriorBuilder()
    for k in range(4):
        b.add(pair(tag_of("train", k), decks=(tuple(D1), tuple(D2) if k % 2 else tuple(D3))))
    b.parts.append("part-00000.parquet")
    calls = []

    def fake(corpus, *, parts, names, progress):
        calls.append((corpus, parts, progress, len(names)))
        return b

    monkeypatch.setattr(OM, "build_from_corpus", fake)
    out = tmp_path / "deck_prior.json"
    assert OM.main(["build-prior", "--split", "train", "--out", str(out), "--parts", "1",
                    "--max-decks", "2", "--min-count", "1"]) == 0
    assert calls and calls[0][1] == 1 and calls[0][2] is True
    printed = capsys.readouterr().out
    assert "commit it" in printed and "anonymous" in printed
    prior = OM.DeckPrior.load(out)
    assert len(prior.decks) == 2 and prior.kept["max_decks"] == 2
    assert prior.source["parts"] == ["part-00000.parquet"] and prior.source["card_map"]["sha256"]
    assert prior.created and prior.source["modes"]
    for split in ("dev", "val", "test", "final"):
        with pytest.raises(SystemExit) as exc:
            OM.main(["build-prior", "--split", split, "--out", str(out)])
        assert exc.value.code == 2
    assert OM.main(["build-prior", "--out", str(out), "--min-count", "0"]) == 2
    assert OM.main(["show", "--prior", str(out), "--known", "A,B,Nope", "--samples", "2"]) == 0
    shown = capsys.readouterr().out
    assert "not in the prior: Nope" in shown and "most likely 5" in shown     # 8 - 3 known
    assert shown.count("sample ") == 2
    assert OM.main(["show", "--prior", str(tmp_path / "none.json")]) == 1


def test_imports_neither_torch_royalesim_rs_match_nor_the_memory_reader():
    code = ("import sys\n"
            "import tools.opp_model\n"
            "heavy = ('torch', 'royalesim', 'rs_match', 'rs_eval', 'rs_rows', 'pyarrow',\n"
            "         'scripts.gamestate', 'scripts.vision', 'gamestate.memread')\n"
            "print(','.join(m for m in heavy if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


# ---------------------------------------------------------------------------
# the corpus, and the committed file
# ---------------------------------------------------------------------------

@pytest.mark.il_replay
def test_the_built_prior_holds_no_dev_or_test_pairing(rs_match):
    pytest.importorskip("pyarrow")
    corpus = il_replay_dir()
    b = OM.build_from_corpus(corpus, parts=1)
    prior = b.prior(max_decks=0, min_count=1)
    train = rs_match.load_pairs(corpus, "train", parts=1, exclude_mirror=False)
    dev = rs_match.load_pairs(corpus, "dev", parts=1, exclude_mirror=False)
    final = rs_match.load_pairs(corpus, "final", parts=1, exclude_mirror=False, allow_final=True)
    assert len(train) and prior.source["digest"] == train.digest()
    assert prior.source["pairs"] == len(train) and prior.seats == 2 * len(train)
    assert all(OM.deck_stats.split_of(t) == "train" for t in train.tags)
    held = set(dev.tags) | set(final.tags)
    assert held and not held & set(train.tags)
    seats = Counter(frozenset(d) for p in train.pairs for d in p.decks)
    got = {frozenset(prior.cards[i] for i in idx): n for n, idx in prior.decks}
    assert got == dict(seats)                                  # the train seats, nothing else
    held_only = {frozenset(d) for p in dev.pairs + final.pairs for d in p.decks} - set(seats)
    assert not held_only & set(got)
    held_cards = {c for p in dev.pairs + final.pairs for d in p.decks for c in d}
    train_cards = {c for p in train.pairs for d in p.decks for c in d}
    assert set(prior.cards) == train_cards and not (held_cards - train_cards) & set(prior.cards)


SOURCE_KEYS = {"pairs", "digest", "distinct_decks", "split_rule", "exclude_mirror", "skipped",
               "other_splits", "parts", "modes", "card_map"}


def test_a_committed_prior_is_train_and_anonymous():
    if not OM.DEFAULT_PATH.exists():
        pytest.skip(f"no {OM.DEFAULT_PATH.relative_to(ROOT).as_posix()} yet: the owner runs "
                    "tools/opp_model.py build-prior --split train")
    data = json.loads(OM.DEFAULT_PATH.read_text(encoding="utf-8-sig"))
    assert tuple(data) == OM.FILE_KEYS and data["split"] == "train"
    assert set(data["source"]) <= SOURCE_KEYS, set(data["source"]) - SOURCE_KEYS
    assert all(re.fullmatch(r"part-[\w.-]+\.parquet", p) for p in data["source"]["parts"])
    assert re.fullmatch(r"[0-9a-f]{16}", data["source"]["digest"])
    assert OM.DEFAULT_PATH.stat().st_size <= OM.COMMIT_BYTES
    prior = OM.DeckPrior.load(OM.DEFAULT_PATH)
    assert prior.seats == 2 * data["source"]["pairs"] and len(prior.decks) >= 1
