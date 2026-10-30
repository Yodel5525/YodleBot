"""L03: tools/rs_league_eval.py, the league evaluation and the promotion gate.

What is checked (the plan's test list first): synthetic result logs give the right pass, fail,
cycle and incomplete decisions, worked by hand at the boundaries (n = 1,000: a score of 551 has
Wilson LB 0.5200346 > 0.52, 550.5 has 0.5195333; an anchor drop of exactly 2 pp passes, 2.125 pp
fails; the pool's reference less one standard error of 3/280); the final split is refused
without --final by report(), evaluate() (before anything is played) and both subcommands. Then
the report's tables (Wilson, the pairing bootstrap, the paired CI and McNemar, the BTD table),
the log (torn lines, duplicates), the worker cap, the protocol, players, and, on conftest's fake
royalesim with stub players: the games are rs_match.pairing_specs' (both seats, the decks
exchanged for a head-to-head, the same positions for every player against an anchor), a run
resumes without replaying, stops early, refuses a changed player, logs and retries errors, plays
the champion against the pool members (a cycle is found), refuses final-split pairs given as a
plain list, and plays idle seats through rs_match.run_match under the delay. One engine-marked
smoke test.

Synthetic tags (as in test_rs_match.py): of "synthetic-0000" .. "synthetic-0199" the 12 in VAL
are the val (dev) split and the 10 in TEST the test (final) split.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from tests.pipeline.conftest import HOG26_KEYS, OPPONENT_KEYS, il_payload, il_replay_row

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.league import stats as LS  # noqa: E402

VAL = (4, 16, 40, 56, 74, 91, 114, 132, 140, 164, 169, 193)
TEST = (3, 24, 27, 53, 84, 107, 118, 151, 190, 196)
NAMES = {"cannon": "Cannon", "fireball": "Fireball", "hog-rider": "HogRider",
         "ice-golem": "IceGolemite", "ice-spirit": "IceSpirits", "musketeer": "Musketeer",
         "skeletons": "Skeletons", "the-log": "Log", "knight": "Knight", "archers": "Archer",
         "giant": "Giant", "arrows": "Arrows", "valkyrie": "Valkyrie", "mini-pekka": "MiniPekka",
         "baby-dragon": "BabyDragon", "zap": "Zap"}
#: rs_match.pairing_specs' engine seed of synthetic-0004 with run seed 0 (test_rs_match.py), and
#: of synthetic-0004 / -0016 with run seed 5: sha1("seed:tag")[:12] mod 2 ** 31.
SEED0_0004 = 1514628266
SEEDS5 = (1663230741, 870327283)


@pytest.fixture
def le(monkeypatch):
    """rs_league_eval alone (its statistics need neither rs_match nor an engine)."""
    monkeypatch.syspath_prepend(str(TOOLS))
    return importlib.import_module("rs_league_eval")


@pytest.fixture
def tools(monkeypatch):
    """rs_rows, rs_match and rs_league_eval as the tools import them, with the module settings
    at their defaults and no engine config from the PC (as tests/pipeline/test_rs_match.py)."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    rm = importlib.import_module("rs_match")
    le_mod = importlib.import_module("rs_league_eval")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "OUT": rr.OUT, "CARD_FLY": {}, "CARD_BUILDING": {}}.items():
        monkeypatch.setattr(rr, name, value)
    return types.SimpleNamespace(rows=rr, match=rm, league=le_mod, engine=eng)


def cfg(tools):
    return tools.engine.EngineConfig.defaults()


def synthetic_rows(n=200):
    """IL_Replay replays rows (test_rs_match.py's): even tags record Hog 2.6 as the team, odd
    tags as the opponent."""
    out = []
    for i in range(n):
        tag = f"synthetic-{i:04d}"
        if i % 2:
            p = il_payload(tag=tag, team_deck=OPPONENT_KEYS, opponent_deck=HOG26_KEYS,
                           team_forms={}, opponent_forms={"musketeer": 2}, plays=(), abilities=())
        else:
            p = il_payload(tag=tag, plays=(), abilities=())
        out.append(il_replay_row(p))
    return out


# ---------------------------------------------------------------------------
# synthetic logs
# ---------------------------------------------------------------------------

#: (a's team, who holds side a's deck) of a comparison's games in comparison_tasks' order: the
#: pairing as drawn, its seat swap, then (decks "both") the decks exchanged and its seat swap.
LAYOUT = {"both": ((0, "a"), (1, "a"), (1, "b"), (0, "b")), "a": ((0, "a"), (1, "a"))}


def game(LE, a, b, s, pairing, a_team=0, holder=None, split="val", protocol="P"):
    """One Game where `a` scores s (1 / 0.5 / 0) sitting on team a_team."""
    teams = (a, b) if a_team == 0 else (b, a)
    winner = -1 if s == 0.5 else (a_team if s == 1 else 1 - a_team)
    holder = a if holder is None else holder
    return LE.Game(key=f"{a}|{b}|{pairing}|{a_team}|{holder}", teams=teams, winner=winner,
                   pairing=pairing, holder=holder, protocol=protocol, split=split)


def comparison(LE, a, b, scores, decks="both", start=0, split="val"):
    """a's games against b, a's scores in order, laid out as comparison_tasks plays them."""
    per = len(LAYOUT[decks])
    out = []
    for i, s in enumerate(scores):
        p, j = divmod(i, per)
        a_team, who = LAYOUT[decks][j]
        out.append(game(LE, a, b, s, f"{split}:p{start + p:04d}", a_team,
                        a if who == "a" else b, split))
    return out


def planted(points, n):
    """n scores summing to `points`: wins, one draw for a half point, then losses."""
    wins = int(points)
    draws = 1 if points != wins else 0
    return [1.0] * wins + [0.5] * draws + [0.0] * (n - wins - draws)


class Table:
    """A rating table stand-in: predict(champion, member) from a dict."""

    def __init__(self, p, rating=None):
        self.p, self.rating, self.anchor_rating = dict(p), dict(rating or {}), 0.0

    def predict(self, a, b):
        return self.p[b]


def record(g):
    """A Game as its league_game log line (the fields the analysis reads)."""
    return {"kind": "league_game", "schema": 1, "key": g.key, "protocol": g.protocol,
            "split": g.split, "pairing": g.pairing, "seat": g.seat, "holder": g.holder,
            "teams": list(g.teams), "winner": g.winner, "ids": list(g.ids)}


# ---------------------------------------------------------------------------
# the plan's tests: pass, fail, cycle on synthetic logs
# ---------------------------------------------------------------------------

def test_gate_pass_fail_and_cycle_on_synthetic_logs(le):
    LE = le
    conf = LE.GateConfig(n_boot=200)
    anchor_h = comparison(LE, "H", "A", planted(240, 400), "a")             # champion 60.0%

    def log(points, anchor_points):
        return (comparison(LE, "C", "H", planted(points, 1000)) + anchor_h
                + comparison(LE, "C", "A", planted(anchor_points, 400), "a"))

    # PASS: 551 / 1000 has Wilson LB 0.5200346 > 0.52; 59.0% vs the anchor is a 1 pp drop.
    d = LE.gate(log(551, 236), "C", "H", anchors=["A"], config=conf)
    assert (d.decision, d.admit, d.promote, d.cycle, d.turned_on()) == ("pass", True, True,
                                                                        False, "none")
    c1 = d.conditions["beats_champion"]
    assert (c1.n, c1.passed, c1.observed, c1.missing, c1.reference) == (1000, True, 0.551, 0, 0.52)
    assert c1.bound == pytest.approx(0.5200345669542097, abs=1e-12)
    assert (c1.detail["wins"], c1.detail["draws"], c1.detail["losses"]) == (551, 0, 449)
    a = d.conditions["anchor:A"]
    assert (a.n, a.passed, a.observed, a.reference) == (400, True, 0.59, 0.6)
    assert a.detail["drop_pp"] == pytest.approx(1.0)
    assert a.bound == pytest.approx(0.58)
    assert (a.detail["candidate_games"], a.detail["champion_games"]) == (400, 400)
    assert d.conditions["pool_collapse"].skipped
    assert d.conditions["pool_collapse"].reason == LE.SKIP_NO_POOL
    assert list(d.conditions) == ["beats_champion", "anchor:A", "pool_collapse"]
    json.dumps(d.to_dict())

    # FAIL on the bound: 550.5 (550 W, 1 D, 449 L) has LB 0.5195333; the anchor still counted.
    d = LE.gate(log(550.5, 236), "C", "H", anchors=["A"], config=conf)
    assert (d.decision, d.admit, d.turned_on()) == ("fail", False, "beats_champion")
    c1 = d.conditions["beats_champion"]
    assert c1.bound == pytest.approx(0.5195333367516204, abs=1e-12) and not c1.passed
    assert c1.detail["draws"] == 1 and c1.observed == 0.5505
    assert d.conditions["anchor:A"].passed and not d.conditions["anchor:A"].skipped

    # A drop of exactly 2 pp passes (0.6 - 0.58 is 0.020000000000000018 in floats) ...
    d = LE.gate(log(560, 232), "C", "H", anchors=["A"], config=conf)
    assert d.decision == "pass" and d.conditions["anchor:A"].passed
    assert d.conditions["anchor:A"].detail["drop_pp"] == pytest.approx(2.0)
    # ... and 2.125 pp fails.
    d = LE.gate(log(560, 231.5), "C", "H", anchors=["A"], config=conf)
    assert (d.decision, d.turned_on()) == ("fail", "anchor:A")
    assert d.conditions["anchor:A"].detail["drop_pp"] == pytest.approx(2.125)
    assert d.conditions["pool_collapse"].reason == LE.SKIP_NO_POOL

    # CYCLE: passes 1 and 2, but its pool mean (50% vs M1, 55% vs M2: 0.525) is below the
    # champion's fitted mean (0.7, 0.6: 0.65) less one se. Pairing means: 45 of 0.5 and 5 of
    # 0.75 (M2's last 5 pairings score 3 of 4): sd 0.0757614, se = 3/280.
    pool = (comparison(LE, "C", "M1", [1, 1, 0, 0] * 25)
            + comparison(LE, "C", "M2", [1, 1, 0, 0] * 20 + [1, 1, 1, 0] * 5))
    games = log(560, 236) + pool
    d = LE.gate(games, "C", "H", anchors=["A"], pool=["M1", "M2", "A", "H"], config=conf,
                table=Table({"M1": 0.7, "M2": 0.6}))
    assert (d.decision, d.admit, d.promote, d.cycle, d.turned_on()) == ("cycle", True, False,
                                                                        True, "pool_collapse")
    p = d.conditions["pool_collapse"]
    assert (p.n, p.missing, p.passed) == (200, 0, False)
    assert p.observed == pytest.approx(0.525) and p.reference == pytest.approx(0.65)
    assert p.detail["se"] == pytest.approx(3 / 280)
    assert p.bound == pytest.approx(0.65 - 3 / 280)
    assert p.detail["per_member"] == {"M1": 0.5, "M2": 0.55}
    assert p.detail["predicted"] == {"M1": 0.7, "M2": 0.6} and p.detail["groups"] == 50
    # The same games against a pool the champion is fitted to score 0.525 on: pass.
    d = LE.gate(games, "C", "H", anchors=["A"], pool=["M1", "M2"], config=conf,
                table=Table({"M1": 0.5, "M2": 0.55}))
    assert d.decision == "pass" and d.conditions["pool_collapse"].passed


def test_gate_incomplete_and_settled_conditions(le):
    LE = le
    conf = LE.GateConfig(n_boot=100)
    h2h = comparison(LE, "C", "H", planted(560, 1000))
    anchor_h = comparison(LE, "H", "A", planted(240, 400), "a")
    anchor_c = comparison(LE, "C", "A", planted(236, 400), "a")
    d = LE.gate(h2h[1:] + anchor_h + anchor_c, "C", "H", anchors=["A"], config=conf)
    assert (d.decision, d.missing(), d.turned_on()) == ("incomplete", {"beats_champion": 1},
                                                        "beats_champion")
    assert d.conditions["beats_champion"].incomplete and not d.admit
    d = LE.gate(h2h + anchor_h + anchor_c[:-1], "C", "H", anchors=["A"], config=conf)
    assert (d.decision, d.missing()) == ("incomplete", {"anchor:A": 1})
    # The candidate's anchor games on other pairings: nothing to pair with the champion's.
    elsewhere = comparison(LE, "C", "A", planted(236, 400), "a", start=200)
    d = LE.gate(h2h + anchor_h + elsewhere, "C", "H", anchors=["A"], config=conf)
    a = d.conditions["anchor:A"]
    assert (d.decision, a.n, a.missing) == ("incomplete", 0, 400)
    assert (a.detail["candidate_games"], a.detail["champion_games"]) == (400, 400)
    # A pool member short of games.
    pool = (comparison(LE, "C", "M1", [1, 1, 0, 0] * 25)
            + comparison(LE, "C", "M2", ([1, 1, 0, 0] * 25)[:96]))
    d = LE.gate(h2h + anchor_h + anchor_c + pool, "C", "H", anchors=["A"], pool=["M1", "M2"],
                config=conf, table=Table({"M1": 0.5, "M2": 0.5}))
    assert (d.decision, d.missing(), d.turned_on()) == ("incomplete", {"pool_collapse": 4},
                                                        "pool_collapse")
    # Settled by the champion condition: the conditions with no games are skipped, not missing.
    d = LE.gate(comparison(LE, "C", "H", planted(500, 1000)), "C", "H", anchors=["A"],
                pool=["M1"], config=conf, table=Table({"M1": 0.5}))
    assert (d.decision, d.missing(), d.turned_on()) == ("fail", {}, "beats_champion")
    for name in ("anchor:A", "pool_collapse"):
        c = d.conditions[name]
        assert (c.skipped, c.reason, c.n, c.incomplete) == (True, LE.SKIP_SETTLED, 0, False)
    out = d.to_dict()
    assert out["conditions"]["anchor:A"]["skipped"] and out["missing"] == {}
    json.dumps(out)
    with pytest.raises(ValueError, match="is the champion"):
        LE.gate([], "H", "H")
    # An anchor equal to the champion or the candidate is not a condition.
    d = LE.gate(h2h, "C", "H", anchors=["H", "C"], config=conf)
    assert list(d.conditions) == ["beats_champion", "pool_collapse"] and d.decision == "pass"


# ---------------------------------------------------------------------------
# the report and the final split
# ---------------------------------------------------------------------------

def small_log(LE, split="val"):
    """C beats H 8 of 8; on the same 4 positions against A, H scores 1, .5, 0, 1 and C 1 each."""
    return (comparison(LE, "C", "H", [1.0] * 8, split=split)
            + comparison(LE, "H", "A", [1, 0.5, 0, 1], "a", split=split)
            + comparison(LE, "C", "A", [1, 1, 1, 1], "a", split=split))


def test_report_tables(le):
    LE = le
    conf = LE.GateConfig(champion_games=8, anchor_games=4, n_boot=500)
    games = small_log(LE)
    rep = LE.report(games, candidates=["C"], champion="H", anchors=["A"], config=conf)
    assert (rep["split"], rep["split_name"], rep["n_games"]) == ("val", "dev", 16)
    assert rep["players"] == ["C", "H", "A"]
    assert rep["roles"] == {"candidates": ["C"], "champion": "H", "anchors": ["A"], "pool": []}
    rows = {(m["a"], m["b"]): m for m in rep["matchups"]}
    assert list(rows) == [("C", "H"), ("C", "A"), ("H", "A")]
    ch = rows[("C", "H")]
    assert (ch["n"], ch["wins"], ch["draws"], ch["losses"], ch["score"]) == (8, 8, 0, 0, 1.0)
    assert (ch["lo"], ch["hi"]) == LS.wilson(8, 8)
    assert ch["lo"] == pytest.approx(0.6755843804891231)
    assert (ch["boot_lo"], ch["boot_hi"], ch["pairings"]) == (1.0, 1.0, 2)
    assert ch["a_holds"]["n"] == ch["b_holds"]["n"] == 4
    ha = rows[("H", "A")]
    assert (ha["n"], ha["wins"], ha["draws"], ha["losses"], ha["score"]) == (4, 2, 1, 1, 0.625)
    assert ha["a_holds"]["n"] == 4 and "b_holds" not in ha                 # A never held a
    # Paired on the 4 positions: C - H = 0, .5, 1, 0 (mean .375); pairing sums .5 and 1 of 2
    # games each, so every bootstrap replicate is .25, .375 or .5.
    (p,) = rep["paired"]
    assert (p["candidate"], p["champion"], p["opponent"], p["n"], p["n_groups"]) == \
        ("C", "H", "A", 4, 2)
    assert (p["candidate_score"], p["champion_score"], p["diff"]) == (1.0, 0.625, 0.375)
    assert 0.25 - 1e-12 <= p["lo"] <= p["hi"] <= 0.5 + 1e-12
    assert p["mcnemar"] == {"a_better": 2, "b_better": 0, "ties": 2, "p_value": 0.5}
    table = LS.fit_btd([(g.teams[0], g.teams[1], g.score(g.teams[0])) for g in games], anchor="A")
    r = rep["ratings"]
    assert r["anchor"] == "A" and {x["player"]: x["rating"] for x in r["rows"]} == table.rating
    assert r["rows"][0]["player"] == "C"                                    # 12 wins of 12
    assert {x["player"]: x["games"] for x in r["rows"]} == {"C": 12, "H": 12, "A": 8}
    gate = rep["gates"]["C"]
    assert (gate["decision"], gate["promote"], rep["best"]) == ("pass", True, "C")
    assert gate["conditions"]["anchor:A"]["observed"] == 1.0
    text = LE.format_report(rep)
    for part in ("C vs H: 100.0% [67.6%, 100.0%]", "C - H vs A: +37.5 pp", "C: PASS",
                 "best: C", "anchor:A: drop -37.5 pp over 4 games"):
        assert part in text, part
    json.dumps(rep)
    # The same games under another rating anchor and a failing bound.
    strict = LE.GateConfig(champion_games=8, anchor_games=4, n_boot=50, champion_lower_bound=0.7)
    rep = LE.report(games, candidates=["C"], champion="H", anchors=["A"], config=strict,
                    rating_anchor="H")
    assert rep["ratings"]["anchor"] == "H" and rep["gates"]["C"]["decision"] == "fail"
    assert rep["best"] is None and "best: none passed" in LE.format_report(rep)


def test_final_split_is_reported_only_with_final(le, tmp_path, capsys):
    LE = le
    conf = LE.GateConfig(champion_games=8, anchor_games=4, n_boot=50)
    games = small_log(LE, split="test")
    with pytest.raises(LE.FinalSplitError, match="held out"):
        LE.report(games, candidates=["C"], champion="H", anchors=["A"], config=conf)
    rep = LE.report(games, candidates=["C"], champion="H", anchors=["A"], config=conf, final=True)
    assert (rep["split"], rep["split_name"], rep["gates"]["C"]["decision"]) == ("test", "final",
                                                                               "pass")
    with pytest.raises(ValueError, match="several splits"):
        LE.report(small_log(LE) + games[:1], candidates=["C"], champion="H", config=conf)
    # The report subcommand on a log of the final split.
    path = tmp_path / "final.jsonl"
    LE.append_records(path, [{"kind": "league_run", "protocol": "P", "roles": {
        "candidates": ["C"], "champion": "H", "anchors": ["A"], "pool": []}}])
    LE.append_records(path, [record(g) for g in games])
    argv = ["report", "--log", str(path), "--champion-games", "8", "--anchor-games", "4",
            "--n-boot", "50"]
    with pytest.raises(SystemExit):
        LE.main(argv)
    assert "held out" in capsys.readouterr().err
    assert not (tmp_path / "final.report.json").exists()
    assert LE.main(argv + ["--final"]) == 0
    assert "C: PASS (vs champion H)" in capsys.readouterr().out
    out = json.loads((tmp_path / "final.report.json").read_text(encoding="utf-8"))
    assert (out["split"], out["protocol_digest"], out["gates"]["C"]["decision"]) == ("test", "P",
                                                                                    "pass")
    # Names on the command line override the header; a second protocol needs --protocol.
    LE.append_records(path, [record(dataclasses.replace(g, protocol="Q")) for g in games[:2]])
    with pytest.raises(SystemExit):
        LE.main(argv + ["--final"])
    assert "several protocols" in capsys.readouterr().err
    assert LE.main(argv + ["--final", "--protocol", "P", "--candidate", "C", "--champion",
                           "H"]) == 0


# ---------------------------------------------------------------------------
# the log, workers, the protocol
# ---------------------------------------------------------------------------

def test_log_reading_torn_lines_and_duplicates(le, tmp_path):
    LE = le
    path = tmp_path / "log.jsonl"
    gs = comparison(LE, "C", "H", [1, 0, 0.5])
    gs = [dataclasses.replace(g, ids=tuple(f"id-{t}" for t in g.teams)) for g in gs]
    LE.append_records(path, [{"kind": "league_run", "protocol": "P"}, record(gs[0])])
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write('{"kind": "league_ga')                                     # a crash mid-write
    LE.append_records(path, [record(gs[1]), record(gs[1]), record(gs[2]),
                             {**record(gs[2]), "key": "bad", "winner": 5},
                             {"kind": "league_error", "protocol": "P", "key": "x"}])
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write("[1, 2]\n\n")
    text = path.read_text(encoding="utf-8")
    # 2 lines, the torn one closed by the next append, 5 more, "[1, 2]" and an empty line.
    assert text.count("\n") == 10 and text.splitlines()[2] == '{"kind": "league_ga'
    log = LE.read_log(path)
    assert log.games == gs
    assert (log.bad_lines, log.duplicates, len(log.runs), len(log.errors)) == (3, 1, 1, 1)
    assert log.protocols() == Counter({"P": 3}) and log.done("P") == {g.key for g in gs}
    assert log.of("Q") == [] and log.last_run("P") == {"kind": "league_run", "protocol": "P"}
    assert [g.score("C") for g in log.games] == [1.0, 0.0, 0.5]
    assert gs[1].teams == ("H", "C") and gs[1].position("C") == ("val:p0000", 1, True)
    assert gs[2].position("C") == ("val:p0000", 1, False) and gs[2].other("C") == "H"
    assert LE.read_log(tmp_path / "missing.jsonl").games == []
    with pytest.raises(ValueError):
        LE.Game(key="k", teams=("C", "C"), winner=0)
    with pytest.raises(ValueError):
        LE.Game(key="k", teams=("C", "H"), winner=2)
    with pytest.raises(ValueError, match="did not play"):
        gs[0].score("A")
    assert log.games[1].ids == ("id-H", "id-C")
    # check_identities: a name whose games were another player's is refused.
    LE.check_identities(log, "P", {"C": "id-C", "H": "id-H", "A": "id-A"})
    with pytest.raises(ValueError, match="another player"):
        LE.check_identities(log, "P", {"C": "zz"})
    LE.check_identities(log, "Q", {"C": "zz"})                              # another protocol


def test_worker_cap(le):
    LE = le
    assert LE.worker_cap(8, available=8192, cpus=16) == (
        4, "4 worker(s), not 8: capped by memory 4 (8192 MB free, 1200 MB each, 3072 MB kept)")
    assert LE.worker_cap(2, available=8192, cpus=16) == (2, "2 worker(s)")
    assert LE.worker_cap(8, available=2048, cpus=16)[0] == 1
    assert LE.worker_cap(8, available=8192, cpus=4)[0] == 3                # one CPU kept free
    assert LE.worker_cap(8, available=16000, cpus=32)[0] == 6              # MAX_WORKERS
    assert LE.worker_cap(8, available=12000, cpus=16)[0] == 6
    assert LE.worker_cap(0, available=8192, cpus=16)[0] == 1
    free = LE.available_mb()
    assert free is None or free > 0
    assert 1 <= LE.worker_cap(4)[0] <= 4


def test_protocol_and_gate_config(le):
    LE = le
    p = LE.Protocol()
    assert (p.split, p.delay, p.delay_jitter, p.noise, p.mem_lag, p.h2h_decks, p.base_forms) == \
        ("val", 24, 0, "default", 1, "both", (True, False))
    assert LE.Protocol(split="dev").split == "val" and LE.Protocol(split="final").split == "test"
    assert LE.Protocol(noise="none").noise is None
    assert LE.Protocol(noise=None).digest() == LE.Protocol(noise="off").digest()
    assert LE.Protocol(noise=None).digest() != LE.Protocol(noise=None, delay=21).digest()
    assert LE.Protocol(mem_lag="21").mem_lag == 21 and LE.Protocol(mem_lag="Legacy").mem_lag == \
        "legacy"
    for bad in (dict(split="x"), dict(delay=-1), dict(delay=2, delay_jitter=3),
                dict(base_forms=(1, 0)), dict(h2h_decks="x"), dict(mem_lag="soon"),
                dict(seed=True)):
        with pytest.raises(ValueError):
            LE.Protocol(**bad)
    j = LE.Protocol(seed=5, delay_jitter=3, noise=None)
    assert [j.delay_for(t) for t in ("synthetic-0004", "synthetic-0016", "synthetic-0040")] == \
        [21, 22, 26]
    assert LE.Protocol(delay_jitter=3).delay_for("synthetic-0004") == 25       # seed 0
    assert LE.Protocol(noise=None).delay_for("anything") == 24
    engine = LE.rs_engine.EngineConfig.defaults()
    e = LE.Protocol(noise=None, engine=engine)
    assert e.engine == engine.to_json() and e.resolved() == e
    back = LE.Protocol.from_dict(json.loads(json.dumps(e.to_dict())))
    assert back == e and back.digest() == e.digest()
    with pytest.raises(ValueError):
        LE.Protocol.from_dict({**e.to_dict(), "colour": 1})
    miss = LE.Protocol(noise={"opponent_plays": {"miss_p": 0.5}}, engine=engine).resolved()
    assert miss.noise["opponent_plays"]["miss_p"] == 0.5 and miss.resolved() == miss
    prior = LE.Protocol(engine=engine).resolved()
    assert isinstance(prior.noise, dict) and prior.noise["name"] == "default_prior"
    assert prior.resolved() == prior and prior.digest() != e.digest()
    assert "noise default_prior" in prior.describe() and "delay 24" in prior.describe()
    assert LE.Protocol(noise={"opponent_plays": {"miss_p": 0.0}}, engine=engine).resolved() \
        .noise is None                                                     # all zero: none
    assert (LE.pairings_needed(1000, "both"), LE.pairings_needed(400, "a"),
            LE.pairings_needed(100, "both"), LE.pairings_needed(1001, "both")) == (250, 200, 25,
                                                                                  251)
    g = LE.GateConfig()
    assert (g.champion_games, g.champion_lower_bound, g.anchor_games, g.anchor_tolerance_pp,
            g.pool_members, g.pool_games) == (1000, 0.52, 400, 2.0, 8, 100)
    assert LE.pairings_for(g, LE.Protocol(noise=None)) == 250
    assert LE.pairings_for(g, LE.Protocol(noise=None, h2h_decks="a")) == 500
    for bad in (dict(champion_games=0), dict(champion_lower_bound=1.0),
                dict(anchor_tolerance_pp=-1), dict(z=0), dict(pool_members=-1)):
        with pytest.raises(ValueError):
            LE.GateConfig(**bad)


def test_select_pool_strata(le):
    LE = le
    ratings = {"m1": -100.0, "m2": 0.0, "m3": 50.0, "m4": 200.0}
    table = Table({"m1": 0.9, "m2": 0.6, "m3": 0.45, "m4": 0.2}, ratings)
    # Two strata, [m1, m2] and [m3, m4]: the most even matchup of each (0.24 > 0.09, .2475 > .16).
    assert LE.select_pool(["m4", "m3", "m2", "m1", "H"], table, "H", 2) == ("m2", "m3")
    assert LE.select_pool(["m4", "m1"], table, "H", 2) == ("m4", "m1")     # at most k: all
    assert LE.select_pool(["m1"], table, "H", 0) == ()
    assert LE.select_pool(["m3", "m1", "m2", "m4"], None, "H", 2) == ("m1", "m3")


# ---------------------------------------------------------------------------
# players and planning (rs_match)
# ---------------------------------------------------------------------------

def test_players(tools):
    M, LE = tools.match, tools.league
    assert LE.parse_player("cand=runs/x/best.pt,lead=-24,decide=legacy,noise=none,"
                           "spell-aim=yes,wait_keep=0.25") == \
        M.AgentSpec("runs/x/best.pt", name="cand", lead=-24, decide="legacy", noise="none",
                    spell_aim=True, wait_keep=0.25)
    assert LE.parse_player("runs/tokens/live_big_hog/best.pt").label == "live_big_hog"
    assert LE.parse_player('{"ckpt": "a.pt", "name": "x", "lead": 3}') == \
        M.AgentSpec("a.pt", name="x", lead=3)
    assert LE.parse_player({"ckpt": "a.pt", "name": "y"}) == M.AgentSpec("a.pt", name="y")
    assert LE.parse_player("floor=idle").idle and LE.parse_player("floor=idle").label == "floor"
    assert LE.parse_player("a.pt", decide="legacy").decide == "legacy"
    assert LE.parse_player("a.pt,decide=sample", decide="legacy").decide == "sample"
    spec = M.AgentSpec("a.pt", name="z")
    assert LE.parse_player(spec, decide="legacy") is spec
    for bad, match in (("a.pt,colour=red", "unknown player option"), ("a.pt,lead=x", "an int"),
                       ("a.pt,lead", "KEY=VALUE"), ("a.pt,lead=1,lead=2", "twice"),
                       ("a.pt,decide=greedy", "decide"), ("a.pt,spell_aim=maybe", "true or")):
        with pytest.raises(ValueError, match=match):
            LE.parse_player(bad)
    proto = LE.Protocol(noise={"opponent_plays": {"miss_p": 0.5}}, engine=cfg(tools)).resolved()
    roles = LE.prepare_players(["c=c.pt", "p=p.pt,noise=none"], "h=h.pt", ["floor=idle"],
                               protocol=proto)
    assert (roles.candidates, roles.champion, roles.anchors, roles.pool) == (("c", "p"), "h",
                                                                            ("floor",), ())
    assert roles.players["c"].noise == proto.noise == roles.players["h"].noise
    assert roles.players["p"].noise == "none" and roles.players["floor"].noise is None
    assert roles.to_dict() == {"candidates": ["c", "p"], "champion": "h", "anchors": ["floor"],
                               "pool": []}
    with pytest.raises(ValueError, match="used twice"):
        LE.prepare_players(["runs/a/best.pt"], "runs/a/ep1.pt")
    with pytest.raises(ValueError, match="candidate"):
        LE.prepare_players([], "h.pt")
    assert LE.ckpt_digest("idle") is None and LE.ckpt_digest("no/such.pt") is None


def test_comparison_tasks_are_rs_match_pairings(tools):
    """Golden: a comparison's games are rs_match.pairing_specs' games (L02, unchanged)."""
    M, LE = tools.match, tools.league
    pairs = M.read_pairs(synthetic_rows(), NAMES, "dev", require=M.HOG26)
    assert pairs.tags[:2] == ["synthetic-0004", "synthetic-0016"]
    C, H, A = M.AgentSpec("c.pt", name="C"), M.AgentSpec("h.pt", name="H"), M.AgentSpec(name="A")
    proto = LE.Protocol(noise=None, engine=cfg(tools), seed=5)
    kw = dict(seed=5, base_forms=(True, False), delay=24, engine=proto.engine, mem_lag=1,
              max_ticks=6200)
    h2h = LE.comparison_tasks(pairs, proto, C, H, games=8, decks="both")
    ab = M.pairing_specs(pairs.pairs[:2], (C, H), **kw)
    ba = M.pairing_specs(pairs.pairs[:2], (H, C), **kw)
    assert [t.spec for t in h2h] == [ab[0], ab[1], ba[0], ba[1], ab[2], ab[3], ba[2], ba[3]]
    assert [t.holder for t in h2h] == ["C", "C", "H", "H"] * 2
    assert all((t.a, t.b) == ("C", "H") for t in h2h)
    assert [t.spec.agents for t in h2h[:4]] == [(C, H), (H, C), (H, C), (C, H)]
    assert [t.spec.seat for t in h2h[:4]] == [0, 1, 0, 1]
    assert {t.spec.seed for t in h2h[:4]} == {SEEDS5[0]} and h2h[4].spec.seed == SEEDS5[1]
    # The decks exchanged: the same deals, H on side a's deck (in base forms) at team 0.
    assert h2h[2].spec.decks == h2h[0].spec.decks and h2h[2].spec.forms == h2h[0].spec.forms
    assert h2h[0].spec.forms[0] == (0,) * 8 and h2h[0].spec.delay == 24
    keys = [t.key for t in h2h]
    assert len(set(keys)) == 8 and keys == [t.key for t in LE.comparison_tasks(
        pairs, proto, C, H, games=8, decks="both")]
    assert LE.comparison_tasks(pairs, proto, C, H, games=8, decks="both",
                               digests={"C": "abc"})[0].key != keys[0]
    assert len(LE.comparison_tasks(pairs, proto, C, H, games=7, decks="both")) == 8
    # Against an anchor: 2 games a pairing, and the champion's games are on the candidate's
    # positions (the same game but for the player).
    ca = LE.comparison_tasks(pairs, proto, C, A, games=4)
    ha = LE.comparison_tasks(pairs, proto, H, A, games=4)
    assert [t.spec for t in ca] == M.pairing_specs(pairs.pairs[:2], (C, A), **kw)
    assert len(ca) == len(ha) == 4 and [t.holder for t in ca] == ["C"] * 4
    for c, h in zip(ca, ha):
        swap = tuple(H if x.label == "C" else x for x in c.spec.agents)
        assert c.spec.replace(agents=swap) == h.spec and c.key != h.key
    with pytest.raises(ValueError, match="take 2 pairings"):
        LE.comparison_tasks(pairs.pairs[:1], proto, C, H, games=8, decks="both")
    with pytest.raises(ValueError):
        LE.comparison_tasks(pairs, proto, C, H, games=8, decks="all")
    # A per-pairing delay: 21 for synthetic-0004, 22 for synthetic-0016 (seed 5, +-3).
    jit = LE.comparison_tasks(pairs, proto.replace(delay_jitter=3), C, A, games=4)
    assert [t.spec.delay for t in jit] == [21, 21, 22, 22]


# ---------------------------------------------------------------------------
# evaluate: phases, resume, early stop, identities, errors
# ---------------------------------------------------------------------------

STRENGTH = {"C": 3, "H": 2, "W": 1, "A": 0}


def stub_play(M, calls, strength=STRENGTH):
    """play(spec): the stronger agent wins, equal ones draw; the spec is kept in `calls`."""
    def play(spec):
        calls.append(spec)
        names = tuple(a.label for a in spec.agents)
        s0, s1 = (strength[n] for n in names)
        winner = 0 if s0 > s1 else 1 if s1 > s0 else -1
        return M.MatchResult(
            id=spec.id, tag=spec.tag, pairing=spec.pairing, seat=spec.seat, seed=spec.seed,
            agents=names, winner=winner, crowns=(int(winner == 0), int(winner == 1)),
            game_over=True, end_tick=3600, plays=(30, 30), refused=(0, 0), decisions=(300, 300),
            full_ticks=(0, 0), leak=(0.0, 0.0), towers=((0, 0, 0), (0, 0, 0)),
            delay=(spec.delay, spec.delay))
    return play


def small_eval(tools, tmp_path, play, **kw):
    M, LE = tools.match, tools.league
    pairs = M.read_pairs(synthetic_rows(), NAMES, "dev", require=M.HOG26)
    args = dict(candidates=["C=c.pt", "W=w.pt"], champion="H=h.pt", anchors=["A=a.pt"],
                config=LE.GateConfig(champion_games=8, anchor_games=4, pool_games=4, n_boot=50),
                log_path=tmp_path / "dev.jsonl", play=play)
    args.update(kw)
    return LE.evaluate(pairs, LE.Protocol(noise=None, engine=cfg(tools)), **args)


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def test_evaluate_phases_resume_and_early_stop(tools, tmp_path):
    M, LE = tools.match, tools.league
    calls = []
    rep = small_eval(tools, tmp_path, stub_play(M, calls))
    # C beats H 8 of 8 (LB 0.6756) and A as H does: pass. W loses 8 of 8 to H: fail, and it
    # plays no anchor games (stopped early).
    assert len(calls) == 24
    assert rep["phases"] == {
        "champion": {"planned": 16, "skipped": 0, "played": 16, "errors": 0},
        "anchors": {"planned": 8, "skipped": 0, "played": 8, "errors": 0},
        "pool": {"planned": 0, "skipped": 0, "played": 0, "errors": 0}}
    gates = rep["gates"]
    assert (gates["C"]["decision"], gates["W"]["decision"], rep["best"]) == ("pass", "fail", "C")
    assert gates["C"]["conditions"]["beats_champion"]["bound"] == pytest.approx(0.6755843804891231)
    w_anchor = gates["W"]["conditions"]["anchor:A"]
    assert (w_anchor["skipped"], w_anchor["reason"]) == (True, LE.SKIP_SETTLED)
    assert (rep["n_games"], rep["errors"], rep["split"]) == (24, 0, "val")
    log = lines(tmp_path / "dev.jsonl")
    head, games = log[0], [x for x in log if x["kind"] == "league_game"]
    assert head["kind"] == "league_run" and len(log) == 25 and len(games) == 24
    assert head["roles"] == {"candidates": ["C", "W"], "champion": "H", "anchors": ["A"],
                             "pool": []}
    assert head["protocol"] == rep["protocol_digest"] and head["stop_early"] is True
    first = games[0]
    assert (first["teams"], first["holder"], first["a"], first["b"], first["seat"]) == \
        (["C", "H"], "C", "C", "H", 0)
    assert (first["pairing"], first["winner"], first["seed"]) == ("val:synthetic-0004", 0,
                                                                  SEED0_0004)
    assert first["ids"] == [head["players"]["C"]["id"], head["players"]["H"]["id"]]
    assert first["protocol"] == rep["protocol_digest"] and first["delay"] == [24, 24]
    assert [(g["teams"], g["holder"]) for g in games[1:4]] == [
        (["H", "C"], "C"), (["H", "C"], "H"), (["C", "H"], "H")]
    assert {g["seed"] for g in games[:4]} == {SEED0_0004}
    assert [g["a"] for g in games[16:]] == ["H"] * 4 + ["C"] * 4
    # A rerun plays nothing and reports the same.
    again = small_eval(tools, tmp_path, stub_play(M, calls))
    assert len(calls) == 24 and again["gates"] == rep["gates"]
    assert again["phases"]["champion"] == {"planned": 16, "skipped": 16, "played": 0, "errors": 0}
    # --full: W plays the anchor too (4 games); still a fail, on the champion condition.
    full = small_eval(tools, tmp_path, stub_play(M, calls), stop_early=False)
    assert len(calls) == 28 and full["phases"]["anchors"] == {
        "planned": 12, "skipped": 8, "played": 4, "errors": 0}
    w = full["gates"]["W"]
    assert (w["decision"], w["turned_on"]) == ("fail", "beats_champion")
    assert w["conditions"]["anchor:A"]["passed"] and not w["conditions"]["anchor:A"]["skipped"]
    # A changed player under the same name and protocol is refused before anything is played.
    with pytest.raises(ValueError, match="another player"):
        small_eval(tools, tmp_path, stub_play(M, calls), candidates=["C=other.pt"])
    assert len(calls) == 28 and len(lines(tmp_path / "dev.jsonl")) == 25 + 1 + 1 + 4


def test_evaluate_logs_and_retries_errors(tools, tmp_path):
    M = tools.match
    calls = []
    good = stub_play(M, calls)

    def flaky(spec):                                       # the second game raises
        if len(calls) == 1:
            calls.append(spec)
            raise RuntimeError("boom")
        return good(spec)

    rep = small_eval(tools, tmp_path, flaky)
    c = rep["gates"]["C"]
    # C is one champion game short, so (stop_early) it played no anchor games: the decision
    # turns on beats_champion, and anchor:A, not settled, still lacks all 4 of its games
    # (Condition.missing); the rerun below plays them.
    assert (c["decision"], c["missing"], c["turned_on"]) == (
        "incomplete", {"beats_champion": 1, "anchor:A": 4}, "beats_champion")
    a = c["conditions"]["anchor:A"]
    assert (a["n"], a["skipped"], a["incomplete"]) == (0, False, True)
    assert rep["phases"]["champion"] == {"planned": 16, "skipped": 0, "played": 15, "errors": 1}
    assert rep["phases"]["anchors"]["planned"] == 0 and rep["errors"] == 1
    (err,) = [x for x in lines(tmp_path / "dev.jsonl") if x["kind"] == "league_error"]
    assert err["error"] == "RuntimeError: boom" and (err["a"], err["b"], err["seat"]) == \
        ("C", "H", 1)
    n = len(calls)
    rep = small_eval(tools, tmp_path, good)
    assert len(calls) == n + 1 + 8                         # the failed game, then the anchors
    assert calls[n].seat == 1 and rep["gates"]["C"]["decision"] == "pass"
    assert rep["phases"]["champion"] == {"planned": 16, "skipped": 15, "played": 1, "errors": 0}

    def broken(spec):
        raise RuntimeError("no engine")

    with pytest.raises(RuntimeError, match="more than 0"):
        small_eval(tools, tmp_path, broken, log_path=tmp_path / "other.jsonl", max_errors=0)
    assert [x["kind"] for x in lines(tmp_path / "other.jsonl")] == ["league_run", "league_error"]


#: A cycle: C beats H, H beats M, C only draws M; both beat the anchor A. (team 0, team 1)
#: pairs listed here are won by their first player, every other pair is a draw.
BEATS = frozenset({("C", "H"), ("C", "A"), ("H", "A"), ("H", "M")})


def relation_play(M, calls, beats=BEATS):
    """play(spec): the winner from `beats`, a draw for pairs it does not list."""
    def play(spec):
        calls.append(spec)
        names = tuple(a.label for a in spec.agents)
        winner = 0 if names in beats else 1 if names[::-1] in beats else -1
        return M.MatchResult(
            id=spec.id, tag=spec.tag, pairing=spec.pairing, seat=spec.seat, seed=spec.seed,
            agents=names, winner=winner, crowns=(int(winner == 0), int(winner == 1)),
            game_over=True, end_tick=3600, plays=(30, 30), refused=(0, 0), decisions=(300, 300),
            full_ticks=(0, 0), leak=(0.0, 0.0), towers=((0, 0, 0), (0, 0, 0)),
            delay=(spec.delay, spec.delay))
    return play


def test_evaluate_pool_phase_plays_the_champion_and_finds_a_cycle(tools, tmp_path):
    """C beats H 8 of 8 and A as H does, but only draws M, which H beats 4 of 4. The pool phase
    plays H against M on C's positions, so pool_collapse's reference is H's own 1.0 and C's 0.5
    falls short (se 0: one pairing): a cycle. (A reference fitted on C's games alone puts M level
    with C, above H, and would pass C.)"""
    M, LE = tools.match, tools.league
    calls = []
    rep = small_eval(tools, tmp_path, relation_play(M, calls), candidates=["C=c.pt"],
                     pool=["M=m.pt"])
    assert rep["phases"]["pool"] == {"planned": 8, "skipped": 0, "played": 8, "errors": 0}
    assert len(calls) == 8 + 8 + 8
    played = [x for x in lines(tmp_path / "dev.jsonl") if x["kind"] == "league_game"]
    assert [(x["a"], x["b"]) for x in played[-8:]] == [("H", "M")] * 4 + [("C", "M")] * 4
    g = rep["gates"]["C"]
    assert (g["decision"], g["admit"], g["promote"], g["turned_on"]) == (
        "cycle", True, False, "pool_collapse")
    p = g["conditions"]["pool_collapse"]
    assert (p["n"], p["missing"], p["observed"], p["reference"], p["bound"]) == (4, 0, 0.5, 1.0,
                                                                                 1.0)
    assert p["detail"]["champion_observed"] == {"M": 1.0}
    assert p["detail"]["source"] == {"M": "observed"} and p["detail"]["reference"] == {"M": 1.0}
    assert (p["detail"]["se"], p["detail"]["groups"]) == (0.0, 1)
    assert rep["best"] is None and [r["opponent"] for r in rep["paired"]] == ["A", "M"]
    assert "pool_collapse: mean 50.0% vs the champion's observed 100.0%" in LE.format_report(rep)


def test_evaluate_refuses_the_final_split(tools, tmp_path):
    M, LE = tools.match, tools.league
    final = M.read_pairs(synthetic_rows(), NAMES, "final", require=M.HOG26, allow_final=True)
    calls = []
    with pytest.raises(LE.FinalSplitError, match="held out"):
        LE.evaluate(final, LE.Protocol(split="final", noise=None, engine=cfg(tools)),
                    candidates=["C=c.pt"], champion="H=h.pt", log_path=tmp_path / "f.jsonl",
                    play=stub_play(M, calls))
    assert calls == [] and not (tmp_path / "f.jsonl").exists()
    with pytest.raises(ValueError, match="split"):
        LE.evaluate(final, LE.Protocol(noise=None, engine=cfg(tools)), candidates=["C=c.pt"],
                    champion="H=h.pt", log_path=tmp_path / "f.jsonl", play=stub_play(M, calls))
    # A plain list of the final split's DeckPairs (no set split) is caught by the pairs' own.
    with pytest.raises(ValueError, match="pairs are of the test split"):
        LE.evaluate(list(final.pairs), LE.Protocol(noise=None, engine=cfg(tools)),
                    candidates=["C=c.pt"], champion="H=h.pt", log_path=tmp_path / "f.jsonl",
                    play=stub_play(M, calls))
    assert calls == [] and not (tmp_path / "f.jsonl").exists()
    with pytest.raises(SystemExit):
        LE.main(["run", "--candidate", "c=idle", "--champion", "h=idle", "--split", "final"])
    # With final=True it plays: 8 wins of 8 (at n = 4 even 4 of 4 has LB 0.5101 < 0.52).
    rep = LE.evaluate(final, LE.Protocol(split="final", noise=None, engine=cfg(tools)),
                      candidates=["C=c.pt"], champion="H=h.pt", log_path=tmp_path / "f.jsonl",
                      play=stub_play(M, calls), final=True,
                      config=LE.GateConfig(champion_games=8, anchor_games=2, n_boot=20))
    assert (rep["split_name"], rep["gates"]["C"]["decision"], len(calls)) == ("final", "pass", 8)
    assert rep["matchups"][0]["a"] == "C" and rep["matchups"][0]["wins"] == 8


def test_evaluate_idle_seats_on_the_fake_engine(fake_royalesim, tools, tmp_path):
    """No stub: rs_match.run_match plays idle seats on the fake royalesim under the delay;
    every game is a draw (the fake never ends a game), so the candidate fails the bound."""
    M, LE = tools.match, tools.league
    pairs = M.read_pairs(synthetic_rows(), NAMES, "dev", require=M.HOG26)
    rep = LE.evaluate(pairs, LE.Protocol(noise=None, engine=cfg(tools), max_ticks=200),
                      candidates=["c=idle"], champion="h=idle", anchors=["a=idle"],
                      config=LE.GateConfig(champion_games=8, anchor_games=4, n_boot=50),
                      log_path=tmp_path / "dev.jsonl")
    assert rep["n_games"] == 8 and rep["phases"]["anchors"]["planned"] == 0
    g = rep["gates"]["c"]
    assert g["decision"] == "fail"
    assert g["conditions"]["beats_champion"]["bound"] == pytest.approx(0.21521252682444186)
    (m,) = rep["matchups"]
    assert (m["a"], m["b"], m["draws"], m["score"]) == ("c", "h", 8, 0.5)
    matches = [b for b in fake_royalesim.battles if b.reset_kwargs is not None]
    assert len(matches) == 8 and len(fake_royalesim.battles) == 16       # _setup's and the match's
    for b in matches:
        assert b.calls_named("set_command_delay_ticks")[0].args == (24, 24)
        assert b.reset_kwargs["levels"] == [[11] * 8, [11] * 8]
    assert len({b.reset_kwargs["seed"] for b in matches[:4]}) == 1
    games = [x for x in lines(tmp_path / "dev.jsonl") if x["kind"] == "league_game"]
    assert all(x["winner"] == -1 and x["result"]["end_tick"] == 200 for x in games)
    assert all("log" not in x["result"] for x in games)


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

@pytest.mark.engine
def test_engine_idle_league_smoke(tools, tmp_path):
    M, LE = tools.match, tools.league
    pairs = M.read_pairs(synthetic_rows(), NAMES, "dev", require=M.HOG26)
    rep = LE.evaluate(pairs, LE.Protocol(noise=None, engine=cfg(tools), max_ticks=200),
                      candidates=["c=idle"], champion="h=idle",
                      config=LE.GateConfig(champion_games=4, anchor_games=2, n_boot=20),
                      log_path=tmp_path / "e.jsonl")
    assert rep["n_games"] == 4 and rep["matchups"][0]["draws"] == 4
    assert rep["gates"]["c"]["decision"] == "fail"
