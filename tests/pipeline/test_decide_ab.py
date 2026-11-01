"""tools/decide_ab.py: the decision rules' A/B verdict read from a league log.

The logs are real league logs (tools/rs_league_eval.evaluate on conftest's synthetic IL_Replay
pairings with a stub play, as tests/pipeline/test_league_eval.py plays them): the candidate is the
champion's checkpoint file with the rules on, the champion the same file with them off. Checked:
the verdict's two-interval rule at its boundaries; STRONGER, WEAKER and NO DIFFERENCE on planted
results; INCOMPLETE (exit code 2) below --games; the per-game plays, leak and rule counts from
the game records; the players' rules and digests from the run header and the warning for two
checkpoints; the protocol chosen; the printed lines scripts/decide_ab.ps1 copies.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import FIELD

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


@pytest.fixture
def tools(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    rm = importlib.import_module("rs_match")
    le = importlib.import_module("rs_league_eval")
    ab = importlib.import_module("decide_ab")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    monkeypatch.setattr(rr, "F", dict(FIELD))
    return types.SimpleNamespace(rows=rr, match=rm, league=le, ab=ab, engine=eng)


def league_log(tools, tmp_path, outcome, *, games=48, cand_ckpt=None, name="R",
               rules="wait_floor=0.95,stray_guard=on", log=None, noise=None):
    """A league log of `name` (rules on) against H (rules off), the same checkpoint file unless
    `cand_ckpt`; outcome(i) -> R's score in its i-th game (1, 0.5, 0). The seat info of R's
    games carries floor_waits 2 and stray_moved 1 a game; R plays 30 cards and leaks 0.5 a
    game, H 34 and 0.25."""
    from tests.pipeline.test_league_eval import NAMES, synthetic_rows
    M, LE = tools.match, tools.league
    ckpt = tmp_path / "v2_hog" / "best.pt"
    ckpt.parent.mkdir(exist_ok=True)
    if not ckpt.exists():
        ckpt.write_bytes(b"the same weights")
    cand = cand_ckpt or ckpt
    pairs = M.read_pairs(synthetic_rows(), NAMES, "dev", require=M.HOG26)
    count = [0]

    def play(spec):
        names = tuple(a.label for a in spec.agents)
        t = names.index(name)
        s = outcome(count[0])
        count[0] += 1
        winner = -1 if s == 0.5 else (t if s == 1 else 1 - t)
        info = [M.seat_info(None, a) for a in spec.agents]
        if "rules" in info[t]:
            info[t]["rules"].update(floor_waits=2, stray_moved=1)
        plays, leak = [34, 34], [0.25, 0.25]
        plays[t], leak[t] = 30, 0.5
        return M.MatchResult(id=spec.id, tag=spec.tag, pairing=spec.pairing, seat=spec.seat,
                             seed=spec.seed, agents=names, winner=winner, crowns=(0, 0),
                             game_over=True, end_tick=3600, plays=tuple(plays), refused=(0, 0),
                             decisions=(300, 300), full_ticks=(0, 0), leak=tuple(leak),
                             towers=((0,) * 3, (0,) * 3), delay=(spec.delay, spec.delay),
                             info=tuple(info))

    log = log or tmp_path / "decide_ab.jsonl"
    LE.evaluate(pairs, LE.Protocol(noise=noise, engine=tools.engine.EngineConfig.defaults()),
                candidates=[f"{name}={cand}" + (f",{rules}" if rules else "")],
                champion=f"H={ckpt}", config=LE.GateConfig(champion_games=games, n_boot=200),
                log_path=log, play=play)
    return log


def test_verdict_needs_both_intervals(tools):
    v = tools.ab.verdict_of
    assert v(0.5001, 0.6, 0.5001, 0.61) == tools.ab.STRONGER
    assert v(0.5, 0.6, 0.51, 0.6) == tools.ab.NO_DIFFERENCE          # Wilson touches 50%
    assert v(0.51, 0.6, 0.49, 0.6) == tools.ab.NO_DIFFERENCE         # the bootstrap holds it
    assert v(0.4, 0.4999, 0.39, 0.4999) == tools.ab.WEAKER
    assert v(0.4, 0.49, 0.39, 0.51) == tools.ab.NO_DIFFERENCE
    assert set(tools.ab.NEXT) == {tools.ab.STRONGER, tools.ab.NO_DIFFERENCE, tools.ab.WEAKER}


def test_stronger_on_a_league_log(tools, tmp_path):
    log = league_log(tools, tmp_path, lambda i: 1.0)
    v = tools.ab.ab(log, "R", "H", games=48)
    assert (v["verdict"], v["games"], v["complete"], v["score"]) == ("stronger", 48, True, 1.0)
    assert v["lo"] > 0.9 and v["boot_lo"] > 0.5 and v["pairings"] == 12
    assert (v["wins"], v["draws"], v["losses"]) == (48, 0, 0)
    assert v["players"]["R"]["rules"] == {"wait_floor": 0.95, "stray_guard": True}
    assert v["players"]["H"]["rules"] == {} and v["same_checkpoint"]
    assert v["players"]["R"]["ckpt_digest"] == v["players"]["H"]["ckpt_digest"]
    assert v["per_game"]["R"] == {"games": 48, "plays": 30.0, "leak": 0.5,
                                  "rules": {"floor_waits": 2.0, "stray_moved": 1.0}}
    assert v["per_game"]["H"] == {"games": 48, "plays": 34.0, "leak": 0.25, "rules": {}}
    lines = tools.ab.format_ab(v, "decide_ab.jsonl")
    assert lines[0].startswith("decide_ab: R against H in decide_ab.jsonl (protocol ")
    assert "dev split" in lines[0]
    assert lines[1].startswith("  R: rules wait floor 0.95, stray guard; checkpoint ")
    assert lines[2].startswith("  H: rules none; checkpoint ")
    assert not any("warning" in x for x in lines)
    assert any(x.startswith("  score 100.0% over 48 games (48 W 0 D 0 L); 95% [") for x in lines)
    assert ("  a game: plays 30.0 (H 34.0), elixir leak 0.50 (0.25); R's rules: floor waits "
            "2.00, stray moved 1.00") in lines
    verdict = [x for x in lines if x.startswith("verdict:")]
    assert len(verdict) == 1 and verdict[0].startswith("verdict: STRONGER: R scores 100.0% "
                                                       "against H, 95% [")
    assert verdict[0].endswith("(both 95% intervals above 50%)")
    assert lines[-1] == "  " + tools.ab.NEXT["stronger"]
    assert not any("INCOMPLETE" in x for x in lines)


def test_weaker_no_difference_and_incomplete(tools, tmp_path, capsys):
    (tmp_path / "w").mkdir()
    weak = league_log(tools, tmp_path / "w", lambda i: 0.0)
    v = tools.ab.ab(weak, "R", "H", games=48)
    assert (v["verdict"], v["score"]) == ("weaker", 0.0)
    assert tools.ab.format_ab(v)[-2].startswith("verdict: WEAKER: R scores 0.0% against H")
    (tmp_path / "e").mkdir()
    even = league_log(tools, tmp_path / "e", lambda i: (1.0, 0.0, 0.5, 0.5)[i % 4])
    v = tools.ab.ab(even, "R", "H", games=1000)
    assert (v["verdict"], v["score"], v["complete"], v["games_wanted"]) == \
        ("no difference", 0.5, False, 1000)
    lines = tools.ab.format_ab(v)
    assert any(x.startswith("verdict: NO DIFFERENCE: R scores 50.0% against H") for x in lines)
    assert "  INCOMPLETE: 48 of 1000 games; run the same command again, the league plays only " \
           "the games its log lacks" in lines
    # the command line: 2 for an incomplete verdict, 0 for a complete one, 1 for an error
    assert tools.ab.main(["--log", str(even), "--champion", "H", "--candidate", "R"]) == 2
    assert tools.ab.main(["--log", str(even), "--champion", "H", "--candidate", "R",
                          "--games", "48"]) == 0
    out = capsys.readouterr().out
    assert out.count("verdict: NO DIFFERENCE") == 2
    assert tools.ab.main(["--log", str(even), "--champion", "H", "--candidate", "X"]) == 1
    assert tools.ab.main(["--log", str(tmp_path / "none.jsonl"), "--champion", "H",
                          "--candidate", "R"]) == 1
    assert "no games" in capsys.readouterr().err


def test_two_checkpoints_warn_and_the_protocol_is_the_last_run(tools, tmp_path):
    other = tmp_path / "v2_hog_other" / "best.pt"
    other.parent.mkdir()
    other.write_bytes(b"other weights")
    log = league_log(tools, tmp_path, lambda i: 1.0, cand_ckpt=other)
    v = tools.ab.ab(log, "R", "H", games=48)
    assert not v["same_checkpoint"]
    assert any(x.startswith("  warning: the two players are not the same checkpoint")
               for x in tools.ab.format_ab(v))
    # a second protocol (noise on) in the same log: the last run's protocol is read; --protocol
    # picks the other
    first = v["protocol"]
    league_log(tools, tmp_path, lambda i: 0.0, cand_ckpt=other, log=log,
               noise={"opponent_plays": {"miss_p": 0.5}})
    w = tools.ab.ab(log, "R", "H", games=48)
    assert w["protocol"] != first and w["verdict"] == "weaker"
    assert tools.ab.ab(log, "R", "H", games=48, protocol=first)["verdict"] == "stronger"
    with pytest.raises(ValueError, match="no games"):
        tools.ab.ab(log, "R", "nobody")


def test_a_candidate_without_rules_reads_none(tools, tmp_path):
    log = league_log(tools, tmp_path, lambda i: 0.5, rules="")
    v = tools.ab.ab(log, "R", "H", games=48)
    assert v["players"]["R"]["rules"] == {} and v["per_game"]["R"]["rules"] == {}
    assert "R's rules: never fired" in "\n".join(tools.ab.format_ab(v))


def test_the_verdict_line_prints_both_intervals_and_names_the_one_holding_50(tools):
    """Review finding 4: Wilson above 50% while the bootstrap holds it is NO DIFFERENCE, and
    the line says so with both intervals, not Wilson's alone."""
    base = {"candidate": "R", "champion": "H", "protocol": "p", "split": "", "games": 1000,
            "games_wanted": 1000, "complete": True, "score": 0.54, "pairings": 250,
            "wins": 520, "draws": 40, "losses": 440, "same_checkpoint": True,
            "players": {n: {"ckpt": "x.pt", "ckpt_digest": "d", "rules": {}} for n in "RH"},
            "per_game": {"R": {"games": 0}, "H": {"games": 0}}}
    cases = (((0.51, 0.57, 0.49, 0.59), "the bootstrap 95% interval holds 50%"),
             ((0.49, 0.57, 0.51, 0.56), "the Wilson 95% interval holds 50%"),
             ((0.47, 0.55, 0.46, 0.56), "both 95% intervals hold 50%"),
             ((0.51, 0.57, 0.505, 0.58), "both 95% intervals above 50%"))
    for (lo, hi, blo, bhi), why in cases:
        v = {**base, "lo": lo, "hi": hi, "boot_lo": blo, "boot_hi": bhi}
        v["verdict"] = tools.ab.verdict_of(lo, hi, blo, bhi)
        line = next(x for x in tools.ab.format_ab(v) if x.startswith("verdict:"))
        word = "STRONGER" if v["verdict"] == tools.ab.STRONGER else "NO DIFFERENCE"
        assert line == (f"verdict: {word}: R scores 54.0% against H, 95% [{100 * lo:.1f}%, "
                        f"{100 * hi:.1f}%] Wilson, [{100 * blo:.1f}%, {100 * bhi:.1f}%] "
                        f"bootstrap ({why})"), line
