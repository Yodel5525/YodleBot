"""tools/rl/train_loop.py (R04): the RL driver.

On conftest's fake royalesim with tests/pipeline/test_rollout.py's setup (its `tools` fixture, the
scripted battle that blue wins at tick 200, tiny TokenPolicies), every step in this process
(InlineSteps: rollout.collect with no worker processes, ppo.main on the CPU), and the gate a
stand-in that writes the report rs_league_eval would. Checked:
  - a two-iteration smoke run: it0000.pt is the start with blob["rl"] and no PPO state, a
    collection and a PPO update per iteration (it0001.pt carries the stamp and the PPO state),
    args.json (frozen flags, engine id, digests, the start's wait_keep that rs_decide reads for
    every checkpoint of the run folder), state.json / it<n>.json / log.jsonl / league.json, the
    league's filed results, the rollouts deleted after the update; collect-task replays a task
  - resume picks up the last iteration (one collection and one update, not two); a failed update
    resumes at the update without collecting again
  - a gate failure keeps the champion; a pass promotes the candidate (league, champion.pt,
    champion.json); the gate's command line; --gate-now; status
  - STOP (the run's, or the repo's) ends the run at the next step boundary, and a launch is
    refused while it exists
  - refusals: a resume of nothing, a new run over an existing one, a changed frozen setting,
    another engine, a changed start checkpoint, a legacy start, bad arguments
  - --rollback goes back to an earlier iteration and the run continues from there
  - the opponents' mix (MixDraw, also through rollout.plan_battles), the bench end to end and its
    recommendation, durations, the JSON line a step prints, run_logged on a real subprocess and
    the subprocess steps' command lines
No engine, no GPU, no data, no emulator.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from tests.pipeline import test_rollout as TR

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: test_rollout's fixture: the tools as they import each other, rs_rows' settings at their
#: defaults, no engine config from the PC, the fake catalogue for RSAgent.
tools = TR.tools

WAIT_KEEP = 0.25                            # the start's args.json; the default would be 0.35


@pytest.fixture
def loop(tools, monkeypatch, tmp_path):
    """train_loop on the fake engine: a tiny v2 start checkpoint (with an args.json, as
    sim/train_tokens writes one) and a second one for an extra anchor, the repo's STOP file
    moved under tmp_path, runs under tmp_path/rl."""
    pytest.importorskip("torch")
    TR.fake_engine(monkeypatch, states=TR.script)
    TL = importlib.import_module("tools.rl.train_loop")
    monkeypatch.setattr(TL, "REPO_STOP", tmp_path / "repo" / "STOP")
    start = tmp_path / "tokens" / "v2_hog" / "best.pt"
    TR.save_ckpt(start, TR.meta_for())
    (start.parent / "args.json").write_text(json.dumps({"wait_keep": WAIT_KEEP}), encoding="utf-8")
    other = tmp_path / "tokens" / "other" / "best.pt"
    TR.save_ckpt(other, TR.meta_for(), seed=1)
    return types.SimpleNamespace(TL=TL, tools=tools, start=start, other=other,
                                 runs=tmp_path / "rl", tmp=tmp_path)


def argv(env, *extra, name="r1", start=None):
    """A run on the fake engine: 4 battles an iteration, 2 open at once, in this process, cut
    at tick 400 (the scripted battle ends at 200), one opponent deck, no perception noise."""
    return ["run", "--name", name, "--runs-dir", str(env.runs), "--start", str(start or env.start),
            "--matches", "4", "--battles", "2", "--workers", "0", "--threads", "0",
            "--max-ticks", "400", "--opponent-deck", ",".join(TR.OPP), "--noise", "none",
            "--ppo-device", "cpu", "--precision", "fp32", "--inline", *extra]


def recording(TL, *, gate="fail", fail_update=0, stop=None):
    """InlineSteps that record what they run; the update can fail on purpose (`fail_update`
    times) or touch `stop` after it ran; the gate writes a report with `gate` as the decision."""

    class Recording(TL.InlineSteps):
        def __init__(self):
            self.calls, self.gate_argv, self.fail_update = [], [], fail_update

        def collect(self, task_path, log_path, echo, poll=None):
            self.calls.append(("collect", Path(task_path).parent.name))
            return super().collect(task_path, log_path, echo, poll)

        def update(self, argv, log_path, echo, poll=None):
            self.calls.append(("update", argv[argv.index("--out") + 1]))
            if self.fail_update:
                self.fail_update -= 1
                raise TL.StepError("the update failed on purpose")
            stats = super().update(argv, log_path, echo, poll)
            if stop is not None:
                Path(stop).touch()
            return stats

        def gate(self, argv, log_path, echo, poll=None):
            argv = list(argv)
            self.calls.append(("gate", argv[argv.index("--candidate") + 1]))
            self.gate_argv.append(argv)
            cand = argv[argv.index("--candidate") + 1].split("=", 1)[0]
            report = Path(argv[argv.index("--report") + 1])
            report.parent.mkdir(parents=True, exist_ok=True)
            passed = gate == "pass"
            cond = {"name": "beats_champion", "passed": passed, "n": 1000,
                    "observed": 0.58 if passed else 0.5, "bound": 0.549 if passed else 0.469,
                    "reference": 0.52, "missing": 0, "skipped": False, "reason": "",
                    "detail": {}, "incomplete": False}
            report.write_text(json.dumps({"gates": {cand: {
                "candidate": cand, "decision": gate,
                "turned_on": "none" if passed else "beats_champion",
                "conditions": {"beats_champion": cond}}}}), encoding="utf-8")
            return 0

    return Recording()


def state(paths):
    return json.loads(paths.state_json.read_text(encoding="utf-8"))


def log_records(paths):
    return [json.loads(line) for line in paths.log_jsonl.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def test_two_iterations_inline_then_resume(loop, capsys):
    torch = pytest.importorskip("torch")
    TL, RD = loop.TL, loop.tools.decide
    paths = TL.RunPaths(loop.runs, "r1")
    steps = recording(TL)
    assert TL.main(argv(loop, "--iterations", "1", "--gate-every", "0"), steps=steps) == TL.EXIT_OK
    assert steps.calls == [("collect", "it0000"), ("update", str(paths.ckpt(1)))]
    st = state(paths)
    assert (st["iteration"], st["pending"], st["gate_due"]) == (1, None, None)
    assert st["champion"] == {"id": "il:v2_hog", "name": "v2_hog", "path": str(loop.start),
                              "iteration": 0}
    doc = json.loads(paths.args_json.read_text(encoding="utf-8"))
    fl = doc["flags"]
    assert (doc["wait_keep"], doc["schema"], fl["delay"], fl["level"], fl["noise"]) == \
        (WAIT_KEEP, "rs_rows/v2", 24, 11, "none")
    assert fl["mix"] == [0.5, 0.35, 0.15] and fl["start"] == fl["ref"] == str(loop.start)
    assert fl["opponent_deck"] == ",".join(sorted(TR.OPP)) and fl["deck_stats"] is None
    assert fl["learner_deck"] == sorted(TR.HOG) and fl["presses"] is True
    fr = doc["frozen"]
    assert fr["engine_id"] == doc["engine_id"] == doc["engine_fingerprint"]["id"]
    assert fr["start_sha256"] == TL.file_digest(loop.start) and fr["noise_digest"] is None
    assert fr["start_name"] == "v2_hog" and fr["anchor_sha256"] == {}
    assert len(doc["launches"]) == 1
    # every checkpoint of the run folder plays with the start's wait bias (args.json "wait_keep")
    for ck in (paths.ckpt(0), paths.ckpt(1), paths.champion):
        assert RD.wait_keep_for(str(ck)) == WAIT_KEEP, ck
    blob0 = torch.load(paths.ckpt(0), map_location="cpu")
    blob1 = torch.load(paths.ckpt(1), map_location="cpu")
    assert blob0["rl"]["run"] == "r1" and blob0["rl"]["engine_id"] == doc["engine_id"]
    assert blob0["rl"]["schema"] == "rs_rows/v2" and "ppo" not in blob0
    assert blob1["rl"] == blob0["rl"] and blob1["meta"] == blob0["meta"]
    assert blob1["ppo"]["state"]["updates"] == 1
    assert TL.file_digest(paths.champion) == TL.file_digest(loop.start)
    (rec,) = log_records(paths)
    c = rec["collect"]
    assert (rec["kind"], rec["iteration"], rec["learner"], rec["out"]) == \
        ("iteration", 0, "it0000.pt", "it0001.pt")
    assert c["matches"] == 4 and c["rows"] > 0 and c["errors"] == 0
    assert rec["update"]["rows"] == c["rows"] and rec["update"]["updates"] == 1
    league = TL.LG.League.from_dict(st["league"])
    assert league.iteration == 1 and league.champion == "il:v2_hog"
    assert league.anchor_ids() == ("il:v2_hog",) and league.snapshot_ids() == ()
    assert rec["league"]["filed"] == len(c["league_results"]) == league.member("il:v2_hog").games
    assert json.loads(paths.league_json.read_text(encoding="utf-8")) == st["league"]
    assert json.loads(paths.sidecar(1).read_text(encoding="utf-8"))["record"]["iteration"] == 0
    assert paths.sidecar(0).is_file() and paths.adam(1).is_file()
    assert not list(paths.data(0).glob("*.npz"))                   # deleted after the update
    assert (paths.data(0) / "summary.json").is_file()
    out = capsys.readouterr().out
    assert "it 0000 -> it0001.pt: 4 matches" in out and "stopped: 1 iterations done" in out

    # resume: only iteration 1 runs, its update starts from iteration 0's Adam moments
    steps = recording(TL)
    assert TL.main(argv(loop, "--resume", "--iterations", "2", "--gate-every", "0"),
                   steps=steps) == TL.EXIT_OK
    assert steps.calls == [("collect", "it0001"), ("update", str(paths.ckpt(2)))]
    assert state(paths)["iteration"] == 2
    assert [r["iteration"] for r in log_records(paths)] == [0, 1]
    assert paths.adam(1).is_file() and paths.adam(2).is_file()
    assert len(json.loads(paths.args_json.read_text(encoding="utf-8"))["launches"]) == 2
    assert torch.load(paths.ckpt(2), map_location="cpu")["ppo"]["state"]["updates"] == 2
    task = json.loads((paths.data(1) / "collect.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(task)) == task and task["workers"] == 0
    assert task["settings"]["wait_keep"] == WAIT_KEEP and len(task["plans"]) == 4
    assert all(p["spec"]["agents"][p["spec"]["seat"]]["ckpt"] == str(paths.ckpt(1))
               for p in task["plans"])

    # the collection subprocess's entry point, in this process: the same task again
    capsys.readouterr()
    assert TL.main(["collect-task", "--task", str(paths.data(1) / "collect.json")]) == TL.EXIT_OK
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["matches"] == 4 and "results" not in last and last["workers"]


def test_a_failed_update_resumes_at_the_update(loop, capsys):
    TL = loop.TL
    paths = TL.RunPaths(loop.runs, "r1")
    steps = recording(TL, fail_update=1)
    assert TL.main(argv(loop, "--iterations", "1"), steps=steps) == TL.EXIT_FAILED
    assert "the update failed on purpose" in capsys.readouterr().out
    st = state(paths)
    assert st["iteration"] == 0 and st["pending"]["iteration"] == 0
    assert st["pending"]["update"] is None and st["pending"]["collect"]["matches"] == 4
    assert list(paths.data(0).glob("*.npz")) and not paths.ckpt(1).exists()
    steps = recording(TL)
    assert TL.main(argv(loop, "--resume", "--iterations", "1"), steps=steps) == TL.EXIT_OK
    assert steps.calls == [("update", str(paths.ckpt(1)))]                 # no new collection
    assert "resuming at the update" in capsys.readouterr().out
    assert state(paths)["iteration"] == 1 and paths.ckpt(1).is_file()


@pytest.mark.parametrize("decision", ["fail", "pass"])
def test_the_gate_keeps_or_replaces_the_champion(loop, capsys, decision):
    TL = loop.TL
    paths = TL.RunPaths(loop.runs, "r1")
    extra = ("--anchor", f"other={loop.other}")
    steps = recording(TL, gate=decision)
    assert TL.main(argv(loop, "--iterations", "2", "--gate-every", "2", *extra),
                   steps=steps) == TL.EXIT_OK
    assert [c[0] for c in steps.calls] == ["collect", "update", "collect", "update", "gate"]
    cand_path = paths.ckpt(2)
    cand = f"it0002-{TL.file_digest(cand_path)[:6]}"
    (g,) = steps.gate_argv

    def value(flag):
        return g[g.index(flag) + 1]

    assert g[0] == "run" and value("--candidate") == f"{cand}={cand_path}"
    assert value("--champion") == f"v2_hog={loop.start}"
    assert [g[i + 1] for i, x in enumerate(g) if x == "--anchor"] == [f"other={loop.other}"]
    assert (value("--split"), value("--delay"), value("--noise"), value("--decide")) == \
        ("dev", "24", "none", "live")
    assert (value("--champion-games"), value("--anchor-games"), value("--max-ticks")) == \
        ("1000", "400", "400")
    assert (value("--log"), value("--report"), value("--engine-config")) == \
        (str(paths.gate_log), str(paths.gate_report(2)), str(paths.engine_json))
    assert value("--workers") == "6" and "--presses" in g
    assert json.loads(paths.engine_json.read_text(encoding="utf-8")) == \
        json.loads(paths.args_json.read_text(encoding="utf-8"))["engine"]
    st = state(paths)
    (rec,) = st["gates"]
    assert (rec["iteration"], rec["candidate"], rec["decision"], rec["error"]) == \
        (2, cand, decision, None)
    assert rec["conditions"]["beats_champion"]["n"] == 1000 and st["gate_due"] is None
    assert log_records(paths)[-1]["kind"] == "gate"
    league = TL.LG.League.from_dict(st["league"])
    out = capsys.readouterr().out
    if decision == "fail":
        assert st["champion"]["name"] == "v2_hog" and league.champion == "il:v2_hog"
        assert "snap:it0002" not in league
        assert TL.file_digest(paths.champion) == TL.file_digest(loop.start)
        assert "FAIL: the champion stays v2_hog" in out
        return
    assert st["champion"] == {"id": "snap:it0002", "name": cand, "path": str(cand_path),
                              "iteration": 2}
    assert league.champion == "snap:it0002" and league.champion_chain == ["il:v2_hog",
                                                                         "snap:it0002"]
    assert league.member("snap:it0002").active and league.member("snap:it0002").path == \
        str(cand_path)
    assert TL.file_digest(paths.champion) == TL.file_digest(cand_path)
    champ = json.loads(paths.champion_json.read_text(encoding="utf-8"))
    assert (champ["name"], champ["iteration"], champ["gate"]["decision"]) == (cand, 2, "pass")
    assert "PASS: the new champion" in out
    # a gate now on iteration 1: the start is an anchor again, the new champion is not
    steps = recording(TL, gate="fail")
    assert TL.main(argv(loop, "--resume", "--gate-now", "1", *extra), steps=steps) == TL.EXIT_OK
    assert [c[0] for c in steps.calls] == ["gate"]
    (g,) = steps.gate_argv
    assert g[g.index("--champion") + 1] == f"{cand}={cand_path}"
    assert [g[i + 1] for i, x in enumerate(g) if x == "--anchor"] == \
        [f"v2_hog={loop.start}", f"other={loop.other}"]
    assert state(paths)["champion"]["name"] == cand and len(state(paths)["gates"]) == 2
    # the champion against itself is not a gate
    assert TL.main(argv(loop, "--resume", "--gate-now", *extra),
                   steps=recording(TL)) == TL.EXIT_GATE
    assert state(paths)["gates"][-1]["error"] == "the candidate is the champion"
    capsys.readouterr()
    assert TL.main(["status", "--name", "r1", "--runs-dir", str(loop.runs)]) == TL.EXIT_OK
    shown = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert shown["iteration"] == 2 and shown["champion"]["name"] == cand
    assert len(shown["gates"]) == 3 and shown["pending"] is False


def test_stop_ends_the_run_at_the_next_step_and_blocks_a_launch(loop, capsys):
    TL = loop.TL
    paths = TL.RunPaths(loop.runs, "r1")
    steps = recording(TL, stop=paths.stop)
    assert TL.main(argv(loop, "--iterations", "3"), steps=steps) == TL.EXIT_OK
    assert [c[0] for c in steps.calls] == ["collect", "update"]    # the commit, then STOP
    assert state(paths)["iteration"] == 1
    assert f"stopped: STOP ({paths.stop})" in capsys.readouterr().out
    assert TL.main(argv(loop, "--resume"), steps=recording(TL)) == TL.EXIT_ARGS
    assert "refusing to start" in capsys.readouterr().err
    paths.stop.unlink()
    TL.REPO_STOP.parent.mkdir(parents=True, exist_ok=True)
    TL.REPO_STOP.touch()                                          # the live runner's STOP
    assert TL.main(argv(loop, "--resume"), steps=recording(TL)) == TL.EXIT_ARGS
    assert str(TL.REPO_STOP) in capsys.readouterr().err
    TL.REPO_STOP.unlink()
    steps = recording(TL)
    assert TL.main(argv(loop, "--resume", "--iterations", "2"), steps=steps) == TL.EXIT_OK
    assert steps.calls == [("collect", "it0001"), ("update", str(paths.ckpt(2)))]
    assert state(paths)["iteration"] == 2


def test_refusals(loop, capsys):
    TL = loop.TL
    assert TL.main(argv(loop, "--resume"), steps=recording(TL)) == TL.EXIT_ARGS
    assert "no run to resume" in capsys.readouterr().err
    assert TL.main(argv(loop, "--iterations", "1"), steps=recording(TL)) == TL.EXIT_OK
    capsys.readouterr()
    assert TL.main(argv(loop, "--iterations", "1"), steps=recording(TL)) == TL.EXIT_ARGS
    assert "already exists" in capsys.readouterr().err
    assert TL.main(argv(loop, "--resume", "--delay", "21"), steps=recording(TL)) == TL.EXIT_ARGS
    err = capsys.readouterr().err
    assert "--delay 21" in err and "the run's: 24" in err
    assert TL.main(argv(loop, "--resume", "--mix", "0.6,0.3,0.1"),
                   steps=recording(TL)) == TL.EXIT_ARGS
    assert "--mix" in capsys.readouterr().err
    assert TL.main(argv(loop, "--resume", "--card-table", "client15535"),
                   steps=recording(TL)) == TL.EXIT_FAILED          # another engine
    capsys.readouterr()
    TR.save_ckpt(loop.start, TR.meta_for(), seed=5)               # the start changed under it
    assert TL.main(argv(loop, "--resume"), steps=recording(TL)) == TL.EXIT_ARGS
    assert "start_sha256" in capsys.readouterr().err
    legacy = loop.tmp / "tokens" / "legacy" / "best.pt"
    TR.save_ckpt(legacy, TR.meta_for(schema="rs_rows/legacy"))
    assert TL.main(argv(loop, name="r2", start=legacy), steps=recording(TL)) == TL.EXIT_ARGS
    assert "v2 checkpoint" in capsys.readouterr().err
    assert TL.main(argv(loop, name="bad name"), steps=recording(TL)) == TL.EXIT_ARGS
    assert TL.main(argv(loop, "--rollback", "1", name="r3"), steps=recording(TL)) == TL.EXIT_ARGS
    assert TL.main(argv(loop, "--bench", "never", name="r3"), steps=recording(TL)) == TL.EXIT_ARGS
    assert TL.main(argv(loop, "--anchor", "v2_hog=" + str(loop.other), name="r3"),
                   steps=recording(TL)) == TL.EXIT_ARGS             # the start's name
    assert "name is taken" in capsys.readouterr().err
    assert not (loop.runs / "r2" / "state.json").exists()
    assert not (loop.runs / "r3" / "state.json").exists()


def test_rollback_returns_to_an_earlier_iteration(loop, capsys):
    TL = loop.TL
    paths = TL.RunPaths(loop.runs, "r1")
    assert TL.main(argv(loop, "--iterations", "2", "--gate-every", "0"),
                   steps=recording(TL)) == TL.EXIT_OK
    assert TL.main(argv(loop, "--resume", "--rollback", "1", "--iterations", "1"),
                   steps=recording(TL)) == TL.EXIT_OK
    st = state(paths)
    assert (st["iteration"], st["pending"]) == (1, None) and "record" not in st
    assert paths.ckpt(1).is_file() and paths.sidecar(1).is_file()
    assert not paths.ckpt(2).exists() and not paths.sidecar(2).exists()
    assert not paths.data(1).exists() and paths.data(0).is_dir()
    (archive,) = paths.run.glob("rollback-*")
    assert (archive / "it0002.pt").is_file() and (archive / "it0002.json").is_file()
    assert (archive / "it0001").is_dir() and (archive / "adam" / "it0002.pt").is_file()
    assert paths.adam(1).is_file()
    assert log_records(paths)[-1]["kind"] == "rollback" and "rollback to iteration 1" in \
        capsys.readouterr().out
    assert TL.main(argv(loop, "--resume", "--rollback", "5"), steps=recording(TL)) == TL.EXIT_ARGS
    steps = recording(TL)
    assert TL.main(argv(loop, "--resume", "--iterations", "2", "--gate-every", "0"),
                   steps=steps) == TL.EXIT_OK
    assert steps.calls == [("collect", "it0001"), ("update", str(paths.ckpt(2)))]
    assert state(paths)["iteration"] == 2 and paths.ckpt(2).is_file()


# ---------------------------------------------------------------------------
# the opponents' mix
# ---------------------------------------------------------------------------

def test_the_mix_draws_mirror_pool_and_anchors(loop):
    TL = loop.TL
    LG, R = TL.LG, loop.tools.rollout
    league = LG.League(LG.LeagueConfig(), anchors={"il:a": "a.pt", "il:b": "b.pt"})
    mix = TL.MixDraw(league)
    # no snapshot yet: the pool's share goes to the anchors
    assert mix.shares() == pytest.approx({"mirror": 0.5, "pool": 0.0, "anchors": 0.5})
    assert mix.weights() == pytest.approx({"learner": 0.5, "il:a": 0.25, "il:b": 0.25})
    plans = R.plan_battles(20, learner="me.pt", league=mix, deck_field=TR.opp_field(R), seed=1)
    assert {p.opponent for p in plans} <= {"learner", "il:a", "il:b"}
    for p in plans:
        spec = p.match_spec()
        if p.opponent == "learner":
            assert p.mirror
        else:
            assert spec.agents[1 - spec.seat].ckpt == {"il:a": "a.pt", "il:b": "b.pt"}[p.opponent]
    league.admit("s.pt", member_id="snap:it0005", iteration=5)
    assert mix.shares() == pytest.approx({"mirror": 0.5, "pool": 0.35, "anchors": 0.15})
    assert mix.weights() == pytest.approx({"learner": 0.5, "il:a": 0.075, "il:b": 0.075,
                                           "snap:it0005": 0.35})
    for _ in range(8):
        league.record("il:a", 1.0)                 # expected 0.9 for a, 0.5 for b: PFSP "hard"
    w = mix.weights()
    shape = np.array([0.1 ** 2, 0.5 ** 2])
    want = 0.8 * shape / shape.sum() + 0.1
    assert [w["il:a"], w["il:b"]] == pytest.approx(list(0.15 * want))
    assert sum(w.values()) == pytest.approx(1.0)
    a = mix.draw(np.random.default_rng(3), size=4000)
    assert a == mix.draw(np.random.default_rng(3), size=4000) and set(a) <= set(w)
    freq = Counter(a)
    assert freq["learner"] / 4000 == pytest.approx(0.5, abs=0.03)
    assert freq["snap:it0005"] / 4000 == pytest.approx(0.35, abs=0.03)
    assert isinstance(mix.draw(np.random.default_rng(3)), str)
    assert mix.member("il:a").path == "a.pt"
    assert mix.kinds() == {"learner": "mirror", "il:a": "anchors", "il:b": "anchors",
                           "snap:it0005": "pool"}
    league.evict(["snap:it0005"])
    assert mix.shares() == pytest.approx({"mirror": 0.5, "pool": 0.0, "anchors": 0.5})
    assert TL.MixDraw(league, (1, 0, 0)).weights() == {"learner": 1.0}
    assert TL.parse_mix("0.6,0.3,0.1") == [0.6, 0.3, 0.1]
    for bad in ("0,0,0", "1,2", "-1,1,1", "a,b,c"):
        with pytest.raises(ValueError):
            TL.parse_mix(bad)


# ---------------------------------------------------------------------------
# the bench
# ---------------------------------------------------------------------------

def test_the_bench_measures_and_recommends(loop, capsys):
    TL = loop.TL
    code = TL.main(argv(loop, "--bench", "1s", "--bench-workers", "0", name="b1"),
                   steps=TL.InlineSteps())
    assert code == TL.EXIT_OK
    out = capsys.readouterr().out
    assert "matches/hour" in out and "RAM per worker" in out and "recommended: --workers" in out
    result = json.loads(out.strip().splitlines()[-1])
    (row,) = result["bench"]
    assert (row["workers"], row["matches"], row["errors"]) == (0, 2, 0) and row["rows"] > 0
    assert result["ppo"]["rows"] == row["rows"] and result["recommended_workers"] >= 1
    assert result["rows_per_match"] == pytest.approx(row["rows"] / 2, abs=0.05)
    run = loop.runs / "b1"
    assert not (run / "state.json").exists() and not (run / "args.json").exists()
    (folder,) = run.glob("bench-*")
    assert (folder / "bench.json").is_file() and (folder / "bench.log").is_file()
    assert not list(folder.rglob("*.npz")) and not (folder / "ppo_bench.pt").exists()


def test_the_bench_recommendation(loop):
    TL, R = loop.TL, loop.tools.rollout
    rows = [{"workers": 1, "matches_per_hour": 600.0, "peak_rss_mb_max": 900.0},
            {"workers": 4, "matches_per_hour": 2200.0, "peak_rss_mb_max": 1000.0}]
    rec = TL.recommend(rows, battles=16, flush_rows=20_000, threads=1, budget_gb=8, cpus=28,
                       available=16_000)
    per = math.ceil(1000.0 * TL.BENCH_MARGIN)
    budget, free = 8192 // per, (16_000 - R.RESERVE_MB) // per
    assert (rec["per_worker_mb"], rec["cpu_cap"]) == (per, 26)
    assert rec["ram_cap"] == rec["recommended_workers"] == min(26, budget, free)
    assert rec["extrapolated"] and rec["measured_max"] == 4
    assert rec["matches_per_hour_est"] == pytest.approx(2200.0 * rec["recommended_workers"] / 4,
                                                        abs=0.1)
    assert rec["scaling"] == pytest.approx((2200 / 600) / 4)
    flat = [{"workers": 2, "matches_per_hour": 1000.0}, {"workers": 4, "matches_per_hour": 900.0}]
    rec = TL.recommend(flat, battles=16, flush_rows=20_000, threads=2, budget_gb=8, cpus=28,
                       available=16_000)
    assert rec["per_worker_mb"] == R.worker_mb(16, 20_000) and rec["cpu_cap"] == 13
    assert rec["recommended_workers"] == 2 and "flattens" in rec["basis"]
    assert rec["matches_per_hour_est"] == pytest.approx(1000.0) and not rec["extrapolated"]


# ---------------------------------------------------------------------------
# steps and small parts
# ---------------------------------------------------------------------------

def test_durations_json_lines_and_a_logged_subprocess(loop, tmp_path, capsys):
    TL = loop.TL
    assert TL.parse_duration("10m") == 600 and TL.parse_duration("600s") == 600
    assert TL.parse_duration("1.5h") == 5400 and TL.parse_duration("2") == 120
    for bad in ("0", "abc", "-1m", ""):
        with pytest.raises(ValueError):
            TL.parse_duration(bad)
    assert TL.last_json(["x", '{"a": 1}', "{bad", "y"]) == {"a": 1}
    assert TL.last_json(["[1]", "no"]) is None
    log = tmp_path / "logs" / "step.log"
    code, last = TL.run_logged(
        [sys.executable, "-c", "import json, sys; print('hello'); print('warn', file=sys.stderr);"
                               " print(json.dumps({'ok': 1}))"], log, TL.Echo())
    assert (code, last) == (0, {"ok": 1})
    text = log.read_text(encoding="utf-8")
    assert "hello" in text and "warn" in text and "== exit 0" in text
    assert "    hello" in capsys.readouterr().out
    assert TL.run_logged([sys.executable, "-c", "raise SystemExit(3)"], log, TL.Echo()) == (3, None)


def test_the_subprocess_steps_command_lines(loop, monkeypatch, tmp_path):
    TL = loop.TL
    seen = []

    def fake_run(cmd, log_path, echo, *, poll=None, env=None):
        seen.append([str(c) for c in cmd])
        return 0, {"ok": True}

    monkeypatch.setattr(TL, "run_logged", fake_run)
    steps = TL.SubprocessSteps("E/python", "T/python")
    log = tmp_path / "x.log"
    assert steps.collect(tmp_path / "t.json", log, TL.Echo()) == {"ok": True}
    assert steps.update(["--ckpt", "a"], log, TL.Echo()) == {"ok": True}
    assert steps.gate(["run"], log, TL.Echo()) == 0
    assert seen == [
        [str(Path("E/python")), str(Path(TL.__file__).resolve()), "collect-task", "--task",
         str(tmp_path / "t.json")],
        [str(Path("T/python")), str(TL.ROOT / "tools" / "rl" / "ppo.py"), "--ckpt", "a"],
        [str(Path("E/python")), str(TL.ROOT / "tools" / "rs_league_eval.py"), "run"]]
    monkeypatch.setattr(TL, "run_logged", lambda *a, **k: (2, None))
    with pytest.raises(TL.StepError, match="exit 2"):
        steps.collect(tmp_path / "t.json", log, TL.Echo())
    with pytest.raises(TL.StepError, match="PPO update failed"):
        steps.update([], log, TL.Echo())
    assert steps.gate([], log, TL.Echo()) == 2
    assert isinstance(TL.make_steps(types.SimpleNamespace(inline=True)), TL.InlineSteps)
