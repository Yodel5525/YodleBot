"""tools/rs_selfplay.py, the deferred fixes (LANE B step 2): the engine config reaches every
collect worker (pool initializer rs_engine.set_active and the task's engine slot) and is stamped
into the rollouts (.npz), the checkpoints train writes and the run's args.json; loop hands the
engine flags to collect and to the evaluation and keeps one engine per run; the sampling agent
decides on rows seen with its checkpoint's visibility; only legacy-schema checkpoints are played
and trained; --eval-opponent stays ep01.pt. The RL update itself is unchanged.

Everything runs on conftest's fake royalesim, a fake process pool and fake subprocesses: no
engine, no corpus, no GPU. The sampling-agent and train tests need torch (skipped without it).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    FIELD,
    SUBTILE_PER_MILLITILE,
    FakeBattle,
    entity_row,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = SUBTILE_PER_MILLITILE
TABLE = "client160402017_20261019"          # what --card-table 16.402.22 names


@pytest.fixture
def sp(monkeypatch):
    """rs_selfplay and rs_engine as the tools import them, with no engine config from the PC."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    eng = importlib.import_module("rs_engine")
    for var in (eng.CONFIG_ENV, *eng.ENV.values()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    return types.SimpleNamespace(mod=importlib.import_module("rs_selfplay"), engine=eng)


@pytest.fixture
def machine_env(monkeypatch, tmp_path):
    """Interpreters from HASTYCR_PY_ENGINE / HASTYCR_PY_TRAIN only (no machine.json)."""
    from hastycr import machine
    for var in (*machine.ENV.values(), machine.CONFIG_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(machine.CONFIG_ENV, str(tmp_path / "machine.json"))
    engine_py, train_py = tmp_path / "eng" / "python.exe", tmp_path / "gpu" / "python.exe"
    monkeypatch.setenv("HASTYCR_PY_ENGINE", str(engine_py))
    monkeypatch.setenv("HASTYCR_PY_TRAIN", str(train_py))
    machine.get.cache_clear()
    yield str(engine_py), str(train_py)
    machine.get.cache_clear()


def engine_file(tmp_path) -> Path:
    path = tmp_path / "engine.json"
    path.write_text(json.dumps({"card_table": "client15535", "level": 11, "tower_level": 11,
                                "delay": [0, 0]}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# flags, command lines, schema
# ---------------------------------------------------------------------------

def test_engine_flags_command_lines_and_the_ep01_default(sp, monkeypatch, tmp_path, capsys):
    mod = sp.mod
    assert mod.engine_argv(argparse.Namespace()) == []
    assert mod.engine_argv(argparse.Namespace(engine_config=None, engine_card_table=None)) == []
    with pytest.raises(SystemExit) as exc:
        mod.engine_config(argparse.Namespace(engine_config=str(tmp_path / "nope.json")))
    assert exc.value.code == 2 and "does not exist" in capsys.readouterr().err

    got = {}

    def keep(name):
        def run(args):
            got[name] = args
            return 0
        return run

    monkeypatch.setattr(mod, "collect", keep("collect"))
    monkeypatch.setattr(mod, "loop", keep("loop"))
    monkeypatch.setattr(sys, "argv", ["rs_selfplay.py", "collect", "--ckpt", "a.pt", "--opponent",
                                      "b.pt", "--out", "c.npz", "--card-table", "16.402.22",
                                      "--engine-config", "legacy"])
    assert mod.main() == 0
    monkeypatch.setattr(sys, "argv", ["rs_selfplay.py", "loop"])
    assert mod.main() == 0
    c, lp = got["collect"], got["loop"]
    assert (c.engine_card_table, c.engine_config) == ("16.402.22", "legacy")
    assert mod.engine_argv(c) == ["--engine-config", "legacy", "--card-table", "16.402.22"]
    assert (lp.engine_card_table, lp.engine_config) == (None, None)
    assert lp.eval_opponent == str(ROOT / "runs" / "tokens" / "rs_bc" / "ep01.pt")


def test_only_legacy_schema_checkpoints_are_played_and_trained(sp):
    for meta in (None, {}, {"schema": None}, {"schema": "rs_rows/legacy"}, {"schema": "legacy"}):
        sp.mod.check_schema(meta)
    with pytest.raises(ValueError, match="it003.pt: a checkpoint of row schema 'rs_rows/v2'"):
        sp.mod.check_schema({"schema": "rs_rows/v2"}, "it003.pt")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def test_collect_hands_the_engine_to_every_worker_and_records_it(sp, fake_royalesim, monkeypatch,
                                                                 tmp_path, capsys):
    mod, eng = sp.mod, sp.engine
    picks = []

    def pick(n, seed, split="train", engine=None):
        picks.append((n, seed, split, engine))
        return ["m0", "m1"]

    class Pool:
        made: list = []

        def __init__(self, workers, initializer=None, initargs=()):
            self.workers, self.initializer, self.initargs = workers, initializer, initargs
            Pool.made.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, tasks):
            self.fn, self.tasks = fn, list(tasks)
            return iter([("win", [1, 0], 1.0, [{"gate_a": 1, "unit_ids": np.array([5, 0], np.int16)},
                                               {"gate_a": 0, "unit_ids": np.array([0, 7], np.int16)}]),
                         None])                 # a match with no records (skipped)

    monkeypatch.setattr(mod, "pick_matches", pick)
    monkeypatch.setattr(mod, "ProcessPoolExecutor", Pool)
    out = tmp_path / "it000.npz"
    mod.collect(argparse.Namespace(ckpt="me.pt", opponent="opp.pt", matches=2, workers=3, seed=5,
                                   out=str(out), engine_config=str(engine_file(tmp_path)),
                                   engine_card_table="16.402.22"))
    # the file says client15535; --card-table wins
    cfg = eng.EngineConfig(card_table=TABLE, level=11, tower_level=11, delay=(0, 0))
    cfg_json = {"card_table": TABLE, "level": 11, "tower_level": 11, "delay": [0, 0],
                "calibration_overrides": {}}
    assert cfg.to_json() == cfg_json
    assert eng.active() == cfg and picks == [(2, 5, "train", cfg)]
    (pool,) = Pool.made
    assert pool.workers == 3 and pool.fn is mod._collect_one
    assert pool.initializer is eng.set_active and pool.initargs == (cfg_json,)
    assert pool.tasks == [("m0", "me.pt", "agent", 5000, "opp.pt", 21, -21, False, False, False,
                           cfg_json),
                          ("m1", "me.pt", "agent", 5001, "opp.pt", 21, -21, False, False, False,
                           cfg_json)]
    # in a fresh worker the initializer makes it the process's config
    monkeypatch.setattr(eng, "_ACTIVE", None)
    pool.initializer(*pool.initargs)
    assert eng.active() == cfg
    # the one Battle collect built: the fingerprint's, under cfg, undelayed
    (battle,) = fake_royalesim.battles
    assert battle.init_kwargs == {"card_names": None, "slot_of_k": [[0, 1, 2], [0, 1, 2]],
                                  "calibration_overrides": {"cards.CARD_TABLE": json.dumps(TABLE)},
                                  "level": 11, "tower_level": 11}
    assert battle.names() == ["set_command_delay_ticks"] and battle.delay == (0, 0)
    # the decisions as before, plus the engine record
    with np.load(out) as d:
        assert sorted(d.files) == ["engine_json", "gate_a", "ret", "unit_ids"]
        assert d["gate_a"].tolist() == [1, 0] and d["ret"].tolist() == [1.0, 1.0]
        assert d["unit_ids"].tolist() == [[5, 0], [0, 7]]
        rec = mod.read_engine_record(d)
    assert mod.read_engine_record(out) == rec == mod.read_engine_record(str(out))
    assert mod.read_engine_record(tmp_path / "missing.npz") is None
    fp = rec["engine_fingerprint"]
    assert rec["engine"] == {**cfg_json, "delay": [21, 21]}
    assert rec["engine_id"] == fp["id"] == eng.fingerprint_id(fp)
    assert (fp["version"], fp["unit_types"], fp["catalogue"]) == (
        "0.0.0+fake", len(DEFAULT_UNIT_TYPES), len(DEFAULT_CATALOGUE))
    assert fp["calibration_overrides"] == {"cards.CARD_TABLE": json.dumps(TABLE)}
    assert fp == eng.fingerprint(eng.make_battle(cfg.undelayed(), fake_royalesim),
                                 fake_royalesim, cfg)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (summary["matches"], summary["decisions"], summary["win_rate"],
            summary["engine_id"]) == (1, 2, 1.0, fp["id"])


def test_collect_one_plays_the_task_as_given(sp, monkeypatch):
    ev = importlib.import_module("rs_eval")
    monkeypatch.setattr(ev, "AGENT_CLASS", ev.AGENT_CLASS)       # restored after the test
    outs = [{"result": "draw", "crowns": [1, 1], "records": [{"gate_a": 0}]},
            {"tag": "t", "mode": "agent", "skipped": "orient", "detail": ""}, None]
    seen = []

    def play(task):
        seen.append(task)
        return outs[len(seen) - 1]

    monkeypatch.setattr(ev, "play", play)
    task = ("m0", "me.pt", "agent", 0, "opp.pt", 21, -21, False, False, False,
            {"card_table": TABLE})
    assert sp.mod._collect_one(task) == ("draw", [1, 1], 0.5, [{"gate_a": 0}])
    assert sp.mod._collect_one(task) is None            # skipped: no records
    assert sp.mod._collect_one(task) is None            # no deal order fits
    assert seen == [task] * 3
    assert ev.AGENT_CLASS.__name__ == "SPAgent" and issubclass(ev.AGENT_CLASS, ev.RSAgent)


# ---------------------------------------------------------------------------
# the sampling agent (torch)
# ---------------------------------------------------------------------------

class StubPolicy:
    """Gate: slot 0 for certain (the wait and the other slots at -1e4); tile 100 the best of the
    18 x 32; value logit 0. Keeps the card ids it was asked about."""

    def __init__(self, torch):
        self.torch, self.card_ids = torch, []

    def __call__(self, b, card_ids=None):
        t = self.torch
        self.card_ids.append(card_ids.tolist())
        tile = t.zeros(1, 576)
        tile[0, 100] = 10.0
        return t.tensor([[-1e4, 0.0, -1e4, -1e4, -1e4]]), tile, t.zeros(1, 2)


#: Tile 100 is column 10, row 5 of blue's own frame: world (10500, 32000 - 5500) millitiles.
LEGAL = (10_500 * SUB, 26_500 * SUB)


def board(tick=200):
    """Blue's Hog Rider (unit type 3, token 5) and an invisible red Knight (8, token 10)."""
    own_unit = entity_row(uid=50, team=0, card_id=2, x=9000 * SUB, y=10_000 * SUB, hp=500,
                          unit_type=3)
    hidden = entity_row(uid=51, team=1, card_id=8, x=9000 * SUB, y=20_000 * SUB, hp=600,
                        unit_type=8, status_flags=2)            # STATUS_INVISIBLE
    return state_dict(tick=tick, entities=[own_unit, hidden])


def _agent(sp, monkeypatch, tmp_path, meta, policy):
    """The sampling agent on a checkpoint whose load_policy gives `policy` and `meta`."""
    torch = policy.torch
    rc = importlib.import_module("rs_checkpoint")
    monkeypatch.setattr(rc, "load_policy",
                        lambda ckpt, engine=None, allow_drift=None: (policy, dict(meta), []))
    threads = torch.get_num_threads()
    try:
        return sp.mod._sampling_agent()(str(tmp_path / "best.pt"), seed=0)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("meta, override, sees", [
    ({}, None, "legacy"),                                   # a checkpoint from before E12
    ({"visibility": "public"}, None, "public"),
    ({"visibility": "public"}, "legacy", "legacy"),        # rs_eval.play's --visibility over it
    ({"visibility": "legacy"}, "public", "public"),
])
def test_sampling_agent_sees_with_its_checkpoints_visibility(sp, monkeypatch, tmp_path, meta,
                                                            override, sees):
    torch = pytest.importorskip("torch")
    rr = importlib.import_module("rs_rows")
    for name, value in {"F": dict(FIELD), "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "VISIBILITY": rr.VIS_LEGACY}.items():
        monkeypatch.setattr(rr, name, value)
    policy = StubPolicy(torch)
    agent = _agent(sp, monkeypatch, tmp_path, meta, policy)
    assert agent.visibility == rr.visibility_of(meta)
    if override is not None:
        agent.visibility = override
    state = board()
    mem = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}
    battle = FakeBattle(deploy=lambda team, slot, x, y: 0 if (x, y) == LEGAL else 7)
    assert agent.act(battle, state, 0, mem, 200, SUB, None) == (0, 0, *LEGAL)
    assert policy.card_ids == [[2], [2]]                    # the Cannon (card 0, token 2)
    (rec,) = agent.records
    want = rr.obs_row(state, 0, SUB, mem, 200, live=False, visibility=sees)
    for key in sp.mod.KEYS:
        assert np.array_equal(rec[key], want[key]), key
    assert {int(u) for u in rec["unit_ids"] if u} == ({5} if sees == "public" else {5, 10})
    # the only play has probability 1, and the only legal tile too
    assert (rec["gate_a"], rec["logq_gate"], rec["tile_a"], rec["logq_tile"]) == (1, 0.0, 0, 0.0)
    assert rec["tile_cands"].tolist() == [100] + [-1] * 15
    assert (rec["next"], rec["v"], rec["tick"]) == (6, 0.5, 200)
    assert rec["mask"].tolist() == [True] * 5


def test_sampling_agent_refuses_another_row_schema(sp, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="row schema 'rs_rows/v2'"):
        _agent(sp, monkeypatch, tmp_path, {"schema": "rs_rows/v2"}, StubPolicy(torch))


# ---------------------------------------------------------------------------
# train (torch): the stamp
# ---------------------------------------------------------------------------

def rollout_columns() -> dict:
    """Four decisions as collect writes them: two waits, a play of slot 0, one of slot 1."""
    n = 4
    unit_ids = np.zeros((n, 48), np.int16)
    unit_ids[:, 0] = 3
    cands = np.full((n, 16), -1, np.int16)
    cands[1, :2] = (7, 50)
    cands[3, :2] = (7, 50)
    return {"unit_ids": unit_ids, "unit_feats": np.zeros((n, 48, 7), np.float16),
            "towers": np.ones((n, 6), np.float32), "scalars": np.zeros((n, 4), np.float32),
            "hand": np.tile(np.array([2, 3, 4, 5], np.int16), (n, 1)),
            "opp_seen": np.zeros((n, 8), np.int16), "opp_last": np.zeros((n, 4), np.int16),
            "opp_last_age": np.ones((n, 4), np.float16), "next": np.full(n, 6),
            "mask": np.ones((n, 5), bool), "gate_a": np.array([0, 1, 0, 2]),
            "logq_gate": np.zeros(n), "v": np.full(n, 0.5), "tile_a": np.array([-1, 0, -1, 1]),
            "logq_tile": np.zeros(n), "tile_cands": cands, "tick": np.array([100, 110, 120, 130]),
            "ret": np.array([1.0, 1.0, 0.0, 0.0])}


def test_train_stamps_the_rollout_engine_into_the_checkpoint(sp, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    from sim.model_tokens import TokenPolicy
    mod = sp.mod
    meta = {"n_cards": 20, "n_units": 20, "d": 16, "layers": 1, "heads": 2,
            "schema": "rs_rows/legacy"}
    net = TokenPolicy(20, 20, d=16, layers=1, heads=2)
    ckpt = tmp_path / "it000.pt"
    torch.save({"model": net.state_dict(), "meta": meta, "selfplay": {"engine_id": "stale"}}, ckpt)
    record = {"engine": {"card_table": TABLE, "level": 11, "tower_level": 11, "delay": [21, 21],
                         "calibration_overrides": {}},
              "engine_fingerprint": {"id": "0123456789abcdef", "version": "0.1.42",
                                     "provenance": None, "unit_types": 237},
              "engine_id": "0123456789abcdef"}
    for stamped in (True, False):
        cols = rollout_columns()
        if stamped:
            cols[mod.ENGINE_KEY] = np.array(json.dumps(record))
        data = tmp_path / f"roll{int(stamped)}.npz"
        np.savez_compressed(data, **cols)
        out = tmp_path / f"it001_{int(stamped)}.pt"
        mod.train(argparse.Namespace(ckpt=str(ckpt), ref=str(ckpt), data=str(data), out=str(out),
                                     lr=0.0, kl=0.05, clip=0.2, epochs=1, batch=4,
                                     wait_keep="0.25"))
        blob = torch.load(out, map_location="cpu", weights_only=True)
        assert blob["meta"] == meta and set(blob["model"]) == set(net.state_dict())
        summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert summary["rows"] == 4
        if stamped:
            assert blob["selfplay"] == record and summary["engine_id"] == "0123456789abcdef"
        else:                       # an .npz from before the record: no stamp, not the stale one
            assert "selfplay" not in blob and summary["engine_id"] is None


# ---------------------------------------------------------------------------
# loop
# ---------------------------------------------------------------------------

def test_loop_hands_the_engine_on_and_keeps_one_per_run(sp, machine_env, monkeypatch, tmp_path,
                                                        capsys):
    mod = sp.mod
    rg, bb = machine_env
    monkeypatch.setattr(mod, "RUNS", tmp_path / "runs")
    init = tmp_path / "imit" / "best.pt"
    init.parent.mkdir()
    init.write_bytes(b"imitation")
    (init.parent / "args.json").write_text(json.dumps({"wait_keep": "0.25"}), encoding="utf-8")
    efile = str(engine_file(tmp_path))
    want = {"card_table": "client15535", "level": 11, "tower_level": 11, "delay": [21, 21],
            "calibration_overrides": {}}
    reported = {"engine": want, "engine_fingerprint": {"id": "f00d", "version": "fake"},
                "engine_id": "f00d"}
    calls = []

    def fake_run(argv, **kw):
        argv = [str(a) for a in argv]
        calls.append(argv)
        if argv[1].endswith("rs_eval.py"):
            return types.SimpleNamespace(returncode=0, stdout="x: 3/10 wins (30.0%)\n", stderr="")
        out = argv[argv.index("--out") + 1]
        if argv[2] == "collect":
            np.savez_compressed(out, **{mod.ENGINE_KEY: np.array(json.dumps(reported))})
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"matches": 2, "engine_id": reported["engine_id"]})
                + "\n", stderr="")
        Path(out).write_bytes(b"ppo")
        return types.SimpleNamespace(returncode=0, stdout='{"rows": 4}\n', stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    args = argparse.Namespace(name="t", init=str(init), ref="", opponent="opp.pt", iters=2,
                              matches=2, workers=1, lr=1e-5, kl=0.05, eval_every=2,
                              eval_matches=10, eval_opponent="ep01.pt", engine_config=efile,
                              engine_card_table=None)
    assert mod.loop(args) == 0
    run = tmp_path / "runs" / "t"
    assert [c[0] for c in calls] == [rg, bb, rg, bb, rg]
    assert ["eval" if c[1].endswith("rs_eval.py") else c[2] for c in calls] == [
        "collect", "train", "collect", "train", "eval"]
    assert calls[0][2:] == ["collect", "--ckpt", str(run / "it000.pt"), "--opponent", "opp.pt",
                            "--matches", "2", "--workers", "1", "--seed", "1",
                            "--out", str(run / "it000.npz"), "--engine-config", efile]
    assert "--engine-config" not in calls[1] and "--card-table" not in calls[1]
    assert calls[4][2:] == ["--ckpt", str(run / "it002.pt"), "--opponent-ckpt", "ep01.pt",
                            "--matches", "10", "--workers", "1", "--delay", "21", "--lead", "-21",
                            "--engine-config", efile]
    saved = json.loads((run / "args.json").read_text(encoding="utf-8"))
    assert {k: saved[k] for k in mod.STAMP_FIELDS} == reported
    assert (saved["wait_keep"], saved["engine_config"], saved["iters"]) == ("0.25", efile, 2)
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in log] == [0, 1] and log[0]["collect_engine_id"] == "f00d"
    assert log[1]["eval"] == "x: 3/10 wins (30.0%)"
    assert sorted(p.name for p in run.glob("it*")) == ["it000.pt", "it001.pt", "it002.pt"]
    capsys.readouterr()

    # continued under another card table: refused before anything runs
    calls.clear()
    other = argparse.Namespace(**{**vars(args), "iters": 3, "engine_card_table": "16.402.22"})
    assert mod.loop(other) == 1 and calls == []
    assert "start a new --name" in capsys.readouterr().err
    assert json.loads((run / "args.json").read_text(encoding="utf-8")) == saved

    # continued as before, but the collect ran on another engine: refused before the update
    reported = {"engine": want, "engine_fingerprint": {"id": "beef"}, "engine_id": "beef"}
    again = argparse.Namespace(**{**vars(args), "iters": 3})
    assert mod.loop(again) == 1
    assert [c[2] for c in calls] == ["collect"]
    assert calls[0][calls[0].index("--seed") + 1] == "3"          # resumed at it002
    assert "start a new --name" in capsys.readouterr().err
    kept = json.loads((run / "args.json").read_text(encoding="utf-8"))
    assert {k: kept[k] for k in mod.STAMP_FIELDS} == {k: saved[k] for k in mod.STAMP_FIELDS}

    # a new run whose first collect resolved another config than the launch: refused, not stamped
    calls.clear()
    reported = {"engine": {**want, "level": 12}, "engine_fingerprint": {"id": "cafe"}, "engine_id": "cafe"}
    assert mod.loop(argparse.Namespace(**{**vars(args), "name": "u"})) == 1
    assert [c[2] for c in calls] == ["collect"]
    assert "start a new --name" in capsys.readouterr().err
    fresh = json.loads((tmp_path / "runs" / "u" / "args.json").read_text(encoding="utf-8"))
    assert fresh["engine"] == want and "engine_id" not in fresh
