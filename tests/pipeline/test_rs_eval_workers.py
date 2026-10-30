"""tools/rs_eval.py --workers: capped by free memory and CPUs as tools/rs_league_eval.py caps its
own (rs_eval.capped_workers), with results that do not depend on the count.

The PC froze about 6 minutes into an 8-worker v2 rs_eval run on 2026-10-30 (docs/SPARRING_RUNBOOK.md,
"If the PC freezes"). rs_eval now lowers --workers the way the league does (worker_cap: the free
memory less its 3 GB reserve at 1.2 GB a worker, 1.8 GB with --search or --shadow as for a league
worker with a searching player, and the CPUs less one; at least 1), without the league's 6-worker
maximum, and prints the reason. What it must not change is a result: the tasks are
the held-out matches in corpus order, match k is seeded k (the battle, the agent; the reactive
opponent k + 7) and its noise by its replay tag, every match builds its own agents, and pool.map
returns the results in task order. So the count only changes which process plays which match; on
the PC the same run gave 155/300, -0.15, 38.3 plays at 8 workers and at 4.

No engine, no corpus, no torch: rs_eval is imported as tools/rs_eval.py imports itself, and main()
is read, not run.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
RS_EVAL = TOOLS / "rs_eval.py"


@pytest.fixture
def ev(monkeypatch):
    """tools/rs_eval.py as the tools import it (royaleimitate kept out, as test_noise_hooks does)."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    return importlib.import_module("rs_eval")


def _main() -> ast.FunctionDef:
    tree = ast.parse(RS_EVAL.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def test_capped_workers_is_the_league_cap_without_its_maximum(ev, monkeypatch):
    LE = importlib.import_module("rs_league_eval")
    mb, reserve = LE.WORKER_MB, LE.RESERVE_MB
    assert (mb, reserve) == (1200, 3072)
    # Room for all: the count asked for, also above the league's MAX_WORKERS (6).
    assert ev.capped_workers(8, available=64000, cpus=16) == (8, "8 worker(s)")
    assert LE.worker_cap(8, available=64000, cpus=16)[0] == LE.MAX_WORKERS == 6
    assert ev.capped_workers(6, available=64000, cpus=16) == (6, "6 worker(s)")
    # Memory: (8000 - 3072) // 1200 = 4, the league's own reason word for word.
    got = ev.capped_workers(8, available=8000, cpus=16)
    assert got == (4, "4 worker(s), not 8: capped by memory 4 (8000 MB free, 1200 MB each, "
                      "3072 MB kept)")
    assert got == LE.worker_cap(8, available=8000, cpus=16)
    # About what a 16 GB PC has free with MuMu and the browser closed: 8 asked, 6 played.
    assert ev.capped_workers(8, available=10500, cpus=16)[0] == (10500 - reserve) // mb == 6
    # The CPUs less one.
    assert ev.capped_workers(8, available=64000, cpus=4) == (
        3, "3 worker(s), not 8: capped by cpus 3 (64000 MB free, 1200 MB each, 3072 MB kept)")
    # At least 1, also for 0 (ProcessPoolExecutor refuses 0) and for no memory to spare.
    assert ev.capped_workers(0, available=64000, cpus=16) == (1, "1 worker(s)")
    assert ev.capped_workers(4, available=1000, cpus=16)[0] == 1
    # This machine's own memory and CPUs: never more than asked, never fewer than 1.
    workers, why = ev.capped_workers(2)
    assert 1 <= workers <= 2 and why.startswith(f"{workers} worker(s)")
    # Free memory unreadable: the league's assumption (8192 MB): 4 workers.
    monkeypatch.setattr(LE, "available_mb", lambda: None)
    assert ev.capped_workers(8, cpus=16)[0] == (LE.ASSUMED_AVAILABLE_MB - reserve) // mb == 4


def test_a_search_or_shadow_worker_counts_the_leagues_search_memory(ev):
    # --search (its battle copies, --rebuild's rebuilt battle) and --shadow (its re-drive) keep a
    # second battle in each worker: the league counts a worker with a searching player at
    # SEARCH_WORKER_MB (rs_league_eval default_worker_mb), and so does rs_eval.
    LE = importlib.import_module("rs_league_eval")
    assert LE.SEARCH_WORKER_MB == 1800
    # (8000 - 3072) // 1800 = 2 where a plain run gets 4.
    assert ev.capped_workers(8, available=8000, cpus=16, search=True) == (
        2, "2 worker(s), not 8: capped by memory 2 (8000 MB free, 1800 MB each, 3072 MB kept)")
    assert ev.capped_workers(8, available=8000, cpus=16)[0] == 4
    assert ev.capped_workers(8, available=8000, cpus=16, search=True) == LE.worker_cap(
        8, available=8000, cpus=16, per_worker_mb=LE.SEARCH_WORKER_MB)
    # About a 16 GB PC with MuMu and the browser closed: 4 searching workers, 6 plain ones.
    assert ev.capped_workers(8, available=10500, cpus=16, search=True)[0] == 4
    assert ev.capped_workers(4, available=10500, cpus=16, search=True) == (4, "4 worker(s)")
    assert ev.capped_workers(8, available=64000, cpus=16, search=True) == (8, "8 worker(s)")
    # main() asks for it in exactly those modes.
    src = ast.unparse(_main())
    assert ("second_battle = mode_ in ('search', 'search_rebuild') or bool(args.shadow)"
            in src)
    assert "mode_ = 'pro' if args.pro else 'search_rebuild' if args.search and args.rebuild " \
           "else 'search' if args.search else 'agent'" in src


def test_capped_workers_imports_nothing_heavy(ev):
    # rs_league_eval at import: numpy and modules rs_eval imports already (no engine, no torch).
    LE = importlib.import_module("rs_league_eval")
    tree = ast.parse(Path(LE.__file__).read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not top & {"torch", "royalesim", "rs_eval", "rs_match", "rs_rows"}, top
    for name in ("perception_noise", "rs_engine", "rs_memory", "numpy"):
        assert name in top and name in sys.modules, name
    # Imported inside the function, so a worker process (which imports rs_eval) does not.
    fn = next(n for n in ast.parse(RS_EVAL.read_text(encoding="utf-8")).body
              if isinstance(n, ast.FunctionDef) and n.name == "capped_workers")
    assert any(isinstance(n, ast.Import) and n.names[0].name == "rs_league_eval"
               for n in ast.walk(fn))


def test_main_caps_the_pool_and_prints_why():
    main = _main()
    src = ast.unparse(main)
    assert "workers, why = capped_workers(args.workers, search=second_battle)" in src
    assert "print(f'workers: {why}', flush=True)" in src
    pools = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "ProcessPoolExecutor"]
    assert len(pools) == 1
    assert isinstance(pools[0].args[0], ast.Name) and pools[0].args[0].id == "workers"
    # args.workers is read once, by the cap; the cap comes before the pool.
    reads = [n for n in ast.walk(main) if isinstance(n, ast.Attribute) and n.attr == "workers"
             and isinstance(n.value, ast.Name) and n.value.id == "args"]
    assert len(reads) == 1
    assert src.index("capped_workers(args.workers, ") < src.index("ProcessPoolExecutor(")
    # --workers keeps its default.
    text = RS_EVAL.read_text(encoding="utf-8")
    assert 'ap.add_argument("--workers", type=int, default=6,' in text


def test_match_seeds_and_order_do_not_depend_on_the_worker_count():
    main = _main()
    maps = [n for n in ast.walk(main) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "map"
            and getattr(n.func.value, "id", None) == "pool"]
    assert len(maps) == 1, "one pool.map (results in task order)"
    call = maps[0]
    assert getattr(call.args[0], "id", None) == "play"
    comp = call.args[1]
    assert isinstance(comp, ast.ListComp) and len(comp.generators) == 1
    gen = comp.generators[0]
    assert [getattr(e, "id", None) for e in getattr(gen.target, "elts", ())] == ["k", "t"]
    assert ast.unparse(gen.iter) == "enumerate(picked)"
    # The task tuple: (payload, ckpt, mode, seed = k, ...) [+ extra].
    task = comp.elt.left if isinstance(comp.elt, ast.BinOp) else comp.elt
    assert isinstance(task, ast.Tuple)
    assert [ast.unparse(e) for e in task.elts[:4]] == ["t", "args.ckpt", "mode_", "k"]
    src = ast.unparse(main)
    for unordered in ("as_completed", "imap_unordered", ".submit("):
        assert unordered not in src, unordered
    # No task field depends on the worker count.
    assert "workers" not in ast.unparse(comp)
    # play(): the match's seed drives the battle and both agents; its noise is seeded by its tag.
    text = RS_EVAL.read_text(encoding="utf-8")
    play = text[text.index("\ndef play(args):"):text.index("\ndef main()")]
    for needle in ("payload_text, ckpt, mode_, seed = args[:4]", "b.reset(seed=seed,",
                   "AGENT_CLASS(ckpt, seed=seed))", "RSAgent(opp_ckpt, seed=seed + 7)",
                   "PN.match_seed(key, noise_seed)", 'f"{tag}|opponent"'):
        assert needle in play, needle
    # Each RSAgent sets one torch thread itself, whatever the process count.
    assert "torch.set_num_threads(1)" in text
