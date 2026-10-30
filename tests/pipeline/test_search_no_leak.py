"""S08 leak test: a search seat sees only its public view (docs/LIVE_SEARCH_PLAN.md 7.4, S08).

In the simulator run_match hands every seat the TRUE battle and an observed state that still holds
what no screen shows (the opponent's hand, elixir, next card and pending plays, the hidden entity
columns, invisible units, projectiles). A search seat must decide on the public view only. The
plan's two checks, through rs_search.SearchSeat's own act() (a stub of RSAgent's attributes, no
checkpoint and no torch; tools/rs_rebuild.py's real Rebuilder on test_rs_rebuild's CycleBattle;
test_rs_search's FakePolicy):
  * two observed states that differ only in privileged fields (test_rs_rebuild.push_state and
    its noised twin) give the same public view, the same RootSpec (its key), the same rows for
    the policy, the same search record and the same decision;
  * a proxy of the match battle records every attribute the seat reaches: a public preset
    (s_root, s_idle) reaches check_deploy only (d0's legality and the candidates' tap-time
    checks through the same object); only the privileged s_true reaches more (save, load,
    step, state_json: its true-copy trials), it says so in every record, and it leaves the
    battle as it found it.
Needs numpy and tools/public_state (an R1 spec sanitises with it); skipped without them.
"""
from __future__ import annotations

import copy
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import FIELD, FakeBattle

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.pipeline.test_rs_search import FakePolicy, ScriptRng  # noqa: E402

#: The push of test_rs_rebuild (our Hog deploying, an enemy Mini P.E.K.K.A deploying, our Log in
#: flight): [wait, Cannon, Fireball, Skeletons, Log] (test_rs_search's integration test).
GATE = [0.0, 3.0, -1.0, 0.5, 1.0]
BEST = {0: (404, 405, 440, 406), 2: (477, 478), 3: (200, 201)}


class Proxy:
    """The match battle as the seat gets it: every attribute it reaches is recorded."""

    def __init__(self, battle):
        object.__setattr__(self, "_battle", battle)
        object.__setattr__(self, "touched", [])

    def __getattr__(self, name):
        self.touched.append(name)
        return getattr(self._battle, name)


@pytest.fixture
def leak(monkeypatch):
    """test_rs_rebuild's helpers, rs_search / rs_rows / rs_engine as the tools import them,
    rs_rows' entity fields unset (the seat reads the engine module's) and rs_elixir's constant
    law."""
    monkeypatch.syspath_prepend(str(TOOLS))
    try:
        importlib.import_module("public_state")
    except Exception as exc:                    # numpy, scripts/vision
        pytest.skip(f"tools/public_state does not import ({type(exc).__name__}: {exc})")
    from tests.pipeline import test_rs_rebuild as RBT
    rr = importlib.import_module("rs_rows")
    sr = importlib.import_module("rs_search")
    eng = rr.rs_engine
    monkeypatch.setattr(rr, "F", None)
    law = rr.rs_elixir.ElixirLaw.constants()
    monkeypatch.setattr(rr.rs_elixir, "default_law", lambda: law)
    return SimpleNamespace(RBT=RBT, search=sr, rows=rr, law=law,
                           cfg=eng.EngineConfig.defaults())


def leak_seat(env, search, *, rb):
    """A SearchSeat with RSAgent's attributes and no checkpoint (no torch), its policy
    test_rs_search's FakePolicy (V = our elixir after the trial), its rebuilder `rb`; it keeps
    every view it built and every row it asked for. Set up as rs_match.build_seat would: team 0,
    delay 24, the spec seed 7; its sends so far are test_rs_rebuild.SENT on both clocks (no
    noise: run_match's seat.sent and its own agree)."""
    SR, RBT = env.search, env.RBT

    class LeakSeat(SR.SearchSeat):
        def __init__(self):
            self.engine = env.cfg
            self.meta, self.model, self.torch, self.vocab_drift = {}, None, None, False
            self.visibility, self.live, self.v2 = None, False, False
            self.rng = ScriptRng(0.99)
            self.wait_bias, self.decide_mode, self.cadence = 0.0, "legacy", 10
            self.lead, self.hold_pending, self.spell_aim = 0, False, False
            self.sent, self.unran = list(RBT.SENT), Counter()
            self.views, self.rows = [], []

        def _catalogue(self):
            return RBT.CATALOGUE

        def search_policy(self):
            return FakePolicy(gate=GATE, best=BEST,
                              value=lambda row: row["state"]["players"][0]["elixir_milli"] / 1e4)

        def search_rebuilder(self, royalesim, sub, fields):
            assert fields == dict(FIELD)
            return rb

        def search_row(self, state, team, mem, sub, royalesim, tick):
            self.rows.append((int(tick), copy.deepcopy(state)))
            return {"tick": int(tick), "state": state}

        def search_law(self):
            return env.law

        def search_view(self, state, team, mem):
            view = super().search_view(state, team, mem)
            self.views.append(copy.deepcopy(view))
            return view

    seat = LeakSeat()
    seat.setup_search(search, team=0, delay=24, opp_delay=24, seed=7)
    seat._search_sent = list(RBT.SENT)
    return seat


def play(env, monkeypatch, search, state, battle=None):
    """One act() of a fresh seat on `state` (seat 0's observed state at T0), the match battle a
    Proxy of `battle`; (the seat, the command, the proxy, the rebuilder)."""
    RBT = env.RBT
    rb, mod = RBT.rebuilder()
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    seat = leak_seat(env, search, rb=rb)
    proxy = Proxy(battle if battle is not None else FakeBattle())
    cmd = seat.act(proxy, state, 0, copy.deepcopy(RBT.PUSH_MEM), RBT.T0, 18, mod)
    return seat, cmd, proxy, rb


def canon(x):
    return json.dumps(x, sort_keys=True, default=str)


def record(seat):
    return {k: v for k, v in seat.last_search.to_json().items() if k != "ms"}


def test_privileged_fields_change_nothing_the_search_sees(leak, monkeypatch):
    RBT = leak.RBT
    plain_state, noised_state = RBT.push_state(), RBT.push_state(noised=True)
    # the noised twin holds what no screen shows
    them = noised_state["players"][1]
    assert them["hand"] != plain_state["players"][1]["hand"] and them["pending"]
    assert noised_state["projectiles"] and len(noised_state["entities"]) == len(
        plain_state["entities"]) + 1
    a, cmd_a, _, rb_a = play(leak, monkeypatch, "s_idle", plain_state)
    b, cmd_b, _, rb_b = play(leak, monkeypatch, "s_idle", noised_state)
    # one public view: the opponent's block is the screen's, hidden columns reset, the invisible
    # Knight and the projectile cut (and the count of what was cut is not in it either)
    assert canon(a.views) == canon(b.views) and len(a.views) == 1
    view = a.views[0]
    assert leak.search.PV.is_public(view, 0, fields=FIELD) == []
    assert "dropped" not in view["public"]
    assert view["players"][1]["hand"] == [] and view["players"][1]["elixir_milli"] == 0
    # one RootSpec (its sha1 key), one set of rows, one search, one decision
    assert a.last_search.key and a.last_search.key == b.last_search.key
    assert a.last_search.lookahead and a.last_search.outcome == "searched"
    assert canon(a.rows) == canon(b.rows) and len(a.rows) > 1
    assert canon(record(a)) == canon(record(b))
    assert a.last_decision == b.last_decision and cmd_a == cmd_b
    assert len(rb_a.stats) and dict(rb_a.stats) == dict(rb_b.stats)


@pytest.mark.parametrize("search", ["s_root", "s_idle"])
def test_a_public_search_reaches_only_check_deploy(leak, monkeypatch, search):
    seat, _, proxy, _ = play(leak, monkeypatch, search, leak.RBT.push_state(noised=True))
    assert set(proxy.touched) <= {"check_deploy"}
    if search == "s_idle":
        assert proxy.touched.count("check_deploy") > 1          # d0's and the candidates'
    assert proxy._battle.calls == [] and proxy._battle.state_reads == 0
    assert not seat.last_search.privileged and not seat.search_info()["privileged"]


def test_only_the_privileged_ceiling_reaches_the_true_battle(leak, monkeypatch):
    battle = FakeBattle()
    battle.tick = leak.RBT.T0
    before = battle.state_json()
    seat, _, proxy, rb = play(leak, monkeypatch, "s_true", leak.RBT.push_state(noised=True),
                              battle=battle)
    assert {"save", "load", "step", "state_json", "check_deploy"} <= set(proxy.touched)
    rec = seat.last_search
    assert rec.privileged and rec.to_json()["privileged"] is True and rec.outcome == "searched"
    assert seat.search_info()["privileged"] is True
    # its trials ran there, from one save, and the battle is as it was
    names = battle.names()
    assert names[0] == "save" and names[-1] == "load" and names.count("step") == rec.trials
    assert battle.state_json() == before and battle.tick == leak.RBT.T0
    # the public root still decided d0 and the candidates' legality at B: the rebuilder built it
    # (one save) and ran no trial on it (a trial steps the horizon's 60 ticks, or 1 + 59)
    assert rb.stats["built"] == 1 and rb.battle.names().count("save") == 1
    assert all(n not in (59, 60) for _, n in leak.RBT.steps(rb.battle))
