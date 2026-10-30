"""S06: tools/rs_search.py, the search core (docs/LIVE_SEARCH_PLAN.md section 3, S06).

No engine, no torch, no data. FakeRebuilder speaks rs_rebuild.Rebuilder's interface (spec, root,
legal, trial, settings, F, SF, tables, last_error, summary) over hand-made states: the root at B
is a state_dict, a trial's after-state is the root with our plays paid for, a tower hit per card
and a game won by a card, as each test scripts it. FakePolicy is decide()'s policy (batch, gate,
tiles) plus values(rows, kind): scripted gate logits, tile logits with a few boosted tiles per
slot, V per after-row by the card played. conftest's FakeBattle is the TRUE battle behind d0's
legality (rs_decide.BattleLegality), so the tests see every call the search makes on it. One
test runs the real Rebuilder on tests/pipeline/test_rs_rebuild.py's CycleBattle (it needs
tools/public_state, numpy only).

The plan's S06 checks: candidate order, pruning and the cap rule removing W; the trust region
keeps P below tau; only P and W finished -> those two are scored, P missing -> d0; every fallback
reason; pending plays land during the advance; slots are remapped by card; a chosen Decision has
the fields vision_play._decide reads; the search never draws from the seat's random generator.
Then the presets, the privileged s_true guard, the public-view guard, choose / cvar on hand-made
matrices, the leak, the hold and the triggers. S09 (the end of the file): s_true's trials run on
the TRUE battle (TrueBattle: conftest's FakeBattle that queues each play under the delay, lands
it on its tick and keeps it all in save() / load()), from one save, each candidate sent now from
its present slot and stepped D + H; the battle is restored and checked after (the plan's S09
test), a play refused at send or when it ran did not run, a broken restore or a delay that is
not the seat's raises TrueCopyError (never a fallback), and every record says privileged.
"""
from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import (DEFAULT_CATALOGUE, FIELD, KING_HP, PRINCESS_HP, Call,
                                     FakeBattle, entity_row, player_dict, state_dict)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rs_search as SR  # noqa: E402

import rs_decide as RD  # noqa: E402  (tools/ is on sys.path once rs_search is imported)
import rs_elixir  # noqa: E402
import rs_public_view as PV  # noqa: E402
import rs_rebuild as RB  # noqa: E402

SUB = 18
T0, D, H = 1000, 24, 60
B = T0 + D
LAW = rs_elixir.ElixirLaw.constants()
CANNON, FIREBALL, HOG, ICEGOLEM, KNIGHT = 0, 1, 2, 3, 8
COST = {i: 1000 * int(row[1]) for i, row in enumerate(DEFAULT_CATALOGUE)}
#: Own-frame tiles (ty * 18 + tx; team 0's half is ty >= 17). Hog: (3,17), (4,17) next to it,
#: (3,19), (5,17); Cannon (8,22), (9,22); Fireball (3,6), (14,6).
HOG_TILES = (309, 310, 345, 311)
CANNON_TILES = (404, 405)
FIREBALL_TILES = (111, 122)
BEST = {0: CANNON_TILES, 1: FIREBALL_TILES, 2: HOG_TILES}
#: [wait, Cannon, Fireball, Hog, Ice Golem]: the gate plays the Hog (slot 2) unless it waits.
GATE = [0.0, 1.0, 0.5, 3.0, 0.0]
DCFG = RD.DecideConfig.legacy(0.0)
NOT_ENOUGH = 4                                  # conftest DEPLOY_REASONS "NOT_ENOUGH_ELIXIR"


def tap(tile, team=0):
    wx, wy = RD.tile_centre_world(tile, team)
    return wx * SUB, wy * SUB


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class ScriptRng:
    """The seat's generator as legacy decide uses it: random() from a script, every draw
    counted."""

    def __init__(self, *values):
        self.values = list(values) or [0.99]
        self.draws = 0

    def random(self):
        v = self.values[min(self.draws, len(self.values) - 1)]
        self.draws += 1
        return v


class FakePolicy:
    """batch / gate / tiles as decide reads them, values(rows, kind) as the search does. Tile
    logits: -0.001 x tile everywhere, the slot's boosted tiles 10, 9, 8, ... in order."""

    value_kind = SR.VALUE_SWING

    def __init__(self, gate=GATE, best=BEST, value=None, kind=SR.VALUE_SWING):
        self.g = np.asarray(gate, dtype=np.float32)
        self.best = {int(k): tuple(v) for k, v in dict(best).items()}
        self.value = value or (lambda row: 0.0)
        self.value_kind = kind
        self.batches = 0
        self.tile_calls = []
        self.value_calls = []

    def batch(self, row):
        self.batches += 1
        return {"row": row}

    def gate(self, b):
        return self.g.copy()

    def tiles(self, b, slot):
        self.tile_calls.append(int(slot))
        lg = -np.arange(RD.TILES, dtype=np.float32) * 0.001
        for k, t in enumerate(self.best.get(int(slot), ())):
            lg[t] = 10.0 - k
        return lg

    def values(self, rows, kind=None):
        rows = list(rows)
        self.value_calls.append((len(rows), kind))
        return np.asarray([self.value(r) for r in rows], dtype=np.float64)


def value_by_card(values):
    """V of an after-row: values[the card our trial played] (values["wait"] for none)."""
    def value(row):
        played = row["state"].get("played") or ["wait"]
        return float(values.get(played[-1], 0.0))
    return value


class FakeRebuilder:
    """rs_rebuild.Rebuilder's interface over a scripted root (module doc).

    elixir / hand    our elixir and hand at B (the root)
    fidelity, refuse (a Refusal reason), spec_error (a RebuildError message), over (the root is
    a finished game), later (our plays in flight landing after B), legal_fn(card, x, y) -> code
    damage {card: hp off the enemy's left princess tower}, wins {card: the trial ends won}
    refused  cards whose trial plays the engine refuses
    clock    a FakeClock: root() and trial() advance it by root_s / trial_s"""

    def __init__(self, *, elixir=5000, hand=(CANNON, FIREBALL, HOG, ICEGOLEM), fidelity=1.0,
                 refuse=None, spec_error=None, over=False, later=(), legal_fn=None, damage=None,
                 wins=(), refused=(), clock=None, root_s=0.0, trial_s=0.0):
        self.settings = RB.RebuildSettings()
        self.F = dict(FIELD)
        self.SF = None
        self.tables = SimpleNamespace(cost=dict(COST))
        self.last_error = None
        self.elixir, self.hand, self.fidelity = elixir, list(hand), fidelity
        self.refuse, self.spec_error, self.over = refuse, spec_error, over
        self.later = tuple(tuple(p) for p in later)
        self.legal_fn = legal_fn
        self.damage, self.wins, self.refused = dict(damage or {}), set(wins), set(refused)
        self.clock, self.root_s, self.trial_s = clock, root_s, trial_s
        self.specs, self.roots, self.legal_calls, self.trials = [], [], [], []

    def spec(self, view, *, land_tick=None, seed=0, level=None, opp_fill=()):
        if self.spec_error:
            raise RB.RebuildError(self.spec_error)
        spec = SimpleNamespace(tick=int(view["tick"]), land_tick=int(land_tick), seed=seed,
                               level=level, opp_fill=list(opp_fill))
        self.specs.append(spec)
        return spec

    def root(self, spec):
        self.roots.append(spec)
        if self.clock is not None:
            self.clock.t += self.root_s
        if self.refuse:
            self.last_error = RB.Refusal(self.refuse, "scripted")
            return None
        me = player_dict(0, elixir_milli=self.elixir, hand=list(self.hand))
        state = state_dict(spec.land_tick, players=[me, player_dict(1)], game_over=self.over)
        rs = RB.RootSpec(level=RB.R1, team=0, tick=spec.tick, land_tick=spec.land_tick,
                         later=self.later)
        return RB.Built(spec=rs, key="fake", battle=None, blob=None, state=state, raw="",
                        labels={}, fidelity=self.fidelity, flags=("fake",), report={})

    def legal(self, card, x, y, built=None, side=None):
        slot = built.slot(card)
        self.legal_calls.append((int(card), int(x), int(y), slot))
        if slot is None:
            return RB.NOT_IN_HAND
        return int(self.legal_fn(card, x, y)) if self.legal_fn else 0

    def trial(self, built, plays, ticks, *, later=True, verify=False):
        self.trials.append((list(plays), int(ticks), bool(verify)))
        if self.clock is not None:
            self.clock.t += self.trial_s
        after = copy.deepcopy(built.state)
        after["tick"] = built.tick + int(ticks)
        run = {"ran": [], "refused": [], "skipped": [], "game_over": False}
        for i, (tick, team, card, x, y) in enumerate(plays):
            if card in self.refused:
                run["refused"].append(("play", i, tick, NOT_ENOUGH))
                continue
            run["ran"].append(("play", i, tick, 0))
            if team != 0:
                continue
            me = after["players"][0]
            me["elixir_milli"] = max(0, me["elixir_milli"] - COST[card])
            after.setdefault("played", []).append(card)
            red = after["players"][1]
            red["tower_hp"][1] = max(0, red["tower_hp"][1] - self.damage.get(card, 0))
            if card in self.wins:
                after["game_over"], after["winner"] = True, 0
        return after, run

    def summary(self):
        return {"fake": len(self.roots)}


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_view(*, tick=T0, elixir=7000, opp_plays=(), entities=(), hand=None):
    """Team 0's public view at `tick`: hand Cannon, Fireball, Hog, Ice Golem (3, 4, 4, 2)."""
    me = player_dict(0, elixir_milli=elixir, hand=hand)
    state = state_dict(tick, players=[me, player_dict(1)], entities=entities)
    mem = {"last": {0: [], 1: [(c + 2, t) for c, t in opp_plays]},
           "seen": {0: set(), 1: {c + 2 for c, _ in opp_plays}}}
    return PV.public_view(state, 0, mem, fields=FIELD, opp_elixir=(5000, True))


def rows_log():
    log = []

    def row_fn(state, tick):
        log.append((int(tick), state))
        return {"tick": int(tick), "state": state}
    return row_fn, log


def searcher(config="s_idle", *, policy=None, rb=None, **kw):
    kw.setdefault("dcfg", DCFG)
    kw.setdefault("law", LAW)
    return SR.Searcher(policy or FakePolicy(), rb or FakeRebuilder(), config, **kw)


def run(s, view=None, *, rng=None, battle=None, row_fn=None, **kw):
    """One decide on team 0 with D = 24; the TRUE battle (a FakeBattle) behind d0's legality."""
    battle = battle if battle is not None else FakeBattle()
    if row_fn is None:
        row_fn, _ = rows_log()
    return s.decide(view if view is not None else make_view(), team=0, land_ticks=D,
                    legality=RD.BattleLegality(battle), rng=rng or ScriptRng(0.99),
                    row_fn=row_fn, **kw)


def roles(rec):
    return [(c.role, c.slot, c.card, c.tile) for c in rec.candidates]


# ---------------------------------------------------------------------------
# the configuration, the presets, the privileged arm
# ---------------------------------------------------------------------------

def test_presets_expand_to_their_full_content():
    root, idle = SR.SearchConfig.named("s_root"), SR.SearchConfig.named("s_idle")
    reply, true = SR.SearchConfig.named("s_reply"), SR.SearchConfig.named("s_true")
    assert (root.lookahead, root.max_trials, root.opponent, root.trials) == (True, 0, "none",
                                                                             False)
    assert (idle.max_trials, idle.horizon, idle.worlds, idle.opponent, idle.deadline_ms) == (
        8, 60, 1, "idle", 80.0)
    assert (reply.max_trials, reply.horizon, reply.worlds, reply.opponent,
            reply.deadline_ms) == (8, 80, 4, "reply", 120.0)
    assert (true.horizon, true.worlds, true.opponent) == (60, 1, "true")
    assert (idle.trust, idle.prior_weight, idle.leak_weight, idle.prior_drop, idle.z,
            idle.cvar_alpha, idle.hold_ticks, idle.elixir_trigger) == (0.05, 0.05, 0.05, 4.0, 1.0,
                                                                       0.5, 10, 8000)
    full = SR.expand("s_idle")
    assert set(full) == {f.name for f in dataclasses.fields(SR.SearchConfig)}
    assert full["preset"] == "s_idle" and json.loads(json.dumps(full)) == full
    assert SR.SearchConfig.from_json(full) == idle and SR.resolve(full) == idle
    # a key records the content: an override changes it, verify (a check only) does not
    assert idle.key == SR.SearchConfig.from_json(full).key
    assert idle.replace(horizon=40).key != idle.key
    assert idle.replace(verify=True).key == idle.key
    assert SR.resolve(" s_idle ") == idle and SR.resolve(idle) is idle
    custom = SR.resolve('{"preset": "s_idle", "horizon": 40, "max_trials": 5}')
    assert (custom.preset, custom.horizon, custom.max_trials, custom.deadline_ms) == (
        "s_idle", 40, 5, 80.0)
    for off in (None, "off", "none", "", " OFF "):
        assert SR.resolve(off) is None and SR.expand(off) is None
    assert SR.SearchConfig(max_trials=8.0).max_trials == 8


@pytest.mark.parametrize("bad", [
    {"max_trials": 1},                      # fewer trials than min_candidates
    {"opponent": "none"},                   # trials need an opponent arm
    {"worlds": 2},                          # idle worlds are all alike
    {"level": "R0"},                        # R0 reads the seat's privileged state
    {"horizon": True}, {"trust": float("nan")}, {"wait": 1}, {"deadline_ms": 0},
    {"opponent": "ghost"}, {"cvar_alpha": 0.0}, {"min_candidates": 1}, {"max_errors": 0},
])
def test_bad_configs_are_refused(bad):
    with pytest.raises(SR.SearchConfigError):
        SR.SearchConfig(**bad)


def test_resolve_refuses_what_it_does_not_know():
    for value in ("s_fast", '{"preset": "s_idle", "horizn": 40}', "{nope", 3, ["s_idle"]):
        with pytest.raises(SR.SearchConfigError):
            SR.resolve(value)
    with pytest.raises(SR.SearchConfigError):
        SR.SearchConfig().replace(horizn=40)


def test_s_true_is_privileged_and_never_promoted_or_played_live():
    true = SR.SearchConfig.named("s_true")
    assert true.privileged and not true.promotable
    assert not SR.SearchConfig.named("s_idle").privileged
    assert SR.is_privileged("s_true") and SR.is_privileged({"preset": "s_idle", "opponent": "true"})
    assert not SR.is_privileged("s_reply") and not SR.is_privileged("off")
    with pytest.raises(SR.PrivilegedSearchError, match="never a promoted"):
        SR.check_promotable("s_true")
    with pytest.raises(SR.PrivilegedSearchError, match="played live"):
        SR.check_live(SR.expand("s_true"))
    assert SR.check_promotable("s_idle") == SR.SearchConfig.named("s_idle")
    assert SR.check_live("off") is None
    # a Searcher refuses it unless told privileged=True (S09 builds it then); its decide needs
    # the true battle, and no public preset takes one
    with pytest.raises(SR.PrivilegedSearchError):
        searcher("s_true")
    s = searcher("s_true", privileged=True)
    assert s.config.privileged and s.summary()["privileged"]
    with pytest.raises(SR.SearchConfigError, match="TRUE battle") as err:
        run(s)
    assert not isinstance(err.value, SR.PrivilegedSearchError)
    with pytest.raises(SR.PrivilegedSearchError, match="is public"):
        run(searcher(), true_battle=FakeBattle())
    with pytest.raises(SR.SearchConfigError, match="no worlds"):
        searcher("s_true", privileged=True, worlds=lambda *a: [])


def test_a_searcher_refuses_what_it_cannot_run():
    with pytest.raises(SR.SearchConfigError, match="off"):
        searcher("off")
    with pytest.raises(SR.SearchConfigError, match="S07"):
        searcher("s_reply")                                 # no opponent world model yet

    class NoValues(FakePolicy):
        values = None

    with pytest.raises(SR.SearchConfigError, match="values"):
        searcher("s_idle", policy=NoValues())
    searcher("s_root", policy=NoValues())                   # the look-ahead needs no V


# ---------------------------------------------------------------------------
# choose and cvar on hand-made matrices
# ---------------------------------------------------------------------------

def test_cvar_is_the_mean_of_the_worst_share():
    assert SR.cvar([3.0, 1.0, 2.0, 10.0], 0.5) == pytest.approx(1.5)
    assert SR.cvar([1.0, 2.0, 3.0], 0.5) == pytest.approx(1.5)     # ceil(1.5) = 2 values
    assert SR.cvar([5.0], 0.5) == 5.0 and SR.cvar([1.0, 3.0], 1.0) == 2.0
    with pytest.raises(ValueError):
        SR.cvar([], 0.5)


def test_choose_one_world_needs_the_trust_margin():
    c = SR.choose([[0.0], [0.04], [0.06]], [0.0, 0.0, 0.0])
    assert (c.best, c.override) == (2, True) and c.gain == pytest.approx(0.06)
    c = SR.choose([[0.0], [0.04]], [0.0, 0.0])
    assert (c.best, c.override, c.threshold) == (1, False, pytest.approx(0.05))
    # the prior term counts: 0.06 better on the board, 1 nat less likely: 0.01 better in all
    c = SR.choose([[0.0], [0.06]], [0.0, -1.0])
    assert c.best == 1 and not c.override and c.gain == pytest.approx(0.01)
    # a tie keeps the incumbent; a candidate that did not run scores -inf
    c = SR.choose([[0.1], [0.1], None], [0.0, 0.0, 0.0])
    assert (c.best, c.override) == (0, False) and c.totals[2] == -math.inf
    # the incumbent did not run: nothing to compare with
    c = SR.choose([None, [1.0]], [0.0, 0.0])
    assert (c.best, c.override, c.gain) == (0, False, None)


def test_choose_k_worlds_needs_the_paired_standard_error():
    inc = [0.0, 0.0, 0.0, 0.0]
    c = SR.choose([inc, [0.2, 0.2, 0.2, 0.2]], [0.0, 0.0])
    assert c.override and c.threshold == pytest.approx(0.05)       # sd 0
    # mean gain 0.1 > tau, but sd 0.2309 over 4 worlds: 0.05 + 0.2309 / 2 = 0.1655 > 0.1
    c = SR.choose([inc, [0.3, -0.1, 0.3, -0.1]], [0.0, 0.0], alpha=1.0)
    assert c.best == 1 and not c.override
    assert c.gain == pytest.approx(0.1) and c.threshold == pytest.approx(0.05 + 0.23094 / 2,
                                                                          abs=1e-4)
    # the ranking is CVaR at alpha 0.5 (worst half): a candidate good on average but bad in its
    # worst worlds does not win
    c = SR.choose([inc, [1.0, 1.0, -0.6, -0.6]], [0.0, 0.0])
    assert c.totals[1] == pytest.approx(-0.6) and c.best == 0
    with pytest.raises(ValueError):
        SR.choose([[0.0], [0.0, 1.0]], [0.0, 0.0])


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

def test_candidate_order_pruning_and_the_shared_pass():
    policy, rb = FakePolicy(), FakeRebuilder()
    battle = FakeBattle()
    d, rec = run(searcher(policy=policy, rb=rb), battle=battle)
    # P (d0's Hog), W, the Hog at two more tiles two apart (310 is next to 309), then the two
    # best other slots (Cannon, Fireball; not the Ice Golem) at their two best tiles each
    assert roles(rec) == [("P", 2, HOG, 309), ("W", -1, -1, -1), ("T", 2, HOG, 345),
                          ("T", 2, HOG, 311), ("S", 0, CANNON, 404), ("S", 0, CANNON, 405),
                          ("S", 1, FIREBALL, 111), ("S", 1, FIREBALL, 122)]
    assert rec.dropped == 0 and rec.outcome == SR.R_SEARCHED and rec.triggers == ("play",)
    # priors: log q(slot) + log-softmax(tile), q decide's gate distribution
    g = np.asarray(GATE, dtype=np.float64)
    q = RD.gate_probs(g, np.ones(5, dtype=bool), 10)
    lg = np.asarray(policy.tiles(None, 2), dtype=np.float64)
    lse = float(np.log(np.exp(lg - lg.max()).sum()) + lg.max())
    p, w = rec.candidates[0], rec.candidates[1]
    assert p.prior == pytest.approx(math.log(q[3]) + 10.0 - lse)
    assert w.prior == pytest.approx(math.log(q[0]))
    # each tile passed d0's legality on the true battle (its present slot) and check_deploy at B
    assert {(c.card, c.x, c.y) for c in rec.candidates if c.is_play} <= {
        (card, x, y) for card, x, y, _ in rb.legal_calls}
    assert {(2, *tap(t)) for t in (345, 311)} <= {(s, x, y) for _, s, x, y in battle.deploy_checks}
    # one batch for the decision's row: decide, the priors and every tile read its one pass
    assert policy.batches == 1
    # every candidate was tried, by card at B, for H ticks; one batched V for the after-rows
    assert [t[0] for t in rb.trials] == [[c.play(B, 0)] if c.is_play else []
                                         for c in rec.candidates]
    assert {t[1] for t in rb.trials} == {H} and policy.value_calls == [(8, "swing")]
    # V 0 everywhere: the prior decides and P is kept
    assert rec.chosen == 0 and not rec.override and d.command == (0, 2, *tap(309))
    assert d.mode == "legacy+search" and rec.decision["mode"] == "legacy+search"
    json.dumps(rec.to_json())


def test_candidates_below_the_prior_drop_are_pruned():
    # Fireball barely in the gate: still one of the two other slots, but 4 nats below P
    policy = FakePolicy(gate=[0.0, 1.0, -5.0, 3.0, -6.0])
    _, rec = run(searcher(policy=policy))
    assert [r[0] for r in roles(rec)] == ["P", "W", "T", "T", "S", "S"]
    assert rec.dropped == 2 and all(c.card != FIREBALL for c in rec.candidates)
    floor = rec.candidates[0].prior - 4.0
    assert all(c.prior >= floor for c in rec.candidates)
    # at most max_trials, the incumbent first
    _, rec = run(searcher(SR.SearchConfig.named("s_idle", max_trials=3)))
    assert [r[0] for r in roles(rec)] == ["P", "W", "T"]


def test_the_cap_rule_removes_w():
    # the live preset's cap rule: at 9600 elixir d0 may not wait, so W is no candidate
    dcfg = RD.DecideConfig.live(0.0)
    _, rec = run(searcher(dcfg=dcfg), make_view(elixir=9600))
    assert rec.d0["kind"] == RD.PLAY and "W" not in [r[0] for r in roles(rec)]
    assert "elixir" in rec.triggers and rec.candidates[0].role == "P"
    # a drawn wait: W is the incumbent, then the two best slots (Hog, Cannon)
    _, rec = run(searcher(), make_view(elixir=8500), rng=ScriptRng(0.0))
    assert rec.d0["kind"] == RD.WAIT and rec.triggers == ("elixir",)
    assert roles(rec) == [("W", -1, -1, -1), ("S", 2, HOG, 309), ("S", 2, HOG, 310),
                          ("S", 0, CANNON, 404), ("S", 0, CANNON, 405)]


def test_slots_are_remapped_by_card():
    # at B the hand is in another order (the rebuild's): the trials send cards, the rebuild maps
    # each to its slot there, and the Decision keeps the PRESENT slot (the command is sent now)
    rb = FakeRebuilder(hand=(HOG, ICEGOLEM, CANNON, FIREBALL))
    policy = FakePolicy(value=value_by_card({CANNON: 1.0}))
    d, rec = run(searcher(policy=policy, rb=rb))
    pick = rec.candidates[rec.chosen]
    assert (pick.role, pick.card, pick.slot) == ("S", CANNON, 0) and rec.override
    assert d.slot == 0 and d.card == CANNON and d.command == (0, 0, *tap(404))
    assert (CANNON, *tap(404), 2) in rb.legal_calls                # slot 2 at B
    assert [(CANNON, *tap(404))] in [[p[2:] for p in t[0]] for t in rb.trials]


def test_a_card_not_in_hand_at_b_is_no_candidate():
    # the Fireball has left the hand by B (our play in flight cycles it): no Fireball candidate
    rb = FakeRebuilder(hand=(CANNON, 12, HOG, ICEGOLEM))
    _, rec = run(searcher(rb=rb))
    assert all(c.card != FIREBALL for c in rec.candidates)
    assert (FIREBALL, *tap(111), None) in rb.legal_calls


# ---------------------------------------------------------------------------
# the trust region, the scores, the deadline
# ---------------------------------------------------------------------------

def priors_of(**kw):
    _, rec = run(searcher(**kw))
    return {(c.role, c.card, c.tile): c.prior for c in rec.candidates}


@pytest.mark.parametrize("margin, override", [(-0.01, False), (0.01, True)])
def test_the_trust_region_keeps_p_below_tau(margin, override):
    pr = priors_of()
    pw = 0.05 * (pr[("W", -1, -1)] - pr[("P", HOG, 309)])
    # W's board value beats P's by tau + margin once the prior term is taken into account
    values = {HOG: 0.2, "wait": 0.2 + 0.05 + margin - pw}
    d, rec = run(searcher(policy=FakePolicy(value=value_by_card(values))))
    assert rec.outcome == SR.R_SEARCHED and rec.override is override
    assert rec.gain == pytest.approx(0.05 + margin)
    if override:
        assert rec.chosen == 1 and d.kind == RD.WAIT and d.reason == SR.REASON_WAIT
    else:
        assert rec.chosen == 1 and d.command == (0, 2, *tap(309))   # best, but vetoed
    assert d.mode == "legacy+search"


def test_swing_value_terminal_and_leak_make_the_score():
    rb = FakeRebuilder(elixir=9500, damage={CANNON: PRINCESS_HP // 2}, wins={FIREBALL})
    policy = FakePolicy(value=value_by_card({HOG: 0.3, CANNON: 0.1}))
    d, rec = run(searcher(policy=policy, rb=rb))
    by = {(c.role, c.card, c.tile): c for c in rec.candidates}
    w, p = by[("W", -1, -1)], by[("P", HOG, 309)]
    cannon, fireball = by[("S", CANNON, 404)], by[("S", FIREBALL, 111)]
    # leak: 9500 at B, 60 ticks at 1x regenerate 1071 milli; the cap is 10000
    assert w.parts[0]["leak"] == pytest.approx(0.571) and p.parts[0]["leak"] == 0.0
    assert w.world_scores == [pytest.approx(-0.05 * 0.571)]
    # swing: the enemy's tower fractions lost (summed, tools/rl/teacher.tower_swing): half a
    # princess tower; V from the policy
    assert cannon.parts[0]["swing"] == pytest.approx(0.5)
    assert cannon.world_scores[0] == pytest.approx(cannon.parts[0]["swing"] + 0.1)
    assert p.world_scores == [pytest.approx(0.3)]
    # a won game scores the result x 3 (the value head's units), and needs no V
    assert fireball.parts[0]["terminal"] and fireball.parts[0]["value"] == 3.0
    assert policy.value_calls == [(6, "swing")]                    # 8 trials, 2 Fireballs won
    assert rec.candidates[rec.chosen].card == FIREBALL and d.card == FIREBALL and d.slot == 1
    # our later play in flight (landing before B + H) is paid out of the bar: no leak
    rb = FakeRebuilder(elixir=9500, later=[(B + 6, CANNON, *tap(404))])
    _, rec = run(searcher(rb=rb))
    assert rec.candidates[1].parts[0]["leak"] == 0.0


def test_a_critic_scores_tanh_plus_the_potential():
    rb = FakeRebuilder(damage={CANNON: PRINCESS_HP})
    policy = FakePolicy(value=lambda row: 0.25, kind=SR.VALUE_CRITIC)
    _, rec = run(searcher(policy=policy, rb=rb))
    cannon = next(c for c in rec.candidates if c.card == CANNON)
    # Phi = 0.1 x (own - enemy tower fraction): the enemy lost one of three towers' hp
    assert cannon.parts[0]["value"] == pytest.approx(0.25 + 0.1 * (1.0 - 2.0 / 3.0))
    assert policy.value_calls == [(8, "critic")]


def test_only_p_and_w_finished_are_scored():
    clock = FakeClock()
    rb = FakeRebuilder(clock=clock, trial_s=0.045)
    s = searcher(rb=rb, clock=clock, realtime=True)              # s_idle: 80 ms
    d, rec = run(s)
    assert rec.outcome == SR.R_SEARCHED and rec.trials == 2 and len(rb.trials) == 2
    assert [c.ran for c in rec.candidates] == [True, True] + [False] * 6
    assert [c.score is not None for c in rec.candidates] == [True, True] + [False] * 6
    # without realtime (the simulator) the same search runs every trial: no clock in a game
    rb = FakeRebuilder(clock=clock, trial_s=0.045)
    _, rec = run(searcher(rb=rb, clock=clock))
    assert rec.trials == 8


def test_p_missing_plays_d0():
    rb = FakeRebuilder(refused={HOG})                            # P (and the T tiles) refused
    d, rec = run(searcher(rb=rb))
    assert rec.outcome == SR.R_INCUMBENT and not rec.candidates[0].ran
    assert d.command == (0, 2, *tap(309)) and d.mode == "legacy"   # d0 itself, untagged
    # the deadline passed before P could run (a slow rebuild)
    clock = FakeClock()
    rb = FakeRebuilder(clock=clock, root_s=0.1)
    d, rec = run(searcher(rb=rb, clock=clock, realtime=True))
    assert rec.outcome == SR.R_DEADLINE and rb.trials == [] and d.mode == "legacy"


# ---------------------------------------------------------------------------
# fallbacks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rb_kw, cfg, outcome", [
    ({"refuse": "reset_refused"}, {}, "rebuild:reset_refused"),
    ({"refuse": "units_dropped"}, {}, "rebuild:units_dropped"),
    ({"spec_error": "no card left"}, {}, SR.R_SPEC),
    ({"fidelity": 0.5}, {}, SR.R_FIDELITY),
    ({"over": True}, {}, SR.R_ROOT_OVER),
    ({}, {"wait": False, "extra_tiles": 0, "other_slots": 0}, SR.R_FEW_CANDS),
    ({"legal_fn": lambda card, x, y: NOT_ENOUGH if card == HOG else 0}, {},
     SR.R_INCUMBENT_ILLEGAL),
    ({"refused": {HOG}}, {}, SR.R_INCUMBENT),
    ({"refused": {CANNON, FIREBALL}}, {"wait": False, "extra_tiles": 0}, SR.R_FEW_TRIALS),
])
def test_every_fallback_plays_d0(rb_kw, cfg, outcome):
    rb = FakeRebuilder(**rb_kw)
    row_fn, log = rows_log()
    s = searcher(SR.SearchConfig.named("s_idle", **cfg), rb=rb)
    d, rec = run(s, row_fn=row_fn)
    assert rec.outcome == outcome
    assert d.command == (0, 2, *tap(309)) and d.mode == "legacy"
    assert s.summary()["outcomes"] == {outcome: 1}
    if outcome.startswith("rebuild") or outcome in (SR.R_SPEC, SR.R_FIDELITY, SR.R_ROOT_OVER):
        # no usable root: no look-ahead either, d0 on the present row, and no trials
        assert not rec.lookahead and log[0][0] == T0 and rb.trials == []
    else:
        assert rec.lookahead and log[0][0] == B


def test_not_triggered_plays_d0_and_each_trigger_fires():
    rb = FakeRebuilder()
    s = searcher(rb=rb)
    d, rec = run(s, rng=ScriptRng(0.0))                         # a drawn wait, nothing else
    assert rec.outcome == SR.R_NOT_TRIGGERED and d.kind == RD.WAIT and d.reason == RD.R_WAIT
    assert rb.trials == [] and rec.lookahead                    # the root was built for d0
    # a new opponent play: once
    view = make_view(tick=T0 + 10, opp_plays=[(KNIGHT, T0 + 5)])
    _, rec = run(s, view, rng=ScriptRng(0.0))
    assert rec.triggers == ("opp_play",) and rec.outcome == SR.R_SEARCHED
    _, rec = run(s, make_view(tick=T0 + 20, opp_plays=[(KNIGHT, T0 + 5)]), rng=ScriptRng(0.0))
    assert rec.outcome == SR.R_NOT_TRIGGERED
    # an enemy push in our half (rs_decide.threat_lane)
    knight = entity_row(uid=50, team=1, kind=0, card_id=KNIGHT, x=3500 * SUB, y=9000 * SUB,
                        hp=1766, unit_type=8)
    _, rec = run(s, make_view(tick=T0 + 30, entities=[knight]), rng=ScriptRng(0.0))
    assert rec.triggers == ("push",)
    assert s.summary()["triggers"] == {"opp_play": 1, "push": 1}


def test_three_exceptions_turn_search_off_for_the_battle():
    rb = FakeRebuilder()
    s = searcher(rb=rb)

    def row_fn(state, tick):
        if tick == B + H + 10 * (len(rb.roots) - 1):            # every after-row raises
            raise RuntimeError("no row")
        return {"tick": tick, "state": state}

    for k in range(3):
        d, rec = s.decide(make_view(tick=T0 + 10 * k), team=0, land_ticks=D,
                          legality=RD.BattleLegality(FakeBattle()), rng=ScriptRng(0.99),
                          row_fn=row_fn)
        assert rec.outcome == SR.R_EXCEPTION and "no row" in rec.errors[0]
        assert d.mode == "legacy" and d.slot == 2
    roots = len(rb.roots)
    d, rec = s.decide(make_view(tick=T0 + 30), team=0, land_ticks=D,
                      legality=RD.BattleLegality(FakeBattle()), rng=ScriptRng(0.99),
                      row_fn=row_fn)
    assert rec.outcome == SR.R_DISABLED and not rec.lookahead and len(rb.roots) == roots
    assert d.slot == 2 and s.summary()["disabled"] == 1 and s.summary()["errors"] == 3
    # a new battle (the tick goes back) turns it on again
    _, rec = run(s, make_view(tick=200))
    assert rec.outcome == SR.R_SEARCHED and rec.lookahead
    # a rebuilder refusal "error" (an exception inside it) counts as one
    s = searcher(rb=FakeRebuilder(refuse=RB.REFUSED_ERROR))
    _, rec = run(s)
    assert rec.outcome == "rebuild:error" and s.summary()["errors"] == 1


def test_the_root_only_preset_decides_at_b_and_tries_nothing():
    rb, policy = FakeRebuilder(), FakePolicy()
    row_fn, log = rows_log()
    d, rec = run(searcher("s_root", rb=rb, policy=policy), row_fn=row_fn)
    assert rec.outcome == SR.R_ROOT and rec.lookahead and rb.trials == []
    assert log[0][0] == B and len(log) == 1 and policy.value_calls == []
    # the row's board is the root's, its hand, elixir and next card the present ones
    assert log[0][1]["tick"] == B and log[0][1]["players"][0]["elixir_milli"] == 7000
    assert d.mode == "legacy" and d.command == (0, 2, *tap(309))   # exactly decide's
    assert rb.specs[0].land_tick == B and rb.specs[0].tick == T0


def test_a_decision_time_checkpoint_rebuilds_only_for_the_trials():
    rb = FakeRebuilder()
    row_fn, log = rows_log()
    d, rec = run(searcher(rb=rb, landing=False), row_fn=row_fn)
    assert not rec.lookahead and log[0][0] == T0 and rec.outcome == SR.R_SEARCHED
    assert len(rb.roots) == 1 and len(rb.trials) == 8
    rb = FakeRebuilder()
    _, rec = run(searcher(rb=rb, landing=False), rng=ScriptRng(0.0))
    assert rec.outcome == SR.R_NOT_TRIGGERED and rb.roots == []   # never rebuilt


# ---------------------------------------------------------------------------
# the hold, the worlds
# ---------------------------------------------------------------------------

def test_a_wait_verdict_holds_until_a_new_opponent_play():
    rb = FakeRebuilder()
    s = searcher(rb=rb, policy=FakePolicy(value=value_by_card({"wait": 1.0})))
    d, rec = run(s)
    assert rec.override and d.kind == RD.WAIT and d.reason == SR.REASON_WAIT
    n = len(rb.trials)
    d, rec = run(s, make_view(tick=T0 + 5))                     # d0 plays again: held
    assert rec.outcome == SR.R_HELD and d.kind == RD.WAIT and d.reason == SR.REASON_HOLD
    assert d.mode == "legacy+search" and len(rb.trials) == n
    d, rec = run(s, make_view(tick=T0 + 6, opp_plays=[(KNIGHT, T0 + 6)]))
    assert rec.outcome == SR.R_SEARCHED and len(rb.trials) > n  # a new play ends it
    _, rec = run(s, make_view(tick=T0 + 30, opp_plays=[(KNIGHT, T0 + 6)]))
    assert rec.outcome == SR.R_SEARCHED                          # and it runs out after 10 ticks


def test_worlds_run_every_candidate_once_each_with_their_own_generator():
    seen = []

    def worlds(searcher_, root, view, team, land, rng):
        seen.append(rng)
        knight = (land + 5, 1, KNIGHT, 9000 * SUB, 20000 * SUB)
        return [SR.World(0, root), SR.World(1, root, plays=(knight,))]

    rb = FakeRebuilder()
    seat_rng = ScriptRng(0.99)
    s = searcher(SR.SearchConfig.named("s_reply", worlds=2), rb=rb, worlds=worlds)
    d, rec = run(s, rng=seat_rng)
    assert rec.worlds == 2 and len(rb.trials) == 2 * len(rec.candidates)
    assert rb.trials[1][0][0][2] == KNIGHT and rb.trials[1][0][1][2] == HOG   # theirs, then ours
    assert all(len(c.world_scores) == 2 for c in rec.candidates if c.ran)
    assert isinstance(seen[0], np.random.Generator) and seat_rng.draws == 1
    assert rec.outcome == SR.R_SEARCHED and {t[1] for t in rb.trials} == {80}


# ---------------------------------------------------------------------------
# the decision, the generator, the public view, the true battle
# ---------------------------------------------------------------------------

def test_a_chosen_decision_has_what_vision_play_reads():
    policy = FakePolicy(value=value_by_card({FIREBALL: 1.0}))
    d, rec = run(searcher(policy=policy))
    assert rec.override and d.is_play
    assert (d.slot, d.card, d.cost, d.tile, d.tx, d.ty) == (1, FIREBALL, 4, 111, 3, 6)
    assert (d.x_sub, d.y_sub) == tap(111) and d.command == (0, 1, *tap(111))
    assert d.world_mt == RD.tile_centre_world(111, 0) and d.world_tile == (3, 25)
    kw = d.suggest_kwargs()
    assert {"tick", "p_wait", "card", "slot", "tile", "model_tile", "capped",
            "fallback"} <= set(kw)
    assert kw["card"] == FIREBALL and kw["slot"] == 1 and kw["model_tile"] == [3, 6]
    rec_d = d.to_record()
    json.dumps(rec_d)
    assert rec_d["command"] == [0, 1, *tap(111)] and rec_d["mode"] == "legacy+search"
    assert d.tick == T0 and d.legality == "battle" and d.p_wait == pytest.approx(1 / 26.4526,
                                                                              rel=1e-4)
    # a chosen wait: no command, the search's reason
    d, _ = run(searcher(policy=FakePolicy(value=value_by_card({"wait": 1.0}))))
    assert d.command is None and d.suggest_kwargs()["reason"] == SR.REASON_WAIT
    assert (d.slot, d.card, d.tile, d.x_sub) == (-1, -1, -1, None)


def test_the_search_never_draws_from_the_seats_generator():
    policy = FakePolicy(value=value_by_card({CANNON: 1.0}))
    rng = ScriptRng(0.99)
    d, rec = run(searcher(policy=policy), rng=rng)
    assert rec.override and rng.draws == 1                       # decide's one draw only
    # d0 is decide's own, call for call: the same row, state, legality and draw
    root = FakeRebuilder().root(SimpleNamespace(tick=T0, land_tick=B))
    row = {"tick": B, "state": RB.with_present(root.state, make_view())}
    plain = RD.decide(FakePolicy(), row, make_view(), 0, RD.BattleLegality(FakeBattle()), DCFG,
                      rng=ScriptRng(0.99), tick=T0)
    assert rec.d0 == {"kind": plain.kind, "reason": plain.reason, "slot": plain.slot,
                      "card": plain.card, "tile": plain.tile, "mode": plain.mode}
    # a numpy Generator: one draw, whatever the search does
    g1, g2 = np.random.default_rng(5), np.random.default_rng(5)
    run(searcher(policy=policy), rng=g1)
    g2.random()
    assert g1.random() == g2.random()


def test_the_search_reads_public_views_only():
    s = searcher()
    view = make_view()
    leaky = copy.deepcopy(view)
    leaky["players"][1]["hand"] = [8, 9, 10, 11]
    with pytest.raises(SR.SearchLeakError, match="opponent hand"):
        run(s, leaky)
    raw = state_dict(T0)                                         # a seat's state, not a view
    with pytest.raises(SR.SearchLeakError, match="public"):
        run(s, raw)
    other = PV.public_view(state_dict(T0), 1, None, fields=FIELD)
    with pytest.raises(SR.SearchLeakError):
        run(s, other)                                           # team 1's view for team 0
    assert s.summary()["errors"] == 0                           # never swallowed as a fallback


def test_the_true_battle_is_touched_only_through_d0s_legality():
    battle = FakeBattle()
    run(searcher(), battle=battle)
    assert battle.calls == [] and battle.state_reads == 0        # no step, save, load, reset
    assert len(battle.deploy_checks) > 1                         # d0's and the candidates'


def test_summary_counts_outcomes_overrides_and_stages():
    s = searcher(policy=FakePolicy(value=value_by_card({CANNON: 1.0})))
    run(s)
    run(s, make_view(tick=T0 + 10), rng=ScriptRng(0.0))
    out = s.summary()
    assert out["decisions"] == 2 and out["searches"] == 1 and out["overrides"] == 1
    assert out["override_by"] == {f"{HOG}>{CANNON}": 1} and out["vetoes"] == 0
    assert out["outcomes"] == {SR.R_SEARCHED: 1, SR.R_NOT_TRIGGERED: 1}
    assert out["lookahead"] == 2 and out["rebuild"] == {"fake": 2}
    assert {"rebuild", "row", "d0", "candidates", "trials", "values", "total"} <= set(out["ms"])
    assert out["config"] == SR.SearchConfig.named("s_idle").key
    json.dumps(out)


# ---------------------------------------------------------------------------
# the real rebuilder: our play in flight lands during the advance
# ---------------------------------------------------------------------------

@pytest.fixture
def RBT():
    """tests/pipeline/test_rs_rebuild.py (CycleBattle, its catalogue, the push position) and
    tools/public_state (numpy), which an R1 spec needs."""
    importlib.import_module("public_state")
    from tests.pipeline import test_rs_rebuild
    return test_rs_rebuild


def test_pending_plays_land_during_the_advance(RBT):
    rb, mod = RBT.rebuilder()
    view = RBT.view_of(RBT.push_state(), 0, RBT.PUSH_MEM, opp_elixir=(5000, True),
                       sent=RBT.SENT)
    # [wait, Cannon, Fireball, Skeletons, Log]: d0 plays the Cannon; the Log (in flight, in the
    # hand until it runs at 1014) is a top slot but not in the hand at B
    gate = [0.0, 3.0, -1.0, 0.5, 1.0]
    best = {0: (404, 405, 440, 406), 2: (477, 478), 3: (200, 201)}
    policy = FakePolicy(gate=gate, best=best,
                        value=lambda row: row["state"]["players"][0]["elixir_milli"] / 10_000)
    row_fn, log = rows_log()
    battle = FakeBattle()
    s = SR.Searcher(policy, rb, "s_idle", dcfg=DCFG, law=LAW)
    d, rec = s.decide(view, team=0, land_ticks=24, legality=RD.BattleLegality(battle),
                      rng=ScriptRng(0.99), row_fn=row_fn)
    b = mod.battles[-1]
    # the root: the Log stepped in at its exec tick from its slot, then on to B, saved once
    assert RBT.steps(b)[:3] == [([], 14), ([(0, 3, *RBT.LOG_AIM)], 1), ([], 9)]
    assert rec.lookahead and rec.fidelity == 1.0 and log[0][0] == RBT.T0 + 24
    root_row_state = log[0][1]
    assert root_row_state["tick"] == RBT.T0 + 24
    assert root_row_state["players"][0]["hand"] == [RBT.CANNON, RBT.FIREBALL, RBT.SKELS, RBT.LOG]
    assert rec.d0["card"] == RBT.CANNON and rec.d0["tile"] == 404
    # no Log candidate (it has cycled by B); the Cannon's trial plays from its slot at B
    assert all(c.card != RBT.LOG for c in rec.candidates)
    assert [r[0] for r in roles(rec)] == ["P", "W", "T", "T", "S", "S"]
    assert RBT.steps(b)[3:5] == [([(0, 0, *tap(404))], 1), ([], 59)]
    assert ([], 60) in RBT.steps(b)[5:]
    # V = our elixir after the trial (4000 at B, the Log paid): waiting keeps it all
    assert rec.override and rec.candidates[rec.chosen].role == "W"
    assert d.kind == RD.WAIT and d.reason == SR.REASON_WAIT
    # the true battle: d0's and the candidates' legality only
    assert battle.calls == [] and battle.state_reads == 0 and battle.deploy_checks


# ---------------------------------------------------------------------------
# S09: the true-copy trials (s_true, privileged)
# ---------------------------------------------------------------------------

#: Team 0's hand in the true battle (conftest player_dict): the view's (make_view).
TRUE_HAND = (CANNON, FIREBALL, HOG, ICEGOLEM)


class TrueBattle(FakeBattle):
    """The TRUE match battle at T0: a play sent on T from team 0's slot runs on T + delay (in the
    step that reaches that tick) and takes damage[card] off red's left princess tower; a card in
    `refuse` is refused at send, one in `refuse_run` when it runs (step_commands_run reports it).
    save() / load() keep the tick, the queue and the tower; restore=False: load() leaves the tower
    as it is (an engine whose restore is not exact)."""

    def __init__(self, *, delay=D, damage=None, refuse=(), refuse_run=(), restore=True):
        super().__init__()
        self.delay = (int(delay), int(delay))
        self.damage, self.refuse, self.refuse_run = dict(damage or {}), set(refuse), set(refuse_run)
        self.restore = restore
        self.tick, self.queued, self.red_left, self.last_ran = T0, [], PRINCESS_HP, []

    def command_delay_ticks(self):
        return self.delay

    def step(self, cmds=(), n=1):
        cmds = [tuple(c) for c in cmds]
        self.calls.append(Call("step", (cmds, n), {}))
        res = []
        for team, slot, x, y in cmds:
            card = TRUE_HAND[slot]
            code = NOT_ENOUGH if card in self.refuse else 0
            if code == 0:
                self.queued.append((self.tick + self.delay[team], team, card))
            res.append([team, code])
        end = self.tick + int(n)
        self.last_ran = []
        for when, team, card in sorted(q for q in self.queued if q[0] <= end):
            code = NOT_ENOUGH if card in self.refuse_run else 0
            self.last_ran.append((when, team, "deploy", card, code))
            if code == 0 and team == 0:
                self.red_left = max(0, self.red_left - self.damage.get(card, 0))
        self.queued = [q for q in self.queued if q[0] > end]
        self.tick, self.steps = end, self.steps + 1
        return res

    def step_commands_run(self):
        return list(self.last_ran)

    def state(self):
        red = player_dict(1, tower_hp=(KING_HP, self.red_left, PRINCESS_HP))
        return state_dict(self.tick, players=[player_dict(0), red])

    def save(self):
        self.calls.append(Call("save", (), {}))
        return {"tick": self.tick, "steps": self.steps, "queued": list(self.queued),
                "red": self.red_left}

    def load(self, snap):
        self.calls.append(Call("load", (), {}))
        self.tick, self.steps, self.queued = snap["tick"], snap["steps"], list(snap["queued"])
        if self.restore:
            self.red_left = snap["red"]


def true_searcher(**kw):
    return searcher("s_true", privileged=True, **kw)


def test_s_true_tries_each_candidate_on_the_true_battle_and_restores_it():
    battle = TrueBattle(damage={CANNON: PRINCESS_HP // 2})
    rb = FakeRebuilder()
    before = battle.state_json()
    s = true_searcher(rb=rb)
    d, rec = run(s, battle=battle, true_battle=battle)
    # the same candidates as the public search (the public root at B decides their legality)
    assert roles(rec) == [("P", 2, HOG, 309), ("W", -1, -1, -1), ("T", 2, HOG, 345),
                          ("T", 2, HOG, 311), ("S", 0, CANNON, 404), ("S", 0, CANNON, 405),
                          ("S", 1, FIREBALL, 111), ("S", 1, FIREBALL, 122)]
    # one save, then per candidate load + step([its command sent now], D + H), then load back
    assert battle.names() == ["save"] + ["load", "step"] * 8 + ["load"]
    sent = [c.args for c in battle.calls_named("step")]
    assert sent == [([c.command(0)] if c.is_play else [], D + H) for c in rec.candidates]
    assert sent[0] == ([(0, 2, *tap(309))], D + H)               # P from its PRESENT slot
    assert rb.trials == []                                       # no public trial at all
    # restored: the battle is what it was, at T0
    assert battle.state_json() == before and battle.tick == T0 and battle.queued == []
    # scored as the public search: the swing from the true state at t (half a princess tower)
    by = {(c.role, c.card, c.tile): c for c in rec.candidates}
    cannon, p = by[("S", CANNON, 404)], by[("P", HOG, 309)]
    assert cannon.parts[0]["swing"] == pytest.approx(0.5) and p.parts[0]["swing"] == 0.0
    assert rec.outcome == SR.R_SEARCHED and rec.override and rec.worlds == 1
    assert rec.candidates[rec.chosen] is cannon
    assert d.command == (0, 0, *tap(404)) and d.mode == "legacy+search"
    # marked in every record and in the summary
    assert rec.privileged and rec.to_json()["privileged"] is True
    assert s.summary()["privileged"] and s.stats["true_trials"] == 8
    json.dumps(rec.to_json())
    # a public search's record says it is not privileged
    _, pub = run(searcher())
    assert pub.privileged is False and pub.to_json()["privileged"] is False


def test_s_true_plays_refused_at_send_or_when_run_did_not_run():
    battle = TrueBattle(refuse={HOG})                            # P and the T tiles
    before = battle.state_json()
    d, rec = run(true_searcher(), battle=battle, true_battle=battle)
    assert rec.outcome == SR.R_INCUMBENT and not rec.candidates[0].ran
    assert d.command == (0, 2, *tap(309)) and d.mode == "legacy"   # d0 itself
    assert battle.state_json() == before
    battle = TrueBattle(refuse_run={FIREBALL}, damage={CANNON: 100})
    _, rec = run(true_searcher(), battle=battle, true_battle=battle)
    ran = {(c.role, c.card): c.ran for c in rec.candidates}
    assert ran[("S", FIREBALL)] is False and ran[("S", CANNON)] and ran[("P", HOG)]
    assert rec.outcome == SR.R_SEARCHED and battle.red_left == PRINCESS_HP


def test_s_true_a_broken_restore_or_another_delay_is_never_a_fallback():
    battle = TrueBattle(restore=False, damage={CANNON: 100})
    s = true_searcher()
    with pytest.raises(SR.TrueCopyError, match="restore"):
        run(s, battle=battle, true_battle=battle)
    assert s.summary()["errors"] == 0                             # raised, not swallowed
    battle = TrueBattle(delay=10)                                 # the seat says D = 24
    with pytest.raises(SR.TrueCopyError, match="delay"):
        run(true_searcher(), battle=battle, true_battle=battle)
    assert battle.names() == []                                   # nothing stepped
