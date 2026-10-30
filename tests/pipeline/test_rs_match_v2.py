"""tools/rs_match.py and tools/rs_league_eval.py with v2 seats (rs_match module doc, V2 SEATS;
rs_league_eval module doc, V2 PLAYERS).

What is checked: rs_rows/v2 is a schema rs_match plays (short names too); AgentSpec's presses
switch (off by default, left out of to_dict while off so the specs, game keys and identities of
earlier runs do not change); MatchAgent's rules (decide presets press only with the switch on and
trained buttons, no own_pending for landing-time rows, the opponent's elixir less the plays the
engine refused, and the count redone when a counted play turns out refused); run_match naming a
v2 seat's buttons from its deal and checking the status bits; the opponent's elixir leaving out
a play refused when it was sent and one refused when it ran under the delay (also seen late
through perception noise); press commands (taken, refused at send, refused when run, refused
from a seat whose switch is off); a tiny v2 checkpoint (torch) through build_seat, its v2 rows
and decide configs; the league's presses option, its default, idle players, presses_effective
and the game keys. Everything runs on conftest's fake royalesim (its catalogue keeps the cost in
column 1, so the fake gets CATALOGUE_FIELDS = FAKE_FIELDS) and rs_elixir's constant law: 6 elixir
at tick 0 is 168000 fine units, 500 a tick at 1x, an elixir 28000, milli = fine // 28; a Knight
(red's hand slot 0, catalogue 8, token 10) costs 3, 84000. One engine-marked test checks the
count against RoyaleSim's own bar.
"""
from __future__ import annotations

import copy
import importlib
import json
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    DEPLOY_REASONS,
    FakeBattle,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOG = ("Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons", "Log")
OPP = ("Archer", "Arrows", "BabyDragon", "Giant", "Knight", "MiniPekka", "Valkyrie", "Zap")
#: A hero Musketeer (catalogue 5, deal position 5) for Hog; an evolved Knight (position 4) for OPP.
HERO_MUSKETEER = (0, 0, 0, 0, 0, 2, 0, 0)
EVO_KNIGHT = (0, 0, 0, 0, 1, 0, 0, 0)
#: The fake catalogue's columns by name (conftest keeps the cost in column 1).
FAKE_FIELDS = ("name", "elixir", "c2", "c3", "c4", "flying", "c6", "c7", "card_kind")
KNIGHT = 10                                     # red's slot-0 card in player_dict, as a token
NOT_ENOUGH = DEPLOY_REASONS.index("NOT_ENOUGH_ELIXIR")
NOT_READY = DEPLOY_REASONS.index("ABILITY_NOT_READY")
TRAINED = {"schema": "rs_rows/v2", "train_rows": {"press_rows": "trained"}}
#: AgentSpec.to_dict's keys before the presses switch.
OLD_AGENT_KEYS = ["ckpt", "name", "schema", "lead", "noise", "noise_seed", "decide", "visibility",
                  "wait_keep", "spell_aim", "hold_pending", "seed_offset", "allow_drift"]
EMPTY_VIEW = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}


@pytest.fixture
def tools(monkeypatch):
    """rs_rows, rs_eval, rs_match and rs_league_eval as the tools import them, rs_rows' settings
    at their defaults (restored after the test), no engine config from the PC, rs_elixir's
    constant law and the fake catalogue cached for RSAgent."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    ev = importlib.import_module("rs_eval")
    rm = importlib.import_module("rs_match")
    le = importlib.import_module("rs_league_eval")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "TRUST": rr.TrustRule(), "OUT": rr.OUT,
                        "CARD_FLY": {}, "CARD_BUILDING": {}, "PRESSES": "off",
                        "BUTTONS": 2, "HAND_SIZE": 4, "DECISION": rr.DecisionRule(),
                        "ROW_SCHEMA": "rs_rows/legacy", "SF": None, "UNIT_FORMS": None,
                        "SPELL_CARDS": None}.items():
        monkeypatch.setattr(rr, name, value)
    real_law = rr.rs_elixir.default_law
    law = rr.rs_elixir.ElixirLaw.constants()
    monkeypatch.setattr(rr.rs_elixir, "default_law", lambda: law)
    monkeypatch.setattr(ev.RSAgent, "_CAT", [list(r) for r in DEFAULT_CATALOGUE])
    return types.SimpleNamespace(rows=rr, eval=ev, match=rm, league=le, engine=eng, law=law,
                                 real_law=real_law)


def cfg(tools):
    return tools.engine.EngineConfig.defaults()


class RunBattle(FakeBattle):
    """A FakeBattle with step_commands_run(): a command accepted (reason 0) in a step from T,
    under a delay d > 0 of its team, runs on T + d and is reported in the step that covers that
    tick as (tick, team, kind, what, reason): kind "ability" for a slot >= 4 (what: the slot),
    else "deploy" (what: the card in that slot of the hand it was sent against); reason
    run_reason(team, kind, tick), default 0. Without a delay nothing is reported (RoyaleSim)."""

    def __init__(self, *args, run_reason=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_reason = run_reason or (lambda team, kind, tick: 0)
        self.waiting: list[tuple] = []
        self.last_ran: list[tuple] = []

    def step(self, cmds=(), n=1):
        start = self.tick
        cmds = [tuple(c) for c in cmds]
        before = self.state() if cmds else None
        res = super().step(cmds, n)
        for cmd, result in zip(cmds, res):
            team, slot = cmd[0], cmd[1]
            if result[1] != 0 or self.delay[team] <= 0:
                continue
            kind, what = (("ability", slot) if slot >= 4
                          else ("deploy", before["players"][team]["hand"][slot]))
            self.waiting.append((start + self.delay[team], team, kind, what))
        due = sorted(w for w in self.waiting if w[0] < self.tick)
        self.waiting = [w for w in self.waiting if w[0] >= self.tick]
        self.last_ran = [(t, team, kind, what, self.run_reason(team, kind, t))
                         for t, team, kind, what in due]
        return res

    def step_commands_run(self):
        return list(self.last_ran)


def fake_engine(monkeypatch, *, states=None, step_results=None, run_reason=None):
    """conftest's fake royalesim with RunBattles and FAKE_FIELDS, as sys.modules["royalesim"]."""
    mod = make_fake_royalesim(states=states, step_results=step_results)

    def Battle(*args, **kwargs):
        battle = RunBattle(*args, run_reason=run_reason, **{**mod.battle_kwargs, **kwargs})
        mod.battles.append(battle)
        return battle

    mod.Battle = Battle
    mod.CATALOGUE_FIELDS = list(FAKE_FIELDS)
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    return mod


def refuse(team, ticks, reason=NOT_ENOUGH):
    """step_results: `team`'s commands sent in the steps from `ticks` are refused with `reason`."""
    def results(battle, cmds, n):
        start = battle.tick - n
        return [[c[0], reason if c[0] == team and start in ticks else 0] for c in cmds]
    return results


class Plays:
    """Plays hand slot 0 at the (true) ticks in `at`, in its own half."""

    def __init__(self, at=()):
        self.at = tuple(at)

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        if battle.tick not in self.at:
            return None
        return (team, 0, 9000 * sub, (8000 if team == 0 else 24_000) * sub)


class Presser:
    """Sends (team, slot, 0, 0), a press of button slot - 4, at the (true) ticks in `at`."""

    def __init__(self, at=(), slot=4):
        self.at, self.slot = tuple(at), slot

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        return (team, self.slot, 0, 0) if battle.tick in self.at else None


class Views:
    """Keeps a copy of the memory view of every act, by true tick; never plays."""

    def __init__(self):
        self.views = {}

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        self.views[battle.tick] = copy.deepcopy(mem)
        return None


class V2Seat:
    """A seat that says it is v2 (run_match names its buttons and checks the status bits)."""

    v2 = True
    buttons = None

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        return None


def counting(M, engine):
    """A MatchAgent with no checkpoint (no torch): each act keeps, by true tick, its count of
    the opponent's bar (opp_elixir of the view it is handed), the opponent's plays in that view
    and the opponent's true bar; it never plays."""
    class Counting(M.MatchAgent):
        def __init__(self):
            self.engine = engine
            self.counts, self.views, self.bars = {}, {}, {}

        def act(self, battle, state, team, mem, tick, sub, royalesim):
            self.counts[battle.tick] = self.opp_elixir(team, mem, tick, royalesim)
            self.views[battle.tick] = [tuple(p) for p in mem["last"][1 - team]]
            self.bars[battle.tick] = int(state["players"][1 - team]["elixir_milli"])
            return None
    return Counting()


# ---------------------------------------------------------------------------
# the spec: schema names, the presses switch, configure
# ---------------------------------------------------------------------------

def test_v2_is_a_row_schema_and_the_presses_switch_keeps_old_specs(tools):
    M = tools.match
    assert {"rs_rows/legacy", "rs_rows/v2"} <= M.ROW_SCHEMAS
    assert M.AgentSpec("a.pt", schema="v2").schema == "rs_rows/v2"
    assert M.AgentSpec("a.pt", schema=" legacy ").schema == "rs_rows/legacy"
    assert M.AgentSpec("a.pt", schema="rs_rows/v9").schema == "rs_rows/v9"      # configure checks
    plain = M.AgentSpec("a.pt", name="x")
    on = M.AgentSpec("a.pt", name="x", presses=True)
    assert plain.presses is False and list(plain.to_dict()) == OLD_AGENT_KEYS
    assert list(on.to_dict()) == OLD_AGENT_KEYS + ["presses"] and on.to_dict()["presses"] is True
    assert M.AgentSpec.from_dict(json.loads(json.dumps(on.to_dict()))) == on != plain
    assert M.AgentSpec.from_dict(plain.to_dict()) == plain
    assert M.AgentSpec("a.pt", presses=np.bool_(True)).presses is True
    for bad in ("yes", 1, None):
        with pytest.raises(ValueError, match="presses"):
            M.AgentSpec("a.pt", presses=bad)
    spec = M.MatchSpec(decks=(HOG, OPP), agents=(on, M.AgentSpec(name="y")))
    assert spec.swapped().agents == (M.AgentSpec(name="y"), on)
    assert M.MatchSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert "presses" not in spec.to_dict()["agents"][1]

    c = cfg(tools)
    seat = types.SimpleNamespace(meta={"schema": "rs_rows/v2"})
    M.configure(seat, M.AgentSpec("x.pt", schema="v2", presses=True), c)
    assert seat.presses is True and M.seat_schema(seat) == "rs_rows/v2"
    assert M.seat_schema(types.SimpleNamespace(meta={"schema": " v2 "})) == "rs_rows/v2"
    M.configure(types.SimpleNamespace(meta={"schema": "v2"}), M.AgentSpec("x.pt"), c)
    assert M.configure(types.SimpleNamespace(meta={}), M.AgentSpec("x.pt"), c).presses is False
    with pytest.raises(ValueError, match="agent spec says"):
        M.configure(types.SimpleNamespace(meta={"schema": "rs_rows/v2"}),
                    M.AgentSpec("x.pt", schema="legacy"), c)
    with pytest.raises(ValueError, match="rs_rows/v9"):
        M.configure(types.SimpleNamespace(meta={"schema": "rs_rows/v9"}), M.AgentSpec("x.pt"), c)


# ---------------------------------------------------------------------------
# MatchAgent's rules (no checkpoint, no torch)
# ---------------------------------------------------------------------------

def test_match_agent_presses_only_when_on_and_trained(tools):
    M = tools.match
    D = M.rs_decide.DecideConfig
    agent = M.MatchAgent.__new__(M.MatchAgent)
    agent.wait_bias = 1.25
    want = {"legacy": lambda **kw: D.legacy(1.25, dt=10, **kw),
            "live": lambda **kw: D.live(1.25, guard_cards=[0], dt=10, **kw),   # Cannon: id 0
            "sample": lambda **kw: D.sampling(1.25, dt=10, **kw)}
    for meta, presses, on in ((None, False, False), (TRAINED, False, False),
                              (TRAINED, True, True), ({"schema": "rs_rows/v2"}, True, False),
                              ({"train_rows": {"press_rows": "trained"}}, True, False)):
        agent.meta, agent.presses = meta, presses
        assert agent.presses_on() is on, meta
        for mode, make in want.items():
            agent.decide_mode = mode
            got = agent.decide_config()
            assert got == (make(presses=True) if on else make()) and got.presses is on, mode


def test_match_agent_pending_follows_the_rows_it_trained_on(tools):
    M = tools.match
    agent = M.MatchAgent.__new__(M.MatchAgent)
    me = player_dict(0, pending=[[0, 2, 0, 0, 12, 4], [1, 5, 0, 0, 3, 1], [0, 5, 0, 0, 5, 4]],
                     pending_cost=8)
    state = state_dict(100, players=[me, player_dict(1)])
    agent.meta = {"obs_delay": {"ticks": 24, "jitter": 3}}
    assert agent.own_pending(state, 0) == [(7, 5), (4, 12)]       # RSAgent.own_pending's rule
    for meta in ({}, None, {"obs_delay": {"ticks": 0}}):
        agent.meta = meta
        assert agent.own_pending(state, 0) == [], meta              # landing-time rows: none


def test_match_agent_counts_only_the_plays_that_ran(tools):
    """Red's Knights ran (or not) at 100 and 150; blue's Cannon (token 2, 3 elixir) at 100."""
    M = tools.match
    rs = types.SimpleNamespace(CATALOGUE_FIELDS=FAKE_FIELDS)
    view = {"seen": {0: {2}, 1: {KNIGHT}},
            "last": {0: [(2, 100)], 1: [(KNIGHT, 100), (KNIGHT, 150)]}}
    keep = copy.deepcopy(view)

    def agent(unran):
        a = counting(M, cfg(tools))
        a.unran = unran
        return a

    assert agent(None).opp_elixir(0, view, 160, rs) == (2857, True)      # 80000 // 28: both paid
    assert agent(Counter()).opp_elixir(0, view, 160, rs) == (2857, True)
    gone = Counter({(1, KNIGHT, 150): 1, (0, 2, 100): 1})               # blue's entry: not red's
    assert agent(gone).opp_elixir(0, view, 160, rs) == (5857, True)      # 164000 // 28
    assert agent(Counter({(1, KNIGHT, 150): 2})).opp_elixir(0, view, 160, rs) == (5857, True)
    assert agent(gone).opp_elixir(1, view, 160, rs) == (8857, True)      # red's count: 248000
    assert agent(None).opp_elixir(1, view, 160, rs) == (5857, True)
    assert view == keep                                                  # the view is not changed
    # Mem lag 0: the play was counted before its refusal was known; the count starts again.
    late = agent(Counter())
    assert late.opp_elixir(0, view, 160, rs) == (2857, True)
    late.unran[(1, KNIGHT, 150)] += 1
    assert late.opp_elixir(0, view, 170, rs) == (6035, True)             # 169000 // 28


# ---------------------------------------------------------------------------
# run_match: buttons, status bits, the counted elixir, presses
# ---------------------------------------------------------------------------

def test_v2_seats_get_the_buttons_of_their_deal(tools, monkeypatch):
    M = tools.match
    mod = fake_engine(monkeypatch)
    spec = M.MatchSpec(decks=(HOG, OPP), forms=(HERO_MUSKETEER, EVO_KNIGHT), engine=cfg(tools),
                       max_ticks=100)
    a, b, plain = V2Seat(), V2Seat(), Plays()
    r = M.run_match(spec, seats=[a, b])
    assert (a.buttons, b.buttons) == ([5], [])                  # an evolution is no button
    assert (r.info[0]["buttons"], r.info[1]["buttons"], r.info[0]["presses"]) == ([5], [], False)
    M.run_match(spec.swapped(), seats=[b, a])                   # a keeps Hog 2.6, on team 1
    assert (a.buttons, b.buttons) == ([5], [])
    M.run_match(spec.replace(forms=None), seats=[a, plain])
    assert a.buttons == [] and not hasattr(plain, "buttons")
    mod.STATUS_BITS = ["burrowing", "cloaked", "inside"]
    with pytest.raises(M.rs_engine.EngineError, match="STATUS_BITS"):
        M.run_match(spec, seats=[a, None])
    assert M.run_match(spec, seats=[plain, None]).decisions == (1, 0)   # legacy: no bits read


def test_the_count_leaves_out_a_play_refused_when_sent(tools, monkeypatch):
    """Delay 0, mem lag 1: red's Knight at 100 runs; the one at 150 is refused at send. The
    memory still lists it (opp_last shows it, as always); blue's count does not pay it."""
    M = tools.match
    mod = fake_engine(monkeypatch, step_results=refuse(1, (150,)))
    spec = M.MatchSpec(decks=(HOG, OPP), engine=cfg(tools), delay=0, max_ticks=200,
                       agents=(M.AgentSpec(name="blue"), M.AgentSpec(name="red")))
    blue = counting(M, cfg(tools))
    r = M.run_match(spec, seats=[blue, Plays((100, 150))])
    assert (r.plays, r.refused, r.decisions) == ((0, 2), (0, 1), (11, 11))
    assert [e[6] for e in r.log] == [0, NOT_ENOUGH]
    assert blue.unran == Counter({(1, KNIGHT, 150): 1})
    assert blue.views[150] == [(KNIGHT, 100)]
    assert blue.views[160] == [(KNIGHT, 100), (KNIGHT, 150)]
    assert {t: blue.counts[t] for t in (90, 100, 110, 150, 160, 190)} == {
        90: (7607, True),       # 213000 // 28
        100: (7785, True),      # 218000: the Knight run at 100 is not on the board yet
        110: (4964, True),      # 168000 + 55000 - 84000
        150: (5678, True),      # 159000
        160: (5857, True),      # 164000: the refused Knight is not paid (2857 if it were)
        190: (6392, True)}      # 179000
    other = counting(M, cfg(tools))                     # outside run_match: it would be paid
    assert other.opp_elixir(0, {"last": {1: blue.views[160]}}, 160, mod) == (2857, True)


@pytest.mark.parametrize("noise", [None, "late"])
def test_the_count_leaves_out_a_play_refused_when_it_ran(tools, monkeypatch, noise):
    """Delay 21: red's Knights sent at 100 and 150 run at 121 and 171 (reported in the steps
    from 120 and 170); the engine refuses the second when it runs. With noise blue's screen
    sees red's plays 4 ticks late, and that is the entry left out."""
    M = tools.match
    fake_engine(monkeypatch,
                run_reason=lambda team, kind, tick: NOT_ENOUGH if (team, tick) == (1, 171) else 0)
    late = {"opponent_plays": {"late_ticks": 4}} if noise else None
    spec = M.MatchSpec(decks=(HOG, OPP), engine=cfg(tools), delay=21, max_ticks=200,
                       agents=(M.AgentSpec(name="blue", noise=late), M.AgentSpec(name="red")))
    blue = counting(M, cfg(tools))
    r = M.run_match(spec, seats=[blue, Plays((100, 150))])
    shift = 4 if noise else 0
    first, second = (KNIGHT, 121 + shift), (KNIGHT, 171 + shift)
    assert (r.plays, r.refused, r.delay, r.decisions) == ((0, 2), (0, 1), (21, 21), (11, 11))
    assert [e[6] for e in r.log] == [0, 0]                      # both taken when sent
    assert blue.unran == Counter({(1, *second): 1})
    assert blue.views[170] == [first] and blue.views[180] == [first, second]
    assert {t: blue.counts[t] for t in (120, 130, 170, 180, 190)} == {
        120: (8142, True),      # 228000 // 28
        130: (5321, True),      # 149000: the first Knight paid
        170: (6035, True),      # 169000
        180: (6214, True),      # 174000: the second never ran (3214 if it were paid)
        190: (6392, True)}      # 179000


def full_blue(battle):
    """Blue sits at 10 elixir with button 0 (hero Musketeer, card 5, 2 elixir) available; red at
    5. The game ends at 200, blue the winner."""
    over = battle.tick >= 200
    players = [player_dict(0, elixir_milli=10_000, abilities=[[1, 0, 2, 5, 0]],
                           crowns=int(over)),
               player_dict(1, elixir_milli=5_000)]
    return state_dict(battle.tick, players=players, game_over=over, winner=0 if over else -1)


@pytest.mark.parametrize("case", ["taken", "refused_at_send", "refused_when_run"])
def test_a_seat_that_may_press_presses(tools, monkeypatch, case):
    """Blue presses button 0 at 100. Taken (delay 0): its 2 elixir end the full bar's leak in
    the step from 100 (190 full ticks); refused at send: nothing spent (200); taken at send and
    refused when it runs (delay 21, the step from 120): spent at 121 as a play would be (191)."""
    M = tools.match
    delay = 21 if case == "refused_when_run" else 0
    results = None
    if case == "refused_at_send":
        results = lambda battle, cmds, n: [[c[0], NOT_READY if c[1] >= 4 else 0] for c in cmds]  # noqa: E731
    mod = fake_engine(monkeypatch, states=full_blue, step_results=results,
                      run_reason=lambda team, kind, tick: NOT_READY if kind == "ability" else 0)
    spec = M.MatchSpec(decks=(HOG, OPP), forms=(HERO_MUSKETEER, (0,) * 8), engine=cfg(tools),
                       delay=delay, agents=(M.AgentSpec(name="blue", presses=True),
                                            M.AgentSpec(name="red")))
    blue, red = Presser((100,)), Views()
    r = M.run_match(spec, seats=[blue, red], law=M.rs_elixir.ElixirLaw.constants())
    assert (r.presses, r.plays, r.refused, r.decisions) == ((1, 0), (0, 0), (0, 0), (11, 11))
    assert r.press_refused == ((0, 0) if case == "taken" else (1, 0))
    reason = NOT_READY if case == "refused_at_send" else 0
    assert r.log == ((100, 0, 5, 4, 0, 0, reason),)            # the button's card, its slot
    ticks = {"taken": 190, "refused_at_send": 200, "refused_when_run": 191}[case]
    assert r.full_ticks == (ticks, 0) and r.leak[0] == pytest.approx(ticks * 500 / 28_000)
    assert [c.args[0] for c in mod.battles[-1].calls_named("step") if c.args[0]] == [[(0, 4, 0, 0)]]
    assert not hasattr(blue, "sent")                            # nothing in flight to look ahead
    assert red.views[190] == EMPTY_VIEW                         # a press is no card play
    assert r.winner == 0
    back = M.MatchResult.from_dict(json.loads(json.dumps(r.to_dict())))
    assert (back.presses, back.press_refused) == (r.presses, r.press_refused)


def test_a_press_needs_the_switch(tools, monkeypatch):
    M = tools.match
    fake_engine(monkeypatch, states=full_blue)
    spec = M.MatchSpec(decks=(HOG, OPP), engine=cfg(tools), max_ticks=120)
    with pytest.raises(ValueError, match=r"hand slot 0-3$"):
        M.run_match(spec, seats=[Presser((100,)), None])
    on = spec.replace(agents=(M.AgentSpec(name="blue", presses=True), M.AgentSpec()))
    with pytest.raises(ValueError, match=r"hand slot 0-3 or a button slot 4-5$"):
        M.run_match(on, seats=[Presser((100,), slot=6), None])
    r = M.run_match(on, seats=[Presser((100,)), None])
    assert (r.presses, r.press_refused, r.plays) == ((1, 0), (0, 0), (0, 0))
    old = M.MatchResult.from_dict({k: v for k, v in r.to_dict().items()
                                   if k not in ("presses", "press_refused")})
    assert (old.presses, old.press_refused) == ((0, 0), (0, 0))      # records before the switch


# ---------------------------------------------------------------------------
# a tiny v2 checkpoint through build_seat (torch)
# ---------------------------------------------------------------------------

def meta_for(**extra) -> dict:
    """A checkpoint meta as sim/train_tokens writes it for conftest's fake catalogue (as in
    tests/pipeline/test_model_tokens_v2.py)."""
    names = [str(r[0]) for r in DEFAULT_CATALOGUE]
    meta = {"n_cards": len(names) + 2, "n_units": len(DEFAULT_UNIT_TYPES) + 2,
            "live_compat": False, "cards": {c: i + 2 for i, c in enumerate(names)},
            "units": {u: i + 2 for i, u in enumerate(DEFAULT_UNIT_TYPES)},
            "card_table": "cards-160402017-20261019.1", "d": 16, "layers": 1, "heads": 2}
    meta.update(extra)
    return meta


@pytest.mark.parametrize("case", ["trained", "untrained", "off"])
def test_a_v2_checkpoint_plays_its_v2_rows(tools, monkeypatch, tmp_path, case):
    """build_seat on a tiny v2 checkpoint: the MatchAgent gets the buttons of its deal ([5]),
    v2 rows with its pending HogRider (card 2, 12 ticks to land) when its rows were
    decision-time ones (obs_delay 24), the opponent's elixir without red's Knight refused at 150,
    and a decide config that presses only with the switch on and trained buttons. decide is a
    spy that waits, so nothing depends on the untrained weights."""
    torch = pytest.importorskip("torch")
    from sim import model_tokens as mt
    M = tools.match
    rc = importlib.import_module("rs_checkpoint")
    monkeypatch.delenv(rc.ALLOW_DRIFT_ENV, raising=False)
    monkeypatch.setattr(rc, "_VOCABS", {})
    redrive = sys.modules.get("royalesim_redrive")
    if redrive is not None and hasattr(redrive, "_ENGINE"):
        monkeypatch.setattr(redrive, "_ENGINE", {})
    mine = player_dict(0, pending=[[0, 2, 0, 0, 12, 4]], pending_cost=4)
    fake_engine(monkeypatch, step_results=refuse(1, (150,)),
                states=lambda b: state_dict(b.tick, players=[mine, player_dict(1)],
                                            game_over=b.tick >= 200))
    extra = {"schema": "rs_rows/v2"}
    if case != "untrained":
        extra.update(obs_delay={"ticks": 24, "jitter": 3}, train_rows={"press_rows": "trained"})
    meta = meta_for(**extra)
    torch.manual_seed(0)
    model = mt.policy_from_meta(meta)
    ckpt = tmp_path / "v2_hog" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "meta": meta, "epoch": 0}, ckpt)
    calls = {}

    def spy(policy, row, state, team, legality, dcfg=None, **kw):
        calls[int(kw["tick"])] = (team, row, dcfg)
        return M.rs_decide.Decision(team=team, tick=kw["tick"])

    monkeypatch.setattr(M.rs_decide, "decide", spy)
    made = []

    def factory(agent, team, spec, cfg_):
        if team == 1:
            return Plays((100, 150))
        made.append(M.build_seat(agent, team, spec, cfg_))
        return made[-1]

    spec = M.MatchSpec(decks=(HOG, OPP), forms=(HERO_MUSKETEER, (0,) * 8), engine=cfg(tools),
                       delay=0, agents=(M.AgentSpec(str(ckpt), name="v2", presses=case != "off"),
                                        M.AgentSpec(name="red")))
    threads = torch.get_num_threads()
    try:
        r = M.run_match(spec, seat_factory=factory)
    finally:
        torch.set_num_threads(threads)                  # RSAgent sets 1 for the process
    (seat,) = made
    pressing = case == "trained"
    assert isinstance(seat, M.MatchAgent) and seat.v2 and seat.buttons == [5]
    assert seat.presses is (case != "off") and seat.presses_on() is pressing
    assert seat.unran == Counter({(1, KNIGHT, 150): 1})
    assert sorted(calls) == list(range(90, 200, 10))
    pending = [0, 0, 0, 0] if case == "untrained" else [4, 0, 0, 0]
    for tick, (team, row, dcfg) in calls.items():
        assert team == 0 and M.rs_schema.V2.check_row(row, buttons=2) == [], tick
        assert row["ability_card"].tolist() == [7, 0], tick
        assert row["own_pending"].tolist() == pending, tick
        assert dcfg.presses is pressing and dcfg.name == "live", tick
    land = 0 if case == "untrained" else np.float16(12 / 40)
    assert calls[100][1]["own_pending_land"].tolist() == [land, 0, 0, 0]
    for tick, milli in ((90, 7607), (100, 7785), (110, 4964), (160, 5857)):
        assert calls[tick][1]["scalars"][5] == np.float32(milli / 10_000), tick
        assert calls[tick][1]["scalars"][6] == 1.0, tick
    assert (r.refused, r.decisions) == ((0, 1), (11, 11))
    assert (r.info[0]["schema"], r.info[0]["buttons"], r.info[0]["presses"]) == \
        ("rs_rows/v2", [5], pressing)


# ---------------------------------------------------------------------------
# the league
# ---------------------------------------------------------------------------

def test_league_players_and_the_presses_switch(tools):
    M, LE = tools.match, tools.league
    assert LE.parse_player("v2=runs/tokens/v2_hog/best.pt,presses=on") == \
        M.AgentSpec("runs/tokens/v2_hog/best.pt", name="v2", presses=True)
    assert LE.parse_player("a.pt", presses=True).presses is True
    assert LE.parse_player("a.pt,presses=off", presses=True).presses is False
    assert LE.parse_player("a.pt").presses is False
    assert LE.parse_player("a.pt", presses=None).presses is False
    with pytest.raises(ValueError, match="true or"):
        LE.parse_player("a.pt,presses=maybe")
    roles = LE.prepare_players(["c=c.pt"], "h=h.pt,presses=off", ["floor=idle"], presses=True)
    assert [roles.players[n].presses for n in ("c", "h", "floor")] == [True, False, False]
    off, on = M.AgentSpec("c.pt", name="c"), M.AgentSpec("c.pt", name="c", presses=True)
    assert "presses" not in LE.identity(off, None)
    assert LE.identity_digest(LE.identity(off, None)) != LE.identity_digest(LE.identity(on, None))
    assert LE.presses_effective(on, TRAINED) is on
    assert LE.presses_effective(on, {"schema": "rs_rows/v2"}) == off
    assert LE.presses_effective(on, {}) == off and LE.presses_effective(off, TRAINED) is off
    run = ["run", "--candidate", "c.pt", "--champion", "h.pt"]
    assert LE.build_parser().parse_args(run + ["--presses"]).presses is True
    assert LE.build_parser().parse_args(run).presses is False
    # A comparison's games: the same games for either switch, other keys when it is on.
    pair = M.DeckPair("synthetic-0004", "val", (HOG, OPP), ((0,) * 8, (0,) * 8), (HOG, OPP))
    proto = LE.Protocol(noise=None, engine=cfg(tools))
    champ = M.AgentSpec("h.pt", name="h")
    t_off = LE.comparison_tasks([pair], proto, off, champ, games=4, decks="both")
    t_on = LE.comparison_tasks([pair], proto, on, champ, games=4, decks="both")
    assert len(t_off) == len(t_on) == 4
    assert [t.spec.agents for t in t_on] == [(on, champ), (champ, on), (champ, on), (on, champ)]
    for a, b in zip(t_off, t_on):
        assert a.spec.replace(agents=tuple(on if x == off else x for x in a.spec.agents)) == b.spec
    assert not {t.key for t in t_off} & {t.key for t in t_on}


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

@pytest.mark.engine
def test_engine_the_count_is_the_engines_bar(tools, monkeypatch):
    """RoyaleSim at a 24-tick delay: red plays its slot-0 card every 30 ticks from 100 (some
    may be refused); at every decision blue's count of red's bar is the bar in the engine's
    state (plain decks, so nothing rs_elixir does not model)."""
    M = tools.match
    monkeypatch.setattr(tools.eval.RSAgent, "_CAT", None)        # the engine's own catalogue
    monkeypatch.setattr(tools.rows.rs_elixir, "default_law", tools.real_law)
    spec = M.MatchSpec(decks=(HOG, OPP), engine=cfg(tools), delay=24, max_ticks=600,
                       agents=(M.AgentSpec(name="blue"), M.AgentSpec(name="red")))
    blue = counting(M, cfg(tools))
    r = M.run_match(spec, seats=[blue, Plays(range(100, 600, 30))])
    assert r.plays[1] == len(range(100, 600, 30)) and len(blue.counts) >= 40
    assert {t: c[0] for t, c in blue.counts.items()} == blue.bars
    assert all(exact for _, exact in blue.counts.values())
