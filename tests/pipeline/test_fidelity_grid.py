"""E03: the fidelity grid (tools/rs_fidelity_grid.py) and the robust re-drive
(tools/royalesim_redrive.py --sample / --payloads / --tick-semantics, per-match errors and
descriptive columns).

Re-drive, on conftest's fake royalesim (no engine, no corpus). The fixture match is conftest's
payload (plays at 100 Musketeer, 160 Knight, 240 HogRider, 300 Arrows, 420 Skeletons; the team
at low y plays blue), with hands that hold every played card (as in test_presses), so the
commands are SENT below. Worked by hand:

* golden: --tick-semantics exec (the default) is the re-drive before E03: the same commands at
  the same ticks, the same steps (100, 1, 59, 1, 79, 1, 59, 1, 119, 1, then 199 to tick 620 =
  420 + 200 and 6000 more, end 6620), and the same values in every column the report read; the
  new columns only add game_mode, split, tower_card, tower_card_level, levels, tower_level 11
  (the mode of 8 + 4 level-11 cards) and refusals.
* tap: set_command_delay_ticks(21, 21) comes before reset, then the very same reset and steps
  as exec. DelayBattle runs a command accepted on tick T at T + delay and reports it in
  step_commands_run(): the Knight sent at 160 runs at 181 (inside the step 161 -> 240) and is
  refused NOT_ENOUGH_ELIXIR there ("run"); the HogRider refused CARD_PENDING when sent at 240
  counts as before ("send"). With a 200-tick delay the HogRider sent at 240 is still pending at
  420, so the Skeletons missing from the hand count "hand_pending"; with 21 it ran at 261, so
  "hand". Presses accepted when sent count when they run: the Musketeer's (240 -> 261) and the
  Ice Golem's (360 -> 381) ok, the Golden Knight's (330 -> 351) refused NO_HERO.
* a match that raises is a row: error "exception <Type>: <message>", "where" file:line.
* --sample: the N tags with the smallest sha1("<seed>:<tag>")[:16] among the modes and split
  asked; the expected list for the 400-tag index below was computed from that rule.

Grid, pure (numpy for tools/league/stats.py): cells and commands, Wilson intervals (k = 3 of
n = 4: 75.0% [30.1, 95.4]; 2 of 4: 50.0% [15.0, 85.0]; 4 of 5: 80.0% [37.6, 96.4]; 2 of 5:
40.0% [11.8, 76.9]; 3 of 3: 100.0% [43.8, 100.0]; 2 of 3: 66.7% [20.8, 93.9]), refusals per 100
plays, strata, the paired McNemar (1 vs 1 discordant: p = min(1, 2 * 3 / 4) = 1; 6 vs 0:
2 / 64 = 0.03125), the report text, and main with a fake cell runner (a failed cell, then
--reuse). The `engine`-marked tests run the fixture through the real royalesim under both tick
semantics, and the whole grid on two fixture matches in subprocesses.

U10 (candidate cells, shared payloads), worked by hand:
* Candidate cells come after the main cells and before the 0.1.22 ones, named
  <label>-<table>-<ticks>-presses-<presses>; a table the checkout does not know keeps its arm
  name. The first cell stays the base, so a candidate cell's "vs base" is its McNemar against
  the old engine.
* The guard (docs/SEASON_UPDATES.md section 9) against a reference of 100 matches that all
  agree, 10 plays each: 1 match lost is -1 pp, p = min(1, 2 * 1/2) = 1, pass; 2 lost is -2 pp
  but p = 2 * 1/4 = 0.5, pass; 10 lost is -10 pp, p = 2 / 2**10 = 0.00195, BLOCKED. Of 1000
  matches, 15 lost is p = 2 / 2**15 but only -1.5 pp, pass; 20 lost is -2 pp with
  p = 2 / 2**20, BLOCKED. Refusals: 2 per match is 200 / 1000 plays = +20.00 per 100,
  BLOCKED; 1 in 10 matches is +1.00, not more than 1.0, pass. Failed matches: the limit is
  0 + 1% of 100 = 1, so 2 failed is BLOCKED and 1 is not. A cell that did not run, or whose
  reference did not run, is BLOCKED.
* The base and tap cells above as base and candidate (exec, presses off), 7 matches: vs the
  reference m1 1 -> 0, m3 0 -> 1, so +0.0 pp (p 1, n 3); refusals 1.25 - 4.00 = -2.75; failed
  7 - 5 = 2 against 7 - 4 = 3 plus 0.07, limit 3.07: pass.
* A shared sample is picked once and never rewritten; a later run with other modes, or after
  its payload file changed, is refused.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEPLOY_REASONS,
    SUBTILE_PER_MILLITILE,
    FakeBattle,
    catalogue_row,
    il_payload,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = SUBTILE_PER_MILLITILE

#: Every card a side plays in the payload is in its hand (as in test_presses).
HOG_HAND, HOG_NEXT = [5, 2, 6, 0], 1          # Musketeer, HogRider, Skeletons, Cannon
OPP_HAND, OPP_NEXT = [8, 11, 9, 10], 12       # Knight, Arrows, Archer, Giant
#: (tick, commands) of each step that carried any: hand slots of HOG_HAND / OPP_HAND.
SENT = [
    (100, [(0, 0, 3500 * SUB, 7500 * SUB)]),      # Musketeer, blue slot 0
    (160, [(1, 0, 14500 * SUB, 24500 * SUB)]),    # Knight, red slot 0
    (240, [(0, 1, 3500 * SUB, 14500 * SUB)]),     # HogRider, blue slot 1
    (300, [(1, 1, 3500 * SUB, 8000 * SUB)]),      # Arrows, red slot 1
    (420, [(0, 2, 9000 * SUB, 5000 * SUB)]),      # Skeletons, blue slot 2
]
STEP_TICKS = [100, 1, 59, 1, 79, 1, 59, 1, 119, 1, 199, 6000]
#: The descriptive columns of il_payload() (split_of("synthetic-0001") is "train"; the
#: opponent's levels are conftest's 12, 11, 13, 11, 12, 11, 14, 11).
INFO = {"game_mode": "Ladder", "split": "train",
        "tower_card": {"team": "tower-princess", "opponent": "tower-princess"},
        "tower_card_level": {"team": 11, "opponent": 11},
        "levels": {"team": {"11": 8}, "opponent": {"11": 4, "12": 2, "13": 1, "14": 1}}}
#: The columns the report read before E03, with their values for the fixture: the fake's
#: state has no crowns and no winner ("draw"), the recording is a 1-0 victory.
LEGACY = {"tag": "synthetic-0001", "ok": True, "error": "", "forced": {}, "plays": 5,
          "engine_crowns": {"team": 0, "opponent": 0},
          "recorded_crowns": {"team": 1, "opponent": 0},
          "winner_agree": False, "crowns_agree": False, "end": 6620}
EXEC_OUT = {**LEGACY, **INFO, "tower_level": 11, "refusals": []}

NOT_ENOUGH = DEPLOY_REASONS.index("NOT_ENOUGH_ELIXIR")
CARD_PENDING = DEPLOY_REASONS.index("CARD_PENDING")
NO_HERO = DEPLOY_REASONS.index("NO_HERO")
#: test_presses' press match: a Golden Knight (catalogue id 16) in the opponent's deck, hero
#: Musketeer and hero Ice Golem in the team's; presses at 240, 330 and 360.
CAT = [list(r) for r in DEFAULT_CATALOGUE] + [catalogue_row("GoldenKnight", 4, "TROOP")]
OPP_DECK = ("knight", "archers", "giant", "arrows", "golden-knight", "mini-pekka", "baby-dragon",
            "zap")
TEAM_FORMS = {"skeletons": 1, "musketeer": 2, "ice-golem": 2}
PRESS_EVENTS = ((240, "team", "musketeer"), (330, "opponent", "golden-knight"),
                (360, "team", "ice-golem"))
PLAIN = {"team_forms": {}, "opponent_forms": {}, "opponent_levels": 11}


class DelayBattle(FakeBattle):
    """A FakeBattle with a command delay: a command accepted (reason 0) in a step that starts
    on tick T runs on T + delay of its team, and step_commands_run() gives the rows of the
    commands that ran in the last step, (tick, team, kind, what, reason): kind "ability" for a
    slot >= 4 (what: the slot), else "deploy" (what: the card in that slot of the hand the
    command was sent against). run_reason(team, kind, what) gives each reason (default 0).
    pending_commands(team) lists the commands accepted and not run yet."""

    def __init__(self, *args, run_reason=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_reason = run_reason or (lambda team, kind, what: 0)
        self.waiting: list[tuple[int, int, str, int]] = []
        self.last_ran: list[tuple] = []

    def step(self, cmds=(), n=1):
        start = self.tick
        cmds = [tuple(c) for c in cmds]
        before = self.state() if cmds else None
        res = super().step(cmds, n)
        for cmd, result in zip(cmds, res):
            if result[1] != 0:
                continue
            team, slot = cmd[0], cmd[1]
            kind, what = (("ability", slot) if slot >= 4
                          else ("deploy", before["players"][team]["hand"][slot]))
            self.waiting.append((start + self.delay[team], team, kind, what))
            self.pending[team].append((kind, what))
        due = sorted(w for w in self.waiting if w[0] < self.tick)
        self.waiting = [w for w in self.waiting if w[0] >= self.tick]
        for _, team, kind, what in due:
            self.pending[team].remove((kind, what))
        self.last_ran = [(t, team, kind, what, self.run_reason(team, kind, what))
                         for t, team, kind, what in due]
        return res

    def step_commands_run(self):
        return list(self.last_ran)


class FormsRefused(FakeBattle):
    """reset() refuses the forms, as royalesim does for a hero the card table lacks."""

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        raise RuntimeError("forms: IceWizard_hero is not in cards-15535.1")


def states(team_hand=HOG_HAND):
    """FakeBattle states: hands holding the played cards (team_hand for blue)."""
    def make(battle):
        return state_dict(battle.tick, players=[
            player_dict(0, hand=team_hand, next_card=HOG_NEXT),
            player_dict(1, hand=OPP_HAND, next_card=OPP_NEXT)])
    return make


def sent(battle):
    """(tick, commands) of every step of a FakeBattle that carried commands."""
    out, tick = [], int((battle.reset_kwargs or {}).get("start_tick") or 0)
    for call in battle.calls:
        if call.name == "step":
            cmds, n = call.args
            if cmds:
                out.append((tick, list(cmds)))
            tick += int(n)
    return out


def step_ticks(battle):
    return [call.args[1] for call in battle.calls_named("step")]


@pytest.fixture
def rd(monkeypatch):
    """tools/royalesim_redrive.py as the tools import it, its engine cache empty and no
    engine config or card-map setting from the PC."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    mod = importlib.import_module("royalesim_redrive")
    eng = mod.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), mod.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    monkeypatch.setattr(mod, "_ENGINE", {})
    return mod


@pytest.fixture
def engine(monkeypatch):
    """install(cls=FakeBattle, **battle_kwargs): a fake royalesim whose Battle builds `cls`."""
    def install(cls=FakeBattle, **kwargs):
        mod = make_fake_royalesim(**kwargs)

        def Battle(*args, **kw):
            battle = cls(*args, **{**mod.battle_kwargs, **kw})
            mod.battles.append(battle)
            return battle
        mod.Battle = Battle
        monkeypatch.setitem(sys.modules, "royalesim", mod)
        return mod
    return install


@pytest.fixture
def grid(rd):
    pytest.importorskip("numpy")
    mod = importlib.import_module("rs_fidelity_grid")
    assert mod.RD is rd
    return mod


def cfg(rd):
    return rd.rs_engine.EngineConfig.defaults()


def _function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        pytest.skip(f"{path.name} no longer defines {name}")
    ns = dict(namespace)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


# ---------------------------------------------------------------------------
# re-drive: settings, sample, names (pure)
# ---------------------------------------------------------------------------

def test_split_rule_and_modes_are_the_row_builders(rd):
    legacy = _function(TOOLS / "rs_rows.py", "split_of", {"hashlib": hashlib})
    tags = [f"#{i:08X}" for i in range(2000)] + ["synthetic-0001", "", "åäö"]
    assert [rd.split_of(t) for t in tags] == [legacy(t) for t in tags]
    assert (rd.split_of("synthetic-0001"), rd.split_of("")) == ("train", "val")
    tree = ast.parse((TOOLS / "rs_rows.py").read_text(encoding="utf-8"))
    node = next((n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "MODES" for t in n.targets)), None)
    if node is None:
        pytest.skip("tools/rs_rows.py no longer assigns MODES")
    assert set(ast.literal_eval(node.value)) == set(rd.MODES)


MODE_CYCLE = ("Ladder", "Ranked", "2v2", "Party")
#: 400 tags, modes in turn; 25 are in val, 15 of them in rs_rows's modes (Ladder, Ranked).
INDEX = [(f"#T{i:04d}", MODE_CYCLE[i % 4], f"part-{i // 100:05d}.parquet") for i in range(400)]
#: sample_tags(INDEX, 10, "val", 0): computed from the rule.
VAL10_SEED0 = ["#T0301", "#T0280", "#T0380", "#T0120", "#T0377", "#T0172", "#T0088", "#T0036",
               "#T0205", "#T0288"]


def test_sample_key_and_tags(rd):
    assert rd.sample_key("abc", 7) == int(hashlib.sha1(b"7:abc").hexdigest()[:16], 16)
    got = rd.sample_tags(INDEX, 10, "val", 0)
    assert got == VAL10_SEED0
    assert all(rd.split_of(t) == "val" for t in got)
    assert {m for t, m, _ in INDEX if t in got} <= rd.MODES
    # the order the corpus is read in does not matter, a tag listed twice counts once, and a
    # smaller sample is the start of a larger one
    assert rd.sample_tags(INDEX[::-1] + INDEX[:50], 10, "val", 0) == got
    assert rd.sample_tags(INDEX, 4, "val", 0) == got[:4]
    pool = [t for t, m, _ in INDEX if rd.split_of(t) == "val" and m in rd.MODES]
    assert len(pool) == 15
    assert sorted(rd.sample_tags(INDEX, 99, "val", 0)) == sorted(pool)
    assert rd.sample_tags(INDEX, 0, "val", 0) == []
    assert rd.sample_tags(INDEX, 10, "val", 1) != got
    # every mode, every split: the same rule over the whole index
    every = rd.sample_tags(INDEX, 10, "all", 0, modes=None)
    ref = sorted((t for t, _, _ in INDEX), key=lambda t: (rd.sample_key(t, 0), t))[:10]
    assert every == ref
    assert {m for t, m, _ in INDEX if t in every} - rd.MODES      # 2v2 / Party come in
    assert len(rd.sample_tags(INDEX, 99, "val", 0, modes=None)) == 25
    with pytest.raises(ValueError, match="--split"):
        rd.sample_tags(INDEX, 10, "dev")


def test_settings_parsers(rd):
    assert rd.TICK_MODES == ("exec", "tap") and rd.TAP_DELAY == 21
    assert rd.tick_mode("exec") == "exec" and rd.tick_mode("tap") == "tap"
    for bad in ("Tap", "", None, 1):
        with pytest.raises(ValueError, match="--tick-semantics"):
            rd.tick_mode(bad)
    assert rd.tap_ticks(21) == 21 and rd.tap_ticks("22") == 22 and rd.tap_ticks(" 22 ") == 22
    assert rd.tap_ticks(200) == 200
    for bad in (0, 201, -1, 21.5, True, "x", None):
        with pytest.raises(ValueError, match="--tap-delay"):
            rd.tap_ticks(bad)
    assert rd.parse_modes(None) is rd.MODES and rd.parse_modes(" ") is rd.MODES
    assert rd.parse_modes("all") is None and rd.parse_modes("ALL") is None
    assert rd.parse_modes("Ladder, Ranked") == frozenset({"Ladder", "Ranked"})
    with pytest.raises(ValueError, match="--modes"):
        rd.parse_modes(",")
    assert rd.describe_ticks("exec").startswith("exec (")
    assert "command delay 22" in rd.describe_ticks("tap", 22)
    eng = rd.rs_engine
    base = eng.EngineConfig.defaults()
    assert rd.match_config(base) == base and rd.match_config(base, "exec", 5) == base
    assert rd.match_config(base, "tap").delay == (21, 21)
    assert rd.match_config(base, "tap", 22).delay == (22, 22)
    assert rd.match_config(eng.EngineConfig(delay=(21, 21))).delay == (0, 0)
    legacy = eng.EngineConfig.legacy()
    assert rd.match_config(legacy).delay is None
    assert rd.match_config(legacy, "tap").delay == (21, 21)


def test_names_and_paths(rd):
    reports = ROOT / "reports" / "redrive"
    assert rd.resolve_pair(None) == "b2000"                         # as before
    assert rd.resolve_pair("b2000") == "b2000" and rd.resolve_pair("x", 2000) == "x"
    assert rd.resolve_pair(None, 2000, "val", 0) == "sample2000_val_seed0"
    assert rd.resolve_pair(None, 0, payloads=Path("a") / "grid.jsonl") == "grid"
    assert rd.sample_name(5, "test", 3) == "sample5_test_seed3"
    assert rd.output_path("b2000") == reports / "b2000_royalesim.jsonl"   # unchanged
    out = rd.output_path("b2000", "on")
    assert rd.tick_path(out) == out and rd.tick_path(out, "exec", 99) == out
    assert rd.tick_path(out, "tap") == reports / "b2000_royalesim_presses_tap21.jsonl"
    assert rd.tick_path(out, "tap", 22).name == "b2000_royalesim_presses_tap22.jsonl"
    assert rd.meta_path(Path("w") / "main-16.402.22-exec-presses-off.jsonl") == \
        Path("w") / "main-16.402.22-exec-presses-off.meta.json"
    assert rd.error_kind("orient: blue 0/3") == "orient"
    assert rd.error_kind("exception KeyError: 'x'") == "exception KeyError"
    assert rd.error_kind(None) == ""


def test_match_info(rd):
    payload = il_payload()
    assert rd.match_info(payload) == INFO
    payload["source"]["game_mode"] = "Ranked"                 # the corpus column wins
    assert rd.match_info(payload)["game_mode"] == "Ranked"
    payload["battle"]["team"]["players"][0]["deck"][0]["level"] = None   # read as 11
    payload["battle"]["opponent"]["players"][0]["tower_card"] = {"card_key": "cannoneer"}
    info = rd.match_info(payload)
    assert info["levels"]["team"] == {"11": 8}
    assert info["tower_card"] == {"team": "tower-princess", "opponent": "cannoneer"}
    assert info["tower_card_level"] == {"team": 11, "opponent": None}
    assert rd.match_info({"battle": [1], "source": None}) == {
        "game_mode": "", "split": "val", "tower_card": {"team": "", "opponent": ""},
        "tower_card_level": {"team": None, "opponent": None},
        "levels": {"team": {}, "opponent": {}}}
    assert rd.match_info(None)["split"] == "val"


def test_payload_file_round_trip(rd, tmp_path):
    texts = [json.dumps(il_payload(tag=f"t{k}")) for k in range(3)]
    path = rd.write_payload_file(tmp_path / "sub" / "p.jsonl",
                                 [json.dumps(json.loads(t), indent=1) for t in texts])
    assert path == tmp_path / "sub" / "p.jsonl"
    assert rd.read_payload_file(path) == texts


# ---------------------------------------------------------------------------
# re-drive on the fake engine
# ---------------------------------------------------------------------------

def test_exec_is_the_redrive_before(rd, engine):
    mod = engine(states=states())
    payload = json.dumps(il_payload())
    out = rd.one(payload, engine=cfg(rd))
    battle = mod.battles[-1]
    assert out == EXEC_OUT
    assert {k: out[k] for k in LEGACY} == LEGACY
    assert list(out)[:5] == ["tag", "ok", "error", "forced", "plays"]
    assert battle.names()[:2] == ["set_command_delay_ticks", "reset"]
    assert battle.calls[0].args == (0, 0)
    assert sent(battle) == SENT and step_ticks(battle) == STEP_TICKS
    assert battle.init_kwargs == cfg(rd).constructor_kwargs()
    # --tick-semantics exec named: the same row and the same calls
    again = rd.one(payload, engine=cfg(rd), tick_semantics="exec", tap_delay=99)
    assert again == out and mod.battles[-1].calls == battle.calls
    assert len(mod.battles) == 3                       # the card map's, then one per match


def test_tap_sets_the_delay_first_and_sends_at_the_recorded_ticks(rd, engine):
    mod = engine(DelayBattle, states=states())
    payload = json.dumps(il_payload())
    exec_out = rd.one(payload, engine=cfg(rd))
    exec_battle = mod.battles[-1]
    out = rd.one(payload, engine=cfg(rd), tick_semantics="tap")
    battle = mod.battles[-1]
    assert battle.names()[:2] == ["set_command_delay_ticks", "reset"]
    assert battle.calls[0].args == (21, 21) and exec_battle.calls[0].args == (0, 0)
    assert battle.calls[1:] == exec_battle.calls[1:]      # the same reset, steps and commands
    assert sent(battle) == SENT and step_ticks(battle) == STEP_TICKS
    assert battle.init_kwargs == exec_battle.init_kwargs == cfg(rd).constructor_kwargs()
    assert out == exec_out == EXEC_OUT                   # every play ran: the same row
    assert battle.pending == {0: [], 1: []} and battle.waiting == []
    rd.one(payload, engine=cfg(rd), tick_semantics="tap", tap_delay=22)
    assert mod.battles[-1].calls[0].args == (22, 22)
    rd.one(payload, engine=rd.rs_engine.EngineConfig.legacy(), tick_semantics="tap")
    legacy = mod.battles[-1]                             # a legacy config gets the call too
    assert legacy.names()[:2] == ["set_command_delay_ticks", "reset"]
    assert legacy.calls[0].args == (21, 21)
    with pytest.raises(ValueError, match="--tick-semantics"):
        rd.one(payload, engine=cfg(rd), tick_semantics="later")
    with pytest.raises(ValueError, match="--tap-delay"):
        rd.one(payload, engine=cfg(rd), tick_semantics="tap", tap_delay=0)


def pending_hog(battle, cmds, n):
    """Step results: the HogRider (blue slot 1) refused CARD_PENDING when sent."""
    return [[c[0], CARD_PENDING if c[:2] == (0, 1) else 0] for c in cmds]


def knight_short(team, kind, what):
    """Run reasons: the Knight (red, card 8) refused NOT_ENOUGH_ELIXIR when it runs."""
    return NOT_ENOUGH if (team, kind, what) == (1, "deploy", 8) else 0


def test_tap_counts_refusals_when_sent_and_when_run(rd, engine):
    engine(DelayBattle, states=states(), step_results=pending_hog, run_reason=knight_short)
    payload = json.dumps(il_payload())
    out = rd.one(payload, engine=cfg(rd), tick_semantics="tap")
    assert out["ok"] and out["plays"] == 5
    assert out["forced"] == {"not_enough_elixir": 1, "card_pending": 1}
    assert out["refusals"] == [[181, "opponent", "Knight", "not_enough_elixir", "run"],
                               [240, "team", "HogRider", "card_pending", "send"]]
    # exec on the same script: the refusal when sent counts as before, the runs are never read
    out = rd.one(payload, engine=cfg(rd))
    assert out["forced"] == {"card_pending": 1}
    assert out["refusals"] == [[240, "team", "HogRider", "card_pending", "send"]]


def test_a_card_missing_while_a_play_is_pending(rd, engine):
    engine(DelayBattle, states=states(team_hand=[5, 2, 0, 1]))   # no Skeletons in hand
    payload = json.dumps(il_payload())
    for kwargs, why in (({}, "hand"), ({"tick_semantics": "tap"}, "hand"),
                        ({"tick_semantics": "tap", "tap_delay": 200}, "hand_pending")):
        out = rd.one(payload, engine=cfg(rd), **kwargs)
        assert out["ok"] and out["forced"] == {why: 1}, kwargs
        assert out["refusals"] == [[420, "team", "Skeletons", why, "send"]], kwargs


def gk_dead(team, kind, what):
    """Run reasons: the Golden Knight's press (red, ability) refused NO_HERO when it runs."""
    return NO_HERO if (team, kind) == (1, "ability") else 0


def test_tap_counts_presses_when_they_run(rd, engine):
    mod = engine(DelayBattle, catalogue=CAT, states=states(), run_reason=gk_dead)
    payload = json.dumps(il_payload(opponent_deck=OPP_DECK, team_forms=TEAM_FORMS,
                                    abilities=PRESS_EVENTS))
    out = rd.one(payload, engine=cfg(rd), presses="on", tick_semantics="tap")
    tap_battle = mod.battles[-1]
    assert out["ok"] and out["forced"] == {} and out["refusals"] == []
    assert out["presses"] == {"events": 3, "sent": 3, "ok": 2, "refused": {"NO_HERO": 1},
                              "skipped": {}, "order_mismatch": 0}
    # exec: every press counted when sent (all accepted there), the same commands
    out = rd.one(payload, engine=cfg(rd), presses="on")
    assert out["presses"] == {"events": 3, "sent": 3, "ok": 3, "refused": {}, "skipped": {},
                              "order_mismatch": 0}
    assert tap_battle.calls[1:] == mod.battles[-1].calls[1:]
    assert [c for _, cmds in sent(tap_battle) for c in cmds if c[1] >= 4] == \
        [(0, 4, 0, 0), (1, 4, 0, 0), (0, 5, 0, 0)]


def test_a_bad_match_is_a_row_not_a_crash(rd, engine):
    mod = engine(states=states())
    payload = il_payload()
    del payload["events"][0]["coordinates"]              # the Musketeer at 100
    out = rd.one(json.dumps(payload), engine=cfg(rd))
    assert out == {"tag": "synthetic-0001", "ok": False,
                   "error": "exception KeyError: 'coordinates'", "forced": {}, "plays": 0,
                   **INFO, "where": out["where"]}
    assert out["where"].startswith("royalesim_redrive.py:")
    assert rd.error_kind(out["error"]) == "exception KeyError"
    assert len(mod.battles) == 1                         # the card map's: no match battle
    assert rd.one(json.dumps(il_payload()), engine=cfg(rd)) == EXEC_OUT   # the next one runs
    bad = rd.one("{not json", engine=cfg(rd))
    assert set(bad) == {"tag", "ok", "error", "forced", "plays", "where"}
    assert bad["tag"] == "" and bad["ok"] is False
    assert bad["error"].startswith("exception JSONDecodeError: ")


def test_a_refused_reset_is_a_row(rd, engine):
    engine(FormsRefused, states=states())
    out = rd.one(json.dumps(il_payload()), engine=cfg(rd))
    assert out["ok"] is False and out["plays"] == 0
    assert out["error"] == "exception RuntimeError: forms: IceWizard_hero is not in cards-15535.1"
    assert rd.error_kind(out["error"]) == "exception RuntimeError"
    assert out["where"].startswith("test_fidelity_grid.py:")
    assert "tower_level" not in out and out["tower_card"] == INFO["tower_card"]


def test_engine_errors_and_bad_settings_still_stop_the_run(rd, engine):
    engine(states=states())
    with pytest.raises(ValueError, match="--presses"):
        rd.one(json.dumps(il_payload()), engine=cfg(rd), presses="maybe")


# ---------------------------------------------------------------------------
# totals (shared by the re-drive's console and the grid)
# ---------------------------------------------------------------------------

PRINCESS = ("tower-princess", "tower-princess")
LV11 = {"team": {"11": 8}, "opponent": {"11": 8}}


def row(tag, ok=True, w=True, c=True, plays=10, forced=None, ec=(1, 0), rc=(1, 0),
        tower=PRINCESS, mode="Ladder", levels=LV11, error=""):
    """A re-drive row as royalesim_redrive.one writes it (the columns the grid reads)."""
    r = {"tag": tag, "ok": ok, "error": error, "forced": dict(forced or {}),
         "plays": plays if ok else 0, "game_mode": mode, "levels": levels}
    if tower is not None:
        r["tower_card"] = {"team": tower[0], "opponent": tower[1]}
    if ok:
        r.update(engine_crowns={"team": ec[0], "opponent": ec[1]},
                 recorded_crowns={"team": rc[0], "opponent": rc[1]},
                 winner_agree=w, crowns_agree=c)
    return r


#: The base cell: 4 of 7 ran; winner 3 of 4, crowns 2 of 4; 100 plays, 4 refused; 3-crown
#: endings: engine m2, m4, recording m3, m4.
BASE = [
    row("m1", plays=10),
    row("m2", c=False, plays=20, forced={"hand": 1}, ec=(3, 0), rc=(2, 0),
        levels={"team": {"11": 6, "12": 2}, "opponent": {"13": 8}}),
    row("m3", w=False, c=False, plays=30, forced={"hand": 1, "not_enough_elixir": 2},
        ec=(0, 1), rc=(3, 0), tower=("cannoneer", "tower-princess"), mode="Ranked",
        levels={"team": {"14": 8}, "opponent": {"11": 8}}),
    row("m4", plays=40, ec=(0, 3), rc=(0, 3), mode="Ranked"),
    row("m5", ok=False, error="orient: blue 0/3 own half", tower=("dagger-duchess",
                                                                   "tower-princess")),
    row("m6", ok=False, error="card not in RoyaleSim: void", tower=None, mode="", levels={}),
    row("m7", ok=False, error="exception KeyError: 'coordinates'"),
]
#: A second cell: 5 of 6 ran; winner 4 of 5 (m1 no), crowns 2 of 5 (m2, m5); 80 plays, 1
#: refused. With the base it shares m1, m2, m3.
TAP = [
    row("m1", w=False, c=False),
    row("m2", plays=20),
    row("m3", c=False, plays=30, forced={"hand": 1}, tower=("cannoneer", "tower-princess"),
        mode="Ranked"),
    row("m4", ok=False, error="exception RuntimeError: boom", mode="Ranked"),
    row("m5"),
    row("m8", c=False),
]


def test_totals(rd):
    assert rd.totals(BASE) == {
        "matches": 7, "ran": 4,
        "errors": {"orient": 1, "card not in RoyaleSim": 1, "exception KeyError": 1},
        "winner_agree": 3, "crowns_agree": 2, "plays": 100,
        "refused": {"hand": 2, "not_enough_elixir": 2},
        "three_crowns": {"engine": 2, "recorded": 2}}
    assert rd.totals([]) == {"matches": 0, "ran": 0, "errors": {}, "winner_agree": 0,
                             "crowns_agree": 0, "plays": 0, "refused": {},
                             "three_crowns": {"engine": 0, "recorded": 0}}


def test_redrive_main_wires_the_new_flags():
    text = (TOOLS / "royalesim_redrive.py").read_text(encoding="utf-8")
    main = text[text.index("def main()"):]
    for needle in ('"--tick-semantics"', "default=TICK_EXEC", '"--tap-delay"', '"--sample"',
                   '"--split"', '"--seed"', '"--modes"', '"--corpus"', '"--payloads"',
                   '"--save-payloads"', '"--out"', "rs_engine.add_engine_args(ap)",
                   "match_config(cfg, ticks, tap_delay)",
                   "partial(job, tick_semantics=ticks, tap_delay=tap_delay)",
                   "sample_payloads(args.corpus, args.sample, args.split, args.seed, modes)",
                   "resolve_pair(args.pair, args.sample, args.split, args.seed, args.payloads)",
                   "initializer=rs_engine.set_active", "meta_path(out).write_text(",
                   "tick_path(output_path(args.pair, args.presses), ticks, tap_delay)"):
        assert needle in main, needle


# ---------------------------------------------------------------------------
# grid (pure)
# ---------------------------------------------------------------------------

TARGET = "client160402017_20261019"


def test_cells_names_and_commands(grid, rd):
    cells = grid.grid_cells([TARGET, "client15535"], ["exec", "tap"], ["off", "on"], "py",
                            rs022="py22")
    names = [c.name for c in cells]
    assert names == [f"main-{t}-{k}-presses-{p}" for t in ("16.402.22", "15.535.29")
                     for k in ("exec", "tap") for p in ("off", "on")] + \
        [f"rs022-default-{k}-presses-{p}" for k in ("exec", "tap") for p in ("off", "on")]
    assert cells[0] == grid.Cell("main", "py", TARGET, "exec", "off")
    assert {c.python for c in cells[8:]} == {"py22"} and {c.table for c in cells[8:]} == \
        {"default"}
    assert grid.grid_cells(["default"], ["exec"], ["off"], "py") == \
        [grid.Cell("main", "py", "default", "exec", "off")]
    payloads, out = Path("w") / "payloads.jsonl", Path("w") / "o.jsonl"
    cmd = grid.cell_command(cells[2], payloads, out, workers=3, tap_delay=22)
    assert cmd == ["py", str(grid.REDRIVE), "--payloads", str(payloads), "--out", str(out),
                   "--pair", "main-16.402.22-tap-presses-off", "--card-table", TARGET,
                   "--tick-semantics", "tap", "--tap-delay", "22", "--presses", "off",
                   "--workers", "3"]
    assert grid.REDRIVE == TOOLS / "royalesim_redrive.py"
    text = grid.REDRIVE.read_text(encoding="utf-8")
    main = text[text.index("def main()"):]
    for flag in cmd[2::2]:                                 # every flag the cells pass exists
        assert f'"{flag}"' in main or flag == "--card-table", flag
    assert "--card-table" in (TOOLS / "rs_engine.py").read_text(encoding="utf-8")
    assert cells[0].settings() == {"interp": "main", "python": "py", "table": TARGET,
                                   "ticks": "exec", "presses": "off"}


def test_parse_tables_choices_and_labels(grid):
    assert grid.parse_tables("16.402.22, 15.535,default,client15535") == \
        [TARGET, "client15535", "default"]
    with pytest.raises(ValueError, match="--tables"):
        grid.parse_tables(" , ")
    assert grid.parse_choices("tap,exec,tap", ("exec", "tap"), "--ticks") == ["tap", "exec"]
    with pytest.raises(ValueError, match="--ticks"):
        grid.parse_choices("later", ("exec", "tap"), "--ticks")
    with pytest.raises(ValueError, match="--presses"):
        grid.parse_choices("", ("off", "on"), "--presses")
    assert grid.table_label(TARGET) == "16.402.22"
    assert grid.table_label("client15535") == "15.535.29"
    assert grid.table_label("client160402017_20261006") == "16.402.19"
    assert grid.table_label("default") == grid.table_label(None) == "default"
    assert grid.table_label("client999") == "client999"


def test_summarize_rates_and_refusals(grid):
    LS = grid.LS
    s = grid.summarize(BASE)
    assert {k: s[k] for k in ("matches", "ran", "winner_agree", "crowns_agree", "plays",
                              "refused_total")} == \
        {"matches": 7, "ran": 4, "winner_agree": 3, "crowns_agree": 2, "plays": 100,
         "refused_total": 4}
    assert s["winner"] == {"k": 3, "n": 4, "rate": 0.75, "lo": LS.wilson(3, 4)[0],
                           "hi": LS.wilson(3, 4)[1]}
    assert (s["winner"]["lo"], s["winner"]["hi"]) == pytest.approx((0.300636, 0.954414),
                                                                    abs=1e-6)
    assert (s["crowns"]["k"], s["crowns"]["n"], s["crowns"]["rate"]) == (2, 4, 0.5)
    assert (s["crowns"]["lo"], s["crowns"]["hi"]) == pytest.approx((0.150036, 0.849964),
                                                                    abs=1e-6)
    assert s["refusals_per_100"] == pytest.approx(4.0)
    assert s["refusals_per_100_by_reason"] == pytest.approx({"hand": 2.0,
                                                             "not_enough_elixir": 2.0})
    assert s["presses"] == {}
    empty = grid.summarize([])
    assert empty["winner"] == {"k": 0, "n": 0, "rate": None, "lo": 0.0, "hi": 0.0}
    assert empty["refusals_per_100"] == 0.0
    t = grid.summarize(TAP)
    assert (t["ran"], t["winner"]["k"], t["crowns"]["k"], t["plays"]) == (5, 4, 2, 80)
    assert t["refusals_per_100"] == pytest.approx(1.25)
    assert t["errors"] == {"exception RuntimeError": 1}


def test_strata(grid):
    tower = grid.stratify(BASE, grid.tower_stratum)
    assert list(tower) == ["tower-princess", "cannoneer + tower-princess",
                           "dagger-duchess + tower-princess", "unknown"]
    tp = tower["tower-princess"]                          # m1, m2, m4 ran; m7 did not
    assert (tp["matches"], tp["ran"], tp["winner"]["k"], tp["crowns"]["k"], tp["plays"]) == \
        (4, 3, 3, 2, 70)
    assert tp["refusals_per_100"] == pytest.approx(100 / 70)
    cannon = tower["cannoneer + tower-princess"]
    assert (cannon["ran"], cannon["winner"]["k"], cannon["refusals_per_100"]) == (1, 0, 10.0)
    assert tower["dagger-duchess + tower-princess"]["ran"] == 0
    assert tower["unknown"]["errors"] == {"card not in RoyaleSim": 1}
    mode = grid.stratify(BASE, grid.mode_stratum)
    assert list(mode) == ["Ladder", "Ranked", "unknown"]
    assert (mode["Ladder"]["matches"], mode["Ladder"]["ran"], mode["Ladder"]["winner"]["k"],
            mode["Ladder"]["crowns"]["k"]) == (4, 2, 2, 1)
    assert mode["Ladder"]["refusals_per_100"] == pytest.approx(100 / 30)
    assert (mode["Ranked"]["ran"], mode["Ranked"]["winner"]["k"],
            mode["Ranked"]["crowns"]["k"]) == (2, 1, 1)
    assert mode["Ranked"]["refusals_per_100"] == pytest.approx(300 / 70)
    assert grid.tower_stratum({"tower_card": {"team": "tower-princess"}}) == \
        "tower-princess + unknown"
    assert grid.tower_stratum({}) == "unknown" and grid.mode_stratum({}) == "unknown"


def test_paired_mcnemar(grid):
    w = grid.paired(BASE, TAP, "winner_agree")         # m1 1 -> 0, m2 1 -> 1, m3 0 -> 1
    assert w == {"n": 3, "rate": pytest.approx(2 / 3), "base_rate": pytest.approx(2 / 3),
                 "diff": 0.0, "better": 1, "worse": 1, "p_value": 1.0}
    c = grid.paired(BASE, TAP, "crowns_agree")         # m1 1 -> 0, m2 0 -> 1, m3 0 -> 0
    assert c == {"n": 3, "rate": pytest.approx(1 / 3), "base_rate": pytest.approx(1 / 3),
                 "diff": 0.0, "better": 1, "worse": 1, "p_value": 1.0}
    worse = [row(f"x{k}", w=False) for k in range(6)]
    better = [row(f"x{k}", w=True) for k in range(6)]
    p = grid.paired(worse, better)
    assert (p["n"], p["diff"], p["better"], p["worse"]) == (6, 1.0, 6, 0)
    assert p["p_value"] == pytest.approx(0.03125)
    assert grid.paired(BASE, []) is None and grid.paired([], TAP) is None


def test_level_histogram_and_best_cell(grid):
    assert grid.level_histogram(BASE) == {"11": 78, "12": 2, "13": 8, "14": 8}
    assert grid.level_histogram([]) == {}
    a = {"name": "a", "summary": grid.summarize(BASE)}
    b = {"name": "b", "summary": grid.summarize(TAP)}
    none = {"name": "c", "summary": grid.summarize([])}
    best = grid.best_cell([a, none, b])
    assert best["name"] == "b" and best["winner"]["k"] == 4
    assert best["meets_target"] is True and best["ci_above_target"] is False   # lo 37.6%
    assert grid.best_cell([a, none])["name"] == "a"
    assert grid.best_cell([a, none], target=0.8)["meets_target"] is False
    assert grid.best_cell([none]) is None
    twin = {"name": "twin", "summary": grid.summarize(BASE)}
    assert grid.best_cell([a, twin])["name"] == "a"                  # a tie keeps the first


def test_fresh(grid):
    cell = grid.Cell("main", "py", TARGET, "tap", "on")
    meta = {"source": {"sha256": "abc"}, "engine": {"card_table": TARGET}, "presses": "on",
            "tick_semantics": "tap", "tap_delay": 21}
    assert grid.fresh(meta, cell, "abc", 21)
    assert not grid.fresh(meta, cell, "abd", 21)                      # other payloads
    assert not grid.fresh(meta, cell, "abc", 22)                      # another delay
    assert not grid.fresh({**meta, "presses": "off"}, cell, "abc", 21)
    assert not grid.fresh({**meta, "tick_semantics": "exec"}, cell, "abc", 21)
    assert not grid.fresh({**meta, "engine": {"card_table": "client15535"}}, cell, "abc", 21)
    assert not grid.fresh(None, cell, "abc", 21) and not grid.fresh({}, cell, "abc", 21)
    plain = grid.Cell("rs022", "py22", "default", "exec", "off")
    meta = {"source": {"sha256": "abc"}, "engine": {"card_table": None}, "presses": "off",
            "tick_semantics": "exec", "tap_delay": 0}
    assert grid.fresh(meta, plain, "abc", 21)                         # exec has no delay


def test_rs022_python(grid, tmp_path):
    assert grid.rs022_python("off", root=tmp_path) is None
    assert grid.rs022_python("auto", root=tmp_path) is None           # no .venvs/rs022
    python = tmp_path / ".venvs" / "rs022" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    assert grid.rs022_python("auto", root=tmp_path) == python
    assert grid.rs022_python(None, root=tmp_path) == python
    assert grid.rs022_python(str(python)) == python
    with pytest.raises(FileNotFoundError, match="--rs022"):
        grid.rs022_python(str(tmp_path / "nope" / "python.exe"))


#: Fingerprints as royalesim_redrive's meta.json carries them.
FP_EXEC = {"version": "0.1.42", "card_table": "cards-160402017-20261019.1",
           "command_delay_ticks": [0, 0]}
FP_TAP = {**FP_EXEC, "command_delay_ticks": [21, 21]}
SAMPLE = {"kind": "sample", "n": 7, "split": "val", "seed": 0, "modes": ["Ladder", "Ranked"],
          "picked": 7, "found": 7, "sha256": "0123456789abcdef"}


def entries(grid):
    return [
        {"cell": grid.Cell("main", "py", TARGET, "exec", "off"), "status": "ran",
         "returncode": 0, "seconds": 2.0, "rows": BASE, "meta": {"fingerprint": FP_EXEC}},
        {"cell": grid.Cell("main", "py", TARGET, "tap", "off"), "status": "ran",
         "returncode": 0, "seconds": 2.0, "rows": TAP, "meta": {"fingerprint": FP_TAP}},
        {"cell": grid.Cell("rs022", "py22", "default", "exec", "off"), "status": "failed",
         "returncode": 1, "seconds": 3.0, "rows": [], "meta": {}, "log": "work/rs022.log",
         "log_tail": ["Traceback (most recent call last):", "ImportError: no royalesim"]},
    ]


def test_build_report(grid):
    rep = grid.build_report(SAMPLE, entries(grid), generated="2026-10-30 12:00 UTC")
    assert rep["base"] == "main-16.402.22-exec-presses-off" and rep["schema"] == 1
    assert [c["status"] for c in rep["cells"]] == ["ran", "ran", "failed"]
    assert rep["cells"][0]["vs_base"] is None
    assert rep["cells"][1]["vs_base"]["winner"]["n"] == 3
    assert rep["cells"][2]["vs_base"] == {"winner": None, "crowns": None}
    assert rep["cells"][1]["engine"]["command_delay_ticks"] == [21, 21]
    assert rep["best"]["name"] == "main-16.402.22-tap-presses-off"
    assert rep["levels"] == {"11": 78, "12": 2, "13": 8, "14": 8}
    assert list(rep["cells"][0]["strata"]) == ["tower", "mode"]
    assert rep["target"] == {"winner": 0.626, "crowns": 0.49}
    json.dumps(rep)                                       # all of it is JSON


def test_render(grid):
    md = grid.render(grid.build_report(SAMPLE, entries(grid),
                                       generated="2026-10-30 12:00 UTC"))
    lines = md.splitlines()
    for line in (
        "# RoyaleSim fidelity grid",
        "- Matches: 7 of split `val`, seed 0, modes Ladder, Ranked (7 payloads found of 7 "
        "picked by tag hash); payload file sha256 `0123456789ab`.",
        "| main-16.402.22-exec-presses-off | 0.1.42 | cards-160402017-20261019.1 | exec | off "
        "| 4 / 7 | 75.0% [30.1, 95.4] | 50.0% [15.0, 85.0] | 4.00 | 2 / 2 | base | base |",
        "| main-16.402.22-tap-presses-off | 0.1.42 | cards-160402017-20261019.1 | tap | off "
        "| 5 / 6 | 80.0% [37.6, 96.4] | 40.0% [11.8, 76.9] | 1.25 | 0 / 0 "
        "| +0.0 pp (p 1, n 3) | +0.0 pp (p 1, n 3) |",
        "| rs022-default-exec-presses-off (failed) | n/a | default | exec | off | 0 / 0 | n/a "
        "| n/a | 0.00 | 0 / 0 | n/a | n/a |",
        "| cell | plays | hand | not_enough_elixir |",
        "| main-16.402.22-exec-presses-off | 100 | 2.00 | 2.00 |",
        "| main-16.402.22-tap-presses-off | 80 | 1.25 | 0.00 |",
        "| main-16.402.22-exec-presses-off | orient | 1 |",
        "| main-16.402.22-tap-presses-off | exception RuntimeError | 1 |",
        "- `rs022-default-exec-presses-off`: exit code 1 after 3 s; log `work/rs022.log`.",
        "ImportError: no royalesim",
        "| tower-princess | main-16.402.22-exec-presses-off | 4 | 3 | 100.0% [43.8, 100.0] "
        "| 66.7% [20.8, 93.9] | 1.43 |",
        "| tower-princess | main-16.402.22-tap-presses-off | 5 | 4 | 75.0% [30.1, 95.4] "
        "| 50.0% [15.0, 85.0] | 0.00 |",
        "| Ranked | main-16.402.22-exec-presses-off | 2 | 2 | 50.0% [9.5, 90.5] "
        "| 50.0% [9.5, 90.5] | 4.29 |",
        "| 12 | 2 | 2.1% |",
        "| 13 | 8 | 8.3% |",
        "| main-16.402.22-tap-presses-off | `py` | 0.1.42 | cards-160402017-20261019.1 | n/a "
        "| n/a | 21/21 | n/a |",
    ):
        assert line in lines, line
    assert ("**Best winner agreement: `main-16.402.22-tap-presses-off`, 80.0% [37.6, 96.4] "
            "(crowns 40.0% [11.8, 76.9]); it reaches the 62.6% target.**") in md
    assert "## Ability presses" not in md                  # no cell with presses on
    # the failed cell: "(failed)" in the cells table, then the refusals and engines tables
    # only (no errors, no strata)
    assert md.count("| rs022-default-exec-presses-off |") == 2
    empty = grid.render(grid.build_report({"kind": "payloads", "path": "p.jsonl"}, []))
    assert "**No cell ran a match.**" in empty and "No match was read." in empty


# ---------------------------------------------------------------------------
# grid main (no subprocess: a fake cell runner)
# ---------------------------------------------------------------------------

def test_main_dry_run_prints_every_cell(grid, tmp_path, capsys):
    payloads = tmp_path / "p.jsonl"
    payloads.write_text(json.dumps(il_payload()) + "\n", encoding="utf-8")
    work = tmp_path / "work"
    code = grid.main(["--dry-run", "--payloads", str(payloads), "--python", "py", "--rs022",
                      "off", "--tables", "16.402.22,default", "--ticks", "exec,tap",
                      "--presses", "off", "--work", str(work)])
    out = capsys.readouterr().out
    assert code == 0
    assert "grid: 4 cells" in out
    for name in ("main-16.402.22-exec-presses-off", "main-16.402.22-tap-presses-off",
                 "main-default-exec-presses-off", "main-default-tap-presses-off"):
        assert name in out
    assert "--card-table default" in out and "--tick-semantics tap --tap-delay 21" in out
    assert not work.exists()
    with pytest.raises(SystemExit):
        grid.main(["--dry-run", "--payloads", str(payloads), "--rs022", "off",
                   "--ticks", "later"])
    with pytest.raises(SystemExit):
        grid.main(["--dry-run", "--payloads", str(tmp_path / "missing.jsonl"),
                   "--rs022", "off"])


def test_main_stops_without_matches(grid, tmp_path, monkeypatch):
    payloads = tmp_path / "empty.jsonl"
    payloads.write_text("\n", encoding="utf-8")
    monkeypatch.setattr(grid, "run_cell", lambda *a, **k: pytest.fail("no cell should run"))
    assert grid.main(["--payloads", str(payloads), "--python", "py", "--rs022", "off",
                      "--work", str(tmp_path / "work"), "--out",
                      str(tmp_path / "grid.md")]) == 2
    assert grid.count_lines(payloads) == 0 and not (tmp_path / "grid.md").exists()


#: Rows the fake runner writes per tick semantics.
FAKE_ROWS = {"exec": [row("m1"), row("m2", w=False, c=False)],
             "tap": [row("m1", w=False, c=False), row("m2")]}


def test_main_runs_each_cell_and_writes_the_report(grid, tmp_path, monkeypatch):
    payloads = tmp_path / "p.jsonl"
    grid.RD.write_payload_file(payloads, [json.dumps(il_payload(tag=f"m{k}")) for k in (1, 2)])
    calls, fail = [], {"main-16.402.22-tap-presses-off"}

    def fake_run(cmd, log, timeout=None):
        args = dict(zip(cmd[2::2], cmd[3::2]))
        calls.append(args["--pair"])
        if args["--pair"] in fail:
            Path(log).write_text("Traceback (most recent call last):\nRuntimeError: boom\n",
                                 encoding="utf-8")
            return 1, 0.5
        out = Path(args["--out"])
        out.write_text("".join(json.dumps(r) + "\n" for r in FAKE_ROWS[args["--tick-semantics"]]),
                       encoding="utf-8")
        tap = args["--tick-semantics"] == "tap"
        meta = {"source": {"kind": "payloads",
                           "sha256": grid.rs_cards.file_sha256(args["--payloads"])},
                "engine": {"card_table": grid.rs_engine.table_name(args["--card-table"])},
                "presses": args["--presses"], "tick_semantics": args["--tick-semantics"],
                "tap_delay": int(args["--tap-delay"]) if tap else 0,
                "fingerprint": FP_TAP if tap else FP_EXEC, "seconds": 1.5}
        grid.RD.meta_path(out).write_text(json.dumps(meta), encoding="utf-8")
        Path(log).write_text("ok\n", encoding="utf-8")
        return 0, 1.5

    monkeypatch.setattr(grid, "run_cell", fake_run)
    md_path = tmp_path / "grid.md"
    argv = ["--payloads", str(payloads), "--python", "py", "--rs022", "off", "--tables",
            "16.402.22", "--ticks", "exec,tap", "--presses", "off", "--work",
            str(tmp_path / "work"), "--out", str(md_path)]
    assert grid.main(argv) == 1                              # a cell failed
    assert calls == ["main-16.402.22-exec-presses-off", "main-16.402.22-tap-presses-off"]
    report = json.loads(md_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert [c["status"] for c in report["cells"]] == ["ran", "failed"]
    assert report["sample"]["kind"] == "payloads"
    assert report["sample"]["sha256"] == grid.rs_cards.file_sha256(payloads)
    assert report["cells"][0]["summary"]["winner"]["k"] == 1
    assert report["cells"][1]["log_tail"][-1] == "RuntimeError: boom"
    md = md_path.read_text(encoding="utf-8")
    assert "## Cells that failed" in md and "RuntimeError: boom" in md
    # again with --reuse: the cell that ran is kept, the failed one runs
    fail.clear()
    calls.clear()
    assert grid.main(argv + ["--reuse"]) == 0
    assert calls == ["main-16.402.22-tap-presses-off"]
    report = json.loads(md_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert [c["status"] for c in report["cells"]] == ["reused", "ran"]
    vs = report["cells"][1]["vs_base"]["winner"]
    assert (vs["n"], vs["better"], vs["worse"], vs["diff"]) == (2, 1, 1, 0.0)
    assert report["cells"][1]["engine"]["command_delay_ticks"] == [21, 21]
    assert report["best"]["name"] == "main-16.402.22-exec-presses-off"   # a tie: the base


# ---------------------------------------------------------------------------
# U10: candidate cells and shared payloads
# ---------------------------------------------------------------------------

NEW_ARM = "client160635001_20261102"


def test_candidate_cells_names_and_commands(grid, tmp_path):
    cells = grid.grid_cells([TARGET], ["exec"], ["off"], "py", rs022="py22", candidate="pyc",
                            candidate_tables=[TARGET, NEW_ARM])
    assert [c.name for c in cells] == [
        "main-16.402.22-exec-presses-off", "candidate-16.402.22-exec-presses-off",
        f"candidate-{NEW_ARM}-exec-presses-off", "rs022-default-exec-presses-off"]
    assert cells[1] == grid.Cell("candidate", "pyc", TARGET, "exec", "off")
    cmd = grid.cell_command(cells[2], Path("p.jsonl"), Path("o.jsonl"), workers=2)
    assert cmd[0] == "pyc" and cmd[cmd.index("--card-table") + 1] == NEW_ARM
    assert cmd[cmd.index("--pair") + 1] == f"candidate-{NEW_ARM}-exec-presses-off"
    # default: the main tables; another label; no candidate: the grid as before
    assert [c.name for c in grid.grid_cells([TARGET, "client15535"], ["exec"], ["off"], "py",
                                            candidate="pyc", candidate_label="rs0.1.43")] == [
        "main-16.402.22-exec-presses-off", "main-15.535.29-exec-presses-off",
        "rs0.1.43-16.402.22-exec-presses-off", "rs0.1.43-15.535.29-exec-presses-off"]
    assert grid.grid_cells([TARGET], ["exec"], ["off"], "py", rs022="py22") == \
        grid.grid_cells([TARGET], ["exec"], ["off"], "py", "py22", None, [NEW_ARM])
    assert grid.parse_tables(f"16.402.22,{NEW_ARM}") == [TARGET, NEW_ARM]
    assert grid.table_label(NEW_ARM) == NEW_ARM
    for good in ("candidate", "rs0.1.43", "next_2"):
        assert grid.candidate_label(good) == good
    for bad in ("main", "rs022", "a/b", "", "-x", "a-b", None):
        with pytest.raises(ValueError, match="--candidate-label"):
            grid.candidate_label(bad)
    assert grid.candidate_python(None) is None and grid.candidate_python(" ") is None
    python = tmp_path / "python.exe"
    python.write_text("", encoding="utf-8")
    assert grid.candidate_python(str(python)) == python
    with pytest.raises(FileNotFoundError, match="--candidate"):
        grid.candidate_python(str(tmp_path / "nope.exe"))


def guard_cell(grid, name, rows, status="ran", interp="candidate", ticks="exec",
               presses="off"):
    return {"name": name, "status": status, "summary": grid.summarize(rows),
            "settings": {"interp": interp, "python": "py", "table": TARGET, "ticks": ticks,
                         "presses": presses}}


def test_regression_guard(grid):
    ref_rows = [row(f"g{k}") for k in range(100)]
    ref = guard_cell(grid, "base", ref_rows, interp="main")

    def check(rows, status="ran", matches=None, reference=ref, reference_rows=ref_rows):
        return grid.regression(reference, reference_rows, guard_cell(grid, "c", rows, status),
                               rows, matches)

    def lost(n, total=100):
        return [row(f"g{k}", w=k >= n) for k in range(total)]

    g = check(lost(1))
    assert g == {"reference": "base", "winner": {
        "n": 100, "rate": pytest.approx(0.99), "base_rate": 1.0, "diff": pytest.approx(-0.01),
        "better": 0, "worse": 1, "p_value": 1.0}, "refusals_rise": 0.0, "failed": 0,
        "failed_reference": 0, "failed_limit": pytest.approx(1.0), "blocked": False,
        "reasons": []}
    g = check(lost(2))                                   # 2 points, p 0.5
    assert g["winner"]["p_value"] == 0.5 and not g["blocked"]
    g = check(lost(10))
    assert g["blocked"] and g["reasons"] == ["winner agreement -10.0 pp (p 0.00195, n 100)"]
    big = [row(f"g{k}") for k in range(1000)]
    g = grid.regression(guard_cell(grid, "base", big, interp="main"), big,
                        guard_cell(grid, "c", lost(15, 1000)), lost(15, 1000))
    assert g["winner"]["p_value"] < 1e-4 and not g["blocked"]          # only 1.5 points
    g = grid.regression(guard_cell(grid, "base", big, interp="main"), big,
                        guard_cell(grid, "c", lost(20, 1000)), lost(20, 1000))
    assert g["blocked"] and g["reasons"][0].startswith("winner agreement -2.0 pp")
    g = check([row(f"g{k}", forced={"hand": 2}) for k in range(100)])
    assert g["refusals_rise"] == pytest.approx(20.0)
    assert g["reasons"] == ["refusals +20.00 per 100 plays"]
    g = check([row(f"g{k}", forced={"hand": 1} if k < 10 else None) for k in range(100)])
    assert g["refusals_rise"] == pytest.approx(1.0) and not g["blocked"]
    bad = [row(f"g{k}", ok=False, error="exception RuntimeError: x") for k in (98, 99)]
    g = check(ref_rows[:98] + bad)
    assert (g["failed"], g["blocked"]) == (2, True)
    assert g["reasons"] == ["2 matches did not run (limit 1)"]
    assert not check(ref_rows[:99] + bad[1:])["blocked"]                 # 1 is the limit
    g = check([], status="failed")
    assert g["winner"] is None
    assert g["reasons"] == ["the cell did not run", "100 matches did not run (limit 1)"]
    gone = guard_cell(grid, "base", [], status="failed", interp="main")
    g = check(ref_rows, reference=gone, reference_rows=[])
    assert g["reasons"] == ["its reference base did not run"] and g["failed_reference"] == 100
    # the sample's size, when given, counts matches missing from a cell's output
    g = check(ref_rows[:50], matches=100)
    assert (g["failed"], g["blocked"]) == (50, True)
    cells = [guard_cell(grid, n, [], interp=i, ticks=t)
             for n, i, t in (("a", "main", "exec"), ("b", "main", "tap"),
                             ("c", "candidate", "tap"), ("d", "rs022", "exec"))]
    assert grid.reference_index(cells, cells[2]) == 1
    assert grid.reference_index(cells, guard_cell(grid, "e", [], presses="on")) is None
    assert [grid.is_candidate(c) for c in cells] == [False, False, True, False]


FP_NEXT = {"version": "0.1.43", "card_table": "cards-160402017-20261019.1",
           "command_delay_ticks": [0, 0]}


def candidate_entries(grid, status="ran"):
    return [
        {"cell": grid.Cell("main", "py", TARGET, "exec", "off"), "status": "ran",
         "returncode": 0, "seconds": 2.0, "rows": BASE, "meta": {"fingerprint": FP_EXEC}},
        {"cell": grid.Cell("candidate", "pyc", TARGET, "exec", "off"), "status": status,
         "returncode": 0 if status == "ran" else 1, "seconds": 2.0,
         "rows": TAP if status == "ran" else [], "meta": {"fingerprint": FP_NEXT},
         "log": "work/candidate.log", "log_tail": []},
    ]


def test_report_with_a_candidate(grid):
    plain = grid.build_report(SAMPLE, entries(grid), generated="2026-10-30 12:00 UTC")
    assert "candidate" not in plain and all("guard" not in c for c in plain["cells"])
    assert "## Candidate engine" not in grid.render(plain)
    rep = grid.build_report(SAMPLE, candidate_entries(grid), generated="2026-10-30 12:00 UTC",
                            matches=7)
    base, cand = rep["cells"]
    name = "candidate-16.402.22-exec-presses-off"
    assert rep["base"] == base["name"] == "main-16.402.22-exec-presses-off"
    assert cand["name"] == name and "guard" not in base
    vs = cand["vs_base"]["winner"]                       # McNemar against the base
    assert (vs["n"], vs["better"], vs["worse"], vs["diff"], vs["p_value"]) == (3, 1, 1, 0.0, 1.0)
    assert cand["guard"] == {"reference": base["name"], "winner": vs,
                             "refusals_rise": pytest.approx(-2.75), "failed": 2,
                             "failed_reference": 3, "failed_limit": pytest.approx(3.07),
                             "blocked": False, "reasons": []}
    assert rep["candidate"] == {"cells": [name], "target_cells": [name], "blocked": False,
                                "blocked_cells": [], "noted_cells": []}
    json.dumps(rep)
    lines = grid.render(rep).splitlines()
    for line in (
        "## Candidate engine against the base",
        f"| {name} | 0.1.43 | cards-160402017-20261019.1 | exec | off | 5 / 6 "
        "| 80.0% [37.6, 96.4] | 40.0% [11.8, 76.9] | 1.25 | 0 / 0 "
        "| +0.0 pp (p 1, n 3) | +0.0 pp (p 1, n 3) |",
        f"| {name} | main-16.402.22-exec-presses-off | +0.0 pp (p 1, n 3) | 1.25 (-2.75) "
        "| 2 (3.07) | pass |",
        "**Regression guard: pass.**",
    ):
        assert line in lines, line
    failed = grid.build_report(SAMPLE, candidate_entries(grid, "failed"), matches=7)
    assert failed["candidate"] == {"cells": [name], "target_cells": [name], "blocked": True,
                                   "blocked_cells": [name], "noted_cells": []}
    assert failed["cells"][1]["guard"]["reasons"] == ["the cell did not run",
                                                      "7 matches did not run (limit 3.07)"]
    md = grid.render(failed)
    assert f"**Regression guard: BLOCKED ({name}).**" in md
    assert "| BLOCKED: the cell did not run; 7 matches did not run (limit 3.07) |" in md


def test_report_guard_blocks_on_the_target_cells_only(grid):
    """docs/SEASON_UPDATES.md section 9 blocks on the target cell: with --candidate-target
    naming the new arm, the new engine failing on the old table is noted, not blocking."""
    ents = candidate_entries(grid, "failed") + [
        {"cell": grid.Cell("candidate", "pyc", NEW_ARM, "exec", "off"), "status": "ran",
         "returncode": 0, "seconds": 2.0, "rows": BASE, "meta": {"fingerprint": FP_NEXT}}]
    old_name = "candidate-16.402.22-exec-presses-off"
    new_name = f"candidate-{NEW_ARM}-exec-presses-off"
    every = grid.build_report(SAMPLE, ents, matches=7)          # default: every candidate cell
    assert every["candidate"] == {"cells": [old_name, new_name],
                                  "target_cells": [old_name, new_name], "blocked": True,
                                  "blocked_cells": [old_name], "noted_cells": []}
    rep = grid.build_report(SAMPLE, ents, matches=7, target_tables=[NEW_ARM])
    assert rep["cells"][2]["guard"]["blocked"] is False                 # BASE against BASE
    assert rep["candidate"] == {"cells": [old_name, new_name], "target_cells": [new_name],
                                "blocked": False, "blocked_cells": [],
                                "noted_cells": [old_name]}
    md = grid.render(rep)
    assert "**Regression guard: pass.**" in md
    assert (f"Not the target cell, so not blocking: {old_name} (section 9 blocks on the "
            "target cell only).") in md
    assert ("| not the target cell, would block: the cell did not run; 7 matches did not run "
            "(limit 3.07) |") in md
    # no candidate cell on the target tables cannot pass
    none = grid.build_report(SAMPLE, ents, matches=7, target_tables=["client15535"])
    assert none["candidate"]["blocked"] is True and none["candidate"]["blocked_cells"] == []
    assert grid.guard_verdict(none["candidate"]) == "BLOCKED (no target cell)"
    assert grid.guard_verdict(rep["candidate"]) == "pass"
    assert grid.guard_verdict(every["candidate"]) == f"BLOCKED ({old_name})"


def two_payloads():
    return [json.dumps(il_payload(tag=f"m{k}")) for k in (1, 2)]


def test_shared_sample(grid, tmp_path):
    rd = grid.RD
    root, corpus = tmp_path / "shared", tmp_path / "corpus"
    picks = []

    def pick(corpus_, n, split, seed, modes):
        picks.append((str(corpus_), n, split, seed, modes))
        return ["m1", "m2"], two_payloads()

    path, sample = grid.shared_sample(root, n=2, split="val", seed=0, modes=rd.MODES,
                                      corpus=corpus, pick=pick)
    folder = root / "sample2_val_seed0"
    assert path == folder / "payloads.jsonl"
    assert picks == [(str(corpus), 2, "val", 0, rd.MODES)]
    assert rd.read_payload_file(path) == [json.dumps(json.loads(t)) for t in two_payloads()]
    spec = json.loads((folder / "sample.json").read_text(encoding="utf-8"))
    assert sample == {**spec, "shared": str(folder)}
    assert {k: spec[k] for k in ("kind", "n", "split", "seed", "modes", "rule", "corpus",
                                 "picked", "found")} == {
        "kind": "sample", "n": 2, "split": "val", "seed": 0, "modes": sorted(rd.MODES),
        "rule": rd.SAMPLE_RULE, "corpus": str(corpus), "picked": 2, "found": 2}
    assert spec["sha256"] == grid.rs_cards.file_sha256(path) and spec["created"].endswith("UTC")

    def never(*args):
        pytest.fail("a shared sample is never picked again")

    # again, from another corpus folder: the same file, untouched
    again = grid.shared_sample(root, n=2, split="val", seed=0, modes=rd.MODES,
                               corpus=tmp_path / "elsewhere", pick=never)
    assert again == (path, sample)
    with pytest.raises(ValueError, match=r"other settings \(modes\)"):
        grid.shared_sample(root, n=2, split="val", seed=0, modes=None, corpus=corpus,
                           pick=never)
    text = path.read_text(encoding="utf-8")
    path.write_text(text + text.splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since it was picked"):
        grid.shared_sample(root, n=2, split="val", seed=0, modes=rd.MODES, corpus=corpus,
                           pick=never)
    (folder / "sample.json").unlink()
    with pytest.raises(ValueError, match="sample.json is missing"):
        grid.shared_sample(root, n=2, split="val", seed=0, modes=rd.MODES, corpus=corpus,
                           pick=never)
    path.unlink()                                       # neither file: picked afresh
    assert grid.shared_sample(root, n=2, split="val", seed=0, modes=rd.MODES, corpus=corpus,
                              pick=pick)[0] == path
    assert len(picks) == 2
    with pytest.raises(ValueError, match="no match"):
        grid.shared_sample(tmp_path / "empty", n=2, split="val", seed=0, modes=rd.MODES,
                           corpus=corpus, pick=lambda *a: ([], []))
    assert not (tmp_path / "empty").exists()


def test_main_candidate_flags_and_dry_run(grid, tmp_path, capsys):
    payloads = tmp_path / "p.jsonl"
    payloads.write_text(json.dumps(il_payload()) + "\n", encoding="utf-8")
    cand = tmp_path / "rs_next_python"
    cand.write_text("", encoding="utf-8")
    base = ["--dry-run", "--python", "py", "--rs022", "off", "--tables", "16.402.22",
            "--ticks", "exec", "--presses", "off", "--work", str(tmp_path / "work")]
    for extra in (["--payloads", str(payloads), "--shared-payloads"],
                  ["--payloads", str(payloads), "--shared-root", str(tmp_path)],
                  ["--payloads", str(payloads), "--candidate-tables", "16.402.22"],
                  ["--payloads", str(payloads), "--candidate-label", "next"],
                  ["--payloads", str(payloads), "--candidate", str(tmp_path / "nope")],
                  ["--payloads", str(payloads), "--candidate", str(cand),
                   "--candidate-label", "main"],
                  ["--payloads", str(payloads), "--candidate-target", "16.402.22"],
                  ["--payloads", str(payloads), "--candidate", str(cand),
                   "--candidate-target", NEW_ARM]):          # not one of the candidate tables
        with pytest.raises(SystemExit):
            grid.main(base + extra)
    capsys.readouterr()
    # --reuse on a shared sample needs a work folder of the run's own
    no_work = [a for a in base if a not in ("--work", str(tmp_path / "work"))]
    with pytest.raises(SystemExit) as exc:
        grid.main(no_work + ["--shared-payloads", "--reuse", "--shared-root",
                             str(tmp_path / "s")])
    assert exc.value.code == 2
    assert "--reuse with --shared-payloads: give --work" in capsys.readouterr().err
    shared = tmp_path / "shared"
    assert grid.main(base + ["--shared-payloads", "--shared-root", str(shared), "--sample",
                             "5", "--candidate", str(cand), "--candidate-tables",
                             f"16.402.22,{NEW_ARM}"]) == 0
    out = capsys.readouterr().out
    assert "grid: 3 cells" in out and f"candidate cells (candidate): {cand}" in out
    want = shared.resolve() / "sample5_val_seed0" / "payloads.jsonl"
    assert f"--payloads {want}" in out
    assert f"{cand} {grid.REDRIVE} " in out and f"--card-table {NEW_ARM}" in out
    assert f"candidate-{NEW_ARM}-exec-presses-off" in out
    assert not shared.exists() and not (tmp_path / "work").exists()
    assert grid.main(base + ["--payloads", str(payloads), "--candidate", str(cand),
                             "--candidate-label", "rs0.1.43"]) == 0
    assert "rs0.1.43-16.402.22-exec-presses-off" in capsys.readouterr().out


#: Rows the fake runner writes per cell of the update grid below.
UPDATE_ROWS = {
    "main-16.402.22-exec-presses-off": [row("m1"), row("m2")],
    "candidate-16.402.22-exec-presses-off": [row("m1"), row("m2", w=False, c=False)],
    f"candidate-{NEW_ARM}-exec-presses-off": [
        row("m1", ok=False, error="exception RuntimeError: no such arm"),
        row("m2", ok=False, error="exception RuntimeError: no such arm")],
}


def test_main_update_grid_on_a_shared_sample(grid, tmp_path, monkeypatch, capsys):
    root = tmp_path / "shared"
    path, sample = grid.shared_sample(root, n=2, split="val", seed=0, modes=grid.RD.MODES,
                                      corpus=tmp_path, pick=lambda *a: (["m1", "m2"],
                                                                        two_payloads()))
    cand = tmp_path / "rs_next_python"
    cand.write_text("", encoding="utf-8")
    calls = []

    def fake_run(cmd, log, timeout=None):
        args = dict(zip(cmd[2::2], cmd[3::2]))
        calls.append((args["--pair"], cmd[0], args["--card-table"]))
        assert args["--payloads"] == str(path.resolve())
        out = Path(args["--out"])
        out.write_text("".join(json.dumps(r) + "\n" for r in UPDATE_ROWS[args["--pair"]]),
                       encoding="utf-8")
        meta = {"source": {"kind": "payloads",
                           "sha256": grid.rs_cards.file_sha256(args["--payloads"])},
                "engine": {"card_table": grid.rs_engine.table_name(args["--card-table"])},
                "presses": args["--presses"], "tick_semantics": args["--tick-semantics"],
                "tap_delay": 0, "fingerprint": FP_NEXT if cmd[0] == str(cand) else FP_EXEC,
                "seconds": 1.0}
        grid.RD.meta_path(out).write_text(json.dumps(meta), encoding="utf-8")
        Path(log).write_text("ok\n", encoding="utf-8")
        return 0, 1.0

    monkeypatch.setattr(grid, "run_cell", fake_run)
    md_path, work = tmp_path / "grid.md", tmp_path / "work"
    argv = ["--shared-payloads", "--shared-root", str(root), "--sample", "2", "--split", "val",
            "--python", "py", "--rs022", "off", "--tables", "16.402.22", "--ticks", "exec",
            "--presses", "off", "--candidate", str(cand), "--candidate-tables",
            f"16.402.22,{NEW_ARM}", "--work", str(work), "--out", str(md_path)]
    assert grid.main(argv) == 0                          # every cell ran
    names = list(UPDATE_ROWS)
    assert calls == [(names[0], "py", TARGET), (names[1], str(cand), TARGET),
                     (names[2], str(cand), NEW_ARM)]
    out = capsys.readouterr().out
    assert f"regression guard: BLOCKED ({names[2]})" in out and "series -> " in out
    report = json.loads(md_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert [c["name"] for c in report["cells"]] == names
    assert Path(report["sample"]["shared"]) == Path(sample["shared"]).resolve()
    assert report["sample"]["sha256"] == sample["sha256"]
    assert report["candidate"] == {"cells": names[1:], "target_cells": names[1:],
                                   "blocked": True, "blocked_cells": [names[2]],
                                   "noted_cells": []}
    old, new = report["cells"][1]["guard"], report["cells"][2]["guard"]
    assert old["blocked"] is False and old["winner"]["diff"] == -0.5   # p 1: not a regression
    assert new["reasons"] == ["2 matches did not run (limit 0.02)"]
    assert report["cells"][1]["vs_base"]["winner"]["n"] == 2
    md = md_path.read_text(encoding="utf-8")
    assert f"**Regression guard: BLOCKED ({names[2]}).**" in md
    assert "every run on it appends its numbers to `series.jsonl` there." in md
    series_path = root / "sample2_val_seed0" / "series.jsonl"
    lines = [json.loads(x) for x in series_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    line = lines[0]
    assert (line["report"], line["work"], line["base"]) == (str(md_path), str(work.resolve()),
                                                             names[0])
    assert line["payloads_sha256"] == sample["sha256"]
    assert [c["blocked"] for c in line["cells"]] == [None, False, True]
    assert line["cells"][2] == {
        "name": names[2], "status": "ran", "interp": "candidate", "table": NEW_ARM,
        "ticks": "exec", "presses": "off", "version": "0.1.43",
        "card_table": "cards-160402017-20261019.1", "card_table_fnv1a64": None,
        "fingerprint": None, "matches": 2, "ran": 0, "winner": None, "crowns": None,
        "refusals_per_100": 0.0, "blocked": True}
    assert line["cells"][0]["winner"] == 1.0 and line["cells"][1]["winner"] == 0.5
    # again with --reuse: no cell runs, the shared file is as it was, the series grows
    calls.clear()
    assert grid.main(argv + ["--reuse"]) == 0
    assert calls == []
    assert grid.rs_cards.file_sha256(path) == sample["sha256"]
    assert len(series_path.read_text(encoding="utf-8").splitlines()) == 2
    # the target cell named: the old table's candidate cell passes, the new arm's failure is
    # noted and does not block (section 9 blocks on the target cell)
    capsys.readouterr()
    assert grid.main(argv + ["--reuse", "--candidate-target", "16.402.22"]) == 0
    assert calls == []
    assert (f"regression guard: pass (not the target cell, not blocking: {names[2]})"
            in capsys.readouterr().out)
    report = json.loads(md_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["candidate"] == {"cells": names[1:], "target_cells": [names[1]],
                                   "blocked": False, "blocked_cells": [],
                                   "noted_cells": [names[2]]}
    # a shared sample picked with other modes is refused before any cell runs
    other = tmp_path / "other"
    grid.shared_sample(other, n=2, split="val", seed=0, modes=None, corpus=tmp_path,
                       pick=lambda *a: (["m1", "m2"], two_payloads()))
    capsys.readouterr()
    assert grid.main([*argv[:2], str(other), *argv[3:]]) == 2
    assert calls == [] and "other settings (modes)" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

@pytest.mark.engine
def test_real_engine_runs_the_fixture_under_both_tick_semantics(rd):
    cfg_real = rd.rs_engine.EngineConfig.load()
    payload = json.dumps(il_payload(**PLAIN))
    for kwargs in ({}, {"tick_semantics": "tap"}):
        out = rd.one(payload, engine=cfg_real, **kwargs)
        assert out["ok"], (kwargs, out)
        assert out["plays"] == 5 and out["tower_level"] == 11
        assert out["tower_card"] == INFO["tower_card"]
        assert out["levels"] == {"team": {"11": 8}, "opponent": {"11": 8}}
        assert sum(out["forced"].values()) == len(out["refusals"])


@pytest.mark.engine
def test_real_engine_grid_end_to_end(grid, tmp_path):
    """Two fixture matches, cells exec and tap, each re-driven by royalesim_redrive.py in a
    subprocess of this interpreter."""
    payloads = tmp_path / "fixture.jsonl"
    grid.RD.write_payload_file(payloads, [json.dumps(il_payload(tag=f"fixture-{k}", **PLAIN))
                                          for k in range(2)])
    md_path = tmp_path / "grid.md"
    code = grid.main(["--payloads", str(payloads), "--python", sys.executable, "--rs022", "off",
                      "--tables", "16.402.22", "--ticks", "exec,tap", "--presses", "off",
                      "--workers", "1", "--work", str(tmp_path / "work"), "--out",
                      str(md_path)])
    report = json.loads(md_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert code == 0, [c["log_tail"] for c in report["cells"]]
    assert [c["summary"]["matches"] for c in report["cells"]] == [2, 2]
    assert [c["summary"]["ran"] for c in report["cells"]] == [2, 2]
    meta = json.loads((tmp_path / "work" / "main-16.402.22-tap-presses-off.meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["tick_semantics"] == "tap" and meta["tap_delay"] == 21
    assert meta["engine"]["delay"] == [21, 21]
    assert meta["totals"]["matches"] == 2


@pytest.mark.engine
def test_real_engine_candidate_cells_on_a_shared_sample(grid, tmp_path):
    """The update grid on two fixture matches: this interpreter as the old engine and as the
    candidate. The same engine re-drives the same matches the same way, so the guard passes."""
    texts = [json.dumps(il_payload(tag=f"fixture-{k}", **PLAIN)) for k in range(2)]
    root = tmp_path / "shared"
    grid.shared_sample(root, n=2, split="val", seed=0, modes=grid.RD.MODES, corpus=tmp_path,
                       pick=lambda *a: ([f"fixture-{k}" for k in range(2)], texts))
    md_path = tmp_path / "grid.md"
    code = grid.main(["--shared-payloads", "--shared-root", str(root), "--sample", "2",
                      "--python", sys.executable, "--rs022", "off", "--tables", "16.402.22",
                      "--candidate", sys.executable, "--ticks", "exec", "--presses", "off",
                      "--workers", "1", "--work", str(tmp_path / "work"), "--out",
                      str(md_path)])
    report = json.loads(md_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert code == 0, [c["log_tail"] for c in report["cells"]]
    assert [c["name"] for c in report["cells"]] == ["main-16.402.22-exec-presses-off",
                                                    "candidate-16.402.22-exec-presses-off"]
    assert [c["summary"]["ran"] for c in report["cells"]] == [2, 2]
    vs = report["cells"][1]["vs_base"]["winner"]
    assert (vs["n"], vs["diff"]) == (2, 0.0)
    assert report["candidate"]["blocked"] is False
    assert (root / "sample2_val_seed0" / "series.jsonl").is_file()
