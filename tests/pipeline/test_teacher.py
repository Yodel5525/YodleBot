"""R05: tools/rl/teacher.py, the optional search teacher, and its hook in tools/rl/ppo.py.

Pure numpy, on a FakeBattle whose state follows what is played (Board: a play lands `delay`
ticks after it is sent and damages red's tower 1; save / load snapshot all of it):
  - candidates are restored between trials: every trial starts from the saved state (its hash)
    and the battle hashes as before afterwards; the calls are save, (load, step) per candidate,
    load; verify catches a battle whose load() does not restore it
  - delay-aware: each trial steps delay + horizon ticks with the candidate sent first, and our
    plays already in flight land in every trial; the delay comes from the call, the config or
    the battle, and an unknown one is refused
  - the target by hand (wait against the drawn slot's two tiles; keep_unsearched on and off),
    refused candidates dropped, another slot's legal-tile scan, the buttons and a capped wait,
    the critic on the after-state's row (its tick, V + Phi), a battle that ended reads its result,
    the prior; on rs_decide's own sampled decisions the targets sum to 1 inside the mask
  - rate / wants / record / blank, the config and its flags, seat_row_fn's obs_row call, the
    swing against rs_rows.towers_vec, a one-row policy refused for the critic
  - ppo: write_rollout(extra=) round trip, ppo_inputs' checks, teacher_coef off by default
With torch (importorskip): kl_rows by hand and 0 for the student itself; the batched critic
equals row by row on a tiny TokenPolicy v2; an update at lr 0 with the student as teacher
measures a KL of 0 and changes nothing; one update where only the teacher's term pulls moves
the policy toward the teacher.

No engine, no data, no emulator.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import Call, FakeBattle, player_dict, state_dict

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tools.rl import ppo as P  # noqa: E402
from tools.rl import teacher as T  # noqa: E402
import rs_decide as RD  # noqa: E402
import rs_schema as RS  # noqa: E402

KING, PRINCESS = 4824, 3052
#: Red's princess towers' max hp on the Board: 100 damage is 0.1 of one.
P_MAX = 1000
SUB = 18
NOW = 500
#: conftest's DEFAULT_CATALOGUE cards 0-3 (Cannon, Fireball, HogRider, IceGolemite), blue's
#: hand in slots 0-3: their elixir, and the damage a landed play does to red's tower 1 per
#: (world column + 1).
COSTS = (3, 4, 4, 2)
DMG = (50, 0, 100, 0)
#: The drawn decision's gate distribution: wait, slots 0-3.
Q = (0.5, 0.1, 0.1, 0.25, 0.05)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class Board(FakeBattle):
    """A FakeBattle whose state follows what is played. Blue (team 0) holds cards 0-3 in slots
    0-3. A play is refused NOT_ENOUGH_ELIXIR (4) when blue's elixir is short, else paid at send
    and landed `delay` ticks later, taking DMG[card] x (world column + 1) off red's tower 1
    (card `kill`: also red's king, the battle over, blue the winner). A press (slot 4+) is
    accepted and does nothing. `pending`: blue's plays in flight, (land tick, card, x_sub,
    y_sub, cost). Legal: blue's own half (world y below 15000 mt). save() snapshots all of it,
    load() restores it; `starts` holds the state's digest at the start of every step."""

    def __init__(self, *, delay=8, elixir=5000, pending=(), kill=None, **kw):
        super().__init__(deploy=lambda team, slot, x, y: 0 if y // SUB < 15000 else 7, **kw)
        self.delay = (int(delay), int(delay))
        self.tick = NOW
        self.elixir = int(elixir)
        self.queue = [tuple(p) for p in pending]
        self.red = [KING, P_MAX, P_MAX]
        self.kill = kill
        self.starts = []

    def command_delay_ticks(self):
        return list(self.delay)

    def state(self):
        pend = [[0, card, x, y, max(0, land - self.tick), cost]
                for land, card, x, y, cost in self.queue]
        blue = player_dict(0, hand=[0, 1, 2, 3], elixir_milli=self.elixir, pending=pend,
                           pending_cost=sum(p[5] for p in pend))
        red = player_dict(1, tower_hp=list(self.red), tower_max_hp=(KING, P_MAX, P_MAX))
        over = self.red[0] <= 0
        return state_dict(self.tick, players=[blue, red], game_over=over,
                          winner=0 if over else -1)

    def digest(self):
        return hashlib.sha1(json.dumps(self.state(), sort_keys=True).encode()).hexdigest()

    def step(self, cmds=(), n=1):
        cmds = [tuple(int(v) for v in c) for c in cmds]
        self.calls.append(Call("step", (cmds, int(n)), {}))
        self.starts.append(self.digest())
        out = []
        for team, slot, x, y in cmds:
            if team != 0:
                out.append([team, 1])                       # BAD_TEAM
            elif slot >= 4:
                out.append([team, 0])                       # a press: nothing on the board
            elif self.elixir < COSTS[slot] * 1000:
                out.append([team, 4])                       # NOT_ENOUGH_ELIXIR
            else:
                self.elixir -= COSTS[slot] * 1000
                self.queue.append((self.tick + self.delay[0], slot, x, y, COSTS[slot]))
                out.append([team, 0])
        self.tick += int(n)
        self.steps += 1
        keep = []
        for item in sorted(self.queue):
            land, card, x, _, _ = item
            if land > self.tick:
                keep.append(item)
                continue
            if card == self.kill:
                self.red[0] = 0
            self.red[1] = max(0, self.red[1] - DMG[card] * (x // SUB // 1000 + 1))
        self.queue = keep
        return out

    def save(self):
        self.calls.append(Call("save", (), {}))
        return copy.deepcopy({"tick": self.tick, "steps": self.steps, "elixir": self.elixir,
                              "queue": self.queue, "red": self.red})

    def load(self, snap):
        self.calls.append(Call("load", (), {}))
        s = copy.deepcopy(snap)
        self.tick, self.steps, self.elixir = s["tick"], s["steps"], s["elixir"]
        self.queue, self.red = s["queue"], s["red"]


class Leaky(Board):
    """A Board whose load() forgets to restore red's towers."""

    def load(self, snap):
        red = list(self.red)
        super().load(snap)
        self.red = red


def tile_logits():
    """Own-frame tile 306 (row 17, column 0: blue's first own-half row) first, then 307, 308,
    ...; the enemy half (tiles below 306) far behind. Strictly decreasing: no ties."""
    t = np.arange(576, dtype=np.float64)
    return np.where(t >= 306, -(t - 306) / 100, -10 - t / 100).astype(np.float32)


class Policy:
    """rs_decide's policy protocol without torch: the batch is the row, the gate logits fixed,
    the critic's won logit the row's "z", tile_logits() for every slot. `read`: the rows the
    critic was asked about."""

    def __init__(self, gate=(0.0, 1.0, 0.5, 2.0, -1.0)):
        self.g = np.asarray(gate, dtype=np.float32)
        self.read = []

    def batch(self, row):
        return row

    def heads(self, b):
        self.read.append(b)
        return self.g.copy(), (0.0, float(b.get("z", 0.0)))

    def gate(self, b):
        return self.g.copy()

    def tiles(self, b, slot):
        return tile_logits()


class OneRow:
    """tools/rl/rollout.py Heads' shape: batch(row) is itself, one row's outputs."""

    def batch(self, row):
        return self

    def heads(self, b):
        return np.zeros(5, dtype=np.float32), (0.0, 0.4)

    def tiles(self, b, slot):
        return tile_logits()


def row_fn(state, tick):
    """The critic's row of an after-state: z = blue's elixir in thousands."""
    return {"z": state["players"][0]["elixir_milli"] / 1000, "tick": int(tick)}


def drawn(q=Q, tiles=(306, 307), tile_a=1, slot=2):
    """A sampling-preset Decision that drew `slot` and tiles[tile_a] (a play, no fallback), its
    logq_tile what the student's tile distribution (explore 0.15, temperature 1) gives it."""
    cfg = RD.DecideConfig.sampling(0.0)
    lg = tile_logits().astype(np.float64)
    pt = RD.tile_probs(lg[list(tiles)], cfg.explore, cfg.tile_temp)
    t = tiles[tile_a]
    wx, wy = RD.tile_centre_world(t, 0)
    return RD.Decision(team=0, tick=NOW, kind=RD.PLAY, slot=slot, card=slot, cost=COSTS[slot],
                       tile=t, tx=t % 18, ty=t // 18, x_sub=wx * SUB, y_sub=wy * SUB,
                       mask=(True,) * len(q), gate_q=tuple(q), gate_a=slot + 1,
                       logq_gate=math.log(q[slot + 1]), tile_cands=tuple(tiles), tile_a=tile_a,
                       logq_tile=math.log(pt[tile_a]))


#: The hand-computed search: the drawn slot only, its best tile (and the drawn one), 4 ticks
#: after the landing, temperature 0.1, the score the swing alone.
HAND = T.TeacherConfig(k_cards=1, k_tiles=1, horizon=4, temperature=0.1, value_weight=0.0,
                       prior=0.0)
ROW = {"tick": NOW}


def logp(p):
    p = np.asarray(p, dtype=np.float64)
    return np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), T.NEG).astype(np.float32)


# ---------------------------------------------------------------------------
# the search on the fake battle
# ---------------------------------------------------------------------------

def test_the_search_restores_the_battle_between_trials_and_after():
    """Tile 306 is own column 0, row 17: world (500, 14500) mt, (9000, 261000) sub-tiles; 307
    is column 1. Each trial steps delay 8 + horizon 4 = 12 ticks with the candidate sent first;
    the play lands at 508 and takes 100 x (column + 1) off red's 1000-hp tower."""
    board = Board(delay=8)
    start = board.digest()
    target = T.Teacher(HAND).annotate(board, 0, drawn(), Policy(), ROW)
    assert board.digest() == start and board.tick == NOW and board.elixir == 5000
    assert board.names() == ["save"] + ["load", "step"] * 3 + ["load"]
    assert board.starts == [start] * 3                   # every trial began from the saved state
    assert [c.args for c in board.calls_named("step")] == [
        ([], 12), ([(0, 2, 9000, 261000)], 12), ([(0, 2, 27000, 261000)], 12)]
    assert board.deploy_checks == []                     # the drawn slot's tiles are its own
    assert [(t.entry, t.tile, t.cand, t.in_gate, t.ran) for t in target.trials] == [
        (0, -1, -1, True, True), (3, 306, 0, True, True), (3, 307, 1, False, True)]
    assert [t.swing for t in target.trials] == pytest.approx([0.0, 0.1, 0.2])


def test_the_target_by_hand():
    """Scores / temperature: wait 0, slot 2 at 306 1, at its drawn tile 307 2 (not in the gate:
    the k_tiles = 1 best only). Gate: the searched {wait, slot 2} keep their gate_q mass 0.75
    and split it e^0 : e^1, the rest keep theirs; tile: e^1 : e^2 over the two candidates (both
    searched, so all of the student's tile mass)."""
    e = math.e
    w, s2 = 1 / (1 + e), e / (1 + e)
    target = T.Teacher(HAND).annotate(Board(delay=8), 0, drawn(), Policy(), ROW)
    assert target.gate_ok and target.tile_ok and target.refused == 0
    assert target.gate.tolist() == pytest.approx([0.75 * w, 0.1, 0.1, 0.75 * s2, 0.05])
    assert target.tile.tolist() == pytest.approx([1 / (1 + e), e / (1 + e)])
    assert target.gate.sum() == pytest.approx(1.0) and target.tile.sum() == pytest.approx(1.0)
    assert target.best.tile == 307
    rec = target.record()
    assert set(rec) == set(T.FIELDS)
    assert rec["teacher_ok"] is True and rec["teacher_tile_ok"] is True
    assert rec["teacher_gate"].dtype == np.float32 and rec["teacher_tile"].shape == (P.CANDS,)
    assert np.exp(rec["teacher_gate"].astype(np.float64)).tolist() == pytest.approx(
        target.gate.tolist(), abs=1e-6)
    assert np.exp(rec["teacher_tile"][:2].astype(np.float64)).tolist() == pytest.approx(
        target.tile.tolist(), abs=1e-6)
    assert (rec["teacher_tile"][2:] == T.NEG).all() and int(rec["teacher_n"]) == 3
    # keep_unsearched off: the searched entries take all the mass, the others none
    alone = T.Teacher(HAND.replace(keep_unsearched=False)).annotate(
        Board(delay=8), 0, drawn(), Policy(), ROW)
    assert alone.gate.tolist() == pytest.approx([w, 0.0, 0.0, s2, 0.0])
    assert (alone.record()["teacher_gate"][[1, 2, 4]] == T.NEG).all()
    assert alone.tile.tolist() == pytest.approx(target.tile.tolist())


def test_an_entry_is_judged_by_its_best_trial_not_by_how_many_it_has():
    """The drawn IceGolemite (slot 3, no damage on the Board) at three tiles against the wait:
    every trial swings 0, so the wait and slot 3 split their gate_q mass 0.7 evenly (a
    log-sum-exp would give slot 3 three votes to the wait's one: 0.525 : 0.175), and the tile
    target is uniform. Slot 2's two tiles score 1 and 2 (over the temperature): the slot reads
    its best, e^0 : e^2 against the wait."""
    e = math.e
    q = (0.4, 0.1, 0.1, 0.1, 0.3)
    d = drawn(q=q, tiles=(306, 307, 308), tile_a=0, slot=3)
    target = T.Teacher(HAND.replace(k_tiles=3)).annotate(Board(delay=8), 0, d, Policy(), ROW)
    assert [(t.entry, t.cand, t.in_gate, t.ran) for t in target.trials] == [
        (0, -1, True, True), (4, 0, True, True), (4, 1, True, True), (4, 2, True, True)]
    assert [t.score for t in target.trials] == pytest.approx([0.0] * 4)
    assert target.gate.tolist() == pytest.approx([0.35, 0.1, 0.1, 0.1, 0.35])
    assert target.tile.tolist() == pytest.approx([1 / 3] * 3)
    two = T.Teacher(HAND.replace(k_tiles=2)).annotate(Board(delay=8), 0, drawn(), Policy(), ROW)
    assert [(t.entry, t.in_gate) for t in two.trials] == [(0, True), (3, True), (3, True)]
    assert two.gate.tolist() == pytest.approx(
        [0.75 / (1 + e ** 2), 0.1, 0.1, 0.75 * e ** 2 / (1 + e ** 2), 0.05])
    assert two.tile.tolist() == pytest.approx([1 / (1 + e), e / (1 + e)])


def test_our_plays_in_flight_land_first_and_the_prior():
    """A blue Cannon in flight, landing at 504 in column 3 (50 x 4 = 200 damage): it lands in
    every trial, each swing is 0.2 more and the target is the same. prior 1: the student's
    log-probability joins the score, log gate_q of the entry plus the tile's log-softmax."""
    pend = [(504, 0, 3500 * SUB, 14500 * SUB, 3)]
    board = Board(delay=8, pending=pend)
    target = T.Teacher(HAND).annotate(board, 0, drawn(), Policy(), ROW)
    assert [t.swing for t in target.trials] == pytest.approx([0.2, 0.3, 0.4])
    assert board.queue == pend                           # restored: still in flight
    base = T.Teacher(HAND).annotate(Board(delay=8), 0, drawn(), Policy(), ROW)
    assert target.gate.tolist() == pytest.approx(base.gate.tolist())
    assert target.tile.tolist() == pytest.approx(base.tile.tolist())
    lg = tile_logits().astype(np.float64)
    ls = lg - np.logaddexp.reduce(lg)
    pri = T.Teacher(HAND.replace(prior=1.0)).annotate(Board(delay=8), 0, drawn(), Policy(), ROW)
    s = [t.score for t in pri.trials]
    assert s[0] == pytest.approx(math.log(0.5))
    assert s[1] == pytest.approx(0.1 + math.log(0.25) + ls[306])
    assert s[2] - s[1] == pytest.approx(0.1 + ls[307] - ls[306])


def test_refused_candidates_are_dropped_and_other_slots_scan_for_a_legal_tile():
    """Blue has 3 elixir on the board (the decision's mask, read off a noisy HUD, let slot 2
    in): the drawn HogRider (4) is refused at send on both its tiles, so slot 2 keeps the
    student's mass and there is no tile target. k_cards 2 also searches slot 0 (the Cannon, 3):
    its first legal tile in logit order, 306, after one check_deploy. Gate: {wait, slot 0}
    keep 0.5 + 0.1 and split it e^0 : e^0.5 (swings 0 and 0.05)."""
    board = Board(delay=8, elixir=3000)
    start = board.digest()
    teacher = T.Teacher(HAND.replace(k_cards=2))
    target = teacher.annotate(board, 0, drawn(), Policy(), ROW)
    assert board.digest() == start
    assert [(t.entry, t.tile, t.ran) for t in target.trials] == [
        (0, -1, True), (3, 306, False), (3, 307, False), (1, 306, True)]
    assert target.refused == 2 and teacher.stats["refused"] == 2
    assert board.deploy_checks == [(0, 0, 9000, 261000)]
    r = math.exp(0.5)
    assert target.gate.tolist() == pytest.approx(
        [0.6 / (1 + r), 0.6 * r / (1 + r), 0.1, 0.25, 0.05])
    assert target.gate_ok and not target.tile_ok
    rec = target.record()
    assert rec["teacher_ok"] and not rec["teacher_tile_ok"]
    assert (rec["teacher_tile"] == T.NEG).all()


def test_buttons_are_searched_and_a_capped_wait_is_not():
    """Capped (gate_q gives the wait nothing): no wait trial. Button 0 is in the mask and gets
    a press trial, (0, HAND_SIZE + 0, 0, 0); button 1 is masked. Gate: {slot 2 (score 0.1),
    button 0 (0)} keep 0.3 + 0.3 and split it e^1 : e^0."""
    e = math.e
    q = (0.0, 0.2, 0.1, 0.3, 0.1, 0.3, 0.0)
    d = dataclasses.replace(drawn(q=q), mask=(True,) * 6 + (False,), capped=True)
    board = Board(delay=8)
    target = T.Teacher(HAND).annotate(board, 0, d, Policy(), ROW)
    assert [t.entry for t in target.trials] == [3, 3, 5]
    assert board.calls_named("step")[2].args == ([(0, 4, 0, 0)], 12)
    assert target.gate.tolist() == pytest.approx(
        [0.0, 0.2, 0.1, 0.6 * e / (1 + e), 0.1, 0.6 / (1 + e), 0.0])
    # without the buttons only slot 2 is searched: no gate comparison, the tile one stays
    one = T.Teacher(HAND.replace(presses=False)).annotate(Board(delay=8), 0, d, Policy(), ROW)
    assert [t.entry for t in one.trials] == [3, 3]
    assert not one.gate_ok and one.tile_ok
    rec = one.record()
    assert rec["teacher_ok"] is False and rec["teacher_tile_ok"] is True
    assert (rec["teacher_gate"] == T.NEG).all()


def test_the_value_reads_the_critic_on_the_after_state_and_the_result_at_the_end():
    """swing_weight 0, value_weight 1: the score is V = tanh(z / 2) of row_fn's row of the
    after-state (z = blue's elixir in thousands, read at the trial's last tick, 512), plus with
    potential Phi of the after-state. An after-state where the battle ended (the drawn HogRider
    kills the king here) reads its result and the critic is not asked about it."""
    cfg = HAND.replace(swing_weight=0.0, value_weight=1.0, potential=False)
    policy = Policy()
    target = T.Teacher(cfg).annotate(Board(delay=8), 0, drawn(), policy, ROW, row_fn=row_fn)
    assert [r["tick"] for r in policy.read] == [512, 512, 512]
    assert [r["z"] for r in policy.read] == [5.0, 1.0, 1.0]
    v5, v1 = math.tanh(2.5), math.tanh(0.5)
    assert [t.score for t in target.trials] == pytest.approx([v5, v1, v1])
    # Phi = 0.1 x (own 1 - enemy (1 + 0.9 + 1) / 3) after 100 damage, (1 + 0.8 + 1) / 3 after 200
    phi = T.Teacher(cfg.replace(potential=True)).annotate(Board(delay=8), 0, drawn(), Policy(),
                                                          ROW, row_fn=row_fn)
    assert [t.score for t in phi.trials] == pytest.approx([v5, v1 + 0.1 / 30, v1 + 0.2 / 30])
    policy = Policy()
    end = T.Teacher(cfg).annotate(Board(delay=8, kill=2), 0, drawn(), policy, ROW,
                                  row_fn=row_fn)
    assert [t.terminal for t in end.trials] == [False, True, True]
    assert [t.score for t in end.trials] == pytest.approx([v5, 1.0, 1.0])
    assert len(policy.read) == 1
    board = Board(delay=8)
    with pytest.raises(T.TeacherError):                  # the critic needs the after-state rows
        T.Teacher(cfg).annotate(board, 0, drawn(), Policy(), ROW)
    assert board.calls == []
    # a value_fn stands in for the policy's critic, one value per row
    vf = T.Teacher(cfg).annotate(Board(delay=8), 0, drawn(), Policy(), ROW, row_fn=row_fn,
                                 value_fn=lambda rows: [r["z"] / 10 for r in rows])
    assert [t.score for t in vf.trials] == pytest.approx([0.5, 0.1, 0.1])
    with pytest.raises(T.TeacherError):
        T.Teacher(cfg).annotate(Board(delay=8), 0, drawn(), Policy(), ROW, row_fn=row_fn,
                                value_fn=lambda rows: [0.0])


def test_the_delay_comes_from_the_call_the_config_or_the_battle():
    """Trials of delay + horizon 4 ticks: the call's delay=, else TeacherConfig.delay, else the
    battle's command_delay_ticks() (8 here); none of them: refused before anything runs."""
    five = HAND.replace(delay=5)
    for kw, cfg, steps in (({}, HAND, 12), ({"delay": 2}, HAND, 6), ({}, five, 9),
                           ({"delay": 0}, five, 4)):
        board = Board(delay=8)
        T.Teacher(cfg).annotate(board, 0, drawn(), Policy(), ROW, **kw)
        assert {c.args[1] for c in board.calls_named("step")} == {steps}, (kw, cfg.delay)
    plain = FakeBattle()                                 # reports no command delay
    with pytest.raises(T.TeacherError):
        T.Teacher(HAND).annotate(plain, 0, drawn(), Policy(), ROW)
    assert plain.calls == []
    assert T.battle_delay(Board(delay=7), 1) == 7 and T.battle_delay(plain, 0) is None
    assert T.battle_delay(SimpleNamespace(command_delay_ticks=lambda: 24), 0) == 24


def test_verify_and_the_decisions_the_teacher_refuses():
    T.Teacher(HAND).annotate(Leaky(delay=8), 0, drawn(), Policy(), ROW)      # goes unnoticed
    with pytest.raises(T.TeacherError):
        T.Teacher(HAND.replace(verify=True)).annotate(Leaky(delay=8), 0, drawn(), Policy(), ROW)
    T.Teacher(HAND.replace(verify=True)).annotate(Board(delay=8), 0, drawn(), Policy(), ROW)
    # a logq_tile the student's distribution does not give: not the decision's row / policy
    with pytest.raises(T.TeacherError):
        T.Teacher(HAND).annotate(Board(delay=8), 0,
                                 dataclasses.replace(drawn(), logq_tile=math.log(0.9)),
                                 Policy(), ROW)
    with pytest.raises(T.TeacherError):                  # not the sampling preset: no gate_q
        T.Teacher(HAND).annotate(Board(delay=8), 0, dataclasses.replace(drawn(), gate_q=()),
                                 Policy(), ROW)
    none = dataclasses.replace(drawn(), kind=RD.WAIT, reason=RD.R_NO_CARD, gate_a=-1)
    board = Board(delay=8)
    assert T.Teacher(HAND).annotate(board, 0, none, Policy(), ROW) is None and board.calls == []


@pytest.mark.parametrize("seed", range(6))
def test_rs_decides_sampled_decisions_get_targets_inside_their_mask(seed):
    """rs_decide.decide (sampling preset) on the Board through the Policy, then the default
    teacher (2 cards x 3 tiles, horizon 60, the board's own delay 24, verify): the battle is as
    it was and every trial began from it, each trial is 84 ticks, and a target's probabilities
    sum to 1 inside the decision's mask (the tile's inside its candidates)."""
    rng = np.random.default_rng(seed)
    board = Board(delay=24, elixir=int(rng.choice([2000, 3500, 5000, 9000])))
    policy = Policy(gate=rng.normal(0, 1, 5))
    cfg = RD.DecideConfig.sampling(0.3)
    state = json.loads(board.state_json())
    d = RD.decide(policy, ROW, state, 0, RD.BattleLegality(board), cfg,
                  rng=np.random.default_rng(seed + 10))
    assert d.gate_a >= 0
    start = board.digest()
    target = T.Teacher(T.TeacherConfig(verify=True)).annotate(board, 0, d, policy, ROW,
                                                              row_fn=row_fn, dcfg=cfg)
    assert board.digest() == start and set(board.starts) == {start}
    assert target is not None
    assert [c.args[1] for c in board.calls_named("step")] == [84] * len(target.trials)
    mask = np.asarray(d.mask)
    assert target.gate.sum() == pytest.approx(1.0) and (target.gate >= 0).all()
    assert (target.gate[~mask] == 0).all()
    rec = target.record()
    if target.gate_ok:
        pg = np.exp(rec["teacher_gate"].astype(np.float64))
        assert pg[mask].sum() == pytest.approx(1.0, abs=1e-5) and pg[~mask].sum() == 0
    assert target.tile_ok == (d.kind == RD.PLAY)         # a play compares its tiles here
    if target.tile_ok:
        assert len(target.tile) == len(d.tile_cands) and target.tile.sum() == pytest.approx(1.0)
        pt = np.exp(rec["teacher_tile"].astype(np.float64))
        assert pt[:len(d.tile_cands)].sum() == pytest.approx(1.0, abs=1e-5)
        assert pt[len(d.tile_cands):].sum() == 0


def test_rate_wants_record_and_blank():
    d = drawn()
    none = dataclasses.replace(d, kind=RD.WAIT, reason=RD.R_NO_CARD, gate_a=-1)
    off = T.Teacher(HAND.replace(rate=0.0))
    before = off.rng.bit_generator.state
    assert not off.wants(none) and off.rng.bit_generator.state == before   # no draw, no number
    board = Board(delay=8)
    rec = off.record(board, 0, d, Policy(), ROW)
    assert rec["teacher_ok"] is False and rec["teacher_tile_ok"] is False
    assert int(rec["teacher_n"]) == 0 and rec["teacher_gate"].shape == (5,)
    assert (rec["teacher_gate"] == T.NEG).all() and (rec["teacher_tile"] == T.NEG).all()
    assert board.calls == [] and off.stats["skipped"] == 1
    on = T.Teacher(HAND.replace(rate=1.0))
    rec = on.record(board, 0, d, Policy(), ROW)
    assert rec["teacher_ok"] is True and rec["teacher_tile_ok"] is True
    assert int(rec["teacher_n"]) == 3 and on.stats["searched"] == 1
    # the share searched follows the rate, drawn from the teacher's own seeded generator
    share = T.Teacher(HAND.replace(rate=0.3), seed=4)
    got = [share.wants(d) for _ in range(2000)]
    assert 0.25 < float(np.mean(got)) < 0.35
    again = T.Teacher(HAND.replace(rate=0.3), seed=4)
    assert [again.wants(d) for _ in range(2000)] == got
    assert T.blank(7)["teacher_gate"].shape == (7,)
    assert T.blank(7)["teacher_tile"].shape == (P.CANDS,)


def test_the_config_and_its_flags():
    d = T.TeacherConfig()
    assert (d.k_cards, d.k_tiles, d.horizon, d.rate, d.delay) == (2, 3, 60, 0.1, None)
    assert d.keep_unsearched and d.potential and not d.verify
    assert T.TeacherConfig.from_json(json.loads(json.dumps(d.to_json()))) == d
    for bad in ({"k_cards": 0}, {"temperature": 0.0}, {"rate": 1.5}, {"horizon": 0},
                {"delay": -1}, {"prior": float("nan")}):
        with pytest.raises(ValueError):
            T.TeacherConfig(**bad)
    ap = T.add_args(argparse.ArgumentParser())
    assert T.config_from_args(ap.parse_args([])) is None          # off by default
    cfg = T.config_from_args(ap.parse_args(
        ["--teacher", "--teacher-rate", "0.5", "--teacher-k-tiles", "2", "--teacher-search-all",
         "--teacher-no-potential", "--teacher-verify"]))
    assert (cfg.rate, cfg.k_tiles, cfg.k_cards) == (0.5, 2, 2)
    assert not cfg.keep_unsearched and not cfg.potential and cfg.verify


def test_seat_row_fn_builds_the_search_agents_look_ahead_row(monkeypatch):
    import rs_rows as RR
    calls = []
    monkeypatch.setattr(RR, "obs_row", lambda *a, **k: calls.append((a, k)) or {"row": 1})
    seat = SimpleNamespace(live=False, visibility="public",
                           v2_kw=lambda state, team, mem, tick, rs: {
                               "schema": RS.SCHEMA_V2, "opp_elixir": (4000, True),
                               "pending": [], "asked": (team, tick, rs)})
    mem = {"seen": {}, "last": {}}
    state = state_dict(612)
    assert T.seat_row_fn(seat, 1, mem, 18, "RS")(state, 612) == {"row": 1}
    (args, kw), = calls
    assert args == (state, 1, 18, mem, 612)
    assert kw == {"live": False, "visibility": "public", "schema": RS.SCHEMA_V2,
                  "opp_elixir": (4000, True), "pending": [], "asked": (1, 612, "RS")}


def test_the_swing_is_the_search_agents_and_a_one_row_policy_is_refused():
    import rs_rows as RR
    before = state_dict(500)
    after = state_dict(560, players=[player_dict(0, tower_hp=(KING, 2000, PRINCESS)),
                                     player_dict(1, tower_hp=(4000, PRINCESS, 1000))])
    for team in (0, 1):
        b, a = RR.towers_vec(before, team), RR.towers_vec(after, team)
        want = float((b[3:] - a[3:]).sum() - (b[:3] - a[:3]).sum())
        assert T.tower_swing(before, after, team) == pytest.approx(want, abs=1e-6)
    assert T.tower_swing(before, after, 0) == pytest.approx(
        824 / 4824 + 2052 / 3052 - 1052 / 3052)
    # rollout.Heads answers one row only: the critic of the after-states needs value_fn
    with pytest.raises(T.TeacherError):
        T.policy_values(OneRow(), [{}])
    assert T.forward_values(lambda rows: [OneRow() for _ in rows])([{}, {}]).tolist() == (
        pytest.approx([math.tanh(0.2)] * 2))
    assert T.policy_values(SimpleNamespace(values=lambda rows: [0.25] * len(rows)),
                           [{}, {}, {}]).tolist() == [0.25] * 3


# ---------------------------------------------------------------------------
# the rollout fields and ppo's side (numpy)
# ---------------------------------------------------------------------------

BUTTONS = 2
G = P.GATE_PLAYS + BUTTONS
META = {"format": P.FORMAT, "schema": RS.SCHEMA_V2, "wait_bias": 0.4, "explore": 0.15,
        "tile_temp": 1.0}
TEACH_GATE = (0.7, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0)


def synthetic(episodes=2, steps=3):
    """Rollout columns by hand (test_ppo's): every row draws a wait, each episode's last row
    is done and won; plus the teacher's fields, a gate target TEACH_GATE on every row."""
    n = episodes * steps
    a = {name: np.zeros((n, *RS.V2.shape(name, BUTTONS)), dtype=RS.V2.dtype(name))
         for name in P.OBS_FIELDS}
    a["hand"][:] = [4, 7, 8, 2]
    a["next"][:] = 5
    a["episode"] = np.repeat(np.arange(episodes), steps).astype(np.int32)
    a["step"] = np.tile(np.arange(steps), episodes).astype(np.int32)
    a["done"] = a["step"] == steps - 1
    a["result"] = np.where(a["done"], 1.0, 0.0).astype(np.float32)
    a["hp_frac"] = np.ones((n, 2), dtype=np.float32)
    a["crowns"] = np.zeros((n, 2), dtype=np.int8)
    a["mask"] = np.ones((n, G), dtype=bool)
    a["gate_a"] = np.zeros(n, dtype=np.int16)
    a["logq_gate"] = np.full(n, -0.5, dtype=np.float32)
    a["tile_cands"] = np.full((n, P.CANDS), -1, dtype=np.int16)
    a["tile_a"] = np.full(n, -1, dtype=np.int16)
    a["logq_tile"] = np.zeros(n, dtype=np.float32)
    a["teacher_ok"] = np.ones(n, dtype=bool)
    a["teacher_tile_ok"] = np.zeros(n, dtype=bool)
    a["teacher_gate"] = np.tile(logp(TEACH_GATE), (n, 1))
    a["teacher_tile"] = np.full((n, P.CANDS), T.NEG, dtype=np.float32)
    a["teacher_n"] = np.full(n, 4, dtype=np.int16)
    return a


def test_the_rollout_carries_the_targets_and_ppo_checks_them(tmp_path):
    a = synthetic()
    n = len(a["gate_a"])
    records = [{k: v[i] for k, v in a.items()} for i in range(n)]
    path = P.write_rollout(tmp_path / "teach", records, META, extra=T.FIELDS)
    plain = P.write_rollout(tmp_path / "plain", records, META)
    r = P.load_rollout([path])
    for name, dtype in T.FIELDS.items():
        assert r.arrays[name].dtype == np.dtype(dtype) and len(r.arrays[name]) == n, name
    assert np.array_equal(r.arrays["teacher_gate"], a["teacher_gate"])
    assert "teacher_gate" not in P.load_rollout([plain]).arrays
    assert "teacher_gate" not in P.load_rollout([path, plain]).arrays   # not in every file
    assert "teacher_gate" in P.load_rollout([path, path]).arrays
    with pytest.raises(ValueError):                      # the format's own names are not extra
        P.write_rollout(tmp_path / "x", records, META, extra={"mask": "bool"})
    short = [dict(rec) for rec in records]
    del short[3]["teacher_n"]
    with pytest.raises(P.RolloutError):                  # one row has it: every row must
        P.write_rollout(tmp_path / "y", short, META, extra=T.FIELDS)
    # the CLI refuses --teacher-coef on rollouts without the fields: exit 2 before any model
    assert P.main(["--check", "--data", str(path), "--teacher-coef", "0.5"]) == 0
    assert P.main(["--check", "--data", str(plain), "--teacher-coef", "0.5"]) == 2
    assert P.main(["--check", "--data", str(plain)]) == 0
    m = P.row_masks(r.arrays)
    got = T.ppo_inputs(r.arrays, m.choice, m.tile)
    assert got.gate_ok.all() and not got.tile_ok.any() and int(got.rows.sum()) == n
    assert got.gate.dtype == np.float32 and got.tile.shape == (n, P.CANDS)

    def copy_of():
        return {k: np.array(v, copy=True) for k, v in r.arrays.items()}

    def refused(b, tile_rows=m.tile):
        with pytest.raises(P.RolloutError):
            T.ppo_inputs(b, m.choice, tile_rows)

    b = copy_of()
    del b["teacher_gate"]
    refused(b)
    b = copy_of()
    b["mask"][0, 5] = False                              # mass 0.2 where the student cannot go
    b["teacher_gate"][0] = logp([0.5, 0.1, 0.1, 0.05, 0.05, 0.2, 0.0])
    refused(b)
    b = copy_of()
    b["teacher_gate"][1, 0] = np.log(0.9)                # sums to 1.2
    refused(b)
    b = copy_of()
    b["capped"] = np.ones(n, dtype=bool)                 # a capped row never waits
    refused(b)
    b = copy_of()
    b["teacher_tile_ok"][:] = True
    b["teacher_tile"][:, 0] = 0.0                        # all on a padding candidate
    refused(b, np.ones(n, dtype=bool))
    # a row without a target may hold anything; a narrower tile target is padded
    b = copy_of()
    b["teacher_ok"][2] = False
    b["teacher_gate"][2] = 0.0
    b["teacher_tile"] = b["teacher_tile"][:, :8]
    got = T.ppo_inputs(b, m.choice, m.tile)
    assert got.gate_ok.tolist() == [True, True, False, True, True, True]
    assert got.tile.shape == (n, P.CANDS) and (got.tile == T.NEG).all()


def test_ppo_keeps_the_teacher_off_by_default():
    assert P.PPOConfig().teacher_coef == 0.0
    with pytest.raises(ValueError):
        P.PPOConfig(teacher_coef=-0.1)
    ap = P.build_parser()
    assert P.config_from_args(ap.parse_args(["--data", "x"])).teacher_coef == 0.0
    args = ap.parse_args(["--data", "x", "--teacher-coef", "0.3"])
    assert P.config_from_args(args).teacher_coef == 0.3


# ---------------------------------------------------------------------------
# the loss and the update (torch)
# ---------------------------------------------------------------------------

N_CARDS, N_UNITS = 18, 20
AVAILABLE = RS.ABILITY_FEATS.index("available")
CANNON, SKELETONS = 0, 6


def _torch():
    return pytest.importorskip("torch")


def tiny_model(seed=0):
    torch = _torch()
    from sim import model_tokens as mt
    torch.manual_seed(seed)
    return mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=1, heads=2, schema="v2").eval()


def v2_row(rng, buttons=BUTTONS):
    """An obs_row of schema v2 (test_ppo's), varied by `rng`; button 0 holds card token 7 and
    reads available."""
    row = {"tick": 500}
    for name in RS.V2.model_inputs:
        shape, dtype = RS.V2.shape(name, buttons), RS.V2.dtype(name)
        row[name] = np.zeros(shape, dtype=dtype) if shape else np.dtype(dtype).type(0)
    row["hand"][:] = rng.permutation([4, 7, 8, 2])
    row["next"] = np.int16(5)
    k = int(rng.integers(1, 6))
    row["unit_ids"][:k] = rng.integers(1, N_UNITS, k)
    row["unit_feats"][:k] = rng.random((k, 9))
    row["towers"][:] = rng.random(6)
    row["scalars"][:] = rng.random(7)
    row["opp_seen"][:3] = rng.integers(1, N_CARDS, 3)
    row["opp_last"][:2] = rng.integers(1, N_CARDS, 2)
    row["opp_last_age"][:2] = rng.random(2)
    row["spell_ids"][0] = 3
    row["spell_feats"][0] = rng.random(6)
    row["own_pending"][0] = 7
    row["own_pending_land"][0] = 0.25
    row["ability_card"][0] = 7
    row["ability_state"][0, AVAILABLE] = 1.0
    return row


def legal(slot, card, x_sub, y_sub):
    """Skeletons never have a legal tile (a drawn Skeletons falls back); the Cannon only on even
    world columns; everything else anywhere."""
    if card == SKELETONS:
        return False
    return card != CANNON or (x_sub // 18000) % 2 == 0


def decide_rollout(model, folder, *, seed=0, episodes=3, steps=6):
    """test_ppo's: rows sampled by rs_decide.decide (sampling preset, presses on) through a
    tiny TokenPolicy v2, the red seat; episodes 0 (won) and 1 (lost) end, the last is cut short
    (its last row is not valid)."""
    adapter = RD.TokenPolicyAdapter(model)
    cfg = RD.DecideConfig.sampling(0.4, presses=True)
    rng = np.random.default_rng(seed)
    obs_rng = np.random.default_rng(seed + 100)
    records = []
    for e in range(episodes):
        for s in range(steps):
            row = v2_row(obs_rng)
            hand = [int(t) - 2 for t in row["hand"]]
            elixir = int(obs_rng.choice([0, 1500, 3500, 9000]))
            abilities = [[1, 0, 2, 5, 0]] if s % 2 == 0 else []
            me = player_dict(1, hand=hand, elixir_milli=elixir, abilities=abilities)
            state = state_dict(500, players=[player_dict(0), me])
            d = RD.decide(adapter, row, state, 1, legal, cfg, rng=rng)
            done = e < episodes - 1 and s == steps - 1
            records.append({**{k: row[k] for k in P.OBS_FIELDS}, **P.decision_record(d),
                            "episode": e, "step": s, "done": done,
                            "result": (1.0 if e == 0 else -1.0) if done else 0.0,
                            "hp_frac": [1.0 - 0.05 * s, 1.0], "crowns": [0, 0]})
    path = P.write_rollout(Path(folder) / "rollout.npz", records, P.rollout_meta(cfg))
    return P.load_rollout([path])


def student(model, rollout):
    """The student's (gate log q, tile log p) on every row of `rollout`, as the update
    computes them (ppo.log_probs on the rollout's masks), and the action tensors."""
    torch = _torch()
    a = rollout.arrays
    m = P.row_masks(a)
    idx = np.arange(rollout.n)
    act = P.action_tensors(a, idx, "cpu", m.tile, P.card_ids(a["hand"], a["gate_a"]))
    with torch.no_grad():
        out = model.outputs(P.obs_batch(a, idx, "cpu"), card_ids=act["card"])
    lq, lpt = P.log_probs(out["gate"], out["tile"], act, P.Sampling.from_meta(rollout.meta))
    return lq, lpt, act


def test_kl_rows_by_hand_and_zero_for_the_student_itself():
    torch = _torch()
    ln = math.log
    lq = torch.tensor([[ln(0.5), ln(0.25), ln(0.25), T.NEG]] * 3)
    mask = torch.tensor([[True, True, True, False]] * 3)
    tg = torch.tensor([[ln(0.25), ln(0.25), ln(0.5), T.NEG]] * 3)
    lpt = torch.tensor([[ln(0.2), ln(0.3), ln(0.5)]] * 3)
    tt = torch.tensor([[ln(0.5), ln(0.5), T.NEG]] * 3)
    cmask = torch.ones(3, 3, dtype=torch.bool)
    gate_ok = torch.tensor([True, False, True])
    tile_ok = torch.tensor([True, True, False])
    kl = T.kl_rows(lq, lpt, tg, tt, gate_ok, tile_ok, mask, cmask)
    g = 0.25 * ln(2)                                     # 0.25 ln(1/2) + 0.5 ln 2
    t = 0.5 * ln(0.5 / 0.2) + 0.5 * ln(0.5 / 0.3)
    assert kl.tolist() == pytest.approx([g + t, t, g], abs=1e-6)
    same = T.kl_rows(lq, lpt, lq, lpt, gate_ok, tile_ok, mask, cmask)
    assert same.abs().max().item() < 1e-7
    mb = {"teacher_gate": tg, "teacher_tile": tt, "teacher_gate_ok": gate_ok,
          "teacher_tile_ok": tile_ok, "mask": mask, "cmask": cmask,
          "teacher_rows": torch.tensor([1.0, 1.0, 0.0])}
    mean, rows = T.kl_term(lq, lpt, mb)
    assert rows.item() == 2 and mean.item() == pytest.approx((g + t + t) / 2, abs=1e-6)


def test_the_batched_critic_equals_row_by_row():
    _torch()
    adapter = RD.TokenPolicyAdapter(tiny_model())
    rng = np.random.default_rng(3)
    rows = [v2_row(rng) for _ in range(5)]
    single = [math.tanh(adapter.heads(adapter.batch(r))[1][1] / 2) for r in rows]
    assert T.policy_values(adapter, rows).tolist() == pytest.approx(single, abs=1e-4)


def test_an_update_with_the_student_as_teacher_measures_kl_0(tmp_path):
    """The teacher's targets are the student's own distributions (the gate on every drawn row,
    the tile on every play that ran): at lr 0 the KL term reads 0 and nothing changes. Off
    (teacher_coef 0) the update reads no teacher field and reports none; on, a rollout without
    the fields is refused."""
    torch = _torch()
    model = tiny_model()
    rollout = decide_rollout(model, tmp_path)
    a = rollout.arrays
    n = rollout.n
    m = P.row_masks(a)
    lq, lpt, _ = student(model, rollout)
    a.update(teacher_ok=m.choice.copy(), teacher_tile_ok=m.tile.copy(),
             teacher_gate=lq.numpy().astype(np.float32),
             teacher_tile=lpt.numpy().astype(np.float32),
             teacher_n=np.full(n, 3, dtype=np.int16))
    valid = np.ones(n, dtype=bool)
    valid[-1] = False                                    # episode 2 is cut short
    cfg = P.PPOConfig(lr=0.0, critic_lr=0.0, epochs=1, minibatch=7, precision="fp32",
                      teacher_coef=1.0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    stats = P.update(model, copy.deepcopy(model), rollout, cfg, P.PPOState.new(cfg))
    assert stats["teacher_coef"] == 1.0
    assert stats["rows_teacher"] == int((m.choice & valid).sum()) > 0
    assert stats["rows_teacher_tile"] == int((m.tile & valid).sum())
    assert abs(stats["kl_teacher"]) < 1e-5 and abs(stats["kl_teacher_last"]) < 1e-5
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
    off = P.update(model, copy.deepcopy(model), rollout, cfg.replace(teacher_coef=0.0),
                   P.PPOState.new(cfg))
    assert "kl_teacher" not in off and "rows_teacher" not in off
    bare = P.Rollout({k: v for k, v in a.items() if not k.startswith("teacher_")}, rollout.meta)
    with pytest.raises(P.RolloutError):
        P.update(model, copy.deepcopy(model), bare, cfg, P.PPOState.new(cfg))


def test_the_teacher_term_pulls_the_policy_toward_the_teacher(tmp_path):
    """Every drawn row's teacher puts all of the gate on the wait, and every row is flagged
    fallback, so there is no policy-gradient term; the reference is the model itself (its KL
    terms have no gradient at the start) and the entropy bonus is off. One Adam step: the
    student's KL to the teacher, as the update measured it before the step, goes down."""
    torch = _torch()
    model = tiny_model(2)
    rollout = decide_rollout(model, tmp_path)
    a = rollout.arrays
    n = rollout.n
    m = P.row_masks(a)
    valid = np.ones(n, dtype=bool)
    valid[-1] = False
    gate = np.full(a["mask"].shape, T.NEG, dtype=np.float32)
    gate[:, 0] = 0.0                                     # log 1 on the wait
    a.update(fallback=np.ones(n, dtype=bool), teacher_ok=m.choice.copy(),
             teacher_tile_ok=np.zeros(n, dtype=bool), teacher_gate=gate,
             teacher_tile=np.full((n, P.CANDS), T.NEG, dtype=np.float32),
             teacher_n=np.full(n, 2, dtype=np.int16))
    ok = torch.from_numpy(m.choice & valid)

    def kl_now():
        lq, lpt, act = student(model, rollout)
        rows = T.kl_rows(lq, lpt, torch.from_numpy(gate), torch.from_numpy(a["teacher_tile"]),
                         ok, torch.zeros(n, dtype=torch.bool), act["mask"], act["cmask"])
        return float(rows[ok].mean())

    start = kl_now()
    cfg = P.PPOConfig(lr=1e-4, critic_lr=0.0, epochs=1, minibatch=1000, ent_coef=0.0,
                      precision="fp32", teacher_coef=1.0)
    stats = P.update(model, copy.deepcopy(model), rollout, cfg, P.PPOState.new(cfg))
    assert stats["rows_pg"] == 0 and stats["rows_teacher"] == int(ok.sum()) > 0
    assert stats["kl_teacher"] == pytest.approx(start, abs=1e-5)
    assert kl_now() < start
