r"""One decision function for every seat the token policy plays: rs_eval, the live runner, RL.

    import rs_decide as RD                       # tools/ on sys.path, as the tools import
    policy = RD.TokenPolicyAdapter(model)        # model from rs_checkpoint.load_policy(ckpt, cfg)
    dcfg = RD.DecideConfig.live(RD.wait_bias_for(ckpt), guard_cards=RD.card_ids(cat, RD.GUARD_CARD_NAMES),
                                presses=RD.presses_allowed(meta))     # E15: v2 buttons, if trained
    legality = RD.TowersOnlyLegality(engine_cfg, catalogue=cat)
    d = RD.decide(policy, row, state, team, legality, dcfg, rng=rng, pending=pending, tick=now, dt=dt)
    d.command            # (team, slot, x_sub, y_sub) for Battle.step / the tap, or None (wait);
                         # a press: (team, HAND_SIZE + button, 0, 0)
    d.value              # the critic's raw [outcome, won logit] (v2 / R01), () without one
    d.suggest_kwargs()   # ControlState.suggest(**...) (scripts/vision/control.py), also in MANUAL
    # v2 rows: obs_row(..., **RD.v2_row_kwargs(opp_elixir,
    #                  RD.pending_inputs(pending, now, RD.obs_delay_ticks(meta))))

Why (V06): rs_eval, live_play and rs_selfplay each had their own copy of the gate (RSAgent.act,
live_play.py:505-586, rs_selfplay.SPAgent), and they had drifted: only live play blocked the
slots of its own plays in flight, only eval read the engine's pending_cost, nobody tried the next
card when the chosen one had no legal tile (missed issue 3: under a command delay RoyaleSim keeps
a pending card in hand and refuses it with CARD_PENDING, so argmax picked it, all 60 tiles
failed and the decision was lost). One function, three presets.

THE DECISION, in order (decide):
  1. gate logits [wait, slot 0..3] from the policy (a v2 model: [wait, slot 0..3, button
     0..B-1]); wait += cfg.wait_bias, log(1 / wait_keep) with wait_keep from the run's args.json
     (wait_bias_for): training kept only that share of the wait rows, so the raw wait logit is
     low by exactly that much.
  2. masks: a slot is out when it is empty (no card, hand -1), or it costs more than the SPENDABLE
     elixir, elixir_milli - 1000 x the elixir of our plays in flight (pending), or (live) it
     holds a card in flight, or (live) hand_costs says no play resolves (-1, a Mirror with
     nothing to copy). A missing hand_costs entry costs 99, as before. A button is out unless
     cfg.presses and button_mask let it in (V2 CHECKPOINTS below).
  3. p_wait = softmax(gate)[0]; the bot waits with probability p_wait ** (dt / 20): the same wait
     rate per second at any decision cadence dt (rs_eval decides every 10 ticks: exponent 0.5,
     as it always did; live, dt = ticks since the last decision, cadence_dt). One rng.random()
     is drawn per decision, also when it is not needed, so a seeded run replays exactly.
     With the cap rule (live) the bot never waits while spendable >= 9500 and a card is
     playable. Nothing playable and no button pressable: wait ("no_card").
  4. the argmax over the unmasked slots and buttons. A slot: its 576 tile logits are tried best
     first, the first 60 at most, each at the tile centre: the first tile legality accepts is
     the play. The lane guard (live, cards in guard_cards: the Cannon) skips tiles outside the
     threatened lane but for the centre columns 7-10 (live_play.py:65-90). blocked_tiles
     (bool[576], e.g. geometry.ability_button_mask(n, "model").reshape(-1)) are skipped too.
     A button: a PRESS, command (team, HAND_SIZE + button, 0, 0) (royalegym protocol: x and y
     are not read), no tile and no legality call.
  5. (live) no legal tile: the next slot or button by gate logit, and so on ("fallback");
     none: wait ("no_tile").

PRESETS (DecideConfig):
  legacy()    exactly RSAgent.act before V06: pending_cost from the state, no pending-slot mask,
              no fallback, no cap rule, no lane guard. tools/rs_eval.py's default
              (tests/pipeline/test_rs_decide.py replays the old code against it, decision for
              decision and draw for draw).
  live()      the live runner and rs_eval --decide live: pending-slot mask, fallback,
              unplayable mask, cap rule 9500, lane guard on guard_cards.
  sampling()  RL rollouts: slot and tile are SAMPLED, not argmaxed, as rs_selfplay samples them:
              the gate as a distribution (wait p_wait ** (dt / 20), the rest in proportion to the
              slot softmax among playable slots), the tile among the 8 best legal tiles of the
              first 60 plus 8 random legal ones, (1 - 0.15) softmax + 0.15 uniform. The decision
              carries gate_q, gate_a, logq_gate, tile_cands, tile_a, logq_tile and the mask.
              A fallback play (the sampled slot had no legal tile) is flagged "fallback"; the RL
              learner decides what to do with it. No cap rule, no lane guard.
  Every preset takes presses= (default False: every button masked, as before E15).

DECISION RULES (DecideConfig.wait_floor, stray_cards; both OFF by default, so every preset
decides as before, draw for draw; tests/pipeline/test_decide_rules.py). Two hand-written rules
from the over-defence audit of the owner's live friendlies of 2026-11-01, switched on per seat
for an A/B in the simulator first (rs_eval --wait-floor / --stray-guard, rs_match and
rs_league_eval players' wait_floor= / stray_guard=, scripts/decide_ab.ps1). The argmax presets
only: the sampling preset refuses them, so RL rollouts and their probabilities never change.
  wait floor  (rule 6, wait_floor = P, the A/B's 0.95) the bot waits whenever the model's own
              p_wait (per 20 ticks, the gate after the wait bias and the masks: what the
              decision records) is at least P, i.e. the model rates playing in the next second
              under 1 - P. Without it, the draw u >= p_wait ** (dt / 20) still plays now and
              then: in the audit about one play a game came from p_wait >= 0.95 (game 1 t2178,
              a Cannon at p_wait 0.986, a 0.3% draw over 12 ticks, which left the bot empty).
              The cap rule wins: the floor never holds a playable card at or above the cap
              (cfg.cap_milli; 9500 when the preset has no cap rule too), so elixir never leaks
              to it. The one random() is still drawn first; a drawn wait keeps its reason
              ("wait"), and a wait the floor made is reason "floor" (R_FLOOR): a play the draw
              would have tried, so at most the plays it removed (one of them might have found
              no tile and waited "no_tile" anyway).
  stray guard (rule 7, stray_cards = card_ids(catalogue, STRAY_CARD_NAMES): Skeletons and Ice
              Spirits, the 1-elixir troops of the bot's Hog 2.6) while an enemy THREAT is on
              the board, a tile of such a card on our half farther than STRAY_REACH_MT (5.5
              tiles) from every threat's PATH is skipped, as the lane guard skips its tiles:
              the model's next tile is tried. When the guard turns down every legal tile of the
              card in the first 60, the decision WAITS (no_tile, stray > 0) in every preset: the
              guard holds the play, it never hands it to another card (the live fallback would
              otherwise go on to, say, a 4-elixir Hog Rider at the bridge: the over-commit the
              rule is there to stop). Episode 9 of the audit (game 1 t1587): Skeletons at
              centre-river (own tile (8, 17)) while a P.E.K.K.A walked down the other lane at
              own (15.3, 13.5); with nothing near, two of the three walked up the far lane and
              died to the enemy tower. A threat: an enemy TROOP (as the lane guard's
              threat_lane: no building, no crown tower) with hp left at own-frame y >=
              GUARD_DEEP_MT (13 tiles), the lane guard's line; the P.E.K.K.A was there, 1.5
              tiles short of the river. An enemy building alone (a Cannon at their river edge)
              walks nowhere, so it switches nothing on. A threat's path: from where it is, for
              a ground troop not yet past the river by its lane's bridge exit (own x 3.5 or
              14.5, y 17), to our princess tower in its lane (the king once that one is down);
              and from the bridge exit (or where it is) to each building of ours on our half,
              which pulls it. 5.5 tiles is half the 11 tiles between the lanes: beyond it a
              troop is nearer the other lane, the one it walks when it finds nothing to hit
              (own column 8 is 6.0 tiles from a right-lane path, column 9 is 5.0). Kept
              whatever the distance: no threat at all (back-of-king cycle plays, any tile), the
              enemy half (own rows below 17: offence, the pocket of a fallen princess), the
              bridge tiles of a lane where a troop of ours is at our bridge or beyond (own
              columns 2-4 or 13-15 of rows 17-18, with a troop of ours in that lane at own y <
              STRAY_PUSH_MT, 19 tiles: a bridge play with or behind the Hog; without one, a
              far-lane bridge play is episode 9's walk up the other lane, and the threat's own
              bridge is near its path anyway) and tiles next to one of our living crown towers
              (within STRAY_TOWER_MT, 1 tile, of its body: the back of the king, a princess
              tower's side). Ice Golem (2) is left out, as the lane guard leaves it out: pros
              pull with it, far from the path on purpose; the lane guard's Cannon is a
              building. Decision.stray counts the legal tiles the guard turned down (the guard
              is asked after legality): a play with stray > 0 is the same card the guard moved
              to another tile, a wait with stray > 0 one the guard made.

PENDING (plays we sent that have not run): pending=None reads them from the state, as RoyaleSim
reports them under a command delay (players[team]["pending"] rows [kind, what, x, y, ticks_left,
cost] and "pending_cost", the elixir they speak for): that is rs_eval. The live runner passes
its own list of Pending(card, cost, slot, tick), one per tap, pruned each frame with
prune_pending (live_play's rule: drop a tap once its card has left the slot it was tapped
from, the client's hand changing when the play runs, or after 60 ticks). A press tap is
Pending(-1, cost, tick=, button=k): it blocks button k and holds its cost until it is 60 ticks
old or (prune_pending's abilities=) the button no longer reads available.

ELIXIR CAP (live only: ElixirCap, decide's elixir_cap=). The card leaves its slot 2-4 ticks
after the tap, but the HUD's elixir reading can lag the payment by up to 20+ ticks, and while
the counter animates it reads high (owner's first --tap friendly, 2026-11-01: Cannon at tick
610 with 4.26 elixir, slot empty from 614, the HUD read 3.99 then 3.00 until 619 and the true
1.43 from 621; prune_pending had dropped the Cannon at 614, so at 619 the bot tapped a Log
with 1.4 elixir and the game refused it). So once a tap of ours has been paid (its card left
the slot, prune_pending's rule), the runner keeps an upper bound on our elixir for a while:

    cap = spendable at the tap - 1000 x cost + the most regen possible since the tap

(the regen of the law's rate, 1x / 2x / 3x by the tick, counted as if dated 20 ticks later:
the clock reads whole seconds and the rate only rises), and decide uses min(elixir_milli,
cap) in place of elixir_milli. "Spendable at the tap" is the decision's own spendable_milli
with the costs it held for the taps then in flight added back: it already holds the earlier
taps' caps, and each earlier tap still in flight is linked to the new one, its cost taken off
the new cap once that tap is paid (a refused one is never paid: decide holds its cost while
it is pending, so it is not taken twice), so plays in a row stack. A cap ends once the HUD
has read at most 50 milli above it on two frames in a row (the HUD has caught up), or 30
ticks after its play was paid (a timeout: a refused tap, a misread base), or when its tap was
never paid (60 ticks, as prune_pending). Its card read back in its slot within 20 ticks of
the payment (a one-frame misread of the slot) makes the tap unpaid again, its cap off; a
played slot reads empty for ~22 ticks and then shows another card. Why +50 on two frames:
on that friendly the settled HUD read within -50..+40 milli of the cap; at +500 the cap of a
Log paid at tick 3069 ended on a reading of 3.99 (cap 3.66) and the HUD's 4.01 of 3070-3073
let a Fireball in with ~3.5 real elixir; one low reading inside a stale window ended a cap
at once, and +200 still let a stale reading 0.2 above the real bar through. Replayed on that
friendly, +50 on two frames ends all 33 caps 1-4 ticks later than +200 on one, none times
out. rs_eval, rs_match and RL never pass elixir_cap: their decisions are exactly as before.

LEGALITY (anything with bind(state, team) -> check(slot, card, x_sub, y_sub) -> bool):
  BattleLegality(battle)   check_deploy on the battle itself (rs_eval: the match battle).
  TowersOnlyLegality(cfg)  the live case: a RoyaleSim battle (rs_engine.make_battle, no command
                           delay) reset with nothing but the crown towers (alive 1 hp, destroyed
                           0), our hand dealt into its first four slots, start_tick 100 (past the
                           engine's 90-tick deploy lockout) and full elixir; check_deploy on it.
                           Cached by (team, hand, destroyed towers): an evicted battle is reset
                           again rather than rebuilt. It holds no units, so unlike the rebuilt
                           look-ahead (rs_eval.RSAgent.rebuilt) nothing seen on screen can make
                           it fail; the cost is that it does not know about units or buildings in
                           the way, which the client then refuses or moves.
  PureLegality(catalogue)  no engine at all (royalesim missing, or it refused the reset): own
                           half, the pockets that destroyed enemy princess towers open (the
                           NoDeploySize rects, royalegym DeployRules), Miner / Goblin Drill
                           anywhere off the water, spells anywhere. CONSERVATIVE: legal here
                           means legal in the engine (the engine-marked test checks every tile);
                           the river rows and tower bodies are never offered.
  TowersOnlyLegality falls back to PureLegality when given a catalogue (or fallback=).

V2 CHECKPOINTS (E15; sim/model_tokens.py, tools/rs_schema.py). A checkpoint whose meta "schema"
is rs_rows/v2 has a TokenPolicy(schema="v2"); a legacy one (no "schema", or rs_rows/legacy) is
decided on exactly as before, call for call and draw for draw (tests/pipeline/test_rs_decide.py).
  batch    TokenPolicyAdapter's own batch is the model's: every rs_schema.V2.model_inputs column
           for a v2 model (row_batch(row, torch, schema), the batch model_tokens.row_batch
           builds), the legacy keys otherwise.
  buttons  the gate entries after the four slots, one per ability button (B from the row's
           ability_card). The model already puts an unavailable button at -1e4. decide lets a
           button in only with cfg.presses (default False, every preset) and when button_mask
           does: players[team]["abilities"] (royalegym AbilityRow: available, spent, cost, ...)
           says it is available and not spent, its cost fits the spendable elixir, and (with
           the pending mask) no press of it is in flight (the engine refuses that CARD_PENDING:
           the state's pending rows of kind 1, or a Pending with button >= 0). Callers set
           cfg.presses from presses_allowed(meta): only a v2 model trained on press-labelled
           rows (meta train_rows.press_rows "trained", sim/train_tokens.py) has trained button
           logits; one trained on rows built with --presses off has not. The bot's own deck has
           no hero or champion, so live every button stays masked; RL and other decks press.
  pending  obs_row's `pending` (own_pending, own_pending_land): pending_inputs(pending, now,
           delay) from the live runner's taps, delay = obs_delay_ticks(meta), the tap-to-run
           delay the decision-time rows were built with (rs_rows --obs-delay; 0: landing-time
           rows, which never show a pending play, so nothing is given); engine_pending_inputs
           (state, team, lead) from the state's pending rows (rs_eval.RSAgent.own_pending's
           rule). v2_row_kwargs gives obs_row's v2 keywords as RSAgent.v2_kw does.
  critic   a model with a critic (value2, R01 / R02) puts its raw output [10 s outcome, won
           logit] into Decision.value (win_p: the sigmoid of the second), from the same
           forward pass as the gate (TokenPolicyAdapter.heads); () without a critic.

Pure Python and numpy; royalesim only inside TowersOnlyLegality, torch only inside
TokenPolicyAdapter, rs_elixir (pure Python, the law) only inside ElixirCap. Never imports
scripts/gamestate.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import math
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:      # also when imported as tools.rs_decide
    sys.path.insert(0, str(ROOT / "tools"))
import rs_engine  # noqa: E402
import rs_schema  # noqa: E402  (pure Python: the v2 batch columns)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DECIDE = 10                  # ticks between rs_eval decisions (0.5 s): the default cadence
WAIT_REF_TICKS = 20          # p_wait is a wait probability per this many ticks (exponent dt / 20)
DEFAULT_WAIT_KEEP = 0.35     # RSAgent's default when the run has no args.json
TOP_TILES = 60               # tiles tried per slot, best first
GRID_W, GRID_H = 18, 32
TILES = GRID_W * GRID_H
SUBTILE_PER_MILLITILE = 18   # royalesim.SUBTILE_PER_MILLITILE
HAND_SIZE = 4
GATE_PLAYS = 1 + HAND_SIZE   # gate entries [wait, slot 0..3]; any after them are ability buttons
MASKED = -1e9                # a masked gate logit
UNPLAYABLE = -1e8            # the best slot logit below this: nothing to play
NO_COST = 99                 # the price of a slot hand_costs does not cover (as RSAgent had it)
CAP_ELIXIR = 9500            # milli; never sit at full elixir with a playable card (live_play)
PENDING_TICKS = 60           # a tap not landed after 3 s is assumed lost (live_play)
FULL_ELIXIR = 10_000
# The live elixir cap after our own plays (ElixirCap; module doc, ELIXIR CAP).
ELIXIR_CAP_TOL = 50          # milli: a cap ends once the HUD reads at most this above it ...
ELIXIR_CAP_FRAMES = 2        # ... on this many frames in a row (the settled HUD's noise)
ELIXIR_CAP_TICKS = 30        # a cap ends this many ticks after its play was paid at the latest
ELIXIR_CAP_SLACK = 20        # ticks: regen counted as if dated this much later (1 s clock)
ELIXIR_CAP_BACK = 20         # ticks: a card read back in its slot this soon after it left was not paid
BACK_FRAMES = 2              # ... when it reads back on this many frames in a row (one is a misread)
LEGAL_START_TICK = 100       # the legality battle's tick: past match.DEPLOY_LOCKOUT_TICKS (90)

# Wrong-lane guard (a hand-written rule, live_play.py:65-90). Live logs of 2026-10-09: 24% of
# defensive plays within 3 s of an opponent's play went to the other lane. When one lane clearly
# holds the threat, these cards may only go to that lane or the centre columns.
GUARD_CARD_NAMES = frozenset({"Cannon"})   # Ice Golem left out: pros pull with it 17% of the time
GUARD_MIN_HP = 1200          # enemy hp past their deep half before the rule acts
GUARD_SHARE = 0.75           # share of that hp in one lane for it to be "the" lane
GUARD_CENTRE = (7, 10)       # own-frame tile columns always allowed
GUARD_DEEP_MT = 13_000       # an enemy at own-frame y below this is still deep on their side

# The decision rules (module doc, DECISION RULES), both off unless a DecideConfig says so.
WAIT_FLOOR_AB = 0.95         # rule 6's floor in the A/B (scripts/decide_ab.ps1)
STRAY_CARD_NAMES = frozenset({"Skeletons", "IceSpirits"})   # rule 7: the deck's 1-elixir troops
STRAY_REACH_MT = 5_500       # a tile farther than this from every threat's path is a stray
STRAY_TOWER_MT = 1_000       # tiles within this of one of our living crown towers' body are kept
STRAY_OWN_ROW = 17           # own-frame rows from here are our half (below: offence, kept)
STRAY_BRIDGE_COLS = ((2, 4), (13, 15))   # own-frame columns of the two bridge plays ...
STRAY_BRIDGE_ROWS = (17, 18)             # ... in our first two rows: kept with a push
STRAY_PUSH_MT = 19_000       # a troop of ours at own y below this (our bridge or beyond)
                             # keeps its lane's bridge tiles
LANE_X_MT = (3_500, 14_500)  # own-frame x of the two lanes: the bridges and the princess towers
BRIDGE_EXIT_MT = 17_000      # own-frame y where a ground troop leaves its lane's bridge
LANE_SPLIT_MT = 9_000        # own-frame x below which a unit is in the left lane (threat_lane)
KIND_BUILDING = 1            # entity kind (threat_lane, rs_rows: 1 = building)

# Sampling (rs_selfplay.TOP_TILES, EXTRA_TILES, EXPLORE, TILE_TEMP; 120 random tiles scanned).
SAMPLE_TOP, SAMPLE_EXTRA, SAMPLE_SCAN, SAMPLE_EXPLORE, SAMPLE_TEMP = 8, 8, 120, 0.15, 1.0

MODE_LEGACY, MODE_LIVE, MODE_SAMPLE = "legacy", "live", "sample"
MODES = (MODE_LEGACY, MODE_LIVE, MODE_SAMPLE)

PLAY, WAIT, PRESS = "play", "wait", "press"
R_WAIT, R_NO_CARD, R_NO_TILE = "wait", "no_card", "no_tile"
R_FLOOR = "floor"            # a wait the wait floor made (module doc, DECISION RULES)

#: An abilities row's columns by position (royalegym protocol.AbilityRow / ability_row, and
#: tools/rs_presses._row): a column the engine does not send reads its default.
ABILITY_COLUMNS = ("available", "spent", "cost", "card_id", "cooldown_ticks")
ABILITY_DEFAULTS = (0, 0, 0, -1, -1)
#: Pending row kinds (players[team]["pending"]): a play, and a button press (what = HAND_SIZE + k).
PENDING_PLAY, PENDING_PRESS = 0, 1
#: The checkpoint meta of a model whose ability buttons trained (sim/train_tokens.py
#: TRAIN_ROWS_KEY, "press_rows" == PRESS_TRAINED); a test checks the names against it.
TRAIN_ROWS_KEY, PRESS_ROWS_KEY, PRESS_TRAINED = "train_rows", "press_rows", "trained"
#: A row's card token is the catalogue id + this (rs_rows: 0 pads, 1 unknown).
TOKEN_OFFSET = 2

#: Entity field name -> column (the layout rs_engine checks royalesim against; no engine needed).
F: dict[str, int] = {name: i for i, name in enumerate(rs_engine.ENTITY_FIELDS_EXPECTED)}


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def to_world(ox, oy, team):
    """Acting player's own frame (millitiles) -> RoyaleSim world millitiles (rs_eval.to_world)."""
    return (ox, 32_000 - oy) if team == 0 else (18_000 - ox, oy)


def own(x_sub, y_sub, team, sub=SUBTILE_PER_MILLITILE):
    """RoyaleSim sub-tiles -> the acting player's frame, millitiles (rs_rows.own)."""
    x, y = x_sub / sub, y_sub / sub
    return (x, 32_000 - y) if team == 0 else (18_000 - x, y)


def tile_centre_world(tile: int, team: int) -> tuple[int, int]:
    """World millitiles of the centre of model tile `tile` (ty * 18 + tx, own frame)."""
    tx, ty = int(tile) % GRID_W, int(tile) // GRID_W
    return to_world(tx * 1000 + 500, ty * 1000 + 500, team)


def cadence_dt(now: int, last: int | None, cap: int = DECIDE) -> int:
    """The ticks a decision at `now` stands for (live_play): min(cap, max(1, now - last));
    `cap` for the first decision."""
    if last is None:
        return int(cap)
    return min(int(cap), max(1, int(now) - int(last)))


# ---------------------------------------------------------------------------
# the run's wait bias
# ---------------------------------------------------------------------------

def wait_keep_for(ckpt: Any, default: float = DEFAULT_WAIT_KEEP) -> float:
    """The share of wait rows the checkpoint trained on: "wait_keep" in the args.json next to
    it (sim/train_tokens.py writes it), else `default`. Read as RSAgent read it: a missing
    file, a missing key or a value float() refuses keeps the default."""
    if ckpt is None or isinstance(ckpt, Mapping):
        return default
    args_json = Path(ckpt).with_name("args.json")
    if args_json.exists():
        try:
            return float(json.loads(args_json.read_text(encoding="utf-8"))["wait_keep"])
        except (KeyError, ValueError):
            pass
    return default


def wait_bias_for(ckpt: Any, default_keep: float = DEFAULT_WAIT_KEEP) -> float:
    """log(1 / wait_keep): what decide adds to the wait logit (DecideConfig.wait_bias)."""
    return math.log(1 / wait_keep_for(ckpt, default_keep))


def is_v2_meta(meta: Mapping[str, Any] | None) -> bool:
    """Whether a checkpoint meta is a v2 model's: its "schema" is rs_rows/v2 (or "v2"), as
    sim/model_tokens.model_schema and rs_eval.checkpoint_v2 read it; absent: legacy."""
    name = (meta or {}).get("schema")
    return isinstance(name, str) and name.strip() in (rs_schema.SCHEMA_V2, "v2")


def presses_allowed(meta: Mapping[str, Any] | None) -> bool:
    """Whether decide may press for the checkpoint `meta` describes (DecideConfig.presses): a v2
    model whose rows were press-labelled, meta "train_rows" "press_rows" == "trained"
    (sim/train_tokens.py). A v2 model trained on rows built with --presses off has untrained
    button logits, and a legacy model has no buttons."""
    meta = meta or {}
    rows = meta.get(TRAIN_ROWS_KEY)
    return (is_v2_meta(meta) and isinstance(rows, Mapping)
            and rows.get(PRESS_ROWS_KEY) == PRESS_TRAINED)


def obs_delay_ticks(meta: Mapping[str, Any] | None) -> int:
    """The tap-to-run delay the checkpoint's rows were built with (vocab.json "obs_delay"
    "ticks", tools/rs_rows.py --obs-delay; copied into the meta): what pending_inputs takes.
    0 when absent, unreadable or negative (landing-time rows)."""
    entry = (meta or {}).get("obs_delay")
    if not isinstance(entry, Mapping):
        return 0
    try:
        ticks = int(entry.get("ticks") or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, ticks)


def card_ids(catalogue: Sequence[Sequence[Any]], names: Iterable[str]) -> frozenset[int]:
    """Catalogue ids (catalogue_json rows, [0] = name) of the cards called `names`."""
    want = set(names)
    return frozenset(i for i, row in enumerate(catalogue) if row and row[0] in want)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DecideConfig:
    """How decide() decides. DecideConfig() is legacy(); see the module doc for the presets."""

    wait_bias: float = math.log(1 / DEFAULT_WAIT_KEEP)
    dt: int = DECIDE
    top_tiles: int = TOP_TILES
    mask_pending: bool = False       # block the slots of our plays in flight
    fallback: bool = False           # no legal tile: try the next slot
    mask_unplayable: bool = False    # block slots hand_costs prices at -1
    cap_milli: int | None = None     # cap rule: no wait at or above this spendable elixir
    guard_cards: frozenset[int] = frozenset()   # lane guard on these card ids (empty: off)
    sample: bool = False
    sample_top: int = SAMPLE_TOP
    sample_extra: int = SAMPLE_EXTRA
    sample_scan: int = SAMPLE_SCAN
    explore: float = SAMPLE_EXPLORE
    tile_temp: float = SAMPLE_TEMP
    name: str = MODE_LEGACY
    presses: bool = False            # E15: ability buttons may be pressed (presses_allowed)
    # The decision rules (module doc, DECISION RULES): off unless set (apply_rules).
    wait_floor: float | None = None  # rule 6: wait at p_wait >= this (None: off)
    stray_cards: frozenset[int] = frozenset()   # rule 7, the stray guard, on these cards

    def __post_init__(self) -> None:
        if self.wait_floor is not None:
            object.__setattr__(self, "wait_floor", check_wait_floor(self.wait_floor))
        if not isinstance(self.stray_cards, frozenset):
            object.__setattr__(self, "stray_cards", frozenset(int(c) for c in self.stray_cards))
        if self.sample and self.has_rules:
            raise ValueError("the decision rules (wait_floor, the stray guard) are for the "
                             "argmax presets: the sampling preset (RL rollouts) refuses them")

    @property
    def has_rules(self) -> bool:
        """Whether a decision rule is on (module doc, DECISION RULES)."""
        return self.wait_floor is not None or bool(self.stray_cards)

    @classmethod
    def legacy(cls, wait_bias: float = math.log(1 / DEFAULT_WAIT_KEEP), *, dt: int = DECIDE,
               presses: bool = False) -> "DecideConfig":
        """RSAgent.act before V06, exactly (with presses=True, also the v2 buttons)."""
        return cls(wait_bias=float(wait_bias), dt=int(dt), name=MODE_LEGACY,
                   presses=bool(presses))

    @classmethod
    def live(cls, wait_bias: float = math.log(1 / DEFAULT_WAIT_KEEP), *,
             guard_cards: Iterable[int] = (), cap_milli: int | None = CAP_ELIXIR,
             dt: int = DECIDE, presses: bool = False) -> "DecideConfig":
        """The live runner: pending-slot mask, fallback, unplayable mask, cap rule, lane guard
        on `guard_cards` (card_ids(catalogue, GUARD_CARD_NAMES); empty turns it off)."""
        return cls(wait_bias=float(wait_bias), dt=int(dt), mask_pending=True, fallback=True,
                   mask_unplayable=True, cap_milli=cap_milli,
                   guard_cards=frozenset(int(c) for c in guard_cards), name=MODE_LIVE,
                   presses=bool(presses))

    @classmethod
    def sampling(cls, wait_bias: float = math.log(1 / DEFAULT_WAIT_KEEP), *, dt: int = DECIDE,
                 **changes: Any) -> "DecideConfig":
        """RL rollouts: sampled slot and tile, with the live masks and fallback, no cap rule
        and no lane guard (unless `changes` say otherwise; presses=True lets the buttons in)."""
        base = cls(wait_bias=float(wait_bias), dt=int(dt), mask_pending=True, fallback=True,
                   mask_unplayable=True, sample=True, name=MODE_SAMPLE)
        return base.replace(**changes) if changes else base

    @classmethod
    def for_mode(cls, mode: str, wait_bias: float, *, guard_cards: Iterable[int] = (),
                 dt: int = DECIDE, presses: bool = False) -> "DecideConfig":
        """The preset called `mode` (one of MODES)."""
        if mode == MODE_LEGACY:
            return cls.legacy(wait_bias, dt=dt, presses=presses)
        if mode == MODE_LIVE:
            return cls.live(wait_bias, guard_cards=guard_cards, dt=dt, presses=presses)
        if mode == MODE_SAMPLE:
            return cls.sampling(wait_bias, dt=dt, presses=bool(presses))
        raise ValueError(f"decide mode: one of {', '.join(MODES)}, got {mode!r}")

    def replace(self, **changes: Any) -> "DecideConfig":
        if "guard_cards" in changes:
            changes["guard_cards"] = frozenset(int(c) for c in changes["guard_cards"])
        return dataclasses.replace(self, **changes)


def check_wait_floor(value: Any) -> float:
    """A wait floor as a float above 0 and at most 1 (ValueError otherwise)."""
    try:
        floor = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"wait floor: a number above 0 and at most 1, got {value!r}") from None
    if isinstance(value, bool) or not 0.0 < floor <= 1.0:      # NaN fails the comparison too
        raise ValueError(f"wait floor: above 0 and at most 1, got {value!r}")
    return floor


def apply_rules(cfg: DecideConfig, catalogue: Sequence[Sequence[Any]] = (), *,
                wait_floor: float | None = None, stray_guard: bool = False) -> DecideConfig:
    """`cfg` with the decision rules a seat's switches turn on (module doc, DECISION RULES): the
    wait floor, and the stray guard on STRAY_CARD_NAMES of `catalogue` (catalogue_json rows).
    Both off: `cfg` itself, unchanged."""
    if wait_floor is None and not stray_guard:
        return cfg
    changes: dict[str, Any] = {}
    if wait_floor is not None:
        changes["wait_floor"] = check_wait_floor(wait_floor)
    if stray_guard:
        cards = card_ids(catalogue, STRAY_CARD_NAMES)
        if not cards:
            raise ValueError(f"stray guard: the catalogue holds none of {sorted(STRAY_CARD_NAMES)}")
        changes["stray_cards"] = cards
    return cfg.replace(**changes)


def rules_record(wait_floor: float | None = None, stray_guard: bool = False) -> dict[str, Any]:
    """The switches as reports record them: only those on ({} when both are off)."""
    out: dict[str, Any] = {}
    if wait_floor is not None:
        out["wait_floor"] = check_wait_floor(wait_floor)
    if stray_guard:
        out["stray_guard"] = True
    return out


def rules_label(wait_floor: float | None = None, stray_guard: bool = False) -> str:
    """The switches as a run's name says them: " wait-floor 0.95 stray-guard" ("" when off)."""
    out = ""
    if wait_floor is not None:
        out += f" wait-floor {check_wait_floor(wait_floor):g}"
    if stray_guard:
        out += " stray-guard"
    return out


# ---------------------------------------------------------------------------
# plays in flight
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Pending:
    """One of our plays sent and not run yet: its card, its elixir, the hand slot it was
    tapped from (-1: any slot holding the card) and the tick of the tap (None: unknown). A
    button press (E15) is Pending(-1, cost, tick=..., button=k)."""

    card: int
    cost: int = 0
    slot: int = -1
    tick: int | None = None
    button: int = -1

    @property
    def is_press(self) -> bool:
        return self.button >= 0


def engine_pending(state: Mapping[str, Any], team: int) -> list[Pending]:
    """The state's own pending rows (RoyaleSim under a command delay): [kind, what, x, y,
    ticks_left, cost]; a play (kind 0) holds card `what`, a button press (kind 1) no card."""
    out = []
    for r in state["players"][team].get("pending") or ():
        if len(r) < 2:
            continue
        cost = int(r[5]) if len(r) > 5 else 0
        out.append(Pending(int(r[1]) if int(r[0]) == 0 else -1, cost))
    return out


def engine_pressing(state: Mapping[str, Any], team: int) -> frozenset[int]:
    """The buttons with a press of `team` accepted and not run (the state's pending rows of kind
    1, what = HAND_SIZE + button): the engine refuses another press of them CARD_PENDING."""
    out = set()
    for r in state["players"][team].get("pending") or ():
        if len(r) > 1 and int(r[0]) == PENDING_PRESS and int(r[1]) >= HAND_SIZE:
            out.add(int(r[1]) - HAND_SIZE)
    return frozenset(out)


def prune_pending(pending: Iterable[Pending], hand: Sequence[int], now: int,
                  max_age: int = PENDING_TICKS, *,
                  abilities: Sequence[Sequence[int]] | None = None) -> list[Pending]:
    """The live runner's taps still in flight (live_play): a tap stays while its card is still in
    the slot it was tapped from (any slot, for slot -1) and it is younger than `max_age` ticks.
    The client's hand and elixir change when the play runs, so a tap whose card has left was
    paid for and is no longer pending. A press (Pending.button >= 0) stays while it is younger
    than `max_age` and, given the side's `abilities` rows, its button still reads available
    (a press that ran starts the ability or spends the hero's charge)."""
    hand = list(hand)
    keep = []
    for p in pending:
        if p.tick is not None and now - p.tick >= max_age:
            continue
        if getattr(p, "button", -1) >= 0:
            if abilities is None or (p.button < len(abilities)
                                     and ability_row(abilities[p.button])[0]):
                keep.append(p)
            continue
        if p.slot >= 0:
            if p.slot < len(hand) and hand[p.slot] == p.card:
                keep.append(p)
        elif p.card in hand:
            keep.append(p)
    return keep


@dataclasses.dataclass(eq=False)
class CapPlay:
    """One of our taps as ElixirCap follows it: the Pending, the elixir its decision had before
    the plays then in flight (base_milli: the decision's spendable plus the costs it held for
    `linked`), the tick its card was first seen gone (None: not paid yet), the ledger's plays
    that were in flight at the tap (linked: their costs come off the cap once they are paid)
    and how many frames in a row the HUD has read at most tol_milli above the cap (caught) and
    its card has read back in its slot (backs: two in a row unpay it, one is a misread)."""

    pending: Pending
    base_milli: int
    paid_tick: int | None = None
    linked: tuple["CapPlay", ...] = ()
    caught: int = 0
    backs: int = 0


class ElixirCap:
    """The live runner's upper bound on our elixir in the ticks after its own plays (module
    doc, ELIXIR CAP). One per battle; never used by rs_eval, rs_match or RL.

        cap = RD.ElixirCap()                          # law: rs_elixir.default_law()
        cap.sent(RD.Pending(card, cost, slot, tick), d.spendable_milli,     # a tap sent, with
                 in_flight=pending_before)                                  # decide's pending
        c = cap.observe(now, hud_hand, hud_elixir_milli)   # each decision, after prune_pending
        RD.decide(..., pending=pending, elixir_cap=c)      # None: no cap on

    sent: the base is the decision's spendable plus the costs it held for the plays then in
    flight (`in_flight`, the pending list decide was given: unpaid taps, their cards still in
    their slots) that the ledger follows; those plays are linked to the new one, and each one's
    cost comes off its cap only once that play is paid (a refused one never is: decide holds
    its cost while it is pending, so it is not taken twice). A play in flight the ledger does
    not follow stays taken off the base, as before.
    observe: a tap whose card has left its slot (prune_pending's rule; its age aside) is paid
    on that frame and its cap goes on; its card read back in its slot within `back_ticks` of
    that (a one-frame misread of the slot) makes it unpaid again (its cap off; it is paid again
    when the card leaves again); one never paid is forgotten at `max_age` ticks. The cap of a
    paid tap is base_milli - 1000 x (cost + the linked plays paid by now) + regen_milli(tap
    tick, now), at most FULL_ELIXIR; it is on from the frame its tap is paid for
    `timeout_ticks` ticks (then it times out), and it ends once the HUD has read at most
    `tol_milli` above it on `release_frames` frames in a row (caught up; the last of them is
    its last frame; an unread elixir neither counts nor breaks the run). The frame's cap is the
    lowest of the caps on (None: none). Counts for the battle_end record: summary()."""

    def __init__(self, law: Any = None, *, tol_milli: int = ELIXIR_CAP_TOL,
                 timeout_ticks: int = ELIXIR_CAP_TICKS, slack_ticks: int = ELIXIR_CAP_SLACK,
                 max_age: int = PENDING_TICKS, release_frames: int = ELIXIR_CAP_FRAMES,
                 back_ticks: int = ELIXIR_CAP_BACK):
        if law is None:
            import rs_elixir                       # pure Python; only the live runner gets here
            law = rs_elixir.default_law()
        self.law = law
        self.tol_milli = int(tol_milli)
        self.timeout_ticks = int(timeout_ticks)
        self.slack_ticks = int(slack_ticks)
        self.max_age = int(max_age)
        self.release_frames = max(1, int(release_frames))
        self.back_ticks = int(back_ticks)
        self.plays: list[CapPlay] = []
        self.value: int | None = None              # the last frame's cap (None: none on)
        self.counts = {"plays": 0, "paid": 0, "caught_up": 0, "timed_out": 0, "lost": 0,
                       "frames_on": 0, "unpaid": 0}

    def regen_milli(self, tick_from: int, tick_to: int) -> int:
        """The most elixir (milli, rounded up) the bar can gain from tick_from to tick_to: the
        law's regen over those ticks dated slack_ticks later (the rate only ever rises)."""
        if tick_to <= tick_from:
            return 0
        s = self.slack_ticks
        fine = int(self.law.regen(int(tick_from) + s, int(tick_to) + s))
        return -(-fine * 1000 // int(self.law.scale))

    def cap_of(self, play: CapPlay, now: int) -> int:
        """The bound one paid play puts on our elixir at `now`."""
        p = play.pending
        tick = int(p.tick) if p.tick is not None else int(play.paid_tick or now)
        held = sum(max(0, int(c.pending.cost)) for c in play.linked if c.paid_tick is not None)
        return min(FULL_ELIXIR, int(play.base_milli) - 1000 * (max(0, int(p.cost)) + held)
                   + self.regen_milli(tick, now))

    def sent(self, pending: Pending, spendable_milli: int, *,
             in_flight: Iterable[Pending] = ()) -> None:
        """A tap sent (a play; a press is not followed): `spendable_milli` is the decision's,
        the elixir it had for this play, and `in_flight` the pending list it was decided with
        (its own costs taken off spendable_milli); see the class doc."""
        if getattr(pending, "button", -1) >= 0 or pending.card < 0:
            return
        links: list[CapPlay] = []
        held = 0
        for q in in_flight:
            if getattr(q, "button", -1) >= 0 or q.card < 0:
                continue
            c = next((c for c in self.plays if c.pending == q), None)
            if c is not None and c not in links:
                links.append(c)
                held += 1000 * max(0, int(q.cost))
        self.plays.append(CapPlay(pending, int(spendable_milli) + held, linked=tuple(links)))
        self.counts["plays"] += 1

    def observe(self, now: int, hand: Sequence[int], elixir_milli: int | None) -> int | None:
        """This frame's cap (None: no cap on); see the class doc."""
        now = int(now)
        hand = list(hand)
        keep: list[CapPlay] = []
        for c in self.plays:
            p = c.pending
            if c.paid_tick is None:
                if p.tick is not None and now - int(p.tick) >= self.max_age:
                    self.counts["lost"] += 1          # never paid: refused or lost
                    continue
                if not prune_pending([p], hand, now, max_age=10 ** 9):
                    c.paid_tick = now                 # its card left the slot: paid
                    c.caught = 0
                    self.counts["paid"] += 1
            elif now - int(c.paid_tick) >= self.timeout_ticks:
                self.counts["timed_out"] += 1
                continue
            elif (now - int(c.paid_tick) <= self.back_ticks
                  and prune_pending([p], hand, now, max_age=10 ** 9)):
                c.backs += 1                          # its card reads back in its slot ...
                if c.backs >= BACK_FRAMES:            # ... on two frames in a row: it was a misread
                    c.paid_tick = None
                    c.caught = c.backs = 0
                    self.counts["unpaid"] += 1
            else:
                c.backs = 0                           # one read-back frame keeps the cap on
            keep.append(c)
        on = [(c, self.cap_of(c, now)) for c in keep if c.paid_tick is not None]
        self.value = min(v for _, v in on) if on else None
        if on:
            self.counts["frames_on"] += 1
        if elixir_milli is not None:                  # caught up on release_frames in a row
            done = set()
            for c, v in on:
                c.caught = c.caught + 1 if int(elixir_milli) <= v + self.tol_milli else 0
                if c.caught >= self.release_frames:
                    done.add(id(c))
            self.counts["caught_up"] += len(done)
            keep = [c for c in keep if id(c) not in done]
        self.plays = keep
        return self.value

    def summary(self) -> dict[str, int]:
        return dict(self.counts)

    def config(self) -> dict[str, int]:
        return {"tol_milli": self.tol_milli, "release_frames": self.release_frames,
                "timeout_ticks": self.timeout_ticks, "slack_ticks": self.slack_ticks,
                "max_age": self.max_age, "back_ticks": self.back_ticks}


def _spendable(p: Mapping[str, Any], state: Mapping[str, Any], team: int,
               pending: Iterable[Pending] | None, cfg: DecideConfig,
               elixir_cap: int | None = None) -> tuple[int, list[Pending]]:
    """(elixir_milli, at most elixir_cap, minus our plays in flight; the plays whose slots
    are blocked)."""
    if pending is None:
        cost = int(p.get("pending_cost", 0) or 0)
        held = engine_pending(state, team) if cfg.mask_pending else []
    else:
        held = list(pending)
        cost = sum(max(0, int(x.cost)) for x in held)
    elixir = p["elixir_milli"]
    if elixir_cap is not None and int(elixir_cap) < elixir:
        elixir = int(elixir_cap)
    return elixir - cost * 1000, held


def _pending_slots(held: Iterable[Pending], hand: Sequence[int]) -> set[int]:
    out: set[int] = set()
    for x in held:
        if x.card < 0:
            continue
        if 0 <= x.slot < len(hand):
            if hand[x.slot] == x.card:
                out.add(x.slot)
        elif x.slot < 0:
            out.update(k for k, c in enumerate(hand[:HAND_SIZE]) if c == x.card)
    return out


def _pending_buttons(state: Mapping[str, Any], team: int, pending: Iterable[Pending] | None,
                     held: Iterable[Pending]) -> frozenset[int]:
    """The buttons with a press in flight: the state's (pending None) or the explicit list's."""
    if pending is None:
        return engine_pressing(state, team)
    return frozenset(int(x.button) for x in held if getattr(x, "button", -1) >= 0)


def ability_row(row: Sequence[Any]) -> tuple[int, int, int, int, int]:
    """(available, spent, cost, card_id, cooldown_ticks) of an abilities row, by position
    (royalegym protocol.ability_row): the first five columns, a missing one ABILITY_DEFAULTS."""
    vals = [int(v) for v in list(row)[:len(ABILITY_COLUMNS)]]
    return tuple(vals + list(ABILITY_DEFAULTS[len(vals):]))  # type: ignore[return-value]


def abilities_of(state: Mapping[str, Any], team: int) -> list:
    """players[team]["abilities"] (one row per button, button order; [] without any)."""
    return list(state["players"][team].get("abilities") or ())


def press_cost(state: Mapping[str, Any], team: int, button: int) -> int:
    """The elixir `team`'s `button` costs (its abilities row's cost; 0 with no row), as
    tools/rs_rows.press_cost reads it."""
    rows = abilities_of(state, team)
    return max(0, ability_row(rows[button])[2]) if 0 <= button < len(rows) else 0


def button_card(state: Mapping[str, Any], team: int, button: int) -> int:
    """The catalogue id behind `team`'s `button` (its abilities row's card_id), -1 unknown."""
    rows = abilities_of(state, team)
    return ability_row(rows[button])[3] if 0 <= button < len(rows) else -1


def button_mask(state: Mapping[str, Any], team: int, n_buttons: int,
                spendable_milli: int, *, pressing: Iterable[int] = ()) -> list[bool]:
    """Which of the `n_buttons` ability-button gate entries may be pressed now (E15):
    players[team]["abilities"] has a row for the button (button order: heroes, then the
    champion), it reads available (a living unit behind it, off cooldown, not mid-ability:
    the engine would take the press but for the elixir) and not spent (a hero's one charge),
    its cost x 1000 fits `spendable_milli`, and it is not in `pressing` (a press of it in
    flight). decide asks only with cfg.presses; without it every button is masked."""
    rows = abilities_of(state, team)
    pressing = frozenset(int(k) for k in pressing)
    out = []
    for k in range(int(n_buttons)):
        if k >= len(rows) or k in pressing:
            out.append(False)
            continue
        available, spent, cost, _, _ = ability_row(rows[k])
        out.append(bool(available) and not spent
                   and int(spendable_milli) >= max(0, cost) * 1000)
    return out


# ---------------------------------------------------------------------------
# v2 row inputs: our plays in flight
# ---------------------------------------------------------------------------

def pending_inputs(pending: Iterable[Pending], now: int, delay: int) -> list[tuple[int, int]]:
    """obs_row's `pending` from the live runner's taps: (card token, ticks to land) of each play
    tapped before `now` (not a press; a tap at `now` is the decision's own), soonest first (ties
    in list order). Ticks to land: max(0, tap + delay - now), the decision-time rows' exec - t
    with `delay` = obs_delay_ticks(meta); `delay` for a tap of unknown tick (just tapped).
    delay <= 0: [] (landing-time rows never show a pending play)."""
    delay = int(delay)
    if delay <= 0:
        return []
    out = []
    for p in pending:
        if p.card is None or int(p.card) < 0 or getattr(p, "button", -1) >= 0:
            continue
        if p.tick is not None and int(p.tick) >= int(now):
            continue
        land = delay if p.tick is None else max(0, int(p.tick) + delay - int(now))
        out.append((int(p.card) + TOKEN_OFFSET, land))
    out.sort(key=lambda x: x[1])
    return out


def engine_pending_inputs(state: Mapping[str, Any], team: int, lead: int = 0
                          ) -> list[tuple[int, int]]:
    """obs_row's `pending` from the state's pending rows (RoyaleSim under a command delay;
    rs_eval.RSAgent.own_pending's rule): (card + 2, ticks_left - lead) of each play row (kind
    0, a card) with ticks_left >= lead, soonest first. Presses are not pending tokens."""
    lead = int(lead or 0)
    out = []
    for r in state["players"][team].get("pending") or ():
        if len(r) > 4 and int(r[0]) == PENDING_PLAY and int(r[1]) >= 0 and int(r[4]) >= lead:
            out.append((int(r[1]) + TOKEN_OFFSET, int(r[4]) - lead))
    out.sort(key=lambda x: x[1])
    return out


def v2_row_kwargs(opp_elixir: Any, pending: Iterable[tuple[int, int]] = (),
                  buttons: Sequence[int] | None = None) -> dict:
    """tools/rs_rows.obs_row's keywords for a v2 row, as rs_eval.RSAgent.v2_kw gives them:
    schema rs_rows/v2, `opp_elixir` ((milli, exact): rs_rows.SeenElixir.view, lag 0 live),
    `pending` (pending_inputs / engine_pending_inputs) and, when known, `buttons` (the side's
    button cards, rs_presses.side_buttons)."""
    kw = {"schema": rs_schema.SCHEMA_V2, "opp_elixir": opp_elixir, "pending": list(pending)}
    if buttons is not None:
        kw["buttons"] = [int(c) for c in buttons]
    return kw


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------

BATCH_KEYS = ("unit_ids", "unit_feats", "towers", "scalars", "hand", "opp_seen", "opp_last",
              "opp_last_age")


def _schema(value: Any) -> Any:
    """rs_schema.get(value), also for a Schema of another import of rs_schema (by its name, as
    sim/model_tokens._as_schema reads it); None stays None (legacy)."""
    if value is None:
        return None
    if not isinstance(value, str) and hasattr(value, "name"):
        value = value.name
    return rs_schema.get(value)


def model_schema_of(model: Any) -> Any:
    """The rs_schema Schema of a v2 TokenPolicy (model.schema.v2), else None (legacy)."""
    spec = getattr(model, "schema", None)
    return spec if getattr(spec, "v2", False) else None


def row_batch(row: Mapping[str, Any], torch: Any, schema: Any = None) -> dict:
    """One obs_row (tools/rs_rows.py) as a batch of one for TokenPolicy. schema None or legacy:
    as rs_eval.RSAgent.batch builds it, exactly. v2 (E14): every column of
    rs_schema.V2.model_inputs in its order, int columns long and float columns float, the
    scalar `next` as [1]: what sim/model_tokens.row_batch(row, "v2") builds (a test checks),
    without importing the model."""
    spec = _schema(schema)
    t = torch
    if spec is None or not spec.v2:
        b = {k: t.as_tensor(np.asarray(row[k])[None]) for k in BATCH_KEYS}
        for k in ("unit_ids", "hand", "opp_seen", "opp_last"):
            b[k] = b[k].long()
        b["unit_feats"] = b["unit_feats"].float()
        b["opp_last_age"] = b["opp_last_age"].float()
        b["next"] = t.as_tensor([int(row["next"])]).long()
        return b
    out = {}
    for name in spec.model_inputs:
        col = spec.column(name)
        if not col.shape:
            arr = np.asarray([int(row[name])], dtype=np.int64)
        else:
            arr = np.asarray(row[name])[None]
        x = t.as_tensor(arr)
        out[name] = x.long() if col.dtype.startswith(("int", "bool")) else x.float()
    return out


class TokenPolicyAdapter:
    """decide()'s view of a TokenPolicy (sim/model_tokens.py): batch(row), gate(batch) -> the
    gate logits of the row (numpy float32, [wait, slot 0..3, buttons...]), tiles(batch, slot)
    -> the 576 tile logits for the card in that slot, heads(batch) -> (gate, the critic's value
    or None). The calls are RSAgent.act's: the gate with card_ids = hand slot 0 (clamped), the
    tiles with card_ids = the slot's card. A model with a critic (has_critic, outputs()) gives
    the gate and value2 from one outputs() call; any other the gate from forward(), as always.
    The batch is the model's (row_batch with model_schema_of(model)) unless `batch` is given."""

    def __init__(self, model: Any, torch: Any = None, *,
                 batch: Callable[[Mapping[str, Any]], dict] | None = None):
        if torch is None:
            torch = importlib.import_module("torch")
        self.model = model
        self.torch = torch
        self._batch = batch

    @property
    def schema(self) -> Any:
        """The model's v2 Schema, or None for a legacy model."""
        return model_schema_of(self.model)

    @property
    def has_critic(self) -> bool:
        return bool(getattr(self.model, "has_critic", False)) and hasattr(self.model, "outputs")

    def batch(self, row: Mapping[str, Any]) -> dict:
        if self._batch is not None:
            return self._batch(row)
        return row_batch(row, self.torch, self.schema)

    def heads(self, b: Mapping[str, Any]) -> tuple[np.ndarray, tuple[float, ...] | None]:
        """(the gate logits, the critic's raw [outcome, won logit] or None), one forward pass."""
        card_ids = b["hand"][:, 0].clamp(min=0)
        value = None
        with self.torch.no_grad():
            if self.has_critic:
                out = self.model.outputs(b, card_ids=card_ids)
                gate = out["gate"]
                if out.get("value2") is not None:
                    value = tuple(float(v) for v in
                                  np.asarray(out["value2"][0].float().numpy()).reshape(-1))
            else:
                gate, _, _ = self.model(b, card_ids=card_ids)
        return gate[0].float().numpy(), value

    def gate(self, b: Mapping[str, Any]) -> np.ndarray:
        return self.heads(b)[0]

    def tiles(self, b: Mapping[str, Any], slot: int) -> np.ndarray:
        with self.torch.no_grad():
            _, tile, _ = self.model(b, card_ids=b["hand"][:, slot])
        return tile[0].float().numpy()


def as_policy(model: Any) -> Any:
    """`model` when it already speaks batch / gate / tiles, else a TokenPolicyAdapter of it."""
    if all(hasattr(model, k) for k in ("batch", "gate", "tiles")):
        return model
    return TokenPolicyAdapter(model)


# ---------------------------------------------------------------------------
# legality
# ---------------------------------------------------------------------------

class Bound:
    """A legality check bound to one state: check(slot, card, x_sub, y_sub) -> bool."""

    __slots__ = ("name", "fn")

    def __init__(self, name: str, fn: Callable[[int, int, int, int], Any]):
        self.name = name
        self.fn = fn

    def __call__(self, slot: int, card: int, x_sub: int, y_sub: int) -> bool:
        return bool(self.fn(slot, card, x_sub, y_sub))


class BattleLegality:
    """check_deploy on a battle that holds the state's hand in the state's slot order: the match
    battle itself (rs_eval). Exactly the call RSAgent.act made."""

    name = "battle"

    def __init__(self, battle: Any):
        self.battle = battle

    def bind(self, state: Mapping[str, Any], team: int) -> Bound:
        b, team = self.battle, int(team)
        return Bound(self.name, lambda slot, card, x, y: b.check_deploy(team, slot, x, y) == 0)


#: The arena as whole tiles, WORLD frame (row 0 = y 0..1000 mt, blue's back row; column 0 = x
#: 0..1000): '.' every half-cell of the tile is plain ground, 'N' one is NO_DEPLOY (back-row
#: strips, river corners, the king blocks), 'W' one is WATER (the river; the dry tiles in rows
#: 15-16 are the bridges, columns 3 and 14). Reduced from royalesim 0.1.42
#: data/derived/arena.json "grid" (the ~2018 default arena, retroroyale tilemap.csv; sha256
#: 7fbe4b0d...); symmetric under both mirrors. The engine-marked test compares it to the file.
ARENA_TILES: tuple[str, ...] = (
    'NNNNNN......NNNNNN', '.......NNNN.......', '.......NNNN.......', '.......NNNN.......',
    '.......NNNN.......', '..................', '..................', '..................',
    '..................', '..................', '..................', '..................',
    '..................', '..................', 'N................N', 'WWW.WWWWWWWWWW.WWW',
    'WWW.WWWWWWWWWW.WWW', 'N................N', '..................', '..................',
    '..................', '..................', '..................', '..................',
    '..................', '..................', '..................', '.......NNNN.......',
    '.......NNNN.......', '.......NNNN.......', '.......NNNN.......', 'NNNNNN......NNNNNN',
)
RIVER_ROWS = (15, 16)
#: Crown tower centres in world TILES, (team, slot), slot as players[team]["tower_hp"] holds
#: it: 0 king, 1 the WORLD-left (x 3.5) princess, 2 the world-right (x 14.5) one, for BOTH
#: teams. That is RoyaleSim's engine tower order k (king, left lane, right lane), which state_json
#: and reset(tower_hp=) use as they stand because rs_engine builds every Battle with slot_of_k
#: the identity (rs_engine.SLOT_OF_K), and it is what rs_rows.towers_vec and tools/public_state.py
#: (screen = world sides, "own_left" = screen left) read and write. royalegym's own-frame
#: TowerSlot, where red's slot 1 is world right (and conftest's TOWER_MILLI, which follows it),
#: is a different numbering. The engine-marked test kills team 1's slot 1 to check this.
TOWER_CENTRE: dict[tuple[int, int], tuple[float, float]] = {
    (0, 0): (9.0, 3.0), (0, 1): (3.5, 6.5), (0, 2): (14.5, 6.5),
    (1, 0): (9.0, 29.0), (1, 1): (3.5, 25.5), (1, 2): (14.5, 25.5),
}
#: Half the side of a tower's body in tiles: the king is 4 x 4, a princess 3 x 3.
TOWER_BODY_HALF = {0: 2.0, 1: 1.5, 2: 1.5}
#: Half the NoDeploySize rect in tiles (cards.json no_deploy_size_tiles: king 18 x 16, princess
#: 11 x 21), centred on the tower: a troop may not be placed inside (closed) the rect of an
#: ALIVE enemy tower (royalegym DeployRules). All up, that is the whole enemy side; a fallen
#: princess opens the 4 rows between the river and the enemy king's rect in its lane.
NO_DEPLOY_HALF = {0: (9.0, 8.0), 1: (5.5, 10.5), 2: (5.5, 10.5)}

TROOP, BUILDING, SPELL, ROLLING, SPELL_NOT_ON_WATER, TUNNEL, MIRROR = (
    "troop", "building", "spell", "rolling", "spell_not_on_water", "tunnel", "mirror")
#: Spells placed in troop territory (royalegym Placement.ROLLING and its kin): catalogue names.
ROLLING_SPELLS = frozenset({"Log", "BarbLog", "Heal", "RoyalDelivery"})
#: Cards whose unit travels under ground to the tap (Placement.TUNNEL).
TUNNEL_CARDS = frozenset({"Miner", "GoblinDrill"})
#: Spells whose projectile spawns characters: never on water (spells.SPAWNING_SPELL_WATER_RULE).
SPAWNING_SPELLS = frozenset({"GoblinBarrel"})
MIRROR_CARDS = frozenset({"Mirror"})


def _tower_hp(state: Mapping[str, Any], team: int) -> list[Any]:
    hp = list(state["players"][team].get("tower_hp") or [])
    return (hp + [None] * 3)[:3]


def _alive(hp: Any) -> bool:
    """A tower is alive unless its hp reads 0 or less; unread (None) counts as alive."""
    return hp is None or hp > 0


def destroyed_towers(state: Mapping[str, Any]) -> tuple[tuple[bool, bool, bool], ...]:
    """((king, left, right) destroyed for team 0), (... team 1)."""
    return tuple(tuple(not _alive(hp) for hp in _tower_hp(state, t)) for t in (0, 1))


class PureLegality:
    """Deploy legality without an engine, conservative (module doc). Tile-centre taps only
    (decide taps tile centres): a tap is judged by the whole tile it is in."""

    name = "pure"

    def __init__(self, catalogue: Sequence[Sequence[Any]], sub: int = SUBTILE_PER_MILLITILE):
        self.names = {i: str(row[0]) for i, row in enumerate(catalogue)}
        self.kinds = {i: str(row[8]).upper() if len(row) > 8 else "" for i, row in
                      enumerate(catalogue)}
        self.sub = sub

    def placement(self, card: int) -> str | None:
        """The card's deploy rule: TROOP, BUILDING, SPELL, ROLLING, SPELL_NOT_ON_WATER, TUNNEL,
        MIRROR, or None for a card the catalogue does not hold."""
        name, kind = self.names.get(int(card)), self.kinds.get(int(card), "")
        if name is None:
            return None
        if name in MIRROR_CARDS:
            return MIRROR
        if name in TUNNEL_CARDS:
            return TUNNEL
        if kind == "SPELL":
            return (ROLLING if name in ROLLING_SPELLS else
                    SPELL_NOT_ON_WATER if name in SPAWNING_SPELLS else SPELL)
        if kind == "BUILDING":
            return BUILDING
        if kind == "TROOP":
            return TROOP
        return None

    @staticmethod
    def tile_legal(rule: str | None, team: int, tx: int, ty: int,
                   alive: Mapping[tuple[int, int], bool]) -> bool:
        """May a card with deploy rule `rule` go on world tile (tx, ty) for `team`?"""
        if rule is None or rule == MIRROR or not (0 <= tx < GRID_W and 0 <= ty < GRID_H):
            return False
        cell = ARENA_TILES[ty][tx]
        if rule == SPELL:
            return True
        if rule == SPELL_NOT_ON_WATER:
            return cell != "W"
        cx, cy = tx + 0.5, ty + 0.5
        for (t, s), (x0, y0) in TOWER_CENTRE.items():
            half = TOWER_BODY_HALF[s]
            if alive[(t, s)] and abs(cx - x0) < half and abs(cy - y0) < half:
                return False
        if rule == TUNNEL:
            return cell != "W"
        if cell != "." or ty in RIVER_ROWS:
            return False
        if rule == BUILDING:
            return ty >= 17 if team == 1 else ty <= 14
        enemy = 1 - team
        for s in (0, 1, 2):
            if not alive[(enemy, s)]:
                continue
            (x0, y0), (hw, hh) = TOWER_CENTRE[(enemy, s)], NO_DEPLOY_HALF[s]
            if abs(cx - x0) <= hw and abs(cy - y0) <= hh:
                return False
        return True

    def bind(self, state: Mapping[str, Any], team: int) -> Bound:
        team = int(team)
        dead = destroyed_towers(state)
        alive = {(t, s): not dead[t][s] for t in (0, 1) for s in (0, 1, 2)}
        mt = state["players"][team].get("mirror_target", -1)
        target = -1 if mt is None else int(mt)
        sub = self.sub

        def check(slot: int, card: int, x_sub: int, y_sub: int) -> bool:
            rule = self.placement(card)
            if rule == MIRROR:
                rule = self.placement(target) if target >= 0 else None
            tx, ty = math.floor(x_sub / sub / 1000), math.floor(y_sub / sub / 1000)
            return self.tile_legal(rule, team, tx, ty, alive)

        return Bound(self.name, check)


def legality_deck(player: Mapping[str, Any], n_cards: int,
                  skip: Iterable[int] = ()) -> list[int]:
    """8 distinct catalogue ids for the legality battle: the hand in its slots (an empty or
    repeated slot gets a stand-in), then the next card and the rest of the deck, then other
    catalogue ids (not in `skip`) in id order."""
    def ok(c: Any) -> bool:
        return isinstance(c, (int, np.integer)) and not isinstance(c, bool) and 0 <= c < n_cards

    hand = (list(player.get("hand") or []) + [-1] * HAND_SIZE)[:HAND_SIZE]
    used: set[int] = set()
    slots: list[int | None] = []
    for c in hand:
        if ok(c) and int(c) not in used:
            slots.append(int(c))
            used.add(int(c))
        else:
            slots.append(None)
    skip = set(skip)
    known = [player.get("next_card", -1), *(player.get("deck") or [])]

    def pool():
        for c in known:
            if ok(c):
                yield int(c)
        for c in range(n_cards):
            if c not in skip:
                yield c

    source = (c for c in pool() if c not in used)

    def take() -> int:
        for c in source:
            if c not in used:
                used.add(c)
                return c
        raise ValueError(f"legality_deck: the catalogue has fewer than 8 usable cards ({n_cards})")

    deck = [c if c is not None else take() for c in slots]
    while len(deck) < 8:
        deck.append(take())
    return deck


_FAILED = object()


class TowersOnlyLegality:
    """check_deploy on a RoyaleSim battle holding only the crown towers (module doc).

    engine: an rs_engine.EngineConfig (or its to_json(); None = rs_engine.active()); the battle
    runs it undelayed. rs: the royalesim module (None: imported on first use). catalogue (or
    fallback=): the PureLegality used when royalesim is missing, the reset is refused, or a card
    is not in the battle's hand; without one those checks are False and last_error says why.
    """

    name = "towers"

    def __init__(self, engine: Any = None, rs: ModuleType | None = None, *,
                 catalogue: Sequence[Sequence[Any]] | None = None, fallback: Any = None,
                 cache_size: int = 8, start_tick: int = LEGAL_START_TICK):
        self.engine = engine
        self._rs = rs
        self.fallback = (fallback if fallback is not None else
                         PureLegality(catalogue) if catalogue is not None else None)
        self.cache_size = max(1, int(cache_size))
        self.start_tick = int(start_tick)
        self._cache: OrderedDict = OrderedDict()
        self._spare: list[Any] = []
        self._cat: list | None = None
        self._unavailable: str | None = None
        self.builds = 0
        self.hits = 0
        self.last_error: str | None = None

    @staticmethod
    def key(state: Mapping[str, Any], team: int) -> tuple:
        """What legality depends on here: the team, its hand in slot order, the destroyed towers."""
        hand = tuple(-1 if c is None else int(c)
                     for c in list(state["players"][team].get("hand") or [])[:HAND_SIZE])
        return int(team), hand, destroyed_towers(state)

    def _module(self) -> ModuleType | None:
        if self._unavailable is not None:
            return None
        try:
            return rs_engine.royalesim_module(self._rs)
        except Exception as exc:                # ImportError, a wheel for another Python
            self._unavailable = self.last_error = (
                f"royalesim does not import ({type(exc).__name__}: {exc})")
            return None

    def _build(self, state: Mapping[str, Any], team: int, key: tuple) -> Any:
        rs = self._module()
        if rs is None:
            return _FAILED
        try:
            battle = (self._spare.pop() if self._spare else
                      rs_engine.make_battle(rs_engine.resolve(self.engine).undelayed(), rs))
            if self._cat is None:
                self._cat = json.loads(battle.catalogue_json())
        except Exception as exc:                # the config refused: no battle, ever
            self._unavailable = self.last_error = (
                f"no legality battle ({type(exc).__name__}: {exc})")
            return _FAILED
        mirror = [i for i, row in enumerate(self._cat) if row and row[0] in MIRROR_CARDS]
        try:
            deck = legality_deck(state["players"][team], len(self._cat), skip=mirror)
            towers = [[0 if dead else 1 for dead in key[2][t]] for t in (0, 1)]
            battle.reset(seed=0, decks=[list(deck), list(deck)], shuffle=0,
                         start_tick=self.start_tick, elixir_milli=[FULL_ELIXIR, FULL_ELIXIR],
                         tower_hp=towers, spawns=[])
            hand = json.loads(battle.state_json())["players"][team]["hand"]
        except Exception as exc:                # this hand / these towers only
            self.last_error = f"legality battle reset refused ({type(exc).__name__}: {exc})"
            return _FAILED
        slot_of: dict[int, int] = {}
        for k, c in enumerate(list(hand)[:HAND_SIZE]):
            if c >= 0 and int(c) not in slot_of:
                slot_of[int(c)] = k
        self.builds += 1
        return battle, slot_of

    def _store(self, key: tuple, entry: Any) -> None:
        self._cache[key] = entry
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            if old is not _FAILED:
                self._spare.append(old[0])      # reset again for the next key, not rebuilt

    def bind(self, state: Mapping[str, Any], team: int) -> Bound:
        team = int(team)
        key = self.key(state, team)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._build(state, team, key)
            self._store(key, entry)
        else:
            self._cache.move_to_end(key)
            self.hits += 1
        fb = self.fallback.bind(state, team) if self.fallback is not None else None
        if entry is _FAILED:
            if fb is not None:
                return fb
            return Bound("none", lambda slot, card, x, y: False)
        battle, slot_of = entry

        def check(slot: int, card: int, x_sub: int, y_sub: int) -> bool:
            k = slot_of.get(int(card))
            if k is None:
                return fb(slot, card, x_sub, y_sub) if fb is not None else False
            return battle.check_deploy(team, k, x_sub, y_sub) == 0

        return Bound(self.name, check)


def _bind(legality: Any, state: Mapping[str, Any], team: int) -> Bound:
    if hasattr(legality, "bind"):
        return legality.bind(state, team)
    if hasattr(legality, "check_deploy"):
        return BattleLegality(legality).bind(state, team)
    if callable(legality):
        return Bound("callable", legality)
    raise TypeError(f"legality: a Legality (bind), a Battle (check_deploy) or a callable, "
                    f"got {type(legality).__name__}")


# ---------------------------------------------------------------------------
# the lane guard
# ---------------------------------------------------------------------------

def threat_lane(state: Mapping[str, Any], team: int, sub: int = SUBTILE_PER_MILLITILE
                ) -> int | None:
    """0 / 1 (own-frame x half) holding the enemy push, or None (live_play.threat_lane): enemy
    troops (not buildings) past GUARD_DEEP_MT, GUARD_MIN_HP of them, GUARD_SHARE in one lane."""
    hp = [0.0, 0.0]
    for e in state["entities"]:
        if e[F["team"]] == team or e[F["kind"]] == 1:
            continue
        ox, oy = own(e[F["x"]], e[F["y"]], team, sub)
        if oy < GUARD_DEEP_MT:
            continue
        hp[0 if ox < 9_000 else 1] += max(0, e[F["hp"]])
    tot = hp[0] + hp[1]
    if tot < GUARD_MIN_HP:
        return None
    for lane in (0, 1):
        if hp[lane] >= GUARD_SHARE * tot:
            return lane
    return None


def _guarded(lane: int | None, tx: int) -> bool:
    """True when the lane guard skips own-frame column tx."""
    return (lane is not None and not GUARD_CENTRE[0] <= tx <= GUARD_CENTRE[1]
            and (0 if tx < 9 else 1) != lane)


# ---------------------------------------------------------------------------
# the stray guard (rule 7; module doc, DECISION RULES)
# ---------------------------------------------------------------------------

def _own_mt(x_mt: float, y_mt: float, team: int) -> tuple[float, float]:
    """World millitiles -> the acting player's frame (to_world's inverse; it is its own)."""
    return to_world(x_mt, y_mt, team)


def _segment_mt(px: float, py: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance from (px, py) to the segment a-b, millitiles."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / n))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


@dataclasses.dataclass(frozen=True)
class StrayGuard:
    """The stray guard of one decision (module doc, DECISION RULES), own frame, millitiles:
    `paths`, one polyline per threat (and per threat and building of ours), `towers`, our
    living crown towers as (x, y, body half), and `push_lanes`, the lanes (0 left, 1 right)
    with a troop of ours at our bridge or beyond, whose bridge tiles are kept."""

    paths: tuple
    towers: tuple
    push_lanes: frozenset = frozenset()

    def reach(self, tx: int, ty: int) -> float:
        """Millitiles from own-frame tile (tx, ty)'s centre to the nearest threat's path."""
        cx, cy = tx * 1000 + 500, ty * 1000 + 500
        return min(_segment_mt(cx, cy, p[i], p[i + 1]) for p in self.paths
                   for i in range(len(p) - 1))

    def keeps(self, tx: int, ty: int) -> bool:
        """Whether own-frame tile (tx, ty) stays in for a stray card: on the enemy half, a
        bridge tile of a lane we push in, next to one of our towers, or within STRAY_REACH_MT
        of a threat's path."""
        if ty < STRAY_OWN_ROW:
            return True
        if ty in STRAY_BRIDGE_ROWS and any(lo <= tx <= hi and lane in self.push_lanes
                                           for lane, (lo, hi) in enumerate(STRAY_BRIDGE_COLS)):
            return True
        cx, cy = tx * 1000 + 500, ty * 1000 + 500
        for x0, y0, half in self.towers:
            if max(abs(cx - x0), abs(cy - y0)) <= half + STRAY_TOWER_MT:
                return True
        return self.reach(tx, ty) <= STRAY_REACH_MT


def stray_guard(state: Mapping[str, Any], team: int, sub: int = SUBTILE_PER_MILLITILE
                ) -> StrayGuard | None:
    """The stray guard of `team` on `state`, or None when no enemy threatens (module doc,
    DECISION RULES): every enemy troop (not a building, as threat_lane, nor a crown tower)
    with hp left at own-frame y >= GUARD_DEEP_MT."""
    hp = _tower_hp(state, team)
    towers, princess = [], {}
    for s in (0, 1, 2):
        if not _alive(hp[s]):
            continue
        wx, wy = TOWER_CENTRE[(team, s)]
        ox, oy = _own_mt(wx * 1000, wy * 1000, team)
        towers.append((ox, oy, TOWER_BODY_HALF[s] * 1000))
        if s:
            princess[0 if ox < LANE_SPLIT_MT else 1] = (ox, oy)
    king = _own_mt(TOWER_CENTRE[(team, 0)][0] * 1000, TOWER_CENTRE[(team, 0)][1] * 1000, team)
    ours, pushes, threats = [], set(), []
    for e in state["entities"]:
        if e[F["hp"]] <= 0 or e[F["tower_slot"]] >= 0:
            continue
        ox, oy = own(e[F["x"]], e[F["y"]], team, sub)
        building = e[F["kind"]] == KIND_BUILDING
        if e[F["team"]] == team:
            if building:
                if oy >= STRAY_OWN_ROW * 1000:
                    ours.append((ox, oy))
            elif oy < STRAY_PUSH_MT:
                pushes.add(0 if ox < LANE_SPLIT_MT else 1)
        elif not building and oy >= GUARD_DEEP_MT:
            threats.append((ox, oy, bool(e[F["flying"]])))
    if not threats:
        return None
    paths = []
    for ox, oy, flying in threats:
        lane = 0 if ox < LANE_SPLIT_MT else 1
        entry = (ox, oy)
        if not flying and oy < BRIDGE_EXIT_MT:
            entry = (LANE_X_MT[lane], BRIDGE_EXIT_MT)
        target = princess.get(lane, king)
        paths.append(((ox, oy), entry, target) if entry != (ox, oy) else ((ox, oy), target))
        paths += [(entry, b) for b in ours]
    return StrayGuard(tuple(paths), tuple(towers), frozenset(pushes))


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Decision:
    """What decide() chose. kind PLAY (command set) or WAIT (reason: R_WAIT drawn, R_NO_CARD
    nothing playable, R_NO_TILE no legal tile, R_FLOOR the wait floor) or PRESS (E15: `button`,
    its card in `card` when the abilities row names it, its elixir in `cost`; command (team,
    HAND_SIZE + button, 0, 0)). Tiles: `tile` = ty * 18 + tx, the model's own frame; x_sub /
    y_sub the engine-frame tap of its centre. `value`: the critic's raw [outcome, won logit]
    (module doc, V2 CHECKPOINTS). `stray`: the legal tiles the stray guard turned down (DECISION
    RULES)."""

    team: int
    tick: int | None
    kind: str = WAIT
    reason: str = ""
    slot: int = -1
    card: int = -1
    cost: int = 0
    tile: int = -1
    tx: int = -1
    ty: int = -1
    x_sub: int | None = None
    y_sub: int | None = None
    p_wait: float = 1.0
    q_wait: float = 1.0
    capped: bool = False
    fallback: bool = False
    tried: tuple[int, ...] = ()
    mask: tuple[bool, ...] = ()
    spendable_milli: int = 0
    lane: int | None = None
    legality: str = ""
    mode: str = ""
    button: int = -1                         # E15: the ability button pressed
    gate_q: tuple[float, ...] = ()           # sampling: the gate distribution
    gate_a: int = -1                         # sampling: the gate entry drawn
    logq_gate: float = 0.0
    tile_cands: tuple[int, ...] = ()         # sampling: the legal tiles the tile was drawn from
    tile_a: int = -1                         # sampling: index into tile_cands
    logq_tile: float = 0.0
    value: tuple[float, ...] = ()            # E15: the critic (value2), raw; () without one
    stray: int = 0                           # rule 7: legal tiles the stray guard turned down

    @property
    def is_play(self) -> bool:
        return self.kind == PLAY

    @property
    def is_press(self) -> bool:
        return self.kind == PRESS

    @property
    def command(self) -> tuple[int, int, int, int] | None:
        """(team, slot, x_sub, y_sub) for Battle.step; a press (team, HAND_SIZE + button, 0, 0);
        None for a wait."""
        if self.is_press:
            return (self.team, HAND_SIZE + self.button, 0, 0)
        if not self.is_play:
            return None
        return (self.team, self.slot, self.x_sub, self.y_sub)

    @property
    def win_p(self) -> float | None:
        """The critic's win probability (sigmoid of value[1]), None without a critic."""
        if len(self.value) < 2:
            return None
        z = float(self.value[1])
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    @property
    def model_tile(self) -> tuple[int, int] | None:
        return (self.tx, self.ty) if self.is_play else None

    @property
    def world_mt(self) -> tuple[int, int] | None:
        """World millitiles of the tap (the tile centre)."""
        return to_world(self.tx * 1000 + 500, self.ty * 1000 + 500, self.team) if self.is_play else None

    @property
    def world_tile(self) -> tuple[int, int] | None:
        """World tile of the tap. For the live bot (team 1) it is the SCREEN tile,
        (17 - tx, ty) = live_play.screen_tile = geometry.model_tile_to_screen_tile."""
        w = self.world_mt
        return None if w is None else (w[0] // 1000, w[1] // 1000)

    def suggest_kwargs(self, name: str = "") -> dict[str, Any]:
        """Keyword arguments for scripts/vision/control.ControlState.suggest(**...): the
        suggestion shown and logged in MANUAL and AUTO alike (the tile is world_tile). A press
        is suggested as card -1 with reason "press" and its button: the control state never
        taps it (the live runner has no press tap; our deck has no ability button)."""
        kw: dict[str, Any] = {"tick": self.tick, "p_wait": self.p_wait}
        if name:
            kw["name"] = name
        if self.is_play:
            kw.update(card=self.card, slot=self.slot, tile=self.world_tile,
                      model_tile=[self.tx, self.ty], capped=self.capped, fallback=self.fallback)
        elif self.is_press:
            kw.update(card=-1, reason=PRESS, button=self.button, button_card=self.card)
        else:
            kw.update(card=-1, reason=self.reason)
        return kw

    def to_record(self) -> dict[str, Any]:
        """A JSON-ready dict (lists for tuples) with the command."""
        rec = dataclasses.asdict(self)
        for k, v in rec.items():
            if isinstance(v, tuple):
                rec[k] = list(v)
        rec["command"] = None if self.command is None else list(self.command)
        return rec


def decide(policy: Any, row: Mapping[str, Any], state: Mapping[str, Any], team: int,
           legality: Any, cfg: DecideConfig | None = None, *, rng: Any,
           pending: Iterable[Pending] | None = None, tick: int | None = None,
           dt: int | None = None, sub: int = SUBTILE_PER_MILLITILE,
           aim: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
           blocked_tiles: np.ndarray | None = None, elixir_cap: int | None = None) -> Decision:
    """Wait, play or (cfg.presses, a v2 gate) press for `team` (module doc, THE DECISION).

    policy    a TokenPolicyAdapter (or a TokenPolicy, wrapped), or anything with batch(row),
              gate(batch) and tiles(batch, slot); heads(batch) -> (gate, value or None), when
              it has one, is called in place of gate (the critic's value for the record)
    row       the obs_row of the board the policy decides on (rs_rows.obs_row)
    state     the present state: hand, hand_costs, elixir, pending, towers, entities, and
              abilities (the buttons' rows)
    legality  BattleLegality / TowersOnlyLegality / PureLegality, a Battle, or a callable
              check(slot, card, x_sub, y_sub) -> bool
    cfg       a DecideConfig (None: legacy)
    rng       a numpy Generator; one random() per decision (argmax), or the samples (sampling)
    pending   our plays in flight (None: the state's own, rs_eval under a delay)
    tick      the decision tick for the record (None: row["tick"])
    dt        ticks this decision stands for (None: cfg.dt)
    aim       aim(order, slot, card) -> order: re-ranks a slot's tiles (rs_eval --spell-aim)
    blocked_tiles  bool[576], True = never tap there (ability buttons on screen)
    elixir_cap     the live runner's upper bound on our elixir after its own plays
                   (ElixirCap.observe; module doc, ELIXIR CAP): spendable is min(elixir_milli,
                   elixir_cap) minus the plays in flight. None (rs_eval, rs_match, RL): as before
    """
    cfg = DecideConfig() if cfg is None else cfg
    policy = as_policy(policy)
    team = int(team)
    if tick is None and "tick" in row:
        tick = int(row["tick"])
    dt = cfg.dt if dt is None else max(1, int(dt))
    if blocked_tiles is not None:
        blocked_tiles = np.asarray(blocked_tiles, dtype=bool).reshape(-1)
    p = state["players"][team]
    hand = list(p.get("hand") or [])
    costs = list(p.get("hand_costs") or [])
    b = policy.batch(row)
    heads = getattr(policy, "heads", None)
    if callable(heads):                     # TokenPolicyAdapter: the gate and the critic
        raw, value = heads(b)
        raw = np.asarray(raw)
    else:
        raw, value = np.asarray(policy.gate(b)), None
    if cfg.sample:
        g = raw.astype(np.float64)
    else:
        g = raw.copy() if raw.dtype == np.float32 else raw.astype(np.float32)
    g[0] += cfg.wait_bias
    spendable, held = _spendable(p, state, team, pending, cfg, elixir_cap)
    blocked = _pending_slots(held, hand) if cfg.mask_pending else set()
    mask = np.zeros(len(g), dtype=bool)
    mask[0] = True
    for slot in range(HAND_SIZE):
        cost = costs[slot] if slot < len(costs) and costs[slot] is not None else NO_COST
        ok = not (slot >= len(hand) or spendable < cost * 1000 or hand[slot] is None
                  or hand[slot] < 0)
        if ok and cfg.mask_unplayable and cost < 0:
            ok = False
        if ok and slot in blocked:
            ok = False
        mask[1 + slot] = ok
        if not ok:
            g[1 + slot] = MASKED
    n_buttons = len(g) - GATE_PLAYS
    if n_buttons > 0:
        # E15: ability buttons, only with cfg.presses (button_mask); else all masked.
        if cfg.presses:
            pressing = (_pending_buttons(state, team, pending, held) if cfg.mask_pending
                        else frozenset())
            allowed = button_mask(state, team, n_buttons, spendable, pressing=pressing)
        else:
            allowed = [False] * n_buttons
        for k, ok in enumerate(allowed):
            mask[GATE_PLAYS + k] = bool(ok)
            if not ok:
                g[GATE_PLAYS + k] = MASKED
    pr = np.exp(g - g.max())
    pr /= pr.sum()
    p_wait = float(pr[0])
    q_wait = p_wait ** (dt / WAIT_REF_TICKS)
    cards_ok = not g[1:GATE_PLAYS].max() < UNPLAYABLE
    playable = cards_ok or bool(mask[GATE_PLAYS:].any())
    capped = bool(cfg.cap_milli is not None and cards_ok and spendable >= cfg.cap_milli)
    base = dict(team=team, tick=tick, p_wait=p_wait, q_wait=q_wait, capped=capped,
                mask=tuple(bool(m) for m in mask), spendable_milli=int(spendable), mode=cfg.name,
                value=() if value is None else tuple(float(v) for v in value))

    def play(slot: int, t: int, wx: int, wy: int, lane: int | None, tried: list[int],
             check: Bound, **extra: Any) -> Decision:
        cost = costs[slot] if slot < len(costs) else 0
        return Decision(kind=PLAY, slot=int(slot), card=int(hand[slot]), cost=max(0, int(cost)),
                        tile=int(t), tx=int(t) % GRID_W, ty=int(t) // GRID_W,
                        x_sub=int(wx * sub), y_sub=int(wy * sub), fallback=len(tried) > 1,
                        tried=tuple(tried), lane=lane, legality=check.name, **base, **extra)

    def press(button: int, tried: list[int], check: Bound | None, **extra: Any) -> Decision:
        return Decision(kind=PRESS, button=int(button), card=button_card(state, team, button),
                        cost=press_cost(state, team, button), fallback=bool(tried),
                        tried=tuple(tried), legality="" if check is None else check.name,
                        **base, **extra)

    if cfg.sample:
        return _decide_sampled(policy, b, state, team, legality, cfg, rng, g, mask, hand, dt,
                               sub, aim, blocked_tiles, playable, capped, base, play, press)
    u = float(rng.random())                 # always drawn: a seeded run replays exactly
    if not playable:
        return Decision(kind=WAIT, reason=R_NO_CARD, **base)
    if not capped and u < q_wait:
        return Decision(kind=WAIT, reason=R_WAIT, **base)
    if cfg.wait_floor is not None and p_wait >= cfg.wait_floor:
        # Rule 6 (module doc, DECISION RULES): the cap wins, also without a cap rule.
        cap = CAP_ELIXIR if cfg.cap_milli is None else cfg.cap_milli
        if not (cards_ok and spendable >= cap):
            return Decision(kind=WAIT, reason=R_FLOOR, **base)
    # The argmax over the slots and buttons (g[1:]: slot s is entry s, button k is
    # HAND_SIZE + k); the fallback goes on in gate order. A masked entry is -1e9, so with every
    # button masked this is the slots' argmax and order, exactly as before E15.
    first = int(np.argmax(g[1:]))
    entries = [first]
    if cfg.fallback:
        entries += [int(s) for s in np.argsort(-g[1:], kind="stable")
                    if mask[1 + int(s)] and int(s) != first]
    check: Bound | None = None
    tried: list[int] = []
    strays: list[Any] = []                  # rule 7: [the guard] once a stray card is tried
    for slot in entries:
        if slot >= HAND_SIZE:
            return press(slot - HAND_SIZE, tried, check)
        if check is None:
            check = _bind(legality, state, team)
        tried.append(slot)
        card = int(hand[slot])
        order = np.argsort(-np.asarray(policy.tiles(b, slot)))
        if aim is not None:
            order = aim(order, slot, card)
        lane = threat_lane(state, team, sub) if card in cfg.guard_cards else None
        guard = None
        if card in cfg.stray_cards:
            if not strays:
                strays.append(stray_guard(state, team, sub))
            guard = strays[0]
        skipped = 0                         # rule 7: legal tiles of this card turned down
        for t in order[:cfg.top_tiles]:
            t = int(t)
            tx, ty = t % GRID_W, t // GRID_W
            if blocked_tiles is not None and blocked_tiles[t]:
                continue
            if _guarded(lane, tx):
                continue
            wx, wy = to_world(tx * 1000 + 500, ty * 1000 + 500, team)
            if check(slot, card, wx * sub, wy * sub):
                if guard is not None and not guard.keeps(tx, ty):
                    skipped += 1
                    continue
                return play(slot, t, wx, wy, lane, tried, check,
                            **({"stray": skipped} if skipped else {}))
        if skipped:
            # The guard turned down every legal tile it was shown: it holds the play (module
            # doc, DECISION RULES), never hands it to the next card.
            return Decision(kind=WAIT, reason=R_NO_TILE, tried=tuple(tried),
                            legality=check.name, stray=skipped, **base)
    return Decision(kind=WAIT, reason=R_NO_TILE, tried=tuple(tried),
                    legality="" if check is None else check.name, **base)


def tile_probs(logits: np.ndarray, explore: float = SAMPLE_EXPLORE,
               temp: float = SAMPLE_TEMP) -> np.ndarray:
    """(1 - explore) x softmax(logits / temp) + explore spread evenly (rs_selfplay.tile_probs)."""
    lg = np.asarray(logits, dtype=np.float64)
    q = np.exp((lg - lg.max()) / temp)
    q /= q.sum()
    return (1 - explore) * q + explore / len(lg)


def gate_probs(g: np.ndarray, mask: np.ndarray, dt: int = DECIDE, capped: bool = False
               ) -> np.ndarray:
    """The gate as a distribution (rs_selfplay.gate_probs): wait p_wait ** (dt / 20) (0 when
    capped), the rest in proportion to the slot softmax among the unmasked entries."""
    g = np.where(mask, np.asarray(g, dtype=np.float64), MASKED)
    pr = np.exp(g - g.max())
    pr /= pr.sum()
    qw = 0.0 if capped else float(pr[0]) ** (dt / WAIT_REF_TICKS)
    s = float(pr[1:].sum())
    q = np.zeros(len(g))
    q[0] = qw
    q[1:] = (1 - qw) * pr[1:] / max(s, 1e-12)
    q = np.where(mask, q, 0.0)
    return q / q.sum()


def _decide_sampled(policy, b, state, team, legality, cfg, rng, g, mask, hand, dt, sub, aim,
                    blocked_tiles, playable, capped, base, play, press) -> Decision:
    """decide() with cfg.sample: the gate and the tile are drawn (module doc, PRESETS). A drawn
    button (a >= GATE_PLAYS; q is 0 there unless button_mask let it in) is a PRESS with no tile
    draw; the fallback goes on in q order over the slots and buttons."""
    if not playable:
        return Decision(kind=WAIT, reason=R_NO_CARD, **base)
    q = gate_probs(g, mask, dt, capped)
    a = int(rng.choice(len(q), p=q))
    drawn = dict(gate_q=tuple(float(v) for v in q), gate_a=a, logq_gate=float(math.log(q[a])))
    if a == 0:
        return Decision(kind=WAIT, reason=R_WAIT, **base, **drawn)
    first = a - 1
    entries = [first]
    if cfg.fallback:
        entries += [int(s) for s in np.argsort(-q[1:], kind="stable")
                    if mask[1 + int(s)] and int(s) != first]
    check = None
    tried: list[int] = []
    for slot in entries:
        if slot >= HAND_SIZE:
            return press(slot - HAND_SIZE, tried, check, **drawn)
        if check is None:
            check = _bind(legality, state, team)
        tried.append(slot)
        card = int(hand[slot])
        logits = np.asarray(policy.tiles(b, slot), dtype=np.float64)
        order = np.argsort(-logits)
        if aim is not None:
            order = aim(order, slot, card)
        lane = threat_lane(state, team, sub) if card in cfg.guard_cards else None

        def legal_at(t: int) -> tuple[bool, int, int]:
            tx, ty = t % GRID_W, t // GRID_W
            if (blocked_tiles is not None and blocked_tiles[t]) or _guarded(lane, tx):
                return False, 0, 0
            wx, wy = to_world(tx * 1000 + 500, ty * 1000 + 500, team)
            return check(slot, card, wx * sub, wy * sub), wx, wy

        legal: list[tuple[int, int, int]] = []
        seen: set[int] = set()
        for t in order[:cfg.top_tiles]:
            t = int(t)
            seen.add(t)
            ok, wx, wy = legal_at(t)
            if ok:
                legal.append((t, wx, wy))
                if len(legal) == cfg.sample_top:
                    break
        if not legal:
            continue
        for t in rng.permutation(len(logits))[:cfg.sample_scan]:
            if len(legal) == cfg.sample_top + cfg.sample_extra:
                break
            t = int(t)
            if t in seen:
                continue
            seen.add(t)
            ok, wx, wy = legal_at(t)
            if ok:
                legal.append((t, wx, wy))
        pt = tile_probs(np.array([logits[t] for t, _, _ in legal]), cfg.explore, cfg.tile_temp)
        k = int(rng.choice(len(legal), p=pt))
        t, wx, wy = legal[k]
        return play(slot, t, wx, wy, lane, tried, check, **drawn,
                    tile_cands=tuple(c for c, _, _ in legal), tile_a=k,
                    logq_tile=float(math.log(pt[k])))
    return Decision(kind=WAIT, reason=R_NO_TILE, tried=tuple(tried),
                    legality="" if check is None else check.name, **base, **drawn)
