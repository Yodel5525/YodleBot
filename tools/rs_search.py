r"""Search on a rebuilt board (docs/LIVE_SEARCH_PLAN.md): BatchedPolicy (S05), the search core (S06),
the simulator seat (S08) and the privileged true-copy trials (S09).

    from tools import rs_search as SR             # or `import rs_search` (tools/ on sys.path)
    policy = SR.BatchedPolicy(loaded.model)       # rs_checkpoint.load_policy(...).model, eval mode
    d0 = rs_decide.decide(policy, row, state, team, legality, cfg, rng=rng)   # as with the adapter
    enc = policy.encode_rows(rows)                # many rows, one encoder pass
    gate, value = enc.head(2)                     # TokenPolicyAdapter.heads of row 2
    logits = enc.tiles_for([(0, 1), (0, 3)])      # tile logits of (row, hand slot) pairs, one call
    v = policy.values(after_rows, policy.value_kind)    # the search's V per after-row, one pass

WHY (plan 6.1, "The shared encoder pass"). In TokenPolicy._heads (sim/model_tokens.py:251-270)
the gate, the value head and the critic read only the encoder's output, and the tile logits are
tile_head(cat[cls, card_emb(card_ids)]) (:266): encode_all (:195-236) never reads card_ids, so
`cls` is the same for every card. One outputs() call (:246-249, the dict with "cls") therefore
gives the gate, the values and cls of a whole batch, and the 576 tile logits of any (row, card)
cost only the small tile head. rs_decide.TokenPolicyAdapter (rs_decide.py:612-664) runs a full
forward for heads() and another for each tiles(b, slot) the decision tries (rs_decide.py:1285,
1354); the old search ran one per candidate (rs_eval.py:703-704). The model is in eval mode
(rs_checkpoint.load_policy, rs_checkpoint.py:322), so dropout is off and a row's outputs do not
depend on the pass it is in. This module uses sim/model_tokens read-only: it calls outputs(),
tile_head and card_emb and changes nothing on the model.

BatchedPolicy(model, torch=None, *, batch=None, device=None, value_kind=None)
  TokenPolicyAdapter's interface, so rs_decide.decide takes it in place of the adapter:
  batch(row)         the adapter's batch of one row: `batch` (RSAgent.batch) when given, else
                     rs_decide.row_batch(row, torch, model_schema_of(model)); on the model's
                     device
  heads(b)           (gate logits of row 0 [wait, slot 0..3, buttons...] numpy float32, the
                     critic's raw [outcome, won logit] or None): exactly the adapter's, from the
                     batch's one encoder pass
  gate(b)            heads(b)[0]
  tiles(b, slot)     the 576 tile logits of row 0 for the card in hand slot `slot` (card_ids =
                     b["hand"][:, slot], as the adapter has it), from the cached pass of `b`
  schema, has_critic as the adapter's
  and for many rows:
  stack(rows)        one batch of the rows' batch(row)s, concatenated (rows of one batch share
                     their widths: v2 rows the same ability-button count B)
  encode(b)          the Encoded pass of a batch, cached by the batch's identity (the last
                     CACHE_BATCHES batches): heads(b) then tiles(b, slot) run the encoder once
  encode_rows(rows)  encode(stack(rows)), not cached (the caller holds the Encoded)
  values(rows, kind) V per row, float64 [N], one encoder pass (below; kind None: the critic's,
                     tools/rl/teacher.policy_values' protocol)
  stats              Counter: passes, rows, cache_hits, tile_calls, tile_rows
  clear()            drops the cache

A BATCH IS READ AS IMMUTABLE. The cache holds the batch dict itself and compares by identity
(`is`), so a dict that is freed and a new one at the same address never match; a batch changed in
place after its pass would keep the old pass. rs_decide.decide builds a batch and never changes it
(rs_decide.py:1195). A model in training mode is refused (BatchedPolicyError): with dropout on, a
row's outputs would depend on the pass and the shared pass would not be the adapter's.

Encoded (one pass over N rows; numpy float32 unless said)
  n, len()           the rows
  gate [N, G], value [N, 2], value2 [N, 2] or None (a model with a critic), cls (the encoder's
  [N, d] tensor, on the model's device), batch (the stacked tensors)
  head(i)            the adapter's heads() of row i: (gate[i], tuple(value2[i]) or None)
  values(kind)       V per row (below; kind None: the policy's value_kind, the search's V)
  tiles(i, slot)     [576] for the card in row i's hand slot `slot`
  tiles_for(pairs)   [M, 576] for (row, slot) pairs, one tile-head call
  tiles_cards(rows, tokens)   [M, 576] for (row, card token) pairs, one tile-head call; a token
                     is a card id + 2 (rs_rows), what a row's hand column holds

VALUES (plan 3.1 step 9: score = swing + V + ...). kind:
  "swing"   the value head's column 0 (sim/model_tokens.py:149, 267): the predicted tower swing
            over the next HORIZON ticks (rs_rows.py:292, the "outcome" label), in tower_swing's
            units (tools/rl/teacher.py:282). A legacy model's V.
  "critic"  tanh(z / 2) of the critic's won logit z = value2[:, 1] (ppo.critic_value,
            tools/rl/ppo.py:887-893): exactly what tools/rl/teacher.policy_values reads
            (teacher.py:318-366), which calls a policy's own values(rows) first; so this
            BatchedPolicy gives the teacher the same V as its batched TokenPolicyAdapter path.
  None      BatchedPolicy.values(rows): "critic", the teacher's protocol, unless the constructor
            was given a value_kind; a model without a critic then raises BatchedPolicyError, as
            the teacher's adapter path raises TeacherError, instead of handing the teacher the
            value head's swing (another unit) as if it were tanh(z / 2). Encoded.values() and
            value_kind: the search's V (plan 3.1 step 9), "critic" for a model with a critic,
            else "swing"; the search asks values(rows, policy.value_kind) or
            encode_rows(rows).values(). ("critic" without a critic is refused.)
The potential Phi, the terminal result (ppo.result_of, scaled for the value head) and the swing
are the scorer's (S06), not the policy's.

DEVICE. device None: the device of the model's first parameter (CPU when it has none). On the CPU
the batches are the adapter's tensors, unmoved; on another device every batch moves there and the
outputs come back to the CPU as numpy. The adapter itself is CPU only.

THE SEARCH (S06; plan section 3, the inline backend). One Searcher per seat (the simulator's
SearchSeat, S08; the live SearchDecider, S13), one call per decision:

    searcher = SR.Searcher(policy, rebuilder, SR.SearchConfig.named("s_idle"), dcfg=dcfg,
                           landing=RD.obs_delay_ticks(meta) == 0, law=law)
    d, rec = searcher.decide(view, team=team, land_ticks=D, legality=legality, rng=seat.rng,
                             row_fn=row_fn, tick=tick, pending=pending, seed=spec_seed)
    d.command          # what to send NOW: a Decision, as rs_decide.decide gives one
    rec.to_json()      # the search record (S13: the decision record's extra)
    searcher.summary() # the counters (S08: seat_info)

  view        rs_public_view.public_view (Permanence applied; live, public_view(live=True)).
              Asserted public on every call (rs_public_view.is_public; SearchLeakError): the
              search reads no hidden state. The only real battle it reaches is d0's legality.
  land_ticks  D, our decision-to-landing delay (simulator: the seat's command delay; live: the
              measured land delay): our play would land at B = view tick + D.
  legality    d0's legality object, the plain seat's (BattleLegality(battle) in the simulator,
              TowersOnlyLegality live); the candidates' tap-time check is the same object.
  rng         the seat's generator: rs_decide.decide draws from it exactly as the plain seat's
              decide does, once per decision, whatever the search then does; the search never
              draws from it (opponent worlds, S07, draw from their own, seeded by (seed, tick)).
  row_fn      row_fn(state, tick) -> the policy's row of `state` dated `tick` (the seat's obs_row
              with its memory view; live, the observer's row).
  pending, dt, aim, blocked_tiles, tick   passed to rs_decide.decide as the plain seat passes them.
  opp_fill    cards for the opponent's open cycle slots (a deck prior's draw, S03 / S07).
  seed        the spec seed: the rebuild's seed and the worlds' generator come from (seed, tick).

  1. ROOT (cfg.lookahead and a landing-time checkpoint, `landing`): rebuilder.spec(view,
     land_tick=B) and root() (S04: our in-flight plays stepped in at their exec ticks, then on
     to B). Refused (rs_rebuild.Refusal), a fidelity below rebuilder.settings.min_fidelity, or a
     root already over: no look-ahead and no trials ("rebuild:<reason>", "low_fidelity",
     "root_game_over").
  2. d0 = rs_decide.decide(policy, row, view, team, legality, dcfg, rng=rng, ...) on the ROOT
     ROW, row_fn(rs_rebuild.with_present(root state, view), B): the board at B with the present
     hand, elixir, deck and next card, dated B (rs_eval.py:294-300, 542-544); without a usable
     root, on the present row, row_fn(view, view tick). The batch of that row is built once and
     handed to decide, so decide's gate, the priors and every candidate tile share one encoder
     pass (BatchedPolicy caches it by the batch's identity). decide's own `state` is the view:
     our block is the seat's whole own block, the board is the public one.
  3. cfg.max_trials 0 (s_root) stops here: d0 exactly as decide gave it ("root"). So does a press
     ("press"), and a battle where search is off ("disabled").
  4. HOLD. After the search chose waiting over a play, that verdict holds for cfg.hold_ticks
     (the simulator's cadence, 10) on the decision clock (decide's clock_tick, else the view's
     tick): a play d0 draws meanwhile becomes a wait ("search_hold"; never when d0 is capped).
     A new opponent play ends the hold early. In the simulator the seat's clock counts its
     decisions (one per cadence), so at hold_ticks = cadence the next decision is never held:
     the hold acts live only, where decisions come faster than the cadence.
  5. TRIGGERS (any): d0 is a play ("play"); the view shows more opponent plays than at the last
     search ("opp_play"); d0's spendable elixir >= cfg.elixir_trigger ("elixir"); rs_decide.
     threat_lane reports a push on the view ("push"). None: d0 ("not_triggered"). For a
     decision-time checkpoint (no look-ahead) the root is rebuilt only now.
  6. CANDIDATES (Candidate; at most cfg.max_trials, in this order; the incumbent first):
       P  d0's play (role "P"); it must be legal at B too, else d0 ("incumbent_illegal")
       W  waiting: the incumbent when d0 waited; else a candidate unless d0 was capped (the cap
          rule) or cfg.wait is off
       T  P's card at cfg.extra_tiles more tiles of its best cfg.top_tiles, each at least
          cfg.tile_gap tiles (Chebyshev, own frame) from P and from each other
       S  the cfg.other_slots best other unmasked slots by rs_decide.gate_probs, cfg.other_tiles
          tiles each from their best cfg.top_tiles
     A tile passes d0's tap-time legality (bound to the view, the same object), the lane guard
     (DecideConfig.guard_cards, rs_decide.threat_lane on the view) and blocked_tiles as d0 does,
     and check_deploy on the root at B with the card mapped to its slot there
     (Rebuilder.legal). prior = log q(slot) + log-softmax(tile) (rs_eval.py:642; W: log q(wait)),
     q the gate as decide's distribution (gate_probs with d0's mask, cadence and cap).
     Candidates with a prior below the incumbent's - cfg.prior_drop are dropped; fewer than
     cfg.min_candidates: d0 ("too_few_candidates").
  7. TRIALS. Per candidate and world (World; s_idle: one, the root itself, no opponent play):
     Rebuilder.trial(world root, the world's plays + ours at B, cfg.horizon): load the root,
     step our play in at B (its card mapped to its slot when it is sent; our later in-flight
     plays too), the relabelled after-state. A play the trial refuses or skips did not run. Live
     (realtime=True) the trials stop at cfg.deadline_ms after the call began, the incumbent and
     the next candidate first. The incumbent did not run: d0 ("incumbent_missing", or
     "deadline"); fewer than cfg.min_trials ran: d0 ("too_few_trials").
  8. SCORE, per world: swing + V - cfg.leak_weight x leak, and per candidate the CVaR at
     cfg.cvar_alpha over the worlds (one world: itself) + cfg.prior_weight x prior.
       swing  the tower-fraction swing from the world's root to the after-state
              (tools/rl/teacher.tower_swing)
       V      the policy's value of the after-row, row_fn(after, its tick), one batched pass:
              policy.values(rows, policy.value_kind). "swing" (a legacy model: the value head's
              predicted tower swing, swing's units); "critic": tanh(z / 2) + Phi(after)
              (tools/rl/ppo.potential, the teacher's V + Phi). A finished game scores
              ppo.result_of x cfg.terminal_scale (3) for "swing", x 1 for "critic".
       leak   the regeneration over [B, B + H) that does not fit under the cap (elixir), given
              our elixir at B on the root, the candidate's cost and our later in-flight plays
              landing before B + H (rs_elixir.ElixirLaw.regen)
  9. CHOOSE (choose): the best total replaces the incumbent only by the trust region: one world,
     total(best) >= total(incumbent) + cfg.trust; K worlds, the mean paired gain >= trust + z x
     sd / sqrt(K). Else d0, tagged ("searched", a veto).
 10. THE DECISION: dataclasses.replace(d0, ...) (rs_decide.Decision is frozen); mode d0.mode +
     "+search"; a chosen play keeps its PRESENT slot (the command is sent now) with its tile,
     tap point, cost and lane; a chosen wait is kind WAIT, reason "search_wait". vision_play's
     _decide and to_record read it unchanged.
FALLBACKS (SearchRecord.outcome) are counted per reason; an exception inside the search plays d0
("exception"), and cfg.max_errors of them in one battle turn search off for the rest of it
("disabled"; the decision clock going back, or new_battle(), starts a new battle). A rebuilder
refusal "error" counts as one. SearchLeakError and the config errors are never swallowed. Every
decision whose look-ahead root was not usable is also counted by its reason in summary()'s
root_fallbacks ("rebuild:<reason>", "spec_refused", "low_fidelity", "root_game_over",
"exception"): s_root's outcome is always "root", so its fallbacks show only there.

PRESETS (SearchConfig.named / resolve; plan 3.2). A preset name expands to its full dict
(expand), so a game key records what the preset holds, not just its name:
  s_root   look-ahead alone (max_trials 0, opponent "none")
  s_idle   look-ahead + at most 8 trials, H 60, one world, an idle opponent; deadline 80 ms live
  s_reply  look-ahead + at most 8 trials, H 80, K 4 worlds, opponent "reply": an in-flight play
           and one reply from an opponent model (S07). Until S07 a Searcher refuses it without
           a `worlds` provider (SearchConfigError): it never runs as an idle search in disguise.
  s_true   as s_idle with the trials on the TRUE delayed battle: the privileged ceiling (M2b;
           THE TRUE COPY below). PRIVILEGED, NEVER PROMOTABLE: SearchConfig.privileged;
           check_promotable and check_live raise PrivilegedSearchError for it (the live runner
           is to call check_live, S14); tools/rs_league_eval.py gives a candidate named so, or
           whose games say so, the gate decision "privileged" (never promoted, never admitted;
           it reads is_privileged, not check_promotable); a Searcher refuses it
           unless built with privileged=True, and every SearchRecord and summary() of it says
           privileged.
"off" / None resolve to no search. level (None: the rebuilder's) picks R1 / R2 per search; R0
rebuilds from the seat's privileged state and belongs to the regret instrument (S10), so it is
refused here. resolve() also takes a seat's search (seat_search, below): "name@tokens" text and a
mapping with "rebuild" give the SearchConfig alone.

THE SEAT (S08; tools/rs_match.py). An rs_match.AgentSpec with `search` is played by SearchSeat:

    RM.AgentSpec("runs/tokens/live_big_hog/best.pt", search="s_root", latency_ticks=2)
    .venvs\royalegym\Scripts\python.exe tools\rs_match.py --a CKPT --search-a s_root ...

  seat_search(value)  what AgentSpec.search holds (AgentSpec normalises it, so the game keys of
                      tools/rs_league_eval.py record the content): SearchConfig.to_json() plus
                      "rebuild", the rs_rebuild.RebuildSettings the seat's rebuilder runs, pinned
                      when the spec is made (the provenance fields "probes" and "notes" left out,
                      so the key is the same on every machine). A preset name takes the plan's
                      defaults (RebuildSettings()); "name@probes" the newest S01 engine report's
                      probe-driven settings (RebuildSettings.from_report, read once, now);
                      "name@R2" (or "@probes+R2") the rebuild level; a mapping's "rebuild" is
                      "defaults", "probes", such tokens or a RebuildSettings dict. A rebuild level
                      R0 is refused (the regret instrument's).
  SearchSeat          rs_match.MatchAgent with SearchSeatMixin (rs_search.SearchSeat; the class
                      is made on first use, so importing this module never imports rs_match).
                      rs_match.build_seat builds one and calls setup_search(search, team, delay,
                      opp_delay, seed) with the delays of the spec's engine config (the spec delay
                      plus each agent's latency_ticks: MatchSpec.engine_config), never the battle's.
                      act(battle, seen, team, mem, tick, sub, royalesim), each decision:
                        view   rs_public_view.public_view(seen, team, mem): the opponent's elixir
                               counted by the seat itself (SeatOppElixir: MatchAgent.opp_elixir,
                               the plays that ran), our plays sent on our OWN clock (below), the
                               engine's refusals (unran) left out, then Permanence.apply (a
                               hidden Tesla kept); public["dropped"] (how many unseen enemies
                               the cut removed: privileged itself) is taken out
                        rows   search_row: rs_rows.obs_row(state, team, sub, mem, tick, live,
                               visibility, **v2_kw), RSAgent.act's row with lead 0
                        d      Searcher.decide(view, land_ticks = the seat's delay, legality =
                               rs_decide.BattleLegality(battle), rng = the seat's, seed = the spec
                               seed); its command is what the seat sends
                      OUR OWN CLOCK. run_match dates seat.sent on the true clock (true tick +
                      delay) while the view's tick is the noised one (perception noise's clock),
                      so the seat keeps its own sends as live dates its taps: (view tick + delay,
                      card, x, y), paired by order with run_match's seat.sent (one entry per card
                      play) to leave out the plays the engine refused. The seat never needs the
                      true clock. Clock noise on both the send's tick and now's can misplace a
                      recent play on either side of the in-flight cut, so the cut is checked
                      against our hand, the live rule (rs_decide.prune_pending): a card still in
                      the hand has not run (it stays in flight), one that left it has (_by_hand;
                      counted in search_info "seat": inflight_late, inflight_ran). The noised
                      tick also steps back now and then (sigma 2: rarely; doubled noise: about
                      15 times a game), so nothing in the seat takes that for a new battle: the
                      Searcher's hold and battle memo run on the seat's decision clock (its
                      decisions x cadence: clock_tick), and Permanence and the rebuilder's
                      spell sightings never reset on a rewind (rewind_ticks None). A SearchSeat
                      plays one match (build_seat makes one per match; another match's refusals
                      Counter raises SearchError).
                      search_info() is what rs_match.seat_info records: the preset, its key, the
                      rebuild settings' key and level, privileged, the delays, Searcher.summary()
                      (decisions, searches, overrides by incumbent > choice, vetoes, fallbacks
                      by reason and the root's (root_fallbacks), triggers, ms per stage), the
                      spread of ms per decision (decision_ms) and of what the search adds to
                      the plain decision's row and d0 (search_ms: the latency charge's measure,
                      plan 6.3, latency_ticks_for).
                      A SearchSeat refuses a lead (the look-ahead is the search's), hold_pending
                      (it reads the battle's pending commands) and s_reply until S07 gives the
                      opponent world model (check_seat). The battle it reaches: check_deploy
                      through d0's legality (d0's and the candidates' tap-time checks); s_true
                      also save / load / step / state_json (THE TRUE COPY).
                      tests/pipeline/test_search_no_leak.py holds it to that.
  merge_search_info   sums search_info() over games (rs_match's command line prints it).

THE TRUE COPY (S09; opponent "true", s_true). The privileged ceiling of M2b: the same root, d0,
triggers, candidates (legal at tap time and at B on the PUBLIC root), scoring and choice as the
public search, but each trial runs on the TRUE match battle, as tools/rl/teacher.annotate does
(teacher.py:509-601): one save(), then per candidate load() and step([its command, sent now
from its PRESENT slot], D + H) under the battle's own delay, so it lands at B with the
opponent's real queued commands; the battle is loaded back and its state_json checked equal to
the one before (TrueCopyError otherwise: a match whose battle the search changed cannot go on,
so it is never swallowed). A play refused at send (the step's reason) or when it ran
(step_commands_run) did not run. The swing runs from the true state at t (the teacher's: the
ticks (t, B] are the same for every candidate, so they cancel in the choice), V reads the true
after-state's row, the leak the public root's elixir at B. decide() needs true_battle= for it and
refuses true_battle= for any public preset (PrivilegedSearchError).

Importing this module imports numpy, rs_decide, rs_elixir, rs_public_view and rs_rebuild (all
torch-free; royalesim only when a Battle is built), never torch, sim.model_tokens, royalesim or
rs_match: BatchedPolicy imports torch when it is built, as TokenPolicyAdapter does; tools/rl/
teacher and ppo (numpy) only when a trial is scored; rs_match and rs_rows when SearchSeat is
first asked for or a seat acts. Nothing here taps, captures or touches the emulator, and nothing
imports scripts/gamestate (the memory reader).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:          # also when imported as tools.rs_search
    sys.path.insert(0, str(ROOT / "tools"))

import rs_decide as RD  # noqa: E402  (row_batch, model_schema_of, TILES; numpy only)
import rs_elixir  # noqa: E402  (pure Python: the elixir law, for the leak)
import rs_memory  # noqa: E402  (standard library only: card tokens, for the seat's refusals)
import rs_public_view as PV  # noqa: E402  (pure Python: is_public, the leak guard)
import rs_rebuild as RB  # noqa: E402  (pure Python until a Battle is built: with_present, R1/R2)

__all__ = [
    "BatchedPolicy",
    "BatchedPolicyError",
    "CACHE_BATCHES",
    "Candidate",
    "Choice",
    "Encoded",
    "OPPONENTS",
    "OUTCOMES",
    "PRESETS",
    "PrivilegedSearchError",
    "SearchConfig",
    "SearchConfigError",
    "SearchError",
    "SearchLeakError",
    "SearchRecord",
    "SearchSeatMixin",
    "Searcher",
    "TrueCopyError",
    "TrueRoot",
    "VALUE_CRITIC",
    "VALUE_KINDS",
    "VALUE_SWING",
    "World",
    "as_batched",
    "as_play",
    "as_wait",
    "check_live",
    "check_promotable",
    "check_seat",
    "choose",
    "cvar",
    "expand",
    "is_privileged",
    "latency_ticks_for",
    "merge_search_info",
    "opponent_checkpoint",
    "probe_report_source",
    "probes_source",
    "rebuild_settings",
    "resolve",
    "seat_config",
    "seat_search",
    "tagged",
]
# SearchSeat (rs_match.MatchAgent + SearchSeatMixin) is made on first use by the module
# __getattr__ at the end, so that importing this module never imports rs_match.

#: values(): V from the value head's column 0 (the predicted tower swing; a legacy model).
VALUE_SWING = "swing"
#: values(): V = tanh(z / 2) of the critic's won logit (ppo.critic_value; a model with value2).
VALUE_CRITIC = "critic"
VALUE_KINDS = (VALUE_SWING, VALUE_CRITIC)
#: Batches whose encoder pass encode() keeps, by identity, most recent first.
CACHE_BATCHES = 4
#: What BatchedPolicy needs of a model (sim/model_tokens.TokenPolicy has them all).
MODEL_PARTS = ("outputs", "tile_head", "card_emb")


class BatchedPolicyError(ValueError):
    """A model, a batch or a value kind BatchedPolicy cannot serve."""


def _model_device(model: Any) -> Any:
    """The device of the model's first parameter, None when it has none."""
    params = getattr(model, "parameters", None)
    if not callable(params):
        return None
    for p in params():
        return p.device
    return None


# ---------------------------------------------------------------------------
# one encoder pass
# ---------------------------------------------------------------------------

class Encoded:
    """One encoder pass over N rows (module doc, Encoded)."""

    __slots__ = ("policy", "batch", "n", "gate", "value", "value2", "cls")

    def __init__(self, policy: "BatchedPolicy", batch: Mapping[str, Any],
                 out: Mapping[str, Any]):
        self.policy = policy
        self.batch = batch
        self.cls = out["cls"]
        self.gate = policy._np(out["gate"])
        self.value = policy._np(out["value"])
        v2 = out.get("value2") if policy.has_critic else None     # the adapter's rule
        self.value2 = None if v2 is None else policy._np(v2)
        self.n = int(self.gate.shape[0])

    def __len__(self) -> int:
        return self.n

    def _row(self, index: int) -> int:
        i = int(index)
        if not 0 <= i < self.n:
            raise IndexError(f"row {index} of a pass over {self.n} rows")
        return i

    def head(self, index: int = 0) -> tuple[np.ndarray, tuple[float, ...] | None]:
        """TokenPolicyAdapter.heads of row `index`: (its gate logits, the critic's raw
        [outcome, won logit] or None without a critic)."""
        i = self._row(index)
        value = None
        if self.value2 is not None:
            value = tuple(float(v) for v in self.value2[i].reshape(-1))
        return self.gate[i].copy(), value

    def values(self, kind: str | None = None) -> np.ndarray:
        """V per row, float64 [N] (module doc, VALUES)."""
        kind = self.policy.value_kind if kind is None else kind
        if kind == VALUE_SWING:
            return self.value[:, 0].astype(np.float64)
        if kind == VALUE_CRITIC:
            if self.value2 is None:
                raise BatchedPolicyError("values: kind 'critic' needs a model with a critic "
                                         "(value2); this one has none: use 'swing'")
            return np.tanh(self.value2[:, 1].astype(np.float64) / 2)
        raise BatchedPolicyError(f"values: kind one of {VALUE_KINDS} or None, got {kind!r}")

    def tiles(self, index: int, slot: int) -> np.ndarray:
        """The 576 tile logits of row `index` for the card in its hand slot `slot`."""
        i = self._row(index)
        return self._tile_head([i], self.batch["hand"][i:i + 1, int(slot)])[0]

    def tiles_for(self, pairs: Iterable[Sequence[int]]) -> np.ndarray:
        """[M, 576] tile logits of (row, hand slot) pairs, in order, one tile-head call."""
        pairs = [(self._row(i), int(s)) for i, s in pairs]
        if not pairs:
            return np.zeros((0, RD.TILES), dtype=np.float32)
        torch = self.policy.torch
        dev = self.batch["hand"].device
        rows = torch.as_tensor([i for i, _ in pairs], dtype=torch.long, device=dev)
        slots = torch.as_tensor([s for _, s in pairs], dtype=torch.long, device=dev)
        return self._tile_head([i for i, _ in pairs], self.batch["hand"][rows, slots])

    def tiles_cards(self, rows: Iterable[int], tokens: Iterable[int]) -> np.ndarray:
        """[M, 576] tile logits of (row, card token) pairs, one tile-head call."""
        rows = [self._row(i) for i in rows]
        tokens = [int(t) for t in tokens]
        if len(rows) != len(tokens):
            raise BatchedPolicyError(f"tiles_cards: {len(rows)} rows for {len(tokens)} tokens")
        if not rows:
            return np.zeros((0, RD.TILES), dtype=np.float32)
        torch = self.policy.torch
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.batch["hand"].device)
        return self._tile_head(rows, ids)

    def _tile_head(self, rows: list[int], card_ids: Any) -> np.ndarray:
        """tile_head(cat[cls, card_emb(card_ids)]) for cls of `rows` (model_tokens.py:266)."""
        p = self.policy
        torch, model = p.torch, p.model
        if rows == list(range(self.n)):
            cls = self.cls
        else:
            cls = self.cls[torch.as_tensor(rows, dtype=torch.long, device=self.cls.device)]
        with torch.no_grad():
            tile = model.tile_head(torch.cat([cls, model.card_emb(card_ids)], -1))
        p.stats["tile_calls"] += 1
        p.stats["tile_rows"] += len(rows)
        return p._np(tile)


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------

class BatchedPolicy:
    """TokenPolicyAdapter's interface with one encoder pass per batch, plus values(rows)
    (module doc)."""

    def __init__(self, model: Any, torch: Any = None, *,
                 batch: Callable[[Mapping[str, Any]], dict] | None = None,
                 device: Any = None, value_kind: str | None = None,
                 cache: int = CACHE_BATCHES):
        missing = [k for k in MODEL_PARTS if not hasattr(model, k)]
        if missing:
            raise BatchedPolicyError(f"BatchedPolicy needs a TokenPolicy (sim/model_tokens.py: "
                                     f"{', '.join(MODEL_PARTS)}); {type(model).__name__} has no "
                                     f"{', '.join(missing)}: use rs_decide.TokenPolicyAdapter")
        if torch is None:
            torch = importlib.import_module("torch")
        self.model = model
        self.torch = torch
        self._batch = batch
        self.device = _model_device(model) if device is None else torch.device(device)
        self._move = self.device is not None and getattr(self.device, "type", "cpu") != "cpu"
        if value_kind not in (None, *VALUE_KINDS):
            raise BatchedPolicyError(f"value_kind one of {VALUE_KINDS} or None, got "
                                     f"{value_kind!r}")
        if value_kind == VALUE_CRITIC and not self.has_critic:
            raise BatchedPolicyError("value_kind 'critic' needs a model with a critic (value2)")
        self._value_kind_given = value_kind
        self.value_kind = value_kind or (VALUE_CRITIC if self.has_critic else VALUE_SWING)
        self.cache_size = max(0, int(cache))
        self._cache: list[tuple[Mapping[str, Any], Encoded]] = []
        self.stats: Counter = Counter()

    # ------------------------------------------------------------ the adapter's properties
    @property
    def schema(self) -> Any:
        """The model's v2 Schema, or None for a legacy model (TokenPolicyAdapter.schema)."""
        return RD.model_schema_of(self.model)

    @property
    def has_critic(self) -> bool:
        return bool(getattr(self.model, "has_critic", False)) and hasattr(self.model, "outputs")

    # ------------------------------------------------------------ batches
    def _cpu_batch(self, row: Mapping[str, Any]) -> dict:
        if self._batch is not None:
            return self._batch(row)
        return RD.row_batch(row, self.torch, self.schema)

    def _to_device(self, b: dict) -> dict:
        if not self._move:
            return b
        return {k: v.to(self.device) for k, v in b.items()}

    def _np(self, t: Any) -> np.ndarray:
        t = t.float()
        if self._move:
            t = t.cpu()
        return t.numpy()

    def batch(self, row: Mapping[str, Any]) -> dict:
        """The batch of one row, as TokenPolicyAdapter.batch builds it (on the model's device)."""
        return self._to_device(self._cpu_batch(row))

    def stack(self, rows: Iterable[Mapping[str, Any]]) -> dict:
        """One batch of `rows`: their batch(row)s concatenated along the first axis."""
        parts = [self._cpu_batch(r) for r in rows]
        if not parts:
            raise BatchedPolicyError("stack: no rows")
        if len(parts) == 1:
            return self._to_device(parts[0])
        keys = list(parts[0])
        for k, p in enumerate(parts[1:], 1):
            if list(p) != keys:
                raise BatchedPolicyError(f"stack: row {k} has the columns {list(p)}, row 0 "
                                         f"{keys} (rows of one schema only)")
        for name in keys:
            widths = {tuple(p[name].shape[1:]) for p in parts}
            if len(widths) > 1:
                raise BatchedPolicyError(f"stack: the rows differ in the width of {name} "
                                         f"{sorted(widths)} (v2 rows of one batch need the same "
                                         f"ability-button count)")
        torch = self.torch
        return self._to_device({name: torch.cat([p[name] for p in parts], 0) for name in keys})

    # ------------------------------------------------------------ the pass
    def _encode(self, b: Mapping[str, Any]) -> Encoded:
        if getattr(self.model, "training", False):
            raise BatchedPolicyError("the model is in training mode (dropout on): call "
                                     "model.eval() (rs_checkpoint.load_policy gives it so)")
        card_ids = b["hand"][:, 0].clamp(min=0)          # the adapter's heads() call
        with self.torch.no_grad():
            out = self.model.outputs(b, card_ids=card_ids)
        enc = Encoded(self, b, out)
        self.stats["passes"] += 1
        self.stats["rows"] += enc.n
        return enc

    def encode(self, b: Mapping[str, Any]) -> Encoded:
        """The Encoded pass of batch `b`, from the cache when `b` is a batch it holds."""
        for held, enc in self._cache:
            if held is b:
                self.stats["cache_hits"] += 1
                return enc
        enc = self._encode(b)
        if self.cache_size:
            self._cache.insert(0, (b, enc))
            del self._cache[self.cache_size:]
        return enc

    def encode_rows(self, rows: Iterable[Mapping[str, Any]]) -> Encoded:
        """One encoder pass over `rows` (stack), not cached."""
        return self._encode(self.stack(rows))

    def clear(self) -> None:
        """Drop the cached passes."""
        self._cache.clear()

    # ------------------------------------------------------------ the adapter's calls
    def heads(self, b: Mapping[str, Any]) -> tuple[np.ndarray, tuple[float, ...] | None]:
        """(the gate logits of row 0, the critic's raw [outcome, won logit] or None)."""
        return self.encode(b).head(0)

    def gate(self, b: Mapping[str, Any]) -> np.ndarray:
        return self.heads(b)[0]

    def tiles(self, b: Mapping[str, Any], slot: int) -> np.ndarray:
        """The tile logits of row 0 for the card in hand slot `slot`, without a new pass."""
        return self.encode(b).tiles(0, slot)

    # ------------------------------------------------------------ values
    def values(self, rows: Iterable[Mapping[str, Any]], kind: str | None = None) -> np.ndarray:
        """V per row, float64 [N], from one encoder pass (module doc, VALUES); [] -> [0]. kind
        None: the constructor's value_kind when it was given one, else "critic" (what
        tools/rl/teacher.policy_values expects of values(rows); BatchedPolicyError without a
        critic). The search's own V: values(rows, self.value_kind)."""
        rows = list(rows)
        if kind is None:
            kind = self._value_kind_given or VALUE_CRITIC
            if kind == VALUE_CRITIC and not self.has_critic:
                raise BatchedPolicyError(
                    "values(rows): the default kind is 'critic' (tanh(z / 2), what "
                    "tools/rl/teacher.policy_values reads) and this model has no critic: pass "
                    "kind='swing' (or value_kind='swing') for the value head's tower swing")
        if not rows:
            return np.zeros(0, dtype=np.float64)
        return self.encode_rows(rows).values(kind)


def as_batched(policy: Any, torch: Any = None, **kw: Any) -> Any:
    """A BatchedPolicy for `policy`: itself when it is one; for a TokenPolicyAdapter (anything
    with a model, torch, batch, gate and tiles) one of its model, torch and batch function; for
    a TokenPolicy one of it. A policy whose model lacks MODEL_PARTS, and any other policy (a
    test's scripted batch / gate / tiles), is returned unchanged."""
    if isinstance(policy, BatchedPolicy):
        return policy
    model = getattr(policy, "model", None)
    if (model is not None and all(hasattr(policy, k) for k in ("batch", "gate", "tiles"))
            and hasattr(policy, "torch")):
        if not all(hasattr(model, k) for k in MODEL_PARTS):
            return policy
        return BatchedPolicy(model, policy.torch if torch is None else torch,
                             batch=getattr(policy, "_batch", None), **kw)
    if all(hasattr(policy, k) for k in MODEL_PARTS):
        return BatchedPolicy(policy, torch, **kw)
    return policy


# ===========================================================================
# S06: the search core (module doc, THE SEARCH)
# ===========================================================================

class SearchError(RuntimeError):
    """The search cannot run as asked (module doc, THE SEARCH)."""


class SearchConfigError(ValueError):
    """A SearchConfig, preset or override that is not valid, or a Searcher it cannot build."""


class PrivilegedSearchError(SearchConfigError):
    """A privileged search (s_true: its trials read the TRUE battle) where only a public one may
    run: a promotion, live play, or a Searcher not built with privileged=True (module doc,
    PRESETS)."""


class SearchLeakError(SearchError):
    """The view handed to the search is not public (rs_public_view.is_public says what)."""


class TrueCopyError(SearchError):
    """S09: the true battle is not what it was after the true-copy trials (its save() / load()
    did not restore it), or its delay is not the seat's: the match cannot go on. Never
    swallowed as a fallback."""


#: The opponent in the trials (plan 3.2): none (no trials), idle, reply (S07), true (S09).
OPP_NONE, OPP_IDLE, OPP_REPLY, OPP_TRUE = "none", "idle", "reply", "true"
OPPONENTS = (OPP_NONE, OPP_IDLE, OPP_REPLY, OPP_TRUE)
#: SearchConfig.level: None (the rebuilder's own settings.level) or a public rebuild level.
SEARCH_LEVELS = (None, RB.R1, RB.R2)
#: A searched Decision's mode ends with this.
MODE_SUFFIX = "+search"
#: Decision.reason of a wait the search chose, and of one its hold kept.
REASON_WAIT = "search_wait"
REASON_HOLD = "search_hold"
#: Candidate roles (module doc, THE SEARCH step 6).
ROLE_P, ROLE_W, ROLE_T, ROLE_S = "P", "W", "T", "S"
#: (seed, tick) -> the rebuild's seed.
SEED_MOD = 2 ** 31 - 1
#: resolve(): what means no search.
OFF_NAMES = ("off", "none", "")
#: seat_search(): the key beside SearchConfig's fields that pins the seat's RebuildSettings, the
#: tokens of "name@tokens" ("+" between them), and the provenance fields a pinned dict leaves out.
SEAT_REBUILD = "rebuild"
REBUILD_DEFAULTS, REBUILD_PROBES = "defaults", "probes"
REBUILD_TOKENS = (REBUILD_DEFAULTS, REBUILD_PROBES, RB.R1, RB.R2)
REBUILD_PROVENANCE = ("probes", "notes")
#: SearchConfig fields a seat's search (AgentSpec.search, and so the game keys and identities of
#: tools/rs_league_eval.py) holds only when they differ from what its preset gives back (resolve):
#: fields that only check (verify), and every field added later (S07 adds opp_ckpt here), so a
#: new field leaves the key of every search that does not use it as it was. Every other field is
#: always held: a preset name expands to its content (plan 3.2).
SEAT_OPTIONAL = ("verify",)
#: The same for RebuildSettings fields added later (none yet), against RebuildSettings().
REBUILD_OPTIONAL: tuple[str, ...] = ()

# Outcomes (SearchRecord.outcome): what one call did. Every one but R_SEARCHED plays d0 (R_HELD a
# held wait).
R_ROOT = "root"                         # no trials configured (s_root): d0, on the root at B
R_DISABLED = "disabled"                 # max_errors exceptions in this battle: d0
R_PRESS = "press"                       # d0 presses an ability button: never searched
R_HELD = "held"                         # a wait verdict holds (module doc, step 4)
R_NOT_TRIGGERED = "not_triggered"
R_REBUILD = "rebuild"                   # "rebuild:<rs_rebuild.Refusal reason>"
R_SPEC = "spec_refused"                 # the spec could not be built (rs_rebuild.RebuildError)
R_FIDELITY = "low_fidelity"
R_ROOT_OVER = "root_game_over"
R_FEW_CANDS = "too_few_candidates"
R_INCUMBENT_ILLEGAL = "incumbent_illegal"
R_INCUMBENT = "incumbent_missing"
R_DEADLINE = "deadline"
R_FEW_TRIALS = "too_few_trials"
R_EXCEPTION = "exception"
R_SEARCHED = "searched"                 # the trials were scored: an override or a veto
OUTCOMES = (R_ROOT, R_DISABLED, R_PRESS, R_HELD, R_NOT_TRIGGERED, R_REBUILD, R_SPEC, R_FIDELITY,
            R_ROOT_OVER, R_FEW_CANDS, R_INCUMBENT_ILLEGAL, R_INCUMBENT, R_DEADLINE, R_FEW_TRIALS,
            R_EXCEPTION, R_SEARCHED)
#: Triggers (module doc, step 5).
T_PLAY, T_OPP, T_ELIXIR, T_PUSH = "play", "opp_play", "elixir", "push"


# ---------------------------------------------------------------------------
# the configuration and the presets
# ---------------------------------------------------------------------------

_INT_FIELDS = ("max_trials", "horizon", "worlds", "min_candidates", "min_trials", "extra_tiles",
               "tile_gap", "other_slots", "other_tiles", "top_tiles", "elixir_trigger",
               "hold_ticks", "max_errors")
_FLOAT_FIELDS = ("trust", "z", "cvar_alpha", "prior_weight", "leak_weight", "prior_drop",
                 "terminal_scale")
_BOOL_FIELDS = ("lookahead", "wait", "verify")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclasses.dataclass(frozen=True)
class SearchConfig:
    """What one search does (module doc, THE SEARCH and PRESETS). Builtins only and frozen, like
    tools/rl/teacher.TeacherConfig; to_json / from_json (unknown fields refused), key.
    SearchConfig() is a custom idle search; the plan's arms are SearchConfig.named(...)."""

    preset: str = "custom"
    lookahead: bool = True            # d0 decides on the root at B (a landing-time checkpoint)
    max_trials: int = 8               # candidates tried; 0: the look-ahead alone
    horizon: int = 60                 # H: the ticks each trial runs from B
    worlds: int = 1                   # K opponent worlds (more than one: opponent "reply")
    opponent: str = OPP_IDLE
    level: str | None = None          # the rebuild level of the search (None: the rebuilder's)
    trust: float = 0.05               # tau: the override margin (the trust region)
    z: float = 1.0                    # K worlds: the margin adds z x sd / sqrt(K)
    cvar_alpha: float = 0.5           # K worlds: a score is the mean of the worst alpha share
    prior_weight: float = 0.05
    leak_weight: float = 0.05
    prior_drop: float = 4.0           # candidates below the incumbent's prior - this are dropped
    min_candidates: int = 2
    min_trials: int = 2
    wait: bool = True                 # W is a candidate when d0 plays (the cap rule aside)
    extra_tiles: int = 2              # T: P's card at this many more tiles ...
    tile_gap: int = 2                 # ... each this many tiles from P and from each other
    other_slots: int = 2              # S: this many other slots, by the gate ...
    other_tiles: int = 2              # ... at this many tiles each
    top_tiles: int = RD.TOP_TILES     # tiles scanned per slot, best first
    elixir_trigger: int = 8000        # milli: spendable elixir that triggers a search
    hold_ticks: int = RD.DECIDE       # a wait verdict holds this long
    max_errors: int = 3               # exceptions in one battle before search is off for it
    terminal_scale: float = 3.0       # a finished game scores its result x this ("swing" V)
    deadline_ms: float | None = None  # live only (Searcher(realtime=True)): the trials stop here
    verify: bool = False              # check every restore (Rebuilder.trial(verify=True))

    def __post_init__(self) -> None:
        bad: list[str] = []
        for name in _INT_FIELDS:
            v = getattr(self, name)
            if not _is_number(v) or int(v) != v:
                bad.append(f"{name} an integer (got {v!r})")
            else:
                object.__setattr__(self, name, int(v))
        for name in _FLOAT_FIELDS:
            v = getattr(self, name)
            if not _is_number(v):
                bad.append(f"{name} a finite number (got {v!r})")
            else:
                object.__setattr__(self, name, float(v))
        for name in _BOOL_FIELDS:
            if not isinstance(getattr(self, name), bool):
                bad.append(f"{name} true or false (got {getattr(self, name)!r})")
        if not isinstance(self.preset, str) or not self.preset.strip():
            bad.append(f"preset a name (got {self.preset!r})")
        if self.opponent not in OPPONENTS:
            bad.append(f"opponent one of {OPPONENTS} (got {self.opponent!r})")
        if self.level not in SEARCH_LEVELS:
            bad.append(f"level None, {RB.R1!r} or {RB.R2!r} (got {self.level!r}; {RB.R0} rebuilds "
                       f"from the seat's privileged state: the regret instrument's, S10)")
        if self.deadline_ms is not None:
            if not _is_number(self.deadline_ms) or not self.deadline_ms > 0:
                bad.append(f"deadline_ms None or > 0 (got {self.deadline_ms!r})")
            else:
                object.__setattr__(self, "deadline_ms", float(self.deadline_ms))
        if bad:
            raise SearchConfigError("SearchConfig: need " + "; ".join(bad))

        def need(ok: bool, what: str) -> None:
            if not ok:
                bad.append(what)

        need(self.max_trials >= 0, "max_trials >= 0")
        need(self.opponent != OPP_NONE or self.max_trials == 0,
             "max_trials 0 with opponent 'none' (trials need an opponent arm: idle, reply, true)")
        if self.max_trials > 0:
            need(self.horizon >= 1, "horizon >= 1")
            need(self.max_trials >= self.min_candidates, "max_trials 0 or >= min_candidates")
        need(self.worlds >= 1, "worlds >= 1")
        need(self.worlds == 1 or self.opponent == OPP_REPLY,
             "worlds 1 unless opponent 'reply' (idle worlds are all alike)")
        need(self.min_candidates >= 2, "min_candidates >= 2")
        need(self.min_trials >= 1, "min_trials >= 1")
        for name in ("extra_tiles", "tile_gap", "other_slots", "other_tiles", "elixir_trigger",
                     "hold_ticks"):
            need(getattr(self, name) >= 0, f"{name} >= 0")
        need(self.top_tiles >= 1, "top_tiles >= 1")
        need(self.max_errors >= 1, "max_errors >= 1")
        need(self.trust >= 0 and self.z >= 0, "trust and z >= 0")
        need(0 < self.cvar_alpha <= 1, "cvar_alpha in (0, 1]")
        need(self.prior_weight >= 0 and self.leak_weight >= 0, "prior_weight, leak_weight >= 0")
        need(self.prior_drop > 0 and self.terminal_scale > 0, "prior_drop, terminal_scale > 0")
        if bad:
            raise SearchConfigError("SearchConfig: need " + "; ".join(bad))

    # ------------------------------------------------------------ reading
    @property
    def trials(self) -> bool:
        """Whether this search tries candidates (s_root does not)."""
        return self.max_trials > 0

    @property
    def privileged(self) -> bool:
        """Whether its trials read the TRUE battle (s_true): never promotable, never live."""
        return self.opponent == OPP_TRUE

    @property
    def promotable(self) -> bool:
        return not self.privileged

    @property
    def key(self) -> str:
        """sha1 of what changes a decision (every field but `verify`, which only checks)."""
        data = {k: v for k, v in self.to_json().items() if k != "verify"}
        text = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------ building
    def replace(self, **changes: Any) -> "SearchConfig":
        unknown = sorted(set(changes) - {f.name for f in dataclasses.fields(self)})
        if unknown:
            raise SearchConfigError(f"SearchConfig: unknown field(s) {unknown}")
        return dataclasses.replace(self, **changes)

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "SearchConfig":
        unknown = sorted(set(data) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise SearchConfigError(f"SearchConfig: unknown field(s) {unknown}")
        return cls(**dict(data))

    @classmethod
    def named(cls, name: str, **overrides: Any) -> "SearchConfig":
        """The preset `name` (PRESETS) with `overrides`."""
        if name not in PRESETS:
            raise SearchConfigError(f"search preset: one of {sorted(PRESETS)}, got {name!r}")
        return cls.from_json({**PRESETS[name], **overrides})


#: The plan's arms (3.2): what each preset changes from SearchConfig's defaults.
PRESETS: dict[str, dict[str, Any]] = {
    "s_root": {"preset": "s_root", "max_trials": 0, "opponent": OPP_NONE},
    "s_idle": {"preset": "s_idle", "max_trials": 8, "horizon": 60, "opponent": OPP_IDLE,
               "deadline_ms": 80.0},
    "s_reply": {"preset": "s_reply", "max_trials": 8, "horizon": 80, "worlds": 4,
                "opponent": OPP_REPLY, "deadline_ms": 120.0},
    "s_true": {"preset": "s_true", "max_trials": 8, "horizon": 60, "opponent": OPP_TRUE,
               "deadline_ms": 80.0},
}


def _rebuild_tokens(text: str) -> tuple[str, ...]:
    """The tokens of a seat's "@probes+R2" tail (module doc, THE SEAT), checked."""
    tokens = tuple(t.strip() for t in str(text).split("+") if t.strip())
    bad = [t for t in tokens if t not in REBUILD_TOKENS]
    if bad or not tokens:
        raise SearchConfigError(f"search rebuild: tokens of {list(REBUILD_TOKENS)} joined by "
                                f"'+', got {text!r}")
    if REBUILD_DEFAULTS in tokens and REBUILD_PROBES in tokens:
        raise SearchConfigError(f"search rebuild: {REBUILD_DEFAULTS!r} or {REBUILD_PROBES!r}, "
                                f"not both ({text!r})")
    if RB.R1 in tokens and RB.R2 in tokens:
        raise SearchConfigError(f"search rebuild: one level, got {text!r}")
    return tokens


def resolve(value: Any) -> SearchConfig | None:
    """The SearchConfig `value` names: None or "off" / "none" / "": no search (None); a
    SearchConfig; a preset name; a JSON object text or a mapping (its "preset", when a PRESETS
    name, is the base the other fields override; unknown fields refused). A seat's search
    (seat_search: "name@tokens", a mapping with "rebuild") gives its SearchConfig; the rebuild
    part is seat_search's."""
    if value is None or isinstance(value, SearchConfig):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in OFF_NAMES:
            return None
        if text.startswith("{"):
            try:
                data = json.loads(text)
            except ValueError as exc:
                raise SearchConfigError(f"search config: not a JSON object ({exc})") from None
            if not isinstance(data, Mapping):
                raise SearchConfigError("search config: a JSON object")
            return resolve(data)
        if "@" in text:
            text, tail = (s.strip() for s in text.split("@", 1))
            _rebuild_tokens(tail)
        if text in PRESETS:
            return SearchConfig.named(text)
        raise SearchConfigError(f"search: a preset ({', '.join(sorted(PRESETS))}; a seat's may "
                                f"add @{'/'.join(REBUILD_TOKENS)}, '+' between them), 'off' or a "
                                f"JSON object, got {value!r}")
    if isinstance(value, Mapping):
        data = dict(value)
        data.pop(SEAT_REBUILD, None)
        name = data.get("preset")
        base = dict(PRESETS[name]) if isinstance(name, str) and name in PRESETS else {}
        return SearchConfig.from_json({**base, **data})
    raise SearchConfigError(f"search: a preset name, a JSON object, a mapping or a SearchConfig, "
                            f"got {type(value).__name__}")


def expand(value: Any) -> dict[str, Any] | None:
    """resolve(value).to_json(), the full dict a game key records; None for no search."""
    cfg = resolve(value)
    return None if cfg is None else cfg.to_json()


def is_privileged(value: Any) -> bool:
    """Whether the search `value` names reads the true battle (s_true)."""
    cfg = resolve(value)
    return bool(cfg is not None and cfg.privileged)


def check_promotable(value: Any, what: str = "a promoted candidate") -> SearchConfig | None:
    """resolve(value), or PrivilegedSearchError for a privileged search: s_true is a ceiling,
    never promoted (plan 3.2, 7.4). For a caller that must refuse one outright (check_live).
    The gate (tools/rs_league_eval.py) does not call it: it reads is_privileged instead
    (privileged_players, and Game.privileged from the games' own records) and gives such a
    candidate the decision "privileged", never promoted (rs_league_eval.gate)."""
    cfg = resolve(value)
    if cfg is not None and cfg.privileged:
        raise PrivilegedSearchError(f"search {cfg.preset!r}: its trials read the TRUE battle (a "
                                    f"privileged ceiling, plan 3.2, 7.4); it is never {what}")
    return cfg


def check_live(value: Any) -> SearchConfig | None:
    """check_promotable for the live runner: s_true is refused live (plan 8, --search-preset)."""
    return check_promotable(value, "played live")


# ---------------------------------------------------------------------------
# candidates, worlds, the choice
# ---------------------------------------------------------------------------

def _num(v: Any) -> float | None:
    """A JSON-ready float: None for None, NaN and infinities."""
    if v is None:
        return None
    v = float(v)
    return round(v, 6) if math.isfinite(v) else None


@dataclasses.dataclass
class Candidate:
    """One candidate (module doc, step 6): its role (P, W, T, S), kind (rs_decide PLAY / WAIT),
    PRESENT hand slot, card, own-frame tile and engine tap point, prior and cost (milli), and
    what its trials found: per world parts (swing, value, leak, terminal), world_scores
    (swing + value - leak term) and score (the aggregate + the prior term)."""

    role: str
    kind: str
    slot: int = -1
    card: int = -1
    tile: int = -1
    x: int | None = None
    y: int | None = None
    prior: float = 0.0
    cost: int = 0
    ran: bool = False
    parts: list = dataclasses.field(default_factory=list)
    world_scores: list = dataclasses.field(default_factory=list)
    score: float | None = None
    afters: list | None = dataclasses.field(default=None, repr=False)

    @property
    def is_play(self) -> bool:
        return self.kind == RD.PLAY

    def play(self, tick: int, team: int) -> tuple[int, int, int, int, int] | None:
        """The trial's (tick, team, card, x, y) of this candidate (None: a wait): sent by card, so
        the rebuild maps it to its slot at B (Rebuilder.trial)."""
        if not self.is_play:
            return None
        return (int(tick), int(team), int(self.card), int(self.x), int(self.y))

    def command(self, team: int) -> tuple[int, int, int, int] | None:
        """The (team, PRESENT slot, x, y) command of this candidate sent now (None: a wait): what
        the seat would send, and what a true-copy trial sends (S09)."""
        if not self.is_play:
            return None
        return (int(team), int(self.slot), int(self.x), int(self.y))

    @property
    def label(self) -> str:
        return "wait" if not self.is_play else str(int(self.card))

    def to_json(self) -> dict[str, Any]:
        return {"role": self.role, "kind": self.kind, "slot": self.slot, "card": self.card,
                "tile": self.tile, "x": self.x, "y": self.y, "prior": _num(self.prior),
                "cost": int(self.cost), "ran": bool(self.ran), "score": _num(self.score),
                "worlds": [{"swing": _num(p.get("swing")), "value": _num(p.get("value")),
                            "leak": _num(p.get("leak")), "terminal": bool(p.get("terminal"))}
                           for p in self.parts]}


class World(NamedTuple):
    """One opponent world: the root (an rs_rebuild.Built) its trials start from, the commands
    (tick, team, card, x, y) every candidate shares in it, a weight and tags. The idle search
    has one, the root itself with no commands; S07's model draws K."""

    index: int
    root: Any
    plays: tuple = ()
    weight: float = 1.0
    tags: tuple = ()


class TrueRoot(NamedTuple):
    """S09: what a true-copy trial is scored from (module doc, THE TRUE COPY): the TRUE state at t
    (the swing's before, as tools/rl/teacher.annotate has it), our later plays in flight and the
    PUBLIC root's state at B (the leak reads its elixir; leak_state)."""

    state: dict
    later: tuple = ()
    leak_state: dict | None = None


class Choice(NamedTuple):
    """choose()'s answer: the best candidate, whether it replaces the incumbent, the paired gain
    and the margin it had to reach (None when the incumbent did not run), each total."""

    best: int
    override: bool
    gain: float | None
    threshold: float | None
    totals: tuple


def cvar(values: Iterable[float], alpha: float = 0.5) -> float:
    """CVaR at alpha: the mean of the worst ceil(alpha x K) of K values (one value: itself)."""
    v = sorted(float(x) for x in values)
    if not v:
        raise ValueError("cvar of no values")
    if not 0 < float(alpha) <= 1:
        raise ValueError(f"cvar: alpha in (0, 1], got {alpha}")
    k = max(1, math.ceil(float(alpha) * len(v) - 1e-9))
    return sum(v[:k]) / k


def choose(scores: Sequence[Sequence[float] | None], priors: Sequence[float], incumbent: int = 0,
           *, trust: float = 0.05, z: float = 1.0, alpha: float = 0.5,
           prior_weight: float = 0.05) -> Choice:
    """The trust region (module doc, step 9). scores[c]: candidate c's score in each world
    (swing + V - the leak term; None: it did not run), the same K worlds for every candidate;
    priors[c] its prior. total = cvar(scores[c], alpha) + prior_weight x prior; the best total
    (the incumbent on a tie, then the first) replaces the incumbent only when the mean paired
    gain, scores[best] - scores[incumbent] + prior_weight x (prior difference) per world, is at
    least trust + z x sd / sqrt(K) (one world: total(best) >= total(incumbent) + trust)."""
    n = len(scores)
    if len(priors) != n:
        raise ValueError(f"choose: {n} candidates, {len(priors)} priors")
    if not 0 <= int(incumbent) < n:
        raise ValueError(f"choose: incumbent {incumbent} of {n} candidates")
    inc = int(incumbent)
    totals: list[float] = []
    width = None
    for row, prior in zip(scores, priors):
        vals = None if row is None else [float(x) for x in row]
        if not vals or not all(math.isfinite(x) for x in vals) or not math.isfinite(float(prior)):
            totals.append(-math.inf)
            continue
        if width is None:
            width = len(vals)
        elif len(vals) != width:
            raise ValueError("choose: every candidate needs one score per world")
        totals.append(cvar(vals, alpha) + float(prior_weight) * float(prior))
    if not math.isfinite(totals[inc]):
        return Choice(inc, False, None, None, tuple(totals))
    best = max(range(n), key=lambda i: (totals[i], i == inc, -i))
    if best == inc:
        return Choice(inc, False, 0.0, float(trust), tuple(totals))
    gains = (np.asarray(scores[best], dtype=np.float64) - np.asarray(scores[inc], dtype=np.float64)
             + float(prior_weight) * (float(priors[best]) - float(priors[inc])))
    k = int(gains.size)
    mean = float(gains.mean())
    sd = float(gains.std(ddof=1)) if k > 1 else 0.0
    threshold = float(trust) + float(z) * sd / math.sqrt(k)
    return Choice(best, bool(mean >= threshold), mean, threshold, tuple(totals))


# ---------------------------------------------------------------------------
# the decision the search returns
# ---------------------------------------------------------------------------

#: A chosen action other than d0's own has no draw of its own (sampling fields cleared).
_UNDRAWN = {"gate_a": -1, "logq_gate": 0.0, "tile_cands": (), "tile_a": -1, "logq_tile": 0.0}


def _searched_mode(mode: str) -> str:
    mode = str(mode or "")
    return mode if mode.endswith(MODE_SUFFIX) else mode + MODE_SUFFIX


def tagged(d0: Any) -> Any:
    """d0 as the search kept it: the same action, mode + "+search"."""
    return dataclasses.replace(d0, mode=_searched_mode(d0.mode))


def as_wait(d0: Any, reason: str = REASON_WAIT) -> Any:
    """A WAIT built from d0 (dataclasses.replace: rs_decide.Decision is frozen): no slot, card,
    tile or tap; d0's gate, mask and elixir kept; mode + "+search"."""
    return dataclasses.replace(d0, kind=RD.WAIT, reason=str(reason), slot=-1, card=-1, cost=0,
                               tile=-1, tx=-1, ty=-1, x_sub=None, y_sub=None, fallback=False,
                               tried=(), lane=None, button=-1, mode=_searched_mode(d0.mode),
                               **_UNDRAWN)


def as_play(d0: Any, cand: Candidate, cost: int, lane: int | None = None) -> Any:
    """The PLAY of `cand` built from d0: its PRESENT slot (the command is sent now), card, tile
    (own frame), tap point and cost (elixir, as decide gives it); mode + "+search"."""
    tile = int(cand.tile)
    return dataclasses.replace(d0, kind=RD.PLAY, reason="", slot=int(cand.slot),
                               card=int(cand.card), cost=max(0, int(cost)), tile=tile,
                               tx=tile % RD.GRID_W, ty=tile // RD.GRID_W, x_sub=int(cand.x),
                               y_sub=int(cand.y), fallback=False, tried=(int(cand.slot),),
                               lane=lane, button=-1, mode=_searched_mode(d0.mode), **_UNDRAWN)


def _brief(d: Any) -> dict[str, Any]:
    return {"kind": d.kind, "reason": d.reason, "slot": int(d.slot), "card": int(d.card),
            "tile": int(d.tile), "mode": d.mode}


@dataclasses.dataclass
class SearchRecord:
    """What one Searcher.decide call did (module doc, THE SEARCH): JSON-ready via to_json. The
    incumbent is candidates[0]; chosen the best candidate (-1: no trials scored)."""

    tick: int
    land_tick: int
    delay: int
    preset: str
    config: str = ""
    privileged: bool = False           # S09: the trials read the TRUE battle (s_true)
    outcome: str = ""
    detail: str = ""
    lookahead: bool = False
    triggers: tuple = ()
    fidelity: float | None = None
    flags: tuple = ()
    key: str = ""
    candidates: list = dataclasses.field(default_factory=list)
    dropped: int = 0
    worlds: int = 0
    trials: int = 0
    incumbent: int = 0
    chosen: int = -1
    override: bool = False
    gain: float | None = None
    threshold: float | None = None
    d0: dict = dataclasses.field(default_factory=dict)
    decision: dict = dataclasses.field(default_factory=dict)
    errors: list = dataclasses.field(default_factory=list)
    ms: dict = dataclasses.field(default_factory=dict)

    @property
    def searched(self) -> bool:
        return self.outcome == R_SEARCHED

    def to_json(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        out["triggers"] = list(self.triggers)
        out["flags"] = list(self.flags)
        out["fidelity"] = _num(self.fidelity)
        out["gain"] = _num(self.gain)
        out["threshold"] = _num(self.threshold)
        out["candidates"] = [c.to_json() for c in self.candidates]
        out["errors"] = list(self.errors)
        out["ms"] = {k: _num(v) for k, v in self.ms.items()}
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _RowPolicy:
    """decide()'s policy with one row's batch fixed, so decide's gate, the priors and the tiles of
    every candidate read one encoder pass (BatchedPolicy caches it by the batch's identity)."""

    def __init__(self, policy: Any, row: Mapping[str, Any], b: Any):
        self._policy, self._row, self._b = policy, row, b

    def batch(self, row: Mapping[str, Any]) -> Any:
        return self._b if row is self._row else self._policy.batch(row)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._policy, name)


def _gate_of(policy: Any, b: Any) -> np.ndarray:
    """The raw gate logits decide read: heads(b)[0] when the policy has heads, else gate(b)."""
    heads = getattr(policy, "heads", None)
    if callable(heads):
        return np.asarray(heads(b)[0])
    return np.asarray(policy.gate(b))


def _logp(p: float) -> float:
    return math.log(max(float(p), 1e-300))


def _lse(z: np.ndarray) -> float:
    m = float(np.max(z))
    return m + math.log(float(np.exp(z - m).sum()))


def _cheb(a: int, b: int) -> int:
    """Tiles apart (Chebyshev, own frame) of two model tiles."""
    return max(abs(a % RD.GRID_W - b % RD.GRID_W), abs(a // RD.GRID_W - b // RD.GRID_W))


def _ran(run: Mapping[str, Any], ref: int) -> bool:
    """Whether Rebuilder.trial's run report has the play `ref` (its index in the plays) as run."""
    return any(r[0] == "play" and int(r[1]) == int(ref) for r in run.get("ran") or ())


def _step_code(res: Sequence[Any], i: int) -> int:
    """The reason code of Battle.step's i-th result row ((card, reason, ...) on RoyaleSim,
    [team, reason] on the test fakes); 0 (accepted) when it says nothing."""
    try:
        return int(res[i][1])
    except (IndexError, TypeError, ValueError):
        return 0


def _refused_when_run(battle: Any, team: int, card: int, due: int) -> bool:
    """S09: whether the last step's Battle.step_commands_run() rows (tick, team, kind, what,
    reason) report `team`'s deploy of `card` due on `due` (or later: rs_match._ran_play's rule) as
    refused or dropped; an engine or a fake without the call: never."""
    fn = getattr(battle, "step_commands_run", None)
    if not callable(fn):
        return False
    for row in fn() or ():
        row = tuple(row)
        if len(row) < 5 or row[2] != "deploy":
            continue
        try:
            tick, who, what, reason = int(row[0]), int(row[1]), int(row[3]), int(row[4])
        except (TypeError, ValueError):
            continue
        if who == int(team) and what == int(card) and tick >= int(due) and reason != 0:
            return True
    return False


def _rl(name: str) -> Any:
    """tools/rl/<name> (teacher, ppo: numpy only), imported on first use."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(f"tools.rl.{name}")


@dataclasses.dataclass
class _Memo:
    """One battle's search state: its last tick (on the decision clock: decide's clock_tick, else
    the view's tick), exceptions, the hold, the opponent plays seen."""

    last_tick: int | None = None
    clock: int = 0
    errors: int = 0
    disabled: bool = False
    hold_until: int | None = None
    hold_opp: int = 0
    opp_seen: int = 0


# ---------------------------------------------------------------------------
# the searcher
# ---------------------------------------------------------------------------

class Searcher:
    """The search of one seat (module doc, THE SEARCH).

    policy      the seat's policy: decide's (batch, gate / heads, tiles), plus values(rows, kind)
                and value_kind for the trials (a TokenPolicyAdapter or TokenPolicy becomes a
                BatchedPolicy: as_batched)
    rebuilder   an rs_rebuild.Rebuilder (its settings pinned by the caller, S08)
    config      a SearchConfig, a preset name, a JSON object or a mapping (resolve)
    dcfg        the seat's rs_decide.DecideConfig (None: legacy)
    landing     the checkpoint's rows are landing-time (rs_decide.obs_delay_ticks(meta) == 0):
                d0 decides on the root at B; a decision-time checkpoint decides on the present
                row and the root serves the trials only
    sub         subtiles per millitile
    law         the rs_elixir.ElixirLaw the leak is counted under (None: rs_elixir.default_law())
    worlds      worlds(searcher, root, view, team, land_tick, rng) -> [World] (S07); None: one
                idle world. Opponent "reply" needs one.
    realtime    live: the trials stop at config.deadline_ms. Off in the simulator, whose games
                must not depend on the machine (the search's time is charged as delay instead).
    clock       time.perf_counter, or a test's
    privileged  True only for the privileged ceiling (s_true); any other caller gets
                PrivilegedSearchError for it. Its decide() then needs true_battle= (module doc,
                THE TRUE COPY), which every public preset refuses.
    Every view is asserted public (rs_public_view.is_public; SearchLeakError), always: there is
    no switch to turn that off.
    """

    def __init__(self, policy: Any, rebuilder: Any, config: Any, *, dcfg: Any = None,
                 landing: bool = True, sub: int = RD.SUBTILE_PER_MILLITILE, law: Any = None,
                 worlds: Callable[..., Iterable[Any]] | None = None, realtime: bool = False,
                 clock: Callable[[], float] = time.perf_counter, privileged: bool = False):
        cfg = resolve(config)
        if cfg is None:
            raise SearchConfigError("Searcher: a search config (a preset, a SearchConfig); 'off' "
                                    "is no Searcher")
        if cfg.privileged:
            if not privileged:
                raise PrivilegedSearchError(
                    f"search {cfg.preset!r}: its trials read the TRUE battle; only the privileged "
                    "ceiling (M2b) and the regret instrument build it, with privileged=True")
            if worlds is not None:
                raise SearchConfigError(f"search {cfg.preset!r}: its one world is the true "
                                        "battle; it takes no worlds provider")
        if cfg.opponent == OPP_REPLY and worlds is None:
            raise SearchConfigError(f"search {cfg.preset!r}: opponent 'reply' needs the opponent "
                                    "world model (S07, tools/opp_model.py): pass worlds=")
        self.policy = as_batched(policy)
        kind = getattr(self.policy, "value_kind", VALUE_SWING)
        if cfg.trials:
            if not callable(getattr(self.policy, "values", None)):
                raise SearchConfigError("the trials score V with policy.values(rows, kind) "
                                        "(BatchedPolicy); this policy has none")
            if kind not in VALUE_KINDS:
                raise SearchConfigError(f"policy.value_kind one of {VALUE_KINDS}, got {kind!r}")
        self.value_kind = kind
        self.rebuilder = rebuilder
        self.config = cfg
        self.key = cfg.key
        self.dcfg = RD.DecideConfig() if dcfg is None else dcfg
        self.landing = bool(landing)
        self.sub = int(sub)
        self._law = law
        self._law_now: Any = None
        self.worlds = worlds
        self.realtime = bool(realtime)
        self.clock = clock
        self.memo = _Memo()
        self.stats: Counter = Counter()
        self.ms: Counter = Counter()
        self.last: SearchRecord | None = None

    # ------------------------------------------------------------ battles
    def new_battle(self) -> None:
        """Forget the battle: exceptions, the hold, the opponent plays seen, the spell sightings."""
        self.memo = _Memo()
        sightings = getattr(self.rebuilder, "sightings", None)
        if callable(getattr(sightings, "reset", None)):
            sightings.reset()
        self.stats["battles"] += 1

    def _battle_at(self, now: int) -> None:
        if self.memo.last_tick is not None and now < self.memo.last_tick:
            self.new_battle()
        self.memo.last_tick = self.memo.clock = now

    def _root_failed(self, why: str) -> None:
        """Count a decision whose look-ahead root was not usable, by reason (summary's
        root_fallbacks): s_root reports its fallbacks here, its outcome is always "root"."""
        self.stats[f"root_fallback:{why}"] += 1

    # ------------------------------------------------------------ the decision
    def decide(self, view: Mapping[str, Any], *, team: int, land_ticks: int, legality: Any,
               rng: Any, row_fn: Callable[[Mapping[str, Any], int], Any], tick: int | None = None,
               pending: Iterable[Any] | None = None, dt: int | None = None,
               aim: Callable | None = None, blocked_tiles: Any = None,
               opp_fill: Iterable[int] = (), seed: int = 0,
               true_battle: Any = None,
               clock_tick: int | None = None) -> tuple[Any, SearchRecord]:
        """(the Decision to act on, its SearchRecord) for `team` on the public `view` (module
        doc, THE SEARCH). true_battle: the TRUE match battle, for the privileged s_true's trials
        only (module doc, THE TRUE COPY); a public preset refuses it. clock_tick: the decision
        clock the hold and the new-battle check run on (None: the view's tick, live's HUD
        clock); the simulator seat passes its own elapsed ticks, because its view's tick carries
        perception clock noise and can step back within a battle (module doc, THE SEAT)."""
        cfg = self.config
        if true_battle is not None and not cfg.privileged:
            raise PrivilegedSearchError(f"search {cfg.preset!r} is public: it never takes the "
                                        "true battle (only the privileged s_true does)")
        if cfg.privileged and true_battle is None:
            raise SearchConfigError(f"search {cfg.preset!r}: its trials run on the TRUE battle; "
                                    "pass true_battle= (the seat's match battle)")
        t0 = self.clock()
        team = int(team)
        now = int(view["tick"])
        delay = max(0, int(land_ticks))
        land = now + delay
        clock = now if clock_tick is None else int(clock_tick)
        self._battle_at(clock)
        memo = self.memo
        self._check_view(view, team)
        self._law_now = self._law_for(view)
        rec = SearchRecord(tick=now, land_tick=land, delay=delay, preset=cfg.preset,
                           config=self.key, privileged=cfg.privileged)
        self.stats["decisions"] += 1
        opp_count = len((view.get("public") or {}).get("opp_plays") or ())
        built, why, tried = None, "", False
        if cfg.lookahead and self.landing and not memo.disabled:
            tried = True
            built, why = self._root(view, team, land, seed, opp_fill, rec)
            if built is None:
                self._root_failed(why)
        t = self.clock()
        row = None
        if built is not None:
            try:
                row = row_fn(RB.with_present(built.state, view), land)
                rec.lookahead = True
            except Exception as exc:            # the plain row below; counted
                self._error(rec, f"root row: {type(exc).__name__}: {exc}")
                built, why = None, R_EXCEPTION
                self._root_failed(why)
        if row is None:
            row = row_fn(view, now)
        b = self.policy.batch(row)
        pol = _RowPolicy(self.policy, row, b)
        rec.ms["row"] = self._ms(t)
        t = self.clock()
        d0 = RD.decide(pol, row, view, team, legality, self.dcfg, rng=rng, pending=pending,
                       tick=now if tick is None else tick, dt=dt, sub=self.sub, aim=aim,
                       blocked_tiles=blocked_tiles)
        rec.ms["d0"] = self._ms(t)
        rec.d0 = _brief(d0)
        if memo.disabled:
            return self._done(rec, d0, R_DISABLED, t0)
        if not cfg.trials:
            return self._done(rec, d0, R_ROOT, t0, why)
        if d0.is_press:
            return self._done(rec, d0, R_PRESS, t0)
        if memo.hold_until is not None:
            if clock < memo.hold_until and opp_count <= memo.hold_opp:
                if not d0.is_play:
                    return self._done(rec, d0, R_HELD, t0)
                if not d0.capped:
                    return self._done(rec, as_wait(d0, REASON_HOLD), R_HELD, t0)
            else:
                memo.hold_until = None
        rec.triggers = self._triggers(d0, view, team, opp_count)
        if not rec.triggers:
            return self._done(rec, d0, R_NOT_TRIGGERED, t0)
        memo.opp_seen = opp_count
        if built is None:
            if tried:
                return self._done(rec, d0, why, t0)
            built, why = self._root(view, team, land, seed, opp_fill, rec)
            if built is None:
                self._root_failed(why)
                return self._done(rec, d0, why, t0)
        try:
            return self._search(rec, d0, pol, b, built, view, team, land, legality, row_fn, dt,
                                aim, blocked_tiles, seed, now, t0, true_battle)
        except (SearchLeakError, SearchConfigError, TrueCopyError):
            raise
        except Exception as exc:                # d0; three in a battle turn search off
            self._error(rec, f"{type(exc).__name__}: {exc}")
            return self._done(rec, d0, R_EXCEPTION, t0)

    # ------------------------------------------------------------ the steps
    def _check_view(self, view: Mapping[str, Any], team: int) -> None:
        pub = view.get("public")
        if not isinstance(pub, Mapping) or int(pub.get("team", -1)) != team:
            raise SearchLeakError("the search reads rs_public_view.public_view views of the "
                                  f"acting team {team} only (no 'public' block for it)")
        problems = PV.is_public(view, team, fields=getattr(self.rebuilder, "F", None),
                                spell_fields=getattr(self.rebuilder, "SF", None))
        if problems:
            more = f" (+{len(problems) - 6} more)" if len(problems) > 6 else ""
            raise SearchLeakError("the view is not public: " + "; ".join(problems[:6]) + more)

    def _law_for(self, view: Mapping[str, Any]) -> Any:
        law = self._law if self._law is not None else rs_elixir.default_law()
        regular = view.get("regular_ticks")
        if regular and int(regular) != law.regular_ticks:
            law = law.with_regular_ticks(int(regular))
        return law

    def _seed(self, seed: int, now: int) -> int:
        return (abs(int(seed)) * 1_000_003 + int(now)) % SEED_MOD

    def _root(self, view: Mapping[str, Any], team: int, land: int, seed: int,
              opp_fill: Iterable[int], rec: SearchRecord) -> tuple[Any, str]:
        """(the usable root at B, "") or (None, the outcome why not)."""
        rb = self.rebuilder
        t = self.clock()
        try:
            try:
                spec = rb.spec(view, land_tick=land, seed=self._seed(seed, int(view["tick"])),
                               level=self.config.level, opp_fill=list(opp_fill or ()))
            except RB.RebuildError as exc:
                rec.detail = str(exc)
                return None, R_SPEC
            built = rb.root(spec)
        except Exception as exc:                # counted: three in a battle turn search off
            self._error(rec, f"rebuild: {type(exc).__name__}: {exc}")
            return None, R_EXCEPTION
        finally:
            rec.ms["rebuild"] = self._ms(t)
        if built is None:
            err = getattr(rb, "last_error", None)
            reason = str(getattr(err, "reason", "") or "refused")
            rec.detail = str(getattr(err, "detail", "") or "")
            if reason == RB.REFUSED_ERROR:
                self._error(rec, f"rebuild: {rec.detail}")
            return None, f"{R_REBUILD}:{reason}"
        rec.fidelity = float(built.fidelity)
        rec.flags = tuple(built.flags)
        rec.key = str(built.key)
        if not RB.usable(built, rb.settings.min_fidelity):
            return None, R_FIDELITY
        if built.state.get("game_over") or built.tick < land:
            return None, R_ROOT_OVER
        return built, ""

    def _triggers(self, d0: Any, view: Mapping[str, Any], team: int, opp_count: int) -> tuple:
        fired = []
        if d0.is_play:
            fired.append(T_PLAY)
        if opp_count > self.memo.opp_seen:
            fired.append(T_OPP)
        if int(d0.spendable_milli) >= self.config.elixir_trigger:
            fired.append(T_ELIXIR)
        if RD.threat_lane(view, team, self.sub) is not None:
            fired.append(T_PUSH)
        return tuple(fired)

    def _search(self, rec: SearchRecord, d0: Any, pol: Any, b: Any, built: Any,
                view: Mapping[str, Any], team: int, land: int, legality: Any, row_fn: Callable,
                dt: int | None, aim: Callable | None, blocked: Any, seed: int, now: int,
                t0: float, true_battle: Any = None) -> tuple[Any, SearchRecord]:
        cfg = self.config
        t = self.clock()
        cands, why, dropped = self._candidates(d0, pol, b, view, team, legality, built, dt, aim,
                                               blocked)
        rec.candidates, rec.dropped = cands, dropped
        rec.ms["candidates"] = self._ms(t)
        if why:
            return self._done(rec, d0, why, t0)
        if len(cands) < cfg.min_candidates:
            return self._done(rec, d0, R_FEW_CANDS, t0)
        if cfg.opponent == OPP_TRUE:            # S09: the privileged ceiling
            rec.worlds = 1
            t = self.clock()
            timed_out = self._true_trials(cands, built, true_battle, land, team, now)
        else:
            worlds = self._worlds(built, view, team, land, seed, now)
            rec.worlds = len(worlds)
            t = self.clock()
            timed_out = self._trials(cands, worlds, land, team, t0)
        rec.ms["trials"] = self._ms(t)
        rec.trials = sum(1 for c in cands if c.ran)
        if not cands[0].ran:
            return self._done(rec, d0, R_DEADLINE if timed_out else R_INCUMBENT, t0)
        if rec.trials < cfg.min_trials:
            return self._done(rec, d0, R_FEW_TRIALS, t0)
        t = self.clock()
        self._score(cands, land, team, row_fn)
        rec.ms["values"] = self._ms(t)
        choice = choose([c.world_scores if c.ran else None for c in cands],
                        [c.prior for c in cands], 0, trust=cfg.trust, z=cfg.z,
                        alpha=cfg.cvar_alpha, prior_weight=cfg.prior_weight)
        for c, total in zip(cands, choice.totals):
            c.score = total if math.isfinite(total) else None
        rec.chosen, rec.override = choice.best, choice.override
        rec.gain, rec.threshold = choice.gain, choice.threshold
        self.stats["searches"] += 1
        if not choice.override:
            self.stats["vetoes"] += 1
            return self._done(rec, tagged(d0), R_SEARCHED, t0)
        pick, inc = cands[choice.best], cands[0]
        if pick.is_play:
            decision = as_play(d0, pick, self._cost_units(view, team, pick.slot),
                               self._lane(view, team, pick.card))
        else:
            decision = as_wait(d0, REASON_WAIT)
            if d0.is_play:                      # the verdict holds (module doc, step 4)
                self.memo.hold_until = self.memo.clock + cfg.hold_ticks
                self.memo.hold_opp = self.memo.opp_seen
        self.stats["overrides"] += 1
        self.stats[f"override:{inc.label}>{pick.label}"] += 1
        return self._done(rec, decision, R_SEARCHED, t0)

    def _candidates(self, d0: Any, pol: Any, b: Any, view: Mapping[str, Any], team: int,
                    legality: Any, built: Any, dt: int | None, aim: Callable | None,
                    blocked: Any) -> tuple[list[Candidate], str, int]:
        """(the candidates, incumbent first, at most max_trials; "" or the outcome that ends
        the search; how many the prior drop removed) (module doc, step 6)."""
        cfg, dcfg, rb = self.config, self.dcfg, self.rebuilder
        p = view["players"][team]
        hand = list(p.get("hand") or [])
        g = np.asarray(_gate_of(pol, b), dtype=np.float64).reshape(-1).copy()
        g[0] += dcfg.wait_bias
        mask = np.asarray(d0.mask, dtype=bool)
        if mask.size != g.size:
            raise SearchError(f"d0's mask has {mask.size} entries, the gate {g.size}")
        cadence = dcfg.dt if dt is None else max(1, int(dt))
        q = RD.gate_probs(g, mask, cadence, bool(d0.capped))
        check = RD._bind(legality, view, team)
        if blocked is not None:
            blocked = np.asarray(blocked, dtype=bool).reshape(-1)
        cache: dict[int, np.ndarray] = {}

        def logits(slot: int) -> np.ndarray:
            if slot not in cache:
                cache[slot] = np.asarray(pol.tiles(b, slot), dtype=np.float64).reshape(-1)
            return cache[slot]

        def prior(slot: int, tile: int) -> float:
            lg = logits(slot)
            return _logp(q[1 + slot]) + float(lg[tile]) - _lse(lg)

        def order(slot: int, card: int) -> list[int]:
            o = np.argsort(-logits(slot), kind="stable")
            if aim is not None:
                o = aim(o, slot, card)
            return [int(t) for t in list(o)[:cfg.top_tiles]]

        def lane_of(card: int) -> int | None:
            return self._lane(view, team, card)

        def tap(slot: int, card: int, tile: int, lane: int | None) -> tuple[int, int] | None:
            """The tap point of a tile d0's checks and check_deploy at B accept, else None."""
            if blocked is not None and blocked[tile]:
                return None
            if RD._guarded(lane, tile % RD.GRID_W):
                return None
            wx, wy = RD.tile_centre_world(tile, team)
            x, y = int(wx * self.sub), int(wy * self.sub)
            if not check(slot, card, x, y):
                return None
            if int(rb.legal(card, x, y, built)) != 0:
                return None
            return x, y

        out: list[Candidate] = []
        if d0.is_play:
            inc = Candidate(ROLE_P, RD.PLAY, slot=int(d0.slot), card=int(d0.card),
                            tile=int(d0.tile), x=int(d0.x_sub), y=int(d0.y_sub),
                            prior=prior(int(d0.slot), int(d0.tile)),
                            cost=self._cost_milli(p, int(d0.slot), int(d0.card)))
            if int(rb.legal(inc.card, inc.x, inc.y, built)) != 0:
                return [inc], R_INCUMBENT_ILLEGAL, 0
            out.append(inc)
            if cfg.wait and not d0.capped and mask[0] and q[0] > 0:
                out.append(Candidate(ROLE_W, RD.WAIT, prior=_logp(q[0])))
            lane = lane_of(inc.card)
            taken = [inc.tile]
            for t in order(inc.slot, inc.card):
                if len(taken) >= 1 + cfg.extra_tiles:
                    break
                if t == inc.tile or any(_cheb(t, u) < cfg.tile_gap for u in taken):
                    continue
                xy = tap(inc.slot, inc.card, t, lane)
                if xy is None:
                    continue
                out.append(Candidate(ROLE_T, RD.PLAY, slot=inc.slot, card=inc.card, tile=t,
                                     x=xy[0], y=xy[1], prior=prior(inc.slot, t), cost=inc.cost))
                taken.append(t)
        else:
            out.append(Candidate(ROLE_W, RD.WAIT, prior=_logp(q[0])))
        skip = int(d0.slot) if d0.is_play else -1
        slots = [s for s in range(RD.HAND_SIZE)
                 if s != skip and 1 + s < q.size and mask[1 + s] and q[1 + s] > 0
                 and s < len(hand) and hand[s] is not None and int(hand[s]) >= 0]
        slots.sort(key=lambda s: (-q[1 + s], s))
        for s in slots[:cfg.other_slots]:
            card = int(hand[s])
            lane = lane_of(card)
            found = 0
            for t in order(s, card):
                if found >= cfg.other_tiles:
                    break
                xy = tap(s, card, t, lane)
                if xy is None:
                    continue
                out.append(Candidate(ROLE_S, RD.PLAY, slot=s, card=card, tile=t, x=xy[0],
                                     y=xy[1], prior=prior(s, t),
                                     cost=self._cost_milli(p, s, card)))
                found += 1
        floor = out[0].prior - cfg.prior_drop
        keep = [out[0]] + [c for c in out[1:] if c.prior >= floor]
        return keep[:cfg.max_trials], "", len(out) - len(keep)

    def _worlds(self, built: Any, view: Mapping[str, Any], team: int, land: int, seed: int,
                now: int) -> list[World]:
        if self.worlds is None:
            return [World(0, built)]
        rng = np.random.default_rng([abs(int(seed)) % 2 ** 32, int(now) % 2 ** 32])
        got = [w if isinstance(w, World) else World(*w)
               for w in self.worlds(self, built, view, team, land, rng) or ()]
        if not got:
            raise SearchError("the world model gave no world")
        return got

    def _trials(self, cands: Sequence[Candidate], worlds: Sequence[World], land: int, team: int,
                t0: float) -> bool:
        """Run each candidate in every world, in order (module doc, step 7); True when the
        deadline stopped them."""
        cfg, rb = self.config, self.rebuilder
        limit = cfg.deadline_ms / 1000.0 if (self.realtime and cfg.deadline_ms) else None
        for c in cands:
            if limit is not None and self.clock() - t0 > limit:
                return True
            afters: list | None = []
            for w in worlds:
                plays = [tuple(int(v) for v in p) for p in w.plays]
                mine = None
                if c.is_play:
                    mine = len(plays)
                    plays.append(c.play(land, team))
                after, run = rb.trial(w.root, plays, cfg.horizon, verify=cfg.verify)
                if mine is not None and not _ran(run, mine):
                    afters = None                   # refused or skipped: it did not run
                    break
                afters.append((w, after))
            if afters is not None:
                c.ran, c.afters = True, afters
        return False

    def _true_trials(self, cands: Sequence[Candidate], built: Any, battle: Any, land: int,
                     team: int, now: int) -> bool:
        """S09 (module doc, THE TRUE COPY): from one save() of the TRUE battle, each candidate's
        command sent now (from its present slot) and stepped D + H ticks, then the battle loaded
        back and checked (TrueCopyError). False: no deadline (the ceiling is never live)."""
        cfg = self.config
        delay = int(land) - int(now)
        reported = _rl("teacher").battle_delay(battle, team)
        if reported is not None and int(reported) != delay:
            raise TrueCopyError(f"the true battle's command delay of team {team} is {reported}, "
                                f"the seat's {delay}: its trials would not land at B")
        steps = delay + int(cfg.horizon)
        before_json = battle.state_json()
        before = json.loads(before_json)
        due = int(before.get("tick", now)) + delay
        world = World(0, TrueRoot(state=before, later=tuple(built.later), leak_state=built.state),
                      tags=("true",))
        saved = battle.save()
        failure: Exception | None = None
        try:
            for c in cands:
                battle.load(saved)
                cmd = c.command(team)
                res = list(battle.step([] if cmd is None else [cmd], steps) or ())
                if cmd is not None and (_step_code(res, 0) != 0
                                        or _refused_when_run(battle, team, c.card, due)):
                    continue                    # refused at send or when it ran: did not run
                c.ran, c.afters = True, [(world, json.loads(battle.state_json()))]
        except Exception as exc:                # raised below, once the battle is checked
            failure = exc
        try:
            battle.load(saved)
            restored = battle.state_json() == before_json
        except Exception as exc:
            raise TrueCopyError(f"the true battle could not be loaded back after the true-copy "
                                f"trials ({type(exc).__name__}: {exc})") from exc
        if not restored:
            raise TrueCopyError("the true battle is not what it was before the true-copy trials: "
                                "its save() / load() did not restore it (probe P9); the match "
                                "cannot go on")
        if failure is not None:
            raise failure
        self.stats["true_trials"] += sum(1 for c in cands if c.ran)
        return False

    def _score(self, cands: Sequence[Candidate], land: int, team: int, row_fn: Callable) -> None:
        """Per candidate and world: swing, V (one batched pass), leak (module doc, step 8)."""
        cfg = self.config
        teacher, ppo = _rl("teacher"), _rl("ppo")
        kind = self.value_kind
        scale = cfg.terminal_scale if kind == VALUE_SWING else 1.0
        rows: list[Any] = []
        waiting: list[dict] = []
        for c in cands:
            if not c.ran:
                continue
            c.parts = []
            for w, after in c.afters or ():
                part = {"world": int(w.index), "terminal": bool(after.get("game_over")),
                        "swing": float(teacher.tower_swing(w.root.state, after, team)),
                        "leak": self._leak(w.root, c, land, team), "value": 0.0}
                if part["terminal"]:
                    part["value"] = float(ppo.result_of(after, team)) * scale
                else:
                    if kind == VALUE_CRITIC:
                        part["value"] = float(ppo.potential(*ppo.reward_parts(after, team)))
                    rows.append(row_fn(after, int(after.get("tick", land + cfg.horizon))))
                    waiting.append(part)
                c.parts.append(part)
            c.afters = None
        if rows:
            v = np.asarray(self.policy.values(rows, kind), dtype=np.float64).reshape(-1)
            if v.size != len(rows):
                raise SearchError(f"policy.values gave {v.size} values for {len(rows)} rows")
            for part, x in zip(waiting, v):
                part["value"] += float(x)
        for c in cands:
            if c.ran:
                c.world_scores = [p["swing"] + p["value"] - cfg.leak_weight * p["leak"]
                                  for p in c.parts]

    # ------------------------------------------------------------ small pieces
    def _card_cost(self, card: int) -> int:
        """`card`'s elixir (milli) from the rebuilder's tables, 0 when unknown."""
        try:
            return int((self.rebuilder.tables.cost or {}).get(int(card), 0) or 0)
        except Exception:                       # no catalogue here: a cost is only the leak's
            return 0

    def _cost_milli(self, p: Mapping[str, Any], slot: int, card: int) -> int:
        costs = list(p.get("hand_costs") or [])
        c = costs[slot] if 0 <= slot < len(costs) else None
        if c is not None and int(c) >= 0:
            return 1000 * int(c)
        return self._card_cost(card)

    def _cost_units(self, view: Mapping[str, Any], team: int, slot: int) -> int:
        """The Decision's cost of a play from `slot`, as rs_decide.decide gives it."""
        costs = list(view["players"][team].get("hand_costs") or [])
        cost = costs[slot] if slot < len(costs) and costs[slot] is not None else 0
        return max(0, int(cost))

    def _lane(self, view: Mapping[str, Any], team: int, card: int) -> int | None:
        """The lane guard's threatened lane for `card`, as decide reads it (None: no guard)."""
        if int(card) not in self.dcfg.guard_cards:
            return None
        return RD.threat_lane(view, team, self.sub)

    def _leak(self, root: Any, c: Candidate, land: int, team: int) -> float:
        """Elixir regenerated over [B, B + H) that does not fit under the cap (module doc, step
        8): our elixir at B on the root, less the candidate's cost and our later plays in flight
        that land in the window."""
        law = self._law_now if self._law_now is not None else rs_elixir.default_law()
        end = land + self.config.horizon
        state = getattr(root, "leak_state", None) or root.state     # S09: the public root's
        have = int(state["players"][team].get("elixir_milli") or 0)
        later = sum(self._card_cost(r[1]) for r in (getattr(root, "later", ()) or ())
                    if int(r[0]) < end)
        regen = law.to_milli(law.regen(land, end))
        cap = law.to_milli(law.cap_fine)
        return max(0, have - int(c.cost) - later + regen - cap) / 1000.0

    def _error(self, rec: SearchRecord, what: str) -> None:
        self.memo.errors += 1
        self.stats["errors"] += 1
        rec.errors.append(str(what))
        if self.memo.errors >= self.config.max_errors and not self.memo.disabled:
            self.memo.disabled = True
            self.stats["disabled"] += 1

    def _ms(self, t: float) -> float:
        return round((self.clock() - t) * 1000.0, 3)

    def _done(self, rec: SearchRecord, decision: Any, outcome: str, t0: float,
              detail: str = "") -> tuple[Any, SearchRecord]:
        rec.outcome = outcome
        for c in rec.candidates:                # a search that ended early keeps no boards
            c.afters = None
        if detail and not rec.detail:
            rec.detail = str(detail)
        rec.decision = _brief(decision)
        rec.ms["total"] = self._ms(t0)
        self.stats[f"outcome:{outcome}"] += 1
        self.stats["lookahead"] += int(rec.lookahead)
        for name in rec.triggers:
            self.stats[f"trigger:{name}"] += 1
        for k, v in rec.ms.items():
            self.ms[k] += float(v)
        self.last = rec
        return decision, rec

    def summary(self) -> dict[str, Any]:
        """The counters (S08: seat_info): decisions, searches, overrides and vetoes, outcomes
        (fallbacks by reason), root_fallbacks (decisions whose look-ahead root was not usable,
        by reason: s_root's fallbacks, whose outcome is always "root"), triggers, overrides by
        incumbent > choice (card id or "wait"), exceptions, battles search was turned off in, ms
        per stage (sums), the rebuilder's."""
        s = self.stats

        def prefixed(p: str) -> dict[str, int]:
            return {k[len(p):]: int(v) for k, v in sorted(s.items()) if k.startswith(p)}

        rb_summary = getattr(self.rebuilder, "summary", None)
        return {"preset": self.config.preset, "config": self.key,
                "privileged": bool(self.config.privileged),
                "decisions": int(s["decisions"]), "lookahead": int(s["lookahead"]),
                "searches": int(s["searches"]), "overrides": int(s["overrides"]),
                "vetoes": int(s["vetoes"]), "errors": int(s["errors"]),
                "disabled": int(s["disabled"]), "outcomes": prefixed("outcome:"),
                "root_fallbacks": prefixed("root_fallback:"),
                "triggers": prefixed("trigger:"), "override_by": prefixed("override:"),
                "ms": {k: round(float(v), 3) for k, v in sorted(self.ms.items())},
                "rebuild": rb_summary() if callable(rb_summary) else {}}


# ===========================================================================
# S08: the simulator seat (module doc, THE SEAT)
# ===========================================================================

def _pinned(settings: Any) -> dict[str, Any]:
    """A RebuildSettings as a seat's search pins it: to_json() less the provenance fields, and
    less the REBUILD_OPTIONAL fields that are at RebuildSettings()'s value."""
    base = RB.RebuildSettings().to_json() if REBUILD_OPTIONAL else {}
    return {k: v for k, v in settings.to_json().items() if k not in REBUILD_PROVENANCE
            and not (k in REBUILD_OPTIONAL and v == base.get(k))}


def _seat_json(cfg: SearchConfig) -> dict[str, Any]:
    """cfg.to_json() less the SEAT_OPTIONAL fields that are at the value resolve gives them back
    from the preset alone (its PRESETS entry, else SearchConfig()'s)."""
    out = cfg.to_json()
    base = resolve({"preset": cfg.preset})
    for name in SEAT_OPTIONAL:
        if name in out and out[name] == getattr(base, name):
            del out[name]
    return out


def rebuild_settings(value: Any = None) -> Any:
    """The rs_rebuild.RebuildSettings a seat pins (module doc, THE SEAT): None / "defaults":
    RebuildSettings(), the plan's; "probes": RebuildSettings.from_report(), the newest S01 engine
    report with probes, read now (the defaults when there is none); "R1" / "R2" (joined by "+")
    the level; a mapping: RebuildSettings.from_json; a RebuildSettings as it is. Level R0 is
    refused (SearchConfigError): it rebuilds from the seat's privileged state."""
    if isinstance(value, RB.RebuildSettings):
        settings = value
    elif isinstance(value, Mapping):
        try:
            settings = RB.RebuildSettings.from_json(dict(value))
        except (TypeError, ValueError) as exc:
            raise SearchConfigError(f"search rebuild settings: {exc}") from None
    elif value is None or isinstance(value, str):
        text = "" if value is None else value.strip()
        tokens = _rebuild_tokens(text) if text else (REBUILD_DEFAULTS,)
        settings = (RB.RebuildSettings.from_report() if REBUILD_PROBES in tokens
                    else RB.RebuildSettings())
        for level in (RB.R1, RB.R2):
            if level in tokens:
                settings = settings.replace(level=level)
    else:
        raise SearchConfigError(f"search rebuild: 'defaults', 'probes', a level or a "
                                f"RebuildSettings dict, got {type(value).__name__}")
    if settings.level not in (RB.R1, RB.R2):
        raise SearchConfigError(f"search rebuild level {settings.level!r}: a seat rebuilds at "
                                f"{RB.R1} or {RB.R2} ({RB.R0} reads the seat's privileged "
                                "state: the regret instrument's)")
    return settings


def seat_search(value: Any, *, verify: bool | None = None) -> dict[str, Any] | None:
    """What rs_match.AgentSpec.search holds for `value` (module doc, THE SEAT): None for no
    search; else SearchConfig.to_json() (less the SEAT_OPTIONAL fields at their preset's value:
    verify only when on) and "rebuild", the pinned RebuildSettings (rebuild_settings of
    "name@tokens"' tokens, or of a mapping's "rebuild"; to_json() less the provenance).
    Idempotent: seat_search(seat_search(x)) == seat_search(x). verify, when not None, sets
    SearchConfig.verify (rs_match --search-verify; a verified arm's games are keyed apart from
    the same arm's without it, though they are the same games: verify must reach the workers,
    which get the spec alone)."""
    rebuild: Any = None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                value = json.loads(text)
            except ValueError as exc:
                raise SearchConfigError(f"search config: not a JSON object ({exc})") from None
            if not isinstance(value, Mapping):
                raise SearchConfigError("search config: a JSON object")
        elif "@" in text:
            rebuild = text.split("@", 1)[1].strip()
    if isinstance(value, Mapping):
        rebuild = value.get(SEAT_REBUILD)
    cfg = resolve(value)
    if cfg is None:
        if rebuild is not None:
            raise SearchConfigError("search 'off' takes no rebuild settings")
        return None
    if verify is not None:
        cfg = cfg.replace(verify=bool(verify))
    return {**_seat_json(cfg), SEAT_REBUILD: _pinned(rebuild_settings(rebuild))}


def probes_source(value: Any) -> str | None:
    """Where a seat search's "@probes" settings come from (module doc, THE SEAT): the path of the
    S01 engine report RebuildSettings.from_report() reads now, or "defaults" when there is none
    (the plan's defaults are pinned then); None when `value` does not ask for the probes. The
    pinned settings leave this provenance out of the key, so the command lines print it."""
    rebuild: Any = None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                value = json.loads(text)
            except ValueError:
                return None
        elif "@" in text:
            rebuild = text.split("@", 1)[1].strip()
    if isinstance(value, Mapping):
        rebuild = value.get(SEAT_REBUILD)
    if not isinstance(rebuild, str) or REBUILD_PROBES not in _rebuild_tokens(rebuild):
        return None
    return probe_report_source()


def probe_report_source() -> str:
    """The S01 engine report "@probes" reads now (RebuildSettings.from_report's provenance: its
    path), or "defaults" when there is none."""
    return str(RB.RebuildSettings.from_report().probes or "defaults")


def seat_config(value: Any) -> SearchConfig | None:
    """The SearchConfig of a seat's search (seat_search's dict, or anything resolve takes)."""
    return resolve(value)


def check_seat(value: Any) -> SearchConfig | None:
    """resolve(value), or SearchConfigError for a search a simulator seat cannot run yet: the
    opponent "reply" (s_reply) needs S07's opponent world model (tools/opp_model.py). rs_match's
    and rs_league_eval's command lines call it before any game is played."""
    cfg = resolve(value)
    if cfg is not None and cfg.opponent == OPP_REPLY:
        raise SearchConfigError(f"search {cfg.preset!r}: opponent 'reply' needs the opponent "
                                "world model (S07, tools/opp_model.py), not built yet: use s_root "
                                "or s_idle")
    return cfg


def opponent_checkpoint(value: Any) -> str | None:
    """The checkpoint of the opponent model a search's worlds sample from (plan 5, the reply arm;
    S07 adds it as SearchConfig.opp_ckpt), or None: an idle or true opponent, and every search
    before S07. tools/rs_league_eval.py warns when it is a gate seat's checkpoint."""
    cfg = resolve(value)
    ckpt = getattr(cfg, "opp_ckpt", None) if cfg is not None else None
    return str(ckpt) if ckpt else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    """The nearest-rank q-quantile of `values` (None for none)."""
    if not values:
        return None
    v = sorted(float(x) for x in values)
    return v[min(len(v) - 1, max(0, math.ceil(float(q) * len(v)) - 1))]


def _spread(ms: Sequence[float]) -> dict[str, Any]:
    """n, sum, p50, p95 and max of per-decision milliseconds (search_info)."""
    return {"n": len(ms), "sum": round(sum(ms), 3), "p50": _num(_quantile(ms, 0.5)),
            "p95": _num(_quantile(ms, 0.95)), "max": _num(max(ms)) if ms else None}


def _pooled(spreads: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """_spread()s of several games together: n and sum added, the mean, the largest p95 (an
    upper bound of the pooled p95, not it) and the largest max."""
    dm = [d or {} for d in spreads]
    n = sum(int(d.get("n") or 0) for d in dm)
    total = sum(float(d.get("sum") or 0.0) for d in dm)
    p95 = [float(d["p95"]) for d in dm if d.get("p95") is not None]
    top = [float(d["max"]) for d in dm if d.get("max") is not None]
    return {"n": n, "mean": _num(total / n) if n else None,
            "p95_max": _num(max(p95)) if p95 else None, "max": _num(max(top)) if top else None}


def latency_ticks_for(ms: float | None, tick_ms: float = 50.0) -> int | None:
    """The latency charge for a search's p95 milliseconds per decision: ceil(ms / tick_ms)
    (plan 6.3; 0 for none), None when unmeasured."""
    if ms is None:
        return None
    return max(0, math.ceil(float(ms) / float(tick_ms) - 1e-9))


#: search_info() counts merge_search_info adds over games.
_INFO_COUNTS = ("decisions", "lookahead", "searches", "overrides", "vetoes", "errors", "disabled",
                "fidelity_skips")


def merge_search_info(infos: Iterable[Mapping[str, Any] | None]) -> dict[str, Any] | None:
    """search_info() of several games together (None when there is none): the counts added;
    outcomes, root_fallbacks, triggers, override_by, the rebuilder's counters ("rebuild") and
    the seat's ("seat") merged; ms per stage summed; decision_ms and search_ms (the part the
    search adds to the plain decision: the latency charge's measure) with n and sum added,
    mean, the largest p95 (an upper bound of the pooled p95, not it) and the largest max."""
    infos = [i for i in infos if isinstance(i, Mapping) and i]
    if not infos:
        return None
    out: dict[str, Any] = {
        "games": len(infos), "presets": sorted({str(i.get("preset")) for i in infos}),
        "privileged": any(bool(i.get("privileged")) for i in infos)}
    for k in _INFO_COUNTS:
        out[k] = sum(int(i.get(k) or 0) for i in infos)
    for k in ("outcomes", "root_fallbacks", "triggers", "override_by", "rebuild", "seat"):
        total: Counter = Counter()
        for i in infos:
            total.update({str(name): int(n) for name, n in (i.get(k) or {}).items()
                          if _is_number(n)})
        out[k] = dict(sorted(total.items()))
    ms: Counter = Counter()
    for i in infos:
        ms.update({str(name): float(v or 0.0) for name, v in (i.get("ms") or {}).items()})
    out["ms"] = {k: round(v, 3) for k, v in sorted(ms.items())}
    out["decision_ms"] = _pooled(i.get("decision_ms") for i in infos)
    out["search_ms"] = _pooled(i.get("search_ms") for i in infos)
    return out


#: The E06 spawn table of each engine config (its key) this process has read.
_SPAWN_TABLES: dict[str, dict] = {}


def _spawn_table(engine: Any, royalesim: Any) -> dict:
    """rs_spawn_table.ensure(DEFAULT_PATH, engine, royalesim), read once per process and engine
    (ensure builds a Battle to fingerprint the engine on every call)."""
    key = engine.key()
    if key not in _SPAWN_TABLES:
        import rs_spawn_table
        _SPAWN_TABLES[key] = rs_spawn_table.ensure(rs_spawn_table.DEFAULT_PATH, engine, royalesim)
    return _SPAWN_TABLES[key]


def _rows_module() -> Any:
    """tools/rs_rows (numpy): the seat's rows, entity fields and elixir law; imported when a seat
    first acts (rs_match has imported it by then)."""
    import rs_rows
    return rs_rows


class SearchSeatMixin:
    """The search half of a simulator seat (module doc, THE SEAT). rs_search.SearchSeat is
    rs_match.MatchAgent with this mixed in first; the tests mix it into a stub with MatchAgent's
    attributes and override the hooks search_policy, search_rebuilder and search_row.

    It reads of the seat: rng, meta, policy (search_policy), decide_config(), opp_elixir(team,
    mem, tick) (SeatOppElixir), v2_kw, live and visibility (search_row), spell_aim and
    _spell_aim, engine, _catalogue(), and run_match's sent and unran; it refuses lead,
    hold_pending and noise. Of the match battle it reads check_deploy, through d0's legality, and
    only the privileged s_true more (module doc, THE TRUE COPY)."""

    search_spec: dict | None = None
    search_config: SearchConfig | None = None
    rebuild_settings: Any = None
    search_team: int | None = None
    search_delay: int = 0
    search_opp_delay: int = 0
    search_seed: int = 0
    searcher: Searcher | None = None
    #: Keep every SearchRecord.to_json() in `records` (profiling, S01's search bench). Off by
    #: default: a match makes about 600 of them.
    keep_records: bool = False
    records: list | None = None
    last_search: SearchRecord | None = None

    # ------------------------------------------------------------ set up
    def setup_search(self, search: Any, *, team: int, delay: int, opp_delay: int = 0,
                     seed: int = 0) -> "SearchSeatMixin":
        """Arm the seat (rs_match.build_seat): its search (seat_search's dict, or anything it
        takes), its team, its own command delay D (the spec delay plus its latency_ticks, from the
        spec's engine config), the opponent's, and the spec seed (the search's own generator
        and the rebuild's seed come from (seed, tick))."""
        spec = seat_search(search)
        if spec is None:
            raise SearchConfigError("setup_search: a search (a preset, a JSON object); 'off' is a "
                                    "plain MatchAgent")
        cfg = check_seat(spec)
        lead = int(getattr(self, "lead", 0) or 0)
        if lead:
            raise SearchConfigError(f"a search seat decides on its own look-ahead: lead {lead} "
                                    "would look ahead twice (give it lead 0)")
        if getattr(self, "hold_pending", False):
            raise SearchConfigError("hold_pending reads the match battle's pending commands: a "
                                    "search seat refuses it")
        team = int(team)
        if team not in (0, 1):
            raise SearchConfigError(f"setup_search: team 0 or 1, got {team}")
        self.search_spec = spec
        self.search_config = cfg
        self.rebuild_settings = RB.RebuildSettings.from_json(spec[SEAT_REBUILD])
        self.search_team = team
        self.search_delay = max(0, int(delay))
        self.search_opp_delay = max(0, int(opp_delay))
        self.search_seed = int(seed)
        self.searcher = None
        self.search_stats: Counter = Counter()
        self._search_sent: list[tuple[int, int, int, int]] = []
        self._search_ms: list[float] = []
        self._search_inc_ms: list[float] = []
        self._search_decisions = 0
        self._search_match: Any = None
        self._search_keep: Any = None
        self._search_F: dict | None = None
        self._search_SF: dict | None = None
        self._search_mirror: tuple = ()
        self.records = [] if self.keep_records else None
        return self

    # ------------------------------------------------------------ hooks (tests override)
    def search_policy(self) -> Any:
        """The search's policy: a BatchedPolicy of the seat's TokenPolicyAdapter (as_batched)."""
        return as_batched(self.policy)

    def search_rebuilder(self, royalesim: Any, sub: int, fields: Mapping[str, int]) -> Any:
        """The seat's rs_rebuild.Rebuilder: one Battle for the match, under the seat's own engine
        config (undelayed), the E06 spawn table as rs_eval's rebuild reads it
        (rs_spawn_table.ensure, once per process and engine: _spawn_table) and the pinned
        settings."""
        import rs_engine
        engine = rs_engine.resolve(getattr(self, "engine", None)).undelayed()
        return RB.Rebuilder(engine, royalesim, _spawn_table(engine, royalesim),
                            settings=self.rebuild_settings, fields=fields, sub=sub)

    def search_row(self, state: Mapping[str, Any], team: int, mem: Mapping[str, Any], sub: int,
                   royalesim: Any, tick: int) -> Any:
        """The policy's row of `state` dated `tick`: RSAgent.act's, lead 0 (rs_rows.obs_row with
        the seat's live, visibility and v2 keywords)."""
        RR = _rows_module()
        tick = int(tick)
        return RR.obs_row(state, team, sub, mem, tick, live=self.live, visibility=self.visibility,
                          **self.v2_kw(state, team, mem, tick, royalesim))

    def search_law(self) -> Any:
        """The elixir law the leak is counted under: the seat's engine's (rs_rows.elixir_law)."""
        return _rows_module().elixir_law(getattr(self, "engine", None))

    # ------------------------------------------------------------ one decision
    def _ensure_search(self, team: int, sub: int, royalesim: Any) -> Searcher:
        if self.searcher is not None:
            return self.searcher
        RR = _rows_module()
        if RR.F:
            F = dict(RR.F)
        else:
            import rs_engine
            F = rs_engine.field_index(royalesim)
        cat = self._catalogue()
        names = [str(row[0]) for row in cat]
        self._search_F = F
        self._search_SF = RR.spell_index(royalesim)
        self._search_mirror = tuple(PV.mirror_ids(cat))
        # A seat plays one match (rs_match.build_seat makes one per match), so nothing here
        # starts a new battle when the view's tick steps back: under perception clock noise it
        # does within a match (module doc, THE SEAT, OUR OWN CLOCK).
        self._search_keep = PV.Permanence(fields=F, team=team,
                                          hiders=PV.card_ids(PV.HIDER_NAMES, names),
                                          spell_cards=PV.spell_ids(cat), sub=sub,
                                          rewind_ticks=None)
        rb = self.search_rebuilder(royalesim, sub, F)
        sightings = getattr(rb, "sightings", None)
        if sightings is not None and hasattr(sightings, "rewind_ticks"):
            sightings.rewind_ticks = None
        self.searcher = Searcher(self.search_policy(), rb, self.search_config,
                                 dcfg=self.decide_config(),
                                 landing=RD.obs_delay_ticks(getattr(self, "meta", None)) == 0,
                                 sub=sub, law=self.search_law(),
                                 privileged=self.search_config.privileged)
        return self.searcher

    def _own_rows(self, own_unran: Mapping[tuple, int]) -> list[tuple[int, int, int, int]]:
        """Our sends on our own clock (module doc, THE SEAT), less the plays the engine refused:
        our k-th send is run_match's k-th seat.sent row (true clock), whose (card, exec tick)
        the seat's own `unran` entries name. Under --mem-lag legacy run_match dates the memory
        entry, and so the unran entry, at the decision tick (exec tick - our delay): that key is
        tried when the exec tick's has none."""
        mine = list(self._search_sent)
        true = [tuple(int(v) for v in s[:4]) for s in (getattr(self, "sent", None) or ())]
        if len(true) != len(mine):
            self.search_stats["sent_unpaired"] += 1
        drop = Counter({(int(c) - rs_memory.TOKEN_OFFSET, int(t)): int(n)
                        for (_who, c, t), n in own_unran.items() if int(n) > 0})
        out = []
        for i, row in enumerate(mine):
            if i < len(true):
                if true[i][1] != row[1]:
                    self.search_stats["sent_mismatch"] += 1
                key = (true[i][1], true[i][0])
                if drop[key] <= 0:
                    key = (true[i][1], true[i][0] - int(self.search_delay))
                if drop[key] > 0:
                    drop[key] -= 1
                    continue
            out.append(row)
        return out

    def _by_hand(self, rows: Sequence[tuple[int, int, int, int]], hand: Iterable[int],
                 now: int) -> list[tuple[int, int, int, int]]:
        """Our sends (own clock, refusals out) with the in-flight cut corrected by our hand, the
        live rule (rs_decide.prune_pending): the engine cycles a card out of the hand only when
        its play RUNS, so a recent play whose card is still in the hand has not run, and one
        whose card has left it has. Our own clock is the noised view tick + delay, so it can
        call a play landed that has not run yet, or in flight when it has run (perception
        clock noise on both ticks). A play still in the hand whose own-clock tick has passed is
        dated `now` (it lands at the root's first tick); one that has left the hand but whose
        own-clock tick is still ahead is dated now - 1 (out of own_inflight, still in own_sent,
        where its aim is read). Only sends with fewer than HAND_SIZE later ones are judged: a
        card comes back to the hand only after HAND_SIZE later plays have run."""
        hand = {int(c) for c in hand or () if int(c) >= 0}
        n, now = len(rows), int(now)
        out = []
        for i, (exec_tick, card, x, y) in enumerate(rows):
            if n - 1 - i < RD.HAND_SIZE:
                if int(card) in hand:
                    if exec_tick < now:
                        exec_tick = now
                        self.search_stats["inflight_late"] += 1
                elif exec_tick >= now:
                    exec_tick = now - 1
                    self.search_stats["inflight_ran"] += 1
            out.append((int(exec_tick), int(card), int(x), int(y)))
        return out

    def search_view(self, state: Mapping[str, Any], team: int,
                    mem: Mapping[str, Any] | None) -> dict:
        """The seat's public view of `state` (the observed state run_match hands it) at its tick
        (module doc, THE SEAT): public_view with the seat's own count of the opponent's elixir,
        our sends on our own clock, the refusals left out, the in-flight cut checked against our
        hand (_by_hand), then Permanence."""
        team = int(team)
        reading = PV.SeatOppElixir(self).view(team, mem, int(state["tick"]))
        unran = dict(getattr(self, "unran", None) or {})
        own = Counter({k: int(n) for k, n in unran.items() if int(k[0]) == team})
        opp = Counter({k: int(n) for k, n in unran.items() if int(k[0]) != team})
        # Our cycle as public_view derives it with every refusal of ours left out (our sends,
        # on our own clock, are cut here: public_view gets only the opponent's refusals).
        me = state["players"][team]
        cycle = PV.CardCycle.from_plays([c for c, _ in PV.plays_of(mem, team, own)],
                                        hand=me.get("hand"), next_card=me.get("next_card"),
                                        deck=me.get("deck") or (), mirror=self._search_mirror)
        sent = self._by_hand(self._own_rows(own), me.get("hand") or (), int(state["tick"]))
        view = PV.public_view(state, team, mem, fields=self._search_F,
                              spell_fields=self._search_SF, opp_elixir=reading,
                              sent=sent, unran=opp, mirror=self._search_mirror,
                              own_cycle=cycle)
        # What the cut removed, counted (unseen enemies, projectiles), is itself privileged: an
        # invisible unit's existence. Nothing downstream reads it; the search never gets it.
        view["public"].pop("dropped", None)
        return self._search_keep.apply(view)

    def act(self, battle: Any, state: Mapping[str, Any], team: int, mem: Mapping[str, Any],
            tick: int, sub: int, royalesim: Any) -> tuple[int, int, int, int] | None:
        """One decision (module doc, THE SEAT): the search on the seat's public view; the command
        to send now (None: wait). run_match's seat interface (RSAgent.act's)."""
        cfg = self.search_config
        if cfg is None:
            raise SearchError("a SearchSeat acts only after setup_search (rs_match.build_seat "
                              "calls it)")
        team = int(team)
        if team != self.search_team:
            raise SearchError(f"this search seat was set up for team {self.search_team}, not "
                              f"{team}: its delays are that team's")
        if getattr(self, "noise", None) is not None:
            raise SearchError("a SearchSeat decides on run_match's observed state; it applies no "
                              "perception noise of its own (rs_eval.play's seat.noise)")
        unran = getattr(self, "unran", None)
        if self._search_decisions and unran is not self._search_match:
            raise SearchError("a SearchSeat plays one match (rs_match.build_seat makes one per "
                              "match): run_match handed it a new match's refusals")
        self._search_match = unran
        self._team = team
        searcher = self._ensure_search(team, sub, royalesim)
        view = self.search_view(state, team, mem)
        now = int(view["tick"])
        shift = int(tick) - now                 # 0 under run_match: its tick is the view's
        # The decision clock (the hold, the new-battle check): run_match asks a seat once per
        # cadence, so the decisions so far x cadence is our elapsed time, which never steps back
        # as the noised view tick can (module doc, THE SEAT, OUR OWN CLOCK).
        clock = self._search_decisions * int(getattr(self, "cadence", RD.DECIDE) or RD.DECIDE)
        self._search_decisions += 1

        def row_fn(s: Mapping[str, Any], t: int) -> Any:
            return self.search_row(s, team, mem, sub, royalesim, int(t) + shift)

        aim = self._spell_aim(view, team, sub) if getattr(self, "spell_aim", False) else None
        d, rec = searcher.decide(view, team=team, land_ticks=self.search_delay,
                                 legality=RD.BattleLegality(battle), rng=self.rng, row_fn=row_fn,
                                 tick=int(tick), aim=aim, seed=self.search_seed,
                                 true_battle=battle if cfg.privileged else None,
                                 clock_tick=clock)
        self.last_decision, self.last_search = d, rec
        total = float(rec.ms.get("total") or 0.0)
        self._search_ms.append(total)
        # What the search adds to the plain seat's decision (its row and d0 are the plain
        # seat's work too): the latency charge's measure (plan 6.3, p95_search_ms).
        self._search_inc_ms.append(max(0.0, total - float(rec.ms.get("row") or 0.0)
                                       - float(rec.ms.get("d0") or 0.0)))
        if self.records is not None:
            self.records.append(rec.to_json())
        cmd = d.command
        if cmd is not None and d.is_play:
            self._search_sent.append((now + self.search_delay, int(d.card), int(cmd[2]),
                                      int(cmd[3])))
        return cmd

    # ------------------------------------------------------------ the record
    def search_info(self) -> dict[str, Any]:
        """What rs_match.seat_info records of the search (module doc, THE SEAT); JSON-ready."""
        cfg = self.search_config
        if cfg is None:
            return {}
        s = self.searcher.summary() if self.searcher is not None else {}
        ms = list(self._search_ms)
        out: dict[str, Any] = {
            "preset": cfg.preset, "config": cfg.key, "privileged": bool(cfg.privileged),
            "rebuild_key": self.rebuild_settings.key, "rebuild_level": self.rebuild_settings.level,
            "delay": int(self.search_delay), "opp_delay": int(self.search_opp_delay)}
        out.update({k: v for k, v in s.items() if k not in out})
        out["fidelity_skips"] = int((s.get("root_fallbacks") or {}).get(R_FIDELITY, 0))
        out["decision_ms"] = _spread(ms)
        out["search_ms"] = _spread(list(getattr(self, "_search_inc_ms", None) or ()))
        out["seat"] = {k: int(v) for k, v in sorted(self.search_stats.items())}
        return out


_SEAT_CLASS: Any = None


def _search_seat_class() -> type:
    """SearchSeat: rs_match.MatchAgent with SearchSeatMixin first, made once (rs_match is
    imported here, not when this module is)."""
    global _SEAT_CLASS
    if _SEAT_CLASS is None:
        import rs_match
        _SEAT_CLASS = type("SearchSeat", (SearchSeatMixin, rs_match.MatchAgent), {
            "__module__": __name__, "__qualname__": "SearchSeat",
            "__doc__": ("rs_match.MatchAgent whose decisions are the search's (module doc, THE "
                        "SEAT): rs_match.build_seat builds it for an AgentSpec with `search` and "
                        "calls setup_search.")})
    return _SEAT_CLASS


def __getattr__(name: str) -> Any:
    """rs_search.SearchSeat, made on first use (_search_seat_class)."""
    if name == "SearchSeat":
        return _search_seat_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
