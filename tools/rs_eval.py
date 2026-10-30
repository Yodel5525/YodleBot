"""Ghost screen in RoyaleSim: a token-policy checkpoint plays the Hog 2.6 side
of held-out recorded matches while the other side replays what the real
opponent did, tick for tick.

    .venvs/royalegym/Scripts/python.exe tools/rs_eval.py --ckpt runs/tokens/rs_bc/best.pt \
        --matches 300 [--search]

Two numbers per run, on the same matches:
* `pro`: the recorded Hog player's own plays replayed against the same ghost.
  This is what the real player achieves in this engine - the bar.
* `agent`: the checkpoint in that seat.

Engine: every Battle comes from tools/rs_engine.make_battle under
data/engine.json (card table, levels); `--delay` sets the command delay of the
match battle, and look-ahead rebuilds run without one. `--card-table default`
runs the engine's own table; `--engine-config legacy` builds every Battle
exactly as before rs_engine. A checkpoint whose card / unit vocabulary or card
table is not this engine's is refused (tools/rs_checkpoint.py;
`--allow-vocab-drift` loads it with a warning). The card -> unit-type table
data/rs_rows/spawn_primary.json is regenerated when it is missing or was made
under another engine (tools/rs_spawn_table.py).

Matches come from the test split (hash of the tag, as the row builders use),
so no training row ever came from them. The side whose troops go down at native
low y plays blue, with coordinates as recorded (tools/rs_orient.py); a match
where either side has under 90% of its troop plays on its own half is skipped,
counted and printed. The ghost cannot react - it plays the
recorded commands even when the board has changed - which flatters an agent
that disrupts a scripted push; the `pro` row is measured the same way, so the
comparison between the two is fair even if both numbers are optimistic.

Opponent memory (E08, tools/rs_memory.py; what opp_seen / opp_last show a seat):
every play is dated on the tick it runs, send tick + --delay, and each seat
decides on one view per tick, taken before anything is sent that tick, holding
the plays with exec_tick + --mem-lag <= tick (default BOARD_LAG = 1: what the
board of that tick shows). So no seat sees a card before it is on the board, nor
the other seat's play of the same tick. `--mem-lag legacy` dates plays when they
are sent and reads the memory at each decision, as before E08: identical results,
with a card in memory up to `delay` ticks before it exists. A run whose lag is
not the default says so in its name; reports/rs_ghost.jsonl records "mem_lag".

Decisions (V06): RSAgent.act builds the row and hands it to tools/rs_decide.decide, the one
decision function the live runner and RL use too, with check_deploy on the match battle as the
legality. `--decide legacy` (the default) is the rule this file always had, decision for
decision and random draw for draw (tests/pipeline/test_rs_decide.py replays the old code);
`--decide live` is the live runner's: slots of our plays still in flight under --delay are
blocked (RoyaleSim refuses them CARD_PENDING), a card with no legal tile hands over to the next
one, no wait at 9.5 elixir, and the Cannon's lane guard. A run that is not legacy says so in its
name; reports/rs_ghost.jsonl then records "decide".

Visibility (E12, tools/rs_rows.py --visibility): each checkpoint seat sees what its rows showed,
the "visibility" of its meta (vocab.json; absent, as in every checkpoint before E12: legacy,
every living unit). `--visibility legacy|public` overrides that for every checkpoint seat.
Under public the seat's rows (RSAgent.act, RSSearchAgent), RSAgent.rebuilt (the live look-ahead:
an enemy the screen does not show is not spawned) and aim_spell leave out the enemy units with
STATUS_INVISIBLE or STATUS_HIDDEN (rs_rows.seen_by), and the rows carry unit_burrow. The match
battle, the ghost and the engine's own look-ahead copy are not filtered: they are the battle,
not a view of it. A run given --visibility says so in its name; reports/rs_ghost.jsonl then
records "visibility".

Schema v2 (E15; tools/rs_schema.py, sim/model_tokens.py): a checkpoint whose meta "schema" is
rs_rows/v2 (trained on tools/rs_rows.py --schema v2 rows) gets v2 rows (RSAgent.v2_kw), with
what its rows held at training time and the seat knows: own_pending, its own plays sent and not
run yet (the battle's players[team]["pending"] play rows, card + 2 and ticks to land, soonest
first; with --lead, those still to land after the look-ahead, ticks less the lead); opp_elixir,
the opponent's bar as this seat counts it from the plays its memory view shows
(rs_rows.SeenElixir with lag 0, fed each play of mem["last"] once, priced from the catalogue; a
recorded ghost play the engine refused is counted too, as the memory lists it); and its ability
buttons named from its dealt deck and forms (rs_presses.side_buttons; play() sets them). The
v2 rows read the burrow bit, so play() checks royalesim.STATUS_BITS for a v2 seat. The v2 gate
has the ability buttons after the four slots; rs_decide masks them (its button_mask) and the
search agent reads the four slots only, so no seat presses yet. A legacy checkpoint (no schema
in its meta, or rs_rows/legacy) is played exactly as before.

Perception noise (E17; tools/perception_noise.py, tools/rs_rows.py --noise): without `--noise`
(the default, or `--noise none`) every seat reads the engine's state, as before, and the results
are those of before. `--noise PROFILE` (a profile JSON or "default"; `--noise-only KINDS`,
`--noise-seed N`) makes the AGENT seat (mode agent, the RSAgent.act seat) a screen reader: play()
gives it a perception_noise.MatchNoise for the match (agent.noise, seeded by match_seed(replay
tag, N)), pushes every state the match goes through into its history, and RSAgent.act first
replaces the state with noise.observe(state, team), so the look-ahead (--lead -N rebuilds from
what the screen shows), the row (at the tick its clock reads, tick + lead + tick_shift; the
opponent's counted elixir stays at the true tick + lead) and rs_decide's masks (the HUD's
elixir) all read it; the match battle and check_deploy stay the engine's. The other side's plays
(the ghost's and the reactive opponent's) go into the memory as that seat perceives them
(MatchNoise.record: missed, late, or the unknown card). `--opponent-noise PROFILE` does the same
for the reactive opponent seat (--opponent-ckpt; seeded by match_seed(tag + "|opponent", N)), and
the agent's plays then reach the memory through it; by default the opponent sees perfectly, so
the screen prices the agent's perception only. --pro (no agent seat), --search (its own act) and
--shadow (a re-drive of the true state) are refused with --noise. A noisy run says so in its
name; reports/rs_ghost.jsonl then records "noise" (perception_noise.vocab_entry) and the summed
counts ("noise_stats"; per match in each row's "noise"), and likewise "opponent_noise".

The 95% interval is tools/league/stats.wilson (L01), the formula this file had.

Workers (capped_workers): --workers is capped by the free memory and the CPUs the way
tools/rs_league_eval.py caps its own (worker_cap: the free memory less RESERVE_MB, at WORKER_MB a
worker, which holds torch, royalesim and two policies as a league worker does, and the CPUs less
one), but not at the league's MAX_WORKERS; the run prints "workers: ..." with the reason. With
--search or --shadow a worker counts SEARCH_WORKER_MB, as a league worker with a searching player
does: the search's copies or rebuild, or the shadow's re-drive, are a second battle. The
results do not depend on the count: match k of the held-out list is played with seed k (the
battle's reset, the agent; the reactive opponent k + 7) and its noise is seeded by its replay
tag, every match builds its own agents, and pool.map returns the results in the list's order.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mode

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import rs_rows as RR  # noqa: E402
from royalesim_redrive import CORPUS, base_key, deal_order  # noqa: E402
import perception_noise as PN  # noqa: E402  (E17; rs_rows imports it too)
import rs_decide  # noqa: E402
import rs_engine  # noqa: E402
import rs_memory  # noqa: E402
import rs_orient  # noqa: E402
import rs_presses  # noqa: E402
import rs_schema  # noqa: E402
from tools.league.stats import wilson  # noqa: E402  (rs_eval.wilson, as before L01)

DECIDE = 10           # ticks between agent decisions (0.5 s)


def to_world(ox, oy, team):
    """Acting player's own frame (millitiles) -> RoyaleSim world millitiles."""
    return (ox, 32_000 - oy) if team == 0 else (18_000 - ox, oy)


def seat_visibility(agent) -> str:
    """The --visibility a seat sees with (module doc): its `visibility` (RSAgent: its
    checkpoint's, or the --visibility override), else rs_rows.VISIBILITY. Read with getattr,
    so a plain namespace standing in for an agent (RSAgent.rebuilt(ns, ...)) works too."""
    vis = getattr(agent, "visibility", None)
    return RR.visibility_mode(RR.VISIBILITY if vis is None else vis)


class RSAgent:
    lead = 0      # ticks to look ahead before deciding (delay compensation)
    # The engine config of the battles this agent builds itself (rebuilt look-aheads, the
    # catalogue): an rs_engine.EngineConfig, or None for the process's rs_engine.active().
    engine = None
    # The decision rule (tools/rs_decide.py): MODE_LEGACY, the rule this agent always had, or
    # MODE_LIVE, the live runner's. last_decision is the rs_decide.Decision of the last act.
    decide_mode = rs_decide.MODE_LEGACY
    last_decision = None
    # What this seat's rows show (rs_rows.VISIBILITY_MODES): __init__ reads the checkpoint's
    # (rs_rows.visibility_of), play() sets --visibility over it; None: rs_rows.VISIBILITY.
    visibility = None
    # E15: a v2 checkpoint (meta "schema" rs_rows/v2) reads v2 rows (module doc, "Schema v2");
    # buttons: the seat's ability button cards (catalogue ids, button order) or None, which
    # play() sets for a v2 seat. _seen / _seen_fed / _seen_team: the opponent elixir count.
    v2 = False
    buttons = None
    _seen = None
    _seen_fed = None
    _seen_team = None
    # E17 (--noise): a perception_noise.MatchNoise play() sets for the match; act() then decides
    # on what it observes (module doc, "Perception noise"). None: the engine's state, as before.
    noise = None

    _CAT = None

    def cat_kind(self, card):
        if RSAgent._CAT is None:
            RSAgent._CAT = rs_engine.catalogue(rs_engine.resolve(self.engine).undelayed())
        return RSAgent._CAT[card][8]

    def _catalogue(self):
        """The catalogue rows cat_kind reads (built once per process)."""
        if RSAgent._CAT is None:
            RSAgent._CAT = rs_engine.catalogue(rs_engine.resolve(self.engine).undelayed())
        return RSAgent._CAT

    def aim_spell(self, order, state, team, card, sub, top=5):
        """Among the model's top tiles for a spell, the one whose radius holds
        the most enemy hit points on the visible board. The model picks the
        spell and the neighbourhood; this only corrects the aim. Public only: under
        --visibility public an enemy the screen does not show (rs_rows.seen_by) counts no
        hit points."""
        F = RR.F
        radius = (RSAgent._CAT[card][4] or 2500) / 1000          # tiles
        public = seat_visibility(self) == RR.VIS_PUBLIC
        enemies = [(e[F["x"]] / sub / 1000, e[F["y"]] / sub / 1000, e[F["hp"]])
                   for e in state["entities"] if e[F["team"]] != team and e[F["hp"]] > 0
                   and (not public or RR.seen_by(e, team, F))]
        if not enemies:
            return order
        best, best_hp = None, 0.0
        for t in order[:top]:
            tx, ty = int(t % 18), int(t // 18)
            wx, wy = to_world(tx * 1000 + 500, ty * 1000 + 500, team)
            wx, wy = wx / 1000, wy / 1000
            hp = sum(h for ex, ey, h in enemies if (ex - wx) ** 2 + (ey - wy) ** 2 <= radius ** 2)
            if hp > best_hp:
                best, best_hp = t, hp
        if best is None:
            return order
        return np.concatenate([[best], [t for t in order if t != best]])

    def ahead(self, battle, state):
        """The board `lead` ticks from now, with nothing new played.

        A play sent now lands ~21 ticks later, so the board worth deciding on
        is that one, not the present. Pending plays (ours) execute in the
        copy, which is what they will do. Live, the same is done on a rebuilt
        battle; here the engine's own copy is used.
        """
        if not self.lead:
            return state
        if getattr(self, "shadow", None) is not None:
            # The live-capable path: look ahead on the shadow re-drive, with our
            # own pending plays issued at the ticks they will execute.
            now = state["tick"]
            sim = self.shadow.battle(now)
            saved = sim.save()
            pend = sorted(self.shadow.pending_own(now))
            t = now
            for (te, team, card, x, y) in pend:
                if te >= now + self.lead:
                    break
                if te > t:
                    sim.step([], te - t)
                    t = te
                st = json.loads(sim.state_json())
                hand = st["players"][team]["hand"]
                if card in hand:
                    sim.step([(team, hand.index(card), x, y)], 1)
                    t += 1
            shadow_now = json.loads(sim.state_json())
            if now + self.lead > t:
                sim.step([], now + self.lead - t)
            future = json.loads(sim.state_json())
            sim.load(saved)
            # The shadow drifts over a match (opponent plays seen late, a few
            # refused), so its towers are not trusted outright: the view keeps
            # the REAL tower hit points plus only the change the shadow
            # predicts over the look-ahead.
            for k in (0, 1):
                real = state["players"][k]["tower_hp"]
                a = shadow_now["players"][k]["tower_hp"]
                z = future["players"][k]["tower_hp"]
                future["players"][k]["tower_hp"] = [max(0, r - max(0, x - y))
                                                    for r, x, y in zip(real, a, z)]
                future["players"][k]["tower_max_hp"] = state["players"][k]["tower_max_hp"]
            for k in (0, 1):
                for key in ("hand", "hand_costs", "elixir_milli", "deck", "next_card"):
                    future["players"][k][key] = state["players"][k].get(key, future["players"][k].get(key))
            return future
        if getattr(self, "lead_extrapolate", False):
            # Public dead reckoning: each unit moves on by its own velocity over
            # the last decision interval. No engine, nothing hidden; units that
            # stop to attack overshoot, new units are not predicted.
            F = RR.F
            prev = getattr(self, "_prev", {})
            dt = max(1, state["tick"] - getattr(self, "_prev_tick", state["tick"] - DECIDE))
            future = json.loads(json.dumps(state))
            for e in future["entities"]:
                uid = e[F["uid"]]
                if e[F["tower_slot"]] < 0 and uid in prev:
                    px, py = prev[uid]
                    e[F["x"]] = int(e[F["x"]] + (e[F["x"]] - px) * self.lead / dt)
                    e[F["y"]] = int(e[F["y"]] + (e[F["y"]] - py) * self.lead / dt)
            self._prev = {e[F["uid"]]: (e[F["x"]], e[F["y"]]) for e in state["entities"]}
            self._prev_tick = state["tick"]
            return future
        if getattr(self, "lead_rebuild", False):
            # The live case: only what a player can see is carried forward, so
            # nothing hidden can leak into the view. Our own plays still in
            # flight are known to us and are replayed at the ticks they land;
            # leaving them out made the agent play again on top of them.
            import royalesim
            now = state["tick"]
            pend = [p for p in getattr(self, "sent", []) if p[0] > now]
            sim = self.rebuilt(state, self._team, royalesim,
                               first=[c for (_, c, _, _) in pend])
            if sim is None:
                return state
            t = now
            for (te, card, x, y) in sorted(pend):
                if te >= now + self.lead:
                    break
                if te > t:
                    sim.step([], te - t)
                    t = te
                hand = json.loads(sim.state_json())["players"][self._team]["hand"]
                if card in hand:
                    sim.step([(self._team, hand.index(card), x, y)], 1)
                    t += 1
            if now + self.lead > t:
                sim.step([], now + self.lead - t)
            future = json.loads(sim.state_json())
            for t in (0, 1):
                future["players"][t]["hand"] = state["players"][t]["hand"]
                future["players"][t]["hand_costs"] = state["players"][t]["hand_costs"]
                future["players"][t]["elixir_milli"] = state["players"][t]["elixir_milli"]
                future["players"][t]["deck"] = state["players"][t]["deck"]
                future["players"][t]["next_card"] = state["players"][t].get("next_card", -1)
            return future
        saved = battle.save()
        battle.step([], self.lead)
        future = json.loads(battle.state_json())
        battle.load(saved)
        # No peeking: the copy also lands the OPPONENT's pending plays, which a
        # real player cannot see until they appear. Drop every unit that does
        # not exist now unless it is ours (our own pending plays are known to
        # us). Residual leak: an enemy spell landing inside the window still
        # changes hit points in the view; measured separately by --rebuild.
        F = RR.F
        now = {e[F["uid"]] for e in state["entities"]}
        team = self._team
        future["entities"] = [e for e in future["entities"]
                              if e[F["uid"]] in now or e[F["team"]] == team]
        # Keep the present hand and elixir: the decision is about what we can
        # tap now; only the board is advanced.
        for t in (0, 1):
            future["players"][t]["hand"] = state["players"][t]["hand"]
            future["players"][t]["hand_costs"] = state["players"][t]["hand_costs"]
            future["players"][t]["elixir_milli"] = state["players"][t]["elixir_milli"]
        return future

    def rebuilt(self, state, team, royalesim, first=()):
        """A fresh Battle holding only what a player can see - the live case.

        Units as single spawns (card, team, position, hit points), tower hit
        points, the clock, our own deck in hand-then-queue order and our own
        elixir. The opponent's elixir is unknown and set to 0 (it makes no plays
        in the lookahead anyway). Lost: spells and projectiles in flight, attack
        timers, targets, deploy timers - which is exactly what this measures.
        Under --visibility public (seat_visibility) an enemy unit the screen does
        not show (rs_rows.seen_by: invisible or hidden) is not spawned either.
        """
        F = RR.F
        public = seat_visibility(self) == RR.VIS_PUBLIC
        me = state["players"][team]
        deck = list(me["deck"])
        hand = list(me["hand"])
        nxt = me.get("next_card", -1)
        hand = [c for c in hand if c >= 0 and c not in first]
        lead_cards = [c for c in first if c in deck]
        rest = [c for c in deck if c not in hand and c != nxt and c not in lead_cards]
        mine = lead_cards + hand + ([nxt] if nxt >= 0 and nxt not in lead_cards else []) + rest
        # dedupe while keeping order, then pad to the deck
        seen_c, ordered = set(), []
        for c in mine + deck:
            if c not in seen_c:
                seen_c.add(c)
                ordered.append(c)
        mine = ordered[:8]
        opp = state["players"][1 - team]["deck"]
        spawns = []
        prim, by_unit = _spawn_tables()
        for e in state["entities"]:
            if e[F["tower_slot"]] >= 0 or e[F["hp"]] <= 0:
                continue
            if public and not RR.seen_by(e, team, F):
                continue        # E12: not on the screen, so not in the live rebuild
            # A unit reports the card that made it, so a Witch's skeleton
            # would be rebuilt as a Witch. Rebuild it as the card whose own
            # unit it is; skip a unit no card produces.
            card, unit = e[F["card_id"]], e[F["unit_type"]]
            if prim.get(card) != unit:
                # An evolved or hero form of the card's own unit (Skeleton_EV1,
                # MusketeerHero, IceGolemiteHero) is the card itself, not a
                # child. Treating forms as children dropped every evolved or
                # hero unit from the rebuild: 98.5% of the units it lost.
                names_ = _unit_names()
                base_name = names_[prim[card]].lower() if card in prim else ""
                if base_name and names_[unit].lower().startswith(base_name):
                    pass
                else:
                    card = by_unit.get(unit)
                    if card is None:
                        continue
            spawns.append((e[F["team"]], card, e[F["x"]], e[F["y"]], e[F["hp"]]))
        # No command delay: our pending plays are stepped in at the ticks they land.
        b = rs_engine.make_battle(rs_engine.resolve(self.engine).undelayed(), royalesim)
        decks = [None, None]
        decks[team], decks[1 - team] = mine[:8], list(opp)[:8]
        elixir = [0, 0]
        elixir[team] = me["elixir_milli"]
        towers = [state["players"][t]["tower_hp"] for t in (0, 1)]
        try:
            b.reset(seed=0, decks=decks, shuffle=0, start_tick=state["tick"],
                    elixir_milli=elixir, tower_hp=towers, spawns=spawns)
        except Exception as exc:
            self.rebuild_error = f"{type(exc).__name__}: {exc}"
            return None
        return b

    def __init__(self, ckpt, seed=0, wait_keep=0.35, *, engine=None, allow_drift=None):
        import torch
        torch.set_num_threads(1)
        import rs_checkpoint
        if engine is not None:
            self.engine = engine
        # The checkpoint's card and unit ids are catalogue / unit_types indexes of the
        # engine its rows came from; another card table or royalesim renumbers them.
        # load_policy refuses a checkpoint that does not fit this agent's engine
        # (rs_checkpoint.VocabMismatch); allow_drift None reads HASTYCR_ALLOW_VOCAB_DRIFT.
        self.model, meta, self.vocab_drift = rs_checkpoint.load_policy(
            ckpt, rs_engine.resolve(self.engine), allow_drift=allow_drift)
        self.meta = meta
        self.torch = torch
        # A --units vision checkpoint (E16) reads vision-class unit tokens; the rows built
        # here are unit types (or creating cards). Refused, also when vocab drift is allowed.
        units = meta.get("units_mode", RR.UNITS_TYPES)
        if units != RR.UNITS_TYPES:
            raise ValueError(f"{ckpt}: a --units {units} checkpoint; rs_eval builds "
                             f"--units {RR.UNITS_TYPES} rows only")
        # E12: the rows it trained on showed every unit (legacy) or the screen's (public).
        self.visibility = RR.visibility_of(meta)
        # A live-compatible checkpoint was trained on card-id unit tokens.
        self.live = bool(meta.get("live_compat", False))
        # E15: a v2 checkpoint reads v2 rows; a legacy one (no schema) the rows it always did.
        self.v2 = checkpoint_v2(meta)
        self.rng = np.random.default_rng(seed)
        # Waits were subsampled to `wait_keep` in training, so the wait logit
        # is low by log(1 / wait_keep). Read the value the checkpoint was
        # trained with: a fixed 0.35 against a 0.25-trained model
        # under-corrected and the agent overplayed (~47 plays a match).
        wait_keep = rs_decide.wait_keep_for(ckpt, wait_keep)
        self.wait_bias = math.log(1 / wait_keep)

    def batch(self, row):
        if self.v2:
            from sim.model_tokens import row_batch
            return row_batch(row, rs_schema.SCHEMA_V2, self.torch)
        t = self.torch
        keys = ("unit_ids", "unit_feats", "towers", "scalars", "hand", "opp_seen",
                "opp_last", "opp_last_age")
        b = {k: t.as_tensor(np.asarray(row[k])[None]) for k in keys}
        for k in ("unit_ids", "hand", "opp_seen", "opp_last"):
            b[k] = b[k].long()
        b["unit_feats"] = b["unit_feats"].float()
        b["opp_last_age"] = b["opp_last_age"].float()
        b["next"] = t.as_tensor([int(row["next"])]).long()
        return b

    # ------------------------------------------------------------ E15: v2 rows
    def v2_kw(self, state, team, mem, tick, royalesim=None) -> dict:
        """The obs_row keywords of this seat's v2 row at `tick` (module doc, "Schema v2"): none
        for a legacy checkpoint, so its obs_row calls are exactly what they were."""
        if not self.v2:
            return {}
        kw = {"schema": rs_schema.SCHEMA_V2,
              "opp_elixir": self.opp_elixir(team, mem, tick, royalesim),
              "pending": self.own_pending(state, team)}
        if self.buttons is not None:
            kw["buttons"] = list(self.buttons)
        return kw

    def own_pending(self, state, team) -> list:
        """(card token, ticks to land) of `team`'s plays sent and not run, soonest first: the
        state's pending rows of kind 0 ([kind, what, x, y, ticks_left, cost]), those landing
        after the look-ahead (`lead`), ticks less the lead. Presses are not pending tokens."""
        lead = int(self.lead or 0)
        out = []
        for r in state["players"][team].get("pending") or ():
            if len(r) > 4 and int(r[0]) == 0 and int(r[1]) >= 0 and int(r[4]) >= lead:
                out.append((int(r[1]) + 2, int(r[4]) - lead))
        out.sort(key=lambda p: p[1])
        return out

    def opp_elixir(self, team, mem, tick, royalesim=None) -> tuple:
        """(milli, exact): the opponent's bar at `tick` as this seat counts it, from every play
        of mem["last"][opponent] (card token, exec tick; rs_memory's view lists them all),
        each fed once to an rs_rows.SeenElixir (lag 0: the view already holds only what the
        seat has seen; a token outside the catalogue is not priced). Any tick may be asked (the
        search agent's look-ahead rows ask later ones). A view that no longer holds a play
        already fed, or another team, starts a new count: a new match."""
        team, tick = int(team), int(tick)
        opp = 1 - team
        plays = [(int(c), int(t)) for c, t in (mem.get("last") or {}).get(opp, ())]
        held = Counter(plays)
        if (self._seen is None or self._seen_team != team
                or any(held[k] < n for k, n in self._seen_fed.items())):
            mod = royalesim if royalesim is not None else sys.modules.get("royalesim")
            fields = list(getattr(mod, "CATALOGUE_FIELDS", ()) or ()) or None
            self._seen = RR.SeenElixir(self._catalogue(), fields=fields,
                                       law=RR.elixir_law(self.engine), lag=0)
            self._seen_fed = Counter()
            self._seen_team = team
        n_cards = len(self._catalogue())
        counts = Counter()
        for key in plays:
            counts[key] += 1
            if counts[key] <= self._seen_fed[key]:
                continue
            self._seen_fed[key] += 1
            cid = key[0] - 2
            if 0 <= cid < n_cards and key[1] >= 0:
                self._seen.record(opp, key[1], cid)
        return self._seen.view(team, tick)

    @property
    def policy(self):
        """The model as rs_decide sees it (batch / gate / tiles), with this agent's batch."""
        p = self.__dict__.get("_policy")
        if p is None or p.model is not self.model or p.torch is not self.torch:
            p = self.__dict__["_policy"] = rs_decide.TokenPolicyAdapter(
                self.model, self.torch, batch=self.batch)
        return p

    def decide_config(self):
        """The rs_decide.DecideConfig of `decide_mode`, with this checkpoint's wait bias."""
        if self.decide_mode == rs_decide.MODE_LEGACY:
            return rs_decide.DecideConfig.legacy(self.wait_bias, dt=DECIDE)
        if self.decide_mode == rs_decide.MODE_LIVE:
            guard = rs_decide.card_ids(self._catalogue(), rs_decide.GUARD_CARD_NAMES)
            return rs_decide.DecideConfig.live(self.wait_bias, guard_cards=guard, dt=DECIDE)
        raise ValueError(f"RSAgent.decide_mode: {rs_decide.MODE_LEGACY!r} or "
                         f"{rs_decide.MODE_LIVE!r}, got {self.decide_mode!r}")

    def _spell_aim(self, view, team, sub):
        """rs_decide's aim hook for --spell-aim: aim_spell on a spell's tile order."""
        def aim(order, slot, card):
            if self.cat_kind(card) == "SPELL":
                return self.aim_spell(order, view, team, card, sub)
            return order
        return aim

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        self._team = team
        # E17 (--noise): first, the state this seat's screen reader reports, so the look-ahead,
        # the row and rs_decide's masks (the HUD's elixir) all read it; the row at the tick its
        # clock reads. Without noise, the state and tick as before.
        noise = getattr(self, "noise", None)
        shift = 0
        if noise is not None:
            state = noise.observe(state, team)
            shift = noise.tick_shift
        if getattr(self, "hold_pending", False) and battle.pending_commands(team):
            # Our own last tap has not landed yet. Deciding again now means
            # deciding on a board that does not show it - the double-play the
            # delay produces. We always know what we sent, so this is public.
            self.ahead(battle, state)        # keep the velocity memory current
            return None
        view = self.ahead(battle, state)
        row = RR.obs_row(view, team, sub, mem, tick + self.lead + shift, live=self.live,
                         visibility=self.visibility,
                         **self.v2_kw(state, team, mem, tick + self.lead, royalesim))
        aim = self._spell_aim(view, team, sub) if getattr(self, "spell_aim", False) else None
        # Masks, wait cadence, slot and tile: tools/rs_decide.decide, on the present state
        # (hand, elixir, our pending plays), legality from check_deploy on this battle.
        d = rs_decide.decide(self.policy, row, state, team, rs_decide.BattleLegality(battle),
                             self.decide_config(), rng=self.rng, tick=tick, sub=sub, aim=aim)
        self.last_decision = d
        return d.command


def checkpoint_v2(meta) -> bool:
    """Whether a checkpoint meta is a v2 model's (E15): its "schema" is rs_rows/v2. Absent (every
    checkpoint before E14), rs_rows/legacy or any other name: legacy, as
    sim/model_tokens.model_schema reads it."""
    name = (meta or {}).get("schema")
    return isinstance(name, str) and name.strip() in (rs_schema.SCHEMA_V2, "v2")


_SPAWN = {}
_UNITS = []


def _unit_names():
    """Unit names in unit_type order, under the process's rs_engine.active() config."""
    if not _UNITS:
        _UNITS.extend(rs_engine.unit_types(rs_engine.active().undelayed()))
    return _UNITS


#: The card -> unit-type table tools/rs_spawn_table.py writes.
SPAWN_TABLE = ROOT / "data" / "rs_rows" / "spawn_primary.json"


def _spawn_tables():
    """card id -> the unit type a spawn of it makes, and the inverse (a unit type several
    cards make maps to the lowest card id). Read from SPAWN_TABLE, which is regenerated
    first when it is missing or was made under another engine than the process's
    rs_engine.active() config (rs_spawn_table.ensure compares the fingerprints)."""
    if not _SPAWN:
        import rs_spawn_table
        table = rs_spawn_table.ensure(SPAWN_TABLE, rs_engine.active())
        prim = rs_spawn_table.primary_of(table)
        _SPAWN.update(prim=prim, by_unit=rs_spawn_table.by_unit(prim))
    return _SPAWN["prim"], _SPAWN["by_unit"]


AGENT_CLASS = RSAgent   # tools/rs_selfplay.py swaps in its sampling agent


class RSSearchAgent(RSAgent):
    """RSAgent whose chosen play is checked in the engine first.

    When the gate decides to play, the model's top `k_cards` cards x `k_tiles`
    legal tiles are each played in a restored copy of the battle (RoyaleSim
    `save` / `load`) and run `horizon` ticks forward with nobody else playing.
    Score = own-frame tower swing over the horizon + the value head's 10 s
    prediction at the horizon + `prior` * model log-probability. The opponent
    makes no plays inside the copies, so its hidden hand is never consulted.
    """

    def __init__(self, ckpt, seed=0, k_cards=2, k_tiles=3, horizon=60, prior=0.05,
                 rebuild=False):
        super().__init__(ckpt, seed)
        self.k_cards, self.k_tiles, self.horizon, self.prior = k_cards, k_tiles, horizon, prior
        self.rebuild = rebuild

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        row = RR.obs_row(state, team, sub, mem, tick, live=self.live, visibility=self.visibility,
                         **self.v2_kw(state, team, mem, tick, royalesim))
        b = self.batch(row)
        with self.torch.no_grad():
            gate, _, _ = self.model(b, card_ids=b["hand"][:, 0].clamp(min=0))
        g = gate[0].float().numpy()
        if self.v2:
            g = g[:rs_decide.GATE_PLAYS]        # E15: the search plays cards, it never presses
        g[0] += self.wait_bias
        p = state["players"][team]
        for slot in range(4):
            cost = p["hand_costs"][slot] if slot < len(p["hand_costs"]) else 99
            if slot >= len(p["hand"]) or p["elixir_milli"] < cost * 1000:
                g[1 + slot] = -1e9
        pr = np.exp(g - g.max())
        pr /= pr.sum()
        if self.rng.random() < float(pr[0]) ** (DECIDE / 20) or g[1:].max() < -1e8:
            return None
        slots = [s for s in np.argsort(-g[1:]) if g[1 + s] > -1e8][:self.k_cards]
        cands = []
        for slot in slots:
            with self.torch.no_grad():
                _, tile, _ = self.model(b, card_ids=b["hand"][:, slot])
            t = tile[0].float().numpy()
            logz = np.logaddexp.reduce(t)
            n = 0
            for k in np.argsort(-t)[:40]:
                tx, ty = int(k % 18), int(k // 18)
                wx, wy = to_world(tx * 1000 + 500, ty * 1000 + 500, team)
                if battle.check_deploy(team, int(slot), wx * sub, wy * sub) == 0:
                    cands.append(((team, int(slot), wx * sub, wy * sub),
                                  float(np.log(pr[1 + slot]) + t[k] - logz)))
                    n += 1
                    if n >= self.k_tiles:
                        break
        if not cands:
            return None
        if getattr(self, "shadow", None) is not None:
            sim = self.shadow.battle(state["tick"])
        else:
            sim = self.rebuilt(state, team, royalesim) if self.rebuild else battle
            if (sim is not None and self.rebuild and getattr(self, "lead_rebuild", False)
                    and self.lead):
                # Delay-aware search: a play sent now lands `lead` ticks later,
                # so candidates are tried on the board at that moment - with
                # our own plays already in flight landed first - rather than
                # on the present one. (Stacked naively, search scored the
                # present board and fought the look-ahead: 41.7% vs 47.0%.)
                now = state["tick"]
                pend = [p for p in getattr(self, "sent", []) if p[0] > now]
                sim = self.rebuilt(state, team, royalesim, first=[c for (_, c, _, _) in pend])
                if sim is not None:
                    t = now
                    for (te, card, x, y) in sorted(pend):
                        if te >= now + self.lead:
                            break
                        if te > t:
                            sim.step([], te - t)
                            t = te
                        hand = json.loads(sim.state_json())["players"][team]["hand"]
                        if card in hand:
                            sim.step([(team, hand.index(card), x, y)], 1)
                            t += 1
                    if now + self.lead > t:
                        sim.step([], now + self.lead - t)
        if sim is None:
            return cands[0][0]
        saved = sim.save()
        before = (RR.towers_vec(json.loads(sim.state_json()), team)
                  if (self.rebuild or getattr(self, "shadow", None) is not None)
                  else RR.towers_vec(state, team))
        best, best_score = None, -1e18
        for cmd, logp in cands:
            sim.load(saved)
            if (self.rebuild or getattr(self, "shadow", None) is not None):
                # The shadow's hand order is its own; play by card, not slot.
                st = json.loads(sim.state_json())
                card = state["players"][team]["hand"][cmd[1]]
                sh = st["players"][team]["hand"]
                if card not in sh:
                    continue
                cmd_sim = (cmd[0], sh.index(card), cmd[2], cmd[3])
                if sim.check_deploy(*cmd_sim) != 0:
                    continue
                sim.step([cmd_sim], self.horizon)
                after_state = json.loads(sim.state_json())
                after = RR.towers_vec(after_state, team)
                swing = float((before[3:] - after[3:]).sum() - (before[:3] - after[:3]).sum())
                arow = RR.obs_row(after_state, team, sub, mem, tick + self.horizon, live=self.live,
                                  visibility=self.visibility,
                                  **self.v2_kw(after_state, team, mem, tick + self.horizon,
                                               royalesim))
                with self.torch.no_grad():
                    _, _, v = self.model(self.batch(arow), card_ids=self.batch(arow)["hand"][:, 0].clamp(min=0))
                score = swing + float(v[0, 0]) + self.prior * logp
                if score > best_score:
                    best, best_score = cmd, score
                continue
            sim.step([cmd], self.horizon)
            after_state = json.loads(sim.state_json())
            after = RR.towers_vec(after_state, team)
            swing = float((before[3:] - after[3:]).sum() - (before[:3] - after[:3]).sum())
            arow = RR.obs_row(after_state, team, sub, mem, tick + self.horizon, live=self.live,
                              visibility=self.visibility,
                              **self.v2_kw(after_state, team, mem, tick + self.horizon,
                                           royalesim))
            with self.torch.no_grad():
                _, _, v = self.model(self.batch(arow), card_ids=self.batch(arow)["hand"][:, 0].clamp(min=0))
            score = swing + float(v[0, 0]) + self.prior * logp
            if score > best_score:
                best, best_score = cmd, score
        if not self.rebuild and getattr(self, "shadow", None) is None:
            battle.load(saved)
        else:
            sim.load(saved)
        return best if best is not None else cands[0][0]




def capped_workers(requested, *, available=None, cpus=None, search=False) -> tuple[int, str]:
    """(workers, why): `requested` capped as tools/rs_league_eval.py worker_cap caps the league's
    workers (the free memory less its reserve, at WORKER_MB a worker, and the CPUs less one; at
    least 1), but not at its MAX_WORKERS: the count asked for is the most (module doc, Workers).
    `search`: the agent keeps a second battle besides the policies (--search's copies or rebuild,
    --shadow's re-drive), so a worker counts SEARCH_WORKER_MB, as a league worker with a
    searching player does. `available` (MB) and `cpus` default to the machine's. rs_league_eval
    imports numpy and modules this file imports already; no engine, no torch."""
    import rs_league_eval as LE
    want = max(1, int(requested))
    per = LE.SEARCH_WORKER_MB if search else LE.WORKER_MB
    return LE.worker_cap(want, available=available, cpus=cpus, per_worker_mb=per,
                         max_workers=want)


def check_noise_seats(noise, opp_noise, mode_, opp_ckpt="", shadow=False) -> None:
    """ValueError when a perception-noise profile (E17; None: none) goes to a seat that is not
    there or does not decide through RSAgent.act (module doc, "Perception noise"): `noise`, the
    agent seat's, needs mode "agent" (--pro has no agent seat; the --search agents have an act()
    of their own) and no shadow (rs_shadow re-drives the engine's true state); `opp_noise` needs
    a reactive opponent, `opp_ckpt` (the ghost replays commands and sees nothing)."""
    if noise is not None:
        if mode_ == "pro":
            raise ValueError("--noise is the agent seat's perception; --pro has no agent seat")
        if mode_ != "agent":
            raise ValueError(f"--noise: the {mode_} agent decides in an act() of its own, which "
                             "has no perception-noise hook; leave out --search")
        if shadow:
            raise ValueError("--noise: --shadow re-drives the engine's true state; leave out one "
                             "or the other")
    if opp_noise is not None and not opp_ckpt:
        raise ValueError("--opponent-noise is the reactive opponent's perception: give "
                         "--opponent-ckpt (the ghost replays its commands and sees nothing)")


def skip_record(tag, mode_, why, detail=""):
    """What play() returns for a match it does not play: {"tag", "mode", "skipped": why,
    "detail"}. It has no "result" or "records", so callers that keep only played
    matches (split_results here, rs_selfplay's `out.get("records")`) drop it."""
    return {"tag": tag, "mode": mode_, "skipped": why, "detail": detail}


def split_results(results):
    """play() results -> (played matches, Counter of skip reasons). None (no deal order
    fits the plays) is dropped uncounted, as before."""
    played = [r for r in results if r and not r.get("skipped")]
    skipped = Counter(r["skipped"] for r in results if r and r.get("skipped"))
    return played, skipped


def play(args):
    """One match. args: (payload_text, ckpt, mode, seed[, opp_ckpt, delay, lead, hold,
    shadow, spell_aim, engine, mem_lag, decide, visibility, noise, opponent_noise,
    noise_seed]); `engine` (an rs_engine config's to_json()) becomes this process's
    rs_engine.active() config, else the active one is used. `mem_lag` as --mem-lag (missing or
    None: rs_memory.BOARD_LAG; module doc). `decide` as --decide (missing or None: legacy), for
    both RSAgent seats. `visibility` as --visibility (missing or None: each RSAgent seat keeps
    its checkpoint's; module doc); a seat that sees public checks royalesim.STATUS_BITS first
    (rs_rows.check_status_bits). `noise` / `opponent_noise` as --noise / --opponent-noise (a
    profile's to_dict(), a path, "default"; missing, None or "none": a perfect observer, as
    before) and `noise_seed` as --noise-seed (missing or None: 0); ValueError for noise on a
    seat that is not there or does not decide through RSAgent.act (module doc, "Perception
    noise"). A noisy seat's counts are in the result's "noise" / "opponent_noise"."""
    payload_text, ckpt, mode_, seed = args[:4]
    opp_ckpt = args[4] if len(args) > 4 else ""
    delay = args[5] if len(args) > 5 else 0
    decide_mode = args[12] if len(args) > 12 and args[12] is not None else rs_decide.MODE_LEGACY
    if decide_mode not in (rs_decide.MODE_LEGACY, rs_decide.MODE_LIVE):
        raise ValueError(f"decide: {rs_decide.MODE_LEGACY!r} or {rs_decide.MODE_LIVE!r}, "
                         f"got {decide_mode!r}")
    visibility = args[13] if len(args) > 13 else None
    if visibility is not None:
        RR.visibility_mode(visibility)
    # E17: each seat's perception noise profile (None: a perfect observer) and the run seed.
    noise = RR.noise_profile(args[14] if len(args) > 14 else None)
    opp_noise = RR.noise_profile(args[15] if len(args) > 15 else None)
    noise_seed = RR.noise_seed_value(args[16] if len(args) > 16 and args[16] is not None else 0)
    check_noise_seats(noise, opp_noise, mode_, opp_ckpt, args[8] if len(args) > 8 else False)
    mem_lag = RR.mem_lag_value(args[11] if len(args) > 11 and args[11] is not None
                               else rs_memory.BOARD_LAG)
    legacy = mem_lag == RR.MEM_LEGACY
    if len(args) > 10 and args[10] is not None:
        rs_engine.set_active(args[10])
    cfg = rs_engine.active()
    royalesim, names, ids, cat = RR._setup(cfg)
    sub = royalesim.SUBTILE_PER_MILLITILE
    p = json.loads(payload_text)
    battle = p["battle"]
    decks, forms, levels = [], [], []
    for side in ("team", "opponent"):
        deck, form, level = [], [], []
        for card in battle[side]["players"][0]["deck"]:
            key, f = base_key(card["card_key"])
            deck.append(names[key])
            form.append(f)
            level.append(int(card.get("level") or 11))
        decks.append(deck)
        forms.append(form)
        levels.append(level)
    events = sorted((e for e in p["events"] if e["kind"] == "play_card"),
                    key=lambda e: (e["replay_tick_20hz"], e["source_index"]))
    plays = {0: [], 1: []}
    for e in events:
        seat = 0 if e["side"] == "team" else 1
        u = e["coordinates"]["native_world_units"]
        plays[seat].append((int(e["replay_tick_20hz"]), names[base_key(e["card_key"])[0]],
                            int(u["x"]), int(u["y"])))
    # The seat from troop plays; coordinates go to the engine as recorded (rs_orient).
    ori = rs_orient.orient(plays, rs_orient.troop_names(cat))
    if not ori.ok():
        return skip_record(p["source"]["replay_tag"], mode_, rs_orient.SKIP, ori.describe())
    team_seat = ori.team_seat
    rs = {0: team_seat, 1: 1 - team_seat}
    seat_of = {rs[0]: 0, rs[1]: 1}
    hog_seat = 0 if frozenset(decks[0]) == RR.HOG26 else 1
    hog_team = rs[hog_seat]
    dealt = []
    for seat in (0, 1):
        order = deal_order(decks[seat], [c for _, c, _, _ in plays[seat]])
        if order is None:
            return None
        dealt.append(order)
    perm = [[decks[s].index(c) for c in dealt[s]] for s in (0, 1)]
    tower = max(1, min(16, mode([lv for s in levels for lv in s])))
    # Real taps land 21-22 ticks after they are sent (RoyaleSim's measured
    # client delay). Both seats get it; the recorded ghost commands were
    # logged at the tick they executed, so they are sent `delay` early.
    # It must be set BEFORE reset: it is battle configuration, and set
    # after reset it silently does nothing (found when a delayed run came
    # back identical, match for match, to an undelayed one). make_battle
    # sets it before returning; a legacy config with delay 0 sets nothing.
    b = rs_engine.make_battle(cfg.delayed(delay), royalesim)
    b.reset(seed=seed, decks=[[ids[c] for c in dealt[seat_of[t]]] for t in (0, 1)],
            shuffle=0, start_tick=0, elixir_milli=[6000, 6000], tower_hp=None, spawns=[],
            forms=[[forms[seat_of[t]][i] for i in perm[seat_of[t]]] for t in (0, 1)],
            levels=[[levels[seat_of[t]][i] for i in perm[seat_of[t]]] for t in (0, 1)],
            tower_levels=[tower, tower])
    lead = args[6] if len(args) > 6 else 0
    hold = args[7] if len(args) > 7 else False
    spell_aim = args[9] if len(args) > 9 else False
    agent = (None if mode_ == "pro" else
             RSSearchAgent(ckpt, seed=seed) if mode_ == "search" else
             RSSearchAgent(ckpt, seed=seed, rebuild=True) if mode_ == "search_rebuild" else
             AGENT_CLASS(ckpt, seed=seed))
    use_shadow = args[8] if len(args) > 8 else False
    if agent is not None and use_shadow:
        from rs_shadow import Shadow
        prim, _ = _spawn_tables()
        me_seat = seat_of[hog_team]
        agent.shadow = Shadow(royalesim, cat, hog_team,
                              [ids[c] for c in dealt[me_seat]],
                              [levels[me_seat][i] for i in perm[me_seat]], tower, prim,
                              engine=cfg)
    if agent is not None:
        agent.engine = cfg
        agent.lead = abs(lead) % 1000
        agent.lead_rebuild = -1000 < lead < 0
        agent.lead_extrapolate = lead >= 1000
        agent.hold_pending = hold
        agent.spell_aim = spell_aim
        agent.decide_mode = decide_mode
    ghost_team = 1 - hog_team
    opponent = RSAgent(opp_ckpt, seed=seed + 7) if opp_ckpt else None
    if opponent is not None:
        opponent.engine = cfg
        opponent.decide_mode = decide_mode
    champs = None
    for seat in (agent, opponent):
        if seat is None:
            continue
        if visibility is not None:
            seat.visibility = visibility
        if seat_visibility(seat) == RR.VIS_PUBLIC or getattr(seat, "v2", False):
            RR.check_status_bits(royalesim)       # a v2 row reads the burrow bit too
        if getattr(seat, "v2", False):
            # E15: the seat's ability buttons, from the deck and forms reset() was given.
            if champs is None:
                champs = rs_presses.champion_ids(cat, getattr(royalesim, "CATALOGUE_FIELDS",
                                                              None))
            s = seat_of[hog_team if seat is agent else ghost_team]
            seat.buttons = rs_presses.side_buttons([ids[c] for c in dealt[s]],
                                                   [forms[s][i] for i in perm[s]], champs)
    # E17: the MatchNoise of the seat deciding for each team (None: it sees the engine's state,
    # and every play reaches the memory as it runs, as before).
    noises = {hog_team: None, ghost_team: None}
    tag = p["source"]["replay_tag"]
    for seat, team, prof, key in ((agent, hog_team, noise, tag),
                                  (opponent, ghost_team, opp_noise, f"{tag}|opponent")):
        if seat is not None and prof is not None:
            seat.noise = noises[team] = PN.MatchNoise(RR.noise_model(prof, b, sub),
                                                      PN.match_seed(key, noise_seed))
    watching = [nz for nz in noises.values() if nz is not None]
    scripted = [(t, rs[s], c, x, y) for s in (0, 1) for (t, c, x, y) in plays[s]
                if (mode_ == "pro" or rs[s] == ghost_team) and not
                (opponent is not None and rs[s] == ghost_team)]
    if delay:
        scripted = [(max(0, t - delay), team, c, x, y) for (t, team, c, x, y) in scripted]
    scripted.sort()
    # E08: a play sent on tick t runs on t + delay, and is recorded with that tick; every
    # decision of a tick reads the one view taken before anything is sent. Legacy dates a
    # play when it is sent and reads the memory at each decision (the dict before E08).
    mem = rs_memory.PublicMemory()

    def record(team, card, when):
        """`team`'s play of row token `card`, run on `when`, into the memory: as it runs, or
        (E17) as the other team's seat perceives it when that seat has noise (its MatchNoise
        may miss the play, date it late or not name it)."""
        nz = noises[1 - team]
        if nz is None:
            mem.record(team, card, when)
        else:
            nz.record(mem, team, card, when)

    runs_after = 0 if legacy else delay
    i, tick, n_agent = 0, 0, 0
    while tick < 6200:
        state = json.loads(b.state_json())
        if state["game_over"]:
            break
        for nz in watching:
            nz.push(state)            # E17: every state the match goes through (board lag)
        known = None if legacy else mem.view(tick, mem_lag)
        cmds = []
        while i < len(scripted) and scripted[i][0] <= tick:
            _, team, card, x, y = scripted[i]
            i += 1
            hand = state["players"][team]["hand"]
            cid = ids[card]
            if cid in hand:
                cmds.append((team, hand.index(cid), *rs_orient.to_engine(x, y, sub)))
                record(team, cid + 2, tick + runs_after)
        if agent is not None and getattr(agent, "shadow", None) is not None:
            agent.shadow.observe(state)
        if agent is not None and tick % DECIDE == 0 and tick >= 90:
            cmd = agent.act(b, state, hog_team, mem.view(tick, 0) if legacy else known, tick,
                            sub, royalesim)
            if cmd is not None:
                cmds.append(cmd)
                agent.sent = getattr(agent, "sent", []) + [
                    (tick + delay, state["players"][hog_team]["hand"][cmd[1]], cmd[2], cmd[3])]
                if getattr(agent, "shadow", None) is not None:
                    agent.shadow.own_play(tick + delay, state["players"][hog_team]["hand"][cmd[1]],
                                          cmd[2], cmd[3])
                card = state["players"][hog_team]["hand"][cmd[1]]
                record(hog_team, card + 2, tick + runs_after)
                n_agent += 1
        if opponent is not None and tick % DECIDE == 0 and tick >= 90:
            cmd = opponent.act(b, state, ghost_team, mem.view(tick, 0) if legacy else known,
                               tick, sub, royalesim)
            if cmd is not None:
                cmds.append(cmd)
                card = state["players"][ghost_team]["hand"][cmd[1]]
                record(ghost_team, card + 2, tick + runs_after)
        nxt = scripted[i][0] if i < len(scripted) else 6000
        step = max(1, min(nxt, (tick // DECIDE + 1) * DECIDE) - tick)
        b.step(cmds, step)
        tick += step
    state = json.loads(b.state_json())
    w = state["winner"]
    real = str(battle.get("result", "")).lower()
    real_hog = (("win" if real in ("victory", "win") else "loss" if real in ("defeat", "loss") else "draw")
                if hog_seat == 0 else
                ("loss" if real in ("victory", "win") else "win" if real in ("defeat", "loss") else "draw"))
    out = {"tag": p["source"]["replay_tag"], "mode": mode_, "real": real_hog,
           "game_over": state["game_over"], "end_tick": state["tick"],
           "result": "win" if w == hog_team else "loss" if w in (0, 1) else "draw",
           "crowns": [state["players"][hog_team]["crowns"], state["players"][1 - hog_team]["crowns"]],
           "agent_plays": n_agent,
           "records": getattr(agent, "records", None),
           "hog_team": hog_team}
    # E17: only for a noisy seat, so the result without noise is the dict of before.
    for key, team in (("noise", hog_team), ("opponent_noise", ghost_team)):
        if noises[team] is not None:
            out[key] = dict(sorted(noises[team].stats.items()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--matches", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6,
                    help="worker processes (default 6), capped by the free memory and the CPUs "
                         "as tools/rs_league_eval.py caps its own; the results do not depend on "
                         "the count")
    ap.add_argument("--pro", action="store_true", help="run the recorded-pro baseline instead")
    ap.add_argument("--search", action="store_true", help="decision-time search")
    ap.add_argument("--spell-aim", action="store_true",
                    help="aim spells at the most enemy hp among the model's top tiles")
    ap.add_argument("--shadow", action="store_true",
                    help="search / lead on a shadow re-drive of public plays (live-capable)")
    ap.add_argument("--hold-pending", action="store_true",
                    help="no new play while our own last play is still pending")
    ap.add_argument("--lead", type=int, default=0,
                    help="delay compensation: decide on the board this many ticks ahead "
                         "(negative = look ahead on a battle rebuilt from public observation; "
                         "1000 + n = extrapolate unit motion n ticks, no engine)")
    ap.add_argument("--delay", type=int, default=0,
                    help="command delay in ticks for both seats (real client: ~21)")
    ap.add_argument("--rebuild", action="store_true",
                    help="search on a battle rebuilt from public observation (the live case)")
    ap.add_argument("--opponent-ckpt", default="",
                    help="a reactive opponent: this checkpoint plays the recorded opponent's "
                         "deck instead of the ghost replaying its commands")
    ap.add_argument("--allow-vocab-drift", action="store_true",
                    help="load checkpoints whose card / unit vocabulary or card table is not "
                         "this engine's, with a warning (tools/rs_checkpoint.py)")
    RR.add_mem_lag_arg(ap, help_text=(
        "opponent memory: a seat deciding at tick t knows the plays that ran by t - TICKS "
        f"(default {rs_memory.BOARD_LAG}: what the board shows; a play runs --delay ticks after "
        "it is sent); 'legacy': dated when sent, as before E08"))
    ap.add_argument("--decide", choices=(rs_decide.MODE_LEGACY, rs_decide.MODE_LIVE),
                    default=rs_decide.MODE_LEGACY,
                    help="decision rule of the checkpoint seats (tools/rs_decide.py): 'legacy' "
                         "(default) as this tool always decided; 'live' the live runner's "
                         "(pending-slot mask, next card when no tile is legal, cap rule, lane "
                         "guard). --search keeps its own rule")
    ap.add_argument("--visibility", choices=RR.VISIBILITY_MODES, default=None,
                    help="which units the checkpoint seats see (tools/rs_rows.py --visibility); "
                         "default: each checkpoint's own (its vocab 'visibility'; legacy before "
                         "E12). 'legacy' every living unit; 'public' no invisible or hidden "
                         "enemy, in the rows, the rebuilt look-ahead and spell aim")
    PN.add_noise_args(ap)              # E17: --noise (the agent seat), --noise-only, --noise-seed
    ap.add_argument("--opponent-noise", default=None, metavar="PROFILE",
                    help="perception noise of the reactive opponent seat (--opponent-ckpt), as "
                         "--noise (default: none, it sees the engine's state; --noise-only does "
                         "not apply to it)")
    rs_engine.add_engine_args(ap)
    args = ap.parse_args()
    mode_ = ("pro" if args.pro else "search_rebuild" if args.search and args.rebuild
             else "search" if args.search else "agent")
    try:
        cfg = rs_engine.config_from_args(args)
        noise = PN.profile_from_args(args)                   # None without --noise
        opp_noise = RR.noise_profile(args.opponent_noise)
        check_noise_seats(noise, opp_noise, mode_, args.opponent_ckpt, args.shadow)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    rs_engine.set_active(cfg)
    import pyarrow.parquet as pq
    royalesim, names, ids, cat = RR._setup(cfg)
    vocab_battle = rs_engine.make_battle(cfg.undelayed(), royalesim)
    fp = rs_engine.fingerprint(vocab_battle, royalesim, cfg)
    print(f"engine: {cfg.delayed(args.delay).describe()}\n        {rs_engine.summary(fp)}",
          flush=True)
    for what, prof in (("noise", noise), ("opponent noise", opp_noise)):
        if prof is None:
            continue
        try:        # bound once here, so a profile this engine cannot bind stops the run now
            model = RR.noise_model(prof, vocab_battle, royalesim.SUBTILE_PER_MILLITILE)
        except (ValueError, FileNotFoundError) as exc:
            ap.error(f"--{what.replace(' ', '-')}: {exc}")
        print(f"{what}: {RR.describe_noise(prof, args.noise_seed)}", flush=True)
        for name in model.unresolved:
            print(f"        not bound: {name}", flush=True)
    import rs_checkpoint
    if args.allow_vocab_drift:
        rs_checkpoint.set_allow_drift(True)          # the workers inherit it
    for ck in ([] if args.pro else [args.ckpt]) + [args.opponent_ckpt]:
        if ck:          # once here, rather than a traceback from every worker
            try:
                rs_checkpoint.check_file(ck, cfg)
            except rs_checkpoint.VocabMismatch as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
    if args.shadow or args.rebuild or -1000 < args.lead < 0:
        _spawn_tables()             # checked (or regenerated) once, before the workers read it
    picked = []
    for part in sorted((CORPUS / "replays").glob("part-*.parquet")):
        for r in pq.read_table(part, columns=["replay_tag", "game_mode", "payload_json"]).to_pylist():
            if r["game_mode"] not in RR.MODES or RR.split_of(r["replay_tag"]) != "test":
                continue
            p = json.loads(r["payload_json"])
            try:
                sides = [frozenset(names.get(base_key(c["card_key"])[0])
                                   for c in p["battle"][s]["players"][0]["deck"]) for s in ("team", "opponent")]
            except (KeyError, IndexError):
                continue
            if any(None in s for s in sides) or RR.HOG26 not in sides or sides[0] == sides[1]:
                continue
            p.setdefault("source", {})["replay_tag"] = r["replay_tag"]
            picked.append(json.dumps(p))
            if len(picked) >= args.matches:
                break
        if len(picked) >= args.matches:
            break
    # E17: the noise slots only when a seat has noise, so the tasks are otherwise as before.
    extra = ()
    if noise is not None or opp_noise is not None:
        extra = (None if noise is None else noise.to_dict(),
                 None if opp_noise is None else opp_noise.to_dict(), args.noise_seed)
    # Capped by free memory and CPUs (module doc, Workers): match k is seeded k whatever the count.
    # A search or a shadow keeps a second battle in each worker: the league's search memory.
    second_battle = mode_ in ("search", "search_rebuild") or bool(args.shadow)
    workers, why = capped_workers(args.workers, search=second_battle)
    print(f"workers: {why}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=rs_engine.set_active,
                             initargs=(cfg.to_json(),)) as pool:
        rows, skipped = split_results(list(pool.map(
            play, [(t, args.ckpt, mode_, k, args.opponent_ckpt, args.delay, args.lead,
                    args.hold_pending, args.shadow, args.spell_aim, cfg.to_json(), args.mem_lag,
                    args.decide, args.visibility) + extra
                   for k, t in enumerate(picked)])))
    if skipped:
        print(f"  skipped matches: {dict(skipped)} (orient = a side's troop plays are off "
              "its own half, tools/rs_orient.py)", flush=True)
    wins = sum(r["result"] == "win" for r in rows)
    draws = sum(r["result"] == "draw" for r in rows)
    lo, hi = wilson(wins, len(rows))
    cd = sum(r["crowns"][0] - r["crowns"][1] for r in rows) / max(1, len(rows))
    name = ("recorded pro" if args.pro else
            Path(args.ckpt).parent.name + (" +search" if args.search else "")
            + (" (rebuilt)" if args.rebuild else "")
            + (f" delay {args.delay}" if args.delay else "")
            + (f" lead {args.lead}" if args.lead else "")
            + (" hold" if args.hold_pending else "")
            + (" shadow" if args.shadow else "")
            + (" spell-aim" if args.spell_aim else "")
            + (f" mem-lag {args.mem_lag}" if args.mem_lag != rs_memory.BOARD_LAG else "")
            + (f" decide {args.decide}" if args.decide != rs_decide.MODE_LEGACY else "")
            + (f" visibility {args.visibility}" if args.visibility is not None else "")
            + (f" noise {noise.name}" if noise is not None else "")
            + (f" opponent-noise {opp_noise.name}" if opp_noise is not None else "")
            + (f" noise-seed {args.noise_seed}" if extra and args.noise_seed else "")
            + (f" vs reactive {Path(args.opponent_ckpt).parent.name}" if args.opponent_ckpt else " vs ghost"))
    real_w = sum(r["real"] == "win" for r in rows)
    print(f"  real result of the Hog player in these matches: {real_w}/{len(rows)} "
          f"({real_w / max(1, len(rows)):.1%}); unfinished in engine: "
          f"{sum(not r['game_over'] for r in rows)}")
    print(f"{name}: {wins}/{len(rows)} wins ({wins / max(1, len(rows)):.1%}, 95% [{lo:.1%}, {hi:.1%}]) "
          f"{draws} draws, crown diff {cd:+.2f}, agent plays/match "
          f"{sum(r['agent_plays'] for r in rows) / max(1, len(rows)):.1f} ({time.time() - t0:.0f}s)")
    out = ROOT / "reports" / "rs_ghost.jsonl"
    record = {"name": name, "ckpt": args.ckpt, "wins": wins, "n": len(rows),
              "ci": [lo, hi], "crown_diff": cd,
              "engine": cfg.delayed(args.delay).to_json(), "engine_id": fp["id"],
              "mem_lag": args.mem_lag, "rows": rows}
    if skipped:                       # only then, so a run with none writes what it did
        record["skipped"] = dict(skipped)
    if args.decide != rs_decide.MODE_LEGACY:     # likewise: a legacy run writes what it did
        record["decide"] = args.decide
    if args.visibility is not None:              # likewise: only when given
        record["visibility"] = args.visibility
    for key, prof in (("noise", noise), ("opponent_noise", opp_noise)):
        if prof is None:                         # E17: likewise, only for a noisy seat
            continue
        stats: dict = {}
        for r in rows:
            RR.merge_stats(stats, r.get(key) or {})
        record[key] = PN.vocab_entry(prof, args.noise_seed)
        record[f"{key}_stats"] = dict(sorted(stats.items()))
        print(f"  {key.replace('_', ' ')}: "
              + (", ".join(f"{k} {v:,}" for k, v in sorted(stats.items())) or "nothing observed"))
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
