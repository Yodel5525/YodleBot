r"""General match runner in RoyaleSim (L02): any two decks at friendly levels, two seats.

    import rs_match as RM                        # tools/ on sys.path, as the tools import
    a = RM.AgentSpec("runs/tokens/live_big_hog/best.pt", name="cand", decide="live", noise="default")
    b = RM.AgentSpec("runs/tokens/live_big/ep1.pt", name="ref")
    pairs = RM.load_pairs(split="dev", require=RM.HOG26).sample(500, seed=0)   # IL_Replay deck pairs
    specs = RM.pairing_specs(pairs.pairs, (a, b), seed=0, delay=24, base_forms=(True, False))
    results = RM.run_many(specs, workers=4)
    print(RM.summarize(results, "cand"))

    .venvs\royalegym\Scripts\python.exe tools\rs_match.py --a CKPT_A --b CKPT_B --split dev \
        --pairs 200 --require hog26 --delay 24 [--noise-a default] [--workers 4]

Why (tokens-09, tokens-15): tools/rs_eval.py plays only Hog 2.6 test matches against one fixed
reference (the recorded opponent's deck, rs_eval.py play()), which gives neither the 1,000-pair
evaluations the promotion gate (L03) needs nor multi-deck work. This runner plays any MatchSpec:
two decks with forms and levels, one agent per seat, the command delay, the opponent memory, the
perception noise of each seat, a seed and the engine config. Nothing here touches the emulator.

THE SPEC (MatchSpec). decks: two lists of RoyaleSim card names, team 0 (blue) then team 1 (red),
each in DEAL ORDER (the first four in hand, then the queue; reset with shuffle 0, so a spec says
exactly what is dealt). forms: 0 base, 1 evolution, 2 hero, parallel to the decks (None: all
base). levels: one level for every card (default FRIENDLY_LEVEL, 11: a friendly battle plays every
card and tower at 11, as the owner's frames show on both sides) or a list per deck; tower_levels
likewise. delay: the command delay in ticks, an int for both seats or (blue, red); None keeps the
engine config's. agents: one AgentSpec per team. mem_lag: rs_rows --mem-lag (default BOARD_LAG).
seed: the engine's reset seed; each policy seat's rng is seeded with seed + its seed_offset.
first: the team that decides first within a tick. tag / pairing / seat: where the spec came from
(a deck pair's replay tag, the id both seat-swapped games share, 0 as drawn and 1 swapped).
swapped() mirrors the assignment: decks, forms, levels, agents, per-seat delays, starting elixir
and `first` change sides, so each agent keeps its deck, its seed and its noise on the other seat.

THE AGENTS (AgentSpec). ckpt: a token-policy checkpoint, played by MatchAgent, rs_eval.RSAgent
with a sampling preset added; "" is an idle seat that never plays (a floor anchor). schema: the
row layout the checkpoint must have been trained on, rs_rows/legacy or rs_rows/v2 (ROW_SCHEMAS;
the short names "legacy" and "v2" are taken too); None takes the checkpoint's vocab "schema"
(absent: legacy). The seat builds the rows of its checkpoint's schema (V2 SEATS below).
lead: rs_eval --lead (0 none, n engine copy, -n rebuilt from public information, 1000 + n dead
reckoning); a decision-time checkpoint (E11, vocab obs_delay.ticks > 0) wants 0. noise: a
perception-noise profile (tools/perception_noise.py: a NoiseProfile, its JSON dict, a path,
"default"; None or "none" is a perfect observer). decide: rs_decide's preset, "live" (the live
runner's rule, the default here), "legacy" (rs_eval's) or "sample" (RL's sampled slot and tile).
visibility: rs_rows --visibility (None: the checkpoint's own). presses: whether the seat may
press its ability buttons (V2 SEATS; default False, every seat as before).

V2 SEATS (E14 / E15; tools/rs_schema.py). A checkpoint whose meta "schema" is rs_rows/v2 gets
rs_eval.RSAgent's v2 rows (RSAgent.v2_kw) with what the live runner gives its rows:
  buttons     its ability button cards, named from the deck and forms the spec deals its team
              (rs_presses.side_buttons: hero entries in deal order, then champions), as
              rs_eval.play names them; run_match sets them on every seat whose `v2` is true,
              and such a seat checks royalesim.STATUS_BITS first (its rows read the burrow bit).
  own_pending its own plays sent and not run yet (the state's pending rows, card + 2 and ticks
              to land; RSAgent.own_pending), except for a checkpoint whose rows were built at
              landing time (meta obs_delay ticks 0): those rows never showed a pending play, and
              rs_decide.pending_inputs gives the live runner none for it either.
  opp_elixir  the opponent's bar as the seat counts it from its memory view (rs_rows.SeenElixir,
              lag 0), paid only for the plays the engine RAN. A play the engine refused when it
              was sent (its step result) or when it ran under the delay
              (Battle.step_commands_run) stays in the memory, as it always did (opp_seen and
              opp_last show it), but run_match hands each seat that reads them (MatchAgent.unran)
              the entries of its memory whose play never ran, and MatchAgent.opp_elixir leaves
              them out. With mem_lag >= 1 (the default) a play is known to have run or not before
              any view holds it; with lag 0 or "legacy" the count is redone from the view once a
              refusal is known (RSAgent.opp_elixir starts again when a counted play leaves the
              view). The opponent's presses are not counted: the live screen reader does not see
              them either.
  presses     AgentSpec.presses (default False: every button masked, as before) lets rs_decide
              press the seat's buttons when its checkpoint's buttons trained
              (rs_decide.presses_allowed(meta): v2, train_rows press_rows "trained"), in every
              decide preset. A press command, (team, HAND_SIZE + button, 0, 0), is taken only
              from a seat whose spec has presses on (else it is refused as a slot outside the
              hand, ValueError). It is counted in presses / press_refused (at send, and when it
              ran under a delay), logged with its button's card (the abilities row's, -1
              unknown) and its slot, never recorded in a memory (it is no card play), and when
              it costs elixir it ends a full bar's leak as a play does. AgentSpec.to_dict leaves
              presses out while it is False, so the specs, game keys and player identities of
              earlier runs (tools/rs_league_eval.py) do not change.
A legacy checkpoint's seat (and any seat whose `v2` is not true) is played exactly as before.

THE LOOP (run_match), rs_eval.play's loop with two policy seats and no ghost: make_battle under
the spec's engine config with its delay, reset with the spec's decks, forms, levels and 6 elixir
each; every `cadence` (10) ticks from tick `first_decision` (90) each seat, `first` then the
other, sees its view and may send one command; step to the next decision tick (single ticks past
END_TICK, as rs_eval.play does there; the engine is over by then), until game over or max_ticks
(6200). Decisions go through RSAgent.act, i.e. tools/rs_decide.decide with check_deploy on the
match battle as the legality. A seat's command must be its own team's, (team, slot, x_sub, y_sub).

MEMORY (E08). Each seat has its own rs_memory.PublicMemory: what that seat knows. A play sent on t
by team k is dated t + delay[k], the tick it runs, and recorded into its own seat's memory as is
and into the other seat's through that seat's noise (missed, late, unnamed; no noise: as is).
Without noise the two memories are equal, so this is rs_eval's single memory. Each seat decides on
one view per tick, taken before anything is sent, holding the plays with exec + mem_lag <= tick.
mem_lag "legacy" dates plays when sent and reads the memory at each decision (rs_eval's legacy).

NOISE (E17). A seat with a noise profile decides on MatchNoise.observe(state, team), the state its
screen reader would report, and obs_row's tick moves by the clock noise (tick_shift); its rng
seed is perception_noise.match_seed of the pairing, seed, team and agent label plus the agent's
noise_seed. States are pushed into its history every loop tick; board lag needs a state every
tick, so a lag profile is approximate here (MatchNoise counts board_lag_unavailable).

THE RESULT (MatchResult): winner (team, -1 a draw), crowns, game_over, end_tick, plays sent and
refused (at send, and when they ran under a delay where the engine reports it), decisions,
elixir leak, final tower hp, the delays, per-seat agent info and noise counts, the play log
([tick, team, card, slot, x_sub, y_sub, reason at send]; a press has slot HAND_SIZE + button),
timings (wall seconds: setup, engine, each seat's decisions; never compare them) and the
presses sent and refused (V2 SEATS). score(team) is
1 / 0.5 / 0; score_of(name) is that agent's, whichever seat it sat in; summarize() gives an
agent's score, Wilson interval (tools/league/stats.py) and means.

ELIXIR LEAK. Over each step, a seat whose bar is full (elixir_milli at the cap) at the step's start
wastes the regeneration until the first of its plays that runs inside the step (or the whole
step): full_ticks counts those ticks and leak the elixir they would have given
(rs_elixir.ElixirLaw.regen: 1x, 2x from 2400, 3x from 4800). A bar that fills inside a step is
counted from the next step: at most one step (10 ticks, 0.18 elixir at 1x) per fill is missed.

DECK PAIRS. read_pairs / load_pairs take IL_Replay replays rows (deck_stats.iter_rows) of ONE
split, by tools/rs_rows.split_of of the replay tag: "train", "dev" (= val) or "final" (= test, only
with allow_final=True: the held-out split L03 reports with --final). A pair is the two recorded
decks (sorted base keys, deck_stats.deck_of, as RoyaleSim names) with their recorded forms, of a
1v1 match in the ranked modes rs_rows trains on. `require` (e.g. HOG26) keeps the matches where
one side plays that deck and puts it on side a; mirror matches are dropped (as rs_eval does).
Levels are not kept: friendlies are level 11. PairSet.sample(n, seed) draws n pairs; save / load
keep a pairing list so every candidate meets the same pairings. pairing_specs deals each pair
(a permutation per deck from a seed of (seed, tag)), puts agent a on side a's deck at team 0 and
adds the seat-swapped game: the pairing's two games share pairing id, seed and deals.

THE LEGACY SCREEN (play_legacy_screen) is rs_eval.play's reactive screen kept as a preset: the same
args tuple, a recorded Hog 2.6 match's dealt decks, recorded forms and levels, the tower level at
the most common card level, the agent (rs_eval.AGENT_CLASS) on the Hog seat deciding first, the
reactive opponent (rs_eval.RSAgent, seed + 7) on the recorded opponent's deck, rs_eval's legacy
decision rule unless --decide says otherwise; it returns rs_eval.play's dict (plus "match", the
MatchResult). tests/pipeline/test_rs_match.py checks it against rs_eval.play call for call. The
ghost and pro screens stay in tools/rs_eval.py.

Imports tools/rs_eval.py and tools/rs_rows.py read-only; never imports scripts/gamestate.
"""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mode
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import deck_stats  # noqa: E402
import perception_noise as PN  # noqa: E402
import rs_cards  # noqa: E402
import rs_decide  # noqa: E402
import rs_elixir  # noqa: E402
import rs_engine  # noqa: E402
import rs_eval  # noqa: E402  (read-only: RSAgent, AGENT_CLASS, seat_visibility, skip_record)
import rs_memory  # noqa: E402
import rs_orient  # noqa: E402
import rs_presses  # noqa: E402  (V2 SEATS: side_buttons, champion_ids, button_count, hand_size)
import rs_rows as RR  # noqa: E402  (read-only: _setup, obs_row through RSAgent, HOG26, settings)
import rs_schema  # noqa: E402  (V2 SEATS: the schema names)
from royalesim_redrive import base_key, deal_order  # noqa: E402
from tools.league.stats import score_interval  # noqa: E402

#: Friendly battles play every card and crown tower at level 11 (owner's frames, both sides).
FRIENDLY_LEVEL = 11
START_ELIXIR = 6000              # milli, both seats (rs_eval.play, rs_rows)
DECIDE = rs_eval.DECIDE          # ticks between decisions (0.5 s)
FIRST_DECISION = 90              # rs_eval.play: no decision before tick 90 (the deploy lockout)
END_TICK = 6000                  # regulation + overtime; rs_eval.play steps single ticks past it
MAX_TICKS = 6200                 # rs_eval.play's loop bound
LEGACY_OPPONENT_OFFSET = 7       # rs_eval.play seeds the reactive opponent with seed + 7
HOG26 = RR.HOG26
#: Split names of the evaluation protocol -> rs_rows.split_of's.
SPLIT_ALIASES = {"dev": "val", "final": "test"}
FINAL_SPLIT = "test"
IDLE = "idle"


def _int(value: Any, what: str, lo: int | None = None) -> int:
    """`value` as a plain int (numpy ints too); bools, other types and values below `lo` refused."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{what}: an int, got {value!r}")
    value = int(value)
    if lo is not None and value < lo:
        raise ValueError(f"{what}: at least {lo}, got {value}")
    return value


def _two(value: Any, what: str) -> tuple:
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError(f"{what}: two items (team 0, team 1), got {value!r}") from None
    if len(items) != 2:
        raise ValueError(f"{what}: two items (team 0, team 1), got {len(items)}")
    return items


def _flip(value: Any) -> Any:
    return None if value is None else (value[1], value[0])


def _is_noise_model(noise: Any) -> bool:
    return all(hasattr(noise, a) for a in ("apply", "play_seen", "profile"))


def _is_noise_profile(noise: Any) -> bool:
    return all(hasattr(noise, a) for a in ("active_groups", "to_dict", "digest"))


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AgentSpec:
    """One seat's agent (module doc, THE AGENTS)."""

    ckpt: str = ""
    name: str = ""
    schema: str | None = None
    lead: int = 0
    noise: Any = None
    noise_seed: int = 0
    decide: str = rs_decide.MODE_LIVE
    visibility: str | None = None
    wait_keep: float | None = None
    spell_aim: bool = False
    hold_pending: bool = False
    seed_offset: int = 0
    allow_drift: bool | None = None
    presses: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ckpt", "" if self.ckpt is None else str(self.ckpt))
        object.__setattr__(self, "name", str(self.name or ""))
        if self.decide not in rs_decide.MODES:
            raise ValueError(f"decide: one of {', '.join(rs_decide.MODES)}, got {self.decide!r}")
        if self.visibility is not None:
            RR.visibility_mode(self.visibility)
        if self.schema is not None:
            text = str(self.schema).strip()
            object.__setattr__(self, "schema", rs_schema.SHORT_NAMES.get(text, text))
        for what in ("lead", "noise_seed", "seed_offset"):
            object.__setattr__(self, what, _int(getattr(self, what), what))
        if not isinstance(self.presses, (bool, np.bool_)):
            raise ValueError(f"presses: True or False, got {self.presses!r}")
        object.__setattr__(self, "presses", bool(self.presses))
        if _is_noise_profile(self.noise) and not _is_noise_model(self.noise):
            object.__setattr__(self, "noise", self.noise.to_dict())     # plain data: picklable

    @property
    def idle(self) -> bool:
        return not self.ckpt or self.ckpt.strip().lower() == IDLE

    @property
    def label(self) -> str:
        """The name results use: `name`, else the checkpoint's run folder (or file), else idle."""
        if self.name:
            return self.name
        if self.idle:
            return IDLE
        return Path(self.ckpt).parent.name or self.ckpt

    def to_dict(self) -> dict:
        """The fields as a dict; `presses` only when True (V2 SEATS: the dict, and so the game
        keys and identities built from it, is what it was before the switch while it is off)."""
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        if _is_noise_model(self.noise):
            out["noise"] = self.noise.profile.to_dict()
        if not self.presses:
            del out["presses"]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentSpec":
        if isinstance(data, cls):
            return data
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"AgentSpec: unknown field(s) {unknown}")
        return cls(**dict(data))


def _agents_default() -> tuple:
    return (AgentSpec(), AgentSpec())


@dataclasses.dataclass(frozen=True)
class MatchSpec:
    """One match (module doc, THE SPEC). Values are normalised on construction."""

    decks: tuple
    agents: tuple = dataclasses.field(default_factory=_agents_default)
    forms: tuple | None = None
    levels: Any = FRIENDLY_LEVEL
    tower_levels: Any = FRIENDLY_LEVEL
    delay: Any = None
    seed: int = 0
    engine: Any = None
    mem_lag: Any = rs_memory.BOARD_LAG
    elixir_milli: tuple = (START_ELIXIR, START_ELIXIR)
    shuffle: int = 0
    first: int = 0
    cadence: int = DECIDE
    first_decision: int = FIRST_DECISION
    max_ticks: int = MAX_TICKS
    tag: str = ""
    pairing: str = ""
    seat: int = 0

    def __post_init__(self) -> None:
        put = object.__setattr__
        decks = tuple(tuple(str(c) for c in d) for d in _two(self.decks, "decks"))
        if not all(decks):
            raise ValueError("decks: each team needs at least one card")
        put(self, "decks", decks)
        put(self, "agents", tuple(AgentSpec.from_dict(a) for a in _two(self.agents, "agents")))
        if self.forms is not None:
            forms = tuple(tuple(_int(f, "forms") for f in fs) for fs in _two(self.forms, "forms"))
            for t in (0, 1):
                if len(forms[t]) != len(decks[t]) or any(f not in (0, 1, 2) for f in forms[t]):
                    raise ValueError(f"forms of team {t}: 0 / 1 / 2 for each of its "
                                     f"{len(decks[t])} cards, got {list(forms[t])}")
            put(self, "forms", forms)
        if isinstance(self.levels, (int, np.integer)) and not isinstance(self.levels, bool):
            put(self, "levels", rs_engine.parse_level(int(self.levels), "levels"))
        else:
            levels = tuple(tuple(rs_engine.parse_level(lv, "levels") for lv in ls)
                           for ls in _two(self.levels, "levels"))
            for t in (0, 1):
                if len(levels[t]) != len(decks[t]):
                    raise ValueError(f"levels of team {t}: one per card ({len(decks[t])})")
            put(self, "levels", levels)
        if isinstance(self.tower_levels, (int, np.integer)) and not isinstance(self.tower_levels, bool):
            put(self, "tower_levels", rs_engine.parse_level(int(self.tower_levels), "tower_levels"))
        else:
            put(self, "tower_levels", tuple(rs_engine.parse_level(lv, "tower_levels")
                                            for lv in _two(self.tower_levels, "tower_levels")))
        if self.delay is not None:
            if isinstance(self.delay, (int, np.integer)) and not isinstance(self.delay, bool):
                put(self, "delay", _int(self.delay, "delay", 0))
            else:
                put(self, "delay", rs_engine.parse_delay(list(_two(self.delay, "delay"))))
        if self.engine is not None:
            engine = json.loads(self.engine) if isinstance(self.engine, str) else self.engine
            put(self, "engine", rs_engine.EngineConfig.coerce(engine).key())
        put(self, "mem_lag", RR.mem_lag_value(self.mem_lag))
        put(self, "elixir_milli", tuple(_int(e, "elixir_milli", 0)
                                        for e in _two(self.elixir_milli, "elixir_milli")))
        put(self, "seed", _int(self.seed, "seed", 0))
        put(self, "shuffle", _int(self.shuffle, "shuffle", 0))
        for what in ("first", "seat"):
            value = _int(getattr(self, what), what)
            if value not in (0, 1):
                raise ValueError(f"{what}: 0 or 1, got {value}")
            put(self, what, value)
        put(self, "cadence", _int(self.cadence, "cadence", 1))
        put(self, "first_decision", _int(self.first_decision, "first_decision", 0))
        put(self, "max_ticks", _int(self.max_ticks, "max_ticks", 1))
        put(self, "tag", str(self.tag or ""))
        put(self, "pairing", str(self.pairing or ""))

    # ------------------------------------------------------------ derived
    @property
    def id(self) -> str:
        return f"{self.pairing or self.tag or 'match'}/seat{self.seat}/seed{self.seed}"

    def replace(self, **changes: Any) -> "MatchSpec":
        return dataclasses.replace(self, **changes)

    def swapped(self) -> "MatchSpec":
        """The same game with the seats exchanged (module doc, THE SPEC)."""
        return dataclasses.replace(
            self, decks=_flip(self.decks), agents=_flip(self.agents), forms=_flip(self.forms),
            levels=self.levels if isinstance(self.levels, int) else _flip(self.levels),
            tower_levels=(self.tower_levels if isinstance(self.tower_levels, int)
                          else _flip(self.tower_levels)),
            delay=self.delay if self.delay is None or isinstance(self.delay, int) else _flip(self.delay),
            elixir_milli=_flip(self.elixir_milli), first=1 - self.first, seat=1 - self.seat)

    def base_engine(self) -> rs_engine.EngineConfig:
        """The engine config without this spec's delay: `engine`, else rs_engine.active()."""
        return rs_engine.resolve(None if self.engine is None else json.loads(self.engine))

    def engine_config(self) -> rs_engine.EngineConfig:
        """The match battle's config: base_engine() with the spec's delay (an int as
        rs_eval.play sets it, EngineConfig.delayed; a pair as (blue, red))."""
        cfg = self.base_engine()
        if self.delay is None:
            return cfg
        if isinstance(self.delay, int):
            return cfg.delayed(self.delay)
        return cfg.replace(delay=self.delay)

    def form_lists(self) -> list[list[int]]:
        if self.forms is None:
            return [[0] * len(d) for d in self.decks]
        return [list(f) for f in self.forms]

    def level_lists(self) -> list[list[int]]:
        if isinstance(self.levels, int):
            return [[self.levels] * len(d) for d in self.decks]
        return [list(lv) for lv in self.levels]

    def tower_level_list(self) -> list[int]:
        if isinstance(self.tower_levels, int):
            return [self.tower_levels, self.tower_levels]
        return list(self.tower_levels)

    def deck_ids(self, ids: Mapping[str, int]) -> list[list[int]]:
        out = []
        for t, deck in enumerate(self.decks):
            missing = [c for c in deck if c not in ids]
            if missing:
                raise ValueError(f"team {t}'s deck has card(s) {missing} that this engine's "
                                 "catalogue does not")
            out.append([int(ids[c]) for c in deck])
        return out

    def reset_kwargs(self, ids: Mapping[str, int]) -> dict:
        """Battle.reset keywords, as rs_eval.play passes them (lists, not tuples)."""
        return {"seed": self.seed, "decks": self.deck_ids(ids), "shuffle": self.shuffle,
                "start_tick": 0, "elixir_milli": list(self.elixir_milli), "tower_hp": None,
                "spawns": [], "forms": self.form_lists(), "levels": self.level_lists(),
                "tower_levels": self.tower_level_list()}

    def to_dict(self) -> dict:
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name == "agents":
                value = [a.to_dict() for a in value]
            elif f.name == "engine":
                value = None if value is None else json.loads(value)
            elif isinstance(value, tuple):
                value = [list(v) if isinstance(v, tuple) else v for v in value]
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MatchSpec":
        if isinstance(data, cls):
            return data
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"MatchSpec: unknown field(s) {unknown}")
        return cls(**dict(data))


# ---------------------------------------------------------------------------
# seats
# ---------------------------------------------------------------------------

class MatchAgent(rs_eval.RSAgent):
    """rs_eval.RSAgent (checkpoint loading and checks, rows, look-ahead, rs_decide.decide on the
    match battle) with the decision cadence of the match and the sampling preset added, and for
    a v2 checkpoint the live runner's rules for its own plays in flight, the opponent's elixir
    and the ability buttons (module doc, V2 SEATS)."""

    cadence = DECIDE
    #: AgentSpec.presses (configure sets it): press when the checkpoint's buttons trained.
    presses = False
    #: run_match's Counter of (team, card token, tick) entries of this seat's memory whose play
    #: the engine refused (V2 SEATS); None outside run_match.
    unran = None

    def presses_on(self) -> bool:
        """Whether decide may press for this seat: AgentSpec.presses and a checkpoint whose
        ability buttons trained (rs_decide.presses_allowed)."""
        return bool(self.presses) and rs_decide.presses_allowed(getattr(self, "meta", None))

    def decide_config(self):
        m = self.decide_mode
        kw = {"presses": True} if self.presses_on() else {}
        if m == rs_decide.MODE_LEGACY:
            return rs_decide.DecideConfig.legacy(self.wait_bias, dt=self.cadence, **kw)
        if m == rs_decide.MODE_LIVE:
            guard = rs_decide.card_ids(self._catalogue(), rs_decide.GUARD_CARD_NAMES)
            return rs_decide.DecideConfig.live(self.wait_bias, guard_cards=guard, dt=self.cadence,
                                               **kw)
        if m == rs_decide.MODE_SAMPLE:
            return rs_decide.DecideConfig.sampling(self.wait_bias, dt=self.cadence, **kw)
        raise ValueError(f"decide: one of {', '.join(rs_decide.MODES)}, got {m!r}")

    def own_pending(self, state, team) -> list:
        """RSAgent.own_pending, but none for a checkpoint whose rows were built at landing time
        (meta obs_delay ticks 0), as rs_decide.pending_inputs gives the live runner (V2 SEATS)."""
        if rs_decide.obs_delay_ticks(getattr(self, "meta", None)) <= 0:
            return []
        return super().own_pending(state, team)

    def opp_elixir(self, team, mem, tick, royalesim=None) -> tuple:
        """RSAgent.opp_elixir on the memory view less the opponent's entries in `unran`, the
        plays the engine never ran (V2 SEATS); each entry takes out one equal play of the
        view's "last" list. Without `unran` (or nothing in it): RSAgent.opp_elixir as is."""
        opp = 1 - int(team)
        drop = Counter({(int(c), int(t)): n for (who, c, t), n in (self.unran or {}).items()
                        if int(who) == opp and n > 0})
        if drop:
            last = dict(mem.get("last") or {})
            kept = []
            for card, when in last.get(opp, ()):
                key = (int(card), int(when))
                if drop[key] > 0:
                    drop[key] -= 1
                    continue
                kept.append((card, when))
            last[opp] = kept
            mem = {**mem, "last": last}
        return super().opp_elixir(team, mem, tick, royalesim)


#: The row layout of every checkpoint before E14 (vocab "schema", absent before E04).
LEGACY_SCHEMA = rs_schema.SCHEMA_LEGACY
#: Schemas whose rows RSAgent.act builds (rs_rows.obs_row): legacy (rs_rows.SCHEMA) and v2
#: (E14 / E15: RSAgent.v2_kw; module doc, V2 SEATS).
ROW_SCHEMAS = frozenset({LEGACY_SCHEMA, rs_schema.SCHEMA_V2, str(RR.SCHEMA)})


def seat_schema(seat: Any) -> str | None:
    """The row schema a seat's checkpoint was trained on: its meta "schema" (LEGACY_SCHEMA when
    the meta has none; "legacy" / "v2" as their full names); None for a seat without a meta (a
    stub, an idle seat)."""
    meta = getattr(seat, "meta", None)
    if not isinstance(meta, Mapping):
        return None
    name = str(meta.get("schema") or LEGACY_SCHEMA).strip()
    return rs_schema.SHORT_NAMES.get(name, name)


def check_schema(seat: Any, agent: AgentSpec) -> None:
    """ValueError when the seat's checkpoint schema is not the agent's (when given) or not one
    this runner builds rows for (ROW_SCHEMAS)."""
    got = seat_schema(seat)
    if got is None:
        return
    if agent.schema is not None and agent.schema != got:
        raise ValueError(f"{agent.label}: the checkpoint's rows are schema {got!r}, the agent "
                         f"spec says {agent.schema!r}")
    if got not in ROW_SCHEMAS:
        raise ValueError(f"{agent.label}: schema {got!r}; rs_match builds "
                         f"{sorted(ROW_SCHEMAS)} rows only")


def configure(seat: Any, agent: AgentSpec, cfg: Any) -> Any:
    """Set an RSAgent-like seat up as rs_eval.play sets its agent: engine, lead (rs_eval --lead
    encoding), hold, spell aim, decision rule, the presses switch (MatchAgent reads it) and
    (when given) visibility; checks the schema."""
    seat.engine = cfg
    seat.lead = abs(agent.lead) % 1000
    seat.lead_rebuild = -1000 < agent.lead < 0
    seat.lead_extrapolate = agent.lead >= 1000
    seat.hold_pending = agent.hold_pending
    seat.spell_aim = agent.spell_aim
    seat.decide_mode = agent.decide
    seat.presses = agent.presses
    if agent.visibility is not None:
        seat.visibility = agent.visibility
    check_schema(seat, agent)
    return seat


def build_seat(agent: AgentSpec, team: int, spec: MatchSpec, cfg: Any) -> Any:
    """The default seat factory: None for an idle agent, else a configured MatchAgent of the
    checkpoint, seeded with spec.seed + agent.seed_offset (loads torch)."""
    if agent.idle:
        return None
    kwargs: dict[str, Any] = {"engine": cfg.undelayed(), "allow_drift": agent.allow_drift}
    if agent.wait_keep is not None:
        kwargs["wait_keep"] = float(agent.wait_keep)
    seat = MatchAgent(agent.ckpt, seed=spec.seed + agent.seed_offset, **kwargs)
    seat.cadence = spec.cadence
    return configure(seat, agent, cfg.undelayed())


def seat_info(seat: Any, agent: AgentSpec) -> dict:
    """What the result records about a seat (JSON-ready)."""
    meta = getattr(seat, "meta", None)
    meta = meta if isinstance(meta, Mapping) else {}
    vis = getattr(seat, "visibility", None) if seat is not None else None
    on = getattr(seat, "presses_on", None)
    buttons = getattr(seat, "buttons", None)
    return {"name": agent.label, "ckpt": agent.ckpt, "lead": agent.lead, "decide": agent.decide,
            "visibility": vis if isinstance(vis, str) else agent.visibility,
            "schema": seat_schema(seat), "obs_delay": meta.get("obs_delay"),
            "live_compat": meta.get("live_compat"),
            "presses": bool(on()) if callable(on) else False,
            "buttons": None if buttons is None else [int(c) for c in buttons]}


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------

def noise_model(noise: Any, battle: Any, sub: int, cache: dict | None = None) -> Any:
    """A perception_noise.NoiseModel bound to `battle` for an AgentSpec.noise, or None for a
    perfect observer (None, "none", an all-zero profile). `cache` (a dict) keeps the models of a
    run, keyed by the profile and the engine's vocabulary."""
    if noise is None:
        return None
    if _is_noise_model(noise):
        return None if getattr(noise, "identity", False) else noise
    if isinstance(noise, str) and noise.strip().lower() in PN.NO_NOISE:
        return None
    profile = PN.as_profile(noise)
    if profile.is_identity():
        return None
    key = None
    if cache is not None:
        vocab = hashlib.sha1((battle.unit_types_json() + "\n" + battle.catalogue_json())
                             .encode("utf-8")).hexdigest()
        key = (profile.digest(), int(sub), vocab)
        if key in cache:
            return cache[key]
    model = PN.NoiseModel.from_battle(profile, battle, fields=RR.F, sub=sub)
    if cache is not None:
        cache[key] = model
    return model


def noise_seed(spec: MatchSpec, team: int) -> int:
    agent = spec.agents[team]
    return PN.match_seed(f"{spec.pairing or spec.tag}|{spec.seed}|{team}|{agent.label}",
                         agent.noise_seed)


def _noise_entry(nz: Any) -> dict | None:
    if nz is None:
        return None
    p = nz.model.profile
    return {"name": p.name, "digest": p.digest(), "kinds": p.active_groups(),
            "stats": dict(sorted(nz.stats.items()))}


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MatchResult:
    """One match's outcome (module doc, THE RESULT). Tuples are (team 0, team 1)."""

    id: str
    tag: str
    pairing: str
    seat: int
    seed: int
    agents: tuple
    winner: int
    crowns: tuple
    game_over: bool
    end_tick: int
    plays: tuple
    refused: tuple
    decisions: tuple
    full_ticks: tuple
    leak: tuple
    towers: tuple
    delay: tuple
    info: tuple = (None, None)
    noise: tuple = (None, None)
    log: tuple = ()
    timings: Any = None
    presses: tuple = (0, 0)
    press_refused: tuple = (0, 0)

    def score(self, team: int) -> float:
        """1 a win, 0 a loss, 0.5 a draw, for `team`."""
        if self.winner not in (0, 1):
            return 0.5
        return 1.0 if self.winner == team else 0.0

    def team_of(self, name: str) -> int:
        """The team the agent called `name` sat on; ValueError when absent or on both."""
        teams = [t for t in (0, 1) if self.agents[t] == name]
        if len(teams) != 1:
            raise ValueError(f"agent {name!r} is on {len(teams)} seats of {self.id} "
                             f"({list(self.agents)})")
        return teams[0]

    def score_of(self, name: str) -> float:
        return self.score(self.team_of(name))

    def to_dict(self, *, timings: bool = True, log: bool = True) -> dict:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name == "timings":
                if timings:
                    out["timings"] = value
                continue
            if f.name == "log":
                if log:
                    out["log"] = [list(e) for e in value]
                continue
            if isinstance(value, tuple):
                value = [list(v) if isinstance(v, tuple) else v for v in value]
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MatchResult":
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in data.items() if k in names}
        for k, v in list(kw.items()):
            if k in ("timings", "info", "noise"):
                kw[k] = tuple(v) if k != "timings" else v
            elif k == "log":
                kw[k] = tuple(tuple(e) for e in v)
            elif isinstance(v, list):
                kw[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        return cls(**kw)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def _law_for(cfg: rs_engine.EngineConfig) -> rs_elixir.ElixirLaw:
    if cfg.calibration_overrides:
        return rs_elixir.ElixirLaw.load(overrides=dict(cfg.calibration_overrides))
    return rs_elixir.default_law()


def _command(cmd: Any, team: int, state: Mapping[str, Any], *, buttons: int = 0,
             hand_size: int = rs_presses.FALLBACK_HAND_SIZE) -> tuple[int, int, int, int]:
    """A seat's command as (team, slot, x_sub, y_sub) ints, checked: its own team and a hand
    slot, or (buttons > 0: the seat may press) a button's slot hand_size + k, k < buttons."""
    try:
        out = tuple(int(v) for v in cmd)
    except (TypeError, ValueError):
        raise ValueError(f"team {team}'s seat returned {cmd!r}, not (team, slot, x_sub, y_sub)") from None
    hand = state["players"][team]["hand"]
    if (len(out) == 4 and out[0] == team and buttons > 0
            and hand_size <= out[1] < hand_size + buttons and out[1] >= len(hand)):
        return out
    if len(out) != 4 or out[0] != team or not 0 <= out[1] < len(hand):
        press = f" or a button slot {hand_size}-{hand_size + buttons - 1}" if buttons > 0 else ""
        raise ValueError(f"team {team}'s seat returned {cmd!r}: expected (team, slot, x_sub, "
                         f"y_sub) for its own team and a hand slot 0-{len(hand) - 1}{press}")
    return out


def _reason(row: Any) -> int:
    """The reason code of a Battle.step result entry: (card_id, reason, ...) on RoyaleSim,
    [team, reason] on the test fake; 0 (accepted) when it says nothing."""
    try:
        return int(row[1])
    except (TypeError, IndexError, ValueError):
        return 0


def _ran_rows(battle: Any) -> list[tuple[int, int, Any, int, int]]:
    """The last step's Battle.step_commands_run() rows, (tick, team, kind, what, reason): every
    delayed command that ran or was dropped in it (kind "deploy" a card play, any other a
    press); [] on engines and fakes without it."""
    fn = getattr(battle, "step_commands_run", None)
    if not callable(fn):
        return []
    out = []
    for row in fn() or ():
        row = tuple(row)
        if len(row) < 5:
            continue
        try:
            tick, team, reason = int(row[0]), int(row[1]), int(row[4])
        except (TypeError, ValueError):
            continue
        try:
            what = int(row[3])
        except (TypeError, ValueError):
            what = -1                           # a press may name no card
        if team in (0, 1):
            out.append((tick, team, row[2], what, reason))
    return out


def _ran_refusals(battle: Any) -> list[int]:
    """Teams of the delayed deploys the last step refused or dropped when they ran
    (Battle.step_commands_run; absent on engines and fakes without it)."""
    return [r[1] for r in _ran_rows(battle)
            if r[2] == rs_memory.RAN_DEPLOY and r[4] != rs_memory.RAN_OK]


def _ran_play(flying: list[int], log: Sequence[Sequence[int]], exec_of: Mapping[int, int],
              team: int, card: int, tick: int) -> int | None:
    """The log index of the play a step_commands_run deploy row (`team` ran `card` on `tick`)
    reports, taken out of `flying` (plays accepted under a delay and not reported yet): the
    oldest of that team and card dated `tick`, else the oldest dated before it; None if none."""
    pick = None
    for j, k in enumerate(flying):
        if log[k][1] != team or log[k][2] != card or exec_of[k] > tick:
            continue
        if exec_of[k] == tick:
            pick = j
            break
        if pick is None:
            pick = j
    return None if pick is None else flying.pop(pick)


class _Recorder:
    """A PublicMemory stand-in for MatchNoise.record: records into `mem` and keeps what it was
    handed, (card token, tick), the entry that seat's memory got for the play."""

    __slots__ = ("mem", "got")

    def __init__(self, mem: rs_memory.PublicMemory):
        self.mem, self.got = mem, None

    def record(self, team: int, card: int, exec_tick: int) -> None:
        self.mem.record(team, card, exec_tick)
        self.got = (int(card), int(exec_tick))


def run_match(spec: MatchSpec | Mapping[str, Any], *, seats: Sequence[Any] | None = None,
              seat_factory: Any = None, ctx: Any = None, law: rs_elixir.ElixirLaw | None = None,
              noise_models: dict | None = None, keep_log: bool = True) -> MatchResult:
    """Play one match (module doc, THE LOOP).

    seats         two seat objects (team 0, team 1; None = idle), used as they are; else each
                  is seat_factory(agent_spec, team, spec, cfg), default build_seat
    ctx           rs_rows._setup(cfg)'s tuple, when the caller already has it
    law           the rs_elixir.ElixirLaw the leak is counted under (default: the engine's)
    noise_models  a dict that keeps NoiseModels across matches (run_many passes one)

    A seat whose `v2` is true gets its `buttons` from the deck and forms the spec deals its team,
    and a seat that has an `unran` attribute (MatchAgent) gets the Counter of its memory's
    entries whose play the engine refused (module doc, V2 SEATS).
    """
    t_start = time.perf_counter()
    spec = MatchSpec.from_dict(spec)
    cfg = spec.engine_config()
    if ctx is None:
        ctx = RR._setup(cfg)
    royalesim, _names, ids, cat = ctx
    sub = royalesim.SUBTILE_PER_MILLITILE
    lag = spec.mem_lag
    legacy = lag == RR.MEM_LEGACY
    delays = tuple(int(d) for d in (cfg.delay or (0, 0)))
    engine_s = 0.0
    t0 = time.perf_counter()
    battle = rs_engine.make_battle(cfg, royalesim)
    dealt = spec.reset_kwargs(ids)
    battle.reset(**dealt)
    engine_s += time.perf_counter() - t0
    t0 = time.perf_counter()
    if seats is None:
        factory = seat_factory or build_seat
        seats = [factory(spec.agents[t], t, spec, cfg) for t in (0, 1)]
    else:
        seats = list(_two(seats, "seats"))
    # V2 SEATS: the press slots each team's spec allows (0: none), and per observing seat the
    # entries (team, card token, tick) of its memory whose play the engine refused.
    hand_size = rs_presses.hand_size(royalesim)
    n_buttons = rs_presses.button_count(royalesim)
    press_slots = [n_buttons if spec.agents[t].presses else 0 for t in (0, 1)]
    unran = [Counter(), Counter()]
    champs = None
    for t, seat in enumerate(seats):
        if seat is None:
            continue
        v2 = bool(getattr(seat, "v2", False))
        if rs_eval.seat_visibility(seat) == RR.VIS_PUBLIC or v2:
            RR.check_status_bits(royalesim)       # a v2 row reads the burrow bit too
        if v2:
            if champs is None:
                champs = rs_presses.champion_ids(cat, getattr(royalesim, "CATALOGUE_FIELDS", None))
            seat.buttons = rs_presses.side_buttons(dealt["decks"][t], dealt["forms"][t], champs)
        if hasattr(seat, "unran"):
            seat.unran = unran[t]
    noise = []
    for t in (0, 1):
        model = noise_model(spec.agents[t].noise, battle, sub, noise_models)
        noise.append(None if model is None else PN.MatchNoise(model, noise_seed(spec, t)))
    setup_s = time.perf_counter() - t0

    mems = [rs_memory.PublicMemory(), rs_memory.PublicMemory()]
    order = (spec.first, 1 - spec.first)
    plays, refused, decisions, full = [0, 0], [0, 0], [0, 0], [0, 0]
    pressed, press_refused = [0, 0], [0, 0]
    leak_fine = [0, 0]
    execs: list[list[int]] = [[], []]
    decide_s = [0.0, 0.0]
    log: list[list[int]] = []
    # Per play (its log index): (team, the entry each seat's memory got or None), its exec tick;
    # flying: the plays accepted under a delay that step_commands_run has not reported yet.
    entries: dict[int, tuple] = {}
    exec_of: dict[int, int] = {}
    flying: list[int] = []

    def never_ran(k: int) -> None:
        """Play k was refused: each seat's memory entry of it goes into that seat's unran."""
        who, got = entries.pop(k, (None, ()))
        for obs, entry in enumerate(got):
            if entry is not None:
                unran[obs][(who, entry[0], entry[1])] += 1

    cap = None
    tick = 0
    while tick < spec.max_ticks:
        t0 = time.perf_counter()
        state = json.loads(battle.state_json())
        engine_s += time.perf_counter() - t0
        if state["game_over"]:
            break
        if cap is None:
            law = law or _law_for(cfg.undelayed())
            if state.get("regular_ticks") and int(state["regular_ticks"]) != law.regular_ticks:
                law = law.with_regular_ticks(int(state["regular_ticks"]))
            cap = law.cap_fine * 1000 // law.scale
        deciding = tick % spec.cadence == 0 and tick >= spec.first_decision
        # E08: one view per seat for the tick, before anything is sent (legacy: at each act).
        views = None if legacy else [mems[t].view(tick, lag) for t in (0, 1)]
        cmds, sent = [], []
        for team in order:
            seat, nz = seats[team], noise[team]
            if not deciding or seat is None:
                if nz is not None:
                    nz.push(state)
                continue
            t1 = time.perf_counter()
            seen, shift = state, 0
            if nz is not None:
                seen = nz.observe(state, team)
                shift = nz.tick_shift
            view = mems[team].view(tick, 0) if legacy else views[team]
            cmd = seat.act(battle, seen, team, view, tick + shift, sub, royalesim)
            decide_s[team] += time.perf_counter() - t1
            decisions[team] += 1
            if cmd is None:
                continue
            cmd = _command(cmd, team, state, buttons=press_slots[team], hand_size=hand_size)
            hand = state["players"][team]["hand"]
            cmds.append(cmd)
            exec_tick = tick + delays[team]
            if cmd[1] >= len(hand):
                # V2 SEATS: a press (its spec lets the seat press). No card play: no memory entry,
                # nothing in flight for the look-ahead; its cost (for the leak) from the abilities.
                button = cmd[1] - hand_size
                pressed[team] += 1
                sent.append((team, exec_tick, len(log), rs_decide.press_cost(state, team, button)))
                log.append([tick, team, rs_decide.button_card(state, team, button), cmd[1],
                            cmd[2], cmd[3], 0])
                continue
            card = int(hand[cmd[1]])
            # Our own plays in flight, for the rebuilt look-ahead (rs_eval.play's agent.sent).
            seat.sent = list(getattr(seat, "sent", None) or []) + [(exec_tick, card, cmd[2], cmd[3])]
            when = tick if legacy else exec_tick
            token = card + rs_memory.TOKEN_OFFSET
            got = []
            for obs in (0, 1):
                if obs != team and noise[obs] is not None:
                    seen_as = _Recorder(mems[obs])
                    noise[obs].record(seen_as, team, token, when)
                    got.append(seen_as.got)                 # None: that seat missed the play
                else:
                    mems[obs].record(team, token, when)
                    got.append((token, when))
            plays[team] += 1
            entries[len(log)] = (team, tuple(got))
            exec_of[len(log)] = exec_tick
            sent.append((team, exec_tick, len(log), None))
            log.append([tick, team, card, cmd[1], cmd[2], cmd[3], 0])
        step = max(1, min(END_TICK, (tick // spec.cadence + 1) * spec.cadence) - tick)
        step = max(1, min(step, spec.max_ticks - tick))
        t0 = time.perf_counter()
        res = list(battle.step(cmds, step) or ())
        ran = _ran_rows(battle)
        engine_s += time.perf_counter() - t0
        for i, (team, exec_tick, k, press_cost) in enumerate(sent):
            reason = _reason(res[i]) if i < len(res) else 0
            log[k][6] = reason
            if press_cost is not None:                  # a press
                if reason != 0:
                    press_refused[team] += 1
                elif press_cost > 0:
                    execs[team].append(exec_tick)       # it spends elixir, as a play does
                continue
            if reason == 0:
                execs[team].append(exec_tick)
                if delays[team] > 0:
                    flying.append(k)
            else:
                refused[team] += 1
                never_ran(k)
        # Under a delay: the commands that ran (or were dropped) in this step.
        for run_tick, team, kind, what, reason in ran:
            if kind == rs_memory.RAN_DEPLOY:
                k = _ran_play(flying, log, exec_of, team, what, run_tick)
                if reason != rs_memory.RAN_OK:
                    refused[team] += 1
                    if k is not None:
                        never_ran(k)
            elif reason != rs_memory.RAN_OK:
                press_refused[team] += 1
        # Elixir leak over [tick, tick + step) (module doc, ELIXIR LEAK).
        for t in (0, 1):
            if state["players"][t]["elixir_milli"] < cap:
                continue
            ex = execs[t]
            k = bisect.bisect_left(ex, tick)
            end = ex[k] if k < len(ex) and ex[k] < tick + step else tick + step
            full[t] += end - tick
            leak_fine[t] += law.regen(tick, end)
        tick += step
    t0 = time.perf_counter()
    state = json.loads(battle.state_json())
    engine_s += time.perf_counter() - t0
    w = state.get("winner")
    winner = int(w) if w in (0, 1) else -1
    players = state["players"]
    scale = law.scale if law is not None else 1
    return MatchResult(
        id=spec.id, tag=spec.tag, pairing=spec.pairing, seat=spec.seat, seed=spec.seed,
        agents=(spec.agents[0].label, spec.agents[1].label), winner=winner,
        crowns=tuple(int(players[t].get("crowns", 0)) for t in (0, 1)),
        game_over=bool(state["game_over"]), end_tick=int(state["tick"]),
        plays=tuple(plays), refused=tuple(refused), decisions=tuple(decisions),
        full_ticks=tuple(full), leak=tuple(f / scale for f in leak_fine),
        towers=tuple(tuple(int(h) for h in players[t].get("tower_hp", ())) for t in (0, 1)),
        delay=delays, info=tuple(seat_info(seats[t], spec.agents[t]) for t in (0, 1)),
        noise=tuple(_noise_entry(nz) for nz in noise),
        log=tuple(tuple(e) for e in log) if keep_log else (),
        timings={"total_s": time.perf_counter() - t_start, "setup_s": setup_s,
                 "engine_s": engine_s, "decide_s": list(decide_s)},
        presses=tuple(pressed), press_refused=tuple(press_refused))


def _run_dict(spec_dict: Mapping[str, Any]) -> dict:
    """A worker's task: one spec as a dict in, its result as a dict out."""
    return run_match(MatchSpec.from_dict(spec_dict)).to_dict()


def run_many(specs: Iterable[MatchSpec | Mapping[str, Any]], *, workers: int = 1,
             law: rs_elixir.ElixirLaw | None = None) -> list[MatchResult]:
    """run_match over `specs`, in order. workers > 1 runs them in processes; every spec is first
    given the parent's engine config when it has none, so the workers run the same engine."""
    specs = [MatchSpec.from_dict(s) for s in specs]
    if workers <= 1 or len(specs) <= 1:
        models: dict = {}
        return [run_match(s, law=law, noise_models=models) for s in specs]
    base = rs_engine.active()
    tasks = [(s if s.engine is not None else s.replace(engine=base.to_json())).to_dict()
             for s in specs]
    with ProcessPoolExecutor(workers, initializer=rs_engine.set_active,
                             initargs=(base.to_json(),)) as pool:
        return [MatchResult.from_dict(d) for d in pool.map(_run_dict, tasks)]


def summarize(results: Sequence[MatchResult], name: str) -> dict:
    """Agent `name`'s record over `results`: score and Wilson interval (draws half), and per game
    means of the crown difference, plays, elixir leak and seconds at full elixir; by_team splits
    the score by the team it sat on."""
    rows = [(r, r.team_of(name)) for r in results]
    si = score_interval([r.score(t) for r, t in rows])
    n = max(1, len(rows))
    out = {"agent": name, **si.to_dict(),
           "crown_diff": sum(r.crowns[t] - r.crowns[1 - t] for r, t in rows) / n,
           "plays": sum(r.plays[t] for r, t in rows) / n,
           "leak": sum(r.leak[t] for r, t in rows) / n,
           "full_s": sum(r.full_ticks[t] for r, t in rows) / n / 20,
           "by_team": {}}
    for team in (0, 1):
        sub = [r.score(t) for r, t in rows if t == team]
        if sub:
            out["by_team"][team] = score_interval(sub).to_dict()
    return out


# ---------------------------------------------------------------------------
# the legacy reactive screen (rs_eval.play) as a preset
# ---------------------------------------------------------------------------

def real_result(battle: Mapping[str, Any], hog_seat: int) -> str:
    """The recorded result from the Hog side's view (rs_eval.play's "real")."""
    real = str(battle.get("result", "")).lower()
    if hog_seat == 0:
        return "win" if real in ("victory", "win") else "loss" if real in ("defeat", "loss") else "draw"
    return "loss" if real in ("victory", "win") else "win" if real in ("defeat", "loss") else "draw"


def legacy_screen_spec(payload: Mapping[str, Any], ctx: Any, *, ckpt: str, opp_ckpt: str,
                       seed: int = 0, delay: int = 0, lead: int = 0, hold: bool = False,
                       spell_aim: bool = False, mem_lag: Any = rs_memory.BOARD_LAG,
                       decide: str = rs_decide.MODE_LEGACY, visibility: str | None = None,
                       engine: Any = None) -> tuple[MatchSpec | None, Any]:
    """The MatchSpec of rs_eval.play's reactive screen for one recorded match: (spec, info) with
    info {"hog_seat", "hog_team", "real"}; or (None, rs_eval.skip_record(...)) for a match the
    orientation check skips, (None, None) when no deal order fits the plays (rs_eval: None)."""
    _rs, names, _ids, cat = ctx
    battle = payload["battle"]
    tag = payload["source"]["replay_tag"]
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
    events = sorted((e for e in payload["events"] if e["kind"] == "play_card"),
                    key=lambda e: (e["replay_tick_20hz"], e["source_index"]))
    plays: dict[int, list] = {0: [], 1: []}
    for e in events:
        u = e["coordinates"]["native_world_units"]
        plays[0 if e["side"] == "team" else 1].append(
            (int(e["replay_tick_20hz"]), names[base_key(e["card_key"])[0]], int(u["x"]), int(u["y"])))
    ori = rs_orient.orient(plays, rs_orient.troop_names(cat))
    if not ori.ok():
        return None, rs_eval.skip_record(tag, "agent", rs_orient.SKIP, ori.describe())
    rs = {0: ori.team_seat, 1: 1 - ori.team_seat}
    seat_of = {rs[0]: 0, rs[1]: 1}
    hog_seat = 0 if frozenset(decks[0]) == RR.HOG26 else 1
    hog_team = rs[hog_seat]
    dealt = []
    for side in (0, 1):
        order = deal_order(decks[side], [c for _, c, _, _ in plays[side]])
        if order is None:
            return None, None
        dealt.append(order)
    perm = [[decks[s].index(c) for c in dealt[s]] for s in (0, 1)]
    tower = max(1, min(16, mode([lv for s in levels for lv in s])))
    agents = [AgentSpec(), AgentSpec()]
    agents[hog_team] = AgentSpec(ckpt, name="agent", lead=lead, hold_pending=hold,
                                 spell_aim=spell_aim, decide=decide, visibility=visibility)
    agents[1 - hog_team] = AgentSpec(opp_ckpt, name="opponent", decide=decide,
                                     visibility=visibility, seed_offset=LEGACY_OPPONENT_OFFSET)
    spec = MatchSpec(
        decks=tuple(tuple(dealt[seat_of[t]]) for t in (0, 1)), agents=tuple(agents),
        forms=tuple(tuple(forms[seat_of[t]][i] for i in perm[seat_of[t]]) for t in (0, 1)),
        levels=tuple(tuple(levels[seat_of[t]][i] for i in perm[seat_of[t]]) for t in (0, 1)),
        tower_levels=(tower, tower), delay=int(delay), seed=int(seed),
        engine=None if engine is None else rs_engine.EngineConfig.coerce(engine).to_json(),
        mem_lag=mem_lag, first=hog_team, tag=tag, pairing=tag)
    return spec, {"hog_seat": hog_seat, "hog_team": hog_team,
                  "real": real_result(battle, hog_seat)}


def play_legacy_screen(args: Sequence[Any], *, law: rs_elixir.ElixirLaw | None = None) -> Any:
    """rs_eval.play's reactive screen through run_match. `args` is rs_eval.play's tuple
    (payload_text, ckpt, mode, seed, opp_ckpt, delay, lead, hold, shadow, spell_aim, engine,
    mem_lag, decide, visibility); mode must be "agent", opp_ckpt given and shadow off (the ghost,
    pro, search and shadow screens stay in rs_eval). Returns rs_eval.play's dict plus "match"."""
    payload_text, ckpt, mode_, seed = args[:4]
    opp_ckpt = args[4] if len(args) > 4 else ""
    delay = args[5] if len(args) > 5 else 0
    lead = args[6] if len(args) > 6 else 0
    hold = args[7] if len(args) > 7 else False
    shadow = args[8] if len(args) > 8 else False
    spell_aim = args[9] if len(args) > 9 else False
    decide = args[12] if len(args) > 12 and args[12] is not None else rs_decide.MODE_LEGACY
    visibility = args[13] if len(args) > 13 else None
    if mode_ != "agent" or not opp_ckpt or shadow:
        raise ValueError("the legacy screen preset is rs_eval.play's reactive screen: mode "
                         "'agent', an opponent checkpoint, no shadow (the others: tools/rs_eval.py)")
    if decide not in (rs_decide.MODE_LEGACY, rs_decide.MODE_LIVE):
        raise ValueError(f"decide: {rs_decide.MODE_LEGACY!r} or {rs_decide.MODE_LIVE!r}, "
                         f"got {decide!r}")
    if visibility is not None:
        RR.visibility_mode(visibility)
    mem_lag = RR.mem_lag_value(args[11] if len(args) > 11 and args[11] is not None
                               else rs_memory.BOARD_LAG)
    if len(args) > 10 and args[10] is not None:
        rs_engine.set_active(args[10])
    cfg = rs_engine.active()
    ctx = RR._setup(cfg)
    payload = json.loads(payload_text)
    spec, info = legacy_screen_spec(payload, ctx, ckpt=ckpt, opp_ckpt=opp_ckpt, seed=seed,
                                    delay=delay, lead=lead, hold=hold, spell_aim=spell_aim,
                                    mem_lag=mem_lag, decide=decide, visibility=visibility,
                                    engine=cfg)
    if spec is None:
        return info
    hog_team = info["hog_team"]
    agent = rs_eval.AGENT_CLASS(ckpt, seed=seed)
    opponent = rs_eval.RSAgent(opp_ckpt, seed=seed + LEGACY_OPPONENT_OFFSET)
    seats = [None, None]
    seats[hog_team], seats[1 - hog_team] = agent, opponent
    for t in (0, 1):
        configure(seats[t], spec.agents[t], cfg)
    result = run_match(spec, seats=seats, ctx=ctx, law=law)
    w = result.winner
    return {"tag": spec.tag, "mode": mode_, "real": info["real"], "game_over": result.game_over,
            "end_tick": result.end_tick,
            "result": "win" if w == hog_team else "loss" if w in (0, 1) else "draw",
            "crowns": [result.crowns[hog_team], result.crowns[1 - hog_team]],
            "agent_plays": result.plays[hog_team], "records": getattr(agent, "records", None),
            "hog_team": hog_team, "match": result.to_dict()}


# ---------------------------------------------------------------------------
# deck pairs from IL_Replay
# ---------------------------------------------------------------------------

def corpus_split(split: str, *, allow_final: bool = False) -> str:
    """"train", "dev" / "val" or "final" / "test" -> the rs_rows.split_of name. The final split
    is refused unless allow_final (module doc, DECK PAIRS)."""
    name = SPLIT_ALIASES.get(str(split), str(split))
    if name not in deck_stats.SPLITS:
        raise ValueError(f"split {split!r}: train, dev (val) or final (test)")
    if name == FINAL_SPLIT and not allow_final:
        raise ValueError("the final split (IL_Replay test) is held out: pass allow_final=True "
                         "(--final) only for the reported final evaluation")
    return name


@dataclasses.dataclass(frozen=True)
class DeckPair:
    """A recorded match's two decks: side a and side b, each RoyaleSim names aligned with the
    IL_Replay base keys (sorted) and the recorded forms; score is side a's recorded result."""

    tag: str
    split: str
    decks: tuple
    forms: tuple
    keys: tuple
    score: float = 0.5

    def flipped(self) -> "DeckPair":
        return DeckPair(self.tag, self.split, _flip(self.decks), _flip(self.forms),
                        _flip(self.keys), 1.0 - self.score)

    def to_dict(self) -> dict:
        return {"tag": self.tag, "split": self.split, "decks": [list(d) for d in self.decks],
                "forms": [list(f) for f in self.forms], "keys": [list(k) for k in self.keys],
                "score": self.score}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeckPair":
        return cls(str(data["tag"]), str(data["split"]),
                   tuple(tuple(str(c) for c in d) for d in data["decks"]),
                   tuple(tuple(int(f) for f in fs) for fs in data["forms"]),
                   tuple(tuple(str(k) for k in ks) for ks in data["keys"]),
                   float(data.get("score", 0.5)))


def pair_of(row: Mapping[str, Any], names: Mapping[str, str],
            modes: Iterable[str] | None = deck_stats.MODES) -> tuple[DeckPair | None, str]:
    """A replays row (replay_tag, game_mode, payload_json) -> (DeckPair, "ok"), or (None, why):
    "no tag", "mode", "json", "not 1v1", "deck" (not 8 different cards) or "card" (a key the
    card map lacks), as tools/deck_stats.py counts them."""
    modes = None if modes is None else frozenset(modes)
    tag = row.get("replay_tag")
    if not tag:
        return None, "no tag"
    game_mode = row.get("game_mode")
    if modes is not None and game_mode is not None and game_mode not in modes:
        return None, "mode"
    payload = row.get("payload_json")
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            payload = json.loads(payload)
        except ValueError:
            return None, "json"
    if not isinstance(payload, Mapping):
        return None, "json"
    battle = payload.get("battle") or {}
    if game_mode is None and modes is not None and battle.get("game_mode") not in modes:
        return None, "mode"
    sides = []
    for side in ("team", "opponent"):
        players = (battle.get(side) or {}).get("players") or []
        if len(players) != 1:
            return None, "not 1v1"
        try:
            deck = deck_stats.deck_of(players[0])
        except (AttributeError, KeyError, TypeError):
            deck = None
        if deck is None:
            return None, "deck"
        sides.append(deck)
    if any(k not in names for keys, _ in sides for k in keys):
        return None, "card"
    tag = str(tag)
    return DeckPair(tag=tag, split=deck_stats.split_of(tag),
                    decks=tuple(tuple(names[k] for k in keys) for keys, _ in sides),
                    forms=tuple(tuple(forms) for _, forms in sides),
                    keys=tuple(tuple(keys) for keys, _ in sides),
                    score=deck_stats.team_score(battle.get("result"))), "ok"


@dataclasses.dataclass(frozen=True)
class PairSet:
    """The deck pairs of one split, in corpus order, and how they were read."""

    split: str
    pairs: tuple
    meta: Any = None

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def tags(self) -> list[str]:
        return [p.tag for p in self.pairs]

    def digest(self) -> str:
        """16 hex digits over the pairs (tags, decks, forms): equal means the same pairings."""
        body = json.dumps([[p.tag, [list(d) for d in p.decks], [list(f) for f in p.forms]]
                           for p in self.pairs], separators=(",", ":"))
        return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]

    def sample(self, n: int, seed: int = 0) -> "PairSet":
        """n pairs drawn without replacement by `seed`, kept in corpus order; all when n >= len."""
        n = _int(n, "n", 0)
        if n >= len(self.pairs):
            return self
        rng = np.random.default_rng(_int(seed, "seed", 0))
        keep = sorted(int(i) for i in rng.choice(len(self.pairs), size=n, replace=False))
        meta = dict(self.meta or {})
        meta.update(sampled=n, sample_seed=int(seed), of=len(self.pairs))
        return PairSet(self.split, tuple(self.pairs[i] for i in keep), meta)

    def to_dict(self) -> dict:
        return {"schema": 1, "tool": "tools/rs_match.py", "split": self.split,
                "digest": self.digest(), "meta": dict(self.meta or {}),
                "pairs": [p.to_dict() for p in self.pairs]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PairSet":
        if data.get("schema") != 1:
            raise ValueError(f"pair set schema {data.get('schema')!r}; this code reads 1")
        out = cls(str(data["split"]), tuple(DeckPair.from_dict(p) for p in data["pairs"]),
                  dict(data.get("meta") or {}))
        if data.get("digest") not in (None, out.digest()):
            raise ValueError("pair set digest does not match its pairs (edited by hand?)")
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PairSet":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def read_pairs(rows: Iterable[Mapping[str, Any]], names: Mapping[str, str], split: str = "dev",
               *, require: Iterable[str] | None = None, exclude_mirror: bool = True,
               limit: int | None = None, allow_final: bool = False,
               modes: Iterable[str] | None = deck_stats.MODES) -> PairSet:
    """The deck pairs of one split among replays `rows` (module doc, DECK PAIRS). A row of
    another split is passed over before its payload is parsed. `require`: a deck (card names)
    one side must play, put on side a; `limit`: stop after that many pairs."""
    want = corpus_split(split, allow_final=allow_final)
    need = None if require is None else frozenset(require)
    pairs, seen, skipped = [], set(), Counter()
    other = 0
    for row in rows:
        tag = row.get("replay_tag")
        if tag and deck_stats.split_of(str(tag)) != want:
            other += 1
            continue
        pair, why = pair_of(row, names, modes)
        if pair is None:
            skipped[why] += 1
            continue
        if pair.tag in seen:
            skipped["duplicate"] += 1
            continue
        a, b = frozenset(pair.decks[0]), frozenset(pair.decks[1])
        if exclude_mirror and a == b:
            skipped["mirror"] += 1
            continue
        if need is not None:
            if b == need and a != need:
                pair = pair.flipped()
            elif a != need:
                skipped["require"] += 1
                continue
        seen.add(pair.tag)
        pairs.append(pair)
        if limit is not None and len(pairs) >= limit:
            break
    meta = {"split_rule": deck_stats.SPLIT_RULE, "other_splits": other,
            "skipped": dict(sorted(skipped.items())), "exclude_mirror": bool(exclude_mirror),
            "require": None if need is None else sorted(need), "limit": limit}
    return PairSet(want, tuple(pairs), meta)


def load_pairs(corpus: str | Path | None = None, split: str = "dev", *,
               names: Mapping[str, str] | None = None, parts: int = 0, **kwargs: Any) -> PairSet:
    """read_pairs over the corpus's replays parquet (default <data>/il_replay; pyarrow) with the
    card map rs_rows uses (tools/rs_cards.load()) unless `names` is given."""
    corpus = Path(corpus) if corpus is not None else deck_stats.data_root() / "il_replay"
    files = deck_stats.corpus_parts(corpus, parts)
    if not files:
        raise FileNotFoundError(f"no IL_Replay parquet in {corpus / 'replays'} "
                                "(run tools/il_replay_download.py)")
    out = read_pairs(deck_stats.iter_rows(files), rs_cards.load() if names is None else names,
                     split, **kwargs)
    meta = dict(out.meta or {})
    meta.update(corpus=str(corpus), parts=[f.name for f in files])
    return PairSet(out.split, out.pairs, meta)


def _pair_seed(seed: int, tag: str) -> int:
    return int(hashlib.sha1(f"{int(seed)}:{tag}".encode("utf-8")).hexdigest()[:12], 16)


def deal(cards: Sequence[str], forms: Sequence[int], rng: np.random.Generator
         ) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """A deck and its forms in one random deal order (the same permutation for both)."""
    order = [int(i) for i in rng.permutation(len(cards))]
    return tuple(cards[i] for i in order), tuple(int(forms[i]) for i in order)


def pairing_specs(pairs: Iterable[DeckPair], agents: Sequence[AgentSpec], *, seed: int = 0,
                  seats: int = 2, base_forms: Sequence[bool] = (False, False),
                  **spec_kwargs: Any) -> list[MatchSpec]:
    """MatchSpecs for deck pairs: agent a plays side a's deck at team 0, agent b side b's at
    team 1, both dealt with a rng seeded by (seed, tag); seats=2 adds the seat-swapped game
    (MatchSpec.swapped). The engine seed is derived from (seed, tag) too, so a pairing is the same
    games in every run and for every candidate. base_forms (side a, side b): play that side in
    base forms (the owner's account has no evolutions or heroes). spec_kwargs go to MatchSpec
    (delay, engine, levels, mem_lag, max_ticks, ...)."""
    a, b = (AgentSpec.from_dict(x) for x in _two(agents, "agents"))
    if seats not in (1, 2):
        raise ValueError(f"seats: 1 or 2, got {seats!r}")
    base = tuple(bool(x) for x in _two(base_forms, "base_forms"))
    out = []
    for pair in pairs:
        pair_seed = _pair_seed(seed, pair.tag)
        rng = np.random.default_rng(pair_seed)
        dealt = [deal(pair.decks[s], pair.forms[s], rng) for s in (0, 1)]
        forms = tuple(tuple(0 for _ in f) if base[s] else f for s, (_, f) in enumerate(dealt))
        spec = MatchSpec(decks=(dealt[0][0], dealt[1][0]), forms=forms, agents=(a, b),
                         seed=pair_seed % (2 ** 31), tag=pair.tag,
                         pairing=f"{pair.split}:{pair.tag}", seat=0, **spec_kwargs)
        out.append(spec)
        if seats == 2:
            out.append(spec.swapped())
    return out


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _require(text: str) -> frozenset[str] | None:
    t = (text or "").strip()
    if t.lower() in ("", "none", "any"):
        return None
    if t.lower() in ("hog26", "hog", "hog2.6"):
        return HOG26
    cards = frozenset(c.strip() for c in t.split(",") if c.strip())
    if len(cards) != 8:
        raise ValueError(f"--require: 'hog26', 'none' or 8 comma-separated card names, got {text!r}")
    return cards


def _delay_arg(text: str | None) -> Any:
    if text is None:
        return None
    d = rs_engine.parse_delay(text)
    if d is None:
        return None
    return d[0] if d[0] == d[1] else d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Play two checkpoints on IL_Replay deck pairings in "
                                             "RoyaleSim, both seats (L02).")
    ap.add_argument("--a", required=True, metavar="CKPT", help="agent A (the candidate); 'idle' "
                    "never plays. A plays side a: the --require deck")
    ap.add_argument("--b", required=True, metavar="CKPT", help="agent B (the reference)")
    ap.add_argument("--name-a", default="a")
    ap.add_argument("--name-b", default="b")
    ap.add_argument("--split", default="dev", help="train, dev (IL_Replay val) or final (test)")
    ap.add_argument("--final", action="store_true", help="allow --split final (held out)")
    ap.add_argument("--pairs", type=int, default=100, help="deck pairings (0: every one)")
    ap.add_argument("--seats", type=int, choices=(1, 2), default=2,
                    help="games per pairing: 2 adds the seat-swapped game")
    ap.add_argument("--require", default="hog26",
                    help="a deck one side must play, side a: hog26, none or 8 card names")
    ap.add_argument("--keep-mirrors", action="store_true")
    ap.add_argument("--base-forms", default="a", choices=("a", "b", "ab", "none"),
                    help="sides played in base forms (default a: the bot's account has no "
                         "evolutions or heroes; opponents keep their recorded forms)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", default=None, metavar="TICKS",
                    help="command delay, 'N' or 'B,R' (live: 21-22 client + capture, ~24); "
                         "default: the engine config's")
    ap.add_argument("--decide", choices=rs_decide.MODES, default=rs_decide.MODE_LIVE)
    ap.add_argument("--lead-a", type=int, default=0)
    ap.add_argument("--lead-b", type=int, default=0)
    ap.add_argument("--noise-a", default=None, metavar="PROFILE",
                    help="A's perception noise: a profile JSON, 'default' or 'none'")
    ap.add_argument("--noise-b", default=None, metavar="PROFILE")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--visibility", choices=RR.VISIBILITY_MODES, default=None)
    ap.add_argument("--presses", default="none", choices=("none", "a", "b", "ab"),
                    help="agents that may press their ability buttons, when their checkpoint's "
                         "buttons trained (rs_decide.presses_allowed; default none)")
    RR.add_mem_lag_arg(ap)
    ap.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    ap.add_argument("--pairs-file", type=Path, default=None,
                    help="read the pairings from this JSON, or write them there when missing")
    ap.add_argument("--corpus", type=Path, default=None, help="default <data>/il_replay")
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--allow-vocab-drift", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "rs_match.jsonl")
    rs_engine.add_engine_args(ap)
    args = ap.parse_args(argv)
    try:
        cfg = rs_engine.config_from_args(args)
        delay = _delay_arg(args.delay)
        require = _require(args.require)
        split = corpus_split(args.split, allow_final=args.final)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    rs_engine.set_active(cfg)
    import rs_checkpoint
    if args.allow_vocab_drift:
        rs_checkpoint.set_allow_drift(True)
    for ck in (args.a, args.b):
        if ck and ck.strip().lower() != IDLE:
            try:
                rs_checkpoint.check_file(ck, cfg)
            except rs_checkpoint.VocabMismatch as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
    if args.pairs_file is not None and args.pairs_file.exists():
        try:
            pairs = PairSet.load(args.pairs_file)
        except ValueError as exc:
            print(f"error: {args.pairs_file}: {exc}", file=sys.stderr)
            return 1
        if pairs.split != split:
            ap.error(f"{args.pairs_file} holds the {pairs.split} split, not {split}")
    else:
        try:
            pairs = load_pairs(args.corpus, args.split, parts=args.parts, require=require,
                               exclude_mirror=not args.keep_mirrors, allow_final=args.final)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.pairs:
            pairs = pairs.sample(args.pairs, args.seed)
        if args.pairs_file is not None:
            pairs.save(args.pairs_file)
    print(f"engine: {cfg.describe()}\npairs: {len(pairs)} of the {split} split "
          f"[{pairs.digest()}], {args.seats} seat(s) each", flush=True)
    agents = (AgentSpec(args.a, name=args.name_a, lead=args.lead_a, noise=args.noise_a,
                        noise_seed=args.noise_seed, decide=args.decide, visibility=args.visibility,
                        presses="a" in args.presses),
              AgentSpec(args.b, name=args.name_b, lead=args.lead_b, noise=args.noise_b,
                        noise_seed=args.noise_seed, decide=args.decide, visibility=args.visibility,
                        presses="b" in args.presses))
    base = ("a" in args.base_forms, "b" in args.base_forms)
    specs = pairing_specs(pairs.pairs, agents, seed=args.seed, seats=args.seats, base_forms=base,
                          delay=delay, engine=cfg.to_json(), mem_lag=args.mem_lag,
                          max_ticks=args.max_ticks)
    t0 = time.time()
    results = run_many(specs, workers=args.workers)
    lines = []
    for name in (args.name_a, args.name_b):
        s = summarize(results, name)
        lines.append(f"{name}: score {s['score']:.1%} 95% [{s['lo']:.1%}, {s['hi']:.1%}] over "
                     f"{s['n']} ({s['wins']} W {s['draws']} D {s['losses']} L), crown diff "
                     f"{s['crown_diff']:+.2f}, plays {s['plays']:.1f}, leak {s['leak']:.2f} "
                     f"elixir, full {s['full_s']:.1f} s")
    print("\n".join(lines) + f"\n({time.time() - t0:.0f}s)")
    record = {"kind": "rs_match", "agents": [a.to_dict() for a in agents], "split": split,
              "pairs_digest": pairs.digest(), "seats": args.seats, "seed": args.seed,
              "engine": cfg.to_json(), "delay": delay, "mem_lag": args.mem_lag,
              "summary": {n: summarize(results, n) for n in (args.name_a, args.name_b)},
              "results": [r.to_dict(log=False) for r in results]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
