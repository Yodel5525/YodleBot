r"""Perception noise: what a screen reader gets wrong, applied to an engine state (E17).

    from tools import perception_noise as PN        # or `import perception_noise` (tools/ on sys.path)
    profile = PN.NoiseProfile.load("default")        # data/noise/default_prior.json; "none": no noise
    model = PN.NoiseModel.from_battle(profile, battle, fields=RR.F, sub=sub)    # once per process
    nz = PN.MatchNoise(model, PN.match_seed(tag))    # once per match: its rng and history ring
    seen = nz.observe(state, team)                   # = model.apply(state, team, nz.rng, nz.history)
    row = RR.obs_row(seen, team, sub, mem, tick + nz.tick_shift)
    nz.record(mem, team, card + 2, exec_tick)        # a play the other seat sees: missed, or late
    python tools/perception_noise.py [PROFILE ...] [--only KINDS] [--json]    # describe, check

Why (tokens-08): tools/rs_rows.obs_row and tools/rs_eval read the exact RoyaleSim state, so every
number so far assumes a perfect observer. A NoiseProfile says, kind by kind, what perception gets
wrong; NoiseModel.apply turns a true state into the state a screen reader would report. Training on
it makes the model robust, and evaluating under it, one kind at a time (NoiseProfile.only), prices
each error in win-rate points and so says what to build first.

THE STATE. apply(state, team, rng, history=None) takes a state_json dict (tests/pipeline/conftest
state_dict layout: entities as royalesim.ENTITY_FIELDS rows, players with hand / elixir_milli /
tower_hp / tower_max_hp) and returns a NEW dict of the same shape; the input, and every state the
history holds, is never changed. Unchanged nested values (an entity's facing, buffs, footprint;
spells; a player's pending list) are shared with the input: treat the result as read-only below
its entity rows and player dicts. `team` is the OBSERVER, the seat whose screen this is. `rng` is a
numpy Generator (an int seeds a fresh one); every draw comes from it, in a fixed order, so a seed and
a call sequence give one result. A profile whose noise is all zero (NoiseProfile(), "none") is the
identity: an equal copy, and no draw is taken from `rng`.

THE KINDS, in the order apply runs them (JSON group names; data/noise/default_prior.json says why
each prior value is what it is):

  board            detection delay: the whole observation (units, towers, elixir, clock) is the
                   newest history state at or before tick - lag (lag_ticks +- jitter_ticks). A
                   history that has no state within max_hold_ticks of that tick gives the present
                   state and counts "board_lag_unavailable": push a state every tick (MatchNoise.push)
                   for exact lags. Needs a history.
  new_units        a unit is hidden until it is delay_ticks (+- jitter, per unit) old: the
                   tracker's min_hits confirmation. Its age comes from the history (first push, or
                   its deploy timer at that push); without a history from its deploy timer alone.
  drop             a unit is missing from one observation: p by size bucket (size: kind 1 is
                   "building", else max_hp < small_max_hp "small", >= large_min_hp "large", else
                   "medium"), or by_class (vision class name -> p).
  forms            an evolution / hero unit reads as its base unit type: always when the detector
                   has no label for that form (unlabelled_to_base: classes.json "katacr" lacks it;
                   KataCR has never seen a hero), else with evo_to_base_p / hero_to_base_p.
  confusion        a sparse class-confusion matrix, pairs: class -> {class: p} (the target's base
                   unit type, creating card, flags and level hit points; "unknown" gives unit_type
                   and card_id -1, row token 1), and unknown_p for any class.
  side_flip        the unit's team is read the wrong way round, with p.
  position         gaussian error in the observer's SCREEN frame (sigma_x_mt, sigma_y_mt; +y down
                   the screen), flyers with flyer_sigma_mt more and a bias (flyer_dx_mt,
                   flyer_dy_mt: what the geometry's flyer offset gets wrong). The live bot is team 1,
                   whose screen is the world frame; a team-0 observer's screen is the world turned
                   half a turn. Clamped into the arena (a ground unit may land in the river:
                   tools/public_state.sanitize_for_rebuild before RSAgent.rebuilt, as live).
  hp               unit bars: a full unit has no bar and reads full, exactly. A damaged one reads
                   full with no_bar_p (no bar found, a hit flash), else fill + N(0, sigma),
                   rounded to quant, clipped to [0, 1], full_snap and above reads full
                   (scripts/vision/hpbars.FULL_SNAP). hp = fill * max_hp, at least 1.
  deploy           the deploying flag from track age: a track's age counts from its first
                   sighting, age_offset_ticks after the spawn, so the deploy timer reads that much
                   longer (also after it ran out, with a history).
  false_positives  Poisson(rate) phantom units per observation: a unit of `classes` (weights) at a
                   uniform arena position, its team the half it stands in; with no classes, a copy
                   of a reported unit near_sigma_mt away (a double box). Negative uids.
  towers           princess / king hit points (both teams) of a damaged tower: + bias_frac and
                   N(0, sigma_frac) of the tower's max, rounded to quant_hp, within [1, max]. A
                   full tower reads full (a king shows no number until damaged, a full bar snaps
                   to full) and a fallen one 0.
  elixir           the observer's own elixir: the integer digit, wrong by one with digit_error_p,
                   and a fraction with N(0, fraction_sigma_milli) error, or lost with
                   fraction_lost_p (then fraction_fill_milli). 10 reads 10000.
  clock            the tick: + bias_ticks + N(0, sigma_ticks), floored to quant_ticks. Also
                   tick_ms. MatchNoise.tick_shift is the perceived tick minus the true one, for
                   obs_row's tick argument (board lag included).
  opponent_plays   model.play_seen / MatchNoise.record: a play is missed with miss_p (spell cards:
                   spell_miss_p when set; it needs catalogue rows), else recorded late_ticks
                   (+- late_jitter_ticks) after it ran, as card token 1 ("unknown") with unknown_p.

Per-unit decisions that the tracker would hold for a track's life (its delay, form collapse,
confusion, side flip) are drawn once per (observer, uid) and kept in the history; drops, positions,
bars, phantoms and the HUD are drawn per observation. Without a history every draw is fresh.

CLASS NAMES. confusion.pairs, forms, drop.by_class and false_positives.classes name data/vision
classes (tools/vision_classes.py), so the model must be bound to the engine's unit types (and, for
pairs and phantom classes, the catalogue): NoiseModel(profile, unit_types=battle unit_types_json(),
catalogue=catalogue_json() rows, classes=ClassTable or the shipped table), or from_battle /
for_engine. Names the table or the engine does not have are skipped and listed in model.unresolved.

HOOKS (lane A wires them; this module edits neither file). The options come from add_noise_args
(--noise PROFILE: a JSON path, "default" or "none"; --noise-only KIND,...; --noise-seed N) and
profile_from_args, which gives None for no noise. None is the default and skips every hook, so
today's rows, vocab.json and numbers stay byte for byte. obs_row's own filters (E12 --visibility,
--units, --live-compat) run on the noisy state as on a true one.

  tools/rs_rows.py. NOISE is a NoiseModel each worker builds once after _setup:
  NoiseModel(profile, unit_types=rs_engine.unit_types(cfg), catalogue=cat, fields=F, sub=sub).
  In rows_for, once per match after `mem = rs_memory.PublicMemory()`:
      nz = PN.MatchNoise(NOISE, PN.match_seed(payload["source"]["replay_tag"], SEED)) if NOISE else None
  Then the 3 lines. Each `obs_row(state, team, sub, VIEW, tick)` (play, wait and E11 decision
  rows) becomes:
      seen = nz.observe(state, team) if nz else state
      row = obs_row(seen, team, sub, VIEW, tick + nz.tick_shift if nz else tick)
      row["tick"] = tick          # outcome window and trust cut stay on the true tick
  and `mem.record(team, cid + 2, tick)` becomes
      nz.record(mem, team, cid + 2, tick) if nz else mem.record(team, cid + 2, tick)
  vocab.json gets rec["perception_noise"] = PN.vocab_entry(profile, SEED). vocab_record needs a
  keyword for it, and the key is left out when there is no noise.

  tools/rs_eval.py. In RSAgent.act, the 3 lines go first, before ahead(), so --lead -N rebuilds
  from what the screen shows and rs_decide's masks see the HUD's elixir:
      noise = getattr(self, "noise", None)               # a MatchNoise, set per match by play()
      if noise is not None:
          state = noise.observe(state, team)
  and obs_row's tick becomes `tick + self.lead + (noise.tick_shift if noise is not None else 0)`.
  In play(), for the agent seat: agent.noise = PN.MatchNoise(NOISE, PN.match_seed(tag, seed)).
  agent.noise.push(state) after each state_json is needed only for board lag. The ghost's and
  the reactive opponent's plays go into the memory through
  agent.noise.record(mem, team, cid + 2, tick + runs_after). A noisy run says so in its name and
  in reports/rs_ghost.jsonl ("noise": vocab_entry).

Pure Python plus numpy; imports tools/rs_engine (ENTITY_FIELDS_EXPECTED, and royalesim only in
for_engine) and tools/vision_classes. Never imports scripts/gamestate or the memory reader.
"""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import hashlib
import json
import math
import numbers
import sys
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:          # also when imported as tools.perception_noise
    sys.path.insert(0, str(ROOT / "tools"))

import rs_engine  # noqa: E402  (ENTITY_FIELDS_EXPECTED; royalesim only in for_engine)
import vision_classes  # noqa: E402  (standard library only)

__all__ = ["SCHEMA", "PRIOR_PATH", "GROUPS", "NoiseProfile", "NoiseModel", "NoiseHistory",
           "MatchNoise", "apply", "as_profile", "match_seed", "vocab_entry", "add_noise_args",
           "profile_from_args", "load_profile"]

SCHEMA = 1
PRIOR_PATH = ROOT / "data" / "noise" / "default_prior.json"
#: --noise values that mean no noise (the hooks are skipped), and the names of the shipped prior.
NO_NOISE = frozenset({"", "none", "off", "zero", "0"})
PRIOR_NAMES = frozenset({"default", "prior", "default_prior"})

SUBTILE_PER_MILLITILE = 18                 # royalesim.SUBTILE_PER_MILLITILE; pass the engine's
ARENA_W_MT, ARENA_H_MT = 18_000, 32_000    # world millitiles
RIVER_Y_MT = 16_000                        # team 0 (blue) owns y below it
TICK_MS = 50
FRIENDLY_LEVEL = vision_classes.FRIENDLY_LEVEL
MAX_ELIXIR_MILLI = 10_000
#: Ticks a unit deploys when the class table does not say (most cards deploy for 1 s).
DEFAULT_DEPLOY_TICKS = 20
DEFAULT_HISTORY_TICKS = 64
#: confusion.pairs target that reads as no class: unit_type -1 and card_id -1 (row token 1).
UNKNOWN_CLASS = "unknown"
#: Row token of a card perception could not name (rs_memory.UNKNOWN), and the token offset.
UNKNOWN_CARD, TOKEN_OFFSET = 1, 2
#: Hit points of a phantom whose class has no level hit points in the class table.
PHANTOM_HP = 100
BUCKETS = ("small", "medium", "large", "building")
KIND_BUILDING = 1                          # entity "kind" (obs_row reads kind == 1 as building)
_PRUNE_AT = 4096                           # unit memos kept before stale ones are pruned

#: Entity columns apply reads or writes.
REQUIRED_FIELDS = ("uid", "team", "kind", "card_id", "tower_slot", "x", "y", "hp", "max_hp",
                   "flying", "deploy_ticks", "unit_type")
#: A phantom's value in each column (royalesim's "nothing" values; lists copied per row).
_PHANTOM_DEFAULTS: dict[str, Any] = {
    "uid": 0, "team": 0, "kind": 0, "card_id": -1, "tower_slot": -1, "x": 0, "y": 0, "hp": 0,
    "max_hp": 0, "radius": 0, "flying": False, "deploy_ticks": 0, "stun_ticks": 0,
    "knockback_ticks": 0, "footprint": None, "target_uid": -1, "attack_phase": 0,
    "facing": [0, 0], "shield": 0, "buffs": [], "status_flags": 0, "level": FRIENDLY_LEVEL,
    "mount_uid": -1, "charge": 0, "dest_x": -1, "dest_y": -1, "ability_ticks": 0,
    "unit_type": -1,
}


# ---------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------

# Field kinds: how a JSON value is checked and stored.
_P, _NONNEG, _REAL, _TICKS, _INT, _BOOL = "p", "nonneg", "real", "ticks", "int", "bool"
_PMAP, _WEIGHTS, _PMATRIX, _P_OR_NONE = "pmap", "weights", "pmatrix", "p_or_none"


def _f(default: Any, kind: str) -> Any:
    if isinstance(default, dict):
        return field(default_factory=dict, metadata={"kind": kind})
    return field(default=default, metadata={"kind": kind})


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{where}: a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{where}: a finite number, got {value!r}")
    return v


def _coerce(kind: str, value: Any, where: str) -> Any:
    if kind == _BOOL:
        if not isinstance(value, bool):
            raise ValueError(f"{where}: true or false, got {value!r}")
        return value
    if kind in (_PMAP, _WEIGHTS):
        if not isinstance(value, Mapping):
            raise ValueError(f"{where}: an object of name -> number, got {value!r}")
        sub = _P if kind == _PMAP else _NONNEG
        return {str(k): _coerce(sub, v, f"{where}.{k}") for k, v in value.items()
                if not str(k).startswith("_")}
    if kind == _PMATRIX:
        if not isinstance(value, Mapping):
            raise ValueError(f"{where}: an object of class -> {{class: p}}, got {value!r}")
        out = {}
        for src, row in value.items():
            if str(src).startswith("_"):
                continue
            r = _coerce(_PMAP, row, f"{where}.{src}")
            if str(src) in r:
                raise ValueError(f"{where}.{src}: a class cannot be confused with itself")
            total = sum(r.values())
            if total > 1.0 + 1e-9:
                raise ValueError(f"{where}.{src}: its confusions add up to {total:g} > 1")
            out[str(src)] = r
        return out
    if kind == _P_OR_NONE and value is None:
        return None
    v = _number(value, where)
    if kind in (_TICKS, _INT):
        if not v.is_integer():
            raise ValueError(f"{where}: a whole number of ticks, got {value!r}")
        v = int(v)
        if kind == _TICKS and v < 0:
            raise ValueError(f"{where}: >= 0, got {value!r}")
        return v
    if kind in (_P, _P_OR_NONE) and not 0.0 <= v <= 1.0:
        raise ValueError(f"{where}: a probability in [0, 1], got {value!r}")
    if kind == _NONNEG and v < 0:
        raise ValueError(f"{where}: >= 0, got {value!r}")
    return v


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(value[k]) for k in sorted(value)}
    return value


class _Group:
    """A JSON group of a NoiseProfile: from_dict / to_dict / active."""

    def active(self) -> bool:            # pragma: no cover - every group overrides it
        raise NotImplementedError

    def check(self, where: str) -> None:
        """Cross-field checks (after the per-field ones)."""

    def to_dict(self) -> dict:
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, data: Any, where: str):
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{where}: an object, got {data!r}")
        known = {f.name: f for f in dataclasses.fields(cls)}
        kwargs = {}
        for key, value in data.items():
            key = str(key)
            if key.startswith("_"):
                continue                    # "_why", "_comment": notes, not settings
            if key not in known:
                raise ValueError(f"{where}: unknown key {key!r} (known: {', '.join(known)})")
            kwargs[key] = _coerce(known[key].metadata["kind"], value, f"{where}.{key}")
        group = cls(**kwargs)
        group.check(where)
        return group


@dataclass(frozen=True)
class Drop(_Group):
    """Units missing from an observation, by size bucket (Size) or by vision class."""
    small: float = _f(0.0, _P)
    medium: float = _f(0.0, _P)
    large: float = _f(0.0, _P)
    building: float = _f(0.0, _P)
    by_class: dict = _f({}, _PMAP)

    def active(self) -> bool:
        return any(getattr(self, b) > 0 for b in BUCKETS) or any(v > 0 for v in self.by_class.values())


@dataclass(frozen=True)
class Size(_Group):
    """The size buckets of Drop, by the entity's max_hp (level-scaled); kind 1 is "building"."""
    small_max_hp: float = _f(400.0, _NONNEG)
    large_min_hp: float = _f(2500.0, _NONNEG)

    def active(self) -> bool:
        return False                         # a setting, not a noise

    def check(self, where: str) -> None:
        if self.small_max_hp > self.large_min_hp:
            raise ValueError(f"{where}: small_max_hp {self.small_max_hp:g} is above large_min_hp "
                             f"{self.large_min_hp:g}")


@dataclass(frozen=True)
class FalsePositives(_Group):
    """Phantom units: Poisson(rate) per observation."""
    rate: float = _f(0.0, _NONNEG)
    classes: dict = _f({}, _WEIGHTS)
    near_sigma_mt: float = _f(0.0, _NONNEG)

    def active(self) -> bool:
        return self.rate > 0

    def check(self, where: str) -> None:
        if self.classes and not sum(self.classes.values()) > 0:
            raise ValueError(f"{where}.classes: the weights add up to 0")


@dataclass(frozen=True)
class Position(_Group):
    """Position error in the observer's screen frame, millitiles."""
    sigma_x_mt: float = _f(0.0, _NONNEG)
    sigma_y_mt: float = _f(0.0, _NONNEG)
    flyer_dx_mt: float = _f(0.0, _REAL)
    flyer_dy_mt: float = _f(0.0, _REAL)
    flyer_sigma_mt: float = _f(0.0, _NONNEG)

    def active(self) -> bool:
        return any(getattr(self, f.name) != 0 for f in dataclasses.fields(self))


@dataclass(frozen=True)
class SideFlip(_Group):
    p: float = _f(0.0, _P)

    def active(self) -> bool:
        return self.p > 0


@dataclass(frozen=True)
class Confusion(_Group):
    """pairs: vision class -> {class or "unknown": p}; unknown_p: any class reads as unknown."""
    unknown_p: float = _f(0.0, _P)
    pairs: dict = _f({}, _PMATRIX)

    def active(self) -> bool:
        return self.unknown_p > 0 or self.has_pairs()

    def has_pairs(self) -> bool:
        return any(p > 0 for row in self.pairs.values() for p in row.values())


@dataclass(frozen=True)
class Forms(_Group):
    evo_to_base_p: float = _f(0.0, _P)
    hero_to_base_p: float = _f(0.0, _P)
    unlabelled_to_base: bool = _f(False, _BOOL)

    def active(self) -> bool:
        return self.evo_to_base_p > 0 or self.hero_to_base_p > 0 or self.unlabelled_to_base


@dataclass(frozen=True)
class Hp(_Group):
    """Unit HP bars: no_bar_p, sigma and quant (fractions of the bar), full_snap (0: off)."""
    no_bar_p: float = _f(0.0, _P)
    sigma: float = _f(0.0, _NONNEG)
    quant: float = _f(0.0, _P)
    full_snap: float = _f(0.0, _P)

    def active(self) -> bool:
        return any(getattr(self, f.name) > 0 for f in dataclasses.fields(self))


@dataclass(frozen=True)
class Deploy(_Group):
    age_offset_ticks: int = _f(0, _TICKS)

    def active(self) -> bool:
        return self.age_offset_ticks > 0


@dataclass(frozen=True)
class NewUnits(_Group):
    delay_ticks: int = _f(0, _TICKS)
    jitter_ticks: int = _f(0, _TICKS)

    def active(self) -> bool:
        return self.delay_ticks > 0 or self.jitter_ticks > 0


@dataclass(frozen=True)
class Board(_Group):
    lag_ticks: int = _f(0, _TICKS)
    jitter_ticks: int = _f(0, _TICKS)
    max_hold_ticks: int = _f(2, _TICKS)

    def active(self) -> bool:
        return self.lag_ticks > 0 or self.jitter_ticks > 0


@dataclass(frozen=True)
class Clock(_Group):
    bias_ticks: int = _f(0, _INT)
    sigma_ticks: float = _f(0.0, _NONNEG)
    quant_ticks: int = _f(0, _TICKS)

    def active(self) -> bool:
        return self.bias_ticks != 0 or self.sigma_ticks > 0 or self.quant_ticks > 1


@dataclass(frozen=True)
class Elixir(_Group):
    fraction_sigma_milli: float = _f(0.0, _NONNEG)
    fraction_lost_p: float = _f(0.0, _P)
    fraction_fill_milli: int = _f(0, _TICKS)
    digit_error_p: float = _f(0.0, _P)

    def active(self) -> bool:
        return self.fraction_sigma_milli > 0 or self.fraction_lost_p > 0 or self.digit_error_p > 0

    def check(self, where: str) -> None:
        if self.fraction_fill_milli > 999:
            raise ValueError(f"{where}.fraction_fill_milli: 0..999, got {self.fraction_fill_milli}")


@dataclass(frozen=True)
class Towers(_Group):
    sigma_frac: float = _f(0.0, _NONNEG)
    bias_frac: float = _f(0.0, _REAL)
    quant_hp: int = _f(0, _TICKS)

    def active(self) -> bool:
        return self.sigma_frac > 0 or self.bias_frac != 0 or self.quant_hp > 1


@dataclass(frozen=True)
class OpponentPlays(_Group):
    miss_p: float = _f(0.0, _P)
    spell_miss_p: float | None = _f(None, _P_OR_NONE)
    late_ticks: int = _f(0, _TICKS)
    late_jitter_ticks: int = _f(0, _TICKS)
    unknown_p: float = _f(0.0, _P)

    def active(self) -> bool:
        return (self.miss_p > 0 or bool(self.spell_miss_p) or self.late_ticks > 0
                or self.late_jitter_ticks > 0 or self.unknown_p > 0)


#: JSON group -> its class, in the order the profile file lists them.
_GROUP_TYPES: dict[str, type] = {
    "board": Board, "new_units": NewUnits, "drop": Drop, "size": Size, "forms": Forms,
    "confusion": Confusion, "side_flip": SideFlip, "position": Position, "hp": Hp,
    "deploy": Deploy, "false_positives": FalsePositives, "towers": Towers, "elixir": Elixir,
    "clock": Clock, "opponent_plays": OpponentPlays,
}
#: The noise kinds (every group but the size setting), for only() / without() / --noise-only.
GROUPS: tuple[str, ...] = tuple(g for g in _GROUP_TYPES if g != "size")


@dataclass(frozen=True)
class NoiseProfile:
    """Every noise kind's settings (module doc). NoiseProfile() is all zero: the identity."""
    name: str = "zero"
    description: str = ""
    board: Board = field(default_factory=Board)
    new_units: NewUnits = field(default_factory=NewUnits)
    drop: Drop = field(default_factory=Drop)
    size: Size = field(default_factory=Size)
    forms: Forms = field(default_factory=Forms)
    confusion: Confusion = field(default_factory=Confusion)
    side_flip: SideFlip = field(default_factory=SideFlip)
    position: Position = field(default_factory=Position)
    hp: Hp = field(default_factory=Hp)
    deploy: Deploy = field(default_factory=Deploy)
    false_positives: FalsePositives = field(default_factory=FalsePositives)
    towers: Towers = field(default_factory=Towers)
    elixir: Elixir = field(default_factory=Elixir)
    clock: Clock = field(default_factory=Clock)
    opponent_plays: OpponentPlays = field(default_factory=OpponentPlays)

    # ------------------------------------------------------------ JSON
    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str | None = None) -> "NoiseProfile":
        """A profile from its JSON object. Keys starting with "_" are notes and are skipped;
        any other unknown key, a value out of range or a newer schema is a ValueError."""
        if not isinstance(data, Mapping):
            raise ValueError(f"a noise profile is a JSON object, got {type(data).__name__}")
        schema = data.get("schema", SCHEMA)
        if isinstance(schema, bool) or not isinstance(schema, int) or schema > SCHEMA:
            raise ValueError(f"noise profile schema {schema!r}: this code reads schema {SCHEMA}")
        groups = {}
        for key in data:
            key = str(key)
            if key.startswith("_") or key in ("schema", "name", "description"):
                continue
            if key not in _GROUP_TYPES:
                raise ValueError(f"noise profile: unknown group {key!r} (groups: "
                                 f"{', '.join(_GROUP_TYPES)})")
            groups[key] = _GROUP_TYPES[key].from_dict(data[key], key)
        nm = data.get("name", name if name is not None else "profile")
        desc = data.get("description", "")
        if not isinstance(nm, str) or not isinstance(desc, str):
            raise ValueError("noise profile: name and description are strings")
        return cls(name=nm, description=desc, **groups)

    def to_dict(self) -> dict:
        """The complete profile, every setting written out (schema, name, description, groups)."""
        out: dict[str, Any] = {"schema": SCHEMA, "name": self.name, "description": self.description}
        for g in _GROUP_TYPES:
            out[g] = getattr(self, g).to_dict()
        return out

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def digest(self) -> str:
        """16 hex digits of the settings (name and description left out)."""
        body = {g: getattr(self, g).to_dict() for g in _GROUP_TYPES}
        text = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def load(cls, source: str | Path | None) -> "NoiseProfile":
        """A JSON file, "default" (PRIOR_PATH) or "none" (NoiseProfile())."""
        if source is None or (isinstance(source, str) and source.strip().lower() in NO_NOISE):
            return cls()
        if isinstance(source, str) and source.strip().lower() in PRIOR_NAMES:
            source = PRIOR_PATH
        path = Path(source)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"noise profile {path} does not exist") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"noise profile {path}: not JSON ({exc})") from None
        try:
            return cls.from_dict(data, name=path.stem)
        except ValueError as exc:
            raise ValueError(f"noise profile {path}: {exc}") from None

    # ------------------------------------------------------------ kinds
    def active_groups(self) -> list[str]:
        return [g for g in GROUPS if getattr(self, g).active()]

    def is_identity(self) -> bool:
        return not self.active_groups()

    def _pick(self, groups: Iterable[str], keep: bool) -> "NoiseProfile":
        names = set()
        for g in groups:
            for part in str(g).split(","):
                part = part.strip()
                if not part:
                    continue
                if part not in GROUPS:
                    raise ValueError(f"no noise kind {part!r} (kinds: {', '.join(GROUPS)})")
                names.add(part)
        changes = {g: _GROUP_TYPES[g]() for g in GROUPS if (g in names) != keep}
        tag = ("only " if keep else "without ") + ",".join(sorted(names))
        return dataclasses.replace(self, name=f"{self.name} ({tag})", **changes)

    def only(self, *groups: str) -> "NoiseProfile":
        """This profile with every other noise kind switched off (to price one kind)."""
        return self._pick(groups, True)

    def without(self, *groups: str) -> "NoiseProfile":
        """This profile with these noise kinds switched off."""
        return self._pick(groups, False)

    def describe(self) -> str:
        """One line per active kind, for the console."""
        active = self.active_groups()
        if not active:
            return f"noise {self.name}: none (perfect observer)"
        lines = [f"noise {self.name} [{self.digest()}]: {', '.join(active)}"]
        for g in active:
            vals = getattr(self, g).to_dict()
            text = ", ".join(f"{k} {v}" for k, v in vals.items()
                             if not isinstance(v, dict) or v)
            lines.append(f"  {g:15s} {text}")
        return "\n".join(lines)


def as_profile(profile: Any) -> NoiseProfile:
    """A NoiseProfile from a profile, its JSON dict, a path / "default" / "none", or None."""
    if isinstance(profile, NoiseProfile):
        return profile
    if profile is None or isinstance(profile, (str, Path)):
        return NoiseProfile.load(profile)
    if isinstance(profile, Mapping):
        return NoiseProfile.from_dict(profile)
    raise TypeError(f"a noise profile: NoiseProfile, dict, path or None, got {type(profile).__name__}")


load_profile = NoiseProfile.load


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

class UnitMemo:
    """What the history knows of one uid: its estimated spawn tick (None: on the board before the
    history began), the tick its deploy timer runs out, the last tick it was pushed, and the
    per-(observer, kind) decisions held for its life."""
    __slots__ = ("spawn", "deploy_end", "last", "sticky")

    def __init__(self, spawn: int | None, deploy_end: int | None, last: int):
        self.spawn = spawn
        self.deploy_end = deploy_end
        self.last = last
        self.sticky: dict[tuple[int, str], Any] = {}

    def __repr__(self) -> str:
        return f"UnitMemo(spawn={self.spawn}, deploy_end={self.deploy_end}, last={self.last})"


class NoiseHistory:
    """One match's ring of true states (ticks ascending, max_ticks deep) and unit memos. The
    states are kept by reference: do not change a state after pushing it."""

    def __init__(self, max_ticks: int = DEFAULT_HISTORY_TICKS):
        if int(max_ticks) < 0:
            raise ValueError(f"NoiseHistory max_ticks must be >= 0, got {max_ticks}")
        self.max_ticks = int(max_ticks)
        self.ticks: list[int] = []
        self.states: list[dict] = []
        self.units: dict[Any, UnitMemo] = {}

    def __len__(self) -> int:
        return len(self.ticks)

    @property
    def last_tick(self) -> int | None:
        return self.ticks[-1] if self.ticks else None

    def at(self, tick: int) -> dict | None:
        """The newest state with tick <= `tick`, or None."""
        k = bisect.bisect_right(self.ticks, int(tick))
        return self.states[k - 1] if k else None


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class _Proto(NamedTuple):
    """What a unit of a class reads as: its base unit type, creating card and flags."""
    unit_type: int
    card_id: int
    kind: int
    flying: bool
    max_hp: int | None


def _rint(x: float) -> int:
    """Round half up (not Python's half to even), so hand-worked values are plain."""
    return int(math.floor(float(x) + 0.5))


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, _rint(x)))


def _count(stats: MutableMapping | None, key: str, n: int = 1) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + n


def _jitter(rng: Any, j: int) -> int:
    return int(rng.integers(-j, j + 1)) if j > 0 else 0


def _as_rng(rng: Any) -> Any:
    if isinstance(rng, bool):
        raise TypeError("rng: a numpy Generator or an int seed, not a bool")
    if isinstance(rng, numbers.Integral):
        return np.random.default_rng(int(rng))
    if rng is None or not all(hasattr(rng, a) for a in ("random", "normal", "integers", "poisson")):
        raise TypeError(f"rng: a numpy Generator (np.random.default_rng(seed)), got {rng!r}")
    return rng


def _team(team: Any) -> int:
    if isinstance(team, bool) or not isinstance(team, numbers.Integral) or int(team) not in (0, 1):
        raise ValueError(f"team must be 0 or 1, got {team!r}")
    return int(team)


def _tick_of(state: Mapping) -> int:
    try:
        return int(state["tick"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("a state needs an integer 'tick' (state_json has one)") from None


def _field_index(fields: Mapping[str, int] | Sequence[str] | None) -> dict[str, int]:
    if fields is None:
        fields = rs_engine.ENTITY_FIELDS_EXPECTED
    if isinstance(fields, Mapping):
        index = {str(k): int(v) for k, v in fields.items()}
    else:
        index = {str(name): i for i, name in enumerate(fields)}
    missing = [name for name in REQUIRED_FIELDS if name not in index]
    if missing:
        raise ValueError(f"entity fields lack {missing}: pass royalesim.ENTITY_FIELDS "
                         "(rs_engine.field_index, or rs_rows.F after _setup)")
    return index


def _card_name(row: Any) -> str:
    return str(row[0]) if isinstance(row, (list, tuple)) else str(row)


def _copy_state(state: Mapping, entities: bool = True) -> dict:
    """A copy whose top level, player dicts (and their lists), entity rows (unless `entities` is
    False: the caller builds them) and spell rows are new; deeper values are shared."""
    out = dict(state)
    if isinstance(state.get("players"), list):
        out["players"] = [{k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}
                          for p in state["players"]]
    if entities and isinstance(state.get("entities"), list):
        out["entities"] = [list(e) for e in state["entities"]]
    for key in ("spells", "projectiles"):
        if isinstance(state.get(key), list):
            out[key] = [list(s) if isinstance(s, list) else s for s in state[key]]
    return out


class NoiseModel:
    """A NoiseProfile bound to an engine's entity layout and, for the kinds that name classes,
    its unit types, catalogue and the vision class table (module doc, CLASS NAMES)."""

    def __init__(self, profile: Any = None, *, unit_types: Sequence[str] | None = None,
                 catalogue: Sequence[Any] | None = None, classes: Any = None,
                 fields: Mapping[str, int] | Sequence[str] | None = None,
                 sub: int = SUBTILE_PER_MILLITILE, level: int = FRIENDLY_LEVEL):
        self.profile = as_profile(profile)
        self.F = _field_index(fields)
        self.sub = int(sub)
        if self.sub <= 0:
            raise ValueError(f"sub (sub-tiles per millitile) must be > 0, got {sub}")
        self.level = int(level)
        self.identity = self.profile.is_identity()
        self.unit_types = None if unit_types is None else tuple(str(n) for n in unit_types)
        rows = None if catalogue is None else list(catalogue)
        self.card_names = None if rows is None else tuple(_card_name(r) for r in rows)
        self.spell_cards = None
        if rows is not None and rows and all(isinstance(r, (list, tuple)) and len(r) > 8 for r in rows):
            self.spell_cards = frozenset(i for i, r in enumerate(rows) if r[8] == "SPELL")
        self.unresolved: list[str] = []
        P = self.profile
        named = [what for what, on in (
            ("confusion.pairs", P.confusion.has_pairs()), ("forms", P.forms.active()),
            ("drop.by_class", any(v > 0 for v in P.drop.by_class.values())),
            ("false_positives.classes", P.false_positives.active() and bool(P.false_positives.classes)),
        ) if on]
        if named and self.unit_types is None:
            raise ValueError(f"{', '.join(named)} name vision classes: bind the model to the "
                             "engine's unit_types_json() (NoiseModel(..., unit_types=...), "
                             "from_battle or for_engine)")
        carded = [w for w in named if w in ("confusion.pairs", "false_positives.classes")]
        if carded and self.card_names is None:
            raise ValueError(f"{', '.join(carded)} give units a creating card: pass the "
                             "catalogue (catalogue_json() rows or card names)")
        if P.opponent_plays.spell_miss_p is not None and self.spell_cards is None:
            raise ValueError("opponent_plays.spell_miss_p tells spells apart: pass the catalogue "
                             "rows (catalogue_json(), the card kind in column 8)")
        self.classes = None
        if self.unit_types is not None and (classes is not None or not self.identity):
            self.classes = classes if classes is not None else vision_classes.load()
        self._protos: dict[str, _Proto | None] = {}
        self._prepare()

    # ------------------------------------------------------------ construction
    @classmethod
    def from_battle(cls, profile: Any, battle: Any, **kwargs: Any) -> "NoiseModel":
        """Bound to a Battle's unit_types_json() and catalogue_json()."""
        return cls(profile, unit_types=json.loads(battle.unit_types_json()),
                   catalogue=json.loads(battle.catalogue_json()), **kwargs)

    @classmethod
    def for_engine(cls, profile: Any, cfg: Any = None, rs: Any = None, **kwargs: Any) -> "NoiseModel":
        """Bound to a Battle under rs_engine config `cfg` (imports royalesim)."""
        battle = rs_engine.make_battle(rs_engine.resolve(cfg).undelayed(), rs)
        if "fields" not in kwargs:
            kwargs["fields"] = rs_engine.field_index(rs)
        return cls.from_battle(profile, battle, **kwargs)

    def _prepare(self) -> None:
        P = self.profile
        self._cls: list[Any] = []
        self._form: list[int] = []
        self._deploy_total: list[int] = []
        self._base: list[int | None] = []
        self._labelled: list[bool] = []
        self._tower: list[bool] = []
        self._name_index: dict[str, int] = {}
        self._class_form_index: dict[tuple[int, int], int] = {}
        self._pairs: dict[int, tuple[tuple[float, _Proto | None], ...]] = {}
        self._drop_class: dict[int, float] = {}
        self._fp_classes: tuple[tuple[float, _Proto], ...] = ()
        if self.classes is None:
            return
        look = self.classes.lookup(list(self.unit_types))
        for u, name in enumerate(self.unit_types):
            self._name_index.setdefault(name, u)
            cid, form = int(look.class_ids[u]), int(look.forms[u])
            if cid >= 0:
                self._class_form_index.setdefault((cid, form), u)
        for u in range(len(self.unit_types)):
            cid, form = int(look.class_ids[u]), int(look.forms[u])
            c = self.classes.classes[cid] if cid >= 0 else None
            self._cls.append(c)
            self._form.append(form)
            ms = int(look.deploy_time_ms[u]) if c is not None else 0
            self._deploy_total.append(-(-ms // TICK_MS) if ms > 0 else DEFAULT_DEPLOY_TICKS)
            self._tower.append(c is not None and c.kind == vision_classes.KIND_TOWER)
            self._labelled.append(c is not None and form in c.katacr)
            base = None
            if c is not None and form != c.base_form:
                b = self._unit_index(c.form(c.base_form).unit, c.id, c.base_form)
                base = b if b is not None and b != u else None
            self._base.append(base)
        if P.confusion.has_pairs():
            for src in sorted(P.confusion.pairs):
                c = self.classes.get(src)
                if c is None or c.name != src:
                    self.unresolved.append(f"confusion.pairs: {src!r} is not a class")
                    continue
                cum, row = 0.0, []
                for tgt in sorted(P.confusion.pairs[src]):
                    p = P.confusion.pairs[src][tgt]
                    if p <= 0:
                        continue
                    proto = None
                    if tgt != UNKNOWN_CLASS:
                        proto = self._proto(tgt)
                        if proto is None:
                            self.unresolved.append(f"confusion.pairs: {src} -> {tgt!r} has no unit "
                                                   "type in this engine")
                            continue
                    cum += p
                    row.append((cum, proto))
                if row:
                    self._pairs[c.id] = tuple(row)
        for name in sorted(P.drop.by_class):
            c = self.classes.get(name)
            if c is None or c.name != name:
                self.unresolved.append(f"drop.by_class: {name!r} is not a class")
                continue
            self._drop_class[c.id] = P.drop.by_class[name]
        if P.false_positives.active() and P.false_positives.classes:
            weights = []
            for name in sorted(P.false_positives.classes):
                w = P.false_positives.classes[name]
                if w <= 0:
                    continue
                proto = self._proto(name)
                if proto is None:
                    self.unresolved.append(f"false_positives.classes: {name!r} has no unit type "
                                           "in this engine")
                    continue
                weights.append((w, proto))
            total = sum(w for w, _ in weights)
            cum, table = 0.0, []
            for i, (w, proto) in enumerate(weights):
                cum += w / total
                table.append((1.0 if i == len(weights) - 1 else cum, proto))
            self._fp_classes = tuple(table)

    def _unit_index(self, name: str, cid: int, form: int) -> int | None:
        u = self._name_index.get(name)
        return u if u is not None else self._class_form_index.get((cid, form))

    def _proto(self, name: str) -> _Proto | None:
        """A unit of class `name` (its base form) as this engine writes it, or None."""
        if name in self._protos:
            return self._protos[name]
        proto = None
        c = self.classes.get(name)
        if (c is not None and c.name == name and not getattr(c, "retired", False)
                and c.kind != vision_classes.KIND_TOWER):
            info = c.form(c.base_form)
            u = self._unit_index(info.unit, c.id, c.base_form)
            if u is not None:
                card = (self.classes.card_id_for(c.name, self.card_names)
                        if self.card_names is not None else -1)
                proto = _Proto(u, int(card), KIND_BUILDING if info.building else 0,
                               bool(info.flying), self.classes.hitpoints(c.name, c.base_form, self.level))
        self._protos[name] = proto
        return proto

    # ------------------------------------------------------------ per unit type
    def _known(self, u: Any) -> bool:
        return (self.classes is not None and isinstance(u, numbers.Integral) and not isinstance(u, bool)
                and 0 <= int(u) < len(self._cls))

    def _class_of(self, u: Any) -> Any:
        return self._cls[int(u)] if self._known(u) else None

    def deploy_total(self, u: Any) -> int:
        """Ticks a unit of type u deploys for (class table; DEFAULT_DEPLOY_TICKS otherwise)."""
        return self._deploy_total[int(u)] if self._known(u) else DEFAULT_DEPLOY_TICKS

    def bucket(self, row: Sequence) -> str:
        """The Drop size bucket of an entity row."""
        F, S = self.F, self.profile.size
        if row[F["kind"]] == KIND_BUILDING:
            return "building"
        mx = row[F["max_hp"]]
        if mx < S.small_max_hp:
            return "small"
        if mx >= S.large_min_hp:
            return "large"
        return "medium"

    # ------------------------------------------------------------ history
    def _note_unit(self, history: NoiseHistory, row: Sequence, tick: int, prev: int | None) -> UnitMemo:
        F = self.F
        k = int(row[F["deploy_ticks"]])
        total = self.deploy_total(row[F["unit_type"]])
        if k > 0:
            spawn = tick - max(0, total - k)
            if prev is not None:
                spawn = max(spawn, prev + 1)
            deploy_end = tick + k
        elif prev is not None:
            spawn = max(prev + 1, tick - total)
            deploy_end = spawn + total
        else:
            spawn, deploy_end = None, None          # there before the history began: old
        memo = history.units[row[F["uid"]]] = UnitMemo(spawn, deploy_end, tick)
        return memo

    def push(self, history: NoiseHistory, state: Mapping) -> bool:
        """Add a true state to the history (apply does it for the state it is given). A state of
        the history's last tick is skipped (False); an earlier tick is a ValueError: one history
        per match, fed in tick order."""
        tick = _tick_of(state)
        prev = history.last_tick
        if prev is not None:
            if tick < prev:
                raise ValueError(f"NoiseHistory: a state of tick {tick} after tick {prev}; keep "
                                 "one history per match and push in tick order")
            if tick == prev:
                return False
        F = self.F
        for e in state["entities"]:
            if e[F["tower_slot"]] >= 0 or e[F["hp"]] <= 0:
                continue
            memo = history.units.get(e[F["uid"]])
            if memo is None:
                memo = self._note_unit(history, e, tick, prev)
            memo.last = tick
        history.ticks.append(tick)
        history.states.append(state)
        while len(history.ticks) > 1 and history.ticks[0] < tick - history.max_ticks:
            history.ticks.pop(0)
            history.states.pop(0)
        if len(history.units) > _PRUNE_AT:
            old = tick - history.max_ticks
            for uid in [uid for uid, m in history.units.items() if m.last < old]:
                del history.units[uid]
        return True

    @staticmethod
    def _sticky(memo: UnitMemo | None, team: int, key: str, draw: Callable[[], Any]) -> Any:
        if memo is None:
            return draw()
        k = (team, key)
        if k not in memo.sticky:
            memo.sticky[k] = draw()
        return memo.sticky[k]

    def _age(self, row: Sequence, memo: UnitMemo | None, now: int) -> float:
        if memo is not None:
            return math.inf if memo.spawn is None else now - memo.spawn
        k = int(row[self.F["deploy_ticks"]])
        if k > 0:
            return max(0, self.deploy_total(row[self.F["unit_type"]]) - k)
        return math.inf

    # ------------------------------------------------------------ apply
    def apply(self, state: Mapping, team: int, rng: Any, history: NoiseHistory | None = None, *,
              stats: MutableMapping | None = None) -> dict:
        """The state `team`'s screen reader would report for the true `state` (module doc). With a
        history, `state` is pushed into it first. `stats` (a dict or Counter) gets counts."""
        team = _team(team)
        if history is not None:
            self.push(history, state)
        _count(stats, "observations")
        if self.identity:
            return _copy_state(state)
        rng = _as_rng(rng)
        P = self.profile
        base, now = state, _tick_of(state)
        if P.board.active():
            if history is None:
                raise ValueError("board.lag_ticks (detection delay) reads the past: pass a "
                                 "NoiseHistory (MatchNoise keeps one per match)")
            lag = max(0, P.board.lag_ticks + _jitter(rng, P.board.jitter_ticks))
            if lag:
                target = now - lag
                old = history.at(target)
                if old is None or target - _tick_of(old) > P.board.max_hold_ticks:
                    _count(stats, "board_lag_unavailable")
                else:
                    base, now = old, _tick_of(old)
                    _count(stats, "board_lag")
        out = _copy_state(base, entities=False)
        F = self.F
        ents, units = [], []
        for e in base["entities"]:
            row = list(e)
            if (row[F["tower_slot"]] >= 0 or row[F["hp"]] <= 0
                    or (self._known(row[F["unit_type"]]) and self._tower[int(row[F["unit_type"]])])):
                ents.append(row)
                continue
            _count(stats, "units")
            row = self._unit(row, team, rng, history, now, stats)
            if row is not None:
                ents.append(row)
                units.append(row)
        if P.false_positives.active():
            ents.extend(self._phantoms(units, rng, stats))
        out["entities"] = ents
        if P.towers.active():
            for p in out["players"]:
                self._towers(p, rng)
        if P.elixir.active():
            self._elixir(out["players"][team], rng, stats)
        if P.clock.active():
            C = P.clock
            t = now + C.bias_ticks + (_rint(rng.normal(0.0, C.sigma_ticks)) if C.sigma_ticks > 0 else 0)
            if C.quant_ticks > 1:
                t = (t // C.quant_ticks) * C.quant_ticks
            now = max(0, t)
        if out.get("tick") != now:
            out["tick"] = now
            if "tick_ms" in out:
                out["tick_ms"] = now * TICK_MS
        return out

    def _unit(self, row: list, team: int, rng: Any, history: NoiseHistory | None, now: int,
              stats: MutableMapping | None) -> list | None:
        F, P = self.F, self.profile
        memo = None
        if history is not None:
            memo = history.units.get(row[F["uid"]])
            if memo is None:                        # a state that was not pushed (same tick)
                memo = self._note_unit(history, row, now, None)
        if P.new_units.active():
            N = P.new_units
            d = self._sticky(memo, team, "delay",
                             lambda: max(0, N.delay_ticks + _jitter(rng, N.jitter_ticks)))
            if self._age(row, memo, now) < d:
                _count(stats, "hidden_new")
                return None
        cls = self._class_of(row[F["unit_type"]])
        p = self._drop_class.get(cls.id) if cls is not None else None
        if p is None:
            p = getattr(P.drop, self.bucket(row))
        if p > 0 and rng.random() < p:
            _count(stats, "dropped")
            return None
        if P.forms.active():
            self._collapse(row, team, rng, memo, stats)
        if P.confusion.active():
            self._confuse(row, cls, team, rng, memo, stats)
        if P.side_flip.active():
            q = P.side_flip.p
            if self._sticky(memo, team, "flip", lambda: bool(rng.random() < q)):
                row[F["team"]] = 1 - int(row[F["team"]])
                _count(stats, "flipped")
        if P.position.active():
            self._move(row, team, rng)
        if P.hp.active():
            self._hp_bar(row, rng, stats)
        if P.deploy.active():
            row[F["deploy_ticks"]] = self._deploy_estimate(row, memo, now)
        return row

    def _set_max_hp(self, row: list, new_max: int) -> None:
        F = self.F
        mx, hp = row[F["max_hp"]], row[F["hp"]]
        frac = hp / mx if mx > 0 else 1.0
        row[F["max_hp"]] = int(new_max)
        row[F["hp"]] = max(1, min(int(new_max), _rint(frac * new_max)))

    def _collapse(self, row: list, team: int, rng: Any, memo: UnitMemo | None,
                  stats: MutableMapping | None) -> None:
        F, Fm = self.F, self.profile.forms
        u = row[F["unit_type"]]
        if not self._known(u):
            return
        u = int(u)
        form, base = self._form[u], self._base[u]
        if base is None:
            return
        if Fm.unlabelled_to_base and not self._labelled[u]:
            go = True
        else:
            p = Fm.evo_to_base_p if form == vision_classes.FORM_EVO else Fm.hero_to_base_p
            go = p > 0 and self._sticky(memo, team, "form", lambda: bool(rng.random() < p))
        if not go:
            return
        c = self._cls[u]
        info = c.form(c.base_form)
        row[F["unit_type"]] = base
        row[F["flying"]] = bool(info.flying)
        hp = self.classes.hitpoints(c.name, c.base_form, self.level)
        if hp:
            self._set_max_hp(row, hp)
        _count(stats, "collapsed")

    def _confuse(self, row: list, cls: Any, team: int, rng: Any, memo: UnitMemo | None,
                 stats: MutableMapping | None) -> None:
        F, C = self.F, self.profile.confusion
        pairs = self._pairs.get(cls.id) if cls is not None else None

        def draw():
            if pairs:
                r = rng.random()
                for cum, proto in pairs:
                    if r < cum:
                        return ("class", proto)
            if C.unknown_p > 0 and rng.random() < C.unknown_p:
                return ("class", None)
            return None

        hit = self._sticky(memo, team, "class", draw)
        if hit is None:
            return
        proto = hit[1]
        if proto is None:
            row[F["unit_type"]] = -1
            row[F["card_id"]] = -1
            _count(stats, "unknown")
            return
        row[F["unit_type"]] = proto.unit_type
        row[F["card_id"]] = proto.card_id
        row[F["kind"]] = proto.kind
        row[F["flying"]] = proto.flying
        if proto.max_hp:
            self._set_max_hp(row, proto.max_hp)
        _count(stats, "confused")

    def _move(self, row: list, team: int, rng: Any) -> None:
        F, Q, s = self.F, self.profile.position, self.sub
        dx = rng.normal(0.0, Q.sigma_x_mt) if Q.sigma_x_mt > 0 else 0.0
        dy = rng.normal(0.0, Q.sigma_y_mt) if Q.sigma_y_mt > 0 else 0.0
        if row[F["flying"]]:
            if Q.flyer_sigma_mt > 0:
                dx += rng.normal(0.0, Q.flyer_sigma_mt)
                dy += rng.normal(0.0, Q.flyer_sigma_mt)
            dx += Q.flyer_dx_mt
            dy += Q.flyer_dy_mt
        if dx == 0 and dy == 0:
            return
        sign = 1 if team == 1 else -1               # observer's screen -> world
        row[F["x"]] = _clamp_int(row[F["x"]] + sign * dx * s, 0, ARENA_W_MT * s)
        row[F["y"]] = _clamp_int(row[F["y"]] + sign * dy * s, 0, ARENA_H_MT * s)

    def _hp_bar(self, row: list, rng: Any, stats: MutableMapping | None) -> None:
        F, H = self.F, self.profile.hp
        mx, hp = row[F["max_hp"]], row[F["hp"]]
        if mx <= 0 or hp >= mx:
            return                                  # full: no bar on screen, reads full
        if H.no_bar_p > 0 and rng.random() < H.no_bar_p:
            row[F["hp"]] = mx
            _count(stats, "no_bar")
            return
        frac = hp / mx
        f = frac
        if H.sigma > 0:
            f += rng.normal(0.0, H.sigma)
        if H.quant > 0:
            f = H.quant * _rint(f / H.quant)
        f = min(1.0, max(0.0, f))
        if H.full_snap > 0 and f >= H.full_snap:
            f = 1.0
        if f != frac:
            row[F["hp"]] = max(1, min(int(mx), _rint(f * mx)))

    def _deploy_estimate(self, row: Sequence, memo: UnitMemo | None, now: int) -> int:
        off = self.profile.deploy.age_offset_ticks
        k = int(row[self.F["deploy_ticks"]])
        if k > 0:
            return k + off
        if memo is not None and memo.deploy_end is not None:
            return max(0, memo.deploy_end + off - now)
        return 0

    def _phantom_row(self) -> list:
        width = max(self.F.values()) + 1
        row: list[Any] = [0] * width
        for name, i in self.F.items():
            v = _PHANTOM_DEFAULTS.get(name, 0)
            row[i] = list(v) if isinstance(v, list) else v
        if "level" in self.F:
            row[self.F["level"]] = self.level
        return row

    def _phantoms(self, units: Sequence[list], rng: Any, stats: MutableMapping | None) -> list[list]:
        F, Q, s = self.F, self.profile.false_positives, self.sub
        out = []
        for k in range(int(rng.poisson(Q.rate))):
            if self._fp_classes:
                r = rng.random()
                proto = next(p for cum, p in self._fp_classes if r < cum)
                x = _clamp_int(rng.random() * ARENA_W_MT * s, 0, ARENA_W_MT * s)
                y = _clamp_int(rng.random() * ARENA_H_MT * s, 0, ARENA_H_MT * s)
                row = self._phantom_row()
                hp = proto.max_hp or PHANTOM_HP
                for name, v in (("team", 0 if y < RIVER_Y_MT * s else 1), ("kind", proto.kind),
                                ("card_id", proto.card_id), ("unit_type", proto.unit_type),
                                ("flying", proto.flying), ("hp", hp), ("max_hp", hp),
                                ("x", x), ("y", y)):
                    row[F[name]] = v
            elif units:
                row = list(units[int(rng.integers(len(units)))])
                if Q.near_sigma_mt > 0:
                    row[F["x"]] = _clamp_int(row[F["x"]] + rng.normal(0.0, Q.near_sigma_mt) * s,
                                             0, ARENA_W_MT * s)
                    row[F["y"]] = _clamp_int(row[F["y"]] + rng.normal(0.0, Q.near_sigma_mt) * s,
                                             0, ARENA_H_MT * s)
            else:
                continue                            # nothing on the board to see twice
            row[F["uid"]] = -(k + 1)
            out.append(row)
            _count(stats, "phantoms")
        return out

    def _towers(self, player: dict, rng: Any) -> None:
        T = self.profile.towers
        hp = player.get("tower_hp")
        if not isinstance(hp, list):
            return
        mx = player.get("tower_max_hp") or []
        for k, v in enumerate(hp):
            m = mx[k] if k < len(mx) and mx[k] > 0 else v
            if v <= 0 or v >= m:
                continue                            # fallen, or full: both plain to see
            w = v + T.bias_frac * m
            if T.sigma_frac > 0:
                w += rng.normal(0.0, T.sigma_frac * m)
            if T.quant_hp > 1:
                w = T.quant_hp * _rint(w / T.quant_hp)
            hp[k] = max(1, min(int(m), _rint(w)))

    def _elixir(self, player: dict, rng: Any, stats: MutableMapping | None) -> None:
        E = self.profile.elixir
        e = int(player.get("elixir_milli", 0))
        whole, frac = divmod(max(0, min(MAX_ELIXIR_MILLI, e)), 1000)
        if E.digit_error_p > 0 and rng.random() < E.digit_error_p:
            whole = max(0, min(10, whole + (1 if rng.random() < 0.5 else -1)))
            _count(stats, "elixir_digit_error")
        if whole >= 10:
            player["elixir_milli"] = MAX_ELIXIR_MILLI
            return
        if E.fraction_lost_p > 0 and rng.random() < E.fraction_lost_p:
            frac = E.fraction_fill_milli
            _count(stats, "elixir_fraction_lost")
        elif E.fraction_sigma_milli > 0:
            frac = _clamp_int(frac + rng.normal(0.0, E.fraction_sigma_milli), 0, 999)
        player["elixir_milli"] = whole * 1000 + frac

    # ------------------------------------------------------------ opponent plays
    def play_seen(self, team: int, card: int, exec_tick: int, rng: Any, *,
                  stats: MutableMapping | None = None) -> tuple[int, int] | None:
        """How perception reports `team`'s play of row token `card` (catalogue id + 2) that ran on
        `exec_tick`: None (missed), or (card token, the tick to record it on)."""
        _team(team)
        card, exec_tick = int(card), int(exec_tick)
        O = self.profile.opponent_plays
        _count(stats, "plays")
        if not O.active():
            return card, exec_tick
        rng = _as_rng(rng)
        p = O.miss_p
        if O.spell_miss_p is not None and (card - TOKEN_OFFSET) in self.spell_cards:
            p = O.spell_miss_p
        if p > 0 and rng.random() < p:
            _count(stats, "plays_missed")
            return None
        late = max(0, O.late_ticks + _jitter(rng, O.late_jitter_ticks))
        if O.unknown_p > 0 and rng.random() < O.unknown_p:
            card = UNKNOWN_CARD
            _count(stats, "plays_unknown")
        return card, exec_tick + late


def apply(model: Any, state: Mapping, team: int, rng: Any, history: NoiseHistory | None = None, *,
          stats: MutableMapping | None = None) -> dict:
    """NoiseModel.apply; `model` may be a NoiseProfile (or its JSON / path), bound to nothing."""
    if not isinstance(model, NoiseModel):
        model = NoiseModel(model)
    return model.apply(state, team, rng, history, stats=stats)


class MatchNoise:
    """One match's noise: the model, a numpy rng seeded for the match, a NoiseHistory and counts."""

    def __init__(self, model: Any, seed: int = 0, *, history_ticks: int = DEFAULT_HISTORY_TICKS):
        self.model = model if isinstance(model, NoiseModel) else NoiseModel(model)
        self.rng = np.random.default_rng(int(seed))
        self.history = NoiseHistory(history_ticks)
        self.stats: dict[str, int] = {}
        #: perceived minus true tick of the last observe (board lag and clock), for obs_row.
        self.tick_shift = 0

    def push(self, state: Mapping) -> bool:
        """Remember a true state the match went through (for board lag on ticks with no row)."""
        return self.model.push(self.history, state)

    def observe(self, state: Mapping, team: int) -> dict:
        out = self.model.apply(state, team, self.rng, self.history, stats=self.stats)
        self.tick_shift = int(out["tick"]) - _tick_of(state)
        return out

    def record(self, mem: Any, team: int, card: int, exec_tick: int) -> bool:
        """Record `team`'s play into PublicMemory `mem` as the other seat perceives it. False
        when perception missed it."""
        seen = self.model.play_seen(team, card, exec_tick, self.rng, stats=self.stats)
        if seen is None:
            return False
        mem.record(team, seen[0], seen[1])
        return True

    def summary(self) -> str:
        return ", ".join(f"{k} {v}" for k, v in sorted(self.stats.items())) or "nothing observed"


def match_seed(tag: Any, base: int = 0) -> int:
    """A match's noise seed from its replay tag and a run seed (stable across processes)."""
    return int(hashlib.sha1(f"{int(base)}:{tag}".encode("utf-8")).hexdigest()[:12], 16)


def vocab_entry(profile: Any, seed: int = 0) -> dict:
    """vocab.json "perception_noise": what the rows were made under, so a checkpoint says it."""
    p = as_profile(profile)
    return {"schema": SCHEMA, "name": p.name, "digest": p.digest(), "seed": int(seed),
            "kinds": p.active_groups(), "profile": p.to_dict()}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def add_noise_args(ap: argparse.ArgumentParser) -> None:
    """--noise PROFILE, --noise-only KINDS, --noise-seed N (tools/rs_rows.py, tools/rs_eval.py)."""
    ap.add_argument("--noise", default=None, metavar="PROFILE",
                    help="perception noise (E17): a profile JSON, 'default' "
                         "(data/noise/default_prior.json) or 'none' (default: none, a perfect "
                         "observer, as before)")
    ap.add_argument("--noise-only", default=None, metavar="KINDS",
                    help=f"keep only these noise kinds of the profile ({', '.join(GROUPS)})")
    ap.add_argument("--noise-seed", type=int, default=0, help="run seed of the noise draws")


def profile_from_args(args: argparse.Namespace) -> NoiseProfile | None:
    """The profile of add_noise_args' options, or None for no noise (the hooks are skipped)."""
    src = getattr(args, "noise", None)
    if src is None or (isinstance(src, str) and src.strip().lower() in NO_NOISE):
        if getattr(args, "noise_only", None):
            raise ValueError("--noise-only needs --noise PROFILE")
        return None
    profile = NoiseProfile.load(src)
    if getattr(args, "noise_only", None):
        profile = profile.only(args.noise_only)
    return profile


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Describe and check perception-noise profiles (E17).")
    ap.add_argument("profiles", nargs="*", default=["default"],
                    help="profile JSON files, 'default' or 'none' (default: the shipped prior)")
    ap.add_argument("--only", default=None, metavar="KINDS", help="keep only these kinds")
    ap.add_argument("--json", action="store_true", help="print the complete profile as JSON")
    args = ap.parse_args(argv)
    for src in args.profiles:
        try:
            p = NoiseProfile.load(src)
            if args.only:
                p = p.only(args.only)
        except (ValueError, FileNotFoundError) as exc:
            print(f"{src}: {exc}", file=sys.stderr)
            return 2
        print(p.dumps() if args.json else p.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
