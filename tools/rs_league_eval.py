r"""League evaluation and the promotion gate (L03): candidates against a champion and anchors on
the same deck pairings and seeds, both seats, under the command delay and perception noise.

    .venvs\royalegym\Scripts\python.exe tools\rs_league_eval.py run ^
        --candidate runs/tokens/dt_hog/best.pt --champion runs/tokens/live_big_hog/best.pt ^
        --anchor runs/tokens/live_big/ep1.pt --anchor floor=idle --workers 4
    .venvs\royalegym\Scripts\python.exe tools\rs_league_eval.py report --log runs\league\dev.jsonl

    import rs_league_eval as LE                  # tools/ on sys.path, as the tools import
    pairs = rs_match.load_pairs(split="dev", require=rs_match.HOG26).sample(250, seed=0)
    rep = LE.evaluate(pairs, LE.Protocol(split="val", delay=24, noise="default"),
                      candidates=["dt=runs/tokens/dt_hog/best.pt"],
                      champion="runs/tokens/live_big_hog/best.pt",
                      anchors=["runs/tokens/live_big/ep1.pt"], log_path="runs/league/dev.jsonl")
    rep["gates"]["dt"]["decision"]               # "pass", "cycle", "fail" or "incomplete"

Why (tokens-09, tokens-15): checkpoints were picked by validation card accuracy and by about 300
unpaired matches against one reference (train_tokens.py best.pt, rs_eval.py), whose Wilson
interval is +-5.6 pp: too weak an instrument to say that a model beats the one before it. Here
every player meets its opponents on the same frozen pairings and seeds, from both seats, so two
players' results against the same opponent are paired (tools/league/stats.py), and promotion is
the three-condition gate of royalelearn ladder/gate.py:1-21 (MIT; re-implemented, not imported).

THE PROTOCOL (Protocol): the split of the deck pairings ("val" = dev, the default; "test" =
final, only with --final / final=True), their digest, the run seed, the command delay D (default
24 ticks: 21-22 of client delay plus capture, inference and adb), an optional per-pairing jitter
(D +- j, drawn from the pairing's tag and the seed), the perception noise of every policy seat
(default: the prior, tools/perception_noise.py; resolved() turns "default" into the profile
itself so the digest covers its content), the opponent-memory lag, the tick cap, which sides play
base forms, how a head-to-head deals the decks (h2h_decks) and the engine config. digest() names
it: games are pooled only within one protocol.

THE GAMES. A comparison of player X against Y on the first n pairings of the frozen pair set is
rs_match.pairing_specs: X on side a's deck (the --require deck, Hog 2.6) at team 0 and the
seat-swapped game (rs_match.MatchSpec.swapped: X keeps its deck, seed and noise on the other
seat). h2h_decks "both" (the default for candidate vs champion and vs pool members) adds the same
two games with the decks exchanged (Y on side a), so a head-to-head between two Hog players is
not decided by the opponent deck's strength: 4 games per pairing. Against an anchor the evaluated
player always keeps side a (2 games per pairing): that is the deck it plays live. Comparisons use
prefixes of one pair list, so the champion's games against an anchor are on the very positions
(pairing, team, deck) of every candidate's games against it. Every policy seat gets the
protocol's noise unless its spec sets one (noise="none" is a perfect observer); idle seats none.

V2 PLAYERS. A checkpoint trained on rs_rows/v2 rows (its vocab "schema") plays like any other:
rs_match gives its seat v2 rows (module doc of tools/rs_match.py, V2 SEATS: the ability buttons
of the deck it is dealt, its own plays in flight, the opponent's elixir counted from the plays
the engine ran). Legacy and v2 players meet in one league. A player never presses its ability
buttons unless its spec says presses=on (or the run passes --presses for every player that sets
none), and then only when its checkpoint's buttons trained (rs_decide.presses_allowed): with
h2h decks "both" it also plays the opponent deck, heroes and champions included. The switch is
part of the player's identity (game keys, identity digests) only while it is on, so the logs of
runs before it still resume; the command line turns it off, with a note, for a checkpoint that
cannot press (presses_effective), so --presses changes the identity only of the players whose
play it changes. An idle player never presses.

THE GATE (gate()), evaluated in order on the protocol's games:
  1 beats_champion  the candidate's score against the champion over >= champion_games (1,000)
                    games: Wilson 95% lower bound > champion_lower_bound (0.52). At n = 1,000 a
                    score of 551 passes (LB 0.52003) and 550.5 does not (0.51953).
  2 anchor:NAME     for each anchor (the champion excluded): on the positions both the candidate
                    and the champion played against it, >= anchor_games (400) of them, the
                    candidate's score may be at most anchor_tolerance_pp (2 pp) below the
                    champion's (a drop of exactly 2 pp passes).
  3 pool_collapse   when pool members are given: the candidate's mean score against them, over
                    >= pool_games (100) each, must reach the champion's mean against them less
                    one standard error over pairings. The champion's mean against a member is
                    its own score on the positions both played against it (evaluate() plays the
                    champion against every selected member, as against an anchor) when it has
                    >= pool_games of them, else its BTD-fitted mean (every game of the
                    protocol). A fitted mean alone cannot see a cycle: fitted on the candidate's
                    own games, it follows them by transitivity.
                    No members: not played; promotion rests on 1 and 2.
Decision: "fail" when 1 or 2 fails (later conditions with no games are marked skipped),
"cycle" when 1 and 2 pass and 3 fails (admit as an opponent, do not promote), "pass" (promote)
when all hold, "incomplete" when a condition that would decide has fewer games than it needs.
evaluate() stops early as royalelearn does: a candidate that failed 1 plays no anchor games, one
that failed 2 no pool games (--full plays everything); the champion plays the anchors and the
pool members once some candidate reaches that phase (its games stay in the log for later runs).

THE LOG (JSONL, appended and fsynced game by game, so a restart loses at most the games in
flight): a "league_run" header per run (protocol, players with their spec and checkpoint
digest, roles, gate settings), a "league_game" line per game (game key = sha1 of the MatchSpec
and the checkpoint digests; protocol digest; split; pairing; seat; teams; holder of side a's
deck; winner; the two teams' identity digests; the rs_match.MatchResult without its play log)
and "league_error" lines for games that raised. A rerun plays only the games its protocol lacks
(errors are retried); a name whose games in the log were played by another player (spec or
checkpoint changed) under the same protocol is refused. A torn last line (a crash mid-write) is
skipped when read and never glued to the next record.

THE REPORT (report()): per matchup the score with its Wilson interval and a bootstrap interval
over pairings (the games of one pairing are correlated); per candidate and opponent shared with
the champion the paired bootstrap CI of candidate - champion and an exact McNemar
(tools/league/stats.py); the BTD rating table (anchor: the first anchor, else the champion); the
gate decision of every candidate and the best one that passed. The final split is refused unless
final=True (--final): it is held out for the one reported final evaluation.

WORKERS. Each worker process holds torch, royalesim and two policies (about WORKER_MB = 1.2 GB);
worker_cap() keeps RESERVE_MB (3 GB) for the parent and the system, at most MAX_WORKERS (6) and
one CPU free: on the owner's 16 GB PC with MuMu and the browser closed that is 4-6 workers.

Never imports scripts/gamestate; nothing here touches the emulator.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import perception_noise as PN  # noqa: E402  (numpy and the standard library)
import rs_engine  # noqa: E402  (standard library only; royalesim is imported by the matches)
import rs_memory  # noqa: E402  (standard library only)
from tools.league import stats as LS  # noqa: E402

SCHEMA = 1
TOOL = "tools/rs_league_eval.py"
KIND_RUN, KIND_GAME, KIND_ERROR = "league_run", "league_game", "league_error"

#: rs_rows.split_of names; the protocol's aliases (rs_match.SPLIT_ALIASES) and display names.
SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"dev": "val", "final": "test"}
SPLIT_NAMES = {"train": "train", "val": "dev", "test": "final"}
DEV_SPLIT, FINAL_SPLIT = "val", "test"
FINAL_MESSAGE = ("the final split (IL_Replay test) is held out: it is played and reported only "
                 "with --final (final=True), for the one reported final evaluation")

DEFAULT_DELAY = 24               # ticks: 21-22 client command delay + capture, inference, adb
DEFAULT_NOISE = "default"        # tools/perception_noise.py's prior
MAX_TICKS = 6200                 # rs_match.MAX_TICKS
MEM_LEGACY = "legacy"            # rs_rows.MEM_LEGACY
H2H_BOTH, H2H_A = "both", "a"

PASS, CYCLE, FAIL, INCOMPLETE = "pass", "cycle", "fail", "incomplete"
DECISIONS = (PASS, CYCLE, FAIL, INCOMPLETE)
COND_CHAMPION, COND_POOL, ANCHOR_PREFIX = "beats_champion", "pool_collapse", "anchor:"
SKIP_SETTLED = "not counted: an earlier condition already settled the decision"
SKIP_NO_POOL = "not played: no pool members besides the candidate, the champion and the anchors"
#: Floating-point slack on the anchor and pool comparisons (a drop of exactly 2 pp passes).
EPS = 1e-12

CHAMPION_GAMES = 1000
CHAMPION_LOWER_BOUND = 0.52
ANCHOR_GAMES = 400
ANCHOR_TOLERANCE_PP = 2.0
POOL_MEMBERS = 8
POOL_GAMES = 100

#: Memory per worker process (torch, royalesim, two TokenPolicies, rows), the reserve for the
#: parent and the system, the hard cap for a 16 GB PC, and what is assumed free when the free
#: memory cannot be read.
WORKER_MB = 1200
RESERVE_MB = 3072
MAX_WORKERS = 6
ASSUMED_AVAILABLE_MB = 8192
MAX_ERRORS = 25


class FinalSplitError(ValueError):
    """The held-out final split was asked for without --final."""


def _rm():
    """tools/rs_match.py, imported when games are planned or played (it pulls in rs_rows,
    rs_eval and the corpus readers; the statistics here do not need them)."""
    import rs_match
    return rs_match


def _int(value: Any, what: str, lo: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{what}: an int, got {value!r}")
    value = int(value)
    if lo is not None and value < lo:
        raise ValueError(f"{what}: at least {lo}, got {value}")
    return value


def _mem_lag(value: Any) -> Any:
    """rs_rows.mem_lag_value's rule: an int >= 0 (digits too) or "legacy"."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text == MEM_LEGACY:
            return MEM_LEGACY
        if not text.isdigit():
            raise ValueError(f"mem_lag: ticks >= 0 or 'legacy', got {value!r}")
        value = int(text)
    return _int(value, "mem_lag", 0)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _canon(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def resolve_noise(noise: Any) -> Any:
    """None for a perfect observer (None, "none", an all-zero profile), else the profile as its
    complete JSON dict (a "default" / path / dict / NoiseProfile read once, so a digest of it
    covers the content and not only the name)."""
    if noise is None or (isinstance(noise, str) and noise.strip().lower() in PN.NO_NOISE):
        return None
    profile = PN.as_profile(noise)
    return None if profile.is_identity() else profile.to_dict()


# ---------------------------------------------------------------------------
# the protocol and the gate settings
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Protocol:
    """How every game of an evaluation is played (module doc, THE PROTOCOL)."""

    split: str = DEV_SPLIT
    pairs_digest: str = ""
    seed: int = 0
    delay: int = DEFAULT_DELAY
    delay_jitter: int = 0
    noise: Any = DEFAULT_NOISE
    mem_lag: Any = rs_memory.BOARD_LAG
    max_ticks: int = MAX_TICKS
    base_forms: tuple = (True, False)
    h2h_decks: str = H2H_BOTH
    engine: Any = None

    def __post_init__(self) -> None:
        put = object.__setattr__
        split = SPLIT_ALIASES.get(str(self.split), str(self.split))
        if split not in SPLITS:
            raise ValueError(f"split {self.split!r}: train, dev (val) or final (test)")
        put(self, "split", split)
        put(self, "pairs_digest", str(self.pairs_digest or ""))
        for what, lo in (("seed", 0), ("delay", 0), ("delay_jitter", 0), ("max_ticks", 1)):
            put(self, what, _int(getattr(self, what), what, lo))
        if self.delay_jitter > self.delay:
            raise ValueError(f"delay_jitter {self.delay_jitter} exceeds delay {self.delay}")
        put(self, "mem_lag", _mem_lag(self.mem_lag))
        forms = tuple(self.base_forms)
        if len(forms) != 2 or not all(isinstance(f, bool) for f in forms):
            raise ValueError(f"base_forms: two bools (side a, side b), got {self.base_forms!r}")
        put(self, "base_forms", forms)
        if self.h2h_decks not in (H2H_BOTH, H2H_A):
            raise ValueError(f"h2h_decks: {H2H_BOTH!r} or {H2H_A!r}, got {self.h2h_decks!r}")
        if isinstance(self.noise, str) and self.noise.strip().lower() in PN.NO_NOISE:
            put(self, "noise", None)
        if self.engine is not None:
            engine = json.loads(self.engine) if isinstance(self.engine, str) else self.engine
            put(self, "engine", rs_engine.EngineConfig.coerce(engine).to_json())

    def replace(self, **changes: Any) -> "Protocol":
        return dataclasses.replace(self, **changes)

    def resolved(self) -> "Protocol":
        """The engine filled in (rs_engine.active() when None) and the noise read into its
        profile dict: what a run records and digests."""
        out = self
        if out.engine is None:
            out = out.replace(engine=rs_engine.active().to_json())
        noise = resolve_noise(out.noise)
        if _canon(noise) != _canon(out.noise):
            out = out.replace(noise=noise)
        return out

    def delay_for(self, tag: str) -> int:
        """The command delay of a pairing: `delay`, or delay +- delay_jitter drawn from
        sha1(seed:tag:delay) (the same in every run and for every player)."""
        if not self.delay_jitter:
            return self.delay
        h = int(hashlib.sha1(f"{self.seed}:{tag}:delay".encode("utf-8")).hexdigest()[:8], 16)
        return self.delay - self.delay_jitter + h % (2 * self.delay_jitter + 1)

    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        out["base_forms"] = list(self.base_forms)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Protocol":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"Protocol: unknown field(s) {unknown}")
        return cls(**dict(data))

    def digest(self) -> str:
        """16 hex digits over every field: games are pooled only under one digest."""
        return hashlib.sha1(_canon(self.to_dict()).encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        if self.noise is None:
            noise = "none"
        elif isinstance(self.noise, Mapping):
            noise = str(self.noise.get("name") or "profile")
        else:
            noise = str(self.noise)
        jitter = f" +-{self.delay_jitter}" if self.delay_jitter else ""
        return (f"{SPLIT_NAMES.get(self.split, self.split)} split ({self.split}), delay "
                f"{self.delay}{jitter}, noise {noise}, mem-lag {self.mem_lag}, h2h decks "
                f"{self.h2h_decks}, seed {self.seed}, max ticks {self.max_ticks}")


@dataclasses.dataclass(frozen=True)
class GateConfig:
    """The gate's numbers (module doc, THE GATE; royalelearn config.py:455-466, anchors at 400
    games instead of 200 as the plan asks) and the bootstrap's."""

    champion_games: int = CHAMPION_GAMES
    champion_lower_bound: float = CHAMPION_LOWER_BOUND
    anchor_games: int = ANCHOR_GAMES
    anchor_tolerance_pp: float = ANCHOR_TOLERANCE_PP
    pool_members: int = POOL_MEMBERS
    pool_games: int = POOL_GAMES
    z: float = 1.96
    n_boot: int = 2000
    boot_seed: int = 0

    def __post_init__(self) -> None:
        for what in ("champion_games", "anchor_games", "pool_games", "n_boot"):
            object.__setattr__(self, what, _int(getattr(self, what), what, 1))
        object.__setattr__(self, "pool_members", _int(self.pool_members, "pool_members", 0))
        object.__setattr__(self, "boot_seed", _int(self.boot_seed, "boot_seed", 0))
        if not 0.0 < float(self.champion_lower_bound) < 1.0:
            raise ValueError(f"champion_lower_bound: in (0, 1), got {self.champion_lower_bound}")
        if not float(self.anchor_tolerance_pp) >= 0.0:
            raise ValueError(f"anchor_tolerance_pp: >= 0, got {self.anchor_tolerance_pp}")
        if not float(self.z) > 0.0:
            raise ValueError(f"z: > 0, got {self.z}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def pairings_needed(games: int, decks: str) -> int:
    """Pairings a comparison of `games` games takes: 4 games each with decks "both", else 2."""
    per = 4 if decks == H2H_BOTH else 2
    return -(-int(games) // per)


def pairings_for(config: GateConfig, protocol: Protocol, *, pool: bool = False) -> int:
    """The pairings a whole gate takes (comparisons use prefixes of one list)."""
    need = max(pairings_needed(config.champion_games, protocol.h2h_decks),
               pairings_needed(config.anchor_games, H2H_A))
    if pool:
        need = max(need, pairings_needed(config.pool_games, protocol.h2h_decks))
    return need


# ---------------------------------------------------------------------------
# players
# ---------------------------------------------------------------------------

_INT_OPTS = frozenset({"lead", "noise_seed", "seed_offset"})
_FLOAT_OPTS = frozenset({"wait_keep"})
_BOOL_OPTS = frozenset({"spell_aim", "hold_pending", "allow_drift", "presses"})
_STR_OPTS = frozenset({"decide", "visibility", "schema", "noise"})
_TRUE, _FALSE = frozenset({"1", "true", "yes", "on"}), frozenset({"0", "false", "no", "off"})


def _option(key: str, text: str) -> Any:
    if key in _INT_OPTS:
        try:
            return int(text, 10)
        except ValueError:
            raise ValueError(f"player option {key}: an int, got {text!r}") from None
    if key in _FLOAT_OPTS:
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"player option {key}: a number, got {text!r}") from None
    if key in _BOOL_OPTS:
        low = text.lower()
        if low not in _TRUE | _FALSE:
            raise ValueError(f"player option {key}: true or false, got {text!r}")
        return low in _TRUE
    if key in _STR_OPTS:
        return text
    known = sorted(_INT_OPTS | _FLOAT_OPTS | _BOOL_OPTS | _STR_OPTS)
    raise ValueError(f"unknown player option {key!r} (known: {', '.join(known)})")


def parse_player(value: Any, **defaults: Any) -> Any:
    """An rs_match.AgentSpec from an AgentSpec, its dict, a JSON object, or the command-line
    form [NAME=]CKPT[,KEY=VALUE...] (keys: lead, decide, noise, noise_seed, visibility, schema,
    wait_keep, spell_aim, hold_pending, seed_offset, allow_drift, presses). CKPT "idle" (or
    empty) never plays. `defaults` fill keys the value leaves out (None: not filled)."""
    RM = _rm()
    if isinstance(value, RM.AgentSpec):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
    else:
        raw = str(value).strip()
        if raw.startswith("{"):
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError(f"player {raw!r}: a JSON object")
        else:
            head, *opts = raw.split(",")
            data = {}
            if "=" in head:
                name, head = head.split("=", 1)
                data["name"] = name.strip()
            data["ckpt"] = head.strip()
            for opt in opts:
                if not opt.strip():
                    continue
                if "=" not in opt:
                    raise ValueError(f"player option {opt!r}: KEY=VALUE")
                key, text = (s.strip() for s in opt.split("=", 1))
                key = key.replace("-", "_")
                if key in data:
                    raise ValueError(f"player option {key!r} given twice in {raw!r}")
                data[key] = _option(key, text)
    for key, val in defaults.items():
        if val is not None:
            data.setdefault(key, val)
    return RM.AgentSpec.from_dict(data)


def apply_protocol(agent: Any, protocol: Protocol) -> Any:
    """The agent with the protocol's noise when it sets none; idle seats and agents with their
    own noise (noise="none" included) are kept as they are."""
    if agent.idle or agent.noise is not None or protocol.noise is None:
        return agent
    return dataclasses.replace(agent, noise=protocol.noise)


@dataclasses.dataclass(frozen=True)
class Roles:
    """Who is evaluated against whom: names, and every player's AgentSpec by name."""

    candidates: tuple
    champion: str
    anchors: tuple = ()
    pool: tuple = ()
    players: Mapping = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"candidates": list(self.candidates), "champion": self.champion,
                "anchors": list(self.anchors), "pool": list(self.pool)}


def prepare_players(candidates: Iterable[Any], champion: Any, anchors: Iterable[Any] = (),
                    pool: Iterable[Any] = (), protocol: Protocol | None = None, *,
                    decide: str | None = None, presses: bool | None = None) -> Roles:
    """Parse every player (parse_player, `decide` as the default rule, `presses` as the default
    switch: module doc, V2 PLAYERS), give it the protocol's noise (apply_protocol) and check
    that no name is used twice."""
    specs: dict[str, Any] = {}

    def add(role: str, value: Any) -> str:
        agent = parse_player(value, decide=decide, presses=presses)
        if agent.idle and agent.presses:
            agent = dataclasses.replace(agent, presses=False)     # an idle seat never presses
        if protocol is not None:
            agent = apply_protocol(agent, protocol)
        name = agent.label
        if name in specs:
            raise ValueError(f"player name {name!r} is used twice (the {role} and an earlier "
                             "player): name one of them, NAME=CKPT")
        specs[name] = agent
        return name

    if champion is None:
        raise ValueError("a champion is required")
    cands = tuple(add("candidate", c) for c in candidates)
    if not cands:
        raise ValueError("at least one candidate is required")
    champ = add("champion", champion)
    anch = tuple(add("anchor", a) for a in anchors)
    members = tuple(add("pool member", m) for m in pool)
    return Roles(cands, champ, anch, members, specs)


def presses_effective(agent: Any, meta: Mapping[str, Any] | None) -> Any:
    """The agent with its presses switch off when its checkpoint (`meta`) cannot press anyway
    (rs_decide.presses_allowed: a v2 checkpoint whose buttons trained), so that --presses
    changes the identity only of the players it changes the play of (module doc, V2 PLAYERS)."""
    if not agent.presses:
        return agent
    import rs_decide
    if rs_decide.presses_allowed(meta):
        return agent
    return dataclasses.replace(agent, presses=False)


def ckpt_digest(ckpt: str) -> str | None:
    """16 hex digits of SHA-256 over a checkpoint file; None for an idle seat or no file."""
    if not ckpt or ckpt.strip().lower() == "idle":
        return None
    path = Path(ckpt)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def identity(agent: Any, digest: str | None) -> dict:
    """What a player is in the log: its AgentSpec and its checkpoint's digest."""
    return {**agent.to_dict(), "ckpt_digest": digest}


def identity_digest(ident: Mapping[str, Any]) -> str:
    """16 hex digits over an identity(): every game record carries one per team."""
    return hashlib.sha1(_canon(ident).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# games: planning
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Task:
    """One planned game: its key, its rs_match.MatchSpec, the comparison it belongs to (a is the
    evaluated player, b its opponent) and the player on side a's deck."""

    key: str
    spec: Any
    a: str
    b: str
    holder: str


def game_key(spec: Any, digests: Mapping[str, str | None] | None = None) -> str:
    """20 hex digits of sha1 over the MatchSpec and the two seats' checkpoint digests."""
    digests = digests or {}
    body = {"spec": spec.to_dict(), "ckpt": [digests.get(a.label) for a in spec.agents]}
    return hashlib.sha1(_canon(body).encode("utf-8")).hexdigest()[:20]


def _pair_list(pairs: Any) -> list:
    return list(pairs.pairs if hasattr(pairs, "pairs") else pairs)


def comparison_tasks(pairs: Any, protocol: Protocol, a: Any, b: Any, *, games: int,
                     decks: str = H2H_A, digests: Mapping[str, str | None] | None = None
                     ) -> list[Task]:
    """The games of player spec `a` against `b` (module doc, THE GAMES): the first
    pairings_needed(games, decks) pairings, each through rs_match.pairing_specs with both seats
    (and, with decks "both", again with the decks exchanged). ValueError when the pair list is
    shorter."""
    RM = _rm()
    if decks not in (H2H_BOTH, H2H_A):
        raise ValueError(f"decks: {H2H_BOTH!r} or {H2H_A!r}, got {decks!r}")
    plist = _pair_list(pairs)
    need = pairings_needed(games, decks)
    if len(plist) < need:
        raise ValueError(f"{a.label} vs {b.label}: {games} games take {need} pairings; the pair "
                         f"set has {len(plist)}")
    orders = [(a, b)] if decks == H2H_A else [(a, b), (b, a)]
    out = []
    for pair in plist[:need]:
        kwargs = dict(seed=protocol.seed, seats=2, base_forms=protocol.base_forms,
                      delay=protocol.delay_for(pair.tag), engine=protocol.engine,
                      mem_lag=protocol.mem_lag, max_ticks=protocol.max_ticks)
        for x, y in orders:
            for spec in RM.pairing_specs([pair], (x, y), **kwargs):
                out.append(Task(game_key(spec, digests), spec, a.label, b.label, x.label))
    return out


# ---------------------------------------------------------------------------
# the log
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Game:
    """One played game as the analysis reads it. winner: 0 / 1, -1 a draw; ids: the two
    teams' identity digests (empty when unknown)."""

    key: str
    teams: tuple
    winner: int
    pairing: str = ""
    seat: int = 0
    holder: str = ""
    protocol: str = ""
    split: str = ""
    ids: tuple = ()

    def __post_init__(self) -> None:
        teams = tuple(str(t) for t in self.teams)
        if len(teams) != 2 or teams[0] == teams[1]:
            raise ValueError(f"teams: two different players, got {self.teams!r}")
        object.__setattr__(self, "teams", teams)
        winner = int(self.winner)
        if winner not in (-1, 0, 1):
            raise ValueError(f"winner: 0, 1 or -1 (a draw), got {self.winner!r}")
        object.__setattr__(self, "winner", winner)
        object.__setattr__(self, "seat", int(self.seat))
        object.__setattr__(self, "ids", tuple(None if i is None else str(i) for i in self.ids))

    def has(self, name: str) -> bool:
        return name in self.teams

    def meets(self, a: str, b: str) -> bool:
        return a != b and a in self.teams and b in self.teams

    def team_of(self, name: str) -> int:
        if name == self.teams[0]:
            return 0
        if name == self.teams[1]:
            return 1
        raise ValueError(f"{name!r} did not play game {self.key} ({list(self.teams)})")

    def score(self, name: str) -> float:
        team = self.team_of(name)
        if self.winner == -1:
            return 0.5
        return 1.0 if self.winner == team else 0.0

    def other(self, name: str) -> str:
        return self.teams[1 - self.team_of(name)]

    def position(self, name: str) -> tuple:
        """Where `name` sat: (pairing, team, held side a's deck). Two players' games against
        the same opponent on the same position are the same game but for the player."""
        return (self.pairing, self.team_of(name), self.holder == name)

    @classmethod
    def from_record(cls, rec: Mapping[str, Any]) -> "Game":
        return cls(key=str(rec["key"]), teams=tuple(rec["teams"]), winner=rec["winner"],
                   pairing=str(rec.get("pairing") or ""), seat=int(rec.get("seat") or 0),
                   holder=str(rec.get("holder") or ""), protocol=str(rec.get("protocol") or ""),
                   split=str(rec.get("split") or ""), ids=tuple(rec.get("ids") or ()))


@dataclasses.dataclass
class Log:
    """What read_log found: games (first record of each (protocol, key)), run headers, error
    records, unreadable lines and duplicate games."""

    games: list = dataclasses.field(default_factory=list)
    runs: list = dataclasses.field(default_factory=list)
    errors: list = dataclasses.field(default_factory=list)
    bad_lines: int = 0
    duplicates: int = 0

    def protocols(self) -> Counter:
        return Counter(g.protocol for g in self.games)

    def of(self, protocol: str) -> list[Game]:
        return [g for g in self.games if g.protocol == protocol]

    def done(self, protocol: str) -> set[str]:
        return {g.key for g in self.games if g.protocol == protocol}

    def last_run(self, protocol: str) -> dict | None:
        runs = [r for r in self.runs if r.get("protocol") == protocol]
        return runs[-1] if runs else None


def read_log(path: str | Path) -> Log:
    """The JSONL log at `path` (an empty Log when there is none)."""
    log = Log()
    path = Path(path)
    if not path.is_file():
        return log
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                log.bad_lines += 1
                continue
            if not isinstance(rec, dict):
                log.bad_lines += 1
                continue
            kind = rec.get("kind")
            if kind == KIND_GAME:
                try:
                    game = Game.from_record(rec)
                except (KeyError, TypeError, ValueError):
                    log.bad_lines += 1
                    continue
                k = (game.protocol, game.key)
                if k in seen:
                    log.duplicates += 1
                    continue
                seen.add(k)
                log.games.append(game)
            elif kind == KIND_RUN:
                log.runs.append(rec)
            elif kind == KIND_ERROR:
                log.errors.append(rec)
    return log


def append_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Append JSON lines and fsync. A file whose last line was torn (no newline) gets one
    first, so the new record never joins the broken one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = False
    if path.is_file() and path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        if torn:
            f.write("\n")
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def game_record(task: Task, result: Any, protocol: str, split: str,
                ids: Mapping[str, str] | None = None) -> dict:
    """The log line of a played game (module doc, THE LOG); ids: identity digests by name."""
    names = tuple(a.label for a in task.spec.agents)
    if tuple(result.agents) != names:
        raise ValueError(f"game {task.key}: the result is of {list(result.agents)}, the spec "
                         f"of {list(names)}")
    spec = task.spec
    ids = ids or {}
    return {"kind": KIND_GAME, "schema": SCHEMA, "key": task.key, "protocol": protocol,
            "split": split, "a": task.a, "b": task.b, "holder": task.holder,
            "pairing": spec.pairing, "tag": spec.tag, "seat": spec.seat, "seed": spec.seed,
            "delay": list(result.delay), "teams": list(names), "winner": int(result.winner),
            "crowns": list(result.crowns), "ids": [ids.get(n) for n in names],
            "result": result.to_dict(log=False)}


def error_record(task: Task, exc: BaseException, protocol: str, split: str) -> dict:
    return {"kind": KIND_ERROR, "schema": SCHEMA, "key": task.key, "protocol": protocol,
            "split": split, "a": task.a, "b": task.b, "pairing": task.spec.pairing,
            "seat": task.spec.seat, "error": f"{type(exc).__name__}: {exc}", "time": time.time()}


def check_identities(log: Log, protocol: str, ids: Mapping[str, str]) -> None:
    """ValueError when a game of this protocol in the log was played by another player (spec or
    checkpoint: identity_digest) under a name of this run: the games would be pooled."""
    for g in log.of(protocol):
        for team, name in enumerate(g.teams):
            old = g.ids[team] if len(g.ids) == 2 else None
            if name in ids and old is not None and old != ids[name]:
                raise ValueError(
                    f"player {name!r} played games of protocol {protocol} in this log as "
                    "another player (its spec or its checkpoint changed): give it another "
                    "name (NAME=CKPT) or use another log")


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

def available_mb() -> int | None:
    """Free physical memory in MB (Windows GlobalMemoryStatusEx, Linux /proc/meminfo, else
    sysconf), or None when it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys // (1 << 20))
            return None
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text(encoding="ascii", errors="replace").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
        pages, size = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return int(pages * size // (1 << 20))
    except (OSError, ValueError, AttributeError, IndexError):
        return None


def worker_cap(requested: int, *, available: int | None = None, cpus: int | None = None,
               per_worker_mb: int = WORKER_MB, reserve_mb: int = RESERVE_MB,
               max_workers: int = MAX_WORKERS) -> tuple[int, str]:
    """(workers, why): `requested` capped by free memory ((available - reserve) / per worker;
    ASSUMED_AVAILABLE_MB when unknown), by the CPUs less one and by max_workers; at least 1.
    `available` (MB) and `cpus` default to the machine's."""
    want = max(1, int(requested))
    avail = available_mb() if available is None else int(available)
    if avail is None:
        avail = ASSUMED_AVAILABLE_MB
    ncpu = (os.cpu_count() or 1) if cpus is None else int(cpus)
    caps = {"memory": max(1, (avail - reserve_mb) // per_worker_mb),
            "cpus": max(1, ncpu - 1), "max": max(1, int(max_workers))}
    got = min(want, *caps.values())
    if got == want:
        return got, f"{got} worker(s)"
    limits = ", ".join(f"{k} {v}" for k, v in caps.items() if v == got)
    return got, (f"{got} worker(s), not {want}: capped by {limits} ({avail} MB free, "
                 f"{per_worker_mb} MB each, {reserve_mb} MB kept)")


def local_play() -> Callable[[Any], Any]:
    """In-process play: rs_match.run_match with one noise-model cache for the run."""
    RM = _rm()
    cache: dict = {}
    return lambda spec: RM.run_match(spec, noise_models=cache, keep_log=False)


def pool_play(tasks: Sequence[Task], workers: int, engine: Any):
    """Yield (task, MatchResult or None, exception or None) as worker processes finish games
    (rs_match's worker task; each worker gets the engine config). On exit the queued games are
    cancelled and only the running ones are waited for."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    RM = _rm()
    pool = ProcessPoolExecutor(workers, initializer=rs_engine.set_active, initargs=(engine,))
    try:
        futures = {pool.submit(RM._run_dict, t.spec.to_dict()): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                data = fut.result()
            except BrokenProcessPool:
                raise
            except Exception as exc:                        # one game's failure, logged
                yield task, None, exc
                continue
            yield task, RM.MatchResult.from_dict(data), None
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def run_tasks(tasks: Iterable[Task], log_path: str | Path, *, protocol: str, split: str,
              done: set[str], workers: int = 1, play: Callable[[Any], Any] | None = None,
              engine: Any = None, echo: Callable[[str], Any] | None = None,
              max_errors: int = MAX_ERRORS, ids: Mapping[str, str] | None = None) -> dict:
    """Play the tasks whose key is not in `done` (each once) and append a record per game as
    it finishes; `done` gains the played keys. play(spec) -> MatchResult replaces the engine
    (in process); otherwise workers > 1 runs pool_play. RuntimeError after max_errors errors.
    ids: identity digests by name, for the records. Returns {"planned", "skipped", "played",
    "errors"}."""
    tasks = list(tasks)
    todo, seen = [], set()
    for t in tasks:
        if t.key in done or t.key in seen:
            continue
        seen.add(t.key)
        todo.append(t)
    out = {"planned": len(tasks), "skipped": len(tasks) - len(todo), "played": 0, "errors": 0}
    if not todo:
        return out
    t0 = time.perf_counter()
    every = max(1, len(todo) // 20)

    def file(task: Task, result: Any, exc: BaseException | None) -> None:
        if exc is None:
            append_records(log_path, [game_record(task, result, protocol, split, ids)])
            done.add(task.key)
            out["played"] += 1
        else:
            append_records(log_path, [error_record(task, exc, protocol, split)])
            out["errors"] += 1
            if echo is not None:
                echo(f"  error in {task.spec.id} ({task.a} vs {task.b}): "
                     f"{type(exc).__name__}: {exc}")
            if out["errors"] > max_errors:
                raise RuntimeError(f"{out['errors']} games raised (more than {max_errors}); "
                                   f"see the {KIND_ERROR} lines in {log_path}")
        n = out["played"] + out["errors"]
        if echo is not None and (n % every == 0 or n == len(todo)):
            spent = time.perf_counter() - t0
            echo(f"  {n}/{len(todo)} games, {spent:.0f} s, about "
                 f"{spent / n * (len(todo) - n):.0f} s left")

    if play is not None or workers <= 1:
        fn = play or local_play()
        RM = None
        for task in todo:
            try:
                result = fn(task.spec)
                if isinstance(result, Mapping):
                    RM = RM or _rm()
                    result = RM.MatchResult.from_dict(result)
            except Exception as exc:                        # one game's failure, logged
                file(task, None, exc)
                continue
            file(task, result, None)
    else:
        with contextlib.closing(pool_play(todo, workers, engine)) as results:
            for task, result, exc in results:
                file(task, result, exc)
    return out


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Condition:
    """One gate condition with the numbers that decided it. missing: games it still lacks;
    skipped: not counted (reason says why)."""

    name: str
    passed: bool
    n: int
    observed: float = 0.0
    bound: float = 0.0
    reference: float = 0.0
    missing: int = 0
    skipped: bool = False
    reason: str = ""
    detail: Mapping = dataclasses.field(default_factory=dict)

    @property
    def incomplete(self) -> bool:
        return self.missing > 0 and not self.skipped

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["detail"] = dict(self.detail)
        out["incomplete"] = self.incomplete
        return out


@dataclasses.dataclass(frozen=True)
class GateDecision:
    """A candidate's gate: decision PASS (promote), CYCLE (admit, not promote), FAIL or
    INCOMPLETE, and its conditions in order."""

    candidate: str
    champion: str
    decision: str
    conditions: Mapping

    @property
    def admit(self) -> bool:
        return self.decision in (PASS, CYCLE)

    @property
    def promote(self) -> bool:
        return self.decision == PASS

    @property
    def cycle(self) -> bool:
        return self.decision == CYCLE

    def turned_on(self) -> str:
        """The condition the decision turned on ("none" for a pass)."""
        for name, c in self.conditions.items():
            if c.skipped:
                continue
            if self.decision == INCOMPLETE and c.incomplete:
                return name
            if self.decision in (FAIL, CYCLE) and not c.passed and not c.incomplete:
                return name
        return "none"

    def missing(self) -> dict:
        return {n: c.missing for n, c in self.conditions.items() if c.incomplete}

    def to_dict(self) -> dict:
        return {"candidate": self.candidate, "champion": self.champion,
                "decision": self.decision, "admit": self.admit, "promote": self.promote,
                "cycle": self.cycle, "turned_on": self.turned_on(), "missing": self.missing(),
                "conditions": {n: c.to_dict() for n, c in self.conditions.items()}}


def _matched(games: Iterable[Game], x: str, y: str, opponent: str
             ) -> tuple[list[float], list[float], list[str]]:
    """x's and y's scores against `opponent` on the positions both played, in position order,
    and the pairing of each (the bootstrap's group)."""
    sx = {g.position(x): g.score(x) for g in games if g.meets(x, opponent)}
    sy = {g.position(y): g.score(y) for g in games if g.meets(y, opponent)}
    common = sorted(set(sx) & set(sy))
    return [sx[p] for p in common], [sy[p] for p in common], [p[0] for p in common]


def champion_condition(games: Iterable[Game], candidate: str, champion: str,
                       config: GateConfig) -> Condition:
    """1: Wilson LB of the candidate's score against the champion > champion_lower_bound over
    >= champion_games games."""
    scores = [g.score(candidate) for g in games if g.meets(candidate, champion)]
    si = LS.score_interval(scores, z=config.z)
    missing = max(0, config.champion_games - si.n)
    passed = missing == 0 and si.lo > config.champion_lower_bound
    return Condition(COND_CHAMPION, passed, si.n, observed=si.score, bound=si.lo,
                     reference=float(config.champion_lower_bound), missing=missing,
                     detail={"hi": si.hi, "wins": si.wins, "draws": si.draws,
                             "losses": si.losses})


def anchor_condition(games: Iterable[Game], candidate: str, champion: str, anchor: str,
                     config: GateConfig) -> Condition:
    """2: on the positions both played against `anchor` (>= anchor_games), the candidate's score
    is at most anchor_tolerance_pp below the champion's."""
    games = list(games)
    name = ANCHOR_PREFIX + anchor
    a, b, groups = _matched(games, candidate, champion, anchor)
    n = len(a)
    missing = max(0, config.anchor_games - n)
    tol = float(config.anchor_tolerance_pp) / 100.0
    detail = {"candidate_games": sum(1 for g in games if g.meets(candidate, anchor)),
              "champion_games": sum(1 for g in games if g.meets(champion, anchor))}
    if not n:
        return Condition(name, False, 0, missing=missing, detail=detail,
                         reason="no games of both on the same positions")
    cand, champ = _mean(a), _mean(b)
    drop = champ - cand
    detail.update(drop_pp=100.0 * drop, mcnemar=LS.mcnemar(a, b).to_dict(),
                  paired=LS.paired_bootstrap_ci(a, b, groups=groups, n_boot=config.n_boot,
                                                seed=config.boot_seed).to_dict())
    passed = missing == 0 and drop <= tol + EPS
    return Condition(name, passed, n, observed=cand, bound=champ - tol, reference=champ,
                     missing=missing, detail=detail)


def pool_condition(games: Iterable[Game], candidate: str, champion: str, members: Sequence[str],
                   config: GateConfig, table: Any) -> Condition:
    """3: the candidate's mean score over the members (>= pool_games each) reaches the mean of
    the champion's score against each member less one standard error of the candidate's
    pairing means. The champion's score against a member is its mean on the positions both
    played against it (_matched) when there are >= pool_games of them, else
    table.predict(champion, member) (module doc, THE GATE)."""
    members = [m for m in members if m not in (candidate, champion)]
    if not members:
        return Condition(COND_POOL, False, 0, skipped=True, reason=SKIP_NO_POOL)
    games = list(games)
    per: dict[str, float] = {}
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    n = missing = 0
    for m in members:
        scores = []
        for g in games:
            if g.meets(candidate, m):
                scores.append(g.score(candidate))
                groups[(m, g.pairing)].append(g.score(candidate))
        n += len(scores)
        missing += max(0, config.pool_games - len(scores))
        if scores:
            per[m] = _mean(scores)
    observed = _mean(list(per.values()))
    values = np.array([_mean(v) for _, v in sorted(groups.items())], dtype=np.float64)
    se = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    predicted = {m: float(table.predict(champion, m)) for m in members}
    champion_seen: dict[str, float] = {}
    for m in members:
        _a, b, _groups = _matched(games, candidate, champion, m)
        if len(b) >= config.pool_games:
            champion_seen[m] = _mean(b)
    ref = {m: champion_seen.get(m, predicted[m]) for m in members}
    reference = _mean(list(ref.values()))
    passed = missing == 0 and observed + EPS >= reference - se
    return Condition(COND_POOL, passed, n, observed=observed, bound=reference - se,
                     reference=reference, missing=missing,
                     detail={"per_member": per, "predicted": predicted,
                             "champion_observed": champion_seen, "reference": ref,
                             "source": {m: "observed" if m in champion_seen else "fitted"
                                        for m in members},
                             "se": se, "groups": int(values.size)})


def ratings(games: Iterable[Game], anchor: str, **kwargs: Any) -> Any:
    """tools/league/stats.fit_btd over the games (team 0, team 1, team 0's score)."""
    return LS.fit_btd(((g.teams[0], g.teams[1], g.score(g.teams[0])) for g in games),
                      anchor=anchor, **kwargs)


def gate(games: Iterable[Game], candidate: str, champion: str, *, anchors: Iterable[str] = (),
         pool: Iterable[str] = (), config: GateConfig | None = None, table: Any = None
         ) -> GateDecision:
    """The promotion gate on one protocol's games (module doc, THE GATE). anchors equal to the
    candidate or the champion are left out, so are pool members that are anchors. `table`
    (a RatingTable or anything with predict(a, b)) is the pool's reference; default: ratings()
    over the games, anchored at the champion."""
    config = config or GateConfig()
    if candidate == champion:
        raise ValueError(f"the candidate is the champion ({candidate!r})")
    games = list(games)
    anchors = [a for a in dict.fromkeys(anchors) if a not in (candidate, champion)]
    pool = [m for m in dict.fromkeys(pool) if m not in (candidate, champion) and m not in anchors]
    if pool and table is None:
        table = ratings(games, champion)
    first = champion_condition(games, candidate, champion, config)
    conds: dict[str, Condition] = {COND_CHAMPION: first}
    for a in anchors:
        conds[ANCHOR_PREFIX + a] = anchor_condition(games, candidate, champion, a, config)
    conds[COND_POOL] = pool_condition(games, candidate, champion, pool, config, table)
    held = [conds[ANCHOR_PREFIX + a] for a in anchors]
    last = conds[COND_POOL]
    settled_after = None
    if first.incomplete:
        decision = INCOMPLETE
    elif not first.passed:
        decision, settled_after = FAIL, 0
    elif any(c.incomplete for c in held):
        decision = INCOMPLETE
    elif not all(c.passed for c in held):
        decision, settled_after = FAIL, 1
    elif last.skipped:
        decision = PASS
    elif last.incomplete:
        decision = INCOMPLETE
    else:
        decision = PASS if last.passed else CYCLE
    if settled_after is not None:
        later = [ANCHOR_PREFIX + a for a in anchors] if settled_after == 0 else []
        later.append(COND_POOL)
        for name in later:
            c = conds[name]
            if c.n == 0 and not c.skipped:
                conds[name] = dataclasses.replace(c, skipped=True, reason=SKIP_SETTLED)
    return GateDecision(candidate, champion, decision, conds)


def select_pool(members: Iterable[str], table: Any, champion: str, k: int) -> tuple[str, ...]:
    """At most k members spread over the rating range (royalelearn gate.py stratified_sample):
    members ranked by table rating (then name) in k equal strata, from each the one whose
    predicted score against the champion is nearest even (largest p(1 - p)). All of them when
    there are at most k."""
    members = [m for m in dict.fromkeys(members) if m != champion]
    if k <= 0 or not members:
        return ()
    if len(members) <= k:
        return tuple(members)

    def rating(m: str) -> float:
        return float(table.rating.get(m, table.anchor_rating)) if table is not None else 0.0

    def weight(m: str) -> float:
        p = float(table.predict(champion, m)) if table is not None else 0.5
        return p * (1.0 - p)

    ranked = sorted(members, key=lambda m: (rating(m), m))
    chosen: list[str] = []
    for s in range(k):
        lo = s * len(ranked) // k
        hi = max(lo + 1, (s + 1) * len(ranked) // k)
        window = [m for m in ranked[lo:hi] if m not in chosen]
        if window:
            chosen.append(max(window, key=weight))
    return tuple(chosen)


def pool_selection(games: Iterable[Game], pool: Sequence[str], candidates: Sequence[str],
                   champion: str, config: GateConfig, rating_anchor: str) -> tuple[str, ...]:
    """The pool members a gate plays: select_pool on ratings of the games without the
    candidates (so the choice does not move while the candidates' games come in)."""
    if not pool:
        return ()
    prior = [g for g in games if not any(g.has(c) for c in candidates)]
    return select_pool(pool, ratings(prior, rating_anchor), champion, config.pool_members)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def _rank(order: Sequence[str]) -> Callable[[str], tuple]:
    index = {n: i for i, n in enumerate(order)}
    return lambda name: (index.get(name, len(index)), name)


def matchups(games: Sequence[Game], order: Sequence[str], config: GateConfig) -> list[dict]:
    """Per pair of players that met: the first (in `order`) one's score, its Wilson interval,
    a bootstrap interval over pairings, and the scores with each player on side a's deck."""
    rank = _rank(order)
    by: dict[tuple[str, str], list[Game]] = defaultdict(list)
    for g in games:
        x, y = sorted(g.teams, key=rank)
        by[(x, y)].append(g)
    rows = []
    for x, y in sorted(by, key=lambda k: (rank(k[0]), rank(k[1]))):
        gs = sorted(by[(x, y)], key=lambda g: (g.position(x), g.key))
        scores = [g.score(x) for g in gs]
        si = LS.score_interval(scores, z=config.z)
        boot = LS.paired_bootstrap_ci(scores, [0.5] * len(scores), groups=[g.pairing for g in gs],
                                      n_boot=config.n_boot, seed=config.boot_seed)
        row = {"a": x, "b": y, **si.to_dict(), "boot_lo": 0.5 + boot.lo,
               "boot_hi": 0.5 + boot.hi, "boot_se": boot.se, "pairings": boot.n_groups}
        for who, key in ((x, "a_holds"), (y, "b_holds")):
            sub = [g.score(x) for g in gs if g.holder == who]
            if sub:
                row[key] = LS.score_interval(sub, z=config.z).to_dict()
        rows.append(row)
    return rows


def paired_table(games: Sequence[Game], candidates: Sequence[str], champion: str,
                 order: Sequence[str], config: GateConfig) -> list[dict]:
    """Per candidate and per opponent it shares with the champion: the paired bootstrap CI of
    candidate - champion on the same positions, and an exact McNemar."""
    rank = _rank(order)
    rows = []
    for c in candidates:
        if c == champion:
            continue
        mine = {g.other(c) for g in games if g.has(c)}
        theirs = {g.other(champion) for g in games if g.has(champion)}
        for opp in sorted((mine & theirs) - {c, champion}, key=rank):
            a, b, groups = _matched(games, c, champion, opp)
            if not a:
                continue
            ci = LS.paired_bootstrap_ci(a, b, groups=groups, n_boot=config.n_boot,
                                        seed=config.boot_seed)
            rows.append({"candidate": c, "champion": champion, "opponent": opp,
                         "candidate_score": _mean(a), "champion_score": _mean(b),
                         **ci.to_dict(), "mcnemar": LS.mcnemar(a, b).to_dict()})
    return rows


def rating_rows(table: Any) -> dict:
    """The BTD table as JSON: the fit's settings and one row per player, best first."""
    rows = [{"player": p, "rating": float(table.rating[p]), "se": float(table.se.get(p, 0.0)),
             "games": int(table.n_games.get(p, 0))} for p in table.players]
    rows.sort(key=lambda r: (-r["rating"], r["player"]))
    return {"anchor": table.anchor, "anchor_rating": table.anchor_rating, "nu": table.nu,
            "converged": table.converged, "iterations": table.iterations, "rows": rows}


def report(games: Iterable[Game], *, candidates: Sequence[str], champion: str,
           anchors: Sequence[str] = (), pool: Sequence[str] = (), config: GateConfig | None = None,
           rating_anchor: str | None = None, final: bool = False) -> dict:
    """The report of one protocol's games (module doc, THE REPORT). FinalSplitError for games
    of the final split unless final=True; ValueError for games of several splits."""
    config = config or GateConfig()
    games = list(games)
    splits = sorted({g.split for g in games if g.split})
    if len(splits) > 1:
        raise ValueError(f"games of several splits {splits}: report one protocol at a time")
    split = splits[0] if splits else ""
    if split == FINAL_SPLIT and not final:
        raise FinalSplitError(FINAL_MESSAGE)
    candidates = list(dict.fromkeys(candidates))
    anchors = [a for a in dict.fromkeys(anchors) if a != champion and a not in candidates]
    pool = [m for m in dict.fromkeys(pool) if m not in (champion, *candidates, *anchors)]
    known = [*candidates, champion, *anchors, *pool]
    others = sorted({p for g in games for p in g.teams} - set(known))
    order = known + others
    anchor = rating_anchor or (anchors[0] if anchors else champion)
    table = ratings(games, anchor)
    out: dict[str, Any] = {
        "schema": SCHEMA, "tool": TOOL, "split": split,
        "split_name": SPLIT_NAMES.get(split, split), "n_games": len(games),
        "roles": {"candidates": candidates, "champion": champion, "anchors": anchors,
                  "pool": pool},
        "players": order, "matchups": matchups(games, order, config),
        "paired": paired_table(games, candidates, champion, order, config),
        "ratings": rating_rows(table), "gate_config": config.to_dict(), "gates": {},
        "best": None}
    best, best_lb = None, -1.0
    for c in candidates:
        if c == champion:
            continue
        decision = gate(games, c, champion, anchors=anchors, pool=pool, config=config,
                        table=table)
        out["gates"][c] = decision.to_dict()
        lb = decision.conditions[COND_CHAMPION].bound
        if decision.promote and lb > best_lb:
            best, best_lb = c, lb
    out["best"] = best
    return out


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _condition_line(name: str, c: Mapping[str, Any]) -> str:
    if c["skipped"]:
        return f"{name}: {c['reason']}"
    if c["incomplete"]:
        return f"{name}: incomplete, {c['n']} games, {c['missing']} more needed"
    verdict = "ok" if c["passed"] else "FAILS"
    if name == COND_CHAMPION:
        return (f"{name}: Wilson LB {c['bound']:.4f} {'>' if c['passed'] else '<='} "
                f"{c['reference']:.2f} over {c['n']} games (score {_pct(c['observed'])}) {verdict}")
    if name.startswith(ANCHOR_PREFIX):
        drop = 100.0 * (c["reference"] - c["observed"])
        return (f"{name}: drop {drop:+.1f} pp over {c['n']} games (candidate "
                f"{_pct(c['observed'])}, champion {_pct(c['reference'])}) {verdict}")
    sources = sorted(set((c.get("detail") or {}).get("source", {}).values())) or ["fitted"]
    return (f"{name}: mean {_pct(c['observed'])} vs the champion's "
            f"{'/'.join(sources)} {_pct(c['reference'])} - 1 se = {_pct(c['bound'])} over "
            f"{c['n']} games {verdict}")


def format_report(rep: Mapping[str, Any]) -> str:
    """The report as text for the terminal."""
    lines = [f"league evaluation: {rep.get('split_name') or '?'} split ({rep.get('split') or '?'}),"
             f" {rep['n_games']} games"]
    if rep.get("protocol"):
        lines.append(f"  protocol {rep.get('protocol_digest', '')}: "
                     f"{Protocol.from_dict(rep['protocol']).describe()}")
    lines.append("matchups (the first player's score, Wilson 95%, bootstrap over pairings):")
    for m in rep["matchups"]:
        lines.append(f"  {m['a']} vs {m['b']}: {_pct(m['score'])} [{_pct(m['lo'])}, "
                     f"{_pct(m['hi'])}] boot [{_pct(m['boot_lo'])}, {_pct(m['boot_hi'])}] over "
                     f"{m['n']} ({m['wins']} W {m['draws']} D {m['losses']} L)")
    if rep["paired"]:
        lines.append("paired (candidate - champion against the same opponent, same positions):")
        for p in rep["paired"]:
            lines.append(f"  {p['candidate']} - {p['champion']} vs {p['opponent']}: "
                         f"{100 * p['diff']:+.1f} pp [{100 * p['lo']:+.1f}, {100 * p['hi']:+.1f}]"
                         f" over {p['n']}, McNemar p {p['mcnemar']['p_value']:.3f}")
    r = rep["ratings"]
    lines.append(f"ratings (Bradley-Terry-Davidson, Elo, {r['anchor']} = {r['anchor_rating']:g}):")
    for row in r["rows"]:
        lines.append(f"  {row['player']}: {row['rating']:+.1f} +- {row['se']:.1f} "
                     f"({row['games']} games)")
    lines.append("gate:")
    for name, g in rep["gates"].items():
        lines.append(f"  {name}: {g['decision'].upper()} (vs champion {g['champion']})")
        for cname, c in g["conditions"].items():
            lines.append("    " + _condition_line(cname, c))
    lines.append(f"best: {rep['best'] or 'none passed'}")
    return "\n".join(lines)


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------

def _pairs_digest(pairs: Any, split: str) -> str:
    if hasattr(pairs, "digest") and hasattr(pairs, "pairs"):
        return pairs.digest()
    RM = _rm()
    return RM.PairSet(split, tuple(_pair_list(pairs))).digest()


def evaluate(pairs: Any, protocol: Protocol | None = None, *, candidates: Iterable[Any],
             champion: Any, anchors: Iterable[Any] = (), pool: Iterable[Any] = (),
             config: GateConfig | None = None, log_path: str | Path, workers: int = 1,
             play: Callable[[Any], Any] | None = None, stop_early: bool = True,
             rating_anchor: str | None = None, final: bool = False, decide: str | None = None,
             echo: Callable[[str], Any] | None = None, max_errors: int = MAX_ERRORS,
             presses: bool | None = None) -> dict:
    """Play what the gate needs and report (module doc). pairs: an rs_match.PairSet (or a list
    of DeckPairs) of the protocol's split; players: AgentSpecs, dicts or [NAME=]CKPT[,K=V] text
    (`decide` / `presses`: the defaults of the players that set none; V2 PLAYERS).
    Phases: every candidate against the champion; the champion and the surviving candidates
    against every anchor; the champion and the surviving candidates against the selected pool
    members (pool_selection; the champion's games give pool_collapse its reference on the same
    positions). stop_early=False plays every phase for every candidate. play(spec) ->
    MatchResult replaces the engine (tests); workers > 1 runs worker processes (worker_cap)."""
    config = config or GateConfig()
    protocol = (protocol or Protocol()).resolved()
    if protocol.split == FINAL_SPLIT and not final:
        raise FinalSplitError(FINAL_MESSAGE)
    if not hasattr(pairs, "pairs"):
        pairs = list(pairs)                      # read more than once below
    # The pair set's split and every pair's own (a plain list of DeckPairs has no set split): a
    # final-split pair must not slip in under a dev protocol.
    named = [getattr(pairs, "split", None)] + [getattr(p, "split", None) for p in _pair_list(pairs)]
    other = sorted({SPLIT_ALIASES.get(s, s) for s in named if s} - {protocol.split})
    if other:
        raise ValueError(f"the pairs are of the {', '.join(other)} split, the protocol's is "
                         f"{protocol.split}")
    digest = _pairs_digest(pairs, protocol.split)
    if protocol.pairs_digest and protocol.pairs_digest != digest:
        raise ValueError(f"the protocol is for pair set {protocol.pairs_digest}, these pairs are "
                         f"{digest}")
    protocol = protocol.replace(pairs_digest=digest)
    pd = protocol.digest()
    roles = prepare_players(candidates, champion, anchors, pool, protocol, decide=decide,
                            presses=presses)
    players = roles.players
    digests = {n: ckpt_digest(a.ckpt) for n, a in players.items()}
    idents = {n: identity(a, digests[n]) for n, a in players.items()}
    ids = {n: identity_digest(i) for n, i in idents.items()}
    log_path = Path(log_path)
    log = read_log(log_path)
    check_identities(log, pd, ids)
    append_records(log_path, [{
        "kind": KIND_RUN, "schema": SCHEMA, "tool": TOOL, "protocol": pd,
        "protocol_spec": protocol.to_dict(), "split": protocol.split, "pairs_digest": digest,
        "players": {n: {**i, "id": ids[n]} for n, i in idents.items()},
        "roles": roles.to_dict(), "gate": config.to_dict(), "stop_early": bool(stop_early),
        "time": time.time()}])
    done = log.done(pd)
    if play is None and workers > 1:
        workers = worker_cap(workers)[0]
    plist = _pair_list(pairs)
    champ = roles.champion
    anchor_names = list(roles.anchors)
    rating_anchor = rating_anchor or (anchor_names[0] if anchor_names else champ)

    def plan(a: str, b: str, games: int, decks: str) -> list[Task]:
        return comparison_tasks(plist, protocol, players[a], players[b], games=games,
                                decks=decks, digests=digests)

    def run(tasks: list[Task], phase: str) -> dict:
        if echo is not None and tasks:
            echo(f"{phase}: {len(tasks)} games planned")
        return run_tasks(tasks, log_path, protocol=pd, split=protocol.split, done=done,
                         workers=workers, play=play, engine=protocol.engine, echo=echo,
                         max_errors=max_errors, ids=ids)

    def current() -> list[Game]:
        return read_log(log_path).of(pd)

    phases = {}
    tasks = [t for c in roles.candidates
             for t in plan(c, champ, config.champion_games, protocol.h2h_decks)]
    phases["champion"] = run(tasks, "candidates vs the champion")
    games = current()
    alive = [c for c in roles.candidates
             if not stop_early or champion_condition(games, c, champ, config).passed]
    tasks = []
    if alive and anchor_names:
        for a in anchor_names:
            tasks += plan(champ, a, config.anchor_games, H2H_A)
        for c in alive:
            for a in anchor_names:
                tasks += plan(c, a, config.anchor_games, H2H_A)
    phases["anchors"] = run(tasks, "anchors")
    games = current()
    if stop_early:
        alive = [c for c in alive if all(anchor_condition(games, c, champ, a, config).passed
                                         for a in anchor_names)]
    members = pool_selection(games, roles.pool, roles.candidates, champ, config, rating_anchor)
    tasks = []
    if alive and members:
        # The champion on the very positions the candidates play (pool_condition's reference).
        tasks = [t for m in members for t in plan(champ, m, config.pool_games, protocol.h2h_decks)]
        tasks += [t for c in alive for m in members
                  for t in plan(c, m, config.pool_games, protocol.h2h_decks)]
    phases["pool"] = run(tasks, "pool")
    final_log = read_log(log_path)
    rep = report(final_log.of(pd), candidates=roles.candidates, champion=champ,
                 anchors=anchor_names, pool=members, config=config, rating_anchor=rating_anchor,
                 final=final)
    rep.update(protocol=protocol.to_dict(), protocol_digest=pd, pairs_digest=digest,
               log=str(log_path), phases=phases, pool_offered=list(roles.pool),
               errors=sum(1 for e in final_log.errors if e.get("protocol") == pd))
    return rep


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _add_gate_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("gate (module doc, THE GATE)")
    g.add_argument("--champion-games", type=int, default=CHAMPION_GAMES)
    g.add_argument("--lower-bound", type=float, default=CHAMPION_LOWER_BOUND,
                   help="Wilson 95%% lower bound the candidate's score must exceed")
    g.add_argument("--anchor-games", type=int, default=ANCHOR_GAMES)
    g.add_argument("--anchor-tolerance", type=float, default=ANCHOR_TOLERANCE_PP,
                   metavar="PP", help="largest drop against an anchor, percentage points")
    g.add_argument("--pool-games", type=int, default=POOL_GAMES)
    g.add_argument("--pool-members", type=int, default=POOL_MEMBERS)
    g.add_argument("--n-boot", type=int, default=2000)
    g.add_argument("--boot-seed", type=int, default=0)
    g.add_argument("--rating-anchor", default=None, metavar="NAME",
                   help="the BTD table's 0 (default: the first anchor, else the champion)")
    ap.add_argument("--final", action="store_true",
                    help="allow the final split (IL_Replay test): the held-out final report")
    ap.add_argument("--report", type=Path, default=None,
                    help="write the report JSON here (default: next to the log)")


def _gate_config(args: argparse.Namespace) -> GateConfig:
    return GateConfig(champion_games=args.champion_games, champion_lower_bound=args.lower_bound,
                      anchor_games=args.anchor_games, anchor_tolerance_pp=args.anchor_tolerance,
                      pool_members=args.pool_members, pool_games=args.pool_games,
                      n_boot=args.n_boot, boot_seed=args.boot_seed)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="League evaluation and the promotion gate (L03): "
                                             "paired games in RoyaleSim, JSONL, CIs, ratings.")
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="play the games the gate needs (resumes the log), report")
    player = "[NAME=]CKPT[,KEY=VALUE...]"
    run.add_argument("--candidate", action="append", required=True, metavar=player,
                     help="a candidate (repeatable); CKPT 'idle' never plays; keys: lead, "
                          "decide, noise, noise_seed, visibility, schema, wait_keep, spell_aim, "
                          "hold_pending, seed_offset, allow_drift, presses")
    run.add_argument("--champion", required=True, metavar=player)
    run.add_argument("--anchor", action="append", default=[], metavar=player)
    run.add_argument("--pool", action="append", default=[], metavar=player,
                     help="pool members for the collapse check (cycle detection); the "
                          "champion plays them too, on the candidates' positions")
    run.add_argument("--decide", default=None,
                     help="decision rule of players that set none: live (default), legacy, sample")
    run.add_argument("--presses", action="store_true",
                     help="players that set no presses= may press their ability buttons when "
                          "their checkpoint's buttons trained (v2; default: never press)")
    run.add_argument("--split", default="dev", help="dev (IL_Replay val) or final (test, --final)")
    run.add_argument("--pairs", type=int, default=0,
                     help="pairings to sample when no pairs file exists (0: what the gate needs)")
    run.add_argument("--pairs-file", type=Path, default=None,
                     help="the frozen pairings (default: next to the log; read, or written once)")
    run.add_argument("--require", default="hog26",
                     help="the deck side a plays: hog26, none or 8 card names")
    run.add_argument("--corpus", type=Path, default=None, help="default <data>/il_replay")
    run.add_argument("--parts", type=int, default=0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--delay", type=int, default=DEFAULT_DELAY,
                     help="command delay in ticks for both seats (live: 21-22 + capture, ~24)")
    run.add_argument("--delay-jitter", type=int, default=0, metavar="J",
                     help="per-pairing delay drawn from delay +- J")
    run.add_argument("--noise", default=DEFAULT_NOISE,
                     help="perception noise of every policy seat: 'default', a profile JSON, "
                          "or 'none'")
    run.add_argument("--mem-lag", default=str(rs_memory.BOARD_LAG), metavar="TICKS|legacy")
    run.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    run.add_argument("--base-forms", default="a", choices=("a", "b", "ab", "none"),
                     help="sides played in base forms (default a: the bot's account)")
    run.add_argument("--h2h-decks", default=H2H_BOTH, choices=(H2H_BOTH, H2H_A),
                     help="head-to-head and pool games: both (the decks exchanged too) or a")
    run.add_argument("--full", action="store_true",
                     help="play every phase for every candidate (no early stop)")
    run.add_argument("--workers", type=int, default=4,
                     help=f"worker processes, capped for memory (about {WORKER_MB} MB each)")
    run.add_argument("--max-errors", type=int, default=MAX_ERRORS)
    run.add_argument("--allow-vocab-drift", action="store_true")
    run.add_argument("--log", type=Path, default=None,
                     help="the JSONL log (default runs/league/<split>.jsonl)")
    _add_gate_args(run)
    rs_engine.add_engine_args(run)
    rep = sub.add_parser("report", help="report a log without playing (names, not checkpoints)")
    rep.add_argument("--log", type=Path, required=True)
    rep.add_argument("--protocol", default=None, metavar="DIGEST",
                     help="which protocol of the log (default: the only one)")
    rep.add_argument("--candidate", action="append", default=None, metavar="NAME",
                     help="default: the roles of the protocol's last run")
    rep.add_argument("--champion", default=None, metavar="NAME")
    rep.add_argument("--anchor", action="append", default=None, metavar="NAME")
    rep.add_argument("--pool", action="append", default=None, metavar="NAME")
    _add_gate_args(rep)
    return ap


def _default_log(split: str) -> Path:
    return ROOT / "runs" / "league" / f"{SPLIT_NAMES.get(split, split)}.jsonl"


def _report_path(args: argparse.Namespace, log: Path) -> Path:
    return args.report if args.report is not None else log.with_suffix(".report.json")


def _cmd_report(ap: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    log = read_log(args.log)
    if not log.games:
        print(f"error: no games in {args.log}", file=sys.stderr)
        return 1
    protocols = log.protocols()
    pd = args.protocol
    if pd is None:
        if len(protocols) > 1:
            listed = ", ".join(f"{p} ({n} games)" for p, n in protocols.most_common())
            ap.error(f"{args.log} holds several protocols: {listed}; pass --protocol DIGEST")
        pd = next(iter(protocols))
    elif pd not in protocols:
        ap.error(f"protocol {pd} has no games in {args.log}")
    header = log.last_run(pd) or {}
    roles = header.get("roles") or {}
    candidates = args.candidate or roles.get("candidates") or []
    champion = args.champion or roles.get("champion")
    if not candidates or not champion:
        ap.error("name the --candidate(s) and the --champion (no run header gives them)")
    anchors = args.anchor if args.anchor is not None else roles.get("anchors") or []
    offered = args.pool if args.pool is not None else roles.get("pool") or []
    try:
        config = _gate_config(args)
    except ValueError as exc:
        ap.error(str(exc))
    games = log.of(pd)
    anchors = [a for a in anchors if a != champion]
    rating_anchor = args.rating_anchor or (anchors[0] if anchors else champion)
    members = pool_selection(games, offered, candidates, champion, config, rating_anchor)
    try:
        rep = report(games, candidates=candidates, champion=champion, anchors=anchors,
                     pool=members, config=config, rating_anchor=rating_anchor, final=args.final)
    except FinalSplitError as exc:
        ap.error(str(exc))
    rep.update(protocol=header.get("protocol_spec"), protocol_digest=pd, log=str(args.log),
               pool_offered=list(offered),
               errors=sum(1 for e in log.errors if e.get("protocol") == pd))
    print(format_report(rep))
    if log.bad_lines or log.duplicates:
        print(f"({log.bad_lines} unreadable line(s), {log.duplicates} duplicate game(s) skipped)")
    print(f"report: {write_json(_report_path(args, args.log), rep)}")
    return 0


def _cmd_run(ap: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    RM = _rm()
    try:
        cfg = rs_engine.config_from_args(args)
        split = RM.corpus_split(args.split, allow_final=args.final)
        require = RM._require(args.require)
        protocol = Protocol(split=split, seed=args.seed, delay=args.delay,
                            delay_jitter=args.delay_jitter, noise=args.noise,
                            mem_lag=args.mem_lag, max_ticks=args.max_ticks,
                            base_forms=("a" in args.base_forms, "b" in args.base_forms),
                            h2h_decks=args.h2h_decks, engine=cfg.to_json()).resolved()
        config = _gate_config(args)
        roles = prepare_players(args.candidate, args.champion, args.anchor, args.pool, protocol,
                                decide=args.decide, presses=True if args.presses else None)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        ap.error(str(exc))
    rs_engine.set_active(cfg)
    import rs_checkpoint
    if args.allow_vocab_drift:
        rs_checkpoint.set_allow_drift(True)              # the workers inherit it
    for name, agent in list(roles.players.items()):
        if agent.idle:
            continue
        try:
            rs_checkpoint.check_file(agent.ckpt, cfg)
        except rs_checkpoint.VocabMismatch as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            return 1
        except FileNotFoundError as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            return 1
        if agent.presses:
            # V2 PLAYERS: the switch stays only where the checkpoint's buttons trained.
            meta = rs_checkpoint._load_blob(agent.ckpt)["meta"]
            roles.players[name] = presses_effective(agent, meta)
            if not roles.players[name].presses:
                print(f"note: {name} cannot press (its checkpoint's ability buttons did not "
                      "train): presses off", flush=True)
    log_path = args.log if args.log is not None else _default_log(split)
    pairs_path = (args.pairs_file if args.pairs_file is not None
                  else log_path.with_name(log_path.stem + ".pairs.json"))
    need = pairings_for(config, protocol, pool=bool(roles.pool))
    if pairs_path.exists():
        try:
            pairs = RM.PairSet.load(pairs_path)
        except ValueError as exc:
            print(f"error: {pairs_path}: {exc}", file=sys.stderr)
            return 1
        if pairs.split != split:
            ap.error(f"{pairs_path} holds the {pairs.split} split, not {split}")
    else:
        if args.pairs and args.pairs < need:
            ap.error(f"--pairs {args.pairs}: the gate needs {need} pairings")
        try:
            pairs = RM.load_pairs(args.corpus, args.split, parts=args.parts, require=require,
                                  allow_final=args.final)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        pairs = pairs.sample(args.pairs or need, args.seed)
        if len(pairs) >= need:
            pairs.save(pairs_path)                       # frozen: every later run reads it
    if len(pairs) < need:
        print(f"error: the gate needs {need} pairings; the {split} split gives {len(pairs)}"
              f"{' in ' + str(pairs_path) if pairs_path.exists() else ''}", file=sys.stderr)
        return 1
    workers, why = worker_cap(args.workers)
    print(f"engine: {cfg.describe()}\nprotocol: {protocol.describe()}\npairs: {len(pairs)} of "
          f"the {split} split [{pairs.digest()}] in {pairs_path}\nlog: {log_path}\nworkers: {why}",
          flush=True)
    try:
        rep = evaluate(pairs, protocol, candidates=list(roles.players[c] for c in roles.candidates),
                       champion=roles.players[roles.champion],
                       anchors=[roles.players[a] for a in roles.anchors],
                       pool=[roles.players[m] for m in roles.pool], config=config,
                       log_path=log_path, workers=workers, stop_early=not args.full,
                       rating_anchor=args.rating_anchor, final=args.final,
                       echo=lambda s: print(s, flush=True), max_errors=args.max_errors)
    except FinalSplitError as exc:
        ap.error(str(exc))
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report(rep))
    print(f"report: {write_json(_report_path(args, log_path), rep)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command == "report":
        return _cmd_report(ap, args)
    return _cmd_run(ap, args)


if __name__ == "__main__":
    sys.exit(main())
