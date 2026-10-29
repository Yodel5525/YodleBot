r"""Opponent elixir, counted the way a strong player counts it: from the plays they see.

    .venvs\royalegym\Scripts\python.exe tools\rs_elixir.py           the elixir law in use
    .venvs\royalegym\Scripts\python.exe tools\rs_elixir.py --json    the same as JSON

Why (tokens-14): the screen never shows the opponent's elixir bar, but every play is
public, the starting bar is public and so is the regeneration rate. Pay for each play
seen, regenerate over the ticks that passed, clamp at the cap: the count is the engine's
own arithmetic. This module is pure Python, needs nothing installed, and follows
royalegym's MatchMemory (royalegym 0.1.25 obs.py:518-905 and protocol.py ElixirLaw) without
importing it.

THE LAW (ElixirLaw). Read from royalesim's data/calibration.json when royalesim can be found
(royalesim.data_dir(), else the installed package's data/ folder, found without importing
the compiled engine); otherwise from CONSTANTS, the values royalesim 0.1.42 ships:

    match.MANA_REGEN_MS_1X 28000, _2X 14000, _OVERTIME 9300   ms to fill a 10-elixir bar
    match.MANA_SPEED_UP_WHEN_REMAINING_SECONDS 60             2x from 120 s (tick 2400)
    match.REGULAR_TIME_S 180                                   overtime from tick 3600
    match.MANA_TRIPLE_AFTER_OVERTIME_S 60                      3x from tick 4800
    match.START_MANA 6, match.MAX_MANA 10, time.TICK_MS 50

Integer fine units, as the engine keeps them: one elixir is lcm(regen 1x, regen 2x) =
28000 units, a tick adds 500 at 1x and 1000 at 2x, and 1505 at 3x (truncated from
1505.38, as the engine truncates it). state_json's elixir_milli is fine * 1000 // 28000.
Every value is an int, so the count is bit-exact against the engine's bar, not near it.
The rate is a function of the tick: the tick that runs FROM t is 2x when t >= 2400 and 3x
when t >= 4800 (a battle runs past 3600 only in overtime).

THE COUNT (ElixirCounter, one bar; the bot keeps one for the opponent and may keep one
for itself). It is a pure function of a seed and a log of dated events, so a play reported
late (vision lag) or out of order gives the same bar as one reported on time:

    play(tick, card)       a card play, paid when it EXECUTES (under a command delay that is
                           the accepted tick + the delay; IL_Replay logs execution ticks).
                           Priced from `costs`; a MIRROR costs the side's last non-Mirror
                           play plus its own 1 (match.MIRROR_COST_RULE, MIRROR_RECORD); a
                           Mirror with nothing to copy means a play was missed. cost=
                           overrides the price (a variant card's form, a hand_costs reading)
                           and is what a later Mirror copies.
    press(tick, cost)      an ability press (hero, champion), paid like a play, priced at the
                           button's cost; it is not a play, so a Mirror does not copy it.
    credit(tick, milli)    elixir paid TO the side (an Elixir Collector payout, an Elixir
                           Golem's ManaOnDeathForOpponent). Paid after the tick's regen; one
                           due at the cap is held until the first tick below it
                           (economy.PRODUCTION_AT_CAP); an overflow is lost.

A play dated t is paid on tick t, the engine's convention: state_json at t still shows the
bar before it, the state at t + 1 shows it paid plus one tick of regen. So:

    bar_milli(t)           what the engine's state at t shows (events dated t not yet paid)
    elixir_milli(t)        what the side holds at t after the plays dated t: spendable now

`exact` says whether the count can be trusted. It goes False, and stays False, when the
counted bar disagrees with an observed one (check / resync / observe), when a seen play is
unaffordable by the count (the engine refuses those, so a play or a payout was missed), or
when a Mirror is seen with nothing to copy. A wrong opponent count is never repaired from
the opponent's real bar: repairing it is the cheat. resync() re-seeds a bar the side can
see (its own).

observe(state, team) reads a state_json dict the way MatchMemory does: seeded once from
the bar at the first state, plays read from hand slots changing (dated at the previous
observation, or when a waiting pending row said it would run), presses read from the
`abilities` rows, idempotent by tick, re-seeded when the tick goes back; then checked
against the state's own bar. The hand-slot reading misses a play when a deck repeats a
card (a real deck never does); under a command delay, presses are dated at the previous
observation + the delay, which is approximate: feed press() the ticks they ran instead
(Battle.step_commands_run()).

Not modelled: a Collector's own payout timer (feed credit() its payouts), other battle
modes' base rates (double/triple elixir from the start), and the game's 1/10000-elixir
rounding (178 a tick where the engine adds 178.57: the engine is what is modelled).
Card costs depend on the card table: build the counter from the catalogue of a Battle
built under the same rs_engine.EngineConfig (from_catalogue).
"""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import importlib.util
import json
import math
import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

#: Fallback values: royalesim 0.1.42 data/calibration.json (measured on clients 16.402 and
#: 15.535.29). Used key by key when the file is absent or lacks a key.
CONSTANTS: dict[str, Any] = {
    "time.TICK_MS": 50,
    "match.MANA_REGEN_MS_1X": 28000,
    "match.MANA_REGEN_MS_2X": 14000,
    "match.MANA_REGEN_MS_OVERTIME": 9300,
    "match.MANA_SPEED_UP_WHEN_REMAINING_SECONDS": 60,
    "match.MANA_TRIPLE_AFTER_OVERTIME_S": 60,
    "match.REGULAR_TIME_S": 180,
    "match.START_MANA": 6,
    "match.MAX_MANA": 10,
    "match.MIRROR_COST_RULE": "copied_elixir_plus_own_elixir",
}
MIRROR_RULE = "copied_elixir_plus_own_elixir"
EMPTY_CARD = -1
#: royalesim catalogue columns read by position (royalegym rust_engine.py CATALOGUE_PREFIX);
#: placement code 6 is the Mirror.
CATALOGUE_PREFIX: tuple[str, ...] = (
    "name", "placement", "elixir", "count", "radius", "flying", "hitpoints", "footprint_tiles",
)
MIRROR_PLACEMENT = 6
#: Entity row columns observe() reads (royalesim ENTITY_FIELDS: uid, team, kind, card_id, ...).
ENTITY_TEAM, ENTITY_CARD = 1, 3


# ---------------------------------------------------------------------------
# the law
# ---------------------------------------------------------------------------

def calibration_path() -> Path | None:
    """royalesim's data/calibration.json, or None when royalesim cannot be found.

    An imported royalesim with data_dir() is asked first (a checkout build answers with the
    checkout's data/); otherwise the installed package is located without importing it."""
    mod = sys.modules.get("royalesim")
    candidates: list[Path] = []
    data_dir = getattr(mod, "data_dir", None)
    if callable(data_dir):
        try:
            candidates.append(Path(data_dir()) / "calibration.json")
        except Exception:                        # a broken install must not stop the count
            pass
    try:
        spec = importlib.util.find_spec("royalesim")
    except (ImportError, ValueError):            # ValueError: a module without __spec__
        spec = None
    if spec is not None:
        for folder in spec.submodule_search_locations or ():
            candidates.append(Path(folder) / "data" / "calibration.json")
        if spec.origin:
            candidates.append(Path(spec.origin).parent / "data" / "calibration.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclasses.dataclass(frozen=True)
class ElixirLaw:
    """How a bar fills, in integer fine units (see the module doc)."""

    scale: int                  # fine units per elixir
    gain_1x: int                # fine units a tick at 1x
    gain_2x: int
    gain_3x: int                # 0: no third rate
    cap_fine: int               # MAX_MANA in fine units
    start_fine: int             # START_MANA in fine units
    tick_ms: int
    regular_ticks: int          # regulation length in ticks (3600)
    speedup_ticks: int          # 2x from regular_ticks - speedup_ticks (2400)
    triple_ticks: int           # 3x from regular_ticks + triple_ticks (4800)
    mirror_rule: str = MIRROR_RULE
    source: str = dataclasses.field(default="constants", compare=False)

    # ------------------------------------------------------------ building
    @classmethod
    def constants(cls) -> "ElixirLaw":
        """The law from CONSTANTS alone."""
        return cls.from_calibration({}, source="constants")

    @classmethod
    def load(cls, calibration: Mapping | None = None, *, path: str | Path | None = None,
             overrides: Mapping[str, Any] | None = None) -> "ElixirLaw":
        """From a parsed calibration.json, the file at `path`, or royalesim's own
        (calibration_path()); CONSTANTS when there is none. `overrides` are ledger keys as
        rs_engine.EngineConfig.calibration_overrides holds them ("section.KEY" -> value or
        its JSON text), so a battle run with overrides is counted under them."""
        source = "calibration"
        if calibration is None:
            found = Path(path) if path is not None else calibration_path()
            if found is None:
                calibration, source = {}, "constants"
            else:
                calibration = json.loads(found.read_text(encoding="utf-8"))
                source = str(found)
        return cls.from_calibration(calibration, overrides=overrides, source=source)

    @classmethod
    def from_calibration(cls, calibration: Mapping, *, overrides: Mapping[str, Any] | None = None,
                         source: str = "calibration") -> "ElixirLaw":
        def get(key: str) -> Any:
            if overrides and key in overrides:
                value = overrides[key]
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except ValueError:
                        pass
                return value
            section, _, name = key.partition(".")
            entry = (calibration.get(section) or {}).get(name)
            value = entry.get("value") if isinstance(entry, Mapping) else entry
            return CONSTANTS[key] if value is None else value

        tick_ms = int(get("time.TICK_MS"))
        r1, r2 = int(get("match.MANA_REGEN_MS_1X")), int(get("match.MANA_REGEN_MS_2X"))
        r3 = int(get("match.MANA_REGEN_MS_OVERTIME"))
        max_mana = int(get("match.MAX_MANA"))
        if min(tick_ms, r1, r2, max_mana) <= 0:
            raise ValueError("calibration: TICK_MS, MANA_REGEN_MS_1X / 2X and MAX_MANA must be > 0")
        scale = math.lcm(r1, r2)
        # scale is a multiple of both rates, so these two gains are exact integers.
        gain_1x, gain_2x = (tick_ms * max_mana * scale // r for r in (r1, r2))
        regular_ms = int(get("match.REGULAR_TIME_S")) * 1000
        return cls(
            scale=scale, gain_1x=gain_1x, gain_2x=gain_2x,
            gain_3x=tick_ms * max_mana * scale // r3 if r3 > 0 else 0,
            cap_fine=max_mana * scale,
            start_fine=int(get("match.START_MANA")) * scale,
            tick_ms=tick_ms,
            regular_ticks=regular_ms // tick_ms,
            speedup_ticks=_ceil_div(int(get("match.MANA_SPEED_UP_WHEN_REMAINING_SECONDS")) * 1000,
                                    tick_ms),
            triple_ticks=_ceil_div(int(get("match.MANA_TRIPLE_AFTER_OVERTIME_S")) * 1000, tick_ms),
            mirror_rule=str(get("match.MIRROR_COST_RULE")),
            source=source,
        )

    def with_regular_ticks(self, regular_ticks: int) -> "ElixirLaw":
        """The same law for a battle whose regulation lasts `regular_ticks` (state_json's)."""
        return dataclasses.replace(self, regular_ticks=int(regular_ticks))

    # ------------------------------------------------------------ units
    def to_milli(self, fine: int) -> int:
        """Fine units -> milli-elixir as state_json reports it (a floor)."""
        return fine * 1000 // self.scale

    def from_milli(self, milli: int) -> int:
        """Milli-elixir -> fine units as the engine seeds a bar (reset's elixir_milli)."""
        return milli * self.scale // 1000

    def seed_fine(self, milli: int) -> int:
        """A reported bar back into fine units, snapped onto the lattice a running bar
        lives on (start plus gains minus whole-elixir spends: multiples of
        gcd(gain_1x, gain_2x)); a bar off it (an odd seed) keeps from_milli's floor.
        royalegym ElixirLaw.seed_fine; past the 3x switch the lattice is finer and the
        snap can be off by under one milli, which check() then reports."""
        lo = self.from_milli(milli)
        hi = lo + max(0, self.scale // 1000 - 1)
        spacing = math.gcd(self.gain_1x, self.gain_2x)
        snapped = _ceil_div(lo, spacing) * spacing
        return snapped if snapped <= hi else lo

    # ------------------------------------------------------------ rate
    @property
    def double_from(self) -> int:
        """The first tick that runs at 2x (2400)."""
        return self.regular_ticks - self.speedup_ticks

    @property
    def triple_from(self) -> int | None:
        """The first tick that runs at 3x (4800), None without a third rate."""
        return self.regular_ticks + self.triple_ticks if self.gain_3x else None

    def rate_at(self, tick: int) -> int:
        """The multiplier of the tick that runs FROM `tick`: 1, 2 or 3."""
        triple = self.triple_from
        if triple is not None and tick >= triple:
            return 3
        return 2 if tick >= self.double_from else 1

    def gain(self, rate: int) -> int:
        gains = {1: self.gain_1x, 2: self.gain_2x, 3: self.gain_3x or self.gain_2x}
        if rate not in gains:
            raise ValueError(f"unknown elixir rate {rate}")
        return gains[rate]

    def regen(self, tick_from: int, tick_to: int) -> int:
        """Fine units gained over the ticks [tick_from, tick_to), ignoring the cap."""
        if tick_to <= tick_from:
            return 0
        slow = max(0, min(tick_to, self.double_from) - tick_from)
        triple = self.triple_from
        fastest = 0 if triple is None else max(0, tick_to - max(tick_from, triple))
        fast = (tick_to - tick_from) - slow - fastest
        return slow * self.gain_1x + fast * self.gain_2x + fastest * self.gain_3x

    def summary(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "double_from": self.double_from,
                "triple_from": self.triple_from}


@lru_cache(maxsize=1)
def default_law() -> ElixirLaw:
    """ElixirLaw.load(), read once per process."""
    return ElixirLaw.load()


# ---------------------------------------------------------------------------
# what a player sees
# ---------------------------------------------------------------------------

def cards_that_left(before: Sequence, after: Sequence) -> list:
    """Cards played, read from hand slots changing (royalegym obs.cards_that_left)."""
    if len(before) != len(after):
        return []
    return [b for b, a in zip(before, after) if b != a and b != EMPTY_CARD]


def presses_seen(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]],
                 standing: set) -> list[int]:
    """Costs of the presses between two views of a side's `abilities` rows ([available,
    spent, cost, card_id, cooldown_ticks, ...]): a hero's charge turning spent, or a
    champion's button going dark while a unit of its card still stands. A button gone dark
    with its unit gone is a death, not a press (royalegym obs.presses_seen)."""
    out = []
    for b, a in zip(before, after):
        a_card = int(a[3]) if len(a) > 3 else EMPTY_CARD
        spent = not int(b[1]) and bool(int(a[1]))
        dark = bool(int(b[0])) and not int(a[0]) and not int(a[1]) and a_card in standing
        if spent or dark:
            out.append(int(a[2]))
    return out


# ---------------------------------------------------------------------------
# the count
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Bar:
    """The fold's state at the START of `tick`, before the payments dated `tick`."""

    tick: int
    fine: int
    leak: int = 0                   # fine units lost to the cap
    held: int = 0                   # payouts held at the cap, fine units
    last_card: Hashable | None = None
    last_cost: int = 0
    unaffordable: int = 0
    blind_mirrors: int = 0


@dataclasses.dataclass(frozen=True)
class Event:
    tick: int
    seq: int
    kind: str                       # "play", "press" or "credit"
    card: Hashable | None = None
    amount: int | None = None       # play: explicit cost or None; press: cost; credit: fine units


@dataclasses.dataclass(frozen=True)
class _Seen:
    tick: int
    hand: list
    rows: list
    pending: list


class ElixirCounter:
    """One side's elixir bar, counted from dated public events (see the module doc).

    costs   card -> whole elixir: a Mapping (card ids or names) or a Sequence indexed by
            card id. Every card played without cost= must be in it.
    mirror  the cards that are Mirrors; default: the keys named "Mirror" (any case).
    law     an ElixirLaw; default default_law().
    tick, elixir_milli  the seed; elixir_milli None is the law's START_MANA.
    delay   the side's command delay in ticks, used only by observe().
    """

    def __init__(self, costs: Mapping | Sequence[int] | None = None, *, mirror: Iterable = (),
                 law: ElixirLaw | None = None, tick: int = 0, elixir_milli: int | None = None,
                 delay: int = 0) -> None:
        if costs is None:
            costs = {}
        self.costs: dict = dict(enumerate(costs)) if not isinstance(costs, Mapping) else dict(costs)
        named = {k for k in self.costs if isinstance(k, str) and k.lower() == "mirror"}
        self.mirror: frozenset = frozenset(mirror) | frozenset(named)
        self.law = law if law is not None else default_law()
        if self.law.mirror_rule != MIRROR_RULE:
            raise ValueError(f"match.MIRROR_COST_RULE {self.law.mirror_rule!r}: only "
                             f"{MIRROR_RULE!r} is counted here")
        self.delay = int(delay)
        self.start(tick, elixir_milli)

    @classmethod
    def from_catalogue(cls, rows: Sequence[Sequence], *, fields: Sequence[str] | None = None,
                       by_name: bool = False, **kwargs: Any) -> "ElixirCounter":
        """From Battle.catalogue_json() rows: elixir from the "elixir" column (index 2,
        or where `fields` - royalesim.CATALOGUE_FIELDS - puts it), Mirrors by placement code
        6 or by name. by_name keys the costs by card name instead of catalogue id."""
        fields = list(fields or CATALOGUE_PREFIX)
        if "elixir" not in fields:
            raise ValueError(f"catalogue fields {fields} have no 'elixir' column")
        at_elixir = fields.index("elixir")
        # Without a placement column the Mirror is known by its name alone.
        at_place = fields.index("placement") if "placement" in fields else None
        costs, mirror = {}, set()
        for cid, row in enumerate(rows):
            key = row[0] if by_name else cid
            costs[key] = int(row[at_elixir])
            if row[0] == "Mirror" or (at_place is not None and len(row) > at_place
                                      and row[at_place] == MIRROR_PLACEMENT):
                mirror.add(key)
        return cls(costs, mirror=set(kwargs.pop("mirror", ())) | mirror, **kwargs)

    # ------------------------------------------------------------ lifecycle
    def start(self, tick: int = 0, elixir_milli: int | None = None) -> None:
        """A new match (or a new seed): everything forgotten."""
        fine = self.law.start_fine if elixir_milli is None else self.law.seed_fine(int(elixir_milli))
        self._seed = _Bar(int(tick), fine)
        self._events: list[Event] = []
        self._keys: list[tuple[int, int]] = []
        self._seq = 0
        self._ckpt: _Bar | None = None
        self._broken = False
        self._seen: _Seen | None = None
        self.now = int(tick)
        self.mismatches: list[tuple[int, int, int]] = []    # (tick, counted, observed)

    # ------------------------------------------------------------ events
    def play(self, tick: int, card: Hashable, *, cost: int | None = None) -> None:
        """A play of `card` executed (paid) on `tick`. cost: the debit, when known."""
        if cost is None and card not in self.costs:
            raise ValueError(f"no elixir cost for card {card!r}: pass costs= or cost=")
        if cost is not None and int(cost) < 0:
            raise ValueError(f"negative cost {cost} for card {card!r}")
        self._add(Event(int(tick), 0, "play", card, None if cost is None else int(cost)))

    def press(self, tick: int, cost: int) -> None:
        """An ability press that ran on `tick`, costing `cost` elixir (the button's)."""
        if int(cost) < 0:
            raise ValueError(f"negative press cost {cost}")
        self._add(Event(int(tick), 0, "press", None, int(cost)))

    def credit(self, tick: int, elixir_milli: int) -> None:
        """Elixir paid to the side during the tick that runs from `tick`."""
        if int(elixir_milli) < 0:
            raise ValueError(f"negative credit {elixir_milli}")
        self._add(Event(int(tick), 0, "credit", None, self.law.from_milli(int(elixir_milli))))

    def _add(self, event: Event) -> None:
        if event.tick < self._seed.tick:
            raise ValueError(f"an event at tick {event.tick} is before this counter's seed "
                             f"(tick {self._seed.tick})")
        event = dataclasses.replace(event, seq=self._seq)
        self._seq += 1
        key = (event.tick, event.seq)
        at = bisect.bisect_right(self._keys, key)
        self._keys.insert(at, key)
        self._events.insert(at, event)
        if self._ckpt is not None and event.tick < self._ckpt.tick:
            self._ckpt = None                    # a late event rewrites what came after it

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def advance(self, tick: int) -> int:
        """Move `now` (the default query tick) forward to `tick`; returns elixir_milli(tick)."""
        if int(tick) < self.now:
            raise ValueError(f"tick {tick} is before now ({self.now}); start() a new match")
        self.now = int(tick)
        return self.elixir_milli(self.now)

    # ------------------------------------------------------------ the fold
    def _price(self, bar: _Bar, event: Event) -> int:
        """Whole elixir an event debits; updates the Mirror's copy target."""
        if event.kind == "press":
            return int(event.amount or 0)
        card = event.card
        if card in self.mirror:
            if event.amount is not None:
                return event.amount
            own = int(self.costs.get(card, 1))
            if bar.last_card is None:            # the engine refuses this: a play was missed
                bar.blind_mirrors += 1
                return own
            return bar.last_cost + own
        cost = event.amount if event.amount is not None else int(self.costs[card])
        bar.last_card, bar.last_cost = card, cost
        return cost

    def _fill(self, bar: _Bar, tick: int) -> None:
        if tick > bar.tick:
            raw = bar.fine + self.law.regen(bar.tick, tick)
            bar.fine = min(self.law.cap_fine, raw)
            bar.leak += raw - bar.fine
            bar.tick = tick

    def _run(self, start: _Bar, upto: int) -> _Bar:
        """`start` carried to the start of `upto`: every event dated before it applied."""
        law, cap = self.law, self.law.cap_fine
        bar = dataclasses.replace(start)
        i = bisect.bisect_left(self._keys, (bar.tick, -1))
        events = self._events
        while i < len(events) and events[i].tick < upto:
            tick = events[i].tick
            self._fill(bar, tick)
            due, credit = 0, 0
            while i < len(events) and events[i].tick == tick:
                event = events[i]
                i += 1
                if event.kind == "credit":
                    credit += int(event.amount or 0)
                    continue
                due += self._price(bar, event) * law.scale
                if due > bar.fine:
                    bar.unaffordable += 1
            # The engine pays commands before the tick, then regenerates, then pays
            # producers (state.rs mana_pass); a payout due at the cap waits for the first
            # tick below it.
            raw = max(0, bar.fine - due) + law.gain(law.rate_at(tick))
            bar.fine = min(cap, raw)
            bar.leak += raw - bar.fine
            credit, bar.held = credit + bar.held, 0
            if credit and bar.fine < cap:
                raw = bar.fine + credit
                bar.fine = min(cap, raw)
                bar.leak += raw - bar.fine
            elif credit:
                bar.held = credit
            bar.tick = tick + 1
        self._fill(bar, upto)
        return bar

    def _bar(self, tick: int) -> _Bar:
        tick = int(tick)
        if tick < self._seed.tick:
            raise ValueError(f"tick {tick} is before this counter's seed (tick {self._seed.tick})")
        base = self._ckpt if self._ckpt is not None and self._ckpt.tick <= tick else self._seed
        bar = self._run(base, tick)
        if self._ckpt is None or bar.tick > self._ckpt.tick:
            self._ckpt = bar
        return bar

    def _horizon(self) -> int:
        last = self._events[-1].tick + 1 if self._events else self._seed.tick
        return max(self.now, last)

    # ------------------------------------------------------------ queries
    def bar_fine(self, tick: int | None = None) -> int:
        return self._bar(self.now if tick is None else tick).fine

    def bar_milli(self, tick: int | None = None) -> int:
        """The bar as the engine's state_json shows it at `tick` (default now): the events
        dated `tick` are not paid yet."""
        return self.law.to_milli(self.bar_fine(tick))

    def elixir_milli(self, tick: int | None = None) -> int:
        """What the side holds at `tick` (default now) after the plays and presses dated
        `tick`: the bar it can spend from."""
        tick = self.now if tick is None else int(tick)
        bar = dataclasses.replace(self._bar(tick))
        lo = bisect.bisect_left(self._keys, (tick, -1))
        hi = bisect.bisect_left(self._keys, (tick + 1, -1))
        due = sum(self._price(bar, e) for e in self._events[lo:hi] if e.kind != "credit")
        return self.law.to_milli(max(0, bar.fine - due * self.law.scale))

    def elixir(self, tick: int | None = None) -> float:
        return self.elixir_milli(tick) / 1000

    def leaked_milli(self, tick: int | None = None) -> int:
        """Elixir lost to the cap up to `tick` (default now), milli."""
        return self.law.to_milli(self._bar(self.now if tick is None else tick).leak)

    @property
    def last_play(self) -> Hashable | None:
        """What a Mirror played now would copy (the last non-Mirror play up to now)."""
        return self._bar(self._horizon()).last_card

    @property
    def unaffordable(self) -> int:
        """Plays and presses the count could not pay (0 on an engine's own log)."""
        return self._bar(self._horizon()).unaffordable

    @property
    def exact(self) -> bool:
        """False, for the rest of the match, once the count is known to be wrong."""
        if self._broken:
            return False
        bar = self._bar(self._horizon())
        return not (bar.unaffordable or bar.blind_mirrors)

    # ------------------------------------------------------------ an observed bar
    def check(self, tick: int, elixir_milli: int) -> bool:
        """Compare the count with a bar seen at `tick` (as state_json reports it). A
        mismatch sets exact False for good and is recorded in `mismatches`; the count is
        left as it was."""
        tick = int(tick)
        counted = self.bar_milli(tick)
        self.now = max(self.now, tick)
        if counted == int(elixir_milli):
            return True
        self._broken = True
        self.mismatches.append((tick, counted, int(elixir_milli)))
        return False

    def resync(self, tick: int, elixir_milli: int) -> bool:
        """check(), then re-seed from the seen bar: for a bar the side can see (its own).
        Events before `tick` are dropped (the seen bar has them); the Mirror's copy target,
        the leak and held payouts carry over."""
        ok = self.check(tick, elixir_milli)
        tick = int(tick)
        bar = self._bar(tick)
        if bar.unaffordable or bar.blind_mirrors:
            self._broken = True
        self._seed = _Bar(tick, self.law.seed_fine(int(elixir_milli)), leak=bar.leak,
                          held=bar.held, last_card=bar.last_card, last_cost=bar.last_cost)
        keep = bisect.bisect_left(self._keys, (tick, -1))
        del self._events[:keep], self._keys[:keep]
        self._ckpt = None
        return ok

    # ------------------------------------------------------------ from engine states
    def observe(self, state: Mapping, team: int, *, repair: bool = False) -> bool:
        """Advance to a state_json dict, counting `team`'s bar from what it shows publicly
        (see the module doc). A no-op unless the tick moved forward; a tick that moved back
        re-seeds. Returns whether the count matches the state's bar; repair=True re-seeds
        from it (resync), for the observer's own bar only."""
        tick = int(state["tick"])
        me = state["players"][team]
        seen = _Seen(tick, list(me.get("hand", ())), [list(r) for r in me.get("abilities", ())],
                     [list(r) for r in me.get("pending", ())])
        prev = self._seen
        if prev is None or tick < prev.tick:
            regular = state.get("regular_ticks")
            if regular is not None and int(regular) != self.law.regular_ticks:
                self.law = self.law.with_regular_ticks(int(regular))
            self.start(tick, int(me["elixir_milli"]))
            self._seen = seen
            return True
        if tick == prev.tick:
            return not self.mismatches or self.mismatches[-1][0] != tick
        last = tick - 1
        waiting = list(prev.pending)
        for card in cards_that_left(prev.hand, seen.hand):
            due = prev.tick + self.delay
            for i, row in enumerate(waiting):
                if row and row[0] in (0, "deploy") and row[1] == card:
                    due = prev.tick + int(row[4])
                    del waiting[i]
                    break
            self.play(min(max(due, prev.tick), last), card)
        standing = set()
        for entity in state.get("entities", ()):
            if isinstance(entity, Mapping):
                if entity.get("team") == team:
                    standing.add(entity.get("card_id"))
            elif len(entity) > ENTITY_CARD and entity[ENTITY_TEAM] == team:
                standing.add(entity[ENTITY_CARD])
        for cost in presses_seen(prev.rows, seen.rows, standing):
            self.press(min(prev.tick + self.delay, last), cost)
        self._seen = seen
        milli = int(me["elixir_milli"])
        return self.resync(tick, milli) if repair else self.check(tick, milli)


# ---------------------------------------------------------------------------
# CLI: which law is in use
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print the elixir law the counter uses.")
    ap.add_argument("--json", action="store_true", help="print it as JSON")
    args = ap.parse_args(argv)
    law = ElixirLaw.load()
    info = law.summary()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"elixir law from {law.source}")
    print(f"  one elixir = {law.scale} fine units; a tick adds {law.gain_1x} (1x), "
          f"{law.gain_2x} (2x), {law.gain_3x} (3x)")
    print(f"  start {law.to_milli(law.start_fine)} milli, cap {law.to_milli(law.cap_fine)} milli, "
          f"tick {law.tick_ms} ms")
    print(f"  2x from tick {law.double_from}, overtime from {law.regular_ticks}, "
          f"3x from {law.triple_from}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
