r"""The rebuilder: one reused RoyaleSim Battle reset from a public view (S04).

    from tools import rs_rebuild as RB              # or `import rs_rebuild` (tools/ on sys.path)
    rb = RB.Rebuilder(cfg, royalesim, spawn_table, settings=settings)   # pinned (S08: SearchConfig)
    view = keep.apply(PV.public_view(seen, team, mem, ...))   # S02: public view + Permanence
    spec = rb.spec(view, land_tick=B, seed=seed)    # a RootSpec: pure, picklable, spec.key
    built = rb.root(spec)                           # a Built at B, or None (rb.last_error.reason)
    if built is None or built.fidelity < rb.settings.min_fidelity: play d0
    rb.legal(card, x, y, built)                     # check_deploy at B, the card mapped to its slot
    after, report = rb.trial(built, [(B, team, card, x, y)], H)    # relabelled after-state

Why (docs/LIVE_SEARCH_PLAN.md sections 3.1 steps 3-4, 4.2-4.8; S04). The search decides on the
board at the tick its play would land, B = t + D, rebuilt in RoyaleSim from what the screen shows.
The author's rebuild (tools/rs_eval.py RSAgent.rebuilt, rs_eval.py:324-391, and the look-ahead
stepping of RSAgent.ahead with lead_rebuild, rs_eval.py:269-300) kept about 7 of the 24 points a
search on the true copy is worth (plan 2.1). This module rebuilds from the S02 public view only,
with our queue in cycle order, the counted opponent elixir, the creating card restored on every
unit (relabel), our own plays in flight stepped in at their landing ticks, and deploy-timer and
spell recasts.

LEVELS (plan 4.8; RebuildSettings.level, or spec(level=)):
  R0  RSAgent.rebuilt exactly, from the state the seat sees (spec_r0): opponent elixir 0, our
      queue in deck order, the opponent's deck as the state has it, no relabel, no snap; our
      plays in flight stepped in as RSAgent.ahead does. Its reset arguments are rebuilt's, call for
      call; it exists so the regret instrument can reproduce the author's finding.
  R1  the public view (S02), public_state.sanitize_for_rebuild (public_state.py:935), card cycles
      for both sides, the counted opponent elixir, relabel, Permanence's kept rows (spawned like any
      other row of the view), buildings snapped with Battle.building_placement, one reused Battle.
  R2  R1 plus deploy-timer recasts and spell recasts (below).

THE SPEC (RootSpec, build_spec / Rebuilder.spec). Everything a reset needs, decided before any
engine call but two pure queries (building_placement for the snap; formation_preview for a card's
members, deploy time and tap point; both asked on a blank board, never on whatever root the
battle holds, and cached, so a view rebuilt twice gives one spec): seed, both decks in deal order
(shuffle 0: hand = deck[0:4], queue = deck[4:8]), elixir, tower hp, the spawns (team, card, x, y,
hp) with each one's view uid and creating card, the recasts, our plays in flight (replays: exec
tick < B; later: exec tick >= B, for the trials), and the bookkeeping fidelity reads. A frozen
dataclass of tuples, so it pickles (the procs backend, plan 6.3) and spec.key is the sha1 of its
canonical JSON: two views that differ only in what no screen shows give one key (the S08 leak
test).

THE DEAL (deal()). Each side's target is its hand H and queue Q at t (CardCycle, S02; the
opponent's open slots filled from `opp_fill`, a deck prior's draw, then its seen deck padded with
Hog 2.6 filler; ours from our own deck). Some cards must be PLAYED by the rebuild before the hand
reads H: the recast cards and our in-flight cards (replays, then later ones), in play order R. A
card of R that is in H (our in-flight card while the hand still holds it: the engine and the live
client both pay and cycle a play when it runs, Battle docstring of set_command_delay_ticks,
rs_decide.prune_pending) is played from its slot as it is. A card of R that is in Q (a recast's
card, already cycled; or our in-flight card when the hand already shows it cycled, an engine that
cycles at send, probe P16) is re-dealt: |R in Q| hand slots C are given to those cards, and the C
cards go to the head of the queue in that order, so each play of R[i] from C[i]'s slot brings
C[i] back:
  pre-hand = H with slot(C[i]) := R[i];   pre-queue = C + (Q minus R)
After the plays the hand is H, slot for slot, and the queue is (Q minus R) + R: Q itself when R
is Q's tail in play order (the most recent plays; else the flag "queue_reordered"). C prefers
Mirror and Elixir Collector (calibration economy.OMIT_FROM_STARTING_HAND swaps them out of a
starting hand; probe P14): one left in the pre-hand is flagged "omit_in_hand".

ELIXIR (plan 4.2, 4.3). Ours: the view's (the HUD's), plus the cost of our in-flight plays when
the caller's elixir has already paid them (RebuildSettings.inflight_paid: only for an engine that
charges at send, probe P16; the engine's docstring and the live client charge when the play runs,
so a live HUD has not paid them: public_state's pending_cost, rs_decide._spendable, and
RebuildSettings.live() keeps inflight_paid False). The opponent's: the counted reading in
view["public"]["opp_elixir"] (S02; 0 and inexact when absent). Each side's
recast costs are added, so the count is right once they have run; the sum is clamped to
[0, 10000] (flag "elixir_clamped").

SPAWNS (plan 4.2). sanitize_for_rebuild clamps units into the arena, moves ground units off the
water, caps hp, pads decks and keeps each king at 1 hp or more. One spec per unit (or per
formation, probe P1: RebuildSettings.spec_per), card = public_state.rebuild_card (a Witch's
skeleton spawns as Skeletons), at the sanitized point, hp = the unit's hp fraction x the card's own
hit points at the rebuild battle's level (the E06 spawn table's "hp", unit_hitpoints' own row;
with forms on, a card reset in its evolved / hero form scales by that form's max_hp as the view
shows it), clamped to [1, that]. A building is snapped to where Battle.building_placement says a
tap there lands (asked on a blank board, cached), so its footprint is a real deploy's. Units no
card makes (Golemite, Lava Pup, death bombs: sanitize's "no spawn card") are lost.
A spec that lays more than its own unit (the spawn table's "all" lists every unit type one reset
spec of the card put down; probe P17): a member of such a card is spawned as a card whose spec
lays only that unit when there is one (a Goblin Gang goblin as Goblins, flag "spec_substituted",
relabelled to Goblin Gang), and a unit such a card's spec brings along (a Ram Rider's rider, a
Goblin Giant's Spear Goblins) is not spawned again while a primary unit of its card is in the
view (flag "carried"; not lost).

RECASTS (R2; plan 4.4). A spawn starts deployed (probe P2) and fires no deploy blow, so:
  deploy   one side's deploying members of one play: the units whose creating card (the view's
           card_id) is a troop or building card and that are its members (its own unit, or a unit
           formation_preview lists for it: a Goblin Gang's Spear Goblins; never a child: a Witch's
           skeletons are not recast as a Witch). A play is recast only when it is whole: as many
           members as formation_preview lays (else the catalogue count), their deploy timers
           spread no more than the formation's own stagger (calibration
           formation.DEPLOY_STAGGER: member k waits k x SummonDeployDelay) plus
           deploy_group_ticks, and at least recast_min_left of the card's deploy time (the
           longest member's, else DEFAULT_DEPLOY_TICKS) left on the youngest. Anything else
           (a member dead or unseen, two plays of one card) is spawned as it is (flag
           "recast_partial"): a recast lays the whole formation, so a part recast would double
           units. Its card is played at t, ours at our exact aim (seat.sent / own_sent), the
           opponent's at the formation's tap point (formation_preview's offset from the members'
           centroid, else the centroid; a building at its snapped centre).
  spell    (SPELL cards only; a troop's effect is dropped, flag "effect_dropped") our spell in
           flight at our known aim (own_sent); an enemy projectile (PROJECTILE_NAMES) at the aim
           infer_aim finds; a rolling one (ROLLING_NAMES) at its first sighting (SpellSightings;
           log_from "current": where it is now); any other at its observed centre. A spell whose
           aim cannot be inferred is dropped (flag "spell_unrecast").
A recast card must be in its side's cycle (a missed play's card goes to the queue's last open
slot; with none it is spawned instead, flag "recast_not_in_cycle"). root() asks check_deploy for
each recast at t; a refused deploy recast is spawned instead (the spec is re-dealt without it and
reset again), a refused spell recast is dropped. A recast lands late by the deploy (or flight)
time already gone; the 50% threshold keeps that within about half a second.

SPELL AIM (infer_aim; plan 4.4). A public spell row's aim is a placeholder, its centre
(public_state.py:510-519): never recast a projectile there. The ray from the caster's king tower
through the sighting is walked beyond the sighting (AIM_STEP_MT apart, plus the projection of each
target onto it); the point whose radius (the catalogue's, else DEFAULT_RADIUS_MT) holds the most of
the other side's value (sum of min(hp, DAMAGE_CAP) over its units and its crown towers, a tower
reached at its body's edge) wins, the nearest one on a tie; nothing of value anywhere: None.

THE ROOT (Rebuilder.root). The spec's own refusal (below) first. Then reset the one Battle
(cfg.undelayed(): every command is timed here), refuse a game already over, check the recasts
(and reset again without the refused ones), read the state once for the relabel map, step the
recasts in at t and our replays at their exec ticks, the card mapped to its slot through the hand
the state shows then, one command per team per step (_run), step on to B, save() once and read the
state at B (relabelled). Built carries the blob, the state, the label map, the fidelity, the
flags and a report. legal(card, x, y) loads the root blob when the battle has moved since.
trial(built, plays, ticks) loads it, steps plays dated (tick, team, card, x, y) in the same way
(our `later` in-flight plays too) and returns the relabelled after-state.

RELABEL (plan 4.5). Right after the reset, each entity is matched to its spec by exact (team,
card, x, y) (spawns are not snapped and come out in canonical, not list, order; a tie is broken by
hp), then, for what is left (a formation's members), by the nearest spec of its team and card
within FORMATION_MATCH_MT. uid -> the view's card_id (the creating card); relabel(state) writes it
back on every state read from the battle. A child made during a trial already carries its parent's
card. RebuildSettings.relabel False is the M6 ablation.

FIDELITY AND REFUSALS (plan 4.6). fidelity = the hp share of the view's units that were spawned
or recast (lost: dropped by sanitize, or a recast refused when it ran) x inexact_elixir (0.8) when
the opponent count is not exact, 0 when refused. Below min_fidelity (0.7) the search plays d0
(usable()). root() returns None, with last_error = Refusal(reason, detail), for: the spec's own
"units_dropped" (more than max_dropped_share of the visible units dropped), "towers_unknown" (a
side's tower_hp is not [king, left, right]: a reset without it would put every tower at full hp)
and "overtime_crowns" (overtime with unequal crowns: the game would be over, probe P7),
"reset_refused" (one bad spec fails the whole reset), "game_over_at_reset", "delayed_battle" (the
battle reports a command delay) and "error" (any other exception, so the caller can count them:
three in a battle turn search off).

PROBE-DRIVEN SETTINGS (RebuildSettings.from_report: a report, or the newest
reports/search_bench/engine_*.json of tools/rs_search_bench.py engine that has probes, S01;
without one the safe defaults, which are the plan's). Asked for, never implied: a Rebuilder
without settings uses RebuildSettings() (the defaults), so a report written later cannot change
an arm between gate runs; S08 pins the settings it uses (to_json, key) in its config.
  spec_per        "unit"    P1 FAIL (one Skeletons spec makes several): "formation"
  deploy_recasts  True      P2 FAIL (spawns carry deploy timers), or P13's played HogRider showing
                            no deploy timer: False
  inflight_paid   False     P16: the elixir is charged when the play is SENT: True (the
                            simulator only; RebuildSettings.live() keeps it False: the live
                            client charges when the play runs)
  omit_rule       True      P14 FAIL (Mirror / Elixir Collector stay in the first four): False
  forms           False     P5 PASS (reset forms reach spawns): True, each side's cards seen
                            evolved / hero get that form (a reset refused with forms is retried
                            without, flag "forms_refused")
P4 (levels), P6 (king activation), P11 (riders), P12 (shields) are things a reset cannot carry
(plan 4.7) and change nothing here; P8 is guarded (a battle reporting a delay is refused); P11 and
P17 (what a spec brings along) are read from the spawn table's "all" instead (SPAWNS).

Never imports scripts/gamestate (the memory reader), torch or royalesim at import time; never
taps, captures or touches the emulator. Imports the standard library and tools/rs_engine,
rs_public_view and rs_spawn_table; tools/public_state (numpy, scripts/vision) only when an R1 / R2
spec is built, so a torch-free search worker can import it.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:        # also when imported as tools.rs_rebuild
    sys.path.insert(0, str(ROOT / "tools"))

import rs_engine  # noqa: E402  (royalesim only when a Battle is built)
import rs_public_view as PV  # noqa: E402  (pure Python: views, cycles, permanence)
import rs_spawn_table  # noqa: E402  (pure helpers: primary_of, by_unit, FILLER_NAMES)

__all__ = ["R0", "R1", "R2", "LEVELS", "RebuildSettings", "RebuildError", "Refusal", "Recast",
           "RootSpec", "Built", "Tables", "Rebuilder", "SpellSightings", "deal", "build_spec",
           "r0_spec", "infer_aim", "fidelity", "usable", "with_present", "probe_overrides",
           "load_report"]

R0, R1, R2 = "R0", "R1", "R2"
LEVELS = (R0, R1, R2)
UNIT, FORMATION = "unit", "formation"
LOG_FIRST, LOG_CURRENT = "first", "current"

SUBTILE_PER_MILLITILE = 18
TICK_MS = 50
HAND_SIZE, DECK_SIZE = 4, 8
ELIXIR_MAX_MILLI = 10_000
REGULAR_TICKS = 3600
KIND_BUILDING = 1
#: legal(): the card is not in the hand at B (not a DEPLOY_REASONS code).
NOT_IN_HAND = -1
#: royalesim.CATALOGUE_FIELDS when the engine does not export it.
CATALOGUE_FIELDS_FALLBACK = ("name", "placement", "elixir", "count", "radius", "flying",
                             "hitpoints", "footprint_tiles", "card_kind")
#: Where tools/rs_search_bench.py writes its reports (S01).
BENCH_DIR = ROOT / "reports" / "search_bench"

# Defaults (plan 4.4, 4.6).
MIN_FIDELITY = 0.7
INEXACT_ELIXIR = 0.8
MAX_DROPPED_SHARE = 0.3
RECAST_MIN_LEFT = 0.5
DEPLOY_GROUP_TICKS = 3
#: A card whose deploy time formation_preview cannot give: 1 s, most troops' and buildings'.
DEFAULT_DEPLOY_TICKS = 20
#: The blank board building_placement is asked on: after the deploy lockout (90 ticks).
BLANK_TICK = 600
FORMATION_MATCH_MT = 3000.0
SNAP_CACHE = 4096

# Spells (catalogue names, tools/rs_decide.py's spelling).
PROJECTILE_NAMES = ("Fireball", "Rocket", "Arrows", "Snowball", "GoblinBarrel")
ROLLING_NAMES = ("Log", "BarbLog")
#: About each projectile's damage at level 11: infer_aim's value is min(hp, cap), so a unit counts
#: no more than the spell can take off it. A ranking, not a damage model.
DAMAGE_CAP = {"Fireball": 690, "Rocket": 1480, "Arrows": 370, "Snowball": 200,
              "GoblinBarrel": 1000}
DEFAULT_DAMAGE_CAP = 700
DEFAULT_RADIUS_MT = 2500.0
AIM_STEP_MT = 250.0
#: Crown tower centres, world millitiles, [king, left, right] as tower_hp (rs_decide.TOWER_CENTRE).
TOWER_CENTRE_MT = {(0, 0): (9000, 3000), (0, 1): (3500, 6500), (0, 2): (14500, 6500),
                   (1, 0): (9000, 29000), (1, 1): (3500, 25500), (1, 2): (14500, 25500)}
#: A spell reaches a tower at its body's edge: half the king's 4 tiles, a princess's 3.
TOWER_REACH_MT = {0: 2000.0, 1: 1500.0, 2: 1500.0}
#: SpellSightings: a row is the same spell when seen again within this many ticks and this far.
SIGHT_GAP_TICKS = 40
SIGHT_REACH_MT_PER_TICK = 600.0

# Refusals (Refusal.reason).
REFUSED_SPEC_DROPPED = "units_dropped"
REFUSED_TOWERS = "towers_unknown"
REFUSED_OVERTIME = "overtime_crowns"
REFUSED_RESET = "reset_refused"
REFUSED_GAME_OVER = "game_over_at_reset"
REFUSED_DELAYED = "delayed_battle"
REFUSED_ERROR = "error"
REFUSALS = (REFUSED_SPEC_DROPPED, REFUSED_TOWERS, REFUSED_OVERTIME, REFUSED_RESET,
            REFUSED_GAME_OVER, REFUSED_DELAYED, REFUSED_ERROR)


class RebuildError(ValueError):
    """A view or state the rebuilder cannot take (not a public view, a deal that cannot be
    made)."""


class Refusal(NamedTuple):
    """Why root() returned None: reason (REFUSALS) and detail (the engine's message, ...)."""
    reason: str
    detail: str = ""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _team(value: Any) -> int:
    team = int(value)
    if team not in (0, 1):
        raise ValueError(f"team must be 0 or 1, got {value!r}")
    return team


def _index(fields: Mapping[str, int] | Sequence[str] | None,
           fallback: Sequence[str]) -> dict[str, int]:
    if fields is None:
        fields = fallback
    if isinstance(fields, Mapping):
        return {str(k): int(v) for k, v in fields.items()}
    return {str(name): i for i, name in enumerate(fields)}


def _clamp_milli(value: int) -> int:
    return min(ELIXIR_MAX_MILLI, max(0, int(value)))


def _reason(res: Sequence[Any], i: int) -> int:
    """The reason code of step()'s i-th result row (card, reason, tick, x, y); 0 when absent."""
    try:
        return int(res[i][1])
    except (IndexError, TypeError, ValueError):
        return 0


def _ps() -> ModuleType:
    """tools/public_state, imported on first use (numpy and scripts/vision come with it)."""
    import public_state
    return public_state


def _own_hp(hp_table: Mapping[str, Any], card: int, level: int) -> int | None:
    """public_state.card_hitpoints: the "own" row's hit points of `card` at `level`, or None."""
    rows = (hp_table.get(str(int(card))) or {}).get(str(int(level))) or ()
    for r in rows:
        if len(r) >= 3 and r[0] == "own" and int(r[2]) > 0:
            return int(r[2])
    return None


def with_present(future: Mapping[str, Any], present: Mapping[str, Any]) -> dict:
    """A copy of a state at B whose players keep the PRESENT hand, hand costs, elixir, deck and next
    card (rs_eval.py:295-300): the root row is about what can be tapped now (plan 3.1 step 5)."""
    out = copy.deepcopy(dict(future))
    for t in (0, 1):
        now, then = present["players"][t], out["players"][t]
        for key in ("hand", "hand_costs", "elixir_milli", "deck"):
            if key in now:
                then[key] = copy.deepcopy(now[key])
        then["next_card"] = now.get("next_card", -1)
    return out


# ---------------------------------------------------------------------------
# settings, and the probes that set them
# ---------------------------------------------------------------------------

def _read_report(path: str | Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict):
        data.setdefault("_path", str(path))
        return data
    return None


def load_report(path: str | Path | None = None, out_dir: str | Path = BENCH_DIR) -> dict | None:
    """The S01 engine report at `path`, else the newest engine_*.json in `out_dir` that has
    probes (a later `engine --no-probes` run, timings only, does not hide the probes of an
    earlier one); None when there is none or it does not read as a JSON object."""
    if path is not None:
        return _read_report(path)
    for f in sorted(Path(out_dir).glob("engine_*.json"), reverse=True):
        data = _read_report(f)
        if data is not None and data.get("probes"):
            return data
    return None


def _probe_ok(probes: Mapping[str, Mapping[str, Any]], pid: str) -> bool | None:
    p = probes.get(pid)
    if not isinstance(p, Mapping):
        return None
    ok = p.get("ok")
    return None if ok is None else bool(ok)


def probe_overrides(report: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """RebuildSettings values an S01 engine report decides (module doc, PROBE-DRIVEN SETTINGS), and
    one note per decision. A probe that did not run decides nothing."""
    if not report:
        return {}, []
    probes = {str(p.get("id")): p for p in report.get("probes") or () if isinstance(p, Mapping)}
    out: dict[str, Any] = {}
    notes: list[str] = []
    p1 = _probe_ok(probes, "P1")
    if p1 is False:
        out["spec_per"] = FORMATION
        notes.append("P1 FAIL: one spec per formation")
    elif p1 is True:
        out["spec_per"] = UNIT
    if _probe_ok(probes, "P2") is False:
        out["deploy_recasts"] = False
        notes.append("P2 FAIL: spawns carry deploy timers, no deploy recasts")
    else:
        hog = ((probes.get("P13") or {}).get("answer") or {}).get("HogRider") or {}
        ticks = hog.get("deploy_ticks") if isinstance(hog, Mapping) else None
        if isinstance(ticks, list) and ticks and not all(int(t or 0) > 0 for t in ticks):
            out["deploy_recasts"] = False
            notes.append("P13: a played HogRider shows no deploy timer, no deploy recasts")
    p14 = _probe_ok(probes, "P14")
    if p14 is not None:
        out["omit_rule"] = p14
        if not p14:
            notes.append("P14 FAIL: Mirror / Elixir Collector stay in the first four")
    if _probe_ok(probes, "P5") is True:
        out["forms"] = True
        notes.append("P5 PASS: reset forms reach spawns")
    p16 = (probes.get("P16") or {}).get("answer") or {}
    if isinstance(p16, Mapping) and p16.get("elixir") in ("send", "run"):
        out["inflight_paid"] = p16.get("elixir") == "send"
        if out["inflight_paid"]:
            notes.append("P16: elixir is charged at send, in-flight costs are added back")
    if _probe_ok(probes, "P17") is False:
        notes.append("P17 FAIL: a spec lays its second summon too; the spawn table's 'all' "
                     "decides the substitution (module doc, SPAWNS): regenerate it under this "
                     "engine")
    return out, notes


@dataclass(frozen=True)
class RebuildSettings:
    """What the rebuilder does (module doc). Builtins only; to_json / from_json."""

    level: str = R1
    min_fidelity: float = MIN_FIDELITY
    inexact_elixir: float = INEXACT_ELIXIR
    max_dropped_share: float = MAX_DROPPED_SHARE
    recast_min_left: float = RECAST_MIN_LEFT
    deploy_group_ticks: int = DEPLOY_GROUP_TICKS
    spec_per: str = UNIT
    deploy_recasts: bool = True
    spell_recasts: bool = True
    log_from: str = LOG_FIRST
    inflight_paid: bool = False
    omit_rule: bool = True
    forms: bool = False
    snap_buildings: bool = True
    relabel: bool = True
    check_recasts: bool = True
    #: Where the probe-driven values came from (a report path, or "defaults"), and why.
    probes: str = "defaults"
    notes: tuple = ()

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level: one of {LEVELS}, got {self.level!r}")
        if self.spec_per not in (UNIT, FORMATION):
            raise ValueError(f"spec_per: {UNIT!r} or {FORMATION!r}, got {self.spec_per!r}")
        if self.log_from not in (LOG_FIRST, LOG_CURRENT):
            raise ValueError(f"log_from: {LOG_FIRST!r} or {LOG_CURRENT!r}, got {self.log_from!r}")
        for name in ("min_fidelity", "inexact_elixir", "max_dropped_share", "recast_min_left"):
            v = float(getattr(self, name))
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {v}")
        if int(self.deploy_group_ticks) < 0:
            raise ValueError("deploy_group_ticks must be >= 0")
        object.__setattr__(self, "notes", tuple(str(n) for n in self.notes))

    def replace(self, **changes: Any) -> "RebuildSettings":
        return dataclasses.replace(self, **changes)

    def to_json(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["notes"] = list(self.notes)
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "RebuildSettings":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"RebuildSettings: unknown field(s) {unknown}")
        kw = dict(data)
        if "notes" in kw:
            kw["notes"] = tuple(kw["notes"] or ())
        return cls(**kw)

    @classmethod
    def from_report(cls, report: Mapping[str, Any] | None = None, *, path: str | Path | None = None,
                    search: bool = True, **overrides: Any) -> "RebuildSettings":
        """The defaults, then what an S01 engine report decides (`report`, else the one at `path`,
        else, with search, the newest in BENCH_DIR), then `overrides`."""
        if report is None and (path is not None or search):
            report = load_report(path)
        probed, notes = probe_overrides(report)
        source = str((report or {}).get("_path") or ("report" if report else "defaults"))
        kw = {**probed, "probes": source, "notes": tuple(notes), **overrides}
        return cls(**kw)

    @classmethod
    def live(cls, base: "RebuildSettings | None" = None, **overrides: Any) -> "RebuildSettings":
        """For the live runner (S13): `base` (the settings the gate ran; default the defaults) with
        inflight_paid False. The live client charges our elixir and cycles our hand when a play
        RUNS, not at the tap (Battle.set_command_delay_ticks docstring: "measured on the live
        client"; rs_decide.prune_pending; public_state reports pending_cost apart and
        rs_decide._spendable subtracts it), so the HUD's elixir has not paid our plays in flight
        and each pays when the rebuild steps it in (plan 4.3)."""
        return (base or cls()).replace(**{"inflight_paid": False, **overrides})

    @property
    def key(self) -> str:
        """sha1 of what changes a rebuild (every field but `probes` and `notes`, the
        provenance): equal keys, equal rebuilds. S08 records it beside SearchConfig."""
        data = {k: v for k, v in self.to_json().items() if k not in ("probes", "notes")}
        text = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# tables: what the spec needs to know about the engine's cards
# ---------------------------------------------------------------------------

class Tables:
    """The engine's cards as the spec builder reads them: catalogue_json() rows (in
    royalesim.CATALOGUE_FIELDS order) and the E06 spawn table (tools/rs_spawn_table.py: card_names,
    unit_types, primary, hp). `level`: the rebuild battle's card level (spawn hp)."""

    def __init__(self, catalogue: Sequence[Sequence[Any]], spawn_table: Mapping[str, Any], *,
                 catalogue_fields: Sequence[str] | None = None, level: int = 11):
        self.catalogue = [list(r) for r in catalogue]
        cf = [str(f) for f in (catalogue_fields or CATALOGUE_FIELDS_FALLBACK)]

        def col(name: str) -> int:
            return cf.index(name) if name in cf else CATALOGUE_FIELDS_FALLBACK.index(name)

        at_e, at_k, at_r, at_n = col("elixir"), col("card_kind"), col("radius"), col("count")
        self.names = [str(r[0]) for r in self.catalogue]
        self.ids = {n: i for i, n in enumerate(self.names)}

        def get(row: Sequence[Any], i: int, default: Any) -> Any:
            return row[i] if i < len(row) and row[i] is not None else default

        self.cost = {i: 1000 * int(get(r, at_e, 0) or 0) for i, r in enumerate(self.catalogue)}
        self.kind = {i: str(get(r, at_k, "")).upper() for i, r in enumerate(self.catalogue)}
        self.radius = {i: float(get(r, at_r, 0) or 0) for i, r in enumerate(self.catalogue)}
        self.count = {i: max(1, int(get(r, at_n, 1) or 1)) for i, r in enumerate(self.catalogue)}
        self.spawn_table = dict(spawn_table)
        self.primary = (rs_spawn_table.primary_of(spawn_table) if spawn_table.get("primary")
                        else {})
        self.by_unit = rs_spawn_table.by_unit(self.primary)
        self.unit_names = [str(u) for u in spawn_table.get("unit_types") or ()]
        self.hp_table = spawn_table.get("hp") or {}
        self.level = int(level)
        self.mirror = frozenset(PV.mirror_ids(self.catalogue))
        self.omit = frozenset(PV.card_ids(PV.OMIT_FROM_STARTING_HAND_NAMES, self.names))
        self.projectile = frozenset(PV.card_ids(PROJECTILE_NAMES, self.names))
        self.rolling = frozenset(PV.card_ids(ROLLING_NAMES, self.names))
        self.spells = frozenset(i for i, k in self.kind.items() if k == "SPELL")
        self.spawnable = [i for i, k in sorted(self.kind.items()) if k in ("TROOP", "BUILDING")]
        self.filler = PV.card_ids(rs_spawn_table.FILLER_NAMES, self.names)
        # What one reset spec of a card puts down (the spawn table's "all", measured by
        # rs_spawn_table; absent in an older table: nothing is known, nothing changes).
        self.all_units = {int(c): frozenset(int(u) for u in us)
                          for c, us in (spawn_table.get("all") or {}).items()}
        #: Cards whose spec lays a unit type besides its own (a second summon, a rider).
        self.bundled = frozenset(c for c, us in self.all_units.items()
                                 if c in self.primary and us - {self.primary[c]})
        #: unit type -> the lowest card whose spec lays that unit and nothing else.
        self.alone: dict[int, int] = {}
        for c, u in sorted(self.primary.items()):
            if c in self.all_units and self.all_units[c] <= {u}:
                self.alone.setdefault(u, c)

    def own_hp(self, card: int, level: int | None = None) -> int | None:
        return _own_hp(self.hp_table, card, self.level if level is None else level)

    def unit_name(self, unit: Any) -> str | None:
        try:
            u = int(unit)
        except (TypeError, ValueError):
            return None
        return self.unit_names[u] if 0 <= u < len(self.unit_names) else None

    def unit_id(self, name: Any) -> int:
        """The unit type id of a unit name (unit_types order), -1 when it is not one."""
        try:
            return self.unit_names.index(str(name))
        except ValueError:
            return -1

    def base_card(self, card: Any, unit: Any) -> int | None:
        """public_state.rebuild_card: the card RSAgent.rebuilt spawns for (card_id, unit_type)."""
        return _ps().rebuild_card(int(card), int(unit), self.primary, self.by_unit,
                                  self.unit_names)

    def spawn_card(self, card: Any, unit: Any) -> int | None:
        """The card a spawn of entity (card_id, unit_type) uses: base_card, except that the own
        unit of a bundled card (its spec lays more, module doc SPAWNS) spawns as a card whose
        spec lays that unit alone, when there is one."""
        c = self.base_card(card, unit)
        if c is not None and c in self.bundled and self.primary.get(c) == int(unit):
            return self.alone.get(int(unit), c)
        return c

    def carried(self, card: Any, unit: Any) -> bool:
        """Whether a unit of type `unit` made by `card` comes with the spec of `card`'s own unit
        (a rider, a second summon: the spawn table's "all"), so it is not spawned again, while
        `card`'s own unit spawns as `card` itself (no alone card stands in for it)."""
        c, u = int(card), int(unit)
        own = self.primary.get(c)
        return (c in self.bundled and own is not None and u != own
                and u in self.all_units.get(c, ()) and own not in self.alone)

    def cap(self, card: int) -> int:
        return int(DAMAGE_CAP.get(self.names[card] if 0 <= card < len(self.names) else "",
                                  DEFAULT_DAMAGE_CAP))

    def radius_mt(self, card: int) -> float:
        return float(self.radius.get(card) or 0.0) or DEFAULT_RADIUS_MT


# ---------------------------------------------------------------------------
# the deal
# ---------------------------------------------------------------------------

def deal(hand: Sequence[int], queue: Sequence[int], replay: Sequence[int], *,
         omit: Iterable[int] = ()) -> tuple[list[int], list[int], list[str]]:
    """(pre-hand, pre-queue, notes): the shuffle-0 deal after whose plays of `replay` (in order,
    each from its slot) the hand is `hand` slot for slot and the queue `queue` when `replay`'s
    queued cards are its tail (module doc, THE DEAL). RebuildError when a replay card is in neither,
    or more cards must be re-dealt than hand slots are free."""
    hand, queue = [int(c) for c in hand], [int(c) for c in queue]
    if len(hand) != HAND_SIZE or len(queue) != DECK_SIZE - HAND_SIZE:
        raise RebuildError(f"a deal needs 4 + 4 cards, got {hand} + {queue}")
    if len(set(hand + queue)) != DECK_SIZE:
        raise RebuildError(f"a deal needs 8 distinct cards, got {hand} + {queue}")
    replay = [int(c) for c in replay]
    if len(set(replay)) != len(replay):
        raise RebuildError(f"a card twice among the plays to replay: {replay}")
    missing = [c for c in replay if c not in hand and c not in queue]
    if missing:
        raise RebuildError(f"cards {missing} to replay are in neither the hand nor the queue")
    rq = [c for c in replay if c in queue]
    keep = {c for c in replay if c in hand}
    omit = {int(c) for c in omit}
    order = sorted(range(HAND_SIZE), key=lambda i: (hand[i] not in omit, -i))
    slots = [i for i in order if hand[i] not in keep][:len(rq)]
    if len(slots) < len(rq):
        raise RebuildError(f"{len(rq)} cards to re-deal, {len(slots)} hand slots free")
    pre_hand = list(hand)
    pushed = []
    for card, i in zip(rq, slots):
        pushed.append(hand[i])
        pre_hand[i] = card
    pre_queue = pushed + [q for q in queue if q not in rq]
    notes = []
    if rq and queue[len(queue) - len(rq):] != rq:
        notes.append("queue_reordered")
    if omit & set(pre_hand):
        notes.append("omit_in_hand")
    first_kept = next((k for k, c in enumerate(replay) if c in keep), None)
    if first_kept is not None and any(c in rq for c in replay[first_kept:]):
        notes.append("deal_interleaved")
    return pre_hand, pre_queue, notes


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------

class Recast(NamedTuple):
    """One card played at t by the rebuild (module doc, RECASTS).

    kind    "deploy" (a deploying group), "spell" (a projectile), "log" (rolling), "area"
    cost    its elixir, milli (added to its side's elixir)
    order   the tick it was (about) played: recasts of a side run oldest first
    spawns, labels, uids   a deploy group's fallback: its members as spawns (spawned instead when
                           check_deploy refuses the recast); empty for a spell
    hp      the view hp of a deploy group (fidelity)
    """
    team: int
    card: int
    x: int
    y: int
    kind: str
    cost: int
    order: int
    spawns: tuple = ()
    labels: tuple = ()
    uids: tuple = ()
    hp: int = 0


def _sorted_spawns(rows: Iterable[tuple]) -> tuple[tuple, tuple, tuple]:
    """(spawns, labels, uids) from (spawn, label, uid) triples in canonical order (team, y, x,
    card, hp, uid): list order means nothing to reset, so the spec does not depend on it."""
    items = sorted(rows, key=lambda r: (r[0][0], r[0][3], r[0][2], r[0][1], r[0][4],
                                        -1 if r[2] is None else r[2]))
    return (tuple(tuple(int(v) for v in r[0]) for r in items),
            tuple(None if r[1] is None else int(r[1]) for r in items),
            tuple(None if r[2] is None else int(r[2]) for r in items))


@dataclass(frozen=True)
class RootSpec:
    """Everything a reset needs (module doc, THE SPEC). R0's decks / elixir are rebuilt's own and
    its hands / queues / order are empty; R1 / R2's decks are deal()'s of hands, queues and
    order."""

    level: str
    team: int
    tick: int
    land_tick: int
    seed: int = 0
    decks: tuple = ((), ())
    elixir: tuple = (0, 0)
    tower_hp: tuple | None = None
    spawns: tuple = ()
    labels: tuple = ()
    uids: tuple = ()
    recasts: tuple = ()
    replays: tuple = ()
    later: tuple = ()
    hands: tuple = ((), ())
    queues: tuple = ((), ())
    order: tuple = ((), ())
    base_elixir: tuple = (0, 0)
    omit: tuple = ()
    forms_of: tuple = ((), ())
    unit_hp: int = 0
    lost_hp: int = 0
    units: int = 0
    dropped: int = 0
    opp_exact: bool = True
    flags: tuple = ()
    refusal: str | None = None

    # ------------------------------------------------------------ identity
    def to_json(self) -> dict[str, Any]:
        def plain(v: Any) -> Any:
            if isinstance(v, Recast):
                return {k: plain(x) for k, x in v._asdict().items()}
            if isinstance(v, (tuple, list)):
                return [plain(x) for x in v]
            return v
        return {f.name: plain(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "RootSpec":
        def tup(v: Any) -> Any:
            if isinstance(v, list):
                return tuple(tup(x) for x in v)
            return v
        kw = {}
        for f in dataclasses.fields(cls):
            if f.name not in data:
                continue
            v = data[f.name]
            if f.name == "recasts":
                v = tuple(Recast(**{k: tup(x) for k, x in r.items()}) for r in v or ())
            else:
                v = tup(v)
            kw[f.name] = v
        return cls(**kw)

    @property
    def key(self) -> str:
        """sha1 of the canonical JSON: equal specs, equal keys."""
        text = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @property
    def opp(self) -> int:
        return 1 - self.team

    # ------------------------------------------------------------ the reset
    def forms(self) -> list[list[int]] | None:
        """reset's forms, parallel to the decks, or None when every card is its base card."""
        if not any(self.forms_of):
            return None
        out = []
        for t in (0, 1):
            of = {int(c): int(f) for c, f in self.forms_of[t]}
            out.append([of.get(int(c), 0) for c in self.decks[t]])
        return out if any(f for side in out for f in side) else None

    def reset_kwargs(self, *, forms: bool = True) -> dict[str, Any]:
        """Battle.reset keywords (lists, as rs_eval passes them); forms only when there are."""
        kw: dict[str, Any] = {
            "seed": int(self.seed), "decks": [list(d) for d in self.decks], "shuffle": 0,
            "start_tick": int(self.tick), "elixir_milli": list(self.elixir),
            "tower_hp": None if self.tower_hp is None else [list(t) for t in self.tower_hp],
            "spawns": [tuple(s) for s in self.spawns]}
        f = self.forms() if forms else None
        if f is not None:
            kw["forms"] = f
        return kw

    def redealt(self, recasts: Sequence[Recast], *, flags: Iterable[str] = (),
                spawns: Iterable[tuple] = ()) -> "RootSpec":
        """This spec with these recasts instead (and these (spawn, label, uid) triples added): each
        side's deck re-dealt and elixir recounted. R1 / R2 only."""
        if self.level == R0:
            raise RebuildError("an R0 spec has no deal to redo")
        recasts = tuple(recasts)
        kept = {(r.team, r.card) for r in recasts}
        dropped = {(r.team, r.card) for r in self.recasts} - kept
        order = tuple(tuple(c for c in self.order[t] if (t, c) not in dropped) for t in (0, 1))
        decks, notes = _deal_both(self.hands, self.queues, order, self.omit)
        elixir, clamped = _elixir(self.base_elixir, recasts)
        rows = list(zip(self.spawns, self.labels, self.uids)) + list(spawns)
        sp, lab, uid = _sorted_spawns(rows)
        new_flags = _flags(set(self.flags) - {"queue_reordered", "omit_in_hand", "deal_interleaved",
                                              "elixir_clamped"}, notes, flags,
                           ["elixir_clamped"] if clamped else [])
        return dataclasses.replace(self, decks=decks, elixir=elixir, spawns=sp, labels=lab,
                                   uids=uid, recasts=recasts, order=order, flags=new_flags)

    def without_recasts(self, indices: Iterable[int]) -> "RootSpec":
        """The spec with recasts[i] (i in indices) undone: a deploy group spawned instead (flag
        "recast_spawned"), a spell dropped ("spell_unrecast")."""
        drop = {int(i) for i in indices}
        keep = [r for i, r in enumerate(self.recasts) if i not in drop]
        gone = [r for i, r in enumerate(self.recasts) if i in drop]
        rows = [(s, lab, uid) for r in gone for s, lab, uid in zip(r.spawns, r.labels, r.uids)]
        flags = (["recast_spawned"] if any(r.kind == "deploy" for r in gone) else []) + \
                (["spell_unrecast"] if any(r.kind != "deploy" for r in gone) else [])
        return self.redealt(keep, flags=flags, spawns=rows)

    def bumped(self, side: int, milli: int) -> "RootSpec":
        """The spec with `milli` more elixir for `side` (a world's elixir bump, plan 5)."""
        side = _team(side)
        base = list(self.base_elixir)
        base[side] += int(milli)
        elixir = list(self.elixir)
        elixir[side] = _clamp_milli(elixir[side] + int(milli))
        return dataclasses.replace(self, base_elixir=tuple(base), elixir=tuple(elixir))


def _flags(*groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(f) for g in groups for f in g}))


def _deal_both(hands: Sequence[Sequence[int]], queues: Sequence[Sequence[int]],
               order: Sequence[Sequence[int]], omit: Iterable[int]) -> tuple[tuple, list[str]]:
    decks, notes = [], []
    for t in (0, 1):
        h, q, n = deal(hands[t], queues[t], order[t], omit=omit)
        decks.append(tuple(h + q))
        notes += n
    return tuple(decks), notes


def _elixir(base: Sequence[int], recasts: Iterable[Recast]) -> tuple[tuple[int, int], bool]:
    total = [int(base[0]), int(base[1])]
    for r in recasts:
        total[r.team] += int(r.cost)
    clamped = any(v > ELIXIR_MAX_MILLI or v < 0 for v in total)
    return (_clamp_milli(total[0]), _clamp_milli(total[1])), clamped


# ---------------------------------------------------------------------------
# R0: RSAgent.rebuilt exactly
# ---------------------------------------------------------------------------

def _r0_card(card: int, unit: int, prim: Mapping[int, int], by_unit: Mapping[int, int],
             names: Sequence[str]) -> int | None:
    """RSAgent.rebuilt's own card rule (rs_eval.py:363-376), Python indexing included: unit_type
    -1 reads the last unit name, as rebuilt's names_[unit] does. A unit_type past the list (rebuilt
    would raise) counts as no form."""
    if prim.get(card) == unit:
        return card
    base_name = names[prim[card]].lower() if card in prim else ""
    try:
        own = str(names[unit]).lower()
    except (IndexError, TypeError):
        own = None
    if base_name and own is not None and own.startswith(base_name):
        return card
    return by_unit.get(unit)


def r0_spec(state: Mapping[str, Any], team: int, *, tables: Tables,
            fields: Mapping[str, int] | Sequence[str] | None = None,
            sent: Iterable[Sequence[int]] = (), lead: int = 0, public: bool = False) -> RootSpec:
    """R0 (module doc, LEVELS): RSAgent.rebuilt(state, team, royalesim, first=...) and the stepping
    of RSAgent.ahead with lead_rebuild (rs_eval.py:269-300), on the state the seat sees. sent: the
    seat's (exec tick, card, x, y) plays (rs_match seat.sent); those with exec tick > now are first
    in our deck and replayed below now + lead. public: the seat's visibility is public (an enemy
    rs_rows.seen_by drops is not spawned)."""
    team = _team(team)
    F = _index(fields, rs_engine.ENTITY_FIELDS_EXPECTED)
    now = int(state["tick"])
    pend = [tuple(int(v) for v in p[:4]) for p in sent or () if int(p[0]) > now]
    first = [c for (_, c, _, _) in pend]
    me = state["players"][team]
    deck = list(me["deck"])
    hand = list(me["hand"])
    nxt = me.get("next_card", -1)
    hand = [c for c in hand if c >= 0 and c not in first]
    lead_cards = [c for c in first if c in deck]
    rest = [c for c in deck if c not in hand and c != nxt and c not in lead_cards]
    mine = lead_cards + hand + ([nxt] if nxt >= 0 and nxt not in lead_cards else []) + rest
    seen_c, ordered = set(), []
    for c in mine + deck:
        if c not in seen_c:
            seen_c.add(c)
            ordered.append(c)
    mine = ordered[:8]
    opp = state["players"][1 - team]["deck"]
    spawns, uids = [], []
    unit_hp = lost_hp = units = dropped = 0
    for e in state["entities"]:
        if e[F["tower_slot"]] >= 0 or e[F["hp"]] <= 0:
            continue
        if public and not PV.seen_by(e, team, F):
            continue
        units += 1
        unit_hp += int(e[F["hp"]])
        card = _r0_card(e[F["card_id"]], e[F["unit_type"]], tables.primary, tables.by_unit,
                        tables.unit_names)
        if card is None:
            dropped += 1
            lost_hp += int(e[F["hp"]])
            continue
        spawns.append((e[F["team"]], card, e[F["x"]], e[F["y"]], e[F["hp"]]))
        uids.append(int(e[F["uid"]]))
    decks = [None, None]
    decks[team], decks[1 - team] = mine[:8], list(opp)[:8]
    elixir = [0, 0]
    elixir[team] = me["elixir_milli"]
    towers = tuple(tuple(state["players"][t]["tower_hp"]) for t in (0, 1))
    land = now + int(lead or 0)
    replays = tuple(p for p in sorted(pend) if p[0] < land) if lead else ()
    return RootSpec(level=R0, team=team, tick=now, land_tick=land if lead else now, seed=0,
                    decks=(tuple(decks[0]), tuple(decks[1])), elixir=tuple(elixir),
                    tower_hp=towers, spawns=tuple(spawns), labels=(None,) * len(spawns),
                    uids=tuple(uids), replays=replays, base_elixir=tuple(elixir),
                    unit_hp=unit_hp, lost_hp=lost_hp, units=units, dropped=dropped,
                    opp_exact=False, flags=("r0",))


# ---------------------------------------------------------------------------
# spell aim
# ---------------------------------------------------------------------------

def _targets(view: Mapping[str, Any], side: int, F: Mapping[str, int], sub: int
             ) -> list[tuple[float, float, int, float]]:
    """`side`'s value on the board: (x mt, y mt, hp, reach mt) of its living units (crown-tower
    entities aside) and of its living crown towers (from tower_hp, live has no tower entities)."""
    out = []
    for e in view.get("entities") or ():
        if int(e[F["team"]]) != side or int(e[F["tower_slot"]]) >= 0 or (e[F["hp"]] or 0) <= 0:
            continue
        out.append((e[F["x"]] / sub, e[F["y"]] / sub, int(e[F["hp"]]), 0.0))
    hp = list(view["players"][side].get("tower_hp") or ())
    for k, h in enumerate(hp[:3]):
        if h and h > 0:
            x, y = TOWER_CENTRE_MT[(side, k)]
            out.append((float(x), float(y), int(h), TOWER_REACH_MT[k]))
    return out


def infer_aim(caster: int, x: float, y: float, targets: Sequence[tuple[float, float, int, float]],
              *, radius_mt: float = DEFAULT_RADIUS_MT, cap: int = DEFAULT_DAMAGE_CAP,
              step_mt: float = AIM_STEP_MT, launch: tuple[float, float] | None = None
              ) -> tuple[float, float, int] | None:
    """(x, y, value) in millitiles: where a projectile `caster` cast, now sighted at (x, y), is
    aimed (module doc, SPELL AIM); None when no point beyond the sighting reaches any target.
    targets: (x, y, hp, reach) of the other side (_targets)."""
    kx, ky = launch if launch is not None else TOWER_CENTRE_MT[(_team(caster), 0)]
    dx, dy = float(x) - kx, float(y) - ky
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return None
    ux, uy = dx / dist, dy / dist
    # the ray leaves the board at s_max
    lim = []
    for d, o, hi in ((ux, kx, 18_000.0), (uy, ky, 32_000.0)):
        if d > 1e-9:
            lim.append((hi - o) / d)
        elif d < -1e-9:
            lim.append((0.0 - o) / d)
    s_max = min(lim) if lim else dist
    points = []
    s = dist
    while s <= s_max + 1e-6:
        points.append(s)
        s += float(step_mt)
    for tx, ty, _hp, _reach in targets:
        s = (tx - kx) * ux + (ty - ky) * uy
        if dist <= s <= s_max:
            points.append(s)
    best = None
    for s in sorted(set(round(p, 3) for p in points)):
        px, py = kx + ux * s, ky + uy * s
        value = sum(min(int(hp), int(cap)) for tx, ty, hp, reach in targets
                    if math.hypot(tx - px, ty - py) <= radius_mt + reach)
        if value > 0 and (best is None or value > best[2]):
            best = (px, py, value)
    return best


class SpellSightings:
    """The first sighting of each spell row (team, card) of a battle's views, for a rolling spell's
    re-roll (module doc, RECASTS). update(view) once per decision, ticks in order; a tick that goes
    back starts a new battle; a row seen again within SIGHT_GAP_TICKS and SIGHT_REACH_MT_PER_TICK
    per tick of its last sighting is the same spell."""

    def __init__(self, spell_fields: Mapping[str, int] | Sequence[str] | None = None, *,
                 sub: int = SUBTILE_PER_MILLITILE, gap_ticks: int = SIGHT_GAP_TICKS,
                 reach_mt_per_tick: float = SIGHT_REACH_MT_PER_TICK):
        self.SF = _index(spell_fields, PV.SPELL_FIELDS_FALLBACK)
        self.sub = int(sub)
        self.gap = int(gap_ticks)
        self.reach = float(reach_mt_per_tick) * self.sub
        self.reset()

    def reset(self) -> None:
        self._memo: dict[tuple[int, int], dict[str, int]] = {}
        self.tick: int | None = None

    def update(self, view: Mapping[str, Any]) -> None:
        SF = self.SF
        tick = int(view["tick"])
        if self.tick is not None and tick < self.tick:
            self.reset()
        if self.tick == tick:
            return
        self.tick = tick
        now: dict[tuple[int, int], dict[str, int]] = {}
        for sp in view.get("spells") or ():
            key = (int(sp[SF["team"]]), int(sp[SF["card_id"]]))
            x, y = int(sp[SF["x"]]), int(sp[SF["y"]])
            m = self._memo.get(key)
            gap = tick - m["last_tick"] if m else None
            if (m is None or gap > self.gap
                    or math.hypot(x - m["x"], y - m["y"]) > self.reach * max(1, gap)):
                m = {"first_tick": tick, "first_x": x, "first_y": y}
            m = {**m, "last_tick": tick, "x": x, "y": y}
            now[key] = m
        self._memo = now

    def first(self, team: int, card: int) -> tuple[int, int, int] | None:
        """(tick, x, y) subtiles of the first sighting of `team`'s `card` spell, or None."""
        m = self._memo.get((int(team), int(card)))
        return None if m is None else (m["first_tick"], m["first_x"], m["first_y"])


# ---------------------------------------------------------------------------
# building the R1 / R2 spec
# ---------------------------------------------------------------------------

def _own_aim(own_sent: Iterable[Sequence[int]], card: int, tick: int, window: int
             ) -> tuple[int, int, int] | None:
    """(x, y, exec tick) of our latest play of `card` that ran before `tick`, within `window`."""
    best = None
    for s in own_sent or ():
        ex, c, x, y = (int(v) for v in s[:4])
        if c == card and tick - window <= ex < tick and (best is None or ex > best[2]):
            best = (x, y, ex)
    return best


def _side_cycle(view: Mapping[str, Any], side: int, recast_cards: Sequence[int],
                pool: Sequence[int], omit: Iterable[int], last_resort: Sequence[int]
                ) -> tuple[list[int], list[int], set[int], list[str]]:
    """(hand, queue, unplaceable recast cards, notes) of `side` at t: its CardCycle from the view,
    each recast card it lacks put in the queue's last open slot (the most recent play), the open
    slots then filled from `pool` (CardCycle.fill: omit cards into queue slots first), then from
    `last_resort`."""
    cyc = PV.CardCycle.from_view(view, side)
    hand, queue = list(cyc.hand), list(cyc.queue)
    notes = [] if cyc.exact else ["cycle_inexact"]
    known = {c for c in hand + queue if c is not None}
    bad: set[int] = set()
    for card in reversed(list(recast_cards)):          # the newest play takes the last hole
        if card in known:
            continue
        holes = [i for i, c in enumerate(queue) if c is None]
        if not holes:
            bad.add(card)
            continue
        queue[holes[-1]] = card
        known.add(card)
    cyc = dataclasses.replace(cyc, hand=tuple(hand), queue=tuple(queue),
                              unplaced=tuple(c for c in cyc.unplaced if c not in known))
    cyc = cyc.fill(list(pool), omit=omit)
    hand, queue = list(cyc.hand), list(cyc.queue)
    have = {c for c in hand + queue if c is not None}
    spare = [c for c in last_resort if c not in have]
    for slots in (hand, queue):
        for i, c in enumerate(slots):
            if c is None:
                if not spare:
                    raise RebuildError(f"no card left to fill team {side}'s cycle")
                slots[i] = spare.pop(0)
    return [int(c) for c in hand], [int(c) for c in queue], bad, notes


def build_spec(view: Mapping[str, Any], tables: Tables, settings: RebuildSettings | None = None, *,
               land_tick: int | None = None, seed: int = 0, level: str | None = None,
               fields: Mapping[str, int] | Sequence[str] | None = None,
               spell_fields: Mapping[str, int] | Sequence[str] | None = None,
               sub: int = SUBTILE_PER_MILLITILE, opp_fill: Iterable[int] = (),
               snap: Callable[[int, int, int, int], tuple[int, int] | None] | None = None,
               deploy_total: Callable[[int], int] | None = None,
               first_sighting: Callable[[int, int], tuple[int, int, int] | None] | None = None,
               members: Callable[[int], Sequence[tuple[str, int]] | None] | None = None,
               tap_point: Callable[[int, int, int, int], tuple[int, int] | None] | None = None
               ) -> RootSpec:
    """The R1 / R2 RootSpec of a public view (module doc). view: rs_public_view.public_view, with
    Permanence's kept rows if any; land_tick: B (default the view's tick); opp_fill: cards for the
    opponent's open slots first (a deck prior's draw); snap(team, card, x, y) -> (x, y) or None: a
    building's landing point (Rebuilder.snap); deploy_total(card) -> ticks (Rebuilder.deploy_total);
    first_sighting(team, card) -> (tick, x, y) (SpellSightings.first); members(card) -> the
    (unit name, deploy ticks) a play of the card lays, or None (Rebuilder.members; without it a
    play is its catalogue count of its own unit, all deploying together); tap_point(team, card, x,
    y) -> the tap whose formation is centred on (x, y), or None (Rebuilder.tap_point)."""
    s = settings or RebuildSettings()
    level = level or s.level
    if level == R0:
        raise RebuildError("R0 rebuilds from the seat's state: r0_spec / Rebuilder.spec_r0")
    pub = view.get("public")
    if not isinstance(pub, Mapping) or "team" not in pub:
        raise RebuildError("not a public view (no 'public' block): rs_public_view.public_view")
    team = _team(pub["team"])
    opp = 1 - team
    F = _index(fields, rs_engine.ENTITY_FIELDS_EXPECTED)
    SF = _index(spell_fields, PV.SPELL_FIELDS_FALLBACK)
    sub = int(sub)
    deploy_total = deploy_total or (lambda card: DEFAULT_DEPLOY_TICKS)
    ps = _ps()
    san = ps.sanitize_for_rebuild(view, tables.spawn_table, fields=F, sub=sub)
    tick = int(san["tick"])
    land = tick if land_tick is None else max(tick, int(land_tick))
    flags: list[str] = []

    # --- the view's units, and what sanitize kept of them
    view_units = [e for e in view.get("entities") or ()
                  if int(e[F["tower_slot"]]) < 0 and (e[F["hp"]] or 0) > 0]
    units, unit_hp = len(view_units), sum(int(e[F["hp"]]) for e in view_units)
    kept_rows = [e for e in san["entities"] if int(e[F["tower_slot"]]) < 0]
    left = Counter(int(e[F["uid"]]) for e in kept_rows)
    lost_hp = dropped = 0
    for e in view_units:
        uid = int(e[F["uid"]])
        if left[uid] > 0:
            left[uid] -= 1
        else:
            dropped += 1
            lost_hp += int(e[F["hp"]])

    # --- the spawn card of each kept unit
    first_row: dict[int, Sequence[Any]] = {}
    for e in view_units:
        first_row.setdefault(int(e[F["uid"]]), e)
    # a unit a bundled card's spec brings along is not spawned again while that card's own unit
    # is in the view (module doc, SPAWNS)
    carriers = {(int(e[F["team"]]), int(e[F["card_id"]])) for e in kept_rows
                if tables.primary.get(int(e[F["card_id"]])) == int(e[F["unit_type"]])}
    picked: list[tuple[Sequence[Any], int]] = []
    for e in kept_rows:
        label, unit_type = int(e[F["card_id"]]), int(e[F["unit_type"]])
        if (int(e[F["team"]]), label) in carriers and tables.carried(label, unit_type):
            flags.append("carried")
            continue
        card = tables.spawn_card(label, unit_type)
        if card is None:                    # sanitize keeps only units with a spawn card
            continue
        if card != tables.base_card(label, unit_type):
            flags.append("spec_substituted")
        picked.append((e, card))

    # --- forms (probe P5): a card seen evolved / hero on a side is reset in that form, and its
    # spawns are scaled by that form's max_hp (the view's), not the base card's
    forms_of: list[tuple] = [(), ()]
    form_max: dict[tuple[int, int], int] = {}
    if s.forms:
        for side in (0, 1):
            of: dict[int, int] = {}
            for e, card in picked:
                st = int(e[F["status_flags"]] or 0)
                if int(e[F["team"]]) == side and int(e[F["card_id"]]) == card and st > 0:
                    if st & PV.STATUS_HERO:
                        of[card] = 2
                    elif st & PV.STATUS_EVOLVED:
                        of.setdefault(card, 1)
            forms_of[side] = tuple(sorted(of.items()))
        for e, card in picked:
            side = int(e[F["team"]])
            form = dict(forms_of[side]).get(card, 0)
            bit = PV.STATUS_HERO if form == 2 else PV.STATUS_EVOLVED if form == 1 else 0
            if bit and int(e[F["status_flags"]] or 0) & bit:
                src = first_row.get(int(e[F["uid"]]), e)
                mx = int(src[F["max_hp"]] or 0)
                if mx > 0:
                    form_max[(side, card)] = max(form_max.get((side, card), 0), mx)

    # --- one candidate spawn per kept unit: (spawn, label, uid, row, spawn card). The hp fraction
    # is the view row's (sanitize has capped the hp already); the point is sanitize's.
    cands = []
    for e, card in picked:
        side = int(e[F["team"]])
        own = form_max.get((side, card)) or tables.own_hp(card)
        src = first_row.get(int(e[F["uid"]]), e)
        hp, mx = int(src[F["hp"]]), int(src[F["max_hp"]] or 0)
        if own is None:
            hp_spawn = max(1, int(e[F["hp"]]))
        else:
            frac = min(1.0, max(0.0, hp / mx)) if mx > 0 else 1.0
            hp_spawn = min(own, max(1, int(round(frac * own))))
        x, y = int(e[F["x"]]), int(e[F["y"]])
        if snap is not None and int(e[F["kind"]]) == KIND_BUILDING \
                and tables.kind.get(card) == "BUILDING":
            got = snap(side, card, x, y)
            if got is not None:
                x, y = int(got[0]), int(got[1])
        cands.append(((side, card, x, y, hp_spawn), int(e[F["card_id"]]),
                      int(e[F["uid"]]), e, card))

    # --- R2 deploy recasts: one whole play per side and card (module doc, RECASTS)
    recasts: list[Recast] = []
    in_recast: set[int] = set()            # cands indices
    own_sent = pub.get("own_sent") or ()

    def left_of(i: int) -> int:
        return int(cands[i][3][F["deploy_ticks"]] or 0)

    if level == R2 and s.deploy_recasts:
        groups = defaultdict(list)
        for i, (_sp, label, _uid, e, card) in enumerate(cands):
            if left_of(i) <= 0 or tables.kind.get(label) not in ("TROOP", "BUILDING"):
                continue
            if label != card:               # a second summon of `label`'s play, or a child
                names = {str(m[0]) for m in (members(label) or ())} if members else set()
                if tables.unit_name(e[F["unit_type"]]) not in names:
                    continue                # a Witch's skeleton is never recast as a Witch
            groups[(int(e[F["team"]]), label)].append(i)
        for (side, card), idx in sorted(groups.items()):
            idx.sort(key=lambda i: (-left_of(i), cands[i][2]))
            total = max(1, int(deploy_total(card)))
            youngest = left_of(idx[0])
            if youngest < s.recast_min_left * total:
                continue                    # most of its deploy is gone: spawned as it is
            form = list(members(card) or ()) if members else []
            # the members the view must show (a rider the spec brings along was not kept)
            laid = [m for m in form if not tables.carried(card, tables.unit_id(m[0]))]
            want = len(laid) or int(tables.count.get(card, 1))
            ticks = [int(m[1]) for m in form]
            stagger = max(ticks) - min(ticks) if ticks else 0
            spread = youngest - left_of(idx[-1])
            if len(idx) != want or spread > stagger + int(s.deploy_group_ticks):
                flags.append("recast_partial")      # not one whole play: a recast would lay
                continue                            # the whole formation again
            played = tick - max(0, total - youngest)
            x = int(round(sum(cands[i][0][2] for i in idx) / len(idx)))
            y = int(round(sum(cands[i][0][3] for i in idx) / len(idx)))
            if side == team:
                aim = _own_aim(own_sent, card, int(pub.get("tick", tick)), total + 3)
                if aim is not None:
                    x, y, played = aim
            elif tap_point is not None:
                got = tap_point(side, card, x, y)
                if got is not None:
                    x, y = int(got[0]), int(got[1])
            rows = _sorted_spawns((cands[i][0], cands[i][1], cands[i][2]) for i in idx)
            recasts.append(Recast(side, card, x, y, "deploy", tables.cost.get(card, 0), int(played),
                                  rows[0], rows[1], rows[2],
                                  sum(int(cands[i][3][F["hp"]]) for i in idx)))
            in_recast.update(idx)

    # --- R2 spell recasts (R1: dropped)
    spells = list(view.get("spells") or ())
    if spells and not (level == R2 and s.spell_recasts):
        flags.append("spells_dropped")
    elif spells:
        value = {team: _targets(view, team, F, sub), opp: _targets(view, opp, F, sub)}
        taken: set[tuple[int, int]] = {(r.team, r.card) for r in recasts}
        for sp in spells:
            side, card = int(sp[SF["team"]]), int(sp[SF["card_id"]])
            if card not in tables.spells:
                flags.append("effect_dropped")
                continue
            if (side, card) in taken:
                flags.append("spell_duplicate")
                continue
            x, y = int(sp[SF["x"]]), int(sp[SF["y"]])
            flown = sp[SF["ticks_flown"]] if "ticks_flown" in SF and SF["ticks_flown"] < len(sp) \
                else 0
            played = tick - max(0, int(flown or 0))
            kind = "area"
            aim: tuple[int, int] | None = (x, y)
            mine = _own_aim(own_sent, card, int(pub.get("tick", tick)), PV.SENT_RECENT_TICKS) \
                if side == team else None
            if card in tables.rolling:
                kind = "log"
                if mine is not None:
                    aim, played = (mine[0], mine[1]), mine[2]
                elif s.log_from == LOG_FIRST and first_sighting is not None:
                    seen = first_sighting(side, card)
                    if seen is not None:
                        aim, played = (seen[1], seen[2]), min(played, seen[0])
            elif card in tables.projectile:
                kind = "spell"
                if mine is not None:
                    aim, played = (mine[0], mine[1]), mine[2]
                else:                       # the caster aims at the other side's value
                    got = infer_aim(side, x / sub, y / sub, value[1 - side],
                                    radius_mt=tables.radius_mt(card), cap=tables.cap(card))
                    aim = None if got is None else (int(round(got[0] * sub)),
                                                    int(round(got[1] * sub)))
            elif mine is not None:
                aim, played = (mine[0], mine[1]), mine[2]
            if aim is None:
                flags.append("spell_unrecast")
                continue
            taken.add((side, card))
            recasts.append(Recast(side, card, int(aim[0]), int(aim[1]), kind,
                                  tables.cost.get(card, 0), int(played)))

    # --- our plays in flight
    inflight = sorted(tuple(int(v) for v in r[:4]) for r in pub.get("own_inflight") or ())
    replays = tuple(r for r in inflight if r[0] < land)
    later = tuple(r for r in inflight if r[0] >= land)

    # --- cycles and the deal
    omit = sorted(tables.omit) if s.omit_rule else []
    recasts.sort(key=lambda r: (r.team, r.order, r.card))
    own_deck = [int(c) for c in view["players"][team].get("deck") or () if int(c) >= 0]
    pools = {team: own_deck + tables.filler,
             opp: [int(c) for c in opp_fill] + [int(c) for c in san["players"][opp]["deck"]]
             + tables.filler}
    hands, queues = [[], []], [[], []]
    for side in (0, 1):
        side_recasts = [r.card for r in recasts if r.team == side]
        h, q, bad, notes = _side_cycle(view, side, side_recasts, pools[side], omit,
                                       tables.spawnable + list(range(len(tables.names))))
        hands[side], queues[side] = h, q
        flags += [f"{'own' if side == team else 'opp'}_{n}" for n in notes]
        if bad:
            flags.append("recast_not_in_cycle")
            back = [r for r in recasts if r.team == side and r.card in bad]
            recasts = [r for r in recasts if not (r.team == side and r.card in bad)]
            for r in back:
                if r.kind == "deploy":
                    in_recast.difference_update(i for i, c in enumerate(cands) if c[2] in r.uids
                                                and c[0][0] == r.team)
                else:
                    flags.append("spell_unrecast")
    own_cards = set(hands[team]) | set(queues[team])
    lost_replays = [r for r in replays + later if r[1] not in own_cards]
    if lost_replays:
        flags.append("replay_not_in_cycle")
        replays = tuple(r for r in replays if r[1] in own_cards)
        later = tuple(r for r in later if r[1] in own_cards)

    def orders(recs: Sequence[Recast]) -> list[list[int]]:
        out = [[r.card for r in recs if r.team == t] for t in (0, 1)]
        seen = set(out[team])
        for r in replays + later:
            if r[1] not in seen:
                out[team].append(r[1])
                seen.add(r[1])
        return out

    # more cards to re-deal than a side has hand slots free: its oldest recasts are undone
    dealt: list[tuple] = [(), ()]
    for side in (0, 1):
        while True:
            try:
                h, q, notes = deal(hands[side], queues[side], orders(recasts)[side], omit=omit)
                break
            except RebuildError:
                side_recasts = [r for r in recasts if r.team == side]
                if not side_recasts:
                    raise
                oldest = min(side_recasts, key=lambda r: (r.order, r.card))
                recasts.remove(oldest)
                flags.append("deal_full")
                if oldest.kind == "deploy":
                    in_recast.difference_update(i for i, c in enumerate(cands)
                                                if c[2] in oldest.uids and c[0][0] == side)
                else:
                    flags.append("spell_unrecast")
        dealt[side] = tuple(h + q)
        flags += notes
    decks = (dealt[0], dealt[1])

    # --- the spawns (formation mode groups multi-unit cards)
    rows = [(c[0], c[1], c[2]) for i, c in enumerate(cands) if i not in in_recast]
    if s.spec_per == FORMATION:
        rows = _formations(rows, tables, sub)
    spawns, labels, uids = _sorted_spawns(rows)
    if not s.relabel:
        labels = (None,) * len(spawns)
        recasts = [r._replace(labels=(None,) * len(r.labels)) for r in recasts]

    # --- elixir
    me = view["players"][team]
    base = [0, 0]
    base[team] = int(me.get("elixir_milli") or 0)
    if s.inflight_paid:
        base[team] += sum(tables.cost.get(r[1], 0) for r in replays + later)
    reading = pub.get("opp_elixir")
    base[opp] = int(reading[0]) if reading else 0
    opp_exact = bool(reading[1]) if reading else False
    elixir, clamped = _elixir(base, recasts)
    if clamped:
        flags.append("elixir_clamped")

    # --- towers, refusals (a side without its three towers is refused: tower_hp None would
    # reset every tower at full hp)
    towers = tuple(tuple(int(h) for h in (san["players"][t].get("tower_hp") or ())[:3])
                   for t in (0, 1))
    tower_hp = towers if all(len(t) == 3 for t in towers) else None
    refusal = None
    if units and dropped / units > s.max_dropped_share:
        refusal = REFUSED_SPEC_DROPPED
    elif tower_hp is None:
        refusal = REFUSED_TOWERS
    regular = int(view.get("regular_ticks") or REGULAR_TICKS)
    if refusal is None and tower_hp is not None and tick >= regular:
        crowns = [sum(1 for k in (1, 2) if tower_hp[1 - t][k] <= 0)
                  + (3 if tower_hp[1 - t][0] <= 0 else 0) for t in (0, 1)]
        if crowns[0] != crowns[1]:
            refusal = REFUSED_OVERTIME
    kept = (pub.get("kept") or ())
    if kept:
        flags.append("permanence")
    return RootSpec(level=level, team=team, tick=tick, land_tick=land, seed=int(seed),
                    decks=decks, elixir=elixir, tower_hp=tower_hp, spawns=spawns, labels=labels,
                    uids=uids, recasts=tuple(recasts), replays=replays, later=later,
                    hands=(tuple(hands[0]), tuple(hands[1])),
                    queues=(tuple(queues[0]), tuple(queues[1])),
                    order=tuple(tuple(o) for o in orders(recasts)), base_elixir=tuple(base),
                    omit=tuple(omit), forms_of=tuple(forms_of), unit_hp=unit_hp, lost_hp=lost_hp,
                    units=units, dropped=dropped, opp_exact=opp_exact, flags=_flags(flags),
                    refusal=refusal)


def _formations(rows: Sequence[tuple], tables: Tables, sub: int) -> list[tuple]:
    """spec_per "formation" (probe P1 FAIL): the units of a multi-unit card (catalogue count k > 1)
    of one side, grouped k at a time in (y, x) order, become one spec at their centroid with their
    mean hp and the first one's label and uid."""
    out, groups = [], defaultdict(list)
    for r in rows:
        (team, card, x, y, hp), label, uid = r
        if tables.count.get(card, 1) > 1:
            groups[(team, card)].append(r)
        else:
            out.append(r)
    for (team, card), members in sorted(groups.items()):
        members.sort(key=lambda r: (r[0][3], r[0][2], r[2] if r[2] is not None else -1))
        k = tables.count[card]
        for i in range(0, len(members), k):
            chunk = members[i:i + k]
            x = int(round(sum(r[0][2] for r in chunk) / len(chunk)))
            y = int(round(sum(r[0][3] for r in chunk) / len(chunk)))
            hp = int(round(sum(r[0][4] for r in chunk) / len(chunk)))
            out.append(((team, card, x, y, max(1, hp)), chunk[0][1], chunk[0][2]))
    return out


# ---------------------------------------------------------------------------
# fidelity
# ---------------------------------------------------------------------------

def fidelity(spec: RootSpec, *, lost_hp: int = 0, refused: bool = False,
             inexact_elixir: float = INEXACT_ELIXIR) -> float:
    """The plan's fidelity score (module doc, FIDELITY): the hp share of the view's units spawned
    or recast (less the spec's lost and `lost_hp` more: recasts refused when they ran) x
    inexact_elixir when the opponent count is not exact; 0 when refused."""
    if refused or spec.refusal:
        return 0.0
    if spec.unit_hp <= 0:
        share = 1.0
    else:
        share = max(0.0, min(1.0, (spec.unit_hp - spec.lost_hp - int(lost_hp)) / spec.unit_hp))
    return share * (1.0 if spec.opp_exact else float(inexact_elixir))


def usable(built: "Built | None", min_fidelity: float = MIN_FIDELITY) -> bool:
    """Whether the search may use this root (plan 4.6): it was built and its fidelity is at least
    min_fidelity."""
    return built is not None and built.fidelity >= float(min_fidelity)


# ---------------------------------------------------------------------------
# the built root
# ---------------------------------------------------------------------------

@dataclass
class Built:
    """A root at B (module doc, THE ROOT): the battle, its save() blob, the state at B (relabelled)
    and its raw state_json (restore verification), uid -> creating card, fidelity, flags, report."""

    spec: RootSpec
    key: str
    battle: Any
    blob: Any
    state: dict
    raw: Any
    labels: dict
    fidelity: float
    flags: tuple
    report: dict

    @property
    def tick(self) -> int:
        return int(self.state.get("tick", self.spec.land_tick))

    @property
    def team(self) -> int:
        return self.spec.team

    @property
    def later(self) -> tuple:
        return self.spec.later

    def hand(self, side: int | None = None) -> list[int]:
        side = self.spec.team if side is None else _team(side)
        return [int(c) for c in self.state["players"][side]["hand"]]

    def slot(self, card: int, side: int | None = None) -> int | None:
        """The slot `card` is in at B (the card mapped to its slot there, plan 3.1 step 7)."""
        hand = self.hand(side)
        return hand.index(int(card)) if int(card) in hand else None


class _Cmd(NamedTuple):
    want: int
    team: int
    card: int
    x: int
    y: int
    kind: str
    ref: int


def _run(battle: Any, start: int, end: int, plan: Iterable[_Cmd], *,
         known: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Step `battle` from `start` to `end`, sending each planned command at its tick (or as soon
    after as its team's earlier commands allow: one command per team per step), its card mapped to
    the slot the state's hand shows then. A command whose card is not in the hand is skipped.
    known: the state at `start`, when the caller has it (saves one state_json)."""
    queues = {0: deque(), 1: deque()}
    for c in plan:
        queues[c.team].append(c)
    cur = int(start)
    out: dict[str, Any] = {"ran": [], "refused": [], "skipped": [], "game_over": False}
    state = known
    while queues[0] or queues[1]:
        due = [t for t in (0, 1) if queues[t] and queues[t][0].want <= cur]
        if not due:
            nxt = min(q[0].want for q in queues.values() if q)
            battle.step([], nxt - cur)
            cur, state = nxt, None
            continue
        if state is None:
            state = json.loads(battle.state_json())
        if state.get("game_over"):
            out["game_over"] = True
            break
        cmds, items = [], []
        for t in due:
            c = queues[t].popleft()
            hand = [int(h) for h in state["players"][t]["hand"]]
            if c.card not in hand:
                out["skipped"].append((c.kind, c.ref, cur))
                continue
            cmds.append((t, hand.index(c.card), int(c.x), int(c.y)))
            items.append(c)
        if not cmds:
            continue
        res = list(battle.step(cmds, 1) or ())
        for i, c in enumerate(items):
            code = _reason(res, i)
            (out["ran"] if code == 0 else out["refused"]).append((c.kind, c.ref, cur, code))
        cur, state = cur + 1, None
    if cur < end and not out["game_over"]:
        battle.step([], end - cur)
        cur = end
    out["tick"] = cur
    return out


# ---------------------------------------------------------------------------
# the rebuilder
# ---------------------------------------------------------------------------

class Rebuilder:
    """One Battle per process, reused with reset (module doc). engine: an rs_engine.EngineConfig
    (or its to_json(); None: rs_engine.active()), always undelayed; rs: the royalesim module (None:
    imported when the battle is first needed); spawn_table: the E06 table (None: read from
    data/rs_rows/spawn_primary.json); settings: RebuildSettings (None: RebuildSettings(), the
    plan's defaults; the probes' answers only when the caller passes RebuildSettings.from_report
    (...), so nothing read from disk later changes a rebuild). catalogue / catalogue_fields /
    fields / spell_fields / sub / level: given instead of read from the engine (tests, workers)."""

    def __init__(self, engine: Any = None, rs: ModuleType | None = None,
                 spawn_table: Mapping[str, Any] | None = None, *,
                 settings: RebuildSettings | None = None,
                 catalogue: Sequence[Sequence[Any]] | None = None,
                 catalogue_fields: Sequence[str] | None = None,
                 fields: Mapping[str, int] | Sequence[str] | None = None,
                 spell_fields: Mapping[str, int] | Sequence[str] | None = None,
                 sub: int | None = None, level: int | None = None):
        self.cfg = rs_engine.resolve(engine).undelayed()
        self._rs = rs
        self._spawn_table = None if spawn_table is None else dict(spawn_table)
        self.settings = settings if settings is not None else RebuildSettings()
        self._catalogue = None if catalogue is None else [list(r) for r in catalogue]
        self._catalogue_fields = None if catalogue_fields is None else list(catalogue_fields)
        self._fields = fields
        self._spell_fields = spell_fields
        self._F: dict[str, int] | None = None
        self._SF: dict[str, int] | None = None
        self._sub = sub
        self._level = level
        self._battle: Any = None
        self._tables: Tables | None = None
        self._snaps: dict[tuple[int, int, int, int], tuple[int, int] | None] = {}
        self._taps: dict[tuple[int, int, int, int], tuple[int, int] | None] = {}
        self._previews: dict[int, tuple | None] = {}
        self._blank = False
        self._current: Built | None = None
        self._dirty = True
        self.sightings = SpellSightings(self.SF, sub=self.sub)
        self.last_error: Refusal | None = None
        self.stats: Counter = Counter()

    # ------------------------------------------------------------ the engine
    @property
    def rs(self) -> ModuleType:
        if self._rs is None:
            self._rs = rs_engine.royalesim_module()
        return self._rs

    @property
    def F(self) -> dict[str, int]:
        """Entity field name -> column: the given fields, else royalesim.ENTITY_FIELDS (checked by
        rs_engine.field_index), else rs_engine.ENTITY_FIELDS_EXPECTED."""
        if self._F is None:
            if self._fields is not None:
                self._F = _index(self._fields, rs_engine.ENTITY_FIELDS_EXPECTED)
            elif self._rs is not None and getattr(self._rs, "ENTITY_FIELDS", None):
                self._F = rs_engine.field_index(self._rs)
            else:
                self._F = _index(None, rs_engine.ENTITY_FIELDS_EXPECTED)
        return self._F

    @property
    def SF(self) -> dict[str, int]:
        """Spell field name -> column: the given ones, else royalesim.SPELL_FIELDS, else S02's."""
        if self._SF is None:
            fields = self._spell_fields
            if fields is None and self._rs is not None:
                fields = getattr(self._rs, "SPELL_FIELDS", None)
            self._SF = _index(fields, PV.SPELL_FIELDS_FALLBACK)
        return self._SF

    @property
    def sub(self) -> int:
        if self._sub is None:
            self._sub = int(getattr(self._rs, "SUBTILE_PER_MILLITILE", SUBTILE_PER_MILLITILE)
                            if self._rs is not None else SUBTILE_PER_MILLITILE)
        return int(self._sub)

    @property
    def battle(self) -> Any:
        """The one Battle (rs_engine.make_battle under cfg.undelayed(): its delay is set, if at
        all, before its first reset)."""
        if self._battle is None:
            self._battle = rs_engine.make_battle(self.cfg, self.rs)
            self._blank = False
        return self._battle

    @property
    def tables(self) -> Tables:
        if self._tables is None:
            cat = self._catalogue
            if cat is None:
                cat = json.loads(self.battle.catalogue_json())
            fields = self._catalogue_fields
            if fields is None and self._rs is not None:
                fields = getattr(self._rs, "CATALOGUE_FIELDS", None)
            table = self._spawn_table
            if table is None:
                table = rs_spawn_table.read(rs_spawn_table.DEFAULT_PATH)
                if table is None:
                    raise FileNotFoundError(f"no spawn table at {rs_spawn_table.DEFAULT_PATH}: run "
                                            "tools/rs_spawn_table.py under this engine")
            level = self._level or self.cfg.level
            if level is None and self._battle is not None:
                lv = getattr(self._battle, "card_level", None)
                level = lv() if callable(lv) else lv
            self._tables = Tables(cat, table, catalogue_fields=fields, level=int(level or 11))
        return self._tables

    # ------------------------------------------------------------ cached pure queries
    def _on_blank(self, accessor: str, unavailable: str) -> Callable[..., Any] | None:
        """The battle's `accessor` (building_placement, formation_preview) with the battle on the
        blank board: reset to it first unless it is there already. Every query a spec makes is
        asked there, never on the board the battle happens to hold (a fresh battle, or the last
        root with its units, towers and tick), so a spec, and its key, is a function of the view
        and the settings alone, whichever root was built before it. None without the accessor,
        or when the engine cannot reset to a blank board (stats[unavailable] counts that)."""
        b = self.battle
        fn = getattr(b, accessor, None)
        if not callable(fn):
            return None
        if not self._blank:
            deck = list(dict.fromkeys(self.tables.filler + self.tables.spawnable))[:DECK_SIZE]
            self._current, self._dirty = None, True
            try:
                b.reset(seed=0, decks=[list(deck), list(deck)], shuffle=0, start_tick=BLANK_TICK,
                        elixir_milli=[ELIXIR_MAX_MILLI, ELIXIR_MAX_MILLI], tower_hp=None,
                        spawns=[])
            except Exception:                   # no blank board: no such query on this engine
                self.stats[unavailable] += 1
                return None
            self._blank = True
        return fn

    def snap(self, team: int, card: int, x: int, y: int) -> tuple[int, int] | None:
        """Where Battle.building_placement lands a tap of `card` by `team` at (x, y) subtiles, asked
        on a blank board (other buildings never move it), cached; None without the accessor or
        when the tap is refused there."""
        key = (int(team), int(card), int(x), int(y))
        if key in self._snaps:
            return self._snaps[key]
        fn = self._on_blank("building_placement", "snap_unavailable")
        got = None
        if fn is not None:
            try:
                res = fn(int(team), self.tables.names[int(card)], int(x), int(y))
            except Exception:                   # an odd card: keep the perceived point
                res = None
            if res is not None:
                got = (int(res[0]), int(res[1]))
        if len(self._snaps) >= SNAP_CACHE:
            self._snaps.clear()
        self._snaps[key] = got
        return got

    def _preview(self, card: int) -> tuple | None:
        """formation_preview's members of a blue play of `card` at the spawn point, asked on the
        blank board (cached), or None without the accessor or for a card it refuses."""
        card = int(card)
        if card in self._previews:
            return self._previews[card]
        rows = None
        fn = (self._on_blank("formation_preview", "preview_unavailable")
              if 0 <= card < len(self.tables.names) else None)
        if fn is not None:
            sx, sy = rs_spawn_table.spawn_point(self.sub)
            try:
                rows = tuple(tuple(r) for r in (fn(0, self.tables.names[card], sx, sy) or ())
                             if len(r) >= 4)
            except Exception:                   # not a deployable card here
                rows = None
        self._previews[card] = rows or None
        return self._previews[card]

    def members(self, card: int) -> tuple[tuple[str, int], ...] | None:
        """(unit name, deploy ticks) of each member a play of `card` lays (formation_preview, in
        creation order: member k's ticks include its formation.DEPLOY_STAGGER wait), or None."""
        rows = self._preview(card)
        if not rows:
            return None
        return tuple((str(r[0]), max(0, int(math.ceil(float(r[3]) / TICK_MS)))) for r in rows)

    def deploy_total(self, card: int) -> int:
        """`card`'s deploy time in ticks: formation_preview's longest member (cached), else
        DEFAULT_DEPLOY_TICKS."""
        ticks = [t for _, t in self.members(card) or ()]
        return max(1, max(ticks)) if ticks and max(ticks) > 0 else DEFAULT_DEPLOY_TICKS

    def tap_point(self, team: int, card: int, x: int, y: int) -> tuple[int, int] | None:
        """The tap whose formation_preview members are centred on (x, y) subtiles: (x, y) less
        the offset of the members' centroid from a tap there (one fixed-point step; a building's
        or a single unit's is (x, y)), asked on the blank board, cached; None without the
        accessor or when it refuses."""
        key = (int(team), int(card), int(x), int(y))
        if key in self._taps:
            return self._taps[key]
        got = None
        fn = (self._on_blank("formation_preview", "preview_unavailable")
              if 0 <= int(card) < len(self.tables.names) else None)
        if fn is not None:
            try:
                rows = [r for r in (fn(int(team), self.tables.names[int(card)], int(x), int(y))
                                    or ()) if len(r) >= 3]
            except Exception:                   # an odd card: keep the centroid
                rows = []
            if rows:
                ox = sum(float(r[1]) for r in rows) / len(rows) - int(x)
                oy = sum(float(r[2]) for r in rows) / len(rows) - int(y)
                got = (int(round(int(x) - ox)), int(round(int(y) - oy)))
        if len(self._taps) >= SNAP_CACHE:
            self._taps.clear()
        self._taps[key] = got
        return got

    # ------------------------------------------------------------ specs
    def spec(self, view: Mapping[str, Any], *, land_tick: int | None = None, seed: int = 0,
             level: str | None = None, opp_fill: Iterable[int] = ()) -> RootSpec:
        """build_spec with this rebuilder's tables, snap, deploy times and spell sightings."""
        level = level or self.settings.level
        if level == R2:
            self.sightings.SF, self.sightings.sub = self.SF, self.sub
            self.sightings.update(view)
        snap = self.snap if self.settings.snap_buildings else None
        return build_spec(view, self.tables, self.settings, land_tick=land_tick, seed=seed,
                          level=level, fields=self.F, spell_fields=self.SF, sub=self.sub,
                          opp_fill=opp_fill, snap=snap, deploy_total=self.deploy_total,
                          first_sighting=self.sightings.first, members=self.members,
                          tap_point=self.tap_point)

    def spec_r0(self, state: Mapping[str, Any], team: int, *, sent: Iterable[Sequence[int]] = (),
                lead: int = 0, public: bool = False) -> RootSpec:
        """r0_spec with this rebuilder's tables (module doc, LEVELS: R0)."""
        return r0_spec(state, team, tables=self.tables, fields=self.F, sent=sent, lead=lead,
                       public=public)

    # ------------------------------------------------------------ the root
    def _refuse(self, reason: str, detail: str = "") -> None:
        self.last_error = Refusal(reason, detail)
        self.stats[f"refused:{reason}"] += 1
        return None

    def _reset(self, spec: RootSpec, flags: list[str]) -> Any:
        """reset from the spec (forms retried without when refused); the engine's error raised."""
        b = self.battle
        self._blank, self._current, self._dirty = False, None, True
        kw = spec.reset_kwargs()
        try:
            b.reset(**kw)
        except Exception:
            if "forms" not in kw:
                raise
            b.reset(**spec.reset_kwargs(forms=False))
            flags.append("forms_refused")
        return b

    def _check(self, b: Any, spec: RootSpec, state: Mapping[str, Any]) -> list[int]:
        """Indices of the recasts check_deploy refuses at t (a card not in its hand counts)."""
        bad = []
        for i, r in enumerate(spec.recasts):
            hand = [int(c) for c in state["players"][r.team]["hand"]]
            if r.card not in hand:
                bad.append(i)
                continue
            if int(b.check_deploy(r.team, hand.index(r.card), r.x, r.y)) != 0:
                bad.append(i)
        return bad

    def root(self, spec: RootSpec) -> Built | None:
        """The root at spec.land_tick (module doc, THE ROOT), or None with last_error set."""
        self.last_error = None
        self.stats["roots"] += 1
        if spec.refusal:
            return self._refuse(spec.refusal, "the spec refused itself")
        flags: list[str] = list(spec.flags)
        try:
            b = self._reset(spec, flags)
        except Exception as exc:                # one bad spec fails the whole reset
            return self._refuse(REFUSED_RESET, f"{type(exc).__name__}: {exc}")
        try:
            got = getattr(b, "command_delay_ticks", None)
            delay = got() if callable(got) else None
            if delay is not None and any(int(d) for d in delay):
                return self._refuse(REFUSED_DELAYED, f"command_delay_ticks {list(delay)}")
            state0 = json.loads(b.state_json())
            if state0.get("game_over"):
                return self._refuse(REFUSED_GAME_OVER, f"winner {state0.get('winner')}")
            report: dict[str, Any] = {"resets": 1, "recasts_refused_check": 0}
            if spec.recasts and self.settings.check_recasts:
                bad = self._check(b, spec, state0)
                if bad:
                    report["recasts_refused_check"] = len(bad)
                    spec = spec.without_recasts(bad)
                    flags = list(spec.flags) + [f for f in flags if f == "forms_refused"]
                    try:
                        b = self._reset(spec, flags)
                    except Exception as exc:
                        return self._refuse(REFUSED_RESET, f"{type(exc).__name__}: {exc}")
                    report["resets"] = 2
                    state0 = json.loads(b.state_json())
                    if state0.get("game_over"):
                        return self._refuse(REFUSED_GAME_OVER, f"winner {state0.get('winner')}")
            labels, rel = self._label_map(spec, state0)
            report["relabel"] = rel
            plan = [_Cmd(spec.tick, r.team, r.card, r.x, r.y, "recast", i)
                    for i, r in enumerate(spec.recasts)]
            plan += [_Cmd(te, spec.team, card, x, y, "replay", i)
                     for i, (te, card, x, y) in enumerate(spec.replays)]
            plan.sort(key=lambda c: (c.team, c.kind != "recast",
                                     spec.recasts[c.ref].order if c.kind == "recast" else c.want,
                                     c.ref))
            ran = _run(b, spec.tick, spec.land_tick, plan, known=state0)
            lost = 0
            for kind, ref, _tick, _code in ran["refused"]:
                if kind == "recast":
                    r = spec.recasts[ref]
                    lost += r.hp
                    flags.append("recast_refused" if r.kind == "deploy" else "spell_unrecast")
                else:
                    flags.append("replay_refused")
            for kind, ref, _tick in ran["skipped"]:
                if kind == "recast":
                    lost += spec.recasts[ref].hp
                    flags.append("recast_skipped")
                else:
                    flags.append("replay_skipped")
            if ran["game_over"]:
                flags.append("game_over_before_land")
            report.update({"ran": len(ran["ran"]), "refused": len(ran["refused"]),
                           "skipped": len(ran["skipped"]), "lost_hp_late": lost,
                           "advance_to": ran["tick"]})
            blob = b.save()
            raw = b.state_json()
            state = self.relabel(json.loads(raw), labels)
            if state.get("game_over"):
                flags.append("game_over_before_land")
            fid = fidelity(spec, lost_hp=lost, inexact_elixir=self.settings.inexact_elixir)
            built = Built(spec=spec, key=spec.key, battle=b, blob=blob, state=state, raw=raw,
                          labels=labels, fidelity=fid, flags=_flags(flags), report=report)
        except Exception as exc:                # the caller counts these (plan 4.6)
            return self._refuse(REFUSED_ERROR, f"{type(exc).__name__}: {exc}")
        self._current, self._dirty = built, False
        self.stats["built"] += 1
        for f in built.flags:
            self.stats[f"flag:{f}"] += 1
        return built

    # ------------------------------------------------------------ relabel
    def _label_map(self, spec: RootSpec, state: Mapping[str, Any]) -> tuple[dict, dict]:
        """uid -> creating card for the entities the reset made (module doc, RELABEL), and the
        counts."""
        F = self.F
        exact: dict[tuple, list[int]] = defaultdict(list)
        by_card: dict[tuple, list[int]] = defaultdict(list)
        for i, (team, card, x, y, _hp) in enumerate(spec.spawns):
            exact[(team, card, x, y)].append(i)
            by_card[(team, card)].append(i)

        def label_of(i: int) -> int | None:
            return spec.labels[i] if i < len(spec.labels) else None

        labels: dict[int, int] = {}
        used: set[int] = set()
        matched = near = unmatched = 0
        rest = []
        for e in state.get("entities") or ():
            if int(e[F["tower_slot"]]) >= 0:
                continue
            key = (int(e[F["team"]]), int(e[F["card_id"]]), int(e[F["x"]]), int(e[F["y"]]))
            free = [i for i in exact.get(key, ()) if i not in used]
            if not free:
                rest.append(e)
                continue
            i = next((j for j in free if spec.spawns[j][4] == int(e[F["hp"]])), free[0])
            used.add(i)
            matched += 1
            if label_of(i) is not None:
                labels[int(e[F["uid"]])] = int(label_of(i))
        reach = FORMATION_MATCH_MT * self.sub
        for e in rest:                          # a formation's members: the nearest spec
            best = None
            for i in by_card.get((int(e[F["team"]]), int(e[F["card_id"]])), ()):
                d = math.hypot(spec.spawns[i][2] - e[F["x"]], spec.spawns[i][3] - e[F["y"]])
                if d <= reach and (best is None or d < best[0]):
                    best = (d, i)
            if best is None:
                unmatched += 1
                continue
            near += 1
            used.add(best[1])
            if label_of(best[1]) is not None:
                labels[int(e[F["uid"]])] = int(label_of(best[1]))
        return labels, {"matched": matched, "near": near, "unmatched": unmatched,
                        "unused": len(spec.spawns) - len(used)}

    def relabel(self, state: dict, labels: Mapping[int, int] | None = None) -> dict:
        """Write each entity's creating card back from `labels` (uid -> card; default the current
        root's), in place; returns the state."""
        if labels is None:
            labels = self._current.labels if self._current is not None else {}
        if not labels:
            return state
        F = self.F
        for e in state.get("entities") or ():
            lab = labels.get(int(e[F["uid"]]))
            if lab is not None and e[F["card_id"]] != lab:
                e[F["card_id"]] = lab
        return state

    # ------------------------------------------------------------ using a root
    def restore(self, built: Built, *, verify: bool = False) -> Any:
        """The battle at `built`'s root: load its blob unless it is there untouched. verify: compare
        the state_json after the load with the root's (plan M1; RebuildError when they differ)."""
        b = built.battle
        if self._current is not built or self._dirty:
            b.load(built.blob)
            self._blank = False                 # a root's board: snap() must reset first
            if verify and b.state_json() != built.raw:
                raise RebuildError("the root's state_json differs after load(): the restore is "
                                   "not exact")
            self._current, self._dirty = built, False
        return b

    def legal(self, card: int, x: int, y: int, built: Built | None = None,
              side: int | None = None) -> int:
        """check_deploy at B of `card` (mapped to its slot in the hand at B) at (x, y) subtiles:
        0 OK, a DEPLOY_REASONS code, or NOT_IN_HAND."""
        built = built or self._current
        if built is None:
            raise RebuildError("legal() needs a root: root(spec) first")
        side = built.team if side is None else _team(side)
        slot = built.slot(card, side)
        if slot is None:
            return NOT_IN_HAND
        b = self.restore(built)
        return int(b.check_deploy(side, slot, int(x), int(y)))

    def trial(self, built: Built, plays: Iterable[Sequence[int]], ticks: int, *,
              later: bool = True, verify: bool = False) -> tuple[dict, dict]:
        """From the root, step `ticks` ticks with `plays` ((tick, team, card, x, y), each card
        mapped to its slot when it is sent; one per team per step) and, with later, our in-flight
        plays that land after B; returns (the relabelled state after, the run's report)."""
        b = self.restore(built, verify=verify)
        start = built.tick
        plan = [_Cmd(max(start, int(p[0])), _team(p[1]), int(p[2]), int(p[3]), int(p[4]),
                     "play", i) for i, p in enumerate(plays)]
        if later:
            plan += [_Cmd(max(start, te), built.team, card, x, y, "later", i)
                     for i, (te, card, x, y) in enumerate(built.later)]
        plan.sort(key=lambda c: (c.team, c.want, c.kind != "later", c.ref))
        self._dirty = True
        ran = _run(b, start, start + max(0, int(ticks)), plan, known=built.state)
        after = self.relabel(json.loads(b.state_json()), built.labels)
        return after, ran

    def summary(self) -> dict[str, Any]:
        """The counters (roots, built, refusals by reason, flags), for seat_info and logs."""
        return dict(sorted(self.stats.items()))
