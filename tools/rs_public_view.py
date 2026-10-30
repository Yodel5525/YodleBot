r"""The public view of a state: only what the acting side's screen shows (S02).

    from tools import rs_public_view as PV          # or `import rs_public_view` (tools/ on sys.path)
    elixir = PV.SeatOppElixir(seat)                   # simulator: the seat's MatchAgent.opp_elixir
    elixir = PV.OppElixir(catalogue, law=law, start_tick=ctx.start_tick)   # live: on ctx.memory
    reading = elixir.view(team, mem_view, tick)
    view = PV.public_view(seen, team, mem_view, fields=RR.F, opp_elixir=reading,
                          sent=seat.sent, unran=seat.unran, mirror=PV.mirror_ids(catalogue),
                          tick=true_tick)             # seat.sent's clock (THE CLOCK)
    keep = PV.Permanence(fields=RR.F, team=team, hiders=PV.card_ids(PV.HIDER_NAMES, names))
    view = keep.apply(view)                           # every decision of the battle, in tick order
    PV.CardCycle.from_view(view, 1 - team)            # the opponent's hand and queue

Why (docs/LIVE_SEARCH_PLAN.md 4.1, 5; S02). In the simulator, tools/rs_match.run_match gives a
seat the true battle and a noised state (rs_match.py:936-941), and that state still holds what no
screen shows: the opponent's hand, elixir, next card, pending plays and deck, the hidden entity
columns, the enemy units the screen does not show, exact spell aims and projectiles
(tools/perception_noise._copy_state copies them all, perception_noise.py:736). The live search
rebuilds a battle from the screen, so the simulator search must rebuild from the same inputs:
public_view() cuts a state down to what tools/public_state.PublicStateBuilder.state produces live
(public_state.py:537-651). Nothing here reads a battle, imports royalesim, torch or numpy, or
touches the emulator; the standard library, tools/rs_elixir, rs_memory and rs_engine (its pure
ENTITY_FIELDS_EXPECTED) only, so a torch-free search worker can import it.

THE VIEW, public_view(state, team, mem_view) (team: the observer; the opponent is 1 - team, in
the simulator either seat, live always team 1):
  entities     every living entity of `team`; an enemy only when rs_rows.seen_by passes
               (rs_rows.py:695: no STATUS_INVISIBLE or STATUS_HIDDEN bit, rs_rows.py:330-334;
               seen_by() here is the same rule, pinned by the tests); crown towers always. Each row
               keeps PUBLIC_FIELDS (public_state.REQUIRED_FIELDS, the columns live fills,
               public_state.py:201-204); status_flags keeps only the bits a screen shows
               (PUBLIC_STATUS: tunnelling, evolved, hero, clone); every other column (target,
               attack phase, stun, knockback, charge, buffs, shield, destination, facing, ability
               timer, mount, radius, footprint) is reset to ENTITY_DEFAULTS
               (public_state.ENTITY_DEFAULTS, public_state.py:207-214). Dead units and units of no
               team are dropped. An enemy whose status_flags is unreported (-1) raises
               PublicViewError, as rs_rows.unit_status does: there is no public view of it.
  players      ours: our own block, copied whole (hand, next, the noised HUD elixir, deck,
               pending; all of it is ours to know). The opponent's: the live block
               (public_state.py:632-635): elixir 0, hand [], next -1, pending [], pending_cost 0,
               deck = the cards its plays in `mem_view` showed, in the order first played; crowns,
               towers and king_active as the state has them (the HUD's).
  spells       each spell row in royalesim.SPELL_FIELDS order (`spell_fields`; default the
               protocol order, rs_rows.SPELL_FIELDS_FALLBACK) as live builds it
               (public_state._spell_row, public_state.py:510-519): team, card, centre, aim = the
               centre (a PLACEHOLDER: never recast a projectile there, plan appendix B),
               ticks_flown (>= 0); every other column 0.
  projectiles  [] (live has none).
  clock        tick, tick_ms, regular_ticks, overtime_ticks, elixir_rate, overtime and game_over
               as the state has them; winner only once the game is over (the end screen), else -1.
               A live state's "perception" and "observer" blocks are kept; any other key is not.
  "public"     what the search needs besides the board (all JSON values):
                 team, tick       the observer and the tick of the in-flight cut
                 opp_elixir       [milli, exact] of the opponent elixir wrapper (below), or None
                 opp_missed       the count saw a play it could not pay: a play was missed
                 opp_plays        [[card or None, exec tick], ...] oldest first, the opponent's
                                  plays the view shows less `unran` (None: an unknown play)
                 own_cycle, opp_cycle   CardCycle.to_json() of each side (below)
                 own_inflight     [[exec tick, card, x, y], ...] of `sent` (rs_match seat.sent,
                                  rs_match.py:961: our plays, engine subtiles) still to run at
                                  the cut (exec tick >= tick: a command that runs at t runs in
                                  the step from t, so the state at t does not show it;
                                  rs_memory), less the ones the engine refused (UNRAN), dated
                                  on the view's clock (THE CLOCK)
                 own_sent         the same for exec tick >= tick - recent_ticks: our own recent
                                  aims, for spells already flying (plan 4.4)
                 sent_shift       what was added to their exec ticks (THE CLOCK; 0 by default)
                 dropped          {reason: count}: "unseen", "dead", "team", "projectiles"
                 kept             uids Permanence added (empty here)
  Live (live=True): the state is already public (PublicStateBuilder's), so the view is the state
  itself, copied, plus "public"; is_public() is asserted first and PublicViewError lists what is
  not public. public_view is idempotent: the view of a view is the view.

UNRAN. rs_match gives a MatchAgent `unran`, a Counter of (team, card token, tick) memory entries
whose play the engine refused (rs_match.py:477-479, 905-910): a play that never ran never showed
on a screen, so each entry takes one equal play out of that team's plays here, for both sides
(MatchAgent.opp_elixir does the same for the opponent, rs_match.py:505-523). rs_match appends a
command to seat.sent before the step says whether it was taken (rs_match.py:961, 993-995), so our
own entries also take one equal play (card, exec tick) out of `sent` (sent_rows): the rebuild
must not replay a play that never runs. (A legacy-memory match dates the entries at the send
tick, not the exec tick, and they then match no sent play.)

THE CLOCK. The view's tick is the observer's clock (in the simulator the noised one: perception
noise's clock, + N(0, sigma) ticks). `sent` must be on the clock of `tick`: by default the view's
own (live: taps are dated by the perceived tick; a simulator seat that logs its own sends at the
tick it was given plus its delay). rs_match's seat.sent is on the TRUE clock (exec = true tick +
delay): pass tick= the true tick, and the cut is made there and every row is moved onto the
view's clock (exec + view tick - tick), so the rebuild's start tick, our replays and our aims
share one clock (public["tick"] is always the view's tick).

CARD CYCLES (CardCycle.from_plays, the same rule for both sides; plan section 5). With shuffle 0
a played card goes to the back of the queue and returns to the hand four plays later, so the
last four plays are four distinct cards and are the queue, oldest (the next card) first; a card
known to be in the deck and not among the last four plays is in the hand. With k < 4 plays the
queue's first 4 - k cards are the starting queue's, unknown (None) unless the side's next card
says (ours: the HUD's). Given our hand (the HUD's, in slot order), the hand is that; cards known
to be in the deck whose slot is still open are placed when only one place is left
(`unplaced` otherwise). The cycle is not `exact` (reasons, in this order) for: "unknown_play"
(token 1), "mirror" (the side played a Mirror), "missed_play" (the elixir counter could not pay a
play it saw: ElixirCounter.unaffordable, rs_elixir.py:578, or a Mirror with nothing to copy),
"repeat" (a card twice among the last four plays: a misread, or a Mirror seen as its copy),
"hand_mismatch" (a card of the given hand among the last four plays; the hand wins),
"next_mismatch" (the given next card is not the oldest of the last four plays) and "confusion"
(more than 8 distinct cards, or more known cards than slots). exact says what is known is
consistent, not that every slot is known. fill() fills the open slots (the unplaced cards first,
then given candidates: a deck prior's draw), Mirror and Elixir Collector into queue slots first
(calibration economy.OMIT_FROM_STARTING_HAND swaps them out of a starting hand; probe P14 says
whether a reset at a later tick does too). Live, our own cycle comes from the HUD
(scripts/vision/hud.py:1035-1039, HandState.cycle): CardCycle.from_known(hand, cycle).

OPPONENT ELIXIR (plan 4.1). Two wrappers with one call, view(team, mem_view, tick) ->
ElixirReading(milli, exact, missed, unaffordable):
  SeatOppElixir(seat)   the simulator: the seat's own MatchAgent.opp_elixir (rs_match.py:505),
                        which counts only the plays that ran (its `unran`), so the search reads
                        the number the seat's v2 rows read; `missed` from its counter.
  OppElixir(catalogue)  live, on ctx.memory, for every checkpoint (vision_play's make_observer
                        gives legacy checkpoints none, vision_play.py:841-856): rs_rows.SeenElixir's
                        rule with lag 0 (rs_rows.py:852), fed each play of the memory view once, as
                        RSAgent.opp_elixir feeds it (rs_eval.py:467-495), less `unran`. Exact as
                        public_state.OpponentElixir (public_state.py:683-747): the counter's own
                        exactness, and not blind, not watching from after the deploy lockout
                        (start_tick > 90), and no unpriced play; an unknown play (token 1) is an
                        unpriced one here (RSAgent.opp_elixir skips it silently).
The count is never repaired from the opponent's real bar: that would be the cheat (rs_elixir).

PERMANENCE (plan 4.1). An enemy building that was seen and never seen destroyed stays in the
rebuild, its hit points drained linearly over its lifetime (max_hp / lifetime per tick, from
its last sighting). The main case is a Tesla gone underground: seen_by drops it, yet it is the
main counter to a Hog (calibration hide.STARTS_HIDDEN; probe P9). "Seen destroyed" is read from
public sightings only (never from the state's hidden rows, which are privileged), on the first
update the building is missing from:
  "dying"    its remaining hit points were within the largest drop (beyond the lifetime drain)
             between two sightings of the last rate_window_ticks: it was being hit that hard
  "spell"    one of our own spells (spell_cards) ran within spell_window_ticks before and landed
             within spell_hit_mt of it (our aims are ours to know: "own_sent")
  "expired"  its drained hit points are gone (its lifetime is over); checked on every update
  "gone"     a building that does not hide (not in `hiders`) unseen for more than grace_ticks:
             every building is on the screen, so it is gone (grace covers a dropped detection)
  "reported" destroyed(uids): perception saw it die (live)
A kept row is its last public row with the drained hp (>= 1), the deploy timer run down and
STATUS_HIDDEN set (out of sight: obs_row under --visibility public and public_view leave it out;
the rebuild spawns it); its uid is listed in view["public"]["kept"]. A building seen again under
a new uid (a live track re-acquired) replaces a kept one of the same card within merge_mt.
Negative uids (perception_noise false positives) are never kept. A view whose tick goes back
starts a new battle.

Never imports scripts/gamestate (the memory reader); never taps, captures or touches the
emulator.
"""
from __future__ import annotations

import copy
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:        # also when imported as tools.rs_public_view
    sys.path.insert(0, str(ROOT / "tools"))

import rs_elixir  # noqa: E402  (pure Python: the elixir law and counter)
import rs_engine  # noqa: E402  (ENTITY_FIELDS_EXPECTED; royalesim only on demand, never here)
import rs_memory  # noqa: E402  (pure Python: tokens, empty_view)

__all__ = ["PublicViewError", "public_view", "is_public", "seen_by", "plays_of", "sent_rows",
           "CardCycle",
           "ElixirReading", "OppElixir", "SeatOppElixir", "make_opp_elixir", "Permanence",
           "card_ids", "mirror_ids", "spell_ids", "lifetimes_from_card_table",
           "PUBLIC_FIELDS", "PUBLIC_STATUS", "ENTITY_DEFAULTS", "SPELL_FIELDS_FALLBACK",
           "HIDER_NAMES", "MIRROR_NAMES", "OMIT_FROM_STARTING_HAND_NAMES"]

SUBTILE_PER_MILLITILE = 18
TICK_MS = 50
HAND_SIZE, DECK_SIZE = 4, 8
QUEUE_SIZE = DECK_SIZE - HAND_SIZE
EMPTY_CARD = -1
KIND_BUILDING = 1
#: royalegym protocol.py EntityState.status_flags bits (rs_rows.py:330-334 for the first three).
STATUS_UNDERGROUND, STATUS_INVISIBLE, STATUS_HIDDEN = 1, 2, 4
STATUS_EVOLVED, STATUS_HERO, STATUS_CLONE = 8, 16, 32
#: rs_rows.UNSEEN_STATUS: an enemy with one of these bits is not on the observer's screen.
UNSEEN_STATUS = STATUS_INVISIBLE | STATUS_HIDDEN
#: The status bits a screen shows: a tunnelling unit's dirt trail, an evolved or hero form, a
#: Clone's tint. The others (windup, ability running, charged, grounded) are reset.
PUBLIC_STATUS = STATUS_UNDERGROUND | STATUS_EVOLVED | STATUS_HERO | STATUS_CLONE
#: public_state.DEPLOY_LOCKOUT_TICKS: a count watching from after it may have missed a play.
DEPLOY_LOCKOUT_TICKS = 90

#: The entity columns a public row keeps: public_state.REQUIRED_FIELDS, what live fills.
PUBLIC_FIELDS: tuple[str, ...] = (
    "uid", "team", "kind", "card_id", "tower_slot", "x", "y", "hp", "max_hp", "flying",
    "deploy_ticks", "status_flags", "level", "unit_type",
)
#: public_state.ENTITY_DEFAULTS: every other column's "nothing" value (lists copied per row).
ENTITY_DEFAULTS: dict[str, Any] = {
    "uid": 0, "team": 0, "kind": 0, "card_id": -1, "tower_slot": -1, "x": 0, "y": 0,
    "hp": 0, "max_hp": 0, "radius": 0, "flying": False, "deploy_ticks": 0, "stun_ticks": 0,
    "knockback_ticks": 0, "footprint": None, "target_uid": -1, "attack_phase": 0,
    "facing": [0, 0], "shield": 0, "buffs": [], "status_flags": 0,
    "level": 11, "mount_uid": -1, "charge": 0, "dest_x": -1, "dest_y": -1,
    "ability_ticks": 0, "unit_type": -1,
}
#: rs_rows.SPELL_FIELDS_FALLBACK, royalegym protocol.SpellState's order.
SPELL_FIELDS_FALLBACK: tuple[str, ...] = (
    "team", "card_id", "motion", "x", "y", "aim_x", "aim_y", "delay_ticks", "travelled", "length",
    "hits", "ticks_flown")
#: The spell columns a public spell row keeps (aim_x / aim_y are set to the centre).
SPELL_KEEP = ("team", "card_id", "x", "y")
#: The top-level keys a view keeps from the state (players, entities, spells, projectiles and
#: "public" are built).
TOP_KEYS = ("tick", "tick_ms", "regular_ticks", "overtime_ticks", "elixir_rate", "overtime",
            "game_over", "winner", "perception", "observer")
#: own_sent keeps our plays this many ticks back (a spell's flight is shorter).
SENT_RECENT_TICKS = 100

#: Card names (catalogue names) the helpers resolve with card_ids().
HIDER_NAMES = ("Tesla",)
MIRROR_NAMES = ("Mirror",)
OMIT_FROM_STARTING_HAND_NAMES = ("Mirror", "Elixir Collector")
#: catalogue_json column of the placement code (6: the Mirror) and of card_kind.
CATALOGUE_PLACEMENT, MIRROR_PLACEMENT, CATALOGUE_KIND = 1, 6, 8

# CardCycle reasons, in the order they are listed.
R_UNKNOWN, R_MIRROR, R_MISSED = "unknown_play", "mirror", "missed_play"
R_REPEAT, R_HAND, R_NEXT, R_CONFUSION = "repeat", "hand_mismatch", "next_mismatch", "confusion"
REASONS = (R_UNKNOWN, R_MIRROR, R_MISSED, R_REPEAT, R_HAND, R_NEXT, R_CONFUSION)

# Permanence defaults.
DEFAULT_LIFETIME_TICKS = 600          # 30 s: most buildings' lifetime (Cannon, Inferno, Tombstone)
GRACE_TICKS = 30                      # a building that does not hide, unseen this long, is gone
RATE_WINDOW_TICKS = 64                # two Hog hits (1.6 s apart) fit in it
MERGE_MT = 1500.0
SPELL_HIT_MT = 3000.0                 # a Fireball's 2.5-tile radius plus half a building
SPELL_WINDOW_TICKS = 60
HISTORY = 16                          # sightings kept per building


class PublicViewError(ValueError):
    """The state cannot be cut to a public view (an unreported status), or a state said to be
    public is not (public_view(live=True))."""


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


def _fields(fields: Mapping[str, int] | Sequence[str] | None) -> dict[str, int]:
    F = _index(fields, rs_engine.ENTITY_FIELDS_EXPECTED)
    missing = [n for n in PUBLIC_FIELDS if n not in F]
    if missing:
        raise ValueError(f"entity fields lack {missing}: pass royalesim.ENTITY_FIELDS "
                         "(rs_engine.field_index, or rs_rows.F after _setup)")
    return F


def _spell_fields(fields: Mapping[str, int] | Sequence[str] | None) -> dict[str, int]:
    SF = _index(fields, SPELL_FIELDS_FALLBACK)
    missing = [n for n in SPELL_KEEP + ("aim_x", "aim_y") if n not in SF]
    if missing:
        raise ValueError(f"spell fields lack {missing}: pass royalesim.SPELL_FIELDS")
    return SF


def _card(value: Any) -> int | None:
    """A catalogue card id, or None for an unknown card (None or negative)."""
    if value is None or isinstance(value, bool):
        return None
    c = int(value)
    return c if c >= 0 else None


def _unique(values: Iterable[Any]) -> list:
    out: list = []
    for v in values:
        if v is not None and v not in out:
            out.append(v)
    return out


def _get(row: Sequence[Any], i: int, default: Any) -> Any:
    return row[i] if i < len(row) else default


def _status(value: Any) -> int | None:
    """status_flags as an int; None when the engine did not report it (-1, None, a bool)."""
    if value is None or isinstance(value, bool):
        return None
    v = int(value)
    return v if v >= 0 else None


def card_ids(names: Iterable[str], card_names: Sequence[str]) -> list[int]:
    """The catalogue ids of `names` that `card_names` (catalogue order) has, in that order."""
    index = {str(n): i for i, n in enumerate(card_names)}
    return [index[n] for n in names if n in index]


def mirror_ids(catalogue: Sequence[Sequence[Any]]) -> list[int]:
    """The Mirror cards of catalogue_json() rows: placement code 6 or the name "Mirror"
    (rs_elixir.ElixirCounter.from_catalogue's rule)."""
    return [i for i, row in enumerate(catalogue)
            if row and (row[0] in MIRROR_NAMES
                        or (len(row) > CATALOGUE_PLACEMENT
                            and row[CATALOGUE_PLACEMENT] == MIRROR_PLACEMENT))]


def spell_ids(catalogue: Sequence[Sequence[Any]], kind_column: int = CATALOGUE_KIND) -> list[int]:
    """The SPELL cards of catalogue_json() rows (card_kind, column 8)."""
    return [i for i, row in enumerate(catalogue)
            if len(row) > kind_column and row[kind_column] == "SPELL"]


def lifetimes_from_card_table(table: Mapping[str, Any], card_names: Sequence[str],
                              tick_ms: int = TICK_MS) -> dict[int, int]:
    """card id -> lifetime in ticks, from a RoyaleSim card table document (data/derived/
    cards*.json, or vision_classes.engine_card_table(battle, royalesim)): its "cards" entries'
    lifetime_ms, for the cards `card_names` (catalogue order) names. Cards without one are left
    out (Permanence then uses its default)."""
    index = {str(n): i for i, n in enumerate(card_names)}
    out: dict[int, int] = {}
    for rec in table.get("cards") or ():
        if not isinstance(rec, Mapping):
            continue
        ms, name = rec.get("lifetime_ms"), rec.get("name")
        if name in index and ms is not None and float(ms) > 0:
            out[index[name]] = max(1, int(math.ceil(float(ms) / int(tick_ms))))
    return out


def seen_by(row: Sequence[Any], team: int, fields: Mapping[str, int] | Sequence[str] | None = None
            ) -> bool:
    """rs_rows.seen_by (rs_rows.py:695): `team`'s own units always; an enemy unless it has an
    UNSEEN_STATUS bit. PublicViewError for an enemy whose status_flags is unreported (-1)."""
    return _seen(row, int(team), _fields(fields))


def _seen(row: Sequence[Any], team: int, F: Mapping[str, int]) -> bool:
    if int(row[F["team"]]) == team:
        return True
    status = _status(row[F["status_flags"]])
    if status is None:
        raise PublicViewError(f"the unit uid {row[F['uid']]} has no status_flags "
                              f"({row[F['status_flags']]!r}); this engine does not report them, "
                              "so no public view of it exists")
    return not status & UNSEEN_STATUS


# ---------------------------------------------------------------------------
# plays from a memory view
# ---------------------------------------------------------------------------

def _last(mem_view: Mapping[str, Any] | None, team: int) -> list:
    last = (mem_view or {}).get("last") or {}
    got = last.get(team)
    if got is None:
        got = last.get(str(team))               # a view that went through JSON
    return list(got or ())


def _entries(mem_view: Mapping[str, Any] | None, team: int,
             unran: Mapping[tuple, int] | None = None) -> list[tuple[int, int]]:
    """(card token, exec tick) of `team`'s plays in the view, oldest first, less one equal entry
    for each of `unran`'s (team, token, tick) counts."""
    drop = Counter({(int(c), int(t)): int(n) for (who, c, t), n in (unran or {}).items()
                    if int(who) == team and int(n) > 0})
    out = []
    for token, when in _last(mem_view, team):
        key = (int(token), int(when))
        if drop[key] > 0:
            drop[key] -= 1
            continue
        out.append(key)
    return out


def plays_of(mem_view: Mapping[str, Any] | None, team: int,
             unran: Mapping[tuple, int] | None = None) -> list[tuple[int | None, int]]:
    """`team`'s plays in a memory view (rs_memory.PublicMemory.view: "last", tokens = card + 2,
    1 unknown), oldest first, as (catalogue card or None for an unknown play, exec tick), less
    the `unran` entries (module doc, UNRAN)."""
    team = _team(team)
    out = []
    for token, when in _entries(mem_view, team, unran):
        card = token - rs_memory.TOKEN_OFFSET
        out.append((card if card >= 0 else None, when))
    return out


# ---------------------------------------------------------------------------
# card cycles
# ---------------------------------------------------------------------------

def _slots(values: Iterable[Any] | None, n: int) -> list[int | None]:
    """`n` slots of distinct cards (None: unknown or empty; a repeat reads None), cut or padded."""
    out: list[int | None] = []
    for v in list(values or ())[:n]:
        c = _card(v)
        out.append(None if c is not None and c in out else c)
    return out + [None] * (n - len(out))


@dataclass(frozen=True)
class CardCycle:
    """One side's hand and queue (module doc, CARD CYCLES).

    hand      HAND_SIZE cards (None: unknown or empty); given hands keep their slot order,
              derived ones list the known cards in first-seen order
    queue     QUEUE_SIZE cards in the order they come back; queue[0] is the next card, the
              last ones the most recent plays
    exact     no reason below; what is known is consistent
    reasons   why not exact (REASONS order)
    plays     the plays counted
    seen      every known card of the side, first seen first (plays, deck, hand, next)
    unplaced  cards known to be in the deck whose slot is open (fill places them first)
    filled    slots fill() filled
    """
    hand: tuple = (None,) * HAND_SIZE
    queue: tuple = (None,) * QUEUE_SIZE
    exact: bool = True
    reasons: tuple = ()
    plays: int = 0
    seen: tuple = ()
    unplaced: tuple = ()
    filled: int = 0

    # ------------------------------------------------------------ building
    @classmethod
    def from_plays(cls, plays: Iterable[Any], *, hand: Iterable[Any] | None = None,
                   next_card: Any = None, deck: Iterable[Any] = (), mirror: Iterable[int] = (),
                   missed: bool = False) -> "CardCycle":
        """The cycle after `plays` (catalogue ids oldest first; None or < 0: an unknown play;
        (card, tick) pairs are read as their card). hand / next_card: the side's own HUD (ours);
        deck: cards known to be in the side's deck; mirror: the Mirror card ids; missed: the
        elixir counter found a play it could not pay."""
        cards = [_card(p[0] if isinstance(p, (tuple, list)) else p) for p in plays]
        mirror = frozenset(int(m) for m in mirror)
        reasons: list[str] = []
        if any(c is None for c in cards):
            reasons.append(R_UNKNOWN)
        if any(c in mirror for c in cards if c is not None):
            reasons.append(R_MIRROR)
        if missed:
            reasons.append(R_MISSED)
        # The queue's tail: the last plays, oldest first; a card twice there keeps its latest.
        tail = cards[-QUEUE_SIZE:]
        q = [None if c is not None and c in tail[i + 1:] else c for i, c in enumerate(tail)]
        if q != tail:
            reasons.append(R_REPEAT)
        queue = [None] * (QUEUE_SIZE - len(q)) + q
        given = None if hand is None else _slots(hand, HAND_SIZE)
        if given is not None:
            clash = {c for c in given if c is not None and c in queue}
            if clash:
                reasons.append(R_HAND)
                queue = [None if c in clash else c for c in queue]
        nxt = _card(next_card)
        if nxt is not None:
            if queue[0] is None and nxt not in queue and (given is None or nxt not in given):
                queue[0] = nxt
            elif queue[0] != nxt:
                reasons.append(R_NEXT)
        seen = _unique([c for c in cards if c is not None] + [_card(c) for c in deck]
                       + list(given or ()) + [nxt])
        confused = len(seen) > DECK_SIZE
        if given is not None:
            h = list(given)
        else:
            # A known card not among the last four plays was played four plays ago or more:
            # it is back in the hand. Cards never played may still sit in the queue's head.
            played = {c for c in cards if c is not None}
            last_at = {c: i for i, c in enumerate(cards) if c is not None}
            certain = [c for c in seen if c not in queue and c in played]
            if None not in queue:
                certain += [c for c in seen if c not in queue and c not in played]
            if len(certain) > HAND_SIZE:
                confused = True
                recent = set(sorted(certain, key=lambda c: -last_at.get(c, -1))[:HAND_SIZE])
                certain = [c for c in certain if c in recent]
            h = certain + [None] * (HAND_SIZE - len(certain))
        rest = [c for c in seen if c not in h and c not in queue]
        holes_h, holes_q = h.count(None), queue.count(None)
        if len(rest) > holes_h + holes_q:
            confused = True
        if rest and holes_q == 0:
            for c in rest[:holes_h]:
                h[h.index(None)] = c
            rest = rest[holes_h:]
        elif rest and holes_h == 0 and holes_q == 1 and len(rest) == 1:
            queue[queue.index(None)] = rest.pop()
        if confused:
            reasons.append(R_CONFUSION)
        reasons = [r for r in REASONS if r in reasons]
        return cls(hand=tuple(h), queue=tuple(queue), exact=not reasons, reasons=tuple(reasons),
                   plays=len(cards), seen=tuple(seen),
                   unplaced=tuple(c for c in rest if c not in h and c not in queue))

    @classmethod
    def from_known(cls, hand: Iterable[Any], queue: Iterable[Any], *, exact: bool = True,
                   reasons: Iterable[str] = (), plays: int = 0) -> "CardCycle":
        """A cycle read directly: live, our HUD's hand and HandState.cycle (cards not in hand
        in the order they return, [0] the next card; None for one not seen yet)."""
        h, q = _slots(hand, HAND_SIZE), _slots(queue, QUEUE_SIZE)
        q = [None if c is not None and c in h else c for c in q]
        why = tuple(reasons)
        return cls(hand=tuple(h), queue=tuple(q), exact=bool(exact) and not why, reasons=why,
                   plays=int(plays), seen=tuple(_unique(h + q)))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "CardCycle":
        return cls(hand=tuple(_card(c) for c in data.get("hand") or (None,) * HAND_SIZE),
                   queue=tuple(_card(c) for c in data.get("queue") or (None,) * QUEUE_SIZE),
                   exact=bool(data.get("exact", True)), reasons=tuple(data.get("reasons") or ()),
                   plays=int(data.get("plays") or 0), seen=tuple(data.get("seen") or ()),
                   unplaced=tuple(data.get("unplaced") or ()), filled=int(data.get("filled") or 0))

    @classmethod
    def from_view(cls, view: Mapping[str, Any], team: int) -> "CardCycle":
        """The cycle public_view stored for `team` (the observer's own_cycle or opp_cycle)."""
        pub = view["public"]
        return cls.from_json(pub["own_cycle"] if int(team) == int(pub["team"]) else pub["opp_cycle"])

    # ------------------------------------------------------------ reading
    @property
    def next(self) -> int | None:
        return self.queue[0]

    @property
    def known(self) -> list[int]:
        """The cards whose slot is known (hand, then queue)."""
        return [c for c in self.hand + self.queue if c is not None]

    @property
    def holes(self) -> tuple[int, int]:
        """(open hand slots, open queue slots)."""
        return self.hand.count(None), self.queue.count(None)

    def order(self) -> list[int | None]:
        """hand then queue: the order a shuffle-0 reset deals (deck[0:4] the hand, deck[4:8] the
        queue, deck[4] next)."""
        return list(self.hand) + list(self.queue)

    def fill(self, cards: Iterable[Any] = (), omit: Iterable[int] = ()) -> "CardCycle":
        """A copy with the open slots filled: the unplaced cards first, then `cards` (a deck
        prior's draw, say) not already in the cycle, in order. A card of `omit` (Mirror and Elixir
        Collector: OMIT_FROM_STARTING_HAND_NAMES) takes a queue slot while there is one; any
        other a hand slot while there is one. exact and reasons are kept."""
        hand, queue = list(self.hand), list(self.queue)
        have = set(self.known)
        omit = {int(c) for c in omit}
        pool = _unique([c for c in list(self.unplaced) + [_card(x) for x in cards]
                        if c is not None and c not in have])
        n = 0
        for c in pool:
            hh = [i for i, x in enumerate(hand) if x is None]
            qq = [i for i, x in enumerate(queue) if x is None]
            if not hh and not qq:
                break
            if c in omit and qq:
                queue[qq[0]] = c
            elif hh:
                hand[hh[0]] = c
            else:
                queue[qq[0]] = c
            n += 1
        placed = set(hand) | set(queue)
        return replace(self, hand=tuple(hand), queue=tuple(queue), filled=self.filled + n,
                       unplaced=tuple(c for c in self.unplaced if c not in placed))

    def to_json(self) -> dict[str, Any]:
        return {"hand": list(self.hand), "queue": list(self.queue), "exact": self.exact,
                "reasons": list(self.reasons), "plays": self.plays, "seen": list(self.seen),
                "unplaced": list(self.unplaced), "filled": self.filled}


# ---------------------------------------------------------------------------
# opponent elixir
# ---------------------------------------------------------------------------

class ElixirReading(NamedTuple):
    """The opponent's bar as the observer counts it. milli: elixir_milli at the tick, after the
    plays dated that tick; exact: the count can be trusted (the fidelity score's x 0.8 when
    not); missed: the counter itself saw a play it could not pay (or a Mirror with nothing to
    copy): a play was missed, so the card cycle is not exact either; unaffordable: the counter's
    count of such plays (-1: unknown)."""
    milli: int
    exact: bool
    missed: bool = False
    unaffordable: int = 0

    @property
    def pair(self) -> tuple[int, bool]:
        """(milli, exact): what obs_row's opp_elixir takes."""
        return int(self.milli), bool(self.exact)


class OppElixir:
    """The opponent's elixir from a memory view: live on ctx.memory, for every checkpoint
    (module doc, OPPONENT ELIXIR). One per battle; a view that no longer holds a play already
    fed, or another observer team, starts a new count (a new battle).

    catalogue   Battle.catalogue_json() rows (prices; the Mirror by placement 6 or name)
    fields      royalesim.CATALOGUE_FIELDS (None: rs_elixir.CATALOGUE_PREFIX's order)
    law         the engine's rs_elixir.ElixirLaw (rs_rows.elixir_law(engine)); None: the default
    start_tick  the tick the bot started watching (the runner's battle start)
    blind       no play can be seen (no detector): never exact
    """

    def __init__(self, catalogue: Sequence[Sequence[Any]], *, fields: Sequence[str] | None = None,
                 law: Any = None, start_tick: int | None = 0, blind: bool = False):
        self.catalogue = [list(row) for row in catalogue]
        self.fields = None if fields is None else list(fields)
        self.law = law
        self.start_tick = 0 if start_tick is None else max(0, int(start_tick))
        self.blind = bool(blind)
        self.counter: rs_elixir.ElixirCounter | None = None
        self.unpriced = 0
        self.restarts = 0
        self._team: int | None = None
        self._fed: Counter = Counter()

    @property
    def late(self) -> bool:
        """Watching from after the deploy lockout: a play may have gone unseen."""
        return self.start_tick > DEPLOY_LOCKOUT_TICKS

    def _restart(self, team: int) -> None:
        self.counter = rs_elixir.ElixirCounter.from_catalogue(self.catalogue, fields=self.fields,
                                                              law=self.law)
        self.unpriced = 0
        self.restarts += 1
        self._team = team
        self._fed = Counter()

    def view(self, team: int, mem_view: Mapping[str, Any] | None, tick: int, *,
             unran: Mapping[tuple, int] | None = None) -> ElixirReading:
        """The opponent's (1 - team's) bar at `tick` from the plays `mem_view` shows (lag 0: the
        view holds what the observer has seen), less `unran`."""
        team = _team(team)
        plays = _entries(mem_view, 1 - team, unran)
        held = Counter(plays)
        if (self.counter is None or self._team != team
                or any(held[k] < n for k, n in self._fed.items())):
            self._restart(team)
        counter = self.counter
        counts: Counter = Counter()
        for key in plays:
            counts[key] += 1
            if counts[key] <= self._fed[key]:
                continue
            self._fed[key] += 1
            token, when = key
            card = token - rs_memory.TOKEN_OFFSET
            if when < 0 or card not in counter.costs:
                self.unpriced += 1          # an unknown play (token 1) or an unpriced card
                continue
            counter.play(when, card)
        milli = counter.elixir_milli(max(0, int(tick)))
        own_exact = counter.exact
        exact = own_exact and not self.blind and not self.late and not self.unpriced
        return ElixirReading(int(milli), bool(exact), not own_exact, int(counter.unaffordable))


class SeatOppElixir:
    """The simulator's opponent elixir: the seat's own MatchAgent.opp_elixir(team, mem, tick)
    (rs_match.py:505; it leaves out the plays in the seat's `unran`), so the search reads the
    number the seat's rows read. missed / unaffordable from the seat's counter (RSAgent._seen,
    an rs_rows.SeenElixir) when it has one."""

    def __init__(self, seat: Any):
        if not callable(getattr(seat, "opp_elixir", None)):
            raise TypeError("SeatOppElixir needs a seat with opp_elixir(team, mem, tick) "
                            "(rs_match.MatchAgent)")
        self.seat = seat

    def view(self, team: int, mem_view: Mapping[str, Any] | None, tick: int, *,
             unran: Mapping[tuple, int] | None = None) -> ElixirReading:
        """`unran` is ignored: the seat reads its own."""
        team = _team(team)
        milli, exact = self.seat.opp_elixir(team, mem_view or rs_memory.empty_view(), int(tick))
        seen = getattr(self.seat, "_seen", None)
        counters = getattr(seen, "counters", None)
        counter = counters.get(1 - team) if isinstance(counters, Mapping) else None
        if counter is None:
            return ElixirReading(int(milli), bool(exact), not exact, 0 if exact else -1)
        return ElixirReading(int(milli), bool(exact), not counter.exact, int(counter.unaffordable))


def make_opp_elixir(seat: Any = None, catalogue: Sequence[Sequence[Any]] | None = None,
                    **kwargs: Any) -> OppElixir | SeatOppElixir:
    """SeatOppElixir(seat) for a seat with opp_elixir, else OppElixir(catalogue, **kwargs)."""
    if seat is not None and callable(getattr(seat, "opp_elixir", None)):
        return SeatOppElixir(seat)
    if catalogue is None:
        raise ValueError("make_opp_elixir: pass a seat with opp_elixir, or the catalogue")
    return OppElixir(catalogue, **kwargs)


def _reading(value: Any) -> ElixirReading | None:
    if value is None or isinstance(value, ElixirReading):
        return value
    milli, exact = value
    return ElixirReading(int(milli), bool(exact), False, 0)


# ---------------------------------------------------------------------------
# the view
# ---------------------------------------------------------------------------

def _template(F: Mapping[str, int]) -> list:
    row: list[Any] = [0] * (max(F.values()) + 1)
    for name, i in F.items():
        row[i] = ENTITY_DEFAULTS.get(name, 0)
    return row


def _public_row(e: Sequence[Any], F: Mapping[str, int], template: Sequence[Any]) -> list:
    row = [list(v) if isinstance(v, list) else v for v in template]
    for name in PUBLIC_FIELDS:
        i = F[name]
        row[i] = _get(e, i, row[i])
    status = _status(_get(e, F["status_flags"], 0))
    row[F["status_flags"]] = 0 if status is None else status & PUBLIC_STATUS
    return row


def _spell_row(sp: Sequence[Any], SF: Mapping[str, int]) -> list:
    row: list[Any] = [0] * (max(SF.values()) + 1)
    for name in SPELL_KEEP:
        row[SF[name]] = _get(sp, SF[name], 0)
    row[SF["aim_x"]], row[SF["aim_y"]] = row[SF["x"]], row[SF["y"]]
    at = SF.get("ticks_flown")
    if at is not None:
        flown = _get(sp, at, 0)
        row[at] = max(0, int(flown)) if flown is not None else 0
    return row


def _strip(state: Mapping[str, Any], team: int, F: Mapping[str, int], SF: Mapping[str, int],
           opp_plays: Sequence[tuple[int | None, int]]) -> tuple[dict, Counter]:
    """The public copy of a simulator state (module doc, THE VIEW) and what it dropped."""
    opp = 1 - team
    out = {k: copy.deepcopy(state[k]) for k in TOP_KEYS if k in state}
    if "winner" in out and not state.get("game_over"):
        out["winner"] = -1
    players = state["players"]
    me = copy.deepcopy(dict(players[team]))
    src = players[opp]
    them = {"team": opp, "elixir_milli": 0, "hand": [], "hand_costs": [],
            "next_card": EMPTY_CARD, "deck": _unique(c for c, _ in opp_plays),
            "crowns": int(src.get("crowns", 0) or 0),
            "tower_hp": list(src.get("tower_hp") or ()),
            "tower_max_hp": list(src.get("tower_max_hp") or ()),
            "king_active": bool(src.get("king_active", False)),
            "pending": [], "pending_cost": 0}
    out["players"] = [me, them] if team == 0 else [them, me]
    dropped: Counter = Counter()
    template = _template(F)
    ents = []
    for e in state.get("entities") or ():
        hp = _get(e, F["hp"], 0)
        if hp is None or hp <= 0:
            dropped["dead"] += 1
            continue
        t = _get(e, F["team"], -1)
        if t not in (0, 1):
            dropped["team"] += 1
            continue
        if int(_get(e, F["tower_slot"], -1)) < 0 and not _seen(e, team, F):
            dropped["unseen"] += 1
            continue
        ents.append(_public_row(e, F, template))
    out["entities"] = ents
    out["spells"] = [_spell_row(sp, SF) for sp in state.get("spells") or ()]
    n_proj = len(state.get("projectiles") or ())
    if n_proj:
        dropped["projectiles"] += n_proj
    out["projectiles"] = []
    return out, dropped


def public_view(state: Mapping[str, Any], team: int, mem_view: Mapping[str, Any] | None = None, *,
                fields: Mapping[str, int] | Sequence[str] | None = None,
                spell_fields: Mapping[str, int] | Sequence[str] | None = None,
                opp_elixir: Any = None, sent: Iterable[Sequence[int]] = (),
                unran: Mapping[tuple, int] | None = None, mirror: Iterable[int] = (),
                own_cycle: CardCycle | None = None, opp_cycle: CardCycle | None = None,
                tick: int | None = None, recent_ticks: int = SENT_RECENT_TICKS,
                live: bool = False) -> dict:
    """`team`'s public view of `state` (module doc, THE VIEW): a new dict; the input is not
    changed.

    mem_view      the observer's memory view (rs_memory.PublicMemory.view), already
                  noise-filtered: the opponent's plays (its seen deck and cycle) and, in the
                  simulator, our own run plays (our cycle); None: no plays
    fields, spell_fields  royalesim.ENTITY_FIELDS / SPELL_FIELDS (names or name -> index)
    opp_elixir    an ElixirReading (OppElixir / SeatOppElixir .view) or (milli, exact); None
    sent          our plays sent, (exec tick, card, x, y) engine subtiles (rs_match seat.sent),
                  dated on the clock of `tick` (module doc, THE CLOCK); one sent play per
                  `unran` entry of our own (team, card token, exec tick) is left out: the engine
                  refused it, so it never runs (module doc, UNRAN)
    unran         the seat's Counter of memory entries the engine refused (module doc, UNRAN)
    mirror        the Mirror card ids (mirror_ids): a Mirror play makes a cycle inexact
    own_cycle, opp_cycle  cycles given instead of derived (live: our HUD's, CardCycle.from_known)
    tick          now on the clock `sent` is dated by, where the in-flight cut is made (default
                  the view's own tick: `sent` is on the view's clock, as live). Given (the
                  simulator seat's true tick: seat.sent is on the true clock), the rows are cut
                  there and moved onto the view's clock by view tick - tick
    live          the state is already public (PublicStateBuilder.state): asserted (is_public),
                  then copied unchanged
    """
    team = _team(team)
    F, SF = _fields(fields), _spell_fields(spell_fields)
    reading = _reading(opp_elixir)
    own_plays = plays_of(mem_view, team, unran)
    opp_plays = plays_of(mem_view, 1 - team, unran)
    if live:
        problems = is_public(state, team, fields=F, spell_fields=SF)
        if problems:
            raise PublicViewError("live=True, but the state is not public: " + "; ".join(problems))
        out = copy.deepcopy({k: v for k, v in state.items() if k != "public"})
        dropped: Counter = Counter()
    else:
        out, dropped = _strip(state, team, F, SF, opp_plays)
    me = out["players"][team]
    if own_cycle is None:
        own_cycle = CardCycle.from_plays([c for c, _ in own_plays], hand=me.get("hand"),
                                         next_card=me.get("next_card"),
                                         deck=me.get("deck") or (), mirror=mirror)
    if opp_cycle is None:
        opp_cycle = CardCycle.from_plays([c for c, _ in opp_plays], mirror=mirror,
                                         missed=bool(reading is not None and reading.missed))
    now = int(out.get("tick") or 0)
    t = now if tick is None else int(tick)
    shift = now - t
    rows = sent_rows(sent, team, unran)
    out["public"] = {
        "team": team, "tick": now,
        "opp_elixir": None if reading is None else [int(reading.milli), bool(reading.exact)],
        "opp_missed": bool(reading is not None and reading.missed),
        "opp_plays": [[c, int(w)] for c, w in opp_plays],
        "own_cycle": own_cycle.to_json(), "opp_cycle": opp_cycle.to_json(),
        "own_inflight": [[r[0] + shift, *r[1:]] for r in rows if r[0] >= t],
        "own_sent": [[r[0] + shift, *r[1:]] for r in rows if r[0] >= t - int(recent_ticks)],
        "sent_shift": shift,
        "dropped": dict(sorted(dropped.items())),
        "kept": [],
    }
    return out


def sent_rows(sent: Iterable[Sequence[int]] | None, team: int,
              unran: Mapping[tuple, int] | None = None) -> list[tuple[int, int, int, int]]:
    """Our sent plays (exec tick, card, x, y), sorted, less one play of each of `unran`'s entries
    of our own (team, card token, exec tick): a play the engine refused at send or when it was
    due never runs (rs_match never_ran puts the seat's own entry of it in its unran)."""
    team = _team(team)
    drop = Counter({(int(c) - rs_memory.TOKEN_OFFSET, int(t)): int(n)
                    for (who, c, t), n in (unran or {}).items() if int(who) == team and int(n) > 0})
    out = []
    for r in sorted(tuple(int(v) for v in s[:4]) for s in sent or ()):
        key = (r[1], r[0])
        if drop[key] > 0:
            drop[key] -= 1
            continue
        out.append(r)
    return out


def is_public(state: Mapping[str, Any], team: int, *,
              fields: Mapping[str, int] | Sequence[str] | None = None,
              spell_fields: Mapping[str, int] | Sequence[str] | None = None) -> list[str]:
    """What in `state` a screen does not show to `team` ([] when it is public): an opponent block
    with elixir, hand, next card or pending plays; projectiles; an enemy with an UNSEEN_STATUS
    bit (or unreported status), except the uids Permanence kept (state["public"]["kept"]); an
    entity column outside PUBLIC_FIELDS that is not ENTITY_DEFAULTS'; a status bit outside
    PUBLIC_STATUS (kept rows aside); a spell whose aim is not its centre."""
    team = _team(team)
    F, SF = _fields(fields), _spell_fields(spell_fields)
    problems: list[str] = []
    them = state["players"][1 - team]
    if them.get("elixir_milli"):
        problems.append(f"opponent elixir {them['elixir_milli']}")
    if them.get("hand"):
        problems.append(f"opponent hand {list(them['hand'])}")
    if them.get("next_card", EMPTY_CARD) not in (EMPTY_CARD, None):
        problems.append(f"opponent next card {them['next_card']}")
    if them.get("pending") or them.get("pending_cost"):
        problems.append("opponent pending plays")
    if state.get("projectiles"):
        problems.append(f"{len(state['projectiles'])} projectiles")
    kept = {int(u) for u in ((state.get("public") or {}).get("kept") or ())}
    private = [(name, i) for name, i in F.items() if name not in PUBLIC_FIELDS]
    for e in state.get("entities") or ():
        uid = _get(e, F["uid"], None)
        if uid in kept:
            continue
        status = _status(_get(e, F["status_flags"], 0))
        if int(_get(e, F["team"], team)) != team and int(_get(e, F["tower_slot"], -1)) < 0:
            if status is None or status & UNSEEN_STATUS:
                problems.append(f"enemy uid {uid} not on screen (status {status})")
        if status is not None and status & ~PUBLIC_STATUS:
            problems.append(f"uid {uid} status bits {status & ~PUBLIC_STATUS}")
        for name, i in private:
            if _get(e, i, ENTITY_DEFAULTS.get(name, 0)) != ENTITY_DEFAULTS.get(name, 0):
                problems.append(f"uid {uid} {name} {e[i]!r}")
    for sp in state.get("spells") or ():
        if (_get(sp, SF["aim_x"], None), _get(sp, SF["aim_y"], None)) != (
                _get(sp, SF["x"], None), _get(sp, SF["y"], None)):
            problems.append(f"spell card {_get(sp, SF['card_id'], None)} has an aim")
    return problems


# ---------------------------------------------------------------------------
# permanence
# ---------------------------------------------------------------------------

@dataclass
class _Memo:
    uid: int
    card: int
    row: list
    max_hp: int
    decay: float              # hit points a tick (max_hp / lifetime)
    hider: bool
    first_tick: int
    last_tick: int
    hp: int
    history: list             # (tick, hp) sightings, oldest first
    gone_at: int | None = None


class Permanence:
    """Enemy buildings seen and never seen destroyed (module doc, PERMANENCE). One per battle and
    observer; update / apply every decision, ticks in order.

    fields          royalesim.ENTITY_FIELDS (names or name -> index)
    team            the observer; the buildings kept are 1 - team's
    hiders          card ids of buildings that go under ground (card_ids(HIDER_NAMES, names))
    lifetimes       card id -> lifetime ticks (lifetimes_from_card_table); others the default
    spell_cards     our spell card ids (spell_ids): a spell of ours landing on a building that
                    then vanished destroyed it; empty: the rule is off
    """

    def __init__(self, fields: Mapping[str, int] | Sequence[str] | None = None, *, team: int,
                 hiders: Iterable[int] = (), lifetimes: Mapping[int, int] | None = None,
                 default_lifetime_ticks: int = DEFAULT_LIFETIME_TICKS,
                 grace_ticks: int = GRACE_TICKS, rate_window_ticks: int = RATE_WINDOW_TICKS,
                 merge_mt: float = MERGE_MT, spell_cards: Iterable[int] = (),
                 spell_hit_mt: float = SPELL_HIT_MT, spell_window_ticks: int = SPELL_WINDOW_TICKS,
                 sub: int = SUBTILE_PER_MILLITILE):
        self.F = _fields(fields)
        self.team = _team(team)
        self.enemy = 1 - self.team
        self.hiders = frozenset(int(c) for c in hiders)
        self.lifetimes = {int(k): int(v) for k, v in (lifetimes or {}).items() if int(v) > 0}
        if int(default_lifetime_ticks) <= 0:
            raise ValueError(f"default_lifetime_ticks must be > 0, got {default_lifetime_ticks}")
        self.default_lifetime = int(default_lifetime_ticks)
        self.grace_ticks = int(grace_ticks)
        self.rate_window = int(rate_window_ticks)
        self.sub = int(sub)
        self.merge_sub = float(merge_mt) * self.sub
        self.spell_cards = frozenset(int(c) for c in spell_cards)
        self.spell_hit_sub = float(spell_hit_mt) * self.sub
        self.spell_window = int(spell_window_ticks)
        self.stats: Counter = Counter()
        self.reset()

    def reset(self) -> None:
        """A new battle: everything forgotten (the stats are kept)."""
        self._memos: dict[int, _Memo] = {}
        self._reported: set[int] = set()
        self.tick: int | None = None

    def destroyed(self, uids: Iterable[int]) -> None:
        """Perception saw these buildings die: forget them (live)."""
        for uid in uids:
            uid = int(uid)
            self._reported.add(uid)
            if self._memos.pop(uid, None) is not None:
                self.stats["reported"] += 1

    # ------------------------------------------------------------ internals
    def lifetime(self, card: int) -> int:
        return self.lifetimes.get(int(card), self.default_lifetime)

    def _visible(self, view: Mapping[str, Any]) -> dict[int, list]:
        F = self.F
        kept = {int(u) for u in ((view.get("public") or {}).get("kept") or ())}
        out = {}
        for e in view.get("entities") or ():
            if (int(e[F["team"]]) != self.enemy or int(e[F["tower_slot"]]) >= 0
                    or int(e[F["kind"]]) != KIND_BUILDING):
                continue
            hp, uid = e[F["hp"]], int(e[F["uid"]])
            status = _status(e[F["status_flags"]])
            if hp is None or hp <= 0 or uid < 0 or uid in kept or uid in self._reported:
                continue
            if status is not None and status & UNSEEN_STATUS:
                continue                    # a row kept earlier, or a view that is not public
            out[uid] = e
        return out

    def _merge(self, e: Sequence[Any], visible: Mapping[int, Any]) -> None:
        """A building seen under a new uid replaces the nearest kept one of its card."""
        F = self.F
        x, y, card = e[F["x"]], e[F["y"]], int(e[F["card_id"]])
        best, best_d = None, None
        for uid, m in self._memos.items():
            if uid in visible or m.card != card:
                continue
            d = math.hypot(m.row[F["x"]] - x, m.row[F["y"]] - y)
            if d <= self.merge_sub and (best_d is None or d < best_d):
                best, best_d = uid, d
        if best is not None:
            del self._memos[best]
            self.stats["merged"] += 1

    def _remaining(self, m: _Memo, tick: int) -> float:
        return m.hp - m.decay * (tick - m.last_tick)

    def _judge(self, m: _Memo, tick: int, view: Mapping[str, Any]) -> str | None:
        """Why a building first missing at `tick` is gone, or None (kept)."""
        remaining = self._remaining(m, tick)
        if remaining <= 0:
            return "expired"
        drops = [h0 - h1 - m.decay * (t1 - t0)
                 for (t0, h0), (t1, h1) in zip(m.history, m.history[1:])
                 if m.last_tick - t1 <= self.rate_window]
        if drops and remaining <= max(drops):
            return "dying"
        if self.spell_cards:
            F = self.F
            bx, by = m.row[F["x"]], m.row[F["y"]]
            for s in (view.get("public") or {}).get("own_sent") or ():
                exec_tick, card, x, y = (int(v) for v in s[:4])
                if (card in self.spell_cards
                        and m.last_tick - self.spell_window <= exec_tick <= tick
                        and math.hypot(x - bx, y - by) <= self.spell_hit_sub):
                    return "spell"
        return None

    def _kept_row(self, m: _Memo, tick: int) -> list:
        F = self.F
        row = [list(v) if isinstance(v, list) else v for v in m.row]
        hp = int(round(self._remaining(m, tick)))
        row[F["hp"]] = max(1, min(m.max_hp, hp))
        row[F["deploy_ticks"]] = max(0, int(row[F["deploy_ticks"]] or 0) - (tick - m.last_tick))
        status = _status(row[F["status_flags"]]) or 0
        row[F["status_flags"]] = (status & PUBLIC_STATUS) | STATUS_HIDDEN
        return row

    # ------------------------------------------------------------ the calls
    def update(self, view: Mapping[str, Any]) -> list[list]:
        """Read the enemy buildings `view` (a public view) shows at its tick; return the rows of
        those kept out of sight, at that tick."""
        F = self.F
        tick = int(view["tick"])
        if self.tick is not None and tick < self.tick:
            self.reset()                    # the clock went back: a new battle
        self.tick = tick
        visible = self._visible(view)
        for uid, e in visible.items():
            m = self._memos.get(uid)
            hp = int(e[F["hp"]])
            if m is None:
                self._merge(e, visible)
                card = int(e[F["card_id"]])
                max_hp = max(1, int(e[F["max_hp"]] or 0), hp)
                m = self._memos[uid] = _Memo(
                    uid=uid, card=card, row=list(e), max_hp=max_hp,
                    decay=max_hp / self.lifetime(card), hider=card in self.hiders,
                    first_tick=tick, last_tick=tick, hp=hp, history=[(tick, hp)])
                continue
            if m.history and m.history[-1][0] == tick:
                m.history[-1] = (tick, hp)
            else:
                m.history.append((tick, hp))
                del m.history[:-HISTORY]
            m.row, m.hp, m.last_tick, m.gone_at = list(e), hp, tick, None
            m.max_hp = max(m.max_hp, hp)
        kept = []
        for uid in sorted(self._memos):
            if uid in visible:
                continue
            m = self._memos[uid]
            why = None
            if m.gone_at is None:
                m.gone_at = tick
                why = self._judge(m, tick, view)
            if why is None and self._remaining(m, tick) <= 0:
                why = "expired"
            if why is None and not m.hider and tick - m.last_tick > self.grace_ticks:
                why = "gone"
            if why is not None:
                del self._memos[uid]
                self.stats[why] += 1
                continue
            kept.append(self._kept_row(m, tick))
        return kept

    def apply(self, view: Mapping[str, Any]) -> dict:
        """update(view), and a copy of `view` with the kept rows after its entities and their
        uids in view["public"]["kept"] (a view applied before loses its old kept rows first, so
        applying twice gives one row per building)."""
        kept = self.update(view)
        F = self.F
        old = {int(u) for u in ((view.get("public") or {}).get("kept") or ())}
        out = dict(view)
        out["entities"] = [e for e in view.get("entities") or ()
                           if int(e[F["uid"]]) not in old] + kept
        pub = dict(view.get("public") or {})
        pub["kept"] = [int(r[F["uid"]]) for r in kept]
        out["public"] = pub
        self.stats["kept_rows"] += len(kept)
        return out

    @property
    def memos(self) -> dict[int, dict[str, Any]]:
        """What is remembered, by uid (for logs and tests)."""
        return {uid: {"card": m.card, "hp": m.hp, "last_tick": m.last_tick,
                      "gone_at": m.gone_at, "hider": m.hider} for uid, m in self._memos.items()}
