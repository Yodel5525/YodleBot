"""The refused-tap guard (FIX E): notices a bot tap that the game did not take.

WHY. A tap on a tile where the card may not go does nothing in the game: the card stays in
its slot, no elixir is spent, nothing is deployed. The live runner never looked at whether a
tap was taken. The pending entry (rs_decide.PENDING_TICKS) expired after 3 s without a log
line, and the same row and legality picked the same card and tile again. On the 2026-10-31
friendly (the enemy towers misread as destroyed) the bot would have tapped Ice Golem on an
enemy-half tile every ~3 s. This module watches each tap the runner sends, from the HUD that
perception already reads:

    landed     the card's slot reads away from the card (empty, or another card) for
               LAND_AWAY_TICKS ticks ("hand"); or the elixir drops by the card's cost on
               ELIXIR_READS reads in a row while no other slot reads empty ("elixir"); or the
               event deriver matched a unit to the tap (an OwnLanding with a land tick, "board")
    refused    REFUSE_AFTER_TICKS after the tap the card was read in its slot on at least
               MIN_PRESENT_TICKS distinct ticks, within the last PRESENT_TAIL_TICKS, with no
               away run in progress (one in progress defers the verdict until it ends)
    retracted  a refused tap whose hand or board evidence arrives before FINAL_AFTER_TICKS (a
               late landing): its mask is removed and it is not counted. Elixir evidence
               alone never retracts
    final      a refused tap at FINAL_AFTER_TICKS (= rs_decide.PENDING_TICKS =
               public_state.TAP_MAX_AGE: until then the pending entry still blocks the
               slot, so a late landing retracts the refusal before the mask could have
               changed any decision), or one whose slot was tapped again. Only finals
               count toward k
    unconfirmed  too few reads of the card to say (fewer than MIN_PRESENT_TICKS, or none
               recent) at FINAL_AFTER_TICKS: no mask, not counted

A tap's age is min(ticks since the tap, capture seconds since the tap * 20 + 2): a clock
misread that jumps forward cannot make every watch final at once.

WHAT IT DOES. It never taps and never sets AUTO. With `enforce` (the runner has a tapper) it
narrows legality and the runner switches AUTO to MANUAL after k final refusals in a battle:
    - a refused card is masked where it was refused: on the enemy half (own-frame ty <=
      ENEMY_MAX_TY) the whole enemy half of that lane, for the rest of the battle (a troop
      goes there only once that lane's princess tower is down, and the tower filter never
      raises a tower again, so the bad read lasts the battle); in our half and on the river
      a 5 x 5 box (MASK_BOX_RADIUS 2) for MASK_OWN_TICKS (600 ticks, a Cannon's 30 s);
    - while a real watch is open its card is masked at every tile (a hand misread can drop
      the pending entry early; this keeps the card from being tapped again meanwhile);
    - k = REFUSALS_TO_MANUAL final refusals in a battle -> one ("tap_guard_manual", ...)
      event; the runner switches to MANUAL. rearm() (the owner pressed F8 back to AUTO)
      starts the count again and keeps the masks; new_battle() clears everything.
Without `enforce` (a dry run, a replay) masked() is always False, nothing switches and every
event says dry: the AUTO(dry) plays get virtual watches only to log what the guard would do.

THE SELECTED-CARD HAZARD (toggle_suspect). A refused tap probably leaves its card selected
(raised): the runner's card tap selected it and its tile tap did nothing. Tapping that slot
again then most likely deselects the card, and that tap's tile tap does nothing for another
reason than the tile. So a tap on the slot of the previous refusal, with no other tap sent in
between, is toggle_suspect: refused, it is counted but not masked, and the remembered
selection is cleared. Unverified (docs/SPARRING_RUNBOOK.md 18).

ATTRIBUTION. A drop of elixir credits at most one watch: an open watch whose slot reads away
first, else the oldest that fits; the other open watches start their running maximum again.
A drop seen while another slot reads empty belongs to that slot's play (every deploy in the
2026-10-31 record had its slot empty on the drop frame): it is not evidence for any other
watch, and those watches' running maximum starts again from that frame, so the drop is not
counted for them later either, once the slot has its refill.

A frame on which no slot shows a card (a failed hand read, a covered panel) is skipped for
the hand and elixir evidence: it is not an away reading.

Standard library only. Tiles are the model's own-frame tiles (rs_decide Decision.tx / .ty).
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from scripts.vision.types import EMPTY_CARD

#: Ticks after the tap before a card still in its slot is refused. A deploy empties the slot
#: on the frame the elixir drops; the tap (two `input tap` lines) has not been timed, probably
#: 0.3-0.6 s: 30 ticks leave about 1 s of margin, and a later landing is retracted.
REFUSE_AFTER_TICKS = 30
#: A refusal becomes final here: rs_decide.PENDING_TICKS = public_state.TAP_MAX_AGE (tested).
FINAL_AFTER_TICKS = 60
#: The slot reads away from the card this many ticks in a row: the card left (a landing). HUD
#: misreads lasted 1-3 ticks on 2026-10-31; a landed slot read away for 11 ticks or more.
LAND_AWAY_TICKS = 6
#: A refusal needs the card read in its slot within the last this many ticks.
PRESENT_TAIL_TICKS = 3
#: ... and on at least this many distinct ticks since the tap; else tap_unconfirmed at FINAL.
MIN_PRESENT_TICKS = 10
#: Elixir evidence: running max since the tap - reading >= cost * 1000 - this, ...
ELIXIR_TOL_MILLI = 500
#: ... on this many reads in a row.
ELIXIR_READS = 2
#: Masks: own half and river rows, a box of this radius for this long.
MASK_OWN_TICKS = 600
MASK_BOX_RADIUS = 2
#: Final refusals in a battle before the runner switches to MANUAL.
REFUSALS_TO_MANUAL = 3
#: Own-frame rows 0..14 are the enemy half (15-16 the river, 17-31 ours); columns 0..8 and
#: 9..17 are the two lanes (own frame).
ENEMY_MAX_TY = 14
LANE_SPLIT_TX = 9
GRID_W, GRID_H = 18, 32
#: The age's capture-clock cap: seconds * TICKS_PER_S + AGE_SLACK_TICKS.
TICKS_PER_S = 20
AGE_SLACK_TICKS = 2
#: An open watch with an away run still in progress at this age gets no verdict at all.
GIVE_UP_TICKS = 2 * FINAL_AFTER_TICKS

OPEN, LANDED, REFUSED, RETRACTED, FINAL, UNCONFIRMED = (
    "open", "landed", "refused", "retracted", "final", "unconfirmed")
HAND, ELIXIR, BOARD = "hand", "elixir", "board"
LANE, BOX = "lane", "box"
SUMMARY_KEYS = ("watched", "landed", "refused", "final", "retracted", "toggle", "unconfirmed",
                "switched", "masks")


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RefusalMask:
    """`card` may not go on own-frame tiles x0..x1, y0..y1 (inclusive) from since_tick until
    until_tick (None: the rest of the battle). kind LANE (the enemy half of a lane) or BOX."""

    card: int
    x0: int
    y0: int
    x1: int
    y1: int
    kind: str
    since_tick: int
    until_tick: int | None
    tap_tick: int

    def live(self, tick: int | None) -> bool:
        return self.until_tick is None or tick is None or int(tick) < self.until_tick

    def covers(self, card: int, tx: int, ty: int, tick: int | None) -> bool:
        return (int(card) == self.card and self.x0 <= int(tx) <= self.x1
                and self.y0 <= int(ty) <= self.y1 and self.live(tick))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1,
                "until_tick": self.until_tick}


def mask_for(card: int, tx: int, ty: int, tick: int, tap_tick: int, *,
             own_ticks: int = MASK_OWN_TICKS, radius: int = MASK_BOX_RADIUS) -> RefusalMask:
    """The mask of a refusal at own-frame tile (tx, ty), decided at `tick`: the enemy half of
    that lane for the battle (ty <= ENEMY_MAX_TY), else a box of `radius` for `own_ticks`."""
    tx, ty = int(tx), int(ty)
    if ty <= ENEMY_MAX_TY:
        x0, x1 = (0, LANE_SPLIT_TX - 1) if tx < LANE_SPLIT_TX else (LANE_SPLIT_TX, GRID_W - 1)
        return RefusalMask(int(card), x0, 0, x1, ENEMY_MAX_TY, LANE, int(tick), None, int(tap_tick))
    r = int(radius)
    return RefusalMask(int(card), max(0, tx - r), max(0, ty - r), min(GRID_W - 1, tx + r),
                       min(GRID_H - 1, ty + r), BOX, int(tick), int(tick) + int(own_ticks),
                       int(tap_tick))


def zone_of(tx: int, ty: int, screen_tile: Sequence[int] | None = None) -> str:
    """"own" (our half and the river rows), else "enemy_left" / "enemy_right" by the SCREEN
    side, as the HUD's tower keys name them: screen_tile's x, or, without it, the live bot's
    mirror of the own-frame column (screen x = 17 - tx, team 1)."""
    if int(ty) > ENEMY_MAX_TY:
        return "own"
    x = int(screen_tile[0]) if screen_tile is not None else GRID_W - 1 - int(tx)
    return "enemy_left" if x < LANE_SPLIT_TX else "enemy_right"


# ---------------------------------------------------------------------------
# watches
# ---------------------------------------------------------------------------
@dataclass
class TapWatch:
    """One tap the runner sent (or, `virtual`, a dry run's play it would have sent)."""

    card: int
    slot: int
    cost: int
    tap_tick: int
    tap_t: float | None
    tx: int
    ty: int
    screen_tile: tuple[int, int] | None = None
    elixir_milli: int | None = None
    spendable_milli: int | None = None
    virtual: bool = False
    toggle_suspect: bool = False
    seq: int = 0
    state: str = OPEN
    present: set[int] = field(default_factory=set)
    last_present: int | None = None
    reads: int = 0
    away_start: int | None = None
    away_max: int = 0
    e_max: int | None = None
    e_now: int | None = None
    drop_reads: int = 0
    evidence: str | None = None
    hand_lag: int | None = None
    elixir_lag: int | None = None
    verdict_tick: int | None = None
    refused_age: int | None = None
    mask: RefusalMask | None = None

    def age(self, tick: int, t: float | None = None) -> int:
        """Ticks since the tap, capped by the capture clock (module doc)."""
        a = int(tick) - self.tap_tick
        if t is not None and self.tap_t is not None:
            a = min(a, int(round((float(t) - self.tap_t) * TICKS_PER_S)) + AGE_SLACK_TICKS)
        return a

    @property
    def live(self) -> bool:
        return self.state in (OPEN, REFUSED)


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------
class RefusedTapGuard:
    """Watches the taps sent and judges them from the HUD (module doc).

    sent(...) after each tap (virtual=True for a dry run's play); observe(...) on every
    battle-screen frame with a tick, before the decision: it returns [(kind, data)] events,
    kinds tap_landed, tap_refused, tap_refusal_retracted, tap_refusal_final, tap_unconfirmed and
    tap_guard_manual (the runner switches to MANUAL on it when `enforce`). masked(card, tx, ty,
    tick) for legality (always False without `enforce`), active(tick) its fast check.
    """

    def __init__(self, *, enforce: bool = False, k: int = REFUSALS_TO_MANUAL,
                 refuse_after: int = REFUSE_AFTER_TICKS, final_after: int = FINAL_AFTER_TICKS,
                 land_away: int = LAND_AWAY_TICKS, present_tail: int = PRESENT_TAIL_TICKS,
                 min_present: int = MIN_PRESENT_TICKS, elixir_tol: int = ELIXIR_TOL_MILLI,
                 elixir_reads: int = ELIXIR_READS, mask_own_ticks: int = MASK_OWN_TICKS,
                 mask_radius: int = MASK_BOX_RADIUS, give_up: int = GIVE_UP_TICKS):
        self.enforce = bool(enforce)
        self.k = max(1, int(k))
        self.refuse_after = int(refuse_after)
        self.final_after = int(final_after)
        self.land_away = max(1, int(land_away))
        self.present_tail = max(1, int(present_tail))
        self.min_present = int(min_present)
        self.elixir_tol = int(elixir_tol)
        self.elixir_reads = max(1, int(elixir_reads))
        self.mask_own_ticks = int(mask_own_ticks)
        self.mask_radius = int(mask_radius)
        self.give_up = int(give_up)
        self.totals: Counter = Counter()
        self.all_hand_lags: list[int] = []
        self.new_battle()

    # -- state ------------------------------------------------------------------
    def new_battle(self) -> None:
        """A new battle: no watches, masks, counts or remembered selection."""
        self.watches: list[TapWatch] = []
        self.masks: list[RefusalMask] = []
        self.finals: list[TapWatch] = []          # counted toward k since the battle / rearm
        self.armed = True
        self.refusals = 0                         # tap_refused verdicts this battle
        self.selected_slot: int | None = None
        self.tick: int | None = None
        self.counts: Counter = Counter()
        self.hand_lags: list[int] = []
        self._seq = 0
        self._queue: list[tuple[str, dict[str, Any]]] = []

    def rearm(self, tick: int | None = None) -> None:
        """The owner handed back to AUTO after a switch: count k again; the masks stay."""
        self.finals = []
        self.armed = True
        self._count("rearmed")

    def mark_switched(self) -> None:
        """The runner switched to MANUAL on a tap_guard_manual event."""
        self._count("switched")

    def _count(self, key: str, n: int = 1) -> None:
        self.counts[key] += n
        self.totals[key] += n

    @property
    def open_watches(self) -> list[TapWatch]:
        return [w for w in self.watches if w.state == OPEN]

    # -- taps -------------------------------------------------------------------
    def sent(self, card: int, slot: int, cost: int, tick: int, t: float | None = None, *,
             tile: Sequence[int], screen_tile: Sequence[int] | None = None,
             elixir_milli: int | None = None, spendable_milli: int | None = None,
             virtual: bool = False) -> TapWatch | None:
        """A tap of `card` from hand `slot` on own-frame `tile` was sent at `tick` (capture
        time `t`). A virtual watch (a dry run's play) is skipped while a watch is open on that
        slot. A refused watch on the same slot becomes final at once (counted; its events
        come with the next observe())."""
        card, slot, tick = int(card), int(slot), int(tick)
        if card < 0 or slot < 0:
            return None
        if virtual and any(w.slot == slot and w.state == OPEN for w in self.watches):
            return None
        for w in self.watches:
            if w.slot == slot and w.state == REFUSED:
                self._queue += self._final(w, tick)
        self.watches = [w for w in self.watches if w.live]
        self._seq += 1
        toggle = self.selected_slot == slot
        self.selected_slot = None                 # this tap (de)selects: no selection known
        w = TapWatch(card, slot, max(0, int(cost or 0)), tick, None if t is None else float(t),
                     int(tile[0]), int(tile[1]),
                     None if screen_tile is None else (int(screen_tile[0]), int(screen_tile[1])),
                     None if elixir_milli is None else int(elixir_milli),
                     None if spendable_milli is None else int(spendable_milli),
                     bool(virtual), toggle, self._seq,
                     e_max=None if elixir_milli is None else int(elixir_milli))
        self.watches.append(w)
        self._count("watched")
        return w

    # -- legality ---------------------------------------------------------------
    def masked(self, card: int, tx: int, ty: int, tick: int | None = None) -> bool:
        """May `card` not go on own-frame tile (tx, ty) now? Always False without enforce."""
        if not self.enforce:
            return False
        card = int(card)
        tick = self.tick if tick is None else int(tick)
        for w in self.watches:
            if w.state == OPEN and not w.virtual and w.card == card:
                return True                       # in flight: the card waits for its verdict
        return any(m.covers(card, tx, ty, tick) for m in self.masks)

    def active(self, tick: int | None = None) -> bool:
        """Could masked() say True now (enforce, and a mask or a real open watch)?"""
        if not self.enforce:
            return False
        tick = self.tick if tick is None else int(tick)
        return (any(w.state == OPEN and not w.virtual for w in self.watches)
                or any(m.live(tick) for m in self.masks))

    # -- frames -----------------------------------------------------------------
    def observe(self, tick: int | None, t: float | None, hand: Sequence[int] | None,
                elixir_milli: int | None, landings: Iterable[Any] = ()) -> list[tuple[str, dict]]:
        """One battle-screen frame: its tick, capture time, the HUD's hand and elixir and the
        deriver's own landings. Returns the events (kind, data)."""
        out, self._queue = self._queue, []
        if tick is None:
            return out
        tick = int(tick)
        t = None if t is None else float(t)
        self.tick = tick
        self.masks = [m for m in self.masks if m.live(tick)]
        live = [w for w in self.watches if w.live]
        if not live:
            return out
        # 1. the board: a unit matched to the tap
        for x in landings or ():
            land, slot = getattr(x, "land_tick", None), getattr(x, "slot", -1)
            if land is None or slot is None or int(slot) < 0:
                continue
            key = (int(x.card), int(slot), int(x.tap_tick))
            for w in live:
                if w.live and not w.virtual and (w.card, w.slot, w.tap_tick) == key:
                    out += self._landed(w, tick, t, BOARD, int(land) - w.tap_tick)
        # 2. the hand
        hand = [EMPTY_CARD if c is None else int(c) for c in (hand or ())]
        hand_read = any(c >= 0 for c in hand)
        if hand_read:
            for w in live:
                if not w.live or tick <= w.tap_tick:
                    continue
                w.reads += 1
                if w.slot < len(hand) and hand[w.slot] == w.card:
                    w.present.add(tick)
                    w.last_present = tick
                    w.away_start = None
                    continue
                if w.away_start is None:
                    w.away_start = tick
                run = tick - w.away_start + 1
                w.away_max = max(w.away_max, run)
                if run >= self.land_away:
                    out += self._landed(w, tick, t, HAND, w.away_start - w.tap_tick)
        # 3. the elixir (open watches only: it never retracts)
        if elixir_milli is not None and hand_read:
            out += self._elixir(live, tick, t, int(elixir_milli), hand)
        # 4. verdicts
        for w in live:
            if not w.live:
                continue
            age = w.age(tick, t)
            if w.state == OPEN:
                if age < self.refuse_after:
                    continue
                if w.away_start is not None:      # an away run in progress: wait for its end
                    if age >= self.give_up:
                        out += self._unconfirmed(w, tick)
                    continue
                recent = w.last_present is not None and tick - w.last_present < self.present_tail
                if len(w.present) >= self.min_present and recent:
                    out += self._refuse(w, tick, age)
                elif age >= self.final_after:
                    out += self._unconfirmed(w, tick)
                    continue
                else:
                    continue
            if w.state == REFUSED and age >= self.final_after:
                out += self._final(w, tick)
        self.watches = [w for w in self.watches if w.live]
        return out

    def _elixir(self, live: list[TapWatch], tick: int, t: float | None, e: int,
                hand: Sequence[int]) -> list[tuple[str, dict]]:
        fits = []
        for w in live:
            if w.state != OPEN or tick <= w.tap_tick:
                continue
            w.e_now = e
            if w.e_max is None or e > w.e_max:
                w.e_max = e
            if any(c == EMPTY_CARD for s, c in enumerate(hand) if s != w.slot):
                w.e_max, w.drop_reads = e, 0      # another slot's play: its drop, not ours
                continue
            if w.cost > 0 and w.e_max - e >= w.cost * 1000 - self.elixir_tol:
                w.drop_reads += 1
                if w.drop_reads >= self.elixir_reads:
                    fits.append(w)
            else:
                w.drop_reads = 0
        if not fits:
            return []
        away = [w for w in fits if w.away_start is not None]
        won = away[0] if away else min(fits, key=lambda w: (w.tap_tick, w.seq))
        out = self._landed(won, tick, t, ELIXIR, tick - won.tap_tick)
        for w in live:                            # one drop credits one watch
            if w is not won and w.state == OPEN:
                w.e_max, w.drop_reads = e, 0
        return out

    # -- verdicts -----------------------------------------------------------------
    def _dry(self, w: TapWatch) -> bool:
        return w.virtual or not self.enforce

    def _landed(self, w: TapWatch, tick: int, t: float | None, evidence: str,
                lag: int) -> list[tuple[str, dict]]:
        if w.state == REFUSED:
            if evidence == ELIXIR or w.age(tick, t) >= self.final_after:
                return []
            self._lag(evidence, lag)
            w.state, w.evidence = RETRACTED, evidence
            if w.mask is not None and w.mask in self.masks:
                self.masks.remove(w.mask)
            if self.selected_slot == w.slot and w.seq == self._seq:
                self.selected_slot = None         # it landed: nothing stays selected
            self._count("retracted")
            return [("tap_refusal_retracted", {"tick": tick, "tap_tick": w.tap_tick, "card": w.card,
                                               "slot": w.slot, "evidence": evidence, "lag": int(lag),
                                               "dry": self._dry(w)})]
        if w.state != OPEN:
            return []
        self._lag(evidence, lag)
        w.state, w.evidence = LANDED, evidence
        if evidence == HAND:
            w.hand_lag = int(lag)
        elif evidence == ELIXIR:
            w.elixir_lag = int(lag)
        self._count("landed")
        return [("tap_landed", {"tick": tick, "tap_tick": w.tap_tick, "card": w.card, "slot": w.slot,
                                "evidence": evidence, "hand_lag": w.hand_lag,
                                "elixir_lag": w.elixir_lag,
                                **({"board_lag": int(lag)} if evidence == BOARD else {}),
                                "dry": self._dry(w)})]

    def _lag(self, evidence: str, lag: int) -> None:
        """A hand landing's lag (the slot's first away tick - the tap tick): the tap-to-deploy
        latency the first --tap session measures (docs/SPARRING_RUNBOOK.md 18)."""
        if evidence == HAND:
            self.hand_lags.append(int(lag))
            self.all_hand_lags.append(int(lag))

    def _refuse(self, w: TapWatch, tick: int, age: int) -> list[tuple[str, dict]]:
        w.state, w.verdict_tick, w.refused_age = REFUSED, tick, int(age)
        self.refusals += 1
        self._count("refused")
        if w.toggle_suspect:
            self._count("toggle")
        else:
            w.mask = mask_for(w.card, w.tx, w.ty, tick, w.tap_tick, own_ticks=self.mask_own_ticks,
                              radius=self.mask_radius)
            self._count("masks")
            if not w.virtual:
                self.masks.append(w.mask)
            if w.seq == self._seq:                # no tap sent since: its card stays selected
                self.selected_slot = w.slot
        return [("tap_refused", {
            "tick": tick, "tap_tick": w.tap_tick, "age": int(age), "card": w.card, "slot": w.slot,
            "cost": w.cost, "model_tile": [w.tx, w.ty],
            "screen_tile": None if w.screen_tile is None else list(w.screen_tile),
            "zone": zone_of(w.tx, w.ty, w.screen_tile),
            "mask": None if w.mask is None else w.mask.to_dict(),
            "toggle_suspect": w.toggle_suspect, "present_ticks": len(w.present), "reads": w.reads,
            "away_max": w.away_max, "elixir_at_tap": w.elixir_milli, "elixir_now": w.e_now,
            "elixir_max": w.e_max, "spendable_at_tap": w.spendable_milli,
            "refusals": self.refusals, "dry": self._dry(w),
            "applied": bool(self.enforce and not w.virtual and w.mask is not None)})]

    def _final(self, w: TapWatch, tick: int) -> list[tuple[str, dict]]:
        w.state = FINAL
        self._count("final")
        self.finals.append(w)
        out = [("tap_refusal_final", {"tick": tick, "tap_tick": w.tap_tick, "card": w.card,
                                      "slot": w.slot, "toggle_suspect": w.toggle_suspect,
                                      "finals": len(self.finals), "k": self.k, "dry": self._dry(w)})]
        if self.armed and len(self.finals) >= self.k:
            self.armed = False
            self._count("manual")
            out.append(("tap_guard_manual", {
                "tick": tick, "k": self.k,
                "finals": [{"card": f.card, "slot": f.slot, "tap_tick": f.tap_tick,
                            "tile": [f.tx, f.ty],
                            "screen_tile": None if f.screen_tile is None else list(f.screen_tile),
                            "toggle_suspect": f.toggle_suspect} for f in self.finals],
                "dry": not self.enforce or all(f.virtual for f in self.finals)}))
        return out

    def _unconfirmed(self, w: TapWatch, tick: int) -> list[tuple[str, dict]]:
        w.state = UNCONFIRMED
        self._count("unconfirmed")
        return [("tap_unconfirmed", {"tick": tick, "tap_tick": w.tap_tick, "card": w.card,
                                     "slot": w.slot, "reads": w.reads,
                                     "present_ticks": len(w.present), "dry": self._dry(w)})]

    # -- reports ----------------------------------------------------------------
    def summary(self, *, session: bool = False) -> dict[str, Any]:
        """This battle's counts (session: the whole session's), for battle_end / session_end:
        watched, landed, refused, final, retracted, toggle, unconfirmed, switched, masks (made),
        and the hand lag (first away tick - tap tick: the tap-to-deploy latency) median / max."""
        c = self.totals if session else self.counts
        lags = self.all_hand_lags if session else self.hand_lags
        out: dict[str, Any] = {k: int(c.get(k, 0)) for k in SUMMARY_KEYS}
        out["hand_lag_median"] = statistics.median(lags) if lags else None
        out["hand_lag_max"] = max(lags) if lags else None
        if not session:
            out["open"] = len(self.open_watches)
            out["masks_live"] = len(self.masks)
        return out

    def config(self) -> dict[str, Any]:
        """The constants and enforce, for the config record."""
        return {"enforce": self.enforce, "k": self.k, "refuse_after_ticks": self.refuse_after,
                "final_after_ticks": self.final_after, "land_away_ticks": self.land_away,
                "present_tail_ticks": self.present_tail, "min_present_ticks": self.min_present,
                "elixir_tol_milli": self.elixir_tol, "elixir_reads": self.elixir_reads,
                "mask_own_ticks": self.mask_own_ticks, "mask_radius": self.mask_radius,
                "enemy_max_ty": ENEMY_MAX_TY, "give_up_ticks": self.give_up}
