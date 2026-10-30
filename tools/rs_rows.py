"""Training rows from IL_Replay re-driven through RoyaleSim, in the same column
format as sim/replay_rows.py, so sim/train_tokens.py trains on either.

    .venvs/royalegym/Scripts/python.exe tools/rs_rows.py --others 20000 --workers 7

Why a second generator: on the same 2,000 matches RoyaleSim's endings agree
with the recording far more often than ours (winner 62.6% vs 49.5%, exact
crowns 49.0% vs 24.2%; reports/redrive_fidelity.md), and it refuses almost no
recorded play, so the boards a row shows are much closer to what the player
really saw.

Frame: every row is from the acting player's view, as the bottom player with
y growing toward its own king (the same as sim/replay_rows.py). RoyaleSim's
blue owns low y, so blue is reflected in y and red in x; both are reflections,
so left and right mean the same thing to either seat.

Seat: tools/rs_orient.py decides from troop plays which IL_Replay side owns native
low y; that side plays blue, and every coordinate goes to the engine as recorded.
A match where either side has under 90% of its troop plays on its own half is
skipped as "orient" (the old median test plus half turn cancelled out and checked
nothing). Matches with the team at native low y give the same rows as before; the
one exception, a team whose median play (spells included) was past the river, used
to be put in red and turned, and now plays blue with the same own-frame rows.

Ids: units are RoyaleSim `unit_type` + 2, cards are catalogue ids + 2 (0 pad,
1 unknown). `data/rs_rows/vocab.json` records both lists, and the engine config
and fingerprint they came from (tools/rs_engine.py), the card table, the row
schema, the observation delay and wait_keep_at_build (vocab_record).
tools/rs_checkpoint.py refuses a checkpoint whose lists or card table are not
the engine's it is loaded into.

Engine: every Battle comes from rs_engine.make_battle under data/engine.json
(card table, levels), with no command delay: plays are sent at their recorded
ticks. `--card-table default` (or HASTYCR_CARD_TABLE=default) runs the engine's
own table, as this tool did before rs_engine.

Cards: IL_Replay card keys map to RoyaleSim names through tools/rs_cards.py
(data/cards/: royaleimitate 0.2.15's replay_cards.json, vendored and hash-checked,
plus void, rune-giant and spirit-empress). Matches with an unmapped card are
skipped. HASTYCR_CARD_EXTRAS=0 gives the 119-key map this tool used before.
vocab.json records the map under "card_map".

Outcome and trust as in sim/replay_rows.py: tower swing over the next 10 s;
rows after the first refused play, or after the engine has taken more crowns
than the recording allows, are marked untrusted.

Opponent memory (E08; opp_seen, opp_last, opp_last_age): tools/rs_memory.PublicMemory, every
play dated on the tick it ran (the recorded IL_Replay tick). `--mem-lag L` (default BOARD_LAG,
1): all rows of a tick read one view, taken before that tick's plays are recorded, holding the
plays with exec_tick + L <= tick. A seat no longer sees the other seat's play of the same tick
(tokens-17); 0 and 1 give the same rows here, and a larger L stands for perception lag.
`--mem-lag legacy` keeps the old order (each play recorded as it is handled, the memory read at
each row) and gives rows byte-identical to the rows before E08. vocab.json records the lag as
obs_delay.opp_lag and the order as "opp_memory": "exec"; legacy rows write opp_lag 0 and no
"opp_memory", as before (memory_vocab).

Units (E16): `--units types` (the default) writes RoyaleSim unit_type + 2 and the engine's
flags, as before. `--units vision` writes the class of data/vision/classes.json
(tools/vision_classes.py) + 2, flying and building from the class table (what a screen can
know), and each unit's form (0 base, 0.5 evo, 1 hero) in one more column, "unit_form"
(MAX_UNITS float16). sim/train_tokens.py does not read that column yet: the legacy model keeps
UF = 7 features, and the v2 schema (E14) moves the form into unit_feats. vocab.json then lists
the classes as "units" (n_units = classes + 2) and adds "units_mode": "vision",
"vision_classes_digest" and "extra_columns". --live-compat is a third kind of unit token and
does not combine with --units vision.

Visibility (E12): `--visibility legacy` (the default) shows every living unit, as before, so
the rows and vocab.json are byte-identical to the rows before E12. `--visibility public` shows
the acting player what their screen shows: an ENEMY unit with STATUS_INVISIBLE (a Royal Ghost,
a cloaked Archer Queen) or STATUS_HIDDEN (a Tesla in the ground) is left out before the sort
and the MAX_UNITS cut, so the units shown take its place; a unit tunnelling now
(STATUS_UNDERGROUND: a Miner, a Goblin Drill on its way) stays where the engine has it, the
head of its dirt trail, with 1 in one more column, "unit_burrow" (MAX_UNITS float16); own units
are always shown. The bits (royalegym protocol.py:579-583) are read from status_flags through
the field index (rs_engine.field_index), and an engine's -1, "not reported", is refused
(VisibilityError), never masked (masked, -1 would be every bit at once). When royalesim exports
STATUS_BITS, its first three names must be underground, invisible, hidden (check_status_bits).
sim/train_tokens.py does not read unit_burrow yet (E14 moves it into unit_feats). vocab.json
then adds "visibility": "public", "visibility_rule" and "unit_burrow" in "extra_columns"
(visibility_vocab); tools/rs_eval.py reads "visibility" back from a checkpoint's meta.

Trust v2 (E09): `--trust legacy` (the default) writes `trusted` only, so the rows and
vocab.json are byte-identical to the rows before E09. `--trust v2` keeps `trusted` as it is
and adds a `trusted_v2` column. A row is trusted_v2 when its tick + HORIZON is at or before the
match's first divergence and its tick is at most WAIT_TAIL (20) ticks after the match's last
recorded play (IL_Replay records no end tick, so the wait rows the engine adds after the
recording stopped are not trusted; tokens-16). The first divergence is the earliest of:
the legacy cut (more crowns than the recording, a card not in hand, a refused step); the first
sampled tick where an engine tower has left the recording's final_tower_hitpoints
(sim/replay_corpus._towers): destroyed while the recording ends with it standing
("tower_lost"), or damaged more than the recording by more than --trust-hp-margin (0.10) of its
max hp ("tower_hp"; towers never heal, so the final hp bounds every tick, and the recorded
damage is scaled to a recorded tower level above the engine's, level_scale); and the tick the
engine ended the game while a recorded play was still to come ("game_over"). Which engine
princess a recorded princess_left is has not been verified, so `--trust-sides unordered` (the
default) pairs a side's two princesses in the way most favourable to the engine; `world`
(princess_left is the world-left tower, slot 1, of both seats) and `own` (the owner's own left:
slot 2 for blue, slot 1 for red) pair them by side. main writes <out>/manifest.json: rows,
trusted (and under v2 trusted_v2) counts and rates for all rows and the Hog 2.6 rows, per
match why trust was cut, one entry per shard and, under v2, princess_votes: per seat, the
matches where one princess fell both in the engine and in the recording, "slot1" when it is
the same tower if princess_left is slot 1, else "slot2" (world: slot1 for both seats; own:
blue slot2, red slot1). vocab.json under v2 adds "trust" (TrustRule.to_json) and "trusted_v2"
in "extra_columns" (trust_vocab). sim/train_tokens.py still filters on `trusted`.

Presses (E10; tools/rs_presses.py): `--presses off` (the default) replays play_card events only,
so the rows, vocab.json and manifest.json are byte-identical to the rows before E10. `--presses
on` also replays every activate_ability event: its card ("card_key", else the candidates)
picks one of the side's ability buttons (heroes, form 2, in dealt-deck order, then champions;
the engine's own `abilities` order when it names the cards), and the command (team, HAND_SIZE
+ button, 0, 0) goes in with that tick's plays. Each press gets a row of its own, observed at
its tick like a play row, with action (0, 0, 0) and "press" = button + 1 (shard column
act_press; 0 on every other row); the side has no wait row on that tick, and no wait row in
the 10 ticks before a press. A refused press (NO_HERO, ABILITY_NOT_READY, ABILITY_SPENT, short of
elixir, ...) and a press whose card is no button of the side ("unknown") or several
("ambiguous") cut `trusted` at its tick, reason CUT_PRESS ("press"); a skipped press has no
row. The match's last recorded action (play or press) sets the end of the run and, under
--trust v2, the wait tail. Every row also has "ability_card" (B int16: each button's card id +
2, 0 none) and "ability_state" (B x 5 float16: reported, available, spent, cost / 10, cooldown
/ 400, from players[t]["abilities"] when the engine reports them, else 0); B is
royalesim.ABILITY_BUTTONS (fallback 2). sim/train_tokens.py reads none of them yet (E15 adds the
press head): a press row's act_card is 0, so it trains as a wait. rows_for's info gets
"presses" (rs_presses.PressLog counts), and manifest.json sums them under "presses" with
press_rows. vocab.json adds "presses" (rs_presses.vocab_entry) and the three columns in
"extra_columns" (press_vocab).

Decision time (E11; tokens-01/05/10): `--obs-delay 0` (the default) observes each play at the
tick it ran, the recorded tick, so the rows, vocab.json and manifest.json are byte-identical to
the rows before E11. `--obs-delay D --obs-delay-jitter J` observes each play where its player
tapped it: tap = exec - D_i, D_i = D + an integer drawn uniformly from [-J, J], floored at tick
0 (tap_delays: one draw per play in queue order from numpy default_rng([tap_seed(tag), 0]), the
seed the first 16 hex digits of sha1(replay tag); presses, under --presses on, from stream 1,
so --presses leaves the plays' taps alone). J <= D, and D + J < HORIZON. The engine still runs
every action at its recorded tick with no command delay, and the loop also stops at every tap,
so a row at tick t shows state(t), the board about D ticks before its play lands, and the
memory view(t, max(BOARD_LAG, L)), L = --mem-lag (alias --opp-lag; --mem-lag legacy is refused).
A play row's label is the play it taps (card and tile, as before); when that card is not in the
engine's hand at t (a card a pending play of the same side cycles in), sim/model_tokens
.target_slot gives -1 and the row trains the tile head only (info "not_in_hand" counts them).
A press row (presses on) sits at its tap too; its command still goes in at the recorded tick.
Wait rows: every WAIT_EVERY ticks for a side with no tap (play or press) in [t, t + W],
`--wait-window W` (default 10; with D = 0 that is exactly the legacy rule). Every row has two
more columns: "own_pending" (PENDING_SLOTS int16 card tokens) and "own_pending_land"
(PENDING_SLOTS float16, (exec - t) / PENDING_SCALE clipped at 1): the side's own plays tapped
before t that run at or after t (tap < t <= exec), soonest first. That is the spec's
tap <= t < exec with two boundary changes: the row's own tap is left out (it is the row's
label), and a play that runs at t is still pending, since state(t) is taken before the step
that runs it (tools/rs_memory.py, "the engine's phase"). Presses are not pending tokens.
hand_exact is honest (exact_hand): True once the side has run at least EXACT_PLAYS plays, the
last four distinct, and no Mirror among its last EXACT_MIRROR_WINDOW (sim/replay_rows
.cycle_hand); legacy rows keep True. The trust cut, the outcome window and trusted_v2 count
from the row's tick, as before. vocab.json: obs_delay ticks and jitter, "decision_rows"
(DecisionRule.to_json) and DECISION_FIELDS last in "extra_columns" (decision_vocab);
rows_for's info and manifest.json get "decision" counts (play_rows, press_rows, not_in_hand,
clamped, pending_rows, exact_rows). sim/train_tokens.py reads none of the new columns (E14 /
E15 will): the legacy model trains on decision-time rows as they are, schema rs_rows/legacy.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import operator
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mode

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from royalesim_redrive import CORPUS, base_key, deal_order  # noqa: E402
import rs_cards  # noqa: E402
import rs_engine  # noqa: E402
import rs_memory  # noqa: E402
import rs_orient  # noqa: E402
import rs_presses  # noqa: E402
import vision_classes  # noqa: E402

OUT = ROOT / "data" / "rs_rows"
MAX_UNITS, UF = 48, 7
HORIZON, WAIT_EVERY, CHUNK = 200, 20, 10
HOG26 = frozenset({"Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits",
                   "Musketeer", "Skeletons", "Log"})
MODES = {"Ranked", "Ladder", "1v1 Battle", "Grand Challenge", "Classic Challenge"}
F = None   # field indexes, filled on first use
# --live-compat: unit tokens carry only what the live memory reader provides.
# The reader reports the card that put a unit down (a Witch's skeleton reads as
# the Witch), hit points and max hit points, side and position; not RoyaleSim's
# unit_type, not a deploy timer, not a flying flag. Rows built with this flag
# use: id = creating card, flying / building from that card's catalogue row,
# and log(max hp) in place of the deploy flag, so a model trained on them sees
# in training exactly what it will see live.
LIVE = False
CARD_FLY: dict = {}
CARD_BUILDING: dict = {}

#: --units (module doc): what a unit token names when the rows are not --live-compat.
UNITS_TYPES, UNITS_VISION = "types", "vision"
UNIT_MODES = (UNITS_TYPES, UNITS_VISION)
UNITS = UNITS_TYPES
#: --units vision: the class table (vision_classes.ClassTable) and its lookup of this engine's
#: unit_types_json() (vision_classes.UnitLookup); _setup fills both.
VCT = None
VC = None
#: The columns --units vision adds to FIELDS.
VISION_FIELDS = ("unit_form",)

#: --mem-lag (module doc): a lag in ticks, or MEM_LEGACY for the order before E08.
MEM_LEGACY = "legacy"
MEM_LAG = rs_memory.BOARD_LAG
#: vocab.json "opp_memory" of rows whose memory is dated at execution (absent: legacy).
OPP_MEMORY_EXEC = "exec"

#: --visibility (module doc): which units a row shows. VIS_LEGACY every living one (the rows
#: before E12), VIS_PUBLIC what the acting player's screen shows.
VIS_LEGACY, VIS_PUBLIC = "legacy", "public"
VISIBILITY_MODES = (VIS_LEGACY, VIS_PUBLIC)
VISIBILITY = VIS_LEGACY
#: EntityState.status_flags bits (royalegym protocol.py:579-583). royalesim.STATUS_BITS names
#: bit k (1 << k); STATUS_NAMES are its first three names (check_status_bits).
STATUS_UNDERGROUND, STATUS_INVISIBLE, STATUS_HIDDEN = 1, 2, 4
STATUS_NAMES = ("underground", "invisible", "hidden")
#: --visibility public: an enemy unit with one of these bits is not on the acting player's
#: screen. A tunnelling one (STATUS_UNDERGROUND) is: its dirt trail.
UNSEEN_STATUS = STATUS_INVISIBLE | STATUS_HIDDEN
#: The columns --visibility public adds to FIELDS: 1.0 for a unit tunnelling now.
VISIBILITY_FIELDS = ("unit_burrow",)

#: --presses (module doc, "Presses"): rs_presses.PRESS_OFF (the rows before E10) or PRESS_ON.
PRESSES = rs_presses.PRESS_OFF
#: The engine's ability buttons per side (B) and hand size; _setup reads them from royalesim.
BUTTONS = rs_presses.FALLBACK_BUTTONS
HAND_SIZE = rs_presses.FALLBACK_HAND_SIZE
#: The row columns --presses on adds to FIELDS, and every column it adds to a shard (the press
#: label act_press first), as vocab.json "extra_columns" lists them.
PRESS_FIELDS = ("ability_card", "ability_state")
PRESS_COLUMNS = ("act_press",) + PRESS_FIELDS


def press_vocab(presses) -> dict:
    """vocab_record keywords for --presses: none for off (the vocab.json of the rows before
    E10), else {"presses": rs_presses.vocab_entry(BUTTONS, HAND_SIZE)}."""
    if rs_presses.press_mode(presses) == rs_presses.PRESS_OFF:
        return {}
    return {"presses": rs_presses.vocab_entry(BUTTONS, HAND_SIZE)}


#: --wait-window (module doc, "Decision time"): a decision-time wait row needs no tap of its side
#: in [t, t + W]. The legacy rows' rule (no play in (t, t + 10], none at t) is W = 10 at D = 0.
WAIT_WINDOW = 10
#: own_pending: how many of a side's pending plays a row shows, and the ticks that read 1.0.
PENDING_SLOTS, PENDING_SCALE = 4, 40
#: The columns decision-time rows add to FIELDS.
DECISION_FIELDS = ("own_pending", "own_pending_land")
#: exact_hand (tokens-10): the plays after which the FIFO cycle names the hand whatever the deal,
#: and how far back a Mirror spoils it (sim/replay_rows.cycle_hand); the catalogue name.
EXACT_PLAYS, EXACT_MIRROR_WINDOW, MIRROR = 4, 8, "Mirror"
#: tap_delays' rng streams, one per kind of action.
STREAM_PLAYS, STREAM_PRESSES = 0, 1
#: decision_rule: the settings a DecisionRule's JSON carries, and the keys that only describe it.
_DECISION_KEYS = ("delay", "jitter", "wait_window")
_DECISION_NOTES = ("seed", "streams", "tap_floor", "pending_slots", "pending_scale",
                   "pending_rule", "hand_exact")


@dataclass(frozen=True)
class DecisionRule:
    """--obs-delay, --obs-delay-jitter and --wait-window; decision_rule() gives a checked one.
    delay 0 (the default): the rows at the tick each play ran, the rows before E11."""
    delay: int = 0
    jitter: int = 0
    wait_window: int = WAIT_WINDOW

    @property
    def on(self) -> bool:
        return self.delay > 0

    def to_json(self) -> dict:
        """vocab.json "decision_rows"; decision_rule() reads it back."""
        return {"delay": self.delay, "jitter": self.jitter, "wait_window": self.wait_window,
                "seed": "int(sha1(replay_tag)[:16], 16)",
                "streams": {"plays": STREAM_PLAYS, "presses": STREAM_PRESSES},
                "tap_floor": 0, "pending_slots": PENDING_SLOTS, "pending_scale": PENDING_SCALE,
                "pending_rule": "own plays with tap < t <= exec, soonest first",
                "hand_exact": (f"honest: >= {EXACT_PLAYS} plays run, the last {EXACT_PLAYS} "
                               f"distinct, no {MIRROR} in the last {EXACT_MIRROR_WINDOW}")}


DECISION = DecisionRule()


def _ticks(value, what: str) -> int:
    """`value` as an int >= 0 (numpy integers and strings of digits too); ValueError otherwise."""
    if isinstance(value, str):
        try:
            value = int(value.strip(), 10)
        except ValueError:
            raise ValueError(f"{what}: ticks (an int >= 0), got {value!r}") from None
    if isinstance(value, bool):
        raise ValueError(f"{what}: ticks (an int >= 0), got {value!r}")
    try:
        ticks = operator.index(value)
    except TypeError:
        raise ValueError(f"{what}: ticks (an int >= 0), got {value!r}") from None
    if ticks < 0:
        raise ValueError(f"{what} must be >= 0, got {ticks}")
    return int(ticks)


def decision_rule(value=None) -> DecisionRule:
    """--obs-delay as rows_for, build and vocab_record take it: None (the module's DECISION), a
    delay in ticks (the other settings at their defaults), a DecisionRule or its to_json().
    ValueError for a negative or non-integer setting, a jitter above the delay (a tap after
    its play), delay + jitter >= HORIZON (the outcome window would miss the play), or a
    --wait-window other than WAIT_WINDOW without a delay (the legacy rows have their own)."""
    if value is None:
        value = DECISION
    if isinstance(value, dict):
        unknown = sorted(set(value) - set(_DECISION_KEYS) - set(_DECISION_NOTES))
        if unknown:
            raise ValueError(f"--obs-delay: unknown setting(s) {unknown}")
        value = DecisionRule(**{k: value[k] for k in _DECISION_KEYS if k in value})
    elif not isinstance(value, DecisionRule):
        value = DecisionRule(delay=value)
    delay = _ticks(value.delay, "--obs-delay")
    jitter = _ticks(value.jitter, "--obs-delay-jitter")
    window = _ticks(value.wait_window, "--wait-window")
    if jitter > delay:
        raise ValueError(f"--obs-delay-jitter {jitter} is more than --obs-delay {delay}: a play "
                         f"would be tapped after it ran")
    if delay + jitter >= HORIZON:
        raise ValueError(f"--obs-delay {delay} + jitter {jitter} must stay under the {HORIZON}-tick "
                         f"outcome window")
    if not delay and window != WAIT_WINDOW:
        raise ValueError(f"--wait-window applies to decision-time rows (--obs-delay > 0); the "
                         f"rows at execution keep their {WAIT_WINDOW}-tick rule")
    return DecisionRule(delay, jitter, window)


def decision_vocab(rule=None) -> dict:
    """vocab_record keywords for --obs-delay: none for 0 (the vocab.json of the rows before
    E11), else {"decision": the rule}."""
    rule = decision_rule(rule)
    return {"decision": rule} if rule.on else {}


def describe_decision(rule=None) -> str:
    """One line for the console."""
    rule = decision_rule(rule)
    if not rule.on:
        return "at execution (each play row at the tick the play ran, as before)"
    jitter = f" +- {rule.jitter}" if rule.jitter else ""
    return (f"decision time, {rule.delay}{jitter} ticks before each play runs (own_pending "
            f"columns, honest hand_exact, no wait row within {rule.wait_window} ticks of a tap)")


def tap_seed(tag) -> int:
    """The jitter seed of a match: its replay tag's sha1, first 16 hex digits, as an int."""
    return int(hashlib.sha1(str(tag).encode()).hexdigest()[:16], 16)


def tap_delays(tag, n: int, rule=None, stream: int = STREAM_PLAYS) -> list:
    """D_i for `n` actions of match `tag`, in order: rule.delay + an integer uniform in
    [-jitter, jitter], from np.random.default_rng([tap_seed(tag), stream]).integers(-jitter,
    jitter + 1, size=n); without jitter every D_i is the delay and no rng is made."""
    rule = decision_rule(rule)
    n = int(n)
    if not rule.jitter or n <= 0:
        return [rule.delay] * max(0, n)
    rng = np.random.default_rng([tap_seed(tag), int(stream)])
    return [rule.delay + int(v) for v in rng.integers(-rule.jitter, rule.jitter + 1, size=n)]


def exact_hand(history) -> bool:
    """tokens-10: whether a side's hand and next card follow from its plays alone, `history`
    being the card names of its plays that have run, oldest first. The queue is first in, first
    out, so after four plays it holds the last four played and the hand is the other four
    cards, whatever the deal. True when at least EXACT_PLAYS have run, the last four are distinct
    (always, in a true cycle) and no MIRROR is among the last EXACT_MIRROR_WINDOW (Mirror cycles
    differently; sim/replay_rows.cycle_hand)."""
    names = list(history)
    if len(names) < EXACT_PLAYS or len(set(names[-EXACT_PLAYS:])) < EXACT_PLAYS:
        return False
    return MIRROR not in names[-EXACT_MIRROR_WINDOW:]


def pending_features(pending, slots: int = PENDING_SLOTS) -> tuple:
    """(own_pending int16 [slots], own_pending_land float16 [slots]) from `pending`, (card
    token, ticks to land) soonest first: the first `slots` of them, ticks / PENDING_SCALE clipped
    to [0, 1], zeros after."""
    cards = np.zeros(slots, dtype=np.int16)
    land = np.zeros(slots, dtype=np.float16)
    for k, (card, ticks) in enumerate(list(pending)[:slots]):
        cards[k] = int(card)
        land[k] = min(1.0, max(0.0, float(ticks) / PENDING_SCALE))
    return cards, land


class DecisionPlan:
    """E11: when each recorded action of one match was tapped, and what a row at tick t knows of
    its own side's taps (module doc, "Decision time").

    queue: rows_for's plays, (exec tick, engine team, card name, x, y) in tick order; tokens:
    their row tokens (catalogue id + 2); presses: (exec tick, engine team, rs_presses.Press) in
    tick order; tag: the replay tag; rule: what decision_rule takes. `events` lists the row
    events, (tap, kind 0 play / 1 press, index into queue / presses), in row order: by tap, a
    tick's plays before its presses, each in queue order.
    """

    def __init__(self, queue, tokens, presses, tag, rule=None):
        self.rule = rule = decision_rule(rule)
        self.max_delay = rule.delay + rule.jitter
        delays = tap_delays(tag, len(queue), rule, STREAM_PLAYS)
        pdelays = tap_delays(tag, len(presses), rule, STREAM_PRESSES)
        self.play_taps = [max(0, int(q[0]) - d) for q, d in zip(queue, delays)]
        self.press_taps = [max(0, int(p[0]) - d) for p, d in zip(presses, pdelays)]
        self.play_clamped = [int(q[0]) - d < 0 for q, d in zip(queue, delays)]
        self.press_clamped = [int(p[0]) - d < 0 for p, d in zip(presses, pdelays)]
        self.events = sorted([(t, 0, k) for k, t in enumerate(self.play_taps)]
                             + [(t, 1, k) for k, t in enumerate(self.press_taps)])
        self._taps = {0: [], 1: []}               # every tap of a team, sorted
        self._exec = {0: [], 1: []}               # its plays: exec ticks (sorted), taps, ...
        self._tap = {0: [], 1: []}
        self._token = {0: [], 1: []}
        self._name = {0: [], 1: []}
        for k, q in enumerate(queue):
            team = int(q[1])
            self._taps[team].append(self.play_taps[k])
            self._exec[team].append(int(q[0]))
            self._tap[team].append(self.play_taps[k])
            self._token[team].append(int(tokens[k]))
            self._name[team].append(q[2])
        for k, p in enumerate(presses):
            self._taps[int(p[1])].append(self.press_taps[k])
        for team in (0, 1):
            self._taps[team].sort()

    def tapped(self, team: int, lo: int, hi: int) -> bool:
        """Whether `team` taps anything (a play or a press) at a tick in [lo, hi]."""
        taps = self._taps[team]
        k = bisect.bisect_left(taps, lo)
        return k < len(taps) and taps[k] <= hi

    def pending(self, team: int, tick: int) -> list:
        """`team`'s plays with tap < tick <= exec, as (card token, exec - tick), soonest first."""
        execs, taps = self._exec[team], self._tap[team]
        out = []
        k = bisect.bisect_left(execs, tick)
        while k < len(execs) and execs[k] <= tick + self.max_delay:   # later ones tap after tick
            if taps[k] < tick:
                out.append((self._token[team][k], execs[k] - tick))
            k += 1
        return out

    def exact(self, team: int, tick: int) -> bool:
        """hand_exact of `team`'s rows at `tick`: exact_hand of its plays that ran before it."""
        return exact_hand(self._name[team][:bisect.bisect_left(self._exec[team], tick)])


class VisibilityError(ValueError):
    """--visibility public on a unit whose status_flags the engine did not report (-1)."""


def visibility_mode(value) -> str:
    """--visibility as obs_row, build and tools/rs_eval.py take it: one of VISIBILITY_MODES.
    ValueError otherwise."""
    if not isinstance(value, str) or value not in VISIBILITY_MODES:
        raise ValueError(f"--visibility: one of {', '.join(VISIBILITY_MODES)}, got {value!r}")
    return value


def describe_visibility(visibility) -> str:
    """One line for the console."""
    if visibility_mode(visibility) == VIS_LEGACY:
        return "legacy (every living unit, an invisible or hidden enemy too)"
    return ("public (no enemy with STATUS_INVISIBLE or STATUS_HIDDEN; a tunnelling unit stays "
            "where it is, unit_burrow 1)")


def visibility_rule() -> dict:
    """vocab.json "visibility_rule" of --visibility public rows: what obs_row leaves out and
    flags, by royalesim.STATUS_BITS name, and the bits."""
    return {"drop_enemy": ["invisible", "hidden"], "unit_burrow": "underground",
            "bits": dict(zip(STATUS_NAMES, (STATUS_UNDERGROUND, STATUS_INVISIBLE,
                                            STATUS_HIDDEN)))}


def visibility_vocab(visibility) -> dict:
    """vocab_record keywords for --visibility: none for legacy (the vocab.json of the rows
    before E12), else {"visibility": visibility}."""
    if visibility_mode(visibility) == VIS_LEGACY:
        return {}
    return {"visibility": visibility}


def visibility_of(meta) -> str:
    """The --visibility of the rows a checkpoint trained on: its meta's (vocab.json's)
    "visibility"; VIS_LEGACY when absent (every checkpoint before E12) or with no meta."""
    return visibility_mode((meta or {}).get("visibility", VIS_LEGACY))


def check_status_bits(royalesim) -> None:
    """When the engine exports royalesim.STATUS_BITS (0.1.20 on), its first three names must
    be STATUS_NAMES, the bits --visibility public reads; rs_engine.EngineError otherwise. An
    engine that does not export them passes (the bits are protocol.py's)."""
    names = getattr(royalesim, "STATUS_BITS", None)
    if names is None:
        return
    got = tuple(str(n) for n in list(names)[:len(STATUS_NAMES)])
    if got != STATUS_NAMES:
        raise rs_engine.EngineError(
            f"royalesim.STATUS_BITS starts {list(got)}; --visibility public reads bits 0-2 as "
            f"{list(STATUS_NAMES)}")


def unit_status(e, fields=None) -> int:
    """status_flags of entity row `e` (`fields`: name -> index, default F). The engine's -1,
    "not reported", raises VisibilityError: read as a mask it would be every bit at once
    (royalegym protocol.status_of)."""
    fields = F if fields is None else fields
    status = e[fields["status_flags"]]
    if status is None or isinstance(status, bool) or status < 0:
        raise VisibilityError(
            f"--visibility public: the unit uid {e[fields['uid']]} has no status_flags "
            f"({status!r}); this engine does not report them")
    return int(status)


def seen_by(e, team, fields=None) -> bool:
    """Whether `team`'s screen shows entity row `e` (--visibility public): its own units always
    (their status is not read here); an enemy unit unless it has an UNSEEN_STATUS bit. A
    tunnelling enemy is shown: its dirt trail is on the screen."""
    fields = F if fields is None else fields
    if e[fields["team"]] == team:
        return True
    return not unit_status(e, fields) & UNSEEN_STATUS


def burrowing(e, fields=None) -> bool:
    """Whether entity row `e` is tunnelling now (STATUS_UNDERGROUND): unit_burrow."""
    return bool(unit_status(e, fields) & STATUS_UNDERGROUND)


#: --trust (module doc, "Trust v2"): TRUST_LEGACY, `trusted` only (the rows before E09);
#: TRUST_V2, also `trusted_v2`.
TRUST_LEGACY, TRUST_V2 = "legacy", "v2"
TRUST_MODES = (TRUST_LEGACY, TRUST_V2)
#: --trust-sides: which engine princess a recorded princess_left is. players[t]["tower_hp"] is
#: (king, world-left, world-right) for both seats (tools/rs_decide.py TOWER_CENTRE).
SIDES_UNORDERED, SIDES_WORLD, SIDES_OWN = "unordered", "world", "own"
TRUST_SIDES = (SIDES_UNORDERED, SIDES_WORLD, SIDES_OWN)
#: The engine slot of a recorded princess_left per engine seat (blue, red). "world": the world-left
#: (native low x) tower of both. "own": the owner's own left, taking native x to grow to the
#: right of the player at native high y (red, as on the live screen), so blue's left is world right.
LEFT_SLOT = {SIDES_WORLD: (1, 1), SIDES_OWN: (2, 1)}
#: --trust-hp-margin: the damage beyond the recording's (a share of the engine tower's max hp)
#: a tower may take before trusted_v2 is cut.
TRUST_HP_MARGIN = 0.10
#: tokens-16: under v2 a row more than WAIT_TAIL ticks after the last recorded play is not trusted.
WAIT_TAIL = 20
#: The columns --trust v2 adds to FIELDS.
TRUST_FIELDS = ("trusted_v2",)
#: final_tower_hitpoints keys (sim/replay_corpus._towers).
TOWER_KEYS = ("king", "princess_left", "princess_right")
#: Why a match's trust was cut: the legacy cuts, then the ones only v2 makes; CUT_NONE: no cut.
CUT_CROWNS, CUT_HAND, CUT_REFUSED = "crowns", "hand", "refused"
CUT_TOWER_LOST, CUT_TOWER_HP, CUT_GAME_OVER = "tower_lost", "tower_hp", "game_over"
CUT_NONE = "none"
#: --presses on (E10): a refused press, or one rs_presses.button_of could not place, cuts
#: `trusted` (and so trusted_v2) with this reason.
CUT_PRESS = "press"
#: princess_vote's answers, and the seat names manifest.json counts them under.
VOTES = ("slot1", "slot2")
SEAT_NAMES = ("blue", "red")


@dataclass(frozen=True)
class TrustRule:
    """--trust, --trust-sides and --trust-hp-margin; trust_rule() gives a checked one."""
    mode: str = TRUST_LEGACY
    sides: str = SIDES_UNORDERED
    hp_margin: float = TRUST_HP_MARGIN

    @property
    def v2(self) -> bool:
        return self.mode == TRUST_V2

    def to_json(self) -> dict:
        """What vocab.json "trust" and manifest.json "trust" record; trust_rule() reads it back."""
        return {"mode": self.mode, "sides": self.sides, "hp_margin": self.hp_margin,
                "wait_tail": WAIT_TAIL, "horizon": HORIZON}


TRUST = TrustRule()
_RULE_KEYS = ("mode", "sides", "hp_margin")


def trust_rule(value=None) -> TrustRule:
    """--trust as rows_for, build and vocab_record take it: None (the module's TRUST), a mode
    ("legacy" / "v2", the other settings at their defaults), a TrustRule or its to_json().
    ValueError for an unknown mode or sides, or a margin outside [0, 1]."""
    if value is None:
        value = TRUST
    if isinstance(value, str):
        value = TrustRule(mode=value)
    elif isinstance(value, dict):
        unknown = sorted(set(value) - set(_RULE_KEYS) - {"wait_tail", "horizon"})
        if unknown:
            raise ValueError(f"--trust: unknown setting(s) {unknown}")
        value = TrustRule(**{k: value[k] for k in _RULE_KEYS if k in value})
    if not isinstance(value, TrustRule):
        raise ValueError(f"--trust: one of {', '.join(TRUST_MODES)}, got {value!r}")
    if not isinstance(value.mode, str) or value.mode not in TRUST_MODES:
        raise ValueError(f"--trust: one of {', '.join(TRUST_MODES)}, got {value.mode!r}")
    if not isinstance(value.sides, str) or value.sides not in TRUST_SIDES:
        raise ValueError(f"--trust-sides: one of {', '.join(TRUST_SIDES)}, got {value.sides!r}")
    margin = value.hp_margin
    if (isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not math.isfinite(margin) or not 0 <= margin <= 1):
        raise ValueError(f"--trust-hp-margin: a share of a tower's max hp in [0, 1], "
                         f"got {margin!r}")
    return TrustRule(value.mode, value.sides, float(margin))


def trust_vocab(rule=None) -> dict:
    """vocab_record keywords for --trust: none for legacy (the vocab.json of the rows before
    E09), else {"trust": the rule}."""
    rule = trust_rule(rule)
    return {"trust": rule} if rule.v2 else {}


def describe_trust(rule=None) -> str:
    """One line for the console."""
    rule = trust_rule(rule)
    if not rule.v2:
        return "legacy (`trusted` only: crowns, hand and refused-step cuts)"
    return (f"v2 (`trusted` and `trusted_v2`: also tower divergence from the recorded final "
            f"hp, margin {rule.hp_margin:g} of max hp, princesses {rule.sides}; engine game over "
            f"before the last play; waits more than {WAIT_TAIL} ticks after the last play)")


def _hp(value):
    """A final_tower_hitpoints value as an int; None when absent or not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def recorded_towers(battle: dict, rs: dict) -> dict:
    """{engine seat: {TOWER_KEYS key: final hp}} from each side's final_tower_hitpoints
    (sim/replay_corpus._towers); `rs` maps the IL_Replay side (0 team, 1 opponent) to its
    engine seat. A missing or non-numeric value is left out, and that tower is not checked."""
    out = {}
    for k, side in enumerate(("team", "opponent")):
        players = (battle.get(side) or {}).get("players") or []
        hp = (players[0].get("final_tower_hitpoints") or {}) if players else {}
        hp = hp if isinstance(hp, dict) else {}
        out[rs[k]] = {key: v for key in TOWER_KEYS if (v := _hp(hp.get(key))) is not None}
    return out


def level_scale(recorded, max_hp) -> float:
    """How much stronger the recorded towers of one kind (the king, or the princesses) were than
    the engine's: the largest recorded final hp over the engine's max hp, at least 1. Towers do
    not heal, so a recorded hp above the engine's max is a higher tower level (or a sturdier
    tower troop); an untouched tower gives the ratio exactly. 1 when nothing is recorded."""
    known = [r for r in recorded if r is not None]
    if not known or max_hp <= 0:
        return 1.0
    return max(1.0, max(known) / max_hp)


def tower_breach(hp, rec, max_hp, scale=1.0, margin=TRUST_HP_MARGIN):
    """Why an engine tower at `hp` (of `max_hp`) has left a recording that ends with it at `rec`,
    or None: CUT_TOWER_LOST when the engine destroyed it and the recording ends with it
    standing; CUT_TOWER_HP when the engine's damage, max_hp - hp, exceeds the recorded damage,
    scale x max_hp - rec, by more than margin x max_hp. `rec` None: not recorded, no check."""
    if rec is None:
        return None
    if hp <= 0 < rec:
        return CUT_TOWER_LOST
    if max_hp > 0 and (max_hp - hp) > (scale * max_hp - rec) + margin * max_hp:
        return CUT_TOWER_HP
    return None


def seat_breach(hp, max_hp, rec, seat, rule=None):
    """Why one engine seat's towers have left the recording of that seat, or None. `hp` /
    `max_hp`: its tower_hp / tower_max_hp (king, world-left, world-right); `rec`: its
    recorded_towers entry. The princesses pair as rule.sides says: SIDES_UNORDERED pairs the
    sorted engine hp with the sorted recorded hp (a value not recorded lowest), which is the
    pairing most favourable to the engine; world / own pair them by LEFT_SLOT. CUT_TOWER_LOST
    wins over CUT_TOWER_HP."""
    rule = trust_rule(rule)
    king = rec.get("king")
    why = [tower_breach(hp[0], king, max_hp[0], level_scale([king], max_hp[0]), rule.hp_margin)]
    left, right = rec.get("princess_left"), rec.get("princess_right")
    pmax = max(max_hp[1], max_hp[2])
    scale = level_scale([left, right], pmax)
    if rule.sides == SIDES_UNORDERED:
        recs = sorted([left, right], key=lambda r: -math.inf if r is None else r)
        pairs = [(h, r, pmax) for h, r in zip(sorted(hp[1:3]), recs)]
    else:
        ls = LEFT_SLOT[rule.sides][seat]
        pairs = [(hp[ls], left, max_hp[ls]), (hp[3 - ls], right, max_hp[3 - ls])]
    why += [tower_breach(h, r, m, scale, rule.hp_margin) for h, r, m in pairs]
    if CUT_TOWER_LOST in why:
        return CUT_TOWER_LOST
    return CUT_TOWER_HP if CUT_TOWER_HP in why else None


def divergence(hp_track, max_hp, recorded, rule=None) -> tuple:
    """(tick, why) of the first entry of `hp_track`, [(tick, (blue tower_hp, red tower_hp))] in
    tick order, where a seat's towers have left the recording (seat_breach); (None, None) when
    none has. `max_hp`: (blue, red) tower_max_hp; `recorded`: recorded_towers. A seat with
    nothing recorded is not checked."""
    rule = trust_rule(rule)
    for tick, hps in hp_track:
        whys = [seat_breach(hps[s], max_hp[s], recorded[s], s, rule)
                for s in (0, 1) if recorded.get(s)]
        if CUT_TOWER_LOST in whys:
            return tick, CUT_TOWER_LOST
        if CUT_TOWER_HP in whys:
            return tick, CUT_TOWER_HP
    return None, None


def princess_vote(hp, rec):
    """manifest.json princess_votes for one seat: when exactly one princess fell both in the
    engine (`hp`, its tower_hp at the end) and in the recording (`rec`, its recorded_towers
    entry, both princesses recorded), "slot1" if that is the same tower when princess_left is
    engine slot 1, else "slot2"; None otherwise (no evidence either way)."""
    left, right = rec.get("princess_left"), rec.get("princess_right")
    if left is None or right is None:
        return None
    rec_dead, rs_dead = (left <= 0, right <= 0), (hp[1] <= 0, hp[2] <= 0)
    if sum(rec_dead) != 1 or sum(rs_dead) != 1:
        return None
    return VOTES[0] if rec_dead == rs_dead else VOTES[1]


def _tower_hp(state, key="tower_hp") -> tuple:
    """(blue, red) players[t][key][:3] as int tuples: king, world-left, world-right."""
    return tuple(tuple(int(v) for v in state["players"][t][key][:3]) for t in (0, 1))


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def trust_stats(rows, infos, rule=None) -> dict:
    """One shard's counts for manifest.json, from its rows and the rows_for `info` of each of
    its matches: rows, play_rows, trusted, hog_rows, hog_trusted, matches, and "cuts" (matches
    by why `trusted` was cut, CUT_NONE when it was not). Under --trust v2 also trusted_v2,
    hog_trusted_v2, "cuts_v2" and princess_votes ({seat name: {vote: matches}}). When the
    infos carry "presses" (--presses on), also "presses": press_rows (rows with a press label)
    and the matches' rs_presses.PressLog counts summed. When they carry "decision" (E11,
    --obs-delay > 0), also "decision": those counts summed."""
    rule = trust_rule(rule)
    hog = [r for r in rows if r["hog"]]
    st = {"rows": len(rows), "play_rows": sum(1 for r in rows if r["action"][0]),
          "trusted": sum(1 for r in rows if r["trusted"]), "hog_rows": len(hog),
          "hog_trusted": sum(1 for r in hog if r["trusted"]), "matches": len(infos),
          "cuts": _count(i["trust"]["why"] for i in infos)}
    if rule.v2:
        st["trusted_v2"] = sum(1 for r in rows if r["trusted_v2"])
        st["hog_trusted_v2"] = sum(1 for r in hog if r["trusted_v2"])
        st["cuts_v2"] = _count(i["trust"]["why_v2"] for i in infos)
        st["princess_votes"] = {
            name: {v: sum(1 for i in infos if i["trust"]["votes"][s] == v) for v in VOTES}
            for s, name in enumerate(SEAT_NAMES)}
    if any("presses" in i for i in infos):
        presses = {"press_rows": sum(1 for r in rows if r.get("press", 0))}
        for i in infos:
            merge_stats(presses, i.get("presses", {}))
        st["presses"] = presses
    if any("decision" in i for i in infos):
        decision: dict = {}
        for i in infos:
            merge_stats(decision, i.get("decision", {}))
        st["decision"] = decision
    return st


def merge_stats(total: dict, part: dict) -> dict:
    """Adds trust_stats `part` into `total` (in place, nested counts too) and returns it."""
    for key, value in part.items():
        if isinstance(value, dict):
            merge_stats(total.setdefault(key, {}), value)
        else:
            total[key] = total.get(key, 0) + value
    return total


#: The trust_stats counts a manifest.json shard entry repeats.
SHARD_KEYS = ("rows", "play_rows", "trusted", "trusted_v2", "hog_rows", "hog_trusted",
              "hog_trusted_v2", "matches")


def shard_entry(batch_id: int, st: dict) -> dict:
    """manifest.json's entry for the shard build(batch_id) wrote, from its trust_stats."""
    return {"shard": f"shard-{batch_id:04d}.npz", **{k: st[k] for k in SHARD_KEYS if k in st}}


def _rate(k, n):
    return k / n if n else None


def manifest(totals: dict, shards, skipped, rule=None) -> dict:
    """manifest.json: the run's merged trust_stats `totals` as counts and rates (a share of the
    rows; None with no rows), wait_rows, the skipped matches by reason, the trust rule and the
    shard entries (shard_entry, in shard order). Under --trust v2 also trusted_v2 and its
    rates, cuts_v2 and princess_votes; with press counts in `totals` (--presses on) also
    "presses"; with decision counts (--obs-delay > 0) also "decision"."""
    rule = trust_rule(rule)
    n, hog, plays = totals.get("rows", 0), totals.get("hog_rows", 0), totals.get("play_rows", 0)
    out = {"rows": n, "play_rows": plays, "wait_rows": n - plays,
           "matches": totals.get("matches", 0), "skipped": dict(skipped),
           "trust": rule.to_json()}
    for col in ("trusted",) + (TRUST_FIELDS if rule.v2 else ()):
        k, h = totals.get(col, 0), totals.get(f"hog_{col}", 0)
        out.update({col: k, f"{col}_rate": _rate(k, n), f"hog_{col}": h,
                    f"hog_{col}_rate": _rate(h, hog)})
    out["hog_rows"] = hog
    out["cuts"] = dict(totals.get("cuts", {}))
    if rule.v2:
        out["cuts_v2"] = dict(totals.get("cuts_v2", {}))
        out["princess_votes"] = {name: {v: totals.get("princess_votes", {}).get(name, {})
                                        .get(v, 0) for v in VOTES} for name in SEAT_NAMES}
    if "presses" in totals:
        out["presses"] = totals["presses"]
    if "decision" in totals:
        out["decision"] = totals["decision"]
    out["shards"] = sorted(shards, key=lambda s: s["shard"])
    return out


def describe_manifest(man: dict) -> str:
    """One line for the console: the trust rates of a manifest()."""
    def pct(rate):
        return "-" if rate is None else f"{100 * rate:.1f}%"
    text = (f"trusted {man['trusted']:,} of {man['rows']:,} rows ({pct(man['trusted_rate'])}; "
            f"Hog 2.6 {pct(man['hog_trusted_rate'])})")
    if "trusted_v2" in man:
        text += (f", trusted_v2 {man['trusted_v2']:,} ({pct(man['trusted_v2_rate'])}; "
                 f"Hog 2.6 {pct(man['hog_trusted_v2_rate'])})")
    if "presses" in man:
        text += (f"; {rs_presses.describe_log(man['presses'])}, "
                 f"{man['presses'].get('press_rows', 0):,} press rows")
    if "decision" in man:
        d = man["decision"]
        text += (f"; decision time: {d.get('not_in_hand', 0):,} of {d.get('play_rows', 0):,} "
                 f"play rows tap a card not yet in hand, {d.get('clamped', 0):,} taps floored "
                 f"at tick 0")
    return text


def unit_mode(value, live=False) -> str:
    """--units as rows_for / build take it: one of UNIT_MODES. ValueError otherwise, and for
    "vision" together with --live-compat (live), which is a unit token of its own."""
    if value not in UNIT_MODES:
        raise ValueError(f"--units: one of {', '.join(UNIT_MODES)}, got {value!r}")
    if live and value == UNITS_VISION:
        raise ValueError("--units vision and --live-compat are two different unit tokens (the "
                         "vision class, the creating card); pick one")
    return value


def mem_lag_value(value):
    """--mem-lag as rows_for, build and rs_eval.play take it: a lag in ticks (an int >= 0; a
    string of digits is read as one) or MEM_LEGACY. ValueError otherwise."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text == MEM_LEGACY:
            return MEM_LEGACY
        try:
            value = int(text, 10)
        except ValueError:
            raise ValueError(f"--mem-lag: a lag in ticks (>= 0) or {MEM_LEGACY!r}, "
                             f"got {value!r}") from None
    if isinstance(value, bool):
        raise ValueError(f"--mem-lag: a lag in ticks (>= 0) or {MEM_LEGACY!r}, got {value!r}")
    try:
        lag = operator.index(value)
    except TypeError:
        raise ValueError(f"--mem-lag: a lag in ticks (>= 0) or {MEM_LEGACY!r}, "
                         f"got {value!r}") from None
    if lag < 0:
        raise ValueError(f"--mem-lag must be >= 0 (a negative lag sees plays before they run), "
                         f"got {lag}")
    return lag


def mem_lag_arg(text: str):
    """argparse type of --mem-lag."""
    try:
        return mem_lag_value(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_mem_lag_arg(ap: argparse.ArgumentParser, help_text: str | None = None,
                    aliases=()) -> None:
    """--mem-lag TICKS|legacy, default rs_memory.BOARD_LAG; tools/rs_eval.py adds it too.
    `aliases`: more option strings for it (rs_rows: "--opp-lag", E11's name); dest mem_lag."""
    ap.add_argument("--mem-lag", *aliases, dest="mem_lag", type=mem_lag_arg,
                    default=rs_memory.BOARD_LAG,
                    metavar="TICKS|legacy", help=help_text or (
                        "opponent memory: a row at tick t sees the plays that ran by "
                        f"t - max(1, TICKS) (default {rs_memory.BOARD_LAG}: what the board "
                        "shows); 'legacy': the order before E08, byte-identical rows that "
                        "show the other seat's same-tick plays"))


def describe_memory(mem_lag) -> str:
    """One line for the console."""
    lag = mem_lag_value(mem_lag)
    if lag == MEM_LEGACY:
        return ("legacy (each play recorded as it is handled; a row sees the other seat's "
                "same-tick plays)")
    return (f"dated at execution, lag {lag} (a row at tick t sees the plays that ran by "
            f"t - {max(rs_memory.BOARD_LAG, lag)})")


def memory_vocab(mem_lag) -> dict:
    """vocab_record keywords for --mem-lag. Legacy: none (opp_lag 0, no "opp_memory": the
    vocab.json of the rows before E08). Otherwise the lag the views in rows_for amount to,
    max(BOARD_LAG, lag) (the view is taken before the tick's plays are recorded, so 0 gives
    the rows of 1), and opp_memory OPP_MEMORY_EXEC."""
    lag = mem_lag_value(mem_lag)
    if lag == MEM_LEGACY:
        return {}
    return {"opp_lag": max(rs_memory.BOARD_LAG, lag), "opp_memory": OPP_MEMORY_EXEC}


def split_of(tag: str) -> str:
    h = int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 5 else "val" if h < 10 else "train"


def _setup(engine=None):
    """(royalesim, IL_Replay key -> card name, card name -> id, catalogue rows) under
    `engine` (an rs_engine.EngineConfig or its to_json(); None = rs_engine.active()).
    The card map is rs_cards.load(). Also sets F, after checking royalesim.ENTITY_FIELDS
    (rs_engine.field_index), and under --units vision (UNITS) VCT, the class table, and VC,
    its lookup of this Battle's unit_types_json(). Under --visibility public (VISIBILITY) it
    first checks royalesim.STATUS_BITS (check_status_bits). BUTTONS and HAND_SIZE come from
    royalesim.ABILITY_BUTTONS / HAND_SIZE (rs_presses.button_count / hand_size)."""
    global F, VC, VCT, BUTTONS, HAND_SIZE
    import royalesim
    if VISIBILITY == VIS_PUBLIC:
        check_status_bits(royalesim)
    BUTTONS, HAND_SIZE = rs_presses.button_count(royalesim), rs_presses.hand_size(royalesim)
    names = rs_cards.load()
    b = rs_engine.make_battle(rs_engine.resolve(engine).undelayed(), royalesim)
    cat = json.loads(b.catalogue_json())
    ids = {row[0]: i for i, row in enumerate(cat)}
    F = rs_engine.field_index(royalesim)
    for i, row in enumerate(cat):
        CARD_FLY[i] = bool(row[5])
        CARD_BUILDING[i] = row[8] == "BUILDING"
    if UNITS == UNITS_VISION:
        VCT = vision_classes.load()
        VC = VCT.lookup(json.loads(b.unit_types_json()))
    return royalesim, names, ids, cat


def own(x_sub, y_sub, team, sub):
    """RoyaleSim sub-tiles -> the acting player's frame, millitiles, bottom = high y."""
    x, y = x_sub / sub, y_sub / sub
    return (x, 32_000 - y) if team == 0 else (18_000 - x, y)


def towers_vec(state, team):
    out = []
    for t in (team, 1 - team):
        hp, mx = state["players"][t]["tower_hp"], state["players"][t]["tower_max_hp"]
        # slots: 0 king, 1 world-left, 2 world-right. Red is reflected in x, so
        # its own left is world right.
        lr = (1, 2) if team == 0 else (2, 1)
        out += [hp[lr[0]] / max(1, mx[lr[0]]), hp[lr[1]] / max(1, mx[lr[1]]),
                hp[0] / max(1, mx[0])]
    return np.asarray(out, dtype=np.float32)


def obs_row(state, team, sub, mem, tick, live=None, units=None, visibility=None, presses=None,
            buttons=None, pending=None, hand_exact=None):
    """One row of `team`'s view. `mem` is the opponent memory, {"seen", "last"} as
    rs_memory.PublicMemory.view() gives it. `live` / `units` / `visibility` / `presses` default
    to LIVE / UNITS / VISIBILITY / PRESSES; with units "vision" (and not live) the row also has
    "unit_form", and VC must be set (_setup). With visibility "public" (E12) the enemy units
    `team`'s screen does not show (seen_by) are left out and the row also has "unit_burrow", 1.0
    for a unit tunnelling now (burrowing); VisibilityError when the engine reports no
    status_flags. With presses "on" (E10) the row also has "ability_card" (BUTTONS int16) and
    "ability_state" (BUTTONS x rs_presses.ABILITY_FEATS float16) from players[team]
    ["abilities"] (rs_presses.ability_features; `buttons`, the side's button cards, names a
    button the engine's rows do not). With `pending` (E11, decision-time rows: `team`'s own
    pending plays, (card token, ticks to land) soonest first; [] for none) the row also has
    "own_pending" and "own_pending_land" (pending_features); None leaves them out. `hand_exact`
    is the row's flag (None: True, as the rows before E11 say)."""
    live = LIVE if live is None else live
    units = unit_mode(UNITS if units is None else units)
    public = visibility_mode(VISIBILITY if visibility is None else visibility) == VIS_PUBLIC
    press = rs_presses.press_mode(PRESSES if presses is None else presses) == rs_presses.PRESS_ON
    vision = not live and units == UNITS_VISION
    if vision and VC is None:
        raise RuntimeError("--units vision rows need the class lookup: set UNITS = 'vision' "
                           "before _setup(), which fills VC")
    forms = np.zeros(MAX_UNITS, dtype=np.float16) if vision else None
    burrow = np.zeros(MAX_UNITS, dtype=np.float16) if public else None
    ents = [e for e in state["entities"] if e[F["tower_slot"]] < 0 and e[F["hp"]] > 0]
    if public:
        # E12: only what `team`'s screen shows, before the sort and the MAX_UNITS cut, so the
        # units shown take the places of the units left out.
        ents = [e for e in ents if seen_by(e, team)]
    king = next((e for e in state["entities"]
                 if e[F["team"]] == team and e[F["tower_slot"]] == 0), None)
    # A destroyed king (a lookahead copy can end the battle) has no entity;
    # sort from where it stands instead.
    kx, ky = ((king[F["x"]], king[F["y"]]) if king is not None
              else (9000 * 18, (3000 if team == 0 else 29000) * 18))
    ents.sort(key=lambda e: (e[F["x"]] - kx) ** 2 + (e[F["y"]] - ky) ** 2)
    ids = np.zeros(MAX_UNITS, dtype=np.int16)
    feats = np.zeros((MAX_UNITS, UF), dtype=np.float16)
    for k, e in enumerate(ents[:MAX_UNITS]):
        ox, oy = own(e[F["x"]], e[F["y"]], team, sub)
        if live:
            card = e[F["card_id"]]
            ids[k] = card + 2
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]),
                        1.0 if CARD_FLY.get(card) else 0.0,
                        1.0 if CARD_BUILDING.get(card) else 0.0,
                        float(np.log1p(max(0, e[F["max_hp"]]))) / 10.0)
        elif vision:
            # The vision class of the unit type, flags from the class table; the form goes
            # to its own column until the v2 schema (E14) makes room in unit_feats.
            unit = e[F["unit_type"]]
            ids[k] = VC.token(unit)
            fly, building = VC.flags(unit)
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]), fly, building,
                        1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
            forms[k] = VC.form_feature(unit)
        else:
            ids[k] = e[F["unit_type"]] + 2
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]), 1.0 if e[F["flying"]] else 0.0,
                        1.0 if e[F["kind"]] == 1 else 0.0,
                        1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
        if public:
            burrow[k] = 1.0 if burrowing(e) else 0.0
    p = state["players"][team]
    hand = [c + 2 for c in p["hand"][:4]] + [0] * (4 - len(p["hand"][:4]))
    opp = 1 - team
    seen = sorted(mem["seen"][opp])[:8]
    last = mem["last"][opp][-4:]
    t_ms = tick * 50
    row = {
        "tick": tick, "side": 1 if team == 0 else -1,
        "hand_exact": True if hand_exact is None else bool(hand_exact),
        "unit_ids": ids, "unit_feats": feats, "towers": towers_vec(state, team),
        "scalars": np.asarray([p["elixir_milli"] / 10_000, min(1.0, t_ms / 300_000),
                               1.0 if t_ms >= 120_000 else 0.0,
                               1.0 if t_ms >= 180_000 else 0.0], dtype=np.float32),
        "hand": np.asarray(hand, dtype=np.int16),
        "next": (p["next_card"] + 2) if p.get("next_card", -1) >= 0 else 0,
        "opp_seen": np.asarray(seen + [0] * (8 - len(seen)), dtype=np.int16),
        "opp_last": np.asarray([c for c, _ in last] + [0] * (4 - len(last)), dtype=np.int16),
        "opp_last_age": np.asarray([min(1.0, (tick - t) / 400) for _, t in last]
                                   + [1.0] * (4 - len(last)), dtype=np.float16),
    }
    if vision:
        row["unit_form"] = forms
    if public:
        row["unit_burrow"] = burrow
    if press:
        cards, states = rs_presses.ability_features(p, BUTTONS, buttons)
        row["ability_card"] = np.asarray(cards, dtype=np.int16)
        row["ability_state"] = np.asarray(states, dtype=np.float16).reshape(
            BUTTONS, len(rs_presses.ABILITY_FEATS))
    if pending is not None:
        row["own_pending"], row["own_pending_land"] = pending_features(pending)
    return row


def rows_for(payload: dict, ctx, engine=None, *, mem_lag=None, trust=None, presses=None,
             info=None, decision=None):
    """Rows of one match. `engine` as in _setup; the battle runs without command delay,
    since the recorded ticks are the ticks the plays executed. `decision` as --obs-delay
    (decision_rule; None: DECISION): a delay > 0 observes each play and press at its tap,
    exec - D_i, and gives every row own_pending, own_pending_land and an honest hand_exact
    (module doc, "Decision time"); 0 keeps the rows at execution, as before. `mem_lag` as --mem-lag
    (None: MEM_LAG): how the opponent memory of a row is dated (module doc). `trust` as
    --trust (trust_rule; None: TRUST): under v2 every row also has "trusted_v2" (module doc,
    "Trust v2"). `presses` as --presses (None: PRESSES): under "on" the recorded ability
    presses are replayed, each gets a press row, and every row has "press", "ability_card" and
    "ability_state" (module doc, "Presses"). A dict `info` gets, for a match that gives rows,
    info["trust"] = {"cut": the tick `trusted` was cut at or None, "why": its CUT_* reason or
    CUT_NONE} and under v2 also "cut_v2", "why_v2" and "votes" (princess_vote of blue and red
    at the end of the run); under presses "on" also info["presses"] (rs_presses.PressLog);
    with a delay also info["decision"], the counts manifest.json sums (module doc)."""
    lag = mem_lag_value(MEM_LAG if mem_lag is None else mem_lag)
    legacy = lag == MEM_LEGACY
    drule = decision_rule(DECISION if decision is None else decision)
    if drule.on and legacy:
        raise ValueError("--obs-delay: decision-time rows read the memory dated at execution; "
                         "--mem-lag legacy is the order before E08")
    rule = trust_rule(trust)
    press_on = rs_presses.press_mode(PRESSES if presses is None else presses) == rs_presses.PRESS_ON
    royalesim, names, ids, cat = ctx
    sub = royalesim.SUBTILE_PER_MILLITILE
    battle = payload["battle"]
    decks, forms, levels = [], [], []
    for side in ("team", "opponent"):
        players = battle[side]["players"]
        if len(players) != 1:
            return None, "not 1v1"
        deck, form, level = [], [], []
        for card in players[0]["deck"]:
            key, f = base_key(card["card_key"])
            name = names.get(key)
            if name is None or name not in ids:
                return None, "card"
            deck.append(name)
            form.append(f)
            level.append(int(card.get("level") or 11))
        decks.append(deck)
        forms.append(form)
        levels.append(level)
    events = sorted((e for e in payload["events"] if e["kind"] == "play_card"),
                    key=lambda e: (e["replay_tick_20hz"], e["source_index"]))
    plays = {0: [], 1: []}
    for e in events:
        seat = 0 if e["side"] == "team" else 1
        name = names.get(base_key(e["card_key"])[0])
        u = e["coordinates"]["native_world_units"]
        plays[seat].append((int(e["replay_tick_20hz"]), name, int(u["x"]), int(u["y"])))
    # The seat from troop plays: the side at native low y plays blue, and every
    # coordinate goes to the engine as recorded (tools/rs_orient.py).
    ori = rs_orient.orient(plays, rs_orient.troop_names(cat))
    if not ori.ok():
        return None, rs_orient.SKIP
    team_seat = ori.team_seat
    dealt = []
    for seat in (0, 1):
        order = deal_order(decks[seat], [c for _, c, _, _ in plays[seat]])
        if order is None:
            return None, "deal"
        dealt.append(order)
    perm = [[decks[s].index(c) for c in dealt[s]] for s in (0, 1)]
    rs = {0: team_seat, 1: 1 - team_seat}
    seat_of = {rs[0]: 0, rs[1]: 1}
    tower = max(1, min(16, mode([lv for s in levels for lv in s])))
    b = rs_engine.make_battle(rs_engine.resolve(engine).undelayed(), royalesim)
    b.reset(seed=0, decks=[[ids[c] for c in dealt[seat_of[t]]] for t in (0, 1)],
            shuffle=0, start_tick=0, elixir_milli=[6000, 6000], tower_hp=None, spawns=[],
            forms=[[forms[seat_of[t]][i] for i in perm[seat_of[t]]] for t in (0, 1)],
            levels=[[levels[seat_of[t]][i] for i in perm[seat_of[t]]] for t in (0, 1)],
            tower_levels=[tower, tower])
    rec_crowns = {rs[0]: battle["team"]["crowns"], rs[1]: battle["opponent"]["crowns"]}
    result = str(battle.get("result", "")).lower()
    won_team = 1.0 if result in ("victory", "win") else 0.0 if result in ("defeat", "loss") else 0.5
    won = {rs[0]: won_team, rs[1]: 1.0 - won_team}
    queue = sorted([(t, rs[s], c, x, y) for s in (0, 1) for (t, c, x, y) in plays[s]])
    play_ticks = {0: [q[0] for q in queue if q[1] == 0], 1: [q[0] for q in queue if q[1] == 1]}
    # E10 (--presses on): the recorded presses as (tick, engine team, rs_presses.Press), each
    # team's button cards in button order (the dealt deck and forms reset() was given), and
    # the obs_row keywords of each team's rows. Off: no press, and obs_row is called as before.
    press_q, btns, pkw, plog = [], {0: [], 1: []}, {0: {}, 1: {}}, None
    hand_n = rs_presses.hand_size(royalesim)
    if press_on:
        champs = rs_presses.champion_ids(cat, getattr(royalesim, "CATALOGUE_FIELDS", None))
        for t in (0, 1):
            s = seat_of[t]
            btns[t] = rs_presses.side_buttons([ids[c] for c in dealt[s]],
                                              [forms[s][k] for k in perm[s]], champs)
        press_q = [(p.tick, rs[p.side], p)
                   for p in rs_presses.parse(payload["events"], names, ids)]
        plog = rs_presses.PressLog(events=len(press_q))
        pkw = {t: {"presses": rs_presses.PRESS_ON, "buttons": btns[t]} for t in (0, 1)}
    # A wait row is not taken in the 10 ticks before a team's play, nor (presses on) a press.
    soon_ticks = {t: play_ticks[t] + [q[0] for q in press_q if q[1] == t] for t in (0, 1)}
    mem = rs_memory.PublicMemory()
    rows, track = [], []
    cut, cut_why = 10 ** 9, None
    hp_track, max_hp, over = [], None, None     # trust v2: engine tower hp by tick, game over
    tick, i, j = 0, 0, 0
    # The last recorded action: the last play, or (presses on) a later press.
    last = max([queue[-1][0] if queue else 0] + [q[0] for q in press_q])
    end = min(last + HORIZON, 6000)
    # E11 (--obs-delay > 0): each action's tap, the rows taken there, and what a row knows of
    # its side's own taps. None: the rows at execution, the loop below as before E11.
    plan, tp, dstat = None, 0, None
    if drule.on:
        plan = DecisionPlan(queue, [ids[q[2]] + 2 for q in queue], press_q,
                            (payload.get("source") or {}).get("replay_tag", ""), drule)
        dstat = {"play_rows": 0, "press_rows": 0, "not_in_hand": 0, "clamped": 0,
                 "pending_rows": 0, "exact_rows": 0}

    def drow(state, team, view, tick):
        """A decision-time row of `team` at `tick`: obs_row with its pending plays and honest
        hand_exact (the one place E17's noise hook wraps for these rows)."""
        pending = plan.pending(team, tick)
        exact = plan.exact(team, tick)
        dstat["pending_rows"] += int(bool(pending))
        dstat["exact_rows"] += int(exact)
        return obs_row(state, team, sub, view, tick, **pkw[team], pending=pending,
                       hand_exact=exact)

    while tick <= end:
        state = json.loads(b.state_json())
        if rule.v2:
            hp_track.append((tick, _tower_hp(state)))
            max_hp = max_hp or _tower_hp(state, "tower_max_hp")
        if state["game_over"]:
            track.append((tick, towers_vec(state, 0)))
            over = tick
            break
        track.append((tick, towers_vec(state, 0)))
        for t in (0, 1):
            if state["players"][t]["crowns"] > rec_crowns[t] and cut > tick:
                cut, cut_why = tick, CUT_CROWNS
        if press_on and tick == 0:
            # The engine's own button order where its abilities rows name their cards.
            for t in (0, 1):
                btns[t], differs = rs_presses.engine_order(state["players"][t], btns[t])
                plog.order_mismatch += int(differs)
                pkw[t]["buttons"] = btns[t]
        # E08: one view for every row of this tick, taken before this tick's plays are
        # recorded. Legacy reads the memory at each row instead, after the plays handled so
        # far, which is the dict the rows before E08 read. Decision-time rows (E11) read the
        # same view, at their tap.
        known = None if legacy else mem.view(tick, lag)
        if plan is not None:
            # E11: the rows of the plays and presses tapped at this tick, from state(tick);
            # their commands still go in at the recorded ticks below.
            while tp < len(plan.events) and plan.events[tp][0] <= tick:
                _, kind, k = plan.events[tp]
                tp += 1
                if kind == 0:
                    _, team, card, x, y = queue[k]
                    cid = ids[card]
                    ex, ey = rs_orient.to_engine(x, y, sub)
                    ox, oy = own(ex, ey, team, sub)
                    row = drow(state, team, known, tick)
                    row["action"] = (cid + 2, int(min(17, max(0, ox // 1000))),
                                     int(min(31, max(0, oy // 1000))))
                    rows.append(row)
                    dstat["play_rows"] += 1
                    dstat["not_in_hand"] += int(cid + 2 not in row["hand"].tolist())
                    dstat["clamped"] += int(plan.play_clamped[k])
                else:
                    _, team, press = press_q[k]
                    button, _ = rs_presses.button_of(press.cards, btns[team])
                    if button is None:              # skipped (and counted) when it is due
                        continue
                    row = drow(state, team, known, tick)
                    row["action"] = (0, 0, 0)
                    row["press"] = button + 1
                    rows.append(row)
                    dstat["press_rows"] += 1
                    dstat["clamped"] += int(plan.press_clamped[k])
        due = []
        while i < len(queue) and queue[i][0] <= tick:
            due.append(queue[i])
            i += 1
        cmds = []
        acting = set()
        for (_, team, card, x, y) in due:
            acting.add(team)
            cid = ids[card]
            hand = state["players"][team]["hand"]
            ex, ey = rs_orient.to_engine(x, y, sub)
            ox, oy = own(ex, ey, team, sub)
            if plan is None:                        # E11: decision-time rows were taken above
                row = obs_row(state, team, sub, mem.view(tick, 0) if legacy else known, tick,
                              **pkw[team])
                row["action"] = (cid + 2, int(min(17, max(0, ox // 1000))), int(min(31, max(0, oy // 1000))))
                rows.append(row)
            if cid in hand:
                cmds.append((team, hand.index(cid), ex, ey))
            elif cut > tick:
                cut, cut_why = tick, CUT_HAND
            mem.record(team, cid + 2, tick)        # the recorded tick is the tick it ran
        # E10: this tick's presses, after its plays: a row each (action (0, 0, 0), press =
        # button + 1) and the command (team, HAND_SIZE + button, 0, 0). A press no single
        # button is behind is not sent, and cuts trust.
        pcmds = []
        while j < len(press_q) and press_q[j][0] <= tick:
            _, team, press = press_q[j]
            j += 1
            acting.add(team)
            button, why = rs_presses.button_of(press.cards, btns[team])
            if button is None:
                plog.skip(why)
                if cut > tick:
                    cut, cut_why = tick, CUT_PRESS
                continue
            if plan is None:                        # E11: its row was taken at its tap
                row = obs_row(state, team, sub, mem.view(tick, 0) if legacy else known, tick,
                              **pkw[team])
                row["action"] = (0, 0, 0)
                row["press"] = button + 1
                rows.append(row)
            pcmds.append(rs_presses.command(team, button, hand_n))
            plog.sent += 1
        if tick % WAIT_EVERY == 0:
            for team in (0, 1):
                if plan is not None:
                    # E11: a wait only where the side taps nothing in [tick, tick + W].
                    if tick > 0 and not plan.tapped(team, tick, tick + drule.wait_window):
                        row = drow(state, team, known, tick)
                        row["action"] = (0, 0, 0)
                        rows.append(row)
                    continue
                soon = any(tick < t <= tick + 10 for t in soon_ticks[team])
                if team not in acting and not soon and tick > 0:
                    row = obs_row(state, team, sub, mem.view(tick, 0) if legacy else known, tick,
                                  **pkw[team])
                    row["action"] = (0, 0, 0)
                    rows.append(row)
        nxt = queue[i][0] if i < len(queue) else end + 1
        if j < len(press_q):
            nxt = min(nxt, press_q[j][0])
        if plan is not None and tp < len(plan.events):
            nxt = min(nxt, plan.events[tp][0])     # E11: the loop stops at every tap too
        step = max(1, min(nxt, (tick // WAIT_EVERY + 1) * WAIT_EVERY) - tick)
        res = b.step(cmds + pcmds if pcmds else cmds, step)
        for k, r in enumerate(res):
            reason = royalesim.DEPLOY_REASONS[r[1]]
            if press_on and k >= len(cmds):          # E10: a press's result
                if not plog.result(reason) and cut > tick:
                    cut, cut_why = tick, CUT_PRESS
            elif reason != "OK" and cut > tick:
                cut, cut_why = tick, CUT_REFUSED
        tick += step
    if rule.v2:
        # E09: the first divergence is the earliest of the legacy cut, the first tick an engine
        # tower leaves the recorded final hp, and a game over with a recorded action (a play;
        # presses on, a press too) still to come; on a tie the reason listed first.
        recorded = recorded_towers(battle, rs)
        cuts = [(cut, cut_why)] if cut_why else []
        d_tick, d_why = divergence(hp_track, max_hp, recorded, rule)
        if d_why:
            cuts.append((d_tick, d_why))
        if over is not None and over <= last:
            cuts.append((over, CUT_GAME_OVER))
        cut_v2, why_v2 = min(cuts, key=lambda c: c[0]) if cuts else (10 ** 9, None)
    ticks = np.asarray([t for t, _ in track])
    tv = np.stack([v for _, v in track])            # blue-frame [own L,R,K, enemy L,R,K]
    blue_total, red_total = tv[:, :3].sum(1), tv[:, 3:].sum(1)
    for row in rows:
        if press_on:
            row["press"] = row.get("press", 0)       # 0: no press (play and wait rows)
        team = 0 if row["side"] == 1 else 1
        a = int(np.searchsorted(ticks, row["tick"]))
        z = int(np.searchsorted(ticks, row["tick"] + HORIZON))
        a, z = min(a, len(ticks) - 1), min(z, len(ticks) - 1)
        own_lost = (blue_total if team == 0 else red_total)
        opp_lost = (red_total if team == 0 else blue_total)
        row["outcome"] = float((opp_lost[a] - opp_lost[z]) - (own_lost[a] - own_lost[z]))
        row["trusted"] = row["tick"] + HORIZON <= cut
        row["won"] = won[team]
        row["hog"] = frozenset(decks[seat_of[team]]) == HOG26
        if rule.v2:
            row["trusted_v2"] = row["tick"] + HORIZON <= cut_v2 and row["tick"] <= last + WAIT_TAIL
    if info is not None:
        trust_info = {"cut": None if cut_why is None else cut, "why": cut_why or CUT_NONE}
        if rule.v2:
            trust_info.update(cut_v2=None if why_v2 is None else cut_v2, why_v2=why_v2 or CUT_NONE,
                              votes=[princess_vote(hp_track[-1][1][s], recorded.get(s, {}))
                                     for s in (0, 1)])
        info["trust"] = trust_info
        if press_on:
            info["presses"] = plog.to_json()
        if dstat is not None:
            info["decision"] = dstat
    return rows, "ok"


FIELDS = ("tick", "side", "hand_exact", "unit_ids", "unit_feats", "towers", "scalars",
          "hand", "next", "opp_seen", "opp_last", "opp_last_age", "outcome", "trusted",
          "won", "hog")


#: vocab.json "schema": the row layout of these shards (obs_row and FIELDS above).
SCHEMA = "rs_rows/legacy"


def vocab_record(cat, unit_types, cfg, fp, card_map, live, *, version=None, schema=SCHEMA,
                 obs_delay=0, obs_delay_jitter=0, opp_lag=0, wait_keep_at_build=None,
                 opp_memory=None, classes=None, visibility=None, trust=None, presses=None,
                 decision=None):
    """vocab.json: the id maps of the rows and how they were made.

    The keys it had before E04 come first, unchanged: n_cards, n_units, live_compat, cards
    (catalogue name -> id + 2), units (unit type -> id + 2), engine, version (`version`,
    else the fingerprint's), engine_config, engine_fingerprint, card_map. Then:

      card_table          what the Battle reported (fp["card_table"]; None before 0.1.42)
      schema              the row layout (SCHEMA)
      obs_delay           {"ticks", "jitter", "opp_lag"}: how far before a play's execution
                          its row is observed (0: rows at the tick it executes; E11's
                          --obs-delay D and --obs-delay-jitter J under `decision`), and
                          opp_lag, how long after a play runs a row's memory shows it
                          (memory_vocab: 0 for legacy rows, else at least BOARD_LAG)
      wait_keep_at_build  the share of wait rows the builder kept; None: no thinning at
                          build, every WAIT_EVERY-tick wait row is in the shards
      wait_every          ticks between wait-row samples

    and only when given (so vocab.json of rows built as before E08 / E16 is unchanged):

      opp_memory          `opp_memory`, OPP_MEMORY_EXEC: plays dated at execution, one
                          view per tick (E08); absent: the legacy order
      units_mode, vision_classes_digest, extra_columns
                          with `classes` (a vision_classes.ClassTable, --units vision):
                          "vision", the table's digest, VISION_FIELDS; n_units and units
                          then count and list the classes (class name -> id + 2)
      visibility, visibility_rule
                          with `visibility` "public" (E12): "public" and visibility_rule(),
                          and VISIBILITY_FIELDS in extra_columns (after VISION_FIELDS);
                          "legacy" (or None) adds nothing
      trust               with `trust` v2 (E09; a mode, TrustRule or its to_json): the rule's
                          to_json(), and TRUST_FIELDS last in extra_columns; legacy (or None)
                          adds nothing
      presses             with `presses` "on" (E10; or the rs_presses.vocab_entry dict a
                          checkpoint's meta carries): that entry (B buttons, slot base, the
                          ability_state columns), and PRESS_COLUMNS last in extra_columns;
                          "off" (or None) adds nothing
      decision_rows       with `decision` on (E11, --obs-delay > 0; what decision_rule takes):
                          DecisionRule.to_json(), DECISION_FIELDS last in extra_columns, and
                          obs_delay ticks / jitter from the rule (an `obs_delay` or
                          `obs_delay_jitter` given as well must agree, else ValueError); a
                          delay of 0 (or None) adds nothing

    sim/train_tokens.py copies all of it into every checkpoint's meta, and
    tools/rs_checkpoint.load_policy checks cards, units and card_table against the engine
    a checkpoint is loaded into.
    """
    vision = classes is not None
    if vision:
        unit_mode(UNITS_VISION, live)
    public = visibility is not None and visibility_mode(visibility) == VIS_PUBLIC
    rule = None if trust is None else trust_rule(trust)
    v2 = rule is not None and rule.v2
    if isinstance(presses, dict):
        press_entry = dict(presses)
        rs_presses.press_mode(press_entry.get("mode"))
        if press_entry["mode"] == rs_presses.PRESS_OFF:
            press_entry = None
    elif presses is None or rs_presses.press_mode(presses) == rs_presses.PRESS_OFF:
        press_entry = None
    else:
        press_entry = rs_presses.vocab_entry(BUTTONS, HAND_SIZE)
    drule = None if decision is None else decision_rule(decision)
    if drule is not None and not drule.on:
        drule = None
    if drule is not None:
        for given, want, what in ((obs_delay, drule.delay, "obs_delay"),
                                  (obs_delay_jitter, drule.jitter, "obs_delay_jitter")):
            if int(given) not in (0, want):
                raise ValueError(f"vocab_record: {what} {given} is not the decision rule's {want}")
        obs_delay, obs_delay_jitter = drule.delay, drule.jitter
    rec = {
        "n_cards": len(cat) + 2,
        "n_units": ((len(cat) + 2) if live else (len(classes) + 2) if vision
                    else (len(unit_types) + 2)),
        "live_compat": live,
        "cards": {row[0]: i + 2 for i, row in enumerate(cat)},
        "units": ({c.name: c.id + 2 for c in classes} if vision
                  else {u: i + 2 for i, u in enumerate(unit_types)}),
        "engine": "royalesim", "version": fp.get("version", "?") if version is None else version,
        "engine_config": cfg.to_json(), "engine_fingerprint": fp, "card_map": card_map,
        "card_table": fp.get("card_table"),
        "schema": schema,
        "obs_delay": {"ticks": int(obs_delay), "jitter": int(obs_delay_jitter),
                      "opp_lag": int(opp_lag)},
        "wait_keep_at_build": None if wait_keep_at_build is None else float(wait_keep_at_build),
        "wait_every": WAIT_EVERY,
    }
    if opp_memory is not None:
        rec["opp_memory"] = str(opp_memory)
    if vision:
        rec["units_mode"] = UNITS_VISION
        rec["vision_classes_digest"] = classes.digest
    if public:
        rec["visibility"] = VIS_PUBLIC
        rec["visibility_rule"] = visibility_rule()
    if v2:
        rec["trust"] = rule.to_json()
    if press_entry is not None:
        rec["presses"] = press_entry
    if drule is not None:
        rec["decision_rows"] = drule.to_json()
    extra = ((list(VISION_FIELDS) if vision else []) + (list(VISIBILITY_FIELDS) if public else [])
             + (list(TRUST_FIELDS) if v2 else [])
             + (list(PRESS_COLUMNS) if press_entry is not None else [])
             + (list(DECISION_FIELDS) if drule is not None else []))
    if extra:
        rec["extra_columns"] = extra
    return rec


def build(batch_id, payload_texts, live=False, out_dir=None, engine=None, *, units=None,
          mem_lag=None, visibility=None, trust=None, presses=None, stats=False,
          decision=None):
    """One shard, shard-<batch_id>.npz in `out_dir` (else OUT), from IL_Replay payload texts.
    `units` / `mem_lag` / `visibility` / `trust` / `presses` / `decision` as --units /
    --mem-lag / --visibility / --trust (trust_rule) / --presses / --obs-delay (decision_rule),
    kept as UNITS / MEM_LAG / VISIBILITY / TRUST / PRESSES / DECISION for this process (None:
    keep the module's); main passes all six to every worker, as it passes `live`. Under presses
    "on" the shard also has act_press (int16, from each row's "press") and PRESS_FIELDS; with a
    delay (E11) DECISION_FIELDS. A delay with mem_lag "legacy" is refused (ValueError) before
    any match. Returns (batch_id, rows, skipped by reason); with `stats`, also the shard's
    trust_stats, which main merges into manifest.json."""
    global LIVE, OUT, UNITS, MEM_LAG, VISIBILITY, TRUST, PRESSES, DECISION
    units = UNITS if units is None else units
    unit_mode(units, live)
    lag = MEM_LAG if mem_lag is None else mem_lag_value(mem_lag)
    vis = visibility_mode(VISIBILITY if visibility is None else visibility)
    rule = trust_rule(trust)
    press = rs_presses.press_mode(PRESSES if presses is None else presses)
    drule = decision_rule(decision)
    if drule.on and lag == MEM_LEGACY:
        raise ValueError("--obs-delay: decision-time rows read the memory dated at execution; "
                         "--mem-lag legacy is the order before E08")
    LIVE, UNITS, MEM_LAG, VISIBILITY, TRUST, PRESSES = live, units, lag, vis, rule, press
    DECISION = drule
    if out_dir:
        OUT = Path(out_dir)
    ctx = _setup(engine)
    rows, tags, skipped, infos = [], [], {}, []
    for text in payload_texts:
        p = json.loads(text)
        tag = p["source"]["replay_tag"]
        info = {}
        try:
            r, why = rows_for(p, ctx, engine, info=info)
        except Exception as exc:
            r, why = None, f"error {type(exc).__name__}"
        if r is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        rows += r
        tags += [tag] * len(r)
        infos.append(info)
    if rows:
        press_on = PRESSES == rs_presses.PRESS_ON
        fields = (FIELDS + (VISION_FIELDS if UNITS == UNITS_VISION and not LIVE else ())
                  + (VISIBILITY_FIELDS if VISIBILITY == VIS_PUBLIC else ())
                  + (TRUST_FIELDS if TRUST.v2 else ()) + (PRESS_FIELDS if press_on else ())
                  + (DECISION_FIELDS if DECISION.on else ()))
        data = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in fields}
        act = np.asarray([r["action"] for r in rows], dtype=np.int16)
        data["act_card"], data["act_x"], data["act_y"] = act[:, 0], act[:, 1], act[:, 2]
        if press_on:
            data["act_press"] = np.asarray([r["press"] for r in rows], dtype=np.int16)
        data["match"] = np.asarray(tags)
        data["split"] = np.asarray([split_of(t) for t in tags])
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"shard-{batch_id:04d}.npz", **data)
    if stats:
        return batch_id, len(rows), skipped, trust_stats(rows, infos, TRUST)
    return batch_id, len(rows), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--others", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--limit-parts", type=int, default=0)
    ap.add_argument("--live-compat", action="store_true",
                    help="unit tokens limited to what the live reader provides")
    ap.add_argument("--units", choices=UNIT_MODES, default=UNITS_TYPES,
                    help="unit tokens: RoyaleSim unit types (default, as before) or the vision "
                         "classes of data/vision/classes.json, plus a unit_form column")
    add_mem_lag_arg(ap, aliases=("--opp-lag",))
    ap.add_argument("--obs-delay", type=int, default=0, metavar="D",
                    help="0 (default): each play row at the tick the play ran, as before; D > 0: "
                         "decision-time rows (E11), each play observed D ticks before it ran "
                         "(about 24 live: the client's 21-22 plus capture, inference and adb), "
                         "with own_pending columns and an honest hand_exact")
    ap.add_argument("--obs-delay-jitter", type=int, default=0, metavar="J",
                    help="--obs-delay: each play's delay is D plus an integer uniform in [-J, J], "
                         "seeded by the match tag (default 0; J <= D)")
    ap.add_argument("--wait-window", type=int, default=WAIT_WINDOW, metavar="W",
                    help="--obs-delay: no wait row where the side taps in [t, t + W] "
                         f"(default {WAIT_WINDOW})")
    ap.add_argument("--visibility", choices=VISIBILITY_MODES, default=VIS_LEGACY,
                    help="which units a row shows: 'legacy' (default) every living unit, as "
                         "before; 'public' what the acting player's screen shows (no invisible "
                         "or hidden enemy; a tunnelling unit kept, with a unit_burrow column)")
    ap.add_argument("--trust", choices=TRUST_MODES, default=TRUST_LEGACY,
                    help="'legacy' (default): the `trusted` column only, as before; 'v2': also "
                         "`trusted_v2`, cut where the engine's towers or game end leave the "
                         "recording, and no trust for waits after the last recorded play")
    ap.add_argument("--trust-sides", choices=TRUST_SIDES, default=SIDES_UNORDERED,
                    help="--trust v2: which engine princess a recorded princess_left is "
                         "(default 'unordered': either; see princess_votes in manifest.json)")
    ap.add_argument("--trust-hp-margin", type=float, default=TRUST_HP_MARGIN,
                    help="--trust v2: damage beyond the recording's a tower may take, as a share "
                         f"of its max hp (default {TRUST_HP_MARGIN})")
    ap.add_argument("--presses", choices=rs_presses.PRESS_MODES, default=rs_presses.PRESS_OFF,
                    help="'off' (default): ability presses are not replayed, as before; 'on': "
                         "replay activate_ability events as button commands, add press rows "
                         "(act_press) and ability_card / ability_state columns; a refused or "
                         "unplaceable press cuts trust")
    ap.add_argument("--out", type=Path, default=None)
    rs_engine.add_engine_args(ap)
    args = ap.parse_args()
    try:
        cfg = rs_engine.config_from_args(args).undelayed()
        unit_mode(args.units, args.live_compat)
        rule = trust_rule(TrustRule(args.trust, args.trust_sides, args.trust_hp_margin))
        decision = decision_rule(DecisionRule(args.obs_delay, args.obs_delay_jitter,
                                              args.wait_window))
        if decision.on and args.mem_lag == MEM_LEGACY:
            raise ValueError("--obs-delay needs the memory dated at execution: --mem-lag / "
                             "--opp-lag TICKS, not legacy")
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    rs_engine.set_active(cfg)
    global OUT, LIVE, UNITS, MEM_LAG, VISIBILITY, TRUST, PRESSES, DECISION
    LIVE = args.live_compat
    UNITS, MEM_LAG, VISIBILITY, TRUST = args.units, args.mem_lag, args.visibility, rule
    PRESSES = args.presses
    DECISION = decision
    if args.out:
        OUT = args.out
    import pyarrow.parquet as pq
    royalesim, names, ids, cat = _setup(cfg)
    vocab_battle = rs_engine.make_battle(cfg, royalesim)
    fp = rs_engine.fingerprint(vocab_battle, royalesim, cfg)
    print(f"engine: {cfg.describe()}\n        {rs_engine.summary(fp)}", flush=True)
    card_map = rs_cards.describe(names)
    print(f"cards:  {rs_cards.summary(names)}", flush=True)
    print(f"memory: {describe_memory(MEM_LAG)}", flush=True)
    print(f"seen:   {describe_visibility(VISIBILITY)}", flush=True)
    print(f"trust:  {describe_trust(TRUST)}", flush=True)
    print(f"presses: {rs_presses.describe_presses(PRESSES)}"
          + (f", {BUTTONS} buttons a side" if PRESSES == rs_presses.PRESS_ON else ""), flush=True)
    print(f"observed: {describe_decision(DECISION)}", flush=True)
    if UNITS == UNITS_VISION:
        print(f"units:  vision classes, {VCT.describe()}", flush=True)
        if VC.unknown:
            print(f"        {len(VC.unknown)} engine unit types have no class and read as "
                  f"unknown (token 1): {', '.join(VC.unknown[:12])}"
                  f"{' ...' if len(VC.unknown) > 12 else ''}", flush=True)
    t0 = time.time()
    hog, others = [], []
    parts = sorted((CORPUS / "replays").glob("part-*.parquet"))
    if args.limit_parts:
        parts = parts[:args.limit_parts]
    for part in parts:
        for r in pq.read_table(part, columns=["replay_tag", "game_mode", "payload_json"]).to_pylist():
            if r["game_mode"] not in MODES:
                continue
            p = json.loads(r["payload_json"])
            sides = []
            ok = True
            for side in ("team", "opponent"):
                pl = p["battle"][side]["players"]
                if len(pl) != 1:
                    ok = False
                    break
                sides.append(frozenset(names.get(base_key(c["card_key"])[0]) for c in pl[0]["deck"]))
            if not ok or any(None in s for s in sides):
                continue
            p.setdefault("source", {})["replay_tag"] = r["replay_tag"]
            text = json.dumps(p)
            if HOG26 in sides:
                hog.append(text)
            elif len(others) < args.others:
                others.append(text)
    picked = hog + others
    print(f"selected {len(hog):,} Hog 2.6 + {len(others):,} others ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    unit_types = json.loads(vocab_battle.unit_types_json())
    (OUT / "vocab.json").write_text(json.dumps(vocab_record(
        cat, unit_types, cfg, fp, card_map, LIVE,
        version=getattr(royalesim, "__version__", "?"),
        classes=VCT if UNITS == UNITS_VISION else None,
        **memory_vocab(MEM_LAG), **visibility_vocab(VISIBILITY), **trust_vocab(TRUST),
        **press_vocab(PRESSES), **decision_vocab(DECISION))),
        encoding="utf-8")
    batches = [picked[i:i + args.batch] for i in range(0, len(picked), args.batch)]
    total, skipped, totals, shards = 0, {}, {}, []
    with ProcessPoolExecutor(args.workers, initializer=rs_engine.set_active,
                             initargs=(cfg.to_json(),)) as pool:
        futs = [pool.submit(build, k, b, LIVE, str(OUT), cfg.to_json(), units=UNITS,
                            mem_lag=MEM_LAG, visibility=VISIBILITY, trust=TRUST.to_json(),
                            presses=PRESSES, stats=True, decision=DECISION.to_json())
                for k, b in enumerate(batches)]
        for n, f in enumerate(as_completed(futs), 1):
            bid, rows, sk, st = f.result()
            total += rows
            for k, v in sk.items():
                skipped[k] = skipped.get(k, 0) + v
            merge_stats(totals, st)
            if rows:                                  # build writes no shard without rows
                shards.append(shard_entry(bid, st))
            if n % 10 == 0 or n == len(batches):
                print(f"  {n}/{len(batches)} shards, {total:,} rows, skipped {skipped} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    man = manifest(totals, shards, skipped, TRUST)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"done: {total:,} rows -> {OUT}\n      {describe_manifest(man)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
