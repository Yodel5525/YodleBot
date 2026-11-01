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
plus void, rune-giant, spirit-empress and minion-giant). Matches with an unmapped card are
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
so --presses leaves the plays' taps alone), and never before the tap of the side's previous
play (one player taps in order, so the plays a row sees pending land no later than its own;
only jitter could swap two plays of a side). J <= D, and D + J < HORIZON. The engine still runs
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
E15 will): the legacy model trains on decision-time rows as they are, schema rs_rows/legacy;
only its --drop-pending-waits (a diagnosis option, off by default) reads own_pending, to leave
out the waits with an own play in flight (tools/rs_dt_diag.py counts them).

Schema v2 (E14; tools/rs_schema.py, which holds both layouts as data; tokens-14): `--schema
legacy` (the default) writes the rows above, schema rs_rows/legacy, so the rows, vocab.json and
manifest.json are byte-identical to the rows before E14. `--schema v2` writes rs_rows/v2: every
row has the same columns whatever the other flags (only --trust v2 still adds trusted_v2), and
build() casts each to rs_schema.V2's dtype (tick int32, side int8, next int16, outcome and won
float32). Against legacy:
  unit_feats    9 features: the 7 above (of the --units mode or --live-compat), then "form",
                0 base, 0.5 evolution, 1 hero (under --units vision the class table's, else
                the unit type's name, unit_form_table), and "burrow", 1 for a unit tunnelling
                now (status_flags read as under --visibility public: the engine's -1 is
                refused); there is no unit_form or unit_burrow column. --visibility public
                still leaves the unseen enemies out.
  scalars       7: the 4 above, then "triple" (3x elixir from 240 s, tick 4800, rs_elixir's
                law), "opp_elixir" (/ 10000) and "opp_exact": the opponent's bar as the acting
                side counts it from what it has seen (SeenElixir: one rs_elixir.ElixirCounter a
                side, priced from the catalogue; fed each play the engine ran, and under
                --presses on each press it took, once exec_tick + max(BOARD_LAG, L) <= t, L =
                --mem-lag: the plays the memory view holds), and whether that count is still
                known to be exact. A recorded play the engine refused never reached the board,
                so it is not counted (the memory still lists it, as before). Own elixir stays
                the raw engine bar (tokens-19): unpaid while a play of the side is in flight
                (tap < t <= exec) and paid when it runs. The live client pays its HUD bar
                from the tap + ~3, so the live runner shows the row min(HUD, its elixir cap)
                plus the cost of its taps in flight (scripts/vision_play.py, tools/
                public_state.LiveObserver.row_elixir); the decision still spends the HUD's.
  hand_feats    per hand slot: the card's form in players[t]["deck"] / ["forms"] / 2, 1 when
                its next play is evolved, and its evolution charge (players[t]["evo"]: plays
                since its last evolved play / the plays it needs, at most 1); hand_features.
                Zeros for a deck without evolutions or heroes (the bot's own, live).
  spell_ids, spell_feats
                up to 8 cast spells in flight (state["spells"] read through
                royalesim.SPELL_FIELDS, spell_index, as tools/rs_shadow.py reads them), nearest
                the own king first: the spell card + 2, and own, its centre and its aim in the
                acting player's frame (/ 18000, / 32000), ticks_flown / 40 within [0, 1];
                spell_features. Only objects of a SPELL card (SPELL_CARDS, catalogue
                card_kind): what the screen shows as a spell (vision_classes.KATACR_SPELLS),
                not the effects a troop or building leaves (an evolution's areas, a death bomb,
                a deploy blow, a hero ability).
  own_pending, own_pending_land
                always; zeros for rows at execution (E11 fills them at --obs-delay D > 0).
  ability_card, ability_state
                always, from players[t]["abilities"] as under --presses on (the engine reports
                its buttons whether or not presses are replayed).
  act_press     always; 0 on every row without --presses on.
  hand_exact    honest for every row (exact_hand), also at --obs-delay 0.
--mem-lag legacy is refused. vocab.json: "schema" rs_rows/v2, "schema_spec"
(rs_schema.V2.to_json(B)) and only trusted_v2 in "extra_columns"; rows_for's info and
manifest.json get "observation" counts (rows, opp_inexact, spell_rows). obs_row(...,
schema="v2", opp_elixir=SeenElixir.view(team, t)) gives the same row from any state dict, a
live one from tools/public_state.py included, once _setup has filled UNIT_FORMS, SPELL_CARDS
and SF. sim/train_tokens.py and the model read v2 rows from E15 on (the legacy trainer cannot:
unit_feats has 9 features and scalars 7); rs_eval and the live runner still build legacy rows.

Selection (E18; tokens-13): a match of IL_Replay's replays/*.parquet is a candidate when its
game_mode is in MODES, each side has one player and rs_cards maps every deck card (scan_part).
Every Hog 2.6 candidate is taken. `--others N` (default 20000) adds non-Hog candidates:
`--others-pick first` (the default) the first N in parquet order, as before, so the shards are
the rows of before E18; `--others-pick random` N drawn uniformly without replacement
(select_rng(--select-seed), default seed 0) and taken in parquet order; `--all-matches` every
candidate (--others is then not read). select_matches gives the order: Hog 2.6 first, then the
others, each in parquet order, and shard k holds its k-th --batch matches, as before. The
corpus is read twice: scan_part keeps only (part, row, tag, hog) of each candidate, then
selected_payloads re-reads one part at a time for the chosen rows' payloads, and at most
2 x --workers batches are in flight (bounded), so the builder holds about one part and a few
batches of payloads, not the whole selection (every match would not fit in 16 GB). vocab.json
gets "selection" (selection_record) unless the pick is the legacy one.

Wait thinning (E18): `--wait-keep-at-build p` keeps each wait row of a match (act_card 0 and no
press) with probability p: one draw per row of the match, in row order, from
default_rng([tap_seed(tag), STREAM_WAITS]) (thin_waits), so a match keeps the same rows whatever
shard or worker builds it; play and press rows are all kept. vocab.json "wait_keep_at_build" is
p (sim/train_tokens.py then thins only what is left: resolve_wait_keep) and "wait_thinning" the
rule (wait_rule); manifest.json "wait_thinning" has p and the waits kept and dropped, and its
other counts are of the rows the shards hold (the matches' "decision" and "observation" info
counts are of their rows before the thinning). Without the flag nothing is thinned and
vocab.json says None, as before. tools/rs_rows_compact.py turns the shards into the per-column
files sim/train_tokens.py --memmap reads.

Perception noise (E17; tools/perception_noise.py): without `--noise` (the default, or `--noise
none`) a row reads the engine's state as it is, so the rows, vocab.json and manifest.json are
byte-identical to the rows before E17. `--noise PROFILE` (a profile JSON, or "default":
data/noise/default_prior.json; `--noise-only KINDS` keeps some of its kinds, `--noise-seed N` is
the run seed) observes every row through one perception_noise.MatchNoise per match, seeded by
match_seed(replay tag, N), so a match gets the same noise whatever shard or worker builds it. Its
model is bound once per process to the Battle's unit types and catalogue, F and the sub-tile
scale (noise_model). Each row (play, press and wait rows, at execution or at the tap) is obs_row
of observe(state, acting team), the state that side's screen reader would report (units dropped,
moved, confused, collapsed to the base form, phantoms, bars, towers, own elixir, ...), with the
tick its clock reads (tick + tick_shift: the time scalars and opp_last_age; ages floor at 0); the
row keeps the true "tick", from which the outcome window, the trust cuts and the wait sampling
count. A recorded play goes into the memory as the OTHER seat perceives it (play_seen: missed, a
few ticks late, or as the unknown card, token 1); under --schema v2 the opponent's counted elixir
is fed what that memory got (a play it missed or could not name is not counted, a late one is
counted when seen), so opp_elixir agrees with opp_seen / opp_last as it does in tools/rs_eval.py.
The engine still runs every recorded play, own hands, own pending plays and hand_exact stay
exact, and presses are seen as they ran. rows_for's info and manifest.json get "noise" (the
MatchNoise counts: observations, units, dropped, plays, plays_missed, ...; like "decision" and
"observation", counted before any wait thinning); vocab.json gets
"perception_noise" (perception_noise.vocab_entry: schema, name, digest, seed, kinds and the whole
profile), which sim/train_tokens.py copies into every checkpoint's meta.
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass
from pathlib import Path
from statistics import mode

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from royalesim_redrive import CORPUS, base_key, deal_order  # noqa: E402
import perception_noise as PN  # noqa: E402  (E17: numpy, rs_engine and vision_classes only)
import rs_cards  # noqa: E402
import rs_elixir  # noqa: E402
import rs_engine  # noqa: E402
import rs_memory  # noqa: E402
import rs_orient  # noqa: E402
import rs_presses  # noqa: E402
import rs_schema  # noqa: E402
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

#: --schema (module doc, "Schema v2"): the row layout, rs_schema.SCHEMA_LEGACY (the rows before
#: E14) or rs_schema.SCHEMA_V2; build and main set it like the settings above.
ROW_SCHEMA = rs_schema.SCHEMA_LEGACY
#: The keys of a v2 obs_row row, in order, and its features per unit.
V2_OBS_FIELDS = rs_schema.V2.obs_fields()
V2_UF = rs_schema.V2.unit_features
#: Where "opp_exact" sits in a v2 row's scalars.
V2_OPP_EXACT = rs_schema.V2.index("scalars", "opp_exact")
#: royalesim.SPELL_FIELDS (name -> column) for state["spells"] rows; _setup fills it (spell_index),
#: None reads them in royalegym protocol.SpellState's order, SPELL_FIELDS_FALLBACK.
SF = None
SPELL_FIELDS_FALLBACK = ("team", "card_id", "motion", "x", "y", "aim_x", "aim_y", "delay_ticks",
                         "travelled", "length", "hits", "ticks_flown")
#: The spell columns a v2 row cannot do without (ticks_flown is optional: -1 / absent reads 0).
SPELL_REQUIRED = ("team", "card_id", "x", "y", "aim_x", "aim_y")
#: v2 unit "form" by unit_type for --units types and --live-compat rows (unit_form_table of
#: this Battle's unit_types_json()); _setup fills it.
UNIT_FORMS = None
#: The catalogue ids of the SPELL cards (card_kind, catalogue column CARD_KIND_COLUMN): the spell
#: objects a v2 row shows (spell_features); _setup fills it. tools/rs_orient.py reads the column
#: at the same place, as CARD_BUILDING above does.
SPELL_CARDS = None
CARD_KIND_COLUMN = 8

#: --wait-keep-at-build (module doc, "Wait thinning"): None, no thinning (the rows before E18), or
#: the share of each match's wait rows a shard keeps; build and main set it like the settings above.
WAIT_KEEP_AT_BUILD = None
#: --others-pick (module doc, "Selection"): the first --others non-Hog candidates in parquet order
#: (the rows before E18) or a seeded uniform draw of them.
PICK_FIRST, PICK_RANDOM = "first", "random"
PICK_MODES = (PICK_FIRST, PICK_RANDOM)
#: select_rng's first seed word, so --select-seed 0 is not numpy's plain seed 0.
SELECT_SALT = 18
#: The parquet columns the scan (scan_part) and the payload pass (selected_payloads) read.
SCAN_COLUMNS = ("replay_tag", "game_mode", "payload_json")
PAYLOAD_COLUMNS = ("replay_tag", "payload_json")

#: --noise (module doc, "Perception noise"): None, a perfect observer (the rows before E17), or the
#: perception_noise.NoiseProfile every row is observed through; NOISE_SEED is --noise-seed. build
#: and main set them like the settings above.
NOISE = None
NOISE_SEED = 0
#: noise_model's NoiseModels of this process, by profile digest, sub, field index and vocabulary.
_NOISE_MODELS: dict = {}


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
#: tap_delays' rng streams, one per kind of action, and thin_waits' (E18).
STREAM_PLAYS, STREAM_PRESSES = 0, 1
STREAM_WAITS = 2
#: decision_rule: the settings a DecisionRule's JSON carries, and the keys that only describe it.
_DECISION_KEYS = ("delay", "jitter", "wait_window")
_DECISION_NOTES = ("seed", "streams", "tap_floor", "tap_order", "pending_slots",
                   "pending_scale", "pending_rule", "hand_exact")


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
                "tap_floor": 0,
                "tap_order": "per side in execution order: never before the previous play's tap",
                "pending_slots": PENDING_SLOTS, "pending_scale": PENDING_SCALE,
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
        raw = [int(q[0]) - d for q, d in zip(queue, delays)]
        self.play_clamped = [t < 0 for t in raw]
        # One player taps in order: a play's tap is never before the tap of its side's previous
        # play (queue order), so the plays a row sees pending all land no later than its own,
        # as for the live runner's own sent plays. Without jitter that holds anyway; with it,
        # two plays of a side within 2 J ticks could otherwise swap.
        self.play_taps, last_tap = [], {}
        for q, t in zip(queue, raw):
            t = max(0, t, last_tap.get(int(q[1]), 0))
            last_tap[int(q[1])] = t
            self.play_taps.append(t)
        self.press_taps = [max(0, int(p[0]) - d) for p, d in zip(presses, pdelays)]
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


# ---------------------------------------------------------------------------
# schema v2 (E14; module doc, "Schema v2")
# ---------------------------------------------------------------------------

def schema_mode(value) -> str:
    """--schema as obs_row, rows_for, build and vocab_record take it: "legacy" / "v2", a full
    vocab.json name or an rs_schema.Schema, as its vocab.json name (rs_schema.schema_name).
    ValueError (rs_schema.SchemaError) otherwise."""
    return rs_schema.schema_name(value)


def describe_schema(schema) -> str:
    """One line for the console."""
    name = schema_mode(schema)
    if name == rs_schema.SCHEMA_LEGACY:
        return f"{name} (the rows before E14)"
    return (f"{name} (spells in flight, the opponent's counted elixir, own pending plays, hand "
            f"evolution, {BUTTONS} ability buttons, unit form and burrow; "
            f"{rs_schema.V2.row_bytes(BUTTONS):,} bytes a row)")


def spell_index(royalesim=None) -> dict:
    """royalesim.SPELL_FIELDS as name -> column; SPELL_FIELDS_FALLBACK's when the engine does
    not export them (royalegym checks that the engine's list is a prefix of that order)."""
    names = getattr(royalesim, "SPELL_FIELDS", None) if royalesim is not None else None
    return {str(n): i for i, n in enumerate(names or SPELL_FIELDS_FALLBACK)}


def unit_form_table(unit_types) -> tuple:
    """The v2 unit "form" of each name of unit_types_json(), in order: 0 base, 0.5 evolution
    (an _EV<n> name), 1 hero (a ...Hero name), by vision_classes.split_form's name rules (the
    rules data/vision/classes.json was built with)."""
    names = [str(n) for n in unit_types]
    index = vision_classes.NameIndex(names)
    return tuple(vision_classes.split_form(n, index)[1] / vision_classes.FORM_HERO for n in names)


def unit_form(unit) -> float:
    """UNIT_FORMS of unit_type `unit`; 0.0 for -1 (not reported) and an unknown index."""
    u = int(unit)
    return UNIT_FORMS[u] if 0 <= u < len(UNIT_FORMS) else 0.0


def _opp_elixir(value) -> tuple:
    """obs_row's opp_elixir as (milli, exact): a (milli, exact) pair, or milli alone (exact)."""
    if value is None:
        raise ValueError("--schema v2: obs_row needs opp_elixir, the (milli, exact) the acting "
                         "side counts for its opponent (SeenElixir.view)")
    if isinstance(value, (tuple, list)):
        milli, exact = value
    else:
        milli, exact = value, True
    return int(milli), bool(exact)


def hand_features(player) -> np.ndarray:
    """v2 hand_feats (rs_schema.HAND_FEATS), float16 [4, 3], one row per hand slot of `player`
    (a state_json player): the card's form in the deck ("deck" / "forms"; an "evo" row alone
    reads as an evolution) / 2, 1 when its next play is evolved, and its evolution charge,
    plays since its last evolved play / plays it needs (at most 1; the ready flag when it needs
    none). Zeros for an empty slot and for a player that reports none of these (the live HUD:
    the bot's own cards are all base)."""
    forms = {int(c): int(f) for c, f in zip(player.get("deck") or (), player.get("forms") or ())}
    evo = {int(r[0]): [int(v) for v in r] for r in (player.get("evo") or ()) if len(r)}
    out = np.zeros((rs_schema.HAND_SIZE, len(rs_schema.HAND_FEATS)), dtype=np.float16)
    for k, card in enumerate(list(player.get("hand") or ())[:rs_schema.HAND_SIZE]):
        card = int(card)
        if card < 0:
            continue
        row = evo.get(card)
        ready = progress = 0.0
        if row is not None:
            since = row[1] if len(row) > 1 else 0
            ready = 1.0 if len(row) > 2 and row[2] else 0.0
            need = row[3] if len(row) > 3 else 0
            progress = min(1.0, max(0.0, since / need)) if need > 0 else ready
        form = forms.get(card, vision_classes.FORM_EVO if row is not None else 0)
        out[k] = (form / vision_classes.FORM_HERO, ready, progress)
    return out


def spell_cards(cat) -> frozenset:
    """The catalogue ids of the SPELL cards in catalogue_json() rows `cat` (card_kind,
    CARD_KIND_COLUMN): SPELL_CARDS."""
    return frozenset(i for i, row in enumerate(cat)
                     if len(row) > CARD_KIND_COLUMN and row[CARD_KIND_COLUMN] == "SPELL")


def spell_features(state, team, sub, king, cards=None) -> tuple:
    """v2 (spell_ids int16 [8], spell_feats float16 [8, 6]) of `team`'s view: the objects of
    state["spells"] (read through SF, else SPELL_FIELDS_FALLBACK's order) whose card is a SPELL
    card (`cards`, None: SPELL_CARDS; module doc, "Schema v2"), nearest `king` ((x, y) engine
    subtiles, the point obs_row sorts the units from) first, the first rs_schema.SPELL_SLOTS.
    Each: its card + 2, and (rs_schema.SPELL_FEATS) own, its centre and its aim in `team`'s
    frame (/ 18000, / 32000), ticks_flown / 40 within [0, 1] (0 when not reported, -1).
    RuntimeError before _setup has filled SPELL_CARDS; rs_engine.EngineError when SF lacks a
    SPELL_REQUIRED name."""
    cards = SPELL_CARDS if cards is None else cards
    if cards is None:
        raise RuntimeError("--schema v2 rows need the spell cards: _setup() fills SPELL_CARDS "
                           "from the engine's catalogue")
    sf = SF if SF is not None else spell_index()
    missing = [n for n in SPELL_REQUIRED if n not in sf]
    if missing:
        raise rs_engine.EngineError(f"royalesim.SPELL_FIELDS {list(sf)} has no {missing}: "
                                    "--schema v2 reads them")
    kx, ky = king
    spells = sorted((list(sp) for sp in (state.get("spells") or ())
                     if int(sp[sf["card_id"]]) in cards),
                    key=lambda sp: (sp[sf["x"]] - kx) ** 2 + (sp[sf["y"]] - ky) ** 2)
    ids = np.zeros(rs_schema.SPELL_SLOTS, dtype=np.int16)
    feats = np.zeros((rs_schema.SPELL_SLOTS, len(rs_schema.SPELL_FEATS)), dtype=np.float16)
    at_flown = sf.get("ticks_flown")
    for k, sp in enumerate(spells[:rs_schema.SPELL_SLOTS]):
        ids[k] = int(sp[sf["card_id"]]) + 2
        x, y = own(sp[sf["x"]], sp[sf["y"]], team, sub)
        ax, ay = own(sp[sf["aim_x"]], sp[sf["aim_y"]], team, sub)
        flown = int(sp[at_flown]) if at_flown is not None and len(sp) > at_flown else -1
        feats[k] = (1.0 if sp[sf["team"]] == team else 0.0, x / 18_000, y / 32_000,
                    ax / 18_000, ay / 32_000,
                    min(1.0, flown / rs_schema.SPELL_FLOWN_SCALE) if flown > 0 else 0.0)
    return ids, feats


def press_cost(player, button: int) -> int:
    """The elixir `player`'s ability button `button` costs: its abilities row's cost column
    (royalegym AbilityRow: available, spent, cost, ...); 0 when the engine sends no row."""
    rows = list(player.get("abilities") or ())
    if 0 <= int(button) < len(rows) and len(rows[int(button)]) > 2:
        return max(0, int(rows[int(button)][2]))
    return 0


def elixir_law(engine=None) -> rs_elixir.ElixirLaw:
    """The elixir law of the battles `engine` builds (rs_engine.resolve): royalesim's
    calibration under the config's calibration_overrides (as tools/rs_match.py counts it)."""
    cfg = rs_engine.resolve(engine)
    if cfg.calibration_overrides:
        return rs_elixir.ElixirLaw.load(overrides=dict(cfg.calibration_overrides))
    return rs_elixir.default_law()


class SeenElixir:
    """E14: each side's elixir as the other side counts it from the plays it has seen.

    One rs_elixir.ElixirCounter per side (seeded with START_MANA at tick 0, paid when a play
    RUNS, regenerated by the law, capped, a Mirror priced as its copy + 1), fed only the events
    its observer knows: view(observer, t) first feeds the observer's opponent every recorded
    event with exec_tick + lag <= t, then returns (elixir_milli(t), exact). With lag >=
    BOARD_LAG (rows_for: max(BOARD_LAG, --mem-lag), the memory view's) no event dated t is fed
    at t, so that is the bar the engine's state(t) shows once every play has been seen (and the
    costs are the engine's: no Collector or Elixir Golem payouts, which rs_elixir does not
    model); a play still unseen at t is not paid yet at t, and once seen it is paid at its own
    tick (rs_elixir re-folds the count). Live, where perception records a play when it first
    sees it, lag 0. Queries in nondecreasing tick order give each observer's view.

    record(team, exec_tick, card) logs `team`'s play of catalogue card `card` (cost= overrides
    the catalogue price); press(team, exec_tick, cost) a press that ran. rows_for records only
    what the engine ran (step results "OK").
    """

    def __init__(self, catalogue, *, fields=None, law=None, lag: int = rs_memory.BOARD_LAG):
        self.lag = _ticks(lag, "SeenElixir lag")
        self.counters = {t: rs_elixir.ElixirCounter.from_catalogue(catalogue, fields=fields,
                                                                   law=law) for t in (0, 1)}
        self._waiting: dict = {0: [], 1: []}       # events recorded and not fed yet

    def _team(self, team) -> int:
        team = _ticks(team, "team")
        if team not in (0, 1):
            raise ValueError(f"team must be 0 or 1, got {team}")
        return team

    def record(self, team, exec_tick, card, cost=None) -> None:
        self._waiting[self._team(team)].append(
            (_ticks(exec_tick, "exec_tick"), "play", int(card), cost))

    def press(self, team, exec_tick, cost) -> None:
        self._waiting[self._team(team)].append(
            (_ticks(exec_tick, "exec_tick"), "press", None, int(cost)))

    def view(self, observer, tick) -> tuple:
        """(milli, exact): `observer`'s count of its opponent's bar at `tick`."""
        side = 1 - self._team(observer)
        tick = int(tick)
        counter, waiting = self.counters[side], self._waiting[side]
        if waiting:
            due = [e for e in waiting if e[0] + self.lag <= tick]
            if due:
                self._waiting[side] = [e for e in waiting if e[0] + self.lag > tick]
                for when, kind, card, cost in due:
                    if kind == "play":
                        counter.play(when, card, cost=cost)
                    else:
                        counter.press(when, cost)
        return counter.elixir_milli(tick), counter.exact


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
    --obs-delay > 0), also "decision": those counts summed; "observation" (E14, --schema v2)
    and "noise" (E17, --noise) likewise."""
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
    if any("observation" in i for i in infos):          # E14, --schema v2
        observation: dict = {}
        for i in infos:
            merge_stats(observation, i.get("observation", {}))
        st["observation"] = observation
    if any("noise" in i for i in infos):                # E17, --noise
        noise: dict = {}
        for i in infos:
            merge_stats(noise, i.get("noise", {}))
        st["noise"] = noise
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


def manifest(totals: dict, shards, skipped, rule=None, wait_keep=None) -> dict:
    """manifest.json: the run's merged trust_stats `totals` as counts and rates (a share of the
    rows; None with no rows), wait_rows, the skipped matches by reason, the trust rule and the
    shard entries (shard_entry, in shard order). Under --trust v2 also trusted_v2 and its
    rates, cuts_v2 and princess_votes; with press counts in `totals` (--presses on) also
    "presses"; with decision counts (--obs-delay > 0) also "decision"; with noise counts (E17,
    --noise) also "noise". With `wait_keep` (--wait-keep-at-build, E18) also "wait_thinning":
    keep, and the waits kept and dropped (build's counts in `totals`)."""
    rule = trust_rule(rule)
    wait_keep = wait_keep_value(wait_keep)
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
    if "observation" in totals:
        out["observation"] = totals["observation"]
    if "noise" in totals:
        out["noise"] = totals["noise"]
    if wait_keep is not None:
        thin = totals.get("wait_thinning", {})
        out["wait_thinning"] = {"keep": wait_keep, "kept": int(thin.get("kept", 0)),
                                "dropped": int(thin.get("dropped", 0))}
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
    if "observation" in man:
        o = man["observation"]
        text += (f"; schema v2: {o.get('spell_rows', 0):,} of {o.get('rows', 0):,} rows show a "
                 f"spell in flight, {o.get('opp_inexact', 0):,} count the opponent's elixir "
                 f"inexactly")
    if "wait_thinning" in man:
        w = man["wait_thinning"]
        text += (f"; waits kept at build with p {w['keep']:g}: {w['kept']:,} kept, "
                 f"{w['dropped']:,} dropped")
    if "noise" in man:
        n = man["noise"]
        text += (f"; noise: {n.get('dropped', 0) + n.get('hidden_new', 0):,} of "
                 f"{n.get('units', 0):,} units unseen, {n.get('phantoms', 0):,} phantoms, "
                 f"{n.get('plays_missed', 0):,} of {n.get('plays', 0):,} plays missed")
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


# ---------------------------------------------------------------------------
# E18: which matches (module doc, "Selection"), and wait thinning at build
# ---------------------------------------------------------------------------

def others_pick(value) -> str:
    """--others-pick as select_matches takes it: one of PICK_MODES; ValueError otherwise."""
    if value not in PICK_MODES:
        raise ValueError(f"--others-pick: one of {', '.join(PICK_MODES)}, got {value!r}")
    return value


def _nonneg(value, what: str) -> int:
    """`value` as an int >= 0 (numpy integers too); ValueError otherwise."""
    if isinstance(value, bool):
        raise ValueError(f"{what}: an int >= 0, got {value!r}")
    try:
        n = operator.index(value)
    except TypeError:
        raise ValueError(f"{what}: an int >= 0, got {value!r}") from None
    if n < 0:
        raise ValueError(f"{what} must be >= 0, got {n}")
    return int(n)


def select_rng(seed: int = 0) -> np.random.Generator:
    """The generator --others-pick random draws from: default_rng([SELECT_SALT, seed])."""
    return np.random.default_rng([SELECT_SALT, _nonneg(seed, "--select-seed")])


def match_decks(payload: dict, names: dict):
    """(team deck, opponent deck) as frozensets of catalogue names, or None for a match the rows
    are not built from: a side without exactly one player, or a card rs_cards does not map
    (`names`: IL_Replay key -> catalogue name, rs_cards.load())."""
    decks = []
    for side in ("team", "opponent"):
        players = payload["battle"][side]["players"]
        if len(players) != 1:
            return None
        decks.append(frozenset(names.get(base_key(c["card_key"])[0]) for c in players[0]["deck"]))
    if any(None in d for d in decks):
        return None
    return tuple(decks)


def scan_part(rows, names) -> list:
    """The candidates among one parquet part's rows (dicts with SCAN_COLUMNS, in table order):
    (row index, replay tag, hog) for each match whose game_mode is in MODES and whose decks
    match_decks reads, hog when either side plays Hog 2.6. The payloads are parsed, not kept."""
    out = []
    for k, r in enumerate(rows):
        if r["game_mode"] not in MODES:
            continue
        decks = match_decks(json.loads(r["payload_json"]), names)
        if decks is not None:
            out.append((k, r["replay_tag"], HOG26 in decks))
    return out


def select_matches(cands, others: int = 20000, pick: str = PICK_FIRST, seed: int = 0,
                   all_matches: bool = False) -> list:
    """The candidates the rows are built from, in shard order: every Hog 2.6 one, then the
    others, both in parquet order (`cands`: (part, row, tag, hog) in parquet order). pick
    "first": the first `others` non-Hog candidates (main before E18); "random": `others` of
    them drawn uniformly without replacement with select_rng(seed); all_matches: every one, and
    `others` is not read. Fewer non-Hog candidates than `others` gives them all."""
    pick = others_pick(pick)
    hog = [c for c in cands if c[3]]
    rest = [c for c in cands if not c[3]]
    if all_matches:
        return hog + rest
    n = min(_nonneg(others, "--others"), len(rest))
    if pick == PICK_FIRST:
        return hog + rest[:n]
    idx = np.sort(select_rng(seed).choice(len(rest), n, replace=False)) if n else ()
    return hog + [rest[int(i)] for i in idx]


def selection_record(picked, candidates: int, parts: int, others: int, pick: str, seed: int,
                     all_matches: bool):
    """vocab.json "selection" for select_matches' `picked` out of `candidates` matches in
    `parts` parquet parts, or None for the legacy pick (the first --others, no --all-matches),
    whose vocab.json has no such key."""
    pick = others_pick(pick)
    if pick == PICK_FIRST and not all_matches:
        return None
    hog = sum(1 for c in picked if c[3])
    rec = {"pick": "all" if all_matches else pick,
           "others": None if all_matches else int(others),
           "seed": int(seed) if pick == PICK_RANDOM and not all_matches else None,
           "parts": int(parts), "candidates": int(candidates), "hog": hog,
           "other_candidates": int(candidates) - hog, "others_taken": len(picked) - hog,
           "order": "Hog 2.6 first, then the others, each in parquet order"}
    if rec["seed"] is not None:
        rec["rng"] = (f"numpy default_rng([{SELECT_SALT}, seed]).choice(other candidates, "
                      "others, replace=False), sorted")
    return rec


def _read_part(path, columns) -> list:
    """One parquet part's rows as dicts of `columns`, in table order."""
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=list(columns)).to_pylist()


def selected_payloads(picked, parts, read_part=None):
    """The payload text build() takes for each picked (part, row, tag, hog), in order: the
    payload with source.replay_tag set, as main always passed it. `read_part(path, columns)`
    (default: pyarrow) reads parts[part] when the order comes to it, and the part is dropped at
    the next one, so Hog first and the others after read each part at most twice. RuntimeError
    when a row's tag is not the one scanned (the file changed under the run)."""
    read = _read_part if read_part is None else read_part
    held, rows = None, None
    for part, row, tag, _ in picked:
        if part != held:
            rows = None                       # let the last part go before reading the next
            rows = read(parts[part], PAYLOAD_COLUMNS)
            held = part
        r = rows[row]
        if r["replay_tag"] != tag:
            raise RuntimeError(f"{parts[part]}: row {row} is {r['replay_tag']!r}, scanned as "
                               f"{tag!r}; the corpus changed during the run")
        p = json.loads(r["payload_json"])
        p.setdefault("source", {})["replay_tag"] = r["replay_tag"]
        yield json.dumps(p)


def chunked(items, size: int):
    """`items` in lists of `size` (the last one may be shorter), lazily."""
    if size < 1:
        raise ValueError(f"--batch must be >= 1, got {size}")
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def bounded(submit, jobs, limit: int):
    """submit(k, job) for each of `jobs` in order (k from 0), with at most `limit` futures not
    done at a time; yields each future once it is done, in the order they finish."""
    limit = max(1, int(limit))
    pending: set = set()
    for k, job in enumerate(jobs):
        pending.add(submit(k, job))
        if len(pending) >= limit:
            done, pending = wait_futures(pending, return_when=FIRST_COMPLETED)
            yield from done
    while pending:
        done, pending = wait_futures(pending, return_when=FIRST_COMPLETED)
        yield from done


def wait_keep_value(value):
    """--wait-keep-at-build as build and main take it: None (no thinning) or a share in (0, 1];
    ValueError otherwise."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"--wait-keep-at-build: a share in (0, 1], got {value!r}")
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"--wait-keep-at-build: a share in (0, 1], got {value!r}") from None
    if not 0.0 < p <= 1.0:
        raise ValueError(f"--wait-keep-at-build: a share in (0, 1], got {value!r}")
    return p


def wait_rule(keep) -> dict | None:
    """vocab.json "wait_thinning" of rows thinned with `keep` (wait_keep_value); None for none."""
    p = wait_keep_value(keep)
    if p is None:
        return None
    return {"keep": p, "waits": "rows with act_card 0 and no press",
            "seed": "int(sha1(replay_tag)[:16], 16)", "stream": STREAM_WAITS,
            "draws": "one per row of the match, in row order; a wait is kept when its draw < keep"}


def is_wait(row) -> bool:
    """A wait row of rows_for: no card (action card 0) and no press ("press" 0 or absent)."""
    return not row["action"][0] and not row.get("press", 0)


def thin_waits(rows, tag, keep) -> tuple:
    """(the rows of one match a shard keeps, the waits dropped) under --wait-keep-at-build
    `keep` (module doc, "Wait thinning"): all of them when `keep` is None; otherwise a wait row
    (is_wait) is kept when its draw is below `keep`, one draw per row in row order from
    default_rng([tap_seed(tag), STREAM_WAITS]), and play and press rows are always kept. The
    rows keep their order."""
    p = wait_keep_value(keep)
    if p is None or not rows:
        return list(rows), 0
    draws = np.random.default_rng([tap_seed(tag), STREAM_WAITS]).random(len(rows))
    kept = [r for r, u in zip(rows, draws) if not is_wait(r) or u < p]
    return kept, len(rows) - len(kept)


# ---------------------------------------------------------------------------
# E17: perception noise (module doc, "Perception noise")
# ---------------------------------------------------------------------------

def noise_profile(value):
    """--noise as rows_for, build and tools/rs_eval.py take it: None, or a word for no noise
    ("none", "off", ...: perception_noise.NO_NOISE), gives None, a perfect observer; anything
    else a perception_noise.NoiseProfile: a profile (also one of another import of that module,
    read through its to_dict()), its JSON dict, a profile file or "default" (the shipped prior).
    An all-zero profile stays a profile (the hooks run and change nothing). ValueError /
    FileNotFoundError as NoiseProfile.load raises them; TypeError for anything else."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in PN.NO_NOISE:
        return None
    if isinstance(value, PN.NoiseProfile):
        return value
    if not isinstance(value, (str, Path, dict)) and callable(getattr(value, "to_dict", None)):
        value = value.to_dict()
    return PN.as_profile(value)


def noise_seed_value(value) -> int:
    """--noise-seed as rows_for, build and tools/rs_eval.py take it: an int (numpy integers too;
    negative ones are fine, perception_noise.match_seed hashes it). ValueError otherwise."""
    if isinstance(value, bool):
        raise ValueError(f"--noise-seed: an int, got {value!r}")
    try:
        return int(operator.index(value))
    except TypeError:
        raise ValueError(f"--noise-seed: an int, got {value!r}") from None


def noise_model(profile, battle, sub):
    """The perception_noise.NoiseModel of `profile` (noise_profile; None gives None) bound to
    `battle`'s unit_types_json() and catalogue_json(), the field index F (_setup) and `sub`
    (royalesim.SUBTILE_PER_MILLITILE). Built once per profile and vocabulary in a process
    (_NOISE_MODELS): binding reads the vision class table. Raises what NoiseModel raises (a kind
    the engine cannot bind); the names it could not resolve are in its `unresolved`."""
    profile = noise_profile(profile)
    if profile is None:
        return None
    if F is None:
        raise RuntimeError("--noise: _setup() fills the field index F before a model is bound")
    units, cards = battle.unit_types_json(), battle.catalogue_json()
    vocab = hashlib.sha1(f"{units}\n{cards}".encode("utf-8")).hexdigest()
    key = (profile.digest(), int(sub), tuple(sorted(F.items())), vocab)
    model = _NOISE_MODELS.get(key)
    if model is None:
        model = _NOISE_MODELS[key] = PN.NoiseModel(
            profile, unit_types=json.loads(units), catalogue=json.loads(cards), fields=F,
            sub=int(sub))
    return model


def noise_vocab(profile, seed=0) -> dict:
    """vocab_record keywords for --noise: none for no noise (the vocab.json of the rows before
    E17), else {"perception_noise": perception_noise.vocab_entry(profile, seed)}."""
    profile = noise_profile(profile)
    if profile is None:
        return {}
    return {"perception_noise": PN.vocab_entry(profile, noise_seed_value(seed))}


def describe_noise(profile, seed=0) -> str:
    """For the console: the profile's kinds (NoiseProfile.describe) and the run seed."""
    profile = noise_profile(profile)
    if profile is None:
        return "none (a perfect observer, as before)"
    return f"{profile.describe()}\n        seeded per match by its replay tag, run seed {seed}"


def _setup(engine=None):
    """(royalesim, IL_Replay key -> card name, card name -> id, catalogue rows) under
    `engine` (an rs_engine.EngineConfig or its to_json(); None = rs_engine.active()).
    The card map is rs_cards.load(). Also sets F, after checking royalesim.ENTITY_FIELDS
    (rs_engine.field_index), and under --units vision (UNITS) VCT, the class table, and VC,
    its lookup of this Battle's unit_types_json(). Under --visibility public (VISIBILITY) it
    first checks royalesim.STATUS_BITS (check_status_bits). BUTTONS and HAND_SIZE come from
    royalesim.ABILITY_BUTTONS / HAND_SIZE (rs_presses.button_count / hand_size). For --schema v2
    rows (E14) it also fills SF (spell_index), UNIT_FORMS (unit_form_table) and SPELL_CARDS
    (spell_cards), whatever the settings (the live runner's rows need them too), and under v2
    (ROW_SCHEMA) it checks STATUS_BITS, since every v2 row reads the burrow bit."""
    global F, VC, VCT, BUTTONS, HAND_SIZE, SF, UNIT_FORMS, SPELL_CARDS
    import royalesim
    if VISIBILITY == VIS_PUBLIC or ROW_SCHEMA == rs_schema.SCHEMA_V2:
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
    unit_types = json.loads(b.unit_types_json())
    SF, UNIT_FORMS = spell_index(royalesim), unit_form_table(unit_types)
    SPELL_CARDS = spell_cards(cat)
    if UNITS == UNITS_VISION:
        VCT = vision_classes.load()
        VC = VCT.lookup(unit_types)
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
            buttons=None, pending=None, hand_exact=None, schema=None, opp_elixir=None):
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
    is the row's flag (None: True, as the rows before E11 say).

    `schema` (None: ROW_SCHEMA) "v2" (E14) gives the row of rs_schema.V2, keys in its
    obs_fields order (module doc, "Schema v2"): `opp_elixir` is then required, the (milli,
    exact) count of the opponent's bar the acting side holds (SeenElixir.view; an int alone is
    exact), `pending` None reads as [], the ability columns are there whatever `presses` says,
    and unit_form / unit_burrow go into unit_feats (UNIT_FORMS must be filled, by _setup, for
    the types and live tokens)."""
    live = LIVE if live is None else live
    units = unit_mode(UNITS if units is None else units)
    v2 = schema_mode(ROW_SCHEMA if schema is None else schema) == rs_schema.SCHEMA_V2
    public = visibility_mode(VISIBILITY if visibility is None else visibility) == VIS_PUBLIC
    press = rs_presses.press_mode(PRESSES if presses is None else presses) == rs_presses.PRESS_ON
    vision = not live and units == UNITS_VISION
    if vision and VC is None:
        raise RuntimeError("--units vision rows need the class lookup: set UNITS = 'vision' "
                           "before _setup(), which fills VC")
    if v2:
        opp_milli, opp_exact = _opp_elixir(opp_elixir)
        if not vision and UNIT_FORMS is None:
            raise RuntimeError("--schema v2 rows need the unit form table: _setup() fills "
                               "UNIT_FORMS from the engine's unit types")
    forms = np.zeros(MAX_UNITS, dtype=np.float16) if vision and not v2 else None
    burrow = np.zeros(MAX_UNITS, dtype=np.float16) if public and not v2 else None
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
    feats = np.zeros((MAX_UNITS, V2_UF if v2 else UF), dtype=np.float16)
    for k, e in enumerate(ents[:MAX_UNITS]):
        ox, oy = own(e[F["x"]], e[F["y"]], team, sub)
        if live:
            card = e[F["card_id"]]
            ids[k] = card + 2
            vals = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                    e[F["hp"]] / max(1, e[F["max_hp"]]),
                    1.0 if CARD_FLY.get(card) else 0.0,
                    1.0 if CARD_BUILDING.get(card) else 0.0,
                    float(np.log1p(max(0, e[F["max_hp"]]))) / 10.0)
        elif vision:
            # The vision class of the unit type, flags from the class table; the form goes
            # to its own column, or (v2, E14) into unit_feats.
            unit = e[F["unit_type"]]
            ids[k] = VC.token(unit)
            fly, building = VC.flags(unit)
            vals = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                    e[F["hp"]] / max(1, e[F["max_hp"]]), fly, building,
                    1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
            if forms is not None:
                forms[k] = VC.form_feature(unit)
        else:
            ids[k] = e[F["unit_type"]] + 2
            vals = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                    e[F["hp"]] / max(1, e[F["max_hp"]]), 1.0 if e[F["flying"]] else 0.0,
                    1.0 if e[F["kind"]] == 1 else 0.0,
                    1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
        if v2:
            # E14: the form and the burrow flag are unit features 7 and 8.
            unit = e[F["unit_type"]]
            vals += (VC.form_feature(unit) if vision else unit_form(unit),
                     1.0 if burrowing(e) else 0.0)
        feats[k] = vals
        if burrow is not None:
            burrow[k] = 1.0 if burrowing(e) else 0.0
    p = state["players"][team]
    hand = [c + 2 for c in p["hand"][:4]] + [0] * (4 - len(p["hand"][:4]))
    opp = 1 - team
    seen = sorted(mem["seen"][opp])[:8]
    last = mem["last"][opp][-4:]
    t_ms = tick * 50
    scalars = [p["elixir_milli"] / 10_000, min(1.0, t_ms / 300_000),
               1.0 if t_ms >= 120_000 else 0.0, 1.0 if t_ms >= 180_000 else 0.0]
    if v2:
        # E14: 3x elixir from 240 s, and the opponent's bar as this side counts it.
        scalars += [1.0 if t_ms >= 240_000 else 0.0, opp_milli / 10_000,
                    1.0 if opp_exact else 0.0]
    row = {
        "tick": tick, "side": 1 if team == 0 else -1,
        "hand_exact": True if hand_exact is None else bool(hand_exact),
        "unit_ids": ids, "unit_feats": feats, "towers": towers_vec(state, team),
        "scalars": np.asarray(scalars, dtype=np.float32),
        "hand": np.asarray(hand, dtype=np.int16),
        "next": (p["next_card"] + 2) if p.get("next_card", -1) >= 0 else 0,
        "opp_seen": np.asarray(seen + [0] * (8 - len(seen)), dtype=np.int16),
        "opp_last": np.asarray([c for c, _ in last] + [0] * (4 - len(last)), dtype=np.int16),
        # Ages floor at 0: only a noisy clock (E17, a tick read early) makes one negative.
        "opp_last_age": np.asarray([max(0.0, min(1.0, (tick - t) / 400)) for _, t in last]
                                   + [1.0] * (4 - len(last)), dtype=np.float16),
    }
    if forms is not None:
        row["unit_form"] = forms
    if burrow is not None:
        row["unit_burrow"] = burrow
    if press or v2:
        cards, states = rs_presses.ability_features(p, BUTTONS, buttons)
        row["ability_card"] = np.asarray(cards, dtype=np.int16)
        row["ability_state"] = np.asarray(states, dtype=np.float16).reshape(
            BUTTONS, len(rs_presses.ABILITY_FEATS))
    if pending is not None or v2:
        row["own_pending"], row["own_pending_land"] = pending_features(pending or ())
    if v2:
        row["hand_feats"] = hand_features(p)
        row["spell_ids"], row["spell_feats"] = spell_features(state, team, sub, (kx, ky))
        row = {k: row[k] for k in V2_OBS_FIELDS}
    return row


def rows_for(payload: dict, ctx, engine=None, *, mem_lag=None, trust=None, presses=None,
             info=None, decision=None, schema=None, noise=None, noise_seed=None):
    """Rows of one match. `engine` as in _setup; the battle runs without command delay,
    since the recorded ticks are taken to be the ticks the plays executed (an assumption until
    tools/rs_dt_diag.py's headroom check reports, runbook section 19). `decision` as --obs-delay
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
    with a delay also info["decision"], the counts manifest.json sums (module doc). `schema`
    as --schema (schema_mode; None: ROW_SCHEMA): "v2" gives rs_schema.V2's rows (module doc,
    "Schema v2"), the opponent's elixir counted by a SeenElixir at the memory view's lag, and
    info["observation"] (rows, opp_inexact, spell_rows); legacy calls obs_row as before.
    `noise` as --noise (noise_profile; None: NOISE, "none": no noise) and `noise_seed` as
    --noise-seed (None: NOISE_SEED): with a profile every row is obs_row of the state the acting
    side perceives, the memory and (v2) the opponent's counted elixir get each play as the other
    side perceives it, and info["noise"] has the match's MatchNoise counts (module doc,
    "Perception noise"); without one nothing of this runs and obs_row is called as before."""
    lag = mem_lag_value(MEM_LAG if mem_lag is None else mem_lag)
    legacy = lag == MEM_LEGACY
    drule = decision_rule(DECISION if decision is None else decision)
    if drule.on and legacy:
        raise ValueError("--obs-delay: decision-time rows read the memory dated at execution; "
                         "--mem-lag legacy is the order before E08")
    v2 = schema_mode(ROW_SCHEMA if schema is None else schema) == rs_schema.SCHEMA_V2
    if v2 and legacy:
        raise ValueError("--schema v2: the rows read the memory dated at execution; --mem-lag "
                         "legacy is the order before E08")
    rule = trust_rule(trust)
    press_on = rs_presses.press_mode(PRESSES if presses is None else presses) == rs_presses.PRESS_ON
    noise_prof = noise_profile(NOISE if noise is None else noise)
    nseed = noise_seed_value(NOISE_SEED if noise_seed is None else noise_seed)
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
    # E17 (--noise): one MatchNoise for the match, seeded by its tag; both sides observe through
    # it (its per-unit decisions are kept per observer). None: a perfect observer, as before.
    nz = None
    if noise_prof is not None:
        nz = PN.MatchNoise(noise_model(noise_prof, b, sub),
                           PN.match_seed((payload.get("source") or {}).get("replay_tag", ""),
                                         nseed))
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
    if press_on or v2:                              # v2 rows show the buttons with presses off
        champs = rs_presses.champion_ids(cat, getattr(royalesim, "CATALOGUE_FIELDS", None))
        for t in (0, 1):
            s = seat_of[t]
            btns[t] = rs_presses.side_buttons([ids[c] for c in dealt[s]],
                                              [forms[s][k] for k in perm[s]], champs)
    if press_on:
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
    # E14 (--schema v2): the opponent's elixir each side counts from what its memory view holds
    # (the same lag), and an honest hand_exact for the rows at execution too (a DecisionPlan
    # without a delay knows which plays ran before a tick). Legacy: none of it.
    seen_elixir, hist, ostat = None, plan, None
    if v2:
        seen_elixir = SeenElixir(cat, fields=list(getattr(royalesim, "CATALOGUE_FIELDS", ())
                                                  or ()) or None,
                                 law=elixir_law(engine), lag=max(rs_memory.BOARD_LAG, lag))
        if hist is None:
            hist = DecisionPlan(queue, [ids[q[2]] + 2 for q in queue], [],
                                (payload.get("source") or {}).get("replay_tag", ""),
                                DecisionRule())
        ostat = {"rows": 0, "opp_inexact": 0, "spell_rows": 0}

    def vkw(team, tick, decision_row=False):
        """The obs_row keywords --schema v2 adds for `team`'s row at `tick` (none for legacy,
        so obs_row is called exactly as before E14). A decision-time row brings its own
        pending plays and hand_exact."""
        if not v2:
            return {}
        kw = {"schema": rs_schema.SCHEMA_V2, "opp_elixir": seen_elixir.view(team, tick)}
        if not press_on:                            # presses on: pkw has the buttons
            kw["buttons"] = btns[team]
        if not decision_row:
            kw["pending"] = []
            kw["hand_exact"] = hist.exact(team, tick)
        return kw

    def look(state, team, view, tick, **kw):
        """obs_row of `team`'s row at `tick`. E17 (--noise): of the state `team`'s screen reader
        reports, at the tick its clock reads; the row keeps the true tick, from which the
        outcome window and the trust cuts count. Without noise, obs_row exactly as before."""
        if nz is None:
            return obs_row(state, team, sub, view, tick, **kw)
        seen = nz.observe(state, team)              # sets nz.tick_shift
        row = obs_row(seen, team, sub, view, tick + nz.tick_shift, **kw)
        row["tick"] = tick
        return row

    def record(team, token, tick):
        """`team`'s play of row token `token`, run on `tick`, into the memory: as it ran, or (E17
        --noise) as the other side perceives it (MatchNoise.record's draws: missed, recorded
        late, or as the unknown card). The (token, tick) the memory got; None: missed."""
        if nz is None:
            mem.record(team, token, tick)
            return token, tick
        seen = nz.model.play_seen(team, token, tick, nz.rng, stats=nz.stats)
        if seen is not None:
            mem.record(team, seen[0], seen[1])
        return seen

    def drow(state, team, view, tick):
        """A decision-time row of `team` at `tick`: obs_row with its pending plays and honest
        hand_exact (through look, E17's noise hook)."""
        pending = plan.pending(team, tick)
        exact = plan.exact(team, tick)
        dstat["pending_rows"] += int(bool(pending))
        dstat["exact_rows"] += int(exact)
        return look(state, team, view, tick, **pkw[team], pending=pending, hand_exact=exact,
                    **vkw(team, tick, True))

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
        cmds, sent_plays = [], []
        acting = set()
        for (_, team, card, x, y) in due:
            acting.add(team)
            cid = ids[card]
            hand = state["players"][team]["hand"]
            ex, ey = rs_orient.to_engine(x, y, sub)
            ox, oy = own(ex, ey, team, sub)
            if plan is None:                        # E11: decision-time rows were taken above
                row = look(state, team, mem.view(tick, 0) if legacy else known, tick,
                           **pkw[team], **vkw(team, tick))
                row["action"] = (cid + 2, int(min(17, max(0, ox // 1000))), int(min(31, max(0, oy // 1000))))
                rows.append(row)
            # The recorded tick is the tick it ran (E17: the memory gets what the other side
            # saw). Recorded above the hand check, which reads no memory: the rows of before.
            seen = record(team, cid + 2, tick)
            if cid in hand:
                cmds.append((team, hand.index(cid), ex, ey))
                sent_plays.append((team, cid, seen))    # E14: counted once the engine runs it
            elif cut > tick:
                cut, cut_why = tick, CUT_HAND
        # E10: this tick's presses, after its plays: a row each (action (0, 0, 0), press =
        # button + 1) and the command (team, HAND_SIZE + button, 0, 0). A press no single
        # button is behind is not sent, and cuts trust.
        pcmds, pcost = [], []
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
                row = look(state, team, mem.view(tick, 0) if legacy else known, tick,
                           **pkw[team], **vkw(team, tick))
                row["action"] = (0, 0, 0)
                row["press"] = button + 1
                rows.append(row)
            pcmds.append(rs_presses.command(team, button, hand_n))
            pcost.append((team, press_cost(state["players"][team], button)))
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
                    row = look(state, team, mem.view(tick, 0) if legacy else known, tick,
                               **pkw[team], **vkw(team, tick))
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
                taken = plog.result(reason)
                if not taken and cut > tick:
                    cut, cut_why = tick, CUT_PRESS
                elif taken and seen_elixir is not None:
                    # E14: a press that ran is public, and costs its button's elixir.
                    who, cost = pcost[k - len(cmds)]
                    seen_elixir.press(who, tick, cost)
            elif reason != "OK":
                if cut > tick:
                    cut, cut_why = tick, CUT_REFUSED
            elif seen_elixir is not None and k < len(sent_plays):
                # E14: a play the engine ran is on the board, and its side has paid for it. E17:
                # counted as the memory got it; a play missed, or not named (token 1), is not.
                who, cid, seen = sent_plays[k]
                if nz is None:
                    seen_elixir.record(who, tick, cid)
                elif seen is not None and seen[0] >= rs_memory.TOKEN_OFFSET:
                    seen_elixir.record(who, seen[1], seen[0] - rs_memory.TOKEN_OFFSET)
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
        if ostat is not None:
            ostat["rows"] += 1
            ostat["opp_inexact"] += int(not row["scalars"][V2_OPP_EXACT])
            ostat["spell_rows"] += int(bool(row["spell_ids"].any()))
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
        if ostat is not None:
            info["observation"] = ostat
        if nz is not None:
            info["noise"] = dict(sorted(nz.stats.items()))
    return rows, "ok"


FIELDS = ("tick", "side", "hand_exact", "unit_ids", "unit_feats", "towers", "scalars",
          "hand", "next", "opp_seen", "opp_last", "opp_last_age", "outcome", "trusted",
          "won", "hog")


#: vocab.json "schema": the row layout of these shards (obs_row and FIELDS above).
SCHEMA = "rs_rows/legacy"


def vocab_record(cat, unit_types, cfg, fp, card_map, live, *, version=None, schema=SCHEMA,
                 obs_delay=0, obs_delay_jitter=0, opp_lag=0, wait_keep_at_build=None,
                 opp_memory=None, classes=None, visibility=None, trust=None, presses=None,
                 decision=None, selection=None, perception_noise=None):
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
      schema_spec         with `schema` rs_schema.SCHEMA_V2 (E14; "v2" is read as it):
                          rs_schema.V2.to_json(BUTTONS), the layout of every column; and
                          extra_columns then lists only the columns outside it (trusted_v2
                          under --trust v2), since unit_form, unit_burrow, act_press, the
                          ability and the pending columns are v2's own
      wait_thinning       with `wait_keep_at_build` not None (E18): wait_rule(), how the
                          builder kept that share of the waits
      selection           with `selection` (E18, selection_record; None for the legacy pick):
                          how the matches were picked
      perception_noise    with `perception_noise` (E17, perception_noise.vocab_entry; noise_vocab
                          gives the keyword, none without --noise): the profile and run seed
                          every row was observed under

    sim/train_tokens.py copies all of it into every checkpoint's meta, and
    tools/rs_checkpoint.load_policy checks cards, units and card_table against the engine
    a checkpoint is loaded into.
    """
    vision = classes is not None
    if vision:
        unit_mode(UNITS_VISION, live)
    if isinstance(schema, rs_schema.Schema) or schema in rs_schema.SHORT_NAMES:
        schema = schema_mode(schema)                 # any other string is written as given
    rows_v2 = schema == rs_schema.SCHEMA_V2
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
    if wait_keep_at_build is not None:
        rec["wait_thinning"] = wait_rule(wait_keep_at_build)
    if selection is not None:
        rec["selection"] = dict(selection)
    if perception_noise is not None:
        rec["perception_noise"] = dict(perception_noise)
    if rows_v2:
        rec["schema_spec"] = rs_schema.V2.to_json(BUTTONS)
        extra = list(TRUST_FIELDS) if v2 else []
    else:
        extra = ((list(VISION_FIELDS) if vision else [])
                 + (list(VISIBILITY_FIELDS) if public else [])
                 + (list(TRUST_FIELDS) if v2 else [])
                 + (list(PRESS_COLUMNS) if press_entry is not None else [])
                 + (list(DECISION_FIELDS) if drule is not None else []))
    if extra:
        rec["extra_columns"] = extra
    return rec


def build(batch_id, payload_texts, live=False, out_dir=None, engine=None, *, units=None,
          mem_lag=None, visibility=None, trust=None, presses=None, stats=False,
          decision=None, schema=None, wait_keep_at_build=None, noise=None, noise_seed=None):
    """One shard, shard-<batch_id>.npz in `out_dir` (else OUT), from IL_Replay payload texts.
    `units` / `mem_lag` / `visibility` / `trust` / `presses` / `decision` / `schema` /
    `wait_keep_at_build` as --units / --mem-lag / --visibility / --trust (trust_rule) /
    --presses / --obs-delay (decision_rule) / --schema (schema_mode) / --wait-keep-at-build
    (wait_keep_value), kept as UNITS / MEM_LAG / VISIBILITY / TRUST / PRESSES / DECISION /
    ROW_SCHEMA / WAIT_KEEP_AT_BUILD for this process (None: keep the module's); main passes all
    eight to every worker, as it passes `live`. Under presses "on" the shard also has act_press
    (int16, from each row's "press") and PRESS_FIELDS; with a delay (E11) DECISION_FIELDS. Under
    schema v2 (E14) the shard has rs_schema.V2's columns (shard_columns, with trusted_v2 under
    --trust v2), each cast to its dtype. With a wait share (E18) each match's rows go through
    thin_waits before they are stacked. A delay or schema v2 with mem_lag "legacy" is refused
    (ValueError) before any match. `noise` / `noise_seed` as --noise (noise_profile: a profile,
    its to_dict(), a path, "default"; "none" turns it off) / --noise-seed, kept as NOISE /
    NOISE_SEED (None: keep the module's); main passes the profile as its to_dict(). A profile
    the engine cannot bind is refused (noise_model's error) before any match. Returns (batch_id,
    rows, skipped by reason); with `stats`, also the shard's trust_stats, which main merges into
    manifest.json (with a wait share it has "wait_thinning": the waits kept and dropped)."""
    global LIVE, OUT, UNITS, MEM_LAG, VISIBILITY, TRUST, PRESSES, DECISION, ROW_SCHEMA
    global WAIT_KEEP_AT_BUILD, NOISE, NOISE_SEED
    units = UNITS if units is None else units
    unit_mode(units, live)
    lag = MEM_LAG if mem_lag is None else mem_lag_value(mem_lag)
    vis = visibility_mode(VISIBILITY if visibility is None else visibility)
    rule = trust_rule(trust)
    press = rs_presses.press_mode(PRESSES if presses is None else presses)
    drule = decision_rule(decision)
    name = schema_mode(ROW_SCHEMA if schema is None else schema)
    keep = wait_keep_value(WAIT_KEEP_AT_BUILD if wait_keep_at_build is None
                           else wait_keep_at_build)
    prof = noise_profile(NOISE if noise is None else noise)
    nseed = noise_seed_value(NOISE_SEED if noise_seed is None else noise_seed)
    if drule.on and lag == MEM_LEGACY:
        raise ValueError("--obs-delay: decision-time rows read the memory dated at execution; "
                         "--mem-lag legacy is the order before E08")
    if name == rs_schema.SCHEMA_V2 and lag == MEM_LEGACY:
        raise ValueError("--schema v2: the rows read the memory dated at execution; --mem-lag "
                         "legacy is the order before E08")
    LIVE, UNITS, MEM_LAG, VISIBILITY, TRUST, PRESSES = live, units, lag, vis, rule, press
    DECISION, ROW_SCHEMA, WAIT_KEEP_AT_BUILD = drule, name, keep
    NOISE, NOISE_SEED = prof, nseed
    if out_dir:
        OUT = Path(out_dir)
    ctx = _setup(engine)
    if NOISE is not None:
        # E17: bound here once, so a profile this engine cannot bind fails the shard, not every
        # match as an "error" skip; rows_for finds the model in _NOISE_MODELS.
        royalesim = ctx[0]
        noise_model(NOISE, rs_engine.make_battle(rs_engine.resolve(engine).undelayed(), royalesim),
                    royalesim.SUBTILE_PER_MILLITILE)
    rows, tags, skipped, infos = [], [], {}, []
    waits_kept = waits_dropped = 0
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
        if keep is not None:
            # E18: the match's waits thinned, seeded by its tag (thin_waits).
            r, dropped = thin_waits(r, tag, keep)
            waits_dropped += dropped
            waits_kept += sum(1 for row in r if is_wait(row))
        rows += r
        tags += [tag] * len(r)
        infos.append(info)
    if rows and ROW_SCHEMA == rs_schema.SCHEMA_V2:
        # E14: every column of rs_schema.V2 at its dtype, act_press on every row.
        spec = rs_schema.V2
        data = {k: np.stack([np.asarray(r[k]) for r in rows]).astype(spec.dtype(k))
                for k in spec.row_fields(TRUST_FIELDS if TRUST.v2 else ())}
        act = np.asarray([r["action"] for r in rows], dtype=np.int16)
        data["act_card"], data["act_x"], data["act_y"] = act[:, 0], act[:, 1], act[:, 2]
        data["act_press"] = np.asarray([r.get("press", 0) for r in rows], dtype=np.int16)
        data["match"] = np.asarray(tags)
        data["split"] = np.asarray([split_of(t) for t in tags])
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"shard-{batch_id:04d}.npz", **data)
    elif rows:
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
        st = trust_stats(rows, infos, TRUST)
        if keep is not None:
            st["wait_thinning"] = {"kept": waits_kept, "dropped": waits_dropped}
        return batch_id, len(rows), skipped, st
    return batch_id, len(rows), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--others", type=int, default=20000,
                    help="non-Hog matches added to every Hog 2.6 match (default 20000)")
    ap.add_argument("--others-pick", choices=PICK_MODES, default=PICK_FIRST,
                    help="which --others: 'first' (default) the first in parquet order, as "
                         "before; 'random' a uniform draw seeded by --select-seed (E18)")
    ap.add_argument("--select-seed", type=int, default=0,
                    help="--others-pick random: the draw's seed (default 0)")
    ap.add_argument("--all-matches", action="store_true",
                    help="every match the rows can be built from (--others is not read; E18). "
                         "Add --wait-keep-at-build so the shards stay a sensible size")
    ap.add_argument("--wait-keep-at-build", type=float, default=None, metavar="P",
                    help="keep each wait row with probability P, seeded by the match tag "
                         "(E18; default: every wait row, as before); train_tokens reads it "
                         "from vocab.json")
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
    ap.add_argument("--schema", choices=rs_schema.CHOICES, default="legacy",
                    help="the row layout (tools/rs_schema.py): 'legacy' (default) the rows as "
                         "before, schema rs_rows/legacy; 'v2' (E14) adds spells in flight, the "
                         "opponent's counted elixir, own pending plays, hand evolution, the "
                         "ability buttons and unit form and burrow (schema rs_rows/v2)")
    PN.add_noise_args(ap)                          # E17: --noise, --noise-only, --noise-seed
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
        schema = schema_mode(args.schema)
        if schema == rs_schema.SCHEMA_V2 and args.mem_lag == MEM_LEGACY:
            raise ValueError("--schema v2 needs the memory dated at execution: --mem-lag / "
                             "--opp-lag TICKS, not legacy")
        wait_keep = wait_keep_value(args.wait_keep_at_build)
        _nonneg(args.others, "--others")
        _nonneg(args.select_seed, "--select-seed")
        if args.batch < 1 or args.workers < 1:
            raise ValueError("--batch and --workers must be 1 or more")
        noise = PN.profile_from_args(args)          # E17: None without --noise
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    rs_engine.set_active(cfg)
    global OUT, LIVE, UNITS, MEM_LAG, VISIBILITY, TRUST, PRESSES, DECISION, ROW_SCHEMA
    global WAIT_KEEP_AT_BUILD, NOISE, NOISE_SEED
    LIVE = args.live_compat
    UNITS, MEM_LAG, VISIBILITY, TRUST = args.units, args.mem_lag, args.visibility, rule
    PRESSES = args.presses
    DECISION = decision
    ROW_SCHEMA = schema
    WAIT_KEEP_AT_BUILD = wait_keep
    NOISE, NOISE_SEED = noise, args.noise_seed
    if args.out:
        OUT = args.out
    royalesim, names, ids, cat = _setup(cfg)
    vocab_battle = rs_engine.make_battle(cfg, royalesim)
    unresolved = []
    if NOISE is not None:
        # E17: bound once here, so a profile this engine cannot bind stops the run at once.
        try:
            unresolved = noise_model(NOISE, vocab_battle,
                                     royalesim.SUBTILE_PER_MILLITILE).unresolved
        except (ValueError, FileNotFoundError) as exc:
            ap.error(f"--noise: {exc}")
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
    print(f"schema: {describe_schema(ROW_SCHEMA)}", flush=True)
    if UNITS == UNITS_VISION:
        print(f"units:  vision classes, {VCT.describe()}", flush=True)
        if VC.unknown:
            print(f"        {len(VC.unknown)} engine unit types have no class and read as "
                  f"unknown (token 1): {', '.join(VC.unknown[:12])}"
                  f"{' ...' if len(VC.unknown) > 12 else ''}", flush=True)
    if WAIT_KEEP_AT_BUILD is not None:
        print(f"waits:  each kept with p {WAIT_KEEP_AT_BUILD:g}, seeded by the match tag",
              flush=True)
    if NOISE is not None:
        print(f"noise:  {describe_noise(NOISE, NOISE_SEED)}", flush=True)
        for what in unresolved:
            print(f"        not bound: {what}", flush=True)
    t0 = time.time()
    parts = sorted((CORPUS / "replays").glob("part-*.parquet"))
    if args.limit_parts:
        parts = parts[:args.limit_parts]
    # E18, pass 1: which rows are candidates; no payload is kept (module doc, "Selection").
    cands = []
    for n, part in enumerate(parts):
        cands += [(n, k, tag, hog) for k, tag, hog in scan_part(_read_part(part, SCAN_COLUMNS),
                                                                names)]
    picked = select_matches(cands, args.others, args.others_pick, args.select_seed,
                            args.all_matches)
    n_hog = sum(1 for c in picked if c[3])
    print(f"selected {n_hog:,} Hog 2.6 + {len(picked) - n_hog:,} others ({time.time() - t0:.0f}s)", flush=True)
    selection = selection_record(picked, len(cands), len(parts), args.others, args.others_pick,
                                 args.select_seed, args.all_matches)
    if selection is not None:
        print(f"        pick {selection['pick']}"
              + (f", seed {selection['seed']}" if selection["seed"] is not None else "")
              + f": {selection['others_taken']:,} of {selection['other_candidates']:,} non-Hog "
                f"candidates in {len(parts)} parts", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    unit_types = json.loads(vocab_battle.unit_types_json())
    (OUT / "vocab.json").write_text(json.dumps(vocab_record(
        cat, unit_types, cfg, fp, card_map, LIVE,
        version=getattr(royalesim, "__version__", "?"), schema=ROW_SCHEMA,
        wait_keep_at_build=WAIT_KEEP_AT_BUILD, selection=selection,
        classes=VCT if UNITS == UNITS_VISION else None,
        **memory_vocab(MEM_LAG), **visibility_vocab(VISIBILITY), **trust_vocab(TRUST),
        **press_vocab(PRESSES), **decision_vocab(DECISION), **noise_vocab(NOISE, NOISE_SEED))),
        encoding="utf-8")
    n_batches = -(-len(picked) // args.batch)
    total, skipped, totals, shards = 0, {}, {}, []
    with ProcessPoolExecutor(args.workers, initializer=rs_engine.set_active,
                             initargs=(cfg.to_json(),)) as pool:
        def submit(k, b):
            return pool.submit(build, k, b, LIVE, str(OUT), cfg.to_json(), units=UNITS,
                               mem_lag=MEM_LAG, visibility=VISIBILITY, trust=TRUST.to_json(),
                               presses=PRESSES, stats=True, decision=DECISION.to_json(),
                               schema=ROW_SCHEMA, wait_keep_at_build=WAIT_KEEP_AT_BUILD,
                               noise=None if NOISE is None else NOISE.to_dict(),
                               noise_seed=NOISE_SEED)

        # E18, pass 2: the picked payloads a part at a time, at most 2 x workers batches out.
        jobs = chunked(selected_payloads(picked, parts), args.batch)
        for n, f in enumerate(bounded(submit, jobs, 2 * args.workers), 1):
            bid, rows, sk, st = f.result()
            total += rows
            for k, v in sk.items():
                skipped[k] = skipped.get(k, 0) + v
            merge_stats(totals, st)
            if rows:                                  # build writes no shard without rows
                shards.append(shard_entry(bid, st))
            if n % 10 == 0 or n == n_batches:
                print(f"  {n}/{n_batches} shards, {total:,} rows, skipped {skipped} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    man = manifest(totals, shards, skipped, TRUST, wait_keep=WAIT_KEEP_AT_BUILD)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"done: {total:,} rows -> {OUT}\n      {describe_manifest(man)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
