r"""The public game state from screen perception: the dict tools/rs_rows.obs_row reads (V05).

    from tools import public_state as PS      # or `import public_state` with tools/ on sys.path
    build = PS.PublicStateBuilder(classes, spawn_table, deck=deck_ids, catalogue=cat,
                                  fields=RR.F, sub=royalesim.SUBTILE_PER_MILLITILE)
    state = build.state(vf, opp_deck=known_opponent_cards,
                        track_cards=deriver.track_cards(vf.tracks))
    row = RR.obs_row(state, PS.OWN_TEAM, sub, memory.view(state["tick"], 0), state["tick"])
    rebuild_state = PS.sanitize_for_rebuild(state, spawn_table, fields=RR.F, sub=sub)

`rs_state_from_vision(vf, deck, classes, spawn_tables, levels=11, ...)` is the one-call form;
the live loop keeps one PublicStateBuilder (it caches its lookups). Inputs: a
scripts/vision VisionFrame (tracks from scripts/vision/tracker.py labelled with the
data/vision/classes.json classes, hud from hud.HudReader, events from events.py), our
deck, the vision_classes.ClassTable, the E06 spawn table (data/rs_rows/spawn_primary.json,
tools/rs_spawn_table.py: card_names, unit_types, primary, hp) and the card levels (11 in a
friendly). The output is the state_json-shaped dict obs_row, towers_vec and RSAgent.act /
ahead / rebuilt read, so a frame gives the row a RoyaleSim state with the same board gives.

FRAMES AND TEAMS (scripts/vision/geometry.py, scripts/vision/types.py). World = RoyaleSim's
engine frame in millitiles: x 0..18000 left to right on screen, y 0..32000 top to bottom.
RoyaleSim team 0 (blue) defends low y, team 1 (red) high y. The bot sits at the bottom of
the screen, so it is always team 1 (OWN_TEAM) and world orientation equals the screen:
nothing is flipped between a Track's (x_mt, y_mt) and an entity's (x, y) = mt * sub (sub =
royalesim.SUBTILE_PER_MILLITILE, 18). obs_row(state, 1, ...) then puts the board in the
model's own frame, ox = 18000 - x, oy = y (rs_rows.own for team 1, a mirror in x), and
rs_eval.to_world / geometry.ArenaGeometry.model_tile_to_screen_tile ((17 - tx, ty)) turn a
model tile back.
Track.team is already a RoyaleSim team id; a track whose team vote is still undecided goes to
the half of the board it stands in (unknown_team="half") or is left out ("drop"). own_team=0
serves offline runs on an engine seat 0; it only changes which player the HUD fills.

TOWERS live in players[t]["tower_hp"] / ["tower_max_hp"] as [king, left, right], left and
right being WORLD (= screen) sides for both teams (types.TOWER_KEYS; rs_rows.towers_vec reads
slot 2 as team 1's own left because its frame mirrors x). They are not entities: obs_row
leaves crown towers out of the unit tokens and, with no king entity, sorts the units by
distance from where the king stands, (9000, 29000) mt for team 1, the king entity's centre.
An unread princess counts as full (the HUD's level-11 maximum, LEVEL11_TOWER_MAX_HP); a
king's number shows only once it is damaged and the HUD reports it full until then.

ENTITIES, one per unit track, built BY NAME from the engine's ENTITY_FIELDS
(rs_engine.field_index; rsim-07), every other field at the engine's "nothing" value:
  uid            the track's uid
  team           the track's team (see above)
  kind           1 building, 0 troop (the class table; obs_row reads kind == 1 as building)
  card_id        the CREATING card (the --live-compat unit token): track_cards[uid] when
                 given (scripts/vision/events.EventDeriver.track_cards), else
                 ClassTable.card_id_for(class, catalogue, deck): our deck for our units, the
                 opponent's known cards for theirs (a Witch's skeleton is the Witch's), else
                 the class's primary card
  tower_slot     -1
  x, y           world millitiles * sub, rounded
  hp, max_hp     max_hp: the E06 unit_hitpoints(card, level) row naming the entity's unit
                 (or another unit of its class and form), then the class's primary card's
                 rows, then ClassTable.hitpoints(class, form, level), else UNKNOWN_MAX_HP;
                 hp = the bar fill (no bar read: full) * max_hp, within [1, max_hp]
  flying         the class table's flag for the track's form
  deploy_ticks   deploy time minus the track's age (scripts/vision/tracker.deploy_ticks_left),
                 within [0, the deploy time] (a track first seen after the state's tick)
  status_flags   0 (a hidden or burrowing unit is not on screen; see LATER)
  level          the team's card level
  unit_type      an engine unit_types index of the track's class and form: the creating
                 card's E06 primary unit when it is one (Goblins -> Goblin_Stab: what the
                 engine shows, and what RSAgent.rebuilt can spawn), else the class's canonical
                 unit (ClassTable.unit_name), else another engine unit of the class, else -1
                 (row token 1, "unknown")

THE ROW SCHEMAS this state serves:
  --live-compat (the first live model): card_id + 2, flying / building from the card's
      catalogue row, log1p(max_hp); exact when the creating card and the E06 hit points are.
  --units vision (E16, obs_row(..., units="vision")): unit_type -> the class token, form and
      flags through vision_classes.UnitLookup; exact for every unit of a class.
  --units types (the default unit-type rows): unit_type + 2 with the engine flags; exact for
      a card's own unit and canonical units, not for other aliases of a class (a Goblin Curse
      goblin, GoblinCurseGoblin, reads as the class's Goblin).

SPELLS AND PROJECTILES are not board units: a spell track (a "spell" label: "Fireball",
"the-log") never becomes an entity. With spell_fields (royalesim.SPELL_FIELDS) it becomes a
"spells" row (team, card_id, x/y and aim at the track, ticks_flown = its age; an undecided
team counts as the opponent's); else "spells" is []. "projectiles" is always []. Crown-tower
and tower-troop labels, objects, UI and unknown labels are left out; state["perception"]
counts what was left out and what was assumed. With unknown_units (N02, off by default) an
enemy track of an unknown label stays as an unknown unit instead: card_id -1, unit_type -1
(row token 1 in --live-compat, unit-type and vision rows alike, the "unknown" token perception
noise trains), a ground troop of UNKNOWN_MAX_HP. A legacy obs_row does not read "spells", a v2
one does (spell_ids / spell_feats, the SPELL cards only); RSAgent.rebuilt drops spells in flight.

THE CLOCK: tick, tick_ms, overtime, and elixir_rate by tools/rs_elixir.py's law (2x from
tick 2400, 3x from 4800); game_over on the end screen.

NOT ON SCREEN: the opponent's hand, next card and elixir (empty / -1 / 0), and our own
pending plays (pass pending_cost; RSAgent.act subtracts it from our elixir).

V2 ROWS (E14 live; tools/rs_rows.py --schema v2, tools/rs_schema.py). A v2 row also reads what
the bot knows besides the board: the opponent's elixir as it counts it, and its own plays sent
and not on the board yet. One LiveObserver per battle holds them; state(vf, observer=...) puts
its inputs at the state's tick under state["observer"] (only then: a legacy state is what it
always was) and row_inputs(state) gives obs_row's v2 keywords from it:
  opp_elixir   OpponentElixir: a tools/rs_elixir.ElixirCounter fed the opponent plays
               scripts/vision/events reports (OppPlay card, exec_tick), rs_rows.SeenElixir's
               count with lag 0 (the rule rs_eval.RSAgent.opp_elixir uses): START_MANA at tick
               0, each play paid at its execution tick, regenerated by the law, capped. Its
               exactness (the row's opp_exact) is the counter's (no unaffordable play, no Mirror
               with nothing to copy), and False for the whole battle when the bot cannot have
               seen every play: no detector (blind), watching from after the deploy lockout
               and one more displayed second (start_tick > 110: DEPLOY_LOCKOUT_TICKS +
               LATE_MARGIN_TICKS), or a play of a card the catalogue does not price. An UNKNOWN
               play (card -1, N02) is paid its OppPlay.cost (the expected cost) and makes the
               count not exact too.
  own_pending  OwnTaps: the runner's own sent taps (card, slot, tap tick), each until the event
               deriver reports its OwnLanding (landed, or missed after its wait) or it is
               TAP_MAX_AGE ticks old; shown as rs_decide.pending_inputs(taps, now, delay), the
               decision-time rows' (card + 2, ticks to land) with delay = the checkpoint's
               --obs-delay (rs_decide.obs_delay_ticks; 0: landing-time rows show none). A
               human's own play (MANUAL) is never a sent tap: it shows once it is on the board.
  spells       the spell tracks as "spells" rows (spell_fields, as above): the detector's spell
               classes, x / y / aim at the track, ticks_flown its age.
Hand evolution, ability buttons and the burrow bit read zero live: the bot's deck has no
evolutions or heroes, the HUD reader reads no ability button, and a tunnelling unit is not on
screen.

sanitize_for_rebuild(state, spawn_table) returns a copy RSAgent.rebuilt can reset a Battle
from (vision-02: one refused spawn fails the whole rebuild and the decision is skipped): it
keeps the units rebuilt can spawn (a troop or building card, resolved exactly as rebuilt
does, rebuild_card), clamps them into the arena, moves ground units off the river (to the
nearest bank or bridge, WATER_MARGIN_MT clear), caps hp to [1, the spawned card's hit
points], pads both decks to 8 distinct cards and clamps elixir and tower hit points.

LATER (hooks, not implemented here):
  E14 / E15  the v2 schema and model read the form (unit_type carries it, UnitLookup.form)
             and may want the level (each entity carries it) or a perception-confidence
             feature (Track.conf; add it to the row there, not to this dict).
  E11        decision-time rows observe at t_exec - D: build the state at that tick
             (state(vf, tick=...) takes the frame's own tick by default).
  RL         self-play seats use engine states; this converter is the live side only.

Never imports scripts/gamestate (the memory reader) or scripts/cr_bot.py; imports the
standard library, scripts/vision (numpy), the stdlib-only tools vision_classes,
rs_spawn_table, rs_elixir and rs_engine (royalesim only when no field index is passed) and
rs_decide (numpy; torch and royalesim only inside its policy and legality classes).
"""
from __future__ import annotations

import copy
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:          # also when imported as tools.public_state
    sys.path.insert(0, str(ROOT / "tools"))
if str(ROOT) not in sys.path:                    # scripts.vision
    sys.path.append(str(ROOT))

import rs_decide  # noqa: E402  (Pending, pending_inputs, v2_row_kwargs: the v2 rows' rules)
import rs_elixir  # noqa: E402
import rs_spawn_table  # noqa: E402
import vision_classes  # noqa: E402
from scripts.vision import geometry as G  # noqa: E402
from scripts.vision.hud import LEVEL11_TOWER_MAX_HP  # noqa: E402
from scripts.vision.tracker import MID_Y_MT, deploy_ticks_left  # noqa: E402
from scripts.vision.types import (  # noqa: E402
    EMPTY_CARD, ENEMY_TEAM, FORM_BASE, HAND_SIZE, OWN_TEAM, PHASE_ENDED, PHASE_OVERTIME, SCREEN_END,
    UNKNOWN_CARD)

__all__ = ["OWN_TEAM", "ENEMY_TEAM", "SUBTILE_PER_MILLITILE", "REQUIRED_FIELDS", "ENTITY_DEFAULTS",
           "PublicStateBuilder", "rs_state_from_vision", "sanitize_for_rebuild", "rebuild_card",
           "card_hitpoints", "off_water", "into_arena", "team_levels", "load_classes",
           "load_spawn_table", "default_fields", "OpponentElixir", "OwnTaps", "LiveObserver",
           "row_inputs", "DEPLOY_LOCKOUT_TICKS", "LATE_MARGIN_TICKS", "TAP_MAX_AGE"]

#: royalesim.SUBTILE_PER_MILLITILE (18 sub-tiles per millitile); pass the engine's as `sub`.
SUBTILE_PER_MILLITILE = 18
FRIENDLY_LEVEL = vision_classes.FRIENDLY_LEVEL           # 11: friendlies play every card at 11
KIND_TROOP, KIND_BUILDING = 0, 1                          # entity "kind" (obs_row: 1 = building)
STATUS_NONE = 0
#: tools/rs_elixir.py's law from royalesim 0.1.42's constants: the clock (50 ms ticks, 3:00 of
#: regulation) and the elixir rate (rate_at: 2x from tick 2400, 3x from 4800).
ELIXIR_LAW = rs_elixir.ElixirLaw.constants()
TICK_MS = ELIXIR_LAW.tick_ms
REGULATION_TICKS = ELIXIR_LAW.regular_ticks
OVERTIME_TICKS = 2400                                      # state_json's overtime_ticks (2:00)
ELIXIR_MAX_MILLI = 10_000
#: max_hp of a unit no table gives hit points for (hp is then the bar fill * this).
UNKNOWN_MAX_HP = 1000
#: unknown_units: the deploy time of a unit of a label the class table cannot place (most
#: troops deploy in 1 s), so it reads as deploying for its first 20 ticks.
UNKNOWN_DEPLOY_MS = 1000
#: catalogue_json() column of the elixir cost, 2 (royalesim CATALOGUE_FIELDS begin name,
#: placement, elixir, ...: rs_elixir.CATALOGUE_PREFIX).
CATALOGUE_ELIXIR = rs_elixir.CATALOGUE_PREFIX.index("elixir")
#: sanitize_for_rebuild: how far a ground unit's centre is kept from river water, and from the
#: arena's outer edge, in millitiles. Half a tile is one arena half-cell: royalegym's reset rule
#: (protocol.spawn_violation, stated after the Rust core's) refuses a non-flying spawn whose
#: centre touches a water half-cell, the bank line included, so a centre one half-cell away is
#: on land; the engine-marked test confirms it on the PC.
WATER_MARGIN_MT = 500.0
EDGE_MARGIN_MT = 500.0
UNKNOWN_TEAM_POLICIES = ("half", "drop")
#: RoyaleSim takes no play before this tick (match.DEPLOY_LOCKOUT_TICKS): a count that was
#: watching by then has had the chance to see every play of the battle.
DEPLOY_LOCKOUT_TICKS = 90
#: ... and a watch that started within one displayed second after it is still in time: the
#: runner's start tick is its first accepted clock reading, base + 10 of the second shown (2:56
#: is tick 90, 2:55 tick 110), and a play made in that second is still a unit on the board when
#: the detector starts (deploy times are 1 s or more), so the deriver reports it.
LATE_MARGIN_TICKS = 20
#: A sent tap not seen landing after this many ticks is assumed lost (rs_decide.PENDING_TICKS,
#: live_play's 3 s).
TAP_MAX_AGE = rs_decide.PENDING_TICKS

#: The entity fields this module fills and the pipeline reads (obs_row, RSAgent.rebuilt).
REQUIRED_FIELDS: tuple[str, ...] = (
    "uid", "team", "kind", "card_id", "tower_slot", "x", "y", "hp", "max_hp", "flying",
    "deploy_ticks", "status_flags", "level", "unit_type",
)
#: The value of each royalesim.ENTITY_FIELDS column when there is nothing to report (fields
#: not listed default to 0). Lists are copied per row.
ENTITY_DEFAULTS: dict[str, Any] = {
    "uid": 0, "team": 0, "kind": KIND_TROOP, "card_id": -1, "tower_slot": -1, "x": 0, "y": 0,
    "hp": 0, "max_hp": 0, "radius": 0, "flying": False, "deploy_ticks": 0, "stun_ticks": 0,
    "knockback_ticks": 0, "footprint": None, "target_uid": -1, "attack_phase": 0,
    "facing": [0, 0], "shield": 0, "buffs": [], "status_flags": STATUS_NONE,
    "level": FRIENDLY_LEVEL, "mount_uid": -1, "charge": 0, "dest_x": -1, "dest_y": -1,
    "ability_ticks": 0, "unit_type": -1,
}

_SPELL, _UNIT, _TOWER, _UNKNOWN = "spell", "unit", "tower", "unknown"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_classes(path: str | Path | None = None) -> Any:
    """data/vision/classes.json (or `path`) as a vision_classes.ClassTable."""
    return vision_classes.load(path)


def load_spawn_table(path: str | Path | None = None) -> dict:
    """data/rs_rows/spawn_primary.json (or `path`); FileNotFoundError when it is missing."""
    path = Path(path) if path is not None else rs_spawn_table.DEFAULT_PATH
    table = rs_spawn_table.read(path)
    if table is None:
        raise FileNotFoundError(f"no spawn table at {path}: run tools/rs_spawn_table.py (E06) "
                                "under the engine the model was trained on")
    return table


def default_fields() -> dict[str, int]:
    """royalesim.ENTITY_FIELDS by name, checked (rs_engine.field_index; imports royalesim)."""
    import rs_engine
    return rs_engine.field_index()


def _field_index(fields: Mapping[str, int] | Sequence[str] | None) -> dict[str, int]:
    """name -> column from a mapping or a sequence of names (None: default_fields())."""
    if fields is None:
        fields = default_fields()
    if isinstance(fields, Mapping):
        index = {str(k): int(v) for k, v in fields.items()}
    else:
        index = {str(name): i for i, name in enumerate(fields)}
    return index


def _entity_fields(fields: Mapping[str, int] | Sequence[str] | None) -> dict[str, int]:
    index = _field_index(fields)
    missing = [name for name in REQUIRED_FIELDS if name not in index]
    if missing:
        raise ValueError(f"entity fields lack {missing}: pass royalesim.ENTITY_FIELDS "
                         "(rs_engine.field_index, or rs_rows.F after _setup)")
    return index


def team_levels(levels: Any = FRIENDLY_LEVEL) -> tuple[int, int]:
    """Card levels as (team 0, team 1): one int for both, a pair, or {team: level}."""
    if isinstance(levels, Mapping):
        out = (int(levels.get(0, FRIENDLY_LEVEL)), int(levels.get(1, FRIENDLY_LEVEL)))
    elif isinstance(levels, (list, tuple)):
        if len(levels) != 2:
            raise ValueError(f"levels: one int, (team 0, team 1) or {{team: level}}, "
                             f"got {levels!r}")
        out = (int(levels[0]), int(levels[1]))
    else:
        out = (int(levels), int(levels))
    if min(out) < 1:
        raise ValueError(f"card levels start at 1, got {out}")
    return out


def _card_list(cards: Iterable[Any], card_ids: Mapping[str, int], what: str) -> list[int]:
    """Catalogue ids (ints, or card names) without repeats or negatives, order kept."""
    out: list[int] = []
    for c in () if cards is None else cards:
        if isinstance(c, str):
            if c not in card_ids:
                raise ValueError(f"{what}: unknown card name {c!r}")
            c = card_ids[c]
        c = int(c)
        if c >= 0 and c not in out:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# the converter
# ---------------------------------------------------------------------------

class PublicStateBuilder:
    """VisionFrame -> the state dict obs_row reads, for one engine, deck and class table.

    classes       vision_classes.ClassTable (None: data/vision/classes.json)
    spawn_table   the E06 table (dict; None: data/rs_rows/spawn_primary.json): card_names,
                  unit_types and hp; made under the engine the model was trained on
    deck          our 8 cards (catalogue ids or names), the order players[t]["deck"] gets
    levels        card levels (team_levels); 11 in a friendly
    catalogue     Battle.catalogue_json() rows (costs; must name the cards the spawn table
                  does); None: costs from the class table's cards
    fields        royalesim.ENTITY_FIELDS by name (rs_rows.F); None: rs_engine.field_index()
    sub           royalesim.SUBTILE_PER_MILLITILE
    own_team      our RoyaleSim team (OWN_TEAM, 1, live)
    unknown_team  "half" or "drop" for tracks whose team is undecided
    tower_max_hp  TOWER_KEYS -> max hit points where the HUD has none (LEVEL11_TOWER_MAX_HP)
    spell_fields  royalesim.SPELL_FIELDS: spell tracks become "spells" rows (else [])
    unknown_units N02 (docs/NEW_CARDS_PLAN.md 2.2; off by default): an ENEMY track of a label
                  the class table cannot place stays on the board as an unknown unit, card_id
                  -1 (or its track_cards card) and unit_type -1, row token 1 in every unit
                  schema, instead of being dropped; state["perception"]["unknown_units"]
                  counts them. Our own such tracks are dropped as before
    """

    def __init__(self, classes: Any, spawn_table: Mapping[str, Any] | None, *, deck: Iterable[Any],
                 levels: Any = FRIENDLY_LEVEL, catalogue: Sequence[Sequence[Any]] | None = None,
                 fields: Mapping[str, int] | Sequence[str] | None = None,
                 sub: int = SUBTILE_PER_MILLITILE, own_team: int = OWN_TEAM,
                 unknown_team: str = "half", tower_max_hp: Mapping[str, int] | None = None,
                 spell_fields: Mapping[str, int] | Sequence[str] | None = None,
                 unknown_units: bool = False):
        if own_team not in (0, 1):
            raise ValueError(f"own_team must be 0 or 1, got {own_team!r}")
        if unknown_team not in UNKNOWN_TEAM_POLICIES:
            raise ValueError(f"unknown_team: one of {UNKNOWN_TEAM_POLICIES}, got {unknown_team!r}")
        if int(sub) <= 0:
            raise ValueError(f"sub must be > 0, got {sub!r}")
        self.classes = classes if classes is not None else load_classes()
        table = spawn_table if spawn_table is not None else load_spawn_table()
        names = [str(n) for n in table.get("card_names") or ()]
        if catalogue is not None:
            cat_names = [str(row[0]) for row in catalogue]
            if names and names != cat_names:
                raise ValueError("the catalogue and the spawn table name different cards: "
                                 "regenerate data/rs_rows/spawn_primary.json under this engine "
                                 "(tools/rs_spawn_table.py)")
            names = cat_names
        if not names:
            raise ValueError("no card names: pass the engine's catalogue or a spawn table "
                             "with card_names")
        self.card_names = names
        self.card_ids = {n: i for i, n in enumerate(names)}
        self.unit_types = [str(u) for u in table.get("unit_types") or ()]
        self.unit_index = {u: i for i, u in enumerate(self.unit_types)}
        self.hp_table: Mapping[str, Any] = table.get("hp") or {}
        self.primary = rs_spawn_table.primary_of(table) if table.get("primary") else {}
        self.F = _entity_fields(fields)
        width = max(self.F.values()) + 1
        template: list[Any] = [0] * width
        for name, i in self.F.items():
            template[i] = ENTITY_DEFAULTS.get(name, 0)
        self._template = tuple(template)
        self.SF = None if spell_fields is None else _field_index(spell_fields)
        self.sub = int(sub)
        self.own_team = int(own_team)
        self.enemy_team = 1 - self.own_team
        self.unknown_team = unknown_team
        self.unknown_units = bool(unknown_units)
        self.levels = team_levels(levels)
        self.deck = _card_list(deck, self.card_ids, "deck")
        self.deck_names = [names[c] for c in self.deck]
        self.costs = self._costs(catalogue)
        self.tower_max_hp = {**LEVEL11_TOWER_MAX_HP, **{str(k): int(v) for k, v in
                                                        (tower_max_hp or {}).items()}}
        self._labels: dict[str, tuple[str, int | None, int, str | None]] = {}
        self._unit_of: dict[tuple[int, int], int] = {}
        self._hp: dict[tuple, int | None] = {}
        self._resolved: dict[str, tuple[int | None, int]] = {}
        self._by_class: dict[tuple[int, int], int] | None = None

    # ------------------------------------------------------------ lookups
    def _costs(self, catalogue: Sequence[Sequence[Any]] | None) -> dict[int, int]:
        if catalogue is not None:
            return {i: int(row[CATALOGUE_ELIXIR]) for i, row in enumerate(catalogue)
                    if len(row) > CATALOGUE_ELIXIR}
        cards = getattr(self.classes, "cards", {}) or {}
        return {i: int((cards.get(n) or {}).get("elixir") or 0)
                for i, n in enumerate(self.card_names)}

    def _resolve(self, unit: str) -> tuple[int | None, int]:
        hit = self._resolved.get(unit)
        if hit is None:
            cid, form = self.classes.resolve(unit)
            hit = self._resolved[unit] = (cid, int(form))
        return hit

    def label(self, cls: str) -> tuple[str, int | None, int, str | None]:
        """A track label -> (type, class id, form, spell card name). type is "unit" (troops
        and buildings), "spell", "tower" (crown towers and tower troops) or what else the
        class table says ("object", "ui", "unknown", ...)."""
        hit = self._labels.get(cls)
        if hit is None:
            lab = self.classes.from_detector(cls)
            typ, cid = str(lab.type), lab.class_id
            if typ == _UNIT and cid is None:
                typ = "unknown"
            elif typ == _UNIT and self.classes[cid].kind == vision_classes.KIND_TOWER:
                typ = _TOWER
            hit = self._labels[cls] = (typ, cid, int(lab.form), lab.card)
        return hit

    def unit_type(self, class_id: int, form: int = FORM_BASE, card: int = -1) -> int:
        """The engine unit_types index for a unit of (class, form) that `card` put down: the
        card's E06 primary unit when it is of that class and form (Goblins -> Goblin_Stab,
        what the engine shows and what RSAgent.rebuilt can spawn); else the class's
        canonical unit of that form (ClassTable.unit_name), another engine unit of that
        class and form, the base form's (either way), any unit of the class; else -1."""
        c = self.classes[class_id]
        form = int(form) if int(form) in c.forms else c.base_form
        prim = self.primary.get(int(card))
        if (prim is not None and 0 <= prim < len(self.unit_types)
                and self._resolve(self.unit_types[prim]) == (c.id, form)):
            return prim
        key = (c.id, form)
        hit = self._unit_of.get(key)
        if hit is None:
            by_class = self._class_units()
            hit = self.unit_index.get(c.form(form).unit)
            if hit is None:
                hit = by_class.get((c.id, form))
            if hit is None:
                hit = self.unit_index.get(c.form(c.base_form).unit)
            if hit is None:
                hit = by_class.get((c.id, c.base_form))
            if hit is None:
                hit = min((i for (cid, _), i in by_class.items() if cid == c.id), default=-1)
            self._unit_of[key] = hit
        return hit

    def _class_units(self) -> dict[tuple[int, int], int]:
        """(class id, form) -> the first engine unit type of it, in unit_types order."""
        if self._by_class is None:
            out: dict[tuple[int, int], int] = {}
            for i, name in enumerate(self.unit_types):
                cid, form = self._resolve(name)
                if cid is not None:
                    out.setdefault((cid, form), i)
            self._by_class = out
        return self._by_class

    def max_hp(self, card: int, class_id: int, form: int = FORM_BASE,
               level: int = FRIENDLY_LEVEL, unit: str | None = None) -> int | None:
        """A unit's maximum hit points; None when no table knows. In order: the E06
        unit_hitpoints(card, level) row naming `unit` (the engine unit the entity gets) or
        the class's canonical unit of that form, else a row whose unit is of that class and
        form; the same in the rows of the class's primary card; ClassTable.hitpoints(class,
        form, level)."""
        c = self.classes[class_id]
        form = int(form) if int(form) in c.forms else c.base_form
        names = tuple(dict.fromkeys(n for n in (unit, c.form(form).unit) if n))
        key = (int(card), c.id, form, int(level), names)
        if key in self._hp:
            return self._hp[key]
        cards = [int(card)] if int(card) >= 0 else []
        primary = self.card_ids.get(c.primary_card) if c.primary_card else None
        if primary is not None and primary not in cards:
            cards.append(primary)
        hp = None
        for cc in cards:
            hp = self._row_hp(cc, int(level), names, (c.id, form))
            if hp is not None:
                break
        if hp is None:
            hp = self.classes.hitpoints(c.id, form, int(level))
            hp = int(hp) if hp and hp > 0 else None
        self._hp[key] = hp
        return hp

    def _row_hp(self, card: int, level: int, names: Sequence[str],
                want: tuple[int, int]) -> int | None:
        rows = [r for r in ((self.hp_table.get(str(card)) or {}).get(str(level)) or ())
                if len(r) >= 3 and int(r[2]) > 0]
        for name in names:
            for r in rows:
                if str(r[1]) == name:
                    return int(r[2])
        for r in rows:
            if self._resolve(str(r[1])) == want:
                return int(r[2])
        return None

    # ------------------------------------------------------------ the state
    def _team(self, track: Any) -> int | None:
        team = int(track.team)
        if team in (0, 1):
            return team
        if self.unknown_team == "drop":
            return None
        high = float(track.y_mt) > MID_Y_MT                # RoyaleSim team 1 defends high y
        return self.own_team if high == (self.own_team == 1) else self.enemy_team

    def _tick(self, vf: Any, tick: int | None, notes: list[str]) -> int:
        if tick is not None:
            return max(0, int(tick))
        if vf.tick is not None:
            return max(0, int(vf.tick))
        if vf.hud.clock_tick is not None:
            notes.append("tick from the HUD clock")
            return max(0, int(vf.hud.clock_tick))
        notes.append("tick unread: the newest track sighting")
        return max([int(tr.last_seen_tick) for tr in vf.tracks] or [0])

    def _row(self, values: Mapping[str, Any]) -> list:
        row = [list(v) if isinstance(v, list) else v for v in self._template]
        for name, v in values.items():
            i = self.F.get(name)
            if i is not None:
                row[i] = v
        return row

    def _spell_row(self, track: Any, team: int, card: int, tick: int) -> list:
        SF = self.SF
        row: list[Any] = [0] * (max(SF.values()) + 1)
        x, y = int(round(float(track.x_mt) * self.sub)), int(round(float(track.y_mt) * self.sub))
        values = {"team": team, "card_id": card, "x": x, "y": y, "aim_x": x, "aim_y": y,
                  "ticks_flown": max(0, int(tick) - int(track.first_seen_tick))}
        for name, v in values.items():
            if name in SF:
                row[SF[name]] = v
        return row

    def _unknown_entity(self, track: Any, tick: int, given: Mapping[int, int],
                        entities: list[list]) -> str | None:
        """unknown_units: enemy `track`, whose label the class table cannot place, appended to
        `entities` as an unknown unit (card_id its track_cards card or -1, unit_type -1, a
        ground troop, UNKNOWN_MAX_HP or its card's own hit points, deploying for
        UNKNOWN_DEPLOY_MS from its first sighting). None when added, else the reason it was
        not, as state()'s "dropped" counts it ("unknown" for a track of ours)."""
        x_mt, y_mt = float(track.x_mt), float(track.y_mt)
        if not (math.isfinite(x_mt) and math.isfinite(y_mt)):
            return "position"
        team = self._team(track)
        if team is None:
            return "team"
        if team != self.enemy_team:
            return "unknown"
        card = int(given.get(int(track.uid), UNKNOWN_CARD))
        card = card if card >= 0 else UNKNOWN_CARD
        level = self.levels[team]
        max_hp = (card_hitpoints(self.hp_table, card, level) if card >= 0 else None) or UNKNOWN_MAX_HP
        frac = 1.0 if track.hp_frac is None else min(1.0, max(0.0, float(track.hp_frac)))
        deploy_total = int(round(UNKNOWN_DEPLOY_MS / TICK_MS))
        entities.append(self._row({
            "uid": int(track.uid), "team": team, "kind": KIND_TROOP, "card_id": card,
            "tower_slot": -1, "x": int(round(x_mt * self.sub)), "y": int(round(y_mt * self.sub)),
            "hp": min(max_hp, max(1, int(round(frac * max_hp)))), "max_hp": max_hp,
            "flying": False,
            "deploy_ticks": min(deploy_total, deploy_ticks_left(track, tick, UNKNOWN_DEPLOY_MS)),
            "status_flags": STATUS_NONE, "level": level, "unit_type": -1,
        }))
        return None

    def _towers(self, hud: Any, side: str, notes: list[str]) -> tuple[list[int], list[int]]:
        hp, mx = [], []
        for pos in ("king", "left", "right"):
            key = f"{side}_{pos}"
            th = hud.towers.get(key)
            m = th.max if th is not None and th.max else self.tower_max_hp.get(key)
            h = th.hp if th is not None else None
            if h is None:
                h = m
                notes.append(f"{key} unread: full")
            h = max(0, int(h or 0))
            m = max(int(m or 0), h, 1)
            hp.append(h)
            mx.append(m)
        return hp, mx

    def state(self, vf: Any, *, tick: int | None = None, opp_deck: Iterable[Any] = (),
              track_cards: Mapping[int, int] | None = None, pending_cost: int = 0,
              observer: Any = None) -> dict:
        """The state dict of VisionFrame `vf`.

        tick          the battle tick (default: vf.tick, else the HUD clock, else the newest
                      sighting); entity deploy timers count from it
        opp_deck      the opponent cards known so far (ids or names): players[enemy]["deck"],
                      and the creating card of their units
        track_cards   uid -> creating card (EventDeriver.track_cards(vf.tracks)); preferred
                      over the class table's guess, which fills only the uids it lacks. A
                      negative card says the card is unknown: card_id -1 (live token 1,
                      "unknown"), the class's own unit type and hit points
        pending_cost  elixir of our plays sent but not yet on the board (RSAgent.act)
        observer      v2 rows (module doc, V2 ROWS): the battle's LiveObserver (anything with
                      inputs(tick) -> dict), or its inputs as a dict; the state then carries
                      them, at its tick, as state["observer"] (row_inputs reads them). None
                      (legacy rows): no such key, the state as it always was
        """
        hud = vf.hud
        notes: list[str] = []
        t = self._tick(vf, tick, notes)
        opp = _card_list(opp_deck, self.card_ids, "opp_deck")
        side_decks = {self.own_team: self.deck_names,
                      self.enemy_team: [self.card_names[c] for c in opp]}
        given = {int(k): int(v) for k, v in (track_cards or {}).items()}
        guessed: dict[tuple[int, int], int] = {}
        entities: list[list] = []
        spells: list[list] = []
        dropped: Counter = Counter()
        n_spells = n_unknown = 0
        for tr in vf.tracks:
            typ, cid, lform, spell_card = self.label(tr.cls)
            if typ == _SPELL:
                n_spells += 1
                card = given.get(int(tr.uid), self.card_ids.get(spell_card or "", -1))
                if self.SF is not None and card >= 0:
                    team = int(tr.team) if int(tr.team) in (0, 1) else self.enemy_team
                    spells.append(self._spell_row(tr, team, card, t))
                continue
            if typ != _UNIT:
                if typ == _UNKNOWN and self.unknown_units:      # N02: kept as an unknown unit
                    why = self._unknown_entity(tr, t, given, entities)
                    if why is None:
                        n_unknown += 1
                    else:
                        dropped[why] += 1
                    continue
                dropped[typ] += 1
                continue
            x_mt, y_mt = float(tr.x_mt), float(tr.y_mt)
            if not (math.isfinite(x_mt) and math.isfinite(y_mt)):
                dropped["position"] += 1
                continue
            team = self._team(tr)
            if team is None:
                dropped["team"] += 1
                continue
            c = self.classes[cid]
            form = int(tr.form) if int(tr.form) != FORM_BASE else lform
            form = form if form in c.forms else c.base_form
            info = c.form(form)
            card = given.get(int(tr.uid))
            if card is None:
                card = guessed.get((c.id, team))
                if card is None:
                    card = guessed[(c.id, team)] = int(
                        self.classes.card_id_for(c.id, self.card_names, side_decks[team]))
            level = self.levels[team]
            unit_type = self.unit_type(c.id, form, card)
            max_hp = self.max_hp(card, c.id, form, level,
                                 self.unit_types[unit_type] if unit_type >= 0 else None)
            if max_hp is None:
                max_hp = UNKNOWN_MAX_HP
                notes.append(f"no hit points for {c.name}: max_hp {UNKNOWN_MAX_HP}")
            frac = 1.0 if tr.hp_frac is None else min(1.0, max(0.0, float(tr.hp_frac)))
            hp = min(max_hp, max(1, int(round(frac * max_hp))))
            deploy_total = int(round(float(info.deploy_time_ms) / TICK_MS))
            entities.append(self._row({
                "uid": int(tr.uid), "team": team,
                "kind": KIND_BUILDING if info.building else KIND_TROOP,
                "card_id": card, "tower_slot": -1,
                "x": int(round(x_mt * self.sub)), "y": int(round(y_mt * self.sub)),
                "hp": hp, "max_hp": max_hp, "flying": bool(info.flying),
                "deploy_ticks": min(deploy_total, deploy_ticks_left(tr, t, info.deploy_time_ms)),
                "status_flags": STATUS_NONE, "level": level, "unit_type": unit_type,
            }))

        own_hp, own_max = self._towers(hud, "own", notes)
        opp_hp, opp_max = self._towers(hud, "enemy", notes)
        crowns = tuple(hud.crowns) if hud.crowns is not None else (0, 0)
        hand = [int(c) for c in list(hud.hand)[:HAND_SIZE]]
        hand += [EMPTY_CARD] * (HAND_SIZE - len(hand))
        if hud.elixir_milli is None:
            notes.append("elixir unread: 0")
        elixir = min(ELIXIR_MAX_MILLI, max(0, int(hud.elixir_milli or 0)))
        me = {"team": self.own_team, "elixir_milli": elixir, "hand": hand,
              "hand_costs": [self.costs.get(c, 0) if c >= 0 else 0 for c in hand],
              "next_card": int(hud.next) if int(hud.next) >= 0 else EMPTY_CARD,
              "deck": list(self.deck), "crowns": int(crowns[0]),
              "tower_hp": own_hp, "tower_max_hp": own_max,
              "king_active": _king_active(own_hp, own_max), "pending_cost": int(pending_cost)}
        them = {"team": self.enemy_team, "elixir_milli": 0, "hand": [], "hand_costs": [],
                "next_card": EMPTY_CARD, "deck": opp, "crowns": int(crowns[1]),
                "tower_hp": opp_hp, "tower_max_hp": opp_max,
                "king_active": _king_active(opp_hp, opp_max), "pending_cost": 0}
        players: list[dict] = [{}, {}]
        players[self.own_team], players[self.enemy_team] = me, them
        out = {
            "tick": t, "tick_ms": t * TICK_MS, "regular_ticks": REGULATION_TICKS,
            "overtime_ticks": OVERTIME_TICKS,
            "elixir_rate": ELIXIR_LAW.rate_at(t),
            "overtime": hud.phase == PHASE_OVERTIME or t >= REGULATION_TICKS,
            "players": players, "entities": entities, "spells": spells, "projectiles": [],
            "game_over": vf.screen == SCREEN_END or hud.phase == PHASE_ENDED, "winner": -1,
            "perception": {"tracks": len(vf.tracks), "units": len(entities), "spells": n_spells,
                           "dropped": dict(sorted(dropped.items())), "notes": notes},
        }
        if self.unknown_units:
            out["perception"]["unknown_units"] = n_unknown
        if observer is not None:
            inputs = observer if isinstance(observer, Mapping) else observer.inputs(t)
            out["observer"] = _observer_block(inputs)
        return out


def _king_active(hp: Sequence[int], mx: Sequence[int]) -> bool:
    """The king tower wakes when it is damaged or a princess tower falls."""
    return hp[0] < mx[0] or hp[1] == 0 or hp[2] == 0


def rs_state_from_vision(vf: Any, deck: Iterable[Any], classes: Any = None,
                         spawn_tables: Mapping[str, Any] | None = None,
                         levels: Any = FRIENDLY_LEVEL, *,
                         catalogue: Sequence[Sequence[Any]] | None = None,
                         fields: Mapping[str, int] | Sequence[str] | None = None,
                         sub: int = SUBTILE_PER_MILLITILE, own_team: int = OWN_TEAM,
                         unknown_team: str = "half", tower_max_hp: Mapping[str, int] | None = None,
                         spell_fields: Mapping[str, int] | Sequence[str] | None = None,
                         tick: int | None = None, opp_deck: Iterable[Any] = (),
                         track_cards: Mapping[int, int] | None = None,
                         pending_cost: int = 0, observer: Any = None,
                         unknown_units: bool = False) -> dict:
    """PublicStateBuilder(classes, spawn_tables, deck=deck, ...).state(vf, ...) in one call
    (the arguments are those two's). A loop should keep one builder: it caches lookups."""
    return PublicStateBuilder(classes, spawn_tables, deck=deck, levels=levels, catalogue=catalogue,
                              fields=fields, sub=sub, own_team=own_team, unknown_team=unknown_team,
                              tower_max_hp=tower_max_hp, spell_fields=spell_fields,
                              unknown_units=unknown_units).state(
        vf, tick=tick, opp_deck=opp_deck, track_cards=track_cards, pending_cost=pending_cost,
        observer=observer)


# ---------------------------------------------------------------------------
# v2 rows: what the bot knows besides the board (module doc, V2 ROWS)
# ---------------------------------------------------------------------------

class OpponentElixir:
    """The opponent's elixir bar as the bot counts it from the plays perception reports.

    catalogue  Battle.catalogue_json() rows of the engine the model was trained on (the
               prices; a Mirror by placement code 6 or name); fields: royalesim.
               CATALOGUE_FIELDS (None: rs_elixir.CATALOGUE_PREFIX's order)
    law        the rs_elixir.ElixirLaw of that engine (rs_rows.elixir_law(engine): its
               calibration under the config's overrides); None: rs_elixir.default_law()
    start_tick the battle tick the bot started watching at (the runner's battle start)
    blind      True when no play can be seen (no detector)

    record(plays) takes scripts/vision OppPlays (card, exec_tick; the first sighting when
    exec_tick is None) and keeps each until view reaches its tick; view(tick) feeds the ones due
    (exec_tick <= tick) to the counter and returns (elixir_milli(tick), exact): the bar after
    the plays dated `tick`, what rs_rows.SeenElixir(lag=0).view gives for the same plays. A
    play reported late is paid at its own tick. A play with an OppPlay.cost is paid that
    (ElixirCounter.play(cost=): a Spirit Empress form's 3 or 6). An UNKNOWN play (card -1, N02:
    events.EventConfig.unknown_plays) is paid its cost, the expected one the deriver gives it,
    and counted in `unknown`; the count is then not exact (the rows' opp_exact 0), since the
    card and so its price are a guess. A card the catalogue does not price, with no cost, is
    skipped and counted in `unpriced` (an UNKNOWN play without a cost, too); the count is then
    not exact. Training rows price an unknown play differently: rs_rows / rs_eval skip it (not
    paid, exactness kept), so a count with UNKNOWN plays is one the model has not seen.
    """

    def __init__(self, catalogue: Sequence[Sequence[Any]], *,
                 fields: Sequence[str] | None = None, law: Any = None,
                 start_tick: int | None = 0, blind: bool = False):
        self.counter = rs_elixir.ElixirCounter.from_catalogue(catalogue, fields=fields, law=law)
        self.start_tick = 0 if start_tick is None else max(0, int(start_tick))
        self.blind = bool(blind)
        self.recorded = 0
        self.unpriced = 0
        self.unknown = 0
        self._waiting: list[tuple[int, int, int | None]] = []     # (tick, card, cost or None)

    @property
    def late(self) -> bool:
        """Whether the bot started watching after the deploy lockout and one more displayed
        second (LATE_MARGIN_TICKS: a first clock reading of 2:55 is in time): plays may be
        unseen."""
        return self.start_tick > DEPLOY_LOCKOUT_TICKS + LATE_MARGIN_TICKS

    @property
    def exact(self) -> bool:
        """The row's opp_exact: the counter's own exactness, and the bot can have seen every
        play (not blind, not late, every play priced, no UNKNOWN play)."""
        return bool(self.counter.exact and not self.blind and not self.late
                    and not self.unpriced and not self.unknown)

    def record(self, plays: Iterable[Any]) -> int:
        """Log the opponent plays `plays` (OppPlay); returns how many were taken."""
        n = 0
        for p in () if plays is None else plays:
            card = int(p.card)
            when = p.exec_tick if p.exec_tick is not None else p.tick
            cost = getattr(p, "cost", None)
            if cost is None and card not in self.counter.costs:
                self.unpriced += 1
                continue
            if card < 0:                                   # UNKNOWN_CARD: its cost is a guess
                self.unknown += 1
            self._waiting.append((max(0, int(when)), card, None if cost is None else max(0, int(cost))))
            n += 1
        self.recorded += n
        return n

    def view(self, tick: int) -> tuple[int, bool]:
        """(milli, exact): the opponent's bar at `tick` as the bot counts it."""
        tick = max(0, int(tick))
        if self._waiting:
            due = [e for e in self._waiting if e[0] <= tick]
            if due:
                self._waiting = [e for e in self._waiting if e[0] > tick]
                for when, card, cost in due:
                    self.counter.play(when, card, cost=cost)
        return int(self.counter.elixir_milli(tick)), self.exact


class OwnTaps:
    """Our plays sent and not on the board yet, what a v2 row's own_pending shows live.

    sent(card, tick, slot=, cost=) logs a tap the runner sent (rs_decide.Pending); landed(
    landings) drops the tap each scripts/vision OwnLanding of a tap (slot >= 0: landed, or
    missed: land_tick None) names by (card, slot, tap_tick); a tap at least `max_age` ticks old
    is dropped too (rs_decide.prune_pending's age rule). inputs(now, delay) is
    rs_decide.pending_inputs of what is left: (card + 2, max(0, tap + delay - now)) soonest
    first, nothing for a tap at `now` (the decision's own) or for delay 0.
    """

    def __init__(self, *, max_age: int = TAP_MAX_AGE):
        self.max_age = int(max_age)
        self.taps: list[Any] = []
        self.landed_count = 0
        self.expired = 0

    def sent(self, card: int, tick: int, *, slot: int = -1, cost: int = 0) -> None:
        self.taps.append(rs_decide.Pending(int(card), max(0, int(cost)), int(slot), int(tick)))

    def landed(self, landings: Iterable[Any]) -> int:
        """Drop the taps `landings` report; returns how many were dropped. An OwnLanding of
        slot -1 is a play no tap explains (a human's): no tap of ours."""
        n = 0
        for x in () if landings is None else landings:
            if int(x.slot) < 0:
                continue
            key = (int(x.card), int(x.slot), int(x.tap_tick))
            for k, p in enumerate(self.taps):
                if (p.card, p.slot, p.tick) == key:
                    del self.taps[k]
                    n += 1
                    break
        self.landed_count += n
        return n

    def prune(self, now: int) -> int:
        """Drop the taps at least max_age ticks old at `now`; returns how many."""
        now = int(now)
        keep = [p for p in self.taps if p.tick is None or now - p.tick < self.max_age]
        dropped = len(self.taps) - len(keep)
        self.taps = keep
        self.expired += dropped
        return dropped

    def inputs(self, now: int, delay: int) -> list[tuple[int, int]]:
        """obs_row's `pending` at `now` (prune first)."""
        self.prune(now)
        return rs_decide.pending_inputs(self.taps, int(now), int(delay))


class LiveObserver:
    """One battle's v2 row inputs (module doc, V2 ROWS): OpponentElixir `opp` and OwnTaps `own`.

    delay      the tap-to-run delay the checkpoint's rows were built with
               (rs_decide.obs_delay_ticks(meta)); 0 for landing-time rows (no pending shown)
    catalogue, fields, law, start_tick, blind   OpponentElixir's
    max_age    OwnTaps'

    events(vf.events) once per frame of the battle (the opponent plays and our landings it
    reports); sent(card, tick, slot=, cost=) for each tap sent; inputs(tick) -> {"opp_elixir":
    [milli, exact], "own_pending": [[card token, ticks to land], ...], "delay": delay}, what
    PublicStateBuilder.state(..., observer=) stores and row_inputs reads.
    """

    def __init__(self, catalogue: Sequence[Sequence[Any]], *, delay: int = 0,
                 fields: Sequence[str] | None = None, law: Any = None,
                 start_tick: int | None = 0, blind: bool = False, max_age: int = TAP_MAX_AGE):
        self.delay = max(0, int(delay or 0))
        self.opp = OpponentElixir(catalogue, fields=fields, law=law, start_tick=start_tick,
                                  blind=blind)
        self.own = OwnTaps(max_age=max_age)

    def events(self, events: Any) -> None:
        """Feed one frame's scripts/vision Events: opp_plays and own_landings."""
        if events is None:
            return
        self.opp.record(getattr(events, "opp_plays", None) or ())
        self.own.landed(getattr(events, "own_landings", None) or ())

    def sent(self, card: int, tick: int, *, slot: int, cost: int = 0) -> None:
        self.own.sent(card, tick, slot=slot, cost=cost)

    def inputs(self, tick: int) -> dict:
        milli, exact = self.opp.view(tick)
        return {"opp_elixir": [int(milli), bool(exact)],
                "own_pending": [[int(c), int(t)] for c, t in self.own.inputs(tick, self.delay)],
                "delay": self.delay}


def _observer_block(inputs: Mapping[str, Any]) -> dict:
    """state["observer"]: the observer's inputs as plain JSON values (lists)."""
    milli, exact = inputs["opp_elixir"]
    return {"opp_elixir": [int(milli), bool(exact)],
            "own_pending": [[int(c), int(t)] for c, t in inputs.get("own_pending") or ()],
            "delay": int(inputs.get("delay") or 0)}


def row_inputs(state: Mapping[str, Any], buttons: Sequence[int] | None = None) -> dict:
    """tools/rs_rows.obs_row's v2 keywords from a state built with an observer
    (rs_decide.v2_row_kwargs: schema rs_rows/v2, opp_elixir (milli, exact), pending [(card
    token, ticks to land)], and `buttons` when given). ValueError for a state without one."""
    obs = state.get("observer")
    if not isinstance(obs, Mapping) or "opp_elixir" not in obs:
        raise ValueError("this state has no observer inputs: v2 rows need PublicStateBuilder."
                         "state(..., observer=the battle's LiveObserver)")
    milli, exact = obs["opp_elixir"]
    pending = [(int(c), int(t)) for c, t in obs.get("own_pending") or ()]
    return rs_decide.v2_row_kwargs((int(milli), bool(exact)), pending, buttons)


# ---------------------------------------------------------------------------
# the rebuild sanitiser (vision-02)
# ---------------------------------------------------------------------------

def rebuild_card(card: int, unit: int, primary: Mapping[int, int], by_unit: Mapping[int, int],
                 unit_names: Sequence[str]) -> int | None:
    """The card RSAgent.rebuilt (tools/rs_eval.py) spawns for an entity (card_id, unit_type),
    the same rule: the card itself when `unit` is its primary unit or a form of it
    (Skeleton_EV1 from Skeletons), else the lowest card whose primary unit `unit` is (a
    Witch's skeleton spawns as Skeletons), else None (rebuilt skips it). primary / by_unit:
    rs_spawn_table.primary_of / by_unit. Every such card is a troop or building. One
    difference: unit_type -1 is never matched by name (rebuilt would compare unit_names[-1])."""
    card, unit = int(card), int(unit)
    if primary.get(card) == unit:
        return card
    if card in primary and 0 <= unit < len(unit_names) and 0 <= primary[card] < len(unit_names):
        base = str(unit_names[primary[card]]).lower()
        if base and str(unit_names[unit]).lower().startswith(base):
            return card
    hit = by_unit.get(unit)
    return None if hit is None else int(hit)


def card_hitpoints(hp_table: Mapping[str, Any], card: int, level: int) -> int | None:
    """The hit points of `card`'s own unit at `level` (E06 "hp": the "own" row), or None."""
    rows = (hp_table.get(str(int(card))) or {}).get(str(int(level))) or ()
    for r in rows:
        if len(r) >= 3 and r[0] == "own" and int(r[2]) > 0:
            return int(r[2])
    return None


def into_arena(x_mt: float, y_mt: float, margin_mt: float = EDGE_MARGIN_MT) -> tuple[float, float]:
    """A world point clamped onto the 18x32 board, `margin_mt` inside its edge."""
    m = max(0.0, float(margin_mt))
    return (min(max(float(x_mt), m), G.WIDTH_MT - m), min(max(float(y_mt), m), G.HEIGHT_MT - m))


def off_water(x_mt: float, y_mt: float, margin_mt: float = WATER_MARGIN_MT, *,
              own_high: bool = True) -> tuple[float, float]:
    """A ground unit's world point moved off the river (geometry.RIVER_Y_MT / BRIDGES_X_MT):
    unchanged when it is `margin_mt` clear of water or on a bridge `margin_mt` inside its
    sides; else the nearest of the two banks (`margin_mt` from the water, x kept) and the
    bridges (x clamped `margin_mt` inside, y kept). Ties go to the unit's own bank first
    (own_high: its team defends high y, RoyaleSim team 1), then the other, then the bridges
    left to right."""
    m = max(1.0, float(margin_mt))
    x, y = float(x_mt), float(y_mt)
    lo, hi = G.RIVER_Y_MT
    if not lo - m < y < hi + m:
        return x, y
    bridges = [(a + m, b - m) for a, b in G.BRIDGES_X_MT if b - a > 2 * m]
    if any(a <= x <= b for a, b in bridges):
        return x, y
    banks = [(x, hi + m), (x, lo - m)] if own_high else [(x, lo - m), (x, hi + m)]
    candidates = banks + [(min(max(x, a), b), y) for a, b in bridges]
    return min(candidates, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)


def _deck8(deck: Iterable[Any], filler: Sequence[int]) -> list[int]:
    out: list[int] = []
    for c in () if deck is None else deck:
        c = int(c)
        if c >= 0 and c not in out:
            out.append(c)
    out = out[:8]
    for c in filler:
        if len(out) >= 8:
            break
        if c not in out:
            out.append(c)
    return out


def sanitize_for_rebuild(state: Mapping[str, Any], spawn_table: Mapping[str, Any] | None = None, *,
                         fields: Mapping[str, int] | Sequence[str] | None = None,
                         sub: int = SUBTILE_PER_MILLITILE, levels: Any = FRIENDLY_LEVEL,
                         water_margin_mt: float = WATER_MARGIN_MT,
                         edge_margin_mt: float = EDGE_MARGIN_MT,
                         filler: Sequence[Any] | None = None) -> dict:
    """A copy of `state` that RSAgent.rebuilt can reset a Battle from (module doc; vision-02).

    Units (tower_slot < 0): dropped when dead, of no team, or when rebuilt would spawn no card
    for them (rebuild_card; spells never qualify); moved into the arena (edge_margin_mt) and,
    unless flying, off the river (off_water, water_margin_mt); hp capped to [1, the hit points
    of the card rebuilt spawns at the unit's level] (card_hitpoints, else the entity's max_hp).
    Crown-tower entities are kept as they are (rebuilt reads tower_hp). Each player's deck
    becomes 8 distinct cards: repeats and negatives dropped, cut to 8, padded from `filler`
    (ids or names; default rs_spawn_table.FILLER_NAMES, Hog 2.6, then the spawnable cards by
    id) - the opponent's padding never plays in a look-ahead. Elixir is clamped to
    [0, 10000], tower hp to [0, max] (a king to at least 1: reset refuses a king at 0, so a
    misread king would cost the decision), the tick to >= 0. The input is not changed; the copy
    carries "sanitized": {"dropped": {reason: n}, "into_arena", "off_water", "hp_capped",
    "padded": {team: cards added}}. Live, give rebuilt (ahead with lead_rebuild) this copy;
    obs_row may read either (positions move by at most the margins).
    """
    table = spawn_table if spawn_table is not None else load_spawn_table()
    F = _entity_fields(fields)
    sub = int(sub)
    primary = rs_spawn_table.primary_of(table) if table.get("primary") else {}
    by_unit = rs_spawn_table.by_unit(primary)
    unit_names = [str(u) for u in table.get("unit_types") or ()]
    card_ids = {str(n): i for i, n in enumerate(table.get("card_names") or ())}
    hp_table = table.get("hp") or {}
    lv = team_levels(levels)
    out = copy.deepcopy(dict(state))
    dropped: Counter = Counter()
    report: dict[str, Any] = {"dropped": {}, "into_arena": 0, "off_water": 0, "hp_capped": 0,
                              "padded": {}}
    kept = []
    for e in out.get("entities") or ():
        if e[F["tower_slot"]] >= 0:
            kept.append(e)
            continue
        if e[F["hp"]] is None or e[F["hp"]] <= 0:
            dropped["dead"] += 1
            continue
        team = int(e[F["team"]])
        if team not in (0, 1):
            dropped["team"] += 1
            continue
        card = rebuild_card(e[F["card_id"]], e[F["unit_type"]], primary, by_unit, unit_names)
        if card is None:
            dropped["no spawn card"] += 1
            continue
        level = int(e[F["level"]] or 0) or lv[team]
        cap = card_hitpoints(hp_table, card, level)
        if cap is None and e[F["max_hp"]] and int(e[F["max_hp"]]) > 0:
            cap = int(e[F["max_hp"]])
        hp = max(1, int(e[F["hp"]]))
        if cap is not None:
            hp = min(hp, cap)
        if hp != e[F["hp"]]:
            e[F["hp"]] = hp
            report["hp_capped"] += 1
        x, y = e[F["x"]] / sub, e[F["y"]] / sub
        ax, ay = into_arena(x, y, edge_margin_mt)
        if (ax, ay) != (x, y):
            report["into_arena"] += 1
        if not e[F["flying"]]:
            wx, wy = off_water(ax, ay, water_margin_mt, own_high=team == 1)
            if (wx, wy) != (ax, ay):
                report["off_water"] += 1
            ax, ay = wx, wy
        if (ax, ay) != (x, y):
            e[F["x"]], e[F["y"]] = int(round(ax * sub)), int(round(ay * sub))
        kept.append(e)
    out["entities"] = kept

    fill: list[int] = []
    for c in (rs_spawn_table.FILLER_NAMES if filler is None else filler):
        c = card_ids.get(c) if isinstance(c, str) else int(c)
        if c is not None and c >= 0 and c not in fill:
            fill.append(c)
    fill += [c for c in sorted(primary) if c not in fill]
    for t, p in enumerate(out.get("players") or ()):
        before = len(_deck8(p.get("deck"), ()))
        p["deck"] = _deck8(p.get("deck"), fill)
        if len(p["deck"]) > before:
            report["padded"][t] = len(p["deck"]) - before
        if "elixir_milli" in p:
            p["elixir_milli"] = min(ELIXIR_MAX_MILLI, max(0, int(p["elixir_milli"] or 0)))
        if "tower_hp" in p:
            mx = list(p.get("tower_max_hp") or [])
            hp = [max(0, min(int(h or 0), int(mx[k]))) if k < len(mx) and mx[k]
                  else max(0, int(h or 0)) for k, h in enumerate(p["tower_hp"])]
            if hp and hp[0] < 1:
                hp[0] = 1           # reset refuses a king at 0 (a princess at 0 starts destroyed)
            p["tower_hp"] = hp
    out["tick"] = max(0, int(out.get("tick") or 0))
    report["dropped"] = dict(sorted(dropped.items()))
    out["sanitized"] = report
    return out
