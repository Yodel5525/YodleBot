"""Opponent plays and our own landings, derived from the tracker's tracks (V04).

The token model reads the opponent's public memory: the cards it has been seen
to play and its last 4 plays with their ages (tools/rs_rows.obs_row,
tools/rs_memory.PublicMemory). On screen there is no play log, only units and
spell effects appearing. This module turns the tracks of each frame into
`types.Events`: opponent plays (OppPlay) with an estimated execution tick, and
our own taps matched to the units they produced (OwnLanding), with latency and
miss statistics.

THE RULES. A new track (a uid the deriver has not seen) is classified once its
team is decided (or after ``team_wait_ticks``, by a fallback). In order:

1. Re-acquired. A track of the same class and team that vanished at most
   ``reid_ticks`` ago, close to where it was lost: the same unit under a new
   uid, not a play.
2. Merged. Another unit of a recent play of the same team whose card puts this
   class down: within PLAY_WINDOW = 20 ticks of that play's first sighting and
   ``merge_radius_mt`` of it (3 Skeletons, a Goblin Gang's spear goblins). For
   spells the window chains from the play's LAST sighting (a rolling Log
   re-acquired downstream; a Poison seen for 8 s).
3. Child. A living or just-dead parent of the same team (seen at most
   ``just_dead_ticks`` before the child, within ``child_radius_mt``) whose
   class creates this class (EventTables.children: spawners, death spawns,
   projectile and area spawns; "*" for Clone): a Witch's skeletons, a Golem's
   golemites, a Goblin Barrel's goblins.
4. Own tap (our team only). The oldest pending tap of a card that puts this
   class down, sent ``tap_min_lag_ticks``..``tap_max_lag_ticks`` before the
   first sighting and (units) landed within ``landing_radius_mt`` of the aim;
   failing that, any pending tap aimed within ``aim_near_mt`` (perception
   misread the card). That is a landing; taps never seen are reported as misses.
5. Otherwise a NEW PLAY of the card whose deploy-time units include the class
   (a card's "primary unit"), or the spell a spell class shows. When several
   cards share the class (Skeleton: Skeletons or Skeleton Army; Goblin: Goblins
   or Goblin Gang), the play waits ``ambiguity_wait_ticks``, collects the units
   that join it, and takes the card whose composition fits the counts best.
   Our own unmatched plays (a human playing in MANUAL mode) are reported as an
   OwnLanding with slot -1 (scripts/vision/control.as_own_play reads that).
6. UNKNOWN (EventConfig.unknown_plays, OFF by default; docs/NEW_CARDS_PLAN.md 2.2).
   An ENEMY track rules 1-5 leave unexplained (no card's units include its class:
   a card the tables lack, a hero's or an evolution's extra unit, a misread) joins
   the newest UNKNOWN play of its team first seen less than PLAY_WINDOW ticks from
   it and within ``merge_radius_mt`` (another unit of the same unnamed card), or
   starts one: OppPlay(card=UNKNOWN_CARD (-1), cost=``unknown_cost``).
   record_plays writes it as row token 1 ("unknown", rs_memory.UNKNOWN), and
   tools/public_state.OpponentElixir charges it that cost and stops calling its
   count exact. Its tracks keep card_of None: the board names them by the class
   table, as before. With the option off such a track is only counted in
   OppStats.unknown, and nothing is reported.

Every classified track also gets a CARD-LEVEL identity (card_of / card_for):
the card that put it on the board, so a Witch's skeleton is the Witch's. That
is what the legacy --live-compat rows use as the unit token (rs_rows.py: unit
id = creating card); the finer vision class stays on the Track. how_of(uid)
says which rule explained a track (N05: the harvest tools read it).

TIMING. OppPlay.tick is the first sighting (Track.first_seen_tick). exec_tick
estimates when the play ran: tick minus a sight lag (``unit_sight_lag_ticks``,
``spell_sight_lag_ticks``, per card ``sight_lag_by_card``; 0 by default, since a
troop is drawn with its deploy stopwatch from the tick it runs). Feed it to
PublicMemory with record_plays(); LandingStats.sight_lag_estimate() measures the
lag from our own landings. Ticks are battle ticks (20 per second).

COST. OppPlay.cost is None (the catalogue's price) except for a card whose forms
cost differently, where it is EventTables.play_cost of the classes the play
showed (the Spirit Empress: 3 or 6), and for an UNKNOWN play (rule 6).

TABLES. EventTables holds, per RoyaleSim card id, the classes one play puts
down (with counts) or, for a spell, the classes that show it; and the
parent -> child class map. A spell that deploys a troop has a composition, not
spell classes (Heal -> HealSpirit). A card with forms (the Spirit Empress
MergeMaiden: MergeMaiden_Normal below 6 elixir, MergeMaiden_Mounted at 6 or more)
holds its forms' classes, so their units name the card; the forms' own ids are
aliases of it, and play_cost(card, classes) gives the form's cost (3 or 6). Build
it from:
* EventTables.from_names(load_spawn_children(), card_ids,
  class_of=unit_class_map(classes), ignore=cardless_classes(classes)):
  data/vision/spawn_children.json, generated from RoyaleSim's card tables (the
  16.402.22 one data/engine.json selects, then 15.535's) by
  tools/vision_spawn_children.py (names; card_ids from Battle.catalogue_json;
  classes = data/vision/classes.json, E16);
* EventTables.for_engine(spawn_primary_json, class_of=...): E06's
  data/rs_rows/spawn_primary.json (this engine's card ids and primary unit
  types) on top of from_names (counts, children, spells);
* or directly, as the tests do.

Conventions are those of scripts/vision/types.py: world millitiles, RoyaleSim
team ids; the deriver's own team is OWN_TEAM (1) unless told otherwise (an
offline run on an engine seat 0 passes own_team=0).

    from scripts.vision import events as EV
    deriver = EV.EventDeriver(tables, own_deck=deck_ids)       # EventConfig(unknown_plays=True): rule 6
    deriver.note_own_tap(card, tick, slot=slot, aim_x_mt=x, aim_y_mt=y, t=EV.now(), tracker=tracker)
    vf.events = deriver.update(tracks, vf.tick, t=vf.t_capture, tracker=tracker)
    EV.record_plays(public_memory, vf.events.opp_plays)        # card + 2 at exec_tick
    card_ids = deriver.track_cards(tracks)                     # uid -> creating card

StateTracks turns RoyaleSim state_json dicts into Tracks, so the same deriver
runs on engine battles (evaluation with perception-derived memory).

Imports only the standard library.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .tracker import MID_Y_MT, TICK_MS
from .types import (ENEMY_TEAM, OWN_TEAM, UNKNOWN_CARD, UNKNOWN_TEAM, Events, OppPlay, OwnLanding,
                    Track, now)

__all__ = ["PLAY_WINDOW", "TOKEN_OFFSET", "KIND_UNIT", "KIND_SPELL", "WILDCARD", "SPAWN_CHILDREN",
           "KATACR_SPELL_CLASSES", "UNKNOWN_CARD", "UNKNOWN_PLAY_COST", "EventTables", "EventConfig",
           "EventDeriver", "OwnTap", "LandingStats", "OppStats", "StateTracks", "load_spawn_children",
           "unit_class_map", "cardless_classes", "record_plays", "strip_form", "now"]

ROOT = Path(__file__).resolve().parents[2]
SPAWN_CHILDREN = ROOT / "data" / "vision" / "spawn_children.json"

PLAY_WINDOW = 20               # ticks: same card, same play (scripts/gamestate/live_play.py)
TOKEN_OFFSET = 2               # row token = catalogue id + 2 (tools/rs_memory.TOKEN_OFFSET)
KIND_UNIT, KIND_SPELL = "unit", "spell"     # OppPlay.kind
WILDCARD = "*"                 # in EventTables.children: explains a child of any class
TICK_S = TICK_MS / 1000.0
SPELL_UID_OFFSET = 1 << 30     # StateTracks: spell uids live above the entity uids
#: Elixir an UNKNOWN play is charged (EventConfig.unknown_cost): about the mean cost of the
#: cards KataCR has no class for, weighted by the 1,467 top-ladder decks that hold them
#: (4.04, docs/NEW_CARDS_PLAN.md 1.2), and a common card cost.
UNKNOWN_PLAY_COST = 4
#: EventDeriver.how_of of a track still waiting for its team.
HOW_PENDING = "pending"

#: KataCR detector names of spells, by RoyaleSim card name (the "spell" entries of
#: data/vision/classes.json "katacr", E16). EventTables.from_names adds them to each
#: spell's classes.
KATACR_SPELL_CLASSES: dict[str, tuple[str, ...]] = {
    "Arrows": ("arrows",), "BarbLog": ("barbarian-barrel",), "Clone": ("clone",),
    "Earthquake": ("earthquake",), "Fireball": ("fireball",), "Freeze": ("freeze",),
    "GoblinBarrel": ("goblin-barrel",), "Graveyard": ("graveyard",), "Lightning": ("lightning",),
    "Log": ("the-log",), "Mirror": ("mirror",), "Poison": ("poison",), "Rage": ("rage",),
    "Rocket": ("rocket",), "RoyalDelivery": ("royal-delivery",), "Snowball": ("giant-snowball",),
    "Tornado": ("tornado",), "Zap": ("zap", "zap-evolution"),
}
#: Spells whose children can be of any class (the copies a Clone makes).
CLONE_CARDS = ("Clone",)

_FORM_SUFFIX = re.compile(r"(_EV\d+|Hero)$")


def strip_form(name: str) -> str:
    """A RoyaleSim unit name without its form suffix: Skeleton_EV1 -> Skeleton,
    MusketeerHero -> Musketeer (the default class mapping of StateTracks)."""
    return _FORM_SUFFIX.sub("", name) or name


def _card_id(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("a card id must be an int, not a bool")
    card = int(value)                 # JSON object keys arrive as strings
    if card < 0:
        raise ValueError(f"card ids are >= 0, got {card}")
    return card


def _counts(value: Any) -> dict[str, int]:
    """'A' | ['A', 'A', 'B'] | {'A': 2} -> {'A': 2, ...}."""
    if isinstance(value, str):
        return {value: 1}
    if isinstance(value, Mapping):
        return {str(k): int(v) for k, v in value.items()}
    return dict(Counter(str(v) for v in value))


def _class_mapper(class_of: Callable[[str], str | None] | Mapping[str, str | None] | None
                  ) -> Callable[[str], str | None]:
    """None -> identity; a mapping -> lookup (unlisted names keep their name); a callable as is."""
    if class_of is None:
        return lambda name: name
    if isinstance(class_of, Mapping):
        return lambda name: class_of.get(name, name)
    return class_of


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EventTables:
    """What a play puts on the board, per RoyaleSim card id.

    compositions  troop/building card -> {class: units one play deploys}
                  (Skeletons {Skeleton: 3}, Goblin Gang {Goblin: 3, SpearGoblin: 3}); also
                  a spell that deploys a troop (Heal {HealSpirit: 1}) and a card with
                  forms, which holds every form's classes (aliases).
    spells        spell card -> the classes that show it (a projectile, an area).
    children      class -> classes it creates later while alive or as it dies
                  (Witch -> Skeleton, Golem -> Golemite, a Goblin Barrel's class
                  -> Goblin); WILDCARD explains any class.
    ignore        classes never counted (towers, tower troops, effects).
    aliases       card id of a FORM -> the card a play of it is reported as (the
                  Spirit Empress's MergeMaiden_Normal and _Mounted -> MergeMaiden). The
                  form's classes are in the card's composition; a form has no
                  composition or spell of its own.
    variant_costs card -> {class: elixir of the form that puts it down}, for a card
                  whose forms cost differently (MergeMaiden: MergeMaiden_Normal 3,
                  MergeMaiden_Mounted 6); play_cost reads it.
    Mapping keys may be strings (JSON); values may be lists, sets or count dicts.
    """
    compositions: Mapping[int, Mapping[str, int]]
    spells: Mapping[int, Iterable[str]] = field(default_factory=dict)
    children: Mapping[str, Iterable[str]] = field(default_factory=dict)
    ignore: Iterable[str] = frozenset()
    aliases: Mapping[int, int] = field(default_factory=dict)
    variant_costs: Mapping[int, Mapping[str, int]] = field(default_factory=dict)
    _unit_by: dict = field(init=False, repr=False, compare=False)
    _spell_by: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        comps: dict[int, dict[str, int]] = {}
        for card, comp in dict(self.compositions).items():
            card = _card_id(card)
            cc: dict[str, int] = {}
            for cls, n in _counts(comp).items():
                if n < 1:
                    raise ValueError(f"card {card}: {cls} count must be >= 1, got {n}")
                cc[cls] = cc.get(cls, 0) + n
            if not cc:
                raise ValueError(f"card {card} has an empty composition")
            comps[card] = cc
        spells: dict[int, frozenset[str]] = {}
        for card, classes in dict(self.spells).items():
            card = _card_id(card)
            if card in comps:
                raise ValueError(f"card {card} is both a unit card and a spell")
            s = frozenset([classes] if isinstance(classes, str) else (str(c) for c in classes))
            if not s:
                raise ValueError(f"spell {card} has no classes")
            spells[card] = s
        children = {str(p): frozenset([k] if isinstance(k, str) else (str(c) for c in k))
                    for p, k in dict(self.children).items()}
        unit_by: dict[str, list[int]] = {}
        for card in sorted(comps):
            for cls in comps[card]:
                unit_by.setdefault(cls, []).append(card)
        spell_by: dict[str, list[int]] = {}
        for card in sorted(spells):
            for cls in spells[card]:
                spell_by.setdefault(cls, []).append(card)
        clash = sorted(set(unit_by) & set(spell_by))
        if clash:
            raise ValueError(f"classes both a unit and a spell: {clash}")
        aliases: dict[int, int] = {}
        for form, card in dict(self.aliases).items():
            form, card = _card_id(form), _card_id(card)
            if form == card:
                raise ValueError(f"card {card} cannot be a form of itself")
            if form in comps or form in spells:
                raise ValueError(f"card {form} is a form of card {card}: it has no composition "
                                 "or spell of its own")
            aliases[form] = card
        chained = sorted(set(aliases) & set(aliases.values()))
        if chained:
            raise ValueError(f"cards both a form and a card with forms: {chained}")
        costs: dict[int, dict[str, int]] = {}
        for card, by_cls in dict(self.variant_costs).items():
            card = _card_id(card)
            if card not in comps:
                raise ValueError(f"variant costs for card {card}, which has no composition")
            cc: dict[str, int] = {}
            for cls, cost in dict(by_cls).items():
                cls, cost = str(cls), int(cost)
                if cls not in comps[card]:
                    raise ValueError(f"card {card}: no form of it puts down {cls}")
                if cost < 0:
                    raise ValueError(f"card {card}: {cls} costs {cost} < 0")
                cc[cls] = cost
            if cc:
                costs[card] = cc
        object.__setattr__(self, "compositions", comps)
        object.__setattr__(self, "spells", spells)
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "ignore", frozenset(self.ignore))
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "variant_costs", costs)
        object.__setattr__(self, "_unit_by", {k: tuple(v) for k, v in unit_by.items()})
        object.__setattr__(self, "_spell_by", {k: tuple(v) for k, v in spell_by.items()})

    # -- lookups ---------------------------------------------------------------
    def unit_cards(self, cls: str) -> tuple[int, ...]:
        """Cards whose deploy-time units include `cls`, by id."""
        return self._unit_by.get(cls, ())

    def spell_cards(self, cls: str) -> tuple[int, ...]:
        """Spell cards `cls` shows, by id."""
        return self._spell_by.get(cls, ())

    def is_spell(self, cls: str) -> bool:
        return cls in self._spell_by

    def kind_of(self, card: int) -> str | None:
        return KIND_SPELL if card in self.spells else KIND_UNIT if card in self.compositions else None

    def classes_of(self, card: int) -> frozenset[str]:
        """Every class one play of `card` shows (for the tracker's note_own_tap)."""
        if card in self.spells:
            return self.spells[card]
        return frozenset(self.compositions.get(card, ()))

    def units_per_play(self, card: int) -> int | None:
        """Units one play deploys (3 for Skeletons); None for a spell, an unknown card or a
        card with forms (which form a play puts down is not known before it lands)."""
        comp = self.compositions.get(card)
        if comp is None or card in self.variant_costs:
            return None
        return sum(comp.values())

    def reported_card(self, card: int) -> int:
        """The card a play of `card` is reported as: a form's card (MergeMaiden_Normal ->
        MergeMaiden, the card the deck, the hand and our rows' memory name), else `card`."""
        card = int(card)
        return self.aliases.get(card, card)

    def play_cost(self, card: int, classes: Iterable[str] | str = ()) -> int | None:
        """The elixir one play of `card` cost when the classes it showed tell its form
        (variant_costs): the Spirit Empress as MergeMaiden_Normal 3, as MergeMaiden_Mounted
        6 (the engine charges the form's own cost, royalesim state.rs resolve_play); the
        dearer one when both show. None for a card every play of which costs the
        catalogue's price, or when no class shown names a form."""
        costs = self.variant_costs.get(int(card))
        if not costs:
            return None
        shown = [classes] if isinstance(classes, str) else list(classes)
        found = [costs[c] for c in shown if c in costs]
        return max(found) if found else None

    def explains(self, parent_cls: str, child_cls: str) -> bool:
        kids = self.children.get(parent_cls, ())
        return child_cls in kids or WILDCARD in kids

    def spell_class(self, card: int) -> str | None:
        """One class standing for spell `card` (StateTracks uses it for engine spells)."""
        s = self.spells.get(card)
        return None if not s else min(s)

    def card_guess(self, cls: str, prefer: Iterable[int] | None = None) -> int | None:
        """The card a lone track of `cls` most likely belongs to: a spell's card, else
        the lowest-id card whose units include it (cards in `prefer` first)."""
        cands = self.spell_cards(cls) or self.unit_cards(cls)
        if not cands:
            return None
        if prefer is not None:
            pref = set(prefer)
            mine = [c for c in cands if c in pref]
            if mine:
                return mine[0]
        return cands[0]

    # -- builders --------------------------------------------------------------
    @classmethod
    def from_names(cls, table: Mapping[str, Any], card_ids: Mapping[str, int], *,
                   class_of: Callable[[str], str | None] | Mapping[str, str | None] | None = None,
                   spell_classes: Mapping[str, Iterable[str]] | None = None, katacr: bool = True,
                   ignore: Iterable[str] = ()) -> "EventTables":
        """Tables from data/vision/spawn_children.json (RoyaleSim names).

        card_ids: card name -> catalogue id ({row[0]: i for i, row in enumerate(catalogue)});
        cards it lacks are skipped. class_of: unit name -> vision class (None drops the
        unit; a mapping keeps unlisted names). spell_classes: spell card name -> its
        vision classes; by default the card's name, its RoyaleSim projectile / area
        names and (katacr=True) its KataCR names. A spell class that is also a unit
        class is dropped from the spell. A spell card with a composition (Heal ->
        HealSpirit, a spell that deploys a troop) is a unit card, not a spell.

        VARIANTS ("variants": card -> {form card: elixir}; the Spirit Empress). When
        card_ids has the card, its forms' compositions become ITS composition (each class
        at the most any one form deploys: a play shows one form), every form card_ids
        has becomes an alias of it, and each class that only one form puts down gets that
        form's cost (variant_costs, play_cost). Without the card's id the forms stay
        cards of their own, as in a table without "variants".
        """
        cmap = _class_mapper(class_of)

        def mapped(comp: Any) -> dict[str, int]:
            cc: dict[str, int] = {}
            for unit, n in _counts(comp).items():
                c = cmap(unit)
                if c:
                    cc[c] = cc.get(c, 0) + int(n)
            return cc

        variants = {str(name): {str(f): int(cost) for f, cost in dict(forms).items()}
                    for name, forms in (table.get("variants") or {}).items()
                    if card_ids.get(name) is not None}
        form_of = {form: name for name, forms in variants.items() for form in forms}
        compositions = table.get("compositions", {})
        comps: dict[int, dict[str, int]] = {}
        for name, comp in compositions.items():
            cid = card_ids.get(name)
            if cid is None or name in form_of:
                continue
            cc = mapped(comp)
            if cc:
                comps[int(cid)] = cc
        aliases: dict[int, int] = {}
        variant_costs: dict[int, dict[str, int]] = {}
        for name, forms in sorted(variants.items()):
            cid = int(card_ids[name])
            merged = dict(comps.get(cid, {}))
            cost_of: dict[str, set[int]] = {}
            for form, cost in sorted(forms.items()):
                for c, n in mapped(compositions.get(form, {})).items():
                    merged[c] = max(merged.get(c, 0), n)
                    cost_of.setdefault(c, set()).add(cost)
            if not merged:
                continue
            comps[cid] = merged
            for form in forms:
                fid = card_ids.get(form)
                if fid is not None and int(fid) != cid:
                    aliases[int(fid)] = cid
            costs = {c: next(iter(s)) for c, s in cost_of.items() if len(s) == 1}
            if costs:
                variant_costs[cid] = costs
        unit_classes = {c for comp in comps.values() for c in comp}
        spells: dict[int, set[str]] = {}
        objects = table.get("spell_objects", {})
        for name, kind in table.get("kinds", {}).items():
            cid = card_ids.get(name)
            if kind != "spell" or cid is None or int(cid) in comps:
                continue
            if spell_classes is not None:
                classes = set(spell_classes.get(name, ()))
            else:
                classes = {name, *objects.get(name, ())}
                if katacr:
                    classes |= set(KATACR_SPELL_CLASSES.get(name, ()))
            classes -= unit_classes
            if classes:
                spells[int(cid)] = classes
        children: dict[str, set[str]] = {}
        for unit, kids in table.get("children", {}).items():
            p = cmap(unit)
            if not p:
                continue
            for k in kids:
                c = cmap(k)
                if c:
                    children.setdefault(p, set()).add(c)
        for name, kids in table.get("spell_children", {}).items():
            cid = card_ids.get(name)
            for sc in spells.get(cid, ()) if cid is not None else ():
                children.setdefault(sc, set()).update(c for c in map(cmap, kids) if c)
        for name in CLONE_CARDS:
            cid = card_ids.get(name)
            for sc in spells.get(cid, ()) if cid is not None else ():
                children.setdefault(sc, set()).add(WILDCARD)
        return cls(comps, spells, children, frozenset(ignore), aliases=aliases,
                   variant_costs=variant_costs)

    @classmethod
    def from_spawn_primary(cls, raw: Mapping[str, Any], *,
                           class_of: Callable[[str], str | None] | Mapping[str, str | None] | None = None,
                           base: "EventTables | None" = None,
                           unit_names: Sequence[str] | None = None) -> "EventTables":
        """Tables from E06's data/rs_rows/spawn_primary.json (tools/rs_spawn_table.py):
        "primary" {card id: unit type} and "all" {card id: [unit types a one-spawn reset
        put down]}. Unit types are indexes into unit_names (default: the file's
        "unit_types") or names; "all" may also be a {unit: count} dict or one unit.

        E06's "all" lists each type once, so it has no counts (Skeletons and Skeleton
        Army both read {Skeleton: 1}): pass `base` (for_engine does) for counts. Without
        `base`, a card's composition is its "all" units (the primary at least once).
        With `base`, base compositions win for the cards base knows, the primary is added
        where base lacks it, and cards base lacks come from "all". The non-primary units of
        "all" also become children of the primary (a Witch's skeletons spawned while E06's
        spawn settled), except, with `base`, units the card's base composition deploys with
        it: a Goblin Gang's spear goblins are not children of its goblins, or a later Spear
        Goblins play next to a living goblin would be explained away. Spells come from
        base only (spawn_primary has none). A card base holds as a form (base.aliases:
        MergeMaiden_Normal) counts as its card, so its units still name MergeMaiden.
        """
        cmap = _class_mapper(class_of)
        names = list(unit_names if unit_names is not None else raw.get("unit_types") or ())
        aliases = dict(base.aliases) if base is not None else {}

        def unit_class(u: Any) -> str | None:
            if isinstance(u, int) and not isinstance(u, bool) or (isinstance(u, str) and u.isdigit()):
                k = int(u)
                return cmap(names[k]) if 0 <= k < len(names) else None
            return cmap(str(u))

        allm = {_card_id(k): v for k, v in (raw.get("all") or {}).items()}
        comps: dict[int, dict[str, int]] = (
            {c: dict(comp) for c, comp in base.compositions.items()} if base is not None else {})
        children: dict[str, set[str]] = (
            {p: set(k) for p, k in base.children.items()} if base is not None else {})
        for key, unit in (raw.get("primary") or {}).items():
            own = _card_id(key)
            cid = aliases.get(own, own)          # a form's units are its card's
            pc = unit_class(unit)
            if not pc:
                continue
            listed = allm.get(own, ())
            if isinstance(listed, (str, int)):
                listed = [listed]
            pairs = listed.items() if isinstance(listed, Mapping) else ((u, 1) for u in listed)
            seen: Counter = Counter()
            for u, n in pairs:
                c = unit_class(u)
                if c:
                    seen[c] += int(n)
            if seen[pc] < 1:
                seen[pc] = 1
            co_deployed = comps.get(cid, {}) if base is not None else {}
            for c in seen:
                if c != pc and c not in co_deployed:
                    children.setdefault(pc, set()).add(c)
            if base is not None and cid in comps:
                comps[cid].setdefault(pc, seen[pc])
            elif base is None or cid not in base.spells:
                comps[cid] = dict(seen)
        spells = dict(base.spells) if base is not None else {}
        unit_classes = {c for comp in comps.values() for c in comp}
        spells = {c: s - unit_classes for c, s in spells.items() if c not in comps and s - unit_classes}
        ignore = base.ignore if base is not None else frozenset()
        costs = base.variant_costs if base is not None else {}
        return cls(comps, spells, children, ignore, aliases=aliases, variant_costs=costs)

    @classmethod
    def for_engine(cls, spawn_primary: Mapping[str, Any], *,
                   spawn_children: Mapping[str, Any] | None = None,
                   class_of: Callable[[str], str | None] | Mapping[str, str | None] | None = None,
                   ignore: Iterable[str] = ()) -> "EventTables":
        """The tables for one engine: E06's spawn_primary.json (ids, primary units) on top
        of from_names(spawn_children) (counts, children, spells), with card ids from the
        file's "card_names". spawn_children defaults to load_spawn_children()."""
        card_ids = {str(n): i for i, n in enumerate(spawn_primary.get("card_names") or ())}
        base = None
        if card_ids:
            base = cls.from_names(load_spawn_children() if spawn_children is None else spawn_children,
                                  card_ids, class_of=class_of, ignore=ignore)
        return cls.from_spawn_primary(spawn_primary, class_of=class_of, base=base)


def load_spawn_children(path: str | Path | None = None) -> dict:
    """data/vision/spawn_children.json (or `path`) as a dict for EventTables.from_names."""
    return json.loads(Path(path or SPAWN_CHILDREN).read_text(encoding="utf-8"))


def unit_class_map(classes_doc: Mapping[str, Any]) -> dict[str, str]:
    """RoyaleSim unit name -> vision class name, from data/vision/classes.json (E16):
    Goblin_Stab -> Goblin, Skeleton_EV1 -> Skeleton, DeliveryRecruit -> Recruit. Pass it
    as EventTables.from_names(class_of=...)."""
    names = {int(c["id"]): str(c["name"]) for c in classes_doc.get("classes", ())}
    return {str(u): names[int(v[0])] for u, v in classes_doc.get("units", {}).items()
            if int(v[0]) in names}


def cardless_classes(classes_doc: Mapping[str, Any]) -> frozenset[str]:
    """Classes of classes.json no card puts down (crown towers, tower troops): the
    `ignore` set for EventTables, so they are not counted as unknown tracks."""
    return frozenset(str(c["name"]) for c in classes_doc.get("classes", ())
                     if c.get("kind") == "tower" or not c.get("cards"))


# ---------------------------------------------------------------------------
# Configuration, taps and statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EventConfig:
    """Deriver parameters: times in battle ticks, distances in world millitiles."""
    play_window_ticks: int = PLAY_WINDOW     # same card within this = the same play
    merge_radius_mt: float = 5000.0          # ... and (units) this close to the play
    just_dead_ticks: int = 30                # a parent seen this recently still explains a child
    child_radius_mt: float = 4500.0          # ... born this close to it (Graveyard radius 4000)
    reid_ticks: int = 30                     # a lost track may come back under a new uid this late
    reid_radius_mt: float = 1200.0           # ... this close to where it was lost
    reid_speed_mt_per_tick: float = 50.0     # ... plus this much per tick it was gone
    reid_max_dead_hp: float = 0.2            # a track lost at or below this HP fraction died
    team_wait_ticks: int = 10                # wait this long for the tracker's team vote
    ambiguity_wait_ticks: int = 6            # a class several cards deploy: collect units this long
    excess_weight: float = 1.0               # composition fit: units the card cannot explain
    missing_weight: float = 0.25             # ... units of the card not seen (detector misses)
    unit_sight_lag_ticks: int = 0            # first sighting - execution, troops and buildings
    spell_sight_lag_ticks: int = 0           # ... spells
    sight_lag_by_card: Mapping[int, int] = field(default_factory=dict)
    tap_min_lag_ticks: int = -2              # control.MIN_LAG_TICKS (clock-read noise)
    tap_max_lag_ticks: int = 60              # control.MAX_LAG_TICKS (live_play.PENDING_TICKS)
    landing_radius_mt: float | None = 4000.0  # a unit landing this far from the aim is not that tap's
    aim_near_mt: float | None = 2000.0       # any tap aimed this close explains an own unit
    forget_ticks: int = 600                  # forget tracks and plays unseen this long
    prime: bool = True                       # tracks present at the first update are not plays
    unknown_plays: bool = False              # rule 6: an unexplained enemy track is an UNKNOWN play
    unknown_cost: int = UNKNOWN_PLAY_COST    # ... charged this much elixir (OppPlay.cost)

    def __post_init__(self):
        for name in ("play_window_ticks", "just_dead_ticks", "reid_ticks", "team_wait_ticks",
                     "ambiguity_wait_ticks", "unit_sight_lag_ticks", "spell_sight_lag_ticks",
                     "tap_max_lag_ticks", "forget_ticks"):
            if getattr(self, name) < 0:
                raise ValueError(f"EventConfig.{name} must be >= 0")
        try:
            cost = math.nan if isinstance(self.unknown_cost, (bool, str, bytes)) else float(self.unknown_cost)
        except (TypeError, ValueError):
            cost = math.nan
        if not (math.isfinite(cost) and cost == int(cost) and cost >= 0):
            raise ValueError(f"EventConfig.unknown_cost must be a whole elixir cost >= 0, "
                             f"got {self.unknown_cost!r}")
        for name in ("merge_radius_mt", "child_radius_mt", "reid_radius_mt", "reid_speed_mt_per_tick",
                     "excess_weight", "missing_weight"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"EventConfig.{name} must be >= 0")
        if self.tap_min_lag_ticks > self.tap_max_lag_ticks:
            raise ValueError("EventConfig needs tap_min_lag_ticks <= tap_max_lag_ticks")


@dataclass(slots=True)
class OwnTap:
    """A tap we sent and have not yet seen land."""
    card: int
    slot: int
    tick: int
    t: float | None = None
    aim_x_mt: float | None = None
    aim_y_mt: float | None = None

    @property
    def aim(self) -> tuple[float, float] | None:
        if self.aim_x_mt is None or self.aim_y_mt is None:
            return None
        return self.aim_x_mt, self.aim_y_mt


def _describe(values: Sequence[float]) -> dict[str, Any] | None:
    """n, mean, median, nearest-rank p90, min, max of `values` (None when empty)."""
    if not values:
        return None
    s = sorted(values)
    p90 = s[max(0, math.ceil(0.9 * len(s)) - 1)]
    return {"n": len(s), "mean": statistics.fmean(s), "median": statistics.median(s),
            "p90": p90, "min": s[0], "max": s[-1]}


@dataclass
class LandingStats:
    """Our taps against what the board showed. Kept across battles (a property of the rig).

    lag_ticks: first sighting - tap tick; lag_s: the same on the vision clock when
    both times were given; offsets_mt: (x - aim_x, y - aim_y) of unit landings.
    unmatched counts own plays no tap explains (a human's).
    """
    taps: int = 0
    landed: int = 0
    missed: int = 0
    unmatched: int = 0
    lag_ticks: list[int] = field(default_factory=list)
    lag_s: list[float] = field(default_factory=list)
    offsets_mt: list[tuple[float, float]] = field(default_factory=list)

    @property
    def miss_rate(self) -> float | None:
        n = self.landed + self.missed
        return None if n == 0 else self.missed / n

    def sight_lag_estimate(self, base_lag_ticks: int = 26) -> int | None:
        """Median landing lag minus the tap -> execution lag (live_play.LEAD, measured
        with the memory reader): how much later vision sees a play than it runs.
        None until a landing was measured."""
        if not self.lag_ticks:
            return None
        return max(0, int(round(statistics.median(self.lag_ticks))) - int(base_lag_ticks))

    def summary(self) -> dict[str, Any]:
        dx = [o[0] for o in self.offsets_mt]
        dy = [o[1] for o in self.offsets_mt]
        dist = [math.hypot(a, b) for a, b in self.offsets_mt]
        return {"taps": self.taps, "landed": self.landed, "missed": self.missed,
                "unmatched": self.unmatched, "miss_rate": self.miss_rate,
                "lag_ticks": _describe(self.lag_ticks), "lag_s": _describe(self.lag_s),
                "offset_mt": {"dx": _describe(dx), "dy": _describe(dy), "dist": _describe(dist)}}


@dataclass
class OppStats:
    """How the opponent's new tracks were explained (one battle)."""
    plays: int = 0
    spells: int = 0
    merged: int = 0
    children: int = 0
    reacquired: int = 0
    ambiguous: int = 0             # plays decided by composition fit
    fallback_team: int = 0         # tracks whose team the fallback decided (both teams)
    unknown: Counter = field(default_factory=Counter)    # class -> tracks no card explains
    unknown_plays: int = 0         # UNKNOWN plays reported (rule 6; not in `plays`)

    def summary(self) -> dict[str, Any]:
        return {"plays": self.plays, "spells": self.spells, "merged": self.merged,
                "children": self.children, "reacquired": self.reacquired,
                "ambiguous": self.ambiguous, "fallback_team": self.fallback_team,
                "unknown": dict(self.unknown), "unknown_plays": self.unknown_plays}


# ---------------------------------------------------------------------------
# Internal records
# ---------------------------------------------------------------------------
PENDING, DONE = "pending", "done"


@dataclass(slots=True, eq=False)
class _Play:
    team: int
    kind: str
    tick: int                         # first sighting
    x: float
    y: float
    last_tick: int
    uids: list[int]
    card: int | None = None           # None while an ambiguous group waits
    candidates: tuple[int, ...] = ()
    resolve_at: int | None = None
    counts: Counter = field(default_factory=Counter)
    record: Any = None                # the OppPlay / OwnLanding reported for it


@dataclass(slots=True, eq=False)
class _Rec:
    uid: int
    cls: str
    first_tick: int
    last_tick: int
    x: float
    y: float
    bx: float                         # birth position
    by: float
    first_t: float | None = None
    hp: float | None = None
    seen_team: int = UNKNOWN_TEAM     # the tracker's latest team
    team: int = UNKNOWN_TEAM          # the team it was classified under
    state: str = PENDING
    how: str = ""                     # play | merged | child | reacquired | landing | unknown |
    #                                   unknown_play | ignored | primed (EventDeriver.how_of)
    card: int | None = None
    play: _Play | None = None
    succeeded: bool = False           # a newer track was taken as this one re-acquired


# ---------------------------------------------------------------------------
# The deriver
# ---------------------------------------------------------------------------
class EventDeriver:
    """Opponent plays and own landings from one battle's tracks (see the module doc).

    tables: EventTables. own_deck: our card ids (preferred when one of our classes
    could be several cards). own_team: the RoyaleSim team we play (1 live).
    Call reset() at the start of every battle and update() once per frame.
    """

    def __init__(self, tables: EventTables, config: EventConfig | None = None, *,
                 own_deck: Iterable[int] | None = None, own_team: int = OWN_TEAM):
        if own_team not in (0, 1):
            raise ValueError(f"own_team must be 0 or 1, got {own_team}")
        self.tables = tables
        self.config = config or EventConfig()
        self.own_deck = None if own_deck is None else frozenset(int(c) for c in own_deck)
        self.own_team = int(own_team)
        self.enemy_team = 1 - self.own_team
        self.landing_stats = LandingStats()
        self.reset()

    def reset(self, *, clear_landing_stats: bool = False) -> None:
        """Forget the battle: tracks, plays, pending taps and opponent statistics.
        Landing statistics are kept unless clear_landing_stats."""
        self._recs: dict[int, _Rec] = {}
        self._plays: list[_Play] = []
        self._taps: list[OwnTap] = []
        self._opp_history: list[OppPlay] = []
        self._own_history: list[OwnLanding] = []
        self._tick: int | None = None
        self._primed = not self.config.prime
        self._out_opp: list[OppPlay] = []
        self._out_own: list[OwnLanding] = []
        self.backwards_ticks = 0
        self.opp_stats = OppStats()
        if clear_landing_stats:
            self.landing_stats = LandingStats()

    # -- inputs ---------------------------------------------------------------
    def note_own_tap(self, card: int, tick: int, *, slot: int, aim_x_mt: float | None = None,
                     aim_y_mt: float | None = None, t: float | None = None,
                     tracker: Any = None) -> OwnTap:
        """Record a play we sent: card id, hand slot (0..3), the tick the tap was sent,
        the world point aimed at and the vision-clock time (types.now()). With
        `tracker`, also tells the tracker (Tracker.note_own_tap) which classes to
        expect there, so the same tap drives the team vote."""
        if int(slot) < 0:
            raise ValueError(f"a tap needs its hand slot (>= 0), got {slot}")
        tap = OwnTap(int(card), int(slot), int(tick), t,
                     None if aim_x_mt is None else float(aim_x_mt),
                     None if aim_y_mt is None else float(aim_y_mt))
        self._taps.append(tap)
        self.landing_stats.taps += 1
        if tracker is not None and tap.aim is not None:
            classes = self.tables.classes_of(tap.card)
            if classes:
                tracker.note_own_tap(classes, tap.aim[0], tap.aim[1], tap.tick,
                                     max_units=self.tables.units_per_play(tap.card))
        return tap

    def update(self, tracks: Iterable[Track], tick: int, *, t: float | None = None,
               tracker: Any = None) -> Events:
        """Feed one frame's tracks (Tracker.update's output) at battle tick `tick`.
        t: the frame's capture time (types.now clock), for latencies in seconds.
        tracker: the Tracker, for birth positions of new tracks. Returns the events
        derived on this frame (not cumulative)."""
        tick = int(tick)
        if self._tick is not None and tick < self._tick:
            self.backwards_ticks += 1
            tick = self._tick
        self._tick = tick
        self._out_opp, self._out_own = [], []

        for tr in tracks:
            rec = self._recs.get(tr.uid)
            if rec is None:
                rec = self._new_rec(tr, tick, t, tracker)
                self._recs[tr.uid] = rec
                if not self._primed:
                    rec.state, rec.how, rec.team = DONE, "primed", tr.team
                    rec.card = self.card_for(tr)
                    continue
            else:
                rec.last_tick = max(rec.last_tick, int(tr.last_seen_tick))
                rec.x, rec.y = float(tr.x_mt), float(tr.y_mt)
                if tr.hp_frac is not None:
                    rec.hp = float(tr.hp_frac)
                if rec.state == PENDING:
                    rec.cls = tr.cls            # the tracker may still fix the label
                if rec.play is not None:
                    rec.play.last_tick = max(rec.play.last_tick, rec.last_tick)
            rec.seen_team = tr.team
        self._primed = True

        cfg = self.config
        pending = sorted((r for r in self._recs.values() if r.state == PENDING),
                         key=lambda r: (r.first_tick, r.uid))
        for rec in pending:
            team = rec.seen_team if rec.seen_team in (0, 1) else UNKNOWN_TEAM
            if team == UNKNOWN_TEAM:
                if tick - rec.first_tick < cfg.team_wait_ticks:
                    continue
                team = self._fallback_team(rec)
                self.opp_stats.fallback_team += 1
            self._classify(rec, team)
        for play in self._plays:
            if play.card is None and play.resolve_at is not None and tick >= play.resolve_at:
                self._resolve(play)
        self._expire_taps(tick)
        self._forget(tick)
        return Events([_copy_opp(p) for p in self._out_opp], [replace(x) for x in self._out_own])

    # -- outputs --------------------------------------------------------------
    def card_of(self, uid: int) -> int | None:
        """The card that put track `uid` on the board, once classified (None for the tracks of
        an UNKNOWN play, rule 6: the deriver does not know their card)."""
        rec = self._recs.get(uid)
        return None if rec is None else rec.card

    def how_of(self, uid: int) -> str | None:
        """Which rule explained track `uid` (module doc, THE RULES): "reacquired" (1), "merged"
        (2, or a further unit of an UNKNOWN play), "child" (3), "landing" (4, one of our taps),
        "play" (5: a new play's first track, ours without a tap too), "unknown_play" (6: the
        first track of an UNKNOWN play), "unknown" (no rule; an enemy one is counted in
        OppStats.unknown), "ignored" (EventTables.ignore), "primed" (on the board at the first
        update); "pending" while it waits for its team; None for a uid this battle has not
        shown, or one forgotten after ``forget_ticks``."""
        rec = self._recs.get(int(uid))
        if rec is None:
            return None
        return rec.how if rec.state == DONE else HOW_PENDING

    def card_for(self, track: Track) -> int | None:
        """card_of(track.uid), or the tables' best guess for its class while it is
        pending (our deck first for our own tracks)."""
        rec = self._recs.get(track.uid)
        if rec is not None and rec.card is not None:
            return rec.card
        prefer = self.own_deck if track.team == self.own_team else None
        return self.tables.card_guess(track.cls, prefer)

    def track_cards(self, tracks: Iterable[Track]) -> dict[int, int]:
        """uid -> creating card for the tracks whose card is known or guessable."""
        out = {}
        for tr in tracks:
            card = self.card_for(tr)
            if card is not None:
                out[tr.uid] = card
        return out

    def opp_plays(self) -> list[OppPlay]:
        """Every opponent play reported this battle (copies), oldest first."""
        return [_copy_opp(p) for p in self._opp_history]

    def own_landings(self) -> list[OwnLanding]:
        """Every own landing, unmatched own play and missed tap reported this battle."""
        return [replace(x) for x in self._own_history]

    @property
    def pending_taps(self) -> list[OwnTap]:
        return [replace(tp) for tp in self._taps]

    def exec_tick(self, card: int, kind: str, tick: int) -> int:
        """Estimated execution tick of a play of `card` first seen at `tick`."""
        cfg = self.config
        lag = cfg.sight_lag_by_card.get(card)
        if lag is None:
            lag = cfg.spell_sight_lag_ticks if kind == KIND_SPELL else cfg.unit_sight_lag_ticks
        return max(0, int(tick) - int(lag))

    def stats(self) -> dict[str, Any]:
        return {"opponent": self.opp_stats.summary(), "own": self.landing_stats.summary(),
                "backwards_ticks": self.backwards_ticks}

    # -- classification -------------------------------------------------------
    def _new_rec(self, tr: Track, tick: int, t: float | None, tracker: Any) -> _Rec:
        first = int(tr.first_seen_tick)
        bx, by = float(tr.x_mt), float(tr.y_mt)
        state = tracker.state(tr.uid) if tracker is not None else None
        if state is not None:
            bx, by = float(state.birth_x_mt), float(state.birth_y_mt)
        first_t = None if t is None else float(t) - (tick - first) * TICK_S
        return _Rec(uid=tr.uid, cls=tr.cls, first_tick=first, last_tick=int(tr.last_seen_tick),
                    x=float(tr.x_mt), y=float(tr.y_mt), bx=bx, by=by, first_t=first_t,
                    hp=None if tr.hp_frac is None else float(tr.hp_frac))

    def _fallback_team(self, rec: _Rec) -> int:
        """Team of a track the tracker could not decide: ours if a pending tap explains
        it; a spell is the enemy's; a unit belongs to the half it was born in."""
        if self._match_tap(rec, self.tables.is_spell(rec.cls)) is not None:
            return self.own_team
        if self.tables.is_spell(rec.cls):
            return self.enemy_team
        own_high = self.own_team == OWN_TEAM          # team 1 defends high y
        in_own_half = rec.by > MID_Y_MT if own_high else rec.by < MID_Y_MT
        return self.own_team if in_own_half else self.enemy_team

    def _classify(self, rec: _Rec, team: int) -> None:
        rec.team, rec.state = team, DONE
        if rec.cls in self.tables.ignore:
            rec.how = "ignored"
            return
        spell = self.tables.is_spell(rec.cls)
        enemy = team == self.enemy_team
        old = self._reacquired(rec, team)
        if old is not None:
            old.succeeded = True
            self._attach(rec, old.play, old.card, "reacquired")
            if enemy:
                self.opp_stats.reacquired += 1
            return
        play = self._mergeable(rec, team, spell)
        if play is not None:
            self._attach(rec, play, play.card, "merged")
            if play.card is None:
                play.counts[rec.cls] += 1
                play.tick = min(play.tick, rec.first_tick)
            if enemy:
                self.opp_stats.merged += 1
            return
        parent = self._parent(rec, team)
        if parent is not None:
            self._attach(rec, parent.play, parent.card, "child")
            if enemy:
                self.opp_stats.children += 1
            return
        if not enemy:
            tap = self._match_tap(rec, spell)
            if tap is not None:
                self._taps.remove(tap)
                self._land(rec, tap, spell)
                return
        cands = self.tables.spell_cards(rec.cls) if spell else self.tables.unit_cards(rec.cls)
        if not enemy and self.own_deck is not None:
            cands = tuple(c for c in cands if c in self.own_deck) or cands
        if not cands:
            rec.how = "unknown"
            if enemy:
                self.opp_stats.unknown[rec.cls] += 1
                if self.config.unknown_plays:
                    self._unknown_play(rec, team)
            return
        play = _Play(team, KIND_SPELL if spell else KIND_UNIT, rec.first_tick, rec.bx, rec.by,
                     rec.last_tick, [rec.uid], candidates=cands, counts=Counter({rec.cls: 1}))
        self._plays.append(play)
        rec.play, rec.how = play, "play"
        if spell or len(cands) == 1:
            self._resolve(play)
        else:
            play.resolve_at = rec.first_tick + self.config.ambiguity_wait_ticks

    def _unknown_play(self, rec: _Rec, team: int) -> None:
        """Rule 6 (EventConfig.unknown_plays): enemy track `rec`, which no card explains, joins
        the newest UNKNOWN play of `team` first seen less than play_window_ticks from it and
        within merge_radius_mt of its birth point, or starts one, reported at once as
        OppPlay(card=UNKNOWN_CARD, cost=unknown_cost). card_of stays None for its tracks."""
        cfg = self.config
        for play in reversed(self._plays):
            if (play.team == team and play.card == UNKNOWN_CARD
                    and abs(rec.first_tick - play.tick) < cfg.play_window_ticks
                    and _dist(play.x, play.y, rec.bx, rec.by) <= cfg.merge_radius_mt):
                self._attach(rec, play, None, "merged")
                play.counts[rec.cls] += 1
                self.opp_stats.merged += 1
                return
        play = _Play(team, KIND_UNIT, rec.first_tick, rec.bx, rec.by, rec.last_tick, [rec.uid],
                     card=UNKNOWN_CARD, candidates=(UNKNOWN_CARD,), counts=Counter({rec.cls: 1}))
        self._plays.append(play)
        rec.play, rec.how = play, "unknown_play"
        opp = OppPlay(card=UNKNOWN_CARD, tick=play.tick,
                      exec_tick=self.exec_tick(UNKNOWN_CARD, KIND_UNIT, play.tick),
                      x_mt=play.x, y_mt=play.y, kind=KIND_UNIT, uids=list(play.uids),
                      cost=int(cfg.unknown_cost))
        play.record = opp
        self.opp_stats.unknown_plays += 1
        self._out_opp.append(opp)
        self._opp_history.append(opp)

    def _attach(self, rec: _Rec, play: _Play | None, card: int | None, how: str) -> None:
        rec.how, rec.play, rec.card = how, play, card
        if play is None:
            return
        play.uids.append(rec.uid)
        play.last_tick = max(play.last_tick, rec.last_tick)
        if isinstance(play.record, OppPlay):
            play.record.uids.append(rec.uid)

    def _reacquired(self, rec: _Rec, team: int) -> _Rec | None:
        cfg = self.config
        best, best_d = None, math.inf
        for r in self._recs.values():
            if (r is rec or r.state != DONE or r.team != team or r.cls != rec.cls or r.succeeded
                    or r.how in ("unknown", "ignored")):
                continue
            gap = rec.first_tick - r.last_tick
            if not 0 < gap <= cfg.reid_ticks:
                continue
            if r.hp is not None and r.hp <= cfg.reid_max_dead_hp:
                continue
            d = _dist(r.x, r.y, rec.bx, rec.by)
            if d <= cfg.reid_radius_mt + cfg.reid_speed_mt_per_tick * gap and d < best_d:
                best, best_d = r, d
        return best

    def _mergeable(self, rec: _Rec, team: int, spell: bool) -> _Play | None:
        cfg = self.config
        for play in reversed(self._plays):
            if play.team != team or (play.kind == KIND_SPELL) != spell:
                continue
            if play.card is None:
                classes = set().union(*(self.tables.classes_of(c) for c in play.candidates))
            else:
                classes = self.tables.classes_of(play.card)
            if rec.cls not in classes:
                continue
            if spell:
                ok = (rec.first_tick - play.last_tick <= cfg.play_window_ticks
                      and rec.first_tick - play.tick > -cfg.play_window_ticks)
            else:
                ok = (abs(rec.first_tick - play.tick) < cfg.play_window_ticks
                      and _dist(play.x, play.y, rec.bx, rec.by) <= cfg.merge_radius_mt)
            if ok:
                return play
        return None

    def _parent(self, rec: _Rec, team: int) -> _Rec | None:
        cfg = self.config
        best, best_d = None, math.inf
        for r in self._recs.values():
            if r is rec or r.state != DONE or r.team != team:
                continue
            if r.first_tick > rec.first_tick or rec.first_tick - r.last_tick > cfg.just_dead_ticks:
                continue
            if not self.tables.explains(r.cls, rec.cls):
                continue
            d = _dist(r.x, r.y, rec.bx, rec.by)
            if d <= cfg.child_radius_mt and d < best_d:
                best, best_d = r, d
        return best

    def _match_tap(self, rec: _Rec, spell: bool) -> OwnTap | None:
        cfg = self.config
        in_window = [tp for tp in self._taps
                     if cfg.tap_min_lag_ticks <= rec.first_tick - tp.tick <= cfg.tap_max_lag_ticks]
        for tp in in_window:                                  # oldest first
            if rec.cls not in self.tables.classes_of(tp.card):
                continue
            aim = tp.aim
            if (not spell and cfg.landing_radius_mt is not None and aim is not None
                    and _dist(aim[0], aim[1], rec.bx, rec.by) > cfg.landing_radius_mt):
                continue
            return tp
        if spell or cfg.aim_near_mt is None:
            return None
        best, best_d = None, cfg.aim_near_mt
        for tp in in_window:                                  # misread card: nearest aim
            aim = tp.aim
            if aim is None:
                continue
            d = _dist(aim[0], aim[1], rec.bx, rec.by)
            if d <= best_d:
                best, best_d = tp, d
        return best

    def _land(self, rec: _Rec, tap: OwnTap, spell: bool) -> None:
        kind = KIND_SPELL if spell else KIND_UNIT
        play = _Play(rec.team, kind, rec.first_tick, rec.bx, rec.by, rec.last_tick, [rec.uid],
                     card=tap.card, candidates=(tap.card,), counts=Counter({rec.cls: 1}))
        self._plays.append(play)
        rec.play, rec.card, rec.how = play, tap.card, "landing"
        landing = OwnLanding(card=tap.card, slot=tap.slot, tap_tick=tap.tick, land_tick=rec.first_tick,
                             tap_t=tap.t, aim_x_mt=tap.aim_x_mt, aim_y_mt=tap.aim_y_mt,
                             x_mt=rec.bx, y_mt=rec.by, uid=rec.uid)
        play.record = landing
        st = self.landing_stats
        st.landed += 1
        st.lag_ticks.append(rec.first_tick - tap.tick)
        if tap.t is not None and rec.first_t is not None:
            st.lag_s.append(rec.first_t - tap.t)
        if not spell and tap.aim is not None:
            st.offsets_mt.append((rec.bx - tap.aim[0], rec.by - tap.aim[1]))
        self._emit_own(landing)

    def _score(self, card: int, counts: Counter) -> tuple[float, int, int]:
        comp = self.tables.compositions.get(card, {})
        matched = sum(min(n, comp.get(c, 0)) for c, n in counts.items())
        excess = sum(max(0, n - comp.get(c, 0)) for c, n in counts.items())
        missing = sum(max(0, m - counts.get(c, 0)) for c, m in comp.items())
        cfg = self.config
        # best fit, then the smaller card, then the lower id
        return matched - cfg.excess_weight * excess - cfg.missing_weight * missing, -sum(comp.values()), -card

    def _resolve(self, play: _Play) -> None:
        if len(play.candidates) == 1 or play.kind == KIND_SPELL:
            play.card = play.candidates[0]
        else:
            play.card = max(play.candidates, key=lambda c: self._score(c, play.counts))
            if play.team == self.enemy_team:
                self.opp_stats.ambiguous += 1
        play.resolve_at = None
        for uid in play.uids:
            rec = self._recs.get(uid)
            if rec is not None and rec.play is play:
                rec.card = play.card
        if play.team == self.enemy_team:
            # cost: a form's own cost (EventTables.variant_costs), else None (the catalogue's)
            opp = OppPlay(card=play.card, tick=play.tick,
                          exec_tick=self.exec_tick(play.card, play.kind, play.tick),
                          x_mt=play.x, y_mt=play.y, kind=play.kind, uids=list(play.uids),
                          cost=self.tables.play_cost(play.card, play.counts))
            play.record = opp
            self.opp_stats.plays += 1
            if play.kind == KIND_SPELL:
                self.opp_stats.spells += 1
            self._out_opp.append(opp)
            self._opp_history.append(opp)
        else:
            landing = OwnLanding(card=play.card, slot=-1, tap_tick=play.tick, land_tick=play.tick,
                                 x_mt=play.x, y_mt=play.y, uid=play.uids[0])
            play.record = landing
            self.landing_stats.unmatched += 1
            self._emit_own(landing)

    def _emit_own(self, landing: OwnLanding) -> None:
        self._out_own.append(landing)
        self._own_history.append(landing)

    def _expire_taps(self, tick: int) -> None:
        horizon = self.config.tap_max_lag_ticks + self.config.team_wait_ticks
        keep = []
        for tp in self._taps:
            if tick - tp.tick > horizon:
                miss = OwnLanding(card=tp.card, slot=tp.slot, tap_tick=tp.tick, land_tick=None,
                                  tap_t=tp.t, aim_x_mt=tp.aim_x_mt, aim_y_mt=tp.aim_y_mt)
                self.landing_stats.missed += 1
                self._emit_own(miss)
            else:
                keep.append(tp)
        self._taps = keep

    def _forget(self, tick: int) -> None:
        horizon = self.config.forget_ticks
        stale = [uid for uid, r in self._recs.items() if r.state == DONE and tick - r.last_tick > horizon]
        for uid in stale:
            del self._recs[uid]
        self._plays = [p for p in self._plays if p.card is None or tick - p.last_tick <= horizon]


def _copy_opp(p: OppPlay) -> OppPlay:
    return OppPlay(p.card, p.tick, p.exec_tick, p.x_mt, p.y_mt, p.kind, list(p.uids), p.cost)


# ---------------------------------------------------------------------------
# Public memory
# ---------------------------------------------------------------------------
def record_plays(memory: Any, plays: Iterable[OppPlay], *, team: int = ENEMY_TEAM,
                 token_offset: int = TOKEN_OFFSET) -> int:
    """Record opponent plays into a PublicMemory-like object: memory.record(team,
    card + token_offset, exec_tick) (the first sighting when exec_tick is None). An
    UNKNOWN play (card -1) is row token 1, rs_memory.UNKNOWN, the token the model reads
    for a play perception could not name (legacy and v2 rows alike: opp_seen / opp_last).
    `team` is the opponent's RoyaleSim team (0 live). Returns how many were recorded."""
    n = 0
    for p in plays:
        memory.record(int(team), int(p.card) + int(token_offset),
                      int(p.exec_tick if p.exec_tick is not None else p.tick))
        n += 1
    return n


# ---------------------------------------------------------------------------
# Engine states -> tracks (offline runs of the deriver)
# ---------------------------------------------------------------------------
def _field_index(fields: Mapping[str, int] | Sequence[str]) -> dict[str, int]:
    if isinstance(fields, Mapping):
        return {str(k): int(v) for k, v in fields.items()}
    return {str(name): i for i, name in enumerate(fields)}


class StateTracks:
    """RoyaleSim state_json dicts -> Tracks, as a perfect perception would report them.

    fields: royalesim.ENTITY_FIELDS (or name -> index); unit_names: unit_types_json();
    sub: SUBTILE_PER_MILLITILE. class_of maps a unit name to its vision class (default
    strip_form; None drops the unit). With spell_fields (royalesim.SPELL_FIELDS) and
    spell_class (card id -> class, e.g. EventTables.spell_class), the state's spells are
    tracks too, first seen at the tick they were cast (tick - ticks_flown) and placed at
    their aim. Crown towers and dead entities are left out. Call reset() per battle.

    royalesim's SpellState rows carry no uid (team, card_id, motion, x, y, aim_x, aim_y,
    delay_ticks, travelled, length, hits, ticks_flown), so without a "uid" column a spell
    is identified by (team, card_id, aim_x, aim_y), which stays fixed for its whole life
    (aim is the landing point, a roll's end point or an area's centre). A key that leaves
    the state frees it: the next cast at the same point gets a new uid. ticks_flown -1
    (not reported by old engines) counts as 0.
    """

    def __init__(self, fields: Mapping[str, int] | Sequence[str], unit_names: Sequence[str], *,
                 sub: int = 18,
                 class_of: Callable[[str], str | None] | Mapping[str, str | None] | None = strip_form,
                 spell_fields: Mapping[str, int] | Sequence[str] | None = None,
                 spell_class: Callable[[int], str | None] | None = None):
        self.F = _field_index(fields)
        self.SF = None if spell_fields is None else _field_index(spell_fields)
        self.unit_names = list(unit_names)
        self.sub = float(sub)
        self.class_of = _class_mapper(class_of)
        self.spell_class = spell_class
        self.reset()

    def reset(self) -> None:
        self._first: dict[int, int] = {}
        self._spell_uid: dict[tuple[int, int, int, int], int] = {}
        self._spells_made = 0

    def _spell_uid_of(self, sp: Sequence[Any], live: set) -> int:
        SF = self.SF
        if "uid" in SF:
            return SPELL_UID_OFFSET + int(sp[SF["uid"]])
        key = (int(sp[SF["team"]]), int(sp[SF["card_id"]]), int(sp[SF["aim_x"]]), int(sp[SF["aim_y"]]))
        live.add(key)
        uid = self._spell_uid.get(key)
        if uid is None:
            self._spells_made += 1
            uid = self._spell_uid[key] = SPELL_UID_OFFSET + self._spells_made
        return uid

    def tracks(self, state: Mapping[str, Any]) -> list[Track]:
        F, sub = self.F, self.sub
        tick = int(state["tick"])
        out = []
        for e in state.get("entities", ()):
            if e[F["tower_slot"]] >= 0 or e[F["hp"]] <= 0:
                continue
            ut = int(e[F["unit_type"]])
            name = self.unit_names[ut] if 0 <= ut < len(self.unit_names) else None
            cls = self.class_of(name) if name else None
            if not cls:
                continue
            uid = int(e[F["uid"]])
            first = self._first.setdefault(uid, tick)
            hp, mx = e[F["hp"]], e[F["max_hp"]]
            out.append(Track(uid=uid, cls=cls, team=int(e[F["team"]]), conf=1.0,
                             x_mt=e[F["x"]] / sub, y_mt=e[F["y"]] / sub,
                             hp_frac=None if not mx else max(0.0, min(1.0, hp / mx)),
                             first_seen_tick=first, last_seen_tick=tick))
        if self.SF is not None and self.spell_class is not None:
            SF = self.SF
            live: set = set()
            for sp in state.get("spells", ()):
                cls = self.spell_class(int(sp[SF["card_id"]]))
                if not cls:
                    continue
                uid = self._spell_uid_of(sp, live)
                flown = max(0, int(sp[SF["ticks_flown"]] or 0)) if "ticks_flown" in SF else 0
                first = self._first.setdefault(uid, max(0, tick - flown))
                out.append(Track(uid=uid, cls=cls, team=int(sp[SF["team"]]), conf=1.0,
                                 x_mt=sp[SF["aim_x"]] / sub, y_mt=sp[SF["aim_y"]] / sub,
                                 first_seen_tick=first, last_seen_tick=tick))
            for key in [k for k in self._spell_uid if k not in live]:
                del self._spell_uid[key]          # that cast is over
        return out
