"""Ability presses from IL_Replay, as RoyaleSim commands (E10; tokens-02, rsim-02).

IL_Replay records each press of a hero's or a champion's ability button as an event of kind
"activate_ability". The tools re-drove only "play_card" events, so a recorded hero Musketeer
(3 elixir) or hero Ice Golem (2 elixir) press never happened in the engine: the simulated
player ended up richer than the real one, nothing was refused, and the rows stayed trusted.
tools/rs_rows.py and tools/royalesim_redrive.py replay them under `--presses on` (default
`off`: no press is sent and every output is what it was before E10).

WHICH CARD. An event names its card as sim/replay_corpus.read_part reads it: "card_key" when it
is set, else "ability_source_candidates" (event_keys). Each key loses its form suffix ("-ev1",
"-hero"; split_key, the same rule as royalesim_redrive.base_key) and maps to a catalogue id
through the card map (tools/rs_cards.py). The candidates that are one of the side's buttons
decide: one button, it is pressed; none ("unknown") or several ("ambiguous"), the press is
skipped, and the caller cuts trust at its tick (button_of).

WHICH BUTTON. royalegym protocol.py:219-223: command slot HAND_SIZE + k presses button k; x and
y are not read. A side's buttons are its hero entries (form 2) in deck order, then its
champions in deck order (side_buttons). The deck order is the order the decks are given to
reset(), which for the tools is the DEALT order (royalesim_redrive.deal_order), not the
recorded one. Champions come from the catalogue's "champion" column where
royalesim.CATALOGUE_FIELDS names one, else from CHAMPIONS_BEFORE_THE_COLUMN (royalegym
rust_engine.py, the same rule; champion_ids). When the engine's own `abilities` rows name their
cards (players[t]["abilities"][k][3] >= 0), their order is the one used (engine_order): the
engine is what presses the slot, and a disagreement is counted.

THE COMMAND. (team, HAND_SIZE + k, 0, 0) at the event's tick, in the same step as that tick's
plays (command). HAND_SIZE is royalesim.HAND_SIZE (fallback 4) and the number of buttons B is
royalesim.ABILITY_BUTTONS (fallback 2; 3 from RoyaleSim 2245f9f). A press the engine refuses
(NO_HERO: no living hero or champion behind it; ABILITY_NOT_READY; ABILITY_SPENT; or any other
reason, NOT_ENOUGH_ELIXIR included) is counted by reason name (PressLog) and cuts trust.

FEATURES. ability_features turns players[t]["abilities"] (royalegym AbilityRow: available,
spent, cost, card_id, cooldown_ticks; read by position, extra columns ignored) into B button
cards (catalogue id + 2; 0 no button, 1 a button whose card nobody names) and B x ABILITY_FEATS
floats: reported (the engine has a row for the button), available, spent, cost / 10 and
cooldown_ticks / COOLDOWN_TICKS (clipped to [0, 1]; 0 when not reported). An engine with no
rows gives zeros, with the cards from the deck when the caller passes its button list.

Pure Python: no royalesim, numpy or torch import. tools/rs_rows.py and
tools/royalesim_redrive.py read the engine's constants through button_count / hand_size.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

#: --presses: PRESS_OFF (the default) replays no press, as before E10; PRESS_ON replays them.
PRESS_OFF, PRESS_ON = "off", "on"
PRESS_MODES = (PRESS_OFF, PRESS_ON)
#: The IL_Replay event kind of a press (sim/replay_corpus.Play.kind).
PRESS_KIND = "activate_ability"
#: royalegym protocol.py HAND_SIZE / ABILITY_BUTTONS, for an engine that does not export them.
FALLBACK_HAND_SIZE = 4
FALLBACK_BUTTONS = 2
EMPTY_CARD = -1
#: MatchSetup.forms: 0 the card itself, 1 its evolution, 2 its hero (an ability button).
FORM_HERO = 2
#: IL_Replay form suffixes (royalesim_redrive.base_key): key suffix -> form.
FORM_SUFFIXES = (("-ev1", 1), ("-hero", 2))
#: royalegym rust_engine.CHAMPIONS_BEFORE_THE_COLUMN: the champions of an engine whose
#: catalogue has no "champion" column.
CHAMPIONS_BEFORE_THE_COLUMN = frozenset({
    "MightyMiner", "SkeletonKing", "ArcherQueen", "GoldenKnight", "Monk", "LittlePrince",
    "Goblinstein", "BossBandit",
})
#: button_of's answers: the press resolved to one button, or why it is skipped.
PRESS_OK, SKIP_UNKNOWN, SKIP_AMBIGUOUS = "ok", "unknown", "ambiguous"
#: The deploy reasons only a press gets (royalegym protocol.DeployStatus 14-16).
PRESS_REASONS = ("NO_HERO", "ABILITY_NOT_READY", "ABILITY_SPENT")
#: ability_features: the floats per button, in column order.
ABILITY_FEATS = ("reported", "available", "spent", "cost", "cooldown")
#: cooldown_ticks / COOLDOWN_TICKS, clipped to 1: 20 s.
COOLDOWN_TICKS = 400
#: What ability_features divides a button's elixir cost by.
COST_SCALE = 10


def press_mode(value) -> str:
    """--presses as the tools take it: one of PRESS_MODES. ValueError otherwise."""
    if not isinstance(value, str) or value not in PRESS_MODES:
        raise ValueError(f"--presses: one of {', '.join(PRESS_MODES)}, got {value!r}")
    return value


def describe_presses(mode) -> str:
    """One line for the console."""
    if press_mode(mode) == PRESS_OFF:
        return "off (activate_ability events are not replayed, as before)"
    return ("on (activate_ability events replayed as (team, HAND_SIZE + button, 0, 0) at their "
            "tick; a refused, unknown or ambiguous press cuts trust)")


def _engine_int(royalesim, name: str, fallback: int) -> int:
    value = getattr(royalesim, name, None)
    if value is None or isinstance(value, bool):
        return fallback
    try:
        value = int(value)
    except (TypeError, ValueError):
        return fallback
    return value if value >= 0 else fallback


def button_count(royalesim) -> int:
    """B, the ability buttons per side: royalesim.ABILITY_BUTTONS, else FALLBACK_BUTTONS."""
    return _engine_int(royalesim, "ABILITY_BUTTONS", FALLBACK_BUTTONS)


def hand_size(royalesim) -> int:
    """royalesim.HAND_SIZE, else FALLBACK_HAND_SIZE: a press's command slot is this + button."""
    return _engine_int(royalesim, "HAND_SIZE", FALLBACK_HAND_SIZE)


def split_key(key: str) -> tuple[str, int]:
    """An IL_Replay card key -> (base key, form), as royalesim_redrive.base_key."""
    for suffix, form in FORM_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], form
    return key, 0


def champion_ids(cat: Sequence[Sequence], fields: Sequence[str] | None = None) -> frozenset:
    """Catalogue ids of the champions in Battle.catalogue_json() rows `cat`: by the "champion"
    column where `fields` (royalesim.CATALOGUE_FIELDS) names one and the row reaches it, else
    by name (CHAMPIONS_BEFORE_THE_COLUMN), as royalegym's RustEngine reads them."""
    fields = list(fields or ())
    at = fields.index("champion") if "champion" in fields else None
    out = set()
    for cid, row in enumerate(cat):
        if at is not None and len(row) > at:
            if row[at]:
                out.add(cid)
        elif row and row[0] in CHAMPIONS_BEFORE_THE_COLUMN:
            out.add(cid)
    return frozenset(out)


def side_buttons(deck: Sequence[int], forms: Sequence[int] | None,
                 champions: Iterable[int]) -> list[int]:
    """The catalogue id behind each of a side's buttons, in button order: its hero entries
    (form 2) in `deck` order, then its champions in `deck` order. `deck` and `forms` are the
    lists given to reset() (the dealt order); `forms` None or empty: every card as itself."""
    forms = list(forms or ()) + [0] * (len(deck) - len(forms or ()))
    champions = frozenset(champions)
    heroes = [int(c) for c, f in zip(deck, forms) if int(f) == FORM_HERO]
    champs = [int(c) for c, f in zip(deck, forms) if int(f) != FORM_HERO and c in champions]
    return heroes + champs


def engine_order(player: Mapping[str, Any], computed: Sequence[int]) -> tuple[list[int], bool]:
    """(the button cards to use, whether they differ from `computed`). The engine's own order
    when every one of its `abilities` rows names its card (column 3 >= 0), else `computed`."""
    rows = list(player.get("abilities") or ())
    cards = [int(r[3]) if len(r) > 3 else EMPTY_CARD for r in rows]
    if not cards or any(c < 0 for c in cards):
        return list(computed), False
    return cards, cards != [int(c) for c in computed]


def event_keys(event: Mapping[str, Any]) -> tuple[str, ...]:
    """The card keys a press event names: its "card_key" when set, else its distinct
    "ability_source_candidates" in their order (sim/replay_corpus.read_part's rule, all of
    the candidates rather than the first)."""
    key = event.get("card_key")
    if isinstance(key, str) and key:
        return (key,)
    out: list[str] = []
    for cand in event.get("ability_source_candidates") or ():
        if isinstance(cand, str) and cand and cand not in out:
            out.append(cand)
    return tuple(out)


class Press(NamedTuple):
    """One recorded press: its tick, IL_Replay side (0 team, 1 opponent), the distinct
    catalogue ids its keys map to (in key order; keys the card map or the catalogue lack are
    left out), its keys and its source_index."""
    tick: int
    side: int
    cards: tuple[int, ...]
    keys: tuple[str, ...]
    index: int


def parse(events: Iterable[Mapping[str, Any]], names: Mapping[str, str],
          ids: Mapping[str, int]) -> list[Press]:
    """The activate_ability events of a match as Press, in (tick, source_index) order.
    `names`: IL_Replay base key -> card name (rs_cards.load()); `ids`: card name -> catalogue
    id. A missing tick reads 0 and a missing source_index 0, as sim/replay_corpus reads them."""
    out = []
    for e in events:
        if e.get("kind") != PRESS_KIND:
            continue
        keys = event_keys(e)
        cards: list[int] = []
        for key in keys:
            name = names.get(split_key(key)[0])
            cid = ids.get(name) if name is not None else None
            if cid is not None and cid not in cards:
                cards.append(int(cid))
        out.append(Press(int(e.get("replay_tick_20hz") or 0), 0 if e.get("side") == "team" else 1,
                         tuple(cards), keys, int(e.get("source_index") or 0)))
    out.sort(key=lambda p: (p.tick, p.index))
    return out


def button_of(cards: Sequence[int], buttons: Sequence[int]) -> tuple[int | None, str]:
    """(button, PRESS_OK) when exactly one button of `buttons` (side_buttons / engine_order)
    is behind the candidate `cards`; (None, SKIP_UNKNOWN) when none is, (None, SKIP_AMBIGUOUS)
    when several are."""
    order = [int(c) for c in buttons]
    hits = sorted({order.index(int(c)) for c in cards if int(c) in order})
    if len(hits) == 1:
        return hits[0], PRESS_OK
    return None, (SKIP_UNKNOWN if not hits else SKIP_AMBIGUOUS)


def command(team: int, button: int, hand: int = FALLBACK_HAND_SIZE) -> tuple[int, int, int, int]:
    """The step() command that presses `team`'s `button`: (team, hand + button, 0, 0)."""
    return (int(team), int(hand) + int(button), 0, 0)


def _row(row: Sequence) -> tuple[int, int, int, int, int]:
    """(available, spent, cost, card_id, cooldown_ticks) of an abilities row, by position
    (royalegym ability_row); a column the engine does not send reads 0, EMPTY_CARD, -1."""
    vals = [int(v) for v in list(row)[:5]]
    return tuple(vals + [0, 0, 0, EMPTY_CARD, -1][len(vals):])


def ability_features(player: Mapping[str, Any], n_buttons: int,
                     buttons: Sequence[int] | None = None) -> tuple[list[int], list[list[float]]]:
    """(cards, feats) of one side's ability buttons (module doc, FEATURES): `n_buttons` entries
    each. A button's card is its abilities row's card_id, else the caller's `buttons`
    (side_buttons) entry, else 1 (a reported button nobody names); 0 for no button."""
    rows = list(player.get("abilities") or ())
    known = [int(c) for c in (buttons or ())]
    cards: list[int] = []
    feats: list[list[float]] = []
    for k in range(int(n_buttons)):
        card = known[k] if k < len(known) else EMPTY_CARD
        if k >= len(rows):
            cards.append(card + 2 if card >= 0 else 0)
            feats.append([0.0] * len(ABILITY_FEATS))
            continue
        available, spent, cost, card_id, cooldown = _row(rows[k])
        if card_id >= 0:
            card = card_id
        cards.append(card + 2 if card >= 0 else 1)
        cool = min(1.0, cooldown / COOLDOWN_TICKS) if cooldown > 0 else 0.0
        feats.append([1.0, 1.0 if available else 0.0, 1.0 if spent else 0.0,
                      cost / COST_SCALE, cool])
    return cards, feats


@dataclasses.dataclass
class PressLog:
    """One match's presses: events (activate_ability events), sent (commands sent), ok
    (accepted), refused (by DEPLOY_REASONS name), skipped (by button_of reason) and
    order_mismatch (sides whose engine button order differed from side_buttons')."""
    events: int = 0
    sent: int = 0
    ok: int = 0
    refused: dict = dataclasses.field(default_factory=dict)
    skipped: dict = dataclasses.field(default_factory=dict)
    order_mismatch: int = 0

    def skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def result(self, reason: str) -> bool:
        """Counts one press result by its DEPLOY_REASONS name; True when it was accepted."""
        if reason == "OK":
            self.ok += 1
            return True
        self.refused[reason] = self.refused.get(reason, 0) + 1
        return False

    def to_json(self) -> dict:
        return {"events": self.events, "sent": self.sent, "ok": self.ok,
                "refused": dict(sorted(self.refused.items())),
                "skipped": dict(sorted(self.skipped.items())),
                "order_mismatch": self.order_mismatch}


def vocab_entry(n_buttons: int, hand: int = FALLBACK_HAND_SIZE) -> dict:
    """vocab.json "presses" of rows built under --presses on."""
    return {"mode": PRESS_ON, "buttons": int(n_buttons), "slot_base": int(hand),
            "button_order": "heroes (form 2) in dealt-deck order, then champions; the engine's "
                            "abilities rows when they name their cards",
            "act_press": "0 none, k + 1 button k",
            "ability_feats": list(ABILITY_FEATS), "cost_scale": COST_SCALE,
            "cooldown_ticks": COOLDOWN_TICKS}


def describe_log(log: Mapping[str, Any]) -> str:
    """One line for the console from PressLog.to_json() counts (or their sum over matches)."""
    refused = log.get("refused") or {}
    skipped = log.get("skipped") or {}
    text = (f"{log.get('sent', 0):,} of {log.get('events', 0):,} presses sent, "
            f"{log.get('ok', 0):,} accepted")
    if refused:
        text += ", refused " + ", ".join(f"{k} {v:,}" for k, v in sorted(refused.items()))
    if skipped:
        text += ", skipped " + ", ".join(f"{k} {v:,}" for k, v in sorted(skipped.items()))
    if log.get("order_mismatch"):
        text += f", engine button order differed on {log['order_mismatch']:,} sides"
    return text
