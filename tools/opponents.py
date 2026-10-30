r"""The opponents' decks from the official API: Yodle2's spars, the clanmates' battle logs and the
top players' decks, pulled into one local store (U03 of docs/SEASON_UPDATES.md).

    .venvs\royalegym\Scripts\python.exe tools\opponents.py init --clan "#..." --owner "#..."
                                                         --bot "#..." [--friend "#..."]
    .venvs\royalegym\Scripts\python.exe tools\opponents.py probe
    .venvs\royalegym\Scripts\python.exe tools\opponents.py pull
    .venvs\royalegym\Scripts\python.exe tools\opponents.py meta [--if-due]
    .venvs\royalegym\Scripts\python.exe tools\opponents.py add --deck "hog-rider,musketeer-hero,..."
    .venvs\royalegym\Scripts\python.exe tools\opponents.py status

Spec: docs/SEASON_UPDATES.md 4.1 (the daily pull, about 55 calls; Mondays the top players,
about 1,000 calls at 2 per second), 7.1 (the source and what is unconfirmed), 7.2 (what is
stored), 7.3 (card names: cr_api.slug), 7.7 (courtesy and privacy), 12 (U03) and 13.1-13.3 (the
owner's decisions of 2026-10-30: an API key in .secrets\supercell_token, the clan told, no
names or tags committed). HTTP goes through tools/cr_api.py (U02): the shared rate, the 429
back-off, the custom User-Agent, IPv4, and HTTP codes instead of exits.

Commands:
  init   writes <data>/opponents/config.json: the clan tag, the owner's main account, Yodle2
         (the bot) and any friends who spar Yodle2 from outside the clan. Run again to change a
         field; --friend adds and --drop-friend removes. --check looks each tag up (one call
         each). data/opponents.example.json is the committed template: placeholders only.
  pull   the daily pull: the clan roster, the battle log of Yodle2, the owner, every member and
         every friend, then /v1/cards. New 1v1 battles go into battles.jsonl.
  meta   the top Path of Legends players and their battle logs; their Path of Legends 1v1
         battles go into meta_battles.jsonl, kept for META_WEEKS weeks. --if-due skips the run
         when the last complete one is less than META_DUE_DAYS days old (for the daily task).
  add    an owner-entered deck (manual.json), as RoyaleAPI keys with -ev1 / -hero: the same deck
         strings as data/ladder/battles_2p6.csv. --list shows them, --remove ID drops one.
  probe  what the PC's real battle log says about the fields the documented API leaves open
         (cr_api.UNCONFIRMED: battlelog_forms, hero_slot_missing, battle_types, log_length,
         friendly_levels). Run it after a friendly in which the main account plays an
         evolution in slot 1 and a hero in slot 2 against Yodle2 (section 7.1). It reads the
         main account's profile (its collection and currentDeck) and battle log, and Yodle2's
         log, and it recommends a form rule.
  status what the store holds: last pull, battles by kind, Yodle2's spar record, meta, cards.

The store, <data>/opponents/ (<data> is the machine's data root, src/hastycr/machine.py):
  config.json          the tags (init)
  battles.jsonl        one normalised 1v1 battle per line, deduplicated on the battle time plus
                       the two tags sorted, as tools/ladder_pull.py does
  meta_battles.jsonl   the top players' Path of Legends battles, rolling META_WEEKS weeks
  manual.json          owner-entered decks (add)
  roster.json          every clan member ever seen, with first_seen, last_seen and left
  cards_api.json       the latest /v1/cards: {"fetched_at", "digest", "data": <the API body>}
  cards_api.prev.json  the snapshot before the last change of the body (for the roster radar)
  state.json           pulled_at (the last pull without a fatal error), per-player log state,
                       meta_at, the cards digest, the last pull's and meta's summaries
  probe.json           the last probe's findings
  raw/<date>/          every response, gzipped, kept keep_raw_days (30) days, so a parser fix
                       can re-parse
  .gitignore           "*": the folder stays out of git even under another data root
Everything there carries tags and names, so it is gitignored (data/opponents/ in .gitignore,
and the folder's own .gitignore). Exports for RL and evaluation (U16) hold decks only.

A battle record (battles.jsonl, meta_battles.jsonl; "v" RECORD_VERSION):
  id, battle_time (the API's 20261029T201500.000Z), type, game_mode {id, name},
  deck_selection, hosted, form_rule, kind, seen_at, and two sides sorted by tag. A side: tag,
  name, clan, role, crowns, king_hp, princess_hp, trophies, trophy_change, elixir_leaked,
  tower (the first supportCards key), support, n_cards, complete (8 different cards), deck
  (the sorted RoyaleAPI deck string, forms as -ev1 / -hero), ambiguous (keys whose form the
  rule cannot tell), and cards in the order the API listed them: key (cr_api.slug), slot,
  form (0 base, 1 evolution, 2 hero, None when ambiguous), name, id, level, max_level,
  evolution_level and max_evolution_level; a field the API did not give is left out (read
  with .get). Meta records are slim: a side keeps tag, role, crowns, tower, rating, rank and
  the deck fields, and a card keeps key, slot, form and the two evolution fields, no names or
  levels (about 1.5 KB a battle; 8 weeks of the top 1,000 are of the order of 100,000).
  Both files are read and rewritten line by line, never loaded whole (16 GB RAM).
  Roles: "bot" (Yodle2), "owner", "member" (anyone the roster has ever listed), "friend"
  (config), "top" (on the board, meta only) or "" (anyone else). Kinds: "spar" (Yodle2 against
  the owner, a member or a friend), "bot_unknown" (Yodle2 against anyone else: add them with
  init --friend if they are a sparring partner), "clan" (an owner's, member's or friend's own
  battle), "top" (meta) and "other". Spars are recognised by tag, never by the battle's type
  string, which is unconfirmed (section 7.1). Every pull recomputes roles, kinds and forms of
  the whole file from the current config, roster and form rule, so a new friend or a changed
  rule reaches old battles with no re-pull.
  Dropped, and counted: "2v2" and other battles that are not one against one, "no_deck" (a side
  without cards), "not_one_deck" (more than 8 cards on a side: duels may list several decks,
  unconfirmed), "no_time", "no_tag".

Forms (FORM_RULES). The API's evolutionLevel is a bit mask, 1 evolution and 2 hero
(API_NOTES.md). In a profile's currentDeck it is what the player OWNS, not what is equipped,
and the deck's slots fix the form: slot 1 takes an evolution, slot 2 a hero (or a champion),
slot 3 either (API_NOTES.md). Whether a battle-log card's evolutionLevel is the form played or
what the player owns is unconfirmed; the probe settles it. The raw bits and the slot are kept on
every card, so any rule can be applied later:
  slot   (default) a form only where the slot takes it: slot 1 evolution if bit 1, slot 2 hero
         if bit 2, slot 3 whichever bit is set (None when both are), slots 4-8 base. Right under
         either meaning, as long as the battle log lists cards in slot order.
  field  evolutionLevel is the form played: 1 evolution, 2 hero, 3 None, whatever the slot.
         For a log whose card order is not the slot order but which carries the played form.
  base   every card in its base form (the worst case of section 14.5).
A side with 7 cards is kept, complete false: API_NOTES.md found 77 of 300 currentDeck arrays
with 7 cards, the hero in the champion slot left out. Its cards are read as slots 1, 3, 4, ...
8 (MISSING_SLOT 2 absent). Resolution to engine names and forms (exact, downgraded, auto,
blocked, incomplete, inferred) is U16's, from these fields.

Sources (docs/SEASON_UPDATES.md 15):
* https://gist.github.com/loganlinn/d813ba0f9ccb47f34462ec8abeab8e14 (Swagger 2.0 of
  api.clashroyale.com/v1): GET /players/{playerTag}/battlelog returns a BattleLog, an array of
  battles with type, battleTime (string), arena, gameMode {id, name}, deckSelection, team and
  opponent; a side has tag, name, startingTrophies, trophyChange, crowns, clan and cards
  (name, level, maxLevel, iconUrls). type and deckSelection are plain strings with no enum.
  GET /clans/{clanTag}/members returns {items: [ClanMember]}: tag, name, role (a string),
  expLevel, trophies, clanRank, donations. That spec predates evolutions: evolutionLevel,
  supportCards, kingTowerHitPoints and princessTowersHitPoints are not in it, so every one of
  them is read when present and None otherwise.
* https://raw.githubusercontent.com/samaksh-bajaj/cr-deck-builder/main/API_NOTES.md: the
  bit masks (maxEvolutionLevel on /cards matches the card's icons; hero-only cards never show
  evolutionLevel 1), the slot layout, "no way to tell what a player actually has equipped",
  the 7-card currentDeck, displayed level = level + (16 - maxLevel), and the Path of Legends
  rankings of completed seasons at /locations/global/pathoflegend/{seasonId}/rankings/players.
  It does not mention the battle log.
* https://github.com/RoyaleAPI/cr-api-docs/blob/master/docs/proxy.md: --base proxy (U02).
* tools/ladder_pull.py: the board path locations/global/pathoflegend/players, the ranked types
  (TOP_TYPES here equals its RANKED_TYPES) and the dedupe key.

UNCONFIRMED (here, beyond cr_api.UNCONFIRMED) lists what this module assumes; the probe and the
pull's own counts report on each.

Courtesy (7.7): DEFAULT_RATE 2 requests per second (cr_api.DEFAULT_RATE), a config rate above
MAX_RATE is refused, a 429 backs off (cr_api), and a 403 or 503 stops the run at once, since
every later call would fail the same way. The daily pull is about 55 calls.

Exit codes: 0 ok; 1 the API answered with an error, a log failed, or another pull holds the
lock; 2 a bad setting (no config, a placeholder tag, no token, a bad deck).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import cr_api  # noqa: E402

EXAMPLE_CONFIG = ROOT / "data" / "opponents.example.json"
FOLDER = "opponents"
CONFIG = "config.json"
BATTLES = "battles.jsonl"
META_BATTLES = "meta_battles.jsonl"
MANUAL = "manual.json"
ROSTER = "roster.json"
CARDS = "cards_api.json"
CARDS_PREV = "cards_api.prev.json"
STATE = "state.json"
PROBE = "probe.json"
RAW = "raw"
LOCK = "pull.lock"
IGNORE_TEXT = ("# tools/opponents.py: battle logs, tags and names of real players. Local only.\n"
               "*\n")

SCHEMA = 1
RECORD_VERSION = 1
EXIT_OK, EXIT_API, EXIT_SETTING = cr_api.EXIT_OK, cr_api.EXIT_API, cr_api.EXIT_SETTING

#: A 403 (key or IP) and a 503 (maintenance) fail every later call too: the run stops.
FATAL = frozenset({403, 503})
DEFAULT_RATE = cr_api.DEFAULT_RATE
MAX_RATE = 5.0                  # requests per second; a config asking for more is refused
KEEP_RAW_DAYS = 30
META_TOP = 1000
META_WEEKS = 8
META_DUE_DAYS = 6.5             # meta --if-due runs when the last complete one is older
META_WORKERS = 2
META_SEASONS_BACK = 3           # completed seasons tried when the current board gives nothing
LOCK_STALE_S = 3 * 3600

DECK_SIZE = 8
EVO_BIT, HERO_BIT = 1, 2
#: The slots that take a form (API_NOTES.md): slot 1 an evolution, slot 2 a hero, slot 3 either.
EVO_SLOTS = frozenset({1, 3})
HERO_SLOTS = frozenset({2, 3})
#: The slot a 7-card deck lacks: the hero in the champion slot (API_NOTES.md, for currentDeck).
MISSING_SLOT = 2
FORM_RULES = ("slot", "field", "base")
FORM_SUFFIX = {1: "-ev1", 2: "-hero"}
#: Suffixes `add` reads (cr_api.slug of what the owner typed).
ADD_SUFFIXES = (("-ev1", 1), ("-evo", 1), ("-evolution", 1), ("-hero", 2))

#: Path of Legends battle types, as tools/ladder_pull.RANKED_TYPES.
TOP_TYPES = frozenset({"pathOfLegend", "pathOfLegendMatch"})
#: Roles that make Yodle2's opponent a sparring partner.
PARTNER_ROLES = frozenset({"owner", "member", "friend"})
KINDS = ("spar", "bot_unknown", "clan", "top", "other")
#: Real tags use only these characters; another one is warned about, not refused.
TAG_ALPHABET = frozenset("0289PYLQGRJCUV")

BOARD_PATH = "locations/global/pathoflegend/players"
SEASON_BOARD_PATH = "locations/global/pathoflegend/{season}/rankings/players"

#: What this module assumes beyond cr_api.UNCONFIRMED. probe and pull report on each.
UNCONFIRMED: dict[str, str] = {
    "card_order": "whether a battle-log side lists its cards in deck-slot order (the slot rule "
                  "needs it); probe compares the log with the profile's currentDeck",
    "seven_card_slots": "that a 7-card battle-log side lacks its slot-2 card, so its cards are "
                        "slots 1, 3-8 (API_NOTES.md shows it for currentDeck only)",
    "friendly_type": "the type string of a friendly; spars are found by tag, and probe prints "
                     "the type of every spar it sees",
    "big_decks": "whether a duel lists more than 8 cards on a side; such sides are dropped and "
                 "counted as not_one_deck",
    "tower_fields": "kingTowerHitPoints, princessTowersHitPoints, supportCards and "
                    "elixirLeaked on a battle-log side (not in the Swagger spec); kept when "
                    "present",
    "board_path": "whether locations/global/pathoflegend/players answers for the current "
                  "season; when it gives nothing, meta tries the rankings of the last "
                  "completed seasons, with ids written YYYY-MM (also unconfirmed)",
    "member_paging": "whether /clans/{tag}/members returns all 50 members in one answer; pull "
                     "follows paging.cursors.after when one is given",
}


class ConfigError(ValueError):
    """The config is missing, a tag is a placeholder, or a value is out of range."""


class Busy(RuntimeError):
    """Another pull or meta run holds the store's lock."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(value: dt.datetime) -> str:
    """UTC, seconds, with a Z (the form tools/game_versions.parse_time reads)."""
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


def parse_iso(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        got = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return (got if got.tzinfo else got.replace(tzinfo=dt.timezone.utc)).astimezone(
        dt.timezone.utc)


def parse_battle_time(value: Any) -> dt.datetime | None:
    """The API's battleTime, "20261029T201500.000Z", as an aware UTC datetime, or None."""
    for fmt in ("%Y%m%dT%H%M%S.%fZ", "%Y%m%dT%H%M%SZ"):
        try:
            return dt.datetime.strptime(str(value), fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _bits(value: Any) -> int:
    got = _int(value)
    return got if got is not None and got > 0 else 0


def clean_tag(value: Any) -> str:
    """cr_api.norm_tag, or "" when the value is no tag."""
    try:
        return cr_api.norm_tag(str(value or ""))
    except ValueError:
        return ""


def odd_tag_letters(tag: str) -> str:
    """The characters of `tag` that real tags never use (TAG_ALPHABET), or ""."""
    return "".join(sorted(set(tag.lstrip("#")) - TAG_ALPHABET))


def battle_id(battle_time: str, tag_a: str, tag_b: str) -> str:
    """The dedupe key: battle time plus the two tags sorted (tools/ladder_pull.py's key)."""
    low, high = sorted((tag_a, tag_b))
    return f"{battle_time}|{low}|{high}"


def deck_id(deck: str) -> str:
    """A short, stable name for a deck string (manual entries; U16's pool tags)."""
    return hashlib.sha1(deck.encode("utf-8")).hexdigest()[:10]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError):
        return default


def write_json(path: Path, obj: Any) -> None:
    """Atomically: a temporary file next to it, then os.replace."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                   newline="\n")
    os.replace(tmp, path)


def scan_jsonl(path: Path) -> Iterator[dict[str, Any] | None]:
    """Every non-empty line of a .jsonl file, one at a time: the record, or None for a line
    that is not a JSON object with an id. Nothing is held in memory (the files grow for
    months; the PC has 16 GB)."""
    try:
        handle = Path(path).open(encoding="utf-8-sig")
    except FileNotFoundError:
        return
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                yield None
                continue
            yield obj if isinstance(obj, dict) and obj.get("id") else None


def read_jsonl(path: Path) -> tuple[list[dict], int]:
    """All records of a .jsonl file and the number of unreadable lines (skipped). For small
    files and tests; the store itself streams (scan_jsonl)."""
    out: list[dict] = []
    bad = 0
    for record in scan_jsonl(path):
        if record is None:
            bad += 1
        else:
            out.append(record)
    return out, bad


def _line(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_line(record))


def _time_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("battle_time") or ""), str(record.get("id") or "")


# ---------------------------------------------------------------------------
# forms and decks
# ---------------------------------------------------------------------------

def slots_for(n: int) -> list[int]:
    """The deck slot (1-8) of each listed card: 1..n, or for 7 cards 1, 3, 4, ... 8 (the
    slot-2 card is the one left out)."""
    if n == DECK_SIZE - 1:
        return [s for s in range(1, DECK_SIZE + 1) if s != MISSING_SLOT]
    return list(range(1, n + 1))


def form_of(slot: int | None, bits: Any, rule: str = "slot") -> int | None:
    """The form a battle-log card was played in: 0 base, 1 evolution, 2 hero, or None when the
    rule cannot tell. `bits` is the card's evolutionLevel (None when absent), `slot` its deck
    slot (1-8). See FORM_RULES in the module docstring."""
    if rule not in FORM_RULES:
        raise ValueError(f"form rule {rule!r}: use one of {', '.join(FORM_RULES)}")
    got = _bits(bits)
    if rule == "base" or not got:
        return 0
    evo, hero = bool(got & EVO_BIT), bool(got & HERO_BIT)
    if rule == "slot":
        evo = evo and slot in EVO_SLOTS
        hero = hero and slot in HERO_SLOTS
    if evo and hero:
        return None
    return 1 if evo else 2 if hero else 0


def card_token(key: str, form: int | None) -> str:
    """A RoyaleAPI deck token: the key, plus -ev1 or -hero for a form (None reads as base)."""
    return key + FORM_SUFFIX.get(form or 0, "")


def deck_string(cards: Iterable[Mapping[str, Any]]) -> str:
    """The sorted, comma-joined tokens, as data/ladder/battles_2p6.csv writes a deck."""
    return ",".join(sorted(card_token(str(c.get("key") or ""), c.get("form")) for c in cards))


def split_form(key: str) -> tuple[str, int]:
    """(base key, form) of a deck token `add` was given; -ev1 / -evo / -evolution and -hero."""
    for suffix, form in ADD_SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)], form
    return key, 0


# ---------------------------------------------------------------------------
# roles and kinds
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Partners:
    """Who is who, by tag. `members` holds every tag the roster has ever listed, so a spar with
    someone who later left stays a spar."""
    bot: str = ""
    owner: str = ""
    members: frozenset[str] = frozenset()
    friends: frozenset[str] = frozenset()

    def role_of(self, tag: str) -> str:
        if tag and tag == self.bot:
            return "bot"
        if tag and tag == self.owner:
            return "owner"
        if tag in self.members:
            return "member"
        if tag in self.friends:
            return "friend"
        return ""


def kind_of(sides: Sequence[Mapping[str, Any]]) -> str:
    roles = [str(s.get("role") or "") for s in sides]
    if "bot" in roles:
        other = roles[1] if roles[0] == "bot" else roles[0]
        return "spar" if other in PARTNER_ROLES else "bot_unknown"
    if any(role in PARTNER_ROLES for role in roles):
        return "clan"
    if "top" in roles:
        return "top"
    return "other"


# ---------------------------------------------------------------------------
# normalising a battle
# ---------------------------------------------------------------------------

def card_entry(card: Mapping[str, Any], slot: int, rule: str,
               slim: bool = False) -> dict[str, Any]:
    """One card: key, slot, form, and the API's evolutionLevel / maxEvolutionLevel when it gave
    them; unless slim, also name, id, level and max_level. Fields the API left out are left
    out here too (readers use .get), except form, whose None means ambiguous."""
    name = str(card.get("name") or "")
    bits = _int(card.get("evolutionLevel"))
    entry: dict[str, Any] = {"key": cr_api.slug(name), "slot": slot,
                             "form": form_of(slot, bits, rule)}
    fields = [("evolution_level", "evolutionLevel"), ("max_evolution_level", "maxEvolutionLevel")]
    if not slim:
        entry["name"] = name
        fields += [("id", "id"), ("level", "level"), ("max_level", "maxLevel")]
    for field, source in fields:
        value = _int(card.get(source))
        if value is not None:
            entry[field] = value
    return entry


def _support(side: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in side.get("supportCards") or []:
        if isinstance(item, Mapping):
            name = str(item.get("name") or "")
            out.append({"name": name, "key": cr_api.slug(name), "id": _int(item.get("id")),
                        "level": _int(item.get("level")), "max_level": _int(item.get("maxLevel"))})
    return out


def _deck_fields(side: dict[str, Any]) -> None:
    cards = side["cards"]
    keys = [c["key"] for c in cards]
    side["n_cards"] = len(cards)
    side["complete"] = len(cards) == DECK_SIZE and len(set(keys)) == DECK_SIZE
    side["deck"] = deck_string(cards)
    side["ambiguous"] = [c["key"] for c in cards if c.get("form") is None]


def side_entry(side: Mapping[str, Any], role: str, rule: str,
               slim: bool = False) -> dict[str, Any]:
    """One side of a battle. Slim (meta records): tag, role, crowns, tower and slim cards only,
    no names, levels or tower HP."""
    cards = [c for c in side.get("cards") or [] if isinstance(c, Mapping)]
    entries = [card_entry(c, s, rule, slim) for c, s in zip(cards, slots_for(len(cards)))]
    support = _support(side)
    out: dict[str, Any] = {"tag": clean_tag(side.get("tag")), "role": role,
                           "crowns": _int(side.get("crowns")),
                           "tower": support[0]["key"] if support else None, "cards": entries}
    if not slim:
        clan = side.get("clan") if isinstance(side.get("clan"), Mapping) else {}
        princess = side.get("princessTowersHitPoints")
        out.update({
            "name": str(side.get("name") or ""), "clan": clean_tag(clan.get("tag")) or None,
            "king_hp": _int(side.get("kingTowerHitPoints")),
            "princess_hp": ([_int(v) for v in princess] if isinstance(princess, list)
                            else None),
            "trophies": _int(side.get("startingTrophies")),
            "trophy_change": _int(side.get("trophyChange")),
            "elixir_leaked": _float(side.get("elixirLeaked")), "support": support})
    _deck_fields(out)
    return out


def normalise(battle: Any, *, role_of: Callable[[str], str], rule: str = "slot",
              seen_at: str | None = None, slim: bool = False
              ) -> tuple[dict[str, Any] | None, str]:
    """(record, "kept") for a 1v1 battle of the API's battle log, else (None, why). Slim
    records (meta) leave names, levels and tower HP out."""
    if not isinstance(battle, Mapping):
        return None, "not_a_battle"
    team, opp = battle.get("team"), battle.get("opponent")
    if not isinstance(team, list) or not isinstance(opp, list) or len(team) != 1 \
            or len(opp) != 1:
        two = (isinstance(team, list) and len(team) == 2) or \
              (isinstance(opp, list) and len(opp) == 2)
        return None, "2v2" if two else "not_1v1"
    if not isinstance(team[0], Mapping) or not isinstance(opp[0], Mapping):
        return None, "not_1v1"
    when = str(battle.get("battleTime") or "")
    if parse_battle_time(when) is None:
        return None, "no_time"
    tags = [clean_tag(team[0].get("tag")), clean_tag(opp[0].get("tag"))]
    if not all(tags) or tags[0] == tags[1]:
        return None, "no_tag"
    counts = [len([c for c in s.get("cards") or [] if isinstance(c, Mapping)])
              for s in (team[0], opp[0])]
    if min(counts) == 0:
        return None, "no_deck"
    if max(counts) > DECK_SIZE:
        return None, "not_one_deck"
    sides = sorted((side_entry(team[0], role_of(tags[0]), rule, slim),
                    side_entry(opp[0], role_of(tags[1]), rule, slim)), key=lambda s: s["tag"])
    mode = battle.get("gameMode") if isinstance(battle.get("gameMode"), Mapping) else {}
    record = {
        "v": RECORD_VERSION, "id": battle_id(when, *tags), "battle_time": when,
        "type": str(battle.get("type") or ""),
        "game_mode": {"id": _int(mode.get("id")), "name": str(mode.get("name") or "")},
        "deck_selection": str(battle.get("deckSelection") or ""),
        "hosted": battle.get("isHostedMatch") if isinstance(battle.get("isHostedMatch"),
                                                            bool) else None,
        "form_rule": rule, "kind": kind_of(sides), "sides": sides, "seen_at": seen_at}
    return record, "kept"


def refresh(record: Mapping[str, Any], *, rule: str,
            role_of: Callable[[str], str] | None = None,
            kind: str | None = None) -> dict[str, Any]:
    """The record with forms re-derived under `rule` from the kept bits and slots, roles from
    `role_of` (kept when None) and the kind recomputed (or set to `kind`)."""
    out = json.loads(json.dumps(record))
    for side in out.get("sides") or []:
        if role_of is not None:
            side["role"] = role_of(str(side.get("tag") or ""))
        for card in side.get("cards") or []:
            card["form"] = form_of(card.get("slot"), card.get("evolution_level"), rule)
        _deck_fields(side)
    out["form_rule"] = rule
    out["kind"] = kind if kind is not None else kind_of(out.get("sides") or [])
    return out


def side_of(record: Mapping[str, Any], tag: str) -> dict[str, Any] | None:
    for side in record.get("sides") or []:
        if side.get("tag") == tag:
            return side
    return None


def other_side(record: Mapping[str, Any], tag: str) -> dict[str, Any] | None:
    sides = record.get("sides") or []
    for i, side in enumerate(sides):
        if side.get("tag") == tag and len(sides) == 2:
            return sides[1 - i]
    return None


def result_of(record: Mapping[str, Any], tag: str) -> str:
    """"W", "D" or "L" for `tag` by crowns, or "" when unknown."""
    me, them = side_of(record, tag), other_side(record, tag)
    if me is None or them is None or me.get("crowns") is None or them.get("crowns") is None:
        return ""
    return "W" if me["crowns"] > them["crowns"] else "L" if me["crowns"] < them["crowns"] else "D"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    clan: str
    owner: str
    bot: str
    friends: tuple[str, ...] = ()
    base: str = ""
    transport: str = "urllib"
    rate: float = DEFAULT_RATE
    form_rule: str = "slot"
    keep_raw_days: int = KEEP_RAW_DAYS
    meta_top: int = META_TOP
    meta_weeks: int = META_WEEKS
    warnings: tuple[str, ...] = ()

    def partners(self, members: Iterable[str] = ()) -> Partners:
        return Partners(self.bot, self.owner, frozenset(members), frozenset(self.friends))

    def to_json(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out.pop("warnings")
        out["friends"] = list(self.friends)
        return {"schema": SCHEMA, **out}


def _tag_value(value: Any, where: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ConfigError(f"{where} is not set: run tools/opponents.py init")
    got = clean_tag(text)
    if not got:
        if "_" in text or "TAG" in text.upper():
            raise ConfigError(f"{where} is {text!r}, the template's placeholder: set the real tag "
                              "with tools/opponents.py init")
        raise ConfigError(f"{where}: {text!r} is not a tag")
    return got


def parse_config(data: Any, where: str = CONFIG) -> Config:
    """A Config from the JSON object. ConfigError names the first bad field."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where} is not a JSON object")
    warnings = []
    known = {f.name for f in dataclasses.fields(Config)} - {"warnings"} | {"schema"}
    for key in data:
        if not str(key).startswith("_") and key not in known:
            warnings.append(f"{where}: unknown setting {key!r} ignored")
    tags = {name: _tag_value(data.get(name), f"{where} {name}") for name in ("clan", "owner",
                                                                             "bot")}
    friends_raw = data.get("friends") or []
    if not isinstance(friends_raw, list):
        raise ConfigError(f"{where} friends must be a list of tags")
    friends = tuple(dict.fromkeys(_tag_value(t, f"{where} friends[{i}]")
                                  for i, t in enumerate(friends_raw)))
    for name, tag in [*tags.items(), *(("friend", t) for t in friends)]:
        odd = odd_tag_letters(tag)
        if odd:
            warnings.append(f"{where} {name} {tag}: real tags never use {odd!r} (only "
                            f"{''.join(sorted(TAG_ALPHABET))}); check it")
    if tags["owner"] == tags["bot"]:
        raise ConfigError(f"{where}: owner and bot are the same tag")
    base = str(data.get("base") or "").strip()
    if base:
        try:
            cr_api.resolve_base(base, {})
        except ValueError as exc:
            raise ConfigError(f"{where} base: {exc}") from None
    transport = str(data.get("transport") or "urllib").strip().lower()
    if transport not in ("urllib", "curl"):
        raise ConfigError(f"{where} transport {transport!r}: use urllib or curl")
    rate = data.get("rate", DEFAULT_RATE)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0.1 <= rate <= MAX_RATE:
        raise ConfigError(f"{where} rate {rate!r}: between 0.1 and {MAX_RATE:g} requests per "
                          "second (courtesy, section 7.7)")
    rule = str(data.get("form_rule") or "slot")
    if rule not in FORM_RULES:
        raise ConfigError(f"{where} form_rule {rule!r}: use one of {', '.join(FORM_RULES)}")
    ints = {}
    for name, low, high, default in (("keep_raw_days", 0, 3650, KEEP_RAW_DAYS),
                                     ("meta_top", 1, 2000, META_TOP),
                                     ("meta_weeks", 1, 52, META_WEEKS)):
        value = data.get(name, default)
        if _int(value) is None or not low <= value <= high:
            raise ConfigError(f"{where} {name} {value!r}: a whole number from {low} to {high}")
        ints[name] = value
    return Config(clan=tags["clan"], owner=tags["owner"], bot=tags["bot"], friends=friends,
                  base=base, transport=transport, rate=float(rate), form_rule=rule,
                  warnings=tuple(warnings), **ints)


def template() -> dict[str, Any]:
    """data/opponents.example.json without its tags, or the defaults when it is missing."""
    data = read_json(EXAMPLE_CONFIG)
    if not isinstance(data, dict):
        data = {"schema": SCHEMA, "friends": [], "base": "", "transport": "urllib",
                "rate": DEFAULT_RATE, "form_rule": "slot", "keep_raw_days": KEEP_RAW_DAYS,
                "meta_top": META_TOP, "meta_weeks": META_WEEKS}
    data = {k: v for k, v in data.items() if k not in ("clan", "owner", "bot", "_comment")}
    data["friends"] = []
    return data


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------

def data_root() -> Path:
    """The machine's data root (src/hastycr/machine.py), else <repo>/data."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from hastycr import machine
        return Path(machine.data_root())
    except Exception as exc:                    # a broken machine.json: say so, use data/
        print(f"machine settings unreadable ({type(exc).__name__}: {exc}); using "
              f"{ROOT / 'data'}", file=sys.stderr)
        return ROOT / "data"


class Store:
    """<data>/opponents/ and its files."""

    def __init__(self, folder: str | os.PathLike):
        self.folder = Path(folder)

    def __getattr__(self, name: str) -> Path:
        files = {"config": CONFIG, "battles": BATTLES, "meta": META_BATTLES, "manual": MANUAL,
                 "roster": ROSTER, "cards": CARDS, "cards_prev": CARDS_PREV, "state": STATE,
                 "probe": PROBE, "raw": RAW, "lock": LOCK}
        if name in files:
            return self.folder / files[name]
        raise AttributeError(name)

    def ensure(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        ignore = self.folder / ".gitignore"
        if not ignore.exists():
            ignore.write_text(IGNORE_TEXT, encoding="utf-8", newline="\n")

    def load_config(self) -> Config:
        if not self.config.is_file():
            raise ConfigError(f"no config at {self.config}: run tools/opponents.py init --clan "
                              "\"#...\" --owner \"#...\" --bot \"#...\"")
        data = read_json(self.config)
        if data is None:
            raise ConfigError(f"{self.config} is not valid JSON")
        return parse_config(data, str(self.config))

    def read_state(self) -> dict[str, Any]:
        got = read_json(self.state, {})
        return got if isinstance(got, dict) else {}

    def write_state(self, state: Mapping[str, Any]) -> None:
        write_json(self.state, {"schema": SCHEMA, **{k: v for k, v in state.items()
                                                     if k != "schema"}})

    def read_roster(self) -> dict[str, Any]:
        got = read_json(self.roster, {})
        if not isinstance(got, dict):
            got = {}
        if not isinstance(got.get("members"), dict):
            got["members"] = {}
        return got

    @staticmethod
    def ids(path: Path) -> set[str]:
        return {r["id"] for r in scan_jsonl(path) if r is not None}

    def merge(self, path: Path, records: Iterable[Mapping[str, Any]],
              ids: set[str] | None = None) -> list[dict[str, Any]]:
        """Appends the records whose id the file lacks (and drops repeats among them); returns
        the ones added, oldest first. `ids`, when given, is the file's ids, and is updated."""
        ids = self.ids(path) if ids is None else ids
        added = []
        for record in records:
            if record["id"] not in ids:
                ids.add(record["id"])
                added.append(dict(record))
        added.sort(key=_time_key)
        if added:
            append_jsonl(path, added)
        return added

    def rewrite(self, path: Path, fn: Callable[[dict[str, Any]], dict[str, Any] | None]) -> int:
        """Applies fn to every record (None drops it) and keeps the first record of each id.
        When anything changed (a record, a drop, a repeat or an unreadable line), the file is
        rewritten, streaming, in its order. Returns that count. Two passes, so a store that is
        already current is only read."""
        def walk() -> Iterator[tuple[dict[str, Any] | None, bool]]:
            seen: set[str] = set()
            for record in scan_jsonl(path):
                if record is None:
                    yield None, True
                    continue
                new = fn(record)
                if new is None or new["id"] in seen:
                    yield None, True
                    continue
                seen.add(new["id"])
                yield new, new != record

        changed = sum(1 for _, differs in walk() if differs)
        if changed:
            path = Path(path)
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("w", encoding="utf-8", newline="\n") as handle:
                for new, _ in walk():
                    if new is not None:
                        handle.write(_line(new))
            os.replace(tmp, path)
        return changed

    def count(self, path: Path) -> int:
        return sum(1 for r in scan_jsonl(path) if r is not None)


class Lock:
    """An exclusive lock file for pull and meta; one older than LOCK_STALE_S is taken over."""

    def __init__(self, path: Path, *, stale: float = LOCK_STALE_S):
        self.path = Path(path)
        self.stale = stale

    def __enter__(self) -> Lock:
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except OSError:
                    continue
                if age <= self.stale:
                    raise Busy(f"another pull holds {self.path} (started {age / 60:.0f} min "
                               "ago); delete the file if no pull is running") from None
                try:
                    self.path.unlink()
                except OSError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "started": iso(utcnow())}))
            return self
        raise Busy(f"cannot take {self.path}")

    def __exit__(self, *exc) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def save_raw(store: Store, what: str, resp: cr_api.Response, stamp: dt.datetime,
             keep_days: int) -> Path | None:
    """The response, gzipped, under raw/<date>/<HHMMSS>_<what>.json.gz (none when keep_days
    is 0)."""
    if keep_days <= 0:
        return None
    folder = store.raw / stamp.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp.strftime('%H%M%S')}_{what}.json.gz"
    body = {"status": resp.status, "path": resp.path, "fetched_at": iso(stamp),
            "reason": resp.reason, "message": resp.message, "error": resp.error,
            "data": resp.data}
    path.write_bytes(gzip.compress(json.dumps(body, ensure_ascii=False).encode("utf-8")))
    return path


def prune_raw(store: Store, today: dt.date, keep_days: int) -> int:
    """Removes raw/<date> folders older than keep_days; returns how many."""
    if not store.raw.is_dir():
        return 0
    removed = 0
    for child in store.raw.iterdir():
        try:
            day = dt.date.fromisoformat(child.name)
        except ValueError:
            continue
        if child.is_dir() and (today - day).days > keep_days:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


def _tag_file(tag: str) -> str:
    return tag.lstrip("#")


def update_roster(roster: dict[str, Any], clan: str, items: Sequence[Any],
                  now: dt.datetime) -> list[str]:
    """Merges today's members into the roster (first_seen, last_seen, left); returns today's
    member tags in the API's order."""
    members = roster.setdefault("members", {})
    current: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        tag = clean_tag(item.get("tag"))
        if not tag or tag in current:
            continue
        current.append(tag)
        entry = members.setdefault(tag, {"first_seen": iso(now)})
        entry.update(name=str(item.get("name") or ""), role=str(item.get("role") or ""),
                     last_seen=iso(now))
        entry.pop("left", None)
    for tag, entry in members.items():
        if tag not in current and not entry.get("left"):
            entry["left"] = iso(now)
    roster["clan"] = clan
    roster["updated_at"] = iso(now)
    return current


def store_cards(store: Store, body: Any, now: dt.datetime) -> dict[str, Any]:
    """Writes cards_api.json; the old file becomes cards_api.prev.json when the body changed."""
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    old = read_json(store.cards)
    old_digest = old.get("digest") if isinstance(old, dict) else None
    changed = old_digest is not None and old_digest != digest
    if changed:
        os.replace(store.cards, store.cards_prev)
    write_json(store.cards, {"fetched_at": iso(now), "digest": digest, "data": body})
    items = body.get("items") if isinstance(body, dict) else None
    support = body.get("supportItems") if isinstance(body, dict) else None
    out = {"fetched_at": iso(now), "digest": digest, "changed": changed,
           "items": len(items) if isinstance(items, list) else None,
           "support_items": len(support) if isinstance(support, list) else None}
    if changed:
        out.update(changed_at=iso(now), previous_digest=old_digest)
    return out


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------

def build_client(cfg: Config | None, args: argparse.Namespace, store: Store, *,
                 transport: cr_api.Transport | None = None,
                 rate: cr_api.Rate | None = None) -> cr_api.Client:
    """The API client from the config and the command line (the command line wins)."""
    base = getattr(args, "base", None) or (cfg.base if cfg else "") or None
    name = getattr(args, "transport", None) or (cfg.transport if cfg else "urllib")
    per = getattr(args, "rate", None) or (cfg.rate if cfg else DEFAULT_RATE)
    if not 0.1 <= per <= MAX_RATE:
        raise ConfigError(f"--rate {per:g}: between 0.1 and {MAX_RATE:g} requests per second")
    try:
        if transport is None:
            transport = cr_api.make_transport(name, scratch=store.folder / ".scratch")
        return cr_api.Client(token_file=getattr(args, "token_file", None), base=base,
                             transport=transport, rate=rate or cr_api.Rate(per),
                             log=lambda line: print(f"  {line}", flush=True))
    except cr_api.TokenError:
        raise
    except ValueError as exc:                   # a bad base or transport
        raise ConfigError(str(exc)) from None


def _failure(what: str, resp: cr_api.Response) -> dict[str, Any]:
    return {"what": what, "status": resp.status, "reason": resp.reason,
            "error": resp.error, "hint": resp.hint()}


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

def _all_members(client: cr_api.Client, clan: str) -> tuple[list[Any] | None, cr_api.Response]:
    """Every member, following paging.cursors.after if the API pages (UNCONFIRMED)."""
    path = f"clans/{cr_api.tag_path(clan)}/members"
    resp = client.get(path)
    if not resp.ok or not isinstance(resp.data, dict) or not isinstance(
            resp.data.get("items"), list):
        return None, resp
    items = list(resp.data["items"])
    after = ((resp.data.get("paging") or {}).get("cursors") or {}).get("after")
    pages = 1
    while after and pages < 5:
        more = client.get(path, {"after": after})
        if not more.ok or not isinstance(more.data, dict):
            return None, more
        items.extend(more.data.get("items") or [])
        after = ((more.data.get("paging") or {}).get("cursors") or {}).get("after")
        pages += 1
    return items, resp


def pull(cfg: Config, client: cr_api.Client, store: Store, *,
         now: dt.datetime | None = None) -> dict[str, Any]:
    """The daily pull. Returns its summary (also written to state.json)."""
    now = now or utcnow()
    store.ensure()
    state = store.read_state()
    roster = store.read_roster()
    players_state = state.setdefault("players", {})
    failed: list[dict[str, Any]] = []
    fatal: cr_api.Response | None = None
    keep = cfg.keep_raw_days

    items, resp = _all_members(client, cfg.clan)
    save_raw(store, "clan_members", resp, now, keep)
    if items is not None:
        current = update_roster(roster, cfg.clan, items, now)
        write_json(store.roster, roster)
    else:
        failed.append(_failure("clan members", resp))
        fatal = resp if resp.status in FATAL else None
        current = [t for t, e in roster["members"].items() if not e.get("left")]
    partners = cfg.partners(roster["members"])
    order = list(dict.fromkeys([cfg.bot, cfg.owner, *current, *cfg.friends]))

    records: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    gaps: list[str] = []
    logs = logs_ok = 0
    for tag in order if fatal is None else ():
        logs += 1
        resp = client.battlelog(tag)
        save_raw(store, f"battlelog_{_tag_file(tag)}", resp, now, keep)
        if resp.status in FATAL:
            fatal = resp
            failed.append(_failure(f"battle log of a {partners.role_of(tag) or 'player'}", resp))
            break
        if not resp.ok or not isinstance(resp.data, list):
            failed.append(_failure(f"battle log of a {partners.role_of(tag) or 'player'}", resp))
            continue
        logs_ok += 1
        times = [t for t in (parse_battle_time(b.get("battleTime")) for b in resp.data
                             if isinstance(b, Mapping)) if t]
        before = players_state.get(tag) if isinstance(players_state.get(tag), dict) else {}
        last_newest = parse_iso(before.get("newest"))
        if last_newest and times and min(times) > last_newest:
            gaps.append(tag)
        players_state[tag] = {
            "pulled_at": iso(now), "battles": len(resp.data),
            "newest": iso(max(times)) if times else before.get("newest"),
            "oldest": iso(min(times)) if times else None, "role": partners.role_of(tag)}
        for battle in resp.data:
            record, why = normalise(battle, role_of=partners.role_of, rule=cfg.form_rule,
                                    seen_at=iso(now))
            if record is None:
                dropped[why] += 1
            else:
                records.append(record)

    cards = None
    if fatal is None:
        resp = client.cards()
        save_raw(store, "cards", resp, now, keep)
        if resp.ok and isinstance(resp.data, dict):
            cards = store_cards(store, resp.data, now)
            state["cards"] = {**(state.get("cards") or {}), **cards}
        else:
            failed.append(_failure("cards", resp))
            fatal = resp if resp.status in FATAL else None

    added = store.merge(store.battles, records)
    recomputed = store.rewrite(store.battles, lambda r: refresh(
        r, rule=cfg.form_rule, role_of=partners.role_of))
    prune_raw(store, now.date(), keep)

    spars = [r for r in added if r.get("kind") == "spar"]
    record_of = Counter(result_of(r, cfg.bot) for r in spars)
    strangers = sorted({(other_side(r, cfg.bot) or {}).get("tag", "") for r in added
                        if r.get("kind") == "bot_unknown"})
    result = {
        "started_at": iso(now), "ok": fatal is None and not failed,
        "fatal": fatal.status if fatal else None, "calls": client.calls,
        "statuses": {str(k): v for k, v in sorted(client.statuses.items())},
        "logs": logs, "logs_ok": logs_ok, "failed": failed, "added": len(added),
        "added_by_kind": dict(sorted(Counter(r.get("kind") for r in added).items())),
        "dropped": dict(sorted(dropped.items())), "recomputed": recomputed,
        "spars": {"W": record_of["W"], "D": record_of["D"], "L": record_of["L"]},
        "bot_unknown_opponents": strangers, "gaps": gaps, "cards": cards,
        "members": len(current), "form_rule": cfg.form_rule}
    state["last_attempt_at"] = iso(now)
    if fatal is None and logs_ok:
        state["pulled_at"] = iso(now)
    state["pull"] = result
    store.write_state(state)
    return result


# ---------------------------------------------------------------------------
# meta: the top Path of Legends players
# ---------------------------------------------------------------------------

def recent_seasons(now: dt.datetime, count: int = META_SEASONS_BACK) -> list[str]:
    """Season ids YYYY-MM from this month backwards (the id format is UNCONFIRMED)."""
    year, month = now.year, now.month
    out = []
    for _ in range(count):
        out.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return out


def top_board(client: cr_api.Client, top: int, now: dt.datetime
              ) -> tuple[list[dict], cr_api.Response | None, str]:
    """(players, last response, the path that answered). The current board first, then the
    rankings of the last completed seasons."""
    paths = [BOARD_PATH, *(SEASON_BOARD_PATH.format(season=s) for s in recent_seasons(now))]
    last = None
    for path in paths:
        players, last = client.paged(path, want=top)
        players = [p for p in players if isinstance(p, Mapping) and clean_tag(p.get("tag"))]
        if players:
            return players, last, path
        if last is not None and last.status in FATAL:
            break
    return [], last, ""


def meta(cfg: Config, client: cr_api.Client, store: Store, *, top: int | None = None,
         workers: int = META_WORKERS, now: dt.datetime | None = None) -> dict[str, Any]:
    now = now or utcnow()
    store.ensure()
    state = store.read_state()
    top = top or cfg.meta_top
    board, last, source = top_board(client, top, now)
    failed: list[dict[str, Any]] = []
    fatal = last if (not board and last is not None and last.status in FATAL) else None
    if not board:
        if last is not None:
            failed.append(_failure("the Path of Legends board", last))
        result = {"started_at": iso(now), "ok": False, "fatal": fatal.status if fatal else None,
                  "board": 0, "board_path": "", "logs_ok": 0, "failed": failed,
                  "failed_count": len(failed), "added": 0, "dropped": {},
                  "held": store.count(store.meta), "weeks": cfg.meta_weeks,
                  "calls": client.calls}
        state["meta"] = result
        store.write_state(state)
        return result
    standing = {clean_tag(p.get("tag")): (_int(p.get("eloRating")), _int(p.get("rank")))
                for p in board}
    abort = threading.Event()

    def one(tag: str) -> tuple[str, cr_api.Response | None]:
        if abort.is_set():
            return tag, None
        resp = client.battlelog(tag)
        if resp.status in FATAL:
            abort.set()
        return tag, resp

    def role_of(tag: str) -> str:
        return "top" if tag in standing else ""

    ids = store.ids(store.meta)
    added = 0
    dropped: Counter = Counter()
    failures = logs_ok = 0
    with ThreadPoolExecutor(max(1, int(workers))) as pool:
        # One log at a time: its records are appended before the next is read, so a run over
        # 1,000 players holds one log in memory, not all of them.
        for tag, resp in pool.map(one, list(standing)):
            if resp is None:
                continue
            save_raw(store, f"meta_battlelog_{_tag_file(tag)}", resp, now, cfg.keep_raw_days)
            if resp.status in FATAL and fatal is None:
                fatal = resp
            if not resp.ok or not isinstance(resp.data, list):
                failures += 1
                if len(failed) < 20:
                    failed.append(_failure("a top player's battle log", resp))
                continue
            logs_ok += 1
            records = []
            for battle in resp.data:
                if not isinstance(battle, Mapping) or battle.get("type") not in TOP_TYPES:
                    dropped["not_path_of_legends"] += 1
                    continue
                record, why = normalise(battle, role_of=role_of, rule=cfg.form_rule,
                                        seen_at=iso(now), slim=True)
                if record is None:
                    dropped[why] += 1
                    continue
                record["kind"] = "top"
                for side in record["sides"]:
                    side["rating"], side["rank"] = standing.get(side["tag"], (None, None))
                records.append(record)
            added += len(store.merge(store.meta, records, ids))
    cutoff = now - dt.timedelta(weeks=cfg.meta_weeks)

    def keep(record: dict[str, Any]) -> dict[str, Any] | None:
        when = parse_battle_time(record.get("battle_time"))
        if when is None or when < cutoff:
            return None
        return refresh(record, rule=cfg.form_rule, kind="top")

    store.rewrite(store.meta, keep)
    result = {
        "started_at": iso(now), "ok": fatal is None and not failures,
        "fatal": fatal.status if fatal else None, "board": len(board), "board_path": source,
        "logs_ok": logs_ok, "failed": failed, "failed_count": failures, "added": added,
        "dropped": dict(sorted(dropped.items())), "held": store.count(store.meta),
        "weeks": cfg.meta_weeks, "calls": client.calls}
    if fatal is None and logs_ok >= max(1, len(board) // 2):
        state["meta_at"] = iso(now)
    state["meta"] = result
    store.write_state(state)
    return result


def meta_due(state: Mapping[str, Any], now: dt.datetime, days: float = META_DUE_DAYS) -> bool:
    last = parse_iso(state.get("meta_at"))
    return last is None or (now - last).total_seconds() >= days * 86400


# ---------------------------------------------------------------------------
# add: owner-entered decks
# ---------------------------------------------------------------------------

def canonical_deck(text: str) -> tuple[str, list[str]]:
    """(deck string, problems) of a comma-separated deck: RoyaleAPI keys or API names, each
    with an optional -ev1 / -evo / -hero."""
    tokens = [t.strip() for t in re.split(r"[,;\n]", str(text)) if t.strip()]
    pairs = [split_form(cr_api.slug(t)) for t in tokens]
    problems = []
    if len(pairs) != DECK_SIZE:
        problems.append(f"{len(pairs)} cards; a deck has {DECK_SIZE}")
    bases = [key for key, _ in pairs]
    twice = sorted({key for key in bases if bases.count(key) > 1})
    if twice:
        problems.append(f"twice: {', '.join(twice)}")
    if any(not key for key in bases):
        problems.append("an empty card name")
    evo = sum(1 for _, form in pairs if form == 1)
    hero = sum(1 for _, form in pairs if form == 2)
    if evo > 2 or hero > 2 or evo + hero > 3:
        problems.append(f"{evo} evolutions and {hero} heroes: a deck has three form slots "
                        "(slot 1 an evolution, slot 2 a hero, slot 3 either)")
    return ",".join(sorted(card_token(k, f) for k, f in pairs)), problems


def read_manual(store: Store) -> dict[str, Any]:
    got = read_json(store.manual, {})
    if not isinstance(got, dict) or not isinstance(got.get("decks"), list):
        got = {"schema": SCHEMA, "decks": []}
    return got


def add_deck(store: Store, deck: str, *, label: str = "", partner: str = "",
             weight: float = 1.0, note: str = "", now: dt.datetime | None = None
             ) -> tuple[dict[str, Any], bool]:
    """Adds (or updates) a manual deck; returns (entry, created)."""
    now = now or utcnow()
    store.ensure()
    manual = read_manual(store)
    ident = deck_id(deck)
    for entry in manual["decks"]:
        if entry.get("id") == ident:
            entry.update(label=label or entry.get("label", ""),
                         partner=partner or entry.get("partner"), weight=weight,
                         note=note or entry.get("note", ""), updated=iso(now))
            write_json(store.manual, manual)
            return entry, False
    entry = {"id": ident, "deck": deck, "label": label, "partner": partner or None,
             "weight": weight, "note": note, "added": iso(now)}
    manual["decks"].append(entry)
    write_json(store.manual, manual)
    return entry, True


def unknown_keys(deck: str) -> list[str]:
    """Base keys of the deck that tools/rs_cards.load() lacks ([] when the map cannot load)."""
    try:
        import rs_cards
        known = set(rs_cards.load())
    except Exception:
        return []
    return sorted({split_form(t)[0] for t in deck.split(",")} - known)


# ---------------------------------------------------------------------------
# probe: the unconfirmed battle-log fields, from the PC's real answers
# ---------------------------------------------------------------------------

def _raw_side(battle: Mapping[str, Any], tag: str) -> Mapping[str, Any] | None:
    for key in ("team", "opponent"):
        for side in battle.get(key) or []:
            if isinstance(side, Mapping) and clean_tag(side.get("tag")) == tag:
                return side
    return None


def _one_v_one(battle: Any) -> bool:
    return (isinstance(battle, Mapping) and isinstance(battle.get("team"), list)
            and isinstance(battle.get("opponent"), list) and len(battle["team"]) == 1
            and len(battle["opponent"]) == 1 and isinstance(battle["team"][0], Mapping)
            and isinstance(battle["opponent"][0], Mapping))


def _raw_id(battle: Mapping[str, Any]) -> str:
    return battle_id(str(battle.get("battleTime") or ""), clean_tag(battle["team"][0].get("tag")),
                     clean_tag(battle["opponent"][0].get("tag")))


def _foe(battle: Mapping[str, Any], tag: str) -> str:
    a, b = clean_tag(battle["team"][0].get("tag")), clean_tag(battle["opponent"][0].get("tag"))
    return b if a == tag else a if b == tag else ""


def _cards(side: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [c for c in side.get("cards") or [] if isinstance(c, Mapping)]


def _displayed(card: Mapping[str, Any]) -> int | None:
    """The level the game shows: level + (16 - maxLevel) (API_NOTES.md)."""
    level, top = _int(card.get("level")), _int(card.get("maxLevel"))
    return None if level is None or top is None else level + (16 - top)


def _friendly_type(kind: Any) -> bool:
    """A guess for the probe only (pull finds spars by tag): a type naming a friendly."""
    text = str(kind or "")
    return "friendly" in text.lower() or text == "clanMate"


def probe_report(tag: str, player: Any, log: Any, *, bot: str = "", bot_log: Any = None,
                 rule: str = "slot") -> dict[str, Any]:
    """What `tag`'s profile (/players/{tag}) and battle log, and the bot's log, say about the
    unconfirmed battle-log fields. Pure: no calls. `rule` is the config's current form rule."""
    tag = clean_tag(tag)
    bot = clean_tag(bot)
    player = player if isinstance(player, Mapping) else {}
    log = [b for b in (log if isinstance(log, list) else []) if isinstance(b, Mapping)]
    bot_log = [b for b in (bot_log if isinstance(bot_log, list) else [])
               if isinstance(b, Mapping)]
    collection = {str(c.get("name")): c for c in player.get("cards") or []
                  if isinstance(c, Mapping)}
    owned = {name: _bits(c.get("evolutionLevel")) for name, c in collection.items()}
    current = [str(c.get("name")) for c in player.get("currentDeck") or []
               if isinstance(c, Mapping)]
    mine = [(b, _raw_side(b, tag)) for b in log if _one_v_one(b)]
    mine = [(b, s) for b, s in mine if s is not None]

    # forms: does a card show a form it owns but cannot play in its slot?
    total = with_field = playable = mirror = hidden = odd = 0
    unknown_bits: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for battle, side in mine:
        cards = _cards(side)
        for card, slot in zip(cards, slots_for(len(cards))):
            total += 1
            with_field += "evolutionLevel" in card
            name = str(card.get("name"))
            seen = _bits(card.get("evolutionLevel"))
            if seen & ~(EVO_BIT | HERO_BIT):
                unknown_bits.add(name)
            own = owned.get(name, 0)
            if not own:
                continue
            mask = (EVO_BIT if slot in EVO_SLOTS else 0) | (HERO_BIT if slot in HERO_SLOTS else 0)
            playable += bool(own & mask)
            if not own & ~mask:
                continue
            what = ("mirror" if seen == own else "hidden" if seen == own & mask else "other")
            mirror += what == "mirror"
            hidden += what == "hidden"
            odd += what == "other"
            if len(evidence) < 12:
                evidence.append({"battle_time": battle.get("battleTime"), "card": name,
                                 "slot": slot, "owned": own, "log": seen, "reads": what})
    if with_field == 0 and playable:
        forms = "absent"
    elif mirror and not hidden:
        forms = "ownership"
    elif hidden and not mirror:
        forms = "played"
    elif mirror and hidden:
        forms = "mixed"
    else:
        forms = "undecided"

    # card order against currentDeck, and the 7-card sides
    same = differs = 0
    missing: list[dict[str, Any]] = []
    for battle, side in mine:
        names = [str(c.get("name")) for c in _cards(side)]
        if current and sorted(names) == sorted(current):
            same += names == current
            differs += names != current
        if (len(names) == DECK_SIZE - 1 and len(current) == DECK_SIZE
                and set(names) < set(current)):
            gone = sorted(set(current) - set(names))[0]
            missing.append({"battle_time": battle.get("battleTime"), "card": gone,
                            "current_deck_slot": current.index(gone) + 1})
    order = "differs" if differs else "same" if same else ("no_match" if current else "unknown")
    # every battle once: a spar is in both logs (1v1 battles by id; others as they come)
    unique: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for b in [*log, *bot_log]:
        if _one_v_one(b):
            if _raw_id(b) in seen_ids:
                continue
            seen_ids.add(_raw_id(b))
        unique.append(b)
    every = [s for b in unique if _one_v_one(b) for s in (b["team"][0], b["opponent"][0])]
    sides_7 = sum(1 for s in every if len(_cards(s)) == DECK_SIZE - 1)
    spars = [(b, s) for b, s in mine if bot and _foe(b, tag) == bot]
    hero_listed = [b.get("battleTime") for b, s in mine if len(_cards(s)) == DECK_SIZE
                   and owned.get(str(_cards(s)[MISSING_SLOT - 1].get("name")), 0) & HERO_BIT]
    if missing or any(len(_cards(s)) == DECK_SIZE - 1 for _, s in spars):
        hero = "missing"
    elif hero_listed:
        hero = "present"
    else:
        hero = "unknown"

    # battle types, the spars' types, log lengths
    def described(b: Mapping[str, Any]) -> str:
        mode = b.get("gameMode") if isinstance(b.get("gameMode"), Mapping) else {}
        return (f"type {b.get('type')!r}, deckSelection {b.get('deckSelection')!r}, "
                f"gameMode {mode.get('name')!r} ({mode.get('id')})")
    types = Counter(described(b) for b in unique)
    spar_types = Counter(described(b) for b, _ in spars)

    # levels in friendlies
    friendlies = [(b, s) for b, s in mine if (bot and _foe(b, tag) == bot)
                  or _friendly_type(b.get("type"))]
    shown = Counter()
    as_collection = differ_collection = 0
    for _, side in friendlies:
        for card in _cards(side):
            level = _displayed(card)
            if level is None:
                continue
            shown[level] += 1
            mine_level = _displayed(collection.get(str(card.get("name")), {}))
            if mine_level is not None:
                as_collection += mine_level == level
                differ_collection += mine_level != level
    if not shown:
        levels = "none"
    elif set(shown) == {11}:
        levels = "11"
    elif as_collection and not differ_collection:
        levels = "collection"
    else:
        levels = "mixed"

    # the same spar seen from the bot's side
    view = "no_spar"
    view_diffs: list[str] = []
    by_id = {_raw_id(b): b for b in bot_log if _one_v_one(b)}
    for battle, side in spars:
        twin = by_id.get(_raw_id(battle))
        other = _raw_side(twin, tag) if twin is not None else None
        if other is None:
            continue
        here = [(c.get("name"), c.get("evolutionLevel"), c.get("level")) for c in _cards(side)]
        there = [(c.get("name"), c.get("evolutionLevel"), c.get("level")) for c in _cards(other)]
        if here == there:
            view = "same" if view != "differs" else view
        else:
            view = "differs"
            view_diffs.append(str(battle.get("battleTime")))

    # the battle to show card by card: the newest spar, else friendly, else battle
    pick = (spars or friendlies or mine)
    latest = []
    if pick:
        battle, side = max(pick, key=lambda bs: str(bs[0].get("battleTime") or ""))
        cards = _cards(side)
        for card, slot in zip(cards, slots_for(len(cards))):
            bits = card.get("evolutionLevel")
            latest.append({
                "slot": slot, "name": card.get("name"), "evolution_level": bits,
                "owned": owned.get(str(card.get("name"))), "level": card.get("level"),
                "max_level": card.get("maxLevel"), "displayed": _displayed(card),
                "slot_rule": form_of(slot, bits, "slot"),
                "field_rule": form_of(slot, bits, "field")})
        latest_meta = {"battle_time": battle.get("battleTime"), "type": battle.get("type"),
                       "against_bot": bool(bot and _foe(battle, tag) == bot),
                       "cards": len(cards)}
    else:
        latest_meta = {}

    towers = Counter(str((s.get("supportCards") or [{}])[0].get("name"))
                     for s in every if s.get("supportCards"))
    keys = {"battle": sorted({k for b in unique for k in b}),
            "side": sorted({k for s in every for k in s}),
            "card": sorted({k for s in every for c in _cards(s) for k in c})}

    shows = "what the player owns" if forms == "ownership" else "the form played"
    if forms == "absent":
        recommend, why = "base", "the log carries no forms at all"
    elif forms in ("ownership", "played") and order == "differs":
        recommend = "base" if forms == "ownership" else "field"
        why = f"the log shows {shows}, and its card order is not the slot order"
    elif forms in ("ownership", "played"):
        recommend, why = "slot", f"the log shows {shows}; card order {order}"
    else:
        recommend, why = rule, ("not settled: play a friendly with a card that owns an evolution "
                                "outside slots 1 and 3 (or a hero outside 2 and 3)")

    answers = {
        "battlelog_forms": (f"{forms}: {mirror} cards show a form they own but cannot play in "
                            f"their slot, {hidden} show only the playable form, {odd} neither; "
                            f"{with_field} of {total} cards carry evolutionLevel"),
        "hero_slot_missing": (
            f"{hero}: {sides_7} of {len(every)} 1v1 sides in the logs have 7 cards"
            + (f"; {missing[0]['card']} (currentDeck slot {missing[0]['current_deck_slot']}) is "
               "the card left out" if missing else "")
            + f"; currentDeck has {len(current)} cards"),
        "battle_types": ("spars with the bot: " + ("; ".join(f"{k} x{n}" for k, n in
                                                          spar_types.most_common()) or "none")),
        "log_length": f"{len(log)} battles in the player's log, {len(bot_log)} in the bot's",
        "friendly_levels": (f"{levels}: displayed levels {dict(sorted(shown.items()))} in "
                            f"{len(friendlies)} friendlies"
                            + (f"; {as_collection} equal the collection's, "
                               f"{differ_collection} differ" if shown else "")),
        "card_order": f"{order}: {same} battles list currentDeck's cards in its order, "
                      f"{differs} in another order",
        "opponent_view": f"{view}" + (f": {', '.join(view_diffs)}" if view_diffs else ""),
    }
    return {
        "forms": forms, "mirror": mirror, "hidden": hidden, "other": odd,
        "with_field": with_field, "cards": total, "evidence": evidence,
        "unknown_bits": sorted(unknown_bits), "order": order, "hero_slot": hero,
        "missing": missing, "sides_7": sides_7, "sides": len(every),
        "current_deck": len(current), "types": dict(types.most_common()),
        "spar_types": dict(spar_types.most_common()), "spars": len(spars),
        "log_length": len(log), "bot_log_length": len(bot_log), "friendly_levels": levels,
        "levels_shown": {str(k): v for k, v in sorted(shown.items())},
        "opponent_view": view, "towers": dict(towers.most_common()), "keys": keys,
        "latest": latest, "latest_battle": latest_meta, "rule": rule,
        "recommend": recommend, "why": why, "answers": answers}


def print_probe(report: Mapping[str, Any]) -> None:
    answers = report["answers"]
    print(f"log       {answers['log_length']}")
    print(f"types     {answers['battle_types']}")
    for text, count in report["types"].items():
        print(f"          {text} x{count}")
    print(f"forms     {answers['battlelog_forms']}")
    for item in report["evidence"][:6]:
        print(f"          {item['card']} in slot {item['slot']}: owns {item['owned']}, the log "
              f"shows {item['log']} ({item['reads']})")
    if report["unknown_bits"]:
        print(f"          UNKNOWN evolutionLevel bits on: {', '.join(report['unknown_bits'])}")
    print(f"order     {answers['card_order']}")
    print(f"hero      {answers['hero_slot_missing']}")
    print(f"levels    {answers['friendly_levels']}")
    print(f"view      the player's cards seen from the bot's log: {answers['opponent_view']}")
    print(f"towers    {', '.join(f'{k} x{n}' for k, n in report['towers'].items()) or 'none'}")
    print(f"fields    side: {', '.join(report['keys']['side'])}")
    print(f"          card: {', '.join(report['keys']['card'])}")
    latest = report["latest_battle"]
    if latest:
        print(f"latest    {latest['battle_time']} ({latest['type']}, "
              f"{'against the bot' if latest['against_bot'] else 'not against the bot'}), "
              f"{latest['cards']} cards, slot by slot:")
        for row in report["latest"]:
            print(f"  {row['slot']}  {str(row['name']):<20} evolutionLevel "
                  f"{row['evolution_level']!s:<4} owned {row['owned']!s:<4} level "
                  f"{row['level']}/{row['max_level']} (shows {row['displayed']})  slot rule "
                  f"{row['slot_rule']}, field rule {row['field_rule']}")
        print("          check this against what was played: an evolution in slot 1, a hero in "
              "slot 2")
    change = "" if report["recommend"] == report["rule"] else \
        f" (config has {report['rule']}: set \"form_rule\": \"{report['recommend']}\")"
    print(f"verdict   form_rule {report['recommend']}{change}: {report['why']}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def summary(store: Store, bot: str = "") -> dict[str, Any]:
    """What the store holds, for the status command and for tools/game_update.py (U04)."""
    state = store.read_state()
    kinds: Counter = Counter()
    results: Counter = Counter()
    bad = battles = incomplete = 0
    for record in scan_jsonl(store.battles):
        if record is None:
            bad += 1
            continue
        battles += 1
        kinds[record.get("kind")] += 1
        if bot and record.get("kind") == "spar":
            results[result_of(record, bot)] += 1
        incomplete += sum(1 for s in record.get("sides") or [] if not s.get("complete"))
    return {
        "folder": str(store.folder), "pulled_at": state.get("pulled_at"),
        "last_attempt_at": state.get("last_attempt_at"),
        "last_ok": (state.get("pull") or {}).get("ok"), "battles": battles,
        "unreadable_lines": bad, "by_kind": dict(sorted(kinds.items())),
        "spars": {"W": results["W"], "D": results["D"], "L": results["L"]},
        "incomplete_decks": incomplete, "meta_at": state.get("meta_at"),
        "meta_battles": store.count(store.meta), "cards": state.get("cards"),
        "manual": len(read_manual(store)["decks"])}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--folder", type=Path, default=None,
                        help=f"the store (default <data>/{FOLDER}, <data> from "
                             "src/hastycr/machine.py)")
    api = argparse.ArgumentParser(add_help=False)
    api.add_argument("--base", default=None,
                     help=f"official (default), proxy ({cr_api.PROXY_BASE}, with "
                          f"{cr_api.PROXY_IP} on the key) or an https URL ending in /v1")
    api.add_argument("--transport", choices=("urllib", "curl"), default=None,
                     help="default: the config's (urllib)")
    api.add_argument("--token-file", type=Path, default=None,
                     help="default .secrets/supercell_token")
    api.add_argument("--rate", type=float, default=None,
                     help=f"requests per second (default the config's, {DEFAULT_RATE:g}; at "
                          f"most {MAX_RATE:g})")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init", parents=[common, api], help="write or change the config")
    init.add_argument("--clan", help="the clan's tag")
    init.add_argument("--owner", help="the owner's main account")
    init.add_argument("--bot", help="Yodle2, the account the bot plays")
    init.add_argument("--friend", action="append", default=[],
                      help="a sparring partner outside the clan (repeatable)")
    init.add_argument("--drop-friend", action="append", default=[], help="remove a friend")
    init.add_argument("--check", action="store_true",
                      help="look every tag up through the API (one call each)")
    sub.add_parser("pull", parents=[common, api], help="the daily pull")
    met = sub.add_parser("meta", parents=[common, api],
                         help="the top Path of Legends players' battles")
    met.add_argument("--top", type=int, default=None, help=f"players (default {META_TOP})")
    met.add_argument("--workers", type=int, default=META_WORKERS)
    met.add_argument("--if-due", action="store_true",
                     help=f"only when the last complete run is {META_DUE_DAYS:g}+ days old")
    add = sub.add_parser("add", parents=[common], help="an owner-entered deck")
    add.add_argument("--deck", help="8 cards, comma-separated, -ev1 / -hero for forms")
    add.add_argument("--label", default="")
    add.add_argument("--partner", default="", help="the tag of who plays it")
    add.add_argument("--weight", type=float, default=1.0)
    add.add_argument("--note", default="")
    add.add_argument("--list", action="store_true", help="show the manual decks")
    add.add_argument("--remove", default=None, metavar="ID", help="drop a manual deck")
    probe = sub.add_parser("probe", parents=[common, api],
                           help="report the unconfirmed battle-log fields")
    probe.add_argument("--player", default=None, help="default the config's owner")
    probe.add_argument("--bot", default=None, help="default the config's bot")
    status = sub.add_parser("status", parents=[common], help="what the store holds")
    status.add_argument("--json", action="store_true")
    return ap


def _print_pull(result: Mapping[str, Any], store: Store) -> None:
    kinds = ", ".join(f"{k} {n}" for k, n in result["added_by_kind"].items()) or "none"
    dropped = ", ".join(f"{k} {n}" for k, n in result["dropped"].items()) or "none"
    print(f"pull      {result['logs_ok']} of {result['logs']} logs, {result['members']} members; "
          f"{result['added']} new battles ({kinds}); dropped: {dropped}")
    sp = result["spars"]
    print(f"spars     {sum(sp.values())} new: Yodle2 {sp['W']}-{sp['D']}-{sp['L']} (W-D-L)")
    if result["bot_unknown_opponents"]:
        print(f"unknown   Yodle2 played {', '.join(result['bot_unknown_opponents'])}: not in the "
              "roster or friends; add sparring partners with init --friend")
    if result["gaps"]:
        print(f"gaps      {len(result['gaps'])} logs start after the newest battle stored last "
              "time: battles in between may be lost (pull at least daily)")
    cards = result.get("cards")
    if cards:
        state = "CHANGED (previous kept as cards_api.prev.json)" if cards["changed"] else \
            "unchanged"
        print(f"cards     {cards['items']} cards, {cards['support_items']} support items, {state}")
    if result["recomputed"]:
        print(f"refresh   {result['recomputed']} stored battles re-derived (roles, kinds, forms)")
    for item in result["failed"]:
        print(f"FAILED    {item['what']}: {item['hint']}")
    print(f"store     {store.battles}")


def _cmd_init(args, store: Store, transport, rate) -> int:
    existing = read_json(store.config) if store.config.is_file() else None
    data = dict(existing) if isinstance(existing, dict) else template()
    data["_comment"] = ("Local settings for tools/opponents.py (gitignored with the folder). "
                        "Change them with tools/opponents.py init.")
    for name in ("clan", "owner", "bot"):
        value = getattr(args, name)
        if value:
            got = clean_tag(value)
            if not got:
                print(f"opponents: --{name} {value!r} is not a tag", file=sys.stderr)
                return EXIT_SETTING
            data[name] = got
    friends = [clean_tag(t) or t for t in data.get("friends") or []]
    for value in args.friend:
        got = clean_tag(value)
        if not got:
            print(f"opponents: --friend {value!r} is not a tag", file=sys.stderr)
            return EXIT_SETTING
        friends.append(got)
    drop = {clean_tag(t) for t in args.drop_friend}
    data["friends"] = [t for t in dict.fromkeys(friends) if t not in drop]
    try:
        cfg = parse_config(data, "config")
    except ConfigError as exc:
        print(f"opponents: {exc}", file=sys.stderr)
        return EXIT_SETTING
    store.ensure()
    write_json(store.config, {"_comment": data["_comment"], **cfg.to_json()})
    for line in cfg.warnings:
        print(f"WARN      {line}")
    print(f"config    {store.config}: clan, owner, bot and {len(cfg.friends)} friends")
    if not args.check:
        return EXIT_OK
    client = build_client(cfg, args, store, transport=transport, rate=rate)
    code = EXIT_OK
    lookups = [("clan", client.clan, cfg.clan), ("owner", client.player, cfg.owner),
               ("bot", client.player, cfg.bot),
               *((f"friend {t}", client.player, t) for t in cfg.friends)]
    for what, call, tag in lookups:
        resp = call(tag)
        if resp.ok and isinstance(resp.data, dict):
            print(f"check     {what}: {resp.data.get('name')!r}")
        else:
            print(f"check     {what}: {resp.hint()}")
            code = EXIT_API
            if resp.status in FATAL:
                break
    return code


def _cmd_add(args, store: Store) -> int:
    if args.list:
        for entry in read_manual(store)["decks"]:
            print(f"{entry['id']}  w {entry.get('weight', 1):g}  {entry['deck']}  "
                  f"{entry.get('label') or ''}")
        return EXIT_OK
    if args.remove:
        manual = read_manual(store)
        kept = [e for e in manual["decks"] if e.get("id") != args.remove]
        if len(kept) == len(manual["decks"]):
            print(f"opponents: no manual deck {args.remove}", file=sys.stderr)
            return EXIT_SETTING
        manual["decks"] = kept
        write_json(store.manual, manual)
        print(f"removed   {args.remove}")
        return EXIT_OK
    if not args.deck:
        print("opponents: add needs --deck (or --list, --remove)", file=sys.stderr)
        return EXIT_SETTING
    deck, problems = canonical_deck(args.deck)
    partner = clean_tag(args.partner) if args.partner else ""
    if args.partner and not partner:
        problems.append(f"--partner {args.partner!r} is not a tag")
    if not 0 < args.weight <= 100:
        problems.append(f"--weight {args.weight:g}: above 0, at most 100")
    if problems:
        print(f"opponents: {deck}: " + "; ".join(problems), file=sys.stderr)
        return EXIT_SETTING
    entry, created = add_deck(store, deck, label=args.label, partner=partner,
                              weight=args.weight, note=args.note)
    print(f"{'added' if created else 'updated':<9} {entry['id']}  {deck}")
    missing = unknown_keys(deck)
    if missing:
        print(f"WARN      not in data/cards: {', '.join(missing)} (blocked until mapped)")
    return EXIT_OK


def _cmd_probe(args, store: Store, transport, rate, now) -> int:
    cfg = None
    if store.config.is_file():
        cfg = store.load_config()
    tag = clean_tag(args.player) if args.player else (cfg.owner if cfg else "")
    bot = clean_tag(args.bot) if args.bot else (cfg.bot if cfg else "")
    if not tag:
        print("opponents: probe needs --player, or a config (init)", file=sys.stderr)
        return EXIT_SETTING
    client = build_client(cfg, args, store, transport=transport, rate=rate)
    now = now or utcnow()
    store.ensure()
    keep = cfg.keep_raw_days if cfg else KEEP_RAW_DAYS
    answers: dict[str, Any] = {}
    for what, call in (("player", lambda: client.player(tag)),
                       ("log", lambda: client.battlelog(tag)),
                       ("bot_log", (lambda: client.battlelog(bot)) if bot and bot != tag
                        else None)):
        if call is None:
            continue
        resp = call()
        save_raw(store, f"probe_{what}_{_tag_file(bot if what == 'bot_log' else tag)}", resp,
                 now, keep)
        if not resp.ok:
            print(f"GET {resp.path}: {resp.hint()}")
            return EXIT_API
        answers[what] = resp.data
    report = probe_report(tag, answers.get("player"), answers.get("log"), bot=bot,
                          bot_log=answers.get("bot_log"),
                          rule=cfg.form_rule if cfg else "slot")
    write_json(store.probe, {"probed_at": iso(now), "player": tag, "bot": bot, **report})
    print_probe(report)
    print(f"saved     {store.probe}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None, *, transport: cr_api.Transport | None = None,
         rate: cr_api.Rate | None = None, now: dt.datetime | None = None) -> int:
    """The CLI. `transport`, `rate` and `now` are for tests."""
    args = _parser().parse_args(argv)
    store = Store(args.folder if args.folder is not None else data_root() / FOLDER)
    try:
        if args.cmd == "init":
            return _cmd_init(args, store, transport, rate)
        if args.cmd == "add":
            return _cmd_add(args, store)
        if args.cmd == "status":
            cfg = store.load_config() if store.config.is_file() else None
            got = summary(store, cfg.bot if cfg else "")
            if args.json:
                print(json.dumps(got, indent=1))
            else:
                print(f"store     {got['folder']}: {got['battles']} battles "
                      f"{got['by_kind']}, {got['meta_battles']} meta, {got['manual']} manual")
                print(f"pulled    {got['pulled_at'] or 'never'} (last attempt "
                      f"{got['last_attempt_at'] or 'never'}, ok {got['last_ok']})")
                sp = got["spars"]
                print(f"spars     Yodle2 {sp['W']}-{sp['D']}-{sp['L']} (W-D-L); "
                      f"{got['incomplete_decks']} sides with an incomplete deck")
                print(f"meta      {got['meta_at'] or 'never'}")
            return EXIT_OK
        if args.cmd == "probe":
            return _cmd_probe(args, store, transport, rate, now)
        cfg = store.load_config()
        for line in cfg.warnings:
            print(f"WARN      {line}")
        if args.cmd == "meta" and args.if_due and not meta_due(store.read_state(),
                                                               now or utcnow()):
            print(f"meta      not due (last complete run {store.read_state().get('meta_at')})")
            return EXIT_OK
        client = build_client(cfg, args, store, transport=transport, rate=rate)
        store.ensure()
        with Lock(store.lock):
            if args.cmd == "pull":
                result = pull(cfg, client, store, now=now)
                _print_pull(result, store)
                return EXIT_OK if result["ok"] else EXIT_API
            result = meta(cfg, client, store, top=args.top, workers=args.workers, now=now)
            print(f"meta      board {result['board']} players ({result['board_path'] or '-'})"
                  f"; {result['logs_ok']} logs, {result['failed_count']} failed; "
                  f"{result['added']} new battles; {result['held']} held "
                  f"({cfg.meta_weeks} weeks)")
            for item in result["failed"][:5]:
                print(f"FAILED    {item['what']}: {item['hint']}")
            return EXIT_OK if result["ok"] else EXIT_API
    except (ConfigError, cr_api.TokenError) as exc:
        print(f"opponents: {exc}", file=sys.stderr)
        return EXIT_SETTING
    except Busy as exc:
        print(f"opponents: {exc}", file=sys.stderr)
        return EXIT_API


if __name__ == "__main__":
    raise SystemExit(main())
