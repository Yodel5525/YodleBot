# tests/pipeline/fixtures/opponents

Hand-made answers of the official Clash Royale API for `tests/pipeline/test_opponents.py` (U03 of
`docs/SEASON_UPDATES.md`). Nothing here was pulled from the API. The meta tests also use the U02
board and battle logs in `../cr_api/`, and the pull tests use `../cr_api/cards.json`.

**Every tag, name, clan, time and number is invented.** The tags use only the characters real tags
use (0289PYLQGRJCUV), so the tag rules accept them, and every one starts with `0`, which no real tag
does: a tag is an account id written in base 14 with those characters, most significant digit first,
and the encoder stops at the last non-zero digit, so a tag never starts with the zero digit (the
`tagfromid` loop in https://gist.github.com/samyk/66148f77ef073384112123be8944e6cb). These tags can
name no real player or clan. The shapes follow the U02 fixtures and the sources `tools/opponents.py`
cites (the Swagger spec, API_NOTES.md).

What the fixtures assume, and the probe checks on the PC (`cr_api.UNCONFIRMED`,
`opponents.UNCONFIRMED`):

- a battle-log card's `evolutionLevel` is what the player **owns** (bits 1 evolution, 2 hero), as
  API_NOTES.md found for `currentDeck`. So the owner's Ice Spirit, whose evolution the owner owns,
  shows 1 from slot 6, and Musketeer shows 3 from slot 2.
- cards are listed in deck-slot order, and a 7-card side lacks its slot-2 card;
- friendlies show every card at level 11 (`level + 16 - maxLevel`);
- the types `friendly`, `clanMate` and `clanMate2v2`, the deck selection `collection`, the game
  modes and their ids, and the tower fields (`kingTowerHitPoints`, `princessTowersHitPoints`,
  `supportCards`, `elixirLeaked`).

## Who is who

| Tag | Who | In |
|---|---|---|
| `#09QJ0RV2` | the clan | config |
| `#0PL0QY8` | the owner (main account; coLeader) | config, roster |
| `#0Y0GRC2` | Yodle2, the bot (also a clan member) | config, roster |
| `#02GJUV9` | M1, leader | roster |
| `#08RPYCQ` | M2, elder | roster |
| `#0LUV2R8` | a friend outside the clan | config friends |
| `#09YRJQP` | a stranger who played Yodle2 | nobody's list |
| `#0CJ8P0L`, `#0U2CRQ9`, `#0RJ9QC0`, `#0G2PV8L` | Path of Legends and 2v2 opponents | nobody's list |

## Decks, in slot order

- **Owner, Hog 2.6**: Cannon (bits 1), Musketeer (bits 3), Skeletons (1), Hog Rider, Ice Golem,
  Ice Spirit (1), Fireball, The Log. Under the slot rule:
  `cannon-ev1,fireball,hog-rider,ice-golem,ice-spirit,musketeer-hero,skeletons-ev1,the-log`
  (the string `data/ladder/battles_2p6.csv` uses). Under the field rule, Musketeer is ambiguous and
  Ice Spirit an evolution.
- **Yodle2, Hog 2.6**: Cannon (1), Musketeer (2, max 3), Skeletons (1), the rest base: the same deck
  string.
- **M1, 7 cards** (the slot-2 hero, a Giant, left out): Knight (1, max 3), Bats (1), Fireball, Zap,
  Goblin Gang, Arrows, Mini P.E.K.K.A (2). The slots are 1, 3, 4, 5, 6, 7, 8, so Bats (slot 3) is
  an evolution and Mini P.E.K.K.A (slot 8) is base:
  `arrows,bats-ev1,fireball,goblin-gang,knight-ev1,mini-pekka,zap`.
- **Friend**: Mirror (1), Giant (2), Knight (3), Fisherman, Electro Spirit, Hunter, Earthquake,
  Royal Giant, tower Dagger Duchess:
  `earthquake,electro-spirit,fisherman,giant-hero,hunter,knight,mirror-ev1,royal-giant`, with
  Knight ambiguous (slot 3, both bits). Mirror keeps its evolution: the engine lacking it is U16's
  business, not the store's.

## Battles

| Time (battleTime) | Type | Sides (crowns) | In the logs of | Stored as |
|---|---|---|---|---|
| 20261030T183000.000Z | friendly | Yodle2 1, owner 2 | bot, owner | spar, Yodle2 L |
| 20261030T181500.000Z | clanMate | Yodle2 2, M1 1 | bot, M1 | spar, Yodle2 W |
| 20261030T170000.000Z | pathOfLegend | owner 1, `#0CJ8P0L` 2 | owner | clan |
| 20261030T160000.000Z | clanMate2v2 | owner + M2 against two | owner | dropped (2v2) |
| 20261030T150000.000Z | pathOfLegend | M1 2 (7 cards), `#0U2CRQ9` 0 | M1 | clan |
| 20261030T120000.000Z | friendly | Yodle2 1, stranger 1 | bot | bot_unknown |
| 20261029T210000.000Z | friendly | Yodle2 3, friend 0 | bot, friend | spar, Yodle2 W |

So one pull stores 6 battles (spar 3, clan 2, bot_unknown 1) and drops one 2v2. Yodle2's new spars
go 2-0-1. M2's log is empty.

| File | What |
|---|---|
| `members.json` | `/clans/%2309QJ0RV2/members`: M1, the owner, M2, Yodle2 |
| `battlelog_bot.json` | Yodle2's log (4 battles, newest first) |
| `battlelog_owner.json` | the owner's log (3 battles, one of them the 2v2) |
| `battlelog_m1.json` | M1's log (2) |
| `battlelog_m2.json` | M2's log: empty |
| `battlelog_friend.json` | the friend's log (1) |
| `player_owner.json` | `/players/%230PL0QY8`: the collection (owned bits) and `currentDeck` in slot order |

The probe on these files reads: forms `ownership` (4 cards show a form they own but cannot play in
their slot, 0 show only the playable one), card order `same`, hero slot `present`, 1 of 10 sides
with 7 cards, friendly levels `11`, the spar the `same` from the bot's side, form rule `slot`.

The files were written by a short generator from the specs above, then checked by hand against
this table.
