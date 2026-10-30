# tests/pipeline/fixtures/cr_api

Hand-written answers of the official Clash Royale API, for `tests/pipeline/test_cr_api.py`
(U02 of `docs/SEASON_UPDATES.md`). Nothing here was pulled from the API.

**Every tag, name, clan, time and number is invented.** The tags use only the letters real tags
use (0289PYLQGRJCUV), so the client's tag rules accept them, but they belong to no one. The
IP address in `error_403_invalid_ip.json` is from the documentation range 198.51.100.0/24.

The shapes follow the sources `tools/cr_api.py` cites:

- the Swagger 2.0 spec (https://gist.github.com/loganlinn/d813ba0f9ccb47f34462ec8abeab8e14):
  list answers wrap results in `items` with a `paging` marker; a battle log is an array of
  battles with `type`, `battleTime`, `arena`, `gameMode`, `deckSelection`, `team` and
  `opponent`; a side has `tag`, `name`, `crowns`, `clan` and `cards`; a card has `name`,
  `level`, `maxLevel` and `iconUrls.medium`; an error body has `reason` and `message`;
  the 429 and 503 messages are the spec's descriptions;
- API_NOTES.md (https://raw.githubusercontent.com/samaksh-bajaj/cr-deck-builder/main/API_NOTES.md):
  ranking entries carry `tag`, `name`, `expLevel`, `eloRating`, `rank` and `clan`;
  `evolutionLevel` and `maxEvolutionLevel` are bitmasks (1 evolution, 2 hero); the icon keys
  `medium`, `evolutionMedium` and `heroMedium`;
- `tools/ladder_pull.py`: `accessDenied.invalidIp`, the Path of Legends board path, and the
  `paging.cursors.after` cursor.

Unconfirmed, and asserted nowhere as fact (see `cr_api.UNCONFIRMED`): the `id`, `elixirCost`,
`rarity` and `supportItems`/`supportCards` fields; the `requestThrottled`, `notFound` and
`inMaintenance` reason strings and the wording of the 403 message; the battle `type` of a
friendly and of a 2v2 (`friendly`, `clanMate2v2`); the `deckSelection` and `gameMode` values;
what `evolutionLevel` on a battle-log card means. U03's `probe` checks them on the PC.

| File | What |
|---|---|
| `cards.json` | `/v1/cards`: 8 cards (Minion Giant has no key in `data/cards`) and 1 support item |
| `pol_page1.json`, `pol_page2.json` | the Path of Legends board in two pages, players A and B, then C |
| `battlelog_a.json` | player A: a ranked win over B, a friendly, a ranked loss to D (not on the board) |
| `battlelog_b.json` | player B: the same battle with A, seen from B's side, and a 2v2 |
| `error_*.json` | error bodies: 403 (wrong IP, and plain), 404, 429, 503 |
