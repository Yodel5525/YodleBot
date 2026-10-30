"""What top ladder is actually playing, from Supercell's own API.

Every policy this project has trained comes out strong against whatever it was
selected on and weaker against the rest. The last run held out at 93.5% against
the rule engine and 78.0% against meta decks, on a training diet that is half
rule engine and a hand-picked list of meta decks. This is where a real
opponent pool comes from: the Path of Legends leaderboard, then each of those
players' recent ranked battles, then the deck distribution that falls out.

Why this and not `tools/royale_scrape.py`: RoyaleAPI's replay endpoint is
blocked (see that module), its crawl needs a person to clear a checkbox, and
one sitting yielded 1,467 battles in twenty minutes. This is sanctioned, needs
no browser, and returns roughly thirty battles for every player on the
leaderboard. What it cannot give is the card-by-card timeline - the official
API exposes decks and results, not replays.

    python tools/ladder_pull.py --top 1000

The token is read from `.secrets/supercell_token`, which is gitignored. Keys
are bound to a single public IP, so a changed address fails every request with
`accessDenied.invalidIp` and the fix is to edit the key rather than the code.

The HTTP side is `tools/cr_api.py` (U02 of docs/SEASON_UPDATES.md): the shared
rate, the 429 back-off (a throttled battle log used to be dropped without a
word), the custom User-Agent and `--base proxy` for RoyaleAPI's proxy. It
answers with HTTP codes; this tool stops on a 403, as it always has, and also
on a 429 the back-off could not clear and on a 503 (maintenance): cr_api.STOP.
A stop in one worker reaches the others, so the rest of the board is not walked
against a refusing API. Courtesy (docs/SEASON_UPDATES.md 7.7): the rate is
cr_api.DEFAULT_RATE (2 per second) unless --rate says otherwise, never above
cr_api.MAX_RATE (5), shared by 2 workers. The default transport stays curl over
IPv4. The two output files are unchanged: `ladder_battles.csv` and
`ladder_meta.json` under `--out`, both gitignored because they carry player
tags.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
import cr_api  # noqa: E402

TOKEN_FILE = cr_api.TOKEN_FILE
BASE = cr_api.OFFICIAL_BASE

# Ranked ladder only. A battlelog also carries challenges, 2v2 and friendlies,
# and none of those are the metagame this bot plays in.
RANKED_TYPES = {"pathOfLegend", "pathOfLegendMatch"}

#: The shared ceiling on requests per second, now in tools/cr_api.py.
Rate = cr_api.Rate

FORBIDDEN = ("403 from the API. The key is bound to one public IP and this "
             "machine's has probably changed; the response body names the "
             "address it saw. Edit the key at developer.clashroyale.com "
             "rather than making a new one.")
THROTTLED = ("429 from the API after the client's back-off: the key is being "
             "throttled. The pull stops rather than keep asking; run it again "
             "later, at a lower --rate.")
MAINTENANCE = ("503 from the API: it is in maintenance (common during a game "
               "update). The pull stops; run it again later.")
WORKERS = 2


def stop_message(resp: cr_api.Response) -> str:
    """Why the pull stops on `resp` (a status in cr_api.STOP)."""
    said = (f" The API said: {resp.reason}: {resp.message}"
            if resp.reason or resp.message else "")
    head = {403: FORBIDDEN, 429: THROTTLED}.get(resp.status, MAINTENANCE)
    return head + said


def stop_if_fatal(resp: cr_api.Response | None) -> None:
    """A 403, a 429 the back-off could not clear or a 503 fails every later
    call too, so the pull stops (a 403 always did)."""
    if resp is not None and resp.status in cr_api.STOP:
        raise SystemExit(stop_message(resp))


def leaderboard(client: cr_api.Client, want: int) -> list[dict]:
    """The Path of Legends global top, paged until `want` players."""
    players, last = client.paged("locations/global/pathoflegend/players", want=want)
    stop_if_fatal(last)
    return players


def battles_for(client: cr_api.Client, tag: str) -> tuple[list[dict], cr_api.Response | None]:
    """A player's battle log and the response (None for a value that is no tag,
    which gives no log, as a 404 would)."""
    try:
        path = f"players/{cr_api.tag_path(tag)}/battlelog"
    except ValueError:
        return [], None
    resp = client.get(path)
    data = resp.data if resp.ok else None
    return (data if isinstance(data, list) else []), resp


def deck_of(side: dict) -> str:
    """A deck as a stable, comparable string: card names, sorted."""
    return ",".join(sorted(c["name"] for c in side.get("cards", [])))


def rows_from(battle: dict, elo: dict) -> list[dict]:
    """One row per battle. Both sides are recorded, so either can be the meta.

    A battle between two ranked players appears in both of their logs, so the
    caller dedupes on the key returned here.
    """
    if battle.get("type") not in RANKED_TYPES:
        return []
    team, opp = battle.get("team") or [], battle.get("opponent") or []
    if len(team) != 1 or len(opp) != 1:
        return []                       # 2v2, not this metagame
    t, o = team[0], opp[0]
    return [{
        "battle_time": battle.get("battleTime", ""),
        "player_tag": t.get("tag", ""),
        "opponent_tag": o.get("tag", ""),
        "player_deck": deck_of(t),
        "opponent_deck": deck_of(o),
        "player_crowns": t.get("crowns", 0),
        "opponent_crowns": o.get("crowns", 0),
        "player_elo": elo.get(t.get("tag", ""), ""),
        "opponent_elo": elo.get(o.get("tag", ""), ""),
    }]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=1000,
                    help="how many Path of Legends players to walk")
    ap.add_argument("--rate", type=float, default=cr_api.DEFAULT_RATE,
                    help=f"requests per second ceiling, kept deliberately low "
                         f"(default {cr_api.DEFAULT_RATE:g}, at most "
                         f"{cr_api.MAX_RATE:g})")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"threads sharing that rate (default {WORKERS})")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ladder")
    ap.add_argument("--base", default=None,
                    help=f"API base: official (default), proxy ({cr_api.PROXY_BASE}, "
                         f"with {cr_api.PROXY_IP} on the key) or an https URL ending "
                         f"in /v1; {cr_api.BASE_ENV} sets the default")
    ap.add_argument("--transport", choices=("curl", "urllib"), default="curl",
                    help="curl over IPv4, as before (default), or Python's urllib, "
                         "also over IPv4")
    args = ap.parse_args(argv)
    if not 0.1 <= args.rate <= cr_api.MAX_RATE:
        ap.error(f"--rate {args.rate:g}: between 0.1 and {cr_api.MAX_RATE:g} "
                 "requests per second (courtesy, docs/SEASON_UPDATES.md 7.7)")
    if not 1 <= args.workers <= 8:
        ap.error(f"--workers {args.workers}: from 1 to 8")

    try:
        token = cr_api.read_token(TOKEN_FILE)
    except cr_api.TokenError as exc:
        raise SystemExit(str(exc)) from None
    scratch = args.out / ".scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        client = cr_api.Client(
            token, base=args.base, rate=Rate(args.rate),
            transport=cr_api.make_transport(args.transport, scratch=scratch),
            log=lambda line: print(f"  {line}", flush=True))
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    started = time.time()
    players = leaderboard(client, args.top)
    if not players:
        raise SystemExit("the leaderboard came back empty")
    elo = {p["tag"]: p.get("eloRating", "") for p in players}
    print(f"{len(players):,} players, elo {players[0].get('eloRating')} down to "
          f"{players[-1].get('eloRating')}")

    rows: dict = {}
    done = [0]
    stop = threading.Event()
    fatal: list[cr_api.Response] = []

    def walk(player):
        # A stop seen by any worker ends the walk for all of them: the queued
        # players are skipped, not asked against a refusing API.
        if stop.is_set():
            return
        battles, resp = battles_for(client, player["tag"])
        if resp is not None and resp.status in cr_api.STOP:
            fatal.append(resp)
            stop.set()
            return
        for battle in battles:
            for row in rows_from(battle, elo):
                # The same battle is in both players' logs; key it so it lands
                # once regardless of whose log it came from.
                key = (row["battle_time"],
                       *sorted((row["player_tag"], row["opponent_tag"])))
                rows.setdefault(key, row)
        done[0] += 1
        if done[0] % 100 == 0:
            print(f"  {done[0]:,}/{len(players):,} players, "
                  f"{len(rows):,} battles", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(walk, players))
    if fatal:
        raise SystemExit(stop_message(fatal[0]))
    if not rows:
        raise SystemExit("no ranked battles came back")

    args.out.mkdir(parents=True, exist_ok=True)
    battles_csv = args.out / "ladder_battles.csv"
    fields = list(next(iter(rows.values())))
    with battles_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows.values())

    # The metagame: how often each deck appears on either side.
    seen = collections.Counter()
    wins = collections.Counter()
    for row in rows.values():
        for me, them, mine, theirs in (
                ("player_deck", "opponent_deck", "player_crowns", "opponent_crowns"),
                ("opponent_deck", "player_deck", "opponent_crowns", "player_crowns")):
            seen[row[me]] += 1
            if row[mine] > row[theirs]:
                wins[row[me]] += 1
    meta = [{"deck": d, "seen": n, "wins": wins[d],
             "win_rate": round(wins[d] / n, 3), "share": round(n / sum(seen.values()), 5)}
            for d, n in seen.most_common()]
    (args.out / "ladder_meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8")

    elapsed = time.time() - started
    print(f"\n{len(rows):,} ranked battles from {len(players):,} players "
          f"in {elapsed / 60:.1f} min")
    print(f"{len(meta):,} distinct decks")
    print(f"  {battles_csv}")
    print(f"  {args.out / 'ladder_meta.json'}")
    print("\nmost played decks at top ladder:")
    for entry in meta[:10]:
        print(f"  {entry['share']:>6.2%}  {entry['win_rate']:>5.0%}  "
              f"{entry['deck'][:78]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
