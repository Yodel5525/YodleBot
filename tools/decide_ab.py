r"""The decision rules' A/B, read from a league log: is the bot stronger with them on?

    .venvs\royalegym\Scripts\python.exe tools\decide_ab.py --log runs\league\decide_ab.jsonl ^
        --champion v2_hog --candidate v2_hog_rules [--candidate v2_hog_floor ...] [--games 1000]

scripts\decide_ab.ps1 plays the games first (tools\rs_league_eval.py run): each candidate is the
champion's own checkpoint with decision rules on (tools/rs_decide.py DECISION RULES: the wait
floor, the stray cheap-troop guard; the league's player keys wait_floor= and stray_guard=), the
champion the same checkpoint with them off, 1,000 games on both seats and with the decks
exchanged, at a 24-tick delay with the noise prior on every seat. Then it runs this.

THE VERDICT, per candidate, on its games against the champion in one protocol of the log (by
default the protocol of the last run whose players hold both names; --protocol DIGEST):
  the candidate's score (a win 1, a draw 0.5) with two 95% intervals: Wilson over the games, as
  the promotion gate counts them, and the bootstrap over pairings (rs_league_eval's matchup
  table: the 4 games of a pairing share decks and seeds, so they are not independent);
  STRONGER when both intervals lie above 50%, WEAKER when both lie below, NO DIFFERENCE
  otherwise; INCOMPLETE as well when the log holds fewer than --games of them (run the same
  scripts\decide_ab.ps1 command again: the league plays only the games its log lacks).
Also, per game and for both players: plays, elixir leak, and what the rules did (the seat info
"rules" of the game records: floor waits, stray moves, stray waits; tools/rs_match.py
rules_summary); and each player's rules and checkpoint digest from the run header, with a
warning when the two checkpoints differ (then the A/B is not of the rules alone).

Exit codes: 0 every verdict complete; 1 an error (no log, no games of a candidate); 2 a verdict
on fewer games than --games. Reads the log only; plays nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import rs_league_eval as LE  # noqa: E402  (no engine: the log and the statistics)

STRONGER, NO_DIFFERENCE, WEAKER = "stronger", "no difference", "weaker"
DEFAULT_GAMES = 1000
RULE_KEYS = ("wait_floor", "stray_guard")
#: What each verdict means for the live runner (printed under it).
NEXT = {STRONGER: "the rules make the bot stronger in the simulator: they can be switched on "
                  "for live friendlies (the live runner's switch comes in a later change)",
        NO_DIFFERENCE: "no measurable effect at this many games: keep them off, or play more "
                       "games (-Games) for a finer interval",
        WEAKER: "the rules make the bot weaker in the simulator: keep them off"}


def verdict_of(lo: float, hi: float, boot_lo: float, boot_hi: float) -> str:
    """STRONGER when both 95% intervals lie above 0.5, WEAKER when both lie below, else NO
    DIFFERENCE."""
    if lo > 0.5 and boot_lo > 0.5:
        return STRONGER
    if hi < 0.5 and boot_hi < 0.5:
        return WEAKER
    return NO_DIFFERENCE


def why_of(v: Mapping[str, Any]) -> str:
    """Why verdict `v` is what it is, naming the interval that holds 50% (both are printed)."""
    if v["verdict"] == STRONGER:
        return "both 95% intervals above 50%"
    if v["verdict"] == WEAKER:
        return "both 95% intervals below 50%"
    holds = [name for name, lo, hi in (("Wilson", v["lo"], v["hi"]),
                                        ("bootstrap", v["boot_lo"], v["boot_hi"]))
             if lo <= 0.5 <= hi]
    if len(holds) == 2:
        return "both 95% intervals hold 50%"
    if holds:
        return f"the {holds[0]} 95% interval holds 50%"
    return "the two 95% intervals lie on either side of 50%"


def game_records(path: str | Path) -> list[dict]:
    """The league_game records of a log, the first of each (protocol, key), as read_log keeps
    them; unreadable lines skipped."""
    out, seen = [], set()
    path = Path(path)
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict) or rec.get("kind") != LE.KIND_GAME:
                continue
            k = (rec.get("protocol"), rec.get("key"))
            if k in seen:
                continue
            seen.add(k)
            out.append(rec)
    return out


def pick_protocol(log: Any, candidate: str, champion: str, protocol: str | None = None) -> str:
    """The protocol to read: `protocol` when given, else that of the last run header whose
    players hold both names, else the one with the most games between them (ValueError when
    they never met)."""
    if protocol:
        return protocol
    for run in reversed(log.runs):
        players = run.get("players") or {}
        if candidate in players and champion in players and run.get("protocol"):
            return str(run["protocol"])
    met = Counter(g.protocol for g in log.games if set(g.teams) == {candidate, champion})
    if not met:
        raise ValueError(f"{candidate} and {champion} have no games in the log")
    return met.most_common(1)[0][0]


def player_of(log: Any, protocol: str, name: str) -> dict:
    """A player's identity from the last run header of `protocol` that lists it ({} if none)."""
    for run in reversed(log.runs):
        if run.get("protocol") == protocol and name in (run.get("players") or {}):
            return dict(run["players"][name])
    return {}


def per_game(records: Sequence[Mapping[str, Any]], name: str) -> dict:
    """`name`'s means per game over `records` (league_game lines): plays, leak, and the counts
    of its seat info "rules" (each 0 in a game whose seat has no rules)."""
    n, plays, leak = 0, 0.0, 0.0
    counts: Counter = Counter()
    for rec in records:
        teams = list(rec.get("teams") or ())
        if name not in teams:
            continue
        t = teams.index(name)
        res = rec.get("result") or {}
        n += 1
        plays += float((res.get("plays") or [0, 0])[t])
        leak += float((res.get("leak") or [0.0, 0.0])[t])
        info = (res.get("info") or [None, None])[t]
        rules = info.get("rules") if isinstance(info, Mapping) else None
        for k, v in (rules or {}).items():
            if k not in RULE_KEYS:
                counts[k] += int(v)
    if not n:
        return {"games": 0}
    return {"games": n, "plays": plays / n, "leak": leak / n,
            "rules": {k: v / n for k, v in sorted(counts.items())}}


def ab(log_path: str | Path, candidate: str, champion: str, *, games: int = DEFAULT_GAMES,
       protocol: str | None = None, config: Any = None) -> dict:
    """The verdict of `candidate` against `champion` (module doc, THE VERDICT), JSON-ready."""
    log = LE.read_log(log_path)
    pd = pick_protocol(log, candidate, champion, protocol)
    played = [g for g in log.of(pd) if set(g.teams) == {candidate, champion}]
    if not played:
        raise ValueError(f"{candidate} and {champion} have no games in protocol {pd}")
    config = config or LE.GateConfig()
    row = LE.matchups(played, [candidate, champion], config)[0]
    if row["a"] != candidate:                              # matchups puts the first name first
        raise AssertionError(row)
    keys = {g.key for g in played}
    records = [r for r in game_records(log_path) if r.get("protocol") == pd and r.get("key") in keys]
    ident = {n: player_of(log, pd, n) for n in (candidate, champion)}
    digests = {n: ident[n].get("ckpt_digest") for n in ident}
    out = {"candidate": candidate, "champion": champion, "protocol": pd,
           "split": next((g.split for g in played if g.split), ""), "games": len(played),
           "games_wanted": int(games), "complete": len(played) >= int(games),
           "score": row["score"], "lo": row["lo"], "hi": row["hi"], "boot_lo": row["boot_lo"],
           "boot_hi": row["boot_hi"], "pairings": row["pairings"], "wins": row["wins"],
           "draws": row["draws"], "losses": row["losses"],
           "verdict": verdict_of(row["lo"], row["hi"], row["boot_lo"], row["boot_hi"]),
           "players": {n: {"ckpt": ident[n].get("ckpt"), "ckpt_digest": digests[n],
                           "rules": {k: ident[n][k] for k in RULE_KEYS if k in ident[n]}}
                       for n in ident},
           "same_checkpoint": bool(digests[candidate]) and digests[candidate] == digests[champion],
           "per_game": {n: per_game(records, n) for n in (candidate, champion)}}
    return out


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _rules_text(rules: Mapping[str, Any]) -> str:
    if not rules:
        return "none"
    parts = []
    if "wait_floor" in rules:
        parts.append(f"wait floor {float(rules['wait_floor']):g}")
    if rules.get("stray_guard"):
        parts.append("stray guard")
    return ", ".join(parts)


def format_ab(v: Mapping[str, Any], log: str = "") -> list[str]:
    """The lines printed for one verdict."""
    c, h = v["candidate"], v["champion"]
    lines = [f"decide_ab: {c} against {h}{' in ' + log if log else ''} (protocol {v['protocol']}"
             f"{', ' + LE.SPLIT_NAMES.get(v['split'], v['split']) + ' split' if v['split'] else ''})"]
    for name in (c, h):
        p = v["players"][name]
        lines.append(f"  {name}: rules {_rules_text(p['rules'])}; checkpoint {p['ckpt'] or '?'} "
                     f"[{p['ckpt_digest'] or 'no digest'}]")
    if not v["same_checkpoint"]:
        lines.append("  warning: the two players are not the same checkpoint (digests above): "
                     "the difference is not the rules' alone")
    lines.append(f"  score {_pct(v['score'])} over {v['games']} games ({v['wins']} W "
                 f"{v['draws']} D {v['losses']} L); 95% [{_pct(v['lo'])}, {_pct(v['hi'])}] "
                 f"Wilson, [{_pct(v['boot_lo'])}, {_pct(v['boot_hi'])}] bootstrap over "
                 f"{v['pairings']} pairings")
    pc, ph = v["per_game"][c], v["per_game"][h]
    if pc.get("games") and ph.get("games"):
        fired = ", ".join(f"{k.replace('_', ' ')} {x:.2f}" for k, x in pc["rules"].items())
        lines.append(f"  a game: plays {pc['plays']:.1f} ({h} {ph['plays']:.1f}), elixir leak "
                     f"{pc['leak']:.2f} ({ph['leak']:.2f}); {c}'s rules: "
                     f"{fired or 'never fired'}")
    word = {STRONGER: "STRONGER", NO_DIFFERENCE: "NO DIFFERENCE", WEAKER: "WEAKER"}[v["verdict"]]
    lines.append(f"verdict: {word}: {c} scores {_pct(v['score'])} against {h}, 95% "
                 f"[{_pct(v['lo'])}, {_pct(v['hi'])}] Wilson, [{_pct(v['boot_lo'])}, "
                 f"{_pct(v['boot_hi'])}] bootstrap ({why_of(v)})")
    if not v["complete"]:
        lines.append(f"  INCOMPLETE: {v['games']} of {v['games_wanted']} games; run the same "
                     "command again, the league plays only the games its log lacks")
    lines.append(f"  {NEXT[v['verdict']]}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The decision rules' A/B verdict from a league log "
                                             "(scripts/decide_ab.ps1 plays the games).")
    ap.add_argument("--log", type=Path, required=True, help="the league's JSONL log")
    ap.add_argument("--champion", required=True, metavar="NAME",
                    help="the player with the rules off")
    ap.add_argument("--candidate", action="append", required=True, metavar="NAME",
                    help="a player with rules on (repeatable)")
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES,
                    help=f"games each candidate should have (default {DEFAULT_GAMES})")
    ap.add_argument("--protocol", default=None, metavar="DIGEST",
                    help="the log's protocol (default: the last run's with both names)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.log.is_file():
        print(f"error: {args.log} does not exist", file=sys.stderr)
        return 1
    code = 0
    for cand in args.candidate:
        try:
            v = ab(args.log, cand, args.champion, games=args.games, protocol=args.protocol)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            code = 1
            continue
        print("\n".join(format_ab(v, str(args.log))), flush=True)
        if not v["complete"] and code == 0:
            code = 2
    return code


if __name__ == "__main__":
    sys.exit(main())
