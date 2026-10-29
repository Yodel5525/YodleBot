"""Which end of the board each IL_Replay side owns, decided once per match from troop
plays, with one coordinate mapping and a checked invariant (plan E07, audit tokens-11).

IL_Replay gives every play in native world units: millitiles on the 18,000 x 32,000
board, one side at low y and the other at high y. RoyaleSim's world is the same frame
with blue (team 0) at low y. So the one mapping is the SEAT: the side whose troops go
down at native low y plays blue, the other red, and every coordinate goes to the engine
as recorded (`to_engine`: millitiles x SUBTILE_PER_MILLITILE). Nothing is rotated.

tools/rs_rows.py, rs_eval.py and royalesim_redrive.py used to take the seat from the
median y of ALL the team's plays (spells included), and to turn the board half a turn
whenever that put the team in red. Seat swap plus half turn is a symmetry of the game,
so the two cancelled: the result was always "team at native low y", right only if the
dataset always says so, and the median test was dead logic. `orient` replaces both:

* only troop and building plays vote (catalogue card_kind TROOP or BUILDING, column
  KIND_COLUMN), not spells and not the tunnellers in ANYWHERE (Miner, Goblin Drill:
  RoyaleSim's can_deploy_on_enemy_side cards), which may go down on the enemy half;
* both sides vote: "team low" is team troops at low y plus opponent troops at high y,
  "team high" the reverse; a tie (no troop play at all, say) keeps the team at low y;
* each side's own-half share is the fraction of its troop plays on the half its seat
  owns (y < HALF_Y for blue, y >= HALF_Y for red); a side with no troop play has share
  1.0, since nothing contradicts it;
* a match with either share below MIN_SHARE (0.9) is skipped as SKIP ("orient") and
  counted by the caller: either the frame is not what this module assumes or the side
  labels are wrong. Troops are legal on the enemy half only in a pocket opened by a
  fallen princess tower, hence 0.9 and not 1.0.

Golden: when the team owns native low y, team_seat is 0 and the coordinates are the
ones the tools always sent, so their output is unchanged. (The one exception was never
right by construction: a team at low y whose median play, spells included, was past
the river used to be put in red and turned; it now plays blue unturned. By the symmetry
above its own-frame rows are the same; `side` and the engine seat follow the half.)

Survey the corpus before building rows (needs pyarrow and royalesim):

    .venvs/royalegym/Scripts/python.exe tools/rs_orient.py --parts 4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

BOARD_W, BOARD_H = 18_000, 32_000      # native / RoyaleSim world millitiles
HALF_Y = 16_000                        # blue (team 0) owns y < HALF_Y, red the rest
MIN_SHARE = 0.9                        # below this own-half share a match is skipped
SKIP = "orient"                        # the skip reason the tools count
#: catalogue_json column of card_kind ("TROOP" / "BUILDING" / "SPELL"); royalesim's
#: CATALOGUE_FIELDS begin name, placement, elixir, count, radius, flying, hitpoints,
#: footprint_tiles, card_kind (the column tools/rs_rows.py reads as row[8]).
KIND_COLUMN = 8
TROOP_KINDS = frozenset({"TROOP", "BUILDING"})
#: Troop / building cards that may legally go down on the enemy half: the tunnellers
#: (RoyaleSim cards.json can_deploy_on_enemy_side, placement code TUNNEL).
ANYWHERE = frozenset({"Miner", "GoblinDrill"})

#: A play as the tools hold it: (tick, RoyaleSim card name or None, x, y), native millitiles.
Play = Sequence[Any]


class Orientation(NamedTuple):
    """team_seat: the RoyaleSim team of IL_Replay's "team" (0 = blue = native low y).
    share: (team, opponent), each side's share of its troop plays on its own half."""
    team_seat: int
    share: tuple[float, float]

    @property
    def seats(self) -> dict[int, int]:
        """IL side (0 team, 1 opponent) -> RoyaleSim team."""
        return {0: self.team_seat, 1: 1 - self.team_seat}

    def ok(self, min_share: float = MIN_SHARE) -> bool:
        """Both sides put at least `min_share` of their troop plays on their own half."""
        return min(self.share) >= min_share

    def describe(self) -> str:
        return (f"team at native {'low' if self.team_seat == 0 else 'high'} y, own-half "
                f"share team {self.share[0]:.2f} opponent {self.share[1]:.2f}")


def troop_names(catalogue: Iterable[Sequence[Any]]) -> frozenset[str]:
    """Names of the catalogue cards whose placement says which half a player owns:
    card_kind TROOP or BUILDING, less the ANYWHERE tunnellers. A row without the
    card_kind column counts as no troop."""
    out = set()
    for row in catalogue:
        if len(row) > KIND_COLUMN and str(row[KIND_COLUMN]).upper() in TROOP_KINDS:
            if str(row[0]) not in ANYWHERE:
                out.add(str(row[0]))
    return frozenset(out)


def half_counts(plays: Mapping[int, Sequence[Play]] | Sequence[Sequence[Play]],
                troop: Collection[str]) -> tuple[tuple[int, int], tuple[int, int]]:
    """((team low, team high), (opponent low, opponent high)): troop plays per native
    half. `plays[0]` are the team's, `plays[1]` the opponent's, each (tick, name, x, y)."""
    counts = []
    for side in (0, 1):
        low = high = 0
        for play in plays[side]:
            if play[1] in troop:
                if play[3] < HALF_Y:
                    low += 1
                else:
                    high += 1
        counts.append((low, high))
    return counts[0], counts[1]


def orient(plays: Mapping[int, Sequence[Play]] | Sequence[Sequence[Play]],
           troop: Collection[str]) -> Orientation:
    """The match's seat and own-half shares from its troop plays (`plays` as half_counts
    reads them, `troop` = troop_names(catalogue)). See the module doc."""
    (t_low, t_high), (o_low, o_high) = half_counts(plays, troop)
    team_seat = 1 if t_high + o_low > t_low + o_high else 0
    own = (t_low, o_high) if team_seat == 0 else (t_high, o_low)
    total = (t_low + t_high, o_low + o_high)
    share = (own[0] / total[0] if total[0] else 1.0, own[1] / total[1] if total[1] else 1.0)
    return Orientation(team_seat, share)


def to_engine(x: int, y: int, sub: int) -> tuple[int, int]:
    """The one coordinate mapping: native world millitiles -> RoyaleSim engine subtiles
    (sub = royalesim.SUBTILE_PER_MILLITILE). The seat, not the coordinates, follows the
    half a side owns."""
    return x * sub, y * sub


# ---------------------------------------------------------------------------
# corpus survey
# ---------------------------------------------------------------------------

def plays_of(payload: Mapping[str, Any], names: Mapping[str, str],
             key_of: Callable[[str], tuple[str, int]]) -> dict[int, list[tuple]]:
    """A payload's play_card events as the tools read them: {0: team, 1: opponent}, each
    (tick, RoyaleSim name or None, x, y) in (tick, source_index) order. key_of is
    royalesim_redrive.base_key."""
    events = sorted((e for e in payload.get("events") or () if e["kind"] == "play_card"),
                    key=lambda e: (e["replay_tick_20hz"], e["source_index"]))
    plays: dict[int, list[tuple]] = {0: [], 1: []}
    for e in events:
        u = e["coordinates"]["native_world_units"]
        plays[0 if e["side"] == "team" else 1].append(
            (int(e["replay_tick_20hz"]), names.get(key_of(e["card_key"])[0]),
             int(u["x"]), int(u["y"])))
    return plays


#: survey() share bins: (label, lowest min-share in the bin).
SHARE_BINS = (("1.00", 1.0), ("0.90-0.99", 0.9), ("0.80-0.89", 0.8), ("0.50-0.79", 0.5),
              ("<0.50", 0.0))


def survey(payloads: Iterable[Mapping[str, Any]], names: Mapping[str, str],
           troop: Collection[str], key_of: Callable[[str], tuple[str, int]],
           min_share: float = MIN_SHARE) -> dict[str, Any]:
    """Counts over 1v1 payloads: the seat orient picks, how many it skips, the lower of
    the two own-half shares in SHARE_BINS, and the raw troop plays per native half
    (no orientation logic: does "team" always own native low y?)."""
    out: dict[str, Any] = {"matches": 0, "not_1v1": 0, "team_low": 0, "team_high": 0,
                           "skipped": 0, "no_troop_plays": 0,
                           "team_troops": [0, 0], "opponent_troops": [0, 0],
                           "min_share": {label: 0 for label, _ in SHARE_BINS}}
    for payload in payloads:
        battle = payload["battle"]
        if any(len(battle[side]["players"]) != 1 for side in ("team", "opponent")):
            out["not_1v1"] += 1
            continue
        plays = plays_of(payload, names, key_of)
        (t_low, t_high), (o_low, o_high) = half_counts(plays, troop)
        ori = orient(plays, troop)
        out["matches"] += 1
        out["team_low" if ori.team_seat == 0 else "team_high"] += 1
        out["skipped"] += not ori.ok(min_share)
        out["no_troop_plays"] += t_low + t_high + o_low + o_high == 0
        out["team_troops"][0] += t_low
        out["team_troops"][1] += t_high
        out["opponent_troops"][0] += o_low
        out["opponent_troops"][1] += o_high
        low = min(ori.share)
        out["min_share"][next(label for label, edge in SHARE_BINS if low >= edge)] += 1
    return out


def _pct(a: int, b: int) -> str:
    return f"{100 * a / max(1, b):.1f}%"


def main(argv: list[str] | None = None) -> int:
    tools = str(Path(__file__).resolve().parent)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import rs_cards
    import rs_engine
    from royalesim_redrive import base_key
    ap = argparse.ArgumentParser(
        description="Which end of the board IL_Replay's 'team' owns, match by match, as "
                    "rs_rows / rs_eval / royalesim_redrive decide it (needs pyarrow and "
                    "royalesim for the catalogue's card kinds).")
    ap.add_argument("--corpus", type=Path, default=rs_cards.CORPUS,
                    help="IL_Replay folder with replays/part-*.parquet (default data/il_replay)")
    ap.add_argument("--parts", type=int, default=4, help="read the first N parts (0 = all)")
    ap.add_argument("--min-share", type=float, default=MIN_SHARE,
                    help=f"own-half share below which a match is skipped (tools: {MIN_SHARE})")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    rs_engine.add_engine_args(ap)
    args = ap.parse_args(argv)
    try:
        cfg = rs_engine.config_from_args(args).undelayed()
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    files = rs_cards.corpus_parts(args.corpus, args.parts)
    if not files:
        print(f"no part-*.parquet in {Path(args.corpus) / 'replays'} "
              "(run tools/il_replay_download.py)", file=sys.stderr)
        return 1
    troop = troop_names(rs_engine.catalogue(cfg))
    out = survey(rs_cards.iter_payloads(files), rs_cards.load(), troop, base_key,
                 args.min_share)
    out["parts"] = len(files)
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    n = out["matches"]
    tl, th = out["team_troops"]
    ol, oh = out["opponent_troops"]
    print(f"{n:,} 1v1 matches in {len(files)} parts ({out['not_1v1']:,} others)")
    print(f"  team at native low y {out['team_low']:,} ({_pct(out['team_low'], n)}), "
          f"high y {out['team_high']:,} ({_pct(out['team_high'], n)}); "
          f"no troop play {out['no_troop_plays']:,}")
    print(f"  skipped as '{SKIP}' (an own-half share < {args.min_share:.2f}): "
          f"{out['skipped']:,} ({_pct(out['skipped'], n)})")
    print("  lower own-half share: " + ", ".join(
        f"{label} {k:,}" for label, k in out["min_share"].items()))
    print(f"  troop plays at native low / high y: team {tl:,} / {th:,} "
          f"({_pct(tl, tl + th)} low), opponent {ol:,} / {oh:,} ({_pct(oh, ol + oh)} high)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
