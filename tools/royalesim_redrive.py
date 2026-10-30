"""Re-drive the same IL_Replay matches through RoyaleSim, for a paired
fidelity comparison with our own engine (tools/redrive_bench.py).

Runs in the isolated RoyaleGym environment:

    .venvs/royalegym/Scripts/python.exe tools/royalesim_redrive.py --pair b2000

It reads the match tags from reports/redrive/<pair>.jsonl, so both engines
play exactly the same matches, and writes reports/redrive/<pair>_royalesim.jsonl
with the same columns the report reads (winner / crowns agreement, refusals).

How a match is played, kept as close to our harness as the API allows:
* decks dealt unshuffled in an order under which every recorded play was in
  hand (searched, as RoyaleImitate does), so a hand refusal means the engine's
  cycle disagreed, not that the start was unknown;
* every card at its recorded level and form (evolution / hero), towers at the
  match's most common card level, both sides on 6 starting elixir;
* each play at its recorded tick and native tile. RoyaleSim cannot be told to
  ignore elixir, so a play it refuses is counted and dropped rather than
  forced. That is a harder test than ours, which tops elixir up;
* the side whose troops go down at native low y plays blue, with coordinates
  as recorded (tools/rs_orient.py); a match where either side has under 90% of
  its troop plays on its own half fails with error "orient: ...".

The engine (card table, levels) is data/engine.json's, through
tools/rs_engine.make_battle, with no command delay (plays go in at their
recorded ticks). `--card-table default` runs the engine's own table, as this
tool did before rs_engine; `--card-table client15535` names it explicitly.

Cards map to RoyaleSim names through tools/rs_cards.py (data/cards/, no royaleimitate
needed). HASTYCR_CARD_EXTRAS=0 drops the void / rune-giant / spirit-empress extras and
leaves royaleimitate 0.2.15's 119-key map, which this tool read before. Matches with those
cards then fail with "card not in RoyaleSim" as they used to, so runs stay comparable.

Presses (E10, tools/rs_presses.py): `--presses off` (the default) replays play_card events
only, as before, and writes reports/redrive/<pair>_royalesim.jsonl unchanged. `--presses on`
also replays every activate_ability event as (team, HAND_SIZE + button, 0, 0) at its tick, in
the same step as that tick's plays (the button: the card of the event among the side's heroes,
form 2, in dealt-deck order, then its champions; the engine's own order when its abilities
rows name the cards). Each result goes to the match's "presses" counts (events, sent, ok,
refused by reason, skipped "unknown" / "ambiguous", order_mismatch), not to "forced", so the
refusals per 100 plays stay comparable; the run ends at the last play or press + 200. Its
output is <pair>_royalesim_presses.jsonl, so both runs can sit side by side and their winner
and crown agreement be compared (rsim-02).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations, permutations
from pathlib import Path
from statistics import mode

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "il_replay"
if str(ROOT / "tools") not in sys.path:      # also when imported as tools.royalesim_redrive
    sys.path.insert(0, str(ROOT / "tools"))
import rs_cards  # noqa: E402
import rs_engine  # noqa: E402
import rs_orient  # noqa: E402
import rs_presses  # noqa: E402


def deal_order(deck, plays):
    for hand in combinations(deck, 4):
        rest = [c for c in deck if c not in hand]
        for queue in permutations(rest):
            held, waiting = set(hand), list(queue)
            for card in plays:
                if card not in held:
                    break
                held.remove(card)
                waiting.append(card)
                held.add(waiting.pop(0))
            else:
                return [*hand, *queue]
    return None


def base_key(key):
    for suffix in ("-ev1", "-hero"):
        if key.endswith(suffix):
            return key[: -len(suffix)], (1 if suffix == "-ev1" else 2)
    return key, 0


_ENGINE = {}


def _engine(engine=None):
    """royalesim, the card map (rs_cards.load()), catalogue ids, the troop names that decide
    the seat (rs_orient.troop_names) and the champions' ids (rs_presses.champion_ids) under
    `engine` (an rs_engine.EngineConfig or its to_json(); None = rs_engine.active()), cached
    per config, delay forced to 0."""
    cfg = rs_engine.resolve(engine).undelayed()
    if _ENGINE.get("key") != cfg.key():
        import royalesim
        names = rs_cards.load()
        b = rs_engine.make_battle(cfg, royalesim)
        cat = json.loads(b.catalogue_json())
        ids = {row[0]: i for i, row in enumerate(cat)}
        _ENGINE.clear()
        _ENGINE.update(royalesim=royalesim, names=names, ids=ids,
                       troop=rs_orient.troop_names(cat), cfg=cfg, key=cfg.key(),
                       champions=rs_presses.champion_ids(
                           cat, getattr(royalesim, "CATALOGUE_FIELDS", None)))
    return _ENGINE


def one(payload_text: str, engine=None, presses: str = rs_presses.PRESS_OFF) -> dict:
    """Re-drive one match (module doc). `presses` as --presses: under "on" the recorded
    ability presses are sent too and the result has "presses" (rs_presses.PressLog counts)."""
    press_on = rs_presses.press_mode(presses) == rs_presses.PRESS_ON
    eng = _engine(engine)
    royalesim, names, ids = eng["royalesim"], eng["names"], eng["ids"]
    p = json.loads(payload_text)
    battle = p["battle"]
    tag = p["source"]["replay_tag"]
    out = {"tag": tag, "ok": False, "error": "", "forced": {}, "plays": 0}
    sides = ("team", "opponent")
    decks, forms, levels = [], [], []
    for side in sides:
        players = battle[side]["players"]
        deck, form, level = [], [], []
        for card in players[0]["deck"]:
            key, f = base_key(card["card_key"])
            name = names.get(key)
            if name is None or name not in ids:
                out["error"] = f"card not in RoyaleSim: {key}"
                return out
            deck.append(name)
            form.append(f)
            level.append(int(card.get("level") or 11))
        decks.append(deck)
        forms.append(form)
        levels.append(level)
    events = sorted((e for e in p["events"] if e["kind"] == "play_card"),
                    key=lambda e: (e["replay_tick_20hz"], e["source_index"]))
    plays = {0: [], 1: []}
    for e in events:
        seat = 0 if e["side"] == "team" else 1
        key, _ = base_key(e["card_key"])
        name = names.get(key)
        u = e["coordinates"]["native_world_units"]
        plays[seat].append((int(e["replay_tick_20hz"]), name, int(u["x"]), int(u["y"])))
    # Which end each side owns, from troop plays (tools/rs_orient.py): the side at native
    # low y plays blue, and every coordinate goes in as recorded.
    ori = rs_orient.orient(plays, eng["troop"])
    if not ori.ok():
        out["error"] = f"{rs_orient.SKIP}: {ori.describe()}"
        return out
    team_seat = ori.team_seat                  # RoyaleSim team index of "team"
    dealt = []
    for seat in (0, 1):
        order = deal_order(decks[seat], [c for _, c, _, _ in plays[seat]])
        if order is None:
            out["error"] = "no deal order fits the plays"
            return out
        dealt.append(order)
    perm = [[decks[s].index(c) for c in dealt[s]] for s in (0, 1)]
    rs_side = {0: team_seat, 1: 1 - team_seat}          # our seat -> RoyaleSim team
    by_team = [None, None]
    for seat in (0, 1):
        by_team[rs_side[seat]] = seat
    lv_all = [lv for s in levels for lv in s]
    tower = max(1, min(16, mode(lv_all)))
    b = rs_engine.make_battle(eng["cfg"], royalesim)
    b.reset(seed=0,
            decks=[[ids[c] for c in dealt[by_team[t]]] for t in (0, 1)],
            shuffle=0, start_tick=0, elixir_milli=[6000, 6000], tower_hp=None, spawns=[],
            forms=[[forms[by_team[t]][i] for i in perm[by_team[t]]] for t in (0, 1)],
            levels=[[levels[by_team[t]][i] for i in perm[by_team[t]]] for t in (0, 1)],
            tower_levels=[tower, tower])
    SUB = royalesim.SUBTILE_PER_MILLITILE
    queue = sorted([(t, seat, c, x, y) for seat in (0, 1) for (t, c, x, y) in plays[seat]])
    # E10 (--presses on): the recorded presses, (tick, RoyaleSim team, rs_presses.Press) in
    # tick order, and each team's button cards (the dealt decks and forms given to reset).
    press_q, btns, plog, checked = [], {}, None, not press_on
    if press_on:
        for t in (0, 1):
            seat = by_team[t]
            btns[t] = rs_presses.side_buttons([ids[c] for c in dealt[seat]],
                                              [forms[seat][k] for k in perm[seat]],
                                              eng["champions"])
        press_q = [(pr.tick, rs_side[pr.side], pr)
                   for pr in rs_presses.parse(p["events"], names, ids)]
        plog = rs_presses.PressLog(events=len(press_q))
    hand_n = rs_presses.hand_size(royalesim)
    forced = Counter()
    tick, i, j = 0, 0, 0
    last = max([queue[-1][0] if queue else 0] + [q[0] for q in press_q])
    while i < len(queue) or j < len(press_q):
        t = queue[i][0] if i < len(queue) else press_q[j][0]
        if j < len(press_q):
            t = min(t, press_q[j][0])
        if t > tick:
            b.step([], t - tick)
            tick = t
        state = json.loads(b.state_json())
        if state["game_over"]:
            break
        if not checked:              # the engine's own button order, where it names the cards
            for team in (0, 1):
                btns[team], differs = rs_presses.engine_order(state["players"][team], btns[team])
                plog.order_mismatch += int(differs)
            checked = True
        cmds = []
        while i < len(queue) and queue[i][0] == t:
            _, seat, card, x, y = queue[i]
            team = rs_side[seat]
            hand = state["players"][team]["hand"]
            cid = ids[card]
            out["plays"] += 1
            if cid not in hand:
                forced["hand"] += 1
            else:
                cmds.append((team, hand.index(cid), *rs_orient.to_engine(x, y, SUB)))
            i += 1
        n_plays = len(cmds)
        while j < len(press_q) and press_q[j][0] == t:
            _, team, press = press_q[j]
            j += 1
            button, why = rs_presses.button_of(press.cards, btns[team])
            if button is None:
                plog.skip(why)
            else:
                cmds.append(rs_presses.command(team, button, hand_n))
                plog.sent += 1
        if cmds:
            res = b.step(cmds, 1)
            tick += 1
            for k, row in enumerate(res):
                reason = royalesim.DEPLOY_REASONS[row[1]]
                if press_on and k >= n_plays:      # a press's result: counted apart
                    plog.result(reason)
                elif reason != "OK":
                    forced[reason.lower()] += 1
    end = min(last + 200, 6000)
    if tick < end:
        b.step([], end - tick)
    state = json.loads(b.state_json())
    if not state["game_over"]:
        b.step([], 6000)
        state = json.loads(b.state_json())
    crowns = {"team": state["players"][team_seat]["crowns"],
              "opponent": state["players"][1 - team_seat]["crowns"]}
    w = state["winner"]
    winner = ("team" if w == team_seat else "opponent" if w in (0, 1) else "draw")
    rec = battle["team"]["crowns"], battle["opponent"]["crowns"]
    result = str(battle.get("result", "")).lower()
    rec_winner = ("team" if result in ("victory", "win") else
                  "opponent" if result in ("defeat", "loss") else
                  ("team" if rec[0] > rec[1] else "opponent" if rec[1] > rec[0] else "draw"))
    out.update(ok=True, forced=dict(forced), engine_crowns=crowns,
               recorded_crowns={"team": rec[0], "opponent": rec[1]},
               winner_agree=winner == rec_winner,
               crowns_agree=crowns == {"team": rec[0], "opponent": rec[1]},
               end=state["tick"])
    if press_on:
        out["presses"] = plog.to_json()
    return out


def press_totals(rows) -> dict:
    """The "presses" counts of every re-driven match summed (rs_presses.PressLog.to_json's
    keys), for main's summary line; {} when no row has them (--presses off)."""
    total: dict = {}
    for r in rows:
        for key, value in (r.get("presses") or {}).items():
            if isinstance(value, dict):
                bucket = total.setdefault(key, {})
                for k, v in value.items():
                    bucket[k] = bucket.get(k, 0) + v
            else:
                total[key] = total.get(key, 0) + value
    return total


def output_path(pair: str, presses: str = rs_presses.PRESS_OFF) -> Path:
    """reports/redrive/<pair>_royalesim.jsonl, or <pair>_royalesim_presses.jsonl under
    --presses on, so a run with presses does not overwrite the run without."""
    suffix = "_presses" if rs_presses.press_mode(presses) == rs_presses.PRESS_ON else ""
    return ROOT / "reports" / "redrive" / f"{pair}_royalesim{suffix}.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="b2000")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--presses", choices=rs_presses.PRESS_MODES, default=rs_presses.PRESS_OFF,
                    help="'off' (default): play_card events only, as before; 'on': also replay "
                         "activate_ability events as button presses (output "
                         "<pair>_royalesim_presses.jsonl)")
    rs_engine.add_engine_args(ap)
    args = ap.parse_args()
    try:
        cfg = rs_engine.config_from_args(args).undelayed()
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    rs_engine.set_active(cfg)
    fp = rs_engine.fingerprint(rs_engine.make_battle(cfg), cfg=cfg)
    print(f"engine: {cfg.describe()}\n        {rs_engine.summary(fp)}", flush=True)
    print(f"cards:  {rs_cards.summary(rs_cards.load())}", flush=True)
    print(f"presses: {rs_presses.describe_presses(args.presses)}", flush=True)
    import pyarrow.parquet as pq
    wanted = {json.loads(l)["tag"] for l in
              (ROOT / "reports" / "redrive" / f"{args.pair}.jsonl").open(encoding="utf-8")}
    payloads = {}
    for part in sorted((CORPUS / "replays").glob("part-*.parquet")):
        acts = pq.read_table(CORPUS / "actions" / part.name,
                             columns=["replay_tag", "event_json"]).to_pylist()
        events = {}
        for a in acts:
            if a["replay_tag"] in wanted:
                events.setdefault(a["replay_tag"], []).append(json.loads(a["event_json"]))
        for r in pq.read_table(part, columns=["replay_tag", "payload_json"]).to_pylist():
            if r["replay_tag"] in wanted:
                payload = json.loads(r["payload_json"])
                payload["events"] = events.get(r["replay_tag"], [])
                payload.setdefault("source", {})["replay_tag"] = r["replay_tag"]
                payloads[r["replay_tag"]] = json.dumps(payload)
        if len(payloads) >= len(wanted):
            break
    print(f"{len(payloads)} of {len(wanted)} matches found", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=rs_engine.set_active,
                             initargs=(cfg.to_json(),)) as pool:
        rows = list(pool.map(partial(one, presses=args.presses), list(payloads.values()),
                             chunksize=8))
    out = output_path(args.pair, args.presses)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if r["ok"]]
    errs = Counter(r["error"].split(":")[0] for r in rows if not r["ok"])
    forced = Counter()
    for r in ok:
        forced.update(r["forced"])
    plays = sum(r["plays"] for r in ok)
    e3 = sum(1 for r in ok if 3 in r["engine_crowns"].values())
    r3 = sum(1 for r in ok if 3 in r["recorded_crowns"].values())
    print(f"RoyaleSim: {len(ok)} ran ({time.time() - t0:.0f}s), errors {dict(errs)}")
    print(f"  winner agree {sum(r['winner_agree'] for r in ok) / max(1, len(ok)):.1%}, "
          f"crowns agree {sum(r['crowns_agree'] for r in ok) / max(1, len(ok)):.1%}, "
          f"3-crown {e3} vs recorded {r3}")
    print("  refused per 100 plays: " + ", ".join(
        f"{k} {100 * v / max(1, plays):.2f}" for k, v in forced.most_common()))
    if args.presses == rs_presses.PRESS_ON:
        print(f"  presses: {rs_presses.describe_log(press_totals(ok))}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
