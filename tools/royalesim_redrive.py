"""Re-drive the same IL_Replay matches through RoyaleSim, for a paired
fidelity comparison with our own engine (tools/redrive_bench.py).

Runs in the isolated RoyaleGym environment:

    .venvs/royalegym/Scripts/python.exe tools/royalesim_redrive.py --pair b2000
    .venvs/royalegym/Scripts/python.exe tools/royalesim_redrive.py --sample 2000 --split val

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

Fidelity grid (E03; tools/rs_fidelity_grid.py runs this tool once per cell):

* Which matches. `--pair NAME` (the default, b2000) reads the tags of
  reports/redrive/<pair>.jsonl, as before; that file is not in the repo. `--sample N --split
  val --seed S` instead picks N matches of the corpus by tag hash: among the matches of the
  game modes in --modes (default MODES, the ranked modes tools/rs_rows.py trains on; "all"
  for every mode) whose split is --split (split_of, tools/rs_rows.split_of's rule; "all" for
  every split), the N with the smallest sample_key(tag, seed) = the first 16 hex digits of
  sha1("<seed>:<tag>"). The pick does not depend on the order the corpus is read in, and the
  sample of N is the first N of a larger one. `--payloads FILE` re-drives the payloads of a
  JSON-lines file (one payload_json per line, events attached, as `--save-payloads FILE`
  writes them), so a grid reads the corpus once and an interpreter without pyarrow (the
  0.1.22 venv) can run it. The output name follows --pair, else sample<N>_<split>_seed<S>,
  else the payload file's stem; `--out FILE` names the output instead.
* One bad match no longer stops the run (rsim-09): an exception inside a match becomes that
  match's row, ok false, error "exception <Type>: <message>" (main counts it as
  "exception <Type>"), "where" the file:line it came from. A missing royalesim, or a bad
  --presses / --tick-semantics, still stops it.
* `--card-table T` (tools/rs_engine.py) as before.
* `--tick-semantics exec` (the default) sends each play at its recorded tick with no command
  delay, reading IL_Replay ticks as execution ticks: the run, its commands and its output are
  the ones before E03. `--tick-semantics tap` reads them as tap ticks: the Battle gets command
  delay --tap-delay (default 21, the live client's 21-22) on both seats, set by
  rs_engine.make_battle before the first reset, and each play is still SENT at its recorded
  tick, so it runs --tap-delay ticks later. A play the engine refuses when it is sent counts as
  before; one it accepts and refuses when it runs (Battle.step_commands_run, read after every
  step) counts under its reason too; a card not in hand while the side has a command pending
  counts as "hand_pending" (the hand only changes when a play runs). Under --presses on a
  press accepted when sent is counted when it runs. Output <name>_tap<delay>.jsonl.
* Every row has, beside the columns above: "game_mode" (the corpus column, else the payload's),
  "split" (split_of), "tower_card" and "tower_card_level" per side (players[0]["tower_card"],
  as sim/replay_corpus.py reads it; RoyaleSim always plays the Tower Princess), "levels" (per
  side, the deck's card levels as {"level": cards}), and once the match ran "tower_level" (the
  crown-tower level it ran at) and "refusals": up to REFUSAL_CAP refused plays as [tick, side,
  card, reason, "send" | "run"]. Columns are only added: every column the report read before
  has the value it had.
* <out>.meta.json (meta_path) records the source of the matches, the engine config and
  fingerprint the matches ran under (command delay included), the card map, the settings and
  the totals (totals()).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
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

#: --tick-semantics: IL_Replay play ticks read as execution ticks (no delay, as before) or as
#: tap ticks (command delay TAP_DELAY, each play sent at its recorded tick).
TICK_EXEC, TICK_TAP = "exec", "tap"
TICK_MODES = (TICK_EXEC, TICK_TAP)
#: The live client's command delay, ticks (RoyaleSim's set_command_delay_ticks doc: 21-22).
TAP_DELAY = 21
MAX_TAP_DELAY = 200
#: tools/rs_rows.split_of's names; "all" takes every split.
SPLITS = ("train", "val", "test")
SPLIT_ALL = "all"
DEFAULT_SPLIT = "val"
#: The game modes tools/rs_rows.py trains on (rs_rows.MODES); --modes all takes every mode.
MODES = frozenset({"Ranked", "Ladder", "1v1 Battle", "Grand Challenge", "Classic Challenge"})
SAMPLE_RULE = ("the N tags with the smallest int(sha1('<seed>:<tag>')[:16], 16) among the "
               "matches of the modes and split asked")
SIDES = ("team", "opponent")
#: The most refused plays a row lists in "refusals" ("forced" counts them all).
REFUSAL_CAP = 64
#: The error of a match that raised: "exception <Type>: <message>".
EXCEPTION = "exception"
META_SCHEMA = 1


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


# ---------------------------------------------------------------------------
# settings (E03)
# ---------------------------------------------------------------------------

def tick_mode(value) -> str:
    """--tick-semantics as the tool takes it: one of TICK_MODES. ValueError otherwise."""
    if not isinstance(value, str) or value not in TICK_MODES:
        raise ValueError(f"--tick-semantics: one of {', '.join(TICK_MODES)}, got {value!r}")
    return value


def tap_ticks(value) -> int:
    """--tap-delay: whole ticks, 1 to MAX_TAP_DELAY. ValueError otherwise."""
    if isinstance(value, bool):
        raise ValueError(f"--tap-delay: a number of ticks, got {value!r}")
    try:
        ticks = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"--tap-delay: a number of ticks, got {value!r}") from None
    if ticks != value and not (isinstance(value, str) and value.strip() == str(ticks)):
        raise ValueError(f"--tap-delay: a whole number of ticks, got {value!r}")
    if not 1 <= ticks <= MAX_TAP_DELAY:
        raise ValueError(f"--tap-delay: 1-{MAX_TAP_DELAY} ticks, got {ticks}")
    return ticks


def describe_ticks(semantics: str, delay: int = TAP_DELAY) -> str:
    """One line for the console."""
    if tick_mode(semantics) == TICK_EXEC:
        return ("exec (IL_Replay ticks read as execution ticks: each play at its recorded tick, "
                "no command delay, as before)")
    return (f"tap (IL_Replay ticks read as tap ticks: command delay {tap_ticks(delay)} set before "
            "reset, each play sent at its recorded tick and run that many ticks later)")


def match_config(cfg, semantics: str = TICK_EXEC, delay: int = TAP_DELAY):
    """The EngineConfig a match's Battle is built with: `cfg` undelayed under exec, with both
    seats' command delay `delay` under tap (rs_engine.make_battle sets it before reset)."""
    cfg = rs_engine.resolve(cfg).undelayed()
    if tick_mode(semantics) == TICK_EXEC:
        return cfg
    return cfg.delayed(tap_ticks(delay))


def parse_modes(text) -> frozenset | None:
    """--modes: None or "" -> MODES; "all" -> None (every mode); else the comma-separated
    names."""
    if text is None or not str(text).strip():
        return MODES
    if str(text).strip().lower() == "all":
        return None
    names = frozenset(p.strip() for p in str(text).split(",") if p.strip())
    if not names:
        raise ValueError(f"--modes: game mode names separated by commas, or 'all'; got {text!r}")
    return names


def split_of(tag: str) -> str:
    """tools/rs_rows.split_of, verbatim (tests/pipeline/test_fidelity_grid.py checks it)."""
    h = int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 5 else "val" if h < 10 else "train"


def sample_key(tag: str, seed: int = 0) -> int:
    """The order --sample picks matches in: the first 16 hex digits of sha1("<seed>:<tag>")."""
    return int(hashlib.sha1(f"{int(seed)}:{tag}".encode("utf-8")).hexdigest()[:16], 16)


def sample_tags(index, n: int, split: str = DEFAULT_SPLIT, seed: int = 0,
                modes=MODES) -> list[str]:
    """The tags --sample N re-drives (SAMPLE_RULE), smallest sample_key first.

    index: (replay_tag, game_mode, ...) rows (corpus_index). split: a SPLITS name or "all".
    modes: the game modes taken (None: every mode). A tag listed twice counts once.
    """
    if split != SPLIT_ALL and split not in SPLITS:
        raise ValueError(f"--split: one of {', '.join((*SPLITS, SPLIT_ALL))}, got {split!r}")
    keys: dict[str, int] = {}
    for row in index:
        tag, game_mode = str(row[0]), str(row[1] or "")
        if tag in keys:
            continue
        if modes is not None and game_mode not in modes:
            continue
        if split != SPLIT_ALL and split_of(tag) != split:
            continue
        keys[tag] = sample_key(tag, seed)
    ranked = sorted(keys, key=lambda t: (keys[t], t))
    return ranked[:max(0, int(n))]


def sample_name(n: int, split: str = DEFAULT_SPLIT, seed: int = 0) -> str:
    return f"sample{int(n)}_{split}_seed{int(seed)}"


def resolve_pair(pair, sample: int = 0, split: str = DEFAULT_SPLIT, seed: int = 0,
                 payloads=None) -> str:
    """The run's name: --pair when given; else sample_name under --sample, the payload file's
    stem under --payloads, and "b2000" otherwise (as before)."""
    if pair:
        return str(pair)
    if sample:
        return sample_name(sample, split, seed)
    if payloads:
        return Path(payloads).stem
    return "b2000"


# ---------------------------------------------------------------------------
# what a match is (E03)
# ---------------------------------------------------------------------------

def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _int_or(value, default):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def match_info(payload) -> dict:
    """The descriptive columns of a match's row: game_mode, split, tower_card and
    tower_card_level per side (players[0]["tower_card"], sim/replay_corpus.py), and levels
    per side ({"level": cards} of the deck, a missing level read as 11, as the re-drive plays
    it). Never raises: what is missing reads "" / None / {}."""
    payload = _dict(payload)
    battle, source = _dict(payload.get("battle")), _dict(payload.get("source"))
    tag = str(source.get("replay_tag", ""))
    info = {"game_mode": str(source.get("game_mode") or battle.get("game_mode") or ""),
            "split": split_of(tag), "tower_card": {}, "tower_card_level": {}, "levels": {}}
    for side in SIDES:
        players = _dict(battle.get(side)).get("players")
        first = _dict(players[0]) if isinstance(players, list) and players else {}
        tower = _dict(first.get("tower_card"))
        info["tower_card"][side] = str(tower.get("card_key") or "")
        info["tower_card_level"][side] = _int_or(tower.get("level"), None)
        deck = first.get("deck") if isinstance(first.get("deck"), list) else []
        hist = Counter(_int_or(_dict(card).get("level") or 11, 11) for card in deck)
        info["levels"][side] = {str(k): v for k, v in sorted(hist.items())}
    return info


# ---------------------------------------------------------------------------
# one match
# ---------------------------------------------------------------------------

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


def one(payload_text: str, engine=None, presses: str = rs_presses.PRESS_OFF,
        tick_semantics: str = TICK_EXEC, tap_delay: int = TAP_DELAY) -> dict:
    """Re-drive one match (module doc). `presses` as --presses: under "on" the recorded
    ability presses are sent too and the result has "presses" (rs_presses.PressLog counts).
    `tick_semantics` / `tap_delay` as --tick-semantics / --tap-delay. An exception inside the
    match is its row's error (rsim-09); bad settings and a missing royalesim raise."""
    press_on = rs_presses.press_mode(presses) == rs_presses.PRESS_ON
    delay = tap_ticks(tap_delay) if tick_mode(tick_semantics) == TICK_TAP else None
    eng = _engine(engine)
    try:
        return _redrive(payload_text, eng, press_on, delay)
    except Exception as exc:        # one bad match must not stop a 2,000-match run
        return failure(payload_text, exc)


def failure(payload_text: str, exc: BaseException) -> dict:
    """The row of a match that raised `exc`: ok false, error "exception <Type>: <message>",
    "where" the file:line it was raised at, and the descriptive columns when the payload
    parses."""
    try:
        payload = json.loads(payload_text)
    except Exception:
        payload = None
    tag = str(_dict(_dict(payload).get("source")).get("replay_tag", ""))
    out = {"tag": tag, "ok": False, "error": f"{EXCEPTION} {type(exc).__name__}: {exc}",
           "forced": {}, "plays": 0}
    if isinstance(payload, dict):
        out.update(match_info(payload))
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ is not None else []
    out["where"] = f"{Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else ""
    return out


def _redrive(payload_text: str, eng: dict, press_on: bool, delay: int | None) -> dict:
    """one()'s match. delay None: exec semantics; else tap semantics with that command delay."""
    royalesim, names, ids = eng["royalesim"], eng["names"], eng["ids"]
    p = json.loads(payload_text)
    battle = p["battle"]
    tag = p["source"]["replay_tag"]
    out = {"tag": tag, "ok": False, "error": "", "forced": {}, "plays": 0}
    out.update(match_info(p))
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
    # E03 --tick-semantics tap: the delay is set before reset (rs_engine.make_battle).
    match_cfg = eng["cfg"] if delay is None else eng["cfg"].delayed(delay)
    b = rs_engine.make_battle(match_cfg, royalesim)
    b.reset(seed=0,
            decks=[[ids[c] for c in dealt[by_team[t]]] for t in (0, 1)],
            shuffle=0, start_tick=0, elixir_milli=[6000, 6000], tower_hp=None, spawns=[],
            forms=[[forms[by_team[t]][i] for i in perm[by_team[t]]] for t in (0, 1)],
            levels=[[levels[by_team[t]][i] for i in perm[by_team[t]]] for t in (0, 1)],
            tower_levels=[tower, tower])
    out["tower_level"] = tower
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
    refusals = []                       # [tick, side, card, reason, "send" | "run"]
    side_of = {rs_side[seat]: sides[seat] for seat in (0, 1)}     # RoyaleSim team -> side

    def refused(tick_, team_, card_, reason_, phase):
        forced[reason_] += 1
        if len(refusals) < REFUSAL_CAP:
            refusals.append([int(tick_), side_of.get(team_, str(team_)), card_, reason_, phase])

    card_of = {}

    def ran():
        """--tick-semantics tap: the delayed commands that ran in the last step. A play the
        engine accepted when it was sent and refused when it ran counts under its reason; a
        press is counted here, when it runs."""
        if delay is None:
            return
        rows = getattr(b, "step_commands_run", None)
        for row in (rows() if rows is not None else ()):
            r_tick, team, kind, what, code = row[:5]
            reason = royalesim.DEPLOY_REASONS[int(code)]
            if kind == "ability":
                if press_on:
                    plog.result(reason)
                continue
            if reason != "OK":
                if not card_of:
                    card_of.update((i, n) for n, i in ids.items())
                refused(r_tick, team, card_of.get(what, str(what)), reason.lower(), "run")

    tick, i, j = 0, 0, 0
    last = max([queue[-1][0] if queue else 0] + [q[0] for q in press_q])
    while i < len(queue) or j < len(press_q):
        t = queue[i][0] if i < len(queue) else press_q[j][0]
        if j < len(press_q):
            t = min(t, press_q[j][0])
        if t > tick:
            b.step([], t - tick)
            tick = t
            ran()
        state = json.loads(b.state_json())
        if state["game_over"]:
            break
        if not checked:              # the engine's own button order, where it names the cards
            for team in (0, 1):
                btns[team], differs = rs_presses.engine_order(state["players"][team], btns[team])
                plog.order_mismatch += int(differs)
            checked = True
        cmds, sent = [], []
        while i < len(queue) and queue[i][0] == t:
            _, seat, card, x, y = queue[i]
            team = rs_side[seat]
            hand = state["players"][team]["hand"]
            cid = ids[card]
            out["plays"] += 1
            if cid not in hand:
                why = ("hand_pending" if delay is not None and b.pending_commands(team)
                       else "hand")
                refused(t, team, card, why, "send")
            else:
                cmds.append((team, hand.index(cid), *rs_orient.to_engine(x, y, SUB)))
                sent.append((team, card))
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
                    if delay is None or reason != "OK":   # tap: counted when it runs
                        plog.result(reason)
                elif reason != "OK":
                    refused(t, sent[k][0], sent[k][1], reason.lower(), "send")
            ran()
    end = min(last + 200, 6000)
    if tick < end:
        b.step([], end - tick)
        ran()
    state = json.loads(b.state_json())
    if not state["game_over"]:
        b.step([], 6000)
        ran()
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
    out["refusals"] = refusals
    if press_on:
        out["presses"] = plog.to_json()
    return out


# ---------------------------------------------------------------------------
# totals and files
# ---------------------------------------------------------------------------

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


def error_kind(error) -> str:
    """How main counts a failed match: its error up to the first ':' ("orient", "card not in
    RoyaleSim", "exception KeyError", ...)."""
    return str(error or "").split(":")[0]


def totals(rows) -> dict:
    """A run's counts: matches, ran (ok), errors by error_kind (most common first),
    winner_agree / crowns_agree (matches that ran and agree), plays and refused (by reason,
    most common first) over the matches that ran, and three_crowns (matches that ran with a
    3-crown side, in the engine and in the recording)."""
    ok = [r for r in rows if r.get("ok")]
    errors = Counter(error_kind(r.get("error")) for r in rows if not r.get("ok"))
    refused = Counter()
    for r in ok:
        refused.update(r.get("forced") or {})
    return {
        "matches": len(rows), "ran": len(ok), "errors": dict(errors.most_common()),
        "winner_agree": sum(1 for r in ok if r.get("winner_agree")),
        "crowns_agree": sum(1 for r in ok if r.get("crowns_agree")),
        "plays": sum(int(r.get("plays") or 0) for r in ok),
        "refused": dict(refused.most_common()),
        "three_crowns": {
            "engine": sum(1 for r in ok if 3 in (r.get("engine_crowns") or {}).values()),
            "recorded": sum(1 for r in ok if 3 in (r.get("recorded_crowns") or {}).values())},
    }


def output_path(pair: str, presses: str = rs_presses.PRESS_OFF) -> Path:
    """reports/redrive/<pair>_royalesim.jsonl, or <pair>_royalesim_presses.jsonl under
    --presses on, so a run with presses does not overwrite the run without."""
    suffix = "_presses" if rs_presses.press_mode(presses) == rs_presses.PRESS_ON else ""
    return ROOT / "reports" / "redrive" / f"{pair}_royalesim{suffix}.jsonl"


def tick_path(path: Path, semantics: str = TICK_EXEC, delay: int = TAP_DELAY) -> Path:
    """`path` under exec; <stem>_tap<delay><suffix> under tap, so the runs sit side by side."""
    if tick_mode(semantics) == TICK_EXEC:
        return Path(path)
    path = Path(path)
    return path.with_name(f"{path.stem}_tap{tap_ticks(delay)}{path.suffix}")


def meta_path(out: Path) -> Path:
    """<out stem>.meta.json beside a run's output."""
    out = Path(out)
    return out.with_name(f"{out.stem}.meta.json")


def corpus_index(corpus: Path = CORPUS) -> list[tuple[str, str, str]]:
    """(replay_tag, game_mode, part file name) of every match in <corpus>/replays (needs
    pyarrow; reads the two columns only)."""
    import pyarrow.parquet as pq
    out = []
    for part in sorted((Path(corpus) / "replays").glob("part-*.parquet")):
        for r in pq.read_table(part, columns=["replay_tag", "game_mode"]).to_pylist():
            out.append((str(r["replay_tag"]), str(r["game_mode"] or ""), part.name))
    return out


def load_payloads(wanted, corpus: Path = CORPUS, parts=None) -> dict[str, str]:
    """{tag: payload_json text} of the wanted tags, with the match's events attached
    ("events", from actions/<part>) and source.replay_tag and source.game_mode set (needs
    pyarrow). parts: the part file names to read (None: every part, until all are found)."""
    import pyarrow.parquet as pq
    corpus = Path(corpus)
    wanted = set(wanted)
    payloads: dict[str, str] = {}
    for part in sorted((corpus / "replays").glob("part-*.parquet")):
        if parts is not None and part.name not in parts:
            continue
        acts = pq.read_table(corpus / "actions" / part.name,
                             columns=["replay_tag", "event_json"]).to_pylist()
        events = {}
        for a in acts:
            if a["replay_tag"] in wanted:
                events.setdefault(a["replay_tag"], []).append(json.loads(a["event_json"]))
        for r in pq.read_table(part, columns=["replay_tag", "game_mode",
                                              "payload_json"]).to_pylist():
            if r["replay_tag"] in wanted:
                payload = json.loads(r["payload_json"])
                payload["events"] = events.get(r["replay_tag"], [])
                source = payload.setdefault("source", {})
                source["replay_tag"] = r["replay_tag"]
                source.setdefault("game_mode", r["game_mode"] or "")
                payloads[r["replay_tag"]] = json.dumps(payload)
        if len(payloads) >= len(wanted):
            break
    return payloads


def sample_payloads(corpus: Path = CORPUS, n: int = 0, split: str = DEFAULT_SPLIT,
                    seed: int = 0, modes=MODES) -> tuple[list[str], list[str]]:
    """--sample N: (the tags picked, sample_tags; the payload texts found for them, in the same
    order). Reads the corpus's tags and modes, then only the parts that hold a picked tag."""
    index = corpus_index(corpus)
    tags = sample_tags(index, n, split, seed, modes)
    chosen = set(tags)
    found = load_payloads(chosen, corpus, {part for tag, _, part in index if tag in chosen})
    return tags, [found[t] for t in tags if t in found]


def read_payload_file(path: Path) -> list[str]:
    """The payload texts of a --payloads file: one JSON payload per non-empty line."""
    with Path(path).open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_payload_file(path: Path, texts) -> Path:
    """Write payload texts one per line (--save-payloads; read back by read_payload_file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps(json.loads(text)) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-drive IL_Replay matches through RoyaleSim and compare each ending with "
                    "the recording (see the module doc).")
    ap.add_argument("--pair", default=None,
                    help="the run's name. Without --sample / --payloads: read the match tags of "
                         "reports/redrive/<pair>.jsonl (default b2000), as before. Default name "
                         "under --sample: sample<N>_<split>_seed<S>; under --payloads: the "
                         "file's stem")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--presses", choices=rs_presses.PRESS_MODES, default=rs_presses.PRESS_OFF,
                    help="'off' (default): play_card events only, as before; 'on': also replay "
                         "activate_ability events as button presses (output "
                         "<pair>_royalesim_presses.jsonl)")
    ap.add_argument("--tick-semantics", choices=TICK_MODES, default=TICK_EXEC,
                    help="'exec' (default): IL_Replay ticks are execution ticks, each play runs "
                         "at its recorded tick, as before; 'tap': they are tap ticks, the battle "
                         "has command delay --tap-delay and each play is sent at its recorded "
                         "tick (output <name>_tap<delay>.jsonl)")
    ap.add_argument("--tap-delay", type=int, default=TAP_DELAY, metavar="TICKS",
                    help=f"--tick-semantics tap: the command delay (default {TAP_DELAY})")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="re-drive N corpus matches picked by tag hash (module doc) instead of "
                         "reading reports/redrive/<pair>.jsonl")
    ap.add_argument("--split", choices=(*SPLITS, SPLIT_ALL), default=DEFAULT_SPLIT,
                    help=f"--sample: the split to pick from (default {DEFAULT_SPLIT}; "
                         "tools/rs_rows.split_of)")
    ap.add_argument("--seed", type=int, default=0, help="--sample: the hash seed (default 0)")
    ap.add_argument("--modes", default=None,
                    help="--sample: game modes, comma-separated, or 'all' (default: the ranked "
                         "modes tools/rs_rows.py trains on)")
    ap.add_argument("--corpus", type=Path, default=CORPUS,
                    help="IL_Replay folder with replays/ and actions/ (default data/il_replay)")
    ap.add_argument("--payloads", type=Path, default=None, metavar="FILE",
                    help="re-drive the payloads of this JSON-lines file (--save-payloads writes "
                         "one) instead of reading the corpus; needs no pyarrow")
    ap.add_argument("--save-payloads", type=Path, default=None, metavar="FILE",
                    help="also write the payloads re-driven to FILE, one per line")
    ap.add_argument("--out", type=Path, default=None, metavar="FILE",
                    help="output JSON lines (default reports/redrive/<name>_royalesim*.jsonl); "
                         "the run's meta goes beside it as <stem>.meta.json")
    rs_engine.add_engine_args(ap)
    args = ap.parse_args()
    try:
        cfg = rs_engine.config_from_args(args).undelayed()
        ticks = tick_mode(args.tick_semantics)
        tap_delay = tap_ticks(args.tap_delay)
        modes = parse_modes(args.modes)
        if args.sample < 0:
            raise ValueError(f"--sample: a number of matches, got {args.sample}")
        if args.sample and args.payloads:
            raise ValueError("--sample and --payloads both pick the matches: give one")
        if args.payloads and not args.payloads.is_file():
            raise FileNotFoundError(f"--payloads {args.payloads} does not exist")
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    args.pair = resolve_pair(args.pair, args.sample, args.split, args.seed, args.payloads)
    rs_engine.set_active(cfg)
    run_cfg = match_config(cfg, ticks, tap_delay)
    fp = rs_engine.fingerprint(rs_engine.make_battle(run_cfg), cfg=run_cfg)
    print(f"engine: {run_cfg.describe()}\n        {rs_engine.summary(fp)}", flush=True)
    names = rs_cards.load()
    print(f"cards:  {rs_cards.summary(names)}", flush=True)
    print(f"presses: {rs_presses.describe_presses(args.presses)}", flush=True)
    print(f"ticks:  {describe_ticks(ticks, tap_delay)}", flush=True)
    if args.payloads:
        texts = read_payload_file(args.payloads)
        wanted_n = len(texts)
        source = {"kind": "payloads", "path": str(args.payloads),
                  "sha256": rs_cards.file_sha256(args.payloads)}
    elif args.sample:
        tags, texts = sample_payloads(args.corpus, args.sample, args.split, args.seed, modes)
        wanted_n = len(tags)
        source = {"kind": "sample", "n": args.sample, "split": args.split, "seed": args.seed,
                  "modes": "all" if modes is None else sorted(modes),
                  "corpus": str(args.corpus), "rule": SAMPLE_RULE}
    else:
        with (ROOT / "reports" / "redrive" / f"{args.pair}.jsonl").open(encoding="utf-8") as f:
            wanted = {json.loads(line)["tag"] for line in f}
        texts = list(load_payloads(wanted, args.corpus).values())
        wanted_n = len(wanted)
        source = {"kind": "pair", "tags": f"reports/redrive/{args.pair}.jsonl"}
    if args.save_payloads:
        write_payload_file(args.save_payloads, texts)
        print(f"payloads: {len(texts)} -> {args.save_payloads}", flush=True)
    print(f"{len(texts)} of {wanted_n} matches found", flush=True)
    t0 = time.time()
    job = partial(one, presses=args.presses)
    job = partial(job, tick_semantics=ticks, tap_delay=tap_delay)
    with ProcessPoolExecutor(args.workers, initializer=rs_engine.set_active,
                             initargs=(cfg.to_json(),)) as pool:
        rows = list(pool.map(job, texts, chunksize=8))
    seconds = time.time() - t0
    out = args.out or tick_path(output_path(args.pair, args.presses), ticks, tap_delay)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tot = totals(rows)
    ok = [r for r in rows if r["ok"]]
    plays = tot["plays"]
    print(f"RoyaleSim: {len(ok)} ran ({seconds:.0f}s), errors {tot['errors']}")
    print(f"  winner agree {tot['winner_agree'] / max(1, len(ok)):.1%}, "
          f"crowns agree {tot['crowns_agree'] / max(1, len(ok)):.1%}, "
          f"3-crown {tot['three_crowns']['engine']} vs recorded {tot['three_crowns']['recorded']}")
    print("  refused per 100 plays: " + ", ".join(
        f"{k} {100 * v / max(1, plays):.2f}" for k, v in tot["refused"].items()))
    presses = press_totals(ok)
    if args.presses == rs_presses.PRESS_ON:
        print(f"  presses: {rs_presses.describe_log(presses)}")
    meta = {"schema": META_SCHEMA, "tool": "tools/royalesim_redrive.py", "pair": args.pair,
            "source": source, "engine": run_cfg.to_json(), "fingerprint": fp,
            "cards": rs_cards.describe(names), "presses": args.presses,
            "tick_semantics": ticks, "tap_delay": tap_delay if ticks == TICK_TAP else 0,
            "seconds": round(seconds, 1), "totals": tot, "press_totals": presses,
            "output": str(out)}
    meta_path(out).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
