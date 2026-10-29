"""Training rows from IL_Replay re-driven through RoyaleSim, in the same column
format as sim/replay_rows.py, so sim/train_tokens.py trains on either.

    .venvs/royalegym/Scripts/python.exe tools/rs_rows.py --others 20000 --workers 7

Why a second generator: on the same 2,000 matches RoyaleSim's endings agree
with the recording far more often than ours (winner 62.6% vs 49.5%, exact
crowns 49.0% vs 24.2%; reports/redrive_fidelity.md), and it refuses almost no
recorded play, so the boards a row shows are much closer to what the player
really saw.

Frame: every row is from the acting player's view, as the bottom player with
y growing toward its own king (the same as sim/replay_rows.py). RoyaleSim's
blue owns low y, so blue is reflected in y and red in x; both are reflections,
so left and right mean the same thing to either seat.

Seat: tools/rs_orient.py decides from troop plays which IL_Replay side owns native
low y; that side plays blue, and every coordinate goes to the engine as recorded.
A match where either side has under 90% of its troop plays on its own half is
skipped as "orient" (the old median test plus half turn cancelled out and checked
nothing). Matches with the team at native low y give the same rows as before; the
one exception, a team whose median play (spells included) was past the river, used
to be put in red and turned, and now plays blue with the same own-frame rows.

Ids: units are RoyaleSim `unit_type` + 2, cards are catalogue ids + 2 (0 pad,
1 unknown). `data/rs_rows/vocab.json` records both lists, and the engine config
and fingerprint they came from (tools/rs_engine.py), the card table, the row
schema, the observation delay and wait_keep_at_build (vocab_record).
tools/rs_checkpoint.py refuses a checkpoint whose lists or card table are not
the engine's it is loaded into.

Engine: every Battle comes from rs_engine.make_battle under data/engine.json
(card table, levels), with no command delay: plays are sent at their recorded
ticks. `--card-table default` (or HASTYCR_CARD_TABLE=default) runs the engine's
own table, as this tool did before rs_engine.

Cards: IL_Replay card keys map to RoyaleSim names through tools/rs_cards.py
(data/cards/: royaleimitate 0.2.15's replay_cards.json, vendored and hash-checked,
plus void, rune-giant and spirit-empress). Matches with an unmapped card are
skipped. HASTYCR_CARD_EXTRAS=0 gives the 119-key map this tool used before.
vocab.json records the map under "card_map".

Outcome and trust as in sim/replay_rows.py: tower swing over the next 10 s;
rows after the first refused play, or after the engine has taken more crowns
than the recording allows, are marked untrusted.

Opponent memory (E08; opp_seen, opp_last, opp_last_age): tools/rs_memory.PublicMemory, every
play dated on the tick it ran (the recorded IL_Replay tick). `--mem-lag L` (default BOARD_LAG,
1): all rows of a tick read one view, taken before that tick's plays are recorded, holding the
plays with exec_tick + L <= tick. A seat no longer sees the other seat's play of the same tick
(tokens-17); 0 and 1 give the same rows here, and a larger L stands for perception lag.
`--mem-lag legacy` keeps the old order (each play recorded as it is handled, the memory read at
each row) and gives rows byte-identical to the rows before E08. vocab.json records the lag as
obs_delay.opp_lag and the order as "opp_memory": "exec"; legacy rows write opp_lag 0 and no
"opp_memory", as before (memory_vocab).

Units (E16): `--units types` (the default) writes RoyaleSim unit_type + 2 and the engine's
flags, as before. `--units vision` writes the class of data/vision/classes.json
(tools/vision_classes.py) + 2, flying and building from the class table (what a screen can
know), and each unit's form (0 base, 0.5 evo, 1 hero) in one more column, "unit_form"
(MAX_UNITS float16). sim/train_tokens.py does not read that column yet: the legacy model keeps
UF = 7 features, and the v2 schema (E14) moves the form into unit_feats. vocab.json then lists
the classes as "units" (n_units = classes + 2) and adds "units_mode": "vision",
"vision_classes_digest" and "extra_columns". --live-compat is a third kind of unit token and
does not combine with --units vision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import operator
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mode

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from royalesim_redrive import CORPUS, base_key, deal_order  # noqa: E402
import rs_cards  # noqa: E402
import rs_engine  # noqa: E402
import rs_memory  # noqa: E402
import rs_orient  # noqa: E402
import vision_classes  # noqa: E402

OUT = ROOT / "data" / "rs_rows"
MAX_UNITS, UF = 48, 7
HORIZON, WAIT_EVERY, CHUNK = 200, 20, 10
HOG26 = frozenset({"Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits",
                   "Musketeer", "Skeletons", "Log"})
MODES = {"Ranked", "Ladder", "1v1 Battle", "Grand Challenge", "Classic Challenge"}
F = None   # field indexes, filled on first use
# --live-compat: unit tokens carry only what the live memory reader provides.
# The reader reports the card that put a unit down (a Witch's skeleton reads as
# the Witch), hit points and max hit points, side and position; not RoyaleSim's
# unit_type, not a deploy timer, not a flying flag. Rows built with this flag
# use: id = creating card, flying / building from that card's catalogue row,
# and log(max hp) in place of the deploy flag, so a model trained on them sees
# in training exactly what it will see live.
LIVE = False
CARD_FLY: dict = {}
CARD_BUILDING: dict = {}

#: --units (module doc): what a unit token names when the rows are not --live-compat.
UNITS_TYPES, UNITS_VISION = "types", "vision"
UNIT_MODES = (UNITS_TYPES, UNITS_VISION)
UNITS = UNITS_TYPES
#: --units vision: the class table (vision_classes.ClassTable) and its lookup of this engine's
#: unit_types_json() (vision_classes.UnitLookup); _setup fills both.
VCT = None
VC = None
#: The columns --units vision adds to FIELDS.
VISION_FIELDS = ("unit_form",)

#: --mem-lag (module doc): a lag in ticks, or MEM_LEGACY for the order before E08.
MEM_LEGACY = "legacy"
MEM_LAG = rs_memory.BOARD_LAG
#: vocab.json "opp_memory" of rows whose memory is dated at execution (absent: legacy).
OPP_MEMORY_EXEC = "exec"


def unit_mode(value, live=False) -> str:
    """--units as rows_for / build take it: one of UNIT_MODES. ValueError otherwise, and for
    "vision" together with --live-compat (live), which is a unit token of its own."""
    if value not in UNIT_MODES:
        raise ValueError(f"--units: one of {', '.join(UNIT_MODES)}, got {value!r}")
    if live and value == UNITS_VISION:
        raise ValueError("--units vision and --live-compat are two different unit tokens (the "
                         "vision class, the creating card); pick one")
    return value


def mem_lag_value(value):
    """--mem-lag as rows_for, build and rs_eval.play take it: a lag in ticks (an int >= 0; a
    string of digits is read as one) or MEM_LEGACY. ValueError otherwise."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text == MEM_LEGACY:
            return MEM_LEGACY
        try:
            value = int(text, 10)
        except ValueError:
            raise ValueError(f"--mem-lag: a lag in ticks (>= 0) or {MEM_LEGACY!r}, "
                             f"got {value!r}") from None
    if isinstance(value, bool):
        raise ValueError(f"--mem-lag: a lag in ticks (>= 0) or {MEM_LEGACY!r}, got {value!r}")
    try:
        lag = operator.index(value)
    except TypeError:
        raise ValueError(f"--mem-lag: a lag in ticks (>= 0) or {MEM_LEGACY!r}, "
                         f"got {value!r}") from None
    if lag < 0:
        raise ValueError(f"--mem-lag must be >= 0 (a negative lag sees plays before they run), "
                         f"got {lag}")
    return lag


def mem_lag_arg(text: str):
    """argparse type of --mem-lag."""
    try:
        return mem_lag_value(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_mem_lag_arg(ap: argparse.ArgumentParser, help_text: str | None = None) -> None:
    """--mem-lag TICKS|legacy, default rs_memory.BOARD_LAG; tools/rs_eval.py adds it too."""
    ap.add_argument("--mem-lag", type=mem_lag_arg, default=rs_memory.BOARD_LAG,
                    metavar="TICKS|legacy", help=help_text or (
                        "opponent memory: a row at tick t sees the plays that ran by "
                        f"t - max(1, TICKS) (default {rs_memory.BOARD_LAG}: what the board "
                        "shows); 'legacy': the order before E08, byte-identical rows that "
                        "show the other seat's same-tick plays"))


def describe_memory(mem_lag) -> str:
    """One line for the console."""
    lag = mem_lag_value(mem_lag)
    if lag == MEM_LEGACY:
        return ("legacy (each play recorded as it is handled; a row sees the other seat's "
                "same-tick plays)")
    return (f"dated at execution, lag {lag} (a row at tick t sees the plays that ran by "
            f"t - {max(rs_memory.BOARD_LAG, lag)})")


def memory_vocab(mem_lag) -> dict:
    """vocab_record keywords for --mem-lag. Legacy: none (opp_lag 0, no "opp_memory": the
    vocab.json of the rows before E08). Otherwise the lag the views in rows_for amount to,
    max(BOARD_LAG, lag) (the view is taken before the tick's plays are recorded, so 0 gives
    the rows of 1), and opp_memory OPP_MEMORY_EXEC."""
    lag = mem_lag_value(mem_lag)
    if lag == MEM_LEGACY:
        return {}
    return {"opp_lag": max(rs_memory.BOARD_LAG, lag), "opp_memory": OPP_MEMORY_EXEC}


def split_of(tag: str) -> str:
    h = int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 5 else "val" if h < 10 else "train"


def _setup(engine=None):
    """(royalesim, IL_Replay key -> card name, card name -> id, catalogue rows) under
    `engine` (an rs_engine.EngineConfig or its to_json(); None = rs_engine.active()).
    The card map is rs_cards.load(). Also sets F, after checking royalesim.ENTITY_FIELDS
    (rs_engine.field_index), and under --units vision (UNITS) VCT, the class table, and VC,
    its lookup of this Battle's unit_types_json()."""
    global F, VC, VCT
    import royalesim
    names = rs_cards.load()
    b = rs_engine.make_battle(rs_engine.resolve(engine).undelayed(), royalesim)
    cat = json.loads(b.catalogue_json())
    ids = {row[0]: i for i, row in enumerate(cat)}
    F = rs_engine.field_index(royalesim)
    for i, row in enumerate(cat):
        CARD_FLY[i] = bool(row[5])
        CARD_BUILDING[i] = row[8] == "BUILDING"
    if UNITS == UNITS_VISION:
        VCT = vision_classes.load()
        VC = VCT.lookup(json.loads(b.unit_types_json()))
    return royalesim, names, ids, cat


def own(x_sub, y_sub, team, sub):
    """RoyaleSim sub-tiles -> the acting player's frame, millitiles, bottom = high y."""
    x, y = x_sub / sub, y_sub / sub
    return (x, 32_000 - y) if team == 0 else (18_000 - x, y)


def towers_vec(state, team):
    out = []
    for t in (team, 1 - team):
        hp, mx = state["players"][t]["tower_hp"], state["players"][t]["tower_max_hp"]
        # slots: 0 king, 1 world-left, 2 world-right. Red is reflected in x, so
        # its own left is world right.
        lr = (1, 2) if team == 0 else (2, 1)
        out += [hp[lr[0]] / max(1, mx[lr[0]]), hp[lr[1]] / max(1, mx[lr[1]]),
                hp[0] / max(1, mx[0])]
    return np.asarray(out, dtype=np.float32)


def obs_row(state, team, sub, mem, tick, live=None, units=None):
    """One row of `team`'s view. `mem` is the opponent memory, {"seen", "last"} as
    rs_memory.PublicMemory.view() gives it. `live` / `units` default to LIVE / UNITS; with
    units "vision" (and not live) the row also has "unit_form", and VC must be set (_setup)."""
    live = LIVE if live is None else live
    units = unit_mode(UNITS if units is None else units)
    vision = not live and units == UNITS_VISION
    if vision and VC is None:
        raise RuntimeError("--units vision rows need the class lookup: set UNITS = 'vision' "
                           "before _setup(), which fills VC")
    forms = np.zeros(MAX_UNITS, dtype=np.float16) if vision else None
    ents = [e for e in state["entities"] if e[F["tower_slot"]] < 0 and e[F["hp"]] > 0]
    king = next((e for e in state["entities"]
                 if e[F["team"]] == team and e[F["tower_slot"]] == 0), None)
    # A destroyed king (a lookahead copy can end the battle) has no entity;
    # sort from where it stands instead.
    kx, ky = ((king[F["x"]], king[F["y"]]) if king is not None
              else (9000 * 18, (3000 if team == 0 else 29000) * 18))
    ents.sort(key=lambda e: (e[F["x"]] - kx) ** 2 + (e[F["y"]] - ky) ** 2)
    ids = np.zeros(MAX_UNITS, dtype=np.int16)
    feats = np.zeros((MAX_UNITS, UF), dtype=np.float16)
    for k, e in enumerate(ents[:MAX_UNITS]):
        ox, oy = own(e[F["x"]], e[F["y"]], team, sub)
        if live:
            card = e[F["card_id"]]
            ids[k] = card + 2
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]),
                        1.0 if CARD_FLY.get(card) else 0.0,
                        1.0 if CARD_BUILDING.get(card) else 0.0,
                        float(np.log1p(max(0, e[F["max_hp"]]))) / 10.0)
        elif vision:
            # The vision class of the unit type, flags from the class table; the form goes
            # to its own column until the v2 schema (E14) makes room in unit_feats.
            unit = e[F["unit_type"]]
            ids[k] = VC.token(unit)
            fly, building = VC.flags(unit)
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]), fly, building,
                        1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
            forms[k] = VC.form_feature(unit)
        else:
            ids[k] = e[F["unit_type"]] + 2
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]), 1.0 if e[F["flying"]] else 0.0,
                        1.0 if e[F["kind"]] == 1 else 0.0,
                        1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
    p = state["players"][team]
    hand = [c + 2 for c in p["hand"][:4]] + [0] * (4 - len(p["hand"][:4]))
    opp = 1 - team
    seen = sorted(mem["seen"][opp])[:8]
    last = mem["last"][opp][-4:]
    t_ms = tick * 50
    row = {
        "tick": tick, "side": 1 if team == 0 else -1, "hand_exact": True,
        "unit_ids": ids, "unit_feats": feats, "towers": towers_vec(state, team),
        "scalars": np.asarray([p["elixir_milli"] / 10_000, min(1.0, t_ms / 300_000),
                               1.0 if t_ms >= 120_000 else 0.0,
                               1.0 if t_ms >= 180_000 else 0.0], dtype=np.float32),
        "hand": np.asarray(hand, dtype=np.int16),
        "next": (p["next_card"] + 2) if p.get("next_card", -1) >= 0 else 0,
        "opp_seen": np.asarray(seen + [0] * (8 - len(seen)), dtype=np.int16),
        "opp_last": np.asarray([c for c, _ in last] + [0] * (4 - len(last)), dtype=np.int16),
        "opp_last_age": np.asarray([min(1.0, (tick - t) / 400) for _, t in last]
                                   + [1.0] * (4 - len(last)), dtype=np.float16),
    }
    if vision:
        row["unit_form"] = forms
    return row


def rows_for(payload: dict, ctx, engine=None, *, mem_lag=None):
    """Rows of one match. `engine` as in _setup; the battle runs without command delay,
    since the recorded ticks are the ticks the plays executed. `mem_lag` as --mem-lag
    (None: MEM_LAG): how the opponent memory of a row is dated (module doc)."""
    lag = mem_lag_value(MEM_LAG if mem_lag is None else mem_lag)
    legacy = lag == MEM_LEGACY
    royalesim, names, ids, cat = ctx
    sub = royalesim.SUBTILE_PER_MILLITILE
    battle = payload["battle"]
    decks, forms, levels = [], [], []
    for side in ("team", "opponent"):
        players = battle[side]["players"]
        if len(players) != 1:
            return None, "not 1v1"
        deck, form, level = [], [], []
        for card in players[0]["deck"]:
            key, f = base_key(card["card_key"])
            name = names.get(key)
            if name is None or name not in ids:
                return None, "card"
            deck.append(name)
            form.append(f)
            level.append(int(card.get("level") or 11))
        decks.append(deck)
        forms.append(form)
        levels.append(level)
    events = sorted((e for e in payload["events"] if e["kind"] == "play_card"),
                    key=lambda e: (e["replay_tick_20hz"], e["source_index"]))
    plays = {0: [], 1: []}
    for e in events:
        seat = 0 if e["side"] == "team" else 1
        name = names.get(base_key(e["card_key"])[0])
        u = e["coordinates"]["native_world_units"]
        plays[seat].append((int(e["replay_tick_20hz"]), name, int(u["x"]), int(u["y"])))
    # The seat from troop plays: the side at native low y plays blue, and every
    # coordinate goes to the engine as recorded (tools/rs_orient.py).
    ori = rs_orient.orient(plays, rs_orient.troop_names(cat))
    if not ori.ok():
        return None, rs_orient.SKIP
    team_seat = ori.team_seat
    dealt = []
    for seat in (0, 1):
        order = deal_order(decks[seat], [c for _, c, _, _ in plays[seat]])
        if order is None:
            return None, "deal"
        dealt.append(order)
    perm = [[decks[s].index(c) for c in dealt[s]] for s in (0, 1)]
    rs = {0: team_seat, 1: 1 - team_seat}
    seat_of = {rs[0]: 0, rs[1]: 1}
    tower = max(1, min(16, mode([lv for s in levels for lv in s])))
    b = rs_engine.make_battle(rs_engine.resolve(engine).undelayed(), royalesim)
    b.reset(seed=0, decks=[[ids[c] for c in dealt[seat_of[t]]] for t in (0, 1)],
            shuffle=0, start_tick=0, elixir_milli=[6000, 6000], tower_hp=None, spawns=[],
            forms=[[forms[seat_of[t]][i] for i in perm[seat_of[t]]] for t in (0, 1)],
            levels=[[levels[seat_of[t]][i] for i in perm[seat_of[t]]] for t in (0, 1)],
            tower_levels=[tower, tower])
    rec_crowns = {rs[0]: battle["team"]["crowns"], rs[1]: battle["opponent"]["crowns"]}
    result = str(battle.get("result", "")).lower()
    won_team = 1.0 if result in ("victory", "win") else 0.0 if result in ("defeat", "loss") else 0.5
    won = {rs[0]: won_team, rs[1]: 1.0 - won_team}
    queue = sorted([(t, rs[s], c, x, y) for s in (0, 1) for (t, c, x, y) in plays[s]])
    play_ticks = {0: [q[0] for q in queue if q[1] == 0], 1: [q[0] for q in queue if q[1] == 1]}
    mem = rs_memory.PublicMemory()
    rows, track = [], []
    cut = 10 ** 9
    tick, i = 0, 0
    end = min((queue[-1][0] if queue else 0) + HORIZON, 6000)
    while tick <= end:
        state = json.loads(b.state_json())
        if state["game_over"]:
            track.append((tick, towers_vec(state, 0)))
            break
        track.append((tick, towers_vec(state, 0)))
        for t in (0, 1):
            if state["players"][t]["crowns"] > rec_crowns[t] and cut > tick:
                cut = tick
        # E08: one view for every row of this tick, taken before this tick's plays are
        # recorded. Legacy reads the memory at each row instead, after the plays handled so
        # far, which is the dict the rows before E08 read. (Decision-time rows, E11, take
        # the view at their observation tick.)
        known = None if legacy else mem.view(tick, lag)
        due = []
        while i < len(queue) and queue[i][0] <= tick:
            due.append(queue[i])
            i += 1
        cmds = []
        acting = set()
        for (_, team, card, x, y) in due:
            acting.add(team)
            cid = ids[card]
            hand = state["players"][team]["hand"]
            ex, ey = rs_orient.to_engine(x, y, sub)
            ox, oy = own(ex, ey, team, sub)
            row = obs_row(state, team, sub, mem.view(tick, 0) if legacy else known, tick)
            row["action"] = (cid + 2, int(min(17, max(0, ox // 1000))), int(min(31, max(0, oy // 1000))))
            rows.append(row)
            if cid in hand:
                cmds.append((team, hand.index(cid), ex, ey))
            elif cut > tick:
                cut = tick
            mem.record(team, cid + 2, tick)        # the recorded tick is the tick it ran
        if tick % WAIT_EVERY == 0:
            for team in (0, 1):
                soon = any(tick < t <= tick + 10 for t in play_ticks[team])
                if team not in acting and not soon and tick > 0:
                    row = obs_row(state, team, sub, mem.view(tick, 0) if legacy else known, tick)
                    row["action"] = (0, 0, 0)
                    rows.append(row)
        nxt = queue[i][0] if i < len(queue) else end + 1
        step = max(1, min(nxt, (tick // WAIT_EVERY + 1) * WAIT_EVERY) - tick)
        res = b.step(cmds, step)
        for r in res:
            if royalesim.DEPLOY_REASONS[r[1]] != "OK" and cut > tick:
                cut = tick
        tick += step
    ticks = np.asarray([t for t, _ in track])
    tv = np.stack([v for _, v in track])            # blue-frame [own L,R,K, enemy L,R,K]
    blue_total, red_total = tv[:, :3].sum(1), tv[:, 3:].sum(1)
    for row in rows:
        team = 0 if row["side"] == 1 else 1
        a = int(np.searchsorted(ticks, row["tick"]))
        z = int(np.searchsorted(ticks, row["tick"] + HORIZON))
        a, z = min(a, len(ticks) - 1), min(z, len(ticks) - 1)
        own_lost = (blue_total if team == 0 else red_total)
        opp_lost = (red_total if team == 0 else blue_total)
        row["outcome"] = float((opp_lost[a] - opp_lost[z]) - (own_lost[a] - own_lost[z]))
        row["trusted"] = row["tick"] + HORIZON <= cut
        row["won"] = won[team]
        row["hog"] = frozenset(decks[seat_of[team]]) == HOG26
    return rows, "ok"


FIELDS = ("tick", "side", "hand_exact", "unit_ids", "unit_feats", "towers", "scalars",
          "hand", "next", "opp_seen", "opp_last", "opp_last_age", "outcome", "trusted",
          "won", "hog")


#: vocab.json "schema": the row layout of these shards (obs_row and FIELDS above).
SCHEMA = "rs_rows/legacy"


def vocab_record(cat, unit_types, cfg, fp, card_map, live, *, version=None, schema=SCHEMA,
                 obs_delay=0, obs_delay_jitter=0, opp_lag=0, wait_keep_at_build=None,
                 opp_memory=None, classes=None):
    """vocab.json: the id maps of the rows and how they were made.

    The keys it had before E04 come first, unchanged: n_cards, n_units, live_compat, cards
    (catalogue name -> id + 2), units (unit type -> id + 2), engine, version (`version`,
    else the fingerprint's), engine_config, engine_fingerprint, card_map. Then:

      card_table          what the Battle reported (fp["card_table"]; None before 0.1.42)
      schema              the row layout (SCHEMA)
      obs_delay           {"ticks", "jitter", "opp_lag"}: how far before a play's execution
                          its row is observed (0 here: rows at the tick it executes), and
                          opp_lag, how long after a play runs a row's memory shows it
                          (memory_vocab: 0 for legacy rows, else at least BOARD_LAG)
      wait_keep_at_build  the share of wait rows the builder kept; None: no thinning at
                          build, every WAIT_EVERY-tick wait row is in the shards
      wait_every          ticks between wait-row samples

    and only when given (so vocab.json of rows built as before E08 / E16 is unchanged):

      opp_memory          `opp_memory`, OPP_MEMORY_EXEC: plays dated at execution, one
                          view per tick (E08); absent: the legacy order
      units_mode, vision_classes_digest, extra_columns
                          with `classes` (a vision_classes.ClassTable, --units vision):
                          "vision", the table's digest, VISION_FIELDS; n_units and units
                          then count and list the classes (class name -> id + 2)

    sim/train_tokens.py copies all of it into every checkpoint's meta, and
    tools/rs_checkpoint.load_policy checks cards, units and card_table against the engine
    a checkpoint is loaded into.
    """
    vision = classes is not None
    if vision:
        unit_mode(UNITS_VISION, live)
    rec = {
        "n_cards": len(cat) + 2,
        "n_units": ((len(cat) + 2) if live else (len(classes) + 2) if vision
                    else (len(unit_types) + 2)),
        "live_compat": live,
        "cards": {row[0]: i + 2 for i, row in enumerate(cat)},
        "units": ({c.name: c.id + 2 for c in classes} if vision
                  else {u: i + 2 for i, u in enumerate(unit_types)}),
        "engine": "royalesim", "version": fp.get("version", "?") if version is None else version,
        "engine_config": cfg.to_json(), "engine_fingerprint": fp, "card_map": card_map,
        "card_table": fp.get("card_table"),
        "schema": schema,
        "obs_delay": {"ticks": int(obs_delay), "jitter": int(obs_delay_jitter),
                      "opp_lag": int(opp_lag)},
        "wait_keep_at_build": None if wait_keep_at_build is None else float(wait_keep_at_build),
        "wait_every": WAIT_EVERY,
    }
    if opp_memory is not None:
        rec["opp_memory"] = str(opp_memory)
    if vision:
        rec["units_mode"] = UNITS_VISION
        rec["vision_classes_digest"] = classes.digest
        rec["extra_columns"] = list(VISION_FIELDS)
    return rec


def build(batch_id, payload_texts, live=False, out_dir=None, engine=None, *, units=None,
          mem_lag=None):
    """One shard, shard-<batch_id>.npz in `out_dir` (else OUT), from IL_Replay payload texts.
    `units` / `mem_lag` as --units / --mem-lag, kept as UNITS / MEM_LAG for this process
    (None: keep the module's); main passes both to every worker, as it passes `live`."""
    global LIVE, OUT, UNITS, MEM_LAG
    units = UNITS if units is None else units
    unit_mode(units, live)
    lag = MEM_LAG if mem_lag is None else mem_lag_value(mem_lag)
    LIVE, UNITS, MEM_LAG = live, units, lag
    if out_dir:
        OUT = Path(out_dir)
    ctx = _setup(engine)
    rows, tags, skipped = [], [], {}
    for text in payload_texts:
        p = json.loads(text)
        tag = p["source"]["replay_tag"]
        try:
            r, why = rows_for(p, ctx, engine)
        except Exception as exc:
            r, why = None, f"error {type(exc).__name__}"
        if r is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        rows += r
        tags += [tag] * len(r)
    if rows:
        fields = FIELDS + (VISION_FIELDS if UNITS == UNITS_VISION and not LIVE else ())
        data = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in fields}
        act = np.asarray([r["action"] for r in rows], dtype=np.int16)
        data["act_card"], data["act_x"], data["act_y"] = act[:, 0], act[:, 1], act[:, 2]
        data["match"] = np.asarray(tags)
        data["split"] = np.asarray([split_of(t) for t in tags])
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"shard-{batch_id:04d}.npz", **data)
    return batch_id, len(rows), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--others", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--limit-parts", type=int, default=0)
    ap.add_argument("--live-compat", action="store_true",
                    help="unit tokens limited to what the live reader provides")
    ap.add_argument("--units", choices=UNIT_MODES, default=UNITS_TYPES,
                    help="unit tokens: RoyaleSim unit types (default, as before) or the vision "
                         "classes of data/vision/classes.json, plus a unit_form column")
    add_mem_lag_arg(ap)
    ap.add_argument("--out", type=Path, default=None)
    rs_engine.add_engine_args(ap)
    args = ap.parse_args()
    try:
        cfg = rs_engine.config_from_args(args).undelayed()
        unit_mode(args.units, args.live_compat)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    rs_engine.set_active(cfg)
    global OUT, LIVE, UNITS, MEM_LAG
    LIVE = args.live_compat
    UNITS, MEM_LAG = args.units, args.mem_lag
    if args.out:
        OUT = args.out
    import pyarrow.parquet as pq
    royalesim, names, ids, cat = _setup(cfg)
    vocab_battle = rs_engine.make_battle(cfg, royalesim)
    fp = rs_engine.fingerprint(vocab_battle, royalesim, cfg)
    print(f"engine: {cfg.describe()}\n        {rs_engine.summary(fp)}", flush=True)
    card_map = rs_cards.describe(names)
    print(f"cards:  {rs_cards.summary(names)}", flush=True)
    print(f"memory: {describe_memory(MEM_LAG)}", flush=True)
    if UNITS == UNITS_VISION:
        print(f"units:  vision classes, {VCT.describe()}", flush=True)
        if VC.unknown:
            print(f"        {len(VC.unknown)} engine unit types have no class and read as "
                  f"unknown (token 1): {', '.join(VC.unknown[:12])}"
                  f"{' ...' if len(VC.unknown) > 12 else ''}", flush=True)
    t0 = time.time()
    hog, others = [], []
    parts = sorted((CORPUS / "replays").glob("part-*.parquet"))
    if args.limit_parts:
        parts = parts[:args.limit_parts]
    for part in parts:
        for r in pq.read_table(part, columns=["replay_tag", "game_mode", "payload_json"]).to_pylist():
            if r["game_mode"] not in MODES:
                continue
            p = json.loads(r["payload_json"])
            sides = []
            ok = True
            for side in ("team", "opponent"):
                pl = p["battle"][side]["players"]
                if len(pl) != 1:
                    ok = False
                    break
                sides.append(frozenset(names.get(base_key(c["card_key"])[0]) for c in pl[0]["deck"]))
            if not ok or any(None in s for s in sides):
                continue
            p.setdefault("source", {})["replay_tag"] = r["replay_tag"]
            text = json.dumps(p)
            if HOG26 in sides:
                hog.append(text)
            elif len(others) < args.others:
                others.append(text)
    picked = hog + others
    print(f"selected {len(hog):,} Hog 2.6 + {len(others):,} others ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    unit_types = json.loads(vocab_battle.unit_types_json())
    (OUT / "vocab.json").write_text(json.dumps(vocab_record(
        cat, unit_types, cfg, fp, card_map, LIVE,
        version=getattr(royalesim, "__version__", "?"),
        classes=VCT if UNITS == UNITS_VISION else None,
        **memory_vocab(MEM_LAG))), encoding="utf-8")
    batches = [picked[i:i + args.batch] for i in range(0, len(picked), args.batch)]
    total, skipped = 0, {}
    with ProcessPoolExecutor(args.workers, initializer=rs_engine.set_active,
                             initargs=(cfg.to_json(),)) as pool:
        futs = [pool.submit(build, k, b, LIVE, str(OUT), cfg.to_json(), units=UNITS,
                            mem_lag=MEM_LAG)
                for k, b in enumerate(batches)]
        for n, f in enumerate(as_completed(futs), 1):
            bid, rows, sk = f.result()
            total += rows
            for k, v in sk.items():
                skipped[k] = skipped.get(k, 0) + v
            if n % 10 == 0 or n == len(batches):
                print(f"  {n}/{len(batches)} shards, {total:,} rows, skipped {skipped} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {total:,} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
