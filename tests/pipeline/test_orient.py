"""E07: tools/rs_orient.py, the seat from troop plays, and the three tools that use it
(tools/rs_rows.py, royalesim_redrive.py, rs_eval.py).

Pure tests need nothing. The tool tests drive the tools with conftest's fake royalesim
(royaleimitate unimportable; the card map comes from data/cards). The `engine`-marked tests
need the real royalesim, the `il_replay` one the corpus too.

Golden: on conftest's payload (the team at native low y) each tool sends the engine exactly
the commands of the legacy rule (seat from the median y of all the team's plays; coordinates
x 18, turned half a turn only when that median put the team in red, which it does not here)
and rs_rows labels the plays as that rule did.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    SUBTILE_PER_MILLITILE,
    catalogue_row,
    il_payload,
    il_replay_dir,
    il_replay_row,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = SUBTILE_PER_MILLITILE


def _load(name: str):
    """tools/<name>.py as the module `name`, the name the tools import it by, without
    putting tools/ on sys.path for the whole session."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rs_orient = _load("rs_orient")
rs_engine = _load("rs_engine")
rs_cards = _load("rs_cards")

#: troop_names of conftest's catalogue: its troops and its building, not its four spells.
TROOP = rs_orient.troop_names(DEFAULT_CATALOGUE)
#: IL_Replay key -> RoyaleSim name for conftest's two decks (as data/cards maps them).
NAMES = {"cannon": "Cannon", "fireball": "Fireball", "hog-rider": "HogRider",
         "ice-golem": "IceGolemite", "ice-spirit": "IceSpirits", "musketeer": "Musketeer",
         "skeletons": "Skeletons", "the-log": "Log", "knight": "Knight", "archers": "Archer",
         "giant": "Giant", "arrows": "Arrows", "valkyrie": "Valkyrie",
         "mini-pekka": "MiniPekka", "baby-dragon": "BabyDragon", "zap": "Zap"}

#: A match whose team puts most of its troops on the wrong half: skipped by every tool.
GARBAGE_PLAYS = ((100, "team", "musketeer", 3500, 7500), (160, "team", "hog-rider", 3500, 24_000),
                 (180, "team", "ice-golem", 14_500, 24_000),
                 (200, "opponent", "knight", 14_500, 24_500))
GARBAGE_DETAIL = "team at native low y, own-half share team 0.33 opponent 1.00"


def key_of(key: str) -> tuple[str, int]:
    """royalesim_redrive.base_key, without importing the tools."""
    for suffix, form in (("-ev1", 1), ("-hero", 2)):
        if key.endswith(suffix):
            return key[: -len(suffix)], form
    return key, 0


def side(n_low: int, n_high: int, card: str = "Knight") -> list[tuple]:
    """n_low plays at native low y and n_high at high y, all of `card`."""
    return ([(k, card, 9000, 7500) for k in range(n_low)]
            + [(100 + k, card, 9000, 24_500) for k in range(n_high)])


# ---------------------------------------------------------------------------
# rs_orient
# ---------------------------------------------------------------------------

def test_constants():
    assert (rs_orient.HALF_Y, rs_orient.MIN_SHARE, rs_orient.SKIP) == (16_000, 0.5, "orient")
    assert rs_orient.ANYWHERE == {"Miner", "GoblinDrill"}


def test_troop_names_are_troops_and_buildings_not_spells_or_tunnellers():
    assert TROOP == {"Cannon", "HogRider", "IceGolemite", "IceSpirits", "Musketeer",
                     "Skeletons", "Knight", "Archer", "Giant", "Valkyrie", "MiniPekka",
                     "BabyDragon"}
    cat = [catalogue_row("Miner", 3, "TROOP"), catalogue_row("GoblinDrill", 4, "BUILDING"),
           catalogue_row("Mirror", 1, "SPELL"), catalogue_row("Bats", 2, "troop"),
           ["Short", 3, 0]]
    assert rs_orient.troop_names(cat) == {"Bats"}


def test_team_at_low_y_plays_blue_with_full_shares():
    plays = rs_orient.plays_of(il_payload(), NAMES, key_of)
    assert [p[1] for p in plays[0]] == ["Musketeer", "HogRider", "Skeletons"]
    assert [p[1] for p in plays[1]] == ["Knight", "Arrows"]
    assert rs_orient.half_counts(plays, TROOP) == ((3, 0), (0, 1))     # Arrows: a spell
    ori = rs_orient.orient(plays, TROOP)
    assert ori == (0, (1.0, 1.0))
    assert ori.ok() and ori.seats == {0: 0, 1: 1}
    assert ori.describe() == "team at native low y, own-half share team 1.00 opponent 1.00"


def test_team_at_high_y_plays_red_and_its_plays_stay_on_its_own_half():
    low = rs_orient.plays_of(il_payload(), NAMES, key_of)
    high = rs_orient.plays_of(il_payload(team_low=False), NAMES, key_of)
    assert rs_orient.half_counts(high, TROOP) == ((0, 3), (1, 0))
    ori = rs_orient.orient(high, TROOP)
    assert ori == (1, (1.0, 1.0)) and ori.ok() and ori.seats == {0: 1, 1: 0}
    for s in (0, 1):                    # the payload turned the board; nothing turns it back
        assert [p[2:] for p in high[s]] == [(18_000 - x, 32_000 - y) for _, _, x, y in low[s]]
    team = [rs_orient.to_engine(x, y, SUB) for _, _, x, y in high[0]]
    assert team and all(ey >= rs_orient.HALF_Y * SUB for _, ey in team)     # red's half


def test_spells_do_not_vote():
    plays = {0: [(10, "Musketeer", 3500, 7500), (20, "Fireball", 9000, 26_000),
                 (30, "Fireball", 9000, 26_000), (40, "Log", 9000, 25_000)],
             1: [(15, "Knight", 14_500, 24_500)]}
    ys = sorted(p[3] for p in plays[0])
    assert ys[len(ys) // 2] >= 16_000           # the old median rule put this team in red
    assert rs_orient.orient(plays, TROOP) == (0, (1.0, 1.0))


def test_tunnellers_and_unmapped_cards_do_not_vote():
    troop = rs_orient.troop_names([*DEFAULT_CATALOGUE, catalogue_row("Miner", 3, "TROOP"),
                                   catalogue_row("GoblinDrill", 4, "BUILDING")])
    assert troop == TROOP
    plays = {0: [(10, "HogRider", 3500, 14_500), (20, "Miner", 3500, 27_000),
                 (30, "GoblinDrill", 14_500, 27_000), (40, "Miner", 3500, 27_000),
                 (50, None, 9000, 30_000)],
             1: [(15, "Knight", 14_500, 24_500)]}
    assert rs_orient.half_counts(plays, troop) == ((1, 0), (0, 1))
    assert rs_orient.orient(plays, troop) == (0, (1.0, 1.0))


def test_the_midline_belongs_to_red():
    plays = {0: [(1, "Knight", 9000, 15_999), (2, "Knight", 9000, 16_000)], 1: []}
    assert rs_orient.half_counts(plays, TROOP) == ((1, 1), (0, 0))


def test_no_troop_play_keeps_the_team_blue_and_the_opponent_alone_can_decide():
    spells = {0: [(1, "Fireball", 9000, 26_000)], 1: [(2, "Zap", 9000, 7000)]}
    assert rs_orient.orient(spells, TROOP) == (0, (1.0, 1.0))
    assert rs_orient.orient({0: [], 1: []}, TROOP) == (0, (1.0, 1.0))
    opponent_low = {0: [(1, "Fireball", 9000, 7000)],
                    1: [(2, "Knight", 3500, 7500), (3, "Giant", 3500, 6000)]}
    assert rs_orient.orient(opponent_low, TROOP) == (1, (1.0, 1.0))
    as_lists = [opponent_low[0], opponent_low[1]]                  # a sequence works too
    assert rs_orient.orient(as_lists, TROOP) == (1, (1.0, 1.0))


@pytest.mark.parametrize("plays, expected", [
    # the team's troops mostly on the wrong half (a vote tie keeps it blue)
    ({0: side(2, 3), 1: side(0, 1, "HogRider")}, (0, (0.4, 1.0))),
    # both sides on the same half: a tie, so blue, and the opponent is all off its half
    ({0: side(1, 0), 1: side(1, 0, "HogRider")}, (0, (1.0, 0.0))),
    # the opponent's troops mostly on the wrong half, the team clearly high
    ({0: side(0, 4), 1: side(1, 2, "HogRider")}, (1, (1.0, 1 / 3))),
])
def test_garbage_is_skipped(plays, expected):
    ori = rs_orient.orient(plays, TROOP)
    assert ori == expected and not ori.ok()


def test_the_share_threshold_is_inclusive():
    at = rs_orient.orient({0: side(1, 1), 1: side(0, 5)}, TROOP)
    assert at.share == (0.5, 1.0) and at.ok()
    under = rs_orient.orient({0: side(2, 3), 1: side(0, 5)}, TROOP)
    assert under.share[0] == pytest.approx(0.4) and not under.ok()
    assert under.ok(min_share=0.35)


def test_pocket_heavy_matches_are_kept():
    # 7 of 10 troops on the own half, 3 in a pocket after a tower fell: kept by default,
    # and the old 0.9 bar would have skipped it
    ori = rs_orient.orient({0: side(7, 3), 1: side(0, 6)}, TROOP)
    assert ori.share == (0.7, 1.0) and ori.ok() and not ori.ok(min_share=0.9)


def test_to_engine_is_the_one_mapping():
    assert rs_orient.to_engine(3500, 7500, SUB) == (63_000, 135_000)
    assert rs_orient.to_engine(0, 31_999, 1) == (0, 31_999)


def _two_v_two() -> dict:
    payload = il_payload(tag="2v2")
    players = payload["battle"]["team"]["players"]
    players.append(dict(players[0]))
    return payload


def test_survey_counts_seats_skips_and_raw_halves():
    payloads = [il_payload(), il_payload(tag="high", team_low=False),
                il_payload(tag="garbage", plays=GARBAGE_PLAYS), _two_v_two()]
    out = rs_orient.survey(payloads, NAMES, TROOP, key_of)
    assert (out["matches"], out["not_1v1"]) == (3, 1)
    assert (out["team_low"], out["team_high"], out["skipped"], out["no_troop_plays"]) == \
        (2, 1, 1, 0)
    assert out["team_troops"] == [3 + 0 + 1, 0 + 3 + 2]          # low, high per match summed
    assert out["opponent_troops"] == [0 + 1 + 0, 1 + 0 + 1]
    assert out["min_share"] == {"1.00": 2, "0.90-0.99": 0, "0.80-0.89": 0, "0.50-0.79": 0,
                                "<0.50": 1}
    assert rs_orient.survey(payloads, NAMES, TROOP, key_of, min_share=0.3)["skipped"] == 0


# ---------------------------------------------------------------------------
# the tools, on the fake engine
# ---------------------------------------------------------------------------

#: The legacy rule on conftest's payload with the fake's hands (blue 0-3, red 8-11): the
#: team's median play y is 7500, so the team was blue and nothing turned. Each command is
#: (team, hand slot, x * 18, y * 18), at the tick it was sent.
LEGACY_SENT = [
    (160, [(1, 0, 14_500 * SUB, 24_500 * SUB)]),     # opponent Knight (8), red slot 0
    (240, [(0, 2, 3_500 * SUB, 14_500 * SUB)]),      # team HogRider (2), blue slot 2
    (300, [(1, 3, 3_500 * SUB, 8_000 * SUB)]),       # opponent Arrows (11), red slot 3
]
#: (tick, side, action) of rs_rows' play rows under the legacy rule: card id + 2 and the
#: tile of own(x * 18, y * 18, team) (blue: (x, 32000 - y); red: (18000 - x, y)).
LEGACY_PLAY_ROWS = [(100, 1, (7, 3, 24)), (160, -1, (10, 3, 24)), (240, 1, (4, 3, 17)),
                    (300, -1, (13, 14, 8)), (420, 1, (8, 9, 27))]

#: Hands for seat_royalesim: every card a side plays in conftest's payload is in its hand.
HOG_HAND, HOG_NEXT = [5, 2, 6, 0], 1       # Musketeer, HogRider, Skeletons, Cannon; Fireball
OPP_HAND, OPP_NEXT = [8, 11, 9, 10], 12    # Knight, Arrows, Archer, Giant; Valkyrie


def legacy_team_seat(payload: dict) -> int:
    """The rule the tools had: the median y of all the team's plays."""
    ys = sorted(e["coordinates"]["native_world_units"]["y"] for e in payload["events"]
                if e["kind"] == "play_card" and e["side"] == "team") or [0]
    return 0 if ys[len(ys) // 2] < 16_000 else 1


def sent(battle) -> list[tuple[int, list[tuple]]]:
    """(tick, commands) of every step of a FakeBattle that carried commands."""
    out, tick = [], int((battle.reset_kwargs or {}).get("start_tick") or 0)
    for call in battle.calls:
        if call.name == "step":
            cmds, n = call.args
            if cmds:
                out.append((tick, list(cmds)))
            tick += int(n)
    return out


def turned(commands: list[tuple[int, list[tuple]]]) -> list[tuple[int, list[tuple]]]:
    """The same commands for the other seat on the board turned half a turn."""
    return [(t, [(1 - team, slot, 18_000 * SUB - x, 32_000 * SUB - y)
                 for team, slot, x, y in cmds]) for t, cmds in commands]


def play_rows(rows) -> list[tuple]:
    return [(r["tick"], r["side"], tuple(r["action"])) for r in rows if r["action"][0]]


def by_side(rows, team_seat: int) -> list[tuple]:
    """Every row with its IL_Replay side in place of its engine seat, sorted."""
    out = []
    for r in rows:
        il = "team" if (0 if r["side"] == 1 else 1) == team_seat else "opponent"
        out.append((il, r["tick"], tuple(r["action"]), r["hand"].tolist(), int(r["next"]),
                    r["opp_seen"].tolist(), r["opp_last"].tolist(),
                    r["opp_last_age"].tolist(), r["towers"].tolist(), r["scalars"].tolist(),
                    r["unit_ids"].tolist(), r["outcome"], r["trusted"], r["won"], r["hog"]))
    return sorted(out)


@pytest.fixture(autouse=True)
def clean_engine(monkeypatch):
    """No HASTYCR_* engine or card-map variable from the PC, and no active config left."""
    for var in (rs_engine.CONFIG_ENV, *rs_engine.ENV.values(), rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rs_engine, "_ACTIVE", None)


@pytest.fixture
def tools(monkeypatch):
    """rs_rows, royalesim_redrive and rs_eval as they import each other, caches empty."""
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rs_rows = importlib.import_module("rs_rows")
    redrive = importlib.import_module("royalesim_redrive")
    rs_eval = importlib.import_module("rs_eval")
    assert rs_rows.rs_orient is rs_orient and redrive.rs_orient is rs_orient
    assert rs_eval.rs_orient is rs_orient and rs_rows.rs_engine is rs_engine
    monkeypatch.setattr(rs_rows, "F", None)
    monkeypatch.setattr(redrive, "_ENGINE", {})
    return types.SimpleNamespace(rows=rs_rows, redrive=redrive, eval=rs_eval)


@pytest.fixture
def seat_royalesim(monkeypatch):
    """install(hog_team): a fake royalesim whose `hog_team` holds HOG_HAND and the other
    team OPP_HAND, so every play of conftest's payload is in hand."""
    def install(hog_team: int):
        def states(battle):
            hand = {hog_team: (HOG_HAND, HOG_NEXT), 1 - hog_team: (OPP_HAND, OPP_NEXT)}
            return state_dict(battle.tick, players=[
                player_dict(t, hand=hand[t][0], next_card=hand[t][1]) for t in (0, 1)])
        mod = make_fake_royalesim(states=states)
        monkeypatch.setitem(sys.modules, "royalesim", mod)
        return mod
    return install


def _cfg():
    return rs_engine.EngineConfig.defaults()


def _play(rs_eval, payload: dict):
    return rs_eval.play((json.dumps(payload), "", "pro", 0, "", 0, 0, False, False, False,
                         _cfg().to_json()))


def test_tools_have_no_median_rule_or_half_turn_left():
    for name in ("rs_rows.py", "rs_eval.py", "royalesim_redrive.py"):
        text = (TOOLS / name).read_text(encoding="utf-8")
        assert "rs_orient.orient(" in text and "rs_orient.to_engine(" in text, name
        assert "ys[len(ys) // 2]" not in text, name
        assert "18_000 - x, 32_000 - y" not in text, name


def test_rows_golden_with_the_team_at_low_y(fake_royalesim, tools):
    RR = tools.rows
    payload = il_payload()
    assert legacy_team_seat(payload) == 0
    rows, why = RR.rows_for(payload, RR._setup(_cfg()), _cfg())
    assert why == "ok"
    match = fake_royalesim.battles[-1]
    assert sent(match) == LEGACY_SENT
    assert play_rows(rows) == LEGACY_PLAY_ROWS
    decks = match.reset_kwargs["decks"]
    assert set(decks[0]) == set(range(8)) and set(decks[1]) <= set(range(8, 16))


def test_redrive_golden_with_the_team_at_low_y(fake_royalesim, tools):
    out = tools.redrive.one(json.dumps(il_payload()), engine=_cfg())
    assert out["ok"] and out["error"] == "" and out["forced"] == {"hand": 2}
    assert sent(fake_royalesim.battles[-1]) == LEGACY_SENT
    assert tools.redrive._ENGINE["troop"] == TROOP


def test_eval_golden_with_the_team_at_low_y(fake_royalesim, tools):
    out = _play(tools.eval, il_payload())
    assert out["hog_team"] == 0 and "skipped" not in out
    assert sent(fake_royalesim.battles[-1]) == LEGACY_SENT


def test_rows_with_the_team_at_high_y_are_the_same_rows_from_red(seat_royalesim, tools):
    RR = tools.rows
    runs = {}
    for hog_team, team_low in ((0, True), (1, False)):
        mod = seat_royalesim(hog_team)
        rows, why = RR.rows_for(il_payload(team_low=team_low), RR._setup(_cfg()), _cfg())
        assert why == "ok"
        runs[team_low] = rows, mod.battles[-1]
    (low_rows, low), (high_rows, high) = runs[True], runs[False]
    assert set(low.reset_kwargs["decks"][0]) == set(range(8))     # the team plays blue ...
    assert set(high.reset_kwargs["decks"][1]) == set(range(8))    # ... and here red
    assert len(sent(low)) == 5 and sent(high) == turned(sent(low))
    # the team's troops land on red's own half, the opponent's Knight on blue's
    assert all(y >= 16_000 * SUB for _, cmds in sent(high) for team, _, _, y in cmds
               if team == 1)
    knight = [c for t, cmds in sent(high) if t == 160 for c in cmds]
    assert knight == [(0, 0, 3_500 * SUB, 7_500 * SUB)]
    # every row is the same from the player's own side; only `side` follows the seat
    assert by_side(high_rows, 1) == by_side(low_rows, 0)
    assert play_rows(high_rows) == [(t, -s, a) for t, s, a in play_rows(low_rows)]
    assert all(r["trusted"] for r in high_rows)


def test_redrive_and_eval_with_the_team_at_high_y(seat_royalesim, tools):
    out, cmds, ev = {}, {}, {}
    for hog_team, team_low in ((0, True), (1, False)):
        mod = seat_royalesim(hog_team)
        tools.redrive._ENGINE.clear()                 # the card map's battle, on this fake
        out[team_low] = tools.redrive.one(json.dumps(il_payload(team_low=team_low)),
                                          engine=_cfg())
        cmds[team_low] = sent(mod.battles[-1])
        ev[team_low] = _play(tools.eval, il_payload(team_low=team_low))
        cmds[team_low, "eval"] = sent(mod.battles[-1])
    assert out[True]["ok"] and out[True]["forced"] == {}
    assert out[False] == out[True]
    assert len(cmds[True]) == 5 and cmds[False] == turned(cmds[True])
    assert (ev[True]["hog_team"], ev[False]["hog_team"]) == (0, 1)
    assert {**ev[False], "hog_team": 0} == ev[True]
    assert cmds[False, "eval"] == turned(cmds[True, "eval"]) and len(cmds[True, "eval"]) == 5


def test_rows_skip_garbage_as_orient(fake_royalesim, tools, tmp_path, monkeypatch):
    RR = tools.rows
    garbage = il_payload(tag="garbage-1", plays=GARBAGE_PLAYS)
    assert RR.rows_for(garbage, RR._setup(_cfg()), _cfg()) == (None, "orient")
    assert len(fake_royalesim.battles) == 1                        # only _setup's
    monkeypatch.setattr(RR, "OUT", RR.OUT)
    monkeypatch.setattr(RR, "LIVE", RR.LIVE)
    batch, n_rows, skipped = RR.build(5, [json.dumps(garbage), json.dumps(il_payload())],
                                      False, str(tmp_path), _cfg().to_json())
    assert (batch, skipped) == (5, {"orient": 1}) and n_rows > 0
    assert (tmp_path / "shard-0005.npz").is_file()


def test_redrive_skips_garbage_as_an_orient_error(fake_royalesim, tools):
    out = tools.redrive.one(json.dumps(il_payload(tag="garbage-1", plays=GARBAGE_PLAYS)),
                            engine=_cfg())
    assert not out["ok"] and out["plays"] == 0
    assert out["error"] == f"orient: {GARBAGE_DETAIL}"
    assert out["error"].split(":")[0] == "orient"                 # how main() counts errors
    assert len(fake_royalesim.battles) == 1                        # only the card map's


def test_eval_skips_garbage_and_counts_it(fake_royalesim, tools):
    RE = tools.eval
    skip = _play(RE, il_payload(tag="garbage-1", plays=GARBAGE_PLAYS))
    assert skip == {"tag": "garbage-1", "mode": "pro", "skipped": "orient",
                    "detail": GARBAGE_DETAIL}
    assert len(fake_royalesim.battles) == 1                        # only _setup's
    played = _play(RE, il_payload())
    rows, skipped = RE.split_results([None, skip, played, skip])
    assert rows == [played] and skipped == Counter({"orient": 2})
    assert RE.split_results([None, played]) == ([played], Counter())


def test_survey_cli_on_a_tiny_corpus(fake_royalesim, tmp_path, monkeypatch, capsys):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    (tmp_path / "replays").mkdir()
    payloads = [il_payload(), il_payload(tag="high", team_low=False),
                il_payload(tag="garbage", plays=GARBAGE_PLAYS)]
    pq.write_table(pa.Table.from_pylist([il_replay_row(p) for p in payloads]),
                   tmp_path / "replays" / "part-00000.parquet")
    assert rs_orient.main(["--corpus", str(tmp_path), "--parts", "0", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["parts"], out["matches"], out["team_low"], out["team_high"], out["skipped"]) \
        == (1, 3, 2, 1, 1)
    assert rs_orient.main(["--corpus", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "3 1v1 matches in 1 parts" in text and "skipped as 'orient'" in text
    assert rs_orient.main(["--corpus", str(tmp_path / "nowhere")]) == 1


# ---------------------------------------------------------------------------
# the real engine and corpus
# ---------------------------------------------------------------------------

@pytest.mark.engine
def test_troop_names_on_the_real_catalogue():
    import royalesim
    cat = rs_engine.catalogue(rs_engine.EngineConfig.load(), royalesim)
    fields = list(getattr(royalesim, "CATALOGUE_FIELDS", ()))
    if fields:
        assert fields[rs_orient.KIND_COLUMN] == "card_kind"
    troop = rs_orient.troop_names(cat)
    assert {"Cannon", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons"} <= troop
    assert not troop & {"Fireball", "Log", "Arrows", "Zap", "Mirror", "Miner", "GoblinDrill"}
    # every tunneller (catalogue placement code 5, royalegym Placement.TUNNEL) is in ANYWHERE
    assert {row[0] for row in cat if row[1] == 5} <= rs_orient.ANYWHERE


@pytest.mark.engine
def test_the_real_engine_plays_both_orientations_alike(monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(TOOLS))
    redrive = importlib.import_module("royalesim_redrive")
    monkeypatch.setattr(redrive, "_ENGINE", {})
    cfg = rs_engine.EngineConfig.load()
    plain = {"team_forms": {}, "opponent_forms": {}, "opponent_levels": 11}
    low = redrive.one(json.dumps(il_payload(**plain)), engine=cfg)
    high = redrive.one(json.dumps(il_payload(team_low=False, **plain)), engine=cfg)
    assert low["ok"] and high["ok"], (low, high)
    assert high["forced"] == low["forced"] and high["plays"] == low["plays"] == 5


@pytest.mark.engine
@pytest.mark.il_replay
def test_survey_reads_the_corpus(monkeypatch):
    pytest.importorskip("pyarrow")
    import royalesim
    monkeypatch.syspath_prepend(str(TOOLS))
    redrive = importlib.import_module("royalesim_redrive")
    files = rs_cards.corpus_parts(il_replay_dir(), parts=1)
    troop = rs_orient.troop_names(rs_engine.catalogue(rs_engine.EngineConfig.load(), royalesim))
    out = rs_orient.survey(rs_cards.iter_payloads(files), rs_cards.load(), troop,
                           redrive.base_key)
    assert out["matches"] > 0
    assert out["team_low"] + out["team_high"] == out["matches"]
    assert sum(out["min_share"].values()) == out["matches"]
    assert 0 <= out["skipped"] <= out["matches"]
