"""N02: the UNKNOWN safety net (docs/NEW_CARDS_PLAN.md 2.2), scripts/vision/events.py rule 6.

An enemy track no rule explains becomes OppPlay(card=-1) behind EventConfig.unknown_plays,
OFF by default, charged an expected cost (OppPlay.cost); scripts/vision/types.py gives OppPlay
an optional cost that old recordings load without; scripts/vision_play.py keeps card -1 out of
the opponent's known cards and counts UNKNOWN plays; tools/vision_replay.py gets
--unknown-plays and counts them. The golden check: with the switch off the deriver reports
exactly what it always did.

How the token model receives card -1 (checked in the code, nothing run):
* opponent memory, legacy and v2 rows alike: record_plays writes card + 2 = 1, rs_memory.UNKNOWN
  ("1 is the unknown card", tools/rs_memory.py PAD, UNKNOWN, TOKEN_OFFSET); obs_row puts
  sorted(seen)[:8] in opp_seen and the last 4 plays in opp_last (tools/rs_rows.py obs_row), and
  sim/model_tokens.py embeds both with card_emb, whose padding_idx is 0: token 1 is a real,
  attended token, the one perception noise trains (opponent_plays.unknown_p, 0.02 in
  data/noise/default_prior.json, tools/perception_noise.py play_seen).
* the v2 opponent elixir: rs_rows / rs_eval skip a token-1 play (not paid, exactness kept); live,
  public_state.OpponentElixir pays its expected cost and calls the count not exact, as the plan
  asks (tests/pipeline/test_public_state_unknown.py).

Synthetic tracks and tables from tests/pipeline/test_events.py; no detector, engine or device.
"""
from __future__ import annotations

import functools
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import events as EV  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from tests.pipeline.test_events import (EMPRESS, HEAL, KNIGHT, N01_IDS, WITCH, deriver,  # noqa: E402
                                        started, tk)

E, O = VT.ENEMY_TEAM, VT.OWN_TEAM
GS, DOC = "Goblinstein", "GoblinsteinDoctor"        # classes no card of test_events.tables() has


def script():
    """(tick, tracks) per frame: a Knight; a two-unit card the tables do not know (Goblinstein
    and its doctor, 2 ticks apart); the same card in the other lane, lost for 15 ticks and found
    again under a new uid; a lone Golemite (its Golem never seen); a Witch; and an unexplained
    unit of OURS."""
    gs4 = tk(4, GS, E, 14500, 9000, 230, 231)
    return [
        (101, [tk(1, "Knight", E, 9000, 8000, 100, 101)]),
        (201, [tk(2, GS, E, 3500, 9000, 200, 201)]),
        (203, [tk(2, GS, E, 3500, 9000, 200, 203), tk(3, DOC, E, 3800, 9300, 202, 203)]),
        (231, [gs4]),
        (235, [replace(gs4, last_seen_tick=235)]),
        (251, [tk(6, GS, E, 14500, 9500, 250, 251)]),
        (301, [tk(5, "Golemite", E, 3000, 5000, 300, 301)]),
        (401, [tk(7, "Witch", E, 9000, 8000, 400, 401)]),
        (501, [tk(8, "Golemite", O, 9000, 25000, 500, 501)]),
    ]


def run(config=None):
    d = deriver(config=config)
    return d, [(tick, d.update(tracks, tick).opp_plays) for tick, tracks in script()]


KNIGHT_PLAY = {"card": KNIGHT, "tick": 100, "exec_tick": 100, "x_mt": 9000.0, "y_mt": 8000.0,
               "kind": "unit", "uids": [1]}
WITCH_PLAY = {"card": WITCH, "tick": 400, "exec_tick": 400, "x_mt": 9000.0, "y_mt": 8000.0,
              "kind": "unit", "uids": [7]}


def unknown_play(tick, x, y, uids, cost=EV.UNKNOWN_PLAY_COST):
    return {"card": -1, "tick": tick, "exec_tick": tick, "x_mt": float(x), "y_mt": float(y),
            "kind": "unit", "uids": uids, "cost": cost}


def dicts(frames):
    return {tick: [p.to_dict() for p in plays] for tick, plays in frames}


# ---------------------------------------------------------------------------
# golden: off by default, and then exactly the old reports
# ---------------------------------------------------------------------------
def test_off_by_default_the_deriver_reports_what_it_always_did():
    assert EV.EventConfig().unknown_plays is False and EV.EventConfig().unknown_cost == 4
    d, frames = run()
    assert dicts(frames) == {101: [KNIGHT_PLAY], 201: [], 203: [], 231: [], 235: [], 251: [],
                             301: [], 401: [WITCH_PLAY], 501: []}
    assert [p.to_dict() for p in d.opp_plays()] == [KNIGHT_PLAY, WITCH_PLAY]
    st = d.opp_stats
    assert (st.plays, st.merged, st.reacquired, st.children, st.unknown_plays) == (2, 0, 0, 0, 0)
    assert st.unknown == {GS: 3, DOC: 1, "Golemite": 1}     # an unknown track is never re-acquired
    assert {u: d.how_of(u) for u in range(1, 9)} == {
        1: "play", 2: "unknown", 3: "unknown", 4: "unknown", 5: "unknown", 6: "unknown", 7: "play",
        8: "unknown"}
    assert [d.card_of(u) for u in range(1, 9)] == [KNIGHT, None, None, None, None, None, WITCH, None]
    # the default config, given or not, and an explicit False are the same run
    for config in (EV.EventConfig(), EV.EventConfig(unknown_plays=False)):
        d2, frames2 = run(config)
        assert dicts(frames2) == dicts(frames) and d2.opp_stats == d.opp_stats


def test_unknown_plays_report_the_unexplained_enemy_units_and_change_nothing_else():
    d, frames = run(EV.EventConfig(unknown_plays=True))
    assert dicts(frames) == {
        101: [KNIGHT_PLAY],
        201: [unknown_play(200, 3500, 9000, [2])],
        203: [],                                   # the doctor, 2 ticks and 424 mt away: merged
        231: [unknown_play(230, 14500, 9000, [4])],    # 30 ticks after the first: another play
        235: [], 251: [],                          # uid 6 is uid 4 re-acquired
        301: [unknown_play(300, 3000, 5000, [5])],     # the lone Golemite
        401: [WITCH_PLAY],
        501: []}                                   # ours: never an opponent play
    off = run()[1]
    named = [p.to_dict() for _, plays in frames for p in plays if p.card >= 0]
    assert named == [p.to_dict() for _, plays in off for p in plays]
    hist = d.opp_plays()
    assert [(p.card, p.tick, p.uids, p.cost) for p in hist] == [
        (KNIGHT, 100, [1], None), (-1, 200, [2, 3], 4), (-1, 230, [4, 6], 4), (-1, 300, [5], 4),
        (WITCH, 400, [7], None)]
    st = d.opp_stats
    assert (st.plays, st.unknown_plays, st.merged, st.reacquired, st.children) == (2, 3, 1, 1, 0)
    assert st.unknown == {GS: 2, DOC: 1, "Golemite": 1}
    assert st.summary()["unknown_plays"] == 3
    assert {u: d.how_of(u) for u in range(1, 9)} == {
        1: "play", 2: "unknown_play", 3: "merged", 4: "unknown_play", 5: "unknown_play",
        6: "reacquired", 7: "play", 8: "unknown"}
    # the deriver does not know their card: the board keeps naming them by the class table
    assert [d.card_of(u) for u in range(2, 7)] == [None] * 5
    tracks = [tk(u, cls, E, 0, 0, 0, 0) for u, cls in ((1, "Knight"), (2, GS), (6, GS), (7, "Witch"))]
    assert d.track_cards(tracks) == run()[0].track_cards(tracks) == {1: KNIGHT, 7: WITCH}


def test_the_expected_cost_and_the_sight_lag_are_the_configs():
    cfg = EV.EventConfig(unknown_plays=True, unknown_cost=6, unit_sight_lag_ticks=3)
    d = deriver(config=cfg)
    (p,) = d.update([tk(2, GS, E, 3500, 9000, 200, 201)], 201).opp_plays
    assert (p.card, p.tick, p.exec_tick, p.kind, p.cost) == (VT.UNKNOWN_CARD, 200, 197, "unit", 6)
    assert EV.EventConfig(unknown_cost=0).unknown_cost == 0
    assert EV.EventConfig(unknown_cost=5.0).unknown_cost == 5.0


@pytest.mark.parametrize("bad", [-1, 2.5, True, "4", float("nan"), float("inf"), None])
def test_the_expected_cost_must_be_a_whole_elixir_cost(bad):
    with pytest.raises(ValueError):
        EV.EventConfig(unknown_cost=bad)


def test_an_unknown_play_is_row_token_1_in_the_public_memory():
    rs_memory = pytest.importorskip("tools.rs_memory")
    _, frames = run(EV.EventConfig(unknown_plays=True))
    mem = rs_memory.PublicMemory()
    for _, plays in frames:
        EV.record_plays(mem, plays)
    view = mem.view(500, 0)
    assert rs_memory.UNKNOWN == 1 == VT.UNKNOWN_CARD + EV.TOKEN_OFFSET
    assert view["last"][E] == [(KNIGHT + 2, 100), (1, 200), (1, 230), (1, 300), (WITCH + 2, 400)]
    assert view["seen"][E] == {KNIGHT + 2, 1, WITCH + 2}


# ---------------------------------------------------------------------------
# OppPlay.cost: optional, old frames load, and the Spirit Empress's form cost (N01's follow-up)
# ---------------------------------------------------------------------------
def test_opp_play_cost_is_optional_and_old_frames_load_unchanged():
    old = {"card": 17, "tick": 1200, "exec_tick": 1198, "x_mt": 3500.0, "y_mt": 13000.0,
           "kind": "unit", "uids": [7]}
    p = VT.OppPlay.from_dict(old)
    assert p.cost is None and p.to_dict() == old          # written exactly as before
    unk = VT.OppPlay(card=VT.UNKNOWN_CARD, tick=200, exec_tick=200, cost=4)
    assert unk.to_dict()["cost"] == 4 and VT.OppPlay.from_dict(unk.to_dict()) == unk
    vf = VT.VisionFrame(t_capture=1.0, tick=201, events=VT.Events([unk, p]), screen=VT.SCREEN_BATTLE)
    assert VT.VisionFrame.from_json(vf.to_json()) == vf
    line = json.loads(json.dumps({"v": 1, "t_capture": 2.0, "tick": 1234,
                                  "events": {"opp_plays": [old], "own_landings": []}}))
    assert VT.VisionFrame.from_dict(line).events.opp_plays == [
        VT.OppPlay(17, 1200, 1198, 3500.0, 13000.0, "unit", [7])]


def test_a_spirit_empress_play_carries_its_forms_cost():
    tb = EV.EventTables.from_names(EV.load_spawn_children(), N01_IDS)
    d = started(tb)
    (mounted,) = d.update([tk(1, "MergeMaiden_Mounted", E, 9000, 9000, 200, 201)], 201).opp_plays
    (normal,) = d.update([tk(2, "MergeMaiden_Normal", E, 3500, 9000, 600, 601)], 601).opp_plays
    (heal,) = d.update([tk(3, "HealSpirit", E, 14500, 12000, 900, 901)], 901).opp_plays
    assert [(p.card, p.cost) for p in (mounted, normal, heal)] == [(EMPRESS, 6), (EMPRESS, 3), (HEAL, None)]
    assert mounted.to_dict()["cost"] == 6 and "cost" not in heal.to_dict()
    assert [p.cost for p in d.opp_plays()] == [6, 3, None]


# ---------------------------------------------------------------------------
# vision_play: card -1 stays out of the known cards; the battle record counts UNKNOWN plays
# ---------------------------------------------------------------------------
def test_vision_play_keeps_unknown_plays_out_of_the_opponents_cards(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    from scripts import vision_play as VP
    from tests.pipeline import test_vision_play as TVP

    args = VP.build_parser().parse_args([])
    assert args.unknown_plays is False
    assert VP.build_parser().parse_args(["--unknown-plays"]).unknown_plays is True
    unknown = VT.OppPlay(card=VT.UNKNOWN_CARD, tick=104, exec_tick=104, x_mt=4000.0, y_mt=8000.0, cost=4)
    giant = VT.OppPlay(card=10, tick=106, exec_tick=106, x_mt=14000.0, y_mt=8000.0)
    script = TVP.standard_script(opp_at={104: [unknown], 106: [giant]})

    def go(sub, unknown_plays):
        if unknown_plays:
            monkeypatch.setattr(VP, "VisionPlay", functools.partial(VP.VisionPlay, unknown_plays=True))
        captured = {}
        res = TVP.run_script(tmp_path / sub, script,
                             decider=TVP.make_decider(TVP.FakePolicy(()), captured=captured))
        return res, captured

    res, captured = go("on", True)
    views = dict(captured["views"])
    assert views[104]["last"][0] == [(1, 104)] and views[104]["seen"][0] == {1}
    assert views[106]["last"][0] == [(1, 104), (12, 106)] and views[106]["seen"][0] == {1, 12}
    known = dict(captured["opp_cards"])
    assert known[104] == [] and known[106] == [10]
    [end] = res.records["battle_end"]
    assert (end["opp_plays"], end["opp_unknown"], end["opp_cards"]) == (2, 1, ["Giant"])
    monkeypatch.undo()
    res, _ = go("off", False)                    # the default runner: no new key in its records
    [end] = res.records["battle_end"]
    assert "opp_unknown" not in end and end["opp_cards"] == ["Giant"]


# ---------------------------------------------------------------------------
# vision_replay --unknown-plays
# ---------------------------------------------------------------------------
def test_vision_replay_counts_unknown_plays(tmp_path):
    """test_latency_and_replay's recording, with tables that know no Knight: its enemy Knight
    (battle frames 12 on) is unexplained; with --unknown-plays it is one UNKNOWN play."""
    pytest.importorskip("numpy")
    from scripts.vision.capture import FileReplaySource
    from scripts.vision.labels import load_class_table
    from scripts.vision.tracker import Tracker
    from tests.pipeline import test_latency_and_replay as TLR

    VR = TLR.VR
    session = TLR.write_recording(tmp_path / "rec")
    names = VR.Names(load_class_table())
    tables = EV.EventTables({2: {"HogRider": 1}})            # no card puts down a Knight

    def replay(unknown):
        perception = VR.ReplayPerception(
            hud_reader=TLR.ReplayHud(), tracker=Tracker(own_classes={"HogRider"}),
            deriver=EV.EventDeriver(tables, EV.EventConfig(unknown_plays=unknown), own_deck=range(8)),
            geometry=TLR.GEOMETRY, detector=TLR.ReplayDetector(), names=names)
        taps, _ = VR.read_recording_records(session)
        return VR.Replay(FileReplaySource(session), perception, taps=taps, names=names,
                         card_ids={n: i for i, n in TLR.NAMES.items()}, card_names=TLR.NAMES,
                         live_names=TLR.NAMES, unknown_plays=unknown).run()

    off = replay(False)["events"]
    assert (off["opp_plays"], off["opp_cards"], off["own_landed"]) == (0, {}, 1)
    assert "opp_unknown" not in off and "unknown_tracks" not in off
    s = replay(True)
    on = s["events"]
    assert (on["opp_plays"], on["opp_cards"], on["opp_unknown"], on["own_landed"]) == (
        1, {"UNKNOWN": 1}, 1, 1)
    assert on["unknown_tracks"] == {"Knight": 1}
    assert any(line.startswith("  UNKNOWN plays (--unknown-plays): 1; enemy tracks no card "
                               "explained: Knight 1") for line in VR.format_summary(s))
    json.dumps(s)


def test_vision_replay_unknown_plays_flag(tmp_path, monkeypatch, capsys):
    pytest.importorskip("numpy")
    from tests.pipeline import test_latency_and_replay as TLR

    assert TLR.VR.build_parser().parse_args(["rec"]).unknown_plays is False
    rc, _, s = TLR._replay_main(tmp_path, monkeypatch, "default")
    assert rc == 0 and "unknown_plays" not in s and "opp_unknown" not in s["events"]
    assert "UNKNOWN plays on" not in capsys.readouterr().out
    rc, _, s = TLR._replay_main(tmp_path, monkeypatch, "unknown", "--unknown-plays")
    assert rc == 0 and s["unknown_plays"] == {"cost": EV.UNKNOWN_PLAY_COST}
    assert (s["events"]["opp_unknown"], s["events"]["unknown_tracks"]) == (0, {})
    assert "unknown    UNKNOWN plays on (card -1, 4 elixir each)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --unknown-cost (review fix): the gate (N14) can set the charge without a code change
# ---------------------------------------------------------------------------
def test_event_config_from_the_unknown_flags():
    pytest.importorskip("numpy")
    from scripts import vision_play as VP

    parse = VP.build_parser().parse_args
    args = parse([])
    assert (args.unknown_plays, args.unknown_cost) == (False, None)
    assert VP.event_config(args) == EV.EventConfig.live()           # the live guards (GUARDS)
    # the deriver's defaults but for our own side's drop-in rule (events.LIVE_OWN)
    assert VP.event_config(parse(["--legacy-opp-plays"])) == EV.EventConfig(**EV.LIVE_OWN)
    assert VP.event_config(parse(["--unknown-plays"])) == EV.EventConfig.live(unknown_plays=True)
    cfg = VP.event_config(parse(["--unknown-plays", "--unknown-cost", "0"]))
    assert (cfg.unknown_plays, cfg.unknown_cost) == (True, 0)
    assert VP.event_config(parse(["--unknown-plays", "--unknown-cost", "6"])).unknown_cost == 6
    with pytest.raises(ValueError, match="applies to --unknown-plays"):
        VP.event_config(parse(["--unknown-cost", "2"]))
    with pytest.raises(ValueError):
        VP.event_config(parse(["--unknown-plays", "--unknown-cost", "-1"]))
    with pytest.raises(SystemExit):
        parse(["--unknown-plays", "--unknown-cost", "2.5"])                  # whole elixir only


def test_vision_play_refuses_a_bad_unknown_cost_before_anything(monkeypatch, capsys):
    pytest.importorskip("numpy")
    from scripts import vision_play as VP

    monkeypatch.setattr(VP, "InstanceLock", lambda *a, **k: pytest.fail("took the instance lock"))
    monkeypatch.setattr(VP, "_main", lambda args: pytest.fail("started"))
    assert VP.main(["--unknown-cost", "2"]) == 2
    assert "refusing to start: --unknown-cost applies to --unknown-plays" in capsys.readouterr().out
    assert VP.main(["--unknown-plays", "--unknown-cost", "-1"]) == 2


def test_vision_replay_unknown_cost_flag(tmp_path, monkeypatch, capsys):
    pytest.importorskip("numpy")
    from tests.pipeline import test_latency_and_replay as TLR

    rc, _, s = TLR._replay_main(tmp_path, monkeypatch, "free", "--unknown-plays", "--unknown-cost", "0")
    assert rc == 0 and s["unknown_plays"] == {"cost": 0}
    assert "unknown    UNKNOWN plays on (card -1, 0 elixir each)" in capsys.readouterr().out
    rc, calls, s = TLR._replay_main(tmp_path, monkeypatch, "bad", "--unknown-cost", "3")
    assert (rc, calls, s) == (2, [], None)                 # refused before the detector or any output
    assert "UNKNOWN plays: --unknown-cost applies to --unknown-plays" in capsys.readouterr().out
