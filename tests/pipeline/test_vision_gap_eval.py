"""N14: tools/vision_gap_eval.py, the gate before a fine-tuned detector is pinned.

docs/NEW_CARDS_PLAN.md 6 and N14. Checked on hand-made data:
* play matching: same-card pairs first (a right read is not taken by a nearer wrong one), within
  1 s and 3 tiles, time only without positions, never across sessions, the most pairs in a burst;
* play scores: per new card recall and precision, an evolution's card and form accuracy, pooled
  play-ID accuracy, phantoms and the elixir error per minute (missed + phantom + wrong-card cost
  difference, an UNKNOWN play at its charged cost), the UNKNOWN safety net and its evidence floor;
* detection: card-level against form-aware (a MinionHorde_EV1 box read as a base Minion), a
  prediction inside an ignore region is no false positive while one on open ground is, side
  accuracy and our labels' recall on owner-vetted boxes (tonight's own pseudo-labels: info), new
  labels on our own boxes, what the model reads each label as;
* the birth window (the top box on 2 of the first 4 images; later parts, ours and short tracks
  left out), the seeded bootstrap, the checks and the verdict (PASS / FAIL / INSUFFICIENT);
* the whole gate on a hand-made Round 0 gold session with a fake vision_replay runner: replays
  cached by key, t = t_origin + tick / 20, gate.json and report.md, nothing written into the gold
  store or labels, refused gold (unconfirmed plays, a stale emit, not gold), the leak check
  failing a candidate whose mix holds the gold recording and not trusting a mix built again
  after training, other card ids or data replayed again, the gain over M0 once a fine-tune is
  the baseline, a pinned baseline that is not the pinned file refused;
* the KataCR regression arithmetic with a fake YOLO (pass at a drop of 0.009, fail at 0.011,
  cached per weights, yaml and val split); a real YOLO val (a 2-name and a 4-name model on a
  2-name yaml) runs only under the gpu marker.
numpy is needed (vision_replay imports it); no GPU, engine, emulator or image codec otherwise.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "tools"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vision_autolabel as VA  # noqa: E402
import vision_gap_eval as GE  # noqa: E402
import vision_replay as VR  # noqa: E402
from scripts.vision.detector import file_sha256  # noqa: E402
from scripts.vision.record import VisionRecorder  # noqa: E402
from scripts.vision.types import (ENEMY_TEAM, OWN_TEAM, SCREEN_BATTLE, SCREEN_MENU, Detection, Events,  # noqa: E402
                                  OppPlay, Track, VisionFrame)
from tests.pipeline.test_autolabel import EVENTS, TABLE, fake_recording, run_ingest  # noqa: E402

R0_SID = "20261029_R0_1"
R0_DECK = "boss-bandit,elite-barbarians,executioner,freeze,guards,hog-rider,minion-horde-ev1,zap"
CFG = GE.GateConfig()
KEYS = GE.Keys(TABLE)


@pytest.fixture(scope="module")
def keys():
    key_map, note = VA.load_key_map()
    assert note is None, note
    return key_map


def pp(card, t, *, cost=None, form=None, x=None, y=None, sid=None, card_id=0):
    return GE.PredPlay(card, card_id, t, x, y, cost=cost, form=form, sid=sid)


def gp(card, t, cost, *, form=0, kind="unit", x=None, y=None):
    return {"card": card, "t": t, "cost": cost, "form": form, "kind": kind, "x_mt": x, "y_mt": y}


def gimg(stem, t, boxes, ignore=(), sid="s"):
    return GE.GoldImage(sid, stem, 0, None, t, 1, (1000, 1000), boxes, [tuple(r) for r in ignore])


def gbox(label, box, side, *, key=None, src="owner", typ=None):
    typ = typ or KEYS.of(label)[2]
    return {"label": label, "side": side, "box": list(box), "type": typ, "src": src, "how": "smooth", "key": key}


def pred(cls, box, p_enemy, conf=0.8, form=0):
    c, label, typ = KEYS.of(cls, form)
    return GE.Pred(c, label, VR.side_of(p_enemy), conf, tuple(v / 1000.0 for v in box), typ, form)


# ---------------------------------------------------------------------------
# plays
# ---------------------------------------------------------------------------

def test_match_plays_same_card_first():
    gold = [{"t": 10.0, "card": "BossBandit", "x_mt": 9000, "y_mt": 10000},
            {"t": 10.4, "card": "Zap", "x_mt": 9500, "y_mt": 10500}]
    pred_ = [pp("GoldenKnight", 10.3, x=9000, y=10100), pp("Zap", 10.5, x=9400, y=10400),
             pp("Minions", 30.0, x=3000, y=5000)]
    pairs, missed, phantom = GE.match_plays(gold, pred_, dt_s=1.0, tiles=3.0)
    assert pairs == [(0, 0, False), (1, 1, True)] and missed == [] and phantom == [2]
    one = [{"t": 10.0, "card": "Zap", "x_mt": 0, "y_mt": 0}]
    match = lambda p: GE.match_plays(one, [p], dt_s=1.0, tiles=3.0)[0]  # noqa: E731
    assert match(pp("Zap", 10.9)) == [(0, 0, True)]                      # no position: time only
    assert match(pp("Zap", 11.01, x=0, y=0)) == []                       # 1.01 s
    assert match(pp("Zap", 10.0, x=3010, y=0)) == []                     # 3.01 tiles
    assert match(pp("Zap", 10.0, x=2990, y=0)) == [(0, 0, True)]
    assert GE.match_plays([dict(one[0], sid="a")], [pp("Zap", 10.0, sid="b")], dt_s=1.0, tiles=3.0)[0] == []
    # a burst within the window: the most pairs, not a greedy pass (which pairs 0.0 with 0.45 and loses 0.9)
    two = [{"t": 0.0, "card": "Zap"}, {"t": 0.9, "card": "Zap"}]
    pairs, missed, phantom = GE.match_plays(two, [pp("Zap", 0.45), pp("Zap", -0.6)], dt_s=1.0, tiles=3.0)
    assert pairs == [(0, 1, True), (1, 0, True)] and missed == [] and phantom == []
    # stage 1 keeps its most pairs before any other-card pair is taken
    gold3 = [{"t": 0.0, "card": "Zap"}, {"t": 0.5, "card": "Knight"}]
    pairs, missed, _ = GE.match_plays(gold3, [pp("Knight", 0.2), pp("Zap", 0.6)], dt_s=1.0, tiles=3.0)
    assert pairs == [(0, 1, True), (1, 0, True)] and missed == []


def test_play_scores():
    gold = ([gp("BossBandit", 10.0 * k, 6) for k in range(8)]
            + [gp("Zap", 10.0 * k, 2, kind="spell") for k in range(8, 12)]
            + [gp("Knight", 10.0 * k, 3) for k in range(12, 18)]
            + [gp("MinionHorde", 10.0 * k, 5, form=1) for k in range(18, 22)])
    pred_ = ([pp("BossBandit", 10.0 * k + 0.2, cost=6) for k in range(6)]
             + [pp("GoldenKnight", 10.0 * k + 0.3, cost=4) for k in (6, 7)]            # BossBandit read as GoldenKnight
             + [pp("Zap", 10.0 * k + 0.1, cost=2) for k in (8, 9, 10)]                 # the 4th Zap missed
             + [pp("Knight", 10.0 * k, cost=3) for k in range(12, 17)]
             + [pp(GE.UNKNOWN, 170.4, cost=4, card_id=-1)]                             # an UNKNOWN on a real Knight
             + [pp("MinionHorde", 10.0 * k + 0.5, cost=5, form=1 if k < 21 else 0) for k in range(18, 22)]
             + [pp(GE.UNKNOWN, 500.0, cost=4, card_id=-1), pp("GoldenKnight", 600.0, cost=4),
                pp("GoldenKnight", 610.0, cost=4)])                                     # three phantoms
    s = GE.score_plays(gold, pred_, minutes=4.0, new_cards={"BossBandit"}, config=CFG)
    assert (s["gold"], s["pred"], s["right"], s["wrong"], s["missed"], s["phantoms"]) == (22, 24, 18, 3, 1, 3)
    assert s["play_id"] == pytest.approx(18 / 22) and s["phantom_per_min"] == pytest.approx(3 / 4)
    el = s["elixir"]
    assert (el["missed"], el["phantom"], el["wrong_card"], el["error"]) == (2, 12, 5, 19)
    # missed Zap 2 + phantoms (UNKNOWN 4, GoldenKnight 4 twice) + |4 - 6| x 2 (GoldenKnight for BossBandit) + |4 - 3|
    assert el["per_min"] == pytest.approx((2 + (4 + 4 + 4) + abs(4 - 6) * 2 + abs(4 - 3)) / 4)
    assert (el["gold_spent"], el["pred_spent"]) == (94, 101)
    bb = s["per_card"]["BossBandit"]
    assert (bb["recall"], bb["precision"], bb["new"]) == (0.75, 1.0, True)
    assert bb["read_as"] == {"BossBandit": 6, "GoldenKnight": 2}
    assert s["per_card"]["Zap"]["read_as"] == {"Zap": 3, "missed": 1}
    assert s["per_card"]["GoldenKnight"]["precision"] == 0.0 and s["per_card"]["GoldenKnight"]["gold"] == 0
    mh = s["forms"]["MinionHorde.1"]
    assert (mh["gold"], mh["right"], mh["card_accuracy"], mh["form_accuracy"]) == (4, 4, 1.0, 0.75)
    unk = s["unknown"]
    assert (unk["pred"], unk["matched"], unk["matched_share"]) == (2, 1, 0.5) and unk["share"] == pytest.approx(2 / 24)
    checks = {c.id: c for c in GE.build_checks({"plays": s}, None, CFG)}
    assert checks["play.recall.BossBandit"].status == "fail" and checks["play.precision.BossBandit"].status == "pass"
    assert checks["play.recall.BossBandit"].n == 8 and checks["play.recall.BossBandit"].ci is not None
    assert "play.recall.Zap" not in checks and "play.recall.GoldenKnight" not in checks     # no new cards
    assert checks["play.unknown"].status == "insufficient"           # 2 UNKNOWN plays: below the 5 it needs
    assert checks["play.id"].status == "fail" and checks["play.id"].required
    assert checks["play.id_gain"].status == "insufficient"            # no baseline
    assert checks["play.phantom"].status == "fail" and checks["play.elixir"].status == "fail"
    assert checks["play.form.MinionHorde.1"].status == "pass" and checks["play.form_acc.MinionHorde.1"].status == "info"
    assert GE.score_plays(gold, pred_, minutes=0.0, config=CFG)["phantom_per_min"] is None


# ---------------------------------------------------------------------------
# detection and the birth window
# ---------------------------------------------------------------------------

def test_detection_with_ignore_regions_and_forms():
    mh, bb = (100, 100, 150, 150), (300, 300, 360, 380)
    a1, a2, a3, a4 = (500, 700, 560, 780), (600, 700, 660, 780), (700, 700, 760, 780), (200, 800, 260, 880)
    a5, a6 = (300, 800, 360, 880), (400, 800, 460, 880)
    gold = {
        "s/a": gimg("a", 10.0, [gbox("MinionHorde_EV1", mh, 1, key="40"), gbox("BossBandit", bb, 1, key="30"),
                                gbox("HogRider", a1, 0, src="live"), gbox("clock", (0, 0, 20, 20), 1, src="live")],
                    ignore=[(700, 100, 800, 200)]),
        "s/b": gimg("b", 10.5, [gbox("BossBandit", bb, 1, key="30"), gbox("HogRider", a2, 0, src="live"),
                                gbox("HogRider", a3, 0, src="live")]),
        "s/c": gimg("c", 12.0, [gbox("Musketeer", a4, 0, src="live"),
                                gbox("HogRider", a5, 0, src="live", key="70"),         # one of our o tracks: vetted
                                gbox("HogRider", a6, 0, src="live", key="70")]),
        "s/d": gimg("d", 13.0, [gbox("BossBandit", bb, 1, key="30")]),          # not replayed
    }
    preds = {
        "s/a": [pred("Minion", mh, 0.8), pred("BossBandit", bb, 0.9, conf=0.9),
                pred("Knight", (710, 110, 790, 190), 0.9),                     # inside the ignore region
                pred("Knight", (850, 850, 900, 900), 0.9),                     # on open ground (an x track, say)
                pred("HogRider", a1, 0.1)],
        "s/b": [pred("BossBandit", bb, 0.2),                                   # the wrong side
                pred("HogRider", a2, 0.1), pred("BossBandit", a3, 0.1)],       # a new label on our box
        "s/c": [pred("Musketeer", a4, 0.1), pred("HogRider", a5, 0.1)],              # a6 missed
    }
    own = TABLE.deck_labels(list(VA.OUR_DECK))
    d = GE.score_detection(gold, preds, TABLE, new_labels={"BossBandit", "MinionHorde_EV1"}, own_labels=own, config=CFG,
                           new_classes={"BossBandit"})
    card, form = d["per_class"]["card"], d["per_class"]["form"]
    assert (card["Minion"]["tp"], card["Minion"]["fp"], card["Minion"]["fn"]) == (1, 0, 0)
    assert (card["BossBandit"]["tp"], card["BossBandit"]["fp"], card["BossBandit"]["fn"]) == (2, 1, 1)
    assert (card["Knight"]["tp"], card["Knight"]["fp"]) == (0, 1)            # the one in the ignore region is no FP
    assert (card["HogRider"]["tp"], card["HogRider"]["fn"]) == (3, 2) and card["Musketeer"]["tp"] == 1
    assert "clock" not in card                                                # only units and spells are scored
    assert d["ignored_fp"] == {"card": 1, "form": 1} and (d["images"], d["unreplayed"]) == (4, 1)
    assert (form["MinionHorde_EV1"]["fn"], form["Minion"]["fp"]) == (1, 1)    # form-aware: the base Minion is wrong
    mhl, bbl = d["labels"]["MinionHorde_EV1"], d["labels"]["BossBandit"]
    assert (mhl["card"]["recall"], mhl["form"]["recall"], mhl["tracks"], mhl["new"]) == (1.0, 0.0, 1, True)
    assert (bbl["tracks"], bbl["boxes"], bbl["card"]["recall"], bbl["card"]["precision"]) == (1, 3, pytest.approx(2 / 3),
                                                                                             pytest.approx(2 / 3))
    assert bbl["card"]["ci"] == "bootstrap" and d["labels"]["HogRider"]["card"]["ci"] == "wilson"
    # side and our labels' recall on owner-vetted boxes only: tonight's own pseudo-labels agree with
    # tonight's model by construction, so they are reported apart
    assert (d["side"]["n"], d["side"]["ok"]) == (4, 3) and (d["side_live"]["n"], d["side_live"]["ok"]) == (3, 3)
    assert d["newlabel_on_own"] == {"n": 6, "hits": 1, "share": pytest.approx(1 / 6),
                                    "ci": pytest.approx(list(VA.wilson(1, 6)), abs=1e-6), "by_label": {"BossBandit": 1}}
    assert d["own"] == {"HogRider": {"tp": 1, "n": 2, "recall": 0.5}}
    assert d["own_live"]["HogRider"] == {"tp": 2, "n": 3, "recall": pytest.approx(2 / 3)}
    assert d["own_live"]["Musketeer"]["recall"] == 1.0
    assert GE.read_table(gold, preds, config=CFG) == {"BossBandit": {"BossBandit": 2, "none": 1},
                                                      "MinionHorde_EV1": {"Minion": 1}}
    checks = {c.id: c for c in GE.build_checks({"detection": d}, None, CFG)}
    assert checks["det.recall.BossBandit"].status == "insufficient"          # 1 gold track, below 15
    assert checks["det.side"].status == "insufficient" and checks["det.side"].value == pytest.approx(3 / 4)
    assert checks["det.side_live"].status == "info" and checks["det.side_live"].value == 1.0
    assert checks["reg.own.HogRider"].value == 0.5 and checks["reg.own.HogRider"].n == 2
    assert checks["reg.own_live"].status == "info" and checks["reg.own_live"].value == pytest.approx(3 / 4)
    assert "det.recall.HogRider" not in checks and "reg.own.Musketeer" not in checks          # not a new label


def test_birth_window_recall():
    box = (300, 300, 360, 380)
    gold, preds, tracks = {}, {}, {}

    def track(key, label, n_images, hits_at, *, part=0, side=1, decoy_at=()):
        cls, flab, _typ = KEYS.of(label)
        stems = []
        for j in range(n_images):
            stem = f"{key}_{j}"
            gold[f"s/{stem}"] = gimg(stem, 0.1 * j, [gbox(label, box, side, key=key)])
            ps = [pred(label, box, 0.9, conf=0.6)] if j in hits_at else []
            if j in decoy_at:
                ps.append(pred("Knight", (305, 305, 365, 385), 0.9, conf=0.9))    # a stronger box of another class
            preds[f"s/{stem}"] = ps
            stems.append(stem)
        tracks[f"s/{key}"] = GE.GoldTrack("s", key, int(key.split(".")[0]), part, label, cls, flab, 0, side, None, 0.0,
                                          stems)

    track("10", "BossBandit", 6, {0, 1, 3}, decoy_at={1})     # hits on images 0 and 3: counts
    track("11", "BossBandit", 6, {1, 4, 5})                   # one hit in the first 4 (4 and 5 are too late): does not
    track("12.1", "BossBandit", 4, {0, 1, 2, 3}, part=1)      # a later part of a split track: no birth
    track("13", "HogRider", 4, {0, 1, 2, 3}, side=0)          # ours: not scored
    track("14", "zap", 1, {0})                                # one image: short
    b = GE.birth_window_recall(tracks, gold, preds, CFG)
    assert (b["tracks"], b["hits"], b["recall"], b["short"]) == (2, 1, 0.5, 1)
    assert b["per_label"] == {"BossBandit": {"hits": 1, "tracks": 2, "recall": 0.5}}
    assert b["span_s"] == pytest.approx(0.3) and (b["images"], b["need"]) == (4, 2)
    del gold["s/10_0"]                                         # a bad frame left out: the window moves on
    assert GE.birth_window_recall(tracks, gold, preds, CFG)["hits"] == 0


def test_bootstrap_is_seeded():
    units = [(1, 2), (0, 1), (3, 4), (2, 2), (0, 3), (4, 5), (1, 1)]
    a = GE.bootstrap_ratio(units, boot=500, seed=7)
    assert a == GE.bootstrap_ratio(units, boot=500, seed=7)
    point = sum(x for x, _ in units) / sum(y for _, y in units)
    assert 0.0 <= a[0] <= point <= a[1] <= 1.0 and a[0] < a[1]
    assert GE.bootstrap_ratio([(2, 2), (1, 1), (3, 3)], boot=100, seed=0) == (1.0, 1.0)
    assert GE.bootstrap_ratio([], boot=100, seed=0) == (0.0, 1.0) == GE.bootstrap_ratio([(0, 0)], boot=100, seed=0)


# ---------------------------------------------------------------------------
# checks and the verdict
# ---------------------------------------------------------------------------

def scores(*, play_id=0.90, side=0.99, birth=0.90, birth_tracks=30, phantom=0.2, elixir=1.0, on_own=0.0,
           kat=(0.600, 0.605), leak=(), measured=True, labels=None):
    """A model's scores as evaluate_model returns them, reduced to what build_checks reads."""
    return {"detection": {"labels": labels or {}, "side": {"n": 100, "ok": round(100 * side), "accuracy": side},
                          "own": {"HogRider": {"tp": 99, "n": 100, "recall": 0.99}},
                          "newlabel_on_own": {"n": 200, "hits": round(200 * on_own), "share": on_own}},
            "birth": {"tracks": birth_tracks, "hits": round(birth_tracks * birth), "recall": birth},
            "plays": {"gold": 40, "pred": 41, "play_id": play_id, "phantoms": 1, "minutes": 5.0, "phantom_per_min": phantom,
                      "elixir": {"per_min": elixir}, "unknown": {"pred": 0, "matched_share": None, "share": 0.0},
                      "per_card": {}, "forms": {}},
            "katacr": None if kat is None else {"candidate": {"map50": kat[0]}, "baseline": {"map50": kat[1]}},
            "leak": {"measured": measured, "leaks": list(leak), "sessions": 3, "note": None if measured else "no mix"},
            "landing": {"taps": 0, "miss_rate": None}}


def test_verdict():
    base = scores(play_id=0.62)
    good = GE.build_checks(scores(), base, CFG)
    assert GE.verdict(good) == GE.PASS
    assert {c.id for c in good if c.required} == {"det.side", "det.birth", "play.id", "play.id_gain", "play.phantom",
                                                  "play.elixir", "reg.katacr", "reg.newlabel_on_own", "leak"}
    assert {c.id: c.status for c in good}["reg.landing"] == "insufficient"       # no taps: thin, does not gate
    gain = {c.id: c for c in GE.build_checks(scores(play_id=0.80), base, CFG)}
    assert gain["play.id_gain"].value == pytest.approx(0.18) and gain["play.id_gain"].status == "pass"
    assert gain["play.id"].status == "fail" and gain["play.id_gain"].baseline == pytest.approx(0.62)
    thin_bb = {"BossBandit": {"new": True, "class": "BossBandit", "tracks": 3, "card": {"recall": 0.5, "precision": 0.5}}}
    assert GE.verdict(GE.build_checks(scores(labels=thin_bb), base, CFG)) == GE.PASS           # thin: reported only
    bad_bb = {"BossBandit": dict(thin_bb["BossBandit"], tracks=20)}
    checks = GE.build_checks(scores(labels=bad_bb), base, CFG)
    assert GE.verdict(checks) == GE.FAIL and "det.recall.BossBandit" in GE.verdict_reason(checks, GE.FAIL)
    few = GE.build_checks(scores(birth_tracks=5), base, CFG)
    assert GE.verdict(few) == GE.INSUFFICIENT and GE.verdict_reason(few, GE.INSUFFICIENT).endswith("det.birth")
    assert GE.verdict(GE.build_checks(scores(leak=["the mix holds gold session X"]), base, CFG)) == GE.FAIL
    assert GE.verdict(GE.build_checks(scores(measured=False), base, CFG)) == GE.INSUFFICIENT
    no_kat = {c.id: c for c in GE.build_checks(scores(kat=None), base, CFG)}
    assert no_kat["reg.katacr"].status == "insufficient" and no_kat["reg.katacr"].required
    assert GE.verdict(list(no_kat.values())) == GE.INSUFFICIENT
    assert GE.verdict(GE.build_checks(scores(birth_tracks=5, leak=["x"]), base, CFG)) == GE.FAIL     # a failure wins
    assert GE.verdict(GE.build_checks(scores(kat=(0.590, 0.605)), base, CFG)) == GE.FAIL             # drop 0.015
    worse_own = scores()
    worse_own["detection"]["own"]["HogRider"]["recall"] = 0.97                                       # 0.99 -> 0.97
    assert {c.id: c.status for c in GE.build_checks(worse_own, base, CFG)}["reg.own.HogRider"] == "fail"
    with pytest.raises(ValueError):
        GE.GateConfig(birth_hits=5, birth_images=4)


# ---------------------------------------------------------------------------
# the whole gate on a hand-made Round 0 gold session, with a fake vision_replay
# ---------------------------------------------------------------------------

GOLD_N = 30
BB_BOX, ZAP_BOX, HOG_BOX = (300.0, 400.0, 340.0, 460.0), (400.0, 500.0, 480.0, 580.0), (100.0, 900.0, 140.0, 960.0)
ORIGIN = 9.9                       # the replay's first image (a menu frame): time ticks count from it


def stem_of(n):
    return f"{5 * n + 5:08d}"


def t_of(n):
    return round(10.0 + 0.1 * n, 4)


def gold_part(key, label, n0, n1, birth, typ="unit"):
    return {"key": key, "h": int(key), "part": 0, "decision": "label", "label": label, "side": 1, "src": "owner",
            "type": typ, "card": None, "form": 0, "first_n": n0, "last_n": n1, "first_t": t_of(n0), "last_t": t_of(n1),
            "hk": 1, "uid": None, "birth": list(birth), "last": list(birth), "skipped": 0, "boxes": []}


PARTS = [gold_part("30", "BossBandit", 0, 10, (9000, 10000)), gold_part("50", "zap", 20, 22, (9000, 12000), "spell")]


def gold_session(tmp_path, keys):
    """Round 0 ingested as gold (natural), its tracks.json, a gold emit written by hand (30 battle
    images from t 10.0 every 0.1 s: Boss Bandit t30 on images 0-10, a Zap t50 on 20-22, our Hog
    Rider on all as tonight's pseudo-label) and its play list, derived and confirmed with plays."""
    root = tmp_path / "gold_rec" / R0_SID
    fake_recording(root)
    run_ingest(tmp_path, root, R0_DECK, key_map=keys, sid=R0_SID, role="gold", capture="natural")
    paths = VA.session_paths(R0_SID, tmp_path / "labels", tmp_path / "store")
    entries = [{"key": p["key"], "h": p["h"], "part": 0, "label": p["label"], "code": None, "source": "owner",
                "sheet": 1, "cluster": 0, "at": "2026-11-01T21:00:00"} for p in PARTS]
    VA.write_json(paths.tracks_file, {"schema": VA.TRACKS_SCHEMA, "sid": R0_SID, "tracks": entries, "split": []})
    paths.gold_dir.mkdir(parents=True)
    (paths.gold_dir / VA.PARTS_FILE).write_text("".join(json.dumps(p) + "\n" for p in PARTS), encoding="utf-8")
    rows = []
    for n in range(GOLD_N):
        boxes = [{"label": "HogRider", "side": 0, "box": list(HOG_BOX), "type": "unit", "src": "live", "how": "live",
                  "key": None}]
        if n <= 10:
            boxes.append({"label": "BossBandit", "side": 1, "box": list(BB_BOX), "type": "unit", "src": "owner",
                          "how": "smooth", "key": "30"})
        if 20 <= n <= 22:
            boxes.append({"label": "zap", "side": 1, "box": list(ZAP_BOX), "type": "spell", "src": "owner",
                          "how": "circle", "key": "50"})
        rows.append({"image": f"images/{stem_of(n)}.jpg", "stem": stem_of(n), "i": 5 * n + 5, "n": n, "t": t_of(n),
                     "hk": 1, "size": [747, 1328], "boxes": boxes, "ignore": [[600.0, 200.0, 660.0, 260.0]] if n == 5 else []})
    (paths.gold_dir / VA.GOLD_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    VA.write_json(paths.gold_dir / VA.EMIT_JSON, {
        "schema": VA.EMIT_SCHEMA, "sid": R0_SID, "kind": VA.KIND_GOLD, "role": VA.ROLE_GOLD, "capture": "natural",
        "created": "2026-11-01T22:00:00", "tracks_sha1": VA.sha1_file(paths.tracks_file), "bad_tracks": [],
        "battle_t0": {"1": 10.0}, "battle_seconds": {"1": 2.9}, "pending": []})
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE, key_map=keys, event_cards=EVENTS)
    first = VA.plays_session(R0_SID, **kw)
    assert [(p["card"], p["t"], p["cost"]) for p in first.plays] == [("BossBandit", 10.0, 6), ("Zap", 12.0, 2)]
    VA.plays_session(R0_SID, list_no=1, confirm=True, **kw)
    assert VA.plays_fresh(paths) == []
    return paths


class FakeReplay:
    """What vision_replay writes for one recording (a recorder session, image_every 0, and
    summary.json): the candidate reads Boss Bandit, the baseline GoldenKnight; both read the Zap
    and our Hog Rider; each reports its plays dated at the first sighting (tick 2 = t 10.0)."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        out = Path(argv[argv.index("--out") + 1])
        weights = Path(argv[argv.index("--detector") + 1])
        spawn = argv[argv.index("--spawn-table") + 1] if "--spawn-table" in argv else None
        _spawn, ids, ids_from = VR.engine_card_ids(spawn, TABLE)
        boss = "BossBandit" if weights.read_bytes().startswith(b"candidate") else "GoldenKnight"
        rec = VisionRecorder(out, session="recording_replay_test", image_every=0, clock=lambda: 0.0)
        rec.write(VisionFrame(t_capture=ORIGIN, screen=SCREEN_MENU, frame_size=(747, 1328)), None,
                  replay={"image": "00000000.jpg", "tick_source": None, "track_tick": None})
        for n in range(GOLD_N):
            t = t_of(n)
            dets = [Detection("HogRider", 0.1, 0.9, HOG_BOX)]
            tracks = [Track(1, "HogRider", OWN_TEAM, 0.9, 3000.0, 25000.0)]
            plays = []
            if n <= 10:
                dets.append(Detection(boss, 0.9, 0.8, BB_BOX))
                tracks.append(Track(7, boss, ENEMY_TEAM, 0.8, 9100.0, 10100.0))
            if 20 <= n <= 22:
                dets.append(Detection("Zap", 0.9, 0.7, ZAP_BOX))
            if n == 4:
                plays.append(OppPlay(ids[boss], 2, None, 9100.0, 10100.0, "unit", [7]))
            if n == 22:
                plays.append(OppPlay(ids["Zap"], 42, None, 9000.0, 12000.0, "spell", []))
            vf = VisionFrame(t_capture=t, tracks=tracks, events=Events(opp_plays=plays), screen=SCREEN_BATTLE,
                             detections=dets, frame_size=(747, 1328))
            rec.write(vf, None, replay={"image": f"{stem_of(n)}.jpg", "tick_source": "time",
                                        "track_tick": int(round((t - ORIGIN) * 20))})
        summary = {"schema": VR.SCHEMA, "frames": GOLD_N + 1, "ticks": "time",
                   "detector": f"{weights} sha256 {file_sha256(weights)[:12]} on fake",
                   "events": {"opp_plays": 2, "own_landed": 0, "own_missed": 0}, "unknown_plays": {"cost": 4},
                   "card_ids": ids_from}
        rec.log("replay_summary", **summary)
        rec.close()
        (rec.session_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return 0


class FakeYOLO:
    """ultralytics.YOLO for katacr_val: .val() returns the metrics shape (box.map50, box.ap50,
    box.ap_class_index, names); the mAP50 is looked up by the weights file's bytes."""
    maps: dict = {}
    vals: list = []

    def __init__(self, path):
        self.path = Path(path)

    def val(self, **kw):
        FakeYOLO.vals.append((self.path.name, kw))
        m = FakeYOLO.maps[self.path.read_bytes()]
        return SimpleNamespace(box=SimpleNamespace(map50=m, ap50=[m + 0.05, m - 0.05], ap_class_index=[0, 3]),
                               names={0: "HogRider@ally", 1: "HogRider@enemy", 2: "Knight@ally", 3: "Knight@enemy"})


def snapshot(*roots):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for r in roots for p in sorted(Path(r).rglob("*"))
            if p.is_file()}


def test_read_replay_and_gate_with_a_fake_runner(tmp_path, keys, capsys):
    paths = gold_session(tmp_path, keys)
    cand = tmp_path / "runs" / "ft1" / "weights" / "best.pt"
    cand.parent.mkdir(parents=True)
    cand.write_bytes(b"candidate weights")
    base = tmp_path / "m0" / "best.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"baseline weights")
    runner = FakeReplay()
    opts = GE.ReplayOptions(spawn_table=str(tmp_path / "no_spawn_table.json"))     # replay-only card ids, both sides
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", out_root=tmp_path / "gate", options=opts,
              runner=runner, table=TABLE, katacr=False, base_labels=None, clock=lambda: 1793800000.0,
              log=lambda m: None)
    before = snapshot(tmp_path / "store", tmp_path / "labels")

    res = GE.run_gate(cand, baseline=base, gold=[R0_SID], **kw)
    assert len(runner.calls) == 2                                          # the baseline, then the candidate
    argv = runner.calls[0]
    assert argv[:5] == [str(paths.recording), "--detector", str(base), "--ticks", "time"] and "--unknown-plays" in argv
    assert res.verdict == GE.INSUFFICIENT and res.name.startswith("ft1_")
    cs = {c.id: c for c in res.checks}
    assert (cs["play.id"].value, cs["play.id"].baseline, cs["play.id"].status) == (1.0, 0.5, "insufficient")
    assert cs["play.id_gain"].value == pytest.approx(0.5)
    # side on the owner's boxes only (Boss Bandit 11, Zap 3); our Hog Rider is tonight's pseudo-label: info
    assert (cs["det.side"].value, cs["det.side"].n, cs["det.side"].status) == (1.0, 14, "insufficient")
    assert (cs["det.side_live"].value, cs["det.side_live"].n, cs["det.side_live"].status) == (1.0, 30, "info")
    assert (cs["det.birth"].value, cs["det.birth"].n) == (1.0, 2)
    assert (cs["det.recall.BossBandit"].value, cs["det.recall.BossBandit"].baseline, cs["det.recall.BossBandit"].n) == (
        1.0, 0.0, 1)
    assert cs["play.recall.BossBandit"].value == 1.0 and cs["play.recall.BossBandit"].baseline == 0.0
    assert "reg.own.HogRider" not in cs                                    # no o track: nothing owner-vetted of ours
    assert (cs["reg.own_live"].value, cs["reg.own_live"].n, cs["reg.own_live"].status) == (1.0, 30, "info")
    assert "--no-katacr-val" in cs["reg.katacr"].note and cs["reg.katacr"].status == "insufficient"
    assert cs["leak"].status == "insufficient" and "detector_manifest.json" in cs["leak"].note
    d = res.doc
    assert d["reads"] == {"BossBandit": {"GoldenKnight": 11}, "zap": {"Zap": 3}}             # what M0 reads
    assert d["baseline_plays"] == {"BossBandit": {"GoldenKnight": 1}, "Zap": {"Zap": 1}}
    assert d["gold"][0]["sid"] == R0_SID and d["gold"][0]["plays"] == 2 and d["gold"][0]["images"] == GOLD_N
    assert "BossBandit" in d["labels"]["new_cards"] and "Zap" not in d["labels"]["new_cards"]
    # GoldenKnight (classes.json: 4 elixir) read for a Boss Bandit (6): an elixir error of 2
    assert d["candidate_scores"]["plays"]["elixir"]["error"] == 0 and d["baseline_scores"]["plays"]["elixir"]["error"] == 2
    on_disk = json.loads((res.out_dir / GE.GATE_FILE).read_text(encoding="utf-8"))
    assert on_disk["schema"] == GE.GATE_SCHEMA and on_disk["verdict"] == GE.INSUFFICIENT
    report = (res.out_dir / GE.REPORT_FILE).read_text(encoding="utf-8")
    assert report.startswith("# Gap gate: INSUFFICIENT") and "What the baseline reads each gold label as" in report
    assert GE.main(["report", str(res.out_dir)]) == GE.EXIT_OK and "# Gap gate: INSUFFICIENT" in capsys.readouterr().out
    assert snapshot(tmp_path / "store", tmp_path / "labels") == before    # gold is only read

    # read_replay: t = t_origin + tick / 20 (ticks count from the first image, a menu frame here)
    session = Path(d["candidate_scores"]["replays"][0]["session"])
    _spawn, ids, _from = VR.engine_card_ids(opts.spawn_table, TABLE)
    rr = GE.read_replay(session, TABLE, {i: n for n, i in ids.items()})
    assert rr.t_origin == pytest.approx(ORIGIN) and rr.frames == GOLD_N + 1
    assert [(p.card, p.t, p.tick, p.cost, p.form, p.kind) for p in rr.plays] == [
        ("BossBandit", pytest.approx(10.0), 2, 6, 0, "unit"), ("Zap", pytest.approx(12.0), 42, 2, None, "spell")]
    assert rr.images["00000000"] == [] and [p.cls for p in rr.images[stem_of(0)]] == ["HogRider", "BossBandit"]
    with pytest.raises(GE.GateError, match="other weights"):
        GE.read_replay(session, TABLE, {}, sha256="ffffffffffffffff")
    with pytest.raises(GE.GateError, match="card ids"):                    # decoded with other ids: refused
        GE.read_replay(session, TABLE, {}, ids_from="engine catalogue (elsewhere)")
    assert GE.read_replay(session, TABLE, {}, ids_from=_from).frames == GOLD_N + 1

    # the cache: the same key replays nothing; --no-reuse and other flags replay again
    res2 = GE.run_gate(cand, baseline=base, gold=[R0_SID], **kw)
    assert len(runner.calls) == 2 and res2.out_dir != res.out_dir and res2.doc["checks"] == d["checks"]
    assert all(r["reused"] for r in res2.doc["candidate_scores"]["replays"])

    # the baseline report (M0 on Round 0): no candidate, verdict BASELINE; through the command line too
    m0 = GE.run_gate(None, baseline=base, mode=GE.MODE_BASELINE, gold=[R0_SID], **kw)
    assert m0.verdict == GE.BASELINE and m0.checks == [] and m0.name.startswith("M0_") and len(runner.calls) == 2
    assert m0.report.startswith("# Gap baseline") and m0.doc["candidate_scores"] is None
    assert {c["id"] for c in m0.doc["baseline_checks"]} >= {"det.side", "det.birth", "play.id", "reg.katacr"}
    assert m0.doc["reads"] == d["reads"] and m0.doc["baseline_plays"] == d["baseline_plays"]
    cli = ["baseline", "--weights", str(base), "--gold", R0_SID, "--labels-root", str(tmp_path / "labels"),
           "--store", str(tmp_path / "store"), "--out", str(tmp_path / "gate"), "--spawn-table", opts.spawn_table,
           "--no-katacr-val", "--name", "m0_cli"]
    assert GE.main(cli) == GE.EXIT_OK and len(runner.calls) == 2           # the cached replay: no runner needed
    assert json.loads((tmp_path / "gate" / "m0_cli" / GE.GATE_FILE).read_text(encoding="utf-8"))["verdict"] == GE.BASELINE
    assert "baseline BASELINE" in capsys.readouterr().out
    GE.run_gate(cand, baseline=base, gold=[R0_SID], **dict(kw, reuse=False))
    assert len(runner.calls) == 4
    GE.run_gate(cand, baseline=base, gold=[R0_SID], **dict(kw, options=GE.ReplayOptions(
        unknown_plays=False, spawn_table=opts.spawn_table)))
    assert len(runner.calls) == 6 and "--unknown-plays" not in runner.calls[-1]
    # other card ids or other data the chain reads: replayed again (the key holds their digests)
    g0 = GE.load_gold(R0_SID, labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE)
    _session, reused = GE.cached_replay(g0, base, file_sha256(base), tmp_path / "gate" / GE.CACHE_DIR, opts, runner=runner,
                                        log=lambda m: None, card_ids="other ids")
    assert not reused and len(runner.calls) == 7
    key = json.loads((_session.parent / GE.KEY_FILE).read_text(encoding="utf-8"))
    assert key["card_ids"] == "other ids" and key["spawn_table"] == "absent" and len(key["data"]) == 40
    root = tmp_path / "chain_root"
    (root / "data" / "vision").mkdir(parents=True)
    (root / "data" / "vision" / "classes.json").write_text("{}", encoding="utf-8")
    d1 = GE.data_digest(root, spawn_table=root / "spawn.json")
    (root / "data" / "vision" / "classes.json").write_text('{"cards": {}}', encoding="utf-8")
    d2 = GE.data_digest(root, spawn_table=root / "spawn.json")
    (root / "spawn.json").write_text('{"card_names": ["Knight"]}', encoding="utf-8")
    d3 = GE.data_digest(root, spawn_table=root / "spawn.json")
    assert d1["data"] != d2["data"] and d1["spawn_table"] == d2["spawn_table"] == "absent" != d3["spawn_table"]

    # M0: the first baseline run recorded it; once a fine-tune is the baseline (pinned), the gain is over M0
    m0_doc = json.loads((tmp_path / "gate" / GE.M0_FILE).read_text(encoding="utf-8"))
    assert m0_doc["sha256"] == file_sha256(base) and Path(m0_doc["weights"]) == base.resolve()
    ft0 = tmp_path / "runs" / "ft0" / "weights" / "best.pt"
    ft0.parent.mkdir(parents=True)
    ft0.write_bytes(b"candidate ft0, pinned")                              # reads Boss Bandit, as the candidate
    on_ft0 = GE.run_gate(cand, baseline=ft0, gold=[R0_SID], **kw)
    c0 = {c.id: c for c in on_ft0.checks}
    assert (c0["play.id_gain"].value, c0["play.id_gain"].baseline) == (pytest.approx(0.5), 0.5)   # over M0
    assert c0["play.id_gain_baseline"].value == 0.0 and c0["play.id_gain_baseline"].status == "info"
    assert c0["play.id"].baseline == 1.0 and on_ft0.doc["m0"]["from"] == GE.M0_FILE
    assert on_ft0.doc["m0"]["play_id"] == 0.5
    pin = tmp_path / "pin.json"
    from scripts.vision.detector import write_pin
    write_pin(ft0, path=pin)
    ft0.write_bytes(b"candidate ft0, changed after the pin")
    with pytest.raises(GE.GateError, match="not the pinned file") as err:
        GE.run_gate(cand, gold=[R0_SID], pin=pin, **kw)
    assert err.value.code == GE.EXIT_CHECK

    # the leak check: the candidate's mix holds the gold recording (by frames_sha1) and trains on its store
    mix = tmp_path / "mix" / "ft1"
    mix.mkdir(parents=True)
    (mix / "cr_mix.yaml").write_text(f'path: {mix.as_posix()}\ntrain: ["{(mix / "train" / "images").as_posix()}", '
                                     f'"{(paths.recording / "images").as_posix()}"]\nval: ["{mix.as_posix()}/val"]\n',
                                     encoding="utf-8")
    sha = VA.load_session(R0_SID, tmp_path / "labels")["recording"]["frames_sha1"]
    VA.write_json(mix / "mix.json", {"schema": VA.MIX_SCHEMA, "name": "ft1",
                                     "sessions": [{"sid": "20261101_A_1", "role": "train", "frames_sha1": sha}]})
    VA.write_json(cand.parent.parent / GE.MANIFEST_FILE, {"weights": cand.as_posix(), "data": (mix / "cr_mix.yaml").as_posix(),
                                                          "label_set_digest": "abc"})
    yaml_path = tmp_path / "yolo" / "cr.yaml"
    yaml_path.parent.mkdir()
    yaml_path.write_text("names:\n  0: HogRider@ally\n", encoding="utf-8")
    FakeYOLO.maps, FakeYOLO.vals = {b"candidate weights": 0.600, b"baseline weights": 0.605}, []
    res3 = GE.run_gate(cand, baseline=base, gold=[R0_SID], **dict(kw, katacr=True, katacr_data=yaml_path,
                                                                  yolo_factory=FakeYOLO))
    cs3 = {c.id: c for c in res3.checks}
    assert res3.verdict == GE.FAIL and cs3["leak"].status == "fail" and cs3["leak"].value == 2
    assert "20261101_A_1 is the recording of gold session" in cs3["leak"].note and "under gold session" in cs3["leak"].note
    assert cs3["reg.katacr"].status == "pass" and len(FakeYOLO.vals) == 2
    assert res3.doc["candidate"]["label_set_digest"] == "abc"
    g0 = GE.load_gold(R0_SID, labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE)
    lk = dict(labels_root=tmp_path / "labels", store=tmp_path / "store")
    # the mix the run trained on (its SHA-1 in the manifest), clean: measured, no leak
    VA.write_json(mix / "mix.json", {"schema": VA.MIX_SCHEMA, "name": "ft1", "sessions": [{"sid": "20261101_A_1"}]})
    (mix / "cr_mix.yaml").write_text(f'path: {mix.as_posix()}\ntrain: ["{(mix / "train" / "images").as_posix()}"]\n',
                                     encoding="utf-8")
    VA.write_json(cand.parent.parent / GE.MANIFEST_FILE, {"data": (mix / "cr_mix.yaml").as_posix(),
                                                          "mix_sha1": GE._sha1_file(mix / "mix.json"),
                                                          "data_sha1": GE._sha1_file(mix / "cr_mix.yaml")})
    lc = GE.leak_check(cand, [g0], **lk)
    assert lc["measured"] and lc["leaks"] == []
    # built again after training (the gold session moved to gold_excluded, say): not measured
    VA.write_json(mix / "mix.json", {"schema": VA.MIX_SCHEMA, "name": "ft1", "sessions": [], "gold_excluded": ["x"]})
    lc = GE.leak_check(cand, [g0], **lk)
    assert not lc["measured"] and "changed since this run was trained" in lc["note"]
    # an older manifest (no mix_sha1): a mix.json newer than the weights is not trusted either
    VA.write_json(cand.parent.parent / GE.MANIFEST_FILE, {"data": (mix / "cr_mix.yaml").as_posix()})
    os.utime(cand, (1_700_000_000, 1_700_000_000))
    lc = GE.leak_check(cand, [g0], **lk)
    assert not lc["measured"] and "newer than the weights" in lc["note"]
    os.utime(mix / "mix.json", (1_600_000_000, 1_600_000_000))
    assert GE.leak_check(cand, [g0], **lk)["measured"]

    # refused gold: unconfirmed plays, a stale emit, a session that is not gold
    plays_doc = json.loads(paths.plays_file.read_text(encoding="utf-8"))
    VA.write_json(paths.plays_file, dict(plays_doc, confirmed=False))
    with pytest.raises(GE.GateError, match="not confirmed") as err:
        GE.load_gold(R0_SID, labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE)
    assert err.value.code == GE.EXIT_CHECK
    with pytest.raises(GE.GateError, match="no gold session is ready"):
        GE.run_gate(cand, baseline=base, **kw)
    assert GE.gold_ready(tmp_path / "labels", tmp_path / "store")[0] == []
    VA.write_json(paths.plays_file, plays_doc)
    assert GE.gold_ready(tmp_path / "labels", tmp_path / "store")[0] == [R0_SID]
    tdoc = json.loads(paths.tracks_file.read_text(encoding="utf-8"))
    VA.write_json(paths.tracks_file, dict(tdoc, history=["edited"]))
    with pytest.raises(GE.GateError, match="gold emit cannot be scored") as err:
        GE.load_gold(R0_SID, labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE)
    assert err.value.code == GE.EXIT_CHECK
    doc = json.loads(paths.session_file.read_text(encoding="utf-8"))
    VA.write_json(paths.session_file, dict(doc, role=VA.ROLE_TRAIN, capture=VA.CAPTURE_HARVEST))
    with pytest.raises(GE.GateError, match="not a gold session") as err:
        GE.load_gold(R0_SID, labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE)
    assert err.value.code == GE.EXIT_INPUT


def test_replay_argv_and_command_line_input(tmp_path, capsys):
    assert GE.replay_argv(Path("rec"), Path("w.pt"), Path("out")) == [
        "rec", "--detector", "w.pt", "--ticks", "time", "--out", "out", "--unknown-plays"]
    argv = GE.replay_argv(Path("rec"), Path("w.pt"), Path("out"),
                          GE.ReplayOptions(unknown_cost=5, device="cpu", spawn_table="s.json", geometry="g.json"))
    assert argv[7:] == ["--unknown-plays", "--unknown-cost", "5", "--device", "cpu", "--spawn-table", "s.json",
                        "--geometry", "g.json"]
    off = GE.replay_argv(Path("rec"), Path("w.pt"), Path("out"), GE.ReplayOptions(unknown_plays=False))
    assert "--unknown-plays" not in off and "--unknown-cost" not in off
    assert GE.ReplayOptions().flags() == {"ticks": "time", "unknown_plays": True, "unknown_cost": None, "spawn_table": None,
                                          "geometry": None}
    missing = str(tmp_path / "none.pt")
    roots = ["--labels-root", str(tmp_path / "labels"), "--store", str(tmp_path / "store"), "--out", str(tmp_path / "g")]
    assert GE.main(["gate", "--candidate", missing, "--no-unknown-plays", "--unknown-cost", "3", *roots]) == GE.EXIT_INPUT
    assert GE.main(["gate", "--candidate", missing, *roots]) == GE.EXIT_INPUT
    assert "candidate weights" in capsys.readouterr().out
    assert GE.main(["report", str(tmp_path / "nothing")]) == GE.EXIT_INPUT
    assert not (tmp_path / "g").exists()


def test_katacr_regression_arithmetic(tmp_path):
    data = tmp_path / "cr.yaml"
    data.write_text("path: x\ntrain: images/train\nval: images/val\nnames:\n  0: HogRider@ally\n", encoding="utf-8")
    for name, blob, m in (("base.pt", b"base", 0.600), ("ok.pt", b"ok", 0.591), ("bad.pt", b"bad", 0.589)):
        (tmp_path / name).write_bytes(blob)
    FakeYOLO.maps, FakeYOLO.vals = {b"base": 0.600, b"ok": 0.591, b"bad": 0.589}, []
    kw = dict(yolo_factory=FakeYOLO, cache=tmp_path / "cache", log=lambda m: None)
    ok = GE.katacr_regression(tmp_path / "ok.pt", tmp_path / "base.pt", data, **kw)
    assert ok["drop"] == pytest.approx(0.009)
    assert ok["candidate"]["ap50"] == {"HogRider@ally": pytest.approx(0.641), "Knight@enemy": pytest.approx(0.541)}
    check = {c.id: c for c in GE.build_checks({"katacr": ok}, None, CFG)}["reg.katacr"]
    assert check.status == "pass" and check.value == pytest.approx(0.591) and check.threshold == pytest.approx(0.59)
    bad = GE.katacr_regression(tmp_path / "bad.pt", tmp_path / "base.pt", data, **kw)
    assert bad["drop"] == pytest.approx(0.011)
    assert {c.id: c for c in GE.build_checks({"katacr": bad}, None, CFG)}["reg.katacr"].status == "fail"
    assert sorted(n for n, _ in FakeYOLO.vals) == ["bad.pt", "base.pt", "ok.pt"]          # the baseline once: cached
    first = FakeYOLO.vals[0][1]
    assert (first["data"], first["split"], first["imgsz"], first["plots"]) == (str(data), "val", 640, False)
    cached = json.loads((tmp_path / "cache" / file_sha256(tmp_path / "base.pt")[:12] / GE.KATACR_FILE).read_text())
    assert cached["map50"] == 0.6 and cached["classes"] == 2 and cached["weights_sha256"] == file_sha256(tmp_path / "base.pt")
    data.write_text(data.read_text(encoding="utf-8") + "  1: HogRider@enemy\n", encoding="utf-8")
    GE.katacr_val(tmp_path / "base.pt", data, **kw)                                         # another yaml: measured again
    assert len(FakeYOLO.vals) == 4
    GE.katacr_val(tmp_path / "base.pt", data, **kw)
    assert len(FakeYOLO.vals) == 4                                                           # the same split: cached
    (tmp_path / "x" / "labels" / "val").mkdir(parents=True)                                  # path: x, val: images/val
    (tmp_path / "x" / "labels" / "val" / "f0.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    sig = GE.val_signature(data)
    GE.katacr_val(tmp_path / "base.pt", data, **kw)                                         # another val split, same yaml
    assert len(FakeYOLO.vals) == 5 and sig != "unreadable"
    # baseline mode: the model's own KataCR val is reported as info, not "not run"
    info = {c.id: c for c in GE.build_checks({"katacr_val": {"map50": 0.6, "classes": 2}}, None, CFG)}["reg.katacr"]
    assert info.status == "info" and info.value == 0.6 and not info.required


@pytest.mark.gpu
def test_katacr_val_on_a_real_model(tmp_path):
    """A real YOLO trained 1 epoch on 4 tiny frames of 2 names, and a fine-tune of it on 4 names (a
    mix's extended set): both validated on the 2-name yaml, as reg.katacr validates a candidate
    with more classes than tonight's 310. The same model drops 0; the 4-name one is scored over the
    yaml's classes."""
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO
    try:
        import cv2
        write = lambda p, img: cv2.imwrite(str(p), img)  # noqa: E731
    except ImportError:
        Image = pytest.importorskip("PIL.Image")
        write = lambda p, img: Image.fromarray(img).save(p)  # noqa: E731
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        for k in range(4):
            img = np.zeros((64, 64, 3), np.uint8)
            img[20:44, 20:44] = (40 * k, 200, 80)
            write(tmp_path / "images" / split / f"f{k}.png", img)
            (tmp_path / "labels" / split / f"f{k}.txt").write_text(f"{k % 2} 0.5 0.5 0.375 0.375\n", encoding="utf-8")
    data = tmp_path / "cr.yaml"
    data.write_text(f"path: {tmp_path.as_posix()}\ntrain: images/train\nval: images/val\n"
                    "names:\n  0: HogRider@ally\n  1: HogRider@enemy\n", encoding="utf-8")
    data4 = tmp_path / "cr_mix.yaml"
    data4.write_text(f"path: {tmp_path.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: HogRider@ally\n"
                     "  1: HogRider@enemy\n  2: BossBandit@ally\n  3: BossBandit@enemy\n", encoding="utf-8")
    common = dict(epochs=1, imgsz=64, batch=2, workers=0, device=0, plots=False, amp=False, project=str(tmp_path / "runs"),
                  exist_ok=True, verbose=False)
    YOLO("yolov8n.yaml").train(data=str(data), name="t", **common)
    last = tmp_path / "runs" / "t" / "weights" / "last.pt"
    reg = GE.katacr_regression(last, last, data, device=0, cache=tmp_path / "cache", log=lambda m: None)
    assert reg["drop"] == 0.0 and reg["candidate"]["ap50"] == reg["baseline"]["ap50"]
    assert (tmp_path / "cache" / file_sha256(last)[:12] / GE.KATACR_FILE).is_file()
    YOLO(str(last)).train(data=str(data4), name="t4", **common)
    last4 = tmp_path / "runs" / "t4" / "weights" / "last.pt"
    assert len(YOLO(str(last4)).names) == 4
    reg4 = GE.katacr_regression(last4, last, data, device=0, cache=tmp_path / "cache", log=lambda m: None)
    assert math.isfinite(reg4["candidate"]["map50"]) and math.isfinite(reg4["baseline"]["map50"])
    assert set(reg4["candidate"]["ap50"]) <= {"HogRider@ally", "HogRider@enemy"}
    assert set(reg4["baseline"]["ap50"]) <= {"HogRider@ally", "HogRider@enemy"}
