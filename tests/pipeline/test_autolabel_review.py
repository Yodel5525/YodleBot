"""N12: tools/vision_autolabel.py review, mark and plays (gold), and their pages in
scripts/vision/unit_harvest.py.

docs/NEW_CARDS_PLAN.md 5 and N12. Checked:
* the Wilson interval of a reviewed share (precision is 1 minus it);
* review on the synthetic scan of test_autolabel.py, emitted as in test_autolabel_emit.py: page A
  samples per (sheet, cluster) and auto-accepted label (gold: every labelled track), page B half
  in a birth window, the same seed draws the same pages, the pages land in <store>/<sid>/review/
  (never emit/ or gold/), a deleted raw falls back to the emitted image;
* mark: bad entries store nothing; keys and image names are stored (never tile numbers); more
  than 2 bad frames name the groups to check again; a mark makes the emit stale and is spent
  once the track is relabelled; a review recorded with nothing bad;
* gold plays from hand-made gold tracks of the Round 0 deck (Guards x3, Elite Barbarians x2, Boss
  Bandit continued by a look change and a live uid, the evo Minion Horde x6 at 5 elixir, Zap, a
  BushGoblin child of an off-deck Suspicious Bush, one of our units), M0's reads, deletions and
  additions by list, confirmation, and the freshness the gate checks; a live uid passed on to a
  later unit, or shared by two at once, joins no play; M0 at the play's own tick; a dry run's
  next step; tracks left unlabelled (emit --allow-pending) and a changed derivation refuse a
  confirmation; review numbers go on after the review folder is deleted;
* fragmented gold tracks as Round 0's list 1 had them, each one play under the capacity, reach and
  deploy-zone rules (and several under the old ones): Guards found again at their spot 4 s on and
  down the lane, a Minion Horde seen as six plays, an Elite Barbarian pair walking down a lane, three
  Hog Rider parts in 7 s; a new play of the card 20 s later stays separate; the speed and unit
  tables, and the river pinned to scripts/vision/geometry;
* the review's breaks, each refused: a part past the river joins only a unit that could have walked
  there (or one still seen within the margin); a part in its deploy zone behind a lost unit, or once
  its card has cycled (4 other plays), is a new play, unless it is a close fit (a long-lived unit's
  track lost 3 s, flagged); a champion cycles from her last sighting; a merge one missed play short
  of the cycle is flagged; a part past the river no unit explains is a play whose deploy went
  unseen, which a part back in the deploy zone never continues (rule 1, a live uid or reach); a
  second track leaves a unit's latest part alone; spell cards' units that no card summons (Heal)
  keep the spell rules, the Spirit Empress' forms are troops; a capacity overflow in the deploy window
  joins the deploy; a deploy-unseen play's earlier parts under another card; the
  two sides' deploy zones mirror each other; the list prints each play's merges by rule and warns
  of zone merges, long reach gaps and cycle gaps until confirmed, and an --add-play replaces a
  deploy-unseen play; the CLI's rule flags.
Synthetic fixtures only (npy images): no image codec, GPU, engine or emulator. The plays and
mark-parsing tests need no numpy.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "tools"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vision_autolabel as VA  # noqa: E402
from tests.pipeline.test_autolabel import EVENTS, TABLE, cli, fake_recording, run_ingest  # noqa: E402
from tests.pipeline.test_autolabel_emit import jsonl, label_synthetic  # noqa: E402

R0_SID = "20261029_R0_1"
R0_DECK = "boss-bandit,elite-barbarians,executioner,freeze,guards,hog-rider,minion-horde-ev1,zap"


@pytest.fixture(scope="module")
def keys():
    key_map, note = VA.load_key_map()
    assert note is None, note
    return key_map


# ---------------------------------------------------------------------------
# arithmetic and small pieces
# ---------------------------------------------------------------------------

def test_wilson():
    # values from a scratch run of the formula (z 1.96), to 4 decimals
    for (k, n), want in (((0, 24), (0.0, 0.1380)), ((2, 24), (0.0232, 0.2585)), ((3, 24), (0.0434, 0.3100)),
                         ((24, 24), (0.8620, 1.0)), ((1, 3), (0.0615, 0.7923)), ((0, 0), (0.0, 1.0))):
        assert VA.wilson(k, n) == pytest.approx(want, abs=5e-5), (k, n)
    assert VA.wilson(0, 24)[0] == 0.0 and VA.wilson(24, 24)[1] == 1.0
    with pytest.raises(ValueError):
        VA.wilson(5, 3)
    assert VA.share_text(3, 24) == "3 of 24: 12.5% [4.3%-31.0%] (precision 87.5% [69.0%-95.7%])"


def test_keys_groups_and_clock():
    assert [VA.track_key(t) for t in ("t17", "17", "T40.1", " t040.01 ", "x7", "17.", "")] == [
        "17", "17", "40.1", "40.1", None, None, None]
    assert VA.review_group({"source": "auto", "label": "Ronin"}, False) == "auto:Ronin"
    assert VA.review_group({"source": "owner", "sheet": 2, "cluster": 0}, False) == "s2c0"
    assert VA.review_group({"source": "owner", "sheet": 3, "cluster": None}, False) == "s3t"
    assert VA.review_group({"source": "auto", "label": "Ronin"}, True) == "gold:Ronin"
    assert VA.review_group(None, True, "BossBandit") == "gold:BossBandit"
    assert VA.review_seed("20261101_A_1", 2) == VA.review_seed("20261101_A_1", 2) != VA.review_seed("20261101_A_1", 3)
    assert (VA.clock_text(112.44), VA.clock_text(5), VA.clock_text(None)) == ("1:52.4", "0:05.0", "-")
    assert VA.bad_frame_numbers({"bad_frames": {"a.jpg": {"i": 12}, "b.jpg": {"i": None}, "00000030": {"i": 30}}}) == {12, 30}
    # the restated names of unit_harvest stay equal to its own
    pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH
    assert (VA.PART_LABEL, VA.PART_OURS) == (UH.DECISION_LABEL, UH.DECISION_OURS)
    assert VA.OUR_DECK == tuple(UH.DEFAULT_OUR_DECK)


def test_review_drawing_helpers():
    np = pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH

    img = np.zeros((20, 30, 3), np.uint8)
    UH.draw_box(img, (2.0, 3.0, 12.0, 9.0), (255, 0, 0), thickness=2)
    assert (img[3, 2:12] == (255, 0, 0)).all() and (img[4, 2:12] == (255, 0, 0)).all() and (img[8, 2:12] == (255, 0, 0)).all()
    assert (img[3:9, 2] == (255, 0, 0)).all() and (img[3:9, 11] == (255, 0, 0)).all()
    assert (img[5:7, 4:10] == 0).all() and (img[10:] == 0).all()
    edge = np.zeros((20, 30, 3), np.uint8)
    UH.draw_box(edge, (-5.0, -5.0, 10.0, 10.0), (0, 255, 0), thickness=1)     # the parts outside are left out
    assert (edge[9, 0:10] == (0, 255, 0)).all() and (edge[0:9, 9] == (0, 255, 0)).all() and (edge[0, :9] == 0).all()
    UH.draw_box(edge, (40.0, 40.0, 50.0, 50.0), (0, 0, 255))                    # nothing of it inside
    assert not (edge == (0, 0, 255)).all(axis=2).any()
    assert UH.resize_nearest(np.arange(12, dtype=np.uint8).reshape(3, 4, 1).repeat(3, 2), 2, 3)[:, :, 0].tolist() == [
        [1, 3], [5, 7], [9, 11]]
    tile = UH.text_tile("RAW GONE", 72)
    assert tile.shape == (72, 72, 3) and (tile[0, 0] == UH.CELL_BG).all() and (tile == UH.DIM_COLOUR).all(axis=2).any()
    grass = np.zeros((60, 40, 3), np.uint8)
    grass[:] = (82, 152, 70)
    dim = UH.sheet_crop(grass, (10.0, 10.0, 20.0, 30.0), 48, colour=UH.DIM_COLOUR)
    assert (dim == UH.DIM_COLOUR).all(axis=2).any() and not (dim == UH.BOX_COLOUR).all(axis=2).any()
    assert (UH.sheet_crop(grass, (10.0, 10.0, 20.0, 30.0), 48) == UH.BOX_COLOUR).all(axis=2).any()
    tiles = [UH.FrameTile("f0 t=0:01.0", grass, [((10, 10, 20, 30), "BOSSBANDIT E", UH.BOX_COLOUR)],
                          [((0, 40, 10, 50), UH.FLAG_COLOUR)], ["n3 i9 birth"]),
             UH.FrameTile("f1 t=0:02.0", None, note="raw deleted")]
    page = UH.render_frames(tiles, width=300, cols=2, header=["page B"])
    assert page.dtype == np.uint8 and page.shape[1] <= 300 and page.shape[2] == 3
    assert (page == UH.BOX_COLOUR).all(axis=2).any() and (page == UH.FLAG_COLOUR).all(axis=2).any()


# ---------------------------------------------------------------------------
# review and mark on the synthetic scan
# ---------------------------------------------------------------------------

def emitted(tmp_path, keys, sid="20261101_A_1"):
    """label_synthetic (h1 GiantBuffer auto, h2/h3 goblinstein on sheet 1 cluster 1, h4 Vines on
    sheet 2 cluster 0, 5.0 x, 5.1 ?), emitted with birth_s 0.3 as npy (20 images, 14 births)."""
    from scripts.vision import unit_harvest as UH

    paths, geom = label_synthetic(tmp_path, keys, sid)
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store")
    VA.emit_session(sid, config=UH.EmitConfig(birth_s=0.3, image_format="npy"), geometry=geom, log=lambda m: None,
                    table=TABLE, **kw)
    return paths, geom, kw


def test_review_pages_sample_groups_and_birth_frames(tmp_path, keys, capsys):
    np = pytest.importorskip("numpy")
    sid = "20261101_A_1"
    paths, _geom, kw = emitted(tmp_path, keys, sid)
    rows = jsonl(paths.emit_dir / VA.BOXES_FILE)
    assert len(rows) == 20 and sum(r["birth"] for r in rows) == 14
    before = {p for p in paths.store_dir.rglob("*") if p.is_file()}

    res = VA.review_session(sid, per_group=1, frames=8, log=lambda m: None, **kw)
    d = res.doc
    assert (res.review, res.kind, d["schema"], d["seed"]) == (1, "train", VA.REVIEW_SCHEMA, VA.review_seed(sid, 1))
    assert [g["group"] for g in d["page_a"]] == ["auto:GiantBuffer", "s1c1", "s2c0"]
    assert [g["tracks"] for g in d["page_a"]] == [1, 2, 1] and all(len(g["keys"]) == 1 for g in d["page_a"])
    assert d["page_a"][0]["keys"] == ["1"] and d["page_a"][1]["keys"][0] in ("2", "3") and d["page_a"][2]["keys"] == ["4"]
    pb = d["page_b"]
    assert len(pb) == 8 and sum(e["birth"] for e in pb) == 4 and [e["tile"] for e in pb] == list(range(8))
    assert [e["n"] for e in pb] == sorted(e["n"] for e in pb)
    by_image = {r["image"]: r for r in rows}
    assert all(e["image"] in by_image and by_image[e["image"]]["birth"] == e["birth"] for e in pb)
    assert all(e["keys"] == sorted({str(b[7]) for b in by_image[e["image"]]["boxes"] if b[7] is not None},
                                   key=VA._key_order) for e in pb)
    assert [im.stem for im in res.images] == ["review_1_a", "review_1_b"] and all(im.exists() for im in res.images)
    new = {p for p in paths.store_dir.rglob("*") if p.is_file()} - before
    assert new and all(paths.review_dir in p.parents for p in new)           # only the review folder is written
    assert json.loads((paths.review_dir / VA.REVIEW_FILE).read_text(encoding="utf-8")) == json.loads(json.dumps(d))
    assert (paths.review_dir / "review_1.json").is_file()

    # the same seed draws the same pages; every review gets the next number
    again = VA.review_session(sid, seed=d["seed"], per_group=1, frames=8, **kw)
    assert again.review == 2 and (again.doc["page_a"], again.doc["page_b"]) == (d["page_a"], d["page_b"])
    full = VA.review_session(sid, **kw)
    assert full.review == 3 and len(full.doc["page_b"]) == 20 and sum(e["birth"] for e in full.doc["page_b"]) == 14
    assert sorted(k for g in full.doc["page_a"] for k in g["keys"]) == ["1", "2", "3", "4"]
    st = {r["sid"]: r for r in VA.status(kw["labels_root"], kw["store"])["sessions"]}
    assert "review" in st[sid]["stages"] and any("review 3 not recorded" in n for n in st[sid]["notes"])
    # raws deleted after emit (D14): page A crops the emitted images instead
    for f in paths.images.glob("*.npy"):
        f.unlink()
    gone = VA.review_session(sid, per_group=1, frames=4, **kw)
    from scripts.vision import unit_harvest as UH
    assert gone.review == 4 and UH.load_image(gone.images[0]).shape[2] == 3 and np.ndim(UH.load_image(gone.images[1])) == 3
    assert cli(tmp_path, "review", sid, "--frames", "4", "--per-cluster", "1") == VA.EXIT_OK
    out = capsys.readouterr().out
    assert f"{sid} review 5 (train" in out and f"mark {sid} --review 5" in out and "page ->" in out


def test_review_shows_every_gold_track(tmp_path, keys):
    pytest.importorskip("numpy")
    from tests.pipeline.test_autolabel import synthetic_scan

    sid = "20261101_A_2"                                   # friendly 2: gold
    paths, geom = synthetic_scan(tmp_path, keys, sid)
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store")
    VA.make_sheet(sid, geometry=geom, log=lambda m: None, table=TABLE, **kw)
    VA.label_session(sid, "t1=GiantBuffer,t2=goblinstein,t3=goblinstein,t4=Vines,t5.0=x,t5.1=?", table=TABLE, **kw)
    with pytest.raises(VA.LabelError, match="run emit 20261101_A_2") as err:     # no emit yet
        VA.review_session(sid, **kw)
    assert err.value.code == VA.EXIT_CHECK
    VA.emit_session(sid, geometry=geom, log=lambda m: None, table=TABLE, **kw)
    res = VA.review_session(sid, per_group=1, frames=6, **kw)
    assert res.kind == "gold"
    assert [(g["group"], g["keys"]) for g in res.doc["page_a"]] == [
        ("gold:GiantBuffer", ["1"]), ("gold:Vines", ["4"]), ("gold:goblinstein", ["2", "3"])]
    pb = res.doc["page_b"]
    assert len(pb) == 6 and sum(e["birth"] for e in pb) == 3
    gold = {g["stem"]: g for g in jsonl(paths.gold_dir / VA.GOLD_FILE)}
    assert all(e["image"] in gold and gold[e["image"]]["n"] == e["n"] for e in pb)      # gold: the recorder stem
    assert not paths.emit_dir.exists() and all(paths.review_dir in im.parents for im in res.images)


def test_mark_stores_keys_and_images_and_refuses_bad_entries(tmp_path, keys, capsys):
    pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH

    sid = "20261101_A_1"
    paths, geom, kw = emitted(tmp_path, keys, sid)
    session = VA.load_session(sid, kw["labels_root"])
    rv = VA.review_session(sid, **kw).doc                   # review 1: 4 tracks, 20 frames
    pb = rv["page_b"]

    with pytest.raises(VA.LabelError) as err:               # every bad entry is listed, nothing is stored
        VA.mark_session(sid, bad_tracks="t99,t5,x7", bad_frames="30,f2", **kw)
    text = " | ".join(err.value.problems)
    for needle in ("t99 is not a labelled track", "t5 is not a labelled track of tracks.json (it was split: t5.0, t5.1)",
                   "'x7' is not a track key", "f30 is not on page B of review 1"):
        assert needle in text, needle
    assert len(err.value.problems) == 4 and err.value.code == VA.EXIT_INPUT and not paths.marks_file.exists()
    with pytest.raises(VA.LabelError, match="nothing to mark"):
        VA.mark_session(sid, **kw)
    with pytest.raises(VA.LabelError, match="no review 7"):
        VA.mark_session(sid, review=7, bad_frames="1", **kw)
    assert cli(tmp_path, "mark", sid, "--bad-tracks", "t99") == VA.EXIT_INPUT
    assert "nothing stored" in capsys.readouterr().out and not paths.marks_file.exists()

    # one bad track and three bad frames of review 1: stored as keys and image names
    m = VA.mark_session(sid, bad_tracks="t2", bad_frames="0,1,13", **kw)
    doc = json.loads(paths.marks_file.read_text(encoding="utf-8"))
    entries = {t["key"]: t for t in json.loads(paths.tracks_file.read_text(encoding="utf-8"))["tracks"]}
    assert doc["schema"] == VA.MARKS_SCHEMA and list(doc["bad_tracks"]) == ["2"]
    assert doc["bad_tracks"]["2"]["label_at"] == entries["2"]["at"] and doc["bad_tracks"]["2"]["review"] == 1
    assert sorted(doc["bad_frames"]) == sorted(pb[t]["image"] for t in (0, 1, 13))
    assert all(name.startswith(f"{sid}_") for name in doc["bad_frames"])                 # names, not tile numbers
    assert VA.bad_frame_numbers(doc) == {pb[t]["i"] for t in (0, 1, 13)}
    t = m.tally
    assert (t["review"], t["page_a_tracks"], t["bad_tracks"], t["page_b_frames"], t["bad_frames"]) == (1, 4, 1, 20, 3)
    assert t["wilson_frames"] == pytest.approx(VA.wilson(3, 20), abs=1e-4)
    assert t["precision_frames"] == pytest.approx((1 - VA.wilson(3, 20)[1], 1 - VA.wilson(3, 20)[0]), abs=1e-4)
    on_bad = {k for tile in (0, 1, 13) for k in pb[tile]["keys"]}
    assert m.recheck == sorted({VA.review_group(entries[k], False) for k in on_bad}) and m.recheck
    assert len(m.hints) == len(m.recheck) and all("--force" in h for h in m.hints)
    assert doc["reviews"] == [json.loads(json.dumps(t))]
    # the mark makes the emit stale (the next emit masks t2) until it is emitted again
    assert VA.MARKS_CHANGED in VA.emit_fresh(paths, session)
    st = {r["sid"]: r for r in VA.status(kw["labels_root"], kw["store"])["sessions"]}
    assert any("marks newer than emit" in n for n in st[sid]["notes"])
    # take one frame back; record review 2 with nothing bad
    VA.mark_session(sid, clear=pb[13]["image"], **kw)
    doc = json.loads(paths.marks_file.read_text(encoding="utf-8"))
    assert len(doc["bad_frames"]) == 2 and doc["reviews"][0]["bad_frames"] == 2 and doc["reviews"][0]["recheck"] == []
    with pytest.raises(VA.LabelError, match="neither a marked track"):
        VA.mark_session(sid, clear="t3", **kw)
    with pytest.raises(VA.LabelError, match="fresh emit"):                # a stale emit is not reviewed
        VA.review_session(sid, **kw)
    VA.emit_session(sid, config=UH.EmitConfig(birth_s=0.3, image_format="npy"), geometry=geom, log=lambda m: None,
                    table=TABLE, force=True, **kw)
    parts = {p["key"]: p for p in jsonl(paths.emit_dir / VA.PARTS_FILE)}
    assert (parts["2"]["decision"], parts["2"]["src"]) == ("mask", "bad") and VA.emit_fresh(paths, session) == []
    rv2 = VA.review_session(sid, **kw).doc
    assert sorted(k for g in rv2["page_a"] for k in g["keys"]) == ["1", "3", "4"]     # t2 is masked now
    VA.mark_session(sid, review=2, **kw)
    doc = json.loads(paths.marks_file.read_text(encoding="utf-8"))
    assert [(r["review"], r["bad_tracks"], r["bad_frames"]) for r in doc["reviews"]] == [(1, 1, 2), (2, 0, 0)]
    shown = doc["reviews"][1]["page_b_frames"]
    assert shown == len(rv2["page_b"]) and doc["reviews"][1]["wilson_frames"] == [
        0.0, pytest.approx(VA.wilson(0, shown)[1], abs=1e-4)]
    before = paths.marks_file.read_text(encoding="utf-8")
    dry = VA.mark_session(sid, bad_tracks="t3", dry_run=True, **kw)
    assert not dry.written and paths.marks_file.read_text(encoding="utf-8") == before
    # the review folder (gitignored, large) deleted: the numbers go on from marks.json, never reused
    shutil.rmtree(paths.review_dir)
    assert VA.next_review_number(paths) == 3 and VA.review_session(sid, **kw).review == 3
    # label --force writes a newer "at": the mark is spent
    VA.label_session(sid, "t2=goblinstein", force=True, table=TABLE, clock=lambda: time.time() + 120, **kw)
    tdoc = json.loads(paths.tracks_file.read_text(encoding="utf-8"))
    assert VA.effective_bad_tracks(doc, tdoc) == set()
    fresh = VA.emit_fresh(paths, session)
    assert VA.TRACKS_CHANGED in fresh and VA.MARKS_CHANGED in fresh       # the emit masked t2: emit again
    with pytest.raises(VA.LabelError, match="fresh emit") as err:          # review refuses a stale emit
        VA.review_session(sid, **kw)
    assert err.value.code == VA.EXIT_CHECK


# ---------------------------------------------------------------------------
# gold plays (no numpy)
# ---------------------------------------------------------------------------

def gold_part(key, label, t, xy, *, last_t=None, last=None, uid=None, h=None, part=0, decision="label", n=None):
    """A parts.jsonl line as a gold emit writes it (positions in world mt, n = (t - 10) * 10)."""
    return {"key": key, "h": int(key.split(".")[0]) if h is None else h, "part": part, "decision": decision,
            "label": label if decision == "label" else None, "side": 1 if decision == "label" else 0, "src": "owner",
            "type": "unit", "card": None, "form": 0, "first_n": int(round((t - 10.0) * 10)) if n is None else n,
            "last_n": 0, "first_t": t, "last_t": t + 2.0 if last_t is None else last_t, "hk": 1, "uid": uid,
            "birth": list(xy), "last": list(last or xy), "skipped": 0, "boxes": []}


# the opponent deploys in the top half (y < 15 tiles; the river is y 15-17), we in the bottom one
R0_PARTS = [gold_part("10", "SkeletonWarrior", 20.0, (9000, 12000), uid="1:10"),       # Guards: 3 units
            gold_part("11", "SkeletonWarrior", 20.1, (9500, 11700), uid="1:11"),
            gold_part("12", "SkeletonWarrior", 20.3, (8600, 11600), uid="1:12"),
            gold_part("20", "AngryBarbarian", 30.0, (4000, 10000)),                      # Elite Barbarians: 2
            gold_part("21", "AngryBarbarian", 30.05, (4800, 10000)),
            # Boss Bandit: a look change (part 1 of h30), then a new harvest track of the same live uid
            gold_part("30.0", "BossBandit", 40.0, (14000, 8000), last_t=43.0, last=(14000, 12000), uid="1:30", h=30),
            gold_part("30.1", "BossBandit", 43.05, (14000, 12100), last_t=44.0, last=(14000, 14000), h=30, part=1),
            gold_part("31", "BossBandit", 45.0, (14000, 15000), last_t=50.0, uid="1:30"),
            *[gold_part(str(40 + j), "Minion" if j < 3 else "MinionHorde_EV1", 50.0 + 0.05 * j,
                        (9000 + 300 * j, 14000 + 200 * j), uid=f"1:{40 + j}") for j in range(6)],   # the evo Horde
            gold_part("50", "zap", 60.0, (9000, 12000)),
            gold_part("59", "SuspiciousBush", 65.0, (5000, 12000)),                     # off the declared deck
            gold_part("60", "BushGoblin", 70.0, (5100, 12100)),                         # its child
            gold_part("70", None, 40.0, (3000, 26000), decision="ours", n=300)]         # one of ours (o)


def hand_gold(tmp_path, keys, parts=R0_PARTS):
    """The Round 0 session ingested as gold (natural capture), a tracks.json, and a gold emit
    written by hand: parts.jsonl, gold.jsonl (800 battle images from t 10.0 every 0.1 s, i = 5 n,
    our HogRider on images 300-301 as part 70), emit.json; plus a scan_frames.jsonl whose live
    chain read GoldenKnight at 40.3, Zap at 60.2 and Minions at 80.0."""
    root = tmp_path / "gold_rec" / R0_SID
    fake_recording(root)
    run_ingest(tmp_path, root, R0_DECK, key_map=keys, sid=R0_SID, role="gold", capture="natural")
    paths = VA.session_paths(R0_SID, tmp_path / "labels", tmp_path / "store")
    entries = [{"key": p["key"], "h": p["h"], "part": p["part"], "label": p["label"],
                "code": None if p["decision"] == "label" else "o", "source": "owner", "sheet": 1, "cluster": 0,
                "at": "2026-11-01T21:00:00"} for p in parts]
    VA.write_json(paths.tracks_file, {"schema": VA.TRACKS_SCHEMA, "sid": R0_SID, "tracks": entries, "split": []})
    paths.gold_dir.mkdir(parents=True)
    (paths.gold_dir / VA.PARTS_FILE).write_text("".join(json.dumps(p) + "\n" for p in parts), encoding="utf-8")
    rows = []
    for n in range(800):
        ours = [{"label": "HogRider", "side": 0, "box": [1, 2, 3, 4], "type": "unit", "src": "live", "how": "smooth",
                 "key": "70"}] if n in (300, 301) else []
        rows.append({"image": f"images/{5 * n:08d}.jpg", "stem": f"{5 * n:08d}", "i": 5 * n, "n": n,
                     "t": round(10.0 + 0.1 * n, 4), "hk": 1, "size": [747, 1328], "boxes": ours, "ignore": []})
    (paths.gold_dir / VA.GOLD_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    VA.write_json(paths.gold_dir / VA.EMIT_JSON, {
        "schema": VA.EMIT_SCHEMA, "sid": R0_SID, "kind": VA.KIND_GOLD, "role": VA.ROLE_GOLD, "created": "2026-11-01T22:00:00",
        "tracks_sha1": VA.sha1_file(paths.tracks_file), "bad_tracks": [], "battle_t0": {"1": 10.0}})
    paths.scan_dir.mkdir(parents=True)
    live = [{"n": 0, "t": 10.0, "plays": []},
            {"n": 303, "t": 40.3, "plays": [{"card": 12, "card_name": "GoldenKnight", "tick": 606, "x_mt": 14100,
                                             "y_mt": 8100, "kind": "unit"}]},
            {"n": 502, "t": 60.2, "plays": [{"card": 40, "card_name": "Zap", "tick": 1004, "x_mt": 9000, "y_mt": 12000,
                                             "kind": "spell"}]},
            {"n": 700, "t": 80.0, "plays": [{"card": 7, "card_name": "Minions", "tick": 1400, "x_mt": 3000,
                                             "y_mt": 5000, "kind": "unit"}]}]
    (paths.scan_dir / "scan_frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in live), encoding="utf-8")
    return paths


def test_plays_from_hand_made_gold_tracks(tmp_path, keys, capsys):
    paths = hand_gold(tmp_path, keys)
    session = VA.load_session(R0_SID, tmp_path / "labels")
    assert VA.is_gold(session) and VA.emit_fresh(paths, session) == []
    assert [c.card for c in VA.session_deck(session)] == ["BossBandit", "AngryBarbarians", "AxeMan", "Freeze",
                                                         "SkeletonWarriors", "HogRider", "MinionHorde", "Zap"]
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE, key_map=keys, event_cards=EVENTS)
    deck = VA.deck_cards(session)
    for label, want in (("SkeletonWarrior", True), ("MinionHorde_EV1", True), ("zap", True), ("BossBandit", True),
                        ("BushGoblin", False)):
        assert VA.starts_play(label, deck, TABLE) is want, label
    assert VA.starts_play("HealSpirit", ["Heal"], TABLE)                        # a spell seen as its units
    assert VA.starts_play("MergeMaiden_Mounted", ["MergeMaiden"], TABLE)
    assert not VA.starts_play("Goblin", ["GoblinCurse"], TABLE, {"GoblinCurse"})   # the curse's goblins: children
    assert VA.play_cost("MergeMaiden", ["MergeMaiden_Normal"], TABLE, EVENTS) == 3
    assert VA.play_cost("MergeMaiden", ["MergeMaiden_Mounted"], TABLE, EVENTS) == 6

    res = VA.plays_session(R0_SID, **kw)
    got = [(p["id"], p["card"], len(p["units"]), p["form"], p["cost"], p["kind"], p["m0"]) for p in res.plays]
    assert got == [("p1", "SkeletonWarriors", 3, 0, 3, "unit", None), ("p2", "AngryBarbarians", 2, 0, 6, "unit", None),
                   ("p3", "BossBandit", 1, 0, 6, "unit", "GoldenKnight"), ("p4", "MinionHorde", 6, 1, 5, "unit", None),
                   ("p5", "Zap", 1, 0, 2, "spell", "Zap"), ("p6", "SuspiciousBush", 1, 0, 2, "unit", None)]
    boss = res.plays[2]
    assert boss["parts"] == ["30.0", "30.1", "31"] and boss["continued"] == {"30.1": "split", "31": "uid"}
    # comparable with EventDeriver's OppPlay: absolute t (the capture clock), the first unit's position,
    # the time tick vision_replay --ticks time would give (20 a second from the first image)
    assert (boss["t"], boss["t_rel"], boss["x_mt"], boss["y_mt"], boss["tick"], boss["n"], boss["i"]) == (
        40.0, 30.0, 14000.0, 8000.0, 600, 300, 1500)
    assert res.plays[3]["labels"] == {"Minion": 3, "MinionHorde_EV1": 3}
    assert res.plays[5]["off_deck"] and res.plays[5]["children"] == ["60"]
    assert [(c["key"], c["play"]) for c in res.children] == [("60", "p6")]
    # ours at y 26 is in our own deploy zone (the bottom one): no deploy-unseen play
    assert [(p["card"], p["units"], p["deploy_unseen"]) for p in res.own] == [("HogRider", ["70"], False)]
    assert res.doc["m0"] == {"matched": 1, "read": 2, "of": 6, "live": 3, "phantoms": 1}
    assert any("p6 SuspiciousBush is not in the declared deck" in w for w in res.warnings)
    assert res.list_no == 1 and not res.doc["confirmed"] and VA.plays_fresh(paths) == [VA.PLAYS_UNCONFIRMED]
    st = {r["sid"]: r for r in VA.status(tmp_path / "labels", tmp_path / "store")["sessions"]}
    assert f"plays not confirmed: plays {R0_SID} --list 1 --confirm" in st[R0_SID]["notes"]

    # edits name list 1; every entry is checked first and nothing is stored on a bad one
    with pytest.raises(VA.LabelError, match="pass --list 1"):
        VA.plays_session(R0_SID, bad_plays="p1", **kw)
    with pytest.raises(VA.LabelError) as err:
        VA.plays_session(R0_SID, list_no=1, bad_plays="p9", add_plays=["0:75=freeze", "0:42.5=knight"], **kw)
    assert len(err.value.problems) == 3 and "p9 is not on list 1" in err.value.problems[0]
    assert "seconds must be below 60" in err.value.problems[1] and "not in the declared deck" in err.value.problems[2]
    assert json.loads(paths.plays_file.read_text(encoding="utf-8"))["list"] == 1
    res2 = VA.plays_session(R0_SID, list_no=1, bad_plays="p1", add_plays=["0:42.5@9,12=freeze"], **kw)
    assert [(p["id"], p["card"], p["source"], p["t"]) for p in res2.plays] == [
        ("p1", "AngryBarbarians", "derived", 30.0), ("p2", "BossBandit", "derived", 40.0),
        ("p3", "MinionHorde", "derived", 50.0), ("p4", "Freeze", "added", 52.5), ("p5", "Zap", "derived", 60.0),
        ("p6", "SuspiciousBush", "derived", 65.0)]
    fr = res2.plays[3]
    assert (fr["kind"], fr["cost"], fr["x_mt"], fr["y_mt"], fr["t_rel"]) == ("spell", 4, 9000.0, 12000.0, 42.5)
    assert fr["tick"] == 850                                   # (52.5 - 10.0) * 20, as a derived play's
    assert res2.doc["deleted"][0]["card"] == "SkeletonWarriors" and res2.list_no == 2
    with pytest.raises(VA.LabelError, match=f"the latest list of {R0_SID} is 2"):
        VA.plays_session(R0_SID, list_no=1, confirm=True, **kw)
    VA.plays_session(R0_SID, list_no=2, confirm=True, **kw)
    assert VA.plays_fresh(paths) == [] and VA.plays_notes(paths) == []
    again = VA.plays_session(R0_SID, **kw)                     # derived again: the same list stays confirmed
    assert again.list_no == 4 and again.doc["confirmed"] and len(again.plays) == 6
    gone = VA.plays_session(R0_SID, list_no=4, bad_plays="p4", **kw)          # the added Freeze taken back
    assert [p["card"] for p in gone.plays] == ["AngryBarbarians", "BossBandit", "MinionHorde", "Zap", "SuspiciousBush"]
    assert not gone.doc["confirmed"] and any("changed since list 4 was confirmed" in w for w in gone.warnings)
    VA.plays_session(R0_SID, list_no=5, confirm=True, **kw)
    assert cli(tmp_path, "plays", R0_SID, "--dry-run") == VA.EXIT_OK
    out = capsys.readouterr().out
    assert "play list 7 (dry run: not stored)" in out and "M0: GoldenKnight" in out and "confirmed" in out
    assert json.loads(paths.plays_file.read_text(encoding="utf-8"))["list"] == 6

    # tracks.json changed: the play list is stale for the gate, and plays refuses the stale emit
    tdoc = json.loads(paths.tracks_file.read_text(encoding="utf-8"))
    tdoc["tracks"][0]["at"] = "2026-11-02T10:00:00"
    VA.write_json(paths.tracks_file, tdoc)
    assert VA.plays_fresh(paths) == [VA.PLAYS_TRACKS]
    assert VA.plays_notes(paths) == [f"plays stale (tracks.json changed since): plays {R0_SID}"]
    with pytest.raises(VA.LabelError, match="fresh emit") as err:
        VA.plays_session(R0_SID, **kw)
    assert err.value.code == VA.EXIT_CHECK
    # a training session has no gold plays
    doc = json.loads(paths.session_file.read_text(encoding="utf-8"))
    VA.write_json(paths.session_file, dict(doc, role=VA.ROLE_TRAIN, capture=VA.CAPTURE_HARVEST))
    with pytest.raises(VA.LabelError, match="not a gold session") as err:
        VA.plays_session(R0_SID, **kw)
    assert err.value.code == VA.EXIT_INPUT


def test_play_specs_deletions_and_m0_matching(keys):
    deck = VA.parse_deck(R0_DECK, VA.CardResolver(TABLE, keys, EVENTS))
    resolver = VA.CardResolver(TABLE, keys, EVENTS)
    battles, frame_t = {1: 10.0, 2: 300.0}, {1234: (77.7, 1)}
    add = lambda spec, **k: VA.parse_add_play(spec, battles, resolver, deck, frame_t=frame_t, table=TABLE, **k)  # noqa: E731
    p, why = add("b2/1:52=freeze")
    assert why is None and (p["t"], p["hk"], p["t_rel"], p["card"], p["kind"], p["source"]) == (412.0, 2, 112.0, "Freeze",
                                                                                                "spell", "added")
    p, _ = add("n1234@9.1,12=minion-horde-ev1")
    assert (p["t"], p["hk"], p["n"], p["card"], p["form"], p["cost"], p["x_mt"], p["y_mt"], p["kind"]) == (
        77.7, 1, 1234, "MinionHorde", 1, 5, 9100.0, 12000.0, "unit")
    for spec, needle in (("1:52=freeze", "write b<K>/ first"), ("b3/0:10=zap", "no battle b3"), ("n99=zap", "not a battle image"),
                         ("0:61=zap", "below 60"), ("b1/0:10=ronin", "not in the declared deck"), ("1.52=zap", "is not [bK/]"),
                         ("b1/0:10=super-knight", "SuperKnight")):
        q, why = add(spec)
        assert q is None and needle in why, (spec, why)
    q, why = add("b1/0:10=ronin", off_deck=True)
    assert why is None and q["off_deck"] and q["card"] == "Ronin"
    # a stored deletion holds within 0.2 s of its card; the rest stays, numbered by time
    derived = [{"t": 10.0, "card": "Zap"}, {"t": 20.0, "card": "BossBandit"}, {"t": 20.1, "card": "Zap"}]
    children = [{"key": "60", "derived": 1}, {"key": "61", "derived": None}]
    final, kids, stale = VA.assemble_plays(derived, children, [{"card": "BossBandit", "t": 20.15},
                                                               {"card": "Freeze", "t": 5.0}],
                                           [{"t": 15.0, "card": "Freeze", "source": "added"}])
    assert [(p["id"], p["card"]) for p in final] == [("p1", "Zap"), ("p2", "Freeze"), ("p3", "Zap")]
    assert [(c["key"], c["play"]) for c in kids] == [("60", None), ("61", None)] and stale == [{"card": "Freeze", "t": 5.0}]
    # M0 matching: a right read is not taken by a nearer wrong one (same card first), within 1 s and 3 tiles
    gold = [{"t": 10.0, "card": "BossBandit", "x_mt": 9000, "y_mt": 10000},
            {"t": 10.4, "card": "Zap", "x_mt": 9500, "y_mt": 10500}]
    live = [{"t": 10.3, "card_name": "GoldenKnight", "x_mt": 9000, "y_mt": 10100},
            {"t": 10.5, "card_name": "Zap", "x_mt": 9400, "y_mt": 10400},
            {"t": 30.0, "card_name": "Minions", "x_mt": 3000, "y_mt": 5000},
            {"t": 10.1, "card_name": "BossBandit", "x_mt": 20000, "y_mt": 30000}]       # too far away
    assert VA.match_m0(gold, live) == {0: 0, 1: 1}
    assert VA.match_m0(gold, [{"t": 11.5, "card_name": "BossBandit", "x_mt": None, "y_mt": None}]) == {}
    assert VA.match_m0(gold, [{"t": 11.01, "card_name": "BossBandit", "x_mt": None, "y_mt": None}]) == {1: 0}
    assert VA.match_m0(gold, [{"t": 10.9, "card_name": "BossBandit", "x_mt": None, "y_mt": None}]) == {0: 0}


def test_a_live_uid_alone_does_not_merge_plays(keys):
    deck = [c.card for c in VA.parse_deck(R0_DECK, VA.CardResolver(TABLE, keys, EVENTS))]
    # Guards at 20.0; 16 s later a new Guards play whose first unit got the uid 1:10 back (the tracker
    # keeps a uid for a unit of the class born where one died): two plays, the second whole
    later = [gold_part("80", "SkeletonWarrior", 36.1, (9000, 12000), uid="1:10"),
             gold_part("81", "SkeletonWarrior", 36.2, (9400, 11800), uid="1:81"),
             gold_part("82", "SkeletonWarrior", 36.3, (8700, 11700), uid="1:82")]
    plays, _kids = VA.derive_plays(R0_PARTS[:3] + later, deck, TABLE, event_cards=EVENTS)
    assert [(p["card"], p["t"], p["units"]) for p in plays] == [("SkeletonWarriors", 20.0, ["10", "11", "12"]),
                                                               ("SkeletonWarriors", 36.1, ["80", "81", "82"])]
    assert all(not p["continued"] for p in plays)
    # two Horde minions at once with one uid: still six units of one play, none a continuation
    horde = [gold_part(str(40 + j), "Minion", 50.0 + 0.05 * j, (9000 + 300 * j, 14000 + 200 * j),
                       uid="1:40" if j < 2 else f"1:{40 + j}") for j in range(6)]
    plays, _kids = VA.derive_plays(horde, deck, TABLE, event_cards=EVENTS)
    assert len(plays) == 1 and len(plays[0]["units"]) == 6 and plays[0]["continued"] == {}
    # the same uid born soon after the unit's last part ended continues it (Boss Bandit 30.1 -> 31: 1.0 s)
    plays, _kids = VA.derive_plays(R0_PARTS[5:8], deck, TABLE, event_cards=EVENTS)
    assert len(plays) == 1 and plays[0]["continued"] == {"30.1": "split", "31": "uid"}
    # 1.6 s after 30.1 ended (past reacquire_s), 11 tiles from where it was (past its reach): the uid alone
    # joins nothing, a new play
    late = dict(R0_PARTS[7], first_t=45.6, last_t=50.0, birth=[3000, 12000], last=[3000, 12000])
    plays, _kids = VA.derive_plays(R0_PARTS[5:7] + [late], deck, TABLE, event_cards=EVENTS)
    assert [p["units"] for p in plays] == [["30.0"], ["31"]]
    # the same 1.6 s later 1 tile on (within its reach: 1.5 tiles/s x 1.25 x 1.6 s + 2): the unit found again
    late = dict(R0_PARTS[7], first_t=45.6, last_t=50.0)
    plays, _kids = VA.derive_plays(R0_PARTS[5:7] + [late], deck, TABLE, event_cards=EVENTS)
    assert [p["units"] for p in plays] == [["30.0"]] and plays[0]["continued"] == {"30.1": "split", "31": "reach"}


# Round 0's play list 1 (60 derived plays for about 18) over-counted fragmented tracks; the tests below
# rebuild its patterns (keys as on that list). OLD_RULES: no reach and no deploy-zone rule, as before.
OLD_RULES = VA.PlayConfig(reach_gap_s=0.0, zone_y_tiles=32.0)


def r0_cards(keys):
    return [c.card for c in VA.parse_deck(R0_DECK, VA.CardResolver(TABLE, keys, EVENTS))]


def test_play_tables_and_the_river():
    speeds = VA.load_speeds()
    # RoyaleAPI's Speed / 60 (tiles a second); the Boss Bandit from SPEED_EXTRA
    assert {c: speeds[c] for c in ("SkeletonWarrior", "Minion", "AngryBarbarian", "HogRider", "AxeMan", "BossBandit")} == {
        "SkeletonWarrior": 1.5, "Minion": 1.5, "AngryBarbarian": 1.5, "HogRider": 2.0, "AxeMan": 1.0, "BossBandit": 1.5}
    units = VA.load_units_per_play()
    assert {c: units.get(c) for c in ("SkeletonWarriors", "MinionHorde", "AngryBarbarians", "HogRider", "AxeMan",
                                      "BossBandit", "Zap", "MergeMaiden")} == {
        "SkeletonWarriors": 3, "MinionHorde": 6, "AngryBarbarians": 2, "HogRider": 1, "AxeMan": 1, "BossBandit": 1,
        "Zap": None, "MergeMaiden": None}                     # a spell, and a card with forms: no cap
    assert VA.load_speeds("no/such/table.json") == VA.SPEED_EXTRA and VA.load_units_per_play("no/such.json") == {}
    pytest.importorskip("numpy")
    from scripts.vision import geometry as G
    assert VA.ARENA_Y_MT == G.HEIGHT_MT and VA.PlayConfig().zone_y_tiles * VA.MT_PER_TILE == G.RIVER_Y_MT[1]


def test_a_guards_play_fragmented_at_its_spot_is_one_play(keys):
    deck = r0_cards(keys)
    # two guards put down at the right princess tower and lost at 21.0; 3.0 and 3.3 s later (past reacquire_s)
    # two parts near where they were lost; then one 11 tiles down the lane, in Yodle2's half (p1, p3, p4, p7)
    parts = [gold_part("36", "SkeletonWarrior", 20.0, (14400, 8700), last_t=21.0, last=(14500, 8800)),
             gold_part("38.0", "SkeletonWarrior", 20.1, (15000, 8500), last_t=21.0, last=(15100, 8600), h=38),
             gold_part("54", "SkeletonWarrior", 24.0, (16500, 7700), last_t=26.0),
             gold_part("56", "SkeletonWarrior", 24.3, (13500, 9000), last_t=26.0, last=(13600, 12000)),
             gold_part("106.0", "SkeletonWarrior", 31.9, (14100, 20300), h=106)]
    plays, kids = VA.derive_plays(parts, deck, TABLE, event_cards=EVENTS)
    assert kids == [] and [(p["card"], p["t"], p["units"]) for p in plays] == [("SkeletonWarriors", 20.0, ["36", "38.0"])]
    # each a lost guard found again within 1.5 tiles/s x 1.25 x the gap + 2 tiles of where it was last seen
    assert plays[0]["continued"] == {"54": "reach", "56": "reach", "106.0": "reach"}
    assert plays[0]["parts"] == ["36", "38.0", "54", "56", "106.0"]
    old, _kids = VA.derive_plays(parts, deck, TABLE, event_cards=EVENTS, config=OLD_RULES)
    assert [p["units"] for p in old] == [["36", "38.0"], ["54"], ["56"], ["106.0"]]          # list 1's p1 p3 p4 p7


def test_a_minion_horde_seen_as_six_plays_is_one(keys):
    deck = r0_cards(keys)

    def walk(key, t, xy, last_t, dy, **kw):
        return gold_part(key, "Minion", t, xy, last_t=last_t, last=(xy[0], xy[1] + dy), **kw)
    # 3 minions boxed at 39.7, 3 more 1.2 s and 3 tiles off (past window_s and radius_tiles), then 7 parts more
    # of the same 6 (p10-p15 of list 1): one of them a second track on a minion while all 6 are seen
    parts = [walk("169.0", 39.7, (15900, 6700), 42.0, 3000, h=169), walk("171.0", 39.75, (16400, 6900), 42.0, 3000, h=171),
             walk("174.1", 39.8, (15400, 6900), 42.0, 3000, h=174, part=1),
             walk("178.0", 40.9, (13900, 9000), 42.2, 1500, h=178), walk("177.1", 40.95, (14400, 9200), 42.2, 1500, h=177,
                                                                          part=1),
             walk("196", 41.0, (13400, 9200), 42.2, 1500),
             gold_part("197", "Minion", 41.1, (13950, 9150), last_t=41.6),            # on top of 178.0, which is seen
             gold_part("198", "Minion", 42.5, (14600, 10300)), gold_part("199.0", "Minion", 42.55, (15300, 10500), h=199),
             gold_part("202", "Minion", 44.1, (14600, 9800)),
             gold_part("221.0", "Minion", 47.9, (13200, 14200), h=221), gold_part("224", "Minion", 47.95, (14000, 14000)),
             gold_part("237", "Minion", 50.6, (14800, 19900))]
    plays, kids = VA.derive_plays(parts, deck, TABLE, event_cards=EVENTS)
    assert kids == [] and len(plays) == 1
    p = plays[0]
    # six units (MinionHorde deploys 6: capacity); the second three join as units unseen since the deploy
    assert p["units"] == ["169.0", "171.0", "174.1", "178.0", "177.1", "196"] and len(p["parts"]) == 13
    # 197 is a second track on 178.0, seen while the play is full (capacity)
    assert p["continued"] == {"197": "capacity", "198": "reacquired", "199.0": "reacquired", "202": "reach",
                              "221.0": "reach", "224": "reach", "237": "reach"}
    assert p["joined"] == {"178.0": "reach", "177.1": "reach", "196": "reach"} and p["gaps"]["178.0"] == 1.2
    old, _kids = VA.derive_plays(parts, deck, TABLE, event_cards=EVENTS, config=OLD_RULES)
    assert [q["t"] for q in old] == [39.7, 40.9, 44.1, 47.9, 50.6]


def test_an_elite_barbarian_pair_walking_down_a_lane_is_one_play(keys):
    deck = r0_cards(keys)
    # put down behind the king tower (1.1 s apart: past window_s), a second track on the right one while both are
    # seen, then the left one found again three times down the left lane and the right one once, in Yodle2's
    # half (p16-p22)
    parts = [gold_part("246", "AngryBarbarian", 57.9, (7800, 300), last_t=60.0, last=(6800, 3500)),
             gold_part("248", "AngryBarbarian", 59.0, (9800, 1300), last_t=61.0, last=(11000, 4800)),
             gold_part("247.1", "AngryBarbarian", 59.05, (9900, 1400), last_t=59.6, h=247, part=1),
             gold_part("252", "AngryBarbarian", 63.5, (4900, 9700), last_t=64.5, last=(4300, 11500)),
             gold_part("226.1", "AngryBarbarian", 67.9, (3800, 16900), last_t=69.0, last=(4500, 18500), h=226, part=1),
             gold_part("280", "AngryBarbarian", 69.3, (14300, 18800)),
             gold_part("283", "AngryBarbarian", 70.6, (6000, 18600))]
    plays, kids = VA.derive_plays(parts, deck, TABLE, event_cards=EVENTS)
    assert kids == [] and [(p["t"], p["units"], p["cost"]) for p in plays] == [(57.9, ["246", "248"], 6)]
    assert plays[0]["continued"] == {"247.1": "capacity", "252": "reach", "226.1": "reach", "280": "reach",
                                     "283": "reach"}
    old, _kids = VA.derive_plays(parts, deck, TABLE, event_cards=EVENTS, config=OLD_RULES)
    assert [q["units"] for q in old] == [["246"], ["248", "247.1"], ["252"], ["226.1"], ["280"], ["283"]]


def test_three_hog_rider_parts_in_7_s_are_one_play(keys):
    deck = r0_cards(keys)
    # put down at the right bridge, lost over the river, found in Yodle2's half 1.8 s later, then again 2.9 s
    # later where the Cannon pulled it (p6, p8, p9: three plays on list 1, and a Hog Rider costs 4)
    hog = [gold_part("100", "HogRider", 22.0, (14300, 14300), last_t=23.0, last=(14300, 18300)),
           gold_part("72.1", "HogRider", 24.8, (14600, 19900), last_t=26.0, last=(14000, 21500), h=72, part=1),
           gold_part("120.1", "HogRider", 28.9, (12900, 22100), h=120, part=1)]
    plays, kids = VA.derive_plays(hog, deck, TABLE, event_cards=EVENTS)
    assert kids == [] and [(p["t"], p["units"], p["cost"]) for p in plays] == [(22.0, ["100"], 4)]
    assert plays[0]["continued"] == {"72.1": "reach", "120.1": "reach"}
    # without the reach rule the deploy zone still holds them: neither starts in Yodle2's half, and the play
    # is full (1 unit), so both continue its unit
    plays, _kids = VA.derive_plays(hog, deck, TABLE, event_cards=EVENTS, config=VA.PlayConfig(reach_gap_s=0.0))
    assert len(plays) == 1 and plays[0]["continued"] == {"72.1": "zone", "120.1": "zone"}
    assert len(VA.derive_plays(hog, deck, TABLE, event_cards=EVENTS, config=OLD_RULES)[0]) == 3
    # its deploy unlabelled: the first part seen is in Yodle2's half, where no play starts, and no earlier Hog
    # Rider could have walked there: a play whose deploy went unseen, dated at that first sighting (as M0
    # dates one), and the next part continues it
    plays, kids = VA.derive_plays(hog[1:], deck, TABLE, event_cards=EVENTS)
    assert kids == [] and [(p["t"], p["units"], p["parts"], p["continued"], p["deploy_unseen"], p["earlier"])
                           for p in plays] == [(24.8, ["72.1"], ["72.1", "120.1"], {"120.1": "reach"}, True, [])]
    # its track began as another card's part (the deploy, labelled Boss Bandit?): named for the owner to check
    boss = gold_part("72.0", "BossBandit", 21.9, (14300, 14000), last_t=24.7, last=(14500, 19500), h=72)
    plays, _kids = VA.derive_plays([boss] + hog[1:], deck, TABLE, event_cards=EVENTS)
    assert [(p["card"], p["deploy_unseen"]) for p in plays] == [("BossBandit", False), ("HogRider", True)]
    assert plays[1]["earlier"] == [{"key": "72.0", "label": "BossBandit", "t": 21.9}]
    # our own units deploy in the bottom half: the defender's half is the top one (71: 28 s after 70 was last
    # seen, past zone_gap_s)
    ours = [dict(gold_part("70", None, 40.0, (3000, 26000), decision="ours")),
            dict(gold_part("71", None, 70.0, (3000, 6000), decision="ours"))]
    own, kids = VA.derive_plays(ours, list(VA.OUR_DECK), TABLE, event_cards=EVENTS,
                                labels={"70": "HogRider", "71": "HogRider"}, owner_top=False)
    assert kids == [] and [(p["units"], p["deploy_unseen"]) for p in own] == [(["70"], False), (["71"], True)]
    # the two deploy zones mirror each other: ours ends at y 17 where the opponent's defender half begins
    for y, past in ((14900, True), (15000, True), (15100, False), (17000, False)):
        own, _ = VA.derive_plays([gold_part("70", None, 0.0, (3000, y), decision="ours")], list(VA.OUR_DECK), TABLE,
                                 event_cards=EVENTS, labels={"70": "HogRider"}, owner_top=False)
        opp, _ = VA.derive_plays([gold_part("70", "HogRider", 0.0, (3000, 32000 - y))], deck, TABLE, event_cards=EVENTS)
        assert own[0]["deploy_unseen"] is past and opp[0]["deploy_unseen"] is past, y


def test_a_new_play_of_the_card_20_s_later_stays_separate(keys):
    deck = r0_cards(keys)

    def guards(k0, t, last_t, last_y):
        return [gold_part(str(k0 + j), "SkeletonWarrior", t + 0.1 * j, (14400 + 500 * j, 8700), last_t=last_t,
                          last=(14200 + 500 * j, last_y)) for j in range(3)]
    # walked to the bridge and last seen 14 s before the next Guards at the same tower (past reach_gap_s)
    plays, kids = VA.derive_plays(guards(10, 20.0, 26.0, 14000) + guards(20, 40.0, 46.0, 14000), deck, TABLE,
                                  event_cards=EVENTS)
    assert kids == [] and [(p["t"], p["units"]) for p in plays] == [(20.0, ["10", "11", "12"]), (40.0, ["20", "21", "22"])]
    assert all(not p["continued"] for p in plays)
    # still held at the tower when the next ones land: a seen unit is no unit found again, and the full play is
    # more than reach_gap_s old, so the new parts are no second tracks on it either
    plays, kids = VA.derive_plays(guards(10, 20.0, 45.0, 9000) + guards(20, 40.0, 46.0, 14000), deck, TABLE,
                                  event_cards=EVENTS)
    assert [(p["t"], p["units"]) for p in plays] == [(20.0, ["10", "11", "12"]), (40.0, ["20", "21", "22"])]
    assert all(not p["continued"] for p in plays)


def derive(parts, keys, **kw):
    return VA.derive_plays(parts, r0_cards(keys), TABLE, event_cards=EVENTS, **kw)


def describe(plays):
    return [(p["t"], p["card"], p["units"], p["continued"], p["deploy_unseen"], p["new_by"]) for p in plays]


def test_a_part_past_the_river_joins_only_a_unit_that_could_have_walked_there(keys):
    # an Executioner lost at 10 s in the right lane; 15 s later (past reach_gap_s, within zone_gap_s) one first
    # seen past the river 8 tiles down that lane: within its walk (1.0 tile/s x 1.25 x 15 s + 2 = 20.75), the
    # unit found again (zone); one 22.6 tiles off in the other lane is not, so its deploy went unseen
    exe = gold_part("10", "AxeMan", 0.0, (14000, 8000), last_t=10.0, last=(14000, 10000))
    near = gold_part("20", "AxeMan", 25.0, (14000, 18000), last_t=27.0)
    far = gold_part("20", "AxeMan", 25.0, (3500, 30000), last_t=27.0)
    plays, kids = derive([exe, near], keys)
    assert kids == [] and describe(plays) == [(0.0, "AxeMan", ["10"], {"20": "zone"}, False, None)]
    assert plays[0]["gaps"] == {"20": 15.0} and VA.flagged_merges(plays[0]) == [("20", "zone", 15.0)]
    assert describe(derive([exe, far], keys)[0]) == [(0.0, "AxeMan", ["10"], {}, False, None),
                                                     (25.0, "AxeMan", ["20"], {}, True, None)]
    # Elite Barbarians still seen at Yodle2's right tower: a pair first seen past the river in the other lane is a
    # play whose deploy went unseen; a part on top of one of them (within reach_margin_tiles) is a second track
    pair = [gold_part("10", "AngryBarbarian", 0.0, (14000, 9000), last_t=35.0, last=(14000, 26000)),
            gold_part("11", "AngryBarbarian", 0.05, (14800, 9000), last_t=35.0, last=(14800, 26000))]
    other = [gold_part("20", "AngryBarbarian", 30.0, (3500, 17200), last_t=34.0, last=(3500, 24000)),
             gold_part("21", "AngryBarbarian", 30.05, (4300, 17200), last_t=34.0, last=(4300, 24000))]
    on_top = gold_part("30", "AngryBarbarian", 31.0, (14000, 23500), last_t=32.0)
    plays, _kids = derive(pair + other + [on_top], keys)
    assert describe(plays) == [(0.0, "AngryBarbarians", ["10", "11"], {"30": "capacity"}, False, None),
                               (30.0, "AngryBarbarians", ["20", "21"], {}, True, None)]


def others(t0, t1, skip, n=4):
    """One play each of n other cards between t0 and t1 (4: what the card cycle needs before a card comes back)."""
    labels = [lab for card, lab in (("Zap", "zap"), ("HogRider", "HogRider"), ("BossBandit", "BossBandit"),
                                    ("AxeMan", "AxeMan"), ("MinionHorde", "MinionHorde_EV1")) if card != skip][:n]
    return [gold_part(str(900 + j), lab, t0 + (t1 - t0) * (j + 1) / (n + 1.0), (2000 + 4000 * j, 6000), last_t=t0 + 1.0)
            for j, lab in enumerate(labels)]


def test_a_new_play_behind_the_last_units_or_after_a_cycle_is_new(keys):
    def ebarbs(k0, t, xy, last_t, last):
        return [gold_part(str(k0), "AngryBarbarian", t, xy, last_t=last_t, last=last),
                gold_part(str(k0 + 1), "AngryBarbarian", t + 0.05, (xy[0] + 800, xy[1]), last_t=last_t,
                          last=(last[0] + 800, last[1]))]
    # the first pair died at Yodle2's tower (y 26) 10 s before the next is put down at y 9: within their walk
    # (20.75 tiles), but 17 tiles behind where they were last seen, in the deploy zone: a new play
    pairs = ebarbs(10, 0.0, (4000, 9000), 15.0, (4000, 26000)) + ebarbs(20, 25.0, (4000, 9000), 30.0, (4000, 20000))
    plays, _kids = derive(pairs, keys)
    assert describe(plays) == [(0.0, "AngryBarbarians", ["10", "11"], {}, False, None),
                               (25.0, "AngryBarbarians", ["20", "21"], {}, False, "behind")]
    # an Executioner replaced 6 s after the old one was lost, 2 tiles behind (not more than back_tiles): alone,
    # nothing tells them apart, so it continues the old one, and the list flags the 6 s gap (flag_gap_s 4)
    exe = [gold_part("10", "AxeMan", 0.0, (9000, 6000), last_t=30.0, last=(9000, 8000)),
           gold_part("20", "AxeMan", 36.0, (9000, 6000), last_t=45.0, last=(9000, 12000))]
    plays, _kids = derive(exe, keys)
    assert describe(plays) == [(0.0, "AxeMan", ["10"], {"20": "reach"}, False, None)]
    assert VA.flagged_merges(plays[0]) == [("20", "reach", 6.0)]
    # with the 4 other plays a game has between two plays of a card, the card is back in the hand: a new play
    plays, _kids = derive(exe + others(0.5, 35.5, "AxeMan"), keys)
    assert [(p["t"], p["card"], p["units"], p["continued"], p["new_by"]) for p in plays if p["card"] == "AxeMan"] == [
        (0.0, "AxeMan", ["10"], {}, None), (36.0, "AxeMan", ["20"], {}, "cycle")]
    plays, _kids = derive(exe + others(0.5, 35.5, "AxeMan"), keys, config=VA.PlayConfig(cycle_plays=0))   # off
    assert [p["units"] for p in plays if p["card"] == "AxeMan"] == [["10"]]
    # a guard found again 1 s after its track ended (rule 1) still continues it once the card has cycled, but the
    # list asks the owner to look
    g = [gold_part("10", "SkeletonWarrior", 20.0, (14400, 8700), last_t=39.0, last=(14400, 9000)),
         gold_part("20", "SkeletonWarrior", 40.0, (14400, 8700), last_t=46.0, last=(14400, 12000))]
    plays, _kids = derive(g + others(20.5, 39.5, "SkeletonWarriors"), keys)
    guards = [p for p in plays if p["card"] == "SkeletonWarriors"]
    assert [(p["units"], p["continued"], p["cycled"]) for p in guards] == [(["10"], {"20": "reacquired"}, {"20": 4})]
    assert VA.flagged_merges(guards[0]) == [("20", "reacquired after its card had cycled", 1.0)]
    # the cycle over a final list: other plays since the previous play of the card, per battle
    final = [{"card": "A", "hk": 1}, {"card": "B", "hk": 1}, {"card": "A", "hk": 2}, {"card": "C", "hk": 1},
             {"card": "A", "hk": 1}, {"card": "B", "hk": 1}]
    assert VA.cycle_since(final) == [None, None, None, None, (2, 0), (2, 1)]


def test_spell_cards_seen_as_units_and_a_full_deploy_window(keys):
    # Heal's spirits are a spell's units (no card summons HealSpirit): no reach, capacity or zone rule, so two Heals
    # 8 s apart are two plays though the second is ahead of the first's last position, within its walk (the patch
    # merged them by reach), and one first seen past the river, 21 s on, is no deploy unseen
    deck = [c.card for c in VA.parse_deck("heal-spirit,hog-rider,musketeer,cannon,ice-golem,skeletons,fireball,the-log",
                                          VA.CardResolver(TABLE, keys, EVENTS))]
    known = VA.known_labels(TABLE, None)
    heal = [gold_part("10", "HealSpirit", 10.0, (9000, 10000), last_t=11.0, last=(9000, 13000)),
            gold_part("20", "HealSpirit", 18.0, (9000, 14000), last_t=19.0, last=(12000, 13000)),
            gold_part("30", "HealSpirit", 40.0, (12000, 20000), last_t=41.0)]
    plays, _kids = VA.derive_plays(heal, deck, TABLE, known, event_cards=EVENTS)
    assert [(p["t"], p["card"], p["units"], p["continued"], p["deploy_unseen"]) for p in plays] == [
        (10.0, "Heal", ["10"], {}, False), (18.0, "Heal", ["20"], {}, False), (40.0, "Heal", ["30"], {}, False)]
    # the Spirit Empress' forms are troops her variant cards summon (MergeMaiden is kind spell only for its variant
    # cost): one walking Empress lost 2 s and 2.5 s is one play, in either form; one first seen past the river with
    # no earlier unit is a play whose deploy went unseen
    sdeck = [c.card for c in VA.parse_deck("spirit-empress,hog-rider,musketeer,cannon,ice-golem,skeletons,fireball,"
                                           "the-log", VA.CardResolver(TABLE, keys, EVENTS))]
    for lab, cost in (("MergeMaiden_Normal", 3), ("MergeMaiden_Mounted", 6)):
        se = [gold_part("10", lab, 10.0, (9000, 10000), last_t=12.0, last=(9000, 13000)),
              gold_part("11", lab, 14.0, (9000, 15500), last_t=16.0, last=(9000, 18000)),
              gold_part("12", lab, 18.5, (9000, 21000), last_t=20.0),
              gold_part("20", lab, 60.0, (3500, 20000), last_t=62.0)]
        plays, _kids = VA.derive_plays(se, sdeck, TABLE, known, event_cards=EVENTS)
        assert [(p["t"], p["card"], p["kind"], p["cost"], p["units"], p["continued"], p["deploy_unseen"])
                for p in plays] == [(10.0, "MergeMaiden", "unit", cost, ["10"], {"11": "reach", "12": "reach"}, False),
                                    (60.0, "MergeMaiden", "unit", cost, ["20"], {}, True)], lab
    # a 4th Guards part 0.4 s into a full deploy, 2.5 tiles off (within radius_tiles of its first unit, past
    # reach_margin_tiles of any guard): a second track on the nearest guard, not a play
    g = [gold_part(str(10 + j), "SkeletonWarrior", 20.0 + 0.1 * j, (9000 + 500 * j, 12000), last_t=25.0,
                   last=(9000 + 500 * j, 14000)) for j in range(3)]
    plays, _kids = derive(g + [gold_part("13", "SkeletonWarrior", 20.4, (6500, 12000), last_t=21.0)], keys)
    assert describe(plays) == [(20.0, "SkeletonWarriors", ["10", "11", "12"], {"13": "capacity"}, False, None)]


def test_a_close_fit_holds_after_the_cycle_and_late_merges_are_flagged(keys):
    cfg = VA.PlayConfig()
    # an Executioner tanking 20 s while 4 other plays came: its track lost 3 s and found 0.4 tiles on is the same
    # unit (a close fit: seen up to flag_gap_s before, within reach_margin_tiles), flagged since the card had cycled
    exe = [gold_part("10", "AxeMan", 0.0, (9000, 5000), last_t=20.0, last=(9000, 9000)),
           gold_part("11", "AxeMan", 23.0, (9100, 9400), last_t=30.0, last=(9100, 10000))]
    plays, kids = derive(exe + others(1.0, 19.0, "AxeMan"), keys)
    axe = [p for p in plays if p["card"] == "AxeMan"]
    assert [(p["units"], p["continued"], p["cycled"]) for p in axe] == [(["10"], {"11": "reach"}, {"11": 4})]
    assert VA.flagged_merges(axe[0]) == [("11", "reach after its card had cycled", 3.0)]
    final, _k, _s = VA.assemble_plays(plays, kids, [], [], cfg)
    checks = VA._play_checks(final, {1: 0.0}, cfg)
    assert checks == [
        "p1 AxeMan: check the parts taken for its units: t11 (reach after its card had cycled, 3.0 s unseen); if one "
        "is a new AxeMan play, --add-play it (a card is back in the hand after 4 other plays: a part merged after that "
        "may be a new play, and so may one merged a play short of it when the labels missed a play between, such as a "
        "Zap)"]
    # not close (6 s unseen; past the river, 6 tiles on): a new play, and past the river one whose deploy went
    # unseen, though a unit could have walked there: the warning says why
    hog = [gold_part("10", "HogRider", 0.0, (14500, 9000), last_t=10.0, last=(14500, 14000)),
           gold_part("11", "HogRider", 13.0, (14500, 20000), last_t=15.0)]
    plays, kids = derive(hog + others(0.5, 9.5, "HogRider"), keys)
    assert [(p["units"], p["deploy_unseen"], p["new_by"]) for p in plays if p["card"] == "HogRider"] == [
        (["10"], False, None), (["11"], True, "cycle")]
    final, _k, _s = VA.assemble_plays(plays, kids, [], [], cfg)
    assert [w for w in VA._play_checks(final, {1: 0.0}, cfg) if w.startswith("p6 ")] == [
        "p6 HogRider is a play whose deploy went unseen: first seen 0:13.0 at (14.5, 20.0), in Yodle2's half, where an "
        "earlier HogRider unit could have walked, but 4 or more other plays came since its card's last play, so it is "
        "taken for a new play; it is dated at that first sighting, as M0 dates one. --add-play \"m:ss=HogRider\" with "
        "the deploy time replaces it; --bad-plays p6 if it is no play"]
    # a new Executioner 3 s after the old one was last seen, 2 tiles on, with 3 labelled plays between: one missed
    # play (a Zap) would make it a new play, so the merge is flagged; with 2 between it is not
    s14 = [gold_part("10", "AxeMan", 0.0, (9000, 4000), last_t=25.0, last=(9000, 5000)),
           gold_part("20", "AxeMan", 28.0, (9000, 7000), last_t=40.0, last=(9000, 12000))]
    plays, _kids = derive(s14 + others(0.5, 27.5, "AxeMan", n=3), keys)
    axe = [p for p in plays if p["card"] == "AxeMan"]
    assert [(p["continued"], p["cycled"]) for p in axe] == [({"20": "reach"}, {"20": 3})]
    assert VA.flagged_merges(axe[0]) == [("20", "reach after 3 other plays", 3.0)]
    plays, _kids = derive(s14 + others(0.5, 27.5, "AxeMan", n=2), keys)
    axe = [p for p in plays if p["card"] == "AxeMan"]
    assert [(p["continued"], p["cycled"]) for p in axe] == [({"20": "reach"}, {})] and VA.flagged_merges(axe[0]) == []
    # Guards replaced at the left bridge 3 s after the old ones were last seen, 3 labelled plays between: flagged
    g = [gold_part(str(10 + j), "SkeletonWarrior", 20.0 + 0.1 * j, (3500 + 500 * j, 12000), last_t=40.0,
                   last=(3500 + 500 * j, 14500)) for j in range(3)]
    g2 = [gold_part(str(20 + j), "SkeletonWarrior", 43.0 + 0.1 * j, (3500 + 500 * j, 13000), last_t=50.0) for j in range(3)]
    plays, _kids = derive(g + g2 + others(20.5, 42.5, "SkeletonWarriors", n=3), keys)
    guards = [p for p in plays if p["card"] == "SkeletonWarriors"]
    assert [p["cycled"] for p in guards] == [{"20": 3, "21": 3, "22": 3}]
    # 16 s after a Guards deploy and 4 other plays, a part past the river 0.2 tiles from a guard still seen there
    # (the reach rule, refused by the cycle, found only guards lost 2 s and 11 s before, 3 and 7 tiles off): the
    # zone rule's close fit, a second track on that guard, flagged
    g = [gold_part("10", "SkeletonWarrior", 0.0, (9000, 12000), last_t=20.0, last=(9000, 22000)),
         gold_part("11", "SkeletonWarrior", 0.1, (9500, 12000), last_t=14.0, last=(9500, 17000)),
         gold_part("12", "SkeletonWarrior", 0.2, (8600, 12000), last_t=5.0, last=(8600, 13000)),
         gold_part("13", "SkeletonWarrior", 16.0, (9200, 20000), last_t=17.0)]
    plays, _kids = derive(g + others(1.0, 13.0, "SkeletonWarriors"), keys)
    guards = [p for p in plays if p["card"] == "SkeletonWarriors"]
    assert [(p["units"], p["continued"], p["cycled"]) for p in guards] == [
        (["10", "11", "12"], {"13": "capacity"}, {"13": 4})]
    assert VA.flagged_merges(guards[0]) == [("13", "capacity after its card had cycled", 0.0)]


def test_a_champion_cycles_from_her_last_sighting():
    deck = ["BossBandit", "AngryBarbarians", "AxeMan", "Freeze", "SkeletonWarriors", "HogRider", "MinionHorde", "Zap"]
    known = VA.known_labels(TABLE, None)
    assert VA.CHAMPION_CARDS == __import__("rs_presses").CHAMPIONS_BEFORE_THE_COLUMN       # the restated names
    assert all(c in TABLE.cards for c in VA.CHAMPION_CARDS)                                # as classes.json has them
    # the Boss Bandit alive 18 s while 4 other plays came, lost 5 s (her dash) and found 1.5 tiles on: no new Boss
    # Bandit can be put down while she lives, and none came since she was last seen, so it is her (the 5 s gap is
    # flagged, as any reach over flag_gap_s). Counted from her deploy instead (no champions), the card had cycled
    bb = [gold_part("10", "BossBandit", 0.0, (14000, 8000), last_t=18.0, last=(14000, 12000)),
          gold_part("11", "BossBandit", 23.0, (14000, 13500), last_t=26.0)]
    between = others(1.0, 17.0, "BossBandit")
    plays, _kids = VA.derive_plays(bb + between, deck, TABLE, known, event_cards=EVENTS)
    boss = [p for p in plays if p["card"] == "BossBandit"]
    assert [(p["units"], p["continued"], p["cycled"]) for p in boss] == [(["10"], {"11": "reach"}, {})]
    assert VA.flagged_merges(boss[0]) == [("11", "reach", 5.0)]
    plays, _kids = VA.derive_plays(bb + between, deck, TABLE, known, event_cards=EVENTS, champions=())
    assert [(p["units"], p["new_by"]) for p in plays if p["card"] == "BossBandit"] == [(["10"], None), (["11"], "cycle")]
    # dead at 18 s, then 4 other plays: her card is back, so a Boss Bandit 11 s later a tile on is a new play
    again = [bb[0], gold_part("11", "BossBandit", 29.0, (14000, 13000), last_t=33.0)]
    plays, _kids = VA.derive_plays(again + others(18.5, 28.5, "BossBandit"), deck, TABLE, known, event_cards=EVENTS)
    assert [(p["units"], p["new_by"]) for p in plays if p["card"] == "BossBandit"] == [(["10"], None), (["11"], "cycle")]


def test_parts_the_rules_must_keep_apart(keys):
    # a deploy-unseen Hog Rider whose track came back to y 14 (its owner's deploy zone): a Hog there 1 s later,
    # 0.5 tiles off, is a new play, neither found again (rule 1) nor reached (rule 5); nor 2 s later
    unseen = gold_part("10", "HogRider", 10.0, (14500, 17500), last_t=11.0, last=(14500, 14000))
    for t in (12.0, 13.0):
        plays, _kids = derive([unseen, gold_part("20", "HogRider", t, (14500, 13500), last_t=t + 3.0)], keys)
        assert [(p["units"], p["deploy_unseen"]) for p in plays] == [(["10"], True), (["20"], False)], t
    # the same live uid handed on to a new Hog at the bridge 1 s after a deploy-unseen one was last seen
    uid = [gold_part("10", "HogRider", 10.0, (14500, 17500), last_t=11.0, last=(14500, 17200), uid="1:5"),
           gold_part("20", "HogRider", 12.0, (14500, 14000), last_t=15.0, last=(14500, 22000), uid="1:5")]
    plays, _kids = derive(uid, keys)
    assert [(p["units"], p["continued"]) for p in plays] == [(["10"], {}), (["20"], {})]
    # an Elite Barbarian seen alone at the bridge: a second pair put down 7 tiles behind 5 s later is a new play,
    # not the unit of the first one not seen since its deploy
    eb = [gold_part("10", "AngryBarbarian", 0.0, (9000, 12000), last_t=10.0, last=(9000, 13000)),
          gold_part("20", "AngryBarbarian", 5.0, (9000, 5000), last_t=8.0)]
    plays, _kids = derive(eb, keys)
    assert [(p["units"], p["joined"], p["new_by"]) for p in plays] == [(["10"], {}, None), (["20"], {}, "behind")]
    # a second track on an Executioner for 1 s (capacity) leaves the unit's latest part the long one, so the unit
    # found again 1 s after that one ended continues it
    exe = [gold_part("10", "AxeMan", 0.0, (9000, 6000), last_t=20.0, last=(9000, 9000)),
           gold_part("11", "AxeMan", 5.0, (9000, 6800), last_t=6.0, last=(9000, 7000)),
           gold_part("12", "AxeMan", 21.0, (9000, 9200), last_t=25.0)]
    plays, _kids = derive(exe, keys)
    assert [(p["units"], p["continued"]) for p in plays] == [(["10"], {"11": "capacity", "12": "reacquired"})]
    # a deploy-unseen play's "earlier" names the earlier parts of its track under another card only (the Hog
    # part 80.0 is a play of its own)
    track = [gold_part("80.0", "HogRider", 0.0, (14500, 10000), last_t=2.0, last=(14500, 12000), h=80),
             gold_part("80.1", "AxeMan", 2.2, (14500, 12500), last_t=3.0, last=(14500, 13000), h=80, part=1),
             gold_part("80.2", "HogRider", 3.2, (14500, 20000), last_t=5.0, h=80, part=2)]
    plays, _kids = derive(track, keys)
    assert [(p["card"], p["units"], p["deploy_unseen"], p.get("earlier")) for p in plays] == [
        ("HogRider", ["80.0"], False, None), ("AxeMan", ["80.1"], False, None),
        ("HogRider", ["80.2"], True, [{"key": "80.1", "label": "AxeMan", "t": 2.2}])]
    # an added play that replaces a deploy-unseen one takes its children too
    derived = [{"t": 20.0, "card": "Witch", "hk": 1, "deploy_unseen": True, "units": ["90"], "x_mt": 9000.0,
                "y_mt": 20000.0}]
    final, kids, _stale = VA.assemble_plays(derived, [{"key": "95", "derived": 0}], [], [{"t": 18.0, "card": "Witch",
                                                                                           "hk": 1}])
    assert [(p["id"], p["replaces"]["t"], p["children"]) for p in final] == [("p1", 20.0, ["95"])]
    assert [c["play"] for c in kids] == ["p1"]


def test_a_deploy_unseen_play_is_listed_checked_and_replaced_by_an_added_play(tmp_path, keys, capsys):
    # a guard lost 5.5 s (reach, past flag_gap_s), a Hog Rider first seen past the river (its deploy unlabelled)
    # and found again 13 s after it was lost, deep in Yodle2's half (zone)
    extra = [gold_part("83", "SkeletonWarrior", 27.5, (9200, 14500)),
             gold_part("90", "HogRider", 75.0, (14300, 20300)), gold_part("91", "HogRider", 90.0, (14000, 27000))]
    paths = hand_gold(tmp_path, keys, parts=R0_PARTS + extra)
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE, key_map=keys, event_cards=EVENTS)
    res = VA.plays_session(R0_SID, **kw)
    assert [(p["id"], p["card"], p["t"], p.get("deploy_unseen")) for p in res.plays][::6] == [
        ("p1", "SkeletonWarriors", 20.0, False), ("p7", "HogRider", 75.0, True)]
    assert res.plays[6]["continued"] == {"91": "zone"}
    assert [(c["key"], c["play"]) for c in res.children] == [("60", "p6")]
    checks = [w for w in res.warnings if w.startswith(("p1 ", "p7 "))]
    assert checks == [
        "p1 SkeletonWarriors: check the parts taken for its units: t83 (reach, 5.5 s unseen); if one is a "
        "new SkeletonWarriors play, --add-play it",
        "p7 HogRider is a play whose deploy went unseen: first seen 1:05.0 at (14.3, 20.3), in Yodle2's half, with no "
        "earlier HogRider unit that could have walked there; it is dated at that first sighting, as M0 dates one. "
        '--add-play "m:ss=HogRider" with the deploy time replaces it; --bad-plays p7 if it is no play',
        "p7 HogRider: check the parts taken for its units: t91 (zone, 13.0 s unseen); if one is a new "
        "HogRider play, --add-play it"]
    assert json.loads(paths.plays_file.read_text(encoding="utf-8"))["plays"][6]["deploy_unseen"] is True
    # the list shows each play's merges by rule, and marks the ones to check
    assert cli(tmp_path, "plays", R0_SID, "--dry-run") == VA.EXIT_OK
    out = capsys.readouterr().out
    assert "1:05.0  HogRider" in out and "(deploy unseen)" in out and "orphan" not in out
    assert "+ reach t83 5.5 s   <- check t83" in out and "+ zone t91 13.0 s   <- check t91" in out
    assert "+ uid t31; split t30.1\n" in out                          # nothing to check there
    # the deploy time given: the added play replaces the deploy-unseen one (one Hog Rider play) and its warning goes
    added = VA.plays_session(R0_SID, list_no=1, add_plays=["1:03.0=HogRider"], **kw)
    hogs = [(p["id"], p["source"], p["t"], p.get("replaces")) for p in added.plays if p["card"] == "HogRider"]
    assert hogs == [("p7", "added", 73.0, {"t": 75.0, "units": ["90"], "x_mt": 14300.0, "y_mt": 20300.0})]
    assert not any("deploy went unseen" in w or "already on the list" in w for w in added.warnings)
    # confirmed: the checks are the owner's to have done, no longer warned of
    ok = VA.plays_session(R0_SID, list_no=2, confirm=True, **kw)
    assert ok.doc["confirmed"] and not any("check the parts" in w for w in ok.warnings)
    # the rule flags reach PlayConfig and plays.json's config: reach and zone off, the guard part is its own play,
    # and so is each Hog Rider part, none a deploy-unseen play the added one would replace
    flags = ["--reach-gap-s", "0", "--zone-y-tiles", "32", "--cycle-plays", "0", "--flag-gap-s", "2.5", "--back-tiles", "1",
             "--zone-gap-s", "9", "--reach-margin-tiles", "1.5"]
    assert cli(tmp_path, "plays", R0_SID, *flags) == VA.EXIT_OK
    out = capsys.readouterr().out
    doc = json.loads(paths.plays_file.read_text(encoding="utf-8"))
    assert {k: doc["config"][k] for k in ("reach_gap_s", "zone_y_tiles", "cycle_plays", "flag_gap_s", "back_tiles",
                                          "zone_gap_s", "reach_margin_tiles")} == {
        "reach_gap_s": 0.0, "zone_y_tiles": 32.0, "cycle_plays": 0, "flag_gap_s": 2.5, "back_tiles": 1.0,
        "zone_gap_s": 9.0, "reach_margin_tiles": 1.5}
    assert [p["card"] for p in doc["plays"]].count("SkeletonWarriors") == 2 and "(deploy unseen)" not in out
    assert [(p["source"], p["t"]) for p in doc["plays"] if p["card"] == "HogRider"] == [
        ("added", 73.0), ("derived", 75.0), ("derived", 90.0)]
    assert not any("comes only" in w for w in doc["warnings"])          # --cycle-plays 0: no cycle check either
    defaults = VA.build_parser().parse_args(["plays", R0_SID])
    assert all(getattr(defaults, k) == getattr(VA.PlayConfig(), k) for k in (
        "window_s", "radius_tiles", "reach_gap_s", "reach_margin_tiles", "zone_y_tiles", "zone_gap_s", "back_tiles",
        "cycle_plays", "flag_gap_s"))


def test_m0_dates_a_live_play_by_its_own_tick(tmp_path):
    scan = tmp_path / "scan"
    scan.mkdir()
    recs = [{"n": 0, "t": 10.0, "tick": 0, "plays": []},
            {"n": 1, "t": 10.05, "tick": 1, "plays": []},
            {"n": 13, "t": 11.3, "tick": 26, "plays": [{"card": 40, "card_name": "Zap", "tick": 10, "x_mt": 9000,
                                                        "y_mt": 12000, "kind": "spell"}]},
            {"n": 14, "t": 11.35, "plays": [{"card": 7, "card_name": "Minions", "tick": 20}]}]       # no record tick
    # unit_harvest writes compact JSON, as here
    (scan / "scan_frames.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs),
                                           encoding="utf-8")
    live, t0 = VA.scan_plays(scan)
    assert t0 == 10.0 and [(p["card_name"], p["t"], p["tick"]) for p in live] == [("Zap", 10.5, 10), ("Minions", 11.35, 20)]
    assert VA.scan_plays(tmp_path / "none") is None


def test_pending_tracks_and_a_changed_derivation_refuse_a_confirmation(tmp_path, keys, capsys):
    paths = hand_gold(tmp_path, keys)
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE, key_map=keys, event_cards=EVENTS)
    # a dry run stores no list: the next step is a plain run, not an edit of a list that does not exist
    assert cli(tmp_path, "plays", R0_SID, "--dry-run") == VA.EXIT_OK
    out = capsys.readouterr().out
    assert "play list 1 (dry run: not stored)" in out and "without --dry-run: stores list 1" in out
    assert not paths.plays_file.exists()
    VA.plays_session(R0_SID, **kw)
    assert cli(tmp_path, "plays", R0_SID, "--dry-run") == VA.EXIT_OK
    out = capsys.readouterr().out
    assert "play list 2 (dry run: not stored)" in out and "--list 1 --bad-plays" in out
    # the gold emit left a track unlabelled (emit --allow-pending): its plays are missing
    doc = json.loads((paths.gold_dir / VA.EMIT_JSON).read_text(encoding="utf-8"))
    VA.write_json(paths.gold_dir / VA.EMIT_JSON, dict(doc, pending=["90"]))
    res = VA.plays_session(R0_SID, **kw)
    assert res.list_no == 2 and res.doc["pending"] == ["90"] and any("were not labelled" in w for w in res.warnings)
    with pytest.raises(VA.LabelError, match="left 1 track") as err:
        VA.plays_session(R0_SID, list_no=2, confirm=True, **kw)
    assert json.loads(paths.plays_file.read_text(encoding="utf-8"))["list"] == 2       # nothing stored
    assert cli(tmp_path, "plays", R0_SID, "--list", "2", "--confirm") == VA.EXIT_INPUT
    assert "--allow-pending" in capsys.readouterr().out
    ok = VA.plays_session(R0_SID, list_no=2, confirm=True, allow_pending=True, **kw)
    assert ok.doc["confirmed"] and ok.list_no == 3
    # other settings derive other plays (Guards, Elite Barbarians and the Horde split at a 0.01 s window without the
    # reach rule): no edit and no confirmation of list 3 then; a plain run shows the new list first
    narrow = VA.PlayConfig(window_s=0.01, reach_gap_s=0.0)
    with pytest.raises(VA.LabelError, match="differ from list 3") as err:
        VA.plays_session(R0_SID, list_no=3, confirm=True, allow_pending=True, config=narrow, **kw)
    assert err.value.code == VA.EXIT_CHECK
    shown = VA.plays_session(R0_SID, config=narrow, **kw)
    assert shown.list_no == 4 and not shown.doc["confirmed"] and len(shown.plays) == len(ok.plays) + 8   # 2 + 1 + 5
    VA.plays_session(R0_SID, list_no=4, confirm=True, allow_pending=True, config=narrow, **kw)
    assert VA.plays_fresh(paths) == []
