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
  confirmation; review numbers go on after the review folder is deleted.
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


R0_PARTS = [gold_part("10", "SkeletonWarrior", 20.0, (9000, 20000), uid="1:10"),       # Guards: 3 units
            gold_part("11", "SkeletonWarrior", 20.1, (9500, 20300), uid="1:11"),
            gold_part("12", "SkeletonWarrior", 20.3, (8600, 20400), uid="1:12"),
            gold_part("20", "AngryBarbarian", 30.0, (4000, 22000)),                      # Elite Barbarians: 2
            gold_part("21", "AngryBarbarian", 30.05, (4800, 22000)),
            # Boss Bandit: a look change (part 1 of h30), then a new harvest track of the same live uid
            gold_part("30.0", "BossBandit", 40.0, (14000, 24000), last_t=43.0, last=(14000, 20000), uid="1:30", h=30),
            gold_part("30.1", "BossBandit", 43.05, (14000, 19900), last_t=44.0, last=(14000, 18000), h=30, part=1),
            gold_part("31", "BossBandit", 45.0, (14000, 15000), last_t=50.0, uid="1:30"),
            *[gold_part(str(40 + j), "Minion" if j < 3 else "MinionHorde_EV1", 50.0 + 0.05 * j,
                        (9000 + 300 * j, 14000 + 200 * j), uid=f"1:{40 + j}") for j in range(6)],   # the evo Horde
            gold_part("50", "zap", 60.0, (9000, 12000)),
            gold_part("59", "SuspiciousBush", 65.0, (5000, 20000)),                     # off the declared deck
            gold_part("60", "BushGoblin", 70.0, (5100, 20100)),                         # its child
            gold_part("70", None, 40.0, (3000, 6000), decision="ours", n=300)]          # one of ours (o)


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
                                             "y_mt": 23900, "kind": "unit"}]},
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
        40.0, 30.0, 14000.0, 24000.0, 600, 300, 1500)
    assert res.plays[3]["labels"] == {"Minion": 3, "MinionHorde_EV1": 3}
    assert res.plays[5]["off_deck"] and res.plays[5]["children"] == ["60"]
    assert [(c["key"], c["play"]) for c in res.children] == [("60", "p6")]
    assert [(p["card"], p["units"]) for p in res.own] == [("HogRider", ["70"])]
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
    later = [gold_part("80", "SkeletonWarrior", 36.1, (9000, 20000), uid="1:10"),
             gold_part("81", "SkeletonWarrior", 36.2, (9400, 20200), uid="1:81"),
             gold_part("82", "SkeletonWarrior", 36.3, (8700, 20300), uid="1:82")]
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
    late = dict(R0_PARTS[7], first_t=45.6, last_t=50.0)                  # 1.6 s after 30.1 ended: a new play
    plays, _kids = VA.derive_plays(R0_PARTS[5:7] + [late], deck, TABLE, event_cards=EVENTS)
    assert [p["units"] for p in plays] == [["30.0"], ["31"]]


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
    # other settings derive other plays (Guards, Elite Barbarians and the Horde split at a 0.01 s window): no edit and no
    # confirmation of list 3 then; a plain run shows the new list first
    narrow = VA.PlayConfig(window_s=0.01)
    with pytest.raises(VA.LabelError, match="differ from list 3") as err:
        VA.plays_session(R0_SID, list_no=3, confirm=True, allow_pending=True, config=narrow, **kw)
    assert err.value.code == VA.EXIT_CHECK
    shown = VA.plays_session(R0_SID, config=narrow, **kw)
    assert shown.list_no == 4 and not shown.doc["confirmed"] and len(shown.plays) == len(ok.plays) + 8   # 2 + 1 + 5
    VA.plays_session(R0_SID, list_no=4, confirm=True, allow_pending=True, config=narrow, **kw)
    assert VA.plays_fresh(paths) == []
