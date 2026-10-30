"""N08: tools/vision_autolabel.py ingest and status on tiny fake recordings.

A fake recording is a scripts/vision/record.py session written by hand in tmp_path: a header,
14 frame lines (an image listed every third frame, one of them already pruned), a last line
cut short by a crash, and a few bytes per "jpg" (ingest never decodes an image). One test
writes the session with the real VisionRecorder (npy images) to pin the layout. The class
table, event cards and RoyaleAPI key map are the shipped ones (data/vision, data/cards).

Checked: images are hard-linked (the same file, so a pruned recording keeps them in the store);
session.json holds the checked deck, role, client and counts, and no machine path; a deck with
SuperKnight (or any other problem) writes nothing and exits 2; roles from the session id;
changed roles or decks need --force; ambiguous recordings are refused; status and its coverage.
No image library, GPU, engine or emulator.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "tools"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vision_autolabel as VA  # noqa: E402
from scripts.vision import labels as L  # noqa: E402

TABLE = L.load_class_table()
EVENTS = VA.load_event_cards()
CLIENT = "16.635.1"
DECK_A = ("goblinstein,rune-giant,goblin-demolisher,suspicious-bush,vines,goblin-curse,"
          "elite-barbarians-ev1,baby-dragon-ev1")
DECK_B = "boss-bandit,berserker-hero,inferno-dragon-ev1,ronin,goblin-machine,MinionGiant,void,three-musketeers"
HOG = "hog-rider,musketeer,cannon,ice-golem,ice-spirit,skeletons,fireball,the-log"
HOG_CARDS = ["HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons", "Fireball", "Log"]


@pytest.fixture(scope="module")
def keys():
    key_map, note = VA.load_key_map()
    assert note is None, note
    return key_map


def fake_recording(root: Path, name: str = "capture_20261101_201500", images=(0, 3, 6, 9),
                   pruned=(12,), age_s: float = 3600.0) -> Path:
    """<root>/<name>/frames.jsonl + images/: 14 frames, `images` on disk, `pruned` listed but gone."""
    session = root / name
    (session / "images").mkdir(parents=True)
    lines = [{"kind": "session", "schema": "hastycr.vision.record/1", "created": "2026-11-01T20:15:00",
              "image_every": 3, "image_format": "jpg", "image_width": None, "frame_schema": 1}]
    for i in range(14):
        listed = i in images or i in pruned
        lines.append({"kind": "frame", "i": i, "t_wall": 1000.0 + i / 57.0,
                      "image": f"images/{i:08d}.jpg" if listed else None,
                      "frame": {"v": 1, "t_capture": 10.0 + i / 57.0}})
    for i in images:
        (session / "images" / f"{i:08d}.jpg").write_bytes(b"\xff\xd8 fake jpeg %d" % i)
    text = "\n".join(json.dumps(x) for x in lines) + "\n" + '{"kind": "frame", "i": 14, "t_w'
    frames = session / "frames.jsonl"
    frames.write_text(text, encoding="utf-8")
    old = time.time() - age_s
    os.utime(frames, (old, old))
    return session


def run_ingest(tmp_path: Path, recording: Path, deck: str = DECK_A, key_map=None, **kw):
    kw.setdefault("client", CLIENT)
    if key_map is None:
        key_map, _ = VA.load_key_map()
    return VA.ingest(recording, deck, labels_root=tmp_path / "labels", store=tmp_path / "store",
                     table=TABLE, key_map=key_map, event_cards=EVENTS, **kw)


def cli(tmp_path: Path, *args: str) -> int:
    return VA.main([*args, "--labels-root", str(tmp_path / "labels"), "--store", str(tmp_path / "store")])


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def test_ingest_hard_links_the_images_and_writes_session_json(tmp_path, keys):
    root = tmp_path / "harvest" / "20261101_A_1"
    session = fake_recording(root)
    res = run_ingest(tmp_path, root, key_map=keys)
    assert res.sid == "20261101_A_1" and res.created
    paths = res.paths
    assert paths.store_dir == tmp_path / "store" / "20261101_A_1"
    assert paths.recording == paths.store_dir / "recording"
    for i in (0, 3, 6, 9):
        src, dst = session / "images" / f"{i:08d}.jpg", paths.images / f"{i:08d}.jpg"
        assert os.path.samefile(src, dst)                    # a hard link, not a copy
    assert (res.counts["linked"], res.counts["copied"], res.counts["present"]) == (4, 0, 0)
    assert paths.frames.read_bytes() == (session / "frames.jsonl").read_bytes()

    doc = json.loads(paths.session_file.read_text(encoding="utf-8"))
    assert doc == VA.load_session("20261101_A_1", tmp_path / "labels") == res.session
    assert paths.session_file == tmp_path / "labels" / "20261101_A_1" / "session.json"
    assert (doc["schema"], doc["sid"], doc["role"], doc["capture"], doc["client"]) == (
        VA.SESSION_SCHEMA, "20261101_A_1", "train", "harvest", CLIENT)
    assert VA.deck_cards(doc) == ["Goblinstein", "GiantBuffer", "GoblinDemolisher", "SuspiciousBush",
                                  "Vines", "GoblinCurse", "AngryBarbarians", "BabyDragon"]
    assert [c["form_name"] for c in doc["deck"]] == ["base"] * 6 + ["evo", "evo"]
    assert doc["deck"][0] == {"given": "goblinstein", "card": "Goblinstein", "form": 0, "form_name": "base",
                              "kind": "troop", "elixir": 5}
    assert doc["slots"] == {"evo": 2, "hero": 0, "champion": 1}
    rec = doc["recording"]
    assert (rec["name"], rec["frames"], rec["images_listed"], rec["images_missing"], rec["bad_lines"]) == (
        "capture_20261101_201500", 14, 5, 1, 1)
    assert (rec["image_every"], rec["created"], rec["images_unlisted"]) == (3, "2026-11-01T20:15:00", 0)
    assert rec["seconds"] == pytest.approx(13 / 57.0, abs=1e-3)
    assert len(rec["frames_sha1"]) == 40 and rec["frames_bytes"] == (session / "frames.jsonl").stat().st_size
    assert doc["store"]["images"] == 4
    for shown in (rec["source"], doc["store"]["path"]):     # never a machine path in committed JSON
        assert shown is None or not Path(shown).is_absolute()
    assert doc["classes_digest"] == TABLE.digest and doc["registry_digest"] == VA.Registry.load().digest
    warnings = " | ".join(res.warnings)
    assert "gone before ingest" in warnings and "unreadable line" in warnings
    assert "AngryBarbarian_EV1" in warnings and "BabyDragon_EV1" in warnings   # no evo labels yet
    assert "still running" not in warnings and "copied" not in warnings
    assert doc["warnings"] == res.warnings

    # The recorder prunes its copy: the store keeps every image.
    for f in (session / "images").iterdir():
        f.unlink()
    assert all((paths.images / f"{i:08d}.jpg").read_bytes() == b"\xff\xd8 fake jpeg %d" % i
               for i in (0, 3, 6, 9))


def test_ingest_again_links_only_what_is_new(tmp_path, keys):
    root = tmp_path / "harvest" / "20261101_A_3"
    session = fake_recording(root, age_s=0.0)
    first = run_ingest(tmp_path, root, key_map=keys)
    assert any("still running" in w for w in first.warnings)
    (session / "images" / "00000012.jpg").write_bytes(b"\xff\xd8 late")       # the pruned one, back
    old = time.time() - 3600
    os.utime(session / "frames.jsonl", (old, old))
    again = run_ingest(tmp_path, root, deck=",".join(reversed(DECK_A.split(","))), key_map=keys)
    assert not again.created
    assert (again.counts["linked"], again.counts["present"]) == (1, 4)
    assert again.session["recording"]["images_missing"] == 0 and again.session["store"]["images"] == 5
    assert again.session["ingested"] == first.session["ingested"] and again.session["history"] == []
    assert not any("still running" in w for w in again.warnings)


def test_a_deck_with_superknight_is_refused_and_nothing_is_written(tmp_path, keys, capsys):
    root = tmp_path / "harvest" / "20261101_A_1"
    fake_recording(root)
    deck = DECK_A.replace("rune-giant", "SuperKnight")
    with pytest.raises(VA.IngestError) as err:
        run_ingest(tmp_path, root, deck=deck, key_map=keys)
    assert err.value.code == VA.EXIT_INPUT
    assert any("SuperKnight" in p and "no friendly deck" in p for p in err.value.problems)
    assert not (tmp_path / "labels").exists() and not (tmp_path / "store").exists()
    assert cli(tmp_path, "ingest", str(root), "--deck", deck, "--client", CLIENT) == VA.EXIT_INPUT
    out = capsys.readouterr().out
    assert "nothing written" in out and "SuperKnight" in out
    assert not (tmp_path / "labels").exists() and not (tmp_path / "store").exists()
    for bad in ("MergeMaiden_Normal", "super-knight", "PrinceBuff"):
        with pytest.raises(VA.IngestError, match="no friendly deck"):
            run_ingest(tmp_path, root, deck=DECK_A.replace("rune-giant", bad), key_map=keys)
    with pytest.raises(VA.IngestError, match="give MergeMaiden instead"):
        run_ingest(tmp_path, root, deck=DECK_A.replace("rune-giant", "MergeMaiden_Mounted"), key_map=keys)
    assert not (tmp_path / "labels").exists() and not (tmp_path / "store").exists()


def test_the_deck_check(keys):
    resolver = VA.CardResolver(TABLE, keys, EVENTS)
    deck_b = VA.parse_deck(DECK_B, resolver)
    assert [(c.card, c.form) for c in deck_b] == [
        ("BossBandit", 0), ("Berserker", 2), ("InfernoDragon", 1), ("Ronin", 0), ("GoblinMachine", 0),
        ("MinionGiant", 0), ("DarkMagic", 0), ("ThreeMusketeers", 0)]
    assert VA.slot_counts(deck_b) == {"evo": 1, "hero": 1, "champion": 1} and VA.slot_warnings(deck_b) == []
    # RoyaleAPI keys and catalogue names give the same deck; spacing and case are forgiven
    by_key = VA.parse_deck(HOG, resolver)
    by_name = VA.parse_deck(" HogRider, Musketeer,Cannon , IceGolemite,IceSpirits,Skeletons,Fireball,Log,", resolver)
    assert [c.card for c in by_key] == [c.card for c in by_name] == HOG_CARDS
    assert resolver.resolve("minion-giant")[0].card == "MinionGiant"
    assert resolver.resolve("spirit-empress")[0].card == "MergeMaiden"
    assert resolver.resolve("heal-spirit")[0].card == "Heal"
    assert (resolver.resolve("Cannon-ev1")[0].card, resolver.resolve("Cannon-ev1")[0].form) == ("Cannon", 1)
    assert VA.split_form_suffix("musketeer-hero") == ("musketeer", VA.FORM_HERO)
    # every problem at once
    with pytest.raises(VA.DeckError) as err:
        VA.parse_deck("hog-rider-ev1,ronin-hero,hog-ryder,MergeMaiden_Normal,super-knight,ronin,ronin", resolver)
    text = " | ".join(err.value.problems)
    for needle in ("no evo form of HogRider", "no hero form of Ronin", "unknown card 'hog-ryder'",
                   "hog-rider", "give MergeMaiden instead", "SuperKnight", "appears twice: Ronin",
                   "a deck has 8 cards; 7 given"):
        assert needle in text, needle
    # the slot rules warn, they do not refuse
    three_evos = VA.parse_deck("cannon-ev1,skeletons-ev1,ice-spirit-ev1,hog-rider,musketeer,ice-golem,"
                               "fireball,the-log", resolver)
    assert any("3 evolution slots" in w for w in VA.slot_warnings(three_evos))
    four = VA.parse_deck("cannon-ev1,skeletons-ev1,musketeer-hero,archer-queen,hog-rider,ice-golem,"
                         "fireball,the-log", resolver)
    assert VA.slot_counts(four) == {"evo": 2, "hero": 1, "champion": 1}
    assert any("at most 3" in w for w in VA.slot_warnings(four))


def test_label_gaps_name_only_what_the_registry_lacks(keys):
    resolver = VA.CardResolver(TABLE, keys, EVENTS)
    known = set(VA.crosswalk_base(TABLE).labels) | set(VA.Registry.load().labels)
    assert VA.deck_label_gaps(VA.parse_deck(HOG, resolver), TABLE, known) == []   # MusketeerTurret: spawned
    gaps = VA.deck_label_gaps(VA.parse_deck(DECK_B, resolver), TABLE, known)
    assert len(gaps) == 1                                                  # only the evo slot's form
    assert gaps[0].startswith("inferno-dragon-ev1:") and "InfernoDragon_EV1" in gaps[0]
    assert "as the base form" in gaps[0]
    bare = VA.deck_label_gaps(VA.parse_deck(DECK_B, resolver), TABLE, VA.crosswalk_base(TABLE).labels)
    text = " | ".join(bare)
    for label in ("BossBandit", "Ronin", "MinionGiant", "DarkMagic", "ThreeMusketeer_Rework_Character_1"):
        assert label in text, label
    assert "can only be masked" in text


def test_role_assignment():
    assert VA.auto_role("20261101_A_1")[0] == VA.ROLE_TRAIN
    assert VA.auto_role("20261101_A_2")[0] == VA.ROLE_GOLD
    assert VA.auto_role("20261101_A_3")[0] == VA.ROLE_TRAIN
    assert VA.auto_role("20261101_A_4")[0] == VA.ROLE_DEV
    assert VA.auto_role("20261101_a_4")[0] == VA.ROLE_DEV
    assert VA.auto_role("20261101_B_4")[0] == VA.ROLE_TRAIN
    assert VA.auto_role("20261102_D_2")[0] == VA.ROLE_GOLD
    assert VA.auto_role("20261102_K_3")[0] == VA.ROLE_TRAIN
    with pytest.raises(ValueError, match="--role"):
        VA.auto_role("capture_20261029_201500")
    assert VA.auto_role("capture_20261029_201500", VA.CAPTURE_NATURAL)[0] == VA.ROLE_GOLD
    assert VA.choose_role("auto", "20261101_A_1", VA.CAPTURE_FIELD)[0] == VA.ROLE_GOLD
    assert VA.choose_role("dev", "any_name") == (VA.ROLE_DEV, "given")
    for capture in (VA.CAPTURE_NATURAL, VA.CAPTURE_FIELD):
        with pytest.raises(ValueError, match="gold only"):
            VA.choose_role("train", "20261029_R0_1", capture)
    with pytest.raises(ValueError):
        VA.choose_role("bogus", "20261101_A_1")
    with pytest.raises(ValueError):
        VA.choose_role("auto", "20261101_A_1", "scraped")


def test_a_changed_role_or_deck_needs_force(tmp_path, keys):
    root = tmp_path / "harvest" / "20261101_A_2"
    fake_recording(root)
    first = run_ingest(tmp_path, root, key_map=keys)
    assert first.session["role"] == VA.ROLE_GOLD and "friendly 2" in first.session["role_rule"]
    for change, needle in (({"role": "train"}, "role 'gold'"), ({"deck": DECK_B}, "has deck"),
                           ({"client": "16.700.0"}, "has client"), ({"capture": "field"}, "has capture")):
        with pytest.raises(VA.IngestError, match=needle) as err:
            run_ingest(tmp_path, root, key_map=keys, **change)
        assert err.value.code == VA.EXIT_CHECK
    assert VA.load_session("20261101_A_2", tmp_path / "labels")["role"] == VA.ROLE_GOLD
    forced = run_ingest(tmp_path, root, key_map=keys, role="train", force=True)
    assert forced.session["role"] == VA.ROLE_TRAIN
    assert forced.session["history"][-1]["changed"] == {"role": ["gold", "train"]}
    assert forced.session["ingested"] == first.session["ingested"]
    back = run_ingest(tmp_path, root, key_map=keys, force=True)          # auto: gold again
    assert back.session["role"] == VA.ROLE_GOLD and len(back.session["history"]) == 2
    assert any("training mix" in w for w in back.warnings)


def test_the_recording_must_be_one_session(tmp_path, keys):
    root = tmp_path / "rec"
    fake_recording(root, "capture_1")
    fake_recording(root, "capture_2")
    with pytest.raises(VA.IngestError, match="2 sessions"):
        run_ingest(tmp_path, root, key_map=keys)
    with pytest.raises(VA.IngestError, match="--role"):            # a capture_ name is no date_deck_n
        run_ingest(tmp_path, root / "capture_1", key_map=keys)
    res = run_ingest(tmp_path, root / "capture_1", sid="20261101_C_1", key_map=keys)
    assert (res.sid, res.session["role"], res.session["recording"]["name"]) == ("20261101_C_1", "train", "capture_1")
    assert VA.find_recording(root / "capture_2" / "frames.jsonl") == (root / "capture_2", "capture_2")
    with pytest.raises(VA.IngestError, match="was ingested from recording capture_1"):
        run_ingest(tmp_path, root / "capture_2", sid="20261101_C_1", key_map=keys)
    # another recording under that id with --force: its images differ, so it is still refused
    (root / "capture_2" / "images" / "00000000.jpg").write_bytes(b"another, longer image")
    with pytest.raises(VA.IngestError, match="differ") as err:
        run_ingest(tmp_path, root / "capture_2", sid="20261101_C_1", key_map=keys, force=True)
    assert err.value.code == VA.EXIT_CHECK
    assert VA.load_session("20261101_C_1", tmp_path / "labels")["recording"]["name"] == "capture_1"
    empty = tmp_path / "empty_root"
    empty.mkdir()
    with pytest.raises(VA.IngestError, match="no recorded session"):
        run_ingest(tmp_path, empty, key_map=keys)
    with pytest.raises(VA.IngestError, match="no recording at"):
        run_ingest(tmp_path, tmp_path / "missing", key_map=keys)
    with pytest.raises(VA.IngestError, match="session id"):
        run_ingest(tmp_path, root / "capture_1", sid="../x", key_map=keys)
    with pytest.raises(VA.IngestError, match="client"):
        run_ingest(tmp_path, root / "capture_1", sid="20261101_C_1", key_map=keys, client="")
    with pytest.raises(VA.IngestError, match="client version"):
        run_ingest(tmp_path, root / "capture_1", sid="20261101_C_1", key_map=keys, client="latest")
    with pytest.raises(VA.IngestError, match="overlap"):
        VA.ingest(root / "capture_1", DECK_A, client=CLIENT, sid="20261101_C_9", labels_root=tmp_path / "labels",
                  store=root / "capture_1" / "store", table=TABLE, key_map=keys, event_cards=EVENTS)


def test_ingest_reads_what_the_real_recorder_writes(tmp_path, keys):
    np = pytest.importorskip("numpy")
    from scripts.vision import record as R
    from scripts.vision.types import VisionFrame
    assert (VA.FRAMES_FILE, VA.IMAGES_DIR, VA.IMAGE_NAME_RE.pattern) == (
        R.FRAMES_FILE, R.IMAGES_DIR, R.IMAGE_NAME_RE.pattern)
    root = tmp_path / "harvest" / "20261101_B_1"
    with R.VisionRecorder(root, prefix="capture", image_every=2, image_format="npy",
                          wall_clock=lambda: 1000.0) as rec:
        for i in range(6):
            rec.write(VisionFrame(t_capture=5.0 + 0.05 * i, frame_size=(8, 8)), np.zeros((8, 8, 3), np.uint8))
    res = run_ingest(tmp_path, root, deck=DECK_B, key_map=keys)
    rec_doc = res.session["recording"]
    assert (rec_doc["frames"], rec_doc["images_listed"], rec_doc["images_missing"]) == (6, 3, 0)
    assert rec_doc["image_format"] == "npy" and rec_doc["seconds"] == pytest.approx(0.25)
    assert res.session["store"]["images"] == 3 and res.session["role"] == VA.ROLE_TRAIN
    reader = R.VisionLogReader(res.paths.recording)            # the store is itself a recording
    assert sum(1 for _ in reader.frames(with_images_only=True)) == 3
    assert R.list_sessions(res.paths.store_dir.parent) == []    # a recorder there would prune nothing


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_lists_sessions_and_what_to_record_next(tmp_path, keys, capsys):
    for sid, deck in (("20261101_A_1", DECK_A), ("20261101_A_2", DECK_A), ("20261101_B_1", DECK_B)):
        root = tmp_path / "harvest" / sid
        fake_recording(root)
        run_ingest(tmp_path, root, deck=deck, key_map=keys)
    reg = VA.Registry.load()
    doc = VA.status(tmp_path / "labels", tmp_path / "store", registry=reg)
    assert [r["sid"] for r in doc["sessions"]] == ["20261101_A_1", "20261101_A_2", "20261101_B_1"]
    assert doc["roles"] == {"train": 2, "gold": 1, "dev": 0} and doc["problems"] == []
    assert all(r["stages"] == ["ingest"] and r["images"] == r["images_ingested"] == 4 for r in doc["sessions"])
    cov = {c["label"]: c for c in doc["coverage"]}
    assert set(cov) == set(reg.labels)
    assert (cov["goblinstein"]["train"], cov["goblinstein"]["gold"]) == (1, 1)
    assert (cov["BerserkerHero"]["train"], cov["BerserkerHero"]["gold"]) == (1, 0)
    assert (cov["MergeMaiden_Normal"]["train"], cov["Vines"]["gold"], cov["DarkMagic"]["gold"]) == (0, 1, 0)
    assert all(c["tracks"] is None for c in doc["coverage"])        # nothing labelled yet

    # labelled tracks (N10's tracks.json) are counted over training sessions only
    VA.write_json(tmp_path / "labels" / "20261101_A_1" / VA.TRACKS_FILE,
                  {"tracks": [{"label": "goblinstein"}, {"label": "goblinstein"}, {"label": "BushGoblin"},
                              {"label": "x"}, "junk"]})
    VA.write_json(tmp_path / "labels" / "20261101_A_2" / VA.TRACKS_FILE, {"tracks": [{"label": "goblinstein"}]})
    gold_images = VA.session_paths("20261101_A_2", tmp_path / "labels", tmp_path / "store").images
    for f in list(gold_images.iterdir())[:2]:
        f.unlink()
    doc = VA.status(tmp_path / "labels", tmp_path / "store", registry=reg)
    cov = {c["label"]: c for c in doc["coverage"]}
    assert (cov["goblinstein"]["tracks"], cov["BushGoblin"]["tracks"], cov["Ronin"]["tracks"]) == (2, 1, 0)
    rows = {r["sid"]: r for r in doc["sessions"]}
    assert rows["20261101_A_1"]["stages"] == ["ingest", "label"]
    assert any("keep gold raws" in n for n in rows["20261101_A_2"]["notes"])
    assert VA.status(tmp_path / "labels", tmp_path / "store", sids=["20261101_B_1"])["roles"]["train"] == 1
    assert VA.status(tmp_path / "nothing_here")["sessions"] == []

    assert cli(tmp_path, "status") == 0
    out = capsys.readouterr().out
    for needle in ("3 session(s)", "train 2, gold 1, dev 0", "20261101_B_1", "keep gold raws",
                   "no gold session yet", "MergeMaiden_Normal", "ingest,label"):
        assert needle in out, needle
    assert cli(tmp_path, "status", "--json") == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["roles"] == {"train": 2, "gold": 1, "dev": 0}
    assert Counter(r["role"] for r in parsed["sessions"]) == Counter({"train": 2, "gold": 1})


def test_ingest_command_prints_what_it_did(tmp_path, capsys):
    root = tmp_path / "harvest" / "20261101_B_2"
    fake_recording(root)
    assert cli(tmp_path, "ingest", str(root), "--deck", DECK_B, "--client", CLIENT) == 0
    out = capsys.readouterr().out
    assert "ingested 20261101_B_2: role gold" in out and "Berserker hero" in out and "4 linked" in out
    assert cli(tmp_path, "ingest", str(root), "--deck", DECK_B, "--client", CLIENT, "--role", "train") == 1
    assert "pass --force" in capsys.readouterr().out
