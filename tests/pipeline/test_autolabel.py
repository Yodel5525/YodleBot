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
N10: the legend, the label map, label refusals (cluster maps need --sheet), sheet and label on a
synthetic scan (paging, labels on other images refused), gold sessions never auto-accepted,
the side a label trains on, labelled units (not parts) counted, machine paths cut from warnings.
No image library, GPU, engine or emulator (the synthetic scan needs numpy).
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
    assert "label those tracks as InfernoDragon_EV1 all the same" in gaps[0] and "base form" not in gaps[0]
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


# ---------------------------------------------------------------------------
# N10: the legend, the label map, sheet and label
# ---------------------------------------------------------------------------

DECK_C = "spirit-empress,balloon-hero,cannon-ev1,royal-ghost-ev1,heal-spirit,zap,freeze,lightning"
LEGEND_A = ["goblinstein", "goblinstein_doctor", "GiantBuffer", "GoblinDemolisher", "GoblinDemolisher_kamikaze_form",
            "SuspiciousBush", "BushGoblin", "Vines", "Goblin", "GoblinCurse", "AngryBarbarian", "AngryBarbarian_EV1",
            "BabyDragon", "BabyDragon_EV1"]


def known():
    return VA.known_labels(TABLE, VA.Registry.load(), base=None)       # the preview: the same on every machine


def test_the_legend_lists_only_the_declared_deck(keys):
    resolver = VA.CardResolver(TABLE, keys, EVENTS)
    have = known()
    assert len(have) == 155 + len(VA.Registry.load())         # tonight's 155 labels and the registry's entries
    legend = VA.build_legend(VA.parse_deck(DECK_A, resolver), TABLE, have)
    assert [e.label for e in legend] == LEGEND_A and [e.n for e in legend] == list(range(14))
    assert [e.label for e in legend if not e.known] == ["AngryBarbarian_EV1", "BabyDragon_EV1"]
    assert (legend[7].type, legend[9].type, legend[11].form, legend[11].given) == ("spell", "spell", 1,
                                                                                  "elite-barbarians-ev1")
    lines = VA.legend_lines(legend)
    assert lines[0] == " 0 goblinstein" and lines[11] == "11 AngryBarbarian_EV1 (evo) *"
    assert " x an effect, not a unit" in lines and any(line.startswith(" * no detector label yet") for line in lines)
    c = [e.label for e in VA.build_legend(VA.parse_deck(DECK_C, resolver), TABLE, have)]
    assert c[:4] == ["MergeMaiden_Mounted", "MergeMaiden_Normal", "Balloon", "BalloonHero"]   # the Empress by her forms
    assert "HealSpirit" in c and "Heal" not in c and "MergeMaiden" not in c                  # spells seen as units
    assert c[-3:] == ["zap", "freeze", "lightning"] and "Cannon_EV1" in c and "Ghost_EV1" in c
    assert VA.LegendEntry.from_dict(legend[11].to_dict()) == legend[11]


def test_the_label_map_is_parsed_strictly():
    entries, problems = VA.parse_map("0=3, #1=x,2-4=?, t12=goblinstein ,t7.1=s,T013=5")
    assert problems == []
    assert entries == [("cluster", 0, "3"), ("cluster", 1, "x"), ("cluster", 2, "?"), ("cluster", 3, "?"),
                       ("cluster", 4, "?"), ("track", "12", "goblinstein"), ("track", "7.1", "s"), ("track", "13", "5")]
    entries, problems = VA.parse_map("0,a=3,0=3,0=4,5-3=x,t1.=2,=4,7=")
    assert entries == [("cluster", 0, "3")] and len(problems) == 7
    assert "cluster 0 is labelled twice" in problems and "'5-3=x': the range runs backwards" in problems
    assert VA.parse_map("") == ([], []) and VA.parse_map(None) == ([], [])


def test_names_resolve_through_classes_json_and_the_legend(keys):
    resolver = VA.CardResolver(TABLE, keys, EVENTS)
    have = known()
    legend = VA.build_legend(VA.parse_deck(DECK_A, resolver), TABLE, have)

    def res(v, leg=legend, **kw):
        return VA.resolve_value(v, leg, TABLE, have, key_map=keys, event_cards=EVENTS, **kw)

    for value, label in (("3", "GoblinDemolisher"), ("goblinstein", "goblinstein"), ("Goblinstein", "goblinstein"),
                         ("vines", "Vines"), ("Vines", "Vines"), ("rune-giant", "GiantBuffer"),
                         ("baby-dragon-ev1", "BabyDragon_EV1"), ("BabyDragon_EV1", "BabyDragon_EV1"),
                         ("goblin-curse", "GoblinCurse")):
        assert res(value) == (VA.Decision(label=label), None), value
    for value in ("x", "X", "o", "?", "s"):
        assert res(value) == (VA.Decision(code=value.lower()), None)
    refused = {v: res(v)[1] for v in ("Goblinstien", "Knight", "99", "PrincessTower", "zap")}
    assert "unknown name 'Goblinstien'" in refused["Goblinstien"] and "did you mean goblinstein" in refused["Goblinstien"]
    assert "not on the declared deck's legend" in refused["Knight"] and "--off-deck" in refused["Knight"]
    assert refused["99"] == "99 is not a legend number (0-13)" and "is a tower" in refused["PrincessTower"]
    assert "not on the declared deck's legend" in refused["zap"]
    assert res("Knight", off_deck=True) == (VA.Decision(label="Knight"), None)
    assert "no friendly deck holds" in res("SuperKnight", off_deck=True)[1]
    legend_c = VA.build_legend(VA.parse_deck(DECK_C, resolver), TABLE, have)
    for value in ("MergeMaiden", "spirit-empress"):
        assert "seen as its units: name one of MergeMaiden_Mounted, MergeMaiden_Normal" in res(value, legend_c)[1]
    assert "HealSpirit" in res("Heal", legend_c)[1]
    assert res("Zap", legend_c) == res("zap", legend_c) == (VA.Decision(label="zap"), None)
    assert res("MergeMaiden_Normal", legend_c) == (VA.Decision(label="MergeMaiden_Normal"), None)
    # suggestions: the live read in its form, else the harvest candidates; none when the form is open
    assert VA.suggest({"read": {"live": "BabyDragon", "form": 1}}, legend, TABLE, have) == 13
    open_read = {"read": {"live": "BabyDragon", "form": 0, "form_open": "base or evo? (a base read ...)"}}
    assert VA.suggest(open_read, legend, TABLE, have) is None
    assert VA.form_choice(open_read, legend, TABLE) == [12, 13]
    assert VA.form_choice({"read": {"live": "BabyDragon", "form": 0}}, legend, TABLE) == []
    assert VA.suggest({"read": {"live": "Knight", "form": 0}, "cands": [["Giant", 0.2], ["GiantBuffer", 0.1]]},
                      legend, TABLE, have) == 2
    assert VA.suggest({"read": {}, "cands": [["Knight", 0.3]]}, legend, TABLE, have) is None


def hand_sheet(tmp_path, keys, sid="20261101_A_1"):
    """An ingested session, a scan in its store (two small files: label only hashes them) and
    a sheet.json written by hand: clusters 0 (tracks 1, 2), 1 (track 3), 2 (track 4 split in two
    parts), track 5 auto-accepted as BabyDragon."""
    root = tmp_path / "harvest" / sid
    fake_recording(root)
    run_ingest(tmp_path, root, key_map=keys)
    paths = VA.session_paths(sid, tmp_path / "labels", tmp_path / "store")
    paths.scan_dir.mkdir(parents=True)
    (paths.scan_dir / VA.SCAN_JSON).write_text('{"created": "2026-11-01T21:00:00"}', encoding="utf-8")
    (paths.scan_dir / VA.SCAN_TRACKS).write_text('{"harvest": []}', encoding="utf-8")
    legend = VA.build_legend(VA.parse_deck(DECK_A, VA.CardResolver(TABLE, keys, EVENTS)), TABLE, known())

    def part(key, h, p=0, parts=1, live=None):
        return {"key": key, "h": h, "part": p, "parts": parts, "hk": 1, "side": "enemy", "n": 10, "first_n": 10 * h,
                "last_n": 10 * h + 9, "first_t": 0.5 * h, "last_t": 0.5 * h + 0.45, "flags": {"S1": "no live track"},
                "read": {"live": live, "form": 0, "card": None, "how": None}, "cands": [], "live_classes": []}

    doc = {"schema": VA.SHEET_SCHEMA, "sid": sid, "sheet": 1, "scan": VA.scan_identity(paths.scan_dir),
           "legend": [e.to_dict() for e in legend],
           "clusters": [{"id": 0, "tracks": ["1", "2"]}, {"id": 1, "tracks": ["3"]}, {"id": 2, "tracks": ["4.0", "4.1"]}],
           "auto": [{"key": "5", "label": "BabyDragon", "legend": 12, "n": 10}],
           "tracks": {"1": part("1", 1), "2": part("2", 2), "3": part("3", 3), "4.0": part("4.0", 4, 0, 2),
                      "4.1": part("4.1", 4, 1, 2), "5": part("5", 5, live="BabyDragon")}}
    VA.write_json(paths.sheet_file, doc)
    return paths


def test_bad_names_store_nothing_and_good_maps_write_tracks_json(tmp_path, keys, capsys):
    sid = "20261101_A_1"
    paths = hand_sheet(tmp_path, keys, sid)
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE, event_cards=EVENTS)
    kw1 = dict(kw, sheet=1)
    with pytest.raises(VA.LabelError) as err:            # every bad entry is listed, nothing is stored
        VA.label_session(sid, "0=Goblinstien,1=Knight,9=x,t99=?,2=goblinstein,t3=s", **kw1)
    text = " | ".join(err.value.problems)
    for needle in ("cluster 0: unknown name 'Goblinstien'", "cluster 1: 'Knight' (Knight) is not on the declared deck",
                   "no cluster 9 on sheet 1", "no track t99 on sheet 1", "track t3: s splits a cluster"):
        assert needle in text, needle
    assert len(err.value.problems) == 5 and not paths.tracks_file.exists()
    for bad, needle in (("1=s", "has one track"), ("0=3,0=4", "labelled twice"), ("0=goblinstein,=", "not ID=VALUE"),
                        ("t4=x", "it was split, use t4.0, t4.1")):
        with pytest.raises(VA.LabelError, match=needle):
            VA.label_session(sid, bad, **kw1)
    with pytest.raises(VA.LabelError, match="renumbered on every sheet") as err:   # cluster ids need --sheet
        VA.label_session(sid, "0=goblinstein", **kw)
    assert len(err.value.problems) == 1
    assert not paths.tracks_file.exists()
    assert cli(tmp_path, "label", sid, "--map", "0=Goblinstien", "--sheet", "1") == VA.EXIT_INPUT
    assert "nothing stored" in capsys.readouterr().out and not paths.tracks_file.exists()
    assert cli(tmp_path, "label", sid, "--map", "0=goblinstein") == VA.EXIT_INPUT
    assert "--sheet" in capsys.readouterr().out and not paths.tracks_file.exists()

    res = VA.label_session(sid, "0=goblinstein,1=x,2=s", **kw1)
    assert (res.stored, res.auto, res.split, res.open_clusters) == (Counter({"goblinstein": 2, "s": 2, "x": 1}), 1, 2, [])
    doc = json.loads(paths.tracks_file.read_text(encoding="utf-8"))
    assert doc["schema"] == VA.TRACKS_SCHEMA and doc["scan"] == VA.scan_identity(paths.scan_dir)
    assert [(t["key"], t["label"], t["code"], t["source"], t["sheet"], t["cluster"]) for t in doc["tracks"]] == [
        ("1", "goblinstein", None, "owner", 1, 0), ("2", "goblinstein", None, "owner", 1, 0),
        ("3", None, "x", "owner", 1, 1), ("5", "BabyDragon", None, "auto", 1, None)]
    assert doc["split"] == ["4.0", "4.1"] and doc["tracks"][0]["flags"] == ["S1"] and "role" not in doc
    assert [(t["side"], t["side_read"]) for t in doc["tracks"]] == [("enemy", "enemy")] * 4   # labels train as enemy
    assert doc["history"][-1]["map"] == "0=goblinstein,1=x,2=s" and doc["history"][-1]["stored"] == 3
    # labelled before: refused without --force; --force changes it and keeps the history
    with pytest.raises(VA.LabelError, match="labelled already"):
        VA.label_session(sid, "0=GiantBuffer", **kw1)
    with pytest.raises(VA.LabelError, match="labelled already"):
        VA.label_session(sid, "t3=?", **kw)
    VA.label_session(sid, "0=2,t3=?", force=True, **kw1)
    doc = json.loads(paths.tracks_file.read_text(encoding="utf-8"))
    assert [(t["key"], t["label"], t["code"]) for t in doc["tracks"]] == [
        ("1", "GiantBuffer", None), ("2", "GiantBuffer", None), ("3", None, "?"), ("5", "BabyDragon", None)]
    assert len(doc["history"]) == 2 and doc["history"][-1]["force"] is True
    # a sheet number or a scan that is not the sheet's: refused
    with pytest.raises(VA.LabelError, match="the latest sheet of 20261101_A_1 is 1"):
        VA.label_session(sid, None, sheet=2, **kw)
    # status: the stages and the labelled training tracks
    st = VA.status(tmp_path / "labels", tmp_path / "store", registry=VA.Registry.load())
    assert st["sessions"][0]["stages"] == ["ingest", "scan", "sheet", "label"]
    cov = {c["label"]: c["tracks"] for c in st["coverage"]}
    assert cov["GiantBuffer"] == 2 and cov["goblinstein"] == 0
    (paths.scan_dir / VA.SCAN_TRACKS).write_text('{"harvest": [1]}', encoding="utf-8")      # scanned again
    with pytest.raises(VA.LabelError, match="made from another scan"):
        VA.label_session(sid, "", **kw)
    assert cli(tmp_path, "label", sid) == VA.EXIT_INPUT and "another scan" in capsys.readouterr().out


def synthetic_scan(tmp_path, keys, sid="20261101_A_1"):
    """An ingested session with a hand-made scan (scan_frames.jsonl, tracks.json, one
    background) over 40 real npy images of a grass board: h1 an enemy Rune Giant the live
    chain reads well (auto-accepted in a training session); h2, h3 red blobs; h4 a blue blob;
    h5 white, then blue from its 13th image (split)."""
    np = pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH
    from scripts.vision.geometry import ArenaGeometry
    from scripts.vision.record import VisionRecorder
    from scripts.vision.types import SCREEN_BATTLE, VisionFrame

    geom, size, frames = ArenaGeometry.builtin(), (360, 640), 40
    grass = (82, 152, 70)
    colours = {"magenta": (200, 40, 200), "red": (220, 40, 40), "blue": (40, 40, 220), "white": (240, 240, 240)}

    def box_at(x, y):
        gx, gy = (float(v) for v in geom.world_to_px(x, y, size))
        return [round(gx - 7, 1), round(gy - 14, 1), round(gx + 7, 1), round(gy, 1)]

    plan = {1: (range(0, 20), 4500, ["magenta"] * 20), 2: (range(0, 12), 9000, ["red"] * 12),
            3: (range(20, 32), 9000, ["red"] * 12), 4: (range(0, 20), 13000, ["blue"] * 20),
            5: (range(0, 24), 16000, ["white"] * 12 + ["blue"] * 12)}
    recs = [{"n": n, "i": n, "image": f"images/{n:08d}.npy", "t": round(n * 0.05, 4), "screen": SCREEN_BATTLE,
             "size": list(size), "seg": 1, "battle": 1, "hk": 1, "epoch": "b1e0", "live": [], "raw": [], "plays": [],
             "props": [], "blobs": [], "tracks": []} for n in range(frames)]
    images = []
    for n in range(frames):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        img[:] = grass
        images.append(img)
    for h, (ns, x, cols) in plan.items():
        for j, n in enumerate(ns):
            box = box_at(x, 6000 + 100 * j)
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            images[n][y0:y1, x0:x1] = colours[cols[j]]
            p = {"box": box, "src": "det" if h == 1 else "blob", "type": "unit" if h == 1 else "blob", "conf": 0.8,
                 "cands": [["GiantBuffer", 1, 0.8]] if h == 1 else [], "blob": None, "live": None, "live_cls": None,
                 "x": x, "y": 6000 + 100 * j, "h": h}
            if h == 1:
                p["live"], p["live_cls"] = [1, 1], "GiantBuffer"
                recs[n]["live"].append([1, "GiantBuffer", "unit", 0.8, 0.9, 0, *box])
            recs[n]["props"].append(p)
    rec = VisionRecorder(tmp_path / "harvest" / sid, session="capture_t", image_every=1, image_format="npy",
                         clock=lambda: 0.0, wall_clock=lambda: 0.0)
    for n, img in enumerate(images):
        rec.write(VisionFrame(t_capture=round(n * 0.05, 4), screen=SCREEN_BATTLE, frame_size=size), img)
    rec.close()
    run_ingest(tmp_path, tmp_path / "harvest" / sid, key_map=keys)
    paths = VA.session_paths(sid, tmp_path / "labels", tmp_path / "store")
    tracks: dict = {}
    for r in recs:
        for k, p in enumerate(r["props"]):
            tracks.setdefault(p["h"], UH.HarvestTrack(p["h"], 1)).add(r["n"], k, r["t"], UH.Proposal.from_dict(p))
    live = {"1:1": {"first_t": 0.0, "last_t": 0.95, "frames": 20, "cls": {"GiantBuffer": 20}, "team": 0, "how": "play",
                    "card": 7, "card_name": "GiantBuffer", "forms": {"0": 20}}}
    (paths.scan_dir / "background").mkdir(parents=True)
    np.save(paths.scan_dir / "background" / "b1e0.npy", images[39])          # frame 39: the empty board
    (paths.scan_dir / UH.TRACKS_JSON).write_text(json.dumps({"schema": UH.SCAN_SCHEMA, "live": live, "harvest": [
        tracks[h].to_dict() for h in sorted(tracks)]}), encoding="utf-8")
    (paths.scan_dir / UH.SCAN_FRAMES).write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (paths.scan_dir / UH.SCAN_JSON).write_text(json.dumps({
        "schema": UH.SCAN_SCHEMA, "created": "2026-11-01T21:00:00", "session_dir": "elsewhere",
        "epochs": [{"id": "b1e0", "image": "background/b1e0.npy"}], "meta": {}}), encoding="utf-8")
    return paths, geom


def test_sheet_and_label_on_a_synthetic_scan(tmp_path, keys):
    pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH

    assert (VA.SCAN_DIR, VA.SCAN_JSON, VA.SCAN_TRACKS) == (UH.SCAN_DIR, UH.SCAN_JSON, UH.TRACKS_JSON)
    sid = "20261101_A_1"
    paths, geom = synthetic_scan(tmp_path, keys, sid)
    assert VA.scan_identity(paths.scan_dir) == UH.scan_identity(paths.scan_dir)

    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE)
    logs = []
    res = VA.make_sheet(sid, geometry=geom, log=logs.append, **kw)
    assert (res.sheet, res.clusters, res.pending, res.auto) == (1, 3, 5, 1)
    doc = res.doc
    assert [c["tracks"] for c in doc["clusters"]] == [["4", "5.1"], ["2", "3"], ["5.0"]]
    assert [c["flags"] for c in doc["clusters"]] == [{"S1": 2, "S7": 1}, {"S1": 2}, {"S1": 1, "S7": 1}]
    assert doc["auto"] == [{"key": "1", "label": "GiantBuffer", "legend": 2, "n": 20}] and doc["auto_on"] is True
    assert doc["tracks"]["1"]["suggest"] == 2 and doc["tracks"]["2"]["suggest"] is None
    assert [e["label"] for e in doc["legend"]] == LEGEND_A and doc["scan"] == VA.scan_identity(paths.scan_dir)
    assert res.image is not None and res.image.exists() and res.image.parent == paths.sheet_dir
    assert res.images == [res.image] and doc["images"] == [res.image.name] and res.image.stem == "sheet_1"
    assert (paths.sheet_dir / UH.ANALYSIS_JSON).is_file() and paths.sheet_file.is_file()
    legend_text = res.legend_file.read_text(encoding="utf-8")
    assert " 0 goblinstein" in legend_text and "#0" in legend_text and f"label {sid} --sheet 1 --map" in legend_text
    again = VA.make_sheet(sid, geometry=geom, log=logs.append, page_tiles=2, **kw)   # the analysis is reused
    assert again.sheet == 2 and any("reused" in line for line in logs)
    assert [c["tracks"] for c in again.doc["clusters"]] == [["4", "5.1"], ["2", "3"], ["5.0"]]
    assert [im.stem for im in again.images] == ["sheet_2_p1", "sheet_2_p2"] and again.image == again.images[0]
    assert all(im.exists() for im in again.images)

    with pytest.raises(VA.LabelError, match="the latest sheet"):
        VA.label_session(sid, "0=s", sheet=1, **kw)
    first = VA.label_session(sid, "0=s,1=goblinstein,2=x", sheet=2, **kw)
    assert (first.auto, first.split, first.open_clusters) == (1, 2, [])
    third = VA.make_sheet(sid, geometry=geom, log=logs.append, **kw)          # the split tracks, one per tile
    assert [c["tracks"] for c in third.doc["clusters"]] == [["4"], ["5.1"]] and third.auto == 0
    VA.label_session(sid, "0=Vines,1=?", sheet=3, **kw)
    doc = json.loads(paths.tracks_file.read_text(encoding="utf-8"))
    assert [(t["key"], t["label"], t["code"], t["source"], t["sheet"]) for t in doc["tracks"]] == [
        ("1", "GiantBuffer", None, "auto", 2), ("2", "goblinstein", None, "owner", 2),
        ("3", "goblinstein", None, "owner", 2), ("4", "Vines", None, "owner", 3), ("5.0", None, "x", "owner", 2),
        ("5.1", None, "?", "owner", 3)]
    assert doc["split"] == [] and (doc["tracks"][4]["first_n"], doc["tracks"][4]["last_n"]) == (0, 11)
    assert [(t["side"], t["side_read"]) for t in doc["tracks"][:4]] == [("enemy", "enemy")] + [("enemy", "unknown")] * 3
    assert doc["tracks"][0]["read"]["uid"] == "1:1" and doc["tracks"][0]["read"]["held"] == [["1:1", 0, "GiantBuffer", 0,
                                                                                             20, 0, 19]]
    done = VA.make_sheet(sid, geometry=geom, log=logs.append, **kw)
    assert (done.clusters, done.image, done.images) == (0, None, [])
    st = VA.status(tmp_path / "labels", tmp_path / "store", registry=VA.Registry.load())
    assert st["sessions"][0]["stages"] == ["ingest", "scan", "sheet", "label"]
    cov = {c["label"]: c["tracks"] for c in st["coverage"]}
    assert (cov["goblinstein"], cov["Vines"], cov["GiantBuffer"]) == (2, 1, 1)
    # labels on other images than the analysis cuts (other split settings): refused
    moved = json.loads(paths.tracks_file.read_text(encoding="utf-8"))
    moved["tracks"][5]["first_n"] += 1
    VA.write_json(paths.tracks_file, moved)
    with pytest.raises(VA.SheetError, match="on other images than this analysis cuts them"):
        VA.make_sheet(sid, geometry=geom, log=logs.append, **kw)
    moved["tracks"][5]["first_n"] -= 1
    VA.write_json(paths.tracks_file, moved)
    # a new scan would get new track ids: refused while tracks.json holds labels
    assert cli(tmp_path, "scan", sid, "--force", "--no-detector") == VA.EXIT_CHECK
    (paths.scan_dir / UH.TRACKS_JSON).write_text("{}", encoding="utf-8")
    with pytest.raises(VA.SheetError, match="labelled on another scan"):
        VA.make_sheet(sid, geometry=geom, log=logs.append, **kw)


def test_a_gold_session_is_never_auto_accepted(tmp_path, keys):
    pytest.importorskip("numpy")
    sid = "20261101_A_2"                                   # friendly 2: gold
    paths, geom = synthetic_scan(tmp_path, keys, sid)
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE)
    assert VA.load_session(sid, tmp_path / "labels")["role"] == VA.ROLE_GOLD
    with pytest.raises(VA.SheetError, match="gold session"):
        VA.make_sheet(sid, geometry=geom, log=lambda m: None, auto=True, **kw)
    res = VA.make_sheet(sid, geometry=geom, log=lambda m: None, **kw)
    assert (res.auto, res.pending, res.doc["auto_on"]) == (0, 6, False)
    assert any(c["tracks"] == ["1"] for c in res.doc["clusters"])                  # the Rune Giant is on the sheet
    assert "(gold: no auto-accept)" in res.legend_file.read_text(encoding="utf-8")
    # a sheet with auto-accepted tracks (made while the session trained): none stored now
    sheet = json.loads(paths.sheet_file.read_text(encoding="utf-8"))
    sheet["auto"] = [{"key": "1", "label": "GiantBuffer", "legend": 2, "n": 20}]
    VA.write_json(paths.sheet_file, sheet)
    out = VA.label_session(sid, "t2=goblinstein", **kw)
    assert out.auto == 0 and any("gold session now" in w for w in out.warnings)
    assert [t["key"] for t in out.doc["tracks"]] == ["2"]
    # status flags auto labels left in a gold session
    tdoc = json.loads(paths.tracks_file.read_text(encoding="utf-8"))
    tdoc["tracks"][0]["source"] = "auto"
    VA.write_json(paths.tracks_file, tdoc)
    rows = {r["sid"]: r for r in VA.status(tmp_path / "labels", tmp_path / "store")["sessions"]}
    assert any("auto-accepted label(s) in a gold session" in n for n in rows[sid]["notes"])


def test_sheet_helpers_mixed_clusters_and_counting_units():
    legend = [VA.LegendEntry(i, label, "Card") for i, label in enumerate(["BushGoblin", "Goblin", "Knight"])]
    parts = {"a": {"suggest": 0}, "b": {"suggest": 0}, "c": {"suggest": 1}, "d": {"suggest": None}, "e": {"suggest": 0}}
    assert VA._mix_text(list(parts.values()), legend) == ("mix 3x0 1x1", "mixed: 3x BushGoblin, 1x Goblin")
    assert VA._mix_text([parts["a"], parts["b"]], legend) == ("", "")
    assert VA._shown(["a", "b", "c", "d", "e"], parts) == ["a", "b", "c", "d"]
    assert VA._shown(["a", "b", "e"], parts) == []                                 # one label: spread as before
    side = VA.entry_side
    assert (side(VA.Decision(label="Vines"), {"side": "unknown"}), side(VA.Decision(code="o"), {"side": "enemy"}),
            side(VA.Decision(code="x"), {"side": "unknown"})) == ("enemy", "ally", "unknown")


def test_labelled_tracks_count_units_not_parts(tmp_path):
    VA.write_json(tmp_path / "s1" / VA.TRACKS_FILE, {"tracks": [
        {"label": "GoblinDemolisher", "h": 4, "part": 0}, {"label": "GoblinDemolisher", "h": 4, "part": 2},
        {"label": "GoblinDemolisher_kamikaze_form", "h": 4, "part": 1},
        {"label": "BushGoblin", "h": 7, "read": {"uid": "1:9"}}, {"label": "BushGoblin", "h": 8, "read": {"uid": "1:9"}},
        {"label": "BushGoblin", "h": 9, "read": {"uid": None}}, {"code": "x", "h": 10}, {"label": "Vines"}, {"label": "Vines"}]})
    got = VA.labelled_tracks(["s1"], tmp_path)
    assert got == Counter({"GoblinDemolisher": 1, "GoblinDemolisher_kamikaze_form": 1, "BushGoblin": 2, "Vines": 2})


def test_machine_paths_are_cut_from_stored_warnings():
    root = str(VA.ROOT)
    text = (f"the RoyaleAPI key map did not load (CardMapError: vendored card map {root}{os.sep}data{os.sep}cards"
            f"{os.sep}replay_cards.json is missing); registry at C:\\Users\\owner\\Desktop\\reg.json, "
            "/home/someone/x/reg.json and train/dev stay")
    out = VA.scrub_paths(text)
    assert root not in out and "owner" not in out and "someone" not in out
    assert f"data{os.sep}cards{os.sep}replay_cards.json" in out and "<path>/reg.json," in out and "train/dev" in out
