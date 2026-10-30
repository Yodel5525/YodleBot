"""N13: tools/vision_autolabel.py mix, and labels --pin-base.

docs/NEW_CARDS_PLAN.md 2.3 (training on KataCR plus the owner's frames), 6 (splits by session;
gold never trained on) and N13. Checked:
* pin_base writes the registry's base digest (and its label counts) and nothing else, once; a
  second run changes nothing; another base is refused; the `labels --pin-base` command;
* cr_mix.yaml: the extended names (the base's a prefix), train and val as lists of folders (the
  mix's and KataCR's), the extended detector_labels.json next to it, whose digest reaches
  scripts/train_detector.py's manifest; label lines that decode to the emitted boxes after
  trains_as; birth repeats as hard links; dev sessions in val only, never repeated; a frame
  marked bad left out; an npy emit refused by default;
* gold is never mixed: a named gold session, a session of a gold recording, an emit.json of
  kind gold and a stale emit are refused with nothing written; with the default selection no
  path of a gold session is in the mix and mix.json lists it as excluded;
* YOLO names equal under ultralytics' strip().lower(); the dataset yaml reader;
* the thin-form rule counts the labelled units of the mix's train sessions whose client may
  teach the label; images left out by kind; the command line, --force and a foreign file;
* `gpu`: one epoch from weights trained on the base names logs ultralytics' "Remapped 2/4 cls
  head rows from pretrained weights by class name".
Synthetic fixtures: hand-made emits (standard library), and emit on test_autolabel's synthetic
scan for the yaml test (numpy). The gpu test needs ultralytics, CUDA and OpenCV.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "tools"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vision_autolabel as VA  # noqa: E402
from scripts.vision import labels as L  # noqa: E402
from tests.pipeline.test_autolabel import DECK_B, TABLE, cli, fake_recording, run_ingest  # noqa: E402
from tests.pipeline.test_autolabel_emit import jsonl, label_synthetic  # noqa: E402

BASE_KATACR = ["hog-rider", "knight", "minion", "fireball"]


@pytest.fixture(scope="module")
def keys():
    key_map, note = VA.load_key_map()
    assert note is None, note
    return key_map


def write_katacr(k: Path, base: L.LabelSet, images=None) -> Path:
    """tmp/yolo as scripts/prepare_yolo_dataset.py writes it: cr.yaml (the base's names),
    detector_labels.json, images/{train,val} and labels/{train,val}. images(path, split, j)
    writes image j; by default one undecoded a.jpg per split (mix never decodes an image)."""
    for split in ("train", "val"):
        (k / "images" / split).mkdir(parents=True)
        (k / "labels" / split).mkdir(parents=True)
        if images is None:
            (k / "images" / split / "a.jpg").write_bytes(b"\xff\xd8 not decoded")
            (k / "labels" / split / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        else:
            images(k, split)
    names = "".join(f"  {i}: {VA._yaml_name(n)}\n" for i, n in enumerate(base.yolo_names()))
    (k / "cr.yaml").write_text(f"path: {k.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}",
                               encoding="utf-8")
    VA.write_json(k / "detector_labels.json", base.to_dict())
    return k


def tiny_katacr(root: Path):
    """(katacr folder, base, registry path): a small tmp/yolo over BASE_KATACR, and a copy of the
    shipped registry pinned to that base with pin_base."""
    base, _ = L.build_label_set(TABLE, BASE_KATACR)
    k = write_katacr(root / "tmp" / "yolo", base)
    reg = root / "detector_extra_labels.json"
    shutil.copyfile(VA.REGISTRY_PATH, reg)
    changed, _msg = VA.pin_base(reg, base)
    assert changed
    return k, base, reg


def session(tmp_path, keys, sid, *, role="auto", capture="harvest", client="16.635.1", rec=None, images=(0, 3, 6, 9),
            pruned=(12,), name="capture_20261101_201500"):
    """An ingested session (no scan): a fake recording of its own unless `rec` is given."""
    rec = rec or fake_recording(tmp_path / "rec" / sid, name, images=images, pruned=pruned)
    run_ingest(tmp_path, rec, deck=DECK_B, key_map=keys, sid=sid, role=role, capture=capture, client=client)
    return VA.session_paths(sid, tmp_path / "labels", tmp_path / "store"), rec


def hand_emit(paths, rows, *, tracks=None, suffix=".jpg", write=None):
    """An emit as emit_session leaves it, by hand: tracks.json (`tracks` entries), emit/emit.json
    with what emit_fresh compares, boxes.jsonl rows (frame i, boxes, repeat) and one image per row
    (write(path, i), else a few undecoded bytes)."""
    doc = VA.load_session(paths.sid, paths.labels_dir.parent)
    if tracks is not None:
        VA.write_json(paths.tracks_file, {"schema": VA.TRACKS_SCHEMA, "tracks": tracks, "split": []})
    images = paths.emit_dir / VA.IMAGES_DIR
    images.mkdir(parents=True)
    lines = []
    for i, boxes, repeat in rows:
        name = f"{paths.sid}_{i:08d}{suffix}"
        if write is None:
            (images / name).write_bytes(b"emitted %d" % i)
        else:
            write(images / name, i)
        lines.append(json.dumps({"image": name, "i": i, "n": i, "t": round(i * 0.05, 4), "hk": 1, "repeat": repeat,
                                 "birth": repeat > 1, "boxes": boxes, "masks": []}) + "\n")
    (paths.emit_dir / VA.BOXES_FILE).write_text("".join(lines), encoding="utf-8")
    VA.write_json(paths.emit_dir / VA.EMIT_JSON, {
        "schema": VA.EMIT_SCHEMA, "sid": paths.sid, "kind": VA.KIND_TRAIN, "role": doc["role"],
        "created": "2026-11-01T22:00:00", "tracks_sha1": VA.sha1_file(paths.tracks_file), "bad_tracks": [],
        "image_format": suffix.lstrip(".")})
    return paths


def track(key, label):
    return {"key": str(key), "h": int(str(key).split(".")[0]), "label": label, "code": None, "source": "owner",
            "at": "2026-11-01T21:00:00", "read": {}}


def box(label, side=1, cx=0.5, cy=0.5, key="1"):
    return [label, side, cx, cy, 0.1, 0.12, "owner", key]


def train_detector_module():
    """scripts/train_detector.py as the module `train_detector`, as test_train_detector_flags.py loads it."""
    if "train_detector" in sys.modules:
        return sys.modules["train_detector"]
    spec = importlib.util.spec_from_file_location("train_detector", ROOT / "scripts" / "train_detector.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_detector"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# labels --pin-base
# ---------------------------------------------------------------------------

def test_pin_base(tmp_path, capsys):
    base, _ = L.build_label_set(TABLE, BASE_KATACR)
    reg = tmp_path / "detector_extra_labels.json"
    shutil.copyfile(VA.REGISTRY_PATH, reg)
    before = reg.read_bytes()
    assert VA.Registry.load(reg).base_pin.get("digest") is None
    changed, msg = VA.pin_base(reg, base)
    assert changed and base.digest in msg and "base.labels 155 -> 4" in msg and "base.yolo_classes 310 -> 8" in msg
    after = reg.read_bytes()
    old, new = before.decode("utf-8").splitlines(), after.decode("utf-8").splitlines()
    assert len(old) == len(new)
    changed_lines = [(a, b) for a, b in zip(old, new) if a != b]
    assert [b.strip() for _a, b in changed_lines] == ['"labels": 4,', '"yolo_classes": 8,', f'"digest": "{base.digest}",']
    entries = [line for line in new if line.lstrip().startswith('{"label":')]
    assert len(entries) == len(VA.Registry.load(VA.REGISTRY_PATH)) and entries == [
        line for line in old if line.lstrip().startswith('{"label":')]                  # one entry per line, as it was
    doc_old, doc_new = json.loads(before), json.loads(after)
    doc_old["base"].update(digest=base.digest, labels=4, yolo_classes=8)
    assert doc_new == doc_old
    reg2 = VA.Registry.load(reg)
    assert reg2.base_pin["digest"] == base.digest and reg2.validate(TABLE) == []
    assert reg2.extend(base, require_pin=True).yolo_names()[:8] == base.yolo_names()
    # again: nothing changes; another base: refused, nothing written
    assert VA.pin_base(reg, base) == (False, f"{reg.name} is pinned to this base already ({base.digest})")
    assert reg.read_bytes() == after
    other, _ = L.build_label_set(TABLE, BASE_KATACR + ["zap"])
    with pytest.raises(VA.RegistryError, match="pinned to another base"):
        VA.pin_base(reg, other)
    assert reg.read_bytes() == after
    # a registry whose base this edit cannot reach: refused with the edit to make by hand
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"schema": VA.REGISTRY_SCHEMA, "labels": []}), encoding="utf-8")
    with pytest.raises(VA.RegistryError, match="by hand"):
        VA.pin_base(bare, base)
    # the command: a real base only, never the preview
    reg3 = tmp_path / "reg3.json"
    shutil.copyfile(VA.REGISTRY_PATH, reg3)
    base_file = VA.write_json(tmp_path / "detector_labels.json", base.to_dict())
    assert VA.main(["labels", "--registry", str(reg3), "--base", str(tmp_path / "none.json"), "--pin-base"]) == VA.EXIT_INPUT
    assert VA.main(["labels", "--registry", str(reg3), "--base", str(base_file), "--pin-base"]) == VA.EXIT_OK
    out = capsys.readouterr().out
    assert f"pinned base digest {base.digest}" in out and "commit " in out and reg3.read_bytes() == after
    assert VA.main(["labels", "--registry", str(reg3), "--base", str(base_file), "--pin-base"]) == VA.EXIT_OK
    assert "pinned to this base already" in capsys.readouterr().out
    other_file = VA.write_json(tmp_path / "other.json", other.to_dict())
    assert VA.main(["labels", "--registry", str(reg3), "--base", str(other_file), "--pin-base"]) == VA.EXIT_CHECK


def test_name_collisions_and_the_dataset_yaml(tmp_path):
    got = VA.name_collisions(["HogRider@ally", "Knight@ally", "hogrider@ALLY "])
    assert len(got) == 1 and "0 'HogRider@ally' and 2 'hogrider@ALLY '" in got[0]
    base, _ = L.build_label_set(TABLE, BASE_KATACR)
    reg = VA.Registry.load()
    pinned = VA.Registry(reg.entries, thin_tracks=reg.thin_tracks, base_pin={"digest": base.digest})
    assert VA.name_collisions(pinned.extend(base, require_pin=True).yolo_names()) == []
    assert VA.name_collisions(VA.crosswalk_base(TABLE).yolo_names()) == []              # tonight's names (the preview)
    # the yaml as prepare_yolo_dataset and mix write it, read back without a yaml library
    assert [VA._yaml_name(n) for n in ("HogRider@ally", "yes", "a b@enemy", "1x")] == [
        "HogRider@ally", '"yes"', '"a b@enemy"', '"1x"']
    y = tmp_path / "d.yaml"
    y.write_text('# a comment\npath: "C:/x y/tmp/mix/ft1"\ntrain: ["C:/a/train/images", "C:/b/images/train"]\n'
                 "val: images/val  # KataCR's\nnames:\n  0: HogRider@ally\n  1: \"a b@enemy\"\n\n  2: 'it''s'\n",
                 encoding="utf-8")
    assert VA.read_dataset_yaml(y) == {"path": "C:/x y/tmp/mix/ft1", "train": ["C:/a/train/images", "C:/b/images/train"],
                                       "val": "images/val", "names": ["HogRider@ally", "a b@enemy", "it's"]}
    y.write_text("names: [\"a@ally\", \"a@enemy\"]\ntrain: t\n", encoding="utf-8")
    assert VA.read_dataset_yaml(y)["names"] == ["a@ally", "a@enemy"]
    y.write_text("names:\n  0: a\n  2: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="skip an index"):
        VA.read_dataset_yaml(y)
    assert VA.same_recording({"frames_sha1": "ab"}, {"frames_sha1": "ab", "name": "x"})
    assert VA.same_recording({"name": "capture_1", "created": "t"}, {"name": "capture_1", "created": "t"})
    assert not VA.same_recording({"name": "capture_1", "created": "t"}, {"name": "capture_1", "created": "u"})
    assert not VA.same_recording({}, {}) and not VA.same_recording(None, {"frames_sha1": None})


# ---------------------------------------------------------------------------
# the mix on emitted synthetic sessions
# ---------------------------------------------------------------------------

def test_mix_yaml_and_labels(tmp_path, keys, monkeypatch):
    pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH

    k, base, reg_path = tiny_katacr(tmp_path)
    reg = VA.Registry.load(reg_path)
    ext = reg.extend(base, require_pin=True)
    assert ext.yolo_names()[:8] == base.yolo_names() and len(ext.yolo_names()) == 2 * (4 + len(reg))
    kwl = dict(labels_root=tmp_path / "labels", store=tmp_path / "store")
    cfg = UH.EmitConfig(birth_s=0.3, image_format="npy")
    paths = {}
    for sid in ("20261101_A_1", "20261101_A_4"):                                   # train, dev (deck A friendly 4)
        paths[sid], geom = label_synthetic(tmp_path, keys, sid)
        VA.emit_session(sid, config=cfg, table=TABLE, geometry=geom, log=lambda m: None, **kwl)
    p1, p4 = paths["20261101_A_1"], paths["20261101_A_4"]
    assert VA.load_session("20261101_A_4", tmp_path / "labels")["role"] == VA.ROLE_DEV
    VA.write_json(p1.marks_file, {"schema": VA.MARKS_SCHEMA, "bad_tracks": {},
                                  "bad_frames": {"20261101_A_1_00000009.npy": {"i": 9, "n": 9}}})
    assert VA.emit_fresh(p1, VA.load_session("20261101_A_1", tmp_path / "labels")) == []   # a frame mark needs no re-emit
    kw = dict(kwl, katacr=k, registry=reg, table=TABLE)
    out = tmp_path / "mix" / "ft1"

    # npy is no image to ultralytics: refused unless the caller accepts it (as here, for the test)
    with pytest.raises(VA.MixError, match="--format jpg") as err:
        VA.mix_sessions("ft1", out=out, **kw)
    assert err.value.code == VA.EXIT_CHECK and not out.exists()
    res = VA.mix_sessions("ft1", out=out, image_suffixes=(".npy",), clock=lambda: 0.0, **kw)
    assert res.out == out and res.yaml == out / VA.MIX_YAML

    # cr_mix.yaml: the extended names; train and val as lists of folders
    y = VA.read_dataset_yaml(res.yaml)
    assert y["names"] == ext.yolo_names() and y["names"][:len(base.yolo_names())] == base.yolo_names()
    text = res.yaml.read_text(encoding="utf-8")
    assert all(f"\n  {i}: {n}\n" in text for i, n in enumerate(ext.yolo_names()))
    assert y["path"] == out.resolve().as_posix()
    assert y["train"] == [(out / "train" / "images").resolve().as_posix(), (k / "images" / "train").resolve().as_posix()]
    assert y["val"] == [(k / "images" / "val").resolve().as_posix(), (out / "val" / "images").resolve().as_posix()]
    # the extended label set next to it, and its digest in train_detector's manifest
    ldoc = json.loads((out / VA.MIX_LABELS).read_text(encoding="utf-8"))
    assert ldoc["digest"] == ext.digest and ldoc["yolo_names"] == ext.yolo_names() and ldoc["base"]["digest"] == base.digest
    assert ldoc["mix"]["sessions"] == ["20261101_A_1", "20261101_A_4"]
    td = train_detector_module()
    assert td._label_set_digest(res.yaml) == ext.digest

    class FakeYOLO:
        kwargs: dict = {}

        def __init__(self, model):
            self.model = model

        def train(self, **kw_):
            FakeYOLO.kwargs = kw_
            w = Path(kw_["project"]) / kw_["name"] / "weights"
            w.mkdir(parents=True, exist_ok=True)
            (w / "best.pt").write_bytes(b"best")
            self.trainer = SimpleNamespace(best=str(w / "best.pt"))

    monkeypatch.setattr(td, "_import_torch", lambda: None)
    monkeypatch.setattr(td, "_import_yolo", lambda: FakeYOLO)
    assert td.main(["--model", str(tmp_path / "best.pt"), "--data", str(res.yaml), "--name", "ft1", "--optimizer", "AdamW",
                    "--lr0", "0.001", "--epochs", "1", "--no-pin", "--device", "cpu", "--project", str(tmp_path / "runs"),
                    "--pin-file", str(tmp_path / "pin.json")]) == 0
    manifest = json.loads((tmp_path / "runs" / "ft1" / "detector_manifest.json").read_text(encoding="utf-8"))
    assert FakeYOLO.kwargs["data"] == str(res.yaml) and manifest["label_set_digest"] == ext.digest
    assert manifest["data"] == res.yaml.resolve().as_posix()                      # the gate finds mix.json beside it
    # what the run trained on, as it started: the gate's leak check compares them with the mix.json it finds
    mix_doc = json.loads((out / VA.MIX_JSON).read_text(encoding="utf-8"))
    assert manifest["mix_sha1"] == hashlib.sha1((out / VA.MIX_JSON).read_bytes()).hexdigest()
    assert manifest["data_sha1"] == hashlib.sha1(res.yaml.read_bytes()).hexdigest()
    assert manifest["mix_created"] == mix_doc["created"]
    # the sessions were emitted against another base (the classes.json preview, or tmp/yolo's): said so
    assert any("emitted against another base label set" in w for w in res.warnings)

    # train: every label line decodes to the emitted box after trains_as; repeats are hard links
    timages, tlabels = out / "train" / "images", out / "train" / "labels"
    rows = jsonl(p1.emit_dir / VA.BOXES_FILE)
    assert len(rows) == 20 and sum(r["repeat"] for r in rows) == 34
    for r in rows:
        stem = Path(r["image"]).stem
        copies = sorted(f for f in timages.iterdir() if f.stem == stem or f.stem.startswith(stem + VA.REPEAT_SEP))
        if r["i"] == 9:                                                         # marked bad: left out
            assert copies == [] and not (tlabels / f"{stem}.txt").exists()
            continue
        assert [f.name for f in copies] == [r["image"]] + [f"{stem}__r{j}.npy" for j in range(1, r["repeat"])]
        for f in copies:
            assert os.path.samefile(f, p1.emit_dir / VA.IMAGES_DIR / r["image"])
            assert (tlabels / f"{f.stem}.txt").read_text() == (tlabels / f"{stem}.txt").read_text()
        lines = (tlabels / f"{stem}.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(r["boxes"]) > 0
        for line, b in zip(lines, r["boxes"]):
            idx, *xywh = line.split()
            assert ext.decode(int(idx)) == (VA.trains_as(b[0], ext, reg, 0, TABLE), b[1])
            assert [float(v) for v in xywh] == pytest.approx(b[2:6], abs=1e-6)
    assert ext.index("GiantBuffer") == 8 and {line.split()[0] for f in tlabels.iterdir() for line in f.read_text().splitlines()} \
        == {str(2 * ext.index(x) + 1) for x in ("GiantBuffer", "goblinstein", "Vines")}
    assert len(list(timages.iterdir())) == len(list(tlabels.iterdir())) == 33
    # dev: val only, one copy each
    vfiles = sorted(f.name for f in (out / "val" / "images").iterdir())
    assert vfiles == sorted(r["image"] for r in jsonl(p4.emit_dir / VA.BOXES_FILE)) and len(vfiles) == 20
    assert not any(VA.REPEAT_SEP in f for f in vfiles) and not any(f.startswith("20261101_A_4") for f in os.listdir(timages))

    # mix.json
    doc = json.loads((out / VA.MIX_JSON).read_text(encoding="utf-8"))
    assert doc == json.loads(json.dumps(res.doc))
    assert (doc["schema"], doc["name"], doc["label_set_digest"], doc["base_digest"]) == (
        VA.MIX_SCHEMA, "ft1", ext.digest, base.digest)
    s1, s4 = doc["sessions"]
    assert (s1["sid"], s1["split"], s1["images"], s1["files"], s1["repeats"]) == ("20261101_A_1", "train", 19, 33, 14)
    assert s1["skipped"] == {"bad_frame": 1, "no index": 0, "client": 0, "side": 0, "missing": 0, "duplicate": 0}
    assert (s4["sid"], s4["split"], s4["images"], s4["files"], s4["repeats"]) == ("20261101_A_4", "val", 20, 20, 0)
    assert s1["frames_sha1"] == VA.load_session("20261101_A_1", tmp_path / "labels")["recording"]["frames_sha1"]
    assert s1["recording"]["name"] == "capture_t" and s1["linked"] == 33 and s1["copied"] == 0
    assert doc["counts"] == {"owner_train_files": 33, "owner_val_files": 20} and doc["katacr_images"] == {"train": 1, "val": 1}
    assert doc["owner_share"] == round(33 / 34, 4) and doc["gold_excluded"] == []
    assert doc["trains_as"] == {"GiantBuffer": "GiantBuffer", "Vines": "Vines", "goblinstein": "goblinstein"}
    assert doc["track_counts"] == {"GiantBuffer": 1, "Vines": 1, "goblinstein": 2}             # train sessions only
    want = {}
    for r in rows:
        if r["i"] != 9:
            for b in r["boxes"]:
                want[VA.trains_as(b[0], ext, reg, 0, TABLE)] = want.get(VA.trains_as(b[0], ext, reg, 0, TABLE), 0) + 1
    assert doc["thin"] == [] and doc["boxes_by_label"] == dict(sorted(want.items())) and want["Vines"] == 10
    assert doc["val_boxes_by_label"] == {"GiantBuffer": 11, "Vines": 11, "goblinstein": 17}       # dev: image 9 too
    assert any("one recording" in w and "as train and as dev" in w for w in res.warnings)         # one synthetic recording
    assert "--data" in res.command and "--no-pin" in res.command and "--name ft1" in res.command


# ---------------------------------------------------------------------------
# gold is never mixed
# ---------------------------------------------------------------------------

def test_mix_refuses_gold(tmp_path, keys):
    k, base, reg_path = tiny_katacr(tmp_path)
    reg = VA.Registry.load(reg_path)
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", katacr=k, registry=reg, table=TABLE)
    gp, grec = session(tmp_path, keys, "20261029_R0_1", role="gold", capture="natural", images=(0, 3), pruned=(),
                       name="capture_20261029_195831")
    (gp.gold_dir / VA.GOLD_LABELS_DIR).mkdir(parents=True)                     # a gold emit's files: never mixed
    VA.write_json(gp.gold_dir / VA.EMIT_JSON, {"schema": VA.EMIT_SCHEMA, "kind": VA.KIND_GOLD})
    (gp.gold_dir / VA.GOLD_LABELS_DIR / "00000000.txt").write_text("19 0.5 0.5 0.1 0.1\n")
    tp, _ = session(tmp_path, keys, "20261101_B_1")
    hand_emit(tp, [(0, [box("BossBandit")], 2), (3, [box("BossBandit", cx=0.2)], 1), (6, [], 1)])

    # a named gold session: refused, nothing written
    out = tmp_path / "mix" / "g1"
    with pytest.raises(VA.MixError, match="gold is never mixed") as err:
        VA.mix_sessions("g1", ["20261101_B_1", "20261029_R0_1"], out=out, **kw)
    assert err.value.code == VA.EXIT_CHECK and not out.exists()
    with pytest.raises(VA.MixError, match="no ingested session") as err:
        VA.mix_sessions("g1", ["20261101_C_9"], out=out, **kw)
    assert err.value.code == VA.EXIT_INPUT and not out.exists()

    # the default selection leaves gold out, and no path of it reaches the mix
    res = VA.mix_sessions("g2", out=tmp_path / "mix" / "g2", **kw)
    doc = res.doc
    assert [s["sid"] for s in doc["sessions"]] == ["20261101_B_1"]
    assert doc["gold_excluded"] == [{"sid": "20261029_R0_1", "frames_sha1": VA.load_session(
        "20261029_R0_1", tmp_path / "labels")["recording"]["frames_sha1"],
        "recording": {"name": "capture_20261029_195831", "created": "2026-11-01T20:15:00"}}]
    assert "20261029_R0_1: gold, never mixed" in doc["left_out"]
    texts = [res.yaml.read_text(encoding="utf-8"), (res.out / VA.MIX_JSON).read_text(encoding="utf-8"),
             (res.out / VA.MIX_LABELS).read_text(encoding="utf-8")]
    for where in (gp.store_dir, gp.gold_dir, grec):
        for form in {where.resolve().as_posix(), str(where.resolve()), str(where)}:
            assert not any(form in t for t in texts), form
    gold_files = [f for f in gp.store_dir.rglob("*") if f.is_file()]
    for f in res.out.rglob("*"):
        if f.is_file():
            assert not any(os.path.samefile(f, g) for g in gold_files), f
    assert sorted(f.name for f in (res.out / "train" / "images").iterdir()) == [
        "20261101_B_1_00000000.jpg", "20261101_B_1_00000000__r1.jpg", "20261101_B_1_00000003.jpg",
        "20261101_B_1_00000006.jpg"]
    assert (res.out / "train" / "labels" / "20261101_B_1_00000006.txt").read_text() == ""      # a background image
    assert (res.out / "train" / "labels" / "20261101_B_1_00000003.txt").read_text() == \
        f"{2 * reg.extend(base, require_pin=True).index('BossBandit') + 1} 0.200000 0.500000 0.100000 0.120000\n"
    assert not (res.out / "val").exists() and VA.read_dataset_yaml(res.yaml)["val"] == [(k / "images" / "val").resolve().as_posix()]

    # a stale emit: refused, naming the re-emit
    VA.write_json(tp.tracks_file, {"schema": VA.TRACKS_SCHEMA, "tracks": [track(1, "BossBandit")], "split": []})
    with pytest.raises(VA.MixError, match="emit 20261101_B_1 --force") as err:
        VA.mix_sessions("g3", out=tmp_path / "mix" / "g3", **kw)
    assert err.value.code == VA.EXIT_CHECK and not (tmp_path / "mix" / "g3").exists()
    tp.tracks_file.unlink()
    # an emit.json of kind gold in a train session's emit folder: refused
    edoc = json.loads((tp.emit_dir / VA.EMIT_JSON).read_text(encoding="utf-8"))
    VA.write_json(tp.emit_dir / VA.EMIT_JSON, dict(edoc, kind=VA.KIND_GOLD))
    with pytest.raises(VA.MixError, match="only a training emit is mixed"):
        VA.mix_sessions("g4", out=tmp_path / "mix" / "g4", **kw)
    assert not (tmp_path / "mix" / "g4").exists()
    VA.write_json(tp.emit_dir / VA.EMIT_JSON, edoc)
    # a train session ingested from the gold recording: refused, named or not
    xp, _ = session(tmp_path, keys, "20261029_X_1", role="train", rec=grec)
    hand_emit(xp, [(0, [box("BossBandit")], 1)])
    for sids in (["20261029_X_1"], None):
        with pytest.raises(VA.MixError, match="the same recording is gold as 20261029_R0_1"):
            VA.mix_sessions("g5", sids, out=tmp_path / "mix" / "g5", **kw)
    assert not (tmp_path / "mix" / "g5").exists()
    # and a gold folder is no KataCR dataset
    with pytest.raises(VA.MixError):
        VA.mix_sessions("g6", ["20261101_B_1"], **dict(kw, katacr=gp.gold_dir), out=tmp_path / "mix" / "g6")
    assert not (tmp_path / "mix" / "g6").exists()
    # an emitted image that is a file of the gold folder (a link into it): refused
    img = tp.emit_dir / VA.IMAGES_DIR / "20261101_B_1_00000006.jpg"
    img.unlink()
    try:
        img.symlink_to(gp.gold_dir / VA.GOLD_LABELS_DIR / "00000000.txt")
    except (OSError, NotImplementedError):
        return                                  # no symlinks here (Windows without the privilege)
    with pytest.raises(VA.MixError, match="lies in a gold folder"):
        VA.mix_sessions("g7", ["20261101_B_1"], out=tmp_path / "mix" / "g7", **kw)
    assert not (tmp_path / "mix" / "g7").exists()


# ---------------------------------------------------------------------------
# the thin-form rule, what is left out, the command line
# ---------------------------------------------------------------------------

def test_thin_rule_skips_and_the_command(tmp_path, keys, capsys):
    base, _ = L.build_label_set(TABLE, BASE_KATACR)
    k = write_katacr(tmp_path / "tmp" / "yolo", base)
    entries = [{"label": "Berserker", "card": "Berserker", "type": "unit", "form": "base"},
               {"label": "BerserkerHero", "card": "Berserker", "type": "unit", "form": "hero", "base": "Berserker",
                "valid_from_client": "16.700.0"},
               {"label": "BossBandit", "card": "BossBandit", "type": "unit", "form": "base"}]
    reg_path = VA.write_json(tmp_path / "reg.json", {"schema": VA.REGISTRY_SCHEMA, "base": {"digest": base.digest},
                                                     "labels": entries})
    reg = VA.Registry.load(reg_path)
    ext = reg.extend(base, require_pin=True)
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", katacr=k, registry=reg, table=TABLE)
    # X (16.635.1) labelled 5 BerserkerHero units, Y (16.700.0) 4: only Y's count (valid from 16.700.0)
    xp, _ = session(tmp_path, keys, "20261101_B_1", images=(0, 3, 6, 9))
    yp, _ = session(tmp_path, keys, "20261101_B_3", client="16.700.0", images=(0, 3, 6), pruned=(),
                    name="capture_20261101_203000")
    hand_emit(xp, [(0, [box("BerserkerHero")], 1), (3, [box("BossBandit")], 2), (6, [box("GoldenKnight")], 1),
                   (9, [box("BossBandit", side=2)], 1), (12, [box("BossBandit")], 1)],
              tracks=[track(h, "BerserkerHero") for h in range(1, 6)] + [track(9, "BossBandit")])
    (xp.emit_dir / VA.IMAGES_DIR / "20261101_B_1_00000012.jpg").unlink()                      # gone: "missing"
    hand_emit(yp, [(0, [box("BerserkerHero"), box("Berserker", cx=0.3)], 1)],
              tracks=[track(h, "BerserkerHero") for h in range(1, 5)] + [track("6.0", "BerserkerHero"),
                                                                          track("6.1", "BerserkerHero")])
    marks = {"schema": VA.MARKS_SCHEMA, "bad_frames": {}, "bad_tracks": {
        "4": {"label_at": "2026-11-01T21:00:00"}, "6.1": {"label_at": "2026-11-01T21:00:00"}}}
    VA.write_json(yp.marks_file, marks)                       # masked by the emit: h4 does not count, h6 does (6.0)
    yp_emit = json.loads((yp.emit_dir / VA.EMIT_JSON).read_text(encoding="utf-8"))
    VA.write_json(yp.emit_dir / VA.EMIT_JSON, dict(yp_emit, bad_tracks=["4", "6.1"]))
    out = tmp_path / "mix" / "ft2"
    res = VA.mix_sessions("ft2", out=out, **kw)
    doc = res.doc
    assert doc["track_counts"] == {"BerserkerHero": 4, "BossBandit": 1}             # Y's h1-h3 and h6; none of X's 5
    assert doc["trains_as"] == {"BerserkerHero": "Berserker", "Berserker": "Berserker", "BossBandit": "BossBandit"}
    assert doc["thin"] == ["BerserkerHero trains as Berserker: 4 of 8 tracks"]
    sx, sy = doc["sessions"]
    assert sx["skipped"] == {"bad_frame": 0, "no index": 1, "client": 1, "side": 1, "missing": 1, "duplicate": 0}
    assert (sx["images"], sx["files"], sy["images"], sy["files"]) == (1, 2, 1, 1)
    yl = (out / "train" / "labels" / "20261101_B_3_00000000.txt").read_text().splitlines()
    assert [ext.decode(int(line.split()[0])) for line in yl] == [("Berserker", 1), ("Berserker", 1)]
    assert any(w.startswith("1 image(s) left out: no index: GoldenKnight") for w in res.warnings)
    assert any("client: BerserkerHero is not taught by client 16.635.1" in w for w in res.warnings)
    # with Y's tracks at 8 the form trains as itself
    VA.write_json(yp.tracks_file, {"schema": VA.TRACKS_SCHEMA, "split": [], "tracks": [
        track(h, "BerserkerHero") for h in range(1, 9)]})
    VA.write_json(yp.emit_dir / VA.EMIT_JSON, dict(yp_emit, tracks_sha1=VA.sha1_file(yp.tracks_file)))
    yp.marks_file.unlink()
    again = VA.mix_sessions("ft2", out=out, force=True, **kw)
    assert again.doc["trains_as"]["BerserkerHero"] == "BerserkerHero"
    assert again.doc["thin"] == ["BerserkerHero trains as itself: 8 tracks (8 needed)"]
    yl = (out / "train" / "labels" / "20261101_B_3_00000000.txt").read_text().splitlines()
    assert [ext.decode(int(line.split()[0])) for line in yl] == [("BerserkerHero", 1), ("Berserker", 1)]

    # the command line: the counts, the thin decisions and the train command; then --force
    args = ["mix", "--name", "ft2", "--katacr", str(k), "--registry", str(reg_path), "--out", str(out)]
    assert cli(tmp_path, *args) == VA.EXIT_CHECK
    assert "holds a mix already" in capsys.readouterr().out
    (out / "train" / "labels.cache").write_bytes(b"ultralytics' cache")
    assert cli(tmp_path, *args, "--force") == VA.EXIT_OK
    printed = capsys.readouterr().out
    assert "BerserkerHero trains as itself: 8 tracks (8 needed)" in printed
    assert "20261101_B_1           train -> train     1 image(s) + 1 repeat(s); left out: 1 no index, 1 client, 1 side, " \
           "1 missing" in printed
    assert "train: 3 owner file(s) and 1 KataCR: 75.0% owner; val: 0 owner and 1 KataCR" in printed
    assert "Remapped 8/14 cls head rows" in printed and len(base.yolo_names()) == 8 and len(ext.yolo_names()) == 14
    assert "scripts\\train_detector.py --model <pinned best.pt> --data" in printed and "--name ft2" in printed
    assert not (out / "train" / "labels.cache").exists()                     # the mix's own folder is rebuilt
    (out / "notes.txt").write_text("mine")
    assert cli(tmp_path, *args, "--force") == VA.EXIT_CHECK
    assert "holds other files (notes.txt)" in capsys.readouterr().out and (out / "notes.txt").exists()
    # never inside KataCR's folder (its label cache) or the store
    for bad in (k / "mix", tmp_path / "store" / "mix"):
        with pytest.raises(VA.MixError, match="overlaps") as err:
            VA.mix_sessions("ft3", out=bad, **kw)
        assert err.value.code == VA.EXIT_CHECK and not bad.exists()
    with pytest.raises(VA.MixError, match="not pinned yet"):
        VA.mix_sessions("ft3", out=tmp_path / "mix" / "ft3",
                        **dict(kw, registry=VA.Registry.from_dict({"schema": VA.REGISTRY_SCHEMA, "labels": entries})))
    assert not (tmp_path / "mix" / "ft3").exists()


# ---------------------------------------------------------------------------
# ultralytics: the class rows are copied by name
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_mix_finetune_logs_class_remap(tmp_path, keys):
    pytest.importorskip("ultralytics")
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from ultralytics import YOLO

    rng = np.random.default_rng(0)

    def png(path, frac=(0.3, 0.3, 0.6, 0.7)):
        img = rng.integers(0, 60, (64, 64, 3), dtype=np.uint8)
        x0, y0, x1, y1 = (int(v * 64) for v in frac)
        img[y0:y1, x0:x1] = (40, 200, 230)
        assert cv2.imwrite(str(path), img)

    def katacr_images(root, split):
        for j in range(4 if split == "train" else 2):
            png(root / "images" / split / f"k{j}.png")
            (root / "labels" / split / f"k{j}.txt").write_text(f"{j % 2} 0.45 0.5 0.3 0.4\n", encoding="utf-8")

    base, _ = L.build_label_set(TABLE, ["hog-rider"])
    assert base.yolo_names() == ["HogRider@ally", "HogRider@enemy"]
    k = write_katacr(tmp_path / "tmp" / "yolo", base, images=katacr_images)
    reg = VA.Registry.from_dict({"schema": VA.REGISTRY_SCHEMA, "base": {"digest": base.digest}, "labels": [
        {"label": "BossBandit", "card": "BossBandit", "type": "unit", "form": "base"}]})
    common = dict(epochs=1, imgsz=64, batch=2, workers=0, device=0, amp=False, plots=False, exist_ok=True,
                  project=str(tmp_path / "runs"))
    YOLO("yolov8n.yaml").train(data=str(k / "cr.yaml"), name="base", **common)
    last = tmp_path / "runs" / "base" / "weights" / "last.pt"
    assert last.is_file()

    tp, _ = session(tmp_path, keys, "20261101_B_1")
    hand_emit(tp, [(i, [box("BossBandit", cx=0.35, cy=0.4)], 1) for i in range(4)], suffix=".png",
              write=lambda p, i: png(p, (0.2, 0.2, 0.5, 0.6)))
    res = VA.mix_sessions("ft1", out=tmp_path / "mix" / "ft1", katacr=k, registry=reg, table=TABLE,
                          labels_root=tmp_path / "labels", store=tmp_path / "store")
    assert VA.read_dataset_yaml(res.yaml)["names"] == ["HogRider@ally", "HogRider@enemy", "BossBandit@ally",
                                                        "BossBandit@enemy"]
    messages: list[str] = []

    class Grab(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    katacr_cache = (k / "labels" / "train.cache").read_bytes()                   # written by the first run
    logger, grab = logging.getLogger("ultralytics"), Grab()
    level = logger.level
    logger.addHandler(grab)
    logger.setLevel(logging.INFO)
    try:
        YOLO(str(last)).train(data=str(res.yaml), name="ft1", **common)
    finally:
        logger.removeHandler(grab)
        logger.setLevel(level)
    assert any("Remapped 2/4 cls head rows from pretrained weights by class name" in m for m in messages), messages[-30:]
    assert (tmp_path / "mix" / "ft1" / "train" / "labels.cache").is_file()          # D16: the cache is the mix's,
    assert (k / "labels" / "train.cache").read_bytes() == katacr_cache                # KataCR's is not rewritten
