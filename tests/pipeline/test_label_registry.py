"""N07: the append-only detector label registry (data/vision/detector_extra_labels.json).

tools/vision_autolabel.py's Registry is read against the shipped data/vision/classes.json. The
base label set is the one scripts/prepare_yolo_dataset.py builds from classes.json's own KataCR
names (vision_autolabel.crosswalk_base): 155 labels x 2 sides = 310 YOLO classes, what the
owner's PC wrote on 2026-10-30. The checks:

* the seed is the plan's 19 phase-1 labels (16 units, 3 spells) in the plan's order, and later
  entries may only follow them;
* every entry resolves through scripts/vision/labels.resolve_label as what it claims to be, and
  the extension equals prepare_yolo_dataset's --extra-label path;
* the old yolo_names() stays a prefix (also with an extra unmapped base name, and without the
  joint side), and check_prefix catches a reorder or a removal;
* train_as writes the base index while a form is thin; valid_from_client and reworks;
* the `labels` command on files in tmp_path.

Standard library and the class table only: no image, GPU or engine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "tools"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vision_autolabel as VA  # noqa: E402
from scripts.vision import labels as L  # noqa: E402

TABLE = L.load_class_table()
REG = VA.Registry.load()
KATACR_NAMES = list(TABLE.doc["katacr_names"])
PHASE1 = ["goblinstein", "goblinstein_doctor", "Berserker", "BerserkerHero", "GiantBuffer",
          "GoblinDemolisher", "GoblinDemolisher_kamikaze_form", "SuspiciousBush", "BushGoblin",
          "BossBandit", "Ronin", "GoblinMachine", "MergeMaiden_Normal", "MergeMaiden_Mounted",
          "MinionGiant", "ThreeMusketeer_Rework_Character_1", "DarkMagic", "Vines", "GoblinCurse"]


def entry(label, card, type="unit", form="base", **kw):
    return {"label": label, "card": card, "type": type, "form": form, **kw}


def registry(*entries, **top):
    return VA.Registry.from_dict({"schema": VA.REGISTRY_SCHEMA, "labels": list(entries), **top})


@pytest.fixture()
def base():
    return VA.crosswalk_base(TABLE)


# ---------------------------------------------------------------------------
# the seed
# ---------------------------------------------------------------------------

def test_seed_is_the_plan_phase1_labels_in_order():
    assert REG.labels[:len(PHASE1)] == PHASE1          # append-only: new entries go after these
    assert [e.type for e in REG.entries[:19]] == [L.TYPE_UNIT] * 16 + [L.TYPE_SPELL] * 3
    assert REG.thin_tracks == 8
    hero = REG.get("BerserkerHero")
    assert (hero.form, hero.base, hero.card, hero.train_as) == (VA.FORM_HERO, "Berserker", "Berserker", None)
    assert all(e.form == VA.FORM_BASE and e.base is None for e in REG.entries[:19] if e is not hero)
    cards = {e.label: e.card for e in REG}
    assert cards["BushGoblin"] == "SuspiciousBush"
    assert cards["goblinstein_doctor"] == "Goblinstein"
    assert cards["MergeMaiden_Normal"] == cards["MergeMaiden_Mounted"] == "MergeMaiden"
    assert cards["ThreeMusketeer_Rework_Character_1"] == "ThreeMusketeers"
    assert REG.base_pin.get("labels") == 155 and REG.base_pin.get("yolo_classes") == 310


def test_every_entry_resolves_through_resolve_label():
    assert REG.validate(TABLE) == []
    assert REG.table_note(TABLE) is None               # written against this classes.json
    for e in REG:
        lab = L.resolve_label(TABLE, e.label)
        assert lab.type == e.type, e.label
        assert lab.form == e.form, e.label
        if e.type == L.TYPE_SPELL:
            assert lab.card == e.label == e.card
        else:
            assert TABLE.detector_key(lab.class_id, lab.form) == e.label
            assert e.card in TABLE.classes[lab.class_id].card_names
            assert e.card in TABLE.cards and e.card not in VA.load_event_cards()
        for side in (L.ALLY_SIDE, L.ENEMY_SIDE):         # what the detector reads back
            name, got = L.split_side(L.with_side(e.label, side))
            assert (name, got) == (e.label, side)
            assert L.resolve_label(TABLE, name).type == e.type


def test_the_forms_and_versions_mean_what_the_comment_says():
    vc = L.vision_classes_module()
    assert (vc.FORM_BASE, vc.FORM_EVO, vc.FORM_HERO) == (VA.FORM_BASE, VA.FORM_EVO, VA.FORM_HERO)
    assert tuple(vc.FORM_NAMES) == VA.FORM_NAMES
    assert all(REG.valid_for(e.label, "16.635.1") and REG.valid_for(e.label, None) for e in REG)
    for e in REG:                                      # the file round-trips through ExtraLabel
        assert VA._parse_entry(e.to_dict(), e.label) == (e, [])


# ---------------------------------------------------------------------------
# the prefix rule
# ---------------------------------------------------------------------------

def test_old_yolo_names_stay_a_prefix(base):
    assert len(base) == 155 and len(base.yolo_names()) == 310      # tonight's set on the owner's PC
    ext = REG.extend(base)
    assert ext.yolo_names()[:310] == base.yolo_names()
    assert VA.check_prefix(base.yolo_names(), ext.yolo_names()) == []
    assert len(ext) == 155 + len(REG) and len(base) == 155          # the base is not changed
    for k, e in enumerate(REG):
        assert ext.index(e.label) == 155 + k
        assert ext.yolo_index(e.label, L.ENEMY_SIDE) == 2 * (155 + k) + 1
        assert ext.status[e.label] == L.STATUS_EXTRA
    phase1 = VA.Registry(REG.entries[:19]).extend(base)
    assert len(phase1) == 174 and len(phase1.yolo_names()) == 348   # plan 2.3: 38 new classes
    assert REG.in_base(base) == []
    # the same set prepare_yolo_dataset.py --extra-label would build
    via_extra, _ = L.build_label_set(TABLE, KATACR_NAMES, extra=REG.labels)
    assert via_extra.yolo_names() == ext.yolo_names()


def test_the_registry_appends_after_whatever_the_base_holds(base):
    base.add("a-katacr-name-the-class-list-lacks", L.STATUS_UNMAPPED)   # plan 8: it can happen
    ext = REG.extend(base)
    assert ext.index("goblinstein") == 156
    assert VA.check_prefix(base.yolo_names(), ext.yolo_names()) == []
    flat, _ = L.build_label_set(TABLE, KATACR_NAMES, joint_side=False)
    ext_flat = REG.extend(flat)
    assert ext_flat.yolo_index("Vines") == ext_flat.index("Vines") == 155 + PHASE1.index("Vines")
    assert ext_flat.yolo_names()[:155] == flat.yolo_names()


def test_check_prefix_catches_a_reorder_or_a_removal(base):
    old = REG.extend(base).yolo_names()
    entries = list(REG.entries)
    swapped = VA.Registry([entries[1], entries[0], *entries[2:]]).extend(base).yolo_names()
    problems = VA.check_prefix(old, swapped)
    assert len(problems) == 1 and "YOLO class 310" in problems[0] and "would move" in problems[0]
    removed = VA.Registry(entries[:-1]).extend(base).yolo_names()
    assert "fewer than the earlier" in VA.check_prefix(old, removed)[0]
    hero = VA.ExtraLabel("BalloonHero", "Balloon", L.TYPE_UNIT, VA.FORM_HERO, base="Balloon")
    appended = VA.Registry([*entries, hero])
    assert appended.validate(TABLE) == []
    assert VA.check_prefix(old, appended.extend(base).yolo_names()) == []


# ---------------------------------------------------------------------------
# bad entries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doc, needle", [
    ({"schema": "something/1", "labels": []}, "schema"),
    ({"schema": VA.REGISTRY_SCHEMA, "labels": [entry("Ronin", "Ronin"), entry("Ronin", "Ronin")]}, "twice"),
    ({"schema": VA.REGISTRY_SCHEMA, "labels": [entry("Ronin@enemy", "Ronin")]}, "must not contain"),
    ({"schema": VA.REGISTRY_SCHEMA, "labels": [entry("Ronin", "Ronin", type="object")]}, "type must be"),
    ({"schema": VA.REGISTRY_SCHEMA, "labels": [entry("Ronin", "Ronin", form="super")]}, "form must be"),
    ({"schema": VA.REGISTRY_SCHEMA, "labels": [entry("Ronin", "Ronin", base="Knight")]}, "evo and hero entries only"),
    ({"schema": VA.REGISTRY_SCHEMA, "labels": [entry("Ronin", "Ronin", valid_from_client="new")]}, "valid_from_client"),
    ({"schema": VA.REGISTRY_SCHEMA, "labels": [entry("Ronin", "Ronin", train_as="Ronin")]}, "self"),
    ({"schema": VA.REGISTRY_SCHEMA, "labels": [], "reworks": {"Ronin": "soon"}}, "reworks"),
    ({"schema": VA.REGISTRY_SCHEMA, "labels": [], "thin_tracks": 0}, "thin_tracks"),
])
def test_malformed_registries_are_refused(doc, needle):
    with pytest.raises(VA.RegistryError, match=needle):
        VA.Registry.from_dict(doc)


@pytest.mark.parametrize("bad, needle", [
    (entry("Nope", "Ronin"), "does not resolve"),
    (entry("berserker_hero", "Berserker", form="hero", base="Berserker"), "write 'BerserkerHero'"),
    (entry("Ronin", "Knight"), "made by Ronin, not Knight"),
    (entry("Vines", "DarkMagic", type="spell"), "not card 'DarkMagic'"),
    (entry("Vines", "Vines"), "is a spell in the class table, not a unit"),
    (entry("MergeMaiden_Normal", "MergeMaiden_Normal"), "not a deck card (internal)"),
    (entry("BerserkerHero", "Berserker", form="hero", base="Knight"), "base must be 'Berserker'"),
    (entry("BerserkerHero", "Berserker", form="hero"), "needs base"),
    (entry("SkeletonArmy_EV1_General", "SkeletonArmy", form="evo", base="Skeleton"), "has no base form"),
    (entry("BerserkerHero", "Berserker"), "reads as the hero form"),
    (entry("Ronin", "Ronin", train_as="Knight"), "train_as 'Knight'"),
    (entry("Ronin", "NoSuchCard"), "not a card of the class table"),
])
def test_entries_that_contradict_classes_json_are_reported(bad, needle):
    problems = registry(bad).validate(TABLE)
    assert any(needle in p for p in problems), problems


def test_a_class_that_only_exists_evolved_is_an_evo_entry_without_base(base):
    """The evolved Skeleton Army's General and the evolved Witch's skeletons have no base form in
    classes.json: their entries keep base null, validate clean, and train as themselves (a '*'
    legend label the registry lacks would leave its images out of the mix)."""
    reg = registry(entry("SkeletonArmy_EV1_General", "SkeletonArmy", form="evo"),
                   entry("Witch_EV1_Healing_Skeleton", "Witch", form="evo"))
    assert reg.validate(TABLE) == []
    ext = reg.extend(base)
    for label in ("SkeletonArmy_EV1_General", "Witch_EV1_Healing_Skeleton"):
        assert reg.get(label).base is None
        for tracks in (0, 100):
            assert reg.train_label(label, tracks) == label
        assert reg.train_index(ext, label, L.ENEMY_SIDE, 0) == ext.yolo_index(label, L.ENEMY_SIDE)


def test_extend_refuses_a_missing_base_or_an_unpinned_base(base):
    lone_form = registry(entry("BerserkerHero", "Berserker", form="hero", base="Berserker"))
    with pytest.raises(VA.RegistryError, match="no label of the extended set"):
        lone_form.extend(base)
    with pytest.raises(VA.RegistryError, match="pinned"):
        registry(entry("Ronin", "Ronin"), base={"digest": "0" * 40}).extend(base)
    assert registry(entry("Ronin", "Ronin"), base={"digest": base.digest}).extend(base).index("Ronin") == 155
    katacr, _ = L.build_label_set(TABLE, KATACR_NAMES, mode=L.MODE_KATACR)
    with pytest.raises(VA.RegistryError, match="mode"):
        REG.extend(katacr)


# ---------------------------------------------------------------------------
# thin forms and client versions
# ---------------------------------------------------------------------------

def test_train_as_writes_the_base_index_while_a_form_is_thin(base):
    ext = REG.extend(base)
    berserker, hero = ext.index("Berserker"), ext.index("BerserkerHero")
    for side in (L.ALLY_SIDE, L.ENEMY_SIDE):
        for tracks in (0, 7):
            assert REG.train_index(ext, "BerserkerHero", side, tracks) == 2 * berserker + side
        assert REG.train_index(ext, "BerserkerHero", side, 8) == 2 * hero + side
        assert REG.train_index(ext, "Ronin", side, 0) == ext.yolo_index("Ronin", side)        # base labels
        assert REG.train_index(ext, "HogRider", side, 0) == ext.yolo_index("HogRider", side)  # not in it
    assert REG.train_label("BerserkerHero", 3) == "Berserker"
    pinned_self = registry(entry("Berserker", "Berserker"),
                           entry("BerserkerHero", "Berserker", form="hero", base="Berserker", train_as="self"))
    assert pinned_self.train_label("BerserkerHero", 0) == "BerserkerHero"
    pinned_base = registry(entry("Berserker", "Berserker"),
                           entry("BerserkerHero", "Berserker", form="hero", base="Berserker",
                                 train_as="Berserker"))
    assert pinned_base.validate(TABLE) == []
    assert pinned_base.train_label("BerserkerHero", 100) == "Berserker"
    ext2 = pinned_base.extend(base)
    assert pinned_base.train_index(ext2, "BerserkerHero", 1, 100) == ext2.yolo_index("Berserker", 1)
    # a form labelled before the registry has it (a '*' legend entry) trains as its class's base form
    assert REG.train_label("BabyDragon_EV1", 0, TABLE) == "BabyDragon" == REG.train_label("BabyDragon_EV1", 99, TABLE)
    assert REG.train_label("BabyDragon_EV1", 0) == "BabyDragon_EV1"                 # without the table: as before
    assert REG.train_label("Knight", 0, TABLE) == "Knight" and REG.train_label("Vines", 0, TABLE) == "Vines"
    assert REG.train_index(ext, "BabyDragon_EV1", 1, 0, TABLE) == ext.yolo_index("BabyDragon", 1)


def test_class_indices_need_a_pinned_base(base):
    with pytest.raises(VA.RegistryError, match="not pinned yet"):
        registry(entry("Ronin", "Ronin")).extend(base, require_pin=True)
    pinned = registry(entry("Ronin", "Ronin"), base={"digest": base.digest})
    assert pinned.extend(base, require_pin=True).index("Ronin") == 155
    assert registry(entry("Ronin", "Ronin")).extend(base).index("Ronin") == 155      # a preview needs no pin


def test_valid_from_client_and_reworks():
    reg = registry(entry("Ronin", "Ronin", valid_from_client="16.635.1"),
                   reworks={"GoblinHut_Rework": "16.500"})
    assert reg.validate(TABLE) == []
    assert reg.valid_for("Ronin", "16.635.1") and reg.valid_for("Ronin", "16.700")
    assert not reg.valid_for("Ronin", "16.402.22")
    assert not reg.valid_for("Ronin", None)                 # unknown client: left out
    assert reg.valid_for("GoblinHut_Rework", "16.635.1") and not reg.valid_for("GoblinHut_Rework", "16.402.22")
    assert reg.valid_for("HogRider", None)                  # no start version: every session
    assert VA.parse_version("16.635.0") == VA.parse_version("16.635") == (16, 635)
    assert VA.parse_version(" 16.402.22 ") == (16, 402, 22)
    for bad in ("16.x", "", "v16", "16..1"):
        with pytest.raises(ValueError):
            VA.parse_version(bad)


# ---------------------------------------------------------------------------
# files and the command
# ---------------------------------------------------------------------------

def test_extended_doc_round_trips_and_names_its_parts(tmp_path, base):
    doc = VA.extended_doc(base, REG)
    path = VA.write_json(tmp_path / "detector_labels.json", doc)
    back = VA.load_label_set(path)
    assert back.yolo_names() == REG.extend(base).yolo_names() == doc["yolo_names"]
    assert doc["base"] == {"labels": 155, "yolo_classes": 310, "digest": base.digest,
                           "classes_digest": TABLE.digest}
    assert doc["registry"]["digest"] == REG.digest and doc["registry"]["labels"] == len(REG)
    assert doc["registry"]["path"] == "data/vision/detector_extra_labels.json"
    assert doc["registry"]["forms"]["BerserkerHero"] == {"base": "Berserker", "train_as": None}
    assert VA.read_yolo_names(path) == doc["yolo_names"]
    edited = dict(doc, labels=list(reversed(doc["labels"])))
    VA.write_json(tmp_path / "edited.json", edited)
    with pytest.raises(VA.RegistryError, match="digest"):
        VA.load_label_set(tmp_path / "edited.json")
    with pytest.raises(VA.RegistryError, match="no label set"):
        VA.load_label_set(tmp_path / "missing.json")


def test_labels_command(tmp_path, capsys, base):
    base_path = VA.write_json(tmp_path / "base.json", base.to_dict())
    out = tmp_path / "extended.json"
    assert VA.main(["labels", "--base", str(base_path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    n = 155 + len(REG)
    assert f"extended {n} labels, {2 * n} YOLO classes" in text
    assert "every entry resolves" in text and "BerserkerHero" in text
    assert VA.read_yolo_names(out) == REG.extend(base).yolo_names()

    assert VA.main(["labels", "--base", str(base_path), "--against", str(out)]) == 0
    assert "stay a prefix" in capsys.readouterr().out
    entries = [e.to_dict() for e in REG]
    entries[0], entries[1] = entries[1], entries[0]
    swapped = VA.write_json(tmp_path / "swapped.json", {"schema": VA.REGISTRY_SCHEMA, "labels": entries})
    assert VA.main(["labels", "--base", str(base_path), "--registry", str(swapped),
                    "--against", str(out), "--out", str(tmp_path / "not_written.json")]) == 1
    assert "would move" in capsys.readouterr().out
    assert not (tmp_path / "not_written.json").exists()

    bad = VA.write_json(tmp_path / "bad.json", {"schema": VA.REGISTRY_SCHEMA, "labels": [entry("Nope", "Ronin")]})
    assert VA.main(["labels", "--base", str(base_path), "--registry", str(bad)]) == 1
    assert "does not resolve" in capsys.readouterr().out
    assert VA.main(["labels", "--base", str(tmp_path / "none.json")]) == 2
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert VA.main(["labels", "--base", str(base_path), "--registry", str(broken)]) == 2
    json.loads(out.read_text(encoding="utf-8"))            # the written file is plain JSON
