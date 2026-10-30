"""U08: tools/card_table_diff.py, the card-table and ledger diff (docs/SEASON_UPDATES.md 4.3,
4.4, 6.3 and 12).

The two tiny tables (old_table(), new_table()), worked by hand:

* Roster: the evolution Giant_EV1 (form_of Giant) and its unit Giant_EV1 are new. (Giant,
  evolution) is a new form, so the class is NEW_FORMS.
* Provenance ignored: the version, vintage_warning, provenance and conventions texts, the
  Skeletons' display_name and damage_source, Skeletons_EV1's overlays and the Skeleton unit's
  source_table all change, and none of them is a change.
* An echo folded: Dart Goblin's damage 59 -> 56 is in its card, its card's projectile, its
  projectile record and its unit: 4 leaves, one change with 3 echoes. Its unit's load_time_ms
  1300 -> 1400 is also in the raw CSV column LoadTime: one change with 1 echo.
* Tags: Skeletons and Skeletons_EV1 are OUR DECK (Hog 2.6 and its form). Dart Goblin is OTHER,
  or POOL with a pool weight of 5%.
* The Orphan unit no card reaches is "unattached"; thin_slice is "other".

Leaves: Skeletons 1, Skeletons_EV1 1, the Dart Goblin card 2, its projectile 1, its unit 3, the
Orphan 1 = 9. Listed after folding: 1 + 1 + 2 + 1 = 5. Percentages: 400 -> 500 is +25.00;
59 -> 56 is -5.08 (-3/59 = -5.0847); 1300 -> 1400 is +7.69 (100/1300 = 7.6923); 60 -> 70 is
+16.67. Balance size: Skeletons +25% is past the 5% OUR DECK bar, so "major".

The `engine`-marked test diffs the real tables a royalesim wheel ships (royalesim.data_dir()).
"""
from __future__ import annotations

import ast
import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


@pytest.fixture
def ctd(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    return importlib.import_module("card_table_diff")


def old_table():
    return {
        "version": "cards-1-1.1", "vintage_warning": "an old build",
        "provenance": {"source": "files of build 1"},
        "conventions": {"distance_units": "millitiles"},
        "rarities": {"Common": {"multiplier_percent_by_level": [100, 110, 121]}},
        "globals": {"MIRROR_LEVEL_OFFSET": 1},
        "thin_slice": ["Knight"],
        "cards": [
            {"name": "Skeletons", "display_name": "Skeletons", "kind": "troop", "elixir": 1,
             "summon_character": "Skeleton", "summon_radius_milli": 400,
             "damage_source": "characters.Skeleton.Damage"},
            {"name": "BlowdartGoblin", "display_name": "Dart Goblin", "kind": "troop",
             "elixir": 3, "summon_character": "BlowdartGoblin", "damage": 59,
             "projectile": {"name": "BlowdartGoblinProjectile", "damage": 59}},
            {"name": "Giant", "kind": "troop", "elixir": 5, "summon_character": "Giant",
             "hitpoints": 2000},
            {"name": "Mirror", "kind": "spell", "elixir": 1},
        ],
        "evolutions": [
            {"name": "Skeletons_EV1", "form_of": "Skeletons", "summon_character": "Skeleton_EV1",
             "summon_radius_milli": 400, "overlays": ["characters_evo.toml"]},
        ],
        "hero_forms": [
            {"name": "Giant_hero", "form_of": "Giant", "summon_character": "GiantHero",
             "ability": {"elixir": 2, "ignore_buffs": ["Rage"]}},
        ],
        "towers": [{"name": "PrincessTower", "hitpoints": 1400},
                   {"name": "KingTower", "hitpoints": 2400}],
        "units": {
            "Skeleton": {"name": "Skeleton", "hitpoints": 32, "source_table": "characters"},
            "Skeleton_EV1": {"name": "Skeleton_EV1", "hitpoints": 32},
            "BlowdartGoblin": {"name": "BlowdartGoblin", "damage": 59, "load_time_ms": 1300,
                               "raw": {"Projectile": "BlowdartGoblinProjectile",
                                       "LoadTime": 1300}},
            "Giant": {"name": "Giant", "hitpoints": 2000},
            "GiantHero": {"name": "GiantHero", "hitpoints": 2000},
            "Orphan": {"name": "Orphan", "speed": 60},
        },
        "projectiles": {"BlowdartGoblinProjectile": {"name": "BlowdartGoblinProjectile",
                                                     "damage": 59}},
        "area_effect_objects": {},
        "buffs": {"Rage": {"name": "Rage", "speed_multiplier": 135}},
    }


def cards_of(table):
    return {c["name"]: c for c in table["cards"]}


def provenance_only(table):
    """Changes the diff must ignore."""
    t = copy.deepcopy(table)
    t["version"] = "cards-1-2.1"
    t["vintage_warning"] = "a newer build"
    t["provenance"] = {"source": "files of build 2"}
    t["conventions"]["distance_units"] = "tiles / 1000"
    skel = cards_of(t)["Skeletons"]
    skel["display_name"] = "Skeleton Trio"
    skel["damage_source"] = "units.Skeleton.Damage"
    t["evolutions"][0]["overlays"] = ["characters/skeletons_ev1.toml"]
    t["units"]["Skeleton"]["source_table"] = "characters_evo"
    return t


def new_table():
    t = provenance_only(old_table())
    cards_of(t)["Skeletons"]["summon_radius_milli"] = 500
    dart = cards_of(t)["BlowdartGoblin"]
    dart["damage"] = 56
    dart["projectile"]["damage"] = 56
    t["evolutions"][0]["summon_radius_milli"] = 500
    t["evolutions"].append({"name": "Giant_EV1", "form_of": "Giant",
                            "summon_character": "Giant_EV1", "hitpoints": 2100})
    units = t["units"]
    units["BlowdartGoblin"].update(damage=56, load_time_ms=1400)
    units["BlowdartGoblin"]["raw"]["LoadTime"] = 1400
    units["Giant_EV1"] = {"name": "Giant_EV1", "hitpoints": 2100}
    units["Orphan"]["speed"] = 70
    t["projectiles"]["BlowdartGoblinProjectile"]["damage"] = 56
    t["thin_slice"] = ["Knight", "Archer"]
    return t


SKELETONS = {"path": "cards.Skeletons.summon_radius_milli", "field": "summon_radius_milli",
             "kind": "changed", "old": 400, "new": 500, "pct": 25.0, "echoes": []}
DART = [
    {"path": "cards.BlowdartGoblin.damage", "field": "damage", "kind": "changed", "old": 59,
     "new": 56, "pct": -5.08,
     "echoes": ["cards.BlowdartGoblin.projectile.damage",
                "projectiles.BlowdartGoblinProjectile.damage", "units.BlowdartGoblin.damage"]},
    {"path": "units.BlowdartGoblin.load_time_ms", "field": "load_time_ms", "kind": "changed",
     "old": 1300, "new": 1400, "pct": 7.69, "echoes": ["units.BlowdartGoblin.raw.LoadTime"]},
]
ORPHAN = {"path": "units.Orphan.speed", "field": "speed", "kind": "changed", "old": 60,
          "new": 70, "pct": 16.67}
EMPTY = {"added": [], "removed": []}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_small_helpers(ctd):
    assert ctd.path_text(("units", "Knight", "raw", "Speed")) == "units.Knight.raw.Speed"
    assert ctd.path_text(("cards", "X", "tiers", 1, "dps")) == "cards.X.tiers[1].dps"
    assert ctd.flatten({"a": {"b": 1, "provenance": "p"}, "l": [1, 2], "r": [{"x": 1}, [2]],
                        "x_provenance": "q", "$c": 3}) == \
        {("a", "b"): 1, ("l",): [1, 2], ("r", 0, "x"): 1, ("r", 1): [2]}
    assert ctd.pct(59, 56) == -5.08 and ctd.pct(400, 500) == 25.0
    assert ctd.pct(0, 5) is None and ctd.pct(True, False) is None and ctd.pct("a", "b") is None
    for parts, want in ((("load_time_ms",), "loadtime"), (("raw", "LoadTime"), "loadtime"),
                        (("range_milli",), "range"), (("raw", "Range"), "range"),
                        (("tiers", 1), "tiers"), (("ms",), "ms")):
        assert ctd.norm_field(parts) == want, parts
    assert ctd.ignored("provenance") and ctd.ignored("no_deploy_size_provenance")
    assert ctd.ignored("$comment") and ctd.ignored("damage_source")
    assert not ctd.ignored("damage") and not ctd.ignored("hit_tick_from_source")
    assert ctd.worst() == "SAME"
    assert ctd.worst("BALANCE", None, "NEW_FORMS", "ENGINE_ONLY") == "NEW_FORMS"
    with pytest.raises(ValueError, match="unknown update class"):
        ctd.worst("MAJOR")
    # list indexes sort by number, not as text
    changes = ctd.compare({"t": [{"v": k} for k in range(11)]},
                          {"t": [{"v": k + 1} for k in range(11)]})
    assert [c["field"] for c in changes] == [f"t[{k}].v" for k in range(11)]


def test_kind_of(ctd):
    assert ctd.kind_of(old_table()) == "table"
    assert ctd.kind_of(old_ledger()) == "ledger"
    for other in ([], {"cards": {}}, {"a": {"b": 1}}, {"$comment": {"x": {"value": 1}}}, None):
        assert ctd.kind_of(other) is None, other


def test_hog26_is_the_row_builders(ctd):
    tree = ast.parse((TOOLS / "rs_rows.py").read_text(encoding="utf-8"))
    node = next((n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "HOG26" for t in n.targets)), None)
    if node is None:
        pytest.skip("tools/rs_rows.py no longer assigns HOG26")
    assert isinstance(node.value, ast.Call) and node.value.func.id == "frozenset"
    assert ctd.HOG26 == frozenset(ast.literal_eval(node.value.args[0]))


def test_standard_library_only():
    names = getattr(sys, "stdlib_module_names", None)
    if names is None:
        pytest.skip("sys.stdlib_module_names needs Python 3.10")
    tree = ast.parse((TOOLS / "card_table_diff.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported - {"__future__"} <= set(names), imported - set(names)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def test_the_fixture_diff(ctd):
    t = ctd.diff_tables(old_table(), new_table())
    assert (t["old_version"], t["new_version"]) == ("cards-1-1.1", "cards-1-2.1")
    assert t["sections"] == EMPTY
    r = t["roster"]
    assert r["evolutions"] == {"added": [{"name": "Giant_EV1", "form_of": "Giant"}],
                               "removed": []}
    assert r["units"] == {"added": ["Giant_EV1"], "removed": []}
    for s in ("cards", "hero_forms", "towers", "projectiles", "area_effect_objects", "buffs"):
        assert r[s] == EMPTY, s
    assert r["forms"] == {"added": [{"base": "Giant", "form": "evolution",
                                     "names": ["Giant_EV1"]}],
                          "removed": [], "renamed": []}
    assert [c["name"] for c in t["cards"]] == ["Skeletons", "Skeletons_EV1", "BlowdartGoblin"]
    skel, evo, dart = t["cards"]
    assert skel == {"name": "Skeletons", "section": "cards", "kind": "card", "base": "Skeletons",
                    "tag": "OUR DECK", "weight": None, "changes": [SKELETONS], "max_pct": 25.0,
                    "unsized": 0, "via": []}
    assert (evo["section"], evo["kind"], evo["base"], evo["tag"]) == \
        ("evolutions", "evolution", "Skeletons", "OUR DECK")
    assert evo["changes"] == [{**SKELETONS, "path": "evolutions.Skeletons_EV1.summon_radius_milli"}]
    assert dart["tag"] == "OTHER" and dart["changes"] == DART
    assert dart["max_pct"] == 7.69 and dart["unsized"] == 0
    assert dart["via"] == ["projectiles.BlowdartGoblinProjectile", "units.BlowdartGoblin"]
    assert t["unattached"] == [ORPHAN] and t["shared"] == []
    assert t["other"] == [{"path": "thin_slice", "field": "(value)", "kind": "changed",
                           "old": ["Knight"], "new": ["Knight", "Archer"], "pct": None}]
    assert t["counts"] == {"leaves": 9, "folded": 5, "cards": 3,
                           "by_tag": {"OUR DECK": 2, "POOL": 0, "OTHER": 1}}
    v = ctd.classify(t)
    assert v == {"class": "NEW_FORMS", "balance": "major", "reasons": [
        "NEW_FORMS: new forms: Giant evolution",
        "BALANCE: 3 cards changed (2 OUR DECK, 1 OTHER); 1 unattached values; units +1 -0 "
        "(major)"]}
    json.dumps(ctd.diff(old_table(), new_table()))          # all of it is JSON


def test_provenance_is_ignored(ctd):
    old = old_table()
    cards_of(old)["Skeletons"]["level_scaling"] = {"rounding": "floor", "reading": "a",
                                                   "base_level": 1}
    old["towers"][0]["no_deploy_size_provenance"] = "buildings.csv of build 1"
    cards_of(old)["Giant"]["defaults_applied"] = ["mass"]
    new = provenance_only(old)
    cards_of(new)["Skeletons"]["level_scaling"].update(rounding="floor -- MEASURED",
                                                       reading="b")
    new["towers"][0]["no_deploy_size_provenance"] = "buildings.csv of build 2"
    cards_of(new)["Giant"]["defaults_applied"] = ["mass", "speed"]
    t = ctd.diff_tables(old, new)
    assert t["cards"] == [] and t["unattached"] == [] and t["shared"] == [] and t["other"] == []
    assert t["counts"]["leaves"] == 0
    assert all(t["roster"][s] == EMPTY for s in (*ctd.ROSTER, *ctd.MAPS))
    assert ctd.classify(t) == {"class": "SAME", "reasons": [], "balance": None}
    # the same field as a value is compared
    cards_of(new)["Skeletons"]["level_scaling"]["base_level"] = 2
    t = ctd.diff_tables(old, new)
    assert [h["field"] for c in t["cards"] for h in c["changes"]] == ["level_scaling.base_level"]


def test_pool_tags(ctd):
    t = ctd.diff_tables(old_table(), new_table(), weights={"BlowdartGoblin": 0.05})
    dart = t["cards"][-1]
    assert (dart["name"], dart["tag"], dart["weight"]) == ("BlowdartGoblin", "POOL", 0.05)
    assert t["cards"][0]["weight"] == 0.0                      # Skeletons: not in the pool
    assert t["counts"]["by_tag"] == {"OUR DECK": 2, "POOL": 1, "OTHER": 0}
    assert "(2 OUR DECK, 1 POOL)" in ctd.classify(t)["reasons"][1]
    low = ctd.diff_tables(old_table(), new_table(), weights={"BlowdartGoblin": 0.02})
    assert low["cards"][-1]["tag"] == "OTHER"
    assert ctd.diff_tables(old_table(), new_table(), weights={"BlowdartGoblin": 0.02},
                           pool_min=0.01)["cards"][-1]["tag"] == "POOL"
    # another deck: Skeletons are OTHER, Dart Goblin is ours
    mine = ctd.diff_tables(old_table(), new_table(), deck={"BlowdartGoblin"})
    assert [(c["name"], c["tag"]) for c in mine["cards"]] == [
        ("BlowdartGoblin", "OUR DECK"), ("Skeletons", "OTHER"), ("Skeletons_EV1", "OTHER")]


def test_pool_weights(ctd):
    stats = {"schema": 1, "tool": "tools/deck_stats.py", "splits": {"train": {
        "matches": 10, "seats": 20, "distinct": 4, "decks": [
            {"cards": ["a"], "names": ["BlowdartGoblin", "Giant", "Knight"], "count": 6,
             "share": 0.3, "score": 3.0, "forms": []},
            {"cards": ["b"], "names": ["Giant", "Mirror"], "count": 3, "share": 0.15,
             "score": 1.0, "forms": []},
            {"cards": ["c"], "names": None, "count": 1, "share": 0.05, "score": 0.0},
            {"cards": ["d"], "names": ["Knight", "Knight"], "count": 1, "share": 0.05,
             "score": 1.0},
        ]}}}
    w = ctd.pool_weights(stats)                         # over 0.3 + 0.15 + 0.05 = 0.5
    assert w == pytest.approx({"BlowdartGoblin": 0.6, "Giant": 0.9, "Knight": 0.7,
                               "Mirror": 0.3})
    with pytest.raises(ctd.DiffError, match="no split 'val'"):
        ctd.pool_weights(stats, "val")
    stats["splits"]["train"]["decks"] = [{"names": None, "share": 0.5}]
    with pytest.raises(ctd.DiffError, match="no deck with card names"):
        ctd.pool_weights(stats)
    assert ctd.pool_weights({"Giant": 0.04, "_comment": "hand-kept"}) == {"Giant": 0.04}
    with pytest.raises(ctd.DiffError, match="not a number"):
        ctd.pool_weights({"Giant": "lots"})
    with pytest.raises(ctd.DiffError):
        ctd.pool_weights([1, 2])


def balance(ctd, edit, weights=None):
    new = old_table()
    edit(new)
    t = ctd.diff_tables(old_table(), new, weights=weights)
    return ctd.classify(t), t


def test_balance_size(ctd):
    def giant(t):                                       # +5%, an OTHER card
        cards_of(t)["Giant"]["hitpoints"] = 2100
    v, t = balance(ctd, giant)
    assert (v["class"], v["balance"]) == ("BALANCE", "minor")
    assert [c["name"] for c in t["cards"]] == ["Giant"]     # the hero's own unit is GiantHero

    def skel(radius):
        def edit(t):
            cards_of(t)["Skeletons"]["summon_radius_milli"] = radius
        return edit
    assert balance(ctd, skel(416))[0]["balance"] == "minor"   # +4%
    assert balance(ctd, skel(424))[0]["balance"] == "major"   # +6%

    def flies(t):                                       # no size: past every bar
        cards_of(t)["Skeletons"]["flying"] = True
    v, t = balance(ctd, flies)
    assert v["balance"] == "major" and t["cards"][0]["unsized"] == 1
    assert t["cards"][0]["changes"][0]["kind"] == "added"

    def tower(t):                                       # +3.57%, every battle's
        t["towers"][0]["hitpoints"] = 1450
    v, t = balance(ctd, tower)
    assert (t["cards"][0]["name"], t["cards"][0]["tag"], v["balance"]) == \
        ("PrincessTower", "OUR DECK", "minor")

    pool = {"Giant": 0.1, "Mirror": 0.1, "BlowdartGoblin": 0.1}

    def three(t):                                       # +20%, +100%, +18.6%
        cards_of(t)["Giant"]["hitpoints"] = 2400
        cards_of(t)["Mirror"]["elixir"] = 2
        cards_of(t)["BlowdartGoblin"]["damage"] = 70

    def two(t):
        cards_of(t)["Giant"]["hitpoints"] = 2400
        cards_of(t)["Mirror"]["elixir"] = 2
    assert balance(ctd, three, pool)[0]["balance"] == "major"
    assert balance(ctd, two, pool)[0]["balance"] == "minor"
    assert balance(ctd, three)[0]["balance"] == "minor"     # no pool: all OTHER

    def shared(t):
        t["globals"]["MIRROR_LEVEL_OFFSET"] = 2
    v, t = balance(ctd, shared)
    assert (v["class"], v["balance"]) == ("BALANCE", "major")
    assert t["shared"] == [{"path": "globals.MIRROR_LEVEL_OFFSET", "field": "MIRROR_LEVEL_OFFSET",
                            "kind": "changed", "old": 1, "new": 2, "pct": 100.0}]


def test_roster_classes(ctd):
    def gone(t):
        t["cards"] = [c for c in t["cards"] if c["name"] != "Mirror"]
    v, _ = balance(ctd, gone)
    assert v["class"] == "BREAKING" and v["reasons"] == ["BREAKING: removed from the table: Mirror"]

    def tower_gone(t):
        t["towers"] = t["towers"][:1]
    assert balance(ctd, tower_gone)[0]["class"] == "BREAKING"

    def new_card(t):
        t["cards"].append({"name": "MinionGiant", "kind": "troop", "elixir": 6})
    v, t = balance(ctd, new_card)
    assert v == {"class": "NEW_CARDS", "reasons": ["NEW_CARDS: new cards: MinionGiant"],
                 "balance": None}
    assert t["roster"]["cards"] == {"added": ["MinionGiant"], "removed": []}

    def no_hero(t):
        t["hero_forms"] = []
    v, t = balance(ctd, no_hero)
    assert v["class"] == "NEW_FORMS" and v["reasons"] == ["NEW_FORMS: removed forms: Giant hero"]
    assert t["roster"]["hero_forms"]["removed"] == [{"name": "Giant_hero", "form_of": "Giant"}]

    def renamed(t):                                     # the same form under a new name
        t["hero_forms"][0]["name"] = "GiantHero_hero"
    v, t = balance(ctd, renamed)
    assert t["roster"]["forms"] == {"added": [], "removed": [],
                                    "renamed": ["GiantHero_hero", "Giant_hero"]}
    assert v == {"class": "BALANCE", "balance": "minor", "reasons": [
        "BALANCE: form records renamed: GiantHero_hero, Giant_hero (minor)"]}

    def rage(t):                                        # only an ignore_buffs list names it
        t["buffs"]["Rage"]["speed_multiplier"] = 140
    v, t = balance(ctd, rage)
    assert t["cards"] == [] and [c["path"] for c in t["unattached"]] == \
        ["buffs.Rage.speed_multiplier"]
    assert v["class"] == "BALANCE"

    def new_projectile(t):
        t["projectiles"]["Spare"] = {"name": "Spare", "damage": 1}
    v, t = balance(ctd, new_projectile)
    assert v == {"class": "BALANCE", "balance": "minor",
                 "reasons": ["BALANCE: projectiles +1 -0 (minor)"]}


def test_a_map_change_is_listed_under_every_card_that_reaches_it(ctd):
    old = old_table()
    old["cards"].append({"name": "GoblinPartyHut", "kind": "building", "elixir": 3,
                         "spawn": {"character": "BlowdartGoblin"}})
    new = copy.deepcopy(old)
    new["units"]["BlowdartGoblin"]["damage"] = 56
    t = ctd.diff_tables(old, new)
    assert [(c["name"], c["changes"][0]["path"]) for c in t["cards"]] == [
        ("BlowdartGoblin", "units.BlowdartGoblin.damage"),
        ("GoblinPartyHut", "units.BlowdartGoblin.damage")]
    assert t["counts"]["leaves"] == 1 and t["counts"]["folded"] == 2


# ---------------------------------------------------------------------------
# ledgers
# ---------------------------------------------------------------------------

def old_ledger():
    return {
        "$comment": ["the ledger"],
        "meta": {"target_game_version": "live-2026", "schema_version": 1,
                 "target_note": "prose"},
        "time": {"TICK_MS": {"value": 50, "status": "measured", "provenance": "traces",
                             "confidence": "HIGH"}},
        "combat": {"$comment": "a section note",
                   "A": {"value": "x", "candidates": ["x", "y"], "status": "candidate",
                         "provenance": "p1"},
                   "B": {"value": 1, "status": "measured"},
                   "C": {"value": True}},
        "cards": {"CARD_TABLE": {"value": "client15535",
                                 "candidates": ["client15535", "client160402017_20261019"],
                                 "status": "datamined", "provenance": "the arms"}},
    }


def new_ledger():
    led = copy.deepcopy(old_ledger())
    led["$comment"].append("more")
    led["meta"].update(schema_version=2, target_note="other prose")
    led["time"]["TICK_MS"].update(provenance="more traces", confidence="SETTLED",
                                  history=[{"value": 33}])
    combat = led["combat"]
    combat["A"]["value"] = "y"
    combat["B"]["status"] = "shipped"
    del combat["C"]
    combat["D"] = {"value": 7, "status": "measured"}
    led["cards"]["CARD_TABLE"]["candidates"].append("client160635001_20261102")
    return led


def test_ledger_diff(ctd):
    d = ctd.diff_ledgers(old_ledger(), new_ledger())
    assert d["sections"] == {
        "cards": {"added": [], "removed": [], "changed": [
            {"key": "CARD_TABLE", "listed_only": True, "fields": {"candidates": {
                "old": ["client15535", "client160402017_20261019"],
                "new": ["client15535", "client160402017_20261019", "client160635001_20261102"],
                "added": ["client160635001_20261102"], "removed": []}}}]},
        "combat": {"added": ["D"], "removed": ["C"], "changed": [
            {"key": "A", "listed_only": False, "fields": {"value": {"old": "x", "new": "y"}}},
            {"key": "B", "listed_only": True,
             "fields": {"status": {"old": "measured", "new": "shipped"}}}]},
    }
    assert d["counts"] == {"added": 1, "removed": 1, "value": 1, "listed_only": 2}
    assert d["meta"] == [{"key": "schema_version", "old": 1, "new": 2}]
    assert d["arms"] == {"old": ["client15535", "client160402017_20261019"],
                         "new": ["client15535", "client160402017_20261019",
                                 "client160635001_20261102"],
                         "added": ["client160635001_20261102"], "removed": [],
                         "default_old": "client15535", "default_new": "client15535"}
    assert ctd.classify(ledger=d) == {"class": "ENGINE_ONLY", "balance": None, "reasons": [
        "ENGINE_ONLY: ledger: 1 keys changed, 1 added, 1 removed"]}
    # candidates, status, meta and prose only: listed, not classed
    led = copy.deepcopy(old_ledger())
    led["cards"]["CARD_TABLE"]["candidates"].append("client160635001_20261102")
    led["combat"]["B"]["status"] = "shipped"
    led["meta"]["schema_version"] = 2
    led["time"]["TICK_MS"]["provenance"] = "more traces"
    d = ctd.diff_ledgers(old_ledger(), led)
    assert d["counts"] == {"added": 0, "removed": 0, "value": 0, "listed_only": 2}
    assert ctd.classify(ledger=d)["class"] == "SAME"
    same = ctd.diff_ledgers(old_ledger(), old_ledger())
    assert same["sections"] == {} and same["meta"] == []
    # the table and the ledger together: the worse class, reasons most severe first
    both = ctd.diff(old_table(), new_table(), old_ledger(), new_ledger())
    assert both["class"] == "NEW_FORMS"
    assert [r.split(":")[0] for r in both["reasons"]] == ["NEW_FORMS", "BALANCE", "ENGINE_ONLY"]


# ---------------------------------------------------------------------------
# text, markdown and the command line
# ---------------------------------------------------------------------------

def squash(lines):
    return [" ".join(line.split()) for line in lines]


def test_summary_and_markdown(ctd):
    result = ctd.diff(old_table(), new_table(), weights={"BlowdartGoblin": 0.05},
                      sources={"old": {"path": "a.json", "kind": "table",
                                       "version": "cards-1-1.1"}})
    lines = squash(ctd.summary_lines(result))
    for want in (
        "old cards-1-1.1 a.json",
        "roster: cards +0 -0, evolutions +1 -0, hero forms +0 -0, towers +0 -0, units +1 -0, "
        "projectiles +0 -0, area effect objects +0 -0, buffs +0 -0",
        "form added: Giant evolution (Giant_EV1)",
        "evolutions added: Giant_EV1",
        "values: 3 cards changed (2 OUR DECK, 1 POOL, 0 OTHER), 0 shared, 1 unattached; "
        "9 changed leaves, 5 changes listed after folding",
        "OUR DECK Skeletons summon_radius_milli 400 -> 500 (+25.00%)",
        "POOL BlowdartGoblin damage 59 -> 56 (-5.08%); load_time_ms 1300 -> 1400 (+7.69%)",
        "unattached units.Orphan.speed: speed 60 -> 70 (+16.67%)",
        "class: NEW_FORMS (major)",
        "NEW_FORMS: new forms: Giant evolution",
    ):
        assert want in lines, want
    md = ctd.render_md(result).splitlines()
    for want in (
        "## Card data",
        "Generated by `tools/card_table_diff.py`. Class **NEW_FORMS** (balance major).",
        "| evolutions | Giant_EV1 | - |",
        "| units | Giant_EV1 | - |",
        "Form added: Giant evolution (Giant_EV1).",
        "| OUR DECK | Skeletons | summon_radius_milli | 400 | 500 | +25.00 | 0 |",
        "| OUR DECK | Skeletons_EV1 | summon_radius_milli | 400 | 500 | +25.00 | 0 |",
        "| POOL | BlowdartGoblin | damage | 59 | 56 | -5.08 | 3 |",
        "| POOL | BlowdartGoblin | load_time_ms | 1300 | 1400 | +7.69 | 1 |",
        "- `units.Orphan.speed`: speed 60 -> 70 (+16.67%)",
        "- `thin_slice`: (value) [\"Knight\"] -> [\"Knight\", \"Archer\"]",
        "| old | table | cards-1-1.1 | `a.json` |",
    ):
        assert want in md, want
    led = ctd.render_md(ctd.diff(old_ledger=old_ledger(), new_ledger=new_ledger())).splitlines()
    for want in ("| combat | D | added |", "| combat | C | removed |",
                 '| combat | A | value "x" -> "y" |',
                 '| combat | B | status "measured" -> "shipped" |',
                 "| cards | CARD_TABLE | candidates added client160635001_20261102 |",
                 "- meta `schema_version`: 1 -> 2"):
        assert want in led, want


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_main(ctd, tmp_path, capsys):
    old_t, new_t = write(tmp_path / "old.json", old_table()), write(tmp_path / "new.json",
                                                                    new_table())
    old_l, new_l = write(tmp_path / "old_l.json", old_ledger()), write(tmp_path / "new_l.json",
                                                                       new_ledger())
    pool = write(tmp_path / "pool.json", {"BlowdartGoblin": 0.05})
    out_json, out_md = tmp_path / "out" / "diff.json", tmp_path / "out" / "diff.md"
    assert ctd.main([str(old_t), str(new_t), "--ledgers", str(old_l), str(new_l), "--pool",
                     str(pool), "--json", str(out_json), "--md", str(out_md)]) == 0
    out = capsys.readouterr().out
    assert "class: NEW_FORMS (major)" in out and f"-> {out_md}" in out
    result = json.loads(out_json.read_text(encoding="utf-8"))
    assert result["class"] == "NEW_FORMS" and result["schema"] == 1
    assert result["pool"] == {"cards": 1, "min": 0.03}
    assert result["sources"]["new"] == {"path": str(new_t), "kind": "table",
                                        "version": "cards-1-2.1"}
    assert result["sources"]["new_ledger"]["kind"] == "ledger"
    assert result["ledger"]["counts"]["value"] == 1
    assert result["table"]["cards"][-1]["tag"] == "POOL"
    assert "| POOL | BlowdartGoblin | damage | 59 | 56 | -5.08 | 3 |" in \
        out_md.read_text(encoding="utf-8")
    # two ledgers alone
    assert ctd.main([str(old_l), str(new_l)]) == 0
    assert "class: ENGINE_ONLY" in capsys.readouterr().out
    # inputs that are not a pair of tables or ledgers: exit 1
    bad = write(tmp_path / "list.json", [1, 2])
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    for argv, why in (([old_t, old_l], "give two of one kind"),
                      ([old_t, tmp_path / "missing.json"], "cannot read"),
                      ([old_t, tmp_path / "broken.json"], "not JSON"),
                      ([bad, bad], "neither a card table"),
                      ([old_l, new_l, "--ledgers", old_l, new_l], "ledgers already"),
                      ([old_t, new_t, "--ledgers", old_t, new_t], "not ledgers"),
                      ([old_t, new_t, "--pool", bad], "pool:")):
        assert ctd.main([str(a) for a in argv]) == 1, argv
        assert why in capsys.readouterr().err, argv
    with pytest.raises(SystemExit) as exc:
        ctd.main([str(old_t), str(new_t), "--pool-min", "2"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# the real tables a wheel ships
# ---------------------------------------------------------------------------

@pytest.mark.engine
def test_real_shipped_table_and_ledger(ctd):
    import royalesim
    data = Path(royalesim.data_dir())
    table_path, ledger_path = data / "derived" / "cards-15.535.json", data / "calibration.json"
    if not (table_path.is_file() and ledger_path.is_file()):
        pytest.skip(f"no shipped cards-15.535.json and calibration.json under {data}")
    table, ledger = ctd.load_json(table_path), ctd.load_json(ledger_path)
    assert (ctd.kind_of(table), ctd.kind_of(ledger)) == ("table", "ledger")
    same = ctd.diff(table, table, ledger, ledger)
    assert (same["class"], same["reasons"]) == ("SAME", [])
    assert same["table"]["counts"]["leaves"] == 0
    assert same["ledger"]["counts"] == {"added": 0, "removed": 0, "value": 0, "listed_only": 0}
    cards = ctd.records(table, "cards")
    assert ctd.HOG26 <= set(cards)
    assert all(base in cards for base, _ in ctd.forms(table))     # every form has its card
    arms = same["ledger"]["arms"]
    assert arms["default_new"] in arms["new"] and "client15535" in arms["new"]
    # an update made by hand: the Skeletons' summon radius, the Hog Rider's hit points in the
    # card and its unit (an echo), and a new evolution of a card that has none
    new = copy.deepcopy(table)
    new_cards = ctd.records(new, "cards")
    radius = new_cards["Skeletons"].get("summon_radius_milli") or 0
    new_cards["Skeletons"]["summon_radius_milli"] = radius + 100
    hog = new_cards["HogRider"]
    unit = ctd.records(new, "units")[hog["summon_character"]]
    assert unit["hitpoints"] == hog["hitpoints"]
    hog["hitpoints"] += 100
    unit["hitpoints"] += 100
    have = ctd.forms(table)
    base = next(c["name"] for c in table["cards"] if (c["name"], "evolution") not in have)
    new["evolutions"].append({**copy.deepcopy(table["evolutions"][0]), "name": f"{base}_EVX",
                              "form_of": base})
    got = ctd.diff(table, new)
    assert got["class"] == "NEW_FORMS"
    assert got["table"]["roster"]["forms"]["added"] == [
        {"base": base, "form": "evolution", "names": [f"{base}_EVX"]}]
    by = {c["name"]: c for c in got["table"]["cards"]}
    assert by["Skeletons"]["tag"] == "OUR DECK"
    assert [h["field"] for h in by["Skeletons"]["changes"]] == ["summon_radius_milli"]
    heads = by["HogRider"]["changes"]
    assert [h["field"] for h in heads] == ["hitpoints"]
    assert heads[0]["echoes"] == [f"units.{hog['summon_character']}.hitpoints"]
