"""Units other units create are their children, not opponent plays (the spawner gap).

tools/vision_spawn_children.py now reads RoyaleSim's newer spawner data (its module doc,
SPAWNERS and REACH): interval spawners and action graphs (the reworked Furnace's Fire Spirits
every 5 s for its whole life), the evolution records (the evolved Goblin Giant's goblins below
half HP), the hero forms' abilities and embedded records, each unit's descendants (a Skeleton
Barrel's skeletons come out of an unlabelled falling barrel) and how far a shooter's shot-borne
spawns land (the evolved Mortar's goblin, up to 11.5 tiles away). scripts/vision/events.py rule 3
then explains each such unit as its parent's child: no time window ties a child to its parent's
play, so a correct entry is enough for a spawner that lives 40 s; EventTables.child_reach lets a
shooter's shot-borne children land farther than EventConfig.child_radius_mt (per parent and child
class), but a far child is a play's unit when the rest of a play is born next to it (a Goblin Gang
8 tiles from an evolved Mortar).

Synthetic tracks in world millitiles (we are team 1 at high y, the opponent team 0), one frame
per battle tick, through the shipped data/vision/spawn_children.json and data/vision/classes.json.
The table before this change is tests/pipeline/fixtures/spawn_children/before_spawners.json.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import events as EV  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from scripts.vision.tracker import MID_Y_MT  # noqa: E402

E = VT.ENEMY_TEAM
CLASSES = ROOT / "data" / "vision" / "classes.json"
BEFORE = ROOT / "tests" / "pipeline" / "fixtures" / "spawn_children" / "before_spawners.json"
CONFIGS = {"rules": EV.EventConfig(), "live": EV.EventConfig.live()}


def before() -> dict:
    return json.loads(BEFORE.read_text(encoding="utf-8"))


def vision_tables(table: dict | None = None) -> tuple[dict, EV.EventTables]:
    """(every card of the table -> an id, its EventTables with classes.json's class map): what
    the live chain builds."""
    if not CLASSES.exists():
        pytest.skip("data/vision/classes.json (E16) is not there")
    doc = json.loads(CLASSES.read_text(encoding="utf-8"))
    table = EV.load_spawn_children() if table is None else table
    ids = {name: i for i, name in enumerate(sorted(table["kinds"]))}
    return ids, EV.EventTables.from_names(table, ids, class_of=EV.unit_class_map(doc),
                                          ignore=EV.cardless_classes(doc))


@dataclass
class Unit:
    """One enemy unit's track: seen on every tick from `born` to `gone`, moving (vx, vy)
    millitiles per tick until `stop`; hp_end is the HP fill of its last sighting (0.05: seen
    dying, so rule 1 never takes a newer track for it)."""
    uid: int
    cls: str
    born: int
    gone: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    stop: int | None = None
    hp_end: float | None = None

    def pos(self, t: int) -> tuple[float, float]:
        dt = (t if self.stop is None else min(t, self.stop)) - self.born
        return self.x + self.vx * dt, self.y + self.vy * dt

    def track(self, t: int) -> VT.Track:
        x, y = self.pos(t)
        moving = self.stop is None or t < self.stop
        return VT.Track(uid=self.uid, cls=self.cls, team=E, conf=0.9, x_mt=x, y_mt=y,
                        hp_frac=self.hp_end if t == self.gone else None,
                        first_seen_tick=self.born, last_seen_tick=t,
                        vx=self.vx if moving else 0.0, vy=self.vy if moving else 0.0)


@dataclass
class Battle:
    """A played-out battle: its deriver and each unit's (how_of, card_of) on the tick after its
    last sighting (the deriver forgets a track forget_ticks, 30 s, after it)."""
    d: EV.EventDeriver
    fate: dict

    def how(self, uid: int) -> str | None:
        return self.fate[uid][0]

    def card(self, uid: int) -> int | None:
        return self.fate[uid][1]

    def plays(self) -> list[tuple[int, int, int]]:
        """(card, tick, first uid) of every opponent play, oldest first."""
        return [(p.card, p.tick, p.uids[0]) for p in self.d.opp_plays()]


def play_out(tb: EV.EventTables, units, config: EV.EventConfig | None = None) -> Battle:
    """A battle on an empty board, one update per tick until 40 ticks after the last unit."""
    d = EV.EventDeriver(tb, config)
    d.update([], 0)
    fate: dict = {}
    for t in range(1, max(u.gone for u in units) + 41):
        d.update([u.track(t) for u in units if u.born <= t <= u.gone], t)
        for u in units:
            if t == u.gone + 1:
                fate[u.uid] = (d.how_of(u.uid), d.card_of(u.uid))
    return Battle(d, fate)


# ---------------------------------------------------------------------------
# the scenarios: (units, the uids a new entry explains, their parent's card name)
# ---------------------------------------------------------------------------
FURNACE_WAVES = range(138, 900, 100)        # its first spirit 38 ticks in, then one every 5 s


def furnace_battle() -> tuple[list[Unit], set[int]]:
    """A reworked Furnace walking up its lane for 40 s (stopping at the bridge at 700) with a
    Fire Spirit 1.5 tiles ahead of it every 5 s; it dies at 900."""
    furnace = Unit(1, "Furnace_rework", 100, 900, 4500, 6000, vy=10, stop=700, hp_end=0.05)
    spirits = [Unit(10 + i, "FireSpirits", t, t + 25, furnace.pos(t)[0], furnace.pos(t)[1] + 1500, vy=60)
               for i, t in enumerate(FURNACE_WAVES)]
    return [furnace, *spirits], {u.uid for u in spirits}


def goblin_giant_battle() -> tuple[list[Unit], set[int]]:
    """An evolved Goblin Giant with its two spear goblin riders (boxed as `spear-goblin`); below
    half HP a Goblin 2.5 tiles behind it every 2.2 s; when it dies its riders drop as spear
    goblins."""
    giant = Unit(1, "GoblinGiant", 100, 700, 14000, 7000, vy=15, stop=500, hp_end=0.05)
    riders = [Unit(2 + i, "SpearGoblin", 102, 700, 14000 + dx, 7000, vy=15, stop=500, hp_end=0.05)
              for i, dx in enumerate((-400, 400))]
    goblins = [Unit(10 + i, "Goblin", t, t + 30, 14000, giant.pos(t)[1] - 2500, vy=40, hp_end=0.05)
               for i, t in enumerate(range(300, 700, 44))]
    drops = [Unit(30 + i, "SpearGoblin", 710, 760, 14000 + dx, 13000, hp_end=0.05)
             for i, dx in enumerate((-400, 400))]
    return [giant, *riders, *goblins, *drops], {u.uid for u in (*riders, *goblins, *drops)}


def skeleton_barrel_battle() -> tuple[list[Unit], set[int]]:
    """An evolved Skeleton Barrel flying to our tower: at 75% HP it drops a barrel (7 skeletons
    12 ticks later), and where it is popped the falling barrel (no label: SkeletonContainerNew)
    bursts into 7 more past the river."""
    balloon = Unit(1, "SkeletonBalloon", 100, 400, 4000, 8000, vy=40, hp_end=0.05)
    ring = [(0, 0), (1000, 0), (-1000, 0), (0, 1000), (0, -1000), (700, 700), (-700, -700)]
    bx, by = balloon.pos(250)
    drop = [Unit(10 + i, "Skeleton", 265, 305, bx + dx, by + dy, hp_end=0.05) for i, (dx, dy) in enumerate(ring)]
    burst = [Unit(20 + i, "Skeleton", 415, 455, 4000 + dx, 20000 + dy, hp_end=0.05)
             for i, (dx, dy) in enumerate(ring)]
    return [balloon, *drop, *burst], {u.uid for u in (*drop, *burst)}


MORTAR_SHOTS = range(180, 800, 94)          # a shell every 4.7 s


def mortar_battle() -> tuple[list[Unit], set[int]]:
    """An evolved Mortar at its bridge shelling our left tower 11 tiles away: every shell drops a
    Goblin where it lands, for the Mortar's 30 s."""
    mortar = Unit(1, "Mortar", 100, 800, 3500, 14000, hp_end=0.05)
    goblins = [Unit(10 + i, "Goblin", t, t + 40, 3500, 25000, hp_end=0.05) for i, t in enumerate(MORTAR_SHOTS)]
    return [mortar, *goblins], {u.uid for u in goblins}


SCENARIOS = {"FirespiritHut": furnace_battle, "GoblinGiant": goblin_giant_battle,
             "SkeletonBalloon": skeleton_barrel_battle, "Mortar": mortar_battle}


# ---------------------------------------------------------------------------
# the regenerated table: only additions
# ---------------------------------------------------------------------------
def test_the_regenerated_table_only_adds_to_the_one_before():
    old, new = before(), EV.load_spawn_children()
    for sec in ("about", "source", "kinds", "compositions", "variants", "spell_objects"):
        assert new[sec] == old[sec], sec                  # every existing entry as it was
    for sec in ("children", "spell_children"):
        for name, kids in old[sec].items():
            assert set(kids) <= set(new[sec].get(name, ())), (sec, name)
    assert set(new) - set(old) == {"child_reach"}
    added = {(sec, p, k) for sec in ("children", "spell_children") for p, kids in new[sec].items()
             for k in kids if k not in old[sec].get(p, ())}
    assert len(added) == 93
    ch = new["children"]
    # the four gaps that made phantom plays with today's labels (and their evolved forms)
    assert ch["Furnace_rework"] == ["FireSpirits"] and ch["Furnace_EV1"] == ["FireSpirits"]
    assert ch["GoblinGiant_EV1"] == ["Goblin", "SpearGoblin", "SpearGoblinGiant", "SpearGoblinGiant_EV1"]
    assert ch["GoblinGiant"] == ["SpearGoblin", "SpearGoblinGiant"]                  # riders' drop
    assert ch["SkeletonBalloon"] == ["Skeleton", "SkeletonContainerNew"]             # descendants
    assert ch["SkeletonBalloon_EV1"] == ["Skeleton", "SkeletonBalloonEvoDummyAeO_DEATH",
                                         "SkeletonBalloonEvoDummyAeO_EXTRA"]
    # evolutions, transforms, stages, heroes and the records they carry
    assert ch["Witch_EV1"] == ["Skeleton", "Witch_EV1_Healing_Skeleton", "Witch_EV1_Skeleton_Interval"]
    assert ch["Wallbreaker_EV1"] == ["Wallbreaker_mini"]
    assert ch["SkeletonArmy_EV1_Soldier"] == ["SkeletonArmy_EV1_Spectral"]
    assert ch["Ghost_EV1"] == ["Ghost_EV1_Summon_Left", "Ghost_EV1_Summon_Right"]   # not its Clone copy
    assert ch["GoblinDemolisher"] == ["GoblinDemolisher_kamikaze_form"]
    assert ch["MovingCannon"] == ["BrokenCannon"]
    assert ch["GoblinRocketSilo"] == ["GoblinRocketSilo1", "GoblinRocketSilo2"]
    assert ch["BalloonHero"] == ["BalloonHero_Bomb", "SkeletonTrooper"]
    assert ch["MusketeerHero"] == ["MusketeerTurret"]
    assert ch["DarkPrinceHero"] == ["DarkPrinceHero_Mount", "DarkPrinceHero_Walking"]
    assert ch["IceWizardHero"] == ["IceWizardHero_IceCube"]
    assert "RoyalHog_EV1" not in ch                       # its grounded self is the same unit
    assert not any("SkeletonArmy_EV1_General" in kids for kids in ch.values())     # deployed with it
    assert new["spell_children"]["GoblinBarrel"] == ["BarrelGoblin", "Goblin", "GoblinDummy"]
    assert new["spell_children"]["BarbLog"] == ["BarbLogBarbarianHero", "Barbarian"]
    # each shooter's reach is for the children its shot spawns only (SuperWitch's Skeletons and
    # Bats keep child_radius_mt)
    assert new["child_reach"] == {"BarbarianLauncher": {"Barbarian": 13600}, "GoblinCannon": {"Goblin": 24100},
                                  "Mortar_EV1": {"Goblin": 13600}, "PrincessTower_SpookyChess": {"VoodooHog": 9400},
                                  "SuperWitch": {"VoodooHog": 7500}, "WitchMother": {"VoodooHog": 7500}}
    for parent, kids in new["child_reach"].items():
        assert set(kids) <= set(ch[parent]), parent


def test_no_child_is_of_its_parents_vision_class():
    # a child of its parent's own class would explain a second play of that card away
    # (the evolved Royal Hog's grounded self, the hero Dark Prince's walking form if merged)
    if not CLASSES.exists():
        pytest.skip("data/vision/classes.json (E16) is not there")
    cmap = EV.unit_class_map(json.loads(CLASSES.read_text(encoding="utf-8")))
    table = EV.load_spawn_children()
    loops = sorted((p, k) for p, kids in table["children"].items() for k in kids
                   if cmap.get(p, p) == cmap.get(k, k))
    assert loops == []
    _, tb = vision_tables(table)
    assert [p for p, kids in tb.children.items() if p in kids] == []


def test_the_new_entries_name_the_classes_todays_labels_box():
    ids, tb = vision_tables()
    _, tb_old = vision_tables(before())
    for parent, child in (("Furnace_rework", "FireSpirits"), ("GoblinGiant", "Goblin"),
                          ("GoblinGiant", "SpearGoblin"), ("SkeletonBalloon", "Skeleton")):
        assert tb.explains(parent, child) and not tb_old.explains(parent, child), (parent, child)
    assert tb.explains("Mortar", "Goblin") and tb_old.explains("Mortar", "Goblin")   # was too near
    assert tb.child_reach["Mortar"] == {"Goblin": 13600} and tb_old.child_reach == {}
    assert tb.child_reach["WitchMother"] == {"VoodooHog": 7500}
    assert "Furnace_rework" not in tb.child_reach and "Witch" not in tb.child_reach
    # the engine's own unit names (StateTracks) read the same entries
    raw = EV.EventTables.from_names(EV.load_spawn_children(), ids)
    assert raw.explains("Furnace_rework", "FireSpirits") and raw.explains("GoblinGiant_EV1", "Goblin")
    assert raw.child_reach["Mortar_EV1"] == {"Goblin": 13600} and "Mortar" not in raw.child_reach


# ---------------------------------------------------------------------------
# the deriver: one play for the parent, its spawns as children
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config", sorted(CONFIGS))
@pytest.mark.parametrize("card", sorted(SCENARIOS))
def test_a_spawners_units_are_its_children_for_its_whole_life(card, config):
    ids, tb = vision_tables()
    units, spawned = SCENARIOS[card]()
    d = play_out(tb, units, CONFIGS[config])
    assert d.plays() == [(ids[card], 100, 1)]                     # ONE play, the spawner's
    assert {d.how(u) for u in spawned} == {"child"}
    assert {d.card(u) for u in spawned} == {ids[card]}
    assert d.d.opp_stats.children == len(spawned) and d.d.opp_stats.unknown == {}


@pytest.mark.parametrize("card", sorted(SCENARIOS))
def test_before_the_entries_the_spawns_were_phantom_plays(card):
    ids, tb_old = vision_tables(before())
    units, spawned = SCENARIOS[card]()
    d = play_out(tb_old, units, EV.EventConfig())
    phantom = [p for p in d.plays() if p[0] != ids[card]]
    assert phantom and all(uid in spawned for _, _, uid in phantom)
    if card == "FirespiritHut":                  # each spirit a 1-elixir Fire Spirits play
        assert [p[0] for p in phantom] == [ids["FireSpirits"]] * len(FURNACE_WAVES)
    if card == "Mortar":                         # each shell's goblin a Goblins play
        assert [p[0] for p in phantom] == [ids["Goblins"]] * len(MORTAR_SHOTS)


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_a_real_fire_spirits_play_is_still_one_when_it_cannot_be_the_furnaces(config):
    ids, tb = vision_tables()
    units, spawned = furnace_battle()
    far = Unit(40, "FireSpirits", 400, 430, 14000, 8000, vy=60)               # 9.5 tiles from it
    near = Unit(41, "FireSpirits", 450, 480, 6500, 9500, vy=60)               # 2 tiles from it
    later = Unit(42, "FireSpirits", 1000, 1030, 4500, 12000, vy=60)           # where it died 5 s ago
    d = play_out(tb, [*units, far, near, later], CONFIGS[config])
    spirits = ids["FireSpirits"]
    assert d.plays() == [(ids["FirespiritHut"], 100, 1), (spirits, 400, 40), (spirits, 1000, 42)]
    # the cost, as with the Witch's skeletons: a real play within child_radius_mt (4.5 tiles)
    # of the living Furnace is taken for one of its spirits
    assert d.how(41) == "child" and d.card(41) == ids["FirespiritHut"]
    assert {d.how(u) for u in spawned} == {"child"}


def test_a_spirit_born_past_the_river_is_still_the_furnaces_child():
    # rule 3 runs before the live guards' deploy zone (and age): a Furnace across the bridge
    ids, tb = vision_tables()
    furnace = Unit(1, "Furnace_rework", 100, 400, 9000, 12000, vy=20)
    spirit = Unit(2, "FireSpirits", 338, 363, 9000, furnace.pos(338)[1] + 1500, vy=60)
    assert spirit.y - MID_Y_MT > EV.LIVE_GUARDS["deploy_zone_mt"]           # 2.3 tiles past the centre
    d = play_out(tb, [furnace, spirit], EV.EventConfig.live())
    assert d.plays() == [(ids["FirespiritHut"], 100, 1)] and d.how(2) == "child"



@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_a_goblin_beyond_the_mortars_reach_is_still_a_play(config):
    ids, tb = vision_tables()
    units, spawned = mortar_battle()
    stray = Unit(40, "Goblin", 600, 640, 12000, 26500, hp_end=0.05)         # 15.1 tiles from it
    d = play_out(tb, [*units, stray], CONFIGS[config])
    assert d.plays() == [(ids["Mortar"], 100, 1), (ids["Goblins"], 600, 40)]
    assert {d.how(u) for u in spawned} == {"child"}


def test_child_reach_widens_only_its_parent_and_child_class():
    tb = EV.EventTables({1: {"P": 1}, 2: {"Q": 1}, 3: {"C": 1}, 4: {"D": 1}},
                        children={"P": {"C", "D"}, "Q": {"C"}}, child_reach={"P": {"C": 8000}})
    assert tb.child_reach == {"P": {"C": 8000.0}} and tb.reach_of("P", "C") == 8000.0
    assert tb.reach_of("P", "D") == 0.0 and tb.reach_of("Q", "C") == 0.0

    def child_at(parent: str, dist: float, kid: str = "C") -> str | None:
        d = EV.EventDeriver(tb)
        d.update([], 0)
        d.update([VT.Track(1, parent, E, 0.9, 9000.0, 5000.0, first_seen_tick=100, last_seen_tick=101)], 101)
        d.update([VT.Track(1, parent, E, 0.9, 9000.0, 5000.0, first_seen_tick=100, last_seen_tick=200),
                  VT.Track(2, kid, E, 0.9, 9000.0, 5000.0 + dist, first_seen_tick=200, last_seen_tick=200)], 200)
        return d.how_of(2)

    assert child_at("P", 7900) == "child" and child_at("P", 8100) == "play"
    assert child_at("P", 6000, "D") == "play" and child_at("P", 4400, "D") == "child"   # its other child
    assert child_at("Q", 4400) == "child" and child_at("Q", 4600) == "play"      # child_radius_mt
    assert child_at("P", 4400) == "child"                                        # never narrower
    # the engine tables keep it (for_engine builds from_spawn_primary on from_names)
    raw = {"card_names": ["Mortar"], "primary": {"0": "Mortar_EV1"}, "all": {"0": ["Mortar_EV1"]}}
    spawn = {"kinds": {"Mortar": "building"}, "compositions": {"Mortar": {"Mortar_EV1": 1}},
             "children": {"Mortar_EV1": ["Goblin"]},
             "child_reach": {"Mortar_EV1": {"Goblin": 13600, "Gone": 9000}, "Gone": {"Goblin": 9000}}}
    eng = EV.EventTables.for_engine(raw, spawn_children=spawn)
    assert eng.child_reach == {"Mortar_EV1": {"Goblin": 13600.0}} and eng.explains("Mortar_EV1", "Goblin")
    for bad in ({"C": -1}, {"C": float("nan")}, {"C": "far"}, {"C": True}, {"C": None}, 8000):
        with pytest.raises(ValueError):
            EV.EventTables({1: {"P": 1}}, child_reach={"P": bad})
    with pytest.raises(ValueError):              # the format before per-child reach
        EV.EventTables.from_names({"children": {"P": ["C"]}, "child_reach": {"P": 8000}}, {})


def goblin_gang(uid: int, born: int, x: float, y: float, *, spears_born: int | None = None,
                goblins: int = 3) -> list[Unit]:
    """A Goblin Gang walking to the bridge: `goblins` Goblins (3 in a play) in a row at y and its
    3 Spear Goblins a tile behind them (born at spears_born, default born)."""
    sb = born if spears_born is None else spears_born
    gob = [Unit(uid + i, "Goblin", born, born + 60, x + dx, y, vy=40, hp_end=0.05)
           for i, dx in enumerate((-600, 0, 600)[:goblins])]
    spear = [Unit(uid + 3 + i, "SpearGoblin", sb, sb + 60, x + dx, y - 1000, vy=40, hp_end=0.05)
             for i, dx in enumerate((-600, 0, 600))]
    return [*gob, *spear]


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_a_goblin_gang_within_the_mortars_reach_is_a_goblin_gang(config):
    # 8 tiles from a living evolved Mortar (within its 13.6-tile reach for its shell's Goblin),
    # between two shells: the gang's goblins are not taken for the shell's, so the play is the
    # 3-elixir Goblin Gang, not 2-elixir Spear Goblins (its spear goblins alone)
    ids, tb = vision_tables()
    units, spawned = mortar_battle()
    gang = goblin_gang(50, 330, 11500, 13000)
    assert all(abs(u.born - t) >= EV.PLAY_WINDOW for u in gang for t in MORTAR_SHOTS)
    assert all(EV.EventConfig().child_radius_mt < math.dist((3500, 14000), (u.x, u.y)) < 13600
               for u in gang if u.cls == "Goblin")
    d = play_out(tb, [*units, *gang], CONFIGS[config])
    assert d.plays() == [(ids["Mortar"], 100, 1), (ids["GoblinGang"], 330, 50)]
    assert [d.how(u.uid) for u in gang] == ["play"] + ["merged"] * 5
    assert {d.card(u.uid) for u in gang} == {ids["GoblinGang"]}
    assert {d.how(u) for u in spawned} == {"child"}


def test_with_the_age_guard_a_far_goblin_waits_for_the_rest_of_its_play():
    # one goblin of a gang seen (two missed), its spear goblins 5 ticks later: the live config
    # holds the goblin's far parent until it is 10 ticks old, by when the spear goblins are seen
    ids, tb = vision_tables()
    units, spawned = mortar_battle()
    gang = goblin_gang(50, 330, 11500, 13000, spears_born=335, goblins=1)
    d = play_out(tb, [*units, *gang], EV.EventConfig.live())
    assert d.plays() == [(ids["Mortar"], 100, 1), (ids["GoblinGang"], 330, 50)]
    assert {d.how(u) for u in spawned} == {"child"}
    # the rules config (no age guard) decides at the first sighting: the cost in the module doc
    d = play_out(tb, [*units, *gang], EV.EventConfig())
    assert d.how(50) == "child" and d.plays()[1:] == [(ids["SpearGoblins"], 335, 53)]


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_the_mother_witchs_hogs_are_her_children_when_several_come_at_once(config):
    # no card puts a VoodooHog down, so there is no play to lose: a far one is always a child
    ids, tb = vision_tables()
    assert not tb.unit_cards("VoodooHog") and tb.reach_of("WitchMother", "VoodooHog") == 7500
    witch = Unit(1, "WitchMother", 100, 600, 9000, 10000, hp_end=0.05)
    hogs = [Unit(10 + i, "VoodooHog", 300, 340, 9000 + dx, 16000, vy=80, hp_end=0.05)
            for i, dx in enumerate((-800, 0, 800))]
    assert all(4500 < math.dist((9000, 10000), (h.x, h.y)) < 7500 for h in hogs)
    d = play_out(tb, [witch, *hogs], CONFIGS[config])
    assert d.plays() == [(ids["WitchMother"], 100, 1)]
    assert {d.how(h.uid) for h in hogs} == {"child"} and {d.card(h.uid) for h in hogs} == {ids["WitchMother"]}


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_events_change_only_where_a_new_child_explains_a_track(config):
    # one battle of every scenario (one after another) between plays the old table already
    # explained: every other track keeps its rule and card, and the plays are the old ones but
    # for those of the tracks the new entries explain
    ids, tb = vision_tables()
    _, tb_old = vision_tables(before())
    units = [Unit(1, "HogRider", 20, 160, 14000, 9000, vy=50, stop=120, hp_end=0.05),
             Unit(2, "Musketeer", 30, 3000, 9000, 3000),
             Unit(3, "Witch", 40, 300, 12000, 5000, hp_end=0.05),
             *[Unit(4 + i, "Skeleton", 80, 110, 12000 + dx, 5000 + dy, hp_end=0.05)
               for i, (dx, dy) in enumerate([(1500, 0), (-1500, 0), (0, 1500), (0, -1500)])],
             Unit(10, "Golem", 50, 900, 15000, 3000, vy=10, stop=800, hp_end=0.05),
             *[Unit(11 + i, "Golemite", 910, 960, 15000 + dx, 10500) for i, dx in enumerate((-600, 600))]]
    newly: set[int] = set()
    start = 1000
    for k, (_, scenario) in enumerate(sorted(SCENARIOS.items())):
        part, spawned = scenario()
        base = 1000 * (k + 1)
        for u in part:
            units.append(replace(u, uid=base + u.uid, born=u.born + start, gone=u.gone + start,
                                 stop=None if u.stop is None else u.stop + start))
        newly |= {base + uid for uid in spawned}
        start += 1100
    d_old = play_out(tb_old, units, CONFIGS[config])
    d_new = play_out(tb, units, CONFIGS[config])
    for u in units:
        if u.uid in newly:
            assert d_new.how(u.uid) == "child", u
        else:
            assert (d_new.how(u.uid), d_new.card(u.uid)) == (d_old.how(u.uid), d_old.card(u.uid)), u
    assert d_new.plays() == [p for p in d_old.plays() if p[2] not in newly]
    assert len(d_new.plays()) == 4 + len(SCENARIOS) and len(d_old.plays()) > len(d_new.plays())


# ---------------------------------------------------------------------------
# the generator, on a table in RoyaleSim's layout just big enough for it
# ---------------------------------------------------------------------------
def _rec(name: str, table: str = "characters", **fields) -> dict:
    return {"name": name, "source_table": table, **fields}


def spawner_table(version: str = "cards-new.1", *, mortar_range: int = 11500) -> dict:
    units = {
        "Furnace_rework": {"interval_spawner": {"character": "FireSpirits", "interval_ms": 5000},
                           "action_graph": {"roots": {"OnStartingAction": "spawn"},
                                            "spawns": ["CharacterType:FireSpirits"]}},
        "Furnace_EV1": {"action_graph": {"spawns": ["ProjectileType:SpiritShot"]}},
        "FireSpirits": {}, "Goblin": {}, "Skeleton": {}, "SpearGoblin": {}, "BarrelGoblin": {},
        "GoblinDummy": {},
        "GoblinDemolisher": {"transform_at_hp": {"into": "GoblinDemolisher_kamikaze_form", "pct": 50,
                                                 "noop_spawns": ["AreaEffectType:CancelTaunt"]}},
        "GoblinDemolisher_kamikaze_form": {},
        "GoblinRocketSilo": {"stage_schedule": [{"at_ms": 6200, "into": "GoblinRocketSilo1"},
                                                {"at_ms": 12000, "into": "GoblinRocketSilo2"}]},
        "GoblinRocketSilo1": {}, "GoblinRocketSilo2": {},
        "GoblinHut_Rework": {"life_state_spawner": {"character": "SpearGoblin_Dummy"}}, "SpearGoblin_Dummy": {},
        "SkeletonBalloon": {"death_spawn": {"character": "SkeletonContainerNew"}},
        "SkeletonContainerNew": {"death_spawn": {"character": "Skeleton"}},
        "GoblinGiant": {"spawner": {"character": "SpearGoblinGiant"}},
        "SpearGoblinGiant": {"death_spawn": {"character": "SpearGoblin"}},
        "GoblinGiant_EV1": {}, "SkeletonBalloon_EV1": {}, "RoyalHog_EV1": {},
        "Ghost_EV1": {}, "Ghost_EV1_Summon_Left": {}, "Ghost_EV1_Summon_Right": {}, "Ghost_EV1_Clone": {},
        "SkeletonArmy_EV1_Soldier": {}, "SkeletonArmy_EV1_Spectral": {}, "SkeletonArmy_EV1_General": {},
        "Mortar_EV1": {"range_milli": mortar_range, "collision_radius_milli": 600, "projectile": "MortarShell"},
        "Zoner": {"action_graph": {"spawns": ["AreaEffectType:Zone"]}},
        "Buffer": {"action_graph": {"spawns": ["BuffType:Goblin"]}},       # a buff, not the unit
    }
    evolutions = [
        {"name": "GoblinGiant_EV1", "form_of": "GoblinGiant", "summon_character": "GoblinGiant_EV1",
         "spawner": {"character": "SpearGoblinGiant", "attach": True},
         "interval_spawner": {"character": "Goblin", "interval_ms": 2200, "health_pct": 50}},
        {"name": "Ghost_EV1", "form_of": "Ghost", "summon_character": "Ghost_EV1", "cloned_version": "Ghost_EV1_Clone",
         "evo_ghost": {"left": "Ghost_EV1_Summon_Left", "right": "Ghost_EV1_Summon_Right", "clone": "Ghost_EV1_Clone"}},
        {"name": "SkeletonArmy_EV1", "form_of": "SkeletonArmy", "summon_character": "SkeletonArmy_EV1_Soldier",
         "action_graph": {"spawns": ["CharacterType:SkeletonArmy_EV1_Spectral"]},
         "evo_army": {"general": "SkeletonArmy_EV1_General", "spectral": "SkeletonArmy_EV1_Spectral"}},
        {"name": "SkeletonBalloon_EV1", "form_of": "SkeletonBalloon", "summon_character": "SkeletonBalloon_EV1",
         "evo_barrel": {"at_hp_pct": 75,
                        "extra": _rec("DropBarrel", "buildings", death_spawn={"character": "Skeleton"})}},
        {"name": "RoyalHogs_EV1", "form_of": "RoyalHogs", "summon_character": "RoyalHog_EV1",
         "evo_fall": {"grounded": _rec("RoyalHog_EV1_Grounded")}},
        {"name": "GoblinBarrel_EV1", "form_of": "GoblinBarrel", "summon_character": None, "kind": "spell",
         "projectile": {"name": "GoblinBarrelSpell_EV1"},
         "evo_mirror": {"projectile": "GoblinBarrelSpell_EV1_Decoy", "unit": "GoblinDummy"}},
    ]
    hero_forms = [
        {"name": "Balloon_hero", "summon_character": "BalloonHero", "kind": "troop",
         "death_spawn": {"character": "BalloonHero_Bomb"},
         "ability": {"effect": {"kind": "throw", "unit": "SkeletonTrooper"}},
         "tables": {"units": {"BalloonHero": _rec("BalloonHero"), "BalloonHero_Bomb": _rec("BalloonHero_Bomb"),
                              "SkeletonTrooper": _rec("SkeletonTrooper")}}},
        {"name": "DarkPrince_hero", "summon_character": "DarkPrinceHero", "kind": "troop",
         "ability": {"effect": {"unit": "DarkPrinceHero_Walking", "mount": "DarkPrinceHero_Mount",
                                "steps": [{"do": "form", "form": "DarkPrinceHero"}]}},
         "tables": {"units": {"DarkPrinceHero_Walking": _rec("DarkPrinceHero_Walking"),
                              "DarkPrinceHero_Mount": _rec("DarkPrinceHero_Mount")}}},
        {"name": "IceWizard_hero", "summon_character": "IceWizardHero", "kind": "troop",
         "ability": {"effect": {"cube": {"unit": "IceWizardHero_IceCube"}}},
         "tables": {"units": {"IceWizardHero_IceCube": _rec("IceWizardHero_IceCube", "buildings")}}},
        {"name": "BarbLog_hero", "summon_character": None, "kind": "spell", "form_of": "BarbLog",
         "projectile": {"name": "HeroLog", "spawn_projectile": {"name": "HeroLogRolling",
                                                                "spawn_character": "BarbLogBarbarianHero"}},
         "ability": {"effect": {"kind": "reroll", "unit": "BarbLogBarbarianHero"}},
         "tables": {"units": {"BarbLogBarbarianHero": _rec("BarbLogBarbarianHero")}}},
    ]
    cards = [
        {"name": "FirespiritHut", "kind": "troop", "elixir": 4, "summon_character": "Furnace_rework"},
        {"name": "GoblinGiant", "kind": "troop", "elixir": 6, "summon_character": "GoblinGiant"},
        {"name": "Mortar", "kind": "building", "elixir": 4, "summon_character": "Mortar"},
        {"name": "GoblinBarrel", "kind": "spell", "elixir": 3, "projectile": {"name": "GoblinBarrelSpell"}},
        {"name": "BarbLog", "kind": "spell", "elixir": 2, "projectile": {"name": "BarbLogProjectile"}},
    ]
    projectiles = {"SpiritShot": {"spawn_character": "FireSpirits"}, "MortarShell": {"spawn_character": "Goblin"},
                   "GoblinBarrelSpell": {"spawn_character": "BarrelGoblin"},
                   "GoblinBarrelSpell_EV1": {"spawn_character": "BarrelGoblin"},
                   "GoblinBarrelSpell_EV1_Decoy": {"spawn_character": "GoblinDummy"},
                   "Zone": {"spawn_character": "Skeleton"}}          # an area of the same name below
    return {"version": version, "cards": cards, "units": units, "evolutions": evolutions,
            "hero_forms": hero_forms, "projectiles": projectiles,
            "area_effect_objects": {"Zone": {"spawn_character": "Goblin"}}}


def test_the_generator_reads_the_newer_spawner_data():
    VSC = pytest.importorskip("tools.vision_spawn_children")
    doc = VSC.build(spawner_table(), files=["new.json"])
    assert doc["children"] == {
        "Furnace_rework": ["FireSpirits"],                                       # interval spawner
        "Furnace_EV1": ["FireSpirits"],                                          # ProjectileType:
        "GoblinDemolisher": ["GoblinDemolisher_kamikaze_form"],                  # transform_at_hp
        "GoblinRocketSilo": ["GoblinRocketSilo1", "GoblinRocketSilo2"],          # stage_schedule
        "GoblinHut_Rework": ["SpearGoblin_Dummy"],                               # life_state_spawner
        "SkeletonBalloon": ["Skeleton", "SkeletonContainerNew"],                 # + descendants
        "SkeletonContainerNew": ["Skeleton"],
        "GoblinGiant": ["SpearGoblin", "SpearGoblinGiant"],
        "SpearGoblinGiant": ["SpearGoblin"],
        "GoblinGiant_EV1": ["Goblin", "SpearGoblin", "SpearGoblinGiant"],        # its evolution record
        "Ghost_EV1": ["Ghost_EV1_Summon_Left", "Ghost_EV1_Summon_Right"],        # not clone
        "SkeletonArmy_EV1_Soldier": ["SkeletonArmy_EV1_Spectral"],               # not general
        "SkeletonBalloon_EV1": ["DropBarrel", "Skeleton"],                       # an embedded record
        "DropBarrel": ["Skeleton"],
        "Mortar_EV1": ["Goblin"],
        "Zoner": ["Goblin"],                                                     # AreaEffectType: only
        "BalloonHero": ["BalloonHero_Bomb", "SkeletonTrooper"],                  # hero record + ability
        "DarkPrinceHero": ["DarkPrinceHero_Mount", "DarkPrinceHero_Walking"],
        "IceWizardHero": ["IceWizardHero_IceCube"],
    }                                                     # RoyalHog_EV1 (grounded), Buffer: none
    assert doc["spell_children"] == {"GoblinBarrel": ["BarrelGoblin", "Goblin", "GoblinDummy"],
                                     "BarbLog": ["BarbLogBarbarianHero"]}
    assert doc["spell_objects"] == {"GoblinBarrel": ["GoblinBarrelSpell"], "BarbLog": []}   # base card's only
    assert doc["child_reach"] == {"Mortar_EV1": {"Goblin": 11500 + 600 + VSC.REACH_SLACK_MT}}
    assert doc["compositions"]["FirespiritHut"] == {"Furnace_rework": 1}


def test_the_generator_still_only_adds_from_an_older_table():
    VSC = pytest.importorskip("tools.vision_spawn_children")
    old = spawner_table("cards-old.1", mortar_range=9000)
    old["units"]["OldOnly"] = {"interval_spawner": {"character": "Skeleton"}}
    doc = VSC.build([spawner_table(), old], files=["new.json", "old.json"])
    assert doc["children"]["OldOnly"] == ["Skeleton"]                         # kept from the old
    assert doc["child_reach"] == {"Mortar_EV1": {"Goblin": 13600}}            # the first table's
    assert VSC.build(old)["child_reach"] == {"Mortar_EV1": {"Goblin": 9000 + 600 + 1500}}
    assert VSC.build({"version": "v.1", "cards": [], "units": {}, "projectiles": {},
                      "area_effect_objects": {}})["child_reach"] == {}


def test_the_generator_closes_descendants_and_reach_over_the_merged_tables():
    VSC = pytest.importorskip("tools.vision_spawn_children")

    def table(version: str, units: dict, projectiles: dict | None = None) -> dict:
        return {"version": version, "cards": [], "units": units, "projectiles": projectiles or {},
                "area_effect_objects": {}}

    first = table("a.1", {"X": {"spawner": {"character": "A"}}, "A": {},
                          "Gun": {"range_milli": 5000, "projectile": "Shell"}, "Pod": {}},
                  {"Shell": {"spawn_character": "Pod"}})
    second = table("b.1", {"A": {"death_spawn": {"character": "B"}}, "B": {},
                           "Pod": {"death_spawn": {"character": "BarrelGoblin"}}, "BarrelGoblin": {}})
    doc = VSC.build([first, second], files=["a.json", "b.json"])
    assert doc["children"]["X"] == ["A", "B"]                 # X -> A in one table, A -> B in the other
    assert doc["children"]["Gun"] == ["BarrelGoblin", "Goblin", "Pod"]      # + the look-alike
    # the shell's pod and what it creates in turn (and its look-alike) land where the shell does
    assert doc["child_reach"] == {"Gun": {"BarrelGoblin": 6500, "Goblin": 6500, "Pod": 6500}}
    assert VSC.build(first)["child_reach"] == {"Gun": {"Pod": 6500}}
