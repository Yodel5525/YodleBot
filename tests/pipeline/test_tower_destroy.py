"""A princess tower is destroyed only on its label's absence, and only that opens a pocket (skin
audit fix list P0-2, P1-1).

The owner's 2026-10-31 friendly destroyed both enemy towers at tick ~266 while they stood: the
chicken skin moved their labels off the HUD's boxes, the boxes read the chicken's comb (a 0.22
bar), then nothing, and "nothing read" became a zero. Any zero opened a pocket for legality and
the event rules. Now (scripts/vision/hud.py TowerHpTracker label control, tower_calib's absence
clock): with calibration a tower falls only once its calibrated label (badge and bar) has been
absent at the lock for 2.5 s of uncovered, non-overlay frames, from <= 0.75, with no detector
label box there; a body box does not stop it (ruins). destroyed_by "label" is the only kind
Hud.towers_destroyed holds, and legality (tools/public_state.py -> rs_decide.destroyed_towers)
and the pockets (events.EventDeriver.note_towers) act on that set alone.

Every frame is drawn by tests/pipeline/skin_synth.py at the owner's 747x1328; no real frame is
used. The detector is its boxes (skin_synth.det).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rs_decide as RD  # noqa: E402
from scripts import vision_play as VP  # noqa: E402
from scripts.vision import events as EV  # noqa: E402
from scripts.vision import hud as H  # noqa: E402
from scripts.vision import tower_labels as TL  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from tests.pipeline import skin_synth as S  # noqa: E402
from tests.pipeline import test_public_state as TPS  # noqa: E402

SIZE = S.OWNER_SIZE
DT = 0.125
PRINCESS = TL.PRINCESS_KEYS
CHICKEN = {k: S.Label(dv=S.SKIN_DV if k in S.ENEMY else 0.0) for k in PRINCESS}
HOG = TPS.CID["HogRider"]
FULL = (VT.TowerHp(3052, 3052), VT.TowerHp(None, 3052))


@lru_cache(maxsize=None)
def frame(floor: str = "sand", *, labels: tuple = (), decorations: tuple = (), flyers: tuple = ()) -> np.ndarray:
    """A cached skin_synth battle frame: CHICKEN labels with `labels` ((key, Label | None) pairs)
    over them."""
    lab = {**CHICKEN, **dict(labels)}
    img = S.battle_frame(floor, labels=lab, decorations=list(decorations), flyers=list(flyers), size=SIZE)
    img.setflags(write=False)
    return img


@lru_cache(maxsize=1)
def store():
    return S.digit_store(SIZE)


def reader(**kw) -> H.HudReader:
    return H.HudReader(store(), {}, regions=S.REGIONS, towers_wait_for_clock=False, **kw)


def read(r: H.HudReader, img, t, dets=None) -> VT.Hud:
    hud = r.read(img, t, towers=False)
    return r.read_towers(img, t, hud, None if dets is None else r.tower_evidence(dets, SIZE))


def label_dv(key: str) -> float:
    return S.SKIN_DV if key in S.ENEMY else 0.0


def dets(*, labels=PRINCESS, bodies=PRINCESS, body_cls="PrincessTower", extra=()) -> list:
    """The detector's boxes: tower-bar labels at the CHICKEN positions of `labels`, bodies (a ruin
    is boxed as a tower, finding 6) at `bodies`, then `extra`."""
    out = [S.label_det(k, label_dv(k)) for k in labels]
    out += [S.body_det(k, body_cls) for k in bodies]
    return out + list(extra)


def state(hud: VT.Hud, tick: int = 1500) -> dict:
    """tools/public_state's state of a battle frame with this HUD."""
    vf = VT.VisionFrame(t_capture=1.0, tick=tick, hud=hud, screen=VT.SCREEN_BATTLE, frame_size=SIZE)
    return TPS.builder().state(vf)


def pocket_legal(hud: VT.Hud) -> bool:
    """A Hog Rider of ours (team 1) at world tile (3, 12): the enemy's left pocket."""
    c = RD.PureLegality(TPS.CATALOGUE).bind(state(hud), 1)
    return bool(c(0, HOG, 3500 * 18, 12500 * 18))


def deriver() -> EV.EventDeriver:
    return EV.EventDeriver(EV.EventTables({8: {"Knight": 1}}), EV.EventConfig.live(), own_deck=range(8))


def feed_deriver(d: EV.EventDeriver, hud: VT.Hud, tick: int) -> None:
    """What LivePerception does with a frame's HUD (vision_play.confirmed_towers)."""
    d.update([], tick, towers=hud.towers, **VP.confirmed_towers(hud))


# ---------------------------------------------------------------------------
# the 10-31 regression
# ---------------------------------------------------------------------------
def test_the_10_31_friendly_destroys_nothing_when_calibration_fails():
    """120 s of chicken-like frames (enemy labels 36 px up, the red comb in the fixed bar band,
    a sand floor), calibration forced to fail (a one-frame window), the detector boxing the
    enemy bodies as Cannoneer with no label boxes (as the 10-31 model did): the enemy towers
    are never read, never destroyed; legality and the pockets stay shut."""
    img = frame(decorations=tuple((k, "comb", 0.0, 0.0) for k in S.ENEMY))
    r = reader()
    r.calib.window_s, r.calib.window_frames = 0.0, 1
    d = deriver()
    boxes = [S.body_det(k, "Cannoneer") for k in S.ENEMY] + dets(labels=S.OWN, bodies=S.OWN)
    huds = []
    for i in range(240):                                       # 2 frames a second
        hud = read(r, img, i * 0.5, boxes)
        feed_deriver(d, hud, 100 + 10 * i)
        huds.append(hud)
    for k in S.ENEMY:
        assert r.calib.towers[k].status == "expired" and not r.calib.calibrated(k)
        assert all(h.towers[k] in FULL for h in huds), k
        assert r.report.tower_bars[k] is None and r.report.tower_digits[k].hp is None
    assert all(h.crowns == (0, 0) and h.towers_destroyed == () for h in huds)
    assert r.towers.destroyed_by == {} and d.fallen_towers == set()
    assert RD.destroyed_towers(state(huds[-1])) == ((False, False, False), (False, False, False))
    assert not pocket_legal(huds[-1])


def test_the_old_rules_still_destroy_but_nothing_opens():
    """The same battle read the old way (calibrate=False, no placement: the fixed boxes): the comb
    reads 0.25 for 4 s, then nothing, and the enemy towers come out destroyed (by the detector's
    absence: a body box alone no longer holds them, P1-1). Without label evidence those zeros are
    unconfirmed: legality reads them full, no pocket opens, the status line says so."""
    comb = frame(decorations=tuple((k, "comb", 0.0, 0.0) for k in S.ENEMY))
    plain = frame()
    r = reader(calibrate=False, place_labels=False, search_labels=False)
    d = deriver()
    boxes = [S.body_det(k, "Cannoneer") for k in S.ENEMY] + dets(labels=S.OWN, bodies=S.OWN)
    huds = []
    for i in range(120):
        hud = read(r, comb if i < 32 else plain, i * DT, boxes)
        feed_deriver(d, hud, 100 + 2 * i)
        huds.append(hud)
    last = huds[-1]
    assert all(last.towers[k] == VT.TowerHp(0, 3052) for k in S.ENEMY) and last.crowns == (2, 0)
    assert r.towers.destroyed_by == {"enemy_left": "detector", "enemy_right": "detector"}
    assert last.towers_destroyed == () and d.fallen_towers == set()
    assert RD.destroyed_towers(state(last)) == ((False, False, False), (False, False, False))
    assert not pocket_legal(last)
    notes = state(last)["perception"]["notes"]
    assert "enemy_left 0 unconfirmed: full" in notes
    assert VP.towers_text(last).endswith("crowns 2-0; enemy_left 0 (unconfirmed); enemy_right 0 (unconfirmed)")


def test_a_hud_without_a_confirmed_set_counts_every_zero_as_before():
    """Another HUD reader or an older recording (Hud.towers_destroyed None): every 0 counts."""
    hud = VT.Hud(towers={"enemy_left": VT.TowerHp(0, 3052), "own_right": VT.TowerHp(0, 3052)})
    assert RD.destroyed_towers(state(hud)) == ((False, True, False), (False, False, True))
    d = deriver()
    feed_deriver(d, hud, 200)
    assert d.fallen_towers == {"enemy_left", "own_right"}
    assert VT.Hud.from_dict(hud.to_dict()).towers_destroyed is None and "towers_destroyed" not in hud.to_dict()
    told = VT.Hud(towers=dict(hud.towers), towers_destroyed=("own_right",))
    assert VT.Hud.from_dict(told.to_dict()) == told and told.destroyed("own_right") and not told.destroyed("enemy_left")
    assert RD.destroyed_towers(state(told)) == ((False, False, False), (False, False, True))
    d = deriver()
    feed_deriver(d, told, 200)
    assert d.fallen_towers == {"own_right"}


# ---------------------------------------------------------------------------
# a real fall
# ---------------------------------------------------------------------------
def fall_run(r: H.HudReader, *, keys=("enemy_left", "own_left"), boxes_after=None, frames_after: int = 48):
    """Full for 1.25 s (the lock), `keys` at 0.3 for 1.5 s, then their labels gone (ruins), the
    detector boxing every body (ruins as towers) and the standing towers' labels. Returns
    (huds, t of the first ruin frame, i of it)."""
    full, low = frame(), frame(labels=tuple((k, S.Label(dv=label_dv(k), frac=0.3)) for k in keys))
    ruin = frame(labels=tuple((k, None) for k in keys))
    standing = [k for k in PRINCESS if k not in keys]
    before = dets()
    after = boxes_after if boxes_after is not None else dets(labels=standing)
    seq = [(full, before)] * 10 + [(low, before)] * 12 + [(ruin, after)] * frames_after
    huds = [read(r, img, i * DT, b) for i, (img, b) in enumerate(seq)]
    return huds, 22 * DT, 22


@pytest.mark.parametrize("body_cls", ["PrincessTower", "Cannoneer"])
def test_a_real_fall_is_destroyed_by_label_and_opens_the_pockets(body_cls):
    r = reader()
    d = deriver()
    standing = ("enemy_right", "own_right")
    huds, t_gone, i_gone = fall_run(r, boxes_after=dets(labels=standing, body_cls=body_cls))
    for i, hud in enumerate(huds):
        feed_deriver(d, hud, 100 + 2 * i)
    for k in ("enemy_left", "own_left"):
        assert huds[i_gone - 1].towers[k] == VT.TowerHp(916, 3052), k          # 0.3, accepted
        first = next(i for i, h in enumerate(huds) if h.towers[k].hp == 0)
        assert 2.5 <= first * DT - t_gone <= 3.5, (k, first * DT - t_gone)
        assert r.towers.destroyed_by[k] == "label" and k in huds[first].towers_destroyed
        assert r.report.evidence[k]["kind"] == "label"
    assert huds[-1].towers_destroyed == ("enemy_left", "own_left") and huds[-1].crowns == (1, 1)
    assert all(huds[-1].towers[k] in FULL for k in standing)
    # legality: the enemy's left pocket opens for our Hog Rider, the other lane stays shut
    assert not pocket_legal(huds[i_gone - 1]) and pocket_legal(huds[-1])
    c = RD.PureLegality(TPS.CATALOGUE).bind(state(huds[-1]), 1)
    assert not c(0, HOG, 14500 * 18, 12500 * 18)
    assert RD.destroyed_towers(state(huds[-1])) == ((False, True, False), (False, True, False))
    # the event rules: enemy units deployed in our left pocket are plays, not late sightings
    assert d.fallen_towers == {"enemy_left", "own_left"}
    pocket = SimpleNamespace(cls="Knight", bx=3000.0, by=21000.0)
    assert d._out_of_zone(pocket, 0) is False
    assert d._out_of_zone(SimpleNamespace(cls="Knight", bx=15000.0, by=21000.0), 0) is True
    before = deriver()
    feed_deriver(before, huds[i_gone - 1], 100)
    assert before._out_of_zone(pocket, 0) is True
    assert r.calib.towers["enemy_left"].researches == 0                      # destroyed before the re-check
    assert r.towers_info()["confirmed"] == ["enemy_left", "own_left"]


def test_a_fall_from_above_075_is_never_destroyed():
    """The occlusion rule stays: a label gone while the tower read 0.8 is no destruction."""
    r = reader()
    full, high = frame(), frame(labels=(("enemy_left", S.Label(dv=S.SKIN_DV, frac=0.8)),))
    ruin = frame(labels=(("enemy_left", None),))
    seq = [full] * 10 + [high] * 12 + [ruin] * 60
    huds = [read(r, img, i * DT, dets(labels=PRINCESS if i < 22 else ("enemy_right", "own_left", "own_right")))
            for i, img in enumerate(seq)]
    assert huds[-1].towers["enemy_left"] == VT.TowerHp(2442, 3052) and huds[-1].towers_destroyed == ()
    assert r.calib.towers["enemy_left"].absent_max >= 2.5               # counted, then unlatched (re-check)
    assert r.towers_info()["calib"]["enemy_left"]["absent_max_s"] >= 2.5


def test_a_label_box_at_the_lock_stops_the_count():
    """The detector still boxing the label at the lock: no destruction, whatever the pixels."""
    r = reader()
    huds, _, _ = fall_run(r, keys=("enemy_left",), boxes_after=dets())
    assert huds[-1].towers["enemy_left"] == VT.TowerHp(916, 3052) and huds[-1].towers_destroyed == ()
    assert "absent_s" not in r.report.evidence.get("enemy_left", {})


# ---------------------------------------------------------------------------
# occlusion
# ---------------------------------------------------------------------------
def test_a_balloon_over_a_half_label_for_6_s_destroys_nothing():
    k = "enemy_left"
    half = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.5)),))
    box = S.flyer_box(k, 0.0, S.SKIN_DV)
    covered = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.5)),), flyers=(box,))
    balloon = S.det("Balloon", box)
    r = reader()
    for i in range(24):
        read(r, half, i * DT, dets())
    held = r.towers.frac[k]
    huds = [read(r, covered, (24 + i) * DT, dets(labels=()) + [balloon]) for i in range(48)]
    assert held == pytest.approx(0.5, abs=0.02)
    assert all(h.towers[k] == VT.TowerHp(int(round(held * 3052)), 3052) for h in huds)
    assert all(h.towers_destroyed == () for h in huds) and r.towers.destroyed_by == {}
    assert r.calib.towers[k].absent_max == 0.0 and r.report.calib[k]["covered"] is True


def test_a_covered_full_tower_is_never_destroyed():
    """Covered for 6 s, even by a unit the detector does not box (the count runs: the label is
    gone to the pixels), a full tower stays full: it was never read <= 0.75."""
    k = "enemy_left"
    box = S.flyer_box(k, 0.0, S.SKIN_DV)
    covered = frame(flyers=(box,))
    r = reader()
    for i in range(12):
        read(r, frame(), i * DT, dets())
    huds = [read(r, covered, (12 + i) * DT, dets(labels=("enemy_right", "own_left", "own_right"))) for i in range(48)]
    assert all(h.towers[k] in FULL and h.towers_destroyed == () for h in huds)
    assert r.calib.towers[k].absent_max >= 2.5 and r.towers.destroyed_by == {}


# ---------------------------------------------------------------------------
# a fall with the label covered now and then, or the bot tapping: the lock stays, the count goes on
# ---------------------------------------------------------------------------
def fall_with(cover=None, tap_every=None, seconds: float = 15.0, unit: str = "Minions"):
    """enemy_left at 0.3, then its label gone (a ruin, boxed as PrincessTower: the re-check's
    trigger), `cover(s after the fall)` -> a `unit` box (and its pixels) over the dead label on
    that frame, the bot tapping every `tap_every` s (HudReader.note_tap: its overlay window).
    Returns (the reader, s after the fall of the first destroyed frame or None, huds)."""
    k = "enemy_left"
    low = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.3)),))
    box = S.flyer_box(k, 0.0, S.SKIN_DV)
    ruin, ruin_cov = frame(labels=((k, None),)), frame(labels=((k, None),), flyers=(box,))
    standing = ("enemy_right", "own_left", "own_right")
    r = reader()
    for i in range(10):
        read(r, frame(), i * DT, dets())
    for i in range(10, 22):
        read(r, low, i * DT, dets())
    first, huds, next_tap = None, [], 0.3
    for j in range(int(seconds / DT)):
        ta, t = j * DT, (22 + j) * DT
        if tap_every and ta >= next_tap:
            r.note_tap(t)
            next_tap += tap_every
        c = bool(cover and cover(ta))
        hud = read(r, ruin_cov if c else ruin, t, dets(labels=standing) + ([S.det(unit, box)] if c else []))
        huds.append(hud)
        if first is None and k in (hud.towers_destroyed or ()):
            first = ta
    return r, first, huds


@pytest.mark.parametrize("name, cover, within", [
    ("one frame in 3", lambda ta: int(round(ta / DT)) % 3 == 0, (3.0, 5.0)),
    ("the first 2 s", lambda ta: ta < 2.0, (4.0, 5.0)),
    ("0-2 s and 3.1-5.5 s", lambda ta: ta < 2.0 or 3.1 <= ta < 5.5, (5.5, 8.5)),
], ids=["1in3", "first2s", "two_stretches"])
def test_a_fall_with_a_unit_over_the_dead_label_now_and_then_is_destroyed(name, cover, within):
    """Ruins are boxed as tower bodies, so the re-check would start 3 s after the label went;
    covered frames used to count as failed re-search seconds and removed the lock before the
    2.5-s count was done (the fall was then never counted: 916 HP all battle, no crown). Now a
    tower read <= 0.75 whose absence count runs keeps its lock, and a covered frame pauses the
    count without losing it."""
    r, first, huds = fall_with(cover)
    assert first is not None, name
    assert within[0] <= first <= within[1], (name, first)
    st = r.calib.towers["enemy_left"]
    assert st.status == "locked" and st.researches == 0 and st.unlatches == 0
    assert r.towers.destroyed_by["enemy_left"] == "label" and huds[-1].crowns == (1, 0)


@pytest.mark.parametrize("every, within", [(4.0, (2.5, 4.5)), (2.5, (2.5, 6.0)), (2.0, (2.5, 9.0))])
def test_a_fall_while_the_bot_taps_is_destroyed(every, within):
    """The bot's own taps (1.3-s overlay windows, every 2 s at worst): overlay frames pause the
    count; the fall is destroyed, never lost to an unlatch."""
    r, first, huds = fall_with(tap_every=every)
    assert first is not None and within[0] <= first <= within[1], (every, first)
    assert r.calib.towers["enemy_left"].unlatches == 0 and huds[-1].crowns == (1, 0)


@pytest.mark.parametrize("miss_every", [0, 40, 8])
def test_a_tower_that_falls_under_a_balloon_is_destroyed_after_it_leaves(miss_every):
    """A Balloon over enemy_left's 0.5 label for 8 s, the detector missing it 1 frame in
    `miss_every`; the tower falls under it, the Balloon stays 1 s, then leaves: the lock stays all
    along (no covered second counts as a failed search) and the fall is destroyed 2.5 s after."""
    k = "enemy_left"
    lab = S.Label(dv=S.SKIN_DV, frac=0.5)
    box = S.flyer_box(k, 0.0, S.SKIN_DV)
    half, covered = frame(labels=((k, lab),)), frame(labels=((k, lab),), flyers=(box,))
    ruin_cov, ruin = frame(labels=((k, None),), flyers=(box,)), frame(labels=((k, None),))
    others = ("enemy_right", "own_left", "own_right")
    r = reader()
    for i in range(24):
        read(r, half, i * DT, dets())

    def step(img, cov, i):
        hit = cov and not (miss_every and i % miss_every == miss_every - 1)
        return read(r, img, i * DT, dets(labels=others) + ([S.det("Balloon", box)] if hit else []))

    huds = [step(covered, True, 24 + j) for j in range(64)]
    assert r.calib.towers[k].status == "locked" and all(h.towers[k].hp == 1526 for h in huds)
    huds = [step(ruin_cov, True, 88 + j) for j in range(8)] + [step(ruin, False, 96 + j) for j in range(40)]
    first = next(j for j, h in enumerate(huds) if h.towers[k].hp == 0)
    assert 8 + 2.5 / DT - 1 <= first <= 8 + 3.0 / DT, first
    assert r.calib.towers[k].unlatches == 0 and huds[-1].towers_destroyed == (k,)


@pytest.mark.parametrize("pattern", ["always", "1in10", "1in3", "1in2"])
def test_a_flyer_boxed_on_most_frames_never_destroys_a_standing_tower(pattern):
    """A flyer over a standing 0.5 tower's label for 15 s, its pixels hiding the label, and the
    detector missing it on some frames: those frames are no evidence while the unit is boxed on
    at least as many, so the tower stands (only absent frames in a row count there)."""
    k = "enemy_left"
    lab = S.Label(dv=S.SKIN_DV, frac=0.5)
    box = S.flyer_box(k, 0.0, S.SKIN_DV)
    half, covered = frame(labels=((k, lab),)), frame(labels=((k, lab),), flyers=(box,))
    boxed = {"always": lambda j: True, "1in10": lambda j: j % 10 != 9,
             "1in3": lambda j: j % 3 != 2, "1in2": lambda j: j % 2 == 0}[pattern]
    others = ("enemy_right", "own_left", "own_right")
    r = reader()
    for i in range(24):
        read(r, half, i * DT, dets())
    huds = [read(r, covered, (24 + j) * DT, dets(labels=others) + ([S.det("Balloon", box)] if boxed(j) else []))
            for j in range(120)]
    assert all(h.towers[k].hp == 1526 and h.towers_destroyed == () for h in huds), pattern
    assert r.calib.towers[k].absent_max < 0.5 and r.calib.calibrated(k)


# ---------------------------------------------------------------------------
# P1-1: ruins boxed as towers
# ---------------------------------------------------------------------------
def test_old_rules_a_ruin_body_alone_holds_only_3_s():
    """calibrate=False (the old rules): the tower at 0.5, its label then gone, a ruin body box on
    every frame. The body bridged the zero for all battle before; now for BODY_BRIDGE_S, then the
    detector's absence and the filter's 2.5 s destroy it ("detector": still unconfirmed)."""
    k = "enemy_right"
    labels = {**{x: S.Label() for x in PRINCESS}, k: S.Label(frac=0.5)}
    half = S.battle_frame("default", labels=labels, size=SIZE)
    ruin = S.battle_frame("default", labels={**labels, k: None}, size=SIZE)
    r = reader(calibrate=False, place_labels=False, search_labels=False)
    standing = [x for x in PRINCESS if x != k]
    for i in range(16):
        read(r, half, i * DT, [S.label_det(x) for x in PRINCESS] + [S.body_det(x) for x in PRINCESS])
    t_gone = 16 * DT
    huds = [read(r, ruin, (16 + i) * DT, [S.label_det(x) for x in standing] + [S.body_det(x) for x in PRINCESS])
            for i in range(80)]
    first = next(i for i, h in enumerate(huds) if h.towers[k].hp == 0)
    assert TL.BODY_BRIDGE_S <= (16 + first) * DT - t_gone <= TL.BODY_BRIDGE_S + 3.5
    assert r.towers.destroyed_by[k] == "detector" and huds[-1].towers_destroyed == ()
    kinds = r.towers_info()["holds_by_kind"][k]
    assert kinds.get("body", 0) >= 20 and "label" not in kinds        # label gone: body holds only
    assert r.report.tower_present[k] is False


def test_old_rules_a_body_with_label_boxes_now_and_then_is_held():
    """The detector misses the label 7 frames in 8 (gaps under BODY_BRIDGE_S) while the body is
    boxed on every frame and the HUD reads nothing: held for 20 s; once the label boxes stop for
    good, the body alone holds only BODY_BRIDGE_S."""
    from tests.pipeline.test_tower_tracker import FRAME, K, reader_with
    reads = [2101] * 8 + [None] * 200
    r = reader_with(None, reads, calibrate=False, towers_wait_for_clock=False, label_hold=False)
    label = TL.TowerEvidence(labels={K: (0.0, 0.0, 1.0, 1.0)}, bodies={K})
    body = TL.TowerEvidence(bodies={K})
    huds = []
    for i in range(168):
        hud = r.read(FRAME, i * DT, towers=False)
        huds.append(r.read_towers(FRAME, i * DT, hud, label if i % 8 == 0 else body))
    assert all(h.towers[K] == VT.TowerHp(2101, 3052) for h in huds[6:]) and huds[-1].crowns == (0, 0)
    assert r.hold_kinds[K]["body"] > 100 and r.hold_kinds[K]["label"] > 10
    for i in range(168, 208):
        hud = r.read(FRAME, i * DT, towers=False)
        huds.append(r.read_towers(FRAME, i * DT, hud, body))
    first = next(i for i, h in enumerate(huds) if h.towers[K].hp == 0)
    assert 167 * DT + TL.BODY_BRIDGE_S <= first * DT <= 167 * DT + TL.BODY_BRIDGE_S + 4.0


# ---------------------------------------------------------------------------
# overtime: the revive closes the pocket
# ---------------------------------------------------------------------------
def badge_box(key: str) -> tuple[float, float, float, float]:
    """A box over `key`'s CHICKEN badge only (its bar's start stays in sight)."""
    x0, y0, x1, y1 = S.label_parts(key, 0.0, label_dv(key))["badge"]
    return (x0 - 4.0, y0 - 4.0, x1 + 1.0, y1 + 4.0)


def test_the_overtime_check_revives_a_false_fall_and_closes_its_pocket():
    """A unit the detector does not box hides enemy_left's label (at 0.3) for 3.5 s: a false
    label destruction (1-0), and the pocket opens. Then only its badge stays hidden (the bar's
    start shows: the label is confirmed, but never seen whole, so the label-back revive does not
    apply). Overtime only follows a tie, so its first reading revives enemy_left:
    hud.towers_destroyed, legality and the deriver's pockets close again, and its label keeps it
    standing."""
    from tests.pipeline.test_hud import TickClock
    k = "enemy_left"
    low = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.3)),))
    hidden = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.3)),), flyers=(S.flyer_box(k, 0.0, S.SKIN_DV),))
    badge_hidden = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.3)),), flyers=(badge_box(k),))
    r = H.HudReader(store(), {}, regions=S.REGIONS)
    r.clock_reader = TickClock(3300)                  # 2.5 ticks a frame: overtime from frame 120
    d = deriver()
    seq = [frame()] * 14 + [low] * 12 + [hidden] * 28 + [badge_hidden] * 80
    out = []
    for i, img in enumerate(seq):
        boxes = dets(labels=PRINCESS if img is not hidden else ("enemy_right", "own_left", "own_right"))
        hud = read(r, img, i * DT, boxes)
        if r.report.towers_revived:                   # LivePerception._revive_towers
            d.revive_towers(r.report.towers_revived)
        feed_deriver(d, hud, hud.clock_tick or 0)
        out.append((hud, list(r.report.towers_revived), set(d.fallen_towers)))
    falls = [i for i, (h, _, _) in enumerate(out) if h.towers[k].hp == 0]
    ot = next(i for i, (h, _, _) in enumerate(out) if h.phase == VT.PHASE_OVERTIME)
    assert falls and falls[0] < ot and r.towers.revived == [k]
    assert r.calib.seen_run(k)[1] == 0                # never seen whole: no label-back revive
    before, (at_ot, revived, fallen) = out[ot - 1][0], out[ot]
    assert before.crowns == (1, 0) and before.towers_destroyed == (k,) and pocket_legal(before)
    assert out[ot - 1][2] == {k}
    assert revived == [k] and at_ot.crowns == (0, 0) and at_ot.towers_destroyed == ()
    assert at_ot.towers[k] == VT.TowerHp(916, 3052) and not pocket_legal(at_ot) and fallen == set()
    assert all(h.towers[k].hp == 916 and h.towers_destroyed == () and not f for h, _, f in out[ot:])


def test_a_false_fall_is_revived_once_its_label_is_back():
    """The same false fall (a unit the detector does not box hides the 0.3 label for 3.5 s), then
    the label in sight again: once it has been seen whole (badge and bar) for REVIVE_S on
    REVIVE_FRAMES frames the tower stands again (towers_revived, "label back"): the confirmed
    set, legality and the deriver's pocket close long before any overtime; zero_locked from then
    on, like the overtime revive."""
    from scripts.vision import tower_calib as TC
    k = "enemy_left"
    low = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.3)),))
    hidden = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.3)),), flyers=(S.flyer_box(k, 0.0, S.SKIN_DV),))
    r = reader()
    d = deriver()
    seq = [frame()] * 10 + [low] * 12 + [hidden] * 28 + [low] * 40
    out = []
    for i, img in enumerate(seq):
        boxes = dets(labels=PRINCESS if img is not hidden else ("enemy_right", "own_left", "own_right"))
        hud = read(r, img, i * DT, boxes)
        if r.report.towers_revived:                   # LivePerception._revive_towers
            d.revive_towers(r.report.towers_revived)
        feed_deriver(d, hud, 100 + 2 * i)
        out.append((hud, list(r.report.towers_revived), set(d.fallen_towers), r.report.towers_misread))
    falls = [i for i, (h, *_) in enumerate(out) if h.towers[k].hp == 0]
    assert falls and falls[0] < 50 and out[falls[-1]][0].towers_destroyed == (k,) and pocket_legal(out[falls[-1]][0])
    assert out[falls[-1]][2] == {k}
    back = next(i for i, (_, rv, *_) in enumerate(out) if rv)
    assert TC.REVIVE_S <= (back - 50) * DT <= TC.REVIVE_S + 0.5          # the label shows again at frame 50
    hud, revived, fallen, misread = out[back]
    assert revived == [k] and misread["why"] == "label back" and misread["crowns_after"] == [0, 0]
    assert hud.towers[k] == VT.TowerHp(916, 3052) and hud.towers_destroyed == () and hud.crowns == (0, 0)
    assert not pocket_legal(hud) and fallen == set() and k in r.towers.zero_locked
    assert all(h.towers[k].hp == 916 and h.towers_destroyed == () and not f for h, _, f, _ in out[back:])


def test_a_real_fall_is_never_revived():
    """Ruins never show the label whole: a tower destroyed on label evidence stays destroyed."""
    r = reader()
    huds, _, _ = fall_run(r, keys=("enemy_left",), frames_after=120)
    assert huds[-1].towers["enemy_left"].hp == 0 and huds[-1].towers_destroyed == ("enemy_left",)
    assert r.towers.revived == [] and r.calib.seen_run("enemy_left") == (0.0, 0)


# ---------------------------------------------------------------------------
# the battle's last tower: the end screen may count it, the HUD never does
# ---------------------------------------------------------------------------
def test_the_tower_that_ends_the_battle_is_counted_by_the_end_screen_only():
    """Round 0 ended in overtime as enemy_right fell: its label still showed on the last battle
    frame, the next was the menu. The HUD reads 0-0, the end screen 1-0: the result book takes
    the end screen's and calls it explained (no crowns_disagree, no "HUD read" in the result line);
    a crown the HUD missed mid-battle (the label long gone) is still a disagreement."""
    from scripts.vision import tower_calib as TC
    r = reader()
    k = "enemy_right"
    low = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.2)),))
    for i in range(10):
        read(r, frame(), i * DT, dets())
    for i in range(10, 30):
        hud = read(r, low, i * DT, dets())
    could = TC.could_fall_at_end(r.calib.absent(), [x for x, v in hud.towers.items() if v.hp == 0])
    assert could == {"own": ["enemy_left", "enemy_right"], "enemy": ["own_left", "own_right"]}
    errors = []
    runner = SimpleNamespace(perception=SimpleNamespace(hud_reader=r), _error=lambda *a, **kw: errors.append(a))
    assert VP.VisionPlay._could_fall(runner, hud) == could and errors == []      # what _end_battle passes
    book = VP.ResultBook()
    book.open(1, "end screen", 50.0, hud.crowns, last_tick=4413, could_fall=could)
    book.end_frame((1, 0), 56.5)
    rec = book.finish()
    assert (rec["crowns"], rec["source"], rec["outcome"]) == ([1, 0], "end screen", "win")
    assert "crowns_disagree" not in rec and rec["crowns_explained"].startswith("own crowns: the battle ended as ")
    # the label gone for 10 s before the end: the HUD should have counted it
    ruin = frame(labels=((k, None),))
    for i in range(30, 110):
        hud = read(r, ruin, i * DT, dets(labels=("enemy_left", "own_left", "own_right"), bodies=()))
    assert hud.crowns == (1, 0)                                         # it did: label evidence
    could = TC.could_fall_at_end(r.calib.absent(), [x for x, v in hud.towers.items() if v.hp == 0])
    assert could["own"] == ["enemy_left"]
    book.open(2, "end screen", 90.0, (0, 0), last_tick=3000, could_fall={"own": [], "enemy": []})
    book.end_frame((1, 0), 91.0)
    rec = book.finish()
    assert rec["crowns_disagree"] == {"hud": [0, 0], "end": [1, 0]} and "no tower's label" in rec["crowns_why"]
    book.open(3, "end screen", 130.0, (1, 0), last_tick=2000)
    book.end_frame((3, 0), 131.0)
    rec = book.finish()
    assert rec["crowns_explained"] == "own crowns: a king fell (3 crowns)" and "crowns_disagree" not in rec
