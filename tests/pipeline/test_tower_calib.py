"""Per-battle tower label calibration and the read gate (scripts/vision/tower_calib.py; skin audit
fix list P0-1 and P1-4), and the deploy-overlay test it skips (scripts/vision/overlay.py).

The owner picks a random tower skin every battle, clanmates use many and the arena changes with
the season and the mode: every frame here is drawn by tests/pipeline/skin_synth.py (7 floors, 5
decorations in a label's area, labels at any (du, dv), the deploy overlay, flying units), at
1080x1920 and at the owner's 747x1328. No real frame is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import hud as H  # noqa: E402
from scripts.vision import overlay as OV  # noqa: E402
from scripts.vision import templates as TP  # noqa: E402
from scripts.vision import tower_calib as TC  # noqa: E402
from scripts.vision import tower_labels as TL  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from tests.pipeline import skin_synth as S  # noqa: E402

PRINCESS = TL.PRINCESS_KEYS
REFS, _ = S.REGIONS.label_refs()
ANCHORS = {k: S.REGIONS.get(f"bar_{k}").u0 * 1080.0 + H.BAR_MARGIN_PX for k in PRINCESS}
#: the spec's grid: du in {-40, 0, +40}, dv in {-110, -36, 0, +50}
POSITIONS = [(du, dv) for du in (-40.0, 0.0, 40.0) for dv in (-110.0, -36.0, 0.0, 50.0)]
CHICKEN = {k: S.Label(dv=S.SKIN_DV if k in S.ENEMY else 0.0) for k in PRINCESS}
DT = 0.125
SIZE_IDS = ["1080", "747"]


def calibrator(**kw) -> TC.TowerCalibrator:
    return TC.TowerCalibrator(REFS, ANCHORS, legal_full=H.LEGAL_TOWER_MAX_HP, **kw)


def feed(cal: TC.TowerCalibrator, frames, t0: float = 0.0, dt: float = DT, **kw) -> list[TC.CalibFrame]:
    out = []
    for i, f in enumerate(frames):
        evidence = kw.pop("evidence_fn", None)
        out.append(cal.update(f, t0 + i * dt, running=kw.get("running", True),
                              evidence=None if evidence is None else evidence(i)))
        if evidence is not None:
            kw["evidence_fn"] = evidence
    return out


def ev(dets, size) -> TL.TowerEvidence:
    return TL.TowerEvidence.from_detections(dets, size, REFS)


def has_pil_digits() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return S.pil_glyph_fn() is not None


def store_for(size):
    """Tower digit templates for frames of `size` (needs Pillow at 747x1328, skin_synth.glyph_fn)."""
    if size == S.OWNER_SIZE and not has_pil_digits():
        return TP.TemplateStore.memory()
    return S.digit_store(size)


def reader(size, **kw) -> H.HudReader:
    return H.HudReader(store_for(size), {}, regions=S.REGIONS, towers_wait_for_clock=False, **kw)


def read(r: H.HudReader, img, t, dets=None, size=None) -> VT.Hud:
    hud = r.read(img, t, towers=False)
    evidence = None if dets is None else r.tower_evidence(dets, size or (img.shape[1], img.shape[0]))
    return r.read_towers(img, t, hud, evidence)


# ---------------------------------------------------------------------------
# the signature: every floor, every decoration, both sizes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", S.SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("decoration", (None,) + S.DECORATIONS)
@pytest.mark.parametrize("floor", S.FLOOR_NAMES)
def test_the_search_finds_every_label_within_2_px(floor, decoration, size):
    """The 12 (du, dv) of the spec over the four towers in three frames per case (rotated by
    floor, so each tower sees most of them), the decoration drawn at each label."""
    shift = S.FLOOR_NAMES.index(floor)
    for j in range(3):
        pos = {k: POSITIONS[(4 * j + i + shift) % len(POSITIONS)] for i, k in enumerate(PRINCESS)}
        labels = {k: S.Label(du=du, dv=dv) for k, (du, dv) in pos.items()}
        decos = [] if decoration is None else [(k, decoration, du, dv) for k, (du, dv) in pos.items()]
        img = S.battle_frame(floor, labels=labels, decorations=decos, size=size)
        for k, (du, dv) in pos.items():
            c = TC.find_label(img, k, ANCHORS[k], REFS[k])
            assert c is not None and abs(c.du - du) <= 2.0 and abs(c.dv - dv) <= 2.0, (k, du, dv, c)
            assert c.full


@pytest.mark.parametrize("size", S.SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("floor", S.FLOOR_NAMES)
def test_every_tower_locks_after_8_frames_over_1_s(floor, size):
    labels = {"enemy_left": S.Label(du=-40.0, dv=S.SKIN_DV), "enemy_right": S.Label(du=40.0, dv=-110.0),
              "own_left": S.Label(dv=50.0), "own_right": S.Label()}
    img = S.battle_frame(floor, labels=labels, decorations=[("enemy_left", "comb", -40.0, S.SKIN_DV)],
                         size=size)
    cal = calibrator()
    frames = feed(cal, [img] * 8)
    assert not any(cal.calibrated(k) for k in PRINCESS)                  # 8 frames, 0.875 s
    assert all(r.status == TC.STATUS_SEARCHING and r.candidate is not None for r in frames[-1].reads.values())
    fr = cal.update(img, 8 * DT, running=True)                           # the 9th: 1.0 s
    for k, lab in labels.items():
        lk = cal.lock(k)
        assert lk is not None and fr.reads[k].calibrated and fr.reads[k].confirmed is True, k
        assert abs(lk.du - lab.du) <= 2.0 and abs(lk.dv - lab.dv) <= 2.0, (k, lk.du, lk.dv)
        assert (lk.source, lk.frames, lk.locked_t, lk.size) == ("search", 9, 8 * DT, tuple(size))
        assert lk.badge.size and lk.fill_rgb.shape == (lk.rows[1] - lk.rows[0], 3) and lk.digit_patch is not None
        assert fr.reads[k].seen is True and fr.reads[k].ncc > 0.95
    placed = cal.placement()
    assert placed["own_right"] == (0.0, 0.0)                             # inside the dead band
    assert placed["enemy_right"][0] == pytest.approx(40.0, abs=2.0)


def test_a_sparse_recording_locks_on_three_frames_over_3_s():
    """vision_play's dry run kept an image every 20th frame: three agreeing frames 1.5 s apart
    lock, and the window lasts until 8 frames were searched."""
    img = S.battle_frame("sand", labels=CHICKEN, size=S.OWNER_SIZE)
    cal = calibrator()
    feed(cal, [img] * 2, dt=1.5)
    assert not cal.calibrated("enemy_left")
    cal.update(img, 3.0, running=True)
    assert cal.calibrated("enemy_left") and cal.lock("enemy_left").frames == 3
    bare = S.battle_frame("sand", labels={k: S.Label(bar=False, badge=False) for k in PRINCESS}, size=S.OWNER_SIZE)
    late = calibrator()
    frames = feed(late, [bare] * 7, dt=2.0)                              # 12 s, 7 frames searched
    assert all(r.status == TC.STATUS_SEARCHING for r in frames[-1].reads.values())
    assert late.update(bare, 14.0, running=True).reads["enemy_left"].status == TC.STATUS_SEARCHING   # the 8th
    assert late.update(bare, 16.0, running=True).reads["enemy_left"].status == TC.STATUS_EXPIRED


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", S.SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("floor", S.FLOOR_NAMES)
def test_gold_without_a_bar_and_a_comb_without_a_badge_are_no_label(floor, size):
    cases = {
        "flowers, badge, no bar": ({k: S.Label(bar=False) for k in PRINCESS}, "flowers"),
        "pad, no badge, no bar": ({k: S.Label(bar=False, badge=False) for k in PRINCESS}, "pad"),
        "pad, badge, no bar": ({k: S.Label(bar=False) for k in PRINCESS}, "pad"),
        "comb, bar, no badge": ({k: S.Label(badge=False) for k in PRINCESS}, "comb"),
    }
    for name, (labels, deco) in cases.items():
        img = S.battle_frame(floor, labels=labels, decorations=[(k, deco) for k in PRINCESS], size=size)
        for k in PRINCESS:
            assert TC.find_candidates(img, k, ANCHORS[k], REFS[k]) == [], (name, k)


def test_refused_frames_never_lock_and_the_window_expires():
    img = S.battle_frame("red", labels={k: S.Label(badge=False) for k in PRINCESS},
                         decorations=[(k, "comb") for k in PRINCESS], size=S.OWNER_SIZE)
    cal = calibrator()
    frames = feed(cal, [img] * 80)                                       # 10 s
    assert not any(cal.calibrated(k) for k in PRINCESS)
    assert all(r.status == TC.STATUS_EXPIRED and r.confirmed is None and r.seen is None
               for r in frames[-1].reads.values())
    assert cal.placement() == {}


@pytest.mark.parametrize("size", S.SIZES, ids=SIZE_IDS)
def test_the_overlay_is_flagged_on_every_floor(size):
    for floor in S.FLOOR_NAMES:
        tinted = S.battle_frame(floor, labels=CHICKEN, overlay=True, size=size)
        plain = S.battle_frame(floor, labels=CHICKEN, size=size)
        assert OV.overlay_on(tinted) and OV.mirror_tint(tinted) > 0.9, floor
        assert not OV.overlay_on(plain) and OV.mirror_tint(plain) < 0.1, floor
    flyers = [S.flyer_box(k, 0.0, S.SKIN_DV if k in S.ENEMY else 0.0) for k in PRINCESS]
    flyers += [(x, y, x + 120.0, y + 120.0) for x, y in ((380, 520), (620, 600), (150, 650))]
    assert OV.overlay_on(S.battle_frame("sand", labels=CHICKEN, overlay=True, flyers=flyers, size=size))
    assert not OV.overlay_on(S.battle_frame("sand", labels=CHICKEN, flyers=flyers, size=size))
    assert len(OV.patch_centres()) >= 40


def test_overlay_frames_contribute_nothing_to_a_lock():
    tinted = S.battle_frame("default", labels=CHICKEN, overlay=True, size=S.OWNER_SIZE)
    plain = S.battle_frame("default", labels=CHICKEN, size=S.OWNER_SIZE)
    cal = calibrator()
    frames = feed(cal, [tinted] * 16)                                    # 2 s of overlay
    assert all(f.overlay is True for f in frames)
    assert all(r.candidate is None for f in frames for r in f.reads.values())
    assert not any(cal.calibrated(k) for k in PRINCESS) and not any(cal.towers[k].votes for k in PRINCESS)
    frames = feed(cal, [plain] * 9, t0=2.0)
    assert frames[-1].overlay is False
    assert all(cal.lock(k) is not None and cal.lock(k).frames == 9 and cal.lock(k).locked_t == 3.0 for k in PRINCESS)
    told = calibrator()                                                  # the caller's flag wins
    for i in range(12):
        told.update(plain, i * DT, running=True, overlay=True)
    assert not told.calibrated("enemy_left")


# ---------------------------------------------------------------------------
# HudReader: a label it cannot find is never read
# ---------------------------------------------------------------------------
def test_a_label_outside_the_window_is_never_read():
    """enemy_left's label 200 px up: never calibrated, no reading for 60 s (2 frames a second),
    full all along; the other towers read. The pre-calibration reader read its fixed boxes."""
    size = S.OWNER_SIZE
    labels = {**CHICKEN, "enemy_left": S.Label(dv=-200.0)}
    img = S.battle_frame("sand", labels=labels, size=size)
    r = reader(size)
    huds = [read(r, img, i * 0.5) for i in range(120)]
    k = "enemy_left"
    c = r.report.summary()["calib"][k]
    assert (c["calibrated"], c["state"], c["source"]) == (False, TC.STATUS_EXPIRED, "none")
    assert all(h.towers[k] in (VT.TowerHp(None, 3052), VT.TowerHp(3052, 3052)) for h in huds)
    assert r.towers.frac[k] in (None, 1.0) and huds[-1].crowns == (0, 0)
    assert r.report.tower_digits[k].hp is None and r.report.tower_digits[k].reason == "label not confirmed"
    assert r.report.tower_bars[k] is None and "enemy_left" not in r.tower_reader.placement
    assert r.report.tower_shift[k] == [0.0, 0.0, "uncalibrated"]
    assert all(r.calib.calibrated(x) for x in ("enemy_right", "own_left", "own_right"))
    assert r.report.tower_bars["enemy_right"] == 1.0
    old = reader(size, calibrate=False, search_labels=False)
    read(old, img, 0.0)
    assert old.report.tower_bars[k] == 0.0                               # the fixed box: floor


def test_the_chicken_battle_reads_full_towers_never_the_comb():
    """The 10-31 friendly drawn: the enemy labels 36 px up, the red comb in the calibrated bar
    band, a sand floor. The fixed boxes read the comb as a 0.25 bar; the calibrated reader
    reads nothing before the lock and 1.0 (and 3052) from it."""
    size = S.OWNER_SIZE
    decos = [(k, "comb", 0.0, 0.0) for k in S.ENEMY]
    img = S.battle_frame("sand", labels=CHICKEN, decorations=decos, size=size)
    old = reader(size, calibrate=False, search_labels=False, place_labels=False)
    read(old, img, 0.0)
    assert 0.1 < old.report.tower_bars["enemy_left"] < 0.95
    r = reader(size)
    bars = []
    for i in range(40):
        hud = read(r, img, i * DT, [S.body_det(k, size=size) for k in PRINCESS], size)
        bars.append(r.report.tower_bars["enemy_left"])
    assert bars[:8] == [None] * 8 and bars[8:] == [1.0] * 32
    assert hud.towers["enemy_left"] in (VT.TowerHp(3052, 3052), VT.TowerHp(None, 3052)) and hud.crowns == (0, 0)
    if has_pil_digits():
        assert r.report.tower_digits["enemy_left"].hp == 3052 and hud.towers["enemy_left"] == VT.TowerHp(3052, 3052)
    du, dv, src = r.report.tower_shift["enemy_left"]
    assert src == "search" and du == 0.0 and abs(dv - S.SKIN_DV) <= 2.0
    assert r.report.tower_shift["own_left"] == [0.0, 0.0, "search"]


@pytest.mark.parametrize("size", S.SIZES, ids=SIZE_IDS)
def test_a_covered_label_reads_nothing_and_keeps_its_hp(size):
    """A flying unit over a half-HP label for 2 s: no digits and no bar (None, not 0), the HP
    held, the record says covered; then read again. A full tower covered for 6 s stays full."""
    k = "enemy_left"
    labels = {**CHICKEN, k: S.Label(dv=S.SKIN_DV, frac=0.5)}
    img = S.battle_frame("forest", labels=labels, size=size)
    box = S.flyer_box(k, 0.0, S.SKIN_DV)
    covered = S.battle_frame("forest", labels=labels, flyers=[box], size=size)
    bodies = [S.body_det(x, size=size) for x in PRINCESS]
    balloon = S.det("Balloon", box, size=size)
    r = reader(size)
    for i in range(16):
        read(r, img, i * DT, bodies, size)
    held = read(r, img, 16 * DT, bodies, size).towers[k]
    assert held == VT.TowerHp(1526, 3052) and r.report.tower_bars[k] == pytest.approx(0.5, abs=0.02)
    for i in range(17, 33):
        hud = read(r, covered, i * DT, bodies + [balloon], size)
        assert r.report.tower_digits[k].hp is None and r.report.tower_bars[k] is None
        assert hud.towers[k] == held, i
        c = r.report.calib[k]
        assert c["covered"] is True and c["confirmed"] is False and "seen" not in c      # unsure, not absent
    hud = read(r, img, 33 * DT, bodies, size)
    assert r.report.calib[k]["confirmed"] is True and hud.towers[k] == held
    assert r.calib.towers[k].researches == 0
    full = reader(size)
    fl_img = S.battle_frame("forest", labels=CHICKEN, size=size)
    fl_cov = S.battle_frame("forest", labels=CHICKEN, flyers=[box], size=size)
    for i in range(12):
        read(full, fl_img, i * DT, bodies, size)
    huds = [read(full, fl_cov, (12 + i) * DT, bodies + [balloon], size) for i in range(48)]
    assert all(h.towers[k].hp in (None, 3052) for h in huds) and huds[-1].crowns == (0, 0)
    assert full.calib.towers[k].status == TC.STATUS_LOCKED and full.calib.towers[k].researches == 0


class IntroClock:
    """No reading on the intro frames, then 2:56 on every frame."""
    ready = True

    def __init__(self, intro: int):
        self.n = 0
        self.intro = intro

    def read(self, rgb):
        self.n += 1
        if self.n <= self.intro:
            return H.ClockRead(None, None, 0.0, None, "no glyphs")
        return H.ClockRead(176, None, 0.95, "2:56")


def test_the_battle_intro_never_locks():
    """Frames before the third clock reading never search: the intro (labels covered by the
    intro's banners, drawn as flying units) and the first two readings (labels shown); the
    search starts on the third and locks 1 s later."""
    size = S.OWNER_SIZE
    shown = S.battle_frame("default", labels=CHICKEN, size=size)
    flyers = [S.flyer_box(k, 0.0, S.SKIN_DV if k in S.ENEMY else 0.0) for k in PRINCESS]
    intro = S.battle_frame("default", labels=CHICKEN, flyers=flyers, size=size)
    r = H.HudReader(store_for(size), {}, regions=S.REGIONS)
    r.clock_reader = IntroClock(6)
    for i in range(6):
        read(r, intro, i * DT)
        assert all(c["state"] == TC.STATUS_WAITING and not c["calibrated"] for c in r.report.calib.values())
    for i in range(6, 8):                                                # readings 1 and 2
        read(r, shown, i * DT)
        assert all(c["state"] == TC.STATUS_WAITING for c in r.report.calib.values())
        assert not any(r.calib.towers[k].votes for k in PRINCESS)
    read(r, shown, 8 * DT)                                               # the third: the clock runs
    assert all(c["state"] == TC.STATUS_SEARCHING for c in r.report.calib.values())
    for i in range(9, 17):
        read(r, shown, i * DT)
    assert all(r.calib.lock(k) is not None and r.calib.lock(k).locked_t == 16 * DT for k in PRINCESS)
    fr = r.calib.update(shown, 17 * DT, running=False)                   # a rejected reading later on
    assert all(x.confirmed is True for x in fr.reads.values())          # ... does not stop the gate


# ---------------------------------------------------------------------------
# re-check, re-search, unlatch; the detector's votes
# ---------------------------------------------------------------------------
def locked(img, size=S.OWNER_SIZE) -> TC.TowerCalibrator:
    cal = calibrator()
    feed(cal, [img] * 9)
    assert all(cal.calibrated(k) for k in PRINCESS)
    return cal


def test_a_wrong_lock_is_searched_again_and_moves():
    """Locked at dv -36; the label is then at dv 0 (a damaged bar, 0.6) while the detector boxes
    the body: 3 s without a match start a re-search, which locks the new place."""
    size = S.OWNER_SIZE
    k = "enemy_left"
    cal = locked(S.battle_frame("sand", labels=CHICKEN, size=size))
    moved = S.battle_frame("sand", labels={**CHICKEN, k: S.Label(dv=0.0, frac=0.6)}, size=size)
    body = ev([S.body_det(x, size=size) for x in PRINCESS], size)
    t0 = 9 * DT
    statuses = []
    for i in range(48):
        fr = cal.update(moved, t0 + i * DT, running=True, evidence=body)
        statuses.append(fr.reads[k].status)
        if i < 23:
            assert fr.reads[k].confirmed is False
    start = statuses.index(TC.STATUS_RESEARCH)
    assert (start + 1) * DT >= TC.LOST_S                                 # >= 3 s after the last match
    lk = cal.lock(k)
    assert lk.source == "re-search" and abs(lk.dv) <= 2.0 and cal.towers[k].relocks == 1
    assert statuses[-1] == TC.STATUS_LOCKED and fr.reads[k].confirmed is True
    assert cal.lock("enemy_right").source == "search"                    # the others kept theirs


def test_a_vanished_label_is_unlatched_after_three_searches():
    size = S.OWNER_SIZE
    k = "enemy_left"
    cal = locked(S.battle_frame("sand", labels=CHICKEN, size=size))
    gone = S.battle_frame("sand", labels={**CHICKEN, k: None}, size=size)    # a ruin
    body = ev([S.body_det(x, size=size) for x in PRINCESS], size)
    t0 = 9 * DT
    seen = []
    for i in range(80):
        fr = cal.update(gone, t0 + i * DT, running=True, evidence=body)
        seen.append((fr.reads[k].status, fr.reads[k].seen))
    assert seen[0] == (TC.STATUS_LOCKED, False)
    first = next(i for i, (s, _) in enumerate(seen) if s == TC.STATUS_UNLATCHED)
    assert 5.5 <= (first + 1) * DT <= 7.5                                # 3 s lost + 3 empty seconds
    assert not cal.calibrated(k) and fr.reads[k].confirmed is None and cal.towers[k].unlatches == 1
    assert cal.calibrated("enemy_right")


def test_the_detectors_votes_confirm_a_lock_or_send_it_searching():
    size = S.OWNER_SIZE
    k = "enemy_left"
    base = S.battle_frame("sand", labels=CHICKEN, size=size)
    blob = S.label_parts(k, 0.0, S.SKIN_DV)["badge"]
    hidden = S.battle_frame("sand", labels=CHICKEN, size=size)
    S.fill_box(hidden, (blob[0] - 4, blob[1] - 4, S.label_parts(k, 0.0, S.SKIN_DV)["bar"][2] + 4, blob[3] + 4),
               S.FLYER)                                                  # no box for it: not "covered"
    bodies = [S.body_det(x, size=size) for x in PRINCESS]
    agree = ev(bodies + [S.label_det(k, S.SKIN_DV, size=size)], size)
    plain = ev(bodies, size)
    for evidence, researches in ((agree, 0), (plain, 1)):
        cal = locked(base)
        for i in range(48):
            fr = cal.update(hidden, (9 + i) * DT, running=True, evidence=evidence)
            assert fr.reads[k].confirmed is not True
        assert cal.towers[k].researches == researches
    far = ev([S.label_det(k, S.SKIN_DV + 40.0, size=size)], size)        # no body: votes elsewhere
    cal = locked(base)
    for i in range(30):
        cal.update(hidden, (9 + i) * DT, running=True, evidence=far)
    assert cal.towers[k].researches == 1
    gone = ev([], size)
    cal = locked(base)
    for i in range(48):                                                  # no body, no votes: kept
        cal.update(hidden, (9 + i) * DT, running=True, evidence=gone)
    assert cal.towers[k].researches == 0 and cal.calibrated(k)


def hidden_label(base_labels, k, size):
    """`base_labels` drawn with `k`'s label painted over by a unit the detector does not box."""
    img = S.battle_frame("sand", labels=base_labels, size=size)
    parts = S.label_parts(k, 0.0, S.SKIN_DV if k in S.ENEMY else 0.0)
    blob = parts["badge"]
    S.fill_box(img, (blob[0] - 4, blob[1] - 4, parts["bar"][2] + 4, blob[3] + 4), S.FLYER)
    return img


def test_a_low_tower_whose_label_is_gone_keeps_its_lock():
    """update's low (the tower read <= 0.75): while its absence count runs it is never re-checked,
    whatever the detector boxes there (ruins are boxed as bodies); the count goes on to the end.
    The same tower not low is re-searched and unlatched as before."""
    size = S.OWNER_SIZE
    k = "enemy_left"
    gone = S.battle_frame("sand", labels={**CHICKEN, k: None}, size=size)
    body = ev([S.body_det(x, size=size) for x in PRINCESS], size)
    for low, researches in (({k}, 0), (set(), 1)):
        cal = locked(S.battle_frame("sand", labels=CHICKEN, size=size))
        for i in range(80):
            fr = cal.update(gone, (9 + i) * DT, running=True, evidence=body, low=low)
        assert cal.towers[k].researches == researches, low
        if low:
            assert cal.calibrated(k) and fr.reads[k].status == TC.STATUS_LOCKED
            assert fr.reads[k].absent_s == pytest.approx(79 * DT)
        else:
            assert cal.towers[k].status == TC.STATUS_UNLATCHED


def test_a_covered_second_is_no_failed_search():
    """A re-search under way (the label hidden 3 s by a unit the detector does not box), then a
    unit box over the label for 6 s: no search runs on covered frames, so those seconds count
    neither way and the lock stays; the label then shows and ends the re-search."""
    size = S.OWNER_SIZE
    k = "enemy_left"
    base = S.battle_frame("sand", labels=CHICKEN, size=size)
    hidden = hidden_label(CHICKEN, k, size)
    bodies = [S.body_det(x, size=size) for x in PRINCESS]
    cover = ev(bodies + [S.det("Balloon", S.flyer_box(k, 0.0, S.SKIN_DV), size=size)], size)
    cal = locked(base)
    t = 9 * DT
    for i in range(30):                                                  # 3.75 s: a re-search starts
        cal.update(hidden, t, running=True, evidence=ev(bodies, size))
        t += DT
    assert cal.towers[k].status == TC.STATUS_RESEARCH and cal.towers[k].failed <= 1
    for i in range(48):                                                  # 6 s covered
        fr = cal.update(hidden, t, running=True, evidence=cover)
        t += DT
        assert fr.reads[k].covered and cal.calibrated(k), i
    assert cal.towers[k].status == TC.STATUS_RESEARCH and cal.towers[k].unlatches == 0
    fr = cal.update(base, t, running=True, evidence=ev(bodies, size))
    assert fr.reads[k].status == TC.STATUS_LOCKED and fr.reads[k].confirmed is True


def test_an_unlatched_lock_comes_back_once_its_label_shows_whole():
    """A label hidden long enough (a unit the detector misses) is unlatched; its old lock is still
    checked, and once the label is seen whole there on LOCK_FRAMES frames over LOCK_SPAN_S it is
    the lock again (source "re-lock"), with no detector vote needed. A ruin never brings it back."""
    size = S.OWNER_SIZE
    k = "enemy_left"
    base = S.battle_frame("sand", labels=CHICKEN, size=size)
    hidden = hidden_label(CHICKEN, k, size)
    gone = S.battle_frame("sand", labels={**CHICKEN, k: None}, size=size)
    body = ev([S.body_det(x, size=size) for x in PRINCESS], size)
    for back, relocked in ((base, True), (gone, False)):
        cal = locked(base)
        old = cal.lock(k)
        t = 9 * DT
        for i in range(64):
            cal.update(hidden, t, running=True, evidence=body)
            t += DT
        assert cal.towers[k].status == TC.STATUS_UNLATCHED and not cal.calibrated(k)
        reads = []
        for i in range(24):
            reads.append(cal.update(back, t, running=True, evidence=body).reads[k])
            t += DT
        assert cal.calibrated(k) is relocked
        if relocked:
            first = next(i for i, r in enumerate(reads) if r.calibrated)
            assert TC.LOCK_FRAMES - 1 <= first <= TC.LOCK_FRAMES + 1
            lk = cal.lock(k)
            assert lk.source == TC.SOURCE_RELOCK and (lk.du, lk.dv) == (old.du, old.dv) and cal.towers[k].relocks == 1
            assert reads[-1].confirmed is True and reads[-1].seen is True


def test_the_label_back_run_counts_whole_sightings():
    """seen_run: the frames and seconds the label has been seen whole since it was last not seen;
    a frame without it ends the run."""
    size = S.OWNER_SIZE
    k = "enemy_left"
    base = S.battle_frame("sand", labels=CHICKEN, size=size)
    gone = S.battle_frame("sand", labels={**CHICKEN, k: None}, size=size)
    cal = locked(base)
    for i in range(3):
        cal.update(base, (9 + i) * DT, running=True)
    assert cal.seen_run(k) == (pytest.approx(2 * DT), 3)
    cal.update(gone, 2.0, running=True)
    assert cal.seen_run(k) == (0.0, 0)
    for i in range(5):
        cal.update(base, 2.125 + i * DT, running=True)
    assert cal.seen_run(k) == (pytest.approx(4 * DT), 5)


def test_a_destroyed_tower_is_not_searched_again():
    size = S.OWNER_SIZE
    k = "enemy_left"
    cal = locked(S.battle_frame("sand", labels=CHICKEN, size=size))
    gone = S.battle_frame("sand", labels={**CHICKEN, k: None}, size=size)
    body = ev([S.body_det(x, size=size) for x in PRINCESS], size)
    for i in range(60):
        cal.update(gone, (9 + i) * DT, running=True, evidence=body, destroyed={k})
    assert cal.towers[k].researches == 0 and cal.calibrated(k)


def test_an_expired_tower_is_searched_again_when_the_detector_asks():
    """No label in the window (here: hidden), then shown with the detector's tower-bar votes."""
    size = S.OWNER_SIZE
    k = "enemy_left"
    hidden = S.battle_frame("sand", labels={**CHICKEN, k: S.Label(dv=S.SKIN_DV, bar=False)}, size=size)
    shown = S.battle_frame("sand", labels=CHICKEN, size=size)
    cal = calibrator()
    feed(cal, [hidden] * 72)                                             # 9 s: expired
    assert cal.towers[k].status == TC.STATUS_EXPIRED
    votes = ev([S.label_det(k, S.SKIN_DV, size=size)], size)
    for i in range(20):
        cal.update(shown, 9.0 + i * DT, running=True, evidence=votes)
    assert cal.calibrated(k) and cal.lock(k).source == "re-search" and abs(cal.lock(k).dv - S.SKIN_DV) <= 2.0


def test_a_new_frame_size_starts_over():
    cal = locked(S.battle_frame("sand", labels=CHICKEN, size=S.OWNER_SIZE))
    big = S.battle_frame("sand", labels=CHICKEN, size=S.REF_SIZE)
    fr = cal.update(big, 9 * DT, running=True)
    assert fr.reads["enemy_left"].status == TC.STATUS_SEARCHING and not cal.calibrated("enemy_left")


# ---------------------------------------------------------------------------
# the read gate on damaged bars; P1-4: the badge by its template
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", S.SIZES, ids=SIZE_IDS)
def test_damaged_bars_read_at_the_lock(size):
    k = "enemy_right"
    r = reader(size)
    full = S.battle_frame("snow", labels=CHICKEN, size=size)
    for i in range(10):
        read(r, full, i * DT)
    for frac in (0.6, 0.3, 0.02):
        img = S.battle_frame("snow", labels={**CHICKEN, k: S.Label(dv=S.SKIN_DV, frac=frac)}, size=size)
        read(r, img, 2.0)
        assert r.report.calib[k]["confirmed"] is True and r.report.calib[k].get("seen") in (True, None)
        if frac >= 0.3:
            assert r.report.tower_bars[k] == pytest.approx(frac, abs=0.02)


@pytest.mark.parametrize("size", S.SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("level", range(9, 16))
def test_real_badges_are_seen_at_every_level(level, size):
    labels = {k: S.Label(dv=S.SKIN_DV if k in S.ENEMY else 0.0, level=level) for k in PRINCESS}
    img = S.battle_frame("default", labels=labels, size=size)
    cal = locked(img, size)
    fr = cal.update(img, 2.0, running=True)
    assert all(fr.reads[k].seen is True for k in PRINCESS) and cal.seen() == {k: True for k in PRINCESS}


@pytest.mark.parametrize("size", S.SIZES, ids=SIZE_IDS)
def test_gold_at_the_badge_column_is_not_the_badge(size):
    """After the lock, the badge is gone and a 40-px pad, flowers or gold trim (the roof) is there:
    never seen; with the bar gone too (a ruin): False."""
    k = "enemy_left"
    cal = locked(S.battle_frame("sand", labels=CHICKEN, size=size), size)
    for deco in ("pad", "flowers", "roof"):
        no_badge = S.battle_frame("sand", labels={**CHICKEN, k: S.Label(dv=S.SKIN_DV, badge=False)},
                                  decorations=[(k, deco, 0.0, S.SKIN_DV)], size=size)
        r = cal.update(no_badge, 2.0, running=True).reads[k]
        assert r.seen is None and r.ncc < TC.NCC_MIN and r.bar_start is True, deco    # the bar shows
        ruin = S.battle_frame("sand", labels={**CHICKEN, k: None}, decorations=[(k, deco, 0.0, S.SKIN_DV)],
                              size=size)
        r = cal.update(ruin, 2.125, running=True).reads[k]
        assert r.seen is False and r.confirmed is False and r.ncc < TC.NCC_MIN, deco


@pytest.mark.parametrize("size", S.SIZES, ids=SIZE_IDS)
def test_a_half_covered_badge_is_seen_or_unsure_never_absent(size):
    """A unit's HP bar (fill, track and outline) over the top or the bottom half of the badge:
    seen or None, never False; the bar's start still confirms the label. Over the bar's start
    too: unconfirmed for that frame, and still never False."""
    k = "own_left"
    cal = locked(S.battle_frame("ice", labels=CHICKEN, size=size), size)
    x0, y0, x1, y1 = S.label_parts(k)["badge"]
    mid = (y0 + y1) / 2.0
    for top in (True, False):
        for reach in (-3.0, 8.0):                                        # short of the anchor, over it
            img = S.battle_frame("ice", labels=CHICKEN, size=size)
            band = (x0 - 6.0, y0 - 2.0, x1 + reach, mid) if top else (x0 - 6.0, mid, x1 + reach, y1 + 2.0)
            S.draw_bar(img, band, 0.7, "own")
            r = cal.update(img, 2.0, running=True).reads[k]
            assert r.seen in (True, None), (top, reach)
            if reach < 0:
                assert r.confirmed is True and r.bar_start is True, top


def test_tower_evidence_keeps_the_unit_boxes():
    size = S.OWNER_SIZE
    e = ev([S.det("Balloon", (100, 100, 200, 200), size=size), S.det("king-tower-bar", (480, 40, 600, 70), size=size),
            S.body_det("enemy_left", size=size), S.label_det("enemy_left", size=size)], size)
    assert [name for name, _ in e.others] == ["Balloon"]
    assert e.others[0][1] == pytest.approx((100.0, 100.0, 200.0, 200.0), abs=1.0)


# ---------------------------------------------------------------------------
# what HudReader writes
# ---------------------------------------------------------------------------
def test_the_records_carry_the_calibration():
    size = S.OWNER_SIZE
    img = S.battle_frame("sand", labels=CHICKEN, size=size)
    r = reader(size)
    read(r, img, 0.0)
    s = r.report.summary()
    assert set(s["calib"]) == set(PRINCESS) and s["overlay"] is False
    assert s["calib"]["enemy_left"] == {"calibrated": False, "state": "searching", "du": 0.0, "dv": 0.0,
                                        "source": "none"}
    assert s["tower_shift"]["enemy_left"] == [0.0, 0.0, "uncalibrated"]
    for i in range(1, 10):
        read(r, img, i * DT)
    s, c = r.report.summary(), r.report.towers_compact()
    el = s["calib"]["enemy_left"]
    assert (el["calibrated"], el["state"], el["source"], el["frames"], el["locked_t"], el["confirmed"], el["seen"]) == (
        True, "locked", "search", 9, 1.0, True, True)
    assert abs(el["dv"] - S.SKIN_DV) <= 2.0
    assert set(c["calib"]["enemy_left"]) == set(H.CALIB_COMPACT) and c["overlay"] is False
    info = r.towers_info()
    assert info["calib"]["enemy_left"]["calibrated"] is True and info["source"]["enemy_left"] == "search"
    assert info["shift_px"]["own_left"] == [0.0, 0.0]
    read(r, S.battle_frame("sand", labels=CHICKEN, overlay=True, size=size), 1.25)
    assert r.report.summary()["overlay"] is True
    legacy = reader(size, calibrate=False)
    read(legacy, img, 0.0)
    assert "calib" not in legacy.report.summary() and legacy.report.towers_compact()["overlay"] is False
    assert "calib" not in legacy.towers_info()
