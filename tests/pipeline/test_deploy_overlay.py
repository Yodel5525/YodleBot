"""The deploy overlay in any arena, and what the HUD skips on it (skin audit fix list P1-2).

While a card is selected the game tints the enemy half toward red (finding 7): the tint passes
the enemy bar colour and the end screen's red-banner cue, so an overlay frame read a damaged
tower's bar as 1.0 (each such frame restarted the fall's confirmation) and could pass for an end
screen. scripts/vision/overlay.OverlayDetector flags a frame by the tint against the mirrored
half, against the battle's own floor medians, or inside the bot's own tap window; HudReader then
reads no tower and counts nothing on it, and vision_play.ScreenCues.end_screen is False.

Every frame is drawn by tests/pipeline/skin_synth.py (7 floors, both sizes); no real frame.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import vision_play as VP  # noqa: E402
from scripts.vision import hud as H  # noqa: E402
from scripts.vision import overlay as OV  # noqa: E402
from scripts.vision import tower_labels as TL  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from tests.pipeline import skin_synth as S  # noqa: E402

PRINCESS = TL.PRINCESS_KEYS
CHICKEN = {k: S.Label(dv=S.SKIN_DV if k in S.ENEMY else 0.0) for k in PRINCESS}
SIZE = S.OWNER_SIZE
DT = 0.125
SIZE_IDS = ["1080", "747"]
WHOLE = (0.0, 0.0, 1080.0, 1920.0)


@lru_cache(maxsize=None)
def frame(floor: str = "default", size=SIZE, *, overlay: bool = False, labels: tuple = (),
          flyers: tuple = ()) -> np.ndarray:
    img = S.battle_frame(floor, labels={**CHICKEN, **dict(labels)}, overlay=overlay, flyers=list(flyers), size=size)
    img.setflags(write=False)
    return img


def tinted_all(img: np.ndarray) -> np.ndarray:
    """The overlay's blend over the whole frame: the mirror sees nothing (both halves alike)."""
    return S.deploy_overlay(np.array(img), box=WHOLE)


def patch_flyers(share: float) -> tuple:
    """Flying-unit boxes (px at 1080) over `share` of the overlay's patches, spread over the half."""
    centres = OV.patch_centres()
    step = max(1, int(round(1.0 / share)))
    return tuple((x - 16.0, y - 16.0, x + 16.0, y + 16.0) for i, (x, y) in enumerate(centres) if i % step == 0)


def history(det: OV.OverlayDetector, img, n: int = 5, t0: float = 0.0) -> None:
    for i in range(n):
        assert not det.update(img, t0 + i * DT).on


@lru_cache(maxsize=1)
def store():
    return S.digit_store(SIZE)


def reader(**kw) -> H.HudReader:
    return H.HudReader(store(), {}, regions=S.REGIONS, towers_wait_for_clock=False, **kw)


def read(r: H.HudReader, img, t, dets=None) -> VT.Hud:
    hud = r.read(img, t, towers=False)
    return r.read_towers(img, t, hud, None if dets is None else r.tower_evidence(dets, SIZE))


# ---------------------------------------------------------------------------
# flagged on every floor; not flagged otherwise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", S.SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("floor", S.FLOOR_NAMES)
def test_the_overlay_is_flagged_on_every_floor(floor, size):
    plain, tinted = frame(floor, size), frame(floor, size, overlay=True)
    det = OV.OverlayDetector()
    first = det.update(tinted, 0.0)                         # the battle's first frame: the mirror
    assert first.on and first.why == "mirror" and first.median is None and first.mirror > 0.9
    det = OV.OverlayDetector()
    history(det, plain)
    blind = tinted_all(plain)                               # our half tinted too: the mirror is blind
    r = det.update(blind, 1.0)
    assert OV.mirror_tint(blind) < 0.1 and r.on and r.why == "median" and r.median > 0.9, (floor, r)
    assert not det.update(plain, 1.125).on                   # and off again


@pytest.mark.parametrize("floor", S.FLOOR_NAMES)
def test_units_over_30_percent_of_the_patches(floor):
    covered = frame(floor, flyers=patch_flyers(0.3))
    covered_ov = frame(floor, overlay=True, flyers=patch_flyers(0.3))
    det = OV.OverlayDetector()
    history(det, frame(floor))
    for i in range(8):
        r = det.update(covered, 1.0 + i * DT)
        assert not r.on and r.mirror < 0.1 and r.median < 0.1, (floor, r)
    r = det.update(covered_ov, 2.0)
    assert r.on and 0.5 <= r.mirror <= 0.8, (floor, r)


@pytest.mark.parametrize("floor", ["default", "snow", "red", "forest"])
def test_a_slow_change_of_light_is_no_overlay(floor):
    plain = frame(floor).astype(np.float64)
    det = OV.OverlayDetector()
    for i in range(80):                                     # 10 s from full light to 75%, and back up
        f = 1.0 - 0.25 * (i / 40.0 if i < 40 else (80 - i) / 40.0)
        r = det.update(np.clip(plain * f, 0, 255).astype(np.uint8), i * DT)
        assert not r.on, (floor, i, r)
    assert det.flagged == 0 and det.frames == 80


def test_the_bots_own_tap_window():
    """note_tap(t): frames from t to t + OWN_TAP_PAIR_S + OWN_TAP_TAIL_S are overlay frames,
    whatever the pixels; a reset (a new battle) keeps the window."""
    det = OV.OverlayDetector()
    plain = frame()
    det.note_tap(10.0)
    assert not det.update(plain, 9.9).on
    for t in (10.0, 10.5, 11.0, 10.0 + OV.OWN_TAP_PAIR_S + OV.OWN_TAP_TAIL_S):
        r = det.update(plain, t)
        assert r.on and r.tap and r.why == "tap", t
    assert not det.update(plain, 11.4).on
    det.note_tap(20.0, 20.2)
    det.reset()
    assert det.update(plain, 20.1).on and not det.update(plain, 20.3).on
    assert det.update(np.zeros((4, 4, 3), dtype=np.uint8), 20.1).on      # too small for pixels: the window


# ---------------------------------------------------------------------------
# what HudReader skips on overlay frames
# ---------------------------------------------------------------------------
def test_overlay_frames_read_no_tower_and_change_nothing():
    k = "enemy_left"
    half = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.5)),))
    half_ov = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.5)),), overlay=True)
    r = reader()
    for i in range(16):
        read(r, half, i * DT)
    before = dict(r.towers.frac)
    for i in range(16, 40):
        hud = read(r, half_ov, i * DT)
        rep = r.report
        assert rep.overlay is True and rep.overlay_info["on"] and rep.overlay_info["mirror"] > 0.9
        assert all(v.hp is None and v.reason == "deploy overlay" for v in rep.tower_digits.values())
        assert all(v is None for v in rep.tower_bars.values())
        assert all(c.get("confirmed") is None and "seen" not in c for c in rep.calib.values())
        assert r.towers.frac == before and hud.towers[k] == VT.TowerHp(int(round(before[k] * 3052)), 3052)
    s = r.report.summary()
    assert s["overlay"] is True and s["overlay_info"]["on"] is True and "tower_reads" in s
    assert r.report.towers_compact()["overlay"] is True and r.report.towers_compact()["bars"] == {}
    info = r.towers_info()
    assert info["overlay_frames"] == [24, 40]
    hud = read(r, half, 40 * DT)
    assert r.report.overlay is False and r.report.tower_bars[k] == pytest.approx(0.5, abs=0.02)


def old_reader(**kw) -> H.HudReader:
    """The fixed boxes, no digit templates (bars only, as before the PC's harvest), no skip."""
    return H.HudReader(H.TemplateStore.memory(), {}, regions=S.REGIONS, towers_wait_for_clock=False,
                       calibrate=False, place_labels=False, search_labels=False, **kw)


def default_labels(k: str, frac: float) -> tuple:
    return tuple((x, S.Label(frac=frac if x == k else 1.0)) for x in PRINCESS)


def test_without_the_skip_an_overlay_frame_reads_a_half_bar_full():
    """The defect: read as before (skip_overlay=False), the tinted floor around a half bar reads
    1.0. (The calibrated reader skips such a frame even then: its own stateless check.)"""
    k = "enemy_left"
    r = old_reader(skip_overlay=False)
    read(r, frame(labels=default_labels(k, 0.5)), 0.0)
    assert r.report.tower_bars[k] == pytest.approx(0.5, abs=0.02) and r.report.overlay is None
    read(r, frame(labels=default_labels(k, 0.5), overlay=True), DT)
    assert r.report.tower_bars[k] == 1.0
    cal = reader(skip_overlay=False)
    for i in range(12):
        read(cal, frame(), i * DT)
    read(cal, frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.5)),), overlay=True), 12 * DT)
    assert cal.report.overlay is True and cal.report.tower_bars[k] is None


def fall_time(r: H.HudReader, half, half_ov, k: str, warm, t0: int = 12) -> float | None:
    """Full for t0 frames, then the tower at 0.5 with the overlay on 2 frames in 5 (cards played
    every second): seconds until the fall is accepted (None: never in 6 s)."""
    for i in range(t0):
        read(r, warm, i * DT)
    for i in range(t0, t0 + 48):
        hud = read(r, half_ov if i % 5 in (1, 3) else half, i * DT)
        if hud.towers[k].hp is not None and hud.towers[k].hp < 2000:
            return (i - t0) * DT
    return None


def test_damage_is_confirmed_with_the_overlay_on_40_percent_of_frames():
    """A tower at 0.5 with the overlay on 40% of frames: its fall is accepted within 1 s
    (calibrated, and the old reader with the skip). Without the skip the old reader's overlay
    frames read it full, each restarting the fall's 0.6 s, and it never falls."""
    k = "enemy_left"
    lab = ((k, S.Label(dv=S.SKIN_DV, frac=0.5)),)
    r = reader()
    t = fall_time(r, frame(labels=lab), frame(labels=lab, overlay=True), k, frame())
    assert t is not None and t <= 1.0 and r.towers.frac[k] == pytest.approx(0.5, abs=0.02)
    half, half_ov = frame(labels=default_labels(k, 0.5)), frame(labels=default_labels(k, 0.5), overlay=True)
    warm = frame(labels=default_labels(k, 1.0))
    old = old_reader()
    t = fall_time(old, half, half_ov, k, warm)
    assert t is not None and t <= 1.0
    before = old_reader(skip_overlay=False)
    assert fall_time(before, half, half_ov, k, warm) is None and before.towers.frac[k] == 1.0


def test_the_label_absence_clock_pauses_on_overlay_frames():
    """A tower at 0.3 whose label goes while a card is held for 4 s: the overlay frames are not
    counted, so it is destroyed 2.5 s of plain frames later, not during the overlay."""
    k = "enemy_left"
    low = frame(labels=((k, S.Label(dv=S.SKIN_DV, frac=0.3)),))
    gone_ov = frame(labels=((k, None),), overlay=True)
    gone = frame(labels=((k, None),))
    r = reader()
    seq = [frame()] * 10 + [low] * 12 + [gone_ov] * 32 + [gone] * 40
    huds = [read(r, img, i * DT) for i, img in enumerate(seq)]
    first = next(i for i, h in enumerate(huds) if h.towers[k].hp == 0)
    assert 54 + 20 <= first <= 54 + 24 and r.towers.destroyed_by[k] == "label"
    assert all(h.towers[k].hp == 916 for h in huds[22:54])


def test_the_runner_notes_its_own_taps_as_overlay_frames():
    """LivePerception.note_own_tap (vision_play: right after a tap pair is sent) opens the HUD's
    overlay window: the frames captured until the tile tap is done read no tower."""
    from scripts.vision import events as EV
    from tests.pipeline import test_vision_play as TVP
    r = reader()
    p = VP.LivePerception(hud_reader=r, tracker=None,
                          deriver=EV.EventDeriver(EV.EventTables({}), own_deck=range(8)),
                          geometry=TVP.GEOMETRY, detector=None)
    plain = frame()
    for i in range(12):
        read(r, plain, i * DT)
    p.note_own_tap(2, 300, slot=0, aim_mt=(3000.0, 20000.0), t=12 * DT)
    read(r, plain, 12 * DT + 0.5)
    assert r.report.overlay is True and r.report.overlay_info["tap"] is True
    assert r.report.tower_bars["enemy_left"] is None
    read(r, plain, 12 * DT + 1.5)
    assert r.report.overlay is False and r.report.tower_bars["enemy_left"] == 1.0


# ---------------------------------------------------------------------------
# the end screen
# ---------------------------------------------------------------------------
def overlay_like_end(floor: str) -> np.ndarray:
    """An overlay battle frame whose blue-banner and OK-button boxes show blue (an icy floor, a
    blue unit, card art): the overlay passes the red-banner cue, so all three cues pass."""
    img = np.array(frame(floor, overlay=True))
    for box in (VP.END_BLUE_BANNER, VP.END_OK_BUTTON):
        S.fill_box(img, (box[0] * 1080, box[1] * 1920, box[2] * 1080, box[3] * 1920), S.END_BLUE)
    return img


@pytest.mark.parametrize("floor", S.FLOOR_NAMES)
def test_an_overlay_frame_is_no_end_screen(floor):
    img = overlay_like_end(floor)
    cues = VP.read_cues(img)
    assert min(cues.red_banner, cues.blue_banner, cues.ok_button) >= VP.END_MIN       # the old test passes
    assert cues.overlay is not None and cues.overlay >= OV.OVERLAY_SHARE and not cues.end_screen
    assert "overlay" in cues.to_dict()
    assert "overlay" not in VP.read_cues(frame(floor)).to_dict()                     # not measured in battle
    end = S.end_screen(floor, own=1, size=SIZE)
    cues = VP.read_cues(end)
    assert cues.end_screen and cues.overlay < 0.1, (floor, cues)
