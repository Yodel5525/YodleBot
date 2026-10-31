"""The HUD never destroys a king during a battle (skin audit fix list P0-3).

A king shows its number only once it is damaged and has no bar to contradict a misread, and the
detector cannot see the enemy king's label (finding 4). The skin audit's script (verify/v1.py)
showed one misread of 1200 at conf 0.61 for 0.8 s, then nothing read, giving a king at 0 and
three crowns, and legality then opened the enemy king's no-deploy rectangle. Now
(scripts/vision/hud.py TowerHpTracker, "Kings"): in a battle a king keeps its last nonzero HP,
falls only on >= 3 readings in a row at conf >= 0.8 within 150 HP of each other, and its 0 and
the three crowns come from the end screen. No real frame: the end screen is skin_synth's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rs_decide as RD  # noqa: E402
from scripts import vision_play as VP  # noqa: E402
from scripts.vision import hud as H  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from tests.pipeline import skin_synth as S  # noqa: E402
from tests.pipeline import test_public_state as TPS  # noqa: E402

KK = "enemy_king"
KING = H.LEVEL11_TOWER_MAX_HP[KK]                          # 4824
HOG = TPS.CID["HogRider"]


def tracker(**kw) -> H.TowerHpTracker:
    return H.TowerHpTracker(prior_max_hp=H.LEVEL11_TOWER_MAX_HP, **kw)


def run(tr: H.TowerHpTracker, reads, *, dt: float = 0.1, t0: float = 0.0, conf: float = 0.9, key: str = KK,
        present=None) -> list[VT.TowerHp]:
    """One update per reading (an HP or None) dt apart, every reading at `conf`."""
    out = []
    for i, hp in enumerate(reads):
        out.append(tr.update({key: hp}, t0 + i * dt, {}, hp_conf={key: conf}, clock_running=None,
                             present=present)[key])
    return out


def state(towers: dict, destroyed=None) -> dict:
    hud = VT.Hud(towers=towers, towers_destroyed=destroyed)
    vf = VT.VisionFrame(t_capture=1.0, tick=1500, hud=hud, screen=VT.SCREEN_BATTLE, frame_size=S.OWNER_SIZE)
    return TPS.builder().state(vf)


def king_rect_open(st: dict) -> bool:
    """A Hog Rider of ours (team 1) at world tile (3, 8): inside the enemy king's rectangle, behind
    the left pocket (rows 11-14). Standing princesses' rectangles cover all of the king's, so it
    matters once one has fallen: the states below have enemy_left at 0."""
    return bool(RD.PureLegality(TPS.CATALOGUE).bind(st, 1)(0, HOG, 3500 * 18, 8500 * 18))


def pocket_open(st: dict) -> bool:
    """... and at (3, 12): in the left pocket."""
    return bool(RD.PureLegality(TPS.CATALOGUE).bind(st, 1)(0, HOG, 3500 * 18, 12500 * 18))


# ---------------------------------------------------------------------------
# the audit's scenario
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("present", [None, {KK: True}, {KK: False}])
def test_one_low_confidence_misread_then_nothing_keeps_the_king_full(present):
    """verify/v1.py: 1200 at conf 0.61 for 0.8 s, then 60 s unread: the king stays full, no
    crowns, whatever the detector says of its body."""
    tr = tracker()
    out = run(tr, [1200] * 8 + [None] * 600, conf=0.61, present=present)
    assert all(h == VT.TowerHp(KING, KING) for h in out)
    assert tr.crowns() == (0, 0) and tr.frac[KK] is None and tr.destroyed_by == {}


def test_five_confident_readings_damage_the_king_and_nothing_destroys_it():
    """5 readings of 2400 at 0.9 (4 a second): the king reads 2400, and never 0 after 60 s unread
    (the old rule: 2.5 s of nothing from <= 0.75 was a zero, and three crowns)."""
    tr = tracker()
    out = run(tr, [2400] * 5, dt=0.25)
    assert out[-1] == VT.TowerHp(2400, KING)
    later = run(tr, [None] * 240, dt=0.25, t0=1.25)
    assert all(h == VT.TowerHp(2400, KING) for h in later) and tr.crowns() == (0, 0)


def test_a_king_falls_only_on_three_agreeing_confident_readings():
    tr = tracker()                                           # two agree, then a far one: no fall
    assert run(tr, [3000, 3000, 2000, None, None] * 8, dt=0.25)[-1] == VT.TowerHp(KING, KING)
    tr = tracker()                                           # spread wider than 150 HP: no fall
    assert run(tr, [3000, 3200, 3400, 3600] * 6, dt=0.25)[-1] == VT.TowerHp(KING, KING)
    tr = tracker()                                           # under conf 0.8: no fall
    assert run(tr, [3000] * 20, dt=0.25, conf=0.79)[-1] == VT.TowerHp(KING, KING)
    tr = tracker()                                           # within 150 HP: falls, at the run's time
    out = run(tr, [3000, 3100, 2990, 3050, 3010], dt=0.25)
    assert out[2] == VT.TowerHp(KING, KING) and out[3].hp in range(2990, 3101)
    tr = tracker()                                           # frames without digits leave the run
    out = run(tr, [3000, None, 3000, None, 3000, None, 3000], dt=0.125)
    assert out[-1] == VT.TowerHp(3000, KING)


def test_a_damaged_king_keeps_its_hp_through_unread_frames_and_bar_zeros():
    tr = tracker()
    run(tr, [2400] * 6, dt=0.25)
    out = [tr.update({KK: None}, 2.0 + i * 0.25, {KK: 0.0}, clock_running=None)[KK] for i in range(40)]
    assert all(h == VT.TowerHp(2400, KING) for h in out) and tr.crowns() == (0, 0)


def test_hud_reader_never_destroys_a_king():
    """Through HudReader (two phases): a tower reader that misreads the king once at 0.61 and then
    reads nothing, a detector that never boxes the enemy king: full and no crowns for 60 s."""
    from tests.pipeline.test_tower_tracker import FRAME

    class KingReads:
        def __init__(self):
            self.n = 0

        def read_digits(self, rgb):
            self.n += 1
            return {KK: H.TowerRead(1200, 0.61, "1200") if self.n <= 8 else H.TowerRead(None, 0.0, None, "x")}

        def read_bars(self, rgb):
            return {}

    r = H.HudReader(H.TemplateStore.memory(), {}, regions=H.HudRegions.builtin(hand_centres=[(0.5, 0.9)] * 4),
                    towers_wait_for_clock=False)
    r.tower_reader = KingReads()
    from scripts.vision.tower_labels import TowerEvidence
    for i in range(480):
        hud = r.read(FRAME, i * 0.125, towers=False)
        hud = r.read_towers(FRAME, i * 0.125, hud, TowerEvidence())
        assert hud.towers[KK] == VT.TowerHp(KING, KING) and hud.crowns == (0, 0), i
    assert hud.towers_destroyed == () and r.towers.destroyed_by == {}


# ---------------------------------------------------------------------------
# legality
# ---------------------------------------------------------------------------
FULL = {k: VT.TowerHp(KING if k.endswith("king") else 3052, KING if k.endswith("king") else 3052)
        for k in VT.TOWER_KEYS}
FALLEN = {**FULL, "enemy_left": VT.TowerHp(0, 3052)}         # enemy_left fell (label evidence)


def test_legality_keeps_the_enemy_king_rectangle_closed():
    st = state(FALLEN, ("enemy_left",))
    assert pocket_open(st) and not king_rect_open(st)
    tr = tracker()                                           # the audit's scenario through the tracker
    hp = run(tr, [1200] * 8 + [None] * 100, conf=0.61)[-1]
    st = state({**FALLEN, KK: hp}, ("enemy_left",))
    assert not king_rect_open(st) and RD.destroyed_towers(st)[0] == (False, True, False)
    # a king read 0 by another reader without a confirmed set would open it (the old path) ...
    assert king_rect_open(state({**FALLEN, KK: VT.TowerHp(0, KING)}, None))
    # ... a 0 the HUD did not confirm does not
    st = state({**FALLEN, KK: VT.TowerHp(0, KING)}, ("enemy_left",))
    assert pocket_open(st) and not king_rect_open(st) and st["players"][0]["tower_hp"][0] == KING
    assert "enemy_king 0 unconfirmed: full" in st["perception"]["notes"]


def test_a_confirmed_princess_fall_activates_the_king_but_keeps_its_rectangle():
    st = state(FALLEN, ("enemy_left",))
    assert st["players"][0]["king_active"] is True and not king_rect_open(st)
    st = state(FALLEN, ())                                    # unconfirmed: nothing changes
    assert st["players"][0]["king_active"] is False and not pocket_open(st)


# ---------------------------------------------------------------------------
# the end screen gives the three crowns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("floor", ["forest", "sand", "snow"])
def test_an_end_screen_with_three_crowns_gives_3_x(floor):
    """The king fell: the HUD's crowns were 1-0 at most (it never reads a king 0); the end screen
    shows 3-1, and the result is 3-1 (a win), with the HUD's disagreement logged."""
    end = S.end_screen(floor, own=3, enemy=1, size=S.OWNER_SIZE)
    cues = VP.read_cues(end)
    assert cues.end_screen and not cues.hud_present and cues.overlay is not None and cues.overlay < 0.5
    assert VP.end_screen_crowns(end) == (3, 1)
    book = VP.ResultBook()
    book.open(1, "end_screen", 100.0, (1, 0), last_tick=3400)
    for i in range(4):
        book.end_frame(VP.end_screen_crowns(end), 101.0 + i)
    rec = book.finish()
    assert rec["crowns"] == [3, 1] and rec["source"] == "end screen" and rec["outcome"] == "win"
    assert rec["crowns_disagree"] == {"hud": [1, 0], "end": [3, 1]}
