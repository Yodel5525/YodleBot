"""tools/tower_audit.py: the tower-skin audit of a recorded session (skin audit fix list, sec. 2).

Sessions are synthetic frames.jsonl files written with the recorder (scripts/vision/record.py)
in the formats the runners write today: tools/vision_replay.py's records (extra.replay.hud_report
= HudReport.summary(), built with the real HudReport) and scripts/vision_play.py's
(extra.towers_raw = HudReport.towers_compact(), battle_end and result records). One session is
read by the real HudReader from tests/pipeline/skin_synth.py frames. The fields later stages add
(calib, evidence, overlay) are passed as extra keys of the summary. Images for --sheet are
skin_synth frames. No device, no detector.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import tower_audit as TA  # noqa: E402
from scripts.vision import hud as H  # noqa: E402
from scripts.vision import record as R  # noqa: E402
from scripts.vision import templates as TP  # noqa: E402
from scripts.vision import tower_labels as TL  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from tests.pipeline import skin_synth as S  # noqa: E402

KEYS = VT.TOWER_KEYS
SIZE = S.OWNER_SIZE
DT = 0.25                                   # 4 frames a second
T0 = 10.0


def summary(t, *, frac=None, bars=None, reads=None, shift=None, present=None, held=(), tick=None,
            clock="read", extra=None) -> dict:
    """HudReport.summary() as HudReader writes it (everything full unless given)."""
    rep = H.HudReport(t=t)
    rep.clock = None if clock is None else H.ClockState(tick, VT.PHASE_REGULATION, 1.0, True, clock)
    rep.tower_frac = {**{k: 1.0 for k in KEYS}, **(frac or {})}
    rep.tower_bars = dict(bars or {})
    rep.tower_digits = {k: H.TowerRead(hp, conf) for k, (hp, conf) in (reads or {}).items()}
    rep.tower_shift = {**{k: [0.0, 0.0, TL.SOURCE_FIXED] for k in TL.PRINCESS_KEYS},
                       **{k: list(v) for k, v in (shift or {}).items()}}
    rep.tower_present = dict(present or {})
    rep.towers_held = list(held)
    rep.towers_live = True
    out = rep.summary()
    out.update(extra or {})                 # the fields later stages add
    return out


def compact(**kw) -> dict:
    """HudReport.towers_compact() for the same frame (vision_play's extra.towers_raw)."""
    s = summary(0.0, **kw)
    return {"reads": s["tower_reads"], "bars": s["tower_bars"], "frac": s["towers"],
            "shift": s["tower_shift"], "present": s["tower_present"]}


def dets_of(objs) -> list:
    return [VT.Detection(d.cls, 0.5, float(d.conf), tuple(d.box)) for d in objs]


class Writer:
    """A session written by VisionRecorder, as vision_replay ("replay") or vision_play ("play")
    write theirs."""

    def __init__(self, root: Path, name: str, kind: str = "replay", image_every: int = 0):
        self.kind = kind
        self.t = 0.0
        self.n = 0
        self.rec = R.VisionRecorder(root, session=name, image_every=image_every, image_format="npy",
                                    clock=lambda: self.t)
        self.path = self.rec.session_dir

    def frame(self, t, *, battle=1, screen="battle", tick=None, hud=None, crowns=(0, 0), hp=None,
              dets=(), det_ran=True, image=None, rgb=None, destroyed=None):
        self.t = t
        self.n += 1
        towers = {k: VT.TowerHp(*v) for k, v in (hp or {}).items()}
        vf = VT.VisionFrame(t_capture=t, tick=tick, hud=VT.Hud(crowns=tuple(crowns), towers=towers,
                                                               towers_destroyed=destroyed),
                            screen=screen, detections=dets_of(dets), frame_size=SIZE)
        extra = {"battle": battle, "state": "battle" if battle else "idle"}
        if self.kind == "replay":
            extra["replay"] = {"i": self.n, "image": image, "det": ({"n": len(dets)} if det_ran else None),
                               "hud_report": hud}
        else:
            extra["ms"] = {}
            if hud is not None:
                extra["towers_raw"] = hud
        self.rec.write(vf, rgb, force_image=rgb is not None, **extra)

    def log(self, kind, **data):
        self.rec.log(kind, **data)

    def close(self) -> Path:
        self.rec.close()
        return self.path


def battle_session(root: Path, name: str = "rep", *, kind: str = "replay", extra_at=None, king_zero: bool = False,
                   end_crowns=(1, 0), overlay_read: bool = False) -> Path:
    """One 40-s battle (4 frames a second, i 0-159 at t 10.0-49.75):
    * enemy_left / enemy_right labels raised 36 px, the detector's placement from frame 4;
    * enemy_right: damaged to 0.6 at i 40; raw zeros held at i 60-63 (its label box seen), 64-65
      (the detector not run: the badge) and 80-95 (its body box only: the label boxes stop at
      i 79, the body at i 95); absent from i 96, accepted zero at i 104 (present False);
    * our towers read 3052 by their digits, the enemy's by their bars;
    * a menu frame before, five end-screen frames after (the first two still in battle 1).
    extra_at(i) -> the later stages' fields for frame i (or None): a frame it flags as an overlay
    frame reads nothing (P1-2), unless overlay_read (then frame 20's enemy_left bar is read);
    king_zero: the enemy king reads 0 from i 120; end_crowns: the end screen's."""
    w = Writer(root, name, kind)
    w.frame(T0 - 1.0, battle=None, screen="menu", hud=None, det_ran=False)
    for i in range(160):
        t = T0 + i * DT
        tick = 100 + 5 * i
        frac, held, present = {}, [], {"enemy_left": True, "own_left": True, "own_right": True}
        bars = {"enemy_left": 1.0}
        dets = [S.label_det("enemy_left", S.SKIN_DV), S.body_det("enemy_left"), S.label_det("own_left"),
                S.label_det("own_right"), S.body_det("own_left"), S.body_det("own_right")]
        det_ran = True
        if i >= 40:
            frac["enemy_right"] = 0.6
        if i < 100:
            bars["enemy_right"] = 0.6 if i >= 40 else 1.0
        if i < 80:
            dets.append(S.label_det("enemy_right", S.SKIN_DV))
        if i < 96:
            dets.append(S.body_det("enemy_right"))
            present["enemy_right"] = True
        else:
            present["enemy_right"] = False
        if 60 <= i < 66 or 80 <= i < 96:
            held.append("enemy_right")
        if 64 <= i < 66:
            dets, det_ran = [], False
        if i >= 104:
            frac["enemy_right"] = 0.0
        if king_zero and i >= 120:
            frac["enemy_king"] = 0.0
        shift = {k: [0.0, S.SKIN_DV, TL.SOURCE_DETECTOR] for k in S.ENEMY} if i >= 4 else {}
        reads = {"own_left": (3052, 0.95), "own_right": (3052, 0.95)}
        extra = extra_at(i) if extra_at else None
        if extra and extra.get("overlay"):
            reads, bars = {}, ({"enemy_left": 0.9} if overlay_read and i == 20 else {})
        kw = dict(frac=frac, bars=bars, reads=reads, shift=shift, present=present, held=held, tick=tick,
                  extra=extra)
        crowns = (1, 0) if i >= 104 else (0, 0)
        crowns = (3, 0) if king_zero and i >= 120 else crowns
        hp = {"enemy_right": (0, 3052)} if i >= 104 else {"enemy_right": (1831 if i >= 40 else 3052, 3052)}
        if king_zero and i >= 120:
            hp["enemy_king"] = (0, 4824)
        if kind == "replay":
            hud = summary(t, **kw)
        else:
            hud = compact(**kw)
            hud.update(kw["extra"] or {})
        w.frame(t, tick=tick, hud=hud, crowns=crowns, hp=hp, dets=dets, det_ran=det_ran, image=f"{i:08d}.jpg")
    for j in range(5):
        w.frame(T0 + 40.0 + j * DT, battle=1 if j < 2 else None, screen="end", crowns=end_crowns, det_ran=False)
    if kind == "play":
        w.log("battle_end", battle=1, reason="end screen", tick=895, hud_crowns=[1, 0],
              towers_info={"destroyed_by": {"enemy_right": "detector"}})
        w.log("result", battle=1, crowns=list(end_crowns), source="end screen", outcome="win",
              hud_crowns=[1, 0])
    return w.close()


def new_fields(i: int) -> dict:
    """calib, evidence and overlay as the later stages write them: every princess label locked
    from frame 8 (the enemy's at dv -36), enemy_right's zero on label evidence, frames 20-22 the
    deploy overlay with nothing read."""
    calib = {k: {"calibrated": i >= 8, "du": 0.0, "dv": S.SKIN_DV if k.startswith("enemy") else 0.0,
                 "source": "badge"} for k in TL.PRINCESS_KEYS}
    out = {"calib": calib, "overlay": 20 <= i <= 22}
    if i >= 104:
        out["evidence"] = {"enemy_right": {"kind": "label"}}
    return out


def audit(path, *args) -> tuple[int, dict, list[str]]:
    lines: list[str] = []
    out = Path(path).parent / "audit.json"                  # next to the session (or root) folder
    code = TA.main([str(path), *args, "--json", str(out)], out=lines.append)
    return code, json.loads(out.read_text(encoding="utf-8")), lines


# ---------------------------------------------------------------------------
# today's replay fields
# ---------------------------------------------------------------------------
def test_a_replay_with_todays_fields(tmp_path):
    path = battle_session(tmp_path / "replays")
    s = TA.load_session(path)
    assert s.kind == "replay" and len(s.frames) == 1 + 160 + 5 and s.bad_lines == 0
    battles = TA.battles_of(s)
    assert list(battles) == [1] and len(battles[1]["frames"]) == 162 and len(battles[1]["end"]) == 5
    code, rep, lines = audit(path)
    assert code == TA.EXIT_OK and rep["schema"] == TA.SCHEMA and rep["checks"] == {}
    b = rep["sessions"][0]["battles"][0]
    assert b["battle_frames"] == 160 and (b["first_tick"], b["last_tick"]) == (100, 895)
    assert b["clock3_t"] == pytest.approx(0.5)                     # the third clock reading: frame 2
    assert b["recorded"] == {"calib": False, "evidence": False, "overlay": False, "confirmed": False, "holds": True,
                             "detections": True}
    el, er, ol = (b["towers"][k] for k in ("enemy_left", "enemy_right", "own_left"))
    # calibration from today's tower_shift: placed by the detector at frame 4
    assert el["calib"] == {"status": "placed", "recorded": False, "du": 0.0, "dv": -36.0, "source": "detector",
                           "locked_t": 1.0, "locked_tick": 120, "frames_to_lock": 5, "after_clock3_s": 0.5}
    assert ol["calib"]["status"] == "fixed" and "locked_t" not in ol["calib"]
    assert b["towers"]["enemy_king"]["calib"] is None
    # reads
    assert el["reads"] == {"digits": 0, "bars": 160, "none": 0}
    assert ol["reads"] == {"digits": 160, "bars": 0, "none": 0}
    assert er["reads"] == {"digits": 0, "bars": 100, "none": 60}
    # timeline: damaged at i 40 (10 s), zero at i 104 (26 s) with the detector's absence (inferred)
    assert er["first_damage"] == {"t": 10.0, "tick": 300, "frac": 0.6}
    assert er["zero"] == {"t": 26.0, "tick": 620, "evidence": "detector", "evidence_from": "inferred", "confirmed": None,
                          "present": False, "label_boxes_before": [0, 12], "revived_t": None}
    assert el["zero"] is None and el["first_damage"] is None
    # holds by kind, the body-only run (i 80-95: 3.75 s), body boxes 4 s after the last label box
    assert er["holds"] == {"label": 4, "body": 16, "badge": 2}
    assert er["body_only_s"] == pytest.approx(3.75) and er["body_after_label_s"] == pytest.approx(4.0)
    assert el["holds"] == {"label": 0, "body": 0, "badge": 0}
    # detector boxes at each tower
    assert el["boxes"]["frames"] == 158 and el["boxes"]["share"] == 1.0
    assert el["boxes"]["classes"] == {"tower-bar": 158, "PrincessTower": 158}
    assert er["boxes"]["with_box"] == 94 and er["boxes"]["classes"]["tower-bar"] == 78
    assert b["towers"]["enemy_king"]["boxes"]["with_box"] == 0
    # crowns: the HUD's 1-0 and the end screen's (5 frames)
    assert b["crowns"] == {"hud": [1, 0], "end": [1, 0], "end_frames": 5, "agree": True}
    assert b["overlay"] is None
    text = "\n".join(lines)
    assert "battle 1: 39.75 s, 160 battle frames" in text and "not recorded: calib, evidence, overlay" in text
    assert "    enemy_left   placed +0,-36 detector @1.0s/5f" in text and "    own_left     fixed " in text
    assert "26.0s t620 detector (inferred)" in text and "crowns: HUD 1-0, end screen 1-0 (5 frames): agree" in text


def test_checks_on_todays_fields(tmp_path):
    """Before the later stages: no calibration, overlay or evidence records, so calib, overlay and
    destroy fail; king and crowns pass."""
    path = battle_session(tmp_path / "replays")
    code, rep, lines = audit(path, "--check", "calib,destroy,king,overlay,crowns")
    assert code == TA.EXIT_CHECK and rep["ok"] is False
    c = rep["checks"]
    assert c["king"] == {"ok": True, "failures": []} and c["crowns"]["ok"] is True
    assert c["calib"]["failures"] == ["rep battle 1: no calibration records (the session predates label calibration)"]
    assert c["overlay"]["failures"] == ["rep battle 1: no overlay records"]
    assert c["destroy"]["failures"] == ["rep battle 1: enemy_right destroyed at 26.0 s without label evidence "
                                        "(detector, inferred)"]
    assert "check destroy: FAIL" in lines and "check king: PASS" in lines
    assert TA.main([str(path), "--check", "king,crowns"], out=lambda s: None) == TA.EXIT_OK


# ---------------------------------------------------------------------------
# the later stages' fields
# ---------------------------------------------------------------------------
def test_calibration_evidence_and_overlay_records_pass(tmp_path):
    path = battle_session(tmp_path / "replays", extra_at=new_fields)
    code, rep, lines = audit(path, "--check", ",".join(TA.CHECKS), "--expect-overlay", "00000020.jpg,00000022.jpg")
    assert code == TA.EXIT_OK and rep["ok"] and all(c["ok"] for c in rep["checks"].values())
    b = rep["sessions"][0]["battles"][0]
    el = b["towers"]["enemy_left"]["calib"]
    assert el == {"status": "calibrated", "recorded": True, "du": 0.0, "dv": -36.0, "source": "badge",
                  "locked_t": 2.0, "locked_tick": 140, "frames_to_lock": 9, "after_clock3_s": 1.5}
    assert b["towers"]["own_left"]["calib"]["status"] == "calibrated"
    z = b["towers"]["enemy_right"]["zero"]
    assert (z["evidence"], z["evidence_from"]) == ("label", "frame")
    assert b["overlay"] == {"frames": 3, "names": ["00000020.jpg", "00000021.jpg", "00000022.jpg"],
                            "read_on_flagged": []}
    assert b["towers"]["enemy_left"]["reads"] == {"digits": 0, "bars": 157, "none": 3}
    assert "    overlay: 3 frames skipped" in lines


def test_overlay_check_wants_nothing_read_on_flagged_frames(tmp_path):
    path = battle_session(tmp_path / "replays", extra_at=new_fields, overlay_read=True)
    code, rep, _ = audit(path, "--check", "overlay")
    assert code == TA.EXIT_CHECK                      # enemy_left's bar read on overlay frame 20
    assert rep["checks"]["overlay"]["failures"] == ["rep battle 1: towers read on overlay frames: enemy_left"]
    path = battle_session(tmp_path / "replays", "rep2", extra_at=new_fields)
    code, rep, _ = audit(path, "--check", "overlay", "--expect-overlay", "00000020.jpg,00000030.jpg")
    assert code == TA.EXIT_CHECK
    assert rep["checks"]["overlay"]["failures"] == ["00000030.jpg: not flagged as an overlay frame"]


def test_a_king_at_zero_and_crowns_that_differ_fail(tmp_path):
    path = battle_session(tmp_path / "replays", extra_at=new_fields, king_zero=True, end_crowns=(1, 1))
    code, rep, lines = audit(path, "--check", "king,crowns,destroy")
    assert code == TA.EXIT_CHECK
    assert rep["checks"]["king"]["failures"] == ["rep battle 1: enemy_king read 0 at 30.0 s (tick 700) "
                                                 "before the end screen"]
    assert rep["checks"]["crowns"]["failures"] == ["rep battle 1: HUD crowns 3-0 differ from the end screen's 1-1"]
    assert rep["checks"]["destroy"]["ok"]               # kings are the king check's
    assert "crowns: HUD 3-0, end screen 1-1 (5 frames): DIFFER" in "\n".join(lines)


def ending_session(root: Path, name: str, *, hud_crowns=(0, 0), end_crowns=(1, 0), absent_last=None,
                   calib: bool = True, zero=()) -> Path:
    """A 10-s battle (every tower calibrated, standing unless in `zero`) whose last battle frame
    records each princess tower's label absence (absent_last: key -> s; 0 by default), then five
    end-screen frames with `end_crowns`. The HUD's crowns are `hud_crowns` all along."""
    w = Writer(root, name)
    w.frame(T0 - 1.0, battle=None, screen="menu", hud=None, det_ran=False)
    n = 40
    for i in range(n):
        t = T0 + i * DT
        last = i == n - 1
        cal = {k: {"calibrated": True, "du": 0.0, "dv": 0.0, "source": "search",
                   **({"absent_s": (absent_last or {}).get(k, 0.0)} if last and (absent_last or {}).get(k) else {})}
               for k in TL.PRINCESS_KEYS}
        frac = {k: 0.0 for k in zero}
        hud = summary(t, tick=100 + 5 * i, frac=frac, extra={"calib": cal} if calib else None)
        w.frame(t, tick=100 + 5 * i, hud=hud, crowns=hud_crowns, hp={k: (0, 3052) for k in zero})
    for j in range(5):
        w.frame(T0 + n * DT + 3.0 + j * DT, battle=1 if j < 2 else None, screen="end", crowns=end_crowns, det_ran=False)
    return w.close()


@pytest.mark.parametrize("case, hud, end, absent, zero, ok", [
    ("overtime: the tower that ended it was never counted", (0, 0), (1, 0), {}, (), True),
    ("a 3-crown win: kings never fall in the HUD", (1, 0), (3, 0), {}, ("enemy_left",), True),
    ("a loss in overtime: our tower fell as it ended", (0, 0), (0, 1), {}, (), True),
    ("a fall the HUD missed mid-battle: its label gone 10 s", (0, 0), (1, 0),
     {"enemy_left": 10.0, "enemy_right": 10.0}, (), False),
    ("the HUD counted a crown the end screen did not show", (1, 0), (0, 0), {}, ("enemy_left",), False),
    ("two more than the HUD, one tower could fall", (0, 0), (2, 0), {"enemy_left": 9.0}, (), False),
], ids=["overtime", "three_crowns", "own_overtime", "missed_fall", "hud_ahead", "two_more"])
def test_the_crowns_check_accepts_what_the_battles_end_explains(tmp_path, case, hud, end, absent, zero, ok):
    """The tower whose fall ends a battle is never counted by the HUD (the battle screen goes on the
    next frame, before its label has been absent 2.5 s; a king never falls in the HUD), so Round 0
    (HUD 0-0, end screen 1-0 after overtime) failed the check the runbook asks to pass. Now the end
    screen may lead by 3 for a king and by one per standing tower whose label was seen in the last
    2.5 s; anything else still fails."""
    path = ending_session(tmp_path / "rep", "s", hud_crowns=hud, end_crowns=end, absent_last=absent, zero=zero)
    code, rep, lines = audit(path, "--check", "crowns")
    c = rep["sessions"][0]["battles"][0]["crowns"]
    assert rep["checks"]["crowns"]["ok"] is ok, (case, c)
    assert c["explained"] is ok and c["agree"] is False
    text = "\n".join(lines)
    if ok:
        assert ": explained (" in text and code == TA.EXIT_OK
    else:
        assert ": DIFFER (" in text and code == TA.EXIT_CHECK


def test_without_calibration_records_only_a_king_explains_a_difference(tmp_path):
    path = ending_session(tmp_path / "rep", "s", calib=False)
    code, rep, _ = audit(path, "--check", "crowns")
    assert code == TA.EXIT_CHECK and rep["sessions"][0]["battles"][0]["crowns"]["could_fall"] == {"own": [], "enemy": []}
    path = ending_session(tmp_path / "rep2", "s", calib=False, hud_crowns=(0, 0), end_crowns=(3, 0))
    assert audit(path, "--check", "crowns")[0] == TA.EXIT_OK


# ---------------------------------------------------------------------------
# vision_play sessions
# ---------------------------------------------------------------------------
def test_a_vision_play_session(tmp_path):
    path = battle_session(tmp_path / "play", "vision_play_x", kind="play")
    code, rep, lines = audit(path, "--check", "destroy,crowns")
    s = rep["sessions"][0]
    assert s["kind"] == "play" and code == TA.EXIT_CHECK
    b = s["battles"][0]
    er = b["towers"]["enemy_right"]
    assert er["holds"] is None and b["recorded"]["holds"] is False         # towers_compact has no holds
    assert (er["zero"]["evidence"], er["zero"]["evidence_from"]) == ("detector", "battle_end")
    assert b["towers"]["enemy_left"]["calib"]["status"] == "placed"
    assert b["clock3_t"] is None                                          # no clock source there
    assert b["crowns"]["result"] == {"crowns": [1, 0], "source": "end screen", "outcome": "win"}
    assert rep["checks"]["crowns"]["ok"] and not rep["checks"]["destroy"]["ok"]
    assert rep["checks"]["destroy"]["failures"] == ["vision_play_x battle 1: enemy_right destroyed at 26.0 s "
                                                    "without label evidence (detector, battle_end)"]
    text = "\n".join(lines)
    assert "not recorded: calib, evidence, overlay, holds" in text
    assert "26.0s t620 detector (battle_end)" in text and " n/a " in text
    assert "; result [1, 0] (end screen)" in text


# ---------------------------------------------------------------------------
# finding sessions, bad input, no battle
# ---------------------------------------------------------------------------
def test_sessions_under_a_root(tmp_path):
    root = tmp_path / "replays"
    a = battle_session(root, "a_old")
    b = battle_session(root, "b_new", extra_at=new_fields)
    (root / "sheets").mkdir()                                          # not a session
    os.utime(a / TA.FRAMES_FILE, (1_000_000, 1_000_000))
    os.utime(b / TA.FRAMES_FILE, (2_000_000, 2_000_000))
    assert TA.find_sessions(root) == [b]
    os.utime(a / TA.FRAMES_FILE, (3_000_000, 3_000_000))
    assert TA.find_sessions(root) == [a]                               # the newest frames.jsonl
    assert TA.find_sessions(root, every=True) == [a, b]
    assert TA.find_sessions(b / TA.FRAMES_FILE) == [b] and TA.find_sessions(b, every=True) == [b]
    assert TA.find_sessions(tmp_path / "nothing") == []
    code, rep, _ = audit(root, "--all", "--check", "calib")
    assert [s["name"] for s in rep["sessions"]] == ["a_old", "b_new"] and code == TA.EXIT_CHECK
    assert rep["checks"]["calib"]["failures"] == [
        "a_old battle 1: no calibration records (the session predates label calibration)"]
    out: list[str] = []
    assert TA.main([str(tmp_path / "nothing")], out=out.append) == TA.EXIT_INPUT and "no session" in out[0]
    out.clear()
    assert TA.main([str(root), "--check", "calib,skins"], out=out.append) == TA.EXIT_INPUT
    assert "unknown check(s) skins" in out[0]


def test_a_session_without_a_battle_fails_every_check(tmp_path):
    w = Writer(tmp_path, "menu_only")
    for i in range(5):
        w.frame(1.0 + i, battle=None, screen="menu", det_ran=False)
    with open(w.close() / TA.FRAMES_FILE, "a", encoding="utf-8") as fh:
        fh.write('{"kind": "frame", "i": 9, "frame": {"t_capture"')        # cut short by a crash
    code, rep, lines = audit(w.path, "--check", "king")
    assert code == TA.EXIT_CHECK and rep["sessions"][0]["battles"] == [] and rep["sessions"][0]["bad_lines"] == 1
    assert rep["checks"]["king"]["failures"] == ["menu_only: no battle"]
    assert "  no battle (no frame carries a battle number)" in lines


# ---------------------------------------------------------------------------
# the live reader's own records (HudReader on skin_synth frames)
# ---------------------------------------------------------------------------
def test_the_audit_reads_what_hudreader_writes(tmp_path):
    """HudReader.read / read_towers on the chicken-like frames (enemy labels 36 px up), the detector's
    label and body boxes as live; enemy_right, at half HP, falls at frame 20 (its label and boxes
    gone): its label's absence at the lock destroys it 2.5 s later (evidence "label", written
    in the frame's evidence record: skin audit P0-2). The records are written as
    vision_replay writes them. The reader calibrates each label (tower_calib): locked on its
    ninth frame (8 searches over 1 s), nothing read before, the gate's confirmed frames counted;
    the pre-calibration reader (calibrate=False) placed them from the detector's votes."""
    r = H.HudReader(TP.TemplateStore.memory(), {}, regions=S.REGIONS, towers_wait_for_clock=False)
    labels = {"enemy_left": S.Label(dv=S.SKIN_DV, digits=False),
              "enemy_right": S.Label(dv=S.SKIN_DV, frac=0.5, digits=False)}
    standing = S.battle_frame("sand", labels=labels, size=SIZE)
    fallen = S.battle_frame("sand", labels={**labels, "enemy_right": None}, size=SIZE)
    w = Writer(tmp_path, "live")
    for i in range(60):
        t = 1.0 + i * 0.125
        gone = i >= 20
        img = fallen if gone else standing
        dets = [S.label_det(k, S.SKIN_DV) for k in S.ENEMY if not (gone and k == "enemy_right")]
        dets += [S.label_det(k) for k in S.OWN] + [S.body_det(k) for k in S.ENEMY + S.OWN
                                                   if not (gone and k == "enemy_right")]
        hud = r.read(img, t, towers=False)
        r.read_towers(img, t, hud, r.tower_evidence(dets, SIZE))
        w.frame(t, hud=r.report.summary(), crowns=hud.crowns,
                hp={k: (v.hp, v.max) for k, v in hud.towers.items()}, dets=dets)
    code, rep, lines = audit(w.close(), "--check", "crowns,king,calib,destroy")
    b = rep["sessions"][0]["battles"][0]
    assert code == TA.EXIT_OK and b["recorded"]["holds"] and b["recorded"]["detections"] and b["recorded"]["calib"]
    for k in S.ENEMY:
        c = b["towers"][k]["calib"]
        assert (c["status"], c["source"], c["frames_to_lock"], c["votes"], c["locked_t"]) == (
            "calibrated", "search", 9, 9, 1.0), k
        assert abs(c["dv"] - S.SKIN_DV) <= 2.0 and abs(c["du"]) <= 2.0, k
    for k in S.OWN:
        c = b["towers"][k]["calib"]
        assert c["status"] == "calibrated" and abs(c["dv"]) <= 2.0 and c["gate"] == {"confirmed": 52,
                                                                                     "unconfirmed": 0}, k
    assert b["towers"]["enemy_left"]["reads"] == {"digits": 0, "bars": 52, "none": 8}    # none before the lock
    assert b["towers"]["enemy_left"]["zero"] is None
    er = b["towers"]["enemy_right"]
    assert er["calib"]["gate"] == {"confirmed": 12, "unconfirmed": 40}       # frames 8-19 confirmed, then gone
    assert er["first_damage"]["frac"] == pytest.approx(0.5, abs=0.01) and 1.0 <= er["first_damage"]["t"] <= 2.0
    z = er["zero"]
    assert z is not None and (z["evidence"], z["evidence_from"], z["present"]) == ("label", "frame", False)
    assert 20 * 0.125 + 2.5 <= z["t"] <= 20 * 0.125 + 3.5                # the label's 2.5 s absence
    assert z["label_boxes_before"][0] > 0                                 # its label boxes until frame 19
    assert r.towers.destroyed_by == {"enemy_right": "label"} and r.towers.confirmed() == {"enemy_right"}
    assert b["crowns"]["hud"] == [1, 0]
    assert any(line.lstrip().startswith("enemy_left") and "calibrated" in line and "search @1.0s/9f gate 100%" in line
               for line in lines)
    assert r.towers_info()["calib"]["enemy_left"]["calibrated"] is True


def test_the_audit_reads_vision_plays_compact_calibration(tmp_path):
    """vision_play records HudReport.towers_compact() (extra.towers_raw): its calib fields carry
    no locked_t, so the lock is the first frame of the battle's last calibrated run. enemy_left's
    label lies 200 px up (outside the search window): uncalibrated, and --check calib names it."""
    r = H.HudReader(TP.TemplateStore.memory(), {}, regions=S.REGIONS, towers_wait_for_clock=False)
    labels = {"enemy_left": S.Label(dv=-200.0, digits=False), "enemy_right": S.Label(dv=S.SKIN_DV, digits=False)}
    img = S.battle_frame("snow", labels=labels, size=SIZE)
    w = Writer(tmp_path, "play", "play")
    for i in range(80):
        t = 1.0 + i * 0.125
        hud = r.read(img, t, towers=False)
        r.read_towers(img, t, hud, None)
        w.frame(t, hud=r.report.towers_compact(), crowns=hud.crowns,
                hp={k: (v.hp, v.max) for k, v in hud.towers.items()}, det_ran=False)
    code, rep, lines = audit(w.close(), "--check", "calib")
    b = rep["sessions"][0]["battles"][0]
    assert rep["sessions"][0]["kind"] == "play" and code == TA.EXIT_CHECK
    er = b["towers"]["enemy_right"]["calib"]
    assert (er["status"], er["source"], er["frames_to_lock"], er["locked_t"]) == ("calibrated", "search", 9, 1.0)
    assert abs(er["dv"] - S.SKIN_DV) <= 2.0 and er["gate"] == {"confirmed": 72, "unconfirmed": 0}
    assert er["state"] == "locked" and "votes" not in er                 # no frames in the compact record
    el = b["towers"]["enemy_left"]["calib"]
    assert el["status"] == "uncalibrated" and b["towers"]["enemy_left"]["reads"]["bars"] == 0
    assert rep["checks"]["calib"]["failures"] == ["play battle 1: enemy_left not calibrated (expired)"]
    assert any(line.lstrip().startswith("enemy_left") and "uncalibrated" in line for line in lines)


def test_the_audit_reads_the_destruction_evidence_and_the_overlay(tmp_path):
    """HudReader's later records (skin audit P0-2, P1-2) as vision_play writes them (the compact
    record and the frame's hud with its towers_destroyed): enemy_right falls (label evidence,
    confirmed), the frames of a card held (the overlay) and of the bot's own tap read nothing and
    are counted by why. The same battle read the old way (calibrate=False, no detector) destroys
    enemy_right without label evidence and outside the confirmed set: --check destroy fails."""
    labels = {"enemy_left": S.Label(digits=False), "enemy_right": S.Label(frac=0.3, digits=False)}
    standing = S.battle_frame("sand", labels=labels, size=SIZE)          # default skin: the old boxes see it
    held = S.battle_frame("sand", labels=labels, overlay=True, size=SIZE)
    fallen = S.battle_frame("sand", labels={**labels, "enemy_right": None}, size=SIZE)

    def run(r, name):
        w = Writer(tmp_path, name, "play")
        for i in range(80):
            t = 1.0 + i * 0.125
            img = held if 12 <= i < 16 else fallen if i >= 24 else standing
            if i == 18:
                r.note_tap(t)                        # the bot's tap pair: frames 18-28
            hud = r.read(img, t, towers=False)
            r.read_towers(img, t, hud, None)
            w.frame(t, hud=r.report.towers_compact(), crowns=hud.crowns, destroyed=hud.towers_destroyed,
                    hp={k: (v.hp, v.max) for k, v in hud.towers.items()}, det_ran=False)
        return w.close()

    new = H.HudReader(TP.TemplateStore.memory(), {}, regions=S.REGIONS, towers_wait_for_clock=False)
    code, rep, lines = audit(run(new, "new"), "--check", "destroy,overlay,king")
    b = rep["sessions"][0]["battles"][0]
    z = b["towers"]["enemy_right"]["zero"]
    assert code == TA.EXIT_OK and (z["evidence"], z["evidence_from"], z["confirmed"]) == ("label", "frame", True)
    assert b["towers"]["enemy_right"]["absent_max_s"] >= 2.5 and b["recorded"]["confirmed"]
    assert b["overlay"]["frames"] == 15 and b["overlay"]["why"] == {"mirror": 4, "tap": 11}
    assert b["overlay"]["read_on_flagged"] == []
    assert any("overlay: 15 frames skipped (mirror 4, tap 11)" in line for line in lines)
    assert any(line.lstrip().startswith("enemy_right") and "label absent" in line for line in lines)
    old = H.HudReader(TP.TemplateStore.memory(), {}, regions=S.REGIONS, towers_wait_for_clock=False,
                      calibrate=False, place_labels=False, search_labels=False, skip_overlay=False)
    code, rep, lines = audit(run(old, "old"), "--check", "destroy")
    z = rep["sessions"][0]["battles"][0]["towers"]["enemy_right"]["zero"]
    assert code == TA.EXIT_CHECK and z["evidence"] == "hud" and z["confirmed"] is False
    assert rep["checks"]["destroy"]["failures"][0].startswith("old battle 1: enemy_right destroyed at")
    assert any(line.lstrip().startswith("enemy_right") and "unconfirmed" in line for line in lines)


# ---------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------
def test_pick_sheet_frames():
    """One image frame per 10-s mark, and up to three image frames each side of a zero within 2 s
    (labelled for the zero, also when a mark picked them)."""
    frames = [TA.FrameView(i=i, t=100.0 + i * 0.5, battle=1, screen="battle", tick=i) for i in range(80)]
    every_third = TA.pick_sheet_frames(frames, [120.0], 10.0, 2.0, lambda f: f.i % 3 == 0)
    assert [(f.i, why) for f, why in every_third] == [
        (0, "0s"), (21, "10s"), (36, "zero -2.0s"), (39, "zero -0.5s"), (42, "zero +1.0s"), (60, "30s")]
    every = TA.pick_sheet_frames(frames, [120.0], 10.0, 2.0, lambda f: True)
    assert [(f.i, why) for f, why in every if why.startswith("zero")] == [
        (37, "zero -1.5s"), (38, "zero -1.0s"), (39, "zero -0.5s"), (40, "zero +0.0s"), (41, "zero +0.5s"),
        (42, "zero +1.0s")]
    assert [why for _, why in every if not why.startswith("zero")] == ["0s", "10s", "30s"]
    assert TA.pick_sheet_frames(frames, [120.0], 10.0, 2.0, lambda f: False) == []


def test_sheets_of_a_vision_play_session(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    w = Writer(tmp_path, "play_imgs", kind="play")
    img = S.battle_frame("forest", labels={k: S.Label(dv=S.SKIN_DV) for k in S.ENEMY}, size=SIZE)
    for i in range(48):                                        # 24 s at 2 frames a second
        t = 5.0 + i * 0.5
        frac = {"enemy_right": 0.0} if i >= 30 else {}
        hud = compact(frac=frac, bars={"enemy_left": 1.0},
                      shift={k: [0.0, S.SKIN_DV, TL.SOURCE_DETECTOR] for k in S.ENEMY})
        w.frame(t, tick=100 + 10 * i, hud=hud, rgb=img if i % 4 == 0 else None)
    path = w.close()
    out_dir = tmp_path / "sheets"
    lines: list[str] = []
    assert TA.main([str(path), "--sheet", str(out_dir)], out=lines.append) == TA.EXIT_OK
    sheets = sorted(out_dir.glob("*.png"))
    assert [p.name for p in sheets] == ["play_imgs_b1_p1.png"]
    with Image.open(sheets[0]) as im:
        assert im.size[0] == TA.SHEET_COLS * TA.SHEET_TILE_W
    assert f"sheet -> {sheets[0]}" in lines
    # the frames picked: 0, 10 and 20 s, and the images within 2 s of the zero at 15 s
    s = TA.load_session(path)
    a = TA.audit_session(s)
    frames = [f for f in TA.battles_of(s)[1]["frames"]]
    zeros = [frames[0].t + a["battles"][0]["towers"]["enemy_right"]["zero"]["t"]]
    picks = TA.pick_sheet_frames(frames, zeros, 10.0, 2.0, lambda f: TA.image_path(s, f) is not None)
    assert [(round(f.t - 5.0, 1), why) for f, why in picks] == [
        (0.0, "0s"), (10.0, "10s"), (14.0, "zero -1.0s"), (16.0, "zero +1.0s"), (20.0, "20s")]


def test_a_replays_images_come_from_its_input_recording(tmp_path):
    rec = tmp_path / "recordings" / "capture_x"
    (rec / "images").mkdir(parents=True)
    (rec / "images" / "00000004.jpg").write_bytes(b"jpg")
    w = Writer(tmp_path / "replays", "capture_x_replay")
    w.frame(1.0, hud=summary(1.0), image="00000004.jpg")
    w.frame(1.25, hud=summary(1.25), image="00000005.jpg")
    w.log("replay_summary", input=str(rec))
    s = TA.load_session(w.close())
    assert TA.image_path(s, s.frames[0]) == rec / "images" / "00000004.jpg"
    assert TA.image_path(s, s.frames[1]) is None
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "00000005.jpg").write_bytes(b"jpg")
    assert TA.image_path(s, s.frames[1], other) == other / "00000005.jpg"
