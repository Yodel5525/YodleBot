"""V01: arena geometry, tap mapping and the 9:16 frame check (scripts/vision/geometry.py).

Ties the default geometry to the code the live bot used before it: the
author's pixel fit (scripts/brain/arena.py, scripts/gamestate/live_play.py)
and the own-frame convention (tools/rs_eval.to_world, live_play.screen_tile,
tools/rs_rows.own). Those files import the engine or the memory reader, so the
functions are compiled out of their source with ast instead of imported.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import geometry as G  # noqa: E402

FULL = (1080, 1920)

# royalegym 0.1.25 action.py:817-820 (own frame of the seat; for Red, team 1,
# world = (17 - gx, 31 - gy)).
UI_BUTTON_TILES = {
    "right": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "left": ((15, 0), (16, 0), (16, 1), (17, 0), (17, 1)),
}


def _function_from(path: Path, name: str):
    """One top-level function of a repo file, compiled alone (no module imports)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns: dict = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            return ns[name]
    pytest.skip(f"{name}() is no longer defined in {path.relative_to(ROOT)}")


def _literal_from(path: Path, names: tuple[str, ...]):
    """The literal assigned to `names` (one name or a tuple of names) in a repo file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        got = (t.id,) if isinstance(t, ast.Name) else tuple(
            e.id for e in getattr(t, "elts", []) if isinstance(e, ast.Name))
        if got == names:
            return ast.literal_eval(node.value)
    return None


def _brain_arena():
    """scripts/brain/arena.py alone (the brain package __init__ pulls in the old bot)."""
    spec = importlib.util.spec_from_file_location("_brain_arena_v01", ROOT / "scripts" / "brain" / "arena.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the default reproduces the author's fit
# ---------------------------------------------------------------------------
def test_default_file_matches_builtin_constants():
    geo = G.ArenaGeometry.load(G.DEFAULT_GEOMETRY_PATH)
    ref = G.ArenaGeometry.builtin()
    assert geo.frame_size == FULL
    np.testing.assert_allclose(geo.homography_px, ref.homography_px, rtol=0, atol=1e-12)
    assert [geo.card_slot_px(i) for i in range(4)] == list(G.DEFAULT_CARD_SLOTS_PX)
    assert geo.ability_buttons == ref.ability_buttons
    assert geo.ability_button_places == ref.ability_button_places
    assert [m.name for m in geo.landmarks] == list(G.LANDMARKS_TILES)
    assert G.ArenaGeometry.default().frame_size == FULL


def test_default_reproduces_arena_to_pixels_on_all_576_tiles():
    arena = _brain_arena()
    geo = G.ArenaGeometry.default()
    worst = 0.0
    for sx in range(18):
        for sy in range(32):
            px, py = geo.tile_center_px(sx, sy, FULL)
            ax, ay = arena.to_pixels(sx, sy)
            worst = max(worst, abs(px - ax), abs(py - ay))
    assert worst <= 0.5 + 1e-9          # to_pixels rounds; the geometry does not


def test_default_matches_live_play_constants_and_card_centres():
    lp = ROOT / "scripts" / "gamestate" / "live_play.py"
    consts = _literal_from(lp, ("_X0", "_XS", "_Y0", "_YS"))
    if consts is None:
        pytest.skip("live_play.py no longer defines _X0, _XS, _Y0, _YS")
    x0, xs, y0, ys = consts
    screen_tile = _function_from(lp, "screen_tile")
    geo = G.ArenaGeometry.default()
    for tx in range(18):
        for ty in range(32):
            sx, sy = screen_tile(tx, ty)
            px, py = geo.tile_center_px(sx, sy, FULL)
            assert abs(px - (x0 + xs * sx)) < 1e-6 and abs(py - (y0 + ys * sy)) < 1e-6
            # live_play taps (round(_X0 + _XS * sx), round(_Y0 + _YS * sy)); ties may round apart
            tap = geo.tap_px(tx, ty, FULL)
            assert abs(tap[0] - round(x0 + xs * sx)) <= 1 and abs(tap[1] - round(y0 + ys * sy)) <= 1
    centres = _literal_from(lp, ("CARD_CENTRES",))
    if centres is not None:
        assert tuple(geo.card_slot_px(i, FULL) for i in range(4)) == tuple(tuple(c) for c in centres)


# ---------------------------------------------------------------------------
# own frame -> screen -> world agrees with the evaluator's to_world(team=1)
# ---------------------------------------------------------------------------
def test_own_screen_world_agree_with_rs_eval_to_world_team1():
    to_world = _function_from(ROOT / "tools" / "rs_eval.py", "to_world")
    screen_tile = _function_from(ROOT / "scripts" / "gamestate" / "live_play.py", "screen_tile")
    geo = G.ArenaGeometry.default()
    for tx in range(18):
        for ty in range(32):
            ox, oy = tx * 1000 + 500, ty * 1000 + 500
            want = to_world(ox, oy, 1)
            assert geo.model_tile_to_world(tx, ty) == want
            assert geo.own_to_world(ox, oy) == want
            assert geo.model_tile_to_screen_tile(tx, ty) == tuple(screen_tile(tx, ty))
            px, py = geo.tile_center_px(*geo.model_tile_to_screen_tile(tx, ty), FULL)
            wx, wy = geo.px_to_world(px, py, FULL)
            assert abs(wx - want[0]) < 1e-6 and abs(wy - want[1]) < 1e-6


def test_world_to_own_matches_rs_rows_own_team1():
    own = _function_from(ROOT / "tools" / "rs_rows.py", "own")
    sub = 18
    for x, y in ((0, 0), (500, 31500), (9000, 16000), (17999, 1), (3500, 25500)):
        assert G.ArenaGeometry.world_to_own(x, y) == own(x * sub, y * sub, 1, sub)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def _landmarks_mt():
    return [(x * 1000.0, y * 1000.0) for x, y in G.LANDMARKS_TILES.values()]


H_TRUE = np.array([[0.05, 0.002, 70.0], [0.001, 0.04, 150.0], [1e-7, 2e-6, 1.0]])


def test_exact_synthetic_fit_has_zero_reprojection_error():
    world = _landmarks_mt()
    px = G.project_points(H_TRUE, world)
    H = G.fit_homography(world, px)
    np.testing.assert_allclose(H, H_TRUE, rtol=1e-6, atol=1e-10)
    assert G.reprojection_errors(H, world, px).max() < 1e-6
    geo = G.ArenaGeometry.fit(world, px, FULL, names=list(G.LANDMARKS_TILES), name="synthetic")
    assert geo.meta["rms_px"] < 1e-6
    assert all(e is not None and e < 1e-6 for e in geo.meta["loo_px"])
    # everywhere on the board, not only at the landmarks
    xs, ys = np.meshgrid(np.linspace(0, 18000, 7), np.linspace(0, 32000, 9))
    u, v = geo.world_to_px(xs.ravel(), ys.ravel(), FULL)
    tu, tv = G._project(H_TRUE, xs.ravel(), ys.ravel())
    assert np.abs(u - tu).max() < 1e-6 and np.abs(v - tv).max() < 1e-6
    back = geo.px_to_world(u, v, FULL)
    assert np.abs(back[0] - xs.ravel()).max() < 1e-6 and np.abs(back[1] - ys.ravel()).max() < 1e-6


def test_noisy_fit_reports_fit_and_held_out_errors():
    world = _landmarks_mt()
    px = G.project_points(H_TRUE, world) + np.random.default_rng(0).normal(0.0, 1.0, (len(world), 2))
    geo = G.ArenaGeometry.fit(world, px, FULL)
    assert 0.1 < geo.meta["rms_px"] < 3.0
    assert len(geo.meta["loo_px"]) == len(world)
    assert all(e is not None and 0.0 <= e < 15.0 for e in geo.meta["loo_px"])
    with pytest.raises(G.GeometryError, match="RMS"):
        G.ArenaGeometry.fit(world, px, FULL, max_rms_px=0.01)


def test_fit_from_default_landmarks_reproduces_the_default():
    d = G.ArenaGeometry.default().to_dict()
    del d["homography_px"]
    geo = G.ArenaGeometry.from_dict(d)
    np.testing.assert_allclose(geo.homography_px, G.ArenaGeometry.default().homography_px,
                               rtol=1e-9, atol=1e-9)


def test_fit_rejects_bad_input():
    world = _landmarks_mt()
    px = G.project_points(H_TRUE, world)
    with pytest.raises(G.GeometryError, match="at least 4"):
        G.fit_homography(world[:3], px[:3])
    with pytest.raises(G.GeometryError):
        G.fit_homography(world, px[:-1])
    line = [(i * 1000.0, i * 1000.0) for i in range(5)]
    with pytest.raises(G.GeometryError, match="degenerate"):
        G.fit_homography(line, [(2.0 * x + 5, 2.0 * y + 7) for x, y in line])
    bad = np.array(px, dtype=float)
    bad[0, 0] = np.nan
    with pytest.raises(G.GeometryError, match="finite"):
        G.fit_homography(world, bad)
    # left and right swapped on screen: refused as mirrored
    default = G.ArenaGeometry.default()
    mirrored = [(1080.0 - default.world_to_px(x, y)[0], default.world_to_px(x, y)[1]) for x, y in world]
    with pytest.raises(G.GeometryError, match="mirrored"):
        G.ArenaGeometry.fit(world, mirrored, FULL)


# ---------------------------------------------------------------------------
# resolution independence and the 9:16 check
# ---------------------------------------------------------------------------
def test_pixels_scale_with_any_9_16_frame():
    geo = G.ArenaGeometry.default()
    for x, y in ((0, 0), (9000, 16000), (17999, 31999), (3500, 25500)):
        ax, ay = geo.world_to_px(x, y, FULL)
        bx, by = geo.world_to_px(x, y, (540, 960))
        assert abs(bx - ax / 2) < 1e-9 and abs(by - ay / 2) < 1e-9
        cx, cy = geo.world_to_px(x, y, (560, 996))          # MuMu's PrintWindow surface
        assert abs(cx - ax * 560 / 1080) < 1e-9 and abs(cy - ay * 996 / 1920) < 1e-9
        wx, wy = geo.px_to_world(cx, cy, (560, 996))
        assert abs(wx - x) < 1e-6 and abs(wy - y) < 1e-6
    # model tile (0, 31) -> screen tile (17, 31) -> world (17500, 31500) -> (970.5, 1455.3) at 1080
    assert geo.tap_px(0, 31, (540, 960)) == (485, 728)
    assert geo.model_tile_to_screen_tile(0, 31) == (17, 31)
    # arrays in, arrays out
    u, v = geo.world_to_px(np.array([0.0, 9000.0]), np.array([0.0, 16000.0]), FULL)
    assert u.shape == (2,) and abs(u[1] - geo.world_to_px(9000, 16000, FULL)[0]) < 1e-9
    with pytest.raises(G.GeometryError):
        geo.tap_px(18, 0)


@pytest.mark.parametrize("size", [(1080, 1920), (720, 1280), (540, 960), (560, 996), (1440, 2560)])
def test_check_frame_size_accepts_9_16_portrait(size):
    assert G.check_frame_size(*size) == size


@pytest.mark.parametrize("size", [(1080, 2400), (1920, 1080), (1080, 1080), (100, 178), (0, 0), (1080, 1800)])
def test_check_frame_size_refuses_everything_else(size):
    with pytest.raises(G.FrameSizeError):
        G.check_frame_size(*size)


def test_geometry_refuses_a_1080x2400_frame():
    geo = G.ArenaGeometry.default()
    with pytest.raises(G.FrameSizeError, match="9:16"):
        geo.world_to_px(9000, 16000, (1080, 2400))
    with pytest.raises(G.FrameSizeError):
        geo.tap_px(3, 20, (1080, 2400))
    with pytest.raises(G.FrameSizeError):
        G.ArenaGeometry(G.DEFAULT_HOMOGRAPHY_PX, (1080, 2400))


# ---------------------------------------------------------------------------
# ability buttons, card slots, ground points
# ---------------------------------------------------------------------------
def _cells(mask: np.ndarray) -> set[tuple[int, int]]:
    """(x, y) of the True cells of a [y, x] mask."""
    ys, xs = np.nonzero(mask)
    return {(int(x), int(y)) for x, y in zip(xs, ys)}


def test_ability_button_tiles_follow_royalegym_ui_button_tiles():
    geo = G.ArenaGeometry.default()
    for place, own in UI_BUTTON_TILES.items():
        assert geo.ability_buttons[place] == {(17 - gx, 31 - gy) for gx, gy in own}
    assert not geo.ability_button_mask(0).any()
    assert not geo.ability_button_mask(3).any()            # three buttons: unmeasured, nothing masked
    right_model = {(gx, 31 - gy) for gx, gy in UI_BUTTON_TILES["right"]}
    left_model = {(gx, 31 - gy) for gx, gy in UI_BUTTON_TILES["left"]}
    assert _cells(geo.ability_button_mask(1)) == right_model == {(0, 30), (0, 31), (1, 30), (1, 31)}
    assert _cells(geo.ability_button_mask(2)) == right_model | left_model
    assert _cells(geo.ability_button_mask(1, frame="screen")) == {(16, 30), (16, 31), (17, 30), (17, 31)}
    flat = geo.ability_button_mask(1).reshape(-1)          # the 576-way tile head, t = ty * 18 + tx
    assert flat.shape == (576,) and flat[31 * 18 + 0] and not flat[31 * 18 + 17]


def test_ability_button_tiles_match_installed_royalegym():
    try:
        import royalegym.action as action
    except Exception as exc:                 # not installed, or its data is missing
        pytest.skip(f"royalegym.action does not import ({type(exc).__name__}: {exc})")
    assert {k: tuple(map(tuple, v)) for k, v in action.UI_BUTTON_TILES.items()} == UI_BUTTON_TILES
    assert dict(action.UI_BUTTON_PLACES) == {1: ("right",), 2: ("left", "right")}


def test_ability_button_rects_and_card_slots_scale():
    geo = G.ArenaGeometry.default()
    x0, y0, x1, y1 = geo.ability_button_rect_px("right", FULL)
    for got, want in zip((x0, y0, x1, y1), (894, 1393, 996, 1476)):
        assert abs(got - want) <= 1
    cx, cy = geo.ability_button_px("right", FULL)
    assert x0 < cx < x1 and y0 < cy < y1
    lx0, _, lx1, _ = geo.ability_button_rect_px("left", FULL)
    assert abs(lx0 - 78) <= 1 and abs(lx1 - 231) <= 1
    for slot, (px, py) in enumerate(G.DEFAULT_CARD_SLOTS_PX):
        hx, hy = geo.card_slot_px(slot, (540, 960))
        assert abs(hx - px / 2) <= 0.5 and abs(hy - py / 2) <= 0.5
    with pytest.raises(G.GeometryError):
        geo.card_slot_px(4)


def test_ground_point_box_bottom_flyer_offset_and_class_override():
    geo = G.ArenaGeometry.default()
    cx, cy = geo.tile_center_px(5, 20, FULL)
    box = (cx - 20.0, cy - 60.0, cx + 20.0, cy)
    x, y = geo.ground_point(box, FULL)
    assert abs(x - 5500) < 1e-6 and abs(y - 20500) < 1e-6
    xf, yf = geo.ground_point(box, FULL, flying=True)
    assert abs(xf - 5500) < 1e-6 and abs(yf - (20500 + geo.ground.flying.dy_mt)) < 1e-6
    gx, gy = geo.ground_px(box, FULL)
    assert abs(gx - cx) < 1e-6 and abs(gy - cy) < 1e-6
    custom = G.ArenaGeometry(G.DEFAULT_HOMOGRAPHY_PX, FULL, ground=G.GroundOffsets(
        classes={"Balloon": G.GroundOffset(anchor_y=0.5, dx_mt=0.0, dy_mt=250.0)}))
    bx, by = custom.ground_point(box, FULL, cls="Balloon", flying=True)
    wx, wy = custom.px_to_world(cx, cy - 30.0, FULL)
    assert abs(bx - wx) < 1e-6 and abs(by - (wy + 250.0)) < 1e-6


def test_arena_rect_and_board_tests():
    geo = G.ArenaGeometry.default()
    for got, want in zip(geo.arena_rect_px(FULL), (78, 151, 996, 1476)):
        assert abs(got - want) <= 1
    for got, want in zip(geo.arena_rect_px(FULL, margin_mt=1000), (27, 109, 1047, 1518)):
        assert abs(got - want) <= 1
    assert G.in_arena(0, 0) and G.in_arena(18000, 32000) and not G.in_arena(-1, 0)
    assert G.in_arena(-1, 0, margin_mt=10)
    assert G.on_water(9000, 16000) and G.on_water(0, 15000)
    assert not G.on_water(3500, 16000) and not G.on_water(14500, 16999)      # bridges
    assert not G.on_water(9000, 14999) and not G.on_water(9000, 17000)


# ---------------------------------------------------------------------------
# files and the command line
# ---------------------------------------------------------------------------
def test_save_load_round_trip(tmp_path):
    world = _landmarks_mt()
    geo = G.ArenaGeometry.fit(world, G.project_points(H_TRUE, world), FULL, names=list(G.LANDMARKS_TILES))
    path = geo.save(tmp_path / "sub" / "geo.json")
    back = G.ArenaGeometry.load(path)
    np.testing.assert_allclose(back.homography_px, geo.homography_px, rtol=1e-12, atol=1e-15)
    assert back.frame_size == geo.frame_size
    assert [back.card_slot_px(i) for i in range(4)] == [geo.card_slot_px(i) for i in range(4)]
    assert back.ability_buttons == geo.ability_buttons
    assert back.ground == geo.ground
    assert [m.name for m in back.landmarks] == list(G.LANDMARKS_TILES)
    assert back.meta["rms_px"] == pytest.approx(geo.meta["rms_px"])
    with pytest.raises(G.GeometryError):
        G.ArenaGeometry.load(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text(json.dumps({"schema": "something/else"}), encoding="utf-8")
    with pytest.raises(G.GeometryError, match="schema"):
        G.ArenaGeometry.load(tmp_path / "bad.json")


def test_cli_check_template_and_fit(tmp_path, capsys):
    assert G.main(["check", "--size", "1080x2400"]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert G.main(["check", "--size", "1080x1920"]) == 0
    points, out = tmp_path / "points.json", tmp_path / "geo_pc.json"
    assert G.main(["template", "--size", "540x960", "--out", str(points)]) == 0
    doc = json.loads(points.read_text(encoding="utf-8"))
    assert doc["frame_size"] == [540, 960] and len(doc["points"]) == len(G.LANDMARKS_TILES)
    assert G.main(["fit", "--points", str(points), "--out", str(out)]) == 0
    geo = G.ArenaGeometry.load(out)
    assert geo.frame_size == (540, 960) and geo.name == "geo_pc"
    ref = G.ArenaGeometry.default()
    for sx, sy in ((0, 0), (17, 31), (8, 16), (3, 25)):
        a, b = geo.tile_center_px(sx, sy), ref.tile_center_px(sx, sy, (540, 960))
        assert abs(a[0] - b[0]) < 0.5 and abs(a[1] - b[1]) < 0.5      # template px are rounded to 0.1
    for slot in range(4):
        a, b = geo.card_slot_px(slot), ref.card_slot_px(slot, (540, 960))
        assert abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1
    doc["frame_size"] = [1080, 2400]
    points.write_text(json.dumps(doc), encoding="utf-8")
    assert G.main(["fit", "--points", str(points), "--out", str(tmp_path / "x.json")]) == 2
    assert G.main(["show"]) == 0


def test_runs_as_a_file_despite_the_types_module_name():
    """scripts/vision/types.py must not shadow the stdlib when geometry.py is run directly."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "vision" / "geometry.py"),
                        "check", "--size", "1080x1920"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
