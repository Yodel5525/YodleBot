"""E14: row schema v2, tools/rs_schema.py and tools/rs_rows.py --schema legacy|v2.

Pure tests, worked by hand: the column specs of both schemas (names, order, shapes, dtypes,
feature names, groups, mirror rules, bytes a row, JSON and digest, the meta checks), the v2
helpers of rs_rows (unit forms from unit-type names, spell cards, hand features, spell objects,
press costs, the opponent's counted elixir) and the v2 obs_row on a synthetic state: the spec's
shapes and dtypes (rs_schema.V2.check_row), every value of the blue row, and the legacy columns
it shares. Golden: the legacy obs_row is a frozen copy of obs_row at fe0bd18 (before E14) for
every flag; rows_for, build and vocab_record write what they wrote unless --schema v2 is given.

rows_for and build on conftest's fake royalesim (no engine, no corpus), with the fake's catalogue
read as name / elixir / ... / card_kind (FAKE_FIELDS: the fake keeps the cost in column 1) and
rs_elixir's constant law (6 elixir at tick 0, 500 of 28000 fine units a tick at 1x, so a bar's
milli is fine // 28). The match is test_decision_rows' (tag decision-0001), plays executing at

    105 red Knight (3 elixir)   290 red Arrows (3)   300 blue HogRider (4)   320 blue Musketeer (4)

and a red Fireball in flight in the states 300 to 339. Blue's count of red at a row's tick t
(the plays with exec + 1 <= t):

    t <= 105        168000 + 500 t
    106 .. 290      137000 + 500 (t - 106)      (220500 at 105, - 84000, + 500)
    291 ..          145500 + 500 (t - 291)      (229000 at 290, - 84000, + 500)

and red's count of blue: min(280000, 168000 + 500 t) to 300 (full from 224), 168500 + 500 (t -
301) to 320, then 66500 + 500 (t - 321). The engine-marked test checks that count against the
real engine's own bar on every v2 row.
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    DEPLOY_REASONS,
    FIELD,
    SUBTILE_PER_MILLITILE,
    entity_row,
    il_payload,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = SUBTILE_PER_MILLITILE
TAG = "decision-0001"
WAIT = (0, 0, 0)
PLAYS = ((105, "opponent", "knight", 14500, 24500), (290, "opponent", "arrows", 3500, 8000),
         (300, "team", "hog-rider", 3500, 14500), (320, "team", "musketeer", 3500, 7500))
BLUE_EARLY, BLUE_LATE = ([2, 0, 6, 3], 5), ([5, 0, 6, 3], 1)
OPP_HAND, OPP_NEXT = [8, 11, 9, 10], 12
#: (exec, side, action) of the four plays.
QUEUE = ((105, -1, (10, 3, 24)), (290, -1, (13, 14, 8)), (300, 1, (4, 3, 17)),
         (320, 1, (7, 3, 24)))
#: The fake catalogue's columns by name: the cost is column 1, the card kind column 8.
FAKE_FIELDS = ("name", "elixir", "c2", "c3", "c4", "flying", "c6", "c7", "card_kind")
#: A red Fireball in flight from tick 300 to 339 (SPELL_FIELDS_FALLBACK order).
FIREBALL = [1, 1, 0, 9000 * SUB, 20000 * SUB, 3500 * SUB, 6500 * SUB, 0, 0, 0, 0, 10]
#: The unit types of the obs_row tests: the defaults, then an evolution, a hero, an evolution.
UNIT_TYPES = DEFAULT_UNIT_TYPES + ("Skeleton_EV1", "MusketeerHero", "Knight_EV1")
UNIT_FORMS = (0.0,) * 14 + (0.5, 1.0, 0.5)
V2_OBS = ("tick", "side", "hand_exact", "unit_ids", "unit_feats", "towers", "scalars", "hand",
          "hand_feats", "next", "opp_seen", "opp_last", "opp_last_age", "spell_ids",
          "spell_feats", "own_pending", "own_pending_land", "ability_card", "ability_state")
V2_SHARD = V2_OBS + ("outcome", "trusted", "won", "hog", "act_card", "act_x", "act_y",
                     "act_press", "match", "split")
LEGACY_ROW = ("tick", "side", "hand_exact", "unit_ids", "unit_feats", "towers", "scalars",
              "hand", "next", "opp_seen", "opp_last", "opp_last_age", "outcome", "trusted",
              "won", "hog")
DSTAT_A = {"play_rows": 4, "press_rows": 0, "not_in_hand": 1, "clamped": 0, "pending_rows": 6,
           "exact_rows": 0}
REFUSED = DEPLOY_REASONS.index("NOT_ENOUGH_ELIXIR")


def f16(values):
    return np.asarray(values, dtype=np.float16).tolist()


def f32(values):
    return np.asarray(values, dtype=np.float32).tolist()


def same(a, b, names):
    for k in names:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and np.array_equal(x, y), k


def same_row(a, b):
    assert list(a) == list(b)
    same(a, b, a)


def keys(rows):
    return [(r["tick"], r["side"], tuple(int(v) for v in r["action"])) for r in rows]


def red_bar(t, arrows=True):
    """Blue's count of red's bar at a row's tick t, milli (module doc)."""
    if t <= 105:
        fine = 168000 + 500 * t
    elif t <= 290 or not arrows:
        fine = 137000 + 500 * (t - 106)
    else:
        fine = 145500 + 500 * (t - 291)
    return min(280000, fine) // 28


def blue_bar(t, press_at=None, press_cost=0):
    """Red's count of blue's bar at a row's tick t, milli; a press of `press_cost` at press_at."""
    if t <= 300:
        fine = min(280000, 168000 + 500 * t)
    elif t <= 320:
        fine = 168500 + 500 * (t - 301)
    elif press_at is None or t <= press_at:
        fine = 66500 + 500 * (t - 321)
    else:
        at = 66500 + 500 * (press_at - 321)
        fine = at - 28000 * press_cost + 500 + 500 * (t - press_at - 1)
    return min(280000, fine) // 28


def payload_a(**kwargs):
    kwargs.setdefault("tag", TAG)
    kwargs.setdefault("plays", PLAYS)
    kwargs.setdefault("abilities", ())
    return il_payload(**kwargs)


def states_a(battle, abilities=()):
    hand, nxt = BLUE_EARLY if battle.tick <= 300 else BLUE_LATE
    spells = [FIREBALL] if 300 <= battle.tick < 340 else []
    return state_dict(battle.tick, spells=spells,
                      players=[player_dict(0, hand=hand, next_card=nxt,
                                           abilities=[list(r) for r in abilities]),
                               player_dict(1, hand=OPP_HAND, next_card=OPP_NEXT)])


def legacy_order():
    """(tick, side, action) of the rows at execution (test_decision_rows.legacy_order)."""
    plays = {e: (s, a) for e, s, a in QUEUE}
    out = []
    for t in sorted(set(range(20, 521, 20)) | set(plays)):
        if t in plays:
            side, act = plays[t]
            out.append((t, side, act))
            if t % 20 == 0:
                out.append((t, -side, WAIT))
            continue
        out.append((t, 1, WAIT))
        if t not in (100, 280):
            out.append((t, -1, WAIT))
    return out


def decision_order():
    """(tick, side, action) of the --obs-delay 21 rows (test_decision_rows.decision_order)."""
    taps = {84: (-1, QUEUE[0][2]), 269: (-1, QUEUE[1][2]), 279: (1, QUEUE[2][2]),
            299: (1, QUEUE[3][2])}
    out = []
    for t in sorted(set(range(20, 521, 20)) | set(taps)):
        if t in taps:
            side, act = taps[t]
            out.append((t, side, act))
            continue
        out.append((t, 1, WAIT))
        if t not in (80, 260):
            out.append((t, -1, WAIT))
    return out


def shard(path):
    with np.load(path) as z:
        return list(z.files), {k: z[k] for k in z.files}


@pytest.fixture
def rs(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    return importlib.import_module("rs_schema")


@pytest.fixture
def rr(monkeypatch, rs):
    """tools/rs_rows.py as the tools import it, its module settings at the defaults (restored
    after the test, ROW_SCHEMA and the v2 tables too), no engine config from the PC, and
    rs_elixir's constant law."""
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "TRUST": rr.TrustRule(), "OUT": rr.OUT,
                        "CARD_FLY": {}, "CARD_BUILDING": {}, "PRESSES": "off",
                        "BUTTONS": 2, "HAND_SIZE": 4, "DECISION": rr.DecisionRule(),
                        "ROW_SCHEMA": rs.SCHEMA_LEGACY, "SF": None, "UNIT_FORMS": None,
                        "SPELL_CARDS": None}.items():
        monkeypatch.setattr(rr, name, value)
    law = rr.rs_elixir.ElixirLaw.constants()
    monkeypatch.setattr(rr.rs_elixir, "default_law", lambda: law)
    return rr


@pytest.fixture
def engine(monkeypatch):
    """install(**battle_kwargs): a fake royalesim in sys.modules for this test, its catalogue
    read by FAKE_FIELDS."""
    def install(**kwargs):
        mod = make_fake_royalesim(**kwargs)
        mod.CATALOGUE_FIELDS = list(FAKE_FIELDS)
        monkeypatch.setitem(sys.modules, "royalesim", mod)
        return mod
    return install


def cfg(rr):
    return rr.rs_engine.EngineConfig.defaults()


def run(rr, payload=None, **kwargs):
    """rows_for on `payload` (payload_a()): (rows, info)."""
    info = {}
    rows, why = rr.rows_for(payload_a() if payload is None else payload,
                            rr._setup(cfg(rr)), cfg(rr), info=info, **kwargs)
    assert why == "ok"
    return rows, info


# ---------------------------------------------------------------------------
# tools/rs_schema.py: the column specs
# ---------------------------------------------------------------------------

def _train_cols():
    """sim/train_tokens.py COLS, read without importing torch."""
    tree = ast.parse((ROOT / "sim" / "train_tokens.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "COLS" for t in n.targets))
    return tuple(ast.literal_eval(node.value))


def test_schema_names(rs):
    assert rs.SCHEMAS == ("rs_rows/legacy", "rs_rows/v2") and rs.CHOICES == ("legacy", "v2")
    assert rs.schema_name() == rs.schema_name(None) == rs.schema_name("legacy") == rs.SCHEMA_LEGACY
    assert rs.schema_name(" v2 ") == rs.schema_name("rs_rows/v2") == rs.SCHEMA_V2
    assert rs.schema_name(rs.V2) == rs.SCHEMA_V2 and rs.get("v2") is rs.V2
    assert rs.get() is rs.LEGACY and rs.LEGACY.short == "legacy" and rs.V2.short == "v2"
    assert rs.V2.v2 and not rs.LEGACY.v2 and rs.V2 != rs.LEGACY and rs.V2 == rs.get("v2")
    for bad in ("v3", "", 2, b"v2", "rs_rows/v9"):
        with pytest.raises(ValueError, match="row schema"):
            rs.schema_name(bad)
    assert issubclass(rs.SchemaError, ValueError)
    assert rs.of_meta(None) is rs.LEGACY and rs.of_meta({}) is rs.LEGACY
    assert rs.of_meta({"schema": None}) is rs.LEGACY
    assert rs.of_meta({"schema": "rs_rows/v2"}) is rs.V2
    with pytest.raises(rs.SchemaError):
        rs.of_meta({"schema": "rs_rows/v9"})
    with pytest.raises(rs.SchemaError, match="no column"):
        rs.V2.column("unit_form")


def test_legacy_spec_is_the_legacy_shard(rs, rr):
    L = rs.LEGACY
    assert L.row_fields() == rr.FIELDS == LEGACY_ROW
    assert L.shard_columns() == rr.FIELDS + ("act_card", "act_x", "act_y", "match", "split")
    assert L.extras == ("unit_form", "unit_burrow", "trusted_v2", "ability_card",
                        "ability_state", "own_pending", "own_pending_land", "act_press")
    every = (rr.VISION_FIELDS + rr.VISIBILITY_FIELDS + rr.TRUST_FIELDS + rr.PRESS_COLUMNS
             + rr.DECISION_FIELDS)                    # vocab.json extra_columns, in its order
    assert L.shard_columns(every) == (rr.FIELDS + rr.VISION_FIELDS + rr.VISIBILITY_FIELDS
                                      + rr.TRUST_FIELDS + rr.PRESS_FIELDS + rr.DECISION_FIELDS
                                      + ("act_card", "act_x", "act_y", "act_press", "match",
                                         "split"))
    assert L.shard_columns(["act_press"])[-3:] == ("act_press", "match", "split")
    assert L.obs_fields() == LEGACY_ROW[:12]
    assert L.train_columns == _train_cols()
    assert L.model_inputs == _train_cols()[:9] and L.labels == ("outcome", "won", "act_card",
                                                                  "act_x", "act_y")
    assert (L.unit_features, L.n_scalars, L.state_features) == (7, 4, 10)
    assert L.groups == {"state": ("towers", "scalars"), "hand": ("hand",), "next": ("next",),
                        "units": ("unit_ids", "unit_feats"), "opp_seen": ("opp_seen",),
                        "opp_last": ("opp_last", "opp_last_age")}
    assert [L.dtype(c) for c in ("tick", "side", "next", "outcome", "won")] == ["int64"] * 3 + [
        "float64"] * 2
    with pytest.raises(rs.SchemaError, match="not extra columns"):
        L.shard_columns(["spell_ids"])
    with pytest.raises(rs.SchemaError, match="not extra columns"):
        L.row_fields("hand")


def test_v2_spec_columns_and_shapes(rs):
    V = rs.V2
    assert V.obs_fields() == V2_OBS
    assert V.shard_columns() == V2_SHARD and V.extras == ("trusted_v2",)
    assert V.shard_columns(("trusted_v2",)) == V2_OBS + ("outcome", "trusted", "won", "hog",
                                                        "trusted_v2") + V2_SHARD[-6:]
    assert V.row_fields() == V2_OBS + ("outcome", "trusted", "won", "hog")
    assert V.model_inputs == V2_OBS[3:]
    assert V.labels == ("outcome", "won", "act_card", "act_x", "act_y", "act_press")
    assert V.train_columns == V.model_inputs + V.labels
    assert (V.unit_features, V.n_scalars, V.state_features) == (9, 7, 13)
    assert V.shape("ability_state", 3) == (3, 5) and V.shape("ability_state") == (2, 5)
    assert V.shape("ability_card", np.int64(4)) == (4,)
    assert V.shape("unit_feats") == (48, 9) and V.shape("spell_feats") == (8, 6)
    assert V.shape("hand_feats") == (4, 3) and V.shape("tick") == () and V.shape("scalars") == (7,)
    assert [V.dtype(c) for c in ("tick", "side", "next", "outcome", "won", "act_press")] == [
        "int32", "int8", "int16", "float32", "float32", "int16"]
    assert V.feats("unit_feats") == ("own", "x", "y", "hp", "flying", "building", "deploying",
                                     "form", "burrow")
    assert V.feats("unit_feats", live=True)[6:] == ("log_max_hp", "form", "burrow")
    assert V.feats("scalars") == ("elixir", "time", "double", "overtime", "triple",
                                  "opp_elixir", "opp_exact")
    assert V.feats("hand_feats") == ("form", "evo_ready", "evo_progress")
    assert V.feats("spell_feats") == ("own", "x", "y", "aim_x", "aim_y", "flown")
    assert V.index("scalars", "opp_elixir") == 5 and V.index("unit_feats", "burrow") == 8
    assert V.index("unit_feats", "log_max_hp", live=True) == 6
    with pytest.raises(rs.SchemaError, match="no feature"):
        rs.LEGACY.index("unit_feats", "deploying", live=True)
    assert V.groups["hand"] == ("hand", "hand_feats")
    assert V.groups["spells"] == ("spell_ids", "spell_feats")
    assert V.groups["pending"] == ("own_pending", "own_pending_land")
    assert V.groups["buttons"] == ("ability_card", "ability_state")
    assert V.mirror() == {"unit_feats": {"flip": [1], "where": "unit_ids"},
                          "towers": {"swap": [[0, 1], [3, 4]]},
                          "spell_feats": {"flip": [1, 3], "where": "spell_ids"},
                          "act_x": {"tile": 17, "where": "act_card"}}
    for bad in (True, -1, 2.0, "3"):
        with pytest.raises(rs.SchemaError, match="buttons"):
            V.shape("ability_card", bad)


def test_layout_numbers_are_the_row_builders(rs, rr):
    assert (rs.MAX_UNITS, rs.PENDING_SLOTS, rs.PENDING_SCALE) == (rr.MAX_UNITS, rr.PENDING_SLOTS,
                                                                 rr.PENDING_SCALE)
    assert rs.LEGACY.unit_features == rr.UF and rs.V2.unit_features == rr.V2_UF
    assert rs.ABILITY_FEATS == rr.rs_presses.ABILITY_FEATS
    assert rs.HAND_SIZE == rr.rs_presses.FALLBACK_HAND_SIZE == 4
    assert rs.FALLBACK_BUTTONS == rr.rs_presses.FALLBACK_BUTTONS == 2
    assert (rs.GRID_W, rs.GRID_H, rs.SPELL_SLOTS, rs.SPELL_FLOWN_SCALE) == (18, 32, 8, 40)
    assert (rs.OPP_SEEN, rs.OPP_LAST) == (8, 4)
    assert rr.V2_OBS_FIELDS == V2_OBS and rr.V2_OPP_EXACT == 6


def test_row_bytes_by_hand(rs):
    # v2 train columns at B = 2: units 96 + 864, towers 24, scalars 28, hand 8 + 24, next 2,
    # opp 16 + 8 + 8, spells 16 + 96, pending 8 + 8, buttons 4 + 20; labels 4 + 4 + 2 x 4
    assert rs.V2.row_bytes(2) == 1246 and rs.V2.row_bytes() == 1246
    assert rs.V2.row_bytes(3) == 1246 + 2 + 10
    # legacy: units 96 + 672, towers 24, scalars 16, hand 8, next 8, opp 16 + 8 + 8,
    # outcome 8, won 8, act 2 x 3
    assert rs.LEGACY.row_bytes() == 878
    assert rs.V2.row_bytes(2, ["tick", "side", "match"]) == 5


def test_json_digest_and_meta_checks(rs):
    js = rs.V2.to_json(3)
    assert js["name"] == "rs_rows/v2" and js["buttons"] == 3
    assert [c["name"] for c in js["columns"]] == list(V2_SHARD[:23]) + ["trusted_v2"] + list(
        V2_SHARD[23:])
    col = {c["name"]: c for c in js["columns"]}
    assert col["ability_state"]["shape"] == [3, 5]
    assert col["ability_state"]["feats"] == list(rs.ABILITY_FEATS)
    assert col["trusted_v2"]["extra"] == "--trust v2" and "extra" not in col["tick"]
    assert col["unit_feats"]["mirror"] == {"flip": [1], "where": "unit_ids"}
    assert "compact" not in col["tick"] and "doc" not in col["tick"]
    assert rs.LEGACY.to_json()["columns"][0]["compact"] == "int32"
    assert js["groups"]["spells"] == ["spell_ids", "spell_feats"]
    assert js["live_unit_feats"][6] == "log_max_hp"
    assert js["layout"] == {"max_units": 48, "hand": 4, "opp_seen": 8, "opp_last": 4,
                            "pending_slots": 4, "pending_scale": 40, "spell_slots": 8,
                            "spell_flown_scale": 40, "grid": [18, 32]}
    assert json.loads(json.dumps(js)) == js
    d3 = rs.V2.digest(3)
    assert len(d3) == 16 and int(d3, 16) >= 0 and d3 == rs.V2.digest(3)
    assert d3 != rs.V2.digest(2) != rs.LEGACY.digest(2)
    # a meta's schema_spec against this code
    meta = {"schema": "rs_rows/v2", "schema_spec": rs.V2.to_json(3)}
    assert rs.spec_problems(meta) == [] and rs.buttons_of(meta) == 3
    assert rs.spec_problems({}) == [] and rs.spec_problems({"schema": "rs_rows/legacy"}) == []
    changed = json.loads(json.dumps(meta))
    changed["schema_spec"]["columns"][4]["shape"] = [48, 7]
    problems = rs.spec_problems(changed)
    assert len(problems) == 1 and problems[0].startswith("column unit_feats:")
    renamed = json.loads(json.dumps(meta))
    renamed["schema_spec"]["name"] = "rs_rows/legacy"
    assert "schema_spec names 'rs_rows/legacy'" in rs.spec_problems(renamed)[0]
    swapped = json.loads(json.dumps(meta))
    cols = swapped["schema_spec"]["columns"]
    cols[0], cols[1] = cols[1], cols[0]
    assert rs.spec_problems(swapped) == ["columns are in another order"]
    more = json.loads(json.dumps(meta))
    more["schema_spec"]["note"] = 1
    assert rs.spec_problems(more) == ["schema_spec has 'note', which this code does not write"]
    assert rs.spec_problems({"schema": "rs_rows/v2", "schema_spec": [1]}) == [
        "schema_spec is a list, not a dict"]
    assert rs.buttons_of({"presses": {"buttons": 4}}) == 4 and rs.buttons_of(None) == 2
    assert rs.buttons_of({"schema_spec": {"buttons": 3}, "presses": {"buttons": 4}}) == 3


def test_column_and_schema_validation(rs):
    C = rs.Column
    with pytest.raises(rs.SchemaError, match="dtype"):
        C("x", (2,), "int7", rs.ROLE_OBS, rs.SOURCE_OBS)
    with pytest.raises(rs.SchemaError, match="dtype"):
        C("x", (2,), "int16", rs.ROLE_OBS, rs.SOURCE_OBS, compact="int3")
    with pytest.raises(rs.SchemaError, match="role"):
        C("x", (2,), "int16", "input", rs.SOURCE_OBS)
    with pytest.raises(rs.SchemaError, match="feature names"):
        C("x", (2, 3), "float16", rs.ROLE_OBS, rs.SOURCE_OBS, feats=("a", "b"))
    ok = C("x", (rs.BUTTONS, 2), "float16", rs.ROLE_OBS, rs.SOURCE_OBS, feats=("a", "b"))
    assert ok.resolved(5) == (5, 2) and ok.resolved() == (2, 2)
    with pytest.raises(rs.SchemaError, match="twice"):
        rs.Schema("t", [ok, ok], {}, ())


def test_check_row_and_shard_report_problems(rs):
    row = {"tick": 3, "side": -1, "hand_exact": True,
           "unit_ids": np.zeros(48, np.int16), "unit_feats": np.zeros((48, 9), np.float16),
           "towers": np.ones(6, np.float32), "scalars": np.zeros(7, np.float32),
           "hand": np.zeros(4, np.int16), "hand_feats": np.zeros((4, 3), np.float16),
           "next": np.int64(5), "opp_seen": np.zeros(8, np.int16),
           "opp_last": np.zeros(4, np.int16), "opp_last_age": np.ones(4, np.float16),
           "spell_ids": np.zeros(8, np.int16), "spell_feats": np.zeros((8, 6), np.float16),
           "own_pending": np.zeros(4, np.int16), "own_pending_land": np.zeros(4, np.float16),
           "ability_card": np.zeros(3, np.int16),
           "ability_state": np.zeros((3, 5), np.float16)}
    assert rs.V2.check_row(row, buttons=3) == []
    assert rs.V2.check_row(row) == ["ability_card: shape (3,), the spec says (2,)",
                                    "ability_state: shape (3, 5), the spec says (2, 5)"]
    bad = dict(row, scalars=np.zeros(7, np.float64), side=1.5, extra=1)
    del bad["hand"]
    assert rs.V2.check_row(bad, buttons=3) == [
        "unexpected keys ['extra']", "side: float 1.5 is not a int8 scalar",
        "scalars: dtype float64, the spec says float32", "hand: missing"]
    assert rs.V2.check_row({"hand_exact": 1}, fields=["hand_exact"]) == [
        "hand_exact: int 1 is not a bool scalar"]
    n = 2
    arrays = {c: np.zeros((n,) + rs.V2.shape(c, 3), dtype=rs.V2.dtype(c))
              for c in rs.V2.shard_columns() if rs.V2.dtype(c) != "str"}
    arrays["match"] = np.asarray(["a", "b"])
    arrays["split"] = np.asarray(["train", "val"])
    ordered = {c: arrays[c] for c in rs.V2.shard_columns()}
    assert rs.V2.check_shard(ordered, buttons=3) == []
    wrong = dict(ordered, tick=np.zeros(2, np.int64), split=np.asarray(["train"]))
    assert rs.V2.check_shard(wrong, buttons=3) == ["tick: dtype int64, the spec says int32",
                                                   "split: 1 rows, the first column has 2"]
    assert rs.V2.check_shard({"tick": ordered["tick"]}, buttons=3)[0].startswith(
        "columns ['tick'], the spec says ['tick', 'side'")


def test_describe_and_cli(rs, capsys):
    head = rs.V2.describe(2).splitlines()[0]
    assert head == ("rs_rows/v2: 30 columns (1 added by a flag), state token 13, unit features "
                    "9, 1,246 bytes a row of train columns")
    assert rs.LEGACY.describe().splitlines()[0].startswith(
        "rs_rows/legacy: 29 columns (8 added by a flag), state token 10, unit features 7, 878")
    assert rs.main(["--schema", "v2", "--buttons", "3", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == rs.V2.to_json(3)
    assert rs.main(["--schema", "legacy"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("rs_rows/legacy: 29 columns") and f"digest {rs.LEGACY.digest(2)}" in out


# ---------------------------------------------------------------------------
# tools/rs_rows.py: the v2 helpers, by hand
# ---------------------------------------------------------------------------

def test_schema_mode_and_describe(rr, rs):
    assert rr.schema_mode(None) == rr.schema_mode("legacy") == rs.SCHEMA_LEGACY
    assert rr.schema_mode("v2") == rr.schema_mode(rs.V2) == rs.SCHEMA_V2
    with pytest.raises(ValueError):
        rr.schema_mode("v3")
    assert rr.ROW_SCHEMA == rs.SCHEMA_LEGACY
    assert rr.describe_schema("legacy") == "rs_rows/legacy (the rows before E14)"
    text = rr.describe_schema("v2")
    assert text.startswith("rs_rows/v2 (spells in flight") and "2 ability buttons" in text
    assert text.endswith("1,246 bytes a row)")


def test_unit_forms_spell_cards_and_spell_fields(rr):
    assert rr.unit_form_table(UNIT_TYPES) == UNIT_FORMS
    assert rr.unit_form_table(["Knight", "Bat_EV1", "Archer_EV2_1", "GiantHero", "Giant_hero",
                               "IceGolemite"]) == (0.0, 0.5, 0.5, 1.0, 1.0, 0.0)
    assert rr.spell_cards(DEFAULT_CATALOGUE) == frozenset({1, 7, 11, 15})
    assert rr.spell_cards([["Short", 3]]) == frozenset()
    assert rr.spell_index() == {n: i for i, n in enumerate(rr.SPELL_FIELDS_FALLBACK)}
    mod = types.SimpleNamespace(SPELL_FIELDS=["team", "card_id", "x", "y", "aim_x", "aim_y"])
    assert rr.spell_index(mod) == {"team": 0, "card_id": 1, "x": 2, "y": 3, "aim_x": 4, "aim_y": 5}
    assert rr.spell_index(types.SimpleNamespace()) == rr.spell_index()
    rr.UNIT_FORMS = UNIT_FORMS
    assert [rr.unit_form(u) for u in (-1, 0, 14, 15, 16, 17, np.int64(15))] == [
        0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 1.0]


def test_hand_features(rr):
    H = rr.hand_features
    deck, forms = [0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 0, 2, 1, 0]
    p = player_dict(0, hand=[2, 5, 6, 0], deck=deck, forms=forms, evo=[[6, 1, 0, 2]])
    out = H(p)
    assert out.dtype == np.float16 and out.shape == (4, 3)
    assert out.tolist() == f16([[0, 0, 0], [1, 0, 0], [0.5, 0, 0.5], [0, 0, 0]])
    ready = player_dict(0, hand=[6, -1, 5], deck=deck, forms=forms, evo=[[6, 2, 1, 2]])
    assert H(ready).tolist() == f16([[0.5, 1, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0]])
    # an evo row alone reads as an evolution; a charge past its need is 1; need 0: the flag
    bare = player_dict(0, hand=[6, 3, 4, 2], evo=[[6, 5, 0, 2], [3, 0, 1, 0], [4, 0, 0, 0]])
    assert H(bare).tolist() == f16([[0.5, 0, 1], [0.5, 1, 1], [0.5, 0, 0], [0, 0, 0]])
    assert not H(player_dict(0)).any()                        # conftest: evo [] and no deck
    assert not H({"hand": [1, 2, 3, 4]}).any() and not H({}).any()   # the live HUD's player


def _spells():
    log = [0, 7, 1, 9000 * SUB, 9000 * SUB, 9000 * SUB, 17000 * SUB, 0, 0, 0, 0, -1]
    hog = [1, 2, 2, 9000 * SUB, 4000 * SUB, 9000 * SUB, 4000 * SUB, 0, 0, 0, 0, 0]
    zap = [1, 15, 2, 9000 * SUB, 26000 * SUB, 9000 * SUB, 26000 * SUB, 0, 0, 0, 0, 80]
    return [FIREBALL, log, hog, zap]


def test_spell_features_by_hand(rr):
    rr.SPELL_CARDS = rr.spell_cards(DEFAULT_CATALOGUE)
    state = state_dict(400, spells=_spells())
    blue_king, red_king = (9000 * SUB, 3000 * SUB), (9000 * SUB, 29000 * SUB)
    ids, feats = rr.spell_features(state, 0, SUB, blue_king)
    assert ids.dtype == np.int16 and feats.dtype == np.float16 and feats.shape == (8, 6)
    # the HogRider object (a troop's effect) is left out; the Log, the Fireball, the Zap
    assert ids.tolist() == [9, 3, 17, 0, 0, 0, 0, 0]
    assert feats[:3].tolist() == f16([[1, 0.5, 23000 / 32000, 0.5, 15000 / 32000, 0],
                                      [0, 0.5, 12000 / 32000, 3500 / 18000, 25500 / 32000, 0.25],
                                      [0, 0.5, 6000 / 32000, 0.5, 6000 / 32000, 1.0]])
    assert not feats[3:].any()
    ids, feats = rr.spell_features(state, 1, SUB, red_king)
    assert ids.tolist() == [17, 3, 9, 0, 0, 0, 0, 0]
    assert feats[:3].tolist() == f16([[1, 0.5, 26000 / 32000, 0.5, 26000 / 32000, 1.0],
                                      [1, 0.5, 20000 / 32000, 14500 / 18000, 6500 / 32000, 0.25],
                                      [0, 0.5, 9000 / 32000, 0.5, 17000 / 32000, 0]])
    # more than eight: the eight nearest, ticks_flown absent (a short row) reads 0
    many = [[1, 11, 0, 9000 * SUB, (4000 + 1000 * k) * SUB, 0, 0] for k in range(10)]
    ids, feats = rr.spell_features(state_dict(0, spells=many[::-1]), 0, SUB, blue_king)
    assert ids.tolist() == [13] * 8
    assert feats[:, 2].tolist() == f16([(28000 - 1000 * k) / 32000 for k in range(8)])
    assert not feats[:, 5].any()
    assert rr.spell_features(state_dict(0), 0, SUB, blue_king)[0].tolist() == [0] * 8
    # only the cards given
    assert rr.spell_features(state, 0, SUB, blue_king, frozenset({15}))[0].tolist()[:2] == [17, 0]
    rr.SF = {"team": 0, "card_id": 1}
    with pytest.raises(rr.rs_engine.EngineError, match="SPELL_FIELDS"):
        rr.spell_features(state, 0, SUB, blue_king)
    rr.SPELL_CARDS = None
    with pytest.raises(RuntimeError, match="spell cards"):
        rr.spell_features(state, 0, SUB, blue_king)


def test_press_cost_and_opp_elixir_argument(rr):
    p = {"abilities": [[1, 0, 3, 5, 0], [0, 0], [1, 0, -2, 7, 0]]}
    assert [rr.press_cost(p, b) for b in (0, 1, 2, 3, -1)] == [3, 0, 0, 0, 0]
    assert rr.press_cost({}, 0) == 0
    assert rr._opp_elixir((7321, False)) == (7321, False)
    assert rr._opp_elixir([np.int64(10000), np.bool_(True)]) == (10000, True)
    assert rr._opp_elixir(5000) == (5000, True)
    with pytest.raises(ValueError, match="opp_elixir"):
        rr._opp_elixir(None)


def test_elixir_law_follows_the_config(rr):
    law = rr.elixir_law(cfg(rr))
    assert law == rr.rs_elixir.ElixirLaw.constants()
    over = rr.rs_engine.EngineConfig(calibration_overrides={"match.START_MANA": 7})
    assert rr.elixir_law(over).start_fine == 7 * 28000
    assert rr.elixir_law(over).gain_1x == 500


def test_seen_elixir_by_hand(rr):
    cat = [list(r) for r in DEFAULT_CATALOGUE]
    law = rr.rs_elixir.ElixirLaw.constants()

    def seen(lag=1):
        return rr.SeenElixir(cat, fields=FAKE_FIELDS, law=law, lag=lag)

    se = seen()
    se.record(1, 100, 8)                                 # red's Knight runs at 100
    assert se.view(0, 100) == (218000 // 28, True)       # not seen yet: 7785
    assert se.view(0, 101) == (134500 // 28, True)       # 218000 - 84000 + 500: 4803
    assert se.view(0, 300) == (234000 // 28, True)       # + 199 ticks: 8357
    assert se.view(1, 300) == (10000, True)              # red counts blue: full since 224
    se.press(0, 300, 2)                                  # a press that ran
    assert se.view(1, 300) == (10000, True)
    assert se.view(1, 301) == ((280000 - 56000 + 500) // 28, True)      # 8017
    # a play the count cannot pay (a play missed, or elixir it was given) ends exactness
    short = seen()
    short.record(1, 100, 8)
    short.record(1, 102, 10)                             # Giant, 5: 135000 < 140000
    assert short.view(0, 200) == (49000 // 28, False)    # 500 at 103, + 97 ticks: 1750
    assert short.view(0, 300) == ((49000 + 50000) // 28, False)
    # a longer lag sees the play later, then pays it on its own tick
    late = seen(lag=5)
    late.record(1, 100, 8)
    assert late.view(0, 104) == (220000 // 28, True)
    assert late.view(0, 105) == (136500 // 28, True)
    # the Mirror: its copy + 1 (catalogue name)
    mirror = rr.SeenElixir(cat + [["Mirror", 1, 0, 0, 0, False, 0, 0, "SPELL"]],
                           fields=FAKE_FIELDS, law=law)
    mirror.record(1, 10, 10)                             # Giant 5, then a Mirror of it: 6
    mirror.record(1, 300, 16)
    # 173000 at 10, - 140000 + 500: 33500 at 11; + 289 ticks: 178000 at 300; - 168000 + 500
    assert mirror.view(0, 301) == ((178000 - 168000 + 500) // 28, True)    # 375
    with pytest.raises(ValueError, match="team"):
        se.view(2, 400)
    with pytest.raises(ValueError, match="lag"):
        seen(lag=-1)


# ---------------------------------------------------------------------------
# obs_row: the spec's shapes and dtypes, every value of one v2 row
# ---------------------------------------------------------------------------

@pytest.fixture
def v2rr(rr, engine):
    """rs_rows set up by _setup on the fake engine with UNIT_TYPES."""
    engine(unit_types=list(UNIT_TYPES))
    rr._setup(cfg(rr))
    assert rr.UNIT_FORMS == UNIT_FORMS and rr.SPELL_CARDS == frozenset({1, 7, 11, 15})
    assert rr.SF == rr.spell_index()
    return rr


def _units():
    return [
        entity_row(uid=101, team=0, unit_type=16, x=9000 * SUB, y=6000 * SUB, hp=500,
                   max_hp=1000),                                   # own evolved Knight
        entity_row(uid=102, team=1, unit_type=9, x=14000 * SUB, y=25000 * SUB, hp=300,
                   status_flags=1),                                # an enemy tunnelling
        entity_row(uid=103, team=1, unit_type=15, x=3000 * SUB, y=20000 * SUB, hp=600,
                   status_flags=16),                               # a hero, STATUS_HERO
        entity_row(uid=104, team=0, kind=1, unit_type=2, card_id=0, x=9000 * SUB,
                   y=9000 * SUB, hp=200, max_hp=400, deploy_ticks=5),   # own Cannon
        entity_row(uid=105, team=1, unit_type=11, x=9000 * SUB, y=12000 * SUB, hp=700,
                   status_flags=2),                                # an invisible enemy
    ]


BLUE = dict(elixir_milli=5000, hand=[2, 5, 6, 0], next_card=4, deck=[0, 1, 2, 3, 4, 5, 6, 7],
            forms=[0, 0, 0, 0, 0, 2, 1, 0], evo=[[6, 1, 0, 2]], abilities=[[1, 0, 1, 5, 0]])
MEM = {"seen": {0: {4, 7}, 1: {10, 13}}, "last": {0: [(4, 4000)], 1: [(10, 4700), (13, 4790)]}}
EMPTY_MEM = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}


def big_state():
    return state_dict(4800, entities=_units(), spells=_spells(),
                      players=[player_dict(0, **BLUE), player_dict(1)])


def test_v2_obs_row_by_hand(v2rr, rs):
    rr = v2rr
    row = rr.obs_row(big_state(), 0, SUB, MEM, 4800, schema="v2", opp_elixir=(7321, False),
                     pending=[(4, 10), (7, 30)], hand_exact=False)
    assert tuple(row) == V2_OBS and rs.V2.check_row(row, buttons=2) == []
    assert (row["tick"], row["side"], row["hand_exact"], row["next"]) == (4800, 1, False, 6)
    # units nearest the blue king first: Knight_EV1, Cannon, the invisible one, the hero,
    # the tunneller; form and burrow are features 7 and 8
    assert row["unit_ids"].tolist()[:6] == [18, 4, 13, 17, 11, 0]
    assert row["unit_feats"][:5].tolist() == f16([
        [1, 0.5, 26000 / 32000, 0.5, 0, 0, 0, 0.5, 0],
        [1, 0.5, 23000 / 32000, 0.5, 0, 1, 1, 0, 0],
        [0, 0.5, 20000 / 32000, 1, 0, 0, 0, 0, 0],
        [0, 3000 / 18000, 12000 / 32000, 1, 0, 0, 0, 1, 0],
        [0, 14000 / 18000, 7000 / 32000, 1, 0, 0, 0, 0, 1]])
    assert not row["unit_feats"][5:].any()
    assert row["towers"].tolist() == [1.0] * 6
    assert row["scalars"].tolist() == f32([0.5, 0.8, 1, 1, 1, 0.7321, 0])
    assert row["hand"].tolist() == [4, 7, 8, 2]
    assert row["hand_feats"].tolist() == f16([[0, 0, 0], [1, 0, 0], [0.5, 0, 0.5], [0, 0, 0]])
    assert row["opp_seen"].tolist() == [10, 13, 0, 0, 0, 0, 0, 0]
    assert row["opp_last"].tolist() == [10, 13, 0, 0]
    assert row["opp_last_age"].tolist() == f16([0.25, 0.025, 1, 1])
    assert row["spell_ids"].tolist() == [9, 3, 17, 0, 0, 0, 0, 0]
    assert row["spell_feats"][1].tolist() == f16([0, 0.5, 0.375, 3500 / 18000, 0.796875, 0.25])
    assert row["own_pending"].tolist() == [4, 7, 0, 0]
    assert row["own_pending_land"].tolist() == f16([0.25, 0.75, 0, 0])
    assert row["ability_card"].tolist() == [7, 0]
    assert row["ability_state"].tolist() == f16([[1, 1, 0, 0.1, 0], [0, 0, 0, 0, 0]])
    # defaults: no pending plays, hand_exact True, the exact flag of an int count
    plain = rr.obs_row(big_state(), 0, SUB, MEM, 2500, schema="v2", opp_elixir=10000)
    assert not plain["own_pending"].any() and not plain["own_pending_land"].any()
    assert plain["hand_exact"] is True
    assert plain["scalars"].tolist() == f32([0.5, 2500 * 50 / 300000, 1, 0, 0, 1, 1])
    # the red seat's row of the same state fits the spec too
    red = rr.obs_row(big_state(), 1, SUB, MEM, 4800, schema="v2", opp_elixir=(4000, True))
    assert rs.V2.check_row(red, buttons=2) == [] and red["side"] == -1
    assert red["spell_ids"].tolist()[:3] == [17, 3, 9] and not red["hand_feats"].any()
    assert red["ability_card"].tolist() == [0, 0]


def test_v2_shares_the_legacy_columns(v2rr):
    rr = v2rr
    for team in (0, 1):
        old = rr.obs_row(big_state(), team, SUB, MEM, 4800)
        new = rr.obs_row(big_state(), team, SUB, MEM, 4800, schema="v2", opp_elixir=(1, True))
        same(old, new, ("tick", "side", "unit_ids", "towers", "hand", "next", "opp_seen",
                        "opp_last", "opp_last_age"))
        assert np.array_equal(new["unit_feats"][:, :7], old["unit_feats"])
        assert np.array_equal(new["scalars"][:4], old["scalars"])
        # the module setting chooses the schema too
        rr.ROW_SCHEMA = "rs_rows/v2"
        same_row(rr.obs_row(big_state(), team, SUB, MEM, 4800, opp_elixir=(1, True)), new)
        rr.ROW_SCHEMA = "rs_rows/legacy"


def test_v2_obs_row_under_the_other_flags(v2rr, rs):
    rr = v2rr
    kw = dict(schema="v2", opp_elixir=(5000, True))
    types_row = rr.obs_row(big_state(), 0, SUB, MEM, 4800, **kw)
    # --visibility public: the invisible enemy is left out, no unit_burrow column
    public = rr.obs_row(big_state(), 0, SUB, MEM, 4800, visibility="public", **kw)
    assert tuple(public) == V2_OBS and public["unit_ids"].tolist()[:5] == [18, 4, 17, 11, 0]
    assert public["unit_feats"][:4].tolist() == types_row["unit_feats"][[0, 1, 3, 4]].tolist()
    # --live-compat: the creating card and its flags, log max hp; form and burrow as types
    live = rr.obs_row(big_state(), 0, SUB, MEM, 4800, live=True, **kw)
    assert live["unit_ids"].tolist()[:5] == [1, 2, 1, 1, 1]
    assert live["unit_feats"][0].tolist() == f16([1, 0.5, 0.8125, 0.5, 0, 0,
                                                  float(np.log1p(1000)) / 10, 0.5, 0])
    assert live["unit_feats"][1, 5] == 1.0                    # the Cannon card is a building
    assert np.array_equal(live["unit_feats"][:, 7:], types_row["unit_feats"][:, 7:])
    # --units vision: class tokens, flags and the form from the class lookup
    rr.VC = StubLookup()
    vision = rr.obs_row(big_state(), 0, SUB, MEM, 4800, units="vision", **kw)
    assert tuple(vision) == V2_OBS and vision["unit_ids"].tolist()[:5] == [19, 5, 14, 18, 12]
    assert vision["unit_feats"][:5, 7].tolist() == [0.0] * 5  # the stub's forms, not the names'
    assert vision["unit_feats"][1, 5] == 1.0 and vision["unit_feats"][2, 4] == 1.0
    assert vision["unit_feats"][4, 8] == 1.0                  # the burrow bit still
    rr.UNIT_FORMS = None                                      # vision needs no form table
    assert rr.obs_row(big_state(), 0, SUB, MEM, 4800, units="vision", **kw)["unit_ids"][0] == 19
    with pytest.raises(RuntimeError, match="unit form table"):
        rr.obs_row(big_state(), 0, SUB, MEM, 4800, **kw)
    rr.UNIT_FORMS = UNIT_FORMS
    # presses on: the same columns, the buttons named from `buttons` past the engine's rows
    press = rr.obs_row(big_state(), 0, SUB, MEM, 4800, presses="on", buttons=[5, 9], **kw)
    assert tuple(press) == V2_OBS and press["ability_card"].tolist() == [7, 11]


def test_v2_obs_row_on_a_live_screen_state(v2rr, rs):
    """The dict tools/public_state.py builds: no abilities, evo or forms; status 0; spell rows
    with x / y / aim at the track and ticks_flown its age."""
    rr = v2rr
    me = {"team": 1, "elixir_milli": 4000, "hand": [2, 0, 6, 3], "hand_costs": [4, 3, 1, 2],
          "next_card": 3, "deck": [0, 1, 2, 3, 4, 5, 6, 7], "crowns": 0,
          "tower_hp": [4824, 3052, 3052], "tower_max_hp": [4824, 3052, 3052],
          "king_active": False, "pending_cost": 0}
    them = {"team": 0, "elixir_milli": 0, "hand": [], "hand_costs": [], "next_card": -1,
            "deck": [8, 11], "crowns": 0, "tower_hp": [4824, 3052, 3052],
            "tower_max_hp": [4824, 3052, 3052], "king_active": False, "pending_cost": 0}
    spell = [0] * 12
    for name, v in (("team", 0), ("card_id", 11), ("x", 9000 * SUB), ("y", 20000 * SUB),
                    ("aim_x", 9000 * SUB), ("aim_y", 20000 * SUB), ("ticks_flown", 4)):
        spell[rr.SF[name]] = v
    state = {"tick": 900, "tick_ms": 45000, "regular_ticks": 3600, "overtime_ticks": 2400,
             "elixir_rate": 1, "overtime": False, "players": [them, me],
             "entities": [entity_row(uid=7, team=0, unit_type=8, x=9000 * SUB, y=12000 * SUB,
                                     hp=900)],
             "spells": [spell], "projectiles": [], "game_over": False, "winner": -1}
    row = rr.obs_row(state, 1, SUB, EMPTY_MEM, 900, schema="v2", opp_elixir=(6000, True))
    assert rs.V2.check_row(row, buttons=2) == []
    assert row["hand"].tolist() == [4, 2, 8, 5] and row["next"] == 5
    assert not row["hand_feats"].any()
    assert row["ability_card"].tolist() == [0, 0] and not row["ability_state"].any()
    assert row["spell_ids"].tolist()[0] == 13
    assert row["spell_feats"][0].tolist() == f16([0, 0.5, 0.625, 0.5, 0.625, 0.1])
    assert row["unit_ids"].tolist()[0] == 10 and row["unit_feats"][0, 8] == 0.0


def test_v2_obs_row_errors(v2rr, rs):
    rr = v2rr
    with pytest.raises(ValueError, match="opp_elixir"):
        rr.obs_row(big_state(), 0, SUB, MEM, 4800, schema="v2")
    with pytest.raises(rs.SchemaError):
        rr.obs_row(big_state(), 0, SUB, MEM, 4800, schema="v3", opp_elixir=1)
    unreported = state_dict(100, entities=[entity_row(uid=9, team=1, unit_type=8,
                                                      x=9000 * SUB, y=20000 * SUB,
                                                      status_flags=-1)])
    with pytest.raises(rr.VisibilityError, match="status_flags"):
        rr.obs_row(unreported, 0, SUB, EMPTY_MEM, 100, schema="v2", opp_elixir=1)
    assert "unit_burrow" not in rr.obs_row(unreported, 0, SUB, EMPTY_MEM, 100)
    rr.SPELL_CARDS = None
    with pytest.raises(RuntimeError, match="spell cards"):
        rr.obs_row(big_state(), 0, SUB, MEM, 4800, schema="v2", opp_elixir=1)


# ---------------------------------------------------------------------------
# golden: legacy obs_row is obs_row before E14
# ---------------------------------------------------------------------------

class StubLookup:
    """vision_classes.UnitLookup's surface obs_row reads (token, flags, form_feature)."""

    def token(self, unit):
        return 1 if unit < 0 else unit + 3

    def flags(self, unit):
        return (1.0 if unit % 2 else 0.0, 1.0 if unit == 2 else 0.0)

    def form_feature(self, unit):
        return 0.5 if unit == 3 else 0.0


def frozen_obs_row(rr, state, team, sub, mem, tick, live=False, units="types",
                   visibility="legacy", presses="off", buttons=None, pending=None,
                   hand_exact=None):
    """tools/rs_rows.obs_row at fe0bd18 (before E14), frozen; module names through `rr`,
    MAX_UNITS 48 and UF 7 the values it had."""
    public = visibility == "public"
    press = presses == "on"
    vision = not live and units == "vision"
    F = rr.F
    forms = np.zeros(48, dtype=np.float16) if vision else None
    burrow = np.zeros(48, dtype=np.float16) if public else None
    ents = [e for e in state["entities"] if e[F["tower_slot"]] < 0 and e[F["hp"]] > 0]
    if public:
        ents = [e for e in ents if rr.seen_by(e, team)]
    king = next((e for e in state["entities"]
                 if e[F["team"]] == team and e[F["tower_slot"]] == 0), None)
    kx, ky = ((king[F["x"]], king[F["y"]]) if king is not None
              else (9000 * 18, (3000 if team == 0 else 29000) * 18))
    ents.sort(key=lambda e: (e[F["x"]] - kx) ** 2 + (e[F["y"]] - ky) ** 2)
    ids = np.zeros(48, dtype=np.int16)
    feats = np.zeros((48, 7), dtype=np.float16)
    for k, e in enumerate(ents[:48]):
        ox, oy = rr.own(e[F["x"]], e[F["y"]], team, sub)
        if live:
            card = e[F["card_id"]]
            ids[k] = card + 2
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]),
                        1.0 if rr.CARD_FLY.get(card) else 0.0,
                        1.0 if rr.CARD_BUILDING.get(card) else 0.0,
                        float(np.log1p(max(0, e[F["max_hp"]]))) / 10.0)
        elif vision:
            unit = e[F["unit_type"]]
            ids[k] = rr.VC.token(unit)
            fly, building = rr.VC.flags(unit)
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]), fly, building,
                        1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
            forms[k] = rr.VC.form_feature(unit)
        else:
            ids[k] = e[F["unit_type"]] + 2
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]), 1.0 if e[F["flying"]] else 0.0,
                        1.0 if e[F["kind"]] == 1 else 0.0,
                        1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
        if public:
            burrow[k] = 1.0 if rr.burrowing(e) else 0.0
    p = state["players"][team]
    hand = [c + 2 for c in p["hand"][:4]] + [0] * (4 - len(p["hand"][:4]))
    opp = 1 - team
    seen = sorted(mem["seen"][opp])[:8]
    last = mem["last"][opp][-4:]
    t_ms = tick * 50
    row = {
        "tick": tick, "side": 1 if team == 0 else -1,
        "hand_exact": True if hand_exact is None else bool(hand_exact),
        "unit_ids": ids, "unit_feats": feats, "towers": rr.towers_vec(state, team),
        "scalars": np.asarray([p["elixir_milli"] / 10_000, min(1.0, t_ms / 300_000),
                               1.0 if t_ms >= 120_000 else 0.0,
                               1.0 if t_ms >= 180_000 else 0.0], dtype=np.float32),
        "hand": np.asarray(hand, dtype=np.int16),
        "next": (p["next_card"] + 2) if p.get("next_card", -1) >= 0 else 0,
        "opp_seen": np.asarray(seen + [0] * (8 - len(seen)), dtype=np.int16),
        "opp_last": np.asarray([c for c, _ in last] + [0] * (4 - len(last)), dtype=np.int16),
        "opp_last_age": np.asarray([min(1.0, (tick - t) / 400) for _, t in last]
                                   + [1.0] * (4 - len(last)), dtype=np.float16),
    }
    if vision:
        row["unit_form"] = forms
    if public:
        row["unit_burrow"] = burrow
    if press:
        cards, states = rr.rs_presses.ability_features(p, rr.BUTTONS, buttons)
        row["ability_card"] = np.asarray(cards, dtype=np.int16)
        row["ability_state"] = np.asarray(states, dtype=np.float16).reshape(
            rr.BUTTONS, len(rr.rs_presses.ABILITY_FEATS))
    if pending is not None:
        row["own_pending"], row["own_pending_land"] = rr.pending_features(pending)
    return row


GOLDEN_FLAGS = (
    {}, {"live": True}, {"units": "vision"}, {"visibility": "public"},
    {"presses": "on", "buttons": [5]}, {"pending": [(4, 10), (9, 50)], "hand_exact": False},
    {"pending": []}, {"units": "vision", "visibility": "public", "presses": "on",
                      "pending": [(7, 3)], "hand_exact": True},
    {"live": True, "visibility": "public", "presses": "on"},
)


@pytest.mark.parametrize("flags", GOLDEN_FLAGS, ids=lambda f: ",".join(sorted(f)) or "plain")
def test_legacy_obs_row_is_the_obs_row_before(v2rr, rs, flags):
    rr = v2rr
    rr.VC = StubLookup()
    at = FIELD["status_flags"]
    unreported = state_dict(300, entities=[e[:at] + [-1] + e[at + 1:] for e in _units()],
                            spells=_spells(), players=[player_dict(0, **BLUE), player_dict(1)])
    states = [big_state()] + ([] if flags.get("visibility") == "public" else [unreported])
    for state in states:
        for team in (0, 1):
            for tick in (4800, 300):
                want = frozen_obs_row(rr, state, team, SUB, MEM, tick, **flags)
                # E17 (b4455d7) floors opp_last_age at 0, the one change since fe0bd18: only
                # a memory holding a play after the row's tick (a clock read early) has a
                # negative age; a view never does. MEM's plays at 4000-4790 do at tick 300.
                if tick < 4790:
                    assert want["opp_last_age"].min() < 0
                want["opp_last_age"] = np.maximum(want["opp_last_age"], np.float16(0))
                same_row(rr.obs_row(state, team, SUB, MEM, tick, **flags), want)
                same_row(rr.obs_row(state, team, SUB, MEM, tick, schema="legacy", **flags), want)
                same_row(rr.obs_row(state, team, SUB, MEM, tick, schema=rs.LEGACY,
                                    opp_elixir=(1, True), **flags), want)
    extra = [k for k, on in (("unit_form", flags.get("units") == "vision"
                              and not flags.get("live")),
                             ("unit_burrow", flags.get("visibility") == "public"),
                             ("ability_card", flags.get("presses") == "on"),
                             ("ability_state", flags.get("presses") == "on"),
                             ("own_pending", "pending" in flags),
                             ("own_pending_land", "pending" in flags)) if on]
    row = rr.obs_row(big_state(), 0, SUB, MEM, 4800, **flags)
    assert tuple(row) == rs.LEGACY.obs_fields(extra)
    assert rs.LEGACY.check_row(row, buttons=2, extra=extra) == []


# ---------------------------------------------------------------------------
# rows_for and build
# ---------------------------------------------------------------------------

def test_rows_for_v2_on_the_fake_engine(rr, rs, engine, monkeypatch):
    mod = engine(states=states_a)
    old, old_info = run(rr)
    old_battle = mod.battles[-1]
    assert keys(old) == legacy_order() and len(old) == 52
    again, again_info = run(rr, schema="legacy")
    assert again_info == old_info and len(again) == 52
    for a, b in zip(again, old):
        same_row(a, b)
    rows, info = run(rr, schema="v2")
    assert keys(rows) == keys(old)
    assert mod.battles[-1].calls == old_battle.calls              # the engine is told the same
    assert info == {**old_info, "observation": {"rows": 52, "opp_inexact": 0, "spell_rows": 4}}
    for r, o in zip(rows, old):
        assert tuple(r) == V2_OBS + ("action", "outcome", "trusted", "won", "hog")
        assert rs.V2.check_row({k: r[k] for k in V2_OBS}, buttons=2) == []
        same(r, o, ("tick", "side", "unit_ids", "towers", "hand", "next", "opp_seen",
                    "opp_last", "opp_last_age"))
        assert (r["outcome"], r["trusted"], r["won"], r["hog"]) == (o["outcome"], o["trusted"],
                                                                    o["won"], o["hog"])
        assert np.array_equal(r["unit_feats"][:, :7], o["unit_feats"])
        assert np.array_equal(r["scalars"][:4], o["scalars"])
        team = 0 if r["side"] == 1 else 1
        want = red_bar(r["tick"]) if team == 0 else blue_bar(r["tick"])
        assert r["scalars"][4:].tolist() == f32([0, want / 10_000, 1]), (r["tick"], team)
        assert r["hand_exact"] is False and o["hand_exact"] is True   # two plays a side
        assert not r["own_pending"].any() and not r["hand_feats"].any()
        assert r["ability_card"].tolist() == ([7, 0] if team == 0 else [0, 0])
        assert not r["ability_state"].any()
        assert r["spell_ids"].tolist()[0] == (3 if r["tick"] in (300, 320) else 0)
    by = {(r["tick"], r["side"], int(r["action"][0])): r for r in rows}
    assert by[(300, 1, 4)]["scalars"][5] == np.float32(5357 / 10_000)
    assert by[(320, 1, 7)]["scalars"][5] == np.float32(5714 / 10_000)
    assert by[(105, -1, 10)]["scalars"][5] == np.float32(7875 / 10_000)
    assert by[(320, -1, 0)]["scalars"][5] == np.float32(6357 / 10_000)
    assert by[(300, -1, 0)]["spell_feats"][0].tolist() == f16([1, 0.5, 0.625, 14500 / 18000,
                                                               0.203125, 0.25])
    # obs_row's keywords: none on the legacy path, the v2 ones under v2
    seen = []
    real = rr.obs_row

    def spy(*args, **kwargs):
        seen.append(sorted(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(rr, "obs_row", spy)
    run(rr, schema="legacy")
    assert len(seen) == 52 and all(kw == [] for kw in seen)
    seen.clear()
    run(rr, schema="v2")
    assert len(seen) == 52
    assert all(kw == ["buttons", "hand_exact", "opp_elixir", "pending", "schema"] for kw in seen)


def test_v2_counts_only_the_plays_the_engine_ran(rr, engine):
    def results(battle, cmds, n):
        start = battle.tick - n
        return [[c[0], REFUSED if (start == 290 and c[0] == 1) else 0] for c in cmds]

    engine(states=states_a, step_results=results)
    rows, info = run(rr, schema="v2")
    assert info["trust"] == {"cut": 290, "why": "refused"}
    for r in rows:
        if r["side"] == 1:
            assert r["scalars"][5] == np.float32(red_bar(r["tick"], arrows=False) / 10_000)
    by = {(r["tick"], r["side"], int(r["action"][0])): r for r in rows}
    assert by[(300, 1, 4)]["scalars"][5] == np.float32(8357 / 10_000)    # no Arrows paid
    assert by[(520, 1, 0)]["scalars"][5] == np.float32(1.0)               # full since 392


def test_v2_decision_rows(rr, engine):
    engine(states=states_a)
    old, old_info = run(rr, decision=21)
    rows, info = run(rr, decision=21, schema="v2")
    assert keys(rows) == keys(old) == decision_order() and len(rows) == 54
    assert info == {**old_info, "observation": {"rows": 54, "opp_inexact": 0, "spell_rows": 4}}
    assert info["decision"] == DSTAT_A
    for r, o in zip(rows, old):
        same(r, o, ("own_pending", "own_pending_land", "hand_exact", "unit_ids", "hand"))
        team = 0 if r["side"] == 1 else 1
        want = red_bar(r["tick"]) if team == 0 else blue_bar(r["tick"])
        assert r["scalars"][5] == np.float32(want / 10_000), (r["tick"], team)
    by = {(r["tick"], r["side"], int(r["action"][0])): r for r in rows}
    hog, musk = by[(279, 1, 4)], by[(299, 1, 7)]
    assert hog["scalars"][5] == np.float32(7982 / 10_000)      # the Arrows (290) not yet run
    assert musk["scalars"][5] == np.float32(5339 / 10_000)
    assert musk["own_pending"].tolist() == [4, 0, 0, 0]
    assert musk["own_pending_land"].tolist() == f16([1 / 40, 0, 0, 0])
    assert by[(84, -1, 10)]["scalars"][5] == np.float32(0.75)
    assert by[(269, -1, 13)]["scalars"][5] == np.float32(1.0)


def test_v2_presses_are_paid_in_the_count(rr, engine):
    ability = [[1, 0, 2, 5, 0]]                       # the hero Musketeer: available, 2 elixir
    engine(states=lambda battle: states_a(battle, ability))
    rows, info = run(rr, payload_a(abilities=((500, "team", "musketeer"),)), schema="v2",
                     presses="on")
    assert info["presses"]["ok"] == 1 and info["presses"]["sent"] == 1
    by = {(r["tick"], r["side"], int(r["action"][0]), r["press"]): r for r in rows}
    press = by[(500, 1, 0, 1)]
    assert press["ability_card"].tolist() == [7, 0]
    assert press["ability_state"].tolist() == f16([[1, 1, 0, 0.2, 0], [0] * 5])
    assert tuple(press)[-6:] == ("action", "press", "outcome", "trusted", "won", "hog")
    for t, want in ((500, 5571), (520, 3928), (700, 7142)):
        assert by[(t, -1, 0, 0)]["scalars"][5] == np.float32(want / 10_000), t
        assert blue_bar(t, press_at=500, press_cost=2) == want
    assert max(r["tick"] for r in rows) == 700                # the press moves the end to 500 + 200


def test_build_v2_shard_and_manifest(rr, rs, engine, tmp_path):
    engine(states=states_a)
    payload = json.dumps(payload_a())
    bid, n, skipped, st = rr.build(1, [payload], False, str(tmp_path / "v2"), cfg(rr).to_json(),
                                   schema="v2", stats=True)
    assert (bid, n, skipped) == (1, 52, {}) and rr.ROW_SCHEMA == rs.SCHEMA_V2
    names, data = shard(tmp_path / "v2" / "shard-0001.npz")
    assert tuple(names) == V2_SHARD == rs.V2.shard_columns()
    assert rs.V2.check_shard(data, buttons=2) == []
    assert data["tick"].dtype == np.int32 and data["side"].dtype == np.int8
    assert data["scalars"].shape == (52, 7) and data["unit_feats"].shape == (52, 48, 9)
    assert not data["act_press"].any() and not data["hand_exact"].any()
    assert set(data["split"].tolist()) == {rr.split_of(TAG)}
    assert st["observation"] == {"rows": 52, "opp_inexact": 0, "spell_rows": 4}
    man = rr.manifest(st, [rr.shard_entry(1, st)], skipped)
    assert man["observation"] == st["observation"]
    assert rr.describe_manifest(man).endswith(
        "; schema v2: 4 of 52 rows show a spell in flight, 0 count the opponent's elixir "
        "inexactly")
    # --trust v2 adds its column; the module setting stays v2 until told otherwise
    rr.build(2, [payload], False, str(tmp_path / "v2"), cfg(rr).to_json(), trust="v2")
    names, data = shard(tmp_path / "v2" / "shard-0002.npz")
    assert tuple(names) == rs.V2.shard_columns(("trusted_v2",))
    assert rs.V2.check_shard(data, buttons=2, extra=("trusted_v2",)) == []
    # back to legacy: the legacy shard, and no observation counts
    _, n, _, st = rr.build(3, [payload], False, str(tmp_path / "old"), cfg(rr).to_json(),
                           schema="legacy", trust="legacy", stats=True)
    names, data = shard(tmp_path / "old" / "shard-0003.npz")
    assert n == 52 and tuple(names) == rs.LEGACY.shard_columns()
    assert rs.LEGACY.check_shard(data) == [] and "observation" not in st
    assert data["tick"].dtype == np.int64 and data["unit_feats"].shape == (52, 48, 7)
    with pytest.raises(ValueError, match="--schema v2"):
        rr.build(4, [payload], False, str(tmp_path), cfg(rr).to_json(), schema="v2",
                 mem_lag="legacy")
    with pytest.raises(ValueError, match="--schema v2"):
        rr.rows_for(payload_a(), rr._setup(cfg(rr)), cfg(rr), schema="v2", mem_lag="legacy")


def test_vocab_record_v2(rr, rs):
    cat, units = [list(r) for r in DEFAULT_CATALOGUE], list(DEFAULT_UNIT_TYPES)
    eng, fp = cfg(rr), {"version": "0.0.0+fake", "card_table": None}
    base = rr.vocab_record(cat, units, eng, fp, {}, False)
    for schema in ("legacy", "rs_rows/legacy", rs.LEGACY):
        assert rr.vocab_record(cat, units, eng, fp, {}, False, schema=schema) == base
    assert base["schema"] == "rs_rows/legacy" and "schema_spec" not in base
    flags = dict(trust="v2", presses="on", decision=21, visibility="public", opp_lag=1,
                 opp_memory="exec")
    old = rr.vocab_record(cat, units, eng, fp, {}, False, **flags)
    assert old["extra_columns"] == ["unit_burrow", "trusted_v2", "act_press", "ability_card",
                                    "ability_state", "own_pending", "own_pending_land"]
    new = rr.vocab_record(cat, units, eng, fp, {}, False, schema="v2", **flags)
    assert new["schema"] == "rs_rows/v2" and new["schema_spec"] == rs.V2.to_json(2)
    assert new["extra_columns"] == ["trusted_v2"] and list(new)[-2:] == ["schema_spec",
                                                                        "extra_columns"]
    assert {k: v for k, v in new.items() if k not in ("schema", "schema_spec", "extra_columns")
            } == {k: v for k, v in old.items() if k not in ("schema", "extra_columns")}
    assert rs.spec_problems(new) == [] and rs.of_meta(new) is rs.V2
    plain = rr.vocab_record(cat, units, eng, fp, {}, False, schema=rs.V2)
    assert "extra_columns" not in plain and plain["schema_spec"]["buttons"] == 2
    assert json.loads(json.dumps(new)) == new


def test_main_wires_the_schema(rr):
    text = (TOOLS / "rs_rows.py").read_text(encoding="utf-8")
    main = text[text.index("def main()"):]
    for needle in ('"--schema"', "choices=rs_schema.CHOICES", 'default="legacy"',
                   "ROW_SCHEMA = schema", "describe_schema(ROW_SCHEMA)",
                   "schema == rs_schema.SCHEMA_V2 and args.mem_lag == MEM_LEGACY"):
        assert needle in main, needle
    assert main.count("schema=ROW_SCHEMA") == 2               # vocab.json and every worker


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

@pytest.mark.engine
def test_real_engine_v2_rows_count_the_opponents_own_bar(rr, rs, monkeypatch):
    """Every v2 row's opponent elixir is the engine's own bar for that side at the row's tick
    (plain decks: no form, no Collector), and every row fits the spec."""
    import royalesim
    names = list(getattr(royalesim, "SPELL_FIELDS", ()) or ())
    assert set(rr.SPELL_REQUIRED) | {"ticks_flown"} <= set(names), names
    monkeypatch.setattr(rr.rs_elixir, "default_law", rr.rs_elixir.ElixirLaw.load)
    cfg_real = rr.rs_engine.EngineConfig.load()
    rr.ROW_SCHEMA = rs.SCHEMA_V2
    ctx = rr._setup(cfg_real)
    assert rr.SPELL_CARDS and rr.UNIT_FORMS and rr.SF == rr.spell_index(royalesim)
    seen = []
    real = rr.obs_row

    def spy(state, team, sub, mem, tick, **kwargs):
        seen.append((tick, team, state["players"][1 - team]["elixir_milli"],
                     kwargs["opp_elixir"]))
        return real(state, team, sub, mem, tick, **kwargs)

    monkeypatch.setattr(rr, "obs_row", spy)
    info = {}
    rows, why = rr.rows_for(il_payload(team_forms={}, opponent_forms={}, abilities=()), ctx,
                            cfg_real, info=info)
    assert why == "ok" and rows and len(seen) == len(rows)
    off = [s for s in seen if s[3] != (s[2], True)]
    assert not off, off[:5]
    for r in rows:
        assert rs.V2.check_row({k: r[k] for k in V2_OBS}, buttons=rr.BUTTONS) == []
    assert info["observation"]["opp_inexact"] == 0
