"""Shared setup for tests/pipeline: skip markers and fakes for the RoyaleSim pipeline.

Markers (also registered in pyproject.toml). A marked test is skipped, with the
reason shown, when what it needs is missing:

    engine     royalesim imports (.venvs/royalegym or .venvs/buildabot)
    il_replay  <data root>/il_replay/replays/part-*.parquet exists; the data root is
               HASTYCR_DATA / machine.json "data" / data/ (src/hastycr/machine.py)
    gpu        torch imports and torch.cuda.is_available()
    device     HASTYCR_DEVICE_TESTS=1 is set; the test may talk to the emulator

Helpers, importable as ``from tests.pipeline.conftest import FakeBattle``, and the
fixtures that hand them out:

    FakeBattle             fixtures make_fake_battle (the class), fake_battle (an instance)
    make_fake_royalesim    fixture fake_royalesim (installed as sys.modules["royalesim"])
    ENTITY_FIELDS, FIELD   the 28 entity columns in royalesim's order, and name -> index
    entity_row, tower_row  fixtures make_entity, make_tower
    player_dict, state_dict  fixtures make_player, make_state (state_json-shaped dicts)
    il_payload             fixture make_il_payload (a synthetic IL_Replay payload_json)
    il_replay_row, il_action_rows  the same match as parquet-style rows

Coordinates follow RoyaleSim: entity x / y are ENGINE-frame subtiles
(SUBTILE_PER_MILLITILE = 18 per millitile), blue (team 0) owns low y. IL_Replay
coordinates are native world units (millitiles, 18000 x 32000).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)          # hastycr, also without `pip install -e .`


# ---------------------------------------------------------------------------
# markers
# ---------------------------------------------------------------------------

MARKERS = {
    "engine": "needs royalesim; skipped when it does not import",
    "il_replay": "needs data/il_replay/replays/*.parquet; skipped when absent",
    "gpu": "needs torch with CUDA; skipped otherwise",
    "device": "talks to the emulator; runs only with HASTYCR_DEVICE_TESTS=1",
}


def pytest_configure(config):
    for name, text in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {text}")


def il_replay_dir() -> Path:
    """Where IL_Replay is expected: the machine's data root + il_replay."""
    try:
        from hastycr import machine
        return machine.get().data / "il_replay"
    except Exception:                       # a broken machine.json must not hide the suite
        return ROOT / "data" / "il_replay"


def _engine_missing() -> str | None:
    try:
        importlib.import_module("royalesim")
    except Exception as exc:                # ImportError, or a wheel built for another Python
        return f"royalesim does not import ({type(exc).__name__}: {exc})"
    return None


def _il_replay_missing() -> str | None:
    folder = il_replay_dir() / "replays"
    if not any(folder.glob("part-*.parquet")):
        return f"no IL_Replay parquet in {folder} (run tools/il_replay_download.py)"
    return None


def _gpu_missing() -> str | None:
    try:
        import torch
    except Exception as exc:
        return f"torch does not import ({type(exc).__name__})"
    if not torch.cuda.is_available():
        return f"CUDA not available (torch {torch.__version__}, CUDA build {torch.version.cuda})"
    return None


def _device_missing() -> str | None:
    if os.environ.get("HASTYCR_DEVICE_TESTS", "").strip() != "1":
        return "device test: set HASTYCR_DEVICE_TESTS=1 with the emulator running"
    return None


_CHECKS: dict[str, Callable[[], str | None]] = {
    "engine": _engine_missing,
    "il_replay": _il_replay_missing,
    "gpu": _gpu_missing,
    "device": _device_missing,
}
_REASONS: dict[str, str | None] = {}        # each check runs once per session


def pytest_runtest_setup(item):
    for name, check in _CHECKS.items():
        if item.get_closest_marker(name) is None:
            continue
        if name not in _REASONS:
            _REASONS[name] = check()
        if _REASONS[name]:
            pytest.skip(_REASONS[name])


# ---------------------------------------------------------------------------
# engine layout: entity rows, players, states
# ---------------------------------------------------------------------------

SUBTILE_PER_MILLITILE = 18

#: royalesim.ENTITY_FIELDS (royalegym protocol.py EntityState, 28 columns, in order).
ENTITY_FIELDS: tuple[str, ...] = (
    "uid", "team", "kind", "card_id", "tower_slot", "x", "y", "hp", "max_hp", "radius",
    "flying", "deploy_ticks", "stun_ticks", "knockback_ticks", "footprint", "target_uid",
    "attack_phase", "facing", "shield", "buffs", "status_flags", "level", "mount_uid",
    "charge", "dest_x", "dest_y", "ability_ticks", "unit_type",
)
FIELD: dict[str, int] = {name: i for i, name in enumerate(ENTITY_FIELDS)}

#: Values a royalesim entity row has when nothing else is said (engine-reported "none"s).
ENTITY_DEFAULTS: dict[str, Any] = {
    "uid": 0, "team": 0, "kind": 0, "card_id": -1, "tower_slot": -1, "x": 0, "y": 0,
    "hp": 100, "max_hp": 100, "radius": 500, "flying": False, "deploy_ticks": 0,
    "stun_ticks": 0, "knockback_ticks": 0, "footprint": None, "target_uid": -1,
    "attack_phase": 0, "facing": [0, 0], "shield": 0, "buffs": [], "status_flags": 0,
    "level": 11, "mount_uid": -1, "charge": 0, "dest_x": -1, "dest_y": -1,
    "ability_ticks": 0, "unit_type": -1,
}

KIND_TROOP, KIND_BUILDING, KIND_KING_TOWER, KIND_PRINCESS_TOWER = 0, 1, 2, 3
KING_HP, PRINCESS_HP = 4824, 3052          # level 11

#: Crown tower centres, (team, slot) -> engine-frame millitiles. Slot 0 is the king,
#: 1 and 2 the owner's own left and right (red is reflected in x, so its own left is
#: world right; tools/rs_rows.towers_vec).
TOWER_MILLI: dict[tuple[int, int], tuple[int, int]] = {
    (0, 0): (9000, 3000), (0, 1): (3500, 6500), (0, 2): (14500, 6500),
    (1, 0): (9000, 29000), (1, 1): (14500, 25500), (1, 2): (3500, 25500),
}

#: Deploy reason names, index = reason code in the fake. The real engine has these names
#: (checked by the engine-marked fixture test) but its order is only promised for "OK" = 0,
#: so tests look a code up by name: DEPLOY_REASONS.index("NO_HERO").
DEPLOY_REASONS: tuple[str, ...] = (
    "OK", "BAD_TEAM", "BAD_SLOT", "EMPTY_SLOT", "NOT_ENOUGH_ELIXIR", "OUT_OF_ARENA",
    "WATER", "NO_DEPLOY", "OUT_OF_TERRITORY", "OCCUPIED", "GAME_OVER", "DUPLICATE_TEAM",
    "ENGINE_ERROR", "TOO_EARLY", "NOTHING_TO_MIRROR", "NO_HERO", "ABILITY_NOT_READY",
    "ABILITY_SPENT", "CARD_PENDING",
)


def _copy(value):
    return list(value) if isinstance(value, list) else value


def entity_row(**fields) -> list:
    """One entity row in ENTITY_FIELDS order. hp alone also sets max_hp."""
    unknown = sorted(set(fields) - set(ENTITY_FIELDS))
    if unknown:
        raise TypeError(f"unknown entity field(s): {unknown}")
    values = {k: _copy(v) for k, v in ENTITY_DEFAULTS.items()}
    if "hp" in fields and "max_hp" not in fields:
        values["max_hp"] = fields["hp"]
    values.update(fields)
    return [values[name] for name in ENTITY_FIELDS]


def tower_row(team: int, slot: int, hp: int | None = None, max_hp: int | None = None,
              uid: int | None = None) -> list:
    """A crown tower entity (slot 0 king, 1 / 2 princess) at its TOWER_MILLI centre."""
    full = KING_HP if slot == 0 else PRINCESS_HP
    x, y = TOWER_MILLI[(team, slot)]
    return entity_row(
        uid=1 + 3 * team + slot if uid is None else uid, team=team,
        kind=KIND_KING_TOWER if slot == 0 else KIND_PRINCESS_TOWER, card_id=-1,
        tower_slot=slot, x=x * SUBTILE_PER_MILLITILE, y=y * SUBTILE_PER_MILLITILE,
        hp=full if hp is None else hp, max_hp=full if max_hp is None else max_hp,
        radius=1000 if slot == 0 else 900)


# ---------------------------------------------------------------------------
# catalogue (only the columns the pipeline reads are meaningful)
# ---------------------------------------------------------------------------

def catalogue_row(name: str, cost: int, kind: str, flying: bool = False) -> list:
    """A catalogue_json row: [0] name, [1] elixir, [5] flying, [8] TROOP/BUILDING/SPELL."""
    return [name, cost, 0, 0, 0, flying, 0, 0, kind]


DEFAULT_CATALOGUE: tuple[list, ...] = (
    catalogue_row("Cannon", 3, "BUILDING"), catalogue_row("Fireball", 4, "SPELL"),
    catalogue_row("HogRider", 4, "TROOP"), catalogue_row("IceGolemite", 2, "TROOP"),
    catalogue_row("IceSpirits", 1, "TROOP"), catalogue_row("Musketeer", 4, "TROOP"),
    catalogue_row("Skeletons", 1, "TROOP"), catalogue_row("Log", 2, "SPELL"),
    catalogue_row("Knight", 3, "TROOP"), catalogue_row("Archer", 3, "TROOP"),
    catalogue_row("Giant", 5, "TROOP"), catalogue_row("Arrows", 3, "SPELL"),
    catalogue_row("Valkyrie", 4, "TROOP"), catalogue_row("MiniPekka", 4, "TROOP"),
    catalogue_row("BabyDragon", 4, "TROOP", flying=True), catalogue_row("Zap", 2, "SPELL"),
)
DEFAULT_UNIT_TYPES: tuple[str, ...] = (
    "KingTower", "PrincessTower", "Cannon", "HogRider", "IceGolemite", "IceSpirit",
    "Musketeer", "Skeleton", "Knight", "Archer", "Giant", "Valkyrie", "MiniPekka",
    "BabyDragon",
)
_COST = {i: row[1] for i, row in enumerate(DEFAULT_CATALOGUE)}


def player_dict(team: int, *, elixir_milli: int = 5000, hand: Sequence[int] | None = None,
                next_card: int | None = None, crowns: int = 0,
                tower_hp: Sequence[int] = (KING_HP, PRINCESS_HP, PRINCESS_HP),
                tower_max_hp: Sequence[int] | None = None, king_active: bool = False,
                hand_costs: Sequence[int] | None = None, **extra) -> dict:
    """A state_json player. Team 0 holds catalogue cards 0-3 (next 4), team 1 cards 8-11 (next 12)."""
    base = 0 if team == 0 else 8
    hand = list(range(base, base + 4)) if hand is None else list(hand)
    player = {
        "team": team, "elixir_milli": elixir_milli, "hand": hand,
        "next_card": base + 4 if next_card is None else next_card, "crowns": crowns,
        "tower_hp": list(tower_hp),
        "tower_max_hp": list((KING_HP, PRINCESS_HP, PRINCESS_HP) if tower_max_hp is None
                             else tower_max_hp),
        "king_active": king_active, "abilities": [], "evo": [],
        "hand_costs": ([_COST.get(c, 3) for c in hand] if hand_costs is None
                       else list(hand_costs)),
        "mirror_target": -1, "pending": [], "pending_cost": 0,
    }
    player.update(extra)
    return player


def state_dict(tick: int = 0, *, players: Sequence[dict] | None = None,
               entities: Sequence[Sequence] = (), towers: bool = True,
               game_over: bool = False, winner: int = -1, spells: Sequence = (),
               projectiles: Sequence = (), overtime: bool | None = None,
               elixir_rate: int = 1, **extra) -> dict:
    """A state_json dict. With towers=True, each living crown tower (players' tower_hp > 0)
    is an entity ahead of `entities`, as the engine lists them."""
    players = [player_dict(0), player_dict(1)] if players is None else [dict(p) for p in players]
    rows: list[list] = []
    if towers:
        for team in (0, 1):
            hp, mx = players[team]["tower_hp"], players[team]["tower_max_hp"]
            rows += [tower_row(team, slot, hp=hp[slot], max_hp=mx[slot])
                     for slot in (0, 1, 2) if hp[slot] > 0]
    rows += [list(e) for e in entities]
    state = {
        "tick": tick, "tick_ms": tick * 50, "regular_ticks": 3600, "overtime_ticks": 2400,
        "elixir_rate": elixir_rate, "overtime": tick >= 3600 if overtime is None else overtime,
        "players": players, "entities": rows, "game_over": game_over, "winner": winner,
        "spells": [list(s) for s in spells], "projectiles": [list(p) for p in projectiles],
    }
    state.update(extra)
    return state


# ---------------------------------------------------------------------------
# FakeBattle and a fake royalesim module
# ---------------------------------------------------------------------------

class Call(NamedTuple):
    name: str
    args: tuple
    kwargs: dict


class FakeBattle:
    """A royalesim.Battle stand-in that records what it is told and replays a script.

    states        list of state dicts: state_json() returns states[k] after k step()
                  calls (the last one from then on); or callable(battle) -> dict;
                  default: state_dict(tick=battle.tick).
    step_results  list with one entry per step() call, used in order (then the default);
                  or callable(battle, cmds, n) -> list, called after the tick advanced.
                  Default: [[team, 0] for each command] (reason 0 = "OK").
    deploy        callable(team, slot, x, y) -> reason code, for check_deploy; default 0.
    catalogue, unit_types  what catalogue_json() / unit_types_json() return.

    Other keyword arguments (card_names, slot_of_k, calibration_overrides, level, ...)
    are kept in `init_kwargs`. `calls` lists Call(name, args, kwargs) for
    set_command_delay_ticks, reset, step, save and load in the order they came;
    `names()` gives just the names. check_deploy calls go to `deploy_checks`.
    """

    def __init__(self, *args, states=None, step_results=None, deploy=None,
                 catalogue=None, unit_types=None, **init_kwargs):
        self.init_args = args
        self.init_kwargs = init_kwargs
        self.calls: list[Call] = []
        self.deploy_checks: list[tuple[int, int, int, int]] = []
        self.state_reads = 0
        self._states = states
        self._results = step_results if callable(step_results) else list(step_results or [])
        self._deploy = deploy
        self.catalogue = [list(row) for row in (catalogue or DEFAULT_CATALOGUE)]
        self.unit_types = list(unit_types or DEFAULT_UNIT_TYPES)
        self.tick = 0
        self.steps = 0
        self.delay = (0, 0)
        self.reset_kwargs: dict | None = None
        self.pending: dict[int, list] = {0: [], 1: []}

    def names(self) -> list[str]:
        return [call.name for call in self.calls]

    def calls_named(self, name: str) -> list[Call]:
        return [call for call in self.calls if call.name == name]

    def set_command_delay_ticks(self, blue, red) -> None:
        self.calls.append(Call("set_command_delay_ticks", (blue, red), {}))
        self.delay = (int(blue), int(red))

    def reset(self, *args, **kwargs) -> None:
        self.calls.append(Call("reset", args, dict(kwargs)))
        self.reset_kwargs = dict(kwargs)
        self.tick = int(kwargs.get("start_tick") or 0)
        self.steps = 0

    def step(self, cmds=(), n=1):
        cmds = [tuple(c) for c in cmds]
        self.calls.append(Call("step", (cmds, n), {}))
        self.tick += int(n)
        self.steps += 1
        if callable(self._results):
            return self._results(self, cmds, n)
        if self._results:
            return self._results.pop(0)
        return [[c[0], 0] for c in cmds]

    def state(self) -> dict:
        if callable(self._states):
            return self._states(self)
        if not self._states:
            return state_dict(tick=self.tick)
        return self._states[min(self.steps, len(self._states) - 1)]

    def state_json(self) -> str:
        self.state_reads += 1
        return json.dumps(self.state())

    def check_deploy(self, team, slot, x, y) -> int:
        self.deploy_checks.append((int(team), int(slot), int(x), int(y)))
        return int(self._deploy(team, slot, x, y)) if self._deploy else 0

    def pending_commands(self, team) -> list:
        return list(self.pending.get(team, []))

    def save(self) -> dict:
        self.calls.append(Call("save", (), {}))
        return {"tick": self.tick, "steps": self.steps,
                "results": None if callable(self._results) else list(self._results)}

    def load(self, snapshot: dict) -> None:
        self.calls.append(Call("load", (), {}))
        self.tick, self.steps = snapshot["tick"], snapshot["steps"]
        if snapshot["results"] is not None:
            self._results = list(snapshot["results"])

    def catalogue_json(self) -> str:
        return json.dumps(self.catalogue)

    def unit_types_json(self) -> str:
        return json.dumps(self.unit_types)


def make_fake_royalesim(**battle_kwargs) -> types.ModuleType:
    """A module that looks enough like royalesim for the pipeline's own code.

    Battle(...) builds a FakeBattle with `module.battle_kwargs` (states, step_results,
    deploy, ...) merged under the caller's arguments, and appends it to `module.battles`.
    """
    mod = types.ModuleType("royalesim")
    mod.__version__ = "0.0.0+fake"
    mod.SUBTILE_PER_MILLITILE = SUBTILE_PER_MILLITILE
    mod.ENTITY_FIELDS = list(ENTITY_FIELDS)
    mod.DEPLOY_REASONS = list(DEPLOY_REASONS)
    mod.HAND_SIZE = 4
    mod.ABILITY_BUTTONS = 2
    mod.EMPTY_CARD = -1
    mod.battles = []
    mod.battle_kwargs = dict(battle_kwargs)

    def Battle(*args, **kwargs):
        battle = FakeBattle(*args, **{**mod.battle_kwargs, **kwargs})
        mod.battles.append(battle)
        return battle

    mod.Battle = Battle
    return mod


# ---------------------------------------------------------------------------
# synthetic IL_Replay payloads
# ---------------------------------------------------------------------------

#: Hog 2.6 as IL_Replay card keys (royaleimitate replay_cards.json names them
#: Cannon, Fireball, HogRider, IceGolemite, IceSpirits, Musketeer, Skeletons, Log).
HOG26_KEYS: tuple[str, ...] = ("cannon", "fireball", "hog-rider", "ice-golem", "ice-spirit",
                               "musketeer", "skeletons", "the-log")
OPPONENT_KEYS: tuple[str, ...] = ("knight", "archers", "giant", "arrows", "valkyrie",
                                  "mini-pekka", "baby-dragon", "zap")
FORM_SUFFIX = {0: "", 1: "-ev1", 2: "-hero"}

#: (tick, side, base card key, x, y): x / y native world units, team at low y.
DEFAULT_PLAYS: tuple[tuple[int, str, str, int, int], ...] = (
    (100, "team", "musketeer", 3500, 7500),
    (160, "opponent", "knight", 14500, 24500),
    (240, "team", "hog-rider", 3500, 14500),
    (300, "opponent", "arrows", 3500, 8000),
    (420, "team", "skeletons", 9000, 5000),
)
#: (tick, side, base card key): the hero Musketeer's ability.
DEFAULT_ABILITIES: tuple[tuple[int, str, str], ...] = ((500, "team", "musketeer"),)
DEFAULT_OPPONENT_LEVELS = (12, 11, 13, 11, 12, 11, 14, 11)


def _deck(keys: Sequence[str], forms: Mapping[str, int], levels) -> list[dict]:
    if len(keys) != 8 or len(set(keys)) != 8:
        raise ValueError(f"a deck is 8 distinct card keys, got {list(keys)}")
    if isinstance(levels, int):
        levels = [levels] * 8
    if len(levels) != 8:
        raise ValueError("levels: one int, or 8 of them")
    unknown = sorted(set(forms) - set(keys))
    if unknown:
        raise ValueError(f"forms for cards not in the deck: {unknown}")
    return [{"card_key": key + FORM_SUFFIX[int(forms.get(key, 0))], "level": int(level)}
            for key, level in zip(keys, levels)]


def il_payload(*, tag: str = "synthetic-0001",
               team_deck: Sequence[str] = HOG26_KEYS,
               opponent_deck: Sequence[str] = OPPONENT_KEYS,
               team_forms: Mapping[str, int] | None = None,
               opponent_forms: Mapping[str, int] | None = None,
               team_levels: int | Sequence[int] = 11,
               opponent_levels: int | Sequence[int] = DEFAULT_OPPONENT_LEVELS,
               plays: Sequence[tuple[int, str, str, int, int]] = DEFAULT_PLAYS,
               abilities: Sequence[tuple[int, str, str]] = DEFAULT_ABILITIES,
               team_low: bool = True, result: str = "victory",
               crowns: tuple[int, int] = (1, 0), game_mode: str = "Ladder",
               final_towers: Mapping[str, Mapping[str, int]] | None = None) -> dict:
    """A synthetic IL_Replay `payload_json` dict, shaped as tools/rs_rows.py reads it.

    Decks are base card keys; forms map a base key to 1 (evolution, "-ev1") or 2
    (hero, "-hero") and default to an evolved Skeletons and a hero Musketeer for the
    team, an evolved Knight for the opponent. Pass {} for plain decks. Levels are one
    int or 8. Plays are (tick, side, base key, x, y) with the team at low y; with
    team_low=False every coordinate is turned half a turn (team at high y). Play and
    ability events carry the deck's full key (suffix included); ability events also
    list it in ability_source_candidates, with zero coordinates.
    """
    team_forms = {"skeletons": 1, "musketeer": 2} if team_forms is None else dict(team_forms)
    opponent_forms = {"knight": 1} if opponent_forms is None else dict(opponent_forms)
    decks = {"team": _deck(team_deck, team_forms, team_levels),
             "opponent": _deck(opponent_deck, opponent_forms, opponent_levels)}
    full_key = {side: {d["card_key"].removesuffix("-ev1").removesuffix("-hero"): d["card_key"]
                       for d in deck} for side, deck in decks.items()}
    raw = []
    for tick, side, key, x, y in plays:
        if key not in full_key[side]:
            raise ValueError(f"{side} plays {key!r}, which is not in its deck")
        if not team_low:
            x, y = 18_000 - x, 32_000 - y
        raw.append((int(tick), "play_card", side, full_key[side][key], int(x), int(y)))
    for tick, side, key in abilities:
        if key not in full_key[side]:
            raise ValueError(f"{side} uses the ability of {key!r}, which is not in its deck")
        raw.append((int(tick), "activate_ability", side, full_key[side][key], 0, 0))
    raw.sort(key=lambda e: e[0])            # stable: same-tick events keep their order
    events = []
    for i, (tick, kind, side, key, x, y) in enumerate(raw):
        event = {"kind": kind, "side": side, "card_key": key, "replay_tick_20hz": tick,
                 "source_index": i, "coordinates": {"native_world_units": {"x": x, "y": y}}}
        if kind == "activate_ability":
            event["ability_source_candidates"] = [key]
        events.append(event)
    towers = final_towers or {
        "team": {"king": KING_HP, "princess_left": PRINCESS_HP, "princess_right": PRINCESS_HP},
        "opponent": {"king": KING_HP, "princess_left": 0, "princess_right": PRINCESS_HP},
    }

    def side_record(side: str, n_crowns: int) -> dict:
        return {"crowns": n_crowns, "players": [{
            "tag": f"#{side.upper()}0001", "name": side.title(), "deck": decks[side],
            "final_tower_hitpoints": dict(towers[side]),
            "tower_card": {"card_key": "tower-princess", "level": 11}}]}

    return {
        "source": {"replay_tag": tag, "dataset": "synthetic"},
        "battle": {"game_mode": game_mode, "battle_type": "PvP", "result": result,
                   "team": side_record("team", crowns[0]),
                   "opponent": side_record("opponent", crowns[1])},
        "events": events,
    }


def il_replay_row(payload: dict) -> dict:
    """The match as a row of IL_Replay's replays/*.parquet."""
    battle = payload["battle"]
    return {"replay_tag": payload["source"]["replay_tag"], "battle_type": battle["battle_type"],
            "game_mode": battle["game_mode"], "result": battle["result"],
            "payload_json": json.dumps(payload)}


def il_action_rows(payload: dict) -> list[dict]:
    """The match's events as rows of IL_Replay's actions/*.parquet (sim/replay_corpus.read_part)."""
    tag = payload["source"]["replay_tag"]
    return [{"replay_tag": tag, "event_index": e["source_index"], "kind": e["kind"],
             "side": e["side"], "card_key": e["card_key"],
             "replay_tick_20hz": e["replay_tick_20hz"],
             "native_x": e["coordinates"]["native_world_units"]["x"],
             "native_y": e["coordinates"]["native_world_units"]["y"],
             "ability_source_candidates": e.get("ability_source_candidates")}
            for e in payload["events"]]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_fake_battle():
    """The FakeBattle class: call it with states= / step_results= / deploy= to script one."""
    return FakeBattle


@pytest.fixture
def fake_battle():
    """A FakeBattle with the default script (state_dict at the current tick, OK results)."""
    return FakeBattle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]])


@pytest.fixture
def fake_royalesim(monkeypatch):
    """A fake royalesim in sys.modules for this test. Code that imported the real one
    earlier keeps it; import the code under test inside the test, or monkeypatch it."""
    mod = make_fake_royalesim()
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    return mod


@pytest.fixture
def make_entity():
    return entity_row


@pytest.fixture
def make_tower():
    return tower_row


@pytest.fixture
def make_player():
    return player_dict


@pytest.fixture
def make_state():
    return state_dict


@pytest.fixture
def make_il_payload():
    return il_payload
