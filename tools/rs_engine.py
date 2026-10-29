r"""One factory for every RoyaleSim battle the pipeline builds, and the engine's fingerprint.

    .venvs\royalegym\Scripts\python.exe tools\rs_engine.py                  the resolved settings
    .venvs\royalegym\Scripts\python.exe tools\rs_engine.py --probe          build a Battle, print
                                                                            its fingerprint
    .venvs\royalegym\Scripts\python.exe tools\rs_engine.py --probe --card-table client15535 --json

Why (rsim-01, rsim-03): the card table, the levels and the command delay used to be set
implicitly, and differently, at each of the 11 places that built a Battle. Every battle in
tools/rs_rows.py, rs_eval.py, royalesim_redrive.py and rs_shadow.py is now built by
make_battle(cfg), from one EngineConfig:

  card_table             a cards.CARD_TABLE arm of royalesim 0.1.42: client15535 (the engine's
                         own default, cards-15535.1), client160402017_20261006 (16.402.19),
                         client160402017_20261019 (16.402.22, the 2026-10-19 Season 88
                         mid-season update; TARGET_TABLE). Passed as
                         calibration_overrides={"cards.CARD_TABLE": '"<arm>"'}. Aliases: the
                         content version ("16.402.22") or the name Battle.card_table() reports
                         ("cards-160402017-20261019.1"). None / "default" passes no override.
  level, tower_level     the Battle constructor's unified card level and crown-tower level.
                         None leaves the engine's own (11 in 0.1.42). Friendly battles: 11.
  delay                  (blue, red) command delay in ticks. make_battle sets it with
                         set_command_delay_ticks BEFORE the first reset: set after a reset it
                         silently does nothing (rs_eval.py found that). None never calls it,
                         which leaves the engine's own 0.
  calibration_overrides  more ledger keys, "section.KEY" -> value as royalesim's
                         data/calibration.json writes it; JSON-encoded for the engine here.

Resolution (EngineConfig.load), first match wins per setting:
  1. explicit values: a tool's --card-table, or EngineConfig.load(card_table=...)
  2. environment variables HASTYCR_CARD_TABLE, HASTYCR_CARD_LEVEL, HASTYCR_TOWER_LEVEL,
     HASTYCR_COMMAND_DELAY ("21" or "21,22") and HASTYCR_CALIBRATION_OVERRIDES (a JSON object)
  3. data/engine.json, or the file HASTYCR_ENGINE_CONFIG names (relative to the repo root)
  4. built-in defaults, the same values data/engine.json ships with (BUILTIN)
"default", "none" or "null" sets a value back to "the engine's own".

Legacy: EngineConfig.legacy() (all None) builds exactly what every tool built before,
Battle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]]), with no delay call. A tool's
--engine-config legacy, or HASTYCR_ENGINE_CONFIG=legacy (worker processes included), starts
from it instead of data/engine.json. Only the old card table: --card-table default, or
HASTYCR_CARD_TABLE=default.

Worker processes: on Windows a ProcessPoolExecutor starts fresh interpreters, so a config a
tool's main() chose does not reach them through module state. The tools pass it with
ProcessPoolExecutor(..., initializer=rs_engine.set_active, initargs=(cfg.to_json(),)); code in
the worker reads it with active(). Environment variables reach workers anyway.

Re-drives of recordings (rs_rows, royalesim_redrive, rs_shadow, rs_eval's rebuilt look-ahead)
send each play at its recorded tick, which IL_Replay logs as the tick it executed. They run
with no command delay whatever the config says (EngineConfig.undelayed()).

fingerprint(battle) says which engine a battle is: royalesim version, card table name and
FNV-1a 64, unit_types_digest (FNV-1a 64 of unit_types_json(), computed here on engines that do
not report it), a SHA-1 of the catalogue names in id order, and the calibration overrides.
Unit vocabularies differ between tables (237 unit types on the default table on the PC); never
assume a count. field_index() checks royalesim.ENTITY_FIELDS against the 28 names the
pipeline reads (ENTITY_FIELDS_EXPECTED) and gives name -> index.

Exit codes of the CLI: 0 ok; 1 royalesim missing, refused the config, or ENTITY_FIELDS
differs; 2 a bad setting.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "data" / "engine.json"
CONFIG_ENV = "HASTYCR_ENGINE_CONFIG"
#: --engine-config / HASTYCR_ENGINE_CONFIG value that reads no file: EngineConfig.legacy(),
#: every Battle exactly as the tools built it before rs_engine.
LEGACY_CONFIG = "legacy"

#: The settings, in order, and the environment variable for each.
FIELDS: tuple[str, ...] = ("card_table", "level", "tower_level", "delay", "calibration_overrides")
ENV: dict[str, str] = {
    "card_table": "HASTYCR_CARD_TABLE",
    "level": "HASTYCR_CARD_LEVEL",
    "tower_level": "HASTYCR_TOWER_LEVEL",
    "delay": "HASTYCR_COMMAND_DELAY",
    "calibration_overrides": "HASTYCR_CALIBRATION_OVERRIDES",
}

#: The card table of the game build the sparring bot plays against: 16.402.22, the
#: 2026-10-19 Season 88 mid-season balance update (docs/SPARRING_PLAN.md).
TARGET_TABLE = "client160402017_20261019"
#: cards.CARD_TABLE arms in royalesim 0.1.42's ledger -> the version Battle.card_table()
#: reports for each (.so docstring of card_table). client160402017 (16.402.2, the install-time
#: pack) is read from a checkout only: a wheel carries no copy of it.
KNOWN_TABLES: dict[str, str] = {
    "client15535": "cards-15535.1",
    "client160402017_20261006": "cards-160402017-20261006.1",
    "client160402017_20261019": "cards-160402017-20261019.1",
    "client160402017": "cards-160402017.1",
}
#: Content versions people say -> arm names.
TABLE_ALIASES: dict[str, str] = {
    "15.535": "client15535",
    "15.535.29": "client15535",
    "16.402.2": "client160402017",
    "16.402.19": "client160402017_20261006",
    "16.402.22": "client160402017_20261019",
}
#: Values that mean "the engine's own": no override, no constructor argument, no delay call.
ENGINE_DEFAULT = frozenset({"", "default", "none", "null", "legacy", "engine"})
CARD_TABLE_KEY = "cards.CARD_TABLE"
SLOT_OF_K: tuple[tuple[int, ...], ...] = ((0, 1, 2), (0, 1, 2))
MAX_LEVEL = 16          # Commons go to 16 (cards-15.535.json level counts)

#: What data/engine.json ships with; used when the file is missing.
BUILTIN: dict[str, Any] = {
    "card_table": TARGET_TABLE,
    "level": 11,
    "tower_level": 11,
    "delay": (0, 0),
    "calibration_overrides": {},
}

#: royalesim.ENTITY_FIELDS as this pipeline reads it (royalegym protocol.py:497-526
#: EntityState, identical in royalesim 0.1.22 and 0.1.42). state_json sends each entity as
#: a JSON array decoded by position, so a field inserted or swapped must stop the pipeline.
ENTITY_FIELDS_EXPECTED: tuple[str, ...] = (
    "uid", "team", "kind", "card_id", "tower_slot", "x", "y", "hp", "max_hp", "radius",
    "flying", "deploy_ticks", "stun_ticks", "knockback_ticks", "footprint", "target_uid",
    "attack_phase", "facing", "shield", "buffs", "status_flags", "level", "mount_uid",
    "charge", "dest_x", "dest_y", "ability_ticks", "unit_type",
)

#: Fingerprint keys that identify the engine build and its data (fingerprint_id hashes these).
IDENTITY_KEYS: tuple[str, ...] = ("engine", "version", "card_table", "card_table_fnv1a64",
                                  "unit_types_digest", "catalogue_sha1", "calibration_overrides")

FNV1A64_OFFSET = 0xCBF29CE484222325
FNV1A64_PRIME = 0x100000001B3


class EngineError(RuntimeError):
    """royalesim refused or ignored the configuration, or its layout is not the expected one."""


# ---------------------------------------------------------------------------
# parsing settings
# ---------------------------------------------------------------------------

def _is_engine_default(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in ENGINE_DEFAULT)


def table_name(value: Any) -> str | None:
    """A card-table setting -> the cards.CARD_TABLE arm name, or None for the engine's own.

    Accepts an arm ("client160402017_20261019"), a content version ("16.402.22"), the
    version Battle.card_table() reports ("cards-160402017-20261019.1") and the JSON-quoted
    form the override takes ('"client15535"'). Other names pass through unchanged, so a
    newer royalesim's table can be named before this file knows it; the engine refuses a
    name it does not have.
    """
    if _is_engine_default(value):
        return None
    if not isinstance(value, str):
        raise ValueError(f"card_table: {value!r} is not a table name")
    text = value.strip().strip('"').strip()
    if text.lower() in ENGINE_DEFAULT:
        return None
    if text in TABLE_ALIASES:
        return TABLE_ALIASES[text]
    if text.startswith("cards-"):
        stem, dot, revision = text[len("cards-"):].rpartition(".")
        if dot and stem and revision.isdigit():
            return "client" + stem.replace("-", "_")
    return text


def table_version_stem(name: str) -> str | None:
    """'client160402017_20261019' -> 'cards-160402017-20261019': what Battle.card_table()
    reports for that arm, without its revision. None for a name not of that form."""
    if not name.startswith("client") or len(name) == len("client"):
        return None
    return "cards-" + name[len("client"):].replace("_", "-")


def parse_level(value: Any, what: str = "level") -> int | None:
    if _is_engine_default(value):
        return None
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{what}: {value!r} is not a level (1-{MAX_LEVEL}, or 'default')")
    try:
        level = int(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: {value!r} is not a level (1-{MAX_LEVEL}, or 'default')") from None
    if not 1 <= level <= MAX_LEVEL:
        raise ValueError(f"{what}: {level} is outside 1-{MAX_LEVEL}")
    return level


def parse_delay(value: Any) -> tuple[int, int] | None:
    """None / 'default' -> None; 21, '21', [21] -> (21, 21); '21,22', [21, 22] -> (21, 22)."""
    if _is_engine_default(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"delay: {value!r} is not a number of ticks")
    if isinstance(value, str):
        items: list[Any] = [p.strip() for p in value.replace("/", ",").split(",")]
    elif isinstance(value, (int, float)):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            raise ValueError(f"delay: {value!r} is not ticks or [blue, red]") from None
    if len(items) == 1:
        items = items * 2
    if len(items) != 2:
        raise ValueError(f"delay: {value!r} is not ticks or [blue, red]")
    try:
        blue, red = (int(v) for v in items)
    except (TypeError, ValueError):
        raise ValueError(f"delay: {value!r} is not ticks or [blue, red]") from None
    if blue < 0 or red < 0:
        raise ValueError(f"delay: {value!r} is negative")
    return blue, red


def parse_overrides(value: Any) -> dict[str, Any]:
    """{"section.KEY": value} (or its JSON text) -> a sorted dict; "_" keys are comments."""
    if value is None:
        return {}
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"calibration_overrides: not a JSON object ({exc})") from None
    if not isinstance(value, Mapping):
        raise ValueError("calibration_overrides: expected an object of 'section.KEY': value")
    out: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"calibration_overrides: key {key!r} is not text")
        if key.startswith("_"):
            continue
        section, dot, name = key.partition(".")
        if not (section and dot and name):
            raise ValueError(f"calibration_overrides: {key!r} is not 'section.KEY'")
        if key == CARD_TABLE_KEY:
            raise ValueError("calibration_overrides: set the card table with card_table, "
                             f"not {CARD_TABLE_KEY}")
        out[key] = item
    return dict(sorted(out.items()))


_PARSERS: dict[str, Callable[[Any], Any]] = {
    "card_table": table_name,
    "level": lambda v: parse_level(v, "level"),
    "tower_level": lambda v: parse_level(v, "tower_level"),
    "delay": parse_delay,
    "calibration_overrides": parse_overrides,
}


def read_config_file(path: Path) -> dict[str, Any]:
    """The settings in an engine.json; keys starting with "_" are comments."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not JSON ({exc})") from None
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    unknown = sorted(k for k in data if not str(k).startswith("_") and k not in FIELDS)
    if unknown:
        raise ValueError(f"{path}: unknown setting(s) {unknown}; the settings are {list(FIELDS)}")
    return {k: data[k] for k in FIELDS if k in data}


def _shown(source: str) -> str:
    """A source for messages: a file under the repo as a repo path, anything else as is."""
    path = Path(source)
    if not path.is_absolute():
        return source
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return source


# ---------------------------------------------------------------------------
# the configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EngineConfig:
    """How to build a Battle. EngineConfig() is legacy: nothing passed, nothing set.

    Values are normalised on construction (table aliases, "default" -> None, delay -> a
    (blue, red) tuple), so equal settings compare equal. `sources` says where each came
    from, for messages; it is not compared.
    """

    card_table: str | None = None
    level: int | None = None
    tower_level: int | None = None
    delay: tuple[int, int] | None = None
    calibration_overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    sources: Mapping[str, str] = dataclasses.field(default_factory=dict, compare=False,
                                                   repr=False)

    def __post_init__(self) -> None:
        for name in FIELDS:
            object.__setattr__(self, name, _PARSERS[name](getattr(self, name)))
        object.__setattr__(self, "sources", dict(self.sources or {}))

    def __hash__(self) -> int:
        return hash(self.key())

    # ----------------------------------------------------------- constructors
    @classmethod
    def legacy(cls) -> "EngineConfig":
        """Exactly what the tools built before: Battle(card_names=None, slot_of_k=...), no delay call."""
        return cls(sources=dict.fromkeys(FIELDS, "legacy"))

    @classmethod
    def defaults(cls) -> "EngineConfig":
        """The built-in defaults (BUILTIN), what data/engine.json ships with."""
        return cls(**BUILTIN, sources=dict.fromkeys(FIELDS, "built-in"))

    @classmethod
    def load(cls, path: str | os.PathLike | None = None, *, env: Mapping[str, str] | None = None,
             **explicit: Any) -> "EngineConfig":
        """Resolve every setting: explicit > environment > engine.json > BUILTIN.

        path: an engine.json to read instead of data/engine.json (as given, relative to the
        working directory); otherwise HASTYCR_ENGINE_CONFIG (relative to the repo root), then
        data/engine.json. A named file must exist; a missing data/engine.json falls back to
        BUILTIN. The name "legacy" (LEGACY_CONFIG) reads no file and starts from
        EngineConfig.legacy() instead. Explicit values that are None count as not given;
        pass "default" for the engine's own.
        """
        unknown = sorted(set(explicit) - set(FIELDS))
        if unknown:
            raise TypeError(f"EngineConfig.load: unknown setting(s) {unknown}")
        env = os.environ if env is None else env
        values: dict[str, Any] = dict(BUILTIN)
        sources: dict[str, str] = dict.fromkeys(FIELDS, "built-in")
        named = ""
        file: Path | None = CONFIG_PATH
        given = os.fspath(path) if path is not None else (env.get(CONFIG_ENV) or "").strip().strip('"')
        if given.strip().lower() == LEGACY_CONFIG:
            file = None
            values = {**dict.fromkeys(FIELDS), "calibration_overrides": {}}
            sources = dict.fromkeys(FIELDS, "legacy")
        elif path is not None:
            file, named = Path(path), "--engine-config"
        elif given:
            file, named = Path(os.path.expandvars(os.path.expanduser(given))), CONFIG_ENV
            if not file.is_absolute():
                file = ROOT / file
        if file is not None and file.is_file():
            shown = str(file.resolve())
            for key, value in read_config_file(file).items():
                values[key], sources[key] = value, shown
        elif named:
            raise FileNotFoundError(f"engine config {file} ({named}) does not exist")
        for key, var in ENV.items():
            raw = env.get(var)
            if raw is not None and raw.strip():
                values[key], sources[key] = raw, var
        for key, value in explicit.items():
            if value is not None:
                values[key], sources[key] = value, "caller"
        for key in FIELDS:              # say where a bad value came from
            try:
                _PARSERS[key](values[key])
            except ValueError as exc:
                raise ValueError(f"{exc} (from {_shown(sources[key])})") from None
        return cls(**values, sources=sources)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "EngineConfig":
        """The inverse of to_json. Missing settings are the engine's own."""
        if not isinstance(data, Mapping):
            raise TypeError(f"expected an engine config object, got {type(data).__name__}")
        unknown = sorted(k for k in data if not str(k).startswith("_") and k not in FIELDS)
        if unknown:
            raise ValueError(f"engine config: unknown setting(s) {unknown}")
        return cls(**{k: data[k] for k in FIELDS if k in data},
                   sources=dict.fromkeys((k for k in FIELDS if k in data), "json"))

    @classmethod
    def coerce(cls, value: Any) -> "EngineConfig":
        """An EngineConfig (also one from a second import of this module) or its to_json()."""
        if type(value) is cls:
            return value
        if hasattr(value, "to_json") and hasattr(value, "card_table"):
            return cls.from_json(value.to_json())
        if isinstance(value, Mapping):
            return cls.from_json(value)
        raise TypeError(f"not an engine config: {value!r}")

    # ---------------------------------------------------------------- derived
    def replace(self, **changes: Any) -> "EngineConfig":
        unknown = sorted(set(changes) - set(FIELDS))
        if unknown:
            raise TypeError(f"EngineConfig.replace: unknown setting(s) {unknown}")
        sources = dict(self.sources)
        sources.update(dict.fromkeys(changes, "caller"))
        return dataclasses.replace(self, **changes, sources=sources)

    def delayed(self, ticks: int) -> "EngineConfig":
        """Both seats' command delay set to `ticks`. A legacy config (delay None) asked for 0
        stays as it is, so no set_command_delay_ticks call is added where there was none."""
        if not ticks and self.delay is None:
            return self
        return self.replace(delay=(int(ticks), int(ticks)))

    def undelayed(self) -> "EngineConfig":
        """For re-driving recorded plays at their recorded (execution) ticks: delay 0."""
        return self.delayed(0)

    @property
    def is_legacy(self) -> bool:
        return self == EngineConfig()

    def encoded_overrides(self) -> dict[str, str]:
        """calibration_overrides as the engine takes them: "section.KEY" -> JSON text."""
        out: dict[str, str] = {}
        if self.card_table is not None:
            out[CARD_TABLE_KEY] = json.dumps(self.card_table)
        out.update((k, json.dumps(v)) for k, v in self.calibration_overrides.items())
        return out

    def constructor_kwargs(self) -> dict[str, Any]:
        """The keyword arguments make_battle passes to royalesim.Battle. A setting that is
        the engine's own is left out, so legacy() gives exactly the old call."""
        kwargs: dict[str, Any] = {"card_names": None, "slot_of_k": [list(s) for s in SLOT_OF_K]}
        overrides = self.encoded_overrides()
        if overrides:
            kwargs["calibration_overrides"] = overrides
        if self.level is not None:
            kwargs["level"] = self.level
        if self.tower_level is not None:
            kwargs["tower_level"] = self.tower_level
        return kwargs

    def to_json(self) -> dict[str, Any]:
        return {"card_table": self.card_table, "level": self.level,
                "tower_level": self.tower_level,
                "delay": None if self.delay is None else list(self.delay),
                "calibration_overrides": dict(self.calibration_overrides)}

    def key(self) -> str:
        """A stable text form, for caches and comparisons."""
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))

    def describe(self) -> str:
        def src(name: str) -> str:
            where = self.sources.get(name)
            return f" [{_shown(where)}]" if where else ""
        delay = "engine default (not set)" if self.delay is None else f"{self.delay[0]}/{self.delay[1]}"
        parts = [
            f"card table {self.card_table or 'engine default (no override)'}{src('card_table')}",
            f"level {self.level or 'engine default'}{src('level')}",
            f"tower level {self.tower_level or 'engine default'}{src('tower_level')}",
            f"delay {delay}{src('delay')}",
        ]
        if self.calibration_overrides:
            parts.append(f"overrides {json.dumps(dict(self.calibration_overrides))}"
                         f"{src('calibration_overrides')}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# the process's config (set in each worker by the pool initializer)
# ---------------------------------------------------------------------------

_ACTIVE: EngineConfig | None = None


def set_active(cfg: Any) -> EngineConfig | None:
    """Make `cfg` (an EngineConfig or its to_json()) the config active() returns in this
    process; None forgets it, so the next active() loads again. Picklable by reference, for
    ProcessPoolExecutor(initializer=rs_engine.set_active, initargs=(cfg.to_json(),))."""
    global _ACTIVE
    _ACTIVE = None if cfg is None else EngineConfig.coerce(cfg)
    return _ACTIVE


def active() -> EngineConfig:
    """The config set with set_active, else EngineConfig.load() (cached for the process)."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = EngineConfig.load()
    return _ACTIVE


def resolve(engine: Any = None) -> EngineConfig:
    """`engine` as an EngineConfig; None means active()."""
    return active() if engine is None else EngineConfig.coerce(engine)


# ---------------------------------------------------------------------------
# building battles
# ---------------------------------------------------------------------------

def royalesim_module(rs: ModuleType | None = None) -> ModuleType:
    """`rs`, or royalesim imported now (so a module put in sys.modules by a test is used)."""
    return importlib.import_module("royalesim") if rs is None else rs


def make_battle(cfg: Any = None, rs: ModuleType | None = None) -> Any:
    """A new royalesim.Battle under `cfg` (None: active()), command delay already set.

    The delay is set before the caller's first reset, which is the only time it takes effect.
    With a card table named, the Battle's card_table() (royalesim 0.1.42+) must report it.
    """
    cfg = resolve(cfg)
    rs = royalesim_module(rs)
    kwargs = cfg.constructor_kwargs()
    try:
        battle = rs.Battle(**kwargs)
    except Exception as exc:
        extra = {k: v for k, v in kwargs.items() if k not in ("card_names", "slot_of_k")}
        if not extra:
            raise
        version = getattr(rs, "__version__", "?")
        raise EngineError(
            f"royalesim {version} would not build a Battle with {extra}: "
            f"{type(exc).__name__}: {exc}. A card table needs royalesim 0.1.42 or later "
            "(pinned in requirements/); --card-table default or HASTYCR_CARD_TABLE=default "
            "runs the engine's own table.") from exc
    if cfg.delay is not None:
        battle.set_command_delay_ticks(*cfg.delay)
    check_card_table(battle, cfg)
    return battle


def _read(obj: Any, name: str) -> Any:
    """obj.name() when it is a method, obj.name when it is a value, None when absent."""
    value = getattr(obj, name, None)
    return value() if callable(value) else value


def check_card_table(battle: Any, cfg: EngineConfig) -> str | None:
    """Raise EngineError when `battle` reports a card table other than the one `cfg` names.
    Returns what it reports (None for an engine without card_table(), or no table named)."""
    if cfg.card_table is None:
        return None
    got = _read(battle, "card_table")
    if got is None:
        return None
    got = str(got)
    stem = table_version_stem(cfg.card_table)
    if stem and got != stem and got.rpartition(".")[0] != stem:
        raise EngineError(f"asked for card table {cfg.card_table} ({stem}.*), but the Battle "
                          f"runs {got}")
    return got


def catalogue(cfg: Any = None, rs: ModuleType | None = None) -> list:
    """catalogue_json() of a Battle under `cfg`: one row per card id."""
    return json.loads(make_battle(cfg, rs).catalogue_json())


def unit_types(cfg: Any = None, rs: ModuleType | None = None) -> list[str]:
    """unit_types_json() of a Battle under `cfg`: unit names in unit_type id order."""
    return json.loads(make_battle(cfg, rs).unit_types_json())


# ---------------------------------------------------------------------------
# entity layout
# ---------------------------------------------------------------------------

def check_entity_fields(rs: ModuleType | None = None) -> tuple[str, ...]:
    """royalesim.ENTITY_FIELDS, after checking it is ENTITY_FIELDS_EXPECTED (EngineError if not)."""
    rs = royalesim_module(rs)
    got = tuple(getattr(rs, "ENTITY_FIELDS", None) or ())
    if got == ENTITY_FIELDS_EXPECTED:
        return got
    if not got:
        raise EngineError(f"royalesim {getattr(rs, '__version__', '?')} exports no ENTITY_FIELDS")
    for i, (have, want) in enumerate(zip(got, ENTITY_FIELDS_EXPECTED)):
        if have != want:
            raise EngineError(f"royalesim.ENTITY_FIELDS[{i}] is {have!r}, the pipeline reads "
                              f"{want!r} there ({len(got)} fields, expected "
                              f"{len(ENTITY_FIELDS_EXPECTED)})")
    raise EngineError(f"royalesim.ENTITY_FIELDS has {len(got)} fields, the pipeline reads "
                      f"{len(ENTITY_FIELDS_EXPECTED)}")


def field_index(rs: ModuleType | None = None) -> dict[str, int]:
    """Entity field name -> column, after check_entity_fields."""
    return {name: i for i, name in enumerate(check_entity_fields(rs))}


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------

def fnv1a64(data: bytes) -> str:
    """FNV-1a 64 of `data`, 16 lowercase hex digits (royalegym protocol.fnv1a64)."""
    h = FNV1A64_OFFSET
    for byte in data:
        h = ((h ^ byte) * FNV1A64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    return f"{h:016x}"


def _hex64(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value & 0xFFFF_FFFF_FFFF_FFFF:016x}"
    return str(value)


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def fingerprint(battle: Any, rs: ModuleType | None = None, cfg: Any = None) -> dict[str, Any]:
    """Which engine and data `battle` runs. Keys:

    engine, version, provenance ([commit, tree] or None), card_table and card_table_fnv1a64
    (None before royalesim 0.1.42), unit_types (count) and unit_types_digest, catalogue
    (count) and catalogue_sha1 (SHA-1 of the card names, one per line, in id order),
    calibration_overrides (what the battle reports; else `cfg`'s, encoded), card_level,
    tower_level, command_delay_ticks (None when the battle does not report them),
    entity_fields (len of royalesim.ENTITY_FIELDS) and id (fingerprint_id).
    """
    rs = royalesim_module(rs)
    units_text = battle.unit_types_json()
    units = json.loads(units_text)
    names = [str(row[0]) for row in json.loads(battle.catalogue_json())]
    digest = _read(battle, "unit_types_digest")
    overrides = _read(battle, "calibration_overrides")
    if isinstance(overrides, str):          # JSON text of the object, should an engine say so
        overrides = json.loads(overrides) if overrides.strip() else {}
    if overrides is None:
        overrides = resolve(cfg).encoded_overrides() if cfg is not None else {}
    provenance = _read(rs, "provenance")
    delay = _read(battle, "command_delay_ticks")
    table = _read(battle, "card_table")
    fp: dict[str, Any] = {
        "engine": "royalesim",
        "version": str(getattr(rs, "__version__", "?")),
        "provenance": None if provenance is None else [str(p) for p in provenance],
        "card_table": None if table is None else str(table),
        "card_table_fnv1a64": _hex64(_read(battle, "card_table_fnv1a64")),
        "unit_types": len(units),
        "unit_types_digest": (str(digest) if digest is not None
                              else fnv1a64(units_text.encode("utf-8"))),
        "catalogue": len(names),
        "catalogue_sha1": hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest(),
        "calibration_overrides": {str(k): (v if isinstance(v, str) else json.dumps(v))
                                  for k, v in sorted(dict(overrides).items())},
        "card_level": _opt_int(_read(battle, "card_level")),
        "tower_level": _opt_int(_read(battle, "tower_level")),
        "command_delay_ticks": None if delay is None else [int(d) for d in delay],
        "entity_fields": len(getattr(rs, "ENTITY_FIELDS", None) or ()),
    }
    fp["id"] = fingerprint_id(fp)
    return fp


def fingerprint_id(fp: Mapping[str, Any]) -> str:
    """16 hex digits over IDENTITY_KEYS: equal means the same engine build, table and vocabulary."""
    blob = json.dumps({k: fp.get(k) for k in IDENTITY_KEYS}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# command line, shared with the tools
# ---------------------------------------------------------------------------

def add_engine_args(ap: argparse.ArgumentParser, *, levels: bool = False,
                    delay: bool = False) -> argparse.ArgumentParser:
    """--card-table and --engine-config (and --level / --tower-level / --delay) on `ap`."""
    group = ap.add_argument_group("engine (tools/rs_engine.py)")
    group.add_argument("--card-table", dest="engine_card_table", default=None, metavar="TABLE",
                       help=f"cards.CARD_TABLE arm, e.g. {TARGET_TABLE} or client15535; "
                            "'default' = no override, the engine's own table (legacy). "
                            "Default: HASTYCR_CARD_TABLE, then data/engine.json")
    group.add_argument("--engine-config", dest="engine_config", default=None, metavar="FILE",
                       help="an engine.json to read instead of data/engine.json; 'legacy' = "
                            "build every Battle exactly as before rs_engine")
    if levels:
        group.add_argument("--level", dest="engine_level", default=None,
                           help="constructor card level (1-16, or 'default')")
        group.add_argument("--tower-level", dest="engine_tower_level", default=None,
                           help="constructor crown-tower level (1-16, or 'default')")
    if delay:
        group.add_argument("--delay", dest="engine_delay", default=None, metavar="TICKS",
                           help="command delay, 'B' or 'B,R' ticks, or 'default'")
    return ap


def config_from_args(args: argparse.Namespace, *, env: Mapping[str, str] | None = None
                     ) -> EngineConfig:
    """EngineConfig.load with what add_engine_args parsed."""
    return EngineConfig.load(getattr(args, "engine_config", None), env=env,
                             card_table=getattr(args, "engine_card_table", None),
                             level=getattr(args, "engine_level", None),
                             tower_level=getattr(args, "engine_tower_level", None),
                             delay=getattr(args, "engine_delay", None))


def summary(fp: Mapping[str, Any]) -> str:
    """One line for a tool's log."""
    return (f"royalesim {fp.get('version')} card table {fp.get('card_table') or 'n/a'} "
            f"({len(fp.get('calibration_overrides') or {})} overrides), "
            f"{fp.get('unit_types')} unit types, {fp.get('catalogue')} cards, id {fp.get('id')}")


def _report(cfg: EngineConfig, fp: Mapping[str, Any], fields_error: str | None) -> str:
    delay = fp.get("command_delay_ticks")
    prov = fp.get("provenance")
    lines = [
        f"config      {cfg.describe()}",
        f"royalesim   {fp['version']}" + (f"  provenance {' '.join(prov)}" if prov else ""),
        f"card table  {fp['card_table'] or 'n/a (royalesim before 0.1.42)'}"
        + (f"  fnv1a64 {fp['card_table_fnv1a64']}" if fp.get("card_table_fnv1a64") else ""),
        f"levels      card {fp['card_level'] if fp['card_level'] is not None else 'n/a'}, "
        f"tower {fp['tower_level'] if fp['tower_level'] is not None else 'n/a'}, command delay "
        + (f"{delay[0]}/{delay[1]}" if delay else "n/a"),
        f"units       {fp['unit_types']} unit types, digest {fp['unit_types_digest']}",
        f"catalogue   {fp['catalogue']} cards, sha1 {fp['catalogue_sha1']}",
        f"overrides   {json.dumps(fp['calibration_overrides'])}",
        f"entities    {fp['entity_fields']} ENTITY_FIELDS, "
        + ("as expected" if fields_error is None else f"DIFFERENT: {fields_error}"),
        f"id          {fp['id']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="RoyaleSim engine settings; --probe builds a Battle and prints its fingerprint.")
    ap.add_argument("--probe", action="store_true",
                    help="build a Battle with these settings and print its fingerprint")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    add_engine_args(ap, levels=True, delay=True)
    args = ap.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"engine config: {exc}", file=sys.stderr)
        return 2
    if not args.probe:
        if args.json:
            print(json.dumps({"config": cfg.to_json(),
                              "sources": {k: _shown(v) for k, v in cfg.sources.items()}}, indent=2))
        else:
            print(cfg.describe())
        return 0
    try:
        rs = royalesim_module()
    except Exception as exc:
        print(f"royalesim does not import ({type(exc).__name__}: {exc}). Run this with the "
              r"engine venv: .venvs\royalegym\Scripts\python.exe tools\rs_engine.py --probe",
              file=sys.stderr)
        return 1
    try:
        battle = make_battle(cfg, rs)
        fp = fingerprint(battle, rs, cfg)
    except Exception as exc:                # EngineError, or the engine's own refusal
        print(f"engine: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    fields_error = None
    try:
        check_entity_fields(rs)
    except EngineError as exc:
        fields_error = str(exc)
    if args.json:
        print(json.dumps({"config": cfg.to_json(), "fingerprint": fp,
                          "entity_fields_ok": fields_error is None}, indent=2))
    else:
        print(_report(cfg, fp, fields_error))
    if fields_error is not None:
        print(f"engine: {fields_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
