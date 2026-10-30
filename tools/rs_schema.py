r"""Row schemas: the columns tools/rs_rows.py writes into a shard, and what each one holds (E14).

    from tools import rs_schema as RS            # or `import rs_schema` with tools/ on sys.path
    spec = RS.get("v2")                           # RS.of_meta(meta) for a checkpoint or vocab.json
    spec.shape("ability_state", buttons=3)        # (3, 5)
    spec.train_columns                            # the columns sim/train_tokens.py loads
    spec.check_row(row, buttons=3)                # [] when obs_row's row fits the spec
    .venvs\royalegym\Scripts\python.exe tools\rs_schema.py --schema v2 --buttons 3

Why (tokens-14): the legacy row has no spells in flight, no opponent elixir, no own pending
plays, no evolution or ability state and no unit form, although the screen shows all of them and
strong players use them. Schema v2 adds them. A row's layout is a contract between the row
builder (tools/rs_rows.py), the trainer (sim/train_tokens.py), the model (sim/model_tokens.py),
the evaluators and the live runner, so it lives here, once, as data: every consumer reads the
column names, shapes, dtypes and feature names from this module instead of repeating them.

TWO SCHEMAS. vocab.json "schema" names the one a shard was built with, and every checkpoint
copies it into its meta; a meta without the key is legacy (of_meta).

  rs_rows/legacy  the rows before E14, byte for byte: rs_rows.FIELDS plus the columns the
                  batch-3/4 flags add (`extra`: --units vision unit_form, --visibility public
                  unit_burrow, --trust v2 trusted_v2, --presses on act_press / ability_card /
                  ability_state, --obs-delay own_pending / own_pending_land). The legacy model
                  reads none of the extras.
  rs_rows/v2      tools/rs_rows.py --schema v2. Every row has the same columns whatever the
                  flags (only --trust v2 still adds trusted_v2):
                    unit_feats  the 7 legacy features plus "form" (0 base, 0.5 evolution,
                                1 hero) and "burrow" (1 for a unit tunnelling now); the legacy
                                unit_form / unit_burrow columns are gone
                    scalars     the 4 legacy ones plus "triple" (3x elixir, from 240 s),
                                "opp_elixir" (the opponent's bar / 10000 as the acting player
                                counts it from the plays it has seen, tools/rs_elixir.py) and
                                "opp_exact" (1 while that count is known to be exact). Own
                                elixir stays the raw bar, as the client shows it (tokens-19).
                    hand_feats  per hand slot: the card's form in the deck / 2, whether its
                                next play is evolved, and its evolution charge (players[t]
                                ["evo"]: plays since its last evolved play / plays it needs)
                    spell_ids, spell_feats
                                up to SPELL_SLOTS cast spells in flight (the objects of a
                                SPELL card in state["spells"], what the screen shows as a
                                spell): the card (catalogue id + 2), and own, its centre and
                                its aim in the acting player's frame, ticks_flown / 40
                    own_pending, own_pending_land
                                the side's own plays tapped and not yet run (E11; zeros for
                                rows at execution)
                    ability_card, ability_state
                                the side's B ability buttons (E10; B = royalesim
                                ABILITY_BUTTONS, resolved when a shape is asked for)
                    act_press   0, or button + 1 on a press row (E10)
                  and narrower dtypes (tick int32, side int8, next int16, outcome and won
                  float32), which build() casts to.

A COLUMN (Column): name; shape per row (ints, BUTTONS = "B" for the ability buttons); dtype as
the shard stores it ("str" for the match tag and split); role (ROLE_OBS a model input, ROLE_INFO
a fact about the row, ROLE_LABEL a training target, ROLE_FILTER a row filter, ROLE_KEY the match
and split); source (SOURCE_OBS what obs_row returns, SOURCE_ROWS what rows_for adds,
SOURCE_BUILD what build adds); extra (None: always there, else the flag that adds it); feats
(the names along the last axis); mirror (below); compact (E18: the dtype a compacted column can
take without losing anything; None: the dtype itself). A shard's columns are its schema's
`columns` that are not extras, plus the extras vocab.json "extra_columns" lists, in the order
of `columns` (shard_columns): first the row columns, then act_card, act_x, act_y, act_press,
match and split, as build writes them.

MIRROR (E18, --mirror): how a left-right flip of the arena changes a column, as data:
  {"flip": [k, ...], "where": ID}   feature k becomes 1 - k, in the slots whose ID column is
                                    not 0 (own-frame x / 18000; padding stays 0)
  {"swap": [[a, b], ...]}           the values at a and b trade places (towers: left <-> right
                                    on both sides)
  {"tile": 17, "where": ID}         the value becomes 17 - value where ID > 0 (act_x of a play)
Every other column is unchanged by a flip: the units are sorted by distance from the own king,
which stands on the centre line, so their order does not change.

GROUPS (E15): the token groups a TokenPolicy builds, each (id column, feature columns ...): the
id column's 0 is padding. legacy: state, hand, next, units, opp_seen, opp_last; v2 also spells,
pending and buttons, and hand gets hand_feats. state_features is the state token's input size,
6 towers + the scalars.

Pure Python, standard library only: sim/train_tokens.py, the model and the tests import it
without the engine, numpy or torch (check_row and check_shard use numpy when it is there).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import operator
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: vocab.json "schema" values, and the short names `--schema` takes.
SCHEMA_LEGACY = "rs_rows/legacy"
SCHEMA_V2 = "rs_rows/v2"
SCHEMAS = (SCHEMA_LEGACY, SCHEMA_V2)
SHORT_NAMES = {"legacy": SCHEMA_LEGACY, "v2": SCHEMA_V2}
CHOICES = tuple(SHORT_NAMES)

#: Layout numbers (tools/rs_rows.py and tools/rs_presses.py use the same values; a test checks).
MAX_UNITS = 48
HAND_SIZE = 4
OPP_SEEN, OPP_LAST = 8, 4
PENDING_SLOTS, PENDING_SCALE = 4, 40
SPELL_SLOTS, SPELL_FLOWN_SCALE = 8, 40
GRID_W, GRID_H = 18, 32
#: The symbolic dimension of the ability buttons, and B for an engine that does not say.
BUTTONS = "B"
FALLBACK_BUTTONS = 2

#: Feature names along the last axis.
UNIT_FEATS_LEGACY = ("own", "x", "y", "hp", "flying", "building", "deploying")
#: --live-compat rows: the seventh unit feature is log1p(max_hp) / 10, not the deploy flag.
LIVE_UNIT_FEATS = ("own", "x", "y", "hp", "flying", "building", "log_max_hp")
UNIT_FEATS_V2 = UNIT_FEATS_LEGACY + ("form", "burrow")
LIVE_UNIT_FEATS_V2 = LIVE_UNIT_FEATS + ("form", "burrow")
TOWER_FEATS = ("own_left", "own_right", "own_king", "opp_left", "opp_right", "opp_king")
SCALARS_LEGACY = ("elixir", "time", "double", "overtime")
SCALARS_V2 = SCALARS_LEGACY + ("triple", "opp_elixir", "opp_exact")
HAND_FEATS = ("form", "evo_ready", "evo_progress")
SPELL_FEATS = ("own", "x", "y", "aim_x", "aim_y", "flown")
ABILITY_FEATS = ("reported", "available", "spent", "cost", "cooldown")

#: Column roles and sources (module doc).
ROLE_OBS, ROLE_INFO, ROLE_LABEL, ROLE_FILTER, ROLE_KEY = "obs", "info", "label", "filter", "key"
ROLES = (ROLE_OBS, ROLE_INFO, ROLE_LABEL, ROLE_FILTER, ROLE_KEY)
SOURCE_OBS, SOURCE_ROWS, SOURCE_BUILD = "obs_row", "rows_for", "build"
SOURCES = (SOURCE_OBS, SOURCE_ROWS, SOURCE_BUILD)
#: The flags that add a legacy extra column (tools/rs_rows.py), and --trust v2 under both.
FLAG_VISION, FLAG_PUBLIC, FLAG_TRUST = "--units vision", "--visibility public", "--trust v2"
FLAG_PRESSES, FLAG_DECISION = "--presses on", "--obs-delay D > 0"
DTYPES = ("bool", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "str")


class SchemaError(ValueError):
    """An unknown schema name, column or feature, or a spec that does not match its schema."""


@dataclasses.dataclass(frozen=True)
class Column:
    """One shard column (module doc, "A COLUMN")."""
    name: str
    shape: tuple
    dtype: str
    role: str
    source: str
    extra: str | None = None
    feats: tuple = ()
    mirror: Mapping[str, Any] | None = None
    compact: str | None = None
    doc: str = ""

    def __post_init__(self):
        if self.dtype not in DTYPES or (self.compact is not None and self.compact not in DTYPES):
            raise SchemaError(f"column {self.name}: dtype {self.dtype!r} / {self.compact!r}")
        if self.role not in ROLES or self.source not in SOURCES:
            raise SchemaError(f"column {self.name}: role {self.role!r}, source {self.source!r}")
        if self.feats and (not self.shape or self.shape[-1] != len(self.feats)):
            raise SchemaError(f"column {self.name}: {len(self.feats)} feature names for shape "
                              f"{self.shape}")

    def resolved(self, buttons: int | None = None) -> tuple:
        """The shape with BUTTONS replaced by `buttons` (None: FALLBACK_BUTTONS)."""
        b = FALLBACK_BUTTONS if buttons is None else _buttons(buttons)
        return tuple(b if d == BUTTONS else int(d) for d in self.shape)

    def to_json(self, buttons: int | None = None) -> dict:
        out = {"name": self.name, "shape": list(self.resolved(buttons)), "dtype": self.dtype,
               "role": self.role, "source": self.source}
        if self.extra is not None:
            out["extra"] = self.extra
        if self.feats:
            out["feats"] = list(self.feats)
        if self.mirror is not None:
            out["mirror"] = json.loads(json.dumps(dict(self.mirror)))
        if self.compact is not None and self.compact != self.dtype:
            out["compact"] = self.compact
        return out


def _buttons(value) -> int:
    """B as an int >= 0 (numpy integers too); SchemaError otherwise."""
    try:
        b = None if isinstance(value, bool) else operator.index(value)
    except TypeError:
        b = None
    if b is None or b < 0:
        raise SchemaError(f"buttons: an int >= 0, got {value!r}")
    return int(b)


def _c(name, shape, dtype, role, source, **kw) -> Column:
    return Column(name, tuple(shape), dtype, role, source, **kw)


_UNIT_FLIP = {"flip": [UNIT_FEATS_LEGACY.index("x")], "where": "unit_ids"}
_TOWER_SWAP = {"swap": [[0, 1], [3, 4]]}
_TILE_FLIP = {"tile": GRID_W - 1, "where": "act_card"}
_SPELL_FLIP = {"flip": [SPELL_FEATS.index("x"), SPELL_FEATS.index("aim_x")], "where": "spell_ids"}

_LEGACY_COLUMNS = (
    _c("tick", (), "int64", ROLE_INFO, SOURCE_OBS, compact="int32",
       doc="the tick the row observes (state(t)), 50 ms each"),
    _c("side", (), "int64", ROLE_INFO, SOURCE_OBS, compact="int8",
       doc="1: the acting player is blue (engine team 0), -1: red"),
    _c("hand_exact", (), "bool", ROLE_INFO, SOURCE_OBS,
       doc="whether the hand follows from the side's plays alone (True before E11)"),
    _c("unit_ids", (MAX_UNITS,), "int16", ROLE_OBS, SOURCE_OBS,
       doc="unit tokens nearest the own king first: unit_type + 2, the vision class + 2 or "
           "(--live-compat) the creating card + 2; 0 pad"),
    _c("unit_feats", (MAX_UNITS, len(UNIT_FEATS_LEGACY)), "float16", ROLE_OBS, SOURCE_OBS,
       feats=UNIT_FEATS_LEGACY, mirror=_UNIT_FLIP,
       doc="own-frame x / 18000 and y / 32000; --live-compat: log_max_hp in place of deploying"),
    _c("towers", (len(TOWER_FEATS),), "float32", ROLE_OBS, SOURCE_OBS, feats=TOWER_FEATS,
       mirror=_TOWER_SWAP, doc="hp / max hp, own then the opponent's, each left, right, king"),
    _c("scalars", (len(SCALARS_LEGACY),), "float32", ROLE_OBS, SOURCE_OBS, feats=SCALARS_LEGACY,
       doc="own elixir / 10000, min(1, ms / 300000), 2x from 120 s, overtime from 180 s"),
    _c("hand", (HAND_SIZE,), "int16", ROLE_OBS, SOURCE_OBS, doc="card tokens, id + 2; 0 empty"),
    _c("next", (), "int64", ROLE_OBS, SOURCE_OBS, compact="int16", doc="next card token; 0 none"),
    _c("opp_seen", (OPP_SEEN,), "int16", ROLE_OBS, SOURCE_OBS,
       doc="the opponent's cards seen so far, sorted"),
    _c("opp_last", (OPP_LAST,), "int16", ROLE_OBS, SOURCE_OBS,
       doc="the opponent's last plays, oldest first"),
    _c("opp_last_age", (OPP_LAST,), "float16", ROLE_OBS, SOURCE_OBS,
       doc="their ages / 400 ticks, clipped at 1; 1 for padding"),
    _c("outcome", (), "float64", ROLE_LABEL, SOURCE_ROWS, compact="float32",
       doc="tower swing over the next HORIZON ticks"),
    _c("trusted", (), "bool", ROLE_FILTER, SOURCE_ROWS),
    _c("won", (), "float64", ROLE_LABEL, SOURCE_ROWS, compact="float32"),
    _c("hog", (), "bool", ROLE_FILTER, SOURCE_ROWS, doc="the side plays Hog 2.6"),
    _c("unit_form", (MAX_UNITS,), "float16", ROLE_OBS, SOURCE_OBS, extra=FLAG_VISION,
       doc="0 base, 0.5 evolution, 1 hero (vision_classes.UnitLookup.form_feature)"),
    _c("unit_burrow", (MAX_UNITS,), "float16", ROLE_OBS, SOURCE_OBS, extra=FLAG_PUBLIC,
       doc="1 for a unit tunnelling now (STATUS_UNDERGROUND)"),
    _c("trusted_v2", (), "bool", ROLE_FILTER, SOURCE_ROWS, extra=FLAG_TRUST),
    _c("ability_card", (BUTTONS,), "int16", ROLE_OBS, SOURCE_OBS, extra=FLAG_PRESSES,
       doc="each button's card + 2; 1 unnamed; 0 no button"),
    _c("ability_state", (BUTTONS, len(ABILITY_FEATS)), "float16", ROLE_OBS, SOURCE_OBS,
       extra=FLAG_PRESSES, feats=ABILITY_FEATS,
       doc="reported, available, spent, cost / 10, cooldown / 400"),
    _c("own_pending", (PENDING_SLOTS,), "int16", ROLE_OBS, SOURCE_OBS, extra=FLAG_DECISION,
       doc="own plays tapped and not run yet, soonest first: card tokens"),
    _c("own_pending_land", (PENDING_SLOTS,), "float16", ROLE_OBS, SOURCE_OBS,
       extra=FLAG_DECISION, doc="their ticks to land / 40, clipped at 1"),
    _c("act_card", (), "int16", ROLE_LABEL, SOURCE_BUILD, doc="the played card's token; 0 wait"),
    _c("act_x", (), "int16", ROLE_LABEL, SOURCE_BUILD, mirror=_TILE_FLIP,
       doc="own-frame tile column 0..17"),
    _c("act_y", (), "int16", ROLE_LABEL, SOURCE_BUILD, doc="own-frame tile row 0..31"),
    _c("act_press", (), "int16", ROLE_LABEL, SOURCE_BUILD, extra=FLAG_PRESSES,
       doc="0 none, k + 1 button k"),
    _c("match", (), "str", ROLE_KEY, SOURCE_BUILD, doc="the replay tag"),
    _c("split", (), "str", ROLE_KEY, SOURCE_BUILD, doc="train / val / test by tag hash"),
)

_V2_COLUMNS = (
    _c("tick", (), "int32", ROLE_INFO, SOURCE_OBS, doc="the tick the row observes (state(t))"),
    _c("side", (), "int8", ROLE_INFO, SOURCE_OBS, doc="1 blue (engine team 0), -1 red"),
    _c("hand_exact", (), "bool", ROLE_INFO, SOURCE_OBS,
       doc="honest (rs_rows.exact_hand) for every row"),
    _c("unit_ids", (MAX_UNITS,), "int16", ROLE_OBS, SOURCE_OBS,
       doc="as legacy: the unit token of the --units mode or --live-compat"),
    _c("unit_feats", (MAX_UNITS, len(UNIT_FEATS_V2)), "float16", ROLE_OBS, SOURCE_OBS,
       feats=UNIT_FEATS_V2, mirror=_UNIT_FLIP,
       doc="the legacy 7, then form (0 / 0.5 / 1) and burrow (STATUS_UNDERGROUND)"),
    _c("towers", (len(TOWER_FEATS),), "float32", ROLE_OBS, SOURCE_OBS, feats=TOWER_FEATS,
       mirror=_TOWER_SWAP),
    _c("scalars", (len(SCALARS_V2),), "float32", ROLE_OBS, SOURCE_OBS, feats=SCALARS_V2,
       doc="the legacy 4, 3x from 240 s, the opponent's counted elixir / 10000, its exactness"),
    _c("hand", (HAND_SIZE,), "int16", ROLE_OBS, SOURCE_OBS),
    _c("hand_feats", (HAND_SIZE, len(HAND_FEATS)), "float16", ROLE_OBS, SOURCE_OBS,
       feats=HAND_FEATS, doc="deck form / 2, next play evolved, evolution charge"),
    _c("next", (), "int16", ROLE_OBS, SOURCE_OBS),
    _c("opp_seen", (OPP_SEEN,), "int16", ROLE_OBS, SOURCE_OBS),
    _c("opp_last", (OPP_LAST,), "int16", ROLE_OBS, SOURCE_OBS),
    _c("opp_last_age", (OPP_LAST,), "float16", ROLE_OBS, SOURCE_OBS),
    _c("spell_ids", (SPELL_SLOTS,), "int16", ROLE_OBS, SOURCE_OBS,
       doc="cast spells in flight (SPELL cards), nearest the own king first: the card + 2"),
    _c("spell_feats", (SPELL_SLOTS, len(SPELL_FEATS)), "float16", ROLE_OBS, SOURCE_OBS,
       feats=SPELL_FEATS, mirror=_SPELL_FLIP,
       doc="own, centre and aim in the own frame (/ 18000, / 32000), ticks_flown / 40"),
    _c("own_pending", (PENDING_SLOTS,), "int16", ROLE_OBS, SOURCE_OBS),
    _c("own_pending_land", (PENDING_SLOTS,), "float16", ROLE_OBS, SOURCE_OBS),
    _c("ability_card", (BUTTONS,), "int16", ROLE_OBS, SOURCE_OBS),
    _c("ability_state", (BUTTONS, len(ABILITY_FEATS)), "float16", ROLE_OBS, SOURCE_OBS,
       feats=ABILITY_FEATS),
    _c("outcome", (), "float32", ROLE_LABEL, SOURCE_ROWS),
    _c("trusted", (), "bool", ROLE_FILTER, SOURCE_ROWS),
    _c("won", (), "float32", ROLE_LABEL, SOURCE_ROWS),
    _c("hog", (), "bool", ROLE_FILTER, SOURCE_ROWS),
    _c("trusted_v2", (), "bool", ROLE_FILTER, SOURCE_ROWS, extra=FLAG_TRUST),
    _c("act_card", (), "int16", ROLE_LABEL, SOURCE_BUILD),
    _c("act_x", (), "int16", ROLE_LABEL, SOURCE_BUILD, mirror=_TILE_FLIP),
    _c("act_y", (), "int16", ROLE_LABEL, SOURCE_BUILD),
    _c("act_press", (), "int16", ROLE_LABEL, SOURCE_BUILD, doc="0 none, k + 1 button k"),
    _c("match", (), "str", ROLE_KEY, SOURCE_BUILD),
    _c("split", (), "str", ROLE_KEY, SOURCE_BUILD),
)

_LEGACY_GROUPS = {"state": ("towers", "scalars"), "hand": ("hand",), "next": ("next",),
                  "units": ("unit_ids", "unit_feats"), "opp_seen": ("opp_seen",),
                  "opp_last": ("opp_last", "opp_last_age")}
_V2_GROUPS = {**_LEGACY_GROUPS, "hand": ("hand", "hand_feats"),
              "spells": ("spell_ids", "spell_feats"),
              "pending": ("own_pending", "own_pending_land"),
              "buttons": ("ability_card", "ability_state")}


class Schema:
    """One row schema: its columns in shard order (module doc)."""

    def __init__(self, name: str, columns: Sequence[Column], groups: Mapping[str, tuple],
                 live_feats: tuple):
        self.name = name
        self.columns = tuple(columns)
        self._by = {c.name: c for c in self.columns}
        if len(self._by) != len(self.columns):
            raise SchemaError(f"{name}: a column name is used twice")
        self.groups = {k: tuple(v) for k, v in groups.items()}
        self._live_feats = tuple(live_feats)

    # ------------------------------------------------------------ names
    @property
    def short(self) -> str:
        return next(k for k, v in SHORT_NAMES.items() if v == self.name)

    @property
    def v2(self) -> bool:
        return self.name == SCHEMA_V2

    def __repr__(self) -> str:
        return f"Schema({self.name!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Schema) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def column(self, name: str) -> Column:
        try:
            return self._by[name]
        except KeyError:
            raise SchemaError(f"{self.name} has no column {name!r}") from None

    def __contains__(self, name) -> bool:
        return name in self._by

    @property
    def extras(self) -> tuple:
        """The columns only a flag adds, in shard order."""
        return tuple(c.name for c in self.columns if c.extra is not None)

    def _pick(self, extra: Iterable[str], parts: tuple) -> tuple:
        extra = {extra} if isinstance(extra, str) else set(extra or ())
        unknown = sorted(extra - set(self.extras))
        if unknown:
            raise SchemaError(f"{self.name}: {unknown} are not extra columns of this schema "
                              f"(extras: {list(self.extras)})")
        return tuple(c.name for c in self.columns
                     if c.source in parts and (c.extra is None or c.name in extra))

    def shard_columns(self, extra: Iterable[str] = ()) -> tuple:
        """Every column of a shard whose vocab.json "extra_columns" is `extra`, in order."""
        return self._pick(extra, (SOURCE_OBS, SOURCE_ROWS)) + self._pick(extra, (SOURCE_BUILD,))

    def row_fields(self, extra: Iterable[str] = ()) -> tuple:
        """The row dict keys build stacks (rs_rows.FIELDS for legacy, extras appended)."""
        return self._pick(extra, (SOURCE_OBS, SOURCE_ROWS))

    def obs_fields(self, extra: Iterable[str] = ()) -> tuple:
        """The keys obs_row returns, in its order."""
        return self._pick(extra, (SOURCE_OBS,))

    @property
    def model_inputs(self) -> tuple:
        """The columns the model of this schema reads (its observation)."""
        return tuple(c.name for c in self.columns if c.role == ROLE_OBS and c.extra is None)

    @property
    def labels(self) -> tuple:
        """The training targets (outcome, won, act_card, act_x, act_y; v2: act_press)."""
        return tuple(c.name for c in self.columns if c.role == ROLE_LABEL and c.extra is None)

    @property
    def train_columns(self) -> tuple:
        """What sim/train_tokens.py loads: model_inputs + labels (legacy: its COLS)."""
        return self.model_inputs + self.labels

    # ------------------------------------------------------------ shapes
    def shape(self, name: str, buttons: int | None = None) -> tuple:
        """The per-row shape of column `name`, BUTTONS resolved (None: FALLBACK_BUTTONS)."""
        return self.column(name).resolved(buttons)

    def dtype(self, name: str) -> str:
        return self.column(name).dtype

    def feats(self, name: str, live: bool = False) -> tuple:
        """The feature names of `name`'s last axis; unit_feats of --live-compat rows have
        log_max_hp in place of deploying."""
        if name == "unit_feats" and live:
            return self._live_feats
        return self.column(name).feats

    def index(self, name: str, feat: str, live: bool = False) -> int:
        """Where feature `feat` sits along column `name`'s last axis."""
        names = self.feats(name, live)
        if feat not in names:
            raise SchemaError(f"{self.name} {name} has no feature {feat!r} ({list(names)})")
        return names.index(feat)

    @property
    def unit_features(self) -> int:
        """UF: features per unit token (legacy 7, v2 9)."""
        return self.column("unit_feats").shape[-1]

    @property
    def n_scalars(self) -> int:
        return self.column("scalars").shape[-1]

    @property
    def state_features(self) -> int:
        """The state token's input size: towers + scalars (legacy 10, v2 13)."""
        return self.column("towers").shape[-1] + self.n_scalars

    def mirror(self) -> dict:
        """{column: its mirror rule} (module doc, MIRROR) for the columns a flip changes."""
        return {c.name: json.loads(json.dumps(dict(c.mirror)))
                for c in self.columns if c.mirror is not None}

    def row_bytes(self, buttons: int | None = None, columns: Iterable[str] | None = None) -> int:
        """Bytes per row of `columns` (default train_columns) at their dtypes; str columns count
        0 (E18's GPU budget)."""
        size = {"bool": 1, "int8": 1, "int16": 2, "int32": 4, "int64": 8, "float16": 2,
                "float32": 4, "float64": 8, "str": 0}
        total = 0
        for name in (self.train_columns if columns is None else columns):
            c = self.column(name)
            total += size[c.dtype] * math.prod(c.resolved(buttons))
        return total

    # ------------------------------------------------------------ JSON
    def to_json(self, buttons: int | None = None) -> dict:
        """vocab.json "schema_spec" of rows of this schema: every column (shapes with B
        resolved), the token groups, the live unit feature names and the layout numbers."""
        b = FALLBACK_BUTTONS if buttons is None else _buttons(buttons)
        return {"name": self.name, "buttons": b,
                "columns": [c.to_json(b) for c in self.columns],
                "groups": {k: list(v) for k, v in self.groups.items()},
                "live_unit_feats": list(self._live_feats),
                "layout": {"max_units": MAX_UNITS, "hand": HAND_SIZE, "opp_seen": OPP_SEEN,
                           "opp_last": OPP_LAST, "pending_slots": PENDING_SLOTS,
                           "pending_scale": PENDING_SCALE, "spell_slots": SPELL_SLOTS,
                           "spell_flown_scale": SPELL_FLOWN_SCALE, "grid": [GRID_W, GRID_H]}}

    def digest(self, buttons: int | None = None) -> str:
        """sha1 of to_json(buttons), compact and key-sorted: 16 hex digits."""
        text = json.dumps(self.to_json(buttons), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------ checks
    def check_row(self, row: Mapping[str, Any], buttons: int | None = None,
                  extra: Iterable[str] = (), fields: Iterable[str] | None = None) -> list:
        """Problems of one row dict against the spec: a missing or unexpected key (of
        obs_fields(extra), or of `fields` when given), and a value whose shape or dtype is not
        the column's. Python ints, floats and bools stand for scalars of an int, float or bool
        column (obs_row returns tick, side, next and hand_exact that way; build casts). []
        when it fits."""
        np = _numpy()
        want = tuple(fields) if fields is not None else self.obs_fields(extra)
        problems = []
        if fields is None:
            unexpected = sorted(set(row) - set(want))
            if unexpected:
                problems.append(f"unexpected keys {unexpected}")
        for name in want:
            if name not in row:
                problems.append(f"{name}: missing")
                continue
            col = self.column(name)
            value = row[name]
            shape = col.resolved(buttons)
            if not shape and not isinstance(value, np.ndarray):
                if not _scalar_fits(value, col.dtype, np):
                    problems.append(f"{name}: {type(value).__name__} {value!r} is not a "
                                    f"{col.dtype} scalar")
                continue
            arr = np.asarray(value)
            if arr.shape != shape:
                problems.append(f"{name}: shape {arr.shape}, the spec says {shape}")
            if col.dtype != "str" and arr.dtype != np.dtype(col.dtype):
                problems.append(f"{name}: dtype {arr.dtype}, the spec says {col.dtype}")
        return problems

    def check_shard(self, arrays: Mapping[str, Any], buttons: int | None = None,
                    extra: Iterable[str] = ()) -> list:
        """Problems of a loaded shard (np.load's NpzFile or a dict of arrays): its column
        names and order, and each column's dtype and per-row shape. [] when it fits."""
        np = _numpy()
        names = list(getattr(arrays, "files", None) or list(arrays))
        want = list(self.shard_columns(extra))
        problems = [] if names == want else [f"columns {names}, the spec says {want}"]
        n = None
        for name in want:
            if name not in names:
                continue
            arr = np.asarray(arrays[name])
            col = self.column(name)
            if arr.shape[1:] != col.resolved(buttons):
                problems.append(f"{name}: rows of shape {arr.shape[1:]}, the spec says "
                                f"{col.resolved(buttons)}")
            if col.dtype == "str":
                if arr.dtype.kind != "U":
                    problems.append(f"{name}: dtype {arr.dtype}, the spec says str")
            elif arr.dtype != np.dtype(col.dtype):
                problems.append(f"{name}: dtype {arr.dtype}, the spec says {col.dtype}")
            if n is None:
                n = len(arr)
            elif len(arr) != n:
                problems.append(f"{name}: {len(arr)} rows, the first column has {n}")
        return problems

    def describe(self, buttons: int | None = None) -> str:
        """The layout, one column a line, for the console."""
        lines = [f"{self.name}: {len(self.columns)} columns ({len(self.extras)} added by a "
                 f"flag), state token {self.state_features}, unit features {self.unit_features},"
                 f" {self.row_bytes(buttons):,} bytes a row of train columns"]
        for c in self.columns:
            shape = "x".join(str(d) for d in c.resolved(buttons)) or "-"
            extra = f"  [{c.extra}]" if c.extra else ""
            feats = f"  ({', '.join(c.feats)})" if c.feats else ""
            lines.append(f"  {c.name:18s} {shape:>6s} {c.dtype:8s} {c.role:6s}{extra}{feats}")
        return "\n".join(lines)


def _numpy():
    import numpy
    return numpy


def _scalar_fits(value, dtype: str, np) -> bool:
    if dtype == "bool":
        return isinstance(value, (bool, np.bool_))
    if isinstance(value, (bool, np.bool_)):
        return False
    if dtype.startswith("int"):
        return isinstance(value, (int, np.integer))
    if dtype.startswith("float"):
        return isinstance(value, (int, float, np.integer, np.floating))
    return isinstance(value, str)


LEGACY = Schema(SCHEMA_LEGACY, _LEGACY_COLUMNS, _LEGACY_GROUPS, LIVE_UNIT_FEATS)
V2 = Schema(SCHEMA_V2, _V2_COLUMNS, _V2_GROUPS, LIVE_UNIT_FEATS_V2)
_ALL = {SCHEMA_LEGACY: LEGACY, SCHEMA_V2: V2}


def schema_name(value=None) -> str:
    """The vocab.json name of `value`: None and "legacy" give SCHEMA_LEGACY, "v2" gives
    SCHEMA_V2, a full name or a Schema itself; SchemaError (a ValueError) otherwise."""
    if value is None:
        return SCHEMA_LEGACY
    if isinstance(value, Schema):
        return value.name
    if isinstance(value, str):
        text = value.strip()
        if text in _ALL:
            return text
        if text in SHORT_NAMES:
            return SHORT_NAMES[text]
    raise SchemaError(f"row schema: one of {', '.join(CHOICES)} (vocab.json "
                      f"{', '.join(SCHEMAS)}), got {value!r}")


def get(value=None) -> Schema:
    """The Schema named `value` (schema_name)."""
    return _ALL[schema_name(value)]


def of_meta(meta: Mapping[str, Any] | None) -> Schema:
    """The schema of a vocab.json or a checkpoint meta: its "schema", legacy when absent (every
    checkpoint before E04 has none). SchemaError for a name this code does not know."""
    return get((meta or {}).get("schema") or SCHEMA_LEGACY)


def buttons_of(meta: Mapping[str, Any] | None) -> int:
    """B of a vocab.json or meta: "schema_spec" buttons, else "presses" buttons (E10), else
    FALLBACK_BUTTONS."""
    meta = meta or {}
    for key in ("schema_spec", "presses"):
        entry = meta.get(key)
        if isinstance(entry, Mapping) and entry.get("buttons") is not None:
            return _buttons(entry["buttons"])
    return FALLBACK_BUTTONS


def spec_problems(meta: Mapping[str, Any] | None) -> list:
    """Differences between a meta's "schema_spec" and what this code's schema of that name says
    (to_json at the spec's own B): [] when they agree or the meta has no spec (legacy rows
    write none). A row layout changed under an old name shows up here, not as garbage."""
    meta = meta or {}
    spec = meta.get("schema_spec")
    if spec is None:
        return []
    schema = of_meta(meta)
    if not isinstance(spec, Mapping):
        return [f"schema_spec is a {type(spec).__name__}, not a dict"]
    if spec.get("name") != schema.name:
        return [f"schema_spec names {spec.get('name')!r}, the meta's schema is {schema.name!r}"]
    want = schema.to_json(buttons_of(meta))
    problems = []
    for key in sorted(set(want) | set(spec)):
        if key not in spec:
            problems.append(f"schema_spec has no {key!r}")
        elif key not in want:
            problems.append(f"schema_spec has {key!r}, which this code does not write")
        elif spec[key] != want[key]:
            if key == "columns":
                have = {c.get("name"): c for c in spec[key] if isinstance(c, Mapping)}
                for c in want[key]:
                    if have.get(c["name"]) != c:
                        problems.append(f"column {c['name']}: {have.get(c['name'])} in the meta, "
                                        f"{c} here")
                extra = sorted(set(have) - {c["name"] for c in want[key]})
                if extra:
                    problems.append(f"columns {extra} are in the meta only")
                if not problems:
                    problems.append("columns are in another order")
            else:
                problems.append(f"{key}: {spec[key]!r} in the meta, {want[key]!r} here")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print a row schema's columns.")
    ap.add_argument("--schema", choices=CHOICES, default="v2")
    ap.add_argument("--buttons", type=int, default=FALLBACK_BUTTONS,
                    help=f"ability buttons per side, B (default {FALLBACK_BUTTONS}; the engine's "
                         "royalesim.ABILITY_BUTTONS for real rows)")
    ap.add_argument("--json", action="store_true", help="print vocab.json's schema_spec")
    args = ap.parse_args(argv)
    schema = get(args.schema)
    if args.json:
        print(json.dumps(schema.to_json(args.buttons), indent=1))
    else:
        print(schema.describe(args.buttons))
        print(f"digest {schema.digest(args.buttons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
