r"""A scaled perception-noise profile: every sigma, spread, rate and probability times k (S11).

    .venvs\royalegym\Scripts\python.exe tools\noise_scale.py           default_prior x 2 -> data\noise\stress_2x.json
    .venvs\royalegym\Scripts\python.exe tools\noise_scale.py --check   is the committed stress_2x.json this tool's? (1: no)
    .venvs\royalegym\Scripts\python.exe tools\noise_scale.py --source data\noise\measured.json --factor 2
                                                                       -> data\noise\measured_2x.json

    from tools import noise_scale as NS               # or `import noise_scale` (tools/ on sys.path)
    data = NS.scaled(json.loads(text), 2.0)           # plain JSON in, plain JSON out
    NS.validate(data)                                 # tools/perception_noise.NoiseProfile.from_dict

Why (docs/LIVE_SEARCH_PLAN.md 7.2 M6, 9 S11). The search must not win only because the simulator's
perception noise is mild: M6 re-runs the champion under doubled noise and asks for no more than
1 pp lost. data/noise/stress_2x.json is that profile, made from data/noise/default_prior.json (the
profile `--noise default` loads, tools/perception_noise.PRIOR_PATH) by this tool. The file is
committed; --check (and tests/pipeline/test_noise_scale.py) confirms it is what the tool makes
from the prior as it is now, so a change to the prior shows up until the file is regenerated.

THE RULES (RULES: every field of every tools/perception_noise group, by its JSON name; a field
or group the table does not know is an error, so a new noise field is classified on purpose):
  p        a probability: drop rates, form collapse, confusion.unknown_p, side flips, missing HP
           bars, elixir misreads, missed and unnamed opponent plays. x k, capped at 1 (None, as
           opponent_plays.spell_miss_p may be, stays None).
  sigma    a gaussian sigma: positions and the flyer spread, HP bars, the phantom distance,
           tower HP, the elixir fraction, the clock. x k.
  spread   a uniform +- spread of whole ticks (board.jitter_ticks, new_units.jitter_ticks,
           opponent_plays.late_jitter_ticks): the sigma of the tick-valued noises. x k, rounded
           half up to whole ticks.
  rate     false_positives.rate, the phantoms' Poisson mean per observation: x k, NOT capped (a
           mean, not a probability).
  pmap     drop.by_class, class -> p: each p x k, capped at 1.
  pmatrix  confusion.pairs, class -> {class: p}: each p x k, capped at 1; a row whose confusions
           then add up to more than 1 is scaled to add up to 1 (perception_noise refuses more).
  keep     unchanged: the delays (board.lag_ticks, new_units.delay_ticks, deploy.age_offset_ticks,
           opponent_plays.late_ticks: the latency the gate already charges, D = 24), the biases
           (position.flyer_dx_mt / flyer_dy_mt, towers.bias_frac, clock.bias_ticks: a doubled bias
           is another miscalibration, not more noise), the reader's resolutions and thresholds
           (hp.quant, hp.full_snap, towers.quant_hp, clock.quant_ticks) and the settings
           (board.max_hold_ticks, size, elixir.fraction_fill_milli, forms.unlabelled_to_base, the
           phantom class weights).
A number keeps its JSON type where it can: an int times 2 is an int (250 -> 500), a float a float.

THE OUTPUT. schema; name (default: the output file's stem, "stress_2x"); a description; "_derived"
(the tool, the source path relative to the repo, its name, the digest of its parsed JSON, factor,
cap); then the source's groups in its order, each with the source's fields only (an absent field
keeps perception_noise's default, which is 0 for every noise). The source's notes ("_why",
"_sources", any key starting with "_") are left out: they argue for the prior's own numbers, not
these. JSON with indent 2 and a final newline, written as bytes (LF on every machine). --check and
the test compare parsed JSON, so a checkout with CRLF line endings still matches.

Standard library only; validate() imports tools/perception_noise (numpy) when it runs. Never reads
a battle, taps or touches the emulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:          # also when imported as tools.noise_scale
    sys.path.insert(0, str(ROOT / "tools"))

__all__ = ["CAP", "FACTOR", "NoiseScaleError", "RULES", "SOURCE", "build", "default_out",
           "main", "render", "scaled", "source_digest", "validate"]

TOOL = "tools/noise_scale.py"
#: The default source: the profile `--noise default` loads (perception_noise.PRIOR_PATH).
SOURCE = ROOT / "data" / "noise" / "default_prior.json"
FACTOR = 2.0
#: Probabilities are capped here.
CAP = 1.0
#: Profile keys that are not groups.
META_KEYS = ("schema", "name", "description")

P, SIGMA, SPREAD, RATE, PMAP, PMATRIX, KEEP = ("p", "sigma", "spread", "rate", "pmap", "pmatrix",
                                               "keep")
RULE_KINDS = (P, SIGMA, SPREAD, RATE, PMAP, PMATRIX, KEEP)

#: group -> field -> rule (module doc, THE RULES), in tools/perception_noise's group order.
RULES: dict[str, dict[str, str]] = {
    "board": {"lag_ticks": KEEP, "jitter_ticks": SPREAD, "max_hold_ticks": KEEP},
    "new_units": {"delay_ticks": KEEP, "jitter_ticks": SPREAD},
    "drop": {"small": P, "medium": P, "large": P, "building": P, "by_class": PMAP},
    "size": {"small_max_hp": KEEP, "large_min_hp": KEEP},
    "forms": {"evo_to_base_p": P, "hero_to_base_p": P, "unlabelled_to_base": KEEP},
    "confusion": {"unknown_p": P, "pairs": PMATRIX},
    "side_flip": {"p": P},
    "position": {"sigma_x_mt": SIGMA, "sigma_y_mt": SIGMA, "flyer_dx_mt": KEEP,
                 "flyer_dy_mt": KEEP, "flyer_sigma_mt": SIGMA},
    "hp": {"no_bar_p": P, "sigma": SIGMA, "quant": KEEP, "full_snap": KEEP},
    "deploy": {"age_offset_ticks": KEEP},
    "false_positives": {"rate": RATE, "classes": KEEP, "near_sigma_mt": SIGMA},
    "towers": {"sigma_frac": SIGMA, "bias_frac": KEEP, "quant_hp": KEEP},
    "elixir": {"fraction_sigma_milli": SIGMA, "fraction_lost_p": P, "fraction_fill_milli": KEEP,
               "digit_error_p": P},
    "clock": {"bias_ticks": KEEP, "sigma_ticks": SIGMA, "quant_ticks": KEEP},
    "opponent_plays": {"miss_p": P, "spell_miss_p": P, "late_ticks": KEEP,
                       "late_jitter_ticks": SPREAD, "unknown_p": P},
}


class NoiseScaleError(ValueError):
    """A profile, field or factor this tool cannot scale."""


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------

def _num(value: Any, where: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise NoiseScaleError(f"{where}: a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise NoiseScaleError(f"{where}: a finite number, got {value!r}")
    return value


def _times(value: Any, k: float, where: str) -> int | float:
    """value * k: an int stays an int when the product is whole, else a float."""
    v = _num(value, where)
    out = float(v) * k
    if isinstance(v, numbers.Integral) and out.is_integer():
        return int(out)
    return out


def _p(value: Any, k: float, where: str) -> int | float | None:
    """A probability x k, capped at CAP; None stays None."""
    if value is None:
        return None
    out = _times(value, k, where)
    if out > CAP:
        out = int(CAP) if isinstance(out, int) else float(CAP)
    return out


def _spread(value: Any, k: float, where: str) -> int:
    """Whole ticks x k, rounded half up."""
    v = _num(value, where)
    return int(math.floor(float(v) * k + 0.5))


def _strip(value: Any) -> Any:
    """`value` with every dict key starting with "_" left out, at any depth."""
    if isinstance(value, Mapping):
        return {str(key): _strip(v) for key, v in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _pmap(value: Any, k: float, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise NoiseScaleError(f"{where}: an object of name -> p, got {value!r}")
    return {str(name): _p(_num(p, f"{where}.{name}"), k, f"{where}.{name}")
            for name, p in value.items() if not str(name).startswith("_")}


def _pmatrix(value: Any, k: float, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise NoiseScaleError(f"{where}: an object of class -> {{class: p}}, got {value!r}")
    out = {}
    for src, row in value.items():
        if str(src).startswith("_"):
            continue
        r = _pmap(row, k, f"{where}.{src}")
        total = sum(float(p) for p in r.values())
        if total > CAP:
            r = {name: float(p) / total * CAP for name, p in r.items()}
        out[str(src)] = r
    return out


def _apply(rule: str, value: Any, k: float, where: str) -> Any:
    if rule == KEEP:
        return _strip(value)
    if rule == P:
        return _p(value, k, where)
    if rule in (SIGMA, RATE):
        return _times(value, k, where)
    if rule == SPREAD:
        return _spread(value, k, where)
    if rule == PMAP:
        return _pmap(value, k, where)
    if rule == PMATRIX:
        return _pmatrix(value, k, where)
    raise NoiseScaleError(f"{where}: no rule {rule!r} (rules: {', '.join(RULE_KINDS)})")


def _factor(factor: Any) -> float:
    k = float(_num(factor, "factor"))
    if not k > 0:
        raise NoiseScaleError(f"factor: above 0, got {factor!r}")
    return k


def source_digest(data: Mapping[str, Any]) -> str:
    """16 hex digits of sha256 over the parsed JSON (sorted keys, compact): the same for a
    file with CRLF or LF line endings."""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def default_name(factor: Any = FACTOR) -> str:
    return f"stress_{_factor(factor):g}x"


# ---------------------------------------------------------------------------
# a profile
# ---------------------------------------------------------------------------

def scaled(data: Mapping[str, Any], factor: Any = FACTOR, *, name: str | None = None,
           source: str | None = None) -> dict:
    """The profile JSON object `data` with every sigma, spread, rate and probability times
    `factor` (module doc, THE RULES and THE OUTPUT). `data` is not changed."""
    if not isinstance(data, Mapping):
        raise NoiseScaleError(f"a noise profile is a JSON object, got {type(data).__name__}")
    k = _factor(factor)
    src_name = data.get("name")
    shown = src_name if isinstance(src_name, str) and src_name else "the source profile"
    out: dict[str, Any] = {
        "schema": data.get("schema", 1),
        "name": default_name(k) if name is None else str(name),
        "description": (f"The perception noise of {shown} with every sigma, spread, rate and "
                        f"probability times {k:g} (probabilities capped at {CAP:g}), made by "
                        f"{TOOL}: the stress profile of docs/LIVE_SEARCH_PLAN.md S11 (M6)."),
        "_derived": {"tool": TOOL, "source": source, "source_name": src_name,
                     "source_digest": source_digest(data), "factor": k, "cap": CAP},
    }
    for group, body in data.items():
        group = str(group)
        if group.startswith("_") or group in META_KEYS:
            continue
        rules = RULES.get(group)
        if rules is None:
            raise NoiseScaleError(f"unknown group {group!r} (groups: {', '.join(RULES)})")
        if body is None:
            out[group] = None
            continue
        if not isinstance(body, Mapping):
            raise NoiseScaleError(f"{group}: an object, got {body!r}")
        part = {}
        for key, value in body.items():
            key = str(key)
            if key.startswith("_"):
                continue
            if key not in rules:
                raise NoiseScaleError(f"{group}: unknown key {key!r} (known: "
                                      f"{', '.join(rules)})")
            part[key] = _apply(rules[key], value, k, f"{group}.{key}")
        out[group] = part
    return out


def render(data: Mapping[str, Any]) -> str:
    """The JSON text this tool writes: indent 2, a final newline."""
    return json.dumps(data, indent=2) + "\n"


def _shown(path: Path) -> str:
    """`path` relative to the repository (posix), else as given."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _read(path: Path) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise FileNotFoundError(f"noise profile {path} does not exist") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise NoiseScaleError(f"noise profile {path}: not JSON ({exc})") from None


def build(source: str | Path = SOURCE, factor: Any = FACTOR, *, name: str | None = None) -> dict:
    """scaled() of a profile file, its path recorded relative to the repository."""
    path = Path(source)
    return scaled(_read(path), factor, name=name, source=_shown(path))


def default_out(source: str | Path = SOURCE, factor: Any = FACTOR) -> Path:
    """data/noise/stress_<k>x.json for the default prior, else <source stem>_<k>x.json beside
    the source."""
    k = _factor(factor)
    path = Path(source)
    if path.resolve() == SOURCE.resolve():
        return SOURCE.parent / f"{default_name(k)}.json"
    return path.parent / f"{path.stem}_{k:g}x.json"


def validate(data: Mapping[str, Any]) -> Any:
    """The tools/perception_noise.NoiseProfile of `data` (ValueError when it does not load)."""
    import perception_noise as PN
    return PN.NoiseProfile.from_dict(data)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scale a perception-noise profile: every sigma, spread, rate and probability "
                    "times a factor, probabilities capped at 1 (docs/LIVE_SEARCH_PLAN.md S11).")
    ap.add_argument("--source", type=Path, default=SOURCE,
                    help="the profile to scale (default: data/noise/default_prior.json)")
    ap.add_argument("--factor", type=float, default=FACTOR, help="the factor (default: 2)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default: data/noise/stress_<factor>x.json for the "
                         "default prior, else <source>_<factor>x.json beside it)")
    ap.add_argument("--name", default=None,
                    help="the profile's name (default: the output file's stem)")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 when --out is not what this would write")
    args = ap.parse_args(argv)
    try:
        out = args.out if args.out is not None else default_out(args.source, args.factor)
        data = build(args.source, args.factor, name=args.name or out.stem)
        profile = validate(data)
    except (ValueError, FileNotFoundError) as exc:
        print(f"noise_scale: {exc}", file=sys.stderr)
        return 2
    if args.check:
        try:
            have = json.loads(Path(out).read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            print(f"noise_scale: {out} does not exist: run {TOOL} to write it", file=sys.stderr)
            return 1
        except json.JSONDecodeError as exc:
            print(f"noise_scale: {out}: not JSON ({exc})", file=sys.stderr)
            return 1
        if have != data:
            print(f"noise_scale: {out} is not what {TOOL} makes from {_shown(args.source)} x "
                  f"{args.factor:g}: run it again to regenerate the file", file=sys.stderr)
            return 1
        print(f"ok {_shown(out)} ({_shown(args.source)} x {args.factor:g})")
        return 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(render(data).encode("utf-8"))
    print(f"wrote {_shown(out)} ({_shown(args.source)} x {args.factor:g})")
    print(profile.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
