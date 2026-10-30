r"""IL_Replay card key -> RoyaleSim card name, the map every re-drive of a recording reads.

    .venvs\royalegym\Scripts\python.exe tools\rs_cards.py                checks the map and shows it
    .venvs\royalegym\Scripts\python.exe tools\rs_cards.py --catalogue    checks every name against the
                                                                         engine's catalogue
    .venvs\royalegym\Scripts\python.exe tools\rs_cards.py --coverage [--parts N]
                                                                         lists the IL_Replay deck keys
                                                                         the map lacks, and confirms
                                                                         the extras

IL_Replay names a card by its RoyaleAPI key ("hog-rider", "the-log", plus "-ev1" or "-hero" for
a form). RoyaleSim names it by the client's name ("HogRider", "Log"). tools/rs_rows.py and
tools/royalesim_redrive.py (and tools/rs_eval.py, through rs_rows._setup) used to import
royaleimitate only to read its replay_cards.json. The map now lives in the repo (E02, rsim-06):

  data/cards/replay_cards.json        royaleimitate 0.2.15's file, byte for byte: 119 keys,
                                      MIT, with the notice in data/cards/LICENSE-royaleimitate.txt.
                                      load() refuses it unless its SHA-256 is VENDORED_SHA256.
  data/cards/replay_cards_extra.json  keys that file lacks: void -> DarkMagic, rune-giant ->
                                      GiantBuffer, spirit-empress -> MergeMaiden, minion-giant ->
                                      MinionGiant. Matches with an unmapped card are skipped, so
                                      before this every match with or against Void, Rune Giant or
                                      Spirit Empress was dropped.
                                      Each extra is "verified": false until --coverage finds its
                                      key in the corpus.

load() merges the two into 123 keys. Either file is refused if it has a key twice, and an extra
may not give an already-mapped key a different name, so no key maps to two names.
HASTYCR_CARD_EXTRAS=0 (or --no-extras here) leaves the extras out. That gives exactly the
119-key map the tools read from royaleimitate 0.2.15 before, so a run can be compared with older
runs on the same matches (for example the royalesim 0.1.22 reference re-drive). The author's
own royaleimitate version is not known (plan open question). If his numbers do not reproduce,
check his match counts first.

The hash is taken with CRLF read as LF, so a Windows checkout that rewrote line ends still
passes (data/cards/.gitattributes asks git not to rewrite them).

Exit codes of the CLI: 0 ok; 1 a check failed (the map, a name missing from the catalogue,
royalesim or the corpus missing); 2 a bad setting.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / "data" / "cards"
VENDORED_PATH = CARDS_DIR / "replay_cards.json"
EXTRA_PATH = CARDS_DIR / "replay_cards_extra.json"
LICENSE_PATH = CARDS_DIR / "LICENSE-royaleimitate.txt"
CORPUS = ROOT / "data" / "il_replay"           # as tools/royalesim_redrive.py CORPUS

#: Where data/cards/replay_cards.json comes from, and its SHA-256 (sha256sum of the wheel's
#: royaleimitate/replay_cards.json; the wheel's RECORD gives the same digest).
VENDORED_SOURCE = "royaleimitate 0.2.15, royaleimitate/replay_cards.json"
VENDORED_SHA256 = "aeca8fdf8eebafadbe3b1f4dba6eebb6132715e935bfe1db63f5968670c039ee"
VENDORED_ENTRIES = 119

EXTRAS_ENV = "HASTYCR_CARD_EXTRAS"
_ON = frozenset({"", "1", "true", "yes", "on"})
_OFF = frozenset({"0", "false", "no", "off"})


class CardMapError(ValueError):
    """A card map file is missing, altered or malformed, or the files contradict each other."""


@dataclasses.dataclass(frozen=True)
class Extra:
    """One entry of replay_cards_extra.json."""
    key: str
    name: str
    verified: bool = False
    evidence: str = ""


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def file_sha256(path: str | os.PathLike) -> str:
    """SHA-256 (hex) of the file's bytes, with CRLF read as LF."""
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CardMapError(f"key {key!r} appears twice")
        out[key] = value
    return out


def _read_json(path: Path, what: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8-sig")     # a Windows editor may add a BOM
    except FileNotFoundError:
        raise CardMapError(f"{what} {path} is missing") from None
    try:
        return json.loads(text, object_pairs_hook=_unique_pairs)
    except json.JSONDecodeError as exc:
        raise CardMapError(f"{what} {path} is not valid JSON: {exc}") from None
    except CardMapError as exc:
        raise CardMapError(f"{what} {path}: {exc}") from None


def _check_entry(key: Any, name: Any, where: str) -> None:
    if not isinstance(key, str) or not key or key != key.strip():
        raise CardMapError(f"{where}: {key!r} is not a card key")
    if not isinstance(name, str) or not name or name != name.strip():
        raise CardMapError(f"{where}: {key!r} -> {name!r}, the name must be a non-empty string")


def read_vendored(path: str | os.PathLike = VENDORED_PATH,
                  sha256: str | None = VENDORED_SHA256) -> dict[str, str]:
    """The vendored key -> name map, after checking its SHA-256 (sha256=None skips the check)."""
    path = Path(path)
    if not path.is_file():
        raise CardMapError(f"vendored card map {path} is missing")
    got = file_sha256(path)
    if sha256 is not None and got != sha256:
        raise CardMapError(
            f"{path} has SHA-256 {got}, not {sha256} ({VENDORED_SOURCE}). The file is vendored "
            "and must stay byte for byte: restore it with `git checkout -- data/cards/"
            "replay_cards.json`. To take a newer royaleimitate's map, copy it in and update "
            "VENDORED_SHA256 in tools/rs_cards.py and data/cards/README.md.")
    data = _read_json(path, "vendored card map")
    if not isinstance(data, dict) or not data:
        raise CardMapError(f"{path}: expected a non-empty JSON object of card key -> name")
    for key, name in data.items():
        _check_entry(key, name, str(path))
    return dict(data)


def read_extras(path: str | os.PathLike = EXTRA_PATH) -> dict[str, Extra]:
    """replay_cards_extra.json as key -> Extra. The file is {"cards": {key: {"name": ...,
    "verified": bool, "evidence": str}}}, and any other top-level key (like "_comment") is
    ignored."""
    path = Path(path)
    data = _read_json(path, "card map extras")
    cards = data.get("cards") if isinstance(data, dict) else None
    if not isinstance(cards, dict):
        raise CardMapError(f'{path}: expected {{"cards": {{key: {{"name": ..., "verified": ...}}}}}}')
    out: dict[str, Extra] = {}
    for key, entry in cards.items():
        if not isinstance(entry, dict):
            raise CardMapError(f'{path}: {key!r} must be an object with a "name"')
        name = entry.get("name")
        _check_entry(key, name, str(path))
        verified = entry.get("verified", False)
        if not isinstance(verified, bool):
            raise CardMapError(f'{path}: {key!r} "verified" must be true or false')
        evidence = entry.get("evidence", "")
        if not isinstance(evidence, str):
            raise CardMapError(f'{path}: {key!r} "evidence" must be a string')
        out[key] = Extra(key=key, name=name, verified=verified, evidence=evidence)
    return out


def merge(vendored: Mapping[str, str], extras: Mapping[str, Extra | str]) -> dict[str, str]:
    """`vendored` plus `extras`. An extra that gives a mapped key another name is an error.
    Repeating the same name is allowed and changes nothing."""
    out = dict(vendored)
    for key, extra in extras.items():
        name = extra.name if isinstance(extra, Extra) else extra
        _check_entry(key, name, "extra")
        have = out.get(key)
        if have is not None and have != name:
            raise CardMapError(f"extra {key!r} -> {name!r}, but the vendored map has {key!r} -> "
                               f"{have!r}: a key maps to one name")
        out[key] = name
    return out


def extras_enabled(env: Mapping[str, str] | None = None) -> bool:
    """False when HASTYCR_CARD_EXTRAS is 0 / false / no / off. True when it is unset, empty,
    1 / true / yes / on."""
    env = os.environ if env is None else env
    value = str(env.get(EXTRAS_ENV, "")).strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    raise CardMapError(f"{EXTRAS_ENV}={value!r}: use 1 (merge the extras) or 0 (vendored map only)")


def load(*, extras: bool | None = None, env: Mapping[str, str] | None = None,
         vendored_path: str | os.PathLike = VENDORED_PATH,
         extra_path: str | os.PathLike = EXTRA_PATH,
         sha256: str | None = VENDORED_SHA256) -> dict[str, str]:
    """The IL_Replay key -> RoyaleSim name map: the hash-checked vendored file, merged with the
    extras. Leave extras=None to follow HASTYCR_CARD_EXTRAS (read from `env`, else the process
    environment, which worker processes inherit). Returns a new dict on every call."""
    vendored = read_vendored(vendored_path, sha256)
    if extras is None:
        extras = extras_enabled(env)
    if not extras:
        return vendored
    return merge(vendored, read_extras(extra_path))


def map_sha256(names: Mapping[str, str]) -> str:
    """SHA-256 of the map as canonical JSON (sorted keys). Two maps with the same pairs get the
    same digest, whatever their order."""
    text = json.dumps(dict(names), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def describe(names: Mapping[str, str], vendored: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What a map is, for vocab.json and the logs: its size, digest, the vendored file it was
    built on and the keys it adds to that file."""
    vendored = read_vendored() if vendored is None else vendored
    return {"entries": len(names), "sha256": map_sha256(names), "vendored": VENDORED_SOURCE,
            "vendored_sha256": VENDORED_SHA256,
            "extras": sorted(k for k in names if k not in vendored)}


def summary(names: Mapping[str, str], vendored: Mapping[str, str] | None = None) -> str:
    """One line for a tool's log."""
    d = describe(names, vendored)
    extras = ", ".join(d["extras"]) or f"none, {EXTRAS_ENV}=0"
    return f"card map {d['entries']} keys (extras: {extras}), sha256 {d['sha256'][:12]}"


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def missing_from_catalogue(names: Mapping[str, str], catalogue: Iterable[Any]) -> dict[str, str]:
    """The keys of `names` whose name is not a catalogue name. `catalogue` holds catalogue_json()
    rows (row[0] is the name) or plain names."""
    have = {row if isinstance(row, str) else row[0] for row in catalogue}
    return {key: name for key, name in names.items() if name not in have}


def _redrive_base_key() -> Callable[[str], tuple[str, int]]:
    """royalesim_redrive.base_key: the key -> (base key, form) rule that rs_rows and the
    re-drive use. It is imported when needed, because that module imports this one."""
    tools = str(Path(__file__).resolve().parent)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from royalesim_redrive import base_key
    return base_key


def _deck_keys(payload: Mapping[str, Any], key_of: Callable[[str], tuple[str, int]]
               ) -> list[set[str]] | None:
    battle = payload.get("battle") or {}
    decks = []
    for side in ("team", "opponent"):
        players = (battle.get(side) or {}).get("players") or []
        if len(players) != 1:
            return None
        decks.append({key_of(card["card_key"])[0] for card in players[0].get("deck") or []})
    return decks


def coverage(payloads: Iterable[Mapping[str, Any]], names: Mapping[str, str],
             watch: Iterable[str] = (),
             key_of: Callable[[str], tuple[str, int]] | None = None) -> dict[str, Any]:
    """How much of a corpus `names` covers, over IL_Replay payload_json dicts.

    matches  1v1 matches read (others are counted in not_1v1)
    mapped   matches whose 16 deck keys (form suffix removed) all have a name
    blocked  unmapped key -> matches it is in, most common first
    watched  for each key of `watch` (the extras): decks that hold it
    keys     distinct deck keys seen
    """
    key_of = _redrive_base_key() if key_of is None else key_of
    watch = sorted(set(watch))
    matches = mapped = not_1v1 = 0
    blocked: Counter[str] = Counter()
    watched: Counter[str] = Counter()
    seen: set[str] = set()
    for payload in payloads:
        decks = _deck_keys(payload, key_of)
        if decks is None:
            not_1v1 += 1
            continue
        matches += 1
        unmapped: set[str] = set()
        for deck in decks:
            seen |= deck
            unmapped |= {k for k in deck if k not in names}
            watched.update(k for k in deck if k in watch)
        if unmapped:
            blocked.update(unmapped)
        else:
            mapped += 1
    return {"matches": matches, "mapped": mapped, "not_1v1": not_1v1,
            "blocked": dict(blocked.most_common()),
            "watched": {k: watched.get(k, 0) for k in watch}, "keys": len(seen)}


def corpus_parts(corpus: str | os.PathLike = CORPUS, parts: int = 0) -> list[Path]:
    """The corpus's replays/part-*.parquet files, the first `parts` of them when parts > 0."""
    files = sorted((Path(corpus) / "replays").glob("part-*.parquet"))
    return files[:parts] if parts > 0 else files


def iter_payloads(files: Iterable[Path]) -> Iterable[dict[str, Any]]:
    """payload_json of every row of these replays parquet files (needs pyarrow)."""
    import pyarrow.parquet as pq
    for part in files:
        for row in pq.read_table(part, columns=["payload_json"]).to_pylist():
            yield json.loads(row["payload_json"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _rs_engine():
    tools = str(Path(__file__).resolve().parent)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import rs_engine
    return rs_engine


def _pct(a: int, b: int) -> str:
    return f"{100 * a / max(1, b):.1f}%"


def main(argv: list[str] | None = None) -> int:
    rs_engine = _rs_engine()
    ap = argparse.ArgumentParser(
        description="The IL_Replay -> RoyaleSim card map (data/cards/): check it, check its "
                    "names against the engine's catalogue, or measure its corpus coverage.")
    ap.add_argument("--no-extras", action="store_true",
                    help=f"the vendored map alone, as {EXTRAS_ENV}=0")
    ap.add_argument("--catalogue", action="store_true",
                    help="check every name against the catalogue of a Battle built under the "
                         "engine config (needs royalesim)")
    ap.add_argument("--coverage", action="store_true",
                    help="count the IL_Replay deck keys the map lacks, and the decks that hold "
                         "each extra (needs pyarrow)")
    ap.add_argument("--corpus", type=Path, default=CORPUS,
                    help="IL_Replay folder with replays/part-*.parquet (default data/il_replay)")
    ap.add_argument("--parts", type=int, default=0, help="read only the first N parts")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    rs_engine.add_engine_args(ap)
    args = ap.parse_args(argv)

    report: dict[str, Any] = {}
    status = 0
    try:
        vendored = read_vendored()
        extras = read_extras()
        use_extras = False if args.no_extras else extras_enabled()
        names = merge(vendored, extras) if use_extras else dict(vendored)
    except CardMapError as exc:
        print(f"card map: {exc}", file=sys.stderr)
        return 1
    report["map"] = describe(names, vendored)
    report["extras"] = {k: {"name": e.name, "verified": e.verified, "merged": k in names}
                        for k, e in sorted(extras.items())}
    lines = [f"vendored    {VENDORED_PATH.relative_to(ROOT).as_posix()}: {len(vendored)} keys, "
             f"sha256 ok ({VENDORED_SOURCE})",
             f"merged      {summary(names, vendored)}"]
    for key, e in sorted(extras.items()):
        lines.append(f"extra       {key} -> {e.name}  "
                     f"{'verified' if e.verified else 'unverified'}"
                     f"{'' if key in names else '  (left out)'}")

    if args.catalogue:
        try:
            cfg = rs_engine.config_from_args(args)
        except (ValueError, FileNotFoundError) as exc:
            print(f"engine config: {exc}", file=sys.stderr)
            return 2
        try:
            rs = rs_engine.royalesim_module()
            catalogue = rs_engine.catalogue(cfg, rs)
        except Exception as exc:            # ImportError, EngineError or the engine's refusal
            print(f"catalogue: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        missing = missing_from_catalogue(names, catalogue)
        report["catalogue"] = {"config": cfg.to_json(), "cards": len(catalogue),
                               "missing": missing}
        lines.append(f"catalogue   {cfg.describe()}: {len(catalogue)} cards, "
                     + ("every name is in it" if not missing else
                        f"{len(missing)} names are NOT in it: "
                        + ", ".join(f"{k} -> {v}" for k, v in sorted(missing.items()))))
        status = status or (1 if missing else 0)

    if args.coverage:
        files = corpus_parts(args.corpus, args.parts)
        if not files:
            print(f"coverage: no part-*.parquet in {Path(args.corpus) / 'replays'} "
                  "(run tools/il_replay_download.py)", file=sys.stderr)
            return 1
        try:
            cov = coverage(iter_payloads(files), names, watch=extras)
        except ImportError as exc:
            print(f"coverage: {exc} (pyarrow is in the engine venv)", file=sys.stderr)
            return 1
        report["coverage"] = {"parts": len(files), **cov}
        lines.append(f"coverage    {len(files)} parts, {cov['matches']:,} 1v1 matches "
                     f"({cov['not_1v1']:,} others), {cov['keys']} deck keys; all 16 cards "
                     f"mapped in {cov['mapped']:,} ({_pct(cov['mapped'], cov['matches'])})")
        for key, n in list(cov["blocked"].items())[:20]:
            lines.append(f"  unmapped  {key}: in {n:,} matches")
        for key, n in cov["watched"].items():
            e = extras[key]
            state = ("seen: the key is right; set \"verified\": true in "
                     "data/cards/replay_cards_extra.json" if n and not e.verified
                     else "seen" if n else "NOT seen in these parts")
            lines.append(f"  extra     {key}: in {n:,} decks, {state}")

    print(json.dumps(report, indent=2) if args.json else "\n".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(main())
