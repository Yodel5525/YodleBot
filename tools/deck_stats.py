r"""The opponents' deck field from IL_Replay (R03): which decks top players play, and how
often, per split; and a sampler that deals them.

    .venvs\royalegym\Scripts\python.exe tools\deck_stats.py [--parts N] [--top 500] [--workers 4]

reads <data>/il_replay/replays/part-*.parquet (pyarrow, imported only here) and writes
<data>/deck_stats.json (<data> is the machine's data root, src/hastycr/machine.py).

    count_rows(rows)        a DeckCounter over IL_Replay replays rows (dicts with
                            replay_tag, game_mode, payload_json): both seats of every
                            1v1 match in the ranked modes tools/rs_rows.py trains on
    DeckCounter.stats(top)  DeckStats: for each split, the decks ranked by how many seats
                            played them, with their share, score and forms
    DeckField               the sampler: a deck of ONE split, drawn with probability
                            proportional to count ** alpha (alpha 1: its frequency)
    load_stats(path)        DeckStats from the JSON this tool writes

Why: tools/rs_selfplay.py plays one opponent deck per recorded match it replays, only
from Hog 2.6 matches (tokens-15). The RL league (tools/rl/league.py) wants the field
the sparring partners actually meet: the corpus's decks, as often as they are played.

Split: a match is in the split tools/rs_rows.split_of gives its replay_tag (sha1 of the
tag, first 8 hex digits mod 100: under 5 test, under 10 val, else train), so the field
of a split is counted only from that split's matches, the matches the imitation model
trained on (train) or was measured on (val, test). Train RL against the train field;
tools/rs_eval.py measures on test.

Decks: a deck is its 8 base IL_Replay card keys, sorted (the "-ev1" / "-hero" form
suffix removed, as tools/royalesim_redrive.base_key). Forms (0 base, 1 evolution, 2
hero, aligned with the sorted keys) are counted per deck, and the sampler deals them in
the proportions they were played. Levels are not kept: friendly battles are level 11.
Names: the keys map to RoyaleSim card names through tools/rs_cards.load(), the map
tools/rs_rows.py uses, and a match with an unmapped card is skipped there and here
(--no-names counts every deck by its keys alone).

Score: per seat, 1 a win, 0 a loss, 1/2 otherwise, from the recorded result, as
tools/rs_rows.py reads it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

SCHEMA = 1
SPLITS: tuple[str, ...] = ("train", "val", "test")
#: The game modes tools/rs_rows.py (and rs_eval, rs_selfplay) read (rs_rows.MODES).
MODES = frozenset({"Ranked", "Ladder", "1v1 Battle", "Grand Challenge", "Classic Challenge"})
#: The replays parquet columns read.
COLUMNS: tuple[str, ...] = ("replay_tag", "game_mode", "payload_json")
#: Hog 2.6 as base IL_Replay keys, for DeckField(exclude=...).
HOG26_KEYS = frozenset({"cannon", "fireball", "hog-rider", "ice-golem", "ice-spirit",
                        "musketeer", "skeletons", "the-log"})
DECK_SIZE = 8
SPLIT_RULE = ("sha1(replay_tag)[:8] hex % 100: < 5 test, < 10 val, else train "
              "(tools/rs_rows.split_of)")


# ---------------------------------------------------------------------------
# the rules shared with tools/rs_rows.py and tools/royalesim_redrive.py
# ---------------------------------------------------------------------------
def split_of(tag: str) -> str:
    """tools/rs_rows.split_of, verbatim (tests/pipeline/test_league_pool.py checks it)."""
    h = int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 5 else "val" if h < 10 else "train"


def base_key(key):
    """tools/royalesim_redrive.base_key, verbatim: (base key, form 0 / 1 evo / 2 hero)."""
    for suffix in ("-ev1", "-hero"):
        if key.endswith(suffix):
            return key[: -len(suffix)], (1 if suffix == "-ev1" else 2)
    return key, 0


def team_score(result: Any) -> float:
    """The team seat's score from a recorded result, as tools/rs_rows.rows_for reads it."""
    text = str(result or "").lower()
    return 1.0 if text in ("victory", "win") else 0.0 if text in ("defeat", "loss") else 0.5


def deck_of(player: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    """(sorted base keys, forms aligned with them) of one player's deck, or None unless
    it is 8 different cards."""
    pairs = [base_key(str(card["card_key"])) for card in player.get("deck") or []]
    if len(pairs) != DECK_SIZE or len({key for key, _ in pairs}) != DECK_SIZE:
        return None
    pairs.sort()
    return tuple(key for key, _ in pairs), tuple(int(form) for _, form in pairs)


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------
@dataclass
class Tally:
    """One deck in one split: seats that played it, their summed score, form tuples."""

    count: int = 0
    score: float = 0.0
    forms: Counter = field(default_factory=Counter)

    def merge(self, other: Tally) -> None:
        self.count += other.count
        self.score += other.score
        self.forms.update(other.forms)


class DeckCounter:
    """Deck frequencies per split over IL_Replay replays rows.

    names  IL_Replay base key -> RoyaleSim name (tools/rs_cards.load()); a match with a
           key it lacks is skipped, as tools/rs_rows.py skips it. None counts every match.
    modes  the game modes counted (default MODES, rs_rows's); None counts every mode.
    """

    def __init__(self, names: Mapping[str, str] | None = None,
                 modes: Iterable[str] | None = MODES) -> None:
        self.names = dict(names) if names is not None else None
        self.modes = frozenset(modes) if modes is not None else None
        self.decks: dict[str, dict[tuple[str, ...], Tally]] = {s: {} for s in SPLITS}
        self.matches: dict[str, int] = {s: 0 for s in SPLITS}
        self.read = 0
        self.skipped: Counter = Counter()

    def _skip(self, why: str) -> str:
        self.skipped[why] += 1
        return why

    def add_row(self, row: Mapping[str, Any]) -> str:
        """Count one replays row. Returns "ok" or why it was skipped."""
        self.read += 1
        tag = row.get("replay_tag")
        if not tag:
            return self._skip("no tag")
        payload = row.get("payload_json")
        mode = row.get("game_mode")
        if self.modes is not None and mode is not None and mode not in self.modes:
            return self._skip("mode")
        if isinstance(payload, (str, bytes, bytearray)):
            try:
                payload = json.loads(payload)
            except ValueError:
                return self._skip("json")
        if not isinstance(payload, Mapping):
            return self._skip("json")
        battle = payload.get("battle") or {}
        if mode is None:
            mode = battle.get("game_mode")
            if self.modes is not None and mode not in self.modes:
                return self._skip("mode")
        decks = []
        for side in ("team", "opponent"):
            players = (battle.get(side) or {}).get("players") or []
            if len(players) != 1:
                return self._skip("not 1v1")
            try:
                deck = deck_of(players[0])
            except (AttributeError, KeyError, TypeError):
                deck = None
            if deck is None:
                return self._skip("deck")
            decks.append(deck)
        if self.names is not None and any(k not in self.names for cards, _ in decks for k in cards):
            return self._skip("card")
        split = split_of(str(tag))
        won = team_score(battle.get("result"))
        for (cards, forms), score in zip(decks, (won, 1.0 - won)):
            tally = self.decks[split].get(cards)
            if tally is None:
                tally = self.decks[split][cards] = Tally()
            tally.count += 1
            tally.score += score
            tally.forms[forms] += 1
        self.matches[split] += 1
        return "ok"

    def add_rows(self, rows: Iterable[Mapping[str, Any]]) -> DeckCounter:
        for row in rows:
            self.add_row(row)
        return self

    def merge(self, other: DeckCounter) -> DeckCounter:
        """Add another counter's counts (one per parquet part, from a worker)."""
        for split in SPLITS:
            mine = self.decks[split]
            for cards, tally in other.decks[split].items():
                if cards in mine:
                    mine[cards].merge(tally)
                else:
                    mine[cards] = Tally(tally.count, tally.score, Counter(tally.forms))
            self.matches[split] += other.matches[split]
        self.read += other.read
        self.skipped.update(other.skipped)
        return self

    def ranked(self, split: str) -> list[tuple[tuple[str, ...], Tally]]:
        """(cards, tally) of one split, most played first (ties by the keys)."""
        _check_split(split)
        return sorted(self.decks[split].items(), key=lambda item: (-item[1].count, item[0]))

    def stats(self, top: int | None = None, min_count: int = 1,
              meta: Mapping[str, Any] | None = None) -> DeckStats:
        """The ranking of every split, the first `top` decks (all with None) played at
        least `min_count` times. Shares are of all the split's seats, not of the kept."""
        splits = {}
        for split in SPLITS:
            seats = 2 * self.matches[split]
            ranked = [(cards, t) for cards, t in self.ranked(split) if t.count >= min_count]
            if top is not None:
                ranked = ranked[:max(0, int(top))]
            decks = tuple(
                DeckStat(cards=cards, count=t.count, share=t.count / seats if seats else 0.0,
                         score=t.score, names=self._names_of(cards),
                         forms=tuple(sorted(t.forms.items(), key=lambda f: (-f[1], f[0]))))
                for cards, t in ranked)
            splits[split] = SplitStats(split=split, matches=self.matches[split], seats=seats,
                                       distinct=len(self.decks[split]), decks=decks)
        info = {"read": self.read, "skipped": dict(sorted(self.skipped.items())),
                "modes": sorted(self.modes) if self.modes is not None else None,
                "named": self.names is not None, "split_rule": SPLIT_RULE}
        info.update(dict(meta or {}))
        return DeckStats(splits=splits, meta=info)

    def _names_of(self, cards: tuple[str, ...]) -> tuple[str, ...] | None:
        if self.names is None:
            return None
        return tuple(self.names[key] for key in cards)


def count_rows(rows: Iterable[Mapping[str, Any]], names: Mapping[str, str] | None = None,
               modes: Iterable[str] | None = MODES) -> DeckCounter:
    """A DeckCounter over replays rows (see DeckCounter for names / modes)."""
    return DeckCounter(names, modes).add_rows(rows)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split {split!r}: use one of {SPLITS}")


@dataclass(frozen=True)
class DeckStat:
    """One deck of one split. `forms` are (form tuple aligned with cards, seats), most
    played first; `names` the RoyaleSim names aligned with cards (None unmapped)."""

    cards: tuple[str, ...]
    count: int
    share: float
    score: float
    names: tuple[str, ...] | None = None
    forms: tuple[tuple[tuple[int, ...], int], ...] = ()

    @property
    def win_rate(self) -> float:
        return self.score / self.count if self.count else 0.0

    @property
    def modal_forms(self) -> tuple[int, ...]:
        return self.forms[0][0] if self.forms else (0,) * len(self.cards)

    def to_dict(self) -> dict[str, Any]:
        return {"cards": list(self.cards), "names": None if self.names is None else list(self.names),
                "count": self.count, "share": self.share, "score": self.score,
                "forms": [[list(forms), n] for forms, n in self.forms]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeckStat:
        names = data.get("names")
        return cls(cards=tuple(data["cards"]), count=int(data["count"]),
                   share=float(data["share"]), score=float(data["score"]),
                   names=None if names is None else tuple(names),
                   forms=tuple((tuple(int(f) for f in forms), int(n))
                               for forms, n in data.get("forms", [])))


@dataclass(frozen=True)
class SplitStats:
    split: str
    matches: int
    seats: int
    #: Distinct decks counted in the split (all of them, not only the kept).
    distinct: int
    decks: tuple[DeckStat, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"matches": self.matches, "seats": self.seats, "distinct": self.distinct,
                "decks": [d.to_dict() for d in self.decks]}


@dataclass(frozen=True)
class DeckStats:
    splits: dict[str, SplitStats]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "tool": "tools/deck_stats.py", **self.meta,
                "splits": {s: self.splits[s].to_dict() for s in SPLITS if s in self.splits}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeckStats:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"deck stats schema {data.get('schema')!r}; this code reads {SCHEMA}")
        splits = {}
        for split, raw in data["splits"].items():
            _check_split(split)
            splits[split] = SplitStats(
                split=split, matches=int(raw["matches"]), seats=int(raw["seats"]),
                distinct=int(raw["distinct"]),
                decks=tuple(DeckStat.from_dict(d) for d in raw["decks"]))
        meta = {k: v for k, v in data.items() if k not in ("schema", "tool", "splits")}
        return cls(splits=splits, meta=meta)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(path)
        return path

    def deck_field(self, split: str = "train", **kwargs: Any) -> DeckField:
        """DeckField.from_stats(self, split, **kwargs)."""
        return DeckField.from_stats(self, split, **kwargs)


def load_stats(path: str | Path) -> DeckStats:
    return DeckStats.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeckChoice:
    """A dealt deck: base keys, RoyaleSim names (None unmapped) and forms, aligned."""

    cards: tuple[str, ...]
    names: tuple[str, ...] | None
    forms: tuple[int, ...]
    split: str


def _generator(rng: Any) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


class DeckField:
    """Deals decks of one split in proportion to count ** alpha.

    alpha 1 follows the corpus's frequencies; 0 is uniform over the kept decks; between
    them flattens the head. With sample_forms the forms are dealt in the proportions that
    deck was played with them; without, its most common forms.
    """

    def __init__(self, decks: Sequence[DeckStat], *, split: str, alpha: float = 1.0,
                 sample_forms: bool = True) -> None:
        _check_split(split)
        self.decks = tuple(decks)
        if not self.decks:
            raise ValueError(f"no decks to deal in split {split!r}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        counts = np.array([d.count for d in self.decks], dtype=np.float64)
        if np.any(counts <= 0):
            raise ValueError("every deck in a field needs a count > 0")
        weights = counts ** float(alpha)
        self.probabilities = weights / weights.sum()
        self.split = split
        self.alpha = float(alpha)
        self.sample_forms = bool(sample_forms)

    @classmethod
    def from_stats(cls, stats: DeckStats, split: str = "train", *, top: int | None = None,
                   min_count: int = 1, exclude: Iterable[Iterable[str]] = (),
                   alpha: float = 1.0, sample_forms: bool = True) -> DeckField:
        """The field of one split: its first `top` decks played at least `min_count`
        times, without the decks in `exclude` (each given by its keys or its names)."""
        _check_split(split)
        if split not in stats.splits:
            raise ValueError(f"these deck stats have no {split!r} split")
        banned = [frozenset(deck) for deck in exclude]
        decks = [d for d in stats.splits[split].decks
                 if d.count >= min_count and not any(
                     b == frozenset(d.cards) or (d.names is not None and b == frozenset(d.names))
                     for b in banned)]
        if top is not None:
            decks = decks[:max(0, int(top))]
        return cls(decks, split=split, alpha=alpha, sample_forms=sample_forms)

    @classmethod
    def from_file(cls, path: str | Path, split: str = "train", **kwargs: Any) -> DeckField:
        return cls.from_stats(load_stats(path), split, **kwargs)

    def __len__(self) -> int:
        return len(self.decks)

    def probability(self, cards: Iterable[str]) -> float:
        """The chance a draw deals this deck (by keys or names); 0 when not in the field."""
        wanted = frozenset(cards)
        for deck, p in zip(self.decks, self.probabilities):
            if wanted == frozenset(deck.cards) or (deck.names is not None
                                                   and wanted == frozenset(deck.names)):
                return float(p)
        return 0.0

    def sample(self, rng: Any = None, size: int | None = None) -> DeckChoice | list[DeckChoice]:
        """One DeckChoice (size None) or a list of `size`, with replacement. `rng` is a
        numpy Generator or a seed."""
        gen = _generator(rng)
        count = 1 if size is None else int(size)
        picks = gen.choice(len(self.decks), size=count, p=self.probabilities)
        out = []
        for index in picks:
            deck = self.decks[int(index)]
            forms = deck.modal_forms
            if self.sample_forms and len(deck.forms) > 1:
                seats = np.array([n for _, n in deck.forms], dtype=np.float64)
                forms = deck.forms[int(gen.choice(len(deck.forms), p=seats / seats.sum()))][0]
            out.append(DeckChoice(cards=deck.cards, names=deck.names, forms=tuple(forms),
                                  split=self.split))
        return out[0] if size is None else out


# ---------------------------------------------------------------------------
# reading the corpus
# ---------------------------------------------------------------------------
def _parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:                  # the engine venv has it (requirements)
        raise RuntimeError("reading IL_Replay needs pyarrow (pip install 'pyarrow>=15'); "
                           f"it does not import: {exc}") from exc
    return pq


def corpus_parts(corpus: str | Path, parts: int = 0) -> list[Path]:
    """replays/part-*.parquet of the corpus, the first `parts` when parts > 0."""
    files = sorted((Path(corpus) / "replays").glob("part-*.parquet"))
    return files[:parts] if parts > 0 else files


def iter_rows(files: Iterable[str | Path], batch_size: int = 1024) -> Iterator[dict[str, Any]]:
    """Rows (replay_tag, game_mode, payload_json) of replays parquet files, a batch at a
    time so one part is never all in memory."""
    pq = _parquet()
    for part in files:
        handle = pq.ParquetFile(str(part))
        have = set(handle.schema_arrow.names)
        missing = {"replay_tag", "payload_json"} - have
        if missing:
            raise ValueError(f"{part} lacks column(s) {sorted(missing)}")
        columns = [c for c in COLUMNS if c in have]
        for batch in handle.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def count_part(path: str | Path, names: Mapping[str, str] | None = None,
               modes: Iterable[str] | None = MODES) -> DeckCounter:
    """One parquet part's counter (the unit of work of --workers)."""
    return count_rows(iter_rows([path]), names, modes)


def count_corpus(files: Sequence[str | Path], names: Mapping[str, str] | None = None,
                 modes: Iterable[str] | None = MODES, workers: int = 1,
                 progress: bool = False) -> DeckCounter:
    """Every part's counts, merged; `workers` > 1 counts parts in that many processes."""
    total = DeckCounter(names, modes)
    t0 = time.time()

    def note(done: int) -> None:
        if progress and (done % 10 == 0 or done == len(files)):
            print(f"  {done}/{len(files)} parts, {total.read:,} rows ({time.time() - t0:.0f}s)",
                  flush=True)

    if workers <= 1 or len(files) <= 1:
        for done, part in enumerate(files, 1):
            total.merge(count_part(part, names, modes))
            note(done)
        return total
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(count_part, str(part), names, modes) for part in files]
        for done, future in enumerate(futures, 1):
            total.merge(future.result())
            note(done)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def data_root() -> Path:
    """The machine's data root (src/hastycr/machine.py), else <repo>/data."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from hastycr import machine
        return Path(machine.data_root())
    except Exception as exc:                    # a broken machine.json: say so, use data/
        print(f"machine settings unreadable ({type(exc).__name__}: {exc}); using {ROOT / 'data'}",
              file=sys.stderr)
        return ROOT / "data"


def card_map() -> tuple[dict[str, str], dict[str, Any]]:
    """tools/rs_cards.load() (the map tools/rs_rows.py uses) and its description."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import rs_cards
    names = rs_cards.load()
    return names, rs_cards.describe(names)


def describe_top(stats: DeckStats, n: int = 10) -> str:
    lines = []
    for split in SPLITS:
        s = stats.splits.get(split)
        if s is None:
            continue
        lines.append(f"{split}: {s.matches:,} matches, {s.distinct:,} distinct decks")
        for rank, d in enumerate(s.decks[:n], 1):
            label = ", ".join(d.names if d.names is not None else d.cards)
            lines.append(f"  {rank:3d}. {100 * d.share:5.2f}%  win {d.win_rate:.3f}  {label}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="IL_Replay folder holding replays/ (default <data>/il_replay)")
    ap.add_argument("--parts", type=int, default=0, help="only the first N parts (0: all)")
    ap.add_argument("--top", type=int, default=500, help="decks kept per split (0: all)")
    ap.add_argument("--min-count", type=int, default=1, help="drop decks played fewer times")
    ap.add_argument("--workers", type=int, default=1, help="processes counting parts")
    ap.add_argument("--all-modes", action="store_true",
                    help=f"count every game mode, not only {sorted(MODES)}")
    ap.add_argument("--no-names", action="store_true",
                    help="do not map keys to RoyaleSim names (and keep unmapped decks)")
    ap.add_argument("--out", type=Path, default=None, help="default <data>/deck_stats.json")
    ap.add_argument("--show", type=int, default=10, help="decks printed per split")
    args = ap.parse_args(argv)
    if args.top < 0 or args.min_count < 1 or args.workers < 1 or args.parts < 0:
        ap.error("--top >= 0, --min-count >= 1, --workers >= 1, --parts >= 0")
    root = None if (args.corpus and args.out) else data_root()
    corpus = args.corpus or root / "il_replay"
    out = args.out or root / "deck_stats.json"
    files = corpus_parts(corpus, args.parts)
    if not files:
        print(f"no IL_Replay parquet in {corpus / 'replays'} (run tools/il_replay_download.py)",
              file=sys.stderr)
        return 1
    names, described = (None, None) if args.no_names else card_map()
    modes = None if args.all_modes else MODES
    t0 = time.time()
    counter = count_corpus(files, names, modes, workers=args.workers, progress=True)
    stats = counter.stats(top=args.top or None, min_count=args.min_count, meta={
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": str(corpus), "parts": [f.name for f in files], "card_map": described,
    })
    stats.save(out)
    print(describe_top(stats, args.show))
    print(f"read {counter.read:,} rows, skipped {dict(counter.skipped)} "
          f"({time.time() - t0:.0f}s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
