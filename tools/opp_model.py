r"""The opponent model's deck prior, from the IL_Replay TRAIN split only (S03).

    .venvs\royalegym\Scripts\python.exe tools\opp_model.py build-prior --split train
        [--parts N] [--max-decks 5000] [--min-count 2] [--out data\opp_prior\deck_prior.json]
    .venvs\royalegym\Scripts\python.exe tools\opp_model.py show [--known HogRider,Log] [--n 6]

    from tools import opp_model as OM               # or `import opp_model` (tools/ on sys.path)
    prior = OM.load_prior(card_names=names)          # names: catalogue order; None before a build
    prior.top(known, n)                              # the n most likely unknown cards (world 0)
    prior.sample(known, n, rng)                      # n unknown cards drawn (worlds 1..K-1)
    opp_fill = prior.fill_cards(cycle, rng, known=recast_cards)   # -> rs_rebuild.build_spec

Why (docs/LIVE_SEARCH_PLAN.md section 5, S03). The old search rebuilt the opponent with its TRUE
deck (rs_eval.py:352): a leak. A screen shows only the cards the opponent has played, so the rest
of its deck must come from a prior. The prior is counted from the IL_Replay train split only
(rs_match.read_pairs, split "train", the rows rs_match.load_pairs reads, rs_match.py:1402-1415),
so the gate's dev pairings (and the held-out test split) never inform it. Before a prior exists,
the rebuild pads the deck with Hog 2.6 (rs_spawn_table.FILLER_NAMES, public_state's
sanitize_for_rebuild): load_prior returns None then.

THE DRAW (plan section 5). An unknown card c is drawn with weight
    w(c | K) = sum over the train seats' decks d holding c of exp(beta * |d & K|),  beta = 1,
K the cards known to be in the deck (the known set). A draw of n cards draws them one at a time,
each drawn card joining K for the next draw, so the n cards complete one coherent deck rather than
n independent marginals; cards of K, of `exclude` and cards the engine lacks (bind) are never
drawn, and no card twice. top() takes the heaviest card each time (ties: the more played card,
then the name), which is the plan's world 0; sample() draws by weight with the caller's rng (a
numpy Generator, random.Random or a seed; only rng.random() is called, once per card), never the
seat's own (plan 3.1: the search has its own generator). Two parts make up the weight:
  kept decks   the most played decks (count >= min_count, the first max_decks), exactly as the
               formula says: one row per deck with its seat count
  the rest     the seats of every other deck, by the cards' co-occurrence: with R(c) of those
               seats holding c and R(c, k) holding c and k,
                   R(c) * prod over k in K of (1 + R(c, k) / R(c) * (exp(beta) - 1)),
               the formula's expectation when the known cards occur independently given c; exact
               for |K| <= 1, an approximation from two known cards on. R comes from the file's
               card and pair counts over ALL train seats less the kept decks' own.
  max_decks 0 with min_count 1 keeps every deck: the formula exactly, in a larger file.
Never together: a card that shares no train deck with a known card although independence would
put them together in at least min_expected seats (25: two champions, which no deck holds) is not
drawn while another card can be; when only such cards are left, they are (the known set may hold
a misread).

THE CYCLE (plan section 5, rs_public_view.CardCycle, S02). fill_cards(cycle, rng, known) returns
the cards for a side's open slots: as many as the cycle's holes, less its unplaced cards (fill
places those first) and less the extra `known` cards (recast cards, a kept building's card), each
of which takes an open slot. With rng None it is top() (world 0); with an rng the draw is shuffled
with it, so which drawn card sits in the hand and which at the queue's head is random, as a real
deal is (CardCycle.fill fills hand holes first; Mirror and Elixir Collector, `omit`, go to the
queue). rs_rebuild.build_spec(opp_fill=...) takes the list (rs_rebuild.py:1083, 1310);
fill_cycle() is the same fill applied to the cycle itself. The world builder (S07, plan section
5: slips, elixir bumps, jitter, the in-flight play and the reply) is not here.

THE FILE (data/opp_prior/deck_prior.json, render()): schema, tool, kind "deck_prior", split
"train" (load refuses any other), created; "source": the train pairs read (pairs, the
PairSet.digest of those pairs, rs_match.py:1314-1318, equal to rs_match.load_pairs(split="train",
exclude_mirror=False).digest() over the same parts), distinct decks, the split rule, the modes,
skipped rows, the rows of other splits passed over, the part file names and the card map's
description (rs_cards.describe); "kept": the kept decks' count, seats and share and the cut;
"seats" (2 per match); "cards" (RoyaleSim names, most played first); "card_seats"; "pairs" (row i:
the seats holding cards[i] and cards[j], j > i); "decks" ([seats, i0..i7] card indexes, most played
first). ANONYMOUS: decks and counts only, never a replay tag, player, clan or path. With the
defaults it is a few hundred KB, so it is committed (`git add data/opp_prior/deck_prior.json`);
build-prior warns above COMMIT_BYTES. Mirror matches count both seats (exclude_mirror False: a deck
prior counts seats, not pairings).

TRAIN ONLY, in code: PriorBuilder refuses a pair set of another split, and any pair whose replay
tag rs_rows.split_of (deck_stats.split_of) does not put in train; `--split` accepts only train; a
file of another split does not load. The prior never reads a battle or a state: its only inputs
are the card ids the caller passes, which the search takes from the public view (S02).

numpy, tools/deck_stats (numpy) and tools/rs_public_view (pure Python) only; tools/rs_match (and
pyarrow, through deck_stats.iter_rows) only when a prior is built. Never imports royalesim, torch
or scripts/gamestate; never taps or touches the emulator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:          # also when imported as tools.opp_model
    sys.path.insert(0, str(ROOT / "tools"))

import deck_stats  # noqa: E402  (split_of, SPLIT_RULE, MODES, corpus_parts, iter_rows; numpy)
import rs_public_view as PV  # noqa: E402  (CardCycle; pure Python)

__all__ = ["BETA", "COMMIT_BYTES", "DEFAULT_PATH", "DeckPrior", "DeckPriorError", "MAX_DECKS",
           "MIN_COUNT", "MIN_EXPECTED", "PriorBuilder", "TRAIN", "build_from_corpus",
           "build_from_rows", "load_prior", "main", "render"]

TOOL = "tools/opp_model.py"
SCHEMA = 1
KIND = "deck_prior"
#: The only split a prior is counted from (rs_rows.split_of's name).
TRAIN = "train"
DEFAULT_PATH = ROOT / "data" / "opp_prior" / "deck_prior.json"
DECK_SIZE = 8
#: The plan's beta: the weight of each known card a deck shares (section 5).
BETA = 1.0
BETA_MAX = 20.0
#: The kept decks (module doc, THE DRAW): at least MIN_COUNT seats, the first MAX_DECKS.
MAX_DECKS = 5000
MIN_COUNT = 2
#: Never together: no shared deck where independence expects at least this many seats.
MIN_EXPECTED = 25.0
#: build-prior warns that a file above this is too large to commit.
COMMIT_BYTES = 1_000_000
#: The keys a file holds (the anonymity test reads them).
FILE_KEYS = ("schema", "tool", "kind", "split", "created", "source", "kept", "seats", "cards",
             "card_seats", "pairs", "decks")
#: Keys rendered one row per line.
ROW_KEYS = ("pairs", "decks")
TOP_CACHE = 256
CHUNK = 20_000


class DeckPriorError(ValueError):
    """A prior that cannot be built or read: a pair of another split than train, a file of
    another split or schema, counts that do not add up."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _label(value: Any) -> Any:
    """A card as the prior's labels hold it: an int for a catalogue id (numpy ints too), else as
    given (a name); None for None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _deck_key(deck: Iterable[Any]) -> tuple[str, ...]:
    """A deck as the prior counts it: its 8 distinct card names, sorted."""
    cards = tuple(sorted(str(c) for c in deck))
    if len(cards) != DECK_SIZE or len(set(cards)) != DECK_SIZE:
        raise DeckPriorError(f"a deck is {DECK_SIZE} distinct cards, got {list(cards)}")
    return cards


def _generator(rng: Any) -> Any:
    """Something with .random(): a numpy Generator, random.Random, or a seed for a Generator."""
    if rng is None:
        raise ValueError("sample() needs an rng (top() is the draw without one)")
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
        return np.random.default_rng(int(rng))
    if not callable(getattr(rng, "random", None)):
        raise TypeError(f"rng: a numpy Generator, random.Random or a seed, got {type(rng).__name__}")
    return rng


def _shuffled(items: Sequence[Any], rng: Any) -> list:
    """Fisher-Yates with rng.random() only (the same for a Generator and random.Random)."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = min(i, int(float(rng.random()) * (i + 1)))
        out[i], out[j] = out[j], out[i]
    return out


def _square(rows: Sequence[Sequence[Any]], n: int) -> np.ndarray:
    """The symmetric pair matrix from its upper-triangle rows (row i: j = i+1 .. n-1)."""
    rows = list(rows)
    if len(rows) != n:
        raise DeckPriorError(f"pairs: {n} rows expected, got {len(rows)}")
    out = np.zeros((n, n), dtype=np.float64)
    for i, row in enumerate(rows):
        row = [float(v) for v in row]
        if len(row) != n - 1 - i:
            raise DeckPriorError(f"pairs row {i}: {n - 1 - i} counts expected, got {len(row)}")
        if any(v < 0 or not math.isfinite(v) for v in row):
            raise DeckPriorError(f"pairs row {i}: counts are >= 0")
        out[i, i + 1:] = row
    return out + out.T


def _triangle(square: np.ndarray) -> list[list[int]]:
    n = square.shape[0]
    return [[int(round(v)) for v in square[i, i + 1:]] for i in range(n)]


class _Draw:
    """One draw's running state: deck overlaps with the known set, the rest's log factors, the
    cards no longer drawable, the cards never seen with a known one."""
    __slots__ = ("ov", "lsum", "blocked", "apart")

    def __init__(self, ov: np.ndarray, lsum: np.ndarray, blocked: np.ndarray, apart: np.ndarray):
        self.ov, self.lsum, self.blocked, self.apart = ov, lsum, blocked, apart


# ---------------------------------------------------------------------------
# the prior
# ---------------------------------------------------------------------------

class DeckPrior:
    """The deck prior (module doc). Build one with PriorBuilder / build-prior (or from_counts for
    a hand-made one), read one with load / load_prior, then bind(card_names) to draw catalogue ids.

    cards       the vocabulary: RoyaleSim card names, most played first
    card_seats  train seats holding each card (all seats)
    pair_rows   upper-triangle rows of the seats holding two cards (all seats)
    decks       the kept decks: (seats, card indexes), most played first
    seats       train seats in all (2 per match)
    beta        the formula's beta (BETA); min_expected the never-together threshold
                (MIN_EXPECTED; math.inf turns it off)
    """

    def __init__(self, cards: Sequence[str], card_seats: Sequence[Any],
                 pair_rows: Sequence[Sequence[Any]], decks: Iterable[tuple[Any, Sequence[int]]], *,
                 seats: int, source: Mapping[str, Any] | None = None,
                 kept: Mapping[str, Any] | None = None, beta: float = BETA,
                 min_expected: float = MIN_EXPECTED, created: str | None = None):
        self.cards = tuple(str(c) for c in cards)
        V = len(self.cards)
        if V == 0 or len(set(self.cards)) != V:
            raise DeckPriorError("cards: one or more distinct names")
        self.seats = int(seats)
        cs = np.array([float(v) for v in card_seats], dtype=np.float64)
        if cs.shape != (V,) or (cs < 0).any() or not np.isfinite(cs).all():
            raise DeckPriorError(f"card_seats: {V} counts >= 0 expected")
        if self.seats < 1 or self.seats < cs.max():
            raise DeckPriorError(f"seats {self.seats}: at least 1 and every card's count")
        P = _square(pair_rows, V)
        out = []
        for n, idx in decks:
            idx = tuple(int(i) for i in idx)
            if (int(n) < 1 or len(idx) != DECK_SIZE or len(set(idx)) != DECK_SIZE
                    or min(idx) < 0 or max(idx) >= V):
                raise DeckPriorError(f"deck {[n, *idx]}: seats >= 1 and {DECK_SIZE} distinct card "
                                     f"indexes below {V}")
            out.append((int(n), idx))
        self.decks = tuple(out)
        self.beta = float(beta)
        if not 0.0 <= self.beta <= BETA_MAX:
            raise DeckPriorError(f"beta {beta!r}: 0 .. {BETA_MAX:g}")
        self.min_expected = float(min_expected)
        if math.isnan(self.min_expected) or self.min_expected < 0:
            raise DeckPriorError(f"min_expected {min_expected!r}: >= 0 (inf: off)")
        self.source = dict(source or {})
        self.kept = dict(kept or {})
        self.created = created
        self._setup(cs, P)
        self._relabel(self.cards)
        self.bound = False

    # ------------------------------------------------------------ building
    @classmethod
    def from_counts(cls, decks: Mapping[Iterable[str], int] | Iterable[tuple[Iterable[str], int]],
                    *, max_decks: int = 0, min_count: int = 1,
                    source: Mapping[str, Any] | None = None,
                    beta: float = BETA, min_expected: float = MIN_EXPECTED,
                    created: str | None = None) -> "DeckPrior":
        """A prior from deck -> seats (8 card names each). max_decks 0 keeps every deck of at least
        min_count seats."""
        items = decks.items() if isinstance(decks, Mapping) else decks
        tally: Counter = Counter()
        for deck, n in items:
            if int(n) < 1:
                raise DeckPriorError(f"deck {sorted(deck)}: seats >= 1, got {n!r}")
            tally[_deck_key(deck)] += int(n)
        if not tally:
            raise DeckPriorError("no decks")
        if int(max_decks) < 0 or int(min_count) < 1:
            raise DeckPriorError("max_decks >= 0 (0: all), min_count >= 1")
        per_card: Counter = Counter()
        pairs: Counter = Counter()
        for deck, n in tally.items():
            for c in deck:
                per_card[c] += n
        cards = sorted(per_card, key=lambda c: (-per_card[c], c))
        index = {c: i for i, c in enumerate(cards)}
        for deck, n in tally.items():
            ids = sorted(index[c] for c in deck)
            for a in range(DECK_SIZE):
                for b in range(a + 1, DECK_SIZE):
                    pairs[ids[a], ids[b]] += n
        V = len(cards)
        rows = [[pairs.get((i, j), 0) for j in range(i + 1, V)] for i in range(V)]
        ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [(n, tuple(sorted(index[c] for c in deck))) for deck, n in ranked
                if n >= int(min_count)]
        if int(max_decks):
            kept = kept[:int(max_decks)]
        seats = sum(tally.values())
        held = sum(n for n, _ in kept)
        info = {"decks": len(kept), "seats": held, "share": round(held / seats, 6),
                "min_count": int(min_count), "max_decks": int(max_decks), "distinct": len(tally)}
        return cls(cards, [per_card[c] for c in cards], rows, kept, seats=seats, source=source,
                   kept=info, beta=beta, min_expected=min_expected, created=created)

    def _setup(self, cs: np.ndarray, P: np.ndarray) -> None:
        V, D = len(self.cards), len(self.decks)
        idx = np.array([i for _, i in self.decks], dtype=np.intp).reshape(D, DECK_SIZE)
        n = np.array([c for c, _ in self.decks], dtype=np.float64)
        kept_seats = np.bincount(idx.ravel(), weights=np.repeat(n, DECK_SIZE), minlength=V)
        kept_pairs = np.zeros((V, V), dtype=np.float64)
        for start in range(0, D, CHUNK):
            part, w = idx[start:start + CHUNK], n[start:start + CHUNK]
            X = np.zeros((len(part), V), dtype=np.float64)
            X[np.arange(len(part))[:, None], part] = 1.0
            kept_pairs += (X * w[:, None]).T @ X
        np.fill_diagonal(kept_pairs, 0.0)
        tol = 1e-6
        if (cs + tol < kept_seats).any() or (P + tol < kept_pairs).any():
            raise DeckPriorError("the kept decks hold more seats than the card or pair counts: "
                                 "the counts do not add up (a hand-edited file?)")
        R = np.clip(cs - kept_seats, 0.0, None)
        rest = np.clip(P - kept_pairs, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(R[:, None] > 0, rest / np.where(R > 0, R, 1.0)[:, None], 0.0)
        ratio = np.clip(ratio, 0.0, 1.0)
        np.fill_diagonal(ratio, 0.0)
        expected = np.outer(cs, cs) / float(self.seats)
        apart = (P <= 0) & (expected >= self.min_expected)
        np.fill_diagonal(apart, False)
        self._idx, self._n, self._R = idx, n, R
        self._L = np.log1p(ratio * math.expm1(self.beta))
        self._apart = apart
        self._etab = np.exp(self.beta * np.arange(DECK_SIZE + 1, dtype=np.float64))
        self._cs, self._P = cs, P
        for a in (self._idx, self._n, self._R, self._L, self._apart, self._etab, self._cs,
                  self._P):
            a.flags.writeable = False
        self._top_cache: dict = {}

    def _relabel(self, labels: Sequence[Any]) -> None:
        labels = tuple(_label(x) for x in labels)
        index = {}
        for i, x in enumerate(labels):
            if x is None:
                continue
            if x in index:
                raise DeckPriorError(f"label {x!r} given to two cards")
            index[x] = i
        self.labels = labels
        self._index = index
        self._drawable = np.array([x is not None for x in labels], dtype=bool)
        self._drawable.flags.writeable = False
        self._top_cache = {}

    def bind(self, card_names: Sequence[str]) -> "DeckPrior":
        """A copy whose cards are catalogue ids: card_names in catalogue order (catalogue_json()
        rows' names, rs_rebuild's Tables.names). A card the catalogue lacks is never drawn."""
        ids: dict[str, int] = {}
        for i, name in enumerate(card_names):
            ids.setdefault(str(name), i)
        out = copy.copy(self)
        out._relabel([ids.get(c) for c in self.cards])
        out.bound = True
        return out

    def __contains__(self, card: Any) -> bool:
        return _label(card) in self._index

    def __len__(self) -> int:
        return len(self.cards)

    # ------------------------------------------------------------ the draw
    def _start(self, known: Iterable[Any], exclude: Iterable[Any]) -> _Draw:
        V = len(self.cards)
        k = sorted({self._index[c] for c in map(_label, known) if c in self._index})
        mask = np.zeros(V, dtype=bool)
        mask[k] = True
        ov = mask[self._idx].sum(axis=1).astype(np.intp)
        lsum = self._L[:, k].sum(axis=1) if k else np.zeros(V, dtype=np.float64)
        blocked = ~self._drawable
        blocked[k] = True
        for c in map(_label, exclude):
            if c in self._index:
                blocked[self._index[c]] = True
        apart = self._apart[:, k].any(axis=1) if k else np.zeros(V, dtype=bool)
        return _Draw(ov, lsum, blocked, apart)

    def _weights(self, st: _Draw) -> np.ndarray:
        w = self._n * self._etab[st.ov]
        kept = np.bincount(self._idx.ravel(), weights=np.repeat(w, DECK_SIZE),
                           minlength=len(self.cards))
        total = kept + self._R * np.exp(st.lsum)
        total[st.blocked] = 0.0
        return total

    @staticmethod
    def _candidates(total: np.ndarray, st: _Draw) -> np.ndarray:
        cand = np.where(st.apart, 0.0, total)
        return cand if cand.sum() > 0 else total

    def _advance(self, st: _Draw, i: int) -> None:
        st.ov = st.ov + (self._idx == i).any(axis=1)
        st.lsum = st.lsum + self._L[:, i]
        st.blocked[i] = True
        st.apart = st.apart | self._apart[:, i]

    def _pick(self, cand: np.ndarray, rng: Any) -> int | None:
        if not cand.sum() > 0:
            return None
        if rng is None:
            return int(np.argmax(cand))
        cum = np.cumsum(cand)
        i = int(np.searchsorted(cum, float(rng.random()) * cum[-1], side="right"))
        if i >= len(cand):                       # u * total rounded up to the total
            i = int(np.flatnonzero(cand > 0)[-1])
        return i

    def _draw(self, known: Iterable[Any], n: int, rng: Any, exclude: Iterable[Any]) -> list:
        n = int(n)
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        st = self._start(known, exclude)
        out = []
        for _ in range(n):
            i = self._pick(self._candidates(self._weights(st), st), rng)
            if i is None:
                break
            out.append(self.labels[i])
            self._advance(st, i)
        return out

    def sample(self, known: Iterable[Any], n: int, rng: Any, exclude: Iterable[Any] = ()) -> list:
        """n unknown cards drawn by weight (module doc, THE DRAW), each joining the known set for
        the next; fewer when no drawable card is left. rng: a numpy Generator, random.Random or a
        seed (one rng.random() per card)."""
        return self._draw(known, n, _generator(rng), exclude)

    def top(self, known: Iterable[Any], n: int, exclude: Iterable[Any] = ()) -> list:
        """The n heaviest unknown cards, one at a time (ties: the more played card, then the name):
        the plan's most likely world. Cached per (known, n, exclude)."""
        known, exclude = list(known), list(exclude)
        key = (frozenset(self._index[c] for c in map(_label, known) if c in self._index), int(n),
               frozenset(self._index[c] for c in map(_label, exclude) if c in self._index))
        got = self._top_cache.get(key)
        if got is None:
            got = tuple(self._draw(known, n, None, exclude))
            if len(self._top_cache) >= TOP_CACHE:
                self._top_cache.clear()
            self._top_cache[key] = got
        return list(got)

    def draw(self, known: Iterable[Any], n: int, rng: Any = None,
             exclude: Iterable[Any] = ()) -> list:
        """top() without an rng (world 0), sample() with one."""
        return self.top(known, n, exclude) if rng is None else self.sample(known, n, rng, exclude)

    def weights(self, known: Iterable[Any] = (), exclude: Iterable[Any] = ()) -> dict:
        """label -> w(c | known), the formula's weight of each drawable card for the first draw
        (before the never-together rule); cards of known and exclude are left out."""
        st = self._start(known, exclude)
        total = self._weights(st)
        return {self.labels[i]: float(total[i]) for i in np.flatnonzero(total > 0)}

    def probabilities(self, known: Iterable[Any] = (), exclude: Iterable[Any] = ()) -> dict:
        """label -> the chance the first card sample() draws is that card (never-together
        applied)."""
        st = self._start(known, exclude)
        cand = self._candidates(self._weights(st), st)
        s = float(cand.sum())
        if not s > 0:
            return {}
        return {self.labels[i]: float(cand[i]) / s for i in np.flatnonzero(cand > 0)}

    def never_together(self, a: Any, b: Any) -> bool:
        """True when the never-together rule keeps b from being drawn beside a (and a beside b)."""
        a, b = _label(a), _label(b)
        if a not in self._index or b not in self._index:
            return False
        return bool(self._apart[self._index[a], self._index[b]])

    # ------------------------------------------------------------ cycles
    def _bound_cycle(self, cycle: Any) -> PV.CardCycle:
        """A CardCycle (duck-typed: rs_public_view and tools.rs_public_view are two modules when
        imported both ways) or its to_json()."""
        if not self.bound:
            raise DeckPriorError("a card cycle holds catalogue ids: bind(card_names) the prior "
                                 "first")
        return PV.CardCycle.from_json(cycle) if isinstance(cycle, Mapping) else cycle

    def fill_cards(self, cycle: Any, rng: Any = None, *, known: Iterable[Any] = (),
                   exclude: Iterable[Any] = ()) -> list[int]:
        """The cards for `cycle`'s open slots (module doc, THE CYCLE): CardCycle or its to_json();
        known: cards in the deck besides the cycle's (recast cards, a kept building's card), each
        taking an open slot; rng None: top() in order (world 0), else a shuffled sample()."""
        cyc = self._bound_cycle(cycle)
        slotted = {c for c in cyc.known}
        have = slotted | {c for c in cyc.seen if c is not None} | set(cyc.unplaced)
        extra = [c for c in dict.fromkeys(_label(x) for x in known)
                 if c is not None and c not in have]
        unplaced = [c for c in cyc.unplaced if c not in slotted]
        n = max(0, sum(cyc.holes) - len(unplaced) - len(extra))
        known_all = list(have) + extra
        if rng is None:
            return [int(c) for c in self.top(known_all, n, exclude)]
        rng = _generator(rng)
        return [int(c) for c in _shuffled(self._draw(known_all, n, rng, exclude), rng)]

    def fill_cycle(self, cycle: Any, rng: Any = None, *, known: Iterable[Any] = (),
                   omit: Iterable[int] = (), exclude: Iterable[Any] = ()) -> PV.CardCycle:
        """`cycle` with every open slot filled: its unplaced cards, the extra `known` cards and
        fill_cards' draw (shuffled together with an rng), by CardCycle.fill (omit: Mirror and
        Elixir Collector into queue slots first)."""
        cyc = self._bound_cycle(cycle)
        rng = None if rng is None else _generator(rng)     # one generator for draw and shuffle
        have = set(cyc.known) | {c for c in cyc.seen if c is not None} | set(cyc.unplaced)
        extra = [c for c in dict.fromkeys(_label(x) for x in known)
                 if c is not None and c not in have]
        drawn = self.fill_cards(cyc, rng, known=extra, exclude=exclude)
        pool = extra + drawn
        if rng is not None:
            pool = _shuffled(pool, rng)
        return cyc.fill(pool, omit=omit)

    # ------------------------------------------------------------ the file
    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "tool": TOOL, "kind": KIND, "split": TRAIN,
                "created": self.created, "source": dict(self.source), "kept": dict(self.kept),
                "seats": self.seats, "cards": list(self.cards),
                "card_seats": [int(round(v)) for v in self._cs], "pairs": _triangle(self._P),
                "decks": [[n, *idx] for n, idx in self.decks]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, beta: float = BETA,
                  min_expected: float = MIN_EXPECTED) -> "DeckPrior":
        if data.get("schema") != SCHEMA or data.get("kind") != KIND:
            raise DeckPriorError(f"not a deck prior of schema {SCHEMA} (schema "
                                 f"{data.get('schema')!r}, kind {data.get('kind')!r})")
        if data.get("split") != TRAIN:
            raise DeckPriorError(f"a deck prior must come from the train split, this one says "
                                 f"{data.get('split')!r}: the gate's dev pairings never inform it")
        try:
            decks = [(row[0], row[1:]) for row in data["decks"]]
            return cls(data["cards"], data["card_seats"], data["pairs"], decks,
                       seats=data["seats"], source=data.get("source"), kept=data.get("kept"),
                       beta=beta, min_expected=min_expected, created=data.get("created"))
        except (KeyError, TypeError, IndexError) as exc:
            raise DeckPriorError(f"a malformed deck prior ({type(exc).__name__}: {exc})") from exc

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(render(self.to_dict()).encode("utf-8"))
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, card_names: Sequence[str] | None = None,
             beta: float = BETA, min_expected: float = MIN_EXPECTED) -> "DeckPrior":
        """The prior in `path`, bound to card_names when given."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise DeckPriorError(f"{path}: not JSON ({exc})") from exc
        prior = cls.from_dict(data, beta=beta, min_expected=min_expected)
        return prior.bind(card_names) if card_names is not None else prior

    def digest(self) -> str:
        """16 hex digits over what the draw reads (counts, decks, beta, min_expected): equal
        means the same draws. For the search's game keys (S08)."""
        body = json.dumps([list(self.cards), [int(round(v)) for v in self._cs], _triangle(self._P),
                           [[n, *idx] for n, idx in self.decks], self.seats, self.beta,
                           None if math.isinf(self.min_expected) else self.min_expected],
                          separators=(",", ":"))
        return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]

    def describe(self, decks: int = 10, cards: int = 12) -> str:
        src = self.source
        lines = [f"deck prior {self.digest()} (train): {src.get('pairs', '?')} pairs, "
                 f"{self.seats} seats, {self.kept.get('distinct', src.get('distinct_decks', '?'))} "
                 f"distinct decks; kept {len(self.decks)} decks holding "
                 f"{100 * float(self.kept.get('share', 0)):.1f}% of the seats; "
                 f"{len(self.cards)} cards; beta {self.beta:g}"]
        top = ", ".join(f"{c} {100 * self._cs[i] / self.seats:.1f}%"
                        for i, c in enumerate(self.cards[:cards]))
        lines.append(f"  most played cards: {top}")
        for rank, (n, idx) in enumerate(self.decks[:decks], 1):
            lines.append(f"  {rank:3d}. {100 * n / self.seats:5.2f}%  "
                         + ", ".join(self.cards[i] for i in idx))
        return "\n".join(lines)


def render(doc: Mapping[str, Any]) -> str:
    """The file's text: one key per line, the rows of ROW_KEYS one per line (a reviewable diff),
    a final newline."""
    items = list(doc.items())
    lines = ["{"]
    for k, (key, value) in enumerate(items):
        end = "," if k < len(items) - 1 else ""
        name = json.dumps(str(key))
        if key in ROW_KEYS and isinstance(value, list) and value:
            lines.append(f" {name}: [")
            for r, row in enumerate(value):
                lines.append("  " + json.dumps(row, separators=(",", ":"))
                             + ("," if r < len(value) - 1 else ""))
            lines.append(" ]" + end)
        else:
            lines.append(f" {name}: {json.dumps(value, separators=(', ', ': '))}{end}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def load_prior(path: str | Path | None = None, *, card_names: Sequence[str] | None = None,
               required: bool = False, beta: float = BETA,
               min_expected: float = MIN_EXPECTED) -> DeckPrior | None:
    """DeckPrior.load(path or DEFAULT_PATH), bound to card_names when given; None when the file
    does not exist yet (the rebuild then pads with Hog 2.6) unless required."""
    path = Path(path) if path is not None else DEFAULT_PATH
    if not path.exists():
        if required:
            raise FileNotFoundError(f"no deck prior at {path}: run {TOOL} build-prior --split train")
        return None
    return DeckPrior.load(path, card_names=card_names, beta=beta, min_expected=min_expected)


# ---------------------------------------------------------------------------
# building from the train split
# ---------------------------------------------------------------------------

class PriorBuilder:
    """Counts the train split's decks pair by pair (rs_match.DeckPair, or anything with tag,
    split, decks and forms), refusing every pair that is not train (module doc, TRAIN ONLY).
    digest() is rs_match.PairSet.digest() of the pairs added, in order."""

    def __init__(self) -> None:
        self.decks: Counter = Counter()
        self.tags: set[str] = set()
        self.pairs = 0
        self.skipped: Counter = Counter()
        self.other_splits = 0
        self.parts: list[str] = []
        self._sha = hashlib.sha1(b"[")
        self._first = True

    def add(self, pair: Any) -> bool:
        """Count one pair's two seats; False for a tag already counted."""
        tag = str(pair.tag)
        split = deck_stats.split_of(tag)
        if str(pair.split) != TRAIN or split != TRAIN:
            raise DeckPriorError(f"pair {tag!r} is in the {split!r} split (says {pair.split!r}): "
                                 "a deck prior counts the train split only")
        if tag in self.tags:
            self.skipped["duplicate"] += 1
            return False
        decks = [_deck_key(d) for d in pair.decks]
        if len(decks) != 2:
            raise DeckPriorError(f"pair {tag!r}: two decks expected, got {len(decks)}")
        self.tags.add(tag)
        self.pairs += 1
        for d in decks:
            self.decks[d] += 1
        item = [tag, [list(d) for d in pair.decks], [list(f) for f in pair.forms]]
        self._sha.update((b"" if self._first else b",")
                         + json.dumps(item, separators=(",", ":")).encode("utf-8"))
        self._first = False
        return True

    def add_set(self, pairs: Any) -> "PriorBuilder":
        """Every pair of an rs_match.PairSet of the train split; its meta's skipped counts."""
        if str(pairs.split) != TRAIN:
            raise DeckPriorError(f"a pair set of the {pairs.split!r} split: a deck prior counts the "
                                 "train split only")
        meta = pairs.meta or {}
        self.skipped.update({str(k): int(v) for k, v in (meta.get("skipped") or {}).items()})
        self.other_splits += int(meta.get("other_splits") or 0)
        for pair in pairs.pairs:
            self.add(pair)
        return self

    def digest(self) -> str:
        h = self._sha.copy()
        h.update(b"]")
        return h.hexdigest()[:16]

    def prior(self, *, max_decks: int = MAX_DECKS, min_count: int = MIN_COUNT,
              beta: float = BETA, min_expected: float = MIN_EXPECTED,
              meta: Mapping[str, Any] | None = None, created: str | None = None) -> DeckPrior:
        if not self.pairs:
            raise DeckPriorError("no train pairs were read")
        source = {"pairs": self.pairs, "digest": self.digest(), "distinct_decks": len(self.decks),
                  "split_rule": deck_stats.SPLIT_RULE, "exclude_mirror": False,
                  "skipped": dict(sorted(self.skipped.items())), "other_splits": self.other_splits,
                  "parts": list(self.parts)}
        source.update(dict(meta or {}))
        return DeckPrior.from_counts(self.decks, max_decks=max_decks, min_count=min_count,
                                     source=source, beta=beta, min_expected=min_expected,
                                     created=created)


def _rs_match():
    import rs_match                              # heavy (rs_eval, rs_rows): only to build
    return rs_match


def build_from_rows(rows: Iterable[Mapping[str, Any]], names: Mapping[str, str], *,
                    modes: Iterable[str] | None = deck_stats.MODES,
                    builder: PriorBuilder | None = None) -> PriorBuilder:
    """IL_Replay replays rows -> the builder, through rs_match.read_pairs(split="train",
    exclude_mirror=False): the rows rs_match.load_pairs reads (a row of another split is passed
    over before its payload is parsed)."""
    b = builder if builder is not None else PriorBuilder()
    b.add_set(_rs_match().read_pairs(rows, names, TRAIN, exclude_mirror=False, modes=modes))
    return b


def build_from_corpus(corpus: str | Path | None = None, *, parts: int = 0,
                      names: Mapping[str, str] | None = None,
                      modes: Iterable[str] | None = deck_stats.MODES,
                      progress: bool = False) -> PriorBuilder:
    """build_from_rows over the corpus's replays parquet (default <data>/il_replay), one part at
    a time (only the counts are kept); a tag in two parts counts once, as load_pairs reads it."""
    corpus = Path(corpus) if corpus is not None else deck_stats.data_root() / "il_replay"
    files = deck_stats.corpus_parts(corpus, parts)
    if not files:
        raise FileNotFoundError(f"no IL_Replay parquet in {corpus / 'replays'} "
                                "(run tools/il_replay_download.py)")
    if names is None:
        import rs_cards
        names = rs_cards.load()
    b = PriorBuilder()
    t0 = time.time()
    for done, part in enumerate(files, 1):
        build_from_rows(deck_stats.iter_rows([part]), names, modes=modes, builder=b)
        b.parts.append(part.name)
        if progress and (done % 5 == 0 or done == len(files)):
            print(f"  {done}/{len(files)} parts, {b.pairs:,} train pairs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return b


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _shown(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _build(args: argparse.Namespace) -> int:
    if args.max_decks < 0 or args.min_count < 1 or args.parts < 0:
        print("opp_model: --max-decks >= 0 (0: all), --min-count >= 1, --parts >= 0",
              file=sys.stderr)
        return 2
    import rs_cards
    names = rs_cards.load()
    t0 = time.time()
    try:
        builder = build_from_corpus(args.corpus, parts=args.parts, names=names, progress=True)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"opp_model: {exc}", file=sys.stderr)
        return 1
    meta = {"modes": sorted(deck_stats.MODES), "card_map": rs_cards.describe(names)}
    prior = builder.prior(max_decks=args.max_decks, min_count=args.min_count, meta=meta,
                          created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out = prior.save(args.out)
    size = out.stat().st_size
    print(prior.describe(decks=args.show))
    print(f"read {builder.pairs:,} train pairs ({builder.other_splits:,} rows of other splits "
          f"passed over, skipped {dict(builder.skipped)}) in {time.time() - t0:.0f}s")
    print(f"wrote {_shown(out)} ({size / 1024:.0f} KB): anonymous (decks and counts only)")
    if size > COMMIT_BYTES:
        print(f"warning: {size / 1e6:.1f} MB is too large to commit; rebuild with the default "
              f"--max-decks {MAX_DECKS} --min-count {MIN_COUNT}", file=sys.stderr)
    else:
        print(f"commit it: git add {_shown(out)}")
    return 0


def _show(args: argparse.Namespace) -> int:
    try:
        prior = load_prior(args.prior, required=True)
    except (FileNotFoundError, DeckPriorError) as exc:
        print(f"opp_model: {exc}", file=sys.stderr)
        return 1
    print(prior.describe(decks=args.decks))
    known = [c.strip() for c in (args.known or "").split(",") if c.strip()]
    if not known and args.n is None:
        return 0
    odd = [c for c in known if c not in prior]
    if odd:
        print(f"  not in the prior: {', '.join(odd)}")
    n = args.n if args.n is not None else max(0, DECK_SIZE - len(set(known)))
    print(f"known {', '.join(known) or '(nothing)'}: most likely {n}: "
          f"{', '.join(prior.top(known, n)) or '-'}")
    probs = sorted(prior.probabilities(known).items(), key=lambda kv: -kv[1])[:8]
    print("  first draw: " + ", ".join(f"{c} {100 * p:.1f}%" for c, p in probs))
    rng = np.random.default_rng(args.seed)
    for k in range(args.samples):
        print(f"  sample {k + 1}: {', '.join(prior.sample(known, n, rng)) or '-'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The opponent deck prior from the IL_Replay train "
                                             "split (docs/LIVE_SEARCH_PLAN.md S03).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-prior", help="count the train split's decks into the prior file")
    b.add_argument("--split", default=TRAIN, choices=[TRAIN],
                   help="always train: the gate's dev pairings and the test split never inform it")
    b.add_argument("--corpus", type=Path, default=None,
                   help="IL_Replay folder holding replays/ (default <data>/il_replay)")
    b.add_argument("--parts", type=int, default=0, help="only the first N parts (0: all)")
    b.add_argument("--max-decks", type=int, default=MAX_DECKS,
                   help=f"decks kept whole (default {MAX_DECKS}; 0: all, a larger file)")
    b.add_argument("--min-count", type=int, default=MIN_COUNT,
                   help=f"keep decks of at least this many seats (default {MIN_COUNT})")
    b.add_argument("--out", type=Path, default=DEFAULT_PATH,
                   help="default data/opp_prior/deck_prior.json")
    b.add_argument("--show", type=int, default=10, help="decks printed")
    s = sub.add_parser("show", help="print a prior and draws for a known set (read-only)")
    s.add_argument("--prior", type=Path, default=DEFAULT_PATH)
    s.add_argument("--known", default="", help="comma-separated RoyaleSim card names")
    s.add_argument("--n", type=int, default=None, help="cards to draw (default 8 - known)")
    s.add_argument("--samples", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--decks", type=int, default=10, help="decks printed")
    args = ap.parse_args(argv)
    return _build(args) if args.cmd == "build-prior" else _show(args)


if __name__ == "__main__":
    sys.exit(main())
