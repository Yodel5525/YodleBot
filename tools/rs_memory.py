r"""Public memory dated at execution: the cards a seat can know the other has played.

    from rs_memory import BOARD_LAG, PublicMemory     # tools/ on sys.path, as the tools import
    mem = PublicMemory()                               # one per match
    mem.record(team, card + 2, exec_tick)              # a play, dated on the tick it RUNS
    row = obs_row(state, team, sub, mem.view(tick, BOARD_LAG), tick)

Why (tokens-04, tokens-17). tools/rs_rows.py and tools/rs_eval.py keep a dict
{"seen": {team: set}, "last": {team: [(card, tick)]}} and append to it when a play is SENT.
Under a command delay the play runs `delay` ticks later (RoyaleSim: "a play accepted on tick T
runs on T + k"), so each seat saw the other's card in opp_seen / opp_last up to 21 ticks before
it existed. And a seat's row built on the tick the other seat played already showed that play.
Here every play is dated when it runs, and a view only holds the plays a seat could know by then.

THE RULE. view(tick, lag) holds exactly the plays with

    exec_tick + lag <= tick

for both teams, in the shape obs_row reads (obs_row reads only the opponent's half):

    {"seen": {0: set, 1: set},                 cards with at least one such play
     "last": {0: [(card, exec_tick), ...], 1: [...]}}   such plays, oldest first

`card` is the row token, catalogue id + 2, as rs_rows stores it (0 is padding and refused;
1, "unknown", is allowed for a play perception could not name). "last" is ordered by
exec_tick, ties in the order they were recorded, so plays recorded out of order (a send dated
`delay` ahead next to one dated now) come out in the order they ran. The ages obs_row derives,
(row tick - exec_tick) / 400, count from the tick the card went down. Every view is a fresh
copy: changing it changes nothing here.

THE ENGINE'S PHASE. A command that runs at tick E runs inside the step FROM E: the state_json
taken at E (before that step) does not show it yet, the state at E + 1 does (tools/rs_elixir.py
dates elixir the same way). So a row built from the state at tick t, which should know only
what its board shows, wants the plays with E < t:

    lag = BOARD_LAG (1)   order-free: any play recorded with exec_tick == t is absent at t.
    lag = 0               the rule above taken literally; a play that runs at t is in the view
                          at t, so the caller builds a tick's rows BEFORE it records that tick's
                          plays (with that order, lag 0 and lag 1 give the same rows).
    lag = 1 + L           perception that needs L more ticks to report a play (live vision).

Legacy rows are lag 0 with the old order (record each play as it is sent, take the view at each
row); the golden tests check that this reproduces the old dict exactly.

LIVE. Perception reports an opponent play when it first sees it; record it at that tick and
read view(now, 0): what the screen showed is by definition what the bot knows.

ENGINE RUN LIST. record_ran() takes the rows of Battle.step_commands_run(), "every delayed
command that ran or was dropped during the last step ... (tick, team, kind, what, reason)",
and records each deploy that ran (reason 0) at the tick it ran. That list is empty with no
delay, so undelayed callers record plays as they send them (they run at once).

Pure Python, no third-party import; the tools import it as `rs_memory`, tests as tools.rs_memory.
"""
from __future__ import annotations

import bisect
import operator
from collections.abc import Iterable, Sequence
from typing import Any

TEAMS: tuple[int, int] = (0, 1)
#: Row tokens: 0 pads opp_seen / opp_last, 1 is the unknown card, catalogue id c is c + 2.
PAD, UNKNOWN, TOKEN_OFFSET = 0, 1, 2
#: The smallest lag at which a view at t holds only plays the state at t shows (see above).
BOARD_LAG = 1
#: Battle.step_commands_run(): the kind of a card play, and the reason of a command that ran.
RAN_DEPLOY, RAN_OK = "deploy", 0


def _int(value: Any, what: str) -> int:
    """`value` as a plain int (numpy integers included); bools and non-integers are refused."""
    if isinstance(value, bool):
        raise TypeError(f"{what} must be an int, not a bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{what} must be an int, got {type(value).__name__}") from None


def _team(value: Any) -> int:
    team = _int(value, "team")
    if team not in TEAMS:
        raise ValueError(f"team must be 0 or 1, got {team}")
    return team


def empty_view() -> dict:
    """The view of a memory with no plays: the dict rs_rows and rs_eval start a match from."""
    return {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}


class PublicMemory:
    """The plays of one match, each dated on the tick it ran; view() filters them by tick."""

    __slots__ = ("_ticks", "_last", "_first")

    def __init__(self) -> None:
        self._ticks: dict[int, list[int]] = {0: [], 1: []}               # exec ticks, sorted
        self._last: dict[int, list[tuple[int, int]]] = {0: [], 1: []}    # (card, exec), same order
        self._first: dict[int, dict[int, int]] = {0: {}, 1: {}}          # card -> earliest exec

    def record(self, team: int, card: int, exec_tick: int) -> None:
        """`team` played row token `card` (catalogue id + 2), which runs on `exec_tick`: the
        recorded tick for IL_Replay rows (taken to be its execution tick, an assumption
        tools/rs_dt_diag.py checks), send tick + command delay in a delayed battle, the tick
        perception first saw it live."""
        team = _team(team)
        card = _int(card, "card")
        exec_tick = _int(exec_tick, "exec_tick")
        if card <= PAD:
            raise ValueError(f"card must be a row token >= {UNKNOWN} (catalogue id + "
                             f"{TOKEN_OFFSET}), got {card}")
        if exec_tick < 0:
            raise ValueError(f"exec_tick must be >= 0, got {exec_tick}")
        ticks = self._ticks[team]
        k = bisect.bisect_right(ticks, exec_tick)       # after equal ticks: record order
        ticks.insert(k, exec_tick)
        self._last[team].insert(k, (card, exec_tick))
        first = self._first[team]
        if exec_tick < first.get(card, exec_tick + 1):
            first[card] = exec_tick

    def record_ran(self, rows: Iterable[Sequence]) -> int:
        """Record the card plays in Battle.step_commands_run() rows, (tick, team, kind, what,
        reason): kind "deploy" with reason 0 is catalogue card `what` that ran on `tick`.
        Ability presses and commands refused or dropped when they ran are skipped. Returns how
        many plays were recorded."""
        n = 0
        for row in rows:
            tick, team, kind, what, reason = tuple(row)[:5]
            if kind != RAN_DEPLOY or _int(reason, "reason") != RAN_OK:
                continue
            self.record(team, _int(what, "what") + TOKEN_OFFSET, tick)
            n += 1
        return n

    def view(self, tick: int, lag: int = 0) -> dict:
        """The plays with exec_tick + lag <= tick, as {"seen": {team: set of cards}, "last":
        {team: [(card, exec_tick)] oldest first}} for both teams: the `mem` obs_row reads."""
        tick = _int(tick, "tick")
        lag = _int(lag, "lag")
        if lag < 0:
            raise ValueError(f"lag must be >= 0 (a negative lag sees plays before they run), "
                             f"got {lag}")
        upto = tick - lag
        out: dict = {"seen": {}, "last": {}}
        for team in TEAMS:
            out["seen"][team] = {card for card, t in self._first[team].items() if t <= upto}
            out["last"][team] = self._last[team][:bisect.bisect_right(self._ticks[team], upto)]
        return out

    def plays(self, team: int) -> list[tuple[int, int]]:
        """Every play recorded for `team`, (card, exec_tick) oldest first (a copy)."""
        return list(self._last[_team(team)])

    def __len__(self) -> int:
        return len(self._ticks[0]) + len(self._ticks[1])

    def __repr__(self) -> str:
        return f"PublicMemory(plays={{0: {len(self._ticks[0])}, 1: {len(self._ticks[1])}}})"
