r"""The game-version ledger, data/game_versions.json (U04 of docs/SEASON_UPDATES.md), and the
local files the daily check keeps next to it. Standard library only, and it imports no other
tool, so selfcheck, the live runner and tools/rs_engine.py can all read it.

    python tools\game_versions.py                   the pin, the clients, the season, the events
    python tools\game_versions.py --season 2026     that year's season starts
    python tools\game_versions.py --client 160635001
    python tools\game_versions.py --currency        the currency line from the last check

What the ledger holds (load() checks it):
  pin             the royalesim release and cards.CARD_TABLE arm every tool runs
  royalesim       per release: the ledger arms, the engine's default arm, and per arm the table
                  version, client build, content version and FNV-1a 64 (from the sdist)
  royaleimitate   the release data/cards/replay_cards.json was vendored from, and its sha256
  clients         client builds (versionCode) with the date each was accepted and why
  seasons         the rule (a season starts on the first Monday of each month), its anchor
                  (Season 89 on 2026-11-02) and the 2026 starts it must reproduce
  content_events  dated balance updates, new cards and new forms, with the arm that has them
  il_replay       the Hugging Face dataset and the revision recorded for it

Client builds. A versionCode is MM mmmm bbb: 160635001 is 16.635.1 (major 16, minor 635, build
1). The client line is (major, minor). A table's line comes from its content version: 16.402.22
is line 16.402. A client on a newer line than the pinned table is "ahead" (accepted on
2026-10-29: 16.635 against 16.402).

Local files, under <runs>/updates/ (runs root: src/hastycr/machine.py; runs/ is gitignored):
  last_seen_client.json  the build selfcheck or the live runner last read (note_client, U05)
  clients.json           builds the daily check accepted by itself (basis auto); the ledger
                         itself is only edited by a commit (promote and accept-client, U12)
  status.json, .txt      what the last check found (tools/game_update.py check)
  state.json             the check's memory between runs (last check, revisions seen)
currency_line(status) renders the one summary selfcheck and the live runner print (4.1).

Exit codes of the CLI: 0 ok; 1 the ledger cannot be read or fails its checks; 2 bad arguments.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "game_versions.json"
SCHEMA = 1
TOOL = "tools/game_versions.py"

#: Season numbering: Season 89 starts on Monday 2026-11-02 (docs/SEASON_UPDATES.md 2, 3.1).
ANCHOR_SEASON = 89
ANCHOR_YEAR, ANCHOR_MONTH = 2026, 11

#: Content event kinds. "season" is a season start; the others are what the ledger lists.
EVENT_KINDS = ("season", "balance", "card", "form", "hero", "evolution", "client", "other")
#: Kinds that change what the simulator has to know (a client build alone does not).
CONTENT_KINDS = frozenset({"season", "balance", "card", "form", "hero", "evolution"})
#: Client entry bases.
BASES = ("table", "owner", "auto", "history")

UPDATES = "updates"
LAST_SEEN = "last_seen_client.json"
LOCAL_CLIENTS = "clients.json"
STATUS_JSON = "status.json"
STATUS_TXT = "status.txt"
STATE_JSON = "state.json"

CODE_MIN, CODE_MAX = 10_000_000, 9_999_999_999
_CONTENT = re.compile(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?\s*$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FNV = re.compile(r"^[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

LINE_WIDTH = 100
LINE_HEAD = "currency  "


class LedgerError(ValueError):
    """data/game_versions.json is missing, not JSON, or fails a check."""


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------

def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_time(value: Any) -> dt.datetime | None:
    """An ISO time ("2026-10-23T04:16:28Z", with or without an offset; a bare date is its
    midnight UTC) as an aware UTC datetime, or None."""
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day, tzinfo=dt.timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        got = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if got.tzinfo is None:
        got = got.replace(tzinfo=dt.timezone.utc)
    return got.astimezone(dt.timezone.utc)


def parse_date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.astimezone(dt.timezone.utc).date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str) and _DATE.match(value.strip()):
        try:
            return dt.date.fromisoformat(value.strip())
        except ValueError:
            return None
    got = parse_time(value)
    return got.date() if got else None


def iso(value: dt.datetime) -> str:
    """UTC, seconds, with a Z."""
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


def age_text(seconds: float | None) -> str:
    """"just now", "12 min ago", "6 h ago", "3 days ago"; "at an unknown time" for None."""
    if seconds is None:
        return "at an unknown time"
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return "just now"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} min ago"
    if seconds < 48 * 3600:
        return f"{round(seconds / 3600)} h ago"
    return f"{round(seconds / 86400)} days ago"


# ---------------------------------------------------------------------------
# seasons
# ---------------------------------------------------------------------------

def first_monday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(0 - first.weekday()) % 7)


def season_number(year: int, month: int) -> int:
    """The season that starts in that month (Season 89 is 2026-11)."""
    return ANCHOR_SEASON + (year - ANCHOR_YEAR) * 12 + (month - ANCHOR_MONTH)


def season_month(number: int) -> tuple[int, int]:
    months = (ANCHOR_YEAR * 12 + ANCHOR_MONTH - 1) + (int(number) - ANCHOR_SEASON)
    return months // 12, months % 12 + 1


def season_start(number: int) -> dt.date:
    return first_monday(*season_month(number))


def season_starts(year: int) -> list[dt.date]:
    """The twelve season starts of `year`, in order."""
    return [first_monday(year, m) for m in range(1, 13)]


def season_on(day: dt.date) -> tuple[int, dt.date]:
    """(number, start) of the season running on `day`."""
    start = first_monday(day.year, day.month)
    if day >= start:
        return season_number(day.year, day.month), start
    year, month = (day.year, day.month - 1) if day.month > 1 else (day.year - 1, 12)
    return season_number(year, month), first_monday(year, month)


def next_season(day: dt.date) -> tuple[int, dt.date]:
    """(number, start) of the first season that starts after `day`."""
    number, _ = season_on(day)
    return number + 1, season_start(number + 1)


def seasons_between(after: dt.date, upto: dt.date) -> list[tuple[int, dt.date]]:
    """Season starts with after < start <= upto."""
    out = []
    number, start = season_on(after)
    while True:
        number += 1
        start = season_start(number)
        if start > upto:
            return out
        if start > after:
            out.append((number, start))


# ---------------------------------------------------------------------------
# client builds and content versions
# ---------------------------------------------------------------------------

def parse_code(code: Any) -> tuple[int, int, int]:
    """versionCode -> (major, minor, build): 160635001 -> (16, 635, 1). ValueError otherwise."""
    if isinstance(code, bool):
        raise ValueError(f"{code!r} is not a versionCode")
    try:
        value = int(str(code).strip()) if not isinstance(code, int) else code
    except ValueError:
        raise ValueError(f"{code!r} is not a versionCode") from None
    if not CODE_MIN <= value <= CODE_MAX:
        raise ValueError(f"{code!r} is not a versionCode (MM mmmm bbb, such as 160635001)")
    return value // 10_000_000, (value // 1000) % 10_000, value % 1000


def version_name(code: Any) -> str:
    """160635001 -> "16.635.1"."""
    return "{}.{}.{}".format(*parse_code(code))


def client_line(code: Any) -> tuple[int, int]:
    major, minor, _ = parse_code(code)
    return major, minor


def parse_content(text: Any) -> tuple[int, int, int] | None:
    """"16.402.22" -> (16, 402, 22); "16.402" -> (16, 402, 0); None otherwise."""
    m = _CONTENT.match(str(text)) if text is not None else None
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def content_line(text: Any) -> tuple[int, int] | None:
    got = parse_content(text)
    return got[:2] if got else None


def relation(code: Any, content: Any) -> str | None:
    """The client's line against a table's content line: "ahead", "same", "behind" (None when
    either cannot be read)."""
    try:
        mine = client_line(code)
    except ValueError:
        return None
    theirs = content_line(content)
    if theirs is None:
        return None
    return "ahead" if mine > theirs else "same" if mine == theirs else "behind"


# ---------------------------------------------------------------------------
# reading the ledger
# ---------------------------------------------------------------------------

def read_json(path: str | os.PathLike) -> Any:
    """A JSON file, or None when it is missing or unreadable (local state files only)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def write_json(path: str | os.PathLike, data: Any) -> Path:
    """Write through a temporary file and os.replace, so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_text(path: str | os.PathLike, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _problems(data: Mapping[str, Any]) -> list[str]:
    bad: list[str] = []
    if data.get("schema") != SCHEMA:
        bad.append(f"schema is {data.get('schema')!r}, this reader knows {SCHEMA}")
    pin = data.get("pin")
    releases = data.get("royalesim")
    if not isinstance(pin, Mapping) or not isinstance(releases, Mapping):
        return bad + ["pin and royalesim must be objects"]
    version, arm = pin.get("royalesim"), pin.get("card_table")
    entry = releases.get(version) if isinstance(version, str) else None
    if not isinstance(entry, Mapping):
        bad.append(f"pin.royalesim {version!r} has no royalesim entry")
    else:
        tables = entry.get("tables")
        if not isinstance(tables, Mapping) or arm not in tables:
            bad.append(f"pin.card_table {arm!r} is not a table of royalesim {version}")
        arms = entry.get("arms")
        if isinstance(arms, list) and arm not in arms:
            bad.append(f"pin.card_table {arm!r} is not an arm of royalesim {version}")
    for ver, rel in releases.items():
        if not isinstance(rel, Mapping):
            bad.append(f"royalesim.{ver} is not an object")
            continue
        sha = rel.get("sdist_sha256")
        if sha is not None and not _SHA256.match(str(sha)):
            bad.append(f"royalesim.{ver}.sdist_sha256 is not a sha256")
        for name, table in (rel.get("tables") or {}).items():
            if not isinstance(table, Mapping):
                bad.append(f"royalesim.{ver}.tables.{name} is not an object")
                continue
            fnv = table.get("fnv1a64")
            if fnv is not None and not _FNV.match(str(fnv)):
                bad.append(f"royalesim.{ver}.tables.{name}.fnv1a64 {fnv!r} is not 16 hex digits")
            if table.get("content") is not None and parse_content(table["content"]) is None:
                bad.append(f"royalesim.{ver}.tables.{name}.content {table['content']!r}")
            if table.get("client_build") is not None:
                try:
                    parse_code(table["client_build"])
                except ValueError as exc:
                    bad.append(f"royalesim.{ver}.tables.{name}.client_build: {exc}")
            if table.get("content_date") is not None and parse_date(table["content_date"]) is None:
                bad.append(f"royalesim.{ver}.tables.{name}.content_date {table['content_date']!r}")
    seen: set[int] = set()
    for k, client in enumerate(data.get("clients") or []):
        where = f"clients[{k}]"
        if not isinstance(client, Mapping):
            bad.append(f"{where} is not an object")
            continue
        try:
            code = int(client.get("version_code"))
            name = version_name(code)
        except (TypeError, ValueError):
            bad.append(f"{where}.version_code {client.get('version_code')!r} is not a versionCode")
            continue
        if code in seen:
            bad.append(f"{where}: versionCode {code} is listed twice")
        seen.add(code)
        if client.get("version_name") not in (None, name):
            bad.append(f"{where}: version_name {client.get('version_name')!r} is not {name} "
                       f"(the encoding of {code})")
        if client.get("basis") not in (None, *BASES):
            bad.append(f"{where}.basis {client.get('basis')!r} is not one of {', '.join(BASES)}")
        if client.get("accepted") is not None and parse_date(client["accepted"]) is None:
            bad.append(f"{where}.accepted {client['accepted']!r} is not a date")
    for k, event in enumerate(data.get("content_events") or []):
        where = f"content_events[{k}]"
        if not isinstance(event, Mapping) or parse_date(event.get("date")) is None:
            bad.append(f"{where} has no date")
            continue
        if event.get("kind") not in EVENT_KINDS:
            bad.append(f"{where}.kind {event.get('kind')!r} is not one of {', '.join(EVENT_KINDS)}")
    seasons = data.get("seasons")
    if isinstance(seasons, Mapping):
        anchor = seasons.get("anchor") or {}
        start = parse_date(anchor.get("start"))
        if start is None or season_start(int(anchor.get("season", 0) or 0)) != start:
            bad.append(f"seasons.anchor {anchor!r} does not follow the rule (Season "
                       f"{ANCHOR_SEASON} on {season_start(ANCHOR_SEASON)})")
    il = data.get("il_replay")
    if il is not None and (not isinstance(il, Mapping) or not il.get("repo")):
        bad.append("il_replay needs a repo")
    return bad


def load(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """The ledger, after its checks (LedgerError names every problem)."""
    path = Path(path) if path is not None else LEDGER_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise LedgerError(f"{path} is missing") from None
    except OSError as exc:
        raise LedgerError(f"{path}: cannot read ({exc.strerror or exc})") from None
    except ValueError as exc:
        raise LedgerError(f"{path} is not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise LedgerError(f"{path} must hold one JSON object")
    bad = _problems(data)
    if bad:
        raise LedgerError(f"{path}: " + "; ".join(bad))
    return data


def pin(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """{"royalesim", "card_table", "since", "table": the pinned arm's table entry, "release":
    the pinned release's entry}."""
    p = ledger["pin"]
    release = ledger["royalesim"][p["royalesim"]]
    return {"royalesim": p["royalesim"], "card_table": p["card_table"], "since": p.get("since"),
            "table": dict(release["tables"][p["card_table"]]), "release": release}


def release(ledger: Mapping[str, Any], version: str) -> dict[str, Any] | None:
    got = (ledger.get("royalesim") or {}).get(version)
    return dict(got) if isinstance(got, Mapping) else None


# ---------------------------------------------------------------------------
# local files
# ---------------------------------------------------------------------------

def _machine():
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from hastycr import machine
    return machine


def updates_dir(runs: str | os.PathLike | None = None) -> Path:
    """<runs>/updates: `runs` when given, else the machine's runs root (runs/ when
    src/hastycr/machine.py cannot be read)."""
    if runs is not None:
        return Path(runs) / UPDATES
    try:
        return Path(_machine().runs_root()) / UPDATES
    except Exception:                       # a broken machine.json: the repo's runs/
        return ROOT / "runs" / UPDATES


def data_root() -> Path:
    try:
        return Path(_machine().data_root())
    except Exception:
        return ROOT / "data"


def note_client(code: Any, name: str | None = None, *, source: str,
                updates: str | os.PathLike | None = None,
                now: dt.datetime | None = None) -> dict[str, Any]:
    """Record the client build a device read showed (selfcheck, the live runner: U05) in
    <updates>/last_seen_client.json. Returns the record. ValueError for a bad versionCode."""
    major, minor, build = parse_code(code)
    value = int(str(code).strip()) if not isinstance(code, int) else code
    record = {"version_code": value, "version_name": name or f"{major}.{minor}.{build}",
              "seen_at": iso(now or utcnow()), "source": str(source)}
    folder = Path(updates) if updates is not None else updates_dir()
    write_json(folder / LAST_SEEN, record)
    return record


def last_seen_client(updates: str | os.PathLike | None = None) -> dict[str, Any] | None:
    folder = Path(updates) if updates is not None else updates_dir()
    got = read_json(folder / LAST_SEEN)
    if not isinstance(got, dict):
        return None
    try:
        parse_code(got.get("version_code"))
    except ValueError:
        return None
    return got


def local_clients(updates: str | os.PathLike | None = None) -> list[dict[str, Any]]:
    folder = Path(updates) if updates is not None else updates_dir()
    got = read_json(folder / LOCAL_CLIENTS)
    items = got.get("clients") if isinstance(got, dict) else None
    out = []
    for item in items if isinstance(items, list) else []:
        try:
            if isinstance(item, dict):
                parse_code(item.get("version_code"))
                out.append(item)
        except ValueError:
            continue
    return out


def record_local_client(entry: Mapping[str, Any],
                        updates: str | os.PathLike | None = None) -> Path:
    """Append (or replace by versionCode) an accepted build in <updates>/clients.json."""
    folder = Path(updates) if updates is not None else updates_dir()
    code = int(entry["version_code"])
    parse_code(code)
    kept = [c for c in local_clients(folder) if int(c["version_code"]) != code]
    kept.append(dict(entry))
    return write_json(folder / LOCAL_CLIENTS, {
        "_comment": ["Client builds the daily check accepted by itself (basis auto) or the "
                     "owner accepted on this PC; tools/game_versions.py merges them with "
                     "data/game_versions.json clients. Local, not committed."],
        "clients": sorted(kept, key=lambda c: int(c["version_code"]))})


def clients(ledger: Mapping[str, Any],
            updates: str | os.PathLike | None = None) -> list[dict[str, Any]]:
    """The ledger's clients plus the local ones (a ledger entry wins over a local one for the
    same versionCode), by versionCode."""
    out = {int(c["version_code"]): dict(c, where="ledger") for c in ledger.get("clients") or []}
    for c in local_clients(updates):
        out.setdefault(int(c["version_code"]), dict(c, where="local"))
    return [out[k] for k in sorted(out)]


def client_entry(ledger: Mapping[str, Any], code: Any,
                 updates: str | os.PathLike | None = None) -> dict[str, Any] | None:
    value = int(str(code).strip()) if not isinstance(code, int) else code
    for c in clients(ledger, updates):
        if int(c["version_code"]) == value:
            return c
    return None


def newest_accepted(ledger: Mapping[str, Any],
                    updates: str | os.PathLike | None = None) -> dict[str, Any] | None:
    got = [c for c in clients(ledger, updates) if c.get("accepted")]
    return got[-1] if got else None


def judge_client(ledger: Mapping[str, Any], code: Any,
                 updates: str | os.PathLike | None = None) -> dict[str, Any]:
    """What the ledger says about client build `code` against the pinned table:
    {"version_code", "version_name", "known", "accepted", "basis", "relation", "table_content",
    "verdict": "ok" (accepted) | "new" (not in the ledger) | "unaccepted" (listed, never
    accepted)}. ValueError for a bad versionCode."""
    name = version_name(code)
    value = int(str(code).strip()) if not isinstance(code, int) else code
    table = pin(ledger)["table"]
    entry = client_entry(ledger, value, updates)
    accepted = entry.get("accepted") if entry else None
    return {"version_code": value, "version_name": name, "known": entry is not None,
            "accepted": accepted, "basis": entry.get("basis") if entry else None,
            "where": entry.get("where") if entry else None,
            "relation": relation(value, table.get("content")),
            "table_content": table.get("content"),
            "verdict": "ok" if accepted else "unaccepted" if entry else "new"}


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

def events(ledger: Mapping[str, Any], *, after: dt.date | None = None,
           upto: dt.date | None = None) -> list[dict[str, Any]]:
    """The ledger's content events plus every season start (by the rule) with
    after < date <= upto, by date. A season start the ledger already lists (kind "season" with
    that date) is not added twice. Each item has "date" as a datetime.date."""
    lo = after or dt.date.min
    hi = upto or dt.date.max
    out: list[dict[str, Any]] = []
    listed_seasons = set()
    for event in ledger.get("content_events") or []:
        day = parse_date(event.get("date"))
        if day is None:
            continue
        if event.get("kind") == "season":
            listed_seasons.add(day)
        if lo < day <= hi:
            out.append(dict(event, date=day))
    if after is not None and upto is not None:
        for number, start in seasons_between(after, upto):
            if start not in listed_seasons:
                out.append({"date": start, "kind": "season", "season": number,
                            "what": f"Season {number} starts", "table": None,
                            "from": "the season rule"})
    return sorted(out, key=lambda e: (e["date"], EVENT_KINDS.index(e.get("kind", "other"))
                                      if e.get("kind") in EVENT_KINDS else 99))


def covered(event: Mapping[str, Any], release_entry: Mapping[str, Any] | None) -> bool:
    """Whether an event's content is in a table of that royalesim release (its "table" arm)."""
    arm = event.get("table")
    tables = (release_entry or {}).get("tables") or {}
    return bool(arm) and arm in tables


def behind(ledger: Mapping[str, Any], today: dt.date) -> dict[str, Any] | None:
    """docs/SEASON_UPDATES.md 10, "days behind": the content events (season starts included)
    after the pinned table's content date, up to `today`, that no table of the pinned release
    covers. None when there are none. {"since": date, "days", "events"}."""
    p = pin(ledger)
    since = parse_date(p["table"].get("content_date")) or parse_date(p.get("since")) or today
    missed = [e for e in events(ledger, after=since, upto=today)
              if e.get("kind") in CONTENT_KINDS and not covered(e, p["release"])]
    if not missed:
        return None
    first = missed[0]["date"]
    return {"since": first, "days": (today - first).days, "events": missed}


def event_text(event: Mapping[str, Any]) -> str:
    what = str(event.get("what") or event.get("kind") or "event")
    return what


# ---------------------------------------------------------------------------
# status and the currency line
# ---------------------------------------------------------------------------

def read_status(updates: str | os.PathLike | None = None) -> dict[str, Any] | None:
    folder = Path(updates) if updates is not None else updates_dir()
    got = read_json(folder / STATUS_JSON)
    return got if isinstance(got, dict) else None


def _ago(now: dt.datetime, when: Any) -> str:
    got = parse_time(when)
    return age_text((now - got).total_seconds() if got else None)


def currency_parts(status: Mapping[str, Any], now: dt.datetime | None = None) -> list[str]:
    """The pieces of the currency line (docs/SEASON_UPDATES.md 4.1), in order."""
    now = now or utcnow()
    parts: list[str] = []
    pin_ = status.get("pin") or {}
    table = pin_.get("content") or pin_.get("card_table") or "?"
    client = status.get("client") or {}
    if client.get("version_code"):
        rel = client.get("relation")
        text = {"ahead": "client ahead", "same": "same line",
                "behind": "client behind the table"}.get(rel, "line unknown")
        if client.get("accepted"):
            text += f", accepted {client['accepted']}"
            if client.get("basis") == "auto":
                text += " (auto)"
        else:
            text += ", NOT ACCEPTED"
        if not client.get("seen"):
            text += ", not read from the device yet"
        parts.append(f"client {client['version_code']} ({client.get('version_name')}) "
                     f"vs table {table}: {text}")
    else:
        parts.append(f"client not read from the device yet vs table {table}")
    rs = status.get("royalesim") or {}
    pinned = pin_.get("royalesim") or "?"
    newer = [i for i in rs.get("newer") or [] if i.get("integrity") is not False]
    if not rs or rs.get("skipped"):
        parts.append(f"royalesim {pinned} = pin; PyPI not checked")
    elif rs.get("error"):
        parts.append(f"royalesim {pinned} = pin; PyPI check failed")
    elif newer:
        parts.append(f"royalesim {pinned} = pin; {newer[-1].get('version')} on PyPI: "
                     "UPDATE AVAILABLE")
    elif rs.get("newer"):
        parts.append(f"royalesim {pinned} = pin; {rs['newer'][-1].get('version')} on PyPI "
                     "failed its sha256 check")
    else:
        parts.append(f"royalesim {pinned} = pin, newest on PyPI")
    parts.append(f"checked {_ago(now, status.get('checked_at'))}")
    pool = status.get("pool") or {}
    parts.append(f"pool pulled {_ago(now, pool['pulled_at'])}" if pool.get("pulled_at")
                 else "pool not pulled yet")
    cal = status.get("calendar") or {}
    late = cal.get("behind")
    if late and late.get("events"):
        first = late["events"][0]
        label = (f"Season {first['season']} started" if first.get("kind") == "season"
                 and first.get("season") else f"{event_text(first)} on")
        tail = ("run scripts\\game_update.ps1 -Update" if newer else
                "run scripts\\game_update.ps1 -Update when RoyaleSim ships it")
        parts.append(f"{label} {first['date']}: {tail}")
    else:
        for event in cal.get("upcoming") or []:
            if event.get("kind") == "season" and event.get("season"):
                parts.append(f"Season {event['season']} starts {event['date']}")
            else:
                parts.append(f"{event_text(event)} on {event['date']}")
            break
    gaps = (status.get("radar") or {}).get("gaps") or []
    if gaps:
        names = [f"{g.get('engine') or g.get('name')} {g.get('form')}" for g in gaps[:4]]
        more = f" and {len(gaps) - 4} more" if len(gaps) > 4 else ""
        parts.append("in the game, not in the table: " + ", ".join(names) + more)
    return parts


def wrap(parts: Sequence[str], width: int = LINE_WIDTH, head: str = LINE_HEAD) -> list[str]:
    """Join parts with "; " into lines of at most `width`, the first after `head`, the others
    indented under it."""
    pad = " " * len(head)
    lines: list[str] = []
    cur, empty = head, True
    for k, part in enumerate(parts):
        piece = part + (";" if k < len(parts) - 1 else "")
        if not empty and len(cur) + 1 + len(piece) > width:
            lines.append(cur)
            cur = pad + piece
        else:
            cur = cur + ("" if empty else " ") + piece
        empty = False
    lines.append(cur.rstrip())
    return lines


def currency_line(status: Mapping[str, Any] | None, now: dt.datetime | None = None) -> str:
    """The currency summary of a status (read_status()), wrapped; a note when there is none."""
    if not status:
        return (LINE_HEAD + "no update check has run yet: run "
                "tools\\game_update.py check --offline (scripts\\game_update.ps1 -Daily)")
    return "\n".join(wrap(currency_parts(status, now)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _describe(ledger: Mapping[str, Any], updates: Path, today: dt.date) -> list[str]:
    p = pin(ledger)
    t = p["table"]
    out = [f"pin        royalesim {p['royalesim']}, {p['card_table']} ({t.get('version')}, "
           f"content {t.get('content')}, client build {t.get('client_build')}, "
           f"fnv {t.get('fnv1a64')}), since {p.get('since')}"]
    for c in clients(ledger, updates):
        acc = f"accepted {c['accepted']} ({c.get('basis')})" if c.get("accepted") else \
            f"not accepted ({c.get('basis') or 'unknown'})"
        out.append(f"client     {c['version_code']} {version_name(c['version_code']):>10}  "
                   f"{relation(c['version_code'], t.get('content'))}  {acc}"
                   + ("  [local]" if c.get("where") == "local" else ""))
    number, start = season_on(today)
    nxt, nstart = next_season(today)
    out.append(f"season     {number} since {start}; Season {nxt} starts {nstart}")
    late = behind(ledger, today)
    if late:
        out.append(f"behind     {late['days']} days, since {late['since']}: "
                   + "; ".join(event_text(e) for e in late["events"]))
    for e in events(ledger, after=today, upto=today + dt.timedelta(days=31)):
        out.append(f"upcoming   {e['date']}  {event_text(e)}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=None, help="another ledger file")
    ap.add_argument("--updates", type=Path, default=None,
                    help="the local folder (default <runs>/updates)")
    ap.add_argument("--season", type=int, metavar="YEAR", help="print that year's season starts")
    ap.add_argument("--client", metavar="CODE", help="judge one client versionCode")
    ap.add_argument("--currency", action="store_true", help="the currency line of the last check")
    args = ap.parse_args(argv)
    if args.season is not None:
        if not 2000 <= args.season <= 2100:
            ap.error(f"--season {args.season}: a year, such as 2026")
        for start in season_starts(args.season):
            print(f"Season {season_number(start.year, start.month)}  {start}")
        return 0
    updates = args.updates if args.updates is not None else updates_dir()
    if args.currency:
        print(currency_line(read_status(updates)))
        return 0
    try:
        ledger = load(args.ledger)
    except LedgerError as exc:
        print(f"ledger: {exc}", file=sys.stderr)
        return 1
    if args.client is not None:
        try:
            got = judge_client(ledger, args.client, updates)
        except ValueError as exc:
            ap.error(f"--client: {exc}")
        print(json.dumps(got, indent=1))
        return 0
    print("\n".join(_describe(ledger, updates, utcnow().date())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
