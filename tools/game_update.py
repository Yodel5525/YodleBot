r"""Keeping the bot current after a game update (docs/SEASON_UPDATES.md 4). This file has the
daily `check` (U04). probe and stage (U09), and promote, update, accept-client and rollback
(U12), come later.

    .venvs\royalegym\Scripts\python.exe tools\game_update.py check --offline
    .venvs\royalegym\Scripts\python.exe tools\game_update.py check --offline --json
    .venvs\royalegym\Scripts\python.exe tools\game_update.py check --offline --skip il_replay
    .venvs\royalegym\Scripts\python.exe tools\game_update.py check --cards cards.json --client 160635001

check makes HTTPS calls only: PyPI, the official Clash Royale API (one GET /v1/cards, through
tools/cr_api.py) and the Hugging Face API. It never calls adb, never starts MuMu, never taps and
reads no game memory. --offline means "no device", as in scripts/selfcheck.py: the client build
comes from <runs>/updates/last_seen_client.json, which selfcheck and the live runner write (U05,
tools/game_versions.note_client). check has no device step at all, so the flag only keeps the
daily command line the same as selfcheck's.

What check looks at (section 4.1, step 2):
  pypi           royalesim releases newer than the pin in data/game_versions.json (pre-releases
                 with --pre; a release whose files are all yanked is listed, not counted). For
                 each, newest last, up to --max-sdists: the sdist is downloaded into
                 <runs>/updates/cache/, checked against PyPI's sha256 (a mismatch is an error and
                 the file is not kept), and read without unpacking: data/calibration.json's
                 cards.CARD_TABLE arms and default, and every data/derived/cards-*.json table's
                 version, client build, content version, roster counts, forms and FNV-1a 64 of
                 its bytes (the value Battle.card_table_fnv1a64() reports; 3.3). Compared with
                 the pinned release in the ledger: new and removed arms, a changed default arm,
                 new tables, and same-named tables whose bytes changed. A release without an
                 sdist (0.1.10 had none) is listed as such: the stage probes its wheel.
  royaleimitate  releases newer than the one data/cards/replay_cards.json was vendored from: its
                 sdist (or wheel) is read the same way, and royaleimitate/replay_cards.json is
                 compared key by key with the vendored map (keys added, removed and renamed).
  radar          the day's /v1/cards against the previous check's snapshot and against the
                 engine's card forms (data/vision/classes.json, which tools/vision_classes.py
                 --engine writes from the pinned table): new card ids, removed ids, renames, new
                 maxEvolutionLevel bits (1 evolution, 2 hero) and every form the game has and
                 the table lacks ("Giant gained an evolution; the simulator lacks it"). A newer
                 release's table that has the form is named. API names are resolved as in 7.3:
                 cr_api.slug -> RoyaleAPI key -> tools/rs_cards.py map -> catalogue name, then
                 the official card id through data/royaleapi/cards.json, then an "auto" match
                 (the display name in CamelCase with the same elixir; Minion Giant). The snapshot
                 is <data>/opponents/cards_api.json when U03's pull wrote one in the last
                 CARDS_FRESH_H hours, else one call through tools/cr_api.py (imported when the
                 check runs; without it, or without a key, the radar is skipped with a note), or
                 --cards FILE.
  calendar       season starts (the first Monday of each month) and the ledger's content_events
                 since the last check (the last FIRST_WINDOW_DAYS on the first run), the ones in
                 the next UPCOMING_DAYS, and "days behind" (section 10): content events after the
                 pinned table's content date that no pinned table covers. From UPSTREAM_DAYS
                 behind with no newer royalesim, a warning says to ask upstream.
  device         the client build last seen (last_seen_client.json, or --client), judged against
                 the ledger's clients plus the ones the check accepted itself
                 (<runs>/updates/clients.json).
  il_replay      the dataset's latest revision (sha, lastModified) when the Hugging Face API
                 answers; a change from the ledger's or the last check's revision is reported.
  pool           when U03 last pulled the opponent pool (<data>/opponents/state.json).

A new client build is accepted by itself (basis auto, into <runs>/updates/clients.json) only on
the day a check first sees it (state.json "pending_client" keeps that day), and only when that
day's check found no newer royalesim, ran the radar against a previous snapshot and found no
roster or form change, and found no calendar event since the last check or today, and no
earlier check the same day found any of these (state.json "noisy_on"). Otherwise it stays a
warning and exit 11 until the owner accepts it (scripts\game_update.ps1 -AcceptClient -Note):
a later quiet day does not accept it (section 4.1, "the same day's check").

What it writes, all under <runs>/updates/ (runs/ is gitignored):
  status.json   everything above, the verdict, the warnings and the currency line
  status.txt    the currency line, then one WARN / INFO / ERROR line each, then the verdict
  state.json    the last check's time, revisions seen, the radar's snapshot name
  cards_api.last.json  the /v1/cards snapshot the radar compared (card data only)
  cache/        verified sdists and their parsed summaries (each release is read once)
Nothing it writes holds a player name, a tag or the API token.

Exit codes of check (section 4.1): 0 nothing to act on; 10 UPDATE AVAILABLE (a royalesim newer
than the pin: run scripts\game_update.ps1 -Update); 11 a client build the ledger has not
accepted, and no newer royalesim; 1 an error (the ledger, PyPI unreachable, a sha256 mismatch);
2 bad arguments. Radar, calendar, royaleimitate and IL_Replay findings are warnings in the status
and do not change the exit code.

Courtesy: every request sends USER_AGENT; one host gets at most one request per HOST_GAP
seconds; 429 and 503 back off (Retry-After when given, else 5, 10, 20 s), and a Retry-After
longer than MAX_WAIT is not waited out.

Unconfirmed from the sandbox that wrote this (it reached PyPI, not the Clash Royale API or
huggingface.co); check records what it sees in status.json "observed" on the PC:
  * PyPI: the project JSON's "releases" key is documented as deprecated; without it the check
    reads info.version and the per-version JSON's "urls".
  * /v1/cards: whether maxEvolutionLevel is omitted for 0 (read as 0) and whether tower troops
    come as "supportItems". radar.observed lists the fields and values seen.
  * Hugging Face: that GET /api/datasets/<repo> carries "sha" and "lastModified".
  * Table provenance text: the client build and content version are parsed from prose
    ("client build 160402020", "content_version 16.402.22"); a release that words it
    differently reports them as null, and the stage (U09) probes the engine.
  * The season and content dates in the ledger marked "confirmed": false.

Sources: https://docs.pypi.org/api/json/ (project and release JSON, digests, yanked);
https://huggingface.co/docs/hub/api (GET /api/datasets/{repo_id}); the Clash Royale API notes
docs/SEASON_UPDATES.md 15 cites (https://raw.githubusercontent.com/samaksh-bajaj/cr-deck-builder/
main/API_NOTES.md: maxEvolutionLevel is a bitmask, 123 cards; https://gist.github.com/loganlinn/
d813ba0f9ccb47f34462ec8abeab8e14: the Swagger spec), and tools/cr_api.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import email.utils
import hashlib
import http.client
import importlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import game_versions as gv  # noqa: E402  (standard library)
import rs_engine  # noqa: E402  (standard library at import; royalesim is not imported)

TOOL = "tools/game_update.py"
SCHEMA = 1

PYPI_BASE = "https://pypi.org/pypi"
HF_API = "https://huggingface.co/api/datasets"
ENGINE_PROJECT = "royalesim"
MAP_PROJECT = "royaleimitate"
USER_AGENT = ("yodlebot-game-update/1.0 (Clash Royale friendly-battle sparring research; "
              "tools/game_update.py; one check a day)")

EXIT_OK, EXIT_ERROR, EXIT_ARGS, EXIT_UPDATE, EXIT_NEW_CLIENT = 0, 1, 2, 10, 11
VERDICTS = {EXIT_OK: "OK", EXIT_UPDATE: "UPDATE AVAILABLE", EXIT_NEW_CLIENT: "NEW CLIENT",
            EXIT_ERROR: "ERROR"}
#: --skip choices: the network parts of the check.
PARTS = ("pypi", "royaleimitate", "radar", "il_replay")

HOST_GAP = 1.0              # seconds between two requests to one host
RETRIES = 3                 # retries after a 429, 5xx or no response
BACKOFF = 5.0               # first wait without Retry-After; doubles: 5, 10, 20 s
MAX_WAIT = 60.0             # a longer Retry-After is not waited out
TIMEOUT = 30.0
DOWNLOAD_TIMEOUT = 180.0
MAX_DOWNLOAD = 100 * 1024 * 1024
MAX_MEMBER = 32 * 1024 * 1024
MAX_SDISTS = 6
CARDS_FRESH_H = 20.0
FIRST_WINDOW_DAYS = 7
UPCOMING_DAYS = 7
UPSTREAM_DAYS = 7           # section 10: after 7 days behind with no release, ask upstream
POOL_STALE_H = 48.0

CALIBRATION = "data/calibration.json"
TABLE_FILE = re.compile(r"^data/derived/cards[^/]*\.json$")
UPSTREAM_TEST = re.compile(r"^crates/[^/]+/tests/[^/]+\.rs$")
MAP_MEMBER = "royaleimitate/replay_cards.json"
CLASSES_PATH = ROOT / "data" / "vision" / "classes.json"
ROYALEAPI_CARDS = ROOT / "data" / "royaleapi" / "cards.json"
CARDS_SNAPSHOT = "cards_api.last.json"
CACHE = "cache"

#: maxEvolutionLevel bits (API_NOTES.md) and the classes.json form ids they need.
FORM_BITS: tuple[tuple[int, int, str], ...] = ((1, 1, "evolution"), (2, 2, "hero"))


class CheckError(RuntimeError):
    """A part of the check could not run (PyPI unreachable, a bad response)."""


class IntegrityError(CheckError):
    """A download's sha256 is not the digest PyPI published."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Fetched:
    """One response: the HTTP status, the body and the headers (lower-case names)."""
    status: int
    body: bytes = b""
    headers: dict[str, str] = dataclasses.field(default_factory=dict)


def _transport(url: str, headers: Mapping[str, str], timeout: float) -> Fetched:
    """One GET through urllib (the tests replace this or fetch)."""
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read(MAX_DOWNLOAD + 1)
            return Fetched(int(resp.status), body,
                           {k.lower(): v for k, v in resp.headers.items()})
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(65536)
        except OSError:
            body = b""
        return Fetched(int(exc.code), body,
                       {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])})


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _clock() -> float:
    return time.monotonic()


_last_call: dict[str, float] = {}


def retry_after(value: str | None) -> float | None:
    """A Retry-After header in seconds (a number or an HTTP date), None when absent or bad."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return max(0.0, (when - gv.utcnow()).total_seconds())


def fetch(url: str, *, accept: str = "application/json", timeout: float = TIMEOUT,
          retries: int = RETRIES) -> Fetched:
    """GET `url` with USER_AGENT. Waits HOST_GAP between two calls to one host; backs off on
    429 and 503 (Retry-After, else BACKOFF doubling) and tries a 500, 502, 504 or a missing
    response again. Returns the last response whatever its status; raises CheckError when no
    response came back at all, or for a URL that is not https."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.netloc:
        raise CheckError(f"{url}: only https URLs are fetched")
    host = parts.netloc.lower()
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    attempt = 0
    while True:
        last = _last_call.get(host)
        if last is not None:
            gap = HOST_GAP - (_clock() - last)
            if gap > 0:
                _sleep(gap)
        _last_call[host] = _clock()
        error = ""
        try:
            got: Fetched | None = _transport(url, headers, timeout)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            got, error = None, f"{type(exc).__name__}: {exc}"
        if got is not None and got.status not in (429, 500, 502, 503, 504):
            return got
        if attempt >= retries:
            if got is None:
                raise CheckError(f"{url}: no response ({error})")
            return got
        wait = retry_after(got.headers.get("retry-after")) if got is not None else None
        if wait is not None and wait > MAX_WAIT:
            return got                      # not waited out
        _sleep(max(1.0, wait) if wait is not None else BACKOFF * 2 ** attempt)
        attempt += 1


def fetch_json(url: str, **kw: Any) -> tuple[int, Any]:
    """(status, parsed body or None)."""
    got = fetch(url, **kw)
    try:
        return got.status, json.loads(got.body.decode("utf-8")) if got.body else None
    except (UnicodeDecodeError, ValueError):
        return got.status, None


# ---------------------------------------------------------------------------
# PyPI
# ---------------------------------------------------------------------------

_VERSION = re.compile(
    r"^\s*v?(?P<release>\d+(?:\.\d+)*)"
    r"(?:[-_.]?(?P<pre>a|b|c|rc|alpha|beta|pre|preview)[-_.]?(?P<pre_n>\d*))?"
    r"(?:(?:[-_.]?(?:post|rev|r)[-_.]?(?P<post_n>\d*))|(?:-(?P<post_n2>\d+)))?"
    r"(?:[-_.]?(?P<dev>dev)[-_.]?(?P<dev_n>\d*))?"
    r"(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?\s*$", re.I)
_PRE_RANK = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2, "preview": 2}


def version_key(text: Any) -> tuple | None:
    """A sort key for a PEP 440 version (release, pre, post, dev; local ignored), None when it
    is not one."""
    m = _VERSION.match(str(text)) if text is not None else None
    if not m:
        return None
    release = [int(x) for x in m.group("release").split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    post_n = m.group("post_n") if m.group("post_n") is not None else m.group("post_n2")
    has_post = post_n is not None
    if m.group("pre"):
        pre = (_PRE_RANK[m.group("pre").lower()], int(m.group("pre_n") or 0))
    elif m.group("dev") and not has_post:
        pre = (-1, 0)                       # 1.0.dev1 < 1.0a1
    else:
        pre = (9, 0)
    post = (int(post_n or 0),) if has_post else (-1,)
    dev = (0, int(m.group("dev_n") or 0)) if m.group("dev") else (1, 0)
    return tuple(release), pre, post, dev


def is_prerelease(text: Any) -> bool:
    m = _VERSION.match(str(text))
    return bool(m and (m.group("pre") or m.group("dev")))


def pypi_json(project: str, version: str | None = None) -> dict[str, Any]:
    """PyPI's project JSON (GET /pypi/<project>/json) or one release's
    (GET /pypi/<project>/<version>/json). CheckError when it is not a 200 with an "info"."""
    url = (f"{PYPI_BASE}/{urllib.parse.quote(project)}/json" if version is None else
           f"{PYPI_BASE}/{urllib.parse.quote(project)}/{urllib.parse.quote(version)}/json")
    status, data = fetch_json(url)
    if status != 200 or not isinstance(data, dict) or not isinstance(data.get("info"), dict):
        raise CheckError(f"PyPI {url}: HTTP {status}"
                         + ("" if status != 200 else ", not the project JSON"))
    return data


def release_files(project_json: Mapping[str, Any], project: str, version: str
                  ) -> list[dict[str, Any]]:
    """The files of one release: the project JSON's "releases" entry, else the release JSON's
    "urls" (one more request)."""
    releases = project_json.get("releases")
    if isinstance(releases, Mapping) and isinstance(releases.get(version), list):
        return [f for f in releases[version] if isinstance(f, Mapping)]
    data = pypi_json(project, version)
    return [f for f in data.get("urls") or [] if isinstance(f, Mapping)]


def newer_versions(project_json: Mapping[str, Any], pinned: str, *, pre: bool = False
                   ) -> dict[str, list[str]]:
    """{"newer": releases newer than `pinned`, oldest first; "yanked": newer releases whose
    files are all yanked; "pre": newer pre-releases left out (empty with pre=True)}."""
    floor = version_key(pinned)
    if floor is None:
        raise CheckError(f"the pin {pinned!r} is not a version")
    releases = project_json.get("releases")
    names = set(releases) if isinstance(releases, Mapping) else set()
    latest = (project_json.get("info") or {}).get("version")
    if latest:
        names.add(str(latest))
    out: dict[str, list[str]] = {"newer": [], "yanked": [], "pre": []}
    for name in names:
        key = version_key(name)
        if key is None or key <= floor:
            continue
        files = releases.get(name) if isinstance(releases, Mapping) else None
        if isinstance(files, list) and files and all(f.get("yanked") for f in files
                                                     if isinstance(f, Mapping)):
            out["yanked"].append(name)
        elif is_prerelease(name) and not pre:
            out["pre"].append(name)
        else:
            out["newer"].append(name)
    for names_ in out.values():
        names_.sort(key=version_key)
    return out


def pick_sdist(files: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    for f in files:
        name = str(f.get("filename") or "")
        if (f.get("packagetype") == "sdist" and not f.get("yanked")
                and name.endswith((".tar.gz", ".zip"))):
            return dict(f)
    return None


def pick_wheel(files: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    for f in files:
        name = str(f.get("filename") or "")
        if f.get("packagetype") == "bdist_wheel" and not f.get("yanked") and name.endswith(".whl"):
            return dict(f)
    return None


def upload_time(files: Iterable[Mapping[str, Any]]) -> str | None:
    got = [str(f.get("upload_time_iso_8601") or f.get("upload_time") or "") for f in files]
    got = [t for t in got if t]
    return min(got) if got else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(name: str) -> str:
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    if not base or base in (".", "..") or not re.fullmatch(r"[A-Za-z0-9._+-]+", base):
        raise CheckError(f"{name!r} is not a plain file name")
    return base


def download_verified(entry: Mapping[str, Any], cache: Path) -> bytes:
    """A release file's bytes, checked against PyPI's sha256 and kept in `cache` (read from
    there when it is already there and still matches). IntegrityError on a mismatch after one
    more download; the bad bytes are never written."""
    name = _safe_name(str(entry.get("filename") or ""))
    expect = str((entry.get("digests") or {}).get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expect):
        raise IntegrityError(f"{name}: PyPI gives no sha256 digest")
    path = cache / name
    if path.is_file():
        data = path.read_bytes()
        if _sha256(data) == expect:
            return data
    url = str(entry.get("url") or "")
    got_sha = ""
    for _ in range(2):
        got = fetch(url, accept="*/*", timeout=DOWNLOAD_TIMEOUT)
        if got.status != 200:
            raise CheckError(f"{name}: HTTP {got.status} from {url}")
        if len(got.body) > MAX_DOWNLOAD:
            raise CheckError(f"{name}: larger than {MAX_DOWNLOAD // (1024 * 1024)} MB")
        got_sha = _sha256(got.body)
        if got_sha == expect:
            cache.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(got.body)
            tmp.replace(path)
            return got.body
    raise IntegrityError(f"{name}: sha256 {got_sha} is not PyPI's {expect}; not installed, "
                         "not kept. Check again later; if it stays, do not update.")


# ---------------------------------------------------------------------------
# reading a release archive (never unpacked to disk)
# ---------------------------------------------------------------------------

def _rel(name: str, wheel: bool) -> str:
    name = name.replace("\\", "/").lstrip("./")
    if wheel or "/" not in name:
        return name
    return name.split("/", 1)[1]            # an sdist's members sit under <project>-<version>/


def archive_names(data: bytes, filename: str) -> list[str]:
    """Every regular member's path inside the archive (an sdist's top folder dropped)."""
    wheel = filename.endswith(".whl")
    if filename.endswith((".zip", ".whl")):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return [_rel(i.filename, wheel) for i in zf.infolist() if not i.is_dir()]
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        return [_rel(m.name, False) for m in tf.getmembers() if m.isfile()]


def archive_read(data: bytes, filename: str, want: Callable[[str], bool]) -> dict[str, bytes]:
    """{path: bytes} of the regular members `want` accepts, each at most MAX_MEMBER bytes. Read
    in memory: nothing is extracted, links and special files are skipped."""
    out: dict[str, bytes] = {}
    wheel = filename.endswith(".whl")
    if filename.endswith((".zip", ".whl")):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                rel = _rel(info.filename, wheel)
                if not info.is_dir() and info.file_size <= MAX_MEMBER and want(rel):
                    out[rel] = zf.read(info)
        return out
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        for member in tf.getmembers():
            rel = _rel(member.name, False)
            if member.isfile() and member.size <= MAX_MEMBER and want(rel):
                handle = tf.extractfile(member)
                if handle is not None:
                    out[rel] = handle.read()
    return out


def ledger_arms(calibration: Any) -> tuple[list[str], str | None]:
    """(cards.CARD_TABLE candidates, its value: the engine's default arm)."""
    section = calibration.get("cards") if isinstance(calibration, Mapping) else None
    entry = section.get("CARD_TABLE") if isinstance(section, Mapping) else None
    if not isinstance(entry, Mapping):
        return [], None
    cands = entry.get("candidates")
    value = entry.get("value")
    return ([str(c) for c in cands] if isinstance(cands, list) else []), (
        str(value).strip('"') if value is not None else None)


_BUILD_PATTERNS = (re.compile(r"client build (\d{8,10})"),
                   re.compile(r"MANIFEST\.json build (\d{8,10})"),
                   re.compile(r"\bbuild (\d{8,10})"))
_CONTENT_PATTERNS = (re.compile(r"content_version (\d+\.\d+(?:\.\d+)?)"),
                     re.compile(r"\bcontent (\d+\.\d+(?:\.\d+)?)"),
                     re.compile(r"MANIFEST\.json version (\d+\.\d+(?:\.\d+)?)"))
_DATED = re.compile(r"-(\d{8})(?:\.\d+)?$")


def table_dates(version: str | None) -> str | None:
    """"cards-160402017-20261019.1" -> "2026-10-19" (the content update's date), else None."""
    m = _DATED.search(str(version or ""))
    if not m:
        return None
    text = m.group(1)
    try:
        return dt.date(int(text[:4]), int(text[4:6]), int(text[6:])).isoformat()
    except ValueError:
        return None


def provenance_versions(table: Mapping[str, Any]) -> dict[str, Any]:
    """{"client_build", "client", "content"} from a table's provenance prose (None each when
    the text does not say)."""
    prov = table.get("provenance") if isinstance(table.get("provenance"), Mapping) else {}
    texts = [str(prov.get(k) or "") for k in ("source", "roster_dating", "vintage")]
    texts.append(str(table.get("vintage_warning") or ""))
    joined = "\n".join(texts)
    build = None
    for pattern in _BUILD_PATTERNS:
        m = pattern.search(joined)
        if m:
            try:
                gv.parse_code(m.group(1))
                build = int(m.group(1))
                break
            except ValueError:
                continue
    content = None
    for pattern in _CONTENT_PATTERNS:
        m = pattern.search(joined)
        if m:
            content = m.group(1)
            break
    client = gv.version_name(build) if build else None
    if client is None:
        m = _CONTENT_PATTERNS[2].search(joined)
        client = m.group(1) if m else None
    return {"client_build": build, "client": client, "content": content}


def table_summary(raw: bytes, path: str) -> dict[str, Any]:
    """What check records about one card-table file."""
    try:
        table = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        return {"file": path, "error": f"not JSON ({exc})", "fnv1a64": rs_engine.fnv1a64(raw),
                "sha256": _sha256(raw), "bytes": len(raw)}
    if not isinstance(table, Mapping):
        return {"file": path, "error": "not a JSON object", "fnv1a64": rs_engine.fnv1a64(raw),
                "sha256": _sha256(raw), "bytes": len(raw)}
    version = table.get("version")
    arm = rs_engine.table_name(version) if isinstance(version, str) else None
    forms: dict[str, set[str]] = {}
    for section, kind in (("evolutions", "evolution"), ("hero_forms", "hero")):
        for rec in table.get(section) or []:
            if isinstance(rec, Mapping) and rec.get("form_of"):
                forms.setdefault(str(rec["form_of"]), set()).add(kind)
    cards = sorted(str(c["name"]) for c in table.get("cards") or []
                   if isinstance(c, Mapping) and c.get("name"))
    prov = table.get("provenance") if isinstance(table.get("provenance"), Mapping) else {}
    return {"file": path, "version": version, "arm": arm, **provenance_versions(table),
            "content_date": table_dates(version), "vintage_key": prov.get("vintage_key"),
            "fnv1a64": rs_engine.fnv1a64(raw), "sha256": _sha256(raw), "bytes": len(raw),
            "roster": {"cards": len(cards), "evolutions": len(table.get("evolutions") or []),
                       "hero_forms": len(table.get("hero_forms") or []),
                       "towers": len(table.get("towers") or []),
                       "units": len(table.get("units") or {})},
            "forms": {k: sorted(v) for k, v in sorted(forms.items())}, "cards": cards}


def read_engine_archive(data: bytes, filename: str) -> dict[str, Any]:
    """A royalesim sdist: {"ledger": {"found", "arms", "default_arm"}, "tables": [...],
    "upstream_tests": the count of crates/*/tests/*.rs}."""
    names = archive_names(data, filename)
    members = archive_read(data, filename,
                           lambda rel: rel == CALIBRATION or bool(TABLE_FILE.match(rel)))
    cal = None
    if CALIBRATION in members:
        try:
            cal = json.loads(members[CALIBRATION].decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError):
            cal = None
    arms, default = ledger_arms(cal)
    tables = [table_summary(raw, rel) for rel, raw in sorted(members.items())
              if TABLE_FILE.match(rel)]
    return {"ledger": {"found": isinstance(cal, Mapping), "arms": arms, "default_arm": default},
            "tables": tables, "upstream_tests": sum(1 for n in names if UPSTREAM_TEST.match(n)),
            "members": len(names)}


def _newest(tables: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    best, best_key = None, None
    for t in tables:
        content = gv.parse_content(t.get("content"))
        if content is None:
            continue
        key = (content, t.get("content_date") or "", t.get("client_build") or 0)
        if best_key is None or key > best_key:
            best, best_key = dict(t), key
    return best


def compare_release(base: Mapping[str, Any] | None, summary: Mapping[str, Any]) -> dict[str, Any]:
    """A new release's sdist summary against a ledger release entry (the pin's): arms added and
    removed, the default arm, and per arm "same", "new", "new version" (a table version string
    the arm did not have), "bytes changed" (the same version, other bytes) or "no file"."""
    base = base or {}
    old_tables = base.get("tables") or {}
    old_arms = list(base.get("arms") or old_tables)
    new_arms = list((summary.get("ledger") or {}).get("arms") or [])
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for t in summary.get("tables") or []:
        if t.get("arm"):
            by_arm.setdefault(str(t["arm"]), []).append(t)
    rows = []
    for arm in new_arms + [a for a in old_arms if a not in new_arms]:
        got = by_arm.get(arm) or []
        old = old_tables.get(arm) if isinstance(old_tables.get(arm), Mapping) else None
        if arm not in new_arms:
            status = "removed"
        elif not got:
            status = "no file"
        elif old is None:
            status = "new"
        elif got[0].get("version") != old.get("version"):
            status = "new version"
        elif old.get("fnv1a64") and got[0].get("fnv1a64") != old.get("fnv1a64"):
            status = "bytes changed"
        else:
            status = "same"
        row = {"arm": arm, "status": status}
        if got:
            row.update({k: got[0].get(k) for k in ("version", "content", "client_build",
                                                    "content_date", "fnv1a64")})
            if len({t.get("fnv1a64") for t in got}) > 1:
                row["note"] = "two files of this arm differ: " + ", ".join(
                    f"{t['file']} {t['fnv1a64']}" for t in got)
        if old is not None:
            row["old"] = {k: old.get(k) for k in ("version", "content", "fnv1a64")}
        rows.append(row)
    in_arms = [t for arm in new_arms for t in by_arm.get(arm, [])[:1]]
    default_old, default_new = base.get("default_arm"), (summary.get("ledger") or {}).get(
        "default_arm")
    return {"arms_added": [a for a in new_arms if a not in old_arms],
            "arms_removed": [a for a in old_arms if a not in new_arms],
            "default_arm": {"old": default_old, "new": default_new,
                            "changed": bool(default_new) and default_new != default_old},
            "tables": rows,
            "changed_tables": [r["arm"] for r in rows if r["status"] in (
                "new", "new version", "bytes changed", "removed")],
            "not_arms": sorted({str(t.get("version")) for t in summary.get("tables") or []
                                if t.get("arm") not in new_arms}),
            "newest_table": _newest(in_arms)}


def _summary_path(cache: Path, filename: str) -> Path:
    return cache / (_safe_name(filename) + ".summary.json")


def read_release(project: str, version: str, files: Sequence[Mapping[str, Any]], cache: Path,
                 reader: Callable[[bytes, str], dict[str, Any]]) -> dict[str, Any]:
    """Download (verified) and read one release's sdist, or its wheel when there is no sdist
    and `project` is royaleimitate; the parsed summary is cached by sha256."""
    entry = pick_sdist(files)
    kind = "sdist"
    if entry is None and project == MAP_PROJECT:
        entry, kind = pick_wheel(files), "wheel"
    if entry is None:
        return {"version": version, "archive": None, "note": "no sdist on PyPI: the stage "
                "(U09) probes the wheel"}
    sha = str((entry.get("digests") or {}).get("sha256") or "").lower()
    cached = gv.read_json(_summary_path(cache, str(entry["filename"])))
    if isinstance(cached, dict) and isinstance(cached.get("archive"), dict) and \
            cached["archive"].get("sha256") == sha and cached.get("version") == version:
        return cached
    data = download_verified(entry, cache)
    out = {"version": version, "archive": {"file": entry["filename"], "kind": kind,
                                           "sha256": sha, "verified": True,
                                           "bytes": len(data)}}
    out.update(reader(data, str(entry["filename"])))
    gv.write_json(_summary_path(cache, str(entry["filename"])), out)
    return out


def read_map_archive(data: bytes, filename: str) -> dict[str, Any]:
    """royaleimitate: its replay_cards.json as {"map": {key: name}, "map_sha256"}."""
    got = archive_read(data, filename, lambda rel: rel == MAP_MEMBER)
    if MAP_MEMBER not in got:
        return {"map": None, "note": f"no {MAP_MEMBER} in {filename}"}
    raw = got[MAP_MEMBER]
    try:
        mapping = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        return {"map": None, "note": f"{MAP_MEMBER} is not JSON"}
    if not isinstance(mapping, dict):
        return {"map": None, "note": f"{MAP_MEMBER} is not an object"}
    return {"map": {str(k): str(v) for k, v in mapping.items()},
            "map_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


def compare_map(old: Mapping[str, str], new: Mapping[str, str]) -> dict[str, Any]:
    return {"added": {k: new[k] for k in sorted(new) if k not in old},
            "removed": sorted(k for k in old if k not in new),
            "renamed": {k: [old[k], new[k]] for k in sorted(old) if k in new and old[k] != new[k]}}


def check_engine(ledger: Mapping[str, Any], cache: Path, *, pre: bool, max_sdists: int
                 ) -> dict[str, Any]:
    """The pypi part. CheckError when PyPI cannot be read."""
    pinned = gv.pin(ledger)
    project = pypi_json(ENGINE_PROJECT)
    found = newer_versions(project, pinned["royalesim"], pre=pre)
    out: dict[str, Any] = {"pin": pinned["royalesim"], "newest": pinned["royalesim"],
                           "newer": [], "yanked": found["yanked"], "pre": found["pre"],
                           "errors": [], "observed": {"releases_key": isinstance(
                               project.get("releases"), Mapping)}}
    newer = found["newer"]
    if newer:
        out["newest"] = newer[-1]
    unread = newer[:-max_sdists] if max_sdists > 0 else list(newer)
    for version in newer:
        if version in unread:
            out["newer"].append({"version": version, "read": False,
                                 "note": f"not read (--max-sdists {max_sdists})"})
            continue
        try:
            files = release_files(project, ENGINE_PROJECT, version)
            item = read_release(ENGINE_PROJECT, version, files, cache, read_engine_archive)
        except IntegrityError as exc:
            out["errors"].append(str(exc))
            out["newer"].append({"version": version, "read": False, "integrity": False,
                                 "error": str(exc)})
            continue
        except (CheckError, OSError, tarfile.TarError, zipfile.BadZipFile, EOFError) as exc:
            out["newer"].append({"version": version, "read": False,
                                 "error": f"{type(exc).__name__}: {exc}"})
            continue
        item["uploaded"] = upload_time(files)
        item["read"] = item.get("archive") is not None
        if item["read"]:
            item["compare"] = compare_release(pinned["release"], item)
        out["newer"].append(_slim(item))
    return out


def _slim(item: Mapping[str, Any]) -> dict[str, Any]:
    """A release summary without the per-table card lists (kept for the radar as forms)."""
    out = dict(item)
    out["tables"] = [{k: v for k, v in t.items() if k != "cards"} for t in item.get("tables")
                     or []]
    return out


def check_card_map(ledger: Mapping[str, Any], cache: Path, *, pre: bool, max_sdists: int
                   ) -> dict[str, Any]:
    """The royaleimitate part (CheckError when PyPI cannot be read)."""
    vendored = (ledger.get("royaleimitate") or {}).get("vendored")
    if not vendored:
        return {"skipped": "the ledger names no vendored royaleimitate"}
    try:
        import rs_cards
        mine = rs_cards.read_vendored()
    except Exception as exc:                # CardMapError, or the file is missing
        return {"vendored": vendored, "error": f"the vendored map: {exc}"}
    project = pypi_json(MAP_PROJECT)
    found = newer_versions(project, vendored, pre=pre)
    out: dict[str, Any] = {"vendored": vendored, "newest": found["newer"][-1] if found["newer"]
                           else vendored, "newer": [], "yanked": found["yanked"],
                           "pre": found["pre"]}
    for version in (found["newer"][-max_sdists:] if max_sdists > 0 else []):
        try:
            files = release_files(project, MAP_PROJECT, version)
            item = read_release(MAP_PROJECT, version, files, cache, read_map_archive)
        except IntegrityError as exc:
            out["newer"].append({"version": version, "integrity": False, "error": str(exc)})
            continue
        except (CheckError, OSError, tarfile.TarError, zipfile.BadZipFile, EOFError) as exc:
            out["newer"].append({"version": version, "error": f"{type(exc).__name__}: {exc}"})
            continue
        entry = {"version": version, "uploaded": upload_time(files),
                 "archive": item.get("archive"), "note": item.get("note")}
        if item.get("map") is not None:
            entry["map_sha256"] = item.get("map_sha256")
            entry["same_as_vendored"] = item.get("map_sha256") == (
                ledger.get("royaleimitate") or {}).get("sha256")
            entry.update(compare_map(mine, item["map"]))
        out["newer"].append(entry)
    return out


# ---------------------------------------------------------------------------
# the roster radar
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slug_fallback(name: str) -> str:
    """The section 7.3 rule, used only when tools/cr_api.py is missing (cr_api.slug is the one
    rule): lower case, dots dropped, other runs of non-alphanumerics -> "-"."""
    return _NON_ALNUM.sub("-", str(name).strip().lower().replace(".", "")).strip("-")


def _cr_api():
    """tools/cr_api.py (U02), imported when the check runs; None when it is not there."""
    try:
        return importlib.import_module("cr_api")
    except ImportError:
        return None


def slug_function() -> Callable[[str], str]:
    mod = _cr_api()
    fn = getattr(mod, "slug", None) if mod is not None else None
    return fn if callable(fn) else _slug_fallback


_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


def mask_ips(text: str, keep: Iterable[str] = ()) -> str:
    """`text` with every IPv4 address but those in `keep` written as <ip>. A 403 invalidIp
    message names the PC's public address, and status.txt is printed in every selfcheck and
    live header (and U12 copies reports into git): the address must not get there."""
    kept = {str(k) for k in keep if k}
    return _IPV4.sub(lambda m: m.group(0) if m.group(0) in kept else "<ip>", str(text))


def api_settings(data_root: Path | None) -> dict[str, str]:
    """base and transport from U03's <data>/opponents/config.json, when it sets them (read
    loosely: tools/opponents.py validates the file). An owner on the proxy because of a changing
    IP set it there, and this call must go the same way."""
    if data_root is None:
        return {}
    cfg = gv.read_json(Path(data_root) / "opponents" / "config.json")
    if not isinstance(cfg, Mapping):
        return {}
    out: dict[str, str] = {}
    base = cfg.get("base")
    if isinstance(base, str) and base.strip():
        out["base"] = base.strip()
    transport = cfg.get("transport")
    if isinstance(transport, str) and transport.strip().lower() in ("urllib", "curl"):
        out["transport"] = transport.strip().lower()
    return out


def api_cards(data_root: Path | None = None) -> tuple[Any, str]:
    """(GET /v1/cards body or None, what happened). Through cr_api.Client, at one request per
    second, with the opponents config's base and transport; never raises. What happened holds
    no IP address (mask_ips)."""
    mod = _cr_api()
    if mod is None or not hasattr(mod, "Client"):
        return None, "tools/cr_api.py (U02) is not there"
    keep = [getattr(mod, "PROXY_IP", "")]
    try:
        client = mod.Client(per_second=1.0, **api_settings(data_root))
    except Exception as exc:                # TokenError: no key; never prints the token
        return None, mask_ips(f"no API client ({type(exc).__name__}: {exc})", keep)
    try:
        resp = client.cards()
    except Exception as exc:
        return None, mask_ips(f"GET /v1/cards failed ({type(exc).__name__}: {exc})", keep)
    if not getattr(resp, "ok", False):
        hint = resp.hint() if hasattr(resp, "hint") else f"HTTP {getattr(resp, 'status', '?')}"
        return None, mask_ips(f"GET /v1/cards: {hint}", keep)
    return resp.data, "GET /v1/cards (tools/cr_api.py)"


def unwrap_cards(obj: Any) -> dict[str, Any] | None:
    """A /v1/cards body ({"items": [...], "supportItems": [...]}), also when a snapshot file
    wraps it under "data", "response" or "cards"."""
    if isinstance(obj, Mapping) and isinstance(obj.get("items"), list):
        return dict(obj)
    if isinstance(obj, Mapping):
        for key in ("data", "response", "cards", "payload"):
            inner = obj.get(key)
            if isinstance(inner, Mapping) and isinstance(inner.get("items"), list):
                return dict(inner)
    return None


def _snapshot_time(obj: Any, path: Path | None) -> dt.datetime | None:
    if isinstance(obj, Mapping):
        for key in ("fetched_at", "pulled_at", "at", "time"):
            got = gv.parse_time(obj.get(key))
            if got:
                return got
    if path is not None:
        try:
            return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
        except OSError:
            return None
    return None


def roster(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """id -> {"id", "name", "max_evo", "elixir", "rarity", "support"} for items and
    supportItems."""
    out: dict[int, dict[str, Any]] = {}
    for key, support in (("items", False), ("supportItems", True)):
        for item in payload.get(key) or []:
            if not isinstance(item, Mapping) or item.get("id") is None:
                continue
            try:
                cid = int(item["id"])
            except (TypeError, ValueError):
                continue
            try:
                evo = int(item.get("maxEvolutionLevel") or 0)
            except (TypeError, ValueError):
                evo = 0
            out[cid] = {"id": cid, "name": str(item.get("name") or ""), "max_evo": evo,
                        "elixir": item.get("elixirCost"), "rarity": item.get("rarity"),
                        "support": support}
    return out


def observed(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What the PC's /v1/cards actually carries (the shapes not confirmed from the sandbox)."""
    items = [i for i in payload.get("items") or [] if isinstance(i, Mapping)]
    fields: dict[str, int] = {}
    icons: dict[str, int] = {}
    evo_values: dict[str, int] = {}
    for item in items:
        for k in item:
            fields[k] = fields.get(k, 0) + 1
        for k in (item.get("iconUrls") or {}) if isinstance(item.get("iconUrls"), Mapping) else {}:
            icons[k] = icons.get(k, 0) + 1
        v = str(item.get("maxEvolutionLevel", "absent"))
        evo_values[v] = evo_values.get(v, 0) + 1
    return {"items": len(items), "supportItems": (len(payload["supportItems"]) if isinstance(
                payload.get("supportItems"), list) else None),
            "fields": dict(sorted(fields.items())), "icon_keys": dict(sorted(icons.items())),
            "maxEvolutionLevel": dict(sorted(evo_values.items()))}


def _camel(name: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z0-9]+", name) if w)


def engine_cards(path: Path = CLASSES_PATH) -> dict[str, dict[str, Any]] | None:
    """Catalogue name -> {"forms": [0 base, 1 evo, 2 hero], "elixir"} from classes.json."""
    got = gv.read_json(path)
    cards = got.get("cards") if isinstance(got, Mapping) else None
    if not isinstance(cards, Mapping):
        return None
    return {str(k): {"forms": [int(f) for f in (v.get("forms") or [])],
                     "elixir": v.get("elixir")}
            for k, v in cards.items() if isinstance(v, Mapping)}


def royaleapi_keys(path: Path = ROYALEAPI_CARDS) -> dict[int, str]:
    """Official card id -> RoyaleAPI key (data/royaleapi/cards.json)."""
    got = gv.read_json(path)
    out: dict[int, str] = {}
    for c in (got.get("cards") if isinstance(got, Mapping) else None) or []:
        try:
            out[int(c["id"])] = str(c["key"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def resolve(card: Mapping[str, Any], *, slug: Callable[[str], str], card_map: Mapping[str, str],
            keys_by_id: Mapping[int, str], engine: Mapping[str, Mapping[str, Any]]
            ) -> dict[str, Any]:
    """Section 7.3's chain for one API card: {"key", "engine" (catalogue name or None), "how":
    "key" | "id" | "auto" | None}."""
    key = slug(card["name"])
    name = card_map.get(key)
    how = "key" if name else None
    if name is None and card["id"] in keys_by_id:
        alt = keys_by_id[card["id"]]
        if card_map.get(alt):
            key, name, how = alt, card_map[alt], "id"
    if name is None:
        guess = _camel(card["name"])
        info = engine.get(guess)
        if info is not None and (card.get("elixir") is None or info.get("elixir") is None
                                 or info.get("elixir") == card.get("elixir")):
            name, how = guess, "auto"
    return {"key": key, "engine": name, "how": how,
            "in_engine": bool(name) and name in engine}


def radar(current: Mapping[str, Any], previous: Mapping[str, Any] | None, *,
          engine: Mapping[str, Mapping[str, Any]] | None, card_map: Mapping[str, str],
          keys_by_id: Mapping[int, str], slug: Callable[[str], str],
          newer_forms: Sequence[tuple[str, str, Mapping[str, Sequence[str]]]] = ()
          ) -> dict[str, Any]:
    """The radar: today's /v1/cards against the previous snapshot and the engine's forms.
    newer_forms: (royalesim version, arm, {card: [kinds]}) of newer releases' tables, newest
    first, to say which one has a missing form."""
    now = roster(current)
    before = roster(previous) if previous is not None else None
    has_engine = bool(engine)               # without classes.json, forms are not compared
    engine = engine or {}
    resolved = {cid: resolve(c, slug=slug, card_map=card_map, keys_by_id=keys_by_id,
                             engine=engine) for cid, c in now.items() if not c["support"]}

    def has_it(name: str | None, kind: str) -> str | None:
        for version, arm, forms in newer_forms:
            if name and kind in (forms.get(name) or ()):
                return f"royalesim {version} ({arm})"
        return None

    def item(cid: int, **extra: Any) -> dict[str, Any]:
        c = now.get(cid) or (before or {}).get(cid) or {"id": cid, "name": "?"}
        r = resolved.get(cid) or {}
        return {"id": cid, "name": c["name"], "engine": r.get("engine"), "how": r.get("how"),
                **extra}

    gaps = []
    for cid, c in sorted(now.items()):
        r = resolved.get(cid)
        if r is None or not r["in_engine"]:
            continue
        forms = engine[r["engine"]]["forms"]
        for bit, form_id, kind in FORM_BITS:
            if c["max_evo"] & bit and form_id not in forms:
                gaps.append(item(cid, form=kind, newer=has_it(r["engine"], kind)))
    unmapped = [item(cid, key=r["key"], auto=r["engine"] if r["how"] == "auto" else None,
                     in_engine=r["in_engine"])
                for cid, r in sorted(resolved.items()) if r["how"] in (None, "auto")]
    out: dict[str, Any] = {"cards": sum(1 for c in now.values() if not c["support"]),
                           "support": sum(1 for c in now.values() if c["support"]),
                           "baseline": before is None, "gaps": gaps, "unmapped": unmapped,
                           "not_in_engine": [item(cid) for cid, r in sorted(resolved.items())
                                             if has_engine and r["engine"]
                                             and not r["in_engine"]],
                           "new_cards": [], "removed": [], "renamed": [], "bits_gained": [],
                           "bits_lost": [], "new_support": [], "changed": False,
                           "engine_forms": "data/vision/classes.json" if has_engine else None}
    if before is not None:
        for cid in sorted(set(now) - set(before)):
            entry = item(cid, key=(resolved.get(cid) or {}).get("key"),
                         in_engine=(resolved.get(cid) or {}).get("in_engine", False))
            out["new_support" if now[cid]["support"] else "new_cards"].append(entry)
        out["removed"] = [{"id": cid, "name": before[cid]["name"]}
                          for cid in sorted(set(before) - set(now))]
        for cid in sorted(set(now) & set(before)):
            a, b = before[cid], now[cid]
            if a["name"] != b["name"]:
                out["renamed"].append({"id": cid, "old": a["name"], "new": b["name"],
                                       "engine": (resolved.get(cid) or {}).get("engine")})
            gained = b["max_evo"] & ~a["max_evo"]
            lost = a["max_evo"] & ~b["max_evo"]
            for bit, form_id, kind in FORM_BITS:
                r = resolved.get(cid) or {}
                if gained & bit:
                    lacks = bool(r.get("in_engine")) and form_id not in \
                        engine[r["engine"]]["forms"]
                    out["bits_gained"].append(item(cid, form=kind, simulator_lacks=lacks,
                                                   newer=has_it(r.get("engine"), kind)))
                if lost & bit:
                    out["bits_lost"].append(item(cid, form=kind))
        out["changed"] = any(out[k] for k in ("new_cards", "removed", "renamed", "bits_gained",
                                              "bits_lost", "new_support"))
    return out


def radar_messages(r: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(warnings, notes) for the status."""
    warn: list[str] = []
    info: list[str] = []
    article = {"evolution": "an evolution", "hero": "a hero form"}
    for g in r.get("bits_gained") or []:
        tail = ("the simulator lacks it" if g.get("simulator_lacks") else
                "the simulator has it" if g.get("engine") else
                "the card is not mapped to the simulator")
        if g.get("simulator_lacks") and g.get("newer"):
            tail += f"; {g['newer']} has it"
        warn.append(f"{g['name']} gained {article[g['form']]}; {tail}")
    gained = {(g["id"], g["form"]) for g in r.get("bits_gained") or []}
    for g in r.get("gaps") or []:
        if (g["id"], g["form"]) in gained:
            continue
        tail = f"; {g['newer']} has it" if g.get("newer") else ""
        warn.append(f"{g['engine'] or g['name']}: {g['form']} in the game, not in the table "
                    f"(simulated as base){tail}")
    for c in r.get("new_cards") or []:
        where = ("in the engine as " + str(c["engine"])) if c.get("in_engine") else \
            "not in the engine"
        warn.append(f"new card in the game: {c['name']} (id {c['id']}), {where}")
    for c in r.get("renamed") or []:
        warn.append(f"card {c['id']} renamed: {c['old']} -> {c['new']}")
    for c in r.get("removed") or []:
        warn.append(f"card {c['id']} ({c['name']}) is gone from /v1/cards")
    for c in r.get("new_support") or []:
        info.append(f"new tower troop in the game: {c['name']} (id {c['id']}); RoyaleSim plays "
                    "the Tower Princess only")
    for c in r.get("bits_lost") or []:
        info.append(f"{c['name']} lost its {c['form']} bit in /v1/cards")
    for c in r.get("unmapped") or []:
        if c.get("auto"):
            info.append(f"{c['name']} has no card-map key; auto match {c['auto']} (propose "
                        f'"{c["key"]}": {{"name": "{c["auto"]}", "verified": false}} for '
                        "data/cards/replay_cards_extra.json)")
        else:
            warn.append(f"{c['name']} (id {c['id']}) has no card-map key and no catalogue match")
    for c in r.get("not_in_engine") or []:
        warn.append(f"{c['name']} maps to {c['engine']}, which the engine catalogue lacks")
    return warn, info


def check_radar(updates: Path, data_root: Path, cards_file: Path | None,
                ledger_newer: Sequence[Mapping[str, Any]], now: dt.datetime) -> dict[str, Any]:
    """The radar part: picks the snapshot, compares, and saves it for the next check."""
    payload, source, fetched = None, None, None
    if cards_file is not None:
        raw = gv.read_json(cards_file)
        payload = unwrap_cards(raw)
        if payload is None:
            return {"error": f"--cards {cards_file}: not a /v1/cards response"}
        source, fetched = f"--cards {cards_file.name}", _snapshot_time(raw, cards_file)
    else:
        pulled = data_root / "opponents" / "cards_api.json"
        raw = gv.read_json(pulled) if pulled.is_file() else None
        when = _snapshot_time(raw, pulled) if raw is not None else None
        if unwrap_cards(raw) is not None and when is not None and \
                (now - when).total_seconds() <= CARDS_FRESH_H * 3600:
            payload, source, fetched = unwrap_cards(raw), "<data>/opponents/cards_api.json", when
        else:
            body, note = api_cards(data_root)
            payload = unwrap_cards(body)
            if payload is None:
                return {"skipped": f"no /v1/cards snapshot: {note}"}
            source, fetched = note, now
    engine = engine_cards(CLASSES_PATH)
    try:
        import rs_cards
        card_map = rs_cards.load()
    except Exception as exc:
        return {"error": f"the card map: {exc}"}
    previous = unwrap_cards(gv.read_json(updates / CARDS_SNAPSHOT))
    newer_forms = []
    for release_ in reversed(list(ledger_newer)):
        best = ((release_.get("compare") or {}).get("newest_table") or {})
        for t in release_.get("tables") or []:
            if t.get("arm") == best.get("arm") and t.get("forms") is not None:
                newer_forms.append((release_["version"], t["arm"], t["forms"]))
    result = radar(payload, previous, engine=engine, card_map=card_map,
                   keys_by_id=royaleapi_keys(ROYALEAPI_CARDS), slug=slug_function(),
                   newer_forms=newer_forms)
    result.update({"source": source, "fetched_at": gv.iso(fetched) if fetched else None,
                   "observed": observed(payload)})
    if engine is None:
        result["note"] = "data/vision/classes.json is missing: forms not compared"
    gv.write_json(updates / CARDS_SNAPSHOT, {"fetched_at": result["fetched_at"],
                                             "source": source, "data": payload})
    return result


# ---------------------------------------------------------------------------
# calendar, IL_Replay, pool, client
# ---------------------------------------------------------------------------

def _event_json(e: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in e.items() if k != "date"}
    out["date"] = e["date"].isoformat() if isinstance(e["date"], dt.date) else str(e["date"])
    return out


def calendar(ledger: Mapping[str, Any], since: dt.datetime | None, now: dt.datetime
             ) -> dict[str, Any]:
    today = now.date()
    after = since.date() if since else today - dt.timedelta(days=FIRST_WINDOW_DAYS)
    number, start = gv.season_on(today)
    nxt, nstart = gv.next_season(today)
    late = gv.behind(ledger, today)
    return {"since": after.isoformat(), "first_check": since is None,
            "passed": [_event_json(e) for e in gv.events(ledger, after=after, upto=today)],
            "today": [_event_json(e) for e in gv.events(
                ledger, after=today - dt.timedelta(days=1), upto=today)],
            "upcoming": [_event_json(e) for e in gv.events(
                ledger, after=today, upto=today + dt.timedelta(days=UPCOMING_DAYS))],
            "season": {"number": number, "start": start.isoformat()},
            "next_season": {"number": nxt, "start": nstart.isoformat()},
            "behind": None if late is None else {
                "since": late["since"].isoformat(), "days": late["days"],
                "events": [_event_json(e) for e in late["events"]]}}


def check_il_replay(ledger: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    info = ledger.get("il_replay") or {}
    repo = str(info.get("repo") or "")
    if not repo:
        return {"skipped": "the ledger names no IL_Replay repo"}
    url = f"{HF_API}/{urllib.parse.quote(repo, safe='/')}"
    try:
        status, data = fetch_json(url)
    except CheckError as exc:
        return {"repo": repo, "error": str(exc)}
    if status != 200 or not isinstance(data, Mapping):
        return {"repo": repo, "error": f"HTTP {status} from {url}"}
    sha = data.get("sha")
    previous = ((state.get("il_replay") or {}).get("sha")
                if isinstance(state.get("il_replay"), Mapping) else None)
    recorded = info.get("revision")
    return {"repo": repo, "sha": sha, "last_modified": data.get("lastModified"),
            "recorded": recorded, "previous": previous,
            "changed": bool(sha) and ((recorded is not None and sha != recorded)
                                      or (previous is not None and sha != previous)),
            "first_seen": recorded is None and previous is None,
            "observed": {"keys": sorted(str(k) for k in data)[:40]}}


def pool_status(data_root: Path, now: dt.datetime) -> dict[str, Any]:
    """When U03 last pulled (its <data>/opponents/state.json; the battles file's time as a
    fallback). Read loosely: U03 decides the file's layout."""
    folder = data_root / "opponents"
    state = gv.read_json(folder / "state.json")
    when = None
    if isinstance(state, Mapping):
        for key in ("pulled_at", "last_pull", "last_pulled", "pull_finished", "finished_at"):
            when = gv.parse_time(state.get(key))
            if when:
                break
        if when is None and isinstance(state.get("pull"), Mapping):
            for key in ("finished_at", "finished", "at", "time"):
                when = gv.parse_time(state["pull"].get(key))
                if when:
                    break
    source = "<data>/opponents/state.json" if when else None
    if when is None:
        battles = folder / "battles.jsonl"
        if battles.is_file():
            when = dt.datetime.fromtimestamp(battles.stat().st_mtime, dt.timezone.utc)
            source = "<data>/opponents/battles.jsonl (file time)"
    return {"pulled_at": gv.iso(when) if when else None, "source": source,
            "hours": round((now - when).total_seconds() / 3600, 1) if when else None}


def client_status(ledger: Mapping[str, Any], updates: Path, override: str | None
                  ) -> dict[str, Any]:
    """The device's client build against the ledger. ValueError for a bad --client."""
    if override is not None:
        seen = {"version_code": int(str(override).strip()), "source": "--client",
                "seen_at": None}
        gv.parse_code(seen["version_code"])
    else:
        seen = gv.last_seen_client(updates)
    if seen is not None:
        judged = gv.judge_client(ledger, seen["version_code"], updates)
        return {**judged, "seen": True, "seen_at": seen.get("seen_at"),
                "source": seen.get("source")}
    stand_in = gv.newest_accepted(ledger, updates)
    if stand_in is None:
        return {"seen": False}
    judged = gv.judge_client(ledger, stand_in["version_code"], updates)
    return {**judged, "seen": False, "source": "the newest accepted build in the ledger (no "
            "device read recorded yet: selfcheck and the live runner record one, U05)"}


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Options:
    updates: Path
    data_root: Path
    ledger: Path | None = None
    client: str | None = None
    cards: Path | None = None
    skip: frozenset[str] = frozenset()
    pre: bool = False
    max_sdists: int = MAX_SDISTS
    now: dt.datetime | None = None


def _pin_json(ledger: Mapping[str, Any]) -> dict[str, Any]:
    p = gv.pin(ledger)
    t = p["table"]
    return {"royalesim": p["royalesim"], "card_table": p["card_table"],
            "version": t.get("version"), "content": t.get("content"),
            "client_build": t.get("client_build"), "content_date": t.get("content_date"),
            "fnv1a64": t.get("fnv1a64"), "since": p.get("since")}


def engine_messages(rs: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    warn: list[str] = []
    info: list[str] = []
    for item in rs.get("newer") or []:
        if not item.get("read"):
            if item.get("error") and item.get("integrity") is not False:
                info.append(f"royalesim {item['version']}: could not read it ({item['error']})")
            elif item.get("note") and item.get("integrity") is not False:
                info.append(f"royalesim {item['version']}: {item['note']}")
            continue
        cmp_ = item.get("compare") or {}
        bits = []
        if cmp_.get("arms_added"):
            bits.append("new arms " + ", ".join(cmp_["arms_added"]))
        if cmp_.get("arms_removed"):
            bits.append("arms removed " + ", ".join(cmp_["arms_removed"]))
        for row in cmp_.get("tables") or []:
            if row["status"] == "bytes changed":
                bits.append(f"{row['arm']} ({row.get('version')}): same name, other bytes "
                            f"(fnv {row['old'].get('fnv1a64')} -> {row.get('fnv1a64')})")
            elif row["status"] == "new version":
                bits.append(f"{row['arm']}: table {row['old'].get('version')} -> "
                            f"{row.get('version')}")
        if (cmp_.get("default_arm") or {}).get("changed"):
            bits.append(f"engine default arm {cmp_['default_arm']['old']} -> "
                        f"{cmp_['default_arm']['new']}")
        newest = cmp_.get("newest_table") or {}
        if newest:
            bits.append(f"newest table {newest.get('arm')} (content {newest.get('content')}, "
                        f"client build {newest.get('client_build')})")
        if not cmp_.get("changed_tables") and not cmp_.get("arms_added"):
            bits.append("no table changed: engine only, promoting is optional")
        warn.append(f"royalesim {item['version']} on PyPI (uploaded {item.get('uploaded')}): "
                    + "; ".join(bits))
    for v in rs.get("yanked") or []:
        info.append(f"royalesim {v} is yanked on PyPI")
    for v in rs.get("pre") or []:
        info.append(f"royalesim {v} is a pre-release (left out; --pre reads it)")
    return warn, info


def map_messages(ri: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    warn: list[str] = []
    info: list[str] = []
    for item in ri.get("newer") or []:
        if item.get("integrity") is False:
            warn.append(f"royaleimitate {item['version']}: {item['error']}")
            continue
        if item.get("error"):
            info.append(f"royaleimitate {item['version']}: could not read it ({item['error']})")
            continue
        if item.get("same_as_vendored"):
            info.append(f"royaleimitate {item['version']}: replay_cards.json unchanged")
            continue
        added, removed, renamed = (item.get("added") or {}, item.get("removed") or [],
                                   item.get("renamed") or {})
        if added or removed or renamed:
            parts = []
            if added:
                parts.append("keys added " + ", ".join(f"{k} -> {v}" for k, v in added.items()))
            if removed:
                parts.append("keys removed " + ", ".join(removed))
            if renamed:
                parts.append("renamed " + ", ".join(f"{k}: {a} -> {b}"
                                                     for k, (a, b) in renamed.items()))
            warn.append(f"royaleimitate {item['version']} has a new card map ("
                        + "; ".join(parts) + "): take it as data/cards/README.md says")
        elif item.get("note"):
            info.append(f"royaleimitate {item['version']}: {item['note']}")
    return warn, info


def next_step(code: int, status: Mapping[str, Any]) -> str:
    if code == EXIT_UPDATE:
        return ("close the browser, then run scripts\\game_update.ps1 -Update "
                f"(royalesim {status['royalesim'].get('newest')})")
    if code == EXIT_NEW_CLIENT:
        c = status.get("client") or {}
        return (f"client {c.get('version_code')} is new: play one dry-run friendly; if nothing "
                "in it matters to the bot, run scripts\\game_update.ps1 -AcceptClient "
                "-Note \"why\"")
    if code == EXIT_ERROR:
        return "read the ERROR lines; the check runs again tomorrow"
    late = (status.get("calendar") or {}).get("behind")
    if late:
        return ("nothing to install yet: run scripts\\game_update.ps1 -Update when RoyaleSim "
                "ships a table for the events since " + late["since"])
    return "nothing to do"


def run_check(opts: Options) -> tuple[int, dict[str, Any]]:
    """The whole check. Writes status.json, status.txt and state.json under opts.updates and
    returns (exit code, status)."""
    now = opts.now or gv.utcnow()
    updates = opts.updates
    warnings: list[str] = []
    notes: list[str] = []
    errors: list[str] = []
    status: dict[str, Any] = {"schema": SCHEMA, "tool": f"{TOOL} check", "checked_at": gv.iso(now)}
    try:
        ledger = gv.load(opts.ledger)
    except gv.LedgerError as exc:
        status.update({"exit": EXIT_ERROR, "verdict": VERDICTS[EXIT_ERROR],
                       "errors": [f"ledger: {exc}"], "warnings": [], "notes": []})
        status["next"] = "fix data/game_versions.json (git checkout -- data/game_versions.json)"
        _write_status(updates, status, now)
        return EXIT_ERROR, status
    state = gv.read_json(updates / gv.STATE_JSON)
    state = state if isinstance(state, dict) else {}
    since = gv.parse_time(state.get("last_check"))
    status["pin"] = _pin_json(ledger)
    cache = updates / CACHE

    # 1. PyPI: royalesim
    if "pypi" in opts.skip:
        rs: dict[str, Any] = {"skipped": "--skip pypi", "newer": []}
    else:
        try:
            rs = check_engine(ledger, cache, pre=opts.pre, max_sdists=opts.max_sdists)
        except (CheckError, OSError) as exc:
            rs = {"error": str(exc), "newer": []}
            errors.append(f"PyPI: {exc}")
        errors += rs.get("errors") or []
        w, n = engine_messages(rs)
        warnings += w
        notes += n
    status["royalesim"] = rs
    update_available = any(item.get("integrity") is not False for item in rs.get("newer") or [])

    # 2. royaleimitate
    if "royaleimitate" in opts.skip:
        status["royaleimitate"] = {"skipped": "--skip royaleimitate"}
    else:
        try:
            ri = check_card_map(ledger, cache, pre=opts.pre, max_sdists=opts.max_sdists)
        except (CheckError, OSError) as exc:
            ri = {"error": str(exc)}
        if ri.get("error"):
            notes.append(f"royaleimitate: {ri['error']}")
        w, n = map_messages(ri)
        warnings += w
        notes += n
        status["royaleimitate"] = ri

    # 3. the roster radar
    if "radar" in opts.skip:
        rad: dict[str, Any] = {"skipped": "--skip radar"}
    else:
        try:
            rad = check_radar(updates, opts.data_root, opts.cards,
                              [i for i in rs.get("newer") or [] if i.get("read")], now)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            rad = {"error": f"{type(exc).__name__}: {exc}"}
        if rad.get("skipped"):
            notes.append(f"radar skipped: {rad['skipped']}")
        elif rad.get("error"):
            notes.append(f"radar: {rad['error']}")
        else:
            w, n = radar_messages(rad)
            warnings += w
            notes += n
            if rad.get("baseline"):
                notes.append("radar: first snapshot recorded; changes show from the next check")
    status["radar"] = rad

    # 4. the calendar
    cal = calendar(ledger, since, now)
    status["calendar"] = cal
    for e in cal["passed"]:
        warnings.append(f"{e['date']}: {gv.event_text(e)} (since the last check)")
    if cal["behind"]:
        b = cal["behind"]
        warnings.append(f"{b['days']} days behind: no pinned table has "
                        + "; ".join(f"{e['date']} {gv.event_text(e)}" for e in b["events"]))
        if b["days"] >= UPSTREAM_DAYS and not update_available:
            warnings.append(f"{b['days']} days with no RoyaleSim release for it: ask upstream "
                            "for a table (docs/SEASON_UPDATES.md 10; the PyPI page links "
                            "github.com/RoyaleGym/RoyaleSim and a Discord)")
    for e in cal["upcoming"]:
        notes.append(f"upcoming {e['date']}: {gv.event_text(e)}")

    # 5. the device's client build
    try:
        client = client_status(ledger, updates, opts.client)
    except ValueError as exc:
        client = {"seen": False, "error": str(exc)}
        errors.append(f"client: {exc}")
    # Section 4.1: a new build is accepted by itself only when the check of the day it appears
    # finds nothing else. "The day it appears" is the day a check first saw it (state.json
    # pending_client.first_checked); any later day it stays a WARN until -AcceptClient, so the
    # build that came with a season's new forms is never accepted the morning after. A change
    # seen by any check that day keeps the day noisy (noisy_on), so running the check again
    # the same day (the radar then compares with its own snapshot) does not accept it either.
    today = now.date().isoformat()
    noisy = bool(rs.get("newer") or rad.get("changed") or cal["passed"] or cal["today"])
    quiet = (not noisy and state.get("noisy_on") != today
             and not rs.get("skipped") and not rs.get("error")
             and not rad.get("skipped") and not rad.get("error") and not rad.get("baseline"))
    new_build = bool(client.get("seen") and client.get("verdict") in ("new", "unaccepted"))
    pending = state.get("pending_client") if isinstance(state.get("pending_client"), dict) \
        else {}
    first_checked = None
    if new_build:
        first_checked = (str(pending.get("first_checked") or "")
                         if pending.get("version_code") == client.get("version_code")
                         else "") or today
    if new_build and quiet and first_checked == today:
        entry = {"version_code": client["version_code"], "version_name": client["version_name"],
                 "first_seen": (gv.parse_date(client.get("seen_at")) or now.date()).isoformat(),
                 "accepted": now.date().isoformat(), "basis": "auto",
                 "note": (f"recorded by {TOOL} check on {now.date()}: no new table, no roster "
                          "or form change, no calendar event")}
        gv.record_local_client(entry, updates)
        client = {**client, "accepted": entry["accepted"], "basis": "auto", "where": "local",
                  "verdict": "ok", "auto_accepted": True}
        notes.append(f"client {client['version_code']} ({client['version_name']}) accepted by "
                     "itself (basis auto): nothing else changed today")
    elif new_build:
        why = ("not in data/game_versions.json" if client["verdict"] == "new"
               else "listed but never accepted")
        warnings.append(f"client {client['version_code']} ({client['version_name']}) is {why}; "
                        f"line {client.get('relation')} of table {client.get('table_content')}")
        client = {**client, "first_checked": first_checked}
        if first_checked != today:
            notes.append(f"client {client['version_code']} was first checked on {first_checked}, "
                         "a day that was not quiet: it is not accepted by itself "
                         "(scripts\\game_update.ps1 -AcceptClient -Note \"why\")")
    elif client.get("relation") == "ahead" and client.get("accepted"):
        notes.append(f"client {client.get('version_code')} is ahead of table "
                     f"{client.get('table_content')}, accepted {client['accepted']} "
                     f"({client.get('basis')})")
    status["client"] = client

    # 6. IL_Replay
    if "il_replay" in opts.skip:
        il: dict[str, Any] = {"skipped": "--skip il_replay"}
    else:
        il = check_il_replay(ledger, state)
        if il.get("error"):
            notes.append(f"IL_Replay: {il['error']} (not needed today)")
        elif il.get("changed"):
            notes.append(f"IL_Replay re-published: revision {il.get('sha')} (lastModified "
                         f"{il.get('last_modified')}); before {il.get('previous') or il.get('recorded')}."
                         " A re-published corpus is the only source of imitation rows for new "
                         "cards.")
        elif il.get("first_seen"):
            notes.append(f"IL_Replay revision {il.get('sha')} recorded (first check)")
    status["il_replay"] = il

    # 7. the opponent pool
    pool = pool_status(opts.data_root, now)
    status["pool"] = pool
    if pool["hours"] is not None and pool["hours"] > POOL_STALE_H:
        warnings.append(f"the opponent pool was last pulled {gv.age_text(pool['hours'] * 3600)}")

    if update_available:
        code = EXIT_UPDATE
    elif errors:
        code = EXIT_ERROR
    elif client.get("seen") and client.get("verdict") in ("new", "unaccepted"):
        code = EXIT_NEW_CLIENT
    else:
        code = EXIT_OK
    status.update({"exit": code, "verdict": VERDICTS[code], "warnings": warnings,
                   "notes": notes, "errors": errors})
    status["next"] = next_step(code, status)
    _write_status(updates, status, now)
    new_state = dict(state)
    new_state.update({"schema": SCHEMA, "last_check": gv.iso(now), "last_exit": code})
    if noisy:
        new_state["noisy_on"] = today
    if new_build and not client.get("auto_accepted"):
        new_state["pending_client"] = {"version_code": client.get("version_code"),
                                       "first_checked": first_checked}
    else:
        new_state.pop("pending_client", None)
    if il.get("sha"):
        new_state["il_replay"] = {"sha": il["sha"], "last_modified": il.get("last_modified"),
                                  "seen_at": gv.iso(now)}
    if not rs.get("skipped") and not rs.get("error"):
        new_state["royalesim_newest"] = rs.get("newest")
    if rad.get("fetched_at"):
        new_state["radar"] = {"snapshot": CARDS_SNAPSHOT, "fetched_at": rad["fetched_at"]}
    gv.write_json(updates / gv.STATE_JSON, new_state)
    return code, status


def status_text(status: Mapping[str, Any], now: dt.datetime | None = None) -> str:
    lines = [status.get("line") or gv.currency_line(status, now), ""]
    lines += [f"ERROR {e}" for e in status.get("errors") or []]
    lines += [f"WARN  {w}" for w in status.get("warnings") or []]
    lines += [f"INFO  {n}" for n in status.get("notes") or []]
    lines += ["", f"exit {status.get('exit')}: {status.get('verdict')}; next: "
              f"{status.get('next')}"]
    return "\n".join(lines) + "\n"


def _write_status(updates: Path, status: dict[str, Any], now: dt.datetime) -> None:
    status["line"] = gv.currency_line(status, now)
    gv.write_json(updates / gv.STATUS_JSON, status)
    gv.write_text(updates / gv.STATUS_TXT, status_text(status, now))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parts(text: str) -> frozenset[str]:
    got = {p.strip() for p in str(text).split(",") if p.strip()}
    bad = sorted(got - set(PARTS))
    if bad:
        raise argparse.ArgumentTypeError(f"{', '.join(bad)}: the parts are {', '.join(PARTS)}")
    return frozenset(got)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="game_update.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="the daily version check (HTTPS only, no device)")
    c.add_argument("--offline", action="store_true",
                   help="no device (the only mode: check never calls adb)")
    c.add_argument("--updates", type=Path, default=None,
                   help="where status, state and cache live (default <runs>/updates)")
    c.add_argument("--data", type=Path, default=None,
                   help="the data root, for <data>/opponents (default the machine's)")
    c.add_argument("--ledger", type=Path, default=None, help="another ledger file")
    c.add_argument("--client", metavar="CODE", default=None,
                   help="the device's versionCode (default <updates>/last_seen_client.json)")
    c.add_argument("--cards", type=Path, default=None, metavar="FILE",
                   help="a saved /v1/cards response for the radar (no API call)")
    # nargs="+": in PowerShell 5.1 an unquoted pypi,radar is an array, which reaches python.exe
    # as two arguments; "pypi,radar", "pypi radar" and two --skip all read the same.
    c.add_argument("--skip", type=_parts, nargs="+", action="extend", default=[],
                   metavar="PARTS",
                   help=f"parts not to check, comma- or space-separated: {', '.join(PARTS)}")
    c.add_argument("--pre", action="store_true", help="count pre-releases on PyPI")
    c.add_argument("--max-sdists", type=int, default=MAX_SDISTS, metavar="N",
                   help=f"read at most the N newest new releases (default {MAX_SDISTS})")
    c.add_argument("--json", action="store_true", help="print status.json instead")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command == "check":
        if args.max_sdists < 0:
            ap.error("--max-sdists: 0 or more")
        if args.client is not None:
            try:
                gv.parse_code(args.client)
            except ValueError as exc:
                ap.error(f"--client: {exc}")
        opts = Options(updates=args.updates if args.updates is not None else gv.updates_dir(),
                       data_root=args.data if args.data is not None else gv.data_root(),
                       ledger=args.ledger, client=args.client, cards=args.cards,
                       skip=frozenset().union(*args.skip), pre=args.pre,
                       max_sdists=args.max_sdists)
        code, status = run_check(opts)
        if args.json:
            print(json.dumps(status, indent=1))
        else:
            print(status_text(status), end="")
        return code
    ap.error(f"unknown command {args.command}")
    return EXIT_ARGS


if __name__ == "__main__":
    raise SystemExit(main())
