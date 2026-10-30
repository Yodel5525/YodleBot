r"""The official Clash Royale API: one client for every tool that reads it (U02).

    .venvs\royalegym\Scripts\python.exe tools\cr_api.py cards
    .venvs\royalegym\Scripts\python.exe tools\cr_api.py cards --base proxy --transport curl

`cards` makes one call, GET /v1/cards. It checks the key, the slug rule and the client on the
PC. It prints the HTTP status and the card count. It shows which form fields the roster carries
(maxEvolutionLevel, icon keys) and which API names have no key in the repo's card map. It also
reports the response headers this module cannot know in advance (UNCONFIRMED below). Exit codes:
0 ok; 1 the API answered with an error or nothing came back; 2 a bad setting, such as no token
or a bad --base.

Spec: docs/SEASON_UPDATES.md sections 7.1 (U02 bullets), 7.3 (the slug rule), 7.7 (courtesy and
privacy), 12 (U02) and 13.1 (the API key). The shapes and behaviours below come from the sources
that section 15 cites. Nothing here has been run against the live API from the sandbox that
wrote it.

* https://gist.github.com/loganlinn/d813ba0f9ccb47f34462ec8abeab8e14 is Swagger 2.0 for
  api.clashroyale.com/v1. Auth is `authorization: Bearer <JWT>`. A tag in a path is
  URL-encoded ("#2ABC" -> "%232ABC"). List responses wrap their results in "items", with a
  "paging" marker. Errors are 400, 403, 404, 429, 500 and 503, with a body of
  {"reason", "message"}. 429 is "Request was throttled, because amount of requests was above
  the threshold defined for the used API token". 503 is maintenance. No rate-limit headers and
  no Retry-After are documented.
* https://github.com/RoyaleAPI/cr-api-docs/blob/master/docs/proxy.md covers RoyaleAPI's proxy.
  Replace https://api.clashroyale.com with https://proxy.royaleapi.dev and put 45.79.218.79
  on the key. The token then passes through a third party (section 13.1).
* https://raw.githubusercontent.com/samaksh-bajaj/cr-deck-builder/main/API_NOTES.md says the
  proxy 403s Python's default urllib User-Agent and that any custom User-Agent fixes it. It
  also says evolutionLevel / maxEvolutionLevel are bitmasks (1 evolution, 2 hero), that
  /cards served 123 cards, and that a deck with a hero can come back with 7 cards.
* tools/ladder_pull.py (this repo, before U02): the key is bound to one IPv4 address, so a
  changed address fails every call with accessDenied.invalidIp. The client forces IPv4 as
  ladder_pull's `curl -4` did.

What the client does:

* `Client.get` returns a `Response` with the HTTP status for every outcome: 200, 403, 404,
  429, 503, or 0 when no response came back. It never exits or raises for an HTTP error. A
  caller that wants to stop, as tools/ladder_pull.py does on 403, decides that itself, and
  `Response.hint()` says what a status usually means.
* 429 backs off. It waits the response's Retry-After (seconds or an HTTP date), or 2, 4, 8
  and 16 s without one, up to RETRIES times. The wait is put on the shared `Rate`, so every
  thread pauses, not only the one that was refused. A Retry-After longer than MAX_WAIT is not
  waited out: the 429 is returned at once rather than asked again and again. 500, 502, 504
  and "no response" are tried once more after TRANSIENT_WAIT. 503 (maintenance) is not
  retried.
* The User-Agent is always USER_AGENT, never Python's or curl's default.
* The base is the official API, RoyaleAPI's proxy, or another https URL ending in /v1. It
  comes from `--base` / `base=` or HASTYCR_CR_API_BASE. Plain http is refused, because the
  token travels in a header.
* The token is read from .secrets/supercell_token (gitignored), BOM and line end stripped. It
  never appears in an argv, a log line, a repr or an error message. The curl transport passes
  it on stdin (`curl -K -`).
* Rate: DEFAULT_RATE, 2 requests per second, shared by all threads (section 4.1 plans about
  1,000 calls in about 8 minutes at 2 per second).
* `slug` is the one card-name rule (section 7.3): lower case, dots dropped, every other run of
  non-alphanumerics becomes "-", with no "-" at either end. It turns an API display name into
  the RoyaleAPI key that data/cards (tools/rs_cards.py) and IL_Replay use: "Mini P.E.K.K.A" ->
  "mini-pekka", "X-Bow" -> "x-bow", "The Log" -> "the-log". It reproduces all 120 keys of
  data/royaleapi/cards.json, and leaves every key of data/cards/replay_cards.json unchanged.

Privacy (section 7.7): this module writes no files. Log lines show a path with its tag
redacted. The curl transport's temporary files are deleted after each call. Pulled data, with
names and tags, belongs in gitignored folders (tools/ladder_pull.py output; <data>/opponents/
for U03).

Transports: "urllib" (the default here) is Python's urllib over IPv4, with gzip. "curl" is
curl -4 --compressed, the path tools/ladder_pull.py has always used (curl.exe ships with
Windows 10 and later). A transport is any callable (url, headers, timeout) -> Raw, which is
how the tests replace the network.

UNCONFIRMED lists what the documented shapes do not settle. `cards` reports the items it can
see. The battle-log items are for U03's `probe` (tools/opponents.py), after a friendly in which
the main account plays an evolution in slot 1 and a hero in slot 2 (section 7.1).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import email.utils
import gzip
import http.client
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOKEN_FILE = ROOT / ".secrets" / "supercell_token"

OFFICIAL_BASE = "https://api.clashroyale.com/v1"
#: RoyaleAPI's proxy (cr-api-docs proxy.md). The key must list PROXY_IP.
PROXY_BASE = "https://proxy.royaleapi.dev/v1"
PROXY_IP = "45.79.218.79"
BASE_ENV = "HASTYCR_CR_API_BASE"
BASES = {"official": OFFICIAL_BASE, "proxy": PROXY_BASE}

#: Sent on every call. The proxy refuses Python's default (API_NOTES.md); a name that says what
#: is calling is the courteous default for the official API as well.
USER_AGENT = ("yodlebot-sparring/1.0 (Clash Royale friendly-battle sparring research; "
              "tools/cr_api.py; low request rate)")

DEFAULT_RATE = 2.0          # requests per second, across all threads
DEFAULT_TIMEOUT = 30.0      # seconds per request
RETRIES = 4                 # 429 retries before the 429 is returned
BACKOFF = 2.0               # first wait without Retry-After; doubles per retry: 2, 4, 8, 16 s
MIN_WAIT = 1.0              # the shortest 429 wait, also for "Retry-After: 0"
MAX_WAIT = 120.0            # a longer Retry-After is not waited out: the 429 is returned
TRANSIENT = frozenset({0, 500, 502, 504})   # tried once more; 503 is maintenance, not retried
TRANSIENT_WAIT = 5.0

#: Exit codes of the CLI.
EXIT_OK, EXIT_API, EXIT_SETTING = 0, 1, 2

#: What the documented shapes leave open, checked on the PC (section 7.1). `cards` reports the
#: first group; U03's `probe` reports the battle-log group.
UNCONFIRMED: dict[str, str] = {
    # the client (tools/cr_api.py cards)
    "retry_after": "whether a 429 carries Retry-After, in seconds or as an HTTP date; without it "
                   "the client waits 2, 4, 8, 16 s",
    "rate_headers": "whether responses carry rate-limit headers; the Swagger spec documents none",
    "throttle_reason": "the 429 body's reason string; the fixtures use requestThrottled",
    "gzip": "whether the API answers Accept-Encoding: gzip with a gzip body",
    "ipv6": "whether api.clashroyale.com is reachable over IPv6; the client always uses IPv4, "
            "because the key is bound to an IPv4 address",
    "proxy_user_agent": "whether proxy.royaleapi.dev still refuses Python's default "
                        "User-Agent; the client always sends USER_AGENT",
    "support_items": "whether /cards lists tower troops under supportItems",
    "slug_other_names": "the RoyaleAPI key of a name with an apostrophe or an accent; no card "
                        "has one today",
    # the battle log (tools/opponents.py probe, U03)
    "battlelog_forms": "whether battle-log cards carry the form played (evolutionLevel on the "
                       "card), or only what the player owns",
    "hero_slot_missing": "whether a hero-slot card goes missing from a battle-log deck (77 of "
                         "300 top players' currentDeck had 7 cards)",
    "battle_types": "the battle type strings and the deckSelection value of a friendly",
    "log_length": "how many battles one battle log holds (about 25)",
    "friendly_levels": "the card levels a friendly shows (the game plays every card at 11)",
}


class TokenError(ValueError):
    """The token file is missing, empty or not a single token."""


# ---------------------------------------------------------------------------
# card names and tags
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slug(name: str) -> str:
    """API display name -> RoyaleAPI card key, as data/cards and IL_Replay spell it: lower case,
    dots dropped, every other run of non-alphanumerics -> "-", none at the ends."""
    text = str(name).strip().lower().replace(".", "")
    return _NON_ALNUM.sub("-", text).strip("-")


_TAG = re.compile(r"#[0-9A-Z]+")


def norm_tag(tag: str) -> str:
    """A player or clan tag as the API writes it: upper case, leading "#", O read as 0 (tags have
    no letter O, and a typed one is a zero). A "%23" prefix is read as "#"."""
    text = str(tag).strip().upper()
    if text.startswith("%23"):
        text = "#" + text[3:]
    if not text.startswith("#"):
        text = "#" + text
    text = text.replace("O", "0")
    if not _TAG.fullmatch(text):
        raise ValueError(f"{tag!r} is not a player or clan tag")
    return text


def tag_path(tag: str) -> str:
    """A tag as a URL path segment: "#2PYQ0" -> "%232PYQ0" (the spec's encoding)."""
    return urllib.parse.quote(norm_tag(tag), safe="")


def redact(text: str) -> str:
    """A path or URL with every encoded or plain tag replaced, for log lines."""
    return re.sub(r"(%23|#)[0-9A-Za-z]+", r"\1<tag>", str(text))


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def resolve_base(value: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """The API base URL. `value` wins, then HASTYCR_CR_API_BASE, then the official API. It may be
    "official", "proxy" or an https URL ending in /v1 (a trailing "/" is dropped)."""
    text = value if value not in (None, "") else (os.environ if env is None else env).get(
        BASE_ENV, "")
    text = str(text).strip()
    if text.lower() in ("", "official"):
        return OFFICIAL_BASE
    if text.lower() == "proxy":
        return PROXY_BASE
    text = text.rstrip("/")
    if not text.lower().startswith("https://"):
        raise ValueError(f"API base {text!r}: only https is allowed (the token is a header)")
    if not text.endswith("/v1"):
        raise ValueError(f"API base {text!r} must end with /v1, e.g. {PROXY_BASE}")
    return text


def base_kind(base: str) -> str:
    """'official', 'proxy' or 'custom'."""
    for kind, url in BASES.items():
        if base == url:
            return kind
    return "custom"


def check_token(token: str, where: str = "the token") -> str:
    """The token with surrounding whitespace removed. Raises TokenError when it is empty or holds
    whitespace, quotes, backslashes or control characters (a pasted second line, say). Such a
    token cannot be a header value, and it is not what developer.clashroyale.com issues."""
    text = str(token).strip()
    if not text:
        raise TokenError(f"{where} is empty")
    if any(ch.isspace() or ch in "\"'\\" or ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise TokenError(f"{where} is not one token: it holds whitespace, quotes or control "
                         "characters. Paste only the key from developer.clashroyale.com")
    return text


def read_token(path: str | os.PathLike | None = None) -> str:
    """The API token from `path` (default .secrets/supercell_token). A BOM and line ends from a
    Windows editor are stripped."""
    path = Path(TOKEN_FILE if path is None else path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise TokenError(f"no token at {path}") from None
    except OSError as exc:
        raise TokenError(f"cannot read the token at {path}: {exc.strerror or exc}") from None
    return check_token(text, f"the token at {path}")


# ---------------------------------------------------------------------------
# rate
# ---------------------------------------------------------------------------

class Rate:
    """A simple ceiling on requests per second, shared across threads.

    The developer tier is throttled and the documented limit is not published as a number, so
    this stays well under anything plausible rather than probing for it. `hold` pushes every
    thread's next request back, which is how a 429 pauses the whole pull. (Moved here from
    tools/ladder_pull.py, where the name still imports.)
    """

    def __init__(self, per_second: float, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.gap = 1.0 / max(0.1, per_second)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            due = max(now, self._next)
            self._next = due + self.gap
        if due > now:
            self._sleep(due - now)

    def hold(self, seconds: float) -> None:
        """No request starts within `seconds` from now, on any thread."""
        with self._lock:
            self._next = max(self._next, self._clock() + max(0.0, float(seconds)))


# ---------------------------------------------------------------------------
# transports
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Raw:
    """What a transport returns: the status (0 when no response came back), lower-cased
    headers, the decoded body bytes and, for status 0, why."""
    status: int
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    body: bytes = b""
    error: str = ""


Transport = Callable[[str, Mapping[str, str], float], Raw]


def _connect_ipv4(address, *args, **kwargs):
    """socket.create_connection for IPv4 addresses only. The key is bound to an IPv4 address, so
    a connection over IPv6 would fail with accessDenied.invalidIp."""
    host, port = address[0], address[1]
    last: OSError | None = None
    for _family, _type, _proto, _name, sockaddr in socket.getaddrinfo(
            host, port, socket.AF_INET, socket.SOCK_STREAM):
        try:
            return socket.create_connection(sockaddr[:2], *args, **kwargs)
        except OSError as exc:
            last = exc
    raise last or OSError(f"no IPv4 address for {host}")


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection whose TCP connection uses IPv4. TLS still checks the host name."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_ipv4


class _IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_IPv4HTTPSConnection, req, context=self._context)


def _lower_headers(items: Iterable[tuple[str, str]] | None) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (items or ())}


class UrllibTransport:
    """Python's urllib, over IPv4 unless ipv4=False, asking for gzip and decoding it."""

    def __init__(self, *, ipv4: bool = True):
        handlers = [_IPv4HTTPSHandler()] if ipv4 else []
        self._opener = urllib.request.build_opener(*handlers)

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> Raw:
        req = urllib.request.Request(url, headers={**headers, "Accept-Encoding": "gzip"})
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", None) or resp.getcode())
                got = _lower_headers(resp.headers.items())
                body = resp.read()
        except urllib.error.HTTPError as exc:          # 4xx / 5xx: still a response
            status = int(exc.code)
            got = _lower_headers(exc.headers.items() if exc.headers is not None else None)
            try:
                body = exc.read() or b""
            except (OSError, ValueError, AttributeError, http.client.HTTPException):
                body = b""
        except urllib.error.URLError as exc:
            return Raw(0, error=f"no response: {exc.reason}")
        except (OSError, http.client.HTTPException, ValueError) as exc:
            return Raw(0, error=f"no response: {type(exc).__name__}: {exc}")
        if got.get("content-encoding", "").lower() == "gzip" and body:
            try:
                body = gzip.decompress(body)
            except (OSError, EOFError, zlib.error) as exc:
                return Raw(status, got, b"", f"bad gzip body: {exc}")
        return Raw(status, got, body)


def _curl_quote(text: str) -> str:
    if "\r" in text or "\n" in text:
        raise ValueError("a header value holds a line break")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _parse_head(text: str) -> dict[str, str]:
    """The headers of the last response in a `curl -D` dump (a proxy's CONNECT reply or a
    100 Continue comes first when there is one)."""
    blocks = [b for b in re.split(r"\r?\n\r?\n", text) if b.strip().startswith("HTTP/")]
    if not blocks:
        return {}
    out: dict[str, str] = {}
    for line in blocks[-1].strip().splitlines()[1:]:
        name, sep, value = line.partition(":")
        if sep:
            out[name.strip().lower()] = value.strip()
    return out


class CurlTransport:
    """curl -4 -sS --compressed, as tools/ladder_pull.py always called it. The headers, the
    token among them, go in on stdin as a curl config (`-K -`), so no argv carries the token.
    Body and headers land in per-thread files under `scratch`, deleted after each call."""

    def __init__(self, *, ipv4: bool = True, scratch: str | os.PathLike | None = None,
                 curl: str = "curl"):
        self.ipv4 = ipv4
        self.curl = curl
        self._scratch = Path(scratch) if scratch is not None else None
        self._lock = threading.Lock()

    def scratch(self) -> Path:
        with self._lock:
            if self._scratch is None:
                self._scratch = Path(tempfile.mkdtemp(prefix="cr_api_"))
            self._scratch.mkdir(parents=True, exist_ok=True)
            return self._scratch

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> Raw:
        folder = self.scratch()
        ident = threading.get_ident()
        body_path, head_path = folder / f"resp_{ident}.json", folder / f"head_{ident}.txt"
        config = "".join(f"header = {_curl_quote(f'{k}: {v}')}\n" for k, v in headers.items())
        argv = [self.curl, *(["-4"] if self.ipv4 else []), "-sS", "--compressed",
                "--max-time", str(max(1, int(math.ceil(timeout)))),
                "-o", str(body_path), "-D", str(head_path), "-w", "%{http_code}",
                "-K", "-", url]
        try:
            done = subprocess.run(argv, input=config, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout + 15)
            code = (done.stdout or "").strip()
            status = int(code) if code.isdigit() else 0
            if status == 0:
                why = (done.stderr or "").strip() or f"curl exit {done.returncode}"
                return Raw(0, error=f"no response: {why}")
            try:
                got = _parse_head(head_path.read_text(encoding="latin-1"))
            except OSError:
                got = {}
            try:
                body = body_path.read_bytes()
            except OSError:
                body = b""
            return Raw(status, got, body)
        except FileNotFoundError:
            return Raw(0, error=f"no response: {self.curl} was not found")
        except subprocess.TimeoutExpired:
            return Raw(0, error="no response: curl timed out")
        finally:
            for path in (body_path, head_path):
                try:
                    path.unlink()
                except OSError:
                    pass


def make_transport(spec: str | Transport | None = None, *,
                   scratch: str | os.PathLike | None = None) -> Transport:
    """A transport from a name ("urllib", the default, or "curl") or a callable, passed through."""
    if callable(spec):
        return spec
    name = (spec or "urllib").strip().lower()
    if name == "urllib":
        return UrllibTransport()
    if name == "curl":
        return CurlTransport(scratch=scratch)
    raise ValueError(f"unknown transport {spec!r}: use urllib or curl")


# ---------------------------------------------------------------------------
# responses and the client
# ---------------------------------------------------------------------------

def retry_after_seconds(value: str | None, now: float | None = None) -> float | None:
    """A Retry-After header in seconds from now: a number of seconds, or an HTTP date. None when
    absent or unreadable."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:                     # "-0000": a UTC time, the RFC says
            when = when.replace(tzinfo=datetime.timezone.utc)
        seconds = when.timestamp() - (time.time() if now is None else now)
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


@dataclasses.dataclass
class Response:
    """One API answer. `status` is the HTTP status, or 0 when nothing came back. `data` is the
    parsed JSON body (errors have one too), or None. `reason` and `message` are an error body's
    fields, such as accessDenied.invalidIp. `path` has its tags redacted."""
    status: int
    data: Any = None
    reason: str = ""
    message: str = ""
    error: str = ""
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    attempts: int = 1
    retry_after: float | None = None
    path: str = ""
    base: str = OFFICIAL_BASE

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error and self.data is not None

    def hint(self) -> str:
        """What this answer usually means, and the fix."""
        said = f" The API said: {self.reason}: {self.message}" if self.reason or self.message \
            else ""
        if self.ok:
            return "ok"
        if self.status == 0:
            return f"No response ({self.error}). Check the network."
        if self.status == 200:
            return f"A 200 without a usable body ({self.error or 'empty'})."
        if self.status == 403:
            if self.reason == "accessDenied.invalidIp":
                where = (f"RoyaleAPI's proxy: the key must list {PROXY_IP}"
                         if base_kind(self.base) == "proxy" else
                         "The key is bound to one public IP, and this machine's has probably "
                         "changed; the message names the address the API saw. Edit the key at "
                         "developer.clashroyale.com rather than making a new one, or use "
                         f"--base proxy with {PROXY_IP} on the key")
                return f"403: {where}.{said}"
            return ("403: access denied. The token in .secrets/supercell_token is wrong, revoked "
                    "or for another IP; through the proxy, the User-Agent can also be refused."
                    + said)
        if self.status == 404:
            return f"404: not found. Check the tag.{said}"
        if self.status == 429:
            wait = ("" if self.retry_after is None
                    else f" Retry-After was {self.retry_after:.0f} s.")
            return (f"429: throttled after {self.attempts} attempts.{wait} Lower the rate "
                    f"or wait.{said}")
        if self.status == 503:
            return f"503: the API is in maintenance (common during a game update). Try later.{said}"
        return f"HTTP {self.status}.{said}"


def _response(raw: Raw, *, path: str, base: str, attempts: int) -> Response:
    data, error = None, raw.error
    if raw.body:
        try:
            data = json.loads(raw.body)
        except (UnicodeDecodeError, ValueError):
            if raw.status == 200 and not error:
                error = "the body is not JSON"
    elif raw.status == 200 and not error:
        error = "empty body"
    reason = message = ""
    if raw.status != 200 and isinstance(data, dict):
        reason, message = str(data.get("reason") or ""), str(data.get("message") or "")
    retry_after = retry_after_seconds(raw.headers.get("retry-after")) if raw.status == 429 \
        else None
    return Response(raw.status, data, reason, message, error, dict(raw.headers), attempts,
                    retry_after, path, base)


class Client:
    """The official API (or the proxy), with a shared rate, 429 back-off and a custom
    User-Agent. Thread-safe: share one client between worker threads."""

    def __init__(self, token: str | None = None, *, token_file: str | os.PathLike | None = None,
                 base: str | None = None, per_second: float = DEFAULT_RATE,
                 rate: Rate | None = None, transport: str | Transport | None = None,
                 user_agent: str = USER_AGENT, timeout: float = DEFAULT_TIMEOUT,
                 retries: int = RETRIES, backoff: float = BACKOFF, max_wait: float = MAX_WAIT,
                 transient_wait: float = TRANSIENT_WAIT,
                 env: Mapping[str, str] | None = None,
                 log: Callable[[str], None] | None = None):
        self._token = check_token(token) if token is not None else read_token(token_file)
        self.base = resolve_base(base, env)
        self.rate = rate if rate is not None else Rate(per_second)
        self.transport = transport if callable(transport) else make_transport(transport)
        self.user_agent = str(user_agent).strip() or USER_AGENT
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.backoff = float(backoff)
        self.max_wait = float(max_wait)
        self.transient_wait = float(transient_wait)
        self.log = log
        self._lock = threading.Lock()
        self.statuses: Counter = Counter()      # every status seen, retries included
        self.retry_after_seen: list[float] = []
        self.calls = 0

    def __repr__(self) -> str:                  # never the token
        return f"Client(base={self.base!r}, rate={1 / self.rate.gap:g}/s)"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json",
                "User-Agent": self.user_agent}

    def url(self, path: str, params: Mapping[str, Any] | None = None) -> str:
        url = f"{self.base}/{str(path).lstrip('/')}"
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})
        return url

    def _note(self, text: str) -> None:
        if self.log is not None:
            self.log(text)

    def _send(self, url: str) -> Raw:
        try:
            return self.transport(url, self.headers(), self.timeout)
        except (OSError, ValueError, http.client.HTTPException,
                subprocess.SubprocessError) as exc:
            return Raw(0, error=f"no response: {type(exc).__name__}: {exc}")

    def _throttle_wait(self, resp: Response, done: int) -> float | None:
        """Seconds to wait before retry number done + 1, or None to return the 429 now."""
        if resp.retry_after is not None:
            if resp.retry_after > self.max_wait:
                return None
            return max(MIN_WAIT, resp.retry_after)
        return max(MIN_WAIT, min(self.max_wait, self.backoff * 2 ** done))

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Response:
        """GET base/path?params: the answer with its status, after any 429 back-off. Never raises
        for an HTTP error."""
        url = self.url(path, params)
        shown = redact(url[len(self.base):])
        attempts = throttles = 0
        transient_left = 1
        while True:
            self.rate.wait()
            attempts += 1
            resp = _response(self._send(url), path=shown, base=self.base, attempts=attempts)
            with self._lock:
                self.calls += 1
                self.statuses[resp.status] += 1
                if resp.retry_after is not None:
                    self.retry_after_seen.append(resp.retry_after)
            if resp.status == 429:
                wait = self._throttle_wait(resp, throttles)
                if wait is None or throttles >= self.retries:
                    self._note(f"429 on {shown}: giving up after {attempts} attempts")
                    return resp
                throttles += 1
                self._note(f"429 on {shown}: waiting {wait:.0f} s "
                           f"(retry {throttles}/{self.retries})")
                self.rate.hold(wait)
                continue
            if resp.status in TRANSIENT and transient_left > 0:
                transient_left -= 1
                self._note(f"{resp.status or 'no response'} on {shown}: trying once more "
                           f"in {self.transient_wait:.0f} s")
                self.rate.hold(self.transient_wait)
                continue
            return resp

    def paged(self, path: str, *, want: int, page_size: int = 1000,
              params: Mapping[str, Any] | None = None) -> tuple[list[Any], Response | None]:
        """Up to `want` items of a list endpoint, following paging.cursors.after. Returns the
        items and the last response (None when want <= 0). It stops early on an error, an empty
        page or a missing cursor; check the last response's status for why."""
        items: list[Any] = []
        after = None
        last: Response | None = None
        while len(items) < want:
            query = dict(params or {})
            query["limit"] = min(page_size, want - len(items))
            if after:
                query["after"] = after
            last = self.get(path, query)
            data = last.data if last.ok else None
            if not isinstance(data, dict) or not data.get("items"):
                break
            items.extend(data["items"])
            paging = data.get("paging") if isinstance(data.get("paging"), dict) else {}
            cursors = paging.get("cursors") if isinstance(paging.get("cursors"), dict) else {}
            after = cursors.get("after")
            if not after:
                break
        return items[:max(0, want)], last

    # endpoints (paths as in the Swagger spec)
    def cards(self) -> Response:
        return self.get("cards")

    def player(self, tag: str) -> Response:
        return self.get(f"players/{tag_path(tag)}")

    def battlelog(self, tag: str) -> Response:
        return self.get(f"players/{tag_path(tag)}/battlelog")

    def clan(self, tag: str) -> Response:
        return self.get(f"clans/{tag_path(tag)}")

    def clan_members(self, tag: str) -> Response:
        return self.get(f"clans/{tag_path(tag)}/members")


# ---------------------------------------------------------------------------
# the roster check
# ---------------------------------------------------------------------------

def cards_report(payload: Any, known_keys: Iterable[str] | None = None) -> dict[str, Any]:
    """What a /cards answer says: the count, maxEvolutionLevel values (1 evolution, 2 hero,
    3 both), the icon keys, and slug collisions. With `known_keys` (the card map's keys), it also
    lists the names whose slug has no key there."""
    body = payload if isinstance(payload, dict) else {}
    items = [c for c in body.get("items") or [] if isinstance(c, dict)]
    support = body.get("supportItems")
    levels: Counter = Counter()
    without = 0
    icons: set[str] = set()
    names_of: dict[str, list[str]] = {}
    for card in items:
        name = str(card.get("name") or "")
        names_of.setdefault(slug(name), []).append(name)
        level = card.get("maxEvolutionLevel")
        if isinstance(level, int) and not isinstance(level, bool):
            levels[level] += 1
        else:
            without += 1
        if isinstance(card.get("iconUrls"), dict):
            icons |= set(card["iconUrls"])
    report: dict[str, Any] = {
        "items": len(items),
        "support_items": len(support) if isinstance(support, list) else None,
        "max_evolution_level": {str(k): n for k, n in sorted(levels.items())},
        "without_max_evolution_level": without,
        "icon_keys": sorted(icons),
        "collisions": {k: v for k, v in sorted(names_of.items()) if len(v) > 1},
    }
    if known_keys is not None:
        known = set(known_keys)
        unmapped = [{"name": names[0], "key": key} for key, names in sorted(names_of.items())
                    if key not in known]
        report["mapped"] = len(names_of) - len(unmapped)
        report["unmapped"] = unmapped
    return report


def header_report(headers: Mapping[str, str]) -> dict[str, Any]:
    """The headers that settle UNCONFIRMED items: content-encoding and anything about rates."""
    rate = sorted(k for k in headers if re.search(r"rate|limit|retry|throttl", k))
    return {"content_encoding": headers.get("content-encoding", ""), "rate_headers": rate,
            "names": sorted(headers)}


def _card_map_keys() -> tuple[set[str] | None, str]:
    """The keys of tools/rs_cards.load(), or None and why not."""
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    try:
        import rs_cards
        return set(rs_cards.load()), ""
    except (ImportError, ValueError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _print_cards(client: Client, resp: Response) -> None:
    keys, why = _card_map_keys()
    rep = cards_report(resp.data, keys)
    support = "" if rep["support_items"] is None else f", {rep['support_items']} support items"
    print(f"cards     {rep['items']} cards{support}")
    levels = ", ".join(f"{k}: {n}" for k, n in rep["max_evolution_level"].items()) or "none"
    print(f"forms     maxEvolutionLevel {levels}; {rep['without_max_evolution_level']} without it")
    print(f"icons     {', '.join(rep['icon_keys']) or 'none'}")
    if keys is None:
        print(f"card map  not read ({why})")
    else:
        names = "; ".join(f"{u['name']} -> {u['key']}" for u in rep["unmapped"]) or "none"
        print(f"card map  {rep['mapped']} of {rep['mapped'] + len(rep['unmapped'])} names slug "
              f"to a key in data/cards; no key: {names}")
    for key, names in rep["collisions"].items():
        print(f"COLLISION {key}: {', '.join(names)}")
    heads = header_report(resp.headers)
    print(f"headers   content-encoding {heads['content_encoding'] or 'none'}; rate-limit "
          f"headers: {', '.join(heads['rate_headers']) or 'none'}")
    seen = dict(sorted(client.statuses.items()))
    after = (f"Retry-After seen: {client.retry_after_seen}" if client.retry_after_seen
             else "no Retry-After seen")
    print(f"calls     {client.calls}, statuses {seen}; {after}")
    print("unconfirmed, for U03's probe: " + ", ".join(
        k for k in ("battlelog_forms", "hero_slot_missing", "battle_types", "log_length",
                    "friendly_levels")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cards = sub.add_parser("cards", help="GET /v1/cards: key, slug rule and client check")
    cards.add_argument("--base", default=None,
                       help=f"official (default), proxy ({PROXY_BASE}, with {PROXY_IP} on the "
                            f"key) or an https URL ending in /v1; {BASE_ENV} sets the default")
    cards.add_argument("--transport", choices=("urllib", "curl"), default="urllib")
    cards.add_argument("--token-file", type=Path, default=None,
                       help="default .secrets/supercell_token")
    cards.add_argument("--rate", type=float, default=DEFAULT_RATE)
    args = ap.parse_args(argv)
    try:
        client = Client(token_file=args.token_file, base=args.base, transport=args.transport,
                        per_second=args.rate, log=lambda m: print(f"  {m}", flush=True))
    except ValueError as exc:                   # TokenError is a ValueError
        print(f"cr_api: {exc}", file=sys.stderr)
        return EXIT_SETTING
    started = time.monotonic()
    resp = client.cards()
    print(f"GET {client.base}/cards: {resp.status or 'no response'} in "
          f"{time.monotonic() - started:.1f} s ({base_kind(client.base)})")
    if not resp.ok:
        print(resp.hint())
        return EXIT_API
    _print_cards(client, resp)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
