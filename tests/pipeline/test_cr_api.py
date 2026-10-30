"""U02: tools/cr_api.py, the shared official-API client, and tools/ladder_pull.py on top of it
(docs/SEASON_UPDATES.md 7.1, 7.3, 12).

No test touches the network. The client's transport is a FakeTransport that answers from the
hand-written fixtures in tests/pipeline/fixtures/cr_api/ (see the README there: every tag and
name is invented). The urllib and curl transports are checked with their HTTP layer replaced:
the opener's open() and subprocess.run. Waits go through a FakeClock, so a 429 back-off is a
list of numbers, not a pause.

Worked by hand for the ladder_pull golden test (--top 3 --workers 1):
* The board: page 1 (limit=3) gives A (#02PYQ0, 3600) and B (#08LGRJ, 3550) and the cursor
  eyJwb3MiOjJ9; page 2 (limit=1&after=...) gives C (#09CUV2, 3500).
* A's log: a ranked 1-0 over B at 20261029T201500.000Z, a friendly (skipped), and a ranked 0-2
  against D (#0P0LYJ, not on the board, so no elo) at 20261029T194000.000Z. B's log: the same
  battle with A from B's side (deduplicated on time and sorted tags; A's row is kept) and a 2v2
  (skipped). C's log: 404, so no battles.
* Decks are the card names sorted. A: Cannon, Fireball, Hog Rider, Ice Golem, Ice Spirit,
  Musketeer, Skeletons, The Log. B: Bats, Goblin Gang, Inferno Dragon, Mega Knight, Miner,
  Spear Goblins, Wall Breakers, Zap. D: Baby Dragon, Barbarian Barrel, Elixir Collector, Golem,
  Lightning, Lumberjack, Night Witch, Tornado.
* Meta: A seen 2 (won 1: 0.5), B seen 1 (won 0), D seen 1 (won 1); 4 sides in all, so shares
  0.5, 0.25 and 0.25. Ties keep the order of first appearance (Counter.most_common).
"""
from __future__ import annotations

import ast
import datetime
import gzip
import http.client
import importlib
import io
import json
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cr_api"
TOKEN = "fixture.token-value_1"
JSON = {"content-type": "application/json; charset=utf-8"}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.delenv("HASTYCR_CR_API_BASE", raising=False)
    return importlib.import_module("cr_api")


@pytest.fixture
def ladder(api):
    return importlib.import_module("ladder_pull")


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def raw(api, status: int, name: str | None = None, headers=None, body: bytes | None = None,
        error: str = ""):
    data = fixture(name) if name else (body or b"")
    return api.Raw(status, {**JSON, **(headers or {})} if data else dict(headers or {}), data,
                   error)


class FakeTransport:
    """Answers by exact URL. A route's answers are used in order; the last one repeats. An
    unknown URL fails the test (AssertionError is not a transport error the client absorbs)."""

    def __init__(self, routes):
        self.routes = {url: list(answers) for url, answers in routes.items()}
        self.calls: list[tuple[str, dict, float]] = []
        self._lock = threading.Lock()

    def __call__(self, url, headers, timeout):
        with self._lock:
            self.calls.append((url, dict(headers), timeout))
            answers = self.routes.get(url)
            assert answers, f"unexpected request {url}"
            return answers.pop(0) if len(answers) > 1 else answers[0]

    @property
    def urls(self) -> list[str]:
        return [url for url, _, _ in self.calls]


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 6))
        self.now += seconds


def client(api, transport, clock=None, **kwargs):
    clock = clock or FakeClock()
    rate = api.Rate(1000.0, clock=clock.monotonic, sleep=clock.sleep)
    return api.Client(TOKEN, transport=transport, rate=rate, env={}, **kwargs)


# ---------------------------------------------------------------------------
# the slug rule (section 7.3) and tags
# ---------------------------------------------------------------------------

def test_slug_reproduces_every_royaleapi_key(api):
    cards = json.loads((ROOT / "data" / "royaleapi" / "cards.json").read_text(
        encoding="utf-8"))["cards"]
    assert len(cards) == 120
    wrong = [(c["name"], c["key"], api.slug(c["name"])) for c in cards
             if api.slug(c["name"]) != c["key"]]
    assert wrong == []


def test_slug_leaves_the_card_map_keys_as_they_are(api):
    """The rule shared with data/cards: every key there, and its form keys, is a fixed point."""
    vendored = json.loads((ROOT / "data" / "cards" / "replay_cards.json").read_text(
        encoding="utf-8"))
    extras = json.loads((ROOT / "data" / "cards" / "replay_cards_extra.json").read_text(
        encoding="utf-8"))["cards"]
    keys = [*vendored, *extras]
    assert len(keys) >= 123             # 119 + 4 extras (minion-giant the fourth); more may come
    for key in keys:
        for form in ("", "-ev1", "-hero"):
            assert api.slug(key + form) == key + form


@pytest.mark.parametrize("name, key", [
    ("P.E.K.K.A", "pekka"), ("Mini P.E.K.K.A", "mini-pekka"),
    ("Super Mini P.E.K.K.A", "super-mini-pekka"), ("X-Bow", "x-bow"), ("The Log", "the-log"),
    ("  Hog   Rider ", "hog-rider"), ("Minion Giant", "minion-giant"), ("Goblinstein", "goblinstein"),
])
def test_slug_examples(api, name, key):
    assert api.slug(name) == key


def test_tags(api):
    assert api.norm_tag("02pyq0") == "#02PYQ0"
    assert api.norm_tag("%2302PYQ0") == "#02PYQ0"
    assert api.norm_tag(" #08lgro ") == "#08LGR0"            # a typed O is a zero
    assert api.tag_path("#02PYQ0") == "%2302PYQ0"
    for bad in ("", "#", "#AB-C", "#2PY Q0"):
        with pytest.raises(ValueError):
            api.norm_tag(bad)
    assert api.redact("/players/%2302PYQ0/battlelog") == "/players/%23<tag>/battlelog"
    assert api.redact("clan #02LQ8CV") == "clan #<tag>"


# ---------------------------------------------------------------------------
# settings: base, token
# ---------------------------------------------------------------------------

def test_resolve_base(api):
    assert api.resolve_base(None, {}) == api.OFFICIAL_BASE == "https://api.clashroyale.com/v1"
    assert api.resolve_base("official", {}) == api.OFFICIAL_BASE
    assert api.resolve_base("proxy", {}) == api.PROXY_BASE == "https://proxy.royaleapi.dev/v1"
    assert api.resolve_base(None, {"HASTYCR_CR_API_BASE": "proxy"}) == api.PROXY_BASE
    assert api.resolve_base("official", {"HASTYCR_CR_API_BASE": "proxy"}) == api.OFFICIAL_BASE
    assert api.resolve_base("https://example.invalid/v1/", {}) == "https://example.invalid/v1"
    for bad in ("http://proxy.royaleapi.dev/v1", "https://example.invalid/api", "ftp://x/v1"):
        with pytest.raises(ValueError):
            api.resolve_base(bad, {})
    assert api.base_kind(api.PROXY_BASE) == "proxy"
    assert api.base_kind("https://example.invalid/v1") == "custom"


def test_read_token(api, tmp_path):
    path = tmp_path / "supercell_token"
    path.write_bytes(b"\xef\xbb\xbf" + b"SECRETTOKEN.abc-def_1\r\n")       # BOM and CRLF
    assert api.read_token(path) == "SECRETTOKEN.abc-def_1"
    with pytest.raises(api.TokenError, match="no token at"):
        api.read_token(tmp_path / "missing")
    for text in ("", "  \n", "SECRETTOKEN one\n", "SECRETTOKEN\nsecond line\n",
                 '"SECRETTOKEN"'):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(api.TokenError) as caught:
            api.read_token(path)
        assert "SECRETTOKEN" not in str(caught.value)
    # Windows PowerShell 5.1's `>` and Out-File write UTF-16LE with a BOM: read it all the same
    path.write_bytes(b"\xff\xfe" + "SECRETTOKEN.abc-def_1\r\n".encode("utf-16-le"))
    assert api.read_token(path) == "SECRETTOKEN.abc-def_1"
    path.write_bytes(b"\xfe\xff" + "SECRETTOKEN.abc-def_1\n".encode("utf-16-be"))
    assert api.read_token(path) == "SECRETTOKEN.abc-def_1"
    # other bytes that are not UTF-8: a TokenError that says how to save it, and no byte of it
    path.write_bytes(b"SECRET\x80\x81TOKEN")
    with pytest.raises(api.TokenError, match="not plain text") as caught:
        api.read_token(path)
    assert "SECRET" not in str(caught.value) and "Set-Content -Encoding ascii" in str(caught.value)


def test_client_never_shows_the_token(api):
    fake = FakeTransport({})
    c = client(api, fake)
    assert TOKEN not in repr(c)
    assert c.headers()["Authorization"] == f"Bearer {TOKEN}"
    with pytest.raises(api.TokenError):
        api.Client("two words", transport=fake, env={})


# ---------------------------------------------------------------------------
# requests and answers
# ---------------------------------------------------------------------------

def test_headers_url_and_a_200(api):
    url = f"{api.OFFICIAL_BASE}/cards"
    fake = FakeTransport({url: [raw(api, 200, "cards.json")]})
    resp = client(api, fake).cards()
    assert resp.ok and resp.status == 200 and resp.attempts == 1
    assert len(resp.data["items"]) == 8
    (sent, headers, timeout), = fake.calls
    assert sent == url and timeout == api.DEFAULT_TIMEOUT
    assert headers == {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json",
                       "User-Agent": api.USER_AGENT}
    assert "python" not in api.USER_AGENT.lower() and "curl" not in api.USER_AGENT.lower()


def test_endpoint_paths_and_redacted_path(api):
    base = api.OFFICIAL_BASE
    ok = raw(api, 200, body=b"{}")
    fake = FakeTransport({f"{base}/players/%2302PYQ0/battlelog": [ok],
                          f"{base}/players/%2302PYQ0": [ok],
                          f"{base}/clans/%2302LQ8CV": [ok],
                          f"{base}/clans/%2302LQ8CV/members": [ok]})
    c = client(api, fake)
    resp = c.battlelog("#02PYQ0")
    assert resp.ok and resp.path == "/players/%23<tag>/battlelog"
    c.player("02pyq0")
    c.clan("#02LQ8CV")
    c.clan_members("#02lq8cv")
    assert len(fake.calls) == 4


def test_proxy_base(api):
    fake = FakeTransport({f"{api.PROXY_BASE}/cards": [raw(api, 200, "cards.json")]})
    assert client(api, fake, base="proxy").cards().ok


def test_403_returns_the_code_and_says_why(api):
    url = f"{api.OFFICIAL_BASE}/cards"
    fake = FakeTransport({url: [raw(api, 403, "error_403_invalid_ip.json")]})
    clock = FakeClock()
    resp = client(api, fake, clock).cards()               # no exception, no exit
    assert resp.status == 403 and not resp.ok and resp.attempts == 1
    assert resp.reason == "accessDenied.invalidIp"
    assert "198.51.100.7" in resp.message
    assert len(fake.calls) == 1 and clock.slept == []      # not retried
    hint = resp.hint()
    assert "developer.clashroyale.com" in hint and api.PROXY_IP in hint

    fake = FakeTransport({f"{api.PROXY_BASE}/cards": [raw(api, 403, "error_403_invalid_ip.json")]})
    assert "RoyaleAPI's proxy" in client(api, fake, base="proxy").cards().hint()

    fake = FakeTransport({url: [raw(api, 403, "error_403.json")]})
    resp = client(api, fake).cards()
    assert resp.reason == "accessDenied" and "supercell_token" in resp.hint()


def test_429_waits_retry_after_then_succeeds(api):
    url = f"{api.OFFICIAL_BASE}/cards"
    fake = FakeTransport({url: [raw(api, 429, "error_429.json", {"retry-after": "3"}),
                                raw(api, 200, "cards.json")]})
    clock, notes = FakeClock(), []
    c = client(api, fake, clock, log=notes.append)
    resp = c.cards()
    assert resp.ok and resp.attempts == 2 and len(fake.calls) == 2
    assert clock.slept == [3.0]
    assert dict(c.statuses) == {429: 1, 200: 1} and c.retry_after_seen == [3.0]
    assert notes and "waiting 3 s" in notes[0]


def test_429_backs_off_without_retry_after(api):
    url = f"{api.OFFICIAL_BASE}/cards"
    fake = FakeTransport({url: [raw(api, 429, "error_429.json"), raw(api, 429, "error_429.json"),
                                raw(api, 200, "cards.json")]})
    clock = FakeClock()
    resp = client(api, fake, clock).cards()
    assert resp.ok and resp.attempts == 3
    assert clock.slept == [2.0, 4.0]                       # BACKOFF, doubled


def test_429_gives_up_and_returns_the_code(api):
    url = f"{api.OFFICIAL_BASE}/cards"
    fake = FakeTransport({url: [raw(api, 429, "error_429.json")]})
    clock = FakeClock()
    resp = client(api, fake, clock, retries=2).cards()
    assert resp.status == 429 and not resp.ok
    assert resp.attempts == 3 and len(fake.calls) == 3
    assert clock.slept == [2.0, 4.0]
    assert resp.reason == "requestThrottled" and "throttled" in resp.hint()
    assert 429 in api.STOP                                 # callers stop a bulk run on it
    # given up, the shared Rate still holds for the next back-off step (2 * 2**2 = 8 s): any
    # later request, on any thread, waits it out
    c_rate = api.Rate(1000.0, clock=clock.monotonic, sleep=clock.sleep)
    c = api.Client(TOKEN, transport=fake, rate=c_rate, env={}, retries=2)
    clock.slept.clear()
    c.cards()
    assert clock.slept == [2.0, 4.0]
    c_rate.wait()
    assert clock.slept == [2.0, 4.0, 8.0]


def test_429_with_a_long_retry_after_is_not_waited_out(api):
    url = f"{api.OFFICIAL_BASE}/cards"
    fake = FakeTransport({url: [raw(api, 429, "error_429.json", {"retry-after": "600"})]})
    clock = FakeClock()
    rate = api.Rate(1000.0, clock=clock.monotonic, sleep=clock.sleep)
    resp = api.Client(TOKEN, transport=fake, rate=rate, env={}).cards()
    assert resp.status == 429 and resp.attempts == 1 and resp.retry_after == 600.0
    assert clock.slept == [] and "600" in resp.hint()
    rate.wait()                     # not retried, but no thread asks again within the 600 s
    assert clock.slept == [600.0]


def test_a_429_pauses_every_thread(api):
    """The back-off is put on the shared Rate: the next request on any thread waits too."""
    clock = FakeClock()
    rate = api.Rate(1000.0, clock=clock.monotonic, sleep=clock.sleep)
    rate.wait()
    rate.hold(5.0)
    rate.wait()
    assert clock.slept == [5.0]


def test_rate_spacing(api):
    clock = FakeClock()
    rate = api.Rate(2.0, clock=clock.monotonic, sleep=clock.sleep)
    for _ in range(3):
        rate.wait()
    assert clock.slept == [0.5, 0.5] and clock.now == 1.0


def test_retry_after_seconds(api):
    now = datetime.datetime(2026, 10, 30, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    assert api.retry_after_seconds("3", now) == 3.0
    assert api.retry_after_seconds(" 2.5 ", now) == 2.5
    assert api.retry_after_seconds("-4", now) == 0.0
    assert api.retry_after_seconds("Fri, 30 Oct 2026 12:00:10 GMT", now) == 10.0
    for bad in (None, "", "soon", "nan", "inf"):
        assert api.retry_after_seconds(bad, now) is None


def test_transient_errors_are_tried_once_more(api):
    url = f"{api.OFFICIAL_BASE}/cards"
    fake = FakeTransport({url: [raw(api, 500, body=b"oops"), raw(api, 200, "cards.json")]})
    clock = FakeClock()
    resp = client(api, fake, clock).cards()
    assert resp.ok and resp.attempts == 2 and clock.slept == [api.TRANSIENT_WAIT]

    fake = FakeTransport({url: [raw(api, 503, "error_503.json")]})
    resp = client(api, fake).cards()
    assert resp.status == 503 and len(fake.calls) == 1 and "maintenance" in resp.hint()

    fake = FakeTransport({url: [api.Raw(0, error="no response: refused")]})
    resp = client(api, fake).cards()
    assert resp.status == 0 and len(fake.calls) == 2 and "refused" in resp.error
    assert resp.hint().startswith("No response")


def test_a_transport_error_is_status_0(api):
    calls = []

    def broken(url, headers, timeout):
        calls.append(url)
        raise OSError("boom")

    resp = client(api, broken).cards()
    assert resp.status == 0 and "boom" in resp.error and len(calls) == 2


def test_a_200_that_is_not_json(api):
    url = f"{api.OFFICIAL_BASE}/cards"
    fake = FakeTransport({url: [api.Raw(200, {}, b"<html>")]})
    resp = client(api, fake).cards()
    assert resp.status == 200 and not resp.ok and resp.error == "the body is not JSON"


def test_paged(api):
    path = f"{api.OFFICIAL_BASE}/locations/global/pathoflegend/players"
    fake = FakeTransport({f"{path}?limit=3": [raw(api, 200, "pol_page1.json")],
                          f"{path}?limit=1&after=eyJwb3MiOjJ9": [raw(api, 200, "pol_page2.json")],
                          f"{path}?limit=2": [raw(api, 200, "pol_page1.json")]})
    c = client(api, fake)
    items, last = c.paged("locations/global/pathoflegend/players", want=3)
    assert [p["tag"] for p in items] == ["#02PYQ0", "#08LGRJ", "#09CUV2"] and last.ok
    items, _ = c.paged("locations/global/pathoflegend/players", want=2)
    assert len(items) == 2
    assert fake.urls == [f"{path}?limit=3", f"{path}?limit=1&after=eyJwb3MiOjJ9",
                         f"{path}?limit=2"]
    assert c.paged("locations/global/pathoflegend/players", want=0) == ([], None)

    fake = FakeTransport({f"{path}?limit=3": [raw(api, 403, "error_403_invalid_ip.json")]})
    items, last = client(api, fake).paged("locations/global/pathoflegend/players", want=3)
    assert items == [] and last.status == 403


# ---------------------------------------------------------------------------
# the transports, with their HTTP layer replaced
# ---------------------------------------------------------------------------

class FakeHTTPResponse:
    def __init__(self, status, headers, body):
        self.status = status
        self.headers = http.client.HTTPMessage()
        for name, value in headers.items():
            self.headers[name] = value
        self._body = body

    def read(self):
        return self._body

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_urllib_transport_sends_our_user_agent_and_decodes_gzip(api, monkeypatch):
    transport = api.UrllibTransport()
    https = [h for h in transport._opener.handlers if isinstance(h, urllib.request.HTTPSHandler)]
    assert len(https) == 1 and isinstance(https[0], api._IPv4HTTPSHandler)
    seen = {}

    def fake_open(req, timeout=None):
        seen.update(req=req, timeout=timeout)
        return FakeHTTPResponse(200, {"Content-Type": "application/json",
                                      "Content-Encoding": "gzip"},
                                gzip.compress(b'{"items": []}'))

    monkeypatch.setattr(transport._opener, "open", fake_open)
    got = transport(f"{api.OFFICIAL_BASE}/cards",
                    {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json",
                     "User-Agent": api.USER_AGENT}, 30.0)
    assert got.status == 200 and got.body == b'{"items": []}' and got.error == ""
    assert got.headers["content-encoding"] == "gzip"
    req = seen["req"]
    assert req.get_header("User-agent") == api.USER_AGENT
    assert req.get_header("Authorization") == f"Bearer {TOKEN}"
    assert req.get_header("Accept-encoding") == "gzip" and seen["timeout"] == 30.0
    # the token is an unredirected header: urllib never copies it onto a redirect
    assert "Authorization" not in req.headers and "Authorization" in req.unredirected_hdrs


def test_urllib_transport_never_follows_a_redirect(api, monkeypatch):
    """urllib's own redirect handler copies req.headers onto the request for the Location; the
    transport's handler refuses, so a 3xx comes back as its status and nothing is re-sent."""
    transport = api.UrllibTransport()
    redirectors = [h for h in transport._opener.handlers
                   if isinstance(h, urllib.request.HTTPRedirectHandler)]
    assert len(redirectors) == 1 and isinstance(redirectors[0], api._NoRedirect)
    req = urllib.request.Request(f"{api.OFFICIAL_BASE}/cards")
    assert redirectors[0].redirect_request(req, None, 302, "Found", {},
                                           "http://elsewhere.invalid/v1/cards") is None
    url = f"{api.OFFICIAL_BASE}/cards"
    opened = []

    def moved(req, timeout=None):
        opened.append(req.full_url)
        hdrs = http.client.HTTPMessage()
        hdrs["Location"] = "http://elsewhere.invalid/v1/cards"
        raise urllib.error.HTTPError(url, 302, "Found", hdrs, io.BytesIO(b""))

    monkeypatch.setattr(transport._opener, "open", moved)
    got = transport(url, {"Authorization": f"Bearer {TOKEN}"}, 30.0)
    assert got.status == 302 and opened == [url]
    assert got.headers["location"] == "http://elsewhere.invalid/v1/cards"


def test_urllib_transport_maps_errors(api, monkeypatch):
    transport = api.UrllibTransport()
    url = f"{api.OFFICIAL_BASE}/cards"

    def throttled(req, timeout=None):
        hdrs = http.client.HTTPMessage()
        hdrs["Retry-After"] = "7"
        hdrs["Content-Type"] = "application/json"
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", hdrs,
                                     io.BytesIO(fixture("error_429.json")))

    monkeypatch.setattr(transport._opener, "open", throttled)
    got = transport(url, {}, 30.0)
    assert got.status == 429 and got.headers["retry-after"] == "7"
    assert json.loads(got.body)["reason"] == "requestThrottled"

    def refused(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(transport._opener, "open", refused)
    got = transport(url, {}, 30.0)
    assert got.status == 0 and "connection refused" in got.error

    def timed_out(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(transport._opener, "open", timed_out)
    assert transport(url, {}, 30.0).status == 0


def test_urllib_connects_over_ipv4(api, monkeypatch):
    asked, opened = [], []

    def fake_getaddrinfo(host, port, family=0, type=0, *rest):
        asked.append((host, port, family, type))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.10", port))]

    def fake_create_connection(address, *args, **kwargs):
        opened.append(address)
        return "socket"

    monkeypatch.setattr(api.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(api.socket, "create_connection", fake_create_connection)
    assert api._connect_ipv4(("api.clashroyale.com", 443), 30.0) == "socket"
    assert asked == [("api.clashroyale.com", 443, socket.AF_INET, socket.SOCK_STREAM)]
    assert opened == [("192.0.2.10", 443)]
    conn = api._IPv4HTTPSConnection("api.clashroyale.com")
    assert conn._create_connection is api._connect_ipv4


def test_curl_transport_keeps_the_token_off_argv(api, monkeypatch, tmp_path):
    seen = {}

    def fake_run(argv, input=None, **kwargs):
        seen.update(argv=argv, input=input, kwargs=kwargs)
        Path(argv[argv.index("-o") + 1]).write_bytes(fixture("error_429.json"))
        # bytes, as curl writes the dump: write_text would turn each \n into \r\n on Windows
        Path(argv[argv.index("-D") + 1]).write_bytes(
            b"HTTP/1.1 200 Connection established\r\n\r\n"
            b"HTTP/2 429\r\ncontent-type: application/json\r\nretry-after: 7\r\n\r\n")
        return subprocess.CompletedProcess(argv, 0, stdout="429", stderr="")

    monkeypatch.setattr(api.subprocess, "run", fake_run)
    scratch = tmp_path / "scratch"
    transport = api.CurlTransport(scratch=scratch)
    url = f"{api.OFFICIAL_BASE}/players/%2302PYQ0/battlelog"
    got = transport(url, {"Authorization": f"Bearer {TOKEN}", "User-Agent": api.USER_AGENT},
                    30.0)
    assert got.status == 429 and got.headers == {"content-type": "application/json",
                                                 "retry-after": "7"}
    assert json.loads(got.body)["reason"] == "requestThrottled"
    argv = seen["argv"]
    assert argv[:2] == ["curl", "-q"]                      # -q first: no .curlrc applies
    assert "-4" in argv and "--compressed" in argv and "-L" not in argv
    assert argv[argv.index("-K") + 1] == "-" and argv[-1] == url
    assert not any(TOKEN in part for part in argv)
    assert f'header = "Authorization: Bearer {TOKEN}"\n' in seen["input"]
    assert f'header = "User-Agent: {api.USER_AGENT}"\n' in seen["input"]
    assert list(scratch.iterdir()) == []                   # files deleted after the call


def test_parse_head_reads_every_line_end(api):
    """\r\n as sent, \n, and \r\r\n (a \r\n written through a Windows text-mode file)."""
    want = {"content-type": "application/json", "retry-after": "7"}
    for end in ("\r\n", "\n", "\r\r\n"):
        text = (f"HTTP/1.1 200 Connection established{end}{end}HTTP/2 429{end}"
                f"content-type: application/json{end}retry-after: 7{end}{end}")
        assert api._parse_head(text) == want, repr(end)
    assert api._parse_head("") == {}


def test_curl_transport_without_a_response(api, monkeypatch, tmp_path):
    def unreachable(argv, input=None, **kwargs):
        return subprocess.CompletedProcess(argv, 6, stdout="000",
                                           stderr="curl: (6) Could not resolve host")

    monkeypatch.setattr(api.subprocess, "run", unreachable)
    transport = api.CurlTransport(scratch=tmp_path)
    got = transport(f"{api.OFFICIAL_BASE}/cards", {}, 30.0)
    assert got.status == 0 and "Could not resolve host" in got.error

    def missing(argv, input=None, **kwargs):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(api.subprocess, "run", missing)
    assert "was not found" in transport(f"{api.OFFICIAL_BASE}/cards", {}, 30.0).error


def test_make_transport(api):
    assert isinstance(api.make_transport(None), api.UrllibTransport)
    assert isinstance(api.make_transport("curl"), api.CurlTransport)

    def mine(url, headers, timeout):
        return api.Raw(200)

    assert api.make_transport(mine) is mine
    with pytest.raises(ValueError):
        api.make_transport("wget")


# ---------------------------------------------------------------------------
# the roster check and the CLI
# ---------------------------------------------------------------------------

def test_cards_report_on_the_fixture(api, monkeypatch):
    rs_cards = importlib.import_module("rs_cards")
    payload = json.loads(fixture("cards.json"))
    # the map without the "minion-giant" extra (data/cards/replay_cards_extra.json, added from
    # this very proposal), so the fixture's Minion Giant is still the unmapped card it was
    rep = api.cards_report(payload, set(rs_cards.load()) - {"minion-giant"})
    assert rep["items"] == 8 and rep["support_items"] == 1
    assert rep["max_evolution_level"] == {"1": 2, "2": 1, "3": 1}
    assert rep["without_max_evolution_level"] == 4
    assert rep["icon_keys"] == ["evolutionMedium", "heroMedium", "medium"]
    assert rep["collisions"] == {}
    assert rep["mapped"] == 7
    assert rep["unmapped"] == [{"name": "Minion Giant", "key": "minion-giant"}]
    bare = api.cards_report({"items": [{"name": "Knight"}, {"name": "knight"}]})
    assert bare["collisions"] == {"knight": ["Knight", "knight"]} and "unmapped" not in bare
    assert api.header_report({"x-ratelimit-remaining": "9", "content-encoding": "gzip"}) == {
        "content_encoding": "gzip", "rate_headers": ["x-ratelimit-remaining"],
        "names": ["content-encoding", "x-ratelimit-remaining"]}


def _token_file(tmp_path) -> Path:
    path = tmp_path / "supercell_token"
    path.write_text(TOKEN + "\n", encoding="utf-8")
    return path


def test_cli_cards(api, monkeypatch, tmp_path, capsys):
    fake = FakeTransport({f"{api.OFFICIAL_BASE}/cards": [
        raw(api, 200, "cards.json", {"cache-control": "max-age=600"})]})
    monkeypatch.setattr(api, "make_transport", lambda spec=None, **kw: fake)
    keys, _ = api._card_map_keys()
    monkeypatch.setattr(api, "_card_map_keys", lambda: (set(keys) - {"minion-giant"}, ""))
    assert api.main(["cards", "--token-file", str(_token_file(tmp_path))]) == api.EXIT_OK
    out = capsys.readouterr().out
    assert "/cards: 200" in out and "8 cards, 1 support items" in out
    assert "Minion Giant -> minion-giant" in out
    assert "rate-limit headers: none" in out and TOKEN not in out


def test_cli_cards_errors(api, monkeypatch, tmp_path, capsys):
    fake = FakeTransport({f"{api.OFFICIAL_BASE}/cards": [raw(api, 403,
                                                              "error_403_invalid_ip.json")]})
    monkeypatch.setattr(api, "make_transport", lambda spec=None, **kw: fake)
    assert api.main(["cards", "--token-file", str(_token_file(tmp_path))]) == api.EXIT_API
    assert "developer.clashroyale.com" in capsys.readouterr().out
    assert api.main(["cards", "--token-file", str(tmp_path / "none")]) == api.EXIT_SETTING
    assert "no token at" in capsys.readouterr().err
    assert api.main(["cards", "--token-file", str(_token_file(tmp_path)),
                     "--base", "http://example.invalid/v1"]) == api.EXIT_SETTING
    assert api.main(["cards", "--token-file", str(_token_file(tmp_path)),
                     "--rate", "50"]) == api.EXIT_SETTING
    assert "--rate 50: between 0.1 and 5" in capsys.readouterr().err


def test_cr_api_imports_only_the_standard_library(api):
    tree = ast.parse((TOOLS / "cr_api.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names - set(sys.stdlib_module_names) - {"__future__"} == {"rs_cards"}
    assert not any("gamestate" in name or "memread" in name for name in names)


# ---------------------------------------------------------------------------
# tools/ladder_pull.py on the shared client: the same outputs
# ---------------------------------------------------------------------------

DECK_A = "Cannon,Fireball,Hog Rider,Ice Golem,Ice Spirit,Musketeer,Skeletons,The Log"
DECK_B = "Bats,Goblin Gang,Inferno Dragon,Mega Knight,Miner,Spear Goblins,Wall Breakers,Zap"
DECK_D = ("Baby Dragon,Barbarian Barrel,Elixir Collector,Golem,Lightning,Lumberjack,"
          "Night Witch,Tornado")


def _ladder_routes(api, board_answer=None):
    base = api.OFFICIAL_BASE
    board = f"{base}/locations/global/pathoflegend/players"
    return {
        f"{board}?limit=3": [board_answer or raw(api, 200, "pol_page1.json")],
        f"{board}?limit=1&after=eyJwb3MiOjJ9": [raw(api, 200, "pol_page2.json")],
        f"{base}/players/%2302PYQ0/battlelog": [raw(api, 200, "battlelog_a.json")],
        f"{base}/players/%2308LGRJ/battlelog": [raw(api, 200, "battlelog_b.json")],
        f"{base}/players/%2309CUV2/battlelog": [raw(api, 404, "error_404.json")],
    }


def fast_rates(api, ladder, monkeypatch):
    """ladder_pull's Rate(per_second) replaced by a FakeClock one; returns the rates asked for."""
    asked: list[float] = []
    clock = FakeClock()

    def make(per_second):
        asked.append(per_second)
        return api.Rate(1000.0, clock=clock.monotonic, sleep=clock.sleep)

    monkeypatch.setattr(ladder, "Rate", make)
    return asked


def test_ladder_pull_writes_the_same_outputs(api, ladder, monkeypatch, tmp_path):
    fake = FakeTransport(_ladder_routes(api))
    made = []

    def fake_make(spec=None, **kwargs):
        made.append((spec, kwargs))
        return fake

    monkeypatch.setattr(api, "make_transport", fake_make)
    monkeypatch.setattr(ladder, "TOKEN_FILE", _token_file(tmp_path))
    asked = fast_rates(api, ladder, monkeypatch)
    out = tmp_path / "out"
    assert ladder.main(["--top", "3", "--workers", "1", "--out", str(out)]) == 0
    assert made == [("curl", {"scratch": out / ".scratch"})]       # curl stays the default
    assert asked == [api.DEFAULT_RATE]                             # 2 per second by default
    base = api.OFFICIAL_BASE
    assert fake.urls == [
        f"{base}/locations/global/pathoflegend/players?limit=3",
        f"{base}/locations/global/pathoflegend/players?limit=1&after=eyJwb3MiOjJ9",
        f"{base}/players/%2302PYQ0/battlelog", f"{base}/players/%2308LGRJ/battlelog",
        f"{base}/players/%2309CUV2/battlelog"]
    assert all(h["User-Agent"] == api.USER_AGENT for _, h, _ in fake.calls)

    expected_csv = (
        "battle_time,player_tag,opponent_tag,player_deck,opponent_deck,player_crowns,"
        "opponent_crowns,player_elo,opponent_elo\r\n"
        f'20261029T201500.000Z,#02PYQ0,#08LGRJ,"{DECK_A}","{DECK_B}",1,0,3600,3550\r\n'
        f'20261029T194000.000Z,#02PYQ0,#0P0LYJ,"{DECK_A}","{DECK_D}",0,2,3600,\r\n')
    assert (out / "ladder_battles.csv").read_bytes() == expected_csv.encode("utf-8")
    expected_meta = [
        {"deck": DECK_A, "seen": 2, "wins": 1, "win_rate": 0.5, "share": 0.5},
        {"deck": DECK_B, "seen": 1, "wins": 0, "win_rate": 0.0, "share": 0.25},
        {"deck": DECK_D, "seen": 1, "wins": 1, "win_rate": 1.0, "share": 0.25},
    ]
    text = (out / "ladder_meta.json").read_text(encoding="utf-8")
    assert text == json.dumps(expected_meta, indent=1)


def test_ladder_pull_still_stops_on_403(api, ladder, monkeypatch, tmp_path):
    fake = FakeTransport(_ladder_routes(api, raw(api, 403, "error_403_invalid_ip.json")))
    monkeypatch.setattr(api, "make_transport", lambda spec=None, **kw: fake)
    monkeypatch.setattr(ladder, "TOKEN_FILE", _token_file(tmp_path))
    fast_rates(api, ladder, monkeypatch)
    with pytest.raises(SystemExit) as caught:
        ladder.main(["--top", "3", "--out", str(tmp_path / "out")])
    text = str(caught.value.code)
    assert text.startswith(ladder.FORBIDDEN) and "accessDenied.invalidIp" in text
    assert len(fake.calls) == 1


@pytest.mark.parametrize("status, name, head", [(429, "error_429.json", "THROTTLED"),
                                                (503, "error_503.json", "MAINTENANCE")])
def test_ladder_pull_stops_every_worker_on_429_and_503(api, ladder, monkeypatch, tmp_path,
                                                       status, name, head):
    """A's log is refused (a 429 with a Retry-After too long to wait, or maintenance): the walk
    stops there; B's and C's logs are never asked, and nothing is written."""
    base = api.OFFICIAL_BASE
    routes = _ladder_routes(api)
    routes[f"{base}/players/%2302PYQ0/battlelog"] = [raw(api, status, name,
                                                        {"retry-after": "600"})]
    fake = FakeTransport(routes)
    monkeypatch.setattr(api, "make_transport", lambda spec=None, **kw: fake)
    monkeypatch.setattr(ladder, "TOKEN_FILE", _token_file(tmp_path))
    fast_rates(api, ladder, monkeypatch)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as caught:
        ladder.main(["--top", "3", "--workers", "1", "--out", str(out)])
    assert str(caught.value.code).startswith(getattr(ladder, head))
    assert fake.urls[-1] == f"{base}/players/%2302PYQ0/battlelog" and len(fake.calls) == 3
    assert not (out / "ladder_battles.csv").exists()


def test_ladder_pull_refuses_an_impolite_rate(api, ladder, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ladder, "TOKEN_FILE", _token_file(tmp_path))
    for argv in (["--rate", "50"], ["--rate", "0"], ["--workers", "64"]):
        with pytest.raises(SystemExit) as caught:
            ladder.main([*argv, "--out", str(tmp_path / "out")])
        assert caught.value.code == 2, argv
    assert "--rate 50: between 0.1 and 5" in capsys.readouterr().err


def test_ladder_pull_without_a_token(api, ladder, monkeypatch, tmp_path):
    monkeypatch.setattr(ladder, "TOKEN_FILE", tmp_path / "missing")
    with pytest.raises(SystemExit) as caught:
        ladder.main(["--out", str(tmp_path / "out")])
    assert str(caught.value.code) == f"no token at {tmp_path / 'missing'}"
    assert ladder.Rate is api.Rate and ladder.BASE == api.OFFICIAL_BASE
