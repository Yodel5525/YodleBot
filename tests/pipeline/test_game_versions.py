"""U04: data/game_versions.json, the game-version ledger, and tools/game_versions.py
(docs/SEASON_UPDATES.md 2, 3.1, 3.3, 4.1, 10 and 12).

Worked by hand:
* Seasons. 2026-01-01 is a Thursday, so the first Mondays of 2026 are 01-05, 02-02, 03-02,
  04-06, 05-04, 06-01, 07-06, 08-03, 09-07, 10-05, 11-02 and 12-07. Season 89 starts in
  2026-11, so 2026-01 is Season 79 and 2026-12-07 Season 90. 2026-11-01 is a Sunday, still
  Season 88 (since 10-05).
* Client builds. versionCode is MM mmmm bbb: 160635001 = 16 * 10^7 + 635 * 10^3 + 1 is 16.635.1,
  line (16, 635). The pinned table's content 16.402.22 is line (16, 402): the client is ahead.
  160402020 is 16.402.20, the same line; 150535029 is 15.535.29, behind.
* Days behind. The pinned table's content date is 2026-10-19. On 2026-10-30 no content event
  follows it. On 2026-11-05 Season 89 (2026-11-02, no table) is 3 days old; on 2026-12-08
  Season 90 (12-07, by the rule) is missed too, and the first missed event is 36 days old
  (11-02 -> 12-02 is 30 days, + 6).
* The currency line (section 4.1), wrapped at 100 columns after "currency  " (10 wide): the
  client part is 82 characters with its ";" (92 in all); "royalesim 0.1.42 = pin, newest on
  PyPI;" (39) does not fit after it, so it starts line 2 at 10 + 39 = 49, then "checked 6 h ago;"
  (+17 = 66) and "pool pulled 6 h ago;" (+21 = 87); the 89-character Season part does not fit
  and is line 3 (99 wide). That is the example in section 4.1, character for character.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import importlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
LEDGER = ROOT / "data" / "game_versions.json"
UTC = dt.timezone.utc

DOC_EXAMPLE = (
    "currency  client 160635001 (16.635.1) vs table 16.402.22: client ahead, accepted 2026-10-29;\n"
    "          royalesim 0.1.42 = pin, newest on PyPI; checked 6 h ago; pool pulled 6 h ago;\n"
    "          Season 89 started 2026-11-02: run scripts\\game_update.ps1 -Update when RoyaleSim "
    "ships it")


@pytest.fixture
def gv(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    return importlib.import_module("game_versions")


@pytest.fixture
def ledger(gv):
    return gv.load()


def _write(tmp_path, data, name="ledger.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# the ledger against the repo
# ---------------------------------------------------------------------------

def test_ledger_pins_what_the_repo_pins(gv, ledger, monkeypatch):
    p = gv.pin(ledger)
    assert (p["royalesim"], p["card_table"]) == ("0.1.42", "client160402017_20261019")
    for req in ("requirements/engine.txt", "requirements/gpu.txt"):
        lines = (ROOT / req).read_text(encoding="utf-8").splitlines()
        assert f"royalesim=={p['royalesim']}" in lines, req
    assert f'"royalesim=={p["royalesim"]}"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    engine_json = json.loads((ROOT / "data" / "engine.json").read_text(encoding="utf-8"))
    assert engine_json["card_table"] == p["card_table"]
    rs_engine = importlib.import_module("rs_engine")
    assert rs_engine.TARGET_TABLE == p["card_table"]
    table = p["table"]
    assert rs_engine.KNOWN_TABLES[p["card_table"]] == table["version"]
    assert rs_engine.TABLE_ALIASES[table["content"]] == p["card_table"]
    assert set(p["release"]["arms"]) == set(rs_engine.KNOWN_TABLES)
    assert set(p["release"]["tables"]) == set(p["release"]["arms"])
    assert p["release"]["default_arm"] == "client15535"


def test_ledger_records_the_section_3_3_fnvs_and_the_table_versions(ledger):
    tables = ledger["royalesim"]["0.1.42"]["tables"]
    assert tables["client160402017_20261019"]["fnv1a64"] == "15bb4e60e69d05f3"
    assert tables["client160402017_20261006"]["fnv1a64"] == "87cca9def6207621"
    assert tables["client15535"]["fnv1a64"] == "803ec02c7eac9fd3"
    target = tables["client160402017_20261019"]
    assert (target["client_build"], target["content"], target["content_date"]) == \
        (160402020, "16.402.22", "2026-10-19")
    assert target["roster"] == {"cards": 145, "evolutions": 43, "hero_forms": 18, "towers": 2,
                                "units": 320}
    assert target["lacks"] == ["Giant evolution", "Mirror evolution"]
    # docs/SEASON_UPDATES.md 2: the sdist file hashes to 813bc904...
    assert target["sha256"].startswith("813bc904")


def test_ledger_card_map_entry_is_the_vendored_file(ledger, monkeypatch):
    rs_cards = importlib.import_module("rs_cards")
    entry = ledger["royaleimitate"]
    assert entry["sha256"] == rs_cards.VENDORED_SHA256
    assert entry["keys"] == rs_cards.VENDORED_ENTRIES
    assert entry["vendored"] in rs_cards.VENDORED_SOURCE


def test_selfcheck_recorded_build_is_accepted_in_the_ledger(gv, ledger, tmp_path):
    text = (ROOT / "scripts" / "selfcheck.py").read_text(encoding="utf-8")
    m = re.search(r"^RECORDED_GAME_BUILD\s*=\s*(\d+)", text, re.M)
    if not m:
        pytest.skip("selfcheck reads the ledger now (U05)")
    got = gv.judge_client(ledger, int(m.group(1)), tmp_path)
    assert got["verdict"] == "ok" and got["basis"] == "owner"


def test_ledger_clients_and_events_are_consistent(gv, ledger):
    codes = [c["version_code"] for c in ledger["clients"]]
    assert codes == sorted(codes)
    for c in ledger["clients"]:
        assert c["version_name"] == gv.version_name(c["version_code"])
    accepted = [c["version_code"] for c in ledger["clients"] if c["accepted"]]
    assert accepted == [160402020, 160635001]
    season = [e for e in ledger["content_events"] if e["kind"] == "season"]
    assert [(e["date"], e["season"]) for e in season] == [("2026-11-02", 89)]
    assert {f["card"] for f in season[0]["forms"]} == {"Giant", "Mirror"}
    assert season[0]["table"] is None and season[0]["confirmed"] is False
    arms = ledger["royalesim"]["0.1.42"]["tables"]
    for e in ledger["content_events"]:
        assert e["table"] is None or e["table"] in arms


# ---------------------------------------------------------------------------
# seasons
# ---------------------------------------------------------------------------

def test_season_rule_reproduces_the_2026_starts(gv, ledger):
    observed = [dt.date.fromisoformat(d) for d in ledger["seasons"]["observed"]]
    starts = gv.season_starts(2026)
    assert len(observed) == 11 and starts[:11] == observed
    assert starts[11] == dt.date(2026, 12, 7)
    assert [gv.season_number(d.year, d.month) for d in starts] == list(range(79, 91))
    assert all(d.weekday() == 0 and d.day <= 7 for d in starts)
    assert gv.season_start(89) == dt.date(2026, 11, 2)
    assert gv.season_start(79) == dt.date(2026, 1, 5)
    assert gv.season_month(91) == (2027, 1)


def test_season_on_and_next(gv):
    assert gv.season_on(dt.date(2026, 10, 30)) == (88, dt.date(2026, 10, 5))
    assert gv.season_on(dt.date(2026, 11, 1)) == (88, dt.date(2026, 10, 5))
    assert gv.season_on(dt.date(2026, 11, 2)) == (89, dt.date(2026, 11, 2))
    assert gv.season_on(dt.date(2027, 1, 2)) == (90, dt.date(2026, 12, 7))
    assert gv.next_season(dt.date(2026, 10, 30)) == (89, dt.date(2026, 11, 2))
    assert gv.next_season(dt.date(2026, 11, 2)) == (90, dt.date(2026, 12, 7))
    assert gv.seasons_between(dt.date(2026, 10, 30), dt.date(2026, 12, 31)) == [
        (89, dt.date(2026, 11, 2)), (90, dt.date(2026, 12, 7))]
    assert gv.seasons_between(dt.date(2026, 11, 2), dt.date(2026, 12, 6)) == []


# ---------------------------------------------------------------------------
# client builds
# ---------------------------------------------------------------------------

def test_client_codes_and_lines(gv):
    assert gv.version_name(160635001) == "16.635.1"
    assert gv.parse_code("160402020") == (16, 402, 20)
    assert gv.client_line(160635001) == (16, 635)
    assert gv.parse_content("16.402.22") == (16, 402, 22)
    assert gv.parse_content("16.402") == (16, 402, 0)
    assert gv.parse_content("cards-1") is None
    assert gv.relation(160635001, "16.402.22") == "ahead"
    assert gv.relation(160402020, "16.402.22") == "same"
    assert gv.relation(150535029, "16.402.22") == "behind"
    assert gv.relation(160635001, None) is None
    for bad in (12, "abc", True, None, 10**11):
        with pytest.raises(ValueError):
            gv.parse_code(bad)


def test_judge_client_and_local_acceptance(gv, ledger, tmp_path):
    ok = gv.judge_client(ledger, 160635001, tmp_path)
    assert ok["verdict"] == "ok" and ok["accepted"] == "2026-10-29"
    assert (ok["basis"], ok["relation"], ok["table_content"]) == ("owner", "ahead", "16.402.22")
    assert gv.judge_client(ledger, 160402012, tmp_path)["verdict"] == "unaccepted"
    new = gv.judge_client(ledger, "160700001", tmp_path)
    assert (new["verdict"], new["known"], new["version_name"]) == ("new", False, "16.700.1")
    gv.record_local_client({"version_code": 160700001, "version_name": "16.700.1",
                            "accepted": "2026-11-04", "basis": "auto"}, tmp_path)
    got = gv.judge_client(ledger, 160700001, tmp_path)
    assert (got["verdict"], got["basis"], got["where"]) == ("ok", "auto", "local")
    assert gv.newest_accepted(ledger, tmp_path)["version_code"] == 160700001
    # a local entry never overrides the ledger's own
    gv.record_local_client({"version_code": 160402012, "accepted": "2026-11-04",
                            "basis": "auto"}, tmp_path)
    assert gv.judge_client(ledger, 160402012, tmp_path)["verdict"] == "unaccepted"
    stored = json.loads((tmp_path / gv.LOCAL_CLIENTS).read_text(encoding="utf-8"))
    assert [c["version_code"] for c in stored["clients"]] == [160402012, 160700001]


def test_note_client_round_trip(gv, tmp_path):
    when = dt.datetime(2026, 10, 30, 9, 0, tzinfo=UTC)
    rec = gv.note_client(160635001, "16.635.1", source="selfcheck", updates=tmp_path, now=when)
    assert rec == {"version_code": 160635001, "version_name": "16.635.1",
                   "seen_at": "2026-10-30T09:00:00Z", "source": "selfcheck"}
    assert gv.last_seen_client(tmp_path) == rec
    assert gv.note_client("160402020", source="vision_play", updates=tmp_path,
                          now=when)["version_name"] == "16.402.20"
    with pytest.raises(ValueError):
        gv.note_client("12", source="x", updates=tmp_path)
    (tmp_path / gv.LAST_SEEN).write_text("{not json", encoding="utf-8")
    assert gv.last_seen_client(tmp_path) is None
    assert gv.last_seen_client(tmp_path / "nowhere") is None


# ---------------------------------------------------------------------------
# events and days behind
# ---------------------------------------------------------------------------

def test_events_add_rule_seasons_once(gv, ledger):
    got = gv.events(ledger, after=dt.date(2026, 10, 30), upto=dt.date(2026, 12, 31))
    assert [(e["date"], e["kind"], e.get("season")) for e in got] == [
        (dt.date(2026, 11, 2), "season", 89), (dt.date(2026, 12, 7), "season", 90)]
    assert got[0]["forms"] and got[1]["from"] == "the season rule"
    assert gv.events(ledger, after=dt.date(2026, 11, 2), upto=dt.date(2026, 11, 30)) == []
    every = gv.events(ledger)
    assert [e["date"].isoformat() for e in every][:2] == ["2026-07-13", "2026-08-26"]


def test_behind(gv, ledger):
    assert gv.behind(ledger, dt.date(2026, 10, 30)) is None
    late = gv.behind(ledger, dt.date(2026, 11, 5))
    assert (late["since"], late["days"]) == (dt.date(2026, 11, 2), 3)
    assert [e["season"] for e in late["events"]] == [89]
    later = gv.behind(ledger, dt.date(2026, 12, 8))
    assert later["days"] == 36 and [e["season"] for e in later["events"]] == [89, 90]
    # an event whose arm the pinned release has is covered
    covered = copy.deepcopy(ledger)
    covered["content_events"][-1]["table"] = "client160402017_20261019"
    assert gv.behind(covered, dt.date(2026, 11, 5)) is None


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_a_broken_ledger_is_refused(gv, tmp_path):
    good = json.loads(LEDGER.read_text(encoding="utf-8"))
    with pytest.raises(gv.LedgerError, match="missing"):
        gv.load(tmp_path / "none.json")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(gv.LedgerError, match="not JSON"):
        gv.load(tmp_path / "bad.json")
    cases = []
    d = copy.deepcopy(good)
    d["royalesim"]["0.1.42"]["tables"]["client15535"]["fnv1a64"] = "xyz"
    cases.append((d, "fnv1a64"))
    d = copy.deepcopy(good)
    d["clients"][-1]["version_name"] = "16.635.2"
    cases.append((d, "is not 16.635.1"))
    d = copy.deepcopy(good)
    d["pin"]["card_table"] = "client999"
    cases.append((d, "pin.card_table"))
    d = copy.deepcopy(good)
    d["pin"]["royalesim"] = "0.1.99"
    cases.append((d, "no royalesim entry"))
    d = copy.deepcopy(good)
    d["seasons"]["anchor"]["start"] = "2026-11-03"
    cases.append((d, "seasons.anchor"))
    d = copy.deepcopy(good)
    d["content_events"][0]["kind"] = "rumour"
    cases.append((d, "kind"))
    d = copy.deepcopy(good)
    d["clients"].append(dict(d["clients"][0]))
    cases.append((d, "listed twice"))
    d = copy.deepcopy(good)
    d["schema"] = 2
    cases.append((d, "schema"))
    for k, (data, needle) in enumerate(cases):
        with pytest.raises(gv.LedgerError, match=re.escape(needle)):
            gv.load(_write(tmp_path, data, f"case{k}.json"))


# ---------------------------------------------------------------------------
# the currency line
# ---------------------------------------------------------------------------

def _status(**over):
    status = {
        "checked_at": "2026-11-03T00:00:00Z",
        "pin": {"royalesim": "0.1.42", "card_table": "client160402017_20261019",
                "content": "16.402.22"},
        "client": {"version_code": 160635001, "version_name": "16.635.1", "relation": "ahead",
                   "accepted": "2026-10-29", "basis": "owner", "seen": True},
        "royalesim": {"pin": "0.1.42", "newest": "0.1.42", "newer": []},
        "pool": {"pulled_at": "2026-11-03T00:00:00Z"},
        "calendar": {"behind": {"since": "2026-11-02", "days": 1, "events": [
            {"date": "2026-11-02", "kind": "season", "season": 89, "what": "Season 89"}]},
            "upcoming": []},
        "radar": {"gaps": []},
    }
    status.update(over)
    return status


def test_currency_line_is_the_section_4_1_example(gv):
    now = dt.datetime(2026, 11, 3, 6, 0, tzinfo=UTC)
    assert gv.currency_line(_status(), now) == DOC_EXAMPLE
    assert all(len(line) <= gv.LINE_WIDTH for line in DOC_EXAMPLE.splitlines())


def test_currency_line_variants(gv):
    now = dt.datetime(2026, 10, 30, 5, 47, tzinfo=UTC)
    quiet = _status(checked_at="2026-10-30T05:47:00Z", pool={},
                    calendar={"behind": None, "upcoming": [
                        {"date": "2026-11-02", "kind": "season", "season": 89}]})
    assert gv.currency_line(quiet, now).splitlines() == [
        "currency  client 160635001 (16.635.1) vs table 16.402.22: client ahead, accepted "
        "2026-10-29;",
        "          royalesim 0.1.42 = pin, newest on PyPI; checked just now; pool not pulled yet;",
        "          Season 89 starts 2026-11-02"]
    update = _status(royalesim={"newest": "0.1.43", "newer": [{"version": "0.1.43"}]})
    text = gv.currency_line(update, now)
    assert "royalesim 0.1.42 = pin; 0.1.43 on PyPI: UPDATE AVAILABLE" in text
    assert "Season 89 started 2026-11-02: run scripts\\game_update.ps1 -Update" in text
    assert "when RoyaleSim ships it" not in text
    bad = _status(royalesim={"newer": [{"version": "0.1.43", "integrity": False}]})
    assert "0.1.43 on PyPI failed its sha256 check" in gv.currency_line(bad, now)
    assert "PyPI check failed" in gv.currency_line(_status(royalesim={"error": "x"}), now)
    assert "PyPI not checked" in gv.currency_line(_status(royalesim={"skipped": "x"}), now)
    new = _status(client={"version_code": 160700001, "version_name": "16.700.1",
                          "relation": "ahead", "accepted": None, "seen": True})
    assert "client ahead, NOT ACCEPTED" in gv.currency_line(new, now)
    auto = _status(client={"version_code": 160700001, "version_name": "16.700.1",
                           "relation": "ahead", "accepted": "2026-11-04", "basis": "auto",
                           "seen": False})
    assert "accepted 2026-11-04 (auto), not read from the device yet" in \
        " ".join(gv.currency_line(auto, now).split())
    gaps = _status(radar={"gaps": [{"engine": "Giant", "form": "evolution"},
                                   {"engine": "Mirror", "form": "evolution"}]})
    assert "in the game, not in the table: Giant evolution, Mirror evolution" in \
        " ".join(gv.currency_line(gaps, now).split())
    assert gv.currency_line(None).startswith("currency  no update check has run yet")


def test_age_and_time_helpers(gv):
    assert gv.age_text(30) == "just now"
    assert gv.age_text(600) == "10 min ago"
    assert gv.age_text(6 * 3600) == "6 h ago"
    assert gv.age_text(3 * 86400) == "3 days ago"
    assert gv.age_text(None) == "at an unknown time"
    got = gv.parse_time("2026-10-23T04:16:28.314454Z")
    assert got.tzinfo is not None and gv.iso(got) == "2026-10-23T04:16:28Z"
    assert gv.parse_time("2026-10-23") == dt.datetime(2026, 10, 23, tzinfo=UTC)
    assert gv.parse_time("soon") is None and gv.parse_time(None) is None
    assert gv.parse_date("2026-11-02") == dt.date(2026, 11, 2)


# ---------------------------------------------------------------------------
# CLI and imports
# ---------------------------------------------------------------------------

def test_cli(gv, tmp_path, capsys):
    assert gv.main(["--season", "2026"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Season 79  2026-01-05" and lines[-1] == "Season 90  2026-12-07"
    assert gv.main(["--client", "160635001", "--updates", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "ok"
    assert gv.main(["--updates", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("pin        royalesim 0.1.42, client160402017_20261019")
    assert gv.main(["--currency", "--updates", str(tmp_path)]) == 0
    assert "no update check has run yet" in capsys.readouterr().out
    for argv in (["--client", "12"], ["--season", "1"]):
        with pytest.raises(SystemExit) as exc:
            gv.main(argv)
        assert exc.value.code == 2
    assert gv.main(["--ledger", str(tmp_path / "none.json")]) == 1


def test_game_versions_is_a_standard_library_leaf():
    """selfcheck, the live runner and rs_engine (U05, U11) import it, so it imports no tool."""
    tree = ast.parse((TOOLS / "game_versions.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names <= {"__future__", "argparse", "datetime", "json", "os", "re", "sys",
                     "collections", "pathlib", "typing", "hastycr"}, names
