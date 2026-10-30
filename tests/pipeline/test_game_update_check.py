"""U04: `tools/game_update.py check`, the daily version check (docs/SEASON_UPDATES.md 4.1, 5,
7.3, 10 and 12).

No test touches the network. `game_update.fetch` is replaced by a router (Web) that answers from
the hand-written fixtures in tests/pipeline/fixtures/game_update/ (see the README there) and the
tiny archives built here; an unexpected URL fails the test. tools/cr_api.py is hidden
(sys.modules["cr_api"] = None) unless a test installs a fake, so no test can reach the API.

The tiny RoyaleSim sdists, worked by hand:
* 0.1.42 (the pin): four tables, one per ledger arm, each with Giant (hero form), Mirror and
  Skeletons (an evolution). The ledger copy the tests load carries these tables' FNVs.
* 0.1.43: the same four, except that cards-160402017-20261019 has Skeletons'
  summon_radius_milli 550 instead of 500 (same name, other bytes), plus a fifth arm
  client160635001_20261102 (client build 160635002, content 16.635.4) whose table adds
  Giant_EV1 and Mirror_EV1. cards.json repeats cards-15.535.json. Two upstream tests.
  So: arms added [client160635001_20261102]; statuses same, same, same, bytes changed, new;
  the newest table is 16.635.4.
* The /v1/cards fixtures: on 2026-10-30 every card's forms match data/vision/classes.json
  (no gap), Minion Giant has no card-map key (the world fixture removes the minion-giant
  extra) and matches the catalogue's MinionGiant at 4 elixir (auto). On 2026-11-02 Giant
  (2 -> 3) and Mirror (0 -> 1) gain an evolution, which classes.json lacks (Giant [0, 2],
  Mirror [0]) and 0.1.43's newest table has; The Log is renamed "Log" (slug "log" has no
  key, so the official id 28000011 finds "the-log" through data/royaleapi/cards.json);
  Fixture Golem (26000999) is new and in no table.
* Calendar: the check of 2026-11-03 05:47 after one on 2026-10-30 05:47 has Season 89
  (2026-11-02) since the last check, and is 1 day behind (no table has it).
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import hashlib
import importlib
import io
import json
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
FIX = Path(__file__).resolve().parent / "fixtures" / "game_update"
UTC = dt.timezone.utc
NOW_1030 = dt.datetime(2026, 10, 30, 5, 47, tzinfo=UTC)
NOW_1103 = dt.datetime(2026, 11, 3, 5, 47, tzinfo=UTC)
PYPI_ENGINE = "https://pypi.org/pypi/royalesim/json"
PYPI_MAP = "https://pypi.org/pypi/royaleimitate/json"
HF = "https://huggingface.co/api/datasets/VanguardX101/IL_Replay"
NEW_ARM = "client160635001_20261102"
OLD_ARMS = ["client15535", "client160402017_20261006", "client160402017",
            "client160402017_20261019"]


@pytest.fixture
def gu(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "cr_api", None)     # never the real API client
    return importlib.import_module("game_update")


@pytest.fixture
def gv(gu):
    return importlib.import_module("game_versions")


@pytest.fixture
def rs_engine(gu):
    return importlib.import_module("rs_engine")


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# tiny archives
# ---------------------------------------------------------------------------

def table(version, *, build=None, content=None, manifest=None, evolutions=(), radius=500):
    if build:
        source = f"Supercell csv_logic/ of the fixture tables (client build {build})"
        dating = (f"The build and content are in the files (MANIFEST.json build {build}, "
                  f"content_version {content}).")
    else:
        source = "Supercell csv_logic/ of the fixture client"
        dating = f"The client version is in the file (MANIFEST.json version {manifest})."
    return {
        "version": version,
        "vintage_warning": "A FIXTURE TABLE. Every number in this file is invented.",
        "provenance": {"source": source, "roster_dating": dating, "vintage": "fixture",
                       "vintage_key": version[len("cards-"):-len(".1")]},
        "cards": [{"name": "Giant", "elixir": 5, "hitpoints": 2000},
                  {"name": "Mirror", "elixir": 1},
                  {"name": "Skeletons", "elixir": 1, "summon_radius_milli": radius}],
        "evolutions": [{"name": "Skeletons_EV1", "form_of": "Skeletons"}]
        + [{"name": f"{c}_EV1", "form_of": c} for c in evolutions],
        "hero_forms": [{"name": "Giant_hero", "form_of": "Giant"}],
        "towers": [{"name": "PrincessTower"}, {"name": "KingTower"}],
        "units": {"Giant": {"name": "Giant"}, "Skeleton": {"name": "Skeleton"}},
    }


def dumps(obj) -> bytes:
    return (json.dumps(obj, indent=1) + "\n").encode("utf-8")


def tables_042():
    """arm -> (file name, table) of the pinned release."""
    return {
        "client15535": ("cards-15.535.json", table("cards-15535.1", manifest="15.535.29")),
        "client160402017_20261006": ("cards-160402017-20261006.json", table(
            "cards-160402017-20261006.1", build=160402017, content="16.402.19")),
        "client160402017": ("cards-160402017.json", table(
            "cards-160402017.1", build=160402017, content="16.402.2")),
        "client160402017_20261019": ("cards-160402017-20261019.json", table(
            "cards-160402017-20261019.1", build=160402020, content="16.402.22")),
    }


def tables_043():
    out = tables_042()
    out["client160402017_20261019"] = ("cards-160402017-20261019.json", table(
        "cards-160402017-20261019.1", build=160402020, content="16.402.22", radius=550))
    out[NEW_ARM] = ("cards-160635001-20261102.json", table(
        "cards-160635001-20261102.1", build=160635002, content="16.635.4",
        evolutions=("Giant", "Mirror")))
    return out


def calibration(arms):
    return {"$comment": ["a fixture ledger"],
            "meta": {"target_game_version": "live-2026", "schema_version": 1},
            "cards": {"CARD_TABLE": {"value": "client15535", "candidates": list(arms),
                                     "status": "datamined", "provenance": "fixture"}}}


def tar_bytes(top: str, members: dict[str, bytes], links=()) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel, data in members.items():
            info = tarfile.TarInfo(f"{top}/{rel}")
            info.size, info.mtime = len(data), 0
            tf.addfile(info, io.BytesIO(data))
        for rel, target in links:
            info = tarfile.TarInfo(f"{top}/{rel}")
            info.type, info.linkname = tarfile.SYMTYPE, target
            tf.addfile(info)
    return buf.getvalue()


def engine_sdist(version="0.1.43", tables=None, *, links=()) -> bytes:
    tables = tables_043() if tables is None else tables
    members = {"PKG-INFO": b"Metadata-Version: 2.1\nName: royalesim\n",
               "data/calibration.json": dumps(calibration(tables)),
               "crates/royalesim/tests/card_table.rs": b"// fixture\n",
               "crates/royalesim/tests/evo_giant.rs": b"// fixture\n",
               "crates/royalesim/src/py.rs": b"// fixture\n"}
    for name, t in tables.values():
        members[f"data/derived/{name}"] = dumps(t)
    members["data/derived/cards.json"] = dumps(tables["client15535"][1])
    return tar_bytes(f"royalesim-{version}", members, links)


def vendored_map() -> bytes:
    return (ROOT / "data" / "cards" / "replay_cards.json").read_bytes()


def map_with(extra: dict[str, str]) -> bytes:
    data = json.loads(vendored_map())
    data.update(extra)
    return dumps(data)


def map_sdist(version: str, raw: bytes) -> bytes:
    return tar_bytes(f"royaleimitate-{version}", {"PKG-INFO": b"Name: royaleimitate\n",
                                                   "royaleimitate/replay_cards.json": raw})


def map_wheel(raw: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("royaleimitate/replay_cards.json", raw)
        zf.writestr("royaleimitate-0.2.16.dist-info/RECORD", "")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# the fake web and the world
# ---------------------------------------------------------------------------

class Web:
    """fetch(url, **kw) from a table of routes; records every URL asked for."""

    def __init__(self, gu):
        self.gu = gu
        self.routes: dict[str, object] = {}
        self.calls: list[str] = []

    def json(self, url, obj, status=200):
        self.routes[url] = self.gu.Fetched(status, json.dumps(obj).encode("utf-8"),
                                           {"content-type": "application/json"})

    def raw(self, url, data, status=200):
        self.routes[url] = self.gu.Fetched(status, data, {})

    def __call__(self, url, **kw):
        self.calls.append(url)
        if url not in self.routes:
            raise AssertionError(f"unexpected request: {url}")
        return self.routes[url]


def _fill(project_json, files: dict[str, bytes], good: bool = True):
    """Give every "FIXTURE" digest the sha256 of the archive built for that file name."""
    for entries in project_json["releases"].values():
        for entry in entries:
            if entry["digests"].get("sha256") == "FIXTURE" and entry["filename"] in files:
                digest = hashlib.sha256(files[entry["filename"]]).hexdigest()
                entry["digests"]["sha256"] = digest if good else "0" * 64
    return project_json


def _url_of(project_json, filename):
    for entries in project_json["releases"].values():
        for entry in entries:
            if entry["filename"] == filename:
                return entry["url"]
    raise KeyError(filename)


class World(types.SimpleNamespace):
    def options(self, **over):
        opts = dict(updates=self.updates, data_root=self.data, ledger=self.ledger,
                    cards=FIX / "cards_api_2026-10-30.json", now=NOW_1030)
        opts.update(over)
        return self.gu.Options(**opts)

    def last_check(self, when: str):
        self.gv.write_json(self.updates / "state.json", {"schema": 1, "last_check": when})

    def previous_cards(self, name="cards_api_2026-10-30.json"):
        self.gv.write_json(self.updates / self.gu.CARDS_SNAPSHOT,
                           {"fetched_at": "2026-10-29T05:47:00Z", "data": load(name)})

    def seen(self, code, when="2026-10-30T05:00:00Z"):
        self.gv.note_client(code, source="selfcheck", updates=self.updates,
                            now=self.gv.parse_time(when))


@pytest.fixture
def world(gu, gv, rs_engine, tmp_path, monkeypatch):
    """A tmp ledger whose 0.1.42 FNVs are the tiny tables', PyPI with no newer release, the
    vendored royaleimitate only, and the Hugging Face answer. Tests add what they need."""
    data = json.loads((ROOT / "data" / "game_versions.json").read_text(encoding="utf-8"))
    for arm, (_name, t) in tables_042().items():
        data["royalesim"]["0.1.42"]["tables"][arm]["fnv1a64"] = rs_engine.fnv1a64(dumps(t))
    ledger = tmp_path / "game_versions.json"
    ledger.write_text(json.dumps(data), encoding="utf-8")
    web = Web(gu)
    monkeypatch.setattr(gu, "fetch", web)
    # the radar tests run without the "minion-giant" extra (added from the proposal they check),
    # so the fixtures' Minion Giant is still the unmapped card that makes it
    rs_cards = importlib.import_module("rs_cards")
    live_load = rs_cards.load
    monkeypatch.setattr(rs_cards, "load", lambda *a, **kw: {
        k: v for k, v in live_load(*a, **kw).items() if k != "minion-giant"})
    w = World(gu=gu, gv=gv, web=web, ledger=ledger, updates=tmp_path / "updates",
              data=tmp_path / "data", files={})
    pypi = load("pypi_royalesim.json")
    for v in ("0.1.43", "0.1.44rc1"):
        del pypi["releases"][v]
    pypi["info"]["version"], pypi["urls"] = "0.1.42", []
    web.json(PYPI_ENGINE, pypi)
    ri = load("pypi_royaleimitate.json")
    del ri["releases"]["0.2.16"]
    ri["info"]["version"] = "0.2.15"
    web.json(PYPI_MAP, ri)
    web.json(HF, load("hf_il_replay.json"))
    return w


def with_release(w, *, good=True, sdist=None):
    """PyPI gains 0.1.43 (and the 0.1.44rc1 pre-release) with the tiny sdist."""
    pypi = load("pypi_royalesim.json")
    files = {"royalesim-0.1.43.tar.gz": engine_sdist() if sdist is None else sdist,
             "royalesim-0.1.44rc1.tar.gz": engine_sdist("0.1.44rc1")}
    _fill(pypi, files, good)
    w.web.json(PYPI_ENGINE, pypi)
    for name, data in files.items():
        w.web.raw(_url_of(pypi, name), data)
    w.files.update(files)
    return pypi


def with_map_release(w, raw: bytes, *, sdist=True):
    """PyPI gains royaleimitate 0.2.16: a wheel of `raw` and, unless sdist=False, an sdist.
    Only the files the release lists are routed, so the check asking for one it does not list
    fails the test."""
    ri = load("pypi_royaleimitate.json")
    files = {"royaleimitate-0.2.16.tar.gz": map_sdist("0.2.16", raw),
             "royaleimitate-0.2.16-py3-none-any.whl": map_wheel(raw)}
    if not sdist:
        ri["releases"]["0.2.16"] = [e for e in ri["releases"]["0.2.16"]
                                    if e["packagetype"] != "sdist"]
        del files["royaleimitate-0.2.16.tar.gz"]
    _fill(ri, files)
    w.web.json(PYPI_MAP, ri)
    for name, data in files.items():
        w.web.raw(_url_of(ri, name), data)
    return ri


# ---------------------------------------------------------------------------
# whole checks
# ---------------------------------------------------------------------------

def test_quiet_day_is_exit_0_with_the_currency_line(world, gu):
    world.last_check("2026-10-29T05:47:00Z")
    world.previous_cards()
    world.seen(160635001)
    code, status = gu.run_check(world.options())
    assert code == 0 and status["verdict"] == "OK"
    assert status["warnings"] == [] and status["errors"] == []
    assert world.web.calls == [PYPI_ENGINE, PYPI_MAP, HF]          # no download
    assert status["royalesim"]["newer"] == [] and status["royalesim"]["newest"] == "0.1.42"
    assert status["royaleimitate"]["newest"] == "0.2.15"
    rad = status["radar"]
    assert (rad["cards"], rad["support"], rad["baseline"], rad["changed"]) == (13, 1, False, False)
    assert rad["gaps"] == [] and rad["not_in_engine"] == []
    assert [(u["name"], u["auto"], u["key"]) for u in rad["unmapped"]] == [
        ("Minion Giant", "MinionGiant", "minion-giant")]
    assert rad["observed"]["maxEvolutionLevel"] == {"0": 1, "1": 3, "2": 3, "3": 2, "absent": 4}
    assert rad["observed"]["supportItems"] == 1
    assert any('"minion-giant": {"name": "MinionGiant", "verified": false}' in n
               for n in status["notes"])
    cal = status["calendar"]
    assert cal["passed"] == [] and cal["behind"] is None
    assert [(e["date"], e["season"]) for e in cal["upcoming"]] == [("2026-11-02", 89)]
    assert (cal["season"], cal["next_season"]) == ({"number": 88, "start": "2026-10-05"},
                                                   {"number": 89, "start": "2026-11-02"})
    client = status["client"]
    assert (client["verdict"], client["relation"], client["seen"]) == ("ok", "ahead", True)
    assert status["il_replay"]["first_seen"] is True
    text = (world.updates / "status.txt").read_text(encoding="utf-8").splitlines()
    assert text[:3] == [
        "currency  client 160635001 (16.635.1) vs table 16.402.22: client ahead, accepted "
        "2026-10-29;",
        "          royalesim 0.1.42 = pin, newest on PyPI; checked just now; pool not pulled yet;",
        "          Season 89 starts 2026-11-02"]
    assert text[-1] == "exit 0: OK; next: nothing to do"
    saved = json.loads((world.updates / "status.json").read_text(encoding="utf-8"))
    assert saved["line"] == "\n".join(text[:3]) and saved["exit"] == 0
    state = json.loads((world.updates / "state.json").read_text(encoding="utf-8"))
    assert state["last_check"] == "2026-10-30T05:47:00Z" and state["last_exit"] == 0
    assert state["il_replay"]["sha"] == "a" * 40 and "noisy_on" not in state


def test_a_new_release_is_exit_10_with_arms_and_changed_bytes(world, gu, rs_engine):
    with_release(world)
    world.last_check("2026-10-30T05:47:00Z")
    world.previous_cards()
    code, status = gu.run_check(world.options(cards=FIX / "cards_api_2026-11-02.json",
                                              now=NOW_1103))
    assert code == 10 and status["verdict"] == "UPDATE AVAILABLE"
    rs = status["royalesim"]
    assert (rs["newest"], rs["pre"], rs["yanked"]) == ("0.1.43", ["0.1.44rc1"], [])
    (item,) = rs["newer"]
    assert item["read"] and item["archive"]["verified"] and item["archive"]["kind"] == "sdist"
    assert item["uploaded"] == "2026-11-02T20:10:00.000000Z"
    assert item["upstream_tests"] == 2
    assert item["ledger"] == {"found": True, "arms": OLD_ARMS + [NEW_ARM],
                              "default_arm": "client15535"}
    cmp_ = item["compare"]
    assert cmp_["arms_added"] == [NEW_ARM] and cmp_["arms_removed"] == []
    assert {r["arm"]: r["status"] for r in cmp_["tables"]} == {
        "client15535": "same", "client160402017_20261006": "same", "client160402017": "same",
        "client160402017_20261019": "bytes changed", NEW_ARM: "new"}
    changed = next(r for r in cmp_["tables"] if r["arm"] == "client160402017_20261019")
    assert changed["fnv1a64"] == rs_engine.fnv1a64(dumps(tables_043()[changed["arm"]][1]))
    assert changed["old"]["fnv1a64"] == rs_engine.fnv1a64(dumps(tables_042()[changed["arm"]][1]))
    assert cmp_["changed_tables"] == ["client160402017_20261019", NEW_ARM]
    newest = cmp_["newest_table"]
    assert (newest["arm"], newest["content"], newest["client_build"],
            newest["content_date"]) == (NEW_ARM, "16.635.4", 160635002, "2026-11-02")
    assert {t["file"] for t in item["tables"]} >= {"data/derived/cards.json",
                                                   "data/derived/cards-15.535.json"}
    assert all("cards" not in t for t in item["tables"])          # slimmed for the status
    assert any(w.startswith("royalesim 0.1.43 on PyPI") and f"new arms {NEW_ARM}" in w
               and "client160402017_20261019 (cards-160402017-20261019.1): same name, other "
               "bytes" in w for w in status["warnings"])
    assert any("0.1.44rc1 is a pre-release" in n for n in status["notes"])
    # the radar names the release that has the missing forms
    rad = status["radar"]
    assert rad["changed"] is True
    gained = {(g["name"], g["form"]): g for g in rad["bits_gained"]}
    assert set(gained) == {("Giant", "evolution"), ("Mirror", "evolution")}
    assert all(g["simulator_lacks"] and g["newer"] == f"royalesim 0.1.43 ({NEW_ARM})"
               for g in gained.values())
    assert f"Giant gained an evolution; the simulator lacks it; royalesim 0.1.43 ({NEW_ARM}) " \
           "has it" in status["warnings"]
    assert {(g["engine"], g["form"]) for g in rad["gaps"]} == {("Giant", "evolution"),
                                                              ("Mirror", "evolution")}
    assert rad["renamed"] == [{"id": 28000011, "old": "The Log", "new": "Log", "engine": "Log"}]
    assert [(c["name"], c["in_engine"]) for c in rad["new_cards"]] == [("Fixture Golem", False)]
    assert "new card in the game: Fixture Golem (id 26000999), not in the engine" in \
        status["warnings"]
    assert "card 28000011 renamed: The Log -> Log" in status["warnings"]
    # the calendar
    cal = status["calendar"]
    assert [(e["date"], e["season"]) for e in cal["passed"]] == [("2026-11-02", 89)]
    assert (cal["behind"]["since"], cal["behind"]["days"]) == ("2026-11-02", 1)
    line = " ".join(status["line"].split())
    assert "royalesim 0.1.42 = pin; 0.1.43 on PyPI: UPDATE AVAILABLE" in line
    assert "Season 89 started 2026-11-02: run scripts\\game_update.ps1 -Update;" in line
    assert "in the game, not in the table: Giant evolution, Mirror evolution" in line
    assert status["next"].endswith("-Update (royalesim 0.1.43)")
    cache = world.updates / "cache"
    assert (cache / "royalesim-0.1.43.tar.gz").read_bytes() == world.files[
        "royalesim-0.1.43.tar.gz"]
    assert (cache / "royalesim-0.1.43.tar.gz.summary.json").is_file()
    # a second check reads the cached summary: the sdist is downloaded once
    code2, status2 = gu.run_check(world.options(cards=FIX / "cards_api_2026-11-02.json",
                                                now=NOW_1103 + dt.timedelta(hours=1)))
    sdist_url = _url_of(load("pypi_royalesim.json"), "royalesim-0.1.43.tar.gz")
    assert code2 == 10 and world.web.calls.count(sdist_url) == 1
    assert status2["royalesim"]["newer"][0]["compare"] == cmp_
    assert status2["radar"]["changed"] is False                    # compared with run 1's


def test_pre_releases_count_with_pre(world, gu):
    with_release(world)
    code, status = gu.run_check(world.options(pre=True, skip=frozenset({"radar"})))
    assert code == 10
    assert [i["version"] for i in status["royalesim"]["newer"]] == ["0.1.43", "0.1.44rc1"]
    assert status["royalesim"]["newest"] == "0.1.44rc1" and status["royalesim"]["pre"] == []


def test_max_sdists_lists_older_releases_unread(world, gu):
    with_release(world)
    code, status = gu.run_check(world.options(pre=True, max_sdists=1,
                                              skip=frozenset({"radar"})))
    first, second = status["royalesim"]["newer"]
    assert (first["version"], first["read"]) == ("0.1.43", False)
    assert "--max-sdists 1" in first["note"] and second["read"] is True
    assert code == 10


def test_a_sha256_mismatch_is_an_error_and_nothing_is_kept(world, gu):
    pypi = with_release(world, good=False)
    code, status = gu.run_check(world.options(skip=frozenset({"radar"})))
    assert code == 1 and status["verdict"] == "ERROR"
    assert len(status["errors"]) == 1 and "sha256" in status["errors"][0]
    assert "not PyPI's" in status["errors"][0]
    (item,) = status["royalesim"]["newer"]
    assert item["integrity"] is False and item["read"] is False
    url = _url_of(pypi, "royalesim-0.1.43.tar.gz")
    assert world.web.calls.count(url) == 2                        # one more try, then refused
    assert not (world.updates / "cache" / "royalesim-0.1.43.tar.gz").exists()
    assert "0.1.43 on PyPI failed its sha256 check" in " ".join(status["line"].split())


def test_a_release_without_sdist_still_counts(world, gu):
    pypi = load("pypi_royalesim.json")
    del pypi["releases"]["0.1.44rc1"]
    pypi["releases"]["0.1.43"] = [e for e in pypi["releases"]["0.1.43"]
                                  if e["packagetype"] != "sdist"]
    world.web.json(PYPI_ENGINE, pypi)
    code, status = gu.run_check(world.options(skip=frozenset({"radar"})))
    (item,) = status["royalesim"]["newer"]
    assert code == 10 and item["read"] is False and "no sdist" in item["note"]
    assert any("royalesim 0.1.43: no sdist" in n for n in status["notes"])


def test_pypi_unreachable_is_exit_1(world, gu):
    world.web.json(PYPI_ENGINE, {"message": "unavailable"}, status=503)
    code, status = gu.run_check(world.options())
    assert code == 1 and status["royalesim"]["error"].startswith("PyPI ")
    assert any(e.startswith("PyPI: ") for e in status["errors"])
    assert "PyPI check failed" in status["line"]


def test_a_broken_ledger_is_exit_1_and_still_writes_a_status(world, gu, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    code, status = gu.run_check(world.options(ledger=bad))
    assert code == 1 and status["errors"][0].startswith("ledger: ")
    assert world.web.calls == []
    assert (world.updates / "status.txt").read_text(encoding="utf-8").rstrip().endswith(
        "fix data/game_versions.json (git checkout -- data/game_versions.json)")


def test_eight_days_behind_says_to_ask_upstream(world, gu):
    """Section 10: 2026-11-10 is 8 days after Season 89 with no table for it."""
    world.last_check("2026-11-09T05:47:00Z")
    code, status = gu.run_check(world.options(skip=frozenset(gu.PARTS),
                                              now=dt.datetime(2026, 11, 10, 5, 47, tzinfo=UTC)))
    assert code == 0 and status["calendar"]["behind"]["days"] == 8
    assert any(w.startswith("8 days with no RoyaleSim release for it: ask upstream")
               for w in status["warnings"])
    assert ("Season 89 started 2026-11-02: run scripts\\game_update.ps1 -Update when RoyaleSim "
            "ships it") in " ".join(status["line"].split())
    assert status["next"].startswith("nothing to install yet")


def test_skip_everything_makes_no_request(world, gu):
    world.seen(160635001)
    code, status = gu.run_check(world.options(skip=frozenset(gu.PARTS)))
    assert code == 0 and world.web.calls == []
    assert "royalesim 0.1.42 = pin; PyPI not checked" in status["line"]
    assert all(status[k].get("skipped") for k in ("royalesim", "royaleimitate", "radar",
                                                    "il_replay"))


# ---------------------------------------------------------------------------
# a new client build
# ---------------------------------------------------------------------------

def test_a_new_client_on_a_noisy_day_waits_for_the_owner(world, gu, gv):
    """Section 4.1: only the check of the day the build appears may accept it. 2026-11-03 has
    the Season 89 roster change (Giant and Mirror gain evolutions the table lacks), so the build
    first seen that day stays a WARN and exit 11 on every later day, quiet or not."""
    world.previous_cards()
    world.seen(160700001, "2026-11-03T05:00:00Z")
    world.last_check("2026-11-02T05:47:00Z")
    cards = FIX / "cards_api_2026-11-02.json"
    # 1. the radar changed today: a warning, exit 11
    code, status = gu.run_check(world.options(cards=cards, now=NOW_1103))
    assert code == 11 and status["verdict"] == "NEW CLIENT"
    assert status["client"]["verdict"] == "new"
    assert status["client"]["first_checked"] == "2026-11-03"
    assert any(w.startswith("client 160700001 (16.700.1) is not in data/game_versions.json")
               for w in status["warnings"])
    assert "-AcceptClient" in status["next"]
    assert "client ahead, NOT ACCEPTED" in " ".join(status["line"].split())
    state = json.loads((world.updates / "state.json").read_text(encoding="utf-8"))
    assert state["noisy_on"] == "2026-11-03"
    assert state["pending_client"] == {"version_code": 160700001, "first_checked": "2026-11-03"}
    # 2. the same day again: the radar now compares with run 1's snapshot (no change), but
    # the day stays noisy
    code, status = gu.run_check(world.options(cards=cards, now=NOW_1103 + dt.timedelta(hours=2)))
    assert code == 11 and status["radar"]["changed"] is False
    assert not (world.updates / gv.LOCAL_CLIENTS).exists()
    # 3. the next day nothing changes, but it is not the day the build appeared: still exit 11
    code, status = gu.run_check(world.options(cards=cards, now=NOW_1103 + dt.timedelta(days=1)))
    assert code == 11 and status["radar"]["changed"] is False
    assert "auto_accepted" not in status["client"]
    assert status["client"]["first_checked"] == "2026-11-03"
    assert any("first checked on 2026-11-03, a day that was not quiet" in n
               for n in status["notes"])
    assert not (world.updates / gv.LOCAL_CLIENTS).exists()
    state = json.loads((world.updates / "state.json").read_text(encoding="utf-8"))
    assert state["pending_client"]["first_checked"] == "2026-11-03"
    # 4. a newer build is a new decision, on its own first day
    world.seen(160700002, "2026-11-05T05:00:00Z")
    code, status = gu.run_check(world.options(cards=cards, now=NOW_1103 + dt.timedelta(days=2)))
    assert code == 0 and status["client"]["auto_accepted"] is True
    assert status["client"]["version_code"] == 160700002
    state = json.loads((world.updates / "state.json").read_text(encoding="utf-8"))
    assert "pending_client" not in state


def test_a_new_client_on_a_quiet_first_day_is_accepted_by_itself(world, gu, gv):
    world.previous_cards()
    world.seen(160700001, "2026-10-30T05:00:00Z")
    world.last_check("2026-10-29T05:47:00Z")
    code, status = gu.run_check(world.options())                  # 2026-10-30: nothing new
    assert code == 0 and status["client"]["auto_accepted"] is True
    assert (status["client"]["basis"], status["client"]["accepted"]) == ("auto", "2026-10-30")
    assert any("accepted by itself (basis auto)" in n for n in status["notes"])
    stored = json.loads((world.updates / gv.LOCAL_CLIENTS).read_text(encoding="utf-8"))
    (entry,) = stored["clients"]
    assert (entry["version_code"], entry["basis"], entry["first_seen"]) == (
        160700001, "auto", "2026-10-30")
    ledger = gv.load(world.ledger)
    assert gv.judge_client(ledger, 160700001, world.updates)["verdict"] == "ok"
    assert "accepted 2026-10-30 (auto)" in " ".join(status["line"].split())
    state = json.loads((world.updates / "state.json").read_text(encoding="utf-8"))
    assert "pending_client" not in state
    # and it stays accepted
    code, status = gu.run_check(world.options(now=NOW_1030 + dt.timedelta(days=1)))
    assert code == 0 and "auto_accepted" not in status["client"]


def test_a_new_client_is_not_accepted_on_a_season_start(world, gu):
    world.previous_cards()
    world.seen(160700001, "2026-11-02T09:00:00Z")
    world.last_check("2026-11-01T05:47:00Z")
    code, status = gu.run_check(world.options(now=dt.datetime(2026, 11, 2, 12, 0, tzinfo=UTC)))
    assert status["radar"]["changed"] is False
    assert [e["season"] for e in status["calendar"]["today"]] == [89]
    assert code == 11


def test_a_new_client_is_not_accepted_without_the_radar(world, gu):
    world.previous_cards()
    world.seen(160700001)
    world.last_check("2026-10-29T05:47:00Z")
    code, status = gu.run_check(world.options(cards=None))
    assert status["radar"]["skipped"].endswith("tools/cr_api.py (U02) is not there")
    assert any(n.startswith("radar skipped") for n in status["notes"])
    assert code == 11


def test_bad_client_code_is_an_error(world, gu):
    code, status = gu.run_check(world.options(client="12", skip=frozenset(gu.PARTS)))
    assert code == 1 and any(e.startswith("client: ") for e in status["errors"])


def test_without_a_device_read_the_newest_accepted_build_stands_in(world, gu):
    code, status = gu.run_check(world.options(skip=frozenset(gu.PARTS)))
    client = status["client"]
    assert code == 0 and client["seen"] is False and client["version_code"] == 160635001
    assert "not read from the device yet" in " ".join(status["line"].split())


# ---------------------------------------------------------------------------
# the radar's sources
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, data, ok=True, status=200, hint="ok"):
        self.data, self.ok, self.status, self._hint = data, ok, status, hint

    def hint(self):
        return self._hint


def fake_cr_api(monkeypatch, response=None, error=None):
    made = []

    class Client:
        def __init__(self, **kw):
            if error is not None:
                raise error
            made.append(kw)

        def cards(self):
            return response

    mod = types.ModuleType("cr_api")
    mod.Client = Client
    mod.slug = lambda name: "-".join(
        "".join(ch if ch.isalnum() else " " for ch in name.lower().replace(".", "")).split())
    monkeypatch.setitem(sys.modules, "cr_api", mod)
    return made


def test_the_radar_calls_cr_api_once_at_one_per_second(world, gu, monkeypatch):
    made = fake_cr_api(monkeypatch, FakeResponse(load("cards_api_2026-10-30.json")))
    code, status = gu.run_check(world.options(cards=None, skip=frozenset({"pypi"})))
    assert made == [{"per_second": 1.0}]
    rad = status["radar"]
    assert rad["source"] == "GET /v1/cards (tools/cr_api.py)" and rad["baseline"] is True
    assert any("first snapshot recorded" in n for n in status["notes"])
    saved = json.loads((world.updates / gu.CARDS_SNAPSHOT).read_text(encoding="utf-8"))
    assert saved["data"]["items"][0]["name"] == "Knight"
    assert saved["fetched_at"] == "2026-10-30T05:47:00Z"


def test_the_radar_degrades_when_cr_api_refuses(world, gu, monkeypatch):
    fake_cr_api(monkeypatch, error=ValueError("no token file"))
    _, status = gu.run_check(world.options(cards=None, skip=frozenset({"pypi"})))
    assert status["radar"]["skipped"] == ("no /v1/cards snapshot: no API client (ValueError: "
                                          "no token file)")
    fake_cr_api(monkeypatch, FakeResponse(None, ok=False, status=403, hint="403: access denied"))
    _, status = gu.run_check(world.options(cards=None, skip=frozenset({"pypi"})))
    assert status["radar"]["skipped"] == "no /v1/cards snapshot: GET /v1/cards: 403: access denied"


def test_the_radar_call_follows_the_opponents_config(world, gu, gv, monkeypatch):
    """An owner on the proxy (a changing home IP) set it in <data>/opponents/config.json; the
    radar's own call must go the same way, or it 403s whenever the pull's snapshot is old."""
    made = fake_cr_api(monkeypatch, FakeResponse(load("cards_api_2026-10-30.json")))
    gv.write_json(world.data / "opponents" / "config.json",
                  {"schema": 1, "base": "proxy", "transport": "curl", "rate": 2.0})
    gu.run_check(world.options(cards=None, skip=frozenset({"pypi"})))
    assert made == [{"per_second": 1.0, "base": "proxy", "transport": "curl"}]
    assert gu.api_settings(None) == {} and gu.api_settings(world.data / "none") == {}


def test_the_radar_note_holds_no_ip_address(world, gu, monkeypatch):
    said = ("403: The key is bound to one public IP. Use --base proxy with 45.79.218.79 on the "
            "key. The API said: accessDenied.invalidIp: API key does not allow access from IP "
            "198.51.100.7")
    fake_cr_api(monkeypatch, FakeResponse(None, ok=False, status=403, hint=said))
    sys.modules["cr_api"].PROXY_IP = "45.79.218.79"
    _, status = gu.run_check(world.options(cards=None, skip=frozenset({"pypi"})))
    skipped = status["radar"]["skipped"]
    assert "198.51.100.7" not in skipped and "from IP <ip>" in skipped
    assert "45.79.218.79" in skipped                              # the proxy's address stays
    for name in ("status.json", "status.txt"):
        assert "198.51.100.7" not in (world.updates / name).read_text(encoding="utf-8")
    assert gu.mask_ips("10.0.0.1 and 1.2.3.4.5 and v1.2.3") == "<ip> and 1.2.3.4.5 and v1.2.3"


def test_a_fresh_pull_snapshot_is_used_before_the_api(world, gu, gv, monkeypatch):
    made = fake_cr_api(monkeypatch, FakeResponse(load("cards_api_2026-10-30.json")))
    snap = world.data / "opponents" / "cards_api.json"
    gv.write_json(snap, {"fetched_at": "2026-10-30T04:47:00Z",
                         "data": load("cards_api_2026-11-02.json")})
    _, status = gu.run_check(world.options(cards=None, skip=frozenset({"pypi"})))
    assert made == [] and status["radar"]["source"] == "<data>/opponents/cards_api.json"
    assert status["radar"]["cards"] == 14
    gv.write_json(snap, {"fetched_at": "2026-10-29T04:47:00Z",
                         "data": load("cards_api_2026-11-02.json")})       # 25 h old
    _, status = gu.run_check(world.options(cards=None, skip=frozenset({"pypi"})))
    assert made == [{"per_second": 1.0}] and status["radar"]["cards"] == 13


def test_radar_unit_resolution(gu):
    rs_cards = importlib.import_module("rs_cards")
    engine = gu.engine_cards(gu.CLASSES_PATH)
    keys = gu.royaleapi_keys(gu.ROYALEAPI_CARDS)
    card_map = rs_cards.load()
    out = gu.radar(load("cards_api_2026-11-02.json"), load("cards_api_2026-10-30.json"),
                   engine=engine, card_map=card_map, keys_by_id=keys, slug=gu._slug_fallback)
    assert out["bits_lost"] == [] and out["removed"] == [] and out["new_support"] == []
    assert all(g["newer"] is None for g in out["bits_gained"])
    warn, info = gu.radar_messages(out)
    assert "Giant gained an evolution; the simulator lacks it" in warn
    assert "Fixture Golem (id 26000999) has no card-map key and no catalogue match" in warn
    # without classes.json no form or catalogue claim is made
    bare = gu.radar(load("cards_api_2026-11-02.json"), None, engine=None, card_map=card_map,
                    keys_by_id=keys, slug=gu._slug_fallback)
    assert bare["gaps"] == [] and bare["not_in_engine"] == [] and bare["baseline"] is True
    assert bare["engine_forms"] is None
    assert gu._slug_fallback("Mini P.E.K.K.A") == "mini-pekka"
    assert gu._slug_fallback("X-Bow") == "x-bow"
    assert gu.unwrap_cards({"data": {"items": []}}) == {"items": []}
    assert gu.unwrap_cards({"nothing": 1}) is None


# ---------------------------------------------------------------------------
# royaleimitate and IL_Replay
# ---------------------------------------------------------------------------

def test_royaleimitate_new_keys_are_a_warning_not_an_exit(world, gu):
    with_map_release(world, map_with({"minion-giant": "MinionGiant"}))
    code, status = gu.run_check(world.options(skip=frozenset({"radar"})))
    assert code == 0
    (item,) = status["royaleimitate"]["newer"]
    assert item["added"] == {"minion-giant": "MinionGiant"} and item["removed"] == []
    assert item["same_as_vendored"] is False and item["archive"]["kind"] == "sdist"
    assert ("royaleimitate 0.2.16 has a new card map (keys added minion-giant -> MinionGiant): "
            "take it as data/cards/README.md says") in status["warnings"]


def test_royaleimitate_same_map_from_the_wheel(world, gu):
    with_map_release(world, vendored_map(), sdist=False)
    code, status = gu.run_check(world.options(skip=frozenset({"radar"})))
    (item,) = status["royaleimitate"]["newer"]
    assert code == 0 and item["archive"]["kind"] == "wheel"
    assert item["same_as_vendored"] is True
    assert "royaleimitate 0.2.16: replay_cards.json unchanged" in status["notes"]


def test_il_replay_revision_change_is_reported(world, gu):
    gu.run_check(world.options(skip=frozenset({"radar", "pypi", "royaleimitate"})))
    hf = load("hf_il_replay.json")
    hf["sha"], hf["lastModified"] = "b" * 40, "2026-11-01T00:00:00.000Z"
    world.web.json(HF, hf)
    code, status = gu.run_check(world.options(skip=frozenset({"radar", "pypi", "royaleimitate"}),
                                              now=NOW_1030 + dt.timedelta(days=1)))
    il = status["il_replay"]
    assert code == 0 and il["changed"] is True and il["previous"] == "a" * 40
    assert any(n.startswith(f"IL_Replay re-published: revision {'b' * 40}")
               for n in status["notes"])
    world.web.json(HF, {"error": "down"}, status=503)
    code, status = gu.run_check(world.options(skip=frozenset({"radar", "pypi", "royaleimitate"})))
    assert code == 0 and status["il_replay"]["error"].startswith("HTTP 503")


# ---------------------------------------------------------------------------
# the parts
# ---------------------------------------------------------------------------

def test_read_engine_archive(gu, rs_engine):
    data = engine_sdist(links=[("data/derived/cards-link.json", "../../../etc/passwd")])
    got = gu.read_engine_archive(data, "royalesim-0.1.43.tar.gz")
    by_version = {t["version"]: t for t in got["tables"]}
    assert set(by_version) == {"cards-15535.1", "cards-160402017-20261006.1",
                               "cards-160402017.1", "cards-160402017-20261019.1",
                               "cards-160635001-20261102.1"}
    assert len(got["tables"]) == 6                                # cards.json too, no link
    old = by_version["cards-15535.1"]
    assert (old["arm"], old["client_build"], old["client"], old["content"],
            old["content_date"]) == ("client15535", None, "15.535.29", "15.535.29", None)
    t = by_version["cards-160402017-20261019.1"]
    assert (t["arm"], t["client_build"], t["client"], t["content"], t["content_date"]) == (
        "client160402017_20261019", 160402020, "16.402.20", "16.402.22", "2026-10-19")
    assert by_version["cards-160402017.1"]["content_date"] is None
    new = by_version["cards-160635001-20261102.1"]
    assert new["forms"] == {"Giant": ["evolution", "hero"], "Mirror": ["evolution"],
                            "Skeletons": ["evolution"]}
    assert new["roster"] == {"cards": 3, "evolutions": 3, "hero_forms": 1, "towers": 2,
                             "units": 2}
    assert new["fnv1a64"] == rs_engine.fnv1a64(dumps(tables_043()[NEW_ARM][1]))
    assert got["upstream_tests"] == 2 and got["ledger"]["default_arm"] == "client15535"


def test_provenance_parsing_on_the_real_0_1_42_wording(gu):
    """The texts are copied from the 0.1.42 sdist's tables (read 2026-10-30)."""
    t20261019 = {"provenance": {
        "source": "Supercell csv_logic/ of the 160402017 tables (client build 160402020): the "
                  "2026-10-19 update folder over the install-time pack, decoded by "
                  "tools/decode_sc_assets.py (each pack's MANIFEST.json carries its per-file "
                  "hashes)",
        "roster_dating": "The build and content are in the files (MANIFEST.json build "
                         "160402020, content_version 16.402.22).",
        "vintage": "160402017 tables, content 16.402.22 (the 2026-10-19 Season 88 mid-season "
                   "balance update)"}}
    assert gu.provenance_versions(t20261019) == {"client_build": 160402020,
                                                 "client": "16.402.20", "content": "16.402.22"}
    t20261006 = {"provenance": {
        "source": "Supercell csv_logic/ of the 160402017 client: the 2026-10-06 update folder "
                  "over the install-time pack",
        "roster_dating": "The build and content are in the files (MANIFEST.json build "
                         "160402017, content_version 16.402.19)."}}
    assert gu.provenance_versions(t20261006)["content"] == "16.402.19"
    assert gu.provenance_versions(t20261006)["client_build"] == 160402017
    t15535 = {"provenance": {
        "source": "Supercell csv_logic/ of the 15.535.29 client, decoded by "
                  "tools/decode_sc_assets.py",
        "roster_dating": "The client version is in the file (MANIFEST.json version 15.535.29); "
                         "the roster is the live 2026 one"},
        "vintage_warning": "LIVE 2026 BUILD (the 15.535.29 client's asset pack)"}
    assert gu.provenance_versions(t15535) == {"client_build": None, "client": "15.535.29",
                                              "content": "15.535.29"}
    assert gu.provenance_versions({}) == {"client_build": None, "client": None, "content": None}
    assert gu.table_dates("cards-160402017-20261019.1") == "2026-10-19"
    assert gu.table_dates("cards-160402017.1") is None
    assert gu.table_dates("cards-15535.1") is None


def test_compare_release_against_nothing(gu):
    summary = gu.read_engine_archive(engine_sdist(), "royalesim-0.1.43.tar.gz")
    got = gu.compare_release(None, summary)
    assert {r["status"] for r in got["tables"]} == {"new"}
    assert got["arms_added"] == OLD_ARMS + [NEW_ARM]
    assert got["default_arm"] == {"old": None, "new": "client15535", "changed": True}


def test_version_ordering(gu):
    order = ["0.1.9", "0.1.10", "0.1.42", "0.1.43.dev1", "0.1.43a1", "0.1.43rc1", "0.1.43",
             "0.1.43.post1", "0.2"]
    shuffled = order[::-1]
    assert sorted(shuffled, key=gu.version_key) == order
    assert gu.version_key("1.0") == gu.version_key("1.0.0")
    assert gu.version_key("v1.2") == gu.version_key("1.2")
    assert gu.version_key("not a version") is None
    assert gu.is_prerelease("0.1.44rc1") and gu.is_prerelease("0.1.43.dev2")
    assert not gu.is_prerelease("0.1.43") and not gu.is_prerelease("0.1.43.post1")


def test_newer_versions_on_the_fixture(gu):
    pypi = load("pypi_royalesim.json")
    assert gu.newer_versions(pypi, "0.1.42") == {"newer": ["0.1.43"], "yanked": [],
                                                 "pre": ["0.1.44rc1"]}
    assert gu.newer_versions(pypi, "0.1.42", pre=True)["newer"] == ["0.1.43", "0.1.44rc1"]
    assert gu.newer_versions(pypi, "0.1.41")["newer"] == ["0.1.42", "0.1.43"]
    yanked = copy.deepcopy(pypi)
    for entry in yanked["releases"]["0.1.43"]:
        entry["yanked"] = True
    assert gu.newer_versions(yanked, "0.1.42")["yanked"] == ["0.1.43"]
    assert gu.newer_versions({"info": {"version": "0.1.43"}}, "0.1.42")["newer"] == ["0.1.43"]
    with pytest.raises(gu.CheckError):
        gu.newer_versions(pypi, "latest")


def test_release_files_without_the_releases_key(gu, monkeypatch):
    web = Web(gu)
    monkeypatch.setattr(gu, "fetch", web)
    web.json("https://pypi.org/pypi/royalesim/0.1.43/json",
             {"info": {"version": "0.1.43"}, "urls": load("pypi_royalesim.json")["urls"]})
    files = gu.release_files({"info": {"version": "0.1.43"}}, "royalesim", "0.1.43")
    assert [f["filename"] for f in files] == ["royalesim-0.1.43.tar.gz"]
    assert gu.pick_sdist(files)["filename"] == "royalesim-0.1.43.tar.gz"


# ---------------------------------------------------------------------------
# the HTTP layer
# ---------------------------------------------------------------------------

class Script:
    """A _transport that answers in turn; an exception in the list is raised."""

    def __init__(self, gu, answers):
        self.gu, self.answers, self.seen = gu, list(answers), []

    def __call__(self, url, headers, timeout):
        self.seen.append((url, dict(headers), timeout))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def http(gu, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(gu, "_sleep", sleeps.append)
    monkeypatch.setattr(gu, "_clock", lambda: 100.0)
    monkeypatch.setattr(gu, "_last_call", {})

    def install(answers):
        script = Script(gu, answers)
        monkeypatch.setattr(gu, "_transport", script)
        return script
    return types.SimpleNamespace(sleeps=sleeps, install=install)


def test_fetch_sends_its_user_agent_and_waits_out_a_429(gu, http):
    script = http.install([gu.Fetched(429, b"", {"retry-after": "3"}),
                           gu.Fetched(200, b"{}", {})])
    got = gu.fetch("https://pypi.org/pypi/royalesim/json")
    assert got.status == 200 and http.sleeps == [3.0, gu.HOST_GAP]
    headers = script.seen[0][1]
    assert headers["User-Agent"] == gu.USER_AGENT and "urllib" not in headers["User-Agent"]
    assert headers["Accept"] == "application/json"


def test_fetch_gives_up_on_a_long_retry_after(gu, http):
    http.install([gu.Fetched(429, b"", {"retry-after": "600"})])
    assert gu.fetch("https://pypi.org/x").status == 429 and http.sleeps == []


def test_fetch_backs_off_then_raises_without_a_response(gu, http):
    http.install([OSError("no route")] * 4)
    with pytest.raises(gu.CheckError, match="no response"):
        gu.fetch("https://pypi.org/x")
    waits = [s for s in http.sleeps if s != gu.HOST_GAP]
    assert waits == [5.0, 10.0, 20.0]


def test_fetch_returns_the_last_503(gu, http):
    script = http.install([gu.Fetched(503, b"", {})] * 4)
    assert gu.fetch("https://huggingface.co/x").status == 503 and len(script.seen) == 4


def test_fetch_spaces_calls_to_one_host_and_refuses_http(gu, http):
    http.install([gu.Fetched(200, b"{}", {})] * 3)
    gu.fetch("https://pypi.org/a")
    gu.fetch("https://huggingface.co/b")
    assert http.sleeps == []
    gu.fetch("https://pypi.org/c")
    assert http.sleeps == [gu.HOST_GAP]
    with pytest.raises(gu.CheckError, match="only https"):
        gu.fetch("http://pypi.org/d")


def test_retry_after_values(gu):
    assert gu.retry_after("2.5") == 2.5
    assert gu.retry_after(None) is None and gu.retry_after("soon") is None
    assert gu.retry_after("Wed, 21 Oct 2099 07:28:00 GMT") > gu.MAX_WAIT


# ---------------------------------------------------------------------------
# CLI and rules
# ---------------------------------------------------------------------------

def test_cli_exit_codes(world, gu, capsys):
    base = ["check", "--offline", "--updates", str(world.updates), "--data", str(world.data),
            "--ledger", str(world.ledger)]
    assert gu.main(base + ["--skip", ",".join(gu.PARTS), "--client", "160635001"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("currency  client 160635001 (16.635.1) vs table 16.402.22")
    assert out.rstrip().splitlines()[-1].startswith("exit 0: OK")
    assert gu.main(base + ["--skip", "pypi,royaleimitate,radar,il_replay", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["exit"] == 0
    # PowerShell 5.1 hands an unquoted pypi,royaleimitate,radar,il_replay over as four
    # arguments; that and two --skip read the same
    for skip in (["--skip", *gu.PARTS], ["--skip", "pypi,royaleimitate", "--skip", "radar",
                                         "il_replay"]):
        assert gu.main(base + [*skip, "--json"]) == 0, skip
        assert all(v.get("skipped") for k, v in json.loads(capsys.readouterr().out).items()
                   if k in ("royalesim", "royaleimitate", "radar", "il_replay"))
    for argv in (["check", "--skip", "bogus"], ["check", "--client", "12"],
                 ["check", "--max-sdists", "-1"], [], ["probe"]):
        with pytest.raises(SystemExit) as exc:
            gu.main(argv)
        assert exc.value.code == 2, argv
    assert world.web.calls == []


def test_check_reads_no_device_and_no_game_memory():
    """Section 4.1: HTTPS calls only, never adb (CLAUDE.md: no memory reading)."""
    for name in ("game_update.py", "game_versions.py"):
        tree = ast.parse((TOOLS / name).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
        assert not {n for n in names if n.split(".")[0] in ("subprocess", "scripts", "emulator",
                                                             "memread", "adbutils")}, name
        assert not any("gamestate" in n for n in names), name


def test_status_files_hold_card_data_only(world, gu):
    with_release(world)
    world.previous_cards()
    gu.run_check(world.options(cards=FIX / "cards_api_2026-11-02.json", now=NOW_1103))
    for path in world.updates.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "Bearer" not in text and "supercell_token" not in text, path
        assert '"tag"' not in text, path
