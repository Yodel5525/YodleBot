"""U03: tools/opponents.py, the opponent store (docs/SEASON_UPDATES.md 4.1, 7.1-7.3, 7.7, 12).

No test touches the network. The client's transport is a FakeTransport that answers from
hand-made fixtures (tests/pipeline/fixtures/opponents/, whose README tells the whole story and
the expected results, plus the U02 board and logs in fixtures/cr_api/). Waits go through a
FakeClock. Every tag and name is invented.

Worked by hand (fixtures/opponents/README.md):
* One pull asks, in order: the clan's members; the logs of Yodle2, the owner, M1, M2 (the roster's
  order, without Yodle2 and the owner a second time) and the friend; then /v1/cards. That is
  7 calls.
* It stores 6 battles: three spars (Yodle2 against the owner, M1 and the friend: 1-2 L, 2-1 W,
  3-0 W, so 2-0-1), two clan battles (the owner's and M1's Path of Legends battles), and Yodle2
  against the stranger (bot_unknown). The 2v2 is dropped. The spars with the owner, M1 and the
  friend each appear in two logs and are stored once.
* The owner's Hog 2.6 reads cannon-ev1, musketeer-hero, skeletons-ev1 under the slot rule: Ice
  Spirit owns an evolution but sits in slot 6. M1's 7 cards sit in slots 1, 3-8, so Bats (index
  1, slot 3) is an evolution and Mini P.E.K.K.A (slot 8) is base.
* The meta run on the U02 files (--top 3): board A, B, C; A's log holds A-B and A-D (Path of
  Legends) and a friendly; B's log holds A-B again and a 2v2; C's log is a 404. So 2 battles,
  2 dropped as not Path of Legends, 1 failed log (exit 1).
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import gzip
import importlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
FIX = Path(__file__).resolve().parent / "fixtures" / "opponents"
API_FIX = Path(__file__).resolve().parent / "fixtures" / "cr_api"
TOKEN = "fixture.token-value_3"
NOW = dt.datetime(2026, 10, 30, 19, 0, tzinfo=dt.timezone.utc)
LATER = NOW + dt.timedelta(hours=24)

CLAN, OWNER, BOT, M1, M2 = "#09QJ0RV2", "#0PL0QY8", "#0Y0GRC2", "#02GJUV9", "#08RPYCQ"
FRIEND, STRANGER, OUT1, OUT2, FORMER = "#0LUV2R8", "#09YRJQP", "#0CJ8P0L", "#0U2CRQ9", "#0QC92LJ"
T_OWNER, T_M1, T_FRIEND = ("20261030T183000.000Z", "20261030T181500.000Z",
                           "20261029T210000.000Z")
HOG = "cannon-ev1,fireball,hog-rider,ice-golem,ice-spirit,musketeer-hero,skeletons-ev1,the-log"
M1_DECK = "arrows,bats-ev1,fireball,goblin-gang,knight-ev1,mini-pekka,zap"
FRIEND_DECK = "earthquake,electro-spirit,fisherman,giant-hero,hunter,knight,mirror-ev1,royal-giant"
IDS = [f"{T_FRIEND}|{FRIEND}|{BOT}", f"20261030T120000.000Z|{STRANGER}|{BOT}",
       f"20261030T150000.000Z|{M1}|{OUT2}", f"20261030T170000.000Z|{OUT1}|{OWNER}",
       f"{T_M1}|{M1}|{BOT}", f"{T_OWNER}|{OWNER}|{BOT}"]
KINDS = ["spar", "bot_unknown", "clan", "clan", "spar", "spar"]


@pytest.fixture
def opp(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.delenv("HASTYCR_CR_API_BASE", raising=False)
    return importlib.import_module("opponents")


def load(name: str, folder: Path = FIX):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def answer(opp, status: int, name: str | None = None, *, folder: Path = FIX,
           body: bytes | None = None):
    data = (folder / name).read_bytes() if name else (body or b"")
    return opp.cr_api.Raw(status, {"content-type": "application/json"} if data else {}, data)


class FakeTransport:
    """Answers by exact URL; a route's answers are used in order and the last one repeats. An
    unknown URL fails the test (an AssertionError is not a transport error the client absorbs)."""

    def __init__(self, routes):
        self.routes = {url: list(answers) for url, answers in routes.items()}
        self.calls: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def __call__(self, url, headers, timeout):
        with self._lock:
            self.calls.append((url, dict(headers)))
            answers = self.routes.get(url)
            assert answers, f"unexpected request {url}"
            return answers.pop(0) if len(answers) > 1 else answers[0]

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def fast_rate(opp):
    clock = FakeClock()
    return opp.cr_api.Rate(1000.0, clock=clock.monotonic, sleep=clock.sleep)


def pull_routes(opp, **override):
    base = opp.cr_api.OFFICIAL_BASE
    routes = {
        "clans/%2309QJ0RV2/members": [answer(opp, 200, "members.json")],
        "players/%230Y0GRC2/battlelog": [answer(opp, 200, "battlelog_bot.json")],
        "players/%230PL0QY8/battlelog": [answer(opp, 200, "battlelog_owner.json")],
        "players/%2302GJUV9/battlelog": [answer(opp, 200, "battlelog_m1.json")],
        "players/%2308RPYCQ/battlelog": [answer(opp, 200, "battlelog_m2.json")],
        "players/%230LUV2R8/battlelog": [answer(opp, 200, "battlelog_friend.json")],
        "cards": [answer(opp, 200, "cards.json", folder=API_FIX)],
        **override}
    return {f"{base}/{path}": answers for path, answers in routes.items()}


def write_config(folder: Path, **changes) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    data = {"schema": 1, "clan": CLAN, "owner": OWNER, "bot": BOT, "friends": [FRIEND],
            "form_rule": "slot", "keep_raw_days": 30, **changes}
    (folder / "config.json").write_text(json.dumps(data), encoding="utf-8")


def run(opp, folder: Path, command: str, *args: str, transport=None, now=NOW) -> int:
    token = folder.parent / "token"
    token.write_text(TOKEN + "\n", encoding="utf-8")
    api = ["--token-file", str(token)] if command in ("init", "pull", "meta", "probe") else []
    return opp.main([command, "--folder", str(folder), *api, *args], transport=transport,
                    rate=fast_rate(opp), now=now)


def stored(folder: Path, name: str = "battles.jsonl") -> list[dict]:
    lines = (folder / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def partners(opp, members=(M1, M2, OWNER, BOT), friends=(FRIEND,)):
    return opp.Partners(BOT, OWNER, frozenset(members), frozenset(friends))


# ---------------------------------------------------------------------------
# forms, slots and decks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slot, bits, by_slot, by_field", [
    (1, 1, 1, 1), (1, 3, 1, None), (1, 2, 0, 2),          # slot 1 takes an evolution only
    (2, 2, 2, 2), (2, 3, 2, None), (2, 1, 0, 1),          # slot 2 a hero only
    (3, 1, 1, 1), (3, 2, 2, 2), (3, 3, None, None),       # slot 3 either; both bits: unknown
    (6, 1, 0, 1), (8, 2, 0, 2),                           # slots 4-8 play base
    (1, None, 0, 0), (1, 0, 0, 0), (5, 4, 0, 0),          # no bits; an unknown bit alone
])
def test_form_rules(opp, slot, bits, by_slot, by_field):
    assert opp.form_of(slot, bits, "slot") == by_slot
    assert opp.form_of(slot, bits, "field") == by_field
    assert opp.form_of(slot, bits, "base") == 0


def test_form_rule_names_and_slots(opp):
    with pytest.raises(ValueError, match="form rule"):
        opp.form_of(1, 1, "guess")
    assert opp.slots_for(8) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert opp.slots_for(7) == [1, 3, 4, 5, 6, 7, 8]           # the slot-2 card is missing
    assert opp.slots_for(3) == [1, 2, 3]
    assert opp.card_token("knight", None) == "knight"          # ambiguous reads as base
    assert opp.split_form("musketeer-hero") == ("musketeer", 2)
    assert opp.split_form("skeletons-evo") == ("skeletons", 1)
    assert opp.split_form("hero") == ("hero", 0)


def test_the_owner_spar_record(opp):
    battle = load("battlelog_bot.json")[0]
    record, why = opp.normalise(battle, role_of=partners(opp).role_of, seen_at="x")
    assert why == "kept" and record["id"] == f"{T_OWNER}|{OWNER}|{BOT}"
    assert record["kind"] == "spar" and record["type"] == "friendly"
    assert record["game_mode"] == {"id": 72000009, "name": "Friendly"}
    assert record["deck_selection"] == "collection" and record["form_rule"] == "slot"
    owner, bot = record["sides"]                                # sorted by tag
    assert (owner["tag"], owner["role"], bot["tag"], bot["role"]) == (OWNER, "owner", BOT, "bot")
    assert owner["deck"] == bot["deck"] == HOG
    assert owner["complete"] and owner["n_cards"] == 8 and owner["ambiguous"] == []
    assert (owner["crowns"], owner["king_hp"], owner["princess_hp"]) == (2, 4824, [1200])
    assert owner["tower"] == "tower-princess" and owner["clan"] == CLAN
    assert owner["cards"][1] == {"key": "musketeer", "slot": 2, "form": 2,
                                 "evolution_level": 3, "max_evolution_level": 3,
                                 "name": "Musketeer", "id": 26000014, "level": 9,
                                 "max_level": 14}
    ice_spirit = owner["cards"][5]
    assert (ice_spirit["slot"], ice_spirit["evolution_level"], ice_spirit["form"]) == (6, 1, 0)
    assert "evolution_level" not in owner["cards"][3]           # Hog Rider: the API gave none


def test_a_seven_card_deck_is_kept_incomplete(opp):
    battle = load("battlelog_bot.json")[1]
    record, _ = opp.normalise(battle, role_of=partners(opp).role_of)
    mate = opp.side_of(record, M1)
    assert (mate["n_cards"], mate["complete"]) == (7, False)
    assert [c["slot"] for c in mate["cards"]] == [1, 3, 4, 5, 6, 7, 8]
    assert mate["deck"] == M1_DECK
    field = opp.refresh(record, rule="field")
    assert opp.side_of(field, M1)["deck"] == \
        "arrows,bats-ev1,fireball,goblin-gang,knight-ev1,mini-pekka-hero,zap"


def test_forms_are_kept_as_given_and_ambiguity_is_marked(opp):
    battle = load("battlelog_bot.json")[3]
    record, _ = opp.normalise(battle, role_of=partners(opp).role_of)
    friend = opp.side_of(record, FRIEND)
    assert friend["deck"] == FRIEND_DECK                       # Mirror's evolution is kept
    assert friend["ambiguous"] == ["knight"] and friend["cards"][2]["form"] is None
    assert (friend["tower"], friend["clan"], friend["king_hp"], friend["princess_hp"]) == \
        ("dagger-duchess", None, 0, [])
    base = opp.refresh(record, rule="base")
    assert opp.side_of(base, FRIEND)["deck"] == \
        "earthquake,electro-spirit,fisherman,giant,hunter,knight,mirror,royal-giant"
    assert base["form_rule"] == "base" and opp.side_of(base, FRIEND)["ambiguous"] == []


def test_two_v_two_and_odd_battles_are_dropped(opp):
    role_of = partners(opp).role_of
    two = load("battlelog_owner.json")[2]
    assert opp.normalise(two, role_of=role_of) == (None, "2v2")
    good = load("battlelog_bot.json")[0]
    nine = copy.deepcopy(good)
    nine["team"][0]["cards"].append(dict(nine["team"][0]["cards"][0], name="Zap"))
    assert opp.normalise(nine, role_of=role_of) == (None, "not_one_deck")
    empty = copy.deepcopy(good)
    empty["opponent"][0]["cards"] = []
    assert opp.normalise(empty, role_of=role_of) == (None, "no_deck")
    late = dict(good, battleTime="yesterday")
    assert opp.normalise(late, role_of=role_of) == (None, "no_time")
    nobody = copy.deepcopy(good)
    nobody["team"][0]["tag"] = ""
    assert opp.normalise(nobody, role_of=role_of) == (None, "no_tag")
    assert opp.normalise(["not", "a", "battle"], role_of=role_of) == (None, "not_a_battle")


def test_spars_are_found_by_tag_not_by_type(opp):
    who = partners(opp, members=(M1, M2, OWNER, BOT, FORMER))
    renamed = dict(load("battlelog_bot.json")[1], type="someFutureFriendlyType")
    assert opp.normalise(renamed, role_of=who.role_of)[0]["kind"] == "spar"
    stranger = load("battlelog_bot.json")[2]                   # type "friendly", but a stranger
    assert opp.normalise(stranger, role_of=who.role_of)[0]["kind"] == "bot_unknown"
    former = copy.deepcopy(stranger)
    former["opponent"][0]["tag"] = FORMER                      # left the clan; still a partner
    assert opp.normalise(former, role_of=who.role_of)[0]["kind"] == "spar"
    no_m1 = partners(opp, members=(M2, OWNER, BOT))
    assert opp.normalise(renamed, role_of=no_m1.role_of)[0]["kind"] == "bot_unknown"
    owner_pol = load("battlelog_owner.json")[1]
    assert opp.normalise(owner_pol, role_of=who.role_of)[0]["kind"] == "clan"
    assert opp.kind_of([{"role": "top"}, {"role": ""}]) == "top"
    assert opp.kind_of([{"role": ""}, {"role": ""}]) == "other"
    assert who.role_of(BOT) == "bot" and who.role_of(OWNER) == "owner"   # before "member"


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

def test_pull_stores_each_battle_once(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    write_config(folder)
    fake = FakeTransport(pull_routes(opp))
    assert run(opp, folder, "pull", transport=fake) == opp.EXIT_OK
    base = opp.cr_api.OFFICIAL_BASE
    assert fake.urls == [f"{base}/clans/%2309QJ0RV2/members",
                         f"{base}/players/%230Y0GRC2/battlelog",
                         f"{base}/players/%230PL0QY8/battlelog",
                         f"{base}/players/%2302GJUV9/battlelog",
                         f"{base}/players/%2308RPYCQ/battlelog",
                         f"{base}/players/%230LUV2R8/battlelog", f"{base}/cards"]
    assert all(h["User-Agent"] == opp.cr_api.USER_AGENT for _, h in fake.calls)
    records = stored(folder)
    assert [r["id"] for r in records] == IDS and [r["kind"] for r in records] == KINDS
    spar = records[-1]
    assert opp.side_of(spar, OWNER)["deck"] == HOG and spar["seen_at"] == "2026-10-30T19:00:00Z"
    assert opp.side_of(records[4], M1)["deck"] == M1_DECK
    assert opp.side_of(records[0], FRIEND)["deck"] == FRIEND_DECK
    assert [opp.result_of(r, BOT) for r in records if r["kind"] == "spar"] == ["W", "W", "L"]

    out = capsys.readouterr().out
    assert ("pull      5 of 5 logs, 4 members; 6 new battles (bot_unknown 1, clan 2, spar 3); "
            "dropped: 2v2 1") in out
    assert "spars     3 new: Yodle2 2-0-1 (W-D-L)" in out
    assert f"unknown   Yodle2 played {STRANGER}" in out
    assert "cards     8 cards, 1 support items, unchanged" in out
    assert TOKEN not in out

    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["pulled_at"] == state["last_attempt_at"] == "2026-10-30T19:00:00Z"
    assert state["players"][BOT] == {"pulled_at": "2026-10-30T19:00:00Z", "battles": 4,
                                     "newest": "2026-10-30T18:30:00Z",
                                     "oldest": "2026-10-29T21:00:00Z", "role": "bot"}
    assert state["players"][M2]["battles"] == 0 and state["players"][M2]["newest"] is None
    assert state["pull"]["ok"] and state["pull"]["statuses"] == {"200": 7}
    assert state["pull"]["gaps"] == [] and state["pull"]["recomputed"] == 0

    roster = json.loads((folder / "roster.json").read_text(encoding="utf-8"))
    assert list(roster["members"]) == [M1, OWNER, M2, BOT]
    assert roster["members"][M1]["first_seen"] == "2026-10-30T19:00:00Z"
    assert "left" not in roster["members"][M1]

    # the file contract tools/game_update.py (U04) reads
    cards = json.loads((folder / "cards_api.json").read_text(encoding="utf-8"))
    assert cards["fetched_at"] == "2026-10-30T19:00:00Z" and len(cards["data"]["items"]) == 8
    assert not (folder / "cards_api.prev.json").exists()
    raw = sorted((folder / "raw" / "2026-10-30").iterdir())
    assert len(raw) == 7 and all(p.name.startswith("190000_") for p in raw)
    first = json.loads(gzip.decompress((folder / "raw" / "2026-10-30" /
                                        "190000_battlelog_0Y0GRC2.json.gz").read_bytes()))
    assert first["status"] == 200 and len(first["data"]) == 4
    assert first["path"] == "/players/%23<tag>/battlelog"
    assert (folder / ".gitignore").read_text(encoding="utf-8").splitlines()[-1] == "*"
    assert not (folder / "pull.lock").exists()


def test_a_second_pull_adds_nothing_and_keeps_the_old_cards(opp, tmp_path):
    folder = tmp_path / "opponents"
    write_config(folder)
    changed = load("cards.json", API_FIX)
    changed["items"] = changed["items"][:-1]
    fake = FakeTransport(pull_routes(opp, cards=[
        answer(opp, 200, "cards.json", folder=API_FIX),
        answer(opp, 200, body=json.dumps(changed).encode())]))
    assert run(opp, folder, "pull", transport=fake) == opp.EXIT_OK
    first = json.loads((folder / "cards_api.json").read_text(encoding="utf-8"))
    assert run(opp, folder, "pull", transport=fake, now=LATER) == opp.EXIT_OK
    assert [r["id"] for r in stored(folder)] == IDS
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["pull"]["added"] == 0 and state["pull"]["recomputed"] == 0
    assert state["cards"]["changed"] is True
    assert state["cards"]["changed_at"] == "2026-10-31T19:00:00Z"
    assert state["cards"]["previous_digest"] == first["digest"]
    prev = json.loads((folder / "cards_api.prev.json").read_text(encoding="utf-8"))
    assert prev == first
    assert len(json.loads((folder / "cards_api.json").read_text(encoding="utf-8"))
               ["data"]["items"]) == 7


def test_a_log_that_starts_after_the_last_pull_is_a_gap(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    write_config(folder)
    (folder / "state.json").write_text(json.dumps(
        {"players": {BOT: {"newest": "2026-10-20T00:00:00Z"}}}), encoding="utf-8")
    assert run(opp, folder, "pull", transport=FakeTransport(pull_routes(opp))) == opp.EXIT_OK
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["pull"]["gaps"] == [BOT]
    assert "gaps      1 logs start after the newest battle stored last time" in \
        capsys.readouterr().out


def test_a_config_change_reaches_stored_battles(opp, tmp_path):
    """Dropping the friend and switching to the base rule re-derive all 6 stored battles, and the
    friend's log is no longer asked for."""
    folder = tmp_path / "opponents"
    write_config(folder)
    fake = FakeTransport(pull_routes(opp))
    assert run(opp, folder, "pull", transport=fake) == opp.EXIT_OK
    write_config(folder, friends=[], form_rule="base")
    assert run(opp, folder, "pull", transport=fake, now=LATER) == opp.EXIT_OK
    assert sum("%230LUV2R8" in url for url in fake.urls) == 1
    records = stored(folder)
    assert [r["kind"] for r in records] == ["bot_unknown", *KINDS[1:]]
    assert opp.side_of(records[0], FRIEND)["role"] == ""
    assert opp.side_of(records[-1], OWNER)["deck"] == \
        "cannon,fireball,hog-rider,ice-golem,ice-spirit,musketeer,skeletons,the-log"
    assert {r["form_rule"] for r in records} == {"base"}
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["pull"]["recomputed"] == 6


def test_a_member_who_left_is_still_a_sparring_partner(opp, tmp_path):
    folder = tmp_path / "opponents"
    write_config(folder)
    members = load("members.json")
    members["items"] = [m for m in members["items"] if m["tag"] != M1]
    fake = FakeTransport(pull_routes(opp, **{"clans/%2309QJ0RV2/members": [
        answer(opp, 200, "members.json"),
        answer(opp, 200, body=json.dumps(members).encode())]}))
    assert run(opp, folder, "pull", transport=fake) == opp.EXIT_OK
    assert run(opp, folder, "pull", transport=fake, now=LATER) == opp.EXIT_OK
    roster = json.loads((folder / "roster.json").read_text(encoding="utf-8"))
    assert roster["members"][M1]["left"] == "2026-10-31T19:00:00Z"
    assert sum("%2302GJUV9" in url for url in fake.urls) == 1   # not pulled after leaving
    assert [r["kind"] for r in stored(folder)] == KINDS        # the M1 spar stays a spar


def test_pull_stops_on_403(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    write_config(folder)
    fake = FakeTransport(pull_routes(opp, **{"clans/%2309QJ0RV2/members": [
        answer(opp, 403, "error_403_invalid_ip.json", folder=API_FIX)]}))
    assert run(opp, folder, "pull", transport=fake) == opp.EXIT_API
    assert len(fake.calls) == 1
    out = capsys.readouterr().out
    assert "FAILED    clan members: 403" in out and "developer.clashroyale.com" in out
    assert not (folder / "battles.jsonl").exists()
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert "pulled_at" not in state and state["last_attempt_at"] == "2026-10-30T19:00:00Z"
    assert state["pull"]["fatal"] == 403 and not state["pull"]["ok"]


def test_pull_stops_on_a_429_the_back_off_cannot_clear(opp, tmp_path, capsys):
    """Yodle2's log is throttled with a Retry-After of 600 s (too long to wait): the pull stops
    there instead of asking for the next log, and the shared rate holds for the 600 s."""
    folder = tmp_path / "opponents"
    write_config(folder)
    throttled = opp.cr_api.Raw(429, {"content-type": "application/json", "retry-after": "600"},
                               (API_FIX / "error_429.json").read_bytes())
    fake = FakeTransport(pull_routes(opp, **{"players/%230Y0GRC2/battlelog": [throttled]}))
    assert run(opp, folder, "pull", transport=fake) == opp.EXIT_API
    assert len(fake.calls) == 2                                # the members, then Yodle2's log
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["pull"]["fatal"] == 429 and "pulled_at" not in state
    assert "FAILED    battle log of a bot: 429: throttled" in capsys.readouterr().out
    assert 429 in opp.FATAL and opp.FATAL == opp.cr_api.STOP


def test_pull_goes_on_past_a_failed_log(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    write_config(folder)
    fake = FakeTransport(pull_routes(opp, **{"players/%2308RPYCQ/battlelog": [
        answer(opp, 404, "error_404.json", folder=API_FIX)]}))
    assert run(opp, folder, "pull", transport=fake) == opp.EXIT_API
    assert len(fake.calls) == 7 and [r["id"] for r in stored(folder)] == IDS
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["pulled_at"] == "2026-10-30T19:00:00Z"        # a partial pull still counts
    assert state["pull"]["failed"][0]["what"] == "battle log of a member"
    assert state["pull"]["failed"][0]["status"] == 404
    assert "FAILED    battle log of a member: 404" in capsys.readouterr().out


def test_raw_responses_are_pruned_or_not_kept(opp, tmp_path):
    folder = tmp_path / "opponents"
    write_config(folder)
    for day in ("2026-09-01", "2026-10-15"):
        (folder / "raw" / day).mkdir(parents=True)
        (folder / "raw" / day / "x.json.gz").write_bytes(b"")
    assert run(opp, folder, "pull", transport=FakeTransport(pull_routes(opp))) == opp.EXIT_OK
    assert sorted(p.name for p in (folder / "raw").iterdir()) == ["2026-10-15", "2026-10-30"]
    other = tmp_path / "none"
    write_config(other, keep_raw_days=0)
    assert run(opp, other, "pull", transport=FakeTransport(pull_routes(opp))) == opp.EXIT_OK
    assert not (other / "raw").exists() and len(stored(other)) == 6


def test_a_held_lock_refuses_and_a_stale_one_is_taken(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    write_config(folder)
    (folder / "pull.lock").write_text("{}", encoding="utf-8")
    fake = FakeTransport(pull_routes(opp))
    assert run(opp, folder, "pull", transport=fake) == opp.EXIT_API
    assert fake.calls == [] and "another pull holds" in capsys.readouterr().err
    old = time.time() - opp.LOCK_STALE_S - 60
    os.utime(folder / "pull.lock", (old, old))
    assert run(opp, folder, "pull", transport=fake) == opp.EXIT_OK
    assert not (folder / "pull.lock").exists()


def test_the_lock_beats_and_leaves_a_lock_it_lost(opp, tmp_path):
    path = tmp_path / "pull.lock"
    first = opp.Lock(path)
    with first:
        old = time.time() - 600
        os.utime(path, (old, old))
        assert first.touch() is True and time.time() - path.stat().st_mtime < 60   # heartbeat
        # a run that made no progress for LOCK_STALE_S is taken over by a second run ...
        old = time.time() - opp.LOCK_STALE_S - 60
        os.utime(path, (old, old))
        second = opp.Lock(path)
        second.__enter__()
        assert second.owned() and not first.owned()
        assert first.touch() is False                      # ... which the first one notices
    assert path.exists() and second.owned()                # and does not remove on its way out
    second.__exit__(None, None, None)
    assert not path.exists()


def test_settings_errors_exit_2(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    assert run(opp, folder, "pull", transport=FakeTransport({})) == opp.EXIT_SETTING
    assert "no config at" in capsys.readouterr().err
    write_config(folder)
    assert opp.main(["pull", "--folder", str(folder), "--token-file", str(tmp_path / "nope")],
                    transport=FakeTransport({}), rate=fast_rate(opp)) == opp.EXIT_SETTING
    assert "no token at" in capsys.readouterr().err
    assert run(opp, folder, "pull", "--base", "http://example.invalid/v1",
               transport=FakeTransport({})) == opp.EXIT_SETTING
    assert run(opp, folder, "pull", "--rate", "50",
               transport=FakeTransport({})) == opp.EXIT_SETTING


def test_status(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    write_config(folder)
    assert run(opp, folder, "pull", transport=FakeTransport(pull_routes(opp))) == opp.EXIT_OK
    capsys.readouterr()
    assert run(opp, folder, "status", "--json") == opp.EXIT_OK
    got = json.loads(capsys.readouterr().out)
    assert got["battles"] == 6 and got["by_kind"] == {"bot_unknown": 1, "clan": 2, "spar": 3}
    assert got["spars"] == {"W": 2, "D": 0, "L": 1} and got["incomplete_decks"] == 2
    assert got["pulled_at"] == "2026-10-30T19:00:00Z" and got["meta_battles"] == 0
    assert run(opp, folder, "status") == opp.EXIT_OK
    assert "Yodle2 2-0-1 (W-D-L)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# config, init and privacy
# ---------------------------------------------------------------------------

def test_the_template_has_placeholders_only(opp):
    data = json.loads(opp.EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(opp.ConfigError, match="placeholder"):
        opp.parse_config(data)
    tags = [v for k, v in data.items() if k in ("clan", "owner", "bot")] + data["friends"]
    assert len(tags) == 3 and all(opp.clean_tag(t) == "" for t in tags)
    text = opp.EXAMPLE_CONFIG.read_text(encoding="utf-8")
    assert not re.search(r"#[0289PYLQGRJCUV]{4,}\b", text)
    assert not {"clan", "owner", "bot"} & set(opp.template()) and opp.template()["friends"] == []


def test_config_rules(opp):
    good = {"clan": CLAN, "owner": OWNER, "bot": BOT, "friends": [FRIEND, FRIEND.lower()]}
    cfg = opp.parse_config(good)
    assert (cfg.clan, cfg.friends, cfg.rate, cfg.form_rule) == (CLAN, (FRIEND,), 2.0, "slot")
    assert cfg.to_json()["friends"] == [FRIEND] and cfg.warnings == ()
    odd = opp.parse_config({**good, "colour": "red"})
    assert any("unknown setting 'colour'" in w for w in odd.warnings)
    for bad, match in [({"clan": "#ABC"}, "never use 'AB'"), ({"friends": ["#2PYQX"]}, "'X'"),
                       ({"clan": "#CLANTAG"}, "placeholder"),
                       ({"bot": "#YODLE2TAG"}, "placeholder"),
                       ({"owner": "#MAINTAG"}, "placeholder"),
                       ({"clan": "#<clan tag>"}, "placeholder"),
                       ({"owner": "#..."}, "placeholder"),
                       ({"owner": BOT}, "same tag"), ({"rate": 50}, "rate"),
                       ({"form_rule": "guess"}, "form_rule"), ({"bot": ""}, "not set"),
                       ({"keep_raw_days": -1}, "keep_raw_days"), ({"friends": "x"}, "list"),
                       ({"base": "http://example.invalid/v1"}, "https"),
                       ({"transport": "carrier-pigeon"}, "transport"),
                       ({"clan": "#12!"}, "not a tag")]:
        with pytest.raises(opp.ConfigError, match=match):
            opp.parse_config({**good, **bad})


def test_init_writes_and_changes_the_config(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    assert run(opp, folder, "init", "--clan", "#09qj0rv2") == opp.EXIT_SETTING
    assert "owner is not set" in capsys.readouterr().err and not folder.exists()
    assert run(opp, folder, "init", "--clan", "#09qj0rv2", "--owner", "0PL0QY8", "--bot",
               "#0Y0GRC2", "--friend", "#0luv2r8") == opp.EXIT_OK
    data = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert (data["clan"], data["owner"], data["bot"], data["friends"]) == \
        (CLAN, OWNER, BOT, [FRIEND])
    assert data["form_rule"] == "slot" and data["rate"] == 2.0 and data["schema"] == 1
    assert (folder / ".gitignore").exists()
    assert run(opp, folder, "init", "--drop-friend", FRIEND, "--friend", STRANGER) == opp.EXIT_OK
    data = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert data["friends"] == [STRANGER] and data["clan"] == CLAN
    # the API settings and the form rule are saved, so every later pull uses them
    capsys.readouterr()
    assert run(opp, folder, "init", "--base", "proxy", "--transport", "curl", "--rate", "1.5",
               "--form-rule", "field") == opp.EXIT_OK
    data = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert (data["base"], data["transport"], data["rate"], data["form_rule"]) == \
        ("proxy", "curl", 1.5, "field")
    assert "base proxy, transport curl, rate 1.5/s, form rule field" in capsys.readouterr().out
    cfg = opp.Store(folder).load_config()
    args = argparse.Namespace(token_file=tmp_path / "token")       # written by run()
    assert opp.build_client(cfg, args, opp.Store(folder),
                            transport=FakeTransport({})).base == opp.cr_api.PROXY_BASE
    assert run(opp, folder, "init", "--rate", "9") == opp.EXIT_SETTING
    assert json.loads((folder / "config.json").read_text(encoding="utf-8"))["rate"] == 1.5


def test_init_refuses_placeholders_and_writes_nothing(opp, tmp_path, capsys):
    """The instructions' own words pasted as they are must not become a config."""
    folder = tmp_path / "opponents"
    assert run(opp, folder, "init", "--clan", "#CLANTAG", "--owner", "#MAINTAG", "--bot",
               "#YODLE2TAG", "--check", transport=FakeTransport({})) == opp.EXIT_SETTING
    assert "placeholder" in capsys.readouterr().err and not (folder / "config.json").exists()
    assert run(opp, folder, "init", "--clan", CLAN, "--owner", OWNER, "--bot", BOT,
               "--friend", "#2PYQX") == opp.EXIT_SETTING
    assert "real tags never use 'X'" in capsys.readouterr().err
    assert not (folder / "config.json").exists()


def test_init_check_looks_the_tags_up_and_stops_on_403(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    write_config(folder)
    base = opp.cr_api.OFFICIAL_BASE
    fake = FakeTransport({
        f"{base}/clans/%2309QJ0RV2": [answer(opp, 200, body=b'{"tag": "#09QJ0RV2", '
                                                             b'"name": "Fixture Clan \\u2694"}')],
        f"{base}/players/%230PL0QY8": [answer(opp, 200, "player_owner.json")],
        f"{base}/players/%230Y0GRC2": [answer(opp, 403, "error_403.json", folder=API_FIX)]})
    assert run(opp, folder, "init", "--check", transport=fake) == opp.EXIT_API
    assert len(fake.calls) == 3                                # the friend is not asked
    out = capsys.readouterr().out
    # ascii(): a name with a symbol prints as an escape, never a UnicodeEncodeError on cp1252
    assert "check     clan: 'Fixture Clan \\u2694'" in out
    assert "check     owner: 'Fixture Owner'" in out
    assert "check     bot: 403" in out


def test_the_store_is_ignored_by_git(opp, tmp_path):
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/opponents/" in rules
    assert not any(r.strip() in ("data/*", "data/", "data/opponents*", "*.json") for r in rules)
    store = opp.Store(tmp_path / "s")
    store.ensure()
    assert (tmp_path / "s" / ".gitignore").read_text(encoding="utf-8") == opp.IGNORE_TEXT
    assert "*" in opp.IGNORE_TEXT.splitlines()


def test_opponents_imports_no_memory_reader(opp):
    source = (TOOLS / "opponents.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names - set(sys.stdlib_module_names) - {"__future__"} == \
        {"cr_api", "hastycr", "rs_cards"}
    assert "gamestate" not in source and "memread" not in source


def test_no_flag_or_command_names_a_matchmade_mode(opp):
    parser = opp._parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    words = set(sub.choices)
    for command in sub.choices.values():
        for action in command._actions:
            words |= set(action.option_strings)
    assert {"init", "pull", "meta", "add", "probe", "status", "--if-due"} <= words
    assert not [w for w in words if re.search(r"ladder|ranked|queue", w, re.I)]


def test_top_types_are_ladder_pulls(opp):
    ladder = importlib.import_module("ladder_pull")
    assert opp.TOP_TYPES == ladder.RANKED_TYPES


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

def test_canonical_deck(opp):
    deck, problems = opp.canonical_deck("Hog Rider, Musketeer-hero, Cannon-ev1, Skeletons-evo, "
                                        "Ice Golem, Ice Spirit, Fireball, The Log")
    assert (deck, problems) == (HOG, [])
    assert opp.canonical_deck(HOG.replace(",the-log", ""))[1] == ["7 cards; a deck has 8"]
    assert "twice: fireball" in opp.canonical_deck(HOG.replace("the-log", "fireball"))[1]
    four = "cannon-ev1,fireball-ev1,hog-rider,ice-golem,ice-spirit,musketeer-hero,skeletons-ev1,zap"
    assert "three form slots" in opp.canonical_deck(four)[1][0]


def test_add_list_remove(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    assert run(opp, folder, "add", "--deck", "Hog Rider, Musketeer-hero, Cannon-ev1, "
               "Skeletons-evo, Ice Golem, Ice Spirit, Fireball, The Log", "--label", "hog",
               "--partner", "#02gjuv9") == opp.EXIT_OK
    ident = opp.deck_id(HOG)
    assert f"added     {ident}  {HOG}" in capsys.readouterr().out
    assert run(opp, folder, "add", "--deck", HOG, "--weight", "2") == opp.EXIT_OK
    assert "updated" in capsys.readouterr().out
    manual = json.loads((folder / "manual.json").read_text(encoding="utf-8"))
    assert len(manual["decks"]) == 1
    entry = manual["decks"][0]
    assert (entry["deck"], entry["partner"], entry["weight"], entry["label"]) == \
        (HOG, M1, 2.0, "hog")
    assert run(opp, folder, "add", "--list") == opp.EXIT_OK
    assert ident in capsys.readouterr().out
    assert run(opp, folder, "add", "--remove", ident) == opp.EXIT_OK
    assert json.loads((folder / "manual.json").read_text(encoding="utf-8"))["decks"] == []
    assert run(opp, folder, "add", "--remove", ident) == opp.EXIT_SETTING
    assert run(opp, folder, "add", "--deck", "hog-rider,zap") == opp.EXIT_SETTING
    assert "2 cards; a deck has 8" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def owner_sides(log):
    """The owner's side of every 1v1 battle of a copy of the owner's log."""
    for battle in log:
        if len(battle["team"]) == 1:
            yield battle["team"][0]


def test_probe_on_the_fixtures_reads_ownership(opp):
    report = opp.probe_report(OWNER, load("player_owner.json"), load("battlelog_owner.json"),
                              bot=BOT, bot_log=load("battlelog_bot.json"))
    assert (report["forms"], report["mirror"], report["hidden"], report["other"]) == \
        ("ownership", 4, 0, 0)
    assert (report["with_field"], report["cards"]) == (8, 16)
    assert {(e["card"], e["slot"]) for e in report["evidence"]} == {("Musketeer", 2),
                                                                    ("Ice Spirit", 6)}
    assert report["order"] == "same" and report["hero_slot"] == "present"
    assert (report["sides_7"], report["sides"], report["current_deck"]) == (1, 10, 8)
    assert (report["log_length"], report["bot_log_length"], report["spars"]) == (3, 4, 1)
    assert report["friendly_levels"] == "11" and report["levels_shown"] == {"11": 8}
    assert report["opponent_view"] == "same"
    assert report["towers"] == {"Tower Princess": 9, "Dagger Duchess": 1}
    friendly = ("type 'friendly', deckSelection 'collection', gameMode 'Friendly' (72000009)")
    assert report["types"][friendly] == 3 and report["spar_types"] == {friendly: 1}
    assert [r["slot_rule"] for r in report["latest"]] == [1, 2, 1, 0, 0, 0, 0, 0]
    assert [r["field_rule"] for r in report["latest"]] == [1, None, 1, 0, 0, 1, 0, 0]
    assert report["latest_battle"] == {"battle_time": T_OWNER, "type": "friendly",
                                       "against_bot": True, "cards": 8}
    assert report["recommend"] == "slot" and "owns" in report["why"]
    for key in ("battlelog_forms", "hero_slot_missing", "battle_types", "log_length",
                "friendly_levels"):
        assert key in opp.cr_api.UNCONFIRMED and report["answers"][key]
    assert "evolutionLevel" in report["keys"]["card"]
    assert "supportCards" in report["keys"]["side"]


def test_probe_reads_played_forms_and_card_order(opp):
    played = load("battlelog_owner.json")
    for side in owner_sides(played):
        side["cards"][1]["evolutionLevel"] = 2                 # the hero played, not bits 3
        side["cards"][5].pop("evolutionLevel")                 # Ice Spirit shows no form
        side["cards"][5].pop("maxEvolutionLevel")
    report = opp.probe_report(OWNER, load("player_owner.json"), played, bot=BOT)
    assert (report["forms"], report["mirror"], report["hidden"]) == ("played", 0, 4)
    assert report["recommend"] == "slot" and report["opponent_view"] == "no_spar"
    for side in owner_sides(played):
        side["cards"][3], side["cards"][4] = side["cards"][4], side["cards"][3]
    report = opp.probe_report(OWNER, load("player_owner.json"), played, bot=BOT)
    assert report["order"] == "differs" and report["recommend"] == "field"


def test_probe_reads_absent_and_undecided(opp):
    bare = load("battlelog_owner.json")
    for side in owner_sides(bare):
        for card in side["cards"]:
            card.pop("evolutionLevel", None)
    report = opp.probe_report(OWNER, load("player_owner.json"), bare)
    assert (report["forms"], report["with_field"], report["recommend"]) == ("absent", 0, "base")
    nothing_owned = dict(load("player_owner.json"), cards=[])
    report = opp.probe_report(OWNER, nothing_owned, load("battlelog_owner.json"), rule="field")
    assert report["forms"] == "undecided" and report["recommend"] == "field"


def test_probe_finds_the_missing_hero_card(opp):
    seven = load("battlelog_owner.json")
    del seven[0]["team"][0]["cards"][1]                        # Musketeer, slot 2
    report = opp.probe_report(OWNER, load("player_owner.json"), seven, bot=BOT,
                              bot_log=load("battlelog_bot.json"))
    assert report["hero_slot"] == "missing"
    assert report["missing"] == [{"battle_time": T_OWNER, "card": "Musketeer",
                                  "current_deck_slot": 2}]
    assert "Musketeer (currentDeck slot 2) is the card left out" in \
        report["answers"]["hero_slot_missing"]
    assert report["opponent_view"] == "differs"


def test_probe_cli(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    write_config(folder)
    base = opp.cr_api.OFFICIAL_BASE
    routes = {f"{base}/players/%230PL0QY8": [answer(opp, 200, "player_owner.json")],
              f"{base}/players/%230PL0QY8/battlelog": [answer(opp, 200, "battlelog_owner.json")],
              f"{base}/players/%230Y0GRC2/battlelog": [answer(opp, 200, "battlelog_bot.json")]}
    assert run(opp, folder, "probe", transport=FakeTransport(routes)) == opp.EXIT_OK
    out = capsys.readouterr().out
    assert "verdict   form_rule slot: the log shows what the player owns; card order same" in out
    assert "forms     ownership: 4 cards show a form they own" in out
    saved = json.loads((folder / "probe.json").read_text(encoding="utf-8"))
    assert saved["player"] == OWNER and saved["forms"] == "ownership"
    assert len(list((folder / "raw" / "2026-10-30").iterdir())) == 3
    routes[f"{base}/players/%230PL0QY8"] = [answer(opp, 404, "error_404.json", folder=API_FIX)]
    assert run(opp, folder, "probe", transport=FakeTransport(routes)) == opp.EXIT_API


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------

def meta_routes(opp):
    base = opp.cr_api.OFFICIAL_BASE
    board = f"{base}/locations/global/pathoflegend/players"
    return {
        f"{board}?limit=3": [answer(opp, 200, "pol_page1.json", folder=API_FIX)],
        f"{board}?limit=1&after=eyJwb3MiOjJ9": [answer(opp, 200, "pol_page2.json",
                                                       folder=API_FIX)],
        f"{base}/players/%2302PYQ0/battlelog": [answer(opp, 200, "battlelog_a.json",
                                                      folder=API_FIX)],
        f"{base}/players/%2308LGRJ/battlelog": [answer(opp, 200, "battlelog_b.json",
                                                      folder=API_FIX)],
        f"{base}/players/%2309CUV2/battlelog": [answer(opp, 404, "error_404.json",
                                                      folder=API_FIX)]}


def test_meta_keeps_path_of_legends_battles_for_8_weeks(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    write_config(folder)
    old = {"v": 1, "id": "20260801T000000.000Z|#02PYQ0|#08LGRJ", "kind": "top",
           "battle_time": "20260801T000000.000Z", "sides": []}
    (folder / "meta_battles.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    assert run(opp, folder, "meta", "--top", "3", "--workers", "1",
               transport=FakeTransport(meta_routes(opp))) == opp.EXIT_API   # C's log: 404
    records = stored(folder, "meta_battles.jsonl")
    assert [r["id"] for r in records] == ["20261029T194000.000Z|#02PYQ0|#0P0LYJ",
                                          "20261029T201500.000Z|#02PYQ0|#08LGRJ"]
    assert {r["kind"] for r in records} == {"top"}
    a = opp.side_of(records[1], "#02PYQ0")
    assert (a["role"], a["rating"], a["rank"]) == ("top", 3600, 1)
    assert a["deck"] == \
        "cannon,fireball,hog-rider,ice-golem,ice-spirit,musketeer-hero,skeletons,the-log"
    d = opp.side_of(records[0], "#0P0LYJ")
    assert (d["role"], d["rating"], d["rank"]) == ("", None, None)
    assert "name" not in a and "level" not in a["cards"][0]     # slim: no names or levels
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["meta_at"] == "2026-10-30T19:00:00Z"
    assert state["meta"]["dropped"] == {"not_path_of_legends": 2}
    assert (state["meta"]["added"], state["meta"]["held"], state["meta"]["failed_count"]) == \
        (2, 2, 1)
    out = capsys.readouterr().out
    assert ("meta      board 3 players (locations/global/pathoflegend/players); 2 logs, "
            "1 failed; 2 new battles; 2 held (8 weeks)") in out
    assert "FAILED    a top player's battle log: 404" in out


def test_meta_stops_after_failed_logs_in_a_row(opp, tmp_path, capsys):
    """With the limit at 1 and one worker, A's 404 stops the run: B and C are never asked."""
    folder = tmp_path / "opponents"
    write_config(folder)
    routes = meta_routes(opp)
    base = opp.cr_api.OFFICIAL_BASE
    routes[f"{base}/players/%2302PYQ0/battlelog"] = [answer(opp, 404, "error_404.json",
                                                           folder=API_FIX)]
    fake = FakeTransport(routes)
    client = opp.cr_api.Client(TOKEN, transport=fake, rate=fast_rate(opp), env={})
    cfg = opp.Store(folder).load_config()
    beats = []
    result = opp.meta(cfg, client, opp.Store(folder), top=3, workers=1, now=NOW,
                      heartbeat=lambda: beats.append(1) or True, max_fails_in_a_row=1)
    assert fake.urls[-1] == f"{base}/players/%2302PYQ0/battlelog" and len(fake.calls) == 3
    assert result["stopped"].startswith("1 failed logs in a row") and not result["ok"]
    assert beats == [1]
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert "meta_at" not in state and state["meta"]["stopped"] == result["stopped"]
    # a lost lock stops it too
    fake = FakeTransport(meta_routes(opp))
    client = opp.cr_api.Client(TOKEN, transport=fake, rate=fast_rate(opp), env={})
    result = opp.meta(cfg, client, opp.Store(folder), top=3, workers=1, now=NOW,
                      heartbeat=lambda: False)
    assert result["stopped"] == "another run took the lock over; stopped" and not result["ok"]


def test_meta_falls_back_to_the_last_seasons(opp):
    base = opp.cr_api.OFFICIAL_BASE
    one = b'{"items": [{"tag": "#02PYQ0", "eloRating": 3600, "rank": 1}], "paging": {}}'
    fake = FakeTransport({
        f"{base}/locations/global/pathoflegend/players?limit=3": [
            answer(opp, 404, "error_404.json", folder=API_FIX)],
        f"{base}/locations/global/pathoflegend/2026-10/rankings/players?limit=3": [
            answer(opp, 404, "error_404.json", folder=API_FIX)],
        f"{base}/locations/global/pathoflegend/2026-09/rankings/players?limit=3": [
            answer(opp, 200, body=one)]})
    client = opp.cr_api.Client(TOKEN, transport=fake, rate=fast_rate(opp), env={})
    players, last, path = opp.top_board(client, 3, NOW)
    assert [p["tag"] for p in players] == ["#02PYQ0"] and last.status == 200
    assert path == "locations/global/pathoflegend/2026-09/rankings/players"
    assert opp.recent_seasons(dt.datetime(2027, 1, 5)) == ["2027-01", "2026-12", "2026-11"]


def test_meta_if_due(opp, tmp_path, capsys):
    folder = tmp_path / "opponents"
    write_config(folder)
    (folder / "state.json").write_text(json.dumps({"meta_at": "2026-10-29T19:00:00Z"}),
                                       encoding="utf-8")
    fake = FakeTransport({})
    assert run(opp, folder, "meta", "--if-due", transport=fake) == opp.EXIT_OK
    assert fake.calls == [] and "meta      not due" in capsys.readouterr().out
    assert opp.meta_due({"meta_at": "2026-10-23T19:00:00Z"}, NOW)
    assert opp.meta_due({}, NOW)
