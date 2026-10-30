"""S11: tools/noise_scale.py and data/noise/stress_2x.json, the doubled-noise stress profile
(docs/LIVE_SEARCH_PLAN.md 7.2 M6, 9 S11).

The committed stress profile loads through tools/perception_noise, and each field of it is the
default prior's doubled (sigmas, spreads, the phantom rate), doubled and capped at 1
(probabilities, by-class drops, confusions), or the same (delays, biases, resolutions,
thresholds, settings), field by field against RULES; the tool regenerates the committed file
(parsed JSON: a CRLF checkout matches too) and `--check` says so; RULES covers every field of
every perception_noise group; on a hand-made profile the cap, the confusion row scaled to 1, the
uncapped rate, None, whole-tick spreads, notes left out, unknown keys and bad factors; main writes
LF bytes, names the profile after the file and --check notices a changed file.

Needs numpy (perception_noise imports it); no engine, torch or data beyond data/noise.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import noise_scale as NS  # noqa: E402
import perception_noise as PN  # noqa: E402  (the name NS.validate imports it by)

PRIOR = ROOT / "data" / "noise" / "default_prior.json"
STRESS = ROOT / "data" / "noise" / "stress_2x.json"


def doubled_p(v):
    return None if v is None else min(1.0, 2 * v)


def expected(rule: str, base):
    """What RULES make of one prior value, x 2."""
    if rule == NS.KEEP:
        return base
    if rule == NS.P:
        return doubled_p(base)
    if rule in (NS.SIGMA, NS.RATE, NS.SPREAD):
        return 2 * base
    if rule == NS.PMAP:
        return {k: doubled_p(v) for k, v in base.items()}
    if rule == NS.PMATRIX:
        out = {}
        for src, row in base.items():
            r = {k: doubled_p(v) for k, v in row.items()}
            total = sum(r.values())
            out[src] = {k: v / total for k, v in r.items()} if total > 1 else r
        return out
    raise AssertionError(rule)


# ---------------------------------------------------------------------------
# the committed profile
# ---------------------------------------------------------------------------

def test_the_stress_profile_loads_and_each_field_is_doubled_or_capped():
    base = PN.NoiseProfile.load("default")
    stress = PN.NoiseProfile.load(STRESS)
    assert stress.name == "stress_2x" and "times 2" in stress.description
    assert stress.active_groups() == base.active_groups() and not stress.is_identity()
    checked = 0
    for group, rules in NS.RULES.items():
        b, s = getattr(base, group), getattr(stress, group)
        for f in dataclasses.fields(b):
            want = expected(rules[f.name], getattr(b, f.name))
            got = getattr(s, f.name)
            if isinstance(want, dict):                   # by_class, pairs (rows), classes
                assert set(got) == set(want), (group, f.name)
                for k, v in want.items():
                    assert got[k] == pytest.approx(v), (group, f.name, k)
            elif want is None or isinstance(want, bool):
                assert got == want, (group, f.name)
            else:
                assert got == pytest.approx(want, abs=1e-12), (group, f.name, got, want)
            checked += 1
    assert checked == sum(len(r) for r in NS.RULES.values())
    # spot checks, by hand from data/noise/default_prior.json
    assert (stress.position.sigma_x_mt, stress.position.sigma_y_mt) == (500, 800)
    assert stress.position.flyer_sigma_mt == 800 and stress.position.flyer_dx_mt == 0
    assert (stress.drop.small, stress.drop.medium) == (pytest.approx(0.24), pytest.approx(0.1))
    assert stress.forms.evo_to_base_p == pytest.approx(0.6)
    assert stress.forms.hero_to_base_p == 1.0                    # 1.0 x 2, capped at 1
    assert stress.forms.unlabelled_to_base is True
    assert stress.confusion.pairs["IceSpirits"] == pytest.approx(
        {"FireSpirits": 0.04, "ElectroSpirit": 0.04, "HealSpirit": 0.02})
    assert stress.hp.no_bar_p == pytest.approx(0.3) and stress.hp.sigma == pytest.approx(0.06)
    assert (stress.hp.quant, stress.hp.full_snap) == (0.025, 0.95)          # kept
    assert (stress.new_units.delay_ticks, stress.new_units.jitter_ticks) == (2, 2)
    assert stress.false_positives.rate == pytest.approx(0.1)
    assert stress.false_positives.near_sigma_mt == 1400
    assert stress.elixir.fraction_sigma_milli == 240 and stress.clock.sigma_ticks == 4
    assert stress.opponent_plays.spell_miss_p == pytest.approx(0.3)
    assert (stress.opponent_plays.late_ticks, stress.opponent_plays.late_jitter_ticks) == (3, 4)
    assert stress.deploy.age_offset_ticks == 1 and stress.board.max_hold_ticks == 2


def test_the_tool_regenerates_the_committed_file():
    data = json.loads(STRESS.read_text(encoding="utf-8-sig"))
    assert NS.build() == data
    assert NS.build(PRIOR, 2) == data
    prior = json.loads(PRIOR.read_text(encoding="utf-8-sig"))
    assert data["_derived"] == {"tool": "tools/noise_scale.py",
                                "source": "data/noise/default_prior.json",
                                "source_name": "default_prior",
                                "source_digest": NS.source_digest(prior),
                                "factor": 2.0, "cap": 1.0}
    assert json.loads(NS.render(data)) == data and NS.render(data).endswith("}\n")
    assert NS.default_out() == STRESS
    assert NS.main(["--check"]) == 0
    groups = [g for g in data if not g.startswith("_") and g not in NS.META_KEYS]
    assert groups == [g for g in prior if not g.startswith("_") and g not in NS.META_KEYS]
    for g in groups:                                     # the prior's notes are left out
        assert not any(k.startswith("_") for k in data[g]), g


def test_every_perception_noise_field_has_a_rule():
    groups = PN.NoiseProfile.__dataclass_fields__
    profile = PN.NoiseProfile()
    names = [g for g in groups if g not in NS.META_KEYS]
    assert list(NS.RULES) == names                               # the same groups, in order
    for g in names:
        fields = [f.name for f in dataclasses.fields(getattr(profile, g))]
        assert list(NS.RULES[g]) == fields, g
        assert set(NS.RULES[g].values()) <= set(NS.RULE_KINDS), g
    assert set(PN.GROUPS) == set(names) - {"size"}


# ---------------------------------------------------------------------------
# the rules on a hand-made profile
# ---------------------------------------------------------------------------

HAND = {
    "schema": 1,
    "name": "hand",
    "_sources": ["a note"],
    "drop": {"small": 0.7, "medium": 0, "by_class": {"Knight": 0.6, "Golem": 0.1, "_why": "x"},
             "_why": "notes are left out"},
    "forms": {"evo_to_base_p": 0.25, "hero_to_base_p": 0.6, "unlabelled_to_base": False},
    "confusion": {"unknown_p": 0.5, "pairs": {"A": {"B": 0.4, "C": 0.3}, "D": {"E": 0.2}}},
    "position": {"sigma_x_mt": 250, "flyer_dx_mt": 600, "flyer_sigma_mt": 0.5},
    "false_positives": {"rate": 0.8, "classes": {"Skeleton": 3, "_n": 1}, "near_sigma_mt": 700},
    "clock": {"bias_ticks": -2, "sigma_ticks": 3, "quant_ticks": 2},
    "new_units": {"delay_ticks": 2, "jitter_ticks": 1},
    "opponent_plays": {"miss_p": 0.03, "spell_miss_p": None, "late_jitter_ticks": 3},
    "size": None,
}


def test_the_rules_on_a_hand_made_profile():
    out = NS.scaled(HAND, 2)
    assert HAND["drop"]["small"] == 0.7                       # the input is not changed
    assert list(out)[:4] == ["schema", "name", "description", "_derived"]
    assert out["name"] == "stress_2x" and out["_derived"]["source"] is None
    assert out["_derived"]["source_name"] == "hand" and "_sources" not in out
    assert out["drop"] == {"small": 1.0, "medium": 0, "by_class": {"Knight": 1.0, "Golem": 0.2}}
    assert out["forms"] == {"evo_to_base_p": 0.5, "hero_to_base_p": 1.0,
                            "unlabelled_to_base": False}
    assert out["confusion"]["unknown_p"] == 1.0
    a = out["confusion"]["pairs"]["A"]                        # 0.8 + 0.6 > 1: scaled to 1
    assert a == pytest.approx({"B": 0.8 / 1.4, "C": 0.6 / 1.4}) and sum(a.values()) <= 1 + 1e-12
    assert out["confusion"]["pairs"]["D"] == {"E": 0.4}
    assert out["position"] == {"sigma_x_mt": 500, "flyer_dx_mt": 600, "flyer_sigma_mt": 1.0}
    assert isinstance(out["position"]["sigma_x_mt"], int)
    assert out["false_positives"] == {"rate": 1.6, "classes": {"Skeleton": 3},
                                      "near_sigma_mt": 1400}                 # a rate: no cap
    assert out["clock"] == {"bias_ticks": -2, "sigma_ticks": 6, "quant_ticks": 2}
    assert out["new_units"] == {"delay_ticks": 2, "jitter_ticks": 2}
    assert out["opponent_plays"] == {"miss_p": 0.06, "spell_miss_p": None, "late_jitter_ticks": 6}
    assert out["size"] is None
    profile = NS.validate(out)                                # perception_noise loads it
    assert profile.confusion.pairs["A"] == pytest.approx(a)
    assert profile.false_positives.rate == pytest.approx(1.6)
    half = NS.scaled(HAND, 0.5, name="calm")
    assert half["name"] == "calm" and half["drop"]["small"] == 0.35
    assert half["new_units"]["jitter_ticks"] == 1            # 0.5 rounds half up
    assert half["opponent_plays"]["late_jitter_ticks"] == 2  # 1.5 rounds half up
    assert half["position"]["sigma_x_mt"] == 125
    assert NS.scaled(HAND, 1.5)["position"]["sigma_x_mt"] == 375.0
    NS.validate(half)


@pytest.mark.parametrize("bad, match", [
    ({"drop": {"tiny": 0.1}}, "unknown key"),
    ({"wobble": {}}, "unknown group"),
    ({"drop": {"small": "high"}}, "a number"),
    ({"drop": {"small": True}}, "a number"),
    ({"drop": {"by_class": {"Knight": None}}}, "a number"),
    ({"drop": 0.1}, "an object"),
    ({"confusion": {"pairs": {"A": 0.1}}}, "an object"),
])
def test_bad_profiles_are_refused(bad, match):
    with pytest.raises(NS.NoiseScaleError, match=match):
        NS.scaled(bad, 2)


@pytest.mark.parametrize("factor", [0, -1, float("inf"), float("nan"), "2", True])
def test_bad_factors_are_refused(factor):
    with pytest.raises(NS.NoiseScaleError, match="factor"):
        NS.scaled({}, factor)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main_writes_checks_and_names_the_profile_after_the_file(tmp_path, capsys):
    src = tmp_path / "measured.json"
    src.write_text(json.dumps(HAND), encoding="utf-8")
    assert NS.default_out(src, 2) == tmp_path / "measured_2x.json"
    assert NS.main(["--source", str(src)]) == 0
    out = tmp_path / "measured_2x.json"
    raw = out.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"}\n")
    data = json.loads(raw.decode("utf-8"))
    assert data["name"] == "measured_2x" and data["_derived"]["source"] == str(src)
    assert data == NS.build(src, 2, name="measured_2x")
    assert "wrote" in capsys.readouterr().out
    assert NS.main(["--source", str(src), "--check"]) == 0
    crlf = raw.replace(b"\n", b"\r\n")                        # a Windows checkout still matches
    out.write_bytes(crlf)
    assert NS.main(["--source", str(src), "--check"]) == 0
    data["drop"]["small"] = 0.5
    out.write_text(json.dumps(data), encoding="utf-8")
    assert NS.main(["--source", str(src), "--check"]) == 1
    assert "regenerate" in capsys.readouterr().err
    assert NS.main(["--source", str(src), "--check", "--out", str(tmp_path / "none.json")]) == 1
    three = tmp_path / "x3.json"
    assert NS.main(["--source", str(src), "--factor", "3", "--out", str(three),
                    "--name", "triple"]) == 0
    assert json.loads(three.read_text(encoding="utf-8"))["name"] == "triple"
    assert NS.main(["--source", str(tmp_path / "missing.json")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"drop": {"small": 0.7, "tiny": 1}}), encoding="utf-8")
    assert NS.main(["--source", str(bad)]) == 2
    assert NS.main(["--source", str(src), "--factor", "0"]) == 2
