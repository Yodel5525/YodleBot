"""Friendly battles only, by construction (P03).

A person starts every battle. Nothing may press the lobby Battle button or queue
ladder or any other matchmade mode, and there is no option that would:
Navigator(queue=True) raises, the old --no-queue flags are accepted and change
nothing, and the launchers always run without queueing.
"""
from __future__ import annotations

import ast
import functools
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hastycr import machine

ROOT = Path(__file__).resolve().parents[2]
CODE_DIRS = ("scripts", "tools", "src", "sim")
SKIP_PARTS = {"vendor", "__pycache__", ".venvs", "tmp", "scrcpy"}
BATTLE_XY = (540, 1490)                      # the lobby Battle button, 1080x1920


def _python_files():
    for top in CODE_DIRS:
        for path in sorted((ROOT / top).rglob("*.py")):
            if SKIP_PARTS.isdisjoint(path.relative_to(ROOT).parts):
                yield path


@functools.cache
def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


@functools.cache
def _parse(path: Path):
    try:
        return ast.parse(_source(path), str(path))
    except SyntaxError:
        return None


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# ---------------------------------------------------------------------------
# the switch itself
# ---------------------------------------------------------------------------

def _true_or_alias(value) -> bool:
    return (isinstance(value, ast.Constant) and value.value is True) or _is_friendly_flag(value)


def test_friendly_only_is_a_constant_nobody_turns_off():
    assert machine.FRIENDLY_ONLY is True
    with pytest.raises(machine.FriendlyOnlyError, match="Friendly battles only"):
        machine.refuse_queue("Ladder")
    bad = []
    for path in _python_files():
        if "FRIENDLY_ONLY" not in _source(path):
            continue
        tree = _parse(path)
        for node in ast.walk(tree) if tree is not None else ():
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets, value = [node.target], getattr(node, "value", None)
            else:
                continue
            for target in targets:
                # machine.FRIENDLY_ONLY = ... (another module's switch), or a local
                # FRIENDLY_ONLY set to anything but True / machine.FRIENDLY_ONLY.
                if (getattr(target, "attr", None) == "FRIENDLY_ONLY" or (
                        getattr(target, "id", None) == "FRIENDLY_ONLY"
                        and (isinstance(node, ast.AugAssign) or not _true_or_alias(value)))):
                    bad.append(f"{_rel(path)}:{node.lineno}")
    assert bad == []


def test_there_is_no_ladder_option_anywhere():
    flags = []
    for path in _python_files():
        tree = _parse(path) if "add_argument" in _source(path) else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and re.search(
                            r"ladder|ranked|queue", arg.value, re.I):
                        flags.append((_rel(path), arg.value))
    # Left: the old flags, kept so old command lines still parse (--no-queue is a
    # no-op, --auto-queue is refused), and train_watchdog's entropy "ladder".
    allowed = {("scripts/cr_bot.py", "--no-queue"), ("scripts/gamestate/live_play.py", "--no-queue"),
               ("scripts/mumu_katacr.py", "--auto-queue"), ("scripts/train_watchdog.py", "--ladder")}
    assert ("scripts/cr_bot.py", "--no-queue") in flags, "the scan did not see scripts/"
    assert set(flags) <= allowed, sorted(set(flags) - allowed)
    for ps in [*ROOT.glob("*.ps1"), *(ROOT / "scripts").glob("*.ps1"), *ROOT.glob("*.cmd")]:
        assert not re.search(r"\$Ladder|-Ladder\b|--ladder", ps.read_text(encoding="utf-8"),
                             re.I), ps.name


# ---------------------------------------------------------------------------
# nothing taps the Battle button
# ---------------------------------------------------------------------------

def _is_friendly_flag(node) -> bool:
    return getattr(node, "id", None) == "FRIENDLY_ONLY" or getattr(node, "attr", None) == "FRIENDLY_ONLY"


def _mentions_battle_button(call: ast.Call) -> bool:
    for arg in [*call.args, *(kw.value for kw in call.keywords)]:
        for node in ast.walk(arg):
            if getattr(node, "id", None) == "HOME_BATTLE" or getattr(node, "attr", None) == "HOME_BATTLE":
                return True
            if (isinstance(node, ast.Tuple) and len(node.elts) == 2
                    and all(isinstance(e, ast.Constant) for e in node.elts)
                    and tuple(e.value for e in node.elts) == BATTLE_XY):
                return True
    return False


class _BattleTapFinder(ast.NodeVisitor):
    """Calls that pass the Battle button's position, outside `if not FRIENDLY_ONLY:`
    (or the else of `if FRIENDLY_ONLY:`)."""

    def __init__(self):
        self.guarded = 0
        self.hits: list[int] = []

    def _visit_block(self, nodes, guarded: bool):
        self.guarded += guarded
        for node in nodes:
            self.visit(node)
        self.guarded -= guarded

    def visit_If(self, node):
        test = node.test
        negated = isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and \
            _is_friendly_flag(test.operand)
        self.visit(test)
        self._visit_block(node.body, negated)
        self._visit_block(node.orelse, _is_friendly_flag(test))

    def visit_Call(self, node):
        if not self.guarded and _mentions_battle_button(node):
            self.hits.append(node.lineno)
        self.generic_visit(node)


def _battle_taps(source: str) -> list[int]:
    finder = _BattleTapFinder()
    finder.visit(ast.parse(source))
    return finder.hits


def test_the_finder_catches_the_old_battle_taps():
    old = ("guarded.tap(args.adb, args.serial, guarded.HOME_BATTLE)\n"
           "tap(serial, HOME_BATTLE)\n"
           "tap(args.adb, args.serial, (540, 1490))\n"
           "if FRIENDLY_ONLY:\n    tap(s, HOME_BATTLE)\n")
    assert _battle_taps(old) == [1, 2, 3, 5]
    assert _battle_taps("if not FRIENDLY_ONLY:\n    tap(s, HOME_BATTLE)\n") == []
    assert _battle_taps("if machine.FRIENDLY_ONLY:\n    pass\nelse:\n    tap(s, HOME_BATTLE)\n") == []


def test_no_code_taps_the_battle_button():
    hits, examined = [], set()
    for path in _python_files():
        text = _source(path)
        if "HOME_BATTLE" not in text and "1490" not in text:
            continue
        try:
            hits += [f"{_rel(path)}:{line}" for line in _battle_taps(text)]
        except SyntaxError:
            continue
        examined.add(_rel(path))
    # mumu_overnight_bot.py names the button's position in a comment, so it is examined.
    assert "scripts/mumu_overnight_bot.py" in examined, sorted(examined)
    assert hits == [], "taps on the lobby Battle button: " + ", ".join(hits)


def test_navigator_calls_never_ask_to_queue():
    bad, seen = [], 0
    for path in _python_files():
        tree = _parse(path) if "Navigator" in _source(path) else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "Navigator":
                continue
            seen += 1
            queue = [kw for kw in node.keywords if kw.arg == "queue"]
            if len(node.args) > 1 or any(not (isinstance(kw.value, ast.Constant)
                                              and kw.value.value is False) for kw in queue):
                bad.append(f"{_rel(path)}:{node.lineno}")
    assert seen >= 1, "found no Navigator(...) call (scripts/gamestate/live_play.py has one)"
    assert bad == []


def test_no_queue_flags_are_no_ops():
    """--no-queue is parsed but never read; --auto-queue only leads to a refusal."""
    for rel in ("scripts/cr_bot.py", "scripts/gamestate/live_play.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        reads = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr == "no_queue"]
        assert reads == [], f"{rel} still reads args.no_queue at {reads}"
    katacr = (ROOT / "scripts" / "mumu_katacr.py").read_text(encoding="utf-8")
    tree = ast.parse(katacr)
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "auto_queue"]
    assert len(uses) == 1 and "--auto-queue is disabled" in katacr


# ---------------------------------------------------------------------------
# Navigator behaviour
# ---------------------------------------------------------------------------

@pytest.fixture
def nav(monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from gamestate import nav as module
    return module


def test_navigator_refuses_to_queue(nav):
    with pytest.raises(machine.FriendlyOnlyError):
        nav.Navigator("127.0.0.1:16384", queue=True)
    with pytest.raises(machine.FriendlyOnlyError):
        nav.Navigator("127.0.0.1:16384", True)
    assert nav.Navigator("127.0.0.1:16384").queue is False
    assert not hasattr(nav, "HOME_BATTLE")


def test_navigator_never_taps_the_lobby_or_the_end_screen(nav, monkeypatch):
    screens = iter(["lobby"] * 3 + ["end_of_game"] * 3 + ["bypass_end_of_game"] * 2
                   + ["in_game", "lobby", "end_of_game"])
    taps, adb_calls = [], []
    monkeypatch.setattr(nav, "capture", lambda serial: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(nav, "classify", lambda small: next(screens))
    monkeypatch.setattr(nav, "tap", lambda serial, xy: taps.append(xy))
    monkeypatch.setattr(nav, "home_guard", lambda full: True)    # Battle button on screen
    monkeypatch.setattr(nav.subprocess, "run", lambda *a, **k: adb_calls.append(a))
    navigator = nav.Navigator("127.0.0.1:16384")
    notes = [navigator.step() for _ in range(11)]
    assert taps == [] and adb_calls == []
    assert all(note is None or "tapped" not in note for note in notes)
    assert notes[0] == "    screen None -> lobby" and notes[1] is None


# ---------------------------------------------------------------------------
# launchers
# ---------------------------------------------------------------------------

def test_supervisor_blocks_run_without_queueing(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import supervisor
    monkeypatch.setattr(supervisor, "LOG_PATH", tmp_path / "supervisor.log")
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    assert supervisor.play_block(Path("adb"), "127.0.0.1:16384", 5, 600) == 0
    assert seen[0][1] == "scripts/cr_bot.py" and "--no-queue" in seen[0]
    assert supervisor.play_block(Path("adb"), "127.0.0.1:16384", 5, 600,
                                 rl=Path("x.pt")) == 0
    assert "--no-queue" in seen[1] and "--advisor" not in seen[1]


def _ps_code(text: str) -> str:
    """PowerShell source without comments: `<# ... #>` blocks and whole-line `#` comments.

    The launchers explain in comments that queueing is refused; only code should count.
    """
    text = re.sub(r"<#.*?#>", "", text, flags=re.S)
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def test_ps_code_drops_comments_but_keeps_code():
    src = "<# --auto-queue in a block #>\n# --auto-queue in a line\n& $py bot.py --auto-queue\n"
    assert _ps_code(src).count("--auto-queue") == 1


def test_run_ps1_and_bot_cmd_always_skip_queueing():
    run = (ROOT / "run.ps1").read_text(encoding="utf-8")
    forced = run.index("$NoQueue = $true")
    assert forced < run.index("if ($NoQueue)    { $botArgs += '--no-queue' }")
    assert "HOME_BATTLE" not in run and "1490" not in run
    bot = (ROOT / "bot.cmd").read_text(encoding="utf-8")
    assert "run.ps1\" -Forever -NoQueue" in bot
    for rel in ("scripts/overnight_supervisor.ps1", "scripts/watch_katacr_overnight.ps1"):
        assert "--auto-queue" not in _ps_code((ROOT / rel).read_text(encoding="utf-8")), rel


def test_cr_bot_reports_queue_false():
    text = (ROOT / "scripts" / "cr_bot.py").read_text(encoding="utf-8")
    assert "queue=False" in text and "QUEUE #" not in text
