r"""Auto-labelling of recorded friendlies: the label registry, ingest, status, scan, sheet, label.

docs/NEW_CARDS_PLAN.md 2.3, 4 and 5 (items N07-N10). The owner's main account plays the
post-2024 cards against Yodle2 in friendlies; Yodle2 records them; this tool turns the
recordings into detector labels. This file holds:

  labels   the append-only label registry data/vision/detector_extra_labels.json (N07),
           checked against data/vision/classes.json and, with a base, appended to the label set
           scripts/prepare_yolo_dataset.py wrote (tmp/yolo/detector_labels.json).
  ingest   one recorded friendly into a per-session store, at once (N08): its images are
           hard-linked, so the recorder's disk budget can no longer prune them; the deck is
           checked; the role (train, gold, dev) is fixed; data/vision/owner_labels/<sid>/
           session.json is written.
  status   the ingested sessions, what each has done, and which registry labels still have no
           training or gold session (plan 6.4: what to record next).
  scan     the N09 scan of an ingested session (scripts/vision/unit_harvest.scan_session) into
           <store>/<sid>/scan: tonight's live chain and the harvest chain on every image.
  sheet    N10: the enemy tracks of the scan flagged (S1-S7), split where their look changes,
           auto-accepted when safe, clustered, and drawn as a numbered contact sheet with the
           declared deck's legend (<store>/<sid>/sheet/sheet_<k>.png, legend_<k>.txt).
  label    N10: one map string per sheet ("0=3,1=3,2=x,5=s") names the clusters; the labels go
           to data/vision/owner_labels/<sid>/tracks.json. A bad entry stores nothing.

Later items (emit, review, mark, plays, mix: N11-N13) are added to this file as their own
commands; they find a session through session_paths() and load_session().

Owner commands (PowerShell, from the repo root; $BB = .venvs\buildabot\Scripts\python.exe):

    # the registry: check it, and see the indices it appends to tonight's label set
    & $BB tools\vision_autolabel.py labels --base tmp\yolo\detector_labels.json

    # after each recorded friendly, at once (plan 4.1)
    $SID = '20261101_A_1'      # date, deck letter, friendly number
    & $BB tools\vision_autolabel.py ingest reports\vision\harvest\$SID --deck "goblinstein,rune-giant,goblin-demolisher,suspicious-bush,vines,goblin-curse,elite-barbarians-ev1,baby-dragon-ev1" --role auto --client 16.635.1

    # Round 0: the 2026-10-29 friendly, copied to reports\vision\gold (gold: no auto-accept)
    $R0 = 'reports\vision\gold'                       # its session folder, or a folder holding only it
    $R0DECK = 'card-1,card-2,card-3,card-4,card-5,card-6,card-7,card-8'   # the main account's 8 cards
    & $BB tools\vision_autolabel.py ingest $R0 --sid 20261029_R0_1 --role gold --capture natural --deck $R0DECK --client 16.635.1

    # what is ingested, and what to record next
    & $BB tools\vision_autolabel.py status

    # then, per session (MuMu and the browser closed for the scan: budget 10-20 min on the GPU)
    & $BB tools\vision_autolabel.py scan $SID
    & $BB tools\vision_autolabel.py sheet $SID          # open the sheet PNG(s) it names; the legend is printed
    & $BB tools\vision_autolabel.py label $SID --sheet 1 --map "0=3,1=3,2=x,5=s"
    & $BB tools\vision_autolabel.py sheet $SID          # again while clusters are left or were split (s)
    & $BB tools\vision_autolabel.py label $SID --sheet 2 --map "0=goblinstein,1=?"

    --sheet is the number printed on the sheet the map was typed from: cluster ids are new on
    every sheet, so a map typed from an older sheet is refused instead of labelling other
    tracks. Track keys (t17=5, t40.1=x) stay the same across sheets.

The registry (N07)
  Each entry appends one label: YOLO classes 2 * (len(base) + k) + side. The base is the real
  tmp/yolo/detector_labels.json, not a list in the code (plan 8: prepare_yolo_dataset may append
  KataCR names the class list does not map). Registry.extend() returns the extended LabelSet;
  the base's yolo_names() is always a prefix of it (check_prefix), and `labels --against OLD`
  checks the same against any older extended label set, so a registry edit that removes or
  reorders an entry is caught before a model is trained on it. Every entry must resolve through
  scripts/vision/labels.resolve_label as what it claims to be (unit form of the named card, or the
  spell card itself) and be written as ClassTable.detector_key writes it.
  Thin forms: a hero or evolution entry trains as its base label while it has fewer than
  thin_tracks (8) labelled training tracks, so the card is still named right (train_label,
  train_index); "train_as" pins the choice. valid_from_client (per entry) and "reworks" (for
  base labels) drop sessions recorded on an older client from that label (valid_for).

Ingest (N08)
  The recording is a scripts/vision/record.py session (<session>/frames.jsonl plus
  images/<8 digits>.jpg). `capture --record reports\vision\harvest\$SID` writes one session
  folder (capture_<time>) under that root: ingest accepts the root when it holds exactly one
  session, or the session folder itself (then give --sid). The session id is the root's name
  by default.
  Every image file is hard-linked into <store>/<sid>/recording/images (copied when the store is
  on another drive, and reported), and frames.jsonl is copied next to them, so the store is
  itself a record.py session that harvest.open_recording and VisionLogReader read. The store
  (default reports\vision\autolabel, gitignored) is never committed: frames show account names.
  Warnings stored in session.json have machine paths cut (scrub_paths).
  The store keeps the recording one level down (<sid>/recording), so a recorder pointed at the
  store root by mistake would not see sessions there to prune.
  The deck: 8 cards, each a RoyaleAPI key as in data/ladder/battles_2p6.csv (hog-rider,
  ronin-hero, cannon-ev1: tools/rs_cards.py's map with its extras) or a catalogue name of
  classes.json (MinionGiant, which has no key yet), with -ev1 or -hero for the evolution or hero
  slot. Refused: a card classes.json lacks, a form it lacks, a duplicate, a deck not of 8, and
  the 22 event-only or internal cards of data/vision/event_cards.json (SuperKnight;
  MergeMaiden_Normal, where spirit-empress is meant). Warnings, not refusals: the observed slot
  rules (plan 1.10: at most 2 evolutions, at most 2 heroes and champions together, at most 3 in
  all) and deck units that have no detector label yet.
  Roles (plan 4.4, 4.5, 6): --role auto reads a session id date_DECK_N: friendly 2 of a deck is
  gold (held out, never trained on), deck A's friendly 4 is dev (early stopping only), every
  other friendly trains. --capture natural (Round 0) and field (clanmate friendlies) are gold
  only. A role, deck, client or capture once written is changed only with --force.
  session.json (committed, small): sid, role, capture, client, the deck as checked, the
  recording's counts and frames.jsonl SHA-1, the store counts, digests and warnings. Paths are
  written relative to the repo, or not at all, never as machine paths.
  Ingesting again with the same deck and role is safe: images not linked yet are linked (a
  recording still being written, an interrupted ingest), frames.jsonl is refreshed.
  Disk: the store's images are hard links of the recording's, so deleting the recording root
  (reports\vision\harvest\$SID) frees nothing while <store>/<sid>/recording/images holds the
  same files; delete both to get the space back (training raws only, after N11 emit; keep
  gold raws, plan 4.7). status counts the store's images. The scan adds 150-300 MB per battle
  in <store>/<sid>/scan (unit_harvest module doc).

Scan, sheet and label (N09, N10)
  scan runs scripts/vision/unit_harvest.scan_session on <store>/<sid>/recording into
  <store>/<sid>/scan (Yodle2 plays Hog 2.6; the detector is the pin unless --detector). A scan
  again (--force) is refused while tracks.json holds labels: a new scan has new track ids.
  sheet runs unit_harvest.analyze_scan (cached in <store>/<sid>/sheet/analysis.json and
  crops.npz, keyed by the scan's SHA-1s, the deck and the settings; --rebuild redoes it):
  the enemy and side-unknown harvest tracks, split where the look changes (S7), flagged S1-S7
  (unit_harvest module doc, N10), and auto-accepted when unflagged, on the enemy side, long,
  well covered by one confident live read that is on the legend and none of the no-auto
  reasons holds (form: an evolution or hero slot's class read in another form, ours, units,
  side, half; unit_harvest.flag_part). A gold session (role gold, capture natural or field)
  never auto-accepts: every track goes on its sheet (plan 5, 6); --no-auto does the same for
  any session. The other tracks are clustered by look (two confident reads of different
  labels never share a cluster); each tile of the sheet is one cluster: up to 4 member tracks
  (birth, +1 s, mid-life; one of each suggested label first), its flags ('weak' when none:
  not confident enough), what tonight's model read (live class/how, best harvest class and
  score), a suggested legend number ('12|13 which form?' when the read cannot tell the form:
  type the form you see), and a last line 'mix 3x6 2x8' when members suggest different
  labels (type s to see them one per tile) or 'no auto: ...' with the reasons. Up to
  --page-tiles clusters per image: sheet_<k>.png, or sheet_<k>_p1.png, _p2, ... .
  The legend lists only the declared deck's labels (build_legend), numbered, '*' where the
  label set has none yet, and the codes x (an effect), o (ours), ? (unsure: masked in
  training), s (split: those tracks come back one per tile on the next sheet). Sheet k
  replaces sheet k-1 in <store>/<sid>/sheet/sheet.json; tracks labelled before are left off.
  label reads the latest sheet.json. Map keys: a cluster id (then --sheet N, the sheet's
  number, is needed), a range (4-7), or a track key (t17, t40.1 for a part of a split
  track); values: a legend number, a name classes.json reads that is on the legend
  (goblinstein, Vines, BerserkerHero; --off-deck for a name off it), or a code. Every entry is
  checked first (parse_map, resolve_value): an unknown or off-legend name, a cluster or track
  not on the sheet, a track labelled before (without --force), a sheet of another scan or not
  the latest: all are listed and NOTHING is stored (exit 2). Then the auto-accepted tracks of
  the sheet are stored too (source "auto"; none in a gold session).
  tracks.json (committed, small): {"tracks": [{"key", "h", "part", "first_n", "last_n", "n",
  "label" (a detector label) or "code" (x, o, ?), "side", "side_read", "source", "sheet",
  "cluster", "flags", "read", ...}], "split": [keys], "scan": the scan's identity, "client",
  "history"}. The contract for N11 (emit) and N13 (mix):
    - "side" is what a label trains as: "enemy" for every label (the legend is the declared
      opponent's deck), "ally" for o, the scan's read side for x and ?; "side_read" is the
      scan's (enemy, ally or unknown).
    - A part's images are its harvest track h's (the scan's tracks.json) with n in
      [first_n, last_n]. read.held lists every live track the part held, [key, team, class,
      form, images, first n, last n], so the images of another unit can be left out.
    - session.json is the only source of a session's role (gold is never trained on);
      tracks.json does not copy it, so an `ingest --force` role change cannot go stale here.
    - A label not in the registry yet (a '*' legend entry) trains as its class's base form
      (Registry.train_label with the table) until it is appended and has thin_tracks tracks;
      class indices are written only with Registry.extend(base, require_pin=True).
    - Auto-accepted tracks (source "auto") are on no cluster: N12's review samples them as a
      group of their own.
  status counts the labelled units of training sessions (labelled_tracks: one per harvest or
  live track, not per part).

Exit codes: 0 ok; 1 a check failed or a change was refused; 2 bad input.

Standard library only for labels, ingest, status and label (scripts/vision/labels.py loads
tools/vision_classes.py; tools/rs_cards.py gives the RoyaleAPI keys), so they work whatever the
vision packages do; scan and sheet import scripts/vision/unit_harvest.py (numpy) when they run.
Nothing here talks to the emulator, reads game memory or taps.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:                    # rs_cards
    sys.path.insert(0, str(TOOLS))
if str(ROOT) not in sys.path:                     # scripts.vision.labels
    sys.path.append(str(ROOT))

from scripts.vision import labels as L  # noqa: E402

REGISTRY_PATH = ROOT / "data" / "vision" / "detector_extra_labels.json"
EVENT_CARDS_PATH = ROOT / "data" / "vision" / "event_cards.json"
LABELS_ROOT = ROOT / "data" / "vision" / "owner_labels"      # committed: small JSON only
STORE_ROOT = ROOT / "reports" / "vision" / "autolabel"         # gitignored: images
BASE_LABELS = ROOT / "tmp" / "yolo" / "detector_labels.json"  # scripts/prepare_yolo_dataset.py OUT

REGISTRY_SCHEMA = "hastycr.vision.detector_extra_labels/1"
EVENT_CARDS_SCHEMA = "hastycr.vision.event_cards/1"
SESSION_SCHEMA = "hastycr.vision.autolabel_session/1"

# The recorder's layout (scripts/vision/record.py:58-61), restated so ingest needs no numpy.
FRAMES_FILE = "frames.jsonl"
IMAGES_DIR = "images"
IMAGE_NAME_RE = re.compile(r"^\d{8}\.(jpg|png|npy)$")

SESSION_FILE = "session.json"          # in data/vision/owner_labels/<sid>/
RECORDING_DIR = "recording"            # in <store>/<sid>/: frames.jsonl and images/
#: Track labels of a session (N10's `label` writes it). What status reads of it:
#: {"tracks": [{"label": <detector label>, ...}, ...]}; anything else is ignored.
TRACKS_FILE = "tracks.json"
SCAN_DIR = "scan"                      # in <store>/<sid>/: the N09 scan (unit_harvest.scan_session)
SHEET_DIR = "sheet"                    # in <store>/<sid>/: sheets, legends, the analysis cache
SHEET_FILE = "sheet.json"              # in <store>/<sid>/sheet/: the latest sheet's clusters
#: The scan's own files (scripts/vision/unit_harvest.py SCAN_JSON, TRACKS_JSON), restated so
#: `label` needs no numpy (pinned equal by a test).
SCAN_JSON = "scan.json"
SCAN_TRACKS = "tracks.json"
TRACKS_SCHEMA = "hastycr.vision.autolabel_tracks/1"
SHEET_SCHEMA = "hastycr.vision.autolabel_sheet/1"
#: (stage, "labels" or "store", path in that session folder): a stage is done when the path
#: exists. N11-N13 append theirs.
STAGES: list[tuple[str, str, str]] = [("ingest", "labels", SESSION_FILE),
                                      ("scan", "store", f"{SCAN_DIR}/{SCAN_JSON}"),
                                      ("sheet", "store", f"{SHEET_DIR}/{SHEET_FILE}"),
                                      ("label", "labels", TRACKS_FILE)]

FORM_BASE, FORM_EVO, FORM_HERO = 0, 1, 2        # tools/vision_classes.py FORMS
FORM_NAMES = ("base", "evo", "hero")
#: RoyaleAPI form suffixes, as tools/royalesim_redrive.py:150-154 base_key reads them.
KEY_SUFFIXES = (("-ev1", FORM_EVO), ("-hero", FORM_HERO))
THIN_TRACKS = 8                                 # plan 2.3: below this a form trains as its base
TRAIN_SELF = "self"
ENTRY_TYPES = (L.TYPE_UNIT, L.TYPE_TOWER, L.TYPE_SPELL)

ROLE_TRAIN, ROLE_GOLD, ROLE_DEV = "train", "gold", "dev"
ROLES = (ROLE_TRAIN, ROLE_GOLD, ROLE_DEV)
ROLE_AUTO = "auto"
CAPTURE_HARVEST, CAPTURE_NATURAL, CAPTURE_FIELD = "harvest", "natural", "field"
CAPTURES = (CAPTURE_HARVEST, CAPTURE_NATURAL, CAPTURE_FIELD)
GOLD_ONLY_CAPTURES = frozenset({CAPTURE_NATURAL, CAPTURE_FIELD})     # plan 6: never trained on
SID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
HARVEST_SID_RE = re.compile(r"^(?P<date>\d{8})_(?P<deck>[A-Za-z][A-Za-z0-9]*)_(?P<n>[1-9][0-9]*)$")
GOLD_FRIENDLY = 2                               # plan 4.4 / 4.5: friendly 2 of each deck
DEV_FRIENDLIES = frozenset({("A", 4)})          # plan 4.4: deck A's friendly 4
DECK_SIZE = 8
#: Champions of the 16.402.22 pool, for the slot-rule warnings only: data/royaleapi/cards.json
#: rarity Champion (its terry is an event card), plus Little Prince, Goblinstein and Boss Bandit
#: (plan 1.2), which that file predates.
CHAMPIONS = frozenset({"ArcherQueen", "GoldenKnight", "SkeletonKing", "MightyMiner", "Monk",
                       "LittlePrince", "Goblinstein", "BossBandit"})
RECENT_WRITE_S = 15.0                           # frames.jsonl younger than this: still recording?

EXIT_OK, EXIT_CHECK, EXIT_INPUT = 0, 1, 2


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _read_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))     # a Windows editor may add a BOM


def write_json(path: str | os.PathLike, doc: Any) -> Path:
    """Write `doc` as indented JSON, atomically (temp file + rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def shown_path(path: str | os.PathLike | None) -> str | None:
    """A path as committed JSON may show it: relative to the repo (posix), else None."""
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


_PATH_END = r"[^\s'\"<>|,;()]"
#: An absolute path in a message: a drive (C:\\... or C:/...), a UNC share, or a POSIX path of
#: two parts or more, not inside a word ("train/dev" stays).
_ABS_PATH_RE = re.compile(r"(?<![\w.])(?:[A-Za-z]:[\\/]" + _PATH_END + r"*|\\\\" + _PATH_END + r"+"
                          r"|/(?:[^\s/'\"<>|,;()]+/)+" + _PATH_END + r"*)")


def scrub_paths(text: Any) -> str:
    """A message as committed JSON may hold it: paths under the repo made relative, any other
    absolute path cut to '<path>/' and its last part. Error texts (a card map or
    registry that did not load) carry the machine's paths; session.json is committed."""
    out = str(text)
    for root in sorted({str(ROOT), ROOT.as_posix()}, key=len, reverse=True):
        for sep in ("\\", "/"):
            out = out.replace(root + sep, "")
        out = out.replace(root, ".")

    def short(m: re.Match) -> str:
        whole = m.group(0)
        path = whole.rstrip(".:")
        name = re.split(r"[\\/]", path.rstrip("\\/"))[-1]
        return f"<path>/{name}{whole[len(path):]}"

    return _ABS_PATH_RE.sub(short, out)


def _inside(child: Path, parent: Path) -> bool:
    child, parent = child.resolve(), parent.resolve()
    return child == parent or parent in child.parents


def parse_version(text: Any) -> tuple[int, ...]:
    """"16.635.1" -> (16, 635, 1); trailing zeros dropped, so 16.635.0 == 16.635. ValueError
    for anything that is not dot-separated whole numbers."""
    s = str(text).strip()
    if not re.fullmatch(r"\d+(\.\d+)*", s):
        raise ValueError(f"not a client version like 16.635.1: {text!r}")
    parts = [int(x) for x in s.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def check_sid(sid: Any) -> str:
    s = str(sid)
    if not SID_RE.match(s):
        raise ValueError(f"session id {sid!r}: use letters, digits, '_', '-' and '.' (like 20261101_A_1)")
    return s


def load_table(path: str | os.PathLike | None = None):
    """data/vision/classes.json (or `path`) as a tools/vision_classes.py ClassTable."""
    return L.load_class_table(path)


# ---------------------------------------------------------------------------
# the label registry (N07)
# ---------------------------------------------------------------------------

class RegistryError(ValueError):
    """The registry file is malformed, contradicts classes.json, or would move class indices."""

    def __init__(self, problems: str | Iterable[str]):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class ExtraLabel:
    """One registry entry: a detector label appended to the base label set."""
    label: str
    card: str
    type: str
    form: int = FORM_BASE
    base: str | None = None              # form entries: the base-form label
    train_as: str | None = None          # None: automatic; "self"; or a label name
    valid_from_client: str | None = None
    phase: int | None = None
    added: str | None = None
    note: str = ""

    @property
    def form_name(self) -> str:
        return FORM_NAMES[self.form]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"label": self.label, "card": self.card, "type": self.type,
                             "form": self.form_name}
        if self.base is not None:
            d["base"] = self.base
        d.update({"train_as": self.train_as, "valid_from_client": self.valid_from_client,
                  "phase": self.phase, "added": self.added, "note": self.note})
        return d


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _parse_entry(d: Any, where: str) -> tuple[ExtraLabel | None, list[str]]:
    if not isinstance(d, Mapping):
        return None, [f"{where}: not an object"]
    problems = []
    label, card, typ = d.get("label"), d.get("card"), d.get("type")
    if not _name(label):
        problems.append(f"{where}: label must be a non-empty name, got {label!r}")
    elif L.SIDE_SEP in label:
        problems.append(f"{where}: label {label!r} must not contain {L.SIDE_SEP!r}")
    if not _name(card):
        problems.append(f"{where}: card must be a catalogue name, got {card!r}")
    if typ not in ENTRY_TYPES:
        problems.append(f"{where}: type must be one of {', '.join(ENTRY_TYPES)}, got {typ!r}")
    form = d.get("form", FORM_NAMES[FORM_BASE])
    if form in FORM_NAMES:
        form = FORM_NAMES.index(form)
    elif not (isinstance(form, int) and not isinstance(form, bool) and 0 <= form < len(FORM_NAMES)):
        problems.append(f"{where}: form must be base, evo or hero, got {form!r}")
        form = FORM_BASE
    base, train_as, valid = d.get("base"), d.get("train_as"), d.get("valid_from_client")
    for key, value in (("base", base), ("train_as", train_as)):
        if value is not None and not _name(value):
            problems.append(f"{where}: {key} must be a label name or null, got {value!r}")
    if form == FORM_BASE and base is not None:
        problems.append(f"{where}: base is for evo and hero entries only")
    if form != FORM_BASE and base is None:
        problems.append(f"{where}: an {FORM_NAMES[form]} entry needs base, the base-form label it "
                        "trains as while it is thin")
    if train_as is not None and train_as == label:
        problems.append(f"{where}: train_as names the label itself; use \"{TRAIN_SELF}\"")
    if valid is not None:
        try:
            parse_version(valid)
        except ValueError as exc:
            problems.append(f"{where}: valid_from_client: {exc}")
    phase = d.get("phase")
    if phase is not None and (isinstance(phase, bool) or not isinstance(phase, int)):
        problems.append(f"{where}: phase must be a whole number or null")
    if problems:
        return None, problems
    return ExtraLabel(label, card, typ, form, base, train_as, None if valid is None else str(valid),
                      phase, None if d.get("added") is None else str(d.get("added")),
                      str(d.get("note") or "")), []


class Registry:
    """data/vision/detector_extra_labels.json, loaded. Entries in index order.

        reg = Registry.load()
        reg.validate(table)                      problems against classes.json ([] = fine)
        ext = reg.extend(base)                   the base LabelSet plus every entry, in order
        reg.train_index(ext, "BerserkerHero", 1, tracks=3)   the base index while thin
        reg.valid_for("Vines", "16.635.1")       may a session of that client teach the label
    """

    def __init__(self, entries: Sequence[ExtraLabel], *, thin_tracks: int = THIN_TRACKS,
                 reworks: Mapping[str, str] | None = None, classes_digest: str | None = None,
                 base_pin: Mapping[str, Any] | None = None, path: Path | None = None):
        self.entries: tuple[ExtraLabel, ...] = tuple(entries)
        self.thin_tracks = int(thin_tracks)
        self.reworks = {str(k): str(v) for k, v in (reworks or {}).items()}
        self.classes_digest = classes_digest
        self.base_pin = dict(base_pin or {})
        self.path = path
        self._by_label: dict[str, ExtraLabel] = {}
        dups = []
        for e in self.entries:
            if e.label in self._by_label:
                dups.append(f"label {e.label!r} appears twice")
            self._by_label[e.label] = e
        for label, version in self.reworks.items():
            try:
                parse_version(version)
            except ValueError as exc:
                dups.append(f"reworks[{label!r}]: {exc}")
        if dups:
            raise RegistryError(dups)

    # ------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, doc: Mapping[str, Any], path: Path | None = None) -> "Registry":
        if not isinstance(doc, Mapping):
            raise RegistryError("the registry must be a JSON object")
        if doc.get("schema") != REGISTRY_SCHEMA:
            raise RegistryError(f"schema must be {REGISTRY_SCHEMA!r}, got {doc.get('schema')!r}")
        raw = doc.get("labels")
        if not isinstance(raw, list):
            raise RegistryError("labels must be a list of entries")
        entries, problems = [], []
        for i, d in enumerate(raw):
            e, p = _parse_entry(d, f"labels[{i}]")
            problems += p
            if e is not None:
                entries.append(e)
        thin = doc.get("thin_tracks", THIN_TRACKS)
        if isinstance(thin, bool) or not isinstance(thin, int) or thin < 1:
            problems.append(f"thin_tracks must be a positive whole number, got {thin!r}")
            thin = THIN_TRACKS
        reworks = doc.get("reworks") or {}
        if not isinstance(reworks, Mapping):
            problems.append("reworks must be an object of label -> client version")
            reworks = {}
        for label, version in reworks.items():
            try:
                parse_version(version)
            except ValueError as exc:
                problems.append(f"reworks[{label!r}]: {exc}")
        base_pin = doc.get("base") or {}
        if not isinstance(base_pin, Mapping):
            problems.append("base must be an object")
            base_pin = {}
        if problems:
            raise RegistryError(problems)
        return cls(entries, thin_tracks=thin, reworks=reworks, classes_digest=doc.get("classes_digest"),
                   base_pin=base_pin, path=path)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Registry":
        p = Path(path) if path is not None else REGISTRY_PATH
        try:
            doc = _read_json(p)
        except FileNotFoundError:
            raise RegistryError(f"no registry at {p}") from None
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{p} is not valid JSON: {exc}") from None
        return cls.from_dict(doc, p)

    # ------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __contains__(self, label: object) -> bool:
        return label in self._by_label

    def get(self, label: str) -> ExtraLabel | None:
        return self._by_label.get(str(label))

    @property
    def labels(self) -> list[str]:
        return [e.label for e in self.entries]

    @property
    def digest(self) -> str:
        """SHA-1 of the labels in order, one per line: equal means the same appended indices."""
        return hashlib.sha1("\n".join(self.labels).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------ checks
    def validate(self, table: Any, event_cards: Mapping[str, Any] | None = None) -> list[str]:
        """Problems of the entries against a class table ([] when every entry resolves
        through labels.resolve_label as what it claims to be)."""
        events = load_event_cards() if event_cards is None else event_cards
        problems: list[str] = []
        for e in self.entries:
            where = repr(e.label)
            lab = L.resolve_label(table, e.label)
            if lab.type not in ENTRY_TYPES:
                problems.append(f"{where} does not resolve to a unit, tower or spell of the class "
                                f"table (it reads as {lab.type})")
                continue
            if lab.type != e.type:
                problems.append(f"{where} is a {lab.type} in the class table, not a {e.type}")
                continue
            info = table.cards.get(e.card)
            if info is None:
                problems.append(f"{where}: card {e.card!r} is not a card of the class table")
            elif e.card in events:
                problems.append(f"{where}: card {e.card!r} is not a deck card "
                                f"({events[e.card].get('kind', 'event')}); name the card a deck holds")
            elif e.form not in [int(f) for f in info.get("forms") or [FORM_BASE]]:
                problems.append(f"{where}: card {e.card} has no {e.form_name} form in the class table")
            if lab.form != e.form:
                problems.append(f"{where} reads as the {FORM_NAMES[lab.form] if 0 <= lab.form < 3 else lab.form} "
                                f"form, the entry says {e.form_name}")
            if lab.type == L.TYPE_SPELL:
                if lab.card != e.label:
                    problems.append(f"{where}: a spell label is its card name ({lab.card!r})")
                if lab.card != e.card:
                    problems.append(f"{where} is the spell {lab.card!r}, not card {e.card!r}")
            else:
                key = table.detector_key(lab.class_id, lab.form)
                if key != e.label:
                    problems.append(f"{where} is not how the class table names this form: write {key!r}")
                vc = table.classes[lab.class_id]
                if e.card not in vc.card_names:
                    problems.append(f"{where}: class {vc.name} is made by {', '.join(vc.card_names) or 'no card'}, "
                                    f"not {e.card}")
                if e.form != FORM_BASE:
                    want = table.detector_key(lab.class_id, FORM_BASE)
                    if e.base != want:
                        problems.append(f"{where}: base must be {want!r}, the base form of {vc.name}")
            target = e.train_as if e.train_as != TRAIN_SELF else None
            if target is not None:
                t = L.resolve_label(table, target)
                same = (t.type == lab.type and t.form == FORM_BASE
                        and ((t.class_id is not None and t.class_id == lab.class_id)
                             or (t.type == L.TYPE_SPELL and t.card == lab.card)))
                if not same:
                    problems.append(f"{where}: train_as {target!r} is not the base form of the same "
                                    "class or spell")
        for label in self.reworks:
            if L.resolve_label(table, label).type not in ENTRY_TYPES:
                problems.append(f"reworks: {label!r} does not resolve to a unit, tower or spell")
        return problems

    def table_note(self, table: Any) -> str | None:
        """A note when the class table is not the one the registry was written against."""
        digest = getattr(table, "digest", None)
        if self.classes_digest and digest and digest != self.classes_digest:
            return (f"classes.json digest {digest[:12]} differs from the registry's "
                    f"{self.classes_digest[:12]}: entries are checked by name; update classes_digest "
                    "once they pass")
        return None

    def in_base(self, base: L.LabelSet) -> list[str]:
        """Entries the base label set already has: they add no class."""
        return [e.label for e in self.entries if e.label in base]

    def extend(self, base: L.LabelSet, *, require_pin: bool = False) -> L.LabelSet:
        """The base label set with every entry appended, in registry order. The base's
        yolo_names() stays a prefix (checked). RegistryError for a KataCR-mode base, a base the
        registry was not pinned to, or a base / train_as target that is no label of the result.
        require_pin: also refuse while base.digest is not pinned in the registry; whatever
        writes class indices (N11 emit, N13 mix) passes it, so a base from another
        prepare_yolo_dataset run cannot move the appended indices unnoticed."""
        problems = []
        if base.mode != L.MODE_VISION:
            problems.append(f"the base label set is in {base.mode!r} mode; the registry extends a "
                            f"{L.MODE_VISION!r} label set (prepare_yolo_dataset --labels vision)")
        pin = self.base_pin.get("digest")
        if require_pin and not pin:
            problems.append(f"the registry's base.digest is not pinned yet: run `labels --base "
                            f"tmp\\yolo\\detector_labels.json` and pin the digest it prints ({base.digest} for "
                            "this base) in data/vision/detector_extra_labels.json before writing class indices")
        if pin and base.digest != pin:
            problems.append(f"the base label set has digest {base.digest}, the registry is pinned to "
                            f"{pin}: appending to another base would move every new index")
        if problems:
            raise RegistryError(problems)
        out = L.LabelSet(list(base.labels), base.joint_side, dict(base.status),
                         {k: list(v) for k, v in base.sources.items()}, base.mode, base.classes_digest)
        for e in self.entries:
            if e.label not in out:
                out.add(e.label, L.STATUS_EXTRA)
        for e in self.entries:
            for what, target in (("base", e.base), ("train_as", None if e.train_as == TRAIN_SELF else e.train_as)):
                if target is not None and target not in out:
                    problems.append(f"{e.label}: its {what} {target!r} is no label of the extended set")
        problems += check_prefix(base.yolo_names(), out.yolo_names())
        if problems:
            raise RegistryError(problems)
        return out

    # ------------------------------------------------------------ training
    def train_label(self, label: str, tracks: int, table: Any = None) -> str:
        """The label a box of `label` is written as, given its number of labelled training
        tracks: a form entry trains as its base while it has fewer than thin_tracks, unless
        train_as pins it; every other label trains as itself. With the class table, a hero or
        evolution form that is not in the registry yet (labelled from a '*' legend entry)
        trains as its class's base form, as a thin form does."""
        e = self._by_label.get(str(label))
        if e is None and table is not None:
            lab = L.resolve_label(table, str(label))
            if lab.type == L.TYPE_UNIT and lab.class_id is not None:
                vc = table.classes[lab.class_id]
                if lab.form != vc.base_form and table.detector_key(lab.class_id, lab.form) == str(label):
                    return table.detector_key(lab.class_id, vc.base_form) or str(label)
            return str(label)
        if e is None or e.train_as == TRAIN_SELF:
            return str(label)
        if e.train_as is not None:
            return e.train_as
        if e.base is not None and int(tracks) < self.thin_tracks:
            return e.base
        return e.label

    def train_index(self, labels: L.LabelSet, label: str, side: int | None, tracks: int, table: Any = None) -> int:
        """The YOLO class index a box of `label` on `side` is written with."""
        return labels.yolo_index(self.train_label(label, tracks, table), side)

    def valid_from(self, label: str) -> tuple[int, ...] | None:
        """The first client version whose look `label` learns (entry or rework), or None."""
        versions = []
        e = self.get(label)
        if e is not None and e.valid_from_client:
            versions.append(parse_version(e.valid_from_client))
        if str(label) in self.reworks:
            versions.append(parse_version(self.reworks[str(label)]))
        return max(versions) if versions else None

    def valid_for(self, label: str, client: str | None) -> bool:
        """Whether a session recorded on `client` may teach `label`. A label without a start
        version takes every session; with one, a session of unknown client is left out."""
        start = self.valid_from(label)
        if start is None:
            return True
        if client is None:
            return False
        return parse_version(client) >= start


def check_prefix(old: Sequence[str], new: Sequence[str]) -> list[str]:
    """Problems if `old` (YOLO names of an earlier label set) is not a prefix of `new`."""
    old, new = list(old), list(new)
    for i, (a, b) in enumerate(zip(old, new)):
        if a != b:
            return [f"YOLO class {i} was {a!r} and is now {b!r}: class indices would move "
                    "(the registry is append-only)"]
    if len(new) < len(old):
        return [f"the new label set has {len(new)} YOLO classes, fewer than the earlier {len(old)}: "
                f"{old[len(new)]!r} and later were removed (the registry is append-only)"]
    return []


def label_set_from_doc(doc: Mapping[str, Any], where: str = "label set") -> L.LabelSet:
    """A detector_labels.json document as a LabelSet; its recorded digest must match."""
    if not isinstance(doc, Mapping) or doc.get("schema") != L.SCHEMA:
        raise RegistryError(f"{where} is not a {L.SCHEMA} document")
    labels = L.LabelSet.from_dict(doc)
    if doc.get("digest") and doc["digest"] != labels.digest:
        raise RegistryError(f"{where}: its digest {doc['digest']} does not match its labels "
                            f"({labels.digest}); was it edited by hand?")
    return labels


def load_label_set(path: str | os.PathLike) -> L.LabelSet:
    p = Path(path)
    try:
        doc = _read_json(p)
    except FileNotFoundError:
        raise RegistryError(f"no label set at {p} (scripts/prepare_yolo_dataset.py writes "
                            "tmp/yolo/detector_labels.json)") from None
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{p} is not valid JSON: {exc}") from None
    return label_set_from_doc(doc, str(p))


def read_yolo_names(path: str | os.PathLike) -> list[str]:
    """YOLO class names from a detector_labels.json (its yolo_names, else its labels) or a JSON
    list of names."""
    doc = _read_json(path)
    if isinstance(doc, list) and all(isinstance(x, str) for x in doc):
        return list(doc)
    if isinstance(doc, Mapping) and isinstance(doc.get("yolo_names"), list):
        return [str(x) for x in doc["yolo_names"]]
    if isinstance(doc, Mapping) and isinstance(doc.get("labels"), list):
        return L.LabelSet.from_dict(doc).yolo_names()
    raise RegistryError(f"{path}: expected a detector_labels.json or a JSON list of YOLO names")


def crosswalk_base(table: Any) -> L.LabelSet:
    """The label set prepare_yolo_dataset builds from classes.json's own KataCR names: a
    preview of tonight's (the real one follows the dataset yaml's names)."""
    labels, _ = L.build_label_set(table, list(table.doc.get("katacr_names") or []))
    return labels


def extended_doc(base: L.LabelSet, registry: Registry, extended: L.LabelSet | None = None) -> dict[str, Any]:
    """The extended label set as a detector_labels.json document, with where it came from: the
    base (count, YOLO classes, digest) and the registry (entries, digest, form fallbacks)."""
    ext = registry.extend(base) if extended is None else extended
    doc = ext.to_dict()
    doc["base"] = {"labels": len(base), "yolo_classes": len(base.yolo_names()), "digest": base.digest,
                   "classes_digest": base.classes_digest}
    doc["registry"] = {
        "path": shown_path(registry.path) if registry.path else None,
        "labels": len(registry), "digest": registry.digest, "thin_tracks": registry.thin_tracks,
        "forms": {e.label: {"base": e.base, "train_as": e.train_as} for e in registry
                  if e.base is not None or e.train_as is not None},
        "valid_from_client": {e.label: e.valid_from_client for e in registry if e.valid_from_client},
        "reworks": dict(registry.reworks)}
    return doc


# ---------------------------------------------------------------------------
# event cards and the deck check (N08)
# ---------------------------------------------------------------------------

def load_event_cards(path: str | os.PathLike | None = None) -> dict[str, dict[str, Any]]:
    """data/vision/event_cards.json: catalogue name -> {"kind", "note", "use"?}."""
    p = Path(path) if path is not None else EVENT_CARDS_PATH
    doc = _read_json(p)
    if not isinstance(doc, Mapping) or doc.get("schema") != EVENT_CARDS_SCHEMA:
        raise ValueError(f"{p} is not a {EVENT_CARDS_SCHEMA} document")
    cards = doc.get("cards")
    if not isinstance(cards, Mapping) or not all(isinstance(v, Mapping) for v in cards.values()):
        raise ValueError(f"{p}: cards must be an object of name -> {{kind, note}}")
    return {str(k): dict(v) for k, v in cards.items()}


def load_key_map() -> tuple[dict[str, str], str | None]:
    """RoyaleAPI key -> catalogue name (tools/rs_cards.py load(), extras always merged), and a
    warning when it does not load (then only catalogue names are accepted)."""
    try:
        import rs_cards
        return rs_cards.load(extras=True), None
    except Exception as exc:                     # a broken card map must not block an ingest
        return {}, (f"the RoyaleAPI key map did not load ({type(exc).__name__}: {exc}); "
                    "only catalogue names are accepted")


class DeckError(ValueError):
    def __init__(self, problems: str | Iterable[str]):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class DeckCard:
    """One checked deck slot: what was typed, the catalogue card and the slot's form."""
    given: str
    card: str
    form: int = FORM_BASE
    kind: str = ""
    elixir: int | None = None

    @property
    def form_name(self) -> str:
        return FORM_NAMES[self.form]

    @property
    def champion(self) -> bool:
        return self.card in CHAMPIONS

    def to_dict(self) -> dict[str, Any]:
        return {"given": self.given, "card": self.card, "form": self.form, "form_name": self.form_name,
                "kind": self.kind, "elixir": self.elixir}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DeckCard":
        return cls(str(d.get("given", d["card"])), str(d["card"]), int(d.get("form", FORM_BASE)),
                   str(d.get("kind") or ""), None if d.get("elixir") is None else int(d["elixir"]))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def split_form_suffix(token: str) -> tuple[str, int]:
    """"cannon-ev1" -> ("cannon", 1), "ronin-hero" -> ("ronin", 2), "MinionGiant" -> (it, 0)."""
    low = token.lower()
    for suffix, form in KEY_SUFFIXES:
        if low.endswith(suffix) and len(token) > len(suffix):
            return token[:-len(suffix)], form
    return token, FORM_BASE


class CardResolver:
    """Deck tokens -> DeckCard: RoyaleAPI keys (key_map) first, then catalogue names of the
    class table, then either up to case and punctuation ("minion-giant"); -ev1 / -hero give
    the form. Event-only and internal cards are refused."""

    def __init__(self, table: Any, key_map: Mapping[str, str], event_cards: Mapping[str, Any]):
        self.table = table
        self.keys = dict(key_map)
        self.events = dict(event_cards)
        self._by_norm: dict[str, set[str]] = {}
        for key, name in self.keys.items():
            self._by_norm.setdefault(_norm(key), set()).add(name)
        for name in table.cards:
            self._by_norm.setdefault(_norm(name), set()).add(name)

    def _event_problem(self, given: str, name: str) -> str:
        info = self.events.get(name) or {}
        text = f"{given!r} is {name}, which no friendly deck can hold ({info.get('kind', 'event')} card"
        text += f": {info['note']})" if info.get("note") else ")"
        if info.get("use"):
            text += f"; give {info['use']} instead"
        return text

    def resolve(self, token: str) -> tuple[DeckCard | None, str | None]:
        given = str(token).strip()
        if not given:
            return None, "an empty card name"
        base, form = split_form_suffix(given)
        if base in self.events:
            return None, self._event_problem(given, base)
        if base in self.keys:
            name = self.keys[base]
        elif base in self.table.cards:
            name = base
        else:
            hits = self._by_norm.get(_norm(base), set())
            if len(hits) > 1:
                return None, (f"{given!r} could be any of {', '.join(sorted(hits))}: give the RoyaleAPI key "
                              "or the exact catalogue name")
            if not hits:
                close = difflib.get_close_matches(base, sorted(set(self.keys) | set(self.table.cards)), n=3)
                hint = f" (did you mean {', '.join(close)}?)" if close else ""
                return None, (f"unknown card {given!r}{hint}: give a RoyaleAPI key such as hog-rider, "
                              "cannon-ev1 or ronin-hero, or a catalogue name such as MinionGiant")
            name = next(iter(hits))
        if name in self.events:
            return None, self._event_problem(given, name)
        info = self.table.cards.get(name)
        if info is None:
            return None, (f"{given!r} is {name}, which classes.json does not have: a card newer than "
                          "the class table cannot be labelled until classes.json is regenerated (plan 7)")
        forms = [int(f) for f in info.get("forms") or [FORM_BASE]]
        if form not in forms:
            return None, (f"{given!r}: classes.json has no {FORM_NAMES[form]} form of {name} "
                          f"(it has {', '.join(FORM_NAMES[f] for f in forms)})")
        elixir = info.get("elixir")
        return DeckCard(given, name, form, str(info.get("kind") or ""),
                        None if elixir is None else int(elixir)), None


def parse_deck(deck: str | Sequence[str], resolver: CardResolver) -> list[DeckCard]:
    """A deck string ("a,b,...") or list -> 8 DeckCards. DeckError lists every problem."""
    tokens = deck.split(",") if isinstance(deck, str) else [str(t) for t in deck]
    tokens = [t.strip() for t in tokens if t.strip()]
    cards, problems = [], []
    for token in tokens:
        card, problem = resolver.resolve(token)
        if problem:
            problems.append(problem)
        else:
            cards.append(card)
    twice = sorted(name for name, n in Counter(c.card for c in cards).items() if n > 1)
    if twice:
        problems.append(f"a card appears twice: {', '.join(twice)}")
    if len(tokens) != DECK_SIZE:
        problems.append(f"a deck has {DECK_SIZE} cards; {len(tokens)} given")
    if problems:
        raise DeckError(problems)
    return cards


def slot_counts(deck: Sequence[DeckCard]) -> dict[str, int]:
    return {"evo": sum(c.form == FORM_EVO for c in deck), "hero": sum(c.form == FORM_HERO for c in deck),
            "champion": sum(c.champion for c in deck)}


def slot_warnings(deck: Sequence[DeckCard]) -> list[str]:
    """The slot rules seen in 1,467 ladder decks (plan 1.10); warnings, since they are observed."""
    n = slot_counts(deck)
    out = []
    if n["evo"] > 2:
        out.append(f"{n['evo']} evolution slots; ladder decks show at most 2")
    if n["hero"] + n["champion"] > 2:
        out.append(f"{n['hero']} hero and {n['champion']} champion slots; ladder decks show at most 2 together")
    if sum(n.values()) > 3:
        out.append(f"{sum(n.values())} evolution, hero and champion slots; ladder decks show at most 3")
    return out


def deck_label_gaps(deck: Sequence[DeckCard], table: Any, labels: Iterable[str]) -> list[str]:
    """Deck cards whose deployed unit (the class the card summons, in the slot's forms: an
    evolution slot shows both) or spell has no detector label in `labels`: their plays cannot
    be named until the registry has it. Units a card only spawns (MusketeerTurret, BalloonBomb)
    are left out: the summoned unit decides the card, and classes.json does not say which form
    spawns them. A spell's units (HealSpirit, the Spirit Empress forms) count as its look.
    A missing evo or hero label whose base form is labelled is only a form gap: label such
    tracks in their true form all the same (the legend offers it, marked '*'), and append the
    form to the registry before training; it trains as its base until it has thin_tracks
    labelled tracks (Registry.train_label), which still names the card (plan 1.8, 2.3). A
    form labelled as its base never gets those tracks."""
    have = set(labels)
    out = []
    for c in deck:
        info = table.cards.get(c.card) or {}
        spell = info.get("kind") == "spell"
        forms = (FORM_BASE, FORM_EVO) if c.form == FORM_EVO else (c.form,)
        missing, form_only = [], set()
        for cls in info.get("classes") or ():
            vc = table.get(cls)
            if vc is None or (not spell and (c.card, "summon") not in vc.cards):
                continue
            for form in forms:
                f = form if form in vc.forms else vc.base_form
                key = table.detector_key(vc.id, f)
                if key and key not in have and key not in missing:
                    missing.append(key)
                    if f != vc.base_form and table.detector_key(vc.id, vc.base_form) in have:
                        form_only.add(key)
        if spell:
            if c.card in have or set(info.get("katacr") or ()) & have:
                missing = []                    # the spell itself is labelled
            elif not info.get("classes"):
                missing.append(c.card)
        if not missing:
            continue
        names = ", ".join(missing)
        if set(missing) <= form_only:
            out.append(f"{c.given}: no detector label for {names} yet: label those tracks as {names} all the "
                       "same (the legend marks it '*'), and append it to data/vision/detector_extra_labels.json "
                       "before training; it trains as its base until it has enough tracks")
        else:
            out.append(f"{c.given}: no detector label for {names}; append it to "
                       "data/vision/detector_extra_labels.json before labelling, or those tracks can "
                       "only be masked")
    return out


# ---------------------------------------------------------------------------
# sessions (N08)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionPaths:
    """Where one session lives: its committed labels and its local store."""
    sid: str
    labels_dir: Path            # data/vision/owner_labels/<sid>
    store_dir: Path             # reports/vision/autolabel/<sid>

    @property
    def session_file(self) -> Path:
        return self.labels_dir / SESSION_FILE

    @property
    def tracks_file(self) -> Path:
        return self.labels_dir / TRACKS_FILE

    @property
    def recording(self) -> Path:
        """A record.py session: open_recording / VisionLogReader read it."""
        return self.store_dir / RECORDING_DIR

    @property
    def frames(self) -> Path:
        return self.recording / FRAMES_FILE

    @property
    def images(self) -> Path:
        return self.recording / IMAGES_DIR

    @property
    def scan_dir(self) -> Path:
        return self.store_dir / SCAN_DIR

    @property
    def sheet_dir(self) -> Path:
        return self.store_dir / SHEET_DIR

    @property
    def sheet_file(self) -> Path:
        return self.sheet_dir / SHEET_FILE


def session_paths(sid: str, labels_root: str | os.PathLike | None = None,
                  store: str | os.PathLike | None = None) -> SessionPaths:
    sid = check_sid(sid)
    return SessionPaths(sid, Path(labels_root or LABELS_ROOT) / sid, Path(store or STORE_ROOT) / sid)


def load_session(sid: str, labels_root: str | os.PathLike | None = None) -> dict[str, Any]:
    p = session_paths(sid, labels_root).session_file
    doc = _read_json(p)
    if not isinstance(doc, Mapping) or doc.get("schema") != SESSION_SCHEMA:
        raise ValueError(f"{p} is not a {SESSION_SCHEMA} document")
    return dict(doc)


def session_ids(labels_root: str | os.PathLike | None = None) -> list[str]:
    root = Path(labels_root or LABELS_ROOT)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and (d / SESSION_FILE).is_file())


def session_deck(doc: Mapping[str, Any]) -> list[DeckCard]:
    return [DeckCard.from_dict(d) for d in doc.get("deck") or ()]


def deck_cards(doc: Mapping[str, Any]) -> list[str]:
    """The session's deck as catalogue names (what ClassTable.deck_labels takes)."""
    return [c.card for c in session_deck(doc)]


def auto_role(sid: str, capture: str = CAPTURE_HARVEST) -> tuple[str, str]:
    """(role, why) from a session id date_DECK_N (plan 4.4, 4.5, 6)."""
    if capture in GOLD_ONLY_CAPTURES:
        return ROLE_GOLD, f"{capture} captures are gold only"
    m = HARVEST_SID_RE.match(str(sid))
    if not m:
        raise ValueError(f"--role auto reads a session id like 20261101_A_1 (date_deck_friendly), "
                         f"not {sid!r}: pass --role train, gold or dev")
    deck, n = m["deck"].upper(), int(m["n"])
    if n == GOLD_FRIENDLY:
        return ROLE_GOLD, f"friendly {n} of deck {deck} is gold (held out)"
    if (deck, n) in DEV_FRIENDLIES:
        return ROLE_DEV, f"friendly {n} of deck {deck} is dev (early stopping only)"
    return ROLE_TRAIN, f"friendly {n} of deck {deck} trains"


def choose_role(role: str, sid: str, capture: str = CAPTURE_HARVEST) -> tuple[str, str]:
    if capture not in CAPTURES:
        raise ValueError(f"--capture must be one of {', '.join(CAPTURES)}, not {capture!r}")
    if role == ROLE_AUTO:
        return auto_role(sid, capture)
    if role not in ROLES:
        raise ValueError(f"--role must be auto, {', '.join(ROLES)}; not {role!r}")
    if capture in GOLD_ONLY_CAPTURES and role != ROLE_GOLD:
        raise ValueError(f"a {capture} capture is gold only: it is never trained on (plan 6)")
    return role, "given"


def find_recording(path: str | os.PathLike) -> tuple[Path, str]:
    """(session folder, default session id) for a recorded session, its frames.jsonl, or a
    root holding exactly one session (the id is then the root's name)."""
    p = Path(path)
    if p.is_file() and p.name == FRAMES_FILE:
        return p.parent, p.parent.name
    if (p / FRAMES_FILE).is_file():
        return p, p.name
    if not p.is_dir():
        raise FileNotFoundError(f"no recording at {p}")
    sessions = sorted(d for d in p.iterdir() if d.is_dir() and (d / FRAMES_FILE).is_file())
    if not sessions:
        raise FileNotFoundError(f"no recorded session in {p} (need <session>/{FRAMES_FILE})")
    if len(sessions) > 1:
        raise FileExistsError(f"{p} holds {len(sessions)} sessions ({', '.join(d.name for d in sessions)}): "
                              "pass the session folder and --sid")
    return sessions[0], p.name


def image_files(images_dir: Path) -> list[Path]:
    """The recorder's image files in a session's images/ (record.py's names only)."""
    if not images_dir.is_dir():
        return []
    return sorted(f for f in images_dir.iterdir() if IMAGE_NAME_RE.match(f.name) and f.is_file())


@dataclass
class FramesInfo:
    """What a frames.jsonl holds, read with the standard library (as harvest.open_recording)."""
    header: dict[str, Any] | None = None
    frames: int = 0
    listed: list[str] = field(default_factory=list)     # image file names, in frame order
    bad_lines: int = 0
    t_first: float | None = None
    t_last: float | None = None
    sha1: str = ""
    size: int = 0

    @property
    def seconds(self) -> float | None:
        if self.t_first is None or self.t_last is None:
            return None
        return round(self.t_last - self.t_first, 3)


def _time_of(rec: Mapping[str, Any]) -> float | None:
    frame = rec.get("frame") if isinstance(rec.get("frame"), Mapping) else {}
    for value in (frame.get("t_capture"), rec.get("t_wall")):
        try:
            t = float(value)
        except (TypeError, ValueError):
            continue
        if t == t and abs(t) != float("inf"):
            return t
    return None


def read_frames(path: Path) -> FramesInfo:
    info = FramesInfo()
    sha = hashlib.sha1()
    with open(path, "rb") as fh:
        for raw in fh:
            sha.update(raw)
            info.size += len(raw)
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                info.bad_lines += 1                # a line cut short by a crash
                continue
            if not isinstance(rec, dict):
                info.bad_lines += 1
                continue
            kind = rec.get("kind")
            if kind == "session" and info.header is None:
                info.header = rec
            elif kind == "frame":
                info.frames += 1
                t = _time_of(rec)
                if t is not None:
                    info.t_first = t if info.t_first is None else min(info.t_first, t)
                    info.t_last = t if info.t_last is None else max(info.t_last, t)
                image = rec.get("image")
                if isinstance(image, str) and image:
                    info.listed.append(Path(image).name)
    info.sha1 = sha.hexdigest()
    return info


def _link(src: Path, dst: Path, allow_copy: bool) -> str:
    """Hard-link src to dst ("linked"); "copied" when linking fails and copies are allowed."""
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        if not allow_copy:
            raise
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return "copied"


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


class IngestError(ValueError):
    def __init__(self, problems: str | Iterable[str], code: int = EXIT_INPUT):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        self.code = code
        super().__init__("; ".join(self.problems))


@dataclass
class IngestResult:
    sid: str
    paths: SessionPaths
    session: dict[str, Any]
    created: bool
    counts: Counter
    warnings: list[str]


_COMPARED = ("role", "capture", "client")


def _changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, list[Any]]:
    out = {k: [old.get(k), new.get(k)] for k in _COMPARED if old.get(k) != new.get(k)}
    old_deck = [(c.get("card"), c.get("form")) for c in old.get("deck") or ()]
    new_deck = [(c.get("card"), c.get("form")) for c in new.get("deck") or ()]
    if sorted(old_deck, key=str) != sorted(new_deck, key=str):
        out["deck"] = [[c.get("given") for c in old.get("deck") or ()],
                       [c.get("given") for c in new.get("deck") or ()]]
    return out


def ingest(recording: str | os.PathLike, deck: str | Sequence[str], *, client: str,
           sid: str | None = None, role: str = ROLE_AUTO, capture: str = CAPTURE_HARVEST,
           labels_root: str | os.PathLike | None = None, store: str | os.PathLike | None = None,
           table: Any = None, registry: Registry | None = None,
           key_map: Mapping[str, str] | None = None, event_cards: Mapping[str, Any] | None = None,
           force: bool = False, allow_copy: bool = True,
           clock: Any = time.time) -> IngestResult:
    """Ingest one recorded friendly (see the module docstring). Everything is checked before
    anything is written; IngestError (code 2 bad input, 1 a refused change) otherwise."""
    warnings: list[str] = []
    try:
        session_dir, default_sid = find_recording(recording)
    except (FileNotFoundError, FileExistsError) as exc:
        raise IngestError(str(exc)) from None
    client = str(client or "").strip()
    if not client:
        raise IngestError("--client is needed (the game client version, like 16.635.1)")
    try:
        sid = check_sid(sid or default_sid)
        role, rule = choose_role(role, sid, capture)
        parse_version(client)
    except ValueError as exc:
        raise IngestError(str(exc)) from None
    table = load_table() if table is None else table
    events = load_event_cards() if event_cards is None else event_cards
    if key_map is None:
        key_map, note = load_key_map()
        if note:
            warnings.append(note)
    if registry is None:
        try:
            registry = Registry.load()
        except RegistryError as exc:
            warnings.append(f"label registry not read ({exc}); label gaps not checked")
    try:
        cards = parse_deck(deck, CardResolver(table, key_map, events))
    except DeckError as exc:
        raise IngestError(exc.problems) from None

    paths = session_paths(sid, labels_root, store)
    if _inside(paths.store_dir, session_dir) or _inside(session_dir, paths.store_dir):
        raise IngestError(f"the store {paths.store_dir} and the recording {session_dir} overlap: "
                          "pass a --store outside the recording")

    new = {"role": role, "capture": capture, "client": client, "deck": [c.to_dict() for c in cards]}
    old = None
    if paths.session_file.is_file():
        try:
            old = load_session(sid, labels_root)
        except (ValueError, OSError) as exc:
            raise IngestError(f"{paths.session_file} cannot be read ({exc}); move it away first",
                              EXIT_CHECK) from None
    history = list((old or {}).get("history") or [])
    if old is not None:
        was = (old.get("recording") or {}).get("name")
        problems = []
        if was and was != session_dir.name:
            problems.append(f"session {sid} was ingested from recording {was}, not {session_dir.name}")
        changes = _changes(old, new)
        for key, (a, b) in changes.items():
            problems.append(f"session {sid} has {key} {a!r}; this ingest says {b!r}")
        if problems and not force:
            raise IngestError(problems + ["pass --force to change it"], EXIT_CHECK)
        if changes.get("role") == [ROLE_TRAIN, ROLE_GOLD] or changes.get("role") == [ROLE_DEV, ROLE_GOLD]:
            warnings.append("this session may already be in a training mix: build the mix again before "
                            "trusting gold numbers")
        if problems:
            history.append({"at": datetime.fromtimestamp(clock()).isoformat(timespec="seconds"),
                            "changed": changes, "recording": [was, session_dir.name]})

    src_images = session_dir / IMAGES_DIR
    files = image_files(src_images)
    conflicts = []
    for f in files:
        dst = paths.images / f.name
        try:
            if dst.exists() and not _same_file(f, dst) and dst.stat().st_size != f.stat().st_size:
                conflicts.append(f.name)
        except OSError:
            continue                           # pruned meanwhile: counted when linking
    if conflicts:
        raise IngestError(f"{len(conflicts)} image(s) in {paths.images} differ from the recording's "
                          f"(first {conflicts[0]}): another recording under session id {sid}? "
                          "Pass another --sid or move that store folder away", EXIT_CHECK)

    # Writing starts here: the images first, the recorder may still prune.
    counts: Counter = Counter()
    paths.images.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = paths.images / f.name
        if dst.exists():
            counts["present"] += 1
            continue
        try:
            counts[_link(f, dst, allow_copy)] += 1
        except FileNotFoundError:
            counts["vanished"] += 1            # pruned between the listing and the link
    src_frames = session_dir / FRAMES_FILE
    tmp = paths.frames.with_name(FRAMES_FILE + ".part")
    shutil.copy2(src_frames, tmp)
    os.replace(tmp, paths.frames)
    info = read_frames(paths.frames)
    stored = {f.name for f in image_files(paths.images)}
    listed = set(info.listed)
    missing = len(listed - stored)
    unlisted = len(stored - listed)

    try:
        age = clock() - src_frames.stat().st_mtime
    except OSError:
        age = None
    if age is not None and age < RECENT_WRITE_S:
        warnings.append(f"{FRAMES_FILE} was written {age:.0f} s ago: if the recording is still running, "
                        "ingest again when it stops")
    if missing:
        warnings.append(f"{missing} image(s) listed in {FRAMES_FILE} were gone before ingest "
                        "(pruned by the recorder's disk budget?)")
    if counts["copied"]:
        warnings.append(f"{counts['copied']} image(s) were copied, not hard-linked (the store is on "
                        "another drive?): they take disk space twice")
    if counts["vanished"]:
        warnings.append(f"{counts['vanished']} image(s) vanished while linking (the recorder pruned them)")
    if info.bad_lines:
        warnings.append(f"{info.bad_lines} unreadable line(s) in {FRAMES_FILE}")
    if not stored:
        warnings.append("the recording has no images: record with --image-every")
    warnings += slot_warnings(cards)
    warnings += deck_label_gaps(cards, table, known_labels(table, registry))
    warnings = [scrub_paths(w) for w in warnings]          # session.json is committed: no machine paths

    now = datetime.fromtimestamp(clock()).isoformat(timespec="seconds")
    header = info.header or {}
    doc: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "sid": sid,
        "role": role,
        "role_rule": rule,
        "capture": capture,
        "client": client,
        "deck": new["deck"],
        "slots": slot_counts(cards),
        "recording": {
            "name": session_dir.name, "source": shown_path(session_dir),
            "created": header.get("created"), "image_every": header.get("image_every"),
            "image_format": header.get("image_format"), "image_width": header.get("image_width"),
            "frames": info.frames, "seconds": info.seconds, "images_listed": len(listed),
            "images_missing": missing, "images_unlisted": unlisted, "bad_lines": info.bad_lines,
            "frames_sha1": info.sha1, "frames_bytes": info.size},
        "store": {"path": shown_path(paths.store_dir), "images": len(stored),
                  "linked": counts["linked"], "copied": counts["copied"], "present": counts["present"]},
        "classes_digest": getattr(table, "digest", None),
        "registry_digest": registry.digest if registry else None,
        "ingested": (old or {}).get("ingested") or now,
        "updated": now,
        "history": history,
        "warnings": warnings,
        "tool": "tools/vision_autolabel.py ingest",
    }
    write_json(paths.session_file, doc)
    return IngestResult(sid, paths, doc, old is None, counts, warnings)


# ---------------------------------------------------------------------------
# status (N08)
# ---------------------------------------------------------------------------

def _count_images(images: Path) -> int:
    return len(image_files(images))


def stages_done(paths: SessionPaths) -> list[str]:
    roots = {"labels": paths.labels_dir, "store": paths.store_dir}
    return [name for name, where, rel in STAGES if (roots[where] / rel).exists()]


def labelled_tracks(sids: Iterable[str], labels_root: str | os.PathLike | None = None) -> Counter:
    """Labelled tracks per label over the sessions' tracks.json files (see TRACKS_FILE): one
    per unit, not per entry. The parts of one harvest track with one label count once, and so
    do harvest tracks of one live track (read.uid: a harvest track broken in two); an entry
    with neither counts alone."""
    out: Counter = Counter()
    for sid in sids:
        p = session_paths(sid, labels_root).tracks_file
        try:
            doc = _read_json(p)
        except (OSError, ValueError):
            continue
        seen: set[tuple[str, str]] = set()
        for i, t in enumerate((doc.get("tracks") if isinstance(doc, Mapping) else None) or ()):
            if not (isinstance(t, Mapping) and isinstance(t.get("label"), str)):
                continue
            read = t.get("read") if isinstance(t.get("read"), Mapping) else {}
            unit = (f"u{read['uid']}" if read.get("uid") else f"h{t['h']}" if t.get("h") is not None else f"#{i}")
            if (t["label"], unit) not in seen:
                seen.add((t["label"], unit))
                out[t["label"]] += 1
    return out


def _slot_shows(entry: ExtraLabel, slot: DeckCard) -> bool:
    """Whether a deck slot can put `entry` on screen: any slot of the card shows its base-form
    labels (an evolution card shows both forms in a match), the evo or hero slot its form."""
    if entry.card != slot.card:
        return False
    return entry.form == FORM_BASE or entry.form == slot.form


def coverage(registry: Registry, sessions: Mapping[str, Mapping[str, Any]],
             tracks: Mapping[str, int] | None = None) -> list[dict[str, Any]]:
    """Per registry label: the sessions per role whose declared deck can show it, and its
    labelled training tracks."""
    rows = []
    for e in registry:
        n = Counter()
        for doc in sessions.values():
            if any(_slot_shows(e, c) for c in session_deck(doc)):
                n[str(doc.get("role"))] += 1
        rows.append({"label": e.label, "card": e.card, "form": e.form_name,
                     ROLE_TRAIN: n[ROLE_TRAIN], ROLE_GOLD: n[ROLE_GOLD], ROLE_DEV: n[ROLE_DEV],
                     "tracks": None if tracks is None else int(tracks.get(e.label, 0))})
    return rows


def status(labels_root: str | os.PathLike | None = None, store: str | os.PathLike | None = None,
           sids: Sequence[str] | None = None, registry: Registry | None = None) -> dict[str, Any]:
    """The sessions, their stages and images, and the registry's coverage, as one document."""
    all_sids = session_ids(labels_root)
    chosen = list(sids) if sids else all_sids
    sessions: dict[str, dict[str, Any]] = {}
    rows, problems = [], []
    for sid in chosen:
        try:
            doc = load_session(sid, labels_root)
        except (OSError, ValueError) as exc:
            problems.append(f"{sid}: {exc}")
            continue
        sessions[sid] = doc
        paths = session_paths(sid, labels_root, store)
        now = _count_images(paths.images)
        had = int((doc.get("store") or {}).get("images") or 0)
        notes = []
        if doc.get("role") == ROLE_GOLD and now < had:
            notes.append(f"{had - now} gold raw image(s) deleted: keep gold raws (plan 4.7)")
        elif now < had:
            notes.append(f"{had - now} raw image(s) deleted (fine once emitted)")
        if doc.get("role") == ROLE_GOLD or doc.get("capture") in GOLD_ONLY_CAPTURES:
            try:
                tdoc = _read_json(paths.tracks_file) if paths.tracks_file.is_file() else {}
            except (OSError, ValueError):
                tdoc = {}
            autos = [t for t in (tdoc.get("tracks") if isinstance(tdoc, Mapping) else None) or ()
                     if isinstance(t, Mapping) and t.get("source") == SOURCE_AUTO]
            if autos:
                notes.append(f"{len(autos)} auto-accepted label(s) in a gold session (stored while it was train "
                             f"or dev): check each, e.g. label {sid} --map \"t{autos[0].get('key')}=...\" --force")
        rows.append({"sid": sid, "role": doc.get("role"), "capture": doc.get("capture"),
                     "client": doc.get("client"), "images": now, "images_ingested": had,
                     "seconds": (doc.get("recording") or {}).get("seconds"),
                     "stages": stages_done(paths), "deck": [c.get("given") for c in doc.get("deck") or ()],
                     "notes": notes})
    roles = Counter(str(d.get("role")) for d in sessions.values())
    train = [sid for sid, d in sessions.items() if d.get("role") == ROLE_TRAIN]
    tracks = labelled_tracks(train, labels_root)
    cov = coverage(registry, sessions, tracks if tracks else None) if registry is not None else []
    return {"sessions": rows, "roles": {r: roles[r] for r in ROLES}, "problems": problems,
            "coverage": cov, "thin_tracks": registry.thin_tracks if registry else THIN_TRACKS,
            "labels_root": str(Path(labels_root or LABELS_ROOT))}


# ---------------------------------------------------------------------------
# the legend, the label map, sheet and label (N10)
# ---------------------------------------------------------------------------

CODE_EFFECT, CODE_OURS, CODE_UNSURE, CODE_SPLIT = "x", "o", "?", "s"
#: The four codes a map value may be instead of a label (plan 5).
CODES: dict[str, str] = {
    CODE_EFFECT: "an effect, not a unit",
    CODE_OURS: "ours (Yodle2's)",
    CODE_UNSURE: "unsure (masked in training)",
    CODE_SPLIT: "split: those tracks come back on the next sheet, one per tile",
}
SOURCE_OWNER, SOURCE_AUTO = "owner", "auto"
MAP_KEY_RE = re.compile(r"^(?:(?P<a>\d+)(?:-(?P<b>\d+))?|[tT](?P<h>\d+)(?:\.(?P<p>\d+))?)$")
KIND_CLUSTER, KIND_TRACK = "cluster", "track"


def _unit_harvest():
    """scripts/vision/unit_harvest.py (numpy): imported only by scan and sheet."""
    from scripts.vision import unit_harvest as UH
    return UH


def scan_identity(scan_dir: str | os.PathLike) -> dict[str, Any]:
    """unit_harvest.scan_identity with the standard library (label needs no numpy): the
    SHA-1 of the scan's scan.json and tracks.json, and when it was made."""
    out = Path(scan_dir)

    def sha(p: Path) -> str | None:
        try:
            return hashlib.sha1(p.read_bytes()).hexdigest()
        except OSError:
            return None

    created = None
    try:
        doc = json.loads((out / SCAN_JSON).read_text(encoding="utf-8"))
        created = doc.get("created") if isinstance(doc, Mapping) else None
    except (OSError, ValueError):
        pass
    return {"created": created, "scan_sha1": sha(out / SCAN_JSON), "tracks_sha1": sha(out / SCAN_TRACKS)}


def base_labels(table: Any, base: str | os.PathLike | None = BASE_LABELS) -> L.LabelSet:
    """Tonight's label set: the real one prepare_yolo_dataset wrote (`base`, by default
    tmp/yolo/detector_labels.json) when it is there and reads, else the preview built from
    classes.json's KataCR names (crosswalk_base; plan 8: the real one may hold more)."""
    if base is not None and Path(base).is_file():
        try:
            return load_label_set(base)
        except (RegistryError, OSError, ValueError):
            pass
    return crosswalk_base(table)


def known_labels(table: Any, registry: Registry | None = None,
                 base: str | os.PathLike | None = BASE_LABELS) -> set[str]:
    """Detector labels a training run can write: the base label set (base_labels: the real
    one when it is there; base=None: the preview) and the registry's appended labels."""
    return set(base_labels(table, base).labels) | set(registry.labels if registry is not None else ())


def spell_label(table: Any, card: str, known: Iterable[str] = ()) -> str:
    """The label a spell card trains as: its KataCR name when the label set has it (Fireball ->
    "fireball"), else the card name (the registry's Vines)."""
    have = set(known)
    for name in (table.cards.get(card) or {}).get("katacr") or ():
        if name in have:
            return str(name)
    return str(card)


def canonical_label(table: Any, name: str, known: Iterable[str] = (), *, form: int | None = None):
    """(label, vision_classes.Label) for any name classes.json reads (a class, a form's unit
    name, a KataCR name, a spell card, the same up to case and punctuation): units and towers
    as ClassTable.detector_key writes them (in `form` when given, else the name's own form;
    None when the class has no such form), spells as spell_label; None for anything else."""
    lab = L.resolve_label(table, str(name))
    if lab.type in (L.TYPE_UNIT, L.TYPE_TOWER) and lab.class_id is not None:
        return table.detector_key(lab.class_id, lab.form if form is None else int(form)), lab
    if lab.type == L.TYPE_SPELL and lab.card:
        return spell_label(table, lab.card, known), lab
    return None, lab


@dataclass(frozen=True)
class LegendEntry:
    """One numbered name of the sheet's legend: a detector label the declared deck can show."""
    n: int
    label: str
    card: str
    form: int = FORM_BASE
    type: str = L.TYPE_UNIT
    known: bool = True
    given: str = ""

    def line(self) -> str:
        form = "" if self.form == FORM_BASE else f" ({FORM_NAMES[self.form]})"
        return f"{self.n:>2} {self.label}{form}{'' if self.known else ' *'}"

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "label": self.label, "card": self.card, "form": self.form, "type": self.type,
                "known": self.known, "given": self.given}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LegendEntry":
        return cls(int(d["n"]), str(d["label"]), str(d.get("card", "")), int(d.get("form", FORM_BASE)),
                   str(d.get("type", L.TYPE_UNIT)), bool(d.get("known", True)), str(d.get("given", "")))


def build_legend(deck: Sequence[DeckCard], table: Any, known: Iterable[str] = ()) -> list[LegendEntry]:
    """The legend of a declared deck (plan 5): per card, in deck order, each of its classes
    (classes.json "classes": what it summons and spawns) in the base form and, for an evolution
    or hero slot, also in that form; a spell's own label after its classes, when the label set
    has it or the spell has no class (a spell whose look is its units, Heal or the Spirit
    Empress, is named by them). `known` marks the labels a training run can write already."""
    have = set(known)
    out: list[LegendEntry] = []
    seen: set[str] = set()

    def add(label: str | None, card: str, form: int, typ: str, given: str) -> None:
        if label and label not in seen:
            seen.add(label)
            out.append(LegendEntry(len(out), label, card, int(form), typ, label in have, given))

    for c in deck:
        info = table.cards.get(c.card) or {}
        for cls in info.get("classes") or ():
            vc = table.get(cls)
            if vc is None or vc.retired:
                continue
            forms = [vc.base_form]
            if c.form != FORM_BASE and c.form in vc.forms and c.form != vc.base_form:
                forms.append(c.form)
            for f in forms:
                add(table.detector_key(vc.id, f), c.card, f, L.TYPE_UNIT, c.given)
        if info.get("kind") == "spell":
            label = spell_label(table, c.card, have)
            if label in have or not info.get("classes"):
                add(label, c.card, FORM_BASE, L.TYPE_SPELL, c.given)
    return out


def legend_lines(legend: Sequence[LegendEntry]) -> list[str]:
    lines = [e.line() for e in legend]
    lines += [f" {k} {text}" for k, text in CODES.items()]
    if any(not e.known for e in legend):
        lines.append(" * no detector label yet: label it all the same, and append it to the registry before "
                     "training (a form trains as its base until it has enough tracks)")
    return lines


@dataclass(frozen=True)
class Decision:
    """What one map value says: a detector label, or one of CODES."""
    label: str | None = None
    code: str | None = None

    @property
    def split(self) -> bool:
        return self.code == CODE_SPLIT


def _card_label(value: str, legend: Sequence[LegendEntry], table: Any, known: Iterable[str],
                key_map: Mapping[str, str] | None) -> tuple[str | None, str | None]:
    """(label, problem) for a name labels.resolve_label does not read: a legend label up to
    case and punctuation ("vines"), or a card as a RoyaleAPI key or catalogue name, with -ev1 or
    -hero ("rune-giant", "baby-dragon-ev1"), when the card summons one class or is a spell
    with its own label. (None, None) when it is not a card either."""
    hits = [e.label for e in legend if _norm(e.label) == _norm(value)]
    if len(hits) == 1:
        return hits[0], None
    base, form = split_form_suffix(value)
    card = (key_map or {}).get(base)
    if card is None:
        names = [c for c in table.cards if _norm(c) == _norm(base)]
        card = names[0] if len(names) == 1 else None
    if card is None:
        return None, None
    info = table.cards.get(card) or {}
    classes = [vc for vc in (table.get(c) for c in info.get("classes") or ()) if vc is not None and not vc.retired]
    if info.get("kind") == "spell":
        label = spell_label(table, card, known)
        if label in set(known) or not classes:
            return label, None
        units = [table.detector_key(vc.id, vc.base_form) for vc in classes]
        return None, f"{value!r} is the spell {card}, seen as its units: name one of {', '.join(u for u in units if u)}"
    summons = [vc for vc in classes if (card, "summon") in vc.cards]
    if len(summons) != 1:
        units = [table.detector_key(vc.id, vc.base_form) for vc in (summons or classes)]
        return None, f"{value!r} is the card {card}, which shows several units: name one of {', '.join(u for u in units if u)}"
    vc = summons[0]
    key = table.detector_key(vc.id, form if form != FORM_BASE else vc.base_form)
    if key is None:
        return None, f"{value!r}: {vc.name} has no {FORM_NAMES[form]} form in classes.json"
    return key, None


def resolve_value(value: str, legend: Sequence[LegendEntry], table: Any, known: Iterable[str] = (), *,
                  off_deck: bool = False, event_cards: Mapping[str, Any] | None = None,
                  key_map: Mapping[str, str] | None = None) -> tuple[Decision | None, str | None]:
    """(decision, None) or (None, why not) for one map value: a code (x, o, ?, s), a legend
    number, or a name that is on the legend: one classes.json reads (labels.resolve_label), a
    legend label up to case and punctuation, or a card (RoyaleAPI key or catalogue name) that
    shows one unit (_card_label). A name off the legend is refused unless off_deck (and never
    an event-only card's)."""
    v = str(value).strip()
    if v.lower() in CODES:
        return Decision(code=v.lower()), None
    names = [e.label for e in legend]
    if v.isdigit():
        hit = next((e for e in legend if e.n == int(v)), None)
        if hit is None:
            return None, (f"{v} is not a legend number (0-{len(legend) - 1})" if legend else
                          f"{v} is not a legend number (the legend is empty)")
        return Decision(label=hit.label), None
    label, lab = canonical_label(table, v, known)
    if label is None and lab.type == L.TYPE_UNKNOWN:
        label, why = _card_label(v, legend, table, known, key_map)
        if why:
            return None, why
        if label is not None:
            label, lab = canonical_label(table, label, known)
    if lab.type == L.TYPE_TOWER:
        return None, f"{v!r} is a tower: towers are not labelled here"
    if label is None:
        close = difflib.get_close_matches(v, names + sorted(set(known) - set(names)), n=3)
        hint = f" (did you mean {', '.join(close)}?)" if close else ""
        if lab.type in (L.TYPE_UNIT,) and lab.class_id is not None:
            return None, f"{v!r}: class {table.classes[lab.class_id].name} has no such form in classes.json"
        return None, (f"unknown name {v!r}{hint}: give a legend number, a unit or spell of classes.json, "
                      f"or one of {', '.join(CODES)}")
    if label in names:
        return Decision(label=label), None
    if lab.type == L.TYPE_SPELL:
        units = [table.detector_key(cls) for cls in (table.cards.get(lab.card) or {}).get("classes") or ()]
        units = [u for u in units if u]
        if units and label not in set(known):
            return None, f"{v!r} is the spell {lab.card}, seen as its units: name one of {', '.join(units)}"
    if not off_deck:
        return None, (f"{v!r} ({label}) is not on the declared deck's legend: give a legend number, or pass "
                      "--off-deck if the deck was declared wrong (then ingest it again with --force)")
    card = lab.card if lab.type == L.TYPE_SPELL else table.classes[lab.class_id].primary_card
    events = load_event_cards() if event_cards is None else event_cards
    if card in events:
        return None, f"{v!r} is {card}, which no friendly deck holds ({events[card].get('kind', 'event')} card)"
    return Decision(label=label), None


def parse_map(text: str | None) -> tuple[list[tuple[str, Any, str]], list[str]]:
    """'0=3, 1=x, #2=goblinstein, 4-6=?, t17=5, t40.1=s' -> [(kind, id, value)], problems.
    Keys: a cluster id, a range of them (A-B), or a track key t<h>[.<part>]; '#' is allowed
    before an id, as harvest-label's maps (scripts/vision/harvest.parse_label_map) allow."""
    entries: list[tuple[str, Any, str]] = []
    problems: list[str] = []
    seen: set[tuple[str, Any]] = set()
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        key, value = key.strip().lstrip("#"), value.strip()
        m = MAP_KEY_RE.match(key)
        if not sep or not value or m is None:
            problems.append(f"{part!r} is not ID=VALUE (ID: a cluster, A-B, or t<track>)")
            continue
        if m["h"] is not None:
            targets = [(KIND_TRACK, m["h"] if m["p"] is None else f"{int(m['h'])}.{int(m['p'])}")]
        else:
            a = int(m["a"])
            b = a if m["b"] is None else int(m["b"])
            if b < a:
                problems.append(f"{part!r}: the range runs backwards")
                continue
            targets = [(KIND_CLUSTER, i) for i in range(a, b + 1)]
        for kind, ident in targets:
            if kind == KIND_TRACK:
                ident = str(int(ident)) if "." not in str(ident) else ident
            if (kind, ident) in seen:
                problems.append(f"{'cluster ' if kind == KIND_CLUSTER else 'track t'}{ident} is labelled twice")
                continue
            seen.add((kind, ident))
            entries.append((kind, ident, value))
    return entries, problems


class LabelError(ValueError):
    """A label map (or the session it is for) is refused: nothing is stored."""

    def __init__(self, problems: str | Iterable[str], code: int = EXIT_INPUT):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        self.code = code
        super().__init__("; ".join(self.problems))


class SheetError(LabelError):
    """The sheet cannot be made (no session, no scan, labels from another scan)."""


def read_tracks(paths: SessionPaths) -> dict[str, Any] | None:
    """data/vision/owner_labels/<sid>/tracks.json, or None when there is none."""
    if not paths.tracks_file.is_file():
        return None
    doc = _read_json(paths.tracks_file)
    if not isinstance(doc, Mapping) or doc.get("schema") != TRACKS_SCHEMA:
        raise LabelError(f"{paths.tracks_file} is not a {TRACKS_SCHEMA} document; move it away first", EXIT_CHECK)
    return dict(doc)


def read_sheet(paths: SessionPaths) -> dict[str, Any]:
    try:
        doc = _read_json(paths.sheet_file)
    except FileNotFoundError:
        raise LabelError(f"no sheet for {paths.sid} ({paths.sheet_file}): run `sheet {paths.sid}` first") from None
    except ValueError as exc:
        raise LabelError(f"{paths.sheet_file} is not valid JSON: {exc}") from None
    if not isinstance(doc, Mapping) or doc.get("schema") != SHEET_SCHEMA:
        raise LabelError(f"{paths.sheet_file} is not a {SHEET_SCHEMA} document")
    return dict(doc)


def suggest(part: Mapping[str, Any], legend: Sequence[LegendEntry], table: Any, known: Iterable[str] = ()) -> int | None:
    """The legend number to suggest for a part: its live read (class and form), else its
    harvest candidates by score, the first that is on the legend. None when the read cannot
    tell the form (read["form_open"]: an evolution or hero slot's class read in another form):
    the owner picks the form from the look (form_choice shows both numbers)."""
    by_label = {e.label: e.n for e in legend}
    read = part.get("read") or {}
    if read.get("form_open"):
        return None
    tries: list[tuple[str, int | None]] = []
    if read.get("live"):
        tries.append((str(read["live"]), int(read.get("form") or FORM_BASE)))
        tries.append((str(read["live"]), None))
    tries += [(str(c), None) for c, _s in part.get("cands") or ()]
    for name, form in tries:
        try:
            label, _ = canonical_label(table, name, known, form=form)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if label in by_label:
            return by_label[label]
    return None


def form_choice(part: Mapping[str, Any], legend: Sequence[LegendEntry], table: Any) -> list[int]:
    """The legend numbers of a form-open read's class (unit_harvest.DeckContext.form_open):
    its base form first, then the forms the deck's slot shows."""
    read = part.get("read") or {}
    if not read.get("form_open") or not read.get("live"):
        return []
    lab = L.resolve_label(table, str(read["live"]))
    if lab.type != L.TYPE_UNIT or lab.class_id is None:
        return []
    keys = set()
    for f in table.classes[lab.class_id].forms:
        key = table.detector_key(lab.class_id, f)
        if key:
            keys.add(key)
    return sorted((e.n for e in legend if e.label in keys), key=lambda n: (legend[n].form, n))


def _read_text(part: Mapping[str, Any]) -> tuple[str, str]:
    """(what the live chain read, the best harvest candidate) of a part, short."""
    r = part.get("read") or {}
    live = "-"
    if r.get("live"):
        live = str(r["live"]) + ("" if not r.get("form") else f"#{r['form']}") + (f"/{r['how']}" if r.get("how") else "")
    harvest = "-" if not r.get("harvest") else f"{r['harvest']} {float(r.get('harvest_score') or 0.0):.2f}"
    return live, harvest


@dataclass
class SheetResult:
    sid: str
    sheet: int
    doc: dict[str, Any]
    image: Path | None                   # the first page (None: no cluster to show)
    legend_file: Path
    clusters: int
    pending: int
    auto: int
    images: list[Path] = field(default_factory=list)      # every page


SHEET_PAGE_TILES = 30                    # tiles per sheet page (6 a row at 1400 px: about 2,100 px tall)


def is_gold(session: Mapping[str, Any]) -> bool:
    """Gold sessions (held-out friendlies, Round 0, field) are reviewed track by track and never
    auto-accepted (plan 5, 6): their labels must not be tonight's model's reads."""
    return session.get("role") == ROLE_GOLD or session.get("capture") in GOLD_ONLY_CAPTURES


def _sure(part: Any, cfg: Any) -> bool:
    """A confident live read (cluster_parts keeps two such parts apart when they name two
    labels): the auto-accept thresholds on cover, score and agreement."""
    r = part.read or {}
    return ((r.get("cover") or 0.0) >= cfg.accept_cover and (r.get("conf") or 0.0) >= cfg.accept_conf
            and (r.get("agree") or 0.0) >= cfg.accept_agree)


def _mix_text(members: Sequence[Mapping[str, Any]], legend: Sequence[LegendEntry]) -> tuple[str, str]:
    """(tile text, console text) when a cluster's members suggest more than one label."""
    sugs = Counter(m["suggest"] for m in members if m.get("suggest") is not None)
    if len(sugs) < 2:
        return "", ""
    order = sorted(sugs, key=lambda n: (-sugs[n], n))
    return ("mix " + " ".join(f"{sugs[n]}x{n}" for n in order),
            "mixed: " + ", ".join(f"{sugs[n]}x {legend[n].label}" for n in order))


def _shown(keys: Sequence[str], parts_doc: Mapping[str, Mapping[str, Any]], k: int = 4) -> list[str]:
    """The members a tile shows: one of each suggested label first (most common first, its
    most typical member), then spread over the rest (unit_harvest.spread_members)."""
    by_sug: dict[Any, str] = {}
    counts = Counter(parts_doc[key].get("suggest") for key in keys)
    for key in keys:
        by_sug.setdefault(parts_doc[key].get("suggest"), key)
    first = [by_sug[n] for n in sorted(by_sug, key=lambda n: (-counts[n], -1 if n is None else n))]
    if len(first) <= 1:
        return []
    first = first[:k]
    rest = [key for key in keys if key not in first]
    if len(first) < k and rest:
        step = max(1, len(rest) // (k - len(first)))
        first += rest[::step][:k - len(first)]
    return sorted(first, key=list(keys).index)


def make_sheet(sid: str, *, labels_root: str | os.PathLike | None = None, store: str | os.PathLike | None = None,
               table: Any = None, registry: Registry | None = None, geometry: Any = None, config: Any = None,
               rebuild: bool = False, auto: bool | None = None, width: int = 1400,
               page_tiles: int = SHEET_PAGE_TILES, load: Any = None, log: Any = print,
               clock: Any = time.time) -> SheetResult:
    """The contact sheet of one scanned session (module doc, sheet). SheetError otherwise.
    auto: None follows the session (no auto-accept in a gold session, else on); True for a
    gold session is refused."""
    UH = _unit_harvest()
    paths = session_paths(sid, labels_root, store)
    try:
        session = load_session(sid, labels_root)
    except (OSError, ValueError) as exc:
        raise SheetError(f"no ingested session {sid} ({exc}); run ingest first") from None
    gold = is_gold(session)
    if auto and gold:
        raise SheetError(f"{sid} is a gold session (role {session.get('role')}, capture {session.get('capture')}): "
                         "every track is reviewed by hand and nothing is auto-accepted (plan 5, 6)")
    auto = (not gold) if auto is None else bool(auto)
    deck = session_deck(session)
    try:
        scan = UH.load_scan(paths.scan_dir)
    except UH.ScanError as exc:
        raise SheetError(f"{exc}; run `scan {sid}` first") from None
    identity = scan_identity(paths.scan_dir)
    old = read_tracks(paths)
    if old is not None and old.get("scan") != identity:
        raise SheetError(f"{paths.tracks_file} was labelled on another scan of {sid} (harvest track ids "
                         "differ): move it away (git keeps it) to label this scan", EXIT_CHECK)
    table = load_table() if table is None else table
    if registry is None:
        try:
            registry = Registry.load()
        except RegistryError:
            registry = None
    known = known_labels(table, registry)
    cfg = config if config is not None else UH.FlagConfig()
    pairs = [(c.card, c.form) for c in deck]
    want = UH.analysis_identity(scan, deck=pairs, config=cfg, table=table)
    an = None if rebuild else UH.load_analysis(paths.sheet_dir, want)
    if an is None:
        an = UH.analyze_scan(scan, deck=pairs, table=table, geometry=geometry, session_dir=paths.recording,
                             config=cfg, load=load or UH.load_image, log=log)
        UH.save_analysis(an, paths.sheet_dir)
    else:
        log(f"  analysis: reused {paths.sheet_dir / UH.ANALYSIS_JSON} ({len(an.parts)} parts)")
    legend = build_legend(deck, table, known)
    on_legend = {e.label: e for e in legend}
    entries = [t for t in (old or {}).get("tracks") or () if isinstance(t, Mapping)]
    decided = {str(t.get("key")) for t in entries}
    split = {str(k) for k in (old or {}).get("split") or ()}
    by_key = an.by_key()
    stale = sorted((decided | split) - set(by_key))
    moved = sorted(str(t.get("key")) for t in entries if str(t.get("key")) in by_key
                   and (t.get("first_n"), t.get("last_n")) != (by_key[str(t.get("key"))].first_n,
                                                               by_key[str(t.get("key"))].last_n))
    if stale or moved:
        what = []
        if stale:
            what.append(f"names track(s) {', '.join(stale[:6])} that this analysis does not have")
        if moved:
            what.append(f"labelled track(s) {', '.join(moved[:6])} on other images than this analysis cuts them")
        raise SheetError(f"{paths.tracks_file} {' and '.join(what)} (other split settings?): make the sheet with "
                         "the settings it was labelled with, or move tracks.json away", EXIT_CHECK)
    parts_doc: dict[str, dict[str, Any]] = {}
    auto_list: list[dict[str, Any]] = []
    for p in an.parts:
        d = p.to_dict(with_sig=False)
        d["suggest"] = suggest(d, legend, table, known)
        d["choice"] = form_choice(d, legend, table)
        parts_doc[p.key] = d
        if not auto or p.auto is None or p.key in decided or p.key in split:
            continue
        label, _ = canonical_label(table, p.auto["cls"], known, form=p.auto.get("form"))
        if label in on_legend:
            auto_list.append({"key": p.key, "label": label, "legend": on_legend[label].n, "n": p.n})
    auto_keys = {a["key"] for a in auto_list}
    pending = [p for p in an.parts if p.key not in decided and p.key not in auto_keys]
    singles = [p for p in pending if p.key in split]
    keys = {p.key: parts_doc[p.key]["suggest"] for p in pending
            if p.key not in split and parts_doc[p.key]["suggest"] is not None and _sure(p, cfg)}
    clusters = UH.cluster_parts([p for p in pending if p.key not in split], max_dist=cfg.cluster_dist,
                                size_weight=cfg.size_weight, keys=keys)
    for p in sorted(singles, key=lambda q: (q.h, q.part)):
        clusters.append(UH.TrackCluster(len(clusters), [p.key], [0.0], None, p.n))

    prev = None
    try:
        prev = read_sheet(paths)
    except LabelError:
        prev = None
    number = int((prev or {}).get("sheet") or 0) + 1
    cluster_docs, tiles, lines = [], [], []
    for c in clusters:
        members = [parts_doc[k] for k in c.keys]
        flags = Counter(f for m in members for f in (m.get("flags") or {}))
        sugs = Counter(m["suggest"] for m in members if m.get("suggest") is not None)
        sug = min(sugs, key=lambda n: (-sugs[n], n)) if sugs else None
        choices = Counter(tuple(m.get("choice") or ()) for m in members if m.get("choice"))
        choice = list(min(choices, key=lambda ch: (-choices[ch], ch))) if choices else []
        reads = Counter(_read_text(m) for m in members)
        live, harvest = min(reads, key=lambda r: (-reads[r], r))
        flag_txt = " ".join(f for f in UH.FLAGS if f in flags) or "weak"
        if sug is not None:
            sug_txt = f"{sug} {legend[sug].label}"
        elif choice:
            sug_txt = "|".join(str(n) for n in choice) + " which form?"
        else:
            sug_txt = "-"
        mix_tile, mix_line = _mix_text(members, legend)
        why = Counter(str(w).split(":")[0] for m in members for w in (m.get("read") or {}).get("no_auto") or ())
        why_txt = " ".join(f"{w}x{k}" if k > 1 else w for w, k in sorted(why.items(), key=lambda kv: (-kv[1], kv[0])))
        note = mix_tile or ("" if flags or not why_txt else "no auto: " + why_txt)
        shown = _shown(c.keys, parts_doc) or UH.spread_members(c.keys, 4)
        cluster_docs.append({"id": c.id, "tracks": list(c.keys), "dists": list(c.dists), "images": c.images,
                             "flags": {f: flags[f] for f in UH.FLAGS if f in flags}, "suggest": sug,
                             "choice": choice, "mixed": dict(sugs) if len(sugs) > 1 else {}, "read": live,
                             "harvest": harvest, "shown": shown})
        tiles.append(UH.SheetTile(f"#{c.id} n={len(c.keys)} {flag_txt}",
                                  [f"read {live}", f"hv {harvest}", f"sug {sug_txt}"] + ([note] if note else []),
                                  [an.crops.get(k, []) for k in shown]))
        extra = "  ".join(t for t in (mix_line, ("no auto: " + why_txt) if why_txt else "") if t)
        lines.append(f"#{c.id:<3} n={len(c.keys):<3} {flag_txt:18s} read {live:24s} hv {harvest:24s} sug {sug_txt}"
                     + (f"  {extra}" if extra else ""))
    label_with = (f'tools\\vision_autolabel.py label {sid} --sheet {number} --map "ID=VALUE,..."  '
                  "(VALUE: a legend number, a name, or x o ? s)")
    header = [f"{sid} sheet {number}: {len(clusters)} clusters, {len(pending)} tracks to label, "
              f"{len(auto_list)} auto-accepted, {len(decided)} labelled before"
              + ("" if auto else (" (gold: no auto-accept)" if gold else " (no auto-accept)")),
              f"label with: {label_with}"]
    legend_text = legend_lines(legend)
    images: list[Path] = []
    per = max(1, int(page_tiles))
    pages = [tiles[i:i + per] for i in range(0, len(tiles), per)]
    for j, page in enumerate(pages):
        head = list(header)
        if len(pages) > 1:
            head[0] += f"; page {j + 1} of {len(pages)}: clusters {j * per}-{j * per + len(page) - 1}"
        img = UH.render_sheet(page, cell=cfg.cell, width=width, header=head, legend=legend_text)
        name = f"sheet_{number}" if len(pages) == 1 else f"sheet_{number}_p{j + 1}"
        images.append(UH.write_sheet_image(paths.sheet_dir / name, img))
    legend_file = paths.sheet_dir / f"legend_{number}.txt"
    legend_file.parent.mkdir(parents=True, exist_ok=True)
    legend_file.write_text("\n".join(header + [""] + legend_text + [""] + lines) + "\n", encoding="utf-8")
    now = datetime.fromtimestamp(clock()).isoformat(timespec="seconds")
    doc = {"schema": SHEET_SCHEMA, "sid": sid, "sheet": number, "created": now, "scan": identity,
           "deck": [c.given for c in deck], "legend": [e.to_dict() for e in legend], "codes": dict(CODES),
           "clusters": cluster_docs, "auto": auto_list, "auto_on": auto, "tracks": parts_doc,
           "image": images[0].name if images else None, "images": [im.name for im in images],
           "legend_file": legend_file.name, "stats": an.stats, "markers": an.markers,
           "tool": "tools/vision_autolabel.py sheet"}
    write_json(paths.sheet_file, doc)
    return SheetResult(sid, number, doc, images[0] if images else None, legend_file, len(clusters), len(pending),
                       len(auto_list), images)


@dataclass
class LabelResult:
    sid: str
    sheet: int
    doc: dict[str, Any]
    stored: Counter
    auto: int
    split: int
    open_clusters: list[int]
    warnings: list[str]
    written: bool


SIDE_ENEMY, SIDE_ALLY = "enemy", "ally"


def entry_side(dec: Decision, part: Mapping[str, Any]) -> str | None:
    """The side a stored decision trains on: every label is the enemy's (the legend is the
    declared opponent's deck: the owner types o for one of ours), o is ours, x and ? keep the
    side the scan read (they train no box)."""
    if dec.label:
        return SIDE_ENEMY
    if dec.code == CODE_OURS:
        return SIDE_ALLY
    return part.get("side")


def _track_entry(part: Mapping[str, Any], dec: Decision, source: str, sheet: int, cluster: int | None,
                 at: str) -> dict[str, Any]:
    read = part.get("read") or {}
    return {"key": str(part["key"]), "h": int(part["h"]), "part": int(part.get("part", 0)),
            "parts": int(part.get("parts", 1)), "hk": part.get("hk"), "first_n": part["first_n"],
            "last_n": part["last_n"], "n": part["n"], "first_t": part["first_t"], "last_t": part["last_t"],
            "side": entry_side(dec, part), "side_read": part.get("side"), "label": dec.label, "code": dec.code,
            "source": source, "sheet": sheet, "cluster": cluster, "flags": [f for f in (part.get("flags") or {})],
            "read": {"live": read.get("live"), "form": read.get("form"), "card": read.get("card"),
                     "how": read.get("how"), "uid": read.get("uid"), "held": read.get("held") or [],
                     "no_auto": read.get("no_auto") or []}, "at": at}


def label_session(sid: str, mapping: str | None = None, *, labels_root: str | os.PathLike | None = None,
                  store: str | os.PathLike | None = None, table: Any = None, registry: Registry | None = None,
                  sheet: int | None = None, force: bool = False, off_deck: bool = False, dry_run: bool = False,
                  event_cards: Mapping[str, Any] | None = None, clock: Any = time.time) -> LabelResult:
    """Apply one label map to the latest sheet of a session (module doc, label). Every entry
    is checked first; a single bad one raises LabelError and nothing is stored."""
    paths = session_paths(sid, labels_root, store)
    try:
        session = load_session(sid, labels_root)
    except (OSError, ValueError) as exc:
        raise LabelError(f"no ingested session {sid} ({exc})") from None
    doc = read_sheet(paths)
    number = int(doc.get("sheet") or 0)
    problems: list[str] = []
    if sheet is not None and int(sheet) != number:
        problems.append(f"--sheet {sheet}: the latest sheet of {sid} is {number}; label from that one "
                        f"(sheet_{number}*.png, legend_{number}.txt)")
    now_scan = scan_identity(paths.scan_dir)
    if doc.get("scan") != now_scan:
        problems.append(f"sheet {number} was made from another scan of {sid} (scanned again since?): run sheet again")
    old = read_tracks(paths)
    if old is not None and old.get("scan") != doc.get("scan"):
        problems.append(f"{paths.tracks_file} was labelled on another scan: move it away to label this one")
    table = load_table() if table is None else table
    if registry is None:
        try:
            registry = Registry.load()
        except RegistryError:
            registry = None
    known = known_labels(table, registry)
    legend = [LegendEntry.from_dict(d) for d in doc.get("legend") or ()]
    clusters = {int(c["id"]): c for c in doc.get("clusters") or ()}
    parts = {str(k): v for k, v in (doc.get("tracks") or {}).items()}
    decided = {str(t["key"]): dict(t) for t in (old or {}).get("tracks") or () if isinstance(t, Mapping)}
    split = {str(k) for k in (old or {}).get("split") or ()}
    entries, bad = parse_map(mapping) if mapping else ([], [])
    problems += bad
    if sheet is None and any(kind == KIND_CLUSTER for kind, _i, _v in entries):
        problems.append(f"cluster ids are renumbered on every sheet: pass --sheet with the number of the sheet the "
                        f"map was typed from (the latest is {number}); track keys (t17) need none")
    events = load_event_cards() if event_cards is None else event_cards
    key_map = load_key_map()[0] if entries else {}
    by_cluster: dict[str, tuple[Decision, int]] = {}
    by_track: dict[str, tuple[Decision, int | None]] = {}
    for kind, ident, value in entries:
        where = f"cluster {ident}" if kind == KIND_CLUSTER else f"track t{ident}"
        dec, why = resolve_value(value, legend, table, known, off_deck=off_deck, event_cards=events, key_map=key_map)
        if why:
            problems.append(f"{where}: {why}")
            continue
        if kind == KIND_CLUSTER:
            c = clusters.get(int(ident))
            if c is None:
                span = f"0-{max(clusters)}" if clusters else "none"
                problems.append(f"no cluster {ident} on sheet {number} (clusters {span})")
                continue
            keys = [str(k) for k in c.get("tracks") or ()]
            if dec.split and len(keys) < 2:
                problems.append(f"{where} has one track: give it a label or x, o, ?")
                continue
            done = [k for k in keys if k in decided]
            if done and not force:
                problems.append(f"{where}: track(s) {', '.join(done[:5])} are labelled already (sheet {number} "
                                "is older than the labels?); pass --force to change them")
                continue
            for k in keys:
                by_cluster[k] = (dec, int(ident))
        else:
            key = str(ident)
            if key not in parts:
                pieces = sorted(k for k in parts if k.startswith(f"{key}."))
                hint = f": it was split, use {', '.join('t' + k for k in pieces)}" if pieces else ""
                problems.append(f"no track t{key} on sheet {number}{hint}")
                continue
            if dec.split:
                problems.append(f"{where}: s splits a cluster; give the track a label or x, o, ?")
                continue
            if key in decided and not force:
                problems.append(f"{where} is labelled already ({decided[key].get('label') or decided[key].get('code')}); "
                                "pass --force to change it")
                continue
            cid = next((int(c["id"]) for c in clusters.values() if key in (c.get("tracks") or ())), None)
            by_track[key] = (dec, cid)
    for a in doc.get("auto") or ():
        if a.get("label") is not None and str(a.get("key")) not in parts:
            problems.append(f"auto-accepted track t{a.get('key')} is missing from sheet {number}")
    if problems:
        raise LabelError(problems)

    now = datetime.fromtimestamp(clock()).isoformat(timespec="seconds")
    final = {**by_cluster, **by_track}
    out = dict(decided)
    new_split = set(split)
    stored: Counter = Counter()
    for key in sorted(final, key=lambda k: (int(k.split(".")[0]), k)):
        dec, cid = final[key]
        if dec.split:
            new_split.add(key)
            out.pop(key, None)
            stored[CODE_SPLIT] += 1
            continue
        new_split.discard(key)
        out[key] = _track_entry(parts[key], dec, SOURCE_OWNER, number, cid, now)
        stored[dec.label or dec.code] += 1
    n_auto = 0
    warnings = []
    autos = [a for a in doc.get("auto") or () if a.get("label") is not None]
    if autos and is_gold(session):
        warnings.append(f"sheet {number} auto-accepted {len(autos)} track(s), but {sid} is a gold session now: "
                        "none stored; run sheet again (gold puts every track on the sheet)")
        autos = []
    for a in autos:
        key = str(a.get("key"))
        if key in out or key in final or key in new_split:
            continue
        out[key] = _track_entry(parts[key], Decision(label=str(a["label"])), SOURCE_AUTO, number, None, now)
        n_auto += 1
    open_clusters = [int(cid) for cid, c in sorted(clusters.items())
                     if any(str(k) not in out and str(k) not in new_split for k in c.get("tracks") or ())]
    client = session.get("client")
    for label in sorted({e["label"] for e in out.values() if e.get("label")}):
        if label not in known:
            base = registry.train_label(label, 0, table) if registry is not None else label
            then = (f"until it is there and has {registry.thin_tracks} tracks it trains as {base}"
                    if registry is not None and base != label else "until then its boxes cannot be written")
            warnings.append(f"{label} is not a detector label yet: append it to data/vision/detector_extra_labels.json "
                            f"before training ({then})")
        elif registry is not None and not registry.valid_for(label, client):
            warnings.append(f"{label}: a session of client {client} does not teach it (valid_from_client / reworks)")
    history = list((old or {}).get("history") or [])
    history.append({"at": now, "sheet": number, "map": mapping or "", "stored": sum(v for k, v in stored.items()
                                                                                   if k != CODE_SPLIT),
                    "auto": n_auto, "split": stored[CODE_SPLIT], "force": bool(force)})
    tdoc = {"schema": TRACKS_SCHEMA, "sid": sid, "client": client, "scan": doc.get("scan"), "deck": [c.get("given") for c in session.get("deck") or ()],
            "updated": now, "tracks": sorted(out.values(), key=lambda t: (int(t["h"]), int(t.get("part", 0)))),
            "split": sorted(new_split, key=lambda k: (int(k.split(".")[0]), k)), "history": history,
            "tool": "tools/vision_autolabel.py label"}
    if not dry_run:
        write_json(paths.tracks_file, tdoc)
    return LabelResult(sid, number, tdoc, stored, n_auto, stored[CODE_SPLIT], open_clusters, warnings, not dry_run)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _print_problems(head: str, problems: Iterable[str]) -> None:
    print(head)
    for p in problems:
        print(f"  - {p}")


def _cmd_labels(args) -> int:
    try:
        registry = Registry.load(args.registry)
        table = load_table(args.classes)
    except (RegistryError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return EXIT_INPUT
    print(f"registry {args.registry or REGISTRY_PATH}: {len(registry)} labels, digest {registry.digest[:12]}")
    note = registry.table_note(table)
    if note:
        print(f"note: {note}")
    problems = registry.validate(table)
    if problems:
        _print_problems("the registry does not match classes.json:", problems)
        return EXIT_CHECK
    print(f"every entry resolves through classes.json ({len(table)} classes, digest {table.digest[:12]})")
    base_path = Path(args.base) if args.base else BASE_LABELS
    preview = False
    try:
        base = load_label_set(base_path)
    except RegistryError as exc:
        if args.base or args.out:
            print(f"error: {exc}")
            return EXIT_INPUT
        print(f"no base label set at {base_path}: the indices below follow a preview built from "
              "classes.json's KataCR names, not the trained set")
        base, preview = crosswalk_base(table), True
    try:
        ext = registry.extend(base)
    except RegistryError as exc:
        _print_problems("the registry cannot extend this base:", exc.problems)
        return EXIT_CHECK
    print(f"base {'(preview) ' if preview else ''}{len(base)} labels, {len(base.yolo_names())} YOLO classes, "
          f"digest {base.digest}")
    if not preview and base.classes_digest and base.classes_digest != table.digest:
        print(f"note: the base was built from classes.json {base.classes_digest[:12]}, this is {table.digest[:12]}")
    want = registry.base_pin.get("labels")
    if not preview and isinstance(want, int) and want != len(base):
        print(f"note: the registry was written against a {want}-label base; this one has {len(base)}, "
              f"so the appended labels start at {len(base)}")
    for label in registry.in_base(base):
        print(f"note: {label} is already a label of the base set; it adds no class")
    print(f"extended {len(ext)} labels, {len(ext.yolo_names())} YOLO classes, digest {ext.digest}; "
          f"the base's {len(base.yolo_names())} names stay a prefix")
    print(f"  {'label':>5}  {'yolo':>9}  {'name':34s} {'card':18s} {'form':5s} {'trains as':24s} valid from")
    for e in registry:
        i = ext.index(e.label)
        yolo = f"{ext.yolo_index(e.label, 0)}/{ext.yolo_index(e.label, 1)}" if ext.joint_side else str(i)
        rule = e.train_as or (f"{e.base} below {registry.thin_tracks} tracks" if e.base else "itself")
        print(f"  {i:5d}  {yolo:>9}  {e.label:34s} {e.card:18s} {e.form_name:5s} {rule:24s} "
              f"{e.valid_from_client or '-'}")
    rc = EXIT_OK
    if args.against:
        try:
            old = read_yolo_names(args.against)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}")
            return EXIT_INPUT
        bad = check_prefix(old, ext.yolo_names())
        if bad:
            _print_problems(f"against {args.against}:", bad)
            rc = EXIT_CHECK
        else:
            print(f"against {args.against}: its {len(old)} YOLO names stay a prefix")
    if args.out and rc == EXIT_OK:
        out = write_json(args.out, extended_doc(base, registry, ext))
        print(f"extended label set -> {out}")
    return rc


def _cmd_ingest(args) -> int:
    try:
        table = load_table(args.classes)
        registry = Registry.load(args.registry) if args.registry else None
    except (RegistryError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return EXIT_INPUT
    try:
        res = ingest(args.recording, args.deck, client=args.client, sid=args.sid, role=args.role,
                     capture=args.capture, labels_root=args.labels_root, store=args.store,
                     table=table, registry=registry, force=args.force, allow_copy=not args.no_copy)
    except IngestError as exc:
        _print_problems("ingest refused, nothing written:" if exc.code == EXIT_INPUT else "ingest refused:",
                        exc.problems)
        return exc.code
    except OSError as exc:
        print(f"error: {exc}")
        return EXIT_CHECK
    s = res.session
    rec, store = s["recording"], s["store"]
    print(f"{'ingested' if res.created else 'updated'} {res.sid}: role {s['role']} ({s['role_rule']}), "
          f"capture {s['capture']}, client {s['client']}")
    shown = [c["card"] if c["form"] == FORM_BASE else f"{c['card']} {c['form_name']}" for c in s["deck"]]
    print(f"  deck: {', '.join(shown)}")
    print(f"  recording {rec['name']}: {rec['frames']} frames, {rec['seconds']} s, "
          f"{rec['images_listed']} images listed, {rec['images_missing']} missing")
    print(f"  store {res.paths.recording}: {store['images']} images "
          f"({res.counts['linked']} linked, {res.counts['copied']} copied, {res.counts['present']} already there)")
    print(f"  session -> {res.paths.session_file}")
    for w in res.warnings:
        print(f"  ! {w}")
    return EXIT_OK


def _cmd_status(args) -> int:
    registry = None
    try:
        registry = Registry.load(args.registry)
    except RegistryError as exc:
        print(f"note: registry not read ({exc})")
    doc = status(args.labels_root, args.store, args.sid or None, registry)
    if args.json:
        print(json.dumps(doc, indent=1))
        return EXIT_CHECK if doc["problems"] else EXIT_OK
    rows = doc["sessions"]
    roles = doc["roles"]
    print(f"{len(rows)} session(s) under {doc['labels_root']}: "
          + ", ".join(f"{r} {roles[r]}" for r in ROLES))
    if rows:
        print(f"  {'sid':22s} {'role':5s} {'capture':8s} {'client':10s} {'images':>13s} {'seconds':>8s}  stages")
    for r in rows:
        images = f"{r['images']}/{r['images_ingested']}"
        secs = "-" if r["seconds"] is None else f"{r['seconds']:.0f}"
        print(f"  {r['sid']:22s} {str(r['role']):5s} {str(r['capture']):8s} {str(r['client']):10s} "
              f"{images:>13s} {secs:>8s}  {','.join(r['stages'])}")
        print(f"      deck: {','.join(str(x) for x in r['deck'])}")
        for n in r["notes"]:
            print(f"      ! {n}")
    for p in doc["problems"]:
        print(f"  ! {p}")
    if doc["coverage"]:
        print(f"registry labels by declared decks (sessions per role; labelled training tracks, "
              f"forms train as their base below {doc['thin_tracks']}):")
        print(f"  {'label':34s} {'card':18s} {'form':5s} {'train':>5s} {'gold':>4s} {'dev':>3s} {'tracks':>6s}")
        for c in doc["coverage"]:
            tracks = "-" if c["tracks"] is None else str(c["tracks"])
            print(f"  {c['label']:34s} {c['card']:18s} {c['form']:5s} {c[ROLE_TRAIN]:5d} {c[ROLE_GOLD]:4d} "
                  f"{c[ROLE_DEV]:3d} {tracks:>6s}")
        need_train = [c["label"] for c in doc["coverage"] if not c[ROLE_TRAIN]]
        need_gold = [c["label"] for c in doc["coverage"] if not c[ROLE_GOLD]]
        if need_train:
            print(f"to record next, no training session yet: {', '.join(need_train)}")
        if need_gold:
            print(f"to record next, no gold session yet: {', '.join(need_gold)}")
    return EXIT_CHECK if doc["problems"] else EXIT_OK


def _cmd_scan(args) -> int:
    try:
        paths = session_paths(args.sid, args.labels_root, args.store)
        doc = load_session(args.sid, args.labels_root)
    except (OSError, ValueError) as exc:
        print(f"error: no ingested session {args.sid} ({exc}); run ingest first")
        return EXIT_INPUT
    if not image_files(paths.images):
        print(f"error: {paths.images} holds no image: ingest the recording first (or --store)")
        return EXIT_INPUT
    if args.force and paths.tracks_file.is_file():
        print(f"scan refused: {paths.tracks_file} holds labels of the scan there; a new scan gets new harvest "
              "track ids. Move tracks.json away first (git keeps it), then scan again with --force")
        return EXIT_CHECK
    try:
        UH = _unit_harvest()
        from scripts.vision.detector import DetectorError
    except ImportError as exc:
        print(f"error: the scan needs numpy and the vision packages (the buildabot venv): {exc}")
        return EXIT_INPUT
    try:
        cfg = UH.ScanConfig(blobs=not args.no_blobs)
        meta = {"sid": args.sid, "role": doc.get("role"), "capture": doc.get("capture"),
                "client": doc.get("client"), "deck_cards": deck_cards(doc)}
        UH.scan_session(paths.recording, paths.scan_dir, weights=args.detector, device=args.device,
                        use_detector=not args.no_detector, classes=args.classes, config=cfg, limit=args.limit,
                        overwrite=args.force, meta=meta)
    except (UH.ScanError, DetectorError, ValueError) as exc:
        print(f"scan: {exc}")
        return EXIT_INPUT
    print(f"next: & $BB tools\\vision_autolabel.py sheet {args.sid}")
    return EXIT_OK


def _cmd_sheet(args) -> int:
    try:
        table = load_table(args.classes)
        registry = Registry.load(args.registry) if args.registry else None
    except (RegistryError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return EXIT_INPUT
    try:
        res = make_sheet(args.sid, labels_root=args.labels_root, store=args.store, table=table, registry=registry,
                         rebuild=args.rebuild, auto=False if args.no_auto else None, width=args.width,
                         page_tiles=args.page_tiles)
    except LabelError as exc:
        _print_problems("sheet refused:", exc.problems)
        return exc.code
    except ImportError as exc:
        print(f"error: the sheet needs numpy (the buildabot venv): {exc}")
        return EXIT_INPUT
    doc = res.doc
    for line in (res.legend_file.read_text(encoding="utf-8").splitlines()):
        print(line)
    if not res.images:
        print(f"sheet {res.sheet} -> (no cluster to show)")
    for j, im in enumerate(res.images):
        page = f" page {j + 1} of {len(res.images)}" if len(res.images) > 1 else ""
        print(f"sheet {res.sheet}{page} -> {im}")
    print(f"legend -> {res.legend_file}")
    print(f"session file -> {session_paths(args.sid, args.labels_root, args.store).sheet_file}")
    if not res.clusters:
        print(f"nothing left to label: `label {args.sid}` stores the {res.auto} auto-accepted track(s)")
    else:
        print(f'next: & $BB tools\\vision_autolabel.py label {args.sid} --sheet {res.sheet} --map "0=...,1=..."')
    st = doc.get("stats") or {}
    print(f"  {st.get('parts', 0)} part(s): {st.get('auto', 0)} auto, {st.get('flagged', 0)} flagged, "
          f"{st.get('weak', 0)} weak; stopwatches {st.get('markers', 0)} ({st.get('markers_matched', 0)} matched)")
    return EXIT_OK


def _cmd_label(args) -> int:
    try:
        table = load_table(args.classes)
        registry = Registry.load(args.registry) if args.registry else None
    except (RegistryError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return EXIT_INPUT
    try:
        res = label_session(args.sid, args.map, labels_root=args.labels_root, store=args.store, table=table,
                            registry=registry, sheet=args.sheet, force=args.force, off_deck=args.off_deck,
                            dry_run=args.dry_run)
    except LabelError as exc:
        _print_problems("label refused, nothing stored:", exc.problems)
        return exc.code
    owner = sum(v for k, v in res.stored.items() if k != CODE_SPLIT)
    verb = "would store" if not res.written else "stored"
    print(f"{verb} {owner} track label(s) from sheet {res.sheet} and {res.auto} auto-accepted; "
          f"{res.split} track(s) to show again one per tile")
    for name, n in sorted(res.stored.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        if name != CODE_SPLIT:
            print(f"  {n:4d}  {name}" + (f"  ({CODES[name]})" if name in CODES else ""))
    total = len(res.doc["tracks"])
    labelled = sum(1 for t in res.doc["tracks"] if t.get("label"))
    print(f"  {res.sid}: {total} track(s) decided, {labelled} with a label")
    if res.open_clusters:
        print(f"  not labelled yet on sheet {res.sheet}: clusters {', '.join(str(c) for c in res.open_clusters)}")
    if res.split or res.open_clusters:
        print(f"  next: & $BB tools\\vision_autolabel.py sheet {res.sid}   (a new sheet with what is left)")
    for w in res.warnings:
        print(f"  ! {w}")
    if res.written:
        print(f"  tracks -> {session_paths(args.sid, args.labels_root, args.store).tracks_file}")
    return EXIT_OK


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--classes", default=None, help="the class list (default data/vision/classes.json)")
    p.add_argument("--registry", default=None,
                   help="the label registry (default data/vision/detector_extra_labels.json)")


def _add_session_roots(p: argparse.ArgumentParser) -> None:
    p.add_argument("--labels-root", default=None,
                   help="where session.json and the labels go (default data/vision/owner_labels, committed)")
    p.add_argument("--store", default=None,
                   help="where the images are linked (default reports/vision/autolabel, never committed)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tools/vision_autolabel.py",
                                 description="Auto-label recorded friendlies (docs/NEW_CARDS_PLAN.md).")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("labels", help="check the label registry and show the indices it appends")
    _add_common(p)
    p.add_argument("--base", default=None,
                   help="the label set to extend (default tmp/yolo/detector_labels.json; without it "
                        "a preview from classes.json is shown)")
    p.add_argument("--against", default=None,
                   help="an older extended detector_labels.json (or JSON list of YOLO names) that must "
                        "stay a prefix")
    p.add_argument("--out", default=None, help="write the extended detector_labels.json here")
    p.set_defaults(func=_cmd_labels)

    p = sub.add_parser("ingest", help="hard-link one recorded friendly into the store and write session.json")
    _add_common(p)
    _add_session_roots(p)
    p.add_argument("recording", help="the --record root of one session, or the session folder")
    p.add_argument("--deck", required=True,
                   help="the 8 cards the recorded opponent (the top player) played, comma-separated: "
                        "RoyaleAPI keys (cannon-ev1, ronin-hero) or catalogue names (MinionGiant)")
    p.add_argument("--client", required=True, help="the game client version, like 16.635.1")
    p.add_argument("--sid", default=None, help="session id (default: the folder's name), like 20261101_A_1")
    p.add_argument("--role", default=ROLE_AUTO, choices=(ROLE_AUTO,) + ROLES,
                   help="auto (from the session id: friendly 2 gold, deck A's friendly 4 dev, the rest "
                        "train), train, gold or dev")
    p.add_argument("--capture", default=CAPTURE_HARVEST, choices=CAPTURES,
                   help="harvest (a scripted friendly, default), natural (Round 0) or field (a "
                        "clanmate friendly); natural and field are gold only")
    p.add_argument("--force", action="store_true",
                   help="change the deck, role, client or capture of a session ingested before")
    p.add_argument("--no-copy", action="store_true",
                   help="fail instead of copying when an image cannot be hard-linked")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("status", help="list the ingested sessions and what to record next")
    _add_common(p)
    _add_session_roots(p)
    p.add_argument("sid", nargs="*", help="only these sessions")
    p.add_argument("--json", action="store_true", help="print the status as JSON")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("scan", help="scan an ingested session: live chain, harvest pass, background, harvest "
                                    "tracks (scripts/vision/unit_harvest.py; GPU venv, MuMu closed)")
    _add_common(p)
    _add_session_roots(p)
    p.add_argument("sid", help="the session id, like 20261101_A_1")
    p.add_argument("--detector", default=None, help="detector weights (default: the pin, tonight's model)")
    p.add_argument("--device", default=None, help="detector device (default: CUDA GPU 0, else cpu)")
    p.add_argument("--no-detector", action="store_true", help="no detector: HUD and blobs only")
    p.add_argument("--no-blobs", action="store_true", help="skip the background and the blobs")
    p.add_argument("--limit", type=int, default=0, help="stop after N images (0: all)")
    p.add_argument("--force", action="store_true", help="scan again (refused while tracks.json holds labels)")
    p.set_defaults(func=_cmd_scan)

    p = sub.add_parser("sheet", help="flag, split and cluster the enemy tracks of a scan; write the numbered "
                                     "contact sheet and its legend")
    _add_common(p)
    _add_session_roots(p)
    p.add_argument("sid", help="the session id")
    p.add_argument("--rebuild", action="store_true", help="analyse the scan again (default: reuse the cached analysis)")
    p.add_argument("--no-auto", action="store_true",
                   help="auto-accept nothing: every enemy track goes on the sheet (always so for a gold session: "
                        "role gold, or capture natural or field)")
    p.add_argument("--width", type=int, default=1400, help="sheet width in pixels")
    p.add_argument("--page-tiles", type=int, default=SHEET_PAGE_TILES,
                   help=f"clusters per sheet image (default {SHEET_PAGE_TILES}; more go on further pages)")
    p.set_defaults(func=_cmd_sheet)

    p = sub.add_parser("label", help="label the latest sheet's clusters with one map string; a bad entry stores nothing")
    _add_common(p)
    _add_session_roots(p)
    p.add_argument("sid", help="the session id")
    p.add_argument("--map", default=None,
                   help='"ID=VALUE,...": ID a cluster (3), a range (4-7) or a track (t17, t40.1); VALUE a legend '
                        "number, a name classes.json reads, or x (effect), o (ours), ? (unsure), s (split). "
                        "Without --map only the auto-accepted tracks are stored")
    p.add_argument("--sheet", type=int, default=None,
                   help="the number of the sheet the map was typed from (printed on it): needed when the map "
                        "names clusters, which every sheet numbers anew; refused unless it is the latest")
    p.add_argument("--force", action="store_true", help="change tracks labelled before")
    p.add_argument("--off-deck", action="store_true",
                   help="accept names that are not on the declared deck's legend (the deck was declared wrong)")
    p.add_argument("--dry-run", action="store_true", help="check the map and show what would be stored")
    p.set_defaults(func=_cmd_label)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
