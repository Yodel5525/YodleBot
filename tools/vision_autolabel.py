r"""Auto-labelling of recorded friendlies: registry, ingest, status, scan, sheet, label, emit, review, mark, plays,
mix.

docs/NEW_CARDS_PLAN.md 2.3, 4 and 5 (items N07-N13). The owner's main account plays the
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
  emit     N11: the labels propagated over every image of their tracks, then written: a train
           or dev session as images in the live crop framing with label names and YOLO boxes
           (<store>/<sid>/emit/), a gold session as full-frame labels only (<store>/<sid>/gold/:
           never an image, never in a train or val folder).
  review   N12: pages to check an emit by eye (<store>/<sid>/review/): A, a few labelled tracks
           per cluster (gold: every one) with their propagated boxes; B, 24 frames as emitted.
  mark     N12: what a review showed wrong, by track key and page-B tile, into
           data/vision/owner_labels/<sid>/marks.json, with the reviewed precision (Wilson).
  plays    N12, gold only: the opponent's plays derived from the gold tracks under the declared
           deck, edited and confirmed by the owner into data/vision/owner_labels/<sid>/plays.json
           (what tools/vision_gap_eval.py scores the live chain's plays against).
  mix      N13: a training mix (tmp/mix/<name>): KataCR's frames (tonight's tmp/yolo) plus the
           emitted train sessions (dev sessions as val), class indices under the pinned extended
           label set, cr_mix.yaml for scripts/train_detector.py --data. Gold is never mixed.

The gate (N14) is tools/vision_gap_eval.py; it finds a session through session_paths() and
load_session(), and a candidate's mix through mix.json next to its cr_mix.yaml.

Owner commands (PowerShell, from the repo root; $BB = .venvs\buildabot\Scripts\python.exe):

    # the registry: check it, and see the indices it appends to tonight's label set
    & $BB tools\vision_autolabel.py labels --base tmp\yolo\detector_labels.json

    # after each recorded friendly, at once (plan 4.1)
    $SID = '20261101_A_1'      # date, deck letter, friendly number
    & $BB tools\vision_autolabel.py ingest reports\vision\harvest\$SID --deck "goblinstein,rune-giant,goblin-demolisher,suspicious-bush,vines,goblin-curse,elite-barbarians-ev1,baby-dragon-ev1" --role auto --client 16.635.1

    # Round 0: the 2026-10-29 friendly (capture_20261029_195831), copied to reports\vision\gold
    # (gold: no auto-accept). The deck is the main account's, read from Yodle2's spar record (plan 4.3)
    $R0 = 'reports\vision\gold'                       # its session folder, or a folder holding only it
    $R0DECK = 'boss-bandit,elite-barbarians,executioner,freeze,guards,hog-rider,minion-horde-ev1,zap'
    & $BB tools\vision_autolabel.py ingest $R0 --sid 20261029_R0_1 --role gold --capture natural --deck $R0DECK --client 16.635.1

    # what is ingested, and what to record next
    & $BB tools\vision_autolabel.py status

    # then, per session (MuMu and the browser closed for the scan: budget 10-20 min on the GPU)
    & $BB tools\vision_autolabel.py scan $SID
    & $BB tools\vision_autolabel.py sheet $SID          # open the sheet PNG(s) it names; the legend is printed
    & $BB tools\vision_autolabel.py label $SID --sheet 1 --map "0=3,1=3,2=x,5=s"
    & $BB tools\vision_autolabel.py sheet $SID          # again while clusters are left or were split (s)
    & $BB tools\vision_autolabel.py label $SID --sheet 2 --map "0=goblinstein,1=?"
    & $BB tools\vision_autolabel.py emit $SID --dry-run  # optional: the propagated boxes per label
    & $BB tools\vision_autolabel.py emit $SID            # train/dev: <store>\$SID\emit; gold: <store>\$SID\gold
    & $BB tools\vision_autolabel.py emit $SID --force    # again, after a relabel (or a mark, N12)
    & $BB tools\vision_autolabel.py review $SID          # open review_<k>_a*.png and review_<k>_b.png
    & $BB tools\vision_autolabel.py mark $SID --review 1 --bad-tracks t17,t40.1 --bad-frames 3,9
    & $BB tools\vision_autolabel.py mark $SID --review 1  # nothing wrong: recorded all the same
    & $BB tools\vision_autolabel.py label $SID --map "t17=0,t40.1=4" --force   # then emit --force, review
    # gold only: the play list, then its edits and the confirmation (the gate needs it)
    & $BB tools\vision_autolabel.py plays $SID
    & $BB tools\vision_autolabel.py plays $SID --list 1 --bad-plays p12 --add-play "1:52=freeze" --confirm

    # the first fine-tune: once, pin tonight's base in the registry and commit the file; then the mix
    & $BB tools\vision_autolabel.py labels --base tmp\yolo\detector_labels.json --pin-base
    & $BB tools\vision_autolabel.py mix --name ft1       # every train and dev session with a fresh emit
    & $BB tools\vision_autolabel.py mix --name ft1 --force   # again, after more sessions or a re-emit
    # then the train command mix prints (scripts\train_detector.py --data tmp\mix\ft1\cr_mix.yaml ... --no-pin)

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
  train_index); "train_as" pins the choice. A class that only exists in that form (the evolved
  Skeleton Army's General and Spectral, the evolved Witch's skeletons) has no base label: its
  entry keeps base null and trains as itself. valid_from_client (per entry) and "reworks" (for
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
  ronin-hero, cannon-ev1: tools/rs_cards.py's map with its extras, which since 2026-10-30 map
  minion-giant to MinionGiant, unverified) or a catalogue name of classes.json (MinionGiant),
  with -ev1 or -hero for the evolution or hero slot. Refused: a card classes.json lacks, a form
  it lacks, a duplicate, a deck not of 8, and the 22 event-only or internal cards of
  data/vision/event_cards.json (SuperKnight; MergeMaiden_Normal, where spirit-empress is meant).
  Warnings, not refusals: the observed slot rules (plan 1.10: at most 2 evolutions, at most 2
  heroes and champions together, at most 3 in all) and deck units that have no detector label
  yet.
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
  same files; delete both to get the space back (training raws only, after N11 emit and N12
  review and mark: a re-emit needs them; keep gold raws, plan 4.7). status counts the store's images. The scan adds 150-300 MB per battle
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
      (trains_as: Registry.train_label with the table) until it is appended and has
      thin_tracks tracks; a class with no base form has no index until it is appended, so mix
      leaves its images out; class indices are written only by N13's mix, under
      Registry.extend(base, require_pin=True) (emit stores label names).
    - Auto-accepted tracks (source "auto") are on no cluster: N12's review samples them as a
      group of their own.
  status counts the labelled units of training sessions (labelled_tracks: one per harvest or
  live track, not per part).

Emit (N11)
  emit reads session.json (the role), tracks.json, the latest sheet.json (its "tracks": every
  part of the analysis) and the scan, and refuses (exit 1, nothing written) when there is no
  tracks.json or scan, when either was made on another scan, or when tracks.json labels parts
  the sheet does not have. Each part becomes (unit_harvest.decide_parts): a label (enemy side),
  a mask (? or marked bad in marks.json and not relabelled since), an effect (x: no box and no
  mask: a hard negative), ours (o: an ally pseudo-label when the live read or a candidate is one
  of Yodle2's labels, else a mask) or pending (not labelled yet: masked, with a warning).
  unit_harvest.propagate_parts puts its box on every image of its track (smoothed sizes, short
  gaps filled, a circle for DarkMagic, Vines and GoblinCurse); unit_harvest.plan_frame adds
  tonight's pseudo-labels (Yodle2's units, towers, UI and objects: plan risk 4), masks every
  live unit or spell no decided part explains, and masks the leftovers: a detector proposal (at
  EmitConfig.leftover_conf) of a track no decided part holds there that nothing kept or masked
  has, such as one of Yodle2's units the live model missed or an enemy track too short for the
  sheet (gold: ignore regions). Every image in a birth window (birth_s after a
  labelled track's first image) is kept, and 1 in --stride of the other images with an enemy
  label, 1 in --empty-stride of the rest; an image with more than --max-masks masks, a mask
  over a kept moving box or masks without a background is dropped (emit.json counts why).
  Train and dev write <store>/<sid>/emit/: images/<sid>_<frame>.<fmt> (masks pasted with the
  epoch background, feathered; the live crop, detector.crop_rect_for), boxes.jsonl (per image:
  label NAMES, sides, YOLO boxes of the crop, repeat 2 for a training session's birth images),
  parts.jsonl (each part's propagated boxes in frame pixels) and emit.json last. Names, not
  class indices: the thin-form rule depends on the labelled tracks of every session, so N13's
  mix writes the indices (yolo_rows, trains_as) under the pinned label set. The image names
  never match the recorder's (\d{8}), so a disk budget there can never prune them.
  Gold (role gold, or capture natural or field) writes <store>/<sid>/gold/ and NO image:
  gold.jsonl (every battle image: boxes and ignore regions in frame pixels), parts.jsonl,
  labels/<recorder stem>.txt (full-frame YOLO lines) and labels/detector_labels.json (the gold
  names, self-contained: vision_replay <recording> --labels <store>\<sid>\gold\labels reads
  them). A gold emit is refused while parts are pending (unless --allow-pending: they become
  ignore regions), while tracks.json holds auto-accepted labels, and while <store>/<sid>/emit/
  exists (a training emit made before the session became gold: delete it, build any mix again).
  Training images of a gold session are refused right before each write (_assert_trainable).
  emit_fresh() says why an emit no longer holds (tracks.json, marks, role or scan changed):
  status shows it, and N12-N14 refuse a stale emit. emit again needs --force, which deletes
  only the earlier emit's own files, and is refused (nothing changed) while an image the earlier
  emit holds has lost its raw: --drop-missing emits without those. Keep the raws and the scan
  until review and mark are done: a re-emit needs both (status says when raws were deleted
  after an emit). emit.json records the base label set the pseudo-labels were checked against
  (base_digest, base_source: tmp/yolo's real one, else the classes.json preview, with a warning);
  mix warns when it is not the mix's base.

Review, mark and plays (N12)
  review needs a fresh emit (emit_fresh) and writes <store>/<sid>/review/ (gitignored; never the
  emit or gold folder): review_<k>_a.png (or _a_p1, _a_p2, ...) and review_<k>_b.png (.npy
  without an image codec), review_<k>.json and review.json (the latest). Page A: the labelled
  tracks, per_group (--per-cluster, 3) drawn at random from each group, a group being the
  sheet and cluster an owner label came from (s2c0), or auto:<label> for the auto-accepted;
  a gold session shows EVERY labelled track (gold:<label>). Each tile is t<key> with its
  label, group, source and flags, and three crops of the raw image (birth, +1 s, mid-life,
  among the images the emit wrote) with the propagated box (dim: a filled image); where the
  raw is gone, the emitted image, else RAW GONE. Page B: --frames (24) frames, half of them
  in a birth window when there are enough: a training emit's images as written (masks
  outlined in red), a gold session's battle images with their gold boxes (ignore regions in
  red); E is the enemy side, A ours (tonight's pseudo-labels in blue). The seed defaults to
  sha1("<sid>:<k>"), so review k can be drawn again (--seed).
  mark stores what was wrong: --bad-tracks keys of tracks.json parts (17, t40.1), --bad-frames
  the f numbers of review K's page B (--review K, default the latest), --clear to take one back;
  every entry is checked first and a bad one stores nothing (exit 2). marks.json (committed,
  small) keeps keys and image names (with n and the recorder frame i), never tile numbers, and
  per review the parts and frames shown, how many were bad and the Wilson interval of that
  share (wilson; the reviewed precision is 1 minus it): `mark SID --review K` alone records a
  review with nothing bad, which the interval needs. More than 2 bad frames of one review
  name the groups of the parts on them, to check again (label ... --force hints). A bad track
  is masked by the next emit (gold: an ignore region) until it is labelled again with
  `label --force` (the mark is spent once the track's tracks.json "at" is newer); so: mark,
  relabel, emit --force, review. Bad frames need no re-emit: mix and the gate read them
  (bad_frame_numbers).
  plays (gold only) derives the opponent's plays from the fresh gold emit's labelled parts
  (derive_plays) under the deck declared at ingest (classes.json: the deck card that makes
  each class), as EventDeriver's OppPlay reports them, so the gate can match the two (t, the
  first unit's x_mt / y_mt, card, kind, cost, form): a spell label is a play; a unit its card
  summons is one (Guards x3, Minion Horde x6 and Elite Barbarians x2 join one play born within
  --window-s and --radius-tiles of its first unit), and so is a unit of a spell seen as its
  units (Heal, the Spirit Empress: her cost is the form's, 3 or 6); a unit its card only
  spawns is a child (BushGoblin, a curse's Goblins); a later part of the same unit (a look
  change, the same live uid, or a track found again near where it was lost) continues it.
  Gold tracks fragment: one unit is often several parts, lost and found again later (Round 0's
  list 1 had 60 plays for about 18). So, for unit parts (spells and the units of a spell card
  that no card summons, Heal's spirits, keep the rules above; the Spirit Empress' forms, which
  her variant cards summon, are troops; settings in PlayConfig, stored in plays.json
  "config", flags --reach-gap-s, --reach-margin-tiles, --zone-y-tiles, --zone-gap-s,
  --back-tiles, --cycle-plays, --flag-gap-s):
    capacity  a play holds at most the units its card puts down (data/vision/spawn_children.json:
              Guards 3, Minion Horde 6, Elite Barbarians 2, the others 1; a card with forms, no
              limit). A part beyond that, inside the deploy window or on top of a unit still seen
              in a full play up to 12 s old, is a second track on that unit.
    reach     a part continues a unit of its card lost up to 12 s before when the unit could have
              walked there: within its speed (data/royaleapi/cards_stats_characters.json Speed / 60
              tiles a second; the Boss Bandit, newer than that table, 1.5) x 1.25 x the time lost
              + 2 tiles of where it was last seen. While the play has room, it may also be one of
              its units not seen since the deploy (within --radius-tiles + that walk of the
              first unit).
    zone      a part first seen in the defender's half (17 tiles or more from its owner's back
              edge: y >= 17 for the opponent, y <= 15 for our o tracks; not Miner, Goblin Drill
              or Heal's spirits) cannot have been put down there. It continues a unit of
              its card as reach does, lost up to 20 s before, or one still seen within 2 tiles;
              else it is a play of its own whose deploy went unseen ("deploy_unseen"), dated at
              that first sighting, as M0's deriver dates such a play.
  Never a continuation: a part in its owner's deploy zone more than 2 tiles behind (toward its
  own king) where the unit was last seen, since units walk forward (Elite Barbarians put down
  again behind the pair that died at the tower are a new play: "new_by": "behind"); nor a part
  that would bring a deploy-unseen unit back into the deploy zone (by any rule, a live uid
  too). The card cycle: a card comes back to the hand only after 4 other plays, so once 4
  other plays came since its card's last play, a part is taken for a lost unit by reach, zone
  or capacity only on a close fit, a unit seen up to 4 s before within 2 tiles of it (a
  long-lived unit or a building whose track was lost a moment; flagged); else it is a new play
  ("new_by": "cycle"). A champion (CHAMPION_CARDS) is alive once at a time and her card cycles
  only once she dies, so for her the 4 plays count from her last sighting. Without those 4
  plays (labels missed them) two plays of a card can look like one unit lost and found again;
  so the list flags each reach, zone or capacity merge made after 3 or more other plays
  ("cycled": one missed play, often a Zap, would make it a new play), and a track found again
  near where it was lost (rule 1, which holds) after 4, for the owner to check.
  Each play lists its merges: "continued" names each later part's rule (split, uid,
  reacquired, capacity, reach, zone), "joined" the units that joined after the deploy window,
  "gaps" how long the unit was unseen.
  The list prints each play's merges by rule under it ("+ reach t54 3.0 s, t56 3.3 s; zone
  t106.0 9.5 s"), and until the list is confirmed it warns of what to check: every zone merge,
  reach merges over 4 s, the merges late in the card cycle ("cycled"), each deploy-unseen play
  (with an earlier part of its track under another card, if any: its deploy, labelled as
  something else?), and every play that comes fewer than 4 plays after its card's last one
  (a play the labels missed lies between them, often a Zap: --add-play it; or it is no new
  play: --bad-plays it). An --add-play of a deploy-unseen play's card, up to 20 s before its
  first sighting, gives that play's deploy time and replaces it ("replaces").
  The list is printed with M0's read of each play (the scan's live plays within 1 s and 3
  tiles) and numbered p1, p2, ... as list K. Edits name list K (--list K, the latest):
  --bad-plays p3,p7 (a deletion is stored as card and time, so it holds when the list is
  derived again), --add-play "1:52=freeze" (m:ss since the battle's first image; b2/ for
  battle 2; n1234 for scan record n's time; @x,y in tiles; the card must be in the deck
  unless --off-deck), --confirm. Each run writes list K+1 to plays.json (committed, small);
  a changed final list is no longer confirmed. An edit (and --confirm) is refused when the
  plays derived now differ from list K's (the label set, the event cards, --window-s or the
  play rules changed): the owner confirms only a list he was shown. While the gold emit left tracks
  unlabelled (emit --allow-pending) their plays are missing, so --confirm also needs
  --allow-pending. plays.json also lists the children and our own units the sheet showed (o
  tracks: "own", not scored). The gate needs plays_fresh() == []: confirmed, and derived from
  the current tracks.json and gold emit.

Mix (N13)
  `labels --base tmp\yolo\detector_labels.json --pin-base` writes the real base's digest into the
  registry's base.digest, once (pin_base: a text edit of that value only); commit the file. From
  then on Registry.extend refuses any other base, so the appended indices never move, and mix
  needs the pin (extend(require_pin=True)).
  mix reads tonight's KataCR dataset (--katacr, default tmp/yolo: cr.yaml, images/, labels/, its
  detector_labels.json the base), extends the base with the registry (old names stay a prefix,
  registry order, append-only; no two YOLO names equal under strip().lower(), which ultralytics'
  class-head remap matches by), and takes every train and dev session with an emit (or the
  sessions named). For each emitted image it writes the YOLO lines (yolo_rows: thin forms as
  their base by the labelled units of the mix's train sessions whose client may teach the label;
  an image with a label without an index, a client that may not teach it or a bad side is left
  out whole: a sprite without its box would teach background), and hard-links the image (a copy
  when linking fails, as across drives on Windows; --no-copy refuses that), birth images as
  `repeat` x --repeat copies (<stem>__r1.jpg, ...). Frames marked bad (mark --bad-frames) are
  left out. Train sessions go to train/, dev sessions to val/ (never repeated); KataCR's own
  train and val folders are listed in the yaml as they are, so its labels keep their indices.
  tmp/mix/<name>/ (gitignored; never inside tmp/yolo: ultralytics puts the label cache next to the
  first sorted label folder, so a mix there would rewrite tonight's cache):
    cr_mix.yaml           path, train and val as lists of absolute folders, names (the extended set)
    detector_labels.json  the extended label set (its digest goes into the run's manifest through
                          train_detector), with the mix's trains_as and track counts
    mix.json              the sessions (sid, role, frames_sha1, recording name and created, emit
                          created, images, skipped), the gold sessions left out, trains_as, counts
    train/images, train/labels, val/images, val/labels
  Refused, with nothing written: a named gold session; a session that is one recording with a
  gold session (same frames.jsonl SHA-1, or recording name and created: same_recording); a
  source inside a gold folder; an emit.json not of kind train; a stale emit (emit_fresh: emit SID
  --force); an emit in npy (ultralytics reads jpg and png: emit SID --force --format jpg); an
  unpinned registry or one that does not match classes.json; KataCR's cr.yaml names that are not
  the base's; an --out in the store, the committed labels, tmp/yolo or the repo outside tmp/; a
  mix folder there already (--force removes only the mix's own files) or one holding other files.
  The emitted images are hard links: delete a mix (or emit --force) and the other copy remains.

Exit codes: 0 ok; 1 a check failed or a change was refused; 2 bad input.

Standard library only for labels, ingest, status, label, mark, plays and mix (scripts/vision/labels.py
loads tools/vision_classes.py; tools/rs_cards.py gives the RoyaleAPI keys), so they work whatever
the vision packages do; scan, sheet, emit and review import scripts/vision/unit_harvest.py
(numpy) when they run.
Nothing here talks to the emulator, reads game memory or taps.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
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
# N11-N14 files. In <store>/<sid>/: emit/ (train and dev) or gold/ (gold: no image), review/.
EMIT_DIR, GOLD_DIR = "emit", "gold"
EMIT_JSON, PARTS_FILE, BOXES_FILE, GOLD_FILE = "emit.json", "parts.jsonl", "boxes.jsonl", "gold.jsonl"
GOLD_LABELS_DIR = "labels"                 # in gold/: <8 digits>.txt and detector_labels.json
GOLD_NAMES_FILE = "detector_labels.json"   # vision_replay.find_label_names reads this name
REVIEW_DIR, REVIEW_FILE = "review", "review.json"            # N12
MARKS_FILE, PLAYS_FILE = "marks.json", "plays.json"          # N12, in data/vision/owner_labels/<sid>/
EMIT_SCHEMA = "hastycr.vision.autolabel_emit/1"
REVIEW_SCHEMA = "hastycr.vision.autolabel_review/1"
MARKS_SCHEMA = "hastycr.vision.autolabel_marks/1"
PLAYS_SCHEMA = "hastycr.vision.gold_plays/1"
MIX_SCHEMA = "hastycr.vision.autolabel_mix/1"                # N13
KATACR_ROOT = ROOT / "tmp" / "yolo"                          # prepare_yolo_dataset OUT (BASE_LABELS lives there)
MIX_ROOT = ROOT / "tmp" / "mix"                              # N13: mixes outside tmp/yolo (label cache)
KIND_TRAIN, KIND_GOLD = "train", "gold"                      # emit.json "kind": train (also dev) or gold
#: The files `emit` writes, and the only ones `emit --force` deletes (images: <sid>_*.<fmt>).
EMIT_OWN_FILES = (EMIT_JSON, PARTS_FILE, BOXES_FILE, GOLD_FILE)
#: (stage, "labels" or "store", path in that session folder): a stage is done when the path
#: exists. N11-N13 append theirs.
STAGES: list[tuple[str, str, str]] = [("ingest", "labels", SESSION_FILE),
                                      ("scan", "store", f"{SCAN_DIR}/{SCAN_JSON}"),
                                      ("sheet", "store", f"{SHEET_DIR}/{SHEET_FILE}"),
                                      ("label", "labels", TRACKS_FILE),
                                      ("emit", "store", f"{EMIT_DIR}/{EMIT_JSON}"),
                                      ("emit-gold", "store", f"{GOLD_DIR}/{EMIT_JSON}"),
                                      ("review", "store", f"{REVIEW_DIR}/{REVIEW_FILE}"),
                                      ("mark", "labels", MARKS_FILE),
                                      ("plays", "labels", PLAYS_FILE)]

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
    base: str | None = None              # form entries: the base-form label (None: the class has none)
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
                if e.form != FORM_BASE and lab.form == e.form:
                    # base is the class's base-form label; a class that only exists in this form
                    # (the evolved Skeleton Army's General, the evolved Witch's skeletons) has
                    # none, so its entry keeps base null and trains as itself.
                    want = table.detector_key(lab.class_id, FORM_BASE)
                    if want is None and e.base is not None:
                        problems.append(f"{where}: {vc.name} has no base form: leave base null "
                                        "(the entry trains as itself)")
                    elif want is not None and e.base is None:
                        problems.append(f"{where}: an {e.form_name} entry needs base, the base-form label "
                                        f"({want!r}) it trains as while it is thin")
                    elif e.base != want:
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
        writes class indices (N13 mix, through yolo_rows; N11 emit writes label names only)
        passes it, so a base from another prepare_yolo_dataset run cannot move the appended
        indices unnoticed."""
        problems = []
        if base.mode != L.MODE_VISION:
            problems.append(f"the base label set is in {base.mode!r} mode; the registry extends a "
                            f"{L.MODE_VISION!r} label set (prepare_yolo_dataset --labels vision)")
        pin = self.base_pin.get("digest")
        if require_pin and not pin:
            problems.append(f"the registry's base.digest is not pinned yet: run `labels --base "
                            f"tmp\\yolo\\detector_labels.json --pin-base` (it pins {base.digest} for this base) and "
                            "commit data/vision/detector_extra_labels.json before writing class indices")
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
        train_as pins it; every other label (a form entry without base among them) trains as
        itself. With the class table, a hero or evolution form that is not in the registry yet
        (labelled from a '*' legend entry) trains as its class's base form, as a thin form does;
        a class with no base form gives the label itself, which the extended set lacks."""
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

    @property
    def emit_dir(self) -> Path:
        """Train and dev sessions' emit (N11): images in the live crop framing, boxes.jsonl."""
        return self.store_dir / EMIT_DIR

    @property
    def gold_dir(self) -> Path:
        """Gold sessions' emit (N11): full-frame labels and gold.jsonl, never an image."""
        return self.store_dir / GOLD_DIR

    @property
    def review_dir(self) -> Path:
        return self.store_dir / REVIEW_DIR

    @property
    def marks_file(self) -> Path:
        return self.labels_dir / MARKS_FILE

    @property
    def plays_file(self) -> Path:
        return self.labels_dir / PLAYS_FILE


def out_dir_for(paths: SessionPaths, session: Mapping[str, Any]) -> Path:
    """Where a session's emit goes: gold/ for a gold session (is_gold), emit/ otherwise."""
    return paths.gold_dir if is_gold(session) else paths.emit_dir


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
        out.update(_labelled_units(doc))
    return out


def _labelled_units(doc: Any, exclude: Iterable[str] = ()) -> Counter:
    """One tracks.json's labelled units per label (labelled_tracks), leaving out the part keys
    in `exclude` (mix: the tracks marked bad, which emit masked)."""
    out: Counter = Counter()
    skip = {str(k) for k in exclude}
    seen: set[tuple[str, str]] = set()
    for i, t in enumerate((doc.get("tracks") if isinstance(doc, Mapping) else None) or ()):
        if not (isinstance(t, Mapping) and isinstance(t.get("label"), str)) or str(t.get("key")) in skip:
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
        gold = is_gold(doc)
        emitted = any((d / EMIT_JSON).is_file() for d in (paths.emit_dir, paths.gold_dir))
        if gold and now < had:
            notes.append(f"{had - now} gold raw image(s) deleted: keep gold raws (plan 4.7)")
        elif now < had and emitted:
            notes.append(f"raws deleted: emit cannot be redone for {had - now} image(s) (emit --force refuses while it "
                         "would lose emitted images; --drop-missing emits without them; delete training raws only "
                         "after review and mark are done)")
        elif now < had:
            notes.append(f"{had - now} raw image(s) deleted before emit: emit skips them")
        if emitted:
            stale = emit_fresh(paths, doc)
            if MARKS_CHANGED in stale:
                notes.append(f"marks newer than emit: emit {sid} --force")
            other = [r for r in stale if r != MARKS_CHANGED]
            if other:
                notes.append(f"emit stale: {'; '.join(other)}: emit {sid} --force")
            if gold:
                notes += plays_notes(paths)
        notes += review_notes(paths)
        if gold:
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


def base_label_set(table: Any, base: str | os.PathLike | None = BASE_LABELS) -> tuple[L.LabelSet, bool]:
    """(base_labels, whether it is the real one rather than the preview)."""
    if base is not None and Path(base).is_file():
        try:
            return load_label_set(base), True
        except (RegistryError, OSError, ValueError):
            pass
    return crosswalk_base(table), False


def base_labels(table: Any, base: str | os.PathLike | None = BASE_LABELS) -> L.LabelSet:
    """Tonight's label set: the real one prepare_yolo_dataset wrote (`base`, by default
    tmp/yolo/detector_labels.json) when it is there and reads, else the preview built from
    classes.json's KataCR names (crosswalk_base; plan 8: the real one may hold more)."""
    return base_label_set(table, base)[0]


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
            # the base form and the slot's form, where the class has them: a class that only
            # exists evolved (the evolved Skeleton Army's General) is on an evo slot's legend only
            forms = [f for f in dict.fromkeys((FORM_BASE, c.form)) if f in vc.forms]
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
                     "training (a form trains as its base until it has enough tracks; a class with no base form "
                     "has no index until it is appended, and mix leaves its images out)")
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
# emit (N11)
# ---------------------------------------------------------------------------

#: emit_fresh reasons that status words on their own.
TRACKS_CHANGED = "tracks.json changed since emit"
MARKS_CHANGED = "the bad-track marks changed since emit"
#: yolo_rows problem kinds (mix counts the images it leaves out by these).
YOLO_NO_INDEX, YOLO_CLIENT, YOLO_SIDE = "no index", "client", "side"


class EmitError(LabelError):
    """emit refused (code 1: stale scan, gold with pending or auto labels, an emit already
    there) or bad input (code 2: no session, no numpy, no image codec); nothing is written."""


@dataclass
class EmitResult:
    sid: str
    kind: str                     # "train" (train and dev) or "gold"
    out_dir: Path
    doc: dict[str, Any]           # emit.json (with dry_run: the propagation counts only)
    counts: Counter
    warnings: list[str]
    written: bool


def _key_order(key: Any) -> tuple[int, str]:
    try:
        return int(str(key).split(".")[0]), str(key)
    except ValueError:
        return 1 << 30, str(key)


def sha1_file(path: str | os.PathLike) -> str | None:
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def read_marks(paths: SessionPaths) -> dict[str, Any] | None:
    """data/vision/owner_labels/<sid>/marks.json (N12 `mark` writes it), or None when there
    is none; LabelError (code 1) when it is there but unreadable: a mark is never ignored."""
    if not paths.marks_file.is_file():
        return None
    try:
        doc = _read_json(paths.marks_file)
    except (OSError, ValueError) as exc:
        raise LabelError(f"{paths.marks_file} cannot be read ({exc}); fix or move it away", EXIT_CHECK) from None
    if not isinstance(doc, Mapping) or doc.get("schema") != MARKS_SCHEMA:
        raise LabelError(f"{paths.marks_file} is not a {MARKS_SCHEMA} document", EXIT_CHECK)
    return dict(doc)


def effective_bad_tracks(marks_doc: Mapping[str, Any] | None, tracks_doc: Mapping[str, Any] | None) -> set[str]:
    """The part keys marked bad (marks.json "bad_tracks") that still hold: a mark is spent
    once the track's tracks.json entry is newer ("at") than when it was marked ("label_at"),
    that is, once it was labelled again with `label --force`. emit masks these (gold: ignore
    regions); emit_fresh compares them with the emit's."""
    if not marks_doc:
        return set()
    entries = {str(t.get("key")): t for t in (tracks_doc or {}).get("tracks") or () if isinstance(t, Mapping)}
    out: set[str] = set()
    for key, mark in (marks_doc.get("bad_tracks") or {}).items():
        entry = entries.get(str(key))
        then = mark.get("label_at") if isinstance(mark, Mapping) else None
        now = entry.get("at") if isinstance(entry, Mapping) else None
        if now and then and str(now) > str(then):
            continue
        out.add(str(key))
    return out


def emit_fresh(paths: SessionPaths, session: Mapping[str, Any]) -> list[str]:
    """Why a session's emit no longer holds ([] when it does): none yet or unreadable; made
    while the session had another kind (gold or not) or role; tracks.json changed since
    (TRACKS_CHANGED); the effective bad-track marks changed (MARKS_CHANGED); the scan changed
    (compared only while the scan's files exist: it may be deleted after emit). mix, review,
    plays and the gate refuse a stale emit: run `emit SID --force`."""
    kind = KIND_GOLD if is_gold(session) else KIND_TRAIN
    out = out_dir_for(paths, session)
    other = paths.emit_dir if out == paths.gold_dir else paths.gold_dir
    p = out / EMIT_JSON
    if not p.is_file():
        if (other / EMIT_JSON).is_file():
            was = KIND_TRAIN if kind == KIND_GOLD else KIND_GOLD
            return [f"the emit in {other.name}/ was made while {paths.sid} was {was}; it is {kind} now"]
        return ["no emit yet"]
    try:
        doc = _read_json(p)
    except (OSError, ValueError):
        return [f"{EMIT_JSON} cannot be read"]
    if not isinstance(doc, Mapping) or doc.get("schema") != EMIT_SCHEMA:
        return [f"{EMIT_JSON} is not a {EMIT_SCHEMA} document"]
    reasons = []
    if doc.get("kind") != kind:
        reasons.append(f"emitted as {doc.get('kind')}, the session is {kind} now")
    if doc.get("role") != session.get("role"):
        reasons.append(f"the role changed since emit ({doc.get('role')} -> {session.get('role')})")
    if doc.get("tracks_sha1") != sha1_file(paths.tracks_file):
        reasons.append(TRACKS_CHANGED)
    try:
        bad = effective_bad_tracks(read_marks(paths), read_tracks(paths))
    except (LabelError, OSError, ValueError):
        reasons.append(f"{MARKS_FILE} or {TRACKS_FILE} cannot be read")
    else:
        if sorted(bad) != sorted(str(k) for k in doc.get("bad_tracks") or ()):
            reasons.append(MARKS_CHANGED)
    if (paths.scan_dir / SCAN_JSON).is_file() and (paths.scan_dir / SCAN_TRACKS).is_file():
        if scan_identity(paths.scan_dir) != doc.get("scan"):
            reasons.append("the scan changed since emit")
    return reasons


def plays_notes(paths: SessionPaths) -> list[str]:
    """status notes on a gold session's plays.json (N12, plays_fresh): not derived, stale or not
    confirmed (the first that holds)."""
    sid = paths.sid
    reasons = plays_fresh(paths)
    if not reasons:
        return []
    if PLAYS_MISSING in reasons:
        return [f"gold plays not derived yet: plays {sid}"]
    if PLAYS_TRACKS in reasons:
        return [f"plays stale (tracks.json changed since): plays {sid}"]
    if PLAYS_EMIT in reasons:
        return [f"plays stale (the gold emit is newer): plays {sid}"]
    if PLAYS_UNCONFIRMED in reasons:
        try:
            k = (read_plays(paths) or {}).get("list")
        except LabelError:
            k = None
        return [f"plays not confirmed: plays {sid} --list {k} --confirm"]
    return list(reasons)


def card_of_label(label: str, deck: Sequence[str], table: Any) -> str | None:
    """The deck card a label's unit or spell belongs to: a unit's creating card in the deck
    (ClassTable.card_for: MinionHorde_EV1 -> MinionHorde in a deck with the Horde), a spell
    its card (zap -> Zap); None for anything else."""
    lab = L.resolve_label(table, str(label))
    if lab.type == L.TYPE_SPELL:
        return lab.card
    if lab.type in (L.TYPE_UNIT, L.TYPE_TOWER) and lab.class_id is not None:
        return table.card_for(lab.class_id, list(deck))
    return None


def gold_label_set(labels_used: Sequence[str], table: Any) -> L.LabelSet:
    """The gold labels as a joint-side LabelSet, in order of first use. Gold is never trained
    on, so its names are self-contained and may hold a label no training set has
    (MinionHorde_EV1); its detector_labels.json lets vision_replay --labels read the files."""
    out = L.LabelSet([], joint_side=True, mode=L.MODE_VISION, classes_digest=getattr(table, "digest", None))
    for label in labels_used:
        out.add(str(label), L.STATUS_EXTRA)
    return out


def trains_as(label: str, ext: L.LabelSet, registry: Registry, tracks: int, table: Any) -> str:
    """The label a box of `label` is written as in a training mix: a registry entry by the
    thin-form rule (Registry.train_label with its labelled tracks); else a label the extended
    set has as itself (tonight's Skeleton_EV1 and Knight_EV1 train as themselves: train_label
    with the table would give their base); else (a '*' legend form) its class's base form."""
    label = str(label)
    if label in registry:
        return registry.train_label(label, int(tracks))
    if label in ext:
        return label
    return registry.train_label(label, 0, table)


def yolo_rows(boxes: Iterable[Sequence[Any]], ext: L.LabelSet, registry: Registry, counts: Mapping[str, int],
              table: Any, *, client: str | None = None) -> tuple[list[str], list[str]]:
    """boxes.jsonl boxes ([label, side, cx, cy, w, h, ...]) as YOLO lines "idx cx cy w h" (6
    decimals) under the extended label set, and the problems ("<kind>: <detail>", kind
    YOLO_NO_INDEX for a label whose trains_as is not in `ext`, YOLO_CLIENT for one the
    session's client may not teach, YOLO_SIDE for a side that is not 0 or 1), each with no
    line. counts: labelled training tracks per label (the thin-form rule). The caller (mix)
    leaves the whole image out when there is a problem: a sprite without its box would teach
    background."""
    lines: list[str] = []
    problems: list[str] = []
    for b in boxes:
        label = str(b[0])
        target = trains_as(label, ext, registry, int(counts.get(label, 0)), table)
        if target not in ext:
            problems.append(f"{YOLO_NO_INDEX}: {label}" + (f" (trains as {target})" if target != label else ""))
            continue
        if not (registry.valid_for(label, client) and registry.valid_for(target, client)):
            problems.append(f"{YOLO_CLIENT}: {label} is not taught by client {client}")
            continue
        side = b[1]
        if ext.joint_side and side not in (0, 1):
            problems.append(f"{YOLO_SIDE}: {label} has side {side!r}")
            continue
        cx, cy, w, h = (float(v) for v in b[2:6])
        lines.append(f"{ext.yolo_index(target, None if side is None else int(side))} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, problems


def _assert_trainable(session: Mapping[str, Any]) -> None:
    """Refuse (EmitError) to write a training image of a gold session: called right before
    every image write, so gold cannot reach a train folder whatever routed it there."""
    if is_gold(session):
        raise EmitError(f"{session.get('sid')} is a gold session (role {session.get('role')}, capture "
                        f"{session.get('capture')}): gold is never written as training images (plan 6)", EXIT_CHECK)


def _clear_emit(out: Path, sid: str) -> int:
    """Delete the files an earlier emit wrote in `out` (EMIT_OWN_FILES, images/<sid>_*, the
    gold labels) and nothing else; the number deleted."""
    n = 0
    for name in EMIT_OWN_FILES:
        p = out / name
        if p.is_file():
            p.unlink()
            n += 1
    images = out / IMAGES_DIR
    if images.is_dir():
        for f in images.iterdir():
            if f.is_file() and f.name.startswith(f"{sid}_") and f.suffix.lower() in (".jpg", ".png", ".npy"):
                f.unlink()
                n += 1
    labels = out / GOLD_LABELS_DIR
    if labels.is_dir():
        for f in labels.iterdir():
            if f.is_file() and (f.suffix.lower() == ".txt" or f.name == GOLD_NAMES_FILE):
                f.unlink()
                n += 1
    return n


def _geometry(geometry: Any) -> Any:
    from scripts.vision.geometry import load_geometry
    if geometry is None or isinstance(geometry, (str, os.PathLike)):
        return load_geometry(geometry)
    return geometry


def emit_session(sid: str, *, labels_root: str | os.PathLike | None = None, store: str | os.PathLike | None = None,
                 table: Any = None, registry: Registry | None = None, geometry: Any = None, config: Any = None,
                 force: bool = False, allow_pending: bool = False, drop_missing: bool = False,
                 dry_run: bool = False, load: Any = None, base: str | os.PathLike | None = BASE_LABELS,
                 log: Any = print, clock: Any = time.time) -> EmitResult:
    """Propagate the labels of one session over its scan and emit them (module doc, emit).
    Everything is checked before anything is written; EmitError otherwise. Train and dev
    sessions write <store>/<sid>/emit/, gold sessions <store>/<sid>/gold/ (no image). An emit
    again (--force) is refused while an image of the earlier emit would be lost because its raw
    is gone (D14), unless drop_missing. base: tonight's label set (base_labels; emit.json
    records which one it was)."""
    paths = session_paths(sid, labels_root, store)
    try:
        session = load_session(sid, labels_root)
    except (OSError, ValueError) as exc:
        raise EmitError(f"no ingested session {sid} ({exc}); run ingest first") from None
    gold = is_gold(session)
    kind = KIND_GOLD if gold else KIND_TRAIN
    dev = session.get("role") == ROLE_DEV
    tdoc = read_tracks(paths)
    if tdoc is None:
        raise EmitError(f"{sid} has no labels yet ({paths.tracks_file}): run sheet and label first", EXIT_CHECK)
    try:
        UH = _unit_harvest()
    except ImportError as exc:
        raise EmitError(f"emit needs numpy (the buildabot venv): {exc}") from None
    try:
        scan = UH.load_scan(paths.scan_dir)
    except UH.ScanError as exc:
        raise EmitError(f"{exc}: emit reads the scan the labels were made on (scan {sid})", EXIT_CHECK) from None
    try:
        sheet = read_sheet(paths)
    except LabelError as exc:
        raise EmitError(exc.problems, EXIT_CHECK) from None
    identity = scan_identity(paths.scan_dir)
    problems: list[str] = []
    if tdoc.get("scan") != identity:
        problems.append(f"{paths.tracks_file} was labelled on another scan of {sid} (scanned again since?)")
    if sheet.get("scan") != identity:
        problems.append(f"sheet {sheet.get('sheet')} was made from another scan of {sid}: run sheet again")
    sheet_parts = {str(k): dict(v) for k, v in (sheet.get("tracks") or {}).items() if isinstance(v, Mapping)}
    entries = [t for t in tdoc.get("tracks") or () if isinstance(t, Mapping)]
    off = sorted({str(t.get("key")) for t in entries} - set(sheet_parts), key=_key_order)
    if off:
        problems.append(f"{paths.tracks_file.name} labels track(s) {', '.join('t' + k for k in off[:6])} that sheet "
                        f"{sheet.get('sheet')} does not have (other split settings?): run sheet again")
    moved = sorted((str(t.get("key")) for t in entries if str(t.get("key")) in sheet_parts
                    and (t.get("first_n"), t.get("last_n")) != (sheet_parts[str(t.get("key"))].get("first_n"),
                                                                sheet_parts[str(t.get("key"))].get("last_n"))),
                   key=_key_order)
    if moved:
        problems.append(f"labelled track(s) {', '.join('t' + k for k in moved[:6])} lie on other images than the "
                        "sheet cuts them: run sheet again with the settings they were labelled with")
    marks = None
    try:
        marks = read_marks(paths)
    except LabelError as exc:
        problems += exc.problems
    bad = effective_bad_tracks(marks, tdoc)
    decisions = UH.decide_parts(sheet_parts, tdoc, bad)
    pending = sorted((k for k, d in decisions.items() if d[0] == UH.DECISION_PENDING), key=_key_order)
    if gold:
        autos = [str(t.get("key")) for t in entries if t.get("source") == SOURCE_AUTO]
        if pending and not allow_pending:
            problems.append(f"{len(pending)} track(s) of gold session {sid} are not labelled yet "
                            f"({', '.join('t' + k for k in pending[:6])}): label them (gold: review every track), "
                            "or pass --allow-pending to make them ignore regions")
        if autos:
            problems.append(f"{len(autos)} auto-accepted label(s) in gold session {sid}: check each, e.g. "
                            f"label {sid} --map \"t{autos[0]}=...\" --force")
        if paths.emit_dir.exists():
            problems.append(f"{paths.emit_dir} holds a training emit of {sid}, made while it was not gold: delete "
                            "that folder (and build any mix again) before the gold emit")
    if problems:
        raise EmitError(problems, EXIT_CHECK)

    table = load_table() if table is None else table
    if registry is None:
        try:
            registry = Registry.load()
        except RegistryError:
            registry = None
    base_ls, base_real = base_label_set(table, base)
    known = set(base_ls.labels) | set(registry.labels if registry is not None else ())
    cfg = config if config is not None else UH.EmitConfig()
    try:
        geo = _geometry(geometry)
    except (OSError, ValueError) as exc:
        raise EmitError(f"geometry: {exc}") from None
    dc = scan.header.get("detector_config")
    try:
        det_config = UH.DetectorConfig.from_dict(dc) if isinstance(dc, Mapping) else UH.DetectorConfig()
    except (TypeError, ValueError) as exc:
        raise EmitError(f"the scan's detector_config does not load ({exc})", EXIT_CHECK) from None
    meta = scan.header.get("meta") if isinstance(scan.header.get("meta"), Mapping) else {}
    own_labels = table.deck_labels(list(meta.get("our_deck") or UH.DEFAULT_OUR_DECK))
    if not gold and not dry_run and cfg.image_format != "npy" and not UH.image_codec_available(cfg.image_format):
        raise EmitError(f"saving .{cfg.image_format} needs opencv-python or pillow (the buildabot venv); "
                        "or pass --format npy")

    # propagate
    hs = {int(sheet_parts[k]["h"]) for k in decisions}
    index = UH.load_frame_index(UH.iter_scan_frames(paths.scan_dir), wanted=hs,
                                leftover_conf=cfg.leftover_conf if cfg.mask_leftovers else None)
    harvest = {int(t["h"]): t for t in scan.tracks if t.get("h") is not None and int(t["h"]) in hs}
    live_teams = {str(k): (v or {}).get("team") for k, v in scan.live_tracks.items()}
    plans = UH.propagate_parts(decisions, sheet_parts, harvest, index, live_teams, table=table, geometry=geo,
                               config=cfg)
    deck = deck_cards(session)
    for p in plans.values():
        if p.label:
            lab = L.resolve_label(table, p.label)
            p.card = card_of_label(p.label, deck, table)
            p.form = int(lab.form) if lab.type == L.TYPE_UNIT else FORM_BASE
    by_decision = Counter(p.decision for p in plans.values())
    per_label: Counter = Counter()
    for p in plans.values():
        if p.label:
            per_label[p.label] += len(p.boxes)
    warnings: list[str] = []
    if pending:
        what = ("ignore regions (--allow-pending): the gate scores no box there, and their plays are missing from the "
                "gold play list (plays refuses --confirm without --allow-pending)") if gold else (
            f"masked: sheet {sid} and label them, then emit {sid} --force")
        warnings.append(f"{len(pending)} track(s) not labelled yet ({', '.join('t' + k for k in pending[:6])}) are {what}")
    if not gold and not base_real:
        warnings.append(f"no readable base label set at {shown_path(base) or base}: emitted against the classes.json "
                        "preview, which may lack labels tonight's model has (their boxes are then masked or skipped, "
                        "and mix leaves images out); emit again with --force once prepare_yolo_dataset has written it")
    if not gold:
        new = sorted(label for label in per_label if label not in known)
        falls = {label: (registry.train_label(label, 0, table) if registry is not None else label) for label in new}
        fallback = [label for label in new if falls[label] != label]
        stuck = [label for label in new if falls[label] == label]
        if fallback:
            warnings.append(f"{', '.join(f'{k} (as {falls[k]})' for k in fallback)}: no detector label yet (a '*' "
                            "legend entry); mix trains a form as its base until it is in "
                            "data/vision/detector_extra_labels.json with enough tracks")
        if stuck:
            warnings.append(f"{', '.join(stuck)}: no detector label yet and nothing to train as (no base form): mix "
                            "has no index for them and leaves their images out until they are appended to "
                            "data/vision/detector_extra_labels.json")
        if (paths.gold_dir / EMIT_JSON).is_file():
            warnings.append(f"{paths.gold_dir} holds a gold emit made while {sid} was gold: the gate no longer uses "
                            "it; delete it")
    if dry_run:
        log(f"{sid} ({kind}): {len(plans)} part(s): "
            + ", ".join(f"{by_decision[d]} {d}" for d in UH.DECISIONS if by_decision[d]))
        for label, k in sorted(per_label.items(), key=lambda kv: (-kv[1], kv[0])):
            log(f"  {k:6d} boxes  {label}")
        doc = {"sid": sid, "kind": kind, "parts": {d: by_decision[d] for d in UH.DECISIONS},
               "labels": dict(sorted(per_label.items())), "pending": pending, "bad_tracks": sorted(bad, key=_key_order)}
        return EmitResult(sid, kind, out_dir_for(paths, session), doc, Counter(), warnings, False)

    # the output folder: an emit there is replaced only with force, and only its own files
    out = out_dir_for(paths, session)
    if (out / EMIT_JSON).exists() and not force:
        raise EmitError(f"{out} holds an emit of {sid} already: pass --force to emit again", EXIT_CHECK)
    at = UH.parts_at(plans)
    births = set() if gold else UH.birth_images(plans, index, cfg)
    with_bg = {str(ep.get("id")) for ep in scan.header.get("epochs") or () if isinstance(ep, Mapping) and ep.get("image")}
    framing: dict[tuple[int, int], Any] = {}
    counts: Counter = Counter()
    dropped: Counter = Counter()
    by_src: Counter = Counter()
    by_label: Counter = Counter()
    spans: dict[int, list[float]] = {}
    backgrounds: dict[str, Any] = {}
    load_fn = load or UH.load_image
    gold_rows: list[tuple[Any, Any]] = []
    planned: list[tuple[Any, Any, bool, Any, float]] = []     # training: (record, plan, birth, crop, pad)

    # plan every image first (no pixels): which images the emit writes
    for n in sorted(index):
        rec = index[n]
        span = spans.setdefault(rec.hk, [rec.t, rec.t])
        span[0], span[1] = min(span[0], rec.t), max(span[1], rec.t)
        if not rec.image:
            continue
        counts["battle_images"] += 1
        if rec.size not in framing:
            try:
                framing[rec.size] = UH.emit_framing(geo, det_config, rec.size, cfg)
            except (ValueError, RuntimeError):
                framing[rec.size] = None            # not a 9:16 frame: nothing to frame it with
        if framing[rec.size] is None:
            dropped["size"] += 1
            continue
        crop, pad = framing[rec.size]
        fp = UH.plan_frame(rec, at.get(n, ()), table=table, known=known, own_labels=own_labels,
                           live_teams=live_teams, config=cfg, gold=gold, crop=crop, pad_px=pad,
                           background=rec.epoch in with_bg)
        counts["no_label"] += fp.counts.get("no_label", 0)
        counts["leftover_masks"] += fp.counts.get("leftover_masks", 0)
        if gold:
            gold_rows.append((rec, fp))
            continue
        birth = n in births
        if not UH.keep_frame(n, fp, birth, cfg):
            continue
        counts["considered"] += 1
        if fp.drop:
            dropped[fp.drop] += 1
            continue
        planned.append((rec, fp, birth, crop, pad))
    # an image of the earlier emit whose raw is gone (D14) would be lost for good: refused
    held = []
    if not gold and (out / IMAGES_DIR).is_dir():
        mine = {f.stem for f in (out / IMAGES_DIR).iterdir() if f.name.startswith(f"{sid}_")}
        held = [rec for rec, *_ in planned
                if f"{sid}_{rec.i:08d}" in mine and not (paths.recording / rec.image).is_file()]
    if held and not drop_missing:
        raise EmitError(f"{len(held)} image(s) of the earlier emit of {sid} have no raw image any more (raws deleted "
                        f"after emit?), e.g. {held[0].image}: emitting again would delete them for good. Keep the emit "
                        "as it is, put the raws back, or pass --drop-missing to emit without them", EXIT_CHECK)
    _clear_emit(out, sid)
    out.mkdir(parents=True, exist_ok=True)

    def background(epoch: str | None) -> Any:
        if epoch not in backgrounds:
            try:
                backgrounds[epoch] = scan.background(epoch) if epoch else None
            except Exception:                   # a damaged background file: those images are dropped
                backgrounds[epoch] = None
        return backgrounds[epoch]

    fh = None
    if not gold:
        (out / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        fh = open(out / BOXES_FILE, "w", encoding="utf-8", newline="\n")
    try:
        for rec, fp, birth, crop, pad in planned:
            try:
                rgb = load_fn(paths.recording / rec.image)
            except Exception:                       # deleted (D14) or unreadable
                dropped["no_image"] += 1
                continue
            if UH.image_size(rgb) != tuple(rec.size):
                dropped["size"] += 1
                continue
            if fp.masks:
                bg = background(rec.epoch)
                if bg is None or UH.image_size(bg) != tuple(rec.size):
                    dropped[UH.DROP_NO_BACKGROUND] += 1
                    continue
                rgb = UH.feather_paste(rgb, bg, fp.masks, pad_px=pad, feather_px=cfg.feather_px)
            img = UH.crop_image(rgb, crop)
            name = f"{sid}_{rec.i:08d}.{cfg.image_format}"
            _assert_trainable(session)
            if out != paths.emit_dir:
                raise EmitError(f"refusing to write a training image outside {paths.emit_dir}", EXIT_CHECK)
            UH.save_image(out / IMAGES_DIR / name, img, cfg.image_format, quality=cfg.quality)
            rows = []
            for b in fp.keep:
                y = UH.to_yolo(b.box, crop, 0.0)
                if y is None:
                    continue
                rows.append([b.label, int(b.side), *(round(v, 6) for v in y), b.src, b.key])
                by_src[b.src] += 1
                if b.how in (UH.HOW_FILL, UH.HOW_CIRCLE):
                    by_src[b.how] += 1
                if b.src in (SOURCE_OWNER, SOURCE_AUTO):
                    by_label[b.label] += 1
            repeat = cfg.birth_repeat if birth and not dev else 1
            fh.write(json.dumps({"image": name, "i": rec.i, "n": rec.n, "t": round(rec.t, 4), "hk": rec.hk,
                                 "frame_size": list(rec.size), "crop": list(crop), "repeat": repeat, "birth": birth,
                                 "boxes": rows, "masks": UH.crop_masks(fp.masks, crop, pad)}) + "\n")
            counts["emitted"] += 1
            counts["birth"] += int(birth)
            counts["masks"] += len(fp.masks)
    finally:
        if fh is not None:
            fh.close()

    gold_ls = None
    if gold:
        order: list[str] = []
        for _rec, fp in gold_rows:
            for b in fp.keep:
                if b.label not in order:
                    order.append(b.label)
        gold_ls = gold_label_set(order, table)
        lab_dir = out / GOLD_LABELS_DIR
        lab_dir.mkdir(parents=True, exist_ok=True)
        with open(out / GOLD_FILE, "w", encoding="utf-8", newline="\n") as gf:
            for rec, fp in gold_rows:
                stem = Path(rec.image).stem
                lines, boxes = [], []
                for b in fp.keep:
                    cx, cy, w, h = UH.frame_yolo(b.box, rec.size)
                    lines.append(f"{gold_ls.yolo_index(b.label, b.side)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    boxes.append({"label": b.label, "side": int(b.side), "box": [round(v, 1) for v in b.box],
                                  "type": b.type, "src": b.src, "how": b.how, "key": b.key})
                    by_src[b.src] += 1
                    if b.how in (UH.HOW_FILL, UH.HOW_CIRCLE):
                        by_src[b.how] += 1
                    if b.src in (SOURCE_OWNER, SOURCE_AUTO):
                        by_label[b.label] += 1
                (lab_dir / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")
                gf.write(json.dumps({"image": rec.image, "stem": stem, "i": rec.i, "n": rec.n, "t": round(rec.t, 4),
                                     "hk": rec.hk, "size": list(rec.size), "boxes": boxes,
                                     "ignore": [[round(v, 1) for v in g] for g in fp.ignore]}) + "\n")
                counts["emitted"] += 1
                counts["masks"] += len(fp.ignore)
        write_json(lab_dir / GOLD_NAMES_FILE, gold_ls.to_dict())
    with open(out / PARTS_FILE, "w", encoding="utf-8", newline="\n") as pf:
        for key in sorted(plans, key=_key_order):
            pf.write(json.dumps(plans[key].to_dict()) + "\n")
    if dropped["no_image"]:
        warnings.append(f"{dropped['no_image']} chosen image(s) had no raw image (deleted?): emit cannot redo them"
                        + (" (--drop-missing: the earlier emit's copies are gone too)" if held else ""))
    if not gold and not counts["emitted"]:
        warnings.append("no image was emitted")
    now = datetime.fromtimestamp(clock()).isoformat(timespec="seconds")
    doc = {
        "schema": EMIT_SCHEMA, "sid": sid, "kind": kind, "role": session.get("role"),
        "capture": session.get("capture"), "client": session.get("client"), "created": now, "scan": identity,
        "tracks_sha1": sha1_file(paths.tracks_file), "sheet": sheet.get("sheet"),
        "bad_tracks": sorted(bad, key=_key_order), "config": asdict(cfg), "geometry": getattr(geo, "name", None),
        "detector_config": asdict(det_config),
        "crops": {f"{w}x{h}": list(v[0]) for (w, h), v in sorted(framing.items()) if v is not None},
        "image_format": None if gold else cfg.image_format,
        "counts": {"records": int(scan.header.get("frames") or len(index)), "battle_images": counts["battle_images"],
                   "considered": counts["considered"], "emitted": counts["emitted"], "birth": counts["birth"],
                   "dropped": {k: dropped[k] for k in (UH.DROP_MASKS, UH.DROP_OVERLAP, UH.DROP_NO_BACKGROUND,
                                                       "no_image", "size")},
                   "boxes": {k: by_src[k] for k in (SOURCE_OWNER, SOURCE_AUTO, UH.SRC_LIVE, UH.HOW_FILL, UH.HOW_CIRCLE)},
                   "masks": counts["masks"], "no_label": counts["no_label"],
                   "leftover_masks": counts["leftover_masks"]},
        "parts": {d: by_decision[d] for d in UH.DECISIONS},
        "labels": dict(sorted(by_label.items())),
        "battle_seconds": {str(hk): round(b - a, 3) for hk, (a, b) in sorted(spans.items())},
        "battle_t0": {str(hk): round(a, 4) for hk, (a, _b) in sorted(spans.items())},
        "gold_names_digest": None if gold_ls is None else gold_ls.digest,
        "base_digest": base_ls.digest, "base_source": "real" if base_real else "preview",
        "pending": pending, "warnings": [scrub_paths(w) for w in warnings], "tool": "tools/vision_autolabel.py emit"}
    write_json(out / EMIT_JSON, doc)          # last: the stage marker, so an interrupted emit is not done
    counts.update({f"dropped_{k}": v for k, v in dropped.items()})
    return EmitResult(sid, kind, out, doc, counts, warnings, True)


# ---------------------------------------------------------------------------
# review, mark, plays (N12)
# ---------------------------------------------------------------------------

REVIEW_PAGE_TILES = 30                     # page A tiles per image, as the sheet
#: unit_harvest's part decisions, restated so mark and plays need no numpy (pinned by a test).
PART_LABEL, PART_OURS = "label", "ours"
#: Yodle2's deck (unit_harvest.DEFAULT_OUR_DECK), restated for plays (pinned by a test).
OUR_DECK = ("HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons", "Fireball", "Log")
GROUP_GOLD, GROUP_AUTO = "gold", "auto"
ROLE_SUMMON = "summon"
PLAY_DERIVED, PLAY_ADDED = "derived", "added"
KIND_UNIT_PLAY, KIND_SPELL_PLAY = "unit", "spell"   # OppPlay.kind
TICKS_PER_S = 20                           # vision_replay --ticks time: 20 ticks a second from the first image
MT_PER_TILE = 1000.0
_EPS_S = 1e-6
#: plays' reach rule: each character's Speed in tiles a minute (60 is a tile a second), RoyaleAPI's
#: table, keyed by the names the vision classes use (AngryBarbarian, SkeletonWarrior, Minion, ...).
SPEEDS_PATH = ROOT / "data" / "royaleapi" / "cards_stats_characters.json"
#: plays' capacity rule: compositions, the units one play of a card deploys (MinionHorde {"Minion": 6}).
SPAWN_CHILDREN_PATH = ROOT / "data" / "vision" / "spawn_children.json"
#: Speeds (tiles a second) of the classes SPEEDS_PATH lacks (the table predates them): the in-game
#: card's Speed (Fast is 90 tiles a minute). The table wins once it has the class.
SPEED_EXTRA = {"BossBandit": 1.5}
SPEED_DEFAULT = 2.0                        # a class in neither: Very Fast (120 a minute), so reach errs long
ARENA_Y_MT = 32_000.0                      # scripts/vision/geometry.HEIGHT_MT (restated: geometry needs numpy)
#: Troop cards placed anywhere on the arena: the deploy-zone rule leaves their units out (spell
#: cards seen as their units, Heal, are left out by their kind).
ZONE_FREE_CARDS = frozenset({"Miner", "GoblinDrill"})
#: Champion cards (tools/rs_presses.CHAMPIONS_BEFORE_THE_COLUMN, restated: the names classes.json
#: uses): one is alive at a time and her card goes back to the cycle only when she dies, so plays'
#: card-cycle rule counts the other plays since her last sighting, not since her deploy.
CHAMPION_CARDS = frozenset({"MightyMiner", "SkeletonKing", "ArcherQueen", "GoldenKnight", "Monk", "LittlePrince",
                            "Goblinstein", "BossBandit"})
#: plays_fresh reasons (status words them; the gate refuses a gold session with any).
PLAYS_MISSING = "no plays.json yet"
PLAYS_TRACKS = "tracks.json changed since the play list"
PLAYS_EMIT = "the gold emit is newer than the play list"
PLAYS_UNCONFIRMED = "the play list is not confirmed"
TRACK_KEY_RE = re.compile(r"^[tT]?(?P<h>\d+)(?:\.(?P<p>\d+))?$")
ADD_PLAY_RE = re.compile(r"^(?:[bB](?P<hk>\d+)/)?(?:(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)|[nN](?P<n>\d+))"
                         r"(?:@(?P<x>-?\d+(?:\.\d+)?),(?P<y>-?\d+(?:\.\d+)?))?=(?P<card>.+)$")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of k in n (95% at z 1.96); (0.0, 1.0) for n == 0. review and
    mark give it for the share of bad tracks and frames a page showed: the reviewed precision
    is 1 minus it, so its interval is (1 - hi, 1 - lo)."""
    k, n = int(k), int(n)
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"wilson: k must be in [0, n], got {k} of {n}")
    p = k / n
    zz = float(z) * float(z)
    d = 1.0 + zz / n
    c = (p + zz / (2 * n)) / d
    h = float(z) * math.sqrt(p * (1.0 - p) / n + zz / (4.0 * n * n)) / d
    return (0.0 if k == 0 else max(0.0, c - h), 1.0 if k == n else min(1.0, c + h))


def share_text(k: int, n: int) -> str:
    """'3 of 24: 12.5% [4.3%-31.0%] (precision 87.5% [69.0%-95.7%])'."""
    lo, hi = wilson(k, n)
    if n <= 0:
        return f"{k} of 0"
    return (f"{k} of {n}: {100.0 * k / n:.1f}% [{100 * lo:.1f}%-{100 * hi:.1f}%] "
            f"(precision {100.0 * (n - k) / n:.1f}% [{100 * (1 - hi):.1f}%-{100 * (1 - lo):.1f}%])")


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    """A JSON-lines file as a list of objects (blank lines skipped)."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def track_key(text: Any) -> str | None:
    """'t17' / '17' -> '17', 't40.1' -> '40.1' (tracks.json's keys); None for anything else."""
    m = TRACK_KEY_RE.match(str(text).strip())
    if m is None:
        return None
    return str(int(m["h"])) if m["p"] is None else f"{int(m['h'])}.{int(m['p'])}"


def _split_list(text: str | None) -> list[str]:
    return [t.strip() for t in str(text or "").split(",") if t.strip()]


def review_group(entry: Mapping[str, Any] | None, gold: bool, label: str | None = None) -> str:
    """The page A group of a labelled part (D9): gold:<label> in a gold session (every part is
    shown), auto:<label> for an auto-accepted one, s<sheet>c<cluster> for an owner label (the
    cluster it was labelled in), s<sheet>t when it was labelled by its track key alone."""
    e = entry or {}
    lab = e.get("label") or label
    if gold:
        return f"{GROUP_GOLD}:{lab}"
    if e.get("source") == SOURCE_AUTO:
        return f"{GROUP_AUTO}:{lab}"
    if e.get("cluster") is not None:
        return f"s{e.get('sheet')}c{e.get('cluster')}"
    return f"s{e.get('sheet')}t"


def review_seed(sid: str, review: int) -> int:
    """The default seed of review K: the same page is drawn again for the same session and K."""
    return int(hashlib.sha1(f"{sid}:{int(review)}".encode("utf-8")).hexdigest(), 16) % 2 ** 32


def read_review(paths: SessionPaths, review: int | None = None) -> dict[str, Any] | None:
    """The latest review.json, or review_<K>.json; None when there is none; LabelError (1)
    when it is there but cannot be read."""
    p = paths.review_dir / (REVIEW_FILE if review is None else f"review_{int(review)}.json")
    if not p.is_file():
        return None
    try:
        doc = _read_json(p)
    except (OSError, ValueError) as exc:
        raise LabelError(f"{p} cannot be read ({exc})", EXIT_CHECK) from None
    if not isinstance(doc, Mapping) or doc.get("schema") != REVIEW_SCHEMA:
        raise LabelError(f"{p} is not a {REVIEW_SCHEMA} document", EXIT_CHECK)
    return dict(doc)


def next_review_number(paths: SessionPaths, latest: Mapping[str, Any] | None = None) -> int:
    """The number of the next review: one more than any review the store's latest review.json
    or the committed marks.json knows (its tallies and marks), so a deleted review folder
    (gitignored, large) never reuses a number marks.json already holds."""
    seen = [int((latest or {}).get("review") or 0)]
    try:
        marks = read_marks(paths) or {}
    except LabelError:
        marks = {}
    for r in marks.get("reviews") or ():
        if isinstance(r, Mapping):
            seen.append(_int_or_zero(r.get("review")))
    for group in ("bad_tracks", "bad_frames"):
        for v in (marks.get(group) or {}).values():
            if isinstance(v, Mapping):
                seen.append(_int_or_zero(v.get("review")))
    return max(seen) + 1


def _int_or_zero(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def review_notes(paths: SessionPaths) -> list[str]:
    """status notes on review and mark (N12): the emit is newer than the latest review, or the
    latest review is not recorded in marks.json yet (its precision needs `mark --review K`)."""
    try:
        rv = read_review(paths)
        marks = read_marks(paths) if rv is not None else None
    except LabelError as exc:
        return list(exc.problems)
    if rv is None:
        return []
    k, sid, notes = rv.get("review"), paths.sid, []
    created = None
    for d in (paths.emit_dir, paths.gold_dir):
        try:
            created = created or _read_json(d / EMIT_JSON).get("created")
        except (OSError, ValueError, AttributeError):
            continue
    if created is not None and rv.get("emit_created") != created:
        notes.append(f"the emit is newer than review {k}: review {sid}")
    elif k not in {r.get("review") for r in (marks or {}).get("reviews") or () if isinstance(r, Mapping)}:
        notes.append(f"review {k} not recorded: mark {sid} --review {k} [--bad-tracks ...] [--bad-frames ...] "
                     "(also when nothing is wrong: the precision needs it)")
    return notes


def bad_frame_numbers(marks_doc: Mapping[str, Any] | None) -> set[int]:
    """The recorder frame numbers (i) of the frames marked bad: mix leaves them out of train and
    val, the gate out of detection scoring. The number outlives a re-emit in another format."""
    out: set[int] = set()
    for _name, m in ((marks_doc or {}).get("bad_frames") or {}).items():
        if isinstance(m, Mapping) and m.get("i") is not None:
            out.add(int(m["i"]))
    return out


def _fresh_emit(paths: SessionPaths, session: Mapping[str, Any], what: str) -> tuple[Path, dict[str, Any], dict[str, dict]]:
    """(emit folder, emit.json, parts.jsonl by key) of a fresh emit; LabelError (1) when the emit
    is missing, stale (emit_fresh) or unreadable."""
    stale = emit_fresh(paths, session)
    if stale:
        run = f"emit {paths.sid}" if stale == ["no emit yet"] else f"emit {paths.sid} --force"
        raise LabelError(f"{what} needs a fresh emit of {paths.sid} ({'; '.join(stale)}): run {run}", EXIT_CHECK)
    out = out_dir_for(paths, session)
    try:
        doc = _read_json(out / EMIT_JSON)
        parts = {str(p["key"]): p for p in read_jsonl(out / PARTS_FILE)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LabelError(f"{out}: the emit's files cannot be read ({exc}): run emit {paths.sid} --force",
                         EXIT_CHECK) from None
    return out, dict(doc), parts


def _battle_t0(emit_doc: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> dict[Any, float]:
    """hk -> the first battle image's t (emit.json battle_t0; else the rows' first)."""
    given: dict[int, float] = {}
    for k, v in (emit_doc.get("battle_t0") or {}).items():
        try:
            given[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    seen: dict[int, float] = {}
    for r in rows:
        if r.get("hk") is not None:
            hk = int(r["hk"])
            seen[hk] = min(seen.get(hk, float(r["t"])), float(r["t"]))
    return {**seen, **given}


def clock_text(seconds: float | None) -> str:
    """92.4 -> '1:32.4' (battle time); '-' for None."""
    if seconds is None:
        return "-"
    s = max(0.0, float(seconds))
    m = int(s // 60)
    return f"{m}:{s - 60 * m:04.1f}"


def _pick_frames(rows: Sequence[Mapping[str, Any]], frames: int, rng: random.Random) -> list[Mapping[str, Any]]:
    """min(frames, len(rows)) rows, half of them with "birth" true when there are enough (the
    other half fills in when one side runs short), in scan order."""
    want = min(max(0, int(frames)), len(rows))
    births = [r for r in rows if r.get("birth")]
    others = [r for r in rows if not r.get("birth")]
    nb = min(len(births), max(want // 2, want - len(others)))
    picked = rng.sample(births, nb) + rng.sample(others, want - nb)
    return sorted(picked, key=lambda r: int(r["n"]))


@dataclass
class ReviewResult:
    sid: str
    review: int
    kind: str                     # "train" (train and dev) or "gold"
    doc: dict[str, Any]           # review.json
    images: list[Path]            # page A image(s), then page B


def review_session(sid: str, *, seed: int | None = None, per_group: int = 3, frames: int = 24, width: int = 1400,
                   cols: int = 4, page_tiles: int = REVIEW_PAGE_TILES, cell: int = 72,
                   labels_root: str | os.PathLike | None = None, store: str | os.PathLike | None = None,
                   load: Any = None, log: Any = print, clock: Any = time.time) -> ReviewResult:
    """The review pages of a session's fresh emit (module doc, review): page A, per_group
    labelled parts per group (gold: every one) at birth, +1 s and mid-life with their
    propagated boxes on the raw image; page B, `frames` emitted frames (gold: battle images)
    with their boxes, half of them in a birth window. Written under <store>/<sid>/review/
    (gitignored; never the emit or gold folder) with review.json; LabelError otherwise."""
    paths = session_paths(sid, labels_root, store)
    try:
        session = load_session(sid, labels_root)
    except (OSError, ValueError) as exc:
        raise LabelError(f"no ingested session {sid} ({exc})") from None
    if int(per_group) < 1 or int(frames) < 0 or int(cols) < 1 or int(page_tiles) < 1 or int(cell) < 16:
        raise LabelError("review: --per-cluster, --cols and --page-tiles must be at least 1, --frames at least 0")
    gold = is_gold(session)
    kind = KIND_GOLD if gold else KIND_TRAIN
    out, emit_doc, parts = _fresh_emit(paths, session, "review")
    try:
        UH = _unit_harvest()
    except ImportError as exc:
        raise LabelError(f"review needs numpy (the buildabot venv): {exc}") from None
    rows_file = out / (GOLD_FILE if gold else BOXES_FILE)
    try:
        rows = read_jsonl(rows_file)
    except (OSError, ValueError) as exc:
        raise LabelError(f"{rows_file} cannot be read ({exc}): run emit {sid} --force", EXIT_CHECK) from None
    tdoc = read_tracks(paths) or {}
    entries = {str(t.get("key")): t for t in tdoc.get("tracks") or () if isinstance(t, Mapping)}
    prev = read_review(paths)
    number = next_review_number(paths, prev)
    seed = review_seed(sid, number) if seed is None else int(seed)
    rng = random.Random(seed)
    load_fn = load or UH.load_image
    raws = {p.stem: p for p in image_files(paths.images)}
    by_n = {int(r["n"]): r for r in rows}
    t0 = _battle_t0(emit_doc, rows)

    def load_or_none(path: Path | None) -> Any:
        if path is None or not path.is_file():
            return None
        try:
            return load_fn(path)
        except Exception:                       # unreadable: shown as gone
            return None

    def raw_of(row: Mapping[str, Any]) -> Path | None:
        if gold:
            return paths.recording / str(row["image"]) if row.get("image") else None
        return raws.get(f"{int(row['i']):08d}") if row.get("i") is not None else None

    # page A: the parts, grouped (D9)
    groups: dict[str, list[str]] = {}
    for key, p in parts.items():
        if p.get("decision") != PART_LABEL or not p.get("label") or not p.get("boxes"):
            continue
        groups.setdefault(review_group(entries.get(key), gold, p.get("label")), []).append(key)
    page_a: list[dict[str, Any]] = []
    for g in sorted(groups):
        keys = sorted(groups[g], key=_key_order)
        shown = keys if gold else sorted(rng.sample(keys, min(int(per_group), len(keys))), key=_key_order)
        labels = Counter(str(parts[k]["label"]) for k in keys)
        page_a.append({"group": g, "keys": shown, "label": min(labels, key=lambda lb: (-labels[lb], lb)),
                       "tracks": len(keys)})
    tiles = []
    for grp in page_a:
        for key in grp["keys"]:
            p = parts[key]
            boxes = {int(b[0]): (tuple(float(v) for v in b[1:5]), str(b[5])) for b in p.get("boxes") or ()}
            have = sorted((n, float(by_n[n]["t"])) for n in boxes if n in by_n)
            picks = UH.crop_frames([SimpleNamespace(n=n, t=t) for n, t in have])
            crops = []
            for f in picks:
                box, how = boxes[f.n]
                colour = UH.DIM_COLOUR if how == UH.HOW_FILL else UH.BOX_COLOUR
                row = by_n[f.n]
                rgb = load_or_none(raw_of(row))
                if rgb is not None:
                    crops.append(UH.sheet_crop(rgb, box, cell, colour=colour))
                    continue
                emitted = None if gold else load_or_none(out / IMAGES_DIR / str(row.get("image")))
                if emitted is not None:
                    x0, y0 = float(row["crop"][0]), float(row["crop"][1])
                    crops.append(UH.sheet_crop(emitted, (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0), cell,
                                               colour=colour))
                else:
                    crops.append(UH.text_tile("RAW GONE", cell))
            while len(crops) < 3:
                crops.append(UH.text_tile("NOT EMITTED" if not have else "-", cell))
            entry = entries.get(key) or {}
            fills = sum(1 for _b, how in boxes.values() if how == UH.HOW_FILL)
            flags = " ".join(str(f) for f in entry.get("flags") or ()) or "-"
            tiles.append(UH.SheetTile(f"t{key} {p.get('label')}",
                                      [f"{grp['group']} {p.get('src')}", f"{flags} {len(boxes)}im {fills}fill"],
                                      [crops[:3]]))
    head_a = [f"{sid} review {number} page A ({kind}): {len(tiles)} of {sum(g['tracks'] for g in page_a)} labelled "
              f"track(s), {len(page_a)} group(s), seed {seed} - birth, +1 s, mid - a dim box is a filled image",
              f"wrong label or box: mark {sid} --review {number} --bad-tracks t17,t40.1"]
    images: list[Path] = []
    per = max(1, int(page_tiles))
    pages = [tiles[i:i + per] for i in range(0, len(tiles), per)] or [[]]
    for j, page in enumerate(pages):
        head = list(head_a)
        if len(pages) > 1:
            head[0] += f", page {j + 1} of {len(pages)}"
        img = UH.render_sheet(page, cell=cell, width=width, header=head, rows=1)
        name = f"review_{number}_a" if len(pages) == 1 else f"review_{number}_a_p{j + 1}"
        images.append(UH.write_sheet_image(paths.review_dir / name, img))

    # page B: frames, half in a birth window
    if gold:
        birth_s = float((emit_doc.get("config") or {}).get("birth_s", 1.0))
        births: set[int] = set()
        for p in parts.values():
            if p.get("decision") != PART_LABEL or int(p.get("part", 0)) != 0:
                continue
            for b in p.get("boxes") or ():
                r = by_n.get(int(b[0]))
                if r is not None and -_EPS_S <= float(r["t"]) - float(p.get("first_t", 0.0)) <= birth_s + _EPS_S:
                    births.add(int(b[0]))
        pool = [dict(r, birth=int(r["n"]) in births) for r in rows]
    else:
        pool = list(rows)
    chosen = _pick_frames(pool, frames, rng)
    page_b: list[dict[str, Any]] = []
    ftiles = []
    for j, r in enumerate(chosen):
        n, hk = int(r["n"]), r.get("hk")
        t_rel = float(r["t"]) - t0.get(None if hk is None else int(hk), float(r["t"]))
        boxes, regions = [], []
        if gold:
            img = load_or_none(raw_of(r))
            for b in r.get("boxes") or ():
                colour = UH.BOX_COLOUR if b.get("src") in (SOURCE_OWNER, SOURCE_AUTO) else UH.PSEUDO_COLOUR
                boxes.append((b["box"], f"{str(b['label'])[:12]} {'E' if int(b['side']) == 1 else 'A'}", colour))
            regions = [(g, UH.FLAG_COLOUR) for g in r.get("ignore") or ()]
            keys = sorted({str(b["key"]) for b in r.get("boxes") or () if b.get("key") is not None}, key=_key_order)
            ident = str(r.get("stem") or Path(str(r.get("image"))).stem)
        else:
            img = load_or_none(out / IMAGES_DIR / str(r["image"]))
            crop = r.get("crop") or (0, 0, 1, 1)
            cw, ch = float(crop[2]) - float(crop[0]), float(crop[3]) - float(crop[1])
            for b in r.get("boxes") or ():
                colour = UH.BOX_COLOUR if b[6] in (SOURCE_OWNER, SOURCE_AUTO) else UH.PSEUDO_COLOUR
                boxes.append((UH.from_yolo(b[2:6], (0.0, 0.0, cw, ch)),
                              f"{str(b[0])[:12]} {'E' if int(b[1]) == 1 else 'A'}", colour))
            regions = [(m, UH.FLAG_COLOUR) for m in r.get("masks") or ()]
            keys = sorted({str(b[7]) for b in r.get("boxes") or () if len(b) > 7 and b[7] is not None}, key=_key_order)
            ident = str(r["image"])
        birth = bool(r.get("birth"))
        page_b.append({"tile": j, "image": ident, "i": r.get("i"), "n": n, "t": r.get("t"), "hk": hk,
                       "birth": birth, "keys": keys})
        ftiles.append(UH.FrameTile(f"f{j} t={clock_text(t_rel)}", img, boxes, regions,
                                   [f"n{n} i{r.get('i')}" + (" birth" if birth else "") + (f" b{hk}" if len(t0) > 1 else "")],
                                   "" if img is not None else "raw deleted"))
    head_b = [f"{sid} review {number} page B ({kind}): {len(page_b)} of {len(rows)} {'battle' if gold else 'emitted'} "
              f"frame(s), {sum(1 for e in page_b if e['birth'])} in a birth window - E enemy, A ours, red: "
              + ("ignore regions" if gold else "masked"),
              f"a wrong frame: mark {sid} --review {number} --bad-frames 3,9 (the f number)"]
    img = UH.render_frames(ftiles, width=width, cols=cols, header=head_b)
    images.append(UH.write_sheet_image(paths.review_dir / f"review_{number}_b", img))

    now = datetime.fromtimestamp(clock()).isoformat(timespec="seconds")
    doc = {"schema": REVIEW_SCHEMA, "sid": sid, "review": number, "kind": kind, "seed": seed, "created": now,
           "emit_created": emit_doc.get("created"), "per_group": int(per_group), "frames": int(frames),
           "page_a": page_a, "page_b": page_b, "images": [im.name for im in images],
           "counts": {"groups": len(page_a), "page_a_tracks": sum(len(g["keys"]) for g in page_a),
                      "labelled_tracks": sum(g["tracks"] for g in page_a), "page_b_frames": len(page_b),
                      "page_b_birth": sum(1 for e in page_b if e["birth"]), "rows": len(rows)},
           "tool": "tools/vision_autolabel.py review"}
    write_json(paths.review_dir / f"review_{number}.json", doc)
    write_json(paths.review_dir / REVIEW_FILE, doc)
    return ReviewResult(sid, number, kind, doc, images)


@dataclass
class MarkResult:
    sid: str
    review: int | None
    doc: dict[str, Any]                         # marks.json
    tally: dict[str, Any] | None                # the review's entry of marks.json "reviews"
    recheck: list[str]
    hints: list[str]
    warnings: list[str]
    stored: Counter
    written: bool


def review_tally(review_doc: Mapping[str, Any], marks_doc: Mapping[str, Any], entries: Mapping[str, Mapping[str, Any]],
                 gold: bool) -> dict[str, Any]:
    """One review's line of marks.json "reviews", from every mark made with it: the parts page A
    showed and how many of them were marked bad, the frames page B showed and how many were
    marked bad, their Wilson intervals (of the bad share), and when more than 2 frames are bad
    (plan 5) the groups of the parts with boxes on them, to check again (recheck)."""
    k = int(review_doc.get("review") or 0)
    shown = {str(key) for g in review_doc.get("page_a") or () for key in g.get("keys") or ()}
    tiles = {str(e.get("image")): e for e in review_doc.get("page_b") or ()}
    bad_t = sorted((key for key, m in (marks_doc.get("bad_tracks") or {}).items()
                    if isinstance(m, Mapping) and m.get("review") == k and key in shown), key=_key_order)
    bad_f = sorted(name for name, m in (marks_doc.get("bad_frames") or {}).items()
                   if isinstance(m, Mapping) and m.get("review") == k and name in tiles)
    wt, wf = wilson(len(bad_t), len(shown)), wilson(len(bad_f), len(tiles))
    groups: dict[str, set[str]] = {}
    if len(bad_f) > 2:
        for name in bad_f:
            for key in tiles[name].get("keys") or ():
                groups.setdefault(review_group(entries.get(str(key)), gold), set()).add(str(key))
    return {"review": k, "emit_created": review_doc.get("emit_created"), "page_a_tracks": len(shown),
            "bad_tracks": len(bad_t), "page_b_frames": len(tiles), "bad_frames": len(bad_f),
            "wilson_tracks": [round(v, 4) for v in wt], "wilson_frames": [round(v, 4) for v in wf],
            "precision_tracks": [round(1.0 - wt[1], 4), round(1.0 - wt[0], 4)],
            "precision_frames": [round(1.0 - wf[1], 4), round(1.0 - wf[0], 4)],
            "recheck": sorted(groups), "recheck_keys": {g: sorted(v, key=_key_order) for g, v in sorted(groups.items())}}


def mark_session(sid: str, *, bad_tracks: str | None = None, bad_frames: str | None = None, review: int | None = None,
                 clear: str | None = None, dry_run: bool = False, labels_root: str | os.PathLike | None = None,
                 store: str | os.PathLike | None = None, clock: Any = time.time) -> MarkResult:
    """Mark what a review showed wrong (module doc, mark): --bad-tracks part keys of tracks.json
    (t17, 40.1), --bad-frames page-B tile numbers of review K (default: the latest), --clear
    marks again; with --review K alone, the review is recorded with nothing bad. Every entry is
    checked first; a bad one raises LabelError (2) and nothing is stored. marks.json keeps keys
    and image names, never tile numbers."""
    paths = session_paths(sid, labels_root, store)
    try:
        session = load_session(sid, labels_root)
    except (OSError, ValueError) as exc:
        raise LabelError(f"no ingested session {sid} ({exc})") from None
    gold = is_gold(session)
    tdoc = read_tracks(paths)
    if tdoc is None:
        raise LabelError(f"{sid} has no labels yet ({paths.tracks_file}): nothing to mark", EXIT_CHECK)
    entries = {str(t.get("key")): t for t in tdoc.get("tracks") or () if isinstance(t, Mapping)}
    old = read_marks(paths)
    problems: list[str] = []
    rv = None
    try:
        rv = read_review(paths, review)
    except LabelError as exc:
        problems += exc.problems
    if review is not None and rv is None and not problems:
        problems.append(f"no review {review} of {sid} ({paths.review_dir}): run review {sid}")
    keys: list[str] = []
    for token in _split_list(bad_tracks):
        key = track_key(token)
        if key is None:
            problems.append(f"--bad-tracks {token!r} is not a track key (t17, t40.1)")
        elif key not in entries:
            pieces = sorted((k for k in entries if k.startswith(f"{key}.")), key=_key_order)
            hint = f" (it was split: {', '.join('t' + k for k in pieces)})" if pieces else " (pending tracks cannot be marked)"
            problems.append(f"t{key} is not a labelled track of {paths.tracks_file.name}{hint}")
        elif key not in keys:
            keys.append(key)
    page_b = {int(e["tile"]): e for e in (rv or {}).get("page_b") or () if e.get("tile") is not None}
    tiles: list[int] = []
    for token in _split_list(bad_frames):
        t = token.lstrip("fF")
        if not t.isdigit():
            problems.append(f"--bad-frames {token!r} is not a page-B tile number (3, f9)")
        elif rv is None:
            problems.append(f"--bad-frames {token}: {sid} has no review to take page-B tiles from: run review {sid}")
        elif int(t) not in page_b:
            span = f"f0-f{max(page_b)}" if page_b else "none"
            problems.append(f"f{int(t)} is not on page B of review {rv.get('review')} (tiles {span})")
        elif int(t) not in tiles:
            tiles.append(int(t))
    marked_t = dict((old or {}).get("bad_tracks") or {})
    marked_f = dict((old or {}).get("bad_frames") or {})
    clears: list[tuple[str, str]] = []
    for token in _split_list(clear):
        key = track_key(token)
        if token in marked_f:
            clears.append(("frame", token))
        elif key is not None and key in marked_t:
            clears.append(("track", key))
        else:
            problems.append(f"--clear {token!r} is neither a marked track (t17) nor a marked frame's image name")
    if not keys and not tiles and not clears and review is None and not problems:
        problems.append("nothing to mark: give --bad-tracks, --bad-frames or --clear, or --review K to record a review "
                        "with nothing bad")
    if problems:
        raise LabelError(problems)

    now = datetime.fromtimestamp(clock()).isoformat(timespec="seconds")
    k = None if rv is None else int(rv.get("review") or 0)
    warnings: list[str] = []
    if rv is not None and (tiles or keys):
        try:
            emitted = _read_json(out_dir_for(paths, session) / EMIT_JSON).get("created")
        except (OSError, ValueError, AttributeError):
            emitted = None
        if emitted is not None and rv.get("emit_created") != emitted:
            warnings.append(f"review {k} was drawn from an older emit ({rv.get('emit_created')}; the emit now is "
                            f"{emitted}): review {sid} again to see the boxes the emit writes now")
    stored: Counter = Counter()
    for key in keys:
        e = entries[key]
        marked_t[key] = {"at": now, "review": k, "label_at": e.get("at"), "label": e.get("label") or e.get("code"),
                         "source": e.get("source")}
        stored["bad_tracks"] += 1
    for t in tiles:
        e = page_b[t]
        marked_f[str(e["image"])] = {"at": now, "review": k, "tile": t, "n": e.get("n"), "i": e.get("i"),
                                     "keys": list(e.get("keys") or ())}
        stored["bad_frames"] += 1
    for what, ident in clears:
        (marked_f if what == "frame" else marked_t).pop(ident, None)
        stored["cleared"] += 1
    doc: dict[str, Any] = {"schema": MARKS_SCHEMA, "sid": sid, "updated": now,
                           "bad_tracks": dict(sorted(marked_t.items(), key=lambda kv: _key_order(kv[0]))),
                           "bad_frames": dict(sorted(marked_f.items())),
                           "reviews": [r for r in (old or {}).get("reviews") or () if isinstance(r, Mapping)],
                           "history": list((old or {}).get("history") or []), "tool": "tools/vision_autolabel.py mark"}
    tally = None
    if rv is not None:
        tally = review_tally(rv, doc, entries, gold)
        doc["reviews"] = sorted([r for r in doc["reviews"] if r.get("review") != k] + [tally],
                                key=lambda r: int(r.get("review") or 0))
    doc["history"].append({"at": now, "review": k, "bad_tracks": bad_tracks or None, "bad_frames": bad_frames or None,
                           "clear": clear or None})
    hints = []
    for g, ks in ((tally or {}).get("recheck_keys") or {}).items():
        hints.append(f'{g}: label {sid} --map "{",".join("t" + key + "=..." for key in ks)}" --force')
    if not dry_run:
        write_json(paths.marks_file, doc)
    return MarkResult(sid, k, doc, tally, list((tally or {}).get("recheck") or ()), hints, warnings, stored, not dry_run)


# -- gold plays ------------------------------------------------------------------

@dataclass(frozen=True)
class PlayConfig:
    """How gold tracks become plays (plays, derive_plays). Times in seconds, distances in tiles."""
    window_s: float = 1.0          # units of one card born this close in time ...
    radius_tiles: float = 3.0      # ... and space (to the play's first unit) are one play (Guards x3, Horde x6)
    reacquire_s: float = 1.5       # a same-card part born this soon after one ended ...
    reacquire_tiles: float = 2.5   # ... this close to where that one was last seen continues it (not a play)
    child_s: float = 10.0          # a spawned unit (BushGoblin, a curse's Goblin) joins its card's play this recent
    # reach (module doc): a unit part born up to reach_gap_s after a same-card play's unit was last seen,
    # within speed x reach_speed_x x the gap + reach_margin_tiles of there, continues it (reach_gap_s 0: off)
    reach_gap_s: float = 12.0
    reach_margin_tiles: float = 2.0
    reach_speed_x: float = 1.25    # the table's speed times this: first-box lag, box noise, an old table
    # zone: a unit part first seen at y >= zone_y_tiles (the river's far bank; ours: y <= 32 - it) starts no
    # play; it continues a same-card unit as reach does, but lost up to zone_gap_s before, else it is a
    # play of its own whose deploy went unseen ("deploy_unseen", dated at that first sighting, as M0 does;
    # zone_y_tiles 32: off)
    zone_y_tiles: float = 17.0
    zone_gap_s: float = 20.0
    # direction: a part born in its owner's deploy zone (short of the river) more than back_tiles behind
    # (toward its king) where a unit was last seen continues no unit of it (units walk forward)
    back_tiles: float = 2.0
    # card cycle: a card comes back to the hand after cycle_plays other plays (8 cards, 4 in hand; a champion's
    # after her death). Once that many other plays came since its card's last play, a part continues a unit
    # lost before (reach, zone, capacity) only on a close fit, a unit seen up to flag_gap_s before within
    # reach_margin_tiles of it (flagged): it may be a new play, so otherwise it is one. The list flags merges
    # made after cycle_plays - 1 (one missed play away) and warns of a play with fewer since (0: off)
    cycle_plays: int = 4
    flag_gap_s: float = 4.0        # the list warns of every zone merge and of reach merges over this gap
    delete_s: float = 0.2          # a stored deletion removes a derived play of its card this close in time
    m0_s: float = 1.0              # the scan's live play matched to a gold play: this close in time ...
    m0_tiles: float = 3.0          # ... and space (plan 6.2's 1 s and 3 tiles)

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not float(value) >= 0.0:
                raise ValueError(f"PlayConfig.{name} must be >= 0")


def _dist_mt(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


_PLAY_TABLES: dict[tuple[str, str], dict] = {}


def load_speeds(path: str | os.PathLike | None = None) -> dict[str, float]:
    """Walking speeds in tiles a second by vision class name, for plays' reach rule: Speed / 60
    of each character in SPEEDS_PATH (tiles a minute; a building's 0 kept), over SPEED_EXTRA for
    the classes it lacks. An unreadable table leaves SPEED_EXTRA (and SPEED_DEFAULT for the rest)."""
    p = SPEEDS_PATH if path is None else Path(path)
    key = ("speeds", str(p))
    if key not in _PLAY_TABLES:
        out = dict(SPEED_EXTRA)
        try:
            rows = _read_json(p)
        except (OSError, ValueError):
            rows = None
        for r in rows if isinstance(rows, list) else ():
            v = r.get("speed") if isinstance(r, Mapping) else None
            if r and r.get("name") and isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
                out[str(r["name"])] = float(v) / 60.0
        _PLAY_TABLES[key] = out
    return dict(_PLAY_TABLES[key])


def load_units_per_play(path: str | os.PathLike | None = None) -> dict[str, int]:
    """Units one play of a card deploys, by card, for plays' capacity rule: the sum of its
    SPAWN_CHILDREN_PATH composition (SkeletonWarriors 3, MinionHorde 6, AngryBarbarians 2,
    HogRider 1). A card with forms (variants: the Spirit Empress) or with no composition has
    no cap. An unreadable table gives no caps."""
    p = SPAWN_CHILDREN_PATH if path is None else Path(path)
    key = ("units", str(p))
    if key not in _PLAY_TABLES:
        out: dict[str, int] = {}
        try:
            doc = _read_json(p)
        except (OSError, ValueError):
            doc = None
        doc = doc if isinstance(doc, Mapping) else {}
        variants = doc.get("variants") if isinstance(doc.get("variants"), Mapping) else {}
        comps = doc.get("compositions") if isinstance(doc.get("compositions"), Mapping) else {}
        for card, units in comps.items():
            if card in variants or not isinstance(units, Mapping):
                continue
            n = sum(int(v) for v in units.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
            if n > 0:
                out[str(card)] = n
        _PLAY_TABLES[key] = out
    return dict(_PLAY_TABLES[key])


def _seen_at(q: Any, t: float) -> tuple[tuple[float, float] | None, float]:
    """Where the unit whose latest part is q was at time t, or last before it, and how long
    before t: its last position and the gap once that part ended; while it still lives at t,
    the point between its birth and last position at t (straight-line), and 0."""
    if q.last_t <= t + _EPS_S:
        return (q.last or q.birth), max(0.0, t - q.last_t)
    if q.birth is None or q.last is None or q.last_t - q.first_t <= _EPS_S:
        return (q.birth or q.last), 0.0
    f = min(1.0, max(0.0, (t - q.first_t) / (q.last_t - q.first_t)))
    return (q.birth[0] + f * (q.last[0] - q.birth[0]), q.birth[1] + f * (q.last[1] - q.birth[1])), 0.0


def starts_play(label: str, deck: Sequence[str], table: Any, known: Iterable[str] = ()) -> bool:
    """Whether a gold track of `label` is a play's unit (else a child of one): a spell label;
    a unit its deck card summons (classes.json "cards": role summon); a unit of a spell whose
    look is its units (no own label on the legend: Heal's HealSpirit, the Spirit Empress
    forms, role spawn). Not a play: a unit its card only spawns (a Suspicious Bush's
    BushGoblin, Goblin Curse's Goblins, Witch's skeletons)."""
    lab = L.resolve_label(table, str(label))
    if lab.type == L.TYPE_SPELL:
        return True
    if lab.type != L.TYPE_UNIT or lab.class_id is None:
        return False
    vc = table.classes[lab.class_id]
    card = table.card_for(lab.class_id, list(deck))
    if card is None:
        return False
    if any(c == card and role == ROLE_SUMMON for c, role in vc.cards):
        return True
    info = table.cards.get(card) or {}
    return bool(info.get("kind") == "spell" and info.get("classes") and spell_label(table, card, known) not in set(known))


def play_cost(card: str | None, labels: Iterable[str], table: Any, event_cards: Mapping[str, Any]) -> int | None:
    """The elixir of a play: its card's (classes.json cards[card].elixir), or for a unit whose
    class's primary card is an internal variant card (event_cards kind internal: the Spirit
    Empress at 3 or 6) that card's."""
    for label in labels:
        lab = L.resolve_label(table, str(label))
        if lab.class_id is None:
            continue
        primary = table.classes[lab.class_id].primary_card
        if primary and primary != card and (event_cards.get(primary) or {}).get("kind") == "internal":
            e = (table.cards.get(primary) or {}).get("elixir")
            if e is not None:
                return int(e)
    e = (table.cards.get(card) or {}).get("elixir") if card else None
    return None if e is None else int(e)


@dataclass
class _GoldPart:
    key: str
    h: int
    part: int
    label: str
    form: int
    spell: bool
    card: str | None
    first_n: int | None
    first_t: float
    last_t: float
    hk: int | None
    uid: str | None
    birth: tuple[float, float] | None
    last: tuple[float, float] | None
    speed: float = SPEED_DEFAULT              # its class's walk, tiles a second (reach)
    # its card is a spell and no card summons its class (Heal's spirits): as a spell. The Spirit Empress'
    # forms, summoned by her variant cards, are troops
    spell_card: bool = False


def _xy(v: Any) -> tuple[float, float] | None:
    try:
        return (float(v[0]), float(v[1])) if v is not None and len(v) >= 2 else None
    except (TypeError, ValueError):
        return None


def derive_plays(parts: Iterable[Mapping[str, Any]], deck: Sequence[str], table: Any, known: Iterable[str] = (), *,
                 config: PlayConfig | None = None, battles: Mapping[int, float] | None = None,
                 event_cards: Mapping[str, Any] | None = None, labels: Mapping[str, str] | None = None,
                 t0: float | None = None, owner_top: bool = True, speeds: Mapping[str, float] | None = None,
                 units_per_play: Mapping[str, int] | None = None,
                 champions: Iterable[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Plays from gold parts (parts.jsonl lines with a label; labels: key -> label instead, for
    our o parts), as EventDeriver's OppPlay has them (card, t / tick, x_mt, y_mt of the first
    unit, kind, cost) plus the units, labels and form. owner_top: the parts' owner deploys from
    the top (the opponent; False for our o parts); "behind" is toward the owner's king, its
    deploy zone its half short of the river, the defender's half the other one. Walked by birth
    (first_t):
      1. a part that continues an earlier one is the same unit, in that one's play: part >= 1
         of a harvest track after its part of the same card (a look change), the same live
         uid and card born within reacquire_s after that unit's latest part ended (never
         overlapping it: a uid may pass to a unit born where one died), or born within
         reacquire_s after a same-card part ended, within reacquire_tiles of where it was last
         seen (a track lost and found again); never (uid, reacquired) a part born in its
         owner's deploy zone for a unit of a deploy-unseen play (rule 6);
      2. a part that does not starts_play is a child: it joins the latest play of its card born
         up to child_s before it, else it is listed alone;
      3. a spell part, or a unit of a spell card that no card summons (Heal's spirits; not the
         Spirit Empress, whose forms her variant cards summon: rules 4-7), joins the play of its
         card born within window_s, in its battle, within radius_tiles of that play's first
         unit; else it starts a play;
      4. a unit part joins, as a unit, such a play (window_s, radius_tiles) that has room: a play
         holds at most the units its card deploys (units_per_play, load_units_per_play: Guards 3,
         Minion Horde 6, Elite Barbarians 2, the others 1); when every such play is full, it is
         a second track on the nearest unit of the nearest one and continues it (capacity);
      5. else (reach) a unit part continues a unit of an earlier same-card play whose latest part
         ended up to reach_gap_s before it, born within reach of where that unit was last seen:
         its class's speed (speeds, load_speeds) x reach_speed_x x the gap + reach_margin_tiles.
         Or it joins such a play as a new unit (one unseen since the deploy) while the play has
         room and is at most reach_gap_s old, born within radius_tiles + its speed x
         reach_speed_x x the time since the play + reach_margin_tiles of the play's first unit.
         Or, the play full and at most reach_gap_s old, it is a second track on a unit still
         seen there, born within reach_margin_tiles of where that unit is then (between its birth
         and last position), and continues it (capacity). Never (the play stays new; "new_by"):
           behind  a part born in its owner's deploy zone more than back_tiles behind where that
                   unit was last seen (radius_tiles + back_tiles behind the play's first unit, for
                   a new unit), and never a unit of a deploy-unseen play (rule 6);
           cycle   once cycle_plays other plays (any card, its battle) came since its card's last
                   play (a champion's: since her last sighting, as her card cycles once she dies;
                   champions, CHAMPION_CARDS): the card is back in the hand, so the part may be a
                   new play. Only a close fit holds then: a unit seen up to flag_gap_s before,
                   within reach_margin_tiles of where it was (a long-lived unit's track lost a
                   moment), listed in "cycled";
         a continuation made once max(1, cycle_plays - 1) other plays came is listed in
         "cycled" too: one play the labels missed (a Zap) would make it a new play;
      6. else a unit part first seen in the defender's half (y >= zone_y_tiles for the top owner,
         y <= 32 - zone_y_tiles for ours; not for ZONE_FREE_CARDS or rule 3's units), where
         no play starts, continues a unit as rule 5 does but lost up to zone_gap_s before (zone;
         a second track on a unit still seen there whatever its play's age), else it is a play
         whose deploy went unseen ("deploy_unseen": dated at that first
         sighting, as M0's deriver dates one: scripts/vision/events.py deploy_zone_mt);
      7. else it starts a play.
    Each play lists its continuations in "continued" (key -> split, uid, reacquired, capacity,
    reach, zone), the units joined after the deploy window in "joined" (key -> reach, zone), and
    in "gaps" (key -> s) how long the unit continued was unseen (0: still seen), or how old the
    play was for a joined unit; "cycled" (key -> other plays since its card's last play) the
    continuations made late in the card cycle: rule-1 ones (uid, reacquired) once cycle_plays
    other plays came (rule 1 holds), rule 5-6 ones as there (the list asks the owner to look).
    A deploy-unseen play's "earlier" names the earlier parts of its
    tracks under another card (its deploy, labelled as something else?). Returns (plays in time
    order, children); a child's "derived" is its play's index."""
    cfg = config or PlayConfig()
    deck = [str(c) for c in deck]
    events = load_event_cards() if event_cards is None else event_cards
    speed_of = load_speeds() if speeds is None else speeds
    cap_of = load_units_per_play() if units_per_play is None else units_per_play
    champs = CHAMPION_CARDS if champions is None else frozenset(champions)
    rows: list[_GoldPart] = []
    for p in parts:
        key = str(p.get("key"))
        label = (labels or {}).get(key) if labels is not None else p.get("label")
        if not label:
            continue
        lab = L.resolve_label(table, str(label))
        cls = table.classes[lab.class_id].name if lab.class_id is not None else None
        card = card_of_label(str(label), deck, table)
        first_t = float(p.get("first_t") or 0.0)
        # a spell card's units keep the spell rules (rule 3) unless a card summons them: Heal's spirits do,
        # the Spirit Empress' forms (summoned by her variant cards; walking, long-lived troops) do not
        spell_card = bool(card) and (table.cards.get(card) or {}).get("kind") == "spell" and (
            lab.class_id is None or not any(role == ROLE_SUMMON for _c, role in table.classes[lab.class_id].cards))
        rows.append(_GoldPart(key, int(p.get("h") if p.get("h") is not None else -1), int(p.get("part") or 0), str(label),
                              int(lab.form or 0), lab.type == L.TYPE_SPELL, card,
                              None if p.get("first_n") is None else int(p["first_n"]), first_t,
                              float(p.get("last_t") if p.get("last_t") is not None else first_t),
                              None if p.get("hk") is None else int(p["hk"]), p.get("uid") or None,
                              _xy(p.get("birth")), _xy(p.get("last")),
                              float(speed_of.get(cls, SPEED_DEFAULT)) if cls else SPEED_DEFAULT, spell_card))
    rows.sort(key=lambda r: (r.first_t, _key_order(r.key)))
    plays: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    where: dict[str, tuple[str, int]] = {}
    latest: dict[str, _GoldPart] = {}           # unit (its first part's key) -> its latest part
    unit_of: dict[str, str] = {}                # part key -> unit
    by_track: dict[tuple[int, int], _GoldPart] = {}
    by_uid: dict[str, str] = {}                 # live uid -> unit
    have = set(known)
    line = cfg.zone_y_tiles * MT_PER_TILE
    zone_on = line < ARENA_Y_MT
    margin = cfg.reach_margin_tiles * MT_PER_TILE

    def depth(xy: tuple[float, float]) -> float:
        """How far into the arena from its owner's back edge (y for the top owner, 32 - y for ours)."""
        return xy[1] if owner_top else ARENA_Y_MT - xy[1]

    def own_zone(xy: tuple[float, float] | None) -> bool:
        """In the owner's deploy zone: its half, short of the river (y < 32 - zone_y_tiles for the top owner)."""
        return zone_on and xy is not None and depth(xy) < ARENA_Y_MT - line

    def behind(xy: tuple[float, float] | None, ref: tuple[float, float] | None, slack_mt: float) -> bool:
        """xy is in the owner's deploy zone more than slack behind ref (toward the owner's king)."""
        return xy is not None and ref is not None and own_zone(xy) and depth(xy) < depth(ref) - slack_mt - 1e-6

    def beyond(r: _GoldPart) -> bool:
        """Rule 6: r's first sighting is in the defender's half and its card is not placed anywhere."""
        if r.birth is None or not zone_on or r.card in ZONE_FREE_CARDS or r.spell_card:
            return False
        return depth(r.birth) >= line - 1e-6

    def since(r: _GoldPart) -> int | None:
        """Plays of other cards (r's battle) since the last play of r's card, or for a champion
        since her last sighting (her card cycles once she dies); None: no play of it yet, or the
        cycle rule off."""
        if cfg.cycle_plays <= 0:
            return None
        last = next((i for i in range(len(plays) - 1, -1, -1)
                     if plays[i]["hk"] == r.hk and plays[i]["card"] == r.card), None)
        if last is None:
            return None
        after = None
        if r.card in champs:
            after = max((min(latest[u].last_t, r.first_t) for u in plays[last]["units"] if u in latest), default=None)
        return sum(1 for pl in plays[last + 1:]
                   if pl["hk"] == r.hk and (after is None or pl["t"] > after + _EPS_S))

    def cycled(r: _GoldPart) -> bool:
        """At least cycle_plays other plays came since r's card last could come back (since)."""
        n = since(r)
        return n is not None and n >= cfg.cycle_plays

    def late(r: _GoldPart) -> int | None:
        """since(r) when a continuation of r is to be listed in "cycled" (rules 5-6): at least
        max(1, cycle_plays - 1) other plays came, so one missed play makes it a new play."""
        n = since(r)
        return n if n is not None and n >= max(1, cfg.cycle_plays - 1) else None

    def add_to(kind: str, idx: int, r: _GoldPart, how: str, gap: float | None = None, joined: bool = False) -> None:
        tgt = plays[idx] if kind == "play" else children[idx]
        tgt["parts"].append(r.key)
        tgt["labels"][r.label] = tgt["labels"].get(r.label, 0) + 1
        tgt["forms"].append(r.form)
        if how:
            tgt["joined" if joined else "continued"][r.key] = how
        if gap is not None:
            tgt["gaps"][r.key] = round(max(0.0, gap), 2)
        if r.uid and r.uid not in tgt["uids"]:
            tgt["uids"].append(r.uid)

    def follow(r: _GoldPart, gap_s: float, zone: bool = False) -> tuple[tuple[int, str | None, float] | None, str]:
        """Rules 5 and 6: (play idx, unit, gap) of the best fit (unit None: a new unit of that
        play) among the unit plays of r's card in its battle, units lost up to gap_s before; and
        why a fit was refused ("behind", "cycle"; "" when none was). A lost unit found again (the
        nearest for its reach) comes first, then a new unit of a play with room, then a second
        track on a unit still seen in a full play (zone: in any play without room, however old,
        as no play starts where r is). Once r's card has cycled, only a close fit (a unit seen
        up to flag_gap_s before, within the margin of where it was) is taken."""
        if r.birth is None or gap_s <= 0.0:
            return None, ""
        cap = cap_of.get(r.card) if r.card else None
        best, close, refused = None, None, ""
        for idx, pl in enumerate(plays):
            if pl["card"] != r.card or pl["hk"] != r.hk or pl["kind"] != KIND_UNIT_PLAY:
                continue
            if pl["deploy_unseen"] and own_zone(r.birth):
                continue                                # back from the defender's half to its deploy zone: no
            age = r.first_t - pl["t"]
            fresh = age <= gap_s + _EPS_S
            room = fresh and (cap is None or len(pl["units"]) < cap)
            for u in pl["units"]:
                q = latest[u]
                pos, gap = _seen_at(q, r.first_t)
                if pos is None:
                    continue
                d = _dist_mt(pos, r.birth) or 0.0
                if q.last_t > r.first_t + _EPS_S:          # still seen: r is this unit only as a second track
                    if (fresh or zone) and not room and d <= margin + 1e-6:
                        e = (2, d / margin if margin > 0 else 0.0, 0.0, idx, _key_order(u))
                        if best is None or e < best[0]:
                            best = (e, (idx, u, 0.0))
                        if close is None or e < close[0]:
                            close = (e, (idx, u, 0.0))
                    continue
                if gap > gap_s + _EPS_S:
                    continue
                lim = (q.speed * cfg.reach_speed_x * gap + cfg.reach_margin_tiles) * MT_PER_TILE
                if d > lim + 1e-6:
                    continue
                if behind(r.birth, pos, cfg.back_tiles * MT_PER_TILE):
                    refused = refused or "behind"
                    continue
                e = (0, d / lim if lim > 0 else 0.0, gap, idx, _key_order(u))
                if best is None or e < best[0]:
                    best = (e, (idx, u, gap))
                if gap <= cfg.flag_gap_s + _EPS_S and d <= margin + 1e-6 and (close is None or e < close[0]):
                    close = (e, (idx, u, gap))
            if room and pl["_birth"] is not None:
                # a unit of the play unseen since its deploy: within the play's spread of where it
                # was put down, plus the walk since
                el = max(0.0, age)
                lim = (cfg.radius_tiles + r.speed * cfg.reach_speed_x * el + cfg.reach_margin_tiles) * MT_PER_TILE
                d = _dist_mt(pl["_birth"], r.birth) or 0.0
                if d <= lim + 1e-6:
                    if behind(r.birth, pl["_birth"], (cfg.radius_tiles + cfg.back_tiles) * MT_PER_TILE):
                        refused = refused or "behind"
                    else:
                        e = (1, d / lim if lim > 0 else 0.0, el, idx, ())
                        if best is None or e < best[0]:
                            best = (e, (idx, None, el))
        if best is not None and cycled(r):
            # the card is back in the hand: a new play, unless the fit is close (a long-lived unit, a tower
            # or building, whose track was lost a moment)
            return (None, "cycle") if close is None else (close[1], "")
        return (None if best is None else best[1]), refused

    for r in rows:
        prev, how, gap0 = None, "", None
        if r.part >= 1:
            q = by_track.get((r.h, r.part - 1))
            if q is not None and q.card == r.card:
                prev, how = q, "split"
        if prev is None and r.uid and r.uid in by_uid:
            # the same live uid, born after that unit's latest part ended and soon after: the
            # tracker may keep a uid for a unit of the same class born where one died
            q = latest[by_uid[r.uid]]
            ukind, uidx = where[by_uid[r.uid]]
            back = ukind == "play" and plays[uidx]["deploy_unseen"] and own_zone(r.birth)   # never back into the zone
            if q.card == r.card and q.hk == r.hk and -_EPS_S <= r.first_t - q.last_t <= cfg.reacquire_s and not back:
                prev, how, gap0 = q, "uid", r.first_t - q.last_t
        if prev is None and r.birth is not None:
            near = []
            for unit, q in latest.items():
                d = _dist_mt(q.last, r.birth)
                gap = r.first_t - q.last_t
                if (q.card == r.card and q.hk == r.hk and d is not None and -_EPS_S <= gap <= cfg.reacquire_s
                        and d <= cfg.reacquire_tiles * MT_PER_TILE):
                    kind, idx = where[unit]
                    if kind == "play" and plays[idx]["deploy_unseen"] and own_zone(r.birth):
                        continue                         # never back into the deploy zone (rule 5)
                    near.append((gap, d, _key_order(unit), q))
            if near:
                gap0, _d, _o, prev = min(near, key=lambda e: e[:3])
                how = "reacquired"
        unit = r.key
        if prev is not None:
            unit = unit_of[prev.key]
            kind, idx = where[unit]
            where[r.key] = (kind, idx)
            add_to(kind, idx, r, how, gap0)
            if how in ("uid", "reacquired") and kind == "play" and cycled(r):
                plays[idx]["cycled"][r.key] = since(r)  # rule 1 holds, but the list asks the owner to look
        elif not starts_play(r.label, deck, table, have):
            cands = [i for i, pl in enumerate(plays) if pl["card"] == r.card and pl["hk"] == r.hk
                     and -_EPS_S <= r.first_t - pl["t"] <= cfg.child_s]
            idx = max(cands, key=lambda i: (plays[i]["t"], i)) if cands else None
            children.append({"key": r.key, "label": r.label, "card": r.card, "t": round(r.first_t, 4), "hk": r.hk,
                             "x_mt": None if r.birth is None else round(r.birth[0], 1),
                             "y_mt": None if r.birth is None else round(r.birth[1], 1), "derived": idx,
                             "parts": [], "labels": {}, "forms": [], "continued": {}, "gaps": {}, "uids": []})
            where[r.key] = ("child", len(children) - 1)
            add_to("child", len(children) - 1, r, "")
            if idx is not None:
                plays[idx]["children"].append(r.key)
        else:
            as_spell = r.spell or r.spell_card
            cap = None if as_spell else cap_of.get(r.card) if r.card else None
            cands, full = [], []
            for i, pl in enumerate(plays):
                if pl["card"] != r.card or pl["hk"] != r.hk or not -_EPS_S <= r.first_t - pl["t"] <= cfg.window_s:
                    continue
                d = _dist_mt(pl["_birth"], r.birth)
                if d is not None and d > cfg.radius_tiles * MT_PER_TILE:
                    continue
                e = (r.first_t - pl["t"], d or 0.0, i)
                (full if cap is not None and len(pl["units"]) >= cap else cands).append(e)
            hit, why, rule, n_late = None, "", "", None
            if not cands and full:
                # capacity: the deploy is full, so this is a second track on its nearest unit
                idx = min(full)[2]
                seen = []
                for u in plays[idx]["units"]:
                    pos, gap = _seen_at(latest[u], r.first_t)
                    seen.append((_dist_mt(pos, r.birth) or 0.0, _key_order(u), u, gap))
                _d, _o, u, gap = min(seen)
                hit, rule = (idx, u, gap), "capacity"
            elif not cands and not as_spell:
                hit, why = follow(r, cfg.reach_gap_s)
                rule = "reach"
                if hit is None and beyond(r):
                    # (after a cycle refusal too: zone may still find a close fit, a unit still seen)
                    hit, why2 = follow(r, cfg.zone_gap_s, zone=True)
                    rule, why = "zone", "cycle" if "cycle" in (why, why2) else (why or why2)
                n_late = late(r) if hit is not None else None
            if cands:
                idx = min(cands)[2]
                plays[idx]["units"].append(r.key)
                where[r.key] = ("play", idx)
                add_to("play", idx, r, "")
            elif hit is not None:
                idx, u, gap = hit
                if u is not None:                          # a later part of unit u
                    unit = u
                    if rule != "capacity" and latest[u].last_t > r.first_t + _EPS_S:
                        rule = "capacity"                  # a second track on a unit still seen
                    where[r.key] = ("play", idx)
                    add_to("play", idx, r, rule, gap)
                else:                                      # a new unit of that play (it has room)
                    plays[idx]["units"].append(r.key)
                    where[r.key] = ("play", idx)
                    add_to("play", idx, r, rule, gap, joined=True)
                if n_late is not None:                     # late in the cycle: the list asks the owner to look
                    plays[idx]["cycled"][r.key] = n_late
            else:
                idx = len(plays)
                plays.append({"t": round(r.first_t, 4), "hk": r.hk, "card": r.card,
                              "kind": KIND_SPELL_PLAY if r.spell else KIND_UNIT_PLAY,
                              "x_mt": None if r.birth is None else round(r.birth[0], 1),
                              "y_mt": None if r.birth is None else round(r.birth[1], 1), "n": r.first_n,
                              "units": [r.key], "parts": [], "labels": {}, "forms": [], "continued": {}, "joined": {},
                              "gaps": {}, "cycled": {}, "uids": [], "children": [], "source": PLAY_DERIVED,
                              "deploy_unseen": not as_spell and beyond(r), "new_by": why or None, "_birth": r.birth})
                where[r.key] = ("play", idx)
                add_to("play", idx, r, "")
        unit_of[r.key] = unit
        if unit not in latest or r.last_t >= latest[unit].last_t:
            latest[unit] = r
        by_track[(r.h, r.part)] = r
        if r.uid:
            by_uid[r.uid] = unit
    row_of = {r.key: r for r in rows}
    for i, pl in enumerate(plays):
        pl.pop("_birth", None)
        pl["derived"] = i
        pl["form"] = max(pl.pop("forms") or [FORM_BASE])
        pl["cost"] = play_cost(pl["card"], pl["labels"], table, events)
        hk = pl["hk"]
        pl["t_rel"] = None if not battles or hk not in battles else round(pl["t"] - float(battles[hk]), 4)
        pl["tick"] = None if t0 is None else int(round((pl["t"] - float(t0)) * TICKS_PER_S))
        pl["off_deck"] = pl["card"] not in deck
        if pl["deploy_unseen"]:
            # the earlier parts of its tracks under another card: its deploy, labelled as something else?
            seen = set(pl["parts"])
            earlier = []
            for k in pl["parts"]:
                q = row_of[k]
                for j in range(q.part):
                    e = by_track.get((q.h, j))
                    if e is not None and e.key not in seen and e.card != pl["card"]:
                        seen.add(e.key)
                        earlier.append({"key": e.key, "label": e.label, "t": round(e.first_t, 4)})
            pl["earlier"] = earlier
    for c in children:
        c["form"] = max(c.pop("forms") or [FORM_BASE])
    return plays, children


def parse_add_play(text: str, battles: Mapping[int, float], resolver: CardResolver, deck: Sequence[DeckCard], *,
                   off_deck: bool = False, frame_t: Mapping[int, tuple[float, int]] | None = None, table: Any = None,
                   known: Iterable[str] = ()) -> tuple[dict[str, Any] | None, str | None]:
    """One --add-play: "1:52=freeze" (m:ss[.s] since its battle's first image), "b2/1:52=freeze"
    (battle 2 of several), "n1234=Vines" (the time of scan record n), with "@x,y" in tiles
    before "=" for a position; the card a RoyaleAPI key or catalogue name (-ev1 / -hero: the
    form), in the declared deck unless off_deck. (play, None) or (None, why not)."""
    s = re.sub(r"\s+", "", str(text))
    m = ADD_PLAY_RE.match(s)
    if m is None:
        return None, (f"{text!r} is not [bK/]M:SS[.S][@X,Y]=CARD or nN[@X,Y]=CARD (1:52=freeze, b2/0:42.5@9,12=zap, "
                      "n1234=Vines)")
    n = None
    if m["n"] is not None:
        if m["hk"] is not None:
            return None, f"{text!r}: n{m['n']} is a scan record of its own battle; drop the b{m['hk']}/"
        n = int(m["n"])
        rec = (frame_t or {}).get(n)
        if rec is None:
            return None, f"{text!r}: n{n} is not a battle image of the gold emit"
        t, hk = float(rec[0]), int(rec[1])
    else:
        sec = float(m["s"])
        if sec >= 60.0:
            return None, f"{text!r}: seconds must be below 60 (m:ss)"
        hks = sorted(int(k) for k in battles)
        if not hks:
            return None, "the gold emit has no battle image to count the time from"
        if m["hk"] is None:
            if len(hks) > 1:
                names = ", ".join(f"b{k}" for k in hks)
                return None, f"{text!r}: this session has {len(hks)} battles ({names}): write b<K>/ first"
            hk = hks[0]
        else:
            hk = int(m["hk"])
            if hk not in hks:
                return None, f"{text!r}: no battle b{hk} (battles {', '.join(f'b{k}' for k in hks)})"
        t = float(battles[hk]) + 60.0 * int(m["m"]) + sec
    dc, why = resolver.resolve(m["card"])
    if why:
        return None, why
    slots = [c for c in deck if c.card == dc.card]
    if not slots and not off_deck:
        return None, (f"{dc.card} is not in the declared deck ({', '.join(c.card for c in deck)}): pass --off-deck "
                      "if it was played all the same")
    info = (table.cards.get(dc.card) if table is not None else None) or {}
    own_look = table is None or spell_label(table, dc.card, known) in set(known)
    spell = info.get("kind") == "spell" and (not info.get("classes") or own_look)
    x = None if m["x"] is None else round(float(m["x"]) * MT_PER_TILE, 1)
    y = None if m["y"] is None else round(float(m["y"]) * MT_PER_TILE, 1)
    return {"t": round(t, 4), "hk": hk, "t_rel": round(t - float(battles.get(hk, t)), 4), "card": dc.card,
            "form": int(dc.form), "cost": None if dc.elixir is None else int(dc.elixir),
            "kind": KIND_SPELL_PLAY if spell else KIND_UNIT_PLAY, "x_mt": x, "y_mt": y, "n": n, "units": [],
            "parts": [], "labels": {}, "continued": {}, "uids": [], "children": [], "source": PLAY_ADDED,
            "spec": str(text).strip(), "off_deck": not slots}, None


def assemble_plays(derived: Sequence[Mapping[str, Any]], children: Sequence[Mapping[str, Any]],
                   deleted: Sequence[Mapping[str, Any]], added: Sequence[Mapping[str, Any]],
                   config: PlayConfig | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """The final list: the derived plays no deletion covers (same card, within delete_s) and the
    added ones, in time order, numbered p1, p2, ...; the children with the id of their play
    (None when it was deleted); and the deletions that covered nothing (stale). An added play of
    a deploy-unseen play's card and battle, up to zone_gap_s before its first sighting (the
    nearest), gives that play's deploy time: it replaces the derived one ("replaces": its t and
    units), which then counts once."""
    cfg = config or PlayConfig()
    used: set[int] = set()
    keep = []
    added = [dict(a) for a in added]
    took: dict[int, int] = {}                     # added index -> the deploy-unseen derived play it replaces
    for j, d in enumerate(derived):
        hit = next((k for k, x in enumerate(deleted) if k not in used and x.get("card") == d.get("card")
                    and abs(float(x.get("t", 0.0)) - float(d["t"])) <= cfg.delete_s + _EPS_S), None)
        if hit is not None:
            used.add(hit)
            continue
        if d.get("deploy_unseen"):
            t = float(d["t"])
            near = [(t - float(a["t"]), k) for k, a in enumerate(added)
                    if k not in took and a.get("card") == d.get("card") and a.get("hk") == d.get("hk")
                    and -cfg.delete_s - _EPS_S <= t - float(a["t"]) <= cfg.zone_gap_s + _EPS_S]
            if near:
                k = min(near, key=lambda e: (abs(e[0]), e[1]))[1]
                took[k] = j
                added[k]["replaces"] = {"t": d["t"], "units": list(d.get("units") or ()), "x_mt": d.get("x_mt"),
                                        "y_mt": d.get("y_mt")}
                continue
        keep.append((float(d["t"]), str(d.get("card")), 0, j, dict(d)))
    keep += [(float(a["t"]), str(a.get("card")), 1, k, a) for k, a in enumerate(added)]
    keep.sort(key=lambda e: e[:4])
    final, ids = [], {}
    for idx, (_t, _c, src, j, p) in enumerate(keep, 1):
        p["id"] = f"p{idx}"
        if src == 0:
            ids[j] = p["id"]
        elif j in took:
            ids[took[j]] = p["id"]                # the replaced play's children are the added one's
        final.append(p)
    kids = [dict(c, play=None if c.get("derived") is None else ids.get(int(c["derived"]))) for c in children]
    for p in final:
        p["children"] = [c["key"] for c in kids if c["play"] is not None and c["play"] == p["id"]]
    return final, kids, [dict(x) for k, x in enumerate(deleted) if k not in used]


def cycle_since(plays: Sequence[Mapping[str, Any]]) -> list[tuple[int, int] | None]:
    """Per play (a list in time order): (other plays since the previous play of its card in its
    battle, that play's index), or None for its card's first play. The card cycle needs 4: a card
    comes back to the hand after 4 other plays (8 cards, 4 in hand)."""
    out: list[tuple[int, int] | None] = []
    last: dict[tuple[Any, Any], int] = {}
    count: dict[Any, int] = {}                    # battle -> plays so far
    at: dict[int, int] = {}                       # play index -> its battle's count before it
    for i, p in enumerate(plays):
        hk = p.get("hk")
        n = count.get(hk, 0)
        at[i] = n
        j = last.get((hk, p.get("card")))
        out.append(None if j is None else (n - at[j] - 1, j))
        last[(hk, p.get("card"))] = i
        count[hk] = n + 1
    return out


def _gap_text(s: Any) -> str:
    return "" if s is None else f" {float(s):.1f} s"


def merges_text(p: Mapping[str, Any]) -> str:
    """A play's merged parts by rule, for the list: "reach t54 3.0 s, t56 3.3 s; zone t106.0 9.5 s;
    unit by reach t21 11.4 s; capacity t197; split t30.1" (the gap: how long the unit continued was
    unseen, or how old the play was for a unit joined later)."""
    gaps = p.get("gaps") or {}
    by: dict[str, list[str]] = {}
    for k, rule in (p.get("continued") or {}).items():
        by.setdefault(str(rule), []).append(f"t{k}" + (_gap_text(gaps.get(k)) if rule in ("reach", "zone") else ""))
    for k, rule in (p.get("joined") or {}).items():
        by.setdefault(f"unit by {rule}", []).append(f"t{k}" + _gap_text(gaps.get(k)))
    order = ["reach", "zone", "unit by reach", "unit by zone", "capacity", "reacquired", "uid", "split"]
    return "; ".join(f"{rule} {', '.join(ks)}" for rule, ks in sorted(by.items(), key=lambda e: (
        order.index(e[0]) if e[0] in order else len(order), e[0])))


def flagged_merges(p: Mapping[str, Any], config: PlayConfig | None = None) -> list[tuple[str, str, float | None]]:
    """The merges of a play the owner should look at: every zone one (a part first seen in the
    defender's half taken for a unit of this play), reach ones over flag_gap_s, and the ones
    made late in the card cycle ("cycled": after its card had cycled, or one play short of it, so
    a play the labels missed would make the part a new play): (key, rule, gap)."""
    cfg = config or PlayConfig()
    gaps = p.get("gaps") or {}
    cyc = p.get("cycled") or {}
    cyc = {str(k): v for k, v in cyc.items()} if isinstance(cyc, Mapping) else {str(k): None for k in cyc}
    out = []
    for src in ("continued", "joined"):
        for k, rule in (p.get(src) or {}).items():
            g = gaps.get(k)
            name = ("unit by " if src == "joined" else "") + str(rule)
            if str(k) in cyc:
                n = cyc[str(k)]
                out.append((str(k), f"{name} after its card had cycled" if n is None or int(n) >= cfg.cycle_plays
                            else f"{name} after {int(n)} other plays", g))
            elif rule == "zone" or (rule == "reach" and g is not None and float(g) > cfg.flag_gap_s + _EPS_S):
                out.append((str(k), name, g))
    return out


def _play_sig(plays: Iterable[Mapping[str, Any]]) -> list[tuple]:
    """What a confirmation is of: each play's card, time, form, source, position and units."""
    return sorted((str(p.get("card")), round(float(p.get("t") or 0.0), 3), p.get("form"), p.get("source"),
                   p.get("x_mt"), p.get("y_mt"), tuple(p.get("units") or ())) for p in plays)


def _hungarian(cost: Sequence[Sequence[float]]) -> list[int]:
    """The least-cost assignment of a square cost matrix (the Hungarian method with potentials,
    O(n^3)): row -> column."""
    n = len(cost)
    inf = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (n + 1)
    p, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [inf] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = [0] * n
    for j in range(1, n + 1):
        out[p[j] - 1] = j - 1
    return out


def assign_pairs(edges: Iterable[tuple[int, float, int, int]]) -> dict[int, tuple[int, int]]:
    """Pairs from candidate edges (stage, cost, i, j): stage by stage (0 first), over the i and j
    no earlier stage took, the most pairs and, among those, the least total cost (the Hungarian
    method per connected group of candidates), so a burst of plays within the window keeps
    every pair a greedy pass would lose. Returns i -> (j, stage). The gate (match_plays) and
    M0's column (match_m0) match with it: same-card pairs as stage 0, any card as stage 1."""
    by_stage: dict[int, dict[tuple[int, int], float]] = {}
    for stage, cost, i, j in edges:
        d = by_stage.setdefault(int(stage), {})
        key = (int(i), int(j))
        d[key] = min(float(cost), d.get(key, float("inf")))
    out: dict[int, tuple[int, int]] = {}
    used_j: set[int] = set()
    for stage in sorted(by_stage):
        live = {k: c for k, c in by_stage[stage].items() if k[0] not in out and k[1] not in used_j}
        parent: dict[tuple[str, int], tuple[str, int]] = {}

        def find(x: tuple[str, int]) -> tuple[str, int]:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, j in live:
            a, b = find(("i", i)), find(("j", j))
            if a != b:
                parent[a] = b
        groups: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for i, j in sorted(live):
            groups.setdefault(find(("i", i)), []).append((i, j))
        for group in groups.values():
            rows = sorted({i for i, _j in group})
            cols = sorted({j for _i, j in group})
            k = max(len(rows), len(cols))
            big = 1.0 + sum(abs(live[e]) for e in group)      # any unmatched row costs more than every pair
            cost = [[big] * k for _ in range(k)]
            for i, j in group:
                cost[rows.index(i)][cols.index(j)] = live[(i, j)]
            for r, c in enumerate(_hungarian(cost)):
                if r < len(rows) and c < len(cols) and (rows[r], cols[c]) in live:
                    out[rows[r]] = (cols[c], stage)
                    used_j.add(cols[c])
    return out


def match_m0(gold: Sequence[Mapping[str, Any]], live: Sequence[Mapping[str, Any]], *, dt_s: float = 1.0,
             tiles: float = 3.0) -> dict[int, int]:
    """gold play index -> live play index: pairs within dt_s (and tiles, when both have a
    position), same-card pairs first, then any card, each stage the most pairs at the least
    |dt| (assign_pairs: the gate's rule), so a right read is not taken by a wrong one nearby."""
    edges = []
    for i, g in enumerate(gold):
        for j, p in enumerate(live):
            dt = abs(float(p["t"]) - float(g["t"]))
            if dt > dt_s + _EPS_S:
                continue
            d = None
            if None not in (g.get("x_mt"), g.get("y_mt"), p.get("x_mt"), p.get("y_mt")):
                d = _dist_mt((g["x_mt"], g["y_mt"]), (p["x_mt"], p["y_mt"]))
                if d > tiles * MT_PER_TILE:
                    continue
            edges.append((0 if p.get("card_name") == g.get("card") else 1, dt + (d or 0.0) * 1e-7, i, j))
    return {i: j for i, (j, _stage) in assign_pairs(edges).items()}


def scan_plays(scan_dir: str | os.PathLike) -> tuple[list[dict[str, Any]], float | None] | None:
    """The live chain's opponent plays in a scan (scan_frames records' "plays": t, n, tick,
    card_name, x_mt, y_mt), and the first record's t (vision_replay's time-tick origin); None
    when the scan is gone (plays then has no M0 column)."""
    p = Path(scan_dir) / "scan_frames.jsonl"
    if not p.is_file():
        return None
    out: list[dict[str, Any]] = []
    t0 = None
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or (t0 is not None and ('"plays":[]' in line or '"plays": []' in line)):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if t0 is None and rec.get("t") is not None:
                t0 = float(rec["t"])
            for pl in rec.get("plays") or ():
                # the play's own tick, not the record it was reported on (after the deriver's waits):
                # both ticks are the live chain's track ticks, 20 a second
                t = float(rec.get("t") or 0.0)
                if isinstance(rec.get("tick"), (int, float)) and isinstance(pl.get("tick"), (int, float)):
                    back = (float(rec["tick"]) - float(pl["tick"])) / TICKS_PER_S
                    if 0.0 <= back <= 30.0:
                        t -= back
                out.append({"t": round(t, 4), "n": rec.get("n"), "tick": pl.get("tick"),
                            "card": pl.get("card"), "card_name": pl.get("card_name"), "x_mt": pl.get("x_mt"),
                            "y_mt": pl.get("y_mt"), "kind": pl.get("kind")})
    return out, t0


def read_plays(paths: SessionPaths) -> dict[str, Any] | None:
    """data/vision/owner_labels/<sid>/plays.json, or None; LabelError (1) when unreadable."""
    if not paths.plays_file.is_file():
        return None
    try:
        doc = _read_json(paths.plays_file)
    except (OSError, ValueError) as exc:
        raise LabelError(f"{paths.plays_file} cannot be read ({exc}); fix or move it away", EXIT_CHECK) from None
    if not isinstance(doc, Mapping) or doc.get("schema") != PLAYS_SCHEMA:
        raise LabelError(f"{paths.plays_file} is not a {PLAYS_SCHEMA} document", EXIT_CHECK)
    return dict(doc)


def plays_fresh(paths: SessionPaths) -> list[str]:
    """Why a gold session's play list cannot be scored ([] when it can): none yet, unreadable,
    tracks.json or the gold emit changed since the list was derived, or not confirmed. The
    gate (N14) scores a gold session only when this is []."""
    try:
        doc = read_plays(paths)
    except LabelError as exc:
        return list(exc.problems)
    if doc is None:
        return [PLAYS_MISSING]
    reasons = []
    if doc.get("tracks_sha1") != sha1_file(paths.tracks_file):
        reasons.append(PLAYS_TRACKS)
    try:
        created = _read_json(paths.gold_dir / EMIT_JSON).get("created")
    except (OSError, ValueError, AttributeError):
        created = None
    if created is None or doc.get("emit_created") != created:
        reasons.append(PLAYS_EMIT)
    if not doc.get("confirmed"):
        reasons.append(PLAYS_UNCONFIRMED)
    return reasons


def _play_checks(final: Sequence[Mapping[str, Any]], battles: Mapping[int, float], cfg: PlayConfig) -> list[str]:
    """What the owner should check before --confirm: deploy-unseen plays, merges from afar
    (flagged_merges) and plays the card cycle says need a play the labels missed (cycle_since)."""
    out: list[tuple[int, int, str]] = []         # (play index, order, text): by play

    def when(p: Mapping[str, Any]) -> tuple[str, str]:
        hk = p.get("hk")
        rel = None if hk not in battles else float(p["t"]) - float(battles[hk])
        pre = f"b{hk}/" if len(battles) > 1 and hk is not None else ""
        return pre, clock_text(rel)

    for i, p in enumerate(final):
        if p.get("source") != PLAY_DERIVED:
            continue
        pid, card = p["id"], p["card"]
        if p.get("deploy_unseen"):
            pre, at_t = when(p)
            at = "" if p.get("x_mt") is None else f" at ({p['x_mt'] / MT_PER_TILE:.1f}, {p['y_mt'] / MT_PER_TILE:.1f})"
            earlier = ", ".join(f"t{e['key']} ({e['label']})" for e in p.get("earlier") or ())
            why = (f"where an earlier {card} unit could have walked, but {int(cfg.cycle_plays)} or more other plays came "
                   f"since its card's last play, so it is taken for a new play" if p.get("new_by") == "cycle" else
                   f"with no earlier {card} unit that could have walked there")
            out.append((i, 0, f"{pid} {card} is a play whose deploy went unseen: first seen {pre}{at_t}{at}, in "
                        f"Yodle2's half, {why}; it is dated at that first sighting, as M0 dates one. "
                        + (f"Its track began earlier as {earlier}: if that is its deploy, label it {card} "
                           "(label --force). " if earlier else "")
                        + f'--add-play "{pre}m:ss={card}" with the deploy time replaces it; --bad-plays {pid} if it '
                        "is no play"))
        far = flagged_merges(p, cfg)
        if far:
            cyc = {str(k) for k in p.get("cycled") or ()}
            out.append((i, 1, f"{pid} {card}: check the parts taken for its units: "
                        + ", ".join(f"t{k} ({rule}," + (f"{_gap_text(g)} after the deploy)" if rule.startswith("unit")
                                                       else f"{_gap_text(g)} unseen)") for k, rule, g in far)
                        + f"; if one is a new {card} play, --add-play it"
                        + (f" (a card is back in the hand after {int(cfg.cycle_plays)} other plays: a part merged after "
                           "that may be a new play, and so may one merged a play short of it when the labels missed a "
                           "play between, such as a Zap)" if any(k in cyc for k, _r, _g in far) else "")))
    for i, (p, c) in enumerate(zip(final, cycle_since(final))):
        if c is None or cfg.cycle_plays <= 0 or c[0] >= cfg.cycle_plays:
            continue
        q = final[c[1]]
        out.append((i, 2, f"{p['id']} {p['card']} comes only {c[0]} play(s) after {q['id']} {q['card']}: a card is "
                    f"back in the hand after {int(cfg.cycle_plays)} others, so a play the labels missed lies between "
                    f"them (--add-play it), or {p['id']} is no new play (--bad-plays {p['id']})"))
    return [text for _i, _o, text in sorted(out, key=lambda e: e[:2])]


@dataclass
class PlaysResult:
    sid: str
    list_no: int
    doc: dict[str, Any]                 # plays.json
    plays: list[dict[str, Any]]         # the final list
    children: list[dict[str, Any]]
    own: list[dict[str, Any]]
    warnings: list[str]
    written: bool


def plays_session(sid: str, *, list_no: int | None = None, bad_plays: str | None = None, add_plays: Sequence[str] = (),
                  confirm: bool = False, off_deck: bool = False, config: PlayConfig | None = None, dry_run: bool = False,
                  allow_pending: bool = False,
                  labels_root: str | os.PathLike | None = None, store: str | os.PathLike | None = None,
                  table: Any = None, registry: Registry | None = None, key_map: Mapping[str, str] | None = None,
                  event_cards: Mapping[str, Any] | None = None, clock: Any = time.time) -> PlaysResult:
    """The gold play list of a session (module doc, plays): derived from the fresh gold emit's
    parts (derive_plays, the session's declared deck), less the deletions, plus the additions,
    numbered as list K+1. Edits (--bad-plays, --add-play, --confirm) name the ids of list K and
    need --list K; every entry is checked first, and a bad one raises LabelError (2) and
    nothing is stored. Gold sessions only (2 otherwise); a stale emit is refused (1)."""
    paths = session_paths(sid, labels_root, store)
    try:
        session = load_session(sid, labels_root)
    except (OSError, ValueError) as exc:
        raise LabelError(f"no ingested session {sid} ({exc})") from None
    if not is_gold(session):
        raise LabelError(f"{sid} is not a gold session (role {session.get('role')}, capture {session.get('capture')}): "
                         "plays are derived for gold sessions only (the gate scores them)")
    out, emit_doc, parts = _fresh_emit(paths, session, "plays")
    try:
        gold_rows = read_jsonl(out / GOLD_FILE)
    except (OSError, ValueError) as exc:
        raise LabelError(f"{out / GOLD_FILE} cannot be read ({exc}): run emit {sid} --force", EXIT_CHECK) from None
    old = read_plays(paths)
    cfg = config or PlayConfig()
    table = load_table() if table is None else table
    if registry is None:
        try:
            registry = Registry.load()
        except RegistryError:
            registry = None
    known = known_labels(table, registry)
    events = load_event_cards() if event_cards is None else event_cards
    deck = session_deck(session)
    deck_names = [c.card for c in deck]
    slot_form = {c.card: c.form for c in deck}
    battles = _battle_t0(emit_doc, gold_rows)
    frame_t = {int(r["n"]): (float(r["t"]), int(r["hk"])) for r in gold_rows if r.get("hk") is not None}
    tracks_sha = sha1_file(paths.tracks_file)
    old_list = int((old or {}).get("list") or 0)
    edits = bool(_split_list(bad_plays)) or bool(add_plays) or bool(confirm)
    problems: list[str] = []
    if edits:
        if old is None:
            problems.append(f"no play list of {sid} yet: run plays {sid} first and take the ids from it")
        elif list_no is None:
            problems.append(f"pass --list {old_list}: the ids are those of the list printed last")
        elif int(list_no) != old_list:
            problems.append(f"--list {list_no}: the latest list of {sid} is {old_list}; its ids are the ones to edit")
        elif old.get("tracks_sha1") != tracks_sha or old.get("emit_created") != emit_doc.get("created"):
            problems.append(f"list {old_list} was derived before tracks.json or the gold emit changed: run plays {sid} "
                            "for a new list, then edit that one")
    elif list_no is not None and int(list_no) != old_list:
        problems.append(f"--list {list_no}: the latest list of {sid} is {old_list or 'none'}")
    pending = [str(k) for k in emit_doc.get("pending") or ()]
    if confirm and pending and not allow_pending:
        problems.append(f"the gold emit left {len(pending)} track(s) unlabelled ({', '.join('t' + k for k in pending[:6])}): "
                        "their plays are missing from the list, and a live play there scores as a phantom. Label them, "
                        f"emit {sid} --force and run plays again, or pass --allow-pending to confirm all the same")
    old_final = {str(p.get("id")): p for p in (old or {}).get("plays") or () if isinstance(p, Mapping)}
    ids: list[str] = []
    for token in _split_list(bad_plays):
        pid = "p" + token.lstrip("pP")
        if not token.lstrip("pP").isdigit():
            problems.append(f"--bad-plays {token!r} is not a play id (p3)")
        elif old is not None and pid not in old_final:
            problems.append(f"{pid} is not on list {old_list} (p1-p{len(old_final)})")
        elif pid not in ids:
            ids.append(pid)
    new_added: list[dict[str, Any]] = []
    if add_plays:
        km = load_key_map()[0] if key_map is None else key_map
        resolver = CardResolver(table, km, events)
        for spec in add_plays:
            play, why = parse_add_play(spec, battles, resolver, deck, off_deck=off_deck, frame_t=frame_t, table=table,
                                       known=known)
            if why:
                problems.append(f"--add-play {spec!r}: {why}")
            else:
                new_added.append(play)
    if problems:
        raise LabelError(problems)

    scan = scan_plays(paths.scan_dir)
    live, t0 = (None, None) if scan is None else scan
    label_parts = [p for p in parts.values() if p.get("decision") == PART_LABEL]
    derived, children = derive_plays(label_parts, deck_names, table, known, config=cfg, battles=battles,
                                     event_cards=events, t0=t0)
    if edits and old is not None and _play_sig(derived) != _play_sig(old.get("derived") or ()):
        raise LabelError(f"the plays derived now differ from list {old_list}'s (the label set, the event cards, "
                         "--window-s / --radius-tiles or another play rule changed since): run plays "
                         f"{sid} without edits, check the new list, then edit or confirm that one", EXIT_CHECK)
    # our own units the sheet showed (o): their ally label is gold.jsonl's, the deck Yodle2's
    own_votes: dict[str, Counter] = {}
    for r in gold_rows:
        for b in r.get("boxes") or ():
            if b.get("key") is not None and int(b.get("side", 1)) == 0:
                own_votes.setdefault(str(b["key"]), Counter())[str(b["label"])] += 1
    own_labels = {k: min(v, key=lambda lb: (-v[lb], lb)) for k, v in own_votes.items()}
    own_parts = [p for p in parts.values() if p.get("decision") == PART_OURS]
    own, _own_children = derive_plays(own_parts, list(OUR_DECK), table, known, config=cfg, battles=battles,
                                      event_cards=events, labels=own_labels, t0=t0, owner_top=False)
    now = datetime.fromtimestamp(clock()).isoformat(timespec="seconds")
    deleted = [dict(d) for d in (old or {}).get("deleted") or () if isinstance(d, Mapping)]
    added = [dict(a) for a in (old or {}).get("added") or () if isinstance(a, Mapping)]
    next_aid = 1 + max((int(str(a.get("aid"))[1:]) for a in added if re.fullmatch(r"a\d+", str(a.get("aid")))),
                       default=0)
    for pid in ids:
        p = old_final[pid]
        if p.get("source") == PLAY_ADDED:
            added = [a for a in added if a.get("aid") != p.get("aid")]
        else:
            deleted.append({"card": p.get("card"), "t": p.get("t"), "x_mt": p.get("x_mt"), "y_mt": p.get("y_mt"),
                            "id": pid, "list": old_list, "at": now})
    for a in new_added:
        added.append(dict(a, aid=f"a{next_aid}", at=now, list=old_list + 1))
        next_aid += 1
    final, kids, stale_del = assemble_plays(derived, children, deleted, added, cfg)
    frame_i = {int(r["n"]): r.get("i") for r in gold_rows}
    for p in final + own:
        p["i"] = None if p.get("n") is None else frame_i.get(int(p["n"]))
    for p in final:
        if p.get("source") == PLAY_ADDED:          # the time tick, as derive_plays gives a derived play
            p["tick"] = None if t0 is None else int(round((float(p["t"]) - float(t0)) * TICKS_PER_S))
    warnings: list[str] = []
    if pending:
        warnings.append(f"{len(pending)} track(s) were not labelled (emit --allow-pending): their plays are missing "
                        "from this list")
    if live is not None:
        hit = match_m0(final, live, dt_s=cfg.m0_s, tiles=cfg.m0_tiles)
        for i, p in enumerate(final):
            p["m0"] = live[hit[i]].get("card_name") if i in hit else None
        m0 = {"matched": sum(1 for i, p in enumerate(final) if i in hit and p["m0"] == p.get("card")),
              "read": len(hit), "of": len(final), "live": len(live), "phantoms": len(live) - len(hit)}
    else:
        m0 = None
        warnings.append("the scan is gone: no M0 column (what tonight's model read)")
    for p in final:
        if p.get("off_deck") and p.get("source") == PLAY_DERIVED:
            warnings.append(f"{p['id']} {p['card']} is not in the declared deck: relabel its tracks "
                            f"({', '.join('t' + k for k in p.get('units') or ())}) with label --force, or --bad-plays it")
        elif p.get("card") in slot_form and int(p.get("form") or 0) > FORM_BASE and slot_form[p["card"]] == FORM_BASE:
            warnings.append(f"{p['id']} {p['card']} is read as {FORM_NAMES[int(p['form'])]}, but its deck slot is base: "
                            "check the form of its tracks")
        if p.get("source") == PLAY_ADDED:
            twin = [q for q in final if q is not p and q.get("card") == p.get("card")
                    and abs(float(q["t"]) - float(p["t"])) <= cfg.window_s]
            if twin:
                warnings.append(f"{p['id']} (added) is within {cfg.window_s:g} s of {twin[0]['id']}, a {p['card']} play "
                                "already on the list")
    for d in stale_del:
        warnings.append(f"the deletion of {d.get('card')} at t {d.get('t')} (list {d.get('list')}) covers no derived "
                        "play now")
    if confirm:
        confirmed, confirmed_at = True, now
    elif old and old.get("confirmed") and _play_sig(final) == _play_sig(old.get("plays") or ()):
        confirmed, confirmed_at = True, old.get("confirmed_at")
    else:
        confirmed, confirmed_at = False, None
        if old and old.get("confirmed"):
            warnings.append(f"the play list changed since list {old_list} was confirmed: check it and --confirm again")
    if not confirmed:
        warnings += _play_checks(final, battles, cfg)
    history = list((old or {}).get("history") or [])
    history.append({"at": now, "list": old_list + 1, "bad_plays": bad_plays or None, "add_plays": list(add_plays) or None,
                    "confirm": bool(confirm), "plays": len(final)})
    doc = {"schema": PLAYS_SCHEMA, "sid": sid, "list": old_list + 1, "tracks_sha1": tracks_sha,
           "emit_created": emit_doc.get("created"), "deck": [c.given for c in deck], "config": asdict(cfg),
           "battles": {str(k): v for k, v in sorted(battles.items())}, "t0": t0,
           "plays": final, "derived": derived, "children": kids, "deleted": deleted, "added": added, "own": own,
           "confirmed": confirmed, "confirmed_at": confirmed_at, "m0": m0, "pending": pending,
           "warnings": [scrub_paths(w) for w in warnings], "history": history, "updated": now,
           "tool": "tools/vision_autolabel.py plays"}
    if not dry_run:
        write_json(paths.plays_file, doc)
    return PlaysResult(sid, old_list + 1, doc, final, kids, own, warnings, not dry_run)


# ---------------------------------------------------------------------------
# mix (N13)
# ---------------------------------------------------------------------------

MIX_YAML, MIX_JSON = "cr_mix.yaml", "mix.json"
#: Next to the yaml: scripts/train_detector.py (_label_set_digest) reads its "digest" into the run's
#: detector_manifest.json, and the gate finds mix.json beside the manifest's "data".
MIX_LABELS = "detector_labels.json"
MIX_TRAIN, MIX_VAL = "train", "val"
#: What a mix folder holds; `mix --force` removes these and nothing else.
MIX_OWN = (MIX_TRAIN, MIX_VAL, MIX_YAML, MIX_LABELS, MIX_JSON)
KATACR_YAML = "cr.yaml"                    # prepare_yolo_dataset's yaml in KATACR_ROOT
#: Image suffixes ultralytics 8.4.192 trains on (data/utils.py IMG_FORMATS), restated: an emit
#: in another format (npy) would put labels in the mix whose images training never reads.
YOLO_IMAGE_SUFFIXES = (".avif", ".bmp", ".dng", ".heic", ".heif", ".jp2", ".jpeg", ".jpg", ".mpo", ".png",
                       ".tif", ".tiff", ".webp")
REPEAT_SEP = "__r"                         # <stem>__r1.jpg: the second copy of an oversampled image
SKIP_BAD_FRAME, SKIP_MISSING, SKIP_DUPLICATE = "bad_frame", "missing", "duplicate"
#: Why mix leaves an emitted image out (mix.json "skipped"): marked bad (mark --bad-frames), a
#: yolo_rows problem (an image with two kinds counts under each), no image file, a repeated name.
MIX_SKIPS = (SKIP_BAD_FRAME, YOLO_NO_INDEX, YOLO_CLIENT, YOLO_SIDE, SKIP_MISSING, SKIP_DUPLICATE)
#: Plain YAML 1.1 words PyYAML (ultralytics' loader) reads as a bool or null, not a name.
_YAML_WORDS = frozenset({"y", "n", "yes", "no", "true", "false", "on", "off", "null", "~"})
_BASE_KEY_RE = re.compile(r'"base"\s*:\s*(?=\{)')


class MixError(LabelError):
    """mix refused (code 1: gold, a stale emit, an unpinned registry, a mix there already) or bad
    input (code 2: no such session, no KataCR dataset or base label set); nothing is written."""


@dataclass
class MixResult:
    name: str
    out: Path
    yaml: Path
    doc: dict[str, Any]           # mix.json
    counts: Counter
    warnings: list[str]
    command: str                  # the train_detector command line for this mix


def pin_base(registry_path: str | os.PathLike, base: L.LabelSet) -> tuple[bool, str]:
    """Pin `base`, tonight's real label set, in the registry file's base.digest, once (D4): from
    then on Registry.extend refuses any other base, so the appended indices cannot move. Only
    that value is rewritten, by a text edit inside the "base" object (the file keeps one entry
    per line, its line ends and any BOM); base.labels and base.yolo_classes follow the base when
    they differ. (False, why) when the file is pinned to this base already. RegistryError, with
    nothing written, when it is pinned to another base (appending to it would move every new
    index), when the pinned registry could not extend this base, or when the edit cannot be
    made safely (the message then says the edit to make by hand)."""
    p = Path(registry_path)
    by_hand = (f'edit {p.name} by hand: in its "base" object set "digest": "{base.digest}" '
               f'("labels": {len(base)}, "yolo_classes": {len(base.yolo_names())})')
    try:
        raw = p.read_bytes()
        text = raw.decode("utf-8-sig")
        doc = json.loads(text)
    except FileNotFoundError:
        raise RegistryError(f"no registry at {p}") from None
    except (OSError, ValueError) as exc:
        raise RegistryError(f"{p} cannot be read ({exc})") from None
    registry = Registry.from_dict(doc, p)
    pinned = registry.base_pin.get("digest")
    if pinned == base.digest:
        return False, f"{p.name} is pinned to this base already ({base.digest})"
    if pinned:
        raise RegistryError(f"the registry is pinned to another base ({pinned}; this one is {base.digest}): "
                            "appending to it would move every new index")
    m = _BASE_KEY_RE.search(text)
    if m is None or not isinstance(doc.get("base"), Mapping):
        raise RegistryError(f'{p} has no "base" object to pin: {by_hand}')
    try:
        obj, end = json.JSONDecoder().raw_decode(text, m.end())
    except ValueError:
        raise RegistryError(f'{p}: its "base" object does not read: {by_hand}') from None
    if obj != doc["base"]:
        raise RegistryError(f'{p}: the first "base" object is not the registry\'s: {by_hand}')
    span, n = re.subn(r'("digest"\s*:\s*)(?:null|"")', lambda mm: mm.group(1) + json.dumps(base.digest),
                      text[m.end():end], count=1)
    if n != 1:
        raise RegistryError(f'{p}: no "digest": null in its "base" object: {by_hand}')
    want = json.loads(text)
    want["base"]["digest"] = base.digest
    notes = []
    for key, value in (("labels", len(base)), ("yolo_classes", len(base.yolo_names()))):
        if key not in obj or obj[key] == value:
            continue
        span, n = re.subn(rf'("{key}"\s*:\s*)(?:-?\d+|null)', lambda mm: mm.group(1) + str(value), span, count=1)
        if n != 1:
            raise RegistryError(f'{p}: base.{key} is not a number to update: {by_hand}')
        want["base"][key] = value
        notes.append(f"base.{key} {obj[key]} -> {value}")
    new_text = text[:m.end()] + span + text[end:]
    try:
        new_doc = json.loads(new_text)
    except ValueError:
        raise RegistryError(f"{p}: the edit would not read back: {by_hand}") from None
    if new_doc != want:
        raise RegistryError(f"{p}: the edit would change more than the base: {by_hand}")
    Registry.from_dict(new_doc, p).extend(base, require_pin=True)        # RegistryError: nothing written
    data = new_text.encode("utf-8")
    if raw.startswith(b"\xef\xbb\xbf"):
        data = b"\xef\xbb\xbf" + data
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    if Registry.load(p).base_pin.get("digest") != base.digest:
        raise RegistryError(f"{p}: the pinned digest does not read back: {by_hand}")
    return True, (f"pinned base digest {base.digest} ({len(base)} labels, {len(base.yolo_names())} YOLO classes) "
                  f"in {p.name}" + (f"; {', '.join(notes)}" if notes else ""))


def name_collisions(names: Sequence[str]) -> list[str]:
    """YOLO names equal under strip().lower(), the key ultralytics' class-head remap matches by
    (nn/tasks.py _remap_cls_by_names, D15): two such names would get one name's learned row."""
    seen: dict[str, int] = {}
    out = []
    for i, name in enumerate(names):
        key = str(name).strip().lower()
        if key in seen:
            j = seen[key]
            out.append(f"YOLO classes {j} {names[j]!r} and {i} {name!r} are one name to ultralytics' class remap "
                       "(compared after strip and lower)")
        else:
            seen[key] = i
    return out


def _yaml_value(text: str) -> Any:
    """A `key: value` yaml value as mix and prepare_yolo_dataset write it: a JSON string or list,
    a single-quoted string, else the plain text (a trailing # comment cut)."""
    s = str(text).strip()
    if s[:1] in ('"', "["):
        try:
            return json.loads(s)
        except ValueError:
            pass
    if len(s) >= 2 and s[0] == s[-1] == "'":
        return s[1:-1].replace("''", "'")
    return re.sub(r"\s+#.*$", "", s)


def read_dataset_yaml(path: str | os.PathLike) -> dict[str, Any]:
    """path, train, val (a string or a list) and names (a list in index order) of an ultralytics
    dataset yaml as scripts/prepare_yolo_dataset.py and mix write it, without a yaml dependency:
    top-level `key: value` lines, and names as a block of `  <index>: <name>` (or `- <name>`)
    lines or a [...] list. ValueError when the names skip an index."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    out: dict[str, Any] = {}
    names: dict[int, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#") or line[:1].isspace():
            continue
        m = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$", line)
        if not m:
            continue
        key, rest = m.groups()
        if key != "names":
            out[key] = _yaml_value(rest)
            continue
        if rest:
            value = _yaml_value(rest)
            names = dict(enumerate(str(v) for v in value)) if isinstance(value, list) else {}
            continue
        while i < len(lines) and (not lines[i].strip() or lines[i][:1].isspace()):
            s = lines[i].strip()
            i += 1
            if not s or s.startswith("#"):
                continue
            mm = re.match(r"^(\d+)\s*:\s*(.*)$", s)
            if mm:
                names[int(mm.group(1))] = str(_yaml_value(mm.group(2)))
            elif s.startswith("- "):
                names[len(names)] = str(_yaml_value(s[2:]))
    if sorted(names) != list(range(len(names))):
        raise ValueError(f"{path}: its names skip an index")
    out["names"] = [names[k] for k in range(len(names))]
    return out


def _yaml_name(name: str) -> str:
    """A class name as a YAML scalar: plain when safe (scripts/prepare_yolo_dataset.py's rule),
    else double-quoted; also quoted when PyYAML would read the plain word as a bool or null."""
    plain = re.fullmatch(r"[A-Za-z][A-Za-z0-9_.@-]*", name) and name.lower() not in _YAML_WORDS
    return name if plain else json.dumps(name)


def same_recording(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    """Whether two session.json "recording" blocks are one recording: the same frames.jsonl SHA-1,
    or the same session folder name and creation time (one recording ingested again while it was
    still written, so its SHA-1 moved). mix refuses a session that is one recording with a gold
    session; the gate fails a candidate whose mix.json holds one."""
    a, b = a or {}, b or {}
    if a.get("frames_sha1") and a.get("frames_sha1") == b.get("frames_sha1"):
        return True
    return bool(a.get("name")) and a.get("name") == b.get("name") and a.get("created") == b.get("created")


def _yolo_images(folders: Iterable[Path]) -> int:
    """Image files ultralytics would read under the folders (it globs them recursively)."""
    n = 0
    for folder in folders:
        for _dir, _subdirs, files in os.walk(folder):
            n += sum(1 for f in files if os.path.splitext(f)[1].lower() in YOLO_IMAGE_SUFFIXES)
    return n


def _split_dirs(root: Path, value: Any, default: str) -> list[Path]:
    """A dataset yaml's train or val (a string or a list) as folders: relative ones under root
    (the KataCR folder given, not the yaml's own path:, which names the PC prepare ran on)."""
    values = value if isinstance(value, list) else [value or default]
    out = []
    for v in values:
        q = Path(str(v))
        out.append(q if q.is_absolute() else root / q)
    return out


@dataclass
class _MixSession:
    sid: str
    doc: dict[str, Any]
    paths: SessionPaths
    dev: bool
    emit_doc: dict[str, Any] = field(default_factory=dict)
    marks: dict[str, Any] | None = None
    tracks: dict[str, Any] | None = None
    plan: list[tuple[Path, str, list[str], int]] = field(default_factory=list)   # src, name, lines, copies
    skipped: Counter = field(default_factory=Counter)
    linked: Counter = field(default_factory=Counter)


def mix_sessions(name: str, sids: Sequence[str] | None = None, *, katacr: str | os.PathLike = KATACR_ROOT,
                 out: str | os.PathLike | None = None, base_labels: str | os.PathLike | None = None,
                 labels_root: str | os.PathLike | None = None, store: str | os.PathLike | None = None,
                 registry: Registry | None = None, table: Any = None, repeat: int = 1, force: bool = False,
                 allow_copy: bool = True, image_suffixes: Sequence[str] | None = YOLO_IMAGE_SUFFIXES,
                 clock: Any = time.time) -> MixResult:
    """Build training mix `name` (module doc, mix): KataCR (prepare_yolo_dataset's `katacr`
    folder) plus the emitted train sessions (and dev sessions, as val) under the pinned extended
    label set, in `out` (default tmp/mix/<name>): cr_mix.yaml, detector_labels.json, mix.json,
    train/ and val/ images (hard links of the emitted images) and labels (class indices:
    yolo_rows). sids None: every train and dev session with an emit. Gold is never mixed: a
    named gold session, a session that is one recording with a gold session (same_recording)
    and any source inside a gold folder are refused. Everything is checked before anything is
    written; MixError (code 1 refused, 2 bad input) otherwise. image_suffixes: the emitted image
    suffixes accepted (ultralytics' by default; None accepts any, for tests on npy emits)."""
    bad_input: list[str] = []
    refused: list[str] = []
    name = str(name)
    if not SID_RE.match(name):
        bad_input.append(f"mix name {name!r}: use letters, digits, '_', '-' and '.' (like ft1)")
    try:
        repeat = int(repeat)
    except (TypeError, ValueError):
        repeat = 0
    if repeat < 1:
        bad_input.append("--repeat must be a whole number of 1 or more")
    if bad_input:
        raise MixError(bad_input, EXIT_INPUT)
    katacr = Path(katacr)
    out_dir = Path(out) if out is not None else MIX_ROOT / name
    base_path = Path(base_labels) if base_labels is not None else katacr / MIX_LABELS
    store_root = Path(store or STORE_ROOT)
    labels_dir = Path(labels_root or LABELS_ROOT)

    # where the mix goes: never in the store, the committed labels, KataCR's folder (D16: its label
    # cache) or anywhere else in the repo but the gitignored tmp/
    if _inside(out_dir, store_root) or _inside(store_root, out_dir):
        refused.append(f"--out {out_dir} overlaps the session store {store_root}: a mix lives under tmp/mix")
    if _inside(out_dir, labels_dir) or _inside(labels_dir, out_dir):
        refused.append(f"--out {out_dir} overlaps the committed labels {labels_dir}: a mix holds frames")
    if _inside(out_dir, katacr) or _inside(katacr, out_dir):
        refused.append(f"--out {out_dir} overlaps the KataCR dataset {katacr}: ultralytics would then write the mix's "
                       "label cache over KataCR's (keep mixes under tmp/mix)")
    if _inside(out_dir, ROOT) and not _inside(out_dir, ROOT / "tmp"):
        refused.append(f"--out {out_dir} is in the repo outside tmp/: a mix holds frames and must stay gitignored")
    if _inside(katacr, store_root):
        refused.append(f"--katacr {katacr} is inside the session store: it is prepare_yolo_dataset's output (tmp/yolo)")
    if out_dir.exists():
        held = {p.name for p in out_dir.iterdir()} if out_dir.is_dir() else {out_dir.name}
        foreign = sorted(held - set(MIX_OWN))
        if not out_dir.is_dir() or foreign:
            refused.append(f"{out_dir} holds other files ({', '.join(foreign[:4]) or 'not a folder'}): pick another "
                           "--name or --out")
        elif held and not force:
            refused.append(f"{out_dir} holds a mix already (or an interrupted one): pass --force to build it again "
                           "(only the mix's own files are removed)")

    # the KataCR dataset and the label sets
    table = load_table() if table is None else table
    kyaml = katacr / KATACR_YAML
    ky: dict[str, Any] = {}
    try:
        ky = read_dataset_yaml(kyaml)
    except FileNotFoundError:
        bad_input.append(f"no {kyaml}: run scripts/prepare_yolo_dataset.py (or pass --katacr)")
    except (OSError, ValueError) as exc:
        bad_input.append(f"{kyaml} cannot be read ({exc})")
    ktrain = _split_dirs(katacr, ky.get("train"), "images/train")
    kval = _split_dirs(katacr, ky.get("val"), "images/val")
    for d in ktrain + kval:
        if ky and not d.is_dir():
            bad_input.append(f"the KataCR images {d} are not there (prepare_yolo_dataset's output?)")
    base = ext = None
    try:
        base = load_label_set(base_path)
    except RegistryError as exc:
        bad_input += exc.problems
    if registry is None:
        try:
            registry = Registry.load()
        except RegistryError as exc:
            bad_input += exc.problems
    if base is not None and registry is not None:
        problems = registry.validate(table)
        if problems:
            refused += [f"registry: {p}" for p in problems]
        try:
            ext = registry.extend(base, require_pin=True)
        except RegistryError as exc:
            refused += exc.problems
        if ext is not None:
            refused += name_collisions(ext.yolo_names())
        if ky and ky.get("names") != base.yolo_names():
            refused.append(f"{kyaml} has {len(ky.get('names') or ())} names that are not {base_path.name}'s "
                           f"{len(base.yolo_names())} YOLO names (prepare_yolo_dataset run again since?): the mix needs "
                           "the label set its KataCR labels were written with")

    # the sessions: gold never
    docs: dict[str, dict[str, Any]] = {}
    for sid in session_ids(labels_root):
        try:
            docs[sid] = load_session(sid, labels_root)
        except (OSError, ValueError) as exc:
            refused.append(f"{sid}: session.json cannot be read ({exc}): the gold check needs every session")
    gold = {sid: d for sid, d in docs.items() if is_gold(d)}
    left_out: list[str] = []
    chosen: list[str] = []
    if sids:
        for sid in sids:
            sid = str(sid)
            if sid in chosen:
                continue
            if not SID_RE.match(sid) or not session_paths(sid, labels_root).session_file.is_file():
                bad_input.append(f"no ingested session {sid}")
            elif sid not in docs:
                continue                                   # unreadable: refused above
            elif is_gold(docs[sid]):
                refused.append(f"{sid} is gold (role {docs[sid].get('role')}, capture {docs[sid].get('capture')}): "
                               "gold is never mixed (plan 6)")
            elif docs[sid].get("role") not in (ROLE_TRAIN, ROLE_DEV):
                refused.append(f"{sid} has role {docs[sid].get('role')!r}: mix takes train and dev sessions")
            elif not (session_paths(sid, labels_root, store).emit_dir / EMIT_JSON).is_file():
                refused.append(f"{sid} has no emit yet: emit {sid}")
            else:
                chosen.append(sid)
    else:
        for sid, d in docs.items():
            if is_gold(d):
                left_out.append(f"{sid}: gold, never mixed")
            elif d.get("role") not in (ROLE_TRAIN, ROLE_DEV):
                left_out.append(f"{sid}: role {d.get('role')!r}")
            elif not (session_paths(sid, labels_root, store).emit_dir / EMIT_JSON).is_file():
                left_out.append(f"{sid}: no emit yet (emit {sid})")
            else:
                chosen.append(sid)
        if not chosen and not refused:
            refused.append("nothing to mix: no train or dev session has an emit (ingest, scan, sheet, label, emit)")
    if bad_input:
        raise MixError(bad_input, EXIT_INPUT)

    sessions: list[_MixSession] = []
    for sid in chosen:
        d = docs[sid]
        s = _MixSession(sid, d, session_paths(sid, labels_root, store), d.get("role") == ROLE_DEV)
        for g, gd in gold.items():
            if same_recording(d.get("recording"), gd.get("recording")):
                refused.append(f"{sid} is the same recording as gold session {g} ({(d.get('recording') or {}).get('name')}): "
                               "the same recording is gold as " + g + "; gold is never mixed")
        stale = emit_fresh(s.paths, d)
        if stale:
            refused.append(f"{sid}: its emit no longer holds ({'; '.join(stale)}): emit {sid} --force")
        try:
            s.emit_doc = dict(_read_json(s.paths.emit_dir / EMIT_JSON))
        except (OSError, ValueError, TypeError) as exc:
            refused.append(f"{sid}: {EMIT_JSON} cannot be read ({exc}): emit {sid} --force")
        if s.emit_doc and s.emit_doc.get("kind") != KIND_TRAIN:
            refused.append(f"{s.paths.emit_dir / EMIT_JSON} says kind {s.emit_doc.get('kind')!r}: only a training emit "
                           "is mixed")
        try:
            s.marks = read_marks(s.paths)
            s.tracks = read_tracks(s.paths)
        except LabelError as exc:
            refused += [f"{sid}: {p}" for p in exc.problems]
        sessions.append(s)
    twice: list[str] = []
    for i, a in enumerate(sessions):
        for b in sessions[i + 1:]:
            if same_recording(a.doc.get("recording"), b.doc.get("recording")):
                twice.append(f"{a.sid} and {b.sid} are one recording: its frames are in the mix twice"
                             + (" (as train and as dev: dev no longer measures held-out frames)" if a.dev != b.dev else ""))

    # the thin-form rule: labelled units of the train sessions, per label, where the client may teach it
    track_counts: Counter = Counter()
    if registry is not None:
        for s in sessions:
            if s.dev or s.tracks is None:
                continue
            units = _labelled_units(s.tracks, effective_bad_tracks(s.marks, s.tracks))
            for label, k in units.items():
                if registry.valid_for(label, s.doc.get("client")):
                    track_counts[label] += k

    # every image's lines, before anything is written
    allowed = None if image_suffixes is None else {str(x).lower() for x in image_suffixes}
    gold_dirs = [session_paths(g, labels_root, store).store_dir.resolve() for g in gold]
    gold_dirs += [session_paths(sid, labels_root, store).gold_dir.resolve() for sid in docs]
    problem_text: Counter = Counter()
    labels_seen: set[str] = set()
    boxes_by_label = {MIX_TRAIN: Counter(), MIX_VAL: Counter()}     # per split, unique images (no repeats)
    names_seen: set[str] = set()
    for s in sessions:
        if ext is None or registry is None:
            break                                          # refused above: the label set does not extend
        if not s.emit_doc:
            continue
        client = s.doc.get("client")
        bad_i = bad_frame_numbers(s.marks)
        try:
            rows = read_jsonl(s.paths.emit_dir / BOXES_FILE)
        except (OSError, ValueError) as exc:
            refused.append(f"{s.sid}: {BOXES_FILE} cannot be read ({exc}): emit {s.sid} --force")
            continue
        if allowed is not None:
            wrong = sorted({Path(str(r.get("image"))).suffix.lower() or "(none)" for r in rows} - allowed)
            if wrong:
                refused.append(f"{s.sid}'s emit wrote {', '.join(wrong)} images, which ultralytics does not train on: "
                               f"emit {s.sid} --force --format jpg")
                continue
        for r in rows:
            image = str(r.get("image") or "")
            if not image or Path(image).name != image:
                refused.append(f"{s.sid}: {BOXES_FILE} names an image {image!r} that is not a file name: emit "
                               f"{s.sid} --force")
                break
            if r.get("i") is not None and int(r["i"]) in bad_i:
                s.skipped[SKIP_BAD_FRAME] += 1
                continue
            lines, problems = yolo_rows(r.get("boxes") or (), ext, registry, track_counts, table, client=client)
            if problems:
                for kind in sorted({p.split(":", 1)[0] for p in problems}):
                    s.skipped[kind] += 1
                problem_text.update(set(problems))
                continue
            src = s.paths.emit_dir / IMAGES_DIR / image
            if not src.is_file():
                s.skipped[SKIP_MISSING] += 1
                continue
            if image in names_seen:
                s.skipped[SKIP_DUPLICATE] += 1
                continue
            names_seen.add(image)
            real = src.resolve()
            if any(real == g or g in real.parents for g in gold_dirs):
                refused.append(f"{s.sid}: {src} lies in a gold folder: gold is never mixed")
                break
            copies = 1 if s.dev else max(1, int(r.get("repeat") or 1)) * repeat
            s.plan.append((src, image, lines, copies))
            labels_seen.update(str(b[0]) for b in r.get("boxes") or ())
            for line in lines:
                boxes_by_label[MIX_VAL if s.dev else MIX_TRAIN][ext.decode(int(line.split()[0]))[0]] += 1
    for d in ktrain + kval:
        real = d.resolve()
        if any(real == g or g in real.parents for g in gold_dirs) or _inside(d, store_root):
            refused.append(f"the KataCR images {d} lie in the session store: gold is never mixed")
    if refused or ext is None or base is None or registry is None:
        raise MixError(refused or ["the label set did not load"], EXIT_CHECK)
    if not any(s.plan for s in sessions):
        raise MixError("no emitted image is left to mix (every one was skipped: " + ", ".join(
            f"{v} {k}" for k, v in sum((s.skipped for s in sessions), Counter()).items()) + ")", EXIT_CHECK)

    trains = {label: trains_as(label, ext, registry, int(track_counts.get(label, 0)), table)
              for label in sorted(labels_seen)}
    thin = []                                          # the form decisions, as `mix` prints them
    for label, target in trains.items():
        e, k = registry.get(label), int(track_counts.get(label, 0))
        if e is not None and e.train_as is not None and e.train_as != TRAIN_SELF:
            thin.append(f"{label} trains as {target}: train_as in the registry")
        elif e is not None and e.base is not None and e.train_as is None:
            thin.append(f"{label} trains as {target}: {k} of {registry.thin_tracks} tracks" if target != label else
                        f"{label} trains as itself: {k} tracks ({registry.thin_tracks} needed)")
        elif e is None and target != label:
            thin.append(f"{label} trains as {target}: not in data/vision/detector_extra_labels.json yet")
    warnings = twice + [f"{k} image(s) left out: {p}"
                        for p, k in sorted(problem_text.items(), key=lambda kv: (-kv[1], kv[0]))[:8]]
    for s in sessions:
        made = (s.emit_doc or {}).get("base_digest")
        if base is not None and s.emit_doc and made != base.digest:
            warnings.append(f"{s.sid} was emitted against another base label set ("
                            + (f"{(s.emit_doc or {}).get('base_source')} {made}" if made else "not recorded")
                            + f", the mix's is {base.digest}): its pseudo-labels may name labels this base lacks "
                            f"(those images are left out) or miss ones it has; emit {s.sid} --force")

    # writing starts here: only the mix's own files are replaced
    if out_dir.exists():
        try:
            for item in MIX_OWN:
                q = out_dir / item
                if q.is_dir() and not q.is_symlink():
                    shutil.rmtree(q)
                elif q.exists() or q.is_symlink():
                    q.unlink()
        except OSError as exc:
            raise MixError(f"{out_dir} could not be cleared ({exc}): is a training run reading it? Stop it, then "
                           f"mix --name {name} --force again", EXIT_CHECK) from None
    has_dev = any(s.dev and s.plan for s in sessions)
    for split in (MIX_TRAIN,) + ((MIX_VAL,) if has_dev else ()):
        (out_dir / split / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    files = Counter()
    for s in sessions:
        split = MIX_VAL if s.dev else MIX_TRAIN
        idir, ldir = out_dir / split / IMAGES_DIR, out_dir / split / "labels"
        for src, image, lines, copies in s.plan:
            stem, suffix = os.path.splitext(image)
            text = "".join(line + "\n" for line in lines)
            for k in range(copies):
                fname = image if k == 0 else f"{stem}{REPEAT_SEP}{k}{suffix}"
                try:
                    s.linked[_link(src, idir / fname, allow_copy)] += 1
                except OSError as exc:
                    raise MixError(f"{src} could not be linked into {idir} ({exc}): the mix in {out_dir} is incomplete; "
                                   f"mix --name {name} --force" + (" (without --no-copy)" if not allow_copy else ""),
                                   EXIT_CHECK) from None
                (ldir / f"{os.path.splitext(fname)[0]}.txt").write_text(text, encoding="utf-8")
                files[split] += 1
    copied = sum(s.linked["copied"] for s in sessions)
    if copied:
        warnings.append(f"{copied} image(s) were copied, not hard-linked (the mix is on another drive than the store?): "
                        "they take disk space twice")

    def posix(p: Path) -> str:
        return p.resolve().as_posix()

    train_dirs = [out_dir / MIX_TRAIN / IMAGES_DIR] + ktrain
    val_dirs = kval + ([out_dir / MIX_VAL / IMAGES_DIR] if has_dev else [])
    if files[MIX_TRAIN] and posix(train_dirs[0]) > min(posix(d) for d in ktrain):
        warnings.append(f"{out_dir} sorts after KataCR's images: ultralytics then writes this mix's train label cache "
                        "into KataCR's labels folder (D16; harmless, but the next KataCR-only run rebuilds it): keep "
                        "mixes under tmp/mix")
    yaml_lines = [f"# tools/vision_autolabel.py mix {name}: KataCR ({shown_path(katacr) or katacr.name}) plus "
                  f"{len(sessions)} owner session(s); gold is never here (plan 6)",
                  f"path: {json.dumps(posix(out_dir), ensure_ascii=False)}",
                  f"train: {json.dumps([posix(d) for d in train_dirs], ensure_ascii=False)}",
                  f"val: {json.dumps([posix(d) for d in val_dirs], ensure_ascii=False)}",
                  "names:"] + [f"  {i}: {_yaml_name(n)}" for i, n in enumerate(ext.yolo_names())]
    yaml_path = out_dir / MIX_YAML
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    ldoc = extended_doc(base, registry, ext)
    ldoc["mix"] = {"name": name, "label_set_digest": ext.digest, "sessions": [s.sid for s in sessions],
                   "trains_as": trains, "counts": dict(sorted(track_counts.items()))}
    write_json(out_dir / MIX_LABELS, ldoc)

    kt, kv = _yolo_images(ktrain), _yolo_images(kval)
    share = files[MIX_TRAIN] / (files[MIX_TRAIN] + kt) if files[MIX_TRAIN] + kt else 0.0
    now = datetime.fromtimestamp(clock()).isoformat(timespec="seconds")
    rows_out = []
    for s in sessions:
        rec = s.doc.get("recording") or {}
        rows_out.append({
            "sid": s.sid, "role": s.doc.get("role"), "client": s.doc.get("client"),
            "split": MIX_VAL if s.dev else MIX_TRAIN, "frames_sha1": rec.get("frames_sha1"),
            "recording": {"name": rec.get("name"), "created": rec.get("created"), "frames_sha1": rec.get("frames_sha1")},
            "emit_created": s.emit_doc.get("created"), "images": len(s.plan),
            "files": sum(item[3] for item in s.plan), "linked": s.linked["linked"], "copied": s.linked["copied"],
            "repeats": sum(item[3] - 1 for item in s.plan), "skipped": {k: s.skipped[k] for k in MIX_SKIPS}})
    doc = {
        "schema": MIX_SCHEMA, "name": name, "created": now, "label_set_digest": ext.digest, "base_digest": base.digest,
        "registry_digest": registry.digest, "classes_digest": getattr(table, "digest", None),
        "labels": len(ext), "yolo_classes": len(ext.yolo_names()), "base_yolo_classes": len(base.yolo_names()),
        "katacr": shown_path(katacr) or katacr.name, "katacr_images": {MIX_TRAIN: kt, MIX_VAL: kv},
        "out": shown_path(out_dir), "yaml": MIX_YAML, "repeat": repeat,
        "sessions": rows_out,
        "gold_excluded": [{"sid": g, "frames_sha1": (gd.get("recording") or {}).get("frames_sha1"),
                           "recording": {"name": (gd.get("recording") or {}).get("name"),
                                         "created": (gd.get("recording") or {}).get("created")}}
                          for g, gd in sorted(gold.items())],
        "left_out": left_out, "trains_as": trains, "thin": thin, "track_counts": dict(sorted(track_counts.items())),
        "boxes_by_label": dict(sorted(boxes_by_label[MIX_TRAIN].items())),
        "val_boxes_by_label": dict(sorted(boxes_by_label[MIX_VAL].items())), "owner_share": round(share, 4),
        "counts": {"owner_train_files": files[MIX_TRAIN], "owner_val_files": files[MIX_VAL]},
        "warnings": [scrub_paths(w) for w in warnings], "tool": "tools/vision_autolabel.py mix"}
    write_json(out_dir / MIX_JSON, doc)               # last: an interrupted mix has no mix.json
    shown = shown_path(yaml_path)
    data_arg = shown.replace("/", "\\") if shown else str(yaml_path)
    if " " in data_arg:
        data_arg = f'"{data_arg}"'
    command = (f"& $BB scripts\\train_detector.py --model <pinned best.pt> --data {data_arg} --name {name} "
               "--optimizer AdamW --lr0 0.001 --epochs 40 --no-pin")
    counts = Counter({"train_files": files[MIX_TRAIN], "val_files": files[MIX_VAL], "copied": copied,
                      "katacr_train": kt, "katacr_val": kv})
    return MixResult(name, out_dir, yaml_path, doc, counts, warnings, command)


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
        if args.base or args.out or args.pin_base:
            print(f"error: {exc}" + (" (--pin-base pins the real label set, never the preview)" if args.pin_base else ""))
            return EXIT_INPUT
        print(f"no base label set at {base_path}: the indices below follow a preview built from "
              "classes.json's KataCR names, not the trained set")
        base, preview = crosswalk_base(table), True
    if args.pin_base:
        reg_path = Path(args.registry) if args.registry else REGISTRY_PATH
        try:
            changed, message = pin_base(reg_path, base)
            registry = Registry.load(reg_path)
        except RegistryError as exc:
            _print_problems("pin refused, nothing written:", exc.problems)
            return EXIT_CHECK
        print(message)
        if changed:
            print(f"commit {shown_path(reg_path) or reg_path}: every mix extends this base from now on")
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


def _cmd_emit(args) -> int:
    try:
        table = load_table(args.classes)
        registry = Registry.load(args.registry) if args.registry else None
    except (RegistryError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return EXIT_INPUT
    try:
        UH = _unit_harvest()
    except ImportError as exc:
        print(f"error: emit needs numpy (the buildabot venv): {exc}")
        return EXIT_INPUT
    try:
        cfg = UH.EmitConfig(stride=args.stride, empty_stride=args.empty_stride, birth_s=args.birth_s,
                            birth_repeat=args.birth_repeat, max_masks=args.max_masks, image_format=args.format)
    except ValueError as exc:
        print(f"error: {exc}")
        return EXIT_INPUT
    try:
        res = emit_session(args.sid, labels_root=args.labels_root, store=args.store, table=table, registry=registry,
                           geometry=args.geometry, config=cfg, force=args.force, allow_pending=args.allow_pending,
                           drop_missing=args.drop_missing, dry_run=args.dry_run)
    except LabelError as exc:
        _print_problems("emit refused, nothing written:", exc.problems)
        return exc.code
    doc = res.doc
    parts = doc.get("parts") or {}
    if not res.written:
        print("dry run: nothing written (the propagated boxes per label are above)")
    else:
        print(f"{res.sid} ({res.kind}): {sum(parts.values())} part(s): "
              + ", ".join(f"{v} {k}" for k, v in parts.items() if v))
        c = doc["counts"]
        if res.kind == KIND_GOLD:
            print(f"  {c['emitted']} gold image(s) labelled of {c['battle_images']} battle images "
                  f"({c['masks']} ignore regions); no image written")
        else:
            print(f"  {c['emitted']} image(s) emitted of {c['considered']} chosen ({c['battle_images']} battle images), "
                  f"{c['birth']} in a birth window, {c['masks']} masks")
            drops = {k: v for k, v in c["dropped"].items() if v}
            if drops:
                print("  dropped: " + ", ".join(f"{v} {k}" for k, v in drops.items()))
        print("  boxes: " + ", ".join(f"{v} {k}" for k, v in c["boxes"].items() if v))
        for label, k in sorted((doc.get("labels") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {k:6d}  {label}")
        print(f"  emit -> {res.out_dir}")
    for w in res.warnings:
        print(f"  ! {w}")
    if res.written:
        print(f"next: & $BB tools\\vision_autolabel.py review {res.sid}"
              + (f"   (then: plays {res.sid})" if res.kind == KIND_GOLD else ""))
    return EXIT_OK


def _cmd_review(args) -> int:
    try:
        res = review_session(args.sid, seed=args.seed, per_group=args.per_cluster, frames=args.frames, width=args.width,
                             cols=args.cols, page_tiles=args.page_tiles, labels_root=args.labels_root, store=args.store)
    except LabelError as exc:
        _print_problems("review refused:", exc.problems)
        return exc.code
    d, c, k, sid = res.doc, res.doc["counts"], res.review, res.sid
    print(f"{sid} review {k} ({res.kind}, seed {d['seed']}): page A {c['page_a_tracks']} of {c['labelled_tracks']} "
          f"labelled track(s) in {c['groups']} group(s); page B {c['page_b_frames']} frame(s), {c['page_b_birth']} "
          "in a birth window")
    for g in d["page_a"]:
        print(f"  {g['group']:22s} {str(g['label']):28s} {len(g['keys']):3d} of {g['tracks']:<3d} "
              + " ".join("t" + key for key in g["keys"][:10]) + (" ..." if len(g["keys"]) > 10 else ""))
    for im in res.images:
        print(f"  page -> {im}")
    print(f"  review -> {session_paths(sid, args.labels_root, args.store).review_dir / REVIEW_FILE}")
    print(f"next: & $BB tools\\vision_autolabel.py mark {sid} --review {k} --bad-tracks t17,t40.1 --bad-frames 3,9")
    print(f"      nothing wrong? record it all the same: mark {sid} --review {k}   (the precision interval needs it)")
    if res.kind == KIND_GOLD:
        print(f"then: & $BB tools\\vision_autolabel.py plays {sid}")
    return EXIT_OK


def _cmd_mark(args) -> int:
    try:
        res = mark_session(args.sid, bad_tracks=args.bad_tracks, bad_frames=args.bad_frames, review=args.review,
                           clear=args.clear, dry_run=args.dry_run, labels_root=args.labels_root, store=args.store)
    except LabelError as exc:
        _print_problems("mark refused, nothing stored:", exc.problems)
        return exc.code
    verb = "would mark" if not res.written else "marked"
    print(f"{res.sid}: {verb} {res.stored['bad_tracks']} track(s) and {res.stored['bad_frames']} frame(s) bad, "
          f"{res.stored['cleared']} cleared; {len(res.doc['bad_tracks'])} track(s) and {len(res.doc['bad_frames'])} "
          "frame(s) marked in all")
    t = res.tally
    if t is not None:
        print(f"  review {t['review']}: bad tracks {share_text(t['bad_tracks'], t['page_a_tracks'])}")
        print(f"  {'':{len(str(t['review'])) + 8}s}bad frames {share_text(t['bad_frames'], t['page_b_frames'])}")
    if res.recheck:
        print(f"  more than 2 bad frames: check these groups again ({', '.join(res.recheck)}):")
        for h in res.hints:
            print(f"    {h}")
    for w in res.warnings:
        print(f"  ! {w}")
    if res.written:
        print(f"  marks -> {session_paths(args.sid, args.labels_root, args.store).marks_file}")
    if res.stored["bad_tracks"] or res.stored["cleared"]:
        print(f"next: relabel what was wrong (label {res.sid} --map \"t17=...\" --force), then "
              f"emit {res.sid} --force and review {res.sid}   (a bad track is masked until it is relabelled)")
    elif res.stored["bad_frames"]:
        print("  frame marks need no re-emit: mix and the gate leave those frames out")
    return EXIT_OK


def _cmd_plays(args) -> int:
    try:
        table = load_table(args.classes)
        registry = Registry.load(args.registry) if args.registry else None
        cfg = PlayConfig(window_s=args.window_s, radius_tiles=args.radius_tiles, reach_gap_s=args.reach_gap_s,
                         reach_margin_tiles=args.reach_margin_tiles, zone_y_tiles=args.zone_y_tiles,
                         zone_gap_s=args.zone_gap_s, back_tiles=args.back_tiles, cycle_plays=args.cycle_plays,
                         flag_gap_s=args.flag_gap_s)
    except (RegistryError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return EXIT_INPUT
    try:
        res = plays_session(args.sid, list_no=args.list_no, bad_plays=args.bad_plays, add_plays=args.add_play or (),
                            confirm=args.confirm, off_deck=args.off_deck, config=cfg, dry_run=args.dry_run,
                            allow_pending=args.allow_pending, labels_root=args.labels_root, store=args.store,
                            table=table, registry=registry)
    except LabelError as exc:
        _print_problems("plays refused, nothing stored:", exc.problems)
        return exc.code
    d, sid = res.doc, res.sid
    res_cfg = cfg
    state = "confirmed" if d["confirmed"] else "not confirmed"
    print(f"{sid} play list {res.list_no}{' (dry run: not stored)' if not res.written else ''}: {len(res.plays)} "
          f"play(s), {state}; {len(d['derived'])} derived, {len(d['deleted'])} deleted, {len(d['added'])} added")
    for p in res.plays:
        form = "" if not p.get("form") else f" ({FORM_NAMES[int(p['form'])]})"
        at = "" if p.get("x_mt") is None else f"  at ({p['x_mt'] / MT_PER_TILE:.1f}, {p['y_mt'] / MT_PER_TILE:.1f})"
        m0 = "" if d.get("m0") is None else f"  M0: {p.get('m0') or '-'}"
        units = " ".join("t" + k for k in p.get("units") or ())
        what = f"{len(p.get('units') or ())} unit(s)" if p.get("source") == PLAY_DERIVED else f"added ({p.get('spec')})"
        kids = " ".join("t" + k for k in p.get("children") or ())
        rep = p.get("replaces") or {}
        tags = [t for t, on in (("deploy unseen", p.get("deploy_unseen")),
                                ("new: its card had cycled", p.get("new_by") == "cycle"),
                                ("new: behind the last one's units", p.get("new_by") == "behind"),
                                (f"replaces the deploy unseen at t {rep.get('t')}", bool(rep)))
                if on]
        print(f"  {p['id']:>4}  {clock_text(p.get('t_rel')):>7}  {str(p['card']) + form:26s} {what:16s}{at}{m0}"
              + (f"  [{units}]" if units else "") + (f"  children {kids}" if kids else "")
              + "".join(f"  ({t})" for t in tags))
        merged = merges_text(p)
        if merged:
            far = {k for k, _rule, _g in flagged_merges(p, res_cfg)}
            print(f"          + {merged}" + (f"   <- check {' '.join('t' + k for k in sorted(far, key=_key_order))}"
                                             if far else ""))
    alone = [c for c in res.children if c.get("play") is None]
    if alone:
        print(f"  {len(alone)} spawned unit(s) with no play of their card (not plays): "
              + ", ".join(f"t{c['key']} {c['label']}" for c in alone[:8]))
    if res.own:
        print("  ours (o tracks, not scored): " + ", ".join(f"{clock_text(p.get('t_rel'))} {p['card']}" for p in res.own[:8]))
    if d.get("m0") is not None:
        m = d["m0"]
        print(f"  M0 matched {m['matched']} of {m['of']} (read {m['read']}; {m['phantoms']} of its {m['live']} live "
              "play(s) match no gold play)")
    for w in res.warnings:
        print(f"  ! {w}")
    if res.written:
        print(f"  plays -> {session_paths(sid, args.labels_root, args.store).plays_file}")
    stored = res.list_no if res.written else res.list_no - 1        # a dry run stores no list
    if not res.written and not stored:
        print(f"next: & $BB tools\\vision_autolabel.py plays {sid}   (without --dry-run: stores list 1)")
    elif not d["confirmed"]:
        print(f'next: & $BB tools\\vision_autolabel.py plays {sid} --list {stored} --bad-plays p3 '
              f'--add-play "1:52=freeze" --confirm   (only the edits needed; --confirm alone when it is right)'
              + ("" if res.written else f"; list {stored} is the one stored, its ids may differ from these"))
    else:
        print(f"next: & $BB tools\\vision_gap_eval.py baseline --gold {sid}")
    return EXIT_OK


def _cmd_mix(args) -> int:
    try:
        table = load_table(args.classes)
        registry = Registry.load(args.registry)
    except (RegistryError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return EXIT_INPUT
    try:
        res = mix_sessions(args.name, args.sid or None, katacr=args.katacr or KATACR_ROOT, out=args.out,
                           base_labels=args.base, labels_root=args.labels_root, store=args.store, registry=registry,
                           table=table, repeat=args.repeat, force=args.force, allow_copy=not args.no_copy)
    except LabelError as exc:
        _print_problems("mix refused:", exc.problems)
        return exc.code
    d, c = res.doc, res.doc["counts"]
    print(f"mix {res.name}: KataCR ({d['katacr']}) and {len(d['sessions'])} owner session(s) -> {res.yaml}")
    for s in d["sessions"]:
        skipped = ", ".join(f"{v} {k}" for k, v in s["skipped"].items() if v)
        print(f"  {s['sid']:22s} {str(s['role']):5s} -> {s['split']:5s} {s['images']:5d} image(s)"
              + (f" + {s['repeats']} repeat(s)" if s["repeats"] else "") + (f"; left out: {skipped}" if skipped else ""))
    k = d["katacr_images"]
    print(f"  train: {c['owner_train_files']} owner file(s) and {k[MIX_TRAIN]} KataCR: {100.0 * d['owner_share']:.1f}% "
          f"owner; val: {c['owner_val_files']} owner and {k[MIX_VAL]} KataCR")
    for line in d["thin"]:
        print(f"  {line}")
    print(f"  label set: {d['labels']} labels, {d['yolo_classes']} YOLO classes, digest {d['label_set_digest']}; the base's "
          f"{d['base_yolo_classes']} names stay a prefix")
    if d["gold_excluded"]:
        print(f"  gold, never mixed: {', '.join(g['sid'] for g in d['gold_excluded'])}")
    for note in d["left_out"]:
        print(f"  - {note}")
    for w in res.warnings:
        print(f"  ! {w}")
    print(f"  from weights trained on the base, the run logs 'Remapped {d['base_yolo_classes']}/{d['yolo_classes']} cls "
          "head rows from pretrained weights by class name'")
    print("next (overnight; MuMu and the browser closed):")
    print(f"  {res.command}")
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
    p.add_argument("--pin-base", action="store_true",
                   help="pin --base (the real tmp/yolo/detector_labels.json, never the preview) as the registry's "
                        "base.digest, once, before the first mix; then commit the registry file")
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

    p = sub.add_parser("emit", help="propagate a session's labels over its scan and emit them: training images and "
                                    "boxes (train, dev) or full-frame gold labels (gold: no image)")
    _add_common(p)
    _add_session_roots(p)
    p.add_argument("sid", help="the session id")
    p.add_argument("--force", action="store_true", help="emit again (deletes only the earlier emit's own files)")
    p.add_argument("--allow-pending", action="store_true",
                   help="gold: tracks not labelled yet become ignore regions instead of refusing the emit")
    p.add_argument("--drop-missing", action="store_true",
                   help="with --force: emit although raws of images the earlier emit holds are gone (those images "
                        "are lost)")
    p.add_argument("--dry-run", action="store_true", help="propagate only: print the parts and boxes per label")
    p.add_argument("--stride", type=int, default=3, help="images with an enemy label: keep 1 in this many")
    p.add_argument("--empty-stride", type=int, default=15, help="images without one: keep 1 in this many")
    p.add_argument("--birth-s", type=float, default=1.0,
                   help="keep every image this many seconds after a labelled track's birth")
    p.add_argument("--birth-repeat", type=int, default=2, help="copies of a birth-window image in a training mix")
    p.add_argument("--max-masks", type=int, default=3, help="drop an image with more masked regions than this")
    p.add_argument("--format", default="jpg", choices=("jpg", "png", "npy"), help="emitted image format")
    p.add_argument("--geometry", default=None, help="an ArenaGeometry JSON (default: data/vision's, as scan uses)")
    p.set_defaults(func=_cmd_emit)

    p = sub.add_parser("review", help="draw review pages of a fresh emit: A, labelled tracks with their propagated "
                                      "boxes; B, emitted (gold: battle) frames with their boxes")
    _add_session_roots(p)
    p.add_argument("sid", help="the session id")
    p.add_argument("--seed", type=int, default=None, help="the sampling seed (default: from the session and review number)")
    p.add_argument("--per-cluster", type=int, default=3,
                   help="page A: tracks per cluster, and per auto-accepted label (gold: every track is shown)")
    p.add_argument("--frames", type=int, default=24, help="page B: frames shown, half in a birth window")
    p.add_argument("--width", type=int, default=1400, help="page width in pixels")
    p.add_argument("--cols", type=int, default=4, help="page B: frames per row")
    p.add_argument("--page-tiles", type=int, default=REVIEW_PAGE_TILES, help="page A: tracks per image")
    p.set_defaults(func=_cmd_review)

    p = sub.add_parser("mark", help="mark the tracks and frames a review showed wrong; a bad entry stores nothing")
    _add_session_roots(p)
    p.add_argument("sid", help="the session id")
    p.add_argument("--review", type=int, default=None,
                   help="the review the tile numbers are from (default: the latest); alone, records it with nothing bad")
    p.add_argument("--bad-tracks", default=None, help="track keys, comma-separated (t17,t40.1): masked (gold: ignored) "
                                                      "until relabelled with label --force")
    p.add_argument("--bad-frames", default=None, help="page-B tile numbers, comma-separated (3,9): left out of mix and "
                                                      "the gate")
    p.add_argument("--clear", default=None, help="marks to take back: track keys or frame image names")
    p.add_argument("--dry-run", action="store_true", help="check and show, store nothing")
    p.set_defaults(func=_cmd_mark)

    p = sub.add_parser("plays", help="gold sessions: derive the opponent's plays from the gold tracks; delete, add, "
                                     "confirm")
    _add_common(p)
    _add_session_roots(p)
    p.add_argument("sid", help="the session id (a gold session)")
    p.add_argument("--list", dest="list_no", type=int, default=None,
                   help="the list the play ids were read from (needed with an edit; must be the latest)")
    p.add_argument("--bad-plays", default=None, help="play ids to delete, comma-separated (p3,p7)")
    p.add_argument("--add-play", action="append", default=None,
                   help='a missed play: "1:52=freeze" (m:ss since the battle\'s first image), "b2/1:52=zap" (battle 2), '
                        '"n1234=Vines" (scan record n), "@9.1,12=..." a position in tiles; repeat for more')
    p.add_argument("--confirm", action="store_true", help="confirm the list (after the edits): the gate needs it")
    p.add_argument("--off-deck", action="store_true", help="accept an added card that is not in the declared deck")
    p.add_argument("--allow-pending", action="store_true",
                   help="confirm although the gold emit left tracks unlabelled (emit --allow-pending): their plays "
                        "are missing from the list")
    play_defaults = PlayConfig()
    p.add_argument("--window-s", type=float, default=play_defaults.window_s,
                   help="units of one card born this close in time are one play")
    p.add_argument("--radius-tiles", type=float, default=play_defaults.radius_tiles,
                   help="... and this close to its first unit")
    p.add_argument("--reach-gap-s", type=float, default=play_defaults.reach_gap_s,
                   help="reach: a unit lost up to this long before is found again within its walk (0: off)")
    p.add_argument("--reach-margin-tiles", type=float, default=play_defaults.reach_margin_tiles,
                   help="reach: the walk's slack in tiles (and how near a second track is to a unit still seen)")
    p.add_argument("--zone-y-tiles", type=float, default=play_defaults.zone_y_tiles,
                   help="deploy zone: a unit first seen this far into the arena (the opponent's y; ours 32 - y) is in "
                        "the defender's half, where no play starts (32: off)")
    p.add_argument("--zone-gap-s", type=float, default=play_defaults.zone_gap_s,
                   help="zone: such a unit continues one of its card lost up to this long before, within its walk")
    p.add_argument("--back-tiles", type=float, default=play_defaults.back_tiles,
                   help="a part in its owner's deploy zone this far behind a unit's last sighting is not that unit")
    p.add_argument("--cycle-plays", type=int, default=play_defaults.cycle_plays,
                   help="card cycle: after this many other plays a card may be played again, so a part continues no "
                        "unit lost before (0: off)")
    p.add_argument("--flag-gap-s", type=float, default=play_defaults.flag_gap_s,
                   help="the list warns of reach merges over this gap (and of every zone merge)")
    p.add_argument("--dry-run", action="store_true", help="show the list, store nothing")
    p.set_defaults(func=_cmd_plays)

    p = sub.add_parser("mix", help="build a training mix: KataCR plus the emitted train sessions (dev as val), "
                                   "cr_mix.yaml and the extended detector_labels.json; gold is never mixed")
    _add_common(p)
    _add_session_roots(p)
    p.add_argument("sid", nargs="*", help="only these sessions (default: every train and dev session with an emit)")
    p.add_argument("--name", required=True, help="the mix's name, like ft1: tmp/mix/<name>, and the run name to train it")
    p.add_argument("--katacr", default=None,
                   help="scripts/prepare_yolo_dataset.py's output: cr.yaml, images/, labels/ (default tmp/yolo)")
    p.add_argument("--base", default=None, help="the base label set (default <katacr>/detector_labels.json)")
    p.add_argument("--out", default=None, help="the mix folder (default tmp/mix/<name>; never in tmp/yolo or the store)")
    p.add_argument("--repeat", type=int, default=1,
                   help="copies of every training image, times emit's birth repeat (dev images: always 1)")
    p.add_argument("--force", action="store_true", help="build the mix again (removes only the mix's own files)")
    p.add_argument("--no-copy", action="store_true", help="fail instead of copying when an image cannot be hard-linked")
    p.set_defaults(func=_cmd_mix)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
