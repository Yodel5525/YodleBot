# New cards: detector and event coverage plan

Status: plan, 2026-10-30. Work items are N01-N18 (section 9). The capture decks of 4.4 and 4.5 were rechosen from the clan's own battles on 2026-10-30 (section 4.8).

The problem, in the owner's words: the clanmates play a wide variety of decks, and most of them use one or more cards released since 2024. The live detector learns only from KataCR's 2024 frames.

The facts below come from reading code and data. Nothing was run, and no game image was examined except 4 owner frames. Sources:
- **Card table:** RoyaleSim 0.1.42's 16.402.22 table (`client160402017_20261019`, the one `data/engine.json` selects), compared with the 15.535 and 16.402.19 tables.
- **Vision data:** `data/vision/classes.json`, `data/vision/spawn_children.json`, and the RoyaleAPI key maps `data/cards/replay_cards.json` and `replay_cards_extra.json`.
- **Opponent decks:** 1,467 top-ladder opponent decks in `data/ladder/battles_2p6.csv` (2026-08-27 to 09-01, ratings 2816-3604).
- **KataCR:** the dataset tree at 31b4151 and its `version_info/annotation_v0.14.csv` (real-frame labels per class and side, as of 2024-03-09), plus KataCR's code.
- **ultralytics 8.4.192:** the wheel source, the version pinned in `requirements/gpu.txt`.
- **Clan decks (section 4.8):** the owner's `tools/opponents.py pull` of 2026-10-30, 1,286 battles of 44 clan players. Only aggregate card statistics are written here: no names, no tags, no single player's deck.

## Short version

- **The gap.** 14 of the 123 cards in the regular pool have no KataCR class. 26 of the 43 evolutions and all 18 heroes are not covered. 97.5% of the top-ladder opponent decks hold at least one gap item, 2.25 of 8 cards on average.
- **The gap in the clan (4.8).** 95.0% of the clanmates' ranked decks hold at least one card or form the detector lacks or barely knows, 2.85 of 8 cards on average. 28.9% hold a card it cannot name at all.
  - The most-played gaps are Knight hero (in .207 of decks), Bowler hero (.150), the Inferno Dragon evo (.138), the Baby Dragon evo (.122), and Zap and Lightning (thin on KataCR's enemy side).
  - The ladder-based capture decks would leave 48.4% of clan decks with a gap. **The capture decks in 4.4 and 4.5 are now chosen from the clan data**: same 36 friendlies, and they leave 7.1%. On the clan's event decks, which they were not chosen on, they leave 31.6% (the ladder decks 69.2%).
  - Heroes decide it. With only the Electro Wizard hero owned (the only hero the main account's log proves), 69.8% keep a gap.
  - Forms (-ev1 or -hero) come from the deck slot, as `opponents.py` reads them. The pull backs that rule; only slot-3 reads of Knight, Musketeer, Valkyrie and Wizard could be the other form (4.8.4).
- **Two table bugs.** These hurt even with a perfect detector:
  - Heal Spirit plays are never named.
  - Spirit Empress is named as an internal variant card, not as itself.
- **Plays that go missing.** Today an enemy play the tables cannot explain is dropped. It never reaches the policy, and the opponent's elixir is not charged.
- **The plan has four parts:**
  1. Fix the event tables.
  2. Add a safety net: a play that cannot be named reaches the policy as UNKNOWN and is charged an expected cost.
  3. Record friendlies where the owner's main account plays the new cards against Yodle2. Turn them into auto-labelled real frames, and fine-tune the detector with labels appended to tonight's.
  4. Gate every new detector on held-out friendlies before it is pinned.
- **Tonight.** Run runbook section 15 unchanged (section 8).

## 1. The gap

### 1.1 The card pool

The 16.402.22 table has 145 cards. Each carries the client's `not_visible` flag.

- **123 cards are visible: the regular pool a clanmate can bring.** That is exactly the 122 IL_Replay/RoyaleAPI keys plus **Minion Giant**.
  - Minion Giant is in 0 of the 1,467 ladder decks.
  - RoyaleSim's calibration records it in live 16.402 battles, so it was most likely released after the ladder snapshot.
- **22 cards are not visible (event-only, test or internal).** None of them is in the key map or in any ladder deck. They need no detector work:
  - **Super cards:** SuperArcher, SuperEliteArcher, SuperHogRider (Santa Hog Rider), SuperHogRiderTerry (Terry), SuperIceGolemite, SuperKnight, SuperLavaHound, SuperMiniPekka, SuperWitch. *Event-only.*
  - **Event cards:** PrinceBuff (Raging Prince), TriWizards, RoyalRecruits_Chess, SkeletonWarriors_SpookyChess, GoblinPartyHut, GoblinPartyRocket, GoblinRocketSilo, BarbarianLauncher, GlobalClone, GlobalLightning, WarmSpell. *Event-only.*
  - **Internal forms:** MergeMaiden_Normal and MergeMaiden_Mounted are the two forms of the visible card MergeMaiden (Spirit Empress). They are not deck cards. *Internal.*
- **Name checks** (verified by `rs_cards --coverage` on the owner's PC over 20,000 IL_Replay matches):
  - void = **DarkMagic**
  - rune-giant = **GiantBuffer**
  - spirit-empress = **MergeMaiden**

### 1.2 Whole cards with no KataCR class: 14 of 123

Counts are opponent decks out of 1,467.

| Card (catalogue name) | Rarity, elixir | On screen (16.402.22) | Decks |
|---|---|---|---|
| Goblinstein (`Goblinstein`) | Champion, 5 | two units: `goblinstein` (targets buildings) and `goblinstein_doctor` (ranged) | 133 |
| Vines (`Vines`, spell) | Epic, 3 | `Vines_AeO`, a 2.5-tile area for 1.4 s; pulls flyers down | 111 |
| Berserker (`Berserker`) | Common, 2 | small fast melee unit; also a hero form | 109 (2 base, 107 hero) |
| Three Musketeers (`ThreeMusketeers`, rework) | Rare, 9 | `ThreeMusketeer_Rework_Character_1` ×3 plus Character_2 ×1; mapped to no KataCR class | 47 |
| Rune Giant (`GiantBuffer`) | Epic, 4 | ground unit, targets buildings, buffs friends | 39 |
| Goblin Demolisher (`GoblinDemolisher`) | Rare, 4 | ranged, then `GoblinDemolisher_kamikaze_form` | 35 |
| Suspicious Bush (`SuspiciousBush`) | Rare, 2 | kamikaze bush that releases `BushGoblin` | 24 |
| Goblin Curse (`GoblinCurse`, spell) | Epic, 2 | `GoblinCurseBase`, a 3-tile circle for 6 s; enemies that die inside become goblins | 23 |
| Boss Bandit (`BossBandit`) | Champion, 6 | dashing melee unit | 21 |
| Ronin (`Ronin`) | Legendary, 5 | melee unit | 18 |
| Void (`DarkMagic`, spell) | Epic, 5 | `DarkMagicAOE`, a 2.5-tile circle for 4 s | 15 |
| Goblin Machine (`GoblinMachine`) | Legendary, 5 | big melee unit with a rocket | 6 |
| Spirit Empress (`MergeMaiden`) | Legendary, 3 or 6 | below 6 elixir `MergeMaiden_Normal` (3, ground); at 6 or more `MergeMaiden_Mounted` (6, flying) | 5 |
| Minion Giant (`MinionGiant`) | Rare, 4 | flying, targets buildings | 0 (newer than the snapshot) |

The other 109 pool cards have a base KataCR class. Goblin Drill counts as covered, through `goblin-drill` and `dirt`.

### 1.3 Evolutions: 43 in 16.402.22

**Covered by a KataCR `-evolution` label (16):** Skeletons, Knight, Archers, Barbarians, Bats, Battle Ram, Bomber, Firecracker, Ice Spirit, Mortar, Royal Giant, Royal Recruits, Tesla, Valkyrie, Wall Breakers, Zap.
- Zap evolution has a label but 0 sprites and 0 real boxes.
- Several of these labels are thin on the enemy side: ice-spirit-evo 10 real boxes, bomber-evo 16, valkyrie-evo 27.
- Battle Ram evolution has 0 real boxes in v0.14; 27 later frames (side unknown) add some. Section 4.8 counts it, and every troop label with 25 or fewer enemy boxes, as thin.

**Covered only by a name merge (1):** Skeleton Army evolution (100 decks). Its soldiers merge to `skeleton-evolution`, but its General and Spectral units have no class. The look is unverified.

**Not covered (26).** Decks out of 1,467, with the catalogue name where it differs:

| Evolution | Decks | Evolution | Decks |
|---|---|---|---|
| Elite Barbarians (`AngryBarbarians`) | 237 | P.E.K.K.A (`Pekka`) | 53 |
| Baby Dragon | 174 | Witch | 52 |
| Cannon | 162 | Princess | 45 |
| Inferno Dragon | 145 | Electro Dragon | 42 |
| Royal Ghost (`Ghost`) | 134 | Wizard | 32 |
| Giant Snowball (`Snowball`, spell) | 119 | Goblin Drill | 30 |
| Minion Horde | 97 | Goblin Cage | 30 |
| Royal Hogs (they fly in, then land) | 75 | Mega Knight | 27 |
| Musketeer | 73 | Lumberjack (`RageBarbarian`) | 20 |
| Executioner (`AxeMan`) | 63 | Furnace (`FirespiritHut`) | 14 |
| Skeleton Barrel (`SkeletonBalloon`) | 56 | Hunter | 13 |
| Goblin Barrel (spell; `GoblinDummy` decoys) | 56 | Goblin Giant | 12 |
| | | Dart Goblin (`BlowdartGoblin`) | 6 |
| | | Electro Giant (new in 16.402) | 0 |

Two further limits on evolutions:
- An evolution appears only on its evolved cycle, so one card shows both forms within a match.
- KataCR's `evolution-symbol` has 36 ally-side real boxes and 0 enemy-side, so the enemy's evolution marker cannot be learned from KataCR.

### 1.4 Heroes: 18, none covered

KataCR predates heroes. Every hero has an elixir-paid ability, and its look changes mid-fight. Decks out of 1,467:

| Hero | Decks | Hero | Decks |
|---|---|---|---|
| Balloon | 135 | Wizard | 58 |
| Berserker | 107 | Barbarian Barrel (`BarbLog`) | 55 |
| Knight | 93 | Bowler | 55 |
| Giant | 89 | Dark Prince | 47 |
| Valkyrie | 84 | Goblins | 28 |
| Mini P.E.K.K.A | 74 | Ice Golem (`IceGolemite`) | 25 |
| Tombstone | 69 | Magic Archer (`EliteArcher`) | 15 |
| Musketeer | 65 | Ice Wizard (new in 16.402) | 0 |
| Mega Minion | 60 | Electro Wizard (new in 16.402) | 0 |

Ten hero forms bring extra units with no class (classes.json 282-298). Examples: MusketeerTurret, BalloonHero_Bomb, SkeletonTrooper, DarkPrinceHero_Mount, TombstoneHero_Monster_*, BarbLogBarbarian.

In all, 38 classes have a KataCR base class but lack an evolution or hero form.

### 1.5 Spells

The pool has 23 spell-kind cards.
- **KataCR has a visual for 18 true spells** (the 18 entries of `KATACR_SPELL_CLASSES`, Mirror included, which has no sprites).
- **Two spell-kind cards deploy troops.** Heal Spirit's KataCR label `heal-spirit` is a unit label (`HealSpirit`); see 1.7. Spirit Empress has no class; see 1.2.
- **Not covered:** Void (DarkMagic), Vines and Goblin Curse. `KATACR_SPELL_CLASSES` (`scripts/vision/events.py:113-121`) is the only way a spell play gets named, so these three plays are never reported. The opponent's elixir is then off by 5, 3 or 2.
- **Spell forms with no class:** Giant Snowball evolution, Goblin Barrel evolution (`GoblinDummy` decoys), and Hero Barbarian Barrel (`BarbLogBarbarian`).
- **Thin on the enemy side even where covered** (enemy real boxes in KataCR): zap 6, lightning 9, royal-delivery 9, giant-snowball 10, freeze 12, rocket 20, goblin-barrel 23, clone 0.

### 1.6 Tower troops

- Tower Princess: `queen-tower`, 12,989 enemy real boxes.
- Cannoneer: `cannoneer-tower`, 568 enemy real boxes.
- Dagger Duchess: `dagger-duchess-tower`, 18 sprites and 0 boxes in v0.14. It was added with the April 2024 frames.
- **Royal Chef has no KataCR class.** Its classes are 47 `Chef`, 48 `ChefTower` and 49 `ChefTowerKing`. It is in the pool: 14 of 785 live 16.402 battles in RoyaleSim's calibration had one.

Towers are `ignore` classes in events.py, so a tower troop cannot cause a misread play. It does give the wrong tower stats.

### 1.7 Gaps in the event tables (they hurt even with a perfect detector)

1. **Heal Spirit (107 decks) is never named.**
   - `spawn_children.json` gives `Heal` kind spell, `spell_children` [HealSpirit] and no `spell_objects`. So its only spell class is `Heal`, and no detector label is ever called `Heal`.
   - KataCR's `heal-spirit` maps to the unit `HealSpirit`, which is in no card's composition. `from_spawn_primary` also skips cards that the base tables hold as spells (`events.py:397`).
   - So a HealSpirit track reaches rule 5 with no candidate card, is counted in `opp_stats.unknown`, and no play is emitted (`events.py:832-839`). KataCR has 77 enemy real boxes of heal-spirit, so the detector should see it.
2. **Spirit Empress is named as a variant card.**
   - `MergeMaiden_Normal` and `MergeMaiden_Mounted` have their own troop compositions in `spawn_children.json`. The card `MergeMaiden` is kind spell, with no objects.
   - So once its units have labels, rule 5 names the not-visible variant card (or nothing, if the engine catalogue lacks its id). It should name MergeMaiden, at 3 or 6 elixir.
3. **Goblin Curse goblins.**
   - `GoblinCurseGoblin` is aliased to `Goblin`, and the curse itself has no label.
   - So those goblins can read as a new Goblins or Goblin Gang play (rule 5).
4. **Goblin Barrel drift.**
   - In 16.402.22 the barrel spawns `BarrelGoblin`; in 15.535 it spawned `Goblin`.
   - `spawn_children.json` was built from the 15.535 extraction (its `source` note) and still says Goblin. That is right for vision, because KataCR's box is `goblin`. But the engine's unit token is BarrelGoblin, and `classes.json` gives BarrelGoblin (class 15) no KataCR class.
   - The same file also lacks the Electro Giant evolution and the two new heroes.

### 1.8 What a missing class does today

- **The detector.** It is YOLOv8 with one sigmoid score per class. It has no objectness score and no "unknown" class, and it keeps boxes at conf 0.35 (`detector.py:127`). An unseen sprite therefore gives either:
  - no box: the unit is invisible, with no track, no play, no token and no elixir charge; or
  - a look-alike class, which then names the wrong card.
- **The card is decided early.** `rec.cls` follows the tracker only while the track is pending (`events.py:701-702`). So the label about 2-3 frames after birth decides the card.
- **Unexplained classes are dropped.** A class with no card is counted in `opp_stats.unknown`, and no OppPlay is emitted (`events.py:832-839`).
  - `public_state` drops tracks of unknown type (`tools/public_state.py:577-578`).
  - `OpponentElixir.record` skips plays it cannot price (`public_state.py:724-737`).
  - `PublicMemory` already has an UNKNOWN token, 1 (`tools/rs_memory.py:63`), and `record_plays` maps card -1 to it (`events.py:1024-1034`). Nothing emits card -1 today.
- **Evolution or hero read as its base class.** This still names the right card, because `Detection.cls` drops the form (`detector.py:541-547`). Only the unit token and the form are wrong.
- **Our units misread as an enemy.** The tracker forces any label outside our deck to the enemy team (`tracker.py:402-406`). So our unit misread as a non-deck class becomes a phantom enemy play.

### 1.9 Possible look changes since 2024 (unverified, no image inspected)

Furnace (`Furnace_rework`), Goblin Hut (`GoblinHut_Rework`, now 4 elixir), Graveyard (`Graveyard_rework`), Three Musketeers (rework units), Goblin Barrel (BarrelGoblin), Goblin Drill dig. KataCR's own notes report no look changes.

### 1.10 How common the gap is (1,467 top-ladder opponent decks)

- 33.7% hold a whole card with no KataCR class.
- 89.5% hold an uncovered evolution.
- 68.0% hold a hero.
- 97.5% hold at least one gap item; on average 2.25 of 8 cards per deck.
- **Deck slot rules, as observed** (evolutions, heroes, champions):
  - shapes: (2,1,0) 884, (2,0,1) 425, (1,2,0) 61, (1,1,1) 48, (2,0,0) 39, (1,1,0) 5, (1,0,2) 3;
  - so: at most 2 evolutions, at most 2 heroes and champions together, and at most 3 of all three kinds;
  - never 2 evolutions plus a hero plus a champion.
  - The decks in section 4 follow these rules.

## 2. The approach

Three designs were compared, and two reviewers checked them against the code. The chosen plan uses the "auto-labelled real frames" design as the backbone, with parts of the other two.

### 2.1 Fix the event tables first (N01)

- **Heal Spirit:** give `Heal` a composition {HealSpirit: 1}, so rule 5 can name it.
- **Spirit Empress:** add a card alias, so `MergeMaiden_Normal` and `MergeMaiden_Mounted` plays are reported as MergeMaiden at a variant cost of 3 or 6.
- **Spawn tables:** regenerate `spawn_children.json` from the 16.402.22 table where it can be read. Decide how BarrelGoblin maps for vision.

### 2.2 A safety net for everything not yet labelled (N02, N03)

Clanmates play many decks, so some cards will always be ahead of the labels. They must not vanish.
- An enemy track that no rule explains becomes `OppPlay(card=-1)`. That is the UNKNOWN token 1, which the policy already sees in training through perception noise (`tools/perception_noise.py:44-47`).
- **Elixir.** The unknown play is charged an expected cost through `ElixirCounter.play(tick, card, cost=)` (`tools/rs_elixir.py:427`), and the count is marked not exact.
- **Board.** Unknown enemy units stay on the board as rows with card -1 instead of being dropped.
- **Play marker (N03, off until measured).** The deploy stopwatch (KataCR `clock`) can mark a play whatever the card:
  - it is already one of tonight's labels, with 1,978 ally and 1,023 enemy real boxes;
  - the owner frames show one stopwatch per troop or building play, red for the enemy; spells get none;
  - an enemy stopwatch with no play nearby becomes an UNKNOWN play;
  - it stays off until its recall is measured on gold.

### 2.3 Learn the new cards from the owner's friendlies (N04-N13)

**Why the owner's friendlies.**
- The owner's main account plays gap decks against Yodle2, and the main always shows at the top, as the enemy. That is exactly what the live bot sees, with the same PrintWindow capture, arena, fights and tower attacks.
- Only enemy-side labels are needed, because Yodle2 only ever shows base Hog 2.6.

**Per recorded battle:**
1. **Ingest.** Hard-link the images into a session folder at once, so the recorder's disk budget cannot prune them.
2. **Scan (two chains on the same image).**
   - The exact live chain, run through `tools/vision_replay.ReplayPerception` with tonight's detector.
   - A harvest chain:
     - the same weights at conf 0.05, with no side merge and no cross-class suppression;
     - foreground blobs against a per-battle background median;
     - a position-only tracker over all of these proposals.
3. **Flag enemy tracks the live chain gets wrong.** Seven flags:
   - S1: no live track;
   - S2: unknown;
   - S3: a card not in the declared deck;
   - S4: a flickering label;
   - S5: a ground class over the river (river jumpers exempt);
   - S6: under an unmatched enemy stopwatch;
   - S7: the look changes mid-track (the track is split there).
4. **Contact sheet.**
   - Flagged tracks are clustered per track, and the owner types one map string per battle.
   - The legend lists only the declared deck's classes.
   - Safe, deck-consistent, high-confidence tracks are auto-accepted.
5. **Propagate and emit.**
   - Each label goes to every image of its track.
   - Unlabelled leftovers are masked with background, or the frame is dropped.
   - YOLO frames are written in the live crop framing. The first second of each track is oversampled, because that is when the card is decided.

**Training.**
- **Labels are appended to tonight's label set**, and indices never change. ultralytics 8.4.192 copies the class rows by name when the class count grows:
  - `_remap_cls_by_names`, `nn/tasks.py:383-450`, on by default (`cls_remap`, `cfg/default.yaml:26`);
  - yolov8s's class-branch width is 128 for 310 and for 348 classes (`nn/modules/head.py:123`);
  - so the learned classes survive.
- **Phase-1 labels** (19 labels, 38 YOLO classes, so 348 in all), in this order:
  - units: goblinstein, goblinstein_doctor, Berserker, BerserkerHero, GiantBuffer, GoblinDemolisher, GoblinDemolisher_kamikaze_form, SuspiciousBush, BushGoblin, BossBandit, Ronin, GoblinMachine, MergeMaiden_Normal, MergeMaiden_Mounted, MinionGiant, ThreeMusketeer_Rework_Character_1 (Character_2 is aliased to it in classes.json);
  - spells, by card name: DarkMagic, Vines, GoblinCurse.
  - Hero and evolution forms (detector keys from `ClassTable.detector_key`, in the style of `Skeleton_EV1` and `BerserkerHero`) are appended as they are labelled.
- **Thin forms train under their base label.** A hero or evolution form with fewer than 8 labelled tracks trains as the base card (`train_as:<base>`). The card is still named right.
- **The fine-tune** starts from the pinned weights, on KataCR's frames plus the owner's, with an explicit optimizer (section 8 explains why).

### 2.4 Gate before pinning (N14)

Held-out gold battles are scored per box, per birth window and per play, through the real `vision_replay` chain. The baseline is tonight's model. A model is pinned only on PASS and after one dry-run friendly.

### 2.5 Later and optional

- **Compositor (N17).** It pastes KataCR's 4,627 unused sprite cutouts and the owner-track cutouts onto recorded backgrounds. This is for thin classes only.
  - Port the logic of KataCR's `generator.py` (MIT, keep the notice).
  - Its class balancing never runs and must be rewritten: `generator.py:332` sizes both pools with the noise pool's length, and `:450-453` compare names without a side suffix against `<cls>_<side>` keys.
- **Deck prior (N18).** It breaks look-alike ties, such as Berserker against Knight. It is perception-only, never feeds `PublicMemory`, and is off by default (see open questions).
- **Deferred:** a few-shot crop embedder and gallery for naming. It is revisited only if the gates show that detector fine-tunes cannot keep up.

### 2.6 Why not the other two designs as the backbone

- **Synthetic scenes first.** Sprites cut from isolated blobs lose tower attacks and fights, which is where building-targeters spend their time. It is also the most new code, with a real sim-to-real risk. It is kept as N17.
- **Decoupled identity (embedder plus gallery).** An unseen sprite still gets no box, and placing it would rest on badge colour masks measured on only 4 frames. Void, Vines and Curse have neither badge nor stopwatch. It also changes contracts in eight modules. Its safety net (N02, N03) and the Spirit Empress alias are kept.

### 2.7 Scale facts used by the plan

- **Tile size at the model input.** At imgsz 640, KataCR's 568x896 frames give about 22.1x17.9 px per tile. Our live crop (about 706x1002 at 747x1328) gives about 22.5x18.3 px. They agree within about 2%, so KataCR frames need no rescale.
- **Pasting KataCR sprites into a full 747x1328 frame** needs ×1.143: KataCR's tile is 30.9x25.0 px, ours 35.3x28.6.
- **Level badges are not learnable from KataCR.** `bar-level` has 0 real boxes in annotation_v0.14, and we train on real frames only, so tonight's detector will not emit it. Level badges come only from `hpbars` colour masks.

## 3. What changes in which files

| File | Change | Item |
|---|---|---|
| `tools/vision_spawn_children.py`, `data/vision/spawn_children.json` | Heal composition; MergeMaiden variants; regenerate from 16.402.22 where readable | N01 |
| `scripts/vision/events.py` | `EventTables` card alias and variant cost; `EventConfig.unknown_plays`; markers argument; `EventDeriver.how_of(uid)` | N01, N02, N03, N05 |
| `scripts/vision/types.py` | `OppPlay.cost` (default None); `VisionFrame.markers` (default empty); `from_dict` fills the defaults, so old recordings load | N02, N03 |
| `tools/public_state.py` | unknown enemy tracks become card -1 rows; `OpponentElixir.record` charges the expected cost | N02 |
| `scripts/vision_play.py`, `tools/vision_replay.py` | pass `clock` UI detections to the marker tracker; skip -1 in `ctx.opp_cards`; `--unknown-plays`, `--markers` flags; vision_replay gets `--conf`, `--imgsz`, `--max-det` | N02, N03, N04 |
| `scripts/vision/markers.py` (new) | enemy stopwatch tracks as play markers | N03 |
| `scripts/vision/capture.py` | `--budget-mb`, passed to `VisionRecorder` (today none is passed, `capture.py:636`) | N04 |
| `scripts/vision/tracker.py` | `Tracker.last_assignment`: the uid for each observation | N05 |
| `scripts/vision/detector.py` | `crop_rect_for(geometry, config, size)` as a module function, shared by the emitter | N05 |
| `scripts/train_detector.py` | `--optimizer`, `--lr0`, `--cls-pw`, `--freeze`; a warning when a fine-tune keeps the default `--name` | N06 |
| `data/vision/detector_extra_labels.json` (new) | append-only label registry, seeded with the 19 phase-1 labels | N07 |
| `tools/vision_autolabel.py` (new) | `ingest`, `status`, `scan`, `sheet`, `label`, `emit`, `review`, `mark`, `plays`, `mix` | N08-N13 |
| `scripts/vision/unit_harvest.py` (new) | background model, blobs, proposal merge, harvest chain, flags, track split, clustering, sheet | N09, N10 |
| `data/vision/event_cards.json` (new) | the 22 not-visible cards; `ingest` refuses them | N08 |
| `data/vision/owner_labels/<sid>/` (new, committed) | `session.json`, track labels, gold plays: small JSON only | N08-N12 |
| `tools/vision_gap_eval.py` (new) | the gate: detection, birth window, plays, elixir, regression; writes `gate.json` | N14 |
| `tools/perception_noise.py` | the "labelled form" test reads the deployed detector's label set, not classes.json `katacr` (`perception_noise.py:840`) | N15 |
| `tools/vision_gap_labels.py` (new) | `report` (what is uncovered) and `plan` (which decks to record, under the slot rules) | N16 |
| `tools/synth/compose.py` (new, optional) | KataCR-style compositor with fixed balancing | N17 |
| `scripts/vision/deck_prior.py` (new, optional) | perception-only deck prior, off by default | N18 |

No change is needed to `scripts/prepare_yolo_dataset.py`. `scripts/vision/harvest.py` is edited by another workflow: import only `open_recording` and `parse_label_map` from it, by name. The runbook is edited by another workflow; it should link here once N08-N13 exist.

Images stay on the data drive and are never committed. Recordings show account names (runbook section 18).

## 4. The owner's capture protocol

### 4.1 Every friendly

1. **Start MuMu and Clash Royale on Yodle2 yourself.** Nothing in the code starts the emulator.
2. **Start the recording on the PC.** It runs for `--seconds` and then stops. Give each session its own folder, because the recorder prunes the oldest images under its whole root above 4,000 MB.

   ```powershell
   $SID = '20261101_A_1'      # date, deck letter, match number
   & $BB -m scripts.vision.capture --source auto --seconds 420 --record reports\vision\harvest\$SID --image-every 3
   ```

   `--image-every 3` gives about 19 images per second, so about 1.2-1.7 GB per battle. The default of 5 is enough for field friendlies.
3. **Start the battle.** Send the friendly from the main account on the phone, and accept it on Yodle2 by hand. A human starts every battle. Nothing taps the lobby.
4. **After the battle, ingest it at once** (N08). Until then, keep one root per session.

**Recording is passive.** It needs no taps. Yodle2 can be:
- idle;
- played lightly by you with the mouse;
- or, once runbook section 18 works, the bot with `--tap` while you watch.

### 4.2 How to play each harvest friendly

1. **Neither side plays in the first 10 seconds.** This gives a clean background.
2. **Play every card at least 4 times:**
   - both lanes, at the bridge and at the back;
   - some defensively, into Yodle2's units, so there are fights.
3. **Cast every spell at least 4 times,** on groups and on empty ground.
4. **Cycle the evolution cards.** They appear only on evolved cycles; aim for at least 2 evolved plays each.
5. **Use every hero and champion ability at least twice.**
6. **Play Spirit Empress both below 6 elixir and at 6 or more,** so both forms appear.
7. **Keep the game going to overtime.** Avoid an early 3-crown.
   - If Yodle2 is idle, play mostly in your own half and at the bridge, not straight at the towers.
8. **Swap in any card the main account does not own.** Tell `ingest` the real deck and which slots are evolution or hero.
9. **Single-unit evolutions in Tier 2 need hard cycling.** A form with fewer than 8 labelled tracks trains under its base label (2.3). Two evolved plays per friendly give a Tier 2 evolution only 4 training tracks, so aim for 4 evolved plays a friendly. Multi-unit evolutions (Minion Horde, Skeleton Army, Royal Hogs, Goblin Barrel, Elite Barbarians) get several tracks per play. `status` lists what to top up.
10. **Play each hero 5 times or more.** 4 plays × 2 training friendlies is only just 8 tracks.
11. **Cards with both forms in slot 3** (Valkyrie hero in I, Wizard evo in J): write the form you chose in the `ingest` deck string.

### 4.3 Round 0: the friendly already recorded (no play)

The 2026-10-29 friendly (22,067 frames, about 4,414 images) has the main account at the top. It becomes the first gold battle: the M0 baseline, plus a table of what tonight's model reads each new card as.

**Copy its folder out of `reports\vision\recordings` before section 16 records more there.** All recordings in that root share the 4,000 MB budget, and the oldest images go first.

**The 8 cards are known** (open question 1 is answered): boss-bandit, elite-barbarians, executioner, freeze, guards, hog-rider, minion-horde-ev1, zap. Yodle2's spar record in the pull matches all 8, forms included: the Minion Horde evo in slot 1, Boss Bandit in slot 2, and Elite Barbarians as a base card in slot 3.

```powershell
$R0 = 'reports\vision\gold'      # the copied session folder
& $BB tools\vision_autolabel.py ingest $R0 --sid 20261029_R0_1 --role gold --capture natural --deck "boss-bandit,elite-barbarians,executioner,freeze,guards,hog-rider,minion-horde-ev1,zap" --client 16.635.1
```

- **Its gap items:** Boss Bandit (no class), Zap and Freeze (thin), the Minion Horde evo (base class only). All four are also in capture decks (C; C and G; J; F), so Round 0 is their natural gold.
- **The 10 GoldenKnight plays** the pinned detector reported. The deck has no Golden Knight, so none of them is one. Label each GoldenKnight cluster after looking at it: Boss Bandit, which has no class yet, is the expected answer. The tracker forces any label outside our deck to the enemy side (1.8), so some may be the main account's Executioner, Elite Barbarians or Guards (label them as those), or Yodle2's own units such as Hog Rider (label them `o`). Golden Knight is in no clanmate's ranked deck.
- **The Elite Barbarians evolution.** Slot 3 showed no evolution bit and the card was played as base, so the main account probably lacks that evolution. Deck J names a fallback.
- **The probe** (`opponents.py probe`, SEASON_UPDATES 7.1) can run on the existing log; it needs no new friendly. Its form test counts any owned form in a slot that cannot take it: Round 0's Executioner (slot 4) and Zap (slot 7), and Lumberjack, Balloon and Barbarian Barrel in slots 4-8 of the main account's usual ranked deck (10 battles in the log); each counts if you own its evolution or hero. Deck A's friendly 1 gives that test nothing: its special cards sit in slots that take their forms, and its five plain cards have no forms.
- **Optional Round 0b.** One natural friendly with your usual ranked deck, as a second natural gold and, if wanted, a probe friendly against Yodle2 (an evolution in slot 1, a hero in slot 2, all proven owned). It holds four clan gap items: Vines, the Baby Dragon evo, the Electro Wizard hero and the P.E.K.K.A evo. About 7 minutes of play and 27 minutes of labelling.

  ```powershell
  & $BB tools\vision_autolabel.py ingest reports\vision\harvest\$SID --sid 20261101_R0b_1 --capture natural --role gold --deck "<your 8 cards>" --client 16.635.1
  ```

  Pass `--capture natural`: with `--role auto` and the default capture, a session id ending in `_1` trains.

### 4.4 Tier 1: the 14 whole cards (3 decks × 4 friendlies)

**This table is current** (revised 2026-10-30 from the clan data, 4.8). It replaces the ladder-based Tier 1 table, which put the top ladder's favourite evolutions in the special slots.

Deck lists use RoyaleAPI keys, as in `data/ladder/battles_2p6.csv`; Minion Giant has no key yet, so it is given by its catalogue name. Slot 1 takes an evolution, slot 2 a hero or champion, slot 3 any of them (4.8.3).

| Deck | Slot 1 (evo) | Slot 2 | Slot 3 | Slots 4-8 | Shape (evo, hero, champion) |
|---|---|---|---|---|---|
| A | skeleton-barrel-ev1 | berserker-hero (if owned; else electro-wizard-hero) | inferno-dragon-ev1 | vines, suspicious-bush, goblin-curse, goblin-demolisher, rocket | (2, 1, 0) |
| B | baby-dragon-ev1 | goblinstein | giant-snowball-ev1 | MinionGiant, rune-giant, ronin, berserker, lightning | (2, 0, 1) |
| C | goblin-drill-ev1 | boss-bandit | witch-ev1 | three-musketeers, void, spirit-empress, goblin-machine, zap | (2, 0, 1) |

- **All 14 whole cards stay in Tier 1.** Base Berserker, which was Tier 3, rides in B.
- **The nine special slots** carry the clan's six most-played uncovered evolutions (Inferno Dragon .138, Baby Dragon .122, Giant Snowball .099, Goblin Drill .092, Skeleton Barrel .091, Witch .082), Berserker hero, Goblinstein and Boss Bandit. Evolutions go in Tier 1 because they look evolved only on their evolution cycle and need the third training friendly; a hero shows on every play.
- **A** holds the whole cards clanmates play: Vines (.085), Berserker hero (.050), Goblin Demolisher (.046), Suspicious Bush (.046) and Goblin Curse (.032). Without Berserker hero, play the Electro Wizard hero and get Berserker hero from a clanmate (4.6).
- **C** holds the four whole cards no clanmate plays in ranked (Three Musketeers, Void, Spirit Empress, Goblin Machine), for the top ladder and for later. Without one of them, play a thin spell instead: royal-delivery, freeze, arrows or tornado. C is heavy: play it to overtime.
- **Ownership.** Only the Baby Dragon evo, Boss Bandit and Vines are proven owned. Check the other Tier 1 evolutions, Berserker hero and the whole cards in the game before deck A (open question 2).
- **Roles.** In each deck, friendly 2 is **gold** (held out). Friendlies 1, 3 and 4 are training. Deck A's friendly 4 is **dev**, used for early stopping only.
- **Playing time.** 12 friendlies at about 7 minutes each is about 1.5 h. Split it over evenings if you like.

### 4.5 Tier 2: the common evolutions and heroes (8 decks × 3 friendlies)

**This table is current** (revised 2026-10-30 from the clan data, 4.8). It replaces the ladder-based table of special-slot triples: none of those triples survives, because their items rank far apart in the clan (Knight hero .207 sat beside the Minion Horde evo .029). All eight decks are complete: the plain cards are part of the deck, mostly a playable shell plus spells that are thin on KataCR's enemy side.

| Deck | Slot 1 (evo) | Slot 2 | Slot 3 | Slots 4-8 | Shape |
|---|---|---|---|---|---|
| D | goblin-barrel-ev1 | knight-hero (if owned) | princess-ev1 | goblin-gang, ice-spirit, inferno-tower, rocket, the-log | (2, 1, 0) |
| E | electro-giant-ev1 | bowler-hero (if owned) | pekka-ev1 | electro-wizard, goblin-hut, lightning, tornado, barbarian-barrel | (2, 1, 0) |
| F | minion-horde-ev1 | ice-wizard-hero (if owned; else goblinstein) | skeleton-army-ev1 | mortar, cannon-cart, fireball, barbarian-barrel, royal-delivery | (2, 1, 0) |
| G | royal-ghost-ev1 (else pekka-ev1, with royal-ghost in P.E.K.K.A's place) | electro-wizard-hero | battle-ram-ev1 | bandit, pekka, fireball, magic-archer, zap | (2, 1, 0) |
| H | royal-hogs-ev1 | barbarian-barrel-hero (if owned) | goblin-cage-ev1 | arrows, fireball, flying-machine, royal-recruits, zappies | (2, 1, 0) |
| J | elite-barbarians-ev1 (probably not owned; else lumberjack-ev1) | giant-hero (if owned) | wizard-ev1 | phoenix, elixir-collector, heal-spirit, mini-pekka, freeze | (2, 1, 0) |
| I | musketeer-ev1 | mini-pekka-hero (if owned) | valkyrie-hero (if owned; else inferno-dragon-ev1) | arrows, knight, mega-knight, skeleton-barrel, the-log | (1, 2, 0) |
| K | cannon-ev1 | balloon-hero (if owned) | tombstone-hero (if owned; else furnace-ev1, with tombstone in Furnace's place) | ice-golem, miner, barbarian-barrel, furnace, graveyard | (1, 2, 0) |

- **Together** they cover 10 heroes and 14 evolutions. With Tier 1, every hero and evolution in .025 or more of clan decks is in a capture deck.
- **Play order (greedy, gaps per deck): A, B, D, C, E, F, G, H, J, I, K.** The table lists Tier 2 in that order. Run the first fine-tune after A, B, D and C (15 friendlies), then label E-K with its model, so it auto-accepts what it knows.
- **A slot-2 hero you do not own** (D, E, H, I, J, K): play the next owned hero of the Tier 3 reserve (goblins, magic-archer, mega-minion, musketeer, wizard, then ice-golem), but skip a hero whose card is already in the deck (I skips musketeer, J skips wizard, K skips ice-golem). If none is owned, play the Electro Wizard hero, unless the deck holds an Electro Wizard (E does). Then get the hero from a clanmate friendly (4.6).
- **Slot-3 fallbacks.** I's Inferno Dragon evo repeats A on purpose: it is the third most-played gap, and evolutions need many evolved plays (4.2 rule 9). K's Furnace evo adds a new item.
- **Deck notes:**
  - D carries Knight hero, the clan's most-played gap: 9 of 28 ranked players use it.
  - E is heavy (4.75 elixir on average): play it to overtime. Its P.E.K.K.A evo is proven owned.
  - F's Skeleton Army evo soldiers merge to `skeleton-evolution`, but its General and Spectral units have no class.
  - G: only the Electro Wizard hero is proven owned. Its Battle Ram evo is thin (4.8).
  - J: Heal Spirit checks the N01 fix on gold. Phoenix is thin (22 enemy boxes).
  - K: Furnace's KataCR class is the 2024 building, and Graveyard was reworked. K's gold checks the Furnace interval spawns and the Graveyard skeletons.
- **Look changes.** All six possible look changes of 1.9 are now in a deck: Furnace and Graveyard in K, Goblin Hut in E, Three Musketeers and the Goblin Drill dig in C, Goblin Barrel in D.
- **Roles.** Friendly 2 of each deck is gold; friendlies 1 and 3 train.
- **Playing time.** 24 friendlies is about 2.8 h, over several evenings.

### 4.6 Tier 3 and the clanmates

**The long tail** (current, from the clan data). The Tier 3 list was ladder-based; Goblin Cage, Electro Giant, Ice Wizard, Electro Wizard and base Berserker have moved into capture decks.
- **Evolutions:** Dart Goblin (clan share .019), Furnace (.011; K's fallback), Ice Spirit (thin, .007), Mega Knight, Zap (thin), Executioner, Hunter, Lumberjack (0 in ranked but 8 players in all modes; J's fallback), Electro Dragon, Goblin Giant, Bomber (thin).
- **Heroes:** Dark Prince (.011), Goblins, Mega Minion, Magic Archer, Musketeer, Wizard, Ice Golem.
- **Others:** Clone, Mirror. The Royal Chef tower troop (.036 of clan decks) has no class, but towers are ignore classes and `vision_autolabel label` refuses them.

`vision_gap_labels plan` (N16) builds the decks for these. Re-run the clan ranking after each pull; the Furnace, Lumberjack and Zap evolutions are the first to watch. Until then they rely on the base label (when the look is close) or on the UNKNOWN safety net.

**Field gold** (the true target):
- 2-3 friendlies against clanmates, with the bot only watching (runbook section 17). Pass `--record-images 5` to `vision_play`; its default keeps only one image in 20.
- **Choose partners by deck:** a clanmate whose usual ranked deck holds at least two Tier 1 or Tier 2 gap items and is not itself a capture deck. The pull on the PC shows who.
- **Ask for their usual ranked deck, not an event deck.** A friendly uses their own collection. For the 17 clanmates with both kinds of battles, the Oct 24-30 AllCards event overstates cards they may not own: Minion Giant 0 in ranked against .155 in the event, Ice Wizard hero .063 against .238, base Knight .020 against .140.
- **Record field gold before 2026-11-02,** when Season 89 is planned to bring Evo Giant and Evo Mirror, and record one more after it. Neither can be labelled until RoyaleSim ships a table with them.
- **Take the deck from the battle log.** After the friendly, run `opponents.py pull`; Yodle2's spar record then holds the partner's 8 cards with forms. It matched 8 of 8 on 10-29. Use that string at `ingest` and as the gold deck. Confirm it with the partner if Knight, Musketeer, Valkyrie or Wizard (or Giant after 11-02) sits in slot 3. These are never trained on.
- **Never put a name or tag in a session id or a session note.** `session.json` is committed.
- **Cards and heroes your main account does not own** come from clanmate training friendlies. They play their own deck against Yodle2, which is idle or played lightly. Label them as training, never gold:

  ```powershell
  & $BB tools\vision_autolabel.py ingest reports\vision\harvest\$SID --sid 20261103_CM_1 --role train --deck "<their 8 cards>" --client 16.635.1
  ```

  Pass `--role train`. `--capture field` is gold only, and with `--role auto` a `_CM_2` session would become gold.
- How many clanmates carry each hero or whole card in their ranked decks (and in how many most-played decks): Knight hero 9 (6), Bowler hero 8 (4), Berserker hero 5 (1), Ice Wizard hero 3 (2; 17 players in all modes), Valkyrie hero 3 (0), Mini P.E.K.K.A hero 2 (2), Barbarian Barrel hero 2 (2), Balloon hero 2 (1), Tombstone hero 2 (0), Giant hero 1 (1); Vines 4 (4), Suspicious Bush 2 (2), Goblin Demolisher 2 (1), Goblin Curse 2 (1), Goblinstein 1 (1).

### 4.7 Time and disk

| Step | Playing | Notes |
|---|---|---|
| Round 0 | none | the existing friendly |
| Round 0b (optional) | about 7 min | 1 natural gold friendly with your usual deck |
| Tier 1 | about 1.5 h | 12 friendlies |
| Tier 2 | about 2.8 h | 24 friendlies |
| Field gold | about 20 min | 2-3 friendlies, bot watching |

**Disk.**
- About 1.2-1.7 GB per harvest battle, so about 45-60 GB for Tiers 1 and 2. The disk is 500 GB.
- Keep gold raw images. Delete training raws after `emit`; the labels can always regenerate the boxes while the raws exist.

### 4.8 Clan data (2026-10-30)

The owner's `opponents.py pull` of 2026-10-30 holds 1,286 battles of 44 clan players. The store stays on the PC (gitignored). This section holds aggregate card statistics only: no names, no tags, no single player's deck. Nothing was run against images or the detector.

**What is counted:**
- **Clan decks** (called ranked decks here): the 588 collection-mode decks of the 28 clanmates who have them. That is Path of Legends, ladder, river race, Showdown, and the owner's one Friendly against Yodle2 (the Round 0 deck). The 688 event decks of 33 players are kept apart, because an event deck need not be the player's own cards.
- **Share:** each of the 28 players weighs 1/28, and each of a player's decks 1/n. So one battle of a 2-battle player weighs 1/56, as much as 15 battles of a 30-battle player.
- **Gap:** a card+form whose detector status is not `yes`:
  - `no`: no class, so the card cannot be named;
  - `base-only`: an evolution or hero whose base card has a class;
  - `spell-thin` and `thin`: a label with 25 or fewer enemy real boxes. `thin` is for troops, buildings and towers: the Battle Ram evolution (0 boxes in v0.14) and base Battle Ram (17), Phoenix (22), the Ice Spirit (10) and Bomber (16) evolutions, and Dagger Duchess;
  - `look-changed`: Furnace. KataCR's `furnace` is the 2024 building; 16.402's `Furnace_rework` walks;
  - `not-a-unit`: Mirror.
- "No class" counts `no`, `base-only` and `not-a-unit`; "cannot name" is `no` only.
- **Only the 8 deck cards count.** Tower troops are left out (4.8.5).
- **Covered:** the card+form is in a capture deck; an evolution also covers its base form. This is an upper bound: it assumes the labels reach their thresholds and pass the gate (6).

#### 4.8.1 Headline coverage

"Ladder plan" is the 4.4-4.5 tables before this revision; "revised" is the current tables.

| Stage | Friendlies | Clan decks with a gap | Gaps per deck | No class | Cannot name | Main decks fully covered |
|---|---|---|---|---|---|---|
| Today | 0 | 95.0% | 2.85 | 94.7% | 28.9% | 1-2 of 28 |
| Ladder plan, Tier 1 | 12 | 91.7% | 1.88 | 90.3% | 1.7% | 2-3 |
| Ladder plan, Tier 1+2 | 36 | 48.4% | 0.58 | 38.1% | 1.7% | 14-15 |
| ... with its suggested fillers | 36 | 41.3% | 0.46 | 38.1% | 1.7% | 15-16 |
| **Revised, Tier 1** | 12 | 90.6% | 1.54 | 89.5% | 0.0% | 2-3 |
| **Revised, Tier 1+2** | 36 | **7.1%** | 0.08 | 6.7% | 0.0% | 27-28 |
| Revised, worst case | 36 | 69.8% | 0.80 | 69.3% | 5.0% | 9-10 |
| Ladder plan, worst case (with fillers) | 36 | 84.2% (83.3%) | 1.25 (1.13) | 82.8% | 6.7% | 4-5 |

- **Worst case:** the main account owns no hero but the Electro Wizard hero, and no Elite Barbarians evolution. Each deck then uses its named fallback; a slot-2 hero that has only the reserve rule is lost. Other evolutions and whole cards count as owned.
- **Main decks fully covered:** the clanmates whose most-played deck has no gap left. It is a range because one player has two decks played once each.
- **"Decks with a gap" is harsh:** one leftover card keeps a deck counted. "Gaps per deck" is the fairer view of progress.
- **In-sample.** The revised decks were chosen on these same ranked decks. On decks they were not chosen on:

  | Deck set | Today | Revised Tier 1+2 | Ladder plan Tier 1+2 | ... with fillers |
  |---|---|---|---|---|
  | Event decks, 33 clan players | 98.8% | 31.6% | 69.2% | 66.1% |
  | Ranked and event decks, 44 clan players | 97.2% | 21.3% | 59.1% | 54.7% |

- **Sensitivity** of the revised 7.1%:
  - counting troop labels with up to 30 enemy boxes as thin gives 19.3%, mostly the Firecracker evo (.061, 30 boxes), the Valkyrie evo (.050) and the Royal Giant evo (.017);
  - if the Goblin Hut and Graveyard reworks also changed their look, it stays 7.1% (both are in E and K), and the ladder plan gives 54.2%;
  - counting tower troops gives 14.4%, with 3.6% that cannot be named (Royal Chef);
  - if every slot-3 read of a dual-form card were the other form (4.8.4), it gives 7.4%.
- **Left after the revised Tier 1+2** (clan share): Dart Goblin evo .019, Dark Prince hero .011, Furnace evo .011, Goblins hero .008, Mega Minion hero .007, Ice Spirit evo .007, and six items below .007. Among the event-only players, the largest leftovers are the Furnace, Lumberjack and Zap evolutions.

**Along the play order** (A first: its whole cards cannot be named today):

| After deck | A | B | D | C | E | F | G | H | J | I | K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Friendlies | 4 | 8 | 11 | 15 | 18 | 21 | 24 | 27 | 30 | 33 | 36 |
| Decks with a gap | 94.7% | 94.1% | 86.6% | 74.1% | 62.8% | 47.0% | 40.2% | 32.0% | 23.4% | 17.1% | 7.1% |
| Gaps per deck | 2.27 | 1.83 | 1.48 | 1.18 | 0.93 | 0.73 | 0.58 | 0.44 | 0.31 | 0.18 | 0.08 |

#### 4.8.2 Priority list

The 25 most-played card+forms the detector lacks or barely knows, by clan share.
- **Share:** in ranked decks, with the 90% interval of a bootstrap over players. **All modes** adds event decks.
- **Ladder plan / Revised:** the deck that carries the item before and after the revision.
- **Owned:** "yes" when the main account's own battles show it, "if owned" for every hero not yet proven.
- **Forms** come from the slot rule (4.8.4). Only slot-3 reads could be the other form: Knight hero holds .039 of its .207 in slot 3.

| # | Card+form | Detector | Share [90% CI] | All modes | Players ranked/all | Ladder plan | Revised | Owned |
|---|---|---|---|---|---|---|---|---|
| 1 | knight-hero | base-only | .207 [.10-.32] | .169 | 9/15 | T2 D | D | if owned |
| 2 | bowler-hero | base-only | .150 [.04-.26] | .104 | 8/16 | T2 J | E | if owned |
| 3 | inferno-dragon-ev1 | base-only | .138 [.05-.24] | .099 | 6/10 | T1 B | A | |
| 4 | baby-dragon-ev1 | base-only | .122 [.04-.22] | .128 | 6/16 | T1 A | B | yes |
| 5 | zap | spell-thin | .117 [.04-.21] | .096 | 8/11 | T1 C | C, G | |
| 6 | lightning | spell-thin | .112 [.04-.22] | .134 | 5/17 | T1 C | B, E | |
| 7 | giant-snowball-ev1 | base-only | .099 [.03-.18] | .093 | 6/13 | T2 D | B | |
| 8 | rocket | spell-thin | .093 [.03-.17] | .060 | 8/10 | suggested filler | A, D | |
| 9 | goblin-drill-ev1 | base-only | .092 [.02-.18] | .067 | 5/9 | T2 K | C | |
| 10 | skeleton-barrel-ev1 | base-only | .091 [.03-.17] | .060 | 7/9 | T2 F | A | |
| 11 | vines | no | .085 [.02-.17] | .056 | 4/7 | T1 A | A | yes |
| 12 | witch-ev1 | base-only | .082 [.01-.18] | .018 | 3/5 | T2 H | C | |
| 13 | ice-wizard-hero | base-only | .074 [.01-.15] | .148 | 3/17 | T3 | F | if owned |
| 14 | princess-ev1 | base-only | .072 [.02-.14] | .035 | 5/5 | T2 I | D | |
| 15 | electro-giant-ev1 | base-only | .071 [0-.14] | .087 | 2/11 | T3 | E | |
| 16 | skeleton-army-ev1 | base-only | .069 [.02-.14] | .053 | 3/7 | none | F | |
| 17 | electro-wizard-hero | base-only | .068 [0-.14] | .129 | 2/10 | T3 | G | yes |
| 18 | musketeer-ev1 | base-only | .058 [.01-.13] | .039 | 3/3 | T2 F | I | |
| 19 | royal-hogs-ev1 | base-only | .054 [.01-.12] | .056 | 4/8 | T2 E | H | |
| 20 | berserker-hero | no | .050 [.01-.11] | .049 | 5/7 | T1 B | A | if owned |
| 21 | mini-pekka-hero | base-only | .048 [0-.11] | .066 | 2/7 | T2 G | I | if owned |
| 22 | royal-ghost-ev1 | base-only | .048 [0-.11] | .041 | 2/6 | T1 C | G | |
| 23 | barbarian-barrel-hero | base-only | .046 [0-.12] | .039 | 2/5 | T2 J | H | if owned |
| 24 | goblin-demolisher | no | .046 [0-.11] | .054 | 2/7 | T1 A | A | |
| 25 | suspicious-bush | no | .046 [0-.10] | .020 | 2/3 | T1 A | A | |

**Next 24, down to .01** (share, revised deck):
- goblin-barrel-ev1 .044 (D); elite-barbarians-ev1 .042 (J, probably not owned); cannon-ev1 .039 (K); goblin-cage-ev1 .038 (H);
- giant-snowball .037 (B, via its evo); battle-ram-ev1 .036 (thin, G); wizard-ev1 .036 (J); giant-hero .036 (J); pekka-ev1 .033 (E, owned);
- goblin-curse .032 (A); goblin-barrel .031 (D, via its evo); balloon-hero .030 (K); minion-horde-ev1 .029 (F, owned); goblinstein .025 (B);
- valkyrie-hero .025 (I); royal-delivery .025 (F); dart-goblin-ev1 .019 (Tier 3); ronin .018 (B); berserker .017 (B); tombstone-hero .015 (K);
- furnace .015 (look-changed, K); dark-prince-hero .011 (Tier 3); furnace-ev1 .011 (Tier 3, K's fallback); phoenix .011 (thin, J).

**Below .01:** rune-giant (B), Minion Giant (B; .004 ranked, but .051 and 10 players in all modes), Boss Bandit (C), freeze (J) and the Tier 3 items of 4.6. Three Musketeers, Void, Spirit Empress and Goblin Machine are in no ranked clan deck (C keeps them). Evo Giant and Evo Mirror (planned for 2026-11-02) are not in the 16.402.22 table, so they cannot be labelled until RoyaleSim ships them.

#### 4.8.3 Deck slot rules in the clan data

From the 1,176 collection-mode sides:
- 8 different cards.
- Slot 1 takes an evolution or a base card.
- Slot 2 takes a hero, a champion or a base card.
- Slot 3 takes an evolution, a hero, a champion or a base card (1,020, 116, 15 and 25 sides).
- Slots 4-8 take base cards only. Champions sat only in slots 2 and 3; two champions together, once.
- So: at most 2 evolutions, at most 2 heroes and champions together, at most 3 special cards, as 1.10 says.
- The owner confirmed the rule on 2026-10-30: two evolutions and one hero, or one evolution and two heroes; never three of one kind; fewer is allowed. That statement did not mention champions; the data shows a champion taking a hero's place.
- Every capture deck in 4.4-4.5, and every deck its fallbacks can produce (73 variants), follows these rules with 8 different cards.

#### 4.8.4 Forms: how they are read, and where -ev1 or -hero could be wrong

**How `opponents.py` reads a form.** The API's `evolutionLevel` is a bit mask: 1 evolution, 2 hero. Whether a battle-log card's bits mean the form played or the forms owned is unconfirmed (SEASON_UPDATES 7.1). So the default `form_rule` `slot` reads a form only where the slot can take it:
- slot 1: evolution if bit 1;
- slot 2: hero if bit 2;
- slot 3: whichever bit is set, and no form (ambiguous) when both are;
- slots 4-8: base.

This is right under either meaning, as long as the log lists cards in slot order.

**Where it could be wrong:**
- **Only in slot 3,** and only for cards that have both an evolution and a hero (`maxEvolutionLevel` 3): Knight, Musketeer, Valkyrie and Wizard, and Giant from 11-02. A card with one form cannot be mislabelled, and slots 1 and 2 force the form.
- **A single bit means the same form under both meanings.** If the field is the form played, it is that form. If it is the owned forms, the player owns only that one, so it is still that form. If a player owns both forms, an ownership field shows 3, which the rule reads as ambiguous rather than wrong. So a wrong -ev1/-hero label needs a field that is neither: for example a single "highest owned" value.
- **If the log's card order were not the slot order,** every form would be suspect.

**What the pull shows** (2,572 sides):
- 0 of 5,083 form-capable cards in slots 4-8 carry a bit. An ownership field would show many, so the field behaves like the form played.
- No card carries both bits (0 ambiguous). Slot 1 never shows a hero bit, and slot 2 never an evolution bit.
- No side has 7 cards: battle logs keep the slot-2 hero.
- On every card, the slot rule and the field rule agree.

**Slot-3 reads of the dual-form cards** (all modes, evolution/hero): Knight 51/34, Valkyrie 77/6, Musketeer 35/6, Wizard 17/4. If every one of them were the other form:
- Knight hero would fall from .207 to .185;
- Valkyrie hero would rise from .025 to .072 (about rank 13), and Valkyrie evo fall from .050 to .004;
- today's 95.0% would become 96.8%, and the revised 7.1% would become 7.4%.

**Checking it.** `opponents.py probe` settles the meaning of the field (4.3). Its own verdict skips slot 3, so check slot 3 by hand:
1. After a friendly with a dual-form card in slot 3 (I's Valkyrie hero or J's Wizard evo), read the probe's "latest" card table.
2. If you own both forms of that card, 2 or 1 means the log shows the form played; 3 means it shows ownership.

Knight in slot 3 would be the most useful check, but no capture deck has it.

#### 4.8.5 Tower troops

| Tower troop | Detector | Share of clan decks | Players |
|---|---|---|---|
| Tower Princess | yes | .847 | 25 |
| Dagger Duchess | thin (0 boxes in v0.14) | .037 | 2 |
| Royal Chef | no class | .036 | 1 |
| Cannoneer | yes | .026 | 1 |
| not in the log | | .055 | |

- Towers are ignore classes in `events.py`, so a tower troop cannot cause a misread play, only wrong tower stats (1.6).
- `vision_autolabel label` refuses towers. A Chef or Duchess tower in a capture friendly would only add flagged tracks to mask, so keep the Tower Princess there.
- Royal Chef stays Tier 3 until there is a tower-label path.

#### 4.8.6 What changed in the capture decks, and why

**Kept:**
- all 14 whole cards in Tier 1, 3 decks × 4 friendlies;
- 8 Tier 2 decks × 3 friendlies;
- the roles, the play rules of 4.2 and the dev friendly.

**Moved:**
- **Tier 1 special slots:** from the top ladder's favourite evolutions to the clan's.
- **Down to Tier 2:** the Elite Barbarians, Cannon and Royal Ghost evolutions, Balloon hero, Heal Spirit and Freeze.
- **Promoted from Tier 3:** Ice Wizard hero (.074, and 17 players in all modes), the Electro Giant evo, Electro Wizard hero, the Goblin Cage evo and base Berserker.
- **Into a deck for the first time:** the Skeleton Army evo, Rocket and Royal Delivery (the latter two were only suggested fillers), the Battle Ram evo, Phoenix and Furnace.
- **Dropped to Tier 3:** Dark Prince hero (.011), the Executioner evo (.002), Musketeer hero (.001), Mega Minion hero (.007), Wizard hero (0) and the Electro Dragon evo (0 in ranked decks).

**Why:**
- The ladder tables left five of the clan's 17 most-played gaps in Tier 3 or in no tier: Ice Wizard hero, Electro Giant evo, Skeleton Army evo, Electro Wizard hero and Rocket.
- The revised tables use the same 36 friendlies, and they take the clan's leftover from 48.4% to 7.1%.

## 5. Labelling steps and time

The commands are those planned in N08-N12.

```powershell
& $BB tools\vision_autolabel.py ingest reports\vision\harvest\$SID --deck "skeleton-barrel-ev1,berserker-hero,inferno-dragon-ev1,vines,...,rocket" --role auto --client 16.635.1
& $BB tools\vision_autolabel.py scan $SID            # unattended, a few minutes on the GPU; MuMu closed
& $BB tools\vision_autolabel.py sheet $SID           # writes the numbered contact sheet PNG and its legend
& $BB tools\vision_autolabel.py label $SID --map "0=3,1=3,2=x,5=s"
& $BB tools\vision_autolabel.py review $SID          # spot-check pages
& $BB tools\vision_autolabel.py mark $SID --bad-tracks 17,40 --bad-frames 3,9
& $BB tools\vision_autolabel.py plays $SID           # gold battles only: confirm the list of plays
```

**The sheet.**
- Each tile shows one cluster of tracks:
  - 4 member tracks, each at birth, at +1 s and at mid-life;
  - the number of tracks;
  - what tonight's model read it as;
  - a suggested name.
- The legend is printed from the declared deck: each card's classes and its enabled forms, with numbers. Four extra codes:
  - `x`: an effect, not a unit;
  - `o`: ours;
  - `?`: unsure (masked in training);
  - `s`: split (those tracks are shown again, one per tile).
- **Checks.** Every name is checked against `classes.json` (`labels.resolve_label`). A wrong name stores nothing.

**Review.**
- Page A shows 3 random tracks per cluster with the propagated box.
- Page B shows 24 random emitted frames, half from birth windows.
- More than 2 bad frames in 24 names the clusters to recheck.

**Gold battles.** Review every track, then confirm the derived play list. Delete wrong plays with `--bad-plays`, and add missed ones with `--add-play "1:52=Vines"`.

**Owner time per battle (estimates):**

| Battle | Ingest | Sheet and label | Review | Plays | Total |
|---|---|---|---|---|---|
| Training | 1 min | 5-7 min | 2 min | none | about 9 min |
| Gold | 1 min | 5-7 min | about 15 min | about 5 min | about 27 min |
| Round 0 | 1 min | about 10 min (30-45 clusters) | about 15 min | about 5 min | about 30-45 min |

**Totals:**
- **Tier 1:** about 1.4 h for training battles plus 1.4 h for gold.
- **Tier 2:** shorter per battle, because the Tier 1 model auto-accepts what it knows. About 1.5 h for training battles plus 3 h for gold.
- **Field gold:** about 15 min each.
- **Overall:** about 8-9 h of labelling over 2-3 weeks, alongside about 4.5 h of play.

**Unattended GPU time:**
- The scan takes a few minutes per battle: two detector passes per image.
- A fine-tune takes about 3-5 h, so run it overnight with MuMu and the browser closed.
  - Tonight's 6-10 h for 5,551 images × 150 epochs works out to about 23-38 images per second.
  - 40 epochs on about 10k images then takes 3-5 h.

## 6. Measurement before live use

**Data splits are by session, never by frame.** KataCR's own val split is random over frames, with adjacent near-duplicates, so its mAP is optimistic.
- **Gold:** owner-reviewed and frozen, never trained on. Three kinds:
  - natural (the existing friendly);
  - scripted (friendly 2 of each deck);
  - field (clanmate friendlies).
- **Dev:** used only for early stopping.
- **Baseline:** tonight's model on the same data.

`tools/vision_gap_eval.py` (N14) scores the baseline and the candidate on the same gold. It writes `report.md` and `gate.json`.

### 6.1 Detection

On gold frames, through the live config (conf 0.35, IoU 0.5), using `vision_replay.match_boxes` and `PRCounter`:
- **Each new label with at least 15 gold tracks:**
  - recall ≥ 0.85 and precision ≥ 0.85, card-level (form ignored);
  - form-aware numbers reported;
  - 95% intervals from a bootstrap over tracks.
- **Side accuracy ≥ 0.98.**
- **Birth-window recall ≥ 0.85:** the share of gold enemy tracks whose class is the top box in at least 2 of their first 4 images. These frames decide the card.

### 6.2 Plays

Run `tools/vision_replay.py <gold> --detector <W> --ticks time` (the full live chain), then match its OppPlays to the gold plays within 1 s and 3 tiles.
- **Each new card with at least 4 gold plays:** play recall ≥ 0.85 and card precision ≥ 0.90.
- **Each evolution or hero with at least 4 gold plays:** card accuracy ≥ 0.90. Form accuracy is reported, with 0.8 as the target.
- **Pooled over all gold:**
  - play-ID accuracy ≥ 0.85, and at least 0.15 above the baseline;
  - phantom plays ≤ 0.5 per minute;
  - elixir error ≤ 1.5 per minute (the cost of missed and phantom plays, plus the cost difference of wrong-card plays).
- **Safety net (N02):** at least 90% of UNKNOWN plays must match a real play. The UNKNOWN share at resolve time is reported.
- **Marker (N03):** stopwatch recall is reported. `require_marker` is turned on only at 98% or more.

### 6.3 Regression

- KataCR val mAP50 over the original 310 classes drops by at most 0.01.
- Each of our Hog 2.6 labels loses at most 0.01 recall on gold.
- New labels fire on at most 1% of our own boxes. Under the deck rule, such a box would become a phantom enemy play.
- The own-landing miss rate is no worse.

### 6.4 Thin items

An item below the evidence thresholds is reported as "insufficient" and does not gate. `status` lists it as what to record next.

### 6.5 Live

After a PASS:
1. Run one dry-run friendly (runbook section 17) against a clanmate deck with post-2024 cards.
2. Compare the OppPlay log with what happened.
3. That recording becomes field gold.
4. Only then pin: `& $BB -m scripts.vision.detector pin <best.pt>`. Fine-tunes always train with `--no-pin`.

## 7. A new season

1. Regenerate `classes.json` from the new table (`tools/vision_classes.py --engine`), as part of `docs/SEASON_UPDATES.md` when that exists.
2. Run `vision_gap_labels report`. It lists:
   - every pool class-form with no KataCR name and too few labelled tracks or gold plays;
   - every label whose card has no composition or spell in the event tables. That catches Heal-type gaps automatically.
3. Until labelled, new cards are covered by the safety net (UNKNOWN plus expected cost).
   - A card the client has but `classes.json` lacks cannot be labelled at all: `resolve_label` refuses it. It stays UNKNOWN until RoyaleSim ships the table.
4. Play 1-2 decks with the new cards, 4 friendlies each: about 30-60 min of play and 30-45 min of labelling.
5. Scan with the pinned model, which auto-accepts what it knows, then label the rest.
6. Run `mix`, fine-tune overnight, gate, run a dry run, then pin.
7. **Reworks.** A reworked card gets a `valid_from_client` in the registry, and older sessions of that label are filtered out.
8. **Rebase.** Once in a while, retrain from `yolov8s.pt` on the whole mix overnight, so fine-tunes do not drift.

## 8. Tonight's detector run: go ahead unchanged

**Yes: run runbook section 15 exactly as written.**

1. **No labels need reserving.** ultralytics 8.4.192 copies the class rows by name when labels are appended later (2.3), so do not add `--extra-label`.
2. **The default settings get a sensible optimizer.**
   - With `optimizer=auto`, the iteration count is ceil(images / 64) × epochs (`engine/trainer.py:329`).
   - At 10,000 or fewer, ultralytics picks AdamW at lr0 = 0.01 / (4 + classes). Above that, it picks MuSGD at 0.01 (`trainer.py:1186-1187`).
   - The default run is ceil(5551 / 64) × 150 = 13,050 iterations, so it gets MuSGD at 0.01.
3. **The model becomes M0,** the baseline for every gate in section 6.

**Two cautions:**
- **Keep the default 150 epochs tonight. Do not use the `--epochs 60` shortcut.**
  - Below 115 epochs the run drops under 10,000 iterations.
  - It would then get AdamW at about 3.2e-5 for 310 classes, a setting not tried here.
  - The same rule is why later fine-tunes pass `--optimizer` and `--lr0` explicitly (N06).
- **Send back the full output of `prepare_yolo_dataset`.** It can append KataCR names that the class list does not map. The registry (N07) must extend the real `tmp\yolo\detector_labels.json`, not the documented 155.

**What not to read into tonight's numbers.** Tonight's val mAP comes from a random frame split of the same KataCR episodes. It says nothing about the clanmates' newer cards.

## 9. Work items

Tests go in `tests/pipeline`. They use synthetic fixtures and `FakeModel`-style detectors (as in `test_detector.py`), so they need no GPU and skip cleanly. A test that runs real YOLO is marked `gpu`. No item talks to the emulator, so none needs `device`.

| ID | Title | Files | Depends | Tests |
|---|---|---|---|---|
| N01 | Event-table fixes: Heal Spirit composition, Spirit Empress alias at cost 3 or 6, regenerate spawn children from 16.402.22 where readable, BarrelGoblin mapping | `tools/vision_spawn_children.py`, `data/vision/spawn_children.json`, `scripts/vision/events.py` | none | `test_events.py` additions: a HealSpirit track gives a Heal play; MergeMaiden_Mounted gives MergeMaiden at 6; old tables unchanged otherwise |
| N02 | UNKNOWN safety net: unexplained enemy tracks become `OppPlay(card=-1)`; expected-cost elixir charge; unknown enemy rows kept as card -1; `opp_cards` skips -1 | `scripts/vision/events.py`, `scripts/vision/types.py`, `tools/public_state.py`, `scripts/vision_play.py`, `tools/vision_replay.py` | none | `test_events_unknown.py`, `test_public_state_unknown.py`; golden: with `unknown_plays` off, existing events and states are identical; old recordings load |
| N03 | Stopwatch play marker (off by default): enemy `clock` tracks; an unmatched one gives an UNKNOWN play; `require_marker` toggle | `scripts/vision/markers.py`, `scripts/vision/events.py`, `scripts/vision_play.py`, `tools/vision_replay.py` | N02 | `test_markers.py`: marker tracks, attach window, require_marker drops curse-like goblins |
| N04 | Recording hygiene: `capture --budget-mb`; `vision_replay --conf/--imgsz/--max-det` | `scripts/vision/capture.py`, `tools/vision_replay.py` | none | `test_capture.py`, `test_latency_and_replay.py` additions |
| N05 | Hooks: `Tracker.last_assignment`; `EventDeriver.how_of(uid)`; `detector.crop_rect_for` | `scripts/vision/tracker.py`, `scripts/vision/events.py`, `scripts/vision/detector.py` | none | `test_tracker.py`, `test_events.py`, `test_detector.py`: crop_rect_for equals `Detector.crop_rect` at 747x1328 |
| N06 | Fine-tune flags: `--optimizer`, `--lr0`, `--cls-pw`, `--freeze`; warn on the default `--name` with a `best.pt` start | `scripts/train_detector.py` | none | `test_train_detector_flags.py` with a fake YOLO: the flags reach `model.train` |
| N07 | Append-only label registry, seeded with the 19 phase-1 labels (16 units and 3 spells, 38 YOLO classes); `train_as:<base>` for thin forms; `valid_from_client` | `data/vision/detector_extra_labels.json`, `tools/vision_autolabel.py` | tonight's `detector_labels.json` (for real use) | `test_label_registry.py`: the old `yolo_names()` stays a prefix; every entry resolves through `resolve_label`; train_as writes the base index |
| N08 | Ingest and status: hard links, deck check (RoyaleAPI keys or catalogue names), refuse event cards, roles, `session.json` | `tools/vision_autolabel.py`, `data/vision/event_cards.json` | N04 helpful | `test_autolabel.py`: ingest on a tiny fake recording; refuses SuperKnight; role assignment |
| N09 | Scan: `ScanPerception` over `ReplayPerception`; low-conf second detector; background median in bands; blobs; position-only harvest tracker | `scripts/vision/unit_harvest.py`, `tools/vision_autolabel.py` | N04, N05 | `test_unit_harvest.py`: moving squares on a synthetic arena; proposal merge; last_assignment |
| N10 | Flags S1-S7 (S5 exempts river jumpers and needs 6 or more images); appearance split; auto-accept; track clustering; sheet and legend; `label` | `scripts/vision/unit_harvest.py`, `tools/vision_autolabel.py` | N09 | flags on hand-made tracks; split on a look change; map parsing; bad names store nothing |
| N11 | Propagate and emit: box smoothing, short-gap fill, spell circles (DarkMagicAOE 2.5, Vines_AeO 2.5, GoblinCurse 3.0 tiles), masking, birth-window oversampling, crop-relative YOLO; full-frame labels for gold | `tools/vision_autolabel.py`, `scripts/vision/unit_harvest.py` | N07, N10 | crop round trip against `crop_rect_for`; indices against the LabelSet digest; gold never written to train dirs |
| N12 | Review, mark and gold plays: review pages, Wilson interval, play derivation from gold tracks | `tools/vision_autolabel.py` | N11 | Wilson arithmetic; plays derived from hand-made gold tracks |
| N13 | Mix: `cr_mix.yaml` with list train/val dirs, the extended `detector_labels.json` next to it, oversampling by hard links, refuses gold | `tools/vision_autolabel.py` | N06, N07, N11 | yaml contents; gold refused; a `gpu` smoke test: 1 epoch logs `Remapped N/M cls head rows` |
| N14 | Gate: detection, birth window, plays, elixir, regression, `gate.json` | `tools/vision_gap_eval.py` | N02, N04, N12 | gate arithmetic on hand-made plays and boxes; the YOLO val part is `gpu` |
| N15 | Perception noise follows the deployed label set; refit unknown, miss and confusion rates from the gate | `tools/perception_noise.py`, `tools/fit_noise_profile.py` | N07, N14 | `test_perception_noise.py`: a form in the deployed set is no longer collapsed to base |
| N16 | Season report and deck planner under the slot rules (at most 2 evolutions, 2 heroes and champions, 3 in all) | `tools/vision_gap_labels.py` | N07, N08 | planner never breaks the slot rules; report flags a Heal-type gap on a fixture |
| N17 | Optional compositor for thin classes (KataCR logic port, MIT notice, fixed balancing, ×1.143 for KataCR sprites) | `tools/synth/compose.py` | N09, N11, N14 | same seed gives the same scene; occlusion drop rule on fixtures |
| N18 | Optional deck prior for look-alike ties, perception-only, off by default | `scripts/vision/deck_prior.py`, `scripts/vision/events.py` | N02 | closure and cycle prior; the prior never reaches `PublicMemory` |

**Order.**
- N01, N02, N04, N05 and N06 can start at once.
- Round 0 needs N04, N05 and N08-N12.
- The first fine-tune needs N13 and N14 as well.

## 10. Risks

1. **Box quality.** Low-confidence boxes and blob boxes may not follow KataCR's convention: the full sprite, feet at the bottom, no HP bar. Mitigations:
   - subtract bar boxes and smooth sizes;
   - re-box in the next round with the new model;
   - check on the review pages.

   Gold IoU measures what is left.
2. **Background model.** Units that stay put leak into the median: our Cannon, buildings, tower ruins. Mitigations:
   - mask pixels under any box;
   - use a separate background before and after a tower falls.
3. **Over-fitting to one arena and one player's placements.** Consecutive frames are highly correlated. Field gold and the natural friendly exist to expose this.
4. **Pseudo-labels.** Labels for our own units, towers and UI come from tonight's model, so its errors are learned again. The regression gates bound this.
5. **Masking artefacts.** Pasted background patches can teach "patch edge = background". Mitigations: feathered edges, at most 3 masked regions per frame, and frames are dropped when a mask overlaps a kept box.
6. **Coverage depends on what the main account owns.** Evolutions appear only on evolved cycles, and hero abilities change the look mid-track. The long tail stays thin. The safety net and `train_as` keep it from vanishing or naming the wrong card.
7. **Client drift.** The client is 16.635.1, newer than the 16.402.22 table. Anything added in between cannot be labelled until `classes.json` is regenerated from a newer table.
8. **The policy side.** UNKNOWN plays and card -1 rows exist in training only as far as perception noise produces them. Until N15 refits the noise, the live rate of unknowns may differ from training.
   - Not checked: whether the engine catalogue (`data/rs_rows/spawn_primary.json` `card_names`, not in the repo here) has ids for the new cards. Without an id, rule 5 still finds no card.
   - Also not checked: whether the token policy has trained embeddings for those ids.
9. **Training defaults.**
   - A short fine-tune with `optimizer=auto` gets AdamW at about 2.8e-5 for 348 classes and barely learns the new ones.
   - `train_detector` pins by default and reuses the run folder (`exist_ok=True`).
   - So every fine-tune passes `--optimizer`, `--lr0`, `--no-pin` and a new `--name`.
10. **Disk and RAM.**
    - The recorder prunes the oldest images under a whole root above 4,000 MB. That is why ingest runs right after each battle, with one root per session until N04 lands.
    - The scan's background median works in bands; run it with MuMu closed.
11. **The stopwatch is unmeasured:** its recall, its fill direction, whether buildings and Spirit Empress show one, and our own side's colour. N03 stays off until gold measures it.
12. **Concurrent edits.** `scripts/vision/harvest.py` keeps moving. Import by name only.

## 11. Open questions

1. ~~Which 8 cards did the main account play in the 2026-10-29 friendly?~~ **Answered (2026-10-30):** boss-bandit, elite-barbarians, executioner, freeze, guards, hog-rider, minion-horde-ev1, zap, read from Yodle2's spar record (4.3).
2. Which capture cards does the main account own? The pull proves only the Electro Wizard hero, the Baby Dragon, P.E.K.K.A and Minion Horde evolutions, Boss Bandit and Vines, and the Elite Barbarians evolution is probably missing (4.3).
   - **Before deck A:** the Skeleton Barrel, Inferno Dragon, Giant Snowball, Goblin Drill and Witch evolutions, Berserker hero, and the whole cards Goblinstein, Ronin, Rune Giant, Minion Giant, Goblin Demolisher, Suspicious Bush, Goblin Curse, Goblin Machine, Spirit Empress, Void and Three Musketeers. Minion Giant is in the client: 10 clanmates played it.
   - **Before Tier 2:** its heroes (Knight, Bowler, Ice Wizard, Barbarian Barrel, Mini P.E.K.K.A, Valkyrie, Giant, Balloon, Tombstone) and evolutions (4.5).
3. Does your client (16.635.1) have cards, evolutions or heroes that are missing from the 16.402.22 table?
4. Do clanmates' friendlies show their own tower troop (Royal Chef, Duchess, Cannoneer), or always Tower Princess? Royal Chef has no KataCR class. 15.3% of clan decks use another tower or have none in the log (4.8.5), so field gold will answer it. Do they use other arenas?
5. Which card id does the engine log for a Spirit Empress play: MergeMaiden, or a variant? N01's alias depends on it.
6. Should the deck prior (N18) exist at all? It uses ladder statistics only for naming what is on screen, never as policy input. It is planned off by default. A clan prior would put Golden Knight near 0 in friendlies: it is in no ranked clan deck, yet the pinned detector reported 10 Golden Knight plays in Round 0, most likely Boss Bandit.
7. Will 2-3 clanmates play a friendly against Yodle2 for field gold, with the bot only watching? Choose them by deck (4.6).
8. Does the enemy side ever show the elixir-cost pop or the card-name text? KataCR's data says these are our side only (elixir 1,493 / 0, text 2,050 / 0). This can be checked on the recorded friendly.
9. What does `opponents.py probe` say about battle-log forms? The pull already suggests the form played (4.8.4). Run it on the existing log; then read slot 3 by hand after the first friendly with Valkyrie or Wizard in slot 3.
