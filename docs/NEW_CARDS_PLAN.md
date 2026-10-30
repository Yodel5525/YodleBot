# New cards: detector and event coverage plan

Status: plan, 2026-10-30. Work items are N01-N18 (section 9). **Built** (code on `sparring`, not yet run on the PC): N01, N02, N04-N14; N03 and N15-N18 are not. Round 0 (4.3) can now run end to end up to a confirmed gold set and the M0 baseline. The capture decks of 4.4 and 4.5 were rechosen from the clan's own battles on 2026-10-30 (section 4.8), then fitted to the main account's collection the same day (4.8.7). The forms the main account lacks come from clanmate friendlies (4.8.8).

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
  - **The main account's collection is known** (official API, 2026-10-30; 4.8.7): all 123 cards, 39 of 43 evolutions, and 4 of 18 heroes (Electro Wizard, Goblins, Ice Golem, Musketeer). It lacks the Elite Barbarians, Goblin Drill, Princess and Electro Giant evolutions and 14 heroes, among them Knight and Bowler.
  - **The capture decks in 4.4 and 4.5 are fitted to that collection** (current, 2026-10-30): 30 friendlies instead of 36, and every deck uses only owned forms. They leave 69.0% of clan decks with a gap, 1.01 gaps per deck. No plan with the main account alone can go below 68.2% (0.98): the 18 forms it lacks are in that many clan decks.
  - **Clanmate friendlies close the rest** (4.8.8): eight clanmates, 27 training friendlies with their own decks, take the 69.0% to 5.3%, and every clanmate's main deck is then fully covered.
  - The earlier clan revision assumed every hero owned and claimed 7.1% with 36 friendlies; the ladder-based decks would leave 48.4% on that assumption.
  - Forms (-ev1 or -hero) come from the deck slot, as `opponents.py` reads them. The 2026-10-30 probe confirmed that the battle log shows the form played. Left to check: slot-2 hero reporting, and slot-3 reads of Knight, Musketeer, Valkyrie and Wizard (4.8.4).
- **Two table bugs.** These hurt even with a perfect detector:
  - Heal Spirit plays are never named.
  - Spirit Empress is named as an internal variant card, not as itself.
- **Plays that go missing.** Today an enemy play the tables cannot explain is dropped. It never reaches the policy, and the opponent's elixir is not charged.
- **The plan has four parts:**
  1. Fix the event tables.
  2. Add a safety net: a play that cannot be named reaches the policy as UNKNOWN and is charged an expected cost.
  3. Record friendlies where the owner's main account plays the new cards against Yodle2, and clanmates play the forms it lacks. Turn them into auto-labelled real frames, and fine-tune the detector with labels appended to tonight's.
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
- The main account lacks 14 heroes and 4 evolutions. Those come from clanmates' training friendlies against Yodle2, played with their own decks (4.8.8).

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
8. **Every capture deck uses only forms the main account owns** (API collection, 2026-10-30; 4.8.7), so no swaps are needed. If the deck builder still refuses a card, swap it and tell `ingest` the real deck and which slots are evolution or hero.
9. **Single-unit evolutions in Tier 2 need hard cycling.** A form with fewer than 8 labelled tracks trains under its base label (2.3). Two evolved plays per friendly give a Tier 2 evolution only 4 training tracks, so aim for 4 evolved plays a friendly. These are Goblin Cage (D), Cannon and Musketeer (E), Dart Goblin (F), Royal Ghost and Battle Ram (G), Furnace and Wizard (H), and P.E.K.K.A (I). Multi-unit evolutions (Royal Hogs, Goblin Barrel, Minion Horde, and Skeleton Army in A) get several tracks per play. `status` lists what to top up.
10. **Play each hero 5 times or more.** 4 plays × 2 training friendlies is only just 8 tracks. The heroes are Electro Wizard (A, G, I), Ice Golem (E), Musketeer (F) and Goblins (H).
11. **Cards with both forms in slot 3** (E's Musketeer evo; H's Wizard evo): write the form you chose in the `ingest` deck string. The main account owns both Musketeer forms, so E is the slot-3 check of 4.8.4.

### 4.3 Round 0: the friendly already recorded (no play)

The 2026-10-29 friendly (22,067 frames, about 4,414 images) has the main account at the top. It becomes the first gold battle: the M0 baseline, plus a table of what tonight's model reads each new card as.

**Copy its folder out of `reports\vision\recordings` before section 16 records more there.** All recordings in that root share the 4,000 MB budget, and the oldest images go first.

**The 8 cards are known** (open question 1 is answered): boss-bandit, elite-barbarians, executioner, freeze, guards, hog-rider, minion-horde-ev1, zap. Yodle2's spar record in the pull matches all 8, forms included: the Minion Horde evo in slot 1, Boss Bandit in slot 2, and Elite Barbarians as a base card in slot 3.

The commands, in order (PowerShell from `C:\hcr`, `$BB` as in the runbook; MuMu and the browser closed for the scan):

```powershell
$R0  = 'reports\vision\gold\capture_20261029_195831'     # the copied session folder (it holds frames.jsonl)
$SID = '20261029_R0_1'
& $BB tools\vision_autolabel.py ingest $R0 --sid $SID --role gold --capture natural --deck "boss-bandit,elite-barbarians,executioner,freeze,guards,hog-rider,minion-horde-ev1,zap" --client 16.635.1
& $BB tools\vision_autolabel.py scan $SID                  # live chain + harvest pass on the GPU, 10-20 min
& $BB tools\vision_autolabel.py sheet $SID                 # open the sheet PNG(s) it names; the legend is printed
& $BB tools\vision_autolabel.py label $SID --sheet 1 --map "0=3,1=3,2=x,5=s"   # every cluster: gold has no auto-accept
& $BB tools\vision_autolabel.py sheet $SID                 # again while clusters are left or were split (s)
& $BB tools\vision_autolabel.py label $SID --sheet 2 --map "0=BossBandit,1=o"
& $BB tools\vision_autolabel.py emit $SID                  # gold: full-frame labels in <store>\$SID\gold, no image
& $BB tools\vision_autolabel.py review $SID                # every gold track (about 10 page-A images) and page B
& $BB tools\vision_autolabel.py mark $SID --review 1 --bad-tracks t17 --bad-frames 3,9   # or --review 1 alone
& $BB tools\vision_autolabel.py label $SID --map "t17=AxeMan" --force   # after a mark: relabel, then emit --force, review
& $BB tools\vision_autolabel.py plays $SID                 # the derived play list, numbered, with M0's read of each
& $BB tools\vision_autolabel.py plays $SID --list 1 --bad-plays p12 --add-play "1:52=freeze" --confirm
& $BB tools\vision_gap_eval.py baseline --gold $SID        # M0's numbers and reads; records M0 for every later gate
```

- **Every track is labelled.** A gold emit is refused while a track is unlabelled (`--allow-pending` makes such tracks ignore regions; their plays are then missing, so `plays --confirm` needs `--allow-pending` too) or while a label is auto-accepted.
- **Edits name the latest list** (`--list K`, printed by `plays`). An edit or `--confirm` is refused when the plays derived now differ from list K's: run `plays $SID` again and check the new list first.
- `& $BB tools\vision_autolabel.py status` says what is left: an unlabelled track, a stale emit, a review not recorded, plays not confirmed.
- Keep the gold raws: the gate replays them through the live chain.

- **Its gap items:** Boss Bandit (no class), Zap and Freeze (thin), the Minion Horde evo (base class only). All four are also in capture decks (C; C and G; I; I), so Round 0 is their natural gold.
- **The 10 GoldenKnight plays** the pinned detector reported. The deck has no Golden Knight, so none of them is one. Label each GoldenKnight cluster after looking at it: Boss Bandit, which has no class yet, is the expected answer. The tracker forces any label outside our deck to the enemy side (1.8), so some may be the main account's Executioner, Elite Barbarians or Guards (label them as those), or Yodle2's own units such as Hog Rider (label them `o`). Golden Knight is in no clanmate's ranked deck.
- **The Elite Barbarians evolution.** Slot 3 showed no evolution bit and the card was played as base. The API collection confirms that the main account lacks that evolution (4.8.7); a clanmate's friendlies supply it (4.8.8).
- **The probe** (`opponents.py probe`, SEASON_UPDATES 7.1) can run on the existing log; it needs no new friendly. Its form test counts any owned form in a slot that cannot take it: Round 0's Executioner (slot 4) and Zap (slot 7), and Lumberjack, Balloon and Barbarian Barrel in slots 4-8 of the main account's usual ranked deck (10 battles in the log); each counts if you own its evolution or hero. Deck A's friendly 1 gives that test nothing: its special cards sit in slots that take their forms, and its five plain cards have no forms. It does test slot-2 hero reporting: the Electro Wizard hero sits in slot 2. The probe on the PC (2026-10-30) found the form played (4.8.4).
- **Optional Round 0b.** One natural friendly with your usual ranked deck, as a second natural gold and, if wanted, a probe friendly against Yodle2 (an evolution in slot 1, a hero in slot 2, all proven owned). It holds four clan gap items: Vines, the Baby Dragon evo, the Electro Wizard hero and the P.E.K.K.A evo. About 7 minutes of play and 27 minutes of labelling.

  ```powershell
  & $BB tools\vision_autolabel.py ingest reports\vision\harvest\$SID --sid 20261101_R0b_1 --capture natural --role gold --deck "<your 8 cards>" --client 16.635.1
  ```

  Pass `--capture natural`: with `--role auto` and the default capture, a session id ending in `_1` trains.

### 4.4 Tier 1: the 14 whole cards (3 decks × 4 friendlies)

**This table is current: fitted to the main account's collection, 2026-10-30** (4.8.7). It replaces the clan-data table of the same day, which assumed every hero owned, and the ladder-based table before it. Every form below is owned, so no deck needs a fallback.

Deck lists use RoyaleAPI keys, as in `data/ladder/battles_2p6.csv`. Minion Giant is given by its catalogue name, `MinionGiant`; since 2026-10-30 (commit 0f3a56e) its key `minion-giant` maps to it too (`data/cards/replay_cards_extra.json`, still `"verified": false` until a corpus deck shows the card), and `tools/vision_autolabel.py` takes either. Slot 1 takes an evolution, slot 2 a hero or champion, slot 3 any of them (4.8.3).

| Deck | Slot 1 (evo) | Slot 2 | Slot 3 | Slots 4-8 | Shape (evo, hero, champion) |
|---|---|---|---|---|---|
| A | skeleton-barrel-ev1 | electro-wizard-hero | skeleton-army-ev1 | vines, suspicious-bush, goblin-curse, goblin-demolisher, rocket | (2, 1, 0) |
| B | baby-dragon-ev1 | goblinstein | giant-snowball-ev1 | MinionGiant, rune-giant, ronin, berserker, lightning | (2, 0, 1) |
| C | inferno-dragon-ev1 | boss-bandit | witch-ev1 | three-musketeers, void, spirit-empress, goblin-machine, zap | (2, 0, 1) |

- **All 14 whole cards are in Tier 1.** Base Berserker rides in B.
- **The six evolutions** are the clan's six most-played owned gap evolutions: Inferno Dragon .138, Baby Dragon .122, Giant Snowball .099, Skeleton Barrel .091, Witch .082 and Skeleton Army .069. Evolutions go in Tier 1 because they look evolved only on their evolution cycle and need the third training friendly; a hero shows on every play.
- **What the collection changed:**
  - A: the Electro Wizard hero takes slot 2 (Berserker hero is not owned; a clanmate supplies it, 4.8.8), and the Skeleton Army evo takes slot 3.
  - C: the Inferno Dragon evo takes slot 1 (the Goblin Drill evo is not owned).
  - B is unchanged.
- **A** holds the whole cards clanmates play: Vines (.085), Goblin Demolisher (.046), Suspicious Bush (.046) and Goblin Curse (.032), with the bait evolutions they come with. Its friendly 1 has an owned hero in slot 2: the slot-2 hero check (4.3).
- **C** holds the four whole cards no clanmate plays in ranked (Three Musketeers, Void, Spirit Empress, Goblin Machine), for the top ladder and for later. The Skeleton Army evo is in A, not C: its skeletons would mix with the Witch evo's on the same side. C is heavy: play it to overtime.
- **Ownership.** All forms were checked against the API collection of 2026-10-30. Check each special card in the deck builder before the deck's first friendly.
- **Roles.** In each deck, friendly 2 is **gold** (held out). Friendlies 1, 3 and 4 are training. Deck A's friendly 4 is **dev**, used for early stopping only.
- **Playing time.** 12 friendlies at about 7 minutes each is about 1.4 h. Split it over evenings if you like.

### 4.5 Tier 2: the common evolutions and owned heroes (6 decks × 3 friendlies)

**This table is current: fitted to the main account's collection, 2026-10-30** (4.8.7). It replaces the clan-data table of 8 decks: those carried 10 heroes, and the main account owns only 4. The letters D-I are reassigned and are the play order. D, F, G and H are built on shells clanmates play; I is built on the owner's own ranked deck.

| Deck | Slot 1 (evo) | Slot 2 | Slot 3 | Slots 4-8 | Shape |
|---|---|---|---|---|---|
| D | royal-hogs-ev1 | royal-delivery | goblin-cage-ev1 | arrows, fireball, flying-machine, royal-recruits, zappies | (2, 0, 0) |
| E | cannon-ev1 | ice-golem-hero | musketeer-ev1 | hog-rider, ice-spirit, skeletons, the-log, fireball | (2, 1, 0) |
| F | goblin-barrel-ev1 | musketeer-hero | dart-goblin-ev1 | goblin-gang, heal-spirit, inferno-tower, rocket, the-log | (2, 1, 0) |
| G | royal-ghost-ev1 | electro-wizard-hero | battle-ram-ev1 | bandit, pekka, fireball, magic-archer, zap | (2, 1, 0) |
| H | furnace-ev1 | goblins-hero | wizard-ev1 | barbarian-barrel, giant-skeleton, goblin-hut, graveyard, poison | (2, 1, 0) |
| I | pekka-ev1 | electro-wizard-hero | minion-horde-ev1 | balloon, barbarian-barrel, lumberjack, tornado, freeze | (2, 1, 0) |
| J (optional) | lumberjack-ev1 | mini-pekka | zap-ev1 | golem, night-witch, bomber, arrows, phoenix | (2, 0, 0) |

- **Together with Tier 1** they cover all four owned heroes, 18 owned evolutions, and every owned gap form in .008 or more of clan decks except Phoenix (.011, in J). Every thin spell the clan plays is in a deck except the Zap evo (J).
- **What each deck adds first** (clan share): D the Royal Hogs and Goblin Cage evos and Royal Delivery (.117); E the Cannon and Musketeer evos and the Ice Golem hero (.097); F the Goblin Barrel evo, the Musketeer hero and the Dart Goblin evo (.095); G the Royal Ghost and Battle Ram evos (.083); H the Furnace and Wizard evos, base Furnace and the Goblins hero (.069); I the P.E.K.K.A and Minion Horde evos and Freeze (.066).
- **Play order (greedy on gaps per deck, given what the decks before cover): A, B, C, D, E, F, G, H, I.** Run the first fine-tune after A, B, C and D (15 friendlies), then label E-I with its model, so it auto-accepts what it knows. The Tier 2 steps are close (1.42 to 1.48 gaps per deck after Tier 1), so the order matters little.
- **No hero you lack is needed.** The reserve-hero rule and the slot-3 fallbacks of the earlier table are gone. D's slot 2 holds Royal Delivery, because no owned hero fits that shell.
- **Deck notes:**
  - D: Royal Delivery is spell-thin.
  - E mirrors Yodle2's Hog 2.6. Its forms show only on your side, but its base cards are on both, so label sides by colour; the S3 "not in deck" flag cannot tell them apart. The main account owns both Musketeer forms, so E is the slot-3 check of 4.8.4.
  - F: Heal Spirit checks the N01 fix on gold. The Goblin Barrel evo shows the base barrel on its other cycles. The Musketeer hero (.021 of clan decks in all modes) brings MusketeerTurret, which has no class.
  - G: the Electro Wizard hero repeats A and I on purpose: it is the clan's largest owned hero gap (.068 ranked, .129 in all modes). Its Battle Ram evo is thin (4.8).
  - H: Furnace's KataCR class is the 2024 building, and Graveyard and Goblin Hut were reworked. H's gold checks the Furnace interval spawns, the Graveyard skeletons and the Goblin Hut.
  - I is the easiest to play well. It gives the largest single drop in decks with a gap (75.1% to 69.0%).
  - J (optional): the Lumberjack and Zap evos are in no or almost no ranked clan deck, but they are the largest owned leftovers in all modes (.057 and .043). Play J if the next pull shows them in ranked; on the clan's event decks it is worth 5.7 points (11 with the clanmate friendlies of 4.8.8).
- **Look changes.** Five of the six possible look changes of 1.9 are in a deck: Furnace, Graveyard and Goblin Hut in H, Three Musketeers in C, Goblin Barrel in F. The Goblin Drill dig comes with a clanmate's friendlies (4.8.8).
- **Five decks instead of six** (drop H) would leave 69.8% and 1.08 gaps per deck.
- **Roles.** Friendly 2 of each deck is gold; friendlies 1 and 3 train.
- **Playing time.** 18 friendlies is about 2.1 h, over several evenings.

### 4.6 Tier 3 and the clanmates

**The forms the main account lacks** (14 heroes, 4 evolutions; 4.8.7) come from clanmate training friendlies: eight requests, 27 friendlies (4.8.8).

**The long tail** (current, from the clan data and the collection). What neither the owner's decks nor the eight clanmate requests cover:
- **Owned evolutions:** Ice Spirit (thin, .006), Mega Knight (.005), Zap (thin, .005; J), Executioner (.002), Hunter (.001), Lumberjack (0 in ranked but 8 players in all modes; J), Electro Dragon (0 in ranked, .031 in all modes), Goblin Giant, Bomber (thin). Phoenix (.011, thin) is in J and in clanmate F's deck.
- **Heroes the main account lacks:** Tombstone (.015), Dark Prince (.011), Mega Minion (.007), Magic Archer (.006): each is below the .02 bar, and each has an optional clanmate deck (4.8.8). Wizard hero is in no clanmate's ranked deck.
- **Others:** Clone, Mirror. The Royal Chef tower troop (.036 of clan decks) has no class, but towers are ignore classes and `vision_autolabel label` refuses them.

`vision_gap_labels plan` (N16) builds the decks for these. Re-run the clan ranking after each pull; the Lumberjack and Zap evolutions are the first to watch. Until then they rely on the base label (when the look is close) or on the UNKNOWN safety net.

**Field gold** (the true target):
- **8 single friendlies against clanmates, with the bot only watching** (runbook section 17), instead of 2-3. Pass `--record-images 5` to `vision_play`; its default keeps only one image in 20.
  - **5 for the forms the main account lacks** (4.8.8). The owner's scripted gold cannot hold these forms, and a game used for training is never gold. Each partner is a different clanmate from that form's training source.
  - **3 for owned forms in natural play:** a clanmate's main deck whose gap items the owner's decks all cover (two or more), not itself a capture deck. The scripted gold already holds these forms; these measure another player's placements.
- **Ask for their usual ranked deck, not an event deck.** A friendly uses their own collection. For the 17 clanmates with both kinds of battles, the Oct 24-30 AllCards event overstates cards they may not own: Minion Giant 0 in ranked against .155 in the event, Ice Wizard hero .063 against .238, base Knight .020 against .140.
- **Record field gold before 2026-11-02,** when Season 89 is planned to bring Evo Giant and Evo Mirror, and record one more after it. Neither can be labelled until RoyaleSim ships a table with them.
- **Take the deck from the battle log.** After the friendly, run `opponents.py pull`; Yodle2's spar record then holds the partner's 8 cards with forms. It matched 8 of 8 on 10-29. Use that string at `ingest` and as the gold deck. Confirm it with the partner if Knight, Musketeer, Valkyrie or Wizard (or Giant after 11-02) sits in slot 3. These are never trained on.
- **Never put a name or tag in a session id or a session note.** `session.json` is committed. Use the clanmate's letter of 4.8.8: `20261103_CMA_1`.
- **Clanmate training friendlies** (4.8.8): they play their own deck against Yodle2, which you play lightly by hand so the game reaches overtime. Label them as training, never gold:

  ```powershell
  & $BB tools\vision_autolabel.py ingest reports\vision\harvest\$SID --sid 20261103_CMA_1 --role train --deck "<their 8 cards>" --client 16.635.1
  ```

  Pass `--role train`. `--capture field` is gold only, and with `--role auto` a `_CMA_2` session would become gold.
- How many clanmates carry each hero or whole card in their ranked decks (and in how many most-played decks): Knight hero 9 (6), Bowler hero 8 (4), Berserker hero 5 (1), Ice Wizard hero 3 (2; 17 players in all modes), Valkyrie hero 3 (0), Mini P.E.K.K.A hero 2 (2), Barbarian Barrel hero 2 (2), Balloon hero 2 (1), Tombstone hero 2 (0), Giant hero 1 (1); Vines 4 (4), Suspicious Bush 2 (2), Goblin Demolisher 2 (1), Goblin Curse 2 (1), Goblinstein 1 (1).

### 4.7 Time and disk

| Step | Friendlies | Your time at the PC | Notes |
|---|---|---|---|
| Round 0 | none | none | the existing friendly |
| Round 0b (optional) | 1 | about 7 min | natural gold with your usual deck |
| Tier 1 | 12 | about 1.4 h | your main account plays |
| Tier 2 | 18 | about 2.1 h | your main account plays; J adds 3 (about 20 min) |
| Clanmate training (4.8.8) | 27 | about 3.2 h | the clanmate plays; you start the capture, accept on Yodle2 and play it lightly |
| Field gold | 8 | about 1 h | the clanmate plays; Yodle2 by hand, the bot watching |

A human starts every battle, so every clanmate friendly needs you at the PC.

**Disk.**
- About 1.2-1.7 GB per harvest battle: about 36-51 GB for Tiers 1 and 2, and about 32-46 GB more for the clanmate training friendlies. The disk is 500 GB.
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
- **Which decks:** 4.8.1-4.8.6 describe the clan revision of the capture decks, made before the collection was known ("revised": every hero assumed owned). 4.8.7 fits the decks to the collection (the current 4.4-4.5), and 4.8.8 adds the clanmate friendlies.

#### 4.8.1 Headline coverage

"Ladder plan" is the 4.4-4.5 tables before the clan revision. "Revised" is the clan revision of 2026-10-30, which assumed every hero owned. "Current" is the decks fitted to the collection, now in 4.4-4.5 (4.8.7).

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
| **Current (4.8.7), Tier 1** | 12 | 84.7% | 1.54 | 84.2% | 5.0% | 5-6 |
| **Current, Tier 1+2** | 30 | **69.0%** | 1.01 | 68.5% | 5.0% | 9-10 |
| Floor: every owned gap form covered | - | 68.2% | 0.98 | 68.2% | 5.0% | 9-10 |
| **Current + clanmate friendlies (4.8.8)** | 30 + 27 | **5.3%** | 0.06 | 4.8% | 0.0% | 28 |

- **Revised** rows assumed every hero owned. **Worst case:** the main account owns no hero but the Electro Wizard hero, and no Elite Barbarians evolution. Each deck then uses its named fallback; a slot-2 hero that has only the reserve rule is lost. Other evolutions and whole cards count as owned. The collection turned out close to that worst case (4.8.7).
- **Current** rows are the decks of 4.4-4.5 fitted to the collection; "+ clanmate friendlies" adds the eight requests of 4.8.8.
- **Main decks fully covered:** the clanmates whose most-played deck has no gap left. It is a range because one player has two decks played once each.
- **"Decks with a gap" is harsh:** one leftover card keeps a deck counted. "Gaps per deck" is the fairer view of progress.
- **In-sample.** The revised and current decks were chosen on these same ranked decks. On decks they were not chosen on:

  | Deck set | Today | Revised Tier 1+2 | Ladder plan Tier 1+2 | ... with fillers | Current Tier 1+2 | Current + clanmates |
  |---|---|---|---|---|---|---|
  | Event decks, 33 clan players | 98.8% | 31.6% | 69.2% | 66.1% | 81.7% | 22.0% |
  | Ranked and event decks, 44 clan players | 97.2% | 21.3% | 59.1% | 54.7% | 78.7% | 15.1% |

- **Sensitivity** of the current 69.0%: with troop labels up to 30 enemy boxes counted as thin it is 75.0% (the floor stays 68.2%, because the newly thin Firecracker, Valkyrie and Royal Giant evolutions are owned); with the Goblin Hut and Graveyard look changes it stays 69.0%; with every slot-3 dual-form read swapped it is 67.2%.
- **Sensitivity** of the revised 7.1%:
  - counting troop labels with up to 30 enemy boxes as thin gives 19.3%, mostly the Firecracker evo (.061, 30 boxes), the Valkyrie evo (.050) and the Royal Giant evo (.017);
  - if the Goblin Hut and Graveyard reworks also changed their look, it stays 7.1% (both were in revised decks), and the ladder plan gives 54.2%;
  - counting tower troops gives 14.4%, with 3.6% that cannot be named (Royal Chef);
  - if every slot-3 read of a dual-form card were the other form (4.8.4), it gives 7.4%.
- **Left after the revised Tier 1+2** (clan share): Dart Goblin evo .019, Dark Prince hero .011, Furnace evo .011, Goblins hero .008, Mega Minion hero .007, Ice Spirit evo .007, and six items below .007. Among the event-only players, the largest leftovers are the Furnace, Lumberjack and Zap evolutions.

**Along the current play order** (4.4-4.5; A first: its whole cards cannot be named today; the revised decks' curve is in git history):

| After deck | A | B | C | D | E | F | G | H | I | J (optional) |
|---|---|---|---|---|---|---|---|---|---|---|
| Friendlies | 4 | 8 | 12 | 15 | 18 | 21 | 24 | 27 | 30 | 33 |
| Decks with a gap | 91.0% | 91.0% | 84.7% | 84.7% | 80.8% | 80.7% | 75.9% | 75.1% | 69.0% | 68.5% |
| Gaps per deck | 2.32 | 1.88 | 1.54 | 1.42 | 1.33 | 1.23 | 1.15 | 1.08 | 1.01 | 1.00 |

#### 4.8.2 Priority list

The 25 most-played card+forms the detector lacks or barely knows, by clan share.
- **Share:** in ranked decks, with the 90% interval of a bootstrap over players. **All modes** adds event decks.
- **Ladder plan / Current:** the deck that carries the item in the ladder plan and now: a deck of 4.4-4.5, or "cm X" for clanmate request X of 4.8.8.
- **Owned:** from the API collection of 2026-10-30 (4.8.7). Every whole card is owned.
- **Forms** come from the slot rule (4.8.4). Only slot-3 reads could be the other form: Knight hero holds .039 of its .207 in slot 3.

| # | Card+form | Detector | Share [90% CI] | All modes | Players ranked/all | Ladder plan | Current | Owned |
|---|---|---|---|---|---|---|---|---|
| 1 | knight-hero | base-only | .207 [.10-.32] | .169 | 9/15 | T2 D | cm A | no |
| 2 | bowler-hero | base-only | .150 [.04-.26] | .104 | 8/16 | T2 J | cm B | no |
| 3 | inferno-dragon-ev1 | base-only | .138 [.05-.24] | .099 | 6/10 | T1 B | C | yes |
| 4 | baby-dragon-ev1 | base-only | .122 [.04-.22] | .128 | 6/16 | T1 A | B | yes |
| 5 | zap | spell-thin | .117 [.04-.21] | .096 | 8/11 | T1 C | C, G | yes |
| 6 | lightning | spell-thin | .112 [.04-.22] | .134 | 5/17 | T1 C | B | yes |
| 7 | giant-snowball-ev1 | base-only | .099 [.03-.18] | .093 | 6/13 | T2 D | B | yes |
| 8 | rocket | spell-thin | .093 [.03-.17] | .060 | 8/10 | suggested filler | A, F | yes |
| 9 | goblin-drill-ev1 | base-only | .092 [.02-.18] | .067 | 5/9 | T2 K | cm A | no |
| 10 | skeleton-barrel-ev1 | base-only | .091 [.03-.17] | .060 | 7/9 | T2 F | A | yes |
| 11 | vines | no | .085 [.02-.17] | .056 | 4/7 | T1 A | A | yes |
| 12 | witch-ev1 | base-only | .082 [.01-.18] | .018 | 3/5 | T2 H | C | yes |
| 13 | ice-wizard-hero | base-only | .074 [.01-.15] | .148 | 3/17 | T3 | cm C | no |
| 14 | princess-ev1 | base-only | .072 [.02-.14] | .035 | 5/5 | T2 I | cm D | no |
| 15 | electro-giant-ev1 | base-only | .071 [0-.14] | .087 | 2/11 | T3 | cm B | no |
| 16 | skeleton-army-ev1 | base-only | .069 [.02-.14] | .053 | 3/7 | none | A | yes |
| 17 | electro-wizard-hero | base-only | .068 [0-.14] | .129 | 2/10 | T3 | A, G, I | yes |
| 18 | musketeer-ev1 | base-only | .058 [.01-.13] | .039 | 3/3 | T2 F | E | yes |
| 19 | royal-hogs-ev1 | base-only | .054 [.01-.12] | .056 | 4/8 | T2 E | D | yes |
| 20 | berserker-hero | no | .050 [.01-.11] | .049 | 5/7 | T1 B | cm C | no |
| 21 | mini-pekka-hero | base-only | .048 [0-.11] | .066 | 2/7 | T2 G | cm D | no |
| 22 | royal-ghost-ev1 | base-only | .048 [0-.11] | .041 | 2/6 | T1 C | G | yes |
| 23 | barbarian-barrel-hero | base-only | .046 [0-.12] | .039 | 2/5 | T2 J | cm F | no |
| 24 | goblin-demolisher | no | .046 [0-.11] | .054 | 2/7 | T1 A | A | yes |
| 25 | suspicious-bush | no | .046 [0-.10] | .020 | 2/3 | T1 A | A | yes |

**Next 24, down to .01** (share, current deck; "not owned" items come from clanmates):
- goblin-barrel-ev1 .044 (F); elite-barbarians-ev1 .042 (cm E, not owned); cannon-ev1 .039 (E); goblin-cage-ev1 .038 (D);
- giant-snowball .037 (B, via its evo); battle-ram-ev1 .036 (thin, G); wizard-ev1 .036 (H); giant-hero .036 (cm E, not owned); pekka-ev1 .033 (I);
- goblin-curse .032 (A); goblin-barrel .031 (F, via its evo); balloon-hero .030 (cm G, not owned); minion-horde-ev1 .029 (I); goblinstein .025 (B);
- valkyrie-hero .025 (cm H, not owned); royal-delivery .025 (D); dart-goblin-ev1 .019 (F); ronin .018 (B); berserker .017 (B); tombstone-hero .015 (optional cm I, not owned);
- furnace .015 (look-changed, H); dark-prince-hero .011 (optional cm D, not owned); furnace-ev1 .011 (H); phoenix .011 (thin; cm F and the optional J).

**Below .01:** rune-giant (B), Minion Giant (B; .004 ranked, but .051 and 10 players in all modes), Goblins hero (H), Boss Bandit (C), freeze (I), Musketeer hero (F; .021 in all modes) and the Tier 3 items of 4.6. Three Musketeers, Void, Spirit Empress and Goblin Machine are in no ranked clan deck (C keeps them). Evo Giant and Evo Mirror (planned for 2026-11-02) are not in the 16.402.22 table, so they cannot be labelled until RoyaleSim ships them.

#### 4.8.3 Deck slot rules in the clan data

From the 1,176 collection-mode sides:
- 8 different cards.
- Slot 1 takes an evolution or a base card.
- Slot 2 takes a hero, a champion or a base card.
- Slot 3 takes an evolution, a hero, a champion or a base card (1,020, 116, 15 and 25 sides).
- Slots 4-8 take base cards only. Champions sat only in slots 2 and 3; two champions together, once.
- So: at most 2 evolutions, at most 2 heroes and champions together, at most 3 special cards, as 1.10 says.
- The owner confirmed the rule on 2026-10-30: two evolutions and one hero, or one evolution and two heroes; never three of one kind; fewer is allowed. That statement did not mention champions; the data shows a champion taking a hero's place.
- Every capture deck in 4.4-4.5 (the optional J included) follows these rules with 8 different cards and owned forms only, so none needs a fallback. Every clanmate request deck of 4.8.8 is a deck the clanmate played, so it follows them too.

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
1. After a friendly of deck E (the Musketeer evo in slot 3), read the probe's "latest" card table.
2. The main account owns both Musketeer forms: 1 means the log shows the form played; 3 means it shows ownership.

**The probe on the PC (2026-10-30)** found the form played: an owned evolution in slot 4 shows as base, cards come in deck order, and no side has 7 cards. Friendlies show every card at level 11. Whether a slot-2 hero always reports as a hero is checked with the next friendly (A's friendly 1, or any deck with a hero in slot 2).

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
- The revised tables use the same 36 friendlies, and they take the clan's leftover from 48.4% to 7.1%, if every hero is owned.

**Then fitted to the collection (2026-10-30, 4.8.7):**
- Tier 1: the Electro Wizard hero replaces Berserker hero in A's slot 2, the Skeleton Army evo takes A's slot 3, and the Inferno Dragon evo moves from A to C, in place of the Goblin Drill evo.
- Tier 2: 6 decks instead of 8, with new letters in play order. Every unowned hero and evolution left the owner's decks. The Ice Golem, Musketeer and Goblins heroes and the Furnace and Dart Goblin evolutions came in, and the Electro Wizard hero now plays in three decks.
- The unowned forms moved to clanmate friendlies (4.8.8).

#### 4.8.7 The main account's collection (official API, 2026-10-30)

The API lists every card with `evolutionLevel` and `maxEvolutionLevel`: bit 1 is the evolution, bit 2 the hero, and `maxEvolutionLevel` says which forms exist.

| Kind | Exist | Owned | Not owned |
|---|---|---|---|
| Cards | 123 | all 123, champions included | none |
| Evolutions | 43 | 39 | Elite Barbarians, Goblin Drill, Princess, Electro Giant |
| Heroes | 18 | 4: Electro Wizard, Goblins, Ice Golem, Musketeer | Knight, Bowler, Ice Wizard, Berserker, Mini P.E.K.K.A, Balloon, Giant, Barbarian Barrel, Tombstone, Valkyrie, Wizard, Dark Prince, Mega Minion, Magic Archer |

- **Musketeer** is the only card with both forms owned. That makes deck E's slot-3 Musketeer evo the direct form check of 4.8.4.
- **The 18 forms it lacks are all detector gaps:** 17 are `base-only`, and Berserker hero has no class. 17 of them are in clan ranked decks (Wizard hero is in none), together in 68.2% of them, 0.98 gaps per deck. No plan with the main account alone can go below that floor.
- **The fitted decks** (4.4-4.5, 30 friendlies) leave 69.0%, 1.01 gaps per deck (4.8.1). The earlier revised decks with their unowned forms simply dropped would leave 69.9% with 36 friendlies.
- **What is left after them**, with each remaining deck's weight shared equally by its remaining items: 52.3 points from unowned heroes, 14.8 from unowned evolutions, 1.9 from owned leftovers. The largest items are Knight hero 14.0, Bowler hero 11.4, Ice Wizard hero 5.5, the Goblin Drill evo 4.6, the Princess evo 4.5, Mini P.E.K.K.A hero 4.5, Barbarian Barrel hero 4.0 and the Electro Giant evo 3.6. 36.8 of the 69.0 points are decks missing a single item.
- **Owned leftovers** (not in a Tier 1 or Tier 2 deck): Phoenix .011 (thin; in J and in clanmate F's deck), the Ice Spirit .006, Mega Knight .005, Zap .005 (J), Executioner .002 and Hunter .001 evolutions.
- **Held out:** on the clan's event decks the fitted decks leave 81.7% (floor 76.0%). The decks were chosen on the ranked decks they are scored on.

#### 4.8.8 Clanmate friendlies for forms the main account lacks

Clanmates play their own decks against Yodle2, recorded as training. Letters only: the owner keeps the letter-to-name map off the repo, and the letters are not capture-deck letters.

| Request | Their deck | Forms it adds (clan share) | Training friendlies | Decks with a gap after |
|---|---|---|---|---|
| clanmate A | main deck | Knight hero (.207), Goblin Drill evo (.092) | 4 | 52.5% |
| clanmate B | main deck | Bowler hero (.150), Electro Giant evo (.071) | 4 | 37.6% |
| clanmate C | main deck | Ice Wizard hero (.074), Berserker hero (.050) | 3 | 30.2% |
| clanmate D | another ranked deck (4 battles) | Princess evo (.072), Mini P.E.K.K.A hero (.048) | 4 | 18.7% |
| clanmate E | main deck | Elite Barbarians evo (.042), Giant hero (.036) | 3 | 15.2% |
| clanmate F | main deck | Barbarian Barrel hero (.046), plus the owned Phoenix (.011) | 3 | 10.6% |
| clanmate G | main deck | Balloon hero (.030) | 3 | 7.6% |
| clanmate H | Showdown deck (5 battles) | Valkyrie hero (.025) | 3 | **5.3%** |
| *optional:* clanmate D | another deck | Dark Prince hero (.011) | 2 | 4.8% |
| *optional:* clanmate H | another deck | Mega Minion hero (.007) | 2 | 4.1% |
| *optional:* clanmate F | another deck | Magic Archer hero (.006) | 2 | 3.5% |
| *optional:* clanmate I | a ranked deck | Tombstone hero (.015) | 2 | 2.0% |

- **Coverage** (clan ranked decks with a gap):

  | Stage | Friendlies | Decks with a gap | Gaps per deck | Main decks fully covered |
  |---|---|---|---|---|
  | The main account's decks only (4.4-4.5) | 30 | 69.0% | 1.01 | 9-10 of 28 |
  | ... plus the eight requests | 30 + 27 | **5.3%** | 0.06 | 28 of 28 |
  | ... plus the optional J | 33 + 27 | 4.8% | 0.05 | 28 of 28 |
  | ... plus the four optional decks | 30 + 35 | 2.0% | 0.02 | 28 of 28 |

  - Nothing is left that cannot be named: clanmate C covers Berserker hero.
  - If every owned form were covered (the floor), the eight requests would leave 4.0%.
  - Held out: on the clan's event decks, 81.7% becomes 22.0%, and 10.9% with J. Most of the difference is the Lumberjack and Zap evolutions, which only J holds.
- **How they were chosen.** Greedy over (clanmate, ranked deck) pairs. The gain is the clan share of the unowned forms the deck adds. Ties go to a clanmate active in the last 7 days, then one already asked, then their main deck, then battles with the deck. The main set stops below a gain of .02.
  - Two clanmates tie for F. The one whose deck also holds Phoenix is asked first. The other, clanmate O, is the fallback; with O the eight give 6.4%.
  - Every request clanmate played in the last 7 days. Two clan accounts (B and J) play one identical deck and may be one player.
  - Single sources: Giant hero (E) and Mega Minion hero (H). The Electro Giant evo's two accounts (B and J) may be one player. Every other form has a second clanmate, so a refusal has a fallback.
- **Friendlies per request.** 4 where a single-unit evolution is new (Goblin Drill, Electro Giant, Princess), 3 for a hero, 2 for the optional decks. A form needs 8 labelled tracks to train as itself (2.3), and a natural friendly shows each card roughly 3-6 times.
- **Field gold** (one friendly each, bot watching, `--capture field`; 4.6):

  | Clanmate | Gold for |
  |---|---|
  | G | Knight hero, Goblin Drill evo |
  | J | Bowler hero, Electro Giant evo (no independent gold if J and B are one player) |
  | K | Ice Wizard hero |
  | L | Princess evo |
  | D | Berserker hero |
  | M | owned forms: Goblin Curse, the Inferno Dragon and Skeleton Army evos, Zap |
  | I | owned forms: the Skeleton Army and Skeleton Barrel evos, Suspicious Bush, Vines |
  | N | owned forms: Goblinstein, the Minion Horde evo |

  Each gold partner for an unowned form is a different clanmate from that form's training source. One natural friendly may leave a form below the gate's 4 gold plays; it then reports as "insufficient" (6.4).
- **Running them:**
  - one SID per friendly with the clanmate's letter, for example `20261103_CMA_1`; never a name;
  - `--role train` on every training SID;
  - take `--deck` from `opponents.py pull` after the game;
  - ask the clanmate to play every card several times, cycle the evolution, use the hero ability and go to overtime; play Yodle2 lightly by hand;
  - where Knight sits in slot 3 (the optional F deck), confirm with the clanmate that it is the hero (4.6).
- **Time.** About 3.2 h at the PC for the 27 training friendlies and about 1 h for the 8 field gold friendlies; labelling about 2.7-4 h and 2-3.6 h (5).
- Slot-2 hero reporting is checked with the next friendly (4.8.4). Re-run the ranking after each pull.

## 5. Labelling steps and time

The commands (N08-N13, built; `tools\vision_autolabel.py --help` and each subcommand's `--help` list the options):

```powershell
& $BB tools\vision_autolabel.py ingest reports\vision\harvest\$SID --deck "skeleton-barrel-ev1,electro-wizard-hero,skeleton-army-ev1,vines,...,rocket" --role auto --client 16.635.1
& $BB tools\vision_autolabel.py scan $SID            # unattended, 10-20 min on the GPU; MuMu closed
& $BB tools\vision_autolabel.py sheet $SID           # writes the numbered contact sheet PNG(s) and prints the legend
& $BB tools\vision_autolabel.py label $SID --sheet 1 --map "0=3,1=3,2=x,5=s"
& $BB tools\vision_autolabel.py emit $SID            # training: images and boxes in <store>\$SID\emit; gold: labels only
& $BB tools\vision_autolabel.py review $SID          # page A (tracks) and page B (frames)
& $BB tools\vision_autolabel.py mark $SID --review 1 --bad-tracks t17,t40.1 --bad-frames 3,9
& $BB tools\vision_autolabel.py emit $SID --force    # after a mark or a relabel (label ... --force)
& $BB tools\vision_autolabel.py plays $SID           # gold battles only: then --list K [--bad-plays ...] [--add-play ...] --confirm
```

Before the first fine-tune (N13, N14):

```powershell
& $BB tools\vision_autolabel.py labels --base tmp\yolo\detector_labels.json --pin-base   # once; commit data\vision\detector_extra_labels.json
& $BB tools\vision_autolabel.py mix --name ft1       # KataCR + every fresh training emit (dev as val); gold never
# then the train command mix prints (scripts\train_detector.py --data tmp\mix\ft1\cr_mix.yaml ... --no-pin), overnight
& $BB tools\vision_gap_eval.py gate --candidate tmp\yolo\runs\ft1\weights\best.pt     # PASS / FAIL / INSUFFICIENT; never pins
```

- **Raws.** Keep a training session's raws and scan until review and mark are done: `emit --force` is refused while it would lose emitted images whose raws are gone (`--drop-missing` emits without them). Keep gold raws for good.
- **Leftovers.** Emit masks what nobody labelled: unvetted live units and spells, and detector proposals of tracks no label holds (one of Yodle2's units the live model missed, a track too short for the sheet). On gold these are ignore regions.

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

**Gold battles.** Review every track, then confirm the derived play list. Delete wrong plays with `--bad-plays`, and add missed ones with `--add-play "1:52=Vines"`, naming the list the ids come from (`--list K`).

**Owner time per battle (estimates):**

| Battle | Ingest | Sheet and label | Review | Plays | Total |
|---|---|---|---|---|---|
| Training | 1 min | 5-7 min | 2 min | none | about 9 min |
| Gold | 1 min | 5-7 min | about 15 min | about 5 min | about 27 min |
| Round 0 | 1 min | about 10 min (30-45 clusters) | about 15 min | about 5 min | about 30-45 min |

**Totals:**
- **Tier 1:** about 1.4 h for training battles plus 1.4 h for gold.
- **Tier 2:** shorter per battle, because the Tier 1 model auto-accepts what it knows. About 1.2 h for 12 training battles plus 2.2 h for 6 gold.
- **Clanmate training (4.8.8):** 27 battles at about 6-9 min, about 2.7-4 h.
- **Field gold:** 8 battles at about 15-27 min each, about 2-3.6 h.
- **Overall:** about 6.2 h for the main account's 30 friendlies, and 11-14 h with the clanmate and field gold friendlies, over 3-4 weeks. Your time at the PC is about 3.5 h for your own friendlies and about 4 h for the clanmates' (4.7).

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

`tools/vision_gap_eval.py` (N14) scores the baseline and the candidate on the same gold. It writes `report.md` and `gate.json` under `reports\vision\gate\`.
- The first `baseline` run records tonight's model as M0 (`reports\vision\gate\m0.json`). The play-ID gain is always measured over M0; the regression checks compare with the current pin (`--baseline`, default the pin, its SHA-256 checked).
- Side accuracy and our labels' recall count only owner-vetted boxes (the owner's labels and our `o` tracks). Our other boxes in gold are tonight's own pseudo-labels, so their agreement is reported as info.
- The leak check trusts a candidate's `mix.json` only if it is the one the run trained on (`train_detector` records its SHA-1).

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

**Order.** Built: N01, N02 and N04-N14 (2026-10-30; the first PC test run of N11-N14 is pending). Not started: N03, N15-N18.
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
6. **Coverage depends on what the main account owns.** It lacks 14 heroes and 4 evolutions (4.8.7), in 68.2% of clan decks. Those rest on clanmate friendlies (4.8.8): one or two players' placements per form, and only if they agree to play. Evolutions appear only on evolved cycles, and hero abilities change the look mid-track. The long tail stays thin. The safety net and `train_as` keep it from vanishing or naming the wrong card.
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
2. ~~Which capture cards does the main account own?~~ **Answered (2026-10-30, official API; 4.8.7):** all 123 cards, 39 of 43 evolutions (not Elite Barbarians, Goblin Drill, Princess or Electro Giant) and 4 of 18 heroes (Electro Wizard, Goblins, Ice Golem, Musketeer). The capture decks of 4.4-4.5 are fitted to it, and the forms it lacks come from clanmates (4.8.8). Still check each special card in the deck builder before a deck's first friendly.
3. Does your client (16.635.1) have cards, evolutions or heroes that are missing from the 16.402.22 table?
4. Do clanmates' friendlies show their own tower troop (Royal Chef, Duchess, Cannoneer), or always Tower Princess? Royal Chef has no KataCR class. 15.3% of clan decks use another tower or have none in the log (4.8.5), so field gold will answer it. Do they use other arenas?
5. Which card id does the engine log for a Spirit Empress play: MergeMaiden, or a variant? N01's alias depends on it.
6. Should the deck prior (N18) exist at all? It uses ladder statistics only for naming what is on screen, never as policy input. It is planned off by default. A clan prior would put Golden Knight near 0 in friendlies: it is in no ranked clan deck, yet the pinned detector reported 10 Golden Knight plays in Round 0, most likely Boss Bandit.
7. Will the clanmates of 4.8.8 play? Eight training requests (27 friendlies) and 8 single field gold friendlies with the bot only watching. The owner's private list names other clanmates with the same forms, where they exist.
8. Does the enemy side ever show the elixir-cost pop or the card-name text? KataCR's data says these are our side only (elixir 1,493 / 0, text 2,050 / 0). This can be checked on the recorded friendly.
9. What does `opponents.py probe` say about battle-log forms? **Partly answered (2026-10-30):** the probe on the PC found the form played, in deck order (4.8.4). Still open: whether a slot-2 hero always reports as a hero (A's friendly 1), and slot 3, read by hand after the first friendly of deck E (the Musketeer evo; both forms owned).
