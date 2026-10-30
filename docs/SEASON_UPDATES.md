# Season updates: keeping the bot current after each game update

Status: plan, 2026-10-30. Nothing here is built yet. Work items are U01-U22 (section 12). This is the routine named in `docs/SPARRING_PLAN.md:196`.

The owner's concern: "one of the biggest issues is keeping updated with all the new cards, evolutions, heroes and balance changes."

The goal is a routine with three properties:
- every part of the bot is brought up to date after each game update;
- the owner does as little as possible;
- nothing goes stale without a visible warning.

The hard rules in `CLAUDE.md` apply to every item:
- The routine reads only PyPI, the official Clash Royale API, the repo's own files and `adb dumpsys`.
- The `dumpsys` read is read-only. It runs only after `scripts/emulator.verify()` passes.
- It never starts or restarts MuMu, never taps, and never reads game memory.
- No tool, flag or script queues any battle mode.

Facts are cited as `file:line` on the `sparring` branch at 00343a6. The RoyaleSim facts come from reading the 0.1.42 wheel and the sdists of all 32 releases that have one. Nothing in the repo was run.

## Short version

- **The weak link is upstream.** A new card table needs a Rust change and a new RoyaleSim release, so we cannot add one as data. RoyaleSim followed the last two balance updates within 31 h and about 3 days.
- **What stands today:**
  - The PC's client (16.635.1) is newer than the newest table (16.402.22). The owner accepted that gap on 2026-10-29.
  - Season 89 starts on Monday 2026-11-02 with Evo Giant and Evo Mirror, which the table lacks.
- **One scheduled task runs every day**, with no adb and no emulator. It:
  - pulls the clanmates' and Yodle2's battle logs from the official API;
  - checks PyPI for a new RoyaleSim;
  - checks the card roster for new cards and forms;
  - checks the season calendar;
  - writes one status line, which selfcheck and the live runner print.
- **After a game update, the owner runs one command:** `scripts\game_update.ps1 -Update`. It takes about 1 h of CPU and does the rest:
  1. installs the new RoyaleSim in a side venv;
  2. diffs the card tables;
  3. classifies the update;
  4. runs the tests and a quick fidelity grid;
  5. on one "Y", moves the pin, carries the checkpoints across and refreshes the opponent pool.
- **Checkpoints survive balance updates.** A balance-only table change will no longer refuse every checkpoint (U01). Card ids only grow, so a new card is carried by copying embedding rows by name (U14).
- **Retraining is a separate overnight command**, `scripts\model_update.ps1 -Apply`. It promotes a new model only through the existing gate plus a clan-deck check. It waits until a v2 champion exists.
- **Opponent decks** come from what the clanmates actually play, read through the official API (U03, U16). Today RL and the gate see only IL_Replay decks.
- **Vision** is owned by `docs/NEW_CARDS_PLAN.md`. Section 7 of that plan is its part of this routine.

## 1. What depends on the game's card data

| Link | What it is | How it goes stale | Guard today |
|---|---|---|---|
| Simulator | `royalesim==0.1.42` (`requirements/engine.txt`, `requirements/gpu.txt`, `pyproject.toml`); `data/engine.json` `card_table` `client160402017_20261019` (content 16.402.22); factory `tools/rs_engine.py` | A game update ships before RoyaleSim does. A same-named table can change bytes. | `rs_engine.check_card_table` (`tools/rs_engine.py:552`) compares the version stem only. Selfcheck pins the package version. |
| Client build | versionCode 160635001 (16.635.1), read by `scripts/selfcheck.py` | Builds change every 1-2 weeks. Server-side content can change with no new build. | `judge_game` (`scripts/selfcheck.py:665-677`) PASSes 160635001 and WARNs on any other build. Nothing compares the client with the table. |
| Fidelity | `tools/rs_fidelity_grid.py` runs `tools/royalesim_redrive.py` per cell on IL_Replay val matches | 62.6% / 49.0% were measured on royalesim 0.1.22 with the 15.535 table (`docs/PROGRESS_2026-10-09.md:72-83`). The grid has never run on the PC. | none |
| Vision classes | `data/vision/classes.json` (299 classes, digest `dc897910…`), from `tools/vision_classes.py --engine` | A new unit or form needs a regenerated file. Ids are kept and new classes appended. | `digest` is written into rows (`tools/rs_rows.py:2312`) and label sets. No loader compares it. |
| Spawn relations | `data/vision/spawn_children.json`, from `tools/vision_spawn_children.py` | **Already stale.** Its `source.version` is `cards-15535.1`, not the pinned table. | none; `docs/NEW_CARDS_PLAN.md` item N01 fixes it |
| Card map | `data/cards/replay_cards.json` (119 keys, SHA-pinned at `tools/rs_cards.py:63`) plus `replay_cards_extra.json` (3 keys) = 122 keys | A new card has no key. Minion Giant already has none. | A match with an unmapped key is skipped whole and counted (`tools/rs_rows.py:1376-1402`, `:1849-1851`). |
| Form suffixes | `base_key` knows `-ev1` and `-hero` only (`tools/deck_stats.py:80-85`, copied verbatim from `tools/royalesim_redrive.py`); `tools/rs_presses.py:61` | A new suffix makes the card unmapped and skips the match. | none |
| Self-healing table | `data/rs_rows/spawn_primary.json` (`tools/rs_spawn_table.py`) | Regenerates itself when the engine fingerprint changes (`ensure`, `:353`). | good as is |
| Rows and checkpoints | `vocab.json` and checkpoint meta: card names, unit names, card table, engine fingerprint | Any arm change refuses every checkpoint, even with identical ids (`tools/rs_checkpoint.py:239`). | `rs_checkpoint.load_policy`, used live at `scripts/vision_play.py:1377` |
| RL runs | `tools/rl/train_loop.py` freezes the engine id | Any royalesim bump changes the id, so resume is refused (`COMPARED` / `ENGINE_KEYS` at `:199-201`, refusal at `:1243-1250`). | A new `--name` is needed. That is fine. |
| Opponent decks | RL field `<data>/deck_stats.json` (IL_Replay train split, `tools/rl/rollout.py:1390`); gate pairs from IL_Replay val | They never contain a card newer than IL_Replay. | `deck_stats.load_stats` checks nothing (`tools/deck_stats.py:342`). |
| Hand-kept lists | `tools/rs_decide.py:168`, `:745-750`; `tools/rs_orient.py:62`; `tools/rs_presses.py:64`; `scripts/vision/events.py:113` | A new tunneller, rolling spell or champion is missed. | none |
| HUD templates | own-deck hand art in `data/vision/templates/` (gitignored) | Card art or HUD changes in a client update. | none; re-checked by a dry-run friendly |
| Live search | `docs/LIVE_SEARCH_PLAN.md`, if it lands | The simulator would then run during live play, so a stale table affects live decisions directly. | It must use the same `data/engine.json` and pin. |

## 2. Where we stand on 2026-10-30

- **RoyaleSim.** PyPI's latest is 0.1.42, uploaded 2026-10-23 04:16Z. I checked with curl today; there are 33 releases. Nothing exists for client line 16.635.
- **Client.** 160635001 is versionName 16.635.1 (`tests/pipeline/test_selfcheck.py:650`). The encoding is `MM mmmm bbb`.
  - The table's client build is 160402020, with content 16.402.22. So the PC runs a newer client line than any table.
  - The owner judged that nothing major changed since the 2026-10-19 balance update (`docs/SPARRING_PLAN.md:23-27`).
- **Not yet measured.** Fidelity of 0.1.42 and of the 16.402.22 table. The grid code is done (batch 4) but has not run.
- **Stale files.**
  - `data/vision/spawn_children.json` was built from `cards-15535.1`. N01 fixes it.
  - The `classes.json` source note quotes sha256 `71c118f4…` for the 16.402.22 bytes. The sdist file hashes to `813bc904…`.
    - The note says its bytes were extracted from the `.so`, so the extraction boundary may explain the difference. That is unproven.
    - U07 records the engine's own `card_table_fnv1a64()` instead.
- **Next event.** Season 89 starts on 2026-11-02 with Evo Giant and Evo Mirror. A third-party source reports no balance changes at that season start; November's changes were moved to 2026-10-19.
  - The 20261019 table has no `Giant_EV1` or `Mirror_EV1`. Its giant evolutions are ElectroGiant, GoblinGiant and RoyalGiant only.
  - In `classes.json`, Mirror has forms `[0]` and Giant `[0, 2]`.
  - So both evolutions will be simulated as base forms until RoyaleSim ships them.
- **Minion Giant** (released 2026-09-07) is in the engine catalogue (`MinionGiant`, id 132) and in the visible card pool (`docs/NEW_CARDS_PLAN.md` 1.1). No key in the card map reaches it, so no IL_Replay row trains its embedding.

## 3. How updates arrive

### 3.1 The game

From web research (search snippets; most official pages were blocked from the sandbox):
- **Seasons** start on the first Monday of each month. That rule reproduces all eleven 2026 season starts. The next ones are 2026-11-02 and 2026-12-07.
- **Balance changes** come at season starts, plus mid-season patches. In 2026 these were 08-26, 09-16 and 10-19.
- **Heroes:** about 1-2 a month since 2025-12-01.
- **Evolutions:** about every other month in 2026.
- **New base cards:** rare. In 2026 there were Ronin (07-13) and Minion Giant (09-07).
- **Client builds** (APKMirror upload dates): 160402002 on 09-01, 160402012 on 09-15, 160402017 on 10-02, 160402020 on 10-07, 160635001 on 10-27.
- **Friendlies** play every card and tower at level 11 (`docs/SPARRING_PLAN.md:19`).

### 3.2 RoyaleSim

- **Releases.** 0.1.10 (2026-10-04) to 0.1.42 (2026-10-23). There is no changelog on PyPI. The GitHub repo and docs site gave 403 from the sandbox.
- **Each sdist is a full changelog.** It holds the Rust source, `data/calibration.json` (the ledger: every mechanic key with value, status and provenance) and every `data/derived/cards-*.json` table. Diffing two sdists needs no code execution.
- **Arms are named in the ledger.** Key `cards.CARD_TABLE`: candidates `client15535`, `client160402017_20261006`, `client160402017` (checkout-only, a wheel cannot load it) and `client160402017_20261019`. The engine's default is still `client15535`.
- **Lag.**
  - The 2026-10-06 update reached 0.1.23 on 10-09 08:54Z, about 2.5-3 days later.
  - The 2026-10-19 16:48Z update reached 0.1.40 on 10-20 23:28Z, about 31 h later.
- **The last table change** (10-06 to 10-19) kept the roster: 145 cards, 43 evolutions, 18 hero forms, 320 unit records. It changed 55 values.
  - One hits our deck: Skeletons `summon_radius_milli` 400 to 500 (also Skeletons_EV1).
  - Others: Dart Goblin damage 59 to 56; Electro Dragon load time 1400 to 1600 ms; Void (`DarkMagic`) dps 1155 to 1200.
- **`client_arms()`** accepts only `15.535.x` and `16.402`. The engine does not know 16.635.

### 3.3 Three facts that shape the design

1. **A table's name is not its identity.** `cards-160402017-20261006.1` kept its name while its bytes changed 6 times (0.1.23 to 0.1.36). Only `Battle.card_table_fnv1a64()` (0.1.40+) and the royalesim version identify content. Expected values:
   - 20261019: `15bb4e60e69d05f3`;
   - 20261006: `87cca9def6207621`;
   - 15535: `803ec02c7eac9fd3`.
2. **Card ids only grow.** `Battle(card_names=None)` (`tools/rs_engine.py:38`) lists the pinned `CATALOGUE_ORDER` first and appends every other loadable card after it.
   - The sdist says a new card "gets the next id and moves no other card's", and the list "is never edited, only grown past" (0.1.42 `crates/royalesim/src/py.rs:241-246`).
   - All 145 cards of the 16.402.22 table are in that list.
   - So a new card does not renumber the others. Only a pinned card that stops loading would.
3. **Unit types are sorted by name.** A new unit record, such as a new `_EV1` or hero unit, renumbers the unit ids after it.
   - The production models use `--live-compat` rows (`scripts/third_model.ps1:218-221`), whose unit tokens are card ids (`tools/rs_rows.py:1726`).
   - So a new form needs no new embedding row for them. Only `--units types` checkpoints are hit.

## 4. The routine

### 4.1 Every day, automatically

One Windows scheduled task runs `scripts\game_update.ps1 -Daily` (U06). It runs with `.venvs\royalegym`, makes HTTPS calls only and never calls adb. Steps:

1. **Opponent pull** (U03): the clan roster, every member's battle log, Yodle2's and the owner's battle logs, and `/v1/cards`. About 55 calls, under a minute. The battle log holds only about 25 battles, so a missed day can lose games for good.
2. **Version check** (U04), `tools/game_update.py check --offline`:
   - **PyPI:** any royalesim newer than the pin. For each one, it downloads the sdist and checks it against PyPI's sha256. It then reads the ledger arms and every table's version, client build, content version and FNV.
     - It flags new arms, and same-named tables whose bytes changed.
     - It does the same for royaleimitate's `replay_cards.json` keys.
   - **Roster radar:** it diffs the day's `/v1/cards` snapshot against the previous one and against the engine table.
     - It flags new card ids, renames, and new `maxEvolutionLevel` bits (1 evolution, 2 hero) whose form the table lacks.
     - Example after 2026-11-02: "Giant gained an evolution; the simulator lacks it".
   - **Calendar:** season starts and the `content_events` listed in `data/game_versions.json` since the last check.
   - **Device:** the client build last seen by selfcheck or the live runner (`last_seen_client`).
   - **IL_Replay:** the dataset's latest revision, when the Hugging Face API answers. A re-published corpus would be the only source of imitation rows for new cards.
3. **Mondays only:** the top Path of Legends players' battle logs (`opponents.py meta`, about 1,000 calls, about 8 minutes at 2 requests per second).
4. **Status:** it writes `<runs>/updates/status.json` and `status.txt`.

**Exit codes of `check`:**
- 0: nothing new;
- 10: UPDATE AVAILABLE, run `-Update`;
- 11: a new client build with no newer table;
- 1: error;
- 2: bad arguments.

**The currency line.** Selfcheck's `game` and `update` lines and the live runner's header print one summary from `status.json`, for example:

```
currency  client 160635001 (16.635.1) vs table 16.402.22: client ahead, accepted 2026-10-29;
          royalesim 0.1.42 = pin, newest on PyPI; checked 6 h ago; pool pulled 6 h ago;
          Season 89 started 2026-11-02: run scripts\game_update.ps1 -Update when RoyaleSim ships it
```

**A new client build** is recorded by itself, with basis `auto`, when the same day's check finds:
- no new table;
- no roster or form change;
- no calendar event.

Otherwise it stays a WARN until the owner runs `-AcceptClient -Note "why"`. This avoids an owner step every two weeks; there were 5 builds in 8 weeks.

### 4.2 After a game update: one command

**When:** the currency line says UPDATE AVAILABLE, or a season start or balance post has passed.

**How:** close the browser, then run `scripts\game_update.ps1 -Update`. MuMu can stay closed. The command is `tools/game_update.py update` (U09, U12), and it resumes with `--reuse`.

1. **Stage** (about 1 h of CPU, unattended).
   - **Side venv.** It creates `.venvs\rs_next`: the engine venv with only royalesim changed, installed with `--require-hashes` from PyPI's digest. The live venvs are not touched.
   - **Probe.** It probes both engines (U09):
     - version, `provenance()`, ENTITY_FIELDS, ABILITY_BUTTONS, SPELL_FIELDS and STATUS_BITS;
     - for each ledger arm: `card_table()`, `card_table_fnv1a64()`, catalogue names in id order, unit types and `rs_engine.fingerprint`.
   - **Target arm.** It picks the newest arm the new wheel builds. The table bytes from the sdist must hash to the engine's own FNV; if they do not, the update is BLOCKED.
   - **Diff.** It runs `tools/card_table_diff.py` (U08) on the old and new tables and ledgers, and on the same arm in both versions.
   - **Checkpoints.** For every checkpoint (the live one, the KL ref, the stand-in, `runs/tokens/*/best.pt`, `runs/rl/*/champion.pt`) it reports: loads, loads with a note, needs `grow`, or refused.
   - **Derived data**, regenerated into the stage folder:
     - `classes.json` (`tools/vision_classes.py --engine`, ids kept);
     - `spawn_children.json` from the target table (the generator as fixed by N01);
     - `spawn_primary.json`;
     - `rs_cards --catalogue`;
     - a check that every Hog 2.6, card-map, deck_stats and pool name is in the new catalogue.
   - **Tests** in a throwaway git worktree with the runbook's sparse pattern: `pytest tests/pipeline -m "not device and not gpu"` on the promoted tree with the new engine. If that fails, the unpromoted tree runs too, to tell an engine break from a promotion break.
   - **Quick fidelity grid**, 3 cells (U10): the old engine, the new engine on the old table, the new engine on the new table. All three run on the same 2,000 val matches.
2. **Verdict and report** (section 4.3).
3. **Promote**, on one "Y". This only happens when the verdict is not BLOCKED and the class is not BREAKING. The "Y" is the owner's consent to the commit. Steps in section 4.5.
4. **Opponent pool.** It rebuilds the pool under the new engine, then writes a new RL field export and a new pair file (U16).
5. **Model plan.** It prints the update class, the cold cards, the share of the clan meta the engine cannot simulate, and the suggested retrain budget.

**Later, optional, overnight:** `scripts\model_update.ps1 -Apply -Hours 8` (U18). Close MuMu and the browser first. Section 6.3 covers it.

**When the client build changed:** play one dry-run friendly (runbook section 17) to re-check the HUD templates and vision. New cards follow `docs/NEW_CARDS_PLAN.md` section 7.

**If anything is wrong:** `scripts\game_update.ps1 -Rollback`.

### 4.3 The report

The stage writes `<runs>/updates/stage-<stamp>/report.md`. Promote copies it to `reports/updates/<stamp>.md`, which is committed.

**The top block:**
- one verdict line;
- "safe to keep playing now?";
- client line against table line;
- what to run next.

**Then the impact matrix:**
- **Simulator:** pin, arm and FNV, old and new; ledger keys changed, by section.
- **Roster:** cards, evolutions, hero forms, towers and units added or removed. New forms are found through `form_of`.
- **Values:** changed values per card, with unit and projectile echoes folded onto the card. Each is tagged OUR DECK (Hog 2.6, `tools/rs_rows.py:291`, and its forms), POOL (at least 3% of opponent-pool weight) or OTHER.
- **Checkpoints:** the table from the stage.
- **Derived data:** classes added and the new digest; spawn relations changed; card-map entries proposed with `"verified": false`.
- **Grid:** the 3 cells, with McNemar against the base cell. Rows per tower troop (`tools/rs_fidelity_grid.py:87`) are always printed.
- **Hand-offs:**
  - new classes and forms the detector lacks, for `docs/NEW_CARDS_PLAN.md`;
  - pool decks blocked or downgraded;
  - hand-kept lists to review (section 11);
  - for an engine without a sdist: modelled or not, i.e. whether a named upstream test exists for each new card or form (`crates/royalesim/tests/*.rs`; 0.1.42 has 351 of them).

### 4.4 Update classes and verdicts

One classifier in the stage decides the class. The model plan reads the same class; there is no second taxonomy.

| Class | Example | Checkpoints (after U01, U14) | Action |
|---|---|---|---|
| SAME | nothing new | load | nothing |
| ENGINE_ONLY | 0.1.41, 0.1.42: one ledger key each | load, with a drift note | promote is optional; RL runs need a new `--name` |
| BALANCE | 0.1.40: 10-06 to 10-19, 55 values | load, with a note | promote; fine-tune if an OUR DECK or POOL card changed |
| NEW_FORMS | Evo Giant, Evo Mirror (expected) | live-compat loads with a note; `types` checkpoints are refused | promote; fine-tune |
| NEW_CARDS | a new base card, appended | refused until `grow`, which promote runs | promote with `grow`; fine-tune |
| BREAKING | a pinned card stops loading; ENTITY_FIELDS, ABILITY_BUTTONS, SPELL_FIELDS or STATUS_BITS change; the v2 schema digest changes | refused | stay on the old pin; rebuild rows and retrain |

**BLOCKED** overrides the class when any of these hold:
- the target arm does not build;
- a table FNV does not match the engine's;
- the tests fail;
- the grid regresses (section 9);
- a Hog 2.6 or card-map name is missing from the new catalogue.

The report then names what to wait for (an upstream release) or what to fix (the failing tests, to hand to Claude).

**BREAKING** should be rare. ENTITY_FIELDS last changed in 0.1.17 (27 to 28 fields), and card ids only grow. If it happens:
- live play stays on the old pin;
- rows can be built with `.venvs\rs_next` into new folders;
- Claude writes the retrain plan then.

There is no second-worktree flow: `hastycr.machine` resolves `data/` and `runs/` from the repo root (`src/hastycr/machine.py:350-356`), so a second checkout would see none of the data.

### 4.5 Promote and rollback

**`promote` refuses when:**
- the verdict is BLOCKED or the class is BREAKING;
- the files it edits have uncommitted changes;
- the staged inputs changed since the stage;
- the job lock (U12) shows an RL run, an evaluation or a grid running. Reinstalling the engine under a running rollout breaks it.

**Steps:**
1. Tag HEAD `pin/<date>-rs<old>`.
2. Edit only data and pin files (after U11):
   - the `royalesim==` line in `requirements/engine.txt`, `requirements/gpu.txt` and `pyproject.toml`;
   - `data/engine.json` `card_table`;
   - `data/game_versions.json` (current pin, history);
   - the staged `classes.json` and `spawn_children.json`;
   - the report.
3. Commit (the "Y" is the consent).
4. Run `scripts\setup_windows.ps1`, which reinstalls both venvs, then `selfcheck.py --offline` in each.
5. **Load check.** The live checkpoint, the KL ref and the opponent stand-in must load under the new pin. For NEW_CARDS, `tools/rs_vocab.py grow` writes carried copies first (U14). If any of them fails, promote rolls back by itself and says why.
6. Move the live pointer to the carried checkpoint after a smoke test (section 6.5).

**`rollback [--to TAG]`:**
- `git revert` of the promotion commit;
- reinstall (no release has been yanked, and the old wheel is also cached in `<runs>/updates/cache/`);
- selfcheck;
- the live pointer back to its previous entry.

Nothing is deleted, so old checkpoints and rows load again at once.

### 4.6 Who does what

| Step | Automatic | Owner |
|---|---|---|
| Daily pull, version check, roster radar, calendar, status | yes (scheduled task) | register the task once |
| Weekly top-player pull | yes (Mondays) | none |
| Deciding an update is due | the currency line says so | read it |
| Stage, classify, test, grid | yes, in `-Update` | close the browser; about 1 h |
| Promote, reinstall, carry checkpoints, refresh pool | yes, in `-Update` | one "Y" |
| Accepting a new client build | yes, when nothing else changed | `-AcceptClient -Note` otherwise |
| RL fine-tune and gates | yes, in `model_update -Apply` | start it in the evening, close MuMu |
| Card-map entry for a renamed or new card | proposed with `"verified": false` | confirm, or ask Claude to commit it |
| Detector labels for new cards | partly (`docs/NEW_CARDS_PLAN.md`) | record friendlies with the main account |
| HUD templates after a client change | none | one dry-run friendly |
| Asking upstream for a table | the report says when it is overdue | send the request (section 10) |
| Rollback | one command | decide to run it |

## 5. How a stale bot is stopped from playing

**The live runner refuses taps, but still runs dry so the owner can watch, when:**
1. the checkpoint does not fit the running engine (exists: `scripts/vision_play.py:1377`);
2. the installed royalesim differs from the pin in `data/game_versions.json` (new, U05);
3. the existing STOP file or `emulator.verify` refusals apply.

**It warns, in the header and the session log, when:**
- the client build is not in the ledger;
- a season start or listed balance date passed with no check since;
- the last check is more than 7 days old;
- the last pull is more than 48 h old;
- a newer table is known but not promoted;
- the roster radar shows cards or forms the table lacks;
- the checkpoint carries a drift note (royalesim version or table FNV differs from its rows);
- the client line is ahead of the table line (shown as accepted, with the date).

**Why not block taps on a new client build.** Builds arrive about every two weeks, and the owner has already judged one such gap acceptable. A WARN in every header keeps the gap visible, and blocking stays one owner decision away (section 13).

Selfcheck prints the same judgement.

## 6. The model: carrying, generalising, retraining

### 6.1 Carrying checkpoints across a new table

**Today** an arm change refuses every checkpoint even when every card and unit id is the same (`tools/rs_checkpoint.py:238-241`). The only escape is `HASTYCR_ALLOW_VOCAB_DRIFT=1`, which is all-or-nothing and also hides real id shifts.

**The new rule** (U01, in `vocab_differences`):
- An arm mismatch becomes a note, not an error, when the card names match in id order, and also the unit names for a checkpoint that is not `--live-compat`.
  - The guard exists to catch renumbering (`tools/rs_checkpoint.py:10-16`), and the name comparisons at `:226-235` already catch it.
- New warn-only notes, for cases that load silently today:
  - the royalesim version recorded in the meta differs from the engine's;
  - the arm is the same but `card_table_fnv1a64` differs.
- The refusal message names `tools/rs_vocab.py grow` as the fix for appended cards.

No checkpoint copies and no lineage ledger are needed, and rollback stays trivial.

**`tools/rs_vocab.py`** (U14) has four commands:
- **`diff`:** cards added at the end, removed or moved; units added or removed; arm, FNV and version.
- **`grow`:** copies `card_emb` rows by name into the new catalogue order. Under live-compat, `unit_emb` uses the same map. Rows for new names are set to the unknown row (token 1).
  - It checks function preservation (max |Δlogit| below 1e-4 on old tokens) before writing.
  - It writes `best.pt` plus `args.json` carrying `wait_keep`. Without that file, `rs_decide.wait_keep_for` (`tools/rs_decide.py:232-244`) falls back to the default wait bias.
  - The result must pass a strict `load_policy`.
  - A removed card makes it BREAKING.
- **`reset-cold`:** sets the rows of cold cards (section 6.2) to the unknown row. That fixes Minion Giant's random starting row today.
- **`check`:** OK, REFUSED (with the fix), ENGINE_DRIFT, or COLD (cold cards that are in the current pool).

### 6.2 Cards the model has never seen

**How the model sees cards.**
- Cards and units are pure id embeddings (`sim/model_tokens.py:133-134`). There are no stat features.
- A card never seen in training keeps its random N(0,1) starting row:
  - the 22 cards outside the regular pool (event, test and internal; `docs/NEW_CARDS_PLAN.md` 1.1);
  - Minion Giant;
  - every future card.
- Token 1 means "unknown". It is trained only by perception noise (`unknown_p` 0.01 for units and 0.02 for plays, `data/noise/default_prior.json:45,116`).

**Definitions used everywhere in this routine:**
- **Cold card:** fewer than 500 IL_Replay train seats (`deck_stats` `meta.card_seats`, added in U16), or added since the champion's rows were built.
- **Matters:** a card is in Hog 2.6, or carries at least 3% of the opponent pool's weight. Until the pool exists, IL_Replay `deck_stats` stands in.

**Fixes in order of cost:**
1. **`reset-cold`** (U14): cold rows become the unknown row. Cheap, no retrain.
2. **Live visibility** (U15): the live runner counts token-1 units and plays per match and logs `unnamed opponent cards: N`.
3. **The safety net** of `docs/NEW_CARDS_PLAN.md` N02: a play vision cannot name reaches the policy as UNKNOWN and is charged an expected cost.
4. **Card stat features** (U20, later, opt-in):
   - a fixed stat vector per catalogue id from `catalogue_json` (13 columns, `py.rs:423`) and the card table;
   - a zero-initialised stat path in `TokenPolicy`;
   - id-dropout during imitation, so the model can act on stats alone.

   It needs a full imitation retrain. It is adopted only if a cold-card evaluation shows it helps (U20's rule). It is last on the list.
5. **Opponent forms in history tokens** (U21, optional): `opp_seen` and `opp_last` carry card ids only (`sim/model_tokens.py:203-209`), so an opponent's evolution or hero cycle is invisible there. Adding it changes the v2 schema and needs rebuilt rows.

### 6.3 Retraining rules by class

**RL fine-tunes start only once a v2 champion exists.** The live model is the legacy `live_big_hog`, and the v2 decision-time model is being diagnosed (29.3% against 46.0%, `docs/SPARRING_PLAN.md`). Until then `model_update` stops after carry-over and the smoke test.

| Class | Rows | Imitation | RL fine-tune (iterations at `--matches 400`) |
|---|---|---|---|
| SAME, ENGINE_ONLY | keep | keep | none, unless the grid moves winner agreement by 1 point or more, or the ledger touches a Hog 2.6 card; then as BALANCE minor |
| BALANCE, minor (no Hog 2.6 card changed by more than 5%, fewer than 3 POOL cards by more than 10%) | keep | keep | 10-20 (one night) |
| BALANCE, major, or NEW_FORMS | keep | keep | about 30 |
| NEW_CARDS | keep; rebuild when IL_Replay gains the card | `grow` + `reset-cold` | 60-80 (a weekend) |
| BREAKING | rebuild | retrain (`scripts/third_model.ps1`, 7-12 h) | then RL |

**These budgets are guesses.** The only measured RL timing is about 300 policy matches in 6.5 minutes (`docs/SPARRING_RUNBOOK.md:732`). Run `train_loop --bench` first.

**RL settings:**
- Every RL run after an update uses a new `--name`, because resume refuses a changed engine id.
- `model_update` copies `--ref` and the KL settings from the champion's own run `args.json`, so the KL pull does not undo earlier RL.
- The opponent field is the newest pool export (section 7.5), passed as `--deck-stats` with `--deck-top 0`.
- Cold decks get at least 30% of the field when any exist.

### 6.4 Promotion gates

**A new RL champion goes live only when all of these pass:**
- **G1, general strength:** the existing gate inside `train_loop`, on IL_Replay dev pairs.
  - Against the champion: Wilson 95% lower bound above 0.52 over 1,000 games, so 551 passes (`tools/rs_league_eval.py:56-59`).
  - Each anchor: a drop of at most 2 points over 400 games.
- **G2, the clan meta:** `rs_league_eval run --split train --pairs-file <pool pairs> --anchor pilot=<stand-in> --full`, with the candidate against the carried champion.
  - Against the stand-in on the pool decks, the candidate may score at most 2 points below the champion (`anchor_condition`).
  - Against an anchor the evaluated player always keeps side a, so it plays Hog 2.6 against each pool deck (`tools/rs_league_eval.py:38-40`).
- **G3, the cold slice** (NEW_FORMS and NEW_CARDS only): the same as G2, on pairs built from cold-card decks.

**Outcomes.** Otherwise the carried champion is kept, and the RL champion stays in its run folder. Most balance updates will end KEPT: G1 needs about a 5-point edge. That is intended, and it costs nothing because the carried champion is already live.

**Live friendlies are the last check**, but they cannot prove a drop quickly. At 20 friendlies the Wilson half-width is about ±20 points. The report shows the numbers, and rollback is one command.

### 6.5 The live pointer

**There is one pointer:** `<runs>/live/champion.json` {path, sha256}, with `history.jsonl` (U15).
- `vision_play.default_ckpt` (`scripts/vision_play.py:1269`) reads it and checks the sha256. On a mismatch or a missing file it falls back to today's default with a notice.

**What may move it:**
- **A carried checkpoint** (the same file loading under the new pin, or a function-preserving `grow`) after a smoke test (S0) in the new engine. S0 is `rs_league_eval` against `idle`, 100 games: 0 errors and a score of at least 0.95.
- **An RL champion** only after G1, G2 and, when run, G3.

`scripts\game_update.ps1 -Rollback` and `scripts\model_update.ps1 -Rollback` both step back through `history.jsonl`.

## 7. The opponent deck pool

### 7.1 Source

**The source is the official API** (`api.clashroyale.com/v1`). It is the sanctioned way to read public player and clan data.
- The repo already has a client, `tools/ladder_pull.py`, with its token in `.secrets/supercell_token` (gitignored).
- `ladder_pull` drops forms (`deck_of`, `:118-120`) and skips friendlies (`:42`).
- **U02** extracts a shared client, `tools/cr_api.py`:
  - it returns HTTP codes instead of exiting on 403;
  - it backs off on 429;
  - it always sends a custom User-Agent;
  - it supports the RoyaleAPI proxy (`https://proxy.royaleapi.dev/v1`, with 45.79.218.79 whitelisted on the key).

**What the API gives:**
- `/v1/cards`: the live roster, with `maxEvolutionLevel` bits (1 evolution, 2 hero, 3 both).
- `/clans/{tag}/members`.
- `/players/{tag}/battlelog`: about 25 recent battles, including friendlies, with both decks and crowns.

**What it does not give:** balance numbers or replays. Balance always comes from the RoyaleSim table.

**Unconfirmed until one probe on the PC** (U03 `probe`, after a friendly where the main account plays an evolution in slot 1 and a hero in slot 2):
- whether battle-log cards carry the form played;
- whether a hero-slot card goes missing (77 of 300 top players' `currentDeck` had 7 cards);
- the battle `type` strings and the `deckSelection` value;
- the log length;
- the card levels shown for a friendly.

Spars are recognised by opponent tag (clan roster, owner, friends list), not by type string.

### 7.2 What is stored

Everything lives under `<data>/opponents/`, which is gitignored (U03):
- `config.json`;
- raw responses, kept 30 days so a parser fix can re-parse;
- `battles.jsonl`: one normalised 1v1 battle per line, deduplicated on battle time plus sorted tags, as `ladder_pull` does;
- `meta_battles.jsonl`: rolling 8 weeks;
- `manual.json`: owner-entered decks (`opponents.py add`);
- `cards_api.json` and the previous snapshot, for the radar;
- `state.json`.

**Yodle2's own log is a vision-free record of every sparring result:** both decks, crowns and tower HP.

### 7.3 Card and form resolution

**The chain:**
1. API name to RoyaleAPI key by one slug function (lower case, dots dropped, other non-alphanumerics to `-`). It matches all 120 cards of `data/royaleapi/cards.json`.
2. Key to RoyaleSim name through `rs_cards.load()`.
3. The name must be in the engine catalogue.
4. The form must be in `classes.json` `cards[name].forms`.

**Per-card status:**
- **exact**.
- **downgraded:** the card exists but not that form; it is played as base and counted. Example: Evo Giant and Evo Mirror from 2026-11-02.
- **auto:** no key, but exactly one catalogue name equals the display name in CamelCase with the same elixir. Example: Minion Giant. The entry for `replay_cards_extra.json` is printed with `"verified": false`; it is never merged by itself.
- **blocked:** not in the engine, or an unknown form.
- **incomplete** or **inferred:** 7 cards.

**Rules:**
- Blocked and incomplete decks are never exported. They are always reported with their weight share and real results.
- Resolution reruns on every build, so a promotion clears blocked decks with no re-pull.
- **Tower troops:** RoyaleSim always plays the Tower Princess (`tools/royalesim_redrive.py:77-78`). The pool counts opponents on other tower troops (from `supportCards`), and the report prints that share next to the grid's per-tower rows.

### 7.4 Weights

- **Recency:** 0.5^(age / H), with H 21 days for Yodle2's spars, 14 for clanmates' own battles and 10 for top players.
- **Clanmates:** each member's decks are normalised to 1, so one grinder does not dominate. A member who sparred Yodle2 in the last 30 days counts ×3.
- **Pool weight** = 0.4 spars + 0.4 clan + 0.2 top players, renormalised over the sources that exist.
- **Families:** decks sharing at least 6 of 8 base cards, for the report.

### 7.5 Exports

Both export kinds are immutable and digest-named, in formats the existing code reads.
- **RL field** (`exports/field_<date>_<sha8>.json`): the `deck_stats.json` schema with only a `train` split.
  - Shares: spars 0.34, clan 0.34, top players 0.17, IL_Replay 0.15. The IL share keeps at least 1,000 distinct decks (`MIN_DECKS`, `tools/rl/train_loop.py:181`).
  - Synthetic cold-card decks are an optional source: an IL deck with the closest same-kind card replaced.
  - `train_loop` freezes the file's sha256 (`COMPARED`), so a running run keeps its field.
- **Pairs** (`exports/pairs_<date>_<sha8>.json`): an `rs_match` pair set with split `train`, side a Hog 2.6 in base forms, side b a pool deck.
  - Tags are `pool:<deck_id>:<k>`, so a deck gets the same games in every export, under every engine and for every checkpoint.
  - The split is `train` because pool decks are in the RL field. `Protocol` accepts `train` (`tools/rs_league_eval.py:243-247`).
  - About 600 pairings cover 90% of the spar and clan weight, or 80 decks.

### 7.6 Clan check and matchup report

- **The clan check is G2** (section 6.4), run by `scripts\opponents_eval.ps1` (U17). There is no separate match runner.
- **The report** is a reader over the `rs_league_eval` JSONL game log (`tools/opponent_report.py`). It shows:
  - per family and per deck: weight, games, score with a Wilson interval, crown difference, and the real friendly W-D-L from Yodle2's log;
  - the 10 worst decks;
  - blocked decks with real results only;
  - simulated against real for decks with at least 3 real friendlies;
  - the pool-weighted "sparring score".
- **The stand-in** opponent is the general imitation model. It is weaker than top-1000 players, so simulated scores are optimistic. Yodle2's real results are the calibration.

### 7.7 Courtesy and privacy

- Tell the clan that the bot reads members' public decks and battle logs through the official API.
- The data stays on the PC in gitignored folders. Exports contain decks only.
- `data/ladder/battles_2p6.csv` no longer carries the 1,467 top players' names and tags. The owner decided on 2026-10-30 to keep only the battle time, rating, rank, both decks, the crowns and the result. Older commits, and the original public repository, still hold the names. `tools/ladder_pull.py` output, which has tags, is gitignored.

## 8. Vision: hand-off to docs/NEW_CARDS_PLAN.md

`docs/NEW_CARDS_PLAN.md` owns detector coverage. Its section 7 is the vision part of this routine. This plan does not duplicate it.

**This routine hands it:**
- a regenerated `classes.json` from the new table, through `tools/vision_classes.py --engine` only. Neither plan copies or hand-edits it, because its digest feeds `scripts/vision/labels.py:194,262` and `scripts/vision/detector.py:502`;
- the new classes, and the API forms and cards the detector lacks;
- the pool's card-usage ranking, as labelling priority.

**Owned there:**
- **N01:** spawn relations from the pinned table, plus the Heal Spirit and Spirit Empress fixes.
- **N02:** the UNKNOWN safety net.
- **N07:** `valid_from_client` for reworks.
- **N16:** `vision_gap_labels report`, the season report.

**Added here: a per-session vision-health record** (U13). The live runner writes one summary per session, from what it already measures:
- the hand-slot template match quality;
- the share of unresolved detections;
- tower-HP and elixir read failures;
- unknown opponent plays.

The daily check compares it with a baseline and flags a drift. That catches card art, HUD and skin changes that no version number reveals.

**HUD templates** (`data/vision/templates/`): after any client change, play one dry-run friendly (runbook section 17) and run `python -m scripts.vision.templates list`.

**Royal Chef** has no KataCR class (`docs/NEW_CARDS_PLAN.md` 1.6), and RoyaleSim does not simulate tower troops other than the Princess. Both are reported, not fixed, here.

## 9. Fidelity

**The grid** (`tools/rs_fidelity_grid.py`) re-drives IL_Replay val matches and scores winner and exact-crown agreement. Its target is `TARGET_WINNER = 0.626` (`:79`).
- U10 adds candidate cells for the side venv and a shared payload file. Every update is then scored on the same 2,000 matches, which gives a time series.
- The first `-Update`, even with no new release, is the overdue baseline for 0.1.42 and the 16.402.22 table.

**Pass rule.** This is a regression guard, not a validation of the new table. The update is BLOCKED when:
- the target cell's winner agreement drops by at least 2 points with McNemar p < 0.05;
- refusals rise by more than 1.0 per 100 plays;
- failed matches exceed the base plus 1% of the sample.

**Why the grid cannot validate a new table.** IL_Replay was played under older balance, and its date range is unknown. A table that matches today's game can score slightly lower on it. Only a significant drop blocks.

**Later, optional (U22):** re-drive Yodle2's own recorded friendlies. They are the only matches played under the live balance. The vision event logs give both sides' plays, and Yodle2's battle log gives crowns and tower HP.

## 10. When RoyaleSim is slow

**The floor.** A card or form RoyaleSim lacks cannot be simulated, so RL and the gates cannot train or test against it. Until it ships:
- live play has the UNKNOWN fallback (N02) and token 1;
- the pool downgrades or blocks those decks and reports their share;
- the currency line shows "days behind": days since the first unmatched event (a new client line, season start or listed balance date) with the pool weight affected.

**After 7 days with no release** following a balance change or new cards, the report says to ask upstream. The PyPI README links github.com/RoyaleGym/RoyaleSim and a Discord. The sandbox could not reach either, so the request process is unknown.

**Optional fallback (owner decision): a local source build.**
- A build from the sdist keeps the default `checkout-data` feature, which reads `data/derived/*.json` at run time (`crates/royalesim/Cargo.toml:13-19`).
- So numeric deltas from patch notes could be entered by hand, and the FNV identity changes with them.
- It needs Rust and MSVC on the PC. New cards, forms or mechanics would still need Rust code.

## 11. Hand-kept card lists

**Two can be derived** from the catalogue's `placement` column (`CATALOGUE_FIELDS`, `py.rs:423`; code 3 rolling, 5 tunnel, 6 mirror, `py.rs:60-69`, `:727-756`):
- `ROLLING_SPELLS`, `TUNNEL_CARDS` and `MIRROR_CARDS` in `tools/rs_decide.py:745-750`;
- `ANYWHERE` in `tools/rs_orient.py:62`.

U07 derives them. A test asserts that the derived sets equal today's sets on the pinned table; where they differ, the hand list stays and the reason is recorded.

**The rest stay hand-kept.** The report prints them as a checklist when the roster grows:
- `GUARD_CARD_NAMES` and `SPAWNING_SPELLS` (`tools/rs_decide.py:168`, `:749`);
- `CHAMPIONS_BEFORE_THE_COLUMN` (`tools/rs_presses.py:64`);
- `KATACR_SPELL_CLASSES` (`scripts/vision/events.py:113`), owned by `docs/NEW_CARDS_PLAN.md`.

**Form suffixes** (U07): `base_key` in `tools/deck_stats.py` and `tools/royalesim_redrive.py`, and `FORM_SUFFIXES` in `tools/rs_presses.py`, report an unknown suffix by name (for example a future `-ev2` or `-hero-ev1`) in the manifest and in `rs_cards --coverage`. Today such a match is skipped as an unmapped card.

## 12. Work items

**Testing rules:**
- Tests go in `tests/pipeline`. They use saved JSON fixtures and the conftest FakeBattle, and they never touch the network.
- Markers: `engine` (needs royalesim), `gpu`, `device` (needs `HASTYCR_DEVICE_TESTS=1` and a running emulator). Every marked test skips cleanly.
- New flags and subcommands must not contain "ladder", "ranked" or "queue" (`tests/pipeline/test_friendly_only.py:94-96`). New `.ps1` files must not match `$Ladder`, `-Ladder` or `--ladder` (`:103-104`).

| ID | Title | Files | Depends | Tests |
|---|---|---|---|---|
| U01 | Checkpoint arm rule: an arm mismatch is a note when names match; warn-only notes for royalesim version and table FNV drift; the refusal names `rs_vocab grow` | `tools/rs_checkpoint.py` | none | `test_vocab_guard.py`: same names on another arm loads with a note; an appended card still refuses; a types checkpoint with changed units refuses; a live-compat one only notes; FNV and version notes |
| U02 | Shared API client: `Rate`, `get` returning codes, 429 back-off, custom User-Agent, proxy base, one slug function; `ladder_pull` imports it | `tools/cr_api.py` (new), `tools/ladder_pull.py` | none | `test_cr_api.py`: slug on all 120 cards of `data/royaleapi/cards.json`; 403 and 429 handling with a fake transport |
| U03 | Opponent pull and store: `init`, `pull`, `meta`, `add`, `probe`; normalised `battles.jsonl`; dedupe; spar detection by tag; `.gitignore` | `tools/opponents.py` (new), `data/opponents.example.json` (new), `.gitignore` | U02 | `test_opponents.py`: 2v2 dropped, dedupe across two logs, 7-card deck, each form rule; no import of `scripts/gamestate` |
| U04 | Version ledger and daily check: `data/game_versions.json` seeded with today's facts; season rule; PyPI and sdist reader (ledger arms, client build, content, FNV); royaleimitate keys; roster radar; calendar; IL_Replay revision; status file; exit codes | `data/game_versions.json` (new), `tools/game_versions.py` (new), `tools/game_update.py` (new: `check`) | U02, U03 | `test_game_versions.py`: season starts reproduce 2026; client line; `test_game_update_check.py` on saved PyPI JSON and a tiny sdist fixture; radar on two `/v1/cards` fixtures |
| U05 | Currency line: selfcheck `game` reads the ledger (replacing `RECORDED_GAME_BUILD`, `scripts/selfcheck.py:83`), new `update` line; live header and session log; taps refused when royalesim differs from the pin | `scripts/selfcheck.py`, `scripts/vision_play.py` | U04 | `test_selfcheck.py`: 160635001 still PASSes; an unknown build WARNs; `test_vision_play.py`: a pin mismatch turns taps off, dry run continues |
| U06 | One scheduled task: `-Daily` (pull, check, Mondays top players, status) and `-Register` (Task Scheduler, StartWhenAvailable, owner logged on, no stored password) | `scripts/game_update.ps1` (new) | U03, U04 | `test_game_update_ps1.py`: ASCII only, no ladder flag, no adb call |
| U07 | Cheap fixes: unknown form suffixes reported by name; placement sets derived from the catalogue; `classes.json` source records `card_table_fnv1a64()` | `tools/deck_stats.py`, `tools/royalesim_redrive.py`, `tools/rs_presses.py`, `tools/rs_decide.py`, `tools/rs_orient.py`, `tools/vision_classes.py` | none | `test_deck_stats.py` suffix cases; `test_rs_decide.py` derived sets equal today's on a FakeBattle catalogue, plus an `engine` test on the pinned table; `test_vision_classes.py` |
| U08 | Card-table and ledger diff (stdlib CLI): roster by `form_of`, flattened values with provenance ignored, echoes folded onto cards, OUR DECK / POOL / OTHER tags, ledger keys by section | `tools/card_table_diff.py` (new) | none | `test_card_table_diff.py` on two tiny fixture tables: an added evolution, provenance ignored, an echo folded, OUR DECK tag |
| U09 | Stage: side venv with hash pin, probe of both engines, target arm, FNV check, diff, checkpoint table, derived data into the stage, tests in a worktree, quick grid, one classifier, report | `tools/game_update.py` (`probe`, `stage`) | U01, U04, U08, U10, U11 | `test_game_update_stage.py`: class and verdict rules from synthetic probe and diff results; `engine` test: the probe reports FNV `15bb4e60e69d05f3` on 0.1.42 |
| U10 | Fidelity grid candidate cells: `--candidate PY`, `--candidate-tables`, shared payloads | `tools/rs_fidelity_grid.py` | none | `test_fidelity_grid.py`: candidate cells appear with their own label; McNemar against the base |
| U11 | Promotion edits data, not code: `TARGET_TABLE` and `BUILTIN` from the ledger; `KNOWN_TABLES` / `TABLE_ALIASES` merged with the wheel's ledger arms; shipped-pin tests read the ledger | `tools/rs_engine.py:95-126`, `tests/pipeline/test_rs_engine.py`, `tests/pipeline/test_requirements.py` | U04 | the edited tests; `apply_pin` on temp copies changes only the expected lines |
| U12 | Promote, accept-client, rollback, job lock: refusals, tag, edits, commit on consent, reinstall, selfcheck, load check with `grow`, automatic undo on failure | `tools/game_update.py` (`promote`, `update`, `accept-client`, `rollback`), `tools/job_lock.py` (new), `scripts/game_update.ps1` | U01, U09, U11, U14, U15 | `test_game_update_promote.py` on a temp git repo: a dirty tree refuses; a held lock refuses; a failed load check reverts; rollback restores the pin |
| U13 | Vision-health record per session and its baseline check | `scripts/vision_play.py`, `tools/game_update.py` | U05; coordinate with N02 and V07 | `test_vision_play.py`: the summary is written from a recorded session; the check flags a drift on fixtures |
| U14 | `rs_vocab`: `diff`, `grow`, `reset-cold`, `check` | `tools/rs_vocab.py` (new) | U01 | `test_rs_vocab.py`: FakeBattle catalogues same, appended, removed; function preservation; `args.json` carried; the strict load passes |
| U15 | Live pointer and unknown-card counter: `champion.json` with sha256, `history.jsonl`, `default_ckpt` reads it, fallback notice, `unnamed opponent cards` per match | `tools/live_pointer.py` (new), `scripts/vision_play.py` | none | `test_live_pointer.py`: sha mismatch falls back; rollback steps back |
| U16 | Pool build, export, pairs, status; `card_seats` in deck stats; tower-troop share; Minion Giant as the first `auto` proposal | `tools/opponents.py`, `tools/deck_stats.py` | U03, U07 | `test_opponents.py`: Mirror form 1 downgraded, Giant form 2 exact, Minion Giant auto, an invented card blocked; the export loads with `DeckField.from_file(path, "train")` and keeps its shares within 1%; pairs round-trip through `PairSet.load` with split `train` |
| U17 | Clan check and report: `opponents_eval.ps1` runs `rs_league_eval` on the pool pairs; `opponent_report.py` reads the game log | `scripts/opponents_eval.ps1` (new), `tools/opponent_report.py` (new) | U16 | `test_opponent_report.py`: aggregation from synthetic game lines, exit codes; `engine` smoke run of 2 pairs |
| U18 | Model update: `plan` reads the stage class; `apply` runs S0, RL with the pool field, G1-G3, pointer move; `status`, `rollback`; RAM check; job lock | `tools/model_update.py` (new), `scripts/model_update.ps1` (new) | U12, U14, U15, U16, U17; a v2 champion for the RL steps | `test_model_update.py`: plan from fake diffs, step resume, gate-report reading, KEPT on a G2 failure |
| U19 | Docs: one link line in `docs/SPARRING_RUNBOOK.md` under "Later steps" until the scripts exist (`tests/pipeline/test_requirements.py:254`), then a runbook section | `docs/SPARRING_RUNBOOK.md`, this file | U06, U12 | `test_requirements.py` (runbook only runs scripts that exist) |
| U20 | Card stat features (optional, last): `card_features.py`, the stat path and id-dropout in `TokenPolicy`, `-CardFeats` in the model scripts, cold-card evaluation | `tools/card_features.py`, `sim/model_tokens.py`, `sim/train_tokens.py`, `tools/cold_eval.py`, `scripts/first_model.ps1`, `scripts/third_model.ps1` | U14 | `test_card_features.py`; `test_model_tokens_feats.py`: zero-init gives identical logits, legacy and v2 golden weights unchanged; `test_cold_eval.py`; a `gpu` smoke epoch |
| U21 | Opponent form in history tokens (optional) | `tools/rs_schema.py`, `tools/rs_rows.py`, `sim/model_tokens.py`, `tools/public_state.py` | a v2 champion | schema digest changes only under the new flag; old rows still load |
| U22 | Re-drive recorded friendlies as a current-balance fidelity corpus (optional) | `tools/friendly_redrive.py` (new) | U03, recorded sessions | on a synthetic event log and a battle-log fixture |

**Order:**
1. U01, before the next RoyaleSim release. It will likely follow Season 89 and bring a new arm; without U01 every checkpoint, `live_big_hog` included, would be refused at `scripts/vision_play.py:1377`.
2. U02, U03 and U06 with a minimal `check`. Every day without them loses battle-log history.
3. U04, U05, U07 and U15.
4. U08, U10, U11, U09, U14, U12: the one-command update.
5. U16, U17, U13.
6. U18 once a v2 champion exists.
7. U20, U21 and U22 last, and only if wanted.

**Coordination:** other workflows edit `tools/rl/*`, `scripts/vision/templates.py`, `scripts/vision/harvest.py`, `docs/SPARRING_RUNBOOK.md`, `docs/LIVE_SEARCH_PLAN.md` and `docs/NEW_CARDS_PLAN.md`. No item here edits `tools/rl/*`; RL is driven through its CLI. Re-check line numbers and flags when those edits land.

## 13. Decisions for the owner

1. **API key.** Decided 2026-10-30: yes. The owner creates one at developer.clashroyale.com, bound to the PC's IP, and saves it as `.secrets\supercell_token`. The key stays on the PC.
   - With a changing home IP, either edit the key when the IP changes, or use the RoyaleAPI proxy.
   - The proxy shows the token to a third party.
2. **Clan courtesy.** Decided 2026-10-30: yes. The owner tells the clan.
3. **The committed CSV.** Decided 2026-10-30: names and tags are removed from `data/ladder/battles_2p6.csv`, and the decks stay.
4. **New client builds.** WARN only, as proposed, or block taps until reviewed?
5. **Live pointer.** May a carried checkpoint go live by itself after the smoke test? An RL champion always needs the gates.
6. **Known gaps.** Keep playing while Evo Giant and Evo Mirror are simulated as base forms (from 2026-11-02 until RoyaleSim ships them)?
7. **Pool shares.** Accept the defaults: spars 0.34, clan 0.34, top players 0.17, IL_Replay 0.15?
8. **Upstream contact.** Who asks RoyaleSim for a table after 7 days of silence, and where?
9. **Optional work.** Install Rust and MSVC for a local RoyaleSim build (section 10)? Fund the stat-feature retrain (U20) after the v2 diagnosis?
10. **Schedule.** The default is a daily run at 05:47 local, made up at the next logon if missed. Change it?

## 14. Risks

1. **Upstream is the only source of tables.** The routine can show "waiting" loudly; it cannot make the simulator current.
2. **Server-side content changes** can land with no new build and no new table. The calendar, the roster radar and the owner reading patch notes are the only detectors for balance-only server patches.
3. **The grid guards against regressions only.** It cannot show that a new table matches the live game.
4. **Carry-over keeps play trained under the old balance.** That is the intent for small patches. Big patches need RL in the new table and real friendlies.
5. **The API's form encoding** in battle logs is unconfirmed. The worst case is base forms everywhere, with a WARN.
6. **Simulated opponents are weaker than top players**, so pool scores are optimistic. Cold cards are played from an untrained or unknown row.
7. **The RL budgets are unmeasured.** Run `train_loop --bench` first.
8. **Supply chain.** The routine installs a native wheel from PyPI. Mitigations: hash pin, side venv first, tests and grid before promotion, a wheel cache, one-command rollback.
9. **16 GB RAM.** Stage, RL, grid and evaluations must not overlap. The job lock enforces it, and the scripts remind the owner to close MuMu and the browser.
10. **Supercell's Terms of Service prohibit bots, friendlies included.** The owner accepted that risk for Yodle2 (`docs/SPARRING_PLAN.md:21`). This routine reads only public data and its own files. It adds no evasion, reads no memory, never starts MuMu and never taps.
11. **Facts from search snippets only:** the Season 89 contents, the "no balance changes" claim, and the APKMirror dates. Re-check them on the PC.

## 15. Sources

- **PyPI:** `https://pypi.org/pypi/royalesim/json` and the per-version JSON; the 0.1.11-0.1.42 sdists (`data/calibration.json`, `data/derived/cards-*.json`, `crates/royalesim/src/py.rs`, `Cargo.toml`).
- **Official API notes:**
  - https://github.com/RoyaleAPI/cr-api-docs/blob/master/docs/proxy.md
  - https://raw.githubusercontent.com/samaksh-bajaj/cr-deck-builder/main/API_NOTES.md
  - https://gist.github.com/loganlinn/d813ba0f9ccb47f34462ec8abeab8e14
- **Season 89 and October 2026** (search snippets):
  - https://timesaver.gg/blog/clash-royale-season-89-king-vs-king
  - https://royaletracker.gg/news/clash-royale-update-november-2026
  - https://thephrasemaker.com/2026/10/07/clash-royale-october-2026-balance-changes-all-buffs-nerfs-reworks-and-more/
- **Client builds:** https://www.apkmirror.com/apk/supercell/clash-royale/
- **Repo:** `docs/SPARRING_PLAN.md`, `docs/NEW_CARDS_PLAN.md`, `docs/LIVE_SEARCH_PLAN.md`, `docs/PROGRESS_2026-10-09.md`, and the files cited above.
