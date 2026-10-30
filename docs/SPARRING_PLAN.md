# Sparring bot: build plan

Goal: the strongest practical Clash Royale bot for **friendly battles** against our clanmates (many of them top-1000 players), playing **Hog 2.6** (Cannon, Fireball, Hog Rider, Ice Golem, Ice Spirit, Musketeer, Skeletons, The Log).

Hardware: Windows 11 PC with MuMu Player, an RTX 4070 Super (12 GB), 16 GB RAM and 500 GB disk.

## This PC (measured 2026-10-29)

| Item | Value |
|---|---|
| CPU | Intel i7-14700F: 20 cores, 28 threads |
| RAM / GPU / disk | 16 GB / RTX 4070 Super 12 GB / 500 GB |
| MuMu | instance 0, "Android Device", Android 15 |
| MuMu instance resources | 1 core, 1 GB. Raise to 2-4 cores and 3 GB before live play. |
| adb | `C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe` |
| adb serial | `127.0.0.1:16384` |
| Screen | 1080x1920 at 480 DPI (what the code assumes) |
| Clash Royale `versionCode` | `160635001` |
| Friendly-battle card levels | all cards at **level 11** (confirmed by the owner), so training, RL and evaluation use level 11 for both sides |
| Bot deck | Hog 2.6, **all base cards: no evolutions or heroes** on the bot account. Tower troop: **Tower Princess** (both sides in the first recorded friendly). |
| Bot account | the owner's **second account** ("Yodle2"), which owns the full Hog 2.6 deck. The owner accepts the ToS/ban risk on it. A brand-new account was ruled out because it would not own The Log and the other deck cards. |

The game build is newer than:
- the original author's build, `160402012`;
- RoyaleSim 0.1.42's newest card table, `client160402017_20261019`.

So any balance changes in this update are not yet in the simulator. The owner, who knows the game well, reports that nothing major changed for our purposes since the 19 October Season 88 mid-season balance update. RoyaleSim's 16.402.22 card table is therefore the target, and small differences will be handled later.

Done on the PC so far:
- the fork is cloned at `C:\hcr` with a sparse checkout, on the `sparring` branch;
- IL_Replay is downloaded to `C:\hcr\data\il_replay`: 104 files, 1,888 MB, all SHA-256 verified.
- `.venvs\royalegym` is created with `royalesim==0.1.42`, numpy 2.5.4 and pyarrow 26.0.0. Engine probe on Windows: `0.1.42 cards-15535.1 11 11 237`. That means:
  - the default card table is the older 15.535 one;
  - card level and tower level are both 11;
  - there are **237 unit types**, not the 335 assumed in the plan. Tests must not hard-code unit-type counts.
- PyTorch's `cu128` index only has torch up to 2.11.0. `.venvs\buildabot` now runs `torch==2.14.1+cu130` and `torchvision==0.29.1+cu130`, installed from `https://download.pytorch.org/whl/cu130`. Verified: CUDA 13.0 is available, the device is the RTX 4070 SUPER, and bf16 is supported. **GPU pins: torch 2.14.1, torchvision 0.29.1, index cu130.**
- Batch 1 first run on the PC (2026-10-29): `setup_windows.ps1` gave READY for both venvs. The online `selfcheck.py` passed 12/12: adb, 1080x1920, 480 dpi, game 160635001, emulator verified. `pytest tests/pipeline`: 207 passed and 1 failed in royalegym; 226 passed and 1 failed in buildabot. The one failure was an over-strict test that flagged `--auto-queue` inside a comment; it is fixed.
- Both venvs are fully installed and verified:
  - royalegym: torch 2.14.1+cpu, royalesim 0.1.42, numpy 2.5.4, pyarrow 26.0.0, pytest 9.2.0.
  - buildabot: torch 2.14.1+cu130, torchvision 0.29.1+cu130, royalesim 0.1.42, pyarrow 26.0.0, scipy 1.18.1, opencv-python 5.0.0.93, ultralytics 8.4.192, pytest 9.2.0. CUDA still works after ultralytics was installed.

This file is the plan. `docs/SPARRING_RUNBOOK.md` has the exact commands to run on the PC. Status of each code batch is at the bottom.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Simulator | **RoyaleSim**, pinned at `royalesim==0.1.42` (MIT, Rust, on PyPI), instead of this repo's own `sim/` engine | Replaying 2,000 real matches, RoyaleSim predicts the right winner 62.6% of the time and the exact crowns 49.0%. The repo's own engine manages 49.5% and 24.2% (`docs/PROGRESS_2026-10-09.md`). RoyaleSim is also about 5× faster and ships its own card data, so nothing has to be extracted from the game app. |
| Starting policy | **Imitation of pro replays** from IL_Replay: 252k public ladder matches on Hugging Face | The author's best imitation model already plays about level with the real players' results inside the simulator. |
| Model | The repo's entity-token transformer (`sim/model_tokens.py`), extended | It is measured, and it keeps evolution/hero forms, card levels and exact timings. |
| Improving past the pros | **Self-play RL in RoyaleSim**, against a league of past versions, with the real game's ~1 s command delay built in | Imitation only copies pros. RL is the only way to get past them. A pull toward the imitation model keeps play human-like and stops it learning simulator-only tricks. |
| Live perception | **Screen vision** (YOLO plus screen readers), feeding exactly the same observation used in training | It survives game updates and needs no root access. The memory reader in `scripts/gamestate/` only works on one exact game build and is not used. |
| Live mode | **Friendly battles only, started by a human.** The bot never presses Battle, and there is no ladder option. | Fair to other players, and our use case is sparring. |
| Not built | Anti-detection or ban evasion; reading hidden opponent information | Out of scope by design. |
| Takeover | **F8 hands the whole match** to the owner and back. There are no split modes, such as the bot defending while the owner attacks. | The owner's choice (2026-10-30): a single toggle is enough for practice. |
| Opponent cards | **Covering the newer cards is a priority.** The clanmates play many different decks, and most include at least one card released since 2024. | The KataCR dataset (2024) lacks about 10 newer troop cards, plus the evolved or hero forms of 38 units. `scripts/vision/events.py` names an opponent's play from the detector's class for the unit, so a missing class means a misread play. |

## How it fits together

1. **Data.** IL_Replay matches are replayed in RoyaleSim with their recorded evolution/hero forms and card levels. Each decision becomes a training row: what was on the board, the hand, the elixir, and what the opponent has publicly shown.
   - Rows are taken **at decision time**, about 1.2 s before the card lands, which is when a player actually has to decide. They are not taken when the card lands, as the old pipeline did.
   - Rows contain only what the screen can show.
   - Rows are stored on disk and read in chunks, so 16 GB of RAM is enough.
2. **Imitation training** on the GPU: a general model trained on all decks, then fine-tuned on Hog 2.6.
3. **RL fine-tuning** in RoyaleSim. Settings:
   - the measured command delay, and perception noise matching what our screen reader gets wrong;
   - friendly-battle card levels;
   - a league of past checkpoints plus the imitation models as fixed sparring partners;
   - an adaptive pull toward the imitation model.
4. **Evaluation.** Paired matches on held-out decks and seeds, both sides of the board, with confidence intervals. A new model replaces the champion only when it is clearly better. Real friendlies against clanmates are the final check.
5. **Live loop:** screen capture → unit detector → tracker → screen readers (hand, next card, elixir, clock, tower HP, unit HP bars) → the same observation encoder used in training → model → tap through adb. It runs in dry-run mode by default and stops when a `STOP` file is created.

## Results so far

**First model, `live_big_hog`** (2026-10-29, owner's PC, `scripts/first_model.ps1`):

| Item | Value |
|---|---|
| Rows | `--live-compat`: 13,659 Hog 2.6 matches plus 20,000 others, 135 shards. 3 matches were skipped by the orientation check. |
| General model `live_big` | d256, 6 layers, 8 heads, 3 epochs in 33 min on the RTX 4070 Super. Hog card 65.9%, tile-within-1 54.6%. |
| Hog 2.6 fine-tune `live_big_hog` | 3 epochs in 9.4 min. **Hog card 66.8%, tile exact 31.8%, tile-within-1 55.7%**, which matches the author's best (`rs_big_hog`: 66.8 / 31.7 / 55.7). |
| Simulator evaluation, no delay | vs a reactive `live_big` epoch-1 opponent on 300 held-out Hog 2.6 matches: **56.7%** [51.0, 62.2], crown difference +0.07, 38.9 plays per match. The real Hog players in these matches won 49.0%. |
| Simulator evaluation, 21-tick command delay | **46.0%** [40.4, 51.7], crown difference -0.33, 32.9 plays per match. The delay costs about 11 points, in line with the author's measurement. |

The next lever is delay: decision-time rows (E11), which predict the play about 1.2 s before it lands. This first model is good enough to drive the first live friendlies.

**Second model, `dt_big_hog`** (2026-10-30, owner's PC, `scripts/second_model.ps1`, log `reports/second_model_20261030_101529.log`):

| Item | Value |
|---|---|
| Rows | Decision time (`--obs-delay 24 --obs-delay-jitter 3 --opp-lag 1 --visibility public --trust v2`): 13,947,771 rows in 19 min. `trusted_v2` keeps 73.9% (Hog 2.6: 76.1%). 20,714 of 2,096,396 play rows tap a card that is not yet in hand. |
| General model `dt_big` | 3 epochs in 32 min, on 3,316,529 training rows. Hog card 62.5%, tile-within-1 52.4%. |
| Hog 2.6 fine-tune `dt_big_hog` | 3 epochs in 9.2 min. Hog card 63.3%, tile exact 30.6%, tile-within-1 54.0%. |
| A: `live_big_hog`, 21-tick delay (re-measured) | 46.0% [40.4, 51.7], crown difference -0.34, 33.3 plays per match. Same as before. |
| B: `dt_big_hog`, 21-tick delay | **29.3%** [24.5, 34.7], crown difference -1.02, 28.7 plays per match. |
| C: `dt_big_hog`, no delay | 21.0% [16.8, 26.0], crown difference -1.25, 30.4 plays per match. |

**B is 17 points below A, far outside the noise.** That is the opposite of what decision-time rows were meant to do. `scripts/third_model.ps1` uses the same decision-time rows, so it waits for the diagnosis. The live runner keeps `live_big_hog`.

**Diagnosis so far** (2026-10-30, by reading the code; the PC checks are `scripts/dt_diagnose.ps1`, runbook section 19, "Why the second model lost"):
- **Leading cause (likely, about 60%): wait labels while the bot's own play is in flight.** A decision-time wait row only needs no tap in the next 10 ticks (`tools/rs_rows.py`, the E11 wait rule), so the grid points between a tap and the tick its play runs are kept and labelled "wait". There the re-drive still shows the tapped card in hand, the bar unspent and no unit: the tap's state a moment later. The legacy model does not read `own_pending`, so it learns that "late" looks like "already played". Its gate turns lazy (28.7 plays against 33.3 at the delay, 30.4 against 38.9 without), it cannot learn quick follow-ups, and it misses defences (crown difference -1.02 against -0.34). Landing-time rows have about half as many look-alike waits, and they come before the play label, not after it. The deficit is about 26 points against each model's own best timing, in B against A and in C against the first model without a delay alike. Schema v2 reads `own_pending`, so the third model should not have it.
- **Refuted: the evaluation shows something else after a tap.** Under a command delay RoyaleSim keeps the card in hand and the bar unspent until the play runs (its `set_command_delay_ticks` documentation, measured on the client), exactly what the undelayed re-drive shows at a decision-time row. If anything, the first model is the one that meets states it never trained on.
- **Refuted for the board: the recorded tick is the tap.** `rows_for` runs every play of both sides on the recorded clock, so a different meaning shifts the whole battle rigidly and keeps every relative timing. Only the elixir and the clock would move, by about 21 ticks of regeneration. Whether they do is the `headroom` check (`tools/rs_dt_diag.py`); IL_Replay's "execution tick" is marked as an assumption in the code until it reports. The fidelity grid's exec / tap axis cannot settle this (Phase 2).
- **Small: trust v2's 18% smaller, late-game-thin training set, the 24-tick lead against the 21-tick delay, the 1% of plays tapping a card not yet in hand, public visibility, opponent-memory dating.** None of these changes with the evaluation delay, so none of them explains C being below B; together a few points at most.

What not to do: do not add "pay and cycle the hand when sent" to `rs_eval` or the rows (RoyaleSim already matches the client, and that would create the skew), and do not retune `--obs-delay`, the wait bias, the decision cadence or the dt/20 wait exponent; they are correct for both models.

Gate for the third model: the `ablation` check (the Hog fine-tune again with `sim/train_tokens.py --drop-pending-waits`) or the `waitbias` check lifts `dt_big_hog` to about 40% or more at the 21-tick delay, and `headroom` shows execution ticks. Then `scripts/third_model.ps1` runs, first with `-Others 20000`. Its league now uses `live_big_hog` as the champion and `live_big` epoch 1 as the anchor, and its evaluation adds no-delay, 24-tick and live-rule runs.

## Phases

| # | Phase | Where | Done when | Compute |
|---|---|---|---|---|
| 0 | Write the code and tests | cloud (this branch) | All batches below are merged, with tests that skip cleanly when data, GPU or device are missing | none |
| 1 | PC setup and smoke tests | PC | `scripts/selfcheck.py` passes; `pytest tests/pipeline` passes; IL_Replay downloaded and hash-checked | 1-2 h, mostly downloads |
| 2 | Engine re-baseline | PC | The fidelity grid picks the RoyaleSim card table and settings with winner agreement ≥ 62.6% | about 1 h CPU |
| 3 | Reproduce the author's baseline | PC | Hog card top-1 ≥ 65% and tile-within-1 ≥ 53%. The reactive screen is near the author's numbers. | about half a day |
| 4 | Decision-time imitation model on vision classes | PC | Beats the author's model under realistic delay and noise, over 1,000 paired games | 1-2 days |
| 5 | Screen perception for the current client | PC | Detector precision and recall ≥ 0.90 on our deck and common cards; hand ≥ 99%; capture-to-decision p95 ≤ 100 ms | 6-10 h GPU per detector run, plus labelling time |
| 6 | Live friendlies, imitation model | PC | 20 friendlies in a row with no fatal fault; real delay measured | real time |
| 7 | RL fine-tuning | PC | Passes the promotion gate in the sim and does no worse than the imitation model in real friendlies | 2-7 days unattended; MuMu not needed |
| 8 | More decks, upkeep after game updates | PC | Each deck passes the same gates | per deck: hours to days |

16 GB RAM notes:
- Close MuMu and the browser during training.
- Training reads rows from disk in chunks instead of loading everything.
- Worker counts are capped by available memory.
- Leave the Windows pagefile on "system managed".

## Code batches

Each change has an ID. Full specs live in the session's plan; one-line summaries are below.

**Batch 1: foundation** (done; see `docs/SPARRING_RUNBOOK.md` for the PC steps)
- P01: machine config for adb path, serial, venvs and data folders, via environment variables or `machine.json`.
- P02: remove the original author's hard-coded paths and ports.
- P03: friendly-only by construction.
- P04: pinned requirements, `scripts/setup_windows.ps1`, `scripts/selfcheck.py`.
- P05: new test layout in `tests/pipeline` with markers that skip cleanly.
- V01: arena geometry and tap mapping.
- V02: vision frame types and recorder.
- V03: unit tracker with label and team voting.
- V08: screen capture sources.
- L01: statistics and ratings.

**Batch 2: engine and training core** (done; PC steps in `docs/SPARRING_RUNBOOK.md` sections 13-14)
- E01: a single RoyaleSim factory with a fingerprint.
- E02: card map, including 3 missing cards.
- E04: refuse to load a checkpoint into the wrong engine.
- E05: training-script fixes: GPU guard, per-epoch checkpoints, correct metrics.
- E06: spawn and HP tables.
- E07: checked board orientation.
- E13: opponent elixir counter.
- R03: league pool and deck sampling.

**Batch 3: live path first** (code done 2026-10-30, not yet run on the PC; PC steps in `docs/SPARRING_RUNBOOK.md` sections 15-18)
- E08: opponent memory updated only when a play lands (`tools/rs_memory.py`). Wired into the row builder and the evaluator (`--mem-lag`).
- E16: one shared class list for vision and the model (`data/vision/classes.json`). Wired into the row builder (`rs_rows --units vision`).
- V04: opponent-play and own-landing detection (`scripts/vision/events.py`).
- V05: screen-to-simulator state converter (`tools/public_state.py`).
- V06: shared decision function (`tools/rs_decide.py`); `rs_eval --decide live` uses it.
- V07: screen readers for clock, elixir, tower HP, hand and HP bars, calibrated on the owner's recorded friendly.
- V09: detector wrapper and dataset prep that keep the side column. Training pins the weights by SHA-256.
- V10: `scripts/vision_play.py`, the friendly-only live runner. Dry run by default; `--tap` only while the owner watches; STOP file; `emulator.verify` before the adb shell opens; no tap once the end screen shows.
- V11 (partial): the latency probe (`tools/latency_probe.py`, the decision-to-board delay D) and offline perception replay with precision and recall (`tools/vision_replay.py`). The noise-profile fit waits for E17's format (batch 5).
- V12: takeover toggle (F8, an always-on-top control window, a MANUAL file), wired into the runner. The runner starts in MANUAL.

**Batch 4: training-row correctness and evaluation** (code done 2026-10-30, not yet run on the PC)
- E03: fidelity grid (`tools/rs_fidelity_grid.py`) and robust re-drive.
- E09: trust cut when the simulation diverges from the recording (`--trust v2`).
- E10: replay hero and champion ability presses (`--presses on`). Hold off on using it in training until the model has an ability head (E15).
- E11: decision-time rows (`--obs-delay 24 --obs-delay-jitter 3 --opp-lag 1`).
- E12: hide invisible units (`--visibility public`).
- Moved forward from batches 5 and 6:
  - E17: the perception-noise model core;
  - L02: the general match runner (`tools/rs_match.py`);
  - L03: the league evaluation and promotion gate (`tools/rs_league_eval.py`).
- Next on the PC: `scripts/second_model.ps1` (runbook section 19) trains a decision-time model and compares it fairly with the first one at the 21-tick delay.

**Batch 5: model and data** (code done 2026-10-30, not yet run on the PC)
- E14: observation v2 (`tools/rs_schema.py`, `rs_rows --schema v2`): opponent elixir counted from seen plays, spells in flight, own pending plays, evolution state per hand slot, ability buttons, form and burrow unit features.
- E15: TokenPolicy v2: token groups, ability-button gate, critic head.
- E18: full-corpus path:
  - `rs_rows --all-matches --wait-keep-at-build`;
  - `tools/rs_rows_compact.py` to per-column `.npy`;
  - `train_tokens --memmap --mirror`.
- E17 hooks: `--noise` in rows and `rs_eval`.
- V11 rest: `tools/fit_noise_profile.py` writes `data/noise/measured.json`.
- V06 for v2: presses, own-pending and the critic value in `rs_decide`. The live runner picks legacy or v2 by checkpoint.
- `rs_selfplay`: engine settings carried to workers, visibility from the checkpoint.
- Next on the PC: `scripts/third_model.ps1` (runbook section 20), about 7-12 h.

**Batch 6: RL** (code done 2026-10-30, not yet run on the PC; PC steps in `docs/SPARRING_RUNBOOK.md` section 21)
- `rs_match` and `rs_league_eval` play v2 checkpoints. This covers ability buttons, own pending plays, and the opponent elixir count without refused plays. `--presses` is a switch.
- R01: vectorised rollout workers (`tools/rl/rollout.py`).
- R02: PPO (`tools/rl/ppo.py`). It has GAE (γ 0.997, λ 0.95), potential shaping, and an adaptive KL pull toward the imitation model.
- R04: the RL driver (`tools/rl/train_loop.py`, `scripts/rl_train.ps1`). It covers collection, PPO, league updates, the promotion gate, resume, and a bench to run first.
- R05: an optional search teacher for distillation (`tools/rl/teacher.py`, `ppo --teacher-coef`). It is off by default and not yet wired into the rollout workers, so `--teacher-coef` above 0 stops with exit code 2.
- Open from the reviews:
  - Mirror battles record both seats, so about two thirds of the PPO rows come from decks the bot never plays live. Decide whether to record only the Hog 2.6 seat or keep the mirror share small.
  - The tolerances in the PPO and teacher tests have only been checked by hand. `tests/pipeline/test_train_loop.py` runs first on the PC.

**Designs in progress (2026-10-30)**
- `docs/LIVE_SEARCH_PLAN.md`: search during live play. The bot rebuilds the board in RoyaleSim from what the screen shows (`Battle.reset` takes the clock, elixir, tower HP and spawned units with HP), tries the best candidate plays, and plays the winner. It is measured in the simulator first.
- `docs/NEW_CARDS_PLAN.md`: detector coverage for cards, evolutions and heroes newer than the KataCR dataset. The owner records friendlies in which their main account plays them against Yodle2.
- `docs/SEASON_UPDATES.md`: one routine after each game update, covering the simulator pin, the regenerated data, model retraining and the opponent deck pool.

## First-match milestone (bot plays a friendly against you)

1. **Code:** batches 2 and 3 land on the branch (done).
2. **First model, on the PC, about half a day** (done 2026-10-29; see Results so far). Build `--live-compat` rows, where each unit is identified by the card that created it, which is what screen vision can supply. Train the author's token model on them and fine-tune on Hog 2.6.
3. **Detector, on the PC, overnight (6-10 h GPU).** Train it on the public KataCR detection dataset (`vendor/CR-Detection-Dataset`). Our own friendlies fill gaps later. Runbook section 15.
4. **One calibration friendly.** Check the screen geometry and capture the HUD templates (digits, hand-card art). Runbook section 16.
5. **Dry-run friendly:** the bot only watches and logs. Then a friendly where it plays, with the takeover toggle available. Runbook sections 17 and 18.

This first version is deliberately simple. It uses landing-time rows and no RL, so expect it to be clearly weaker than the final bot. Batches 4-6 then make it stronger.

## Main risks

- **RoyaleSim is about 63% faithful, not 100%.** RL can learn simulator-only tricks. Guards: the pull toward imitation, gates under delay and noise, and real-friendly A/B tests.
- **Our client may be newer than RoyaleSim's newest card table** (16.402.22, 2026-10-19). Re-run the fidelity grid after every game update.
- **Perception.** The public detection dataset (KataCR) is from 2024 and lacks newer cards, evolutions and heroes. Expect some labelling work on our own friendlies.
- **Supercell's Terms of Service prohibit automation, friendlies included.** Use a dedicated bot account.
- **Top players find patterns fast.** Guards: sampled play, rotating checkpoints, and reviewing every loss.
- **This code was written without being run** (the cloud sandbox cannot execute it). Expect some fixes when the tests first run on the PC.
