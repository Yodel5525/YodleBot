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

**Batch 2: engine and training core** (in progress)
- E01: a single RoyaleSim factory with a fingerprint.
- E02: card map, including 3 missing cards.
- E04: refuse to load a checkpoint into the wrong engine.
- E05: training-script fixes: GPU guard, per-epoch checkpoints, correct metrics.
- E06: spawn and HP tables.
- E07: checked board orientation.
- E13: opponent elixir counter.
- R03: league pool and deck sampling.

**Batch 3: training-row correctness**
- E03: fidelity grid.
- E08: opponent memory updated only when a play lands.
- E09: trust cut when the simulation diverges from the recording.
- E10: replay hero and champion ability presses.
- E11: decision-time rows.
- E12: hide invisible units.
- V07: screen readers for clock, elixir, tower HP, hand and HP bars.

**Batch 4: model and data**
- E14: observation v2: spells in flight, opponent elixir, pending plays, abilities.
- E15: model v2 with an ability head and a critic.
- E16: one shared class list for vision and the model.
- E17: perception-noise model.
- E18: full-corpus memory-mapped training with mirroring.
- V09: detector wrapper and dataset prep.

**Batch 5: live loop and evaluation**
- V04: opponent-play detection.
- V05: screen-to-simulator state converter.
- V06: shared decision function.
- V10: `scripts/vision_play.py`, the friendly-only live runner.
- V12: takeover toggle. Press F8, click the control-window button, or create a `MANUAL` file to take control mid-match; do the same again to hand back. While you play, the bot keeps watching and shows what it would do.
- V11: latency probe and noise fitting.
- L02: general match runner.
- L03: league evaluation and promotion gate.

**Batch 6: RL**
- R01: vectorised rollouts.
- R02: PPO with GAE, reward shaping and an adaptive KL pull.
- R04: training driver with gating and resume.
- R05: optional search teacher.

## Main risks

- **RoyaleSim is about 63% faithful, not 100%.** RL can learn simulator-only tricks. Guards: the pull toward imitation, gates under delay and noise, and real-friendly A/B tests.
- **Our client may be newer than RoyaleSim's newest card table** (16.402.22, 2026-10-19). Re-run the fidelity grid after every game update.
- **Perception.** The public detection dataset (KataCR) is from 2024 and lacks newer cards, evolutions and heroes. Expect some labelling work on our own friendlies.
- **Supercell's Terms of Service prohibit automation, friendlies included.** Use a dedicated bot account.
- **Top players find patterns fast.** Guards: sampled play, rotating checkpoints, and reviewing every loss.
- **This code was written without being run** (the cloud sandbox cannot execute it). Expect some fixes when the tests first run on the PC.
