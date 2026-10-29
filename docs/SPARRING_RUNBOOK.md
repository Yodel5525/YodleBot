# Sparring bot: PC runbook

Step-by-step commands for the Windows PC: Windows 11, MuMu Player, RTX 4070 SUPER (12 GB), 16 GB RAM. `docs/SPARRING_PLAN.md` explains the why. This file is the how.

All commands are for **PowerShell**, run from the repo at `C:\hcr`.

This file covers only what exists on the `sparring` branch today. Steps whose scripts are not written yet are listed at the end, under **Later steps**. Each one gets its exact commands here when its batch lands.

## Rules that apply to every step

- **Friendly battles only.** You start every battle yourself, from clan chat or a friend's profile. Nothing in this repo presses Battle or queues ladder or any other matchmade mode, and there is no option that would.
- **The bot account is the second account ("Yodle2").** Never your main account.
- **Do not run the original author's launchers:** `bot.cmd`, `run.ps1`, `scripts\supervisor.py`, `scripts\mumu_overnight_bot.py`, `scripts\mumu_buildabot_native.py`, `scripts\gamestate\live_play.py`. They can no longer queue (change P03). But they belong to his old pipeline: a rule bot, and a memory reader that is out of scope. The sparring bot's live runner comes in a later batch.
- Nothing here starts or restarts MuMu. You start the emulator yourself.

Shorthand used below. Paste it into each new PowerShell window:

```powershell
cd C:\hcr
$RG = 'C:\hcr\.venvs\royalegym\Scripts\python.exe'   # engine venv: CPU torch, royalesim
$BB = 'C:\hcr\.venvs\buildabot\Scripts\python.exe'   # GPU venv: CUDA torch, royalesim, vision
```

## Status on this PC (2026-10-29)

| Step | State |
|---|---|
| 1-4: driver, tools, resources, clone | done; the fork is at `C:\hcr` on `sparring`, with a sparse checkout |
| 6: venvs | both exist. `royalegym` has royalesim 0.1.42, numpy 2.5.4, pyarrow 26.0.0. `buildabot` has torch 2.14.1+cu130 and torchvision 0.29.1+cu130. Run `setup_windows.ps1` once anyway: it keeps what is right and adds the rest. |
| 11: IL_Replay | done; 104 files, 1,888 MB, hash-checked |
| 5, 7-10, 12 | to do |

## 1. GPU driver

Install the latest NVIDIA Game Ready or Studio driver. Then check it:

```powershell
nvidia-smi
```

It must list `NVIDIA GeForce RTX 4070 SUPER`, with `CUDA Version` 13.0 or higher. The GPU venv uses PyTorch's CUDA 13.0 build (`cu130`); PyTorch's cu128 index stops at torch 2.11.0.

## 2. Base tools (once)

```powershell
winget install -e --id Git.Git
winget install -e --id Python.Python.3.12
```

Then open a **new** terminal and run:

```powershell
py -0p                                   # must list -V:3.12
git config --global core.longpaths true
```

## 3. Disk and memory (16 GB RAM)

**Disk.** Keep at least 100 GB free on an SSD:
- training rows: 20-35 GB;
- the two venvs: about 8 GB;
- IL_Replay: 1.9 GB;
- YOLO data and runs: about 10 GB.

**Pagefile.** Leave it on "System managed", with at least 32 GB free on that drive. Every worker process loads torch, and Windows reserves pagefile space for each one.

**16 GB RAM.** The original author's PC ran out of memory with less going on, so:
- Run one heavy job at a time: building rows, evaluation, training, or live play.
- **Close MuMu and the browser before training.** Training and live play never run at the same time.
- Start engine jobs (rows, evaluation, RL workers) with `--workers 6`. Watch Task Manager > Performance > Memory during the first minutes:
  - if "Available" stays above 4 GB, you can try 8;
  - with MuMu open, use 4.

  These are starting points. The scripts that cap workers by free memory come in a later batch.
- MuMu's instance has 1 core and 1 GB today. Raise it to 2-4 cores and 3 GB before live play (MuMu Multi-Instance > Settings), and close it again for training.

## 4. Get the code (done on this PC)

This is a partial, sparse clone of the fork. It skips `data/vod` (1.2 GB of video dumps) and `demo.mp4`, which nothing in the pipeline reads:

```powershell
git clone --filter=blob:none --no-checkout https://github.com/Yodel5525/YodleBot.git C:\hcr
cd C:\hcr
git sparse-checkout set --no-cone "/*" "!/data/vod/" "!/demo.mp4"
git checkout sparring
```

To get later batches: `git pull` (on `sparring`).

## 5. Machine settings: adb, serial, instance

Find MuMu's own adb and manager:

```powershell
Get-ChildItem 'C:\Program Files\Netease' -Recurse -Include MuMuManager.exe,adb.exe -ErrorAction SilentlyContinue | ForEach-Object FullName
```

On this PC, adb is `C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe`, the serial is `127.0.0.1:16384` (instance 0), and the instance is named `Android Device`. Use only MuMu's adb: two adb versions keep restarting each other's server.

Save the settings once, either way:

**Option A: environment variables.** Open a new terminal afterwards.

```powershell
setx HASTYCR_ADB "C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe"
setx HASTYCR_SERIAL 127.0.0.1:16384
setx HASTYCR_INSTANCE "Android Device"
```

**Option B: a machine.json file.** Copy `machine.example.json` to `machine.json` (it is gitignored) and fill in `"adb"`. Use forward slashes, or double every backslash.

Check what the code will use:

```powershell
py -3.12 src\hastycr\machine.py          # before step 6
& $BB -m hastycr.machine                 # after step 6
```

Every value shows where it came from (env, machine.json or default), plus any warnings.

## 6. Create the venvs: scripts\setup_windows.ps1

```powershell
cd C:\hcr
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

It builds two venvs:
- `.venvs\royalegym` from `requirements\engine.txt`: CPU torch, royalesim;
- `.venvs\buildabot` from `requirements\gpu.txt`: CUDA torch and torchvision, royalesim, ultralytics, opencv-python, pillow, scipy, matplotlib.

The pins are the versions already installed and working in this PC's venvs (2026-10-29), so on this PC the script mostly re-checks.

For each one it:
1. installs torch first, from PyTorch's index (`https://download.pytorch.org/whl/cpu` for royalegym, `https://download.pytorch.org/whl/cu130` for buildabot);
2. installs the pinned packages;
3. re-installs the CUDA build if a package replaced it;
4. installs this repo (`pip install --no-deps -e .`);
5. checks torch with a one-liner;
6. writes `reports\freeze-royalegym.txt` / `reports\freeze-buildabot.txt`;
7. runs `scripts\selfcheck.py --offline`.

It is safe to run again: anything already right is kept. The first run downloads about 3-4 GB.

Options:

| Option | Use |
|---|---|
| `-Only royalegym` / `-Only buildabot` | one venv only |
| `-GpuIndexUrl <url>` | another CUDA build of torch. The default is cu130, which is verified on this PC. The index must offer the torch / torchvision pins in `requirements\gpu.txt`. |
| `-Recreate` | delete the venv(s) and build them from scratch |
| `-Reference` | also build `.venvs\rs022` (royalesim 0.1.22), used only to reproduce the author's re-drive numbers (Phase 2) |
| `-SkipSelfcheck` | skip the final offline selfcheck |

If it stops with `ERROR:`:

| Message mentions | Fix |
|---|---|
| `py` is not on PATH | step 2, then a new terminal |
| `CUDA (or bf16) does not work` | step 1 (driver). Or `-GpuIndexUrl` for a CUDA version your driver supports, with matching pins in `requirements\gpu.txt`. |
| `torch is still not ...` | the index does not have that torch for Python 3.12 on Windows. Open `<index>/torch/` in a browser to see what it has. |
| `not 3.12` / `half-made venv` | re-run with `-Recreate` |

## 7. MuMu settings (once)

In MuMu's settings:
- **Display:** Phone, portrait, **1080x1920**, **DPI 480**, 60 FPS.
- **Root:** OFF.
- **Instances:** one.

Restart the instance yourself. Then check it:

```powershell
$ADB = 'C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe'
$S = '127.0.0.1:16384'
& $ADB connect $S
& $ADB -s $S shell wm size        # Physical size: 1080x1920, and no "Override size" line
& $ADB -s $S shell wm density     # Physical density: 480
```

If an `Override size` line shows up (the author's `run.ps1` used to leave 540x960 behind), remove it with `& $ADB -s $S shell wm size reset`.

## 8. Selfcheck

With MuMu running and Clash Royale installed:

```powershell
& $BB scripts\selfcheck.py
& $RG scripts\selfcheck.py --offline
```

Both must end with `READY`, which means exit code 0 (`$LASTEXITCODE`). Each line is PASS, WARN, FAIL, SKIP or INFO:

| Check | Passes when |
|---|---|
| python | Python 3.12 |
| friendly | `FRIENDLY_ONLY` is on and queueing raises |
| machine | `machine.json` (if any) parses; warnings are shown |
| role | `train` for buildabot, `engine` for royalegym (from the interpreter's path; `--role` overrides it) |
| torch | **train:** a CUDA build, CUDA available, bf16, a CUDA torchvision, and a bf16 matmul on the GPU. **engine:** the CPU build. A CUDA build there is a WARN. |
| packages | every pin in the venv's requirements file is installed at that version |
| royalesim | 0.1.42. It also prints the card table, card level, tower level and unit-type count. This PC printed `cards-15535.1`, 11, 11 and 237. |
| hastycr | the repo is installed in the venv |
| adb | the serial answers `device` (`adb connect` is tried once) |
| screen | 1080x1920 portrait with no override |
| density | 480 dpi |
| game | Clash Royale's versionCode. A build other than `160635001` is a WARN, because the game has updated. |
| emulator | `scripts\emulator.py` confirms that the serial is the Clash Royale instance |

`--offline` skips the last five. `--json` prints everything as JSON, ready to paste back to the cloud session. `--strict` treats warnings as failures. `--adb` / `--serial` override the machine settings for one run.

## 9. Account and deck

- Log in to MuMu's Clash Royale with the bot account ("Yodle2"), in the clan.
- Set the active deck to **Hog 2.6**: Cannon, Fireball, Hog Rider, Ice Golem, Ice Spirit, Musketeer, Skeletons, The Log.
- Write down which cards sit in the evolution and hero slots.
- Friendly battles play every card at level 11, so training and evaluation use level 11.

## 10. Screenshots for the screen geometry (no taps)

In a friendly battle you started yourself:

```powershell
$ADB = 'C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe'   # as in step 7
$S = '127.0.0.1:16384'
New-Item -ItemType Directory -Force C:\hcr\tmp | Out-Null
& $ADB -s $S shell screencap -p /sdcard/hcr_snap.png
& $ADB -s $S pull /sdcard/hcr_snap.png C:\hcr\tmp\snap_battle.png
& $BB -m scripts.vision.geometry overlay --image tmp\snap_battle.png --out tmp\snap_battle_grid.png
```

Repeat on the lobby and on the end screen (`snap_lobby.png`, `snap_end.png`). Open `tmp\snap_battle_grid.png` to check that the drawn tile grid sits on the arena. Send the screenshots back to the cloud session.

Do not use `exec-out screencap -p > file.png`: in Windows PowerShell, `>` re-encodes the bytes and corrupts the PNG.

## 11. IL_Replay (done on this PC)

```powershell
& $RG tools\il_replay_download.py
```

It writes 104 parquet files, about 1.9 GB, SHA-256 checked, to `data\il_replay`. Running it again only re-verifies, and resumes anything partial. `--parts 2` is a quick smoke test.

## 12. Tests

```powershell
& $RG -m pytest tests/pipeline -q
& $BB -m pytest tests/pipeline -q
```

Tests skip cleanly when what they need is missing:
- `engine` needs royalesim;
- `gpu` needs CUDA;
- `il_replay` needs `data\il_replay`;
- `device` needs `$env:HASTYCR_DEVICE_TESTS = '1'` and a running MuMu.

This code was written in a sandbox that could not run it, so the first run may show a few failures. Paste the output back to the cloud session.

The legacy suite (`pytest tests`) needs `tmp/gamedata`, has known failures, and is not needed.

## 13. Engine checks (quick, MuMu can stay closed)

Run these after each `git pull` that changes the engine code:

```powershell
& $RG tools\rs_engine.py --probe
```

This must report the card table `cards-160402017-20261019.1` (the 16.402.22 / 19 October balance update), card level 11 and tower level 11. `--engine-config legacy` shows the engine's built-in table instead, for comparison.

```powershell
& $RG tools\rs_cards.py --coverage --parts 4
& $RG tools\rs_orient.py --parts 4
& $RG tools\rs_spawn_table.py
```

- `rs_cards --coverage`: how many IL_Replay matches have every card in the simulator's card map.
- `rs_orient`: how many matches pass the board-orientation check. A large "skipped" share means the check is too strict; send the output back.
- `rs_spawn_table`: writes `data\rs_rows\spawn_primary.json`, the table of which units each card creates. Evaluation needs it.

## 14. First model (Phase 3, live-compatible)

**Close MuMu and the browser first.** These jobs use most of the RAM.

The first model is trained on `--live-compat` rows. In those rows, units are identified by the card that created them, which is what screen vision can supply. It is the author's design, kept as the quickest route to a first live friendly. Later batches replace it with decision-time rows and RL.

**Easiest: one script runs all four steps unattended.** It stops at the first failure and writes everything to `reports\first_model_<time>.log`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\first_model.ps1
```

`-Workers 6` uses less RAM. `-From hog` (or `general`, `eval`) restarts at a later step. Send back the log when it finishes. The steps it runs are below, if you prefer to run them one at a time.

**1. Build the training rows.** This takes about 30-60 minutes. It uses every Hog 2.6 match plus 20,000 others:

```powershell
& $RG tools\rs_rows.py --live-compat --others 20000 --workers 8 --out data\rs_rows_live
```

If Task Manager shows less than 2 GB of RAM available, stop it with Ctrl+C and re-run with `--workers 6`.

**2. Train a general model on all decks.** This takes a few hours on the GPU. It writes `runs\tokens\live_big\` with `ep01.pt`, `ep02.pt`, `ep03.pt`, `best.pt` and `last.pt`:

```powershell
& $BB -m sim.train_tokens --name live_big --data data\rs_rows_live --wait-keep 0.25 --epochs 3 --d 256 --layers 6 --heads 8
```

**3. Fine-tune it on Hog 2.6.** This takes about an hour:

```powershell
& $BB -m sim.train_tokens --name live_big_hog --data data\rs_rows_live --wait-keep 0.25 --epochs 3 --d 256 --layers 6 --heads 8 --finetune-hog --init runs\tokens\live_big\best.pt
```

Each epoch prints `val card ... near ...` and `hog card ... near ...`. The author's comparable model reached about 65% Hog card accuracy and 54% tile-within-1.

**4. Evaluate it in the simulator.** Each run is 300 matches on held-out games against the general model's first epoch. Run it once without the client's command delay and once with it:

```powershell
& $RG tools\rs_eval.py --ckpt runs\tokens\live_big_hog\best.pt --opponent-ckpt runs\tokens\live_big\ep01.pt --matches 300 --workers 8
& $RG tools\rs_eval.py --ckpt runs\tokens\live_big_hog\best.pt --opponent-ckpt runs\tokens\live_big\ep01.pt --matches 300 --workers 8 --delay 21
```

Send back the last lines of each step: the row counts, the per-epoch `val` lines, and both win rates with their intervals.

## If something breaks

Paste these back to the cloud session:
- the full error;
- `& $BB scripts\selfcheck.py --json`;
- `reports\freeze-royalegym.txt` and `reports\freeze-buildabot.txt`.

## Later steps (coming in a later batch)

These need code that is not on the branch yet. Each one gets exact commands here when its batch lands. The IDs refer to `docs/SPARRING_PLAN.md`.

| Phase | What you will run | Waits for |
|---|---|---|
| 2. Engine re-baseline | spawn tables for both card tables; then a fidelity grid over 2,000 validation matches (`tools/rs_fidelity_grid.py`), which picks the RoyaleSim card table and writes `data/engine.json` | E01, E03, E06 |
| 4. Decision-time models | v2 rows on the vision class list, with delay and noise; memory-mapped training; the league evaluation | E11-E18, L01-L03 |
| 5. Screen perception | detector dataset and training; HUD readers (clock, elixir, tower HP, hand) | V07, V09 |
| 6. Live friendlies | `scripts/vision_play.py`: dry run by default; `--tap` only while you watch; stops when `C:\hcr\STOP` exists. Plus a latency probe to measure the real delay. | V04-V06, V10, V11 |
| 7. RL fine-tuning | the RL driver with gating and resume (`tools/rl/train_loop.py`); runs for days with MuMu closed | R01-R05 |
