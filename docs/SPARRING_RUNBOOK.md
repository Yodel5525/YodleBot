# Sparring bot: PC runbook

Step-by-step commands for the Windows PC: Windows 11, MuMu Player, RTX 4070 SUPER (12 GB), 16 GB RAM. `docs/SPARRING_PLAN.md` explains the why. This file is the how.

All commands are for **PowerShell**, run from the repo at `C:\hcr`.

This file covers only what exists on the `sparring` branch today. Steps whose scripts are not written yet are listed at the end, under **Later steps**. Each one gets its exact commands here when its batch lands.

## Rules that apply to every step

- **Friendly battles only.** You start every battle yourself, from clan chat or a friend's profile. Nothing in this repo presses Battle or queues ladder or any other matchmade mode, and there is no option that would.
- **The bot account is the second account ("Yodle2").** Never your main account.
- **Do not run the original author's launchers:** `bot.cmd`, `run.ps1`, `scripts\supervisor.py`, `scripts\mumu_overnight_bot.py`, `scripts\mumu_buildabot_native.py`, `scripts\gamestate\live_play.py`. They can no longer queue (change P03). But they belong to his old pipeline: a rule bot, and a memory reader that is out of scope. The sparring bot's live runner is `scripts\vision_play.py` (sections 17 and 18).
- Nothing here starts or restarts MuMu. You start the emulator yourself.

Shorthand used below. Paste it into each new PowerShell window:

```powershell
cd C:\hcr
$RG = 'C:\hcr\.venvs\royalegym\Scripts\python.exe'   # engine venv: CPU torch, royalesim
$BB = 'C:\hcr\.venvs\buildabot\Scripts\python.exe'   # GPU venv: CUDA torch, royalesim, vision
```

## Status on this PC (2026-10-30)

| Step | State |
|---|---|
| 1-4: driver, tools, resources, clone | done; the fork is at `C:\hcr` on `sparring`, with a sparse checkout |
| 6: venvs | both exist. `royalegym` has royalesim 0.1.42, numpy 2.5.4, pyarrow 26.0.0. `buildabot` has torch 2.14.1+cu130 and torchvision 0.29.1+cu130. Run `setup_windows.ps1` once anyway: it keeps what is right and adds the rest. |
| 5, 7, 8, 12 | done in the batch 1 first run: setup READY, selfcheck 12/12, tests passing after one fix |
| 9: account and deck | done: Yodle2, Hog 2.6, all base cards |
| 11: IL_Replay | done; 104 files, 1,888 MB, hash-checked |
| 14: first model | done: `runs\tokens\live_big_hog\best.pt` (Hog card 66.8%, tile-within-1 55.7%) |
| 10, 13 | not reported back. Section 16 covers what step 10 was for. |
| 15-18: detector, calibration, dry run, first friendly with taps | to do |
| 19: second model (decision-time rows) | done 2026-10-30: `dt_big_hog` 29.3% against `live_big_hog`'s 46.0% at a 21-tick delay. The checks of "Why the second model lost" (`scripts\dt_diagnose.ps1`) are to do |
| 20: third model (v2 rows, the whole corpus) | to do, after the section 19 checks |
| 21: RL fine-tuning (bench, then days of training) | to do, after 20 |

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

## 15. Detector (overnight, MuMu closed)

The bot sees units on the board with a YOLO detector. It is trained on the public KataCR detection dataset: about 640 MB of labelled frames from 2024. Our own friendlies fill its gaps later.

**Close MuMu and the browser first.** Training uses the GPU and most of the RAM.

**1. Get the dataset (once).**

```powershell
cd C:\hcr
git clone --depth 1 https://github.com/wty-yy/Clash-Royale-Detection-Dataset.git vendor\CR-Detection-Dataset
Test-Path vendor\CR-Detection-Dataset\images\part2\ClashRoyale_detection.yaml    # must print True
```

`vendor\` is gitignored. The dataset can also live on another drive: then add `--source D:\datasets\CR-Detection-Dataset` to the next command.

**2. Convert it to the YOLO layout.** This takes a few minutes and writes `tmp\yolo\`:

```powershell
& $BB scripts\prepare_yolo_dataset.py
```

The images are hard-linked, not copied, so it needs almost no extra disk. Each class is a unit label plus a side (ally or enemy). It should print 155 labels x 2 sides = 310 classes, then any names the class list does not know.

**3. Train.** This runs overnight, about 6-10 hours on the RTX 4070 SUPER: yolov8s, 640 px, batch 32, up to 150 epochs. It stops early once the score has not improved for 30 epochs.

```powershell
& $BB scripts\train_detector.py
```

- The first run downloads the starting weights, `yolov8s.pt` (about 22 MB).
- Keep the default 150 epochs. Below 115 epochs ultralytics' automatic optimizer switches to AdamW at about 3e-5, a setting not tried here (`docs\NEW_CARDS_PLAN.md` section 8).
- If the GPU runs out of memory, add `--batch 16`. If RAM runs low, add `--workers 2`.
- Without CUDA it stops with exit code 2 instead of training on the CPU for days. Then check steps 1 and 6.
- If the PC restarts during the run, resume it:

  ```powershell
  & $BB scripts\train_detector.py --model tmp\yolo\runs\cr_detector\weights\last.pt --resume
  ```

**Where the weights land.** `tmp\yolo\runs\cr_detector\weights\best.pt`, with `best.pt.sha256` next to it and `detector_manifest.json` in the run folder.

When training ends it **pins** best.pt: it writes the path and the SHA-256 to `data\vision\detector_pin.json`. That is how the live runner and the replay tool find the detector:
- by default they load the pinned file, and refuse it if its SHA-256 has changed;
- `--detector <file>` uses another file for one run;
- `& $BB -m scripts.vision.detector pin <file>` pins another file;
- a new training run pins its own best.pt. `--no-pin` keeps the old pin.

**4. Check it.**

```powershell
& $BB -m scripts.vision.detector info
& $BB -m scripts.vision.detector image $I     # $I: one battle frame from a recording (section 16, step 3)
```

`info` must show:
- the pinned weights and their `sha256`;
- `"device_desc": "cuda:0 NVIDIA GeForce RTX 4070 SUPER ..."` and `"half": true`;
- `"side_mode": "joint"`;
- an empty `"unresolved_names"`, and no `NOT in the vision class list` line.

`image` lists each detection with its class, confidence, P(enemy) and tile. Our own units should have P(enemy) below 0.5. Nothing should be detected in the card bar.

Send back:
- the output of `prepare_yolo_dataset`;
- the first and last 30 lines of the training;
- `tmp\yolo\runs\cr_detector\results.csv`;
- the output of `info` and of one `image`.

## 16. Calibration friendly: geometry, HUD and templates

The HUD readers match small crops of the game's own digits and card art, taken from your client. Nothing ships them. Without clock templates the bot cannot read the clock, so it never decides.

You record one friendly, then build everything from the recording. MuMu is only needed for the recording, so this can be done before the detector is trained. If the friendly you recorded on 2026-10-29 is still under `reports\vision\recordings`, you can use it and skip step 2.

**Before the friendly:**
- MuMu instance: 2-4 cores and 3 GB (step 3).
- Make the MuMu window as tall as the screen allows. Frames are captured at the window's size: 747x1328 on this PC.
- Keep the window uncovered. A fully covered window stops updating.

**1. Check the capture.** In the lobby:

```powershell
cd C:\hcr
New-Item -ItemType Directory -Force C:\hcr\tmp | Out-Null
& $BB -m scripts.vision.capture --source auto --seconds 10
& $BB -m scripts.vision.geometry check --size 747x1328
```

The first prints the frame size and rate: on this PC `(747, 1328)` at about 57 fps. Give `check` the size it printed. It must say `ok: ... is 9:16 portrait`.

**2. Record one friendly.** Start the recording in the lobby, then start a friendly yourself and play it to the end:

```powershell
& $BB -m scripts.vision.capture --source auto --seconds 400 --record reports\vision\recordings
```

It prints `recording to ...\capture_<date>_<time>` and keeps an image every 5 frames. It only reads pixels. During the friendly, try to get:
- every card in hand at least once, and greyed out (not enough elixir);
- a frame just after a play, while that slot is empty;
- damage on several towers, so the tower numbers show many digits;
- overtime, if the game gets there (optional).

**3. Pick frames.** Set these once in the PowerShell window:

```powershell
$REC = (Get-ChildItem reports\vision\recordings -Directory | Sort-Object Name | Select-Object -Last 1).FullName
$IMG = Get-ChildItem "$REC\images" -Filter *.jpg | Sort-Object Name
$IMG.Count
explorer "$REC\images"
```

Find a clear mid-battle frame in Explorer and point `$I` at it, for example `$I = "$REC\images\00001200.jpg"`.

**4. Geometry.** Draw the tile grid on that frame:

```powershell
& $BB -m scripts.vision.geometry overlay --image $I --out tmp\grid_check.png
```

Open `tmp\grid_check.png`:
- the white grid must sit on the arena floor: 18 tiles across, 32 down;
- the blue lines mark the river;
- red circles mark the tower centres, the bridge ends and the river's corners at the arena edges;
- green crosses mark the four card slots.

Taps use this geometry, scaled to the device's 1080x1920. If the grid is off by more than about a quarter tile, refit it:

```powershell
& $BB -m scripts.vision.geometry template --size 747x1328 --out tmp\points.json
notepad tmp\points.json      # set each "px" to where that point is on $I; drop points you cannot see
& $BB -m scripts.vision.geometry fit --points tmp\points.json --out data\vision\geometry_pc.json
& $BB -m scripts.vision.geometry overlay --image $I --geometry data\vision\geometry_pc.json --out tmp\grid_check_pc.png
```

Then add `--geometry data\vision\geometry_pc.json` to every `vision_play.py` command in sections 17 and 18.

**5. HUD boxes.**

```powershell
& $BB -m scripts.vision.hud overlay --image $I --out tmp\hud_check.png
```

Open `tmp\hud_check.png`:
- yellow: the clock digits, the "Time left" label, the elixir number and the elixir bar;
- cyan: the tower numbers and bars;
- magenta: the four hand cards and the next card.

Each box must cover its number or card. The defaults were measured on your 2026-10-29 frames, so they should fit. If one is off, write your own copy and fix that box:

```powershell
& $BB -m scripts.vision.hud regions --size 747x1328 --out data\vision\hud_regions_pc.json
notepad data\vision\hud_regions_pc.json     # fix the "px" box [x0, y0, x1, y1] of the region that is off
& $BB -m scripts.vision.hud overlay --image $I --out tmp\hud_check.png
```

The readers use `hud_regions_pc.json` automatically when it exists. Do not edit `hud_regions_1080x1920.json`.

**6. Templates.** The fast route reads the whole recording: `clock-sweep` for the clock, and `harvest` plus `harvest-label` for the elixir count, the tower numbers, the hand and the next card. The one-frame commands further down are the fallback, and they are still how the clock's label is stored.

*a) Clock.* Read the clock on one frame (`$I`), and the sweep counts the seconds from it through the whole recording. Check what it would store with `--dry-run` first:

```powershell
& $BB -m scripts.vision.templates clock-sweep --recording $REC --anchor $I --text 1:48 --dry-run
& $BB -m scripts.vision.templates clock-sweep --recording $REC --anchor $I --text 1:48
```

- It prints the range it labelled (for example `labelled 3:00 down to 0:00`), every stretch it skipped as ambiguous, and each glyph it stored. It stores only the characters `templates list` reports missing; add `--all` to store every character.
- `lost the count` or `count check failed` means it could not count through a stretch (a capture stall, a long stretch without a readable clock, or a second the game skipped), and stopped on that side. The message names the image where it stopped. Read the clock on a frame after that point (set `$J` to it, like `$I`) and add it as a second anchor: `--anchor $I --text 1:48 --anchor $J --text 0:31` (one `--text` per `--anchor`, in the same order).
- It stops at the end of regulation. If the friendly went to overtime, add an overtime frame (`$K`) as the last anchor, plus `--until-overtime`: `--anchor $I --text 1:48 --anchor $K --text 1:47 --until-overtime`.

*b) Elixir, towers, hand and next card.* `harvest` crops every image and groups crops that look alike. It writes a contact sheet with one tile per group: `#id n=size` (a `g` marks greyed cards), then the group's most typical crop, a middle one and its least typical one. Open the sheet, then give each group you can read its value:

```powershell
& $BB -m scripts.vision.templates harvest --recording $REC --what elixir --out tmp\harvest_elixir.png
start tmp\harvest_elixir.png
& $BB -m scripts.vision.templates harvest-label --sheet tmp\harvest_elixir.json --map "0=7,1=3,2=1,3=0"
```

Do the same with `--what tower`, `--what hand` and `--what next`, for example `--map "0=HogRider,1=greyed:HogRider,2=Log,3=empty"` for the hand.

- Digits: one group is one digit, so `10` elixir shows up as two groups, `1` and `0`.
- Cards: the simulator's name; put `greyed:` in front of a greyed card (`5=greyed:Cannon`), and `empty` for an empty slot. A group the sheet marks `g` must be labelled `greyed:...`, and only those. Otherwise `harvest-label` stores nothing and says which group is wrong (`--force` stores it anyway).
- Leave out every group whose three crops are not all the same digit or card. Also leave out the tower groups that show a king's level badge (`11`) instead of hit points. Groups you leave out are not stored.
- Several groups for one digit or card is normal, for example the two tower colours. Label them all.
- If a group mixes two digits, run `harvest` again with a higher `--min-ncc` (the default is 0.9 for digits and 0.8 for cards). `--max-clusters` (default 80) sets how many groups the sheet shows, largest first.

*c) One frame at a time (fallback, and the clock label).* Open the frame, read what it shows, and store it. An example frame: the clock at 1:48, 2 elixir, a hand of Log, Hog Rider (greyed), Skeletons and Fireball (greyed), Ice Spirits next, and tower numbers 3052 and 2101 (theirs) and 3052 and 2731 (ours):

```powershell
& $BB -m scripts.vision.templates digits --image $I --font clock --text 1:48
& $BB -m scripts.vision.templates digits --image $I --font elixir --text 2
& $BB -m scripts.vision.templates digits --image $I --font tower --region tower_enemy_left --text 3052
& $BB -m scripts.vision.templates digits --image $I --font tower --region tower_enemy_right --text 2101
& $BB -m scripts.vision.templates digits --image $I --font tower --region tower_own_right --text 2731
& $BB -m scripts.vision.templates hand --image $I --cards Log,HogRider,Skeletons,Fireball --variants base,greyed,base,greyed --next IceSpirits
& $BB -m scripts.vision.templates label --image $I --label regulation
```

- Card names are the simulator's: `HogRider`, `Musketeer`, `Cannon`, `IceGolemite`, `IceSpirits`, `Skeletons`, `Fireball`, `Log`. Slots go left to right; `-` skips one.
- On a frame right after a play: `& $BB -m scripts.vision.templates empty --image $I --slot 2`, with the number (0-3) of the empty slot.
- In overtime: `& $BB -m scripts.vision.templates label --image $I --label overtime`.
- `found 5 glyphs ... but the text has 4 characters` means the box or the frame is not clean. Try another frame, or crop by hand with `--box x0,y0,x1,y1`.

Run `harvest` or the one-frame commands again (on other frames or groups) until this shows `complete` for all three fonts, and `base, greyed` for every card:

```powershell
& $BB -m scripts.vision.templates list --deck HogRider,Musketeer,Cannon,IceGolemite,IceSpirits,Skeletons,Fireball,Log
```

The templates are stored under `data\vision\templates\`.

**7. Read the recording back.** This sends 40 frames from the middle of the battle through the HUD readers. `--dt 0.09` is the time between them: one image every 5 frames at 57 fps.

```powershell
$RUN = ($IMG | Select-Object -Skip 1000 -First 40).FullName
& $BB -m scripts.vision.hud read --image $RUN --deck HogRider,Musketeer,Cannon,IceGolemite,IceSpirits,Skeletons,Fireball,Log --dt 0.09
& $BB tools\vision_replay.py $REC --no-detector
```

- `hud read` prints, per frame: the tick (20 a second, from 0 at 3:00), the elixir in thousandths, the hand, the next card and the tower HP. Compare a few with the images.
- `vision_replay` runs the whole recording and prints how often the clock, elixir, hand, next card and towers were read.
- Once the detector is trained (section 15), run `vision_replay` again without `--no-detector`. It adds detections and tracks per frame.
- The whole runner (HUD, model, decisions, battle result) on the same recording is section 17 step 2.

Send back:
- `tmp\grid_check.png` and `tmp\hud_check.png`;
- the `templates list` output;
- both outputs of step 7.

## 17. Dry-run friendly (the bot watches, no taps)

`scripts\vision_play.py` is the live runner. It runs the whole chain on the screen: capture, HUD, detector, tracker, the model and the decision. By default it taps nothing and opens no adb connection. It never presses Battle, OK or anything outside the arena, and it never starts or restarts MuMu.

You need:
- the templates complete (section 16);
- `runs\tokens\live_big_hog\best.pt` and the `args.json` next to it (section 14);
- for step 3, also the detector pinned (section 15), and MuMu at 2-4 cores and 3 GB, with its window large and uncovered.

**1. Tests and controls.** After `git pull`:

```powershell
& $RG -m pytest tests/pipeline -q
& $BB -m pytest tests/pipeline -q
& $BB -m scripts.vision.control --seconds 60
```

The last one tries the takeover controls without a battle. It never taps. Press F8 a few times, click the button in the small "Sparring bot" window, and create then delete `C:\hcr\MANUAL`. Each switch prints a line such as `MANUAL -> AUTO (hotkey)`.

**2. Rehearse on the recording first (MuMu not needed).** The runner has never run end to end before. This runs all of it, the setup, the HUD, the model, the decisions, the battle and its result and the log, on the friendly you recorded on 2026-10-29. On a recording it refuses taps and touches neither MuMu nor adb. Its session goes to `reports\vision_play\replays\`, so `latency_probe --last 1` in step 4 still finds the live session.

Run one heavy job at a time (section 3). Run the rehearsal before the detector training starts or after it ends, not during it. Without a pinned detector it runs with `detector NONE`: everything except the detector, the tracker and the events is tested, and the board stays empty for the model.

```powershell
$REC = 'C:\hcr\reports\vision\recordings\capture_20261029_195831'   # or its copy under reports\vision\gold\
Test-Path "$REC\frames.jsonl"                                          # must print True
& $BB scripts\vision_play.py --source $REC --replay-fast --no-window --hotkey none --seed 1
```

`--replay-fast` replays every image once, as fast as the chain runs: a few minutes for the 4,414 images. Check the setup lines:
- `source     replay of ...capture_20261029_195831 (4414 images, as fast as the chain runs); no device, no adb`;
- `taps       off (the source is a recording)`;
- `decide     live; wait_keep 0.25 (wait bias 1.386); would-play every 10 ticks`. A `WARNING: no args.json` there means the trained share of waits is unknown: copy `args.json` next to `best.pt` (`--tap` refuses without it);
- `detector   NONE (...)` until the detector is pinned. Once it is pinned: `... on cuda:0, half True, side joint, 310 classes, 0 unresolved, classes dc897910, imgsz 640`. A `WARNING` line under it means the sides are not joint or some detector names are not in the class list: send it back;
- `hud        clock templates ok`;
- `frame 747x1328` on the first frame. A `WARNING` after it means another frame size than the HUD was calibrated on.

Then, on this recording:
- `battle 1 started (tick 110, mode MANUAL)`, about 8 s into the recording;
- status lines with `| would <card> -> (x, y) @<tick>`: what the bot would play now. In MANUAL it decides on nearly every tick and almost every decision is a wait, so on a wait it works this out at most every 10 ticks;
- `battle 1 over (end screen) at tick 44..` (overtime), then `battle 1: WIN 1-0 (end screen + hud); record 1-0-0 (W-L-D)`;
- `stopped: the frame source ended; 4414 frames, 0 taps, results {'win': 1}, errors -`.

Then run it once in real time (about 6.5 minutes, the length of the recording):

```powershell
& $BB scripts\vision_play.py --source $REC --no-window --hotkey none --seed 1
& $BB tools\latency_probe.py reports\vision_play\replays --last 1
```

The recording kept an image of every 5th captured frame, so the replay has about 11 frames a second, against about 57 live. Its frame rate in the latency probe is not the live figure. Rules that count frames (the hand's 3 votes, the 3 clock readings before the hand and the towers start, the battle start's 3 frames) also take longer here than live. Use this run for the time of each stage per frame (perceive, decide).

Send back the setup lines, the battle, result and `stopped:` lines of both runs, and the latency probe output. `errors` other than `-` on the `stopped:` line, or a traceback, means something in the chain is broken: send the whole output.

**Once the detector is pinned** (section 15), run the fast rehearsal once more before the live friendly. The `detector` line must then name the pin. This is the first run of the detector, the tracker, the events and the board in the model's state on real frames. The `stopped:` line must again end in `errors -`, without `detect_error`. Your plays in the recording are now seen on the board, so the log gets `human_play` records, and each should carry what the bot would have played:

```powershell
& $BB scripts\vision_play.py --source $REC --replay-fast --no-window --hotkey none --seed 1
$S = (Get-ChildItem C:\hcr\reports\vision_play\replays -Directory | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
& $BB -c "import json,sys; h=[r['data'] for r in map(json.loads, open(sys.argv[1], encoding='utf-8')) if r.get('kind')=='human_play']; print(len(h), 'human plays,', sum(1 for d in h if d.get('would')), 'with would')" "$S\frames.jsonl"
```

Expect a count close to the number of cards you played in that battle (the HUD saw about 47), nearly all `with would`. Send back that line and the `stopped:` line.

**3. The dry-run friendly.** Start the runner in the lobby, then start a friendly yourself and play it with the mouse as usual:

```powershell
& $BB scripts\vision_play.py --seconds 600
```

**Do not click or select text in the PowerShell window while the runner runs.** With QuickEdit on (the default), a selection pauses the console, and the runner then stops reading the screen until the selection ends. Switch modes with F8 or the control window. Copy the lines to send back from the window after the `stopped:` line, or read them from the session's `frames.jsonl`.

It first prints its setup. Check these lines:
- `decide     live; wait_keep 0.25 ...`, with no `WARNING`.
- `detector` names the pinned `best.pt` on `cuda:0`, `side joint`, `0 unresolved`. `NONE (...)` means no detector: the board stays empty for the model.
- `hud        clock templates ok`. `MISSING` means section 16 is not done, and the bot never decides.
- `source     printwindow (auto: a dry run opens no adb); ...`. Only `--tap` adds the adb fallback, after `scripts\emulator.py` has verified the device.
- `taps       off (dry run: no --tap)`.
- `hotkey     on: F8` and `window     on`.
- `frame 747x1328` once the first frame is in.

Then it prints a status line every second, for example:

```
[battle 1] tick 1440 | elixir 2.3 | hand Log/HogRider/Skeletons/Fireball next IceSpirits | MANUAL | last Skeletons -> (8, 22) [manual] @1436 | would HogRider -> (14, 17) @1438 | 31 fps, 0 taps
```

| Part | Meaning |
|---|---|
| `[battle 1]` | in battle 1. Outside a battle: `[menu]`, `[end]` (the end screen), or `[battle]` (a battle screen before the clock has run for a second) |
| `tick 1440` | the battle clock in ticks: 20 a second, from 0 at 3:00 (1:48 left is 1440). `OT` in overtime; `-` when the clock is not read |
| `elixir 2.3` | our elixir |
| `hand ... next ...` | the hand, left to right, and the next card. `?` is a slot it cannot read, or an empty slot right after a play. |
| `MANUAL` | who plays: `MANUAL` (you), `AUTO(dry)` (the bot, without `--tap`), `AUTO(tap)` (the bot taps), `STOPPED` |
| `last ...` | the bot's last play: card, screen tile, what happened, and the tick. The tile is x 0-17 from the left and y 0-31 from the top; our half is y 17-31. `manual` is a suggestion only, `dry` would have played, `tap` tapped, `dropped` means the mode switched first. |
| `would ...` | while the bot does not tap: what it would play now (its best card at its best tile), worked out on a wait at most every 10 ticks. It is never tapped. |
| `31 fps, 0 taps` | frames processed per second, taps sent, and the error count if there are errors |

During the battle:
- Check that the hand, the elixir and the clock match the screen. After you play two cards quickly, both slots must show the right new cards within about a second.
- Press F8 once. The status shows `AUTO(dry)` and the decisions become `[dry]`. Nothing is tapped. Press F8 again to go back to `MANUAL`.
- The control window shows the bot's suggestion while you play, and `would play ...` while it waits. Each of your plays is logged (`human_play`) next to the bot's suggestion at that moment and the newest play it would have made with your card still in its hand (`would`, `would_same_card`, `would_tile_dist`). The card's slot empties as soon as you start dragging, so a would-play made during the drag could not have chosen your card and is passed over.
- A `battle 1 over (clock reset)` line in the middle of the battle means a misread clock. Note when it happened. About a second later the runner follows the same battle again (`battle 2 started ...; joined in the middle`), with the towers it had read.

At the end it prints `battle 1 over (end screen) at tick ...` and then `battle 1: WIN 2-1 (...); record 1-0-0 (W-L-D)`. After 600 s it stops with `stopped: 600 s; ... frames, 0 taps, ...`. Ctrl+C stops it earlier.

Everything goes to `reports\vision_play\vision_play_<date>_<time>\frames.jsonl`, with an image every 20 recorded frames.

Optional, on a second dry run: `--decide legacy` logs the decision rule the simulator's 46% was measured with. It differs live while one of your cards is in flight: the simulator kept the card in its slot and the bot waited, while live the slot is empty from your drag until the refill (about 1.5 s) and legacy plays its next-best card. `--tap` refuses `--decide legacy`.

**4. Latency.** It reads the log only; MuMu is not needed:

```powershell
& $BB tools\latency_probe.py --last 1 --json reports\vision\latency_dryrun.json
```

It prints the time of each stage (stale, perceive, decide, capture to decision) as median and p90 in ms, the frame rate, and how often the bot decided. `D: not measured` is expected here: D needs taps (section 18). The token model decides on the CPU. If `decide` is slow or MuMu stutters, try `--torch-threads 2` (fewer CPU threads for the model) on the next run and compare.

Send back:
- the setup lines;
- about ten status lines spread over the battle;
- the battle and result lines, and the `stopped:` line;
- the latency probe output;
- anything that did not match the screen.

## 18. First friendly with taps (you watch the whole time)

Only while you sit at the PC and watch MuMu for the whole battle. Tell your clanmate it is a bot test.

You need:
- a clean dry run (section 17): clock, hand and elixir read right, sensible suggestions, no errors;
- MuMu at 1080x1920 with no override size (step 7). The runner checks `wm size` and refuses another size;
- no `C:\hcr\STOP` file. The runner will not start while it exists.

**1. Start it in MANUAL.** In the lobby:

```powershell
& $BB scripts\vision_play.py --tap --battles 1
```

As in section 17, do not click or select text in the runner's PowerShell window: with QuickEdit on, a selection stops the whole loop, perception and taps, until it ends. Use F8, the control window or the STOP file.

Before it opens the adb shell, `scripts\emulator.py` must confirm that the serial is the Clash Royale instance. The setup must then show `taps       ON (verified Clash Royale instance)`. If it prints `refusing to tap: ...`, nothing is tapped: send that line back.

It starts in **MANUAL**: it watches and suggests, but does not tap.

**2. Hand over with F8.** Start the friendly yourself. Once the status line shows `[battle 1]` and the clock is running, press **F8**, or the window's "Hand back to the bot" button:
- The window turns green, `AUTO: the bot plays`, and the status shows `AUTO(tap)`.
- Each play is two taps: the card, then the tile.
- **To take over,** press F8 again. The window turns orange, `MANUAL: you play`. A tap the bot already decided is dropped. Press F8 to hand back.
- Keep your hands off MuMu while the bot plays. Press F8 first.
- The `MANUAL` file does the same from a console: `New-Item -ItemType File -Force C:\hcr\MANUAL` switches to MANUAL, and `Remove-Item C:\hcr\MANUAL` back to AUTO.
- While the runner runs, F8 belongs to it: other programs do not get the key.

**3. Stop.** Any of these works, in any mode:
- Ctrl+C in the runner's window;
- the STOP file, from a second PowerShell window:

  ```powershell
  New-Item -ItemType File -Force C:\hcr\STOP | Out-Null
  ```

  The runner ends. Delete the file before the next run: `Remove-Item C:\hcr\STOP`.

With `--battles 1` it also stops by itself after the result of the first battle.

**4. Watch the whole time:**
- The first taps: the card and tile the status line names must be what lands. Taps go to device pixels (1080x1920) through one adb shell.
- The end: the runner only taps while the battle screen is up, and it never presses OK. You press OK yourself. If a tap lands after the clock has stopped, press F8 and send it back.
- A `battle 1 over (clock reset)` line in the middle of a battle is a misread clock: send it back.
- Do not watch replays or spectate clan battles while the runner runs with `--tap`. Stop it first.

**5. The real delay.** After the battle:

```powershell
& $BB tools\latency_probe.py --last 1 --json reports\vision\latency.json
```

`D = N ticks` is the delay from the bot's decision to its unit on the board, measured on this PC. The simulator evaluation in section 14 assumed 21 ticks. D needs at least 5 troop or building plays by the bot; with fewer, it says so.

Send back:
- the setup lines;
- the status lines around the first taps and around each F8;
- the result line and the `stopped:` line;
- `reports\vision\latency.json`;
- what you saw.

Later, once you trust it:
- `--start-mode auto` starts with the bot in control;
- leaving out `--battles` keeps it running across several friendlies. STOP or Ctrl+C ends it.
- The mode carries over from one battle to the next. If it is AUTO when a friendly starts, the bot plays it. Press F8 between battles if you want to start the next one yourself.

Do not commit `reports\vision_play\`, `reports\vision\`, `STOP` or `MANUAL`. They are not gitignored, and the recordings show account names.

## 19. Second model: decision-time rows

**Close MuMu and the browser first.** These jobs use most of the RAM. Do not run this while the live runner is running.

The first model learned each play at the tick it landed. Live, the bot decides about 24 ticks (1.2 s) earlier: the client's command delay of 21-22 ticks, plus capture and inference. In the simulator the first model loses about 11 points at a 21-tick delay (46.0% against 56.7%). The second model is trained on decision-time rows: each play is shown at the moment its player tapped it, about 24 ticks before it lands. The rows also show only what the screen shows (no invisible enemies), and they drop more of the rows where the simulator has drifted from the recording (trust v2). Unit tokens stay `--live-compat`, like the first model.

It needs the first model from section 14: `runs\tokens\live_big_hog\best.pt` and `runs\tokens\live_big\ep01.pt`. The script checks for them before it starts. The new rows in `data\rs_rows_dt` need about as much disk as `data\rs_rows_live` (section 3: 20-35 GB).

After `git pull`, run the tests of section 12 first. The batch 4 row options and this script's training options have not run on the PC yet, and a failing test costs a minute where a failed step can cost an hour. Paste any failure back before you start the script.

**One script runs everything unattended.** It stops at the first failure and writes everything to `reports\second_model_<time>.log`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\second_model.ps1
```

| Step | What it runs | Time (first model) |
|---|---|---|
| rows | `tools\rs_rows.py --live-compat --obs-delay 24 --obs-delay-jitter 3 --opp-lag 1 --visibility public --trust v2 --presses off --others 20000 --workers 8 --out data\rs_rows_dt` | about 25 min (20 min) |
| general | `sim.train_tokens --name dt_big`, d256, 6 layers, 8 heads, 3 epochs, `--wait-keep 0.25 --trust v2` | about 35 min (33 min) |
| hog | the same with `--name dt_big_hog --finetune-hog --init runs\tokens\dt_big\best.pt` | about 10 min (9.4 min) |
| eval | three runs of `tools\rs_eval.py`, 300 matches each, against `live_big\ep01.pt`: A `live_big_hog` at `--delay 21`, B `dt_big_hog` at `--delay 21`, C `dt_big_hog` with no delay | about 20 min (two runs: 13 min) |

That is about 1.5 hours in all. Options:
- `-Workers 6` uses less RAM.
- `-From general` (or `hog`, `eval`) restarts at a later step.
- `-MaxRows 4000000` trains on a sample, if training runs out of memory.
- `-League` adds an optional last step: `tools\rs_league_eval.py run` with `dt_big_hog` as the candidate and `live_big_hog` as the champion. It plays 400 paired games at a 24-tick delay with the default perception noise, on both seats and with both decks, on at most 4 workers. Allow another 30-60 minutes; this step has not been timed yet. It writes `runs\league\second_model.jsonl` and `runs\league\second_model.report.json`. `-From league` runs that step alone.

**Reading the result.** A and B are the comparison. Both use the same opponent, the same 300 held-out matches and the same delay. A re-measures the first model under the current code, because the evaluator's opponent memory has changed since the 46.0% (E08: plays are now remembered when they land). Compare B with A, not with 46.0%. Each interval is about +-5.6 points, so a difference of less than about 8 points between A and B is not conclusive at 300 matches. The league step is the finer test. C shows how much of B's result is lost to the delay. The 400-game league result is only a first reading: the real promotion gate needs 1,000 games.

Keep using `live_big_hog` in the live runner (sections 17-18) until these numbers are back.

Send back the log, `reports\second_model_<time>.log`. If you cannot send all of it, send:
- the rows step's header lines (`memory:`, `seen:`, `trust:`, `observed:`), its `selected` line, and its last two lines (`done: ... rows` and the trusted / trusted_v2 rates);
- from both trainings, the `rows:` lines and every `epoch N: val ...` line;
- the three `rs_eval` result lines (`...: N/300 wins (...)`);
- with `-League`, the gate decision and the `report:` line.

### Why the second model lost (run this before section 20)

The second model came back on 2026-10-30 (`reports\second_model_20261030_101529.log`):

| Run | Win rate | Crown difference | Plays per match |
|---|---|---|---|
| A `live_big_hog`, 21-tick delay | 46.0% [40.4, 51.7] | -0.34 | 33.3 |
| B `dt_big_hog`, 21-tick delay | 29.3% [24.5, 34.7] | -1.02 | 28.7 |
| C `dt_big_hog`, no delay | 21.0% [16.8, 26.0] | -1.25 | 30.4 |

B was meant to beat A. It lost by 17 points, far outside the noise. Section 20 uses the same decision-time rows, so find out why first.

**The leading explanation** (likely, not yet proven):
- A decision-time row is taken where the player tapped, about 24 ticks before the play lands. The rows also take a "wait" row every 20 ticks wherever the player taps nothing in the next 10 ticks. That includes the ticks just after a tap, while the play is still in flight.
- In those rows the tapped card is still in hand, the elixir is not yet spent and the unit is not on the board: the simulator changes the hand and the elixir only when a play lands, as the client does. So they look like the tap row a moment later, and they say "wait".
- The second model does not read the column that says a play is in flight (`own_pending`), so it cannot tell the two apart. It learns to hold back just when the moment to play has come or passed: fewer and later plays, no quick follow-ups, missed defences. That fits B's 28.7 plays and its crown difference of -1.02.

Ruled out by reading the code:
- The evaluation: under `--delay 21`, `rs_eval` shows the bot's own plays in flight the way the rows do (card in hand, elixir unspent). The wait bias, the decision cadence and the decision rule are the same for A and B.
- What IL_Replay's play tick means (the tick a play ran, or the tap) does not shift the board, because every play of both sides moves together. It can shift only the elixir, by about 1 s. The `headroom` check settles it.

The third model reads `own_pending`, so it should not have this problem. These checks confirm that before a night is spent on it.

After `git pull`, run the tests of section 12 first. The new tests are `tests\pipeline\test_drop_pending_waits.py` and `tests\pipeline\test_rs_dt_diag.py`.

**One script runs the checks, about 85 minutes.** Close MuMu and the browser first. It writes `reports\dt_diagnose_<time>.log`, goes on to the next check when one fails, and ends with a summary of the result lines:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\dt_diagnose.ps1
```

The optional checks are `fullablation` (about 50 minutes), `trust` and `ticks` (about 30 minutes together): add for example `-Steps fullablation`, or `-Steps all` for everything (about 2.5 hours). `-Steps ablation` runs one check alone; `-Workers 6` uses less RAM.

| Check | What it runs | Time | What the result means |
|---|---|---|---|
| headroom | `tools\rs_dt_diag.py rows` on `data\rs_rows_dt` (with `--trust v2`, the rows the second model trained on), then on `data\rs_rows_live` | a few min | `pending waits:` is the share of wait rows with an own play in flight; expect about 15-20%. **What the recorded tick is**, read two ways. On `rs_rows_dt`, the `headroom: < -7` line: about 0 means the tick a play ran, as assumed; a large share (tenths) means the tap. On `rs_rows_live` (skipped with a note if the folder is gone), the `headroom:` shares: a floor near 15-21 ticks, with few rows below it, means the tick a play ran; many rows below 6, with counts from 0, mean the tap. Also on `rs_rows_dt`, `< 0` is the share of play labels below the full bar that the evaluation masks at the tap for lack of elixir (they turn into waits or a cheaper card): the larger it is, the more of B's loss comes from the elixir rather than from the waits. |
| waitbias | `dt_big_hog` with `wait_keep` 0.5 and 1.0 (copies in `runs\tokens\dt_big_hog_wk050`, `_wk100`), and `live_big_hog` with 1.0 as the control (`runs\tokens\live_big_hog_wk100`), all at `--delay 21` | 25 min | Compare each with its own model at the normal bias: `dt_big_hog` 29.3% (28.7 plays), `live_big_hog` 46.0% (33.3 plays). `dt_big_hog` gains clearly more than the control, with plays up to 33-38: its gate is lazy, as the explanation says. Both gain about as much: the wait bias is simply too high at this delay for either model, which the third model would not fix; send the log back. Plays up but wins flat: the play rate is not the cause. |
| live | `live_big_hog` and `dt_big_hog` at `--delay 21 --decide live` (the live runner's rule: no wait at a full bar, no second tap of a card in flight) | 18 min | Compare these two with each other only: both seats change rule. If `dt_big_hog` gains clearly more than `live_big_hog`, decisions lost around its own plays in flight are part of the cause. These are also the numbers closest to live play. |
| offset | `dt_big_hog` at `--delay 24` | 8 min | Within about 3 points of 29.3%: the 24-tick lead it was trained for against the 21-tick delay is not the cause. |
| ablation | the Hog 2.6 fine-tune again with `--drop-pending-waits` (`runs\tokens\dt_big_hog_np`), then `--delay 21` | 20 min | 40% or more, with more plays: the explanation holds, and the third model should fix it. Near 29% is **not** conclusive on its own: the fine-tune starts from `dt_big`, which learnt from those waits too, and its lazy gate can survive three epochs. Run `fullablation` before deciding. |
| fullablation (optional) | the general model and its Hog 2.6 fine-tune again, both with `--drop-pending-waits` (`runs\tokens\dt_big_np`, `runs\tokens\dt_big_np_hog`), then `--delay 21` | 50 min | The decisive check when `ablation` stays near 29%. 40% or more: the explanation holds. Near 29% here too: the cause is elsewhere. |
| trust (optional) | the Hog 2.6 fine-tune again with `--trust legacy` (`runs\tokens\dt_big_hog_tl`), then `--delay 21` | 20 min | Within about 5 points of 29.3%: trust v2's smaller training set is not the cause. |
| ticks (optional) | `rs_eval --pro` (the recorded plays, no model) with and without `--delay 21`, then `tools\rs_dt_diag.py ghost` | 10 min | A second check of the tick meaning. The same result on nearly every match (a few percent differ either way, from plays in the first seconds): the tick a play ran. Many matches differing: the tap. |

The ablations' trainings print `--drop-pending-waits: N wait rows with an own play in flight left out`. Their validation uses the same rows as `dt_big`'s and `dt_big_hog`'s, waits in flight included, so their `gate` figures may read lower. Card and tile compare directly. Judge them by their `rs_eval` lines.

Even 40% is strong evidence rather than proof. The ablated model still decides under the old rule, which may now try to tap again while its first play is in flight (the evaluation refuses or masks those taps). The third model learns that from `own_pending` instead.

The same checks one by one, in the shorthand at the top of this file (the wait-bias copies are easiest with `-Steps waitbias`):

```powershell
$OPP = 'runs\tokens\live_big\ep01.pt'
& $RG tools\rs_dt_diag.py rows data\rs_rows_dt --trust v2
& $RG tools\rs_dt_diag.py rows data\rs_rows_live
& $RG tools\rs_eval.py --ckpt runs\tokens\live_big_hog\best.pt --opponent-ckpt $OPP --matches 300 --workers 8 --delay 21 --decide live
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog\best.pt --opponent-ckpt $OPP --matches 300 --workers 8 --delay 21 --decide live
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog\best.pt --opponent-ckpt $OPP --matches 300 --workers 8 --delay 24
& $BB -m sim.train_tokens --name dt_big_hog_np --data data\rs_rows_dt --wait-keep 0.25 --epochs 3 --d 256 --layers 6 --heads 8 --finetune-hog --init runs\tokens\dt_big\best.pt --trust v2 --drop-pending-waits
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog_np\best.pt --opponent-ckpt $OPP --matches 300 --workers 8 --delay 21
& $BB -m sim.train_tokens --name dt_big_np --data data\rs_rows_dt --wait-keep 0.25 --epochs 3 --d 256 --layers 6 --heads 8 --trust v2 --drop-pending-waits
& $BB -m sim.train_tokens --name dt_big_np_hog --data data\rs_rows_dt --wait-keep 0.25 --epochs 3 --d 256 --layers 6 --heads 8 --finetune-hog --init runs\tokens\dt_big_np\best.pt --trust v2 --drop-pending-waits
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_np_hog\best.pt --opponent-ckpt $OPP --matches 300 --workers 8 --delay 21
& $BB -m sim.train_tokens --name dt_big_hog_tl --data data\rs_rows_dt --wait-keep 0.25 --epochs 3 --d 256 --layers 6 --heads 8 --finetune-hog --init runs\tokens\dt_big\best.pt --trust legacy
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog_tl\best.pt --opponent-ckpt $OPP --matches 300 --workers 8 --delay 21
& $RG tools\rs_eval.py --pro --matches 300 --workers 8
& $RG tools\rs_eval.py --pro --matches 300 --workers 8 --delay 21
& $RG tools\rs_dt_diag.py ghost --delay 21
```

**What to do next:**
- `headroom` says the tick is the tap (a large `headroom: < -7` share on `rs_rows_dt`, or many rows near 0 on `rs_rows_live`): do not run section 20. Send the log back. The rows then need a re-drive with the delay first.
- `ablation` or `fullablation` lifts the model clearly, to about 40% or more, or `waitbias` lifts `dt_big_hog` clearly more than its `live_big_hog` control: run section 20, in two steps.
  1. The second model's corpus size first, about 3-4 hours instead of a night: `third_model.ps1 -Others 20000 -Tag 20k` (add `-SkipDiskCheck` if `C:` has less than the 90 GB the full run checks for). Its model is `v2_hog_20k`.
  2. Only once `v2_hog_20k` beats `live_big_hog` (its league score against it above 50%; a PASS needs at least 551 of 1,000): the whole corpus, `third_model.ps1` with no `-Tag`. Its model is `v2_hog`; the 20k run's files stay as they were. Free the disk first (section 20, "Disk").
- `ablation` stays near 29%: run `-Steps fullablation` before deciding anything.
- Neither ablation moves it, and `waitbias` lifts both models alike or neither: hold section 20, run `-Steps trust`, and send the log back.

Keep `live_big_hog` in the live runner meanwhile.

Send back `reports\dt_diagnose_<time>.log`. If you cannot send all of it, send the `== summary` block at its end.

## 20. Third model: v2 rows from the whole corpus

**Close MuMu and the browser first.** These jobs use most of the RAM. Do not run this while the live runner is running. The whole run takes most of a night.

What is new compared with section 19's rows (decision time, the screen's view, trust v2, all kept):
- **Schema v2** (E14). Each row also shows:
  - the spells in flight;
  - the opponent's elixir, counted from the plays it has shown;
  - our own plays tapped and not yet landed;
  - hand evolution;
  - the ability buttons;
  - each unit's form (evolution, hero) and whether it is burrowing.

  Our deck has no evolutions or heroes, but the opponents' decks do.
- **Ability presses** (E10, `--presses on`). Champion and hero ability presses are replayed and learnt by the model's ability head (E15). Our deck has no ability, so the bot never presses one live.
- **Perception noise** (E17). Every row is seen through the noise profile: what our screen reader gets wrong (missed units, late plays, wrong cards). Until a profile is fitted from our own friendlies, this is the prior, `data\noise\default_prior.json`.
- **The whole corpus** (E18): every usable IL_Replay match, instead of every Hog 2.6 match plus 20,000 others. That is about 7 times the matches.
  - 25% of the wait rows are kept while the rows are built.
  - The rows are then compacted into one file per column (`data\rs_cols_v2`), and the trainer reads them from disk (`--memmap`), so they need not fit in 16 GB of RAM.
  - Each training row is flipped left-right with probability 0.5 (`--mirror 0.5`).
- **A critic** (E15): the value the RL step (batch 6) starts from.

The model keeps the size of `live_big` and `dt_big`: d256, 6 layers, 8 heads. That should take about 3-4 GB of the 12 GB of VRAM (estimated). With the same size, the comparison is about the rows, and the simulator's CPU workers (the league, later RL) run it as fast as the champion.

Unit tokens stay `--live-compat`, as in sections 14 and 19, for two reasons:
- The live runner reads either kind. But the simulator's evaluators (`rs_eval`, `rs_match`, `rs_league_eval`) and the RL workers build only `--live-compat` or unit-type rows: a `--units vision` model could not be evaluated or trained by RL.
- What the vision classes would add, the unit's form, is in every v2 row anyway.

Run section 19's checks ("Why the second model lost") first. These rows are decision-time rows too, and the checks say whether the v2 model should fix what went wrong there.

It needs:
- section 14's `runs\tokens\live_big_hog\best.pt` (the champion: the live model and the best so far) and `runs\tokens\live_big\ep01.pt` (the anchor, and the reactive opponent of `rs_eval`);
- about 90 GB free on `C:` (below).

The script checks these before it starts. Keep `data\engine.json` and the card table as they were for section 14: the league refuses checkpoints made under another engine.

The champion used to be section 19's `dt_big_hog`. At 29.3% against `live_big_hog`'s 46.0%, beating it would not mean v2_hog is better than the live model, so the gate now compares with `live_big_hog`.

After `git pull`, run the tests of section 12 first. None of this batch's code (v2 rows, compaction, `--memmap`, the v2 model) has run on the PC yet. Paste any failure back before you start the script.

**One script runs everything unattended.** It stops at the first failure, except in the evaluation runs: a failed `rs_eval` run is logged, the league still runs, and the script lists the failures and exits with an error at the end. Everything goes to `reports\third_model_<time>.log`. Run it twice, as section 19's "What to do next" says: first on the second model's corpus size, then, once that model beats `live_big_hog`, on the whole corpus:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\third_model.ps1 -Others 20000 -Tag 20k
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\third_model.ps1
```

`-Tag 20k` gives the first run its own files: `data\rs_rows_v2_20k`, `data\rs_cols_v2_20k`, `runs\tokens\v2_big_20k`, `runs\tokens\v2_hog_20k`, and the league player `v2_hog_20k`. The run without `-Tag` uses the names in the table below. Both share the league log `runs\league\third_model.jsonl`, so `live_big_hog`'s 400 games against `live_big` are played once and reused. The league refuses a player name whose games it already holds when the checkpoint has changed. So the script stops at the start, not after hours, when it would train a model whose name already has games in the log: use another `-Tag`, or move the log away. The first run takes about 3-4 hours.

| Step | What it runs | Time (estimated) |
|---|---|---|
| rows | `tools\rs_rows.py --schema v2 --live-compat --obs-delay 24 --obs-delay-jitter 3 --opp-lag 1 --visibility public --trust v2 --presses on --noise data\noise\default_prior.json --all-matches --wait-keep-at-build 0.25 --workers 8 --out data\rs_rows_v2` | about 3.5 h |
| compact | `tools\rs_rows_compact.py --rows data\rs_rows_v2 --out data\rs_cols_v2 --force` | about 30 min |
| general | `sim.train_tokens --name v2_big --data data\rs_cols_v2 --epochs 2 --memmap --mirror 0.5 --trust v2 --wait-keep 0.25 --d 256 --layers 6 --heads 8 --val-max-rows 500000` | about 3 h |
| hog | the same with `--name v2_hog --epochs 3 --finetune-hog --init runs\tokens\v2_big\best.pt` | about 15 min |
| eval | five runs of `tools\rs_eval.py`, 300 matches each, against `live_big\ep01.pt` (`--ckpt runs\tokens\v2_hog\best.pt --opponent-ckpt runs\tokens\live_big\ep01.pt --matches 300 --workers 8`): `v2_hog` at `--delay 21`, with no delay, at `--delay 24`, and at `--delay 21 --decide live`; then `live_big_hog` at `--delay 21 --decide live` for reference | about 45 min |
| league | `tools\rs_league_eval.py run --candidate v2_hog=runs\tokens\v2_hog\best.pt --champion live_big_hog=runs\tokens\live_big_hog\best.pt --anchor live_big=runs\tokens\live_big\ep01.pt --delay 24 --noise data\noise\default_prior.json --champion-games 1000 --anchor-games 400 --full --workers 6 --log runs\league\third_model.jsonl` | about 1-1.5 h |

That is about 9.5 hours in all, and anywhere from 7.5 to 12.5. Start it in the evening.

None of these times is measured yet. They are the first model's measured times (section 14 and `docs/SPARRING_PLAN.md`, "Results so far"), scaled by the data:
- **rows**: the first model's 33,659 matches took 20 min. Here there are about 7 times the matches, with about 1.5 times the work per match (the trust check, the presses, the noise, the v2 columns).
- **compact**: it reads about 15-20 GB of shards and writes about 50 GB of columns.
- **general**: the first model's 3 epochs took 33 min. Here:
  - there are about 7 times the training rows: 7 times the matches, with the waits thinned to the same 25%;
  - there are 2 epochs instead of 3;
  - a v2 row has about 80 tokens instead of about 66.

  In all, about 5.5 times the work.
- **hog**: every Hog 2.6 match was already in the first model's rows, so this is about the first model's 9.4 min with the larger rows.
- **eval**: five runs of 300 matches. Section 19's runs took 8-9.5 min each.
- **league**: 1,800 games at most 6 at a time:
  - 1,000 of v2_hog against live_big_hog;
  - 400 of v2_hog against live_big;
  - 400 of live_big_hog against live_big.

  Each game takes about 10-13 s of one CPU core, as an `rs_eval` match does, with the noise on both seats. `rs_league_eval` runs fewer workers when free memory is short (it keeps 3 GB free, about 1.2 GB per worker).

The training steps print their own time to go. Each prints `ep 0 step 500/S` every 500 steps, where S is the total number of steps. Take the time between two of these lines, multiply by S / 500, and that is how long the whole training takes.

Options:
- `-Workers 6` uses less RAM.
- `-Others 60000` builds rows from every Hog 2.6 match plus 60,000 others drawn at random, instead of every match. That takes about a third of the rows, compact and general time and disk.
- `-Epochs 1` (or 3) changes the general model's epochs. The Hog 2.6 fine-tune always runs 3.
- `-MaxRows 15000000` trains on a sample of that many rows, which halves the general step if training runs long.
- `-Noise <profile.json>` uses another noise profile, such as one fitted by `tools\fit_noise_profile.py`, for both the rows and the league.
- `-From compact` (or `general`, `hog`, `eval`, `league`) restarts at a later step. Give it the same `-Tag` as the run it continues. The script refuses to start from `rows` while `data\rs_rows_v2` (`data\rs_rows_v2_<tag>` with `-Tag`) holds finished rows: use `-From compact`, or delete the folder to build them again.
- `-Tag <name>` names this run's files, as above. Letters, digits, `_` and `-` only.

**Disk.** These sizes are estimated: about 40 million rows of about 1.27 KB each.

| Folder | Size |
|---|---|
| `data\rs_rows_v2` (the shards, compressed) | about 15-20 GB |
| `data\rs_cols_v2` (the compact store, uncompressed so that it can be memory-mapped) | about 50 GB |
| `runs\tokens\v2_big`, `runs\tokens\v2_hog` | under 0.5 GB |

The script stops before it starts if `C:` has less than 90 GB free (60 GB with `-From compact`); `-SkipDiskCheck` starts anyway.

If space runs short:
- After the compact step, `data\rs_rows_v2` is read only to rebuild the store (`-From compact`). Deleting it gives back 15-20 GB.
- Before the whole-corpus run, delete the `-Tag 20k` run's rows and store once its league is done. The models in `runs\tokens\v2_big_20k` and `v2_hog_20k` stay, and no evaluation reads the rows:

  ```powershell
  Remove-Item -Recurse -Force C:\hcr\data\rs_rows_v2_20k, C:\hcr\data\rs_cols_v2_20k
  ```
- `data\rs_rows_live` and `data\rs_rows_dt` are needed only to retrain the first or second model. No evaluation reads them.

**Reading the result.**
- **eval** is section 19's instrument: the same reactive opponent, the same 300 held-out matches and the same 21-tick delay as its runs A (`live_big_hog`, 46.0%) and B (`dt_big_hog`, 29.3%). Compare the first line with those. Each interval is about +-5.6 points, so a difference of less than about 8 points is not conclusive. The other lines:
  - **no delay** and **24-tick delay**: a model trained for the 24-tick lead should do best at 21-24 ticks and worse without a delay, as `dt_big_hog` did (21.0% without one). If v2_hog does much better without a delay than at 21, its timing is off.
  - **live rule**: the two `decide live` lines compare only with each other, because both seats change rule. v2_hog should be at least level with `live_big_hog` there, since these are the numbers closest to live play. The `plays/match` figures show a lazy gate directly: the pros play about 35-40 cards a match.
- **league** is the real test. It plays at a 24-tick delay with perception noise on every seat, from both seats, on deck pairings and seeds that are the same for every player. Against the champion, each pairing is also played with the decks exchanged. Its `gate:` lines give the decision for `v2_hog`:
  - **PASS**: v2_hog beats `live_big_hog` (Wilson 95% lower bound above 52% over 1,000 games, i.e. at least 551 points), and it scores no more than 2 points below `live_big_hog` against `live_big`. Then v2_hog becomes the live model: add `--ckpt runs\tokens\v2_hog\best.pt` (`runs\tokens\v2_hog_20k\best.pt` for the `-Tag 20k` run) to the `vision_play.py` commands of sections 17 and 18. The runner reads the checkpoint's schema and builds v2 rows by itself.
  - **FAIL**, **CYCLE** or **INCOMPLETE**: keep the live model you have.

  The `paired` lines show v2_hog's score minus live_big_hog's against the same opponent on the same positions, with an interval.
- The report is also written to `runs\league\third_model.report.json`. Each run overwrites it; its text is in the run's log too. In the whole-corpus run's report, `v2_hog_20k` appears among the other players.

If the league step fails, the trained models are kept. Send the log back, and after the fix restart with `-From league` (and the same `-Tag`). The league log resumes: games already played are not played again. If an evaluation run failed, `-From eval` runs the evaluations again and then the league, which plays only the games it lacks.

At the start the script also checks that the league can play a v2 checkpoint (`tools\rs_match.py` must build v2 rows). If it cannot, a yellow `warning:` line says so. The other steps still run, and the script stops before the league step instead of failing in it. Run `git pull`, then `-From league`.

Send back the log, `reports\third_model_<time>.log`. Each step's `==` line has its start time, which is how the times above get measured. If you cannot send all of it, send:
- from the rows step:
  - its header lines (`memory:`, `seen:`, `trust:`, `presses:`, `observed:`, `schema:`, `waits:`, `noise:`, and any `not bound:` lines);
  - its `selected` and `pick` lines;
  - its last two lines (`done: ... rows` and the line after it).
- from the compact step: the `compact:` line, the `done:` line (rows and GB) and the `index:` line;
- from both trainings:
  - the `schema` line and both `rows:` lines;
  - every `epoch N: val ...` line;
  - the `done in` line;
- the `rs_eval` result line (`...: N/300 wins (...)`);
- from the league:
  - its `workers:` line;
  - everything from `league evaluation:` to `best:`;
  - the `report:` line.

## 21. RL fine-tuning: the bench, then days of training

This is Phase 7 of `docs/SPARRING_PLAN.md`: reinforcement learning in the simulator, to play better than the pros it copied without learning tricks that only work in RoyaleSim. It is pure simulation. Nothing here starts, touches or taps MuMu, so the emulator is not needed.

It needs:
- section 20's `runs\tokens\v2_hog\best.pt`. This is the start model, the first champion and the model the training stays close to (the KL pull). The script checks for it. If only section 20's `-Tag 20k` run has been done, pass `-Start runs\tokens\v2_hog_20k\best.pt`.
- the imitation models `runs\tokens\dt_big_hog\best.pt` (section 19) and `runs\tokens\live_big_hog\best.pt` (section 14) as anchors. A missing one is skipped with a warning.
- IL_Replay (section 11). It supplies the opponents' decks and the gate's held-out pairings.

After `git pull`, run these tests first. None of the RL code has run on the PC yet. Paste any failure back before you start.

```powershell
& $RG -m pytest tests/pipeline/test_train_loop.py tests/pipeline/test_rollout.py tests/pipeline/test_ppo.py tests/pipeline/test_league_pool.py tests/pipeline/test_league_eval.py -q
& $BB -m pytest tests/pipeline/test_ppo.py -q
```

**What one iteration does** (`tools\rl\train_loop.py`):

1. **Collect.** It plays `-Matches` battles (default 400) in the engine venv, on `-Workers` processes.
   - The learner plays Hog 2.6 in base forms, every card and tower at level 11, as in a friendly.
   - Every command lands 24 ticks after the decision, as live.
   - The bot sees the board through the perception-noise profile: `data\noise\measured.json` once you have fitted one (section 20 explains the fit), else the prior.
   - Opponents:
     - 50%: the learner itself, on both seats;
     - 35%: earlier snapshots of the learner (one joins the pool every 5 iterations);
     - 15%: the imitation models.

     Within each group, the opponents that still beat the learner are drawn more often.
   - The opponents' decks are drawn from the IL_Replay train split, as often as they are played there.
2. **Update.** One PPO update runs in the GPU venv. The learner moves toward what won, and an adaptive pull keeps it within about 0.075 nats per decision of v2_hog. A second pull keeps its play rate near v2_hog's, so it plays about as many cards as the pros.
3. **Record.** The league files the learner's results and the run moves to the next iteration. Each step is saved as it ends, so a crash repeats at most the step that was running.

**The gate** runs every 10 iterations (`-GateEvery`). It is the same test as section 20's league:
- the newest checkpoint against the champion: 1,000 games on the dev split's frozen pairings, both seats, at the 24-tick delay with the run's noise;
- 400 games against each anchor;
- **PASS**: the Wilson lower bound against the champion is above 52%, and the score against no anchor drops more than 2 points below the champion's. The checkpoint becomes the champion and is copied to `runs\rl\hog_rl1\champion.pt`.
- **FAIL** (or a gate that could not run): the champion stays.

The first champion is v2_hog. After a pass, v2_hog becomes an anchor too.

### Step 1: the bench (about 15 minutes)

Close MuMu and the browser. The bench measures what fits in memory, so it must run with them closed.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\rl_train.ps1 -Bench
```

The first time, the script first builds the opponents' deck field (`tools\deck_stats.py --top 2000`, a few minutes) if `data\deck_stats.json` is missing. Then it plays for about 10 minutes at 1, then about half, then all the workers that RAM and CPU allow, and times one PPO update. It prints:
- `workers N: ... matches/hour, ... RAM per worker ... MB at peak` for each count;
- `RAM per worker` and `recommended: --workers N`, with the reason;
- how long an iteration of 400 matches takes, collection and update;
- a last line with the command to start: `rl_train.ps1 -Workers N -Hours 48`.

Options: `-BenchMinutes 20` measures longer. The bench writes `runs\rl\hog_rl1\bench-<time>\bench.json` and trains nothing.

### Step 2: the long run (days, MuMu closed)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\rl_train.ps1 -Workers 8 -Hours 48
```

Use the bench's `-Workers`. Before every collection, the driver lowers the count when free memory is short, and it learns the real memory per worker from the previous collection.

How long (estimated; the bench measures the collection part):
- One iteration is the collection (400 matches at the bench's matches/hour) plus the update (the bench's time).
- One gate is 1,000 games against the champion, plus 400 against each anchor only when the candidate beat the champion. Each game takes about 10-13 s of one core, on 6 workers: about 35-60 minutes. The first time a candidate beats the champion, the champion also plays both anchors once (800 more games); later gates reuse those games.
- Plan 2-7 days in all. The gate's 52% bar is strict, so many gates fail before one passes. That is expected.

Other options:
- `-Iterations 200` stops after 200 iterations in all.
- `-Matches 800` makes larger iterations (the update needs more memory and time).
- `-GateEvery 20` runs fewer gates (more training time). Use `0` for none.
- `-Noise <profile.json>` (new run only) sets the noise profile.
- `-Name hog_rl2` starts a second, separate run.

Disk: each iteration keeps one checkpoint, about as big as `runs\tokens\v2_hog\best.pt` (check with `(Get-Item runs\tokens\v2_hog\best.pt).Length / 1MB`). 200 iterations take 200 times that. The rollout files are deleted after each update. Do not delete `itNNNN.pt` files while the run uses them: the league plays its snapshots from them.

### Stop and resume

- **Stop cleanly:** from a second PowerShell window:

  ```powershell
  New-Item -ItemType File -Force C:\hcr\runs\rl\hog_rl1\STOP
  ```

  The step that is running (a collection, an update, or a gate, which can take an hour) finishes, then the run stops. `C:\hcr\STOP`, the live runner's stop file, stops it too. **Delete the file before you resume:** `Remove-Item C:\hcr\runs\rl\hog_rl1\STOP`. The script refuses to start while either file exists.
- **Stop now:** Ctrl+C. The interrupted step runs again on resume. Use Ctrl+C or the STOP file, not Task Manager: a collection notices that the driver is gone and stops by itself, but a gate that was running would play on. If you did end it from Task Manager, end any `python.exe` that is left before you resume.
- **Resume:**

  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\rl_train.ps1 -From resume -Workers 8 -Hours 24
  ```

  The run keeps the settings it started with: start model, anchors, delay, noise, decks, mix and gate. It also keeps its own copies of the engine settings and the noise profile (`engine.json` and `noise.json` in the run folder), so a `git pull` that changes `data\engine.json` or the noise prior does not change the run. The resume is refused (exit 1 or 2) if royalesim, a checkpoint file (`v2_hog` or an anchor) or `data\deck_stats.json` changed since the start. Then start a new `-Name` instead; so do not retrain `v2_hog` or rebuild the deck stats while a run still needs them. `-Workers`, `-Matches`, `-Hours`, `-Iterations` and `-GateEvery` can change on every launch.
- **Go back:** `-From 40` resumes from iteration 40. Later checkpoints move to `runs\rl\hog_rl1\rollback-<time>\`, and the league and champion return to their state at iteration 40. Use it when a run went wrong after some iteration.
- **See where it is**, also while it runs:

  ```powershell
  & $RG tools\rl\train_loop.py status --name hog_rl1
  ```

  It prints the iteration, the champion, every gate and the last five iterations.

### Reading the output

Every iteration prints four lines:
- `it 0012 -> it0013.pt: 400 matches in 7.5 min (3200/h, peak 900 MB a worker), ... rows`;
- `learner score: anchors 0.55 (60), pool 0.48 (140)`: its mean score per opponent group, over the non-mirror games. Near 0.5 against the pool is normal: the pool is its own recent self. Plays per seat are the learner's cards per match (pros: 35-40).
- `ppo ...`:
  - `kl` should stay near its 0.075 budget. The coefficient after it rises when the policy pulls away, and falls when it stays close.
  - `q_play` is the learner's play rate and `ref` v2_hog's. They should stay close.
  - `explained var` (the critic's fit) should rise above 0 within the first iterations.
- `league: ... active, admitted snap:it0015, ...`.

Every gate prints its decision and a line per condition, as in section 20. The files:
- each gate's report: `runs\rl\hog_rl1\gates\itNNNN.report.json`;
- each step's full output: `runs\rl\hog_rl1\logs\`;
- one record per iteration and gate: `runs\rl\hog_rl1\log.jsonl`.

Warning signs to send back:
- `warning: N opponent decks` in the header: `data\deck_stats.json` holds fewer than the plan's 1,000 decks (an older file built with `--top 500`). Stop, delete the file, and start a new run with another `-Name`: the script rebuilds it with 2,000;
- `q_play` far from `ref`;
- `kl` many times its budget for several iterations;
- `errors` above 0 in many iterations;
- refused plays per seat that keep growing.

### When a checkpoint passes the gate

`champion.pt` is the new champion. It plays with v2_hog's wait setting: the run folder's `args.json` carries it, so keep the file next to `champion.pt`.

Try it live (sections 17 and 18) with `--ckpt runs\rl\hog_rl1\champion.pt`. The plan's last check is real friendlies, alternating it with v2_hog match by match, at least 40 per arm. It must do no worse than v2_hog, with plays per match and elixir leak inside human ranges.

Exit codes of the script:
- 0: done (time or iterations reached, or STOP);
- 1: a step failed;
- 2: refused at the start (bad settings or files);
- 3: done, but a gate could not run;
- 130: interrupted.

Send back the log, `reports\rl_train_<time>.log`. If it is too big, send:
- from the bench:
  - every `workers N:` line;
  - the `ppo:` line;
  - the `RAM per worker`, `recommended` and `at --workers` lines;
- from a run:
  - the `train_loop: run ...` header (8 lines);
  - the four lines of the first 3 and the last 3 iterations;
  - every gate block (`gate at iteration ...` and the lines under it);
  - the `stopped:` line;
  - the `status` output at the end;
- `runs\rl\hog_rl1\log.jsonl` (small) and `runs\rl\hog_rl1\gates\*.report.json`;
- any `error:` or `warning:` line, with the matching file under `runs\rl\hog_rl1\logs\`.

## If something breaks

Paste these back to the cloud session:
- the full error;
- `& $BB scripts\selfcheck.py --json`;
- `reports\freeze-royalegym.txt` and `reports\freeze-buildabot.txt`;
- for the live runner: its console output, and the session folder name under `reports\vision_play\`.

## Later steps (coming in a later batch)

These need code that is not on the branch yet. Each one gets exact commands here when its batch lands. The IDs refer to `docs/SPARRING_PLAN.md`.

| Phase | What you will run | Waits for |
|---|---|---|
| 2. Engine re-baseline | spawn tables for both card tables; then a fidelity grid over 2,000 validation matches (`tools/rs_fidelity_grid.py`), which picks the RoyaleSim card table and writes `data/engine.json` | E01, E03, E06 |
| 4. Decision-time models, v2 | the decision-time model is section 19 and the v2 model (perception noise in the rows, the ability head, memory-mapped training) section 20. Still to come: v2 rows on the vision class list (`--units vision`) | the simulator's evaluators and RL workers reading `--units vision` rows |
| 5. Screen perception, second pass | label frames from our own friendlies for the cards and forms the KataCR dataset lacks, then score the detector on them (`tools/vision_replay.py --labels`); fit the perception-noise profile used in training (`tools/fit_noise_profile.py`) | V11 (rest), E17 |
| 6. Live friendlies with a stronger model | the same runner (sections 17-18) with a model trained on decision-time rows (section 19's, or section 20's v2 model) and the measured delay D | the results of sections 19 and 20 |
| 7. RL fine-tuning, the live check | the driver is section 21. Still to come: the promoted model against the imitation champion in real friendlies, match by match in turn; the optional search teacher (R05) as a second arm | the first champion from section 21 |
