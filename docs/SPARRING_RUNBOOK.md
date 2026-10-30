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
| 19: second model (decision-time rows) | to do |
| 20: third model (v2 rows, the whole corpus) | to do, after 19 |

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
- `--epochs 60` gives a quicker first model.
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

**6. Templates.** Open the frame, read what it shows, and store it. An example frame: the clock at 1:48, 2 elixir, a hand of Log, Hog Rider (greyed), Skeletons and Fireball (greyed), Ice Spirits next, and tower numbers 3052 and 2101 (theirs) and 3052 and 2731 (ours):

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

Repeat on other frames until this shows `complete` for all three fonts, and `base, greyed` for every card:

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

Send back:
- `tmp\grid_check.png` and `tmp\hud_check.png`;
- the `templates list` output;
- both outputs of step 7.

## 17. Dry-run friendly (the bot watches, no taps)

`scripts\vision_play.py` is the live runner. It runs the whole chain on the screen: capture, HUD, detector, tracker, the model and the decision. By default it taps nothing. It never presses Battle, OK or anything outside the arena, and it never starts or restarts MuMu.

You need:
- the detector pinned (section 15);
- the templates complete (section 16);
- `runs\tokens\live_big_hog\best.pt` (section 14);
- MuMu at 2-4 cores and 3 GB, with its window large and uncovered.

**1. Tests and controls.** After `git pull`:

```powershell
& $RG -m pytest tests/pipeline -q
& $BB -m pytest tests/pipeline -q
& $BB -m scripts.vision.control --seconds 60
```

The last one tries the takeover controls without a battle. It never taps. Press F8 a few times, click the button in the small "Sparring bot" window, and create then delete `C:\hcr\MANUAL`. Each switch prints a line such as `MANUAL -> AUTO (hotkey)`.

**2. Replay the recording first (MuMu not needed).** This runs the runner on the friendly you recorded in section 16. On a recording, taps are always refused.

```powershell
$REC = (Get-ChildItem reports\vision\recordings -Directory | Sort-Object Name | Select-Object -Last 1).FullName
& $BB scripts\vision_play.py --source $REC --no-window
```

It should start battle 1 a second after the clock runs, and print decisions in the status line (below).

**3. The dry-run friendly.** Start the runner in the lobby, then start a friendly yourself and play it with the mouse as usual:

```powershell
& $BB scripts\vision_play.py --seconds 600
```

It first prints its setup. Check these lines:
- `detector` names the pinned `best.pt` on `cuda:0`. `NONE (...)` means no detector: the board stays empty for the model.
- `hud        clock templates ok`. `MISSING` means section 16 is not done, and the bot never decides.
- `taps       off (dry run: no --tap)`.
- `hotkey     on: F8` and `window     on`.

Then it prints a status line every second, for example:

```
[battle 1] tick 1440 | elixir 2.3 | hand Log/HogRider/Skeletons/Fireball next IceSpirits | MANUAL | last Skeletons -> (8, 22) [manual] @1436 | 31 fps, 0 taps
```

| Part | Meaning |
|---|---|
| `[battle 1]` | in battle 1. Outside a battle: `[menu]`, `[end]` (the end screen), or `[battle]` (a battle screen before the clock has run for a second) |
| `tick 1440` | the battle clock in ticks: 20 a second, from 0 at 3:00 (1:48 left is 1440). `OT` in overtime; `-` when the clock is not read |
| `elixir 2.3` | our elixir |
| `hand ... next ...` | the hand, left to right, and the next card. `?` is a slot it cannot read. |
| `MANUAL` | who plays: `MANUAL` (you), `AUTO(dry)` (the bot, without `--tap`), `AUTO(tap)` (the bot taps), `STOPPED` |
| `last ...` | the bot's last play: card, screen tile, what happened, and the tick. The tile is x 0-17 from the left and y 0-31 from the top; our half is y 17-31. `manual` is a suggestion only, `dry` would have played, `tap` tapped, `dropped` means the mode switched first. |
| `31 fps, 0 taps` | frames processed per second, taps sent, and the error count if there are errors |

During the battle:
- Check that the hand, the elixir and the clock match the screen.
- Press F8 once. The status shows `AUTO(dry)` and the decisions become `[dry]`. Nothing is tapped. Press F8 again to go back to `MANUAL`.
- The control window shows the bot's suggestion while you play. Your own plays are logged next to what the bot would have played.
- A `battle 1 over (clock reset)` line in the middle of the battle means a misread clock. Note when it happened.

At the end it prints `battle 1 over (end screen) at tick ...` and then `battle 1: WIN 2-1 (...); record 1-0-0 (W-L-D)`. After 600 s it stops with `stopped: 600 s; ... frames, 0 taps, ...`. Ctrl+C stops it earlier.

Everything goes to `reports\vision_play\vision_play_<date>_<time>\frames.jsonl`, with an image every 20 recorded frames.

**4. Latency.** It reads the log only; MuMu is not needed:

```powershell
& $BB tools\latency_probe.py --last 1 --json reports\vision\latency_dryrun.json
```

It prints the time of each stage (stale, perceive, decide, capture to decision) as median and p90 in ms, the frame rate, and how often the bot decided. `D: not measured` is expected here: D needs taps (section 18).

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

It needs:
- section 19's `runs\tokens\dt_big_hog\best.pt` (the champion);
- section 14's `runs\tokens\live_big_hog\best.pt` (the anchor) and `runs\tokens\live_big\ep01.pt` (the reactive opponent of `rs_eval`);
- about 90 GB free on `C:` (below).

The script checks all three before it starts. Do not change `data\engine.json` or the card table between sections 19 and 20: the league refuses checkpoints made under another engine.

After `git pull`, run the tests of section 12 first. None of this batch's code (v2 rows, compaction, `--memmap`, the v2 model) has run on the PC yet. Paste any failure back before you start the script.

**One script runs everything unattended.** It stops at the first failure and writes everything to `reports\third_model_<time>.log`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\third_model.ps1
```

| Step | What it runs | Time (estimated) |
|---|---|---|
| rows | `tools\rs_rows.py --schema v2 --live-compat --obs-delay 24 --obs-delay-jitter 3 --opp-lag 1 --visibility public --trust v2 --presses on --noise data\noise\default_prior.json --all-matches --wait-keep-at-build 0.25 --workers 8 --out data\rs_rows_v2` | about 3.5 h |
| compact | `tools\rs_rows_compact.py --rows data\rs_rows_v2 --out data\rs_cols_v2 --force` | about 30 min |
| general | `sim.train_tokens --name v2_big --data data\rs_cols_v2 --epochs 2 --memmap --mirror 0.5 --trust v2 --wait-keep 0.25 --d 256 --layers 6 --heads 8 --val-max-rows 500000` | about 3 h |
| hog | the same with `--name v2_hog --epochs 3 --finetune-hog --init runs\tokens\v2_big\best.pt` | about 15 min |
| eval | `tools\rs_eval.py --ckpt runs\tokens\v2_hog\best.pt --opponent-ckpt runs\tokens\live_big\ep01.pt --matches 300 --workers 8 --delay 21` | about 10 min |
| league | `tools\rs_league_eval.py run --candidate v2_hog=runs\tokens\v2_hog\best.pt --champion dt_big_hog=runs\tokens\dt_big_hog\best.pt --anchor live_big_hog=runs\tokens\live_big_hog\best.pt --delay 24 --noise data\noise\default_prior.json --champion-games 1000 --anchor-games 400 --full --workers 6 --log runs\league\third_model.jsonl` | about 1-1.5 h |

That is about 9 hours in all, and anywhere from 7 to 12. Start it in the evening.

None of these times is measured yet. They are the first model's measured times (section 14 and `docs/SPARRING_PLAN.md`, "Results so far"), scaled by the data:
- **rows**: the first model's 33,659 matches took 20 min. Here there are about 7 times the matches, with about 1.5 times the work per match (the trust check, the presses, the noise, the v2 columns).
- **compact**: it reads about 15-20 GB of shards and writes about 50 GB of columns.
- **general**: the first model's 3 epochs took 33 min. Here:
  - there are about 7 times the training rows: 7 times the matches, with the waits thinned to the same 25%;
  - there are 2 epochs instead of 3;
  - a v2 row has about 80 tokens instead of about 66.

  In all, about 5.5 times the work.
- **hog**: every Hog 2.6 match was already in the first model's rows, so this is about the first model's 9.4 min with the larger rows.
- **eval**: one run of 300 matches. The first model's two runs took 13 min.
- **league**: 1,800 games at most 6 at a time:
  - 1,000 of v2_hog against dt_big_hog;
  - 400 of v2_hog against live_big_hog;
  - 400 of dt_big_hog against live_big_hog.

  Each game takes about 10-13 s of one CPU core, as an `rs_eval` match does, with the noise on both seats. `rs_league_eval` runs fewer workers when free memory is short (it keeps 3 GB free, about 1.2 GB per worker).

The training steps print their own time to go. Each prints `ep 0 step 500/S` every 500 steps, where S is the total number of steps. Take the time between two of these lines, multiply by S / 500, and that is how long the whole training takes.

Options:
- `-Workers 6` uses less RAM.
- `-Others 60000` builds rows from every Hog 2.6 match plus 60,000 others drawn at random, instead of every match. That takes about a third of the rows, compact and general time and disk.
- `-Epochs 1` (or 3) changes the general model's epochs. The Hog 2.6 fine-tune always runs 3.
- `-MaxRows 15000000` trains on a sample of that many rows, which halves the general step if training runs long.
- `-Noise <profile.json>` uses another noise profile, such as one fitted by `tools\fit_noise_profile.py`, for both the rows and the league.
- `-From compact` (or `general`, `hog`, `eval`, `league`) restarts at a later step. The script refuses to start from `rows` while `data\rs_rows_v2` holds finished rows: use `-From compact`, or delete the folder to build them again.

**Disk.** These sizes are estimated: about 40 million rows of about 1.27 KB each.

| Folder | Size |
|---|---|
| `data\rs_rows_v2` (the shards, compressed) | about 15-20 GB |
| `data\rs_cols_v2` (the compact store, uncompressed so that it can be memory-mapped) | about 50 GB |
| `runs\tokens\v2_big`, `runs\tokens\v2_hog` | under 0.5 GB |

The script stops before it starts if `C:` has less than 90 GB free (60 GB with `-From compact`); `-SkipDiskCheck` starts anyway.

If space runs short:
- After the compact step, `data\rs_rows_v2` is read only to rebuild the store (`-From compact`). Deleting it gives back 15-20 GB.
- `data\rs_rows_live` and `data\rs_rows_dt` are needed only to retrain the first or second model. No evaluation reads them.

**Reading the result.**
- **eval** is section 19's instrument: the same reactive opponent, the same 300 held-out matches and the same 21-tick delay as its run B (`dt_big_hog`). Compare the two numbers. Each interval is about +-5.6 points, so a difference of less than about 8 points is not conclusive.
- **league** is the real test. It plays at a 24-tick delay with perception noise on every seat, from both seats, on deck pairings and seeds that are the same for every player. Against the champion, each pairing is also played with the decks exchanged. Its `gate:` lines give the decision for `v2_hog`:
  - **PASS**: v2_hog beats `dt_big_hog` (Wilson 95% lower bound above 52% over 1,000 games, i.e. at least 551 points), and it scores no more than 2 points below `dt_big_hog` against `live_big_hog`. Then v2_hog becomes the live model: add `--ckpt runs\tokens\v2_hog\best.pt` to the `vision_play.py` commands of sections 17 and 18. The runner reads the checkpoint's schema and builds v2 rows by itself.
  - **FAIL**, **CYCLE** or **INCOMPLETE**: keep the live model you have.

  The `paired` lines show v2_hog's score minus dt_big_hog's against the same opponent on the same positions, with an interval.
- The report is also written to `runs\league\third_model.report.json`.

If the league step fails, the trained models are kept. Send the log back, and after the fix restart with `-From league`. The league log resumes: games already played are not played again.

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
| 7. RL fine-tuning | the RL driver with gating and resume (`tools/rl/train_loop.py`); runs for days with MuMu closed | R01-R05 |
