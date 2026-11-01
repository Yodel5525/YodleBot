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

## Status on this PC (2026-10-30, evening)

| Step | State |
|---|---|
| 1-4: driver, tools, resources, clone | done; the fork is at `C:\hcr` on `sparring`, with a sparse checkout. The NVIDIA Studio driver gets a clean reinstall the evening of 2026-10-30, and Omen Gaming Hub goes to Balanced, after repeated graphics-driver hangs and a freeze ("If something breaks", "If the PC freezes"); section 1's check and a CUDA check in the GPU venv run again on 2026-10-31 |
| 6: venvs | both exist. `royalegym` has royalesim 0.1.42, numpy 2.5.4, pyarrow 26.0.0. `buildabot` has torch 2.14.1+cu130 and torchvision 0.29.1+cu130. Run `setup_windows.ps1` once anyway: it keeps what is right and adds the rest. |
| 5, 7, 8, 12 | done in the batch 1 first run: setup READY, selfcheck 12/12, tests passing after one fix |
| 9: account and deck | done: Yodle2, Hog 2.6, all base cards |
| 11: IL_Replay | done; 104 files, 1,888 MB, hash-checked |
| 14: first model | done: `runs\tokens\live_big_hog\best.pt` (Hog card 66.8%, tile-within-1 55.7%) |
| 10, 13 | not reported back. Section 16 covers what step 10 was for. |
| 15-16: detector, calibration | the detector is trained: the 2026-10-30 replay of the recorded friendly ran `tmp\yolo\runs\cr_detector\weights\best.pt` (sha256 `7a1acec2c94c...`) on the GPU. The templates read the clock and the elixir on 98.8% of its battle frames |
| 17-18: dry run, first friendly with taps | to do, from 2026-10-31 with the champion `v2_hog_20k` (`--ckpt runs\tokens\v2_hog_20k\best.pt`): the offline rehearsal, the replay check of the opponent-play guards, then the watch-only friendly, which a promoted model needs before any `--tap` |
| 19: second model (decision-time rows) | done 2026-10-30: `dt_big_hog` 29.3% against `live_big_hog`'s 46.0% at a 21-tick delay. The checks of "Why the second model lost" done the same day: the waits with our own play in flight were the cause (`dt_big_hog_np` without them: 80.7%), and against `live_big` epoch 1 both models win far more with no wait bias. Next, from 2026-10-31: `scripts\model_check.ps1 -League -Champion runs\tokens\v2_hog_20k\best.pt -Workers 4` on `dt_big_hog_np`, then the champion's own wait_keep 1.0 league (section 19, "Result (2026-10-30)", Decisions) |
| 20: third model (v2 rows, the whole corpus) | step 1 (`-Others 20000 -Tag 20k`) done 2026-10-30, restarted `-From eval -Workers 4` after the PC froze: `v2_hog_20k` 78.7% at a 21-tick delay, league **PASS** against `live_big_hog` (70.6% over 1,000 games). **`v2_hog_20k` is the champion.** Step 2 (the whole corpus, no `-Tag`, `-Workers 4`) is next, overnight (section 20, "Result of step 1 (2026-10-30)") |
| 21: RL fine-tuning (bench, then days of training) | to do, after section 20 step 2 (or with `-Start runs\tokens\v2_hog_20k\best.pt`) |

## 1. GPU driver

Install the newest NVIDIA Studio (or Game Ready) driver, **version 580 or newer**: the GPU venv's CUDA 13.0 needs an R580 driver, and an older one installs without complaint but leaves `torch.cuda.is_available()` False. Then check it:

```powershell
nvidia-smi
```

It must list `NVIDIA GeForce RTX 4070 SUPER`, with `CUDA Version` 13.0 or higher. The GPU venv uses PyTorch's CUDA 13.0 build (`cu130`); PyTorch's cu128 index stops at torch 2.11.0.

After a driver change, also check that the GPU venv still reaches the GPU (step 6 made the venv):

```powershell
& $BB -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x = torch.randn(4096, 4096, device='cuda'); print('matmul ok', float((x @ x).abs().mean()))"
```

It must print `2.14.1+cu130 13.0 True`, the RTX 4070 SUPER and a `matmul ok` line. `False`, or an error that names the driver, means the driver is too old for CUDA 13.0 or did not install: install it again.

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
- **Since the freeze of 2026-10-30, start engine jobs (rows, evaluation, leagues, RL workers) with 4 workers** ("If something breaks", "If the PC freezes"). Watch Task Manager > Performance > Memory during the first minutes:
  - if "Available" stays above about 7 GB at 4, you can try 6;
  - not 8 until the cause of the freeze is known (the PC froze 6 minutes into an `rs_eval` run at 8);
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

It runs 4 workers by default since the freeze of 2026-10-30 ("If something breaks", "If the PC freezes"); `-Workers 6` only after a watched 4-worker run left more than about 7 GB of memory available. `-From hog` (or `general`, `eval`) restarts at a later step. Send back the log when it finishes. The steps it runs are below, if you prefer to run them one at a time.

**1. Build the training rows.** This takes about 30-60 minutes. It uses every Hog 2.6 match plus 20,000 others:

```powershell
& $RG tools\rs_rows.py --live-compat --others 20000 --workers 4 --out data\rs_rows_live
```

If Task Manager shows less than 2 GB of RAM available, stop it with Ctrl+C and re-run with `--workers 3`.

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
& $RG tools\rs_eval.py --ckpt runs\tokens\live_big_hog\best.pt --opponent-ckpt runs\tokens\live_big\ep01.pt --matches 300 --workers 4
& $RG tools\rs_eval.py --ckpt runs\tokens\live_big_hog\best.pt --opponent-ckpt runs\tokens\live_big\ep01.pt --matches 300 --workers 4 --delay 21
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
- the champion's checkpoint and the `args.json` next to it. Since 2026-10-30 that is `runs\tokens\v2_hog_20k\best.pt` (section 20, "Result of step 1"). **Without `--ckpt` the runner plays `runs\tokens\live_big_hog\best.pt`**, the first model, so every `vision_play.py` command in this section and in section 18 carries `--ckpt runs\tokens\v2_hog_20k\best.pt`. A model that becomes the champion later gets its own run of this section, with its own `--ckpt`, before any `--tap` (section 19, "Result (2026-10-30)", Decisions);
- for step 3, also the detector pinned (section 15), and MuMu at 2-4 cores and 3 GB, with its window large and uncovered.

`v2_hog_20k` is a v2 model (schema `rs_rows/v2`). The runner reads the schema from the checkpoint and builds v2 rows by itself: the opponent's elixir counted from the plays it has seen, our own taps in flight until they land, the spells in flight. The v2 live path has run only in the tests so far, with a stand-in model: steps 2 and 3 are its first runs with a real v2 checkpoint. Our taps in flight (`own_pending`) come only from taps the runner sends, so a dry run never shows one: the first `--tap` friendly (section 18) is the first live test of that input.

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
& $BB scripts\vision_play.py --source $REC --replay-fast --no-window --hotkey none --seed 1 --ckpt runs\tokens\v2_hog_20k\best.pt
```

`--replay-fast` replays every image once, as fast as the chain runs: a few minutes for the 4,414 images. Check the setup lines:
- `model      runs\tokens\v2_hog_20k\best.pt (schema rs_rows/v2, live_compat True, units types); CPU, ... torch threads`. `schema rs_rows/legacy` means `--ckpt` is missing;
- `source     replay of ...capture_20261029_195831 (4414 images, as fast as the chain runs); no device, no adb`;
- `taps       off (the source is a recording)`;
- `decide     live; wait_keep 0.25 (wait bias 1.386); would-play every 10 ticks`. A `WARNING: no args.json` there means the trained share of waits is unknown: copy `args.json` next to `best.pt` (`--tap` refuses without it). A `_wk100` copy shows `wait_keep 1 (wait bias 0.000)`;
- `detector   NONE (...)` until the detector is pinned. Once it is pinned: `... on cuda:0, half True, side joint, 310 classes, 0 unresolved, classes dc897910, imgsz 640`. A `WARNING` line under it means the sides are not joint or some detector names are not in the class list: send it back;
- `hud        clock templates ok`, then `towers     each princess tower's HP label found at battle start (tower skins) and read only where it is; ...`. A `towers     note: the HUD regions ... have no label_* boxes` line means your `hud_regions_pc.json` predates the label boxes: nothing to do, the runner places them from its tower and bar boxes. Never copy boxes from `hud_regions_1080x1920.json` into it (the two files are at different pixel sizes). A `WARNING    the HUD regions ...: label_... px ... from where its tower_* and bar_* boxes put them` line means label boxes in it are wrong: delete the `label_*` entries it names from `hud_regions_pc.json`;
- for a v2 model only: `v2 rows    opponent elixir counted from the plays seen; our taps in flight until they land (--obs-delay 24 ticks); spells in flight`. `--obs-delay 0 ticks` means the checkpoint lost its rows' delay: send it back;
- `opp plays  guards on: an enemy unit is a play once seen for 10 ticks; a lost one may come back under another label; no play from a unit first seen over 1000 mt past the river centre where a reported one could have walked`;
- `tap gate   no AUTO tap until the towers are calibrated (and boxed by the detector) and the mode is classic; else MANUAL (dry run: a line only)`. The gate is explained in section 18, step 2;
- `frame 747x1328` on the first frame. A `WARNING` after it means another frame size than the HUD was calibrated on.

Then, on this recording:
- `battle 1 started (tick 110, mode MANUAL)`, about 8 s into the recording;
- a few seconds later `battle 1: mode classic (first clock 2:56, elixir every 2.7 s)` (the elixir figure can differ by 0.1-0.2 s). There must be no line that starts with `dry run: the tower gate would keep tapping off`;
- status lines with `| would <card> -> (x, y) @<tick>`: what the bot would play now. In MANUAL it decides on nearly every tick and almost every decision is a wait, so on a wait it works this out at most every 10 ticks;
- `battle 1 over (end screen) at tick 44..` (overtime), then `battle 1: WIN 1-0 (end screen); record 1-0-0 (W-L-D)`;
- `stopped: the frame source ended; 4414 frames, 0 taps, results {'win': 1}, errors -`.

Then run it once in real time (about 6.5 minutes, the length of the recording):

```powershell
& $BB scripts\vision_play.py --source $REC --no-window --hotkey none --seed 1 --ckpt runs\tokens\v2_hog_20k\best.pt
& $BB tools\latency_probe.py reports\vision_play\replays --last 1
```

The recording kept an image of every 5th captured frame, so the replay has about 11 frames a second, against about 57 live. Its frame rate in the latency probe is not the live figure. Rules that count frames (the hand's 3 votes, the 3 clock readings before the hand and the towers start, the battle start's 3 frames) also take longer here than live. Use this run for the time of each stage per frame (perceive, decide).

Send back the setup lines, the battle, result and `stopped:` lines of both runs, and the latency probe output. `errors` other than `-` on the `stopped:` line, or a traceback, means something in the chain is broken: send the whole output.

**Once the detector is pinned** (section 15), run the fast rehearsal once more before the live friendly. The `detector` line must then name the pin. This is the first run of the detector, the tracker, the events and the board in the model's state on real frames. The `stopped:` line must again end in `errors -`, without `detect_error`. Your plays in the recording are now seen on the board, so the log gets `human_play` records, and each should carry what the bot would have played:

```powershell
& $BB scripts\vision_play.py --source $REC --replay-fast --no-window --hotkey none --seed 1 --ckpt runs\tokens\v2_hog_20k\best.pt
$S = (Get-ChildItem C:\hcr\reports\vision_play\replays -Directory | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
& $BB -c "import json,sys; h=[r['data'] for r in map(json.loads, open(sys.argv[1], encoding='utf-8')) if r.get('kind')=='human_play']; print(len(h), 'human plays,', sum(1 for d in h if d.get('would')), 'with would')" "$S\frames.jsonl"
```

Expect a count close to the number of cards you played in that battle (the HUD saw about 47), nearly all `with would`. Send back that line and the `stopped:` line.

For a v2 model, also check what its rows were given. Every decision record carries the opponent's elixir as the bot counted it (`opp_elixir`: thousandths and whether the count is exact) and our taps in flight (`own_pending`):

```powershell
& $BB -c "import json,sys; d=[r['data'] for r in map(json.loads, open(sys.argv[1], encoding='utf-8')) if r.get('kind')=='decision']; v=[x['opp_elixir'] for x in d if x.get('opp_elixir')]; print(len(d), 'decisions,', len(v), 'with opp_elixir, min', min([a[0] for a in v] or [-1]), 'max', max([a[0] for a in v] or [-1]), ', exact', sum(1 for a in v if a[1]), ', own_pending', sum(1 for x in d if x.get('own_pending')))" "$S\frames.jsonl"
```

Expect nearly every decision with `opp_elixir`, its values between 0 and 10000, and `own_pending 0`: a replay sends no taps, and only sent taps are in flight. No `opp_elixir` at all, or `own_pending` above 0, means the v2 inputs are wired wrong: send the line back.

**The opponent-play guards on the recording** (commits e2188e5, 522c70c and 0f3a56e). First check the pin. `detector info` prints JSON, so every backslash of the `"weights"` path shows twice:

```powershell
& $BB -m scripts.vision.detector info
```

`"sha256"` must start `7a1acec2c94c`, with `"side_mode": "joint"` and an empty `"unresolved_names"`. If nothing is pinned, pin it: `& $BB -m scripts.vision.detector pin C:\hcr\tmp\yolo\runs\cr_detector\weights\best.pt`. Then replay the recording through the perception alone:

```powershell
& $BB tools\vision_replay.py $REC
```

- Its `opp plays  guards on: ...` line must read as above.
- `events: N opponent plays {...}`: commit 522c70c measured about 46 plays of 18 cards (155 elixir) with a re-implementation of the rules, so the real code can differ a little: roughly 40-55 is in line. The run of 2026-10-30, before the guards, had 103 of 27 cards; about 103 means the guards did not run.
- A new line under it: `  enemy tracks: N plays (N spells), merged ..., children ..., re-found ... (... under another label); held back: young dropped ..., fragments ..., late ...`.
- Optional: the same with `--legacy-opp-plays` replays the rules as they were before the guards, for comparison: near the old 103 (the newer spawner table may explain a few more children).

Commit 0f3a56e regenerated `data\vision\spawn_children.json` in the cloud. Check that the PC's RoyaleSim gives the same table. This writes to `tmp\`, never over the committed file:

```powershell
& $RG tools\vision_spawn_children.py --out tmp\spawn_children_pc.json
& $RG -c "import json,sys; a=json.load(open(sys.argv[1], encoding='utf-8')); b=json.load(open(sys.argv[2], encoding='utf-8')); print('same' if a == b else 'DIFFERENT: ' + ', '.join(sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))))" data\vision\spawn_children.json tmp\spawn_children_pc.json
```

The first prints `tmp\spawn_children_pc.json: 145 cards, 114 spawning units, 8 spawning spells, 1 with forms (from cards-160402017-20261019.1, cards-15535.1)`, the second `same`. `DIFFERENT: source` alone is harmless (it says only where the tables were read from). Anything else: send the line, and do not copy the file over the committed one. Send back everything from the `vision_replay` line to `summary -> ...` and both lines of the table check.

**The towers on four recordings: tower skins, arenas and the tap gate.** No MuMu needed. Do this once now, and again before the first `--tap` friendly against a tower skin or an arena the bot has not seen. You pick a random tower skin every battle, your clanmates use many, and the arena changes with the season and the mode. So at the start of every battle the runner looks for each princess tower's HP label (the gold crown badge and the outlined bar next to it), wherever the skin puts it, and from then on reads that tower only where its label is. A tower counts as destroyed only after its label has been gone for 2.5 s; a unit in front of the label, or the pink tint over the enemy half while a card is selected, does not count. A king tower is never counted as destroyed during a battle: its crowns come from the end screen.

Each replay below runs the pinned detector, so close MuMu and the browser first and run one replay at a time. Type or paste the commands one line at a time:

```powershell
$R0 = 'C:\hcr\reports\vision\recordings\capture_20261029_195831'
$DRY = (Get-ChildItem C:\hcr\reports\vision_play -Directory -Filter 'vision_play_20261031_*' | Sort-Object Name | Select-Object -First 1).FullName
$A1 = 'C:\hcr\reports\vision\autolabel\20261031_A_1\recording'
$A2 = 'C:\hcr\reports\vision\autolabel\20261031_A_2\recording'
Test-Path "$R0\frames.jsonl"
Test-Path "$DRY\frames.jsonl"
Test-Path "$A1\frames.jsonl"
Test-Path "$A2\frames.jsonl"
```

`$R0` is Round 0 (forest arena, default towers, end screen 1-0), `$DRY` the 10-31 dry run (the chicken tower skin on the enemy side, sand arena, end screen 1-0; it kept an image of every 20th frame), `$A1` and `$A2` the two 10-31 deck A friendlies. All four `Test-Path` lines must print `True`. If the `$DRY` one prints `False`, list the sessions with `Get-ChildItem C:\hcr\reports\vision_play -Directory`, then set `$DRY` to the full path of the one started at 11:21 on 10-31, for example `$DRY = 'C:\hcr\reports\vision_play\vision_play_20261031_112148'`.

Replay the four recordings through the perception, then audit the towers:

```powershell
& $BB tools\vision_replay.py $R0 --out reports\vision\replays\skin
& $BB tools\vision_replay.py $DRY --out reports\vision\replays\skin
& $BB tools\vision_replay.py $A1 --out reports\vision\replays\skin
& $BB tools\vision_replay.py $A2 --out reports\vision\replays\skin
& $BB tools\tower_audit.py reports\vision\replays\skin --all --sheet reports\vision\replays\skin\sheets
& $BB tools\tower_audit.py reports\vision\replays\skin --all --check calib,destroy,king,crowns
$LASTEXITCODE
```

What to look for:
- The audit prints one block per session and battle. Under `calib`, all four princess towers must be calibrated in every battle, within about 3 s of the battle clock starting (longer on `$DRY`, which kept an image of every 20th frame, about 0.8 s apart). On `$R0` every label shift is under 3 px. On `$DRY` the enemy labels sit about 36 px up (`dv` about -36; the shifts are in 1080x1920 pixels) and ours near 0, and the enemy towers never read below 0.95 before their first real hit (`damage`).
- Under `zero`, every destroyed tower has evidence `label`. On `$R0` and `$DRY` no tower falls during the battle, so `zero` shows `-` for all six: both were won by an enemy tower that fell in overtime as the battle ended, with its label still showing on the last battle frame, and the HUD never counts the tower that ends a battle (it would need its label gone for 2.5 s). No king reads 0 before the end screen.
- `crowns`: the HUD's crowns against the end screen's. On `$R0` and `$DRY`: `crowns: HUD 0-0, end screen 1-0 (... frames): explained (own crowns: the battle ended as enemy_left or enemy_right fell)`. The end screen may be ahead of the HUD only by the tower that ended the battle (3 crowns for a king). `DIFFER` means the HUD missed or invented a fall: send it back. On `$A1` and `$A2`, `agree` or `explained`.
- The last command ends with `check calib: PASS`, `check destroy: PASS`, `check king: PASS` and `check crowns: PASS`, and `$LASTEXITCODE` prints `0`. `1` means a check failed: send back the whole audit output.
- Under `boxes`, each tower starts with `body NN% (first N.Ns/NNf NN%)`: the share of frames with a tower box at that tower by the tap gate's own rule, over the battle and over the frames the gate decides on (`first`: from the battle's start until all four labels are calibrated and, in a `vision_play` session, the mode verdict is in; at least 1.5 s and 10 frames). On `$R0` and `$DRY` both are 100% for all six towers; under 50% in that window is what turns the tap gate off. The share after `any` counts every box there, units too, so it can be higher. A `vision_play` session also shows its `tap gate:` line (open or closed, and the shares it decided on); when the two differ, the `tap gate:` line is what the gate saw.
- The sheets are PNG files in `reports\vision\replays\skin\sheets`. Open a few: the boxes must sit on the towers' HP labels (green while the tower stands). Send any sheet where they do not.

Then rehearse the whole runner on the same four recordings. It never taps on a recording; this shows what the tap gate would do:

```powershell
& $BB scripts\vision_play.py --source $R0 --replay-fast --no-window --hotkey none --seed 1 --ckpt runs\tokens\v2_hog_20k\best.pt
& $BB scripts\vision_play.py --source $DRY --replay-fast --no-window --hotkey none --seed 1 --ckpt runs\tokens\v2_hog_20k\best.pt
& $BB scripts\vision_play.py --source $A1 --replay-fast --no-window --hotkey none --seed 1 --ckpt runs\tokens\v2_hog_20k\best.pt
& $BB scripts\vision_play.py --source $A2 --replay-fast --no-window --hotkey none --seed 1 --ckpt runs\tokens\v2_hog_20k\best.pt
```

In each run:
- after every `battle N started ...` line, a `battle N: mode classic (first clock 2:5x, elixir every 2.x s)` line;
- no line that starts with `dry run: the tower gate would keep tapping off`;
- the status lines' `would` plays as in your last rehearsal of that recording: the tap gate changes no decision in a dry run (the tower fixes can move a few plays after a tower falls);
- on `$R0` and `$DRY`: `battle 1: WIN 1-0 (end screen); ...`, with no `HUD read` in it.

A recording that only started after 2:50 on the battle clock gives `mode unknown (first clock 2:3x ...)` and the dry-run gate line: the check is working, the recording started too late. Say so when you send the lines back.

Send back the audit output, the four `stopped:` lines and every `mode ...` and `tower gate` line.

**3. The dry-run friendly.** Start the runner in the lobby, then start a friendly yourself and play it with the mouse as usual:

```powershell
& $BB scripts\vision_play.py --seconds 600 --ckpt runs\tokens\v2_hog_20k\best.pt
```

**Do not click or select text in the PowerShell window while the runner runs.** With QuickEdit on (the default), a selection pauses the console, and the runner then stops reading the screen until the selection ends. Switch modes with F8 or the control window. Copy the lines to send back from the window after the `stopped:` line, or read them from the session's `frames.jsonl`.

It first prints its setup. Check these lines:
- `model      runs\tokens\v2_hog_20k\best.pt (schema rs_rows/v2, ...)`: the champion, not the default `live_big_hog`.
- `decide     live; wait_keep 0.25 ...`, with no `WARNING`.
- `v2 rows    ... (--obs-delay 24 ticks) ...` and `opp plays  guards on: ...`, as in step 2.
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
| `gate wait`, `gate OFF`, `gate overridden` | shown only when the tap gate (section 18, step 2) is not open: still checking this battle's towers and mode (`gate wait`, the first few seconds), a check failed and the bot will not tap this battle (`gate OFF`), or you pressed F8 after that (`gate overridden`) |
| `last ...` | the bot's last play: card, screen tile, what happened, and the tick. The tile is x 0-17 from the left and y 0-31 from the top; our half is y 17-31. `manual` is a suggestion only, `dry` would have played, `tap` tapped, `dropped` means the mode switched first. |
| `would ...` | while the bot does not tap: what it would play now (its best card at its best tile), worked out on a wait at most every 10 ticks. It is never tapped. |
| `31 fps, 0 taps` | frames processed per second, taps sent, and the error count if there are errors |

During the battle:
- Check that the hand, the elixir and the clock match the screen. After you play two cards quickly, both slots must show the right new cards within about a second.
- About 3 s after `battle 1 started`, the runner prints `battle 1: mode classic (first clock 3:00, elixir every 2.8 s)`: it measured how fast your elixir fills (a card you play then only makes it take a little longer). If it prints a line starting `dry run: the tower gate would keep tapping off this battle (...)` instead, a `--tap` run would not have let the bot tap in this battle (section 18, step 2, says what each reason means): note your tower skin, the opponent's and the arena, and send the line back.
- Press F8 once. The status shows `AUTO(dry)` and the decisions become `[dry]`. Nothing is tapped. Press F8 again to go back to `MANUAL`.
- The control window shows the bot's suggestion while you play, and `would play ...` while it waits. Each of your plays is logged (`human_play`) next to the bot's suggestion at that moment and the newest play it would have made with your card still in its hand (`would`, `would_same_card`, `would_tile_dist`). The card's slot empties as soon as you start dragging, so a would-play made during the drag could not have chosen your card and is passed over.
- A `battle 1 over (clock reset)` line in the middle of the battle means a misread clock. Note when it happened. About a second later the runner follows the same battle again (`battle 2 started ...; joined in the middle`), with the towers it had read.
- Keep an eye on Task Manager > Performance > GPU: the detector runs on every frame. The GPU temperature should stay under about 80-85 °C. Above that, stop the runner (Ctrl+C or the STOP file) and send back the reading.
- MuMu's rendering and the detector load the GPU together here, the PC's known failure ("If the PC freezes"). If the screen goes black or flickers, or MuMu's window freezes for a few seconds, stop the runner (Ctrl+C), note the time, look in Reliability Monitor (`perfmon /rel`) for a LiveKernelEvent at that time, send it back, and play no `--tap` friendly (section 18) that day.

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
- a clean dry run (section 17) with the same `--ckpt`: clock, hand and elixir read right, sensible suggestions, no errors. Since 2026-10-30 the checkpoint is the champion `runs\tokens\v2_hog_20k\best.pt`. Without `--ckpt` the runner would tap with `live_big_hog`, the first model;
- MuMu at 1080x1920 with no override size (step 7). The runner checks `wm size` and refuses another size;
- no `C:\hcr\STOP` file. The runner will not start while it exists;
- the tower checks of section 17, step 2 ("The towers on four recordings"), passed: `$LASTEXITCODE` `0`, and no tower gate line in the four rehearsals;
- against a clanmate whose tower skin the bot has not met yet: first one friendly without `--tap` (section 17, step 3) against that skin, with no `tower gate` line. Only then a `--tap` friendly against it.

**1. Start it in MANUAL.** In the lobby:

```powershell
& $BB scripts\vision_play.py --tap --battles 1 --ckpt runs\tokens\v2_hog_20k\best.pt
```

For a v2 model this is also the first live test of our taps in flight (`own_pending`), because a dry run sends no taps. A play decision made while a bot tap is still in flight carries that card in its `own_pending` (the session's `frames.jsonl`, `decision` records; waits are not logged). Your own plays while you are in MANUAL are never in it: only the bot's taps are. So hand over with F8 only when none of your own cards is still on its way (about 2 s after your last play): until it lands, the model would see your play in flight with nothing in `own_pending`, a state its training rows never had (there every own play in flight is in `own_pending`).

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

**The tap gate.** At the start of every battle the bot does not tap, even in AUTO, until three checks pass:
- it has found the HP labels of all four princess towers (tower skins move them);
- the detector boxes all six towers where the arena puts them (it looks at where each tower's box is centred, so a skin that makes a tower taller or lower still passes; it counts every frame until the other two checks are done, at least 1.5 s);
- the battle is a classic 1v1: the clock started at 3:00 (its first reading shows 2:50-3:00) and your elixir fills at one every 2.8 s.

This takes about 3 s. Until then the status line shows `| gate wait` and a play the bot decides is not sent (`[skipped]`). Normally the gate then opens without a line, and the bot plays. If a check fails, the runner prints one line and switches to MANUAL:

| Line | What it means |
|---|---|
| `towers not calibrated (enemy_left, enemy_right): tapping off this battle; play yourself (switched to MANUAL; F8 hands back to the bot anyway, for this battle only)` | the bot could not find those towers' HP labels (a skin it cannot read), so it would not know when they fall |
| `towers not boxed by the detector (enemy_king 20% of 31 frames): ...` | the detector did not see that tower in its place on half the frames from the battle's start until the labels and the mode were checked (about 2-3 s; at least 1.5 s and 10 frames): a skin it does not box at all, or another arena layout. `tools\tower_audit.py` on that session shows the gate's view: its `tap gate:` line, and `body NN% (first N.Ns/NNf NN%)` under `boxes` for each tower |
| `mode unknown (regen 1.4 s): ...` | your elixir filled faster than classic (double elixir 1.4 s, triple 0.9 s): another game mode |
| `mode unknown (first clock 2:00 (tick 1210), not 2:5x-3:00): ...` | the clock did not start at 3:00: another mode, or the runner was started in the middle of the battle |

When you see one: play the battle yourself and send the line back, with your tower skin, the opponent's and the arena. The control window shows it too, after the mode: `MANUAL: you play  |  tap gate OFF this battle (...)`. F8 hands back to the bot anyway, for the rest of that battle only. It prints `tower gate overridden (hotkey): AUTO taps for the rest of battle 1` and logs `tap_gate_override`. Do that only if the status line looks right (the hand, the elixir, `towers them 3052 3052 us 3052 3052`). The next battle is checked again.

If you were still in MANUAL when the line came (as in step 2, before your first F8), the line says `MANUAL already; your next F8 keeps it MANUAL, the one after hands back to the bot anyway`. So your usual F8 to hand over does not lift the gate: the runner switches straight back to MANUAL and prints the line again, ending `(switched back to MANUAL; F8 again hands back to the bot anyway, for this battle only)`. Only a second F8 after that overrides the gate. When the runner follows the same battle again after losing it for a moment (`battle 2 started ...; joined in the middle`), it keeps that battle's gate: no new check, no new line.

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
- Tower skins move a tower's HP label. At the start of every battle the runner finds each princess tower's label (section 17, step 2, "The towers on four recordings") and then reads that tower only where its label is. A tower counts as destroyed only after its label has been gone for 2.5 s (a unit in front of the label, or a card you hold, pauses that count), and a king never during the battle (its crowns come from the end screen). A tower that comes back into view whole for a second after that stands again: `towers misread: its label is back; revived enemy_left -> 0-0`, which means a unit the detector missed had hidden it. A line such as `enemy labels 36 px above calibration (tower skin; from the search)` is that working. Any other `towers misread` line, or a result line ending in `HUD read 2-0` (the result itself comes from the end screen), means the HUD read a tower wrong: send it back.
- A tap the game did not take (the card still in its slot 1.5 s later) prints `tap did nothing: <card> -> (x, y) at tick N ...`. The bot then keeps that card off that spot: the enemy half of that lane for the rest of the battle, or 2 tiles around it for 30 s on our side. Once such a refusal is final the status line shows `| refused 1/3`. After 3 in a battle the runner prints `REFUSED TAPS: 3 taps this battle did nothing in the game (...). Switched to MANUAL ...` and stops tapping (if you pressed F8 yourself before that, it prints `... Already MANUAL ...` instead). Play yourself, and press F8 to hand back only once the status line looks right: the hand, the elixir, and `towers them 3052 3052 us 3052 3052 crowns 0-0` (the princess towers' HP as the bot reads them, theirs then ours, left to right, and the crowns ours-theirs). A `0` there for a tower that is still standing is the misread that sends troops to the enemy side: keep playing yourself and send it back. A `0` the HUD could not confirm (its label was never found, or it fell by the old rules) is followed by `; enemy_left 0 (unconfirmed)`: the bot keeps that tower's pocket shut and plays as if it stands, so it is no reason to take over; send the line back. After you hand back, the count starts again from 0. A `landed after all` line is only a slow tap. Send all of these lines back. In a dry run nothing is tapped, so these lines are not shown; instead, an AUTO dry run prints `dry run: the refused-tap guard would switch to MANUAL now ...` once per battle, which is expected.
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
- The mode carries over from one battle to the next. If it is AUTO when a friendly starts, the bot plays it (after the tap gate's first seconds). Press F8 between battles if you want to start the next one yourself. An F8 that overrode the tap gate counts for its own battle only: the next battle is checked again.

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
| rows | `tools\rs_rows.py --live-compat --obs-delay 24 --obs-delay-jitter 3 --opp-lag 1 --visibility public --trust v2 --presses off --others 20000 --workers 4 --out data\rs_rows_dt` | about 25 min (20 min) |
| general | `sim.train_tokens --name dt_big`, d256, 6 layers, 8 heads, 3 epochs, `--wait-keep 0.25 --trust v2` | about 35 min (33 min) |
| hog | the same with `--name dt_big_hog --finetune-hog --init runs\tokens\dt_big\best.pt` | about 10 min (9.4 min) |
| eval | three runs of `tools\rs_eval.py`, 300 matches each, against `live_big\ep01.pt`: A `live_big_hog` at `--delay 21`, B `dt_big_hog` at `--delay 21`, C `dt_big_hog` with no delay | about 20 min (two runs: 13 min) |

That is about 1.5 hours in all (at 8 workers, the default when it ran). Options:
- `-Workers`: 4 workers by default since the freeze of 2026-10-30 ("If something breaks", "If the PC freezes"); `-Workers 6` only after a watched 4-worker run left more than about 7 GB of memory available.
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

The optional checks are `fullablation` (about 50 minutes), `trust` and `ticks` (about 30 minutes together): add for example `-Steps fullablation`, or `-Steps all` for everything (about 2.5 hours). `-Steps ablation` runs one check alone. Its `rs_eval` runs use 4 workers by default since the freeze of 2026-10-30 ("If something breaks"); `-Workers 6` only after a watched 4-worker run left more than about 7 GB of memory available. The default, `ablation` and `all` retrain `runs\tokens\dt_big_hog_np` in place: do not run them while that model or its `_wk100` copy is the champion or in the live runner.

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
& $RG tools\rs_eval.py --ckpt runs\tokens\live_big_hog\best.pt --opponent-ckpt $OPP --matches 300 --workers 4 --delay 21 --decide live
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog\best.pt --opponent-ckpt $OPP --matches 300 --workers 4 --delay 21 --decide live
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog\best.pt --opponent-ckpt $OPP --matches 300 --workers 4 --delay 24
& $BB -m sim.train_tokens --name dt_big_hog_np --data data\rs_rows_dt --wait-keep 0.25 --epochs 3 --d 256 --layers 6 --heads 8 --finetune-hog --init runs\tokens\dt_big\best.pt --trust v2 --drop-pending-waits
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog_np\best.pt --opponent-ckpt $OPP --matches 300 --workers 4 --delay 21
& $BB -m sim.train_tokens --name dt_big_np --data data\rs_rows_dt --wait-keep 0.25 --epochs 3 --d 256 --layers 6 --heads 8 --trust v2 --drop-pending-waits
& $BB -m sim.train_tokens --name dt_big_np_hog --data data\rs_rows_dt --wait-keep 0.25 --epochs 3 --d 256 --layers 6 --heads 8 --finetune-hog --init runs\tokens\dt_big_np\best.pt --trust v2 --drop-pending-waits
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_np_hog\best.pt --opponent-ckpt $OPP --matches 300 --workers 4 --delay 21
& $BB -m sim.train_tokens --name dt_big_hog_tl --data data\rs_rows_dt --wait-keep 0.25 --epochs 3 --d 256 --layers 6 --heads 8 --finetune-hog --init runs\tokens\dt_big\best.pt --trust legacy
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog_tl\best.pt --opponent-ckpt $OPP --matches 300 --workers 4 --delay 21
& $RG tools\rs_eval.py --pro --matches 300 --workers 4
& $RG tools\rs_eval.py --pro --matches 300 --workers 4 --delay 21
& $RG tools\rs_dt_diag.py ghost --delay 21
```

#### Result (2026-10-30)

The checks ran on 2026-10-30, in about 80 minutes (`reports\dt_diagnose_20261030_165702.log`). Every `rs_eval` run: 300 held-out matches against `runs\tokens\live_big\ep01.pt`, 8 workers.

| Run | wait_keep | Rule | Delay | Win rate [95% CI] | Crown difference | Plays per match |
|---|---|---|---|---|---|---|
| A `live_big_hog` (`second_model.ps1`) | 0.25 | plain | 21 | 46.0% [40.4, 51.7] | -0.34 | 33.3 |
| B `dt_big_hog` (`second_model.ps1`) | 0.25 | plain | 21 | 29.3% [24.5, 34.7] | -1.02 | 28.7 |
| waitbias: `dt_big_hog` | 0.5 | plain | 21 | 63.3% [57.7, 68.6] | +0.30 | 38.8 |
| waitbias: `dt_big_hog` | 1.0 | plain | 21 | 68.0% [62.5, 73.0] | +0.51 | 43.3 |
| waitbias: `live_big_hog` (the control) | 1.0 | plain | 21 | 65.3% [59.8, 70.5] | +0.47 | 41.8 |
| live: `live_big_hog` | 0.25 | live | 21 | 51.7% [46.0, 57.3] | -0.15 | 38.3 |
| live: `dt_big_hog` | 0.25 | live | 21 | 45.7% [40.1, 51.3] | -0.29 | 40.5 |
| offset: `dt_big_hog` | 0.25 | plain | 24 | 29.0% [24.2, 34.4] | -0.96 | 28.0 |
| ablation: `dt_big_hog_np` | 0.25 | plain | 21 | **80.7%** [75.8, 84.7], 1 draw | +0.79 | 37.4 |

How to read the columns:
- **wait_keep**: `rs_eval` and the live runner add log(1 / wait_keep) to the model's "wait" score, with wait_keep read from the `args.json` next to the checkpoint. Both models trained with 0.25, which adds 1.39; 0.5 adds 0.69 and 1.0 adds nothing. A larger wait_keep means fewer waits and more plays.
- **Rule**: plain is `rs_eval`'s own decision rule. live is `--decide live`, the live runner's rule: no wait at a full bar, no second tap of a card in flight. It applies to both seats, the opponent too, so live rows compare only with live rows.

The headroom check:
- On `rs_rows_dt`, the `< -7` share is 0.019, about 0. So the recorded tick is the tick a play ran, as assumed. The `< 0` share is 0.038.
- On `rs_rows_live`, no row is below 0, 0.2% are below 6 ticks and 1.7% below 15. That is the same answer. (Its folder has no manifest because it comes from an older run; that is harmless.)
- Waits with our own play in flight are 19.8% of the wait rows, and 24.2% in the trusted Hog 2.6 rows.

The ablation's fine-tune left out 105,736 of those waits. Its validation figures (Hog card 63.4%, tile-within-1 54.0%, gate 57.5%) are those of `dt_big_hog` (63.3%, 54.0%), yet it wins 80.7% instead of 29.3%. Leaving those waits out also thins the waits by about a quarter, which is worth about 0.3 of the 1.39 bias (like wait_keep 0.33 instead of 0.25). But wait_keep 0.5 takes away 0.69 of the bias and reaches only 63.3%, with more plays (38.8 against 37.4), so the ablation's gain is not the play rate.

**Reading:**
1. The leading explanation holds strongly. Without the waits in flight the same recipe goes from 29.3% to 80.7%. The other checks agree:
   - waitbias: at wait_keep 1.0, `dt_big_hog` gains 38.7 points (29.3% to 68.0%), twice the control's 19.3 (46.0% to 65.3%), about 3.5 standard errors apart. Its plays go from 28.7 to 43.3 a match, the control's from 33.3 to 41.8, and the two models end level (68.0% against 65.3%). That is the table's first outcome: a lazy gate.
   - live: under the live rule the gap between the two models shrinks from 16.7 points (plain) to 6.0 (51.7% against 45.7%), within the noise. The live rule's in-flight mask recovers most of what `dt_big_hog` lost. The gains themselves (16.4 points against 5.7) differ by about 2 standard errors: suggestive, not conclusive.
2. A second effect, which both models share: against `live_big` epoch 1 in `rs_eval` (no noise, 300 matches), both win far more with no wait bias. wait_keep 1.0 lifts the control `live_big_hog` by 19 points too (46.0% to 65.3%). This is one result against one opponent, and that opponent plays with the 0.25 bias itself, so "a busier model beats a passive 0.25 one" fits it as well as "0.25 is too high". The wait_keep 1.0 runs also play 41.8-43.3 cards a match, above the pros' 35-40. The third model reads `own_pending` (our own plays in flight), which addresses 1 but not 2.
3. Ruled out: what the recorded tick means (headroom), and the 24-tick lead against the 21-tick delay (offset: 29.0% at 24 ticks against 29.3% at 21).

**Caveats:**
- There is one opponent, `live_big` epoch 1, reactive. A model can beat one opponent's habits without being better in general.
- There is no perception noise, and each run is 300 matches, so each interval is about +-5.6 points.
- The league (`tools\rs_league_eval.py`) is the stricter instrument. It plays paired games on the same deals, from both seats, at a 24-tick delay, with the noise prior on every seat and the live rule. Its gate (plan L03; `GateConfig` in that file) needs 1,000 games against the champion with a Wilson 95% lower bound above 52%, that is at least 551 points of 1,000 (a win is 1, a draw 0.5), and no more than 2 points below the champion against `live_big` over 400 games.
- In the league the champion and `live_big` play at wait_keep 0.25 too (unless the champion is a `_wk100` copy). So a wait_keep 1.0 copy that passes has shown that it beats 0.25 models, not that 1.0 is right in general. Real friendlies are the final check (plan, "Evaluation").
- This log's `== summary` block leaves out the wait_keep 0.5 line and repeats some `engine:` lines. `Invoke-WaitBias`'s `-Keep` parameter ('0.5', '1.0') stood in for the summary's pattern: a PowerShell function reads its callers' variables, case ignored. The runs are not affected, and the lines above the summary are complete. `dt_diagnose.ps1` is fixed since 2026-10-30.

**Decisions** (updated the evening of 2026-10-30, after the third model's league):
- Done the night of 2026-10-30: section 20 step 1, `third_model.ps1 -Others 20000 -Tag 20k`, restarted `-From eval -Workers 4` after the PC froze in its first `rs_eval` run at 8 workers. Its league gave `v2_hog_20k: PASS (vs champion live_big_hog)`: 70.6% over 1,000 games (section 20, "Result of step 1 (2026-10-30)").
- **The champion** is the last model whose league said PASS: **since 2026-10-30, `v2_hog_20k`** (`runs\tokens\v2_hog_20k\best.pt`). Every candidate plays the current champion, never a former one, so that two models never both replace the champion without meeting. Since the update of 2026-10-30 `model_check.ps1` and `third_model.ps1` default `-Champion` to `runs\tokens\v2_hog_20k\best.pt`, and every engine script defaults to 4 workers. The commands below still give both, which changes nothing. Once another model passes, give its `best.pt` as `-Champion`. A `_wk100` copy of a model is a candidate like any other, the champion's own copy too: the copy is another file, so another player.
- The live runner moves from `live_big_hog` to `v2_hog_20k` once `v2_hog_20k`'s section 17 dry run is clean ("Switching the live runner", below). 80.7% for `dt_big_hog_np` against one opponent without noise is not a gate pass, so it does not go into the runner.
- **From 2026-10-31, whenever the PC is free** (not during `third_model.ps1`, detector training or a friendly), with MuMu and the browser closed, at `-Workers 4` (the PC froze at 8 `rs_eval` workers on 2026-10-30, "If something breaks"), in this order. Both use `runs\league\model_check.jsonl`.
  1. `scripts\model_check.ps1 -League -Champion runs\tokens\v2_hog_20k\best.pt -Workers 4`: `dt_big_hog_np` at its own wait_keep 0.25. First 8 `rs_eval` runs (wait_keep own and 1.0, rules plain and live, each with `v2_hog_20k` under the same setting as the control; the champion's 1.0 control is the copy `runs\tokens\v2_hog_20k_wk100`), then its league against `v2_hog_20k`: up to 1,800 games, because the champion's 400 games against the anchor are not yet in this log (1,400 if an earlier run already played `dt_big_hog_np`'s anchor games there). About 2.5 hours. On a PASS, the champion is `runs\tokens\dt_big_hog_np\best.pt`.
     - Three of its runs repeat runs of 2026-10-30 with the same flags, and `rs_eval` gives the same result at any worker count (`live_big_hog` under the live rule: 155/300, -0.15, 38.3 plays at 8 workers and at 4). So they check for free that the update changed nothing: `dt_big_hog_np wk own(0.25) plain d21` must read 242/300 with 1 draw, `v2_hog_20k wk own(0.25) plain d21` 236/300, and `v2_hog_20k wk own(0.25) live d21` 209/300. Any other count means a checkpoint, the simulator or the held-out matches changed: send it back.
     - A run stopped during its `rs_eval` runs starts over with the same command (they cannot resume). One stopped in its league step goes on with `-LeagueOnly` added: the league plays only the games its log lacks.
  2. The champion's own wait bias: `scripts\model_check.ps1 -Ckpt runs\tokens\v2_hog_20k\best.pt -LeagueOnly -LeagueKeep 1.0 -Champion runs\tokens\v2_hog_20k\best.pt -Workers 4`. The candidate is the copy `v2_hog_20k_wk100` (the same `best.pt`, with an `args.json` of wait_keep 1.0; item 1 made it), and it plays `v2_hog_20k` at 0.25. 1,400 games, because the champion's anchor games from item 1 are reused: about 51 minutes. On a PASS, the champion is `runs\tokens\v2_hog_20k_wk100\best.pt`. If item 1 passed, give `-Champion runs\tokens\dt_big_hog_np\best.pt` instead.
  3. Dropped: `live_big_hog_wk100`'s league. `live_big_hog` scored 29.4% against `v2_hog_20k`, and its wait_keep 1.0 copy would need at least 55.1% (551 points of 1,000). The bias gave it 19 points in `rs_eval` against a 0.25 opponent, and under the live rule `v2_hog_20k` leads it by 18 points (69.7% against 51.7%). Even a pass would put a legacy model above the v2 line that RL starts from. If it is wanted later: `scripts\model_check.ps1 -Ckpt runs\tokens\live_big_hog\best.pt -LeagueOnly -LeagueKeep 1.0 -Champion <the champion's best.pt> -Workers 4`, about 51 minutes.
  4. `dt_big_hog_np_wk100`'s league (`scripts\model_check.ps1 -LeagueOnly -LeagueKeep 1.0 -Champion <the champion's best.pt> -Workers 4`, about 51 minutes): only if item 1 passed, or if item 1's table shows `dt_big_hog_np wk 1.0 live` at least 8 points above its `v2_hog_20k wk 1.0 live` control.

  Measured at 4 workers on 2026-10-30: the one `rs_eval` run timed at both counts (`live_big_hog`, live rule) took 561 s at 4 workers and 582 s at 8, with identical results; the four `v2_hog_20k` runs at 4 took 10.3-13.3 minutes each, and no v2 run has been timed at 8. A league game took about 2.2 s (1,000 games in 37 minutes, so 1,400 in about 51 and 1,800 in about 66). `model_check.ps1`'s `plan:` line uses these times since the update of 2026-10-30 (before, it scaled 10 minutes a run by 8 / workers and printed about twice the real time at 4).
- **Switching the live runner** to the champion, once that is not `live_big_hog` (`v2_hog_20k` since 2026-10-30):
  - Repeat the section 17 dry run with `--ckpt <the champion's best.pt>` added to its `vision_play.py` command (for `v2_hog_20k`: `--ckpt runs\tokens\v2_hog_20k\best.pt`, steps 2 and 3), and check its `decide` start-up line: `wait_keep 0.25 (wait bias 1.386)` for a model at its own wait_keep (0.25 for every model so far), `wait_keep 1 (wait bias 0.000)` for a `_wk100` copy. Only then add the same `--ckpt` to section 18's `--tap` friendlies.
  - The live runner reads wait_keep from the `args.json` next to the checkpoint. So a model at 1.0 is the copy that `model_check.ps1` makes, `runs\tokens\<name>_wk100\`: the same `best.pt`, with an `args.json` that says `"wait_keep": "1.0"`. Never edit the `args.json` in a trained model's own folder.
  - Do not retrain into the live model's folder: no `dt_diagnose.ps1` default run, `-Steps ablation` or `-Steps all` while `dt_big_hog_np` or its copy is in use, and no `third_model.ps1 -Tag 20k` run (the script refuses a name the league log already holds). `model_check.ps1` never replaces an existing copy whose source has changed; it stops and says so.
  - If the champion's dry run shows a fault: keep `live_big_hog` at 0.25 in the runner until it is fixed. The leagues keep `v2_hog_20k` as the champion.
- Section 20 step 2 (the whole corpus) needed `v2_hog_20k` to beat `live_big_hog`, and it did: step 2 is next (section 20, "Result of step 1"). `third_model.ps1`'s league plays its `-Champion`, by default `v2_hog_20k` since the update of 2026-10-30, so step 2's `v2_hog` meets the champion in that league. If another model has passed by then, give that one as `-Champion` (section 20, "Reading the result").

So the second bullet below applied: section 20 runs in two steps, and step 1 is done.

#### What to do next

- `headroom` says the tick is the tap (a large `headroom: < -7` share on `rs_rows_dt`, or many rows near 0 on `rs_rows_live`): do not run section 20. Send the log back. The rows then need a re-drive with the delay first.
- `ablation` or `fullablation` lifts the model clearly, to about 40% or more, or `waitbias` lifts `dt_big_hog` clearly more than its `live_big_hog` control: run section 20, in two steps.
  1. The second model's corpus size first, about 3-4 hours instead of a night: `third_model.ps1 -Others 20000 -Tag 20k` (add `-SkipDiskCheck` if `C:` has less than the 90 GB the full run checks for). Its model is `v2_hog_20k`.
  2. Only once `v2_hog_20k` beats the champion (its league score against it above 50%; a PASS needs at least 551 of 1,000): the whole corpus, `third_model.ps1` with no `-Tag`. Its model is `v2_hog`; the 20k run's files stay as they were. Free the disk first (section 20, "Disk"). **Met on 2026-10-30** (70.6%, PASS).
- `ablation` stays near 29%: run `-Steps fullablation` before deciding anything.
- Neither ablation moves it, and `waitbias` lifts both models alike or neither: hold section 20, run `-Steps trust`, and send the log back.

Since 2026-10-30 the champion is `v2_hog_20k` (a league PASS). `live_big_hog` stays in the live runner only until `v2_hog_20k`'s section 17 dry run is clean ("Result (2026-10-30)", Decisions).

Send back `reports\dt_diagnose_<time>.log`. If you cannot send all of it, send the `== summary` block at its end.

### Checking a checkpoint: model_check.ps1

`scripts\model_check.ps1` runs this section's simulator checks on any checkpoint, unattended, with the control in the same log. Use it for `dt_big_hog_np` and the champion's own wait_keep 1.0 copy now (Decisions above), and later for `v2_hog` and the RL checkpoints of section 21. It needs no copies made by hand. It uses only the CPU venv (`.venvs\royalegym`), but close MuMu and the browser first. Since the freeze of 2026-10-30, run it with `-Workers 4` ("If something breaks"). Do not run it while `third_model.ps1` or the live runner is running.

After `git pull`, run the tests of section 12 first. The new test is `tests\pipeline\test_model_check_ps1.py`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\model_check.ps1 -League -Champion runs\tokens\v2_hog_20k\best.pt -Workers 4
```

What it does:
- **The runs.** For each checkpoint (`-Ckpt`, default `runs\tokens\dt_big_hog_np\best.pt`), each wait_keep (`-Keeps`, default `own,1.0`) and each rule (`-Rules`, default `plain,live`), one `rs_eval` run: 300 matches against `live_big` epoch 1 at `--delay 21`. Each run has a label such as `dt_big_hog_np wk own(0.25) live d21`.
- **The control.** Right after each setting, the champion (`-Champion`, default `runs\tokens\v2_hog_20k\best.pt`, the champion since 2026-10-30) runs under the same setting. `-NoBaseline` leaves it out. A run that was already made earlier in the same run is not made again.
- **wait_keep.** `own` is the value in the checkpoint's own `args.json`. A checkpoint without one is refused for `own`. A number plays a copy in `runs\tokens\<name>_wkNNN\` (wk100 is 1.0, wk050 is 0.5), as `dt_diagnose.ps1` made them. The copy is made at the start and read back through `rs_decide.wait_keep_for` before anything runs. An existing copy is used only when it holds the same `best.pt` as its source. When the source has changed since (retrained), or another checkpoint with the same name made the copy, the script stops before anything runs, because the copy may be the live model. A folder that is not such a copy is never overwritten. Two checkpoints of one run with the same name are refused too.
- **`-League`.** After the runs, `tools\rs_league_eval.py run` plays the first checkpoint against the champion, with `live_big` epoch 1 as the anchor. Only that one checkpoint, and only at one wait_keep: its own, or `-LeagueKeep`. A `_wk100` copy plays in the `rs_eval` runs only; its league is a separate `-LeagueOnly -LeagueKeep 1.0` run (Decisions above). The settings are those of section 20's league step: 24-tick delay, the noise prior, both seats, 1,000 games against the champion and 400 against the anchor for each (`--full`), at most 6 workers. The log is `runs\league\model_check.jsonl`, with `model_check.pairs.json` (the frozen deals) next to it, and one report per matchup, `<log name>.<candidate>_vs_<champion>.report.json` (for example `model_check.dt_big_hog_np_vs_v2_hog_20k.report.json`). A rerun plays only the games the log lacks, so the champion's 400 games against `live_big` are played once and reused for every later candidate against that champion.
- **League names.** A checkpoint plays under its folder's name (`best.pt`), or its folder and file name (`runs\rl\hog_rl1\it0010.pt` plays as `hog_rl1_it0010`). A copy plays as `<name>_wk100`. One name must mean one checkpoint, so the script stops at the start when the league log (`-LeagueLog`) already uses a name of this run for another checkpoint or path (the league would refuse those games), or when any log under `runs\league\` or `runs\rl\` uses the candidate's name for another checkpoint. `-LeagueName` gives the candidate another name. For the champion's or the anchor's name in another log it prints a yellow `warning:` line and goes on. Never move an RL run's `gate.jsonl`: it is part of that run's state.
- **The log.** Everything goes to `reports\model_check_<time>.log`. Its `== summary` block has a table: label, wins, draws, win %, 95% CI, crown difference, plays per match, and the points above the control at the same wait_keep and rule. With `-League` the league's report follows, and a `league verdict:` line.

Do not use the league's `wait_keep=` player option to change a wait_keep. `rs_eval`'s agent reads the `args.json` next to the checkpoint first, so the option changes nothing for a trained checkpoint. Use the copies.

| Part | Time (measured at 4 workers, 2026-10-30) |
|---|---|
| each `rs_eval` run (300 matches) | 9.4-13.3 min (`live_big_hog` 9.4 min, `v2_hog_20k` 10.3-13.3; the one run timed at 8 workers too was no faster there); the default is 8 runs, about 1.5 h |
| the league, with a champion new to the log | about 66 min (1,800 games at about 2.2 s) |
| the league again with another candidate against the same champion, same log | about 51 min (1,400 games; the champion's anchor games are reused) |

More examples:

Each of these names the champion with `-Champion` and the workers with `-Workers 4`. Both are the defaults since 2026-10-30, so they can be left out until another model passes (Decisions above); then give that model's `best.pt` as `-Champion`.

```powershell
# the champion at wait_keep 1.0 against itself at 0.25 (the copy plays as v2_hog_20k_wk100)
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\model_check.ps1 -Ckpt runs\tokens\v2_hog_20k\best.pt -LeagueOnly -LeagueKeep 1.0 -Champion runs\tokens\v2_hog_20k\best.pt -Workers 4
# the league alone, dt_big_hog_np at wait_keep 1.0 (it plays as dt_big_hog_np_wk100)
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\model_check.ps1 -LeagueOnly -LeagueKeep 1.0 -Champion runs\tokens\v2_hog_20k\best.pt -Workers 4
# section 20 step 2's v2_hog against the champion, which third_model.ps1's league plays too; with v2_hog_20k as the champion, both anchor sets are in third_model.jsonl
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\model_check.ps1 -Ckpt runs\tokens\v2_hog\best.pt -LeagueOnly -Champion runs\tokens\v2_hog_20k\best.pt -LeagueLog runs\league\third_model.jsonl -Workers 4
# an RL champion (section 21): champion.pt changes when RL promotes, so name each one
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\model_check.ps1 -Ckpt runs\rl\hog_rl1\champion.pt -League -LeagueName hog_rl1_c1 -Champion runs\tokens\v2_hog_20k\best.pt -Workers 4
```

Other options: `-Workers` sets the `rs_eval` processes (default 4 since 2026-10-30; `rs_eval` lowers it when free memory is short and prints why); `-Matches`, `-Delay` and `-Opponent` change the `rs_eval` runs; `-Noise` changes the league's noise profile. With `-LeagueLog runs\league\third_model.jsonl`, a candidate with section 20's name, champion and anchor finds its games already played, and its report goes to `third_model.<candidate>_vs_<champion>.report.json`, the name `third_model.ps1` gives its own reports since 2026-10-30, so step 1's `third_model.report.json` stays as it is. Give other candidates the default log.

The same runs one by one, in the shorthand at the top of this file. A `_wk100` copy exists once the script has made it:

```powershell
$OPP = 'runs\tokens\live_big\ep01.pt'
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog_np\best.pt --opponent-ckpt $OPP --matches 300 --workers 4 --delay 21
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog_np\best.pt --opponent-ckpt $OPP --matches 300 --workers 4 --delay 21 --decide live
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog_np_wk100\best.pt --opponent-ckpt $OPP --matches 300 --workers 4 --delay 21
& $RG tools\rs_eval.py --ckpt runs\tokens\dt_big_hog_np_wk100\best.pt --opponent-ckpt $OPP --matches 300 --workers 4 --delay 21 --decide live
& $RG tools\rs_league_eval.py run --candidate dt_big_hog_np=runs\tokens\dt_big_hog_np\best.pt --champion v2_hog_20k=runs\tokens\v2_hog_20k\best.pt --anchor live_big=runs\tokens\live_big\ep01.pt --delay 24 --noise data\noise\default_prior.json --champion-games 1000 --anchor-games 400 --full --workers 4 --log runs\league\model_check.jsonl
& $RG tools\rs_league_eval.py report --log runs\league\model_check.jsonl
```

**Reading the result.**
- In the table, compare each line with its control, in the `vs control` column. At 300 matches a difference under about 8 points is not conclusive.
- The `live` lines are the closest to live play. Pros play about 35-40 cards a match, so `plays` far outside that range shows a lazy or a jumpy model.
- The `league verdict:` line decides. PASS means the candidate is the new champion, and every later candidate plays it. It goes into the live runner after a section 17 dry run with it ("Decisions" in "Result (2026-10-30)" above). FAIL, CYCLE or INCOMPLETE means you keep the champion. INCOMPLETE means games are missing: run it again with `-LeagueOnly` and the same `-Ckpt`, `-LeagueKeep`, `-LeagueName`, `-LeagueLog` and `-Champion`. The league plays only the games its log lacks, and `-LeagueOnly` skips the `rs_eval` runs (about 1.5 hours with the defaults).

Send back `reports\model_check_<time>.log`. If you cannot send all of it, send its `== summary` block.

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
- the champion, `-Champion` (default `runs\tokens\v2_hog_20k\best.pt`, from step 1 below), and section 14's `runs\tokens\live_big\ep01.pt` (the anchor, and the reactive opponent of `rs_eval`);
- about 90 GB free on `C:` (below).

The script checks these before it starts. Keep `data\engine.json` and the card table as they were for section 14: the league refuses checkpoints made under another engine.

The league's champion is `-Champion`: the champion of the day, the last model whose league said PASS, never a former one. Step 1 played `live_big_hog`, the champion then (section 19's `dt_big_hog`, at 29.3% against `live_big_hog`'s 46.0%, would have made the gate mean nothing). Since step 1's PASS on 2026-10-30 the default is `runs\tokens\v2_hog_20k\best.pt`, so step 2's `v2_hog` plays `v2_hog_20k` ("Result of step 1 (2026-10-30)", below). The league names the champion after its folder, as `model_check.ps1` does, and `third_model.ps1` stops at the start when the champion is missing, is this run's candidate (a `-Tag 20k` run with the default), the anchor or this run's general model, or when `runs\league\third_model.jsonl` holds games of its name played with another file, or under another spelling of its path (the league compares the path as written and would refuse it hours later: give `-Champion` as those games did, `runs\tokens\v2_hog_20k\best.pt` for `v2_hog_20k`).

After `git pull`, run the tests of section 12 first. None of this batch's code (v2 rows, compaction, `--memmap`, the v2 model) has run on the PC yet. Paste any failure back before you start the script.

**One script runs everything unattended.** It stops at the first failure, except in the evaluation runs: a failed `rs_eval` run is logged, the league still runs, and the script lists the failures and exits with an error at the end. Everything goes to `reports\third_model_<time>.log`. Run it twice, as section 19's "What to do next" says: first on the second model's corpus size, then, once that model beats `live_big_hog`, on the whole corpus:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\third_model.ps1 -Others 20000 -Tag 20k
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\third_model.ps1
```

`-Tag 20k` gives the first run its own files: `data\rs_rows_v2_20k`, `data\rs_cols_v2_20k`, `runs\tokens\v2_big_20k`, `runs\tokens\v2_hog_20k`, and the league player `v2_hog_20k`. The run without `-Tag` uses the names in the table below. Both share the league log `runs\league\third_model.jsonl`, so a champion's 400 games against `live_big` are played once and reused (`v2_hog_20k`'s are there from step 1). The league refuses a player name whose games it already holds when the checkpoint has changed. So the script stops at the start, not after hours, when it would train a model whose name already has games in the log: use another `-Tag`, or move the log away. The first run takes about 3-4 hours. Step 1 is done; run again, it would need `-Champion runs\tokens\live_big_hog\best.pt`, and the script refuses it anyway while its games are in the log.

| Step | What it runs | Time (estimated) |
|---|---|---|
| rows | `tools\rs_rows.py --schema v2 --live-compat --obs-delay 24 --obs-delay-jitter 3 --opp-lag 1 --visibility public --trust v2 --presses on --noise data\noise\default_prior.json --all-matches --wait-keep-at-build 0.25 --workers 4 --out data\rs_rows_v2` | about 3.5 h at 8 workers; 3.5-7 h at 4 (not timed) |
| compact | `tools\rs_rows_compact.py --rows data\rs_rows_v2 --out data\rs_cols_v2 --force` | about 30 min |
| general | `sim.train_tokens --name v2_big --data data\rs_cols_v2 --epochs 2 --memmap --mirror 0.5 --trust v2 --wait-keep 0.25 --d 256 --layers 6 --heads 8 --val-max-rows 500000` | about 3 h |
| hog | the same with `--name v2_hog --epochs 3 --finetune-hog --init runs\tokens\v2_big\best.pt` | about 15 min |
| eval | five runs of `tools\rs_eval.py`, 300 matches each, against `live_big\ep01.pt` (`--ckpt runs\tokens\v2_hog\best.pt --opponent-ckpt runs\tokens\live_big\ep01.pt --matches 300 --workers 4`): `v2_hog` at `--delay 21`, with no delay, at `--delay 24`, and at `--delay 21 --decide live`; then the champion (`v2_hog_20k`) at `--delay 21 --decide live` for reference | about 1 h (57 min for step 1 at 4 workers) |
| league | `tools\rs_league_eval.py run --candidate v2_hog=runs\tokens\v2_hog\best.pt --champion v2_hog_20k=runs\tokens\v2_hog_20k\best.pt --anchor live_big=runs\tokens\live_big\ep01.pt --delay 24 --noise data\noise\default_prior.json --champion-games 1000 --anchor-games 400 --full --workers 4 --log runs\league\third_model.jsonl` | about 51 min (1,400 games; `v2_hog_20k`'s anchor games are in the log) |

That is about 9.5 hours in all at 8 workers, and anywhere from 7.5 to 12.5; at the default 4, about 9-12.5 ("Result of step 1", below). Start it in the evening.

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
- **league**: up to 1,800 games, on `-Workers` workers (4 by default, at most 6):
  - 1,000 of v2_hog against the champion;
  - 400 of v2_hog against live_big;
  - 400 of the champion against live_big, played once per log: `v2_hog_20k`'s are there from step 1, so step 2 plays 1,400.

  Each game takes about 10-13 s of one CPU core, as an `rs_eval` match does, with the noise on both seats. `rs_league_eval` runs fewer workers when free memory is short (it keeps 3 GB free, about 1.2 GB per worker).

The training steps print their own time to go. Each prints `ep 0 step 500/S` every 500 steps, where S is the total number of steps. Take the time between two of these lines, multiply by S / 500, and that is how long the whole training takes.

Options:
- `-Workers 6` (the default is 4 since the freeze of 2026-10-30) only after a watched 4-worker run left more than about 7 GB of memory available ("If the PC freezes"). Not 8.
- `-Champion <best.pt>` plays another champion: the champion of the day, if a model other than `v2_hog_20k` has passed since.
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
  - **PASS**: v2_hog beats the champion (Wilson 95% lower bound above 52% over 1,000 games, i.e. at least 551 points), and it scores no more than 2 points below the champion against `live_big`. If the `-Champion` it played is still the champion, v2_hog is now the champion (section 19, "Result (2026-10-30)", Decisions), and every later candidate plays it. It becomes the live model after a section 17 dry run with `--ckpt runs\tokens\v2_hog\best.pt` (`runs\tokens\v2_hog_20k\best.pt` for the `-Tag 20k` run) added to the `vision_play.py` command; then add the same `--ckpt` to section 18's. The runner reads the checkpoint's schema and builds v2 rows by itself.
    - If another model became the champion after the run started, v2_hog must pass against that model too: `scripts\model_check.ps1 -Ckpt runs\tokens\v2_hog\best.pt -LeagueOnly -Champion <the champion's best.pt> -Workers 4` (`runs\tokens\v2_hog_20k\best.pt` for the 20k run), about 51 minutes at 4 workers (1,400 games). `-League` instead adds its `rs_eval` runs, about 1.5 hours more.
  - **FAIL**, **CYCLE** or **INCOMPLETE**: keep the champion, and the model the live runner has.

  The `paired` lines show v2_hog's score minus the champion's against the same opponent on the same positions, with an interval.
- The report is also written next to the log, one per matchup, since the update of 2026-10-30: `runs\league\third_model.<candidate>_vs_<champion>.report.json`, for step 2 `third_model.v2_hog_vs_v2_hog_20k.report.json`. Step 1's report stays in `runs\league\third_model.report.json` (rs_league_eval's default then); no later run writes over it. A `model_check.ps1` run on this log names its report the same way, so the same matchup writes the same file, from the same games. The report's text is in the run's log too. In the whole-corpus run's report, `live_big_hog` (step 1's champion) appears among the other players.

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

### Result of step 1 (2026-10-30)

`third_model.ps1 -Others 20000 -Tag 20k` ran the evening of 2026-10-30. The PC froze about 6 minutes into the first `rs_eval` run at 8 workers ("If something breaks", "If the PC freezes"); after the restart, `-From eval -Workers 4` ran the evaluation and the league (`reports\third_model_20261030_194930.log`, 19:49-21:51). The rows, the compact store and both trainings are from the first log of that evening, which has not been sent back yet: it holds the row counts, the `epoch N: val` lines and each step's time.

`rs_eval`, 300 held-out matches against `live_big` epoch 1:

| Run | Win rate [95% CI] | Crown difference | Plays per match | Time |
|---|---|---|---|---|
| `v2_hog_20k`, 21-tick delay | **78.7%** [73.7, 82.9] | +0.83 | 38.0 | 747 s |
| `v2_hog_20k`, no delay | 56.3% [50.7, 61.8] | +0.09 | 42.3 | 798 s |
| `v2_hog_20k`, 24-tick delay | 84.3% [79.8, 88.0] | +1.03 | 37.8 | 618 s |
| `v2_hog_20k`, 21-tick delay, live rule | 69.7% [64.2, 74.6] | +0.46 | 41.4 | 648 s |
| `live_big_hog`, 21-tick delay, live rule (reference) | 51.7% [46.0, 57.3] | -0.15 | 38.3 | 561 s |

The league: dev split, 24-tick delay, the noise prior on every seat, both seats, decks exchanged against the champion, protocol `c0d21e5580dc97d0`, 4 workers:

| Matchup | Score [Wilson 95%] | Games |
|---|---|---|
| `v2_hog_20k` vs `live_big_hog` | **70.6%** [67.7, 73.3] (706 W, 294 L) | 1,000 |
| `v2_hog_20k` vs `live_big` | 57.8% [52.9, 62.5] | 400 |
| `live_big_hog` vs `live_big` | 32.5% [28.1, 37.2] | 400 |

- Paired against `live_big`: `v2_hog_20k` minus `live_big_hog` +25.2 points [+18.9, +31.4], McNemar p 0.000.
- Ratings (Elo, `live_big` = 0): `v2_hog_20k` +43.1 ± 13.7, `live_big_hog` -114.4 ± 13.9.
- Gate: **`v2_hog_20k: PASS (vs champion live_big_hog)`**: Wilson lower bound 0.677 > 0.52, and 25.2 points above the champion against the anchor. No pool members, so the cycle check was not played.
- Times at 4 workers: 1,000 games in 2,193 s and 800 anchor games in 1,694 s, about 2.2 s a game.

**Reading:**
- The v2 rows fix what sank the second model. On section 19's instrument (the same opponent, matches and delay) `v2_hog_20k` wins 78.7%, where `dt_big_hog` won 29.3% and `live_big_hog` 46.0%. Its play rate, 38.0 a match, is inside the pros' 35-40: no lazy gate.
- It is level with the ablation, not above it: `dt_big_hog_np` (the legacy recipe without the waits with our play in flight, no `own_pending`, no noise) won 80.7% [75.8, 84.7] on the same 300 matches, 2.0 points more, within the noise. So `rs_eval` shows that removing the in-flight look-alike waits is the fix, which the ablation does by dropping them and v2 by telling them apart (`own_pending`). It does not show that v2 adds anything on top: what else it brings (the noise in its rows, the opponent's elixir count, `own_pending` live) is not measured by `rs_eval`. That makes Decisions item 1, `dt_big_hog_np`'s league against `v2_hog_20k`, a real contest.
- Its timing is what the 24-tick lead predicts: best at 24 ticks (84.3%), a little lower at 21, and much lower with no delay (56.3%).
- Under the live rule, the closest to live play, it leads `live_big_hog` by 18 points (69.7% against 51.7%), well beyond the about 8 points 300 matches can resolve.
- The league agrees: 70.6% against the champion over 1,000 games is a clear pass. **`v2_hog_20k` is the champion** from 2026-10-30 (section 19, "Result (2026-10-30)", Decisions).
- The anchor is not weak under league conditions: `live_big_hog` scores only 32.5% against `live_big` epoch 1 there (noise on every seat, a 24-tick delay, both seats), and `v2_hog_20k` 57.8%.
- Caveats: the simulator, one noise prior (not yet fitted to our screen reader), and 300 matches per `rs_eval` line. Real friendlies are the final check.

**Next:**
- `v2_hog_20k` goes into the live runner after a section 17 dry run with `--ckpt runs\tokens\v2_hog_20k\best.pt`; then section 18's `--tap` friendlies use the same `--ckpt`.
- Step 2, the whole corpus, overnight, after the day's `model_check.ps1` runs and with MuMu and the browser closed:

  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\third_model.ps1 -Workers 4
  ```

  - `-Workers 6` if Task Manager showed more than about 7 GB of memory "Available" during a 4-worker run that day. Not 8: the PC froze at 8 on 2026-10-30.
  - The disk check wants 90 GB free on `C:`. If it is short, the 20k run's rows and store can go now that its league is done ("Disk", above).
  - Time at 4 workers: about 9-12.5 hours, and up to about 15 if section 20's estimates (none of them measured yet) are low. The rows step is the long one: section 20 estimates 3.5 h at 8 workers. At 4 it takes between that (if 4 workers build rows as fast as 8, as `rs_eval` ran) and about 7 h (if they take twice as long); it has not been timed at 4. The rest is about 5.5 h: compact 0.5 h, the general model about 3 h on the GPU whatever the workers, the fine-tune 0.25 h, the evaluation about 1 h (57 minutes for the 20k run at 4 workers) and the league about 51 minutes. At 6 workers about 9-10.5 hours. Start it by about 19:00. The 20k run's own step times (the first log of 2026-10-30, at 8 workers) will firm this up.
  - Its league plays `v2_hog` against the champion, `-Champion`, by default `v2_hog_20k` since the update of 2026-10-30: 1,400 games, about 51 minutes, since `v2_hog_20k`'s anchor games are already in `runs\league\third_model.jsonl` (from step 1). A PASS there makes `v2_hog` the champion; no second league is needed.
  - If a `model_check.ps1` run that day made another model the champion (`dt_big_hog_np`, `v2_hog_20k_wk100`), give it: for example `third_model.ps1 -Workers 4 -Champion runs\tokens\dt_big_hog_np\best.pt`. Its league then also plays that champion's 400 games against `live_big` in this log: 1,800 games, about 66 minutes.
  - Before the update, the league was hard-wired to `live_big_hog`, and `v2_hog` needed one more league against `v2_hog_20k` (`scripts\model_check.ps1 -Ckpt runs\tokens\v2_hog\best.pt -LeagueOnly -Champion runs\tokens\v2_hog_20k\best.pt -LeagueLog runs\league\third_model.jsonl -Workers 4`). That command still works, and plays nothing new after the night's league; its report goes to `third_model.v2_hog_vs_v2_hog_20k.report.json`, the file step 2's league writes, not over step 1's `third_model.report.json`.

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

The first time, the script first builds the opponents' deck field (`tools\deck_stats.py --top 2000`, a few minutes) if `data\deck_stats.json` is missing. Then it plays for about 10 minutes at 1, half and all of `-Workers`: 1, 2 and 4 workers by default (the freeze of 2026-10-30, "If the PC freezes"; `-Workers 0` leaves the counts to the driver, 1, half and all that RAM and CPU allow, about 9 on this PC: not until the freeze is explained), and times one PPO update. It prints:
- `workers N: ... matches/hour, ... RAM per worker ... MB at peak` for each count;
- `RAM per worker` and `recommended: --workers N`, with the reason;
- how long an iteration of 400 matches takes, collection and update;
- a last line with the command to start: `rl_train.ps1 -Workers N -Hours 48`, with a reminder to use at most 4 until the freeze is explained.

Options: `-BenchMinutes 20` measures longer. The bench writes `runs\rl\hog_rl1\bench-<time>\bench.json` and trains nothing.

### Step 2: the long run (days, MuMu closed)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\rl_train.ps1 -Workers 4 -Hours 48
```

Use the bench's `-Workers`, but at most 4 until the freeze of 2026-10-30 is explained ("If the PC freezes"); 4 is the default since then (`-Workers 0` gives the driver's own 8). Before every collection, the driver lowers the count when free memory is short, and it learns the real memory per worker from the previous collection.

How long (estimated; the bench measures the collection part):
- One iteration is the collection (400 matches at the bench's matches/hour) plus the update (the bench's time).
- One gate is 1,000 games against the champion, plus 400 against each anchor only when the candidate beat the champion. The gate runs on `-Workers` workers, at most 6 (4 by default): at about 2.2 s a game at 4 workers, as section 20's league played, about 37 minutes. The first time a candidate beats the champion, the champion also plays both anchors once (800 more games); later gates reuse those games.
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
  powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\rl_train.ps1 -From resume -Workers 4 -Hours 24
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

### If the PC freezes

On 2026-10-30 the PC froze about 6 minutes into an `rs_eval` run at 8 workers and had to be reset. Windows' Reliability Monitor (`perfmon /rel`) also lists **LiveKernelEvent 117** on 2026-10-16, 10-18 and 10-21, and a Desktop Window Manager crash on 10-26. LiveKernelEvent 117 (0x117, VIDEO_TDR_TIMEOUT_DETECTED) means the graphics driver stopped responding and Windows reset and recovered it. A recovery that fails can hang the PC, or crash it with bugcheck 0x116 (VIDEO_TDR_FAILURE). But the 10-30 freeze came during a CPU-only job with the GPU idle (`rs_eval` runs in the CPU venv), the first time a v2 model ran at 8 workers, while about 80 minutes of 8-worker `rs_eval` runs of the legacy models had run without trouble that afternoon (`dt_diagnose.ps1`). So the driver and memory (8 v2 workers on 16 GB, perhaps with the browser open) are both suspects, and the power mode's heat a third. Windows' logs at the time of the freeze tell them apart: Event Viewer > Windows Logs > System shows "Display driver nvlddmkm stopped responding and has successfully recovered" (Event ID 4101) for a driver timeout, and the Kernel-Power 41 event of the reset has `BugcheckCode` 278 (0x116) when the driver's recovery failed, 0 when the PC hung without a crash. Each fix below covers one suspect: the clean install the driver, 4 workers the memory, Balanced the heat. What to do:
- **A clean driver install.** Close MuMu, the browser and every `python.exe`. Install the newest NVIDIA Studio driver, version 580 or newer (CUDA 13.0 needs R580), with Custom (Advanced) and "Perform a clean installation" ticked, then restart. Run section 1's two checks: `nvidia-smi` must show `CUDA Version` 13.0 or higher, and the torch line must print `True`.
- **Omen Gaming Hub on Balanced**, not Performance, for all long runs.
- **`-Workers 4`** for every engine job: `third_model.ps1`, `model_check.ps1`, `dt_diagnose.ps1` and `rl_train.ps1` take `-Workers` and default to 4 since the update of 2026-10-30, and `tools\rs_eval.py` and `tools\rs_league_eval.py` take `--workers 4`. Both tools also lower the count by themselves when free memory is short (about 1.2 GB a worker, 3 GB kept free), and say so in their `workers:` line; `rs_eval`'s results do not depend on the count. It cost little in the one comparison so far: on 2026-10-30 the same `rs_eval` run (`live_big_hog`, live rule) took 561 s at 4 workers and 582 s at 8, with identical results; the `v2_hog_20k` runs took 10.3-13.3 minutes at 4, and no v2 run has been timed at 8. The league played about 2.2 s a game at 4. The rows step of `third_model.ps1` has not been timed at 4. Go up to 6 only after a watched run with more than about 7 GB of memory "Available" (Task Manager > Performance > Memory). Not 8 until the cause is known.
- Watch the temperatures in the first minutes of a long run: Task Manager > Performance > GPU shows the GPU temperature (keep it under about 80-85 °C), and Omen Gaming Hub shows the CPU's.
- **After a freeze:** reset the PC, then look in Reliability Monitor for a new LiveKernelEvent or other critical event at that time, and send its details back with the log. Resume where it stopped: `third_model.ps1 -From <the step that was running>` with the same `-Tag` (as `-From eval` did on 2026-10-30), or the same `model_check.ps1` command again. Each league resumes from its log and plays only the games it lacks; an `rs_eval` run starts over. Nothing in this repo starts or restarts MuMu: start it yourself if you need it.

## Later steps (coming in a later batch)

These need code that is not on the branch yet. Each one gets exact commands here when its batch lands. The IDs refer to `docs/SPARRING_PLAN.md`.

| Phase | What you will run | Waits for |
|---|---|---|
| 2. Engine re-baseline | spawn tables for both card tables; then a fidelity grid over 2,000 validation matches (`tools/rs_fidelity_grid.py`), which picks the RoyaleSim card table and writes `data/engine.json` | E01, E03, E06 |
| 4. Decision-time models, v2 | the decision-time model is section 19 and the v2 model (perception noise in the rows, the ability head, memory-mapped training) section 20. Still to come: v2 rows on the vision class list (`--units vision`) | the simulator's evaluators and RL workers reading `--units vision` rows |
| 5. Screen perception, second pass | label frames from our own friendlies for the cards and forms the KataCR dataset lacks, then score the detector on them (`tools/vision_replay.py --labels`); fit the perception-noise profile used in training (`tools/fit_noise_profile.py`) | V11 (rest), E17 |
| 6. Live friendlies with a stronger model | the same runner (sections 17-18) with a model trained on decision-time rows (section 19's, or section 20's v2 model) and the measured delay D | the results of sections 19 and 20 |
| 7. RL fine-tuning, the live check | the driver is section 21. Still to come: the promoted model against the imitation champion in real friendlies, match by match in turn; the optional search teacher (R05) as a second arm | the first champion from section 21 |
