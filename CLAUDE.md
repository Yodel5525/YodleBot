# CLAUDE.md

This is a fork of Hasty-CR, a Clash Royale bot research project. The fork has one goal: a strong **sparring bot for friendly battles** against the owner's clanmates, playing **Hog 2.6**.

Read `docs/SPARRING_PLAN.md` for the plan and `docs/SPARRING_RUNBOOK.md` for the PC setup and commands. The original README describes the author's older pipeline; the plan explains what changed and why.

## Hard rules

- **Friendly battles only.** Nothing may tap the lobby Battle button or queue ladder or any other matchmade mode. A human starts every battle. Do not add a ladder option.
- **No memory reading.** `scripts/gamestate/memread.py` and `scripts/gamestate/native/hcr_reader*` read the game's process memory. They are not used. New code must not import them, and do not extend them or re-derive their offsets.
- **No anti-detection or ban-evasion work**, such as tap jitter meant to look human, hiding processes, or defeating anti-cheat.
- **Never start or restart the emulator on your own**, and never touch any other MuMu instance. `scripts/emulator.verify()` must pass before anything taps.
- **Live play is dry-run by default.** Only tap with `--tap` while the owner is watching. A `STOP` file in the repo root stops the live runner.

## Layout

| Path | What |
|---|---|
| `tools/rs_*.py` | RoyaleSim pipeline: building training rows, engine factory, evaluation, self-play |
| `sim/model_tokens.py`, `sim/train_tokens.py` | the entity-token policy and its imitation training |
| `scripts/vision/` | screen perception for the live bot |
| `tools/league/`, `tools/rl/` | evaluation statistics, league and RL |
| `src/hastycr/machine.py` | machine settings (adb, serial, venvs, data roots) from environment variables or a gitignored `machine.json` |
| `tests/pipeline/` | tests for the new pipeline (the legacy `tests/` needs `tmp/gamedata`, which is gitignored, and has known failures) |
| `sim/` (engine), `scripts/brain/`, `scripts/cr_bot.py` | the author's older engine, rule bot and plane-PPO pipeline; kept for reference |

## Environments

There are two venvs, created by `scripts/setup_windows.ps1`:

- `.venvs/royalegym`: CPU torch and `royalesim`. Used for building rows, evaluation and RL rollout workers.
- `.venvs/buildabot`: CUDA torch, `royalesim` and vision packages. Used for training, YOLO and live play.

The machine has 16 GB RAM:
- Keep worker counts modest.
- Prefer the memory-mapped data path for training.
- Ask the owner to close MuMu and the browser before long training runs.

## Tests

```powershell
.venvs\royalegym\Scripts\python.exe -m pytest tests/pipeline -q
.venvs\buildabot\Scripts\python.exe -m pytest tests/pipeline -q
```

Markers:
- `engine`: needs `royalesim`.
- `il_replay`: needs `data/il_replay`.
- `gpu`: needs CUDA.
- `device`: needs `HASTYCR_DEVICE_TESTS=1` and a running emulator.

Tests skip cleanly when these are missing.

## Status

The code on the `sparring` branch was written in a cloud sandbox that cannot execute it. The first test run on the PC is expected to surface some integration bugs. Fix them in small commits, and keep the legacy behaviour reproducible (golden tests).
