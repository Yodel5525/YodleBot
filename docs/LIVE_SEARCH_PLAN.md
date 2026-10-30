# Live search plan

Status: design, 2026-10-30. Nothing in this plan is built yet. The code is written in the cloud sandbox, which cannot run it, and every number is measured on the owner's PC.

The hard rules in `CLAUDE.md` apply to every item:
- friendly battles only;
- no memory reading;
- no anti-detection work;
- never start or restart the emulator;
- dry run unless `--tap`, and `--tap` only while the owner watches;
- a `STOP` file stops the live runner.

Line references are `file:line` on the `sparring` branch as of this commit. RoyaleSim calls are cited by their docstrings in `royalesim` 0.1.42; `help(royalesim.Battle)` shows them on the PC.

## 1. Summary

**The problem.** The live bot decides about 24 ticks (1.2 s) before its play lands, and it decides on the board as it is now. Its policy, `live_big_hog`, was trained on landing-time rows, so it expects the board at the moment the card lands (`docs/SPARRING_PLAN.md`, "First-match milestone").

**What this plan adds.** It rebuilds the board in RoyaleSim from what the screen shows, steps it forward to the landing tick, and then does one or both of these:
1. **Root look-ahead.** The policy decides on the stepped-forward board instead of the present one.
2. **Search.** Up to 8 candidates (the policy's own play, waiting, other tiles and other cards) are tried on that board. The bot plays the best one only when it clearly beats the policy's own choice.

**How it is checked.** Both parts are measured in the simulator first. They run as seats in `tools/rs_match.py` and must pass the L03 gate in `tools/rs_league_eval.py`. The search's own compute time is charged as extra command delay.

**How a weak result stops early.** The work runs in this order, and each step can end it:
1. a benchmark;
2. a privileged ceiling;
3. a cheap regret instrument;
4. games.

**Live use.** It comes only after the gate. Search is off by default (`--search off`), and `--search shadow` only logs.

## 2. Why: the numbers

### 2.1 What was measured before

These are the original author's results, on 300 held-out Hog 2.6 matches in RoyaleSim against a reactive epoch-1 reference (`docs/PROGRESS_2026-10-09.md`).

| Setup | Hog wins | Source |
|---|---|---|
| plain policy (`rs_bc` final), no delay | 40.3% [34.9, 46.0] | PROGRESS:161 |
| + search on the **true** battle copy, no delay | **64.3%** [58.8, 69.5] | PROGRESS:179 |
| + search on a battle **rebuilt from public observation**, no delay | 47.0% [41.4, 52.7] | PROGRESS:180 |
| `rs_hog`, delay 21, nothing | 32.0% | PROGRESS:292 |
| + look-ahead 21 on the true copy, new enemy units hidden | 60.0% [54.4, 65.4] | PROGRESS:203 |
| + look-ahead 21 on the rebuilt battle (after the forms fix) | **47.0%** [41.4, 52.7] | PROGRESS:290 |
| + rebuilt look-ahead + rebuilt search | 41.7% [36.2, 47.3] | PROGRESS:299 |
| + delay-aware search on the rebuilt look-ahead board | 44.0% [38.5, 49.7] | PROGRESS:313 |
| `rs_live60_hog` (live-compatible model), delay 21: none / rebuilt look-ahead | 39.7% / 40.0% | PROGRESS:311-317 |

Our current model, from `docs/SPARRING_PLAN.md:82-83`:
- `live_big_hog` wins 56.7% with no delay and 46.0% at delay 21, against `live_big` epoch 1.
- So the delay costs about **0.51 pp per tick**.

Four things follow from these numbers:
- **Search is worth a lot when the world model is right:** +24 pp on the true copy.
- **The rebuild kept only about 7 of those 24 points** (PROGRESS:183-188). Under delay, rebuilt search made things worse. The author's verdict: "Search on a rebuilt board still adds nothing, delay-aware or not" (PROGRESS:323).
- **The public rebuilt look-ahead alone was worth +15 pp** at delay 21 for the unit-type model `rs_hog`. For the live-compatible model it was worth nothing. PROGRESS:315-317 says this is "likely because" rebuilt units carry the card they were spawned as, while live-compatible tokens expect the card that created them. That is a hypothesis, not a finding.
  - The author's next step was to read unit types from game memory (PROGRESS:318-321). CLAUDE.md forbids that, so this plan restores the creating card instead (section 4.5).
- **Search under a delay was never measured on the true copy.** So the size of the prize at delay 24 with perception noise is unknown. Stage M2 measures it first.

### 2.2 What went wrong in the old search

The old search is `RSSearchAgent` in `tools/rs_eval.py:593-727`.

| Defect | Where | This plan |
|---|---|---|
| It never compared waiting with playing: it searched only after the gate drew a play | rs_eval.py:627 | Waiting is always a candidate (unless the cap rule forbids it) |
| The opponent was idle in trials and had 0 elixir | rs_eval.py:382 | Counted elixir and a known card cycle. One arm adds a sampled in-flight play and one reply from an opponent model |
| The rebuild used the opponent's true deck (a leak, harmless only because the opponent never played) | rs_eval.py:352 | The deck is only the cards seen, plus a prior over the rest |
| Candidate legality came from the true battle | rs_eval.py:640 | Legality at tap time (as today) and at landing time on the rebuilt board |
| Our queue followed deck order, not the true cycle | rs_eval.py:337-349 | Our queue in cycle order |
| A new Battle was built on every call | rs_eval.py:379 | One Battle per process, reused with `reset` |
| The after-row was dated `tick + horizon`, not `tick + lead + horizon` | rs_eval.py:699 | Rows are dated at the tick they describe |
| Each candidate had its own forward pass | rs_eval.py:703-704 | One shared encoder pass plus one batched value forward |
| It replaced the policy's choice on any margin | rs_eval.py:726 | It overrides only by a margin (a trust region) |
| Deploying units were spawned already deployed; spells in flight were lost | rs_eval.py:327-331 | Deploy-timer and spell recasts (section 4.4) |

### 2.3 What the gate needs

- The L03 gate needs a Wilson lower bound above 0.52 over at least 1,000 games. In practice that is at least 551 of 1,000, or 55.1% (`tools/rs_league_eval.py:58-59, 156-157`).
- A search that costs 2 ticks of latency must earn more than about 1 pp to break even.

## 3. What the search does (one decision)

The notation:
- `t` is the decision tick.
- `team` is our side and `opp = 1 - team`. In the simulator our side can be team 0 or 1, because `MatchSpec.swapped` exchanges the seats (`tools/rs_match.py:383-391`). Live, the bot is always team 1.
- `D` is our decision-to-landing delay. In the simulator it is the seat's command delay: the spec delay plus the seat's `latency_ticks`. A command sent at `t` runs at `t + delays[team]` (`rs_match.py:949`). Live, it is the latency_probe median minus 2 sighting ticks, plus `latency_ticks` (section 8).
- `B = t + D` is the tick our play would land.

### 3.1 The steps

1. **Trigger check** (cheap). The search runs when any of these is true:
   - the plain decision `d0` is a play;
   - a new opponent play has appeared in memory since the last search;
   - our spendable elixir is at least 8.0;
   - `rs_decide.threat_lane` reports a push (`tools/rs_decide.py:1018`).

   After a search chooses waiting over a play, that verdict holds for 10 ticks (the simulator's decision cadence, `rs_match.py:177`). A new opponent play ends the hold early.
2. **Public view.** The seat's observation is cut down to what the screen shows (section 4.1). Live, the state is already public, so this step is the identity, and a check asserts it.
3. **Rebuild at `t`.** One reused, undelayed Battle is reset from the public view (section 4).
4. **Advance to `B`.** Our own in-flight plays are stepped in at their landing ticks, then the battle is stepped to `B` and `save()` is called once. This is the stepping logic of `RSAgent.ahead` with `lead_rebuild` (`rs_eval.py:269-300`).
5. **Root row and `d0`.** For a landing-time checkpoint (`rs_decide.obs_delay_ticks(meta) == 0`, `rs_decide.py:270`):
   - The row is built on the root board and dated `B`, as `RSAgent.act` dates it at `tick + lead` (`rs_eval.py:542-544`).
   - The hand, elixir, deck and next card are the present ones (`rs_eval.py:294-300`).
   - `d0 = rs_decide.decide(...)`, the same call, legality and random draw as the plain seat (`rs_eval.py:546-552`).

   For a decision-time checkpoint, the row is the present one.
6. **Fallback.** If the trigger did not fire, the rebuild was refused, or the fidelity score is low (section 4.6), the bot plays `d0`.
7. **Candidates** (at most 8):
   - **P**, `d0`'s play;
   - **W**, waiting, unless `d0` was capped (cap rule);
   - P's card at 2 more tiles, at least 2 tiles from P and from each other;
   - the top 2 other unmasked slots by `rs_decide.gate_probs` (`rs_decide.py:1312`), with 2 tiles each from the best `TOP_TILES = 60` (`rs_decide.py:151`).

   Rules for the candidates:
   - A tile must pass the tap-time legality that `d0` used (the same object) and `check_deploy` on the root at `B`, with the card mapped to its slot there.
   - The Cannon lane guard applies as in `d0` (`DecideConfig.guard_cards`; `GUARD_CARD_NAMES`, `rs_decide.py:168`).
   - prior = log q(slot) + log-softmax(tile), as at `rs_eval.py:642`.
   - Candidates with a prior below prior(P) − 4 are dropped.
   - If fewer than 2 candidates remain, the bot plays `d0`.
8. **Trials.** For each candidate: `load(root)`, `step([cmd], H)`, then the after-state is read with `state_json` and relabelled (section 4.5). With opponent worlds (section 5), each candidate runs once in each world.
9. **Score.**
   ```
   score = swing + V + 0.05 · prior − 0.05 · leak
   ```
   - **swing** is the tower-fraction swing from `B` to `B + H` (`tools/rl/teacher.py:282`; `rs_rows.towers_vec`, `tools/rs_rows.py:1658`).
   - **V** is the value of the after-row, dated `B + H`. A legacy model uses the value head's column 0, its predicted 10 s tower swing in the same units as swing (`sim/model_tokens.py:149, 267`; label horizon `rs_rows.py:290`). A critic model uses tanh(z/2) + Phi (`teacher.py:574-578`; `tools/rl/ppo.py:285`).
   - A finished game scores `ppo.result_of` (`ppo.py:275`), scaled to ±3 for the value head.
   - **leak** is the elixir regeneration over `[B, B + H]` that does not fit under the cap, given our elixir at `B` and the candidate's cost (`rs_elixir.ElixirLaw.regen`, `tools/rs_elixir.py:282`).
10. **Choose.**
    - With one world, the best candidate replaces the incumbent (P, or W when `d0` waited) only if it scores at least τ = 0.05 higher.
    - With K worlds, the override needs the mean paired gain ≥ τ + z·sd/√K, with z = 1.
    - If the incumbent's trial did not run, the bot plays `d0`.
11. **Return a Decision.** `rs_decide.Decision` is frozen (`rs_decide.py:1049-1050`), so the result is a new Decision built with `dataclasses.replace(d0, ...)`:
    - it keeps the present slot (the command is sent now);
    - its mode is `d0.mode + "+search"`;
    - a chosen wait has kind WAIT and reason `search_wait`.

    `vision_play._decide` and `to_record` then work unchanged (`scripts/vision_play.py:1088-1145`).

The search never consumes the seat's random generator. It has its own, seeded by (spec seed, tick). Before the first override, the search seat's `d0` is therefore drawn exactly like the root-only seat's.

### 3.2 Presets

| Preset | Root look-ahead | Trials | Opponent in trials | Role |
|---|---|---|---|---|
| `s_root` | yes | none (`max_trials 0`) | none | look-ahead alone; the champion that isolates the search |
| `s_idle` | yes | ≤ 8, H 60, one world | idle | cheap search |
| `s_reply` | yes | ≤ 8, H 80, K 4 worlds | an in-flight play plus one reply (section 5) | search against a modelled opponent |
| `s_true` | yes | as `s_idle`, but run on the TRUE delayed battle | its real queued commands | **privileged ceiling, never promotable** |

`SearchConfig` is a frozen dataclass of builtins with `to_json` / `from_json`, like `TeacherConfig` (`teacher.py:172-222`). A preset name expands to its full dict, so the game keys record the content of the preset, not just its name.

## 4. How the battle is rebuilt

### 4.1 Inputs: only what the screen gives

In the simulator, `run_match` gives every seat the TRUE battle and a noised state (`rs_match.py:936-941`). That noised state still holds privileged data: the opponent's hand, elixir, pending plays and deck, the hidden entity columns, and exact spells and projectiles (`tools/perception_noise.py:736`, `_copy_state`). So the search reads only `public_view(state, team, mem_view)`. It produces what `PublicStateBuilder.state` produces live (`tools/public_state.py:298, 537-640`):

| Input | Simulator source | Live source |
|---|---|---|
| Enemy units | only those `rs_rows.seen_by` passes (`rs_rows.py:695`; hidden or invisible are dropped, `rs_rows.py:330-334`) | tracker tracks, with `EventDeriver.card_of` as the card (`scripts/vision/events.py:727`) |
| Hidden entity columns | target, attack phase, stun, charge, buffs, shield and destination reset to `ENTITY_DEFAULTS` (`public_state.py:207`) | not present |
| Opponent block | elixir 0, hand [], next −1, pending [], deck = cards seen (`public_state.py:632-635`) | the same |
| Spells | aim = the current position, a placeholder as live (`public_state.py:510-519`); projectiles [] | spell tracks |
| Our hand, next, elixir, pending | our own block (noised HUD elixir) | HUD; `ctx.pending` |
| Our in-flight aims | `seat.sent` = (exec_tick, card, x, y) (`rs_match.py:961`) | `perception.deriver.pending_taps` (`events.py:759`); `ctx.pending` has no aims (`rs_decide.py:365`) |
| Our queue order | our run plays, using the same cycle rule as the opponent's (section 5) | `perception.hud_reader.hand.state().cycle` (`scripts/vision/hud.py:1035-1039`) |
| Opponent elixir | `MatchAgent.opp_elixir` (counts only plays that ran, `rs_match.py:505`) | `rs_rows.SeenElixir` (`rs_rows.py:852`) or `public_state.OpponentElixir` (`public_state.py:683`) on `ctx.memory`, for every checkpoint; today `make_observer` gives legacy checkpoints none (`vision_play.py:841-856`) |
| Opponent plays | the seat's memory view, already noise-filtered (`tools/rs_memory.py:137`) | `ctx.memory`, fed by `EV.record_plays` (`vision_play.py:1053`) |
| Clock, towers, crowns | the noised state | HUD `ClockFilter`, `TowerHpTracker` |

**Permanence.** An enemy building that was seen and never seen destroyed stays in the rebuild, with its hp drained linearly over its lifetime. A Tesla that has gone underground is the main case: `seen_by` drops it, yet it is the main counter to a Hog. A spawned Tesla starts hidden (calibration `hide.STARTS_HIDDEN`; probe P9).

### 4.2 The reset

The rebuild uses one Battle per process, from `rs_engine.make_battle(cfg.undelayed(), rs)` (`tools/rs_engine.py:425-427, 519-543`), and reuses it with `reset`. The delay stays (0, 0), and every command is timed by hand.

`set_command_delay_ticks` applies "for every battle this object starts from the next `reset` on" (Battle docstring). Probe P8 checks that it persists.

The reset call:
```
reset(seed, decks, shuffle=0, start_tick=t, elixir_milli=[blue, red], tower_hp, spawns)
```

- **spawns**
  1. Run `public_state.sanitize_for_rebuild(view, spawn_table, fields=RR.F, sub)` (`public_state.py:935`). It clamps units into the arena, moves ground units off the water, caps hp, pads decks to 8 and keeps each king at 1 hp or more.
  2. Make one spec `(team, card, x_sub, y_sub, hp)` per unit, with `card = public_state.rebuild_card` (`public_state.py:865`).
  - Coordinates: engine = millitiles × 18, with no flip.
  - hp = fraction × `unit_hitpoints(card, 11)`, clamped to [1, max]. The core accepts any hp ≥ 0, so the code must clamp it.
  - Buildings are snapped with `building_placement`, a pure query, so the footprint matches a real deploy.
- **decks** use shuffle 0, so the hand is `deck[0:4]` and the queue is `deck[4:8]`.
  - Ours: pending cards first, then the hand, then next, then the queue in cycle order. This is `RSAgent.rebuilt`'s order (`rs_eval.py:337-349`), except the queue tail follows the true cycle.
  - The opponent's: hand cards first, then the queue, oldest play first (section 5).
- **elixir_milli**: ours from the HUD; the opponent's from the counter. Recast costs are added (section 4.4). Values are clamped to [0, 10000].
- **tower_hp**: [king, left, right] per team in world sides. A king below 1 is raised to 1.
- **start_tick** = the perceived tick.
  - Elixir rates follow the tick: 2× from 2400, overtime from 3600, 3× from 4800.
  - Live, the clock offset between RoyaleSim tick 0 and the timer leaving 3:00 is not measured yet (`hud.py:9-16`).

**Refusals.** One bad spec fails the whole reset. In that case the bot falls back to `d0` and the reason is logged, as `rs_eval.py:385-390` does. The same fallback applies when the reset state is already game over. That happens in overtime when misread crowns are unequal (probe P7).

### 4.3 Our in-flight plays

Our plays that have been sent but have not landed are stepped in at their exec ticks, during the advance to `B`. The card is mapped to its slot through the rebuilt hand (`rs_eval.py:281-293`). Only one command per team is allowed in each `step` call.

The engine changes our hand and elixir only when a command runs (`set_command_delay_ticks` docstring), and the live state follows the same model: our in-flight cost is reported separately as `pending_cost` (`public_state.py:89, 631`). So the elixir at `t` has not paid for these plays yet; each one pays when it is stepped in.

### 4.4 Recasts (rebuild level R2)

A spawn starts already deployed: MockEngine uses `deploy=0`, calibration `rider.DEPLOY` speaks of "a setup spawn", and probe P2 checks it. It also fires no deploy blow (calibration `combat.DEPLOY_PROJECTILE`). So a Hog placed at the bridge half a second ago acts up to 1 s early. Recasts fix this.

**Deploy-timer recast.** Take a unit group (same team and card, with deploy ticks within 3 of each other) that still has at least half of its deploy time left. It is not spawned; its card is played at `t` instead:
- the opponent's at the group's centroid (a building at its snapped centre);
- ours at our exact aim.

The card must be in that side's hand, so the deck is ordered so that the hand and queue are right after the recasts run. With hand H, queue Q and recast cards R (the most recent plays, so the tail of Q), choose C = |R| cards of H. Then:
- pre-hand = R + (H minus C);
- pre-queue = C + (Q minus R).

After the recasts run, the hand holds H and the queue holds Q. The recast's cost is added to that side's elixir, so the counted elixir is right afterwards.

If `check_deploy` refuses the recast, the group is spawned instead.

A recast lands late by the deploy time that had already passed. The 50% threshold keeps that at or below about half a second.

**Spell recast.**
- **Our own spells in flight:** recast at `t` at our known aim. They land late by `t − exec`.
- **Enemy projectile spells** (Fireball, Rocket, Arrows, Snowball, Goblin Barrel): the aim is not public. Live, a spell row's aim is only the track's current position (`public_state.py:510-519`), and a recast there would land mid-flight. So the aim is inferred:
  - take the ray from the casting king tower through the sightings;
  - choose the point beyond the current position that hits the most of our value (units, buildings, towers).
- **A rolling Log** is re-rolled from its first sighting.
- **Area spells** are recast at their observed centre.
- A spell whose aim cannot be inferred is dropped, and the `spell_unrecast` flag is raised.

### 4.5 Labels (relabel)

Right after the reset, and before any step:
1. Match each entity to its spec by exact (team, card, x, y). Spawns are not snapped, and the materialisation order is canonical, not list order (`reset` docstring).
2. Record uid → the creating card from the public view's `card_id`.

Every after-state used for a row is rewritten with that map. A Witch's skeleton then reads as a Witch again, which is what live-compatible tokens expect. Children created during a trial already carry their parent's card.

This targets the PROGRESS:315-317 failure. A relabel-off ablation (M6) tests whether that hypothesis is right.

### 4.6 Fidelity score and fallbacks

```
fidelity = (hp share of public units spawned or recast)
           × 0.8 if the opponent elixir count is not exact
           × 0 if the reset was refused
```

- Units that no card produces count as lost. These are Golemite, Lava Pup, Elixir Golem halves, the Phoenix egg and death bombs.
- Below `min_fidelity` (0.7), the search is skipped and `d0` is played. Bad perception therefore degrades to the plain policy instead of misleading it.
- Fallbacks, each logged by reason:
  - reset refused;
  - overtime with unequal crowns;
  - more than 30% of the visible units dropped;
  - fewer than 2 trials ran by the deadline;
  - a broken worker pool;
  - three exceptions in one battle, after which search is off for the rest of that battle.

### 4.7 What a reset cannot carry

`reset` takes only the clock, decks, elixir, tower hp and unit spawns (Battle docstring). The only in-place edit is `debug_nudge`, a "TEST ENTRY POINT" that moves an entity. This plan does not use it.

| Lost in a reset | Effect for Hog 2.6 | Mitigation in this plan |
|---|---|---|
| Deploy timers | an enemy Hog, Ram or Giant just placed acts up to 1 s early; Cannon and Ice Golem pull timing is misjudged | deploy-timer recasts (R2) |
| Spells and projectiles in flight | an enemy Fireball on our Musketeer, a Goblin Barrel, a rolling Log | spell recasts with an inferred aim (R2) |
| Our Ice Spirit freeze; enemy Zap, Snowball or Log stun and knockback | the Ice Spirits + Cannon defence is misjudged | none in R2 (tag `freeze_active` in M3); the anchored window R3 if M3 shows it matters |
| Attack windups, cooldowns and target locks | every unit and tower re-acquires and re-winds its first hit | partly regenerated by the advance to `B`; otherwise none |
| Charge and Inferno ramp (read-only `charge_states`) | a Prince charge or Inferno ramp restarts | none |
| King activation (Snapshot-only) | a king woken by a Log or Fireball goes back to sleep | none; probe P6 |
| Spawner phase (Tombstone, Hut, Furnace, Witch) | waves restart from activation | none |
| Units no card produces | Golem, Lava and Elixir Golem pushes are underestimated | lowers the fidelity score; cardless proxies (S17) if M3 shows it matters |
| Evolution and hero forms | spawned as the base card | probe P5: do reset `forms` reach spawns? |
| Shields | probably spawned full | probe P12 |
| A tower troop other than the Princess | no Python option to choose one | flag `nonprincess_tower`; open question |
| The opponent's commands in the last ~22 ticks | unknowable live | a sampled in-flight play (section 5) |
| RNG and move order | new seed; canonical creation order | the same seed for every candidate (common random numbers) |

### 4.8 Rebuild levels

| Level | Contents |
|---|---|
| R0 | `RSAgent.rebuilt` exactly (opponent elixir 0, deck-order queue, no relabel). It exists only so the instrument can reproduce the author's finding |
| R1 | public view, sanitize, cycles, counted opponent elixir, relabel, permanence, building snap, one reused Battle |
| R2 | R1 plus deploy-timer and spell recasts |
| R3 (deferred, S16) | an anchored replay window: reset 1-5 s in the past, before any running deploy, flight or freeze, then replay both sides' observed plays as commands up to `t`. Built only if M3 shows that deploys, freezes and spells in flight dominate the regret |

## 5. Opponent handling

The old search's opponent was idle with 0 elixir. Against top-1000 players that makes every Hog look like it connects. This plan models the opponent from public information only.

**Card cycle** (`CardCycle`, used for both sides).
- With shuffle 0, a played card goes to the back of the queue and comes back to the hand only on the 5th play after it. So the last 4 plays are always 4 distinct cards: they are the queue, oldest first. The hand is the other 4 cards.
- The cycle is marked `exact = False` when:
  - a play was unknown (token 1);
  - the opponent played a Mirror;
  - more than 8 distinct cards were seen (a confusion);
  - the elixir counter reports an unaffordable play (`ElixirCounter.unaffordable`, `rs_elixir.py:578`), which means a play was missed.
- Mirror and Elixir Collector are swapped out of a starting hand (calibration `economy.OMIT_FROM_STARTING_HAND`). The deck order keeps them out of the first four (probe P14).

**Deck prior.** `DeckPrior` is built from the train split only, with `rs_match.load_pairs(split="train")` (`rs_match.py:1402`), so the gate's dev pairings never inform it.
- An unknown card c is drawn with weight Σ over decks d containing c of exp(β·|d ∩ known|), with β = 1.
- Before any prior exists, `sanitize_for_rebuild` fills the deck with Hog 2.6 cards (FILLER_NAMES).

**Elixir.** The count is clamped to [0, 10000]. When it is not exact, half of the worlds give the opponent 1000 more elixir. Overestimating the opponent is the safe side for us.

**Worlds** (`s_reply` only; K = 4, with common random numbers).
- **World 0** is the most likely one: top-prior cards, no slip, the counted elixir.
- **Worlds 1 to K−1** each draw:
  - unknown cards from `DeckPrior`;
  - with probability 0.1, a "cycle slip" that puts the oldest queued card back in the hand. The default noise prior misses 3% of troop plays and 15% of spells (`data/noise/default_prior.json`, `opponent_plays`).
  - position and hp jitter from the seat's own noise profile: σ 250 / 400 mt, and hp σ 0.03 on damaged units.
- **In-flight play.** The opponent may have tapped within the last `D_opp` ticks without anything showing yet. Each world samples one such play from the opponent model's gate on the opponent's row at `t`:
  - P(play) = 1 − p_wait^(D_opp/20);
  - the tile comes from the top 3 legal tiles;
  - it lands uniformly in (t, min(B, t + D_opp)].
  - `D_opp` is the spec delay of the other team in the simulator, and 22 live.
- **Reply.** At `B + ρ` (ρ uniform in [6, 16] ticks, shared by all candidates in a world), the opponent model decides on the trial state:
  - its gate is masked to the rebuilt hand and elixir;
  - it is sampled with the world's shared uniforms;
  - the reply lands at `B + ρ + D_opp`.

  All opponent rows go through one batched forward. The opponent's memory of our plays is a `PublicMemory` built from our own run plays.
- **Aggregation.** Each candidate's score is CVaR at α = 0.5 over the worlds (the mean of the worst half), plus the prior. This partly stands in for opponents who are stronger than an imitation sample.
- **Opponent model checkpoint.** `runs\tokens\live_big\best.pt`, the general model. It must not be a seat in the gate.
  - `live_big_hog` is fine-tuned from it, so the search may know its gate opponent too well.
  - `rs_league_eval` warns when the opponent model's digest matches a gate seat's.
  - M6 tests against an unmodelled opponent.

## 6. Compute and latency budget

No RoyaleSim timing exists anywhere in the repo. The figures below are prior estimates that the benchmark (S01) replaces.

The only speed evidence comes from the author's machine:
- 2,000 matches re-driven in 32 s (PROGRESS:83);
- "a few tens of milliseconds" for a whole-match re-drive (`tools/rs_shadow.py:8-9`).

### 6.1 Work per searched decision

| Preset | Engine ticks | `state_json` + `obs_row` | Batched forwards |
|---|---|---|---|
| `s_root` | ≈ D + replays ≈ 30 | 1 | 1 (gate and tiles share one pass) |
| `s_idle` | ≈ 30 + 8 × 60 ≈ 510 | ≈ 9 | 2 |
| `s_reply` | ≈ 4 × 24 + 8 × 4 × 80 ≈ 2,650 | ≈ 64 (32 reply rows + 32 after-rows) | 4 |
| teacher, for comparison | 672 (`teacher.py:73-77`) | | |

**The shared encoder pass (`BatchedPolicy`).**
- In `TokenPolicy._heads`, tile logits are `tile_head(cat[cls, card_emb(card_ids)])`, and `cls` does not depend on `card_ids` (`sim/model_tokens.py:251-272`). The model is in eval mode (`tools/rs_checkpoint.py:322`).
- So one `outputs` call gives the gate, the value and `cls`, and the tiles for any slot cost only the small tile head.
- Today `TokenPolicyAdapter.tiles` runs a full forward for each slot the plain decide tries (`rs_decide.py:661-664, 1285`). This change removes those forwards too.

**Prior estimates per stage:**
- rebuild + advance + save + root row: 4-8 ms;
- one trial (load + step 60 + `state_json`/loads + relabel + `obs_row`): 3-6 ms;
- value forward for 8 rows of `live_big` (d256, 6 layers): 2-5 ms on CUDA, 50-90 ms on one CPU thread.

So `s_idle` takes about 40-60 ms inline with a CUDA forward. `s_reply` takes about 150-400 ms on one core, so live it needs the worker pool or a lighter preset.

### 6.2 In the simulator

- Each `rs_league_eval` game already runs in its own worker process (`pool_play`, `rs_league_eval.py:833`), so the search runs inline, with no nested pools.
- A seat with an opponent model and Battles needs about 1.7-1.8 GB, against the default `WORKER_MB = 1200` (`rs_league_eval.py:166-168`). S08 adds `--worker-mb`, so 4-5 workers fit on 16 GB with MuMu and the browser closed.
- `MatchResult.timings["decide_s"]` measures each seat's wall time (`rs_match.py:942, 1039`).

### 6.3 Live

- The plain chain's target is capture-to-decision p95 ≤ 100 ms (`docs/SPARRING_PLAN.md:109`).
- **Blocking, with a deadline.** The search blocks with a hard deadline: 80 ms for `s_idle`, 120 ms for `s_reply`. After the deadline the bot plays `d0`.
  - The tracker's tentative tracks die after 6 unseen ticks, i.e. 300 ms (`scripts/vision/tracker.py:225-226`).
  - `can_decide` needs the frame to be at most 0.5 s stale (`vision_play.py:126`).
  - So a block of 120 ms or less starves nothing.
- **Anytime.** P and W are submitted first. Whatever has finished by the deadline is scored. If P or W is missing, the bot plays `d0`.
- **Worker pool (`--search-backend procs`).** A warm `ProcessPoolExecutor` (spawn context on Windows) of torch-free workers. Each worker:
  - does load + step + `state_json` + relabel + `obs_row`;
  - returns rows of a few KB.

  The main process runs one batched value forward on CUDA. Workers never touch adb, the emulator or the screen. On `BrokenProcessPool`, the battle continues on plain decisions.
- **Latency charge.** `latency_ticks = ceil(p95_search_ms / 50)` from the benchmark. It is added to the seat's command delay in every simulator game. If a dry run measures a higher p95, the gate is re-run with the measured value.
- **If the budget is exceeded,** cut in this order:
  1. the procs backend;
  2. H 40;
  3. 5 trials;
  4. K = 2;
  5. an asynchronous mode (submit, return WAIT `searching`, poll each frame).

## 7. Measurement in the simulator, and the gate

**The protocol** is L03 (`rs_league_eval.py:136-168`):
- delay 24;
- noise `default`;
- the dev split. It is named `val` internally, and `dev` is an accepted alias (`rs_match.py:184`; `rs_league_eval.py:136-138`).
- Hog 2.6 pairings (`--require hog26`);
- our side in base forms;
- both decks head to head.

Every search seat pays its `latency_ticks`.

### 7.1 Arms, fixed in advance

| Arm | Seat |
|---|---|
| `plain` | `live_big_hog`, lead 0: the current live configuration |
| `author_la` | `live_big_hog`, `lead=-24`: the author's rebuilt look-ahead as is. It runs today through `configure` (`rs_match.py:558-573`); reference only |
| `s_root`, `s_idle`, `s_reply`, `s_true` | SearchSeat with that preset (section 3.2) and the rebuild level M3 picks |

### 7.2 Stages

| Stage | What | Size | Stop or pass rule |
|---|---|---|---|
| M0 | Benchmark and engine probes (S01) | ~1 h | Any probe that contradicts section 4 is fixed in the rebuilder before anything else. Fixes `latency_ticks`, trials, H and K |
| M1 | Tests, and a 20-game smoke run with `verify` (restore checked by `state_json`, as `teacher.py:581-583`) | ~30 min | All pass; no leak (S08 test) |
| M2a | `s_root` vs `plain`, and `s_root` vs `author_la` | 400 games each | The look-ahead track goes on to M5 (c) if `s_root` scores above 50% against `plain`. If `s_root` loses to `author_la`, the new rebuild has a bug: fix it first. If `s_root` does not beat `plain`, run the relabel-off and R1/R2 ablations before parking the track |
| M2b | `s_true` vs `s_root`: the prize for search at delay 24 + noise, never measured before | 400 games | **Search track stops if `s_true` < 55% vs `s_root`.** The look-ahead track goes on |
| M3 | Regret instrument (S10) on about 2-3k sampled decisions from 50 dev pairings | 1-3 h | A public configuration (R0/R1/R2 × `s_idle`/`s_reply`, H in {40, 60, 80}) goes on only if its mean gain has a 95% CI above 0, recovery ≥ 0.3 and harm rate ≤ 15%. R0 must show about 0 gain, or the instrument is suspect. The search track stops if no configuration passes |
| M4 | Screen the best 1-2 public configurations vs `s_root` | 400 games each | ≥ 53%; plays per match within ±20% of `s_root`'s; refused plays not above `s_root`'s |
| M5 | **L03 gate** (`rs_league_eval run`): (a) best search arm vs `s_root`; (b) best search arm vs `plain`; (c) `s_root` vs `plain` | ≥ 1,000 games each | Wilson LB > 0.52; anchors `live_big\ep01.pt` and `floor=idle` within 2 pp (`rs_league_eval.py:156-159`) |
| M6 | Robustness, as differences from the M5 champion | 400 games each | noise `none`; doubled noise (S11) ≥ −1 pp; an unmodelled opponent (`dt_big_hog` or an RL league checkpoint) keeps the sign; `d0` with `TowersOnlyLegality`; relabel off; the opponent at 0 elixir; `latency_ticks` 0 vs measured; `data/noise/measured.json` once V11 fits it; optionally a rebuild on another card table than the match. A gain that vanishes there points to simulator-only tricks |
| M7 | Vision-shaped simulator check (S12): the seat builds its state through `events.StateTracks` (`events.py:1046`), then `PublicStateBuilder.state`, then SearchDecider's own path | 50 games | Rebuild refused in < 5% of searches; ≥ 90% agreement with the M5 seat's choices |

**The regret instrument (M3).**
- A probe seat plays `s_root`'s decisions, so the game follows the incumbent's path and `d0` is the search's incumbent. At sampled decisions where a trigger fires, it takes the candidate set C from the public root.
- **Truth T(c).** On the TRUE delayed battle: `save()`, then for each candidate `load` + `step([cmd at t], D + H)`, then `load` back, with the restore verified. The true after-state is projected to public before scoring, so the same scorer reads the same fields.
- **Opponent-error split.** For reply arms, ORACLE injects the same model replies into the true copy. ORACLE-TRUE-OPP replies with a separate instance of the real opponent's checkpoint on its true hand. The difference between them prices the opponent-model error.
- **Metrics:**
  - gain = T(chosen) − T(d0's choice);
  - oracle = max T − T(d0's choice);
  - recovery = mean gain / mean oracle;
  - top-1 agreement, Kendall τ and harm rate (chosen worse than `d0`'s choice by more than 0.05);
  - board errors at `B`: unit match rate, position RMSE, hp MAE, tower error, hand and elixir agreement.
- **Tags.** Every metric is broken down by:
  - an enemy building-targeter in our half;
  - a spell in flight;
  - an enemy unit deploying;
  - our freeze active;
  - a damaged king;
  - a cardless unit;
  - an enemy evolution or hero;
  - a hidden building;
  - opponent elixir ≥ 7;
  - our Hog alive;
  - an opponent command in flight (known only to the instrument).
- Confidence intervals come from a bootstrap clustered by match.

### 7.3 The result that justifies live use

- **A configuration goes live only if it passes the L03 gate against `plain`.** That is M5 (b), or M5 (c) for the look-ahead alone. It must also pass M6 and M7, and the live dry run must measure a p95 within its charged `latency_ticks`.
- **The search part ships only if it also passes against `s_root`** (M5 (a)). Otherwise the simpler `s_root` ships.
- Real friendlies are the final check. They are an owner-watched A/B log, not a statistical gate.

### 7.4 Statistics and honesty

- 400 paired games resolve differences of about 5 pp. Only M5 counts for promotion, and the arms are fixed before any run.
- The `--final` (test) split is played once, for the one reported result (`rs_league_eval.py:139-140`).
- `s_true` is privileged and can never be promoted. `rs_league_eval` labels any candidate whose search reads the true battle.
- The search code may use the true `battle` only through `d0`'s legality, which is the same call the champion makes. A test enforces this (S08).

## 8. How it plugs into the live runner

**`SearchDecider`** (`tools/live_search.py`) wraps today's `Decider` (`scripts/vision_play.py:736-774`). It has the same interface: `decide(vf, ctx) -> (Decision, state)`, plus `observer_for`. `VisionPlay._decide` is unchanged, and so are the tap path, `ScreenGuard`, the control logic and the dry-run logic.

**What `decide` does:**
1. It prunes `ctx.pending` exactly as now (`vision_play.py:766`).
2. It builds the state with the same `state_fn` (already public).
3. If search is off, not triggered, or held, it runs the base path.
4. Otherwise it gathers the search inputs:
   - our aims from `perception.deriver.pending_taps`;
   - our cycle from `perception.hud_reader.hand.state().cycle`;
   - the opponent's elixir and cycle from `ctx.memory`;
   - `B = vf.tick + D_land`, where `D_land` = the latency_probe median minus 2 sighting ticks (tracker `min_hits`), plus `latency_ticks`. Before any measurement it is 26 − 2 + `latency_ticks` (`scripts/vision/control.py:142`). The root row is dated `B`, as in the simulator.
5. `d0` is the same `RD.decide` call as today, with `TowersOnlyLegality` (`vision_play.py:1386-1389, 1419-1423`). Then come the candidates, the trials and the choice.

**Flags** (`scripts/vision_play.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--search {off,shadow,act}` | `off` | `off`: today's behaviour, and the way to turn search off. `shadow`: plays `d0` and logs the search's choice next to it. `act`: suggests the search's choice |
| `--search-preset NAME` | `s_root` | a preset from section 3.2 (`s_true` is refused live) |
| `--search-config JSON` | none | overrides of `SearchConfig` fields |
| `--search-backend {inline,procs}`, `--search-workers N` | `inline`, 4 | the trial runner |
| `--search-deadline-ms N` | from the preset | after this, play `d0` |
| `--search-land-ticks auto\|N` | `auto` | `D_land` |

**Startup and logging.** The flags are printed in the startup lines and in the `config` record. The pool starts in `_main` before the first battle and closes in its `finally` block (`vision_play.py:1495-1507`). Each decision record's `extra` gets the search record, and its stage times go to `ms["search_*"]`.

**What does not change:**
- `--search act` still taps only with `--tap`, and `--tap` only while the owner watches.
- The search returns a Decision; it never taps.
- The STOP file, F8 and `emulator.verify()` are untouched.
- Nothing touches the lobby or starts a battle.
- No new code imports `scripts/gamestate`.
- No anti-detection logic of any kind.
- New files must pass `tests/pipeline/test_friendly_only.py` and `tests/pipeline/test_no_machine_literals.py`.

## 9. Work items

Every test lives in `tests/pipeline` and skips cleanly:
- `engine` needs `royalesim`;
- `gpu` needs CUDA;
- `il_replay` needs the corpus;
- `device` is never needed here;
- tests that need torch use `pytest.importorskip("torch")`.

Logic tests use `conftest.FakeBattle` (`tests/pipeline/conftest.py:283-370`).

| ID | Title | Files | Depends |
|---|---|---|---|
| S01 | Benchmark and engine probes | `tools/rs_search_bench.py` (new); `tests/pipeline/test_rs_search_bench.py` | none for `engine`; the `search` subcommand needs S08 (and S13 for the procs backend) |
| S02 | Public view, card cycles, opponent elixir, permanence | `tools/rs_public_view.py` (new); `tests/pipeline/test_rs_public_view.py` | none |
| S03 | Deck prior from the train split | `tools/opp_model.py` (new); `data/opp_prior/deck_prior.json` (made on the PC); `tests/pipeline/test_opp_model.py` | S02 |
| S04 | Rebuilder (R0, R1, R2): reused Battle, reset, relabel, own in-flight replay, recasts, spell aim, building snap, fidelity, fallbacks | `tools/rs_rebuild.py` (new); `tests/pipeline/test_rs_rebuild.py` | S02 |
| S05 | BatchedPolicy: shared encoder pass, batched values | `tools/rs_search.py` (new, policy part); `tests/pipeline/test_batched_policy.py` | none |
| S06 | Search core: SearchConfig and presets, triggers, candidates, inline trials, scoring, trust region, Decision, fallbacks | `tools/rs_search.py`; `tests/pipeline/test_rs_search.py` | S04, S05 |
| S07 | Opponent worlds: determiniser, in-flight play, reply phase, CVaR, paired-SE override | `tools/opp_model.py`, `tools/rs_search.py`; `tests/pipeline/test_opp_worlds.py` | S03, S06 |
| S08 | Simulator seat and harness: SearchSeat, `AgentSpec.search` / `latency_ticks`, delay charge, CLI options, leak test | `tools/rs_search.py`; `tools/rs_match.py`; `tools/rs_league_eval.py`; `tests/pipeline/test_search_seat.py`, `tests/pipeline/test_search_no_leak.py` | S06 |
| S09 | True-copy trial mode (`s_true`, privileged) | `tools/rs_search.py`; tests in `test_rs_search.py` | S06, S08 |
| S10 | Regret instrument (M3) | `tools/rs_search_fidelity.py` (new); `tests/pipeline/test_search_fidelity.py` | S07, S09 |
| S11 | Doubled-noise stress profile | `tools/noise_scale.py` (new) → `data/noise/stress_2x.json`; `tests/pipeline/test_noise_scale.py` | none |
| S12 | Vision-shaped simulator check (M7) | `tools/live_search.py` (`simcheck` subcommand); `tests/pipeline/test_live_search.py` | S08, S13 |
| S13 | Live SearchDecider and worker pool | `tools/live_search.py` (new), `tools/rs_search_worker.py` (new); `tests/pipeline/test_live_search.py` | S06 (S07 for `s_reply`) |
| S14 | Runner wiring: flags, construction, log fields, pool shutdown | `scripts/vision_play.py`; `tests/pipeline/test_live_search.py` | S13 |
| S15 | Docs: results tables here, a decision entry in `docs/SPARRING_PLAN.md`, then a runbook section | `docs/LIVE_SEARCH_PLAN.md`, `docs/SPARRING_PLAN.md`, later `docs/SPARRING_RUNBOOK.md` | results; the runbook only after the other workflows' runbook edits merge |
| S16 | (deferred) Anchored replay window, R3 | `tools/rs_rebuild.py`, `tools/rs_effect_windows.py` → `data/rs_rows/effect_windows.json` | S04, S10, and M3 showing deploy, freeze or spell tags dominate |
| S17 | (deferred) Cardless-unit proxies (e.g. Golemite, Lava Pup), death damage subtracted in advance | `tools/rs_spawn_proxy.py` → `data/rs_rows/spawn_proxy.json`; `tools/rs_rebuild.py` | S04, S10, and M3 showing the cardless tag matters |
| S18 | (deferred) The public-rebuild search as a non-privileged R05 distillation teacher (no live latency) | `tools/rl/teacher.py`, `tools/rl/rollout.py` | S06; only after the RL workflow's edits to `tools/rl/*` merge |

### Details

**S01 Benchmark and probes.** The tool is read-only and offline: it never taps, captures or touches the emulator. It has three subcommands.

*`engine`* (royalegym venv) builds Battles with `make_battle` and times:
- construction;
- `reset` with 0, 10, 20 and 40 spawns;
- `step(k)` for k = 1, 10, 60 on busy boards;
- `state_json` + `json.loads`;
- `save` (and `len(save())`), `load`;
- `check_deploy`;
- thread scaling of `step` over 1-16 Battles (`step` releases the GIL, per its docstring).

It also runs these probes, each printing pass or fail:

| Probe | Question |
|---|---|
| P1 | Does one Skeletons spec make 1 entity? |
| P2 | Does a spawned unit have `deploy_ticks` 0? |
| P3 | Does a spawn's max_hp match `unit_hitpoints(card, 11)`? (hp 0 and hp above max are never sent) |
| P4 | Do reset `levels` change a spawn's level? |
| P5 | Do reset `forms` reach spawns? |
| P6 | What is `king_active` after a reset with the king below max? |
| P7 | What are the elixir rate and overtime at start_tick 2399, 2400, 3600 and 4800? Do unequal crowns in overtime end the battle at once? |
| P8 | Does `set_command_delay_ticks` persist across resets? |
| P9 | Does save → step 100 → load give the same `state_json` and `state_hash`? Is a spawned Tesla in `hidden_uids`? |
| P10 | Do two Battles reset with the same seed and commands give the same `state_hash`? |
| P11 | Does a RamRider or GoblinGiant spec bring its rider? |
| P12 | Does a Dark Prince spawn with its shield? |
| P13 | Does a played card get its full deploy time and match `formation_preview`? |
| P14 | Are Mirror and Elixir Collector kept out of the first four under shuffle 0? |
| P15 | Does a blob saved on one Battle load on a sibling Battle? |

*`search`* (buildabot venv) plays a few `rs_match` games of SearchSeat with profiling on. It then replays the recorded searches with each backend, worker count, horizon, trial count and device, and records:
- p50 / p95 / p99 per stage;
- IPC overhead;
- inline and procs giving identical trials;
- worker RSS;
- fallback rates.

*`report`* prints the table and a recommendation: the most trials within the budget, `latency_ticks`, and torch threads for the gate workers.

Tests:
- the report and recommendation logic on canned numbers (no marker);
- P1, P2, P8 and P9 as assertions under `engine`.

**S02 Public view.** It provides:
- `public_view(state, team, mem_view)` (section 4.1);
- `CardCycle.from_plays(plays)`, used for both sides;
- an opponent elixir wrapper (simulator: `MatchAgent.opp_elixir`; live: `SeenElixir` with lag 0 on `ctx.memory`);
- `Permanence`.

Tests (no marker):
- a noised state carrying the opponent's hand, elixir, pending plays and deck, hidden columns, spell aims, projectiles and an unseen enemy comes out with all of them removed;
- the cycle queue and hand, including fewer than 4 plays, unknown plays and Mirror;
- a hidden Tesla is kept by permanence, with its hp drained.

**S03 Deck prior.** `python tools/opp_model.py build-prior --split train`, plus `DeckPrior.sample(known, n, rng)`.

Tests:
- sampling on a tiny hand-made prior (no marker);
- under `il_replay`, the built prior contains no dev or test pairing.

**S04 Rebuilder.** It provides:
- `RootSpec`, picklable, keyed by a sha1 of its canonical form;
- `Rebuilder(engine_cfg, rs, spawn_table)`, which holds one Battle;
- `root(spec) -> Built | None`;
- `relabel(state)`;
- `legal(card, x, y)` at `B`;
- `fidelity`.

A `level` switch selects R0, R1 or R2. Switches from the probes: one spec per unit or per formation (P1), and recasts on or off (P2).

Tests:
- With FakeBattle (no marker):
  - deck order with pending cards and recasts gives back exactly H and Q;
  - recast costs are added to that side's elixir;
  - spawns are clamped;
  - a refused reset falls back with a reason;
  - the fidelity arithmetic;
  - R0's reset arguments equal `RSAgent.rebuilt`'s, captured with a monkeypatched Battle.
- Under `engine`:
  - the relabel round trip covers every spawn;
  - a recast has `deploy_ticks` > 0;
  - a root rebuilt twice gives the same `state_hash`.

**S05 BatchedPolicy.** It has `TokenPolicyAdapter`'s interface (`batch`, `heads`, `gate`, `tiles`; `rs_decide.py:612-664`), plus `values(rows)`, with one `outputs` call per batch.

Tests (torch via `importorskip`): on a tiny random TokenPolicy on CPU,
- gate and tiles equal `TokenPolicyAdapter`'s within 1e-5;
- `rs_decide.decide` gives the same decision draw for draw.

**S06 Search core.** Section 3, inline backend.

Tests (FakeBattle, no marker):
- candidate order, pruning and the cap rule removing W;
- the trust region keeps P below τ;
- only P and W finished → those two are scored; P missing → `d0`;
- every fallback reason;
- pending plays land during the advance;
- slots are remapped by card;
- a chosen Decision has the fields `vision_play._decide` reads;
- the search never draws from the seat's random generator.

**S07 Opponent worlds.** Section 5.

Tests (no marker):
- world 0 is the most likely world;
- slip and elixir bump frequencies over many seeds;
- CVaR and the paired-SE override on hand-made score matrices;
- the reply lands at `B + ρ + D_opp`.

**S08 Seat and harness.**
- `SearchSeat(rs_match.MatchAgent)` (`rs_match.py:467`): `RSSearchAgent` cannot take `build_seat`'s keyword arguments (`rs_eval.py:604-606` vs `rs_match.py:580-584`), so this is a new class.
- `AgentSpec` (`rs_match.py:226-242`) gains `search: Mapping | None = None` and `latency_ticks: int = 0`. Both are validated in `__post_init__`.
  - `to_dict` leaves them out at their defaults, as it does for `presses` (`rs_match.py:275-283`). Existing game keys and identities therefore stay the same (`rs_league_eval.py:518-548`).
  - `from_dict` refuses unknown keys (`rs_match.py:286-293`), so these fields must exist on the class.
- `MatchSpec.engine_config` (`rs_match.py:397-405`; it already takes a (blue, red) pair) adds each agent's `latency_ticks` to its own team's delay. The charge follows the agent through `swapped()`.
- `build_seat` (`rs_match.py:576-586`) builds a SearchSeat when `agent.search` is set. It imports `rs_search` lazily and passes the seat's own and opponent delays from the spec, never from the battle.
- `seat_info` (`rs_match.py:589-605`) gains a search summary: searches, fallbacks by reason, overrides by from → to card, vetoes, fidelity skips and ms per stage.
- `rs_league_eval`:
  - `latency_ticks` goes in `_INT_OPTS` and `search` in `_STR_OPTS`, as a preset name (`rs_league_eval.py:369-372`). JSON players already work (`parse_player`, `:398`).
  - New `--worker-mb` option for `worker_cap` (`:805`).
  - The warning for an opponent model that matches a seat.
  - The privileged label for `s_true`.
- The `rs_match` CLI gains `--search-a/--search-b` (a preset name or JSON), `--latency-a/--latency-b` and `--search-verify` (check every restore, as `teacher.py:581-583`).

Tests (no marker):
- `AgentSpec.to_dict` is unchanged with the defaults;
- round trip;
- `engine_config` with latency, plain and swapped;
- `build_seat` dispatch.

Leak test (FakeBattle):
- two seen states that differ only in privileged fields give an identical `RootSpec` and decision;
- a battle proxy records every attribute the seat touches, and only `d0`'s `check_deploy` is allowed.

**S09 True-copy mode.** It runs the teacher's trial loop on the true delayed battle: one `save`, then for each candidate `load` + `step([cmd], D + H)`, then `load` (`teacher.py:509-601`). It uses the same candidates and scoring as the public search.

Test (FakeBattle): the battle is restored after a search.

**S10 Regret instrument.** Section 7.2.

Tests (no marker):
- gain, recovery, harm, Kendall τ and the clustered bootstrap on hand-made data;
- the probe seat's game equals the `s_root` seat's on FakeBattle (the instrument never changes the trajectory).

**S11 Stress profile.** It doubles every σ and probability in a noise profile, with probabilities capped at 1.

Test (no marker): the profile loads, and each field is doubled or capped.

**S12 Vision-shaped check.** A seat variant: `StateTracks` → (optionally `EventDeriver`) → `PublicStateBuilder.state` → the SearchDecider path, against the true engine.

Test (`engine`): a short smoke game.

**S13 Live SearchDecider and pool.** Section 8.

Tests (no marker, with fake perception, pool and clock):
- the dry run never taps;
- a missed deadline gives `d0`;
- a pool crash turns search off for that battle;
- STOP still stops;
- no new tap path exists;
- a worker imports no torch, checked with a subprocess as `tests/pipeline/test_hud.py:1030-1033` checks heavy imports.

**S14 Runner wiring.**

Tests:
- `--search off` gives today's decisions exactly;
- the flags parse;
- `s_true` is refused;
- `test_friendly_only.py` and `test_no_machine_literals.py` still pass.

## 10. PC steps for the owner

Commands use the runbook's `$RG` (royalegym) and `$BB` (buildabot) variables. Each step needs the work items named in it.

Before the benchmark and every long run:
- close MuMu and the browser (16 GB RAM);
- run no training at the same time.

1. **Update and test** (after S01-S08):
   ```powershell
   git pull
   & $RG -m pytest tests/pipeline -q
   & $BB -m pytest tests/pipeline -q
   ```
   Send back any failures.
2. **Engine benchmark and probes** (S01, about 10 min):
   ```powershell
   & $RG tools\rs_search_bench.py engine
   ```
3. **Search benchmark** (S01, S08, and S13 for the procs backend; about 30 min):
   ```powershell
   & $BB tools\rs_search_bench.py search --ckpt runs\tokens\live_big_hog\best.pt --opp-ckpt runs\tokens\live_big\best.pt --games 6 --delay 24 --noise default
   & $RG tools\rs_search_bench.py report
   ```
   Commit or send back `reports\search_bench\*.json`. The plan's knobs are set from it before the next step.
4. **Deck prior** (S03):
   ```powershell
   & $RG tools\opp_model.py build-prior --split train
   ```
5. **Smoke** (M1): 20 games, `s_root` and `s_idle` against `plain`, with `verify` on:
   ```powershell
   & $BB tools\rs_match.py --a runs\tokens\live_big_hog\best.pt --b runs\tokens\live_big_hog\best.pt --search-a s_idle --latency-a <bench> --search-verify --pairs 10 --seats 2 --delay 24 --noise-a default --noise-b default --out reports\search_smoke.jsonl
   ```
6. **Look-ahead and ceiling** (M2, about 4 × 400 games). These are `rs_match` screens with `--pairs 200 --seats 2 --workers 4`:
   - `s_root` vs `plain`;
   - `s_root` vs `author_la` (`--lead-b -24`);
   - `s_true` vs `s_root`.

   Send back the report lines.
7. **Regret instrument** (M3, S10, 1-3 h):
   ```powershell
   & $BB tools\rs_search_fidelity.py --ckpt runs\tokens\live_big_hog\best.pt --ref runs\tokens\live_big_hog\best.pt --opp-ckpt runs\tokens\live_big\best.pt --pairs 50 --delay 24 --noise default --rate 0.2 --workers 4 --out reports\search_fidelity\m3.jsonl
   ```
8. **Screens** (M4): the configurations M3 keeps, 400 games each against `s_root`.
9. **Gate** (M5, each about 2-5 h on 4-5 workers, unmeasured). For example (b):
   ```powershell
   & $BB tools\rs_league_eval.py run --candidate "srch=runs\tokens\live_big_hog\best.pt,search=<arm>,latency_ticks=<bench>" --champion "plain=runs\tokens\live_big_hog\best.pt" --anchor "ep01=runs\tokens\live_big\ep01.pt" --anchor "floor=idle" --delay 24 --noise default --workers 4 --worker-mb 1800 --log runs\league\search_b.jsonl
   ```
   Then (a), with the champion `"root=runs\tokens\live_big_hog\best.pt,search=s_root,latency_ticks=<bench>"`, and (c), `s_root` against `plain`.
10. **Robustness** (M6) and the **vision-shaped check** (M7):
    ```powershell
    & $BB tools\live_search.py simcheck --games 50
    ```
11. **Dry-run friendly** (S14). The owner starts the friendly; the bot only watches:
    ```powershell
    & $BB scripts\vision_play.py --search shadow --search-preset <winner>
    ```
    Send back the log, to compare p95 search ms with the charged `latency_ticks` and to see the agreement with `d0`.
12. **Played friendlies**, only while the owner watches:
    ```powershell
    & $BB scripts\vision_play.py --tap --search act --search-preset <winner>
    ```
    Alternate with `--search off` against the same clanmates, and log the results.

## 11. Risks

- **The fidelity ceiling.** `reset` cannot carry attack timers, target locks, freezes, charge, king activation or spawner phase. The search may again add about nothing. M2b and M3 stop it cheaply if so.
- **The relabel hypothesis may be wrong.** Then the look-ahead adds about nothing for live-compatible models, as it did for `rs_live60_hog`. M6 has a relabel-off ablation, and M2a decides.
- **The gap between the simulator and the real game.** RoyaleSim predicts the right winner 62.6% of the time (`docs/SPARRING_PLAN.md:48`), and the search optimises the simulator's own dynamics. Guards:
  - candidates come from the policy's own support;
  - the prior term;
  - the margin and the paired standard error;
  - jitter across worlds;
  - the override audit;
  - the engine-mismatch run;
  - live A/B friendlies.
- **A weak or inflated opponent model.** It is an imitation of anonymous ladder players, and the clanmates are top-1000. CVaR helps only partly. A shared lineage with the gate opponent inflates the simulator result; M6 tests against an unmodelled opponent.
- **Bias between waiting and playing.** Against an idle opponent, plays look better than they are. Guards:
  - the leak term;
  - the trust region;
  - triggers;
  - plays per match monitored in M4.
- **Leaks in the simulator harness.** Seats receive the true battle and a privileged noised state. Guards: `public_view`, the leak test, and privileged data only in `s_true` and the instrument.
- **Engine unknowns.** Probes P1-P15 settle them before any game. One bad spec fails a whole reset, and the fallback covers that.
- **Compute.** PC engine speed is unknown, so a gate may take more than 5 h. The benchmark sets the budget, and the cut order is fixed (section 6.3).
- **Live latency and jitter.** A late search falls back to `d0`, which the simulator charge does not model. A dry run must show the p95 inside the charge, or the gate is re-run.
- **Memory on 16 GB.** About 1.8 GB per worker with an opponent model, so 4-5 workers, no training in parallel, and MuMu and the browser closed.
- **Predictability against top players.** `d0` stays sampled, and overrides happen only on a margin, but a deterministic search pick may repeat. Review the override audit, and review losses in friendlies.
- **Process pool on Windows.** Spawn start-up, pickling and `BrokenProcessPool`. The pool is warm before battles, and the battle falls back to plain.
- **Model change.** If a decision-time model replaces `live_big_hog`, the root look-ahead no longer applies (its row is the present one). The search still does.

## 12. Open questions

1. The probe answers P1-P15, especially:
   - the spawn count of multi-unit cards;
   - the deploy state of spawns;
   - whether the delay persists across resets;
   - whether save and load are exact;
   - what overtime does with unequal crowns.
2. PC engine speed, `save()` size, and the per-stage search cost. These decide which presets are affordable live.
3. Does the relabel recover the look-ahead's gain for live-compatible models?
4. Does the search's prize survive delay 24 plus noise (`s_true` vs `s_root`)?
5. The live clock offset (`hud.py` `offset_ticks` 0, unmeasured). It shifts `start_tick` and the 2× elixir boundary in every rebuild.
6. The real decision-to-landing delay (`tools/latency_probe.py`). Until it is measured, 26 ticks to first sighting is assumed (`control.py:142`).
7. How often clanmates use a tower troop other than the Princess. RoyaleSim's Python API cannot choose one.
8. Which opponent model to use, and which unmodelled opponent to use for M6. `live_big` shares a lineage with `live_big_hog`.
9. In the simulator, should `d0` keep the champion's `BattleLegality` (for exact pairing) or use the live `TowersOnlyLegality`? This plan keeps pairing, and measures the other in M6.
10. Does the measured noise profile (`data/noise/measured.json`, V11) change the result? Re-run M5 when it exists.
11. Is the anchored replay window (S16) or are the cardless proxies (S17) worth building? M3's tags decide.

## Appendix A. API facts this plan relies on

| Fact | Where |
|---|---|
| `Battle.reset(seed, decks, shuffle, start_tick, elixir_milli, tower_hp, spawns, forms=None, levels=None, tower_levels=None)`; spawns are `(team, card_id, x, y, hp or -1)`, materialised in canonical order, never list order | `royalesim.Battle.reset` docstring |
| `set_command_delay_ticks(blue, red)` applies from the next `reset`; hand and elixir change only when a command runs; `command_delay_ticks()` is what the next reset uses | Battle docstrings |
| `step(commands, ticks)` releases the GIL, stops at game over, returns (card, reason, tick, resolved x, resolved y); a building that does not fit is moved, not refused | `Battle.step` docstring |
| `debug_nudge` is a test entry point that moves an entity | `Battle.debug_nudge` docstring |
| Spawns: troop or building cards only, centre in the arena, non-flyers off the water, hp −1 or 1..card hitpoints in the adapter (the core accepts any hp ≥ 0) | royalegym 0.1.42 `protocol.py` (`SpawnSpec`, MatchSetup rules) |
| `RSAgent.rebuilt`, `RSAgent.ahead` with `lead_rebuild`, `RSSearchAgent` | `tools/rs_eval.py:324-391`, `269-300`, `593-727` |
| `RSAgent.act` dates the row at `tick + lead + shift` and calls `rs_decide.decide` with `BattleLegality(battle)` | `tools/rs_eval.py:525-552` |
| `AgentSpec`, `to_dict` omitting `presses`, `from_dict` refusing unknown keys | `tools/rs_match.py:226-242, 275-283, 286-293` |
| `MatchSpec.swapped`, `engine_config` (int or (blue, red) delay) | `tools/rs_match.py:383-391, 397-405` |
| `MatchAgent`, `opp_elixir`, `configure`, `build_seat`, `seat_info` | `tools/rs_match.py:467, 505, 558-573, 576-586, 589-605` |
| `run_match`: delays, `seat.act(battle, seen, …)`, `decide_s`, `exec_tick = tick + delays[team]`, `seat.sent` | `tools/rs_match.py:853, 941-942, 949, 961, 1039` |
| Split aliases (`dev` → `val`), `load_pairs` | `tools/rs_match.py:184, 1402`; `tools/rs_league_eval.py:136-138` |
| L03 defaults: delay 24, noise default; 1,000 games, LB 0.52, anchors 400 within 2 pp; worker memory | `tools/rs_league_eval.py:142-143, 156-159, 166-168` |
| Player options, `parse_player`, `worker_cap`, `pool_play`, `run` subcommand | `tools/rs_league_eval.py:369-372, 398, 805, 833, 1507` |
| Tile head depends on `cls` and the card embedding only; eval mode | `sim/model_tokens.py:251-272`; `tools/rs_checkpoint.py:322` |
| `TOP_TILES`, `obs_delay_ticks`, `DecideConfig.live`, `Pending`, `prune_pending`, `TokenPolicyAdapter`, `BattleLegality`, `TowersOnlyLegality`, `threat_lane`, `Decision` (frozen), `decide`, `gate_probs` | `tools/rs_decide.py:151, 270, 323, 365, 403, 612-664, 691, 894, 1018, 1049-1050, 1160, 1312` |
| `tower_swing`, `battle_delay`, `seat_row_fn`, `_batched_values`, `policy_values`, `annotate` (verify), `_candidates`, `_play_cmd` | `tools/rl/teacher.py:282, 288, 302, 318, 340, 509-601 (581), 603, 708` |
| `reward_parts`, `result_of`, `potential` | `tools/rl/ppo.py:263, 275, 285` |
| `ENTITY_DEFAULTS`, `PublicStateBuilder`, `_spell_row`, the public opponent block, `OpponentElixir`, `OwnTaps`, `LiveObserver`, `rebuild_card`, `_deck8`, `sanitize_for_rebuild` | `tools/public_state.py:207, 298, 510, 632-635, 683, 750, 801, 865, 920, 935` |
| `HORIZON` 200, status bits, `seen_by`, `SeenElixir`, `towers_vec`, `obs_row` | `tools/rs_rows.py:290, 330-334, 695, 852, 1658, 1670` |
| `ElixirLaw`, `regen`, `ElixirCounter`, `unaffordable` | `tools/rs_elixir.py:153, 282, 365, 578` |
| `_copy_state`, `NoiseModel.apply`, `MatchNoise.observe` | `tools/perception_noise.py:736, 1006, 1338` |
| `EventDeriver.card_of`, `track_cards`, `pending_taps`; `StateTracks` | `scripts/vision/events.py:727, 741, 759, 1046` |
| `HandState.cycle` | `scripts/vision/hud.py:1035-1039` |
| Tracker `min_hits` 2, `tentative_ttl_ticks` 6 | `scripts/vision/tracker.py:225-226` |
| `DEFAULT_LAG_TICKS = 26` | `scripts/vision/control.py:142` |
| `MAX_STALE_S`, `Decider`, `decide`, `make_observer`, `_after_perception`, `_decide`, Pending append, legality and decider construction, runner `finally` | `scripts/vision_play.py:126, 736-774, 841, 1036, 1088, 1119, 1389, 1419-1423, 1495-1507` |
| `FakeBattle`; markers | `tests/pipeline/conftest.py:47-60, 283-370`; `pyproject.toml:45-49` |

## Appendix B. Implementation traps

- **The opponent is `1 - team` in the simulator.** Swapped specs put the search seat on team 0. Only live is the bot always team 1.
- **A public spell row's aim is a placeholder** (the current position, `public_state.py:510-519`). Never recast a projectile there.
- **Rebuilt uids never match the original uids or the list order.** Match by (team, card, x, y).
- **The core accepts any hp ≥ 0.** Clamp spawns to [1, max] yourself.
- **One bad spec fails the whole reset.**
- **Keep the rebuild battle undelayed** (`cfg.undelayed()`) and time every command by hand. A delay set after a reset does not affect the running battle.
- **`Decision` is frozen.** Build a new one with `dataclasses.replace`.
- **Date the root row at `B`** in the simulator and live alike (`rs_eval.py:542-544`).
- **Never draw from the seat's random generator inside the search**, or pairing with `s_root` breaks.
- **The new `AgentSpec` fields must be omitted by `to_dict` at their defaults**, or every logged game key changes.
- **The memory-reading route to unit types (PROGRESS:318-321) is closed** by CLAUDE.md. Do not use `scripts/gamestate`.
