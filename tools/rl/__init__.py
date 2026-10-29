"""Reinforcement learning for the sparring bot, after imitation (tools/rs_rows.py ->
sim/train_tokens.py).

    league      who the learner plays next: permanent anchors (imitation checkpoints),
                learner snapshots admitted every N iterations, decayed win-rate
                estimates and PFSP "hard" weights with a uniform floor (R03)

Planned beside it: rollout (R01, vectorised RoyaleSim battles), ppo (R02) and
train_loop (R04, the driver that replaces tools/rs_selfplay.py's loop). The deck field
the opponents play comes from tools/deck_stats.py.

This file imports nothing on purpose: each module states its own dependencies, so
importing one never pulls in torch or the engine for another. Nothing here imports the
memory reader (scripts/gamestate/memread.py). The design follows royalelearn's ladder
(MIT), which is not imported.
"""
