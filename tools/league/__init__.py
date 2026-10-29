"""League evaluation for the sparring bot: statistics, ratings and (later) gates.

    stats   Wilson intervals, paired bootstrap CIs for two agents on the same
            pairings, exact McNemar, a Bradley-Terry-Davidson MAP rating fit
            with an anchor, and an online Elo readout for dashboards

Everything here is pure numpy and runs offline on result logs; nothing imports
the engine, torch or the memory reader. The design follows royalelearn's
ladder/rating.py (MIT), which is not imported.
"""
