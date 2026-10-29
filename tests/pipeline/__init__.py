"""Tests for the sparring pipeline: RoyaleSim rows, token model, vision, RL, machine setup.

Run them on their own; they do not depend on the legacy suite in tests/ (which needs
the gitignored tmp/gamedata and has known failures):

    .venvs\\royalegym\\Scripts\\python.exe -m pytest tests/pipeline -q

Shared fixtures and the skip markers live in tests/pipeline/conftest.py.
"""
