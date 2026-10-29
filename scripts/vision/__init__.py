"""Screen-vision perception for the sparring bot (friendly battles only).

The modules here turn emulator frames into the public game state the token
model reads. They never touch the memory reader (scripts/gamestate/memread.py)
and they never navigate menus or queue for battles.

    geometry  screen pixels <-> RoyaleSim world millitiles, tap targets,
              card-slot centres, ability-button tiles, 9:16 frame check
    types     the VisionFrame contract (Detection, Track, Hud, Events)
    record    JSONL + image recorder with a disk budget, and its reader
    capture   frame sources: threaded PrintWindow, adb screencap, file replay
    tracker   position-first tracker with label and team voting (V03)

Importing this package (or any module above) needs only numpy; tracker needs
only the standard library (scipy, if installed, for Hungarian matching).
OpenCV, Pillow and torch are imported lazily, inside the functions that need
them.
"""
