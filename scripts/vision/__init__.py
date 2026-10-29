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
    templates template store (data/vision/templates, captured from your
              client on the PC), numpy NCC, digit OCR, capture CLI (V07)
    hud       HUD readers: clock -> tick, elixir_milli, tower HP, hand with
              a FIFO prior and the next card; HudReader -> types.Hud (V07)
    hpbars    unit HP-bar fill and bar -> unit attachment (V07)
    control   human takeover: AUTO / MANUAL toggled by a hotkey (F8), an
              always-on-top window or a MANUAL file; bot suggestions, and own
              plays attributed to the bot or the human (V12)
    events    opponent plays (children, repeats and re-acquired tracks are
              not plays) with estimated execution ticks for PublicMemory, own
              taps matched to landings (latency, misses), and the card that
              created each track (--live-compat identity) (V04; stdlib only)
    labels    detector labels: classes.json crosswalk x side ("HogRider@enemy"),
              read back into class / form / spell card; dataset label
              conversion for scripts/prepare_yolo_dataset.py (V09; stdlib only)
    detector  YOLO on the arena crop only, boxes off the board dropped,
              ally/enemy boxes merged into side_prob, names mapped through the
              class list; weights pinned by SHA-256, device from torch.cuda;
              ultralytics imported lazily (V09)

The class list they share lives outside the package: tools/vision_classes.py
-> data/vision/classes.json (E16) maps every RoyaleSim unit type to a vision
class and form (types.FORM_*), names the card that creates each class (the
unit id --live-compat rows use) and crosswalks the KataCR detector classes.

Importing this package (or any module above) needs only numpy; tracker needs
only the standard library (scipy, if installed, for Hungarian matching), and
so does control (it loads tkinter and ctypes lazily, in their own threads).
OpenCV, Pillow and torch are imported lazily, inside the functions that need
them (the HUD readers never need OpenCV: their NCC is numpy).
"""
