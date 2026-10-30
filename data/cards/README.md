# data/cards: the IL_Replay to RoyaleSim card map

IL_Replay names each card by its RoyaleAPI key, such as `hog-rider` or `the-log`, with `-ev1` or `-hero` added for a form. RoyaleSim names it by the client's name, such as `HogRider` or `Log`. Every re-drive of a recording goes from one to the other through `tools/rs_cards.py`. That covers `tools/rs_rows.py`, `tools/royalesim_redrive.py`, and `tools/rs_eval.py` through `rs_rows._setup`. A match with a card the map lacks is skipped.

| File | What it is |
|---|---|
| `replay_cards.json` | `royaleimitate/replay_cards.json` from the royaleimitate 0.2.15 wheel, byte for byte. It has 119 keys. SHA-256 `aeca8fdf8eebafadbe3b1f4dba6eebb6132715e935bfe1db63f5968670c039ee`. The wheel's RECORD gives the same digest. |
| `LICENSE-royaleimitate.txt` | royaleimitate's MIT license, copied unchanged from the wheel's `royaleimitate-0.2.15.dist-info/licenses/LICENSE`. |
| `replay_cards_extra.json` | Keys the vendored file lacks: `void` → `DarkMagic`, `rune-giant` → `GiantBuffer`, `spirit-empress` → `MergeMaiden`, `minion-giant` → `MinionGiant`. Each has `"verified": false` until the coverage check below finds it in the corpus. Minion Giant (released 2026-09-07) stays unverified: no match of the 4 parts checked so far has it, and the corpus is believed, not shown, to predate it; a full `--coverage` run settles it. |
| `.gitattributes` | Keeps git from rewriting `replay_cards.json`'s line ends. |

`rs_cards.load()` does three things:

- It refuses `replay_cards.json` unless its SHA-256 matches. CRLF is read as LF, so a Windows checkout still passes.
- It refuses either file if a key appears twice in it.
- It refuses an extra that gives a key already in the vendored file a different name.

It returns the 123-key merged map. Set `HASTYCR_CARD_EXTRAS=0` to get the vendored map alone, exactly what the tools read from royaleimitate before. Use it to compare a run with older runs on the same matches. `tools/rs_rows.py` writes the map it used to `vocab.json` under `"card_map"`: the entry count, the SHA-256 of its canonical JSON, and the extras.

```powershell
.venvs\royalegym\Scripts\python.exe tools\rs_cards.py                 # check the files, list the extras
.venvs\royalegym\Scripts\python.exe tools\rs_cards.py --catalogue     # every name in the engine's catalogue
.venvs\royalegym\Scripts\python.exe tools\rs_cards.py --coverage      # unmapped deck keys; are the extras seen?
```

**Confirming an extra.** Run `--coverage`. Read it in two parts:

- **Extras.** When an extra's key shows up in corpus decks, the key is right. Set its `"verified"` to `true` in `replay_cards_extra.json`.
- **Unmapped keys.** Any other key that blocks many matches is a candidate for a new extra. Give it the RoyaleSim catalogue name: `--catalogue` checks the name, and so does the engine-marked test in `tests/pipeline/test_rs_cards.py`.

**Taking a newer royaleimitate map.** Copy its `replay_cards.json` over this one unchanged. Then put the new `sha256sum` in `VENDORED_SHA256` in `tools/rs_cards.py` and in this file. Copy its LICENSE too if the license changed. Drop any extra the new file now covers. A changed map changes which matches are kept, so rebuild `data/rs_rows`.

## royaleimitate license

`replay_cards.json` is used under the following notice. The same text is in `LICENSE-royaleimitate.txt`.

```text
MIT License

Copyright (c) 2026 RoyaleGym contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
