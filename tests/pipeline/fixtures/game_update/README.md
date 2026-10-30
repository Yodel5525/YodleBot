# tests/pipeline/fixtures/game_update

Hand-written answers for `tests/pipeline/test_game_update_check.py` (U04 of
`docs/SEASON_UPDATES.md`). No test touches the network: the tests replace
`tools/game_update.py`'s `fetch` with a router that answers from these files.

| File | Shape | Source of the shape |
|---|---|---|
| `pypi_royalesim.json` | PyPI project JSON, `GET https://pypi.org/pypi/royalesim/json`, trimmed to a few releases | https://docs.pypi.org/api/json/ and the live answer read on 2026-10-30 (keys `info`, `last_serial`, `releases`, `urls`, `vulnerabilities`; a file has `filename`, `packagetype`, `digests.sha256`, `url`, `upload_time_iso_8601`, `yanked`) |
| `pypi_royaleimitate.json` | the same for royaleimitate | the same |
| `hf_il_replay.json` | Hugging Face dataset info, `GET https://huggingface.co/api/datasets/VanguardX101/IL_Replay` | https://huggingface.co/docs/hub/api (`id`, `sha`, `lastModified`, `siblings`, ...). huggingface.co could not be reached from the sandbox, so the shape is unconfirmed. |
| `cards_api_2026-10-30.json` | official API `GET /v1/cards` (`items`, `supportItems`) | the Swagger spec and API_NOTES.md that `tools/cr_api.py` cites |
| `cards_api_2026-11-02.json` | the same after a made-up Season 89 | the same |

What is real and what is invented:

- The 0.1.41 and 0.1.42 entries of `pypi_royalesim.json` and the 0.2.15 entries of
  `pypi_royaleimitate.json` carry PyPI's real file names, digests and upload times. Everything
  newer (0.1.43, 0.1.44rc1, 0.2.16) is invented. A digest of `"FIXTURE"` is replaced by the
  test with the sha256 of the tiny archive it builds for that file name.
- The `/v1/cards` card ids are the official ids from `data/royaleapi/cards.json`, except
  Minion Giant (26000100) and Fixture Golem (26000999), which are invented. The
  `maxEvolutionLevel` values follow the forms in `data/vision/classes.json` on 2026-10-30, so
  the 2026-10-30 file shows no gap. The 2026-11-02 file gives Giant and Mirror an evolution
  (Season 89), renames The Log to "Log" and adds Fixture Golem.
- The Hugging Face `sha` values are invented.
