# geoguessr-AI

A self-trained AI that plays [OpenGuessr](https://openguessr.com) by watching your screen and moving your mouse. It never reads the page's code.

There are three commands:

| Command | Use it to |
| --- | --- |
| `geoguessr-ai play` | play a game on its own |
| `geoguessr-ai friends` | play against your friends in a multiplayer room |
| `geoguessr-ai train` | make the model better, from the rounds it has played |

## Install (once)

In Windows PowerShell, in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
geoguessr-ai train --shards 0
```

- The last line builds the first model. It downloads about 5.4 GB of street photos, then trains, which takes a while. Skip it if `models/geoguessr.pt` already exists.
- Run `.\.venv\Scripts\Activate.ps1` again in every new terminal. If PowerShell says scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Play on its own

1. Open [openguessr.com](https://openguessr.com) and start a **Singleplayer** game. Put the browser window where it will stay.
2. Run:
   ```powershell
   geoguessr-ai play --rounds 10
   ```
3. **The first time only**, it asks you to show it where the game is. For each step, press Enter in the terminal, then hover the mouse over the spot it names (don't click) within 3 seconds. Near the end it asks you to make a guess yourself, so it can see the Continue button. Then press Continue so that a round shows.
4. You have 5 seconds to click on the browser window. From then on the bot moves the mouse.

**To stop it,** press **F8**, or push the mouse into a corner of the screen.

Add `--learn` to also read and save each round's real location, so that `train` can learn from it. Rounds take a few seconds longer.

## Play against friends

The bot needs a player of its own: run it in a browser where nobody else is playing, and let a friend host the room.

1. A friend hosts a room (**Multiplayer > Host**) on their own device and sends you the room code.
2. In the bot's browser, join the room (**Multiplayer > Join**).
3. Run:
   ```powershell
   geoguessr-ai friends
   ```
4. **The first time only**, it asks you to show it where the game is, as in step 3 above. A round must be showing for this, so ask the host to press Start first.
5. Click on the browser window within 5 seconds and leave it alone.

What it does:

- It plays each round the host starts, and waits on the result screens until the host presses Continue. It never presses Continue itself.
- It keeps track of the round's timer and hurries when time is short. In **Duel** style, a guess cuts everyone else's time to 15 s. When another player guesses, the bot stops looking around and guesses straight away.
- It plays until you stop it with **F8**, or for `--rounds 5` rounds.
- It saves every round with its real location, for `train`.

## Make it better

```powershell
geoguessr-ai train
```

It learns from every round saved by `play --learn` and `friends` (in `runs/`). It switches to the new model only if that one scores better on rounds held back for testing, and keeps the old one as `models/geoguessr-previous.pt`. It uses a quarter of the CPU so the computer stays cool, so let it run for a while.

## Tips

- **Moved or resized the browser?** Add `--calibrate` to show the bot where the game is again, e.g. `geoguessr-ai play --calibrate`.
- **Check it can see the game** with `--dry-run`: it plays one round without clicking.
- Keep Street View's compass (on the right, beside the map) on screen: the bot turns with it.
- Faster but weaker: `--no-text` (don't read signs), `--no-walk`, `--no-look-up`.
- Every command lists its options with `--help`, e.g. `geoguessr-ai play --help`.

## How it works

It looks north, east, south and west, walks on when unsure, reads signs and looks for the sun. A frozen CLIP encoder plus a classifier you train ranks regions of the world. Clues reweigh those regions: scripts, languages, road numbers, towns, phone numbers and the sun. Then it drops the pin on the map.

| Test | Mean score (of 5,000) |
| --- | --- |
| OSV-5M test photos | 2,496 |
| Held-out game rounds, images only / with clues | 3,414 / 3,502 |
| Latest 425 live rounds | 3,450 |

More photos score higher: `train --shards 0 1 2 3` uses four times as many as `--shards 0`. On the test photos, one shard scored 2,238 and four scored 2,496.

## Credits

Data: OpenStreetView-5M (CC-BY-SA 4.0), GeoNames (CC BY 4.0), Natural Earth, `global-land-mask`. Encoder: OpenAI CLIP. Text reading: RapidOCR with PaddleOCR models (Apache 2.0).

For fun and learning. Only play it against friends who know it's a bot, never in ranked or public games.
