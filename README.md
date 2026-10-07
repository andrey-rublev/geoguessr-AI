# geoguessr-AI

A self-trained geolocation AI that plays [OpenGuessr](https://openguessr.com) by watching your screen and controlling your mouse. It never reads the page's code.

## How it works

It looks north, east, south and west (walking on when unsure), reads signs and finds the sun. A frozen CLIP encoder plus a classifier you train ranks regions of the world, clues reweigh them (scripts, languages, road numbers, towns, phone numbers, the sun), and it drops the pin on the minimap. Rounds it saves with their real answers become training data.

## Results

| Test | Mean score (of 5,000) |
| --- | --- |
| OSV-5M test photos | 2,496 |
| Held-out game rounds, images only / with clues | 3,414 / 3,502 |
| Latest 425 live rounds | 3,450 |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # every new terminal
pip install -e ".[dev]"
geoguessr-ai train --shards 0     # first model: downloads ~5.4 GB of photos, embeds and trains
```

If scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. More shards score higher (`--shards 0 1 2 3`): 1 shard scored 2,238 on the test photos, 4 shards 2,496.

## Commands

| Command | What it does |
| --- | --- |
| `geoguessr-ai play` | Play OpenGuessr on your own. |
| `geoguessr-ai friends` | Play your friends in a multiplayer room. |
| `geoguessr-ai train` | Learn from the photos and every round saved with its answer. |

The first time you run `play` or `friends`, it asks you to point at the view, map and buttons. Re-do that with `--calibrate` if you move the browser window. Run `geoguessr-ai <command> --help` for options.

## Play

```powershell
geoguessr-ai play --rounds 10           # start a Singleplayer game first
geoguessr-ai play --learn --rounds 500  # also read and save every answer, for train (slower)
```

- **Stop:** press **F8** or move the mouse into a screen corner.
- Keep Street View's compass on screen.

## Play with friends

```powershell
geoguessr-ai friends    # join your friends' room first (Multiplayer > Join)
```

- A friend hosts and presses Continue: the bot waits on the result and standings screens for them.
- It keeps track of the round's time all along, and skips walking, the sun and signs when it's short.
- The moment another player guesses (the game says so over Street View in duels), it stops looking round and guesses.
- The chat box is blanked out of what it sees. Rounds are saved with their answers, for `train`.

## Train

```powershell
geoguessr-ai train
```

It trains a new model on the photos and your saved rounds, pairs it with the last one (two trained apart guess better than either), and switches only if the pair scores better on held-out rounds. Training uses a quarter of the CPU to stay cool (`--threads`, `--rest`).

## Credits

Data: OpenStreetView-5M (CC-BY-SA 4.0), GeoNames (CC BY 4.0), Natural Earth, `global-land-mask`. Encoder: OpenAI CLIP. Text reading: RapidOCR with PaddleOCR models (Apache 2.0).

For fun and learning; only play it against friends who know it's a bot, never in ranked or public games.
