# geoguessr-AI

A self-trained geolocation AI that plays [OpenGuessr](https://openguessr.com) by watching your screen and controlling your mouse. It never reads the page's code.

## How it works

It looks north, east, south and west (walking on when unsure), reads signs and finds the sun. A frozen CLIP encoder plus a classifier you train ranks regions of the world, clues reweigh them (scripts, languages, road numbers, towns, phone numbers, the sun), and it drops the pin on the minimap. `learn` also reads the real answer and retrains on your rounds.

## Results

| Test | Mean score (of 5,000) |
| --- | --- |
| OSV-5M test photos | 2,496 |
| Held-out game rounds, images only / with clues | 3,337 / 3,419 |
| Latest 500 live rounds | 3,420 |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # every new terminal
pip install -e ".[dev]"
```

If scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Train

```powershell
geoguessr-ai download --shards 0   # OpenStreetView-5M labels + one shard (~5.4 GB)
geoguessr-ai embed --shards 0      # photos -> CLIP features (~20 min a shard on CPU)
geoguessr-ai train                 # saves models/geoguessr.pt
```

More shards score higher: 1 shard scored 2,238 on the test photos, 4 shards 2,496.

## Play

```powershell
geoguessr-ai calibrate            # once: point at the view, minimap and buttons
geoguessr-ai play --rounds 10     # just play
geoguessr-ai learn --rounds 500   # play, read every answer, then retrain
```

- **Stop:** press **F8** or move the mouse into a screen corner.
- Keep Street View's compass on screen, and re-run `calibrate` if you move the browser window.
- `learn` switches to the retrained model only if it scores better on held-out rounds. Training uses a quarter of the CPU to stay cool (`--threads`, `--rest`).

Run `geoguessr-ai <command> --help` for options.

## Credits

Data: OpenStreetView-5M (CC-BY-SA 4.0), GeoNames (CC BY 4.0), Natural Earth, `global-land-mask`. Encoder: OpenAI CLIP. Text reading: RapidOCR with PaddleOCR models (Apache 2.0).

For fun and learning; please don't use it against real players.
