# geoguessr-AI

A self-trained geolocation AI that plays [OpenGuessr](https://openguessr.com) by watching your screen and controlling your mouse. It only uses what's on screen, never the page's code.

## How it works

1. **Look:** faces north with Street View's compass and captures north, east, south and west, waiting for each view to finish loading. Under a clear sky it tilts up to find the sun. If the model is unsure (it expects under 1,750 points) it walks about 50 m on and looks again, and once more if still under 1,000.
2. **Read:** reads signs and road names, also from above, where text painted on the road comes out straight.
3. **Guess:** a frozen CLIP image encoder plus a small classifier you train ranks regions of the world, and clues from signs and the sun reweigh them.
4. **Place:** finds the world on the minimap by its coastlines, zooms in and clicks.
5. **Learn** (`learn` only): reads the real location off the result screen and retrains on your rounds.

## Results

| Test | Mean score (of 5,000) |
| --- | --- |
| OSV-5M test photos, trained on 4 shards | 2,496 |
| 765 held-out game rounds, images only | 3,337 |
| Same rounds, with clues | 3,419 |
| Latest 500 live rounds | 3,420 (country right 70%) |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Activate the venv in every new terminal. If activation says scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Train

The training data is [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m): about 5M geotagged street photos in shards of about 50k, each covering the whole world.

```powershell
geoguessr-ai download --shards 0                  # labels + one shard (~5.4 GB)
geoguessr-ai embed --shards 0                     # photos -> CLIP features (~20 min a shard on CPU)
geoguessr-ai train                                # saves models/geoguessr.pt
geoguessr-ai download --split test --shards 4     # test photos
geoguessr-ai embed --split test --shards 4
geoguessr-ai evaluate --embeddings data/embeddings/openai__clip-vit-base-patch32/osv5m-test-04.npz
```

More shards help: 1 scored 2,238 on the test photos, 4 scored 2,496. Train to a new file with `--out`, evaluate it with `--model`, and copy it over `models/geoguessr.pt` if it's better. A shard's zip can be deleted once it's embedded.

## Play

```powershell
geoguessr-ai calibrate            # once: point at the view, minimap and buttons
geoguessr-ai play --dry-run       # one round, no clicks
geoguessr-ai play --rounds 10     # just play
geoguessr-ai learn --rounds 500   # play, read every answer, then retrain
```

- **Stop:** press **F8**, or move the mouse into a screen corner. Re-run `calibrate` if you move the browser window.
- Keep Street View's compass on screen, or the bot has to drag the view round instead.
- Per round: reading signs takes 3 to 5 seconds (`--no-text`), looking up for the sun up to 7 (`--no-look-up`), each walk about 13 (`--no-walk`), and reading the answer in `learn` 5 to 20.
- If an advert covers the Continue button, the bot waits up to 20 seconds, then stops rather than click the advert.

## Learn from your rounds

`learn` saves every round in `runs/` with the real location, then retrains with your rounds making up 15% of each batch (30% and 50% scored worse). It switches to the new model only if it scores at least as well on the 30% of rounds held out for testing, and keeps the old one as `models/geoguessr-previous.pt`. The last 500 rounds added 53 points on held-out rounds.

Training uses a quarter of the CPU's threads and rests between stretches, since running flat out crashed a laptop at 97 to 100 °C. `--threads` and `--rest 0` change that. Training cut short carries on from its last finished epoch.

## What it knows

| Clue | Examples |
| --- | --- |
| Script | Korean, Japanese, Chinese, Cyrillic, Greek, Thai, Devanagari |
| Language | Telling letters (ł, ř, ğ, ã, ß) and street and shop words in about 40 languages (rua, calle, straße, jalan) |
| Road labels and numbers | `Co Rd 158`, `14th St`, `BR-116`, `I-95`, `US-90`, `SK-29`, `México 175D`, `RN-17` |
| Speeds and postcodes | `35 MPH`, `SW1A 1AA`, `01310-100` |
| Towns | 46,000 towns of over 15,000 people, but not everyday words, surnames or street names |
| Domains, phones, brands, prices | `.com.br`, `+48`, PEMEX, `R$ 9,99` |
| Sun | Its direction and height narrow down the latitude |
| Street View coverage | Countries with little or none count for less |

On rounds nobody tuned them against, clues fit the real country about 90% of the time, and they add **82 points a round** on held-out rounds. `python scripts/check_clues.py` and `python scripts/check_strengths.py` re-measure them on your rounds.

Tried and dropped: road line colours, a separate driving-side detector, CLIP's own guess of the country, votes from similar past rounds, and reading signs at full width.

## Commands

| Command | What it does |
| --- | --- |
| `download` | Download dataset shards |
| `embed` | Turn photos into features |
| `train` | Train the model |
| `evaluate` | Score a model on held-out photos or rounds |
| `predict` | Guess where image files were taken |
| `calibrate` | Record where the game UI is on screen |
| `play` | Play OpenGuessr |
| `learn` | Play, then retrain on your rounds |

Run `geoguessr-ai <command> --help` for options.

## Credits

Data: OpenStreetView-5M (CC-BY-SA 4.0). Borders: Natural Earth (public domain). Land mask: `global-land-mask`. Encoder: OpenAI CLIP. Text reading: RapidOCR with PaddleOCR models (Apache 2.0). Town names: GeoNames (CC BY 4.0).

For fun and learning; please don't use it against real players.
