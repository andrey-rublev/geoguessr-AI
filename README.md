# geoguessr-AI

A self-trained geolocation AI that plays [OpenGuessr](https://openguessr.com) by watching your screen and controlling your mouse.

## How it works

1. **Look:** presses Street View's compass to face north, then turns 90° at a time to capture north, east, south and west, pressing again if a turn didn't happen. If the model is unsure after that (it expects under 1,500 points), the bot walks about 50 m on along the road and looks round again, as players do when a place gives nothing away.
2. **Read:** reads signs and road names, and looks for the sun.
3. **Guess:** a frozen CLIP image encoder plus a small classifier you train ranks regions of the world, then GeoGuessr knowledge reweighs them (see [What it knows](#what-it-knows)).
4. **Place:** finds the world on the minimap by its coastlines, drags the map if the guess is off screen, zooms in, clicks, and checks the pin landed.
5. **Learn** (`learn` only): reads the real location off the result screen, then retrains on your rounds.

It only uses what's on screen. It never reads the page's code.

## Setup (Windows PowerShell)

First time only:

```powershell
cd C:\path\to\geoguessr-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Every time you open a new terminal, enter the venv:

```powershell
cd C:\path\to\geoguessr-AI
.\.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. Type `deactivate` to leave.

If activation fails with "running scripts is disabled", run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Or skip activating and call the venv's copy directly:

```powershell
.\.venv\Scripts\geoguessr-ai.exe play --dry-run
```

## Train

The training data is [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m): about 5M geotagged street photos. It comes in **shards**, zip files of about 50k photos each, and any single shard covers the whole world.

```powershell
geoguessr-ai download --shards 0     # labels + one shard (~5.4 GB)
geoguessr-ai embed --shards 0        # photos -> CLIP features (slow step, ~20 min per shard on CPU)
geoguessr-ai train                   # minutes; saves models/geoguessr.pt
```

Measure it on the separate test photos:

```powershell
geoguessr-ai download --split test --shards 4
geoguessr-ai embed --split test --shards 4
geoguessr-ai evaluate --embeddings data/embeddings/openai__clip-vit-base-patch32/osv5m-test-04.npz
```

Results on the test photos:

| Training data | Mean score (out of 5,000) | Median miss | Within 750 km |
| --- | --- | --- | --- |
| 1 shard (~50k photos) | 2,238 | 1,129 km | 42% |
| 4 shards (~200k photos) | 2,496 | 823 km | 49% |

### Train further

1. Add more shards. There are 98, each about 2.5 GB:

   ```powershell
   geoguessr-ai download --shards 4 5 6 7
   geoguessr-ai embed --shards 4 5 6 7
   ```

2. Retrain on everything embedded so far (test photos are skipped automatically):

   ```powershell
   geoguessr-ai train --out models/geoguessr-new.pt
   ```

3. Score it, and keep it only if the mean score is higher than before:

   ```powershell
   geoguessr-ai evaluate --embeddings data/embeddings/openai__clip-vit-base-patch32/osv5m-test-04.npz --model models/geoguessr-new.pt
   ```

4. If it's better, make it the default model:

   ```powershell
   Copy-Item models/geoguessr-new.pt models/geoguessr.pt
   ```

Other ways to improve it:

- **More shards:** every shard is a random sample of the whole world (each has about 190 of 222 countries, in the same mix), so more shards mostly add photos of rare countries.
- **Better encoder:** add `--backbone geolocal/StreetCLIP` to `embed`. It's much more accurate but a far bigger model: on a CPU expect many hours per shard, and slower rounds. You have to re-embed every shard with it.
- **CLIP's own idea of the country:** `python scripts/try_zero_shot_countries.py` scores the model with CLIP's zero-shot guess of the country ("a Street View photo taken in Kenya") mixed in, using embeddings you already have (a few minutes).
- **Reading signs at a larger size:** `python scripts/compare_ocr_sizes.py` reads your saved rounds at several sizes, showing what more it finds and how long each round takes.
- **Your own photos:** `geoguessr-ai embed --images <folder> --labels <folder>/labels.csv`, where the CSV has `filename,latitude,longitude` columns.

Once a shard is embedded, its zip in `data/osv5m/images/train/` can be deleted to free space.

## Play

```powershell
geoguessr-ai calibrate           # once: point at the view, minimap, and buttons
geoguessr-ai play --dry-run      # one round, no clicks
geoguessr-ai play --rounds 10    # just play, as fast as it can
geoguessr-ai learn --rounds 20   # play, read every answer, then retrain on your rounds
```

- `play` learns nothing. `learn` is slower: reading each answer zooms the result map out (5 to 20 seconds a round), and retraining takes a few minutes at the end.
- Each round prints its clues and likeliest countries, like `clue: Portuguese: farmácia, rua` and `guess -23.550, -46.630 (BR 81%, PT 6%, AR 3%; driving on the right 97%)`.
- Reading signs adds about 3 seconds a round, and its models (about 100 MB) download the first time. `--no-text` skips it.
- Walking on when unsure adds about 13 seconds to those rounds. `--no-walk` skips it. `--look-down` also tilts the camera down at the road and saves those views, for future use: nothing reads them yet.
- Keep Street View's compass (right edge, above the zoom buttons) on screen. Without it the bot drags the view round instead, which doesn't cover every direction.
- `calibrate` also snapshots the Continue button (`layout-continue.png`). If an advert covers the button, the bot waits up to 20 seconds, then stops rather than click the advert. Layouts from before this need `calibrate` again.

**Stop:** press **F8**, or move the mouse into a screen corner. Re-run `calibrate` if you move the browser window.

## What it knows

| Clue | How it's used | Limits |
| --- | --- | --- |
| Street View coverage | Countries with no Google Street View (most of China, Central Asia, much of Africa and the Middle East) are 20× less likely; ones with only a little, 2× | The country list is approximate. `learn` also learns where the game really sends you. `--coverage-strength 0` turns it off. |
| Script | Korean, Japanese, Chinese, Cyrillic, Greek, Thai, Devanagari (Arabic too after `pip install python-bidi`) | Hebrew, Georgian, Khmer, Lao and several Indian scripts can't be read. |
| Language | Telling letters (ł, ř, ğ, ã, ß) and street and shop words (rua, calle, straße, jalan, kinyozi) in about 40 languages, even read without accents (PRACA) or cut off by the frame (DESCONT); letters that set Ukrainian, Serbian or Kazakh apart from Russian | English barely counts: it's on signs everywhere. |
| Road names | Street View writes them along the road; they're read like any sign | Labels running steeply along the road often can't be read, and road numbers (A167, C-13) can't be read even looking down. |
| Domains, phone numbers, brands | `.com.br`, `+48`, Brazil's `99983-2915`, PEMEX, M-PESA, O Boticário (even run together as OBOTICARIO) | About 100 regional chains and 15 local phone formats. |
| Hemisphere | A low sun towards the south means north of the tropics, and the other way round | The sun seldom shows in a level view, and only a clear sun in open sky counts. |
| Driving side | Printed with each guess, from the countries the model favours | A separate detector scored worse on test photos (2,467 vs 2,500), since the model already gets it right 88% of the time, so it isn't counted twice. |

Road line colours (yellow centre lines in the Americas, yellow edges in southern Africa and the Middle East) aren't used either: a detector for yellow paint found it on cars, walls and dry grass as often as on roads, in level views of 104 saved rounds and looking down in 23 live ones, and many rounds are on unmarked roads anyway. Bollards, poles and the Google car aren't looked for separately. The model only picks them up the way it learns anything else, from photos and your rounds.

On 83 saved rounds with known answers, pins used to land at the minimap's edge whenever the guess was off screen (Japan, the US west coast, Australia). Placing them where the model meant raises the mean score from 2,084 to 2,286. Coverage added 49 points (give or take 89, so not yet conclusive). Sign clues changed one round, by +378, but those rounds didn't capture the road below the view. Of 15 later rounds, two had readable signs: the clues now catch a Kenyan barber's KINYOZI and a Brazilian shop's brand, phone number and cut-off DESCONT (Brazil 43% → 94%), but neither guess moved.

## Learn from your rounds

`learn` saves every round in `runs/` with the real location read off the result screen. These are real game images, unlike OSV-5M's phone and dashcam photos. After playing it embeds the new rounds, retrains with them in up to 15% of each batch, and learns where the game tends to send you. It switches to the retrained model only if that scores at least as well on the 30% of rounds held out for testing, and keeps the old one as `models/geoguessr-previous.pt`. `learn --no-train` only collects rounds.

The same steps by hand:

```powershell
geoguessr-ai embed --rounds
geoguessr-ai train --out models/geoguessr-rounds.pt
geoguessr-ai evaluate --rounds --model models/geoguessr-rounds.pt
```

Scores on held-out rounds are noisy: trust a difference bigger than about twice `mean_score_stderr`, which takes a few hundred rounds.

## Commands

| Command | What it does |
| --- | --- |
| `download` | Download dataset shards |
| `embed` | Turn photos into features |
| `train` | Train the model |
| `evaluate` | Score a model on held-out photos |
| `predict` | Guess where image files were taken |
| `calibrate` | Record where the game UI is on screen |
| `play` | Play OpenGuessr |
| `learn` | Play, then retrain on your rounds |

Run `geoguessr-ai <command> --help` for options.

## Credits

Data: OpenStreetView-5M (CC-BY-SA 4.0). Borders: Natural Earth (public domain). Land mask: `global-land-mask`. Encoder: OpenAI CLIP. Text reading: RapidOCR with PaddleOCR models (Apache 2.0).

For fun and learning; please don't use it against real players.
