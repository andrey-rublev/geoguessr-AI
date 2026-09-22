# geoguessr-AI

A self-trained geolocation AI that plays [OpenGuessr](https://openguessr.com) by watching your screen and controlling your mouse.

## How it works

1. **Look:** presses Street View's compass to face north, then turns 90° at a time to capture north, east, south and west, pressing again if a turn didn't happen. If the sky is clear it then tilts up and looks round for the sun, which a level view seldom shows. If the model is unsure after that (it expects under 1,500 points), the bot walks about 50 m on along the road and looks round again, as players do when a place gives nothing away.
2. **Read:** reads signs, and road names also as if looking down on the road, where they come out straight, and looks for the sun.
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
- **Checking the clues:** `python scripts/check_clues.py` scores the clues on your own rounds (see [What the clues are worth](#what-the-clues-are-worth)); rerun it after changing one. `python scripts/check_strengths.py` does the same for how hard the bot leans on what it knows.
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
- Reading signs adds about 3 to 5 seconds a round, and its models (about 100 MB) download the first time. `--no-text` skips it.
- Looking up for the sun adds up to 7 seconds to rounds with blue sky. `--no-look-up` skips it.
- In the first round the bot drags the view sideways once to measure Street View's camera, so it knows which way each pixel looks.
- Walking on when unsure adds about 13 seconds to those rounds. `--no-walk` skips it. `--look-down` also tilts the camera down at the road and saves those views, for future use: nothing reads them yet.
- Keep Street View's compass (right edge, above the zoom buttons) on screen. Without it the bot drags the view round instead, which doesn't cover every direction.
- `calibrate` also snapshots the Continue button (`layout-continue.png`). If an advert covers the button, the bot waits up to 20 seconds, then stops rather than click the advert. Layouts from before this need `calibrate` again.

**Stop:** press **F8**, or move the mouse into a screen corner. Re-run `calibrate` if you move the browser window.

## What it knows

| Clue | How it's used | Limits |
| --- | --- | --- |
| Street View coverage | Countries with no Google Street View (most of China, Central Asia, much of Africa and the Middle East) count for less; ones with only a little, less again | Leant on gently (`--coverage-strength 0.25`), since `learn` also learns where the game really sends you, which says the same thing from experience. |
| Script | Korean, Japanese, Chinese, Cyrillic, Greek, Thai, Devanagari (Arabic too after `pip install python-bidi`) | Hebrew, Georgian, Khmer, Lao and several Indian scripts can't be read. Chinese characters count for Japan as much as anywhere: the reader picks out kanji shop names and no kana at all. |
| Language | Telling letters (ł, ř, ğ, ã, ß) and street and shop words (rua, calle, straße, jalan, kinyozi) in about 40 languages, even read without accents (PRACA), cut off by the frame (DESCONT), shortened as on street signs (C., Tv., Rte., Jl.) or written onto the end of the name (Ivalontie, Eindstraat, Bárðardalsvegur) | English barely counts: it's on signs everywhere. |
| Road names | Street View writes them flat on the road. Each view is also turned into a picture of the road from above, where they come out straight: in 6 of 40 saved rounds it read road names the level views missed, like Mazatlán-Culiacán and Gaspar Rodríguez de Francia | Only once the camera is measured. Road numbers painted on the road (A167, C-13) still can't be read. |
| Road labels | How a country numbers and names its roads: `Co Rd 158`, `Range Rd 20`, `FM213`, `S Triple X Rd`, `14th St`, `Ulitsa`, `Marg`, `Cra. 71c`, `DW785`, `RP 51` | The compass points and numbered streets fit Canada as well as the US. |
| Towns | 46,000 town and city names (all over 15,000 people), in local scripts too (Москва, 東京): towns on a direction sign are usually near | Names that are everyday words (Victoria), short (Lima), makes (TOYOTA), road words (Terrace) or found in more than 4 countries don't count, nor ones after a street or shop word (Rua São João, Tv. Pinheiro). |
| Domains, phone numbers, brands | `.com.br`, `+48`, Brazil's `99983-2915`, PEMEX, M-PESA, O Boticário (even run together as OBOTICARIO) | About 100 regional chains and 15 local phone formats. |
| Prices | `R$ 9,99`, `25 zł`, `Ksh 100`, `Rp 15.000` | About 25 currencies, counting for less than road signs do: a tourist shop can price in euros. The euro and pound signs only narrow it to their regions. |
| Speeds, postcodes, road numbers | `35 MPH`, `SW1A 1AA`, `01310-100`, `BR-116`, `I-95`, `DN1`, `SH58` | What a country's own road authority and post office write, so these count for the most of any clue. |
| Sun | Its direction and height: low in the south means well north of the tropics, high in the north means south of them, overhead means the tropics | Only a whole, round sun glowing into open sky counts, not one cut off by the view's edge, behind a roof or in haze. On the 26 saved rounds where it was found it fitted the real latitude in 24, and it is worth more than any other clue. |
| Driving side | Printed with each guess, from the countries the model favours | A separate detector scored worse on test photos (2,467 vs 2,500), since the model already gets it right 88% of the time, so it isn't counted twice. |

Road line colours (yellow centre lines in the Americas, yellow edges in southern Africa and the Middle East) aren't used either: a detector for yellow paint found it on cars, walls and dry grass as often as on roads, in level views of 104 saved rounds and looking down in 23 live ones, and many rounds are on unmarked roads anyway. Bollards, poles and the Google car aren't looked for separately. The model only picks them up the way it learns anything else, from photos and your rounds.

### What the clues are worth

`python scripts/check_clues.py` replays your saved rounds: the model's own belief, then the same belief reweighed by the clues read from that round. Over 1,568 rounds played so far, a clue fitted the country the round was really in **414 times out of 474**, and on the 454 rounds the model was never trained on the clues are worth **+54 points a round** (give or take 19).

That precision is the number to watch, and it only tells the truth on rounds nobody has tuned against. Measured on rounds the guards were written for it looks like 98%; measured on the next thousand played it was 83%, and the fixes those misses paid for brought it to 87%. Each new batch finds its own misreadings.

They used to be worth less than half: each clue left the countries it pointed away from 30 to 50% of their weight, which the model simply outvoted. Given how seldom a clue is wrong, they now count two to three times as sharply. Rounds the model already had right lose 150 to 600 points to this; badly wrong ones gain thousands.

Sharper clues make a misread expensive, so the guards matter as much as the clues, and every one below was written for a miss that really happened:

- Cyrillic and Greek need a word of four letters and one letter that couldn't be a misread Latin one, and Greek doesn't count beside Cyrillic at all: `со` read out of Francisco sent Mexico to North Macedonia, `по` out of a Spanish `no` sent Spain to Russia, and `ΣΥΠΕΡΜΑΡΚΕΤ` was Russian СУПЕРМАРКЕТ.
- A dot with a space around it only joins a web address when the rest looks like one: `Ctra. de la Rabassa` is a road in Andorra, not a German `.de` domain.
- A word that is the tail of a longer one read nearby is the same sign cut off: `alle` beside `Calle Benito Juárez` is Spanish, not a Danish allé.
- Road numbers are looked for within a line, or a Turkish `811.SH` above a `247.Sk.` becomes the state highway SH 247. Seven Brazilian state codes are US ones too, so `MS-465` means Mississippi as readily as Mato Grosso do Sul.
- A name with a road word on either side is a road, not a town: Rua São João, Lucas Paddock Rd, Monroe Lake.

Towns are the least reliable clue and still worth keeping: they fitted only 52 times in 85, yet dropping them costs 24 points a round (give or take 15). Precision isn't the thing to maximise — a clue that is wrong a third of the time still pays when being right moves the guess thousands of kilometres.

One clue was tried and dropped: a road name shortened the English way (`Sage Rd`) pointing to the countries that sign in English. It fired on 45 rounds and was worth +1 point a round, because Street View labels big roads in Kazakhstan and Mongolia in English too, and those misses cancelled the wins.

### How hard to lean on what it knows

`python scripts/check_strengths.py` measures the three settings that decide that, choosing them on half the held-out rounds and pricing them on the other half. All three wanted to be weaker than the 1.0 they started at — dividing out the training set's bias at 0.6, the game's own prior at 0.5, Street View coverage at 0.25 — which is worth about **+60 points a round** (give or take 43) on the half that only checked. Coverage matters least now because the prior learned from played rounds says the same thing from experience. Rerun it after a lot more training: what suits the model moves as the model gets better.

On 83 earlier rounds, pins used to land at the minimap's edge whenever the guess was off screen (Japan, the US west coast, Australia). Placing them where the model meant raised the mean score from 2,084 to 2,286; on the latest 162 rounds the pin lands a median of 1 km from where the model meant. Coverage added 49 points (give or take 89, so not yet conclusive).

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

Data: OpenStreetView-5M (CC-BY-SA 4.0). Borders: Natural Earth (public domain). Land mask: `global-land-mask`. Encoder: OpenAI CLIP. Text reading: RapidOCR with PaddleOCR models (Apache 2.0). Town names: GeoNames (CC BY 4.0).

For fun and learning; please don't use it against real players.
