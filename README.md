# geoguessr-AI

A self-trained geolocation AI that plays [OpenGuessr](https://openguessr.com) by watching your screen and controlling your mouse.

## How it works

1. **Look:** screenshots the Street View and rotates the camera to capture 4 views.
2. **Guess:** a frozen CLIP image encoder plus a small classifier you train picks the most likely region of the world.
3. **Act:** finds the world on the minimap by matching coastlines, clicks the guessed spot, and presses Guess.
4. **Learn:** reads the real location off the result screen, saves the round, and presses Continue.

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

- **Better encoder:** add `--backbone geolocal/StreetCLIP` to `embed`. It's much more accurate but about 10× slower, and you have to re-embed every shard with it.
- **Your own photos:** `geoguessr-ai embed --images <folder> --labels <folder>/labels.csv`, where the CSV has `filename,latitude,longitude` columns.

Once a shard is embedded, its zip in `data/osv5m/images/train/` can be deleted to free space.

## Play

```powershell
geoguessr-ai calibrate          # once: point at the view, minimap, and buttons
geoguessr-ai play --dry-run     # one round, no clicks
geoguessr-ai play --rounds 10
```

**Stop:** press **F8**, or move the mouse into a screen corner. Re-run `calibrate` if you move the browser window.

## Learn from your rounds

Every round is saved in `runs/` with the real location, read off the result screen. These are real game images, unlike OSV-5M's phone and dashcam photos, so they can make the model better at OpenGuessr:

```powershell
geoguessr-ai embed --rounds                          # 30% of rounds are held out for testing
geoguessr-ai train --out models/geoguessr-rounds.pt  # mixes your rounds into training
geoguessr-ai evaluate --rounds --model models/geoguessr.pt
geoguessr-ai evaluate --rounds --model models/geoguessr-rounds.pt
```

- Training takes up to 15% of each batch from your rounds (`--real-fraction`) and learns where the game tends to send you (`--game-prior-strength`, 0 = off).
- Keep the new model only if its mean score beats the old one by more than about twice `mean_score_stderr`. That takes a few hundred rounds.
- Reading the answer zooms the result map out and adds about 5 to 20 seconds per round (closer guesses take longer). `play --no-answers` skips it.

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

Run `geoguessr-ai <command> --help` for options.

## Credits

Data: OpenStreetView-5M (CC-BY-SA 4.0). Land mask: `global-land-mask`. Encoder: OpenAI CLIP.

For fun and learning; please don't use it against real players.
