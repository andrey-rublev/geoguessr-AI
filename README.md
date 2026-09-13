# geoguessr-AI

A self-trained geolocation AI that plays [OpenGuessr](https://openguessr.com) by watching your screen and controlling your mouse.

## How it works

1. **Look:** screenshots the Street View and rotates the camera to capture 4 views.
2. **Guess:** a frozen CLIP image encoder plus a small classifier you train picks the most likely region of the world.
3. **Act:** finds the world on the minimap by matching coastlines, clicks the guessed spot, then presses Guess and Continue.

It only uses what's on screen. It never reads the page's code.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
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

Trained on one shard, the model averages about 2,270 points (out of 5,000) per location, with a median miss of about 1,100 km.

### Train further

- **More data:** `download` and `embed` more shards (e.g. `--shards 1 2 3`), then `train` again. It uses every embedded shard.
- **Better encoder:** `--backbone geolocal/StreetCLIP` is much more accurate but about 10× slower; best with a GPU.
- **Your own photos:** `embed --images <folder> --labels labels.csv`, where the CSV has `filename,latitude,longitude` columns.

Keep a new model only if `evaluate` shows a higher mean score.

## Play

```powershell
geoguessr-ai calibrate          # once: point at the view, minimap, and buttons
geoguessr-ai play --dry-run     # one round, no clicks
geoguessr-ai play --rounds 10
```

**Stop:** press **F8**, or move the mouse into a screen corner. Re-run `calibrate` if you move the browser window.

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
