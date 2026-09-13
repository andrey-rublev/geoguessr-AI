# geoguessr-AI

**A self-trained geolocation AI that plays [OpenGuessr](https://openguessr.com) by watching your screen and controlling your mouse.**

It looks around the Street View panorama, guesses where in the world it is with a model you train yourself, finds that spot on the in-game map, drops the pin, and moves on to the next round. It needs no browser extension or API key and never reads the page's code. It works only from screen pixels, like a human player.

---

## How it works

Each round has three stages:

**1. Look.** The bot takes a screenshot of the panorama, drags the view to rotate the camera, and repeats (4 views by default). Screen capture uses [`mss`](https://github.com/BoboTiG/python-mss) and the mouse is driven with [`pyautogui`](https://github.com/asweigart/pyautogui).

**2. Guess.** This is the model you train:
- Each view is split into square crops and encoded by a **frozen CLIP image encoder** (`openai/clip-vit-base-patch32` by default).
- A small **MLP head** classifies the embeddings into **geocells**: regions of the globe from k-means on the training locations, so dense areas get small cells.
- Training uses **haversine label smoothing**, so a near miss is penalised less than a guess on the wrong continent.
- All crops vote together. The final pin goes where the **expected GeoGuessr score is highest**, not simply on the likeliest cell. If the model is split between Paris and Brussels, a guess between them beats a coin flip.

**3. Act.** The bot hovers the minimap to expand it and takes a screenshot of it. It then works out the map's exact Web Mercator projection by **matching the water/land pattern against a global land mask**: a coarse search over every scale and position, then sub-pixel refinement. With that projection, any latitude/longitude converts to a screen pixel. The bot clicks there, presses Guess, then Continue.

The map step doesn't depend on window size, browser zoom, or display scaling. On the real OpenGuessr map it recovers the world width to the exact pixel (1024 CSS px, i.e. Leaflet zoom 2). In an end-to-end test the pin landed **within ~13 km** of its target.

## Requirements

- Python 3.10+ on Windows, macOS, or Linux. Developed on Windows 11.
- About 5 GB of disk per training shard: 2.5 GB of images, plus a one-time 2.9 GB label CSV.
- A GPU is optional. Everything runs on CPU; see [Speed](#speed).

## Setup

```bash
git clone <this repo> && cd geoguessr-AI
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

## 1. Train your model

Training data comes from [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m): 5 million geotagged street-level images, CC-BY-SA 4.0, split into 98 train shards of ~50k images each.

```bash
# Download the labels and the first image shard (~5.4 GB total)
geoguessr-ai download --shards 0

# Encode images with the frozen backbone (slow step, runs once; resumable per shard)
geoguessr-ai embed --shards 0 --limit 20000

# Train the geocell head (minutes)
geoguessr-ai train
```

`train` prints validation metrics each epoch: median error in km, mean GeoGuessr score, and accuracy at 25 / 200 / 750 / 2500 km. It keeps the best checkpoint at `models/geoguessr.pt`.

Try the model on any photo:

```bash
geoguessr-ai predict some_street.jpg
```

**Getting a better model**

- **More data.** Download and embed more shards (`--shards 0 1 2 3`). `train` picks up every embedding file in the folder.
- **A better backbone.** Use `--backbone geolocal/StreetCLIP`, a CLIP model pretrained for geolocation. It's much more accurate but ~10× slower to embed, so it's best with a GPU.
- **Your own images.** Put them in a folder with a `labels.csv`:

  ```csv
  filename,latitude,longitude
  shot_001.jpg,48.8584,2.2945
  ```

  ```bash
  geoguessr-ai embed --images my_photos --labels my_photos/labels.csv
  ```

  Embeddings from the same backbone mix freely, so you can train on OSV-5M and your own data together.

### Speed

Embedding is the bottleneck, and the model does most of the work: JPEG decoding and preprocessing each run at several hundred images/s. With the default backbone on an 8-core laptop CPU (Intel Core Ultra, no CUDA), steady-state speed is about **45–50 images/s**, so a 50k-image shard takes about 18 minutes. Finished shards are skipped when you re-run, so `embed` can be stopped and resumed.

Other ways to speed it up:
- **NVIDIA GPU:** install a CUDA build of PyTorch. It's picked up automatically.
- **Intel Arc / Core Ultra iGPU:** install PyTorch's XPU build and pass `--device xpu`.

## 2. Calibrate (once)

Open https://openguessr.com, start a Singleplayer game, and place the browser window where it will stay. Then run:

```bash
geoguessr-ai calibrate
```

It walks you through seven points. For each one, press Enter, then hover the spot within 3 seconds:

1. and 2. Opposite corners of the Street View area. Stay clear of the minimap and buttons.
3. The collapsed minimap.
4. and 5. Opposite corners of the expanded map (map tiles only).
6. The Guess button.
7. The Continue button on the result screen. You make one guess by hand for this step.

The result is saved to `layout.json`, plus `calibration_preview.png` so you can check the regions. Re-run it if you move or resize the window.

## 3. Play

Do a dry run first. It plays one round and never clicks:

```bash
geoguessr-ai play --dry-run
```

Then let it play:

```bash
geoguessr-ai play --rounds 10
```

You get 5 seconds to switch to the browser. **To stop:** press **F8**, or slam the mouse into any screen corner (pyautogui's fail-safe).

Each round is saved to `runs/<timestamp>/round_NN/`:
- the captured views
- the map screenshot with the pin marked
- `round.json` with the guess, the recovered projection, and the click position

`--no-debug` turns this off.

## Commands

| Command | What it does |
| --- | --- |
| `download` | Download OSV-5M labels and image shards |
| `embed` | Encode OSV-5M shards or your own folder into embeddings |
| `train` | Train the geocell head; saves the best checkpoint |
| `predict` | Guess the location of image files |
| `calibrate` | Record where the OpenGuessr UI is on screen |
| `play` | Play OpenGuessr |
| `locate-map` | Debug: find the world on a map screenshot (`--at LAT LON` prints that point's pixel) |

Run `geoguessr-ai <command> --help` for all options. `python -m geoguessr_ai` works too.

## Troubleshooting

- **"Couldn't read the map".** `map_region` probably includes non-map UI, or the map is zoomed far in over land. The bot zooms out once by itself. If it keeps failing, re-run `calibrate` and select only the map tiles. Check with `geoguessr-ai locate-map runs/.../map.png`.
- **Clicks land in the wrong place.** The window moved or the display scale changed; re-run `calibrate`.
- **The view doesn't rotate enough.** Adjust `drag_fraction` in `BotSettings` (`src/geoguessr_ai/game.py`).

## Project layout

```
src/geoguessr_ai/
  cli.py          command-line interface
  game.py         the bot loop (look -> guess -> read map -> pin -> continue)
  mapcal.py       map projection from pixels via land-mask matching
  calibrate.py    interactive UI calibration
  screen.py       DPI-aware screen capture
  controls.py     mouse control, F8 stop key, fail-safe
  config.py       layout.json
  geo.py          haversine, Web Mercator, GeoGuessr scoring
  model/
    backbone.py   frozen CLIP encoder
    geocells.py   k-means geocells, label smoothing, expected-score guessing
    head.py       MLP head + checkpoint
    data.py       OSV-5M download and streaming readers
    embed.py      embedding precompute
    train.py      training loop and metrics
    predictor.py  inference over multiple views
tests/            unit tests, including a synthetic-map end-to-end bot round
```

Run the tests with `pytest`.

## Fair play

This is a learning project about computer vision and screen automation. The bot only uses what's visible on screen. Please don't use it against real people in multiplayer or competitive modes.

## Credits

- Training data: **OpenStreetView-5M**, Astruc et al., CVPR 2024 (CC-BY-SA 4.0).
- Land mask: [`global-land-mask`](https://github.com/toddkarin/global-land-mask), based on NOAA GLOBE.
- Image encoder: OpenAI CLIP via Hugging Face Transformers.
- Geocell classification with haversine smoothing and expected-score guessing is inspired by PlaNet and PIGEON.
