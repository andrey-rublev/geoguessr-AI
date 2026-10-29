# geoguessr-AI

An AI that plays [OpenGuessr](https://openguessr.com) by looking at your screen and moving your mouse. It never reads the page's code, and it learns from the rounds it plays.

## Install

In PowerShell, in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
geoguessr-ai train --shards 0
```

The last line builds the first model: it downloads about 5.4 GB of street photos, then trains. In each new terminal, run `.\.venv\Scripts\Activate.ps1` again.

## Use

1. Open the game in your browser: a Singleplayer game for `play`, a multiplayer room for `friends`.
2. Run the command and let go of the mouse. Press **F8** to stop.

The bot finds the game by itself, on any screen, and switches to its tab if another tab is showing. When the game has moved, for example after you dock your laptop, it finds it again. To check what it found, open `calibration_preview.png` (`calibration_preview_party.png` for a room).

### Commands

**`geoguessr-ai play`** plays a Singleplayer game on its own: 5 rounds, or `--rounds 10`. After each round it reads the answer and says how far off its pin was and whether it got the country right, like `our pin was 111 km away, guessed NG`. When it stops, it gives the average distance and how often the country was right. Each round is saved to `runs/` with its answer, for `train`.

**`geoguessr-ai friends`** plays in a multiplayer room until you stop it. It plays each round the host starts, keeps to the round's timer, and waits for the host to press Continue. It needs a browser of its own, with a friend hosting the room on their device: `friends --join EW2-78U` opens one and joins the room with that code, or `friends --browser` opens one for you to join in yourself. `--hidden` keeps that browser off screen. OpenGuessr's bot check may stop the bot's own browser, though; if it does, use a screen of its own (below). Its rounds are saved like `play`'s.

**`geoguessr-ai train`** learns from the street photos and the rounds in `runs/`. It builds a new model and keeps it only if it scores better than the current one on rounds held out from training, so training never makes it worse. `--shards 1 2` first downloads and adds more photos, about 2.5 GB a shard. Once you have a few thousand rounds, `--backbone geolocal/StreetCLIP` tries StreetCLIP, a sharper image model; it needs a GPU to be quick, and on an Intel one it runs through OpenVINO.

**`geoguessr-ai train-details`** tries to learn the small things players use (bollards, poles, road lines, kerbs) from close-ups of its rounds, kept only if they add points. On 1,800 rounds they didn't yet. The first run takes a while, since it looks at every round's close-ups.

**`geoguessr-ai calibrate`** finds the game on screen and saves where it is. `play` and `friends` do this by themselves, so you need it only if they can't find the game: `calibrate --manual` lets you show it each part yourself by hovering over it.

**`geoguessr-ai evaluate --rounds`** scores the model on the held-out rounds, and **`geoguessr-ai predict photo.jpg`** guesses where photos were taken.

Training runs cool, since running flat out has crashed a laptop: on a quarter of the CPU's threads, resting as long as it works, and pausing above 85 °C. `--threads`, `--rest` and `--max-temp` change that.

### Options for `play` and `friends`

| Option | What it does |
| --- | --- |
| `--browser` | Plays in a Chrome window of the bot's own, so your mouse and screen stay free (`pip install playwright` first). Start the game in that window |
| `--rounds N` | How many rounds to play (`friends` plays as many as the host starts) |
| `--dry-run` | Plays one round without clicking, to check the bot can see the game |
| `--no-answers` | Doesn't read the answer after each round: quicker, but no distances and nothing for `train` |
| `--no-save` | Saves nothing to `runs/` |
| `--no-text`, `--no-walk`, `--no-look-up` | Quicker but less accurate: don't read signs, don't walk on to look again when unsure, don't look up at the sun |

Add `--help` to any command to see all its options.

### Play it on the same laptop

To play against the bot while you use the laptop, give it a screen and mouse of its own: a virtual screen with a normal Chrome, in WSL's Ubuntu. Set it up once (about 1.5 GB; it asks for your Ubuntu password):

```powershell
wsl -d Ubuntu -e bash scripts/wsl_setup.sh
```

Then host a room in your own browser and run, with its code:

```powershell
wsl -d Ubuntu -e bash scripts/wsl_screen.sh friends --join EW2-78U
```

The bot joins the room on its own screen and plays each round you start. To watch it, or to do something there yourself, such as passing a check or starting a Singleplayer game for `play`, open `http://localhost:6080/vnc.html?autoconnect=1&resize=scale`. Ctrl+C stops it. With an Intel GPU, the image model runs on it from WSL too.

## How it works

It looks around, reads signs and checks the sun. StreetCLIP, an image model trained to geolocate street photos, and a classifier trained on its own rounds rank regions of the world. The clues and the photos that look most alike reweigh them, and those photos place the guess within the chosen region. On 1,052 held-out rounds it averages about 3,770 of OpenGuessr's 5,000 points a round, before the game's streak bonus; with plain CLIP it was 3,100.

Data: OpenStreetView-5M (CC-BY-SA 4.0), GeoNames (CC BY 4.0), Natural Earth, `global-land-mask`. Models: StreetCLIP (CC BY-NC 4.0, non-commercial use only), OpenAI CLIP, RapidOCR with PaddleOCR (Apache 2.0), run with OpenVINO (Apache 2.0).

For fun and learning. Only play it against friends who know it's a bot, never in ranked or public games.
