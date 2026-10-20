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

| Command | What it does |
| --- | --- |
| `geoguessr-ai play` | Plays a Singleplayer game on its own (5 rounds, or `--rounds 10`) |
| `geoguessr-ai friends` | Plays in a multiplayer room until you stop it |
| `geoguessr-ai train` | Learns from the rounds it played, and keeps the new model only if it scores better |
| `geoguessr-ai train-details` | Tries to learn the small things players use (bollards, poles, road lines, kerbs) from close-ups of its rounds, kept only if they add points. On 1,800 rounds they didn't yet |

1. Open the game in your browser: a Singleplayer game for `play`, a multiplayer room for `friends`.
2. Run the command and let go of the mouse. Press **F8** to stop.

Add `--browser` to play in a Chrome window of the bot's own instead, so your mouse and screen stay free (`pip install playwright` first). Start the game in that window.

The bot finds the game by itself, on any screen, and switches to its tab if another tab is showing. When the game has moved, for example after you dock your laptop, it finds it again. To check what it found, open `calibration_preview.png`.

**With friends:** the bot needs a browser of its own: `friends --browser` opens one. A friend hosts the room on their own device, and you join it in the bot's browser. The bot plays each round the host starts, keeps to the timer, and waits for the host to press Continue.

**Training:** `friends` and `play --learn` save each round's answer to `runs/`, and `train` learns from them. Once you have a few thousand rounds, `train --backbone geolocal/StreetCLIP` switches to StreetCLIP, a sharper image model, if it scores better. It needs a GPU to be quick: on an Intel one it runs through OpenVINO.

**Tips:**
- `--dry-run` checks that the bot can see the game, without clicking.
- If it can't find the game, `geoguessr-ai calibrate --manual` lets you show it each part yourself.
- Add `--help` to any command to see its options.

## How it works

It looks around, reads signs and checks the sun. StreetCLIP, an image model trained to geolocate street photos, and a classifier trained on its own rounds rank regions of the world, and the clues reweigh them. On 1,052 held-out rounds it averages about 3,700 of OpenGuessr's 5,000 points a round, before the game's streak bonus; with plain CLIP it was 3,100.

Data: OpenStreetView-5M (CC-BY-SA 4.0), GeoNames (CC BY 4.0), Natural Earth, `global-land-mask`. Models: StreetCLIP (CC BY-NC 4.0, non-commercial use only), OpenAI CLIP, RapidOCR with PaddleOCR (Apache 2.0), run with OpenVINO (Apache 2.0).

For fun and learning. Only play it against friends who know it's a bot, never in ranked or public games.
