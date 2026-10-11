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

1. Start the game in your browser, then run the command.
2. The first time, it asks you to show it where the game is. Follow the prompts in the terminal.
3. Click on the browser within 5 seconds. Press **F8** to stop.

**With friends:** the bot needs a browser of its own. A friend hosts the room on their own device, and you join it in the bot's browser before running `friends`. The bot plays each round the host starts, within the timer, and waits for the host to press Continue.

**Training:** `friends` and `play --learn` save each round's answer to `runs/`, and `train` learns from them.

**Tips:** run with `--calibrate` after moving the browser, or `--dry-run` to check it can see the game. Add `--help` to any command to see its options.

## How it works

It looks around, reads signs and checks the sun. A CLIP image encoder and a classifier trained on OpenStreetView-5M and its own rounds rank regions of the world, and the clues reweigh them. On recent live rounds it averages about 3,450 of 5,000 points.

Data: OpenStreetView-5M (CC-BY-SA 4.0), GeoNames (CC BY 4.0), Natural Earth, `global-land-mask`. Models: OpenAI CLIP, RapidOCR with PaddleOCR (Apache 2.0).

For fun and learning. Only play it against friends who know it's a bot, never in ranked or public games.
