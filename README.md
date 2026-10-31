# geoguessr-AI

An AI that plays [OpenGuessr](https://openguessr.com) by looking at the screen and moving a mouse. It never reads the page's code, and it learns from the rounds it plays. On 1,052 held-out rounds it averages about 3,770 of OpenGuessr's 5,000 points a round.

For fun and learning. Only play it against friends who know it's a bot, never in ranked or public games.

## Get it

On Windows 10 or 11, download **[geoguessr-AI-Setup.exe](https://github.com/andrey-rublev/geoguessr-AI/releases/latest/download/geoguessr-AI-Setup.exe)** and run it. It needs no administrator or Python. Windows may warn that it doesn't recognise the installer, as it isn't signed: choose **More info**, then **Run anyway**.

It fetches Python and its libraries as it installs (about 1 GB), then opens the app, which offers to download the model (about 2.2 GB).

## Play

Enter the room's code and press **Start the bot**, or choose **Solo game** and how many rounds.

**Where it plays:**

- **On a hidden screen** leaves your mouse and screen yours, so you can play against it on the same computer. It needs WSL, Windows' own Linux, which the app sets up (about 1.5 GB). **Bot's screen** shows that screen, where you can tick OpenGuessr's box if it asks.
- **In its own browser window** needs nothing more, but OpenGuessr's person check may stop it there.
- **On this screen** plays in your own browser, taking over your mouse until you press F8.

The map plots each round's guess and answer with a line between them. Scroll to zoom, drag to move, and click a round on it or in the list to see it. The switches under **How hard it plays** make it easier to beat: one view only (NMPZ) scores about 3,510 points a round against 3,770, and turning off signs brings it down further.

## From a terminal

To work on it, or use it without the app, in PowerShell, in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,browser]"
geoguessr-ai fetch-model
```

`geoguessr-ai app` opens the app, and `geoguessr-ai app --shortcut` puts an icon for it on your desktop and in the Start menu. Otherwise, open the game in your browser, run a command and let go of the mouse. **F8** stops it. The bot finds the game by itself, on any screen, and finds it again if it moves.

**`geoguessr-ai play`** plays a Singleplayer game on its own: 5 rounds, or `--rounds 10`. After each round it says how far off its pin was, like `our pin at 6.452, 3.391 was 111 km away, guessed NG`, and saves the round to `runs/` for `train`.

**`geoguessr-ai friends`** plays in a multiplayer room until you stop it: each round the host starts, keeping to the timer. `--join EW2-78U` joins that room in a browser of its own.

**`geoguessr-ai train`** learns from the street photos and the rounds in `runs/`, and keeps the new model only if it scores better on rounds held out from training. `--shards 1 2` first downloads and adds more photos, about 2.5 GB a shard. **`train-details`** tries to learn the small things players use from close-ups of its rounds, kept only if they add points; on 1,800 rounds they didn't yet.

**`geoguessr-ai calibrate`** finds the game on screen, if `play` and `friends` can't by themselves; `--manual` lets you show it each part by hovering over it. **`evaluate --rounds`** scores the model on the held-out rounds, and **`predict photo.jpg`** guesses where photos were taken.

Training runs cool, since running flat out has crashed a laptop: on a quarter of the CPU's threads, resting as long as it works, and pausing above 85 °C. `--threads`, `--rest` and `--max-temp` change that.

| Option for `play` and `friends` | What it does |
| --- | --- |
| `--hidden` | Plays on a screen of its own in WSL (set it up once with `wsl -d Ubuntu -e bash scripts/wsl_setup.sh`). Watch it at `http://localhost:6080/vnc.html?autoconnect=1&resize=scale` |
| `--browser` | Plays in a muted browser window of its own. Start the game in that window |
| `--rounds N` | How many rounds to play (`friends` plays as many as the host starts) |
| `--nmpz` | One view a round: for a game with NMPZ on, or to make it easier to beat |
| `--no-text`, `--no-walk`, `--no-look-up` | Quicker but less accurate: don't read signs, walk on when unsure, or look up at the sun |
| `--dry-run` | Plays one round without clicking, to check the bot can see the game |
| `--no-answers`, `--no-save` | Doesn't read each round's answer, or saves nothing to `runs/` |

Add `--help` to any command to see all its options.

## How it works

It looks around, reads signs and checks the sun. StreetCLIP, an image model trained to geolocate street photos, and a classifier trained on its own rounds rank regions of the world. The clues and the photos that look most alike reweigh them, and those photos place the guess within the chosen region.

Data: OpenStreetView-5M (CC-BY-SA 4.0), GeoNames (CC BY 4.0), Natural Earth, `global-land-mask`. Models: StreetCLIP (CC BY-NC 4.0, non-commercial use only), OpenAI CLIP, RapidOCR with PaddleOCR (Apache 2.0), run with OpenVINO (Apache 2.0).
