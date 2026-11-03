# Contributing

Thanks for helping. Bug reports, results from computers unlike ours, and pull requests are all welcome. Issues marked [good first issue](https://github.com/andrey-rublev/geoguessr-AI/labels/good%20first%20issue) are a good place to start, and [Discussions](https://github.com/andrey-rublev/geoguessr-AI/discussions) is the place for questions and ideas.

## Two rules

- **Pixels in, mouse and keys out.** The bot only looks at the screen and moves the mouse. It never reads the page's code, its network traffic or the game's answers before guessing. Changes that break this won't be merged.
- **Friends only.** It's for games with people who know it's a bot. Nothing that helps it pass as a person, get past a person check, or play ranked or public games.

## Set up

On Windows 10 or 11, with [uv](https://docs.astral.sh/uv/):

```powershell
git clone https://github.com/andrey-rublev/geoguessr-AI
cd geoguessr-AI
uv sync --extra dev --extra browser
uv run geoguessr-ai fetch-model
```

Or with plain Python 3.10 or newer, as in the [README](README.md#from-a-terminal). Most of the code and all of the tests run without the model.

## Before a pull request

```powershell
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
```

GitHub runs the same checks on Windows for every pull request.

- Keep a pull request to one change, and say in it what you saw before and after.
- Add a test for a fix, so it stays fixed. The tests use saved screenshots and small made-up data, never a live game.
- Write like the code around it: plain words, and comments that say why rather than what.
- For a change to how it guesses, give the mean score on held-out rounds before and after: `geoguessr-ai evaluate --rounds`, or `python scripts/check_clues.py` for clues. A change that only helps on the rounds it was tuned on doesn't count.

## Long jobs and heat

Training and embedding photos can run for hours. They pace themselves to keep a laptop cool, as running flat out has crashed one: a quarter of the CPU's threads, resting as long as they work, and pausing above 85 °C. Keep that for anything long-running you add, with the `Pacer` in `src/geoguessr_ai/model/train.py`.

`runs/`, `models/` and `data/` hold your rounds, models and photos. They are ignored by git: don't commit them.

## Releases

A tag like `v1.2.0` builds `geoguessr-AI-Setup.exe`, installs it on a clean GitHub computer, checks that what it installed runs there, and publishes the release only if it does. The model is released separately: see `scripts/pack_model.py`.
