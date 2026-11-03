"""Build the live demo: the app's own page playing back a real game the bot played, for anyone to
watch in a browser with nothing installed. GitHub Pages serves what this builds (see
.github/workflows/demo.yml):

    python scripts/build_demo.py docs/demo/game.log site/index.html

There's no app behind the page, so what the app would have told it comes from the game's log, read
the way the app reads the bot's lines as it plays. The buttons that would start the bot say where to
get it instead. Only the page's land and borders at the zoom it opens at come along: zoomed in, the
land's dots stay coarse.
"""

import argparse
import json
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from geoguessr_ai import desktop
from geoguessr_ai.botrun import Choices, Progress

GET_IT = "https://github.com/andrey-rublev/geoguessr-AI#get-it"
PAGE_URL = "https://andrey-rublev.github.io/geoguessr-AI/"
PICTURE = "social-preview.png"
NO_SCREEN = ("The bot's screen:", "Open it in your browser to watch")
"""Lines about watching the bot's own screen, which a replay hasn't got."""
START_S = 0.8
"""How long the page waits, as if someone had just pressed Start."""


def pause_before(line: str) -> float:
    """How long after the line before it the bot printed ``line``, about, at a pace that shows each
    round land on the chart without taking all day: a real round takes a minute or more."""
    if line.startswith("Round "):
        return 0.35
    if line.startswith("  guess "):
        return 0.3
    if line.startswith("  answer "):
        return 0.25
    return 0.1


def steps(log: list[str], choices: Choices) -> list[dict]:
    """What the app would say after each line, as only what changed: how the game is going, the
    rounds from the first that changed on, and the new lines."""
    progress = Progress(choices)
    over = threading.Event()
    run = SimpleNamespace(over=over, choices=choices, progress=progress)
    app = SimpleNamespace(setup={}, saved={})
    said, rounds, at = [], [], START_S

    def say(lines: list[str]) -> None:
        state = desktop.App.state(app, run, 1)
        now = state["progress"].pop("rounds")
        changed = next(
            (i for i, (was, is_) in enumerate(zip(rounds, now, strict=False)) if was != is_),
            min(len(rounds), len(now)),
        )
        said.append(
            {
                "at": round(at, 2),
                "running": state["running"],
                "progress": state["progress"],
                "from": changed,
                "tail": now[changed:],
                "lines": lines,
            }
        )
        rounds[:] = now

    for line in log:
        if line.startswith(NO_SCREEN):
            continue
        at += pause_before(line)
        progress.read(line)
        say([line])
    over.set()
    progress.finish(0, asked_to=False)
    at += 1.0
    say([])
    return said


def build(log_path: Path, out: Path) -> None:
    log = log_path.read_text(encoding="utf-8").splitlines()
    played = sum(line.startswith("Round ") for line in log)
    choices = Choices(mode="play", where="hidden", rounds=played)
    app = desktop.App(program=["demo"])
    app.setup |= {"model": True, "ubuntu": True, "wsl": True, "chrome": True}
    app.saved = asdict(choices)
    demo = {
        "idle": app.state(None, 0) | {"lines": [], "restart": True},
        "base": {"choices": asdict(choices), "setup": app.setup, "saved": app.saved},
        "steps": steps(log, choices),
    }
    page = desktop._page(desktop.land_rows()).decode()
    words = (
        f"A real game of {played} rounds the bot played on OpenGuessr, played back in the app. Drag"
        " and scroll the map, or click a round."
    )
    swaps = [
        ("<title>geoguessr-AI</title>", HEAD.format(words=words, url=PAGE_URL, picture=PICTURE)),
        ("</style>", STYLE + "</style>"),
        ('    <fieldset id="all-choices">', NOTE.format(words=words, get_it=GET_IT)),
        ("\n<script>\n", f"\n<script>\nconst DEMO = {_json(demo)};\n{REPLAY}</script>\n<script>\n"),
    ]
    for old, new in swaps:
        if page.count(old) != 1:
            raise SystemExit(f"desktop.html has changed: no one place for the demo at {old!r}")
        page = page.replace(old, new)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8", newline="\n")
    print(f"Wrote {out}, {out.stat().st_size / 1000:.0f} kB, {len(demo['steps'])} steps")


def _json(value) -> str:
    """``value`` as JSON that can't end the script it's in."""
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


HEAD = """<title>geoguessr-AI: a real game, played back</title>
<meta name="description" content="{words}">
<meta property="og:title" content="geoguessr-AI: a real game, played back">
<meta property="og:description" content="{words}">
<meta property="og:image" content="{url}{picture}">
<meta property="og:url" content="{url}">
<meta name="twitter:card" content="summary_large_image">"""

STYLE = """
/* The demo's note: what this page is, in the drawer, above what it would have you choose. */
.demo-note {
  margin: 0 0 18px; padding: 10px 12px 12px;
  background: var(--paper); border-left: 3px solid var(--plot); border-radius: 0 5px 5px 0;
}
.demo-note .section-name { color: var(--plot); margin-bottom: 4px; }
.demo-note p { margin: 0; font-size: 13px; line-height: 1.4; }
.demo-actions {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin-top: 10px;
}
.demo-get { margin-left: auto; font-weight: 600; color: var(--plot); text-underline-offset: 3px; }
.demo-get:hover { color: var(--ink); }
@media (max-width: 860px) {
  /* On a phone, the game comes first: the choices, and the button to start, are for the app. */
  #all-choices, .go { display: none; }
}
"""

NOTE = """    <aside class="demo-note" aria-labelledby="demo-name">
      <p class="section-name" id="demo-name">Played back</p>
      <p>{words}</p>
      <div class="demo-actions">
        <button type="button" class="quiet-button" id="demo-again">Watch again</button>
        <button type="button" class="quiet-button" id="demo-end">Skip to the end</button>
        <a class="demo-get" href="{get_it}">Get the app</a>
      </div>
    </aside>
    <fieldset id="all-choices">"""

REPLAY = """// Played back: there's no app behind this page, so it hears again what the app said
// while the bot played, at the pace given (?speed=2 doubles it). The bot can't start here.
(() => {
  const NOT_HERE =
    "This page only plays back a game the bot played. Get the app to play one of your own.";
  const speed = Number(new URLSearchParams(location.search).get("speed")) || 1;
  let listener = null, timers = [], game = 0;

  function messages() {
    let rounds = [];
    return DEMO.steps.map((step, i) => {
      rounds = rounds.slice(0, step.from).concat(step.tail);
      const progress = { ...step.progress, rounds };
      const { running, lines } = step;
      return [step.at, { ...DEMO.base, game, running, progress, lines, restart: i === 0 }];
    });
  }

  function say(message) {
    listener?.onmessage?.({ data: JSON.stringify(message) });
    document.getElementById("demo-end").hidden = !message.running;
  }

  function play() {
    timers.forEach(clearTimeout);
    game += 1;
    timers = messages().map(([at, message]) => setTimeout(() => say(message), (at * 1000) / speed));
  }

  function skip() {
    timers.forEach(clearTimeout);
    const all = messages();
    const [, last] = all[all.length - 1];
    say({ ...last, lines: all.flatMap(([, message]) => message.lines), restart: true });
  }

  window.EventSource = class {
    constructor() {
      listener = this;
      setTimeout(() => { say(DEMO.idle); play(); });
    }
    close() { timers.forEach(clearTimeout); }
  };
  window.fetch = async (url, options = {}) => {
    if (options.method !== "POST") return new Response("{}", { status: 404 });  // no finer land
    if (String(url).startsWith("/stop")) window.showProblem?.(NOT_HERE);  // the page says nothing
    return new Response(JSON.stringify({ error: NOT_HERE }), { status: 409 });
  };
  // Leaving the page stops nothing, so it needn't ask first.
  const listen = window.addEventListener.bind(window);
  window.addEventListener = (type, ...rest) =>
    type === "beforeunload" ? undefined : listen(type, ...rest);
  document.getElementById("demo-again").addEventListener("click", play);
  document.getElementById("demo-end").addEventListener("click", skip);
})();
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", type=Path, help="what the bot printed as it played a game")
    parser.add_argument("out", type=Path, help="where to write the page")
    args = parser.parse_args()
    build(args.log, args.out)
