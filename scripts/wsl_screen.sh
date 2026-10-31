#!/bin/bash
# Play on a screen of the bot's own, inside WSL, so your mouse and screen stay yours.
#
# It starts a virtual screen with a normal Chrome on it, and runs geoguessr-ai there, with a
# mouse of its own. Watch that screen in a browser tab (http://localhost:6080), where you can
# also join a room or start a game for it yourself.
#
#   wsl -d Ubuntu -e bash scripts/wsl_screen.sh friends --join EW2-78U
#
# which `geoguessr-ai friends --join EW2-78U --hidden` runs from Windows. Set up once with
# scripts/wsl_setup.sh.
set -u

SCREEN=:99
SIZE=1920x1080x24
VNC_PORT=5999
VIEW_PORT=6080
VENV=${GEOGUESSR_VENV:-$HOME/geoguessr-venv}
PROFILE=$HOME/.geoguessr-chrome
REPO=$(cd "$(dirname "$0")/.." && pwd)
WINDOWS_HOME=${WINDOWS_HOME:-/mnt/c/Users/$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')}

pids=()
stop() {
    kill "${pids[@]}" 2>/dev/null
    wait 2>/dev/null
}
trap stop EXIT
trap 'exit 130' INT TERM HUP

if [ ! -x "$VENV/bin/geoguessr-ai" ]; then
    echo "No geoguessr-ai in $VENV: set it up first, with"
    echo "  wsl -d Ubuntu -e bash scripts/wsl_setup.sh"
    exit 1
fi

# The model files the Windows side already has: the image model, and it compiled for the GPU.
export HF_HOME=${HF_HOME:-$WINDOWS_HOME/.cache/huggingface}
mkdir -p "$HOME/.cache"
if [ ! -e "$HOME/.cache/geoguessr-ai" ] && [ -d "$WINDOWS_HOME/.cache/geoguessr-ai" ]; then
    ln -s "$WINDOWS_HOME/.cache/geoguessr-ai" "$HOME/.cache/geoguessr-ai"
fi

if timeout 1 xdpyinfo -display "$SCREEN" >/dev/null 2>&1; then
    if pgrep -f "$VENV/bin/geoguessr-ai" >/dev/null; then
        echo "The bot is already running on its screen, in another window."
        exit 1
    fi
    # Left behind by a run that was cut off.
    pkill -f "Xvfb $SCREEN"
    pkill -f "rfbport $VNC_PORT"
    pkill -f "websockify .*$VIEW_PORT"
    pkill -f "user-data-dir=$PROFILE"
    pkill openbox
    sleep 1
fi
Xvfb "$SCREEN" -screen 0 "$SIZE" -nolisten tcp >/dev/null 2>&1 &
pids+=($!)
# Everything here draws on that screen, not on WSL's own window on your desktop.
unset WAYLAND_DISPLAY
export XDG_SESSION_TYPE=x11 DISPLAY=$SCREEN
for _ in $(seq 50); do timeout 1 xdpyinfo >/dev/null 2>&1 && break; sleep 0.1; done
openbox >/dev/null 2>&1 &
pids+=($!)

# A plain Chrome, as you'd run it. Street View needs WebGL: on the GPU, through Windows' Direct3D,
# where WSL can reach one, or else on the CPU, more slowly.
graphics=(--enable-unsafe-swiftshader)
if [ -e /dev/dxg ] && [ -e /usr/lib/x86_64-linux-gnu/dri/d3d12_dri.so ]; then
    export GALLIUM_DRIVER=d3d12
    # Only Street View on it: there, the rest of the page came out with stray triangles.
    graphics+=(--ignore-gpu-blocklist --use-gl=angle --use-angle=gl --disable-gpu-rasterization)
    graphics+=(--disable-gpu-compositing)
fi
google-chrome --user-data-dir="$PROFILE" --no-first-run --no-default-browser-check \
    --start-maximized "${graphics[@]}" https://openguessr.com/multiplayer >/dev/null 2>&1 &
pids+=($!)
unset GALLIUM_DRIVER

# Only this computer can see the screen: both listen on localhost alone.
x11vnc -display "$SCREEN" -localhost -rfbport "$VNC_PORT" -forever -shared -nopw -quiet \
    >/dev/null 2>&1 &
pids+=($!)
websockify --web /usr/share/novnc "127.0.0.1:$VIEW_PORT" "127.0.0.1:$VNC_PORT" \
    >/dev/null 2>&1 &
pids+=($!)

echo "The bot's screen: http://localhost:$VIEW_PORT/vnc.html?autoconnect=1&resize=scale"
echo "Open it in your browser to watch, or to join the room or start a game there yourself."
echo "Ctrl+C here stops the bot."

[ $# -eq 0 ] && set -- friends
layout=()
case " $* " in
*" --layout "*) ;;
*) [ "$1" = play ] && layout=(--layout layout-wsl.json) ||
    layout=(--layout layout-party-wsl.json) ;;
esac
export PYTHONUNBUFFERED=1  # say each step as it happens, through wsl too
cd "$REPO" && "$VENV/bin/geoguessr-ai" "$@" --own-screen "${layout[@]}"
