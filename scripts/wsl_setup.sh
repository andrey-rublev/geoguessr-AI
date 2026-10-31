#!/bin/bash
# Set up WSL's Ubuntu once, for scripts/wsl_screen.sh: a virtual screen, a viewer for it in a
# browser tab, Google Chrome, and the bot in a Python environment of its own (~/geoguessr-venv).
#
#   wsl -d Ubuntu -e bash scripts/wsl_setup.sh
#
# It asks for your Ubuntu password, for apt. About 1.5 GB of downloads.
set -eu

REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${GEOGUESSR_VENV:-$HOME/geoguessr-venv}

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq xvfb x11-utils openbox x11vnc \
    novnc websockify fonts-noto-core fonts-noto-cjk fonts-liberation python3-venv python3-dev \
    python3-tk wget libgl1-mesa-dri

if ! command -v google-chrome >/dev/null; then
    deb=$(mktemp --suffix=.deb)
    wget -q -O "$deb" https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$deb"
    rm -f "$deb"
fi

# Intel's GPU driver, so the image model runs on an Intel Arc GPU rather than the CPU.
if [ -e /dev/dxg ] && ! dpkg -s intel-opencl-icd >/dev/null 2>&1; then
    dir=$(mktemp -d)
    igc=https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5
    neo=https://github.com/intel/compute-runtime/releases/download/26.35.39758.10
    for url in "$igc/intel-igc-core-2_2.41.5+22716_amd64.deb" \
        "$igc/intel-igc-opencl-2_2.41.5+22716_amd64.deb" \
        "$neo/intel-opencl-icd_26.35.39758.10-0_amd64.deb" \
        "$neo/libigdgmm12_22.10.0_amd64.deb" \
        "$neo/libze-intel-gpu1_26.35.39758.10-0_amd64.deb"; do
        wget -q -P "$dir" "$url"
    done
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ocl-icd-libopencl1 libze1 \
        "$dir"/*.deb || echo "No GPU driver: the image model will run on the CPU, more slowly."
    rm -rf "$dir"
fi

[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install -q -e "$REPO"
echo "Set up. Play with: wsl -d Ubuntu -e bash scripts/wsl_screen.sh friends"
