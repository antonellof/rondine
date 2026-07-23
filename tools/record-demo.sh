#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_cast="${TMPDIR:-/tmp}/rondine-demo-raw.cast"
cast="${TMPDIR:-/tmp}/rondine-demo.cast"

cd "$root"
asciinema record \
  --quiet \
  --headless \
  --overwrite \
  --return \
  --output-format asciicast-v2 \
  --window-size 100x34 \
  --idle-time-limit 1.5 \
  --command "expect tools/record-demo.exp" \
  "$raw_cast"

python - "$raw_cast" "$cast" <<'PY'
import json
import sys

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    lines = handle.readlines()

events = [json.loads(line) for line in lines[1:]]
offset = events[0][0] if events else 0.0
for event in events:
    event[0] = max(0.0, event[0] - offset)
    if event[0] < 0.32:
        event[0] = 0.0

with open(destination, "w", encoding="utf-8") as handle:
    handle.write(lines[0])
    for event in events:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
PY

agg \
  --theme monokai \
  --font-size 15 \
  --line-height 1.25 \
  --fps-cap 20 \
  --idle-time-limit 1.5 \
  --last-frame-duration 2 \
  --speed 1.15 \
  "$cast" \
  assets/rondine-demo.gif
