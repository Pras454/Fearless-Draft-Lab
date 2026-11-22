# Fearless Draft Lab

A League of Legends draft simulator and opponent-scouting platform for coaching
staff — built for the **Cloud9 x JetBrains: Sky's The Limit Hackathon 2026**
(Category 2: LoL Macro Scouting & Draft Management).

## What it does

1. **Fearless draft simulator** — interactive pick/ban board for Best-of-3 and
   Best-of-5 series with Hard Fearless, Soft Fearless, or standard draft rules.
   The champion pool visibly shrinks game by game as picks lock out, with the
   locked champions listed per game and a full per-game draft history.
2. **Live draft win-rate** — a logistic-regression model trained offline on
   1,732 pro games predicts the win probability of the current draft in real
   time, and suggests the best next pick (or the most valuable deny-ban) with
   the win-rate swing of each option. Held-out accuracy 52.6% vs a 50.6%
   majority baseline — a directional signal, honestly labelled as such.
3. **Opponent tendencies** — pick/ban rates per team per game number, player
   champion-pool depth, and patch/event context, computed from 2025–2026 pro
   series data (GRID feed provided for the hackathon).
4. **"How to Win" insights** — data-backed counter-strategy leads: key bans
   that swing Game 2, comfort picks worth banning, shallow champion pools to
   attack, and observed Fearless pool math. Every insight carries its sample
   size.

## Project layout

```
app/          Static web app (HTML/CSS/JS, no build step). Serve from repo root.
              model.json — trained win-rate model coefficients (baked in).
data/         aggregates.json + patch_event.json — aggregate statistics only (safe to commit).
pipeline/     Python data pipeline (streaming; reads the GRID starter repo)
              + train_model.py (win-rate model + patch/event summaries).
BUILD-NOTE.md Build log: what was used in each phase, incl. JetBrains tooling.
```

## Run the app

Serve this repository from its root folder (the app loads `data/aggregates.json`
relative to the repo root):

```bash
python3 -m http.server 8000
# open http://localhost:8000/app/
```

## Rebuild the data

The GRID starter repo must sit **beside** this project (never inside it):

```bash
git clone https://github.com/cloud9esports/stl-hackathon-2026-grid-starter ../stl-hackathon-2026-grid-starter
python3 -m venv ../.venvs/cloud9 && ../.venvs/cloud9/bin/pip install zstandard
../.venvs/cloud9/bin/python pipeline/extract_lol.py
```

The pipeline streams the `.jsonl.zst` feeds (never decompresses them to disk)
and writes only aggregate statistics to `data/aggregates.json`. The win-rate
model needs no extra dependencies (pure-Python logistic regression):

```bash
../.venvs/cloud9/bin/python pipeline/train_model.py
# writes app/model.json + data/patch_event.json
```

## Data license — hard rule

The GRID feed is proprietary and licensed **only for building this hackathon
project, during the hackathon**. Raw data, extracts, and derivatives
(`.zst`, `.jsonl`, CSV/Parquet extracts, databases) must never be committed —
see `.gitignore`. Aggregate results (rates, counts, win rates) are allowed and
are the only data-derived content in this repository.

## JetBrains tooling

See `BUILD-NOTE.md` for the tooling log. Phases 1–2 were built with a plain
editor; Junie CLI (headless, Junie Lite) is planned for Phase 3 once the API
key is available. No JetBrains evidence folders (`.idea/`, `.junie/`) are
committed unless genuinely produced by the corresponding tool.

## License

MIT — see `LICENSE`. Not endorsed by Riot Games.
