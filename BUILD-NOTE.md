# Build Note — Fearless Draft Lab

Log of what was used in each phase, for the Devpost "Tooling Explanation" and
the Best AI-Powered Workflow Story bonus (`AI_SHOWCASE.md`, Phase 2).

## Phase 1 (21 Nov 2026) — data pipeline + aggregates + skeleton app

**Built:**
- `pipeline/extract_lol.py` — streaming extractor for the GRID LoL feeds.
  Meta CSVs are the primary source (games.csv for winners/teams/game numbers;
  rosters.csv for picks); feeds are used only for bans. Byte-prefilters each
  envelope line for draft/result/game-start events before JSON parsing, so the
  huge `seriesState` snapshots in most lines are never parsed. Handles game
  rollbacks (`series-!started-game`), series without a `team-won-series`
  event (winners derived per game), and string IDs.
- `pipeline/inspect.py` — schema inspector for one series feed + meta CSVs.
- `data/aggregates.json` — aggregate statistics only (see below).
- `app/` — static web app (HTML/CSS/JS, no build step): Fearless draft
  simulator (Bo3/Bo5, Hard/Soft/off), pool-shrink visualization, opponent
  tendency explorer, "How to Win" insight cards with sample sizes.

**Verified numbers (full run, 21 Nov 2026, ~95 s):**
- 744 series processed, 0 skipped; 1,737 finished games with winners;
  1,734 games with draft data (3 games have no draft events in their feeds).
- 45 teams, 282 players, 155 champions.
- Invariants check out exactly: 17,370 champion picks (10 × 1,737 games),
  8,685 wins (5 × 1,737), 3,474 team-games (2 × 1,737).
- Reproducibility: re-running the pipeline against the finished canonical
  clone (`../stl-hackathon-2026-grid-starter`, 2.7 GB, 745 LoL feeds)
  produces a byte-identical `aggregates.json` (cmp clean).
- All 568 multi-game series are Fearless-format (no champion picked twice in
  a series); locked champions per game: G2 10, G3 20, G4 30, G5 40 (exact
  averages, 30 Bo5 series).
- 8 key-ban shifts pass the thresholds (G1 pick rate ≥ 0.30 over ≥ 5 games,
  each arm n ≥ 3). Strongest: FURIA/Taliyah −0.417 (G2 WR 25% when banned,
  n=4, vs 67% when not, n=6); best sample: Movistar KOI/Azir −0.206
  (50%, n=10, vs 71%, n=17).

**Feed quirks found and handled (all verified against real data):**
- Draft events appear BEFORE their game's `series-started-game` event:
  the extractor buffers them and flushes on the started-game event, resolving
  the game UUID via games.csv. Unresolvable UUID = abandoned attempt (9
  rollbacks seen); its draft is discarded.
- Champion-select remakes appear as negation events
  (`team-!picked-character`, `team-!banned-character`): 461 handled — they
  cancel the matching pending pick/ban. Ignoring them inflated pick counts
  (up to 15 events per team per game).
- One game's draft attribution is swapped between the two teams (series
  2782482 game 2, verified against the in-game state): the extractor detects
  the swap by comparing feed picks to roster picks and flips the ban
  attribution. 0 games needed their bans dropped.
- One registered substitute (Jiwoo, series 2782482 game 2) is listed in the
  roster with a champion he never played: dropped from the pick counts using
  the verified feed. One missing roster champion (vital's Kalista, same
  game) is filled from the verified feed. Both corrections are counted in
  the aggregates' `meta` block.

**App smoke test (jsdom, 21 Nov 2026):** data badge shows the real counts;
155 champion buttons and 46 team options render; 8 insight cards render; a
full 20-action Game 1 draft works (5 picks + 5 bans per team); finishing the
game locks exactly the 10 picked champions in Game 2 under Hard Fearless
(pool: 155 total, 145 available, 10 locked); Cloud9 Kia scouting shows real
Game 1 tendencies (Ezreal/Corki 26%, Varus banned 43% over 53 games). Zero
JS errors.

**Aggregates produced (all with sample sizes):**
- Pick/ban rate per team per game number (Game 1–5).
- Champion pool depth per player (games, unique champions, top champions).
- Win rate per champion pick (overall + per game number).
- Fearless stats: observed locked champions per game number; key-ban shifts
  (Game 2 win-rate delta when a Game 1 comfort pick is banned in Game 2).
- Team / player / champion lookup lists from `meta/`.

**Tooling used in Phase 1:**
- Plain text editor + terminal (Python 3.12, `zstandard` 0.25.0 in a venv).
- No JetBrains tool was used in Phase 1, and no `.idea/` or `.junie/`
  evidence folder is committed — those are only added when genuinely produced
  by a JetBrains tool (planned: Junie CLI headless in Phase 2, pending the
  Junie API key).

**Data-license compliance:**
- GRID starter repo cloned BESIDE the project
  (`../stl-hackathon-2026-grid-starter`), never inside it.
- Pipeline writes only aggregate statistics into `data/`. No `.zst`, `.jsonl`,
  CSV/Parquet extract, or database file is written into the project.
- Pre-commit check (must print nothing):
  `git ls-files | grep -Ei '\.(zst|jsonl|parquet|sqlite|duckdb)$|stl-hackathon-2026-grid-starter'`

## Phase 2 (21 Nov 2026) — full app + win-rate model

**Built:**
- `pipeline/train_model.py` — fresh training script (reads ONLY the GRID meta
  CSVs: games.csv + rosters.csv + lol/series.csv; no feed files opened).
  Builds a per-game dataset (1,732 finished LoL games with complete 5v5
  active rosters; 5 dropped for incomplete rosters) and trains an
  L2-regularised logistic regression in pure Python (no numpy/sklearn —
  feature vectors are sparse: 10 picks per game). Features: per-champion
  presence difference (Blue +1 / Red −1) + one-hot game number + intercept.
  Lambda picked on an inner validation split (λ=0.3). Also writes the
  aggregate-only `data/patch_event.json` (22 patches, 20 events).
- `app/model.json` — baked-in model coefficients + metrics.
- `app/` — full app: live draft win-rate bar (updates on every pick),
  pick/ban suggestions ranked by win-rate swing (clickable — they perform the
  draft action), Fearless-locked champion list with per-game provenance,
  per-game draft history, opponent player pool-depth table, patch/event
  context filters, champion win-rate leaders/laggards, and three new insight
  families (comfort picks to ban, shallow pools to attack, Fearless pool
  math) next to the Phase 1 key-ban cards. Responsive CSS (≤640px layout).

**Verified numbers (21 Nov 2026):**
- Model: test accuracy **52.6%** (n=346) vs 50.6% majority baseline;
  train accuracy 63.7%, test log-loss 0.750. Per-game test accuracy:
  G1 57.6% (n=139), G2 50.9%, G3 51.4%, G4 40.9% (n=22), G5 33.3% (n=9) —
  late-game samples are small, so the app labels the prediction a
  directional signal, not a forecast. Intercept +0.32 (Blue-side edge).
- Strongest champion weights: Rek'Sai +0.86, Aurelion Sol +0.86, Zoe +0.73;
  weakest: Mel −0.90, Viego −0.84, Brand −0.67. Rare-pick champions carry
  large weights, so suggestions show the champion's season win rate + sample
  size and flag n<20.
- Insight counts (computed at runtime from aggregates.json): 8 key-ban
  shifts, 4 comfort picks, 5 shallow-pool players, Fearless locked-per-game
  table (G2 10.0 / G3 20.0 / G4 30.0 / G5 40.0 over 568/316/79/30 series).
- Win-rate extremes (min. 30 picks): Camille/Xayah 58% at the top, Smolder
  37% and Yorick 39% at the bottom.

**App smoke test (jsdom, 21 Nov 2026):** 30/30 checks pass, zero JS errors:
data badge, 155 champion buttons, 46 team options, 23 patch options,
21 event options, all insight groups, full 20-action Game 1 draft,
win-rate panel on both sides, Game 2 lock of exactly 10 champions
(pool 145 available), draft history, series score, C9 scouting + pool-depth
table, patch/event info, and suggestion buttons performing draft actions.

**Tooling used in Phase 2:**
- Plain text editor + terminal (Python 3.12, pure-Python training — no new
  dependencies; Node v24 + jsdom for the smoke test).
- No JetBrains tool was used in Phase 2 either (Junie API key still pending),
  and no `.idea/` or `.junie/` evidence folder is committed — those are only
  added when genuinely produced by a JetBrains tool (planned: Junie CLI
  headless in Phase 3).

**Data-license compliance (re-checked 21 Nov 2026):**
- No `.zst`, `.jsonl`, CSV/Parquet extract, or database file exists anywhere
  under the project (verified with `find`, not `git ls-files` — see note
  below). `data/` holds only `aggregates.json` and `patch_event.json`
  (aggregate statistics, explicitly allowed).
- NOTE: the Phase 2 git commit was NOT run by the build agent — the runtime
  flagged the workspace listing with an execution-risk check, which forbids
  running git commands inside the flagged directory. The parent agent should
  run the commit itself after reviewing: `git add -A && git commit -m
  "Phase 2: full app + win-rate model (test accuracy 52.6%)"`, preceded by
  the pre-commit check `git ls-files | grep -Ei
  '\.(zst|jsonl|parquet|sqlite|duckdb)$|stl-hackathon-2026-grid-starter'`
  (must print nothing).

## Phase 3 (planned) — Junie polish + AI_SHOWCASE.md + video
- Junie CLI (headless, Junie Lite) for app polish tasks, with session logs.
- `AI_SHOWCASE.md` for the $1,000 workflow-story bonus.
- Demo video (<3 min, public YouTube) + Devpost submission.
