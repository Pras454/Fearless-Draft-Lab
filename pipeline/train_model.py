#!/usr/bin/env python3
"""Train the draft win-rate model for Fearless Draft Lab (Phase 2).

Reads ONLY the GRID meta CSVs (data files, never committed into this repo):
  meta/games.csv, meta/rosters.csv (title == 'lol', active players only),
  lol/series.csv (event names). No feed files are opened.

Model: L2-regularised logistic regression, pure Python (no numpy/sklearn
dependency — the feature vectors are sparse: 10 picks per game).
  Features: per champion, +1 if picked by Blue, -1 if picked by Red;
            one-hot game number (1-5); intercept (captures the Blue-side edge).
  Target: 1 if the Blue-side team won the game, else 0.
  Split: 80/20 train/test, fixed seed; lambda picked on an inner
         validation split of the training set.

Outputs:
  app/model.json        — champion weights + game weights + intercept + metrics
                          (baked into the app for live win-rate prediction)
  data/patch_event.json — aggregate-only patch/event summaries for the
                          dashboard filters (no raw data, no extracts)

Run:  python3 pipeline/train_model.py
"""
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT.parent / "stl-hackathon-2026-grid-starter"
AGG_PATH = ROOT / "data" / "aggregates.json"
MODEL_PATH = ROOT / "app" / "model.json"
PATCH_EVENT_PATH = ROOT / "data" / "patch_event.json"

SEED = 42
TEST_FRACTION = 0.20
LAMBDAS = [0.3, 1.0, 3.0, 10.0, 30.0]
EPOCHS = 1500
LEARNING_RATE = 0.5


def load_games():
    """Finished LoL games: game_id -> dict(winner, blue_team, game_number, patch, series_id)."""
    games = {}
    with open(GRID / "meta" / "games.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["title"] != "lol":
                continue
            if row["finished"] != "True" or not row["winner_team_id"]:
                continue
            blue_team = row["team1_id"] if row["team1_side"] == "blue" else row["team2_id"]
            games[row["game_id"]] = {
                "winner": row["winner_team_id"],
                "blue_team": blue_team,
                "game_number": int(row["game_number"]),
                "patch": row["patch"] or "unknown",
                "series_id": row["series_id"],
            }
    return games


def load_rosters(games):
    """Active picks per game: game_id -> {'blue': [champ_id x5], 'red': [champ_id x5]}."""
    picks = defaultdict(lambda: {"blue": [], "red": []})
    with open(GRID / "meta" / "rosters.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["title"] != "lol":
                continue
            if row["participation_status"] != "active":
                continue  # registered substitutes who never played
            gid = row["game_id"]
            if gid not in games:
                continue
            side = row["side"]
            if side not in ("blue", "red"):
                continue
            picks[gid][side].append(row["character_id"])
    return picks


def load_events():
    """series_id -> event name (from lol/series.csv)."""
    events = {}
    with open(GRID / "lol" / "series.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            events[row["series_id"]] = row["event"]
    return events


def build_dataset(games, picks, champ_index):
    """Return list of (sparse feature dict, label, game_number, patch, event)."""
    dataset = []
    dropped = 0
    for gid, g in games.items():
        p = picks.get(gid)
        if not p or len(p["blue"]) != 5 or len(p["red"]) != 5:
            dropped += 1
            continue
        feats = {}
        for cid in p["blue"]:
            feats[champ_index[cid]] = feats.get(champ_index[cid], 0) + 1
        for cid in p["red"]:
            feats[champ_index[cid]] = feats.get(champ_index[cid], 0) - 1
        # One-hot game number, offset after the champion features.
        feats[len(champ_index) + g["game_number"] - 1] = 1
        label = 1 if g["winner"] == g["blue_team"] else 0
        dataset.append((feats, label, g["game_number"], g["patch"], g["series_id"]))
    return dataset, dropped


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def train_logreg(train, n_features, lam, epochs=EPOCHS, lr=LEARNING_RATE):
    """Full-batch gradient descent, sparse updates. Returns (weights, intercept)."""
    w = [0.0] * n_features
    b = 0.0
    n = len(train)
    for _ in range(epochs):
        grad = [0.0] * n_features
        grad_b = 0.0
        for feats, label, *_ in train:
            z = b
            for idx, val in feats.items():
                z += w[idx] * val
            err = sigmoid(z) - label
            for idx, val in feats.items():
                grad[idx] += err * val
            grad_b += err
        for i in range(n_features):
            w[i] -= lr * (grad[i] / n + lam * w[i] / n)
        b -= lr * grad_b / n
    return w, b


def evaluate(model, data):
    w, b = model
    correct = 0
    logloss = 0.0
    per_game = defaultdict(lambda: [0, 0])
    for feats, label, game_number, *_ in data:
        z = b
        for idx, val in feats.items():
            z += w[idx] * val
        p = sigmoid(z)
        p = min(max(p, 1e-9), 1 - 1e-9)
        logloss += -(label * math.log(p) + (1 - label) * math.log(1 - p))
        pred = 1 if p >= 0.5 else 0
        ok = pred == label
        correct += ok
        per_game[game_number][0] += ok
        per_game[game_number][1] += 1
    n = len(data)
    return {
        "accuracy": correct / n,
        "logloss": logloss / n,
        "n": n,
        "per_game_accuracy": {
            str(k): round(v[0] / v[1], 3) for k, v in sorted(per_game.items())
        },
        "per_game_n": {str(k): v[1] for k, v in sorted(per_game.items())},
    }


def patch_event_summaries(dataset, events, n_champs):
    """Aggregate-only summaries for the dashboard filters."""
    patch_games = defaultdict(list)
    event_games = defaultdict(list)
    for feats, label, game_number, patch, series_id in dataset:
        patch_games[patch].append((feats, label))
        event_games[events.get(series_id, "unknown")].append((feats, label))

    def summarize(groups, min_games):
        out = {}
        for name, rows in groups.items():
            if len(rows) < min_games:
                continue
            picks = Counter()
            wins = Counter()
            blue_wins = sum(label for _, label in rows)
            for feats, label in rows:
                for idx, val in feats.items():
                    if idx >= n_champs:
                        continue  # game-number one-hot features, not champions
                    if val > 0:
                        picks[idx] += 1
                        wins[idx] += label
                    elif val < 0:
                        picks[idx] += 1
                        wins[idx] += 1 - label
            top_picks = [
                {"champion_index": idx, "picks": c, "winrate": round(wins[idx] / c, 3)}
                for idx, c in picks.most_common(10)
            ]
            out[name] = {
                "games": len(rows),
                "blue_winrate": round(blue_wins / len(rows), 3),
                "top_picks": top_picks,
            }
        return out

    return {
        "patches": summarize(patch_games, min_games=20),
        "events": summarize(event_games, min_games=10),
    }


def main():
    agg = json.loads(AGG_PATH.read_text(encoding="utf-8"))
    champions = agg["champions"]  # [{id, name}] — index order is the feature order
    champ_index = {c["id"]: i for i, c in enumerate(champions)}
    n_champs = len(champions)

    games = load_games()
    picks = load_rosters(games)
    events = load_events()
    dataset, dropped = build_dataset(games, picks, champ_index)
    print(f"games finished (LoL): {len(games)}")
    print(f"games with complete 5v5 active rosters: {len(dataset)} (dropped {dropped})")

    rng = random.Random(SEED)
    idx = list(range(len(dataset)))
    rng.shuffle(idx)
    n_test = int(len(dataset) * TEST_FRACTION)
    test_idx = set(idx[:n_test])
    train = [d for i, d in enumerate(dataset) if i not in test_idx]
    test = [d for i, d in enumerate(dataset) if i in test_idx]

    # Inner split of train for lambda selection.
    rng2 = random.Random(SEED + 1)
    tidx = list(range(len(train)))
    rng2.shuffle(tidx)
    n_val = int(len(train) * 0.15)
    val_set = set(tidx[:n_val])
    inner_train = [d for i, d in enumerate(train) if i not in val_set]
    val = [d for i, d in enumerate(train) if i in val_set]

    n_features = n_champs + 5
    best_lam, best_acc = None, -1.0
    for lam in LAMBDAS:
        model = train_logreg(inner_train, n_features, lam)
        acc = evaluate(model, val)["accuracy"]
        print(f"  lambda={lam:<5} val accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc, best_lam = acc, lam
    print(f"selected lambda={best_lam} (val accuracy {best_acc:.4f})")

    model = train_logreg(train, n_features, best_lam)
    w, b = model
    train_metrics = evaluate(model, train)
    test_metrics = evaluate(model, test)
    majority = max(
        sum(d[1] for d in test) / len(test),
        1 - sum(d[1] for d in test) / len(test),
    )
    print(f"train accuracy={train_metrics['accuracy']:.4f} logloss={train_metrics['logloss']:.4f}")
    print(f"test  accuracy={test_metrics['accuracy']:.4f} logloss={test_metrics['logloss']:.4f}")
    print(f"majority baseline (test)={majority:.4f}")
    print(f"per-game test accuracy: {test_metrics['per_game_accuracy']}")

    model_doc = {
        "trained_from": "GRID LoL meta (games.csv + rosters.csv, 2025-2026 seasons)",
        "n_games": len(dataset),
        "n_train": len(train),
        "n_test": len(test),
        "lambda": best_lam,
        "intercept": round(b, 4),
        "champion_weights": {
            champions[i]["id"]: round(w[i], 4) for i in range(n_champs)
        },
        "game_weights": {str(g): round(w[n_champs + g - 1], 4) for g in range(1, 6)},
        "metrics": {
            "train_accuracy": round(train_metrics["accuracy"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_logloss": round(test_metrics["logloss"], 4),
            "majority_baseline": round(majority, 4),
            "per_game_test_accuracy": test_metrics["per_game_accuracy"],
            "per_game_test_n": test_metrics["per_game_n"],
        },
    }
    MODEL_PATH.write_text(json.dumps(model_doc, indent=1), encoding="utf-8")
    print(f"wrote {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")

    summaries = patch_event_summaries(dataset, events, n_champs)
    # Translate champion_index back to champion ids for the app.
    for group in summaries.values():
        for entry in group.values():
            for tp in entry["top_picks"]:
                tp["champion_id"] = champions[tp.pop("champion_index")]["id"]
    summaries["meta"] = {
        "generated_by": "pipeline/train_model.py (Fearless Draft Lab)",
        "note": "Aggregate statistics only. No raw GRID data, extracts, or copies of meta/series.csv are included.",
        "n_games": len(dataset),
    }
    PATCH_EVENT_PATH.write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    print(f"wrote {PATCH_EVENT_PATH} ({PATCH_EVENT_PATH.stat().st_size} bytes)")
    print(f"patches: {len(summaries['patches'])}, events: {len(summaries['events'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
