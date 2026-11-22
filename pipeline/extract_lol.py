#!/usr/bin/env python3
"""Extract LoL draft aggregates from the GRID starter repo.

Design (verified against the data, 21 Nov 2026):
- PRIMARY source is meta/*.csv (title == "lol"):
    games.csv   -> per-game winner + teams + game_number
    rosters.csv -> per-game picks (player -> champion, active only)
- The .jsonl.zst feeds are used ONLY for bans, which meta does not contain.
  Each feed line is byte-scanned for draft/game-start events before any JSON
  parsing, so the huge seriesState snapshots in most lines are never parsed.
  All 20 draft events of a game appear BEFORE its series-started-game event,
  so they accumulate in a pending buffer and are flushed when the next
  series-started-game event arrives, whose game UUID resolves to a game
  number through games.csv (rolled-back games are simply absent from
  games.csv, so an unresolvable UUID discards the abandoned attempt's draft).
  Champion-select remakes appear as negation events (team-!picked-character
  / team-!banned-character) that cancel earlier picks/bans; the extractor
  applies them to the pending buffer. A few games have the feed's draft
  attribution swapped between the two teams (verified against in-game state);
  the extractor detects the swap by comparing feed picks to roster picks and
  flips the ban attribution, or drops the game's bans when neither
  orientation matches.
- Output is data/aggregates.json — AGGREGATE statistics only. No raw GRID
  data, no extracts, and no copies of meta/series.csv are written here.

Usage:
  extract_lol.py [--data-dir DIR] [--out FILE] [--limit N] [--probe]
  --probe: dump the first matched events of the first series file and exit.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import zstandard

# Byte patterns that mark lines worth parsing (draft / game lifecycle).
INTERESTING = (
    b"picked-character",
    b"banned-character",
    b"started-game",
)


def iter_lines_zst(path):
    """Yield raw (unparsed) lines from a .jsonl.zst file, streaming."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        yield line
            if buf.strip():
                yield buf


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def extract_series_bans(path, game_id_to_number, probe=False):
    """Stream one series feed. Returns {game_number: {bans: {team: [champ]},
    picks: {team: [champ]}}} plus a stats dict.

    Game association (verified on real feeds, 21 Nov 2026): all 20 draft
    events of a game appear BEFORE its series-started-game event. So draft
    events accumulate in a pending buffer and are flushed when the next
    series-started-game event arrives, whose game UUID resolves to a game
    number through games.csv. A UUID absent from games.csv means a
    rolled-back/abandoned game: the pending buffer is discarded (it was the
    abandoned attempt's draft) and the re-draft repopulates it.

    Champion-select remakes appear as NEGATION events (team-!picked-character
    / team-!banned-character) that cancel an earlier pick/ban. Negations
    remove the matching entry from the pending buffer (or, if already
    flushed, from the flushed game). Duplicate events are removed on flush.
    """
    per_game = defaultdict(lambda: {"bans": defaultdict(list), "picks": defaultdict(list)})
    pending = []  # (kind, team_id, champ_id) since the last started-game event
    last_flushed = None
    stats = {"negations": 0, "dupes_removed": 0, "rollbacks": 0}
    probe_shown = 0

    def _remove(target_list, kind, team_id, champ_id):
        # Remove the most recent matching entry; return True if removed.
        for i in range(len(target_list) - 1, -1, -1):
            if target_list[i] == (kind, team_id, champ_id):
                del target_list[i]
                return True
        return False

    for raw in iter_lines_zst(path):
        if not any(p in raw for p in INTERESTING):
            continue
        try:
            env = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for ev in env.get("events", []):
            etype = ev.get("type", "")
            actor = ev.get("actor") or {}
            target = ev.get("target") or {}
            actor_id = actor.get("id") if isinstance(actor, dict) else None
            target_id = target.get("id") if isinstance(target, dict) else None

            if probe and probe_shown < 4 and ("picked-character" in etype or "banned-character" in etype):
                print(json.dumps({k: v for k, v in ev.items() if k in ("type", "actor", "action", "target")}, default=str)[:600])
                probe_shown += 1

            if etype == "series-started-game":
                gid = None
                try:
                    gid = actor.get("stateDelta", {}).get("games", [{}])[0].get("id")
                except (AttributeError, IndexError, KeyError):
                    gid = None
                number = game_id_to_number.get(gid) if gid else None
                if number is None:
                    pending = []  # abandoned attempt; its draft is discarded
                    stats["rollbacks"] += 1
                else:
                    g = per_game[number]
                    seen = set()
                    for kind, team_id, champ_id in pending:
                        key = (kind, team_id, champ_id)
                        if key in seen:
                            stats["dupes_removed"] += 1
                            continue
                        seen.add(key)
                        g[kind][team_id].append(champ_id)
                    pending = []
                    last_flushed = number
                continue
            if "!started-game" in etype:
                # Rollback of a game start: drop the pending draft of the
                # abandoned attempt; the re-draft will repopulate the buffer.
                pending = []
                stats["rollbacks"] += 1
                continue
            if "!picked-character" in etype or "!banned-character" in etype:
                kind = "picks" if "picked" in etype else "bans"
                if not actor_id or not target_id:
                    continue
                stats["negations"] += 1
                if _remove(pending, kind, actor_id, target_id):
                    continue
                if last_flushed is not None:
                    lst = per_game[last_flushed][kind].get(actor_id, [])
                    if target_id in lst:
                        lst.remove(target_id)
                continue
            if "picked-character" in etype:
                if actor_id and target_id:
                    pending.append(("picks", actor_id, target_id))
            elif "banned-character" in etype:
                if actor_id and target_id:
                    pending.append(("bans", actor_id, target_id))
    # Leftover pending (a game whose started-game event never arrived) is
    # deliberately discarded: unattributable draft data must not pollute the
    # aggregates.
    return dict(per_game), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "stl-hackathon-2026-grid-starter"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "aggregates.json"))
    ap.add_argument("--limit", type=int, default=0, help="process only first N series (0 = all)")
    ap.add_argument("--probe", action="store_true", help="dump first matched events of first series file and exit")
    ap.add_argument("--diag", action="store_true",
                    help="print the key-ban threshold table (all pairs with G1 rate >= 0.3) and exit before writing")
    args = ap.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    meta_dir = os.path.join(data_dir, "meta")

    # ---------- meta (title == "lol" only; meta files also contain VALORANT) ----------
    teams = {}
    for row in load_csv(os.path.join(meta_dir, "teams.csv")):
        if row.get("title") != "lol":
            continue
        teams[row["team_id"]] = {"id": row["team_id"], "name": row["name"]}
    players = {}
    for row in load_csv(os.path.join(meta_dir, "players.csv")):
        if row.get("title") != "lol":
            continue
        players[row["player_id"]] = {"id": row["player_id"], "name": row["name"],
                                      "team_id": row.get("latest_team_id")}
    champions = {}
    for row in load_csv(os.path.join(meta_dir, "characters.csv")):
        if row.get("title") != "lol":
            continue
        champions[row["character_id"]] = {"id": row["character_id"], "name": row["name"]}
    print(f"meta: {len(teams)} LoL teams, {len(players)} players, {len(champions)} champions")

    # games.csv: series_id -> game_number -> {game_id, winner, teams}
    games_by_series = defaultdict(dict)
    game_id_to_number = {}   # game UUID -> (series_id, game_number)
    for row in load_csv(os.path.join(meta_dir, "games.csv")):
        if row.get("title") != "lol":
            continue
        if row.get("finished") != "True" or not row.get("winner_team_id"):
            continue  # unfinished / forfeited games have no winner
        sid = row["series_id"]
        gn = int(row["game_number"])
        games_by_series[sid][gn] = {
            "game_id": row["game_id"],
            "winner": row["winner_team_id"],
            "teams": [row["team1_id"], row["team2_id"]],
        }
        game_id_to_number[row["game_id"]] = (sid, gn)
    n_games = sum(len(g) for g in games_by_series.values())
    print(f"meta: {len(games_by_series)} series, {n_games} finished games with winners")

    # rosters.csv (active only): picks per (series, game, team) + player rows.
    # Player pools are built later, during the feed scan, so rows for a
    # 6th (substitute) player can be dropped when the feed shows which 5
    # champions were actually picked.
    roster_picks = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))  # sid -> gn -> team -> {champ}
    roster_rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # sid -> gn -> team -> [(player, champ)]
    for row in load_csv(os.path.join(meta_dir, "rosters.csv")):
        if row.get("title") != "lol" or row.get("participation_status") != "active":
            continue
        if not row.get("character_id"):
            continue  # a few rows lack a champion (e.g. a subbed player) — unusable
        sid, gn = row["series_id"], int(row["game_number"])
        if gn not in games_by_series.get(sid, {}):
            continue  # skip unfinished games
        roster_picks[sid][gn][row["team_id"]].add(row["character_id"])
        roster_rows[sid][gn][row["team_id"]].append((row["player_id"], row["character_id"]))
    print(f"meta: roster picks for {sum(len(g) for g in roster_picks.values())} games")

    # series.csv: series_id -> feed file
    series_files = {}
    for row in load_csv(os.path.join(data_dir, "lol", "series.csv")):
        series_files[row["series_id"]] = row["file"]
    print(f"series.csv: {len(series_files)} series files listed")

    series_ids = sorted(games_by_series)
    if args.limit:
        series_ids = series_ids[: args.limit]

    if args.probe:
        sid = series_ids[0]
        print(f"probe series {sid}: {series_files[sid]}")
        gid_map = {g["game_id"]: gn for gn, g in games_by_series[sid].items()}
        extract_series_bans(os.path.join(data_dir, series_files[sid]), gid_map, probe=True)
        return

    # ---------- feed scan: bans per game ----------
    # accumulators
    team_pb = defaultdict(lambda: defaultdict(lambda: {"games": 0, "picks": Counter(), "bans": Counter()}))
    champ_wr = defaultdict(lambda: {"picks": 0, "wins": 0, "by_game": defaultdict(lambda: {"picks": 0, "wins": 0})})
    team_games = Counter()
    player_pool = defaultdict(lambda: {"games": 0, "champions": Counter()})
    series_records = []   # for fearless analysis: {games: {n: {picks, bans, winner}}}
    fearless_like = 0
    bo5_series = 0

    processed = 0
    games_with_bans = 0
    swaps_detected = 0
    bans_dropped_games = 0
    subs_dropped = 0
    gaps_filled = 0
    total_negations = 0
    total_dupes = 0
    total_rollbacks = 0
    skipped = []

    def _sets_match(feed_set, roster_set):
        # Feed picks match roster picks when at least 4 champions agree.
        # Tolerance of one covers: a roster row with a missing champion
        # (series 2782482 game 2), and a registered substitute listed in the
        # roster (5 games). False positives are impossible in practice:
        # the two teams' pick sets in one game are disjoint, so a swapped
        # orientation can never reach 4 agreements by accident.
        return len(feed_set) >= 4 and len(feed_set & roster_set) >= 4

    for i, sid in enumerate(series_ids):
        rel = series_files.get(sid)
        if not rel:
            skipped.append((sid, "no file in series.csv"))
            continue
        path = os.path.join(data_dir, rel)
        if not os.path.exists(path):
            skipped.append((sid, "file missing"))
            continue
        gid_map = {g["game_id"]: gn for gn, g in games_by_series[sid].items()}
        try:
            feed, fstats = extract_series_bans(path, gid_map)
        except Exception as e:
            skipped.append((sid, f"parse error: {e}"))
            continue
        processed += 1
        total_negations += fstats["negations"]
        total_dupes += fstats["dupes_removed"]
        total_rollbacks += fstats["rollbacks"]

        games = games_by_series[sid]
        rec_games = {}
        for gn, g in sorted(games.items()):
            roster_sets = {t: set(champs) for t, champs in roster_picks.get(sid, {}).get(gn, {}).items()}
            if not roster_sets:
                continue
            feed_picks = {t: set(v) for t, v in feed.get(gn, {}).get("picks", {}).items()}
            feed_bans = {t: list(v) for t, v in feed.get(gn, {}).get("bans", {}).items()}
            game_teams = sorted(roster_sets)

            # Orientation check: the feed's draft attribution is swapped in
            # a few games (verified vs in-game state, series 2782482 game 2).
            # Rosters are the ground truth for who played what.
            orientation = "unknown"
            corrected_feed_picks = {}
            if len(game_teams) == 2 and feed_picks:
                t1, t2 = game_teams
                if (_sets_match(feed_picks.get(t1, set()), roster_sets[t1])
                        and _sets_match(feed_picks.get(t2, set()), roster_sets[t2])):
                    orientation = "normal"
                    corrected_feed_picks = feed_picks
                elif (_sets_match(feed_picks.get(t1, set()), roster_sets[t2])
                        and _sets_match(feed_picks.get(t2, set()), roster_sets[t1])):
                    orientation = "swapped"
                    swaps_detected += 1
                    corrected_feed_picks = {t1: feed_picks.get(t2, set()),
                                            t2: feed_picks.get(t1, set())}
            if orientation == "normal":
                bans_by_team = feed_bans
            elif orientation == "swapped":
                bans_by_team = {t: feed_bans.get(other, [])
                                for t, other in ((game_teams[0], game_teams[1]), (game_teams[1], game_teams[0]))}
            else:
                if feed_picks or feed_bans:
                    bans_dropped_games += 1
                bans_by_team = {}
            if bans_by_team:
                games_with_bans += 1

            # Reconcile picks. Rosters are the ground truth, with two narrow
            # corrections from the verified feed: (a) a 6th roster champion
            # (a registered substitute who never played) is dropped when the
            # feed shows the 5 actually picked; (b) a missing roster
            # champion is filled from the feed when the roster is a subset
            # of the feed's 5 (series 2782482 game 2: vital's Kalista).
            picks_by_team = {}
            for team_id, rset in roster_sets.items():
                fset = corrected_feed_picks.get(team_id, set())
                if len(rset) > 5 and len(fset) == 5 and len(rset & fset) == 5:
                    picks_by_team[team_id] = sorted(fset)
                    subs_dropped += 1
                elif len(rset) < 5 and len(fset) == 5 and rset <= fset:
                    picks_by_team[team_id] = sorted(fset)
                    gaps_filled += 1
                else:
                    picks_by_team[team_id] = sorted(rset)

            rec_games[gn] = {"picks": picks_by_team, "bans": bans_by_team, "winner": g["winner"]}
            for team_id, champs in picks_by_team.items():
                team_games[team_id] += 1
                ent = team_pb[team_id][gn]
                ent["games"] += 1
                ent["picks"].update(champs)
                for c in champs:
                    cw = champ_wr[c]
                    cw["picks"] += 1
                    cw["by_game"][gn]["picks"] += 1
                    if g["winner"] == team_id:
                        cw["wins"] += 1
                        cw["by_game"][gn]["wins"] += 1
            for team_id, bans in bans_by_team.items():
                team_pb[team_id][gn]["bans"].update(bans)
            # Player pools: drop a 6th (substitute) row when the feed shows
            # which 5 champions the team actually picked.
            for team_id, rows in roster_rows.get(sid, {}).get(gn, {}).items():
                ref = corrected_feed_picks.get(team_id, set())
                for pid, cid in rows:
                    if len(rows) > 5 and len(ref) == 5 and cid not in ref:
                        continue
                    pp = player_pool[pid]
                    pp["games"] += 1
                    pp["champions"][cid] += 1

        if rec_games:
            all_picks = [c for g in rec_games.values() for champs in g["picks"].values() for c in champs]
            if len(all_picks) == len(set(all_picks)) and len(rec_games) >= 2:
                fearless_like += 1
            if len(rec_games) == 5:
                bo5_series += 1
            series_records.append({"games": rec_games})

        if processed % 100 == 0:
            print(f"  ... {processed}/{len(series_ids)} series, {games_with_bans} games with bans")

    print(f"processed {processed} series, {games_with_bans} games with bans, "
          f"swaps corrected: {swaps_detected}, bans dropped (unverifiable): {bans_dropped_games}, "
          f"sub picks dropped: {subs_dropped}, gaps filled from feed: {gaps_filled}, "
          f"negations: {total_negations}, dupes removed: {total_dupes}, rollbacks: {total_rollbacks}, "
          f"skipped: {len(skipped)}")
    for sid, why in skipped[:10]:
        print(f"  skipped {sid}: {why}")

    # ---------- (d) key-ban shifts ----------
    # Universe: series where team T played G1 and G2 and did NOT pick C in G1.
    # Arm A: opponent banned C in G2. Arm B: opponent did not ban C in G2.
    # Requires: T's G1 pick rate of C >= 0.30 over >= 5 G1 games; each arm
    # n >= 3. (Thresholds set from the --diag table, 21 Nov 2026: rate 0.40
    # left only 1 pair; 0.30 yields 8, all with honest sample sizes shown in
    # the app. Pairs with a 1-game banned arm are noise and stay excluded.)
    def _arms(team_id, champ_id):
        n_banned = n_not = w_banned = w_not = 0
        for rec in series_records:
            games = rec["games"]
            if 1 not in games or 2 not in games:
                continue
            if champ_id in games[1]["picks"].get(team_id, []):
                continue  # picked in G1 -> locked in G2 under Fearless; ban is moot
            g2 = games[2]
            if team_id not in g2["picks"]:
                continue
            opp_bans = [c for t, bans in g2["bans"].items() if t != team_id for c in bans]
            won = g2["winner"] == team_id
            if champ_id in opp_bans:
                n_banned += 1
                w_banned += 1 if won else 0
            else:
                n_not += 1
                w_not += 1 if won else 0
        return n_banned, w_banned, n_not, w_not

    if args.diag:
        print("team | champ | G1 rate (games) | n_banned (wins) | n_not (wins) | delta")
        rows = []
        for team_id, by_game in team_pb.items():
            g1 = by_game.get(1)
            if not g1 or g1["games"] < 5:
                continue
            for champ_id, count in g1["picks"].items():
                rate = count / g1["games"]
                if rate < 0.3:
                    continue
                n_b, w_b, n_n, w_n = _arms(team_id, champ_id)
                delta = (w_b / n_b - w_n / n_n) if n_b and n_n else None
                rows.append((team_id, champ_id, rate, g1["games"], n_b, w_b, n_n, w_n, delta))
        rows.sort(key=lambda r: (-(r[8] is not None and abs(r[8] or 0)), -(r[4] or 0)))
        tname = {t["id"]: t["name"] for t in
                 [{"id": tid, "name": teams.get(tid, {}).get("name", tid)} for tid in team_pb]}
        cname = {c: champions.get(c, {}).get("name", c) for c in champions}
        for team_id, champ_id, rate, g1g, n_b, w_b, n_n, w_n, delta in rows:
            d = f"{delta:+.3f}" if delta is not None else "n/a"
            print(f"{tname.get(team_id, team_id):22s} | {cname.get(champ_id, champ_id):12s} | "
                  f"{rate:.2f} ({g1g:2d}) | {n_b} ({w_b}) | {n_n} ({w_n}) | {d}")
        print(f"pairs with G1 rate >= 0.3: {len(rows)}")
        return

    key_ban_shifts = []
    for team_id, by_game in team_pb.items():
        g1 = by_game.get(1)
        if not g1 or g1["games"] < 5:
            continue
        for champ_id, count in g1["picks"].items():
            rate = count / g1["games"]
            if rate < 0.3:
                continue
            n_banned, w_banned, n_not, w_not = _arms(team_id, champ_id)
            if n_banned >= 3 and n_not >= 3:
                wr_b = w_banned / n_banned
                wr_n = w_not / n_not
                key_ban_shifts.append({
                    "team_id": team_id,
                    "champion_id": champ_id,
                    "game1_pick_rate": round(rate, 3),
                    "game1_games": g1["games"],
                    "game2_wr_when_banned": round(wr_b, 3),
                    "game2_wr_when_not_banned": round(wr_n, 3),
                    "delta": round(wr_b - wr_n, 3),
                    "n_banned": n_banned,
                    "n_not_banned": n_not,
                })
    key_ban_shifts.sort(key=lambda s: -abs(s["delta"]))
    print(f"key-ban shifts found: {len(key_ban_shifts)}")

    # ---------- locked-per-game (observed across all multi-game series) ----------
    locked_per_game = {}
    for gn in range(2, 6):
        vals = []
        for rec in series_records:
            games = rec["games"]
            if gn not in games:
                continue
            prior = set()
            for pn in range(1, gn):
                if pn in games:
                    for champs in games[pn]["picks"].values():
                        prior.update(champs)
            vals.append(len(prior))
        if vals:
            locked_per_game[str(gn)] = {
                "avg_locked": round(sum(vals) / len(vals), 1),
                "series": len(vals),
            }

    # ---------- assemble output ----------
    out = {
        "meta": {
            "generated_by": "pipeline/extract_lol.py (Fearless Draft Lab)",
            "source": "GRID stl-hackathon-2026-grid-starter (LoL meta + feeds)",
            "source_seasons": "2025-2026",
            "series_processed": processed,
            "games_processed": games_with_bans,
            "games_with_draft": games_with_bans,
            "games_finished_total": n_games,
            "series_skipped": len(skipped),
            "swaps_corrected": swaps_detected,
            "bans_dropped_games": bans_dropped_games,
            "sub_picks_dropped": subs_dropped,
            "gaps_filled_from_feed": gaps_filled,
            "negation_events_handled": total_negations,
            "duplicate_events_removed": total_dupes,
            "rollbacks_seen": total_rollbacks,
            "fearless_like_series": fearless_like,
            "bo5_series": bo5_series,
            "note": "Aggregate statistics only. No raw GRID data, extracts, or copies of meta/series.csv are included.",
        },
        "teams": [
            {"id": tid, "name": teams.get(tid, {}).get("name", tid), "games": team_games.get(tid, 0)}
            for tid in sorted(team_games, key=lambda t: -team_games[t])
        ],
        "players": [
            {"id": pid, "name": players.get(pid, {}).get("name", pid),
             "team_id": players.get(pid, {}).get("team_id")}
            for pid in sorted(player_pool)
        ],
        "champions": [
            {"id": cid, "name": champions.get(cid, {}).get("name", cid)}
            for cid in sorted(champions, key=lambda c: champions[c]["name"])
        ],
        "team_pick_ban_by_game": {
            tid: {
                str(gn): {
                    "games": ent["games"],
                    "picks": {c: {"count": n, "rate": round(n / ent["games"], 3)} for c, n in ent["picks"].most_common(15)},
                    "bans": {c: {"count": n, "rate": round(n / ent["games"], 3)} for c, n in ent["bans"].most_common(15)},
                }
                for gn, ent in sorted(by_game.items())
            }
            for tid, by_game in team_pb.items()
        },
        "player_champion_pool": {
            pid: {
                "games": pp["games"],
                "unique": len(pp["champions"]),
                "champions": {c: n for c, n in pp["champions"].most_common(10)},
            }
            for pid, pp in player_pool.items() if pp["games"] >= 3
        },
        "champion_winrate": {
            cid: {
                "picks": cw["picks"],
                "wins": cw["wins"],
                "winrate": round(cw["wins"] / cw["picks"], 3) if cw["picks"] else 0,
                "by_game": {
                    str(gn): {"picks": v["picks"], "wins": v["wins"],
                              "winrate": round(v["wins"] / v["picks"], 3) if v["picks"] else 0}
                    for gn, v in sorted(cw["by_game"].items())
                },
            }
            for cid, cw in champ_wr.items()
        },
        "fearless": {
            "locked_per_game": locked_per_game,
            "fearless_like_series": fearless_like,
            "bo5_series": bo5_series,
            "key_ban_shifts": key_ban_shifts[:30],
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")
    print(f"teams in output: {len(out['teams'])}, champions: {len(out['champions'])}, "
          f"players with pool data: {len(out['player_champion_pool'])}")


if __name__ == "__main__":
    main()
