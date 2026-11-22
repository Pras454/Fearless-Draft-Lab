#!/usr/bin/env python3
"""Inspect one GRID LoL series feed: print envelope/event structure.

Streaming only — never decompresses a whole file to disk.
Usage: inspect.py [--data-dir DIR] [--file NAME] [--max-lines N]
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import zstandard


def stream_jsonl_zst(path, max_lines=None):
    """Yield parsed JSON objects, one per line, from a .jsonl.zst file."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = b""
            n = 0
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        yield json.loads(line)
                        n += 1
                        if max_lines and n >= max_lines:
                            return
            if buf.strip():
                yield json.loads(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "stl-hackathon-2026-grid-starter"))
    ap.add_argument("--file", default=None, help="specific lol file name, else first file in lol/")
    ap.add_argument("--max-lines", type=int, default=50)
    args = ap.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    # meta CSV headers
    meta_dir = os.path.join(data_dir, "meta")
    print("== meta/ CSVs ==")
    for name in sorted(os.listdir(meta_dir)):
        if not name.endswith(".csv"):
            continue
        p = os.path.join(meta_dir, name)
        with open(p, newline="", encoding="utf-8") as fh:
            r = csv.reader(fh)
            header = next(r)
            row1 = next(r, None)
            print(f"{name}: {len(header)} cols")
            print(f"  header: {header}")
            if row1:
                print(f"  row1:   {row1}")

    lol_dir = os.path.join(data_dir, "lol")
    files = sorted(f for f in os.listdir(lol_dir) if f.endswith(".jsonl.zst"))
    print(f"\n== lol/ files: {len(files)} ==")
    print("first 5:", files[:5])

    fname = args.file or files[0]
    path = os.path.join(lol_dir, fname)
    print(f"\n== inspecting {fname} (first {args.max_lines} envelopes) ==")

    event_types = Counter()
    n_env = 0
    first_env = None
    draft_samples = []
    won_samples = []
    rollback_samples = []
    for env in stream_jsonl_zst(path, max_lines=args.max_lines):
        n_env += 1
        if first_env is None:
            first_env = env
        for ev in env.get("events", []):
            t = ev.get("type", "?")
            event_types[t] += 1
            if "picked-character" in t or "banned-character" in t:
                if len(draft_samples) < 3:
                    draft_samples.append(ev)
            if "won-game" in t or "won-series" in t:
                if len(won_samples) < 3:
                    won_samples.append(ev)
            if "!started-game" in t:
                if len(rollback_samples) < 3:
                    rollback_samples.append(ev)

    print(f"envelopes read: {n_env}")
    print(f"first envelope keys: {sorted(first_env.keys())}")
    print(f"first envelope events[0] keys: {sorted(first_env['events'][0].keys()) if first_env.get('events') else 'no events'}")
    print("\nevent type counts (first N envelopes):")
    for t, c in event_types.most_common(30):
        print(f"  {c:5d}  {t}")

    def show(label, samples):
        print(f"\n-- {label} ({len(samples)} samples) --")
        for s in samples:
            print(json.dumps(s, indent=1, default=str)[:2000])

    show("draft events", draft_samples)
    show("won events", won_samples)
    show("rollback events", rollback_samples)

    # seriesState shape from the first envelope that has one
    for env in stream_jsonl_zst(path, max_lines=5):
        st = None
        for ev in env.get("events", []):
            if ev.get("seriesState"):
                st = ev["seriesState"]
                break
        if st:
            print("\n-- seriesState top-level keys --")
            print(sorted(st.keys()))
            if "games" in st:
                print(f"games: {len(st['games'])} entries; first game keys: {sorted(st['games'][0].keys()) if st['games'] else []}")
            if "teams" in st:
                print(f"teams: {len(st['teams'])} entries; first team keys: {sorted(st['teams'][0].keys()) if st['teams'] else []}")
            break


if __name__ == "__main__":
    main()
