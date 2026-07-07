"""Builds the live-training dashboard HTML from runs/ state.

Usage: .venv/bin/python build_dashboard.py --out /path/to/dashboard.html
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
MAX_POINTS = 240      # downsample metric series for the page
MAX_EVENTS = 12
MAX_TABLE = 30


def read_jsonl(path, limit=None):
    try:
        lines = Path(path).read_text().strip().splitlines()
    except FileNotFoundError:
        return []
    rows = [json.loads(l) for l in lines if l.strip()]
    return rows[-limit:] if limit else rows


def downsample(rows, n):
    if len(rows) <= n:
        return rows
    idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    return [rows[i] for i in sorted(set(idx))]


def build_payload():
    metrics = read_jsonl(RUNS / "metrics.jsonl")
    try:
        status = json.loads((RUNS / "status.json").read_text())
    except FileNotFoundError:
        status = {"state": "starting", "iteration": 0, "config": {}}
    events = read_jsonl(RUNS / "events.jsonl", MAX_EVENTS)
    try:
        replay = json.loads((RUNS / "replay.json").read_text())
    except FileNotFoundError:
        replay = None

    ds = downsample(metrics, MAX_POINTS)
    return {
        "builtAt": time.time(),
        "status": status,
        "events": events,
        "replay": replay,
        "totals": {
            "iterations": metrics[-1]["iter"] if metrics else 0,
            "games": sum(m["games"] for m in metrics),
            "steps": sum(m["steps"] for m in metrics),
        },
        "recent": metrics[-MAX_TABLE:],
        "series": {
            "iter": [m["iter"] for m in ds],
            "entropy": [m["entropy"] for m in ds],
            "value_loss": [m["value_loss"] for m in ds],
            "mean_score": [m["mean_score"] for m in ds],
            "mean_len": [m["mean_len"] for m in ds],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", action="store_true",
                    help="write the raw data payload (for the PWA) instead of HTML")
    args = ap.parse_args()

    payload = build_payload()
    blob = json.dumps(payload, separators=(",", ":"))
    if args.json:
        Path(args.out).write_text(blob)
        print(f"app data built: {args.out} ({len(blob)//1024} KB)")
        return
    tpl = (ROOT / "dashboard_template.html").read_text()
    assert "/*__DATA__*/null" in tpl
    out = tpl.replace("/*__DATA__*/null", blob)
    Path(args.out).write_text(out)
    print(f"dashboard built: {args.out} ({len(out)//1024} KB)")


if __name__ == "__main__":
    main()
