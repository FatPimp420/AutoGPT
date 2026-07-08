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
    if n <= 0:
        return []
    if n == 1:
        return rows[-1:]
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
    keys = ("iter", "entropy", "value_loss", "mean_score", "mean_len")
    series = {k: [m[k] for m in ds] for k in keys}
    totals = {
        "iterations": metrics[-1]["iter"] if metrics else 0,
        "games": sum(m["games"] for m in metrics),
        "steps": sum(m["steps"] for m in metrics),
    }

    # runs/history.json is the last payload pushed before a container reset:
    # splice its series/totals in so curves stay continuous across resets.
    try:
        hist = json.loads((RUNS / "history.json").read_text())
        first = metrics[0]["iter"] if metrics else float("inf")
        keep = [i for i, it in enumerate(hist["series"]["iter"]) if it < first]
        idx = downsample(keep, MAX_POINTS - len(series["iter"]) if metrics else MAX_POINTS)
        for k in keys:
            series[k] = [hist["series"][k][i] for i in idx] + series[k]
        totals["iterations"] = max(totals["iterations"], hist["totals"]["iterations"])
        totals["games"] += hist["totals"]["games"]
        totals["steps"] += hist["totals"]["steps"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass

    return {
        "builtAt": time.time(),
        "status": status,
        "events": events,
        "replay": replay,
        "totals": totals,
        "recent": metrics[-MAX_TABLE:],
        "series": series,
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
