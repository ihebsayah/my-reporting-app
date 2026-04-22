"""Canary Report — analyse the live canary log and print agreement stats.

Reads ``artifacts/canary/canary.jsonl`` (written by app/api/agent_bridge.py)
and shows:
  - Total canary calls processed
  - Agreement rate (main pipeline vs agent)
  - Breakdown by outcome type
  - Safety rails trigger frequency
  - Rolling 50-request agreement trend (last N)
  - go/no-go recommendation for A/B phase

Usage:
    python scripts/canary_report.py [--last N] [--file PATH]
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def load_records(log_file: Path, last: Optional[int]) -> List[Dict]:
    if not log_file.exists():
        return []
    records = [json.loads(l) for l in log_file.open() if l.strip()]
    return records[-last:] if last else records


def run_report(log_file: Path, last: Optional[int]) -> None:
    records = load_records(log_file, last)
    total = len(records)

    if total == 0:
        print(f"No canary records found in {log_file}")
        print("Tip: set AGENT_SERVICE_URL=http://localhost:8001 and "
              "CANARY_RATE=0.05 (or 1.0 for testing) then send traffic.")
        return

    print(f"Canary Report  —  {log_file}")
    print(f"{'─' * 60}")
    print(f"  Records analysed  : {total}{f'  (last {last})' if last else ''}")

    # ── Agreement ─────────────────────────────────────────────────────────
    outcomes = Counter(r["agreement"] for r in records)
    agree_n = outcomes.get("agree", 0)
    agree_rate = agree_n / total

    print(f"  Agreement rate    : {agree_rate:.1%}  ({agree_n}/{total})")
    print(f"  Disagree          : {outcomes.get('disagree', 0)}")
    print()

    # ── Decision breakdown ─────────────────────────────────────────────────
    main_counts  = Counter(r["main_decision"] for r in records)
    agent_counts = Counter(r["agent_action_mapped"] for r in records)
    print("  Main pipeline decisions:")
    for dec, cnt in sorted(main_counts.items()):
        print(f"    {dec:<15} {cnt:>4}  ({cnt/total:.0%})")
    print("  Agent decisions (mapped):")
    for dec, cnt in sorted(agent_counts.items()):
        print(f"    {dec:<15} {cnt:>4}  ({cnt/total:.0%})")
    print()

    # ── Safety rails ───────────────────────────────────────────────────────
    all_rails = []
    for r in records:
        all_rails.extend(r.get("rails_triggered", []))
    if all_rails:
        rail_counts = Counter(all_rails)
        print(f"  Safety rails triggered ({sum(rail_counts.values())} total):")
        for rail, cnt in rail_counts.most_common():
            print(f"    {rail:<50} ×{cnt}")
        print()

    # ── Performance ───────────────────────────────────────────────────────
    avg_conf = sum(r["agent_confidence"] for r in records) / total
    avg_ms   = sum(r.get("duration_ms", 0) for r in records) / total
    fallback_n = sum(1 for r in records if r.get("fallback_used"))
    print(f"  Avg agent confidence : {avg_conf:.1%}")
    print(f"  Avg agent latency    : {avg_ms:.0f} ms")
    print(f"  Fallback used        : {fallback_n} / {total}")
    print()

    # ── Rolling trend (last 50) ────────────────────────────────────────────
    if total >= 10:
        window = min(50, total)
        recent = records[-window:]
        recent_agree = sum(1 for r in recent if r["agreement"] == "agree")
        recent_rate  = recent_agree / len(recent)
        trend = "↑" if recent_rate > agree_rate else ("↓" if recent_rate < agree_rate else "→")
        print(f"  Last {window} requests      : {recent_rate:.1%} agreement {trend}")
        print()

    # ── go/no-go ──────────────────────────────────────────────────────────
    print("─" * 60)
    if agree_rate >= 0.90:
        print("✅ Agreement ≥90% — ready for A/B phase (route 50% to agents)")
    elif agree_rate >= 0.80:
        print("✅ Agreement ≥80% — continue Canary; review disagreements")
    elif agree_rate >= 0.65:
        print("⚠️  Agreement 65–80% — investigate before expanding Canary")
    else:
        print("❌ Agreement <65% — roll back Canary, fix agent calibration")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canary phase agreement report")
    parser.add_argument("--last", type=int, default=None,
                        help="Analyse only the last N canary records")
    parser.add_argument("--file", type=str,
                        default="artifacts/canary/canary.jsonl",
                        help="Path to the canary JSONL log file")
    args = parser.parse_args()
    run_report(Path(args.file), args.last)
