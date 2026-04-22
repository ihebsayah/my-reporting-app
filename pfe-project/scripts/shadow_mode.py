"""Shadow Mode Runner — Phase 1 of the agent deployment plan.

Sends each document through both pipelines in parallel:
  - Main pipeline (port 8000): existing heuristic/RF system
  - Agent pipeline (port 8001): new AI agent layer

Records agreements, disagreements, and timing for the shadow report.
Does NOT act on agent decisions — they are for comparison only.

Usage:
    python scripts/shadow_mode.py [--count N] [--output FILE]

Output:
    JSON Lines file with per-document comparison + summary report.
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
MAIN_URL  = "http://localhost:8000/api/v1/pipeline/run"
AGENT_URL = "http://localhost:8001/agents/extract"
DB_PATH   = Path(__file__).parent.parent / "data" / "reporting_app.db"

TIMEOUT = 20  # seconds per request


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_documents(limit: int) -> list[dict]:
    """Pull document texts from extraction_results DB for shadow testing."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, document_id, source_text, overall_decision "
        "FROM extraction_results "
        "ORDER BY id DESC "
        "LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "document_id": r[1], "text": r[2], "baseline_decision": r[3]}
        for r in rows
        if r[2] and len(r[2].strip()) > 20
    ]


def _call_main(text: str) -> dict:
    """Call the existing FastAPI pipeline."""
    try:
        r = httpx.post(MAIN_URL, json={"text": text}, timeout=TIMEOUT)
        d = r.json()
        return {
            "decision": d.get("overall_decision", d.get("decision", "unknown")),
            "confidence": d.get("overall_confidence", d.get("confidence", 0.0)),
            "scorer": d.get("scorer", "heuristic"),
            "status": "ok",
        }
    except Exception as exc:
        return {"decision": "error", "confidence": 0.0, "scorer": "error", "status": str(exc)}


def _call_agent(text: str, document_id: str) -> dict:
    """Call the agent microservice (shadow — result not acted upon)."""
    try:
        r = httpx.post(
            AGENT_URL,
            json={"text": text, "document_id": f"shadow_{document_id}"},
            timeout=TIMEOUT,
        )
        d = r.json()
        return {
            "action": d.get("action", "unknown"),
            "confidence": d.get("confidence", 0.0),
            "doc_type": d.get("doc_type", "unknown"),
            "agents_used": d.get("agents_used", []),
            "rails_triggered": d.get("safety_rails_triggered", []),
            "field_count": len(d.get("extracted_fields", [])),
            "issue_count": len(d.get("validation_issues", [])),
            "fallback_used": d.get("fallback_used", False),
            "duration_ms": d.get("duration_ms", 0),
            "status": "ok",
        }
    except Exception as exc:
        return {"action": "error", "confidence": 0.0, "status": str(exc)}


def _compare(main: dict, agent: dict) -> str:
    """Classify the relationship between main and agent decisions."""
    m = main.get("decision", "")
    a = agent.get("action", "")
    # Map agent actions to main decision vocabulary
    mapping = {"auto_approve": "auto", "human_review": "review", "reject": "reject"}
    a_mapped = mapping.get(a, a)
    if m == a_mapped:
        return "agree"
    if a_mapped == "human_review" and m == "auto":
        return "agent_more_cautious"
    if a_mapped == "auto_approve" and m == "review":
        return "agent_more_confident"
    if a_mapped == "reject":
        return "agent_rejects"
    return "disagree"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_shadow(count: int, output_path: Path) -> None:
    print(f"Shadow Mode — {count} documents")
    print(f"  Main API:  {MAIN_URL}")
    print(f"  Agent svc: {AGENT_URL}")
    print(f"  Output:    {output_path}")
    print()

    docs = _fetch_documents(count)
    if not docs:
        print("❌ No documents found in the DB. Run the seed script first.")
        sys.exit(1)

    print(f"  Loaded {len(docs)} documents from DB.")
    print()

    results = []
    stats = {"agree": 0, "agent_more_cautious": 0, "agent_more_confident": 0,
             "agent_rejects": 0, "disagree": 0, "error": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fh:
        for i, doc in enumerate(docs, 1):
            agent_result = _call_agent(doc["text"], doc["document_id"])
            main_result  = _call_main(doc["text"])

            outcome = _compare(main_result, agent_result)
            if agent_result.get("status") != "ok" or main_result.get("status") != "ok":
                outcome = "error"

            stats[outcome] = stats.get(outcome, 0) + 1

            record = {
                "document_id": doc["document_id"],
                "baseline_decision": doc["baseline_decision"],
                "main": main_result,
                "agent": agent_result,
                "outcome": outcome,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            results.append(record)
            fh.write(json.dumps(record) + "\n")

            print(
                f"  [{i:>3}/{len(docs)}] {doc['document_id'][:38]:<38} "
                f"main={main_result['decision']:<8} "
                f"agent={agent_result.get('action','err'):<14} "
                f"→ {outcome}"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    total_done = len(results)
    agree_rate = stats["agree"] / total_done if total_done else 0
    avg_agent_conf = (
        sum(r["agent"].get("confidence", 0) for r in results) / total_done
        if total_done else 0
    )
    avg_agent_ms = (
        sum(r["agent"].get("duration_ms", 0) for r in results) / total_done
        if total_done else 0
    )

    print()
    print("=" * 60)
    print("SHADOW MODE REPORT")
    print("=" * 60)
    print(f"  Documents processed : {total_done}")
    print(f"  Agreement rate      : {agree_rate:.1%}  ({stats['agree']}/{total_done})")
    print(f"  Agent more cautious : {stats['agent_more_cautious']}")
    print(f"  Agent more confident: {stats['agent_more_confident']}")
    print(f"  Agent rejects       : {stats['agent_rejects']}")
    print(f"  Disagreements       : {stats['disagree']}")
    print(f"  Errors              : {stats['error']}")
    print(f"  Avg agent confidence: {avg_agent_conf:.1%}")
    print(f"  Avg agent latency   : {avg_agent_ms:.0f}ms")
    print()

    if agree_rate >= 0.80:
        print("✅ Agreement ≥80% — ready for Canary phase (route 5% of traffic to agents)")
    elif agree_rate >= 0.65:
        print("⚠️  Agreement 65–80% — review disagreements before Canary deployment")
    else:
        print("❌ Agreement <65% — investigate agent decisions before any live routing")

    print()
    print(f"Full results: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shadow Mode — compare agent vs main pipeline")
    parser.add_argument("--count", type=int, default=40, help="Number of documents to process (default: 40)")
    parser.add_argument("--output", type=str,
                        default=f"artifacts/shadow_mode/shadow_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()
    run_shadow(args.count, Path(args.output))
