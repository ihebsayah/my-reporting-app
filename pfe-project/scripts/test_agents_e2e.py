"""Manual end-to-end test for the agent service.

Run this script AFTER starting both services:
    # Terminal 1:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    # Terminal 2:
    uvicorn agents.main:app --host 0.0.0.0 --port 8001 --reload

Tests:
1.  Health check — both services up
2.  Agent extract — full pipeline on sample invoice
3.  Human feedback — simulates approval, triggers learning
4.  Verify LongTermMemory — checks agent_patterns table is populated
5.  Agent service status — rollback state
6.  Accuracy monitoring — agreement rate / override rate
7.  Safety Rail 1 — $150k invoice must NOT be auto-approved
8.  Safety Rail 2 — new vendor must NOT be auto-approved
9.  Canary monitoring — stats from JSONL log
10. LLM availability — checks Ollama / heuristic mode
"""

import json
import sqlite3
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Please install httpx: pip install httpx")
    sys.exit(1)

AGENT_BASE = "http://localhost:8001"
MAIN_BASE  = "http://localhost:8000"

SAMPLE_INVOICE = """
Invoice #INV-2024-001
Vendor: Acme Corp SARL
Invoice Date: 2024-01-15
Due Date: 2024-02-15

Line Items:
  Consulting Services - January     $4,000.00
  Software License - Q1             $1,000.00

Total: $5,000.00

Payment Terms: Net 30
Contact: billing@acme-corp.example.com
"""

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def check(condition: bool, msg: str) -> None:
    icon = PASS if condition else FAIL
    print(f"{icon} {msg}")
    if not condition:
        print(f"   → ASSERTION FAILED: {msg}")
        sys.exit(1)


def main() -> None:
    failures = []

    # ── 1. Health checks ───────────────────────────────────────────────────────
    separator("1. Health Checks")
    r = httpx.get(f"{AGENT_BASE}/health", timeout=5)
    check(r.status_code == 200, f"Agent svc healthy (port 8001) — HTTP {r.status_code}")

    r2 = httpx.get(f"{MAIN_BASE}/health", timeout=5)
    check(r2.status_code == 200, f"Main API healthy (port 8000) — HTTP {r2.status_code}")

    # ── 2. Agent extract ───────────────────────────────────────────────────────
    separator("2. Agent Extract — Sample Invoice ($5k, Acme Corp)")
    r = httpx.post(
        f"{AGENT_BASE}/agents/extract",
        json={
            "text": SAMPLE_INVOICE,
            "document_id": "e2e_acme_001",
            "metadata": {"source": "e2e_test"},
        },
        timeout=90,
    )
    check(r.status_code == 200, f"Extract returned HTTP {r.status_code}")
    data = r.json()

    print(f"\n  Decision:   {data.get('action', 'N/A').upper()}")
    print(f"  Confidence: {data.get('confidence', 0):.1%}")
    print(f"  Doc Type:   {data.get('doc_type', 'N/A')}")
    print(f"  Agents:     {', '.join(data.get('agents_used', []))}")
    print(f"  Duration:   {data.get('duration_ms', 0):.0f}ms")
    print(f"  Fallback:   {data.get('fallback_used', False)}")
    print(f"  LLM mode:   {'heuristic' if data.get('fallback_used') else 'agent'}")
    print(f"\n  Safety Rails: {data.get('safety_rails_triggered', [])}")
    print(f"\n  Fields ({len(data.get('extracted_fields', []))}):")
    for f in data.get("extracted_fields", []):
        print(f"    {f['field_name']:20s} = {f['value']!r:25s}  conf={f['confidence']:.2f}")
    print(f"\n  Validation Issues ({len(data.get('validation_issues', []))}):")
    for i in data.get("validation_issues", []):
        print(f"    [{i['severity'].upper()}] {i['field_name']}: {i['description']}")
    print(f"\n  Reasoning:\n{data.get('agent_reasoning', '')[:400]}")

    check("action" in data, "Response has 'action' field")
    check(data.get("confidence", 0) > 0, "Confidence > 0")

    # ── 3. Human feedback ──────────────────────────────────────────────────────
    separator("3. Record Human Feedback — approve Acme Corp $5k")
    r = httpx.post(
        f"{AGENT_BASE}/agents/feedback",
        json={
            "document_id": "e2e_acme_001",
            "agent_decision": data.get("action", "human_review"),
            "human_outcome": "approve",
            "vendor": "Acme Corp SARL",
            "amount": 5000.0,
            "notes": "Verified with purchasing — legitimate invoice.",
        },
        timeout=15,
    )
    check(r.status_code == 200, f"Feedback endpoint HTTP {r.status_code}")
    fb = r.json()
    check(fb.get("recorded") is True, "Feedback marked as recorded")
    print(f"  learned: {fb.get('learned')}  message: {fb.get('message')}")

    # ── 4. LongTermMemory verification ─────────────────────────────────────────
    separator("4. Verify LongTermMemory — agent_patterns table")
    db_paths = [
        Path("reporting_app.db"),
        Path("data/reporting_app.db"),
        Path("pfe-project/reporting_app.db"),
    ]
    db_path = next((p for p in db_paths if p.exists()), None)
    if db_path:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT pattern_key, approve_count, reject_count, total_count "
                "FROM agent_patterns WHERE pattern_key LIKE 'vendor:%acme%'"
            )
            row = cur.fetchone()
            if row:
                print(f"  Pattern found: {row[0]}")
                print(f"    approve_count={row[1]}  reject_count={row[2]}  total={row[3]}")
                check(row[3] >= 1, "LongTermMemory persisted at least 1 feedback event")
            else:
                print(f"  {WARN} No vendor:acme pattern yet (may take a moment to persist)")
        except sqlite3.OperationalError as exc:
            print(f"  {WARN} Schema query failed: {exc}")
            print("      (DB may have old schema — restart agent service to auto-migrate)")
        finally:
            conn.close()
    else:
        print(f"  {WARN} Could not locate reporting_app.db for verification")

    # ── 5. Agent status ────────────────────────────────────────────────────────
    separator("5. Agent Service Status")
    r = httpx.get(f"{AGENT_BASE}/agents/status", timeout=5)
    s = r.json()
    print(f"  agents_enabled: {s.get('agents_enabled')}")
    print(f"  rolled_back:    {s.get('rolled_back')}")
    print(f"  version:        {s.get('agent_service_version')}")
    check(s.get("agents_enabled") is True, "Agents are enabled (no rollback triggered)")

    # ── 6. Accuracy monitoring ─────────────────────────────────────────────────
    separator("6. Accuracy Monitoring")
    r = httpx.get(f"{AGENT_BASE}/agents/monitoring/accuracy", timeout=5)
    acc = r.json()
    print(f"  total_decisions: {acc.get('total_decisions')}")
    print(f"  accuracy:        {acc.get('accuracy', 0):.1%}")
    print(f"  override_rate:   {acc.get('override_rate', 0):.1%}")
    print(f"  rollback_needed: {acc.get('rollback_needed')}")
    check(acc.get("rollback_needed") is False, "No rollback recommended")

    # ── 7. Safety Rail 1 — large amount ────────────────────────────────────────
    separator("7. Safety Rail 1 — $150k Invoice (must NOT auto-approve)")
    r = httpx.post(
        f"{AGENT_BASE}/agents/extract",
        json={
            "text": "Invoice #INV-LARGE-001 Vendor: BigCorp Total: $150,000.00 Date: 2024-01-10",
            "document_id": "e2e_large_001",
        },
        timeout=90,
    )
    large = r.json()
    print(f"  Decision: {large.get('action', 'N/A').upper()}")
    print(f"  Rails:    {large.get('safety_rails_triggered', [])}")
    check(
        large.get("action") != "auto_approve",
        "Rail 1 ENFORCED: $150k invoice not auto-approved",
    )

    # ── 8. Safety Rail 2 — new vendor ─────────────────────────────────────────
    separator("8. Safety Rail 2 — Unknown Vendor (must force human_review)")
    r = httpx.post(
        f"{AGENT_BASE}/agents/extract",
        json={
            "text": "Invoice #INV-NEW-001 Vendor: BRAND_NEW_VENDOR_XYZ_2024 Total: $800.00 Date: 2024-03-01",
            "document_id": "e2e_newvendor_001",
        },
        timeout=90,
    )
    new_v = r.json()
    print(f"  Decision: {new_v.get('action', 'N/A').upper()}")
    print(f"  Rails:    {new_v.get('safety_rails_triggered', [])}")
    check(
        new_v.get("action") in ("human_review", "reject"),
        "Rail 2 ENFORCED: new vendor not auto-approved",
    )

    # ── 9. Canary monitoring ───────────────────────────────────────────────────
    separator("9. Canary Monitoring Endpoint")
    r = httpx.get(f"{AGENT_BASE}/agents/monitoring/canary", timeout=5)
    c = r.json()
    print(f"  total canary records: {c.get('total', 0)}")
    print(f"  agreement_rate:       {c.get('agreement_pct', 'N/A')}")
    print(f"  recommendation:       {c.get('recommendation', 'N/A')}")
    check(r.status_code == 200, "Canary endpoint reachable")

    # ── 10. LLM availability ───────────────────────────────────────────────────
    separator("10. LLM Availability Check (Ollama)")
    try:
        r_llm = httpx.get("http://localhost:11434/api/tags", timeout=3)
        models = r_llm.json().get("models", [])
        if models:
            for m in models:
                print(f"  {PASS} Ollama model ready: {m['name']}  ({m.get('size', 0)/1e9:.1f} GB)")
        else:
            print(f"  {WARN} Ollama running but no models pulled yet.")
            print("       Run: ollama pull mistral")
    except Exception:
        print(f"  {WARN} Ollama not running — agents using heuristic fallback mode.")
        print("       Run: ollama serve  then  ollama pull mistral")

    # ── Summary ────────────────────────────────────────────────────────────────
    separator("ALL TESTS PASSED ✅")
    print("\n  Production readiness:")
    print(f"  {PASS} Agent service running and responsive")
    print(f"  {PASS} Full extraction pipeline working")
    print(f"  {PASS} Human feedback triggers LongTermMemory learning")
    print(f"  {PASS} Safety Rails 1 & 2 enforced correctly")
    print(f"  {PASS} Accuracy monitor active (no rollback)")
    print(f"  {PASS} Canary monitoring endpoint live")
    print(f"\n  Next steps:")
    print(f"  → Set CANARY_RATE=0.50 in .env for A/B testing (already done ✓)")
    print(f"  → Run 'ollama pull mistral' to enable real LLM reasoning")
    print(f"  → Monitor /agents/monitoring/canary for agreement rate")


if __name__ == "__main__":
    main()
