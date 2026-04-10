# AI AGENT INTEGRATION PLAN
## Complete Design for Adding Agents to Your Extraction System

**Status:** All 18 decisions LOCKED ✅
**Complexity:** Medium (augment existing system, don't replace)
**Timeline:** 8-12 weeks (parallel to existing operations)

---

## EXECUTIVE SUMMARY

You're adding **5 AI agents** (1 master + 4 specialized) to your existing extraction system.

**What changes:**
- ❌ NOT replacing spaCy NER, Random Forest, BART (keep them)
- ✅ Adding intelligent reasoning layer on top
- ✅ Agents make smarter decisions using context + history
- ✅ Humans still approve/override all decisions
- ✅ Agents learn from feedback in real-time

**Expected improvements:**
- Auto-approve rate: 65% → 72%+ (fewer human reviews)
- Decision accuracy: 94% → 96%+
- Processing speed: 4ms → 8ms (agents add reasoning overhead)
- Human override rate: 28% → 18% (smarter decisions)

---

## LOCKED DECISIONS (18 TOTAL)

| # | Decision | Answer | Status |
|---|----------|--------|--------|
| 1 | Autonomy | Partial (agents assist, humans decide) | ✅ |
| 2 | Architecture | Hybrid (1 master + 4 sub-agents) | ✅ |
| 3 | LLM | Open-source (Llama 2/Mistral) | ✅ |
| 4 | Framework | LangChain | ✅ |
| 5 | Reasoning | ReAct (Reasoning + Acting loop) | ✅ |
| 6 | Tools | ML models + DB + External APIs | ✅ |
| 7 | Coordination | Sequential (master calls sub-agents) | ✅ |
| 8 | Memory | Hybrid (short-term + long-term) | ✅ |
| 9 | Failures | Retry with fallback strategies | ✅ |
| 10 | Integration | Wrap ML models, replace decision logic | ✅ |
| 11 | Oversight | Humans review flagged decisions only | ✅ |
| 12 | Learning | Real-time from human feedback | ✅ |
| 13 | Deployment | Parallel + A/B test (4 phases) | ✅ |
| 14 | Infrastructure | Separate microservice | ✅ |
| 15 | Team | Master + 4 agents (5 total) | ✅ |
| 16 | Memory Storage | Redis (real-time) + Postgres (long-term) | ✅ |
| 17 | Safety | Monitor everything + safety rails + auto-rollback | ✅ |
| 18 | Adoption | DEFAULT (can be globally disabled) | ✅ |

---

## ARCHITECTURE OVERVIEW

```
USER UPLOADS DOCUMENT
        ↓
FASTAPI (existing code, unchanged)
        ├─ Extract text (OCR, PDF parsing)
        ├─ Run ML models (spaCy, BART, RF)
        ├─ Calculate base confidence
        └─ Call Agent Service (NEW)
        ↓
AGENT SERVICE (new microservice)
        ├─ Master Agent
        │   ├─ Classifier Agent
        │   ├─ Extractor Agent
        │   ├─ Validator Agent
        │   └─ Router Agent
        ├─ Redis (real-time learning)
        └─ PostgreSQL (queries, history)
        ↓
DECISION LOGIC (replaced by agents)
        ├─ Auto-approve
        ├─ Human review
        └─ Reject
        ↓
HUMAN DASHBOARD
        ├─ Agent reasoning shown
        ├─ Can override
        └─ Feedback recorded
        ↓
DATABASE + REDIS
        ├─ Store results
        └─ Agent learns
```

---

## THE 5 AGENTS EXPLAINED

### MASTER AGENT: "Extraction Supervisor"

**Role:** Orchestrate entire extraction workflow

**Thinks:** 
```
"Document uploaded. Let me:
1. Classify document type
2. Extract fields
3. Validate extracted data
4. Make final routing decision
5. Present to human with reasoning"
```

**Tools:**
- Call Classifier Agent
- Call Extractor Agent
- Call Validator Agent
- Call Router Agent
- Query PostgreSQL (history)
- Escalate to human

**Memory:**
- Short-term: Current document conversation
- Long-term: Patterns (what worked before)

**Output:** Final decision + reasoning explanation

---

### AGENT 1: "Document Classifier"

**Role:** Determine document type

**Thinks:**
```
"What type of document is this?
- Invoice? (has amounts, dates, vendors)
- Receipt? (similar but shorter)
- Contract? (longer, legal language)
- Report? (tabular data)
- Letter? (natural language text)"
```

**Tools:**
- BART zero-shot classifier (existing)
- Database lookup: "Have we seen this before?"
- NLP features: word frequency, document length

**Input:** Document text
**Output:** {doc_type: "invoice", confidence: 0.95}

**Example:**
```
Input: "Invoice #123 from Acme Corp for $5000"
Output: {doc_type: "invoice", confidence: 0.98}
```

---

### AGENT 2: "Field Extractor"

**Role:** Extract all relevant fields

**Thinks:**
```
"For an invoice, I need to find:
- Invoice ID: Look for patterns like 'INV-XXXX'
- Date: Look for date patterns
- Vendor: Who sent this? Look for company names
- Amount: Currency values
- Line items: Multiple amounts in a table"
```

**Tools:**
- spaCy NER (trained, existing)
- Regex patterns
- Random Forest confidence scorer
- ML feature extraction

**Input:** Document text + document type
**Output:** {fields: {amount, date, vendor, invoice_id}, confidences}

**Example:**
```
Input: "Invoice #12345 from Acme dated Jan 15 for $5000"
Output: {
  invoice_id: {value: "12345", confidence: 0.95},
  vendor: {value: "Acme", confidence: 0.87},
  date: {value: "2024-01-15", confidence: 0.92},
  amount: {value: 5000, confidence: 0.90}
}
```

---

### AGENT 3: "Validator"

**Role:** Check if extractions make sense

**Thinks:**
```
"Let me verify the extracted data:
- Does vendor exist in our database?
- Is the amount reasonable for this vendor?
- Is the date in the past (not future)?
- Are there any red flags?
- Should I boost or lower confidence?"
```

**Tools:**
- PostgreSQL queries (vendor lookup, history)
- Business rules (amount limits, date ranges)
- Anomaly detection (Isolation Forest)
- Historical patterns

**Input:** Extracted fields
**Output:** {is_valid, anomalies, confidence_adjustment}

**Example:**
```
Input: {vendor: "Acme", amount: 5000}
Database query: "What's the usual range for Acme invoices?"
Response: "Range: $1000-$8000, avg: $4500"
Output: {
  is_valid: true,
  anomalies: [],
  confidence_adjustment: +0.02  # Amount is normal
}
```

---

### AGENT 4: "Router"

**Role:** Decide where document goes (auto-approve, review, reject)

**Thinks:**
```
"Based on everything I know:
- Extraction confidence: 0.88
- Validation score: 0.92
- Vendor history: Usually clean
- Amount reasonableness: Typical
- Should I auto-approve, request review, or reject?"
```

**Tools:**
- Confidence thresholds (configurable)
- Historical patterns from Redis
- Safety rails (business rules)
- Human override history

**Input:** All previous agent outputs
**Output:** {action: "auto_approve"|"human_review"|"reject", reasoning}

**Example:**
```
Input: {
  extraction_confidence: 0.88,
  validation_score: 0.92,
  vendor_history: "clean",
  amount_deviation: "normal"
}
Output: {
  action: "auto_approve",
  reasoning: "Confidence 0.88 is borderline, but vendor is known 
              and amount is within normal range. Safe to approve."
}
```

---

## AGENT WORKFLOW (STEP BY STEP)

### Example: Processing an invoice

```
STEP 1: Human uploads "acme_invoice_2024.pdf"
        ↓
STEP 2: FastAPI extracts text + base metadata
        ↓
STEP 3: FastAPI calls Agent Service
        POST /agents/extract {text: "...", metadata: {...}}
        ↓
STEP 4: MASTER AGENT receives document
        Thinks: "New document, let me process it"
        ↓
STEP 5: MASTER calls CLASSIFIER AGENT
        "What type of document?"
        ↓
STEP 6: CLASSIFIER processes
        Runs BART: "Invoice"
        Checks DB: "Seen many invoices"
        Returns: {doc_type: "invoice", confidence: 0.98}
        ↓
STEP 7: MASTER calls EXTRACTOR AGENT
        "Extract invoice fields"
        ↓
STEP 8: EXTRACTOR processes
        Runs spaCy NER: finds "Acme", "2024-01-15", "$5000"
        Scores confidence: 0.87 (borderline)
        Returns: {
          vendor: {value: "Acme", conf: 0.87},
          date: {value: "2024-01-15", conf: 0.92},
          amount: {value: 5000, conf: 0.90}
        }
        ↓
STEP 9: MASTER calls VALIDATOR AGENT
        "Check if these make sense"
        ↓
STEP 10: VALIDATOR processes
         Queries DB: "Acme's invoices range $1k-$8k"
         Checks: Amount $5000 is normal ✓
         Checks: Date is in past ✓
         Checks: Vendor exists ✓
         Detects: No anomalies
         Returns: {
           is_valid: true,
           anomalies: [],
           confidence_boost: +0.02
         }
        ↓
STEP 11: MASTER calls ROUTER AGENT
         "Make final decision"
        ↓
STEP 12: ROUTER processes
         Checks thresholds:
         - Extraction confidence: 0.87 (borderline)
         - Validation passed: true
         - Vendor known: yes (from memory)
         - Amount normal: yes
         Thinks: "Even though extraction confidence is 0.87,
                  all other signals say safe"
         Decides: "auto_approve"
         Returns: {
           action: "auto_approve",
           reasoning: "Vendor known, amount normal, 
                      validation passed. Safe."
         }
        ↓
STEP 13: MASTER returns to FastAPI
         {
           action: "auto_approve",
           agent_reasoning: "Vendor known...",
           extracted_fields: {...},
           confidence: 0.89
         }
        ↓
STEP 14: FastAPI returns to Dashboard
         Human sees:
         - Extracted fields (vendor, date, amount)
         - Agent decision: "AUTO-APPROVE"
         - Agent reasoning: "Vendor known, amount normal..."
         - Can override: [APPROVE] [REVIEW] [REJECT]
        ↓
STEP 15: Human clicks [APPROVE]
         ✓ Document approved
         Agent learns: "This case approved"
        ↓
STEP 16: (ALTERNATIVE) Human clicks [REJECT]
         Agent learns: "Wait, I was wrong about this"
         Stores to Redis: {vendor:acme:5000 → rejected}
         Next similar case: "I remember this was rejected"
```

---

## AGENT LEARNING (REAL-TIME)

### Example: Agent adapts to feedback

**Initial case:**
```
Document: Acme invoice, $5000
Agent decision: "auto_approve"
Human decision: "approve" ✓
Agent learns: "Acme + $5000 = safe"
Stored in Redis: {vendor:acme:5000: approved}
```

**Second case (one week later):**
```
Document: Acme invoice, $5000 (same pattern)
Agent memory check: "I've seen this before... approved it"
Agent thinks: "Similar case, probably safe"
Agent decision: "auto_approve"
Human decision: "approve" ✓
Confidence increases
```

**Third case (someone new vendor):**
```
Document: Unknown vendor, $5000
Agent memory check: "No history for this vendor"
Agent thinks: "Unknown vendor, need caution"
Agent decision: "human_review" (recommend review)
Human decision: "reject" ✗
Agent learns: "New vendors need scrutiny"
Stored in Redis: {vendor:unknown: needs_review}
```

**Fourth case (same unknown vendor):**
```
Document: Same unknown vendor, $5000
Agent memory check: "This vendor needs review"
Agent thinks: "Remember, this vendor was rejected before"
Agent decision: "human_review" (recommend review)
Human decision: "reject" ✗
Agent confidence in recommendation increases
```

---

## SAFETY RAILS (GUARDRAILS)

### What agents CAN'T do

**Rail 1: Amount Limits**
```
IF extracted_amount > $100,000
THEN agent MUST recommend "human_review"
     Agent CAN'T auto-approve
```

**Rail 2: New Vendors**
```
IF vendor NOT in database
THEN agent MUST recommend "human_review"
     First-time vendors need human approval
```

**Rail 3: Anomaly Escalation**
```
IF anomaly_detected = true
THEN agent MUST flag for human
     Agent CAN'T suppress anomalies
```

**Rail 4: Override Integrity**
```
IF human_decision conflicts with agent_decision
THEN record discrepancy
     Agent learns from override
     Agent CAN'T change human decision retroactively
```

### Auto-Rollback Triggers

**If ANY of these happen:**
```
Condition 1: Agent accuracy < 85% (last 100 docs)
AND Condition 2: Human override rate > 30%
AND Condition 3: Issue is NOT known/seasonal

THEN:
  - Disable all agents
  - Revert to current system (spaCy + RF)
  - Alert ops team
  - Quarantine problematic agent
  - Human investigates
  - Once fixed, re-enable with updated knowledge
```

**Example:**
```
Day 1-10:  Agent accuracy = 92% ✓
Day 11:    New vendor type enters (niche industry)
Day 12:    Agent sees 20 invoices, gets 14 wrong (70%) ✗
Trigger:   Accuracy < 85% AND override rate = 65% > 30%
Action:    DISABLE AGENTS
           Revert to current system
           Human reviews new vendor type
           Agent learns patterns
           Re-enable tomorrow
```

---

## MONITORING & OBSERVABILITY

### Real-time Metrics (Dashboard)

```
┌─ AGENT STATUS ─────────────────────┐
│                                     │
│ Agents: ENABLED ✓                   │
│ Last check: 30 seconds ago          │
│ Next check: in 30 seconds           │
│                                     │
├─ ACCURACY ─────────────────────────┤
│ Last 100 docs: 91% ✓                │
│ Last 1000 docs: 89%                 │
│ Trend: ↑ improving                  │
│                                     │
├─ HUMAN INTERACTION ────────────────┤
│ Override rate: 22%                  │
│ Approve rate: 71%                   │
│ Review rate: 7%                     │
│                                     │
├─ AGENT DECISIONS ──────────────────┤
│ Auto-approved (today): 145          │
│ Flagged for review (today): 28      │
│ Rejected (today): 12                │
│                                     │
├─ ERRORS ──────────────────────────┤
│ Last error: None                    │
│ Error rate: 0.1%                    │
│                                     │
├─ MEMORY ──────────────────────────┤
│ Redis keys: 45,230                  │
│ Patterns learned: 382               │
│ Expiring soon: 12                   │
│                                     │
└─────────────────────────────────────┘

[DISABLE AGENTS] [VIEW LOGS] [VIEW PATTERNS]
```

### Logging & Debugging

Every agent interaction logged:
```
{
  timestamp: "2024-02-15T10:23:45Z",
  document_id: "doc_123",
  agent: "master",
  action: "process_document",
  reasoning: "Document is invoice, extracting fields",
  sub_agents_called: ["classifier", "extractor", "validator", "router"],
  final_decision: "auto_approve",
  confidence: 0.89,
  human_override: false,
  human_feedback: null,
  duration_ms: 245
}
```

---

## DEPLOYMENT TIMELINE (4 PHASES)

### Phase 1: Shadow Mode (Week 1-2)
```
Setup:
- Deploy agent service (separate container)
- Connect to FastAPI + PostgreSQL
- NO traffic sent to agents yet

Testing:
- Process 100 documents with agents
- Compare agent decisions vs current system
- Log all differences
- Fix bugs

Metrics:
- Agents process: 100 docs
- Accuracy: ??? (still testing)
- Human impact: NONE (shadow mode)
```

### Phase 2: Canary Rollout (Week 3-4)
```
Setup:
- Route 5% of NEW documents to agents
- Route 95% to current system

Monitoring:
- Agent accuracy (target: > 85%)
- Human override rate (target: < 30%)
- Processing time (target: < 1000ms)
- Error rate (target: 0%)

Metrics (target):
- Agents process: 50 docs/day
- Accuracy: 88%+
- Override rate: 25%
- Time: 500ms avg

Decision:
- If good: expand to Phase 3
- If bad: rollback to current system
```

### Phase 3: A/B Test (Week 5-6)
```
Setup:
- Route 50% of documents to agents
- Route 50% to current system

Measurement:
- Agent group: 500 docs
- Control group: 500 docs
- Compare: accuracy, speed, human satisfaction

Metrics (target):
- Agent accuracy > control accuracy
- Human override rate acceptable
- Processing speed acceptable

Statistical test:
- Is agent group significantly better?
- Chi-square test, p < 0.05

Decision:
- If significantly better: Phase 4
- If no improvement: stay at Phase 2
- If worse: rollback
```

### Phase 4: Full Rollout (Week 7+)
```
Setup:
- 100% of documents through agents
- Current system as fallback only

Maintenance:
- Monitor 24/7
- Auto-rollback if needed
- Collect feedback
- Continuous improvement

Success metrics:
- Accuracy: 94%+
- Override rate: 18-25%
- Auto-approve rate: 72%+
- Human time saved: 20%+
```

---

## FILE STRUCTURE (NEW AGENT SERVICE)

```
pfe-project/
├── app/                          # EXISTING (unchanged)
│   ├── main.py
│   ├── api/
│   ├── ml/
│   ├── dashboard/
│   └── ...
│
├── agents/                        # NEW (agent microservice)
│   ├── __init__.py
│   ├── main.py                   # Agent service entry point
│   ├── config.py                 # Agent config + LLM settings
│   ├── master_agent.py           # Master agent (orchestration)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── classifier.py         # Document classifier agent
│   │   ├── extractor.py          # Field extractor agent
│   │   ├── validator.py          # Validation agent
│   │   └── router.py             # Routing agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── ml_tools.py           # Call spaCy, RF, BART
│   │   ├── db_tools.py           # Query PostgreSQL
│   │   ├── memory_tools.py       # Redis operations
│   │   └── external_apis.py      # Vendor lookup, etc.
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py         # Current doc conversation
│   │   ├── long_term.py          # Learning across docs
│   │   └── redis_client.py       # Redis integration
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── accuracy_tracker.py   # Track decisions vs human
│   │   ├── safety_rails.py       # Enforce guardrails
│   │   ├── auto_rollback.py      # Auto-disable if issues
│   │   └── logging.py            # Detailed agent logs
│   └── api/
│       ├── __init__.py
│       └── extraction_agent.py   # FastAPI endpoint: POST /agents/extract
│
├── docker-compose.yml            # UPDATED (add agent service)
└── requirements.txt              # UPDATED (add LangChain, Llama deps)
```

---

## IMPLEMENTATION CHECKLIST

### Week 1-2: Setup
- [ ] Install LangChain, Llama2 model
- [ ] Set up separate agent service (docker container)
- [ ] Configure Redis + PostgreSQL for agents
- [ ] Create agent config file (models, prompts, settings)
- [ ] Scaffold 5 agent classes (master + 4 sub-agents)

### Week 3-4: Agent Implementation
- [ ] Implement Classifier Agent
- [ ] Implement Extractor Agent
- [ ] Implement Validator Agent
- [ ] Implement Router Agent
- [ ] Test each agent independently

### Week 5-6: Integration & Memory
- [ ] Connect agents to FastAPI
- [ ] Implement real-time learning (Redis)
- [ ] Implement long-term learning (Postgres aggregation)
- [ ] Build agent communication (master → sub-agents)
- [ ] Test full workflow end-to-end

### Week 7-8: Safety & Monitoring
- [ ] Implement safety rails
- [ ] Build monitoring dashboard
- [ ] Implement auto-rollback logic
- [ ] Add detailed logging
- [ ] Test failure scenarios

### Week 9-10: Deployment Phase 1 (Shadow)
- [ ] Deploy agent service
- [ ] Run shadow mode (100 docs)
- [ ] Compare agent vs current system
- [ ] Fix discrepancies
- [ ] Prepare for Phase 2

### Week 11-12: Deployment Phase 2-4
- [ ] Canary rollout (5%)
- [ ] A/B test (50%)
- [ ] Full rollout (100%)
- [ ] 24/7 monitoring
- [ ] Continuous improvement

---

## EXPECTED OUTCOMES

### Metrics Before Agents
```
Auto-approve rate: 65%
Human review rate: 28%
Rejection rate: 7%
Accuracy: 94%
Human override rate: N/A
Processing time: 4ms
```

### Metrics After Agents (Target)
```
Auto-approve rate: 72% (+7%)
Human review rate: 21% (-7%)
Rejection rate: 7%
Accuracy: 96% (+2%)
Human override rate: 22% (acceptable)
Processing time: 8ms (+4ms for reasoning)

Business impact:
- 1000 docs/day
- Time saved: 70 fewer human reviews/day
- Human time saved: ~3 hours/day
- Value: ~$150/day
```

---

## WHAT'S NOT CHANGING

✅ spaCy NER model (keep it)
✅ Random Forest confidence model (keep it)
✅ BART document classifier (keep it)
✅ File processing pipeline (keep it)
✅ PostgreSQL schema (mostly unchanged)
✅ FastAPI structure (kept, extended)
✅ Dashboard (updated to show agent reasoning)

---

## WHAT IS CHANGING

❌ Decision logic (hardcoded thresholds → agent reasoning)
❌ Routing (fixed destinations → intelligent routing)
❌ Human review workflow (approval bottleneck → flagged review)
✨ Agent reasoning layer (NEW)
✨ Real-time learning (NEW)
✨ Safety rails (NEW)
✨ Agent monitoring (NEW)

---

## RISKS & MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Agent makes wrong decision | Medium | High | Safety rails, human review, monitoring |
| Agent hallucinates reasoning | Medium | Low | Log all reasoning, human oversight |
| Model fails/crashes | Low | High | Fallback to current system, auto-rollback |
| Memory bloat (Redis fills up) | Low | Medium | TTL on Redis keys, cleanup script |
| Agents learn bad patterns | Low | High | Monitoring, human feedback, retraining |
| Performance degradation | Low | Medium | Parallel deployment, A/B test |

---

## SUCCESS CRITERIA

Project is **DONE** when:

✅ All 5 agents deployed and working
✅ Real-time learning functional (agents adapt to feedback)
✅ Safety rails enforced (no violations)
✅ Monitoring dashboard shows all metrics
✅ Auto-rollback triggered and verified to work
✅ Phase 4 rollout complete (100% document volume)
✅ Agent accuracy > 94%
✅ Human override rate < 25%
✅ Processing time < 1000ms per document
✅ Full documentation + runbooks
✅ Team trained on system

---

## NEXT STEPS

1. **Review this plan** with your team
2. **Confirm all 18 locked decisions** are what you want
3. **Start Week 1:** Set up agent infrastructure
4. **Build agent service** (parallel to existing operations)
5. **Run Phase 1 shadow mode** before any user impact

Ready to start building?
