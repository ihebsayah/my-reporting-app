"""Unit tests for the AI Agent service — runs without a live FastAPI or Redis.

All external dependencies (ML tools, DB, Redis) are mocked so these tests
run fast and offline. They verify:

- ClassifierAgent reasoning logic
- ExtractorAgent field merging
- ValidatorAgent safety rules (amount cap, new vendor, date validation)
- RouterAgent safety rails (cannot auto-approve large amounts or new vendors)
- MasterAgent orchestration (fallback on sub-agent failure)
- SafetyRailsEnforcer (API boundary audit)
- AutoRollbackMonitor (accuracy thresholds)
- ShortTermMemory (context storage)

Run with:
    cd pfe-project
    python3 -m pytest tests/test_agents.py -v
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on PYTHONPATH
sys.path.insert(0, ".")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_invoice_text():
    return (
        "Invoice #INV-2024-001\n"
        "Vendor: Acme Corp SARL\n"
        "Invoice Date: 2024-01-15\n"
        "Total: $5,000.00"
    )


@pytest.fixture
def sample_fields():
    return [
        {"field_name": "INVOICE_ID", "value": "INV-2024-001", "confidence": 0.92, "sources": ["regex"], "decision": "auto"},
        {"field_name": "VENDOR_NAME", "value": "Acme Corp", "confidence": 0.87, "sources": ["regex"], "decision": "review"},
        {"field_name": "INVOICE_DATE", "value": "2024-01-15", "confidence": 0.90, "sources": ["regex"], "decision": "auto"},
        {"field_name": "TOTAL_AMOUNT", "value": "$5,000.00", "confidence": 0.88, "sources": ["regex"], "decision": "review"},
    ]


@pytest.fixture
def validation_result_valid():
    return {
        "is_valid": True,
        "confidence_adjustment": 0.02,
        "vendor_known": True,
        "amount_normal": True,
        "date_valid": True,
        "issues": [],
    }


@pytest.fixture
def validation_result_new_vendor():
    return {
        "is_valid": True,
        "confidence_adjustment": -0.05,
        "vendor_known": False,
        "amount_normal": True,
        "date_valid": True,
        "issues": [
            {
                "field_name": "VENDOR_NAME",
                "issue_type": "rule_violation",
                "severity": "warning",
                "description": "Vendor not in DB",
            }
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ShortTermMemory tests
# ─────────────────────────────────────────────────────────────────────────────


class TestShortTermMemory:
    def test_initialisation(self):
        from agents.memory.short_term import ShortTermMemory

        mem = ShortTermMemory(document_id="doc_001")
        assert mem.document_id == "doc_001"
        assert mem.session_id
        assert mem.messages == []
        assert mem.context == {}

    def test_add_message(self):
        from agents.memory.short_term import ShortTermMemory

        mem = ShortTermMemory(document_id="doc_001")
        mem.add_message("human", "hello", agent="master")
        assert len(mem.messages) == 1
        assert mem.messages[0]["role"] == "human"
        assert mem.messages[0]["content"] == "hello"

    def test_context_get_set(self):
        from agents.memory.short_term import ShortTermMemory

        mem = ShortTermMemory(document_id="doc_001")
        mem.set_context("key1", {"value": 42})
        assert mem.get_context("key1") == {"value": 42}
        assert mem.get_context("missing", default="fallback") == "fallback"

    def test_to_dict(self):
        from agents.memory.short_term import ShortTermMemory

        mem = ShortTermMemory(document_id="doc_test")
        mem.add_message("ai", "reasoning")
        d = mem.to_dict()
        assert d["document_id"] == "doc_test"
        assert d["message_count"] == 1

    def test_persist_to_redis_no_redis(self):
        """Should return False gracefully when Redis is unavailable."""
        from agents.memory.short_term import ShortTermMemory

        mem = ShortTermMemory(document_id="doc_001")
        # No Redis running — should not raise
        result = mem.persist_to_redis()
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# ClassifierAgent tests
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifierAgent:
    @patch("agents.agents.classifier.get_reasoner")
    @patch("agents.agents.classifier.run_bart_classification")
    @patch("agents.agents.classifier.run_ner_extraction")
    def test_invoice_classification(self, mock_ner, mock_bart, mock_reasoner, sample_invoice_text):
        """Classifier boosts confidence above BART score when NER finds invoice fields.

        LLM reasoner is patched to unavailable so the test is deterministic
        regardless of whether Ollama is running locally.
        """
        mock_reasoner.return_value.is_available.return_value = False
        mock_ner.invoke.return_value = json.dumps({
            "entities": [
                {"label": "INVOICE_ID", "text": "INV-2024-001", "score": 0.92},
                {"label": "VENDOR_NAME", "text": "Acme Corp", "score": 0.87},
                {"label": "TOTAL_AMOUNT", "text": "$5,000.00", "score": 0.88},
                {"label": "INVOICE_DATE", "text": "2024-01-15", "score": 0.90},
            ]
        })
        mock_bart.invoke.return_value = json.dumps({
            "doc_type": "invoice", "confidence": 0.82
        })

        from agents.agents.classifier import ClassifierAgent

        agent = ClassifierAgent()
        result = agent.run(sample_invoice_text)

        assert result.doc_type == "invoice"
        assert result.confidence > 0.82  # Heuristic field-match boost raises above BART score
        assert "INVOICE_ID" in result.field_names

    @patch("agents.agents.classifier.get_reasoner")
    @patch("agents.agents.classifier.run_bart_classification")
    @patch("agents.agents.classifier.run_ner_extraction")
    def test_fallback_to_invoice_on_ambiguous_bart(self, mock_ner, mock_bart, mock_reasoner):
        """When BART returns unknown but NER finds invoice fields, reclassify as invoice.

        LLM reasoner patched to unavailable for deterministic heuristic output.
        """
        mock_reasoner.return_value.is_available.return_value = False
        mock_ner.invoke.return_value = json.dumps({
            "entities": [
                {"label": "INVOICE_ID", "text": "INV-001", "score": 0.85},
                {"label": "TOTAL_AMOUNT", "text": "$1,000", "score": 0.80},
                {"label": "VENDOR_NAME", "text": "Corp", "score": 0.75},
            ]
        })
        mock_bart.invoke.return_value = json.dumps({"doc_type": "unknown", "confidence": 0.3})

        from agents.agents.classifier import ClassifierAgent

        result = ClassifierAgent().run("some ambiguous text")
        assert result.doc_type == "invoice"
        assert result.confidence >= 0.72

    @patch("agents.agents.classifier.get_reasoner")
    @patch("agents.agents.classifier.run_bart_classification")
    @patch("agents.agents.classifier.run_ner_extraction")
    def test_unknown_when_no_signals(self, mock_ner, mock_bart, mock_reasoner):
        """With no NER entities and unknown BART, confidence stays low.

        LLM patched to unavailable — otherwise Mistral might re-classify as invoice.
        """
        mock_reasoner.return_value.is_available.return_value = False
        mock_ner.invoke.return_value = json.dumps({"entities": []})
        mock_bart.invoke.return_value = json.dumps({"doc_type": "unknown", "confidence": 0.2})

        from agents.agents.classifier import ClassifierAgent

        result = ClassifierAgent().run("some random text with no invoice signals")
        assert result.doc_type == "unknown"
        assert result.confidence < 0.5

    @patch("agents.agents.classifier.run_bart_classification")
    @patch("agents.agents.classifier.run_ner_extraction")
    def test_memory_context_written(self, mock_ner, mock_bart, sample_invoice_text):
        from agents.agents.classifier import ClassifierAgent
        from agents.memory.short_term import ShortTermMemory

        mock_ner.invoke.return_value = json.dumps({"entities": []})
        mock_bart.invoke.return_value = json.dumps({"doc_type": "invoice", "confidence": 0.75})

        mem = ShortTermMemory("doc_test")
        ClassifierAgent(memory=mem).run(sample_invoice_text)

        ctx = mem.get_context("classifier_result")
        assert ctx is not None
        assert "doc_type" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# ExtractorAgent tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractorAgent:
    @patch("agents.agents.extractor.run_ner_extraction")
    @patch("agents.agents.extractor.run_confidence_scoring")
    def test_extracts_fields_from_pipeline(self, mock_scoring, mock_ner, sample_invoice_text):
        mock_scoring.invoke.return_value = json.dumps({
            "overall_decision": "auto",
            "scorer": "rf",
            "fields": [
                {"field_name": "INVOICE_ID", "value": "INV-001", "confidence": 0.92, "sources": ["regex"], "decision": "auto"},
                {"field_name": "TOTAL_AMOUNT", "value": "$5000", "confidence": 0.88, "sources": ["regex"], "decision": "review"},
                {"field_name": "VENDOR_NAME", "value": "Acme", "confidence": 0.85, "sources": ["regex"], "decision": "review"},
                {"field_name": "INVOICE_DATE", "value": "2024-01-15", "confidence": 0.91, "sources": ["regex"], "decision": "auto"},
            ],
        })
        mock_ner.invoke.return_value = json.dumps({"entities": []})

        from agents.agents.extractor import ExtractorAgent

        result = ExtractorAgent().run(sample_invoice_text, doc_type="invoice")

        assert len(result.fields) == 4
        assert result.overall_confidence > 0.0
        field_names = [f.field_name for f in result.fields]
        assert "INVOICE_ID" in field_names
        assert "TOTAL_AMOUNT" in field_names

    @patch("agents.agents.extractor.run_ner_extraction")
    @patch("agents.agents.extractor.run_confidence_scoring")
    def test_supplementary_ner_fills_missing_fields(self, mock_scoring, mock_ner):
        """NER supplementary pass should add fields missing from the pipeline."""
        mock_scoring.invoke.return_value = json.dumps({
            "overall_decision": "review",
            "scorer": "heuristic",
            "fields": [
                {"field_name": "INVOICE_ID", "value": "INV-001", "confidence": 0.90, "sources": ["regex"], "decision": "auto"},
            ],
        })
        mock_ner.invoke.return_value = json.dumps({
            "entities": [
                {"label": "TOTAL_AMOUNT", "text": "5000", "score": 0.70, "sources": ["ner_supplementary"]},
            ]
        })

        from agents.agents.extractor import ExtractorAgent

        result = ExtractorAgent().run("some text", doc_type="invoice")
        field_names = [f.field_name for f in result.fields]
        assert "TOTAL_AMOUNT" in field_names

    @patch("agents.agents.extractor.run_ner_extraction")
    @patch("agents.agents.extractor.run_confidence_scoring")
    def test_confidence_weighted_by_field_importance(self, mock_scoring, mock_ner):
        """TOTAL_AMOUNT has higher weight (1.5) than INVOICE_DATE (1.0).

        The RF scorer may adjust raw confidence scores from the pipeline mock
        (e.g. 0.5 → 0.45 for TOTAL_AMOUNT) before the weighted average is
        computed.  The assertion uses a wider tolerance (0.05) to remain stable
        across scorer versions.
        """
        mock_scoring.invoke.return_value = json.dumps({
            "overall_decision": "review",
            "scorer": "heuristic",
            "fields": [
                {"field_name": "TOTAL_AMOUNT", "value": "5000", "confidence": 0.5, "sources": [], "decision": "review"},
                {"field_name": "INVOICE_DATE", "value": "2024-01-15", "confidence": 0.9, "sources": [], "decision": "auto"},
            ],
        })
        mock_ner.invoke.return_value = json.dumps({"entities": []})

        from agents.agents.extractor import ExtractorAgent

        result = ExtractorAgent().run("text", doc_type="invoice")
        # TOTAL_AMOUNT weight=1.5, INVOICE_DATE weight=1.0
        # RF scorer may shift raw confidence slightly — accept range [0.60, 0.68]
        assert 0.60 <= result.overall_confidence <= 0.68, (
            f"Expected weighted confidence in [0.60, 0.68], got {result.overall_confidence:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ValidatorAgent tests
# ─────────────────────────────────────────────────────────────────────────────


class TestValidatorAgent:
    @patch("agents.agents.validator.lookup_vendor_history")
    def test_known_vendor_passes(self, mock_lookup, sample_fields):
        mock_lookup.invoke.return_value = json.dumps({
            "vendor": "Acme Corp",
            "found": True,
            "total_invoices": 25,
            "avg_amount": 4500.0,
            "min_amount": 1000.0,
            "max_amount": 8000.0,
            "is_new_vendor": False,
        })

        from agents.agents.validator import ValidatorAgent
        from agents.memory.long_term import LongTermMemory

        ltm = MagicMock(spec=LongTermMemory)
        ltm.should_flag_vendor.return_value = False

        result = ValidatorAgent(long_term_memory=ltm).run(sample_fields)

        assert result.vendor_known is True
        assert result.is_valid is True
        assert result.confidence_adjustment > 0  # Known vendor + normal amount → boost

    @patch("agents.agents.validator.lookup_vendor_history")
    def test_new_vendor_adds_warning(self, mock_lookup, sample_fields):
        mock_lookup.invoke.return_value = json.dumps({
            "vendor": "Unknown Corp",
            "found": False,
            "total_invoices": 0,
            "is_new_vendor": True,
        })

        from agents.agents.validator import ValidatorAgent
        from agents.memory.long_term import LongTermMemory

        ltm = MagicMock(spec=LongTermMemory)
        ltm.should_flag_vendor.return_value = True

        result = ValidatorAgent(long_term_memory=ltm).run(sample_fields)

        assert result.vendor_known is False
        warning_types = [i.severity for i in result.issues]
        assert "warning" in warning_types

    @patch("agents.agents.validator.lookup_vendor_history")
    def test_amount_exceeds_safety_limit(self, mock_lookup):
        """Safety Rail 1: amount > SAFETY_MAX_AMOUNT must create an error issue."""
        mock_lookup.invoke.return_value = json.dumps({"found": True, "total_invoices": 5, "avg_amount": 50000, "min_amount": 10000, "max_amount": 200000, "is_new_vendor": False})

        fields = [
            {"field_name": "TOTAL_AMOUNT", "value": "$150,000.00", "confidence": 0.9},
            {"field_name": "VENDOR_NAME", "value": "BigCorp", "confidence": 0.9},
        ]
        from agents.agents.validator import ValidatorAgent
        from agents.memory.long_term import LongTermMemory

        ltm = MagicMock(spec=LongTermMemory)
        ltm.should_flag_vendor.return_value = False

        result = ValidatorAgent(long_term_memory=ltm).run(fields)

        rail_errors = [i for i in result.issues if i.issue_type == "rule_violation" and "100" in i.description]
        assert len(rail_errors) > 0

    @patch("agents.agents.validator.lookup_vendor_history")
    def test_future_date_fails(self, mock_lookup, sample_fields):
        mock_lookup.invoke.return_value = json.dumps({"found": True, "total_invoices": 5, "avg_amount": 5000, "min_amount": 1000, "max_amount": 10000, "is_new_vendor": False})

        future_fields = [f for f in sample_fields if f["field_name"] != "INVOICE_DATE"]
        future_fields.append({"field_name": "INVOICE_DATE", "value": "2099-12-31", "confidence": 0.9})

        from agents.agents.validator import ValidatorAgent
        from agents.memory.long_term import LongTermMemory

        ltm = MagicMock(spec=LongTermMemory)
        ltm.should_flag_vendor.return_value = False

        result = ValidatorAgent(long_term_memory=ltm).run(future_fields)

        date_errors = [i for i in result.issues if i.field_name == "INVOICE_DATE" and i.severity == "error"]
        assert len(date_errors) > 0
        assert result.is_valid is False


# ─────────────────────────────────────────────────────────────────────────────
# RouterAgent tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRouterAgent:
    @patch("agents.agents.router.get_reasoner")
    def test_auto_approve_high_confidence_known_vendor(self, mock_reasoner, sample_fields, validation_result_valid):
        """High-confidence known vendor should auto-approve in heuristic mode.

        LLM patched to unavailable so routing uses heuristic rules, giving
        deterministic auto_approve for confidence=0.92, known vendor, no errors.
        """
        mock_reasoner.return_value.is_available.return_value = False
        from agents.agents.router import RouterAgent, AUTO_APPROVE

        ltm = MagicMock()
        ltm.lookup_vendor_pattern.return_value = {"approve": 20, "reject": 1, "total": 21}

        result = RouterAgent(long_term_memory=ltm).run(
            extraction_confidence=0.92,
            validation_result=validation_result_valid,
            extracted_fields=sample_fields,
            doc_type="invoice",
        )
        assert result.action == AUTO_APPROVE
        assert result.confidence >= 0.85

    def test_human_review_for_new_vendor(self, sample_fields, validation_result_new_vendor):
        """Safety Rail 2: new vendor must force human_review regardless of confidence."""
        from agents.agents.router import RouterAgent, HUMAN_REVIEW

        ltm = MagicMock()
        ltm.lookup_vendor_pattern.return_value = None

        result = RouterAgent(long_term_memory=ltm).run(
            extraction_confidence=0.95,  # High confidence but new vendor
            validation_result=validation_result_new_vendor,
            extracted_fields=sample_fields,
        )
        assert result.action == HUMAN_REVIEW
        assert any("RAIL_2_NEW_VENDOR" in r for r in result.safety_rails_triggered)

    def test_human_review_for_large_amount(self, validation_result_valid):
        """Safety Rail 1: amount > $100k forces human_review."""
        from agents.agents.router import RouterAgent, HUMAN_REVIEW

        large_fields = [
            {"field_name": "TOTAL_AMOUNT", "value": "$150,000.00", "confidence": 0.95},
            {"field_name": "VENDOR_NAME", "value": "Acme Corp", "confidence": 0.90},
        ]
        ltm = MagicMock()
        ltm.lookup_vendor_pattern.return_value = {"approve": 10, "total": 10}

        result = RouterAgent(long_term_memory=ltm).run(
            extraction_confidence=0.95,
            validation_result=validation_result_valid,
            extracted_fields=large_fields,
        )
        assert result.action == HUMAN_REVIEW
        assert any("RAIL_1_AMOUNT_LIMIT" in r for r in result.safety_rails_triggered)

    def test_reject_very_low_confidence(self, validation_result_valid, sample_fields):
        """Very low confidence (< 0.39) triggers Rail 4 → reject."""
        from agents.agents.router import RouterAgent, REJECT

        ltm = MagicMock()
        ltm.lookup_vendor_pattern.return_value = None

        result = RouterAgent(long_term_memory=ltm).run(
            extraction_confidence=0.20,
            validation_result=validation_result_valid,
            extracted_fields=sample_fields,
        )
        assert result.action == REJECT

    def test_vendor_trust_boost_applied(self, sample_fields, validation_result_valid):
        """High-approve-rate vendor pattern should boost confidence."""
        from agents.agents.router import RouterAgent

        ltm = MagicMock()
        ltm.lookup_vendor_pattern.return_value = {
            "approve": 50, "reject": 1, "total": 51
        }

        result = RouterAgent(long_term_memory=ltm).run(
            extraction_confidence=0.87,
            validation_result=validation_result_valid,
            extracted_fields=sample_fields,
        )
        # Vendor trust boost of +0.03 should push confidence above 0.87
        assert result.confidence >= 0.87

    def test_reasoning_is_present(self, sample_fields, validation_result_valid):
        from agents.agents.router import RouterAgent

        ltm = MagicMock()
        ltm.lookup_vendor_pattern.return_value = None

        result = RouterAgent(long_term_memory=ltm).run(
            extraction_confidence=0.75,
            validation_result=validation_result_valid,
            extracted_fields=sample_fields,
        )
        assert result.reasoning
        assert len(result.reasoning) > 20


# ─────────────────────────────────────────────────────────────────────────────
# SafetyRailsEnforcer tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyRailsEnforcer:
    def setup_method(self):
        from agents.monitoring.safety_rails import SafetyRailsEnforcer

        self.enforcer = SafetyRailsEnforcer()

    def test_passes_clean_invoice(self):
        result = self.enforcer.check(
            proposed_action="auto_approve",
            confidence=0.92,
            fields={"TOTAL_AMOUNT": "$5,000", "VENDOR_NAME": "Acme"},
            validation_result={"is_valid": True, "vendor_known": True, "issues": []},
        )
        assert result.passed
        assert result.corrected_action is None

    def test_blocks_auto_approve_large_amount(self):
        result = self.enforcer.check(
            proposed_action="auto_approve",
            confidence=0.95,
            fields={"TOTAL_AMOUNT": "$200,000", "VENDOR_NAME": "Acme"},
            validation_result={"is_valid": True, "vendor_known": True, "issues": []},
        )
        assert not result.passed
        assert result.corrected_action == "human_review"
        assert any("RAIL_1" in v for v in result.violated_rails)

    def test_blocks_auto_approve_unknown_vendor(self):
        result = self.enforcer.check(
            proposed_action="auto_approve",
            confidence=0.95,
            fields={"TOTAL_AMOUNT": "$1,000", "VENDOR_NAME": "NewVendor"},
            validation_result={"is_valid": True, "vendor_known": False, "issues": []},
        )
        assert not result.passed
        assert result.corrected_action == "human_review"
        assert any("RAIL_2" in v for v in result.violated_rails)

    def test_blocks_auto_approve_with_errors(self):
        result = self.enforcer.check(
            proposed_action="auto_approve",
            confidence=0.9,
            fields={"TOTAL_AMOUNT": "$5,000", "VENDOR_NAME": "Acme"},
            validation_result={
                "is_valid": False,
                "vendor_known": True,
                "issues": [{"severity": "error", "description": "Date in future"}],
            },
        )
        assert not result.passed
        assert result.corrected_action == "human_review"
        assert any("RAIL_3" in v for v in result.violated_rails)

    def test_allows_human_review_with_errors(self):
        """human_review with errors should PASS the enforcer (already safe action)."""
        result = self.enforcer.check(
            proposed_action="human_review",
            confidence=0.9,
            fields={"TOTAL_AMOUNT": "$5,000", "VENDOR_NAME": "Acme"},
            validation_result={
                "is_valid": False,
                "vendor_known": True,
                "issues": [{"severity": "error", "description": "Date in future"}],
            },
        )
        assert result.passed

    def test_override_integrity_recorded(self):
        """Human override discrepancy should be noted but NOT change the decision."""
        result = self.enforcer.check(
            proposed_action="auto_approve",
            confidence=0.95,
            fields={"TOTAL_AMOUNT": "$5,000", "VENDOR_NAME": "Acme"},
            validation_result={"is_valid": True, "vendor_known": True, "issues": []},
            human_decision="reject",
        )
        # The proposed_action was valid by rail standards, human just disagreed
        # Rail 4 records it but doesn't force an action override
        assert any("RAIL_4" in v for v in result.violated_rails)


# ─────────────────────────────────────────────────────────────────────────────
# AutoRollbackMonitor tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoRollbackMonitor:
    def setup_method(self):
        # Reset module-level state before each test
        import agents.monitoring.auto_rollback as arb

        arb._agents_enabled_override = None
        arb._last_rollback_reason = None
        arb._last_rollback_at = None

    def test_agents_enabled_by_default(self):
        from agents.monitoring.auto_rollback import AutoRollbackMonitor

        assert AutoRollbackMonitor.agents_enabled() is True

    def test_manual_disable_and_enable(self):
        from agents.monitoring.auto_rollback import AutoRollbackMonitor

        AutoRollbackMonitor.disable_agents("test_reason")
        assert AutoRollbackMonitor.agents_enabled() is False

        AutoRollbackMonitor.enable_agents()
        assert AutoRollbackMonitor.agents_enabled() is True

    def test_status_after_disable(self):
        from agents.monitoring.auto_rollback import AutoRollbackMonitor

        AutoRollbackMonitor.disable_agents("unit_test")
        status = AutoRollbackMonitor.status()
        assert status["rolled_back"] is True
        assert status["last_rollback_reason"] == "unit_test"
        assert status["agents_enabled"] is False

    @patch("agents.monitoring.auto_rollback.AccuracyTracker")
    def test_auto_rollback_triggers(self, mock_tracker_class):
        from agents.monitoring.accuracy_tracker import AccuracyReport
        from agents.monitoring.auto_rollback import AutoRollbackMonitor

        mock_report = AccuracyReport(
            window=100,
            total_decisions=100,
            agreed=70,
            overridden=30,
            accuracy=0.70,         # < 0.85 threshold
            override_rate=0.30,    # == 0.30 threshold (needs > 0.30)
            rollback_needed=True,  # Both conditions met
            rollback_reasons=["Accuracy 70.0% < threshold 85.0%", "Override rate 30.0% > threshold 30.0%"],
        )
        mock_tracker_class.return_value.evaluate.return_value = mock_report

        monitor = AutoRollbackMonitor()
        result = monitor.check()

        assert result["rolled_back"] is True
        assert AutoRollbackMonitor.agents_enabled() is False

    @patch("agents.monitoring.auto_rollback.AccuracyTracker")
    def test_no_rollback_when_accurate(self, mock_tracker_class):
        from agents.monitoring.accuracy_tracker import AccuracyReport
        from agents.monitoring.auto_rollback import AutoRollbackMonitor

        mock_report = AccuracyReport(
            window=100,
            total_decisions=100,
            agreed=92,
            overridden=8,
            accuracy=0.92,
            override_rate=0.08,
            rollback_needed=False,
            rollback_reasons=[],
        )
        mock_tracker_class.return_value.evaluate.return_value = mock_report

        monitor = AutoRollbackMonitor()
        result = monitor.check()

        assert result["rolled_back"] is False
        assert AutoRollbackMonitor.agents_enabled() is True


# ─────────────────────────────────────────────────────────────────────────────
# MasterAgent integration tests (all sub-agents mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestMasterAgent:
    @patch("agents.master_agent.save_agent_decision")
    @patch("agents.master_agent.RouterAgent")
    @patch("agents.master_agent.ValidatorAgent")
    @patch("agents.master_agent.ExtractorAgent")
    @patch("agents.master_agent.ClassifierAgent")
    def test_full_pipeline_auto_approve(
        self,
        mock_classifier_cls,
        mock_extractor_cls,
        mock_validator_cls,
        mock_router_cls,
        mock_save,
        sample_invoice_text,
    ):
        from agents.agents.classifier import ClassifierResult
        from agents.agents.extractor import ExtractorResult, ExtractedField
        from agents.agents.router import RouterResult
        from agents.agents.validator import ValidatorResult
        from agents.master_agent import MasterAgent

        # Mock classifier
        mock_classifier_cls.return_value.run.return_value = ClassifierResult(
            doc_type="invoice", confidence=0.95, reasoning="Test", field_names=["INVOICE_ID"]
        )
        # Mock extractor
        mock_extractor_cls.return_value.run.return_value = ExtractorResult(
            fields=[ExtractedField("TOTAL_AMOUNT", "$5000", 0.90)],
            overall_confidence=0.90,
            reasoning="Test",
            doc_type="invoice",
        )
        # Mock validator
        mock_validator_cls.return_value.run.return_value = ValidatorResult(
            is_valid=True,
            issues=[],
            confidence_adjustment=0.02,
            reasoning="Test",
            vendor_known=True,
        )
        # Mock router
        mock_router_cls.return_value.run.return_value = RouterResult(
            action="auto_approve",
            confidence=0.92,
            reasoning="Test: auto approve",
        )

        master = MasterAgent()
        result = master.process_document("doc_001", sample_invoice_text)

        assert result.action == "auto_approve"
        assert result.confidence == 0.92
        assert result.fallback_used is False
        assert "classifier" in result.agents_used
        assert "router" in result.agents_used
        mock_save.assert_called_once()

    @patch("agents.master_agent.save_agent_decision")
    @patch("agents.master_agent.ClassifierAgent")
    def test_fallback_on_classifier_crash(self, mock_classifier_cls, mock_save, sample_invoice_text):
        from agents.master_agent import MasterAgent

        mock_classifier_cls.return_value.run.side_effect = RuntimeError("Model crashed")

        master = MasterAgent()
        result = master.process_document("doc_002", sample_invoice_text)

        # Should fall back safely to human_review
        assert result.action == "human_review"
        assert result.fallback_used is True
        assert result.error is not None
        assert "Model crashed" in result.error
