"""Tests for dashboard data services."""

from app.dashboard.services import DashboardDataService, dashboard_source_exists


def test_dashboard_data_service_builds_metric_cards_and_kpis() -> None:
    """Ensure the dashboard service builds reusable summary data."""
    service = DashboardDataService()

    data = service.build_dashboard_data(input_dir="docs/source_documents", job_limit=5)

    assert data.metric_cards
    assert any(card.label == "Documents" for card in data.metric_cards)
    assert data.field_kpis
    assert data.field_decisions


def test_dashboard_data_service_loads_document_previews() -> None:
    """Ensure the dashboard service returns source document previews."""
    service = DashboardDataService()

    previews = service.load_document_preview(input_dir="docs/source_documents")

    assert previews
    assert "document_id" in previews[0]
    assert "preview" in previews[0]
    assert "overall_decision" in previews[0]


def test_dashboard_source_exists_returns_expected_state(tmp_path) -> None:
    """Ensure dashboard source existence checks behave correctly."""
    assert dashboard_source_exists("docs/source_documents") is True
    assert dashboard_source_exists(str(tmp_path / "missing")) is False
