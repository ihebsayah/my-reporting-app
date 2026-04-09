"""Streamlit dashboard entry point — Month 4 final version.

Layout
------
Sidebar:  source-directory picker, job-limit slider, retrain trigger button.
Row 1:    metric cards (docs, auto, review, reject, avg confidence, auto-rate from DB).
Row 2:    Field KPI table  |  Drift status + per-field avg confidence bar chart.
Row 3:    Source document preview  |  Recent async jobs table.
Page 4:   Extraction history table for a selected document.
"""

import logging

from app.dashboard.services import DashboardDataService, dashboard_source_exists

logger = logging.getLogger(__name__)


def main() -> None:
    """Render the Streamlit operator dashboard."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install with `pip install streamlit`."
        ) from exc

    st.set_page_config(
        page_title="PFE Extraction Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️  Controls")
    source_dir = st.sidebar.text_input(
        "Source document directory", value="docs/source_documents"
    )
    job_limit = st.sidebar.slider("Recent jobs to show", 1, 25, 10)

    st.sidebar.divider()
    st.sidebar.subheader("🔁 Retraining")
    n_estimators = st.sidebar.number_input(
        "RF n_estimators", min_value=50, max_value=500, value=200, step=50
    )
    run_retrain = st.sidebar.button("▶  Trigger monthly retraining", use_container_width=True)

    # ── Main title ─────────────────────────────────────────────────────────────
    st.title("📊 PFE Extraction & Confidence Dashboard")
    st.caption(
        "Live operator view — extraction quality, confidence drift, async jobs, "
        "and human-in-the-loop retraining."
    )

    # ── Source-dir guard ───────────────────────────────────────────────────────
    if not dashboard_source_exists(source_dir):
        st.error(f"⛔ Source directory not found: `{source_dir}`")
        st.info("Set the correct path in the sidebar or create `docs/source_documents/`.")
        return

    service = DashboardDataService()

    # ── Retrain action ─────────────────────────────────────────────────────────
    if run_retrain:
        with st.spinner("Running monthly RF retraining…"):
            retrain_msg = service.trigger_retraining(n_estimators=int(n_estimators))
        if retrain_msg.get("success"):
            st.success(
                f"✅ Retraining complete — new model: **{retrain_msg['model_version']}**  "
                f"(accuracy {retrain_msg['accuracy']:.2%}, "
                f"{retrain_msg['total_records']} records)"
            )
        else:
            st.error(f"❌ Retraining failed: {retrain_msg.get('error')}")

    # ── Fetch dashboard data ───────────────────────────────────────────────────
    data = service.build_dashboard_data(input_dir=source_dir, job_limit=job_limit)
    storage_kpi = service.build_storage_kpi()
    drift_report = service.build_drift_report()
    previews = service.load_document_preview(input_dir=source_dir)

    # ── Row 1: Metric cards ───────────────────────────────────────────────────
    st.subheader("📈 Pipeline Summary")
    cols = st.columns(len(data.metric_cards) + 2)
    for col, card in zip(cols, data.metric_cards):
        col.metric(card.label, card.value)

    # Storage KPI cards (from DB).
    if storage_kpi.total_documents > 0:
        cols[-2].metric(
            "Auto-rate (DB)",
            f"{storage_kpi.auto_rate:.1%}",
            delta=None,
        )
        cols[-1].metric(
            "DB Records",
            str(storage_kpi.total_documents),
        )

    # ── Row 2a: drift alert banner ────────────────────────────────────────────
    if drift_report is not None:
        if drift_report.drift_detected:
            st.error(
                f"🚨 **Confidence drift detected!**  "
                f"Signals: `{', '.join(drift_report.triggered_signals)}`  |  "
                f"Auto-rate drop: **{drift_report.auto_rate_drop:.1%}**  |  "
                f"Confidence drop: **{drift_report.confidence_drop:.3f}**  |  "
                f"Checked: {drift_report.checked_at[:19]}"
            )
        else:
            st.success(
                f"✅ No drift detected (auto-rate drop: {drift_report.auto_rate_drop:.1%}, "
                f"conf drop: {drift_report.confidence_drop:.3f})"
            )

    st.divider()

    # ── Row 2: Field KPI table | Per-field confidence chart ──────────────────
    left_col, right_col = st.columns([1.2, 1.0])

    with left_col:
        st.subheader("🗂 Field KPI Summary")
        st.dataframe(
            [
                {
                    "Field": item.field_name,
                    "Total": item.total_occurrences,
                    "Auto ✅": item.auto_count,
                    "Review 🔍": item.review_count,
                    "Reject ❌": item.reject_count,
                    "Avg Conf": round(item.average_confidence, 3),
                }
                for item in data.field_kpis
            ],
            use_container_width=True,
            hide_index=True,
        )

    with right_col:
        st.subheader("🎯 Avg Confidence by Field (DB)")
        if storage_kpi.average_confidence_by_field:
            try:
                import pandas as pd
                conf_df = pd.DataFrame(
                    [
                        {"Field": k, "Avg Confidence": v}
                        for k, v in sorted(
                            storage_kpi.average_confidence_by_field.items()
                        )
                    ]
                ).set_index("Field")
                st.bar_chart(conf_df, use_container_width=True)
            except ImportError:
                st.json(storage_kpi.average_confidence_by_field)
        else:
            st.info("No DB records yet — run the pipeline via the API to populate this chart.")

    st.divider()

    # ── Row 3: Document preview | Recent async jobs ───────────────────────────
    bottom_left, bottom_right = st.columns([1.2, 1.0])

    with bottom_left:
        st.subheader("📄 Source Document Preview")
        st.dataframe(
            [
                {
                    "Document ID": p["document_id"],
                    "Decision": p["overall_decision"],
                    "Preview": p["preview"],
                }
                for p in previews
            ],
            use_container_width=True,
            hide_index=True,
        )

    with bottom_right:
        st.subheader("🔄 Recent Async Jobs")
        st.dataframe(
            [
                {
                    "Job ID": j.job_id[:8] + "…",
                    "Status": j.status,
                    "Submitted": j.submitted_at[:19],
                    "Completed": (j.completed_at or "-")[:19],
                    "Error": j.error_message or "-",
                }
                for j in data.recent_jobs
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── Row 4: Extraction history search ─────────────────────────────────────
    st.subheader("🔍 Extraction History by Document")
    doc_search = st.text_input(
        "Enter document ID to view extraction history", placeholder="e.g. doc-001"
    )
    if doc_search:
        history = service.get_extraction_history(doc_search, limit=10)
        if history:
            for rec in history:
                with st.expander(
                    f"Run {rec['record_id']} — {rec['overall_decision'].upper()} "
                    f"(scorer: {rec['scorer']}) — {rec['processed_at'][:19]}"
                ):
                    st.dataframe(
                        [
                            {
                                "Field": f["field_name"],
                                "Value": f["value"] or "-",
                                "Confidence": round(f["confidence"], 3),
                                "Decision": f["decision"],
                                "Sources": ", ".join(f["sources"]),
                                "Scorer": f["scorer"],
                            }
                            for f in rec["fields"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.info(f"No extraction history found for document `{doc_search}`.")


if __name__ == "__main__":
    main()
