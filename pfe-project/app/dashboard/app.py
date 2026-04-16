"""Streamlit dashboard entry point — Agent-enriched version.

Layout
------
Sidebar:    source-directory picker, job-limit slider, retrain trigger.
Tab 1 (Pipeline):    metric cards, field KPI, drift banner, doc preview, async jobs,
                     extraction history.
Tab 2 (AI Agents):   agent status, accuracy metrics, live document tester,
                     feedback submission, admin enable/disable controls.
"""

import logging

from app.dashboard.agent_services import AgentDataService
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
        "AI agent decisions, and human-in-the-loop retraining."
    )

    # ── Tabs ────────────────────────────────────────────────────────────────────
    tab_pipeline, tab_agents = st.tabs(["🔬 Pipeline", "🤖 AI Agents"])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1 — existing pipeline dashboard (unchanged)
    # ════════════════════════════════════════════════════════════════════════════
    with tab_pipeline:
        if not dashboard_source_exists(source_dir):
            st.error(f"⛔ Source directory not found: `{source_dir}`")
            st.info("Set the correct path in the sidebar or create `docs/source_documents/`.")
        else:
            service = DashboardDataService()

            # Retrain action.
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

            data = service.build_dashboard_data(input_dir=source_dir, job_limit=job_limit)
            storage_kpi = service.build_storage_kpi()
            drift_report = service.build_drift_report()
            previews = service.load_document_preview(input_dir=source_dir)

            # Row 1 – metric cards.
            st.subheader("📈 Pipeline Summary")
            cols = st.columns(len(data.metric_cards) + 2)
            for col, card in zip(cols, data.metric_cards):
                col.metric(card.label, card.value)
            if storage_kpi.total_documents > 0:
                cols[-2].metric("Auto-rate (DB)", f"{storage_kpi.auto_rate:.1%}")
                cols[-1].metric("DB Records", str(storage_kpi.total_documents))

            # Drift alert.
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

            # Row 2 – field KPI | confidence chart.
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

            # Row 3 – document preview | async jobs.
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

            # Row 4 – extraction history.
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

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2 — AI Agent panel (NEW)
    # ════════════════════════════════════════════════════════════════════════════
    with tab_agents:
        agent_svc = AgentDataService()
        _render_agent_tab(st, agent_svc)


def _render_agent_tab(st, agent_svc: "AgentDataService") -> None:
    """Render the full AI Agent monitoring and control panel.

    Args:
        st: The streamlit module.
        agent_svc: Initialised AgentDataService instance.
    """
    st.subheader("🤖 AI Agent Service")

    # ── Service status banner ──────────────────────────────────────────────────
    status = agent_svc.get_status()

    if not status.reachable:
        st.warning(
            f"⚠️ Agent service is **not reachable**. "
            f"Set `AGENT_SERVICE_URL` and ensure the service is running on port 8001.  \n"
            f"Reason: `{status.last_rollback_reason}`"
        )
        st.info(
            "**Start locally:**  \n"
            "```bash\n"
            "AGENT_SERVICE_URL=http://localhost:8001 "
            "uvicorn agents.main:app --port 8001 --reload\n"
            "```"
        )
        return

    # Connected — show status row.
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Service", "🟢 Online")
    s_col2.metric("Agents Enabled", "✅ Yes" if status.agents_enabled else "❌ No")
    s_col3.metric("Auto-Rollback", "⚠️ Active" if status.rolled_back else "✅ Off")
    s_col4.metric("Version", status.version)

    if status.rolled_back:
        st.error(
            f"🚨 **Auto-rollback is active** — agents are DISABLED.  \n"
            f"Reason: `{status.last_rollback_reason}`  \n"
            "Investigate, then use the **Admin Controls** below to re-enable."
        )

    st.divider()

    # ── Accuracy monitoring ────────────────────────────────────────────────────
    st.subheader("📊 Agent Accuracy Monitor")
    accuracy = agent_svc.get_accuracy()

    if accuracy is None:
        st.info("Accuracy data unavailable.")
    else:
        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        a_col1.metric("Total Decisions", str(accuracy.total_decisions))
        a_col2.metric(
            "Accuracy",
            f"{accuracy.accuracy:.1%}",
            delta=f"target ≥ {accuracy.accuracy_threshold:.0%}",
            delta_color="normal" if accuracy.accuracy >= accuracy.accuracy_threshold else "inverse",
        )
        a_col3.metric(
            "Override Rate",
            f"{accuracy.override_rate:.1%}",
            delta=f"limit ≤ {accuracy.override_rate_threshold:.0%}",
            delta_color="normal" if accuracy.override_rate <= accuracy.override_rate_threshold else "inverse",
        )
        a_col4.metric("Window", f"{accuracy.window} docs")

        if accuracy.rollback_needed:
            st.error(
                f"🚨 **Rollback recommended!**  \n"
                + "\n".join(f"- {r}" for r in accuracy.rollback_reasons)
            )
        elif accuracy.total_decisions < accuracy.window:
            st.info(
                f"📋 {accuracy.total_decisions}/{accuracy.window} decisions recorded. "
                "Auto-rollback check activates when the window is full."
            )
        else:
            st.success("✅ Agent accuracy is within acceptable thresholds.")

    st.divider()

    # ── Live document tester ───────────────────────────────────────────────────
    st.subheader("🧪 Try a Document")
    st.caption("Send a document directly to the agent pipeline and see the full reasoning chain.")

    test_text = st.text_area(
        "Paste document text",
        height=140,
        placeholder=(
            "Invoice #INV-2024-001\n"
            "Vendor: Acme Corp SARL\n"
            "Invoice Date: 2024-01-15\n"
            "Total: $5,000.00"
        ),
        key="agent_test_text",
    )
    test_doc_id = st.text_input(
        "Document ID (optional)", placeholder="e.g. test-invoice-001", key="agent_test_doc_id"
    )

    if st.button("▶  Run agent pipeline", key="run_agent_btn", use_container_width=False):
        if not test_text.strip():
            st.warning("Please paste document text before running.")
        else:
            with st.spinner("Running all 4 agents…"):
                result = agent_svc.run_agent_extraction(
                    text=test_text,
                    document_id=test_doc_id or None,
                )

            if result is None:
                st.error("Agent service returned no result. Check connectivity.")
            else:
                # Decision badge.
                badge = {"auto_approve": "✅", "human_review": "🔍", "reject": "❌"}.get(
                    result.action, "❓"
                )
                st.markdown(
                    f"### {badge} Decision: **{result.action.upper().replace('_', ' ')}**  "
                    f"— confidence **{result.confidence:.1%}**  "
                    f"— `{result.doc_type}`  "
                    f"— {result.duration_ms} ms"
                )

                # Agents used + safety rails.
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.markdown("**Agents used:**")
                    st.write(" → ".join(result.agents_used) if result.agents_used else "none")
                with r_col2:
                    if result.safety_rails_triggered:
                        st.markdown("**Safety rails triggered:**")
                        for rail in result.safety_rails_triggered:
                            st.markdown(f"- ⛔ `{rail}`")
                    else:
                        st.markdown("**Safety rails:** ✅ none triggered")

                if result.fallback_used:
                    st.warning("⚠️ Fallback was used — agent pipeline encountered an error.")

                # Validation issues.
                if result.validation_issues:
                    with st.expander(f"⚠️ Validation issues ({len(result.validation_issues)})"):
                        st.dataframe(
                            [
                                {
                                    "Field": i.get("field_name", ""),
                                    "Type": i.get("issue_type", ""),
                                    "Severity": i.get("severity", "").upper(),
                                    "Description": i.get("description", ""),
                                }
                                for i in result.validation_issues
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )

                # Full reasoning chain.
                with st.expander("💬 Full agent reasoning chain", expanded=True):
                    for line in result.reasoning.split("\n"):
                        if line.strip():
                            st.markdown(line)

                # Inline feedback.
                st.divider()
                st.markdown("#### Submit feedback on this decision")
                fb_col1, fb_col2 = st.columns([2, 1])
                with fb_col1:
                    human_outcome = st.selectbox(
                        "Your decision",
                        options=["approve", "reject", "review"],
                        key="inline_feedback_outcome",
                    )
                    fb_notes = st.text_input(
                        "Notes (optional)", key="inline_feedback_notes"
                    )
                with fb_col2:
                    fb_vendor = st.text_input("Vendor (optional)", key="inline_fb_vendor")
                    fb_amount = st.number_input(
                        "Amount (optional)", min_value=0.0, value=0.0, key="inline_fb_amount"
                    )

                if st.button("📤 Submit feedback", key="inline_fb_submit"):
                    recorded = agent_svc.submit_feedback(
                        document_id=result.document_id,
                        agent_decision=result.action,
                        human_outcome=human_outcome,
                        vendor=fb_vendor,
                        amount=float(fb_amount),
                        notes=fb_notes,
                    )
                    if recorded:
                        st.success(
                            "✅ Feedback recorded. Agent will learn from this decision."
                        )
                    else:
                        st.error("❌ Failed to record feedback. Check agent service logs.")

    st.divider()

    # ── Standalone feedback form ───────────────────────────────────────────────
    st.subheader("📤 Submit Standalone Feedback")
    st.caption("Record human feedback for a previously processed document.")

    with st.form("standalone_feedback_form"):
        sf_col1, sf_col2 = st.columns(2)
        with sf_col1:
            sf_doc_id = st.text_input("Document ID *", placeholder="e.g. INV-2024-001")
            sf_agent_decision = st.selectbox(
                "Agent's decision", options=["auto_approve", "human_review", "reject"]
            )
            sf_human_outcome = st.selectbox(
                "Your decision", options=["approve", "reject", "review"]
            )
        with sf_col2:
            sf_vendor = st.text_input("Vendor (optional)")
            sf_amount = st.number_input("Amount (optional)", min_value=0.0, value=0.0)
            sf_notes = st.text_area("Notes (optional)", height=80)

        submitted = st.form_submit_button("Submit feedback", use_container_width=True)
        if submitted:
            if not sf_doc_id.strip():
                st.error("Document ID is required.")
            else:
                recorded = agent_svc.submit_feedback(
                    document_id=sf_doc_id.strip(),
                    agent_decision=sf_agent_decision,
                    human_outcome=sf_human_outcome,
                    vendor=sf_vendor,
                    amount=float(sf_amount),
                    notes=sf_notes,
                )
                if recorded:
                    st.success("✅ Feedback recorded and learning signals updated.")
                else:
                    st.error("❌ Failed to submit feedback.")

    st.divider()

    # ── Admin controls ─────────────────────────────────────────────────────────
    st.subheader("🛠 Admin Controls")
    st.caption("Enable or disable agents at runtime without restarting the service.")

    adm_col1, adm_col2 = st.columns(2)
    with adm_col1:
        disable_reason = st.text_input(
            "Reason (for disable)", value="dashboard_manual", key="disable_reason"
        )
        if st.button("🔴  Disable Agents", key="disable_agents_btn", use_container_width=True):
            ok = agent_svc.disable_agents(reason=disable_reason)
            if ok:
                st.warning("⚠️ Agents disabled. All documents will route to human review.")
                st.rerun()
            else:
                st.error("Failed to disable agents. Check connectivity.")

    with adm_col2:
        st.markdown("&nbsp;", unsafe_allow_html=True)  # Vertical alignment spacer
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🟢  Re-enable Agents", key="enable_agents_btn", use_container_width=True):
            ok = agent_svc.enable_agents()
            if ok:
                st.success("✅ Agents re-enabled.")
                st.rerun()
            else:
                st.error("Failed to re-enable agents. Check connectivity.")


if __name__ == "__main__":
    main()

