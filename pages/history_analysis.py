"""
History & Analysis Page — Search, filter, inspect, and export historical quality & traceability records.
Displays defect compliance, heat-by-heat quality distributions, and Excel exports.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from pathlib import Path

from storage.database import get_database
from utils.config import THEME, OUTPUT_DIR, EVIDENCE_DIR


def render_history():
    st.markdown('<div class="page-title">📊 Quality History &amp; Traceability Archive</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Search, filter, analyze defect QC &amp; traceability, and export Excel QC certificates</div>',
        unsafe_allow_html=True,
    )

    db = get_database()

    # ── Filters ───────────────────────────────────────────────────
    with st.expander("🔍 Filter Inspection Records", expanded=True):
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        with fc1:
            status_filter = st.selectbox(
                "Inspection Status",
                ["All", "PASS", "FAIL", "REWORK", "REVIEW_REQUIRED", "UNVERIFIED"],
            )
        with fc2:
            billet_filter = st.text_input("Billet Number", placeholder="e.g. BLT-1042")
        with fc3:
            heat_filter = st.text_input("Heat Number", placeholder="e.g. HT-88215")
        with fc4:
            batch_filter = st.text_input("Batch Number", placeholder="e.g. BATCH-01")
        with fc5:
            date_range = st.selectbox("Date Range", ["All", "Today", "Last 7 days", "Last 30 days"])

    params = {"limit": 500}
    if status_filter != "All":
        params["status"] = status_filter
    if billet_filter:
        params["billet_id"] = billet_filter
    if heat_filter:
        params["heat_number"] = heat_filter
    if batch_filter:
        params["batch_number"] = batch_filter
    if date_range == "Today":
        params["date_from"] = datetime.now().strftime("%Y-%m-%d")
    elif date_range == "Last 7 days":
        params["date_from"] = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    elif date_range == "Last 30 days":
        params["date_from"] = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    records = db.get_inspections(**params)

    st.markdown(f"**{len(records)}** inspection record(s) matching criteria")

    if not records:
        st.info("No records match the current filter selection.")
        return

    # ── Data Table with Traceability & Defect Status ──────────────
    df = db.export_to_dataframe(**{k: v for k, v in params.items() if k != "limit"})

    # Priority visible columns
    preferred_cols = [
        "Billet Number", "Heat Number", "Inspection Status",
        "Defect Detected", "Defect Type", "Defect Count",
        "Timestamp", "Source File"
    ]
    show_cols = [c for c in preferred_cols if c in df.columns]
    try:
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True, height=360)
    except Exception:
        st.table(df[show_cols])

    # ── Export Options ────────────────────────────────────────────
    st.markdown("### 📥 Export Traceability Data")
    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        try:
            excel_path = db.export_to_excel(**{k: v for k, v in params.items() if k != "limit"})
            with open(excel_path, "rb") as f:
                st.download_button(
                    label="📊 Download Excel Spreadsheet (.xlsx)",
                    data=f.read(),
                    file_name=Path(excel_path).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Excel export error: {e}")

    with ec2:
        try:
            csv_path = db.export_to_csv(**{k: v for k, v in params.items() if k != "limit"})
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📄 Download CSV Export (.csv)",
                    data=f.read(),
                    file_name=Path(csv_path).name,
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"CSV export error: {e}")

    with ec3:
        quick_csv = df.to_csv(index=False)
        st.download_button(
            label="⚡ Quick Filtered CSV Export",
            data=quick_csv,
            file_name=f"inspections_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Trend Analysis Charts ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Quality Compliance &amp; Defect Analysis")

    tcol1, tcol2 = st.columns(2)

    with tcol1:
        if "Timestamp" in df.columns:
            df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
            trend = df.groupby(["Date", "Inspection Status"]).size().reset_index(name="Count")
            if not trend.empty:
                fig = go.Figure()
                status_colors = {
                    "PASS": "#16A34A",
                    "FAIL": "#DC2626",
                    "REVIEW_REQUIRED": "#D97706",
                    "REWORK": "#EA580C",
                    "UNVERIFIED": "#94A3B8",
                }
                for status in trend["Inspection Status"].unique():
                    sub = trend[trend["Inspection Status"] == status]
                    fig.add_trace(go.Scatter(
                        x=sub["Date"], y=sub["Count"],
                        mode="lines+markers",
                        name=status,
                        line=dict(color=status_colors.get(status, "#3B82F6"), width=2.5),
                        marker=dict(size=7),
                    ))
                fig.update_layout(
                    title="Inspection Volume & Compliance Over Time",
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font_color="#334155",
                    height=300,
                    margin=dict(l=20, r=20, t=35, b=25),
                    legend=dict(orientation="h", y=-0.2),
                    xaxis=dict(showgrid=False, linecolor="#CBD5E1"),
                    yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1"),
                )
                st.plotly_chart(fig, use_container_width=True)

    with tcol2:
        # Defect Distribution
        if "Defect Type" in df.columns:
            def_counts = df[df["Defect Type"] != "None"]["Defect Type"].value_counts().reset_index()
            def_counts.columns = ["Defect Type", "Count"]
            if not def_counts.empty:
                fig2 = go.Figure(go.Bar(
                    x=def_counts["Defect Type"],
                    y=def_counts["Count"],
                    marker_color="#DC2626",
                ))
                fig2.update_layout(
                    title="Defect Distribution by Type",
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font_color="#334155",
                    height=300,
                    margin=dict(l=20, r=20, t=35, b=25),
                    xaxis=dict(showgrid=False, linecolor="#CBD5E1"),
                    yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", title="Occurrences"),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No defects identified in the filtered inspection set.")

    # ── Detailed Record Inspection ────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Individual Billet Audit Inspector")
    record_ids = [f"{r.get('billet_id', '—')} | {r.get('heat_number', '—')} ({r['timestamp'][:16]})" for r in records[:50]]
    selected_idx = st.selectbox(
        "Select Billet Record to Inspect",
        range(len(record_ids)),
        format_func=lambda i: record_ids[i],
    )

    if selected_idx is not None and selected_idx < len(records):
        rec = records[selected_idx]
        dc1, dc2 = st.columns([1.2, 1.0])

        with dc1:
            st.markdown("##### 🏷️ Identification & Traceability")
            st.write(f"- **Billet Number / ID:** `{rec.get('billet_id', '—')}`")
            st.write(f"- **Heat Number:** `{rec.get('heat_number', '—')}`")
            st.write(f"- **Batch Number:** `{rec.get('batch_number', '—')}`")
            st.write(f"- **QC Status:** `{rec.get('inspection_status', '—')}`")
            st.write(f"- **Defect Detected:** `{'Yes' if rec.get('defect_detected') else 'No'}`")
            st.write(f"- **Defect Type:** `{(rec.get('defect_type') or 'None').replace('_', ' ').title()}`")
            st.write(f"- **Timestamp:** `{rec.get('timestamp', '—')}`")
            st.write(f"- **Source:** `{rec.get('source_file', '—')}`")

        with dc2:
            st.markdown("##### 📸 Visual QC Evidence")
            evidence = rec.get("evidence_path")
            if evidence:
                p = Path(evidence)
                if not p.exists():
                    fallback = EVIDENCE_DIR / p.name
                    if fallback.exists():
                        evidence = str(fallback)
            if evidence and Path(evidence).exists():
                import cv2
                img = cv2.imread(evidence)
                if img is not None:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Evidence: {rec.get('billet_id', '')}", use_container_width=True)
            else:
                st.caption("No visual evidence image archived for this record.")
