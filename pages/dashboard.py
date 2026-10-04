"""
Dashboard Page — Real-time inspection monitoring, executive KPIs, and analytics.
Displays physical dimensional trends in cm, defect rates, and recent traceability logs.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import pandas as pd

from storage.database import get_database
from utils.config import THEME, DECISION_STATUSES


def render_dashboard():
    st.markdown('<div class="page-title">📊 Executive Inspection Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Real-time quality control, dimensional accuracy (cm), and Heat &amp; Billet traceability</div>',
        unsafe_allow_html=True,
    )

    db = get_database()
    stats = db.get_stats()

    # ── Primary KPI Cards ─────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    _kpi(c1, stats["total"], "Total Inspected", "#0F2942")
    _kpi(c2, stats["pass"], "Passed (QC OK)", "#16A34A")
    _kpi(c3, stats["fail"], "Failed (Out of Spec)", "#DC2626")
    _kpi(c4, stats["rework"], "Rework Required", "#EA580C")
    _kpi(c5, stats["review_required"], "Flagged Review", "#D97706")
    _kpi(c6, stats["unverified"], "Unverified", "#64748B")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Secondary Operational Metrics ─────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    _kpi(s1, f"{stats['defect_rate']:.1f}%", "Defect Rate",
         "#DC2626" if stats["defect_rate"] > 5 else "#16A34A")
    _kpi(s2, stats["today_total"], "Today's Billets", "#1E40AF")
    _kpi(s3, stats["today_pass"], "Today Passed", "#16A34A")
    _kpi(s4, stats["today_fail"], "Today Failed", "#DC2626")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Trend & Distribution Charts ───────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 📈 Daily Inspection Volume by Status")
        if stats["daily_trend"]:
            df_trend = pd.DataFrame(stats["daily_trend"])
            fig = go.Figure()
            status_colors = {
                "PASS": "#16A34A",
                "FAIL": "#DC2626",
                "REVIEW_REQUIRED": "#D97706",
                "REWORK": "#EA580C",
                "UNVERIFIED": "#94A3B8",
            }
            for status in df_trend["inspection_status"].unique():
                sub = df_trend[df_trend["inspection_status"] == status]
                fig.add_trace(go.Bar(
                    x=sub["date"], y=sub["cnt"],
                    name=status,
                    marker_color=status_colors.get(status, "#3B82F6"),
                ))
            fig.update_layout(
                barmode="stack",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_color="#334155",
                height=340,
                margin=dict(l=20, r=20, t=25, b=25),
                legend=dict(orientation="h", y=-0.18),
                xaxis=dict(showgrid=False, linecolor="#CBD5E1"),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inspection telemetry yet. Upload a video or frame to begin.")

    with col_right:
        st.markdown("#### 🎯 Overall Quality Yield")
        if stats["total"] > 0:
            labels = ["Passed", "Failed", "Rework", "Review", "Unverified"]
            values = [stats["pass"], stats["fail"], stats["rework"],
                      stats["review_required"], stats["unverified"]]
            colors = ["#16A34A", "#DC2626", "#EA580C", "#D97706", "#94A3B8"]

            fig2 = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker_colors=colors,
                textinfo="label+percent",
                textfont_size=12,
            ))
            fig2.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_color="#334155",
                height=340,
                margin=dict(l=10, r=10, t=25, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No distribution data.")

    # ── Defect Classification Summary ─────────────────────────────
    st.markdown("#### ⚠️ Defect Classification Summary")
    defect_summary = db.get_defect_summary()
    if defect_summary:
        df_def = pd.DataFrame(defect_summary)
        df_def.columns = ["Defect Type", "Count", "Average Confidence"]
        df_def["Defect Type"] = df_def["Defect Type"].apply(lambda s: str(s).replace('_', ' ').title())
        df_def["Average Confidence"] = df_def["Average Confidence"].apply(lambda x: f"{x:.1%}" if x else "—")
        try:
            st.dataframe(df_def, use_container_width=True, hide_index=True)
        except Exception:
            st.table(df_def)
    else:
        st.info("No surface defects recorded.")

    # ── Recent Inspections Table with CM Dimensions & Dual ID ─────
    st.markdown("#### 🕐 Recent Inspected Billets (with CM Dimensions & Traceability)")
    recent = db.get_inspections(limit=12)
    if recent:
        rows = []
        for r in recent:
            status = r.get("inspection_status", "—")
            l_mm = r.get("measured_length_mm")
            w_mm = r.get("measured_width_mm")
            h_mm = r.get("measured_height_mm")

            l_cm = f"{l_mm/10.0:.1f} cm" if l_mm is not None else "—"
            w_cm = f"{w_mm/10.0:.1f} cm" if w_mm is not None else "—"
            h_cm = f"{h_mm/10.0:.1f} cm" if h_mm is not None else "—"
            dim_cm = f"{l_cm} x {w_cm} x {h_cm}" if (l_mm and w_mm) else "—"

            rows.append({
                "Timestamp": r.get("timestamp", "—")[:19],
                "Billet Number": r.get("billet_id") or "—",
                "Heat Number": r.get("heat_number") or "—",
                "Status": status,
                "Dimensions (cm)": dim_cm,
                "Length (cm)": l_cm,
                "Width (cm)": w_cm,
                "Height (cm)": h_cm,
                "Defect": "Yes" if r.get("defect_detected") else "No",
                "Defect Type": (r.get("defect_type") or "").replace("_", " ").title() or "None",
            })
        try:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception:
            st.table(pd.DataFrame(rows))

        st.markdown("<br>", unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns([1, 3])
        with col_exp1:
            try:
                excel_path = db.export_to_excel()
                with open(excel_path, "rb") as f:
                    st.download_button(
                        label="📊 Download Traceability Log (.xlsx)",
                        data=f.read(),
                        file_name=Path(excel_path).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
            except Exception as e:
                st.caption(f"Excel ready in data/outputs: {e}")
    else:
        st.info("No inspection records. Go to **Upload & Process** to start.")

    # ── Recent System Alerts ──────────────────────────────────────
    st.markdown("#### 🔔 Recent Factory QC Alerts")
    alerts = db.get_alerts(acknowledged=False, limit=5)
    if alerts:
        for a in alerts:
            severity = a.get("severity", "INFO")
            css_class = {"CRITICAL": "alert-critical", "WARNING": "alert-warning"}.get(
                severity, "alert-info"
            )
            icon = {"CRITICAL": "🔴", "WARNING": "🟡"}.get(severity, "🔵")
            st.markdown(
                f'<div class="alert-bar {css_class}">{icon} '
                f'<strong>{a.get("alert_type", "")}</strong> — {a.get("message", "")} '
                f'<span style="margin-left:auto;font-size:0.8rem;color:{THEME["text_muted"]}">'
                f'{a.get("timestamp", "")[:19]}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("✓ All quality parameters nominal. No active alerts.")


def _kpi(col, value, label, color):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)
