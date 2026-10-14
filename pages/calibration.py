"""Deprecated — Camera calibration and dimensional analysis have been completely wiped out."""
import streamlit as st


def render_calibration():
    st.markdown('<div class="page-title">🔬 Calibration &amp; Dimensions (Removed)</div>', unsafe_allow_html=True)
    st.info(
        "ℹ️ Size calibration and dimensional analysis features have been completely removed from the inspection pipeline. "
        "Quality control evaluation is strictly determined by surface defect detection and OCR traceability."
    )
