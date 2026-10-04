"""Settings page — billet profiles, model configuration, system settings."""
import streamlit as st
import json
from pathlib import Path

from utils.config import (
    THEME, DEFAULT_BILLET_PROFILES, DETECTION_CONFIG, DEFECT_CONFIG,
    OCR_CONFIG, BASE_DIR, MODELS_DIR, get_all_profiles,
    is_qc_suspended, set_qc_suspended,
)
from storage.database import get_database


def render_settings():
    st.markdown('<div class="page-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Configure billet profiles, models, testing controls, and quality requirements</div>',
        unsafe_allow_html=True,
    )

    # ─── Prominent QC Suspension / Testing Mode Banner ───────────────
    current_qc_suspended = is_qc_suspended()

    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px 20px; margin-bottom:1.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:1.05rem; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:8px;">
                    <span>🧪</span> <span>Testing Mode &amp; QC Requirements Override</span>
                </div>
                <div style="font-size:0.82rem; color:#64748B; margin-top:2px;">
                    Suspend all QC rejection criteria (tolerances, defects, mandatory IDs, and calibration blocks) during prototype and pipeline testing.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    banner_c1, banner_c2 = st.columns([1.6, 2.4])
    with banner_c1:
        toggle_val = st.toggle(
            "🧪 **Suspend All QC Requirements (Testing Mode)**",
            value=current_qc_suspended,
            key="qc_suspend_master_toggle",
            help="When enabled, all billets evaluate to PASS so you can test conveyor speed, camera feeds, and tracking without QC halts.",
        )
        if toggle_val != current_qc_suspended:
            set_qc_suspended(toggle_val)
            st.rerun()

    with banner_c2:
        if toggle_val:
            st.markdown("""
            <div style="background:#FEF3C7; border:1px solid #F59E0B; border-radius:8px; padding:8px 14px;">
                <div style="color:#B45309; font-weight:700; font-size:0.85rem; display:flex; align-items:center; gap:6px;">
                    <span>⚠️</span> <span>TESTING MODE ACTIVE — ALL QC REQUIREMENTS SUSPENDED</span>
                </div>
                <div style="color:#92400E; font-size:0.75rem; margin-top:2px; line-height:1.4;">
                    All inspected billets will evaluate to <b>PASS</b>. Out-of-spec dimensions, surface defects, and missing markings are logged as diagnostic telemetry without rejecting billets.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:8px; padding:8px 14px;">
                <div style="color:#15803D; font-weight:700; font-size:0.85rem; display:flex; align-items:center; gap:6px;">
                    <span>🛡️</span> <span>PRODUCTION MODE ACTIVE — STRICT QC ENFORCED</span>
                </div>
                <div style="color:#166534; font-size:0.75rem; margin-top:2px; line-height:1.4;">
                    Standard industrial tolerances, defect rejection rules, and traceability requirements are strictly enforced for factory dispatch.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    tab_qc, tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 QC & Testing Mode",
        "📏 Billet Profiles",
        "🧠 Model Configuration",
        "🔤 OCR Settings",
        "ℹ️ System Info",
    ])

    # ── QC & Testing Mode Details ─────────────────────────────────
    with tab_qc:
        st.markdown("### 🧪 Quality Control (QC) Suspension Details")
        st.markdown(
            "This toggle allows engineers and quality managers to run integration and sensor tests "
            "without having billets rejected or flagged for manual review."
        )

        qc1, qc2 = st.columns(2)
        with qc1:
            st.markdown("#### ⚙️ Active Rejection Criteria Breakdown")
            st.markdown(f"""
            - **Current Status:** `{'⚠️ SUSPENDED (Testing Mode)' if toggle_val else '🛡️ ENFORCED (Production Mode)'}`
            - **Dimensional Tolerances:** `{'Bypassed (Logged as Telemetry)' if toggle_val else 'Strictly Enforced (±3.0mm width / ±7.5mm length)'}`
            - **Defect Rejection Rule:** `{'Bypassed (Overlays Shown, No Reject)' if toggle_val else 'Enforced (FAIL / REWORK on crack/defect)'}`
            - **Traceability / Stencil OCR:** `{'Bypassed (Fallback ID allowed)' if toggle_val else 'Enforced (ID required for dispatch)'}`
            - **Camera Calibration Requirement:** `{'Bypassed (Uncalibrated OK)' if toggle_val else 'Enforced (Calibration required for PASS)'}`
            """)

            action_btn_text = "🛡️ Switch to Production Mode (Enforce QC)" if toggle_val else "🧪 Switch to Testing Mode (Suspend QC)"
            action_btn_type = "secondary" if toggle_val else "primary"
            if st.button(action_btn_text, type=action_btn_type, key="quick_switch_qc_btn"):
                set_qc_suspended(not toggle_val)
                st.rerun()

        with qc2:
            st.markdown("#### 📋 What happens when QC is suspended?")
            st.markdown("""
            1. **Real-time Telemetry Remains Intact**:
               - YOLO bounding boxes and Center-Lock crosshairs are still drawn.
               - Exponential Moving Average (EMA) smoothing and `0.1x` length scaling remain active.
               - Real width, length, and height are still computed and displayed in centimeters and millimeters.
            2. **No Rejections During Testing**:
               - Billets that are out of tolerance will **not** trigger a `FAIL` decision.
               - Detected surface defects are visualized with confidence percentages, but the billet retains `PASS`.
               - Missing or obscured OCR markings will not cause the pipeline to stall or demand manual review.
            3. **Database Records**:
               - Inspection records in SQLite are marked `PASS` with review reasons tagged as `[QC Suspended]`.
            """)


    # ── Billet Profiles ───────────────────────────────────────────
    with tab1:
        st.markdown("### Active Profiles")
        profiles = get_all_profiles()

        for name, profile in profiles.items():
            with st.expander(f"📐 {name}", expanded=False):
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.write(f"**Description:** {profile.get('description', '—')}")
                    st.write(f"**Cross-section:** {profile.get('cross_section', '—')}")
                    st.write(f"**Defect rule:** {profile.get('defect_rule', '—')}")
                    st.write(f"**Required fields:** {', '.join(profile.get('required_fields', []))}")
                with pc2:
                    nl = profile.get('nominal_length_mm', 0)
                    nw = profile.get('nominal_width_mm', 0)
                    nh = profile.get('nominal_height_mm', 0)
                    st.write(f"**Nominal Length:** {nl} mm ({nl/10.0:.1f} cm)")
                    st.write(f"**Nominal Width:** {nw} mm ({nw/10.0:.1f} cm)")
                    st.write(f"**Nominal Height:** {nh} mm ({nh/10.0:.1f} cm)")
                    st.write(f"**Length Tolerance:** ±{profile.get('length_tolerance_mm', 0)} mm (±{profile.get('length_tolerance_mm', 0)/10.0:.2f} cm)")
                    st.write(f"**Width Tolerance:** ±{profile.get('width_tolerance_mm', 0)} mm (±{profile.get('width_tolerance_mm', 0)/10.0:.2f} cm)")
                    st.write(f"**Height Tolerance:** ±{profile.get('height_tolerance_mm', 0)} mm (±{profile.get('height_tolerance_mm', 0)/10.0:.2f} cm)")

        st.markdown("---")
        st.markdown("### ➕ Create Custom Profile")

        cp1, cp2 = st.columns(2)
        with cp1:
            new_name = st.text_input("Profile Name", placeholder="e.g. Custom Billet 800x160x160")
            new_length = st.number_input("Nominal Length (mm)", 0.0, 5000.0, 750.0, 1.0)
            new_width = st.number_input("Nominal Width (mm)", 0.0, 1000.0, 150.0, 1.0)
            new_height = st.number_input("Nominal Height (mm)", 0.0, 1000.0, 150.0, 1.0)
        with cp2:
            new_ltol = st.number_input("Length Tolerance (mm)", 0.0, 100.0, 7.5, 0.5)
            new_wtol = st.number_input("Width Tolerance (mm)", 0.0, 50.0, 3.0, 0.5)
            new_htol = st.number_input("Height Tolerance (mm)", 0.0, 50.0, 3.0, 0.5)
            new_defect_rule = st.selectbox("Defect Rule", ["FAIL", "REWORK", "REVIEW"])
            new_cross = st.selectbox("Cross-section", ["square", "round", "rectangular"])

        if st.button("💾 Save Profile", type="primary"):
            if new_name:
                config_path = BASE_DIR / "config.yaml"
                try:
                    import yaml
                except ImportError:
                    yaml = None

                if yaml is None:
                    st.error("PyYAML is not installed.")
                    return

                config = {}
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f) or {}

                if "billet_profiles" not in config:
                    config["billet_profiles"] = {}

                config["billet_profiles"][new_name] = {
                    "nominal_length_mm": new_length,
                    "nominal_width_mm": new_width,
                    "nominal_height_mm": new_height,
                    "length_tolerance_mm": new_ltol,
                    "width_tolerance_mm": new_wtol,
                    "height_tolerance_mm": new_htol,
                    "cross_section": new_cross,
                    "defect_rule": new_defect_rule,
                    "required_fields": ["billet_id"],
                    "description": f"Custom profile: {new_name}",
                }

                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)

                st.success(f"✅ Profile '{new_name}' saved to config.yaml")
                st.rerun()
            else:
                st.warning("Please enter a profile name.")

    # ── Model Configuration ───────────────────────────────────────
    with tab2:
        st.markdown("### 🧠 Detection Models")

        st.markdown("#### Billet Detector")
        st.write(f"- **Model path:** `{DETECTION_CONFIG['model_path']}`")
        model_exists = Path(DETECTION_CONFIG["model_path"]).exists()
        st.write(f"- **Status:** {'✅ Loaded' if model_exists else '⚠️ Not found (using fallback)'}")
        st.write(f"- **Device:** `{DETECTION_CONFIG['device']}`")
        st.write(f"- **Confidence threshold:** {DETECTION_CONFIG['confidence_threshold']}")

        st.markdown("#### Defect Detector")
        st.write(f"- **Model path:** `{DEFECT_CONFIG['model_path']}`")
        defect_model_exists = Path(DEFECT_CONFIG["model_path"]).exists()
        st.write(f"- **Status:** {'✅ Loaded' if defect_model_exists else '⚠️ Not found (using heuristic fallback)'}")
        st.write(f"- **Patch size:** {DEFECT_CONFIG['patch_size']}px")
        st.write(f"- **Patch overlap:** {DEFECT_CONFIG['patch_overlap']}")

        st.markdown("---")
        st.markdown("#### Upload Custom Model")
        model_upload = st.file_uploader(
            "Upload a YOLO .pt model file",
            type=["pt"],
            key="model_upload",
        )
        model_target = st.selectbox("Target", ["Billet Detector", "Defect Detector"])

        if model_upload and st.button("📤 Upload Model"):
            if model_target == "Billet Detector":
                target_path = MODELS_DIR / "billet_detector.pt"
            else:
                target_path = MODELS_DIR / "defect_detector.pt"

            with open(target_path, "wb") as f:
                f.write(model_upload.read())

            st.success(f"✅ Model saved to `{target_path}`. Restart the app to load it.")

    # ── OCR Settings ──────────────────────────────────────────────
    with tab3:
        st.markdown("### 🔤 OCR Configuration")
        st.write(f"- **Engine:** `{OCR_CONFIG['engine']}`")
        st.write(f"- **Languages:** {OCR_CONFIG['language']}")
        st.write(f"- **Confidence threshold:** {OCR_CONFIG['confidence_threshold']}")
        st.write(f"- **GPU enabled:** {OCR_CONFIG['gpu_enabled']}")

        st.markdown("#### ID Patterns")
        for name, pattern in OCR_CONFIG["id_patterns"].items():
            st.write(f"- **{name}:** `{pattern}`")

        st.markdown("#### Preprocessing Methods")
        for method in OCR_CONFIG["preprocessing_methods"]:
            st.write(f"- {method}")

    # ── System Info ───────────────────────────────────────────────
    with tab4:
        st.markdown("### ℹ️ System Information")
        import sys
        import platform

        si1, si2 = st.columns(2)
        with si1:
            st.write(f"- **Python:** {sys.version.split()[0]}")
            st.write(f"- **Platform:** {platform.platform()}")
            st.write(f"- **Project dir:** `{BASE_DIR}`")

            # Check dependencies
            deps = {
                "streamlit": "streamlit",
                "opencv": "cv2",
                "numpy": "numpy",
                "pandas": "pandas",
                "plotly": "plotly",
                "torch": "torch",
                "ultralytics": "ultralytics",
                "rapidocr": "rapidocr_onnxruntime",
                "onnxruntime": "onnxruntime",
                "easyocr": "easyocr",
                "openpyxl": "openpyxl",
            }
            st.markdown("#### Dependencies")
            for name, module in deps.items():
                try:
                    mod = __import__(module)
                    ver = getattr(mod, "__version__", "installed")
                    st.write(f"- ✅ {name}: {ver}")
                except Exception:
                    st.write(f"- ❌ {name}: not installed")

        with si2:
            db = get_database()
            stats = db.get_stats()
            st.markdown("#### Database Stats")
            st.write(f"- Total records: {stats['total']}")
            st.write(f"- Database path: `{db.db_path}`")
            st.write(f"- Alerts: {len(db.get_alerts())}")
            st.write(f"- Review flags: {len(db.get_review_flags())}")

            st.markdown("#### Storage")
            from utils.config import DATA_DIR, EVIDENCE_DIR, OUTPUT_DIR
            for label, path in [("Data", DATA_DIR), ("Evidence", EVIDENCE_DIR), ("Output", OUTPUT_DIR)]:
                if path.exists():
                    count = len(list(path.glob("*")))
                    st.write(f"- {label}: {count} file(s)")

        # Database reset
        st.markdown("---")
        st.markdown("### ⚠️ Danger Zone")
        if st.button("🗑️ Clear All Inspection Data", type="secondary"):
            st.warning("This will delete all inspection records, alerts, and review flags.")
            if st.button("⚠️ Confirm Delete", key="confirm_delete"):
                import os
                db.close()
                from utils.config import DATABASE_PATH
                if Path(DATABASE_PATH).exists():
                    os.remove(str(DATABASE_PATH))
                st.success("Database cleared. Restart the app.")
                st.rerun()
