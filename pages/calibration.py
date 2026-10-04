"""Calibration page — camera calibration and dimensional setup."""
import streamlit as st
import cv2
import numpy as np
from datetime import datetime

from measurement.calibration import CameraCalibrator, DimensionalMeasurer, dimensional_analysis
from integrationsbyAbhi.dimensionsCamYOLO import (
    KNOWN_DISTANCE_MM, KNOWN_WIDTH_MM,
    calculate_focal_length, calculate_real_width, calculate_real_length,
    check_length_and_width, draw_yolo_dimension_annotations,
    detect_best_target_box,
)
from utils.config import CALIBRATION_CONFIG, THEME, get_profile
from storage.database import get_database


def render_calibration():
    st.markdown('<div class="page-title">🔬 Camera Calibration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Calibrate pixel-to-physical conversion for accurate dimensional measurement'
        '</div>',
        unsafe_allow_html=True,
    )

    if "calibrator" not in st.session_state:
        st.session_state.calibrator = CameraCalibrator()
    cal: CameraCalibrator = st.session_state.calibrator

    # ── Current calibration status ─────────────────────────────
    st.markdown("### 📐 Current Calibration Status")
    if cal.calibration and cal.calibration.is_valid:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Status", "✅ Valid")
        sc2.metric("Method", cal.calibration.method.replace("_", " ").title())
        fl_val = getattr(cal.calibration, "focal_length", 0.0)
        if fl_val > 0:
            sc3.metric("Focal Length", f"{fl_val:.1f} px", delta=f"{cal.calibration.pixels_per_mm:.3f} px/mm")
        else:
            sc3.metric("Calibration Scale", f"{cal.calibration.pixels_per_mm * 10:.1f} px/cm", delta=f"{cal.calibration.pixels_per_mm:.3f} px/mm")
        sc4.metric("Valid Until", cal.calibration.valid_until[:10] if cal.calibration.valid_until else "—")

        if cal.calibration.reprojection_error:
            st.caption(f"Reprojection error: {cal.calibration.reprojection_error:.4f}")
    elif cal.calibration and not cal.calibration.is_valid:
        st.warning("⚠️ Calibration has expired. Please recalibrate.")
    else:
        st.info(
            "📏 No calibration set. Physical measurements will not be available. "
            "Choose a method below to calibrate."
        )

    st.markdown("---")

    # ── Calibration methods ────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📏 Reference Object",
        "♟ Checkerboard",
        "📱 Pinhole & Phone Reference (dimensionsCamYOLO)",
        "✏️ Manual Entry",
    ])

    with tab1:
        st.markdown("#### Reference Object Calibration")
        st.markdown(
            "Upload an image with a known reference dimension "
            "(ruler, calibration block, etc.). Click two points that "
            "define the known distance."
        )

        ref_image = st.file_uploader(
            "Upload calibration image", type=["jpg", "jpeg", "png", "bmp"],
            key="cal_ref_image",
        )

        if ref_image:
            file_bytes = np.frombuffer(ref_image.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is not None:
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Calibration image — note the pixel coordinates of two reference points",
                    use_container_width=True,
                )
                h, w = image.shape[:2]
                st.caption(f"Image size: {w} × {h} px")

                rc1, rc2 = st.columns(2)
                with rc1:
                    x1 = st.number_input("Point 1 — X", 0, w, 100, key="ref_x1")
                    y1 = st.number_input("Point 1 — Y", 0, h, 100, key="ref_y1")
                with rc2:
                    x2 = st.number_input("Point 2 — X", 0, w, 400, key="ref_x2")
                    y2 = st.number_input("Point 2 — Y", 0, h, 100, key="ref_y2")

                known_mm = st.number_input(
                    "Known distance between points (mm)",
                    min_value=1.0, value=200.0, step=1.0,
                )

                pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                st.info(f"Pixel distance: {pixel_dist:.1f} px → {pixel_dist / known_mm:.4f} px/mm")

                if st.button("✅ Calibrate with Reference", type="primary"):
                    try:
                        cal_data = cal.calibrate_with_reference(
                            image, ((x1, y1), (x2, y2)), known_mm
                        )
                        db = get_database()
                        db.save_calibration(cal_data.to_dict())
                        st.success(
                            f"Calibration saved! {cal_data.pixels_per_mm:.4f} px/mm. "
                            f"Valid until {cal_data.valid_until[:10]}."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Calibration failed: {e}")

    with tab2:
        st.markdown("#### Checkerboard Calibration")
        st.markdown(
            f"Upload **{CALIBRATION_CONFIG['min_calibration_images']}+** images "
            f"of a {CALIBRATION_CONFIG['checkerboard_size'][0]}×"
            f"{CALIBRATION_CONFIG['checkerboard_size'][1]} checkerboard pattern."
        )

        cb_images = st.file_uploader(
            "Upload checkerboard images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="cal_cb_images",
        )

        cb_square_size = st.number_input(
            "Square size (mm)",
            min_value=1.0,
            value=CALIBRATION_CONFIG["square_size_mm"],
            step=0.5,
        )

        if cb_images and len(cb_images) >= CALIBRATION_CONFIG["min_calibration_images"]:
            if st.button("✅ Calibrate with Checkerboard", type="primary"):
                images = []
                for f in cb_images:
                    data = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is not None:
                        images.append(img)

                try:
                    with st.spinner("Detecting checkerboard corners..."):
                        cal_data = cal.calibrate_with_checkerboard(images)
                    db = get_database()
                    db.save_calibration(cal_data.to_dict())
                    st.success(
                        f"Calibration saved! Reprojection error: {cal_data.reprojection_error:.4f}. "
                        f"Valid until {cal_data.valid_until[:10]}."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Checkerboard calibration failed: {e}")
        elif cb_images:
            st.warning(
                f"Need at least {CALIBRATION_CONFIG['min_calibration_images']} images. "
                f"Currently have {len(cb_images)}."
            )

    with tab3:
        st.markdown("#### 📱 Pinhole & Reference Calibration (`integrationsbyAbhi/dimensionsCamYOLO.py`)")
        st.markdown(
            "Uses a standard mobile phone reference (**75.0 mm**) at a known distance (**300.0 mm**) "
            "with YOLO target detection to calculate the camera's pinhole focal length $F = \\frac{P \\cdot D}{W}$."
        )

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            known_w = st.number_input("Known Object Width (mm)", min_value=1.0, value=KNOWN_WIDTH_MM, step=0.1, help="Standard phone width is 75.0 mm (COCO class 67)")
            known_d = st.number_input("Distance to Camera (mm)", min_value=10.0, value=KNOWN_DISTANCE_MM, step=10.0, help="Distance from camera lens to object (e.g. 300 mm = 30 cm)")
        with pcol2:
            meas_px = st.number_input("Measured Pixel Width", min_value=1.0, value=250.0, step=1.0, help="Pixel width on sensor")
            fl_calc = calculate_focal_length(meas_px, known_w, known_d)
            st.metric("Computed Focal Length", f"{fl_calc:.1f} px", delta=f"{fl_calc/known_d:.3f} px/mm")

        pinhole_img = st.file_uploader("Upload Calibration Photo for Auto-Detection", type=["jpg", "jpeg", "png", "bmp"], key="pinhole_cal_img")
        if pinhole_img:
            data = np.frombuffer(pinhole_img.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                best_box = detect_best_target_box(img, target_class_id=67) or detect_best_target_box(img)
                if best_box is not None:
                    bx1, by1, bx2, by2 = best_box
                    pw = float(abs(bx2 - bx1))
                    vis = draw_yolo_dimension_annotations(img, best_box, width_cm=round(known_w / 10.0, 2))
                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Auto-Detected YOLO Target: {pw:.1f} px", use_container_width=True)
                    meas_px = pw

        if st.button("✅ Save Pinhole Calibration (dimensionsCamYOLO)", type="primary", key="save_pinhole_btn"):
            cal_data = cal.calibrate_pinhole(meas_px, known_w, known_d)
            db = get_database()
            db.save_calibration(cal_data.to_dict())
            st.success(
                f"Pinhole calibration saved! Focal length: {cal_data.focal_length:.1f} px ({cal_data.pixels_per_mm:.3f} px/mm). "
                f"Valid until {cal_data.valid_until[:10]}."
            )
            st.rerun()

    with tab4:
        st.markdown("#### Manual Calibration")
        st.markdown(
            "If you know the exact pixels-per-mm ratio, enter it directly."
        )

        manual_ppmm = st.number_input(
            "Pixels per mm", min_value=0.001, value=5.0, step=0.1,
            key="manual_ppmm",
        )
        st.caption(f"1 mm = {manual_ppmm:.3f} pixels  |  1 pixel = {1/manual_ppmm:.4f} mm")

        if st.button("✅ Save Manual Calibration", type="primary"):
            cal_data = cal.calibrate_manual(manual_ppmm)
            db = get_database()
            db.save_calibration(cal_data.to_dict())
            st.success(f"Manual calibration saved. Valid until {cal_data.valid_until[:10]}.")
            st.rerun()

    # ── Dimensional Analysis test panel (dimensionsCamYOLO.py) ─────
    st.markdown("---")
    st.markdown("### 📏 Dimensional Analysis Test Panel (`integrationsbyAbhi/dimensionsCamYOLO.py`)")
    st.markdown("Upload a billet image to verify YOLO Center-Crosshair Lock, Exponential Moving Average smoothing, and calibrated length/width tolerance checks.")

    dim_test_img = st.file_uploader(
        "Upload billet image for dimensional testing", type=["jpg", "jpeg", "png", "bmp"],
        key="dim_test_img_upl",
    )

    if dim_test_img:
        file_bytes = np.frombuffer(dim_test_img.read(), np.uint8)
        d_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if d_image is not None:
            profile_test = get_profile("Billet 750x150x150")
            measurer = DimensionalMeasurer(cal)
            d_results = measurer.measure_from_bbox((0, 0, d_image.shape[1], d_image.shape[0]), profile_test, image=d_image)
            chk_info = check_length_and_width(image=d_image, profile=profile_test)

            dc1, dc2 = st.columns(2)
            with dc1:
                annot_d = d_image.copy()
                l_val = d_results["length"].physical_value_cm
                w_val = d_results["width"].physical_value_cm
                annot_box = chk_info.get("locked_box") or (0, 0, d_image.shape[1], d_image.shape[0])
                annot_d = draw_yolo_dimension_annotations(
                    annot_d, annot_box,
                    length_cm=l_val, width_cm=w_val,
                    is_pass=d_results["width"].is_within_tolerance,
                )
                st.image(cv2.cvtColor(annot_d, cv2.COLOR_BGR2RGB), caption="YOLO Center-Crosshair Lock & Target Annotation", use_container_width=True)

            with dc2:
                st.markdown("##### 📐 Measured Dimensions")
                l_cm = d_results["length"].physical_value_cm
                w_cm = d_results["width"].physical_value_cm
                st.write(f"- **Length:** `{l_cm} cm` ({'PASS' if d_results['length'].is_within_tolerance else 'OUT OF SPEC'})")
                st.write(f"- **Width:** `{w_cm} cm` ({'PASS' if d_results['width'].is_within_tolerance else 'OUT OF SPEC'})")
                st.write(f"- **Method:** `{chk_info.get('calibration_method', 'N/A')}`")
                if d_results["length"].notes:
                    st.caption(d_results["length"].notes)

    # ── OCR test panel ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔤 OCR Test Panel")
    st.markdown("Upload an image of a billet marking to test OCR preprocessing.")

    ocr_test_img = st.file_uploader(
        "Upload marking image", type=["jpg", "jpeg", "png", "bmp"],
        key="ocr_test",
    )

    if ocr_test_img:
        from ocr_module.ocr_extraction import OCREngine

        file_bytes = np.frombuffer(ocr_test_img.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            ocr_eng = OCREngine()
            result = ocr_eng.extract(image)

            tc1, tc2 = st.columns(2)
            with tc1:
                annotated_test = ocr_eng.draw_ocr_annotations(image, result.detections) if result.detections else image
                st.image(cv2.cvtColor(annotated_test, cv2.COLOR_BGR2RGB),
                         caption="RapidOCR Detections Overlay", use_container_width=True)
            with tc2:
                st.markdown("**OCR Results:**")
                st.write(f"- Engine: `{result.engine}`")
                st.write(f"- Raw text: `{result.raw_text}`")
                st.write(f"- Cleaned: `{result.cleaned_text}`")
                st.write(f"- Billet ID: `{result.billet_id or '—'}`")
                st.write(f"- Heat #: `{result.heat_number or '—'}`")
                st.write(f"- Confidence: {result.confidence:.0%}")
                st.write(f"- Preprocessing: {result.preprocessing_method}")
                if result.qr_data:
                    st.write(f"- QR: `{result.qr_data}`")
                if result.validation_warnings:
                    st.warning("Warnings: " + "; ".join(result.validation_warnings))
