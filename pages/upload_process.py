"""
Upload & Process Page — Video and Image Inspection Pipeline.
Executes real-time billet detection, defect classification,
OCR text & 2D DataMatrix/QR traceability extraction, and automated QC evaluation.
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import uuid
import os
from pathlib import Path
from datetime import datetime

from vision.detection import VideoProcessor
from defects.defect_detection import DefectDetector
from ocr_module.ocr_extraction import OCREngine
from rules.decision_engine import DecisionEngine
from storage.database import get_database
from utils.config import (
    THEME, VIDEO_CONFIG, UPLOAD_DIR, DEMO_DIR, EVIDENCE_DIR,
    get_all_profiles, get_profile, is_qc_suspended,
)
from utils.logger import logger


def render_upload_process():
    st.markdown('<div class="page-title">📹 Inspection &amp; Traceability Processing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Upload factory video or still frames for real-time defect QC and Heat/Billet ID capture</div>',
        unsafe_allow_html=True,
    )

    if is_qc_suspended():
        st.markdown("""
        <div style="background:#FFFBEB; border:1px solid #FCD34D; border-radius:8px; padding:10px 14px; margin-bottom:1.2rem;">
            <div style="color:#B45309; font-weight:700; font-size:0.88rem; display:flex; align-items:center; gap:8px;">
                <span>🧪</span> <span>TESTING MODE ACTIVE — ALL QC REQUIREMENTS SUSPENDED</span>
            </div>
            <div style="color:#92400E; font-size:0.78rem; margin-top:2px;">
                All QC rejection rules are suspended via Settings. Inspected billets will evaluate to <b>PASS</b> with diagnostic telemetry displayed for validation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    db = get_database()

    # Initialize session state services
    if "video_processor" not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
    if "defect_detector" not in st.session_state:
        st.session_state.defect_detector = DefectDetector()
    if "ocr_engine" not in st.session_state:
        st.session_state.ocr_engine = OCREngine()
    if "decision_engine" not in st.session_state:
        st.session_state.decision_engine = DecisionEngine()

    vp: VideoProcessor = st.session_state.video_processor
    dd: DefectDetector = st.session_state.defect_detector
    ocr: OCREngine = st.session_state.ocr_engine
    de: DecisionEngine = st.session_state.decision_engine

    # ── Configuration & Traceability Setup ────────────────────────
    st.markdown("### ⚙️ Inspection Configuration & Batch Traceability")
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)

    with cfg_col1:
        profiles = get_all_profiles()
        profile_name = st.selectbox("Billet Profile Specification", list(profiles.keys()), index=0)
        profile = profiles[profile_name]
        st.caption(profile.get("description", ""))

    with cfg_col2:
        use_sparse_sampling = st.toggle("Sparse Sampling (1:30 frames)", value=True, help="Enable this to sample 1 in 30 frames for faster video processing on large files.")
        if use_sparse_sampling:
            frame_skip = 30
            st.caption("Active: Processing 1 in 30 frames")
        else:
            frame_skip = st.slider("Frame Sampling (Every Nth)", 1, 10, VIDEO_CONFIG.get("default_frame_skip", 3))
        conf_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.45, 0.05)

    with cfg_col3:
        plant_info = st.text_input("Mill / Plant Line", value="SMS-2 / Caster-1", placeholder="e.g. Plant-A Line-2")
        shift_info = st.text_input("Shift / Operator ID", value="Shift-A / Operator-12", placeholder="e.g. Shift-B / OP-07")

    with cfg_col4:
        default_billet_prefix = st.text_input("Default Billet Prefix", value="BLT-2024", placeholder="e.g. BLT-2024")
        default_heat_num = st.text_input("Active Heat Number", value="HT-88215", placeholder="e.g. HT-88215")
        st.caption("ℹ️ Stamped tags automatically override defaults if readable.")

    st.markdown("---")

    # ── Input selection ───────────────────────────────────────────
    st.markdown("### 📁 Input Source")
    input_tab1, input_tab2, input_tab3, input_tab4 = st.tabs([
        "🎬 Factory Video",
        "🖼️ Billet Image & RapidOCR",
        "📹 Live Camera & OCR Feed",
        "🎮 Quick Demo Mode",
    ])

    with input_tab1:
        uploaded_video = st.file_uploader(
            "Upload a steel plant MP4, AVI, or MOV video",
            type=["mp4", "avi", "mov", "mkv", "wmv"],
            key="video_upload",
        )
        if uploaded_video:
            suffix = Path(uploaded_video.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(UPLOAD_DIR))
            tmp.write(uploaded_video.read())
            tmp.close()

            info = vp.get_video_info(tmp.name)
            if "error" not in info:
                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.metric("Resolution", f"{info['width']}×{info['height']}")
                ic2.metric("FPS", f"{info['fps']:.1f}")
                ic3.metric("Total Frames", info["total_frames"])
                ic4.metric("Duration", f"{info['duration_seconds']:.1f}s")

                if st.button("▶️ Start Video Inspection", type="primary", use_container_width=True):
                    _process_video(
                        tmp.name, vp, dd, ocr, de, db,
                        profile, profile_name, frame_skip, conf_threshold,
                        plant_info, shift_info, default_billet_prefix, default_heat_num,
                    )
            else:
                st.error(f"Cannot read video: {info['error']}")

    with input_tab2:
        st.markdown("#### 🖼️ Billet Image OCR & QC")
        img_source = st.radio("Image Source", ["Upload File", "Load Synthetic Factory Billet Sample"], horizontal=True, key="img_src_sel")
        image = None
        img_filename = "uploaded_billet.jpg"

        if img_source == "Upload File":
            uploaded_image = st.file_uploader(
                "Upload a high-resolution billet inspection image",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                key="image_upload",
            )
            if uploaded_image:
                file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_filename = uploaded_image.name
        else:
            image = _create_demo_frame()
            img_filename = "synthetic_billet_sample.jpg"

        if image is not None:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Active Image Frame: {img_filename}", use_container_width=True)

            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                run_ocr_scan = st.button("⚡ Run RapidOCR Live Scanner (Green Overlays)", use_container_width=True)
            with btn_c2:
                run_full_qc = st.button("🔍 Run Full Billet Inspection (QC + OCR)", use_container_width=True, type="primary")

            if run_ocr_scan:
                with st.spinner("⚡ Running ONNX-accelerated RapidOCR..."):
                    ocr_res = ocr.extract(
                        image,
                        default_billet_id=default_billet_prefix,
                        default_heat_number=default_heat_num,
                    )
                    annotated_ocr = ocr.draw_ocr_annotations(image, ocr_res.detections) if ocr_res.detections else image

                    sc1, sc2 = st.columns([1.3, 1.0])
                    with sc1:
                        st.image(cv2.cvtColor(annotated_ocr, cv2.COLOR_BGR2RGB), caption="RapidOCR Detection Overlay (Green Bounding Boxes from ocrCam.py)", use_container_width=True)
                    with sc2:
                        st.markdown("##### 🏷️ RapidOCR Extracted Markings")
                        st.markdown(f"**Billet ID:** `{ocr_res.billet_id or 'ID_UNREADABLE'}`")
                        st.markdown(f"**Heat Number:** `{ocr_res.heat_number or '—'}`")
                        st.markdown(f"**Confidence:** `{ocr_res.confidence:.1%}`")
                        st.markdown(f"**Engine:** `{ocr_res.engine.upper()} (ONNX-backed)`")
                        if ocr_res.detections:
                            st.markdown(f"**Detected Blocks:** `{len(ocr_res.detections)}`")
                            with st.expander("Detected Text Blocks", expanded=True):
                                for i, d in enumerate(ocr_res.detections, 1):
                                    st.write(f"- Block {i}: **{d['text']}** (`{d['confidence']:.0%}`)")
                        else:
                            st.info("No text markings detected in this image.")

            if run_full_qc:
                _process_single_image(
                    image, img_filename, vp, dd, ocr, de, db,
                    profile, profile_name, plant_info, shift_info,
                    default_billet_prefix, default_heat_num,
                    is_demo=(img_source != "Upload File"),
                )

    with input_tab3:
        st.markdown("#### 📹 Real-Time Live Feed OCR & Camera Inspection")
        st.markdown(
            "Powered by ONNX-accelerated **RapidOCR**. Runs live character recognition, draws real-time bounding boxes (from `integrationsbyAbhi/ocrCam.py`), and allows instantaneous freeze-frame QC inspection."
        )

        feed_mode = st.radio("Feed Source Mode", ["Live Camera Stream (Webcam / RTSP)", "Browser Camera Snapshot"], horizontal=True, key="live_feed_src_mode")

        if feed_mode == "Live Camera Stream (Webcam / RTSP)":
            cam_c1, cam_c2, cam_c3 = st.columns([1.5, 1.2, 1.2])
            with cam_c1:
                cam_source_str = st.text_input("Camera Index or RTSP Stream URL", value="0", help="0 for default integrated webcam, 1 for secondary USB camera, or rtsp:// stream URL.")
                try:
                    cam_source = int(cam_source_str)
                except ValueError:
                    cam_source = cam_source_str
            with cam_c2:
                sample_interval = st.slider("OCR Sampling (Every Nth Frame)", min_value=1, max_value=30, value=10, help="Process every Nth frame to eliminate CPU bottleneck (matches ocrCam.py).")
            with cam_c3:
                max_live_seconds = st.slider("Max Stream Duration (s)", 10, 180, 45, help="Auto-stops live streaming after this duration to prevent browser lockup.")

            start_col, stop_col, inspect_col = st.columns([1, 1, 1.5])
            with start_col:
                start_live = st.button("▶️ Start Live Feed", type="primary", use_container_width=True, key="start_live_btn")
            with stop_col:
                stop_live = st.button("⏹️ Stop Live Feed", use_container_width=True, key="stop_live_btn")
            with inspect_col:
                inspect_captured = st.button("🔍 Inspect Last Captured Frame", use_container_width=True, key="inspect_cap_btn")

            if "live_streaming" not in st.session_state:
                st.session_state.live_streaming = False
            if "last_live_frame" not in st.session_state:
                st.session_state.last_live_frame = None

            if start_live:
                st.session_state.live_streaming = True
            if stop_live:
                st.session_state.live_streaming = False

            if inspect_captured and st.session_state.last_live_frame is not None:
                st.markdown("---")
                st.markdown("#### 📸 Running QC Inspection on Captured Live Frame")
                _process_single_image(
                    st.session_state.last_live_frame, "live_camera_capture.jpg", vp, dd, ocr, de, db,
                    profile, profile_name, plant_info, shift_info,
                    default_billet_prefix, default_heat_num,
                )

            elif st.session_state.live_streaming:
                cap = cv2.VideoCapture(cam_source)
                if not cap.isOpened():
                    st.error(f"❌ Cannot connect to camera '{cam_source}'. Please verify camera is connected and permissions are granted.")
                    st.session_state.live_streaming = False
                else:
                    frame_placeholder = st.empty()
                    status_placeholder = st.empty()

                    frame_counter = 0
                    current_detections = []
                    start_stream_time = time.time()

                    try:
                        while cap.isOpened() and st.session_state.live_streaming:
                            ret, frame = cap.read()
                            if not ret:
                                status_placeholder.warning("⚠️ End of video stream or camera disconnected.")
                                break

                            # Process every Nth frame with RapidOCR (from ocrCam.py)
                            if frame_counter % sample_interval == 0:
                                res = ocr.extract(
                                    frame,
                                    default_billet_id=default_billet_prefix,
                                    default_heat_number=default_heat_num,
                                )
                                current_detections = res.detections
                                
                                # Log the text detected by OCR during camera feed
                                if current_detections:
                                    detected_texts = [d.get("text", "") for d in current_detections]
                                    logger.info(f"Live Camera OCR detected texts: {detected_texts}")

                            # Draw bounding boxes and text from latest processed frame (from ocrCam.py)
                            annotated_frame = ocr.draw_ocr_annotations(frame, current_detections) if current_detections else frame.copy()

                            # Save last frame
                            st.session_state.last_live_frame = frame.copy()

                            # Stream display in Streamlit
                            frame_placeholder.image(
                                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                                caption=f"Live OCR Pipeline Feed • Frame {frame_counter}",
                                use_container_width=True,
                            )

                            # Live metrics
                            elapsed = time.time() - start_stream_time
                            cur_fps = (frame_counter + 1) / max(elapsed, 0.001)
                            det_texts = [d["text"] for d in current_detections] if current_detections else ["None"]
                            status_placeholder.markdown(
                                f"**Live FPS:** `{cur_fps:.1f}` | **Processed Frames:** `{frame_counter}` | **Elapsed:** `{elapsed:.1f}s` | **Active Markings:** `{', '.join(det_texts[:3])}`"
                            )

                            frame_counter += 1
                            if elapsed > max_live_seconds:
                                status_placeholder.info(f"⏱️ Stream duration reached {max_live_seconds}s limit.")
                                break
                            time.sleep(0.02)
                    finally:
                        cap.release()
                        st.session_state.live_streaming = False

        else:
            # Browser Camera Snapshot mode
            st.info("📷 Snap a photo with your device's camera for instant OCR scanning & full QC inspection.")
            camera_img = st.camera_input("Take a photo of billet marking", key="browser_cam_input")
            if camera_img:
                bytes_data = camera_img.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    ocr_res = ocr.extract(
                        frame,
                        default_billet_id=default_billet_prefix,
                        default_heat_number=default_heat_num,
                    )
                    annotated_frame = ocr.draw_ocr_annotations(frame, ocr_res.detections) if ocr_res.detections else frame

                    snap_c1, snap_c2 = st.columns([1.3, 1.0])
                    with snap_c1:
                        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Captured Frame with RapidOCR Overlays", use_container_width=True)
                    with snap_c2:
                        st.markdown("##### 🏷️ RapidOCR Traceability Results")
                        st.markdown(f"- **Billet ID:** `{ocr_res.billet_id or 'ID_UNREADABLE'}`")
                        st.markdown(f"- **Heat Number:** `{ocr_res.heat_number or '—'}`")
                        st.markdown(f"- **Confidence:** `{ocr_res.confidence:.1%}`")
                        st.markdown(f"- **Detected Blocks:** `{len(ocr_res.detections)}`")

                    if st.button("🔍 Run Full Billet QC Inspection on Snapshot", type="primary", use_container_width=True, key="snap_qc_btn"):
                        _process_single_image(
                            frame, "camera_snapshot.jpg", vp, dd, ocr, de, db,
                            profile, profile_name, plant_info, shift_info,
                            default_billet_prefix, default_heat_num,
                        )

    with input_tab4:
        st.info("💡 **Factory Simulation Mode** — Rapid verification with high-fidelity synthetic billet assets containing markings, surface defects, and calibration scale.")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown("##### 🖼️ Synthetic Image QC")
            st.caption("Inspects a simulated billet with surface crack, Billet ID, and Heat Number.")
            if st.button("▶️ Run Synthetic Image Inspection", use_container_width=True, type="primary"):
                demo_frame = _create_demo_frame()
                _process_single_image(
                    demo_frame, "demo_synthetic.jpg", vp, dd, ocr, de, db,
                    profile, profile_name, plant_info, shift_info,
                    default_billet_prefix, default_heat_num, is_demo=True,
                )

        with dcol2:
            st.markdown("##### 🎬 Conveyor Video Inspection")
            demo_vid_path = str(DEMO_DIR / "demo_video.mp4")
            if Path(demo_vid_path).exists():
                st.caption("Runs automated tracking across a 120-frame simulated conveyor with continuous traceability.")
                if st.button("▶️ Run Demo Conveyor Video", use_container_width=True):
                    _process_video(
                        demo_vid_path, vp, dd, ocr, de, db,
                        profile, profile_name, frame_skip, conf_threshold,
                        plant_info, shift_info, default_billet_prefix, default_heat_num,
                    )
            else:
                st.caption("Generate pre-built factory test video.")
                if st.button("⚡ Generate Test Video", use_container_width=True):
                    from training.generate_synthetic_data import generate_demo_video
                    with st.spinner("Generating factory test video..."):
                        generate_demo_video()
                    st.success("Test video ready! Click above to run inspection.")
                    st.rerun()


# ─── Video processing ────────────────────────────────────────────

def _process_video(
    video_path, vp, dd, ocr, de, db,
    profile, profile_name, frame_skip, conf_threshold,
    plant_info, shift_info, default_billet_prefix, default_heat_num,
):
    """Process a video file frame by frame with real-time telemetry."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_display = st.empty()
    results_area = st.container()

    processed_count = 0
    billet_records = {}
    video_records_list = [] # Store records for the video specific report
    frame_idx = 0
    start_time = time.time()

    # We'll use a specific batch ID to identify this video's run
    video_batch_id = f"VID-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % frame_skip != 0:
            continue

        timestamp = frame_idx / fps
        processed_count += 1

        # Process frame detections
        result = vp.process_frame(frame, frame_idx, timestamp)
        
        annotated_frame = result.annotated_frame if result.annotated_frame is not None else frame.copy()

        for det in result.detections:
            track_id = det.track_id

            if track_id in billet_records:
                billet_records[track_id]["frames_seen"] += 1
                # If we've seen it multiple times, we still want to show defects live, but maybe we don't log a new record
                # Let's let it re-evaluate or just draw previous. We will evaluate and update.
                if billet_records[track_id]["frames_seen"] > 3:
                    continue

            # Surface defects
            defects = dd.detect(frame, roi=det.bbox)

            # OCR / 2D tag reading
            fallback_id = f"{default_billet_prefix}-{track_id.split('-')[-1]}" if default_billet_prefix else None
            ocr_result = ocr.extract(
                frame, roi=det.bbox,
                default_billet_id=fallback_id,
                default_heat_number=default_heat_num,
            )

            # QC Decision based on defects and OCR traceability
            decision = de.evaluate(
                defects=defects,
                ocr_result=ocr_result,
                profile=profile,
            )

            # Evidence crop
            evidence_path = vp.save_evidence(frame, det, prefix="billet")

            # Database record
            record = _build_record(
                det, defects, ocr_result, decision,
                evidence_path, video_path, frame_idx, timestamp,
                profile_name, plant_info, shift_info, is_demo=False,
            )
            # Tag with this video's run batch
            record["batch_number"] = video_batch_id

            if track_id not in billet_records:
                record_id = db.insert_inspection(record)
                billet_records[track_id] = {"record_id": record_id, "frames_seen": 1}
                video_records_list.append(record)
                _generate_alerts(db, record, decision, det, defects, ocr_result)
            else:
                # Update record if needed or just keep track
                pass
                
            # Draw Live Bounding Boxes for Defects
            for d in defects:
                dx1, dy1, dx2, dy2 = d.bbox
                cv2.rectangle(annotated_frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"{d.defect_type} {d.confidence:.0%}",
                            (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Annotated video display with live defect boxes
        display = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_display.image(display, caption=f"Frame {frame_idx}/{total_frames} (Sampling 1:{frame_skip})", use_container_width=True)

        progress = min(frame_idx / max(total_frames, 1), 1.0)
        progress_bar.progress(progress)
        elapsed = time.time() - start_time
        proc_fps = processed_count / max(elapsed, 0.01)
        status_text.markdown(
            f"⏱ Frame **{frame_idx}/{total_frames}** | "
            f"Processed: **{processed_count}** | "
            f"Speed: **{proc_fps:.1f} FPS** | "
            f"Billets identified: **{len(billet_records)}**"
        )

    cap.release()
    progress_bar.progress(1.0)

    elapsed = time.time() - start_time
    st.success(
        f"✅ Video processing complete! {processed_count} frames evaluated in {elapsed:.1f}s. "
        f"{len(billet_records)} billet(s) successfully inspected & traced."
    )
    
    # Generate Separate Video Report
    import pandas as pd
    if video_records_list:
        df = pd.DataFrame(video_records_list)
        report_filename = f"VideoReport_{video_batch_id}.csv"
        report_path = UPLOAD_DIR / report_filename
        df.to_csv(report_path, index=False)
        st.info(f"📄 Generated separate report for this video run: {report_filename}")
        
        with open(report_path, "rb") as f:
            st.download_button(
                label=f"📊 Download Video Report ({report_filename})",
                data=f.read(),
                file_name=report_filename,
                mime="text/csv",
                use_container_width=True,
            )

    with results_area:
        _show_results_summary(db, billet_records)


# ─── Single image processing ─────────────────────────────────────

def _process_single_image(
    image, filename, vp, dd, ocr, de, db,
    profile, profile_name, plant_info, shift_info,
    default_billet_prefix="BLT-2024", default_heat_num="HT-88215",
    is_demo=False,
):
    """Process a single image for defect QC and traceability ID capture."""
    with st.spinner("🔍 Evaluating billet defects and markings..."):
        # Detect billet ROI
        result = vp.process_frame(image, 0, 0.0)

        if not result.detections:
            h, w = image.shape[:2]
            from vision.detection import BilletDetection
            result.detections = [BilletDetection(
                bbox=(0, 0, w, h),
                confidence=0.5,
                class_name="billet_roi",
                track_id="BLT-0001",
            )]

        for det in result.detections:
            col_img, col_info = st.columns([1.3, 1.0])

            defects = dd.detect(image, roi=det.bbox)

            ocr_result = ocr.extract(
                image, roi=det.bbox,
                default_billet_id=default_billet_prefix,
                default_heat_number=default_heat_num,
            )

            decision = de.evaluate(
                defects=defects,
                ocr_result=ocr_result,
                profile=profile,
            )

            evidence_path = vp.save_evidence(image, det, prefix="billet")

            # Annotate
            annotated = image.copy()
            x1, y1, x2, y2 = det.bbox
            status_colors = {
                "PASS": (0, 200, 83), "FAIL": (50, 50, 220),
                "REVIEW_REQUIRED": (0, 165, 255), "REWORK": (0, 120, 255),
                "UNVERIFIED": (130, 130, 130),
            }
            color = status_colors.get(decision.status, (200, 120, 40))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            for d in defects:
                dx1, dy1, dx2, dy2 = d.bbox
                cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                cv2.putText(annotated, f"{d.defect_type} {d.confidence:.0%}",
                            (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw RapidOCR bounding boxes and text overlays (matching integrationsbyAbhi/ocrCam.py)
            if ocr_result.detections:
                annotated = ocr.draw_ocr_annotations(annotated, ocr_result.detections, color=(0, 255, 0))

            # Label on image
            b_label = ocr_result.billet_id or det.track_id
            cv2.putText(annotated, f"{b_label} | {decision.status}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            with col_img:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption=f"Billet Inspection View: {b_label}", use_container_width=True)

            with col_info:
                # Status badge
                pass_label = "PASS - QC SUSPENDED (TESTING)" if is_qc_suspended() else "PASS - READY FOR DISPATCH"
                badge_map = {
                    "PASS": (pass_label, "badge-pass"),
                    "FAIL": ("REJECT - DEFECT DETECTED", "badge-fail"),
                    "REVIEW_REQUIRED": ("FLAGGED FOR MANUAL REVIEW", "badge-review"),
                    "REWORK": ("REWORK REQUIRED", "badge-rework"),
                    "UNVERIFIED": ("SPEC UNVERIFIED", "badge-unverified"),
                }
                badge_text, badge_cls = badge_map.get(decision.status, ("UNKNOWN", "badge-unverified"))
                st.markdown(f'<div class="badge {badge_cls}" style="font-size:0.95rem; margin-bottom:12px; display:block; text-align:center;">{badge_text}</div>', unsafe_allow_html=True)

                # Traceability Identification (Billet ID & Heat Number)
                st.markdown("##### 🏷️ Identification & Traceability")
                id_c1, id_c2 = st.columns(2)
                with id_c1:
                    bid = ocr_result.billet_id if ocr_result.is_readable else (default_billet_prefix or "ID_UNREADABLE")
                    st.markdown(f"""
                    <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:8px 12px; margin-bottom:8px;">
                        <div style="font-size:0.72rem; color:#64748B; font-weight:600; text-transform:uppercase;">Billet Number / ID</div>
                        <div style="font-size:1.15rem; font-weight:700; color:#0F172A; font-family:'JetBrains Mono',monospace;">{bid}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with id_c2:
                    hnum = ocr_result.heat_number or (default_heat_num or "—")
                    st.markdown(f"""
                    <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:8px 12px; margin-bottom:8px;">
                        <div style="font-size:0.72rem; color:#64748B; font-weight:600; text-transform:uppercase;">Heat Number</div>
                        <div style="font-size:1.15rem; font-weight:700; color:#0F172A; font-family:'JetBrains Mono',monospace;">{hnum}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if ocr_result.detections:
                    with st.expander(f"🔤 RapidOCR Detected Blocks ({len(ocr_result.detections)})", expanded=False):
                        for i, d in enumerate(ocr_result.detections, 1):
                            st.write(f"- Block {i}: **{d['text']}** (`{d['confidence']:.0%}`)")

                # Defects
                st.markdown("##### ⚠️ Surface Defect Inspection")
                if defects:
                    for d in defects:
                        st.markdown(f"- **{d.defect_type.replace('_', ' ').title()}** &bull; Confidence: `{d.confidence:.0%}`")
                else:
                    st.markdown("<span style='color:#16A34A; font-weight:600;'>✓ No surface defects detected</span>", unsafe_allow_html=True)

                if decision.failed_rules:
                    st.markdown("##### 🚫 Failed QC Rules")
                    for rule in decision.failed_rules:
                        st.markdown(f"- <span style='color:#DC2626;'>{rule}</span>", unsafe_allow_html=True)

            # Record in SQLite database
            record = _build_record(
                det, defects, ocr_result, decision,
                evidence_path, filename, 0, 0.0,
                profile_name, plant_info, shift_info, is_demo,
            )
            rec_id = db.insert_inspection(record)
            _generate_alerts(db, record, decision, det, defects, ocr_result)

            # Instant Excel Export for this inspection
            st.markdown("---")
            ec1, ec2 = st.columns([1, 1])
            with ec1:
                st.success("✅ Record persisted to SQLite database.")
            with ec2:
                try:
                    excel_path = db.export_to_excel()
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            label="📊 Download Excel Report (.xlsx)",
                            data=f.read(),
                            file_name=Path(excel_path).name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )
                except Exception as e:
                    logger.debug(f"Excel export button error: {e}")


# ─── Helpers ──────────────────────────────────────────────────────

def _build_record(
    det, defects, ocr_result, decision,
    evidence_path, source_file, frame_number, timestamp,
    profile_name, plant_info, shift_info, is_demo,
):
    """Build a complete database record dict from inspection results."""
    # Ensure billet_id is cleanly populated
    billet_num = ocr_result.billet_id if ocr_result.is_readable else "ID_UNREADABLE"
    heat_num = ocr_result.heat_number or None

    return {
        "record_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "source_file": str(source_file),
        "source_type": "demo" if is_demo else "upload",
        "video_timestamp": timestamp,
        "frame_number": frame_number,
        "billet_track_id": det.track_id,
        "billet_id": billet_num,
        "heat_number": heat_num,
        "batch_number": ocr_result.batch_number or None,
        "serial_number": ocr_result.serial_number or None,
        "raw_ocr_text": ocr_result.raw_text or None,
        "ocr_confidence": ocr_result.confidence or 0.0,
        "qr_barcode_result": ocr_result.qr_data or ocr_result.barcode_data or None,
        "measured_length_mm": None,
        "measured_width_mm": None,
        "measured_height_mm": None,
        "nominal_length_mm": None,
        "nominal_width_mm": None,
        "nominal_height_mm": None,
        "length_tolerance_mm": None,
        "width_tolerance_mm": None,
        "height_tolerance_mm": None,
        "calibration_status": "NOT_APPLICABLE",
        "measurement_confidence": decision.overall_confidence,
        "defect_detected": 1 if defects else 0,
        "defect_type": defects[0].defect_type if defects else None,
        "defect_confidence": max((d.confidence for d in defects), default=None) if defects else None,
        "defect_count": len(defects),
        "inspection_status": decision.status,
        "failed_rules": "; ".join(decision.failed_rules) if decision.failed_rules else None,
        "review_reason": "; ".join(decision.review_reasons) if decision.review_reasons else None,
        "profile_name": profile_name,
        "evidence_path": evidence_path,
        "is_demo": 1 if is_demo else 0,
        "plant_info": plant_info or None,
        "shift_info": shift_info or None,
    }


def _generate_alerts(db, record, decision, det, defects, ocr_result):
    """Generate alerts for notable conditions."""
    base = {
        "record_id": record["record_id"],
        "billet_track_id": det.track_id,
        "evidence_path": record.get("evidence_path"),
    }

    if decision.status == "FAIL":
        for rule in decision.failed_rules:
            db.insert_alert({
                **base,
                "alert_type": "DEFECT_DETECTED",
                "severity": "CRITICAL",
                "message": rule,
                "reason": rule,
            })

    if defects:
        max_conf = max(d.confidence for d in defects)
        if max_conf > 0.4:
            db.insert_alert({
                **base,
                "alert_type": "DEFECT_DETECTED",
                "severity": "WARNING",
                "message": f"Surface defect: {defects[0].defect_type} ({max_conf:.0%})",
                "detection_info": f"{len(defects)} defect(s)",
                "reason": "Surface anomaly detected",
            })

    if not ocr_result.is_readable:
        db.insert_alert({
            **base,
            "alert_type": "ID_UNREADABLE",
            "severity": "WARNING",
            "message": "Billet marking could not be decoded",
            "reason": "Marking obscured or OCR confidence below threshold",
        })


def _show_results_summary(db, billet_records):
    """Show rich summary table of inspected billets with dimensions in cm and Excel download."""
    st.markdown("### 📋 Batch Inspection Traceability Summary")
    if not billet_records:
        st.info("No billets processed.")
        return

    rows = []
    for track_id, info in billet_records.items():
        record = db.get_inspection(info["record_id"])
        if record:
            rows.append({
                "Track ID": track_id,
                "Billet Number": record.get("billet_id", "—"),
                "Heat Number": record.get("heat_number", "—"),
                "Status": record.get("inspection_status", "—"),
                "Defect": record.get("defect_type") or ("None" if not record.get("defect_detected") else "Yes"),
                "Defect Count": record.get("defect_count", 0),
                "Frames": info["frames_seen"],
            })

    if rows:
        import pandas as pd
        summary_df = pd.DataFrame(rows)
        try:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        except Exception:
            st.table(summary_df)

        st.markdown("<br>", unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            try:
                excel_path = db.export_to_excel()
                with open(excel_path, "rb") as f:
                    st.download_button(
                        label="📊 Download Batch Excel Spreadsheet (.xlsx)",
                        data=f.read(),
                        file_name=Path(excel_path).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Excel export failed: {e}")

        with col_dl2:
            csv_data = summary_df.to_csv(index=False)
            st.download_button(
                label="📄 Download Summary CSV",
                data=csv_data,
                file_name=f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


def _create_demo_frame():
    """Create a synthetic demo frame with a simulated billet, stamped markings, and QR tag."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Clean industrial background
    frame[:] = (35, 32, 28)

    # Ambient noise
    noise = np.random.randint(0, 12, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)

    # Billet geometry
    x1, y1, x2, y2 = 200, 250, 1080, 470
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 85, 80), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (130, 125, 120), 3)

    # Surface texture
    for _ in range(250):
        sx = np.random.randint(x1 + 5, x2 - 5)
        sy = np.random.randint(y1 + 5, y2 - 5)
        intensity = np.random.randint(70, 105)
        cv2.circle(frame, (sx, sy), 1, (intensity, intensity - 5, intensity - 10), -1)

    # Stamped identification plate on billet
    plate_x1, plate_y1 = 640, 290
    plate_x2, plate_y2 = 1040, 420
    cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (60, 55, 50), -1)
    cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (160, 155, 150), 2)

    # Text markings
    billet_id = "BLT-4092"
    heat_number = "HT-88215"
    cv2.putText(frame, f"BILLET: {billet_id}", (plate_x1 + 15, plate_y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 235, 230), 2)
    cv2.putText(frame, f"HEAT: {heat_number}", (plate_x1 + 15, plate_y1 + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 235, 230), 2)

    # Embed 2D QR traceability tag
    try:
        import qrcode
        qr = qrcode.QRCode(box_size=2, border=1)
        qr.add_data(f"BILLET:{billet_id}|HEAT:{heat_number}")
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        qr_bgr = cv2.cvtColor(np.array(qr_img), cv2.COLOR_RGB2BGR)
        qr_resized = cv2.resize(qr_bgr, (80, 80), interpolation=cv2.INTER_NEAREST)
        frame[plate_y1 + 22:plate_y1 + 102, plate_x2 - 95:plate_x2 - 15] = qr_resized
    except Exception:
        pass

    # Simulated surface anomaly (defect)
    cv2.ellipse(frame, (450, 360), (32, 14), 25, 0, 360, (40, 35, 30), -1)

    # Factory banner
    cv2.putText(frame, "SMS-1 / CASTER-A &bull; DEMO INSPECTION", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    return frame
