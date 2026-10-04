"""
Dimensional Analysis via YOLO Object Detection, Center-Crosshair Lock, and Temporal Smoothing (EMA).
Integrations by Abhi (integrationsbyAbhi/dimensionsCamYOLO.py).
Supports YOLOv8 detection, center-distance lock, exponential moving average smoothing,
pinhole camera focal length calibration, and calibrated length/width tolerance checks.
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any, List

# --- Configuration ---

KNOWN_DISTANCE_MM = 300.0  
KNOWN_WIDTH_MM = 75.0       # Standard mobile phone width (COCO class 67: cell phone)
CALIBRATION_CLASS_ID = 67   # COCO class for cell phone

# --- Temporal Smoothing (EMA) ---
alpha = 0.15  # Lower = smoother measurements, but slightly more lag
_global_smoothed_width = None
_global_smoothed_length = None

# Model cache
_cached_yolo_model = None


def get_yolo_model(model_name: str = "yolov8s.pt"):
    """Loads and caches YOLO model with fallback handling."""
    global _cached_yolo_model
    if _cached_yolo_model is not None:
        return _cached_yolo_model
    try:
        from ultralytics import YOLO
        # Attempt loading requested model (yolov8s.pt or local fallback)
        try:
            _cached_yolo_model = YOLO(model_name)
        except Exception:
            _cached_yolo_model = YOLO("yolov8n.pt")
        return _cached_yolo_model
    except Exception as e:
        return None


def calculate_focal_length(pixel_w: float, real_w: float = KNOWN_WIDTH_MM, distance: float = KNOWN_DISTANCE_MM) -> float:
    """Calculates pinhole focal length (pixels): F = (P * D) / W"""
    if real_w <= 0:
        return 0.0
    return float((pixel_w * distance) / real_w)


def calculate_real_width(pixel_w: float, distance: float = KNOWN_DISTANCE_MM, focal_length: Optional[float] = None) -> Optional[float]:
    """Calculates real-world width (mm): W = (P * D) / F"""
    if focal_length is None or focal_length <= 0:
        return None
    return float((pixel_w * distance) / focal_length)


def calculate_real_length(pixel_l: float, distance: float = KNOWN_DISTANCE_MM, focal_length: Optional[float] = None) -> Optional[float]:
    """
    Calculates physical real-world length (in mm).
    """
    raw_val = calculate_real_width(pixel_l, distance, focal_length)
    if raw_val is None:
        return None
    return round(raw_val, 2)


def calculate_real_dimension(pixel_dim: float, distance: float = KNOWN_DISTANCE_MM, focal_length: Optional[float] = None) -> Optional[float]:
    """Generic dimension helper."""
    return calculate_real_width(pixel_dim, distance, focal_length)


def apply_ema(current_val: float, prev_val: Optional[float], smoothing_factor: float = alpha) -> float:
    """Applies Exponential Moving Average smoothing: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}"""
    if prev_val is None:
        return float(current_val)
    return float((smoothing_factor * current_val) + ((1.0 - smoothing_factor) * prev_val))


def detect_best_target_box(
    frame: np.ndarray,
    model: Any = None,
    target_class_id: Optional[int] = None,
    conf_threshold: float = 0.45,
    center_lock: bool = True,
) -> Optional[Tuple[int, int, int, int]]:
    """
    YOLO Center-Crosshair Lock:
    Runs YOLO prediction, finds objects matching class/confidence,
    and locks onto the detection closest to the center of the camera frame.
    Returns:
        (x1, y1, x2, y2) or None
    """
    if frame is None or frame.size == 0:
        return None

    if model is None:
        model = get_yolo_model("yolov8s.pt")
    if model is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    frame_center = (frame_width // 2, frame_height // 2)

    try:
        results = model.predict(frame, stream=False, verbose=False, imgsz=640)
    except Exception:
        return None

    best_box = None
    min_dist_to_center = float("inf")

    for result in results:
        boxes = getattr(result, "boxes", [])
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Filter by class if specified (e.g. CALIBRATION_CLASS_ID = 67 for phone)
            if (target_class_id is None or class_id == target_class_id) and confidence >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if center_lock:
                    box_center_x = (x1 + x2) // 2
                    box_center_y = (y1 + y2) // 2
                    dist = np.sqrt((box_center_x - frame_center[0]) ** 2 + (box_center_y - frame_center[1]) ** 2)

                    if dist < min_dist_to_center:
                        min_dist_to_center = dist
                        best_box = (x1, y1, x2, y2)
                else:
                    return (x1, y1, x2, y2)

    return best_box


def check_length_and_width(
    image: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    profile: Optional[Dict[str, Any]] = None,
    focal_length: Optional[float] = None,
    known_distance_mm: float = KNOWN_DISTANCE_MM,
    pixels_per_mm: Optional[float] = None,
    use_yolo_detect: bool = True,
    model: Any = None,
) -> Dict[str, Any]:
    """
    Core Dimensional Analysis function using integrationsbyAbhi/dimensionsCamYOLO.py.
    Leverages YOLO Center-Crosshair Lock, EMA temporal smoothing, and pinhole tolerance checking.

    Returns:
        dict with pixel dimensions, physical length/width in mm and cm,
        deviation metrics, tolerance pass/fail, and target lock bounding box.
    """
    global _global_smoothed_width, _global_smoothed_length
    profile = profile or {}
    locked_box = None
    target_locked = False

    # 1. Determine bounding box (via YOLO Center-Crosshair Lock or user bbox)
    if bbox is not None:
        locked_box = bbox
    elif image is not None and use_yolo_detect:
        yolo_box = detect_best_target_box(image, model=model, target_class_id=None, conf_threshold=0.40)
        if yolo_box is not None:
            locked_box = yolo_box
            target_locked = True

    # Fallback to image extents if no box found
    if locked_box is None and image is not None:
        h_img, w_img = image.shape[:2]
        locked_box = (0, 0, w_img, h_img)

    if locked_box is not None:
        x1, y1, x2, y2 = locked_box
        cur_w = float(abs(x2 - x1))
        cur_h = float(abs(y2 - y1))
        raw_pixel_length = max(cur_w, cur_h)
        raw_pixel_width = min(cur_w, cur_h)
    else:
        raw_pixel_length = 0.0
        raw_pixel_width = 0.0

    # 2. Apply Exponential Moving Average (EMA) Temporal Smoothing to kill jitter
    smoothed_w = apply_ema(raw_pixel_width, _global_smoothed_width, alpha)
    smoothed_l = apply_ema(raw_pixel_length, _global_smoothed_length, alpha)
    _global_smoothed_width = smoothed_w
    _global_smoothed_length = smoothed_l

    pixel_width = smoothed_w
    pixel_length = smoothed_l
    pixel_height = pixel_width

    # 3. Physical dimension conversion (Pinhole model or pixels_per_mm)
    phys_len_mm = None
    phys_wid_mm = None
    phys_hgt_mm = None
    cal_method = "none"

    if focal_length is not None and focal_length > 0:
        phys_len_mm = round(calculate_real_length(pixel_length, known_distance_mm, focal_length), 2)
        raw_w = calculate_real_width(pixel_width, known_distance_mm, focal_length)
        phys_wid_mm = round(raw_w, 2) if raw_w is not None else None
        phys_hgt_mm = phys_wid_mm
        cal_method = "yolo_pinhole_focal_length"
    elif pixels_per_mm is not None and pixels_per_mm > 0:
        phys_len_mm = round((pixel_length / pixels_per_mm), 2)
        phys_wid_mm = round((pixel_width / pixels_per_mm), 2)
        phys_hgt_mm = phys_wid_mm
        cal_method = "yolo_pixels_per_mm"

    # 4. Profile specifications & tolerance checks
    nom_len = profile.get("nominal_length_mm")
    nom_wid = profile.get("nominal_width_mm")
    nom_hgt = profile.get("nominal_height_mm", nom_wid)

    tol_len = profile.get("length_tolerance_mm")
    tol_wid = profile.get("width_tolerance_mm")
    tol_hgt = profile.get("height_tolerance_mm", tol_wid)

    failure_reasons = []
    len_within = None
    wid_within = None
    len_dev = None
    wid_dev = None

    if phys_len_mm is not None and nom_len is not None and nom_len > 0:
        target_nom_l = (nom_len / 10.0) if (abs(phys_len_mm - (nom_len / 10.0)) < abs(phys_len_mm - nom_len)) else nom_len
        target_tol_l = (tol_len / 10.0) if (tol_len and target_nom_l == nom_len / 10.0) else tol_len
        len_dev = round(phys_len_mm - target_nom_l, 2)
        if target_tol_l is not None and target_tol_l > 0:
            min_l = target_nom_l - target_tol_l
            max_l = target_nom_l + target_tol_l
            len_within = (min_l <= phys_len_mm <= max_l)
            if not len_within:
                failure_reasons.append(
                    f"Length {phys_len_mm:.1f}mm out of tolerance (nominal {target_nom_l}±{target_tol_l}mm, dev {len_dev:+.1f}mm)"
                )
        else:
            len_within = True

    if phys_wid_mm is not None and nom_wid is not None and nom_wid > 0:
        wid_dev = round(phys_wid_mm - nom_wid, 2)
        if tol_wid is not None and tol_wid > 0:
            min_w = nom_wid - tol_wid
            max_w = nom_wid + tol_wid
            wid_within = (min_w <= phys_wid_mm <= max_w)
            if not wid_within:
                failure_reasons.append(
                    f"Width {phys_wid_mm:.1f}mm out of tolerance (nominal {nom_wid}±{tol_wid}mm, dev {wid_dev:+.1f}mm)"
                )
        else:
            wid_within = True

    # 5. Determine overall pass status
    if phys_len_mm is None or phys_wid_mm is None:
        pass_status = "REVIEW_REQUIRED"
    elif failure_reasons:
        pass_status = "FAIL"
    elif (len_within is True or len_within is None) and (wid_within is True or wid_within is None):
        pass_status = "PASS"
    else:
        pass_status = "REVIEW_REQUIRED"

    return {
        "pixel_length": pixel_length,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "physical_length_mm": phys_len_mm,
        "physical_width_mm": phys_wid_mm,
        "physical_height_mm": phys_hgt_mm,
        "physical_length_cm": round(phys_len_mm / 10.0, 2) if phys_len_mm is not None else None,
        "physical_width_cm": round(phys_wid_mm / 10.0, 2) if phys_wid_mm is not None else None,
        "physical_height_cm": round(phys_hgt_mm / 10.0, 2) if phys_hgt_mm is not None else None,
        "length_within_tolerance": len_within,
        "width_within_tolerance": wid_within,
        "length_deviation_mm": len_dev,
        "width_deviation_mm": wid_dev,
        "locked_box": locked_box,
        "target_locked": target_locked,
        "calibration_method": cal_method,
        "pass_status": pass_status,
        "failure_reasons": failure_reasons,
    }


def draw_yolo_dimension_annotations(
    frame: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    length_cm: Optional[float] = None,
    width_cm: Optional[float] = None,
    is_pass: bool = True,
    draw_crosshair: bool = True,
) -> np.ndarray:
    """
    Renders Center-Crosshair Lock and dimension badges matching dimensionsCamYOLO.py.
    """
    if frame is None:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    frame_center = (w // 2, h // 2)

    # 1. Draw camera center crosshair
    if draw_crosshair:
        cv2.circle(annotated, frame_center, 5, (0, 255, 255), -1)
        cv2.line(annotated, (frame_center[0] - 15, frame_center[1]), (frame_center[0] + 15, frame_center[1]), (0, 255, 255), 1)
        cv2.line(annotated, (frame_center[0], frame_center[1] - 15), (frame_center[0], frame_center[1] + 15), (0, 255, 255), 1)

    # 2. Draw locked target box
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        box_color = (0, 230, 118) if is_pass else (50, 50, 220)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(annotated, "Target Locked (YOLO)", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 0), 2)

        # Dimension labels
        dim_text = []
        if length_cm is not None:
            dim_text.append(f"L: {length_cm:.1f}cm")
        if width_cm is not None:
            dim_text.append(f"W: {width_cm:.1f}cm")

        if dim_text:
            label = " | ".join(dim_text)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
            cv2.rectangle(annotated, (x1, y2 + 5), (x1 + tw + 10, y2 + th + 15), (15, 23, 42), -1)
            cv2.putText(annotated, label, (x1 + 5, y2 + th + 10), font, 0.6, (255, 255, 255), 2)

    return annotated


def dimensional_analysis(
    image: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    profile: Optional[Dict[str, Any]] = None,
    calibrator: Any = None,
    known_distance_mm: float = KNOWN_DISTANCE_MM,
    focal_length: Optional[float] = None,
    pixels_per_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Top-level Dimensional Analysis function using dimensionsCamYOLO.py.
    """
    fl = focal_length
    ppmm = pixels_per_mm

    if calibrator is not None and getattr(calibrator, "calibration", None) is not None:
        cal = calibrator.calibration
        if getattr(cal, "is_valid", False):
            ppmm = getattr(cal, "pixels_per_mm", None) or ppmm
            fl = getattr(cal, "focal_length", None) or fl
            known_distance_mm = getattr(cal, "known_distance_mm", None) or known_distance_mm

    return check_length_and_width(
        image=image,
        bbox=bbox,
        profile=profile,
        focal_length=fl,
        known_distance_mm=known_distance_mm,
        pixels_per_mm=ppmm,
    )


# --- Standalone Live Webcam Pipeline ---

def run_live_yolo_measurement(camera_index: int = 0, model_name: str = "yolov8s.pt"):
    """
    Live Camera pipeline from integrationsbyAbhi/dimensionsCamYOLO.py.
    Uses YOLOv8s, Center-Crosshair Lock, and Exponential Moving Average smoothing.
    Press 'c' to Calibrate, 'q' to Quit.
    """
    model = get_yolo_model(model_name)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    focal_length = None
    smoothed_pixel_width = None

    print(f"Starting YOLO Measurement with {model_name} on Camera {camera_index}...")
    print(f"Calibration reference: {KNOWN_WIDTH_MM}mm at {KNOWN_DISTANCE_MM}mm. Press 'c' to Calibrate, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width // 2, frame_height // 2)

        best_box = detect_best_target_box(frame, model=model, target_class_id=CALIBRATION_CLASS_ID, conf_threshold=0.45)

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cur_w = float(x2 - x1)
            smoothed_pixel_width = apply_ema(cur_w, smoothed_pixel_width, alpha)

            # Draw Crosshair and Box
            cv2.circle(frame, frame_center, 5, (0, 255, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Target Locked", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if focal_length is None:
                cv2.putText(frame, f"Hold phone {KNOWN_DISTANCE_MM}mm away.", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'c' to Calibrate.", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if cv2.waitKey(1) & 0xFF == ord("c"):
                    focal_length = calculate_focal_length(smoothed_pixel_width, KNOWN_WIDTH_MM, KNOWN_DISTANCE_MM)
                    print(f"Calibrated! Focal Length: {focal_length:.2f}")
            else:
                real_w = calculate_real_width(smoothed_pixel_width, KNOWN_DISTANCE_MM, focal_length)
                cv2.putText(frame, f"Width: {real_w:.1f} mm", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Robust YOLO Measurement", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_yolo_measurement()