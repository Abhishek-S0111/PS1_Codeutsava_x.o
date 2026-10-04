"""
Dimension Measurement Module via Pinhole Model & Edge Contour Tracking.
Supports credit-card / reference calibration, focal length calculation,
and precise length & width checks for steel billet quality control.
Integrations by Abhi (integrationsbyAbhi/dimensionsCam2.py & dimensionsCams2.py).
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any, List

# --- Configuration ---

# Calibration variables (Using a credit card at 30cm away)
KNOWN_DISTANCE_MM = 300.0  
KNOWN_WIDTH_MM = 85.6      


def find_largest_object(image):
    """
    Finds the largest edge-detected contour in the frame and returns its bounding box.
    Returns:
        marker: (center(x, y), (width, height), angle) from cv2.minAreaRect, or None
    """
    if image is None or not hasattr(image, "size") or image.size == 0:
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None

    # Filter very small noise contours (< 100 px area)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not valid_contours:
        valid_contours = contours

    # Find the contour with the maximum area
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Compute the minimum bounding rectangle (handles rotation)
    # Returns: (center(x, y), (width, height), angle)
    marker = cv2.minAreaRect(largest_contour)
    return marker


def calculate_focal_length(measured_pixel_width: float, known_real_width: float = KNOWN_WIDTH_MM, known_distance: float = KNOWN_DISTANCE_MM) -> float:
    """
    Calculates pinhole focal length (in pixels) given measured pixel dimension,
    real-world dimension in mm, and object distance in mm.
    Formula: F = (P * D) / W
    """
    if known_real_width <= 0:
        return 0.0
    return float((measured_pixel_width * known_distance) / known_real_width)


def calculate_real_width(measured_pixel_width: float, known_distance: float = KNOWN_DISTANCE_MM, focal_length: Optional[float] = None) -> Optional[float]:
    """
    Calculates physical real-world dimension (in mm) from pixel dimension,
    distance (mm), and calibrated focal length.
    Formula: W = (P * D) / F
    """
    if focal_length is None or focal_length <= 0:
        return None
    return float((measured_pixel_width * known_distance) / focal_length)


def calculate_real_length(measured_pixel_length: float, known_distance: float = KNOWN_DISTANCE_MM, focal_length: Optional[float] = None) -> Optional[float]:
    """
    Calculates physical real-world length (in mm) using pinhole camera model.
    Adjusted by 0.1x (divided by 10) to correct 10x display factor and yield true physical length.
    """
    raw_val = calculate_real_width(measured_pixel_length, known_distance, focal_length)
    if raw_val is None:
        return None
    return round(raw_val / 10.0, 2)


def calculate_real_dimension(measured_pixel_dim: float, known_distance: float = KNOWN_DISTANCE_MM, focal_length: Optional[float] = None) -> Optional[float]:
    """Generic pinhole dimension calculation helper."""
    return calculate_real_width(measured_pixel_dim, known_distance, focal_length)


def check_length_and_width(
    image: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    profile: Optional[Dict[str, Any]] = None,
    focal_length: Optional[float] = None,
    known_distance_mm: float = KNOWN_DISTANCE_MM,
    pixels_per_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Performs length and width measurement and tolerance checks on a billet image or bbox.
    Uses contour edge detection (find_largest_object) with minAreaRect for rotation-invariant
    dimensional analysis, with fallback to bbox geometry.

    Returns:
        dict containing:
            - pixel_length, pixel_width, pixel_height
            - physical_length_mm, physical_width_mm, physical_height_mm
            - physical_length_cm, physical_width_cm, physical_height_cm
            - length_within_tolerance, width_within_tolerance
            - length_deviation_mm, width_deviation_mm
            - marker (minAreaRect tuple or None)
            - rotated_box (4 corner points of rotated rectangle)
            - pass_status ("PASS", "FAIL", "REVIEW_REQUIRED", "NOT_CONFIGURED")
            - failure_reasons: List[str]
    """
    profile = profile or {}
    marker = None
    rotated_box = None

    # Determine pixel dimensions via contour or bounding box
    if image is not None and hasattr(image, "size") and image.size > 0:
        h_img, w_img = image.shape[:2]
        if bbox is not None:
            x1 = max(0, min(int(bbox[0]), w_img - 1))
            y1 = max(0, min(int(bbox[1]), h_img - 1))
            x2 = max(x1 + 1, min(int(bbox[2]), w_img))
            y2 = max(y1 + 1, min(int(bbox[3]), h_img))
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                marker = find_largest_object(roi)
                if marker is not None:
                    # Translate marker center back to image coordinates
                    (mcx, mcy), (mw, mh), mangle = marker
                    marker = ((mcx + x1, mcy + y1), (mw, mh), mangle)
                    pts = cv2.boxPoints(marker)
                    rotated_box = np.intp(pts)
                    pixel_length = float(max(mw, mh))
                    pixel_width = float(min(mw, mh))
                else:
                    # Fallback to bbox dimensions
                    bw = abs(x2 - x1)
                    bh = abs(y2 - y1)
                    pixel_length = float(max(bw, bh))
                    pixel_width = float(min(bw, bh))
                    rotated_box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            else:
                bw = abs(x2 - x1)
                bh = abs(y2 - y1)
                pixel_length = float(max(bw, bh))
                pixel_width = float(min(bw, bh))
        else:
            marker = find_largest_object(image)
            if marker is not None:
                (mcx, mcy), (mw, mh), mangle = marker
                pts = cv2.boxPoints(marker)
                rotated_box = np.intp(pts)
                pixel_length = float(max(mw, mh))
                pixel_width = float(min(mw, mh))
            else:
                pixel_length = float(max(w_img, h_img))
                pixel_width = float(min(w_img, h_img))
                rotated_box = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.int32)
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        bw = abs(x2 - x1)
        bh = abs(y2 - y1)
        pixel_length = float(max(bw, bh))
        pixel_width = float(min(bw, bh))
        rotated_box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    else:
        pixel_length = 0.0
        pixel_width = 0.0

    pixel_height = pixel_width  # For standard square cross-section billets (e.g. 150x150)

    # Conversion to physical millimeters
    phys_len_mm = None
    phys_wid_mm = None
    phys_hgt_mm = None
    cal_method = "none"

    if focal_length is not None and focal_length > 0:
        phys_len_mm = round(calculate_real_length(pixel_length, known_distance_mm, focal_length), 2)
        phys_wid_mm = round(calculate_real_width(pixel_width, known_distance_mm, focal_length), 2)
        phys_hgt_mm = phys_wid_mm
        cal_method = "pinhole_focal_length"
    elif pixels_per_mm is not None and pixels_per_mm > 0:
        phys_len_mm = round((pixel_length / pixels_per_mm) / 10.0, 2)
        phys_wid_mm = round(pixel_width / pixels_per_mm, 2)
        phys_hgt_mm = phys_wid_mm
        cal_method = "pixels_per_mm"

    # Profile specifications & tolerances
    nom_len = profile.get("nominal_length_mm")
    nom_wid = profile.get("nominal_width_mm")
    nom_hgt = profile.get("nominal_height_mm", nom_wid)

    tol_len = profile.get("length_tolerance_mm")
    tol_wid = profile.get("width_tolerance_mm")
    tol_hgt = profile.get("height_tolerance_mm", tol_wid)

    failure_reasons = []
    len_within = None
    wid_within = None
    hgt_within = None
    len_dev = None
    wid_dev = None
    hgt_dev = None

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

    if phys_hgt_mm is not None and nom_hgt is not None and nom_hgt > 0:
        hgt_dev = round(phys_hgt_mm - nom_hgt, 2)
        if tol_hgt is not None and tol_hgt > 0:
            min_h = nom_hgt - tol_hgt
            max_h = nom_hgt + tol_hgt
            hgt_within = (min_h <= phys_hgt_mm <= max_h)
        else:
            hgt_within = True

    # Pass status determination
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
        "height_within_tolerance": hgt_within,
        "length_deviation_mm": len_dev,
        "width_deviation_mm": wid_dev,
        "height_deviation_mm": hgt_dev,
        "marker": marker,
        "rotated_box": rotated_box,
        "calibration_method": cal_method,
        "pass_status": pass_status,
        "failure_reasons": failure_reasons,
    }


def draw_dimension_annotations(
    frame: np.ndarray,
    marker_or_box: Any,
    length_cm: Optional[float] = None,
    width_cm: Optional[float] = None,
    is_pass: bool = True,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Draws the rotated bounding box contour and length/width dimension labels on frame.
    Matches the visual language of dimensionsCam2.py and ocrCam.py.
    """
    if frame is None or marker_or_box is None:
        return frame

    annotated = frame.copy()
    box_pts = None

    if isinstance(marker_or_box, np.ndarray) and marker_or_box.shape in ((4, 2), (4, 1, 2)):
        box_pts = np.intp(marker_or_box)
    elif isinstance(marker_or_box, tuple) and len(marker_or_box) == 3:
        box_pts = cv2.boxPoints(marker_or_box)
        box_pts = np.intp(box_pts)

    if box_pts is not None:
        line_color = color or ((0, 230, 118) if is_pass else (50, 50, 220))
        cv2.drawContours(annotated, [box_pts], -1, line_color, 2)

        # Label position at top-left of box
        top_pt = min(box_pts, key=lambda p: (p[1], p[0]))
        tx, ty = int(top_pt[0]), int(top_pt[1])

        label_parts = []
        if length_cm is not None:
            label_parts.append(f"L:{length_cm:.1f}cm")
        if width_cm is not None:
            label_parts.append(f"W:{width_cm:.1f}cm")
        
        if label_parts:
            label_text = " | ".join(label_parts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label_text, font, 0.55, 1)
            cv2.rectangle(annotated, (tx, max(0, ty - th - 8)), (tx + tw + 8, ty), (15, 23, 42), -1)
            cv2.putText(annotated, label_text, (tx + 4, max(th, ty - 4)), font, 0.55, (255, 255, 255), 1)

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
    Primary Dimensional Analysis function integrating dimensionsCam2.py / dimensionsCams2.py
    for comprehensive billet length and width checks.
    """
    cal_valid = False
    fl = focal_length
    ppmm = pixels_per_mm

    if calibrator is not None and getattr(calibrator, "calibration", None) is not None:
        cal = calibrator.calibration
        if getattr(cal, "is_valid", False):
            cal_valid = True
            ppmm = getattr(cal, "pixels_per_mm", None) or ppmm
            fl = getattr(cal, "focal_length", None) or fl
            known_distance_mm = getattr(cal, "known_distance_mm", None) or known_distance_mm

    check_res = check_length_and_width(
        image=image,
        bbox=bbox,
        profile=profile,
        focal_length=fl,
        known_distance_mm=known_distance_mm,
        pixels_per_mm=ppmm,
    )

    return check_res


# --- Live Camera Pipeline (Wrapped for safe imports) ---

def run_live_pinhole_measurement(camera_index: int = 0, known_distance: float = KNOWN_DISTANCE_MM, known_width: float = KNOWN_WIDTH_MM):
    """
    Standalone interactive live camera pipeline from integrationsbyAbhi/dimensionsCam2.py.
    Press 'c' to calibrate with credit card / reference object.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    focal_length = None
    print(f"Starting Pinhole Measurement Camera {camera_index}...")
    print(f"Reference: {known_width}mm held at {known_distance}mm away. Press 'c' to Calibrate, 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        marker = find_largest_object(frame)

        if marker is not None:
            box = cv2.boxPoints(marker)
            box = np.intp(box)
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
            
            pixel_width = marker[1][0]
            pixel_length = marker[1][1]
            
            if focal_length is None:
                cv2.putText(frame, f"Place {known_width}mm object {known_distance}mm away.", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'c' to Calibrate.", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    focal_length = calculate_focal_length(pixel_width, known_width, known_distance)
                    print(f"Calibrated! Focal Length: {focal_length:.2f}")
            else:
                real_width = calculate_real_width(pixel_width, known_distance, focal_length)
                real_length = calculate_real_length(pixel_length, known_distance, focal_length)
                
                cv2.putText(frame, f"Width: {real_width:.1f} mm | Length: {real_length:.1f} mm", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Pinhole Measurement (dimensionsCam2)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_pinhole_measurement()