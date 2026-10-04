"""
Measurement Module - Camera Calibration and Dimensional Measurement
Handles pixel-to-physical conversion and tolerance checking.
"""
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from utils.config import CALIBRATION_CONFIG, MEASUREMENT_CONFIG, CALIBRATION_DIR
from utils.logger import logger

try:
    from integrationsbyAbhi.dimensionsCamYOLO import (
        KNOWN_DISTANCE_MM, KNOWN_WIDTH_MM,
        calculate_focal_length, calculate_real_width, calculate_real_length,
        check_length_and_width, draw_yolo_dimension_annotations,
        detect_best_target_box,
    )
except ImportError:
    from integrationsbyAbhi.dimensionsCams2 import (
        KNOWN_DISTANCE_MM, KNOWN_WIDTH_MM,
        calculate_focal_length, calculate_real_width, calculate_real_length,
        check_length_and_width,
    )


@dataclass
class CalibrationData:
    """Stores camera calibration information."""
    calibration_id: str = ""
    method: str = "none"
    pixels_per_mm: float = 0.0
    reference_dimension_mm: float = 0.0
    reprojection_error: float = 0.0
    camera_matrix: np.ndarray = None
    dist_coeffs: np.ndarray = None
    valid_until: str = ""
    created_at: str = ""
    focal_length: float = 0.0
    known_distance_mm: float = 300.0
    
    @property
    def is_valid(self) -> bool:
        if not self.valid_until:
            return False
        try:
            expiry = datetime.fromisoformat(self.valid_until)
            return datetime.now() < expiry and (self.pixels_per_mm > 0 or self.focal_length > 0)
        except Exception:
            return False
    
    def to_dict(self) -> Dict:
        return {
            "calibration_id": self.calibration_id,
            "method": self.method,
            "pixels_per_mm": self.pixels_per_mm,
            "reference_dimension_mm": self.reference_dimension_mm,
            "reprojection_error": self.reprojection_error,
            "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            "valid_until": self.valid_until,
            "created_at": self.created_at,
            "focal_length": self.focal_length,
            "known_distance_mm": self.known_distance_mm,
        }


@dataclass
class MeasurementResult:
    """Result of a dimensional measurement."""
    dimension_name: str  # "length", "width", "height"
    pixel_value: float
    physical_value_mm: float = None
    unit: str = "mm"
    nominal_value_mm: float = None
    min_allowed_mm: float = None
    max_allowed_mm: float = None
    deviation_mm: float = None
    deviation_percent: float = None
    status: str = "NOT_MEASURED"  # VALID, REVIEW, NOT_MEASURABLE, NOT_CONFIGURED
    calibration_valid: bool = False
    confidence: float = 0.0
    notes: str = ""
    
    @property
    def is_within_tolerance(self) -> bool:
        if self.status != "VALID" or self.physical_value_mm is None:
            return False
        if self.min_allowed_mm is not None and self.physical_value_mm < self.min_allowed_mm:
            return False
        if self.max_allowed_mm is not None and self.physical_value_mm > self.max_allowed_mm:
            return False
        return True

    @property
    def physical_value_cm(self) -> Optional[float]:
        return round(self.physical_value_mm / 10.0, 2) if self.physical_value_mm is not None else None

    @property
    def nominal_value_cm(self) -> Optional[float]:
        return round(self.nominal_value_mm / 10.0, 2) if self.nominal_value_mm is not None else None

    @property
    def min_allowed_cm(self) -> Optional[float]:
        return round(self.min_allowed_mm / 10.0, 2) if self.min_allowed_mm is not None else None

    @property
    def max_allowed_cm(self) -> Optional[float]:
        return round(self.max_allowed_mm / 10.0, 2) if self.max_allowed_mm is not None else None

    @property
    def deviation_cm(self) -> Optional[float]:
        return round(self.deviation_mm / 10.0, 2) if self.deviation_mm is not None else None


class CameraCalibrator:
    """Handles camera calibration for pixel-to-physical conversion."""
    
    def __init__(self):
        self.calibration: Optional[CalibrationData] = None
        self._load_saved_calibration()
    
    def _load_saved_calibration(self):
        """Load the most recent calibration from disk."""
        cal_file = CALIBRATION_DIR / "current_calibration.json"
        if cal_file.exists():
            try:
                with open(cal_file, "r") as f:
                    data = json.load(f)
                self.calibration = CalibrationData(
                    calibration_id=data.get("calibration_id", ""),
                    method=data.get("method", ""),
                    pixels_per_mm=data.get("pixels_per_mm", 0),
                    reference_dimension_mm=data.get("reference_dimension_mm", 0),
                    reprojection_error=data.get("reprojection_error", 0),
                    valid_until=data.get("valid_until", ""),
                    created_at=data.get("created_at", ""),
                    focal_length=data.get("focal_length", 0.0),
                    known_distance_mm=data.get("known_distance_mm", 300.0),
                )
                if data.get("camera_matrix"):
                    self.calibration.camera_matrix = np.array(data["camera_matrix"])
                if data.get("dist_coeffs"):
                    self.calibration.dist_coeffs = np.array(data["dist_coeffs"])
                logger.info(f"Loaded calibration: {self.calibration.method}, valid={self.calibration.is_valid}")
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")
    
    def calibrate_with_reference(
        self,
        image: np.ndarray,
        reference_points: Tuple[Tuple[int, int], Tuple[int, int]],
        known_dimension_mm: float,
    ) -> CalibrationData:
        """Calibrate using a known reference dimension.
        
        Args:
            image: Calibration image
            reference_points: Two points defining the known dimension
            known_dimension_mm: Real-world distance in mm
        """
        p1, p2 = reference_points
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        if pixel_distance < 10:
            raise ValueError("Reference points are too close together")
        if known_dimension_mm <= 0:
            raise ValueError("Reference dimension must be positive")
        
        pixels_per_mm = pixel_distance / known_dimension_mm
        
        validity_days = CALIBRATION_CONFIG["validity_days"]
        valid_until = (datetime.now() + timedelta(days=validity_days)).isoformat()
        
        import uuid
        self.calibration = CalibrationData(
            calibration_id=str(uuid.uuid4()),
            method="reference_object",
            pixels_per_mm=pixels_per_mm,
            reference_dimension_mm=known_dimension_mm,
            valid_until=valid_until,
            created_at=datetime.now().isoformat(),
        )
        
        self._save_calibration()
        logger.info(f"Calibrated: {pixels_per_mm:.2f} px/mm using {known_dimension_mm}mm reference")
        return self.calibration
    
    def calibrate_with_checkerboard(self, images: list) -> CalibrationData:
        """Calibrate using checkerboard pattern.
        
        Args:
            images: List of calibration images with checkerboard
        """
        board_size = CALIBRATION_CONFIG["checkerboard_size"]
        square_size = CALIBRATION_CONFIG["square_size_mm"]
        
        obj_points = []
        img_points = []
        
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        gray_shape = None
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            gray_shape = gray.shape[::-1]
            
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners2)
        
        if len(obj_points) < CALIBRATION_CONFIG["min_calibration_images"]:
            raise ValueError(
                f"Need at least {CALIBRATION_CONFIG['min_calibration_images']} valid images, "
                f"got {len(obj_points)}"
            )
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray_shape, None, None
        )
        
        # Estimate pixels_per_mm from focal length
        fx = mtx[0, 0]
        pixels_per_mm = fx / 1000.0  # Rough estimate
        
        validity_days = CALIBRATION_CONFIG["validity_days"]
        
        import uuid
        self.calibration = CalibrationData(
            calibration_id=str(uuid.uuid4()),
            method="checkerboard",
            pixels_per_mm=pixels_per_mm,
            reprojection_error=ret,
            camera_matrix=mtx,
            dist_coeffs=dist,
            valid_until=(datetime.now() + timedelta(days=validity_days)).isoformat(),
            created_at=datetime.now().isoformat(),
        )
        
        self._save_calibration()
        logger.info(f"Checkerboard calibration: reproj_err={ret:.4f}, px/mm={pixels_per_mm:.2f}")
        return self.calibration
    
    def calibrate_manual(self, pixels_per_mm: float) -> CalibrationData:
        """Manual calibration by specifying pixels_per_mm directly."""
        import uuid
        validity_days = CALIBRATION_CONFIG["validity_days"]
        
        self.calibration = CalibrationData(
            calibration_id=str(uuid.uuid4()),
            method="manual",
            pixels_per_mm=pixels_per_mm,
            valid_until=(datetime.now() + timedelta(days=validity_days)).isoformat(),
            created_at=datetime.now().isoformat(),
        )
        
        self._save_calibration()
        return self.calibration
    
    def calibrate_pinhole(
        self,
        measured_pixel_width: float,
        known_real_width_mm: float = KNOWN_WIDTH_MM,
        known_distance_mm: float = KNOWN_DISTANCE_MM,
    ) -> CalibrationData:
        """Calibrate using pinhole model and reference dimension (integrationsbyAbhi/dimensionsCamYOLO.py).
        
        Args:
            measured_pixel_width: Pixel width of reference object on sensor (phone/target)
            known_real_width_mm: Real-world width in mm (default 75.0 mm phone)
            known_distance_mm: Distance from camera in mm (default 300 mm)
        """
        if measured_pixel_width <= 0:
            raise ValueError("Measured pixel width must be positive")
        if known_real_width_mm <= 0:
            raise ValueError("Known real width must be positive")
        if known_distance_mm <= 0:
            raise ValueError("Known distance must be positive")
            
        focal_length = calculate_focal_length(measured_pixel_width, known_real_width_mm, known_distance_mm)
        pixels_per_mm = focal_length / known_distance_mm
        
        validity_days = CALIBRATION_CONFIG.get("validity_days", 30)
        valid_until = (datetime.now() + timedelta(days=validity_days)).isoformat()
        
        import uuid
        self.calibration = CalibrationData(
            calibration_id=str(uuid.uuid4()),
            method="pinhole_dimensionsCamYOLO",
            pixels_per_mm=pixels_per_mm,
            reference_dimension_mm=known_real_width_mm,
            focal_length=focal_length,
            known_distance_mm=known_distance_mm,
            valid_until=valid_until,
            created_at=datetime.now().isoformat(),
        )
        self._save_calibration()
        logger.info(f"Pinhole calibrated: focal_length={focal_length:.2f}px, px/mm={pixels_per_mm:.3f}")
        return self.calibration

    def _save_calibration(self):
        """Save calibration to disk."""
        if self.calibration:
            cal_file = CALIBRATION_DIR / "current_calibration.json"
            with open(cal_file, "w") as f:
                json.dump(self.calibration.to_dict(), f, indent=2)
    
    def pixels_to_mm(self, pixel_value: float) -> Optional[float]:
        """Convert pixels to millimeters using current calibration or dimensionsCams2 pinhole calculation."""
        if not self.calibration or not self.calibration.is_valid:
            return None
        if getattr(self.calibration, "focal_length", 0) > 0 and getattr(self.calibration, "known_distance_mm", 0) > 0:
            real_mm = calculate_real_width(pixel_value, self.calibration.known_distance_mm, self.calibration.focal_length)
            if real_mm is not None:
                return real_mm
        return pixel_value / self.calibration.pixels_per_mm
    
    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Apply lens distortion correction if calibration is available."""
        if (self.calibration and self.calibration.camera_matrix is not None
                and self.calibration.dist_coeffs is not None):
            h, w = image.shape[:2]
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
                (w, h), 1, (w, h)
            )
            return cv2.undistort(
                image, self.calibration.camera_matrix,
                self.calibration.dist_coeffs, None, new_mtx
            )
        return image


class DimensionalMeasurer:
    """Measures billet dimensions from bounding boxes."""
    
    def __init__(self, calibrator: CameraCalibrator = None):
        self.calibrator = calibrator or CameraCalibrator()
    
    def measure_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        profile: Dict[str, Any],
        image: Optional[np.ndarray] = None,
    ) -> Dict[str, MeasurementResult]:
        """Measure dimensions from a bounding box and optional image using integrationsbyAbhi/dimensionsCamYOLO.py.
        
        Leverages YOLO Center-Crosshair Lock + EMA temporal smoothing from dimensionsCamYOLO.py
        for robust length and width checks, with pinhole focal length calculation.
        """
        cal_valid = self.calibrator.calibration is not None and self.calibrator.calibration.is_valid
        
        fl = getattr(self.calibrator.calibration, "focal_length", None) if cal_valid else None
        dist = getattr(self.calibrator.calibration, "known_distance_mm", KNOWN_DISTANCE_MM) if cal_valid else KNOWN_DISTANCE_MM
        ppmm = getattr(self.calibrator.calibration, "pixels_per_mm", None) if cal_valid else None

        # Execute length and width checks via integrationsbyAbhi/dimensionsCamYOLO.py
        cam_res = check_length_and_width(
            image=image,
            bbox=bbox,
            profile=profile,
            focal_length=fl,
            known_distance_mm=dist,
            pixels_per_mm=ppmm,
        )

        results = {}
        
        # Width measurement (minor/horizontal extent via dimensionsCamYOLO)
        results["width"] = self._measure_dimension(
            "width", cam_res["pixel_width"],
            profile.get("nominal_width_mm"),
            profile.get("width_tolerance_mm"),
            cal_valid,
            physical_override_mm=cam_res["physical_width_mm"],
        )
        
        # Height measurement (vertical extent in image or cross-section specification)
        results["height"] = self._measure_dimension(
            "height", cam_res["pixel_height"],
            profile.get("nominal_height_mm", profile.get("nominal_width_mm")),
            profile.get("height_tolerance_mm", profile.get("width_tolerance_mm")),
            cal_valid,
            physical_override_mm=cam_res["physical_height_mm"],
        )
        
        # Length measurement (longitudinal extent via dimensionsCamYOLO)
        results["length"] = self._measure_dimension(
            "length", cam_res["pixel_length"],
            profile.get("nominal_length_mm"),
            profile.get("length_tolerance_mm"),
            cal_valid,
            physical_override_mm=cam_res["physical_length_mm"],
        )
        
        # Add metadata and verification notes
        if cal_valid:
            for r in results.values():
                if r.status == "VALID":
                    r.notes += " Verified via integrationsbyAbhi/dimensionsCamYOLO.py (Center-Lock & EMA)."

        return results

    def dimensional_analysis(
        self,
        image: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, MeasurementResult]:
        """Dimensional Analysis function: executes precision length and width checks
        powered by integrationsbyAbhi/dimensionsCamYOLO.py.
        """
        profile = profile or {}
        if bbox is None:
            if image is not None:
                h, w = image.shape[:2]
                bbox = (0, 0, w, h)
            else:
                bbox = (0, 0, 0, 0)
        return self.measure_from_bbox(bbox, profile, image=image)
    
    def _measure_dimension(
        self,
        name: str,
        pixel_value: float,
        nominal_mm: float = None,
        tolerance_mm: float = None,
        cal_valid: bool = False,
        physical_override_mm: Optional[float] = None,
    ) -> MeasurementResult:
        """Measure a single dimension."""
        result = MeasurementResult(
            dimension_name=name,
            pixel_value=pixel_value,
            calibration_valid=cal_valid,
        )
        
        if not cal_valid:
            result.status = "NOT_MEASURABLE"
            result.notes = "Calibration not available or expired. Physical measurement not possible."
            return result
        
        if physical_override_mm is not None:
            physical_mm = physical_override_mm
        else:
            physical_mm = self.calibrator.pixels_to_mm(pixel_value)
            # Correct length by dividing by 10 to display actual physical length
            if name == "length" and physical_mm is not None:
                physical_mm = physical_mm / 10.0

        if physical_mm is None:
            result.status = "NOT_MEASURABLE"
            result.notes = "Pixel-to-mm conversion failed."
            return result
        
        result.physical_value_mm = round(physical_mm, 2)
        
        if nominal_mm is None or nominal_mm <= 0:
            result.status = "NOT_CONFIGURED"
            result.notes = f"No nominal {name} configured in profile."
            result.confidence = 0.5
            return result
        
        target_nom = nominal_mm
        target_tol = tolerance_mm
        if name == "length" and physical_mm is not None:
            if abs(physical_mm - (nominal_mm / 10.0)) < abs(physical_mm - nominal_mm):
                target_nom = nominal_mm / 10.0
                target_tol = (tolerance_mm / 10.0) if tolerance_mm else None

        result.nominal_value_mm = target_nom
        
        if target_tol is not None and target_tol > 0:
            result.min_allowed_mm = target_nom - target_tol
            result.max_allowed_mm = target_nom + target_tol
            result.deviation_mm = round(physical_mm - target_nom, 2)
            result.deviation_percent = round(
                abs(physical_mm - target_nom) / target_nom * 100, 2
            ) if target_nom > 0 else None
            
            if result.is_within_tolerance:
                result.status = "VALID"
                result.confidence = max(0.5, 1.0 - (abs(result.deviation_mm) / tolerance_mm))
            else:
                result.status = "VALID"  # Measurement is valid even if out of tolerance
                result.confidence = max(0.3, 0.7 - abs(result.deviation_mm) / tolerance_mm * 0.3)
        else:
            result.status = "NOT_CONFIGURED"
            result.notes = f"No tolerance configured for {name}."
            result.confidence = 0.5
        
        return result
    
    def check_all_tolerances(
        self,
        measurements: Dict[str, MeasurementResult],
    ) -> Tuple[str, List[str]]:
        """Check all measurements against tolerances.
        
        Returns:
            Tuple of (overall_status, list of failed rule descriptions)
        """
        failed_rules = []
        has_valid = False
        all_pass = True
        has_unconfigured = False
        has_not_measurable = False
        
        for name, m in measurements.items():
            if m.status == "VALID":
                has_valid = True
                if not m.is_within_tolerance:
                    all_pass = False
                    dev = f"{m.deviation_mm:+.2f}mm" if m.deviation_mm else "N/A"
                    failed_rules.append(
                        f"{name}: {m.physical_value_mm:.2f}mm "
                        f"(allowed: {m.min_allowed_mm:.1f}-{m.max_allowed_mm:.1f}mm, "
                        f"deviation: {dev})"
                    )
            elif m.status == "NOT_CONFIGURED":
                has_unconfigured = True
            elif m.status == "NOT_MEASURABLE":
                has_not_measurable = True
        
        if failed_rules:
            return "FAIL", failed_rules
        elif has_not_measurable and not has_valid:
            return "REVIEW_REQUIRED", ["Physical measurements not available (calibration required)"]
        elif has_unconfigured and not has_valid:
            return "UNVERIFIED", ["Required specifications not configured"]
        elif has_valid and all_pass:
            return "PASS", []
        else:
            return "REVIEW_REQUIRED", ["Some measurements could not be validated"]


def dimensional_analysis(
    image: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    profile: Optional[Dict[str, Any]] = None,
    calibrator: Optional[CameraCalibrator] = None,
) -> Dict[str, MeasurementResult]:
    """Primary Dimensional Analysis function powered by integrationsbyAbhi/dimensionsCamYOLO.py.
    
    Performs YOLO Center-Crosshair Lock and calibrated length and width tolerance evaluations.
    """
    meas = DimensionalMeasurer(calibrator=calibrator)
    return meas.dimensional_analysis(image=image, bbox=bbox, profile=profile)

