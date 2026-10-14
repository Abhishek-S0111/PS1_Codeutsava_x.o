"""
Measurement Module (Deprecated / Wiped Out)
Dimensional analysis and camera calibration features have been completely removed.
This module provides lightweight backward-compatibility stubs.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Any
from utils.logger import logger


@dataclass
class CalibrationData:
    """Deprecated: calibration data stub."""
    calibration_id: str = ""
    method: str = "none"
    pixels_per_mm: float = 0.0
    reference_dimension_mm: float = 0.0
    reprojection_error: float = 0.0
    valid_until: str = ""
    created_at: str = ""
    focal_length: float = 0.0
    known_distance_mm: float = 0.0

    @property
    def is_valid(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {}


@dataclass
class MeasurementResult:
    """Deprecated: measurement result stub."""
    dimension_name: str = ""
    pixel_value: float = 0.0
    physical_value_mm: Optional[float] = None
    unit: str = "mm"
    nominal_value_mm: Optional[float] = None
    min_allowed_mm: Optional[float] = None
    max_allowed_mm: Optional[float] = None
    deviation_mm: Optional[float] = None
    deviation_percent: Optional[float] = None
    status: str = "NOT_MEASURED"
    calibration_valid: bool = False
    confidence: float = 0.0
    notes: str = "Dimensional analysis removed."

    @property
    def is_within_tolerance(self) -> bool:
        return True

    @property
    def physical_value_cm(self) -> Optional[float]:
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {}


class CameraCalibrator:
    """Deprecated: camera calibration stub."""
    def __init__(self, calibration_file: Optional[str] = None):
        self.calibration = None

    def load_calibration(self) -> Optional[CalibrationData]:
        return None

    def is_calibrated(self) -> bool:
        return False

    def pixels_to_mm(self, pixels: float) -> Optional[float]:
        return None

    def pixels_to_cm(self, pixels: float) -> Optional[float]:
        return None


class DimensionalMeasurer:
    """Deprecated: dimensional measurer stub."""
    def __init__(self, calibrator: Optional[CameraCalibrator] = None):
        self.calibrator = calibrator or CameraCalibrator()

    def measure_from_bbox(self, bbox: Any, profile: Optional[Dict] = None) -> Dict[str, MeasurementResult]:
        return {}


def dimensional_analysis(*args, **kwargs) -> Dict[str, Any]:
    """Deprecated: dimensional analysis has been completely wiped out."""
    logger.info("dimensional_analysis called, but dimensional analysis has been removed.")
    return {}
