"""
Vision Module - Billet Detection and Frame Processing
Handles video decoding, billet detection, ROI extraction, and tracking.
"""
import cv2
import numpy as np
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from utils.config import DETECTION_CONFIG, VIDEO_CONFIG, EVIDENCE_DIR
from utils.logger import logger


@dataclass
class BilletDetection:
    """Single billet detection in a frame."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str = "billet"
    track_id: str = ""
    frame_number: int = 0
    timestamp: float = 0.0
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class FrameResult:
    """Processing result for a single frame."""
    frame_number: int
    timestamp: float
    original_frame: np.ndarray
    annotated_frame: np.ndarray = None
    detections: List[BilletDetection] = field(default_factory=list)
    processing_time_ms: float = 0.0
    error: str = None


class BilletDetector:
    """Detects steel billets in video frames using YOLO or fallback methods."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_type = "none"
        self._load_model()
    
    def _load_model(self):
        """Attempt to load the YOLO model."""
        
        # 1. Try loading custom vidModel_Util YOLO model (User requested)
        try:
            from integrationsbyAbhi.vidModel_Util import get_vid_model
            vid_model = get_vid_model()
            if vid_model is not None:
                self.model = vid_model
                self.model_loaded = True
                self.model_type = "custom_yolo"
                logger.info("Loaded custom billet detector via vidModel_Util")
                return
        except Exception as e:
            logger.warning(f"Failed to load via vidModel_Util: {e}")
            
        # 2. Try loading custom model from config
        model_path = DETECTION_CONFIG["model_path"]
        if Path(model_path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.model_loaded = True
                self.model_type = "custom_yolo"
                logger.info(f"Loaded custom billet detector: {model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load custom model: {e}")
        
        # 3. Try loading pre-trained YOLO for general object detection fallback
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            self.model_loaded = True
            self.model_type = "pretrained_yolo"
            logger.info("Loaded pre-trained YOLOv8n (general detection)")
        except Exception as e:
            logger.warning(f"YOLO not available: {e}. Using contour-based fallback.")
            self.model_type = "contour_fallback"
    
    def detect(self, frame: np.ndarray, confidence_threshold: float = None) -> List[BilletDetection]:
        """Detect billets in a frame."""
        if confidence_threshold is None:
            confidence_threshold = DETECTION_CONFIG["confidence_threshold"]
        
        if self.model_loaded and self.model_type == "custom_yolo":
            return self._detect_yolo_custom(frame, confidence_threshold)
        elif self.model_loaded and self.model_type == "pretrained_yolo":
            return self._detect_yolo_pretrained(frame, confidence_threshold)
        else:
            return self._detect_contour_fallback(frame)
    
    def _detect_yolo_custom(self, frame: np.ndarray, conf: float) -> List[BilletDetection]:
        """Detection with custom-trained YOLO model."""
        results = self.model(frame, conf=conf, verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names.get(cls, "billet")
                    
                    detections.append(BilletDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        class_name=class_name,
                    ))
        
        return detections
    
    def _detect_yolo_pretrained(self, frame: np.ndarray, conf: float) -> List[BilletDetection]:
        """Use pre-trained YOLO to find rectangular/elongated objects."""
        results = self.model(frame, conf=max(conf, 0.3), verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    frame_area = frame.shape[0] * frame.shape[1]
                    
                    # Heuristic: billets are large, roughly rectangular
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                    if area > frame_area * 0.02 and aspect_ratio > 1.5:
                        detections.append(BilletDetection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=confidence * 0.7,  # Downweight since not billet-specific
                            class_name="billet_candidate",
                        ))
        
        # If no YOLO candidates, try contour fallback
        if not detections:
            detections = self._detect_contour_fallback(frame)
        
        return detections
    
    def _detect_contour_fallback(self, frame: np.ndarray) -> List[BilletDetection]:
        """Contour-based fallback detection for elongated rectangular objects."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Adaptive thresholding for varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < frame_area * 0.02:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            
            if aspect_ratio > 1.3 and area > frame_area * 0.02:
                rect = cv2.minAreaRect(contour)
                solidity = area / (w * h + 1e-6)
                
                detections.append(BilletDetection(
                    bbox=(x, y, x + w, y + h),
                    confidence=min(0.5, solidity * 0.6),  # Lower confidence for fallback
                    class_name="billet_candidate_contour",
                ))
        
        # Sort by area, take top detections
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections[:5]
    
    @property
    def status(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.model_loaded,
            "model_type": self.model_type,
            "device": DETECTION_CONFIG["device"],
        }


class SimpleTracker:
    """Simple IoU-based tracker for billets across frames."""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.tracks: Dict[str, Dict] = {}
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
    
    def update(self, detections: List[BilletDetection], frame_number: int) -> List[BilletDetection]:
        """Update tracks with new detections."""
        if not detections:
            self._age_tracks(frame_number)
            return []
        
        # Match detections to existing tracks
        matched = set()
        for det in detections:
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                iou = self._compute_iou(det.bbox, track["bbox"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id:
                det.track_id = best_track_id
                self.tracks[best_track_id]["bbox"] = det.bbox
                self.tracks[best_track_id]["last_seen"] = frame_number
                self.tracks[best_track_id]["frames_seen"] += 1
                matched.add(best_track_id)
            else:
                # New track
                track_id = f"BLT-{self.next_id:04d}"
                self.next_id += 1
                det.track_id = track_id
                self.tracks[track_id] = {
                    "bbox": det.bbox,
                    "last_seen": frame_number,
                    "first_seen": frame_number,
                    "frames_seen": 1,
                }
        
        self._age_tracks(frame_number)
        return detections
    
    def _age_tracks(self, frame_number: int):
        """Remove old tracks."""
        to_remove = [
            tid for tid, t in self.tracks.items()
            if frame_number - t["last_seen"] > self.max_age
        ]
        for tid in to_remove:
            del self.tracks[tid]
    
    @staticmethod
    def _compute_iou(box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)


class VideoProcessor:
    """Processes uploaded videos frame by frame."""
    
    def __init__(self, detector: BilletDetector = None, tracker: SimpleTracker = None, calibrator = None):
        self.detector = detector or BilletDetector()
        self.tracker = tracker or SimpleTracker()
        self.calibrator = calibrator
        self.is_processing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}
        
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        cap.release()
        return info
    
    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FrameResult:
        """Process a single frame."""
        import time
        start = time.time()
        
        result = FrameResult(
            frame_number=frame_number,
            timestamp=timestamp,
            original_frame=frame.copy(),
        )
        
        try:
            # Detect billets
            detections = self.detector.detect(frame)
            
            # Track
            detections = self.tracker.update(detections, frame_number)
            
            # Annotate
            annotated = frame.copy()
            for det in detections:
                annotated = self._draw_detection(annotated, det)
            
            result.detections = detections
            result.annotated_frame = annotated
            
        except Exception as e:
            logger.error(f"Frame {frame_number} processing error: {e}")
            result.error = str(e)
            result.annotated_frame = frame.copy()
        
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    def _draw_detection(self, frame: np.ndarray, det: BilletDetection) -> np.ndarray:
        """Draw bounding box and label on frame."""
        x1, y1, x2, y2 = det.bbox
        color = (0, 230, 118)  # Green
        thickness = 2
        
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label
        label = f"{det.track_id} | {det.confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), 1)
        
        # Dimensions in pixels
        dim_label = f"{det.width}x{det.height}px"
        cv2.putText(frame, dim_label, (x1, y2 + 20), font, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def save_evidence(self, frame: np.ndarray, detection: BilletDetection, prefix: str = "evidence") -> str:
        """Save evidence image to disk."""
        filename = f"{prefix}_{detection.track_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath = EVIDENCE_DIR / filename
        
        # Crop the detection region with padding
        h, w = frame.shape[:2]
        pad = 20
        x1 = max(0, detection.bbox[0] - pad)
        y1 = max(0, detection.bbox[1] - pad)
        x2 = min(w, detection.bbox[2] + pad)
        y2 = min(h, detection.bbox[3] + pad)
        
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(str(filepath), crop)
        return str(filepath)
