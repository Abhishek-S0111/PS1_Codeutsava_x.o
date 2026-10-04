"""
Defect Detection Module
Surface defect detection using YOLO or image-processing fallback.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from utils.config import DEFECT_CONFIG, EVIDENCE_DIR
from utils.logger import logger


@dataclass
class DefectDetection:
    """A detected defect on a billet surface."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (in original frame coords)
    confidence: float
    defect_type: str = "defect"
    patch_index: int = -1
    evidence_path: str = ""


class DefectDetector:
    """Detects surface defects on steel billets using ResNet50 and frame segmentation."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_type = "none"
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Attempt to load defect detection model."""
        # We enforce using the specific resnet50 model path requested
        model_path = Path("models/resnet50_steel_defect.pt")
        
        if model_path.exists():
            try:
                import torch
                import torchvision.transforms as transforms
                
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # Load the model directly (assuming full model was saved)
                loaded = torch.load(model_path, map_location=self.device)
                
                if isinstance(loaded, dict):
                    # It's a state_dict, we need to initialize a ResNet50
                    from torchvision.models import resnet50
                    self.model = resnet50(num_classes=2) # Adjust num_classes if necessary
                    self.model.load_state_dict(loaded)
                else:
                    self.model = loaded
                    
                self.model.to(self.device)
                self.model.eval()
                
                self.model_loaded = True
                self.model_type = "resnet50"
                
                # Transform for consistent graying and ResNet input
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=3), # Consistent graying out
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                
                logger.info(f"Loaded defect detector: {model_path} on {self.device}")
                return
            except Exception as e:
                logger.warning(f"Failed to load ResNet model: {e}")
        
        logger.info("Defect detector: using image-processing fallback (model not configured or import failed)")
        self.model_type = "image_processing_fallback"
    
    def detect(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int] = None,
        use_patches: bool = True,
    ) -> List[DefectDetection]:
        """Detect defects in an image or ROI.
        
        Args:
            image: Input frame (BGR)
            roi: Optional bounding box (x1, y1, x2, y2) to restrict detection
            use_patches: Whether to use patch-based detection for large images
        """
        if roi:
            x1, y1, x2, y2 = roi
            crop = image[y1:y2, x1:x2]
        else:
            crop = image
            roi = (0, 0, image.shape[1], image.shape[0])
        
        if self.model_loaded and self.model_type == "resnet50":
            detections = self._detect_resnet(crop, roi)
        else:
            detections = self._detect_fallback(crop, roi, use_patches)
        
        # Non-maximum suppression (for ResNet segments, we might not want standard NMS if segments just report localized defects, but let's keep it to remove highly overlapping same defects)
        detections = self._nms(detections, DEFECT_CONFIG.get("nms_threshold", 0.5))
        
        return detections
    
    def _detect_resnet(self, crop: np.ndarray, roi: Tuple) -> List[DefectDetection]:
        """ResNet-based defect detection using segments/patches."""
        import torch
        if crop.size == 0:
            return []
            
        detections = []
        
        # We segment the image into overlapping patches (e.g. 224x224)
        patch_size = 224
        overlap = 0.5
        stride = int(patch_size * (1 - overlap))
        
        h, w = crop.shape[:2]
        
        if h < patch_size and w < patch_size:
            # Too small, just process whole thing
            patches = [{"image": crop, "offset": (0, 0)}]
        else:
            patches = []
            for y in range(0, max(1, h - patch_size + 1), stride):
                for x in range(0, max(1, w - patch_size + 1), stride):
                    py2 = min(y + patch_size, h)
                    px2 = min(x + patch_size, w)
                    # If it's the last patch and smaller than patch_size, we can take it, it will be resized by transforms anyway
                    patch = crop[y:py2, x:px2]
                    if patch.shape[0] > 50 and patch.shape[1] > 50:
                        patches.append({
                            "image": patch,
                            "offset": (x, y),
                        })

        for patch_info in patches:
            patch = patch_info["image"]
            offset_x, offset_y = patch_info["offset"]
            
            input_tensor = self.transform(patch).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                # Assume standard binary classification: index 1 is defect, index 0 is normal. Or similar.
                # If the model outputs 1 value (sigmoid), or 2 values (softmax).
                # We'll use softmax if it's 2 or more classes.
                if outputs.shape[1] >= 2:
                    probs = torch.softmax(outputs, dim=1)[0]
                    # Let's assume class 1 is defect
                    defect_prob = probs[1].item()
                    defect_class_idx = 1
                else:
                    # Single output (sigmoid)
                    defect_prob = torch.sigmoid(outputs)[0].item()
                    defect_class_idx = 1
            
            # If the segment has a defect
            if defect_prob > DEFECT_CONFIG.get("confidence_threshold", 0.5):
                ph, pw = patch.shape[:2]
                detections.append(DefectDetection(
                    bbox=(
                        int(offset_x + roi[0]),
                        int(offset_y + roi[1]),
                        int(offset_x + pw + roi[0]),
                        int(offset_y + ph + roi[1]),
                    ),
                    confidence=defect_prob,
                    defect_type="surface_defect",
                ))
                
        return detections
    
    def _detect_fallback(
        self,
        crop: np.ndarray,
        roi: Tuple,
        use_patches: bool = True,
    ) -> List[DefectDetection]:
        """Image-processing based defect detection fallback.
        
        NOTE: This is a heuristic fallback, NOT a validated deep-learning detector.
        Results should be treated as candidates requiring human review.
        """
        if crop.size == 0:
            return []
        
        detections = []
        
        patch_size = DEFECT_CONFIG.get("patch_size", 256)
        
        if use_patches and max(crop.shape[:2]) > patch_size:
            # Patch-based processing
            patches = self._create_patches(crop, patch_size, DEFECT_CONFIG.get("patch_overlap", 0.5))
            for patch_info in patches:
                patch = patch_info["image"]
                offset_x, offset_y = patch_info["offset"]
                patch_dets = self._analyze_surface(patch)
                
                for det in patch_dets:
                    # Remap patch coordinates to original frame
                    det.bbox = (
                        det.bbox[0] + offset_x + roi[0],
                        det.bbox[1] + offset_y + roi[1],
                        det.bbox[2] + offset_x + roi[0],
                        det.bbox[3] + offset_y + roi[1],
                    )
                    det.patch_index = patch_info["index"]
                    detections.append(det)
        else:
            dets = self._analyze_surface(crop)
            for det in dets:
                det.bbox = (
                    det.bbox[0] + roi[0],
                    det.bbox[1] + roi[1],
                    det.bbox[2] + roi[0],
                    det.bbox[3] + roi[1],
                )
                detections.append(det)
        
        return detections
    
    def _create_patches(self, image: np.ndarray, patch_size: int, overlap: float) -> List[Dict]:
        """Create overlapping patches from a large image."""
        stride = int(patch_size * (1 - overlap))
        
        h, w = image.shape[:2]
        patches = []
        idx = 0
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                py2 = min(y + patch_size, h)
                px2 = min(x + patch_size, w)
                patch = image[y:py2, x:px2]
                
                if patch.shape[0] > 50 and patch.shape[1] > 50:
                    patches.append({
                        "image": patch,
                        "offset": (x, y),
                        "index": idx,
                    })
                    idx += 1
        
        return patches
    
    def _analyze_surface(self, surface: np.ndarray) -> List[DefectDetection]:
        """Analyze a surface patch for anomalies using image processing.
        
        WARNING: This is a heuristic method, not a trained classifier.
        Detections are labelled as candidates.
        """
        if surface.size == 0 or surface.shape[0] < 20 or surface.shape[1] < 20:
            return []
        
        gray = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY) if len(surface.shape) == 3 else surface
        
        detections = []
        
        # Method 1: Edge density anomalies
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_area = gray.shape[0] * gray.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < img_area * 0.005 or area > img_area * 0.5:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Compute local texture variance
            region = gray[y:y+h, x:x+w]
            if region.size > 0:
                local_std = np.std(region)
                global_std = np.std(gray)
                
                if local_std > global_std * 1.5:
                    confidence = min(0.5, (local_std / (global_std + 1e-6) - 1) * 0.2)
                    detections.append(DefectDetection(
                        bbox=(x, y, x + w, y + h),
                        confidence=confidence,
                        defect_type="surface_anomaly_candidate",
                    ))
        
        # Method 2: Dark/bright spots (potential inclusions, pitting)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        diff = cv2.absdiff(gray, blurred)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        spot_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in spot_contours:
            area = cv2.contourArea(contour)
            if area < img_area * 0.003 or area > img_area * 0.3:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2 + 1e-6)
            
            detections.append(DefectDetection(
                bbox=(x, y, x + w, y + h),
                confidence=min(0.4, circularity * 0.3),
                defect_type="inclusion_candidate" if circularity > 0.5 else "scratch_candidate",
            ))
        
        return detections
    
    def _nms(self, detections: List[DefectDetection], threshold: float) -> List[DefectDetection]:
        """Non-maximum suppression."""
        if not detections:
            return []
        
        detections.sort(key=lambda d: d.confidence, reverse=True)
        keep = []
        
        for det in detections:
            overlap = False
            for kept in keep:
                iou = self._compute_iou(det.bbox, kept.bbox)
                if iou > threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)
        
        return keep
    
    @staticmethod
    def _compute_iou(box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        
        return inter / (union + 1e-6)
    
    def save_defect_evidence(self, frame: np.ndarray, defect: DefectDetection) -> str:
        """Save defect crop as evidence."""
        h, w = frame.shape[:2]
        pad = 15
        x1 = max(0, defect.bbox[0] - pad)
        y1 = max(0, defect.bbox[1] - pad)
        x2 = min(w, defect.bbox[2] + pad)
        y2 = min(h, defect.bbox[3] + pad)
        
        crop = frame[y1:y2, x1:x2]
        filename = f"defect_{defect.defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath = EVIDENCE_DIR / filename
        cv2.imwrite(str(filepath), crop)
        defect.evidence_path = str(filepath)
        return str(filepath)
    
    @property
    def status(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.model_loaded,
            "model_type": self.model_type,
            "is_fallback": self.model_type == "image_processing_fallback",
            "note": "Using heuristic fallback. Upload a trained model for reliable detection." if not self.model_loaded else "",
        }
