"""
OCR Module - Optical Character Recognition and Barcode/QR Decoding
Extracts billet IDs, heat numbers, batch numbers from markings.
Offline-first design with built-in industrial character & QR recognition.
"""
import os
import cv2
import numpy as np
import re
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from utils.config import OCR_CONFIG, EVIDENCE_DIR
from utils.logger import logger


@dataclass
class OCRResult:
    """OCR extraction result."""
    raw_text: str = ""
    cleaned_text: str = ""
    billet_id: str = ""
    heat_number: str = ""
    batch_number: str = ""
    serial_number: str = ""
    confidence: float = 0.0
    engine: str = "none"
    qr_data: str = ""
    barcode_data: str = ""
    preprocessing_method: str = ""
    evidence_path: str = ""
    is_readable: bool = False
    validation_warnings: List[str] = field(default_factory=list)
    detections: List[Dict[str, Any]] = field(default_factory=list)


class OCREngine:
    """Handles OCR and 2D code extraction from billet images."""

    _templates_cache: Optional[Dict[str, np.ndarray]] = None

    def __init__(self):
        self.engine = None
        self.engine_name = "none"
        self._init_engine()

    def _init_engine(self):
        """Initialize OCR engine with RapidOCR ONNX runtime as primary engine."""
        preferred = OCR_CONFIG.get("engine", "rapidocr")

        # 1. Primary: RapidOCR (ONNX-backed, ultra-fast and robust)
        if preferred in ("rapidocr", "rapid_ocr", "onnx"):
            try:
                from rapidocr_onnxruntime import RapidOCR
                self.engine = RapidOCR()
                self.engine_name = "rapidocr"
                logger.info("RapidOCR (ONNX Runtime) initialized as primary OCR engine")
                return
            except Exception as e:
                logger.warning(f"RapidOCR initialization failed, falling back: {e}")

        # 2. Check if EasyOCR weights are already cached locally before attempting import
        easyocr_model_dir = Path(os.path.expanduser("~/.EasyOCR/model"))
        craft_file = easyocr_model_dir / "craft_mlt_25k.pth"
        recog_file = easyocr_model_dir / "english_g2.pth"

        if preferred == "easyocr" and craft_file.exists() and recog_file.exists():
            try:
                import easyocr
                self.engine = easyocr.Reader(
                    OCR_CONFIG["language"],
                    gpu=OCR_CONFIG["gpu_enabled"],
                    download_enabled=False,
                )
                self.engine_name = "easyocr"
                logger.info("EasyOCR engine initialized from local weights")
                return
            except Exception as e:
                logger.warning(f"EasyOCR local init failed: {e}")

        # Try RapidOCR if preferred was easyocr but easyocr weights weren't found
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR()
            self.engine_name = "rapidocr"
            logger.info("RapidOCR (ONNX Runtime) initialized as primary OCR engine")
            return
        except Exception as e:
            logger.debug(f"RapidOCR secondary init: {e}")

        # 3. Try Tesseract if available
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.engine_name = "tesseract"
            logger.info("Tesseract OCR engine available")
            return
        except Exception:
            pass

        # 4. Built-in offline industrial template & contour OCR engine
        self.engine_name = "industrial_ocr"
        logger.info("Industrial Offline OCR engine initialized (100% offline, zero network required)")

    @classmethod
    def _pad_glyph(cls, img: np.ndarray, target_h: int = 32, target_w: int = 24) -> np.ndarray:
        """Resize patch to target dimensions preserving aspect ratio with zero padding."""
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((target_h, target_w), dtype=np.uint8)
        scale = min(target_h / max(h, 1), target_w / max(w, 1))
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        y_off = (target_h - nh) // 2
        x_off = (target_w - nw) // 2
        canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
        return canvas

    @classmethod
    def _get_char_templates(cls) -> Dict[str, np.ndarray]:
        """Pre-render and cache character templates for industrial glyph matching."""
        if cls._templates_cache is not None:
            return cls._templates_cache

        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-:#"
        templates = {}
        for c in chars:
            canvas = np.zeros((44, 44), dtype=np.uint8)
            (tw, th), baseline = cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            x = max(0, (44 - tw) // 2)
            y = max(th, (44 + th) // 2)
            cv2.putText(canvas, c, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
            pts = cv2.findNonZero(canvas)
            if pts is not None:
                bx, by, bw, bh = cv2.boundingRect(pts)
                glyph = canvas[by:by + bh, bx:bx + bw]
                templates[c] = cls._pad_glyph(glyph, 32, 24)
            else:
                templates[c] = np.zeros((32, 24), dtype=np.uint8)

        cls._templates_cache = templates
        return templates

    def _run_industrial_ocr(self, crop: np.ndarray) -> Tuple[str, float]:
        """Recognize stamped industrial characters using morphology and aspect-preserved matching."""
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
            templates = self._get_char_templates()
            candidates = []

            # Multi-threshold scan
            threshold_methods = [
                ("plate_otsu", lambda g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                ("bright_thresh", lambda g: cv2.threshold(g, 175, 255, cv2.THRESH_BINARY)[1]),
                ("dark_thresh", lambda g: cv2.threshold(g, 80, 255, cv2.THRESH_BINARY_INV)[1]),
                ("adaptive", lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -8)),
            ]

            for name, get_thresh in threshold_methods:
                thresh = get_thresh(gray)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes = []
                for cnt in contours:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    # Filter for typical character height/width
                    if 10 <= bh <= 65 and 3 <= bw <= 55 and (bw * bh) < (crop.shape[0] * crop.shape[1] * 0.15):
                        boxes.append((x, y, bw, bh))

                if len(boxes) < 3:
                    continue

                # Group boxes into horizontal text lines
                lines: Dict[int, List[Tuple[int, int, int, int]]] = {}
                for b in boxes:
                    placed = False
                    for lk in lines:
                        if abs(b[1] - lk) < 22:
                            lines[lk].append(b)
                            placed = True
                            break
                    if not placed:
                        lines[b[1]] = [b]

                detected_lines = []
                conf_scores = []
                for lk in sorted(lines.keys()):
                    line_boxes = sorted(lines[lk], key=lambda b: b[0])
                    if len(line_boxes) < 3:
                        continue

                    line_str = ""
                    prev_x = None
                    for (x, y, bw, bh) in line_boxes:
                        if prev_x is not None and (x - prev_x) > 13:
                            line_str += " "
                        patch = self._pad_glyph(thresh[y:y + bh, x:x + bw], 32, 24)
                        best_c, best_s = "?", -1.0
                        for c, tpl in templates.items():
                            s = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)[0][0]
                            if s > best_s:
                                best_s = s
                                best_c = c

                        if best_s > 0.45:
                            line_str += best_c
                            conf_scores.append(float(best_s))
                        prev_x = x + bw

                    cleaned_line = line_str.strip()
                    if cleaned_line:
                        detected_lines.append(cleaned_line)

                if detected_lines:
                    full_str = " | ".join(detected_lines)
                    # Disambiguate common OCR confusions in industrial tags
                    full_str = re.sub(r'\b8LT\b', 'BLT', full_str)
                    full_str = re.sub(r'\b8ILLET\b', 'BILLET', full_str)
                    full_str = re.sub(r'\b8L\b', 'BL', full_str)
                    full_str = re.sub(r'\bHE4T\b', 'HEAT', full_str)
                    full_str = re.sub(r'\bH7\b', 'HT', full_str)
                    avg_c = float(np.mean(conf_scores)) if conf_scores else 0.85
                    candidates.append((full_str, avg_c))

            if candidates:
                # Rank candidates by recognizable tokens
                candidates.sort(key=lambda t: len(re.findall(r'[A-Z0-9-]{3,}', t[0])), reverse=True)
                return candidates[0]

            return "", 0.0
        except Exception as e:
            logger.debug(f"Industrial OCR error: {e}")
            return "", 0.0

    def extract(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int] = None,
        default_billet_id: str = None,
        default_heat_number: str = None,
    ) -> OCRResult:
        """Extract Billet ID, Heat Number, and batch markings from image or ROI.

        Args:
            image: Input image (BGR)
            roi: Optional region of interest (x1, y1, x2, y2)
            default_billet_id: Optional fallback Billet ID
            default_heat_number: Optional fallback Heat Number
        """
        result = OCRResult(engine=self.engine_name)

        if roi:
            x1, y1, x2, y2 = roi
            crop = image[y1:y2, x1:x2]
        else:
            crop = image

        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            result.validation_warnings.append("Image region too small for OCR")
            if default_billet_id:
                result.billet_id = default_billet_id
                result.heat_number = default_heat_number or ""
                result.is_readable = True
            return result

        # ─── 1. Check for QR Code and Barcode ─────────────────────────
        qr_result = self._decode_qr_barcode(crop)
        if qr_result:
            result.qr_data = qr_result.get("qr", "")
            result.barcode_data = qr_result.get("barcode", "")
            tag_text = result.qr_data or result.barcode_data
            parsed_qr = self._parse_fields(tag_text)

            if parsed_qr.get("billet_id") or parsed_qr.get("heat_number"):
                result.billet_id = parsed_qr.get("billet_id", "")
                result.heat_number = parsed_qr.get("heat_number", "")
                result.batch_number = parsed_qr.get("batch_number", "")
                result.raw_text = tag_text
                result.cleaned_text = tag_text
                result.confidence = 0.99
                result.is_readable = True
                result.engine = "qr_decoder"
                result.preprocessing_method = "qr_direct"
                # If both are captured, we can return immediately
                if result.billet_id and result.heat_number:
                    return result

        # ─── 2. Run OCR (RapidOCR / EasyOCR / Tesseract / Industrial OCR) ─
        best_text = ""
        best_confidence = 0.0
        best_method = "raw"
        detected_boxes = []

        if self.engine_name == "rapidocr":
            text, conf, dets = self._run_rapidocr(crop)
            if not text.strip():
                # Try key preprocessing methods if raw crop didn't find anything
                for method in ["contrast_enhance", "grayscale", "sharpen"]:
                    prep = self._preprocess(crop, method)
                    p_text, p_conf, p_dets = self._run_rapidocr(prep)
                    if p_text.strip():
                        text, conf, dets = p_text, p_conf, p_dets
                        best_method = method
                        break
            if text.strip():
                best_text = text
                best_confidence = conf
                if best_method == "raw":
                    best_method = "rapidocr_direct"
                detected_boxes = dets
            else:
                # Fallback to industrial template OCR if RapidOCR missed stamped markings
                text_ind, conf_ind = self._run_industrial_ocr(crop)
                if text_ind.strip():
                    best_text = text_ind
                    best_confidence = conf_ind
                    best_method = "industrial_stencil_fallback"
        elif self.engine_name == "easyocr":
            for method in OCR_CONFIG["preprocessing_methods"]:
                prep = self._preprocess(crop, method)
                text, conf = self._run_easyocr(prep)
                if conf > best_confidence and text.strip():
                    best_confidence = conf
                    best_text = text
                    best_method = method
        elif self.engine_name == "tesseract":
            for method in OCR_CONFIG["preprocessing_methods"]:
                prep = self._preprocess(crop, method)
                text, conf = self._run_tesseract(prep)
                if conf > best_confidence and text.strip():
                    best_confidence = conf
                    best_text = text
                    best_method = method
        else:
            # Industrial offline glyph & contour OCR
            text, conf = self._run_industrial_ocr(crop)
            if text.strip():
                best_text = text
                best_confidence = conf
                best_method = "industrial_stencil"

        # Offset detection coordinates to global frame if ROI was given
        if roi and detected_boxes:
            rx1, ry1, _, _ = roi
            adjusted_boxes = []
            for d in detected_boxes:
                d_copy = dict(d)
                if "box" in d and d["box"] is not None:
                    d_copy["box"] = [[float(p[0] + rx1), float(p[1] + ry1)] for p in d["box"]]
                if "pt1" in d:
                    d_copy["pt1"] = (int(d["pt1"][0] + rx1), int(d["pt1"][1] + ry1))
                if "pt2" in d:
                    d_copy["pt2"] = (int(d["pt2"][0] + rx1), int(d["pt2"][1] + ry1))
                adjusted_boxes.append(d_copy)
            result.detections = adjusted_boxes
        else:
            result.detections = detected_boxes

        # ─── 3. Parse fields from text ────────────────────────────────
        if best_text.strip():
            result.raw_text = best_text
            result.confidence = best_confidence or 0.88
            result.preprocessing_method = best_method
            result.cleaned_text = self._clean_text(best_text)

            parsed = self._parse_fields(result.cleaned_text)
            if not result.billet_id and parsed.get("billet_id"):
                result.billet_id = parsed["billet_id"]
            if not result.heat_number and parsed.get("heat_number"):
                result.heat_number = parsed["heat_number"]
            if not result.batch_number and parsed.get("batch_number"):
                result.batch_number = parsed["batch_number"]
            if not result.serial_number and parsed.get("serial_number"):
                result.serial_number = parsed["serial_number"]

        # ─── 4. Secondary Fallbacks & Disambiguation ──────────────────
        if not result.billet_id and result.qr_data:
            result.billet_id = result.qr_data
            result.is_readable = True
        elif not result.billet_id and result.barcode_data:
            result.billet_id = result.barcode_data
            result.is_readable = True

        # Apply operator defaults if provided and field is missing
        if not result.billet_id and default_billet_id:
            result.billet_id = default_billet_id
            result.validation_warnings.append("Billet ID inherited from batch configuration")

        if not result.heat_number and default_heat_number:
            result.heat_number = default_heat_number
            result.validation_warnings.append("Heat number inherited from batch configuration")

        # Determine readability
        if result.billet_id and result.billet_id != "ID_UNREADABLE":
            result.is_readable = True
            if not result.confidence or result.confidence <= 0:
                result.confidence = 0.90
        else:
            result.is_readable = False
            result.validation_warnings.append("OCR could not extract readable Billet ID")

        # Check for commonly confused characters
        if result.cleaned_text:
            confused = self._check_confused_chars(result.cleaned_text)
            if confused:
                result.validation_warnings.extend(confused)

        return result

    def _preprocess(self, image: np.ndarray, method: str) -> np.ndarray:
        """Apply preprocessing for better OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if method == "grayscale":
            return gray
        elif method == "contrast_enhance":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        elif method == "denoise":
            return cv2.fastNlMeansDenoising(gray, h=15)
        elif method == "adaptive_threshold":
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            return cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 5
            )
        elif method == "sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(gray, -1, kernel)

        return gray

    def _run_rapidocr(self, image: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Run RapidOCR on image frame and return (combined_text, avg_confidence, detections)."""
        if self.engine is None:
            return "", 0.0, []
        try:
            result, _ = self.engine(image)
            if not result:
                return "", 0.0, []

            texts = []
            confidences = []
            detections = []
            for det in result:
                # RapidOCR returns: [box_coordinates, text, confidence_score]
                if len(det) >= 3:
                    box = det[0]
                    text = str(det[1]).strip()
                    try:
                        conf = float(det[2])
                    except (ValueError, TypeError):
                        conf = 0.85

                    if text:
                        texts.append(text)
                        confidences.append(conf)
                        try:
                            pt1 = (int(box[0][0]), int(box[0][1]))
                            pt2 = (int(box[2][0]), int(box[2][1]))
                        except Exception:
                            pt1 = (0, 0)
                            pt2 = (0, 0)

                        detections.append({
                            "box": box,
                            "text": text,
                            "confidence": conf,
                            "pt1": pt1,
                            "pt2": pt2,
                        })

            combined_text = " | ".join(texts)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            return combined_text, avg_conf, detections
        except Exception as e:
            logger.warning(f"RapidOCR execution error: {e}")
            return "", 0.0, []

    @staticmethod
    def draw_ocr_annotations(
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        draw_labels: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes and text from RapidOCR (matching integrationsbyAbhi/ocrCam.py)."""
        annotated = image.copy()
        for det in detections:
            box = det.get("box")
            text = det.get("text", "")
            conf = det.get("confidence", 0.0)

            if box is not None:
                try:
                    pts = np.array(box, dtype=np.int32)
                    cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
                    pt1 = (int(box[0][0]), int(box[0][1]))
                except Exception:
                    pt1 = det.get("pt1", (10, 10))
                    pt2 = det.get("pt2", (50, 50))
                    cv2.rectangle(annotated, pt1, pt2, color, 2)
            else:
                pt1 = det.get("pt1", (10, 10))
                pt2 = det.get("pt2", (50, 50))
                cv2.rectangle(annotated, pt1, pt2, color, 2)

            if draw_labels and text:
                label = f"{text}"
                if conf > 0:
                    label += f" ({conf:.0%})"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                ty = max(pt1[1] - 8, th + 4)
                # Background rectangle for crisp contrast on steel surfaces
                cv2.rectangle(annotated, (pt1[0], ty - th - 3), (pt1[0] + tw + 4, ty + bl), (0, 0, 0), -1)
                cv2.putText(
                    annotated, label, (pt1[0] + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        return annotated

    def _run_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Run OCR engine on preprocessed image."""
        if self.engine_name == "rapidocr":
            t, c, _ = self._run_rapidocr(image)
            return t, c
        elif self.engine_name == "easyocr":
            return self._run_easyocr(image)
        elif self.engine_name == "tesseract":
            return self._run_tesseract(image)
        return self._run_industrial_ocr(image)

    def _run_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Run EasyOCR safely."""
        try:
            results = self.engine.readtext(image)
            if not results:
                return "", 0.0

            texts = []
            confidences = []
            for (bbox, text, conf) in results:
                texts.append(text)
                confidences.append(conf)

            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            return combined_text, avg_confidence
        except Exception as e:
            logger.warning(f"EasyOCR error: {e}")
            return "", 0.0

    def _run_tesseract(self, image: np.ndarray) -> Tuple[str, float]:
        """Run Tesseract OCR."""
        try:
            import pytesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            texts = []
            confidences = []
            for i, conf in enumerate(data["conf"]):
                if int(conf) > 30:
                    texts.append(data["text"][i])
                    confidences.append(int(conf) / 100.0)

            combined = " ".join(t for t in texts if t.strip())
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            return combined, avg_conf
        except Exception as e:
            logger.warning(f"Tesseract error: {e}")
            return "", 0.0

    def _decode_qr_barcode(self, image: np.ndarray) -> Optional[Dict[str, str]]:
        """Attempt QR code and barcode decoding using OpenCV and pyzbar."""
        result = {}

        # OpenCV QR detector
        try:
            qr_detector = cv2.QRCodeDetector()
            data, points, _ = qr_detector.detectAndDecode(image)
            if data and data.strip():
                result["qr"] = data.strip()
        except Exception:
            pass

        # pyzbar for barcodes
        try:
            from pyzbar.pyzbar import decode
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            barcodes = decode(gray)
            for barcode in barcodes:
                decoded = barcode.data.decode("utf-8", errors="ignore").strip()
                barcode_type = barcode.type
                if barcode_type == "QRCODE":
                    result["qr"] = decoded
                else:
                    result["barcode"] = decoded
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Barcode decode error: {e}")

        return result if result else None

    def _clean_text(self, text: str) -> str:
        """Clean raw OCR output."""
        text = re.sub(r'\s+', ' ', text).strip()
        text = ''.join(c for c in text if c.isprintable())
        return text

    def _parse_fields(self, text: str) -> Dict[str, str]:
        """Parse billet ID, heat number, batch number from OCR/QR text."""
        fields = {}
        if not text or not text.strip():
            return fields

        upper_text = text.upper()

        # 1. Targeted labeled or code patterns
        # Billet ID: direct code match (e.g. BLT-1000, BLT1042)
        b_direct = re.search(r'\b(BLT[-]?\d{3,8})\b', upper_text)
        if b_direct:
            raw_b = b_direct.group(1).replace(" ", "")
            num_part = raw_b[3:].lstrip("-")
            fields["billet_id"] = f"BLT-{num_part}" if num_part else raw_b
        else:
            # Labeled match (e.g. BILLET: 1042 or BILLET NO 4092)
            b_label = re.search(r'(?:BILLET\s*(?:ID|NO|NUM|#)?[:\s#]*)(?:BLT[-:\s]*)?([A-Z0-9-]{3,12})\b', upper_text)
            if b_label:
                val = b_label.group(1).strip("-:# ")
                fields["billet_id"] = f"BLT-{val}" if (val.isdigit() and len(val) >= 3) else val
            else:
                gen_match = re.search(r'\b([A-Z]{2,4}[-\s]?\d{3,8})\b', upper_text)
                if gen_match:
                    fields["billet_id"] = gen_match.group(1).strip()

        # Heat Number: direct code match (e.g. HT-58210, HT88215)
        h_direct = re.search(r'\b(HT[-]?\d{3,8})\b', upper_text)
        if h_direct:
            raw_h = h_direct.group(1).replace(" ", "")
            hnum_part = raw_h[2:].lstrip("-")
            fields["heat_number"] = f"HT-{hnum_part}" if hnum_part else raw_h
        else:
            # Labeled match (e.g. HEAT: 88215 or HEAT NO 58210)
            h_label = re.search(r'(?:HEAT\s*(?:NO|NUM|#)?[:\s#]*)(?:HT[-:\s]*)?([A-Z0-9-]{3,12})\b', upper_text)
            if h_label:
                hval = h_label.group(1).strip("-:# ")
                clean_bid = fields.get("billet_id", "").replace("BLT-", "").replace("BLT", "")
                if hval != clean_bid:
                    fields["heat_number"] = f"HT-{hval}" if (hval.isdigit() and len(hval) >= 3) else hval

        # Batch Number
        batch_match = re.search(r"(?:BATCH\s*(?:NO|NUM|#)?|BCH|BT)[-:\s#]*([A-Z0-9-]{3,12})", upper_text)
        if batch_match:
            bval = batch_match.group(1).strip("-:# ")
            fields["batch_number"] = f"BATCH-{bval}" if (bval.isdigit() and len(bval) >= 3) else bval

        # 2. Multi-token fallback if unlabelled
        tokens = [t.strip() for t in re.split(r'[\s,/|]+', upper_text) if len(t.strip()) >= 3]
        if not fields.get("billet_id") and tokens:
            fields["billet_id"] = tokens[0]
            if len(tokens) > 1 and not fields.get("heat_number"):
                fields["heat_number"] = tokens[1]
        elif fields.get("billet_id") and not fields.get("heat_number"):
            for t in tokens:
                clean_t = t.replace("BLT-", "").replace("HT-", "").replace("BLT", "").replace("HT", "")
                if clean_t != fields["billet_id"].replace("BLT-", "").replace("BLT", "") and ("H" in t or t.isdigit()):
                    fields["heat_number"] = f"HT-{t}" if t.isdigit() else t
                    break

        return fields

    def _check_confused_chars(self, text: str) -> List[str]:
        """Check for commonly confused character pairs."""
        warnings = []
        confusions = [
            ("O", "0", "Letter O / digit 0"),
            ("I", "1", "Letter I / digit 1"),
            ("B", "8", "Letter B / digit 8"),
            ("S", "5", "Letter S / digit 5"),
            ("Z", "2", "Letter Z / digit 2"),
        ]

        for char1, char2, desc in confusions:
            if char1 in text and char2 in text:
                warnings.append(f"Potential confusion: {desc} in '{text}'")

        return warnings[:3]

    def save_ocr_evidence(
        self,
        original: np.ndarray,
        preprocessed: np.ndarray,
        ocr_result: OCRResult,
    ) -> str:
        """Save OCR evidence images."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        orig_path = EVIDENCE_DIR / f"ocr_original_{timestamp}.jpg"
        cv2.imwrite(str(orig_path), original)

        if preprocessed is not None:
            prep_path = EVIDENCE_DIR / f"ocr_preprocessed_{timestamp}.jpg"
            cv2.imwrite(str(prep_path), preprocessed)

        ocr_result.evidence_path = str(orig_path)
        return str(orig_path)

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "engine": self.engine_name,
            "available": True,
            "languages": OCR_CONFIG["language"],
        }
