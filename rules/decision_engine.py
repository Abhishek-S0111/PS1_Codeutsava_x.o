"""
Decision Engine - Configurable quality rules for inspection status.
Produces PASS / FAIL / REWORK / REVIEW_REQUIRED / UNVERIFIED decisions
based strictly on surface defect detection and OCR/traceability markings.
(Dimensional analysis and camera calibration have been completely removed.)
"""
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from utils.config import DECISION_STATUSES, DEFECT_CONFIG, is_qc_suspended
from utils.logger import logger


@dataclass
class InspectionDecision:
    """Final inspection decision with supporting evidence."""
    status: str = "PROCESSING"
    failed_rules: List[str] = field(default_factory=list)
    review_reasons: List[str] = field(default_factory=list)
    defect_status: str = "NOT_CHECKED"
    ocr_status: str = "NOT_CHECKED"
    overall_confidence: float = 0.0
    explanation: str = ""
    timestamp: str = ""
    qc_suspended: bool = False
    # Legacy compatibility fields (defaults to NOT_APPLICABLE)
    dimension_status: str = "NOT_APPLICABLE"
    calibration_status: str = "NOT_APPLICABLE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inspection_status": self.status,
            "failed_rules": "; ".join(self.failed_rules) if self.failed_rules else "",
            "review_reason": "; ".join(self.review_reasons) if self.review_reasons else "",
        }


class DecisionEngine:
    """Evaluates inspection results against configured rules."""

    def __init__(self, suspend_qc: Optional[bool] = None):
        self.suspend_qc = suspend_qc

    def evaluate(
        self,
        defects: List[Any] = None,
        ocr_result: Any = None,
        profile: Dict[str, Any] = None,
        suspend_qc: Optional[bool] = None,
        measurements: Optional[Any] = None,       # Legacy ignored
        calibration_valid: Optional[bool] = None,  # Legacy ignored
        **kwargs,
    ) -> InspectionDecision:
        """Produce a quality decision from defect and OCR inspection inputs.

        Args:
            defects: List of DefectDetection objects
            ocr_result: OCRResult object
            profile: Active billet profile dict
            suspend_qc: Optional override to suspend all QC rejection criteria (Testing Mode)
        """
        decision = InspectionDecision(timestamp=datetime.now().isoformat())
        profile = profile or {}

        # Determine whether QC requirements are suspended for testing
        qc_suspended = (
            suspend_qc
            if suspend_qc is not None
            else (self.suspend_qc if self.suspend_qc is not None else is_qc_suspended())
        )
        decision.qc_suspended = qc_suspended

        # ── 1. Defect check ───────────────────────────────────────
        if defects is not None:
            defect_status, defect_failures = self._check_defects(defects, profile)
            if qc_suspended:
                decision.defect_status = "PASS (QC Suspended)" if defect_failures else defect_status
                if defect_failures:
                    decision.review_reasons.extend([f"[QC Suspended] {f}" for f in defect_failures])
            else:
                decision.defect_status = defect_status
                decision.failed_rules.extend(defect_failures)
        else:
            decision.defect_status = "NOT_CHECKED"

        # ── 2. OCR / ID check ────────────────────────────────────
        if ocr_result is not None:
            ocr_status, ocr_warnings = self._check_ocr(ocr_result, profile)
            if qc_suspended:
                decision.ocr_status = "PASS (QC Suspended)" if (ocr_status in ("FAIL", "REVIEW") or ocr_warnings) else ocr_status
                if ocr_warnings:
                    decision.review_reasons.extend([f"[QC Suspended] {w}" for w in ocr_warnings])
            else:
                decision.ocr_status = ocr_status
                decision.review_reasons.extend(ocr_warnings)
        else:
            decision.ocr_status = "NOT_CHECKED"

        # ── 3. Aggregate final status ────────────────────────────
        if qc_suspended:
            decision.status = "PASS"
            decision.failed_rules = []  # Clear hard failures so billet passes QC
            raw_conf = self._compute_confidence(decision, defects, ocr_result)
            decision.overall_confidence = max(raw_conf, 0.92) if raw_conf > 0 else 0.95
            decision.explanation = "PASS (QC Requirements Suspended - Testing Mode Active)"
            if decision.review_reasons:
                decision.explanation += " | Notes: " + "; ".join(decision.review_reasons[:3])
        else:
            decision.status = self._aggregate_status(decision, profile)
            decision.overall_confidence = self._compute_confidence(decision, defects, ocr_result)
            decision.explanation = self._build_explanation(decision)

        return decision

    # ── Internal rule evaluators ─────────────────────────────────

    def _check_defects(
        self, defects: List, profile: Dict
    ) -> Tuple[str, List[str]]:
        if not defects:
            return "PASS", []

        failures = []
        review_threshold = DEFECT_CONFIG.get("confidence_review_threshold", 0.7)
        defect_rule = profile.get("defect_rule", "FAIL")

        high_conf_defects = [
            d for d in defects
            if getattr(d, "confidence", 0) >= review_threshold
        ]
        low_conf_defects = [
            d for d in defects
            if getattr(d, "confidence", 0) < review_threshold
        ]

        if high_conf_defects:
            types = set(getattr(d, "defect_type", "defect") for d in high_conf_defects)
            max_conf = max(getattr(d, "confidence", 0) for d in high_conf_defects)
            failures.append(
                f"Defect detected ({', '.join(types)}) – "
                f"confidence {max_conf:.0%} → rule: {defect_rule}"
            )
            return defect_rule, failures

        if low_conf_defects:
            return "REVIEW", [
                "Possible defect(s) detected with low confidence. Manual review recommended."
            ]

        return "PASS", []

    def _check_ocr(self, ocr_result, profile: Dict) -> Tuple[str, List[str]]:
        warnings: List[str] = []
        required_fields = profile.get("required_fields", [])
        is_readable = getattr(ocr_result, "is_readable", False)

        if not is_readable:
            if "billet_id" in required_fields:
                warnings.append("Billet ID required but not readable (ID_UNREADABLE).")
                return "FAIL", warnings
            else:
                warnings.append("OCR could not extract readable text.")
                return "REVIEW", warnings

        # Check required fields
        for field_name in required_fields:
            value = getattr(ocr_result, field_name, "")
            if not value:
                warnings.append(f"Required field '{field_name}' not found in OCR result.")

        # Forward validation warnings from OCR engine
        ocr_warnings = getattr(ocr_result, "validation_warnings", [])
        warnings.extend(ocr_warnings)

        if any("required" in w.lower() and "not found" in w.lower() for w in warnings):
            return "REVIEW", warnings

        return "PASS", warnings

    def _aggregate_status(self, decision: InspectionDecision, profile: Dict) -> str:
        """Determine the aggregated inspection status."""
        statuses = [
            decision.defect_status,
            decision.ocr_status,
        ]

        # 1. Any hard failure -> FAIL
        if "FAIL" in statuses:
            final = "FAIL"
        # 2. Rework required -> REWORK
        elif "REWORK" in statuses:
            final = "REWORK"
        # 3. Review required -> REVIEW_REQUIRED
        elif any(s in ("REVIEW", "REVIEW_REQUIRED") for s in statuses):
            final = "REVIEW_REQUIRED"
        # 4. At least one passed and none failed/needs review -> PASS
        elif "PASS" in statuses:
            final = "PASS"
        # 5. Nothing checked or only unconfigured -> UNVERIFIED
        else:
            final = "UNVERIFIED"

        return final

    def _compute_confidence(
        self, decision: InspectionDecision, defects, ocr_result
    ) -> float:
        scores = []
        if defects:
            for d in defects:
                scores.append(getattr(d, "confidence", 0.5))
        if ocr_result and getattr(ocr_result, "confidence", 0) > 0:
            scores.append(getattr(ocr_result, "confidence", 0.8))
        return round(sum(scores) / max(len(scores), 1), 3) if scores else 0.95

    def _build_explanation(self, decision: InspectionDecision) -> str:
        parts = [f"Status: {decision.status}"]
        if decision.failed_rules:
            parts.append("Failed rules: " + "; ".join(decision.failed_rules))
        if decision.review_reasons:
            parts.append("Review notes: " + "; ".join(decision.review_reasons))
        return " | ".join(parts)
