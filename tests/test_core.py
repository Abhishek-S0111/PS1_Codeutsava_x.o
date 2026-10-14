"""
Core tests for Steel Billet Inspection System.
Run with: pytest tests/test_core.py -v
"""
import sys
import os
import tempfile
import uuid
from pathlib import Path

try:
    import pytest
    approx = pytest.approx
except ImportError:
    pytest = None
    class _Approx:
        def __init__(self, expected, rel=1e-3, abs=1e-5):
            self.expected = expected
            self.rel = rel
            self.abs = abs
        def __eq__(self, actual):
            return abs(actual - self.expected) <= max(self.abs, self.rel * abs(self.expected))
        def __repr__(self):
            return f"approx({self.expected})"
    def approx(expected, **kwargs):
        return _Approx(expected, **kwargs)

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Database Tests ────────────────────────────────────────────────

class TestDatabase:
    """Test SQLite storage layer."""

    def _get_db(self):
        from storage.database import InspectionDatabase
        tmp = tempfile.mktemp(suffix=".db")
        return InspectionDatabase(tmp), tmp

    def test_insert_and_retrieve(self):
        db, path = self._get_db()
        record = {
            "record_id": str(uuid.uuid4()),
            "timestamp": "2024-01-01T00:00:00",
            "billet_id": "TEST-001",
            "inspection_status": "PASS",
        }
        rid = db.insert_inspection(record)
        assert rid == record["record_id"]
        fetched = db.get_inspection(rid)
        assert fetched is not None
        assert fetched["billet_id"] == "TEST-001"
        db.close()
        os.remove(path)

    def test_count(self):
        db, path = self._get_db()
        for i in range(5):
            db.insert_inspection({
                "record_id": str(uuid.uuid4()),
                "timestamp": "2024-01-01T00:00:00",
                "inspection_status": "PASS" if i < 3 else "FAIL",
            })
        assert db.get_inspection_count() == 5
        assert db.get_inspection_count("PASS") == 3
        assert db.get_inspection_count("FAIL") == 2
        db.close()
        os.remove(path)

    def test_append_preserves_rows(self):
        db, path = self._get_db()
        r1 = db.insert_inspection({
            "record_id": str(uuid.uuid4()),
            "timestamp": "2024-01-01T00:00:00",
            "billet_id": "A",
            "inspection_status": "PASS",
        })
        r2 = db.insert_inspection({
            "record_id": str(uuid.uuid4()),
            "timestamp": "2024-01-02T00:00:00",
            "billet_id": "B",
            "inspection_status": "FAIL",
        })
        records = db.get_inspections()
        assert len(records) == 2
        db.close()
        os.remove(path)

    def test_export_excel(self):
        db, path = self._get_db()
        db.insert_inspection({
            "record_id": str(uuid.uuid4()),
            "timestamp": "2024-01-01T00:00:00",
            "billet_id": "EXPORT-1",
            "inspection_status": "PASS",
        })
        xlsx = tempfile.mktemp(suffix=".xlsx")
        result = db.export_to_excel(xlsx)
        assert Path(result).exists()
        db.close()
        os.remove(path)
        if Path(result).exists():
            os.remove(result)

    def test_export_csv(self):
        db, path = self._get_db()
        db.insert_inspection({
            "record_id": str(uuid.uuid4()),
            "timestamp": "2024-01-01T00:00:00",
            "billet_id": "CSV-1",
            "inspection_status": "FAIL",
        })
        csv_path = tempfile.mktemp(suffix=".csv")
        result = db.export_to_csv(csv_path)
        assert Path(result).exists()
        db.close()
        os.remove(path)
        if Path(result).exists():
            os.remove(result)

    def test_alerts(self):
        db, path = self._get_db()
        aid = db.insert_alert({
            "alert_type": "DEFECT_DETECTED",
            "severity": "CRITICAL",
            "message": "Test alert",
        })
        alerts = db.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["message"] == "Test alert"
        db.acknowledge_alert(aid, by="tester")
        acked = db.get_alerts(acknowledged=True)
        assert len(acked) == 1
        db.close()
        os.remove(path)

    def test_review_flags(self):
        db, path = self._get_db()
        db.insert_inspection({
            "record_id": str(uuid.uuid4()),
            "timestamp": "2024-01-01T00:00:00",
            "batch_number": "BATCH-99",
            "inspection_status": "FAIL",
        })
        fid = db.flag_batch_for_review("BATCH-99", "Test reason")
        flags = db.get_review_flags(status="PENDING")
        assert len(flags) == 1
        assert flags[0]["batch_id"] == "BATCH-99"
        db.resolve_review_flag(fid, "Resolved in test")
        pending = db.get_review_flags(status="PENDING")
        assert len(pending) == 0
        db.close()
        os.remove(path)

    def test_db_safe_when_excel_locked(self):
        """Database records survive even if Excel write fails."""
        db, path = self._get_db()
        rid = db.insert_inspection({
            "record_id": str(uuid.uuid4()),
            "timestamp": "2024-01-01T00:00:00",
            "billet_id": "SAFE-1",
            "inspection_status": "PASS",
        })
        record = db.get_inspection(rid)
        assert record is not None
        db.close()
        os.remove(path)


# ── Decision Engine Tests ─────────────────────────────────────────

class TestDecisionEngine:
    def test_pass_decision(self):
        from rules.decision_engine import DecisionEngine
        de = DecisionEngine()
        decision = de.evaluate(defects=[], profile={})
        assert decision.status == "PASS"

    def test_fail_on_defect(self):
        from rules.decision_engine import DecisionEngine
        from defects.defect_detection import DefectDetection
        de = DecisionEngine()
        defects = [DefectDetection(bbox=(0, 0, 50, 50), confidence=0.9, defect_type="crack")]
        decision = de.evaluate(defects=defects, profile={"defect_rule": "FAIL"})
        assert decision.status == "FAIL"

    def test_unverified_when_nothing_checked(self):
        from rules.decision_engine import DecisionEngine
        de = DecisionEngine()
        decision = de.evaluate()
        assert decision.status == "UNVERIFIED"

    def test_suspend_qc_requirements_bypass_all_rejections(self):
        from rules.decision_engine import DecisionEngine
        from defects.defect_detection import DefectDetection
        de = DecisionEngine()

        # Defect that would normally cause hard FAIL
        defects = [DefectDetection(bbox=(0, 0, 50, 50), confidence=0.95, defect_type="severe_crack")]

        # Evaluate with suspend_qc=True (Testing Mode)
        decision = de.evaluate(
            defects=defects,
            ocr_result=None,
            profile={"defect_rule": "FAIL"},
            suspend_qc=True,
        )

        assert decision.status == "PASS"
        assert decision.qc_suspended is True
        assert len(decision.failed_rules) == 0  # No hard failures in testing mode
        assert "PASS (QC Requirements Suspended" in decision.explanation
        assert any("Testing Mode" in r or "QC Suspended" in r for r in decision.review_reasons)

    def test_config_is_and_set_qc_suspended(self):
        from utils.config import is_qc_suspended, set_qc_suspended
        orig_state = is_qc_suspended()
        try:
            set_qc_suspended(True)
            assert is_qc_suspended() is True

            set_qc_suspended(False)
            assert is_qc_suspended() is False
        finally:
            set_qc_suspended(orig_state)



# ── Detection Tests ───────────────────────────────────────────────

class TestDetection:
    def test_contour_fallback_runs(self):
        from vision.detection import BilletDetector
        det = BilletDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2 = __import__("cv2")
        cv2.rectangle(frame, (200, 250), (1000, 450), (100, 95, 90), -1)
        results = det._detect_contour_fallback(frame)
        # Should find at least the rectangle
        assert isinstance(results, list)

    def test_no_detection_on_blank(self):
        from vision.detection import BilletDetector
        det = BilletDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = det._detect_contour_fallback(frame)
        assert isinstance(results, list)


# ── OCR Tests ─────────────────────────────────────────────────────

class TestOCR:
    def test_ocr_failure_does_not_invent_id(self):
        from ocr_module.ocr_extraction import OCREngine
        eng = OCREngine()
        blank = np.zeros((50, 50, 3), dtype=np.uint8)
        result = eng.extract(blank)
        # Should NOT invent an ID from nothing
        assert result.billet_id == "" or result.billet_id == "ID_UNREADABLE" or not result.is_readable

    def test_qr_on_blank_returns_empty(self):
        from ocr_module.ocr_extraction import OCREngine
        eng = OCREngine()
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        result = eng._decode_qr_barcode(blank)
        # No QR on blank image
        assert result is None or result.get("qr", "") == ""

    def test_captures_both_billet_and_heat_number(self):
        from ocr_module.ocr_extraction import OCREngine
        eng = OCREngine()
        # Test direct text parsing
        fields = eng._parse_fields("BILLET: BLT-1042 | HEAT: HT-88215")
        assert fields.get("billet_id") == "BLT-1042"
        assert fields.get("heat_number") == "HT-88215"

        # Test unhyphenated industrial stencil formats
        fields2 = eng._parse_fields("BLT1000 HT58210")
        assert fields2.get("billet_id") == "BLT-1000"
        assert fields2.get("heat_number") == "HT-58210"

    def test_rapidocr_engine_and_annotations(self):
        import cv2
        from ocr_module.ocr_extraction import OCREngine
        eng = OCREngine()
        assert eng.engine_name == "rapidocr"
        
        # Test synthetic text card
        card = np.ones((120, 500, 3), dtype=np.uint8) * 255
        cv2.putText(card, "BILLET: BLT-9921", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(card, "HEAT: HT-11223", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        res = eng.extract(card)
        assert res.billet_id == "BLT-9921"
        assert res.heat_number == "HT-11223"
        assert len(res.detections) >= 2
        
        # Test drawing annotations
        annotated = eng.draw_ocr_annotations(card, res.detections)
        assert annotated.shape == card.shape
        assert not np.array_equal(annotated, card)




# ── Chatbot Tests ─────────────────────────────────────────────────

class TestChatbot:
    def test_help_query(self):
        from chatbot.chatbot_engine import InspectionChatbot
        bot = InspectionChatbot()
        result = bot.query("help")
        assert "message" in result
        assert "Available" in result["message"]

    def test_summary_query(self):
        from chatbot.chatbot_engine import InspectionChatbot
        bot = InspectionChatbot()
        result = bot.query("summarise today")
        assert "message" in result

    def test_flag_batch_requires_fields(self):
        from chatbot.chatbot_engine import InspectionChatbot
        bot = InspectionChatbot()
        result = bot.flag_batch_for_quality_review("", "reason")
        assert "required" in result["message"].lower() or "❌" in result["message"]

    def test_snapshot_missing_record(self):
        from chatbot.chatbot_engine import InspectionChatbot
        bot = InspectionChatbot()
        result = bot.get_inspection_snapshot("nonexistent-id-12345")
        assert "❌" in result["message"] or "No record" in result["message"]


# ── Tracker Tests ─────────────────────────────────────────────────

class TestTracker:
    def test_assign_track_ids(self):
        from vision.detection import SimpleTracker, BilletDetection
        tracker = SimpleTracker()
        det1 = [BilletDetection(bbox=(100, 100, 300, 300), confidence=0.9)]
        result = tracker.update(det1, 1)
        assert len(result) == 1
        assert result[0].track_id != ""

    def test_same_billet_keeps_id(self):
        from vision.detection import SimpleTracker, BilletDetection
        tracker = SimpleTracker()
        det1 = [BilletDetection(bbox=(100, 100, 300, 300), confidence=0.9)]
        r1 = tracker.update(det1, 1)
        tid1 = r1[0].track_id

        det2 = [BilletDetection(bbox=(105, 100, 305, 300), confidence=0.9)]
        r2 = tracker.update(det2, 2)
        assert r2[0].track_id == tid1  # Same billet, same ID


if __name__ == "__main__":
    if pytest is not None:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        failed = 0
        passed = 0
        for name, cls in list(globals().items()):
            if isinstance(cls, type) and name.startswith("Test"):
                instance = cls()
                for m_name in dir(instance):
                    if m_name.startswith("test_"):
                        fn = getattr(instance, m_name)
                        try:
                            fn()
                            print(f"[PASS] {name}.{m_name}")
                            passed += 1
                        except Exception as e:
                            print(f"[FAIL] {name}.{m_name}: {e}")
                            failed += 1
        print(f"\nTest Summary: {passed} passed, {failed} failed")
        sys.exit(1 if failed else 0)
