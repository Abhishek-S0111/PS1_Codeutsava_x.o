"""
Chatbot Engine - Inspection log query chatbot.
Rule-based query interface with optional LLM enhancement.
"""
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from storage.database import get_database
from utils.config import CHATBOT_CONFIG
from utils.logger import logger


class InspectionChatbot:
    """Answers questions about inspection history from the database."""

    HELP_TEXT = """**Available queries:**
- *How many billets failed today?*
- *Show billets with defects*
- *Which heat numbers have repeated defects?*
- *Show billets with unreadable IDs*
- *Which batches need quality review?*
- *Show inspection history for billet ID [ID]*
- *What is the defect rate?*
- *Show recent alerts*
- *Summarise today's inspections*
- *Flag batch [BATCH] for review because [REASON]*
- *Show snapshot for record [ID]*

Type **help** to see this list again."""

    def __init__(self):
        self.db = get_database()

    # ── Public API ────────────────────────────────────────────────

    def query(self, user_input: str) -> Dict[str, Any]:
        """Process a natural-language query and return structured results."""
        text = user_input.strip().lower()

        if not text:
            return self._reply("Please enter a question about the inspection logs.")

        if text in ("help", "?", "commands"):
            return self._reply(self.HELP_TEXT)

        # ── Pattern matching ──────────────────────────────────────
        # Failed / passed counts
        if re.search(r"how many.*(fail|reject)", text):
            return self._count_by_status("FAIL", self._extract_date_hint(text))
        if re.search(r"how many.*(pass|ok|good)", text):
            return self._count_by_status("PASS", self._extract_date_hint(text))
        if re.search(r"how many.*inspect", text):
            return self._count_by_status(None, self._extract_date_hint(text))

        # Defect queries
        if re.search(r"(defect rate|defect %|defect percent)", text):
            return self._defect_rate()
        if re.search(r"(defect|defective|damaged)", text) and re.search(r"(heat|number)", text):
            return self._heat_numbers_with_defects()
        if re.search(r"(defect|defective|damaged)", text):
            return self._list_defective()

        # OCR / ID queries
        if re.search(r"(unreadable|missing).*(id|ocr|marking)", text):
            return self._unreadable_ids()
        if re.search(r"(id|billet).*history", text) or re.search(r"history.*for.*(id|billet)", text):
            bid = self._extract_id(user_input)
            return self._billet_history(bid)

        # Batch queries
        if re.search(r"(batch|batches).*(review|flag|quality)", text):
            return self._batches_needing_review()
        if re.search(r"flag\s+batch", text):
            return self._flag_batch(user_input)

        # Snapshot
        if re.search(r"(snapshot|detail|evidence).*for", text):
            rid = self._extract_id(user_input)
            return self._get_inspection_snapshot(rid)

        # Summary
        if re.search(r"(summar|overview|report|status)", text):
            return self._daily_summary(self._extract_date_hint(text))

        # Alerts
        if re.search(r"(alert|warning|notification)", text):
            return self._recent_alerts()

        # Review required
        if re.search(r"review.*(required|needed|pending)", text):
            return self._list_by_status("REVIEW_REQUIRED")

        # Unverified
        if re.search(r"unverified", text):
            return self._list_by_status("UNVERIFIED")

        # Rework
        if re.search(r"rework", text):
            return self._list_by_status("REWORK")

        # Tolerance / measurement
        if re.search(r"(out of tolerance|exceed|measurement|dimension)", text):
            return self._out_of_tolerance()

        # Drift
        if re.search(r"(drift|trend|shift)", text):
            return self._dimensional_drift()

        return self._reply(
            "I didn't understand that question. " + self.HELP_TEXT
        )

    # ── Chatbot tool: flag_batch_for_quality_review ──────────────

    def flag_batch_for_quality_review(self, batch_id: str, reason: str) -> Dict[str, Any]:
        """Flag a batch for quality-manager review."""
        if not batch_id:
            return self._reply("❌ Batch ID is required.")
        if not reason:
            return self._reply("❌ A reason for flagging is required.")

        # Verify batch exists
        records = self.db.get_inspections(batch_number=batch_id, limit=1)
        if not records:
            return self._reply(f"⚠️ No inspection records found for batch '{batch_id}'.")

        flag_id = self.db.flag_batch_for_review(batch_id, reason, flagged_by="chatbot")
        return self._reply(
            f"✅ Batch **{batch_id}** flagged for quality review.\n"
            f"- Reason: {reason}\n"
            f"- Flag ID: `{flag_id}`\n"
            f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # ── Chatbot tool: get_inspection_snapshot ─────────────────────

    def get_inspection_snapshot(self, record_id: str) -> Dict[str, Any]:
        """Retrieve detailed snapshot for a specific inspection."""
        if not record_id:
            return self._reply("❌ Record ID is required.")

        record = self.db.get_inspection(record_id)
        if not record:
            # Try searching by billet_id
            records = self.db.get_inspections(billet_id=record_id, limit=1)
            if records:
                record = records[0]
            else:
                return self._reply(f"❌ No record found for ID '{record_id}'.")

        missing = []
        fields = {
            "Record ID": record.get("record_id", "—"),
            "Timestamp": record.get("timestamp", "—"),
            "Billet ID": record.get("billet_id") or "ID_UNREADABLE",
            "Heat Number": record.get("heat_number") or "—",
            "Batch Number": record.get("batch_number") or "—",
            "Status": record.get("inspection_status", "—"),
            "Raw OCR": record.get("raw_ocr_text") or "—",
            "QR/Barcode": record.get("qr_barcode_result") or "—",
            "Length (mm)": record.get("measured_length_mm") or "—",
            "Width (mm)": record.get("measured_width_mm") or "—",
            "Height (mm)": record.get("measured_height_mm") or "—",
            "Calibration": record.get("calibration_status", "—"),
            "Defect": "Yes" if record.get("defect_detected") else "No",
            "Defect Type": record.get("defect_type") or "—",
            "Defect Confidence": record.get("defect_confidence") or "—",
            "Failed Rules": record.get("failed_rules") or "None",
            "Review Reason": record.get("review_reason") or "None",
            "Evidence": record.get("evidence_path") or "Not available",
        }

        for k, v in fields.items():
            if v in ("—", "Not available", None):
                missing.append(k)

        lines = [f"📋 **Inspection Snapshot**\n"]
        for k, v in fields.items():
            lines.append(f"- **{k}:** {v}")

        if missing:
            lines.append(f"\n⚠️ Missing evidence: {', '.join(missing)}")

        return self._reply("\n".join(lines), data=record)

    # ── Query implementations ────────────────────────────────────

    def _count_by_status(self, status: Optional[str], date_hint: str) -> Dict:
        filters = {}
        if date_hint:
            filters["date_from"] = date_hint
        if status:
            filters["status"] = status

        records = self.db.get_inspections(**filters, limit=10000)
        count = len(records)
        label = status or "total"
        period = f"since {date_hint}" if date_hint else "all time"
        return self._reply(
            f"📊 **{label.upper()}** inspections ({period}): **{count}**"
        )

    def _defect_rate(self) -> Dict:
        stats = self.db.get_stats()
        total = stats["total"]
        defects = stats["defect_count"]
        rate = stats["defect_rate"]
        return self._reply(
            f"📊 **Defect Rate:** {rate:.1f}%\n"
            f"- Total inspections: {total}\n"
            f"- Defects detected: {defects}"
        )

    def _heat_numbers_with_defects(self) -> Dict:
        records = self.db.get_inspections(limit=5000)
        heat_defects: Dict[str, int] = {}
        for r in records:
            if r.get("defect_detected") and r.get("heat_number"):
                hn = r["heat_number"]
                heat_defects[hn] = heat_defects.get(hn, 0) + 1

        if not heat_defects:
            return self._reply("No heat numbers with repeated defects found.")

        sorted_heats = sorted(heat_defects.items(), key=lambda x: -x[1])
        lines = ["🔥 **Heat numbers with defects:**\n"]
        for hn, count in sorted_heats[:15]:
            lines.append(f"- `{hn}`: {count} defect(s)")
        return self._reply("\n".join(lines))

    def _list_defective(self) -> Dict:
        records = self.db.get_inspections(limit=500)
        defective = [r for r in records if r.get("defect_detected")]
        if not defective:
            return self._reply("✅ No defective billets found.")

        lines = [f"⚠️ **{len(defective)} defective billet(s) found:**\n"]
        for r in defective[:20]:
            bid = r.get("billet_id") or "Unknown"
            dtype = r.get("defect_type") or "defect"
            conf = r.get("defect_confidence") or 0
            lines.append(f"- `{bid}` — {dtype} ({conf:.0%} confidence)")
        return self._reply("\n".join(lines))

    def _unreadable_ids(self) -> Dict:
        records = self.db.get_inspections(limit=5000)
        unreadable = [
            r for r in records
            if not r.get("billet_id") or r.get("billet_id") == "ID_UNREADABLE"
        ]
        if not unreadable:
            return self._reply("✅ All billet IDs are readable.")
        lines = [f"❓ **{len(unreadable)} billet(s) with unreadable IDs:**\n"]
        for r in unreadable[:20]:
            ts = r.get("timestamp", "")
            src = r.get("source_file") or "unknown"
            lines.append(f"- Record `{r['record_id'][:8]}…` at {ts} from {src}")
        return self._reply("\n".join(lines))

    def _billet_history(self, billet_id: str) -> Dict:
        if not billet_id:
            return self._reply("Please specify a billet ID. Example: *Show history for billet AB-12345*")
        records = self.db.get_inspections(billet_id=billet_id, limit=100)
        if not records:
            return self._reply(f"No records found for billet ID '{billet_id}'.")
        lines = [f"📜 **History for billet `{billet_id}`** ({len(records)} records):\n"]
        for r in records:
            status = r.get("inspection_status", "—")
            ts = r.get("timestamp", "—")
            lines.append(f"- [{ts}] Status: **{status}**")
        return self._reply("\n".join(lines))

    def _batches_needing_review(self) -> Dict:
        flags = self.db.get_review_flags(status="PENDING")
        if not flags:
            return self._reply("✅ No batches currently flagged for review.")
        lines = ["🚩 **Batches flagged for review:**\n"]
        for f in flags:
            lines.append(
                f"- Batch `{f['batch_id']}` — {f['reason']} "
                f"(flagged {f['created_at']})"
            )
        return self._reply("\n".join(lines))

    def _flag_batch(self, text: str) -> Dict:
        # Parse: "flag batch BATCH123 for review because reason text"
        match = re.search(
            r"flag\s+batch\s+(\S+)\s+(?:for\s+review\s+)?(?:because\s+)?(.+)",
            text, re.IGNORECASE,
        )
        if match:
            batch_id = match.group(1)
            reason = match.group(2).strip()
            return self.flag_batch_for_quality_review(batch_id, reason)
        return self._reply(
            "Usage: *flag batch BATCH_ID for review because REASON*"
        )

    def _list_by_status(self, status: str) -> Dict:
        records = self.db.get_inspections(status=status, limit=50)
        if not records:
            return self._reply(f"No {status} inspections found.")
        lines = [f"📋 **{status} inspections ({len(records)}):**\n"]
        for r in records[:20]:
            bid = r.get("billet_id") or "Unknown"
            ts = r.get("timestamp", "")
            reason = r.get("review_reason") or ""
            lines.append(f"- `{bid}` at {ts}" + (f" — {reason}" if reason else ""))
        return self._reply("\n".join(lines))

    def _out_of_tolerance(self) -> Dict:
        records = self.db.get_inspections(status="FAIL", limit=200)
        oot = [r for r in records if r.get("failed_rules") and "tolerance" in r.get("failed_rules", "").lower()]
        if not oot:
            return self._reply("✅ No out-of-tolerance measurements found.")
        lines = [f"📏 **{len(oot)} out-of-tolerance measurement(s):**\n"]
        for r in oot[:15]:
            bid = r.get("billet_id") or "Unknown"
            lines.append(f"- `{bid}`: {r.get('failed_rules', '')}")
        return self._reply("\n".join(lines))

    def _dimensional_drift(self) -> Dict:
        records = self.db.get_inspections(limit=500)
        valid = [
            r for r in records
            if r.get("measured_length_mm") is not None
        ]
        if len(valid) < 5:
            return self._reply(
                "⚠️ Insufficient data for drift analysis (need ≥5 measurements)."
            )
        lengths = [r["measured_length_mm"] for r in valid]
        mean_l = sum(lengths) / len(lengths)
        lines = [
            f"📈 **Dimensional Drift Analysis** (last {len(valid)} measurements):\n",
            f"- Mean length: {mean_l:.2f} mm",
            f"- Min: {min(lengths):.2f} mm",
            f"- Max: {max(lengths):.2f} mm",
            f"- Range: {max(lengths) - min(lengths):.2f} mm",
        ]
        return self._reply("\n".join(lines))

    def _daily_summary(self, date_hint: str) -> Dict:
        stats = self.db.get_stats()
        lines = [
            "📊 **Inspection Summary**\n",
            f"- **Total:** {stats['total']}",
            f"- ✅ Pass: {stats['pass']}",
            f"- ❌ Fail: {stats['fail']}",
            f"- 🔧 Rework: {stats['rework']}",
            f"- ⚠️ Review Required: {stats['review_required']}",
            f"- ❓ Unverified: {stats['unverified']}",
            f"- Defect Rate: {stats['defect_rate']:.1f}%",
            "",
            f"**Today:** {stats['today_total']} inspections "
            f"({stats['today_pass']} pass, {stats['today_fail']} fail)",
        ]
        return self._reply("\n".join(lines))

    def _recent_alerts(self) -> Dict:
        alerts = self.db.get_alerts(limit=20)
        if not alerts:
            return self._reply("✅ No recent alerts.")
        lines = ["🔔 **Recent Alerts:**\n"]
        for a in alerts:
            ack = "✔" if a.get("acknowledged") else "⬜"
            lines.append(
                f"- {ack} [{a.get('severity', 'INFO')}] "
                f"{a.get('message', '')} ({a.get('timestamp', '')})"
            )
        return self._reply("\n".join(lines))

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _reply(message: str, data: Any = None) -> Dict[str, Any]:
        return {"message": message, "data": data, "timestamp": datetime.now().isoformat()}

    @staticmethod
    def _extract_date_hint(text: str) -> str:
        if "today" in text:
            return datetime.now().strftime("%Y-%m-%d")
        if "yesterday" in text:
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        if "this week" in text:
            return (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if "this month" in text:
            return datetime.now().strftime("%Y-%m-01")
        return ""

    @staticmethod
    def _extract_id(text: str) -> str:
        # Try to find an ID-like token
        tokens = text.split()
        for t in reversed(tokens):
            cleaned = t.strip(".,;:!?'\"")
            if re.match(r"[A-Za-z0-9_\-]{3,}", cleaned):
                if cleaned.lower() not in (
                    "for", "the", "billet", "id", "record",
                    "show", "history", "snapshot", "detail",
                    "evidence",
                ):
                    return cleaned
        return ""
