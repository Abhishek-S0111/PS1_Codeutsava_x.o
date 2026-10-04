"""
Storage Module - SQLite Database and Excel/CSV Export
Handles persistent inspection records, alerts, and review flags.
"""
import sqlite3
import os
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import csv
try:
    import pandas as pd
except ImportError:
    pd = None

from utils.config import DATABASE_PATH, OUTPUT_DIR, BASE_DIR, DATA_DIR, EVIDENCE_DIR, UPLOAD_DIR
from utils.logger import logger


class InspectionDatabase:
    """Thread-safe SQLite database for inspection records."""
    
    _local = threading.local()
    
    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DATABASE_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conns"):
            self._local.conns = {}
        if self.db_path not in self._local.conns or self._local.conns[self.db_path] is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conns[self.db_path] = conn
        return self._local.conns[self.db_path]
    
    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS inspections (
                record_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                source_file TEXT,
                source_type TEXT DEFAULT 'upload',
                video_timestamp REAL,
                frame_number INTEGER,
                billet_track_id TEXT,
                billet_id TEXT,
                heat_number TEXT,
                batch_number TEXT,
                serial_number TEXT,
                raw_ocr_text TEXT,
                ocr_confidence REAL,
                qr_barcode_result TEXT,
                measured_length_mm REAL,
                measured_width_mm REAL,
                measured_height_mm REAL,
                measured_diameter_mm REAL,
                nominal_length_mm REAL,
                nominal_width_mm REAL,
                nominal_height_mm REAL,
                length_tolerance_mm REAL,
                width_tolerance_mm REAL,
                height_tolerance_mm REAL,
                calibration_status TEXT DEFAULT 'NOT_CALIBRATED',
                calibration_id TEXT,
                measurement_confidence REAL,
                defect_detected INTEGER DEFAULT 0,
                defect_type TEXT,
                defect_confidence REAL,
                defect_count INTEGER DEFAULT 0,
                inspection_status TEXT DEFAULT 'PROCESSING',
                failed_rules TEXT,
                review_reason TEXT,
                profile_name TEXT,
                model_version TEXT,
                evidence_path TEXT,
                defect_evidence_path TEXT,
                ocr_evidence_path TEXT,
                operator_notes TEXT,
                operator_correction TEXT,
                is_demo INTEGER DEFAULT 0,
                plant_info TEXT,
                line_info TEXT,
                shift_info TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                record_id TEXT,
                billet_track_id TEXT,
                alert_type TEXT NOT NULL,
                severity TEXT DEFAULT 'WARNING',
                message TEXT NOT NULL,
                measurement_value REAL,
                detection_info TEXT,
                evidence_path TEXT,
                reason TEXT,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at TEXT,
                operator_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS review_flags (
                flag_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                flagged_by TEXT DEFAULT 'system',
                session_id TEXT,
                status TEXT DEFAULT 'PENDING',
                resolution TEXT,
                resolved_by TEXT,
                resolved_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS calibrations (
                calibration_id TEXT PRIMARY KEY,
                method TEXT NOT NULL,
                pixels_per_mm REAL,
                reference_dimension_mm REAL,
                reprojection_error REAL,
                camera_matrix TEXT,
                dist_coeffs TEXT,
                valid_until TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_inspections_status ON inspections(inspection_status);
            CREATE INDEX IF NOT EXISTS idx_inspections_billet_id ON inspections(billet_id);
            CREATE INDEX IF NOT EXISTS idx_inspections_batch ON inspections(batch_number);
            CREATE INDEX IF NOT EXISTS idx_inspections_heat ON inspections(heat_number);
            CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
            CREATE INDEX IF NOT EXISTS idx_alerts_record ON alerts(record_id);
        """)
        conn.commit()
    
    # ─── Inspection Records ───────────────────────────────────────
    
    def insert_inspection(self, record: Dict[str, Any]) -> str:
        """Insert a new inspection record. Returns the record_id."""
        if "record_id" not in record:
            record["record_id"] = str(uuid.uuid4())
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()
        
        columns = list(record.keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)
        values = [record[c] for c in columns]
        
        conn = self._get_conn()
        try:
            conn.execute(
                f"INSERT OR REPLACE INTO inspections ({col_str}) VALUES ({placeholders})",
                values
            )
            conn.commit()
            logger.info(f"Inserted inspection record: {record['record_id']}")
            return record["record_id"]
        except Exception as e:
            logger.error(f"Failed to insert inspection: {e}")
            conn.rollback()
            raise
    
    def update_inspection(self, record_id: str, updates: Dict[str, Any]):
        """Update an existing inspection record."""
        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [record_id]
        
        conn = self._get_conn()
        try:
            conn.execute(
                f"UPDATE inspections SET {set_clause} WHERE record_id = ?",
                values
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to update inspection {record_id}: {e}")
            conn.rollback()
            raise
    
    @staticmethod
    def resolve_path(path: Optional[str]) -> Optional[str]:
        """Resolve a file path, ensuring portable behavior across machines and imports."""
        if not path:
            return path
        p = Path(path)
        if p.exists():
            return str(p.resolve())
        
        # Check by filename in known directories
        if (EVIDENCE_DIR / p.name).exists():
            return str((EVIDENCE_DIR / p.name).resolve())
        if (UPLOAD_DIR / p.name).exists():
            return str((UPLOAD_DIR / p.name).resolve())
        if (BASE_DIR / path).exists():
            return str((BASE_DIR / path).resolve())
            
        # Match common subdirectories like data/evidence/filename
        norm = path.replace("\\", "/")
        for sub, target_dir in [
            ("data/evidence", EVIDENCE_DIR),
            ("data/uploads", UPLOAD_DIR),
            ("data/outputs", OUTPUT_DIR),
        ]:
            if sub in norm:
                rel = norm.split(sub, 1)[1].lstrip("/")
                cand = target_dir / rel
                if cand.exists():
                    return str(cand.resolve())
                    
        return path

    def _normalize_record_paths(self, record: Optional[Dict]) -> Optional[Dict]:
        """Normalize all path fields in an inspection record."""
        if not record:
            return record
        rec = dict(record)
        for key in ["evidence_path", "defect_evidence_path", "ocr_evidence_path", "source_file"]:
            if key in rec and rec[key]:
                rec[key] = self.resolve_path(rec[key])
        return rec

    def get_inspection(self, record_id: str) -> Optional[Dict]:
        """Get a single inspection record by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM inspections WHERE record_id = ?", (record_id,)
        ).fetchone()
        return self._normalize_record_paths(dict(row)) if row else None
    
    def get_inspections(
        self,
        status: str = None,
        batch_number: str = None,
        heat_number: str = None,
        billet_id: str = None,
        date_from: str = None,
        date_to: str = None,
        limit: int = 500,
        offset: int = 0,
        is_demo: bool = None,
    ) -> List[Dict]:
        """Query inspections with optional filters."""
        query = "SELECT * FROM inspections WHERE 1=1"
        params = []
        
        if status:
            query += " AND inspection_status = ?"
            params.append(status)
        if batch_number:
            query += " AND batch_number LIKE ?"
            params.append(f"%{batch_number}%")
        if heat_number:
            query += " AND heat_number LIKE ?"
            params.append(f"%{heat_number}%")
        if billet_id:
            query += " AND billet_id LIKE ?"
            params.append(f"%{billet_id}%")
        if date_from:
            query += " AND timestamp >= ?"
            params.append(date_from)
        if date_to:
            query += " AND timestamp <= ?"
            params.append(date_to)
        if is_demo is not None:
            query += " AND is_demo = ?"
            params.append(1 if is_demo else 0)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()
        return [self._normalize_record_paths(dict(r)) for r in rows]
    
    def get_inspection_count(self, status: str = None) -> int:
        """Count inspections, optionally filtered by status."""
        query = "SELECT COUNT(*) FROM inspections"
        params = []
        if status:
            query += " WHERE inspection_status = ?"
            params.append(status)
        conn = self._get_conn()
        return conn.execute(query, params).fetchone()[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate inspection statistics."""
        conn = self._get_conn()
        stats = {}
        
        stats["total"] = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        
        for status in ["PASS", "FAIL", "REWORK", "REVIEW_REQUIRED", "UNVERIFIED", "PROCESSING"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM inspections WHERE inspection_status = ?", (status,)
            ).fetchone()[0]
            stats[status.lower()] = count
        
        stats["defect_count"] = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE defect_detected = 1"
        ).fetchone()[0]
        
        stats["defect_rate"] = (
            (stats["defect_count"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        )
        
        # Today's stats
        today = datetime.now().strftime("%Y-%m-%d")
        stats["today_total"] = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE timestamp LIKE ?", (f"{today}%",)
        ).fetchone()[0]
        stats["today_pass"] = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE timestamp LIKE ? AND inspection_status = 'PASS'",
            (f"{today}%",)
        ).fetchone()[0]
        stats["today_fail"] = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE timestamp LIKE ? AND inspection_status = 'FAIL'",
            (f"{today}%",)
        ).fetchone()[0]
        
        # Recent trend
        rows = conn.execute("""
            SELECT DATE(timestamp) as date, inspection_status, COUNT(*) as cnt
            FROM inspections
            GROUP BY DATE(timestamp), inspection_status
            ORDER BY date DESC
            LIMIT 100
        """).fetchall()
        stats["daily_trend"] = [dict(r) for r in rows]
        
        return stats
    
    def get_defect_summary(self) -> List[Dict]:
        """Get defect type summary."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT defect_type, COUNT(*) as count, AVG(defect_confidence) as avg_confidence
            FROM inspections
            WHERE defect_detected = 1 AND defect_type IS NOT NULL
            GROUP BY defect_type
            ORDER BY count DESC
        """).fetchall()
        return [dict(r) for r in rows]
    
    def check_duplicate_billet(self, billet_id: str, time_window_seconds: int = 60) -> bool:
        """Check if a billet ID was recently recorded (dedup)."""
        if not billet_id or billet_id == "ID_UNREADABLE":
            return False
        conn = self._get_conn()
        row = conn.execute("""
            SELECT COUNT(*) FROM inspections
            WHERE billet_id = ? AND timestamp > datetime('now', ? || ' seconds')
        """, (billet_id, f"-{time_window_seconds}")).fetchone()
        return row[0] > 0
    
    # ─── Alerts ───────────────────────────────────────────────────
    
    def insert_alert(self, alert: Dict[str, Any]) -> str:
        if "alert_id" not in alert:
            alert["alert_id"] = str(uuid.uuid4())
        if "timestamp" not in alert:
            alert["timestamp"] = datetime.now().isoformat()
        
        columns = list(alert.keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)
        values = [alert[c] for c in columns]
        
        conn = self._get_conn()
        conn.execute(
            f"INSERT INTO alerts ({col_str}) VALUES ({placeholders})", values
        )
        conn.commit()
        return alert["alert_id"]
    
    def get_alerts(self, acknowledged: bool = None, limit: int = 100) -> List[Dict]:
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        if acknowledged is not None:
            query += " AND acknowledged = ?"
            params.append(1 if acknowledged else 0)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()
        alerts = [dict(r) for r in rows]
        for a in alerts:
            if a.get("evidence_path"):
                a["evidence_path"] = self.resolve_path(a["evidence_path"])
        return alerts
    
    def acknowledge_alert(self, alert_id: str, by: str = "operator", notes: str = ""):
        conn = self._get_conn()
        conn.execute("""
            UPDATE alerts SET acknowledged = 1, acknowledged_by = ?,
            acknowledged_at = ?, operator_notes = ?
            WHERE alert_id = ?
        """, (by, datetime.now().isoformat(), notes, alert_id))
        conn.commit()
    
    # ─── Review Flags ────────────────────────────────────────────
    
    def flag_batch_for_review(self, batch_id: str, reason: str, flagged_by: str = "system") -> str:
        flag_id = str(uuid.uuid4())
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO review_flags (flag_id, batch_id, reason, flagged_by)
            VALUES (?, ?, ?, ?)
        """, (flag_id, batch_id, reason, flagged_by))
        conn.commit()
        logger.info(f"Batch {batch_id} flagged for review: {reason}")
        return flag_id
    
    def get_review_flags(self, status: str = None) -> List[Dict]:
        query = "SELECT * FROM review_flags"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    
    def resolve_review_flag(self, flag_id: str, resolution: str, resolved_by: str = "operator"):
        conn = self._get_conn()
        conn.execute("""
            UPDATE review_flags SET status = 'RESOLVED', resolution = ?,
            resolved_by = ?, resolved_at = ?
            WHERE flag_id = ?
        """, (resolution, resolved_by, datetime.now().isoformat(), flag_id))
        conn.commit()
    
    # ─── Calibration ─────────────────────────────────────────────
    
    def save_calibration(self, cal_data: Dict[str, Any]) -> str:
        if "calibration_id" not in cal_data:
            cal_data["calibration_id"] = str(uuid.uuid4())
        
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO calibrations
            (calibration_id, method, pixels_per_mm, reference_dimension_mm,
             reprojection_error, camera_matrix, dist_coeffs, valid_until)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cal_data["calibration_id"],
            cal_data.get("method", "reference_object"),
            cal_data.get("pixels_per_mm"),
            cal_data.get("reference_dimension_mm"),
            cal_data.get("reprojection_error"),
            cal_data.get("camera_matrix"),
            cal_data.get("dist_coeffs"),
            cal_data.get("valid_until"),
        ))
        conn.commit()
        return cal_data["calibration_id"]
    
    def get_latest_calibration(self) -> Optional[Dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM calibrations ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None
    
    # ─── Export ───────────────────────────────────────────────────
    
    @staticmethod
    def _enrich_records_with_cm(records: List[Dict]) -> List[Dict]:
        """Compute and add centimeter (cm) dimensional values to records."""
        for r in records:
            l_mm = r.get("measured_length_mm")
            w_mm = r.get("measured_width_mm")
            h_mm = r.get("measured_height_mm")
            r["measured_length_cm"] = round(l_mm / 10.0, 2) if l_mm is not None else None
            r["measured_width_cm"] = round(w_mm / 10.0, 2) if w_mm is not None else None
            r["measured_height_cm"] = round(h_mm / 10.0, 2) if h_mm is not None else None
            if l_mm and w_mm and h_mm:
                r["dimensions_cm"] = f"{round(l_mm/10.0, 1)} x {round(w_mm/10.0, 1)} x {round(h_mm/10.0, 1)} cm"
            elif l_mm and w_mm:
                r["dimensions_cm"] = f"{round(l_mm/10.0, 1)} x {round(w_mm/10.0, 1)} cm"
            else:
                r["dimensions_cm"] = "—"
        return records

    def export_to_dataframe(self, **filters) -> Any:
        """Export inspection records to a pandas DataFrame with cm dimensions and clear priority ordering."""
        if pd is None:
            raise RuntimeError("pandas is not installed. Install pandas or use get_inspections() / export_to_csv().")
        records = self.get_inspections(**filters)
        self._enrich_records_with_cm(records)
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Priority column mapping: Billet ID, Heat Number, and cm dimensions first
        priority_cols = [
            ("billet_id", "Billet Number"),
            ("heat_number", "Heat Number"),
            ("batch_number", "Batch Number"),
            ("inspection_status", "Inspection Status"),
            ("dimensions_cm", "Dimensions (cm)"),
            ("measured_length_cm", "Length (cm)"),
            ("measured_width_cm", "Width (cm)"),
            ("measured_height_cm", "Height (cm)"),
            ("defect_detected", "Defect Detected"),
            ("defect_type", "Defect Type"),
            ("defect_count", "Defect Count"),
            ("measured_length_mm", "Length (mm)"),
            ("measured_width_mm", "Width (mm)"),
            ("measured_height_mm", "Height (mm)"),
            ("failed_rules", "Failed Rules"),
            ("calibration_status", "Calibration Status"),
            ("ocr_confidence", "OCR Confidence"),
            ("timestamp", "Timestamp"),
            ("source_file", "Source File"),
        ]

        existing_map = {orig: new_name for orig, new_name in priority_cols if orig in df.columns}
        other_cols = [c for c in df.columns if c not in [p[0] for p in priority_cols]]

        ordered_cols = list(existing_map.keys()) + other_cols
        df = df[ordered_cols]
        df = df.rename(columns=existing_map)
        return df

    def export_to_excel(self, filepath: str = None, **filters) -> str:
        """Export to Excel file with executive industrial styling and safe write handling."""
        if pd is None:
            logger.warning("pandas not installed. Falling back to CSV export.")
            return self.export_to_csv(filepath=filepath.replace(".xlsx", ".csv") if filepath else None, **filters)

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(OUTPUT_DIR / f"inspection_report_{timestamp}.xlsx")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df = self.export_to_dataframe(**filters)

        if df.empty:
            logger.warning("No records to export")
            return filepath

        # Safe write: write to temp then rename
        temp_path = filepath.rsplit(".", 1)[0] + "_temp.xlsx"
        try:
            # If file exists, append
            if os.path.exists(filepath):
                try:
                    existing_df = pd.read_excel(filepath, engine="openpyxl")
                    df = pd.concat([existing_df, df], ignore_index=True)
                    dedup_col = "record_id" if "record_id" in df.columns else ("Billet Number" if "Billet Number" in df.columns else None)
                    if dedup_col:
                        df = df.drop_duplicates(subset=[dedup_col], keep="last")
                except Exception as e:
                    logger.warning(f"Could not read existing Excel: {e}. Creating new file.")

            df.to_excel(temp_path, index=False, engine="openpyxl")

            # Style the spreadsheet with openpyxl
            try:
                import openpyxl
                from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
                from openpyxl.utils import get_column_letter

                wb = openpyxl.load_workbook(temp_path)
                ws = wb.active
                ws.title = "Inspection QC Log"

                header_fill = PatternFill(start_color="0F2942", end_color="0F2942", fill_type="solid")
                header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
                zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
                thin_border = Border(
                    left=Side(style='thin', color='E2E8F0'),
                    right=Side(style='thin', color='E2E8F0'),
                    top=Side(style='thin', color='E2E8F0'),
                    bottom=Side(style='thin', color='E2E8F0')
                )

                # Header row styling
                ws.row_dimensions[1].height = 28
                for col_num in range(1, ws.max_column + 1):
                    cell = ws.cell(row=1, column=col_num)
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

                # Data rows styling
                for row_num in range(2, ws.max_row + 1):
                    ws.row_dimensions[row_num].height = 20
                    is_even = (row_num % 2 == 0)
                    for col_num in range(1, ws.max_column + 1):
                        cell = ws.cell(row=row_num, column=col_num)
                        cell.border = thin_border
                        if is_even:
                            cell.fill = zebra_fill
                        col_header = str(ws.cell(row=1, column=col_num).value or "")
                        if any(k in col_header for k in ["Status", "ID", "Number", "Defect Detected", "Dimensions"]):
                            cell.alignment = Alignment(horizontal="center", vertical="center")
                        else:
                            cell.alignment = Alignment(vertical="center")

                # Auto-fit column widths
                for col in ws.columns:
                    max_len = 0
                    col_letter = get_column_letter(col[0].column)
                    for cell in col:
                        val_str = str(cell.value or "")
                        max_len = max(max_len, len(val_str))
                    ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 40)

                wb.save(temp_path)
            except Exception as e:
                logger.debug(f"Excel styling skipped: {e}")

            # Atomic rename
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_path, filepath)
            logger.info(f"Excel exported: {filepath} ({len(df)} records)")
            return filepath
        except PermissionError:
            logger.error(f"Excel file locked: {filepath}. Records safe in database.")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Fall back to CSV
            csv_path = filepath.replace(".xlsx", ".csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Fallback CSV exported: {csv_path}")
            return csv_path
        except Exception as e:
            logger.error(f"Export failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    def export_to_csv(self, filepath: str = None, **filters) -> str:
        """Export to CSV file with or without pandas."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(OUTPUT_DIR / f"inspection_report_{timestamp}.csv")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        records = self.get_inspections(**filters)
        self._enrich_records_with_cm(records)
        
        if pd is not None:
            df = pd.DataFrame(records) if records else pd.DataFrame()
            df.to_csv(filepath, index=False)
        else:
            # Pure Python CSV fallback
            if records:
                fieldnames = list(records[0].keys())
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(records)
            else:
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    f.write("")
        
        logger.info(f"CSV exported: {filepath} ({len(records)} records)")
        return filepath
    
    def close(self):
        if hasattr(self._local, "conns") and self.db_path in self._local.conns:
            conn = self._local.conns.pop(self.db_path, None)
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass


# Singleton
_db_instance = None

def get_database() -> InspectionDatabase:
    """Get or create the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = InspectionDatabase()
    return _db_instance
