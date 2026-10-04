"""
Steel Billet Inspection System - Core Configuration
CodeUtsava X.0 - Team INIT
"""
import os
try:
    import yaml
except ImportError:
    yaml = None
from pathlib import Path
from datetime import datetime, timedelta

# ─── Project Paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
EVIDENCE_DIR = DATA_DIR / "evidence"
CALIBRATION_DIR = DATA_DIR / "calibration"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

env_db = os.getenv("DATABASE_PATH")
if env_db:
    _p = Path(env_db)
    DATABASE_PATH = _p if _p.is_absolute() else (BASE_DIR / _p)
else:
    DATABASE_PATH = DATA_DIR / "inspection.db"

# Create directories
for d in [UPLOAD_DIR, OUTPUT_DIR, EVIDENCE_DIR, CALIBRATION_DIR, DEMO_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Application Settings ────────────────────────────────────────
APP_TITLE = "Steel Billet Inspection & Traceability System"
APP_VERSION = "1.0.0"
APP_TEAM = "Team INIT - CodeUtsava X.0"
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# ─── Video Processing ────────────────────────────────────────────
VIDEO_CONFIG = {
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv"],
    "default_frame_skip": int(os.getenv("DEFAULT_FRAME_SKIP", "3")),
    "max_resolution": (1920, 1080),
    "default_resolution": (1280, 720),
    "fps_target": 30,
    "batch_size": 8,
}

# ─── Detection Configuration ─────────────────────────────────────
DETECTION_CONFIG = {
    "model_path": str(MODELS_DIR / "billet_detector.pt"),
    "confidence_threshold": float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.5")),
    "nms_threshold": 0.45,
    "input_size": 640,
    "device": "cuda" if os.getenv("GPU_ENABLED", "false").lower() == "true" else "cpu",
    "classes": ["billet"],
}

# ─── Defect Detection ────────────────────────────────────────────
DEFECT_CONFIG = {
    "model_path": str(MODELS_DIR / "defect_detector.pt"),
    "confidence_threshold": 0.5,
    "confidence_review_threshold": 0.7,
    "nms_threshold": 0.3,
    "device": "cpu",
    "patch_size": 640,
    "patch_overlap": 0.2,
    "classes": [
        "defect",  # Unified class for initial model
    ],
    "detailed_classes": [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches",
    ],
}

# ─── OCR Configuration ───────────────────────────────────────────
OCR_CONFIG = {
    "engine": "rapidocr",
    "language": ["en"],
    "confidence_threshold": 0.4,
    "gpu_enabled": False,
    "preprocessing_methods": [
        "grayscale",
        "contrast_enhance",
        "denoise",
        "adaptive_threshold",
        "sharpen",
    ],
    "id_patterns": {
        "billet_id": r"(?:BLT|BILLET|ID|BL)[-\s:#]*[A-Z0-9]{3,10}|[A-Z]{1,4}[-\s]?\d{3,8}",
        "heat_number": r"(?:HEAT|HT|H)[-\s:#]*[A-Z0-9]{3,10}|H(?:EAT)?[-\s:#]?\d{3,8}",
        "batch_number": r"(?:BATCH|BCH|BT)[-\s:#]*[A-Z0-9]{3,10}|B(?:ATCH)?[-\s:#]?\d{3,8}",
    },
}

# ─── Calibration ─────────────────────────────────────────────────
CALIBRATION_CONFIG = {
    "validity_days": 30,
    "checkerboard_size": (9, 6),
    "square_size_mm": 25.0,
    "min_calibration_images": 5,
    "reprojection_error_threshold": 1.0,
}

# ─── Measurement ─────────────────────────────────────────────────
MEASUREMENT_CONFIG = {
    "accuracy_target_percent": 1.0,
    "min_confidence_for_pass": 0.7,
    "units": ["mm", "cm", "m"],
    "default_unit": "mm",
}

# ─── Default Billet Profiles ─────────────────────────────────────
DEFAULT_BILLET_PROFILES = {
    "ISI Standard 750x150x150mm": {
        "nominal_length_mm": 750.0,
        "nominal_width_mm": 150.0,
        "nominal_height_mm": 150.0,
        "length_tolerance_mm": 7.5,
        "width_tolerance_mm": 3.0,
        "height_tolerance_mm": 3.0,
        "cross_section": "square",
        "defect_rule": "FAIL",
        "required_fields": ["billet_id"],
        "description": "Standard square billet per ISI specification",
    },
    "ISI Standard 1000x200x200mm": {
        "nominal_length_mm": 1000.0,
        "nominal_width_mm": 200.0,
        "nominal_height_mm": 200.0,
        "length_tolerance_mm": 10.0,
        "width_tolerance_mm": 4.0,
        "height_tolerance_mm": 4.0,
        "cross_section": "square",
        "defect_rule": "FAIL",
        "required_fields": ["billet_id"],
        "description": "Large square billet per ISI specification",
    },
    "Custom Profile": {
        "nominal_length_mm": 0.0,
        "nominal_width_mm": 0.0,
        "nominal_height_mm": 0.0,
        "length_tolerance_mm": 0.0,
        "width_tolerance_mm": 0.0,
        "height_tolerance_mm": 0.0,
        "cross_section": "square",
        "defect_rule": "REVIEW",
        "required_fields": [],
        "description": "User-defined custom billet profile",
    },
}

# ─── Decision Rules ──────────────────────────────────────────────
DECISION_STATUSES = {
    "PASS": {"color": "#00E676", "icon": "✅", "priority": 5},
    "FAIL": {"color": "#FF1744", "icon": "❌", "priority": 1},
    "REWORK": {"color": "#FF9100", "icon": "🔧", "priority": 2},
    "REVIEW_REQUIRED": {"color": "#FFC107", "icon": "⚠️", "priority": 3},
    "UNVERIFIED": {"color": "#78909C", "icon": "❓", "priority": 4},
    "PROCESSING": {"color": "#42A5F5", "icon": "⏳", "priority": 6},
}

# ─── Alert Configuration ─────────────────────────────────────────
ALERT_CONFIG = {
    "max_alerts_per_billet": 5,
    "dedup_window_seconds": 10,
    "severity_levels": ["CRITICAL", "WARNING", "INFO"],
    "alert_types": [
        "OUT_OF_TOLERANCE",
        "DEFECT_DETECTED",
        "ID_UNREADABLE",
        "DUPLICATE_ID",
        "CALIBRATION_INVALID",
        "LOW_CONFIDENCE",
        "MODEL_UNAVAILABLE",
        "DIMENSIONAL_DRIFT",
    ],
}

# ─── Dashboard Theme (Executive Enterprise Light) ────────────────
THEME = {
    "bg_primary": "#F8FAFC",
    "bg_secondary": "#FFFFFF",
    "bg_card": "#FFFFFF",
    "bg_elevated": "#F1F5F9",
    "accent_primary": "#1E40AF",
    "accent_success": "#16A34A",
    "accent_danger": "#DC2626",
    "accent_warning": "#D97706",
    "accent_info": "#0284C7",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "text_muted": "#94A3B8",
    "border": "#E2E8F0",
    "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
}

# ─── Chatbot ──────────────────────────────────────────────────────
CHATBOT_CONFIG = {
    "llm_provider": os.getenv("LLM_PROVIDER", "none"),
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
    "max_query_results": 100,
    "fallback_to_rules": True,
}


def load_config_yaml(path: str = None) -> dict:
    """Load configuration from YAML file if it exists."""
    if path is None:
        path = str(BASE_DIR / "config.yaml")
    if yaml and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_active_profile(profile_name: str) -> dict:
    """Get billet profile by name."""
    profiles = DEFAULT_BILLET_PROFILES.copy()
    yaml_config = load_config_yaml()
    if "billet_profiles" in yaml_config:
        profiles.update(yaml_config["billet_profiles"])
    return profiles.get(profile_name, profiles.get("Custom Profile", {}))


def get_all_profiles() -> dict:
    """Get all available billet profiles."""
    profiles = DEFAULT_BILLET_PROFILES.copy()
    yaml_config = load_config_yaml()
    if "billet_profiles" in yaml_config:
        profiles.update(yaml_config["billet_profiles"])
    return profiles


get_profile = get_active_profile


# ─── System & Testing Settings ────────────────────────────────────
SYSTEM_SETTINGS_FILE = DATA_DIR / "system_settings.json"


def is_qc_suspended() -> bool:
    """Check if QC requirements are suspended for testing mode."""
    # 1. Streamlit session state takes highest precedence for immediate UI reactivity
    try:
        import streamlit as st
        if "suspend_qc_requirements" in st.session_state:
            return bool(st.session_state["suspend_qc_requirements"])
    except Exception:
        pass

    # 2. Check persistent system_settings.json
    if SYSTEM_SETTINGS_FILE.exists():
        try:
            import json
            with open(SYSTEM_SETTINGS_FILE, "r") as f:
                data = json.load(f)
                if "suspend_qc_requirements" in data:
                    return bool(data["suspend_qc_requirements"])
        except Exception:
            pass

    # 3. Check config.yaml
    yaml_config = load_config_yaml()
    if "quality_control" in yaml_config and "suspend_qc_requirements" in yaml_config["quality_control"]:
        return bool(yaml_config["quality_control"]["suspend_qc_requirements"])

    # 4. Environment variable fallback
    return os.getenv("SUSPEND_QC_REQUIREMENTS", "false").lower() in ("true", "1", "yes")


def set_qc_suspended(suspended: bool) -> None:
    """Enable or disable QC requirements suspension (Testing Mode)."""
    # 1. Update session state
    try:
        import streamlit as st
        st.session_state["suspend_qc_requirements"] = bool(suspended)
    except Exception:
        pass

    # 2. Persist to system_settings.json
    try:
        import json
        settings = {}
        if SYSTEM_SETTINGS_FILE.exists():
            try:
                with open(SYSTEM_SETTINGS_FILE, "r") as f:
                    settings = json.load(f) or {}
            except Exception:
                settings = {}
        settings["suspend_qc_requirements"] = bool(suspended)
        SYSTEM_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SYSTEM_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

    # 3. Update config.yaml if PyYAML is available
    config_path = BASE_DIR / "config.yaml"
    if yaml:
        try:
            cfg = load_config_yaml()
            if "quality_control" not in cfg:
                cfg["quality_control"] = {}
            cfg["quality_control"]["suspend_qc_requirements"] = bool(suspended)
            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
        except Exception:
            pass

