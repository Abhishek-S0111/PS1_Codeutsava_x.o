"""Logging configuration for Steel Billet Inspection System."""
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "steel_inspection", log_dir: str = None) -> logging.Logger:
    """Configure and return application logger."""
    if log_dir is None:
        from utils.config import LOGS_DIR
        log_dir = str(LOGS_DIR)
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path(log_dir) / f"inspection_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()
