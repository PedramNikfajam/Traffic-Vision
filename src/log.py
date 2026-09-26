"""
traffic_ai logging and reporting utilities.

Provides consistent logging across all pipeline stages,
and standardized JSON report generation.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logger(
    name: str = "traffic_ai",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Create a consistently formatted logger.
    
    Args:
        name: Logger name (usually the script/module name).
        level: Logging level.
        log_file: Optional file path to also write logs to.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


# ============================================================
# STATUS LOGGING HELPERS
# ============================================================

def log_pass(logger: logging.Logger, msg: str) -> None:
    """Log a PASS status check."""
    logger.info("PASS  | %s", msg)


def log_fail(logger: logging.Logger, msg: str) -> None:
    """Log a FAIL status check."""
    logger.error("FAIL  | %s", msg)


def log_warn(logger: logging.Logger, msg: str) -> None:
    """Log a WARNING status check."""
    logger.warning("WARN  | %s", msg)


def log_section(logger: logging.Logger, title: str) -> None:
    """Log a section header."""
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


# ============================================================
# REPORT GENERATION
# ============================================================

class Report:
    """Standardized JSON report builder.
    
    All pipeline stages use this to produce consistent reports.
    Reports include metadata (timestamp, stage, version) automatically.
    
    Usage:
        report = Report(stage="02_dataset_preparation")
        report.add("source_paths", {...})
        report.add("statistics", {...})
        report.add_check("split_leakage", passed=True, detail="0 overlaps")
        report.save(output_dir / "phase2_report.json")
    """
    
    def __init__(self, stage: str, description: str = ""):
        self.data: Dict[str, Any] = {
            "_meta": {
                "stage": stage,
                "description": description,
                "generated_at": datetime.now().isoformat(),
                "version": "0.2.0",
            },
        }
        self._checks: List[Dict[str, Any]] = []
    
    def add(self, key: str, value: Any) -> None:
        """Add a top-level section to the report."""
        self.data[key] = value
    
    def add_check(
        self, 
        name: str, 
        passed: bool, 
        detail: str = "",
        severity: str = "error",
    ) -> None:
        """Add an integrity check result.
        
        Args:
            name: Check name (e.g., 'split_leakage', 'class_mapping').
            passed: Whether the check passed.
            detail: Human-readable detail.
            severity: 'error' (must pass) or 'warning' (informational).
        """
        self._checks.append({
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "severity": severity,
            "detail": detail,
        })
    
    @property
    def all_checks_passed(self) -> bool:
        """True if all error-severity checks passed."""
        return all(
            c["status"] == "PASS" 
            for c in self._checks 
            if c["severity"] == "error"
        )
    
    @property
    def has_warnings(self) -> bool:
        """True if any warning-severity checks failed."""
        return any(
            c["status"] == "FAIL" 
            for c in self._checks 
            if c["severity"] == "warning"
        )
    
    def save(self, path: Path) -> Path:
        """Save report to JSON file. Returns the path."""
        self.data["integrity_checks"] = self._checks
        self.data["_meta"]["all_checks_passed"] = self.all_checks_passed
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str, ensure_ascii=False)
        return path
    
    def print_summary(self, logger: logging.Logger) -> None:
        """Print a summary of all checks to the logger."""
        if not self._checks:
            logger.info("No integrity checks recorded.")
            return
        
        log_section(logger, "INTEGRITY CHECK SUMMARY")
        for check in self._checks:
            status = check["status"]
            name = check["name"]
            detail = check["detail"]
            if status == "PASS":
                log_pass(logger, f"{name}: {detail}")
            else:
                if check["severity"] == "error":
                    log_fail(logger, f"{name}: {detail}")
                else:
                    log_warn(logger, f"{name}: {detail}")
        
        if self.all_checks_passed:
            logger.info("Overall: ALL CHECKS PASSED")
        else:
            logger.error("Overall: SOME CHECKS FAILED - review errors above")


# ============================================================
# ENVIRONMENT REPORTING
# ============================================================

def collect_environment() -> Dict[str, Any]:
    """Collect environment information for reproducibility."""
    env = {
        "python_version": sys.version,
        "platform": sys.platform,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        import torch
        env["pytorch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            env["gpu_devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                env["gpu_devices"].append({
                    "index": i,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / 1024**3, 1),
                })
    except ImportError:
        env["pytorch_version"] = "not installed"
    
    try:
        import ultralytics
        env["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        env["ultralytics_version"] = "not installed"
    
    try:
        import cv2
        env["opencv_version"] = cv2.__version__
    except ImportError:
        env["opencv_version"] = "not installed"
    
    try:
        import numpy
        env["numpy_version"] = numpy.__version__
    except ImportError:
        pass
    
    return env
