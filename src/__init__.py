"""
traffic_ai - BDD100K Traffic Computer Vision Research Package

Research-grade pipeline for vehicle detection, tracking, counting,
and traffic flow analysis with day/night robustness evaluation.

Architecture:
    src/config.py           - Central configuration (single source of truth)
    src/log.py              - Logging and standardized reporting
    src/viz.py              - Schema-agnostic training-curve plotting
    src/dataset/convert.py  - BDD100K->YOLO conversion (pure, testable)

Scripts (run in order on Kaggle):
    scripts/01_dataset_inspection.py    - Verify dataset structure/schema
    scripts/02_prepare_vehicle_dataset.py - Convert to YOLO format
    scripts/03_validate_dataset.py      - Validate prepared dataset
    scripts/04_train.py                 - Train detector
    scripts/05_evaluate.py              - Evaluate detector
"""

__version__ = "0.2.0"
__author__ = "Transportation AI Research"
