"""
Dataset module for BDD100K traffic analysis.

Provides pure, testable conversion logic and dataset helpers.
"""

from .convert import (
    AnnotationAccounting,
    BBox,
    ConvertedFrame,
    ConvertedLabel,
    convert_frame,
    fingerprint_config,
    fingerprint_file,
    parse_bdd100k_box2d,
)

__all__ = [
    "AnnotationAccounting",
    "BBox",
    "ConvertedFrame",
    "ConvertedLabel",
    "convert_frame",
    "fingerprint_config",
    "fingerprint_file",
    "parse_bdd100k_box2d",
]
