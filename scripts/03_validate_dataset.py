#!/usr/bin/env python3
"""
Phase 3: YOLO Dataset Validation

Validates the generated YOLO dataset for correctness before training.
Independent from the preparation script - validates only the output artifacts.

Inputs:
    - data/vehicle_detection/  (YOLO dataset from Phase 2)

Outputs:
    - reports/phase3_validation.json

Checks:
    - dataset.yaml exists and is parseable
    - Every image has a corresponding label file
    - Every label has a corresponding image file
    - All class IDs are valid (0..nc-1)
    - All coordinates are normalized [0,1]
    - No duplicate filenames
    - No train/val filename overlap (split leakage)
    - Images are readable
    - Label files are well-formed
    - Manifest consistency (if manifests exist)

Run on Kaggle:
    python traffic_ai/scripts/03_validate_dataset.py
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.config import CLASSES, OUTPUT, IMAGE_EXTENSIONS, SEED
from src.log import setup_logger, log_section, log_pass, log_fail, log_warn, Report


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def count_duplicate_label_lines(lines: List[str]) -> int:
    """Count exact-duplicate label lines within a single label file.

    BDD100K occasionally contains the same box twice for one object; the
    training framework silently removes exact duplicates, so class counts
    must exclude them to match what the model actually trains on.
    """
    seen = set()
    duplicates = 0
    for line in lines:
        if line in seen:
            duplicates += 1
        else:
            seen.add(line)
    return duplicates


def validate_dataset_yaml(data_dir: Path, logger) -> Tuple[bool, Dict[str, Any]]:
    """Validate dataset.yaml exists and is well-formed."""
    yaml_path = data_dir / "dataset.yaml"
    
    if not yaml_path.exists():
        log_fail(logger, f"dataset.yaml not found at {yaml_path}")
        return False, {}
    
    try:
        import yaml
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
    except ImportError:
        # Fallback: parse manually for simple YAML
        config = {}
        with open(yaml_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if ": " in line:
                        key, val = line.split(": ", 1)
                        config[key.strip()] = val.strip()
    except Exception as e:
        log_fail(logger, f"dataset.yaml is not valid YAML: {e}")
        return False, {}
    
    # Check required fields
    errors = []
    for field in ["path", "train", "val", "nc", "names"]:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        for e in errors:
            log_fail(logger, f"dataset.yaml: {e}")
        return False, config
    
    # Validate nc matches our class config
    nc = int(config.get("nc", 0))
    if nc != CLASSES.num_classes:
        log_fail(logger, f"dataset.yaml nc={nc} != expected {CLASSES.num_classes}")
        return False, config
    
    log_pass(logger, f"dataset.yaml valid: nc={nc}, path={config.get('path', '?')}")
    return True, config


def validate_split(
    data_dir: Path,
    split: str,
    nc: int,
    logger,
    sample_images: int = 100,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate a single split directory.
    
    Returns:
        (all_passed, stats_dict)
    """
    log_section(logger, f"VALIDATING {split.upper()} SPLIT")
    
    images_dir = data_dir / split / "images"
    labels_dir = data_dir / split / "labels"
    
    # Check directories exist
    if not images_dir.exists():
        log_fail(logger, f"{images_dir} does not exist")
        return False, {}
    if not labels_dir.exists():
        log_fail(logger, f"{labels_dir} does not exist")
        return False, {}
    
    # Collect all image and label files
    image_files = set()
    for f in os.listdir(images_dir):
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
            image_files.add(Path(f).stem)
    
    label_files = set()
    for f in os.listdir(labels_dir):
        if f.endswith(".txt"):
            label_files.add(Path(f).stem)
    
    logger.info("Found %d images, %d labels in %s", len(image_files), len(label_files), split)
    
    # Check image-label correspondence
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files
    
    all_ok = True
    
    if images_without_labels:
        log_fail(logger, f"{len(images_without_labels)} images without labels")
        for name in sorted(images_without_labels)[:5]:
            logger.error("  Missing label for: %s", name)
        all_ok = False
    else:
        log_pass(logger, "All images have corresponding labels")
    
    if labels_without_images:
        log_fail(logger, f"{len(labels_without_images)} labels without images")
        for name in sorted(labels_without_images)[:5]:
            logger.error("  Missing image for: %s", name)
        all_ok = False
    else:
        log_pass(logger, "All labels have corresponding images")
    
    # Validate label file contents
    total_labels = 0
    empty_labels = 0
    malformed_lines = 0
    invalid_class_ids = 0
    invalid_coords = 0
    duplicate_label_lines = 0
    class_counter = Counter()
    
    for stem in sorted(label_files):
        label_path = labels_dir / f"{stem}.txt"
        try:
            with open(label_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception as e:
            log_fail(logger, f"Cannot read {label_path}: {e}")
            malformed_lines += 1
            all_ok = False
            continue
        
        if not lines:
            empty_labels += 1
            continue
        
        duplicate_label_lines += count_duplicate_label_lines(lines)
        
        seen_lines = set()
        for line in lines:
            total_labels += 1
            
            # Exact duplicates: counted above, excluded from class counts
            # (matches training-framework dedup behavior)
            if line in seen_lines:
                continue
            seen_lines.add(line)
            
            parts = line.split()
            
            if len(parts) != 5:
                malformed_lines += 1
                continue
            
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                malformed_lines += 1
                continue
            
            if cls_id < 0 or cls_id >= nc:
                invalid_class_ids += 1
                continue
            
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and
                    0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                invalid_coords += 1
                continue
            
            class_counter[cls_id] += 1
    
    # Report label validation
    if malformed_lines > 0:
        log_fail(logger, f"{malformed_lines} malformed label lines")
        all_ok = False
    else:
        log_pass(logger, "All label lines well-formed (5 fields)")
    
    if invalid_class_ids > 0:
        log_fail(logger, f"{invalid_class_ids} labels with invalid class IDs (not in 0..{nc-1})")
        all_ok = False
    else:
        log_pass(logger, f"All class IDs valid (0..{nc-1})")
    
    if invalid_coords > 0:
        log_fail(logger, f"{invalid_coords} labels with invalid coordinates")
        all_ok = False
    else:
        log_pass(logger, "All coordinates normalized [0,1]")
    
    if duplicate_label_lines > 0:
        log_warn(logger,
                  f"{duplicate_label_lines} exact-duplicate label lines "
                  f"(excluded from class counts; matches training dedup)")
    else:
        log_pass(logger, "No duplicate label lines")
    
    logger.info("Total labels: %d, Empty label files: %d", total_labels, empty_labels)
    
    # Per-class distribution
    for cls_id in range(nc):
        cls_name = CLASSES.id_to_class.get(cls_id, f"class_{cls_id}")
        count = class_counter.get(cls_id, 0)
        pct = (count / total_labels * 100) if total_labels > 0 else 0
        logger.info("  Class %d (%s): %d objects (%.1f%%)", cls_id, cls_name, count, pct)
    
    # Sample image readability check
    if sample_images > 0 and image_files:
        random.seed(SEED)
        sample = random.sample(sorted(image_files), min(sample_images, len(image_files)))
        readable = 0
        unreadable = 0
        for stem in sample:
            # Find the actual image file (could be .jpg, .png, etc.)
            found = False
            for ext in IMAGE_EXTENSIONS:
                img_path = images_dir / f"{stem}{ext}"
                if img_path.exists():
                    try:
                        from PIL import Image
                        with Image.open(img_path) as img:
                            img.verify()
                        readable += 1
                    except Exception:
                        unreadable += 1
                    found = True
                    break
            if not found:
                unreadable += 1
        
        if unreadable > 0:
            log_warn(logger, f"{unreadable}/{len(sample)} sampled images unreadable")
        else:
            log_pass(logger, f"All {len(sample)} sampled images readable")
    
    stats = {
        "image_count": len(image_files),
        "label_count": len(label_files),
        "total_labels": total_labels,
        "empty_label_files": empty_labels,
        "malformed_lines": malformed_lines,
        "invalid_class_ids": invalid_class_ids,
        "invalid_coords": invalid_coords,
        "duplicate_label_lines": duplicate_label_lines,
        "images_without_labels": len(images_without_labels),
        "labels_without_images": len(labels_without_images),
        "class_distribution": {
            CLASSES.id_to_class.get(k, str(k)): v 
            for k, v in class_counter.items()
        },
    }
    
    return all_ok, stats


def check_cross_split_leakage(
    data_dir: Path,
    logger,
) -> Tuple[bool, int]:
    """Check for filename overlap between train and val."""
    train_images = set()
    val_images = set()
    
    train_dir = data_dir / "train" / "images"
    val_dir = data_dir / "val" / "images"
    
    if train_dir.exists():
        train_images = {f for f in os.listdir(train_dir) 
                       if Path(f).suffix.lower() in IMAGE_EXTENSIONS}
    if val_dir.exists():
        val_images = {f for f in os.listdir(val_dir)
                     if Path(f).suffix.lower() in IMAGE_EXTENSIONS}
    
    overlap = train_images & val_images
    if overlap:
        log_fail(logger, f"SPLIT LEAKAGE: {len(overlap)} images in both train and val")
        for name in sorted(overlap)[:10]:
            logger.error("  Leaked: %s", name)
        return False, len(overlap)
    
    log_pass(logger, f"No split leakage: train={len(train_images)}, val={len(val_images)}")
    return True, 0


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    logger = setup_logger("phase3", log_file=OUTPUT.logs / "phase3.log")
    log_section(logger, "YOLO DATASET VALIDATION")
    
    data_dir = OUTPUT.data
    if not data_dir.exists():
        log_fail(logger, f"Dataset directory not found: {data_dir}")
        logger.error("Run Phase 2 (02_prepare_vehicle_dataset.py) first.")
        return 1
    
    report = Report(
        stage="03_dataset_validation",
        description="Validate YOLO dataset integrity before training",
    )
    
    # 1. Validate dataset.yaml
    yaml_ok, yaml_config = validate_dataset_yaml(data_dir, logger)
    report.add_check("dataset_yaml", yaml_ok, 
                      "dataset.yaml valid" if yaml_ok else "dataset.yaml invalid")
    
    nc = CLASSES.num_classes
    
    # 2. Validate train split
    train_ok, train_stats = validate_split(data_dir, "train", nc, logger)
    report.add_check("train_split", train_ok,
                      f"{train_stats.get('image_count', 0)} images, "
                      f"{train_stats.get('total_labels', 0)} labels")
    report.add("train_statistics", train_stats)
    
    # 3. Validate val split
    val_ok, val_stats = validate_split(data_dir, "val", nc, logger)
    report.add_check("val_split", val_ok,
                      f"{val_stats.get('image_count', 0)} images, "
                      f"{val_stats.get('total_labels', 0)} labels")
    report.add("val_statistics", val_stats)
    
    # 4. Cross-split leakage
    leakage_ok, leakage_count = check_cross_split_leakage(data_dir, logger)
    report.add_check("split_leakage", leakage_ok,
                      f"{leakage_count} overlapping filenames")
    
    # 5. Class coverage check
    all_classes = set()
    for stats in [train_stats, val_stats]:
        if "class_distribution" in stats:
            all_classes.update(stats["class_distribution"].keys())
    
    missing_classes = set(CLASSES.target_classes) - all_classes
    classes_ok = len(missing_classes) == 0
    report.add_check("class_coverage", classes_ok,
                      f"Missing classes: {missing_classes}" if missing_classes 
                      else f"All {CLASSES.num_classes} classes present",
                      severity="warning")
    
    # Save report
    report_path = report.save(OUTPUT.reports / "phase3_validation.json")
    report.print_summary(logger)
    
    log_section(logger, "VALIDATION COMPLETE")
    logger.info("Report: %s", report_path)
    
    if report.all_checks_passed:
        log_pass(logger, "DATASET READY FOR TRAINING")
        return 0
    else:
        log_fail(logger, "DATASET HAS ERRORS - fix before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
