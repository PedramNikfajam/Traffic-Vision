#!/usr/bin/env python3
"""
Phase 1: BDD100K Dataset Inspection

Automatically scans /kaggle/input/ for BDD100K dataset components,
identifies image directories, annotation files, and metadata schemas
by INSPECTING FILE CONTENTS (not filenames), and produces structured reports.

Run on Kaggle:
    python scripts/01_dataset_inspection.py

Outputs:
    /kaggle/working/traffic_ai/reports/dataset_report.json
    /kaggle/working/traffic_ai/reports/dataset_report.txt
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class AnnotationSource:
    """Represents an identified annotation file with its classification."""
    path: str
    relative_path: str
    size_bytes: int
    classification: str  # "detection_bdd100k", "detection_coco", "tracking", "segmentation", "metadata", "unknown"
    schema: Dict[str, Any]
    confidence: float
    reason: str
    sample_annotations: List[Dict] = field(default_factory=list)


@dataclass
class ImageDirectory:
    """Represents a classified image directory."""
    name: str
    path: str
    relative_path: str
    purpose: str  # "100k_train", "100k_val", "100k_test", "10k_train", "10k_val", "trainA", "trainB", "seg_images", "seg_color_labels", "unknown"
    image_count: int
    labeled_count: int = 0
    unlabeled_count: int = 0
    sample_images: List[str] = field(default_factory=list)
    nested_partitions: Dict[str, int] = field(default_factory=dict)


def detect_environment() -> Dict[str, Any]:
    """Detect GPU, framework versions, and available libraries.

    All imports are optional - missing libraries are reported, not fatal.
    """
    env_info = {
        "pytorch_version": "not installed",
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpus": [],
        "tensorflow_version": "not installed",
        "tf_gpu_devices": [],
    }

    try:
        import torch
        env_info["pytorch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        env_info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                env_info["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 1),
                })
    except ImportError:
        pass

    try:
        import tensorflow as tf
        env_info["tensorflow_version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            env_info["tf_gpu_devices"] = [gpu.name for gpu in gpus]
    except ImportError:
        pass

    return env_info


def find_kaggle_input(custom_path: Optional[str] = None) -> Optional[Path]:
    """Find the /kaggle/input/ directory structure."""
    if custom_path:
        path = Path(custom_path)
        if path.exists() and path.is_dir():
            return path

    candidates = [
        Path("/kaggle/input"),
        Path("./kaggle/input"),
        Path("kaggle/input"),
    ]

    for path in candidates:
        if path.exists() and path.is_dir():
            return path

    return None


# --- Efficient JSON inspection helpers ---

def read_json_head(filepath: Path, max_chars: int = 16384) -> str:
    """Read the first N characters of a JSON file as raw text."""
    try:
        with open(filepath, "r") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def extract_top_level_keys(text: str) -> Set[str]:
    """Extract top-level JSON object keys from raw text without full parsing."""
    keys = set()
    if not text or not text.strip().startswith("{"):
        return keys

    # Simple state machine to find top-level keys in a JSON object
    in_string = False
    escape_next = False
    depth = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == "\\" and in_string:
            escape_next = True
            i += 1
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            i += 1
            continue

        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            elif ch == '"' and depth == 1:
                # Start of a top-level key
                i += 1
                key_start = i
                while i < n and text[i] != '"':
                    if text[i] == "\\" and i + 1 < n:
                        i += 2
                        continue
                    i += 1
                if i < n:
                    keys.add(text[key_start:i])
            elif ch == "[" and depth == 1:
                # Array value at top level - we can't easily extract keys, stop
                pass
        i += 1

    return keys


def detect_root_type(text: str) -> str:
    """Detect if JSON root is object, array, or other from raw text."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return "object"
    if stripped.startswith("["):
        return "array"
    return "unknown"


def extract_first_frame_info(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract first frame's keys and first label's keys from raw JSON text.
    Only works for object-root with "frames" array.
    """
    if not text.strip().startswith("{"):
        return None

    # Find "frames" key and its array start
    frames_idx = text.find('"frames"')
    if frames_idx == -1:
        return None

    # Find the opening bracket of frames array
    arr_start = text.find("[", frames_idx)
    if arr_start == -1:
        return None

    # Find the first object in the array
    obj_start = text.find("{", arr_start)
    if obj_start == -1:
        return None

    # Extract first frame object (naive but works for well-formed JSON)
    # We'll parse just enough to get keys
    frame_text = text[obj_start:]
    frame_keys = set()
    label_keys = set()

    in_string = False
    escape_next = False
    depth = 0
    i = 0
    n = len(frame_text)
    current_key = ""

    while i < n:
        ch = frame_text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == "\\" and in_string:
            escape_next = True
            i += 1
            continue

        if ch == '"' and not escape_next:
            if not in_string:
                # Start of key
                in_string = True
                key_start = i + 1
            else:
                # End of key or string value
                in_string = False
                if depth == 1 and not current_key:
                    # This is a key at frame level
                    frame_keys.add(frame_text[key_start:i])
                    current_key = frame_text[key_start:i]
                elif depth == 2 and current_key == "labels":
                    # Inside labels array, capturing first label's keys
                    label_keys.add(frame_text[key_start:i])
            i += 1
            continue

        if in_string:
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        elif ch == "[" and depth == 1 and current_key == "labels":
            # Entering labels array - next object will be first label
            pass

        i += 1

    return {
        "frame_keys": frame_keys,
        "label_keys": label_keys,
    }


def extract_first_coco_info(text: str) -> Optional[Dict[str, Any]]:
    """Extract basic COCO format info from raw text."""
    if not text.strip().startswith("{"):
        return None
    keys = extract_top_level_keys(text)
    return {"top_keys": keys} if keys else None


def extract_list_root_info(text: str) -> Optional[Dict[str, Any]]:
    """Extract info from array-root JSON (BDD100K list format)."""
    if not text.strip().startswith("["):
        return None
    # Can't easily inspect without parsing - return candidate flag
    return {"is_list_root": True}


def classify_annotation_file_lazy(filepath: Path) -> Tuple[str, Dict[str, Any], float, str]:
    """
    Classify annotation file by content with minimal loading.
    Uses raw text inspection - NO json.loads() on truncated data.
    Returns: (classification, schema, confidence, reason)
    """
    text = read_json_head(filepath)
    if not text:
        return "unknown", {}, 0.0, "Empty or unreadable file"

    root_type = detect_root_type(text)

    if root_type == "object":
        keys = extract_top_level_keys(text)

        # COCO detection: has "annotations" + "images" + "categories"
        if {"annotations", "images", "categories"}.issubset(keys):
            return "detection_coco", {"format": "coco", "top_keys": sorted(keys),
                "has_box2d": False,
                "has_bbox": True,
                "has_category": True,
                "has_category_id": True,
                "has_image_id": True,
                "has_object_attributes": False,
                "has_frame_attributes": False,
            }, 0.95, "COCO format: annotations+images+categories"

        # BDD100K frames format: has "frames" array
        if "frames" in keys:
            frame_info = extract_first_frame_info(text)
            if frame_info:
                frame_keys = frame_info["frame_keys"]
                label_keys = frame_info["label_keys"]

                if label_keys:
                    # Tracking: track_id present
                    if "track_id" in label_keys or "tracking_id" in label_keys:
                        return "tracking", {"format": "bdd100k_tracking", "top_keys": sorted(keys)}, 0.95, "BDD100K tracking: track_id in labels"

                    # Segmentation: rle/segmentation/mask in labels
                    if any(k in label_keys for k in ["rle", "segmentation", "mask", "poly2d"]):
                        return "segmentation", {"format": "bdd100k_segmentation", "top_keys": sorted(keys)}, 0.95, "BDD100K segmentation: mask/rle in labels"

                    # Detection: box2d + category in labels
                    if "box2d" in label_keys and "category" in label_keys:
                        has_frame_meta = any(k in frame_keys for k in ["timeofday", "weather", "scene"])
                        # Determine box2d format from label keys
                        box2d_keys = [k for k in label_keys if k.startswith("box2d")] or ["box2d"]
                        if "x1" in label_keys or "x2" in label_keys:
                            box2d_format = "xyxy"
                        elif "x" in label_keys or "w" in label_keys:
                            box2d_format = "xywh"
                        else:
                            box2d_format = "unknown"
                        has_object_attrs = "attributes" in label_keys
                        obj_attr_keys = []
                        if has_object_attrs:
                            # Can't easily determine from text alone
                            obj_attr_keys = ["attributes"]
                        return "detection_bdd100k", {
                            "format": "bdd100k_frames",
                            "has_frame_metadata": has_frame_meta,
                            "has_box2d": True,
                            "box2d_keys": box2d_keys,
                            "box2d_format": box2d_format,
                            "has_category": True,
                            "has_category_id": "category_id" in label_keys,
                            "has_object_attributes": has_object_attrs,
                            "object_attribute_keys": obj_attr_keys,
                            "has_frame_attributes": has_frame_meta,
                            "frame_attribute_keys": [k for k in frame_keys if k in ("timeofday", "weather", "scene")],
                            "frame_keys": sorted(frame_keys),
                            "label_keys": sorted(label_keys),
                            "top_keys": sorted(keys)
                        }, 0.95 if has_frame_meta else 0.85, "BDD100K detection: box2d+category in labels"

                # Empty frames - check if it's just metadata
                if any(k in frame_keys for k in ["timeofday", "weather", "scene"]):
                    return "metadata", {"format": "bdd100k_metadata", "top_keys": sorted(keys)}, 0.90, "BDD100K metadata: frame attributes without labels"

            # Frames key exists but couldn't parse frames - still likely detection
            return "detection_bdd100k", {"format": "bdd100k_frames", "top_keys": sorted(keys)}, 0.70, "BDD100K frames format detected (frames key present)"

        # Segmentation color-label index files
        if any(k in keys for k in ["color_labels", "segmentation", "masks"]):
            return "segmentation", {"format": "segmentation_index", "top_keys": sorted(keys)}, 0.90, "Segmentation index: color_labels/segmentation keys"

        return "unknown", {"format": "unknown_object", "top_keys": sorted(keys)}, 0.2, f"Unknown object format: top_keys={sorted(keys)}"

    elif root_type == "array":
        # BDD100K list-of-frames format - can't inspect without more parsing
        return "candidate_bdd100k_list", {"format": "list", "top_keys": ["array_root"]}, 0.60, "JSON root is array - possible BDD100K list format"

    return "unknown", {"format": "unknown", "top_keys": []}, 0.1, "Unknown root type"


def deep_classify_bdd100k_list(filepath: Path) -> Tuple[str, Dict[str, Any], float, str]:
    """Fully load and classify BDD100K list-of-frames format (only for candidates)."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception as e:
        return "unknown", {}, 0.0, f"Failed to load: {e}"

    if not isinstance(data, list) or not data:
        return "unknown", {}, 0.0, "Not a list format"

    first_frame = data[0]
    if not isinstance(first_frame, dict):
        return "unknown", {}, 0.0, "List elements not dict"

    frame_keys = set(first_frame.keys())
    labels = first_frame.get("labels", [])

    if not labels:
        if any(k in frame_keys for k in ["timeofday", "weather", "scene"]):
            return "metadata", {"format": "bdd100k_metadata_list"}, 0.90, "BDD100K metadata list"
        return "unknown", {}, 0.3, "Frames without labels"

    first_label = labels[0]
    label_keys = set(first_label.keys())

    # Tracking
    if "track_id" in label_keys or "tracking_id" in label_keys:
        return "tracking", {"format": "bdd100k_tracking_list"}, 0.95, "BDD100K tracking list: track_id"

    # Segmentation
    if any(k in label_keys for k in ["rle", "segmentation", "mask", "poly2d"]):
        return "segmentation", {"format": "bdd100k_segmentation_list"}, 0.95, "BDD100K segmentation list: mask/rle"

    # Detection
    if "box2d" in label_keys and "category" in label_keys:
        has_frame_meta = any(k in frame_keys for k in ["timeofday", "weather", "scene"])
        box2d_keys = [k for k in label_keys if k.startswith("box2d")] or ["box2d"]
        if "x1" in label_keys or "x2" in label_keys:
            box2d_format = "xyxy"
        elif "x" in label_keys or "w" in label_keys:
            box2d_format = "xywh"
        else:
            box2d_format = "unknown"
        has_object_attrs = "attributes" in label_keys
        obj_attr_keys = []
        if has_object_attrs:
            obj_attr_keys = ["attributes"]
        return "detection_bdd100k", {
            "format": "bdd100k_frames_list",
            "has_frame_metadata": has_frame_meta,
            "has_box2d": True,
            "box2d_keys": box2d_keys,
            "box2d_format": box2d_format,
            "has_category": True,
            "has_category_id": "category_id" in label_keys,
            "has_object_attributes": has_object_attrs,
            "object_attribute_keys": obj_attr_keys,
            "has_frame_attributes": has_frame_meta,
            "frame_attribute_keys": [k for k in frame_keys if k in ("timeofday", "weather", "scene")],
        }, 0.95 if has_frame_meta else 0.85, "BDD100K detection list"

    return "unknown", {}, 0.3, "Unrecognized list format"


def get_sample_annotations_lazy(filepath: Path, classification: str, max_samples: int = 5) -> List[Dict]:
    """
    Get sample annotations WITHOUT fully loading large files.
    Uses streaming text parsing to extract first few annotations.
    """
    try:
        # Read a larger chunk but still bounded
        text = read_json_head(filepath, max_chars=65536)
        if not text:
            return []

        sample_anns = []

        if classification == "detection_coco" and text.strip().startswith("{"):
            # For COCO, find annotations array and extract first few objects
            ann_idx = text.find('"annotations"')
            if ann_idx != -1:
                arr_start = text.find("[", ann_idx)
                if arr_start != -1:
                    # Extract first few annotation objects from text
                    # This is a best-effort text extraction
                    obj_text = text[arr_start:arr_start + 10000]
                    # Parse what we can from the array
                    sample_anns = extract_json_objects_from_array(obj_text, max_samples)

        elif classification in ("detection_bdd100k", "tracking", "segmentation", "metadata") and text.strip().startswith("{"):
            # BDD100K frames format - extract first frame's labels
            frame_info = extract_first_frame_info(text)
            if frame_info and frame_info["label_keys"]:
                # We can't easily reconstruct full objects from text alone
                # Return minimal schema info for reporting
                sample_anns = [{"_sample": True, "label_keys": list(frame_info["label_keys"])}]

        elif classification == "candidate_bdd100k_list" and text.strip().startswith("["):
            # List root - can't easily sample without parsing
            pass

        return sample_anns[:max_samples]
    except Exception:
        return []


def extract_json_objects_from_array(array_text: str, max_objects: int) -> List[Dict]:
    """Extract first N complete JSON objects from an array text."""
    objects = []
    if not array_text or not array_text.strip().startswith("["):
        return objects

    in_string = False
    escape_next = False
    depth = 0
    obj_start = -1
    i = 0
    n = len(array_text)

    while i < n and len(objects) < max_objects:
        ch = array_text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == "\\" and in_string:
            escape_next = True
            i += 1
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            i += 1
            continue

        if not in_string:
            if ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    obj_text = array_text[obj_start:i+1]
                    try:
                        obj = json.loads(obj_text)
                        objects.append(obj)
                    except Exception:
                        pass
                    obj_start = -1
        i += 1

    return objects


def classify_annotation_file(filepath: Path) -> Tuple[str, Dict[str, Any], float, str, List[Dict]]:
    """
    Full classification with sample annotations for reporting.
    Uses lazy classification first, only deep-loads candidate list formats.
    Returns: (classification, schema, confidence, reason, sample_annotations)
    """
    # First try lazy classification (only reads first 16KB as text)
    classification, schema, confidence, reason = classify_annotation_file_lazy(filepath)

    # If candidate list format, do deep classification (full load)
    if classification == "candidate_bdd100k_list":
        classification, schema, confidence, reason = deep_classify_bdd100k_list(filepath)

    # Get sample annotations - only for files we might report on
    # For unknown/other types, skip sampling to save memory
    sample_anns = []
    if classification in ("detection_bdd100k", "detection_coco", "tracking", "segmentation", "metadata"):
        sample_anns = get_sample_annotations_lazy(filepath, classification)

    return classification, schema, confidence, reason, sample_anns[:5]


# --- Image directory classification ---

def classify_image_directory(rel_path: str, dir_name: str, kaggle_input_dir: Path) -> str:
    """Classify image directory by its path structure and content."""
    rel_lower = rel_path.lower()
    name_lower = dir_name.lower()
    parts = [p for p in rel_lower.split("/") if p]

    # Segmentation directories (bdd100k_seg/.../seg/...)
    if "seg" in parts:
        if "color" in rel_lower or "color_label" in rel_lower:
            return "seg_color_labels"
        if "images" in rel_lower or parts[-1] in ("images", "img"):
            return "seg_images"
        if "labels" in rel_lower or parts[-1] in ("labels", "label"):
            return "seg_labels"
        return "seg_unknown"

    # TrainA/TrainB/TestA/TestB at any level (derived subsets) - check dir_name first
    if name_lower == "traina":
        return "trainA"
    if name_lower == "trainb":
        return "trainB"
    if name_lower == "testa":
        return "testA"
    if name_lower == "testb":
        return "testB"

    # BDD100K 100k structure - CANONICAL splits only (direct children of 100k)
    if "100k" in parts:
        idx = parts.index("100k")
        # Only consider immediate children of 100k as canonical splits
        if idx + 1 < len(parts):
            child = parts[idx + 1]
            # If this is a canonical split AND we're directly in it (not nested deeper)
            if child in ("train", "val", "test"):
                # Check if we're directly in 100k/train or 100k/train/xxx
                if len(parts) == idx + 2:
                    # Direct child: 100k/train, 100k/val, 100k/test
                    return f"100k_{child}"
                # Nested deeper: check if the current dir is a derived subset
                # (already handled above by dir_name check)
        return "100k_unknown"

    # BDD100K 10k structure - separate dataset
    if "10k" in parts:
        idx = parts.index("10k")
        if idx + 1 < len(parts):
            child = parts[idx + 1]
            if child == "train":
                return "10k_train"
            if child == "val":
                return "10k_val"
            if child == "test":
                return "10k_test"
        return "10k_unknown"

    # Other train/val/test at root
    if name_lower in ("train", "training"):
        return "train_root"
    if name_lower in ("val", "validation", "valid"):
        return "val_root"
    if name_lower in ("test",):
        return "test_root"

    return "unknown"


def get_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
    """Get image dimensions without loading full image into memory."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def _collect_all_images_recursive(root_path: Path, extensions: set) -> Tuple[List[str], int]:
    """Collect all image files recursively under a directory."""
    image_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if Path(fname).suffix.lower() in extensions:
                rel_path = Path(dirpath).relative_to(root_path) / fname
                image_files.append(str(rel_path))
    return image_files, len(image_files)


def _get_nested_partitions(root_path: Path, extensions: set) -> Dict[str, int]:
    """Get image counts for immediate subdirectories (partitions) and root direct files."""
    partitions = {}
    try:
        # Count images directly in root
        root_direct = 0
        for item in root_path.iterdir():
            if item.is_file() and item.suffix.lower() in extensions:
                root_direct += 1
        if root_direct > 0:
            partitions["root_direct"] = root_direct

        # Count images in immediate subdirectories
        for item in root_path.iterdir():
            if item.is_dir():
                _, count = _collect_all_images_recursive(item, extensions)
                if count > 0:
                    partitions[item.name] = count
    except Exception:
        pass
    return partitions


def scan_kaggle_input(kaggle_input_dir: Path) -> Tuple[List[ImageDirectory], List[AnnotationSource]]:
    """Recursively scan /kaggle/input/ to find BDD100K dataset components."""
    print(f"\n{'='*60}")
    print(f"SCANNING: {kaggle_input_dir}")
    print(f"{'='*60}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    image_dirs = []
    annotation_sources = []

    # First pass: collect all image directories and JSON file paths
    json_file_paths = []
    raw_image_dirs = []  # All directories with images (before canonical aggregation)

    for root, dirs, files in os.walk(kaggle_input_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(kaggle_input_dir)
        rel_str = str(rel_root)

        image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
        json_files_in_dir = [f for f in files if f.lower().endswith(".json")]

        if image_files:
            purpose = classify_image_directory(rel_str, root_path.name, kaggle_input_dir)
            raw_image_dirs.append(ImageDirectory(
                name=root_path.name,
                path=str(root_path),
                relative_path=rel_str,
                purpose=purpose,
                image_count=len(image_files),
                sample_images=image_files[:5],
            ))

        for fname in json_files_in_dir:
            fpath = root_path / fname
            json_file_paths.append((fpath, rel_str))

    # Post-process: Aggregate canonical 100k splits with recursive counting
    # Find 100k/train, 100k/val, 100k/test directories
    canonical_100k_dirs = {}
    for d in raw_image_dirs:
        if d.purpose in ("100k_train", "100k_val", "100k_test"):
            canonical_100k_dirs[d.purpose] = d

    # For each canonical split, recursively count all images and find partitions
    for purpose, canon_dir in canonical_100k_dirs.items():
        root_path = Path(canon_dir.path)
        all_images, total_count = _collect_all_images_recursive(root_path, image_extensions)
        partitions = _get_nested_partitions(root_path, image_extensions)

        # Update the canonical directory with recursive counts
        canon_dir.image_count = total_count
        canon_dir.sample_images = all_images[:5]
        # Store partition info as metadata
        canon_dir.nested_partitions = partitions

        image_dirs.append(canon_dir)

    # Add other non-100k directories (10k, seg, root-level train/val/test, etc.)
    for d in raw_image_dirs:
        if d.purpose not in ("100k_train", "100k_val", "100k_test"):
            image_dirs.append(d)

    print(f"  Found {len(image_dirs)} image directories (after canonical aggregation)")
    for d in image_dirs:
        part_info = f" | partitions: {d.nested_partitions}" if hasattr(d, 'nested_partitions') and d.nested_partitions else ""
        print(f"    [{d.purpose}] {d.relative_path}: {d.image_count} images{part_info}")

    print(f"  Found {len(json_file_paths)} JSON files")

    # Second pass: classify JSON files by content (lazy inspection only)
    print(f"\n  Classifying {len(json_file_paths)} annotation files by content (lazy)...")
    for fpath, rel_str in json_file_paths:
        classification, schema, confidence, reason, sample_anns = classify_annotation_file(fpath)
        annotation_sources.append(AnnotationSource(
            path=str(fpath),
            relative_path=rel_str + "/" + fpath.name,
            size_bytes=fpath.stat().st_size if fpath.exists() else 0,
            classification=classification,
            schema=schema,
            confidence=confidence,
            reason=reason,
            sample_annotations=sample_anns,
        ))

    return image_dirs, annotation_sources


# --- Annotation loading and processing ---

def load_detection_annotations(annotation_sources: List[AnnotationSource]) -> Tuple[List[Dict], List[Dict]]:
    """Load ALL annotations from confirmed detection sources only."""
    all_annotations = []
    frame_level_data = []

    detection_sources = [a for a in annotation_sources if a.classification in ("detection_bdd100k", "detection_coco")]

    print(f"\n{'='*60}")
    print(f"LOADING ALL ANNOTATIONS FROM {len(detection_sources)} CONFIRMED DETECTION FILES")
    print(f"{'='*60}")

    for source in detection_sources:
        print(f"  Loading: {source.relative_path} ({source.classification})")
        try:
            with open(source.path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                if "annotations" in data:
                    # COCO format - no frame-level metadata
                    for ann in data.get("annotations", []):
                        ann["_source_file"] = source.path
                        ann["_source_classification"] = source.classification
                        all_annotations.append(ann)

                elif "frames" in data:
                    # BDD100K frames format - metadata in frame["attributes"]
                    for frame in data.get("frames", []):
                        frame_name = frame.get("name", "")
                        # Extract frame-level attributes from frame["attributes"] per BDD100K schema
                        frame_attrs = frame.get("attributes", {})
                        for label in frame.get("labels", []):
                            ann = label.copy()
                            ann["_frame_name"] = frame_name
                            ann["_frame_attributes"] = frame_attrs
                            ann["_source_file"] = source.path
                            ann["_source_classification"] = source.classification
                            all_annotations.append(ann)
                        if frame_attrs:
                            frame_level_data.append({
                                "frame_name": frame_name,
                                "attributes": frame_attrs,
                                "num_objects": len(frame.get("labels", [])),
                                "source_file": source.path,
                            })

            elif isinstance(data, list):
                # BDD100K list-of-frames format - metadata in frame["attributes"]
                for frame in data:
                    frame_name = frame.get("name", "")
                    frame_attrs = frame.get("attributes", {})
                    for label in frame.get("labels", []):
                        ann = label.copy()
                        ann["_frame_name"] = frame_name
                        ann["_frame_attributes"] = frame_attrs
                        ann["_source_file"] = source.path
                        ann["_source_classification"] = source.classification
                        all_annotations.append(ann)
                    if frame_attrs:
                        frame_level_data.append({
                            "frame_name": frame_name,
                            "attributes": frame_attrs,
                            "num_objects": len(frame.get("labels", [])),
                            "source_file": source.path,
                        })

        except Exception as e:
            print(f"    Warning: Failed to load {source.path}: {e}")

    print(f"  Total annotations loaded: {len(all_annotations)}")
    print(f"  Frames with metadata: {len(frame_level_data)}")

    return all_annotations, frame_level_data


def extract_frame_metadata(frame_level_data: List[Dict]) -> Dict[str, Counter]:
    """Extract timeofday, weather, scene from FRAME-LEVEL attributes."""
    print(f"\n{'='*60}")
    print("EXTRACTING FRAME-LEVEL METADATA (timeofday/weather/scene)")
    print(f"{'='*60}")

    metadata_counts = {
        "timeofday": Counter(),
        "weather": Counter(),
        "scene": Counter(),
    }

    for frame_data in frame_level_data:
        attrs = frame_data.get("attributes", {})
        for key in ["timeofday", "weather", "scene"]:
            if key in attrs:
                metadata_counts[key][attrs[key]] += 1

    for key in ["timeofday", "weather", "scene"]:
        if metadata_counts[key]:
            print(f"\n  {key.capitalize()} distribution (frames):")
            for k, v in metadata_counts[key].most_common():
                print(f"    {k}: {v}")
        else:
            print(f"\n  {key.capitalize()}: not found in frame attributes")

    return metadata_counts


def _collect_image_names_recursive(root_path: Path, extensions: set) -> Set[str]:
    """Collect all image filenames recursively under a directory (basename only)."""
    image_names = set()
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if Path(fname).suffix.lower() in extensions:
                image_names.add(fname)
    return image_names


def build_image_annotation_index(
    all_annotations: List[Dict],
    image_dirs: List[ImageDirectory]
) -> Tuple[Dict[Tuple[str, str], List], Dict[str, str]]:
    """
    Build robust image->annotations index and image->split mapping.
    Keys are (split_purpose, image_filename) to avoid collisions.
    Uses ONLY canonical 100k_train, 100k_val, 100k_test for detection annotation matching.
    Searches recursively under canonical split directories.
    """
    index = defaultdict(list)
    image_to_split = {}

    # Build image filename set per split - ONLY canonical detection splits
    canonical_splits = {"100k_train", "100k_val", "100k_test"}
    split_image_sets = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    for d in image_dirs:
        if d.purpose not in canonical_splits:
            continue  # Only use canonical 100k splits for detection annotation matching
        # Recursively collect all image filenames under this canonical split
        image_names = _collect_image_names_recursive(Path(d.path), image_extensions)
        split_image_sets[d.purpose] = image_names
        print(f"    [INDEX] {d.purpose}: {len(image_names)} images (recursive)")

    # Match annotations to images using canonical splits
    for ann in all_annotations:
        frame_name = ann.get("_frame_name", "")
        if not frame_name:
            continue

        # Find which canonical split this image belongs to
        matched_split = None
        for purpose, img_set in split_image_sets.items():
            if frame_name in img_set:
                matched_split = purpose
                break

        if matched_split:
            key = (matched_split, frame_name)
            index[key].append(ann)
            image_to_split[frame_name] = matched_split

    return index, image_to_split


def _count_labeled_recursive(root_path: Path, image_to_split: Dict[str, str], purpose: str, extensions: set) -> int:
    """Count labeled images recursively under a directory."""
    labeled = 0
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if Path(fname).suffix.lower() in extensions:
                if fname in image_to_split and image_to_split[fname] == purpose:
                    labeled += 1
    return labeled


def count_images_by_split(
    image_dirs: List[ImageDirectory],
    image_to_split: Dict[str, str]
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, Dict[str, int]]]:
    """Count images per split separately with proper categorization.
    Returns: (canonical_counts, derived_counts, dataset_10k_counts, seg_counts, labeled_counts, unlabeled_counts, nested_partitions_map)
    """
    print(f"\n{'='*60}")
    print("IMAGE COUNTS BY SPLIT (separate reporting)")
    print(f"{'='*60}")

    canonical_counts = {}
    derived_counts = {}
    dataset_10k_counts = {}
    seg_counts = {}
    labeled_counts = {}
    unlabeled_counts = {}
    nested_partitions_map = {}

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for d in image_dirs:
        count = d.image_count
        
        # For canonical 100k splits, count labeled recursively
        if d.purpose.startswith("100k_"):
            labeled = _count_labeled_recursive(Path(d.path), image_to_split, d.purpose, image_extensions)
        else:
            # For others, use direct listing
            labeled = 0
            for fname in os.listdir(d.path):
                if any(fname.lower().endswith(ext) for ext in image_extensions):
                    if fname in image_to_split and image_to_split[fname] == d.purpose:
                        labeled += 1
        
        unlabeled = count - labeled

        d.labeled_count = labeled
        d.unlabeled_count = unlabeled

        # Collect nested partitions for canonical splits
        if d.purpose.startswith("100k_") and d.nested_partitions:
            nested_partitions_map[d.purpose] = d.nested_partitions

        if d.purpose.startswith("100k_"):
            canonical_counts[d.purpose] = count
            labeled_counts[d.purpose] = labeled
            unlabeled_counts[d.purpose] = unlabeled
            part_info = f" | partitions: {d.nested_partitions}" if d.nested_partitions else ""
            print(f"  [CANONICAL] {d.purpose}: {count} images ({labeled} labeled, {unlabeled} unlabeled){part_info}")
        elif d.purpose in ("trainA", "trainB", "testA", "testB"):
            # Use relative path as unique key to avoid overwriting multiple trainA/trainB at different paths
            key = d.relative_path
            derived_counts[key] = count
            labeled_counts[key] = labeled
            unlabeled_counts[key] = unlabeled
            print(f"  [DERIVED] {key} ({d.purpose}): {count} images ({labeled} labeled, {unlabeled} unlabeled)")
        elif d.purpose.startswith("10k_"):
            dataset_10k_counts[d.purpose] = count
            labeled_counts[d.purpose] = labeled
            unlabeled_counts[d.purpose] = unlabeled
            print(f"  [10K] {d.purpose}: {count} images ({labeled} labeled, {unlabeled} unlabeled)")
        elif d.purpose.startswith("seg"):
            seg_counts[d.purpose] = count
            print(f"  [SEG] {d.purpose}: {count} images (segmentation - excluded from detection counts)")
        else:
            print(f"  [OTHER] {d.purpose}: {count} images ({labeled} labeled, {unlabeled} unlabeled)")

    total_canonical = sum(canonical_counts.values())
    total_derived = sum(derived_counts.values())
    total_10k = sum(dataset_10k_counts.values())
    total_seg = sum(seg_counts.values())

    print(f"\n  Canonical detection images (100k): {total_canonical}")
    print(f"  Derived subsets (trainA/trainB/testA/testB): {total_derived}")
    print(f"  10k dataset images: {total_10k}")
    print(f"  Segmentation images: {total_seg}")
    print(f"  TOTAL (all categories): {total_canonical + total_derived + total_10k + total_seg}")

    return canonical_counts, derived_counts, dataset_10k_counts, seg_counts, labeled_counts, unlabeled_counts, nested_partitions_map


def validate_annotations_with_images(
    all_annotations: List[Dict],
    image_dirs: List[ImageDirectory],
    kaggle_input_dir: Path
) -> Dict[str, Any]:
    """Validate bounding boxes using actual image dimensions per split."""
    print(f"\n{'='*60}")
    print("VALIDATING ANNOTATIONS WITH IMAGE DIMENSIONS")
    print(f"{'='*60}")

    # Build image path lookup with split context
    image_dims = {}  # (split_purpose, image_name) -> (width, height)
    image_dims_by_name = {}  # image_name -> (width, height) - fallback

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for d in image_dirs:
        if d.purpose.startswith("seg"):
            continue
        # For canonical 100k splits, recursively walk to get all nested images
        if d.purpose in ("100k_train", "100k_val", "100k_test"):
            for dirpath, _, filenames in os.walk(d.path):
                for img_name in filenames:
                    if Path(img_name).suffix.lower() in image_extensions:
                        img_path = Path(dirpath) / img_name
                        dims = get_image_dimensions(img_path)
                        if dims:
                            image_dims[(d.purpose, img_name)] = dims
                            image_dims_by_name[img_name] = dims
        else:
            # Non-canonical: direct listing only
            for img_name in os.listdir(d.path):
                if Path(img_name).suffix.lower() in image_extensions:
                    img_path = Path(d.path) / img_name
                    dims = get_image_dimensions(img_path)
                    if dims:
                        image_dims[(d.purpose, img_name)] = dims
                        image_dims_by_name[img_name] = dims

    print(f"  Loaded dimensions for {len(image_dims_by_name)} detection images")

    valid_boxes = 0
    invalid_boxes = 0
    unverified_boxes = 0  # boxes where image dims couldn't be found
    non_bbox_annotations = 0  # legitimate annotations without bbox (lane, drivable area, etc.)
    negative_coord = 0
    zero_area = 0
    outside_image = 0
    class_counts = Counter()
    bbox_class_counts = Counter()  # classes with valid box2d/bbox
    non_detection_class_counts = Counter()  # classes without boxes (seg-only like lane, drivable area)

    for ann in all_annotations:
        category = ann.get("category") or ann.get("category_id")
        if category:
            class_counts[category] += 1

        box = None
        box_format = None
        has_box = False
        if "box2d" in ann and isinstance(ann["box2d"], dict):
            box = ann["box2d"]
            if all(k in box for k in ["x1", "y1", "x2", "y2"]):
                box_format = "xyxy"
                has_box = True
            elif all(k in box for k in ["x", "y", "w", "h"]):
                box_format = "xywh"
                has_box = True
        elif "bbox" in ann and isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4:
            box = {"x": ann["bbox"][0], "y": ann["bbox"][1], "w": ann["bbox"][2], "h": ann["bbox"][3]}
            box_format = "xywh_coco"
            has_box = True

        if has_box and category:
            bbox_class_counts[category] += 1
        elif category and not has_box:
            non_detection_class_counts[category] += 1

        if not box:
            non_bbox_annotations += 1
            continue

        if box_format in ("xyxy",):
            xmin, ymin, xmax, ymax = box["x1"], box["y1"], box["x2"], box["y2"]
            w, h = xmax - xmin, ymax - ymin
        elif box_format in ("xywh", "xywh_coco"):
            xmin, ymin = box["x"], box["y"]
            w, h = box["w"], box["h"]
            xmax, ymax = xmin + w, ymin + h
        else:
            invalid_boxes += 1
            continue

        if xmin < 0 or ymin < 0:
            negative_coord += 1
            invalid_boxes += 1
            continue
        if w <= 0 or h <= 0:
            zero_area += 1
            invalid_boxes += 1
            continue

        img_name = ann.get("_frame_name", "")
        split = ann.get("_source_split", "")
        dims = image_dims.get((split, img_name)) or image_dims_by_name.get(img_name)
        if dims:
            img_w, img_h = dims
            if xmax > img_w or ymax > img_h:
                outside_image += 1
                invalid_boxes += 1
                continue
            valid_boxes += 1
        else:
            unverified_boxes += 1

    print(f"  Valid boxes: {valid_boxes}")
    print(f"  Invalid boxes: {invalid_boxes}")
    print(f"    - Negative coordinates: {negative_coord}")
    print(f"    - Zero area: {zero_area}")
    print(f"    - Outside image bounds: {outside_image}")
    print(f"  Non-bbox annotations (legitimate, e.g., lane, drivable area): {non_bbox_annotations}")
    print(f"  Unverified boxes (no image dims found): {unverified_boxes}")

    if class_counts:
        print(f"\n  All classes found in annotations:")
        for cls, count in class_counts.most_common():
            marker = " [BBOX]" if cls in bbox_class_counts else " [NON-BBOX]"
            print(f"    {cls}: {count}{marker}")

    if non_detection_class_counts:
        print(f"\n  Non-detection annotation categories (no bounding box):")
        for cls, count in non_detection_class_counts.most_common():
            print(f"    {cls}: {count}")

    return {
        "valid_boxes": valid_boxes,
        "invalid_boxes": invalid_boxes,
        "unverified_boxes": unverified_boxes,
        "non_bbox_annotations": non_bbox_annotations,
        "negative_coord_boxes": negative_coord,
        "zero_area_boxes": zero_area,
        "outside_image_boxes": outside_image,
        "class_counts": dict(class_counts),
        "bbox_class_counts": dict(bbox_class_counts),
        "non_detection_class_counts": dict(non_detection_class_counts),
        "total_objects": len(all_annotations),
    }


def generate_reports(
    env_info: Dict,
    image_dirs: List[ImageDirectory],
    annotation_sources: List[AnnotationSource],
    all_annotations: List[Dict],
    frame_level_data: List[Dict],
    metadata_counts: Dict[str, Counter],
    validation_results: Dict[str, Any],
    canonical_counts: Dict[str, int],
    derived_counts: Dict[str, int],
    dataset_10k_counts: Dict[str, int],
    seg_counts: Dict[str, int],
    labeled_counts: Dict[str, int],
    unlabeled_counts: Dict[str, int],
    nested_partitions_map: Dict[str, Dict[str, int]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate machine-readable JSON and human-readable text reports."""
    print(f"\n{'='*60}")
    print("GENERATING DATASET REPORTS")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build dataset_findings section ---
    dataset_findings = {
        "image_directories": [
            {
                "name": d.name,
                "path": d.path,
                "relative_path": d.relative_path,
                "purpose": d.purpose,
                "image_count": d.image_count,
                "labeled_count": d.labeled_count,
                "unlabeled_count": d.unlabeled_count,
                "nested_partitions": d.nested_partitions,
            }
            for d in image_dirs
        ],
        "annotation_sources": [
            {
                "path": a.path,
                "relative_path": a.relative_path,
                "size_bytes": a.size_bytes,
                "classification": a.classification,
                "confidence": a.confidence,
                "reason": a.reason,
                "schema": a.schema,
            }
            for a in annotation_sources
        ],
        "schema_identified": {
            "detection_format": None,
            "has_frame_level_metadata": len(frame_level_data) > 0,
            "frame_metadata_keys": list(set().union(*[set(f.get("attributes", {}).keys()) for f in frame_level_data])) if frame_level_data else [],
        }
    }

    for a in annotation_sources:
        if a.classification in ("detection_bdd100k", "detection_coco"):
            dataset_findings["schema_identified"]["detection_format"] = a.classification
            break

    # --- Identify recommended detector training resources ---
    # Find training and validation annotation files (BDD100K detection format)
    train_ann_file = None
    val_ann_file = None
    for a in annotation_sources:
        if a.classification == "detection_bdd100k":
            rel = a.relative_path.lower()
            if "train" in rel and "val" not in rel and "test" not in rel:
                train_ann_file = a.relative_path
            elif "val" in rel:
                val_ann_file = a.relative_path

    # Find canonical image directories
    train_img_dir = None
    val_img_dir = None
    for d in image_dirs:
        if d.purpose == "100k_train":
            train_img_dir = d.relative_path
        elif d.purpose == "100k_val":
            val_img_dir = d.relative_path

    # Vehicle detection classes (from bbox_class_counts)
    vehicle_classes = []
    non_vehicle_classes = []
    exclude_classes = []

    # User-specified vehicle categories for road-vehicle detector
    # Note: bike = bicycle, motor = motorcycle in BDD100K
    vehicle_categories = {"car", "truck", "bus", "bike", "motor", "motorcycle", "bicycle"}
    # Optional vehicle/rail class
    optional_vehicle_categories = {"train"}
    # Categories to exclude from vehicle detection (non-vehicle objects)
    exclude_categories = {"person", "rider", "traffic sign", "traffic light"}
    # Segmentation-only categories (no bounding boxes)
    seg_only_categories = {"lane", "drivable area", "road", "sidewalk"}

    bbox_classes = validation_results.get("bbox_class_counts", {})
    non_bbox_classes = validation_results.get("non_detection_class_counts", {})

    for cls in bbox_classes.keys():
        if cls in vehicle_categories:
            vehicle_classes.append(cls)
        elif cls in optional_vehicle_categories:
            vehicle_classes.append(cls)  # Will be listed as optional in report
        elif cls in exclude_categories:
            exclude_classes.append(cls)
        else:
            non_vehicle_classes.append(cls)

    for cls in non_bbox_classes.keys():
        if cls in seg_only_categories:
            exclude_classes.append(cls)
        else:
            exclude_classes.append(cls)

    # --- Machine-readable JSON report ---
    json_report = {
        "generated_at": datetime.now().isoformat(),
        "dataset": "BDD100K",
        "environment": env_info,
        "dataset_findings": dataset_findings,
        "canonical_detection_images": canonical_counts,
        "derived_subsets": derived_counts,
        "dataset_10k": dataset_10k_counts,
        "segmentation_images": seg_counts,
        "nested_partitions_inside_100k_train": nested_partitions_map.get("100k_train", {}),
        "summary": {
            "total_canonical_detection_images": sum(canonical_counts.values()),
            "total_derived_subset_images": sum(derived_counts.values()),
            "total_10k_images": sum(dataset_10k_counts.values()),
            "total_segmentation_images": sum(seg_counts.values()),
            "total_annotated_objects": validation_results["total_objects"],
            "total_labeled_images": sum(labeled_counts.values()),
            "total_unlabeled_images": sum(unlabeled_counts.values()),
            "per_class_counts": validation_results["class_counts"],
            "bbox_class_counts": validation_results.get("bbox_class_counts", {}),
            "non_detection_class_counts": validation_results.get("non_detection_class_counts", {}),
            "day_night_counts": dict(metadata_counts.get("timeofday", {})),
            "weather_distribution": dict(metadata_counts.get("weather", {})),
            "scene_distribution": dict(metadata_counts.get("scene", {})),
            "box_validation": {
                "valid": validation_results["valid_boxes"],
                "invalid": validation_results["invalid_boxes"],
                "unverified": validation_results["unverified_boxes"],
                "negative_coords": validation_results["negative_coord_boxes"],
                "zero_area": validation_results["zero_area_boxes"],
                "outside_image": validation_results["outside_image_boxes"],
            },
            "annotation_files": {
                "detection_bdd100k": len([a for a in annotation_sources if a.classification == "detection_bdd100k"]),
                "detection_coco": len([a for a in annotation_sources if a.classification == "detection_coco"]),
                "tracking": len([a for a in annotation_sources if a.classification == "tracking"]),
                "segmentation": len([a for a in annotation_sources if a.classification == "segmentation"]),
                "metadata": len([a for a in annotation_sources if a.classification == "metadata"]),
                "unknown": len([a for a in annotation_sources if a.classification == "unknown"]),
            }
        },
        "recommended_for_detector_training": {
            "training_image_directory": train_img_dir,
            "validation_image_directory": val_img_dir,
            "training_annotation_file": train_ann_file,
            "validation_annotation_file": val_ann_file,
            "classes_suitable_for_vehicle_detection": vehicle_classes,
            "classes_to_exclude_from_vehicle_detection": exclude_classes,
            "non_vehicle_bbox_classes": non_vehicle_classes,
            "note": "Test ground truth not available in public BDD100K detection annotations" if not val_ann_file else "Using canonical 100k splits"
        }
    }

    json_path = output_dir / "dataset_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"  Written: {json_path}")

    # --- Human-readable text report ---
    txt_lines = [
        "=" * 60,
        "BDD100K DATASET INSPECTION REPORT - Phase 1",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "ENVIRONMENT",
        "-" * 40,
        f"  PyTorch: {env_info['pytorch_version']}",
        f"  CUDA: {env_info['cuda_available']} ({env_info['cuda_device_count']} devices)",
        f"  TensorFlow: {env_info['tensorflow_version']}",
        "",
        "CANONICAL DETECTION IMAGES (100k splits only)",
        "-" * 40,
    ]

    for purpose in ["100k_train", "100k_val", "100k_test"]:
        count = canonical_counts.get(purpose, 0)
        labeled = labeled_counts.get(purpose, 0)
        unlabeled = unlabeled_counts.get(purpose, 0)
        if count > 0:
            txt_lines.append(f"  {purpose}: {count} images ({labeled} labeled, {unlabeled} unlabeled)")

    # Nested partitions inside 100k_train
    train_partitions = nested_partitions_map.get("100k_train", {})
    if train_partitions:
        txt_lines.extend([
            "",
            "NESTED PARTITIONS INSIDE 100k_TRAIN (metadata only, not additional datasets)",
            "-" * 40,
        ])
        total_partitions = 0
        for part_name in ["trainA", "trainB", "testA", "testB", "root_direct"]:
            count = train_partitions.get(part_name, 0)
            if count > 0:
                txt_lines.append(f"  {part_name}: {count}")
                total_partitions += count
        # Also include any other partitions not in the standard list
        for part_name, count in train_partitions.items():
            if part_name not in ["trainA", "trainB", "testA", "testB", "root_direct"] and count > 0:
                txt_lines.append(f"  {part_name}: {count}")
                total_partitions += count
        txt_lines.append(f"  TOTAL (sum of partitions): {total_partitions}")

    txt_lines.extend([
        "",
        "DERIVED SUBSETS (trainA/trainB/testA/testB - NOT in canonical totals, reported as metadata)",
        "-" * 40,
    ])

    for purpose in ["trainA", "trainB", "testA", "testB"]:
        count = derived_counts.get(purpose, 0)
        labeled = labeled_counts.get(purpose, 0)
        unlabeled = unlabeled_counts.get(purpose, 0)
        if count > 0:
            txt_lines.append(f"  {purpose}: {count} images ({labeled} labeled, {unlabeled} unlabeled)")

    txt_lines.extend([
        "",
        "10K DATASET (separate from 100k)",
        "-" * 40,
    ])

    for purpose in ["10k_train", "10k_val", "10k_test"]:
        count = dataset_10k_counts.get(purpose, 0)
        labeled = labeled_counts.get(purpose, 0)
        unlabeled = unlabeled_counts.get(purpose, 0)
        if count > 0:
            txt_lines.append(f"  {purpose}: {count} images ({labeled} labeled, {unlabeled} unlabeled)")

    txt_lines.extend([
        "",
        "SEGMENTATION IMAGES (NOT detection)",
        "-" * 40,
    ])

    for purpose, count in seg_counts.items():
        txt_lines.append(f"  {purpose}: {count} images")

    txt_lines.extend([
        "",
        "ANNOTATION SOURCES (classified by CONTENT)",
        "-" * 40,
    ])

    for source in annotation_sources:
        txt_lines.append(f"  [{source.classification.upper()}] {source.relative_path}")
        txt_lines.append(f"    Confidence: {source.confidence:.0%} | Reason: {source.reason}")
        if source.schema:
            s = source.schema
            txt_lines.append(f"    Format: {s.get('format', 'unknown')}")
            if 'has_frame_metadata' in s:
                txt_lines.append(f"    Frame metadata: {s['has_frame_metadata']}")
            if 'frame_keys' in s:
                txt_lines.append(f"    Frame keys: {s['frame_keys']}")
            if 'label_keys' in s:
                txt_lines.append(f"    Label keys: {s['label_keys']}")

    txt_lines.extend([
        "",
        "DETECTION SCHEMA",
        "-" * 40,
    ])

    det_sources = [a for a in annotation_sources if a.classification in ("detection_bdd100k", "detection_coco")]
    if det_sources:
        s = det_sources[0].schema
        txt_lines.append(f"  Primary format: {det_sources[0].classification}")
        txt_lines.append(f"  Has box2d: {s.get('has_box2d')}")
        if s.get('box2d_keys'):
            txt_lines.append(f"    box2d keys: {s.get('box2d_keys')}")
        txt_lines.append(f"  Has bbox (COCO): {s.get('has_bbox')}")
        txt_lines.append(f"  Has category: {s.get('has_category')}")
        txt_lines.append(f"  Has category_id: {s.get('has_category_id')}")
        txt_lines.append(f"  Has image_id: {s.get('has_image_id')}")
        txt_lines.append(f"  Has object attributes: {s.get('has_object_attributes')}")
        if s.get('object_attribute_keys'):
            txt_lines.append(f"    object attr keys: {s.get('object_attribute_keys')}")
        txt_lines.append(f"  Has frame attributes: {s.get('has_frame_attributes')}")
        if s.get('frame_attribute_keys'):
            txt_lines.append(f"    frame attr keys: {s.get('frame_attribute_keys')}")
        txt_lines.append(f"  Categories found: {s.get('categories_sample', [])}")

    txt_lines.extend([
        "",
        "METADATA DISTRIBUTION (frame-level attributes - ALL frames)",
        "-" * 40,
    ])

    # Report timeofday with daytime/night/dawn/dusk categorization
    timeofday_counts = metadata_counts.get("timeofday", Counter())
    if timeofday_counts:
        txt_lines.append("  timeofday:")
        daytime_total = 0
        night_total = 0
        dawndusk_total = 0
        for k, v in timeofday_counts.most_common():
            txt_lines.append(f"    {k}: {v}")
            kl = k.lower()
            if kl in ("daytime", "day"):
                daytime_total += v
            elif kl in ("night",):
                night_total += v
            elif kl in ("dawn/dusk", "dawn", "dusk", "twilight"):
                dawndusk_total += v
        txt_lines.append(f"    SUMMARY: daytime={daytime_total}, night={night_total}, dawn/dusk={dawndusk_total}")
    else:
        txt_lines.append("  timeofday: not found")

    for key in ["weather", "scene"]:
        if key in metadata_counts and metadata_counts[key]:
            txt_lines.append(f"  {key}:")
            for k, v in metadata_counts[key].most_common():
                txt_lines.append(f"    {k}: {v}")
        else:
            txt_lines.append(f"  {key}: not found")

    txt_lines.extend([
        "",
        "PER-CLASS OBJECT COUNTS (all annotations)",
        "-" * 40,
    ])

    for cls, count in validation_results.get("class_counts", {}).items():
        marker = " [BBOX]" if cls in validation_results.get("bbox_class_counts", {}) else " [NON-BBOX]"
        txt_lines.append(f"  {cls}: {count}{marker}")

    if validation_results.get("non_detection_class_counts"):
        txt_lines.extend([
            "",
            "NON-DETECTION ANNOTATION CATEGORIES (no bounding box - e.g., lane, drivable area)",
            "-" * 40,
        ])
        for cls, count in validation_results.get("non_detection_class_counts", {}).items():
            txt_lines.append(f"  {cls}: {count}")

    txt_lines.extend([
        "",
        "BOUNDING BOX VALIDATION",
        "-" * 40,
        f"  Valid boxes: {validation_results['valid_boxes']}",
        f"  Invalid boxes: {validation_results['invalid_boxes']}",
        f"    - Negative coordinates: {validation_results['negative_coord_boxes']}",
        f"    - Zero area: {validation_results['zero_area_boxes']}",
        f"    - Outside image bounds: {validation_results['outside_image_boxes']}",
        f"  Unverified boxes (no image dims): {validation_results['unverified_boxes']}",
        "",
        "ANNOTATION FILE SUMMARY",
        "-" * 40,
    ])

    for cat in ["detection_bdd100k", "detection_coco", "tracking", "segmentation", "metadata", "unknown"]:
        count = len([a for a in annotation_sources if a.classification == cat])
        if count:
            txt_lines.append(f"  {cat}: {count} files")

    # RECOMMENDED DETECTOR DATASET section
    txt_lines.extend([
        "",
        "=" * 60,
        "RECOMMENDED DETECTOR DATASET",
        "=" * 60,
        "Training images:",
        f"  exact recursive path/tree: {train_img_dir or 'NOT FOUND'}",
        "Validation images:",
        f"  exact path: {val_img_dir or 'NOT FOUND'}",
        "Training annotations:",
        f"  exact JSON path: {train_ann_file or 'NOT FOUND'}",
        "Validation annotations:",
        f"  exact JSON path: {val_ann_file or 'NOT FOUND'}",
        "",
        "Recommended road-vehicle classes:",
    ])

    # List vehicle classes in the specified order
    ordered_vehicle_classes = ["car", "truck", "bus", "bike", "motor"]
    found_vehicle = False
    for cls in ordered_vehicle_classes:
        if cls in bbox_classes:
            count = bbox_classes.get(cls, 0)
            txt_lines.append(f"  {cls}: {count} instances")
            found_vehicle = True
    
    # Check for motorcycle/bicycle as aliases
    for cls in ["motorcycle", "bicycle"]:
        if cls in bbox_classes and cls not in ordered_vehicle_classes:
            count = bbox_classes.get(cls, 0)
            txt_lines.append(f"  {cls}: {count} instances")
            found_vehicle = True
    
    if not found_vehicle:
        txt_lines.append("  (none found)")

    txt_lines.extend([
        "",
        "Optional:",
    ])
    
    # Optional vehicle/rail class
    if "train" in bbox_classes:
        count = bbox_classes.get("train", 0)
        txt_lines.append(f"  train: {count} instances")
    else:
        txt_lines.append("  train: (not found)")

    txt_lines.extend([
        "",
        "Excluded:",
    ])

    # Excluded classes in specified order
    excluded_order = ["person", "rider", "traffic light", "traffic sign", "lane", "drivable area"]
    for cls in excluded_order:
        count = non_bbox_classes.get(cls, 0) or bbox_classes.get(cls, 0)
        if count > 0:
            txt_lines.append(f"  {cls}: {count} instances")
        else:
            txt_lines.append(f"  {cls}: (not found)")

    txt_lines.extend([
        "",
        "Note: Public BDD100K detection annotations do not include test ground truth.",
        "Use 100k_train (recursive) for training, 100k_val for validation.",
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])

    txt_path = output_dir / "dataset_report.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))
    print(f"  Written: {txt_path}")

    return json_path, txt_path


def main():
    parser = argparse.ArgumentParser(description="BDD100K Dataset Inspection - Phase 1")
    parser.add_argument("--kaggle-input", type=str, default=None, help="Override kaggle input path")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/traffic_ai/reports", help="Output directory for reports")
    args = parser.parse_args()

    print("BDD100K Dataset Inspection - Phase 1")
    print("=" * 60)

    # Step 1: Detect environment
    env_info = detect_environment()
    print(f"\nPyTorch: {env_info['pytorch_version']}")
    print(f"CUDA: {env_info['cuda_available']} ({env_info['cuda_device_count']} devices)")
    for gpu in env_info["gpus"]:
        print(f"  GPU {gpu['index']}: {gpu['name']}, {gpu['total_memory_gb']} GB")
    print(f"TensorFlow: {env_info['tensorflow_version']}")

    # Step 2: Find /kaggle/input/
    kaggle_input_dir = find_kaggle_input(args.kaggle_input)
    if kaggle_input_dir is None:
        print("\nERROR: /kaggle/input/ directory not found.")
        print("This script is designed for Kaggle execution.")
        return 1

    print(f"\n[kaggle/input] Found: {kaggle_input_dir}")

    # Step 3: Scan dataset structure (filesystem only, no JSON loading)
    image_dirs, annotation_sources = scan_kaggle_input(kaggle_input_dir)

    # Step 4: Load ALL annotations from confirmed detection sources
    all_annotations, frame_level_data = load_detection_annotations(annotation_sources)

    # Step 5: Build image->annotation index with split context
    image_to_annots, image_to_split = build_image_annotation_index(all_annotations, image_dirs)

    # Step 6: Extract frame-level metadata
    metadata_counts = extract_frame_metadata(frame_level_data)

    # Step 7: Count images by split (separate reporting)
    canonical_counts, derived_counts, dataset_10k_counts, seg_counts, labeled_counts, unlabeled_counts, nested_partitions_map = count_images_by_split(image_dirs, image_to_split)

    # Step 8: Validate annotations with image dimensions
    validation_results = validate_annotations_with_images(all_annotations, image_dirs, kaggle_input_dir)

    # Step 9: Generate reports
    output_dir = Path(args.output_dir)
    json_path, txt_path = generate_reports(
        env_info, image_dirs, annotation_sources,
        all_annotations, frame_level_data,
        metadata_counts, validation_results,
        canonical_counts, derived_counts, dataset_10k_counts, seg_counts,
        labeled_counts, unlabeled_counts,
        nested_partitions_map,
        output_dir
    )

    # Final summary
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"\nReports generated:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())