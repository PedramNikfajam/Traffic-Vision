"""
BDD100K to YOLO format conversion utilities.

Pure functions for annotation parsing, bbox conversion, and accounting.
No side effects, no file I/O. Testable in isolation.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any


# ============================================================
# BOUNDING BOX
# ============================================================

@dataclass
class BBox:
    """Bounding box in absolute pixel coordinates (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)
    
    def validate(self, img_w: int, img_h: int) -> Tuple[bool, str]:
        """Validate bbox geometry against image dimensions.
        
        Returns:
            (is_valid, error_reason) where error_reason is '' if valid.
        """
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            return False, "zero_or_negative_area"
        if self.x1 < 0 or self.y1 < 0:
            return False, "negative_coordinates"
        if self.x2 > img_w + 1 or self.y2 > img_h + 1:
            # Allow 1px tolerance for floating point
            return False, "outside_image"
        return True, ""
    
    def clip(self, img_w: int, img_h: int) -> "BBox":
        """Clip bbox to image boundaries."""
        return BBox(
            x1=max(0.0, min(self.x1, float(img_w))),
            y1=max(0.0, min(self.y1, float(img_h))),
            x2=max(0.0, min(self.x2, float(img_w))),
            y2=max(0.0, min(self.y2, float(img_h))),
        )
    
    def to_yolo(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert to YOLO format (cx, cy, w, h) normalized to [0, 1].
        
        Returns:
            (center_x, center_y, width, height) all in [0, 1].
        
        Raises:
            ValueError: If resulting coordinates are outside [0, 1].
        """
        cx = (self.x1 + self.x2) / 2.0 / img_w
        cy = (self.y1 + self.y2) / 2.0 / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        
        # Strict validation
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise ValueError(
                f"YOLO center outside [0,1]: cx={cx:.6f}, cy={cy:.6f} "
                f"from bbox [{self.x1},{self.y1},{self.x2},{self.y2}] "
                f"in image {img_w}x{img_h}"
            )
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise ValueError(
                f"YOLO dims outside (0,1]: w={w:.6f}, h={h:.6f} "
                f"from bbox [{self.x1},{self.y1},{self.x2},{self.y2}] "
                f"in image {img_w}x{img_h}"
            )
        
        return cx, cy, w, h


def parse_bdd100k_box2d(box2d: dict) -> Optional[BBox]:
    """Parse a BDD100K box2d dict into a BBox.
    
    BDD100K uses {x1, y1, x2, y2} format.
    
    Returns:
        BBox or None if parsing fails.
    """
    try:
        return BBox(
            x1=float(box2d["x1"]),
            y1=float(box2d["y1"]),
            x2=float(box2d["x2"]),
            y2=float(box2d["y2"]),
        )
    except (KeyError, ValueError, TypeError):
        return None


# ============================================================
# ANNOTATION ACCOUNTING
# ============================================================

@dataclass
class AnnotationAccounting:
    """Rigorous annotation accounting.
    
    Every source annotation MUST end up in exactly one category.
    The total must reconcile: sum of all categories == source_total.
    """
    source_total: int = 0
    
    # Disposition categories (must sum to source_total)
    target_bbox: int = 0          # Selected vehicle with valid bbox -> YOLO label
    excluded_class: int = 0       # person, rider, traffic sign, etc.
    non_bbox_class: int = 0       # lane, drivable area (no bounding box)
    target_no_bbox: int = 0       # Vehicle class but no box2d present
    invalid_bbox: int = 0         # Has box2d but geometry is invalid
    unknown_class: int = 0        # Category not in any known list
    
    # Detail counters (for diagnostics, not part of reconciliation)
    target_by_class: Dict[str, int] = field(default_factory=dict)
    excluded_by_class: Dict[str, int] = field(default_factory=dict)
    invalid_reasons: Dict[str, int] = field(default_factory=dict)
    
    @property
    def accounted_total(self) -> int:
        """Sum of all disposition categories."""
        return (
            self.target_bbox
            + self.excluded_class
            + self.non_bbox_class
            + self.target_no_bbox
            + self.invalid_bbox
            + self.unknown_class
        )
    
    def verify(self) -> Tuple[bool, str]:
        """Verify accounting reconciles.
        
        Returns:
            (passed, message)
        """
        if self.source_total == 0:
            return False, "No annotations processed"
        if self.accounted_total != self.source_total:
            return False, (
                f"Accounting mismatch: source={self.source_total}, "
                f"accounted={self.accounted_total}, "
                f"difference={self.source_total - self.accounted_total}"
            )
        return True, (
            f"Accounting verified: {self.source_total} annotations, "
            f"{self.target_bbox} selected targets"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_total": self.source_total,
            "target_bbox": self.target_bbox,
            "excluded_class": self.excluded_class,
            "non_bbox_class": self.non_bbox_class,
            "target_no_bbox": self.target_no_bbox,
            "invalid_bbox": self.invalid_bbox,
            "unknown_class": self.unknown_class,
            "accounted_total": self.accounted_total,
            "reconciles": self.accounted_total == self.source_total,
            "target_by_class": dict(self.target_by_class),
            "excluded_by_class": dict(self.excluded_by_class),
            "invalid_reasons": dict(self.invalid_reasons),
        }


# ============================================================
# FRAME CONVERSION
# ============================================================

@dataclass
class ConvertedLabel:
    """A single YOLO-format label line."""
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    
    def to_yolo_line(self) -> str:
        """Format as YOLO label line: 'class_id cx cy w h'."""
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class ConvertedFrame:
    """Result of converting a single BDD100K frame."""
    frame_name: str
    split: str
    img_width: int
    img_height: int
    labels: List[ConvertedLabel]
    source_image_path: str
    source_annotation_count: int
    frame_attributes: Dict[str, str]  # timeofday, weather, scene
    
    @property
    def num_objects(self) -> int:
        return len(self.labels)
    
    @property
    def has_objects(self) -> bool:
        return len(self.labels) > 0
    
    @property
    def class_ids_present(self) -> Set[int]:
        return {l.class_id for l in self.labels}
    
    def to_manifest_dict(self) -> Dict[str, Any]:
        """Create manifest entry for this frame."""
        return {
            "frame_name": self.frame_name,
            "split": self.split,
            "source_image": self.source_image_path,
            "width": self.img_width,
            "height": self.img_height,
            "num_objects": self.num_objects,
            "source_annotations": self.source_annotation_count,
            "class_counts": {
                l.class_name: sum(1 for ll in self.labels if ll.class_name == l.class_name) 
                for l in self.labels
            },
            "timeofday": self.frame_attributes.get("timeofday", ""),
            "weather": self.frame_attributes.get("weather", ""),
            "scene": self.frame_attributes.get("scene", ""),
        }


def convert_frame(
    frame: dict,
    img_w: int,
    img_h: int,
    split: str,
    source_image_path: str,
    class_config: Any,  # ClassConfig from config.py
    accounting: AnnotationAccounting,
) -> ConvertedFrame:
    """Convert a single BDD100K frame to YOLO format.
    
    Args:
        frame: BDD100K frame dict with 'name', 'labels', 'attributes'.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        split: 'train' or 'val'.
        source_image_path: Path to the source image file.
        class_config: ClassConfig instance for class mapping.
        accounting: AnnotationAccounting to update (mutated in place).
    
    Returns:
        ConvertedFrame with YOLO labels.
    """
    frame_name = frame.get("name", "")
    frame_attrs = frame.get("attributes", {})
    if not isinstance(frame_attrs, dict):
        frame_attrs = {}
    
    labels_raw = frame.get("labels", [])
    if labels_raw is None:
        labels_raw = []
    
    converted_labels = []
    
    for label in labels_raw:
        accounting.source_total += 1
        
        category = label.get("category", "")
        box2d = label.get("box2d")
        has_box = isinstance(box2d, dict) and len(box2d) > 0
        
        # Classify this annotation
        disposition = class_config.classify_annotation(category, has_box)
        
        if disposition == "non_bbox_class":
            accounting.non_bbox_class += 1
            continue
        
        if disposition == "excluded_class":
            accounting.excluded_class += 1
            accounting.excluded_by_class[category] = (
                accounting.excluded_by_class.get(category, 0) + 1
            )
            continue
        
        if disposition == "unknown_class":
            accounting.unknown_class += 1
            continue
        
        if disposition == "target_no_bbox":
            accounting.target_no_bbox += 1
            continue
        
        # disposition == "target_bbox"
        assert disposition == "target_bbox", f"Unexpected disposition: {disposition}"
        
        # Parse bbox
        bbox = parse_bdd100k_box2d(box2d)
        if bbox is None:
            accounting.invalid_bbox += 1
            accounting.invalid_reasons["parse_failure"] = (
                accounting.invalid_reasons.get("parse_failure", 0) + 1
            )
            continue
        
        # Validate bbox geometry
        is_valid, error = bbox.validate(img_w, img_h)
        if not is_valid:
            accounting.invalid_bbox += 1
            accounting.invalid_reasons[error] = (
                accounting.invalid_reasons.get(error, 0) + 1
            )
            continue
        
        # Clip marginal boxes (within 1px tolerance)
        bbox = bbox.clip(img_w, img_h)
        
        # Convert to YOLO
        try:
            cx, cy, w, h = bbox.to_yolo(img_w, img_h)
        except ValueError:
            accounting.invalid_bbox += 1
            accounting.invalid_reasons["yolo_conversion"] = (
                accounting.invalid_reasons.get("yolo_conversion", 0) + 1
            )
            continue
        
        # Get class ID
        class_id = class_config.get_class_id(category)
        target_name = class_config.get_target_class(category)
        assert class_id is not None, f"Class ID is None for target category: {category}"
        
        converted_labels.append(ConvertedLabel(
            class_id=class_id,
            class_name=target_name,
            cx=cx, cy=cy, w=w, h=h,
        ))
        
        accounting.target_bbox += 1
        accounting.target_by_class[target_name] = (
            accounting.target_by_class.get(target_name, 0) + 1
        )
    
    return ConvertedFrame(
        frame_name=frame_name,
        split=split,
        img_width=img_w,
        img_height=img_h,
        labels=converted_labels,
        source_image_path=source_image_path,
        source_annotation_count=len(labels_raw),
        frame_attributes=frame_attrs,
    )


# ============================================================
# DATASET FINGERPRINTING
# ============================================================

def fingerprint_file(path: Path) -> str:
    """Compute lightweight fingerprint: first 64KB + file size.
    
    Much faster than full file hash but sufficient to detect changes.
    """
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()[:16]


def fingerprint_config(class_config: Any) -> str:
    """Fingerprint the class configuration for provenance tracking."""
    import json
    data = {
        "target_classes": list(class_config.target_classes),
        "source_to_target": dict(class_config.source_to_target),
        "excluded": list(class_config.excluded_categories),
    }
    h = hashlib.sha256(json.dumps(data, sort_keys=True).encode())
    return h.hexdigest()[:16]
