"""
traffic_ai central configuration.

Single source of truth for all project constants, paths, and parameters.
Every script imports from here. No constants scattered in scripts.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


# ============================================================
# ENVIRONMENT DETECTION
# ============================================================

def is_kaggle() -> bool:
    """Detect if running on Kaggle."""
    return os.path.exists("/kaggle/working")


def get_project_root() -> Path:
    """Get the project working directory."""
    if is_kaggle():
        return Path("/kaggle/working/traffic_ai")
    # Local development: find traffic_ai directory
    here = Path(__file__).resolve().parent  # src/
    return here.parent  # traffic_ai/


# ============================================================
# DATASET PATHS (from Phase 1 discovery)
# ============================================================

@dataclass(frozen=True)
class DatasetPaths:
    """BDD100K dataset paths as discovered by Phase 1 inspection.
    
    These are READ-ONLY references to the source dataset on Kaggle.
    They should never be modified by the pipeline.
    """
    # Kaggle input root for BDD100K
    kaggle_input: str = "/kaggle/input"
    
    # Canonical 100k detection image directories
    train_images: str = (
        "/kaggle/input/datasets/solesensei/solesensei_bdd100k/"
        "bdd100k/bdd100k/images/100k/train"
    )
    val_images: str = (
        "/kaggle/input/datasets/solesensei/solesensei_bdd100k/"
        "bdd100k/bdd100k/images/100k/val"
    )
    
    # Annotation files
    train_annotations: str = (
        "/kaggle/input/datasets/solesensei/solesensei_bdd100k/"
        "bdd100k_labels_release/bdd100k/labels/"
        "bdd100k_labels_images_train.json"
    )
    val_annotations: str = (
        "/kaggle/input/datasets/solesensei/solesensei_bdd100k/"
        "bdd100k_labels_release/bdd100k/labels/"
        "bdd100k_labels_images_val.json"
    )
    
    # Expected counts (from Phase 1 verified results)
    expected_train_images: int = 70000
    expected_val_images: int = 10000
    expected_train_labeled: int = 69863
    expected_val_labeled: int = 10000
    
    def verify(self) -> List[str]:
        """Verify all source paths exist. Returns list of errors."""
        errors = []
        for name, path_str in [
            ("train_images", self.train_images),
            ("val_images", self.val_images),
            ("train_annotations", self.train_annotations),
            ("val_annotations", self.val_annotations),
        ]:
            if not Path(path_str).exists():
                errors.append(f"Source path not found: {name} = {path_str}")
        return errors


# ============================================================
# CLASS TAXONOMY
# ============================================================

@dataclass(frozen=True)
class ClassConfig:
    """Vehicle detection class taxonomy.
    
    CRITICAL: BDD100K uses 'bike' and 'motor' as category names,
    NOT 'bicycle' and 'motorcycle'. This was verified in Phase 1.
    The CATEGORY_MAP below maps BDD100K source names to our target names.
    """
    
    # Target detection classes (order = class ID)
    # These are the YOLO class IDs: 0=car, 1=truck, 2=bus, 3=bike, 4=motor
    target_classes: Tuple[str, ...] = ("car", "truck", "bus", "bike", "motor")
    
    # BDD100K source category -> our target class name
    # Only categories that map to a target class are included.
    # BDD100K uses 'bike' for bicycle and 'motor' for motorcycle.
    source_to_target: Dict[str, str] = field(default_factory=lambda: {
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "bike": "bike",
        "motor": "motor",
    })
    
    # All BDD100K categories that are explicitly excluded from vehicle detection
    excluded_categories: Tuple[str, ...] = (
        "person", "rider", "traffic sign", "traffic light",
        "train",  # rail vehicle - excluded by design choice
    )
    
    # BDD100K categories with no bounding box (segmentation-only)
    non_bbox_categories: Tuple[str, ...] = ("lane", "drivable area")
    
    @property
    def num_classes(self) -> int:
        return len(self.target_classes)
    
    @property
    def class_to_id(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.target_classes)}
    
    @property
    def id_to_class(self) -> Dict[int, str]:
        return {i: name for i, name in enumerate(self.target_classes)}
    
    def get_target_class(self, bdd100k_category: str) -> Optional[str]:
        """Map a BDD100K category to our target class, or None."""
        return self.source_to_target.get(bdd100k_category)
    
    def get_class_id(self, bdd100k_category: str) -> Optional[int]:
        """Map a BDD100K category to our target class ID, or None."""
        target = self.get_target_class(bdd100k_category)
        if target is None:
            return None
        return self.class_to_id.get(target)
    
    def classify_annotation(self, category: str, has_box2d: bool) -> str:
        """Classify a single annotation into an accounting category.
        
        Returns one of:
            'target_bbox'     - selected vehicle with valid bbox
            'excluded_class'  - explicitly excluded category
            'non_bbox_class'  - category without bounding boxes (lane, drivable area)
            'target_no_bbox'  - target vehicle class but missing bbox
            'unknown_class'   - category not in any known list
        """
        if category in self.non_bbox_categories:
            return "non_bbox_class"
        if category in self.excluded_categories:
            return "excluded_class"
        if self.get_target_class(category) is not None:
            if has_box2d:
                return "target_bbox"
            else:
                return "target_no_bbox"
        return "unknown_class"


# ============================================================
# OUTPUT PATHS
# ============================================================

@dataclass
class OutputPaths:
    """All output directory paths. Generated under project root."""
    
    project_root: Path = field(default_factory=get_project_root)
    
    @property
    def data(self) -> Path:
        """Prepared dataset root."""
        return self.project_root / "data" / "vehicle_detection"
    
    @property
    def reports(self) -> Path:
        return self.project_root / "reports"
    
    @property
    def experiments(self) -> Path:
        return self.project_root / "experiments"
    
    @property
    def weights(self) -> Path:
        return self.project_root / "weights"
    
    @property
    def plots(self) -> Path:
        return self.project_root / "plots"
    
    @property
    def logs(self) -> Path:
        return self.project_root / "logs"
    
    @property
    def predictions(self) -> Path:
        """Inference outputs (future phase)."""
        return self.project_root / "predictions"
    
    @property
    def videos(self) -> Path:
        """Video outputs (future tracking/counting phases)."""
        return self.project_root / "videos"
    
    def ensure_all(self) -> None:
        """Create all output directories."""
        for d in [self.data, self.reports, self.experiments,
                  self.weights, self.plots, self.logs,
                  self.predictions, self.videos]:
            d.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

@dataclass
class TrainConfig:
    """Training hyperparameters and settings."""
    
    # Model
    model: str = "yolov8m.pt"  # Start with medium for accuracy/speed balance
    
    # Image
    imgsz: int = 640
    
    # Training
    epochs: int = 50
    # TOTAL batch across all GPUs. Ultralytics divides this across DDP ranks
    # internally (batch_size = batch // world_size), so with device="0,1" and
    # batch=32 each GPU trains on 16. 32 total = 16/GPU fits T4-16GB at 640px
    # for yolov8s/m with AMP.
    batch_size: int = 32
    workers: int = 4
    
    # Optimizer
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01  # Final LR = lr0 * lrf
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    
    # Hardware
    device: str = "0"  # Single GPU by default; "0,1" for multi-GPU
    amp: bool = True
    
    # Checkpointing
    save_period: int = 5  # Save checkpoint every N epochs
    patience: int = 15    # Early stopping patience
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Caching
    # False is the only reliable option on Kaggle for the full train set:
    #   "ram"  -> needs ~90GB (31.3GB available) and logs a nondeterminism warning
    #   "disk" -> writes ~190GB of .npy next to the images in /kaggle/working
    #             (quota ~20GB) -> guaranteed mid-run crash
    cache: Any = False
    
    # Validation
    val: bool = True
    
    def to_ultralytics_args(self) -> Dict[str, Any]:
        """Convert to Ultralytics YOLO train() keyword arguments."""
        return {
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "batch": self.batch_size,
            "workers": self.workers,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "device": self.device,
            "amp": self.amp,
            "save_period": self.save_period,
            "patience": self.patience,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "cache": self.cache,
            "val": self.val,
        }


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

@dataclass
class EvalConfig:
    """Evaluation settings."""
    conf_threshold: float = 0.001  # For mAP computation
    iou_threshold: float = 0.6     # NMS threshold
    max_det: int = 300
    
    # Confidence thresholds to analyze
    analysis_thresholds: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75)


# ============================================================
# EXPERIMENT TRACKING
# ============================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    name: str = "baseline"
    description: str = ""
    
    # Components
    dataset: DatasetPaths = field(default_factory=DatasetPaths)
    classes: ClassConfig = field(default_factory=ClassConfig)
    output: OutputPaths = field(default_factory=OutputPaths)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Metadata
    seed: int = 42
    
    @property
    def experiment_id(self) -> str:
        """Generate deterministic experiment ID from name and timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{ts}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON reporting."""
        return {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "classes": {
                "target_classes": list(self.classes.target_classes),
                "num_classes": self.classes.num_classes,
                "excluded": list(self.classes.excluded_categories),
            },
            "train": asdict(self.train),
            "eval": asdict(self.eval),
        }
    
    def save(self, path: Path) -> None:
        """Save experiment config to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# ============================================================
# MODEL PROGRESSION STRATEGY
# ============================================================

MODEL_PROGRESSION = {
    "nano": {
        "weights": "yolov8n.pt",
        "description": "Quick validation, debug runs",
        "batch_size_t4": 32,
    },
    "small": {
        "weights": "yolov8s.pt",
        "description": "Fast baseline experiments",
        "batch_size_t4": 24,
    },
    "medium": {
        "weights": "yolov8m.pt",
        "description": "Strong baseline, good accuracy/speed",
        "batch_size_t4": 16,
    },
    "large": {
        "weights": "yolov8l.pt",
        "description": "High accuracy, slower training",
        "batch_size_t4": 8,
    },
    "xlarge": {
        "weights": "yolov8x.pt",
        "description": "Maximum accuracy, needs gradient accumulation",
        "batch_size_t4": 4,
    },
}


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
SEED = 42

# ============================================================
# GROUND-TRUTH LABEL MAPPING (BDD100K box-track parquet)
# ============================================================
# The BDD100K box-track dump uses its own category names, which do NOT match
# the detection taxonomy: it says 'bicycle'/'motorcycle' where BDD100K
# detection says 'bike'/'motor'. Both map onto our class IDs.
#
# Categories deliberately ABSENT: pedestrian, rider, other person, other
# vehicle, trailer, train, and null. They have no matching detector class, so
# including them in the GT denominator would manufacture false negatives that
# no detector could ever clear. Phase 6/7 log the excluded set per video
# instead of filtering silently.
GT_LABEL_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "bike": "bike",
    "bicycle": "bike",
    "motor": "motor",
    "motorcycle": "motor",
}

# BDD100K standard image dimensions (Phase 1 verified: all 1280x720)
BDD100K_WIDTH = 1280
BDD100K_HEIGHT = 720


# ============================================================
# CONVENIENCE: DEFAULT INSTANCES
# ============================================================

DATASET_PATHS = DatasetPaths()
CLASSES = ClassConfig()
OUTPUT = OutputPaths()
