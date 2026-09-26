"""
Unit tests for traffic_ai core modules.

Runs WITHOUT requiring the BDD100K dataset.
Tests annotation parsing, bbox conversion, class mapping, and accounting.

Run:
    python -m pytest tests/ -v
    python -m pytest tests/test_core.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TEST_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


# ============================================================
# CONFIG TESTS
# ============================================================

class TestClassConfig:
    """Test the ClassConfig class mapping and annotation classification."""

    def setup_method(self):
        from src.config import ClassConfig
        self.config = ClassConfig()

    def test_target_classes_count(self):
        assert self.config.num_classes == 5

    def test_target_classes_order(self):
        assert self.config.target_classes == ("car", "truck", "bus", "bike", "motor")

    def test_class_to_id_mapping(self):
        assert self.config.class_to_id == {
            "car": 0, "truck": 1, "bus": 2, "bike": 3, "motor": 4
        }

    def test_id_to_class_mapping(self):
        assert self.config.id_to_class == {
            0: "car", 1: "truck", 2: "bus", 3: "bike", 4: "motor"
        }

    def test_get_target_class_vehicle(self):
        """BDD100K category names map to our target classes."""
        assert self.config.get_target_class("car") == "car"
        assert self.config.get_target_class("truck") == "truck"
        assert self.config.get_target_class("bus") == "bus"
        assert self.config.get_target_class("bike") == "bike"
        assert self.config.get_target_class("motor") == "motor"

    def test_get_target_class_excluded(self):
        """Excluded categories return None."""
        assert self.config.get_target_class("person") is None
        assert self.config.get_target_class("rider") is None
        assert self.config.get_target_class("traffic sign") is None
        assert self.config.get_target_class("traffic light") is None
        assert self.config.get_target_class("train") is None

    def test_get_target_class_nonbbox(self):
        """Non-bbox categories return None."""
        assert self.config.get_target_class("lane") is None
        assert self.config.get_target_class("drivable area") is None

    def test_get_class_id(self):
        assert self.config.get_class_id("car") == 0
        assert self.config.get_class_id("motor") == 4
        assert self.config.get_class_id("person") is None

    def test_classify_target_bbox(self):
        assert self.config.classify_annotation("car", has_box2d=True) == "target_bbox"
        assert self.config.classify_annotation("truck", has_box2d=True) == "target_bbox"
        assert self.config.classify_annotation("bike", has_box2d=True) == "target_bbox"

    def test_classify_excluded_class(self):
        assert self.config.classify_annotation("person", has_box2d=True) == "excluded_class"
        assert self.config.classify_annotation("rider", has_box2d=True) == "excluded_class"
        assert self.config.classify_annotation("train", has_box2d=True) == "excluded_class"

    def test_classify_non_bbox(self):
        assert self.config.classify_annotation("lane", has_box2d=False) == "non_bbox_class"
        assert self.config.classify_annotation("drivable area", has_box2d=False) == "non_bbox_class"

    def test_classify_target_no_bbox(self):
        assert self.config.classify_annotation("car", has_box2d=False) == "target_no_bbox"

    def test_classify_unknown(self):
        assert self.config.classify_annotation("airplane", has_box2d=True) == "unknown_class"


# ============================================================
# BBOX TESTS
# ============================================================

class TestBBox:
    """Test bounding box validation and YOLO conversion."""

    def test_valid_bbox(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=100, y1=200, x2=300, y2=400)
        is_valid, reason = bbox.validate(1280, 720)
        assert is_valid is True
        assert reason == ""

    def test_zero_area_bbox(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=100, y1=200, x2=100, y2=400)
        is_valid, reason = bbox.validate(1280, 720)
        assert is_valid is False
        assert reason == "zero_or_negative_area"

    def test_negative_area_bbox(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=300, y1=200, x2=100, y2=400)
        is_valid, reason = bbox.validate(1280, 720)
        assert is_valid is False
        assert reason == "zero_or_negative_area"

    def test_negative_coordinates(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=-10, y1=200, x2=300, y2=400)
        is_valid, reason = bbox.validate(1280, 720)
        assert is_valid is False
        assert reason == "negative_coordinates"

    def test_outside_image(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=100, y1=200, x2=1400, y2=400)
        is_valid, reason = bbox.validate(1280, 720)
        assert is_valid is False
        assert reason == "outside_image"

    def test_outside_image_within_tolerance(self):
        """1px floating-point tolerance is allowed."""
        from src.dataset.convert import BBox
        bbox = BBox(x1=100, y1=200, x2=1280.5, y2=400)
        is_valid, reason = bbox.validate(1280, 720)
        assert is_valid is True

    def test_yolo_conversion(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=0, y1=0, x2=640, y2=360)
        cx, cy, w, h = bbox.to_yolo(1280, 720)
        assert abs(cx - 0.25) < 1e-6
        assert abs(cy - 0.25) < 1e-6
        assert abs(w - 0.5) < 1e-6
        assert abs(h - 0.5) < 1e-6

    def test_yolo_full_image(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=0, y1=0, x2=1280, y2=720)
        cx, cy, w, h = bbox.to_yolo(1280, 720)
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6
        assert abs(w - 1.0) < 1e-6
        assert abs(h - 1.0) < 1e-6

    def test_yolo_small_box(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=100, y1=100, x2=110, y2=110)
        cx, cy, w, h = bbox.to_yolo(1280, 720)
        assert 0 < cx < 1
        assert 0 < cy < 1
        assert 0 < w < 1
        assert 0 < h < 1

    def test_clip(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=-10, y1=200, x2=1400, y2=800)
        clipped = bbox.clip(1280, 720)
        assert clipped.x1 == 0
        assert clipped.y1 == 200
        assert clipped.x2 == 1280
        assert clipped.y2 == 720

    def test_area(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=0, y1=0, x2=100, y2=50)
        assert bbox.area == 5000

    def test_width_height(self):
        from src.dataset.convert import BBox
        bbox = BBox(x1=10, y1=20, x2=110, y2=80)
        assert bbox.width == 100
        assert bbox.height == 60


class TestParseBdd100kBox2d:
    """Test parsing BDD100K box2d dictionaries."""

    def test_valid_box(self):
        from src.dataset.convert import parse_bdd100k_box2d
        box = parse_bdd100k_box2d({"x1": 100, "y1": 200, "x2": 300, "y2": 400})
        assert box is not None
        assert box.x1 == 100
        assert box.y2 == 400

    def test_float_values(self):
        from src.dataset.convert import parse_bdd100k_box2d
        box = parse_bdd100k_box2d({"x1": 100.5, "y1": 200.3, "x2": 300.7, "y2": 400.9})
        assert box is not None
        assert abs(box.x1 - 100.5) < 1e-6

    def test_missing_key(self):
        from src.dataset.convert import parse_bdd100k_box2d
        box = parse_bdd100k_box2d({"x1": 100, "y1": 200, "x2": 300})
        assert box is None

    def test_invalid_value(self):
        from src.dataset.convert import parse_bdd100k_box2d
        box = parse_bdd100k_box2d({"x1": "abc", "y1": 200, "x2": 300, "y2": 400})
        assert box is None

    def test_none_input(self):
        from src.dataset.convert import parse_bdd100k_box2d
        box = parse_bdd100k_box2d(None)
        assert box is None


# ============================================================
# ACCOUNTING TESTS
# ============================================================

class TestAnnotationAccounting:
    """Test annotation accounting reconciliation."""

    def test_empty_accounting(self):
        from src.dataset.convert import AnnotationAccounting
        acc = AnnotationAccounting()
        passed, msg = acc.verify()
        assert passed is False  # No annotations = failure

    def test_balanced_accounting(self):
        from src.dataset.convert import AnnotationAccounting
        acc = AnnotationAccounting()
        acc.source_total = 100
        acc.target_bbox = 40
        acc.excluded_class = 30
        acc.non_bbox_class = 20
        acc.target_no_bbox = 5
        acc.invalid_bbox = 3
        acc.unknown_class = 2
        passed, msg = acc.verify()
        assert passed is True

    def test_unbalanced_accounting(self):
        from src.dataset.convert import AnnotationAccounting
        acc = AnnotationAccounting()
        acc.source_total = 100
        acc.target_bbox = 40
        acc.excluded_class = 30
        # Missing 30
        passed, msg = acc.verify()
        assert passed is False
        assert "mismatch" in msg.lower()

    def test_to_dict(self):
        from src.dataset.convert import AnnotationAccounting
        acc = AnnotationAccounting()
        acc.source_total = 10
        acc.target_bbox = 10
        d = acc.to_dict()
        assert d["reconciles"] is True
        assert d["source_total"] == 10


# ============================================================
# CONVERT FRAME TESTS
# ============================================================

class TestConvertFrame:
    """Test full frame conversion pipeline."""

    def _make_frame(self, labels=None, name="test_image.jpg"):
        """Helper to create a BDD100K frame dict."""
        return {
            "name": name,
            "attributes": {
                "timeofday": "daytime",
                "weather": "clear",
                "scene": "highway",
            },
            "labels": labels or [],
        }

    def _make_label(self, category, x1=100, y1=200, x2=300, y2=400):
        """Helper to create a BDD100K label dict."""
        return {
            "category": category,
            "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }

    def test_single_car(self):
        from src.config import ClassConfig
        from src.dataset.convert import AnnotationAccounting, convert_frame

        config = ClassConfig()
        acc = AnnotationAccounting()
        frame = self._make_frame([self._make_label("car")])

        result = convert_frame(frame, 1280, 720, "train", "/test/img.jpg", config, acc)

        assert result.num_objects == 1
        assert result.labels[0].class_id == 0
        assert result.labels[0].class_name == "car"
        assert acc.target_bbox == 1
        assert acc.source_total == 1

    def test_mixed_annotations(self):
        from src.config import ClassConfig
        from src.dataset.convert import AnnotationAccounting, convert_frame

        config = ClassConfig()
        acc = AnnotationAccounting()
        frame = self._make_frame([
            self._make_label("car"),
            self._make_label("person"),
            self._make_label("truck"),
            {"category": "lane"},  # No box2d
            self._make_label("traffic sign"),
        ])

        result = convert_frame(frame, 1280, 720, "train", "/test/img.jpg", config, acc)

        assert result.num_objects == 2  # car + truck
        assert acc.source_total == 5
        assert acc.target_bbox == 2
        assert acc.excluded_class == 2  # person + traffic sign
        assert acc.non_bbox_class == 1  # lane

        passed, _ = acc.verify()
        assert passed is True

    def test_empty_frame(self):
        from src.config import ClassConfig
        from src.dataset.convert import AnnotationAccounting, convert_frame

        config = ClassConfig()
        acc = AnnotationAccounting()
        frame = self._make_frame([])

        result = convert_frame(frame, 1280, 720, "train", "/test/img.jpg", config, acc)

        assert result.num_objects == 0
        assert result.has_objects is False
        assert acc.source_total == 0  # No labels to count

    def test_invalid_bbox_counted(self):
        """Invalid bboxes must be counted in accounting."""
        from src.config import ClassConfig
        from src.dataset.convert import AnnotationAccounting, convert_frame

        config = ClassConfig()
        acc = AnnotationAccounting()
        frame = self._make_frame([
            self._make_label("car", x1=100, y1=200, x2=100, y2=400),  # zero area
            self._make_label("car"),  # valid
        ])

        result = convert_frame(frame, 1280, 720, "train", "/test/img.jpg", config, acc)

        assert result.num_objects == 1
        assert acc.source_total == 2
        assert acc.invalid_bbox == 1
        assert acc.target_bbox == 1

        passed, _ = acc.verify()
        assert passed is True

    def test_frame_attributes_preserved(self):
        from src.config import ClassConfig
        from src.dataset.convert import AnnotationAccounting, convert_frame

        config = ClassConfig()
        acc = AnnotationAccounting()
        frame = self._make_frame([self._make_label("car")])

        result = convert_frame(frame, 1280, 720, "train", "/test/img.jpg", config, acc)

        assert result.frame_attributes["timeofday"] == "daytime"
        assert result.frame_attributes["weather"] == "clear"
        assert result.frame_attributes["scene"] == "highway"

    def test_manifest_dict(self):
        from src.config import ClassConfig
        from src.dataset.convert import AnnotationAccounting, convert_frame

        config = ClassConfig()
        acc = AnnotationAccounting()
        frame = self._make_frame([self._make_label("car"), self._make_label("truck")])

        result = convert_frame(frame, 1280, 720, "train", "/test/img.jpg", config, acc)
        manifest = result.to_manifest_dict()

        assert manifest["frame_name"] == "test_image.jpg"
        assert manifest["split"] == "train"
        assert manifest["width"] == 1280
        assert manifest["height"] == 720
        assert manifest["num_objects"] == 2
        assert manifest["timeofday"] == "daytime"

    def test_yolo_line_format(self):
        from src.dataset.convert import ConvertedLabel
        label = ConvertedLabel(class_id=0, class_name="car",
                               cx=0.5, cy=0.5, w=0.3, h=0.2)
        line = label.to_yolo_line()
        parts = line.split()
        assert len(parts) == 5
        assert parts[0] == "0"
        assert float(parts[1]) == 0.5


# ============================================================
# FINGERPRINT TESTS
# ============================================================

class TestFingerprint:
    """Test dataset fingerprinting."""

    def test_fingerprint_config_deterministic(self):
        from src.config import ClassConfig
        from src.dataset.convert import fingerprint_config

        config = ClassConfig()
        fp1 = fingerprint_config(config)
        fp2 = fingerprint_config(config)
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_fingerprint_config_changes(self):
        from src.config import ClassConfig
        from src.dataset.convert import fingerprint_config

        config1 = ClassConfig()
        config2 = ClassConfig(target_classes=("car", "truck", "bus"))

        fp1 = fingerprint_config(config1)
        fp2 = fingerprint_config(config2)
        assert fp1 != fp2

    def test_fingerprint_file(self):
        import tempfile
        from src.dataset.convert import fingerprint_file

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write('{"test": "data"}')
            path = Path(f.name)

        try:
            fp1 = fingerprint_file(path)
            fp2 = fingerprint_file(path)
            assert fp1 == fp2
            assert len(fp1) == 16

            # Change content -> different fingerprint
            with open(path, "w") as f:
                f.write('{"test": "changed"}')
            fp3 = fingerprint_file(path)
            assert fp3 != fp1
        finally:
            path.unlink(missing_ok=True)


# ============================================================
# REPORT TESTS
# ============================================================

class TestReport:
    """Test the Report class."""

    def test_report_creation(self):
        from src.log import Report
        report = Report(stage="test", description="test report")
        assert report.data["_meta"]["stage"] == "test"

    def test_add_section(self):
        from src.log import Report
        report = Report(stage="test")
        report.add("stats", {"count": 42})
        assert report.data["stats"]["count"] == 42

    def test_checks_pass(self):
        from src.log import Report
        report = Report(stage="test")
        report.add_check("test1", True, "ok")
        report.add_check("test2", True, "ok")
        assert report.all_checks_passed is True

    def test_checks_fail(self):
        from src.log import Report
        report = Report(stage="test")
        report.add_check("test1", True, "ok")
        report.add_check("test2", False, "bad")
        assert report.all_checks_passed is False

    def test_warning_does_not_fail(self):
        from src.log import Report
        report = Report(stage="test")
        report.add_check("test1", True, "ok")
        report.add_check("warn1", False, "minor", severity="warning")
        assert report.all_checks_passed is True
        assert report.has_warnings is True

    def test_save_and_load(self):
        from src.log import Report
        import tempfile

        report = Report(stage="test")
        report.add("value", 42)
        report.add_check("check1", True, "ok")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = Path(f.name)

        try:
            report.save(path)
            with open(path, "r") as f:
                loaded = json.load(f)
            assert loaded["value"] == 42
            assert loaded["integrity_checks"][0]["status"] == "PASS"
        finally:
            path.unlink(missing_ok=True)


# ============================================================
# DATASET YAML GENERATION TEST
# ============================================================

class TestDatasetYaml:
    """Test that dataset.yaml generation produces valid content."""

    def test_yaml_content(self):
        """Verify dataset.yaml format matches Ultralytics expectations."""
        from src.config import CLASSES

        # Simulate what Phase 2 writes
        yaml_content = (
            f"path: /test/data\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"\n"
            f"nc: {CLASSES.num_classes}\n"
            f"names: {list(CLASSES.target_classes)}\n"
        )

        assert "nc: 5" in yaml_content
        assert "train: train/images" in yaml_content
        assert "val: val/images" in yaml_content
        assert "['car', 'truck', 'bus', 'bike', 'motor']" in yaml_content


# ============================================================
# TRAIN CONFIG TESTS
# ============================================================

class TestTrainConfig:
    """Test training configuration."""

    def test_default_config(self):
        from src.config import TrainConfig
        config = TrainConfig()
        assert config.model == "yolov8m.pt"
        assert config.imgsz == 640
        assert config.seed == 42

    def test_to_ultralytics_args(self):
        from src.config import TrainConfig
        config = TrainConfig()
        args = config.to_ultralytics_args()
        assert args["imgsz"] == 640
        assert args["epochs"] == 50
        assert args["seed"] == 42
        assert "data" not in args  # Must be added separately

    def test_cache_default_disabled(self):
        """cache=False is the only reliable option on Kaggle (ram: 90GB needed;
        disk: ~190GB .npy vs ~20GB /kaggle/working quota)."""
        from src.config import TrainConfig
        config = TrainConfig()
        assert config.cache is False
        assert TrainConfig().to_ultralytics_args()["cache"] is False

    def test_model_progression(self):
        from src.config import MODEL_PROGRESSION
        assert "nano" in MODEL_PROGRESSION
        assert "medium" in MODEL_PROGRESSION
        assert "weights" in MODEL_PROGRESSION["nano"]
        assert "batch_size_t4" in MODEL_PROGRESSION["medium"]
