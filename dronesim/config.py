"""Configurable (assumed, not measured) parameters shared across the pipeline."""
from dataclasses import dataclass, field


@dataclass
class VisionConfig:
    model: str = "yolov8s.pt"
    tracker: str = "bytetrack.yaml"
    # Lower than the HUD's 0.6: the obstacle's confidence is low at long range.
    conf: float = 0.3
    # Empty = any class (the COCO label of a plain box is arbitrary, e.g. "refrigerator").
    allowed_classes: tuple = ()
    # Window of recent detections used to fit growth rate and time-to-contact.
    fit_window_s: float = 0.75
    min_fit_samples: int = 5
    # How long to keep extrapolating TTC after the detector loses the obstacle.
    max_gap_s: float = 2.0
    # A fitted TTC beyond this is treated as 'not approaching' (near-zero slope = noise).
    max_ttc_s: float = 30.0


@dataclass
class ObstacleConfig:
    """Real obstacle size. Pixels are not metres: this must be supplied, not inferred."""
    width_m: float = 0.5
    height_m: float = 0.5
    thickness_m: float = 0.05     # panel depth along the approach axis (assumed)
    size_is_assumed: bool = True


@dataclass
class CameraConfig:
    """Pinhole camera. Tello's real FOV is not measured here; 82.6 deg is a nominal value."""
    hfov_deg: float = 82.6
    fov_is_assumed: bool = True


@dataclass
class PipelineConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
