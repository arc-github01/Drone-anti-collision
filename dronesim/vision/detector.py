"""YOLO + tracker wrapper. The only module here that needs ultralytics/OpenCV."""
import os
from dataclasses import dataclass
from typing import Iterator

import cv2

from dronesim.config import VisionConfig


@dataclass
class Detection:
    frame: int
    t: float
    track_id: int
    cls_name: str
    conf: float
    bbox: tuple


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    info = dict(path=path, fps=cap.get(cv2.CAP_PROP_FPS), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def track_video(path: str, cfg: VisionConfig, fps: float) -> Iterator[list[Detection]]:
    """Yield the detections of each frame (possibly empty). t = frame_index / fps."""
    from ultralytics import YOLO  # imported lazily so the pure modules stay light

    path = os.path.abspath(path)  # YOLO can change the working directory
    model = YOLO(cfg.model)
    for i, r in enumerate(model.track(path, stream=True, verbose=False, conf=cfg.conf,
                                      persist=True, tracker=cfg.tracker)):
        dets = []
        if r.boxes.id is not None:
            for xy, tid, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(),
                                     r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                name = model.names[int(c)]
                if cfg.allowed_classes and name not in cfg.allowed_classes:
                    continue
                dets.append(Detection(i, i / fps, int(tid), name, float(p), tuple(xy)))
        yield dets
