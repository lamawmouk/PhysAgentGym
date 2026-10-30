"""Trajectory factories: synthetic for tests, OpenCV CSRT for real videos.

Coordinate convention: physical y-up, x-right. Positions are normalized by frame
height so the trajectory is dimensionless and scale-invariant. Time is in seconds.
"""
from pathlib import Path
import numpy as np
from .types import Trajectory


# ---------- synthetic factories (kept for tests) ----------

def synthetic_vertical_rise(duration: float = 2.0, fps: float = 30,
                             v0: float = 30.0, noise: float = 0.0) -> Trajectory:
    n = int(duration * fps)
    t = np.linspace(0, duration, n)
    y = v0 * t - 0.5 * 9.81 * t * t
    x = np.zeros_like(t) + noise * np.random.randn(n)
    pos = np.stack([x, y], axis=1)
    return Trajectory(t=t, pos=pos, confidence=np.ones_like(t),
                      object_id="rocket", fps=fps)


def synthetic_free_fall(duration: float = 1.5, fps: float = 30,
                         h0: float = 12.0, noise: float = 0.0) -> Trajectory:
    n = int(duration * fps)
    t = np.linspace(0, duration, n)
    y = h0 - 0.5 * 9.81 * t * t
    x = np.zeros_like(t) + noise * np.random.randn(n)
    pos = np.stack([x, y], axis=1)
    return Trajectory(t=t, pos=pos, confidence=np.ones_like(t),
                      object_id="ball", fps=fps)


def synthetic_horizontal_drift(duration: float = 2.0, fps: float = 30,
                                vx: float = 5.0, vy_drift: float = 2.0) -> Trajectory:
    n = int(duration * fps)
    t = np.linspace(0, duration, n)
    pos = np.stack([vx * t, vy_drift * t], axis=1)
    return Trajectory(t=t, pos=pos, confidence=np.ones_like(t), object_id="thing", fps=fps)


def synthetic_perpetual_motion(duration: float = 3.0, fps: float = 30) -> Trajectory:
    n = int(duration * fps)
    t = np.linspace(0, duration, n)
    omega = 2.0
    r = 1.0 + 0.5 * t
    pos = np.stack([r * np.cos(omega * t), r * np.sin(omega * t)], axis=1)
    return Trajectory(t=t, pos=pos, confidence=np.ones_like(t), object_id="impossible", fps=fps)


# ---------- real video → trajectory ----------

def _import_cv2():
    import cv2
    return cv2


def read_video(path: str | Path) -> tuple[np.ndarray, float]:
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames), fps


def _frame_diff_seed(frames: np.ndarray, motion_thresh: int = 18,
                      blur_ksize: int = 7, dilate_ksize: int = 9,
                      max_area_frac: float = 0.6) -> tuple[int, int, int, int] | None:
    """Largest contour in the diff between frame 0 and frame N//4. Returns None if the
    region covers more than max_area_frac of the frame (pan/zoom catastrophe)."""
    cv2 = _import_cv2()
    if frames.shape[0] < 4:
        raise ValueError("need at least 4 frames to seed bbox")
    a = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(frames[len(frames) // 4], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(a, b)
    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    _, thr = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)
    thr = cv2.dilate(thr, np.ones((dilate_ksize, dilate_ksize), np.uint8))
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    H, W = frames.shape[1], frames.shape[2]
    if (w * h) / (H * W) > max_area_frac:
        return None
    return int(x), int(y), int(w), int(h)


_YOLO_MODEL = None


def _yolo_world_seed(frame: np.ndarray, prompt_objects: list[str],
                      conf_thresh: float = 0.05,
                      max_area_frac: float = 0.55) -> tuple[int, int, int, int] | None:
    global _YOLO_MODEL
    try:
        from ultralytics import YOLO
        if _YOLO_MODEL is None:
            _YOLO_MODEL = YOLO("yolov8s-worldv2.pt")
        _YOLO_MODEL.set_classes(prompt_objects)
        res = _YOLO_MODEL.predict(frame, conf=conf_thresh, verbose=False)
    except Exception:
        return None
    if not res or len(res[0].boxes) == 0:
        return None
    H, W = frame.shape[:2]
    boxes = res[0].boxes
    confs = boxes.conf.cpu().numpy()
    xyxys = boxes.xyxy.cpu().numpy()
    # filter out full-frame detections
    keep = []
    for i, (x1, y1, x2, y2) in enumerate(xyxys):
        area = (x2 - x1) * (y2 - y1)
        if area / (H * W) <= max_area_frac:
            keep.append(i)
    if not keep:
        return None
    best = keep[int(np.argmax(confs[keep]))]
    x1, y1, x2, y2 = xyxys[best]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(W, int(x2)); y2 = min(H, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _center_crop_seed(frames: np.ndarray, frac: float = 0.4) -> tuple[int, int, int, int]:
    H, W = frames.shape[1], frames.shape[2]
    w, h = int(W * frac), int(H * frac)
    return (W // 2 - w // 2, H // 2 - h // 2, w, h)


class PerceptionFailure(RuntimeError):
    """No reliable seed bbox could be found — caller should skip the video."""


def find_initial_bbox(frames: np.ndarray,
                       prompt_objects: list[str] | None = None,
                       strict: bool = True) -> tuple[tuple[int, int, int, int], str]:
    """Cascade: YOLO-world → strict frame-diff → loose frame-diff → center crop.
    Returns (bbox, source) where source ∈ {yolo_world, frame_diff_strict,
    frame_diff_loose, center_crop}. If strict=True and only weaker fallbacks
    fired, raises PerceptionFailure so the orchestrator can record and skip."""
    if prompt_objects:
        for frame_idx in [0, len(frames) // 4, len(frames) // 2]:
            bbox = _yolo_world_seed(frames[frame_idx], prompt_objects)
            if bbox is not None:
                return bbox, "yolo_world"
    bbox = _frame_diff_seed(frames, max_area_frac=0.6)
    if bbox is not None:
        return bbox, "frame_diff_strict"
    if strict:
        raise PerceptionFailure("YOLO-world and strict frame-diff both failed")
    bbox = _frame_diff_seed(frames, max_area_frac=1.01)
    if bbox is not None:
        return bbox, "frame_diff_loose"
    return _center_crop_seed(frames), "center_crop"


def estimate_camera_motion(frames: np.ndarray,
                            max_features: int = 600) -> list[np.ndarray]:
    """Frame-to-frame homographies: H[i] maps coords in frame[i+1] to frame[i].
    Uses goodFeaturesToTrack + LK optical flow + RANSAC homography fit. Outlier
    features (the moving subject) are rejected automatically by RANSAC.
    Returns list of length N-1."""
    cv2 = _import_cv2()
    homs: list[np.ndarray] = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_features,
                                      qualityLevel=0.01, minDistance=8)
        if p0 is None or len(p0) < 8:
            homs.append(np.eye(3))
            prev_gray = curr_gray
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        if p1 is None:
            homs.append(np.eye(3))
            prev_gray = curr_gray
            continue
        good_p0 = p0[st.flatten() == 1]
        good_p1 = p1[st.flatten() == 1]
        if len(good_p0) < 8:
            homs.append(np.eye(3))
            prev_gray = curr_gray
            continue
        H, _mask = cv2.findHomography(good_p1, good_p0, cv2.RANSAC, 3.0)
        homs.append(H if H is not None else np.eye(3))
        prev_gray = curr_gray
    return homs


def compensate_for_camera_motion(centers_px: np.ndarray,
                                  homographies: list[np.ndarray],
                                  max_norm: float = 50.0) -> np.ndarray:
    """Reproject each frame's tracked center back into frame[0]'s coordinate system.
    Disables compensation from frame t onward if the cumulative homography blows up
    (a degenerate RANSAC fit on textureless / repetitive scenes)."""
    n = len(centers_px)
    out = np.zeros_like(centers_px)
    cumulative = np.eye(3)
    out[0] = centers_px[0]
    blew_up = False
    for t in range(1, n):
        if not blew_up:
            cumulative = cumulative @ homographies[t - 1]
            if not np.isfinite(cumulative).all() or np.linalg.norm(cumulative) > max_norm:
                blew_up = True
                cumulative = np.eye(3)
        if blew_up:
            out[t] = centers_px[t]
            continue
        v = np.array([centers_px[t, 0], centers_px[t, 1], 1.0])
        w = cumulative @ v
        if abs(w[2]) < 1e-6:
            blew_up = True
            out[t] = centers_px[t]
            continue
        out[t] = w[:2] / w[2]
    return out


def _make_csrt_tracker():
    cv2 = _import_cv2()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("no CSRT tracker — install opencv-contrib-python")


def track_csrt(frames: np.ndarray,
               bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    tracker = _make_csrt_tracker()
    tracker.init(frames[0], tuple(bbox))
    centers = np.zeros((len(frames), 2), dtype=np.float64)
    confidence = np.zeros(len(frames), dtype=np.float64)
    last_cx = bbox[0] + bbox[2] / 2
    last_cy = bbox[1] + bbox[3] / 2
    for i, frame in enumerate(frames):
        ok, b = tracker.update(frame)
        if ok:
            cx = b[0] + b[2] / 2
            cy = b[1] + b[3] / 2
            centers[i] = (cx, cy)
            confidence[i] = 1.0
            last_cx, last_cy = cx, cy
        else:
            centers[i] = (last_cx, last_cy)
            confidence[i] = 0.0
    return centers, confidence


def from_video(video_path: str | Path,
               object_id: str = "object",
               prompt_objects: list[str] | None = None,
               initial_bbox: tuple[int, int, int, int] | None = None,
               compensate_camera: bool = True,
               strict: bool = True) -> Trajectory:
    frames, fps = read_video(video_path)
    H = float(frames.shape[1])
    if initial_bbox is not None:
        bbox, seed_source = initial_bbox, "manual"
    else:
        bbox, seed_source = find_initial_bbox(frames, prompt_objects=prompt_objects, strict=strict)
    print(f"   perception: seed={seed_source}  bbox={bbox}")
    centers_px, confidence = track_csrt(frames, bbox)
    if compensate_camera and len(frames) > 1:
        homographies = estimate_camera_motion(frames)
        centers_px = compensate_for_camera_motion(centers_px, homographies)
    t = np.arange(len(frames)) / fps
    x_norm = centers_px[:, 0] / H
    y_norm = (H - centers_px[:, 1]) / H
    pos = np.stack([x_norm, y_norm], axis=1)
    return Trajectory(t=t, pos=pos, confidence=confidence,
                      object_id=object_id, fps=fps)
