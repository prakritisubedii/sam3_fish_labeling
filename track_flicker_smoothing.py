"""
Lightweight IoU + Kalman tracker (SORT-style) to test ONE narrow question:
does cross-frame tracking bridge single-frame "flicker" gaps in the
ensemble+WBF output -- a fish detected in frame N and frame N+2, but missing
in frame N+1 (its score dipped just under threshold that one frame)?

This is explicitly NOT about recovering camouflaged/missed fish -- that's the
separate, harder problem documented in KNOWN_LIMITATIONS.md, and tracking
can't fix an image-quality ceiling. This only tests whether a track's
Kalman-predicted position can carry a detection through a brief gap instead
of it vanishing for that frame.

This is a standalone test script -- NOT wired into ensemble_wbf.py or any
production path. Inspect the before/after images this produces and decide
whether it's worth adopting before wiring it in anywhere.

Reads: outputs/ensemble_wbf/detections.json (already computed -- this script
makes no SAM3/GPU calls, it's pure post-processing of existing detections).

Output:
    outputs/tracking_flicker_test/case<n>_frame<k>_before.png
    outputs/tracking_flicker_test/case<n>_frame<k>_after.png
    (a handful of real flicker-gap cases found in the actual data)

Usage: python track_flicker_smoothing.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from experiment_common import OUTPUT_ROOT
from extract_frames import frame_path

DETECTIONS_PATH = OUTPUT_ROOT / "ensemble_wbf" / "detections.json"
RUN_DIR = OUTPUT_ROOT / "tracking_flicker_test"

# Tracker params
IOU_MATCH_THRESHOLD = 0.1   # low on purpose -- these boxes are tiny, IoU is
                             # brittle for small objects, same reasoning as
                             # the center-distance approach in Stage 1
MAX_AGE = 3                  # frames a track survives without a matching detection
NUM_EXAMPLE_CASES_TO_SHOW = 4

# For finding real flicker-gap cases directly in the data (empirical, not synthetic)
GAP_CENTER_DIST_PX = 20


def xyxy_to_cxcywh(box):
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def cxcywh_to_xyxy(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def make_kalman_filter(box_xyxy):
    """State x = [cx, cy, vcx, vcy, w, h]; measurement z = [cx, cy, w, h].
    Constant-velocity model for position, size held (slow drift via Q)."""
    kf = KalmanFilter(dim_x=6, dim_z=4)
    cx, cy, w, h = xyxy_to_cxcywh(box_xyxy)
    kf.x = np.array([[cx], [cy], [0.0], [0.0], [w], [h]])
    kf.F = np.array(
        [
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    kf.H = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    kf.P *= 100.0
    kf.P[2:4, 2:4] *= 10  # extra uncertainty on velocity, unknown at init
    kf.R = np.diag([9, 9, 16, 16]).astype(float)   # measurement noise (px^2)
    kf.Q = np.diag([1, 1, 4, 4, 1, 1]).astype(float)  # process noise
    return kf


class Track:
    _next_id = 0

    def __init__(self, box_xyxy, score):
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = make_kalman_filter(box_xyxy)
        self.score = score
        self.time_since_update = 0
        self.last_update_source = "detection"

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.get_box()

    def update(self, box_xyxy, score, source="detection"):
        cx, cy, w, h = xyxy_to_cxcywh(box_xyxy)
        self.kf.update(np.array([[cx], [cy], [w], [h]]))
        self.time_since_update = 0
        self.score = score
        self.last_update_source = source

    def get_box(self):
        cx, cy, w, h = self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[4, 0], self.kf.x[5, 0]
        return cxcywh_to_xyxy(cx, cy, w, h)


def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    boxes_a = np.array(boxes_a)
    boxes_b = np.array(boxes_b)
    x0 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y0 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x1 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y1 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def associate(track_boxes, det_boxes, iou_threshold):
    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return [], list(range(len(track_boxes))), list(range(len(det_boxes)))
    iou = iou_matrix(track_boxes, det_boxes)
    row_ind, col_ind = linear_sum_assignment(-iou)
    matches, matched_t, matched_d = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_t.add(r)
            matched_d.add(c)
    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_t]
    unmatched_dets = [i for i in range(len(det_boxes)) if i not in matched_d]
    return matches, unmatched_tracks, unmatched_dets


def run_tracker(detections: dict, num_frames: int) -> dict:
    tracks: list[Track] = []
    results_per_frame = {}
    for frame_idx in range(num_frames):
        predicted_boxes = [t.predict() for t in tracks]

        frame_dets = detections.get(str(frame_idx), [])
        det_boxes = [d[:4] for d in frame_dets]
        det_scores = [d[4] for d in frame_dets]

        matches, _, unmatched_dets = associate(predicted_boxes, det_boxes, IOU_MATCH_THRESHOLD)
        for t_idx, d_idx in matches:
            tracks[t_idx].update(det_boxes[d_idx], det_scores[d_idx])
        for d_idx in unmatched_dets:
            tracks.append(Track(det_boxes[d_idx], det_scores[d_idx]))

        tracks = [t for t in tracks if t.time_since_update <= MAX_AGE]

        results_per_frame[frame_idx] = [
            {
                "box": t.get_box(),
                "score": t.score,
                "track_id": t.id,
                "source": "detection" if t.time_since_update == 0 else "predicted",
            }
            for t in tracks
        ]
    return results_per_frame


def find_flicker_gap_cases(detections: dict, num_frames: int):
    """Empirically finds real 1-frame gaps in the raw data: a detection in
    frame N with a spatially-close match in frame N+2, but nothing close in
    frame N+1."""

    def centers(frame_key):
        boxes = detections.get(frame_key, [])
        c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes], dtype=np.float32)
        return c, boxes

    cases = []
    for n in range(num_frames - 2):
        c0, b0 = centers(str(n))
        c1, _ = centers(str(n + 1))
        c2, b2 = centers(str(n + 2))
        if len(c0) == 0 or len(c2) == 0:
            continue
        for i in range(len(c0)):
            d2 = np.sqrt(((c2 - c0[i]) ** 2).sum(axis=1))
            if d2.min() > GAP_CENTER_DIST_PX:
                continue
            if len(c1):
                d1 = np.sqrt(((c1 - c0[i]) ** 2).sum(axis=1))
                if d1.min() <= GAP_CENTER_DIST_PX:
                    continue  # not actually a gap
            cases.append({"frame_n": n, "gap_frame": n + 1, "box_before": b0[i], "box_after": b2[int(d2.argmin())]})
    return cases


def draw_frame(frame_idx, raw_boxes=None, tracked_entry=None):
    raw = cv2.imread(str(frame_path(frame_idx)))
    vis = raw.copy()
    if raw_boxes:
        for b in raw_boxes:
            x0, y0, x1, y1 = [int(v) for v in b[:4]]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(vis, f"{b[4]:.2f}", (x0, max(y0 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if tracked_entry:
        for entry in tracked_entry:
            x0, y0, x1, y1 = [int(v) for v in entry["box"]]
            color = (0, 255, 0) if entry["source"] == "detection" else (0, 128, 255)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            label = f"{entry['score']:.2f} ({entry['source']})"
            cv2.putText(vis, label, (x0, max(y0 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def main():
    data = json.load(open(DETECTIONS_PATH))
    detections = data["detections"]
    num_frames = len(detections)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    gap_cases = find_flicker_gap_cases(detections, num_frames)
    print(f"Found {len(gap_cases)} real 1-frame flicker-gap cases in the raw ensemble+WBF output")

    tracked = run_tracker(detections, num_frames)

    # cross-check: for each gap case, did the tracker produce a *predicted*
    # box near the expected gap location on the gap frame?
    bridged = 0
    for case in gap_cases:
        gap_frame = case["gap_frame"]
        expected_cx = (case["box_before"][0] + case["box_before"][2]) / 2
        expected_cy = (case["box_before"][1] + case["box_before"][3]) / 2
        predicted_entries = [e for e in tracked[gap_frame] if e["source"] == "predicted"]
        for e in predicted_entries:
            cx = (e["box"][0] + e["box"][2]) / 2
            cy = (e["box"][1] + e["box"][3]) / 2
            if np.hypot(cx - expected_cx, cy - expected_cy) <= GAP_CENTER_DIST_PX:
                bridged += 1
                break
    print(f"Tracker bridged {bridged}/{len(gap_cases)} of those gaps with a predicted box")

    # render before/after for a handful of concrete example cases
    shown = 0
    for case in gap_cases:
        gap_frame = case["gap_frame"]
        predicted_entries = [e for e in tracked[gap_frame] if e["source"] == "predicted"]
        expected_cx = (case["box_before"][0] + case["box_before"][2]) / 2
        expected_cy = (case["box_before"][1] + case["box_before"][3]) / 2
        match = None
        for e in predicted_entries:
            cx = (e["box"][0] + e["box"][2]) / 2
            cy = (e["box"][1] + e["box"][3]) / 2
            if np.hypot(cx - expected_cx, cy - expected_cy) <= GAP_CENTER_DIST_PX:
                match = e
                break
        if match is None:
            continue  # only show cases where bridging actually worked

        raw_boxes_gap_frame = detections.get(str(gap_frame), [])
        before = draw_frame(gap_frame, raw_boxes=raw_boxes_gap_frame)
        after = draw_frame(gap_frame, tracked_entry=tracked[gap_frame])
        cv2.imwrite(str(RUN_DIR / f"case{shown}_frame{gap_frame}_before.png"), before)
        cv2.imwrite(str(RUN_DIR / f"case{shown}_frame{gap_frame}_after.png"), after)
        print(f"  case {shown}: gap at frame {gap_frame}, predicted box near "
              f"({expected_cx:.0f},{expected_cy:.0f}) -- wrote before/after images")
        shown += 1
        if shown >= NUM_EXAMPLE_CASES_TO_SHOW:
            break

    print(f"\nWrote {shown} before/after example pairs to {RUN_DIR}")


if __name__ == "__main__":
    main()
