"""
Renders the temporal-stability filter (temporal_stability_filter.py /
temporal_stability_deep_dive.py) as an actual video across the FULL 299-frame
clip, not just static before/after crops of a handful of example tracks.

Colors each track's box by the refined rule the deep-dive spot-check
justified (plain score alone wasn't reliable -- e.g. track 2 had a low score
but visually looked like a real fish holding still briefly, while track 309
scored just above the naive 0.2 cutoff yet still looked static 96 frames
apart):
    RED    = low variance AND long-lived (>=50 frames) -- strong false-
             positive signal, confirmed on track 208 (208 frames, a static
             rock that was earlier mistakenly reported as our best-tracked
             fish)
    ORANGE = low variance but short-lived (<50 frames) -- ambiguous, could
             be a real fish that simply held still for a moment (track 2
             looked like this)
    GREEN  = everything else -- real size/shape/contrast change over time,
             consistent with a real, moving fish

Pure CPU: reuses run_tracker() on the already-computed
outputs/ensemble_wbf/detections.json and the already-extracted frame PNGs,
no GPU/SAM3 calls, so this covers the full clip with no OOM risk (unlike the
native-tracker videos, which are capped at ~246-260/299 frames by shared GPU
memory).

Output:
    outputs/diagnostics/temporal_stability/stability_video.mp4

Usage: python diagnostics/temporal_stability_video.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from experiment_common import OUTPUT_ROOT
from extract_frames import frame_path
from track_flicker_smoothing import DETECTIONS_PATH, run_tracker

RUN_DIR = OUTPUT_ROOT / "diagnostics" / "temporal_stability"
VIDEO_PATH = "assets/fish_data/clip1.mp4"
MIN_TRACK_LEN = 15  # below this, not enough data for a meaningful score -- always green
LOW_SCORE_THRESH = 0.25  # slightly above the original 0.2 eyeball cutoff, per
                          # the deep-dive finding that track 309 (0.2021) still
                          # looked static -- pulls it into the flagged zone
LONG_LIVED_THRESH = 50  # frames

RED = (0, 0, 255)
ORANGE = (0, 140, 255)
GREEN = (0, 255, 0)


def patch_contrast(frame_gray, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, frame_gray.shape[1]), min(y1, frame_gray.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(frame_gray[y0:y1, x0:x1].std())


def coeff_of_variation(values):
    values = np.array(values, dtype=np.float64)
    mean = values.mean()
    return float(values.std() / mean) if mean > 1e-6 else 0.0


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    data = json.load(open(DETECTIONS_PATH))
    detections = data["detections"]
    num_frames = len(detections)
    print(f"Running Kalman/IoU tracker on {num_frames} frames...")
    tracked = run_tracker(detections, num_frames)

    track_frames = {}
    for frame_idx in range(num_frames):
        for entry in tracked[frame_idx]:
            track_frames.setdefault(entry["track_id"], []).append((frame_idx, entry["box"]))

    gray_cache = {}

    def get_gray(frame_idx):
        if frame_idx not in gray_cache:
            img = cv2.imread(str(frame_path(frame_idx)))
            gray_cache[frame_idx] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_cache[frame_idx]

    print("Scoring tracks...")
    track_color = {}
    track_score = {}
    for tid, frames in track_frames.items():
        if len(frames) < MIN_TRACK_LEN:
            track_color[tid] = GREEN
            track_score[tid] = None
            continue
        areas, aspects, contrasts = [], [], []
        for frame_idx, box in frames:
            x0, y0, x1, y1 = box
            w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
            areas.append(w * h)
            aspects.append(w / h)
            contrasts.append(patch_contrast(get_gray(frame_idx), box))
        score = coeff_of_variation(areas) + coeff_of_variation(aspects) + coeff_of_variation(contrasts)
        track_score[tid] = score
        if score < LOW_SCORE_THRESH and len(frames) >= LONG_LIVED_THRESH:
            track_color[tid] = RED
        elif score < LOW_SCORE_THRESH:
            track_color[tid] = ORANGE
        else:
            track_color[tid] = GREEN

    n_red = sum(1 for c in track_color.values() if c == RED)
    n_orange = sum(1 for c in track_color.values() if c == ORANGE)
    n_green = sum(1 for c in track_color.values() if c == GREEN)
    print(f"RED (strong false-positive signal): {n_red}")
    print(f"ORANGE (ambiguous, short + low-variance): {n_orange}")
    print(f"GREEN (looks like real fish, or too short to score): {n_green}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_path = RUN_DIR / "stability_video.mp4"
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("Rendering full-clip video...")
    for frame_idx in range(num_frames):
        frame = cv2.imread(str(frame_path(frame_idx)))
        for entry in tracked[frame_idx]:
            tid = entry["track_id"]
            color = track_color[tid]
            x0, y0, x1, y1 = [int(v) for v in entry["box"]]
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            score = track_score[tid]
            label = f"id{tid}" if score is None else f"id{tid} {score:.2f}"
            cv2.putText(frame, label, (x0, max(y0 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        ffmpeg_proc.stdin.write(frame.tobytes())
        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}/{num_frames}")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
