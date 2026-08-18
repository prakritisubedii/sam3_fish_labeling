"""
Tests a new idea (user's, not previously tried): rather than gating candidate
DETECTIONS by motion (already tried twice -- see
project_sam2_motion_gated_reseed.md's fixed-camera/median-background gate,
and run_experiments_enhanced.py's motion-seeded points), use size/shape/
contrast variance over each TRACK's own lifetime as a post-hoc verifier:
a real fish should show real box-size/aspect-ratio/local-contrast change as
it swims, turns, and changes distance from the camera; a false positive
sitting on static coral/rock texture should look almost identical frame to
frame since nothing there is actually moving.

Pure CPU post-processing -- no GPU, no new SAM3 calls. Reuses the Kalman/IoU
tracker from track_flicker_smoothing.py (run_tracker) on the already-computed
outputs/ensemble_wbf/detections.json to get full 299-frame (no OOM
truncation -- this pipeline never touches the GPU) per-track box histories,
then for each track with enough frames computes:
  - box area coefficient of variation (CV = std/mean) over its lifetime
  - aspect ratio (w/h) CV
  - local image-patch contrast (std of pixel intensities under the box) CV,
    sampled from the actual extracted frames

Hypothesis: low CV across all three = static/coral false positive.
High CV = real, moving fish. Ranks tracks by a combined stability score and
writes crops of the most static-looking long-lived tracks (top false-positive
candidates) and the most variable ones (clear real-fish examples) so this can
be checked visually against the real frames rather than trusted on numbers
alone.

Output:
    outputs/diagnostics/temporal_stability/stability_ranking.json
    outputs/diagnostics/temporal_stability/static_candidate_<rank>_track<id>_frame<n>.png
    outputs/diagnostics/temporal_stability/variable_example_<rank>_track<id>_frame<n>.png

Usage: python diagnostics/temporal_stability_filter.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from experiment_common import OUTPUT_ROOT
from extract_frames import frame_path
from track_flicker_smoothing import DETECTIONS_PATH, run_tracker

RUN_DIR = OUTPUT_ROOT / "diagnostics" / "temporal_stability"
MIN_TRACK_LEN = 15  # need enough frames for variance to be meaningful
NUM_EXAMPLES_EACH_SIDE = 5


def patch_contrast(frame_gray, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, frame_gray.shape[1]), min(y1, frame_gray.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = frame_gray[y0:y1, x0:x1]
    return float(patch.std())


def coeff_of_variation(values):
    values = np.array(values, dtype=np.float64)
    mean = values.mean()
    if mean <= 1e-6:
        return 0.0
    return float(values.std() / mean)


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    data = json.load(open(DETECTIONS_PATH))
    detections = data["detections"]
    num_frames = len(detections)
    print(f"Running Kalman/IoU tracker on {num_frames} frames of ensemble_wbf detections...")
    tracked = run_tracker(detections, num_frames)

    # group by track_id across all frames
    track_frames = {}
    for frame_idx in range(num_frames):
        for entry in tracked[frame_idx]:
            track_frames.setdefault(entry["track_id"], []).append((frame_idx, entry["box"]))

    print(f"Total tracks: {len(track_frames)}")
    long_tracks = {tid: frames for tid, frames in track_frames.items() if len(frames) >= MIN_TRACK_LEN}
    print(f"Tracks with >= {MIN_TRACK_LEN} frames: {len(long_tracks)}")

    # cache grayscale frames as we touch them (only long tracks' frames needed)
    gray_cache = {}

    def get_gray(frame_idx):
        if frame_idx not in gray_cache:
            img = cv2.imread(str(frame_path(frame_idx)))
            gray_cache[frame_idx] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_cache[frame_idx]

    results = []
    for tid, frames in long_tracks.items():
        areas, aspects, contrasts = [], [], []
        for frame_idx, box in frames:
            x0, y0, x1, y1 = box
            w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
            areas.append(w * h)
            aspects.append(w / h)
            contrasts.append(patch_contrast(get_gray(frame_idx), box))

        area_cv = coeff_of_variation(areas)
        aspect_cv = coeff_of_variation(aspects)
        contrast_cv = coeff_of_variation(contrasts)
        combined = area_cv + aspect_cv + contrast_cv

        results.append({
            "track_id": tid,
            "num_frames": len(frames),
            "first_frame": frames[0][0],
            "last_frame": frames[-1][0],
            "area_cv": round(area_cv, 4),
            "aspect_cv": round(aspect_cv, 4),
            "contrast_cv": round(contrast_cv, 4),
            "combined_stability_score": round(combined, 4),
        })

    results.sort(key=lambda r: r["combined_stability_score"])  # most static first

    with open(RUN_DIR / "stability_ranking.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Most STATIC tracks (low size/shape/contrast variance -- false-positive candidates) ===")
    for r in results[:NUM_EXAMPLES_EACH_SIDE]:
        print(f"  track {r['track_id']}: {r['num_frames']} frames, "
              f"area_cv={r['area_cv']} aspect_cv={r['aspect_cv']} contrast_cv={r['contrast_cv']} "
              f"combined={r['combined_stability_score']}")

    print(f"\n=== Most VARIABLE tracks (high variance -- likely real, moving fish) ===")
    for r in results[-NUM_EXAMPLES_EACH_SIDE:]:
        print(f"  track {r['track_id']}: {r['num_frames']} frames, "
              f"area_cv={r['area_cv']} aspect_cv={r['aspect_cv']} contrast_cv={r['contrast_cv']} "
              f"combined={r['combined_stability_score']}")

    # write visual crops: first/mid/last frame for the top static candidates and top variable examples
    def write_crops(tid, frames, tag, rank):
        n = len(frames)
        sample_idxs = sorted(set([0, n // 2, n - 1]))
        for si in sample_idxs:
            frame_idx, box = frames[si]
            img = cv2.imread(str(frame_path(frame_idx)))
            x0, y0, x1, y1 = [int(v) for v in box]
            pad = 20
            x0p, y0p = max(x0 - pad, 0), max(y0 - pad, 0)
            x1p, y1p = min(x1 + pad, img.shape[1]), min(y1 + pad, img.shape[0])
            crop = img[y0p:y1p, x0p:x1p].copy()
            cv2.rectangle(crop, (x0 - x0p, y0 - y0p), (x1 - x0p, y1 - y0p), (0, 255, 0), 1)
            crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(RUN_DIR / f"{tag}_{rank}_track{tid}_frame{frame_idx}.png"), crop)

    for rank, r in enumerate(results[:NUM_EXAMPLES_EACH_SIDE]):
        write_crops(r["track_id"], long_tracks[r["track_id"]], "static_candidate", rank)
    for rank, r in enumerate(results[-NUM_EXAMPLES_EACH_SIDE:]):
        write_crops(r["track_id"], long_tracks[r["track_id"]], "variable_example", rank)

    print(f"\nWrote crops and {RUN_DIR / 'stability_ranking.json'} to {RUN_DIR}")


if __name__ == "__main__":
    main()
