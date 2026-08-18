"""
Two techniques to recover fish that camouflage against the coral/background,
building on the best prompt found in run_experiments.py (prompt="small fish", thresh=0.3):

1. CLAHE: white-balance + contrast-limited histogram equalization on every frame
   before it ever reaches SAM3, to fight underwater color cast / low contrast.
2. Motion-seeded points: fish move, coral doesn't. We diff frames to find moving
   blobs the text prompt missed, and add each as a new point-prompted object on
   top of the text-prompt detections (uses SAM3's built-in "refine with points
   after propagation" flow, so it doesn't touch what text-prompting already found).

Each run's results go into its own named folder under outputs/, same convention
as run_experiments.py:
    outputs/thresh0.3_prompt-small-fish_clahe/output.mp4
    outputs/thresh0.3_prompt-small-fish_motion/output.mp4
    outputs/thresh0.3_prompt-small-fish_clahe-motion/output.mp4

Usage: python run_experiments_enhanced.py
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from sam3.model_builder import build_sam3_video_predictor

import sam3_predictor_patch  # noqa: F401 -- applies threshold-routing patch before building the predictor
from experiment_common import OUTPUT_ROOT, render_video, summarize

VIDEO_PATH = "assets/fish_data/clip1.mp4"
PREPROCESSED_DIR = OUTPUT_ROOT / "_preprocessed"

PROMPT = "small fish"
THRESHOLD = 0.3

# motion-blob detection knobs
MOTION_WINDOW_FRAMES = 30      # frames accumulated to build the motion map
MOTION_MIN_AREA_FRAC = 0.0003  # ~460 px^2 on a 1920x1280 frame; drops noise
MOTION_MAX_AREA_FRAC = 0.02    # ~49000 px^2; drops big lighting-flicker blobs
MOTION_TOP_K = 6                # cap on how many new point-prompted objects to add
# Each point-prompted object spawns its own full tracker memory state across
# every frame of the video (much heavier than a text-detected object, which
# only adds a mask entry to the shared detector pass). ~50 combined objects
# OOM'd a 47 GiB A6000 on this video, so keep this small.


# ── 1) CLAHE / underwater color-correction preprocessing ──────────────────

def gray_world_white_balance(frame_bgr: np.ndarray) -> np.ndarray:
    """Corrects the blue/green color cast typical of underwater footage."""
    result = frame_bgr.astype(np.float32)
    avg_b, avg_g, avg_r = (result[:, :, i].mean() for i in range(3))
    avg_gray = (avg_b + avg_g + avg_r) / 3
    result[:, :, 0] *= avg_gray / max(avg_b, 1e-6)
    result[:, :, 1] *= avg_gray / max(avg_g, 1e-6)
    result[:, :, 2] *= avg_gray / max(avg_r, 1e-6)
    return np.clip(result, 0, 255).astype(np.uint8)


def clahe_enhance(frame_bgr: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    """Boosts local contrast on the L channel only, so colors aren't distorted."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def build_enhanced_video(video_path: str, output_path: Path):
    if output_path.exists():
        print(f"Reusing existing enhanced video at {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = gray_world_white_balance(frame)
        frame = clahe_enhance(frame, clahe)
        ffmpeg_proc.stdin.write(frame.tobytes())
    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    print(f"Wrote enhanced video to {output_path}")


# ── 2) Motion-blob detection → point prompts ───────────────────────────────

def compute_motion_points(video_path: str):
    """Accumulates frame-to-frame motion over the first MOTION_WINDOW_FRAMES
    frames and returns normalized (cx, cy) centroids of moving blobs, largest
    first, capped to MOTION_TOP_K."""
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    accum = np.zeros((h, w), dtype=np.float32)

    for _ in range(MOTION_WINDOW_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        accum += cv2.absdiff(gray, prev_gray).astype(np.float32)
        prev_gray = gray
    cap.release()

    accum_norm = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    accum_blur = cv2.GaussianBlur(accum_norm, (9, 9), 0)
    _, motion_mask = cv2.threshold(
        accum_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        motion_mask, connectivity=8
    )
    min_area = MOTION_MIN_AREA_FRAC * h * w
    max_area = MOTION_MAX_AREA_FRAC * h * w
    blobs = []
    for i in range(1, num_labels):  # label 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            blobs.append((area, cx / w, cy / h))
    blobs.sort(key=lambda b: -b[0])
    return [(cx, cy) for _, cx, cy in blobs[:MOTION_TOP_K]]


def filter_points_outside_masks(points_norm, masks, h, w):
    """Drops motion points that already fall inside a text-prompt detection,
    so we only add NEW objects instead of duplicating existing ones."""
    masks_2d = [m if m.ndim == 2 else m[..., 0] for m in masks]
    kept = []
    for cx, cy in points_norm:
        px = min(max(int(cx * w), 0), w - 1)
        py = min(max(int(cy * h), 0), h - 1)
        if not any(m[py, px] for m in masks_2d):
            kept.append((cx, cy))
    return kept


# ── run helpers ──────────────────────────────────────────────────────────

def run_text_prompt_baseline(predictor, video_path, prompt, threshold):
    """Starts a session, runs the text prompt + full-video propagation, and
    returns (session_id, outputs_per_frame, frame0_detections). Session is
    left open so callers can add point prompts on top."""
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=video_path)
    )
    session_id = response["session_id"]

    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
            output_prob_thresh=threshold,
        )
    )
    frame0_detections = len(response["outputs"]["out_obj_ids"])

    outputs_per_frame = {}
    for out in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            output_prob_thresh=threshold,
        )
    ):
        outputs_per_frame[out["frame_index"]] = out["outputs"]

    return session_id, outputs_per_frame, frame0_detections


def add_motion_points_and_repropagate(predictor, session_id, motion_points, existing_ids):
    """Adds each motion-blob centroid as a new point-prompted object, then
    re-propagates once. SAM3 merges the new tracker-driven objects with the
    already-cached text-prompt detections automatically."""
    next_id = (max(existing_ids) + 1) if len(existing_ids) > 0 else 0
    added_ids = []
    for i, (cx, cy) in enumerate(motion_points):
        obj_id = next_id + i
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                points=[[cx, cy]],
                point_labels=[1],
                obj_id=obj_id,
                rel_coordinates=True,
            )
        )
        added_ids.append(obj_id)

    outputs_per_frame = {}
    for out in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[out["frame_index"]] = out["outputs"]

    return outputs_per_frame, added_ids


def save_run(run_name, video_path, outputs_per_frame, elapsed, extra_info=None):
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    render_video(video_path, outputs_per_frame, run_dir / "output.mp4")

    summary = summarize(outputs_per_frame)
    run_info = {
        "run_name": run_name,
        "prompt": PROMPT,
        "output_prob_thresh": THRESHOLD,
        "video_path": video_path,
        "elapsed_seconds": round(elapsed, 1),
        **(extra_info or {}),
        **summary,
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(
        f"[{run_name}] unique objects: {summary['num_unique_objects']} "
        f"| avg detections/frame: {summary['avg_detections_per_frame']:.2f} "
        f"| avg confidence: {summary['avg_confidence']:.3f} ({elapsed:.1f}s)"
    )
    return run_info


def run_clahe(predictor, enhanced_video_path):
    t0 = time.time()
    session_id, outputs_per_frame, frame0_det = run_text_prompt_baseline(
        predictor, str(enhanced_video_path), PROMPT, THRESHOLD
    )
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    elapsed = time.time() - t0
    return save_run(
        "thresh0.3_prompt-small-fish_clahe",
        str(enhanced_video_path),
        outputs_per_frame,
        elapsed,
        extra_info={"technique": "clahe", "frame0_detections": frame0_det},
    )


def run_motion(predictor, video_path, run_name, technique):
    t0 = time.time()
    session_id, outputs_per_frame, frame0_det = run_text_prompt_baseline(
        predictor, video_path, PROMPT, THRESHOLD
    )
    frame0_masks = list(outputs_per_frame[0]["out_binary_masks"])
    h, w = frame0_masks[0].shape[:2] if frame0_masks else (0, 0)
    motion_points = compute_motion_points(video_path)
    if frame0_masks:
        motion_points = filter_points_outside_masks(motion_points, frame0_masks, h, w)
    existing_ids = outputs_per_frame[0]["out_obj_ids"]
    outputs_per_frame, added_ids = add_motion_points_and_repropagate(
        predictor, session_id, motion_points, existing_ids
    )
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    elapsed = time.time() - t0
    return save_run(
        run_name,
        video_path,
        outputs_per_frame,
        elapsed,
        extra_info={
            "technique": technique,
            "frame0_detections": frame0_det,
            "num_motion_points_added": len(added_ids),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which", choices=["clahe", "motion", "clahe_motion"], required=True,
        help="Run a single experiment per process invocation, so GPU memory is "
             "fully released between the heavier multi-object runs (see the "
             "documented allocator-leak notes in sam3_base_predictor.close_session).",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(exist_ok=True)
    enhanced_video_path = PREPROCESSED_DIR / "clip1_clahe.mp4"
    if args.which in ("clahe", "clahe_motion"):
        build_enhanced_video(VIDEO_PATH, enhanced_video_path)

    predictor = build_sam3_video_predictor(gpus_to_use=[0])

    if args.which == "clahe":
        run_clahe(predictor, enhanced_video_path)
    elif args.which == "motion":
        run_motion(predictor, VIDEO_PATH, "thresh0.3_prompt-small-fish_motion", "motion_points")
    elif args.which == "clahe_motion":
        run_motion(
            predictor,
            str(enhanced_video_path),
            "thresh0.3_prompt-small-fish_clahe-motion",
            "clahe+motion_points",
        )

    predictor.shutdown()


if __name__ == "__main__":
    main()
