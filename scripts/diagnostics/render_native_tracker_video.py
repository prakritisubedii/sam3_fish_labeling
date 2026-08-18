"""
Renders an actual video from SAM3's NATIVE video-tracking session (the same
predictor as ../test_sam3_video.py and track_lifespans.py -- a real learned
tracker with per-object visual memory, NOT the independent-per-frame
detection + bolted-on Kalman/IoU tracker in track_flicker_smoothing.py /
track_full_clip_validation.py) with a PERSISTENT color per object id, so
identity-holding is visible directly in the video rather than just in stats.

Why this over the Kalman/ensemble pipeline: track_lifespans.py already
measured this quantitatively -- 10 of 24 objects survived >=80% of the
frames processed (vs 0 of 549 for the Kalman tracker), and 6 objects
recovered their own identity after a real gap with no help from us, because
this tracker has actual appearance memory of each fish instead of just
matching box positions frame to frame.

Known constraint (not fixable from our side -- see
project_sam3_native_point_reprompt.md): this GPU has ~21GB permanently held
by another user's process, and SAM3's own per-object internal state keeps
growing as it discovers more fish, so a full 299-frame run OOMs around frame
~246-260. This script catches that and finalizes the video with whatever
frames it got instead of crashing -- still 82-87% of the clip, enough to
demonstrate the identity-holding behavior.

Output:
    outputs/diagnostics/native_tracker_video/output.mp4
    outputs/diagnostics/native_tracker_video/track_stats.json (same stats
        shape as track_full_clip_validation.py's, for direct comparison)

Usage: python diagnostics/render_native_tracker_video.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_video_predictor

import sam3_predictor_patch  # noqa: F401 -- applies threshold-routing patch before building the predictor
from experiment_common import OUTPUT_ROOT

VIDEO_PATH = "assets/fish_data/clip1.mp4"
PROMPT = "fish"  # same prompt used for the track_lifespans.json baseline
RUN_DIR = OUTPUT_ROOT / "diagnostics" / "native_tracker_video"


def get_color(obj_id, rng_cache={}):
    if obj_id not in rng_cache:
        rng = np.random.default_rng(obj_id)  # deterministic per id
        rng_cache[obj_id] = tuple(int(c) for c in rng.integers(50, 255, size=3))
    return rng_cache[obj_id]


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    predictor = build_sam3_video_predictor(gpus_to_use=[0])
    print(f"model loaded in {time.time() - t0:.1f}s")

    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=VIDEO_PATH,
            offload_state_to_cpu=True,
        )
    )
    session_id = response["session_id"]

    response = predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=PROMPT)
    )
    print(f"Frame 0: {len(response['outputs']['out_obj_ids'])} detections for prompt '{PROMPT}'")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(RUN_DIR / "output.mp4"),
        ],
        stdin=subprocess.PIPE,
    )

    per_obj_frames = {}
    frames_rendered = 0
    crash_error = None
    t1 = time.time()
    try:
        for out in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            frame_idx = out["frame_index"]
            outputs = out["outputs"]

            ret, frame_bgr = cap.read()
            if not ret:
                break

            vis = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
            overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for obj_id, mask, prob, box in zip(
                outputs["out_obj_ids"],
                outputs["out_binary_masks"],
                outputs["out_probs"],
                outputs["out_boxes_xywh"],
            ):
                obj_id = int(obj_id)
                per_obj_frames.setdefault(obj_id, []).append(frame_idx)
                color = get_color(obj_id)

                mask_bool = mask if mask.ndim == 2 else mask[..., 0]
                colored = np.zeros((*mask_bool.shape, 4), dtype=np.uint8)
                colored[mask_bool] = (*color, 90)
                overlay = Image.alpha_composite(overlay, Image.fromarray(colored, mode="RGBA"))
                draw = ImageDraw.Draw(overlay)

                x, y, w, h = box
                box_abs = [x * width, y * height, (x + w) * width, (y + h) * height]
                draw.rectangle(box_abs, outline=(*color, 255), width=3)
                draw.text(
                    (box_abs[0], max(box_abs[1] - 15, 0)),
                    f"id={obj_id} {prob:.2f}",
                    fill=(*color, 255),
                )

            vis = Image.alpha_composite(vis, overlay).convert("RGB")
            frame_out = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
            ffmpeg_proc.stdin.write(frame_out.tobytes())
            frames_rendered += 1

            if frame_idx % 50 == 0:
                print(f"  rendered frame {frame_idx}/{total_frames}, elapsed {time.time() - t1:.1f}s")
    except Exception as e:
        crash_error = repr(e)
        print(f"\nStopped after {frames_rendered} rendered frames: {crash_error}")

    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    try:
        predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        predictor.shutdown()
    except Exception:
        pass  # session/predictor may already be in a bad state after the OOM

    print(f"\nRendered {frames_rendered}/{total_frames} frames "
          f"({frames_rendered / total_frames:.0%} of the clip) to {RUN_DIR / 'output.mp4'}")

    lengths = {oid: len(frames) for oid, frames in per_obj_frames.items()}
    survivors_80pct = [oid for oid, n in lengths.items() if n >= 0.8 * frames_rendered]
    fragments_15 = [oid for oid, n in lengths.items() if n <= 15]
    stats = {
        "frames_rendered": frames_rendered,
        "total_frames_in_clip": total_frames,
        "total_tracks_created": len(lengths),
        "track_length_distribution": sorted(lengths.values()),
        "full_clip_survivor_count": len(survivors_80pct),
        "full_clip_survivor_ids": survivors_80pct,
        "fragment_count": len(fragments_15),
        "crash_error": crash_error,
    }
    with open(RUN_DIR / "track_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Total unique objects: {stats['total_tracks_created']}")
    print(f"Full-clip survivors (>=80% of {frames_rendered} rendered frames): {stats['full_clip_survivor_count']}")
    print(f"Fragments (<=15 frames): {stats['fragment_count']}")
    print(f"Wrote {RUN_DIR / 'track_stats.json'}")


if __name__ == "__main__":
    main()
