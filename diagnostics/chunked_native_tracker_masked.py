"""
Combines chunked_native_tracker.py's chunking + IoU identity-stitching (to
get full 299/299-frame coverage past the shared-GPU OOM ceiling) with
render_native_tracker_video.py's mask+box+persistent-color rendering (the
"native SAM3 style" visualization the box-only chunked video was missing).
No chunk-boundary text overlay -- removed per user request.

Renders directly to a single ffmpeg pipe that stays open across all chunks
(masks are too large to round-trip through JSON the way merged_tracks.json
did for boxes-only), so this is one continuous video write, not a separate
render pass over cached data.

Uses the same tuned thresholds (output_prob_thresh=0.3, new_det_thresh=0.5)
and IoU boundary-matching (threshold 0.3) validated in
chunked_native_tracker.py.

Output: outputs/diagnostics/chunked_native_tracker/output_masked.mp4

Usage: python diagnostics/chunked_native_tracker_masked.py
"""

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
from extract_frames import frame_path

VIDEO_PATH = "assets/fish_data/clip1.mp4"
PROMPT = "fish"
OUTPUT_PROB_THRESH = 0.3
NEW_DET_THRESH = 0.5
NUM_FRAMES_TOTAL = 299
IOU_MATCH_THRESHOLD = 0.3
WIDTH, HEIGHT = 1920, 1280
RUN_DIR = OUTPUT_ROOT / "diagnostics" / "chunked_native_tracker"


def box_rel_to_abs(box_rel):
    x, y, w, h = box_rel
    return [x * WIDTH, y * HEIGHT, (x + w) * WIDTH, (y + h) * HEIGHT]


def iou(b1, b2):
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def get_color(gid, cache={}):
    if gid not in cache:
        rng = np.random.default_rng(gid)
        cache[gid] = tuple(int(c) for c in rng.integers(50, 255, size=3))
    return cache[gid]


def render_frame(frame_bgr, objs):
    """objs: list of (global_id, mask_bool, prob, box_abs)."""
    vis = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for gid, mask, prob, box_abs in objs:
        color = get_color(gid)
        mask_bool = mask if mask.ndim == 2 else mask[..., 0]
        colored = np.zeros((*mask_bool.shape, 4), dtype=np.uint8)
        colored[mask_bool] = (*color, 90)
        overlay = Image.alpha_composite(overlay, Image.fromarray(colored, mode="RGBA"))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(box_abs, outline=(*color, 255), width=3)
        draw.text((box_abs[0], max(box_abs[1] - 15, 0)), f"id={gid} {prob:.2f}", fill=(*color, 255))
    vis = Image.alpha_composite(vis, overlay).convert("RGB")
    return cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    out_path = RUN_DIR / "output_masked.mp4"
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )

    last_frame_boxes = {}  # frame_idx -> {global_id: box_abs}, only need the boundary frame each time
    next_global_id = 0
    start_frame = 0
    chunk_num = 0
    rendered_frames = set()

    while start_frame < NUM_FRAMES_TOTAL - 1:
        chunk_num += 1
        print(f"\n=== Chunk {chunk_num}: starting at frame {start_frame} ===")
        t0 = time.time()
        predictor = build_sam3_video_predictor(gpus_to_use=[0])
        print(f"model loaded in {time.time() - t0:.1f}s")

        response = predictor.handle_request(request=dict(
            type="start_session", resource_path=VIDEO_PATH, offload_state_to_cpu=True,
        ))
        session_id = response["session_id"]

        response = predictor.handle_request(request=dict(
            type="add_prompt", session_id=session_id, frame_index=start_frame, text=PROMPT,
            output_prob_thresh=OUTPUT_PROB_THRESH, new_det_thresh=NEW_DET_THRESH,
        ))
        first_outputs = response["outputs"]

        prev_frame_data = last_frame_boxes.get(start_frame, {})
        local_to_global = {}
        used_global_ids = set()
        for obj_id, box_rel in zip(first_outputs["out_obj_ids"], first_outputs["out_boxes_xywh"]):
            obj_id = int(obj_id)
            box_abs = box_rel_to_abs(box_rel)
            best_gid, best_iou = None, 0.0
            for gid, prev_box in prev_frame_data.items():
                if gid in used_global_ids:
                    continue
                i = iou(box_abs, prev_box)
                if i > best_iou:
                    best_iou, best_gid = i, gid
            if best_gid is not None and best_iou >= IOU_MATCH_THRESHOLD:
                local_to_global[obj_id] = best_gid
                used_global_ids.add(best_gid)
            else:
                local_to_global[obj_id] = next_global_id
                next_global_id += 1
        print(f"  boundary frame {start_frame}: {len(local_to_global)} objects "
              f"({len(used_global_ids)} matched, {len(local_to_global) - len(used_global_ids)} new)")

        def handle_frame(frame_idx, outputs):
            nonlocal next_global_id
            objs = []
            frame_boxes = {}
            for obj_id, mask, prob, box_rel in zip(
                outputs["out_obj_ids"], outputs["out_binary_masks"],
                outputs["out_probs"], outputs["out_boxes_xywh"],
            ):
                obj_id = int(obj_id)
                if obj_id not in local_to_global:
                    local_to_global[obj_id] = next_global_id
                    next_global_id += 1
                gid = local_to_global[obj_id]
                box_abs = box_rel_to_abs(box_rel)
                objs.append((gid, mask, float(prob), box_abs))
                frame_boxes[gid] = box_abs
            last_frame_boxes[frame_idx] = frame_boxes
            if frame_idx not in rendered_frames:
                frame_bgr = cv2.imread(str(frame_path(frame_idx)))
                vis = render_frame(frame_bgr, objs)
                ffmpeg_proc.stdin.write(vis.tobytes())
                rendered_frames.add(frame_idx)

        handle_frame(start_frame, first_outputs)

        last_ok_frame = start_frame
        try:
            for out in predictor.handle_stream_request(request=dict(
                type="propagate_in_video", session_id=session_id,
                propagation_direction="forward", start_frame_index=start_frame,
                output_prob_thresh=OUTPUT_PROB_THRESH, new_det_thresh=NEW_DET_THRESH,
            )):
                frame_idx = out["frame_index"]
                handle_frame(frame_idx, out["outputs"])
                last_ok_frame = frame_idx
                if frame_idx % 50 == 0:
                    print(f"  chunk {chunk_num} propagated frame {frame_idx}")
        except Exception as e:
            print(f"  chunk {chunk_num} stopped after frame {last_ok_frame}: {repr(e)}")

        try:
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
            predictor.shutdown()
        except Exception:
            pass

        print(f"Chunk {chunk_num} covered frames {start_frame}-{last_ok_frame} ({last_ok_frame - start_frame + 1} frames)")
        if last_ok_frame <= start_frame:
            print("Chunk made no progress -- stopping")
            break
        start_frame = last_ok_frame

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    print(f"\n=== Done: {chunk_num} chunks, {len(rendered_frames)}/{NUM_FRAMES_TOTAL} frames rendered "
          f"({len(rendered_frames) / NUM_FRAMES_TOTAL:.0%}) ===")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
