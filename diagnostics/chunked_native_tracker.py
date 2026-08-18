"""
Gets FULL-CLIP (299 frame) coverage from the native SAM3 video-tracking
session despite the known shared-GPU OOM ceiling (~120-250 frames per
session depending on object density -- see
project_sam3_native_point_reprompt.md), by chunking: when one session's
propagate_in_video hits OOM, close it fully (frees the GPU, confirmed by the
"empty_cache freed ... free_pct -> 37.9%" log line every prior run showed),
start a brand-new session, and resume from exactly the frame the previous
chunk stopped at.

The hard part, and the reason this wasn't done earlier: a fresh add_prompt()
call assigns brand-new local object ids starting from 0 again, with no
knowledge of the previous chunk's ids -- naively chaining sessions would
make every fish change color/id at every chunk boundary. This fixes that by
IoU-matching the new chunk's first-frame (= the boundary frame, re-detected
by both chunks) boxes against the previous chunk's LAST recorded boxes at
that exact same frame, and remapping matched local ids back onto the
existing global id (unmatched local ids -- genuinely new fish, or ones the
previous chunk had already lost -- get a fresh global id). Not previously
tried; the id-stitching is a real approximation (a fish that changed
shape/position a lot right at the boundary could fail to match and get
double-counted as two ids) rather than a guarantee, but it's the direct,
cheap way to get continuity without re-architecting the tracker itself.

Uses the tuned thresholds from track_lifespans_tuned.py
(output_prob_thresh=0.3, new_det_thresh=0.5 vs library defaults 0.5/0.7),
which were confirmed to substantially increase recall (40 objects in 121
frames vs 24 in 246 at defaults) with zero new false positives flagged by
the temporal-stability filter.

Output:
    outputs/diagnostics/chunked_native_tracker/merged_tracks.json

Usage: python diagnostics/chunked_native_tracker.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.model_builder import build_sam3_video_predictor

import sam3_predictor_patch  # noqa: F401 -- applies threshold-routing patch before building the predictor
from experiment_common import OUTPUT_ROOT

VIDEO_PATH = "assets/fish_data/clip1.mp4"
PROMPT = "fish"
OUTPUT_PROB_THRESH = 0.3
NEW_DET_THRESH = 0.5
NUM_FRAMES_TOTAL = 299
IOU_MATCH_THRESHOLD = 0.3  # matches track_active_reprompt.py's REPROMPT_MIN_IOU;
                           # this codebase has repeatedly found IoU brittle for
                           # small boxes and settled on ~0.3 as a reasonable floor
WIDTH, HEIGHT = 1920, 1280
RUN_DIR = OUTPUT_ROOT / "diagnostics" / "chunked_native_tracker"
OUT_JSON = RUN_DIR / "merged_tracks.json"


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


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    merged_per_frame = {}  # frame_idx -> {global_id: {"box_rel", "box_abs", "prob"}}
    next_global_id = 0
    start_frame = 0
    chunk_num = 0

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

        # match this chunk's first-frame (boundary) detections against the
        # previous chunk's last-recorded boxes at that SAME frame index
        prev_frame_data = merged_per_frame.get(start_frame, {})
        local_to_global = {}
        used_global_ids = set()
        for obj_id, box_rel, prob in zip(
            first_outputs["out_obj_ids"], first_outputs["out_boxes_xywh"], first_outputs["out_probs"]
        ):
            obj_id = int(obj_id)
            box_abs = box_rel_to_abs(box_rel)
            best_gid, best_iou = None, 0.0
            for gid, info in prev_frame_data.items():
                if gid in used_global_ids:
                    continue
                i = iou(box_abs, info["box_abs"])
                if i > best_iou:
                    best_iou, best_gid = i, gid
            if best_gid is not None and best_iou >= IOU_MATCH_THRESHOLD:
                local_to_global[obj_id] = best_gid
                used_global_ids.add(best_gid)
            else:
                local_to_global[obj_id] = next_global_id
                next_global_id += 1

        n_matched = len(used_global_ids)
        n_new = len(local_to_global) - n_matched
        print(f"  boundary frame {start_frame}: {len(local_to_global)} objects "
              f"({n_matched} matched to existing ids, {n_new} new)")

        def record_frame(frame_idx, outputs):
            merged_per_frame.setdefault(frame_idx, {})
            nonlocal next_global_id
            for obj_id, box_rel, prob in zip(
                outputs["out_obj_ids"], outputs["out_boxes_xywh"], outputs["out_probs"]
            ):
                obj_id = int(obj_id)
                if obj_id not in local_to_global:
                    # a new object this chunk's own continuous re-detection found
                    # after the boundary frame -- fresh global id
                    local_to_global[obj_id] = next_global_id
                    next_global_id += 1
                gid = local_to_global[obj_id]
                box_abs = box_rel_to_abs(box_rel)
                merged_per_frame[frame_idx][gid] = {
                    "box_rel": [float(v) for v in box_rel],
                    "box_abs": box_abs,
                    "prob": float(prob),
                }

        record_frame(start_frame, first_outputs)

        last_ok_frame = start_frame
        crash_error = None
        try:
            for out in predictor.handle_stream_request(request=dict(
                type="propagate_in_video", session_id=session_id,
                propagation_direction="forward", start_frame_index=start_frame,
                output_prob_thresh=OUTPUT_PROB_THRESH, new_det_thresh=NEW_DET_THRESH,
            )):
                frame_idx = out["frame_index"]
                record_frame(frame_idx, out["outputs"])
                last_ok_frame = frame_idx
                if frame_idx % 50 == 0:
                    print(f"  chunk {chunk_num} propagated frame {frame_idx}")
        except Exception as e:
            crash_error = repr(e)
            print(f"  chunk {chunk_num} stopped after frame {last_ok_frame}: {crash_error}")

        try:
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
            predictor.shutdown()
        except Exception:
            pass

        print(f"Chunk {chunk_num} covered frames {start_frame}-{last_ok_frame} "
              f"({last_ok_frame - start_frame + 1} frames)")
        if last_ok_frame <= start_frame:
            print("Chunk made no progress past the boundary frame -- stopping to avoid an infinite loop")
            break
        start_frame = last_ok_frame

    print(f"\n=== Done: {chunk_num} chunks, {len(merged_per_frame)} of {NUM_FRAMES_TOTAL} frames covered "
          f"({len(merged_per_frame) / NUM_FRAMES_TOTAL:.0%}), {next_global_id} total global ids ===")

    serializable = {
        str(f): {str(gid): {"box": v["box_rel"], "prob": v["prob"]} for gid, v in objs.items()}
        for f, objs in merged_per_frame.items()
    }
    with open(OUT_JSON, "w") as fp:
        json.dump({
            "num_frames_total": NUM_FRAMES_TOTAL,
            "frames_covered": sorted(merged_per_frame.keys()),
            "num_global_ids": next_global_id,
            "per_frame_boxes": serializable,
        }, fp)
    print(f"Saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
