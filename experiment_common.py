"""Shared helpers for the SAM3 fish-detection experiment scripts (run_experiments*.py)."""

import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

OUTPUT_ROOT = Path("outputs")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render_video(video_path: str, outputs_per_frame: dict, output_path: Path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    rng = np.random.default_rng(0)
    obj_colors = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = outputs_per_frame.get(frame_idx)
        if out is not None and len(out["out_obj_ids"]) > 0:
            vis = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
            overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for obj_id, mask, prob, box in zip(
                out["out_obj_ids"],
                out["out_binary_masks"],
                out["out_probs"],
                out["out_boxes_xywh"],
            ):
                if obj_id not in obj_colors:
                    obj_colors[obj_id] = tuple(int(c) for c in rng.integers(50, 255, size=3))
                color = obj_colors[obj_id]

                mask_bool = mask if mask.ndim == 2 else mask[..., 0]
                colored = np.zeros((*mask_bool.shape, 4), dtype=np.uint8)
                colored[mask_bool] = (*color, 100)
                overlay = Image.alpha_composite(overlay, Image.fromarray(colored, mode="RGBA"))

                x, y, w, h = box
                box_abs = [x * width, y * height, (x + w) * width, (y + h) * height]
                draw = ImageDraw.Draw(overlay)
                draw.rectangle(box_abs, outline=(*color, 255), width=3)
                draw.text(
                    (box_abs[0], max(box_abs[1] - 15, 0)),
                    f"id={obj_id} {prob:.2f}",
                    fill=(*color, 255),
                )

            vis = Image.alpha_composite(vis, overlay).convert("RGB")
            frame = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

        ffmpeg_proc.stdin.write(frame.tobytes())
        frame_idx += 1

    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()


def summarize(outputs_per_frame: dict) -> dict:
    counts_per_frame = {
        idx: len(out["out_obj_ids"]) for idx, out in outputs_per_frame.items()
    }
    all_obj_ids = sorted({int(oid) for out in outputs_per_frame.values() for oid in out["out_obj_ids"]})
    all_probs = [
        float(p) for out in outputs_per_frame.values() for p in out["out_probs"]
    ]
    counts = list(counts_per_frame.values())
    return {
        "num_frames_with_output": len(outputs_per_frame),
        "unique_object_ids": all_obj_ids,
        "num_unique_objects": len(all_obj_ids),
        "avg_detections_per_frame": sum(counts) / len(counts) if counts else 0,
        "max_detections_in_a_frame": max(counts) if counts else 0,
        "min_detections_in_a_frame": min(counts) if counts else 0,
        "avg_confidence": sum(all_probs) / len(all_probs) if all_probs else 0,
    }
