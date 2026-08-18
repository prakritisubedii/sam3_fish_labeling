"""
SAM3-ensemble + Weighted Box Fusion.

Pivot from the background-subtraction pipeline (stage1_bg_subtraction.py):
that approach fought the video's moving camera + complex 3D coral texture,
the same thing that also broke naive motion-point prompting and CLAHE earlier
this session. SAM3's own text-prompted detection is immune to that problem
(it runs per-frame, independent of motion) and has held up as the most
reliable signal all along -- its limit is recall on hard cases, not false
positives. So instead of an external appearance-based candidate generator,
this runs SAM3 itself at two prompt variants already shown to work well
individually this session ("fish": 24 objects/video, "small fish": 31) and
fuses their per-frame outputs with weighted_boxes_fusion, keeping SAM3's
precision while using disagreement-resolution across prompts for recall
instead of a shaky external detector.

Note: threshold-sweeping was NOT added as a second ensemble dimension --
run_experiments.py already showed thresh=0.5/0.3/0.2 give byte-for-byte
identical results for a fixed prompt on this video, so it adds no diversity.

Uses the stateless per-frame image API (Sam3Processor), consistent with the
rest of today's per-frame work, and reads from the shared extracted-frames
directory (outputs/_frames/) rather than the video directly.

Output:
    outputs/ensemble_wbf/output.mp4       (fused boxes drawn on original frames)
    outputs/ensemble_wbf/detections.json  (per-frame fused boxes + scores)

KNOWN LIMITATION: this recipe does not detect fish within a dense, tightly-
schooling, motion-blurred cluster (confirmed on clip1.mp4 frame 0). This was
tested and ruled out as a threshold/prompt/resolution problem -- see
KNOWN_LIMITATIONS.md before spending time re-chasing it.

Usage: python ensemble_wbf.py
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image, ImageDraw

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from experiment_common import OUTPUT_ROOT
from extract_frames import extract_frames, frame_path

VIDEO_PATH = "assets/fish_data/clip1.mp4"
RUN_DIR = OUTPUT_ROOT / "ensemble_wbf"

PROMPTS = ["fish", "small fish"]
CONFIDENCE_THRESHOLD = 0.3  # per-prompt threshold before fusion; established
                            # this session as the best value for "small fish"
                            # (and threshold doesn't change "fish"'s results)
PROMPT_WEIGHTS = [1.0, 1.0]

# weighted_boxes_fusion params (from the original Stage 5 spec)
IOU_THR = 0.4
SKIP_BOX_THR = 0.1


def detect(processor, image: Image.Image, prompt: str):
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt=prompt, state=state)
    boxes = state["boxes"].float().cpu().numpy() if len(state["boxes"]) else np.zeros((0, 4))
    scores = state["scores"].float().cpu().numpy() if len(state["scores"]) else np.zeros((0,))
    return boxes, scores  # boxes in pixel [x0, y0, x1, y1]


def fuse_frame(per_prompt_boxes, per_prompt_scores, width, height):
    boxes_norm_list = []
    for boxes in per_prompt_boxes:
        norm = boxes.copy().astype(np.float64)
        norm[:, [0, 2]] /= width
        norm[:, [1, 3]] /= height
        norm = np.clip(norm, 0.0, 1.0)
        boxes_norm_list.append(norm.tolist())
    scores_list = [s.tolist() for s in per_prompt_scores]
    labels_list = [[0] * len(s) for s in per_prompt_scores]

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_norm_list,
        scores_list,
        labels_list,
        weights=PROMPT_WEIGHTS,
        iou_thr=IOU_THR,
        skip_box_thr=SKIP_BOX_THR,
    )
    fused_boxes = np.array(fused_boxes).reshape(-1, 4)
    fused_boxes[:, [0, 2]] *= width
    fused_boxes[:, [1, 3]] *= height
    return fused_boxes, np.array(fused_scores)


def draw_boxes(image_bgr: np.ndarray, boxes, scores) -> np.ndarray:
    vis = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(vis)
    for (x0, y0, x1, y1), score in zip(boxes, scores):
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
        draw.text((x0, max(y0 - 12, 0)), f"{score:.2f}", fill=(0, 255, 0))
    return cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)


def main():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    num_frames = extract_frames(VIDEO_PATH)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)

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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    detections = {}
    per_prompt_totals = {p: 0 for p in PROMPTS}
    fused_counts = []

    for idx in range(num_frames):
        frame_bgr = cv2.imread(str(frame_path(idx)))
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        per_prompt_boxes, per_prompt_scores = [], []
        for prompt in PROMPTS:
            boxes, scores = detect(processor, frame_pil, prompt)
            per_prompt_boxes.append(boxes)
            per_prompt_scores.append(scores)
            per_prompt_totals[prompt] += len(scores)

        fused_boxes, fused_scores = fuse_frame(per_prompt_boxes, per_prompt_scores, width, height)
        detections[str(idx)] = [
            [float(x0), float(y0), float(x1), float(y1), float(s)]
            for (x0, y0, x1, y1), s in zip(fused_boxes, fused_scores)
        ]
        fused_counts.append(len(fused_scores))

        vis = draw_boxes(frame_bgr, fused_boxes, fused_scores)
        ffmpeg_proc.stdin.write(vis.tobytes())

        if idx % 50 == 0:
            print(f"frame {idx}/{num_frames}: fused {len(fused_scores)} boxes")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    with open(RUN_DIR / "detections.json", "w") as f:
        json.dump(
            {
                "params": {
                    "prompts": PROMPTS,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "prompt_weights": PROMPT_WEIGHTS,
                    "iou_thr": IOU_THR,
                    "skip_box_thr": SKIP_BOX_THR,
                },
                "detections": detections,
            },
            f,
        )

    fused_counts_arr = np.array(fused_counts)
    print(f"\nFrames processed: {num_frames}")
    for prompt in PROMPTS:
        print(f"  '{prompt}' total detections across video: {per_prompt_totals[prompt]}")
    print(f"Fused avg detections/frame: {fused_counts_arr.mean():.2f}")
    print(f"Fused total detections: {fused_counts_arr.sum()}")
    print(f"\nWrote {RUN_DIR / 'output.mp4'}")
    print(f"Wrote {RUN_DIR / 'detections.json'}")


if __name__ == "__main__":
    main()
