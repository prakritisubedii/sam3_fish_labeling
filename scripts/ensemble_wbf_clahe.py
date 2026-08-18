"""
Full-clip variant of ensemble_wbf.py that reads from the CLAHE-preprocessed
video (outputs/_preprocessed/clip1_clahe.mp4) instead of the raw extracted
frames, to directly test a combination that was never run: does CLAHE help
the two-prompt ("fish" + "small fish") WBF-fused ensemble across the whole
clip, not just the single-prompt run already on record
(outputs/thresh0.3_prompt-small-fish_clahe/run_info.json, which showed a
measurable regression: 31->30 unique objects, 18.51->16.72 avg
detections/frame vs the non-CLAHE baseline).

A diagnostics/clahe_ensemble_quickcheck.py spot check on 7 frames found a
similar net regression (128->118 fused detections) but not uniformly -- some
frames improved, others (esp. frame 250) lost more than half their
detections, visually concentrated in shadowed coral crevices. This full run
exists to see whether that holds up (or reverses) across all 299 frames.

Same detect()/fuse_frame() logic and WBF params as ensemble_wbf.py, same
confidence_threshold=0.3 for both prompts -- only the input video differs, so
threshold isn't a confound (this was explicitly checked before running this).

Reads frames sequentially from the CLAHE video via cv2.VideoCapture (not
pre-extracted to PNGs the way the raw video is via extract_frames.py).

Output:
    outputs/ensemble_wbf_clahe/output.mp4
    outputs/ensemble_wbf_clahe/detections.json

Usage: python ensemble_wbf_clahe.py
"""

import json
import subprocess

import cv2
import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image, ImageDraw

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from experiment_common import OUTPUT_ROOT

CLAHE_VIDEO_PATH = "outputs/_preprocessed/clip1_clahe.mp4"
RUN_DIR = OUTPUT_ROOT / "ensemble_wbf_clahe"

PROMPTS = ["fish", "small fish"]
CONFIDENCE_THRESHOLD = 0.3
PROMPT_WEIGHTS = [1.0, 1.0]
IOU_THR = 0.4
SKIP_BOX_THR = 0.1


def detect(processor, image: Image.Image, prompt: str):
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt=prompt, state=state)
    boxes = state["boxes"].float().cpu().numpy() if len(state["boxes"]) else np.zeros((0, 4))
    scores = state["scores"].float().cpu().numpy() if len(state["scores"]) else np.zeros((0,))
    return boxes, scores


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
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CLAHE_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
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
        idx += 1

    cap.release()
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
                    "source_video": CLAHE_VIDEO_PATH,
                },
                "detections": detections,
            },
            f,
        )

    fused_counts_arr = np.array(fused_counts)
    print(f"\nFrames processed: {idx}")
    for prompt in PROMPTS:
        print(f"  '{prompt}' total detections across video: {per_prompt_totals[prompt]}")
    print(f"Fused avg detections/frame: {fused_counts_arr.mean():.2f}")
    print(f"Fused total detections: {fused_counts_arr.sum()}")
    print(f"\nWrote {RUN_DIR / 'output.mp4'}")
    print(f"Wrote {RUN_DIR / 'detections.json'}")


if __name__ == "__main__":
    main()
