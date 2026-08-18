import subprocess

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_video_predictor

import sam3_predictor_patch  # noqa: F401 -- applies threshold-routing patch before building the predictor

VIDEO_PATH = "assets/fish_data/clip1.mp4"
OUTPUT_PATH = "results/test_clip1_output.mp4"
PROMPT = "fish"

predictor = build_sam3_video_predictor(gpus_to_use=[0])

response = predictor.handle_request(
    request=dict(type="start_session", resource_path=VIDEO_PATH)
)
session_id = response["session_id"]

response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=PROMPT,
    )
)
print(f"Frame 0: {len(response['outputs']['out_obj_ids'])} detections for prompt '{PROMPT}'")

outputs_per_frame = {}
for out in predictor.handle_stream_request(
    request=dict(type="propagate_in_video", session_id=session_id)
):
    outputs_per_frame[out["frame_index"]] = out["outputs"]

predictor.handle_request(request=dict(type="close_session", session_id=session_id))
predictor.shutdown()

print(f"Propagated over {len(outputs_per_frame)} frames")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# cv2.VideoWriter's "mp4v" fourcc encodes old MPEG-4 Part 2, which many
# players (QuickTime, browsers) fail to open from an .mp4 container even
# though ffmpeg/VLC decode it fine. Pipe raw frames to ffmpeg instead so the
# output is standard H.264, which is universally playable.
ffmpeg_proc = subprocess.Popen(
    [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        OUTPUT_PATH,
    ],
    stdin=subprocess.PIPE,
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
print(f"Saved annotated video to {OUTPUT_PATH}")
