"""
Shared frame-extraction utility for the background-subtraction + SAM3 + SR
pipeline (stage1..stage5). Decodes the source video once to a directory of
PNG frames so every stage addresses the same frame by the same index/filename
instead of re-seeking the compressed video (which cv2 doesn't guarantee is
frame-exact) or repeatedly re-decoding it.

Usage: python extract_frames.py
"""

from pathlib import Path

import cv2

VIDEO_PATH = "assets/fish_data/clip1.mp4"
FRAMES_DIR = Path("outputs/_frames")


def frame_path(frame_idx: int) -> Path:
    return FRAMES_DIR / f"frame_{frame_idx:06d}.png"


def extract_frames(video_path: str = VIDEO_PATH, frames_dir: Path = FRAMES_DIR) -> int:
    """Extracts all frames if not already done. Returns the frame count."""
    cap = cv2.VideoCapture(video_path)
    expected_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    existing = sorted(frames_dir.glob("frame_*.png")) if frames_dir.exists() else []
    if len(existing) == expected_count:
        cap.release()
        print(f"Reusing {expected_count} already-extracted frames in {frames_dir}")
        return expected_count

    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(frame_path(frame_idx)), frame)
        frame_idx += 1
    cap.release()
    print(f"Extracted {frame_idx} frames to {frames_dir}")
    return frame_idx


if __name__ == "__main__":
    extract_frames()
