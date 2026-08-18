# SAM3 fish tracking

A zero-shot underwater fish detection and tracking pipeline built on top of
Meta's [SAM3](https://github.com/facebookresearch/sam3), validated against
one real reef clip (`assets/fish_data/clip1.mp4`, moving-camera diver
footage). This repo is only the pipeline built on top of SAM3 — it does not
vendor SAM3 itself.

See [FISH_TRACKING.md](FISH_TRACKING.md) for the full pipeline walkthrough
and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for the main negative
result and why it isn't a quick fix.

## Setup

1. Clone and install upstream SAM3 separately, per its own instructions:
   ```
   git clone https://github.com/facebookresearch/sam3
   cd sam3 && pip install -e .
   ```
2. Install this repo's own dependencies: `pip install -r requirements.txt`.
3. GPU + CUDA required; SAM3 runs its video predictor in bfloat16 autocast.
4. `ffmpeg` on `PATH` (used to mux tracked-output videos via subprocess, not
   `cv2.VideoWriter`, so results play back in browsers/QuickTime).
5. If you're on NixOS: you'll likely need to point `LD_LIBRARY_PATH` at
   `/run/opengl-driver/lib` (driver's libGL/libcuda) and set
   `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` (Triton's NMS kernel — used
   by the SAM3 video predictor — looks up libcuda via
   `/sbin/ldconfig -p`, which doesn't exist on NixOS). Not needed on a
   standard Linux box with the NVIDIA driver installed normally.

All scripts assume they're run with this repo's root as the working
directory, e.g. `python ensemble_wbf.py` or
`python diagnostics/chunked_native_tracker.py`.

## Important: apply the predictor patch

Upstream SAM3's `Sam3VideoPredictor` has two bugs that make requested
detection thresholds silently do nothing (see
[`sam3_predictor_patch.py`](sam3_predictor_patch.py) for the full
explanation). Every script here that builds a video predictor already
imports that patch module as its first local import, so nothing extra is
required to run them — just don't skip that import if you copy code out of
this repo into your own scripts.

## Quick start

- `python test_sam3.py` — single-image text-prompted segmentation on
  `test_fish.png`.
- `python test_sam3_video.py` — native SAM3 video tracking over
  `assets/fish_data/clip1.mp4`; `test_clip1_output.mp4` (checked in) shows
  the expected output.
