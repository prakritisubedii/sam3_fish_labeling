# SAM3 fish tracking

A zero-shot underwater fish detection and tracking pipeline built on top of
Meta's [SAM3](https://github.com/facebookresearch/sam3), validated against
one real reef clip (`assets/fish_data/clip1.mp4`, moving-camera diver
footage). This repository contains only the pipeline built on top of SAM3 —
it does not vendor SAM3 itself.

See [FISH_TRACKING.md](FISH_TRACKING.md) for the full pipeline walkthrough
and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for the main negative
result and why it isn't a quick fix.

## Setup

1. Clone and install upstream SAM3 separately, per its own instructions:
   ```
   git clone https://github.com/facebookresearch/sam3
   cd sam3 && pip install -e .
   ```
2. Install this repository's own dependencies: `pip install -r requirements.txt`.
3. Requires a GPU with CUDA; SAM3 runs its video predictor in bfloat16
   autocast.
4. Requires `ffmpeg` on `PATH` (used to mux tracked-output videos via
   subprocess rather than `cv2.VideoWriter`, so results play back in
   browsers/QuickTime).
5. On NixOS: set `LD_LIBRARY_PATH` to include `/run/opengl-driver/lib`
   (needed for the driver's libGL/libcuda) and set
   `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` (Triton's NMS kernel, used by
   the SAM3 video predictor, looks up libcuda via `/sbin/ldconfig -p`, which
   doesn't exist on NixOS). Not needed on a standard Linux install with the
   NVIDIA driver installed normally.

## Layout

- `scripts/` — every pipeline script (`scripts/diagnostics/` holds the
  tracking and false-positive-filter diagnostics).
- `assets/` — inputs: `fish_data/clip1.mp4`, the test clip everything in
  this repository is validated against.
- `results/` — every output worth keeping: rendered videos, comparison
  images, sweep summaries. Regenerable scratch output goes to `outputs/`
  instead, which is gitignored.

All scripts are run with the repository root as the working directory, for
example:
```
python scripts/ensemble_wbf.py
python scripts/diagnostics/render_native_tracker_video.py
```

## Required: the predictor patch

Upstream SAM3's `Sam3VideoPredictor` has two bugs that make requested
detection thresholds silently do nothing (see
[`scripts/sam3_predictor_patch.py`](scripts/sam3_predictor_patch.py) for the
full explanation). Every script in this repository that builds a video
predictor imports that patch module as its first local import. Any new
script built on top of this pipeline that constructs a
`Sam3VideoPredictor` needs the same import, or requested thresholds will be
silently ignored.

## Quick start

Run `python scripts/test_sam3_video.py` to check the setup: it runs SAM3's
native video predictor over `assets/fish_data/clip1.mp4` and writes an
annotated video to `results/test_clip1_output.mp4`. A pre-rendered copy of
that output is already checked in, so the expected result can be seen
without a GPU.
