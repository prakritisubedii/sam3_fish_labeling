# Fish tracking pipeline

A zero-shot underwater fish detection and tracking pipeline built on top of
Meta's SAM3, built and validated against one real reef clip
(`assets/fish_data/clip1.mp4`, moving-camera diver footage). See
[README.md](README.md) for setup. Everything below is code and results kept
from a larger set of experiments — see
[Investigated but not included](#investigated-but-not-included) for what was
tried and deliberately left out, and why.

## Production detection pipeline

**`scripts/ensemble_wbf.py`** is the recipe `KNOWN_LIMITATIONS.md` refers to
as "production": SAM3 is prompted twice per frame (`"fish"` and
`"small fish"`), and the two detection sets are merged with weighted box
fusion (`ensemble_boxes.weighted_boxes_fusion`) rather than naive NMS, so a
fish either prompt independently finds isn't dropped just because the other
prompt also found it with a slightly different box. It depends on
`scripts/experiment_common.py` (shared render/summarize helpers,
`OUTPUT_ROOT = outputs/`) and `scripts/extract_frames.py` (frame-cache
extraction).

**`scripts/run_experiments.py`** is the sweep that established this recipe:
it compares prompt wording (`"fish"` vs. `"small fish"` vs. `"fish near
coral"`) and detection threshold, and is what produced
`results/comparisons/` and `results/sweep_summary.json`. Re-running it
regenerates the full `outputs/<run-name>/` folders these summaries were
pulled from.

## Tracking across the full clip

Two problems had to be solved to track individual fish across the whole
clip, not just detect them per-frame:

- **Native SAM3 video sessions beat an IoU+Kalman tracker.** Handing SAM3's
  own video predictor session responsibility for identity (instead of a
  from-scratch SORT-style IoU+Kalman tracker over independent per-frame
  detections) held fish identity far better in a head-to-head comparison:
  10 of 24 objects survived ≥80% of the frames processed, vs 0 of 549 for
  the Kalman tracker, and 6 objects recovered their own identity after a
  real gap with no help, because the native tracker has actual appearance
  memory of each fish instead of just matching box positions frame to
  frame. See `scripts/diagnostics/render_native_tracker_video.py`, which
  renders this directly from a live SAM3 session (persistent color per
  object id).
- **A single native session still OOMs partway through the clip.** The fix is
  chunked sessions — process the clip in bounded-length chunks, each its own
  SAM3 video session, then stitch track identities across chunk boundaries
  by IoU-matching the last frame of chunk *N* against the first frame of
  chunk *N+1*. This gets 100% clip coverage where a single full-clip session
  couldn't. See `scripts/diagnostics/chunked_native_tracker.py` (the core
  implementation, writes `merged_tracks.json`) and
  `chunked_native_tracker_masked.py` (a self-contained rerun that renders the
  same chunked+stitched result with masks, not just box outlines).

All three of these scripts build a `Sam3VideoPredictor` and depend on
[`scripts/sam3_predictor_patch.py`](scripts/sam3_predictor_patch.py)
(imported at the top of each) to make their requested
`output_prob_thresh`/`new_det_thresh`/`det_nms_thresh` actually take effect
— see that file for why upstream SAM3 needs this.

## False-positive filtering

`scripts/diagnostics/temporal_stability_filter.py` reads
`outputs/ensemble_wbf/detections.json` (run `scripts/ensemble_wbf.py`
first), turns it into per-track box histories using the Kalman/IoU tracker
in `scripts/track_flicker_smoothing.py` (kept as a dependency for this — see
[Investigated but not included](#investigated-but-not-included) for why
that tracker isn't used as the primary tracker itself), then scores each
track by how much its size/shape/contrast vary over its lifetime. Static
false positives (coral, rock) sit still and look nearly identical frame to
frame — genuinely low variance — which separates them from real fish even
when a confidence-threshold cut can't. It caught a real 253-frame-long
static false positive that a confidence filter alone missed.
`temporal_stability_video.py` renders the result across the full clip,
colored red/orange/green by the stability rule.

## Results

`results/` holds a curated set of outputs referenced above and in
`KNOWN_LIMITATIONS.md`, small enough to check in directly:

- `results/comparisons/` — annotated frames showing the production ensemble
  (`ensemble_wbf.py`) and the winning sweep config (`"small fish"` @ 0.3)
  side by side, at frame 0 and frame 200.
- `results/sweep_summary.json` — per-run detection stats from
  `run_experiments.py`'s prompt/threshold sweep.
- `results/test_clip1_output.mp4` — a pre-rendered
  `scripts/test_sam3_video.py` run, so the native tracker's output can be
  seen without a GPU.

Everything else under `outputs/` is regenerable scratch space (raw extracted
frames, full per-run detection dumps) and is gitignored — running the full
sweep produces several GB of it.

## Known limitations

See [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) for the main negative
result (dense, motion-blurred schooling fish aren't detected by any prompt,
threshold, or preprocessing combination tried) and why it isn't a quick fix.

## Investigated but not included

These were tried and either failed outright or were superseded by what's
kept above. Documented here instead of shipped as code, so the repo stays
limited to what actually works:

- **CLAHE contrast correction**, tested at every stage (single prompt, the
  full two-prompt ensemble, and a multi-frame spot check) — a consistent net
  regression: fewer detections overall, and some frames lost more than half
  their boxes in shadowed coral crevices.
- **Motion-seeded point prompts** — added a tracked object wherever
  frame-to-frame motion was detected, on the theory that fish move and coral
  doesn't. Fails on this footage because the camera itself drifts over
  complex 3D coral, so large motionless coral formations get flagged as
  "moving" and tracked as new fish.
- **Background-subtraction candidate generation** (MOG2 over
  motion-compensated frames), abandoned for the same reason as motion-seeded
  points above: a moving camera over complex 3D coral breaks per-pixel
  motion assumptions, even after compensating for camera motion.
- **Top-crop tiling for small/distant fish** — cropping to just the region a
  fish sits in, for more effective resolution after SAM3's fixed 1008x1008
  resize. Tested directly on the hardest schooling-fish cluster; still
  missed even at large zoom (see `KNOWN_LIMITATIONS.md`).
- **A hand-rolled IoU+Kalman tracker as the primary tracker**, later
  extended with active re-prompting on tracking gaps — even the single most
  reliably-detected fish in the clip fragmented into 3 separate track IDs.
  Superseded by the native-SAM3-session + chunking approach above, which
  held identity far better without a hand-rolled tracker. The base tracker
  (`scripts/track_flicker_smoothing.py`) is still kept, but only as plumbing
  the temporal-stability filter reuses to get per-track box histories, not
  as a tracker in its own right.
- **RVRT multi-frame video deblurring** — tests whether fusing information
  across frames recovers structure in the blurred schooling-fish cluster
  that single-frame methods can't. Excluded because it needs a separate
  external repo (`JingyunLiang/RVRT`) with its own pretrained checkpoints and
  CUDA toolkit setup — not something to vendor into this pipeline.
- **Point re-prompt recovery** — re-prompting SAM3 at a lost fish's
  last-known location. Recovered identity for about 6 frames before dropping
  again, too limited to be production-worthy.
- **A crop-tagging / detection-reranking pipeline** — in progress, not yet a
  validated part of the pipeline.
- **A handful of narrower checks** confirming decisions already reflected in
  the code above: whether high temporal-stability variance alone confirms a
  real fish (mostly, but with enough ambiguous cases it wasn't adopted as a
  standalone rule), and native-tracker survival rates across prompts.
