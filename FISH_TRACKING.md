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
prompt also found it with a slightly different box.
`scripts/ensemble_wbf_clahe.py` is the same recipe with CLAHE contrast
correction applied to frames first — kept because it's a real variant worth
comparing, even though `KNOWN_LIMITATIONS.md` found it doesn't fix the
hardest case (dense motion-blurred schools).

Both depend on `scripts/experiment_common.py` (shared render/summarize
helpers, `OUTPUT_ROOT = outputs/`) and `scripts/extract_frames.py`
(frame-cache extraction) — keep those two alongside them.

`scripts/run_experiments.py` / `scripts/run_experiments_enhanced.py` are the
sweep harness
that generated the threshold/prompt comparison grid — this is what produced
`results/comparisons/` and `results/sweep_summary.json` (a renamed copy of
their `outputs/index.json`). Re-running them regenerates the full
`outputs/<run-name>/` folders these summaries were pulled from.

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

- `results/comparisons/` — before/after frames from the threshold/prompt
  sweep (`scripts/run_experiments*.py`).
- `results/sweep_summary.json` — per-run detection stats for that sweep.
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

Several approaches were tried, produced a clear negative or superseded
result, and were deliberately left out of this repo to keep it to what
actually works. Each is described here instead of shipping the script:

- **Background-subtraction candidate generation**
  (would-be `stage1_bg_subtraction.py`) — MOG2 over motion-compensated
  (ECC-aligned) frames as a SAM3-independent candidate-box generator. Ruled
  out because this footage isn't static-camera: a diver swims through dense
  3D coral, so even small camera drift shifts every high-frequency coral
  edge and the raw MOG2 signal floods with false "motion." The ECC alignment
  fix contains that, but the approach was abandoned in favor of the
  prompted-SAM3 + WBF recipe. Not the same issue as the schooling-fish
  detection miss below — this one is about motion *between* frames, that
  one's about detection *within* a single frame.
- **Top-crop tiling for small/distant fish** (would-be
  `diagnostic_tiling_crop.py`) — cropping to just the region a fish sits in,
  to give it more effective resolution after SAM3's fixed 1008x1008 resize.
  Tested directly on the hardest schooling-fish cluster; still missed even at
  large zoom (see `KNOWN_LIMITATIONS.md`).
- **IoU+Kalman tracking as the primary tracker** (would-be
  `track_active_reprompt.py`, `track_full_clip_validation.py`) — a
  from-scratch SORT-style tracker over independent per-frame WBF detections,
  later extended with active re-prompting on tracking gaps.
  `track_full_clip_validation.py` found the single most reliably-detected
  fish in the clip still fragmented into 3 separate track IDs under pure
  Kalman-coasting. Superseded entirely by the native-SAM3-session + chunking
  approach above, which held identity far better without a hand-rolled
  tracker. `scripts/track_flicker_smoothing.py` (the base Kalman/IoU tracker these
  build on) *is* kept, but only as plumbing `temporal_stability_filter.py`
  reuses to get per-track box histories — not as a tracker in its own right.
- **"High variance positively confirms a real fish" (the flipped
  hypothesis)** (would-be `diagnostics/chunked_tracker_stability_check.py`,
  `_masked.py`, `_video.py`) — the temporal-stability filter's validated rule
  is the negative direction ("barely changes at all" ⇒ probably not a fish).
  This tested whether the flip also holds. Expected to be weaker going in —
  underwater footage has non-fish motion sources (light caustics, camera
  drift, mask-boundary noise on complex coral texture) — and the result was
  exactly that: mostly confirmed real fish, but with enough ambiguous/fading
  cases that it wasn't adopted as a standalone rule.
- **Native-tracker-vs-Kalman as a separate stats pipeline** (would-be
  `diagnostics/native_tracker_stability.py`, `_tuned.py`, `_video.py`,
  `render_native_tracker_video_smallfish.py`) — `native_tracker_stability.py`
  and `_tuned.py` are identical except which cached lifespan file they read;
  both (plus `_video.py`) depend on `outputs/diagnostics/track_lifespans*.json`
  from a lifespan-measurement script that isn't included, so they aren't
  runnable as-is. `render_native_tracker_video_smallfish.py` is the same
  script as `render_native_tracker_video.py` with the prompt swapped to
  `"small fish"` (29 vs 24 objects found, comparable 31% vs 42% full-clip
  survival) — a real but minor variant, not worth a second file.
- **CLAHE preprocessing on the two-prompt ensemble specifically** (would-be
  `diagnostics/clahe_ensemble_quickcheck.py`) — narrower version of the same
  question `ensemble_wbf_clahe.py` answers; consistent with the single-prompt
  regression already documented in `KNOWN_LIMITATIONS.md`, not a distinct
  result.
- **Deep-dive into the stability-score cutoff** (would-be
  `diagnostics/temporal_stability_deep_dive.py`) — the exploratory pass that
  arrived at the refined red/orange/green rule (plain score alone wasn't
  reliable: a low-scoring track could still look like a real fish holding
  still briefly, and vice versa). Its conclusion is already built into
  `temporal_stability_video.py`'s coloring rule, so the exploration itself
  doesn't need to ship separately.
- **RVRT multi-frame video deblurring** (would-be
  `diagnostics/rvrt_deblur_diagnostic.py`) — tests whether fusing information
  across frames (rather than single-frame deblurring, which can't add
  information a single exposure never captured) recovers structure in the
  blurred schooling-fish cluster. Excluded because it requires a separate
  external clone (`JingyunLiang/RVRT`, patched for Python 3.12 —
  `distutils.version.LooseVersion` → `packaging.version.Version` in
  `models/network_rvrt.py` and `models/op/deform_attn.py`) plus its own
  pretrained checkpoints and a CUDA toolkit with `cuda_runtime.h` for its
  JIT-compiled `deform_attn` extension. Not something to vendor into this
  repo; reproduce by cloning RVRT, applying that patch, and pointing
  `CUDA_HOME` at a full toolkit.
- **NMS/detection-threshold sweeps and per-track lifespan measurement**
  (would-be `diagnostics/sweep_det_nms_thresh.py`, `sweep_new_det_thresh.py`,
  `track_lifespans*.py`) — the parameter search that arrived at the
  thresholds baked into the scripts kept above; the sweep scripts themselves
  aren't needed to reproduce the result, only their conclusion is.
- **Point re-prompt recovery** (would-be
  `diagnostics/point_reprompt_recovery.py`) — SAM3 native point re-prompting
  as one-shot recovery for a fish RF-DETR (from a separate, unrelated
  project) kept detecting that SAM3's video tracker lost. Recovery held for
  roughly 6 frames before dropping again — a limited, not-production-worthy
  result.
- **Crop-tagging / reranker training pipeline** (would-be
  `diagnostics/dump_crops.py`, `tag_crops_server.py`, `train_reranker.py`) —
  an in-progress detection-reranking approach, not yet a validated part of
  the pipeline.
- **Miscellaneous one-off checks** (`cotracker_school_diagnostic.py`,
  `moving_object_native_check.py`, `moving_object_prompt_quickcheck.py`,
  `image_vs_video_frame0.py`, `single_image_threshold_check.py`,
  `verify_video_threshold_fix.py`, `score_report.py`,
  `flicker_gap_diagnostic.py`, `flicker_rescue_test.py`) — exploratory
  scripts from the same investigation that didn't produce a result worth
  keeping in the pipeline.
