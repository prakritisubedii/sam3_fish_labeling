# Known limitations — SAM3 fish detection (production recipe: `scripts/ensemble_wbf.py`)

## Dense, tightly-schooling, motion-blurred fish are not detected

**What**: the production detection recipe (SAM3 prompted with `"fish"` and
`"small fish"`, fused with weighted box fusion — see `scripts/ensemble_wbf.py`) does
not detect individual fish within a dense, tightly-packed school when the
fish are small and motion-blurred. This is expected, not a bug to chase.

**Where confirmed**: `assets/fish_data/clip1.mp4`, frame 0, roughly
`x∈[800,1080], y∈[330,480]` in the 1920x1280 source. One large fish in that
same cluster *is* detected reliably (~0.82 confidence); it's specifically the
surrounding loose school of small, pale fish that's missed. Likely present at
other similar schooling moments in this and other clips too, not just this
one frame.

Visual evidence (already in the repo):
- `results/diagnostic_tiling/frame0_cluster_zoom.png` — 4x zoomed
  crop of the region. The school is visible to the eye as faint, blurry
  streaks around the one large, clearly-detected fish.
- `results/diagnostic_tiling/frame0_tightcrop_detected.png` — SAM3
  run on a tight, heavily-zoomed crop of just this region. Still only finds
  the one large fish; the school is still missed even with far more relative
  resolution per fish than the model sees in the full frame.

**Why this isn't a quick fix — what was ruled out** (all tested this
session, zero-shot only, no fine-tuning data available):
- Confidence threshold (0.5 / 0.3 / 0.2 for a fixed prompt gave *identical*
  results — not a thresholding issue)
- Prompt wording (`"small fish"` helps broadly but not this cluster;
  `"fish near coral"` made things worse)
- CLAHE / underwater contrast-correction preprocessing (no improvement,
  slightly worse overall)
- Targeted tiling/cropping for resolution — tested directly on this exact
  cluster, still missed even at large zoom (see the two images above)
- SAM3 ensemble + WBF (current production recipe) — still missed

Ruling out resolution and prompt-coverage leaves the most likely explanation
as genuine motion blur (the school moving during the frame's exposure) plus
video compression, i.e. an image-quality ceiling rather than something a
prompting or preprocessing trick can undo.

**What this means going forward**: don't re-chase this with more
prompt/preprocessing experiments without new data — that ground has been
covered. The real fix is almost certainly fine-tuning on labeled examples of
this failure mode, which is out of scope until more annotated data exists.
Until then, expect annotators to manually box/correct dense schooling
clusters in Label Studio; the automated recipe handles everything else.

**Related, separate finding**: background-subtraction-based candidate
generation (`stage1_bg_subtraction.py`) was tried and abandoned as the
detection strategy for an unrelated reason — this footage's moving camera
over complex 3D coral breaks per-pixel motion/background assumptions
structurally. See that file's docstring for details. Not the same issue as
the schooling-fish case above (that one is about *detecting on a single
frame*; this one is about *motion between frames*), but both stem from the
same underlying footage being genuinely hard.
