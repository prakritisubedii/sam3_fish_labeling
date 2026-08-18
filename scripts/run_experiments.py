"""
Sweep SAM3 video segmentation over (text prompt, output_prob_thresh) combos to see
which settings recover more of the small/occluded fish that get missed at defaults.

Each combo's results go into their own named folder under outputs/, e.g.:
    outputs/thresh0.5_prompt-fish/output.mp4
    outputs/thresh0.5_prompt-fish/run_info.json

Usage: python run_experiments.py
"""

import json
import time

from sam3.model_builder import build_sam3_video_predictor

import sam3_predictor_patch  # noqa: F401 -- applies threshold-routing patch before building the predictor
from experiment_common import OUTPUT_ROOT, render_video, slugify, summarize

VIDEO_PATH = "assets/fish_data/clip1.mp4"

# Each entry: (prompt_text, output_prob_thresh)
EXPERIMENTS = [
    ("fish", 0.5),
    ("fish", 0.3),
    ("fish", 0.2),
    ("small fish", 0.3),
    ("fish near coral", 0.3),
]


def run_name_for(prompt: str, threshold: float) -> str:
    return f"thresh{threshold}_prompt-{slugify(prompt)}"


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    predictor = build_sam3_video_predictor(gpus_to_use=[0])

    results_index = []

    for prompt, threshold in EXPERIMENTS:
        run_name = run_name_for(prompt, threshold)
        run_dir = OUTPUT_ROOT / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running: prompt={prompt!r} threshold={threshold} -> {run_dir} ===")

        t0 = time.time()
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=VIDEO_PATH)
        )
        session_id = response["session_id"]

        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt,
                output_prob_thresh=threshold,
            )
        )
        frame0_detections = len(response["outputs"]["out_obj_ids"])
        print(f"Frame 0: {frame0_detections} detections for prompt {prompt!r} @ thresh={threshold}")

        outputs_per_frame = {}
        for out in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                output_prob_thresh=threshold,
            )
        ):
            outputs_per_frame[out["frame_index"]] = out["outputs"]

        predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        elapsed = time.time() - t0

        video_out_path = run_dir / "output.mp4"
        render_video(VIDEO_PATH, outputs_per_frame, video_out_path)

        summary = summarize(outputs_per_frame)
        run_info = {
            "run_name": run_name,
            "prompt": prompt,
            "output_prob_thresh": threshold,
            "video_path": VIDEO_PATH,
            "elapsed_seconds": round(elapsed, 1),
            "frame0_detections": frame0_detections,
            **summary,
        }
        with open(run_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

        print(
            f"Done in {elapsed:.1f}s | unique objects tracked: {summary['num_unique_objects']} "
            f"| avg detections/frame: {summary['avg_detections_per_frame']:.2f} "
            f"| avg confidence: {summary['avg_confidence']:.3f}"
        )
        results_index.append(run_info)

    predictor.shutdown()

    with open(OUTPUT_ROOT / "index.json", "w") as f:
        json.dump(results_index, f, indent=2)

    print("\n=== Summary across all runs ===")
    for r in results_index:
        print(
            f"{r['run_name']:35s} objects={r['num_unique_objects']:3d} "
            f"avg_det/frame={r['avg_detections_per_frame']:.2f} "
            f"avg_conf={r['avg_confidence']:.3f} ({r['elapsed_seconds']}s)"
        )


if __name__ == "__main__":
    main()
