"""Patches Sam3VideoPredictor so its detection thresholds actually take effect.

Import this once, before building any video predictor
(`from sam3.model_builder import build_sam3_video_predictor`) -- the patch
applies at import time as a side effect. Every script in this repo that
calls `build_sam3_video_predictor` imports this first.

Why this is needed (upstream SAM3, as of this writing, has these two bugs):

1. output_prob_thresh silently does nothing. Sam3VideoInferenceWithInstance-
   Interactivity.add_prompt has no output_prob_thresh parameter, so
   Sam3BasePredictor.add_prompt's kwarg-filtering silently drops it before
   it reaches the model. The real gate is self.model.score_threshold_
   detection (read inside run_backbone_and_detection), so this patch sets
   that directly whenever add_prompt/propagate_in_video is called with a
   requested threshold.
2. new_det_thresh / det_nms_thresh aren't part of the request schema.
   new_det_thresh ("prob threshold for a detection to be added as a new
   object", sam3_video_base.py Sam3VideoBase.__init__) is a separate gate
   from score_threshold_detection/output_prob_thresh above: it decides
   whether a detection that already passed the output filter is allowed to
   become a *new* tracked masklet. Defaults to 0.7, the value baked into
   build_sam3_video_model. det_nms_thresh is the mask-IoU threshold NMS
   suppresses detections at (run_backbone_and_detection ->
   forward_video_grounding_multigpu -> nms_masks in sam3/perflib/nms.py --
   note this is mask IoU, not box IoU, despite the generic-sounding name).
   Defaults to 0.1, also baked into build_sam3_video_model. Setting it to
   0.0 disables NMS entirely (run_nms = det_nms_thresh > 0.0 in
   sam3_video_base.py). This patch routes both from the request dict at
   dispatch time. Omitting either key changes nothing.

Scoped to Sam3VideoPredictor, not Sam3VideoPredictorMultiGPU: every script
here calls build_sam3_video_predictor(gpus_to_use=[0]) (single GPU), so
world_size is always 1 and Sam3VideoPredictorMultiGPU never spawns worker
subprocesses -- it just calls straight through to the patched methods below
via its own handle_request/handle_stream_request overrides (both of which
end in `return/yield from super().handle_request(...)`). If you ever call
build_sam3_video_predictor with more than one GPU, the worker subprocesses
it spawns are separate Python processes that re-import
sam3.model.sam3_video_predictor fresh and would NOT see this patch unless
they also import this module first.
"""

from sam3.model.sam3_video_predictor import Sam3VideoPredictor

_original_add_prompt = Sam3VideoPredictor.add_prompt
_original_propagate_in_video = Sam3VideoPredictor.propagate_in_video
_original_handle_request = Sam3VideoPredictor.handle_request
_original_handle_stream_request = Sam3VideoPredictor.handle_stream_request


def _patched_add_prompt(self, *args, output_prob_thresh: float = 0.5, **kwargs):
    self.model.score_threshold_detection = output_prob_thresh
    return _original_add_prompt(self, *args, output_prob_thresh=output_prob_thresh, **kwargs)


def _patched_propagate_in_video(self, *args, output_prob_thresh: float = 0.5, **kwargs):
    self.model.score_threshold_detection = output_prob_thresh
    yield from _original_propagate_in_video(
        self, *args, output_prob_thresh=output_prob_thresh, **kwargs
    )


def _patched_handle_request(self, request):
    if request["type"] == "add_prompt":
        self.model.new_det_thresh = request.get("new_det_thresh", 0.7)
        self.model.det_nms_thresh = request.get("det_nms_thresh", 0.1)
    return _original_handle_request(self, request)


def _patched_handle_stream_request(self, request):
    if request["type"] == "propagate_in_video":
        self.model.new_det_thresh = request.get("new_det_thresh", 0.7)
        self.model.det_nms_thresh = request.get("det_nms_thresh", 0.1)
    yield from _original_handle_stream_request(self, request)


Sam3VideoPredictor.add_prompt = _patched_add_prompt
Sam3VideoPredictor.propagate_in_video = _patched_propagate_in_video
Sam3VideoPredictor.handle_request = _patched_handle_request
Sam3VideoPredictor.handle_stream_request = _patched_handle_stream_request
