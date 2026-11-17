# Agentic object removal: reproducible development study

## Question

Starting with a 3DGS scene and one target mask, can a VLM collect useful additional masks for GaussianCut? Does the result improve when the harness rejects masks that do not agree with the initial target?

This is a development experiment. It does not establish a new algorithm or a general advantage of VLM view selection.

## Fixed cases and comparison

The configuration was written before the new removal outputs were available. It contains three targets in two pretrained Mip-NeRF360 scenes: the room footstool, the room coffee table, and the counter plant. Coffee table and counter plant are targets in Remove360 Appendix I. The footstool is the existing failure case retained for continuity.

Each policy starts from the same mask and the same eligible camera pool. Each has a maximum of four additional SAM requests. Fixed order requests ascending camera IDs. The VLM sees candidate thumbnails and accepted mask overlays, and selects a camera or, after two additional accepted masks, stop. Both policies use the same segmentation and validation tools. Compare actual requests as well as final scores: early stopping can result in unequal numbers of requests even though the maximum budget is identical.

Three evaluation cameras per target are specified in `v2_config.json`. They never enter policy decisions or GaussianCut input masks. These are held out from **editing**, not from the original 3DGS training.

The footstool uses the original hand-drawn mask. The two other seed masks are produced by SAM from manually specified boxes and must be visually reviewed. They simulate the supplied initial-mask interface; they are not a demonstration of automatic target discovery.

## Pipeline

1. Load the official Graphdeco pretrained scene and known camera poses. Render RGB and alpha-normalized expected camera-space depth.
2. Back-project the initial masked pixels using depth. Project them into candidate cameras. Reject projections largely outside the frame or inconsistent with visible depth.
3. Ask the VLM to choose a camera. This is selection among already rendered thumbnail previews, not physical robot motion or selection of entirely unseen images.
4. Use GroundingDINO to propose category boxes, selecting only proposals that contain the projected target points. SAM ViT-H uses the box plus positive projected points. Fall back to a projected box when no detector box matches.
5. Check target-point coverage, reverse projection agreement with the initial mask, mask area, and image-boundary clipping. Accept or reject the mask and record the evidence. These gates are our harness, not a published GaussianCut component.
6. Run the released GaussianCut implementation on accepted masks, keeping its background partition. Use the Remove360 G.1 settings: foreground threshold 0.9, 10 graph neighbors, 5 foreground and 5 background clusters, leaf size 40. No inpainting.
7. Render the edited and original scenes at identical held-out poses. Run a separate evaluation job that cannot influence selection.

## Papers and code actually used

| Component | Source | Role |
|---|---|---|
| GaussianCut | https://github.com/umangi-jain/gaussiancut | Native mask lifting, graph cut and Gaussian deletion. Pinned revision `93d24a4ba02944ffae7848f9882b083cd9e04640`. |
| Segment Anything | https://github.com/facebookresearch/segment-anything | Official SAM ViT-H predictor and automatic mask generator. Revision in config. |
| GroundingDINO | https://huggingface.co/IDEA-Research/grounding-dino-tiny | Object-box proposals. Editing thresholds 0.30/0.25. |
| SmolVLM | https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct | Selects an allowed action using next-token letter scores. No RL or fine-tuning. |
| Grounded-SAM-2 | https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_hf_model_demo.py | Evaluation inference sequence: DINO tiny, thresholds 0.4/0.3, SAM2.1 large, full-image category prompt, no scoring-mask prompt. |
| Remove360 | https://github.com/simonakocour/anything_left | Released semantic/depth metrics and structural matching helpers. Revision `dd0aad895b4d4dce75bd9f2462ad36114e813bd6`. |
| 3DGSNav / AREA3D | https://arxiv.org/html/2602.12159v1 and https://arxiv.org/html/2512.05131 | References for separating a VLM action loop from geometric tools. Their navigation/reconstruction implementations are not imported here. |

## Benchmark scope and necessary compatibility details

The physical Remove360 release has photographs and masks but no trained Gaussian checkpoints or camera poses. Running that full benchmark requires reconstruction and registration. This study uses the paper's **Mip-NeRF360 reference-free protocol**, Appendix I, on a small declared subset. Do not compare the measured values directly with its full-data leaderboard.

Report pre- and post-removal target IoU, IoU drop, and the fraction of views with post-removal IoU below 0.3. Also report target-region depth modification and `1 - sim_SAM(pre, post)`. A low detector response before editing weakens the meaning of apparent suppression. Structural change is not physical correctness. Background damage must also be inspected.

The frozen full-frame evaluation category prompts are `footstool. ottoman.`, `table.`, and `potted plant.`. The table uses the broad category for the semantic probe rather than its longer editing description. These prompts are fixed before post-edit results are inspected. Detector and text thresholds are 0.4 and 0.3, following the released Grounded-SAM-2 HF example.

Reference masks are generated from original renders using SAM, independently of edited images, then visually reviewed. They are pseudo-reference masks, not physical ground truth. Keep failures and any partial-coverage flags in the report.

The declared held-out camera list contains partial targets, especially for the plant. The primary development summary retains supported visible target regions and flags clipped masks. A separate complete-view summary excludes those views, following the paper's G.3 coverage restriction more closely. If no complete view remains for a target, report that fact rather than substituting another camera after seeing the edited result. This partial-view pilot is a deviation from the paper's full protocol and must be labeled on the slides and report.

The depth renderer produces float scene-unit values. The adapter encodes each pair with one shared linear min/max map to 8-bit values, reserving zero for invalid pixels. The official preprocessing must use `save_intermediate=True` to preserve its inverse-grayscale convention. Without that option its default path reverses the interpretation. Identical-depth and known-changed-region tests verify the bridge. This encoding is explicitly our adapter, not an undocumented claim of exact author export parity.

The released structural evaluator skips empty retained mask sets. Paper G.4 defines both empty as similarity 1 and one empty as similarity 0. The bridge follows the paper for those cases, retains every evaluation view, and otherwise uses the released IoU/resize operations and the same Hungarian normalization.

Do not report the physical-reference RQS for this study. Report the separate axes and limitations.

## Execution and presentation

`build_v2.py` packages a private Kaggle job. `build_evaluation.py` packages an independent evaluator using its outputs. `test_geometry.py` and `test_benchmark_metrics.py` test pose round trips, wrong-object rejection, occlusion, depth-score orientation, and structural empty-set behavior.

The final report must be generated from downloaded run records. Show every target and both policies, accepted/rejected masks, before/after images, measured scores, actual request counts, and remaining failures. Never replace a failed edit with a synthetic or hand-retouched image.
