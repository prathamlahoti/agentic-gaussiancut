# Agentic GaussianCut

A research pipeline for selecting additional views, validating SAM object masks, and removing an object from a pretrained 3D Gaussian Splatting scene with GaussianCut.

## Status

Development code. The revised three-target experiment and independent GPU evaluation have completed. All six edits exported valid paired renders; 240 render/depth/alpha file hashes were checked. Removal is visibly stronger than the initial footstool pilot, but blur and small remnants remain. VLM selection did not consistently beat the fixed-order baseline. This is not a state-of-the-art claim or the full physical Remove360 benchmark.

## Pipeline

1. Input a pretrained 3DGS, known camera poses, and one target mask.
2. Render candidate previews and project the initial target into each camera.
3. A VLM selects a view. GroundingDINO proposes boxes and SAM ViT-H segments the target.
4. The harness checks target coverage, reverse projection consistency, mask area, and clipping.
5. GaussianCut deletes target Gaussians using accepted masks.
6. Evaluate paired held-out renders using the Remove360 reference-free Mip-NeRF360 protocol.

The fixed-order baseline uses the same tools and maximum request budget. Candidate previews are already rendered. There is no physical robot movement or RL.

## Setup

Python 3.10+ and Git are needed for packaging. Execution requires an NVIDIA CUDA environment; the provided scripts package private Kaggle GPU jobs. Authenticate Kaggle locally; never commit credentials.

```bash
pip install -r requirements.txt
python setup_sources.py
python test_geometry.py
python test_benchmark_metrics.py
```

Set `KAGGLE_USERNAME` to your Kaggle username, then:

```bash
python build_v2.py
```

The builder prints a temporary job folder. Submit it using `kaggle kernels push -p <printed-folder>`. After it completes, run `python build_evaluation.py` and submit its printed folder in the same way. The evaluator references your `agentic-gaussiancut-v2` output and the original `agentic-gaussiancut-room-pilot`; remove the latter source in `build_evaluation.py` if you do not have that optional earlier run.

Download outputs into `v2_output/` and `evaluation_output/`, preserving their `experiment/` and `evaluation/` folders. Run `python present_v2.py` to produce a self-contained HTML report. No weights, scans, credentials, logs, or unrelated workspace files are included.

## Examples and evaluation

The declared development cases are room footstool, room coffee table, and counter plant in `v2_config.json`. The latter two occur in Remove360's Mip-NeRF360 experiments. Seed boxes for new cases produce initial SAM masks that require visual review.

This is a small reference-free development subset, not the physical Remove360 dataset benchmark. Depth export uses our declared adapter, and reference masks are SAM pseudo-labels. See [the complete protocol](V2_PROTOCOL.md) for thresholds, held-out views, interpretation, and tested compatibility fixes.

## Research code and attribution

- [GaussianCut](https://github.com/umangi-jain/gaussiancut): native graph-cut editing, pinned revision in `removal_pipeline/upstream_lock.json`.
- [Segment Anything](https://github.com/facebookresearch/segment-anything): SAM predictor and automatic masks.
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2): evaluation probe inference sequence.
- [Remove360](https://github.com/simonakocour/anything_left) and [paper](https://arxiv.org/abs/2508.11431): evaluation functions, fetched separately at a pinned revision.
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct): action selection.
- 3DGSNav and AREA3D inform the agent/tool architecture; their code is not imported.

The action harness, projection checks, adapters, and experiment packaging are project code. Upstream code, pretrained weights, and datasets retain their respective licenses. Consult those licenses before use or redistribution. No blanket license over upstream dependencies is implied.

## Measured development results

[Per-target metrics](results/pilot_metrics.csv) cover three held-out views per target and policy, including flagged partial targets. They use SAM pseudo-reference masks and the declared depth adapter. A separate complete-view summary is available in the generated report. One saved SAM segment was merged into the plant reference to include its visible white pot, equally for both policies; raw GPU scores remain separate. `review_benchmark.py` records this specific review step.

The initial footstool edit and revised edit both escaped the semantic detector. Their depth-change scores differed (37.0% versus 82.6%), as did structural change (34.0% versus 87.1%). Thus semantic non-detection alone did not establish successful removal. Neither change metric establishes physical correctness.
