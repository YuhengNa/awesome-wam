# PV-VAE Multi-Dataset Scaling Plan

## Research Goal

Scale PV-VAE adapter training from Bridge to a larger collection of valid
main/external-view robot videos while preserving the current research line:

```text
RGB clips
  -> frozen SVG-P semantic encoder
  -> PV-VAE temporal compression adapter
  -> compressed semantic latent groups for the downstream WAM
```

PV-VAE is still evaluated as a semantic temporal compression adapter. The
multi-dataset work must not silently change it into a VLA policy or a standalone
future-generation model.

## Current Data Roots

```text
DROID:
/data/user/jhe724/workspace/data/droid_success

Behavior-1K:
/data/user/jhe724/workspace/data/2025-challenge-demos

OXE:
/data/user/jhe724/workspace/data/OXE
```

OXE currently includes at least Bridge and `fractal20220817`.

## Stage 1: Camera And Format Audit

Do not mix the datasets yet. First discover:

- actual dataset format and metadata layout
- camera/video keys
- candidate main/external/wrist view roles
- total files and successful decode ratio
- small or repeated placeholder-file ratio
- black/flat video ratio
- visible-motion ratio
- FPS, resolution, and representative contact sheets

Run:

```bash
PYTHONPATH=$PWD/external/openpi/src:$PYTHONPATH \
python external/openpi/scripts/inspect_robot_dataset_views.py \
  --root /data/user/jhe724/workspace/data/droid_success \
  --root /data/user/jhe724/workspace/data/2025-challenge-demos \
  --root /data/user/jhe724/workspace/data/OXE \
  --output-dir runs/dataset_view_audit/all_sources \
  --decode-videos-per-view 16 \
  --frames-per-video 8 \
  --contact-sheet-videos 4 \
  --small-file-threshold-kb 16 \
  --rgb-motion-threshold 0.02
```

The important outputs are:

```text
runs/dataset_view_audit/all_sources/dataset_view_audit.json
runs/dataset_view_audit/all_sources/view_summary.csv
runs/dataset_view_audit/all_sources/**/*.png
```

The script's `role_hint` is only a naming-based candidate. Confirm the final
view role from contact sheets before training.

If the combined scan is too slow, run one root at a time. Do not use
`--max-files` for the final count report because truncation can bias camera
coverage statistics.

## Stage 1 Decision Gate

A view can enter the first mixed-training experiment only if:

- its semantic role is visually confirmed as a main, head, or external view;
- its decode-success ratio is high;
- placeholder, black, and flat-video ratios are acceptably low or can be
  filtered reliably;
- its time sampling can be mapped to a common physical horizon.

Wrist views are not deleted. They are assigned a separate `view_role` and held
out from the first main-view-only scaling experiment.

## Stage 2: Unified Clip Contract

After Stage 1, implement one dataset adapter per actual storage format. Every
adapter must emit the same minimal contract:

```python
{
    "images": "[T,C,H,W]",
    "timestamps": "[T] or physical time offsets",
    "dataset_id": "str",
    "view_role": "str",
    "episode_uid": "globally unique str",
    "clip_start": "int",
    "valid_mask": "[T]",
}
```

Important:

- one sample uses one selected view; missing extra views must not invalidate it;
- `episode_uid` must include dataset and chunk identity;
- clips from different FPS datasets must align by seconds, not by identical
  frame-index deltas;
- dataset-specific parsing stays inside adapters, not inside the PV-VAE model.

## Stage 3: Mixture Sampling

Use a mixture sampler rather than physically concatenating all files.

The first controlled comparison should keep the PV-VAE architecture and loss
fixed and vary only the data mixture:

```text
Bridge main-view baseline
vs.
Bridge + Fractal main-view mixture
vs.
Bridge + Fractal + DROID/Behavior valid-main-view mixture
```

Start with balanced per-dataset sampling. Do not sample proportional to raw
dataset size until coverage and quality differences are understood.

## Metrics

Keep the current PV-VAE metrics:

- feature reconstruction MSE and cosine
- future feature MSE
- static-copy baseline gap
- predicted versus target delta magnitude
- SVG decoded-RGB and feature PCA visualization

Also report metrics by:

- dataset
- view role
- prediction horizon

This prevents a large or easy dataset from hiding failures on another dataset
or camera role.
