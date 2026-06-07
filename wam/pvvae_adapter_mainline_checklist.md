# PV-VAE Adapter Mainline Checklist

This document is the persistent progress anchor for the SVG-P/DINO + PV-VAE
adapter work. At the start of each research/debug discussion, restate this
checklist and update the status before branching into details.

## Research Position

The target is WAM / FastWAM-style video-action generation, not OpenPI VLA
policy training. OpenPI scripts are only used as utilities for loading robot
clips, extracting frozen semantic features, training the adapter, and running
diagnostics.

Current representation path:

```text
RGB robot clip
  -> frozen semantic teacher encoder, such as SVG-P / DINO
  -> semantic feature clip [B,V,T,N,D]
  -> PV-VAE temporal compression adapter
  -> compressed semantic latent groups [B,V,G,N,d]
  -> WAM / FastWAM-style video-action generation target
```

PV-VAE is an adapter/tokenizer over semantic features. Its decoder is used for
pretraining diagnostics and visualization; it is not assumed to be the final
world-model decoder unless a later paper-grounded design explicitly uses it.

## Mainline Tasks

- [x] OXE/Bridge single-source PV-VAE adapter smoke tests.
- [x] Feature normalization and visualization sanity checks.
- [x] OXE main-view mixed baseline trained to 20k steps, showing structure and
  some temporal dynamics.
- [x] PV-VAE latent export contract dry run:
  `[B,V,G,N,d] -> [B,d,G,H_packed,W_packed]`.
- [ ] Audit OXE / DROID / Behavior dataset formats and camera views:
  main/external view, wrist view, missing or placeholder view ratio, decode
  success, black/flat ratio, motion ratio, FPS, and resolution.
- [ ] Build a multi-dataset main-view training spec.
- [ ] Train the OXE + DROID + Behavior main-view PV-VAE adapter baseline.
- [ ] Run a fixed `observed_groups=4` ablation.
- [ ] Run held-out evaluation matrix:
  checkpoint x `observed_groups=3/4/5`, with per-horizon feature metrics and
  visualization.
- [ ] Move from main-view-only to multi-view adapter training after the
  main-view baseline is stable.
- [ ] Integrate compressed semantic latents into the WAM / FastWAM-style
  video-action generation path.

## Current Next Step

Continue Stage 1 of `wam/pvvae_multidataset_execution_plan.md`: dataset and
view audit for OXE, DROID, and Behavior-1K. Do not start mixed training until
the valid main/external views and dataset-specific FPS/layouts are confirmed.

## Guardrails

- Do not reinterpret this work as OpenPI/VLA training.
- Do not jump to expensive FastWAM training before adapter quality and latent
  contract checks pass.
- Do not treat wrist-only results as evidence that the main-view adapter is
  ready.
- Do not mix datasets silently if their view roles, FPS, or missing-view
  behavior are unknown.
- Use decoded RGB visualizations only as diagnostics for semantic feature
  quality; the primary training/evaluation target is the compressed semantic
  feature representation.

