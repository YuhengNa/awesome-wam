# Repository Guidelines

## Project Structure & Module Organization

This repository contains WAM planning materials plus a local reference link to FastWAM.

- `wam/plan.md`: implementation roadmap, target architecture, phases, benchmarks, and configuration examples.
- `wam/story.md`: project motivation, related WAM papers, design-space rationale, and positioning relative to StarVLA.
- `wam.zip`: archive containing the same `wam/` documentation files.
- `external/FastWAM`: symlink to `/data/LFT-W02_data/junjie/VLA_WM/FastWAM`, the local FastWAM development repository used as the current code base.

Future implementation should follow the structure proposed in `wam/plan.md`: `core/` for pluggable WAM abstractions, `action_decoder/` for StarVLA-compatible action heads, `data/` for dataset loading, `eval/` for benchmark code, `configs/` for experiment definitions, and `docs/` for design notes.

## Build, Test, and Development Commands

Current repository-level commands are documentation and reference-code checks:

- `ls wam`: list project planning files.
- `sed -n '1,160p' wam/plan.md`: review the implementation roadmap.
- `sed -n '1,160p' wam/story.md`: review project motivation and design context.
- `ls external/FastWAM`: inspect the linked FastWAM code tree.
- `git -C external/FastWAM status -sb`: check local FastWAM changes before syncing or training.

Develop locally in `external/FastWAM` and sync to the training server when experiments are ready. Do not place checkpoints, datasets, or run outputs in this repository root.

## Coding Style & Naming Conventions

Preserve the terminology already established in the planning docs: `WorldEncoder`, `WorldBranch`, `DataStrategy`, `Backbone`, and `ActionDecoder`. Use PascalCase for class-like module abstractions, snake_case for Python files and config keys, and lowercase directory names such as `core/world_encoder/` and `core/world_branch/`.

Configuration examples should use YAML and keep option names aligned with `wam/plan.md`, for example `backbone.freeze`, `world_branch.train_video_loss`, and `attention_topology.action_sees_future`.

## Testing Guidelines

No testing framework is configured yet. Once implementation begins, add focused unit tests for config validation and module interfaces, then integration tests for training/evaluation entry points. Name Python tests `test_*.py` and keep benchmark-heavy tests separate from fast CI checks.

## Commit & Pull Request Guidelines

This workspace does not include Git history, so no repository-specific commit convention can be inferred. Use concise imperative commit subjects, such as `Add WAM config validation` or `Implement DINO world encoder`.

Pull requests should include a short purpose statement, changed paths, validation commands run, and links to the relevant phase or design section in `wam/plan.md`. Include benchmark tables or logs for training, evaluation, or model-behavior changes.
