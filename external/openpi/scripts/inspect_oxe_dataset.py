#!/usr/bin/env python
"""Inspect an OXE / RLDS-style dataset tree before wiring PV-VAE training.

This script is intentionally read-only. It prints:

- top-level directory layout
- likely TFDS/RLDS dataset directories
- common data file counts by suffix
- optional TensorFlow Datasets feature/split metadata
- optional first-sample nested keys, shapes, and dtypes

Run it on the training server, for example:

    python external/openpi/scripts/inspect_oxe_dataset.py \
      --root /data/user/jhe724/workspace/data/OXE \
      --try-tfds \
      --peek-sample
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any


COMMON_SUFFIXES = {
    ".tfrecord",
    ".parquet",
    ".mp4",
    ".json",
    ".jsonl",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="OXE root directory.")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-entries", type=int, default=120)
    parser.add_argument("--try-tfds", action="store_true", help="Try reading TFDS metadata from dataset_info.json dirs.")
    parser.add_argument("--peek-sample", action="store_true", help="Load and summarize one sample per TFDS dataset.")
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def print_tree(root: Path, max_depth: int, max_entries: int) -> None:
    print("\n== Directory sample ==")
    printed = 0
    for current, dirs, files in os.walk(root):
        path = Path(current)
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        dirs[:] = sorted(dirs)[:max_entries]
        files = sorted(files)[:8]
        indent = "  " * depth
        name = "." if path == root else path.name
        print(f"{indent}{name}/")
        for file_name in files:
            print(f"{indent}  {file_name}")
        printed += 1
        if printed >= max_entries:
            print(f"... truncated after {max_entries} directories")
            break


def collect_file_stats(root: Path) -> tuple[Counter[str], list[Path]]:
    suffix_counts: Counter[str] = Counter()
    tfds_info_dirs: list[Path] = []
    for current, _, files in os.walk(root):
        path = Path(current)
        if "dataset_info.json" in files:
            tfds_info_dirs.append(path)
        for file_name in files:
            suffix = Path(file_name).suffix
            if suffix in COMMON_SUFFIXES or file_name.endswith(".tfrecord-00000-of-00001"):
                suffix_counts[suffix or ".tfrecord-shard"] += 1
    return suffix_counts, sorted(tfds_info_dirs)


def summarize_value(value: Any, depth: int = 0, max_depth: int = 4) -> Any:
    if depth >= max_depth:
        return "..."
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        return {"shape": tuple(shape) if shape is not None else None, "dtype": str(dtype)}
    if isinstance(value, Mapping):
        return {str(k): summarize_value(v, depth + 1, max_depth) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(summarize_value(v, depth + 1, max_depth) for v in value[:8])
    if isinstance(value, list):
        return [summarize_value(v, depth + 1, max_depth) for v in value[:8]]
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return type(value).__name__
    if isinstance(value, Sequence):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__


def try_tfds(tfds_dirs: list[Path], *, peek_sample: bool) -> None:
    print("\n== TFDS/RLDS metadata ==")
    try:
        import tensorflow_datasets as tfds  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on server env
        print(f"tensorflow_datasets import failed: {type(exc).__name__}: {exc}")
        return

    for dataset_dir in tfds_dirs:
        print(f"\n-- {dataset_dir} --")
        try:
            builder = tfds.builder_from_directory(str(dataset_dir))
            print(f"name={builder.info.name} version={builder.info.version}")
            print(f"splits={builder.info.splits}")
            print(f"features={builder.info.features}")
            if peek_sample:
                split_names = list(builder.info.splits.keys())
                split = split_names[0] if split_names else "train"
                ds = builder.as_dataset(split=f"{split}[:1]")
                for sample in tfds.as_numpy(ds.take(1)):
                    print("sample_summary=")
                    print(summarize_value(sample))
                    break
        except Exception as exc:  # pragma: no cover - depends on data
            print(f"failed: {type(exc).__name__}: {exc}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    print(f"root={root}")
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    print_tree(root, args.max_depth, args.max_entries)
    suffix_counts, tfds_dirs = collect_file_stats(root)

    print("\n== File suffix counts ==")
    for suffix, count in suffix_counts.most_common():
        print(f"{suffix}: {count}")

    print("\n== Candidate TFDS/RLDS dirs ==")
    for path in tfds_dirs[: args.max_entries]:
        print(rel(path, root))
    if len(tfds_dirs) > args.max_entries:
        print(f"... {len(tfds_dirs) - args.max_entries} more")

    if args.try_tfds:
        try_tfds(tfds_dirs, peek_sample=args.peek_sample)


if __name__ == "__main__":
    main()
