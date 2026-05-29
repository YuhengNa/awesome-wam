#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


TRAIN_RE = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+)/(?P<max_step>\d+).*?"
    r"loss=(?P<loss>[0-9.]+) loss_action=(?P<loss_action>[0-9.]+)\s+"
    r"loss_video=(?P<loss_video>[0-9.]+).*?"
    r"speed=(?P<speed>[0-9.]+) step/s, (?P<samples>[0-9.]+) samples/s",
    re.S,
)

EVAL_RE = re.compile(
    r"step=(?P<step>\d+) val_loss=(?P<val_loss>[0-9.]+).*?"
    r"feature_mse=(?P<feature_mse>[0-9.]+).*?"
    r"feature_cosine=(?P<feature_cosine>[0-9.]+) action_l2=(?P<action_l2>[0-9.]+)\s+"
    r"action_l1=(?P<action_l1>[0-9.]+) decoded_psnr=(?P<decoded_psnr>[0-9.]+)\s+"
    r"decoded_ssim=(?P<decoded_ssim>[0-9.]+)",
    re.S,
)


def parse_log(path: Path) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    text = path.read_text(errors="ignore")
    train = {}
    for match in TRAIN_RE.finditer(text):
        step = int(match.group("step"))
        train[step] = {
            "loss_video": float(match.group("loss_video")),
            "loss": float(match.group("loss")),
            "speed": float(match.group("speed")),
            "samples": float(match.group("samples")),
        }
    evals = {}
    for match in EVAL_RE.finditer(text):
        step = int(match.group("step"))
        evals[step] = {
            "val_loss": float(match.group("val_loss")),
            "feature_mse": float(match.group("feature_mse")),
            "feature_cosine": float(match.group("feature_cosine")),
            "decoded_psnr": float(match.group("decoded_psnr")),
            "decoded_ssim": float(match.group("decoded_ssim")),
        }
    return train, evals


def pct_delta(compact: float, full: float) -> float:
    if full == 0:
        return 0.0
    return (compact - full) / full * 100.0


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-log",
        type=Path,
        default=Path(
            "runs/libero_svg_dino_p_2cam256_future4_1e-4/_slurm_logs/fw_svgp_f4_10ep_b8_300362.out"
        ),
    )
    parser.add_argument(
        "--compact-log",
        type=Path,
        default=Path(
            "runs/libero_svg_dino_p_compact_2cam256_future4_1e-4/_slurm_logs/fw_svgpc_f4_10ep_b8_301745.out"
        ),
    )
    parser.add_argument("--last", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    full_train, full_eval = parse_log(args.full_log)
    compact_train, compact_eval = parse_log(args.compact_log)
    common_steps = sorted(set(full_eval) & set(compact_eval))
    rows = common_steps[-args.last :]

    lines = []
    lines.append("# Compact vs Full SVG-DINO-P Feature Loss Monitor")
    lines.append("")
    lines.append(f"- Full log: `{args.full_log}`")
    lines.append(f"- Compact log: `{args.compact_log}`")
    if compact_train:
        latest_train_step = max(compact_train)
        latest = compact_train[latest_train_step]
        lines.append(
            f"- Compact latest train step: `{latest_train_step}`, "
            f"loss_video=`{latest['loss_video']:.4f}`, speed=`{latest['speed']:.2f}` step/s"
        )
    if compact_eval:
        latest_eval_step = max(compact_eval)
        latest = compact_eval[latest_eval_step]
        lines.append(
            f"- Compact latest eval step: `{latest_eval_step}`, "
            f"feature_mse=`{latest['feature_mse']:.4f}`, "
            f"feature_cosine=`{latest['feature_cosine']:.4f}`, "
            f"decoded_psnr=`{latest['decoded_psnr']:.2f}`"
        )
    lines.append("")

    if not rows:
        lines.append("No common eval steps found.")
    else:
        lines.append(
            "| Step | full train loss_video | compact train loss_video | train delta | "
            "full eval feature_mse | compact eval feature_mse | eval delta | "
            "full cos | compact cos | full PSNR | compact PSNR |"
        )
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for step in rows:
            full_tv = full_train.get(step, {}).get("loss_video")
            compact_tv = compact_train.get(step, {}).get("loss_video")
            train_delta = ""
            if full_tv is not None and compact_tv is not None:
                train_delta = f"{compact_tv - full_tv:+.4f} ({pct_delta(compact_tv, full_tv):+.1f}%)"
            fe = full_eval[step]
            ce = compact_eval[step]
            eval_delta = f"{ce['feature_mse'] - fe['feature_mse']:+.4f} ({pct_delta(ce['feature_mse'], fe['feature_mse']):+.1f}%)"
            lines.append(
                f"| {step} | "
                f"{fmt(full_tv) if full_tv is not None else ''} | "
                f"{fmt(compact_tv) if compact_tv is not None else ''} | "
                f"{train_delta} | "
                f"{fmt(fe['feature_mse'])} | {fmt(ce['feature_mse'])} | {eval_delta} | "
                f"{fmt(fe['feature_cosine'])} | {fmt(ce['feature_cosine'])} | "
                f"{fe['decoded_psnr']:.2f} | {ce['decoded_psnr']:.2f} |"
            )

    report = "\n".join(lines) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
