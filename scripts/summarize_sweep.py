#!/usr/bin/env python3
"""Summarize a ts_lora sweep folder into CSV and Markdown.

Usage:
  python scripts/summarize_sweep.py /path/to/sweep_dir

Notes:
  - No external deps. Parses a small set of fields via regex.
  - Assumes each run has config.yaml and out.txt.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, Optional


CONFIG_FIELDS = [
    "lora_lr",
    "lora_rank",
    "lora_alpha",
    "iterations",
    "steps_per_iteration",
    "train_epochs",
    "batch_size",
]


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_config(text: str) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {k: None for k in CONFIG_FIELDS}
    for key in CONFIG_FIELDS:
        m = re.search(rf"^\s*{re.escape(key)}:\s*([^#\n]+)", text, re.M)
        if m:
            out[key] = m.group(1).strip()
    return out


def extract_eval(text: str) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {
        "reward_mean": None,
        "reward_std": None,
        "forward_progress_mean": None,
        "forward_progress_std": None,
        "k1": None,
        "k2": None,
        "k5": None,
        "k10": None,
        "k15": None,
        "k1_std": None,
        "k2_std": None,
        "k5_std": None,
        "k10_std": None,
        "k15_std": None,
    }

    # Evaluation summary metrics
    for key in ["reward_mean", "reward_std", "forward_progress_mean", "forward_progress_std"]:
        m = re.search(rf"^-\s*{key}:\s*([0-9.]+)", text, re.M)
        if m:
            out[key] = m.group(1)

    # Final eval K-step RMSE table (lora row)
    # Example line:
    # lora  |   0.6507 ±   0.0304 |   0.8753 ±   0.0644 |   ...
    m = re.search(
        r"^lora\s*\|\s*([0-9.]+)\s*±\s*([0-9.]+)\s*\|\s*"
        r"([0-9.]+)\s*±\s*([0-9.]+)\s*\|\s*"
        r"([0-9.]+)\s*±\s*([0-9.]+)\s*\|\s*"
        r"([0-9.]+)\s*±\s*([0-9.]+)\s*\|\s*"
        r"([0-9.]+)\s*±\s*([0-9.]+)",
        text,
        re.M,
    )
    if m:
        out["k1"], out["k1_std"] = m.group(1), m.group(2)
        out["k2"], out["k2_std"] = m.group(3), m.group(4)
        out["k5"], out["k5_std"] = m.group(5), m.group(6)
        out["k10"], out["k10_std"] = m.group(7), m.group(8)
        out["k15"], out["k15_std"] = m.group(9), m.group(10)

    return out


def run_status(text: str) -> str:
    has_train_done = "Training finished" in text
    has_eval = "Evaluation summary" in text
    if has_train_done and has_eval:
        return "complete"
    if "Traceback" in text or "Error" in text:
        return "crashed"
    if has_train_done and not has_eval:
        return "ambiguous"
    return "incomplete"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a sweep folder")
    parser.add_argument("sweep_dir", help="Path to sweep directory")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    runs = []

    for name in sorted(os.listdir(sweep_dir)):
        run_dir = os.path.join(sweep_dir, name)
        if not os.path.isdir(run_dir):
            continue
        cfg_path = os.path.join(run_dir, "config.yaml")
        out_path = os.path.join(run_dir, "out.txt")
        if not os.path.exists(cfg_path) or not os.path.exists(out_path):
            runs.append({"folder": name, "status": "malformed"})
            continue

        cfg_text = read_text(cfg_path)
        out_text = read_text(out_path)

        cfg = extract_config(cfg_text)
        ev = extract_eval(out_text)
        status = run_status(out_text)

        row = {"folder": name, "status": status}
        row.update(cfg)
        row.update(ev)
        runs.append(row)

    # Markdown summary to stdout
    headers = [
        "folder",
        "status",
        "lora_lr",
        "lora_rank",
        "lora_alpha",
        "iterations",
        "steps_per_iteration",
        "reward_mean",
        "reward_std",
        "forward_progress_mean",
        "forward_progress_std",
        "k1",
        "k2",
        "k5",
        "k10",
        "k15",
    ]

    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in runs:
        print("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")

    # Optional CSV output
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(runs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
