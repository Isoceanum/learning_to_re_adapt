#!/usr/bin/env python3
"""Summarize TS-LoRA sweep runs from an outputs folder.

Parses config.yaml and out.txt with simple regexes (no external deps).
Prints a compact Markdown table by default.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Any, List

NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def read_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except FileNotFoundError:
        return None


def find_float(text: str, key: str) -> float | None:
    m = re.search(rf"^\s*(?:-\s*)?{re.escape(key)}\s*[:=]\s*({NUM_RE})", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def find_int(text: str, key: str) -> int | None:
    m = re.search(rf"^\s*(?:-\s*)?{re.escape(key)}\s*[:=]\s*(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def parse_config(text: str | None) -> Dict[str, Any]:
    if not text:
        return {}
    cfg: Dict[str, Any] = {}
    m = re.search(r"^\s*experiment_name:\s*(.+)$", text, re.MULTILINE)
    cfg["experiment_name"] = m.group(1).strip() if m else None
    for k in [
        "lora_lr",
        "lora_rank",
        "lora_alpha",
        "iterations",
        "steps_per_iteration",
        "train_epochs",
        "batch_size",
    ]:
        v = find_float(text, k) if k == "lora_lr" else find_int(text, k)
        cfg[k] = v
    return cfg


def parse_out(text: str | None) -> Dict[str, Any]:
    if not text:
        return {}
    out: Dict[str, Any] = {}
    out["has_eval_summary"] = "Evaluation summary" in text
    out["has_error"] = bool(re.search(r"(Traceback|Error:|Exception|CUDA error|Killed)", text))
    for k in [
        "reward_mean",
        "reward_std",
        "forward_progress_mean",
        "forward_progress_std",
        "episode_length_mean",
    ]:
        out[k] = find_float(text, k)

    blocks = list(
        re.finditer(
            r"K-step RMSE on eval rollouts \(real env steps\):\n(?:.*\n){0,5}?base\s*\|(.+)\n\s*lora\s*\|(.+)",
            text,
        )
    )
    if blocks:
        b = blocks[-1]
        base_line = b.group(1)
        lora_line = b.group(2)

        def parse_k_line(line: str) -> List[tuple[float, float]]:
            vals = re.findall(rf"({NUM_RE})\s*±\s*({NUM_RE})", line)
            return [(float(a), float(b)) for a, b in vals]

        base_vals = parse_k_line(base_line)
        lora_vals = parse_k_line(lora_line)
        ks = ["k1", "k2", "k5", "k10", "k15"]
        if len(base_vals) == 5:
            out["eval_rmse_base"] = {k: base_vals[i][0] for i, k in enumerate(ks)}
        if len(lora_vals) == 5:
            out["eval_rmse_lora"] = {k: lora_vals[i][0] for i, k in enumerate(ks)}
    return out


def format_float(v: float | None, nd: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def summarize(root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        cfg_text = read_file(os.path.join(path, "config.yaml"))
        out_text = read_file(os.path.join(path, "out.txt"))
        cfg = parse_config(cfg_text)
        out = parse_out(out_text)
        status = "complete" if out.get("has_eval_summary") and not out.get("has_error") else "incomplete"
        total_steps = None
        if cfg.get("iterations") and cfg.get("steps_per_iteration"):
            total_steps = cfg["iterations"] * cfg["steps_per_iteration"]
        rmse_base = out.get("eval_rmse_base", {})
        rmse_lora = out.get("eval_rmse_lora", {})

        def delta(k: str) -> float | None:
            if k in rmse_base and k in rmse_lora:
                return rmse_base[k] - rmse_lora[k]
            return None

        rows.append(
            {
                "name": name,
                "lr": cfg.get("lora_lr"),
                "rank": cfg.get("lora_rank"),
                "alpha": cfg.get("lora_alpha"),
                "iters": cfg.get("iterations"),
                "steps_per_iter": cfg.get("steps_per_iteration"),
                "total_steps": total_steps,
                "batch": cfg.get("batch_size"),
                "reward_mean": out.get("reward_mean"),
                "reward_std": out.get("reward_std"),
                "forward_mean": out.get("forward_progress_mean"),
                "forward_std": out.get("forward_progress_std"),
                "rmse_k1": rmse_lora.get("k1"),
                "rmse_k10": rmse_lora.get("k10"),
                "rmse_k15": rmse_lora.get("k15"),
                "delta_k1": delta("k1"),
                "delta_k10": delta("k10"),
                "status": status,
            }
        )
    return rows


def print_markdown(rows: List[Dict[str, Any]]) -> None:
    print(
        "| experiment | rank/alpha | lr | iters×steps (total) | batch | reward_mean ± std | forward_mean ± std | lora RMSE k1/k10/k15 | ΔRMSE k1/k10 (base-lora) | status |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lr = format_float(r["lr"], 6).rstrip("0").rstrip(".") if r["lr"] is not None else "—"
        print(
            "| {name} | {ra}/{aa} | {lr} | {iters}×{spi} ({tot}) | {batch} | {rm} ± {rs} | {fm} ± {fs} | {k1}/{k10}/{k15} | {d1}/{d10} | {status} |".format(
                name=r["name"],
                ra=r["rank"] if r["rank"] is not None else "—",
                aa=r["alpha"] if r["alpha"] is not None else "—",
                lr=lr,
                iters=r["iters"] if r["iters"] is not None else "—",
                spi=r["steps_per_iter"] if r["steps_per_iter"] is not None else "—",
                tot=r["total_steps"] if r["total_steps"] is not None else "—",
                batch=r["batch"] if r["batch"] is not None else "—",
                rm=format_float(r["reward_mean"]),
                rs=format_float(r["reward_std"]),
                fm=format_float(r["forward_mean"]),
                fs=format_float(r["forward_std"]),
                k1=format_float(r["rmse_k1"]),
                k10=format_float(r["rmse_k10"]),
                k15=format_float(r["rmse_k15"]),
                d1=format_float(r["delta_k1"]),
                d10=format_float(r["delta_k10"]),
                status=r["status"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default="/Users/isoceanum/Projects/l2ra/learning_to_re_adapt/outputs/2026-03-13-ex3",
        help="Outputs sweep directory",
    )
    args = parser.parse_args()
    rows = summarize(args.root)
    print_markdown(rows)


if __name__ == "__main__":
    main()
