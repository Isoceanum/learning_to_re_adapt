import os
import json
import yaml
import copy
import sys
from datetime import datetime

from algorithms.full_param_mem.trainer import FullParamMemoryTrainer
from utils.seed import set_seed


PRETRAINED_DIR = "/Users/isoceanum/Projects/l2ra/learning_to_re_adapt/outputs/2026-04-13/full_param_mem"

# Hardcoded overrides for debugging. For now, these mirror the values in the YAML
# so the script is behaviorally identical, but you can tweak them to isolate
# memory/eval/perturbation behavior without touching the checkpoint config.
OVERRIDES = {
    "train": {
        "use_memory": True,
        "memory_retrieve_window_size": 48,
        "permanent_window_size": 32,
        "retrieve_abs_threshold": 0.002,
        "retrieve_top2_margin": 0.0,
        "retrieve_diversity_alpha": 0.001,
        "write_abs_threshold": 0.001,
        "write_update_abs_threshold": 0.002,
        "slow_inner_learning_rate": 0.003,
        "debug_memory": True,
        "debug_compare_against_base": True,
        "perturbation": {
            "type": "cripple",
            "probability": 1,
            "candidate_action_indices": [[2, 3], [4, 5], [6, 7]],
        },
    },
    "eval": {
        "episodes": 10,
        "seeds": [42],
        "perturbation": {
            "type": "cripple",
            "probability": 1,
            "candidate_action_indices": [[2, 3], [4, 5], [6, 7]],
        },
    },
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _start_log_capture(output_dir):
    log_dir = os.path.join(output_dir, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_memory_{ts}.txt")
    log_f = open(log_path, "w", buffering=1)

    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = _Tee(stdout_orig, log_f)
    sys.stderr = _Tee(stderr_orig, log_f)
    print(f"[mem] full log file: {log_path}", flush=True)
    return log_f, stdout_orig, stderr_orig


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _print_eval_summary(metrics):
    elapsed = float(metrics["elapsed"])
    elapsed_str = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
    print("\nEvaluation summary:")
    print(f"- reward_mean: {metrics['reward_mean']:.4f}")
    print(f"- reward_std: {metrics['reward_std']:.4f}")
    print(f"- episode_length_mean: {metrics['episode_length_mean']:.2f}")
    print(f"- elapsed: {elapsed_str}")


def _run_eval(trainer, config):
    episodes = int(config["eval"]["episodes"])
    seeds = config["eval"]["seeds"]
    total_runs = len(seeds) * episodes
    print(f"Evaluating {episodes} episode(s) × {len(seeds)} seed(s) = {total_runs} total runs")
    metrics = trainer._evaluate(episodes, seeds)
    _print_eval_summary(metrics)
    return metrics


def _print_memory_report(trainer):
    memory = getattr(trainer, "memory", None)
    entries = getattr(memory, "entries", []) if memory is not None else []

    task_usage_counts = {}
    created_task_counts = {}
    retrievals_total = 0
    updates_total = 0
    entry_summaries = []

    for entry in entries:
        memory_id = str(entry.get("id", "unknown"))
        created_task = str(entry.get("created_task", "nominal"))
        n_retrieved = int(entry.get("n_retrieved", 0))
        n_updated = int(entry.get("n_updated", 0))
        retrieve_task_counts = dict(entry.get("retrieve_task_counts", {}))

        created_task_counts[created_task] = int(created_task_counts.get(created_task, 0)) + 1
        retrievals_total += n_retrieved
        updates_total += n_updated

        normalized_task_counts = {}
        for task, count in retrieve_task_counts.items():
            task_key = str(task)
            task_count = int(count)
            normalized_task_counts[task_key] = task_count
            task_usage_counts[task_key] = int(task_usage_counts.get(task_key, 0)) + task_count

        entry_summaries.append(
            {
                "id": memory_id,
                "created_task": created_task,
                "n_retrieved": n_retrieved,
                "n_returned": n_retrieved,
                "n_updated": n_updated,
                "retrieve_task_counts": normalized_task_counts,
            }
        )

    entry_summaries.sort(key=lambda x: x["id"])

    eval_retrieve_stats = dict(getattr(trainer, "_eval_retrieve_stats", {}) or {})
    transfer = _build_transfer_report(entry_summaries)

    eval_split_report = _build_eval_split_report(trainer)
    report = {
        "memory_report": {
            "entries_total": len(entries),
            "retrievals_total": retrievals_total,
            "returns_total": retrievals_total,
            "updates_total": updates_total,
            "eval_retrieve_stats": eval_retrieve_stats,
            "eval_split_report": eval_split_report,
            "transfer_report": transfer,
            "task_usage_counts": dict(sorted(task_usage_counts.items())),
            "created_task_counts": dict(sorted(created_task_counts.items())),
            "entries": entry_summaries,
        }
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    _print_param_feedback(trainer, report["memory_report"])
    return report


def _stats(values):
    if not values:
        return {"count": 0, "reward_mean": None, "reward_std": None}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) * (x - mean) for x in values) / n
    return {"count": n, "reward_mean": float(mean), "reward_std": float(var ** 0.5)}


def _mean(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def _build_transfer_report(entry_summaries):
    total_retrievals = 0
    same_task_retrievals = 0
    cross_task_retrievals = 0
    per_entry = []
    for entry in entry_summaries:
        created_task = str(entry.get("created_task", "nominal"))
        retrieve_counts = dict(entry.get("retrieve_task_counts", {}) or {})
        n_retrieved = int(entry.get("n_retrieved", 0))
        same = int(retrieve_counts.get(created_task, 0))
        cross = int(max(0, n_retrieved - same))
        total_retrievals += n_retrieved
        same_task_retrievals += same
        cross_task_retrievals += cross
        per_entry.append(
            {
                "id": entry.get("id"),
                "created_task": created_task,
                "n_retrieved": n_retrieved,
                "same_task_retrievals": same,
                "cross_task_retrievals": cross,
                "same_task_ratio": None if n_retrieved == 0 else float(same / n_retrieved),
            }
        )
    same_rate = None if total_retrievals == 0 else float(same_task_retrievals / total_retrievals)
    cross_rate = None if total_retrievals == 0 else float(cross_task_retrievals / total_retrievals)
    return {
        "total_retrievals": int(total_retrievals),
        "same_task_retrievals": int(same_task_retrievals),
        "cross_task_retrievals": int(cross_task_retrievals),
        "same_task_rate": same_rate,
        "cross_task_rate": cross_rate,
        "per_entry": per_entry,
    }


def _build_eval_split_report(trainer):
    records = list(getattr(trainer, "_eval_episode_records", []) or [])
    if not records:
        return {}

    retrieved_rewards = []
    not_retrieved_rewards = []
    by_task = {}
    by_reason = {}
    by_selected_task = {}
    by_store_action = {}
    by_task_and_selected = {}
    pre_adv = []
    post_adv = []
    delta_l2 = []
    delta_max_abs = []
    high67 = []
    low67 = []
    probe_winner_differs = 0
    probe_winner_compared = 0

    for row in records:
        reward = float(row.get("episode_return", 0.0))
        task = str(row.get("task", "nominal"))
        reason = str(row.get("retrieve_reason", "unknown"))
        selected_task = row.get("selected_created_task")
        selected_task = "none" if selected_task is None else str(selected_task)
        store_action = str(row.get("store_action", "unknown"))

        if bool(row.get("retrieved", False)):
            retrieved_rewards.append(reward)
        else:
            not_retrieved_rewards.append(reward)

        by_task.setdefault(task, []).append(reward)
        by_reason.setdefault(reason, []).append(reward)
        by_selected_task.setdefault(selected_task, []).append(reward)
        by_store_action.setdefault(store_action, []).append(reward)
        pair_key = f"{task}|selected={selected_task}"
        by_task_and_selected.setdefault(pair_key, []).append(reward)

        pre = row.get("retrieve_improvement_pre")
        post = row.get("retrieve_improvement_post")
        l2 = row.get("selected_param_delta_l2")
        mabs = row.get("selected_param_delta_max_abs")
        hi = row.get("retrieve_action67_high_improvement")
        lo = row.get("retrieve_action67_low_improvement")
        wd = row.get("retrieve_probe_winner_differs")
        if pre is not None:
            pre_adv.append(float(pre))
        if post is not None:
            post_adv.append(float(post))
        if l2 is not None:
            delta_l2.append(float(l2))
        if mabs is not None:
            delta_max_abs.append(float(mabs))
        if hi is not None:
            high67.append(float(hi))
        if lo is not None:
            low67.append(float(lo))
        if wd is not None:
            probe_winner_compared += 1
            if bool(wd):
                probe_winner_differs += 1

    return {
        "episode_count": len(records),
        "retrieved": _stats(retrieved_rewards),
        "not_retrieved": _stats(not_retrieved_rewards),
        "by_task": {k: _stats(v) for k, v in sorted(by_task.items())},
        "by_retrieve_reason": {k: _stats(v) for k, v in sorted(by_reason.items())},
        "by_selected_created_task": {k: _stats(v) for k, v in sorted(by_selected_task.items())},
        "by_store_action": {k: _stats(v) for k, v in sorted(by_store_action.items())},
        "by_task_and_selected_created_task": {k: _stats(v) for k, v in sorted(by_task_and_selected.items())},
        "retrieve_advantage_pre_mean": _mean(pre_adv),
        "retrieve_advantage_post_mean": _mean(post_adv),
        "selected_param_delta_l2_mean": _mean(delta_l2),
        "selected_param_delta_max_abs_mean": _mean(delta_max_abs),
        "action67_high_improvement_mean": _mean(high67),
        "action67_low_improvement_mean": _mean(low67),
        "probe_winner_differs_count": int(probe_winner_differs),
        "probe_winner_compared_count": int(probe_winner_compared),
        "probe_winner_differs_rate": (
            None if probe_winner_compared == 0 else float(probe_winner_differs / probe_winner_compared)
        ),
    }


def _print_param_feedback(trainer, memory_report):
    train_cfg = trainer.train_config
    current = {
        "memory_retrieve_window_size": int(train_cfg.get("memory_retrieve_window_size", 0)),
        "permanent_window_size": int(train_cfg.get("permanent_window_size", 0)),
        "retrieve_abs_threshold": float(train_cfg.get("retrieve_abs_threshold", 0.0)),
        "retrieve_top2_margin": float(train_cfg.get("retrieve_top2_margin", 0.0)),
        "retrieve_diversity_alpha": float(train_cfg.get("retrieve_diversity_alpha", 0.0)),
        "write_abs_threshold": float(train_cfg.get("write_abs_threshold", 0.0)),
        "write_update_abs_threshold": float(train_cfg.get("write_update_abs_threshold", 0.0)),
        "slow_inner_learning_rate": float(train_cfg.get("slow_inner_learning_rate", 0.0)),
    }

    stats = memory_report.get("eval_retrieve_stats", {}) or {}
    attempts = int(stats.get("attempts", 0))
    accepted = int(stats.get("accepted", 0))
    rejected = int(stats.get("rejected", 0))
    reasons = dict(stats.get("reasons", {}) or {})
    selected_counts = dict(stats.get("selected_created_task_counts", {}) or {})

    entries_total = int(memory_report.get("entries_total", 0))
    retrievals_total = int(memory_report.get("retrievals_total", 0))
    updates_total = int(memory_report.get("updates_total", 0))
    usage_counts = dict(memory_report.get("task_usage_counts", {}) or {})

    accept_rate = (accepted / attempts) if attempts > 0 else 0.0
    max_usage = max(usage_counts.values()) if usage_counts else 0
    usage_share = (max_usage / retrievals_total) if retrievals_total > 0 else 0.0
    no_best = int(reasons.get("no_best_entry", 0))
    below_thr = int(reasons.get("below_threshold", 0))
    min2_fail = int(reasons.get("min2_not_positive", 0))
    margin_fail = int(reasons.get("below_top2_margin", 0))

    print("\n[mem] parameter_feedback:", flush=True)
    print(f"[mem] current_params: {json.dumps(current, sort_keys=True)}", flush=True)
    print(
        f"[mem] run_health: attempts={attempts} accepted={accepted} rejected={rejected} "
        f"accept_rate={accept_rate:.3f} entries_total={entries_total} "
        f"retrievals_total={retrievals_total} updates_total={updates_total}",
        flush=True,
    )
    print(
        f"[mem] failure_breakdown: no_best_entry={no_best} below_threshold={below_thr} "
        f"min2_not_positive={min2_fail} below_top2_margin={margin_fail}",
        flush=True,
    )
    if selected_counts:
        print(f"[mem] selected_created_task_counts: {json.dumps(dict(sorted(selected_counts.items())), sort_keys=True)}", flush=True)
    if usage_counts:
        print(f"[mem] task_usage_counts: {json.dumps(dict(sorted(usage_counts.items())), sort_keys=True)}", flush=True)
    split = memory_report.get("eval_split_report", {}) or {}
    retrieved_split = split.get("retrieved", {}) or {}
    not_retrieved_split = split.get("not_retrieved", {}) or {}
    if retrieved_split.get("count", 0) > 0 or not_retrieved_split.get("count", 0) > 0:
        print(
            "[mem] reward_split: "
            f"retrieved_mean={retrieved_split.get('reward_mean')} n={retrieved_split.get('count', 0)} | "
            f"not_retrieved_mean={not_retrieved_split.get('reward_mean')} n={not_retrieved_split.get('count', 0)}",
            flush=True,
        )
    if split.get("by_task"):
        print(f"[mem] reward_by_task: {json.dumps(split['by_task'], sort_keys=True)}", flush=True)
    if split.get("by_retrieve_reason"):
        print(
            f"[mem] reward_by_retrieve_reason: {json.dumps(split['by_retrieve_reason'], sort_keys=True)}",
            flush=True,
        )
    if split.get("by_selected_created_task"):
        print(
            f"[mem] reward_by_selected_created_task: {json.dumps(split['by_selected_created_task'], sort_keys=True)}",
            flush=True,
        )
    transfer = memory_report.get("transfer_report", {}) or {}
    if transfer:
        print(
            "[mem] transfer: "
            f"same_task_rate={transfer.get('same_task_rate')} "
            f"cross_task_rate={transfer.get('cross_task_rate')}",
            flush=True,
        )
    pre_adv = split.get("retrieve_advantage_pre_mean")
    post_adv = split.get("retrieve_advantage_post_mean")
    if pre_adv is not None or post_adv is not None:
        print(
            f"[mem] retrieve_advantage: pre_mean={pre_adv} post_mean={post_adv}",
            flush=True,
        )
    l2_mean = split.get("selected_param_delta_l2_mean")
    max_abs_mean = split.get("selected_param_delta_max_abs_mean")
    if l2_mean is not None or max_abs_mean is not None:
        print(
            f"[mem] param_delta: l2_mean={l2_mean} max_abs_mean={max_abs_mean}",
            flush=True,
        )
    hi67 = split.get("action67_high_improvement_mean")
    lo67 = split.get("action67_low_improvement_mean")
    if hi67 is not None or lo67 is not None:
        print(
            f"[mem] action67_improvement: high_mean={hi67} low_mean={lo67}",
            flush=True,
        )
    probe_diff_rate = split.get("probe_winner_differs_rate")
    probe_diff_count = split.get("probe_winner_differs_count")
    probe_cmp = split.get("probe_winner_compared_count")
    if probe_diff_rate is not None:
        print(
            f"[mem] probe_winner_differs: rate={probe_diff_rate} count={probe_diff_count}/{probe_cmp}",
            flush=True,
        )

    suggestions = []

    if no_best > 0:
        suggestions.append(
            "P0: `no_best_entry` still occurred; verify retrieve loop is selecting winner before gates."
        )

    if attempts > 0 and accept_rate < 0.35:
        new_thr = max(0.0015, round(current["retrieve_abs_threshold"] * 0.8, 6))
        suggestions.append(
            f"P1: acceptance is low ({accept_rate:.2f}); lower `retrieve_abs_threshold` first: "
            f"{current['retrieve_abs_threshold']:.6f} -> {new_thr:.6f} (about -20%)."
        )

    if attempts > 0 and min2_fail / max(1, attempts) > 0.25:
        new_win = current["memory_retrieve_window_size"] + 16
        suggestions.append(
            f"P1: many `min2_not_positive` rejects; increase `memory_retrieve_window_size`: "
            f"{current['memory_retrieve_window_size']} -> {new_win} (+16)."
        )

    if current["retrieve_top2_margin"] > 0.0 and attempts > 0 and margin_fail / max(1, attempts) > 0.2:
        new_margin = round(current["retrieve_top2_margin"] * 0.5, 6)
        suggestions.append(
            f"P1: many `below_top2_margin` rejects; reduce `retrieve_top2_margin`: "
            f"{current['retrieve_top2_margin']:.6f} -> {new_margin:.6f}."
        )

    if retrievals_total >= 6 and usage_share > 0.75:
        new_alpha = round(max(0.0005, current["retrieve_diversity_alpha"] * 2.0), 6)
        suggestions.append(
            f"P1: retrieval collapsed to one task (top share {usage_share:.2f}); raise "
            f"`retrieve_diversity_alpha`: {current['retrieve_diversity_alpha']:.6f} -> {new_alpha:.6f}."
        )

    if entries_total >= 10 and retrievals_total > 0 and (entries_total / retrievals_total) > 1.5:
        new_insert = round(current["write_abs_threshold"] * 1.25, 6)
        new_update = round(current["write_update_abs_threshold"] * 1.25, 6)
        suggestions.append(
            f"P2: memory is growing faster than use; tighten writes: "
            f"`write_abs_threshold` {current['write_abs_threshold']:.6f} -> {new_insert:.6f}, "
            f"`write_update_abs_threshold` {current['write_update_abs_threshold']:.6f} -> {new_update:.6f}."
        )

    if retrievals_total > 0 and updates_total / retrievals_total < 0.5:
        new_wu = round(max(current["write_abs_threshold"] + 0.0002, current["write_update_abs_threshold"] * 0.85), 6)
        suggestions.append(
            f"P2: low update ratio ({updates_total}/{retrievals_total}); make updates easier: "
            f"`write_update_abs_threshold` {current['write_update_abs_threshold']:.6f} -> {new_wu:.6f}."
        )

    retrieved_mean = retrieved_split.get("reward_mean")
    not_retrieved_mean = not_retrieved_split.get("reward_mean")
    if (
        retrieved_mean is not None
        and not_retrieved_mean is not None
        and retrieved_split.get("count", 0) >= 4
        and not_retrieved_split.get("count", 0) >= 4
    ):
        if retrieved_mean + 20.0 < not_retrieved_mean:
            tighter = round(current["retrieve_abs_threshold"] * 1.2, 6)
            suggestions.append(
                f"P1: retrieved episodes underperform non-retrieved; tighten acceptance first: "
                f"`retrieve_abs_threshold` {current['retrieve_abs_threshold']:.6f} -> {tighter:.6f}."
            )
        elif retrieved_mean > not_retrieved_mean + 20.0 and attempts > 0 and accept_rate < 0.6:
            looser = round(max(0.0015, current["retrieve_abs_threshold"] * 0.9), 6)
            suggestions.append(
                f"P1: retrieval looks beneficial; allow more of it: "
                f"`retrieve_abs_threshold` {current['retrieve_abs_threshold']:.6f} -> {looser:.6f}."
            )

    if not suggestions:
        suggestions.append("No strong red flags; sweep one parameter at a time around current values.")

    print("[mem] suggested_change_order:", flush=True)
    for i, line in enumerate(suggestions, start=1):
        print(f"[mem]   {i}. {line}", flush=True)


def main():
    log_f, stdout_orig, stderr_orig = _start_log_capture(PRETRAINED_DIR)
    try:
        memory_path = os.path.join(PRETRAINED_DIR, "memory.pt")
        if os.path.exists(memory_path):
            os.remove(memory_path)
            print(f"Removed stale memory file: {memory_path}", flush=True)

        config_path = os.path.join(PRETRAINED_DIR, "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        _deep_update(config, OVERRIDES)

        set_seed(int(config["train"]["seed"]))
        trainer = FullParamMemoryTrainer(config, PRETRAINED_DIR)
        trainer.load(PRETRAINED_DIR)
        memory_metrics = _run_eval(trainer, config)
        memory_report = _print_memory_report(trainer)

        compare_base = bool(config["train"].get("debug_compare_against_base", False))
        if compare_base:
            base_config = copy.deepcopy(config)
            base_config["train"]["use_memory"] = False
            base_config["train"]["debug_memory"] = False
            set_seed(int(base_config["train"]["seed"]))
            base_trainer = FullParamMemoryTrainer(base_config, PRETRAINED_DIR)
            base_trainer.load(PRETRAINED_DIR)
            print("\n[mem] running_base_comparison_eval...", flush=True)
            base_metrics = _run_eval(base_trainer, base_config)
            delta = float(memory_metrics["reward_mean"] - base_metrics["reward_mean"])
            compare_payload = {
                "memory_reward_mean": float(memory_metrics["reward_mean"]),
                "base_reward_mean": float(base_metrics["reward_mean"]),
                "reward_mean_delta_memory_minus_base": delta,
                "memory_reward_std": float(memory_metrics["reward_std"]),
                "base_reward_std": float(base_metrics["reward_std"]),
                "memory_entries_total": int(memory_report["memory_report"].get("entries_total", 0)),
            }
            print(f"[mem] base_compare: {json.dumps(compare_payload, sort_keys=True)}", flush=True)
    finally:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_f.close()


if __name__ == "__main__":
    main()
