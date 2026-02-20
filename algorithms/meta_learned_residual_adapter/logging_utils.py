import math
import torch


def _safe_mean(values):
    if len(values) == 0:
        return float("nan")
    return float(torch.stack(values).mean().item())


def _safe_std(values):
    if len(values) == 0:
        return float("nan")
    return float(torch.stack(values).std(unbiased=False).item())


def _grad_norm(parameters):
    sq_sum = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        sq_sum += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(sq_sum) if sq_sum > 0.0 else 0.0


def _param_norm(parameters):
    sq_sum = 0.0
    for p in parameters:
        sq_sum += float(p.detach().pow(2).sum().item())
    return math.sqrt(sq_sum) if sq_sum > 0.0 else 0.0


def compute_meta_update_stats(
    support_losses,
    query_losses_pre,
    query_losses_post,
    residual_adapter,
):
    support_mean = _safe_mean(support_losses)
    query_pre_mean = _safe_mean(query_losses_pre)
    query_post_mean = _safe_mean(query_losses_post)
    query_improve = query_pre_mean - query_post_mean

    support_std = _safe_std(support_losses)
    query_post_std = _safe_std(query_losses_post)

    grad_norm = _grad_norm(residual_adapter.parameters())
    param_norm = _param_norm(residual_adapter.parameters())

    return {
        "support_mean": support_mean,
        "support_std": support_std,
        "query_pre_mean": query_pre_mean,
        "query_post_mean": query_post_mean,
        "query_post_std": query_post_std,
        "query_improve": query_improve,
        "param_norm": param_norm,
        "grad_norm": grad_norm,
    }


def format_meta_update_line(stats, prefix="", decimals=6):
    fmt = f".{decimals}f"
    line = (
        f"{prefix}support={stats['support_mean']:{fmt}}±{stats['support_std']:{fmt}} "
        f"query_pre={stats['query_pre_mean']:{fmt}} "
        f"query_post={stats['query_post_mean']:{fmt}}±{stats['query_post_std']:{fmt}} "
        f"improve={stats['query_improve']:{fmt}} "
        f"param_norm={stats['param_norm']:{fmt}} "
        f"grad_norm={stats['grad_norm']:{fmt}}"
    )
    return line


def summarize_meta_update_logs(
    support_losses,
    query_losses_pre,
    query_losses_post,
    residual_adapter,
    prefix="",
    decimals=6,
):
    stats = compute_meta_update_stats(
        support_losses,
        query_losses_pre,
        query_losses_post,
        residual_adapter,
    )
    return format_meta_update_line(stats, prefix=prefix, decimals=decimals)
