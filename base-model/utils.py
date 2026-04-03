from __future__ import annotations
import torch


def top_k_top_p_filtering(
    logits: torch.tensor, top_k: int | None = None, top_p: float | None = None
):
    B, V = logits.shape
    filtered = logits.clone()

    if top_k is not None and top_k < V:
        top_vals, _ = torch.topk(filtered, top_k, dim=-1)
        kth = top_vals[:, -1].unsqueeze(-1)
        filtered[filtered < kth] = float("-inf")

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum > top_p
        # make a atleast one to be False
        mask[..., 0] = False
        sorted_logits[mask] = float("-inf")
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(1, sorted_idx, sorted_logits)
    return filtered
