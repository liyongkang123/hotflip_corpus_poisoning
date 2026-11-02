"""AGGD rank-interleaved candidate search for token-level corpus poisoning.

Algorithm reference: JinyanSu1/AGGD, utils/AGGD.py::AGGD_candidate,
commit aa0e4fd29a99b2f73361b28cf6644ab2382e19a6.
This implementation masks unused vocabulary rows and handles finite search exhaustion.
"""
import torch


def search_depth_limit(num_positions, num_candidates, num_valid_tokens):
    return (num_positions * num_valid_tokens + num_candidates - 1) // num_candidates


def aggd_candidates(gradient, embedding_weight, depth, num_candidates, candidate_mask):
    """Take one depth slice of rank-major (position, token_id) pairs.

    Every position contributes its highest-scoring token first, then its second,
    and so on. This is the upstream ordering, not a global sort of all dot products.
    """
    if gradient.ndim != 2 or gradient.shape[0] == 0 or depth < 1 or num_candidates < 1:
        raise ValueError("AGGD needs position gradients, positive depth and candidate count")
    n_positions = gradient.shape[0]
    n_tokens = int(candidate_mask.sum().item())
    if not n_tokens:
        raise ValueError("AGGD has no legal candidate tokens")
    start = (depth - 1) * num_candidates
    end = min(start + num_candidates, n_tokens * n_positions)
    if start >= end:
        raise ValueError("AGGD depth exceeds the finite candidate pool")
    # Upstream requests one extra rank even at an exact position boundary.
    n_ranks = min(end // n_positions + 1, n_tokens)
    with torch.no_grad():
        scores = embedding_weight @ gradient.T
        scores.masked_fill_(~candidate_mask.to(scores.device)[:, None], float("-inf"))
        tokens = scores.topk(n_ranks, dim=0).indices.reshape(-1)[start:end]
        positions = torch.arange(start, end, device=scores.device) % n_positions
    return torch.stack([positions, tokens], dim=1)


def score_candidates(encoder, adv_ids, candidates, query_mean):
    """Score single-token edits in chunks, preserving fixed prompt/special-token ids."""
    from utils.rebuttal_utils import rebuttal_get_emb

    scores = []
    with torch.no_grad():
        for pairs in candidates.split(encoder.cand_chunk):
            edited = adv_ids.repeat(len(pairs), 1)
            rows = torch.arange(len(pairs), device=edited.device)
            edited[rows, pairs[:, 0]] = pairs[:, 1]
            reps = rebuttal_get_emb(encoder, {"input_ids": edited})
            scores.append((query_mean @ reps.T).mean(dim=0))
    return torch.cat(scores)
