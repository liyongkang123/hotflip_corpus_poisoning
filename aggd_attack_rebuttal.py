"""AGGD search adapted to the shared LLM encoder and mean-query attack targets.

Upstream: https://github.com/JinyanSu1/AGGD/tree/aa0e4fd29a99b2f73361b28cf6644ab2382e19a6
The candidate ordering and adaptive depth follow AGGD. Initialization, objective,
validation, length and candidate count follow hotflip_rebuttal; the iteration budget is 3000.
"""
import logging
import time

import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from hotflip_attack_rebuttal import main as generation_main
from model.aggd import aggd_candidates, score_candidates, search_depth_limit
from utils.evaluate import evaluate_sim_ours_new
from utils.rebuttal_utils import rebuttal_get_emb

logger = logging.getLogger(__name__)


def run_cluster(encoder, target, args, seed, on_improve):
    set_seed(seed)
    device = encoder.device
    adv = torch.tensor([target.gold_init_ids], dtype=torch.long, device=device)
    loader = DataLoader(torch.from_numpy(target.train_means).to(device), batch_size=1, shuffle=True)
    n_adv, start = adv.shape[1], encoder.doc_head.numel()
    depth_limit = search_depth_limit(n_adv, args.num_cand, int(encoder.cand_mask.sum().item()))
    if not depth_limit:
        raise ValueError("AGGD has no legal candidate tokens")

    def validate(ids):
        with torch.no_grad():
            return evaluate_sim_ours_new(encoder, encoder, rebuttal_get_emb, target.valid_means, ids, False, device)

    def entry(finish=False):
        ids = best_ids[0].tolist()
        return dict(best_sim=best_sim, best_adv_text=encoder.tokenizer.convert_ids_to_tokens(ids),
                    best_adv_passage_ids=ids, tot=target.tot, finish=finish, it=best_it)

    best_ids = adv.clone()
    best_sim, best_it, depth = validate(best_ids), -1, 1
    on_improve(entry())
    began = time.monotonic()
    for iteration in range(args.num_iter):
        train_iter = iter(loader)
        gradient_target = next(train_iter)
        full_ids = torch.cat([encoder.doc_head.unsqueeze(0), adv, encoder.doc_tail.unsqueeze(0)], dim=1)
        _, grad = encoder.doc_embedding_and_grad(full_ids, lambda emb: (gradient_target @ emb.T).mean())
        candidates = aggd_candidates(grad[start:start + n_adv], encoder.embedding_weight,
                                     depth, args.num_cand, encoder.cand_mask)
        score_target = next(train_iter, gradient_target)
        with torch.no_grad():
            baseline = (score_target @ rebuttal_get_emb(encoder, {"input_ids": adv}).T).mean().item()
        scores = score_candidates(encoder, adv, candidates, score_target)
        improved = False
        if scores.max().item() > baseline:
            position, token = candidates[scores.argmax()]
            adv[0, position] = token
            current_sim = validate(adv)
            if current_sim > best_sim:
                best_ids, best_sim, best_it = adv.clone(), current_sim, iteration
                on_improve(entry())
                improved = True
        # Upstream resets only after a new validation best, not after any train improvement.
        # When every legal pair has been visited, start a new pass instead of overflowing topk.
        depth = 1 if improved else depth % depth_limit + 1
        if (iteration + 1) % 100 == 0 or iteration + 1 == args.num_iter:
            logger.info("AGGD iteration %d/%d best_sim=%.6f next_depth=%d seconds/iter=%.3f",
                        iteration + 1, args.num_iter, best_sim, depth,
                        (time.monotonic() - began) / (iteration + 1))
    return entry(finish=True)


def main():
    generation_main(method="aggd_rebuttal", cluster_runner=run_cluster)


if __name__ == "__main__":
    main()
