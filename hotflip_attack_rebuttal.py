"""Generate hotflip_rebuttal documents without changing the legacy attack pipeline."""
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from model.hotflip import hotflip_attack, hotflip_candidate_score_new
from rebuttal_data import (DEFAULT_QUERY_EMBEDDING_ROOT, cluster_seed, load_query_embeddings,
                           prepare_targets, remove_targets_cache)
from utils.evaluate import evaluate_sim_ours_new
from utils.rebuttal_utils import REBUTTAL_MODELS, RebuttalEncoder, rebuttal_get_emb

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT = "output_attack/attacked_text/document/hotflip_rebuttal"
DEFAULT_NUM_ITER = {"hotflip_rebuttal": 5000, "aggd_rebuttal": 3000}


def parse_args(method="hotflip_rebuttal"):
    parser = argparse.ArgumentParser(description=f"Generate {method} documents with the shared LLM encoding pipeline.")
    parser.add_argument("--model", required=True, choices=REBUTTAL_MODELS)
    parser.add_argument("--attack_dataset", required=True, choices=["nq-train", "hotpotqa", "msmarco"])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1999, 5, 27, 2016, 2026])
    parser.add_argument("--num_iter", type=int, default=DEFAULT_NUM_ITER.get(method, 5000))
    parser.add_argument("--num_cand", type=int, default=100)
    parser.add_argument("--num_adv_passage_tokens", type=int, default=50)
    parser.add_argument("--query_embedding_root", type=Path, default=DEFAULT_QUERY_EMBEDDING_ROOT)
    parser.add_argument("--cand_chunk", type=int, default=100, help="Candidate forward batch size; lower this if GPU memory is tight")
    parser.add_argument("--cache_root", type=Path, required=True)
    output_default = Path("output_attack/attacked_text/document") / method
    parser.add_argument("--output_root", type=Path, default=output_default)
    parser.add_argument("--max_queries", type=int, help="Cap unique training query ids for a short check")
    parser.add_argument("--device_map", default=None)
    args = parser.parse_args()
    for name in ("num_iter", "num_cand", "num_adv_passage_tokens", "cand_chunk"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if any(k <= 0 for k in args.ks) or any(s < 0 or s * 1000 + max(args.ks) >= 2**32 for s in args.seeds):
        parser.error("ks must be positive and cluster seeds must fit in uint32")
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max_queries must be positive")
    if args.max_queries is not None and args.output_root == output_default:
        parser.error("Short checks require a separate --output_root")
    return args


def doc_json_path(output_root, dataset, model, seed, k, n_adv=50, num_iter=None, num_cand=100, method="hotflip_rebuttal"):
    if num_iter is None:
        num_iter = DEFAULT_NUM_ITER.get(method, 5000)
    name = (f"{dataset}_train_attacked_documents_seed_{seed}_{model}_{method}_"
            f"N{k}_Len{n_adv}_Iter{num_iter}_Pool{num_cand}_InitTrue.json")
    return Path(output_root) / dataset / name


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temp, path)


def load_progress(path, k):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        saved = json.load(handle)
    if not isinstance(saved, dict) or not set(saved) <= {str(i) for i in range(k)}:
        raise ValueError(f"Invalid cluster keys in {path}")
    for key, entry in saved.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid cluster {key} in {path}")
        if entry.get("finish") is True and not {"best_sim", "best_adv_passage_ids", "best_adv_text", "tot", "it"} <= entry.keys():
            raise ValueError(f"Incomplete finished cluster {key} in {path}")
    return saved


def run_cluster(encoder, target, args, seed, on_improve):
    set_seed(seed)
    device = encoder.device
    adv = torch.tensor([target.gold_init_ids], dtype=torch.long, device=device)
    loader = DataLoader(torch.from_numpy(target.train_means).to(device), batch_size=1, shuffle=True)
    n_adv = adv.shape[1]
    start = encoder.doc_head.numel()
    # Exclude unused embedding rows; special tokens in the vocabulary remain legal.
    candidate_filter = torch.zeros(encoder.embedding_weight.shape[0], dtype=encoder.embedding_weight.dtype, device=device)
    candidate_filter.masked_fill_(~encoder.cand_mask.to(device), float("inf"))
    if args.num_cand > encoder.cand_mask.sum().item():
        raise ValueError("num_cand exceeds the number of tokenizer ids with embedding rows")

    def validate(ids):
        with torch.no_grad():
            return evaluate_sim_ours_new(encoder, encoder, rebuttal_get_emb,
                                         target.valid_means, ids, False, device)

    def entry(ids, score, iteration, finish=False):
        return dict(best_sim=score, best_adv_text=encoder.tokenizer.convert_ids_to_tokens(ids[0].tolist()),
                    best_adv_passage_ids=ids[0].tolist(), tot=target.tot, finish=finish, it=iteration)

    best_ids = adv.clone()
    best_sim = validate(best_ids)
    best_it = -1
    on_improve(entry(best_ids, best_sim, best_it))
    started = time.monotonic()
    for iteration in range(args.num_iter):
        train_iter = iter(loader)
        gradient_target = next(train_iter)
        full_ids = torch.cat([encoder.doc_head.unsqueeze(0), adv, encoder.doc_tail.unsqueeze(0)], dim=1)
        _, gradient = encoder.doc_embedding_and_grad(full_ids, lambda emb: (gradient_target @ emb.T).mean())
        pos = random.randrange(n_adv)
        candidates = hotflip_attack(gradient[start + pos], encoder.embedding_weight,
                                    increase_loss=True, num_candidates=args.num_cand, filter=candidate_filter)
        # Reuse candidate scoring, but compare both options on the SAME next query mean.
        score_target = next(train_iter, gradient_target)
        with torch.no_grad():
            baseline = (score_target @ rebuttal_get_emb(encoder, {"input_ids": adv}).T).mean().item()
            scores = hotflip_candidate_score_new(args, iteration, candidates, range(1), iter([score_target]),
                        rebuttal_get_emb, encoder, adv, pos, device, False)
        if scores.max().item() > baseline:
            adv[0, pos] = candidates[scores.argmax()]
            score = validate(adv)
            if score > best_sim:
                best_ids, best_sim, best_it = adv.clone(), score, iteration
                on_improve(entry(best_ids, best_sim, best_it))
        if (iteration + 1) % 100 == 0 or iteration + 1 == args.num_iter:
            logger.info("Iteration %d/%d best_sim=%.6f seconds/iter=%.3f", iteration + 1, args.num_iter,
                        best_sim, (time.monotonic() - started) / (iteration + 1))
    return entry(best_ids, best_sim, best_it, finish=True)


def main(method="hotflip_rebuttal", cluster_runner=None):
    args = parse_args(method)
    cluster_runner = run_cluster if cluster_runner is None else cluster_runner
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("%s", args)
    jobs = []
    for k in dict.fromkeys(args.ks):
        for seed in dict.fromkeys(args.seeds):
            path = doc_json_path(args.output_root, args.attack_dataset, args.model, seed, k,
                                 args.num_adv_passage_tokens, args.num_iter, args.num_cand, method=method)
            progress = load_progress(path, k)
            if len(progress) == k and all(e.get("finish") is True for e in progress.values()):
                logger.info("Already complete: %s", path)
                remove_targets_cache(args.cache_root, args.attack_dataset, args.model, k, seed)
            else:
                jobs.append((k, seed, path, progress))
    if not jobs:
        return
    # Fail before loading model weights if the prerequisite query cache is missing.
    query_embeddings = load_query_embeddings(args.query_embedding_root, args.model, args.attack_dataset, args.max_queries)
    encoder = RebuttalEncoder(args.model, args.attack_dataset, device_map=args.device_map, cand_chunk=args.cand_chunk)
    for k, seed, path, progress in jobs:
        targets = prepare_targets(encoder, args.attack_dataset, k, seed, args.cache_root,
                    args.num_adv_passage_tokens, args.max_queries, query_embeddings=query_embeddings)
        for cluster, target in enumerate(targets):
            key = str(cluster)
            if progress.get(key, {}).get("finish") is True:
                logger.info("Skipping finished cluster %s (k=%d seed=%d)", key, k, seed)
                continue
            logger.info("Attacking cluster %d/%d seed=%d pairs=%d", cluster + 1, k, seed, target.n_pairs)
            last_write = 0.0

            def save_improvement(entry):
                nonlocal last_write
                progress[key] = entry
                if time.monotonic() - last_write >= 60:
                    atomic_write_json(path, progress)
                    last_write = time.monotonic()

            progress[key] = cluster_runner(encoder, target, args, cluster_seed(seed, cluster), save_improvement)
            atomic_write_json(path, progress)
        remove_targets_cache(args.cache_root, args.attack_dataset, args.model, k, seed)
        logger.info("Finished %s", path)


if __name__ == "__main__":
    main()
