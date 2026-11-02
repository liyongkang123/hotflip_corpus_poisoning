"""Encode each model/dataset's training queries once for all HotFlip seeds and N values."""
import argparse
import logging
from pathlib import Path

from transformers import set_seed

from rebuttal_data import DEFAULT_QUERY_EMBEDDING_ROOT, build_query_embeddings
from utils.rebuttal_utils import REBUTTAL_MODELS, RebuttalEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=REBUTTAL_MODELS)
    parser.add_argument("--attack_dataset", required=True, choices=["nq-train", "hotpotqa", "msmarco"])
    parser.add_argument("--query_embedding_root", type=Path, default=DEFAULT_QUERY_EMBEDDING_ROOT)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--max_queries", type=int, help="For short checks; use a separate query_embedding_root")
    parser.add_argument("--device_map", default=None)
    args = parser.parse_args()
    if args.query_batch_size <= 0 or (args.max_queries is not None and args.max_queries <= 0):
        parser.error("query_batch_size and max_queries must be positive")
    if args.max_queries is not None and args.query_embedding_root.resolve() == DEFAULT_QUERY_EMBEDDING_ROOT.resolve():
        parser.error("Short checks require a separate --query_embedding_root")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(0)
    build_query_embeddings(args.query_embedding_root, args.model, args.attack_dataset,
        lambda: RebuttalEncoder(args.model, args.attack_dataset, device_map=args.device_map),
        batch_size=args.query_batch_size, max_queries=args.max_queries)


if __name__ == "__main__":
    main()
