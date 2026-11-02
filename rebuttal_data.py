"""Training pairs and cached cluster targets for the corrected HotFlip pipeline."""
import csv
import fcntl
import hashlib
import json
import logging
import os
import random
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.cluster import KMeans

from utils.load_data import batch_average, create_batches
from utils.rebuttal_utils import MODEL_SPECS, get_rebuttal_prompts

logger = logging.getLogger(__name__)
CACHE_VERSION = 2
QUERY_CACHE_VERSION = 1
DEFAULT_QUERY_EMBEDDING_ROOT = Path(__file__).resolve().parent / "cache_rebuttal/query_embeddings"


@dataclass
class ClusterTarget:
    train_means: np.ndarray
    valid_means: np.ndarray
    gold_init_ids: list[int]
    tot: int
    n_pairs: int


def cluster_seed(seed, cluster):
    return seed * 1000 + cluster


@lru_cache(maxsize=1)
def _train_queries(data_path, max_queries):
    """Read training query ids/texts without loading the corpus."""
    data_path = Path(data_path)
    qrels = {}
    with (data_path / "qrels/train.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qid = row["query-id"]
            # Select complete query groups, retaining the original qrel weighting.
            if qid not in qrels and max_queries is not None and len(qrels) >= max_queries:
                continue
            qrels.setdefault(qid, []).append(row["corpus-id"])
    if not qrels:
        raise ValueError(f"No training qrels in {data_path}")
    queries = {}
    with (data_path / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["_id"] in qrels:
                queries[item["_id"]] = item["text"]
    # Canonical qrel order is independent of attack seed and k.
    return qrels, {qid: queries[qid] for qid in qrels}


def load_train_queries(dataset, max_queries=None):
    data_path = Path(os.environ.get("HF_HOME", ".")) / "datasets" / dataset
    return _train_queries(str(data_path), max_queries)[1]


@lru_cache(maxsize=1)
def _train_groups(data_path, max_queries):
    """Keep only qrel documents in RAM; small checks need not load the full corpus."""
    data_path = Path(data_path)
    qrels, queries = _train_queries(str(data_path), max_queries)
    needed = {cid for ids in qrels.values() for cid in ids}
    passages = {}
    with (data_path / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            cid = item["_id"]
            if cid in needed:
                passages[cid] = (item.get("title") or "") + " " + (item.get("text") or "")
                if len(passages) == len(needed):
                    break
    # Missing qids/cids deliberately raise rather than silently dropping pairs.
    return [(qid, queries[qid], [passages[cid] for cid in ids]) for qid, ids in qrels.items()]


def load_train_pairs(dataset, seed, max_queries=None, return_ids=False):
    data_path = Path(os.environ.get("HF_HOME", ".")) / "datasets" / dataset
    groups = list(_train_groups(str(data_path), max_queries))
    random.Random(seed).shuffle(groups)
    queries, passages, query_ids = [], [], []
    for qid, query, golds in groups:
        queries.extend([query] * len(golds))
        passages.extend(golds)
        query_ids.extend([qid] * len(golds))
    return (queries, passages, query_ids) if return_ids else (queries, passages)


def query_embedding_dir(root, dataset, model):
    return Path(root) / dataset / model


def _query_meta(model, dataset, queries, max_queries):
    spec = MODEL_SPECS[model]
    digest = hashlib.sha256()
    for qid, text in queries.items():
        digest.update((json.dumps([qid, text], ensure_ascii=False) + "\n").encode("utf-8"))
    return dict(version=QUERY_CACHE_VERSION, model=model, dataset=dataset, split="train", max_queries=max_queries,
                input_sha256=digest.hexdigest(), hf_id=spec.hf_id, tokenizer_kwargs=spec.tokenizer_kwargs,
                model_kwargs=spec.model_kwargs, query_prompt=get_rebuttal_prompts(model, dataset)["query"],
                pooling=spec.pooling, normalize=spec.normalize, max_len=spec.max_len, model_dtype="bfloat16")


@dataclass
class QueryEmbeddings:
    embeddings: np.ndarray
    rows: dict[str, int]
    meta: dict
    cache_id: str

    @property
    def identity(self):
        return self.cache_id

    def for_ids(self, query_ids):
        return self.embeddings[[self.rows[qid] for qid in query_ids]]


def load_query_embeddings(root, model, dataset, max_queries=None):
    """Read the immutable float32 matrix by query id; never encode on a cache miss."""
    path = query_embedding_dir(root, dataset, model)
    manifest, matrix = path / "queries.json", path / "queries.npy"
    if not manifest.exists() or not matrix.exists():
        raise FileNotFoundError(f"Training query cache missing: {path}. Run encode_rebuttal_queries.py first.")
    queries = load_train_queries(dataset, max_queries)
    expected = _query_meta(model, dataset, queries, max_queries)
    with manifest.open(encoding="utf-8") as handle:
        saved = json.load(handle)
    if saved["meta"] != expected or saved["query_ids"] != list(queries):
        raise ValueError(f"Training query cache settings/data differ: {path}. Use a separate query_embedding_root.")
    embeddings = np.load(matrix, mmap_mode="r", allow_pickle=False)
    if (embeddings.ndim != 2 or list(embeddings.shape) != saved["shape"]
            or embeddings.shape[0] != len(queries) or embeddings.dtype != np.float32):
        raise ValueError(f"Invalid training query embedding matrix: {matrix}")
    logger.info("Reading %d cached training queries from %s", len(queries), matrix)
    return QueryEmbeddings(embeddings, {qid: i for i, qid in enumerate(queries)}, expected, saved["cache_id"])


def build_query_embeddings(root, model, dataset, encoder_factory, batch_size=64, max_queries=None):
    """Encode once under a file lock. Publish the manifest only after the matrix is complete."""
    path = query_embedding_dir(root, dataset, model)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".encode.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (path / "queries.json").exists():
            # Completed caches are immutable; a second submit skips even model loading.
            return load_query_embeddings(root, model, dataset, max_queries)
        queries = load_train_queries(dataset, max_queries)
        meta = _query_meta(model, dataset, queries, max_queries)
        texts = list(queries.values())
        encoder = encoder_factory()
        temp_matrix, temp_manifest = path / "queries.npy.tmp", path / "queries.json.tmp"
        matrix = None
        try:
            for start in range(0, len(texts), batch_size):
                batch = encoder.encode_queries(texts[start:start + batch_size], batch_size=batch_size).numpy()
                batch = np.asarray(batch, dtype=np.float32)
                if batch.ndim != 2 or not np.isfinite(batch).all():
                    raise ValueError(f"Invalid query embeddings at row {start}")
                if matrix is None:
                    matrix = np.lib.format.open_memmap(temp_matrix, mode="w+", dtype=np.float32,
                                                      shape=(len(texts), batch.shape[1]))
                matrix[start:start + len(batch)] = batch
                if (start // batch_size + 1) % 100 == 0 or start + len(batch) == len(texts):
                    logger.info("Encoded %d/%d training queries", start + len(batch), len(texts))
            matrix.flush()
            shape = list(matrix.shape)
            del matrix
            matrix = None
            with temp_manifest.open("w", encoding="utf-8") as handle:
                json.dump(dict(meta=meta, query_ids=list(queries), shape=shape, cache_id=uuid.uuid4().hex,
                               transformers=transformers.__version__, batch_size=batch_size), handle)
                handle.write("\n")
            os.replace(temp_matrix, path / "queries.npy")
            os.replace(temp_manifest, path / "queries.json")
        finally:
            del matrix
            temp_matrix.unlink(missing_ok=True)
            temp_manifest.unlink(missing_ok=True)
    return load_query_embeddings(root, model, dataset, max_queries)


def initial_token_id(tokenizer, cand_mask):
    for name in ("mask", "unk", "pad", "bos", "eos"):
        token_id = getattr(tokenizer, f"{name}_token_id", None)
        if token_id is not None and 0 <= token_id < len(cand_mask) and cand_mask[token_id]:
            return token_id
    raise ValueError("No usable special token for padding the gold initialization")


def targets_cache_path(cache_root, dataset, model, k, seed):
    return Path(cache_root) / dataset / model / f"k{k}_seed{seed}.pt"


def prepare_targets(encoder, dataset, k, seed, cache_root, n_adv=50,
                    max_queries=None, query_embeddings=None, query_embedding_root=DEFAULT_QUERY_EMBEDDING_ROOT):
    if query_embeddings is None:
        query_embeddings = load_query_embeddings(query_embedding_root, encoder.model_name, dataset, max_queries)
    if query_embeddings.embeddings.shape[1] != encoder.embedding_weight.shape[1]:
        raise ValueError("Training query embedding dimension does not match the attack model")
    path = targets_cache_path(cache_root, dataset, encoder.model_name, k, seed)
    meta = dict(version=CACHE_VERSION, model=encoder.model_name, dataset=dataset, k=k, seed=seed,
                n_adv=n_adv, max_queries=max_queries, transformers=transformers.__version__,
                pre_sp=encoder.pre_sp, suf_sp=encoder.suf_sp, prompts=encoder.prompts,
                embedding_rows=encoder.embedding_weight.shape[0], query_embeddings=query_embeddings.identity)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["meta"] != meta:
            raise ValueError(f"Target cache settings differ: {path}. Use a separate cache_root for a new run.")
        logger.info("Resuming with cluster targets from %s", path)
        return [ClusterTarget(t["train_means"].numpy(), t["valid_means"].numpy(),
                              t["gold_init_ids"], t["tot"], t["n_pairs"]) for t in saved["targets"]]

    _, golds, query_ids = load_train_pairs(dataset, seed, max_queries, return_ids=True)
    if len(query_ids) < k:
        raise ValueError(f"Only {len(query_ids)} training pairs for k={k}")
    query_reps = query_embeddings.for_ids(query_ids)
    logger.info("Clustering %d pairs into %d clusters", len(query_ids), k)
    labels = KMeans(n_clusters=k, random_state=0).fit_predict(query_reps) if k > 1 else np.zeros(len(query_ids), dtype=int)
    targets = []
    pad_id = initial_token_id(encoder.tokenizer, encoder.cand_mask)
    for cluster in range(k):
        indices = np.flatnonzero(labels == cluster)
        if not len(indices):
            raise ValueError(f"Empty cluster {cluster}: k={k} exceeds the distinct training targets")
        rng = random.Random(cluster_seed(seed, cluster))
        gold = golds[rng.choice(indices.tolist())]
        ids = encoder.tokenizer(gold, add_special_tokens=False).input_ids[:n_adv]
        # Some tokenizers expose ids that have no model embedding row (Harrier-270M).
        ids = [i if 0 <= i < len(encoder.cand_mask) and encoder.cand_mask[i] else pad_id for i in ids]
        ids += [pad_id] * (n_adv - len(ids))
        indices = np.random.RandomState(cluster_seed(seed, cluster)).permutation(indices)
        n_valid = min(1000, int(len(indices) * 0.3))
        n_train = len(indices) - n_valid
        train_means = batch_average(create_batches(query_reps[indices[:n_train]], 64))
        if n_valid:
            valid_means = batch_average(create_batches(query_reps[indices[n_train:]], 64))
        else:
            logger.warning("Cluster %d has no held-out queries; validating on its training means", cluster)
            valid_means = train_means.copy()
        targets.append(ClusterTarget(train_means, valid_means, ids, n_valid, len(indices)))

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(meta=meta, targets=[dict(train_means=torch.from_numpy(t.train_means),
                   valid_means=torch.from_numpy(t.valid_means), gold_init_ids=t.gold_init_ids,
                   tot=t.tot, n_pairs=t.n_pairs) for t in targets])
    temp = path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
    logger.info("Saved cluster targets to %s", path)
    return targets


def remove_targets_cache(cache_root, dataset, model, k, seed):
    targets_cache_path(cache_root, dataset, model, k, seed).unlink(missing_ok=True)
