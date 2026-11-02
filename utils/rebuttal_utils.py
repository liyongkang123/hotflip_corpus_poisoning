import logging
from dataclasses import dataclass

import torch
import transformers
from packaging import version
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from utils.load_model import HFtoSF
from utils.utils import get_model_prompts_tasks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    tokenizer_kwargs: dict
    model_kwargs: dict
    pooling: str
    normalize: bool
    max_len: int
    score_function: str
    post_load: str | None


def _last_token(hf_id, tokenizer_kwargs=None, model_kwargs=None, post_load=None):
    return ModelSpec(hf_id, dict(tokenizer_kwargs or {}), dict(model_kwargs or {}), "last", True, 8192, "cos_sim", post_load)


# must match LLM_Robust/utils/load_model.py::load_model_hf
_LEFT = {"padding_side": "left"}
_REMOTE = {"trust_remote_code": True}
MODEL_SPECS = {
    "contriever": ModelSpec("facebook/contriever-msmarco", {}, dict(_REMOTE), "mean", False, 512, "dot", None),
    "bge_m3": ModelSpec("BAAI/bge-m3", {}, dict(_REMOTE), "cls", True, 8192, "cos_sim", None),
    "qwen3": _last_token("Qwen/Qwen3-Embedding-8B", _LEFT),
    "qwen3_4B": _last_token("Qwen/Qwen3-Embedding-4B", _LEFT),
    "qwen3_0.6B": _last_token("Qwen/Qwen3-Embedding-0.6B", _LEFT),
    "linq": _last_token("Linq-AI-Research/Linq-Embed-Mistral"),
    "gte": _last_token("Alibaba-NLP/gte-Qwen2-7B-instruct", _REMOTE, _REMOTE, post_load="disable_use_cache"),
    "reasonir": ModelSpec("reasonir/ReasonIR-8B", {}, dict(_REMOTE), "mask_prompt_mean", True, 8192, "cos_sim", None),
    "diver": _last_token("AQ-MedAI/Diver-Retriever-4B", _LEFT),
    "diver_0.6B": _last_token("AQ-MedAI/Diver-Retriever-0.6B", _LEFT),
    "bge_reasoner": _last_token("hanhainebula/reason-embed-qwen3-8b-0928"),
    "harrier-oss-v1_27B": _last_token("microsoft/harrier-oss-v1-27b", _LEFT),
    "harrier-oss-v1_0.6B": _last_token("microsoft/harrier-oss-v1-0.6b", _LEFT),
    "harrier-oss-v1_270M": _last_token("microsoft/harrier-oss-v1-270m", _LEFT),
}
REBUTTAL_MODELS = list(MODEL_SPECS)

# must match the instruction list of LLM_Robust/utils/utils.py::get_model_prompts_tasks
INSTRUCT_MODELS = {'qwen3', 'qwen3_4B', 'qwen3_0.6B', 'linq', 'diver', 'diver_1.7B', 'diver_0.6B', 'gte', 'bge_reasoner',
                   'nv_embed_v2', 'harrier-oss-v1_27B', 'harrier-oss-v1_270M', 'harrier-oss-v1_0.6B'}


def get_rebuttal_prompts(model, dataset):
    prompts = get_model_prompts_tasks(model, dataset)
    # utils.utils already wraps browsecomp_plus for every model
    if model.startswith("harrier-oss-v1_") and dataset != "browsecomp_plus":
        prompts["query"] = f"Instruct: {prompts['query']}\nQuery:"
    return prompts


def load_tokenizer(model):
    spec = MODEL_SPECS[model]
    return AutoTokenizer.from_pretrained(spec.hf_id, **spec.tokenizer_kwargs)


def special_token_ids(tok, probe="hello world"):
    full = tok(probe).input_ids
    plain = tok(probe, add_special_tokens=False).input_ids
    for i in range(len(full) - len(plain) + 1):
        if full[i:i + len(plain)] == plain:
            return full[:i], full[i + len(plain):]
    raise ValueError(f"{tok.name_or_path}: {plain} is not a contiguous part of {full}")


def _prefix_ids(tok, passage_prefix):
    return tok(passage_prefix, add_special_tokens=False).input_ids


def build_doc_ids(tok, passage_prefix, adv_ids):
    """Build the shared id-level attack input; text tokenization may merge prefix boundaries."""
    pre_sp, suf_sp = special_token_ids(tok)
    return pre_sp + _prefix_ids(tok, passage_prefix) + list(adv_ids) + suf_sp


def adv_positions(tok, passage_prefix, n_adv):
    start = len(special_token_ids(tok)[0]) + len(_prefix_ids(tok, passage_prefix))
    return slice(start, start + n_adv)


def doc_prefix_len(tok, passage_prefix):
    """Match clean ReasonIR's prompt mask (only used for mask_prompt_mean pooling)."""
    return len(tok(passage_prefix, add_special_tokens=True).input_ids)


def candidate_mask(tok, num_rows):
    ids = torch.tensor(list(tok.get_vocab().values()))
    no_row = ids[ids >= num_rows].sort().values.tolist()
    if no_row:
        logger.warning(f"{tok.name_or_path}: {len(no_row)} token ids >= {num_rows} embedding rows are not candidates: {no_row[:10]}")
    mask = torch.zeros(num_rows, dtype=torch.bool)
    mask[ids[(ids >= 0) & (ids < num_rows)]] = True
    return mask


def bf16_kwargs():
    if version.parse(transformers.__version__) >= version.parse("4.56"):
        return {"dtype": torch.bfloat16}
    return {"torch_dtype": torch.bfloat16}


def load_hf_model(model, device_map=None):
    spec = MODEL_SPECS[model]
    kwargs = {**bf16_kwargs(), **spec.model_kwargs}
    if device_map is not None:
        kwargs["device_map"] = device_map
    hf_model = AutoModel.from_pretrained(spec.hf_id, **kwargs)
    # Embedding and HotFlip forwards never need generation KV caches.
    if hasattr(hf_model.config, "use_cache"):
        hf_model.config.use_cache = False
    return hf_model.eval().requires_grad_(False)


def _hftosf(hf_model, tok, prompt, spec, device):
    sf = HFtoSF(hf_model, tok, prompt=prompt, normalize=spec.normalize, pooling=spec.pooling, max_seq_length=spec.max_len,
                device=device)
    # LLM_Robust prepends the prompt without HFtoSF's trailing space and masks it as in its _pooling
    sf.prompt = prompt
    if spec.pooling == "mask_prompt_mean":
        sf.prompt_tokens = tok(prompt, padding=False, add_special_tokens=True, return_tensors="pt")
    return sf


class RebuttalEncoder:
    def __init__(self, model, dataset, device="cuda", device_map=None, hf_model=None, cand_chunk=100):
        self.model_name, self.spec = model, MODEL_SPECS[model]
        self.tokenizer = load_tokenizer(model)
        self.prompts = get_rebuttal_prompts(model, dataset)
        self.model = (load_hf_model(model, device_map) if hf_model is None else hf_model).eval().requires_grad_(False)
        # HFtoSF calls .to(device), which would gather a multi-device device_map onto one GPU; .to(None) moves nothing
        sf_device = device if device_map is None else None
        self.q_model = _hftosf(self.model, self.tokenizer, self.prompts["query"], self.spec, sf_device)
        self.c_model = _hftosf(self.model, self.tokenizer, self.prompts["passage"], self.spec, sf_device)
        self.embedding_weight = self.model.get_input_embeddings().weight
        self.device = torch.device(device) if device_map is None else self.embedding_weight.device
        self.q_model.device = self.c_model.device = self.device

        self.pre_sp, self.suf_sp = special_token_ids(self.tokenizer)
        split = adv_positions(self.tokenizer, self.prompts["passage"], 0).start
        empty = build_doc_ids(self.tokenizer, self.prompts["passage"], [])
        self.doc_head = torch.tensor(empty[:split], dtype=torch.long, device=self.device)
        self.doc_tail = torch.tensor(empty[split:], dtype=torch.long, device=self.device)
        self.cand_mask = candidate_mask(self.tokenizer, self.embedding_weight.shape[0])
        self.cand_chunk = cand_chunk

    def eval(self):
        self.model.eval()
        return self

    def encode_queries(self, texts, batch_size=64):
        texts = [t if t.strip() else "[PAD]" for t in texts]
        batches = range(0, len(texts), batch_size)
        return torch.cat([self.q_model.encode(texts[i:i + batch_size]).float().cpu()
                          for i in tqdm(batches, desc="Encoding queries", disable=len(texts) <= batch_size)])

    def doc_ids(self, adv_ids):
        return build_doc_ids(self.tokenizer, self.prompts["passage"], adv_ids)

    def _embed(self, ids):
        mask = torch.ones_like(ids)
        hidden = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.c_model._pooling(hidden, mask).float()

    def _embed_chunks(self, ids, chunk):
        return torch.cat([self._embed(part) for part in ids.to(self.device).split(chunk)])

    def embed_doc_ids(self, ids, chunk=100):
        with torch.no_grad():
            return self._embed_chunks(ids, chunk)

    def doc_embedding_and_grad(self, ids, loss_fn):
        outputs = []

        # frozen weights leave the embedding output out of the graph, so a backward hook would never fire
        def track(module, inputs, output):
            output.requires_grad_()
            output.retain_grad()
            outputs.append(output)

        handle = self.model.get_input_embeddings().register_forward_hook(track)
        try:
            with torch.enable_grad():
                loss = loss_fn(self._embed(ids.to(self.device)))
                loss.backward()
        finally:
            handle.remove()
        return loss.item(), outputs[0].grad[0]


def rebuttal_get_emb(c_model, inputs):
    adv = inputs["input_ids"].to(c_model.device)
    n = adv.shape[0]
    ids = torch.cat([c_model.doc_head.expand(n, -1), adv, c_model.doc_tail.expand(n, -1)], dim=1)
    return c_model._embed_chunks(ids, c_model.cand_chunk)
