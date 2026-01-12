from transformers import AutoTokenizer,AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from sentence_transformers import SentenceTransformer
import torch
import logging
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

def _get_underlying_model(encoder):
    """Returns the torch.nn.Module that exposes the embedding layer."""
    if isinstance(encoder, SentenceTransformer):
        transformer_module = encoder[0]
        if hasattr(transformer_module, "auto_model"):
            return transformer_module.auto_model
        raise ValueError("SentenceTransformer encoder does not expose an auto_model for gradients.")
    if hasattr(encoder, "hf_model"):
        return encoder.hf_model
    return encoder

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """

    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    if isinstance(model, DPRContextEncoder):
        embeddings = model.ctx_encoder.bert_model.embeddings.word_embeddings
    elif isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    elif hasattr(model, "hf_model"):
        inner_model = model.hf_model
        if hasattr(inner_model, 'embeddings'): # BERT, Roberta, XLM-R
            embeddings = inner_model.embeddings.word_embeddings
        elif hasattr(inner_model, 'get_input_embeddings'): # 通用兜底
             embeddings = inner_model.get_input_embeddings()
        else:
            raise ValueError(f"Unknown model type: {type(inner_model)}")
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings # size is (30522,768)


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids

    # f(a) --> f(b)  =  f'(a) * (b - a) = f'(a) * b

model_code_to_qmodel_name = {  # query encoder
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-question_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-question_encoder-multiset-base",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "tas-b": "msmarco-distilbert-base-tas-b",   #SentenceTransformer
    "dragon": "nthakur/dragon-plus-query-encoder",
    "condenser":"hlyu/co-condenser-marco-retriever_141011_cls",


    "reasonir":"reasonir/ReasonIR-8B",
    "bge_reasoner": "hanhainebula/reason-embed-qwen3-8b-0928",
    "diver": 'AQ-MedAI/Diver-Retriever-4B',
    "qwen3": "Qwen/Qwen3-Embedding-8B",
    "linq": "Linq-AI-Research/Linq-Embed-Mistral",
    "gte": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "bge_m3": "BAAI/bge-m3",

    "qwen3_4B": "Qwen/Qwen3-Embedding-4B",
    "qwen3_0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "diver_1.7B": "AQ-MedAI/Diver-Retriever-1.7B",
    "diver_0.6B": "AQ-MedAI/Diver-Retriever-0.6B",
}

model_code_to_cmodel_name = {  # ctx  encoder
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-ctx_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-ctx_encoder-multiset-base",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "tas-b":"msmarco-distilbert-base-tas-b",     #SentenceTransformer
    "dragon": "nthakur/dragon-plus-context-encoder",
    "condenser":"hlyu/co-condenser-marco-retriever_141011_cls",

    "reasonir":"reasonir/ReasonIR-8B",
    "bge_reasoner": "hanhainebula/reason-embed-qwen3-8b-0928",
    "diver": 'AQ-MedAI/Diver-Retriever-4B',
    "qwen3": "Qwen/Qwen3-Embedding-8B",
    "linq": "Linq-AI-Research/Linq-Embed-Mistral",
    "gte": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "bge_m3": "BAAI/bge-m3",

    "qwen3_4B": "Qwen/Qwen3-Embedding-4B",
    "qwen3_0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "diver_1.7B": "AQ-MedAI/Diver-Retriever-1.7B",
    "diver_0.6B": "AQ-MedAI/Diver-Retriever-0.6B",
}

import os,json
def get_model_prompts_tasks(model_name, dataset_name):
    '''

    :param model_name:
    :param dataset_name:
    :return: Dict{query : query_task, passage: doc_task}
    '''
    Dataset_NAME_ALIASES = {
        "nq-train": "nq",
    }
    dataset_name = Dataset_NAME_ALIASES.get(dataset_name, dataset_name)
    # 获取父目录路径（调用 utils.py 的路径的父级目录）
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # 构建正确的 prompts 文件路径
    prompts_path = os.path.join(parent_dir, 'prompts', f'{model_name}.json')

    # 检查 prompts 文件是否存在
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(
            f"Prompt file not found at expected location: {prompts_path}\n"
            "Ensure the file exists and the model_name is correct."
        )
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts_all = json.load(f)
    # 检查 dataset_name 是否存在于 JSON 文件中
    if dataset_name not in prompts_all:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in prompts file: {prompts_path}\n"
            "Ensure the dataset exists in the provided file."
        )

    prompts = prompts_all[dataset_name]

    if  model_name in  ['qwen3','linq','diver','gte','bge_reasoner'] : # qwen3 是需要只对query 加上这样的prompt，document不用
        prompts['query'] = f"Instruct: {prompts['query']}\nQuery:"
        # reasonir 和 bge_m3 和 contriever 不需要

    elif dataset_name=='browsecomp_plus':
        prompts['query'] = f"Instruct: {prompts['query']}\nQuery:"

    return  prompts

def compose_model_inputs(input_ids: torch.Tensor,
                         use_token_type_ids: bool) -> Dict[str, torch.Tensor]:
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if use_token_type_ids:
        model_inputs["token_type_ids"] = torch.zeros_like(input_ids, device=input_ids.device)
    return model_inputs