from transformers import AutoTokenizer,AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from sentence_transformers import SentenceTransformer
import torch
import logging


logger = logging.getLogger(__name__)


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
}