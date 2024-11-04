import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer,AutoModel , BertTokenizerFast
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder
from transformers import BertModel
from sentence_transformers import SentenceTransformer
import torch
import logging

logger = logging.getLogger(__name__)

from .utils import model_code_to_qmodel_name,model_code_to_cmodel_name


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":  # average pooling
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def bi_encoder_senctence_transformer_get_emb(model, input): #  All models provided by SentenceTransformer
    # here tas-b ， ance, dragon, condenser are from SentenceTransformer
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def load_models(model_code):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        q_model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = q_model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code],use_fast=True)
        get_emb = contriever_get_emb
    elif 'dpr' in model_code:
        q_model = DPRQuestionEncoder.from_pretrained(model_code_to_qmodel_name[model_code])
        c_model = DPRContextEncoder.from_pretrained(model_code_to_cmodel_name[model_code])
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = dpr_get_emb
    elif any(sub_model_code in model_code for sub_model_code in ['ance', 'tas', 'condenser']):
        q_model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = q_model
        tokenizer = q_model.tokenizer
        get_emb = bi_encoder_senctence_transformer_get_emb
    elif 'dragon' in model_code:
        q_model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        c_model = SentenceTransformer(model_code_to_cmodel_name[model_code])
        tokenizer = q_model.tokenizer
        get_emb = bi_encoder_senctence_transformer_get_emb
    else:
        raise NotImplementedError

    return q_model, c_model, tokenizer, get_emb