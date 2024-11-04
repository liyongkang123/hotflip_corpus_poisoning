from transformers import AutoTokenizer,AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder
from transformers import BertModel
from sentence_transformers import SentenceTransformer
import torch
import logging
import os
import requests
import zipfile
from torch.utils.data import DataLoader
from data_loader import GenericDataLoader #beir.datasets
import random
from datasets import Dataset
from transformers import default_data_collator
from tqdm import tqdm
import numpy as np
import time
from collections import Counter
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

def evaluate_acc(model, c_model, get_emb, dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type,
                 data_collator, valid_emb_dic=None, device='cuda'):
    """The code provided by Zhong et al. and it  is very slow"""
    model.eval()
    c_model.eval()
    acc = 0
    tot = 0
    for idx, (data) in tqdm(enumerate(dataloader)):
        data = data_collator(data)  # [bsz, 2, max_len] ,
        # Get query embeddings
        q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
        q_emb = get_emb(model, q_sent)  # [b x d]
        gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
        gold_emb = get_emb(c_model, gold_pass)  # [b x d]
        sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
        p_sent = {'input_ids': adv_passage_ids,
                  'attention_mask': adv_passage_attention,
                  'token_type_ids': adv_passage_token_type}
        p_emb = get_emb(c_model, p_sent)  # [k x d]
        sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]
        acc += (sim_to_gold > sim).sum().cpu().item()
        tot += q_emb.shape[0]
    print(f'Acc = {acc / tot * 100} ({acc} / {tot})')
    return acc / tot

# Re-optimize the evaluate_acc function by us. The original time on 1080Ti is about 8.6 seconds per run.
# def evaluate_acc(model, c_model, get_emb, dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type,
#                  data_collator, valid_emb_dic=None, device='cuda'):
#     """Returns the 2-way classification accuracy (used during training)"""
#
#     model.eval()
#     c_model.eval()
#     acc = 0
#     tot = 0
#     p_sent = {'input_ids': adv_passage_ids,
#               'attention_mask': adv_passage_attention,
#               'token_type_ids': adv_passage_token_type}
#     p_emb = get_emb(c_model, p_sent)  # [k x d]
#
#     if valid_emb_dic is not None: # Here I added the valid_emb_dic parameter. If this parameter exists, use its value directly. Otherwise, recalculate it.
#         q_emb = torch.from_numpy(valid_emb_dic['valid_q_emb']).to(device)
#         gold_emb = torch.from_numpy(valid_emb_dic['valid_gold_emb']).to(device)
#         sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
#         sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]
#         acc += (sim_to_gold > sim).sum().cpu().item()
#         tot += q_emb.shape[0]
#     else:
#         '''The following is the original
#         '''
#         for idx, (data) in tqdm(enumerate(dataloader)): # The valid_dataloader is passed in here and has been tokenized.
#             data = data_collator(data)  # [bsz, 2, max_len] , 2 is the query and its related document
#
#             # Get query embeddings
#             q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()} # Here q_sent is the text of all queries in the validation set
#             q_emb = get_emb(model, q_sent)  # [b x d]
#             gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()} # Here, gold_pass is all the documents in the validation set, and these documents are the relevant text of the query.
#             gold_emb = get_emb(c_model, gold_pass)  # [b x d]
#             sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
#             sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]
#             acc += (sim_to_gold > sim).sum().cpu().item()
#             tot += q_emb.shape[0]
#
#         print(f'Acc = {acc / tot * 100} ({acc} / {tot})')
#     return acc / tot


def evaluate_sim_ours(model, c_model, get_emb, va_dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type,
                      data_collator, device='cuda'):
    model.eval()
    c_model.eval()
    p_sent = {'input_ids': adv_passage_ids.to(device),
              'attention_mask': adv_passage_attention.to(device),
              'token_type_ids': adv_passage_token_type.to(device)}
    p_emb = get_emb(c_model, p_sent)  # [k x d]
    sim_all=[]

    for idx, (data) in tqdm(enumerate(va_dataloader)):
        sim = torch.mm(data, p_emb.T).squeeze()
        sim_all.append(sim.sum().cpu().item())

    return sum(sim_all) / len(sim_all)

