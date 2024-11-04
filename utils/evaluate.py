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

# def evaluate_acc(model, c_model, get_emb, dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type,  # 作者提供的代码，非常慢
#                  data_collator, device='cuda'):
#     """Returns the 2-way classification accuracy (used during training)"""
#
#     # 感觉是计算生成的  adv_passage_ids 和 验证集里面所有的 document 之间的相似度
#     model.eval()
#     c_model.eval()
#     acc = 0
#     tot = 0
#     for idx, (data) in tqdm(enumerate(dataloader)):
#         data = data_collator(data)  # [bsz, 2, max_len] , 2 是 query 和 它的 相关 document
#
#         # Get query embeddings
#         q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()} # 这里的q_sent 是 验证集里面所有的 query 的文本
#         q_emb = get_emb(model, q_sent)  # [b x d]
#
#         gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()} # 这里的gold_pass 是 验证集里面所有的 document， 然后这些document是query的相关文本
#         gold_emb = get_emb(c_model, gold_pass)  # [b x d]
#
#         sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
#
#         p_sent = {'input_ids': adv_passage_ids,
#                   'attention_mask': adv_passage_attention,
#                   'token_type_ids': adv_passage_token_type}
#         p_emb = get_emb(c_model, p_sent)  # [k x d] # 这是对抗文档的相似度
#
#         sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]
#
#         acc += (sim_to_gold > sim).sum().cpu().item()
#         tot += q_emb.shape[0]
#
#     print(f'Acc = {acc / tot * 100} ({acc} / {tot})')  # 这里的acc 是 datloader 里面的相关文本对比query相似 比 生成的adv_passage_ids 还要高的比例
#     return acc / tot

# 重新优化 evaluate_acc 函数 原来在1080Ti上的时间大概是 8.6 秒 一次
def evaluate_acc(model, c_model, get_emb, dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type,
                 data_collator, valid_emb_dic, device='cuda'):
    """Returns the 2-way classification accuracy (used during training)"""

    # 感觉是计算生成的  adv_passage_ids 和 验证集里面所有的 document 之间的相似度
    model.eval()
    c_model.eval()
    acc = 0
    tot = 0
    p_sent = {'input_ids': adv_passage_ids,
              'attention_mask': adv_passage_attention,
              'token_type_ids': adv_passage_token_type}
    p_emb = get_emb(c_model, p_sent)  # [k x d] # 这是对抗文档的相似度

    if valid_emb_dic is not None: # 这里我增加了 valid_emb_dic 参数，如果有这个参数，就直接使用这个参数的值，否则就重新计算
        q_emb = torch.from_numpy(valid_emb_dic['valid_q_emb']).to(device)
        gold_emb = torch.from_numpy(valid_emb_dic['valid_gold_emb']).to(device)
        sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
        sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]
        acc += (sim_to_gold > sim).sum().cpu().item()
        tot += q_emb.shape[0]
    else:
        '''下面这个是原来的
        '''
        for idx, (data) in tqdm(enumerate(dataloader)): # 这里传入的是 valid_dataloader，并且已经经过了 tokenization
            data = data_collator(data)  # [bsz, 2, max_len] , 2 是 query 和 它的 相关 document

            # Get query embeddings
            q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()} # 这里的q_sent 是 验证集里面所有的 query 的文本
            q_emb = get_emb(model, q_sent)  # [b x d]
            gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()} # 这里的gold_pass 是 验证集里面所有的 document， 然后这些document是query的相关文本
            gold_emb = get_emb(c_model, gold_pass)  # [b x d]
            sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
            sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]
            acc += (sim_to_gold > sim).sum().cpu().item()
            tot += q_emb.shape[0]

        print(f'Acc = {acc / tot * 100} ({acc} / {tot})')  # 这里的acc 是 datloader 里面的相关文本对比query相似 比 生成的adv_passage_ids 还要高的比例
    return acc / tot


def evaluate_sim_ours(model, c_model, get_emb, va_dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type,
                      data_collator, device='cuda'):
    model.eval()
    c_model.eval()
    p_sent = {'input_ids': adv_passage_ids.to(device),
              'attention_mask': adv_passage_attention.to(device),
              'token_type_ids': adv_passage_token_type.to(device)}
    p_emb = get_emb(c_model, p_sent)  # [k x d] # 这是对抗文档的相似度
    sim_all=[]

    for idx, (data) in tqdm(enumerate(va_dataloader)):
        sim = torch.mm(data, p_emb.T).squeeze()
        sim_all.append(sim.sum().cpu().item())

    return sum(sim_all) / len(sim_all)

