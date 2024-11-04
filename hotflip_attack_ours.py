'''
Use cluster centers and batches to speed up calculations
'''

import logging
import time
import torch
import os
import json
import random
from transformers import (
    set_seed,
default_data_collator,
)
import wandb
logger = logging.getLogger(__name__)
import argparse
# from beir import util

from utils.load_model import load_models
from utils.load_data import load_data_ours_batch
from utils.evaluate import evaluate_sim_ours

import utils.utils as utils
import config
from model.hotflip import hotflip_candidate,hotflip_candidate_score

def main():
    prep_start_time = time.time()
    args=  config.parse()
    print(args)
    wandb.init(
        # set the wandb project where this run will be logged
        project="attak_generate",
        config=vars(args),
    )
    file_name = "results/%s-generate/%s/%s/k%d-s%d-seed%d-num_cand%d-num_iter%d-tokens%d-gold_init%s.json" % (
         args.method, args.attack_dataset, args.attack_model_code, args.k, args.kmeans_split, args.seed, args.num_cand,
        args.num_iter, args.num_adv_passage_tokens,args.init_gold)
    args.output_file = file_name
    # create output directory if it doesn't exist
    output_dir_name = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed) # set seed for reproducibility 我喜欢年份

    # Load models
    q_model, c_model, tokenizer, get_emb = load_models(args.attack_model_code) # c_model 是ctx model

    q_model.eval() # query model and context model
    q_model.to(device)
    c_model.eval()
    c_model.to(device)

    # Load datasets
    data_collator, train_loader, valid_loader, num_valid, gold_passage_init = load_data_ours_batch(args, tokenizer, q_model, c_model, get_emb)

    # Set up variables for embedding gradients
    embeddings = utils.get_embeddings(c_model)
    print('Model embedding', embeddings)
    embedding_gradient = utils.GradientStorage(embeddings)

    # Initialize adversarial passage with gold passage or not
    if args.init_gold is True:
        adv_passage_ids = tokenizer(gold_passage_init)["input_ids"]
        if len(adv_passage_ids) < args.num_adv_passage_tokens:
            # 使用 0 填充到指定长度
            adv_passage_ids += [tokenizer.mask_token_id] * (args.num_adv_passage_tokens - len(adv_passage_ids))
        else:
            # 如果足够长，则直接切片
            adv_passage_ids = adv_passage_ids[:args.num_adv_passage_tokens]
    else:
        adv_passage_ids = [tokenizer.mask_token_id] * args.num_adv_passage_tokens # 这里设置要生成 50 个 token, 即 50 个 mask token

    print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids))
    adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(0)

    adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
    adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

    best_adv_passage_ids = adv_passage_ids.clone() #  深拷贝（deep copy）操作， 两个张量相互独立

    best_sim = evaluate_sim_ours(q_model, c_model, get_emb, valid_loader, best_adv_passage_ids, adv_passage_attention,
                                 adv_passage_token_type, data_collator)

    print(best_sim)

    prep_end_time = time.time()
    search_start_time = time.time()

    for it_ in range(args.num_iter):  #这个代码是对单个聚类簇进行攻击，所以直接设置迭代次数 num_iter 5000
        print(f"Iteration: {it_}")

        # print(f'Accumulating Gradient {args.num_grad_iter}')
        c_model.zero_grad()

        pbar = range(args.num_grad_iter)
        train_iter_centrid_embedding = iter(train_loader)
        grad = None

        for _ in pbar: #这里的_ 是占位符，表示不需要使用这个变量，所以不需要赋值 ，实际上只会运行一次这里的for循环
            try:
                it_centrid_embedding = next(train_iter_centrid_embedding)

            except:
                print('Insufficient data!')
                break
            p_sent = {'input_ids': adv_passage_ids,
                      'attention_mask': adv_passage_attention,
                      'token_type_ids': adv_passage_token_type}
            p_emb = get_emb(c_model, p_sent)
            # Compute loss
            sim = torch.mm(it_centrid_embedding, p_emb.T)  # [b x k]
            loss = sim.mean()
            # print('loss', loss.cpu().item())
            loss.backward()

            # train_q_centrid_em 的模长仅有0.538， 而 p_emb 的模长是 4.7
            temp_grad = embedding_gradient.get()
            if grad is None:
                grad = temp_grad.sum(dim=0) / args.num_grad_iter
            else:
                grad += temp_grad.sum(dim=0) / args.num_grad_iter

        # print('Evaluating Candidates')

        token_to_flip, candidates = hotflip_candidate(args, grad, embeddings)
        current_score, candidate_scores = hotflip_candidate_score(args, it_,
                    candidates, pbar, train_iter_centrid_embedding, data_collator, get_emb, q_model, c_model,
                    adv_passage_ids, adv_passage_attention, adv_passage_token_type,token_to_flip, device=device)



        # if find a better one, update
        if (candidate_scores > current_score).any() :
            logger.info('Better adv_passage detected.')
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
            print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))
        else:
            print('No improvement detected!')
            continue
        start_time =time.time()
        cur_sim = evaluate_sim_ours(q_model, c_model, get_emb, valid_loader, adv_passage_ids, adv_passage_attention,
                                    adv_passage_token_type, data_collator)
        end_time = time.time()
        # print(end_time - start_time,'seconds evaluate_acc 一次所需的时间')

        if cur_sim > best_sim: #  cur_acc 越小越好
            best_sim = cur_sim
            best_adv_passage_ids = adv_passage_ids.clone()
            logger.info('!!! Updated best adv_passage')
            print(tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]))
            if args.output_file is not None:
                with open(args.output_file, 'w') as f:
                    json.dump({"it": it_, "best_sim": best_sim,
                               "best_adv_text": tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]), "tot": num_valid}, f)

        print('best_sim', best_sim)

    search_end_time = time.time()
    print(search_end_time-search_start_time,'seconds for searching')
    print(prep_end_time-prep_start_time,'seconds for preparing')
if __name__ == "__main__":
    main()