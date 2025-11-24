'''
batch 版本使用centroid
并且一次centorid 得到多个kmeans 的数据，这样就不需要重复处理数据了

# 复制来源于 attack-baseline 中的 vscode-remote://ssh-remote%2Bsnellius-yli4/gpfs/work4/0/prjs0928/attack_baseline/hotflip_attack_yk_batch_ms_all.py
'''

import logging
import time
import torch
import os
import json
import random
# from transformers import (
#     set_seed,
# default_data_collator,
# )
from transformers.trainer_utils import set_seed
import wandb
logger = logging.getLogger(__name__)
import argparse
# from beir import util

from utils.load_model import load_models
from utils.load_data import load_data_ours_batch_all
from utils.evaluate import  evaluate_sim_ours_new

import utils.utils as utils
import config
from model.hotflip import hotflip_candidate,hotflip_candidate_score_new
from tqdm import tqdm
from pathlib import Path

def resolve_initial_token_id(tokenizer) -> int:
    candidate_ids = [
        getattr(tokenizer, "mask_token_id", None),
        getattr(tokenizer, "unk_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ]
    print('candidate_ids: ', candidate_ids)
    for token_id in candidate_ids:
        if token_id is not None:
            return token_id
    raise ValueError("Tokenizer does not provide any special tokens for initialization.")

def main():
    prep_start_time = time.time()
    args=  config.parse()
    print(args)
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="attak_generate",
    #     # project="attack_time_test",
    #     # track hyperparameters and run metadata
    #     config=vars(args),
    # )
    # 一次聚类k 个,同时输出多个 file_name
    file_name_dic={}
    for k_s in range(args.k):
        file_name_dic[k_s]= "results/%s-generate/%s/%s/k%d-s%d-seed%d-num_cand%d-num_iter%d-tokens%d-gold_init%s.json" % (
         args.method, args.attack_dataset, args.attack_model_code, args.k, k_s, args.seed, args.num_cand,
        args.num_iter, args.num_adv_passage_tokens,args.init_gold)  # 这里的 args.kmeans_split 被替换成了 k_s

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # 使用一个文档再记录所有的 聚类簇对应的对抗文档
    output_dir = Path(f"output_attack/attacked_text/document/{args.attack_dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 构建包含超参数的文件名，方便区分实验配置
    file_name = (
        f"{args.attack_dataset}_train_attacked_documents_"
        f"seed_{args.seed}_"
        f"{args.attack_model_code}_supervised_hotflip_"
        f"N{args.k}_"  # 攻击文档数量
        f"Len{args.num_adv_passage_tokens}_"  # 对抗 Token 长度
        f"Iter{args.num_iter}_"  # 迭代次数
        f"Pool{args.num_cand}_"  # 候选池大小
        f"Init{args.init_gold}" # 初始化方法
        f".json"
    )
    all_output_data = {}

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed) # set seed for reproducibility 我喜欢年份

    # Load models
    q_model, c_model, tokenizer, get_emb = load_models(args.attack_model_code ,args.attack_dataset) # c_model 是ctx model

    q_model.eval() # query model and context model
    q_model.to(device) 
    c_model.eval()
    c_model.to(device)

    use_token_type_ids = hasattr(tokenizer, "model_input_names") and "token_type_ids" in tokenizer.model_input_names
    print('use_token_type_ids:', use_token_type_ids)

    # Load datasets
    # 这里不同的 聚类簇 最后传出来的是一个dic
    data_collator_dic, train_loader_dic, valid_loader_dic, num_valid_dic, gold_passage_init_dic = load_data_ours_batch_all(args, tokenizer, q_model, c_model,get_emb)

    for k_s in range(args.k):
        print(f"================= Attacking kmeans split {k_s} =================")
        all_output_data[k_s] = {}
        args.kmeans_split =k_s
        args.output_file = file_name_dic[k_s]
        # create output directory if it doesn't exist
        output_dir_name = os.path.dirname(args.output_file)
        # if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name, exist_ok=True)

        data_collator = data_collator_dic[k_s]
        train_loader = train_loader_dic[k_s]
        valid_embeddings = valid_loader_dic[k_s]
        num_valid = num_valid_dic[k_s]
        gold_passage_init = gold_passage_init_dic[k_s]


        # Set up variables for embedding gradients
        embeddings = utils.get_embeddings(c_model)
        print('Model embedding', embeddings)
        embedding_gradient = utils.GradientStorage(embeddings)

        mask_token_id = resolve_initial_token_id(tokenizer)
        print('mask_token_id: ', mask_token_id)
        # Initialize adversarial passage with gold passage or not
        if args.init_gold is True:
            adv_passage_ids = tokenizer(gold_passage_init)["input_ids"]
            if len(adv_passage_ids) < args.num_adv_passage_tokens:
                # Use mask_token_id to fill to the specified length
                adv_passage_ids += [mask_token_id] * (args.num_adv_passage_tokens - len(adv_passage_ids))
            else:
                # If it is long enough, cut it directly
                adv_passage_ids = adv_passage_ids[:args.num_adv_passage_tokens]
        else:
            adv_passage_ids = [mask_token_id] * args.num_adv_passage_tokens # Here we set to generate 50 tokens, i.e. 50 mask tokens

        print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids))
        adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(0)

        adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
        adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

        best_adv_passage_ids = adv_passage_ids.clone()

        best_sim = evaluate_sim_ours_new(q_model, c_model, get_emb, valid_embeddings, best_adv_passage_ids, use_token_type_ids,
                                        device=device)

        print(best_sim)

        prep_end_time = time.time()
        search_start_time = time.time()

        # for it_ in range(args.num_iter):  #This code attacks a single cluster, so directly set the number of iterations num_iter 5000
        for it_ in tqdm(range(args.num_iter), desc=f"Attacking..."):
            # print(f"Iteration: {it_}")

            # print(f'Accumulating Gradient {args.num_grad_iter}')
            c_model.zero_grad()

            pbar = range(args.num_grad_iter)
            train_iter_centrid_embedding = iter(train_loader)
            grad = None

            for _ in pbar: #Here _ is a placeholder, indicating that this variable is not needed, so no value is assigned. In fact, the for loop here will only be run once
                try:
                    it_centrid_embedding = next(train_iter_centrid_embedding)

                except:
                    # 数据耗尽，重新从头开始
                    train_iter_centrid_embedding = iter(train_loader)
                    it_centrid_embedding = next(train_iter_centrid_embedding)
                    print('Insufficient data!')
                    break
                
                p_sent = utils.compose_model_inputs(adv_passage_ids, use_token_type_ids)
                p_emb = get_emb(c_model, p_sent)
                # Compute loss
                sim = torch.mm(it_centrid_embedding, p_emb.T)  # [b x k]
                loss = sim.mean()
                # print('loss', loss.cpu().item())
                loss.backward()
                current_score = loss.item() # 直接复用这个值！

                temp_grad = embedding_gradient.get()
                if grad is None:
                    grad = temp_grad.sum(dim=0) / args.num_grad_iter
                else:
                    grad += temp_grad.sum(dim=0) / args.num_grad_iter

            # print('Evaluating Candidates')

            token_to_flip, candidates = hotflip_candidate(args, grad, embeddings)
            candidate_scores = hotflip_candidate_score_new(args, it_,
                        candidates, pbar, train_iter_centrid_embedding, get_emb,  c_model,
                        adv_passage_ids, token_to_flip, device=device, use_token_type_ids=use_token_type_ids)

            # if find a better one, update
            if (candidate_scores > current_score).any() :
                logger.info('Better adv_passage detected.')
                # best_candidate_score = candidate_scores.max()
                best_candidate_idx = candidate_scores.argmax()
                adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0])) #减少 io 操作

                improve_flag = True
            else:
                print('No improvement detected!')
                improve_flag = False
                continue

            if improve_flag: # 只有当 adv_passage_ids 变化时，才评估
                start_time =time.time()
                cur_sim = evaluate_sim_ours_new(q_model, c_model, get_emb, valid_embeddings, adv_passage_ids, use_token_type_ids, device=device)
                end_time = time.time()


                if cur_sim > best_sim: # The larger the cur_sim, the better
                    best_sim = cur_sim
                    best_adv_passage_ids = adv_passage_ids.clone()
                    logger.info('!!! Updated best adv_passage')
                    print(tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]))
                    if args.output_file is not None:
                        with open(args.output_file, 'w') as f:
                            json.dump({"it": it_, "best_sim": best_sim,
                                    "best_adv_text": tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]), 
                                    "best_adv_passage_ids": best_adv_passage_ids[0].tolist(),
                                    "tot": num_valid}, f)

                    print('best_sim', best_sim)

        search_end_time = time.time()
        print(search_end_time-search_start_time,'seconds for searching')
        print(prep_end_time-prep_start_time,'seconds for preparing')

        all_output_data[k_s] = {
            "best_sim": best_sim,
            "best_adv_text": tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]),
            "best_adv_passage_ids": best_adv_passage_ids[0].tolist(),
            "tot": num_valid
        }
    # 保存所有 k 个聚类簇的对抗结果

    output_path = output_dir / file_name
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved adversarial documents to {output_path}")

if __name__ == "__main__":
    main()