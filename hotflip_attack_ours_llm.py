'''
Use cluster centers and batches to speed up calculations
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
from tqdm import tqdm
from typing import Dict, List, Optional, Sequence, Tuple
from utils.load_model import load_models
from utils.load_data import load_data_ours_batch_new
from utils.evaluate import evaluate_sim_ours_new

import utils.utils as utils
import config
from model.hotflip import hotflip_candidate,hotflip_candidate_score_new

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
    #     config=vars(args),
    # )
    file_name = "results/%s-generate/%s/%s/k%d-s%d-seed%d-num_cand%d-num_iter%d-tokens%d-gold_init%s.json" % (
         args.method, args.attack_dataset, args.attack_model_code, args.k, args.kmeans_split, args.seed, args.num_cand,
        args.num_iter, args.num_adv_passage_tokens,args.init_gold)
    args.output_file = file_name
    # create output directory if it doesn't exist
    output_dir_name = os.path.dirname(args.output_file)
    # if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name,exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed) # set seed for reproducibility

    # Load models
    q_model, c_model, tokenizer, get_emb = load_models(args.attack_model_code, args.attack_dataset) # c_model is ctx model

    q_model.eval() # query model and context model
    q_model.to(device)
    c_model.eval()
    c_model.to(device)

    use_token_type_ids = hasattr(tokenizer, "model_input_names") and "token_type_ids" in tokenizer.model_input_names
    print('use_token_type_ids:', use_token_type_ids)

    # Load datasets
    data_collator, train_loader, valid_embeddings, num_valid, gold_passage_init = load_data_ours_batch_new(args, tokenizer, q_model, c_model, get_emb)

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
                print('Insufficient data!')
                break
            
            p_sent = utils.compose_model_inputs(adv_passage_ids, use_token_type_ids)
            p_emb = get_emb(c_model, p_sent)
            # Compute loss
            sim = torch.mm(it_centrid_embedding, p_emb.T)  # [b x k]
            loss = sim.mean()
            # print('loss', loss.cpu().item())
            loss.backward()
            current_score = loss.item() # Reuse this value directly!

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
            print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0])) # Reduce I/O operations

            improve_flag = True
        else:
            print('No improvement detected!')
            improve_flag = False
            continue

        if improve_flag: # Only evaluate when adv_passage_ids has changed
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
if __name__ == "__main__":
    main()