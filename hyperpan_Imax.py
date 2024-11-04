'''
After attacking the document, test the effect of every 1000 steps from 1000 to 20000
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

from utils.load_model import load_models
from utils.load_data import load_data_ours_batch_all
from utils.evaluate import evaluate_sim_ours

import utils.utils as utils
import config
from model.hotflip import hotflip_candidate,hotflip_candidate_score

def main():
    prep_start_time = time.time()
    args=  config.parse()
    args.num_iter = 20001
    print(args)
    wandb.init(
        # set the wandb project where this run will be logged
        project="hyper_generate_Imax",
        config=vars(args),
    )
    file_name_dic={}
    for sub_num_iter in range(1000, 20001, 1000):
        for k_s in range(args.k):
            file_name_dic[str(k_s)+'_'+str(sub_num_iter)]= "results_hyper_iter/%s-generate/%s/%s/k%d-s%d-seed%d-num_cand%d-num_iter%d-tokens%d-gold_init%s.json" % (
             args.method, args.attack_dataset, args.attack_model_code, args.k, k_s, args.seed, args.num_cand,
            sub_num_iter, args.num_adv_passage_tokens,args.init_gold)
    print(file_name_dic)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    num_iter_range = list(range(1000, 20001, 1000))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed) # set seed for reproducibility

    # Load models
    q_model, c_model, tokenizer, get_emb = load_models(args.attack_model_code) # c_model 是ctx model

    q_model.eval() # query model and context model
    q_model.to(device)
    c_model.eval()
    c_model.to(device)

    # Load datasets
    # Here, different clusters are finally transmitted as a dic
    data_collator_dic, train_loader_dic, valid_loader_dic, num_valid_dic, gold_passage_init_dic = load_data_ours_batch_all(args, tokenizer, q_model, c_model, get_emb)

    for k_s in range(args.k):
        args.kmeans_split = k_s

        data_collator = data_collator_dic[k_s]
        train_loader = train_loader_dic[k_s]
        valid_loader = valid_loader_dic[k_s]
        num_valid = num_valid_dic[k_s]
        gold_passage_init = gold_passage_init_dic[k_s]


        # Set up variables for embedding gradients
        embeddings = utils.get_embeddings(c_model)
        print('Model embedding', embeddings)
        embedding_gradient = utils.GradientStorage(embeddings)

        # Initialize adversarial passage with gold passage or not
        if args.init_gold is True:
            adv_passage_ids = tokenizer(gold_passage_init)["input_ids"]
            if len(adv_passage_ids) < args.num_adv_passage_tokens:
                adv_passage_ids += [tokenizer.mask_token_id] * (args.num_adv_passage_tokens - len(adv_passage_ids))
            else:
                adv_passage_ids = adv_passage_ids[:args.num_adv_passage_tokens]
        else:
            adv_passage_ids = [tokenizer.mask_token_id] * args.num_adv_passage_tokens

        print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids))
        adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(0)

        adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
        adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

        best_adv_passage_ids = adv_passage_ids.clone()

        best_sim = evaluate_sim_ours(q_model, c_model, get_emb, valid_loader, best_adv_passage_ids, adv_passage_attention,
                                     adv_passage_token_type, data_collator)

        print(best_sim)

        prep_end_time = time.time()
        search_start_time = time.time()

        output_file_old = ""
        for it_ in range(1, args.num_iter):
            print(f"Iteration: {it_}")

            it_division= it_//1000
            it_remainder = it_%1000
            if it_remainder ==0:
                it_division-=1
            args.output_file = file_name_dic[str(k_s)+'_'+str(num_iter_range[it_division])]
            # print("args.output_file : ",args.output_file )
            # create output directory if it doesn't exist
            output_dir_name = os.path.dirname(args.output_file)
            if not os.path.exists(output_dir_name):
                os.makedirs(output_dir_name)
            # Use the previous output_file_old to initialize, to avoid no file generation when iteration is too high
            if output_file_old!="" and output_file_old!=args.output_file:
                # Read JSON data from source file
                with open(output_file_old, 'r') as f:
                    data_output_file_old = json.load(f)

                # Write the read data to the target file
                with open(args.output_file, 'w') as f:
                    json.dump(data_output_file_old, f)

                output_file_old = args.output_file

            elif output_file_old!=args.output_file:
                output_file_old = args.output_file

            c_model.zero_grad()

            pbar = range(args.num_grad_iter)
            train_iter_centrid_embedding = iter(train_loader)
            grad = None

            for _ in pbar:
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
                loss.backward()
                temp_grad = embedding_gradient.get()
                if grad is None:
                    grad = temp_grad.sum(dim=0) / args.num_grad_iter
                else:
                    grad += temp_grad.sum(dim=0) / args.num_grad_iter

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

            if cur_sim > best_sim:
                best_sim = cur_sim
                best_adv_passage_ids = adv_passage_ids.clone()
                logger.info('!!! Updated best adv_passage')
                print(tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]))
                if args.output_file is not None:
                    with open(args.output_file, 'w') as f:
                        json.dump({"it": it_, "best_sim": best_sim,
                                   "best_adv_text": tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]), "tot": num_valid}, f)
                        print("Write Success")

            print('best_sim', best_sim)

        search_end_time = time.time()
        print(search_end_time-search_start_time,'seconds for searching')
        print(prep_end_time-prep_start_time,'seconds for preparing')

if __name__ == "__main__":
    main()