'''
This is the original code from Zhong et al. (2023)
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
from utils.load_data import load_data
from utils.evaluate import evaluate_acc

import utils.utils as utils
import config


def main():
    prep_start_time = time.time()
    args = config.parse()
    print(args)

    wandb.init(
        # set the wandb project where this run will be logged
        project="hotflip attack raw",
        config=vars(args),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    file_name = "results/%s-generate/%s/%s/k%d-s%d-seed%d-num_cand%d-num_iter%d-tokens%d-gold_init%s.json" % (
         args.method, args.attack_dataset, args.attack_model_code, args.k, args.kmeans_split, args.seed, args.num_cand,
        args.num_iter, args.num_adv_passage_tokens,args.init_gold)
    args.output_file = file_name
    # create output directory if it doesn't exist
    output_dir_name = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)


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
    # data_collator, dataloader, valid_dataloader, num_valid,valid_emb_dic = load_data(args, tokenizer, q_model, get_emb, c_model) # c_model embdding gold_passage
    data_collator, dataloader, valid_dataloader, num_valid = load_data(args, tokenizer, q_model, get_emb )  # c_model embdding gold_passage


    # Set up variables for embedding gradients
    embeddings = utils.get_embeddings(c_model)
    print('Model embedding', embeddings)
    embedding_gradient = utils.GradientStorage(embeddings)

    # Initialize adversarial passage
    adv_passage_ids = [tokenizer.mask_token_id] * args.num_adv_passage_tokens # Here we set to generate 50 tokens, i.e. 50 mask tokens
    print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids))
    adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(0)

    adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
    adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

    best_adv_passage_ids = adv_passage_ids.clone()
    best_acc = evaluate_acc(q_model, c_model, get_emb, valid_dataloader, best_adv_passage_ids, adv_passage_attention,
                            adv_passage_token_type, data_collator,valid_emb_dic=None) # add valid_emb_dic
    # The acc here is the accuracy rate of the relevant document being greater than the adversarial document, so if you want the attack to be effective, the lowest best_acc should be used.

    print(best_acc)

    prep_end_time = time.time()
    search_start_time = time.time()

    for it_ in range(args.num_iter):  ##This code attacks a single cluster, so directly set the number of iterations num_iter 5000
        print(f"Iteration: {it_}")

        print(f'Accumulating Gradient {args.num_grad_iter}')
        c_model.zero_grad()

        pbar = range(args.num_grad_iter)
        train_iter = iter(dataloader)
        grad = None

        for _ in pbar:
            try:
                data = next(train_iter)
                data = data_collator(data)  # [bsz, 3, max_len]
            except:
                print('Insufficient data!')
                break

            q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()} # The data here actually only contains one batch size, so the gradient obtained is only for this batch.
            q_emb = get_emb(q_model, q_sent).detach()

            gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
            gold_emb = get_emb(c_model, gold_pass).detach()

            sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
            sim_to_gold_mean = sim_to_gold.mean().cpu().item()
            print('Avg sim to gold p =', sim_to_gold_mean)

            # Initialize the adversarial passage with a gold passage
            # if it_ == 0 and _ == 0 and best_acc == 1.0 and (not args.dont_init_gold): # The best_acc condition added here is too harsh, so remove it
            if it_ == 0 and _ == 0 and (not args.dont_init_gold):
                print("Init with a gold passage")
                ll = min(len(gold_pass['input_ids'][0]), args.num_adv_passage_tokens)
                adv_passage_ids[0][:ll] = gold_pass['input_ids'][0][:ll]
                print(adv_passage_ids.shape)
                print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))

                best_adv_passage_ids = adv_passage_ids.clone()
                best_acc = evaluate_acc(q_model, c_model, get_emb, valid_dataloader, best_adv_passage_ids,
                                        adv_passage_attention, adv_passage_token_type, data_collator,valid_emb_dic=None)
                print(best_acc)

            p_sent = {'input_ids': adv_passage_ids,
                      'attention_mask': adv_passage_attention,
                      'token_type_ids': adv_passage_token_type}
            p_emb = get_emb(c_model, p_sent)

            # Compute loss
            sim = torch.mm(q_emb, p_emb.T)  # [b x k]
            print(it_, _, 'Avg sim to adv =', sim.mean().cpu().item(), 'sim to gold =', sim_to_gold_mean)

            suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item()
            print('Attack on train: %d / %d' % (suc_att, sim_to_gold.shape[0]), 'best_acc', best_acc)
            loss = sim.mean()
            print('loss', loss.cpu().item())
            loss.backward()

            temp_grad = embedding_gradient.get()
            if grad is None:
                grad = temp_grad.sum(dim=0) / args.num_grad_iter
            else:
                grad += temp_grad.sum(dim=0) / args.num_grad_iter

        print('Evaluating Candidates')
        pbar = range(args.num_grad_iter)
        train_iter = iter(dataloader)

        token_to_flip = random.randrange(args.num_adv_passage_tokens) #  Randomly select a token to attack
        candidates = utils.hotflip_attack(grad[token_to_flip],
                                    embeddings.weight,
                                    increase_loss=True,
                                    num_candidates=args.num_cand,
                                    filter=None) #Get the top-100 optional token candidates,

        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=device)
        current_acc_rate = 0
        candidate_acc_rates = torch.zeros(args.num_cand, device=device)

        for step in pbar:
            try:
                data = next(train_iter)
                data = data_collator(data)  # [bsz, 3, max_len]
            except:
                print('Insufficient data!')
                break

            q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
            q_emb = get_emb(q_model, q_sent).detach()

            gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
            gold_emb = get_emb(c_model, gold_pass).detach()

            sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
            sim_to_gold_mean = sim_to_gold.mean().cpu().item()
            print('Avg sim to gold p =', sim_to_gold_mean)

            p_sent = {'input_ids': adv_passage_ids,
                      'attention_mask': adv_passage_attention,
                      'token_type_ids': adv_passage_token_type}
            p_emb = get_emb(c_model, p_sent)

            # Compute loss
            sim = torch.mm(q_emb, p_emb.T)  # [b x k]
            print(it_, _, 'Avg sim to adv =', sim.mean().cpu().item(), 'sim to gold =', sim_to_gold_mean)
            suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item()
            print('Attack on train: %d / %d' % (suc_att, sim_to_gold.shape[0]), 'best_acc', best_acc)
            loss = sim.mean()
            temp_score = loss.sum().cpu().item()

            current_score += temp_score
            current_acc_rate += suc_att

            start_time = time.time()
            for i, candidate in enumerate(candidates): # # Find all token documents in turn and test which one is the best
                temp_adv_passage = adv_passage_ids.clone() # temp_adv_passage.shape [1,50]
                temp_adv_passage[:, token_to_flip] = candidate
                p_sent = {'input_ids': temp_adv_passage,
                          'attention_mask': adv_passage_attention,
                          'token_type_ids': adv_passage_token_type}
                p_emb = get_emb(c_model, p_sent)
                with torch.no_grad():
                    sim = torch.mm(q_emb, p_emb.T)
                    can_suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item()
                    can_loss = sim.mean()
                    temp_score = can_loss.sum().cpu().item()

                    candidate_scores[i] += temp_score
                    candidate_acc_rates[i] += can_suc_att
            end_time = time.time()
            print(end_time - start_time,'seconds one hot flipping')


        print(current_score, max(candidate_scores).cpu().item())
        print(current_acc_rate, max(candidate_acc_rates).cpu().item())

        # if find a better one, update
        if (candidate_scores > current_score).any() or (candidate_acc_rates > current_acc_rate).any():
            logger.info('Better adv_passage detected.')
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
            print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))
        else:
            print('No improvement detected!')
            continue

        start_time =time.time()
        cur_acc = evaluate_acc(q_model, c_model, get_emb, valid_dataloader, adv_passage_ids, adv_passage_attention,
                               adv_passage_token_type, data_collator,valid_emb_dic=None)
        end_time = time.time()
        print(end_time - start_time,'seconds evaluate_acc Time required for one time')
        #
        if cur_acc < best_acc: #  The smaller cur_acc is, the better
            best_acc = cur_acc
            best_adv_passage_ids = adv_passage_ids.clone()
            logger.info('!!! Updated best adv_passage')
            print(tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]))

            if args.output_file is not None:
                with open(args.output_file, 'w') as f:
                    json.dump({"it": it_, "best_acc": best_acc,
                               "best_adv_text": tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]), "tot": num_valid}, f)

        print('best_acc', best_acc)

    search_end_time = time.time()
    print(search_end_time-search_start_time, 'seconds for searching')
    print(prep_end_time-prep_start_time, 'seconds for preparing')

if __name__ == "__main__":
    main()