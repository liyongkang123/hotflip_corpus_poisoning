'''
Using my optimized code, attack corpus
# Currently attacking ArguAna, FiQA, these two datasets
We perform all adversarial document generation on a single GPU, so that we do not need to perform clustering operations on each GPU, saving time.
'''

import copy
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
import itertools
import wandb
logger = logging.getLogger(__name__)
from utils.data_loader import GenericDataLoader, CustomDataset_encode_corpus,Attack_Batch_Dataset
from utils.load_model import load_models
from utils.evaluate import evaluate_sim_ours
import utils.load_data as ld
import utils.utils as utils
import config
from model.hotflip import hotflip_candidate, hotflip_candidate_score
from torch.utils.data import DataLoader
import numpy as np
import faiss
from tqdm import tqdm
from utils.load_data import create_batches,batch_average
from evaluate_attack import evaluate_recall

def main():
    prep_start_time = time.time()
    args=  config.parse()

    args.split = 'test' # When attacking the corpus, use the test dataset

    print(args)
    wandb.init(
        # set the wandb project where this run will be logged
        project="attak_generate_corpus",
        config=vars(args),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

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

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.attack_dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    eval_data_path = os.path.join(out_dir, args.attack_dataset)
    if not os.path.exists(eval_data_path):
        eval_data_path = ld.download_and_unzip(url, out_dir)
    logging.info(eval_data_path)
    eval_corpus, eval_queries, eval_qrels = GenericDataLoader(eval_data_path).load(split="test")
    # encode corpus and clustering

    attack_number_p = round( len(eval_corpus)*args.attack_rate) # This is the number of documents that need to be generated during the attack
    print('The total number of clusters is： ',attack_number_p)
    # Cluster k at a time and output multiple file_names at the same time
    file_name_dic={}
    for k_s in range(attack_number_p):
        file_name_dic[k_s]= "results_corpus_attack/%s-generate/%s/%s/k%d-s%d-seed%d-num_cand%d-num_iter%d-tokens%d-gold_init%s.json" % (
         args.method, args.attack_dataset, args.attack_model_code, attack_number_p, k_s, args.seed, args.num_cand,
        args.num_iter, args.num_adv_passage_tokens,args.init_gold)  # Here args.kmeans_split is replaced by k_s


    # Load data and cluster the dataset
    batch_size = 64
    encode_corpus_datasets = CustomDataset_encode_corpus(eval_corpus, 128, tokenizer)
    encode_data_loader = DataLoader(dataset=encode_corpus_datasets, batch_size=batch_size,
                                    shuffle=False, drop_last=False,
                                    collate_fn=encode_corpus_datasets.collate_fn)
    # Embedding corpus and clustering

    # Storing embedding Results
    encoded = []

    for batch_idx, batch_data in enumerate(tqdm(encode_data_loader)):
        batch_input = {key: value.cuda() for key, value in batch_data['encoded_inputs'].items()}
        with torch.no_grad():
            batch_corpus_embs = get_emb(c_model, batch_input)
            encoded.append(batch_corpus_embs.cpu().detach().numpy())
    c_embs = np.concatenate(encoded, axis=0)

    print("c_embs", c_embs.shape)
    #Let’s start clustering

    start_time = time.time()
    kmeans = faiss.Kmeans(c_embs.shape[1], attack_number_p, niter=300, verbose=True, gpu=True, seed=args.seed)
    kmeans.train(c_embs)
    # centroids = kmeans.centroids
    # mean = np.mean(centroids, axis=0)
    end_time = time.time()
    # kmeans_sklearn = KMeans(n_clusters=args.k, random_state=0).fit(c_embs)
    execution_time = end_time - start_time
    print(f"Clustering program running time: {execution_time}秒")
    # Centroids is the average embedding

    # Creating an L2 distance index
    index = faiss.IndexFlatL2(c_embs.shape[1])
    index.add(kmeans.centroids)

    # Search for the nearest cluster center for each sample
    D, I = index.search(c_embs, 1)  # Searching for k=1 means returning only the nearest center

    # I is the cluster label of each sample
    labels = I.flatten()

    # labels is the cluster label of each sample
    print(labels)
    return_data_dic={}

    attack_result_dic={} # In ours, you can directly record all attack results and then directly output the results
    attack_result_p_ids=[]

    for k_s in range(attack_number_p):
        split = k_s
        ret_text = []
        ret_emb_index=[]
        for i in range(c_embs.shape[0]):
            if labels[i] == split:
                i_text = encode_corpus_datasets.data[i]
                if i_text['title'] != "":
                    ret_text.append(i_text['title'] + '.' + i_text['text'])
                else:
                    ret_text.append(i_text['text'])
                ret_emb_index.append(i)
        ret_emb = c_embs[ret_emb_index]

        return_data_dic[split] = ret_text
        data_collator = default_data_collator
        gold_passage_init = random.choice(ret_text)


        num_rows = ret_emb.shape[0]
        val_size = min(1000, int(num_rows * 0.3))
        train_size = num_rows - val_size

        indices = np.arange(num_rows)
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_set = ret_emb[train_indices]
        val_set = ret_emb[val_indices]

        train_batches = create_batches(train_set, batch_size)
        val_batches = create_batches(val_set, batch_size)

        train_avg_set = batch_average(train_batches)
        val_avg_set = batch_average(val_batches)

        print("Training set average data set shape:", train_avg_set.shape)
        print("Validation set average data set shape:", val_avg_set.shape)

        train_dataset = Attack_Batch_Dataset(train_avg_set)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        valid_dataset = Attack_Batch_Dataset(val_avg_set)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

        num_valid = val_size


        args.kmeans_split =k_s
        args.output_file = file_name_dic[k_s]

        output_dir_name = os.path.dirname(args.output_file)
        if not os.path.exists(output_dir_name):
            os.makedirs(output_dir_name)


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

        prep_end_time = time.time()
        search_start_time = time.time()

        for it_ in range(args.num_iter):
            # print(f"Iteration: {it_}")

            # print(f'Accumulating Gradient {args.num_grad_iter}')
            c_model.zero_grad()

            pbar = range(args.num_grad_iter)
            # train_iter_centrid_embedding = iter(train_loader)
            train_iter_centrid_embedding = itertools.cycle(train_loader) #  Because there are many clusters, sometimes there is only one batch, so a loop is needed
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
                # print('loss', loss.cpu().item())
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
                # logger.info('Better adv_passage detected.')
                best_candidate_score = candidate_scores.max()
                best_candidate_idx = candidate_scores.argmax()
                adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                # print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))
            else:
                # print('No improvement detected!')
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

            # print('best_sim', best_sim)
        search_end_time = time.time()
        print(search_end_time-search_start_time,'seconds for searching')
        print(prep_end_time-prep_start_time,'seconds for preparing')
        attack_result_dic[k_s] = best_adv_passage_ids
        attack_result_p_ids.append(best_adv_passage_ids)

    # Start evaluation

    # Load the retrieval results of the beir dataset generated by the embedding_index.py file. Here are all eval_model_code
    beir_result_file = f'{"results/beir_result"}/{args.attack_dataset}/{args.attack_model_code}/beir.json'
    with open(beir_result_file, 'r') as f:
        results = json.load(f)

    assert len(eval_qrels) == len(results)
    print('Total samples:', len(results))

    def evaluate_adv(k, qrels, results):

        # The first step is to load all attack results according to the parameters.
        adv_results = copy.deepcopy(results)
        adv_p_ids = torch.cat(attack_result_p_ids, dim=0).cuda()
        adv_attention = torch.ones_like(adv_p_ids, device='cuda')
        adv_token_type = torch.zeros_like(adv_p_ids, device='cuda')
        adv_input = {'input_ids': adv_p_ids, 'attention_mask': adv_attention, 'token_type_ids': adv_token_type}
        with torch.no_grad():
            adv_embs = get_emb(c_model, adv_input)

        adv_qrels = {q: {"adv%d" % (s): 1 for s in range(k)} for q in qrels}

        for i, query_id in tqdm(enumerate(results)):
            query_text = eval_queries[query_id]
            query_input = tokenizer(query_text,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt",
                                    max_length=args.max_seq_length)
            query_input = {key: value.cuda() for key, value in query_input.items()}
            with torch.no_grad():
                query_emb = get_emb(q_model, query_input)
                adv_sim = torch.mm(query_emb, adv_embs.T)
            for s in range(k):
                adv_results[query_id]["adv%d" % (s)] = adv_sim[0][s].cpu().item()
        adv_eval = evaluate_recall(adv_results, adv_qrels)
        return adv_eval
    final_res = evaluate_adv( attack_number_p, eval_queries, results)
    print('eval_dataset: ', args.attack_dataset, ' model: ', args.attack_model_code,'attack percent: ',args.attack_rate ,' k cluster: ', attack_number_p, ' seed: ', args.seed)
    # Because this is an indomain corpus attack, eval_dataset is the same as attack_dataset.
    print('results: ', final_res)

if __name__ == "__main__":
    main()


