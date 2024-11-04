'''
使用我优化后的代码，攻击corpus
'''

# 目前攻击 ArguAna, FiQA,  这三个数据集

'''
batch 版本使用centroid
并且一次centorid 得到多个kmeans 的数据，这样就不需要重复处理数据了

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
import argparse
# from beir import util
from utils.data_loader import GenericDataLoader, CustomDataset_encode_corpus,Attack_Batch_Dataset
from utils.load_model import load_models
from utils.evaluate import evaluate_acc,evaluate_acc_yk ,evaluate_sim_ours
import utils.load_data as ld
import utils.utils as utils
import config
from model.hotflip import hotflip_candidate,hotflip_candidate_score
from torch.utils.data import DataLoader
import numpy as np
import faiss
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.load_data import create_batches,batch_average
from evaluate_attack import evaluate_recall

def main():
    prep_start_time = time.time()
    args=  config.parse()

    args.split = 'test' # attack corpus 的时候，使用 test 数据集

    print(args)
    wandb.init(
        # set the wandb project where this run will be logged
        project="attak_generate_corpus",
        # project="attack_time_test",
        # track hyperparameters and run metadata
        config=vars(args),
    )

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

    '''单独加载数据'''
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.attack_dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    eval_data_path = os.path.join(out_dir, args.attack_dataset)
    if not os.path.exists(eval_data_path):
        eval_data_path = ld.download_and_unzip(url, out_dir)
    logging.info(eval_data_path)
    eval_corpus, eval_queries, eval_qrels = GenericDataLoader(eval_data_path).load(split="test")
    # encode corpus and clustering

    attack_number_p = round( len(eval_corpus)*args.attack_rate) # 这是攻击时需要产生的文档数目
    print('聚类总数为： ',attack_number_p)
    # 一次聚类k 个,同时输出多个 file_name
    file_name_dic={}
    for k_s in range(attack_number_p):
        file_name_dic[k_s]= "results_corpus_attack/%s-generate/%s/%s/k%d-s%d-seed%d-num_cand%d-num_iter%d-tokens%d-gold_init%s.json" % (
         args.method, args.attack_dataset, args.attack_model_code, attack_number_p, k_s, args.seed, args.num_cand,
        args.num_iter, args.num_adv_passage_tokens,args.init_gold)  # 这里的 args.kmeans_split 被替换成了 k_s


    # load data 并对数据集进行聚类
    batch_size = 64
    encode_corpus_datasets = CustomDataset_encode_corpus(eval_corpus, 128, tokenizer)
    encode_data_loader = DataLoader(dataset=encode_corpus_datasets, batch_size=batch_size,
                                    shuffle=False, drop_last=False,
                                    collate_fn=encode_corpus_datasets.collate_fn)
    # embedding corpus并聚类

    # 存储结果
    encoded = []

    for batch_idx, batch_data in enumerate(tqdm(encode_data_loader)):
        batch_input = {key: value.cuda() for key, value in batch_data['encoded_inputs'].items()}
        with torch.no_grad():
            batch_corpus_embs = get_emb(c_model, batch_input)
            encoded.append(batch_corpus_embs.cpu().detach().numpy())
    c_embs = np.concatenate(encoded, axis=0)

    print("c_embs", c_embs.shape)
    #下面开始聚类

    start_time = time.time()
    kmeans = faiss.Kmeans(c_embs.shape[1], attack_number_p, niter=300, verbose=True, gpu=True, seed=args.seed)
    kmeans.train(c_embs)
    centroids = kmeans.centroids
    mean = np.mean(centroids, axis=0)  # 这两个是一样的
    end_time = time.time()
    # kmeans_sklearn = KMeans(n_clusters=args.k, random_state=0).fit(c_embs)
    # 计算运行时间
    execution_time = end_time - start_time
    print(f"聚类程序运行时间: {execution_time}秒")
    # centroids 就是平均的embedding

    # 创建 L2 距离索引
    index = faiss.IndexFlatL2(c_embs.shape[1])
    index.add(kmeans.centroids)

    # 搜索每个样本的最近簇中心
    D, I = index.search(c_embs, 1)  # 搜索 k=1 表示只返回最近的一个中心

    # I 就是每个样本的簇标签
    labels = I.flatten()  # 展平为一维数组

    # labels 是每个样本的聚类标签
    print(labels)
    return_data_dic={}

    attack_result_dic={} # 在ours 中，可以直接记录所有的攻击结果，然后直接输出结果
    attack_result_p_ids=[]

    for k_s in range(attack_number_p):
        split = k_s
        ret_text = []
        ret_emb_index=[]
        for i in range(c_embs.shape[0]):
            if labels[i] == split:  # 这里设置split 是为了在多张卡上进行攻击，每张卡上只攻击一个聚类簇
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

        # Step 1: 按行划分 30% 为验证集，70% 为训练集
        num_rows = ret_emb.shape[0]
        val_size = min(1000, int(num_rows * 0.3))
        train_size = num_rows - val_size
        # 打乱数据的索引
        indices = np.arange(num_rows)
        np.random.shuffle(indices)

        # 根据索引划分训练集和验证集
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_set = ret_emb[train_indices]
        val_set = ret_emb[val_indices]

        # 为训练集和验证集创建 batch
        train_batches = create_batches(train_set, batch_size)
        val_batches = create_batches(val_set, batch_size)

        # 训练集和验证集的平均值数据集
        train_avg_set = batch_average(train_batches)
        val_avg_set = batch_average(val_batches)

        # 输出结果
        print("训练集平均值数据集形状:", train_avg_set.shape)
        print("验证集平均值数据集形状:", val_avg_set.shape)

        # Step 2: 创建 Dataset 实例
        train_dataset = Attack_Batch_Dataset(train_avg_set)

        # Step 3: 构建 DataLoader
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

        # print(best_sim)

        prep_end_time = time.time()
        search_start_time = time.time()

        for it_ in range(args.num_iter):  #这个代码是对单个聚类簇进行攻击，所以直接设置迭代次数 num_iter 5000
            # print(f"Iteration: {it_}")

            # print(f'Accumulating Gradient {args.num_grad_iter}')
            c_model.zero_grad()

            pbar = range(args.num_grad_iter)
            # train_iter_centrid_embedding = iter(train_loader)
            train_iter_centrid_embedding = itertools.cycle(train_loader) # 因为聚类数多的话，有的时候就只有一个batch，所以需要循环
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

            # print('best_sim', best_sim)

        search_end_time = time.time()
        print(search_end_time-search_start_time,'seconds for searching')
        print(prep_end_time-prep_start_time,'seconds for preparing')
        attack_result_dic[k_s] = best_adv_passage_ids
        attack_result_p_ids.append(best_adv_passage_ids)



    # 开始进行evaluation

    # 加载 embedding_index.py 文件生成的 beir数据集的retrieval结果  这里全是 eval_model_code
    beir_result_file = f'{"results/beir_result"}/{args.attack_dataset}/{args.attack_model_code}/beir.json'
    with open(beir_result_file, 'r') as f:
        results = json.load(f)

    assert len(eval_qrels) == len(results)
    print('Total samples:', len(results))

    def evaluate_adv(k, qrels, results):
        '''
        evaluate adversarial results
        :param k:  这里的k = args.k
        :param qrels:
        :param results:
        :return:
        '''
        # 第一步，根据参数，加载所有的攻击结果
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
            # query_input = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")
            query_input = tokenizer(query_text,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt",
                                    max_length=args.max_seq_length) # evaluate_attack 中是否进行截断未知

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
    # 这里因为是indomain 的corpus attack, 所以eval_dataset 就是attack_dataset
    print('results: ', final_res)


if __name__ == "__main__":
    main()


