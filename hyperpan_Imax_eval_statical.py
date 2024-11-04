'''
使用nq 或者msmarco 的测试集query进行测试生成的结果
'''
import os
import json
import sys
import argparse
import torch
import copy
from tqdm import tqdm
import config
import wandb
import numpy as np
from utils.load_model import load_models
from utils.load_data import load_data, load_data_yk, load_data_ours_batch
from utils.evaluate import evaluate_acc,evaluate_acc_yk ,evaluate_sim_ours
from utils.data_loader import GenericDataLoader
from utils.logging import LoggingHandler
import utils.utils as utils
from utils.utils import model_code_to_qmodel_name, model_code_to_cmodel_name
import utils.load_data as ld


def evaluate_recall(results, qrels, k_values=[10, 20, 50, 100, 500, 1000] ):
    cnt = {k: 0 for k in k_values}
    for q in results:
        sims = list(results[q].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        gt = qrels[q]
        found = 0
        for i, (c, _) in enumerate(sims[:max(k_values)]):
            if c in gt:
                found = 1
            if (i + 1) in k_values:
                cnt[i + 1] += found
    #             print(i, c, found)
    recall = {}
    for k in k_values:
        recall[f"Recall@{k}"] = round(cnt[k] / len(results), 5)

    return recall


def main():
    args=  config.parse()
    print(args)
    wandb.init(
        # set the wandb project where this run will be logged
        project="hyper_generate_Imax",
        # track hyperparameters and run metadata
        config=vars(args),
    )
    datasets_list_dic= [('nq-train','nq')]
    model_list = ["contriever-msmarco"]
    # source_model_code_list = [ "contriever", "contriever-msmarco", "dpr-single" ,"dpr-multi" ,"ance" ,"tas-b" ,"dragon" ]
    # target_model_code_list = ["contriever", "contriever-msmarco", "dpr-single", "dpr-multi", "ance", "tas-b", "dragon"]
    seed_list = [1999, 5, 27, 2016, 2024]
    k_list = [1,10, 50]


    args.split ='test'
    args.result_output = "results/beir_result"
    for eval_datasets in datasets_list_dic:
        args.eval_dataset = eval_datasets[1]

        for model_name in model_list:
            args.eval_model_code = model_name

            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.eval_dataset)
            out_dir = os.path.join(os.getcwd(), "datasets")
            data_path = os.path.join(out_dir, args.eval_dataset)
            if not os.path.exists(data_path):
                data_path = ld.download_and_unzip(url, out_dir)
            print(data_path)

            data = GenericDataLoader(data_path)
            if '-train' in data_path:
                args.split = 'train'
            corpus, queries, qrels = data.load(split=args.split)

            # 加载 embedding_index.py 文件生成的 beir数据集的retrieval结果  这里全是 eval_model_code
            beir_result_file = f'{args.result_output}/{args.eval_dataset}/{args.eval_model_code}/beir.json'
            with open(beir_result_file, 'r') as f:
                results = json.load(f)

            assert len(qrels) == len(results)
            print('Total samples:', len(results))

            # Load models
            model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)

            model.eval()
            model.cuda()
            c_model.eval()
            c_model.cuda()

            for k in k_list:

                num_iter_range = list(range(1000, 20001, 1000))
                for num_iter in num_iter_range:
                    top_20_list = []
                    for seed in seed_list:

                        def evaluate_adv(k, qrels, results, num_iter,seed):

                            # 第一步，根据参数，加载所有的攻击结果
                            adv_ps = []
                            for s in range(k):
                                file_name = "results_hyper_iter/%s-generate/%s/%s/k%d-s%d-seed%d-num_cand%d-num_iter%d-tokens%d-gold_init%s.json" % (
                                    args.method, eval_datasets[0], model_name, k, s, seed, # 这里是attack_dataset
                                    args.num_cand, num_iter, args.num_adv_passage_tokens, True)
                                if not os.path.exists(file_name):
                                    print(f"!!!!! {file_name} does not exist!")
                                    continue
                                attack_results = []

                                with open(file_name, 'r') as f:
                                    for line in f:
                                        data = json.loads(line)
                                        attack_results.append(data)

                                    adv_ps.append(attack_results[-1])
                            print('# adversaria passages', len(adv_ps))

                            adv_results = copy.deepcopy(results)

                            adv_p_ids = [tokenizer.convert_tokens_to_ids(p["best_adv_text"]) for p in adv_ps]
                            adv_p_ids = torch.tensor(adv_p_ids).cuda()
                            adv_attention = torch.ones_like(adv_p_ids, device='cuda')
                            adv_token_type = torch.zeros_like(adv_p_ids, device='cuda')
                            adv_input = {'input_ids': adv_p_ids, 'attention_mask': adv_attention, 'token_type_ids': adv_token_type}
                            with torch.no_grad():
                                adv_embs = get_emb(c_model, adv_input)

                            adv_qrels = {q: {"adv%d" % (s): 1 for s in range(k)} for q in qrels}

                            for i, query_id in tqdm(enumerate(results)):
                                query_text = queries[query_id]
                                query_input = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")
                                query_input = {key: value.cuda() for key, value in query_input.items()}
                                with torch.no_grad():
                                    query_emb = get_emb(model, query_input)
                                    adv_sim = torch.mm(query_emb, adv_embs.T)

                                for s in range(len(adv_ps)):
                                    adv_results[query_id]["adv%d" % (s)] = adv_sim[0][s].cpu().item()

                            adv_eval = evaluate_recall(adv_results, adv_qrels)

                            return adv_eval

                        final_res = evaluate_adv(k, qrels, results,num_iter,seed)
                        print(final_res)
                        top_20_list.append(final_res['Recall@20'])
                    #计算同一个 num_iter 下的不同seed 的平均值和标准差
                    top_20_np = np.array(top_20_list)
                    mean = np.mean(top_20_np)
                    std = np.std(top_20_np)
                    var = np.var(top_20_np)
                    print("attack_datasets: ", eval_datasets[0], " eval_datasets: ", eval_datasets[1], " model_code: ", model_name, " k: ", k, " seed_list: ",
                          seed_list, " num_iter: ", num_iter)
                    print("top_20_np: ", top_20_np)
                    print("mean: ", mean)
                    print("std: ", std)
                    print("var: ", var)


if __name__ == "__main__":
    main()