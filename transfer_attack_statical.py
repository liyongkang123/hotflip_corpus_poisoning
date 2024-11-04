'''
这个代码用来统计，分析 transfer attack 的结果
'''
'''
把evaluate_attack 的指标进行计算平均值 ，方差和标准差
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
from utils.load_model import load_models
from utils.load_data import load_data, load_data_yk, load_data_ours_batch
from utils.evaluate import evaluate_acc,evaluate_acc_yk ,evaluate_sim_ours
from utils.data_loader import GenericDataLoader
from utils.logging import LoggingHandler
import utils.utils as utils
from utils.utils import model_code_to_qmodel_name, model_code_to_cmodel_name
import utils.load_data as ld
import numpy as np

def main():
    args=  config.parse()
    print(args)

    # datasets_list =['nq','msmarco']
    datasets_list_dic= [('nq-train','nq'),('msmarco','msmarco')]
    source_model_code_list = [ "contriever", "contriever-msmarco", "dpr-single" ,"dpr-multi" ,"ance" ,"tas-b" ,"dragon" ]
    target_model_code_list = ["contriever", "contriever-msmarco", "dpr-single", "dpr-multi", "ance", "tas-b", "dragon"]
    seed_list = [1999, 5, 27, 2016, 2024]
    k_list = [50]

    for datasets_list in datasets_list_dic:
        for attack_model_code in source_model_code_list:
            for target_model_code in target_model_code_list:
                for k in k_list:
                    top_20_list = []
                    for seed in seed_list:

                        # 创建子目录，用于保存攻击结果的测试结果
                        sub_dir = 'results/attack_results/%s/%s-%s' % (args.method, datasets_list[0], attack_model_code)
                        # 这里是 attack_model_code， 记录攻击数据集，然后下面转到评估数据集


                        # 这里的 filename 是用来保存测试生成的 adversaival documents 效果的文件
                        filename = '%s/%s-%s-k%d-seed%d-num_cand%d-num_iter%d-tokens%d.json' % (
                        sub_dir, datasets_list[1], target_model_code, k, seed, args.num_cand, args.num_iter,
                        args.num_adv_passage_tokens)
                        # if os.path.isfile(filename):
                        #     return

                        # 读取 JSON 文件
                        with open(filename, 'r', encoding='utf-8') as file:
                            data = json.load(file)

                        # 打印读取的数据
                        # print(data)
                        top_20_list.append(data['Recall@20']*100)
                    top_20_np =np.array(top_20_list)
                    mean = np.mean(top_20_np)
                    std = np.std(top_20_np)
                    var = np.var(top_20_np)
                    print("datasets_list: ",datasets_list," source_model_code: ",attack_model_code," target_model_code: ",target_model_code," k: ",k," seed_list: ",seed_list )
                    print("top_20_np: ", top_20_np)
                    print("mean: ",mean)
                    print("std: ",std)
                    print("var: ",var)

if __name__ == '__main__':
    main()