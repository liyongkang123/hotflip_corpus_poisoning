'''
We first perform the original retrieval evaluation on BEIR and save the retrieval results
(i.e., for a given query, we save a list of top passages and similarity values to the query).
'''

import logging
from transformers import (
    set_seed,
default_data_collator,
)
import wandb
logger = logging.getLogger(__name__)

from utils.data_loader import GenericDataLoader
from utils.logging import LoggingHandler
import utils.utils as utils
from utils.utils import model_code_to_qmodel_name, model_code_to_cmodel_name
import utils.load_data as ld
import config
from utils.load_model import Contriever


from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import DPR
from utils.beir_utils import DenseEncoderModel

import logging
import os
import json
import torch
import sys
import transformers


def main():
    args=  config.parse()
    print(args)
    wandb.init(
        # set the wandb project where this run will be logged
        project="hotflip attack",
        config=vars(args),
    )

    set_seed(args.seed)

    result_output = f'{args.result_output}/{args.eval_dataset}/{args.eval_model_code}/beir.json'
    if os.path.isfile(result_output):
        exit()

    def compress(results):
        for y in results:
            k_old = len(results[y])
            break
        sub_results = {}
        for query_id in results:
            sims = list(results[query_id].items())
            sims.sort(key=lambda x: x[1], reverse=True)
            sub_results[query_id] = {}
            for c_id, s in sims[:2000]:
                sub_results[query_id][c_id] = s
        for y in sub_results:
            k_new = len(sub_results[y])
            break
        logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
        return sub_results

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    logging.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #### Download and load eval_dataset
    dataset = args.eval_dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        data_path = ld.download_and_unzip(url, out_dir)
    logging.info(data_path)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)

    logging.info("Loading model...")
    if 'contriever' in args.eval_model_code:
        encoder = Contriever.from_pretrained(model_code_to_cmodel_name[args.eval_model_code]).cuda()
        tokenizer = transformers.BertTokenizerFast.from_pretrained(model_code_to_cmodel_name[args.eval_model_code])
        model = DRES(DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer),
                     batch_size=args.per_gpu_eval_batch_size)
    elif 'dpr' in args.eval_model_code:
        model = DRES(DPR((model_code_to_qmodel_name[args.eval_model_code], model_code_to_cmodel_name[args.eval_model_code])),
                     batch_size=args.per_gpu_eval_batch_size, corpus_chunk_size=5000)
    elif any(model_code in args.eval_model_code for model_code in ['ance','tas', 'condenser']):
        model = DRES(models.SentenceBERT(model_code_to_cmodel_name[args.eval_model_code]),
                     batch_size=args.per_gpu_eval_batch_size)
    elif "dragon" in args.eval_model_code:
        model = DRES(models.SentenceBERT((model_code_to_qmodel_name[args.eval_model_code], model_code_to_cmodel_name[args.eval_model_code])),
                     batch_size=args.per_gpu_eval_batch_size)
    else:
        raise NotImplementedError

    logging.info(f"model: {model.model}")

    retriever = EvaluateRetrieval(model, score_function=args.score_function, k_values=[10, 50, 1000]) # "cos_sim"  or "dot" for dot-product
    results = retriever.retrieve(corpus, queries)

    logging.info("Printing results to %s"%(result_output))
    sub_results = compress(results)
    output_dir_name = os.path.dirname(result_output)
    if not os.path.exists(output_dir_name ):
        os.makedirs(output_dir_name)
    with open(result_output, 'w') as f:
        json.dump(sub_results, f)

if __name__ == '__main__':
    main()