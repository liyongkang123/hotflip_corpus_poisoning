import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import logging
import os
import requests
import zipfile
from torch.utils.data import DataLoader
from data_loader import GenericDataLoader #beir.datasets
from data_loader import Attack_Batch_Dataset
import random
from datasets import Dataset
from transformers import default_data_collator
from tqdm import tqdm
import numpy as np
import time
from collections import Counter
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)

def tokenization(examples, tokenizer, max_seq_length, pad_to_max_length):
    q_feat = tokenizer(examples["sent0"], max_length=max_seq_length, truncation=True, padding="max_length" if pad_to_max_length else False)
    c_feat = tokenizer(examples["sent1"], max_length=max_seq_length, truncation=True, padding="max_length" if pad_to_max_length else False)

    ret = {}
    for key in q_feat:
        ret[key] = [(q_feat[key][i], c_feat[key][i]) for i in range(len(examples["sent0"]))]
    return ret

def create_batches(data, batch_size):
    """Split the data into batch_size and calculate the average of each batch."""
    if len(data) <= batch_size:
        return [data]

    num_batches = len(data) // batch_size
    batches = [data[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    if len(data) % batch_size != 0:
        batches.append(data[num_batches * batch_size:])
    return batches

def batch_average(batches):
    """Calculate the average value for each batch."""
    avg_data = np.array([np.mean(batch, axis=0) for batch in batches])
    return avg_data

def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get('Content-Length', 0))
    with open(save_path, 'wb') as fd, tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)

def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, "r")
    zip_.extractall(path=out_dir)
    zip_.close()

def download_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        logger.info("Downloading {} ...".format(dataset))
        download_url(url, zip_file, chunk_size)

    if not os.path.isdir(zip_file.replace(".zip", "")):
        logger.info("Unzipping {} ...".format(dataset))
        unzip(zip_file, out_dir)

    return os.path.join(out_dir, dataset.replace(".zip", ""))


def load_data(args, tokenizer, q_model, get_emb): # Here is the original version from Zhong et al.
# def load_data(args, tokenizer, q_model, get_emb, c_model):  # add c_model for valid_emb_dic
    # Load datasets
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.attack_dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.attack_dataset)
    if not os.path.exists(data_path):
        data_path = download_and_unzip(url, out_dir)
    print(data_path)
    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        args.split = 'train'
    corpus, queries, qrels = data.load(split=args.split)

    l = list(qrels.items())
    random.shuffle(l)
    qrels = dict(l)

    data_dict = {"sent0": [], "sent1": []}
    for q in qrels:
        q_ctx = queries[q]
        for c in qrels[q]:
            c_ctx = corpus[c].get("title") + ' ' + corpus[c].get("text")
            data_dict["sent0"].append(q_ctx)
            data_dict["sent1"].append(c_ctx)

    train_queries =  data_dict["sent0"]

    # do kmeans
    if args.do_kmeans:
        data_dict = kmeans_split(data_dict, q_model, get_emb, tokenizer, k=args.k, split=args.kmeans_split)

    datasets = {"train": Dataset.from_dict(data_dict)}

    def tokenization(examples):
        q_feat = tokenizer(examples["sent0"], max_length=args.max_seq_length, truncation=True,
                           padding="max_length" if args.pad_to_max_length else False)
        c_feat = tokenizer(examples["sent1"], max_length=args.max_seq_length, truncation=True,
                           padding="max_length" if args.pad_to_max_length else False)
        ret = {}
        for key in q_feat:
            ret[key] = [(q_feat[key][i], c_feat[key][i]) for i in range(len(examples["sent0"]))]
        return ret

    # use 30% examples as dev set during generation
    print('Train data size = %d' % (len(datasets["train"])))
    num_valid = min(1000, int(len(datasets["train"]) * 0.3))
    datasets["subset_valid"] = Dataset.from_dict(datasets["train"][:num_valid])
    datasets["subset_train"] = Dataset.from_dict(datasets["train"][num_valid:])

    train_dataset = datasets["subset_train"].map(tokenization, batched=True,
                                                 remove_columns=datasets["train"].column_names)
    dataset = datasets["subset_valid"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names)
    print('Finished loading datasets')

    data_collator = default_data_collator
    dataloader = DataLoader(train_dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=True,
                            collate_fn=lambda x: x)
    valid_dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=lambda x: x) # Here, change the batch according to the GPU memory you have

    '''The valid_dataloader here can be directly stored as an embedding, but here is the optimized approach'''
    # valid_q_emb = []
    # valid_gold_emb = []
    # for idx, (data) in tqdm(enumerate(valid_dataloader)): # The valid_dataloader is passed in here and has been tokenized.
    #     data = data_collator(data)  # [bsz, 2, max_len] , 2 is the query and its related document
    #
    #     # Get query embeddings
    #     q_sent = {k: data[k][:, 0, :].to('cuda') for k in data.keys()} # Here q_sent is the text of all queries in the validation set
    #     q_emb = get_emb(q_model, q_sent)  # [b x d]
    #     valid_q_emb.append(q_emb.detach().cpu().numpy())
    #
    #     gold_pass = {k: data[k][:, 1, :].to('cuda') for k in data.keys()} # Here, gold_pass is all the documents in the validation set, and these documents are the relevant text of the query.
    #     gold_emb = get_emb(c_model, gold_pass)  # [b x d]
    #     valid_gold_emb.append(gold_emb.detach().cpu().numpy())
    # valid_emb_dic = {"valid_q_emb":np.concatenate(valid_q_emb, axis=0),"valid_gold_emb":np.concatenate(valid_gold_emb, axis=0)}

    return data_collator, dataloader, valid_dataloader, num_valid
    # return data_collator, dataloader, valid_dataloader,num_valid,valid_emb_dic


def kmeans_split(data_dict, model, get_emb, tokenizer, k, split):
    """Get all query embeddings and perform kmeans"""

    # get query embs
    q_embs = []
    for q in tqdm(data_dict["sent0"]):
        query_input = tokenizer(q, padding=True, truncation=True, return_tensors="pt")
        query_input = {key: value.cuda() for key, value in query_input.items()}
        with torch.no_grad():
            query_emb = get_emb(model, query_input)
        q_embs.append(query_emb[0].cpu().numpy())
    q_embs = np.array(q_embs)
    print("q_embs", q_embs.shape)

    # Using batch operations may speed up the process 30 minutes -> 3 minutes
    # Assume that batch_size is defined as a suitable value, such as 16
    # batch_size = 128
    # all_query_inputs = tokenizer(data_dict["sent0"], padding=True, truncation=True, return_tensors="pt")
    # all_query_inputs = {key: value.cuda() for key, value in all_query_inputs.items()}
    # q_embs = []
    # for i in tqdm(range(0, len(data_dict["sent0"]), batch_size)):
    #     batch_query_input = {key: value[i:i + batch_size] for key, value in all_query_inputs.items()}
    #     with torch.no_grad():
    #         batch_query_embs = get_emb(model, batch_query_input)
    #     q_embs.append(batch_query_embs.cpu().numpy())
    # q_embs = np.concatenate(q_embs, axis=0)
    # print("q_embs", q_embs.shape)

    # Start the timer
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(q_embs)
    print(Counter(kmeans.labels_))
    # End the timer
    end_time = time.time()
    # Calculate the runtime
    runtime = end_time - start_time
    print('Clustering took: ', runtime, ' seconds')
    ret_dict = {"sent0": [], "sent1": []}
    for i in range(len(data_dict["sent0"])):
        if kmeans.labels_[i] == split:
            ret_dict["sent0"].append(data_dict["sent0"][i])
            ret_dict["sent1"].append(data_dict["sent1"][i])
    print("K = %d, split = %d, tot num = %d" % (k, split, len(ret_dict["sent0"])))
    return ret_dict


def kmeans_split_ours(data_dict, model, get_emb, tokenizer, k, split):
    """Get all query embeddings and perform kmeans"""

    # get query embs
    batch_size = 128
    all_query_inputs = tokenizer(data_dict["sent0"], padding=True, truncation=True, return_tensors="pt")
    all_query_inputs = {key: value.cuda() for key, value in all_query_inputs.items()}

    q_embs = []
    for i in tqdm(range(0, len(data_dict["sent0"]), batch_size)):
        batch_query_input = {key: value[i:i + batch_size] for key, value in all_query_inputs.items()}
        with torch.no_grad():
            batch_query_embs = get_emb(model, batch_query_input)
        q_embs.append(batch_query_embs.cpu().numpy())

    q_embs = np.concatenate(q_embs, axis=0)

    print("q_embs", q_embs.shape)

    # Start the timer
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(q_embs)
    print(Counter(kmeans.labels_))
    # End the timer
    end_time = time.time()
    # Calculate the runtime
    runtime = end_time - start_time

    ret_dict = {"sent0": [], "sent1": []}
    for i in range(len(data_dict["sent0"])):
        if kmeans.labels_[i] == split:
            ret_dict["sent0"].append(data_dict["sent0"][i])
            ret_dict["sent1"].append(data_dict["sent1"][i])

    del q_embs,all_query_inputs, data_dict
    print("K = %d, split = %d, tot num = %d" % (k, split, len(ret_dict["sent0"])))
    return ret_dict

def load_data_ours_batch(args, tokenizer, q_model, c_model, get_emb):
    # Load datasets
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.attack_dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.attack_dataset)
    if not os.path.exists(data_path):
        data_path = download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        args.split = 'train'
    corpus, queries, qrels = data.load(split=args.split)

    l = list(qrels.items())
    random.shuffle(l)
    qrels = dict(l)

    data_dict = {"sent0": [], "sent1": []}
    for q in qrels:
        q_ctx = queries[q]
        for c in qrels[q]: #  c is corpus passage id
            c_ctx = corpus[c].get("title") + ' ' + corpus[c].get("text")
            data_dict["sent0"].append(q_ctx)
            data_dict["sent1"].append(c_ctx)

    # In fact, from here on, I only need the center of train and vaild
    # The code here clusters first, then splits the clustered code into train, vaild, and then into multiple batches, and then calculates the center of each batch

    # do kmeans
    if args.do_kmeans:
        data_dict = kmeans_split_ours(data_dict, q_model, get_emb, tokenizer, k=args.k, split=args.kmeans_split)

    gold_passage_init = random.choice(data_dict["sent1"])
    if args.attack_query==True:
        batch_size = 64
        print('start tokenizing all query')
        all_inputs = tokenizer(data_dict["sent0"], padding="max_length", truncation=True, max_length = args.max_query_length,
                               return_tensors="pt")
    else:
        batch_size = 64
        print('start tokenizing all documents')
        all_inputs = tokenizer(data_dict["sent1"], padding="max_length", truncation=True, max_length=args.max_seq_length,
                           return_tensors="pt")
    all_inputs = {key: value.cuda() for key, value in all_inputs.items()}
    q_embs = []
    for i in tqdm(range(0, len(data_dict["sent0"]), batch_size)):
        batch_query_input = {key: value[i:i + batch_size] for key, value in all_inputs.items()}

        with torch.no_grad():
            batch_query_embs = get_emb(q_model, batch_query_input)

        q_embs.append(batch_query_embs.cpu().numpy())
    q_embs = np.concatenate(q_embs, axis=0)
    # Step 1: Divide the rows into 30% validation set and 70% training set
    num_rows = q_embs.shape[0]
    val_size = min(1000, int(num_rows * 0.3))
    train_size = num_rows - val_size
    indices = np.arange(num_rows)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_set = q_embs[train_indices]
    val_set = q_embs[val_indices]

    del q_embs, data_dict

    train_batches = create_batches(train_set, batch_size)
    val_batches = create_batches(val_set, batch_size)

    train_avg_set = batch_average(train_batches)
    val_avg_set = batch_average(val_batches)

    # Output results
    print("Training set average data set shape:", train_avg_set.shape)
    print("Validation set average data set shape:", val_avg_set.shape)
    train_dataset = Attack_Batch_Dataset(train_avg_set)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataset = Attack_Batch_Dataset(val_avg_set)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    data_collator = default_data_collator
    return data_collator, train_loader, valid_loader, val_size ,gold_passage_init


def load_data_ours_batch_all(args, tokenizer, q_model, c_model, get_emb):
    # This can load many kmean-splits at one time
    # Load datasets
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.attack_dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.attack_dataset)
    if not os.path.exists(data_path):
        data_path = download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        args.split = 'train'
    corpus, queries, qrels = data.load(split=args.split)
    l = list(qrels.items())
    random.shuffle(l)
    qrels = dict(l)

    data_dict = {"sent0": [], "sent1": []}
    for q in qrels:
        q_ctx = queries[q]
        for c in qrels[q]:
            c_ctx = corpus[c].get("title") + ' ' + corpus[c].get("text")
            data_dict["sent0"].append(q_ctx)
            data_dict["sent1"].append(c_ctx)

    # do kmeans
    if args.do_kmeans:
        data_dict_dic = kmeans_split_ours_all(data_dict, q_model, get_emb, tokenizer, k=args.k)

    data_collator_dic, train_loader_dic, valid_loader_dic, val_size_dic, gold_passage_init_dic={},{},{},{},{}
    for k_s in range(args.k):
        data_dict = data_dict_dic[k_s]
        gold_passage_init = random.choice(data_dict["sent1"])
        if args.attack_query==True:
            batch_size = 64
            print('Start tokenizing all query')
            all_inputs = tokenizer(data_dict["sent0"], padding="max_length", truncation=True, max_length = args.max_query_length,
                                   return_tensors="pt")
        else:
            batch_size = 64
            print('Start tokenizing all documents')
            all_inputs = tokenizer(data_dict["sent1"], padding="max_length", truncation=True, max_length=args.max_seq_length,
                               return_tensors="pt")
        all_inputs = {key: value.cuda() for key, value in all_inputs.items()}
        q_embs = []
        for i in tqdm(range(0, len(data_dict["sent0"]), batch_size)):
            batch_query_input = {key: value[i:i + batch_size] for key, value in all_inputs.items()}
            with torch.no_grad():
                batch_query_embs = get_emb(q_model, batch_query_input)
            q_embs.append(batch_query_embs.cpu().numpy())
        q_embs = np.concatenate(q_embs, axis=0)

        num_rows = q_embs.shape[0]
        val_size = min(1000, int(num_rows * 0.3))
        train_size = num_rows - val_size

        indices = np.arange(num_rows)
        np.random.shuffle(indices)


        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_set = q_embs[train_indices]
        val_set = q_embs[val_indices]

        del q_embs, data_dict

        train_batches = create_batches(train_set, batch_size)
        val_batches = create_batches(val_set, batch_size)

        train_avg_set = batch_average(train_batches)
        val_avg_set = batch_average(val_batches)

        # Output results
        print("Training set average data set shape:", train_avg_set.shape)
        print("Validation set average data set shape:", val_avg_set.shape)

        train_dataset = Attack_Batch_Dataset(train_avg_set)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        valid_dataset = Attack_Batch_Dataset(val_avg_set)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

        data_collator = default_data_collator

        data_collator_dic[k_s] = data_collator
        train_loader_dic[k_s] = train_loader
        valid_loader_dic[k_s] = valid_loader
        val_size_dic[k_s] = val_size
        gold_passage_init_dic[k_s] = gold_passage_init

    return data_collator_dic, train_loader_dic, valid_loader_dic, val_size_dic ,gold_passage_init_dic

def kmeans_split_ours_all(data_dict, model, get_emb, tokenizer, k):
    """Get all query embeddings and perform kmeans"""

    # get query embs
    # Using batch for calculation will speed up the process from 30 minutes to 3 minutes
    # Assume that batch_size is defined as a suitable value, such as 16
    batch_size = 128
    all_query_inputs = tokenizer(data_dict["sent0"], padding=True, truncation=True, return_tensors="pt")
    all_query_inputs = {key: value.cuda() for key, value in all_query_inputs.items()}

    q_embs = []
    for i in tqdm(range(0, len(data_dict["sent0"]), batch_size)):
        batch_query_input = {key: value[i:i + batch_size] for key, value in all_query_inputs.items()}
        with torch.no_grad():
            batch_query_embs = get_emb(model, batch_query_input)

        q_embs.append(batch_query_embs.cpu().numpy())


    q_embs = np.concatenate(q_embs, axis=0)

    print("q_embs", q_embs.shape)

    # Start the timer
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(q_embs)
    print(Counter(kmeans.labels_))
    # End the timer
    end_time = time.time()
    # Calculate the runtime
    runtime = end_time - start_time
    print('Clustering took: ', runtime, ' seconds')

    return_data_dic={}

    for k_s in range(k):
        split = k_s
        ret_dict = {"sent0": [], "sent1": []}
        for i in range(len(data_dict["sent0"])):
            if kmeans.labels_[i] == split:  # The split is set here to attack on multiple GPUs, and only attack one cluster on each gpu
                ret_dict["sent0"].append(data_dict["sent0"][i])
                ret_dict["sent1"].append(data_dict["sent1"][i])

        return_data_dic[split] = ret_dict

    del q_embs, all_query_inputs, data_dict
    # print("K = %d, split = %d, tot num = %d" % (k, split, len(ret_dict["sent0"])))
    return return_data_dic # The output is a dictionary, the key is the cluster, the value is also a dic:  the documents of the cluster