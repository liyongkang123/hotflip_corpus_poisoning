from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv
from itertools import islice
from torch.utils.data import Dataset, DataLoader
import torch

logger = logging.getLogger(__name__)


class GenericDataLoader: # special for BEAR datasets

    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl",
                 query_file: str = "queries.jsonl",
                 qrels_folder: str = "qrels", qrels_file: str = ""):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels() # load qrels
            self.queries = {qid: self.queries[qid] for qid in self.qrels} # Only keep the query in qrels
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:

        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    # def _load_corpus(self): # This is the original code before optimization
    #
    #     num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
    #     with open(self.corpus_file, encoding='utf8') as fIn:
    #         for line in tqdm(fIn, total=num_lines):
    #             line = json.loads(line)
    #             self.corpus[line.get("_id")] = {
    #                 "text": line.get("text"),
    #                 "title": line.get("title"),
    #             }

    def _load_corpus(self): # This is the optimized code I wrote myself
        # Count total lines
        total_lines = sum(1 for i in open(self.corpus_file, 'rb'))

        with open(self.corpus_file, encoding='utf8') as fIn:
            for line_num, line in enumerate(tqdm(fIn, total=total_lines)):
                line = json.loads(line)
                # # If you are sure that the keys "_id", "text", and "title" must exist, you can avoid the get() method and directly access the dictionary fields to increase speed.
                self.corpus[line["_id"]] = {
                    "text": line["text"],
                    "title": line["title"],
                }


    def _load_queries(self):

        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self):

        open_file = open(self.qrels_file, encoding="utf-8")  # Open the file manually

        reader = csv.reader(open_file,delimiter="\t", quoting=csv.QUOTE_MINIMAL)

        next(reader)  # Skip header

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score

        open_file.close()

class Attack_Batch_Dataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float().to('cuda')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class CustomDataset_encode_corpus(Dataset):
    # encode corpus
    def __init__(self, dataset,p_max_length,tokenizer):
        """
        Args:
            data (list): List of data (could be paths to images, text data, etc.).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.max_length = p_max_length
        self.encoder_tokenizer = tokenizer

        self.docid_all = dataset.keys()
        # Doc_id is not needed at present when encoding, so just put all of them in
        self.data = list(dataset.values())

        assert len(self.docid_all) == len(self.data)

    def __len__(self):
        return len(self.docid_all)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

    def collate_fn(self, batch):

        text_list =[]
        for x in batch:
            if x['title']!="":
                text_list.append(x['title']+'.' +x['text'])
            else:
                text_list.append(x['text'])

        encoded_inputs = self.encoder_tokenizer(text_list,
                                                padding='max_length',
                                                truncation=True,
                                                max_length=self.max_length,
                                                return_tensors='pt')
        # return a dic
        return {
            'encoded_inputs': encoded_inputs,
        }