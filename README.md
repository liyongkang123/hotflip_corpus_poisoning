# hotflip_corpus_poisoning
This is the code repository for our ECIR reproducibility paper submission.


## Files structure
- `data/` contains the data files used in the experiments.
- `results/` contains the results of the experiments.
- `scripts/` contains the scripts to run the experiments.
- `utils/` contains the utility functions used in the experiments.


## Requirements
- Python ,PyTorch , numpy, pandas, beir,transformers, sentence_transformers, sklearn, wandb
- If you do not want to use wandb, you can comment out all code with *wandb* in the code.
- You need to install the *beir* library(https://github.com/beir-cellar/beir)

## Details
Things need for the experiments in RQ1:
- 1, `embedding_index.py` is used to index the embeddings of the corpus, and save the retrieval results by BEIR.
- 2.1, `hotflip_raw.py` is used to generate the adversarial examples by hotflip (Zhong et al., 2023).
- 2.2, `hotflip_attack_ours.py` is used to generate the adversarial examples by hotflip (Zhong et al., 2023) with our pipeline optimizing strategy (Mean embedding).
- 3, `evaluate_attack.py` is used to evaluate the attack performance of the adversarial examples. The retrieval results are saved in `results/`.
- 4, `attack_results_statistics.py` is used to calculate the statistics of the attack results. 

Things need for the experiments in RQ2:
- 1, repeat the steps in RQ1 to generate the adversarial examples with all 7 retrievers.
- 2, `evaluate_attack.py` is used to evaluate the attack performance of the adversarial examples. The retrieval results are saved in `results/`.
- 3, `transfer_attack_statical.py` is used to calculate the statistics of the attack results of black-box attacks.

Things need for the experiments in RQ3:
- 1, `hotflip_ours_attack_corpus.py` is used for the corpus poisoning attack with our pipeline optimizing strategy (Mean embedding).

# Steps to reproduce the results
## RQ1
