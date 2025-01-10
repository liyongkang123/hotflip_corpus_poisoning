# hotflip_corpus_poisoning
This is the code repository for our ECIR2025 reproducibility paper [《Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval
》](https://arxiv.org/abs/2501.04802).


## Files structure
- `datasets/` contains the datasets files used in the experiments.
- `results/` contains the results of the experiments.
- `scripts/` contains the scripts to run the experiments.
- `utils/` contains the utility functions used in the experiments.

## Datasets
The datasets used in the experiments are from the BEIR library. The datasets are stored in the `datasets/` folder. These datasets will download automatically when you run the code.

## Requirements
- Python ,PyTorch , numpy, pandas, beir,transformers, sentence_transformers, sklearn, wandb
- If you do not want to use wandb, you can comment out all code with *wandb* in the code.
- You need to install the *beir* library(https://github.com/beir-cellar/beir)

## Introduction of the code
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
- 1, Run `sbatch scripts/embedding_index.sh` to get the retrieval results of the all datsaets with all retrievers. The retrieval results are saved in `results/beir_result`.
- 2, Run `sbatch scripts/generate_hotflip_multi_raw.sh` to generate the adversarial examples by hotflip (Zhong et al., 2023). The adversarial examples are saved in `results/hotflip_raw-generate`.
- 3, Run `sbatch scripts/generate_hotflip_multi.sh` to generate the adversarial examples by hotflip (Zhong et al., 2023) with our pipeline optimizing strategy (Mean embedding). The adversarial examples are saved in `results/hotflip-generate`.
- 4, Run `sbatch scripts/evaluate_attack.sh` to evaluate the attack performance of the adversarial examples. The  results are saved in `results/attack_results`.
- 5, Run `python attack_results_statistics.py` to calculate the statistics of the attack results.

## RQ2
- 1, Since we have already generated the adversarial examples in RQ1, we do not need to repeat the steps in RQ1.
- 2, Run `sbatch scripts/transfer_attack.sh`. The retrieval results are saved in `results/attack_results/hotflip` and `results/attack_results/hotflip_raw`.
- 3, Run `python transfer_attack_statical.py --method hotflip_raw ` and `python transfer_attack_statical.py --method hotflip` to calculate the statistics of the attack results of black-box attacks. Remember to change the method to `hotflip_raw` and `hotflip` respectively. And change `seed_list = [1999]` only for `k_list=[10]` in the evaluation `hotflip_raw` method.

## RQ3
- 1, Run `sbatch scripts/attack_corpus_ous.sh` to generate the adversarial passages for the corpus poisoning attack. The results are saved in `results_corpus_attack/hotflip-generate`.
- 2, When you finish the code, they will output the evaluation results directly. Just record the results.

[//]: # (# Hyperparameter Study of $I_{max}$)

[//]: # ()
[//]: # (The maximum number of iterations $I_{max}$ is an important hyper-parameter affecting the attack result. Zhong et al.&#40;2023&#41;  use $I_{max}=5000$  as the default setting, while Su et al.&#40;2024&#41; use $I_{max}=3000$ as the default setting. However, the impact of $I_{max}$ on experimental results, aside from their effect on runtime, remains unclear. To better show the differences, we select **Contriever-ms** as the retriever, and attack the **NQ** dataset using its training queries. And we generate $|\mathcal{A}| \in \{1, 10, 50\}$ adversarial passages with different number of iterations $I_{max}$. We use five different random seeds and record the experimental results every 1000 iterations, from 1000 to a maximum of 20000. We report the mean attack success rate under different random seeds in the following Figure.)

[//]: # ()
[//]: # (![Image]&#40;results/hyper_parameter_Imax.png&#41;)

[//]: # ()
[//]: # (In this Figure , we can observe that a larger $I_{max}$ generally leads to better attack performance. Moreover, increasing $I_{max}$ leads to a much greater performance improvement when $|\mathcal{A}|=1$ compared to $|\mathcal{A}|=50$. )

[//]: # (However, increasing $I_{max}$ also leads to more time costs, even with our optimized code, each iteration still takes approximately 0.06 seconds.)

[//]: # ( Therefore, the specific choice of $I_{max}$ depends on a trade-off between efficiency and performance.)


# Citation
If you find this code useful, I would greatly appreciate it if you could cite our paper:
```
@inproceedings{li2025reproducinghotflip,
  title={Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval},
  author={Yongkang Li and Panagiotis Eustratiadis and Evangelos Kanoulas},
  booktitle={The 47th European Conference on Information Retrieval, {ECIR} 2025},
  year={2025},
  organization={Springer},
  url={https://arxiv.org/abs/2501.04802}, 
}
```