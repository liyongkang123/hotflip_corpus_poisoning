#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc425
#SBATCH --time=02:00:00 #默认最长
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out

module load anaconda/2024.02

source activate ir

# conda activate ir

# conda env list

# 1. 直接指定你解压环境里的 python
PYTHON=/gpfs/projects/ehpc425/anaconda/envs/ir/bin/python
echo "=== python path ==="
which $PYTHON || echo "PYTHON variable set to $PYTHON"
$PYTHON -V

cd /gpfs/projects/ehpc425/hotflip_corpus_poisoning

# 2. 用这个 python 跑你的脚本
$PYTHON  hotflip_attack_batch_all_llm.py     \
     --attack_dataset  nq-train \
     --attack_model_code  contriever-msmarco   \
     --split train \
     --max_seq_length 512 \
     --max_query_length 32 \
     --num_cand 100 \
     --k 10 \
     --num_iter 5000 \
     --per_gpu_eval_batch_size 64 \
     --init_gold \
     --seed 1999

