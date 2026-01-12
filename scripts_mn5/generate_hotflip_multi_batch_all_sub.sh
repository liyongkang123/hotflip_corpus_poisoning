#!/bin/bash
#SBATCH --job-name=generate_hotflip_sub
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc425
#SBATCH --time=01-20:00:00 #默认最长
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out


module load anaconda/2024.02

source activate ir

# 1. 直接指定你解压环境里的 python
PYTHON=/gpfs/projects/ehpc425/anaconda/envs/ir/bin/python
echo "=== python path ==="
which $PYTHON || echo "PYTHON variable set to $PYTHON"
$PYTHON -V


nvidia-smi

cd /gpfs/projects/ehpc425/hotflip_corpus_poisoning

conda env list


sub_data=$1
sub_model=$2
sub_k=$3
sub_seed=$4

echo "Executing: python hotflip_attack_batch_all_llm.py --attack_dataset ${sub_data} --attack_model_code ${sub_model} --split train --max_seq_length 128 --max_query_length 32 --num_cand 100 --k ${sub_k} --num_iter 5000 --kmeans_split  --per_gpu_eval_batch_size 256 --init_gold True"

# 2. 用这个 python 跑你的脚本
$PYTHON /gpfs/projects/ehpc425/hotflip_corpus_poisoning/hotflip_attack_batch_all_llm.py \
     --attack_dataset  ${sub_data} \
     --attack_model_code  ${sub_model}  \
     --split train \
     --max_seq_length 512 \
     --max_query_length 32 \
     --num_cand 100 \
     --k ${sub_k} \
     --num_iter 5000 \
     --per_gpu_eval_batch_size 64 \
     --init_gold \
     --seed ${sub_seed}