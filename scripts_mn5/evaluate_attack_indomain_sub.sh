#!/bin/bash
#SBATCH --job-name=evaluate_attack_indomain_sub
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

nvidia-smi

# Got the parameters
sub_attack_data=$1
sub_eval_data=$2
sub_attack_model=$3
sub_eval_model=$4
sub_k=$5
sub_seed=$6
sub_method=$7

$PYTHON evaluate_attack.py \
     --attack_dataset   ${sub_attack_data} \
     --attack_model_code  ${sub_attack_model} \
     --split test \
     --max_seq_length 512 \
     --max_query_length 32 \
     --num_cand 100 \
     --k ${sub_k} \
     --num_iter 5000 \
     --kmeans_split 0 \
     --per_gpu_eval_batch_size 256 \
     --eval_dataset ${sub_eval_data} \
     --eval_model_code ${sub_eval_model} \
     --seed ${sub_seed} \
     --init_gold  \
     --method ${sub_method}