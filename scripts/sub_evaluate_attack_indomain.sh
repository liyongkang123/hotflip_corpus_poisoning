#!/bin/sh
#SBATCH --job-name=
# Set-up the environment.

conda env list

conda activate ir

nvidia-smi

# Got the parameters
sub_attack_data=$1
sub_eval_data=$2
sub_attack_model=$3
sub_eval_model=$4
sub_k=$5
sub_seed=$6
sub_method=$7

python evaluate_attack.py \
     --attack_dataset   ${sub_attack_data} \
     --attack_model_code  ${sub_attack_model} \
     --split test \
     --max_seq_length 128 \
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