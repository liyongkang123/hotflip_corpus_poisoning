#!/bin/sh
#SBATCH --job-name=
# Set-up the environment.

conda env list

conda activate ir

nvidia-smi

# got the parameters
sub_data=$1
sub_model=$2
sub_k=$3
sub_s=$4
sub_seed=$5

echo "Executing: python hotflip_raw.py --attack_dataset ${sub_data} --attack_model_code ${sub_model} --split train --max_seq_length 128 --max_query_length 32 --num_cand 100 --k ${sub_k} --num_iter 5000 --kmeans_split  --per_gpu_eval_batch_size 256 --init_gold True"

python hotflip_raw.py \
     --attack_dataset  ${sub_data} \
     --attack_model_code  ${sub_model} \
     --split train \
     --max_seq_length 128 \
     --max_query_length 32 \
     --num_cand 100 \
     --k ${sub_k} \
     --num_iter 5000 \
     --kmeans_split ${sub_s} \
     --per_gpu_eval_batch_size 64 \
      --init_gold \
      --seed ${sub_seed} \
      --method hotflip_raw