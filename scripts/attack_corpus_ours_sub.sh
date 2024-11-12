#!/bin/sh
#SBATCH --job-name=
# Set-up the environment.

conda env list

conda activate ir

nvidia-smi


sub_data=$1
sub_model=$2
sub_seed=$3
sub_attack_rate=$4

echo "Executing: python hptflip_ours_attack_corpus.py --attack_dataset ${sub_data} --attack_model_code ${sub_model} --split train --max_seq_length 128 --max_query_length 32 --num_cand 100  --num_iter 5000 --kmeans_split  --per_gpu_eval_batch_size 256 --init_gold True"

python hptflip_ours_attack_corpus.py \
     --attack_dataset  ${sub_data} \
     --attack_model_code  ${sub_model} \
     --split test \
     --max_seq_length 128 \
     --max_query_length 32 \
     --num_cand 100 \
     --num_iter 5000 \
     --per_gpu_eval_batch_size 64 \
      --init_gold \
      --seed ${sub_seed} \
      --method hotflip \
      --attack_rate ${sub_attack_rate} \
      --result_output results_corpus_attack