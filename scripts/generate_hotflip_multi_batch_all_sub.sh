#!/bin/sh
#SBATCH --job-name=generate_hotflip_sub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=00-20:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
eval "$(/gpfs/home1/yli4/anaconda3/bin/conda shell.bash hook)"
conda activate ir

nvidia-smi

cd /gpfs/work4/0/prjs0928/hotflip_corpus_poisoning

conda env list


sub_data=$1
sub_model=$2
sub_k=$3
sub_seed=$4

echo "Executing: python hotflip_attack_yk_batch_ms_all_llm.py --attack_dataset ${sub_data} --attack_model_code ${sub_model} --split train --max_seq_length 128 --max_query_length 32 --num_cand 100 --k ${sub_k} --num_iter 5000 --kmeans_split  --per_gpu_eval_batch_size 256 --init_gold True"

python hotflip_attack_yk_batch_ms_all_llm.py \
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