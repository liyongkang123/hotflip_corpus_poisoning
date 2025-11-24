#!/bin/sh
#SBATCH --job-name=generate_hotflip_multi_batch_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=00-10:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.
# Activate conda
eval "$(/gpfs/home1/yli4/anaconda3/bin/conda shell.bash hook)"
conda activate ir

nvidia-smi

cd /gpfs/work4/0/prjs0928/hotflip_corpus_poisoning

conda env list

conda activate ir

nvidia-smi


attack_dataset_list=( "nq-train"  "hotpotqa"  ) #"nq-train"  "scifact"  hotpotqa "fiqa" 
#attack_model_list=(  "contriever" "contriever-msmarco" "dpr-single" "dpr-multi" "ance" )
attack_model_list=( "contriever-msmarco" )
k_list=(1 10) #10
seed_list=( 1999 5 27 2016 2024  ) #1999 5 27 2016 2024

for sub_data  in "${attack_dataset_list[@]}"; do
    for sub_model in "${attack_model_list[@]}"; do
      for sub_k in "${k_list[@]}"; do
          for sub_seed in "${seed_list[@]}"; do
        sbatch /gpfs/work4/0/prjs0928/hotflip_corpus_poisoning/scripts/generate_hotflip_multi_batch_all_sub.sh "${sub_data}" "${sub_model}" "${sub_k}" "${sub_seed}"
done
done
done
done
 