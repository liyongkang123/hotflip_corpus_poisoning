#!/bin/sh
#SBATCH --job-name=evaluate_attack_indomain
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


attack_model_list=(   "contriever-msmarco"    )
k_list=(1  10)  #10 # be careful with k=50, please run only one seed 1999 for hotflip_raw
#k_list=(50)
seed_list=(1999 5 27 2016 2024) 
# seed_list=(  2024  ) 

for sub_model in "${attack_model_list[@]}"; do
  for sub_k in "${k_list[@]}"; do
  for sub_seed in "${seed_list[@]}"; do
          # sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "nq-train" "nq" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip_raw"
          # sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "nq-train" "nq" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"

          # sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "msmarco" "msmarco" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip_raw"
          # sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "msmarco" "msmarco" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"


          sbatch /gpfs/work4/0/prjs0928/hotflip_corpus_poisoning/scripts/sub_evaluate_attack_indomain.sh "nq-train" "nq" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"
          sbatch /gpfs/work4/0/prjs0928/hotflip_corpus_poisoning/scripts/sub_evaluate_attack_indomain.sh "fiqa" "fiqa" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"
          # sbatch /gpfs/work4/0/prjs0928/hotflip_corpus_poisoning/scripts/sub_evaluate_attack_indomain.sh "scifact" "scifact" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"
done
done
done
