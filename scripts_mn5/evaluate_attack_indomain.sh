#!/bin/bash
#SBATCH --job-name=evaluate_attack_indomain
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc425
#SBATCH --time=01-20:00:00 #默认最长
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out

module load anaconda/2024.02

source activate ir

attack_model_list=(   "contriever-msmarco" "bge_m3"   )
k_list=( 10)  #10 # be careful with k=50, please run only one seed 1999 for hotflip_raw
#k_list=(50)
seed_list=(1999 5 27 2016 2026) 
# seed_list=(  2024  ) 

for sub_model in "${attack_model_list[@]}"; do
  for sub_k in "${k_list[@]}"; do
  for sub_seed in "${seed_list[@]}"; do
          # sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "nq-train" "nq" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip_raw"
          # sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "nq-train" "nq" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"

          # sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "msmarco" "msmarco" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip_raw"
          # sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "msmarco" "msmarco" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"


          sbatch /gpfs/projects/ehpc425/hotflip_corpus_poisoning/scripts_mn5/evaluate_attack_indomain_sub.sh "nq-train" "nq" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"
          # sbatch /gpfs/projects/ehpc425/hotflip_corpus_poisoning/scripts_mn5/evaluate_attack_indomain_sub.sh "msmarco" "msmarco" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"
          # sbatch /gpfs/work4/0/prjs0928/hotflip_corpus_poisoning/scripts/sub_evaluate_attack_indomain.sh "hotpotqa" "hotpotqa" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"
done
done
done
