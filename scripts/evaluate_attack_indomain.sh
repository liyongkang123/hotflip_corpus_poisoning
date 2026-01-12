#!/bin/sh
#SBATCH --job-name=
# Set-up the environment.

conda env list

conda activate ir

nvidia-smi


attack_model_list=(   "contriever" "contriever-msmarco" "dpr-single" "dpr-multi" "ance" "tas-b" "dragon"  )
k_list=(1 10 ) # be careful with k=50, please run only one seed 1999 for hotflip_raw
#k_list=(50)
seed_list=(1999 5 27 2016 2024) 

for sub_model in "${attack_model_list[@]}"; do
  for sub_k in "${k_list[@]}"; do
  for sub_seed in "${seed_list[@]}"; do
          sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "nq-train" "nq" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip_raw"
          sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "nq-train" "nq" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"

          sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "msmarco" "msmarco" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip_raw"
          sbatch /ivi/ilps/personal/yli8/attack_baseline/scripts/sub_evaluate_attack_indomain.sh "msmarco" "msmarco" "${sub_model}" "${sub_model}" "${sub_k}" "${sub_seed}" "hotflip"
done
done
done
