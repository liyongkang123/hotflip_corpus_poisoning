#!/bin/sh
#SBATCH --job-name=
# Set-up the environment.

conda env list

conda activate ir

nvidia-smi


attack_dataset_list=( "nq-train" "msmarco" )
#attack_model_list=(  "contriever" "contriever-msmarco" "dpr-single" "dpr-multi" "ance" )
attack_model_list=(  "contriever" "contriever-msmarco" "dpr-single" "dpr-multi" "ance" "tas-b" "dragon")
k_list=(1 10 ) # for k=50, please run only one seed 1999
#k_list=(50)
seed_list=(1999 5 27 2016 2024)

for sub_data  in "${attack_dataset_list[@]}"; do
    for sub_model in "${attack_model_list[@]}"; do
      for sub_k in "${k_list[@]}"; do
        for sub_s in $(seq 0 $((sub_k-1))); do
          for sub_seed in "${seed_list[@]}"; do
        sbatch scripts/sub_generate_hotflip_raw.sh "${sub_data}" "${sub_model}" "${sub_k}" "${sub_s}" "${sub_seed}"
done
done
done
done
done