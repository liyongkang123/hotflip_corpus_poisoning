#!/bin/sh
#SBATCH --job-name=
# Set-up the environment.

conda env list

conda activate ir

nvidia-smi
attack_dataset_list=( "arguana" "fiqa"  )
attack_model_list=(  "contriever"  "contriever-msmarco"  "dpr-single"   "dpr-multi" "ance" "tas-b"  "dragon" )
attack_percent=(0.0001 0.0005 0.001 0.005 )
seed_list=(1999 5 27 2016 2024)

for sub_data  in "${attack_dataset_list[@]}"; do
    for sub_model in "${attack_model_list[@]}"; do
      for sub_attack_percent in "${attack_percent[@]}"; do
          for sub_seed in "${seed_list[@]}"; do
        sbatch scripts/attack_corpus_ours_sub.sh "${sub_data}" "${sub_model}"  "${sub_seed}" "${sub_attack_percent}"
done
done
done
done