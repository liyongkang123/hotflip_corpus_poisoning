#!/bin/bash
#SBATCH --job-name=generate_hotflip_multi_batch_all
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc425
#SBATCH --time=01-20:00:00 #默认最长
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out

module load anaconda/2024.02

source activate ir

# conda activate ir

# conda env list

# 1. 直接指定你解压环境里的 python
PYTHON=/gpfs/projects/ehpc425/anaconda/envs/ir/bin/python
echo "=== python path ==="
which $PYTHON || echo "PYTHON variable set to $PYTHON"
$PYTHON -V

cd /gpfs/projects/ehpc425/hotflip_corpus_poisoning


attack_dataset_list=( "nq-train" "hotpotqa"  "msmarco" ) #  "hotpotqa"  "msmarco"      "nq-train"  "scifact"  hotpotqa "fiqa" 
#attack_model_list=(  "contriever" "contriever-msmarco" "dpr-single" "dpr-multi" "ance" )
attack_model_list=( "qwen3_4B"  ) #  bge_m3 "qwen3" "gte" "linq"  reasonir  diver  bge_reasoner  '  # "contriever-msmarco" "bge_m3" "qwen3" "gte" "linq" "diver" "bge_reasoner"  已经提交   qwen3_0.6B  qwen3_4B
k_list=(  50) #10 50
seed_list=(1999 5 27 2016 2026) #1999 5 27 2016 2026 

for sub_data  in "${attack_dataset_list[@]}"; do
    for sub_model in "${attack_model_list[@]}"; do
      for sub_k in "${k_list[@]}"; do
          for sub_seed in "${seed_list[@]}"; do
        sbatch /gpfs/projects/ehpc425/hotflip_corpus_poisoning/scripts_mn5/generate_hotflip_multi_batch_all_sub.sh "${sub_data}" "${sub_model}" "${sub_k}" "${sub_seed}"
done
done
done
done
 