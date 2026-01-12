#!/bin/sh
#SBATCH --job-name=
# Set-up the environment.

conda env list

conda activate ir

nvidia-smi


# the dataset and model list
# after the evaluation, we will have the retrieval results for each model and dataset
#eval_dataset_name_list=( "nq" "msmarco"  "hotpotqa" "fiqa" "trec-covid" "nfcorpus" "arguana" "quora" "scidocs" "fever" "scifact" )
eval_dataset_name_list=( "nq" "msmarco"  "fiqa"  "arguana"  )
eval_model_code_list=(  "contriever" "contriever-msmarco" "dpr-single" "dpr-multi" "ance" "tas-b" "dragon" )

for sub_data_name  in "${eval_dataset_name_list[@]}"; do
    for sub_model in "${eval_model_code_list[@]}"; do
    python embedding_index.py \
        --eval_dataset ${sub_data_name} \
        --eval_model_code ${sub_model} \
        --split test \
        --per_gpu_eval_batch_size 512
done
done