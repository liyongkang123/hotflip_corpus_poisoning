#!/bin/sh
#SBATCH --job-name=embed_index
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=00-30:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
eval "$(/gpfs/home1/yli4/anaconda3/bin/conda shell.bash hook)"
conda activate ir

nvidia-smi

cd /gpfs/work4/0/prjs0928/hotflip_corpus_poisoning


# the dataset and model list
# after the evaluation, we will have the retrieval results for each model and dataset
#eval_dataset_name_list=( "nq" "msmarco"  "hotpotqa" "fiqa" "trec-covid" "nfcorpus" "arguana" "quora" "scidocs" "fever" "scifact" )
eval_dataset_name_list=( "nq" ) # "nq" "msmarco"  "fiqa"  "scifact"
# eval_model_code_list=(  "contriever" "contriever-msmarco" "dpr-single" "dpr-multi" "ance" "tas-b" "dragon" )
eval_model_code_list=( "contriever-msmarco"  )

for sub_data_name  in "${eval_dataset_name_list[@]}"; do
    for sub_model in "${eval_model_code_list[@]}"; do
    python embedding_index.py \
        --eval_dataset ${sub_data_name} \
        --eval_model_code ${sub_model} \
        --split test \
        --per_gpu_eval_batch_size 512
done
done