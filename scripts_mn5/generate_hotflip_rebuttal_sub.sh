#!/usr/bin/env bash
#SBATCH --job-name=hotflip_rebuttal
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
module load anaconda/2024.02
source activate ir
set -u
REPO=${REPO:-/gpfs/projects/ehpc695/hotflip_corpus_poisoning}
PYTHON=${PYTHON:-/gpfs/projects/ehpc695/anaconda/envs/ir/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON")")/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=${HF_HOME:-/gpfs/projects/ehpc695/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "$REPO"
model=$1 dataset=$2
read -r -a ks <<< "$3"
read -r -a seeds <<< "$4"
nvidia-smi
"$PYTHON" hotflip_attack_rebuttal.py --model "$model" --attack_dataset "$dataset" \
    --ks "${ks[@]}" --seeds "${seeds[@]}" \
    --cache_root "${CACHE_ROOT:-$REPO/cache_rebuttal}" \
    --output_root "${OUTPUT_ROOT:-$REPO/output_attack/attacked_text/document/hotflip_rebuttal}" \
    --query_embedding_root "${QUERY_EMBEDDING_ROOT:-$REPO/cache_rebuttal/query_embeddings}" \
    --cand_chunk "${CAND_CHUNK:-100}"
