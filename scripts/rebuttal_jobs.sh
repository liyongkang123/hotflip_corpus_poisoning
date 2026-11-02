#!/usr/bin/env bash
# Shared attack submission logic for the HotFlip and AGGD MN5 launchers.
set -euo pipefail

rebuttal_complete() {
    local dataset=$1 model=$2 seed=$3 k=$4 file count method=${ATTACK_METHOD:-hotflip_rebuttal}
    local num_iter=5000
    if [[ "$method" == aggd_rebuttal ]]; then num_iter=3000; fi
    file="$OUTPUT_ROOT/$dataset/${dataset}_train_attacked_documents_seed_${seed}_${model}_${method}_N${k}_Len50_Iter${num_iter}_Pool100_InitTrue.json"
    [[ -f "$file" ]] || return 1
    count=$(grep -c '"finish": true' "$file" || true)
    [[ "$count" -eq "$k" ]]
}

launch_rebuttal_jobs() {
    local dry_run=false
    if [[ ${1:-} == --dry_run ]]; then dry_run=true; shift; fi
    if [[ $# -gt 2 ]]; then
        echo 'Usage: bash <launcher> [--dry_run] ["model ..."] ["dataset ..."]' >&2
        return 2
    fi
    local -a models=("${MODELS[@]}") datasets=(nq-train hotpotqa msmarco)
    if [[ -n ${1:-} ]]; then read -r -a models <<< "$1"; fi
    if [[ -n ${2:-} ]]; then read -r -a datasets <<< "$2"; fi
    local model dataset seed k all_done allowed found
    local -a planned_models=() planned_datasets=() planned_ks=() planned_seeds=()
    for model in "${models[@]}"; do
        found=false
        for allowed in "${MODELS[@]}"; do [[ "$model" != "$allowed" ]] || found=true; done
        if [[ "$found" != true ]]; then echo "Model not supported by this server's launcher: $model" >&2; return 2; fi
    done
    for dataset in "${datasets[@]}"; do
        case "$dataset" in nq-train|hotpotqa|msmarco) ;; *) echo "Unknown dataset: $dataset" >&2; return 2;; esac
    done
    # Submit the longer N50 jobs first.
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if ! rebuttal_complete "$dataset" "$model" "$seed" 50; then
                    planned_models+=("$model"); planned_datasets+=("$dataset")
                    planned_ks+=(50); planned_seeds+=("$seed")
                fi
            done
        done
    done
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            all_done=true
            for k in 1 10; do
                for seed in "${SEEDS[@]}"; do
                    if ! rebuttal_complete "$dataset" "$model" "$seed" "$k"; then all_done=false; fi
                done
            done
            if [[ "$all_done" != true ]]; then
                planned_models+=("$model"); planned_datasets+=("$dataset")
                planned_ks+=("1 10"); planned_seeds+=("${SEEDS[*]}")
            fi
        done
    done
    local total=${#planned_models[@]} existing=0 i
    if [[ "$dry_run" != true ]]; then
        existing=$(squeue -u "$USER" --noheader --format='%A' | wc -l)
        if (( existing + total > MAX_JOBS )); then
            echo "Job limit: $existing existing + $total planned > $MAX_JOBS. Select fewer models/datasets." >&2
            return 1
        fi
        mkdir -p "$REPO/logs"
    fi
    echo "Planned attack jobs: $total (dry_run=$dry_run)" >&2
    export REPO OUTPUT_ROOT
    for ((i=0; i<total; i++)); do
        local -a command=(sbatch --chdir="$REPO" --job-name="${JOB_PREFIX:-hf}_${planned_models[i]}_${planned_datasets[i]}_N${planned_ks[i]// /_}_s${planned_seeds[i]// /_}")
        if [[ -n ${ACCOUNT:-} ]]; then command+=(--account="$ACCOUNT"); fi
        if [[ -n ${QOS:-} ]]; then command+=(--qos="$QOS"); fi
        command+=("$SUB_SCRIPT" "${planned_models[i]}" "${planned_datasets[i]}" "${planned_ks[i]}" "${planned_seeds[i]}")
        if [[ "$dry_run" == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; else "${command[@]}"; fi
    done
}
