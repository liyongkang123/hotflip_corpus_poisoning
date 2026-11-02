#!/usr/bin/env bash
# Run this launcher with bash on the MN5 login node; it submits the GPU jobs.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ACCOUNT=${ACCOUNT:-ehpc821}
QOS=${QOS:-acc_ehpc}
ATTACK_METHOD=aggd_rebuttal
JOB_PREFIX=aggd
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/output_attack/attacked_text/document/aggd_rebuttal}
MAX_JOBS=${MAX_JOBS:-300}
MODELS=(contriever bge_m3 qwen3 qwen3_4B qwen3_0.6B linq gte reasonir diver bge_reasoner harrier-oss-v1_27B harrier-oss-v1_0.6B harrier-oss-v1_270M)
SEEDS=(1999 5 27 2016 2026)
SUB_SCRIPT="$REPO/scripts_mn5/generate_aggd_rebuttal_sub.sh"
source "$REPO/scripts/rebuttal_jobs.sh"
launch_rebuttal_jobs "$@"
