# AGENTS.md

Guidance for coding agents working in this repository. `CLAUDE.md` is a symlink to this file, so keep only one copy.
Last updated: 2026-11-02. The user communicates in Chinese; code comments are mostly English, newer files have some Chinese.

**Handoff (2026-11-02):** read the latest section of `HANDOFF.md` and `docs/HOTFLIP_REBUTTAL.md` first.
The user narrowed this round to this repository and reduced verification; LLM_Robust changes are deferred.

## Project

Code of the ECIR 2025 reproducibility paper "Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval"
(Li, Eustratiadis, Kanoulas; arXiv 2501.04802). `README.md` describes the paper's RQ1–RQ3 pipeline.

- `main`: the paper code, for BERT-style retrievers (`contriever`, `contriever-msmarco`, `dpr-single`, `dpr-multi`, `ance`,
  `tas-b`, `dragon`, `condenser`).
- `dev_llm` (checked out, tracks `origin/dev_llm`; **all work happens on this branch**, user's decision 2026-11-01, and every
  model, `contriever-msmarco` included, goes through `HFtoSF`): extends the HotFlip generator to LLM-based dense retrievers. It produced
  the corpus-poisoning documents of `../LLM_Robust` (TOIS 2026 submission, RQ2 corpus poisoning; see `../LLM_Robust/CLAUDE.md`).
  Those documents are evaluated in LLM_Robust (`eval_attack_new.py --attack_method supervised_poisoning`,
  `eval_transfer_attack.py`), not here.

Where it ran: all `dev_llm` generation ran on MareNostrum 5 (BSC; repo at `/gpfs/projects/ehpc425/hotflip_corpus_poisoning`,
account `ehpc425`, QoS `acc_ehpc`, scripts in `scripts_mn5/`); `logs/`, `results/hotflip-generate/` and `output_attack/` were
copied to Snellius afterwards, and every log in `logs/` is from MN5. The ECIR experiments ran on the UvA IvI cluster: the old
scripts in `scripts/` call `/ivi/ilps/personal/yli8/attack_baseline/...` and have no resource headers.

## Code map

| File | Role |
|---|---|
| `hotflip_attack_batch_all_llm.py` | **Generator used for LLM_Robust.** One job per (dataset, model, k, seed): k-means once, then HotFlip for all k clusters in turn |
| `hotflip_attack_ours_llm.py` | Same attack, one cluster per job (`--kmeans_split s`) |
| `hotflip_attack_ours.py`, `hotflip_raw.py` | ECIR RQ1: our mean-embedding pipeline vs the original Zhong et al. (2023) code |
| `embedding_index.py`, `evaluate_attack.py` | ECIR: clean BEIR retrieval (`results/beir_result/<ds>/<model>/beir.json`) and attack Recall@k |
| `attack_results_statistical.py`, `transfer_attack_statical.py` | ECIR RQ1 / RQ2 statistics over seeds |
| `hotflip_ours_attack_corpus.py` | ECIR RQ3 (corpus attack by `--attack_rate`) |
| `hyperpan_Imax.py`, `hyperpan_Imax_eval_statical.py` | ECIR I_max study (`results_hyper_iter/`) |
| `print_adv_document.py` | Decodes one aggregated output file (loads the model at import, so compute node only) |
| `config.py` | The single argparse used by every script |
| `utils/load_model.py` | `load_models(model_code, dataset)` -> `(q_model, c_model, tokenizer, get_emb)`; `HFtoSF` wrapper for LLMs |
| `utils/utils.py` | model code -> HF name maps, `get_model_prompts_tasks`, `get_embeddings`, `GradientStorage`, `compose_model_inputs` |
| `utils/load_data.py` | BEIR download, k-means, batch-mean query embeddings (`load_data_ours_batch_all` / `_new`; `load_data` is Zhong's) |
| `model/hotflip.py` | Candidate tokens (gradient x embedding matrix) and batched candidate scoring |
| `prompts/<model>.json` | Per-dataset `query` / `passage` instructions (`nq-train` uses the `nq` entry) |

Run every script from the repository root (`import config`, `utils.*`, relative output paths).

## The LLM generator (`hotflip_attack_batch_all_llm.py`)

1. Load the (query, gold passage) pairs of the training qrels (`--split train`; forced for `nq-train`), embed all queries with
   `q_model`, k-means with `KMeans(n_clusters=k, random_state=0)`.
2. Per cluster: query embeddings (`--max_query_length 32`) are split into train / valid (valid = min(1000, 30%)), and each part is
   averaged in batches of 64 ("mean embedding" optimization of the ECIR paper).
3. The adversarial passage has 50 tokens, initialized from a random gold passage of the cluster (`--init_gold`; otherwise
   mask/unk/pad/bos/eos id). For `--num_iter 5000` iterations: gradient at the input embeddings for one train batch mean, a random
   position, top `--num_cand 100` candidates, scored on the next batch mean; the best passage is kept by mean similarity to the
   valid batch means.

Models (`utils/load_model.py`, HF names in `utils/utils.py`): BERT-style codes as on `main`, and through `HFtoSF` (bf16, query
and passage encoders share one base model, prompts from `prompts/`): `contriever-msmarco` (mean pooling, not normalized), `bge_m3`
(CLS), `reasonir` (mean without prompt tokens), and last-token pooling for `qwen3`, `qwen3_4B`, `qwen3_0.6B`, `linq`, `gte`,
`diver`, `diver_1.7B`, `diver_0.6B`, `bge_reasoner`. LLM_Robust's `contriever` is this repo's `contriever-msmarco`. LLM_Robust's
revision models `harrier-oss-v1_*` and `splade_v3` are not supported here.

Runtime per cluster on one MN5 H100 (last clusters of one hotpotqa log per model): `contriever-msmarco` ~2 min, `bge_m3` ~3.5 min,
`qwen3_0.6B` ~11 min, all other decoder models (including `diver_0.6B` / `_1.7B`) 22–34 min. N50 with a 7–8B model is ~25–28 h,
longer than the 20 h limit in `scripts/generate_hotflip_multi_batch_all_sub.sh`.

## Outputs

- Per cluster: `results/<method>-generate/<attack_dataset>/<model>/k{k}-s{s}-seed{seed}-num_cand100-num_iter5000-tokens50-gold_initTrue.json`,
  one JSON object `{it, best_sim, best_adv_text, best_adv_passage_ids, tot}`, rewritten on every improvement.
- Aggregated (read by LLM_Robust): `output_attack/attacked_text/document/<ds>/<ds>_train_attacked_documents_seed_<seed>_<model>_supervised_hotflip_N<k>_Len50_Iter5000_Pool100_InitTrue.json`,
  a dict `"0".."k-1"` -> `{best_sim, best_adv_text, best_adv_passage_ids, tot}`. It is rewritten after each cluster, so a job that
  stopped early leaves fewer than N entries (LLM_Robust asserts `len == N`). `best_adv_passage_ids` are ids of the attacked model's
  tokenizer; LLM_Robust (`eval_attack_new.py`) encodes them at the id level as
  `tokenizer(passage_prompt, add_special_tokens=False) + best_adv_passage_ids` with an all-ones mask (the decoded text is not used).
  LLM_Robust's loader hard-codes this file name, so keep the naming.
- Inventory (2026-11-01): datasets `nq-train`, `hotpotqa`, `msmarco` x seeds 1999 5 27 2016 2026 (the ECIR runs used 2024 instead
  of 2026); N10 and N50 for `bge_m3`, `bge_reasoner`, `contriever-msmarco`, `diver`, `gte`, `linq`, `qwen3`, `reasonir`; N50 only
  for `qwen3_0.6B`, `qwen3_4B`, `diver_0.6B`, `diver_1.7B` (300 files, plus a stale `..._contriever-msmarco_..._batch_old.json` in
  `nq-train`). `../LLM_Robust/output_attack/attacked_text/document/` is a byte-identical copy (`diff -rq`, 2026-11-01); LLM_Robust
  reads its own copy. Do not overwrite or delete either copy without asking.
- Leftovers in `results/hotflip-generate/`: `nq-train/contriever-msmarco_1`, `nq-train/contriever-msmarco_all` (seed-2024 tests),
  `fiqa/gte` (1 file), `scifact/reasonir` (empty); `diver_0.6B` / `diver_1.7B` have k10 per-cluster files but no N10 aggregate.
- `results/attack_results/hotflip/nq-train-contriever-msmarco/` and `results/beir_result/{nq,fiqa,scifact}/contriever-msmarco/`:
  `evaluate_attack.py` / `embedding_index.py` outputs for contriever-msmarco.
- `logs/`: 385 MN5 Slurm logs, 21 GB, mostly tqdm progress bars. Read them with `tail -c` / `head -c`, not whole; `*.out` is
  gitignored, so `rg` skips them unless given `--no-ignore`.
- Git: `*.out`, `/output_attack/` and `k*.json` (which covers `results/hotflip-generate/`) are ignored; the `.gitignore` lines for
  the last two are an uncommitted change. `._.gitignore` is a macOS AppleDouble file.

## Environment

- Conda env `ir` (`/gpfs/work4/0/prjs0928/conda/envs/ir`): `eval "$($HOME/anaconda3/bin/conda shell.bash hook)" && conda activate ir`,
  plus `export LD_LIBRARY_PATH=/gpfs/work4/0/prjs0928/conda/envs/ir/lib:$LD_LIBRARY_PATH`.
- `HF_HOME=/gpfs/work4/0/prjs0928/cache` (models in `hub/`; BEIR datasets in `datasets/`, which has `nq`, `nq-train`, `hotpotqa`,
  `msmarco`, `fiqa`, `scifact`). The `dev_llm` loaders and `evaluate_attack.py` read `$HF_HOME/datasets`; the older scripts
  (`embedding_index.py`, `hotflip_raw.py`, `hotflip_attack_ours.py`, `hotflip_ours_attack_corpus.py`) download into `./datasets`,
  which does not exist here.
- wandb: `wandb.init` is commented out in the `dev_llm` scripts but still active in the older ones.

## Slurm compute-node policy (mandatory)

The login node may be used only for repository operations: reading/editing files, `rg`, `sed`, `ls`, `git status`,
`git diff`, and inspecting jobs with `squeue`/`scontrol`.

**Do not run project Python on the login node, not even small checks** (the user's instruction of 2026-10-28, restated for
this repository on 2026-11-01; it replaces the earlier allowance for small CPU tests). Importing `beir` or
`sentence_transformers` there also takes minutes. Run every Python command, small statistics included, on a compute node that
belongs to one of the user's `RUNNING` jobs:

- Find the node from the user's interactive job named `embedding`: before each session the user allocates one by hand, so
  `squeue --me --name=embedding --states=RUNNING --format='%A|%N'` gives the node to `ssh` to. A node the user names in the
  conversation takes precedence; if there is no running `embedding` job and no named node, ask the user.
- Before each use, confirm the job is still `RUNNING` on that node (`squeue --me --states=RUNNING --format='%A|%j|%N'`) and
  that the GPU is still free (`nvidia-smi`). If the job has ended or the GPU is taken, stop and ask the user.
- Without a persistent shell, send each batch of commands through one SSH call that sets up the environment itself:

```bash
ssh <node> 'bash -l -s' <<'EOS'
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
export LD_LIBRARY_PATH=/gpfs/work4/0/prjs0928/conda/envs/ir/lib:$LD_LIBRARY_PATH
conda activate ir
export HF_HOME=/gpfs/work4/0/prjs0928/cache
cd /gpfs/work4/0/prjs0928/hotflip_corpus_poisoning
python ...
EOS
```

**If the user has several jobs on the same node, `ssh` may place the session in any of them** (on 2026-11-01 sessions on gcn129
landed in BM25 batch job 28797938 instead of the `embedding` job and shared its 16 cores). To run inside the intended job, start
a step in it from the login node instead of `ssh`:

```bash
srun --jobid=<embedding job id> --overlap -N1 -n1 --gres=gpu:<n> bash -l -s <<'EOS'
... same environment setup as above ...
EOS
```

Check `echo $SLURM_JOB_ID`, `nproc` and `nvidia-smi` inside; never use the cores or GPUs of the user's batch jobs.

### Rules

Anything that uses a GPU or processes data at scale must run on the GPU compute node. This includes loading or running models
(PyTorch/CUDA/FAISS-GPU workloads), encoding or reranking corpora, training, building indexes, full evaluation runs over
datasets, long-running scripts, and compiling CUDA/C++ extensions. Do not run such commands on the login node, even as a
fallback when the queue is busy. When unsure whether a job counts as small, use the compute node.

Experiments run as batch jobs submitted with `sbatch`; the user submits them. The named node above and the interactive
`embedding` allocation described next are only for short checks, such as testing a script on a small dataset before writing
its job script. When the user has named a node, use it instead of looking up or requesting an `embedding` allocation.

Node named by the user on 2026-11-01 for debugging this repository: gcn160 (`embedding` job 28812722, 1 H100, 16 cores). Check
that it is still `RUNNING` before each use, as above.

Update (user, 2026-11-02): current test node is **gcn49**, `embedding` job **28818423**, A100 40 GB.
This supersedes the historical gcn160 allocation; confirm RUNNING/node/GPU availability before each use.

Latest update (user, 2026-11-02 evening): test node is **gcn135**, `embedding` job **28848113**, H100 96 GB.
This supersedes gcn49/gcn160; check RUNNING and GPU availability before each use.

### The `embedding` allocation

The job name `embedding` is intentionally shared with `/gpfs/work4/0/prjs0928/Embedding_Isotropy` and
`/gpfs/work4/0/prjs0928/Decision_Retrieval`: the projects reuse the same allocation and the same GPU. Do not rename it. Before
loading a large model, check free GPU memory with `nvidia-smi` and report a conflict instead of killing the other projects'
processes.

Before requesting a new allocation, check for an existing job owned by the current user with the exact job name `embedding`:

```bash
squeue --me --name=embedding --states=RUNNING,PENDING --noheader --sort=-S --format='%A|%T|%N'
```

Follow this decision order:

1. If an `embedding` job is `RUNNING`, use the first/newest running entry. Read its node from the `%N` field (normally a single
   hostname such as `gcn142`; resolve a compact node list with `scontrol show hostnames <node-list>`).
2. Compare the selected node with `hostname -s`. If they match, the shell is already on the allocated compute node; do not SSH
   or start a nested `srun`.
3. If they differ, use that node through SSH (one SSH call per batch of commands, as above, when no persistent shell is
   available).
4. If an `embedding` job exists but is only `PENDING`, do not submit a duplicate job and do not run the command locally. Wait
   until it is running, or report that execution is waiting for Slurm resources.
5. Only when no `RUNNING` or `PENDING` `embedding` job exists, request a new persistent interactive allocation:

```bash
srun --job-name=embedding --partition=gpu_h100 --gres=gpu:1 --mem=180G --time=05:00:00 --pty bash -l
```

After entering the compute node, verify the execution location before running code: `hostname -s` must belong to the selected
running job, `echo $SLURM_JOB_ID` should show it, `nvidia-smi` must show an H100, and `which python` must resolve to
`/gpfs/work4/0/prjs0928/conda/envs/ir/bin/python`. Never SSH to a compute node that is not assigned to a currently `RUNNING`
job owned by the current user. If the job ends, stop using that node and repeat the lookup procedure.

## Job scripts

- Latest update (user, 2026-11-02): query encoding is complete on Snellius; ALL rebuttal attacks, including Harrier-27B,
  run on MN5. The newer Snellius attack launch/sub scripts were removed; legacy paper scripts remain unchanged.
- Snellius: `scripts/generate_hotflip_multi_batch_all.sh` (edit the lists at the top) submits one
  `scripts/generate_hotflip_multi_batch_all_sub.sh <dataset> <model> <k> <seed>` per combination; both use `gpu_h100`, 1 GPU,
  180G and write `logs/%x-%j.out`. They do not export `HF_HOME` or `LD_LIBRARY_PATH`; add both (as in the SSH template) before use.
- MN5: `scripts_mn5/` (same structure, `module load anaconda/2024.02`, `/gpfs/projects/ehpc425/...` paths); does not run on Snellius.
- Agents do not submit batch jobs: write or edit the scripts, dry-run them with a fake `sbatch` on `PATH` that only echoes, and
  give the user the submit commands.

## Rebuttal generation pipeline (2026-11-02)

- Scope: this repository only. LLM_Robust's new document construction, transfer decoding and evaluation are deferred.
  The old plan's exhaustive checks and per-task subagent workflow are superseded by focused verification.
- Query precomputation: `encode_rebuttal_queries.py`; attack: `hotflip_attack_rebuttal.py`; targets/data: `rebuttal_data.py`;
  encoder/model table: `utils/rebuttal_utils.py`.
  Usage and output format: `docs/HOTFLIP_REBUTTAL.md`.
- Model loader supports 14 models for compatibility; current experiment defaults use 13, excluding the now-retired DIVER-0.6B.
  DIVER-1.7B and SPLADE remain excluded. New `contriever` means contriever-msmarco. Contriever/BGE-M3 still run HotFlip:
  empty prompts avoid the insertion bug, but legacy document special-token handling and query truncation differ from the new pipeline.
  Queries tokenize prompt + text together without extra spaces. Documents are fixed special tokens + prompt ids + 50 mutable
  ids + fixed suffix tokens. Separate prompt/body tokenization can differ from concatenated text tokenization at boundaries.
- Frozen weights, embedding-output gradients only, generation KV caches disabled. Candidate ids must have tokenizer and
  embedding entries. Baseline/candidates are scored on the same query batch; the old loop compared different batches.
- New outputs: `output_attack/attacked_text/document/hotflip_rebuttal/<dataset>/`. Atomic writes; only `finish: true` clusters
  are skipped. Unfinished clusters restart from cached initialization with independent cluster seeds. Targets are cached until
  all clusters of that (k, seed) finish; completed configurations skip model loading.
- Persistent training query cache: `$REPO/cache_rebuttal/query_embeddings/<dataset>/<model>/queries.npy` + `queries.json`.
  Float32 matrix, query IDs, input hash, encoding settings and unique cache ID; read-only mmap. Encoded once per model/dataset,
  with a writer lock and manifest published last; all seed/N jobs read it by query ID. Cache/data mismatch raises; existing
  valid caches skip model loading. Attack jobs never encode queries. Persistent query files are NOT deleted after attacks.
  The user confirmed LLM_Robust's Clean query files cover evaluation splits only, not these training queries.
- Query encoding was manually submitted on Snellius for ALL 14 models and 3 training datasets and is now complete (42 caches):
  `bash scripts/encode_hotflip_rebuttal_queries_snellius.sh [--dry_run]` submits 3 independent jobs; model groups are at the top.
  The user copies `cache_rebuttal/query_embeddings/` to MN5 before manually launching attacks. No automatic encode submission
  or Slurm dependencies in the attack launchers. The previously added MN5 encoding sub script was removed.
- MN5 attacks: `bash scripts_mn5/generate_hotflip_rebuttal.sh [--dry_run] ["models"] ["datasets"]` (234 jobs, 13 active models).
  Account ehpc821, QoS acc_ehpc; repo /gpfs/projects/ehpc695/hotflip_corpus_poisoning; 3-day jobs.
- No new attacks on Snellius: the four HotFlip/AGGD Snellius attack launcher/sub scripts have been removed.
  The reusable Snellius encoding scripts remain. Attack candidate chunk size is uniformly 100 for both methods/all models.
  Shared attack launch logic: `scripts/rebuttal_jobs.sh`. `QUERY_EMBEDDING_ROOT` overrides the project-local default;
  `QUERY_BATCH_SIZE` is for encoding only. Encoding sub script: `scripts/encode_hotflip_rebuttal_queries_snellius_sub.sh`.
- No agent batch submissions or commits. MN5 execution/transformers compatibility remains unverified on that server.
  Harrier-27B passed a two-iteration smoke on gcn160 (3 queries, 5 candidates, chunk 2), peak allocated GPU memory 51.40 GiB;
  this is not a full-dataset memory/time benchmark.
- Updated Harrier-27B capacity test on gcn135: `check_harrier_memory.py`, PyTorch allocator capped at 58 GiB, 100 candidates,
  50 tokens, 9000 GPU target means, 5 iterations per method/configuration. HotFlip/AGGD whole-device NVML peaks: ~52.87 GiB
  with chunk 10, up to 53.96 GiB with chunk 100. Query batch 64 × 256 tokens peaked at 55.40 GiB. All phases passed;
  64 GB should fit for these settings, but MN5's actual hardware/software environment is untested. Report:
  `docs/benchmarks/harrier_27b_memory_gcn135.md` with raw JSON alongside. The user subsequently moved all attacks to MN5.
- Persistent-cache changes: 8 focused cache/pipeline checks passed on gcn160, including a real Qwen-0.6B short attack that
  fails if query encoding is called inside attack preparation. The later manual script simplification uses shell checks only.

## AGGD adaptation (2026-11-02)

- `aggd_attack_rebuttal.py` and `model/aggd.py` implement `aggd_rebuttal` using the same 14-model encoder and persistent train
  query cache. Algorithm source: JinyanSu1/AGGD commit aa0e4fd29a99b2f73361b28cf6644ab2382e19a6.
- Preserves rank-major interleaving across all mutable positions and adaptive depth (reset on a new validation best).
  Masks invalid vocabulary rows, chunks candidate forwards and cycles after finite candidate exhaustion.
- Uses our mean-query objective, gold initialization and 50-token/3000-iteration/100-candidate setup (user decision 2026-11-02;
  HotFlip stays at 5000 iterations). AGGD script completion checks and output filenames use Iter3000. Official per-query/gold
  ASR acceptance/validation is not reproduced. Call this an LLM/mean-target adaptation, not a full official experiment reproduction.
- Shared generation runner in `hotflip_attack_rebuttal.py` accepts method/cluster-runner parameters; its HotFlip defaults stay
  unchanged. Separate output directory `output_attack/attacked_text/document/aggd_rebuttal/`; same JSON fields/resume behavior.
- Launcher: `scripts_mn5/generate_aggd_rebuttal.sh` (234 attacks, 13 active models including Harrier-27B). Supports `--dry_run`,
  uses shared `scripts/rebuttal_jobs.sh`, and submits no query-encoding jobs. Temporary target cache is separate from HotFlip:
  MN5 `cache_aggd_rebuttal/`; old Snellius scratch caches remain untouched. HotFlip+AGGD total 468 jobs, so submit in batches
  under the 300-job cap. Current Snellius queue had no project attacks/encoding to cancel; shared embedding 28848113 was kept.
- Details: `docs/AGGD_REBUTTAL.md`. On gcn49 A100, 13 focused AGGD/HotFlip checks passed, including real Qwen-0.6B generation,
  similarity recomputation and resume. Scripts passed syntax/dry-run and fake-sbatch completion checks. No formal jobs submitted;
  no all-model AGGD smoke or full run; LLM_Robust remains deferred.

## Pitfalls

- `contriever-msmarco` loads differently on the two branches: `main` uses the `Contriever` class (fp32), `dev_llm` uses `HFtoSF`
  (bf16, mean pooling; intended) because `'contriever' == model_code` now matches only `contriever`. ECIR numbers come from `main`.
- `evaluate_attack.py` cannot evaluate LLM models: `--eval_model_code` only accepts the BERT-style codes (the MN5 `bge_m3`
  evaluation jobs 32664008–32664012 failed with `invalid choice: 'bge_m3'`).
- **Prompt position mismatch between attack and evaluation** (verified 2026-11-01 with
  `/scratch-shared/yli4/hotflip_tmp/check_prompt_insertion.py` on gcn160). `HFtoSF.encode_tokenized` inserts the prompt ids after
  position 0, assuming position 0 is [CLS]/BOS and padding is on the right; LLM_Robust prepends the prompt to the query text
  (`"Instruct: ...\nQuery:" + query`, left padding for `qwen3*` / `diver*`) and prepends passage-prompt ids to the adversarial ids.
  Tokenizers loaded here without `padding_side` pad right for `qwen3*`, `diver*`, `bge_reasoner`, `reasonir`, `bge_m3`, left for
  `gte`, `linq`. Results at attack time:
  - `qwen3*`, `diver*`, `bge_reasoner` (no BOS, right padding): `who | Instruct: ... Query: | sang what ...`.
  - `gte`, `linq` (left padding): `<pad> | Instruct: ... Query: | <pad x20> | [<s>] who sang ...`.
  - `reasonir`: `<bos> <bos> <|user|>... <|embed|>\n | who ...` (the prompt ids include BOS because of `mask_prompt_mean`).
  - Passages: unchanged except `diver*` (`a_0 | Represent this text: | a_1 ... a_49` vs `Represent this text: | a_0 ... a_49`) and
    `reasonir`. `bge_m3` and `contriever-msmarco` have empty prompts and are not affected.
  - `HFtoSF` also appends a space to every prompt (`Query: ` vs `Query:`); alone this changes little (cos 0.997–0.999).
  - Measured with the 0.6B models, bf16, 512 nq-train queries: cos(attack-time, eval-time query) mean 0.964 / min 0.74 for `qwen3_0.6B`, 0.958 / 0.82 for
    `diver_0.6B` (dropping the instruction entirely gives 0.887 / 0.940); cos of the 64-query means used as HotFlip targets 0.963 /
    0.991; `diver_0.6B` adversarial docs cos 0.988. Mean query similarity of the 50 N50 docs: optimized 0.304 -> evaluated 0.272
    (`qwen3_0.6B`), 0.705 -> 0.686 (`diver_0.6B`). Effect on ASR and on the 4–8B models not measured.
  The existing documents were generated this way; changing it means regenerating them, so only on the user's decision.
- **Adversarial documents get no special tokens**, both here and in LLM_Robust's evaluation (raw `prompt_ids + adv_ids`), while
  clean documents are tokenized as text with them: Qwen-family tokenizers (`qwen3*`, `gte`, `bge_reasoner`) append `<|endoftext|>`,
  which is the pooled token for clean documents; `bge_m3` and `linq` wrap with `<s> ... </s>`; `diver*` add nothing. HotFlip may
  also flip position 0 (the `<s>` from the gold-passage init survives in 6/50 `bge_m3` and 1/50 `linq` docs). Checked 2026-11-01
  (`/scratch-shared/yli4/hotflip_tmp/check_special_tokens.py`, nq-train seed 1999 N50): cos(raw ids, ids + special tokens) mean
  0.84 / 0.91 / 0.94 for `qwen3_0.6B` / `bge_m3` / `linq`, and mean query similarity drops 0.272 -> 0.247, 0.370 -> 0.349,
  0.316 -> 0.309. Decoding to text and re-tokenizing gives similar numbers, but reproduces the ids for only 8% / 56% / 0% of docs.
- **LLM_Robust's transfer evaluation decodes the source model's ids with the target model's tokenizer**
  (`eval_transfer_attack.py`: tokenizer from `load_model_hf(target)` at line 387, `tokenizer.decode(token_ids)` in
  `load_poisoning_docs` at line 182), so whenever the two vocabularies differ, the injected text is garbage (pairs within the
  Qwen-tokenizer family are unaffected). The first-round "all transfer ASR@20 < 1%" result rests on this.
- Rebuttal rerun (`hotflip_rebuttal`, fixing the three issues above): design in
  `docs/superpowers/specs/2026-11-01-hotflip-rebuttal-design.md`; existing files and results stay untouched.
- `HFtoSF` hard-codes `device='cuda'` and calls `torch.cuda.empty_cache()` after every forward pass.
- `config.py` flags: `--init_gold` / `--no_init_gold` default to False (the scripts pass `--init_gold`, hence `InitTrue`);
  `--do_kmeans` is always True; `--attack_query` defaults to True (clusters and targets are query embeddings).
- `diver_1.7B` results are invalid: the public checkpoint does not reproduce its own README example (see `../LLM_Robust/CLAUDE.md`).
