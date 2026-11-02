# 交接说明

## 最新更新（2026-11-02，本轮优先于下方历史交接）

- **用户停用 DIVER-0.6B**：已从 HotFlip/AGGD MN5 默认列表和 Snellius 编码分组移除；当前每种方法 13 模型、234 jobs，
  合计 468。历史缓存、结果及模型加载兼容性保留。用户同时询问 Contriever/BGE-M3 是否可免重跑 HotFlip：
  已核对旧代码，空 prompt 仅使它们不受 prompt 插入错位影响，旧文档 CLS/SEP 未固定、query 目标截断规则也不同，
  因此当前仍保留两者 HotFlip，建议重跑；用户尚未明确决定跳过。
- **用户最新决定：全部攻击统一 MN5，含 Harrier-27B。** 用户已完成 Snellius 的全部 query 编码，已只读确认 42 份
  `queries.json`。两个 MN5 launcher 均包含全部 14 模型，sub 脚本取消了禁止 Harrier-27B 的限制；各方法 252 个作业。
  HotFlip 5000 步、AGGD 3000 步、分块 100、独立结果目录和共享 query 缓存均保持；无自动编码或依赖。
  四份 `scripts/generate_{hotflip,aggd}_rebuttal_snellius{,_sub}.sh` 已移除，编码脚本保留。不要运行下文历史的 Snellius 攻击命令。
  用户授权取消 Snellius 本项目任务，但队列中没有编码/攻击任务可取消，仅有共享 `embedding` 28848113 @ gcn135；该交互作业保留。
  用户需将更新后的代码/prompts 与 `cache_rebuttal/query_embeddings/` 同步到 MN5 项目相同位置后自行提交。
  两种方法共 504 个配置作业，300 上限仍保留，须分批。Agent 不提交 MN5 batch 作业。
  已通过 shell 语法、dry-run 和假 sbatch 检查：每种方法 252 条，含 18 个 Harrier-27B 配置；已有 50 + 252 超 300 正确拒绝。
  42 份 query manifest 的对应 `.npy` 均存在且非空。本轮无 Python/GPU 运行、无真 batch 提交、无作业取消（队列无匹配任务）。
- **用户最新决定：正式攻击统一 `cand_chunk=100`。** HotFlip/AGGD、Snellius/MN5、所有模型（含 Harrier-27B）
  默认均为 100；已将两份 Snellius sub 脚本从 10 改为 100，Python 和 MN5 原本就是 100。
  显存报告中的 chunk 10/100 仍是历史实测对照，不要据此把正式默认值改回 10。
- **Harrier-27B 显存复测（用户新指定 gcn135，28848113 embedding，96 GB H100）**：新增
  `check_harrier_memory.py`，在 PyTorch allocator 限额 **58 GiB** 下完成 13 个阶段，无 OOM。
  100 候选、50 tokens、GPU 9000 行训练均值、每种攻击每个 chunk 各 5 迭代：chunk 10 时 HotFlip/AGGD 整卡峰值
  约 **52.87 GiB**，chunk 100 时最高 **53.96 GiB**；PyTorch allocated 约 51.58 GiB。
  编码 batch 64 × 256-token 压力输入整卡峰值 **55.40 GiB**。因此在本次代码/环境/输入下，64 GB 容量应足够，
  用户随后决定正式统一 cand_chunk=100；尚未在 MN5 软件环境/实际 64 GB 卡上验证，未更改服务器分工脚本。
  报告与原始数据：`docs/benchmarks/harrier_27b_memory_gcn135.{md,json}`；日志 `logs/harrier-memory-gcn135-28848113.out`。
  本轮未生成正式攻击文档、未修改持久 query 缓存。gcn135 取代历史 gcn49/gcn160，使用前检查 RUNNING/GPU。
- **用户最新决定：AGGD 默认 3000 steps，HotFlip 保持 5000。** 已同步 Python 默认值、两平台 AGGD sub 脚本、
  输出文件名及启动器完成检查（AGGD `Iter3000`）；CLI 显式 `--num_iter` 仍可覆盖。持久 query 缓存无需重算。
- **新增 AGGD 的 LLM 适配（本轮）**：已核对官方 `JinyanSu1/AGGD` 的 `utils/AGGD.py` 与 `main.py`，提交
  `aa0e4fd29a99b2f73361b28cf6644ab2382e19a6`。实现位于 `model/aggd.py`、`aggd_attack_rebuttal.py`，方法名
  `aggd_rebuttal`；按所有位置的 token 排名交错取候选，验证最佳结果改进时重置 depth，否则递增，候选池耗尽后循环。
  沿用本项目均值目标、gold 初始化，当前参数为 50/3000/100，不使用官方逐 query/gold ASR gate，因此是搜索机制移植而非官方实验完整复现。
  复用同一份训练 query 缓存和 14 模型编码器；`hotflip_attack_rebuttal.py` 只增加默认不改变 HotFlip 的 method/runner 接口。
  AGGD 结果单独保存到 `output_attack/attacked_text/document/aggd_rebuttal/<dataset>/`，字段/续跑同 HotFlip。
  临时目标缓存独立：MN5 `cache_aggd_rebuttal/`，Snellius `/scratch-shared/yli4/aggd_rebuttal_cache/`。
  脚本：`scripts_mn5/generate_aggd_rebuttal{,_sub}.sh`（234 jobs），
  `scripts/generate_aggd_rebuttal_snellius{,_sub}.sh`（仅 Harrier-27B，18 jobs）；无编码提交或 Slurm 依赖。
  详细解释及官方来源：`docs/AGGD_REBUTTAL.md`。LLM_Robust 仍未修改。
  验证：用户指定的 **gcn49 / RUNNING embedding 28818423 / A100 40 GB** 上，
  `tests/test_aggd.py tests/test_rebuttal_pipeline.py` 共 13 项通过（57.88s），包括真实 Qwen-0.6B 的生成、分数重算、续跑和 HotFlip 回归。
  候选排序/掩码、depth 递增/重置及有限池耗尽也已验证。新脚本语法与 dry-run（234/18）通过。
  尚未在 14 模型上逐一跑 AGGD，更未做正式长跑；未提交 batch、未 commit。gcn49 目前取代历史 gcn160，使用前仍须检查 RUNNING/GPU。
- **用户最新纠正：编码与攻击必须手动分开，不要自动提交编码或添加依赖。**
  已简化为 `bash scripts/encode_hotflip_rebuttal_queries_snellius.sh [--dry_run]`：在 **Snellius 用 3 个独立 job
  编码全部 14 模型 × 3 训练集**。分组在启动脚本顶部；sub 脚本顺序遍历模型/数据集，已有缓存跳过。
  用户完成编码后手动把 `cache_rebuttal/query_embeddings/` 同步到 MN5，再手动运行攻击启动器。
  攻击仍为 Snellius 仅 Harrier-27B（18 jobs）、MN5 其余 13 模型（234 jobs）。自动编码/afterok 逻辑已移除，
  新增的 MN5 编码 sub 脚本已删除。底层持久缓存及 Python 编码程序保持不变。
  语法及假 sbatch 检查通过：恰好 3 个编码 job、42 个无重复的模型/数据集组合；攻击 dry-run 仍为 18/234，
  无编码任务或依赖参数。本轮未运行 Python 验证或提交真作业。
- **最新决定与实现：用户确认没有 train query 表示，要求一次编码后跨全部 seed/N 作业复用，保存在本项目下。**
  新增 `encode_rebuttal_queries.py`；`rebuttal_data.py` 实现持久 float32 NPY + JSON（query IDs、输入指纹、编码配置、cache ID），
  默认 `$REPO/cache_rebuttal/query_embeddings/<dataset>/<model>/queries.{npy,json}`。按 query ID 对齐，保留多 gold 的配对权重。
  批量写盘、只读 mmap、编码文件锁、完成标记最后发布；已完成缓存跳过模型加载。攻击阶段严格只读缓存，缺少时不会自行编码。
  cache ID 纳入临时 cluster 目标缓存元数据；训练 query 缓存不会在攻击完成后删除。`cache_rebuttal/` 已加入 gitignore。
  编码脚本为 Snellius `scripts/encode_hotflip_rebuttal_queries_snellius{,_sub}.sh`，具体手动流程见上一条。
  `QUERY_EMBEDDING_ROOT` 可覆盖默认路径；`QUERY_BATCH_SIZE` 只用于编码作业。提交命令仍见使用说明。
  上轮 8 项缓存/短攻击检查通过。之前的自动编码依赖验证已不适用于现有脚本。
  未提交真作业，未编码正式数据，未改 LLM_Robust。
- 用户随后提出复用 clean query embeddings。已只读核对 `LLM_Robust/eval.py`：Snellius 根目录
  `/scratch-shared/yli4/LLM_Robust/Clean`，文件为 `output/embeddings/<model>/<dataset>/queries.pkl`，
  pickle 内容 `(embeddings, query_ids)`；用户给出的 MN5 对应目录为
  `/gpfs/scratch/ehpc695/LLM_Robust/Clean/output/embeddings/`。
  **当前不能直接替换 train query 编码**：Qwen-0.6B 的 nq/hotpotqa/msmarco 缓存分别为
  3452/7405/7077 条评估 query，与攻击 train qrels 的 132803/85000/502939 个 query ID 的交集全部为 0。
  指定 Snellius 根目录下未找到 nq-train 或训练 query 文件；eval 的 loader 只返回所选 split 的 queries。
  用户已确认没有另外的训练缓存，因此采用上面新增的一次性 train 编码流程，攻击目标始终保持训练 query。
- 用户要求精简验证，先只完成当前仓库。**不要修改 LLM_Robust**；配套的白盒特殊 token 构造、source tokenizer
  解码迁移、评估和统计留到后续。原计划的逐 Task 子代理流程及 A6/A8 的全模型重验证不再作为前置条件。
- 已新增 `rebuttal_data.py`、`hotflip_attack_rebuttal.py`，接通已有 `utils/rebuttal_utils.py::RebuttalEncoder`。
  旧代码与既有结果未修改，不 commit、不提交 sbatch。
- Query 按 prompt + text 整体 tokenize；文档固定特殊 token/prompt，只修改 50 个正文 id。模型冻结，只对 embedding
  输出求梯度。新增入口还修正了旧循环在不同 query 批次间比较 baseline/candidate 分数的问题。
- 训练数据从 train qrels 配对，流式筛选需要的 corpus 文档；跨作业复用持久 query 表示。缓存目标与初始化，JSON 原子保存。
  `finish: true` 的 cluster 跳过；中断的 cluster 从缓存初始化重新运行；全部完成后删除该 (k, seed) 缓存。
- **攻击服务器分工**：`scripts/generate_hotflip_rebuttal_snellius{,_sub}.sh` 仅 Harrier-27B（18 attack jobs）；
  `scripts_mn5/generate_hotflip_rebuttal{,_sub}.sh` 为其余 13 模型（234 attack jobs）；编码全部在 Snellius 手动提交。
  `scripts/rebuttal_jobs.sh` 为共享启动逻辑。支持 `--dry_run`、模型/数据集筛选、完成跳过与 300 作业上限。
- 使用说明：`docs/HOTFLIP_REBUTTAL.md`。模型短名 `contriever` 在新入口指 contriever-msmarco；新输出单独存于
  `output_attack/attacked_text/document/hotflip_rebuttal/`。MN5 尚未实际执行，不宣称跨服务器验证完成。
- 上轮验证在用户 RUNNING 的 28812722 @ gcn160 上完成（本轮缓存验证另见上方）：
  - 14 项针对性检查通过：5 项既有 encoder 检查（真实梯度、参数冻结、候选评分、ReasonIR mask、加载选项），
    4 项各模型 tokenizer 边界检查，5 项新增 `tests/test_rebuttal_pipeline.py`（query 拼接、缓存/小 cluster、
    真实短攻击分数、原子 JSON、main 只重跑未完成 cluster）。未跑全模型 clean queries.pkl 对比。
  - Contriever、BGE-M3、Harrier-270M、Harrier-0.6B、Harrier-27B 各完成 2 次真实攻击迭代，保存分数重算一致。
    Harrier-27B 峰值 allocated 51.40 GiB（3 queries、5 candidates、chunk 2，仅短检查，不代表完整 batch）。
  - 5 个 shell 文件语法通过；dry-run MN5 234、Snellius 18；假 sbatch 验证不提交真作业；完成跳过变为 233；
    已有 100 jobs + 233 planned 会正确拒绝。CLI `--help` 正常。
- 历史交接中“下一步 A3”已经过时。接下来由用户同步到 MN5 并按使用说明提交生成；不要自动开始修改 LLM_Robust。

---

# 历史交接（2026-11-02 凌晨，Claude → 下一个 agent）

阅读顺序：`AGENTS.md`（仓库概况、Slurm 规则、已知问题）→ 本文件 → 设计
`docs/superpowers/specs/2026-11-01-hotflip-rebuttal-design.md` → 实施计划
`docs/superpowers/plans/2026-11-02-hotflip-rebuttal.md` → 执行记录
`.superpowers/sdd/2026-11-02-hotflip-rebuttal/progress.md`（ledger，含全部 ruling）。用户用中文交流。

## 1. 仓库概况

- ECIR 2025 复现论文 "Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval" 的代码。`main` 是论文代码（BERT 系检索器）；
  当前分支 `dev_llm` 把 HotFlip 扩展到 LLM 检索器，为 `/gpfs/work4/0/prjs0928/LLM_Robust`（TOIS 2026 投稿，RQ2-2 corpus poisoning）
  生成下毒文档。`dev_llm` 的生成全部在 MareNostrum 5 上运行，`logs/`、`results/hotflip-generate/`、`output_attack/` 是事后拷过来的；
  `LLM_Robust/output_attack/attacked_text/document/` 是逐字节相同的副本。文件、输出、模型的细节见 `AGENTS.md`。
- 2026-11-01 发现三个问题（细节与实测数字见 `AGENTS.md` 的 Pitfalls）：
  1. `utils/load_model.py::HFtoSF.encode_tokenized` 把 prompt 插在第 0 个 token 之后，攻击时的 query 目标（以及 diver / reasonir 的文档）
     与评估时不一致；
  2. 对抗文档在生成端和评估端都没有特殊 token（Qwen 系末尾的 `<|endoftext|>` 正是 clean 文档的 pooling 位置）；
  3. `LLM_Robust/eval_transfer_attack.py` 用 target 的 tokenizer 解码 source 的 id（词表不同时注入的是乱码）。用户决定不修旧文件，
     只在新评估代码里做对。

## 2. 用户的需求（当前任务：用新方法 `hotflip_rebuttal` 重做 RQ2-2）

目标：攻击时、评估时、clean 文档三者编码一致，重新生成对抗文档，重做白盒与迁移评估。用户已决定的全部内容（设计 §4 与之后的回答）：

- 方法名 `hotflip_rebuttal`；训练集 nq-train / hotpotqa / msmarco（train），评估 nq test、hotpotqa test、msmarco dev（另存 DL19 / DL20）；
  seed 1999 5 27 2016 2026；白盒 N = 1、10、50；迁移 N = 50。
- 14 个模型（LLM_Robust 短名）：contriever（= contriever-msmarco）、bge_m3、qwen3、qwen3_4B、qwen3_0.6B、linq、gte、reasonir、diver、
  diver_0.6B、bge_reasoner、harrier-oss-v1_27B、harrier-oss-v1_0.6B、harrier-oss-v1_270M。排除 DIVER-1.7B、SPLADE-v3。
  Harrier 的 prompt 从 LLM_Robust 复制（已完成）。
- 对抗文档在 id 层面按 clean 文档编码：`pre_sp + passage_prompt_ids + 50 个对抗 id + suf_sp`；白盒不走"解码再编码"。
- 候选 token 以 tokenizer 为准（特殊 token 可以，embedding 矩阵里 tokenizer 没有的行排除）。
- 续跑：读取对抗文档 JSON，每个条目加 `"finish": true`，没有 `finish` 的条目重新生成；每个 cluster 单独设种子，k-means 结果存盘。
- 作业划分：每个 (模型, 数据集) 1 个作业跑 N1 + N10 × 5 个 seed，N50 每个 seed 1 个作业；MN5 上作业数 ≤ 300（13 个模型 → 234 个）。
- 运行位置：Harrier-27B 在 Snellius 上生成（1 张 96 GB H100，单作业最长 5 天，18 个作业）；其余 13 个模型在 MN5（单作业最长 3 天）。
  Harrier-270M / 0.6B 若在 MN5 上复现不了（preflight 不通过），用户自己升级 MN5 的 transformers。
- MN5：仓库 `/gpfs/projects/ehpc695/hotflip_corpus_poisoning`，`ACCOUNT=ehpc821`，`QOS=acc_ehpc`，
  `PYTHON=/gpfs/projects/ehpc695/anaconda/envs/ir/bin/python`，`HF_HOME=/gpfs/projects/ehpc695/cache`，`HF_HUB_CACHE=$HF_HOME`，
  `HF_DATASETS_CACHE=/gpfs/projects/ehpc695/cache`；存储充足。
- 保存：一个 attack method 一个文件夹 `output_attack/attacked_text/document/hotflip_rebuttal/<ds>/`；评估端 `retrieval_results.json`
  只存 top-100（指标仍在合并后的 top-1000 上算；work4 配额已用 92%）；k-means 缓存放在 MN5 `$REPO/cache_rebuttal/`、Snellius
  `/scratch-shared/yli4/hotflip_rebuttal_cache/`。
- 迁移：source 与 target 的 tokenizer 相同时直接用 id，不同时先用 source 解码、再用 target 按 clean 文档编码。
- RQ3 之后由用户自己切换到新结果；先只做 RQ2-2。
- **工作方式**：尽可能复用已有代码，确实不同的函数才新写；现有文件一律不改（只新增）；不 commit（用户自己 commit；LLM_Robust 里还有
  用户暂存的改动）；用户希望尽快完成，作业时长不是问题；执行方式为 subagent-driven（每个 Task 一个实现子代理 + 一个 reviewer）。

## 3. 工作规则

- 登录节点不运行任何 Python（包括 `python3 -c`、读 JSON；这次会话违反过三次，不要再犯）。所有 Python 都用 `srun --jobid=<embedding 作业>
  --overlap` 进入用户的 `embedding` 作业运行，模板见计划的 Global Constraints（另加 `export USE_TF=0`，见 ledger Ruling 8）。当前作业
  28812722 @ gcn160，约 2026-11-02 12:15 结束；之后按 `AGENTS.md` 重新查找，没有就问用户。
- Agent 不提交 `sbatch`，只写脚本、用假的 `sbatch` dry-run，把提交命令交给用户。
- 在 LLM_Robust 中工作时遵守它的 `CLAUDE.md`：每完成一步在 `EXPERIMENT_LOG.md` 追加记录。
- 子代理只能用 `model: inherit`（环境限制，用户没有指定模型）。

## 4. 进度（2026-11-02 02:10）

| 项目 | 状态 |
|---|---|
| 设计文档 | 完成，用户已确认 |
| 实施计划（含"复用原则"表） | 完成，用户已确认 |
| SDD 工作区 | `.superpowers/sdd/2026-11-02-hotflip-rebuttal/`（git 忽略）：`context.md`（每次派发都给）、各 Task 的 brief、`progress.md`（ledger） |
| Task A1 + A2（模型表、prompt、tokenizer 层构造、候选范围） | **完成**，review 通过（修复 1 轮）；16 个测试通过 |
| Task A3（`RebuttalEncoder`） | **被用户中断，未验证**：代码已写入 `utils/rebuttal_utils.py` 第 107–202 行（`bf16_kwargs`、`load_hf_model`、`_hftosf`、`RebuttalEncoder`、`rebuttal_get_emb`），测试 `tests/test_rebuttal_encoder.py`（15 个）已写；没有报告，不知道测试是否跑过 / 通过，未 review |
| Task A4–A8、B1–B5、最终 review | 未开始；LLM_Robust 尚未改动 |

目前新增的文件（都未 commit）：`utils/rebuttal_utils.py`、`prompts/harrier-oss-v1_{27B,0.6B,270M}.json`、
`tests/test_rebuttal_{prompts,tokens,encoder}.py`、`tests/fixtures/rebuttal_doc_ids.json`（V2 的参照，只有设置
`REBUTTAL_REGEN_FIXTURE=1` 时才重写）、`docs/superpowers/{specs,plans}/…`、`AGENTS.md`、`CLAUDE.md`（→ `AGENTS.md`）、本文件。
临时检查脚本：`/scratch-shared/yli4/hotflip_tmp/check_prompt_insertion.py`、`check_special_tokens.py`。

## 5. 下一步

1. **收尾 Task A3**：在计算节点上运行 `python -m pytest tests/test_rebuttal_encoder.py tests/test_rebuttal_prompts.py
   tests/test_rebuttal_tokens.py -q`；V1 标准为 qwen3_0.6B、bge_m3、contriever、harrier-oss-v1_270M 的 query 与 LLM_Robust clean
   `queries.pkl` 的 cos 均值 ≥ 0.9999、最小值 ≥ 0.999（不达标就停下来报告，不要放宽）。补写 `task-A3-report.md`，然后按 SDD 派发 task review。
2. 按计划顺序继续：A4 → A5 → A6 → A7 → A8 → B1 → B2 → B3 → B4 + B5 → 最终 review。每个 Task 派发时给 brief 路径、`context.md` 路径、
   前面 Task 的实际接口（见下）与 ledger 中的相关 ruling。
3. A8 完成后交给用户：Snellius 上提交 Harrier-27B（18 个作业）；MN5 上同步代码、跑 preflight、核对通过后提交 234 个作业；生成完后拷回
   文档、提交 42 个评估作业，再运行统计。

已有接口（Task A4 起要用）：`utils/rebuttal_utils.py` 中 `MODEL_SPECS`、`REBUTTAL_MODELS`、`get_rebuttal_prompts(model, dataset)`、
`load_tokenizer`、`special_token_ids`、`build_doc_ids(tok, passage_prefix, adv_ids)`、`adv_positions(tok, passage_prefix, n_adv)`、
`doc_prefix_len`、`candidate_mask(tok, num_rows)`；A3（未验证）：`RebuttalEncoder(model, dataset, device="cuda", device_map=None,
hf_model=None, cand_chunk=100)`，属性 `tokenizer, prompts, model, q_model, c_model（HFtoSF）, embedding_weight, device, pre_sp, suf_sp,
doc_head, doc_tail, cand_mask, cand_chunk`，方法 `eval()`、`encode_queries(texts, batch_size=64)`、`doc_ids(adv_ids)`、
`embed_doc_ids(ids, chunk=100)`、`doc_embedding_and_grad(ids, loss_fn) -> (loss, grad[L, h])`；`rebuttal_get_emb(c_model, inputs)`
（与 `model.hotflip.hotflip_candidate_score_new`、`utils.evaluate.evaluate_sim_ours_new` 兼容，`inputs["input_ids"]` 只含 50 个对抗 id）。

## 6. 容易出错的地方

- Harrier-270M 的 tokenizer 有一个 id（262144，`<image_soft_token>`）没有 embedding 行；`candidate_mask` 会排除并打印警告。
- diver 的 passage prompt `Represent this text:` 与正文首词在约 25% 的 clean 文档中会合并成一个 token（设计接受；A2 的测试样本恰好不触发）。
- transformers 4.57.6 对三个 Harrier tokenizer 打印 `fix_mistral_regex` 警告，是误报，不要设置该参数（LLM_Robust 也没设）。
- MN5 的环境据 `LLM_Robust/requirements.txt` 为 transformers 4.52.0 / torch 2.7.1，Harrier 的 config 由 4.57.6 保存；第一轮 6 个 LLM 的
  clean 表示是在 MN5 上算的，Harrier 的在 Snellius 上算的，所以 V1 / V5 用容差判断。
- clean 检索用 faiss GPU fp16（`useFloat16=True`）；对抗文档打分要用同一套设置（计划的复用表：`eval_attack_query_batch.build_dense_index`
  + `dense_search`）。
- 旧代码对全部权重求梯度（8B 多占 16 GB，27B 多占 54 GB）；新代码冻结权重，只对 embedding 层输出求梯度。
- 运行时间（上次 MN5）：每个 cluster contriever 约 2 分钟、bge_m3 约 3.5 分钟、0.6B 11–25 分钟、4–8B 22–34 分钟；Harrier-27B 未测，
  A8 要实测迭代速度与 query 编码吞吐（batch 32 / 128 / 256）。
