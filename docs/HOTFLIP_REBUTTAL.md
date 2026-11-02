# 修正后的 HotFlip 生成流程

2026-11-02：本轮只处理 `hotflip_corpus_poisoning`。LLM_Robust 的白盒编码、迁移解码和评估入口留到后续修改。
旧生成代码和已有攻击文档保留，新增方法使用 `hotflip_rebuttal`。

另已新增 AGGD 搜索适配 `aggd_rebuttal`，共用本文的编码器和持久训练 query 缓存，说明及独立提交命令见
[AGGD_REBUTTAL.md](AGGD_REBUTTAL.md)。

## 编码与攻击

- `utils/rebuttal_utils.py`：14 个模型的配置、prompt、pooling 和编码器。新方法中的 `contriever` 指
  `facebook/contriever-msmarco`，与 LLM_Robust 的短名一致。
- Query 使用 `tokenizer(query_prompt + query_text)`，不经过旧的 `encode_tokenized` 插入逻辑；不额外补空格，
  不再把 query 截断为 32 tokens，最大长度与评估配置一致。
- 对抗文档输入为 `pre_sp + passage_prompt_ids + adv_ids + suf_sp`，只有中间的 50 个 `adv_ids` 可修改。
  保存的 id 不含固定边界。prompt 与正文在 id 层拼接，文本重新 tokenize 时可能发生边界合并，不能据此重建白盒输入。
- Gold 初始化使用 `add_special_tokens=False`；不足 50 tokens 时补齐。候选包括 tokenizer 中的特殊 token，
  但排除没有 embedding 行的 id（包括 Harrier-270M 的 262144）。
- 冻结模型权重，只对 embedding 输出求梯度；禁用 generation KV cache。
- `rebuttal_data.py`：train qrels 的 query/gold 配对、query 编码、k-means、每 64 条 query 的均值目标与缓存。
  只在内存保留 qrels 使用的正文，避免加载完整 corpus；所有 seed 和 N 共享持久保存的训练 query 表示。
- `hotflip_attack_rebuttal.py`：完整生成入口。候选和当前文档在同一 query 均值上比较，验证集决定最终保存文档。
- `encode_rebuttal_queries.py`：一次性编码入口。只读训练 query 和 qrels，不加载 corpus；编码结果写入项目目录。
  攻击入口不再编码 query，缺少训练缓存时会在加载模型前报错。

默认参数：训练集 `nq-train hotpotqa msmarco`；N = 1、10、50；seeds = 1999、5、27、2016、2026；
50 个正文 token，5000 次迭代，100 个候选。DIVER-1.7B 与 SPLADE 不在本轮模型列表中。
用户随后停用 DIVER-0.6B，当前默认编码/攻击列表为 13 个模型。它的已有缓存和历史结果保留。
Contriever/BGE-M3 仍保留 HotFlip：空 prompt 只意味着不受 prompt 插入错位影响；旧文档的 CLS/SEP 没有固定保护，
且旧 query 目标截断至 32 tokens，与修复后的统一输入规则不同。

## 服务器与提交

所有命令从仓库根目录执行。启动器用 bash 运行，只有 sub 脚本在 GPU 作业内执行 Python。
Agent 未提交正式作业。

**第一步：在 Snellius 手动编码当前 13 个模型（3 个独立 job）**

```bash
bash scripts/encode_hotflip_rebuttal_queries_snellius.sh --dry_run
bash scripts/encode_hotflip_rebuttal_queries_snellius.sh
```

三个 job 的模型分组在该脚本顶部，可自行调整：

1. Harrier-27B。
2. Qwen3-8B、4B、0.6B、GTE、BGE reasoner。
3. Contriever、BGE-M3、Linq、ReasonIR、DIVER-4B、Harrier-0.6B、270M。

每个 job 在 1 张 H100 上顺序编码其模型的三个训练数据集（nq-train、hotpotqa、msmarco）；
180G 内存、16 cores、5 天。Harrier-27B query batch = 32，其余模型 = 64。
已有有效缓存会跳过，未完成任务可重提同一脚本。

**第二步：手动同步缓存到 MN5**

用户已在 Snellius 完成全部 42 份训练 query 缓存。把本项目的 `cache_rebuttal/query_embeddings/` 复制到 MN5 项目目录下的同一路径，保留其中的
`queries.npy` 和 `queries.json`。默认 MN5 项目路径为 `/gpfs/projects/ehpc695/hotflip_corpus_poisoning`。

**第三步：手动提交攻击**

**当前 13 个模型（含 Harrier-27B）的 HotFlip/AGGD 攻击统一在 MN5 提交。**
每种方法最多 234 个作业；旧 Snellius 攻击启动/sub 脚本已移除。Snellius 编码脚本仅保留供将来重建缓存时手动使用。

把新增文件连同现有仓库代码和 prompts 同步到 `/gpfs/projects/ehpc695/hotflip_corpus_poisoning` 后：

```bash
bash scripts_mn5/generate_hotflip_rebuttal.sh --dry_run
bash scripts_mn5/generate_hotflip_rebuttal.sh
```

AGGD 的对应命令为 `bash scripts_mn5/generate_aggd_rebuttal.sh`，默认 3000 步；HotFlip 默认 5000 步。
两种方法合计最多 468 个作业，受 300 作业上限限制，请分批选择模型/数据集，或等第一种方法队列减少后再提交另一种。

Account = `ehpc821`，QoS = `acc_ehpc`，1 张 GPU、20 cores、3 天。
Python = `/gpfs/projects/ehpc695/anaconda/envs/ir/bin/python`；HF_HOME = `/gpfs/projects/ehpc695/cache`，
HF_HUB_CACHE = HF_HOME。离线加载，需要 MN5 环境能加载对应 Harrier checkpoint。
候选前向 chunk = 100。Harrier-27B 在 Snellius 的限额测试支持 64 GB 显存容量足够，见
[显存报告](benchmarks/harrier_27b_memory_gcn135.md)；MN5 环境尚未实际执行验证。

编码与攻击由用户分别控制，攻击启动器不自动提交编码作业，不添加 Slurm 依赖。
每个 (model, dataset) 有 5 个 N50 单 seed 攻击作业和 1 个 N1+N10 的五 seed 作业；完成的配置会跳过。
启动器先安排 N50，最后 N1+N10；检查现有排队/运行作业加本次攻击作业数不超过 300。
可分批选模型、数据集：

```bash
bash scripts_mn5/generate_hotflip_rebuttal.sh --dry_run "qwen3_0.6B harrier-oss-v1_270M" "nq-train"
```

可用环境变量覆盖 `REPO`、`OUTPUT_ROOT`、`CACHE_ROOT`、`QUERY_EMBEDDING_ROOT`、`QUERY_BATCH_SIZE`、`CAND_CHUNK`、`MAX_JOBS`；
MN5 另可覆盖 `ACCOUNT`、`QOS`、`PYTHON`。启动器从自身位置确定仓库，sub 脚本有对应服务器的默认路径。

## 持久训练 query 缓存

两台服务器均默认保存在本项目中，不读取 LLM_Robust 的 test/dev query 表示：

```text
cache_rebuttal/query_embeddings/<dataset>/<model>/queries.npy
cache_rebuttal/query_embeddings/<dataset>/<model>/queries.json
```

`.npy` 为 float32 表示，`.json` 记录训练 query IDs、输入文本指纹、模型/prompt/pooling 配置、编码 batch 和版本。
每个训练 query ID 编码一次，所有 seeds 和 N 按 ID 取表示用于聚类和目标均值；一个 query 对应多个 gold 的权重不变。
使用内存映射读取矩阵；表示分 batch 写盘，不必把整份表示同时攒在内存里。

编码时用文件锁防止同一缓存被重复生成，缓存完成标记只在矩阵全部写完后发布。
已有有效缓存会跳过模型加载。中断时未完成的缓存不能被攻击使用，重提编码任务从头编码该 (model, dataset)。
完成的 query 缓存长期保留，攻击结束不会删除；与临时 cluster 目标缓存不同。数据或编码配置不匹配时明确报错，不覆盖旧缓存。
`cache_rebuttal/` 已加入 gitignore。

若只想生成指定模型、数据集的 query 缓存，也可以直接提交 sub 脚本，模型和数据集均可传空格分隔的列表：

```bash
sbatch scripts/encode_hotflip_rebuttal_queries_snellius_sub.sh "qwen3_0.6B harrier-oss-v1_270M" "nq-train hotpotqa"
```

编码阶段只在 Snellius 运行；完成后由用户同步缓存、提交后续攻击。

## 输出与续跑

输出目录：`output_attack/attacked_text/document/hotflip_rebuttal/<dataset>/`。
文件名示例：

```text
nq-train_train_attacked_documents_seed_1999_qwen3_0.6B_hotflip_rebuttal_N50_Len50_Iter5000_Pool100_InitTrue.json
```

条目 `"0"` … `"N-1"` 保存 `best_sim`、`best_adv_text`（token 字符串列表，沿用旧格式）、
`best_adv_passage_ids`、`tot`、`finish`、`it`（最佳文档出现的迭代，初始化为 -1）。
只有 `finish: true` 表示该 cluster 已完成全部迭代。

JSON 原子替换；最优文档改进时最多每分钟落盘一次，cluster 完成后立即保存。
重跑同一命令会跳过已完成 cluster，未完成的 cluster 从缓存的初始化重新运行，而非从中断的迭代继续。
每个 cluster 独立设随机种子。全部完成的配置不加载模型。

临时 cluster 目标缓存路径：MN5 `$REPO/cache_rebuttal/`；Snellius `/scratch-shared/yli4/hotflip_rebuttal_cache/`。
它保存 cluster 的均值目标和初始化，引用持久 query 缓存的 ID；设置不匹配会报错，该 (k, seed) 全部完成后删除。
小样本检查请使用独立的 `--query_embedding_root`、`--output_root` 与 `--cache_root`，编码和攻击使用相同的 `--max_queries`。

## 本轮验证

采用必要的回归检查和少量真实模型迭代，省去原计划全模型 clean 缓存对比、legacy 复现、吞吐调参。
验证结果见 `HANDOFF.md` 最新更新。正式生成与 MN5 上的执行由用户提交；短试跑不代表完整数据集的运行时间或峰值显存。
