# AGGD：LLM dense encoder 适配

入口：`aggd_attack_rebuttal.py`；候选搜索：`model/aggd.py`。

算法参考 [JinyanSu1/AGGD](https://github.com/JinyanSu1/AGGD)，固定核对的提交为
[`aa0e4fd29a99b2f73361b28cf6644ab2382e19a6`](https://github.com/JinyanSu1/AGGD/commit/aa0e4fd29a99b2f73361b28cf6644ab2382e19a6)，
具体为 `utils/AGGD.py::AGGD_candidate` 和 `main.py` 的 depth 更新逻辑。这里按算法重新实现，未引入官方模型加载器或环境依赖。

## 搜索方式与移植范围

HotFlip 每轮随机选一个位置，从该位置梯度产生候选。AGGD 每轮对全部正文位置计算梯度与 embedding 的点积，
先排列“每个位置的第 1 名 token”，再排列“每个位置的第 2 名 token”，依此类推。
`depth=d` 取该序列的第 `(d-1)*num_cand` 到 `d*num_cand` 个位置/token 对。
这不是对所有位置/token 的分数做一次全局 top-k；每个候选仍然只修改一个 token。

与官方一致：只有验证集出现新的最佳文档时，depth 才重置为 1；否则递增。
额外处理：排除没有 tokenizer/embedding 对应关系的 token，候选池完全耗尽后从 depth=1 重新开始，避免 topk 越界。
候选前向按 `--cand_chunk` 分块，正式配置统一为 100，固定 prompt 与 BOS/EOS 不参与修改。

**`aggd_rebuttal` 是适配当前实验设置的版本，不是官方 BERT 实验的完整复现。**
保留的是 AGGD 候选顺序和自适应 depth；多数设置与本项目 `hotflip_rebuttal` 一致，迭代预算单独指定：

- 共享同一份持久训练 query embedding、k-means 和 64-query 均值目标。
- Gold 初始化；50 个可修改 token、**3000 次迭代**、每轮 100 个候选（用户 2026-11-02 决定）；HotFlip 仍为 5000 次。
- 按平均相似度选择更新和验证最优文档。官方代码还有 query 与 gold 的逐条比较及 ASR gate；本适配版不使用该 gate。
- 官方运行脚本采用 30 token、30 候选、2000 次迭代和 mask 初始化，不能把本项目结果直接称为该设置的复现。
- 编码规则、固定特殊 token、冻结权重及 embedding-output 梯度均复用 `RebuttalEncoder`。当前默认使用 13 个模型，已停用 DIVER-0.6B。

## 手动运行与提交

先按 `docs/HOTFLIP_REBUTTAL.md` 在 Snellius 编码训练 query，并手动同步到 MN5。
HotFlip 和 AGGD 共用 `cache_rebuttal/query_embeddings/`，无需重新编码。编码与攻击仍由用户分开提交，无自动依赖。

```bash
# MN5：当前 13 个模型（含 Harrier-27B，不含 DIVER-0.6B），234 个攻击 job
bash scripts_mn5/generate_aggd_rebuttal.sh --dry_run
bash scripts_mn5/generate_aggd_rebuttal.sh
```

用户已完成 Snellius 的 query 编码；同步缓存和代码后，所有攻击均在 MN5 手动提交。
旧 Snellius 攻击脚本已移除。HotFlip 另有 234 个作业，两种方法需按 300 个作业上限分批提交。

主程序也可直接在计算作业内运行（示例为正式超参数）：

```bash
python aggd_attack_rebuttal.py --model qwen3_0.6B --attack_dataset nq-train \
    --ks 1 10 --seeds 1999 5 27 2016 2026 --num_iter 3000 \
    --query_embedding_root cache_rebuttal/query_embeddings \
    --cache_root cache_aggd_rebuttal --cand_chunk 100
```

## 输出与缓存

新结果单独保存，不写入旧 HotFlip 目录：

```text
output_attack/attacked_text/document/aggd_rebuttal/<dataset>/
<dataset>_train_attacked_documents_seed_<seed>_<model>_aggd_rebuttal_N<k>_Len50_Iter3000_Pool100_InitTrue.json
```

条目字段与 `hotflip_rebuttal` 相同：`best_sim`、`best_adv_text`（token 列表）、`best_adv_passage_ids`（只含可修改正文）、
`tot`、`finish`、`it`。共享原子 JSON 保存、完成 cluster 跳过和未完成 cluster 重跑逻辑。

持久 query 缓存共享；临时 cluster 缓存默认单独存放在 MN5 `$REPO/cache_aggd_rebuttal/`，
避免两个方法同时运行时相互清理临时文件。历史 Snellius 试跑缓存仍保留，不会自动搬迁或删除。
`QUERY_EMBEDDING_ROOT`、`CACHE_ROOT`、`OUTPUT_ROOT`、`CAND_CHUNK` 等覆盖方式与 HotFlip 脚本相同。

LLM_Robust 评估入口尚未修改；之后需要同时接入两个方法的新文档路径与编码规则。
