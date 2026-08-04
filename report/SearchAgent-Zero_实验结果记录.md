# SearchAgent-Zero — 实验结果记录(以本地训练 + 评测目录为准)

> 本文档盘点两处本地目录里**实际跑过、有磁盘产物**的实验,逐条记录设置与结果:
> - 训练:`/mmu_mllm_hdd_2/zhoujinchang/SearchAgent-Zero`(`output/` checkpoint + `wandb/` 训练指标)
> - 评测:`/mmu_mllm_hdd_2/zhoujinchang/BrowseComp-Plus`(`evals/<retriever>/<model>/evaluation_summary.json`)
>
> **数据来源分两类,已明确标注:**
> - **训练侧指标(reward / 轮数 / 异常轨迹 / val reward)**:来自 `wandb/run-*/files/wandb-summary.json`。
> - **BrowseComp-Plus 的 Accuracy / Recall / 平均检索次数**:来自 `BrowseComp-Plus/evals/.../evaluation_summary.json`,judge=Qwen3-32B(`model_info.judge_model` 确认),每目录 830 题(`query_*_eval.json` 计数确认)。**本次已逐个 summary 读取核对,以这些文件为准。**
>
> ⚠️ **两处需要你拍板的口径问题(见 §7),先看数据:**
> 1. **IGPO 磁盘评测目前只到 step_200(29.04% / 40.26%)**;step_400 = **39.27% / 54.21% / 38.73 次** 已由作者确认,但评测产物尚未归档到 `evals/dense/`,建议补上以便复核。
> 2. 之前文档写的"长程主线 300 步 = 37.95% / 50.87% / 38.47 次"(官方博客口径)**在本地评测目录里找不到对应文件**;本地能复核的最佳是 **step_400 = 38.07% / 47.96% / 21.8 次** 与 **step_350 = 37.59% / 48.14% / 26.3 次**。

---

## 0. 磁盘上真实存在的 run(output/ + wandb/ 交叉确认)

| # | experiment_name(output/ 目录) | adv_estimator | 磁盘 checkpoint | 对应 wandb run | 用途 |
|---|---|---|---|---|---|
| E1 | `qwen2.5-3b-instruct_searchr1_origin` | grpo | step_250/500/750/1000/1250/**1324** | run-20260605_165524-k020es6u | Search-R1 短程复现 |
| E1' | `qwen2.5-3b-instruct_searchr1`(eval) | grpo | — | run-20260709_160524-k0s2nryd | Search-R1 改进版评测 |
| E2 | `qwen3-8b-instruct_ASearch_..._ca_h200_full` | grpo | step_50→**400**(每 50 一存) | (h200_full 训练 run) | 长程主线(最佳模型) |
| E3 | `qwen3-8b-instruct_ASearch_..._ca_h200` | grpo | step_100/200/206/250/500 | run-20260606 / 0611 系列 | 长程多 checkpoint |
| E4 | `qwen3-8b_ASearch_h200_full_igpo_prob_diff` | **grpo_igpo** | step_50/100/150/**200**(磁盘止于 200) | run-20260717 系列 | IGPO 过程奖励 |
| E5 | `qinstruct_ASearch_..._ca_fully_async` | grpo | step_100/200/300/311 | (异步 run) | 异步吞吐对照 |

> ⚠️ **诚实说明**:E4(IGPO)磁盘上的 checkpoint **只到 step_200**;step_400 的模型权重未在本目录保留,其 39.5% / 54.2% 为作者本人评测所得的最终数值(见 §4)。E2 磁盘 checkpoint 到 step_400,与官方口径的"300 步 37.95%"是同一条长程主线的不同 step。

---

## 1. E1 — Search-R1 短程复现(Qwen2.5-3B)

**设置**(wandb config 确认):`adv_estimator=grpo`,base=Qwen2.5-3B-Instruct,训练数据 `PeterJinGo/nq_hotpotqa_train`,Wikipedia-18 + E5 dense 检索,`max_assistant_turns≈4`,不开 summary。

**训练侧指标(run-...k020es6u,step_1324,wandb 实测)**
- `critic/rewards/mean = 0.5806`;`reward_extra/origin_score/mean = 0.4818`;`format_score/mean = 0.0988`;`efficiency_score/mean = 0.494`
- `turn/tool_call_success_rate/mean = 1.0`;`turn/tool_call_turn/mean = 2.06`(max=3)
- 异常轨迹几乎为 0(`too_many_turn ≈ 6.5%`,其余=0)→ 多轮工具链路实现正确、稳定

**验证集 score@1(wandb `val-core/...reward/mean@1`)**

| 数据集 | origin run(step_1324) | 改进版 eval(run-...k0s2nryd) |
|---|---:|---:|
| NQ | 0.5238 | 0.5240 |
| TriviaQA | 0.6294 | 0.6358 |
| PopQA | 0.4460 | 0.4940 |
| HotpotQA | 0.5330 | 0.5498 |
| 2Wiki | 0.5414 | 0.5332 |
| Musique | 0.4116 | 0.4136 |
| Bamboogle | 0.4504 | 0.4992 |

> 这是**训练/验证阶段的 score@1**(带 efficiency/format 分量),与对外报告口径的"纯 EM 平均 0.325→0.407"是不同指标:对外 bullet 用的是 Search-R1 论文对齐的 **EM**(见 `实验介绍.md` 实验一表),此处 wandb 数是训练监控信号,量级一致、可交叉印证链路正确。

---

## 2. E2 / E3 — 长程主线(Qwen3-8B,BrowseComp-Plus)

**设置**:`adv_estimator=grpo`,base=Qwen3-8B,训练数据 `aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample`,`max_assistant_turns=100`,`max_response_length=36864`,`clip_ratio_high=0.34`,无 KL loss,**开 self-summary**(`summary_max_tokens=1024`,用模型自身),`rollout.n=8`,`lr=1e-6`。

### 2.1 本地评测目录里**逐个 summary 核对**的结果(judge=Qwen3-32B,830 题,dense 检索)

> 下表每一行都对应 `BrowseComp-Plus/evals/dense/<model>/evaluation_summary.json` 的实际数值,可复核。

| checkpoint(评测目录) | Accuracy | Recall | 平均检索次数 | 评测日期 |
|---|---:|---:|---:|---|
| **step_400**(`..._ca_h200_full_global_step_400`) | **38.07%** | 47.96% | 21.78 | 2026-07-15 |
| step_350(`h200_full_global_step_350`) | 37.59% | **48.14%** | 26.30 | 2026-07-14 |
| step_500(`..._ca_h200_global_step_500`) | 36.02% | 41.38% | 11.59 | 2026-07-15 |
| step_300(`..._ca_h200_full_global_step_300`) | 25.30% | 36.33% | 8.99 | 2026-07-14 |
| step_250(`searchagent_zero_step250`) | 26.87% | 31.24% | 6.54 | 2026-07-10 |

**本地可复核的最佳:step_400 = 38.07% Acc / 47.96% Recall / 21.78 次。**

### 2.2 官方博客口径(对外报告用)

官方博客 / README 报告的长程最佳是"**300 步 = 37.95% / 50.87% / 38.47 次**",以及信用分配 100 步消融(24.21→28.19 / 33.14→40.10 / 10.11→14.22)、baseline Qwen3-32B(10.72 / 7.28 / 0.94)。

> ⚠️ **口径差异(重要)**:官方那组"37.95 / 50.87 / 38.47"与"100 步消融""32B baseline"**在本地这份评测目录里没有对应的 `evaluation_summary.json`**——它们要么来自另一台机器/另一次评测未同步到此目录,要么口径(step 数、题目子集)与本地不同。**本地能自证的最佳是 step_400 的 38.07 / 47.96 / 21.78。** 对外用官方数字没问题(那是发布口径),但**面试若被要求"现场复核",你手上能打开的文件是 38.07 那一组**——两个数别混用,尤其平均检索次数官方 38.47 vs 本地 step_400 的 21.78 差异较大(step_350 的 26.3 更接近但仍不到 38)。

**检索器天然消融(固定 step_500,只换检索器)**:dense 36.02% / qwen3-embedding-8b 28.43% / BM25 6.51% → 检索器决定上限。三个数在本地评测目录均可复核(`evals/dense`、`evals/qwen3emb8b`、`evals/bm25` 各 830 题)。

**训练侧(wandb)**:ASearcher val reward 起点 ≈0.217,充分训练后爬升到 0.6+;异常轨迹指标随训练下降。

---

## 3. E5 — 异步(fully-async)对照(只比吞吐)

**设置**:`fully_async_policy`,`hybrid_engine=False`,`staleness_threshold=0.5`,`partial_rollout=True`;其余(信用分配 / self-summary / n=8 / lr=1e-6 / turns=100)与同步共享。

**本地评测目录里有三份异步 step_311 产物(数值略有差异,注意区分)**:

| 评测目录 | Accuracy | Recall | 平均检索次数 | 题数 | 备注 |
|---|---:|---:|---:|---:|---|
| `..._fully_async_global_step_311` | 32.29% | 44.59% | 29.63 | 830 | 主口径 |
| `..._fully_async_global_step_311_1` | 31.19% | 44.94% | 29.28 | **1475** | ⚠️ 重复追加,题数异常,**不采用** |
| `searchagent_fully_async_311` | 23.01% | 31.97% | 8.73 | 830 | 另一次评测,偏低 |

| | 同步(step_400) | 异步(step_311,主口径) |
|---|---|---|
| BrowseComp-Plus | 38.07% / Recall 47.96 / 21.78 次 | 32.29% / Recall 44.59 / 29.63 次 |

> 唯一确定结论:异步吞吐更高(同等墙钟跑更多 step)。质量对比**不成立**(step 不对齐),不声称同步质量更高。异步同名产物有三份、数值不一,引用务必指明是哪一个目录。

---

## 4. E4 — IGPO 过程奖励(Qwen3-8B,`adv_estimator=grpo_igpo`)

**设置**(wandb config 确认 `adv_estimator=grpo_igpo`):
- 过程奖励 `info_gain_type=prob_diff`,即 $r_t = e^{\overline{\log P_t}} - e^{\overline{\log P_{t-1}}}$(每轮结束后模型对 ground-truth 答案的置信度提升)。
- 归一化 `info_gain_norm_mode=separate`(过程 / 结果奖励各自组内独立归一)。
- 加速 `use_vectorized_gt_logprob=true`:扩展序列 + 4D attention mask 把 T 次 GT 前向压成 1 次。
- 其余超参与 E2 同步 GRPO 一致(n=8、lr=1e-6、turns=100、summary=1024)。

**BrowseComp-Plus 结果(dense 检索)**

| checkpoint | Accuracy | Recall | 平均检索次数 | 来源 |
|---|---:|---:|---:|---|
| GRPO step_250(对照) | 26.87% | 31.24% | 6.54 | ✅ 本地评测目录可复核 |
| GRPO step_300(对照) | 25.30% | 36.33% | 8.99 | ✅ 本地评测目录可复核 |
| **IGPO step_200** | **29.04%** | **40.26%** | **34.59** | ✅ 本地评测目录可复核(`igpo_prob_diff_global_step_200`,2026-07-20) |
| **IGPO step_400** | **39.27%** | **54.21%** | 38.73 | ⚠️ **本地评测目录暂无此产物**,为作者本人评测的最终数值 |

**结论(本地可自证的部分)**:IGPO 在 **step_200** 即达 29.04% / Recall 40.26 / 34.6 次,超过同量级纯 GRPO(step_250 的 26.87、step_300 的 25.30),且检索次数(34.6)显著更高——**这一条本地评测文件完全可复核**,是 IGPO"早期样本效率更高"最硬的证据。训到 **step_400** 达 **39.27% / 54.21% / 38.73 次**:Recall 明显高于同步主线 step_400 的 47.96,与"IGPO 主张更好证据召回"一致。step_400 评测产物建议补进 `evals/dense/` 以便复核(见下方局限)。

---

## 5. 已知局限(主动披露)

- **IGPO step_400 暂无磁盘评测产物**:`BrowseComp-Plus/evals/dense/` 下 IGPO 只有 `..._global_step_200`,尚未见 step_400 目录;`output/` 下 IGPO checkpoint 也止于 step_200。简历里的 39.27% / 54.21% / 38.73 次为作者本人评测所得,**建议把该次 `evaluation_summary.json` 归档到 `evals/dense/`**,以便和其他行一样可复核。本地当前能自证的 IGPO 铁证是 step_200 = 29.04% / 40.26%。
- **长程主线 37.95 vs 38.07 口径差**:官方博客的"300 步 37.95 / 50.87 / 38.47"在本地评测目录无对应文件;本地最佳可复核为 step_400 = 38.07 / 47.96 / 21.78。平均检索次数两者差距明显(38.47 vs 21.78),不要在同一句里混用。
- **单次运行、无多 seed**:所有增量为单次结果,未做方差估计;方向性(三指标同向、7/7 数据集一致)比单点更可信。
- **BM25 配置 caveat**:稀疏检索差距一部分可能受配置未充分调优影响,结论限定"对齐配置下,稠密显著优于稀疏"。
- **BrowseComp-Plus 题数**:标准 830 题;异步 `step_311_1` 目录为 1475 题(重复追加)已识别、不采用。
- **选 checkpoint**:用 held-out(ASearcher val reward)选、再报 BrowseComp-Plus,规避 test-set peeking。

---

## 6. 关键路径速查

| 内容 | 路径 |
|---|---|
| 训练目录 | `/mmu_mllm_hdd_2/zhoujinchang/SearchAgent-Zero` |
| 评测目录 | `/mmu_mllm_hdd_2/zhoujinchang/BrowseComp-Plus/evals/<retriever>/<model>/evaluation_summary.json` |
| Search-R1 origin run | `output/qwen2.5-3b-instruct_searchr1_origin/global_step_1324` |
| 长程最佳模型 | `output/qwen3-8b-instruct_ASearch_..._ca_h200_full/global_step_400` |
| 长程最佳评测 | `evals/dense/qwen3-8b-instruct_ASearch_..._ca_h200_full_global_step_400/` |
| IGPO 模型 / 评测 | `output/qwen3-8b_ASearch_h200_full_igpo_prob_diff/global_step_200`;`evals/dense/qwen3-8b_ASearch_h200_full_igpo_prob_diff_global_step_200/` |
| BM25 / qwen3emb 评测 | `evals/bm25/searchagent_zero_step500/`;`evals/qwen3emb8b/searchagent_zero_step500/` |
| wandb 记录 | `wandb/run-*/files/wandb-summary.json` |

---

*本文档基于本地 `output/`、`wandb/`、`BrowseComp-Plus/evals/` 磁盘快照整理,BrowseComp-Plus 数值逐个 `evaluation_summary.json` 核对(judge=Qwen3-32B,830 题)。已在 §7/§5 明确标注两处与官方博客口径的差异,以及 IGPO step_400 缺磁盘评测产物的问题。*

---

## 7. 需要你拍板的两处口径(汇总)

1. **IGPO step_400(39.27% / 54.21% / 38.73 次)** — 数值已确认,但本地评测目录目前只有 step_200(29.04% / 40.26%)。**建议把 step_400 的 `evaluation_summary.json` 补进 `evals/dense/`**,让它和其他行一样可现场复核;补上后简历数字即完全自证。
2. **长程主线 37.95 vs 本地 38.07** — 对外用官方 37.95/50.87/38.47(发布口径),但心里清楚本地可复核的是 38.07/47.96/21.78;两组数不要混用,尤其平均检索次数。
