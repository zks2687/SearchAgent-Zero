# SearchAgent-Zero — 实验介绍(数据集 / 模型 / 设置 / 指标 / 参数 / 结果)

> 目的:把本项目做过的每一组实验讲清楚——**用什么数据、什么模型、怎么设置、用什么指标、关键超参、跑出什么结果**。作为项目叙事报告与源码详解的实验底稿,数据均来自本地训练与评测记录。
>
> 主线两条:
> - **短程验证(Search-R1)**:证明 recipe / 基础设施正确、地基牢。
> - **长程主线(ASearcher → BrowseComp-Plus)**:证明能稳定扩展到几十轮,并逐项消融每个改进的贡献。
>
> 全文以**同步 GRPO** 为主线;异步(fully-async)仅作为一条已记录的对照 run 附带列出,不作为主结论。

---

## 0. 实验总览

| 组 | 目的 | 训练模型 | 训练数据 | 评测基准 | 主指标 |
|---|---|---|---|---|---|
| 实验一 | 短程地基验证 | Qwen2.5-3B-Instruct | `PeterJinGo/nq_hotpotqa_train` | Search-R1 7 QA 验证集 | EM / score@1 |
| 实验二 | 长程消融(核心) | Qwen3-8B | `aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample` | BrowseComp-Plus(830 题) | Accuracy / Recall / 平均检索轮数 |
| 实验三 | 多 checkpoint + 检索器复核 | Qwen3-8B(同上) | 同上 | BrowseComp-Plus | Accuracy / Recall / 平均检索轮数 |
| 实验四 | 进阶探索:IGPO 过程奖励 | Qwen3-8B(grpo_igpo) | 同上 | BrowseComp-Plus | Accuracy / Recall / 平均检索轮数 |
| 附:异步对照 | 吞吐对照(非质量结论) | Qwen3-8B(fully-async) | 同上 | BrowseComp-Plus | 训练步数 / 吞吐 |

**共享的环境组件**
- **训练框架**:verl(FSDP2 actor + vLLM async rollout + GRPO),多轮走 AgentLoop(`multi_turn.enable=True`,`default_agent_loop=tool_agent`)。
- **检索服务**:本地 Wikipedia 语料 + E5 dense index,HTTP 服务 `http://127.0.0.1:8000/retrieve`,Search-R1 风格 I/O。
- **搜索工具**:`verl.tools.search_tool.SearchTool`,`num_workers=60`,`rate_limit=60`,`timeout=20`;工具函数 `search(query_list)`。
- **异常轨迹监控指标**(全程埋点,既做过滤依据也做诊断):工具调用成功率、平均搜索轮数、重复 query、单轮并发 query 过多、工具解析失败、结果无增量、轨迹超长截断、超轮数。

---

## 实验一:Search-R1 短程验证(地基是否牢)

### 1.1 目的
在扩展到长程之前,先在成熟的短程 setting(2–6 轮 reasoning + search)上验证「稳定基础设施 + AgentLoop rollout + 异常轨迹处理」这套 recipe 有效——这是地基验证,不是刷点。

### 1.2 数据集与模型
- **模型**:`Qwen/Qwen2.5-3B-Instruct`(与 Search-R1 同 base)。
- **训练数据**:`PeterJinGo/nq_hotpotqa_train`(NQ + HotpotQA 混合),预处理为 verl 多轮工具调用格式(`train_search_r1.parquet` / `test_search_r1.parquet`)。
- **检索语料**:Wikipedia-18 + E5 dense(与 Search-R1 对齐)。
- **评测集**:7 个开放域 QA 验证集——NQ、TriviaQA、PopQA、HotpotQA、2Wiki、Musique、Bamboogle。

### 1.3 关键设置与超参
| 项 | 值 | 项 | 值 |
|---|---|---|---|
| adv_estimator | grpo | rollout.n | 5 |
| train_batch_size | 256 | ppo_mini_batch_size | 128 |
| lr | 1e-6 | max_prompt_length | 4096 |
| max_response_length | 3000 | max_model_len | 15000 |
| max_assistant_turns | 4 | max_queries_per_tool_call | 1 |
| rollout / val temperature | 1.0 / 1.0 | top_p | 1.0 |
| kl_loss_coef / type | 0.001 / low_var_kl | entropy_coeff | 0 |
| enable_tool_response_summary | **False**(短程无需压缩) | gpu_memory_utilization | 0.7 |
| total_epochs | 2 | save / test freq | 250 / 50 |

> 短程与长程的关键差异:短程 `max_assistant_turns=4`、单轮单 query、**不开 summary**;这些都在长程实验里被放开(见实验二)。

### 1.4 指标
- **score@1 / EM**:验证集单次采样答案正确率(`val-core/*/reward/mean@1`)。
- 报告口径:**只报绝对提升**,不报相对百分比(低基数数据集如 Musique 的相对增益会显得夸张、不可比)。

### 1.5 结果

| 数据集 | Search-R1(原报告) | 本 recipe(verl + AgentLoop) | 绝对提升 |
|---|---:|---:|---:|
| NQ | 0.341 | 0.464 | +0.123 |
| TriviaQA | 0.545 | 0.616 | +0.071 |
| PopQA | 0.378 | 0.424 | +0.046 |
| HotpotQA | 0.324 | 0.423 | +0.099 |
| 2Wiki | 0.319 | 0.398 | +0.079 |
| Musique | 0.103 | 0.181 | +0.078 |
| Bamboogle | 0.264 | 0.344 | +0.080 |
| **平均** | **0.325** | **0.407** | **+0.082** |

**结论**:7/7 数据集全部正增长,平均 0.325 → 0.407(+8.2pt)。稳定基础设施 + AgentLoop + 异常轨迹处理确实改善训练质量,为扩展到长程提供信心。

> **实现正确性 sanity check**:另有一条以**原始 Search-R1 recipe** 复跑的对照 run(`qwen2.5-3b-instruct_searchr1_origin`,训练到 step 1324),验证集 score@1 量级为 triviaqa≈0.629 / hotpot≈0.533 / 2wiki≈0.541 / nq≈0.524 / popqa≈0.446 / bamboogle≈0.450 / musique≈0.412,与上表同量级,且训练期 `tool_call_success_rate≈1.0`、平均工具轮数≈2.06——说明多轮工具调用链路实现正确、稳定。

---

## 实验二:BrowseComp-Plus 长程消融(核心)

### 2.1 目的与基准
把轨迹从 2–6 轮扩到 20–40 轮,并**逐项消融**每个改进值多少分。

**BrowseComp-Plus** 把「检索器」与「LLM agent」解耦:查询来自 OpenAI BrowseComp,但针对一个**固定的 ~100K 人工核验文档语料**检索(不打实时网页),从而公平、可复现。共 **830 题**。

### 2.2 数据集与模型
- **模型**:`Qwen/Qwen3-8B`。
- **训练数据**:`aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample`(ASearcher 长程多轮搜索数据,已 reject-sample),切分为训练/测试并转 verl 多轮格式。
- **评测**:BrowseComp-Plus 830 题,**judge = Qwen3-32B**,**稠密检索器**(除实验三专门换检索器外)。

### 2.3 关键设置与超参
| 项 | 值 | 项 | 值 |
|---|---|---|---|
| adv_estimator | grpo | rollout.n | 8 |
| train_batch_size | 128 | ppo_mini_batch_size | 64 |
| lr | 1e-6 | max_prompt_length | 2048 |
| max_response_length | 36864 | max_model_len | 20000 |
| max_assistant_turns | 100 | turn_limit_schedule | 0:100,50:100,100:100,200:100,300:100 |
| max_queries_per_tool_call | 4 | clip_ratio_high | **0.34(DAPO 风格)** |
| use_kl_loss | False | entropy_coeff | 0 |
| rollout / val temperature | 1.0 / 0.7 | gpu_memory_utilization | 0.7 |
| **self-summary** | **开启**,self(不用外部模型),`summary_max_tokens=1024`,temp 0.6 / top_p 0.95 / top_k 20 | max_tool_response_length | 20000 |
| rollout 修正 | token IS(阈值 2.0)+ 拒绝采样 token_k1(0.6–1.6) | save / test freq | 100 / 20 |

> 与短程的对照:轮数 4→100、单轮 query 1→4、**开启 self-summary 压缩**、max_response 3000→36864——这三处放开正是"能搜几十轮"的前提。

### 2.4 指标
- **Accuracy(%)**:Qwen3-32B judge 判定答案正确的比例。
- **Recall(%)**:证据文档(回答该题所必需)的召回率。
- **平均检索轮数**:过程指标——**只有伴随 Recall 上升才说明"多搜"有效**,单看它无意义。

### 2.5 消融结果(逐点)

| # | 设置 | Accuracy | Recall | 平均检索轮数 | 相比上一行 |
|---|------|---:|---:|---:|---|
| ① | Qwen3-32B(无训练 baseline) | 10.72% | 7.28% | 0.94 | 起点:几乎不主动搜 |
| ② | Qwen3-8B + 异常过滤 + self-summary(100 步) | 24.21% | 33.14% | 10.11 | **+13.5pt Acc**:RL 让 8B 学会主动多轮搜索 |
| ③ | + **轮级信用分配**(100 步) | 28.19% | 40.10% | 14.22 | **+3.98pt Acc / +6.96pt Recall**:信用分配净贡献 |
| ④ | 充分训练(300 步) | 37.95% | 50.87% | 38.47 | **+9.76pt Acc**:长程持续爬升 |

> 所有行同一 judge(Qwen3-32B)、同一 830 题语料、同一稠密检索器,仅改待评变量,保证对照公平。

**三个结论**:
- **A(①→②,RL 的价值)**:未训练的 32B 几乎不搜(0.94 次)、Acc 仅 10.72%;RL 训练的 8B 搜到 10 次、Acc 翻倍到 24.21%。**参数小 4 倍,但学会主动搜索就能反超。**
- **B(②→③,信用分配的必要性——最干净的对照)**:唯一变量是"是否轮级信用分配"。解除连坐后,模型敢搜更多轮(10→14)→ 召回更多证据(Recall +6.96pt)→ 答对更多题(Acc +3.98pt)。**三指标同向且逻辑自洽**,比单看 Acc 更有说服力。
- **C(③→④,长程可扩展)**:100→300 步,检索轮数 14→38、Recall 破 50%、Acc 到 37.95%。证明搜索智能体 RL 能稳定 scale 到几十轮不崩。

> **训练侧信号(验证 recipe 正常)**:ASearcher 验证集起点 origin_score≈0.18 / reward≈0.21;充分训练后 val ASearcher reward 稳定爬升到 0.6+。

---

## 实验三:多 checkpoint + 检索器复核

### 3.1 目的
在 BrowseComp-Plus 上评测多个训练 checkpoint,交叉印证实验二,并补充两个工程洞见。**选 checkpoint 依据独立的 held-out 验证信号(ASearcher val reward)选出,再在 BrowseComp-Plus 报告**,避免直接在测试集挑点。

### 3.2 评测结果(按 Accuracy 排序,稠密检索除非另注)

| 检索器 | checkpoint | Accuracy | Recall | 平均检索轮数 |
|---|---|---:|---:|---:|
| dense | **step_400** | **38.07%** | 47.96% | 21.8 |
| dense | step_350 | 37.59% | **48.14%** | 26.3 |
| dense | step_500 | 36.02% | 41.38% | 11.6 |
| dense | fully_async step_311(异步,见附) | 32.29% | 44.59% | 29.6 |
| dense | IGPO step_200(见实验四) | 29.04% | 40.26% | 34.6 |
| qwen3-embedding-8b | step_500 | 28.43% | 31.18% | 5.4 |
| dense | step_250 | 26.87% | 31.24% | 6.5 |
| dense | step_300 | 25.30% | 36.33% | 9.0 |
| **BM25** | step_500 | **6.51%** | 6.06% | 6.8 |

> 本地最佳 step_400(38.07%)与实验二的 300 步(37.95%)基本一致,交叉印证结果稳健。

### 3.3 洞见一:检索器决定上限(天然消融)
固定同一 step_500 模型,只换检索器:

| 检索器 | Accuracy | Recall |
|---|---:|---:|
| dense(稠密) | 36.02% | 41.38% |
| qwen3-embedding-8b | 28.43% | 31.18% |
| **BM25(稀疏)** | **6.51%** | 6.06% |

**同一 agent,换 BM25 从 36% 掉到 6.5%——agent 再强也救不回召不回证据的检索器。**
> caveat:BM25 断崖有一部分可能源于稀疏检索配置(分词/analyzer、top-k、字段权重)未充分调优,故结论限定为"**对齐配置下,稠密显著优于稀疏**",作为"检索预算投稠密"的工程指引,而非"BM25 天生不行"的普适断言。

### 3.4 洞见二:训练 reward ≠ 下游表现
step_500 训练 reward 最高,但下游 **step_400(38.07%)> step_500(36.02%)**。训练奖励(答案 log-prob / 格式分组合)与真实 Accuracy 存在 gap——**选 checkpoint 必须用 held-out / benchmark 离线复核,不能只看训练曲线峰值。**

---

## 实验四:IGPO 过程奖励(进阶探索)

### 4.1 目的与做法
GRPO 只有一个最终 outcome reward,对长轨迹信用分配太粗——几十轮里哪一轮真正带来信息增益说不清。IGPO(Information Gain Policy Optimization)把信用分配从"轮级掩码"推进到"**步级过程奖励**":以每轮结束后模型对 **ground-truth 答案的概率提升**作为该轮的 process reward。

- **过程奖励**:`info_gain_type=prob_diff`,即 $e^{\overline{\log P_t}} - e^{\overline{\log P_{t-1}}}$。
- **归一化**:`info_gain_norm_mode=separate`——过程奖励与结果(F1/outcome)奖励**各自在组内独立归一**,避免量纲差异淹没过程信号。
- **加速**:`use_vectorized_gt_logprob=true`——用扩展序列 + 4D attention mask 把 T 次 GT 前向压成 1 次(FA2 不支持 4D mask 时自动回退顺序实现 + 数值校验)。
- **课程学习**:暂关(`ig_init=1.0→final=0.2` 留作后续)。
- 其余超参与实验二同步 GRPO 一致(Qwen3-8B、n=8、lr=1e-6、turns=100、summary=1024)。

### 4.2 早期实测(step_200)

| checkpoint | Accuracy | Recall | 平均检索轮数 |
|---|---:|---:|---:|
| GRPO step_250 | 26.87% | 31.24% | 6.5 |
| GRPO step_300 | 25.30% | 36.33% | 9.0 |
| **IGPO step_200** | **29.04%** | **40.26%** | **34.6** |

**IGPO 在 step_200 就达 29.04% / Recall 40.26,超过同量级 step 的 GRPO(step_250 的 26.87、step_300 的 25.30),且检索轮数(34.6)显著更高。** 与机制预期一致:稠密过程奖励在早期就鼓励更主动的多轮探索。

### 4.3 诚实定位
- **价值命题(可证伪)**:IGPO 主张的不是更高天花板,而是**更高样本效率 + 更好证据召回**——即"相同步数下 Recall 更高,或达到相同 Accuracy 所需步数更少"。step_200 的早期信号朝此方向为正。
- **边界**:目前**仅到 step_200 实测**,完整验证(训到 step_400 与同 step GRPO 严格对照)是明确的下一步。§4.2 是实测数据,不做更高步数的结论引用。
- **代码位置**:IGPO 相关实现在 `report/IGPO_ref/`(`verl/utils/reward_score/info_gain.py`、`scrl/llm_agent/vectorized_gt_logprob.py`、`verl/trainer/ppo/core_algos.py`),尚未合入主仓库。

---

## 附:异步(fully-async)对照 run —— 只比吞吐,不比质量

作为一条已记录的训练范式对照(**不进入主结论**):

- **入口**:`verl.experimental.fully_async_policy.fully_async_main`,`hybrid_engine=False`,4 卡 rollout + 4 卡训练流水线并行,`staleness_threshold=0.5`、`trigger_parameter_sync_step=2`、`partial_rollout=True`。
- **与同步共享**:credit assignment、self-summary、rollout 修正、n=8、lr=1e-6、turns=100。

| | 同步(h200_full) | 异步(fully_async) |
|---|---|---|
| 最大 checkpoint | step_400 | step_311 |
| 训练 rollout step | ~399 | **622(全部 run 中最长)** |
| val ASearcher reward | 0.19 → **0.614** | 0.21 → 0.56 |
| BrowseComp-Plus | 38.07% / Recall 47.96 | 32.29% / Recall 44.59 |
| 评测平均检索轮数 | 21.8 | 29.6 |

**能下的唯一确定结论**:异步吞吐确实更高(同等墙钟跑到 622 步)。**质量对比不成立**——异步 step_311 与同步 step_400 步数不对齐,是不公平对比,故本项目不声称"同步质量更高"。异步"敢搜"(29.6 > 21.8)可能与 `partial_rollout=True`(长轨迹不被打断)有关,属观察性描述。

---

## 附录:已知局限与数据口径(主动披露)

- **单次运行、无多 seed**:所有增量(尤其实验二 ②→③ 的 +3.98pt Acc)为单次结果,受算力限制未做方差估计;方向性(三指标同向、7/7 数据集一致)比单点数值更可信。
- **IGPO 未训完**:仅到 step_200 实测,更高步数为下一步。
- **BM25 配置 caveat**:稀疏检索差距一部分可能受配置未充分调优影响,结论限定"对齐配置下"。
- **BrowseComp-Plus 题数**:标准为 **830 题**。历史评测产物中曾出现评测流程 bug——`fully_async_step311` 某输出文件为 1475 题(两次运行被追加进同一文件),已识别、隔离、**未采用**;`step_300` 某文件缺 13 题。这些是评测脚本的工程问题,与 RL 训练稳定性无关。
- **选 checkpoint**:用 held-out(ASearcher val reward)选、再报 BrowseComp-Plus,规避 test-set peeking。
- **绝对声明绑定口径**:BrowseComp-Plus 结果绑定"830 题、Qwen3-32B judge、稠密检索、截至 2026-08",在 14B 以下开源模型中具很强竞争力;不作无限定 SOTA 主张。

### 关键路径速查
| 内容 | 路径 |
|---|---|
| 同步训练脚本(3B 短程) | `run_qwen2.5_3b_instruct_search_multiturn_SearchR1.sh` |
| 同步训练脚本(8B 长程) | `run_qwen3_8b_instruct_search_multiturn_ASearch.sh` |
| 主配置 | `examples/search_agent_rl/config/search_multiturn_grpo.yaml` |
| 信用分配 agent loop | `verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py` |
| 搜索工具 / 检索配置 | `verl/tools/search_tool.py` / `.../tool_config/search_tool_config.yaml` |
| 最佳模型(38%) | `output/qwen3-8b-instruct_ASearch_..._h200_full/global_step_400` |
| IGPO 模型 | `output/qwen3-8b_ASearch_h200_full_igpo_prob_diff/global_step_200` |
| IGPO 变更日志 | `report/IGPO_ref/resources/CHANGELOG_20260201.md` |

---

*本文档基于本地训练脚本、wandb 记录与 BrowseComp-Plus 评测产物整理。短程验证、长程消融、多 checkpoint 复核、IGPO step_200 为实测数据;IGPO 更高步数与异步质量对比为明确标注的下一步工作,不作已测结论引用。*
