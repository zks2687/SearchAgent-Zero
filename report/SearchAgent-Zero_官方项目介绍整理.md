# SearchAgent-Zero 官方项目介绍（整理版）

> 本文整理自作者官方博客（知乎），是项目的**权威一手口径**。用于统一 report 下其他文档的表述与数字。
> 项目地址：https://github.com/NLPJCL/SearchAgent-Zero
> BrowseComp-Plus 榜单：https://huggingface.co/spaces/Tevatron/BrowseComp-Plus
>
> ⚠️ 口径提示：本文数字以官方博客为准（Search-R1 平均 **0.4085**；BrowseComp-Plus **37.95% Acc / 50.87% Recall / 38.47 次**）。官方博客**未提及 IGPO**——IGPO 属于本仓库后续加入的探索方向，不在官方发布口径内。

---

## 一、TL;DR（太长不看）

基于 verl 开源了一套 search agent 的 RL 训练框架。

**动机**：开源一个能在工业界训练 search agent 的 RL 框架，可 scale up 到多轮搜索、多种工具、多模态；同时让个人开发者与研究者也能**低成本**首次训练可扩展到多轮的 Search Agent——**不依赖 Google Search、Jina Reader 等外部商业服务，纯 wiki 搜索**。

**主要做了几件事**：
1. 基于 verl 原生 AgentLoop 支持多轮 search agent rollout；
2. 支持 Search-R1 与 ASearch 两套可复现训练 recipe；
3. 加入异常轨迹监控与过滤（重复搜索、工具调用错误、并发 query 过多等）；
4. 提出**带信用分配的异常轨迹过滤**，避免误伤前面正常的搜索轮次；
5. 支持 self-summary / external-summary，让模型在有限上下文内搜索更多轮；
6. 支持同步与异步 RL 训练；
7. 增加 TIS 和 MIS 等策略，缓解训推不一致。

**结果**：
- Search-R1 相同设置下，平均分从 **0.325 → 约 0.4085（相对提升约 25.3%）**；
- BrowseComp-Plus 上，SearchAgent-Zero-8B 纯 RL 训练，300 step 达 **37.95% Accuracy / 50.87% Recall**，为 **14B 以下模型 SOTA**，测试集平均搜索 **38.47 次**。

项目尚处早期，欢迎 Search Agent / Agent RL / RAG / 多轮搜索方向的交流与 star / issue / PR。

---

## 二、背景：为什么开源 SearchAgent-Zero

搜索与大模型的结合正从传统 RAG 演进到 **Agent Search**——模型围绕复杂问题多轮搜索、阅读、总结与推理。RL 对 Search Agent 至关重要：它不仅决定模型**是否搜索**，还决定模型能否在多轮交互中**持续提出有效 query、过滤无效信息、完成复杂问题求解**。

但训练 Search Agent RL 仍缺一个**趁手、稳定、可扩展**的开源框架。

### 现有框架的问题

verl、slime 等已支持一定程度的 Agent RL，也有专门的 Agent RL 框架（rllm 等），但大多为通用性做了取舍，**没有针对 Search Agent RL 场景深度优化**。训练真正能多轮搜索的 Search Agent，不是把搜索工具接上 RL 就行，需要一系列面向搜索场景的专门设计：

- **训练过程内部指标监控**：串行/并行搜索轮数、异常轨迹比例、环境稳定性、工具调用成功率等——决定训练是否真稳定、模型是否真学会有效搜索。
- **异常轨迹过滤机制**：重复搜索、一次生成多个 query、工具调用格式错误等；不处理则 RL 训练易被低质量样本污染。
- **Summary 模型支持**：多轮搜索不断累积上下文，直接塞入易触长度瓶颈、影响关键信息利用；框架需原生支持对搜索结果做总结压缩。

此外，常用 benchmark（如 BrowseComp）通常要调真实 Google Search + Jina Reader，真实 RL 训练场景下持续调用**成本极高**，使 Search Agent RL 往往只有工业界负担得起。一个领域不能只靠少数工业团队——需要更开放、低成本、可扩展的训练框架。

### Search-R1 的局限

Search-R1 是有代表性的开源实现，但存在明显限制：

1. **没有持续跟随 verl 演进**：基于早期 verl 改造，后续未同步 verl 最新能力，难以直接支持 Qwen3、Qwen3.5 等新模型。
2. **对 verl 的改造不够原生**：其 rollout 逻辑主要自己实现，属"非侵入式但割裂"的方式，很难随 verl 主框架升级自然演进。
3. **缺少可 scale up 的多轮训练验证**：更偏 2–3 轮场景，未充分验证能否稳定扩到 10–20 轮甚至更多；且原论文中 GRPO 训练到一定 step 后会**崩溃**，说明稳定性/可扩展性仍有问题。

---

## 三、主要贡献

### 3.1 一个基于 verl 的可扩展多轮搜索智能体 RL 框架

支持 Qwen3 等新模型在 verl 中进行**同步与异步** Search Agent RL 训练。提供两套可复现 recipe：

- **Search-R1 recipe**：面向 2–3 轮搜索场景；
- **ASearch recipe**：面向 10–20 轮多轮搜索场景。

实验表明框架能稳定支持 Search Agent RL 训练，并扩展到更多轮搜索交互。

### 3.2 多轮训练所需的内部指标监控与异常轨迹过滤

加入更细粒度的**内部（"内科"）指标监控**（环境稳定性、异常轨迹比例等），并实现**异常轨迹过滤**，过滤重复搜索、一次搜多个 query、格式错误、无效工具调用等低质量轨迹。

进一步提出**带信用分配的异常轨迹过滤**：相比简单丢弃整条轨迹，能更细粒度地区分轨迹中哪些行为真正导致异常，**减少对有效搜索行为的误伤**，提高训练稳定性与样本利用率。

### 3.3 搜索结果 Summary，提升多轮搜索能力

为在固定上下文长度下支持更多搜索轮次并提升泛化，支持对检索返回结果做 summary 压缩，两种方式：

- **Self-summary**：用正在训练的 Search Agent 自身模型总结（清空已有上下文，用新的 summary prompt）；
- **Other-summary**：单独部署一个外部 summary 模型总结。

通过 summary，模型在有限上下文内保留更多轮关键信息，支持更长搜索轨迹与更复杂求解。

---

## 四、实验结果

### 4.1 Search-R1 设置下显著超过原论文

相同模型（Qwen2.5）、相同训练数据下，用 SearchAgent-Zero 重训并评估，7 个测试集平均提升约 8 个点：

| 数据集 | Search-R1（原论文，Qwen2.5-3B-Instruct） | verl AgentLoop 复现（Qwen2.5-3B-Instruct） |
|---|---:|---:|
| NQ | 0.341 | 0.4640 |
| TriviaQA⋆ | 0.545 | 0.6164 |
| PopQA⋆ | 0.378 | 0.4239 |
| HotpotQA† | 0.324 | 0.4225 |
| 2Wiki⋆ | 0.319 | 0.3979 |
| Musique⋆ | 0.103 | 0.1808 |
| Bamboogle⋆ | 0.264 | 0.344 |
| **平均** | **0.325** | **0.40707** |

**verl 优化版本（qwen2.5-3b-instruct_searchr1）三处关键改动**：
1. 完善 system prompt，使用**原生 function call** 能力输出 XML 格式内容，而非 Search-R1 原本的固定 pattern（再解析）；
2. 增加异常轨迹约束；
3. 把 agent loop 改为 **verl 原生**的。

**结论**：原始 paper 的 GRPO 训练不稳定、收敛较慢；新框架在相同数据下可**快速收敛（100 step 从 0.4→0.5）且稳定训练**。一个正确、稳定、可扩展的 RL infra 本身就能显著影响 Search Agent 训练效果——呼吁社区在更可靠的框架与更强 baseline 上做创新。

（异常轨迹比例观察：早期会出现重复搜相同 query、并发搜多个 query，但后期因有异常轨迹过滤渐渐消失。）

### 4.2 纯 RL 训练出多轮搜索的 Search Agent（BrowseComp-Plus）

用多轮 QA 数据纯 RL 训练，Qwen3-8B 在 BrowseComp-Plus 测试集达 **14B 以下模型 SOTA：0.3795（37.95% Acc）**。且模型真正学会多轮搜索：

- 训练集平均约 **20 轮**搜索；
- 测试集平均约 **40 轮**搜索；
- 同时能**泛化到浅层搜索**场景，而非只适用超长轨迹。

说明 SearchAgent-Zero 不只提升 benchmark 分数，更验证了 Search Agent RL 可稳定 **scale up 到多轮**。

---

## 五、实验分析与关键发现

### 发现 1：正确的 RL infra 让 Search Agent 稳定 scale up

基于正确的 RL 基础设施，用 GRPO 训练 Search Agent 可稳定扩展到多轮搜索，**不必然出现 Search-R1 那种训练到一定 step 后崩溃**的问题。说明之前的崩溃不一定来自 GRPO 算法本身，而可能与 **rollout 实现、环境稳定性、异常轨迹处理**等工程细节密切相关。

### 发现 2：Summary 对多轮搜索训练较关键

加 summary 相比不加：
1. 相同训练 step 下**收敛更快**；
2. 模型能支持**更多轮**搜索。

对比 self-summary 与 other-summary：**8B 量级下两者最终效果基本一致**。说明中等规模模型上 self-summary 已具较好可用性，为低成本训练多轮 Search Agent 提供更简单的方案。

（Summary 好处补充：① 大大减少轨迹长度，固定上下文内训练更多轮次；② 增强泛化——不同工具返回经 summary 后，agent 接收到的检索结果形式一致，提升鲁棒性。）

### 发现 3：异常轨迹过滤显著提升训练质量

异常轨迹会严重影响训练稳定性与策略学习质量。加入异常轨迹过滤后，能有效降低异常轨迹占比、提高搜索轨迹质量、提升测试指标。在此基础上引入**带信用分配的异常轨迹过滤**，可进一步帮助模型 scale up 到更多搜索轮数并获得更好测试表现。**异常轨迹过滤不只是工程清洗步骤，而是影响多轮 Search Agent RL 训练上限的重要模块。**

### 发现 4：长 Thinking 不一定适合多轮搜索模型

对多轮 Search Agent，think 模型的长 thinking 并不总有利——过长 thinking 占用大量上下文预算，压缩搜索结果与历史信息空间，影响更多轮搜索。因此设计了一种方法：**在保留推理能力的同时，避免长 thinking 干扰多轮搜索过程**，让上下文预算更多用于搜索、阅读、信息整合，而非消耗在过长中间推理上。

> 具体地：基于 Qwen3，带 think 虽初始 reward 高约 2 个点，但原生 think 思维链太长（1k–2k token），上下文很快耗尽、无法 scale 轮数。因此训练时**关闭 think（空 think）**，让模型自己输出推理——实践证明多轮搜索数据集上才能 scale 起更多轮数。

---

## 六、核心机制细节

### 6.1 Search Agent 训练过程内部（"内科"）指标监控

**（1）异常轨迹比例**

*模型错误类*（会处理）：
- `abnormal_trajectory/tool_parser_error_count_percentage`：工具调用格式错误导致 parser 出错的比例；
- `abnormal_trajectory/searched_query_count_percentage`：重复搜索之前搜过的 query 的比例；
- `abnormal_trajectory/too_many_tool_call_count_percentage`：单次并发搜索 query 数超过最大并发限制的比例；
- `abnormal_trajectory/duplicate_search_result_count_percentage`：搜索返回 doc 相比之前无增量信息（未超阈值）的轨迹比例（**暂时只监控，未处理**）。

*模型正常但需 mask 类*（mask 掉这些轨迹，不抑制其测试集泛化性）：
- `abnormal_trajectory/too_many_turn_count_percentage`：工具调用轮数超过最大轮数限制的比例；
- `abnormal_trajectory/too_long_seq_truncated_count_percentage`：轨迹长度超过最大长度；
- `abnormal_trajectory/response_truncated_count_percentage`：单轮回复过程被截断的比例。

**（2）轮数相关指标**
- `turn/tool_call_success_rate/mean`：工具调用成功率（监控环境稳定性）；
- `turn/tool_call_turn/mean`：平均调用轮数（监控串行/并发轮数）；
- `turn/tool_call_success_counts/mean`：所有调用 query 成功的次数；
- `turn/all_call_tool_counts/mean`：所有调用 query 的次数。

### 6.2 带信用分配的异常轨迹过滤

大多数 paper 对模型错误部分都是**停止 rollout、直接给整条 0 reward**，惩罚整条轨迹。但对多轮 Search Agent 而言，**模型错误往往只是当前轮次错误，前面轮次可能正常**，粗暴一起打压不是好策略。

因此提出**带信用分配的异常轨迹过滤**——对模型错误部分**只惩罚当前出错轮次的 token，不惩罚前面的 token**：

- **模型错误**（生成工具错误 / 重复搜 query / 并发 query 过多 / 结果无增量[暂只监控]）：停止 rollout，给 0 reward，但**只对最后一轮出错的 token 惩罚**，不惩罚前面 token；
- **模型正常但超限**（超过最大轮数 / 超过最大长度 / 单轮回复被截断）：**mask 掉这些轨迹，参与优势计算但不更新梯度**（不抑制其测试集泛化性）。

### 6.3 Summary 模型

- **self-summary**：用正在训练的 Search Agent 模型，**清空已有上下文**，用新的 summary prompt 做总结；
- **other-summary**：另外部署一个 summary 模型来做总结。

---

## 七、消融实验（BrowseComp-Plus）

### 7.1 信用分配的价值：普通异常过滤 vs 带信用分配的异常过滤

| 模型 | Accuracy | Recall | 平均搜索次数 |
|---|---:|---:|---:|
| 裸 Qwen3-32B | 10.72% | 7.28% | 0.94 |
| Qwen3-8B 不带异常过滤 + self-summary | 轨迹指标很差（重复+并发搜大量 query），经常超长，无法得到有效结论 | — | — |
| Qwen3-8B + 异常轨迹过滤 + self-summary（100 step） | 24.21% | 33.14% | 10.11 |
| Qwen3-8B + **带信用分配的**异常过滤 + self-summary（100 step） | **28.19%** | **40.1%** | **14.22** |

**结论**：对异常轨迹，只有当前轮次有问题，但多数工作给整条 0 reward、误伤前面正常轮次。带信用分配的过滤只惩罚当前异常轮，使 **Acc 28.19 vs 24.21、Recall 40.1 vs 33.14、平均搜索 14.22 vs 10.11**——三指标同向提升。

### 7.2 最终版本（300 step）

> 训练 300 step。因训练成本较高，这是早期跑的模型，**未加带信用分配的异常轨迹过滤**。

- 训练 reward 可持续 scale up；
- 轮数：纯 RL 可随训练持续 scale up 轮数，达到训练时 **20 轮**；
- 回复长度、异常轨迹过滤指标均正常。

| 模型 | Accuracy | Recall | 平均搜索次数 |
|---|---:|---:|---:|
| 裸 Qwen3-32B | 10.72% | 7.28% | 0.94 |
| Qwen3-8B + 异常过滤 + self-summary（100 step） | 24.21% | 33.14% | 10.11 |
| Qwen3-8B + 带信用分配的异常过滤 + self-summary（100 step） | 28.19% | 40.1% | 14.22 |
| **SearchAgent-Zero（300 step）** | **37.95%** | **50.87%** | **38.47** |

> 局限说明：这是 Qwen3-8B（非专门 agent 模型）纯 RL 场景的结论，且 BrowseComp-Plus 难度超过训练集 Q-A 对，故测试搜索轮数一般多于训练。因此训练中搜索轮数越多的模型，测试集表现往往越好（OOD 越少）：100 step 时"带信用分配"版可训到平均搜索 9 轮，而普通过滤只能训到约 5.5 轮，因此效果更好。

---

## 八、支持 scale 多轮的 verl 参数配置

1. **增加 clip-high、去除 KL loss**：`clip-high=0.34`；
2. **增加 TIS 和 MIS**：缓解训练与推理不一致问题。

（具体配置见仓库：https://github.com/NLPJCL/SearchAgent-Zero）

---

## 九、QA

**为什么基于 verl？**
slime 也可以，甚至因其设计（暴露 rollout 接口和 reward 接口）可能更适合改；但作者对 verl 更熟悉，故基于 verl 做。

**为什么从 verl 独立出来，而不是给 verl 提 PR？**
1. Search Agent 需要自己的特殊监控指标和轨迹过滤逻辑，放在一个 verl AgentLoop 下会很冗余；
2. 侵入式修改难维护——随 verl 更新，这套代码本质上是改一个新的 AgentLoop + 一些内部指标监控，可让 codex 跟进这些逻辑，一键迁移到最新版 verl 或其他框架（后续会给出迁移 prompt）；同时保留了一个分支，通过 commit 记录可清楚看出改了什么。

---

## 十、与本 report 其他文档的口径对照

| 项 | 官方博客口径（本文，权威） | 备注 |
|---|---|---|
| Search-R1 平均分 | 0.325 → **0.40707 / 0.4085**（相对 +25.3%，约 +8pt） | 其他文档写 0.407，等价 |
| BrowseComp-Plus 最终 | **37.95% Acc / 50.87% Recall / 38.47 次**（300 step） | 其他文档偶写 38% / ~50%，以本文为准 |
| 训练/测试平均轮数 | 训练 ~20 轮 / 测试 ~40 轮 | — |
| 信用分配消融 | 24.21→28.19 Acc / 33.14→40.1 Recall / 10.11→14.22 次 | — |
| SOTA 声明 | 14B 以下模型 SOTA（BrowseComp-Plus） | 绑定该榜单口径 |
| **IGPO** | 官方博客**未提及** | IGPO 为本仓库后续探索，不在官方发布口径内；简历若写需标"进行中/探索性" |
| 训推一致性 | 官方提到 **TIS / MIS**（去 KL、clip-high=0.34） | 本 report 早期文档未强调，可补 |
| 长 thinking | 关闭 think（空 think）以 scale 轮数 | 官方明确发现，其他文档未覆盖 |

---

*本文为作者官方博客的忠实整理，数字与表述以官方为准；如与 report 下其他文档冲突，以本文口径修正。*
