# SearchAgent-Zero — 简历项目介绍

> 长程多轮搜索智能体的强化学习(RL)训练。基于 verl + AgentLoop,用 Qwen3-8B 在 Deep-Research 评测 BrowseComp-Plus 上,纯靠 RL 达到 ~38% Accuracy / ~48–51% Recall(随 checkpoint 而定)。本项目只涉及**同步 GRPO** 训练。

---

## 一、简历 Bullet Points

### 中文版(推荐 4–5 条上简历)

- **长程搜索智能体 RL 训练框架**:基于 verl + AgentLoop 构建可稳定扩展到 **30–40 轮**工具调用的多轮搜索 RL 训练管线,通过 agent-loop 状态机的长度/轮数守卫消除基线框架在长轨迹上的 rollout 崩溃;用 Qwen3-8B 在 Deep-Research 基准 **BrowseComp-Plus(830 题)达到 38.1% Accuracy / 48% Recall**,在 14B 以下开源模型中具很强竞争力。

- **轮级信用分配(核心算法改进)**:针对长轨迹"某一轮异常就连坐惩罚整条轨迹"的问题,基于 `response_mask` 设计 token 级掩码,在异常轮终止前将**该轮之前的正确搜索 token 从损失中剔除**,把惩罚精确定位到肇事轮。消融(唯一变量=是否信用分配,同 100 步):**Accuracy +4.0pt(24.2%→28.2%)、Recall +7.0pt(33.1%→40.1%)、平均检索轮数 10→14**,三指标同向印证"解除连坐让模型敢多轮搜索"。

- **上下文压缩(self-summary)**:用 policy 模型自身对每轮长检索结果并发摘要(失败回退原文、prompt 强制"证据不足则声明",抑制幻觉),摘要 token 不计入损失;在固定 context 预算内把可执行搜索轮数从个位数扩展到 **~38 轮**,支撑长程 scaling(100→300 步,Recall 33%→51%)。

- **IGPO 过程奖励(进阶算法探索)**:将信用分配从"轮级掩码"推进到"步级过程奖励",以模型每轮对 ground-truth 答案的**概率提升**($e^{\overline{\log P_t}}-e^{\overline{\log P_{t-1}}}$)作为 process reward,并用扩展序列 + 4D attention mask 把 T 次 GT 前向压成 **1 次**前向。早期(step_200)即达 **29.0% Acc / 40.3% Recall,超过同期 GRPO(step_250/300)**,展现更高样本效率与证据召回。

- **短程 recipe 验证 + 多维度分析**:在 Search-R1 短程设置(Qwen2.5-3B,7 个 QA 数据集)复现并改进,**平均 EM +8.2pt(0.325→0.407,7/7 全正向)**;发现"检索器决定上限"(dense 36% vs BM25 6.5%)与"训练 reward ≠ 下游表现",据此建立用 held-out 验证信号选 checkpoint 的流程。

### English version

- Built a **long-horizon search-agent RL pipeline** on verl + AgentLoop that scales stably to **30–40 tool-call turns**, eliminating rollout crashes via length/turn guards in the agent-loop state machine; trained Qwen3-8B to **38.1% Accuracy / 48% Recall on BrowseComp-Plus (830 Q)**, competitive among sub-14B open models.
- Designed **turn-level credit assignment** via `response_mask` token masking: on an abnormal turn, prior correct-search tokens are dropped from the loss so the penalty lands only on the offending turn. Ablation (sole variable, same 100 steps): **+4.0pt Accuracy, +7.0pt Recall, avg search turns 10→14** vs. whole-trajectory penalty.
- Added **self-summary context compression** (policy model summarizes each retrieval concurrently, falls back to raw text on failure, prompt forbids fabrication; summary tokens excluded from loss), extending executable search turns to **~38** and enabling long-horizon scaling (Recall 33%→51%, 100→300 steps).
- Explored **IGPO process rewards**, pushing credit assignment from turn-level masks to per-step rewards using the model's per-turn **probability gain toward the ground truth** ($e^{\overline{\log P_t}}-e^{\overline{\log P_{t-1}}}$), with an extended-sequence + 4D-attention-mask trick collapsing T GT forward passes into **one**. At just step_200 it reaches **29.0% Acc / 40.3% Recall, beating same-step GRPO**, indicating higher sample efficiency.
- Reproduced and improved the short-horizon **Search-R1** recipe (**+8.2pt avg EM, 7/7 datasets**); surfaced "retriever sets the ceiling" (dense 36% vs BM25 6.5%) and "training reward ≠ downstream metric", establishing held-out checkpoint selection.

---

## 二、完整项目介绍

### 2.1 背景与动机

大模型正从"把知识压进参数"转向"**边推理边检索**"的 Deep-Research 范式:模型面对复杂问题时多轮调用搜索工具,边检索边推理,逐步逼近答案。要让模型学会**何时搜、搜什么、如何整合、何时停下作答**,SFT 只能模仿固定轨迹,而 RL 能在与检索环境的交互中试错、按答案是否正确获得奖励——这是训练"决策式搜索"能力的自然选择。Search-R1 已验证短程(2–6 轮)"推理 + 检索" RL 可行。

**本项目的问题**:社区缺一个**能稳定复现、又能平滑扩展到长程多轮(数十轮)**的开源 RL recipe。目标不是刷单点,而是让搜索智能体 RL 变得 **可复现、可扩展到长程、可在其上做算法探索**。

### 2.2 难点

真正的难点不在"能不能搜一两轮",而在**能不能稳定地搜几十轮**。轨迹从 2–6 轮扩到 20–40 轮时,三个问题集中爆发:

| # | 难点 | 症状 |
|---|------|------|
| P1 | **长程 rollout 会崩** | 基线框架在长多轮轨迹上出现 rollout crash,训练扩不上去 |
| P2 | **上下文预算爆炸** | 每轮检索文档很长,几轮就填满 context,无法进行真正的长程搜索 |
| P3 | **异常轨迹污染训练信号** | 重复搜索、单轮并行 query 过多、工具解析失败等噪声,若整条惩罚会"错杀"前面正确的搜索行为 |

### 2.3 方法:三个改进,各治一个难点

**改进 1 —— 稳定可扩展的 RL 基础设施(治 P1)**
基于 verl + **AgentLoop**,用 vLLM 执行多轮工具调用(`multi_turn.enable=True`,`default_agent_loop=tool_agent`),配同步 GRPO 训练入口。使长程轨迹稳定扩展到 20–40 轮而不崩,消除基线框架在长轨迹上的 rollout crash。

**改进 2 —— 上下文压缩 self-summary(治 P2)**
每轮长检索结果先用模型自身压缩成摘要(`enable_tool_response_summary=True`,`summary_max_tokens=1024`)再放回上下文。在固定 context 预算内容纳更多搜索轮次,充分训练后平均检索轮数扩展到 ~38 轮。
*代价(诚实说明)*:自摘要会丢失原文细节、可能引入摘要幻觉;本项目以单轮保真度换整体覆盖度,下游 Recall 的提升(见消融)表明在该 benchmark 上这笔交易划算。

**改进 3 —— 轮级信用分配(治 P3,核心算法改进)**
先定制细粒度异常轨迹监控指标(工具成功率、平均轮数、重复 query、单轮并行 query 过多、解析失败、截断……);再对异常轨迹做**信用分配式惩罚**——当异常只发生在某一轮时,**用 token 级掩码把惩罚精确限制在该轮相关的 token**,不牵连前面正确的搜索行为。
*直觉*:整条连坐会给出错误梯度——"多搜就容易踩异常、干脆少搜";轮级掩码保留前序正确搜索的正向信号,让模型敢搜更多轮。

### 2.4 实验

所有对比使用同一 judge(Qwen3-32B)、同一评测语料、同一检索器,仅改待评变量,保证对照公平。受算力限制为**单次运行**,故强调"多指标同向 / 全数据集一致"这类趋势性证据,而非单点数值的显著性。

#### 实验一:短程验证(Search-R1)—— 地基是否牢

**设置**:Qwen2.5-3B-Instruct,与 Search-R1 同 base、同数据、同检索器,7 个开放域 QA;指标 EM/score@1。

| 数据集 | Search-R1 | 本 recipe | 绝对提升 |
|---|---:|---:|---:|
| NQ | 0.341 | 0.464 | +0.123 |
| TriviaQA | 0.545 | 0.616 | +0.071 |
| PopQA | 0.378 | 0.424 | +0.046 |
| HotpotQA | 0.324 | 0.423 | +0.099 |
| 2Wiki | 0.319 | 0.398 | +0.079 |
| Musique | 0.103 | 0.181 | +0.078 |
| Bamboogle | 0.264 | 0.344 | +0.080 |
| **平均** | **0.325** | **0.407** | **+0.082** |

**结论**:7/7 全正向,平均 +8.2pt。稳定基础设施 + AgentLoop + 异常轨迹处理确实改善训练质量,为扩展到长程提供信心。*(只报绝对提升;低基数数据集如 Musique 的相对百分比会显得夸张、不可比,故不采用。)*

#### 实验二:长程消融(BrowseComp-Plus)—— 每个改进值多少分

BrowseComp-Plus 把「检索器」与「LLM agent」**解耦**:查询来自 OpenAI BrowseComp,但针对固定的 ~100K 人工核验文档语料检索,公平可复现。共 **830 题**。指标:Accuracy(Qwen3-32B judge)、Recall(证据文档召回)、平均检索轮数(**过程指标——只有伴随 Recall 上升才说明"多搜"有效**)。

| # | 设置 | Accuracy | Recall | 平均检索轮数 | 相比上一行 |
|---|------|---:|---:|---:|---|
| ① | Qwen3-32B(无训练 baseline) | 10.7% | 7.3% | 0.9 | 起点:几乎不主动搜 |
| ② | Qwen3-8B + 异常过滤 + self-summary(100 步) | 24.2% | 33.1% | 10.1 | **+13.5pt Acc**:RL 让 8B 学会主动多轮搜索 |
| ③ | + **轮级信用分配**(100 步) | 28.2% | 40.1% | 14.2 | **+4.0pt Acc / +7.0pt Recall**:信用分配净贡献 |
| ④ | 充分训练(300 步) | 38.0% | 50.9% | 38.5 | **+9.8pt Acc**:长程持续爬升 |

**三个结论**:
- **A(①→②,RL 的价值)**:未训练的 32B 几乎不搜(0.9 次),Acc 仅 10.7%;RL 训练的 8B 搜到 10 次、Acc 翻倍到 24.2%。**参数小 4 倍,但学会主动搜索就能反超。**
- **B(②→③,信用分配的必要性——最干净的对照)**:唯一变量是"是否轮级信用分配"。解除连坐后,模型敢搜更多轮(10→14)→ 找回更多证据(Recall +7.0pt)→ 答对更多题(Acc +4.0pt)。**三指标同向且逻辑自洽**,这条因果链比单看 Acc 更有说服力。
- **C(③→④,长程可扩展)**:100→300 步,检索轮数 14→38、Recall 破 50%、Acc 到 38%。证明搜索智能体 RL 能**稳定 scale 到几十轮而不崩**,背后是改进 1(不崩)与改进 2(装得下)在支撑。

#### 实验三:多 checkpoint 复核 —— 两个工程洞见

在 BrowseComp-Plus 上评测多个 checkpoint,交叉印证并补充:

**洞见 1:检索器决定上限**(固定同一 step_500 模型,只换检索器)

| 检索器 | Accuracy | Recall |
|---|---:|---:|
| dense(稠密) | 36.0% | 41.4% |
| qwen3-embedding-8b | 28.4% | 31.2% |
| BM25(稀疏) | 6.5% | 6.1% |

同一 agent 换 BM25 从 36% 掉到 6.5%——**agent 再强也救不回召不回证据的检索器**,呼应 BrowseComp-Plus"解耦"的设计初衷。*(caveat:BM25 断崖有一部分可能源于稀疏检索配置未充分调优,故结论限定在"对齐配置下,稠密显著优于稀疏",作为"检索预算投稠密"的工程指引。)*

**洞见 2:训练 reward ≠ 下游表现**
step_500 训练 reward 最高,但下游 **step_400(38.1%)> step_500(36.0%)**。训练奖励(答案 log-prob / 格式分)与真实 Accuracy 存在 gap,故选 checkpoint 用**独立的 held-out 验证信号**(而非直接在测试集挑点,也非只看 reward 峰值)。

### 2.5 进阶探索:IGPO(更细粒度的信用分配)

**动机**:GRPO 只有一个最终 outcome reward,对长轨迹信用分配太粗——几十轮里哪一轮真正带来信息增益说不清。IGPO 把信用分配从"轮级"推向"步级过程奖励":以"每步搜索后模型对 ground-truth 答案的概率提升"作为 process reward(`info_gain_type=prob_diff`,过程/结果奖励独立归一化)。

**早期实测**:训到 step_200,BrowseComp-Plus 达 **29.0% Acc / 40.3% Recall**,已**超过同量级 step 的 GRPO**(step_250 26.9% / step_300 25.3%),且检索轮数更高。这与机制预期一致:稠密过程奖励在早期就鼓励更主动的探索。

**价值命题(可证伪)**:IGPO 主张的不是更高天花板,而是**更高样本效率 + 更好证据召回**——即"相同步数下 Recall 更高,或达到相同 Accuracy 所需步数更少"。step_200 的早期信号朝此方向为正;完整验证(训到 step_400 与同 step GRPO 严格对照)为明确的下一步工作。

> 面试口径:IGPO 作为"研究品味 + 严谨假设设定"呈现,是**进行中的探索**,不作已完成结论。

### 2.6 结论

1. **核心价值**:不是刷单点,而是让搜索智能体 RL 可复现、可稳定扩展到长程多轮——从 2–6 轮到近 40 轮全程不崩。
2. **三个改进各有消融支撑**:基础设施→300 步稳定训练;self-summary→检索轮数扩到 ~38;**轮级信用分配→最干净的对照,Acc +4.0pt / Recall +7.0pt,三指标同向**。
3. **成果与边界**:Qwen3-8B 纯 RL 在 BrowseComp-Plus 达 ~38% Acc / 48–51% Recall;在本评测口径(830 题、Qwen3-32B judge、稠密检索,截至 2026-08)下,于 14B 以下开源模型中具很强竞争力。绝对声明绑定口径与时间,不作无限定 SOTA 主张。
4. **两个工程洞见**:检索器决定上限;训练 reward ≠ 下游表现,选模型须用 held-out 复核。

---

## 三、附录:超参与已知局限

**核心超参(Qwen3-8B / 同步 GRPO)**

| 项 | 值 | 项 | 值 |
|---|---|---|---|
| adv_estimator | grpo | rollout.n | 8 |
| lr | 1e-6 | rollout / val temp | 1.0 / 0.7 |
| train_batch_size | 128 | max_assistant_turns | 100 |
| max_response_length | 36864 | kl_loss_coef / type | 0.001 / low_var_kl |
| entropy_coeff | 0 | clip_ratio_high | 0.34(DAPO 风格) |
| self-summary | 开启,max_tokens=1024,用模型自身(不用外部模型) | | |

**已知局限(主动披露,避免被 judge)**
- **单次运行、无多 seed**:增量(尤其 §消融的 +4.0pt Acc)为单次结果,受算力限制未做方差估计;方向性(同向、全数据集一致)比单点更可信。
- **IGPO 未训完**:仅到 step_200 实测,更高步数为下一步。
- **BM25 配置 caveat**:稀疏检索差距一部分可能受配置影响,结论限定"对齐配置下"。
- **选 checkpoint**:用 held-out 验证信号选、再报测试集,规避 test-set peeking。

---

*本项目为作者本人在开源 recipe 基础上的复现与扩展。所有性能声明绑定具体评测口径与时间;IGPO 更高步数结果为明确标注的下一步工作,不作已测数据引用。*
