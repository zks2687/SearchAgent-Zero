# SearchAgent-Zero 源码实现详解(面试防穿版)

> 目的:把简历上 4 个改进点的**代码实现、设计策略、以及"为什么这么写"**讲透,面试官追问到代码级也能接住。
>
> 所有引用均指向仓库 `/mmu_mllm_hdd_2/zhoujinchang/workspace/SearchAgent-Zero`(IGPO 部分在 `report/IGPO_ref/`)。

---

## 0. 先把大框架讲清楚:verl / AgentLoop / Search-R1 的关系

面试官第一个会问的往往是:"你到底在谁的基础上做的?"一句话讲清分层:

```
verl(RL 训练框架:FSDP actor + vLLM rollout + GRPO/PPO 算法)
  └─ AgentLoop(verl 的多轮 agent rollout 抽象:experimental/agent_loop/)
       └─ ToolAgentLoop(通用"LLM ↔ 工具"多轮循环)
            └─ ToolAgentLoop(本项目:search 场景 + 异常过滤 + self-summary + 信用分配)
```

- **verl**:提供 actor(FSDP2)、rollout(vLLM async server)、advantage 估计(GRPO)、PPO clip loss 等基础件。
- **AgentLoop**:verl 里负责"一条多轮轨迹怎么 rollout"的抽象层。核心产物是 `AgentLoopOutput`(`agent_loop.py:188`),里面最关键的两个字段:
  - `response_ids`:整条轨迹的 token(含模型生成 + 工具返回)。
  - **`response_mask`**:与 `response_ids` 等长的 0/1 序列,**1 = 参与训练(算 loss)的 token,0 = 只当上下文、不算 loss 的 token**。这是理解后面所有改进的钥匙。
- **Search-R1 的原始做法**:短程(2–6 轮),把"检索文档原文"直接拼进 prompt,轨迹短、不涉及压缩/信用分配。verl 官方 AgentLoop 也只提供一个"能跑就行"的 `ToolAgentLoop`。

**我在 Search-R1 / verl AgentLoop 基础上做的 4 件事**(下面逐一讲实现):
1. 让长程 rollout 不崩(基础设施 + turn schedule + 截断即终止)。
2. self-summary 压上下文。
3. 异常轨迹过滤 + **轮级信用分配**(response_mask 的精细控制)。
4. IGPO:把 outcome reward 变成 step 级过程奖励(在 IGPO 分支)。

> **关键概念(反复用到)**:多轮轨迹的 `response_mask` 长这样(`agent_loop.py:552`):
> ```
> responses:     |<-第1轮生成->|<-工具返回->|<-第2轮生成->|<-工具返回->| ...
> response_mask: | 1,1,...,1,1 | 0,0,...,0,0 | 1,1,...,1,1 | 0,0,...,0,0 | ...
> ```
> 工具返回的 token 永远是 0(不训练),模型生成的 token 默认是 1。**我所有的"过滤/信用分配"本质上都是在改这个 mask 里哪些 1 该变成 0。**

---

## 改进 1:稳定可扩展的长程 RL 基础设施

**文件**:`verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py`(状态机 `run()`,`:319`);训练脚本 `run_qwen3_8b_instruct_search_multiturn_ASearch.sh`。

### 1.1 rollout 是一个显式状态机

`run()` 用一个状态机跑完一条多轮轨迹(`:358`):

```python
state = AgentState.PENDING
while state != AgentState.TERMINATED:
    if   state == PENDING:          state = _handle_pending_state(...)      # 拼 prompt
    elif state == GENERATING:       state = _handle_generating_state(...)   # 模型生成一轮
    elif state == PROCESSING_TOOLS: state = _handle_processing_tools_state()# 执行工具、拼回观测
```

每轮:`GENERATING`(模型生成 assistant turn)→ 解析 tool_call →`PROCESSING_TOOLS`(执行 search、把结果拼回上下文)→ 回到 `GENERATING`,直到模型输出 `<answer>` 或触发终止条件。

### 1.2 长程为什么会崩,我怎么治

原框架在几十轮长轨迹上崩,根因是**轨迹长度失控**:context 撑爆 max_model_len、或轮数无限增长。我在 `_handle_generating_state` 里加了**多道硬边界,一旦越界立即安全终止**(`:416`、`:464`、`:735`):

```python
# 生成前:prompt 已逼近 max_model_len → 直接终止,整条 mask 置 0(不参与训练)
if len(agent_data.prompt_ids) >= self.max_model_len - 1:
    agent_data.response_mask = [0] * len(agent_data.response_mask)
    agent_data.abnormal_trajectory_dic["too_long_seq_truncated_count"] += 1
    return AgentState.TERMINATED

# 生成后:response 超长 / 轮数超限 → 同样置 0 终止
if len(agent_data.response_mask) >= self.response_length: ...
if agent_data.assistant_turns >= max_assistant_turns: ...
```

**策略要点**:超长/超轮的轨迹不是"报错崩溃",而是"**优雅终止 + 整条 mask 置 0**"——既不炸训练,又不让这条畸形轨迹污染梯度。这是"长程不崩"最核心的一招。

### 1.3 turn limit 课程调度

脚本里 `turn_limit_schedule="0:100,50:100,..."`,在 `_scheduled_turn_limit`(`:310` 附近)按训练 step 动态取当前允许的最大轮数。可以做成"前期少轮、后期多轮"的课程;本项目实测固定 100,给长程留足空间。

### 1.4 异步 vLLM rollout(仅指执行方式,不是异步训练)

脚本里 `rollout.mode=async` + `multi_turn.enable=True`:vLLM 以 async server 形式对外,`_handle_generating_state` 里 `await self.server_manager.generate(...)`(`:428`)非阻塞地并发跑很多条轨迹的生成。**注意**:这只是 rollout 阶段的并发生成,训练仍是同步 GRPO(`main_ppo`,hybrid engine,8 卡共用)。我们已把复杂的 fully-async 训练排除在简历叙事之外,这里的 async 仅指"生成端并发",别和异步训练混为一谈。

> **面试如果追问"崩在哪一层"**:答——崩在 rollout 端,长轨迹把 vLLM 的 KV cache / context 撑爆,或轮数不收敛导致单条轨迹无限跑。治法是"在 agent loop 状态机里加长度/轮数守卫 + 越界即置零终止",而不是改 vLLM 本身。

---

## 改进 2:self-summary 上下文压缩

**文件**:同上,`_handle_processing_tools_state`(`:569`)、`_generate_single_summary`(`:863`)、`PROMPT_TEMPLATE`(`:45`)。

### 2.1 痛点与思路

每次 `search` 返回若干篇长文档,直接拼进 context,几轮就把 max_model_len 填满(改进 1 的守卫会因此提前终止轨迹,搜不了几轮)。**思路**:检索结果不原样拼回,而是**先让模型自己把"这批文档对当前 query 的答案"压成一段短摘要,再拼回上下文**。

### 2.2 实现:逐 query 并发摘要

在工具返回后(`:616`):

```python
if self.enable_tool_response_summary and tool_call.name == "search" and tool_response_text and query_list:
    # 把一次 search 的多个 query 结果按分隔符切开
    tool_response_text_lst = json.loads(tool_response_text)['result'].split(self.summary_result_separator)
    summary_tasks = []
    for q_idx, (query, doc_item) in enumerate(zip(query_list, tool_response_text_lst)):
        # 先按 token 数截断超长原文(防摘要输入本身爆掉)
        doc_item, truncated = await self._truncate_text_by_tokens(doc_item, self.max_tool_response_length, ...)
        summary_tasks.append(self._generate_single_summary(query, doc_item, q_idx + 1))
    summary_results = await asyncio.gather(*summary_tasks)   # 多 query 并发摘要
    all_summary_text = self.summary_result_separator.join(summary_results)
    message_text = all_summary_text if all_summary_text else <截断后的原文兜底>
```

关键策略:
- **每个 query 单独摘要**并 `asyncio.gather` 并发,不串行,省时间。
- **摘要失败/为空 → 回退到截断后的原文**(`:652`),保证不因摘要挂掉丢整轮。
- **截断双保险**:摘要前先按 token 截断原文,记 `response_truncated_count` 指标。

### 2.3 用模型自身摘要(不用外部模型)

`_generate_single_summary`(`:863`)按 `summary_use_external_model` 分流;本项目 `=False`,走 `generate_single_summary_self`(`:904`):

```python
async def generate_single_summary_self(self, query, document, idx):
    summary_prompt = PROMPT_TEMPLATE.format(query=query, documents=document)
    prompt_ids = await self.apply_chat_template([{"role":"user","content":summary_prompt}], ...)
    output = await self.server_manager.generate(request_id=uuid4().hex, prompt_ids=prompt_ids,
                 sampling_params=dict(temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=1024))
    summary_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
    return f"the summary of the query {idx} search result is : {summary_text}"
```

- **同一个 policy 模型**(同一个 vLLM server)兼做 summarizer,不额外起模型 → 省显存、省部署。
- 摘要用**独立的 request_id + 独立采样参数**(temp 0.6),与主生成解耦。
- `PROMPT_TEMPLATE`(`:45`)强约束:先 Reasoning 再 Summary,**证据不足必须显式输出 "Information Insufficient",禁止编造** → 抑制摘要幻觉。

### 2.4 摘要 token 算不算 loss?—— 不算

摘要结果作为 tool 消息拼回时(`:749`):

```python
agent_data.prompt_ids += response_ids
agent_data.response_mask += [0] * len(response_ids)   # 摘要/工具返回一律 mask=0
```

**这点面试必问**:摘要是"环境观测"的一部分,`response_mask=0`,**不参与策略梯度**,只当上下文。模型学的是"如何基于摘要继续决策",而不是"如何写摘要"。

> **代价的诚实回答**:自摘要必然有信息损失、可能引入幻觉;`PROMPT_TEMPLATE` 的"Information Insufficient"约束 + 失败回退原文是缓解手段。收益(Recall 提升)见叙事报告消融——在本 benchmark 上这笔交易划算,但属经验结论。

---

## 改进 3:异常轨迹过滤 + 轮级信用分配(核心)

**文件**:`tool_agent_loop_credit_assignment.py`;对照基类 `tool_agent_loop.py`。**这俩文件的 diff 就是信用分配的全部**,面试官若要你"指出改了哪一行"就是这里。

### 3.1 先定义"异常轨迹",并全程埋点

在 rollout 过程中检测多类异常,累加到 `abnormal_trajectory_dic`(既是过滤依据,也是监控指标):

| 异常类型 | 计数字段 | 检测位置 |
|---|---|---|
| 工具解析失败(query_list 非法/JSON 错) | `tool_parser_error_count` | `:493`、`:510` |
| 单轮并行 query 过多(> `max_queries_per_tool_call=4`) | `too_many_tool_call_count` | `:497` |
| 重复 query(归一化后已搜过) | `searched_query_count` | `:503` |
| 检索结果整体重复(与历史文档高度重叠) | `duplicate_search_result_count` | `:607` |
| 轨迹超长截断 | `too_long_seq_truncated_count` | `:424`、`:465`、`:735` |
| 轮数超限 | `too_many_turn_count` | `:469` |

其中"重复 query"用 `normalize_answer` 归一化后查 `searched_query` 集合(`:502`);"结果重复"用文档签名集合的 Jaccard-min 重叠 ≥ 2/3 判定(`_all_search_results_are_duplicate`,`:553`)。

### 3.2 关键对比:基类"连坐" vs 本项目"信用分配"

看两文件的 diff(实测 `diff` 只差这几行,`_mask_previous_response_tokens` + 4 处调用):

**基类 `tool_agent_loop.py`** —— 检测到异常直接终止,**前面所有轮的 `response_mask` 保持 1**:

```python
if query in agent_data.searched_query:
    agent_data.abnormal_trajectory_dic['searched_query_count'] += 1
    return AgentState.TERMINATED     # ← 整条轨迹带着"坏 reward"进 loss = 连坐
```

**本项目 `..._credit_assignment.py`** —— 多了一个掩码函数和调用:

```python
@staticmethod
def _mask_previous_response_tokens(agent_data, current_turn_start):
    # 把"本轮开始之前"的所有 response_mask 置 0
    agent_data.response_mask[:current_turn_start] = [0] * current_turn_start
```

调用点:每轮生成开始时先记住本轮起点(`:455`)

```python
current_turn_start = len(agent_data.response_mask)   # 本轮 token 的起始下标
agent_data.response_mask += [1] * len(agent_data.response_ids)
```

一旦本轮触发异常,终止前先把**前面正确轮的 mask 清零,只留下出问题的这一轮**(`:494`、`:499`、`:505`、`:511`):

```python
if query in agent_data.searched_query:
    agent_data.abnormal_trajectory_dic['searched_query_count'] += 1
    self._mask_previous_response_tokens(agent_data, current_turn_start)  # ← 只罚这一轮
    return AgentState.TERMINATED
```

### 3.3 为什么这样就实现了"信用分配"

- GRPO 的优势/奖励最终**只作用在 `response_mask==1` 的 token** 上。
- 一条轨迹前面搜了 5 轮都对,第 6 轮重复 query 踩了异常:
  - **连坐(基类)**:6 轮的 token mask 全是 1 → 坏的组优势/低 reward 会拉低**前 5 轮正确行为**的梯度 → 模型学到"多搜有风险,少搜为妙"。
  - **信用分配(本项目)**:`_mask_previous_response_tokens` 把前 5 轮置 0,只有第 6 轮参与 loss → 惩罚精确落在犯错的那一轮,前 5 轮正确搜索**不被牵连** → 模型敢继续多轮搜索。
- 这正是消融里 ②→③ 的机制解释:检索轮数 10→14、Recall +7pt、Acc +4pt(三指标同向)。

> **面试防穿的三个细节**:
> 1. **为什么是 `[:current_turn_start]` 而不是别的区间?** 因为 `current_turn_start` 恰好是"本轮第一个生成 token 的下标",切片前缀 = 本轮之前的全部内容(含历史生成 + 历史工具返回),清零后只剩当前异常轮。
> 2. **工具返回本来就是 0,清零它们有害吗?** 无害,幂等——它们本就是 0。
> 3. **超长/超轮为什么是"整条置 0"而不是"只留本轮"?** 因为那类异常是"轨迹整体失控",没有"某一轮该负责"的语义,直接整条作废最干净(`:465` 等)。而 query 类异常有明确的"肇事轮",才用信用分配。**这个区分是设计的精髓。**

---

## 改进 4:IGPO —— 从轮级信用分配到 step 级过程奖励(进阶探索)

**文件**(在 `report/IGPO_ref/`):过程奖励打分 `verl/utils/reward_score/info_gain.py`;GT log-prob 计算 `scrl/llm_agent/vectorized_gt_logprob.py`;优势归一化 `verl/trainer/ppo/core_algos.py`(`compute_grpo_outcome_advantage`,`:189`)。

### 4.1 动机(和改进 3 一脉相承)

改进 3 解决的是"**坏轮别连坐好轮**",但对"好轮"内部仍是一个笼统的 outcome reward——几十轮里**哪一轮真正带来了信息增益**说不清。IGPO 把信用分配从"轮级 mask"推进到"**每一步一个过程奖励**":哪一步搜索让模型更接近正确答案,哪一步就该得正 reward。

### 4.2 过程奖励怎么算:GT 概率的逐轮提升

核心定义(`vectorized_gt_logprob.py:745` 附近):对第 t 轮结束后的上下文,算模型对 **ground-truth 答案**的平均 log-prob,取相邻两轮之差作为该轮的信息增益奖励:

```python
mean_log_prob = answer_log_probs.mean()          # 当前轮末,模型对 GT 答案的平均 logP
cur_value = math.exp(mean_log_prob)              # prob_diff:转成概率(log_prob_diff 则直接用 logP)
info_gain = cur_value - prev_value               # 相邻两轮之差 = 这一轮带来的"信息增益"
```

- `info_gain_type=prob_diff`(本项目):$e^{\overline{\log P_t}} - e^{\overline{\log P_{t-1}}}$,即"搜完这一轮后,模型答对的概率涨了多少"。
- 直觉:一次真正有用的检索,应该让"正确答案"在模型眼里变得更可能;没用的检索,概率不变甚至下降。

### 4.3 工程难点:T 次 forward 压成 1 次(vectorized GT logprob)

朴素做法:轨迹有 T 轮,就要跑 T 次前向(每轮各拼一次 GT 算概率)。IGPO 的加速(`compute_all_turns_vectorized`,`:377`):

- 把序列扩展成 `[原始轨迹 | GT_0 | GT_1 | ... | GT_{T-1}]`,一次前向算完所有轮(`:147` 构造扩展序列)。
- 用**定制 4D attention mask** 保证 `GT_t` 只能看到"原始轨迹到第 t 轮结束"的 token、以及它自己(因果),看不到别的 GT 拷贝(`build_extended_attention_mask`,`:185`)。
- 用**定制 position_ids** 让每个 `GT_t` 的位置编码从 turn_t 末尾接续,保证 RoPE 正确(`:250`)。
- **FlashAttention-2 不支持任意 4D mask**,代码自动回退到顺序实现(`_check_4d_attention_support`,`:348`;回退 `:414`),并带一个 `validate_vectorized_vs_sequential`(`:570`)在首个 batch 校验两条路径数值一致(`rtol=1e-4`)。

> 这段是很好的"工程深度"素材:面试可讲"我知道为什么慢(T 次前向)、怎么优化(扩展序列 + 4D mask 单次前向)、以及优化的正确性风险(FA2 不兼容)和我怎么兜底(自动回退 + 数值校验)"。

### 4.4 奖励怎么落到 token 上:每轮末 token 承接

`info_gain.py:compute_score`(`:162`)把奖励铺到 token 序列:

- 用分隔符 `"\n<|im_start|>assistant\n"` 切出每一轮(`:209`)。
- **每一轮的最后一个 token** 放该轮的 info_gain 奖励;**最后一轮的最后一个 token** 放最终 F1/outcome 奖励(`:267`–`:291`)。
- 用 offset_mapping 做 char→token 的精确定位(`_char_pos_to_token_idx`,`:141`),避免 subword 错位。

### 4.5 优势归一化:过程奖励与结果奖励分开归一(separate)

`core_algos.py:compute_grpo_outcome_advantage`(`:189`):

- 构造两个 mask:`f1_mask`(每条轨迹最后一个有效 token,承接 outcome)与 `ig_mask`(其余非零过程奖励位)(`:229`)。
- `info_gain_norm_mode=separate`(本项目):**F1 奖励和 info_gain 奖励各自在组内独立做 (x-mean)/std 归一**(`:285`–`:304`)。
  - 为什么要 separate:两种奖励量纲差异大(概率差通常 ≪ F1 的 0/1),混在一起归一会让小量纲的过程奖励被淹没。分开归一让两路信号各自有效。
- 之后按轮做 discounted 累积再 broadcast 回该轮所有 token(`_compute_turn_level_advantage`,`:322`),`gamma=1.0`。
- 另配 `curriculum`(`:232`)可随训练衰减过程奖励权重(本项目未启用,`ig_init=1.0→final=0.2` 留作后续)。

### 4.6 IGPO 的定位(诚实)

实测到 step_200:BrowseComp-Plus 29.0% Acc / 40.3% Recall,已超同期 GRPO(step_250/300)。**价值命题是"更高样本效率 + 更好证据召回",不是更高天花板**,完整对照(训到 step_400 与同 step GRPO 比)是明确的下一步。这一节讲清"机制正确 + 早期信号为正 + 严谨的下一步"即可,不要吹成已完成结论。

---

## 附:面试高频问答速查

| 可能的追问 | 一句话答案 |
|---|---|
| response_mask 是什么? | 与响应等长的 0/1,1 才算策略梯度;工具/摘要返回恒为 0。所有过滤/信用分配都是在改这个 mask。 |
| 信用分配代码具体改了哪? | 基类异常时直接 `return TERMINATED`(前面轮 mask 仍为 1=连坐);我加 `_mask_previous_response_tokens` 在终止前把本轮之前的 mask 清零,只罚肇事轮。 |
| 为什么信用分配能提 Recall? | 解除连坐 → 模型不再"怕搜错就少搜" → 敢多轮搜索(10→14)→ 召回更多证据。 |
| self-summary 会不会让模型学写摘要? | 不会,摘要 token mask=0,不进 loss;模型只学"基于摘要继续决策"。 |
| 摘要用哪个模型? | policy 模型自身(同一 vLLM server),独立 request_id 与采样参数;失败回退原文,prompt 强制"证据不足说 Insufficient"防幻觉。 |
| 长程为什么原来会崩? | rollout 端轨迹长度/轮数失控撑爆 context;治法是状态机里加长度/轮数守卫,越界即"整条 mask 置 0"优雅终止。 |
| IGPO 的过程奖励是什么? | 每轮末模型对 GT 答案概率的提升($e^{\overline{\log P_t}}-e^{\overline{\log P_{t-1}}}$),放在该轮最后一个 token。 |
| IGPO 为什么要 4D mask? | 为把 T 次 GT 前向压成 1 次:扩展序列 + 4D mask 让每个 GT 拷贝只看到对应轮的前缀;FA2 不支持则自动回退并数值校验。 |
| 同步还是异步? | 训练是同步 GRPO(main_ppo,hybrid engine,8 卡共用);`rollout.mode=async` 只是生成端并发,别混淆。 |

---

*本文档基于源码逐行核对整理,行号对应仓库当前版本;IGPO 相关文件在 `report/IGPO_ref/`。*
