# SearchAgent-Zero 面向长程搜索的核心代码改动讲解

> 本文基于**直接源码对照**整理:对照对象为上游原版 verl(`/mmu_mllm_hdd_2/zhoujinchang/verl`,v0.9.0.dev)与本仓库(v0.8.0.dev)。
> 版本漂移带来的无关差异一律忽略,只保留**与长程搜索直接相关**的重点改动,并把关键代码片段提取出来逐一讲解逻辑与功能。
> 文中所有行号对应本仓库当前版本。

---

## 0. 全局背景:改在哪一层

分层关系(从底到上):

```
verl(RL 训练框架:FSDP2 actor + vLLM async rollout + GRPO/PPO)
  └─ AgentLoop(多轮 agent rollout 抽象:verl/experimental/agent_loop/)
       └─ ToolAgentLoop(“LLM ↔ 工具”多轮状态机)
            └─ 本项目的增强:异常检测 + self-summary + 长度守卫 + 轮级信用分配
```

**一个必须先讲清的事实**:状态机架构(`AgentState` 枚举、`run()` 主循环、`_handle_pending / _handle_generating / _handle_processing_tools` 三态)在**上游 verl 里就已存在**,不是本项目发明的。本项目的工作是在这套状态机内部做增强。

**理解一切改动的钥匙 —— `response_mask`**:与 `response_ids` 等长的 0/1 序列。

```
responses:     |<-第1轮生成->|<-工具返回->|<-第2轮生成->|<-工具返回->| ...
response_mask: | 1,1,...,1,1 | 0,0,...,0,0 | 1,1,...,1,1 | 0,0,...,0,0 | ...
                    ↑算 loss       ↑仅上下文      ↑算 loss       ↑仅上下文
```

- `1` = 该 token 参与策略梯度(算 loss);`0` = 只当上下文,不算 loss。
- 模型生成的 token 默认置 1;工具返回 / 摘要结果一律置 0。
- **本项目所有“过滤 / 信用分配 / 优雅终止”本质上都是在改这个 mask 里哪些 1 该变成 0。**

---

## 动机:为什么 search-r1 只能搜 3~4 轮,本项目能搜几十轮还能稳定训练

这其实是**三个不同的瓶颈**,分别对应三招改动。关键配置:`max_model_len=20000`、`enable_tool_response_summary=True`、`summary_max_tokens=1024`、`max_assistant_turns=100`。

| 目标 | 靠哪招(对应改动) | 机制 |
|---|---|---|
| **能搜几十轮** | self-summary 压缩(改动 3) | 检索原文不全量拼回,先压成 ≤1024 token 摘要再拼回,每轮 context 增量降一个数量级 |
| **不崩溃** | 长度/轮数护栏 + 越界置零(改动 1) | 失控轨迹优雅终止(整条 mask 置 0),不报错、不 OOM、不污染梯度 |
| **能正常训练** | 轮级信用分配(改动 5) | 坏轮不连坐好轮,否则模型会学成“少搜为妙”,轮数被负梯度压回去 |

**算术直觉**(20000 context 预算):

```
search-r1(原文全量拼回,每轮 ~3000 token):
  2048 + n×(300 生成 + 3000 原文) ≈ 20000  →  n ≈ 5 轮就撑满 → 截断丢证据 / 超长崩溃

本项目(摘要拼回,每轮 ~500 token):
  2048 + n×(300 生成 + ~500 摘要) ≈ 20000  →  n ≈ 22+ 轮
```

search-r1 卡在 3~4 轮,是因为它**只有“原文全量拼回”一种策略、没有护栏、且异常连坐**——三个瓶颈同时存在。本项目分别打通:**压缩解决“装得下”,护栏解决“不崩”,信用分配解决“训得对”。三者缺一,几十轮训练都跑不起来。**

---

## 前置:搜索 rollout 的 AgentLoop 到底是怎么实现的(原生骨架 vs 本项目增强)

在讲具体改动前,必须先把「一条搜索轨迹是怎么被生成出来的」这条主链讲透。**结论先行**:多轮 rollout 的整套派发机制和状态机骨架**都是上游 verl 原生的**,本项目没有另造一套 rollout 引擎;本项目做的是**在原生状态机的三个 handler 内部插入长程搜索逻辑**,并通过一个 YAML 把 `tool_agent` 这个名字重绑到派生子类 `tool_agent_loop_credit_assignment.ToolAgentLoop`。

### (1) 从 trainer 到单条轨迹的派发链路(全部原生)

```
ray_trainer.fit()                                             【原生】
 └─ async_rollout_manager.generate_sequences(gen_batch)       【原生】
     └─ AgentLoopManager.generate_sequences (agent_loop.py:1214)      【原生】
         切分 batch → 分发到各 Ray AgentLoopWorker → 汇总
         └─ AgentLoopWorker.generate_sequences (agent_loop.py:533)    【原生】
             每条样本起一个 asyncio task
             └─ AgentLoopWorker._run_agent_loop (agent_loop.py:615)   【原生】
                 hydra.instantiate(_target_)  ← 由 YAML 决定实例化哪个类
                 └─ await agent_loop.run(...)                         ← 进入本项目子类
                     ToolAgentLoop.run (tool_agent_loop_credit_assignment.py:327)  【新增子类】
```

**类是怎么被选中的(关键机制,原生但配置是新增)**:
- `agent_loop.py` 里有一个模块级注册表 `_agent_loop_registry` + `@register(name)` 装饰器。base `tool_agent_loop.py` 用 `@register("tool_agent")` 注册了自己(且被 `__init__.py` eager import,所以默认 `tool_agent` 指向 base)。
- 但 `AgentLoopWorker.__init__`(`agent_loop.py:513-518`)会读取配置项 `agent.agent_loop_config_path` 指向的 YAML,**用其内容覆盖注册表**。
- ASearch 训练脚本设了 `AGENT_LOOP_CONFIG=examples/search_agent_rl/config/agent_loop/tool_agent_credit_assignment.yaml`,其内容:
  ```yaml
  - name: tool_agent
    _target_: verl.experimental.agent_loop.tool_agent_loop_credit_assignment.ToolAgentLoop
  ```
  于是 `tool_agent` 被重绑到 credit_assignment 子类。再配合脚本里 `default_agent_loop=tool_agent`,`hydra.instantiate` 就会实例化本项目的 `ToolAgentLoop`。
- **若不给这个 YAML**(如上游默认 `agent_loop_config_path: null`、`default_agent_loop=single_turn_agent`),既不会走多轮 tool loop,也不会走 credit_assignment 版。所以「跑的是哪套 rollout」完全由脚本配置决定,代码里两个 `@register("tool_agent")` 并存不冲突。

### (2) 状态机本体:四态 + `run()` 主循环(骨架原生,逐字对齐上游)

上游 base 和本项目子类的 `run()` 结构**完全一致**(已源码核对:上游 `tool_agent_loop.py:49/125/166-171`,本项目 `:123/327/367-373`):

```python
# tool_agent_loop_credit_assignment.py:123
class AgentState(Enum):
    PENDING = "pending"; GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"; TERMINATED = "terminated"

# :366  run() 的核心循环 —— 与上游逐字相同
state = AgentState.PENDING
while state != AgentState.TERMINATED:
    if state == AgentState.PENDING:
        state = await self._handle_pending_state(agent_data, sampling_params)      # 组 prompt
    elif state == AgentState.GENERATING:
        state = await self._handle_generating_state(agent_data, sampling_params)   # 模型生成一轮 + 解析 tool_call
    elif state == AgentState.PROCESSING_TOOLS:
        state = await self._handle_processing_tools_state(agent_data)              # 执行工具 + 拼回结果
    else:
        state = AgentState.TERMINATED
```

一轮搜索的状态流转:

```
PENDING ──组prompt──▶ GENERATING ──模型产出<thought><tool_call>──▶ PROCESSING_TOOLS
                          ▲                                              │
                          │            执行search、(可选)摘要、拼回上下文  │
                          └──────────────────────────────────────────────┘
                                            (下一轮)
   任一步命中护栏/异常/<answer> ─────────────────────────────▶ TERMINATED
```

- `PENDING`(`_handle_pending_state`,`:412`):`apply_chat_template` 把 messages+工具 schema 拼成 `prompt_ids`,进入 GENERATING。**基本是原生逻辑。**
- `GENERATING`(`_handle_generating_state`,`:424`):调 vLLM 生成一段 assistant 文本,解析其中的 `<tool_call>`。若出现 `<answer>`/无 tool_call → TERMINATED;否则 → PROCESSING_TOOLS。**这里被插入了改动 1(长度/轮数守卫)、改动 4(query 异常检测)、改动 5(信用分配掩码)。**
- `PROCESSING_TOOLS`(`_handle_processing_tools_state`,`:581`):真正调用 `search_tool` 执行检索,把结果(或摘要)以 `mask=0` 拼回 `prompt_ids`,回到 GENERATING 开下一轮。**这里被插入了改动 3(self-summary)、改动 1(c)(拼接前守卫)、改动 4(结果重复检测)、改动 9(记录 igpo 轮边界)。**

### (3) `run()` 收尾:把一条轨迹打包成 `AgentLoopOutput`(骨架原生,字段新增)

`run()` 末尾(`:378-410`)从 `agent_data` 切出 `prompt_ids` / `response_ids` / `response_mask`,组装 `AgentLoopOutput`。**组装动作原生**,但本项目往里塞了新增字段:`all_call_tool_counts`、`all_call_tool_success_counts`、`abnormal_trajectory_dic`,以及 `enable_igpo` 时的 `igpo_turn_end_indices` / `igpo_ground_truth`(`:405-409`)。这些字段随后由 `_postprocess` 展开进 `non_tensor_batch`(见改动 7)。

### (4) 原生 vs 新增(rollout 层)一句话总结

| 组件 | 原生 / 新增 | 说明 |
|---|---|---|
| `AgentLoopManager` / `AgentLoopWorker` / 派发 | **原生** | 切分 batch、Ray 分发、asyncio、汇总,全套沿用 |
| 注册表 + `@register` + `agent_loop_config_path` 覆盖 | **原生机制** | 选类机制是上游的;ASearch 提供的覆盖 YAML 是新增配置 |
| `AgentState` 四态 + `run()` 主循环 + 三 handler 骨架 | **原生** | 逐字对齐上游 `tool_agent_loop.py` |
| base `tool_agent_loop.py`(552→927 行) | **重写增强** | 骨架来自上游,长程逻辑是本项目新增(改动 1~4) |
| `tool_agent_loop_credit_assignment.py`(954 行) | **新文件** | 派生自 base,只加信用分配 + igpo 轮边界(改动 5、9) |

一句话:**rollout 引擎和状态机是 verl 原生的地基;SearchAgent-Zero 的长程搜索能力是灌在这三个 handler 内部的“增强层”,并通过一个 YAML 把入口切到 credit_assignment 子类。** 下面的改动 1~9 就是逐一拆解这些增强层。

---

## 改动清单(先看全景)

| # | 主题 | 文件 | 关键函数 / 位置 |
|---|---|---|---|
| 1 | 长度/轮数硬边界 + 越界置零终止 | `tool_agent_loop.py` | `_handle_generating_state`、`_handle_processing_tools_state` |
| 2 | 轮数课程调度 turn_limit_schedule | `tool_agent_loop.py` | `_parse_turn_limit_schedule` 等 5 个方法 |
| 3 | self-summary 上下文压缩 | `tool_agent_loop.py` | `_handle_processing_tools_state`、`generate_single_summary_self` |
| 4 | 异常轨迹检测 + 埋点 | `tool_agent_loop.py` | `AgentData.__init__`、`_all_search_results_are_duplicate` 等 |
| 5 | 轮级信用分配 | `tool_agent_loop_credit_assignment.py`(新文件) | `_mask_previous_response_tokens` |
| 6 | 检索工具 | `search_tool.py`(新文件) | `SearchTool`、`perform_search_remote` |
| 7 | 指标数据流打通 | `agent_loop.py`、`metric_utils.py` | `AgentLoopOutput`、`_postprocess`、指标聚合 |
| 8 | 配置字段注册 | `rollout.py`、`rollout.yaml` | `MultiTurnConfig` |
| 9 | IGPO 过程奖励(step 级信用分配) | `igpo_gt_logprob.py`(新文件)、`core_algos.py`、`ray_trainer.py`、`tool_agent_loop_credit_assignment.py` | `IGPORewardBuilder`、`compute_grpo_igpo_advantage`、`_compute_igpo_info_gain` |

---

## 改动 1:长程稳定性 —— 长度/轮数硬边界 + “越界即优雅终止”

**文件**:`verl/experimental/agent_loop/tool_agent_loop.py`
**位置**:`_handle_generating_state`(`:408`)、`_handle_processing_tools_state`(`:726`)

### 痛点
原框架在几十轮长轨迹上崩溃,根因是**轨迹长度失控**:context 撑爆 `max_model_len`,或轮数无限增长把 vLLM 的 KV cache / context 打爆。

### 做法:在状态机内插入多道守卫,一旦越界立即安全终止

**(a) 生成前守卫** —— prompt 已逼近 max_model_len:

```python
# _handle_generating_state 开头,:412
if self.max_model_len is not None and len(agent_data.prompt_ids) >= self.max_model_len - 1:
    agent_data.response_mask = [0] * len(agent_data.response_mask)          # 整条 mask 置 0
    agent_data.abnormal_trajectory_dic["too_long_seq_truncated_count"] += 1 # 埋点计数
    return AgentState.TERMINATED
```

**(b) 生成后守卫** —— response 超长 / 轮数超限:

```python
# :459
if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
    agent_data.response_mask = [0]*len(agent_data.response_mask)
    agent_data.abnormal_trajectory_dic['too_long_seq_truncated_count'] += 1
    return AgentState.TERMINATED

max_assistant_turns, max_user_turns = self._effective_turn_limits(agent_data, output.extra_fields)
if max_assistant_turns and agent_data.assistant_turns >= max_assistant_turns:   # :464
    agent_data.response_mask = [0]*len(agent_data.response_mask)
    agent_data.abnormal_trajectory_dic['too_many_turn_count'] += 1
    return AgentState.TERMINATED
```

**(c) 拼接工具返回前守卫** —— 预判加上工具返回会超长:

```python
# _handle_processing_tools_state,:726
if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
    agent_data.response_mask = len(agent_data.response_mask) * [0]
    agent_data.abnormal_trajectory_dic['too_long_seq_truncated_count'] += 1
    return AgentState.TERMINATED
```

### 关键设计
失控轨迹**不是报错崩溃,而是“优雅终止 + 整条 mask 置 0”**:
- 训练不会因为一条畸形轨迹而炸掉;
- 这条轨迹的所有 token 都不算 loss,**不污染梯度**。

这是“长程不崩”最核心的一招。注意:这类“整体失控”异常之所以整条置 0(而不是只留某一轮),是因为它没有“某一轮该负责”的语义 —— 与改动 5 的信用分配形成刻意的对比。

---

## 改动 2:长程轮数课程调度 —— turn_limit_schedule

**文件**:`tool_agent_loop.py`
**新增方法**:`_parse_turn_limit_schedule`(`:236`)、`_resolve_turn_limit_step`(`:273`)、`_scheduled_turn_limit`(`:287`)、`_cap_turn_limit`(`:298`)、`_effective_turn_limits`(`:306`)

### 功能
允许把“允许的最大轮数”做成**随训练 step 变化的课程**,例如配置 `"0:100,50:100"` 表示 step≥0 用 100 轮、step≥50 仍是 100 轮(也可配成前期少轮、后期多轮)。

### 实现逻辑

**解析配置字符串**(把 `"step:limit,step:limit"` 解析成有序里程碑列表):

```python
# _parse_turn_limit_schedule,:236
for raw_item in schedule.split(","):
    step_text, limit_text = item.split(":", maxsplit=1)
    step, limit = int(step_text), int(limit_text)
    milestones.append((step, limit))
milestones.sort(key=lambda x: x[0])   # 按 step 升序,校验非负/正数/无重复
```

**按当前 step 查表**(取最后一个满足 `step >= milestone_step` 的 limit):

```python
# _scheduled_turn_limit,:287
active_limit = self.turn_limit_schedule[0][1]
for milestone_step, limit in self.turn_limit_schedule:
    if step < milestone_step:
        break
    active_limit = limit
return active_limit
```

**动态限制与静态上限取 min**,得到本轮真正生效的轮数上限:

```python
# _effective_turn_limits,:306
dynamic_limit = self._scheduled_turn_limit(step)
assistant_limit = self._cap_turn_limit(dynamic_limit, self.max_assistant_turns)  # min(动态, 静态)
user_limit      = self._cap_turn_limit(dynamic_limit, self.max_user_turns)
return assistant_limit, user_limit
```

当前 step 从 `output.extra_fields` 或 `agent_data.extra_fields` 里的 `global_steps` 系列字段解析(`_resolve_turn_limit_step`,`:273`)。生效点就是改动 1(b) 的轮数守卫。

---

## 改动 3:self-summary 上下文压缩(长程能搜更多轮的关键)

**文件**:`tool_agent_loop.py`
**位置**:`_handle_processing_tools_state`(`:607`)、`_generate_single_summary`(`:854`)、`generate_single_summary_self`(`:895`)、`PROMPT_TEMPLATE`(`:45`)、`_truncate_text_by_tokens`(`:787`)

### 痛点
每次 `search` 返回若干篇长文档,若原样拼进 context,几轮就把 `max_model_len` 填满 —— 改动 1 的守卫会因此提前终止,搜不了几轮。

### 思路
检索结果不原样拼回,而是**先让模型把“这批文档对当前 query 的答案”压成一段短摘要,再拼回上下文**。

### 实现:逐 query 并发摘要 + 失败回退

```python
# _handle_processing_tools_state,:607
if self.enable_tool_response_summary and tool_call.name == "search" and tool_response_text and query_list:
    # 把一次 search 的多个 query 结果按分隔符切开
    tool_response_text_lst = json.loads(tool_response_text)['result'].split(self.summary_result_separator)

    summary_tasks = []
    for q_idx, (query, doc_item) in enumerate(zip(query_list, tool_response_text_lst)):
        # 摘要前先按 token 截断超长原文,防止摘要输入本身爆掉
        doc_item, truncated = await self._truncate_text_by_tokens(
            doc_item, self.max_tool_response_length, self.tool_response_truncate_side)
        if truncated:
            agent_data.abnormal_trajectory_dic['response_truncated_count'] += 1
        summary_tasks.append(self._generate_single_summary(query, doc_item, q_idx + 1))

    summary_results = await asyncio.gather(*summary_tasks) if summary_tasks else []  # 并发摘要
    all_summary_text = self.summary_result_separator.join(summary_results)

    if all_summary_text:
        message_text = all_summary_text          # 用摘要替换原文
    else:
        message_text, _ = await self._truncate_text_by_tokens(...)   # 摘要为空 → 回退截断原文
```

要点:
- **每个 query 单独摘要并 `asyncio.gather` 并发**,不串行,省时间;
- **摘要失败/为空 → 回退到截断后的原文**,保证不因摘要挂掉丢整轮;
- **截断双保险**:摘要前先按 token 截断,记 `response_truncated_count`。

### 用模型自身做摘要(默认,不额外起模型)

`_generate_single_summary`(`:854`)按 `summary_use_external_model` 分流;默认 `False`,走 `generate_single_summary_self`:

```python
# generate_single_summary_self,:895
summary_prompt = PROMPT_TEMPLATE.format(query=query, documents=document)
tool_response_prompt_ids = await self.apply_chat_template(
    [{"role": "user", "content": summary_prompt}], ...)
sampling_params = dict(temperature=self.summary_temperature, top_p=self.summary_top_p,
                       top_k=self.summary_top_k, repetition_penalty=1.0, logprobs=False)
summary_req_id = uuid4().hex                                  # 独立 request_id,与主生成解耦
output = await self.server_manager.generate(                 # 复用同一 vLLM server
    request_id=summary_req_id, prompt_ids=tool_response_prompt_ids, sampling_params=sampling_params, ...)
summary_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
```

- **同一 policy 模型兼做 summarizer**(同一个 vLLM server)→ 省显存、省部署;
- 用独立 request_id + 独立采样参数,与主生成互不干扰;
- 也支持外部模型(`generate_single_summary`,`:859`,轮询 OpenAI 客户端,`_init_summary_external_clients` `:823` 初始化)。

### 防幻觉的 PROMPT_TEMPLATE(`:45`)
强约束“先 Reasoning 再 Summary”,且**证据不足必须显式输出 "Information Insufficient",禁止编造**:

```
- If the documents are irrelevant, empty, or do not provide enough context to answer the query,
  the Summary must explicitly state: "Information Insufficient".
- Do not attempt to fabricate an answer or use outside knowledge not present in the documents.
```

### 摘要 token 算不算 loss?—— 不算
摘要作为 tool 消息拼回时(`:741`):

```python
agent_data.prompt_ids += response_ids
agent_data.response_mask += [0] * len(response_ids)   # 摘要/工具返回一律 mask=0
```

摘要是“环境观测”的一部分,不参与策略梯度。**模型学的是“如何基于摘要继续决策”,而不是“如何写摘要”。**

---

## 改动 4:异常轨迹检测 + 全程埋点

**文件**:`tool_agent_loop.py`
**位置**:`AgentData.__init__`(`:130`)、`_handle_generating_state`(`:478`)、`_extract_search_result_signatures`(`:508`)、`_search_result_overlap`(`:538`)、`_all_search_results_are_duplicate`(`:544`)

### 新增状态字段(`AgentData.__init__`)

```python
self.searched_query = set()                        # 已搜过的(归一化)query 集合
self.searched_result_signatures: list[set[str]] = []  # 历史检索结果的文档签名
self.all_call_tool_counts = 0                      # 工具调用总次数
self.all_call_tool_success_counts = 0              # 工具调用成功次数
self.abnormal_trajectory_dic = {                   # 7 类异常计数(既是监控指标也是过滤依据)
    'searched_query_count': 0,           # 重复 query
    'tool_parser_error_count': 0,        # 工具解析失败(query_list 非法/JSON 错)
    'too_many_tool_call_count': 0,       # 单轮并行 query 过多
    'too_many_turn_count': 0,            # 轮数超限
    'response_truncated_count': 0,       # 工具返回被截断
    'too_long_seq_truncated_count': 0,   # 轨迹超长截断
    'duplicate_search_result_count': 0,  # 检索结果整体重复
}
```

### query 类异常检测(生成后,解析 tool_call 时,`:478`)

```python
query_list = parsed_args.get("query_list")
if not query_list or not isinstance(query_list, list):        # 解析失败
    agent_data.abnormal_trajectory_dic['tool_parser_error_count'] += 1
    return AgentState.TERMINATED
if self.max_queries_per_tool_call and len(query_list) > self.max_queries_per_tool_call:  # 并行过多
    agent_data.abnormal_trajectory_dic['too_many_tool_call_count'] += 1
    return AgentState.TERMINATED
for query in query_list:
    query = normalize_answer(query)                           # 归一化后判重
    if query in agent_data.searched_query:
        agent_data.abnormal_trajectory_dic['searched_query_count'] += 1
        return AgentState.TERMINATED
    else:
        agent_data.searched_query.add(query)
```

其中 `normalize_answer`(`:73`,从 Search-R1 搬来)做小写化、去标点、去冠词、规范空白。

### 检索结果“整体重复”检测(工具返回后,`:596`)
用文档签名集合 + Jaccard-min 重叠判定:

```python
# _search_result_overlap,:538  —— 交集 / 较小集合大小
return len(first & second) / min(len(first), len(second))

# _all_search_results_are_duplicate,:544  —— 本轮每个签名都与某历史签名重叠 ≥ 2/3 才算“整体重复”
for current_signature in current_signatures:
    if not any(self._search_result_overlap(current_signature, hist) >= overlap_threshold
               for hist in historical_signatures):
        return False
return True
```

`_extract_search_result_signatures`(`:508`)按 `Doc N (Title: ...)` 切分文档、归一化成签名集合。命中时 `duplicate_search_result_count += 1`(注意此项当前只计数、不终止,`:604` 的 `return TERMINATED` 被注释掉了)。

**功能定位**:这些计数在训练时会被聚合成监控指标(见改动 7),让 Search Agent 训练过程**可观测**;同时 query 类异常触发终止,是改动 5 信用分配的作用对象。

---

## 改动 5:轮级信用分配(核心)—— tool_agent_loop_credit_assignment.py

**文件**:`verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py`(**新文件**,派生自上面的 base)

这个文件相对 base 的差异**极小且极精准**,全部差异如下:

```python
# 1) 新增静态方法,:315
@staticmethod
def _mask_previous_response_tokens(agent_data: AgentData, current_turn_start: int) -> None:
    agent_data.response_mask[:current_turn_start] = [0] * current_turn_start   # 本轮之前全部置 0

# 2) 每轮生成开始时记录本轮起点,:455
current_turn_start = len(agent_data.response_mask)

# 3) 四个 query 类异常终止点,终止前先调用掩码(:494 :499 :505 :511)
if query in agent_data.searched_query:
    agent_data.abnormal_trajectory_dic['searched_query_count'] += 1
    self._mask_previous_response_tokens(agent_data, current_turn_start)  # ← 只罚这一轮
    return AgentState.TERMINATED
```

### “连坐” vs “信用分配”

- **base `tool_agent_loop.py`(连坐)**:检测到 query 异常直接 `return TERMINATED`,**前面所有正确轮的 `response_mask` 仍是 1** → 整条轨迹带着“坏 reward”进 loss。
- **本文件(信用分配)**:终止前先把 `response_mask[:current_turn_start]` 清零,**只留下出问题的当前轮**参与 loss。

### 为什么这样就实现了“信用分配”
GRPO 的优势/奖励最终只作用在 `response_mask==1` 的 token 上。设想一条轨迹前 5 轮搜得都对、第 6 轮重复 query 踩了异常:

| | 前 5 轮正确搜索 | 第 6 轮(肇事轮) | 后果 |
|---|---|---|---|
| 连坐(base) | mask=1 | mask=1 | 低 reward 拉低前 5 轮梯度 → 模型学到“少搜为妙” |
| 信用分配(本文件) | mask=0 | mask=1 | 惩罚精确落在肇事轮 → 模型**敢继续多轮搜索** |

这解释了“解除连坐 → 检索轮数变多 → 召回更多证据”的机制。

### 三个易被追问的细节
1. **为什么切片是 `[:current_turn_start]`?** 因为 `current_turn_start` 恰好是“本轮第一个生成 token 的下标”,前缀切片 = 本轮之前的全部内容(历史生成 + 历史工具返回),清零后只剩当前异常轮。
2. **工具返回本来就是 0,清零它有害吗?** 无害,幂等 —— 它们本就是 0。
3. **为什么超长/超轮是“整条置 0”而非“只留本轮”?** 那类是“轨迹整体失控”,没有“某一轮该负责”的语义,整条作废最干净;query 类异常才有明确“肇事轮”,才用信用分配。**这个区分是设计精髓。**

---

## 改动 6:检索工具 —— search_tool.py

**文件**:`verl/tools/search_tool.py`(**新文件**,上游无)

### 结构
- `SearchTool(BaseTool)`(`:211`):`execute`(`:259`)接收 `query_list`,委托给 Ray remote task,返回 `(ToolResponse, reward, metadata)`。
- 模块级 `perform_search_remote`(`:87`,`@ray.remote`):HTTP POST 到 `retrieval_service_url`,带超时与重试。
- `GlobalRateLimiter`(`:68`):全局限流,避免检索服务被打爆。

### execute 核心(`:259`)

```python
query_list = parameters.get("query_list")
query_list_len = len(query_list)
if not query_list or not isinstance(query_list, list):        # 参数校验也是一种错误
    return ToolResponse(text=json.dumps({"result": msg})), 0.0, {"error_code": 0, "query_list_len": query_list_len}

future = perform_search_remote.remote(
    query_list=query_list, url=self.retrieval_service_url,
    topk=self.topk, timeout=self.timeout, rate_limiter=self.rate_limit_actor)
result_text, metadata = await future
metadata['query_list_len'] = query_list_len                   # 供上层统计并行 query 数与成功率
```

metadata 里的 `error_code`(1=成功)、`query_list_len` 被 `_handle_processing_tools_state`(`:591`)用来累加 `all_call_tool_success_counts` / `all_call_tool_counts`。

---

## 改动 7:指标数据流打通(让新指标进训练/日志)

### (a) AgentLoopOutput 新增字段 —— `verl/experimental/agent_loop/agent_loop.py`

```python
# class AgentLoopOutput(BaseModel),:213
all_call_tool_counts : int = 0
all_call_tool_success_counts : int = 0
abnormal_trajectory_dic : dict[str, int] = {}
```

`run()` 结束时把 `agent_data` 上的对应字段填进 `AgentLoopOutput`(credit_assignment 文件 `:388`)。

### (b) `_postprocess` 把异常计数展开进 non_tensor_batch(`:982`)

```python
abnormal_trajectory_dics = [input.abnormal_trajectory_dic for input in inputs]
if abnormal_trajectory_dics and abnormal_trajectory_dics[0]:
    for key in abnormal_trajectory_dics[0].keys():
        non_tensor_batch[key] = np.array([dic[key] for dic in abnormal_trajectory_dics], dtype=np.int32)
```

### (c) 指标聚合 —— `verl/trainer/ppo/metric_utils.py`(`:260`)

```python
# 工具调用成功率
metrics["turn/tool_call_success_rate/mean"] = tool_tool_success_sum / all_call_tool_counts_sum
# 各类异常轨迹占比
metrics["abnormal_trajectory/searched_query_count_percentage"] = searched_query_count.sum() / len(...)
metrics["abnormal_trajectory/too_many_tool_call_count_percentage"] = ...
metrics["abnormal_trajectory/tool_parser_error_count_percentage"] = ...
metrics["abnormal_trajectory/response_truncated_count_percentage"] = ...
metrics["abnormal_trajectory/too_many_turn_count_percentage"] = ...
metrics["abnormal_trajectory/too_long_seq_truncated_count_percentage"] = ...
```

这样 README 里提到的“工具调用成功率、平均搜索轮数、重复 query、过度并行、解析失败、轨迹截断”等 Search-Agent 专属指标就能进 wandb / 日志监控。

---

## 改动 8:配置字段注册

**文件**:`verl/workers/config/rollout.py`(`MultiTurnConfig`)+ `verl/trainer/config/rollout/rollout.yaml`

新增约 20 个 multi_turn 字段,让 Hydra 接受训练脚本里的 override。核心几个:

| 字段 | 作用 | 对应改动 |
|---|---|---|
| `turn_limit_schedule` | 轮数课程调度字符串 | 改动 2 |
| `max_queries_per_tool_call` | 单轮并行 query 上限 | 改动 4 |
| `enable_tool_response_summary` | 是否开启 self-summary | 改动 3 |
| `summary_use_external_model` | 摘要用外部模型还是自身 | 改动 3 |
| `summary_temperature/top_p/top_k/max_tokens` | 摘要采样参数 | 改动 3 |
| `summary_external_base_urls/model/api_key/...` | 外部摘要模型配置 | 改动 3 |
| `summary_result_separator` | 多 query 结果分隔符 | 改动 3/4 |
| `duplicate_search_result_overlap_threshold` | 结果重复判定阈值 | 改动 4 |
| `max_tool_response_length` / `tool_response_truncate_side` | 工具返回截断长度与方向 | 改动 3 |

---

## 改动 9:IGPO 过程奖励 —— 从“轮级 mask 信用分配”到“step 级过程奖励”

**核心文件**:
- `verl/utils/igpo_gt_logprob.py`(**新文件**,346 行,IGPO 奖励计算的纯逻辑核心)
- `verl/trainer/ppo/core_algos.py`(`grpo_igpo` 优势估计器)
- `verl/trainer/ppo/ray_trainer.py`(训练循环里的编排 + GPU 前向)
- `verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py`(rollout 时记录轮边界)
- 配置:`verl/trainer/config/algorithm.py`、`verl/workers/config/rollout.py`

> 论文出处:IGPO(Information Gain-based Policy Optimization,arXiv:2510.14967)。
> **诚实标注**:`igpo_gt_logprob.py` 文件头注明,其原始源码曾丢失,本文件是依据编译后的 `.pyc` 符号表、`ray_trainer` 调用契约,以及 `report/IGPO_ref` 官方参考实现**重建**而成,对外接口与 trainer 调用完全对齐。

### 9.1 动机(和改动 5 一脉相承)

改动 5 的轮级信用分配解决的是“**坏轮别连坐好轮**”,但对“好轮”内部仍是一个笼统的 outcome reward —— 几十轮里**到底哪一轮真正带来了信息增益**说不清。IGPO 把信用分配从“轮级 0/1 mask”推进到“**每一轮一个连续过程奖励**”:哪一轮搜索让模型更接近正确答案,哪一轮就得正奖励。

### 9.2 过程奖励怎么定义:GT 概率的逐轮提升(teacher forcing)

核心思想(`igpo_gt_logprob.py` 文件头 + `compute_info_gain_per_turn` `:96`):第 t 轮结束后,用 **teacher forcing**(不是让模型自己生成)测量“模型此刻对正确答案的信念” —— 把 ground-truth 答案拼到“历史到第 t 轮”后面,读出 GT 答案 token 的平均 log-prob。相邻两轮之差就是该轮的信息增益奖励:

```python
# compute_info_gain_per_turn,:96  (turn 0 只做基线,不发奖励)
#   prob_diff (默认):  r_t = exp(mean_logP_t) - exp(mean_logP_{t-1})
#   log_prob_diff   :  r_t = mean_logP_t       - mean_logP_{t-1}
def to_value(lp):
    if lp is None or math.isnan(lp) or math.isinf(lp):
        return None                          # NaN/inf 安全:无效步贡献 0
    return math.exp(lp) if info_gain_type == "prob_diff" else lp

prev = to_value(turn_mean_logprobs[0])
for t in range(1, len(turn_mean_logprobs)):
    curr = to_value(turn_mean_logprobs[t])
    gains.append(0.0 if (prev is None or curr is None) else (curr - prev))
    if curr is not None:
        prev = curr
```

直觉:一次真正有用的检索,应让“正确答案”在模型眼里变得更可能($P$ 上升 → 正奖励);没用/干扰性的检索,概率不变甚至下降 → 零或负奖励。

### 9.3 关键工程:一次批量前向算完所有 (sample, turn) 的 GT 概率

朴素做法:一条轨迹有 T 轮就要跑 T 次前向。IGPO 的做法是把**所有样本、所有轮的 teacher-forcing 行拼成一个大 batch,一次前向算完**。模块被刻意拆成“纯逻辑(可 CPU 测试)+ 注入式 GPU 前向”:

**(a) 构造 teacher-forcing 行**(`build_gt_scoring_rows` `:205`):对样本 i、轮边界 e,构造一行 `prompt_i + response_i[:e] + gt_tokens_i`,并记录 GT 答案 token 的区间 `[ans_start, ans_end)`。

```python
# build_gt_scoring_rows,:236
for turn_idx, e in enumerate(boundaries):
    history = prompt + response[:e]          # 历史到第 t 轮结束
    row_ids = history + list(gt_tokens)      # 拼上 GT(前缀+答案+后缀)
    base = len(history)
    rows.append(GTScoringRow(sample_idx=i, turn_idx=turn_idx, input_ids=row_ids,
                             ans_start=base + ans_start_in_gt, ans_end=base + ans_end_in_gt))
```

- `_usable_boundaries`(`:188`)会丢弃超过 `response_length`(被截断的轮)或超出真实长度的边界;**不足 2 个可用轮的样本直接跳过**(无法构成信息增益)。
- GT 答案 token 区间用 `offset_mapping` 做 char→token 精确定位(`get_answer_token_range` `:63`),避免 subword 错位。GT 文本用固定 `gt_prefix`/`gt_suffix` 包裹(默认 `"\nNow there's enough information to answer\n</thought>\n<answer>\n"` … `"\n</answer><|im_end|>"`),模拟“模型正要给出答案”的语境。

**(b) 一次批量打分**(`IGPORewardBuilder.build` `:292`):所有行在**一个 `logprob_fn` 调用**里打分,再按样本分组、按轮序还原:

```python
# IGPORewardBuilder.build,:322
mean_logprobs = logprob_fn(rows)             # 一次批量前向,返回每行 GT 答案的平均 logP
per_sample_logps = {}                        # 按 sample_idx 分组,保持轮序
for row, lp in zip(rows, mean_logprobs):
    per_sample_logps.setdefault(row.sample_idx, []).append(float(lp))
for i, mean_logps in per_sample_logps.items():
    info_gains = compute_info_gain_per_turn(mean_logps, self.info_gain_type)
    r_row, b_row = place_info_gain_on_tokens(info_gains, boundaries, response_length)
    info_gain_reward[i] = r_row              # (bsz, response_length)
    turn_boundary_mask[i] = b_row
```

**(c) GPU 前向被注入为 `logprob_fn`** —— 这是唯一需要 GPU 的部分,由 trainer 提供,纯逻辑部分因此可用假函数做 CPU 单测。trainer 里的实现(`_igpo_score_rows` `ray_trainer.py:1281`):把每行按 verl 标准的“左 pad prompt + 右 pad response”布局摆放,`prompt = input_ids[:ans_start]`(历史+GT前缀)、`response = input_ids[ans_start:ans_end]`(答案 token),调用 `actor.compute_log_prob` 得到 `log P(answer_i | 前缀 + answer[:i])`,逐行取平均。由于行数通常不能被 dp_size 整除,会先 `pad_dataproto_to_divisor` 补齐、打分后丢掉补齐的尾行。

### 9.4 奖励落到 token 上:每轮末 token 承接

`place_info_gain_on_tokens`(`:137`)把轮级奖励铺到 token 序列:

```python
# 每个轮末位置(该轮最后一个 response token = e-1)标 1,记录轨迹的轮结构
for e in turn_end_indices:
    turn_boundary_row[min(max(e-1,0), response_length-1)] = 1.0
# info_gains[k] 是“第 k+1 轮相对第 k 轮”的增益 → 放在第 k+1 轮的末 token
for k, g in enumerate(info_gains):
    reward_row[min(max(turn_end_indices[k+1]-1,0), response_length-1)] = float(g)
```

返回 `reward_row`(每轮末的 info-gain 奖励,其余为 0)和 `turn_boundary_row`(标出所有轮末位置,供优势估计器识别轮结构)。

**轮边界从哪来?** 在 rollout 时由 credit_assignment agent loop 记录(`tool_agent_loop_credit_assignment.py:771`):每轮工具返回拼回后,把当前 `response_mask` 长度追加到 `igpo_turn_end_indices`,并随 `AgentLoopOutput`(`:407`)带出 `igpo_turn_end_indices` 和 `igpo_ground_truth`。

### 9.5 优势估计器:`grpo_igpo`(过程奖励与结果奖励分开归一)

`core_algos.py:compute_grpo_igpo_advantage`(`:386`,注册名 `grpo_igpo`)。若 IGPO 张量缺失则**安全回退**到普通 outcome GRPO(`:416`)。四步:

```python
# 1) 合并:outcome(权重 1)+ 缩放后的过程奖励
combined = token_level_rewards + coef * info_gain_reward           # :430

# 2) 两个 mask
last_valid_pos = (seq_len-1) - response_mask.flip(dims=[1]).argmax(dim=1)
f1_mask = (pos_idx == last_valid_pos.unsqueeze(1)) & (response_mask == 1)   # 每条轨迹最后一个有效 token = outcome 位
ig_mask = (turn_boundary_mask > 0) & (response_mask == 1) & (~f1_mask)      # 其余轮末 = 过程奖励位

# 3) 组内归一(GRPO 的分组 = 同一 prompt 的多条采样)
if norm_mode == "separate":     # 本项目默认
    masks = (f1_mask, ig_mask)  # F1 奖励、info_gain 奖励各自独立做 (x-mean)/std
else:                           # joint
    masks = (f1_mask | ig_mask,)
for m in masks:
    gm, gs = group_stats(m)     # scatter_add 算组均值/组标准差
    nr = combined - gm[group_ids_exp]
    if norm_adv_by_std_in_grpo:
        nr = nr / (gs[group_ids_exp] + epsilon)
    normalized = torch.where(m, nr, normalized)

# 4) 轮级折扣累积 + 广播到该轮所有 token
boundary = (f1_mask | ig_mask)
advantages = _igpo_compute_turn_level_advantage(normalized, response_mask, gamma, boundary)
```

**为什么要 `separate` 分开归一?** 两种奖励量纲差异大:概率差(prob_diff)通常远小于 F1 的 0/1。混在一起归一,小量纲的过程奖励会被 outcome 淹没;分开归一让两路信号各自有效。

**轮级折扣累积**(`_igpo_compute_turn_level_advantage` `:334`):对每条轨迹,从最后一轮往前 `A_i = r_i + γ·A_{i+1}`,再把 `A_i` 广播到第 i 轮内所有 `mask==1` 的 token。`γ` 由 `igpo_gamma`(默认 1.0)控制。

### 9.6 训练循环编排 + 配置

**编排**(`ray_trainer.py`):
- fit 循环里,当 `adv_estimator == "grpo_igpo"` 时调用 `_compute_igpo_info_gain(batch)`(`:1651`),算出 `igpo_info_gain_reward` / `igpo_turn_boundary_mask` 写回 batch;
- `compute_advantage` 里把这两个张量作为 `adv_kwargs` 传给 `grpo_igpo` 估计器(`:213`);
- `_compute_igpo_info_gain`(`:1210`)负责:用 `attention_mask` 反 pad 出每样本的 prompt/response token → 建 `IGPORewardBuilder` → 注入 `logprob_fn=self._igpo_score_rows` → 得到奖励张量。元数据缺失时写零(no-op)。

**配置**:
- `algorithm.py`(`:680`):`info_gain_type`(`prob_diff`/`log_prob_diff`)、`info_gain_norm_mode`(`separate`/`joint`)、`igpo_gamma`、`igpo_coef`,以及课程退火开关 `use_igpo_curriculum` + `igpo_ig_init/final`、`igpo_f1_init/final`(线性衰减过程奖励权重)。
- `rollout.py`(`:111`):`enable_igpo`(是否在 rollout 记录轮边界)、`igpo_gt_prefix`、`igpo_gt_suffix`。

### 9.7 IGPO 与前面改动的关系

| 层级 | 机制 | 粒度 | 作用对象 |
|---|---|---|---|
| 改动 5 轮级信用分配 | `_mask_previous_response_tokens` 清零 | 轮级 0/1 mask | 惩罚“肇事轮”,好轮不连坐 |
| 改动 9 IGPO 过程奖励 | GT 概率逐轮提升 → 每轮末连续奖励 | step 级连续值 | 奖励“真正带来信息增益的轮” |

改动 5 是“负向:别罚错”;IGPO 是“正向:精准奖对”。二者都是把信用**精确分配到具体的轮**,只是从“0/1 掩码”升级为“连续过程奖励 + 轮级折扣广播”。

---

## 总结:五个支柱

围绕长程搜索,核心代码集中在 **`tool_agent_loop.py`(重写增强)**、**`tool_agent_loop_credit_assignment.py`(轮级信用分配 + IGPO 轮边界)**、**`igpo_gt_logprob.py` / `core_algos.py`(IGPO 过程奖励)** 这几个文件,配套 `search_tool.py`(检索工具)、`agent_loop.py` / `metric_utils.py`(指标数据流)、`ray_trainer.py`(IGPO 编排)与 `rollout` / `algorithm` 配置。五个支柱:

1. **长度/轮数守卫 + 越界置零终止** → 长程 rollout 不崩;
2. **self-summary 压上下文** → 固定 context 预算下能搜更多轮;
3. **异常轨迹检测 + 埋点** → 训练过程可观测、可过滤;
4. **轮级信用分配** → 坏轮不连坐好轮,模型敢多轮搜索;
5. **IGPO 过程奖励** → 把信用从“轮级 0/1”推进到“step 级连续奖励”,精准奖励真正带来信息增益的搜索轮。

> 说明:IGPO 相关代码(`igpo_gt_logprob.py` 等)已合入本仓库,但其 `igpo_gt_logprob.py` 文件头明确标注为“原始源码丢失后的重建版本”(依据 `.pyc` 符号表、trainer 调用契约与 `report/IGPO_ref` 参考实现重建),对外接口与 trainer 调用对齐;涉及 GPU 前向的 `_igpo_score_rows` 无法 CPU 单测,其余纯逻辑均可测。

---

*本文由源码逐一核对整理;上游对照基准为 verl v0.9.0.dev,本仓库为 v0.8.0.dev,已剔除版本漂移带来的无关差异。*
