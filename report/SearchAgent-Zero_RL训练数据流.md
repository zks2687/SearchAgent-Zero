# SearchAgent-Zero RL 训练数据流讲解

> 本文讲清楚一次 RL 训练 step 里，数据从哪来、经过哪些文件/函数、到哪去，并标注每一跳是 **【原生】**（verl 自带）还是 **【新增】**（SearchAgent-Zero 为长程搜索 / IGPO 新加）。文末用一个具体的搜索问答样本从头到尾走一遍，方便对照理解。
>
> 对照基准：上游 verl 源码 `/mmu_mllm_hdd_2/zhoujinchang/verl`。
> 入口脚本：`run_qwen3_8b_instruct_search_multiturn_ASearch.sh`。

---

## 0. 全局数据流一览

```
                    ┌─────────────────────────────────────────────┐
 dataset (prompt+GT)│              fit()  单个 step                 │
        │           │  ray_trainer.py:1424  【原生骨架】            │
        ▼           └─────────────────────────────────────────────┘
  gen_batch.repeat(n)                     每个 prompt 采 n 条轨迹
        │
        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ ① Rollout：多轮搜索轨迹生成                                    │
 │   async_rollout_manager.generate_sequences                     │
 │     → AgentLoopManager → AgentLoopWorker → ToolAgentLoop.run    │
 │     （状态机【原生】 + 护栏/summary/异常检测/credit【新增】）    │
 │     工具调用 → search_tool.py【新增】→ 远端检索服务             │
 │   产出：responses / response_mask / non_tensor_batch(新增字段)  │
 └──────────────────────────────────────────────────────────────┘
        │
        ▼
 ② compute_response_mask / _balance_batch          【原生】
        │
        ▼
 ③ 奖励：extract_reward ← reward_manager ← search_r1_like_qa_em.compute_score
    EM + 格式分，写到最后一个有效 token → rm_scores  【骨架原生 / 打分函数修改】
        │
        ▼
 ④ [仅 grpo_igpo] _compute_igpo_info_gain            【新增】
    → IGPORewardBuilder.build → teacher forcing 测 P(GT)
    → 每轮置信度增益 = 过程奖励，落到该轮末 token
        │
        ▼
 ⑤ old_log_prob / ref_log_prob / (values)           【原生】
        │
        ▼
 ⑥ compute_advantage                                【分发原生】
    grpo → compute_grpo_outcome_advantage           【原生，字节一致】
    grpo_igpo → compute_grpo_igpo_advantage          【新增】
        │
        ▼
 ⑦ _update_critic / _update_actor（GRPO loss, FSDP2）【原生】
        │
        ▼
 ⑧ update_weights：actor → vLLM rollout 权重同步     【原生】
        │
        ▼
 ⑨ validate（走同样 rollout 路径）                   【原生】
```

---

## 1. 装配（启动时一次性）

| 位置 | 作用 | 标签 |
|---|---|---|
| `run_qwen3_8b_instruct_search_multiturn_ASearch.sh` | `python -m verl.trainer.main_ppo`；设 `adv_estimator=grpo`、`rollout.mode=async`、`default_agent_loop=tool_agent`、`agent_loop_config_path=.../tool_agent_credit_assignment.yaml`、`tool_config=.../search_tool_config.yaml` | 【新增】配置 |
| `verl/trainer/main_ppo.py` → `RayPPOTrainer` | 装配 trainer / worker group / rollout manager | 【原生】 |
| `ray_trainer.py:859-876` `AgentLoopManager.create` | 构建 async rollout manager（支持 FQN 覆盖成自定义 manager） | 【原生】(hook 新增) |
| `agent_loop.py:513-518` 读 `agent_loop_config_path` YAML | **关键**：把注册表里 `tool_agent` 的 `_target_` 重绑到 `tool_agent_loop_credit_assignment.ToolAgentLoop` | 【原生机制】+【新增配置】 |

> `__init__.py:25` 会先 eager import base `tool_agent_loop.py`（也 `@register("tool_agent")`）。**只有** ASearch 脚本提供了 `agent_loop_config_path`，才会覆盖成 credit-assignment 版；否则跑的是 base 类。

---

## 2. fit() 主循环骨架

`ray_trainer.py:1424 fit()`【原生骨架】。一个 step 内：

```python
_get_gen_batch → gen_batch.repeat(n)                # 每个 prompt 采 n 条
  → async_rollout_manager.generate_sequences()      # ① rollout
  → batch.union(gen_output) + batch.repeat(n)
  → compute_response_mask() / _balance_batch()       # ②
  → extract_reward()                                 # ③ 奖励 (:1580)
  → [grpo_igpo] _compute_igpo_info_gain()            # ④ IGPO (:1651)
  → old_log_prob / ref_log_prob / (values)           # ⑤
  → compute_advantage()                              # ⑥ (:1687)
  → _update_critic() / _update_actor()               # ⑦ (:1711)
  → update_weights()                                 # ⑧
  → validate()                                       # ⑨
```

---

## 3. ① Rollout：多轮搜索轨迹生成（重头戏）

调用链（`verl/experimental/agent_loop/`）：

```
generate_sequences (ray_trainer.py:1515)                        【原生】
 → AgentLoopManager.generate_sequences (agent_loop.py:1214)      【原生】
     chunk batch → worker.generate_sequences.remote → concat
 → AgentLoopWorker.generate_sequences (agent_loop.py:533)        【原生】
     补 default_agent_loop=tool_agent；每条样本一个 asyncio task
 → AgentLoopWorker._run_agent_loop (agent_loop.py:615)           【原生】
     hydra.instantiate(_target_) → credit_assignment.ToolAgentLoop
     await agent_loop.run(...)                                   ← 进入自定义逻辑
 → ToolAgentLoop.run (tool_agent_loop_credit_assignment.py:327)  【新增】
```

`ToolAgentLoop.run` 就是那套 **状态机**（`_handle_pending / _generating / _processing_tools_state`）——**状态机本身是 verl 原生**，SAZ 在里面加了长程搜索的核心改动【新增】：

- **长度/轮次硬护栏** + 溢出时"优雅终止、整条 mask 置零"
- **turn_limit_schedule** 轮次课程
- **self-summary** 上下文压缩（用策略模型自己并发总结，mask=0）
- **异常轨迹检测**：`abnormal_trajectory_dic` 7 类计数器
- **credit assignment**：`_mask_previous_response_tokens`(:315)，坏 turn 不牵连好 turn
- **IGPO 埋点**：`igpo_turn_end_indices` 每轮追加(:771)，`enable_igpo` 时把 `igpo_turn_end_indices` / `igpo_ground_truth` 塞进 `output.extra_fields`(:405-408)

工具调用经 `verl/tools/search_tool.py`【新增】：`SearchTool.execute`(:259) → `perform_search_remote`(:87, `@ray.remote` HTTP POST) → `GlobalRateLimiter`(:68) 限流打远端检索服务。

**打包成 batch**：

```
_agent_loop_postprocess (agent_loop.py:649) 【原生骨架】
   per-sample 补齐 prompt_ids/response_ids/response_mask/attention_mask/position_ids
   携带新增字段 all_call_tool_counts / all_call_tool_success_counts /
              abnormal_trajectory_dic / extra_fields(含 igpo 列)
_postprocess (agent_loop.py:929) 【原生骨架 + 新增字段打包】
   tensor batch: prompts / responses / response_mask / input_ids / ...
   non_tensor_batch:
     all_call_tool_counts / all_call_tool_success_counts          【新增】
     7 个异常子键 (searched_query_count 等)                       【新增】
     igpo_turn_end_indices / igpo_ground_truth (via extra_fields) 【新增】
   返回 DataProto(batch, non_tensor_batch, meta_info)              【原生容器】
```

> **`response_mask` 是长程稳定的关键**：1=参与策略梯度，0=纯上下文。工具返回、summary 全是 0。

---

## 4. ③ 奖励计算（EM + 格式分）

```
_compute_reward_colocate (ray_trainer.py:522, 调用于 :1576)       【原生骨架】
 → NaiveRewardManager.__call__ / _compute_reward (naive.py:59)    【原生】
     self.compute_score(...)
 → reward_score/__init__.py:104 按 data_source 分发               【修改】(加 asearcher 源 + 传 extra_info)
 → search_r1_like_qa_em.py:190 compute_score                      【修改】
     指标是 EM（不是 F1）：extract_solution 取最后一个 <answer>，
       em_check 归一化后判等
     返回 dict：
       origin_score  = EM命中1.0（>10个answer标签降到0.25，防刷）
       format_score  = 0.1（严格 thought/tool_call/tool_response/answer 状态机）
       efficiency_score = 算了但没进最终 score
       final score   = origin_score + format_score
 → naive.py:100 把标量 score 写到"最后一个有效 response token"位置   【原生】
   其余为 0 → 存成 rm_scores
 → extract_reward (reward.py:154, ray_trainer.py:1580)            【原生, 字节一致】
     只把 rm_scores 读出来，不做计算
```

> 上游 `search_r1_like_qa_em.py` 原本只返回一个 EM float；SAZ 改成带 format 状态机、efficiency、>10标签惩罚的 dict 返回。`extract_reward` 本身没改。

---

## 5. ④ IGPO 过程奖励（仅当 `adv_estimator=grpo_igpo`）【全新】

```
_compute_igpo_info_gain (ray_trainer.py:1210)                    【新增】
 → 读 non_tensor_batch: igpo_turn_end_indices / igpo_ground_truth
 → IGPORewardBuilder.build (verl/utils/igpo_gt_logprob.py:292)    【新增】
     _tokenize_gts: 对每条 GT 用 gt_prefix/gt_suffix 组装并 tokenize
     build_gt_scoring_rows: 每个(sample, turn边界)构造一行
                            [prompt + response[:e] + GT]，teacher forcing
     logprob_fn(rows) 一次批量前向 → 每行 GT 答案 token 的 mean logP
       (trainer 注入 _igpo_score_rows(:1281)，内部走 actor.compute_log_prob) 【GPU前向,原生】
     compute_info_gain_per_turn: r_t = exp(logP_t) - exp(logP_{t-1})  (prob_diff)
     place_info_gain_on_tokens: 每轮增益散布到该轮最后一个 token
   → 返回 (info_gain_reward, turn_boundary_mask)  形状 (bsz, resp_len)
```

含义：每一轮搜索后，用 teacher forcing 测"模型对正确答案的置信度"，相邻两轮的置信度差就是这一轮的过程奖励，落到该轮末 token 上。

---

## 6. ⑤⑥⑦⑧⑨ 概率 / advantage / 更新 / 同步

| 阶段 | 函数 | 标签 |
|---|---|---|
| ⑤ old_log_prob | `_compute_old_log_prob` (`ray_trainer.py:1597`)，bypass 时用 rollout 算好的 | 【原生】 |
| ⑤ ref_log_prob | `_compute_ref_log_prob` (`ray_trainer.py:1634`) | 【原生】 |
| ⑤ rollout/actor 概率对齐 | `rollout_corr_helper.py` | 【原生】(非 SAZ 发明) |
| ⑥ advantage 分发 | `compute_advantage` (`ray_trainer.py:136`) | 【原生】 |
| ⑥ grpo | `compute_grpo_outcome_advantage` (`core_algos.py:267`) | 【原生, 字节一致】 |
| ⑥ grpo_igpo | `compute_grpo_igpo_advantage` (`core_algos.py:386`)：`combined = outcome + coef*info_gain`；f1_mask/ig_mask 分开/联合归一化；`_igpo_compute_turn_level_advantage`(:334) turn 级折扣累加 | 【新增】 |
| ⑦ 更新 | `_update_critic` / `_update_actor` (`ray_trainer.py:1711`)，GRPO loss + FSDP2 | 【原生】 |
| ⑧ 权重同步 | `update_weights()`，actor→vLLM rollout | 【原生】 |
| ⑨ 验证 | `validate()` (`ray_trainer.py:574`) 走同样 rollout 路径 | 【原生】 |

---

## 7. 原生 vs 新增 —— 边界总结

**verl 原生（骨架/引擎）**：`fit()` 主循环、AgentLoopManager/Worker + 注册表 + hydra 分发、AgentLoop 状态机本身、DataProto、reward manager 骨架、`extract_reward`、GRPO advantage、old/ref log_prob、rollout_correction、update_actor/critic、FSDP2 worker、权重同步。

**SearchAgent-Zero 新增（长程搜索 + IGPO）**：
1. `ToolAgentLoop`(credit_assignment)：长度/轮次护栏、turn 课程、self-summary 压缩、异常轨迹 7 类计数、turn-level credit assignment；
2. `search_tool.py` 远端检索工具；
3. 奖励侧 `search_r1_like_qa_em.py` 的 EM+格式状态机 dict 返回、`__init__.py` 的 asearcher 源；
4. 新增字段打包链路（`abnormal_trajectory_dic`、`all_call_tool_counts`、igpo 字段 → non_tensor_batch）；
5. 整套 IGPO：`igpo_gt_logprob.py`、`grpo_igpo` 估计器、`_compute_igpo_info_gain`、相关 config 字段。

---

## 8. 一个具体样本从头走一遍

> 用一条 2-hop 问答样本（HotpotQA 风格），`n=4`（每 prompt 采 4 条），`adv_estimator=grpo_igpo`。为了直观，token 数是示意值。

### 样本输入（来自 dataset）

```json
{
  "data_source": "hotpotqa",
  "prompt": "Question: The director of the 1997 film Titanic also directed which 2009 sci-fi film?",
  "reward_model": { "ground_truth": "Avatar" }
}
```

`gen_batch.repeat(4)` → 同一问题复制 4 份，分别独立 rollout。以下跟踪其中**第 3 条**轨迹。

### ① Rollout —— ToolAgentLoop.run 生成多轮轨迹

状态机跑了 2 轮搜索后作答（每个 `<...>` 段是一轮里的文本）：

```
turn0: <thought>需要先确认Titanic导演</thought>
       <tool_call>{"name":"search","arguments":{"query_list":["director of 1997 film Titanic"]}}</tool_call>
       <tool_response>James Cameron directed Titanic (1997)...</tool_response>   ← response_mask=0
turn1: <thought>James Cameron的2009科幻片</thought>
       <tool_call>{"name":"search","arguments":{"query_list":["James Cameron 2009 sci-fi film"]}}</tool_call>
       <tool_response>Avatar is a 2009 science fiction film by James Cameron...</tool_response> ← mask=0
turn2: <thought>信息足够了</thought>
       <answer>Avatar</answer>
```

对应产出（示意）：

- `responses`：整条轨迹拼接的 token（长约 180 token）
- `response_mask`：模型自己生成的 `<thought>/<tool_call>/<answer>` token = 1；两段 `<tool_response>` = 0
- `non_tensor_batch`（新增字段）：
  - `all_call_tool_counts = 2`（发了 2 次 search）
  - `all_call_tool_success_counts = 2`
  - `searched_query_count=2`，`duplicate_search_result_count=0`，`tool_parser_error_count=0` …（7 类异常计数，本条都正常=0）
  - `igpo_turn_end_indices = [42, 118, 165]`（每轮最后一个 response token 的下标，示意）
  - `igpo_ground_truth = "Avatar"`

> 假设第 1 条轨迹超长溢出 → 触发护栏"优雅终止、整条 mask 置零"，那条不产生梯度；第 2 条把同一 query 搜了 3 次 → `duplicate_search_result_count` 递增。这些都进 metrics，便于观测长程稳定性。

### ② response_mask / balance

`compute_response_mask` 把上面的 0/1 mask 定稿；`_balance_batch` 在多卡间均衡 token 负载。

### ③ 奖励（EM + 格式分）

`search_r1_like_qa_em.compute_score(solution_str, "Avatar", extra_info)`：

- `extract_solution` 取最后一个 `<answer>Avatar</answer>` → 归一化 `"avatar"`
- `em_check("avatar","avatar")` → 命中 → `origin_score = 1.0`
- 轨迹符合 `thought/tool_call/tool_response/answer` 状态机 → `format_score = 0.1`
- `final score = 1.1`

写到该轨迹**最后一个有效 response token**（即 `<answer>Avatar</answer>` 的末 token，下标 164）：

```
rm_scores[3] = [0, 0, ..., 0, 1.1]   # 仅末 token = 1.1，其余 0
```

4 条轨迹的 outcome 分（示意）：`[1.1, 0.1, 1.1, 0.1]`（第 1、3 条答对，2、4 条只有格式分）。

### ④ IGPO 过程奖励

对第 3 条，用 `igpo_turn_end_indices=[42,118,165]` 构造 3 个 teacher-forcing 行：

| 行 | 内容 | mean logP(GT="Avatar") | exp() |
|---|---|---|---|
| turn0 后 | prompt + resp[:42] + GT | -2.30 | 0.10 |
| turn1 后 | prompt + resp[:118] + GT | -0.51 | 0.60 |
| turn2 后 | prompt + resp[:165] + GT | -0.11 | 0.90 |

`compute_info_gain_per_turn`（prob_diff）：

- turn0→turn1 增益 = 0.60 − 0.10 = **+0.50**（第一次搜到 James Cameron，置信度大涨）
- turn1→turn2 增益 = 0.90 − 0.60 = **+0.30**（第二次搜到 Avatar，再涨）

`place_info_gain_on_tokens` 把它们落到对应轮末 token：

```
info_gain_reward[3][117] = 0.50   # turn1 末
info_gain_reward[3][164] = 0.30   # turn2 末
turn_boundary_mask[3]    = 1 at {41, 117, 164}
```

> 直觉：真正带来信息增量的那一步（搜到关键实体）被单独奖励，而不是把功劳全压在最后答对的 token 上。

### ⑤ old/ref log_prob

对 4 条完整轨迹跑一遍 actor（old）和 reference（ref）前向，拿到每个 response token 的 log_prob，用于 PPO ratio 和 KL。

### ⑥ advantage（grpo_igpo）

- **组内基线**：4 条 outcome 分 `[1.1,0.1,1.1,0.1]`，组均值=0.6，标准化后第 3 条的 outcome advantage 为正（答对，高于组均值）。
- **过程项**：turn1/turn2 末 token 各叠加 `coef * info_gain`（0.50 / 0.30 归一化后）。
- `combined = outcome_adv + coef*info_gain`；`_igpo_compute_turn_level_advantage` 做 turn 级折扣累加。

结果：第 3 条的 `<answer>` 末 token 有强正 advantage（答对），turn1/turn2 的关键搜索 token 也有正 advantage（有信息增益）——**梯度信号被合理分摊到"做对的每一步"**。

### ⑦⑧⑨ 更新 & 同步

- `_update_actor`：用上面 advantage + response_mask 算 GRPO loss，只在 mask=1 的 token 上回传（tool_response 那些 mask=0 的 token 不参与）。
- `update_weights`：更新后的 actor 权重同步回 vLLM rollout engine，下个 step 用新策略采样。
- `validate`：周期性在验证集上走同样 rollout 路径评估 EM。

---

### 这条样本的"原生/新增"落点回顾

| 环节 | 这条样本发生了什么 | 谁负责 |
|---|---|---|
| 多轮搜索生成 | 2 轮 search + 作答 | ToolAgentLoop【新增】/ 状态机【原生】 |
| 检索 | 2 次 HTTP 打远端 | search_tool.py【新增】 |
| mask | tool_response=0 | 逻辑【新增】/ 字段【原生】 |
| 异常计数 | 全 0（正常轨迹） | abnormal 计数【新增】 |
| EM 奖励 1.1 | 答对+格式对 | search_r1_like_qa_em【修改】/ 骨架【原生】 |
| 过程奖励 +0.5/+0.3 | 两次搜索信息增益 | IGPO【新增】 |
| advantage | outcome+过程，turn 级 | grpo_igpo【新增】/ 分发【原生】 |
| 更新/同步 | GRPO loss + 权重同步 | 【原生】 |
