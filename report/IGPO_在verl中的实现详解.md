# IGPO 在 verl 中的实现详解(结合源代码)

> 本文档详细解释 **IGPO(Information Gain Policy Optimization,信息增益策略优化)** 在本项目 verl 训练栈中的完整实现,逐段结合源代码。
>
> IGPO 论文:arXiv:2510.14967。本项目的实现是从 `report/IGPO_ref/` 官方参考实现**忠实移植(faithfully port)** 到自研的 verl + credit-assignment agent loop 训练栈上的。

---

## 目录

1. [一句话概括:IGPO 解决什么问题](#1-一句话概括)
2. [代码分布:IGPO 改动落在哪些文件](#2-代码分布)
3. [核心思想:从 outcome 奖励到稠密的轮级信息增益](#3-核心思想)
4. [端到端数据流(五个阶段)](#4-端到端数据流)
5. [阶段一:Rollout 采集轮边界与 GT(agent loop)](#5-阶段一rollout-采集)
6. [阶段二:训练循环触发信息增益计算(fit loop)](#6-阶段二fit-loop-触发)
7. [阶段三:信息增益 reward 构建(IGPORewardBuilder)](#7-阶段三信息增益构建)
8. [阶段四:GT log-prob 打分(teacher forcing)](#8-阶段四gt-log-prob-打分)
9. [阶段五:grpo_igpo advantage 估计器](#9-阶段五advantage-估计器)
10. [配置项完整清单](#10-配置项清单)
11. [健壮性设计:安全回退与默认关闭](#11-健壮性设计)
12. [已知问题:源文件缺失](#12-已知问题)

---

## 1. 一句话概括

**标准 GRPO 只有一个最终的 outcome reward(答案对不对),放在整条轨迹的最后一个 token 上。** 对一条几十轮的长程搜索轨迹而言,这个信用分配太粗糙——模型不知道到底是"哪一轮搜索"真正推进了答案。

**IGPO 的做法**:在每一轮搜索结束后,用 teacher forcing 测量"模型此刻对标准答案(ground-truth)的把握有多大"(即 `log P(GT | 到当前轮为止的历史)`)。相邻两轮之间这个概率的**提升**,就是这一轮的**信息增益(information gain)**,作为一个**稠密的过程奖励(dense process reward)**放在该轮的最后一个 token 上。这样,推进了答案的轮次得到正向信用,没推进的轮次不得利,信用分配从"轨迹级"细化到"轮级"。

---

## 2. 代码分布

IGPO 是一个**端到端集成、可开关、带安全回退**的一等特性,改动分布在 5 个源文件 + 1 个纯计算核心模块:

| 文件 | 角色 | 关键符号 |
|---|---|---|
| `verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py` | **① Rollout 采集** | `enable_igpo`、`igpo_turn_end_indices`、`igpo_ground_truth` |
| `verl/trainer/ppo/ray_trainer.py` | **② 训练循环编排 + ④ GT 打分** | `_compute_igpo_info_gain()`、`_igpo_score_rows()` |
| `verl/utils/igpo_gt_logprob.py` | **③ 纯计算核心** | `IGPORewardBuilder`、`GTScoringRow`、`compute_info_gain_per_turn`、`place_info_gain_on_tokens` |
| `verl/trainer/ppo/core_algos.py` | **⑤ Advantage 估计器** | `@register_adv_est("grpo_igpo")`、`_igpo_compute_turn_level_advantage` |
| `verl/trainer/config/algorithm.py` | 算法侧配置 | `info_gain_type`、`igpo_gamma`、`igpo_coef`、curriculum 字段 |
| `verl/workers/config/rollout.py` | rollout 侧配置 | `enable_igpo`、`igpo_gt_prefix/suffix` |

> ⚠️ 注意:干净的上游 workspace 副本(`workspace/SearchAgent-Zero`)**不含** IGPO(`grpo_igpo` 零匹配)——IGPO 是本项目自研、尚未推上游的改动,只存在于原始仓库 `/mmu_mllm_hdd_2/zhoujinchang/SearchAgent-Zero`。

---

## 3. 核心思想

对一条轨迹,把它按"轮(turn)"切分。设第 t 轮结束时的历史为 `history_t`,GT 为标准答案。定义:

- **每轮的信念(belief)**:`mean_logP_t = mean_token log P(GT | history_t)`,用 teacher forcing 打分得到。
- **每轮的信息增益(info gain)**,两种公式(源码 `igpo_gt_logprob.py` docstring 明确):

  | `info_gain_type` | 公式 |
  |---|---|
  | `prob_diff`(默认) | `r_t = exp(mean_logP_t) − exp(mean_logP_{t-1})` |
  | `log_prob_diff` | `r_t = mean_logP_t − mean_logP_{t-1}` |

- 这个 `r_t` 被放在**第 t 轮最后一个 token** 上,构成稠密过程奖励;最终答案 token 上仍是 outcome reward(EM)。
- Advantage 再对这些"轮末奖励"做**分组归一化 + 轮级折扣累积**(见 §9)。

---

## 4. 端到端数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│ ① Rollout (agent loop, GPU rollout worker)                           │
│   每轮工具响应追加后,记录 response_mask 长度 → igpo_turn_end_indices  │
│   连同 ground_truth 一起放进 output.extra_fields                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ② Fit loop (ray_trainer.fit)                                         │
│   算完 outcome reward 后, 若 adv_estimator=="grpo_igpo":              │
│   调用 self._compute_igpo_info_gain(batch)                            │
└───────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ③ IGPORewardBuilder.build (verl/utils/igpo_gt_logprob.py, 纯 CPU)     │
│   build_gt_scoring_rows: 每 (样本,轮) 造一条 [history_t + GT] 行      │
│   → 调 logprob_fn 打分 → compute_info_gain_per_turn (prob_diff)       │
│   → place_info_gain_on_tokens: scatter 到每轮末 token                 │
│   返回 igpo_info_gain_reward + igpo_turn_boundary_mask                │
└───────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ④ _igpo_score_rows (ray_trainer, GPU actor)                          │
│   teacher forcing: actor.compute_log_prob 打 GT 答案 token 的 log-prob │
│   (这是唯一需要 GPU 的部分, logprob_fn 的实体)                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ⑤ compute_grpo_igpo_advantage (core_algos.py)                        │
│   combined = outcome + igpo_coef * info_gain                         │
│   → 分组归一化(separate/joint) → 轮级折扣累积 A_i=r_i+γA_{i+1}       │
│   → broadcast 到每轮所有 token                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 阶段一:Rollout 采集

**文件**:`verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py`

**5.1 开关与状态初始化**(`AgentData` 构造 & agent loop init):

```python
# AgentData: 每轮结束时记录 response_mask 的长度,标记"到第 t 轮为止的历史"边界
self.igpo_turn_end_indices: list[int] = []          # 约 line 164

# agent loop: 读开关,默认 False => 行为与原来完全一致
self.enable_igpo = getattr(self.rollout_config.multi_turn, "enable_igpo", False)  # line 230
```

**5.2 每轮结束时记录轮边界**(工具响应追加进 prompt 之后,line 761–772):

```python
agent_data.prompt_ids += response_ids
agent_data.response_mask += [0] * len(response_ids)
...
agent_data.user_turns += 1
agent_data.tool_turns += 1
# IGPO: 记录本轮结束时的 response-token 边界(工具响应已追加)。
# 这标记了"到第 t 轮为止的历史",供后续打分 P(GT|history_t)。
# 下游只保留前 response_length 个 token,越界的边界无害(trainer 会 clamp)。
if self.enable_igpo:
    agent_data.igpo_turn_end_indices.append(len(agent_data.response_mask))
```

**5.3 把 IGPO 元数据随 rollout 输出返回**(line 405–408):

```python
output.extra_fields.update({"turn_scores": ..., "tool_rewards": ...})
if self.enable_igpo:
    output.extra_fields.update({
        "igpo_turn_end_indices": agent_data.igpo_turn_end_indices,   # 每轮末位置
        "igpo_ground_truth": list(agent_data.ground_truth_list),     # 标准答案
    })
```

> **要点**:rollout 侧只做"记录",不做打分——它只负责标出每轮在 response token 序列里的结束位置,并把 GT 带出来。真正的信息增益计算全部在训练侧完成。

---

## 6. 阶段二:Fit loop 触发

**文件**:`verl/trainer/ppo/ray_trainer.py`,`fit()` 主循环(line 1647–1653)

```python
batch.batch["token_level_scores"] = reward_tensor   # outcome reward(EM)先算好

# IGPO: 计算稠密的轮级信息增益 reward,并把它 + turn-boundary mask 塞进 batch,
# 供 grpo_igpo advantage 估计器消费。未开启则完全 no-op。
if self.config.algorithm.get("adv_estimator", "") == "grpo_igpo":
    with marked_timer("igpo_info_gain", timing_raw, color="magenta"):
        self._compute_igpo_info_gain(batch)
```

计算完后,在 advantage 分发处(line 215–222)把两个张量喂给估计器:

```python
if adv_estimator in ("grpo_igpo",):
    if "index" not in adv_kwargs and "uid" in data.non_tensor_batch:
        adv_kwargs["index"] = data.non_tensor_batch["uid"]
    adv_kwargs["norm_adv_by_std_in_grpo"] = norm_adv_by_std_in_grpo
    if "igpo_info_gain_reward" in data.batch.keys():
        adv_kwargs["info_gain_reward"] = data.batch["igpo_info_gain_reward"]
    if "igpo_turn_boundary_mask" in data.batch.keys():
        adv_kwargs["turn_boundary_mask"] = data.batch["igpo_turn_boundary_mask"]
```

---

## 7. 阶段三:信息增益构建

**文件**:`verl/trainer/ppo/ray_trainer.py::_compute_igpo_info_gain()`(line 1210–1279)

这一步把 rollout 带出的原始数据整理成 `IGPORewardBuilder` 需要的格式:

**7.1 解 padding,取回每条样本的真实 token 序列**(line 1243–1255):

```python
prompt_mask = attention_mask[:, :prompt_len]
resp_mask   = attention_mask[:, prompt_len:]
for i in range(bsz):
    p = prompt_ids[i][prompt_mask[i].bool()].tolist()     # 去 padding 的 prompt
    r = response_ids[i][resp_mask[i].bool()].tolist()     # 去 padding 的 response
    te = turn_ends[i]                                     # 该样本各轮末位置
    g  = gts[i]                                           # 该样本 GT
    ...
```

**7.2 构造 builder 并调用**(line 1257–1277):

```python
builder = IGPORewardBuilder(
    tokenizer=self.tokenizer,
    info_gain_type=self.config.algorithm.get("info_gain_type", "prob_diff"),
    gt_prefix=...multi_turn.get("igpo_gt_prefix", "\nNow there's enough information to answer\n</thought>\n<answer>\n"),
    gt_suffix=...multi_turn.get("igpo_gt_suffix", "\n</answer><|im_end|>"),
)

def logprob_fn(rows):                     # GPU teacher-forcing 打分(见 §8)
    return self._igpo_score_rows(rows)

info_gain_reward, turn_boundary_mask = builder.build(
    prompt_ids_per_sample=..., response_ids_per_sample=...,
    turn_end_indices_per_sample=..., gt_texts=..., response_length=resp_len,
    logprob_fn=logprob_fn,
)
```

**7.3 `IGPORewardBuilder` 的内部结构**(`verl/utils/igpo_gt_logprob.py`,依据模块符号与 docstring):

| 函数/类 | 职责 |
|---|---|
| `GTScoringRow` | 一条 teacher-forcing 打分行的数据结构(`input_ids`、`ans_start`、`ans_end`) |
| `build_gt_scoring_rows` | 对每个 (样本, 轮边界 e),造一条 `[history 到 e + GT_prefix + GT + GT_suffix]` 行 |
| `compute_info_gain_per_turn` | 对每条样本的逐轮 `mean_logP` 序列,套 `prob_diff` / `log_prob_diff` 公式,算出逐轮增益 |
| `place_info_gain_on_tokens` | 把 `info_gains[k]`(第 k+1 轮相对第 k 轮的增益)scatter 到 `turn_end_indices[k+1]-1` 位置;同时产出 `turn_boundary_row`(每个轮末位置为 1) |
| `IGPORewardBuilder.build` | 编排上述步骤,返回 `(info_gain_reward, turn_boundary_mask)`,形状均为 `(bsz, response_length)` |

> `prob_diff` 与 `log_prob_diff` 的定义(直接取自 `.pyc` 内嵌 docstring):
> ```
> prob_diff  (default):  r_t = exp(mean_logP_t) - exp(mean_logP_{t-1})
> log_prob_diff       :  r_t = mean_logP_t       - mean_logP_{t-1}
> ```
> 使用 tokenizer 的 `offset_mapping` 做精确的字符→token 边界检测,以对齐官方实现。

---

## 8. 阶段四:GT log-prob 打分

**文件**:`verl/trainer/ppo/ray_trainer.py::_igpo_score_rows()`(line 1281–1353)

这是 `logprob_fn` 的实体,也是**整个 IGPO 里唯一需要 GPU 的部分**。它把一批 `GTScoringRow` 摆成标准 verl 的"左填充 prompt + 右填充 response"格式,让 `compute_log_prob` 恰好打分 GT 答案那几个 token。

**8.1 摆放格式**(line 1297–1319):

```python
prompt_parts = [r.input_ids[: r.ans_start] for r in rows]      # 历史 + GT 前缀
resp_parts   = [r.input_ids[r.ans_start : r.ans_end] for r in rows]  # GT 答案 token
...
# prompt 左填充 -> 占 [max_p-lp : max_p]
input_ids[i, max_p - lp : max_p] = torch.tensor(p)
# response 右填充 -> 占 [max_p : max_p+lr]
input_ids[i, max_p : max_p + lr] = torch.tensor(rp)
...
position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)   # 左填充感知
```

**8.2 padding 到 dp_size 整数倍再打分**(line 1338–1350)——因为打分行数 = 所有样本的有效轮数之和,通常不能被 actor 的 dp_size 整除,会触发 dispatch 的整除断言:

```python
dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
td, pad_size = pad_dataproto_to_divisor(td, dp_size)   # 补齐,打分后丢弃尾部重复行
...
# temperature=1.0: 打真实(未缩放)的 GT log-prob,与 rollout 采样温度无关
tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False, temperature=1.0)
output = self.actor_rollout_wg.compute_log_prob(batch_td)
log_probs = no_padding_2_padding(...)[:n]              # 丢弃 padding 行
denom = response_mask.sum(dim=-1).clamp(min=1)
mean_lp = (lp * response_mask).sum(dim=-1) / denom     # 每行 GT 答案 token 的平均 log-prob
return mean_lp.tolist()
```

> **关键设计**:用 `temperature=1.0` 打分,是为了测量"模型对 GT 的真实信念",不受 rollout 时采样温度的影响。这是信息增益作为"belief measurement"的正确性前提。

---

## 9. 阶段五:Advantage 估计器

**文件**:`verl/trainer/ppo/core_algos.py`,`compute_grpo_igpo_advantage`(line 386–478)+ `_igpo_compute_turn_level_advantage`(line 334–383)

**9.1 主函数四步**(docstring 已写明,line 397–414):

```python
@register_adv_est("grpo_igpo")
def compute_grpo_igpo_advantage(token_level_rewards, response_mask, index, ...,
                                turn_boundary_mask=None, info_gain_reward=None):
    # 安全回退:缺 IGPO 张量则退回普通 outcome GRPO
    if turn_boundary_mask is None or info_gain_reward is None:
        return compute_grpo_outcome_advantage(...)

    gamma     = getattr(config, "igpo_gamma", 1.0)
    norm_mode = getattr(config, "info_gain_norm_mode", "separate")
    coef      = getattr(config, "igpo_coef", 1.0)

    with torch.no_grad():
        # 1) 合并:outcome(权重1) + coef * info_gain
        combined = token_level_rewards + coef * info_gain_reward

        # 2) 两类 mask:f1=每条序列最后一个有效 token(outcome 位置);
        #              ig=轮末位置(排除 f1)
        last_valid_pos = (seq_len-1) - response_mask.flip(dims=[1]).long().argmax(dim=1)
        f1_mask = (pos_idx == last_valid_pos.unsqueeze(1)) & (response_mask == 1)
        ig_mask = (turn_boundary_mask > 0) & (response_mask == 1) & (~f1_mask)

        # 3) 分组归一化(按 index 分组做 GRPO 组内标准化)
        if norm_mode == "separate":     # info_gain 与 outcome 各自独立归一化
            masks = (f1_mask, ig_mask)
        else:                            # joint: 一起归一化
            masks = (f1_mask | ig_mask,)
        for m in masks:
            gm, gs = group_stats(m)      # scatter_add 求每组 mean/std
            nr = combined - gm[group_ids_exp]
            if norm_adv_by_std_in_grpo:
                nr = nr / (gs[group_ids_exp] + epsilon)
            normalized = torch.where(m, nr, normalized)

        # 4) 轮级折扣累积 + broadcast
        boundary = (f1_mask | ig_mask).to(combined.dtype)
        advantages = _igpo_compute_turn_level_advantage(normalized, response_mask, gamma, boundary)
    return advantages, advantages
```

**9.2 轮级折扣累积**(`_igpo_compute_turn_level_advantage`,line 358–383):

```python
for b in range(bsz):
    reward_positions = turn_boundary_mask[b].nonzero(...).tolist()   # 该样本所有轮末位置
    # 反向折扣累积: A_i = r_i + gamma * A_{i+1}
    next_adv = 0.0
    turn_data = []
    for pos in reversed(reward_positions):
        adv = rewards[pos].item() + gamma * next_adv
        turn_data.append((pos, adv)); next_adv = adv
    turn_data.reverse()
    # 把每轮的 A_i broadcast 到该轮范围内所有 mask==1 的 token
    prev_end = 0
    for pos, adv in turn_data:
        for t in range(prev_end, pos + 1):
            if mask[t] == 1: returns[b, t] = adv
        prev_end = pos + 1
```

> **直觉**:`gamma`(默认 1.0)是**轮级折扣**——一轮的价值 = 本轮信息增益 + 后续所有轮的折扣回报。这让"为后续几轮铺路"的早期搜索也能得到信用。归一化的 `separate` 模式(默认)让稠密的 info-gain 奖励与稀疏的 outcome 奖励在各自的尺度上标准化,避免一方淹没另一方。

---

## 10. 配置项清单

**算法侧**(`verl/trainer/config/algorithm.py`,line 680–688):

```python
info_gain_type: str = "prob_diff"       # prob_diff | log_prob_diff
info_gain_norm_mode: str = "separate"   # separate | joint
igpo_gamma: float = 1.0                 # 轮级折扣 A_i = r_i + gamma*A_{i+1}
igpo_coef: float = 1.0                  # info-gain 相对 outcome 的权重
use_igpo_curriculum: bool = False       # 是否随训练线性退火 info-gain/outcome 权重
igpo_ig_init: float = 1.0               # curriculum: info-gain 权重起点
igpo_ig_final: float = 0.2              #             终点
igpo_f1_init: float = 1.0               # curriculum: outcome 权重起点
igpo_f1_final: float = 1.0              #             终点
```

**Rollout 侧**(`verl/workers/config/rollout.py`,line 111–116):

```python
enable_igpo: bool = False               # 总开关,默认关(行为等同 outcome-only)
# GT 答案包裹:PREFIX + gt + SUFFIX 后再 teacher-forcing
igpo_gt_prefix: str = "\nNow there's enough information to answer\n</thought>\n<answer>\n"
igpo_gt_suffix: str = "\n</answer><|im_end|>"
```

> ⚠️ **一个易错点(源码注释特别标注)**:本项目的推理标签是 `<thought>`(**不是**官方的 `<think>`)。`igpo_gt_prefix` 必须与训练用的 system prompt 对齐,否则 GT 分布会 mismatch,信息增益打分失真。

**本地实验(`igpo_prob_diff`)实际用的配置**:`adv_estimator=grpo_igpo`、`info_gain_type=prob_diff`、`info_gain_norm_mode=separate`、`igpo_coef=1.0`、`use_igpo_curriculum=False`、`enable_igpo=True`。

---

## 11. 健壮性设计

IGPO 的集成贯彻了"**默认关闭、可安全回退、不影响原路径**"的原则,有三道保险:

1. **总开关默认 False**(`enable_igpo=False`):不开时 rollout 不记录轮边界,行为与原 outcome-only 训练**逐字节一致**。
2. **fit loop 条件触发**:只有 `adv_estimator=="grpo_igpo"` 才调用 `_compute_igpo_info_gain`,否则完全 no-op。
3. **估计器内部安全回退**:即便进了 `grpo_igpo` 估计器,若 `turn_boundary_mask` 或 `info_gain_reward` 缺失(元数据没采到),会自动 `return compute_grpo_outcome_advantage(...)` 退回普通 GRPO(core_algos.py:416–419);`_compute_igpo_info_gain` 在 `turn_ends is None or gts is None` 时也会写全零并直接返回(ray_trainer.py:1238–1241)。

这套设计意味着 IGPO 是一个"叠加"特性——加进主干不会破坏既有的 GRPO / 异步训练路径。

---

## 12. 已知问题:源文件缺失

⚠️ **纯计算核心 `verl/utils/igpo_gt_logprob.py` 的源文件在原始仓库中已丢失**(见 [叙事报告](SearchAgent-Zero_项目叙事报告.md) 记录的 7/31 目录事件),当前**只剩编译缓存** `verl/utils/__pycache__/igpo_gt_logprob.cpython-310.pyc`。

- 本文档 §7 对 `IGPORewardBuilder` / `GTScoringRow` / `compute_info_gain_per_turn` / `place_info_gain_on_tokens` 的描述,是**结合 `ray_trainer.py` 的调用点 + `.pyc` 内嵌的 docstring 与符号表**还原的,逻辑可信,但**不是逐行源码**。
- **恢复建议**:
  1. 该 `.pyc` 是 Python 3.10 编译产物,可用 `decompyle3` / `pycdc` 等工具反编译还原大部分源码;
  2. 或从上游 `report/IGPO_ref/` 参考实现对照移植(注释称本实现"忠实移植"自它);
  3. 只要 `.pyc` 还在,且 Python 版本匹配(3.10),**训练本身仍可运行**(import 的是编译产物)。
- 其余 5 个 IGPO 源文件(agent loop、ray_trainer、core_algos、两个 config)均为**完整 `.py` 源码,未丢失**。

---

## 附:关键源码位置速查

| 功能 | 文件:行 |
|---|---|
| `grpo_igpo` advantage 估计器 | `verl/trainer/ppo/core_algos.py:386` |
| 轮级折扣累积 | `verl/trainer/ppo/core_algos.py:334` |
| fit loop 触发点 | `verl/trainer/ppo/ray_trainer.py:1651` |
| info-gain reward 编排 | `verl/trainer/ppo/ray_trainer.py:1210` |
| GT teacher-forcing 打分(GPU) | `verl/trainer/ppo/ray_trainer.py:1281` |
| advantage 分发注入 | `verl/trainer/ppo/ray_trainer.py:215` |
| rollout 记录轮边界 | `verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py:771` |
| rollout 输出 IGPO 元数据 | `.../tool_agent_loop_credit_assignment.py:405` |
| 算法配置字段 | `verl/trainer/config/algorithm.py:680` |
| rollout 配置字段 | `verl/workers/config/rollout.py:111` |
| 纯计算核心(仅 .pyc) | `verl/utils/igpo_gt_logprob.py`(源缺失) |

---

*本文档基于 `/mmu_mllm_hdd_2/zhoujinchang/SearchAgent-Zero` 原始仓库源码撰写。IGPO 参考实现见 `report/IGPO_ref/`;论文 arXiv:2510.14967。*
