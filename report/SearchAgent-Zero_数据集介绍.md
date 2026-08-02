# SearchAgent-Zero — 数据集格式、构造方法与 Case 集

> 目的:把训练/评测用到的数据讲透——**每种数据长什么样、字段怎么组织、如何从原始 HF 数据集构造成 verl 多轮训练格式、检索/奖励环节的数据流是什么**;附真实 case,并总结数据构建与清洗的方法论。
>
> 三种数据角色:
> - **训练数据**(RL rollout 的题目池):Search-R1 短程 QA、ASearcher 长程多跳 QA。
> - **检索语料**(工具返回的知识源):Wikipedia-18 + E5 dense index。
> - **评测数据**(离线打分):Search-R1 7 QA 验证集、BrowseComp-Plus 830 题。

---

## 1. 统一数据契约:verl 多轮工具调用格式

无论 Search-R1 还是 ASearcher,原始 HF 数据都会被预处理脚本转换成**同一套 6 字段 schema**(verl `RLHFDataset` 约定)。这是全项目的数据契约,先讲清它,后面的 case 才好读。

| 字段 | 类型 | 作用 |
|---|---|---|
| `data_source` | str | 数据来源标签(加 `searchR1_` 前缀),决定用哪个 reward_fn、按源分组统计指标 |
| `prompt` | list[msg] | 对话消息列表 `[{system}, {user}]`,喂给 chat template |
| `ability` | str/None | 能力标签(如 `fact-reasoning`),仅作元信息 |
| `reward_model` | dict | `{"ground_truth": {"target": [...]}, "style": "rule"}`,打分用的标准答案 |
| `extra_info` | dict | `index / split / question / need_tools_kwargs / tools_kwargs` |
| `metadata` | any/None | 原始元信息透传 |

**关键设计点**:

1. **`tools_kwargs` 把 ground_truth 注入到工具会话里**。`extra_info.tools_kwargs.search.create_kwargs` 携带 `{ground_truth, question, data_source}`——这样每条 rollout 的搜索工具实例都知道自己这道题的标答与来源,rollout 结束后可直接算 reward,无需再回表 join。
2. **`need_tools_kwargs=True`** 告诉 verl 这条样本要走工具 AgentLoop(而非纯文本生成)。
3. **`ground_truth` 统一包成 `{"target": [...]}` 列表**:答案天然可能有多个别名(同义写法),打分时对列表做"命中任一即正确"。ASearcher 脚本显式 `np.array([ground_truth])` 保证结构一致。

### 1.1 System Prompt(决定输出协议)

两个数据集共用同一段 system prompt,它定义了 agent 的**输出协议**(后续格式奖励、异常检测、轮次切分全依赖它):

```text
You are a helpful and harmless assistant.
Answer the given question. You must first conduct step by step reasoning
between <thought> and </thought> first every time you get new information.
If you need external information, call the search tool by returning a JSON
object inside <tool_call> tags. ... it will return the top searched results
between <tool_response> and </tool_response>.
You can search as many times as you want. Break down the user's question into
specific sub-questions for searching.
Check previous search history to ensure new queries are unique.
If you find no further external knowledge needed, you can directly provide the
answer inside <answer> and </answer> ... For example, <answer> Beijing </answer>.
```

协议要点(每一条都对应下游一个机制):
- `<thought>…</thought>`:强制"拿到新信息先推理"→ ReAct 式思考。
- `<tool_call>{JSON}</tool_call>`:工具调用,JSON 内是 `query_list`。
- `<tool_response>…</tool_response>`:环境注入的检索结果(mask=0,不训练)。
- `<answer>…</answer>`:终止并作答。
- "Check previous search history to ensure new queries are unique" → 直接对应**重复 query 异常检测**的 prompt 侧约束。

---

## 2. 训练数据集之一:Search-R1(短程 QA)

### 2.1 来源与规模
- **HF 源**:`PeterJinGo/nq_hotpotqa_train`。
- **规模**:处理后 **train ≈ 169,615 行**;test 按 `data_source` 分组、每组截断(短程验证覆盖 7 个开放域 QA:NQ / TriviaQA / PopQA / HotpotQA / 2Wiki / Musique / Bamboogle)。
- **特征**:单跳或 2–3 跳,答案短(实体/数字/是否),适合验证"推理+检索"链路。

### 2.2 真实 Case(处理后)

```jsonc
{
  "data_source": "searchR1_nq",
  "prompt": [
    {"role": "system", "content": "You are a helpful and harmless assistant. ..."},
    {"role": "user",   "content": "total number of death row inmates in the us?"}
  ],
  "ability": "fact-reasoning",
  "reward_model": {"ground_truth": {"target": "['2,718']"}, "style": "rule"},
  "extra_info": {
    "index": 0, "need_tools_kwargs": true, "split": "train",
    "question": "total number of death row inmates in the us?",
    "tools_kwargs": {"search": {"create_kwargs": {
        "ground_truth": {"target": "['2,718']"},
        "question": "total number of death row inmates in the us?",
        "data_source": "searchR1_nq"}}}
  }
}
```

再看一条(答案是拼写数字,提示答案有多种表面形式):
```jsonc
{"data_source": "searchR1_nq",
 "reward_model": {"ground_truth": {"target": "['seven']"}},
 "extra_info": {"question": "big little lies season 2 how many episodes?"}}
```

### 2.3 构造流程(`preprocess_search_r1_dataset_new.py`)
1. 从 HF 下载 `train.parquet` / `test.parquet`。
2. 逐行:取 `question` → 拼 `[system, user]` prompt;从 `reward_model.ground_truth` 取标答(缺失回退 `golden_answers`)。
3. 打 `data_source = "searchR1_" + 原始源`。
4. 组装 `tools_kwargs`、`extra_info`,落 parquet。
5. **test 侧清洗**:`groupby("data_source").head(500)` 每个来源最多取 500 条并 `reset_index`——控制评测规模、避免大源淹没小源。

---

## 3. 训练数据集之二:ASearcher(长程多跳 QA)

### 3.1 来源与规模
- **HF 源**:`aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample`(单文件 `ASearcher_en_nomath_rejectsample.json`)。
- **规模**:按 `train_ratio=0.95` / `seed=42` 切分后 **train ≈ 13,285 行**、test ≈ 700 行。
- **两个关键限定词**(来自数据集名):
  - **`no-math`**:剔除数学题——数学靠推理而非检索,混进来会稀释"搜索决策"信号。
  - **`Qwen3-8B-reject-sample`**:用 Qwen3-8B 做**拒绝采样**过滤——只保留策略模型经过多轮搜索"够得着但不 trivial"的题,难度匹配被训模型,避免全对(无梯度)或全错(噪声)。

### 3.2 真实 Case(处理后)——注意问题的长度与跳数

```jsonc
{
  "data_source": "searchR1_asearcher",
  "prompt": [
    {"role": "system", "content": "You are a helpful and harmless assistant. ..."},
    {"role": "user", "content": "Prior to a 2007 municipal merger in Denmark's Capital Region, the former municipality housing the Louisiana Museum of Modern Art and Fredensborg Palace, whose dialect forms the basis of standard Danish, merged with which other municipality to create the new administrative area?"}
  ],
  "ability": null,
  "reward_model": {"ground_truth": {"target": "['Karlebo Kommune']"}, "style": "rule"},
  "extra_info": {
    "index": 0, "need_tools_kwargs": true, "split": "train",
    "question": "…(同上)…",
    "tools_kwargs": {"search": {"create_kwargs": {
        "ground_truth": {"target": "['Karlebo Kommune']"}, "…": "…"}}},
    "raw_extra_info": { /* 保留原始字段,便于回溯 */ }
  }
}
```

对比 Search-R1 的"total number of death row inmates in the us?"——ASearcher 的问题是**多约束、多实体、需要 3–6+ 次检索逐步收窄**的类型,这正是长程搜索能力的训练目标。

### 3.3 构造流程(`preprocess_ASearcher_dataset.py`)
1. HF 下载单个 JSON → `json.load`。
2. 逐条:从 `extra_info.question` 取题;`reward_model.ground_truth`(缺失回退 `extra_info.ground_truth`);**统一包成 `{"target": np.array([gt])}`**。
3. 打 `searchR1_` 前缀;组装 `tools_kwargs`/`extra_info`;**保留 `raw_extra_info`** 以便追溯。
4. **train/test 切分**:固定 `seed=42` 的 `np.random` shuffle + `train_ratio=0.95`;保证至少各留 1 条,可复现。

### 3.4 与 Search-R1 的构造差异一览
| 维度 | Search-R1 | ASearcher |
|---|---|---|
| 原始格式 | HF parquet(已分 train/test) | 单个 JSON(脚本内切分) |
| 题目难度 | 短程 1–3 跳 | 长程多跳,reject-sample 难度对齐 |
| ground_truth 包装 | 直接透传 `{"target":[...]}` | `np.array([gt])` 强制结构一致 |
| 清洗动作 | test 每源 head(500) | no-math 过滤 + reject-sample + 95/5 切分 |
| 可回溯 | — | 保留 `raw_extra_info` |

---

## 4. 检索语料与工具数据流

### 4.1 检索服务 I/O(Search-R1 风格)
```python
# 请求
{"queries": ["What is Python?", "..."], "topk": 3, "return_scores": true}
# 响应
{"result": [[{"document": {"contents": "Title\nbody..."}, "score": 0.9}, ...], ...]}
```
- 语料:Wikipedia-18(`wiki-18.jsonl`),索引:E5 `e5_Flat.index`(dense)。
- 服务:`http://127.0.0.1:8000/retrieve`;工具 `SearchTool`(`num_workers=60`,`rate_limit=60`,`timeout=20`)。

### 4.2 检索结果如何变成模型看到的文本(`_format_passages`)
每篇 doc 的 `contents` 首行当 Title、其余当正文,拼成:
```text
Doc 1 (Title: Death row)
In the United States, as of ... the total number of death row inmates ...

Doc 2 (Title: Capital punishment in the United States)
...
```
这段文本被包进 `<tool_response>…</tool_response>` 注入上下文。**长程场景下**它先经 self-summary 压缩(见源码详解),摘要/原文都以 `response_mask=0` 拼回,不参与 loss。

### 4.3 一条 rollout 的数据流(端到端)
```
[system + user question]                      ← prompt(来自训练样本)
<thought> 需要先查 X </thought>
<tool_call>{"query_list": ["..."]}</tool_call>  ← 模型生成,mask=1
<tool_response> Doc 1 (Title: ...) ... </tool_response>  ← 检索/摘要注入,mask=0
<thought> 还差 Y,再查 </thought>
<tool_call>{"query_list": ["..."]}</tool_call>
<tool_response> ... </tool_response>
...
<answer> Karlebo Kommune </answer>            ← 终止,触发打分
```

---

## 5. 奖励怎么用数据:EM + 格式 + 效率(`search_r1_like_qa_em.py`)

rollout 文本 + `reward_model.ground_truth.target` → `compute_score` 产出多路分:

| 分项 | 规则 |
|---|---|
| `origin_score` | 抽取最后一个 `<answer>` 内容,`normalize_answer` 后与 `target` 列表做 **EM(命中任一=1)** |
| `format_score` | 全轨迹经 `is_valid_sequence` 状态机校验(thought→tool_call→tool_response→…→answer 严格配对),合法 +0.1 |
| `efficiency_score` | 恰好 1 个 `<answer>`=0.5,多个=0(惩罚啰嗦/复读) |
| 防刷分 | `</answer>` 出现 >10 次,`score/4`(防止刷格式) |

**`normalize_answer`**(EM 的清洗核心):小写化 → 去标点 → 去冠词(a/an/the)→ 压空格。这让 `"The Beijing."` 与 `"beijing"` 判等,是 QA EM 公平打分的通用做法。

> **数据侧 insight**:正因 EM 对表面形式敏感,**ground_truth 必须是"答案别名列表"**(`target: [...]`),否则同义正确答案会被误判为错。这解释了第 1 节为何把 gt 统一成列表结构。

---

## 6. 评测数据:Search-R1 验证集 与 BrowseComp-Plus

### 6.1 Search-R1 验证集(短程)
- 7 个 QA 的 test 集,`val_kwargs.n=1`,指标 `score@1`(即 `origin_score` 的均值)。
- 训练中按 `data_source` 分组上报 `val-core/searchR1_<ds>/reward/mean@1`,天然得到分数据集对比。

### 6.2 BrowseComp-Plus(长程,主评测)
- **规模**:**830 题**,查询源自 OpenAI BrowseComp。
- **核心设计:检索器与 agent 解耦**——不打实时网页,针对**固定 ~100K 人工核验文档语料**检索,保证公平、可复现。
- **三个指标**:
  - **Accuracy**:LLM judge(Qwen3-32B)判定答案正确的比例——因答案是长/开放形式,不能用 EM,故用模型判分。
  - **Recall**:命中"回答该题所必需的证据文档"的召回率——直接衡量"搜到没搜到"。
  - **平均检索轮数**:过程指标,只有伴随 Recall 上升才有意义。

> **数据口径 caveat(已披露)**:标准题数为 830;历史评测产物里 `fully_async_step311` 某文件曾因两次运行结果被追加而变成 1475 题(已隔离、未采用),`step_300` 某文件缺 13 题——这些是**评测脚本工程问题**,与训练无关。评测流程 bug 是"数据产物"层面的,提示了**评测输出也要做去重/条数校验**。

---

## 7. 数据构建 / 清洗方法论与 Insight

把散落在脚本里的做法提炼成可复用的原则:

### 7.1 构建原则
1. **统一数据契约**:不同来源(parquet/json、短程/长程)全部归一到同一 6 字段 schema,下游 rollout/奖励/统计代码零分支。**异构进、同构出。**
2. **标答随样本走,不靠回表**:把 `ground_truth` 通过 `tools_kwargs` 注入工具会话,rollout 结束即可就地打分——避免大规模 join 的一致性风险。
3. **答案一律列表化**(`target: [...]`):QA 答案有多种表面形式,列表 + 归一化 EM 才公平;单值 gt 是误判之源。
4. **来源打标签**(`searchR1_` 前缀):支撑"按源选 reward_fn"和"分数据集上报指标",分析时能定位是哪个源在拖后腿。
5. **可复现与可回溯**:切分固定 `seed=42`;ASearcher 保留 `raw_extra_info`——出问题能回到原始记录。

### 7.2 清洗原则
6. **难度匹配 > 数据量**:ASearcher 的 **reject-sample**(用 Qwen3-8B 采样过滤)只留"够得着但不 trivial"的题——全对无梯度、全错是噪声,RL 数据的价值在中间难度带。
7. **剔除与目标能力无关的样本**:`no-math` 过滤掉数学题——它们靠推理不靠检索,会稀释搜索决策信号。**训练数据要与被训能力对齐。**
8. **评测集抽样要保来源均衡**:test 侧 `groupby(data_source).head(500)`,防止大源淹没小源、指标被单一分布主导。
9. **协议即约束**:system prompt 里"确保 query 唯一""先思考再答"不只是提示,它与下游的重复-query 异常检测、格式奖励状态机一一对应——**prompt 设计和数据/奖励设计要协同**。
10. **评测产物也要校验**:830 题变 1475 / 缺 13 题的教训——评测输出需做条数核对与去重,否则"数据 bug"会被误读成"模型/训练问题"。

### 7.3 一句话总结
> 训练数据的质量不在"多",而在**格式统一(异构进同构出)、标答可靠(列表化+归一化)、难度对齐(reject-sample)、目标纯净(no-math 类过滤)**;评测数据的质量在**解耦可复现(固定语料)+ 抽样均衡 + 产物校验**。

---

## 附录:关键路径

| 内容 | 路径 |
|---|---|
| Search-R1 预处理 | `examples/search_agent_rl/preprocess_search_r1_dataset_new.py` |
| ASearcher 预处理 | `examples/search_agent_rl/preprocess_ASearcher_dataset.py` |
| 奖励打分(EM/格式/效率) | `verl/utils/reward_score/search_r1_like_qa_em.py` |
| 搜索工具 / 结果格式化 | `verl/tools/search_tool.py`(`_format_passages`) |
| 检索服务配置 | `examples/search_agent_rl/config/tool_config/search_tool_config.yaml` |
| 处理后训练数据 | `examples/search_agent_rl/{search_r1_processed,ASearcher}/*.parquet` |

*本文档基于预处理脚本、奖励代码、工具代码与处理后 parquet 样本逐一核对整理;case 为真实样本(截断展示)。*
