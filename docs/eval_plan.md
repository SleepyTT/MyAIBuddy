# Context Management 评估计划（Eval Plan）

> 状态：设计定稿并已实施（2026-06-11）。回放功能延后。
> 关联文档：`design.md`（架构总览，同目录）、`../CLAUDE.md`（开发指引）、`phase1_sliding_window_summary.md` / `phase2_pgvector_rag.md` / `phase3_cross_session_memory.md`（各阶段详细计划）。

## 0. 实施状态与 baseline 记录

| 组件 | 状态 | 位置 |
|---|---|---|
| token 估算 + `context_report` 生成 | ✅ | `context_report.py`，`/debug/run` 每轮推送 `context_report` SSE 事件，`round`/`final` 事件携带 API `usage` |
| Single Run 视图 CONTEXT 区块 | ✅ | `static/debugger.html`（堆叠条、双轨 token、分层明细、tool result token 徽章、生命周期累计） |
| 数据集 — smoke tier | ✅ | `eval/cases/smoke/`，12 用例 / 15 probes（13 needle + 2 负向），五题材，历史 ~1.5k tok/例。快速回归 / 管道自检 |
| 数据集 — long tier | ✅ | `eval/cases/long/`，6 用例 / 15 probes（13 needle + 2 负向），五题材，历史 ~8.7k tok/例（tiktoken），needle 载体 assistant 长回复 800–900 字、针埋中段测 chunking |
| CLI 跑批 | ✅ | `eval/run.py --tier smoke\|long`（需先启动后端） |
| Eval tab | ✅ | debugger 内 [Single Run \| Eval]，策略对比表 + 维度下钻 + probe 列表，列头按 run_id + strategy·model·dataset_version 区分 |
| 用例一键回放 | ⏸ 延后 | 结果 JSON 已自包含回放数据 |

**Smoke baseline（2026-06-11，`2026-06-11_2312_concat.json`）**：strategy=concat，model=deepseek-v4-pro，judge=kimi-k2.5 → ctx 命中 100% · ans 命中 100% · 幻觉 0% · 平均 input 5879 tok/题（本地估算 1729，差值来自上游偶发的计费异常）· 平均延迟 7.5s。另有 grok-4-fast 版（`2026-06-11_2323_concat.json`）同为满分、input 干净（1.3k）、延迟 2.4s。

**Long baseline（2026-06-15，`2026-06-15_1340_concat_long.json`，15 探针版；24 探针版见 `2026-06-15_1846_concat_long.json`）**：strategy=concat，model=grok-4-fast，judge=kimi-k2.5 → ctx 命中 100% · ans 命中 100% · 幻觉 0% · 平均 input 5519 tok/题（**本地估算 8751**，grok 对中文 tokenize 更省，此处估算高于 API 是正常方向）· 平均延迟 1.8s。全量 concat 在两档数据集上都是满分上限；long tier 的 8.7k token 历史是后续压缩策略真正会丢内容的区域，区分度从这里开始体现。

**Phase 1 首轮（2026-06-15，`2026-06-15_2052_window_summary_long.json`）**：strategy=window_summary，model=grok-4-fast，judge=kimi-k2.5，long tier 24 探针 → **ctx 命中 91% · ans 命中 91% · 幻觉 0% · 平均 input 2394 tok（压缩比 0.38，相对 concat 省 57%）· 延迟 16.4s**。详见 `phase1_sliding_window_summary.md` §6。⚠️ 同日有一次错误结果（`...1900_...`，ctx 9%/幻觉 71%）——那是 `ctx_hit` 子串匹配在摘要改写下假阴性 + 幻觉判定漏了"答对不算幻觉"守卫所致的测量伪影，已修 `ctx_hit`（改 LLM 保留判定）与幻觉守卫后重跑，错误结果文件已删除。

**实施中确立的方法论决定**：

1. **eval 不用 `supermind-agent-v1`**：它是上游 agent 封装，内部自跑 agentic loop（2k context 能报出 100k+ 的 usage）且偶发返回空 content，污染效率与准确度指标。eval 默认 `deepseek-v4-pro`。
2. **裁判模型用 `kimi-k2.5`**：`deepseek-v4-flash` 经实测会对明显符合 rubric 的回答误判 NO。裁判原始输出存入结果 JSON 的 `judge_raw` 字段可审计。
3. **载体 = tool result 的针暂缓**：应用的 DB 只持久化 user/assistant 消息，tool 消息不进历史，等 tool result 持久化后再补这一维度。
4. **效率指标双轨呈现**：`avg_input_tokens`（API 真值）与 `avg_estimated_input_tokens`（本地估算）并列，后者对上游计费异常更稳健。

## 1. 背景与目标

当前 `/chat` 把 session 内全部历史暴力 concat 后发给上游 LLM，存在两个问题：

1. 大量无关 context 浪费 token（agentic loop 每轮都重发整个 context，浪费被轮数放大）；
2. context window 被 AI 长回答迅速撑满，导致模型幻觉。

计划引入分层 context 管理（滑动窗口 + 滚动摘要 → pgvector RAG → 跨 session 检索/memory）。**在动 context 管理之前，先建立测量体系**——否则无法回答"新策略到底省了多少 token、答案有没有变差"。

本文档定义：测量机制、效率指标、准确度指标、needle 数据集设计、可视化设计、实现架构。

## 2. 总体框架

- 评估沿**两条轴**：效率（token 成本）与准确度（信息找回能力/幻觉程度）。两者必须**一起报告**——单看效率无意义（context 全删光效率最高），单看准确度也无意义（全量 concat 准确度上限最高）。每个策略是两条轴上的一个点，好策略 = 准确度不降的前提下省 token。
- **所有策略跑同一套固定数据集**，效率与准确度数字才可比。同一次运行同时产出两类指标。
- 测量体系**先于**任何 context 管理改动上线，先量化暴力 concat 的 baseline。

## 3. 测量机制：`context_report`

测量不是日志，而是**一等公民的结构化数据**。后端每次组装 messages 调 LLM 时，同步生成一份 `context_report`，在 `/debug/run` 中作为新 SSE 事件类型 `context_report` 于每轮 LLM 调用前推送。

```jsonc
{
  "type": "context_report",
  "round": 1,
  "layers": [
    // baseline 阶段只有三层；未来扩展为 system/memory/summary/retrieved/recent/current
    { "layer": "history",      "tokens": 6800, "items": 14 },
    { "layer": "current",      "tokens": 85,   "items": 1 },
    { "layer": "tools_schema", "tokens": 350,  "items": 2 },
    // Phase 2（RAG）之后 retrieved 层带明细：
    { "layer": "retrieved", "tokens": 1240, "items": [
        { "msg_id": "…", "source_chat": "…", "position": 12,
          "score": 0.83, "tokens": 310, "preview": "…" }
    ]}
  ],
  "estimated_prompt_tokens": 7235,   // 本地 tiktoken 估算（可分层）
  "api_prompt_tokens": 7410,         // 上游 usage 返回的总量真值
  "full_history_tokens": 7235,       // 不做任何管理时的体积 → 算压缩比
  "context_limit": 128000,
  "candidates": []                   // Phase 2：检索候选全集，含落选项及落选原因
}
```

关键设计点：

- **稳定消息 ID**：每条消息有贯穿全程的唯一 ID，`context_report` 的条目引用它，使"某条原始消息是否进了 context"可确定性回溯（两级判定的前提）。
- **Token 双轨制（已定稿）**：上游 `usage.prompt_tokens` 是总量真值但不能分层；分层明细用本地 **tiktoken（cl100k_base）** 估算。两个数都展示并显示偏差——偏差稳定在几个百分点内即说明估算可信。supermind 的真实 tokenizer 未知，但做相对比较足够准。
- **现在就能跑在 baseline 上**：今天 layer 只有 `history/current/tools_schema` 三层，升级后 layer 自然变多，同一套报表前后可比。

## 4. 效率指标

agentic loop 每轮重发整个 context，因此效率必须按**一次提问的完整生命周期**累计：

| 指标 | 定义 | 回答什么问题 |
|---|---|---|
| `input_tokens` | Σ 各轮 `usage.prompt_tokens` | 大头，context 管理直接优化的对象 |
| `output_tokens` | Σ 各轮 `usage.completion_tokens` | 策略基本不影响它，单独计以免污染对比 |
| `rounds` | 工具调用轮数 | context 质量差 → 模型多搜几轮 → 间接膨胀 input |
| `latency` | 提问到 done 的耗时 | 检索/摘要本身的速度开销 |
| 管理开销 | 摘要调用、embedding 调用花的 token | 省下来的 token 别被管理调用吃回去 |
| 压缩比 | 实际发送 tokens / 全量 history tokens | 管理策略省了多少 |
| 利用率 | prompt_tokens / context_limit | 离撑爆还有多远 |
| 每轮增量 | 第 N 轮 prompt − 第 N−1 轮 | tool result 的膨胀速度 |
| 单条 tool result 体积 | 每个 result 的 tokens | 哪个工具在污染 context（`read_page` 大概率是大头） |

input/output **分开报告**（已定稿）：context 策略几乎只影响 input，且两者单价不同。如需单一排序数字，可加按上游价格比的成本加权和，但分开报告是主体。

## 5. 准确度指标

### 5.1 两级判定（已定稿）

每个 needle 的"找到没找到"测两层，失败原因完全不同：

1. **Context 命中率（retrieval-level）**：needle 所在消息/chunk 是否进入最终组装的 context。从 `context_report` 按消息 ID **确定性判断**，零成本，不需要 LLM。
2. **答案命中率（answer-level）**：最终回答是否包含正确事实。事实型针用子串/正则匹配；软性针（决定/偏好类）用 LLM 判分（已确认接受判分成本）。

诊断逻辑：context 命中但答案错 → 生成问题（context 太乱、模型被干扰）；context 没命中 → 检索/摘要问题。

### 5.2 幻觉率

两个来源计入幻觉：

- **负向探针**：问一个对话中从未出现的事实（"我之前给你的 API key 是什么？"——从没给过）。正确行为是承认不知道；编造即计一次幻觉。
- **Context 未命中但模型给出具体答案**：本该说"不记得"却答了，计一次幻觉。

### 5.3 LLM judge

软性针与负向探针用便宜模型判分，rubric 写在用例文件里。每次回归约多十几次便宜调用，已确认可接受。

## 6. Needle 数据集设计

### 6.1 针的类型

| 类型 | 例子 | 测什么 |
|---|---|---|
| 事实原子 | "staging 数据库端口是 5433" | 基础召回；正则可判 |
| 决定/偏好 | "我们最后决定用 pgvector，不用 Pinecone" | 决策脉络保持；LLM 判分 |
| 代码标识符 | 之前对话定义过的函数名/变量名 | 纯向量检索的弱点（专有名词） |
| 更新型针 | 先说 A，几轮后更正为 B | 能否取到**最新版**；摘要式管理的高危区 |

### 6.2 针的位置

- **深度**：对话早期 / 中期 / 晚期（lost-in-the-middle）。
- **载体**：埋在用户消息 vs **AI 长回答中段** vs tool result。AI 回答载体直接测试 chunking 质量（对应痛点 2）。
- **距离**：针与提问之间隔多少轮 / 多少 token。

### 6.3 问法类型

| 问法 | 例子 | 测什么 |
|---|---|---|
| 直接回忆 | "我之前说的端口是多少？" | 关键词重叠高，BM25 都能中 |
| 改写问法 | "连数据库该用哪个口？"（与针零关键词重叠） | 专测语义检索 |
| 隐式依赖 | "帮我写 staging 的连接字符串"（需要用到针才能做对） | 最接近真实使用，最难 |

### 6.4 干扰项（数据集质量的分水岭）

- 填充对话必须**真实、同题材**——随机噪音下任何检索都满分。
- **近似干扰针**："staging 端口 5433" 旁边放 "本地端口 5432"，测精确度而非只测召回。
- **负向探针**：见 5.2。

### 6.5 题材（已定稿，5 个）

| 题材 | 最适合埋的针 | 理由 |
|---|---|---|
| ML 技术问答（深入讨论某技术） | 代码标识符、更新型针 | 专有名词测向量检索；"先用 A 后改 B"自然 |
| 股票/公司投资价值 | 事实原子 + 近似干扰针 | 数字密集（股价、市盈率、持仓成本）；幻觉高危区，负向探针放这里最有说服力 |
| AI 技术趋势 | 决定/偏好类针 | 观点型内容，测 LLM 判分 |
| 养花种花 | 用户个人事实 | "我家朝北阳台"——为跨 session memory 测试预埋 |
| 健康生活 | 个人事实 + 负向探针 | 编造后果直观，幻觉率指标最有意义 |

**纪律：每个用例内部单一题材**——针和填充同题材才构成真干扰。题材多样性体现在用例之间；将来跨 session 检索时再让多题材同库互扰。

### 6.6 构造方法与规模

- 填充对话用 LLM **生成一次后冻结**（存文件，永不重新生成），针手工插入。数据集必须确定性，否则两次运行分数不可比。
- 脚本里的 assistant 消息**写死**（不真调 LLM 生成历史），只有最后的 probe 真实跑完整 pipeline。
- 第一版规模：**10–12 个用例**（约 4 种针型 × 早/晚两种深度，载体覆盖 user/assistant/tool，加 2 个负向探针、1 个更新型针），每例约 20–40 轮对话。先小而精，跑通后沿表现最差的维度加密。
- **两档数据集（2026-06-11 决定，2026-06-15 long tier 交付）**：
  - **smoke set（`eval/cases/smoke/`，v1-smoke）**：12 例，历史仅 ~1.5k token，任何合理窗口都装得下，因此**测不出 Phase 1 压缩策略的区分度**，但它快且便宜，承担管道/模型/裁判的快速回归职责（已实际抓出 supermind 空回复与 flash 裁判误判两个真问题）。
  - **long set（`eval/cases/long/`，v1-long）**：6 例 × 36–46 轮 × ~8.7k token（tiktoken），needle 载体的 assistant 回复 800–906 字、针埋回复中段以压测 chunking；近似干扰针加密（256/512/1024 vs 384、num_workers 16/4 vs 8、朋友成本 198 vs 242、苯醚甲环唑 1500 vs 代森锰锌 800、尿酸 460 vs 455）；更新型针均带过时旧值（V100→A100、跑步 3km→快走+静蹲、内部工具路线→对客 Q3）。覆盖：5 种针型全到、user×10/assistant×4 载体、direct/paraphrase/implicit/negative 四问法（含 3 个多-needle 多跳算术探针、2 个负向探针）、early/middle/late 深度。
  - 采用 **augment 而非 replace**，保持纵向可比；结果 JSON 带 `dataset_version` 字段（`eval/run.py` 中的 `DATASET_VERSIONS` 映射，换 set 必须 bump），不同版本的分数不可直接对比；`run_id` 含 tier 后缀。
- **long tier 评审（2026-06-15）**：经独立 reviewer subagent 审计，首轮 ACCEPT WITH FIXES，5 项必修已逐一修复并脚本复核通过——核心两点是（a）每个用例结尾的总结性复述会把 needle 泄漏给 recency-only 策略，已改为"详见前文"式不复述具体值；（b）needle 载体回复原本只有 smoke 量级长度，已扩写到 800+ 字并把针压到回复中段，否则 chunking 压测形同虚设。脚本验证：尾部 25% 区间无 needle 泄漏、各针型/载体/问法覆盖达标。

### 6.7 工具结果维度与四 tier 矩阵（2026-06-16 设计；tool-medium 已实施，research 待建）

到目前为止的 smoke / long 数据集都**不含工具调用**——脚本对话只有 user/assistant,实跑时探针也不触发工具(`avg_rounds=1.0`)。这忠实于"过去轮次的工具中间结果不持久化"的应用行为(DB 只存最终回答),但留下一个真实盲点:**单次提问的 agentic loop 内部**,工具结果(尤其 `read_page` 一次可达 `PAGE_TEXT_LIMIT=5000` 字符)是最大的 context 膨胀源,对应最初的痛点 #2,而我们从没测过它。

#### 两轴矩阵:历史长度 × 工具有无

数据集按两条正交轴铺开,填有用的格子:

| | 无工具 | 有工具 |
|---|---|---|
| **短历史** | smoke（管道/模型/裁判自检） | — |
| **中历史** | — | **tool-medium**（一般小任务:几次搜索/读页） |
| **长历史** | long（压缩/chunking,隔离历史维度） | **research**（深度调研:长历史 + loop 内多轮工具) |

#### 两类用途:诊断型 vs 集成型(这是单一变量原则的关键)

- **诊断型(单一变量,隔离一个维度)**:smoke、long、**tool-medium**。每个只变一个量,作用是**定位策略为什么挂**——是历史压缩没搞定,还是工具结果处理没搞定。`tool-medium` 保持**中等历史**,把变量隔离在"工具轮"上,不要又长历史又大工具结果(那会变回混合 tier、归因又糊)。
- **集成/真实型(多变量,还原真实压力)**:**research**(长历史 + loop 内多轮工具,对应用户深度调研场景)。它**不隔离变量**,作用是回答"最难的真实场景下整体扛不扛得住",是验收测试不是诊断。
- 两类**配合用,不是二选一**:research 发现"出问题了",再回到 tool-medium / long 去**归因**。前者发现、后者定位。所以加 research 不破坏干净归因,而是它的上层——前提是诊断型 tier 都在。

#### 工具结果的确定性机制:真实捕获 + 冻结回放(方案 A)

为保证确定性(`read_page` 实时结果每次不同会破坏跨运行可比):

1. **生成时真实捕获**:构造数据集时真的调 `web_search`/`read_page`,把返回**按应用实际工具格式**写进用例(见 §6.8 格式),`read_page` 取真实截断后的 5000 字正文。想测膨胀就挑会产生接近上限的页面。
2. **eval 时冻结回放(方案 A 预置 loop)**:不重新执行工具。哪一轮在生成时决定了调用,eval 就**当它必然按预置的调用和结果走**——harness 直接把冻结的 `[assistant tool_call, tool result]` 塞进 loop,再让模型出最终答案。完全确定,且只测 context 管理对工具结果的处理(膨胀、carry、针存活),不掺入"模型是否决定调工具"的方差。
3. **忠实性约束**:冻结工具轮只挂在**当前被评测探针自己的 loop** 里,不进脚本历史——因为生产里过去的工具结果不持久化。所以 `carrier=tool` 的针测的是"round 1 的工具结果能不能进 round 2+ 的 context",不是"从过去某轮检索回来"。
4. **后端改动**:`/debug/run` 加可选 replay/预置参数;实现见 §10 排期。

#### 分级门控(便宜门控贵,非对称跳过)

eval 流水线按成本递增:smoke(秒级)→ long / tool-medium(分钟级)→ research(最贵)。门控**非对称**:
- "上游便宜 tier 已挂 → 跳过下游贵 tier" 成立(别在已知差的策略上烧资源);
- 但 "long 过了 ≠ tool 一定过"——策略可能文本处理好却栽在 5000 字工具结果膨胀上,所以对**通过的 finalist** 仍需真跑 tool/research。

#### 区分力与排期

`tool-medium` / `research` 的区分力**主要在 Phase 2+**:Phase 1 窗口/摘要对 loop 内工具结果**原样保鲜**(不压缩),所以这两 tier 上 concat≈window_summary,Phase 1 阶段只量个膨胀基线;真正拉开差距要等 Phase 2 对工具结果做 chunk/检索(深度调研里几个 5000 字结果叠起来,正是 concat 爆炸、RAG 价值最大化处)。

**实现优先级**:`tool-medium` 排 Phase 2 一开始(信号最干净、最便宜,直接服务 chunking 调参);`research` 排 Phase 2 中后段(构造和跑都最贵,塔尖、只对 finalist 跑、跑得最少)。

**实施状态(2026-06-16)**:`tool-medium` 已交付——`/debug/run` 加 `tool_rounds` 冻结回放参数(方案 A,工具不重跑、单次出答案)、`context_report` 的 `tool_loop` 层携带工具结果文本供 `ctx_hit` 判定、`eval/run.py` 支持探针级 needle 并集解析与 `carrier:"tool"` 命中判定、`DATASET_VERSIONS` 登记 `v1-tool-medium`/`v1-research`。`research` tier 仅登记版本号,用例与调高 `MAX_TURNS` 待 Phase 2 中后段再建。

**一个实现依赖**:当前 `MAX_TURNS=3` 上限,"loop 内多轮工具"最多两三轮就被强制收尾。`research` 的深度调研多工具场景可能需要调高 `MAX_TURNS`,否则模拟不出真正的多轮累积——Phase 2 实现 research 时一并考虑。

#### tier 登记

`DATASET_VERSIONS` 扩为 `{smoke, long, tool-medium, research}`,各自 `v1-*`;`--tier` 已支持;`run_id` 含 tier 后缀。

### 6.8 用例文件格式

```jsonc
// eval/cases/<tier>/needle_001.json — smoke / long(无工具)
{
  "id": "needle_001",
  "topic": "stocks",
  "conversation": [ /* 完整脚本化的 user/assistant 消息序列，约 20–40 轮 */ ],
  "needles": [
    { "id": "n1", "content": "5433", "position": 6, "carrier": "user", "type": "factual" }
  ],
  "probes": [
    { "question": "帮我写 staging 的连接字符串", "needle_ids": ["n1"],
      "judge": { "method": "regex", "pattern": "5433" }, "style": "implicit" },
    { "question": "我之前给过你 API key 吗？是什么？", "needle_ids": [],
      "judge": { "method": "llm", "rubric": "正确行为是表示没有/不记得" }, "style": "negative" }
  ]
}
```

工具 tier（tool-medium / research）的探针多一个 `tool_rounds` 字段——生成时真实捕获、eval 时冻结回放（§6.7 方案 A）。埋在工具结果里的针用 `tool_round`（指向 `tool_rounds` 下标）代替对话 `position`，`carrier: "tool"`：

```jsonc
// eval/cases/tool-medium/tool_001.json
{
  "id": "tool_001", "topic": "ai_trends",
  "conversation": [ /* 中等长度 user/assistant 历史 */ ],
  "probes": [
    {
      "question": "查一下最新的 X，然后告诉我里面提到的版本号",
      // 预置 loop：eval 时按此顺序冻结回放，不重新执行工具
      "tool_rounds": [
        {
          "assistant": { "role": "assistant", "content": null, "tool_calls": [
            { "id": "call_1", "type": "function",
              "function": { "name": "read_page", "arguments": "{\"url\":\"https://…\"}" } } ] },
          "tool": { "role": "tool", "tool_call_id": "call_1",
                    "content": "<read_page 真实截断后的 5000 字正文>" }
        }
      ],
      "needles": [ { "id": "t1", "content": "v4.2", "tool_round": 0, "carrier": "tool", "type": "factual" } ],
      "needle_ids": ["t1"], "style": "direct",
      "judge": { "method": "regex", "pattern": "v4\\.2" }
    }
  ]
}
```

## 7. 汇总报表（体系的终点产物）

每个策略跑完整个数据集产出一行；该上线哪个策略，看这张表：

```
策略           ctx命中  ans正确  幻觉率   input tok/题  output tok/题  rounds  延迟
全量concat      100%     83%     0%      21,400        510           1.8    8.2s
窗口+摘要        72%     70%    11%       6,100        490           2.1    9.0s
RAG top-5       91%     85%     4%       4,800        505           1.6    9.5s
```

再按维度下钻（深度/载体/问法分桶的命中率），回答**为什么**输。

## 8. 可视化设计（Context Debugger）

按"用户带着什么问题来看图"组织，四个问题对应四块：

### 8.1 "这次请求的 context 长什么样？" → Single Run 视图（现有页面增强）

每轮卡片顶部加 **CONTEXT 区块**：

```
┌─ ROUND 1 ──────────────────────────────────────────────┐
│ CONTEXT   est 7.2k · api 7.4k (+2.4%) · 5.8% of 128k   │
│ ████████████████░░░░  history 94% │ current 1% │ tools 5%│
│ ▸ history      6,800 tok · 14 msgs                     │
│ ▸ retrieved    1,240 tok · 4 chunks      ← Phase 2 后   │
│     ✓ 0.83  「FastAPI 部署」pos 12  310tok  "…预览…"     │
│     ✗ 0.41  「闲聊」pos 3  (落选：低于阈值)               │
│ ▸ current         85 tok                               │
├────────────────────────────────────────────────────────┤
│ ASSISTANT → tool call: web_search(…)                   │
│ TOOL RESULT  web_search   ⚠ +2,130 tok 进入下一轮        │
└────────────────────────────────────────────────────────┘
```

- 堆叠条：各 layer 占比，颜色按 layer 固定。
- 双轨 token 数及偏差。
- 可展开分层明细；检索候选列表**含落选项**及分数、落选原因（低于阈值 / 被预算挤掉）——"模型为什么不知道 X"的两种病因修法完全不同，所以落选项必须展示。
- 现有 TOOL RESULT 区块加 `+N tok` 徽章（下一轮膨胀的直接来源）。
- done 后状态栏显示整次生命周期 input/output 累计与每轮增量。

### 8.2 "策略 A 和 B 谁好？" → 策略对比表（Eval 视图首屏）

第 7 节的汇总表，每策略一行；选两个结果文件 = 对比模式并排。

### 8.3 "输在哪个维度？" → 维度下钻表

行 = 维度分桶（深度早/中/晚、载体 user/assistant/tool、问法直接/改写/隐式），列 = 策略，格 = 该桶命中率，低于阈值标红。例："RAG 在针埋于 assistant 长回答时命中率 50%" → 指向 chunking 问题。

### 8.4 "这个 case 为什么挂？" → 用例下钻 + 一键回放（回放延后）

用例列表每行两个徽章（ctx ✓/✗、ans ✓/✗）+ 幻觉标记。**一键回放功能延后实现**（2026-06-11 决定：第一版只做 CLI 跑批 + Eval tab 看结果；回放作为 debugger 的后续功能再加）。回放设计保留备用：点击失败用例 → 跳到 Single Run 视图回放该次运行：完整对话 + 每轮 `context_report`，针所在消息高亮——绿 = 进了 context，红 = 被裁掉，旁注其检索分数与落选原因。结果 JSON 自包含 `context_report`，已为回放预留数据，届时纯前端工作。

```
┌─ [Single Run] [Eval] ───────────────────────────────────────┐
│ Results: [2026-06-11_concat ▾] vs [2026-06-11_rag-top5 ▾]   │
│                                                             │
│ 策略         ctx命中  ans正确  幻觉   in-tok/题  rounds  延迟 │
│ concat       100%     83%     0%     21.4k      1.8     8.2s│
│ rag-top5      91%     85%     4%      4.8k      1.6     9.5s│
│                                                             │
│ 维度下钻        concat   rag-top5                            │
│ 载体=assistant  100%     50% ⚠                              │
│ 问法=改写        90%     88%                                 │
│                                                             │
│ 用例列表                                                     │
│ needle_003  ctx✗ ans✗  [回放→]   ← 点击跳 Single Run 回放    │
│ needle_007  ctx✓ ans✗  [回放→]                              │
└─────────────────────────────────────────────────────────────┘
```

## 9. 实现架构（已定稿：CLI 跑批 + Debugger 看结果）

```
eval/cases/*.json ──> eval/run.py --strategy=X ──> eval/results/2026-06-11_rag-top5.json
                         （CLI 跑批，可进 CI）          （自包含：每题的 context_report、
                                                        回答、判定结果——可回放，无需重跑）

debugger.html 加 tab 切换：[Single Run | Eval]（同页加 tab，回放复用 Single Run 渲染代码）
  Eval tab 经 GET /debug/eval/results 列出结果文件，选 1–2 个加载渲染。
```

- 跑批放 CLI：长任务、十几次 LLM 调用，不适合浏览器里等；可进 CI。
- 结果文件自包含所有 `context_report`，debugger 读文件即可渲染与回放。
- 后端只加两个只读 debug 端点：列结果文件、读结果文件（与现有 debug 端点一致，无 auth）。

## 10. 实施顺序

1. **Instrumentation**（✅）：tiktoken 计数工具、`context_report` 生成与 SSE 推送、Single Run 视图 CONTEXT 区块 —— 先量化 baseline。
2. **数据集 + 跑批**（✅）：`eval/cases/{smoke,long}/` 用例构造（LLM 生成填充后冻结）、`eval/run.py`、正则/LLM 两种 judge、结果 JSON。
3. **Eval 视图**（✅）：debugger 加 tab、对比表、维度下钻（用例回放延后）。
4. context 管理本体分阶段推进，每阶段用本体系回归：
   - **Phase 1**（✅ 2026-06-15）：滑动窗口 + 滚动摘要。
   - **Phase 2**（pgvector RAG）：开始时先做 **tool-medium tier**（✅ 2026-06-16：§6.7,工具结果维度,`/debug/run` 冻结回放 replay 参数已实现,数据集已建）；中后段做 **research tier**（长历史 + 多工具,塔尖、最贵、只对 finalist 跑,需评估调高 `MAX_TURNS`）。
   - **Phase 3**（跨 session + memory 注入）：需新建 multi-session tier（§3 跨 session 检索）。

## 11. 已定稿决定清单

| 决定 | 结论 |
|---|---|
| 分层 token 计数 | tiktoken（cl100k_base）本地估算 + API usage 总量真值，双轨展示偏差 |
| 准确度判定 | 两级：context 命中（确定性）+ 答案命中（正则/LLM judge） |
| 幻觉测量 | 负向探针 + "context 未命中却给出具体答案"计数 |
| 软性针 LLM 判分 | 接受 |
| 数据集题材 | ML 技术问答、股票/投资、AI 趋势、养花、健康生活；每用例单一题材 |
| 数据集构造 | LLM 生成填充后冻结，针手工插入，assistant 消息写死；首版 10–12 例 |
| 效率口径 | 按提问完整生命周期累计；input/output 分开报告 |
| 跑批方式 | CLI（`eval/run.py`），结果自包含 JSON |
| 可视化载体 | debugger.html 同页加 [Single Run | Eval] tab |
| 用例回放 | 延后（结果 JSON 已自包含回放所需数据，后续纯前端实现） |
| 数据集 tier（2026-06-16） | 两轴矩阵（历史长度 × 工具有无）四 tier：smoke / long / **tool-medium**（诊断,已交付）+ research（§6.7,仅登记版本,用例待建）；诊断型单一变量、research 集成验收；分级门控非对称跳过 |
| 工具结果确定性（2026-06-16） | 真实捕获 + 冻结回放（方案 A 预置 loop）；只挂当前探针 loop、不进历史；`/debug/run` 加 replay 参数 |
| 工具 tier 排期（2026-06-16） | tool-medium 排 Phase 2 起始、research 排 Phase 2 中后段（需评估调高 `MAX_TURNS`） |
