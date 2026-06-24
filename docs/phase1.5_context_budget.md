# Phase 1.5 设计:可调 context budget 与 budget-aware 策略

> 状态:设计待评审(2026-06-22)。本文档定稿、用户认可后再实现。
> 关联:`eval_plan.md`(评测体系,§6.7 工具维度)、`phase1_sliding_window_summary.md`(滑动窗口+摘要)、`phase2_pgvector_rag.md`(RAG 检索)、`context_strategies.py` / `context_report.py` / `main.py` / `eval/run.py`。

## 1. 背景与动机

到目前为止,所有数据集(smoke ~1.5k、long ~8.7k、tool-medium 历史 ~3.2k + 工具结果最多 ~16.6k)**全程 token 都远不到** `CONTEXT_LIMIT`(假设 128k)。后果:

- `concat`(全量)在每个 tier 上都是 100% 满分——什么都没丢;
- **幻觉率几乎恒为 0**——不是模型不幻觉,是没有任何 case 把它逼到"必须丢信息"的境地。

但生产里幻觉真正高发的场景恰恰是:**上下文逼近/超过可用预算 → 被迫做 context 管理 → 丢了关键信息 → 模型该说"不记得"却编造**。我们至今**没测过这个 regime**。当前评测只覆盖"自愿压缩省 token"(松预算下的成本优化),没覆盖"被迫取舍"(紧预算下的正确性失守)。

本阶段补上它:把 **context budget 做成可调参数**,在紧预算下逼各策略真正丢/压信息,从而第一次让**幻觉指标**和**策略间准确度区分度**产生信号。

## 2. 关键认知:`CONTEXT_LIMIT` 现在只是"显示数",不驱动行为

实现前必须澄清一个坑:

> 当前 `context_report.py` 的 `CONTEXT_LIMIT = 128_000` **只用来算"利用率 %"**,策略本身(`context_strategies.py`)压根不读它。`window_summary` 是按**轮数**(`WINDOW_TURNS=6`)切的,不是按 token 预算切的。

所以**单把这个常数调小什么都不会变**,只是利用率读数变了。要让"可调预算"真能测出东西,核心工作量在于:**让策略变成 budget-aware ——按 token 预算压,而不是按轮数压。** 这也顺带让 `window_summary` 更贴近现实(按轮数切本就不合理:一条 30k 的工具结果和一句"好的"各算一轮)。

## 3. 设计总览

1. **预算是横切的 eval 参数,不是新数据集 tier。** 给 eval 加 `--context-budget`,**复用现有 long、tool-medium 数据集**在不同预算下跑。同一套针,松预算下 `concat ≈ window_summary`,紧预算下立刻拉开——**区分度从预算里来,不必造新数据**。
2. **策略按 token 预算 `B` 组装**:默认 `B = 128k → 行为与现在完全一致**(不破坏任何回归)。调小 `B` 才触发压缩/截断。
3. **三个策略各按自己的哲学在预算压力下退化**(§5)。
4. **预算由我们的组装层强制**(策略压到 `B` 以内),与被测模型真实窗口无关——所以即便用 128k 模型,设 `B=8k` 也能逼出紧预算行为,不需要真的换小窗口模型。

## 4. 预算预留与优先级模型

预算 `B` 是**对一次上游调用的 prompt(input)token 上限**。组装时按优先级填充,先保命、再压可压的:

| 优先级 | 内容 | 处理 |
|---|---|---|
| 必保(固定预留) | tools schema + 当前问题 + **工具结果配额**(`TOOL_RESULT_QUOTA`,默认 1000 tok) | 永不被历史挤掉;工具结果由各策略压进自己那份配额(§6) |
| 填充 | 历史,**最近优先、纯按 token** | 装得下→原样保留;装不下的溢出部分,concat **丢弃**、window_summary **进滚动摘要** |

`历史预算 = B − tokens(tools schema) − tokens(当前问题) − 实际工具结果占用(≤ 配额)`,再按"最近优先"填历史。

**关键(2026-06-23 定稿,工业界标准做法)**:① **纯 token 驱动,去掉了 `WINDOW_TURNS` 硬帽子**——预算装得下就不压,只有溢出才压(所以大预算下 window_summary ≡ concat,这是诚实行为);② **工具结果有自己的固定配额**,不抢历史预算、也不许撑爆 context(industry:不把 30k 原文直接灌进 context);③ 另设一个**正交的"主动省钱"旋钮** `proactive_compress_to`:即使预算更大也把历史压到 N token(默认关)——自愿压缩只发生在这里。

## 5. 三个策略在预算压力下的退化

| 策略 | 压缩时看不看"当前问题" | 预算不足时的行为 | 后果 |
|---|---|---|---|
| **concat** | 位置盲 | **截断**超预算内容(见 §6),保最近、保问题 | 针落在被截区就丢 |
| **window_summary** | **query 盲**(不看问题就压) | 历史按 token 窗口保最近、其余进**滚动摘要**;超预算工具结果**总结**到适配(§6) | specific 针(版本号/函数名)可能被摘要压没 |
| **RAG**(Phase 2) | **query 感知**(按相关度挑) | 把历史/工具结果切块、按与问题相关度**检索** top-k 填进预算 | 最可能保住针 |

### 可证伪的核心假设

> **紧预算下,准确度 RAG > window_summary > concat;而 token 成本 RAG / window ≪ concat。**

差异根源是三者**对"当前问题"的知情程度不同**:concat 位置盲、summary query 盲、RAG query 感知。这把"加预算维度"从工程变成**实验**——它让 summary 的盲点和 RAG 的价值第一次可量化。budget sweep 就是验证这条假设。

## 6. 工具结果配额的处理(用户拍板,2026-06-23 定稿)

工具结果有一份**固定配额**(`TOOL_RESULT_QUOTA`,默认 **1000 tok**),像系统提示/当前消息那样的特殊预留;每个策略用自己的方法把工具结果压进这份配额,**在同一目标大小下 apples-to-apples**:

| 策略 | 把工具结果压进配额的方法 |
|---|---|
| **concat** | **头截断**(保留正文开头,与 `PAGE_TEXT_LIMIT` 一致)。针埋在结果**中段(~45%)**,而 1k 配额 ≈ 头部 ~13% → **针几乎必被截掉,concat 在 tool-medium 上 ctx≈0**,这是"头截断丢中段"的真实表现。 |
| **window_summary** | 对工具结果做 **query 盲总结**压到配额。具体针(版本号/函数名)可能被当噪音压没——已实测:30k 结果被压成 ~50 tok 短摘要、针丢失。 |
| **RAG**(Phase 2) | **切块 + 按问题检索**,只把最相关的几段放进配额。query 感知,最可能精确保住针。 |

配额值 1k 的取舍:30k 字符页 ≈ 6–7k tok,压到 1k 是 ~6× 上限,既限住膨胀、摘要/检索又有空间装关键事实;可调。**注意:工具配额是固定值,不随总预算变**——所以 tool-medium 上"概谁能在压缩后保住针"是**策略间**的对比(concat<window<RAG 待验证),不是跨预算的曲线。

补充:
- `window_summary` 的工具结果总结是**额外一次 LLM 调用**,其 token 算进**管理开销**(扩展现有 `summary_cost_tokens`)——别让省下的 token 被摘要调用吃回去。
- concat 的截断、window 的总结、RAG 的检索都要在 `context_report` 里**留痕**(被截/被压/被检索了什么),供 debugger 回放与归因。

## 7. 指标与结果标注

- **结果 JSON 顶层加 `context_budget` 字段**;`run_id` 带后缀(如 `..._window_summary_long_b8k`);debugger Eval tab 列头显示 budget。这样**同数据集、不同预算**的结果能并排比。
- **新增 per-probe 信号 `forced_truncation`**(bool:策略是否为 fit 预算而丢/压了内容),配合现有 `ctx_hit` / `ans_hit` / `hallucination`。
- **幻觉指标终于被激活**:紧预算下 ctx-miss 上升,一部分 → 模型诚实说"不记得"(对),一部分 → 编造具体答案(幻觉)。负向探针 + "ctx-miss 却给具体答案"两路计数正是为此设计。
- `context_report` 的 `context_limit` 字段改为携带本次的 `B`(而非写死 128k),利用率读数随之有意义。

## 8. budget 档位与 sweep 计划

档位(用户定):**128k(baseline,行为不变)/ 32k / 16k / 8k / 4k**。各档对当前数据的"咬合"预期:

| budget | 对 long(~8.7k 历史) | 对 tool-medium(单工具 ~9–10k / 多轮 ~16.6k) |
|---|---|---|
| 128k | 全装下(=现状) | 全装下(=现状) |
| 32k | 装下,不咬 | 多轮也装下,基本不咬 |
| 16k | 装下,不咬 | 多轮(16.6k)开始临界 |
| 8k | 历史临界/溢 | 单工具结果即溢 → 开始逼压 |
| 4k | 历史即溢(无工具也溢) | 全面逼压 |

**sweep**:`concat` 与 `window_summary` × {long, tool-medium} × 5 个 budget。smoke 可选(~1.5k,要到 4k 以下才咬,主要留作 sanity)。产出一张 **budget × strategy** 网格,报告里画"准确度–预算"和"效率–预算"两条曲线。

## 9. 实现改动清单与分期

**改动文件**:

- `context_report.py`:`CONTEXT_LIMIT` → 由调用方传入的 `budget`;report 携带本次 budget;`tool_loop` / 新增 `truncated` / `summarized` 留痕。
- `context_strategies.py`:`assemble_context` 接受 `budget`;`window_summary` 改为 **token 预算窗口**(保最近、按 token 装,其余进摘要)+ 超预算工具结果总结;`concat` 加超预算截断。
- `main.py`:`/debug/run` 与 `/chat` 透传 `budget`;在 loop 内对工具结果应用各策略的超预算 policy(§6)。
- `eval/run.py`:`--context-budget`(可单值或扫描列表);把 budget 写进 `/debug/run` 请求与结果 JSON / `run_id`;聚合按 budget 分组。
- `static/debugger.html`:Eval 列头显示 budget;Single Run 显示被截/被压留痕。

**分期**:

- **Phase 1.5(本阶段,实现)**:budget-aware **concat(截断)** + **window_summary(token 窗口 + 工具结果总结)**;budget 参数贯通;sweep 跑起来。
- **Phase 2(后续)**:RAG 检索版退化(§5/§6 的第三行)——切块 + 按问题检索填预算。本文档把 RAG 行为一并设计,确保预算机制对它前向兼容,但**不在本阶段实现**。

## 10. 不在范围内

- **跨问题的工具结果召回**:"答 Q1 时调了工具,在 Q2 问它的结果"——生产里工具结果**不持久化**,永远 ctx-miss,故意不测(见 `eval_plan.md` §6.7 忠实性约束)。要测它需先实现"工具结果持久化 + 检索",属 Phase 3 范畴。
- **被测模型的真实硬窗口**:budget 是我们组装层施加的**模拟预算**,不依赖模型真实上限。两者解耦是为了用大窗口模型也能跑紧预算实验。

## 11. 待定 / 已决定清单

| 项 | 结论 |
|---|---|
| 预算是参数还是新 tier | **横切参数**,复用现有数据集(§3) |
| 历史窗口怎么定(2026-06-23 改) | **纯 token 驱动、去掉 `WINDOW_TURNS` 硬帽子**(工业界做法):装得下不压,溢出才压。默认大预算下 window_summary ≡ concat |
| budget 档位 | 128k / 32k / 16k / 8k / 4k(§8) |
| 工具结果(2026-06-23 定) | **固定配额** `TOOL_RESULT_QUOTA=1000`;concat 头截断进配额、window_summary 总结进配额、RAG 检索进配额(§6) |
| 主动省钱旋钮(2026-06-23 加) | `proactive_compress_to`:即使预算更大也把历史压到 N token,默认关、与硬预算正交 |
| RAG | 切块 + 按问题检索 top-k(Phase 2,本阶段不实现) |
| 核心**假设**(未验证) | 紧预算下 准确度 RAG > window_summary > concat;token RAG/window ≪ concat(§5)。**待 Phase 2 真跑 RAG 才能下结论** |
| 结果标注 | `context_budget` / `proactive_compress_to` 字段 + `run_id` 后缀(`_bNk`/`_pcNk`)(§7) |
| 幻觉 | 本阶段才被真正激活(紧预算 → ctx-miss → 诚实/编造分流) |

## 12. Sweep 结果(2026-06-22,model=grok-4-fast,judge=kimi-k2.5)

redesign 后(纯 token 历史 + 固定 1k 工具配额)重跑,12 格(2 策略 × 2 tier × **4k/8k/16k**);judge 重试加固后 0 个 judge_error。**32k/128k 未跑**——long 在 16k 已全装下(ctx 1.0),tool 配额固定,这两档只是非咬的平顶、无新信号。结果文件:`eval/results/phase1_5/`。

| 策略 / tier(ctx/ans/幻觉) | 16k | 8k | 4k |
|---|---|---|---|
| concat · long | 1.0 / .91 / 0 | **.55 / .68 / .12** | **.18 / .32 / .21** |
| window_summary · long | 1.0 / .95 / 0 | **.82 / .82 / .08** | **.91 / .91 / .08** |
| concat · tool-medium | **0 / 0 / .25** | **0 / 0 / .5** | **0 / 0 / .25** |
| window_summary · tool-medium | **0 / .17 / .38** | **0 / .33 / .38** | **0 / .17 / .25** |

### 三个核心发现

1. **"溢出 regime" 激活,幻觉指标在两个 tier 都有信号。** concat·long 随预算收紧:16k 全装下(ctx 1.0、幻觉 0)→ 8k 丢历史(ctx .55、幻觉 **.12**)→ 4k 重截(ctx .18、幻觉 **.21**)。预判的因果链坐实——预算不足 → 被迫丢信息 → 模型编造。

2. **历史维度:window_summary 完胜 concat(假设验证)。** 紧预算下 long,concat 崩(ctx .55→.18、幻觉 .21),window_summary 全程 ctx .82–1.0、幻觉 ~0。摘要把挤出窗口的历史压缩保留,concat 直接丢整轮。**佐证 redesign 生效:16k 两者都 ctx 1.0、input 都正好 5539(完全相同)——预算装得下时 window_summary ≡ concat,不再自愿压缩。**

3. **工具结果维度:两种盲压缩都 ctx≈0,且对预算平。** 固定 1k 配额下,concat 头截断把中段针(~45%)切掉、window_summary 盲摘要(30k → ~50 tok)把具体针压没——**两者 ctx 都恒为 0,跨预算不变**(配额固定)。这是设计预告的"盲压缩两条路都不行 → 留给 RAG"的结果。

### 两个测量注意

- **tool-medium 里 ans > ctx**(window 的 ans .17–.33 而 ctx 0):针被压掉后模型靠**训练先验**蒙对了一些——有几个针是真实世界事实(`scaled_dot_product_attention`、2300mg 钠)模型本来就知道。所以 **ctx(确定性)才是"信息有没有真进 context"的真信号,ans 会被先验高估**——这正是留两个指标的价值。
- **tool-medium 当前不区分 concat vs window**(都 ctx≈0、都失败),它现在是"两种盲压缩都不行"的**演示 + RAG 动机**,不是策略判别器;等 Phase 2 RAG 做 query 感知检索接管工具结果才会拉开。

### 推论(边界:RAG 还没实现、零数据点)

**测到的**:历史维度 window_summary > concat;工具结果维度两种盲压缩(截断/摘要)都救不了具体针。**推断的(假设,待 Phase 2 验证)**:既然盲压缩不行,一个 query 感知的方法(按相关度挑 chunk)原理上应能保住针——RAG 是候选。但 §5 那条 "RAG > window_summary > concat" 始终是**未验证假设**,得等 Phase 2 真跑出数才算数。这个 sweep 把"为什么要 RAG"从随口一说升级成有数据支撑的动机。
