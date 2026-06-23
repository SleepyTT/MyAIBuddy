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
| 必保 | 当前问题(current question)、tools schema | 永不压缩/丢弃 |
| 高 | 当前 loop 的工具结果(模型为答这题刚 fetch 的) | 体积最大,是被压/截/检索的主要对象(§6) |
| 中 | 最近历史(verbatim 窗口) | 按 token 装,装不下的下沉到摘要/被丢 |
| 低 | 更早历史 | 进摘要 / 被丢 / 待检索 |

`剩余预算 = B − tokens(tools schema) − tokens(current question)`,再按上表自高而低填。

**要处理的是"总量超预算",不只是"单条工具结果超预算"**——紧预算下常见的是总量爆(如 4k 预算下,光 long 的 8.7k 历史没有任何工具也已溢)。三种情形统一走同一套优先级:①历史总量超 ②单条工具结果超 ③两者都超。

## 5. 三个策略在预算压力下的退化

| 策略 | 压缩时看不看"当前问题" | 预算不足时的行为 | 后果 |
|---|---|---|---|
| **concat** | 位置盲 | **截断**超预算内容(见 §6),保最近、保问题 | 针落在被截区就丢 |
| **window_summary** | **query 盲**(不看问题就压) | 历史按 token 窗口保最近、其余进**滚动摘要**;超预算工具结果**总结**到适配(§6) | specific 针(版本号/函数名)可能被摘要压没 |
| **RAG**(Phase 2) | **query 感知**(按相关度挑) | 把历史/工具结果切块、按与问题相关度**检索** top-k 填进预算 | 最可能保住针 |

### 可证伪的核心假设

> **紧预算下,准确度 RAG > window_summary > concat;而 token 成本 RAG / window ≪ concat。**

差异根源是三者**对"当前问题"的知情程度不同**:concat 位置盲、summary query 盲、RAG query 感知。这把"加预算维度"从工程变成**实验**——它让 summary 的盲点和 RAG 的价值第一次可量化。budget sweep 就是验证这条假设。

## 6. 单条工具结果超预算的处理(用户拍板)

| 策略 | policy |
|---|---|
| **concat** | **截断**工具结果到适配预算的位置(保当前问题,绝不截掉问题)。截断方向:头截断(保留正文开头,与 `PAGE_TEXT_LIMIT` 一致);我们的针埋在结果**中段**,所以预算紧到约结果一半以下时针被截掉 → ctx-miss,可测、确定。 |
| **window_summary** | 对超预算的工具结果做**总结**,压到一个独立的「工具结果摘要预算」(如 ~N token)使其 fit。这是 **query 盲** 的压缩,针可能保住也可能被压掉——正是要测的。 |
| **RAG**(Phase 2) | 把工具结果**切块 + 按问题检索**,只把最相关的几段放进窗口。query 感知,最可能精确保住针。 |

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
| 默认 budget | 128k,保证行为与现状一致、无回归 |
| budget 档位 | 128k / 32k / 16k / 8k / 4k(§8) |
| concat 超预算 | 截断工具结果(头截断),保当前问题(§6) |
| window_summary 超预算 | token 窗口 + 工具结果总结(独立摘要预算),计入管理开销(§6) |
| RAG 超预算 | 切块 + 按问题检索 top-k(Phase 2,本阶段不实现) |
| 核心假设 | 紧预算下 准确度 RAG > window_summary > concat;token RAG/window ≪ concat(§5) |
| 结果标注 | `context_budget` 字段 + `run_id` 后缀 + debugger 列头(§7) |
| 幻觉 | 本阶段才被真正激活(紧预算 → ctx-miss → 诚实/编造分流) |

## 12. Sweep 结果(2026-06-22,model=grok-4-fast,judge=kimi-k2.5)

完整 20 格(2 策略 × 2 tier × 5 budget);judge 重试加固后本轮 0 个 judge_error。

| 策略 / tier | 128k | 32k | 16k | 8k | 4k |
|---|---|---|---|---|---|
| concat · long(ctx/ans/幻觉) | 1.0 / .95 / 0 | 1.0 / .95 / 0 | 1.0 / .95 / 0 | **.55 / .59 / .17** | **.18 / .32 / .38** |
| window_summary · long | .86 / .86 / .04 | .95 / .95 / 0 | .91 / .82 / .04 | .91 / .95 / 0 | **1.0 / .95 / 0** |
| concat · tool-medium | 1.0 / .83 / 0 | 1.0 / 1.0 / 0 | 1.0 / 1.0 / 0 | .83 / .83 / 0 | **.67 / .50 / 0** |
| window_summary · tool-medium | 1.0 / 1.0 / 0 | 1.0 / 1.0 / 0 | 1.0 / 1.0 / 0 | **.50 / .67 / .12** | **.17 / .17 / .38** |

avg input(tok):concat long 5.5k→2.9k、window_summary long 稳定 ~2.3–2.7k;concat tool-medium 8.7k→3.5k、window_summary tool-medium 8.1k→2.4k。

### 三个核心发现

1. **"溢出 regime" 成功激活,幻觉指标第一次有信号。** concat·long 随预算收紧:128k–16k 全装下(ctx 1.0、幻觉 0)→ 8k 丢历史(ctx .55、幻觉 **.17**)→ 4k 重截(ctx .18、幻觉 **.38**)。正是预判的因果链——预算不足 → 被迫丢信息 → 模型编造。此前所有 tier 幻觉恒为 0,现在量出来了。

2. **历史维度:window_summary 完胜 concat(验证假设)。** 紧预算下 long,concat 崩(ctx .55→.18、幻觉冲到 .38),window_summary 全程 ctx .91–1.0、幻觉 ~0,且 token 只有 concat 的一半(~2.5k vs 5k+)。摘要把挤出窗口的历史压缩保留,concat 直接丢整轮。

3. **工具结果维度:反而 concat > window_summary(推翻简单假设,且更要紧)。** tool-medium 紧预算下,concat(头截断)8k ctx .83 / 4k .67;window_summary(对工具结果做 **query 盲**摘要)8k ctx **.50** / 4k **.17**——更差。原因:针是页面里一个**具体事实**(版本号/函数名/数字),query 盲的摘要把它当噪音压掉,而头截断只要预算够到针位(中段)就保住。**对"大工具结果里的具体针",盲压缩比截断还糟。**

### 推论(注意边界:本轮没测 RAG)

**本轮只跑了 concat 和 window_summary,RAG 还没实现、零数据点。** 所以分两层:

- **测到的(有数据)**:历史维度 window_summary > concat;工具结果维度 concat > window_summary。合起来——我们试过的**两种 query 盲压缩(截断 / 摘要),对"大工具结果里的具体针"都救不了**(一个截掉、一个压没)。
- **推断的(假设,待 Phase 2 验证)**:既然盲压缩不行,一个**看问题**的方法(按相关度挑 chunk)原理上应能保住针,RAG 是这样的方法。但 §5 那条 "RAG > window_summary > concat" 始终是**未验证假设**——RAG 行不行、是否真的更优,**得等 Phase 2 真跑出数才算数**,本轮不下结论。sweep 只是把"想试 RAG"从随口一说升级成"有数据支撑的假设"。

**测量注意(重要):很多格子里预算"没咬上"(non-binding)。** 预算 B 只有小于策略本来就会发送的大小时才起约束作用。`window_summary` 靠 `WINDOW_TURNS=6` 已压到 ~3.4k,所以 8k–128k 对它全没咬——那几格是同一配置的**噪声重复采样**;`tool-medium` 全量 ~16k,32k 以上对它也不咬。叠加单次跑 + 实时摘要 + LLM 裁判的非确定性 + 小样本(tool-medium 仅 6 针探针,翻 1 个 = .17),非咬格子的上下浮动就是**噪声底,不是趋势**——直接证据:同一 `(concat,tool-medium,128k)` 跑两次得 1.0 与 .83。因此 **window_summary·long 的非单调、concat·tool-medium 128k 偏低都是噪声**,只有"咬上"的格子(concat·long 8k/4k、concat/window_summary·tool-medium 8k/4k)才是信号。要让非咬区也有干净结论,需每格重复 N 次取均值±区间,或把预算压到各策略自然大小以下(window_summary 要 <3k 才咬)。结果文件:`eval/results/2026-06-22_*_{concat,window_summary}_{long,tool-medium}[_bNk].json`。
