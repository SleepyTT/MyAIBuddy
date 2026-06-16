# Phase 1 — 滑动窗口 + 滚动摘要（`strategy = "window_summary"`）

> 状态：计划，未实施。
> 关联：`design.md`（路线总览）、`eval_plan.md`（评估体系与指标定义）、`phase2_pgvector_rag.md`（下一阶段）。

## 1. 目标

解决最初提出的两个痛点中"context window 被长回答迅速撑满 → 幻觉"的那一个，并以最低工程成本把 token 增长封顶。Phase 1 不引入任何检索或外部依赖（无 embedding、无新表），是三阶段里风险最低的一步，同时它强制完成一个所有后续阶段都依赖的架构改造：**context 组装从前端搬到后端**。

## 2. 策略设计

每次请求组装的 context 分三段：

```
[system prompt]
[滚动摘要]          ← 把"窗口之前"的所有历史压缩成一段
[最近 N 轮原文]      ← 保证指代（"它"、"刚才那个"）和近期连贯性
[当前用户消息]
```

- **窗口**：保留最近 `WINDOW_TURNS` 轮的原始消息（建议初值 N=6，即约 3 个 user/assistant 来回）。窗口必须是原文——指代消解、"你上一段给的代码"这类只能靠原文。
- **滚动摘要**：窗口之外的更早历史压缩成一段不超过 `SUMMARY_MAX_TOKENS`（建议 400）的摘要，放在 system prompt 之后。
- **增量更新**：每当对话推进、有旧轮被挤出窗口，就把"被挤出的那几轮"增量并入已有摘要（一次便宜 LLM 调用，输入 = 旧摘要 + 新挤出的轮次，输出 = 新摘要），而不是每次从全量历史重算。摘要结果持久化（见 §4），避免重复计费。

### 触发与边界
- 历史轮数 ≤ N 时：摘要为空，退化成"全量原文"，行为与 concat 等价（短对话不付出任何摘要成本）。
- 摘要更新时机：在 `POST /api/chats/{id}/messages` 落库新消息后异步触发，或在 `/chat` 组装时发现摘要落后于窗口边界时惰性触发。倾向后者起步（实现简单，无需后台任务），惰性触发的一次摘要延迟计入该问题的"管理开销"指标。

## 3. 架构前提：context 组装移到后端

这是 Phase 1 必须先做、且后续阶段共享的改造。

**现状**：前端持有全部 `history`，`POST /chat` 收 `{message, model, history, strategy}`，后端 `[*history, {user message}]` 直接拼。

**改造后**：
- `/chat` 请求体改为 `{chat_id, message, model, strategy}`，不再传 `history`。
- 后端按 `chat_id` 从 DB 读消息（`models.Message`，按 `position` 排序），按 `strategy` 组装。
- guest 模式（localStorage、无 DB）需要一条兼容路径：要么 guest 继续走 `concat` 老契约，要么前端把 localStorage 历史随请求传上来由后端组装。起步采用前者——**guest 仅支持 `concat`**，`window_summary` 及以后只对登录用户开放（与 `/api/*` 的 `require_user` 边界一致）。
- `/debug/run` 已经收 `history`，保留它直接收 `history` 的形式（debugger 是无状态工具，不依赖 DB），由后端对传入的 `history` 套用策略组装。这样 eval runner 不需要建库造数据，仍然把脚本化对话当 `history` 传入即可。

> 注意：`/debug/run` 与 `/chat` 的组装函数要共用同一份 `assemble_context(strategy, history, current, ...)` 实现，否则 eval 量的和线上跑的不是同一个东西，评估失去意义。

## 4. 代码与数据改动

| 位置 | 改动 |
|---|---|
| `main.py` | 新增 `assemble_context(strategy, history, current_msg, summary=None)`；`/chat` 改为按 `chat_id` 读库组装；`strategy` 白名单加 `"window_summary"` |
| 新模块（如 `context_strategies.py`） | 放窗口切分、摘要增量合并逻辑，保持 `main.py` 瘦 |
| `models.py` | `Chat` 加 `summary`（Text，可空）和 `summary_through_position`（Int，记录摘要已覆盖到哪个 position，用于增量）。无破坏性迁移——`init_db` 重建即可（dev） |
| 摘要 LLM 调用 | 复用现有上游 `/chat/completions`，固定一个便宜模型 + 固定 summarize prompt（"将以下对话压缩成不超过 400 字的要点，保留所有具体数值、决定、专有名词"） |
| `context_report.py` | `build_context_report` 增加 `summary` 层（layer 名 `summary`，带 tokens）；`history` 层改为只统计窗口内的轮次；新增字段 `dropped_turns`（被摘要吸收、未进窗口的轮数）便于 debugger 展示 |
| `static/debugger.html` | CONTEXT 区块的 `LAYER_COLORS` 已预留 `summary` 颜色；无需大改，分层条会自动多出一段 |

## 5. context_report 变化

```jsonc
{
  "strategy": "window_summary",
  "layers": [
    { "layer": "summary", "tokens": 380, "items": 1, "covers_positions": [0, 13] },
    { "layer": "history", "tokens": 1450, "items": 6, "msg_ids": [14,15,16,17,18,19] },
    { "layer": "current", "tokens": 40, "items": 1 }
  ],
  "estimated_prompt_tokens": 1870,
  "full_history_tokens": 8750,   // 不做管理时的体积
  "dropped_turns": 14,           // 被摘要吸收的轮次
  "summary_cost_tokens": 1200    // 本次摘要更新的 LLM 调用花费（管理开销）
}
```

压缩比（实际发送 / 全量历史）从此 < 1，是 Phase 1 的核心效率证据。

## 6. 评估计划

**跑哪个数据集**：**long tier 必跑**（`--tier long`）。理由写在 `eval_plan.md` §6.6：smoke tier 历史仅 ~1.5k token，窗口 N=6 基本能装下全部，压缩比≈1，测不出区分度。smoke tier 仍作为管道自检快速跑一遍（确认改造没把 `concat` 跑挂）。

```bash
# 回归：先确认 concat 没被架构改造破坏
python eval/run.py --strategy concat --tier long --model grok-4-fast
# 新策略
python eval/run.py --strategy window_summary --tier long --model grok-4-fast
# 在 debugger Eval tab 选这两个 result 并排对比
```

**新增 needle/probe**：基本不需要新造。long tier 现有用例正是为此设计的——它们的更新型针（V100→A100、跑步→康复、内部工具→对客 Q3）和早期深度针（埋在对话前段、会被摘要吸收）恰好考验摘要的两个失效点：
- **更新型针**：摘要会不会把"先 A 后 B"压成只剩旧值 A？这是滚动摘要的高危区。
- **早期深度针**（depth=early，被挤出窗口）：摘要有没有保住它的具体值（端口号、函数名、3.9 这种精确数字）？

建议补 1–2 个针对性强的探针（可选）：对 long tier 里 depth=early 的事实原子针，加一个直接回忆探针，专门量化"摘要对精确值的保真度"。

**成功判据**（相对 long tier 的 concat 基线 ctx/ans=100%）：
- 效率：平均 input token 相对 concat 下降 ≥ 40%（含摘要开销摊算后仍净省）。
- 准确度：ans 命中率 ≥ 85%；ctx 命中率单独看——被摘要吸收的针，"ctx 命中"的判定需要调整（见下）。
- 幻觉率：不高于 concat 基线 + 5 个百分点。

> **判定口径调整（已实施，2026-06-15）**：原 `ctx_hit` 是"needle 所在原始消息 position 是否进入 context"。摘要策略下，被吸收进摘要的针其原始 position 不在 context 里，但信息可能在摘要文本里。
>
> 首版实现用"针的关键词是否子串出现在摘要文本中"——**实测证明这个口径不可靠**：滚动摘要会改写事实（加反引号、用"位于"代替括号、数字写法微调），精确子串匹配对保留下来的事实大量假阴性（首跑 ctx 命中被压到 9%，连带"ctx-miss + 自信回答"的幻觉判定误报到 71%，而模型其实答对了）。
>
> 修正后的口径（`eval/run.py` `ctx_hit`，现为 async）：① verbatim 针仍按 position 判定（concat 全走这条，零 LLM 开销、结果不变）；② 被摘要吸收的针，先用子串/数字边界做"命中即确定"的快速路径，子串失败时回退 LLM 判定"这条事实在摘要里是否得到保留（措辞格式不同也算）"。LLM 只在子串失败时触发，开销可控。
>
> 同时修正幻觉判定的一个根本 bug：**答对的回答永远不是幻觉**。原逻辑在 ctx-miss 时无视回答对错就调编造裁判，给"从摘要正确回忆"误扣幻觉。加 `and not hit_ans` 守卫后，幻觉只在"ctx-miss 且回答错误"或负向探针编造时计数。

**实测基线（2026-06-15，修正口径后）**：strategy=window_summary，model=grok-4-fast，judge=kimi-k2.5，long tier（24 探针）→ **ctx 命中 91% · ans 命中 91% · 幻觉 0% · 平均 input 2394 tok（压缩比 0.38，相对 concat 省 57%）· 延迟 16.4s**。对照 concat 基线（同数据集）100%/100%/0%/5539 tok。结论：window_summary 用 ~57% 的 token 节省换约 9 个百分点的准确度、幻觉零增长，权衡相当有利；延迟升高来自每问的摘要 LLM 调用（无状态 eval 每次重算，线上持久化增量更新后远低于此）。注：window_summary 的摘要是 LLM 调用，结果有轮间方差（首次修正前 ans 命中在另一次随机摘要下为 77%），严肃对比宜多跑几次取均值或固定随机性。

## 7. 风险与回退
- 摘要是有损压缩，错了不可逆——这是策略的固有上限，靠 long tier 的更新型/精确值针量化它有多糟，用数据决定 N 和 SUMMARY_MAX_TOKENS。
- 任何时候 `concat` 仍可用，线上可按用户/会话灰度，出问题切回 `concat` 是改一个参数的事。
- 窗口 N 和摘要长度是两个旋钮，建议像 chunk size 那样跑 N∈{4,6,10} 的对比实验，用 long tier 的分数定值，而不是拍脑袋。
