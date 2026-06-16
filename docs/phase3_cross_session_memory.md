# Phase 3 — 跨 session 检索 + memory 注入（`strategy = "rag_xsession"`）

> 状态：计划，未实施。
> 关联：`design.md`（路线总览）、`eval_plan.md`（评估体系）、`phase2_pgvector_rag.md`（前置）。
> 前置依赖：Phase 2 的 pgvector chunk 表与检索链路。

## 1. 目标

实现用户的第二个需求：context 不只来自当前 session，还能从**其他 session** 检索相关内容来构建当前问题的 context。再叠加一层结构化记忆：把对话中反复出现的稳定事实（用户背景、偏好）抽取成持久 memory，每次对话注入。

两件互补的事：
- **跨 session 检索**：把 Phase 2 的检索范围从 `chat_id = X` 放宽到 `user_id = X`。
- **memory 抽取**：稳定的用户事实用 memory 条目表达，比检索旧消息原文更干净（旧消息有大量噪音）。

## 2. 策略设计

```
[system prompt]
[跨 session memory 条目]     ← 本阶段新增，稳定事实
[当前 session 滚动摘要]       ← Phase 1
[检索片段（跨所有 session）]   ← Phase 2 检索，范围放宽到 user_id，标注来源 session
[最近 N 轮原文]              ← Phase 1
[当前用户消息]
```

### 跨 session 检索的两个新问题
1. **多题材互扰**：用户的 ML、股票、养花对话同库，检索必须能在噪音里挑对题材的片段——这正是 Phase 3 区分度的来源（单 session 时同一用例只有一个题材，没有跨题材干扰）。
2. **来源标注与隐私**：检索片段要标注来自哪个 session/何时，注入时让模型知道"这是你在另一次对话里说的"。recency 衰减要跨 session 生效。

### memory 抽取
后台异步从对话里抽取持久事实（"用户在做 FastAPI 项目"、"用户对花生严重过敏"、"用户阳台朝西"），写成独立 memory 条目（每条一个事实 + 来源），每次对话按相关性注入。抽取用便宜模型定时跑，不在请求路径上。

## 3. 代码与数据改动

| 位置 | 改动 |
|---|---|
| `models.py` | `MessageChunk` 检索查询去掉 `chat_id` 过滤、改按 `user_id`（需确保 chunk 表能 join 到 user）。新表 `Memory`：`id`、`user_id`、`content`、`source_chat_id`、`embedding`、`created_at`、`last_used` |
| 检索逻辑 | Phase 2 的检索函数加一个 `scope` 参数（`chat` / `user`）；`rag_xsession` 用 `user` |
| memory 抽取 | 新后台任务（或惰性触发）：对话结束/达到一定长度后，便宜模型抽取候选事实，去重后写 `Memory`。注入时对 memory 做相关性筛选（embedding 相似度），不是全量灌 |
| `main.py` | `strategy` 白名单加 `"rag_xsession"` |
| `context_report.py` | 新增 `memory` 层（颜色已预留）；`retrieved` 层的明细项加 `source_chat` 字段标注来源 session |
| `static/debugger.html` | `memory` 层颜色已预留；检索候选项展示 source session |

## 4. context_report 变化

`retrieved` 层的明细项带上 `source_chat`（`eval_plan.md` §3 的数据结构已预留该字段），新增 `memory` 层。debugger 的检索候选列表要能显示"这条来自哪个会话"，否则跨 session 检索无法调试。

## 5. 评估计划

**这是唯一需要新数据集的阶段。** 现有 smoke/long tier 都是单 session 用例，无法测跨 session 检索。

**新增 multi-session tier**（`eval/cases/multi/`，`dataset_version = "v1-multi"`）：
- 每个用例是**同一用户的多个 session**（如 3–5 段不同题材的对话）+ 一组 probe。
- 关键：probe 问的事实埋在**另一个 session** 里（如在"养花 session"埋"阳台朝西"，在一个新的当前问题里问"我阳台适合种月季吗"，正确回答需要跨 session 取回那个事实）。
- 多题材同库**互为干扰项**——养花针埋在股票/ML 对话之间，检索随便就中说明数据集太easy；这是单 session 数据集给不了的压力。
- 复用五题材，但一个用例横跨多题材（与单 session "每用例单一题材"的纪律相反，正是为了测互扰）。
- 规模建议首版 4–6 个 multi-session 用例。

`eval/run.py` 需要扩展以支持"多 session 历史"的输入形态（目前 case 的 `conversation` 是单条线性历史；multi tier 需要 `sessions: [[...], [...]]` 结构，runner 把"非当前 session"灌进可检索库、把"当前 session"当 history）。

```bash
# 跨 session 必须在 multi tier 上测
python eval/run.py --strategy rag_xsession --tier multi --model grok-4-fast
# 对照：concat 在 multi tier 上会"作弊"地把所有 session 拼进去——它就是跨 session 的准确度天花板（但 token 爆炸）
python eval/run.py --strategy concat        --tier multi --model grok-4-fast
# rag（单 session 检索）在 multi tier 上应当大量 miss——它够不到别的 session，作为下界对照
python eval/run.py --strategy rag            --tier multi --model grok-4-fast
```

**memory 单独评估**：抽取质量另测——给定一段对话，抽出的事实集和人工标注的"应抽取事实"对比（precision/recall）。这是离线评估，不进 needle pipeline。

**成功判据**：
- 跨 session 召回：在 multi tier 上，`rag_xsession` 的 ans 命中率应接近 concat（跨 session 天花板），且**远高于** `rag`（单 session 下界，它够不到别的 session）。这个"高于单 session rag、接近 concat"的夹逼是 Phase 3 成立的证明。
- 效率：input token 远低于在 multi tier 上作弊全拼的 concat。
- 幻觉：跨 session 引入更多噪音，负向探针和"context miss 却给答案"的幻觉率要重点盯，不能因为检索范围变大就更爱编。

## 6. 风险
- 跨 session 检索的噪音放大——多题材同库，错召回别题材的片段会污染回答；hybrid + 题材/recency 加权更关键。
- memory 抽取错误会被每次注入放大（一个错事实污染所有后续对话），抽取宁缺勿滥，且要可被用户/系统修正。
- multi tier 数据集和 runner 改造是这一阶段的主要工程量，排期要把"建数据集"当成和"写策略"同等的任务，别低估。
