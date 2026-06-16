# Phase 2 — pgvector RAG 检索（`strategy = "rag"`）

> 状态：计划，未实施。
> 关联：`design.md`（路线总览）、`eval_plan.md`（评估体系）、`phase1_sliding_window_summary.md`（前置）、`phase3_cross_session_memory.md`（下一阶段）。
> 前置依赖：Phase 1 的"context 组装移到后端"必须先完成。

## 1. 目标

解决另一个痛点："不是所有 context 都有用，暴力 concat 低效费 token"。用检索代替"全带"或"摘要压缩"：对每个新问题，从历史里**精确召回**最相关的若干片段，只把它们放进 context。这是用户最初描述的方案本体。Phase 2 仍限定在**当前 session 内**检索（跨 session 留给 Phase 3）。

相对 Phase 1 的进步：摘要是有损的、丢细节；检索能把旧的精确值（端口号、函数名、某次复查的 LDL 数字）原文取回，对"指回旧细节"类问题远强于摘要。

## 2. 策略设计

分层 context：

```
[system prompt]
[当前 session 滚动摘要]     ← 复用 Phase 1，提供全局脉络
[检索到的相关片段]          ← 本阶段新增，标注来源 position
[最近 N 轮原文]            ← 复用 Phase 1 窗口
[当前用户消息]
```

检索流程：
1. 新问题来，对问题文本算 embedding。
2. 在当前 chat 的所有历史 chunk 上做 **hybrid 检索**：pgvector 余弦相似度 + Postgres 全文检索/ILIKE 关键词分，RRF 或加权合并。
3. 乘 **recency 衰减**（越近的片段分数加成），避免久远内容挤掉近期关键上下文。
4. 取 top-k（建议 k=5），拼入 context，每片段标注其来源 position 和分数。
5. 已在窗口内的片段不重复注入（去重）。

### Chunking（关键）
长 AI 回答整条 embed 会稀释语义（long tier 的 800+ 字 needle 回复正是为此设计）。按段落把消息切成 chunk 分别 embed，chunk 命中后可选择只取 chunk 或带回整条消息。chunk 粒度建议先用"段落级 + 上限 ~300 token"，作为旋钮用 eval 调。

## 3. 代码与数据改动

| 位置 | 改动 |
|---|---|
| Postgres | 启用 `pgvector` 扩展（`CREATE EXTENSION vector`）。Docker 镜像换成带 pgvector 的（`pgvector/pgvector`）或在现有库装扩展 |
| `models.py` | 新表 `MessageChunk`：`id`、`message_id → messages`、`chat_id`（冗余，便于按会话过滤）、`position`（来源消息的 position）、`chunk_index`、`content`、`embedding`（vector 列）、`tokens`。HNSW 索引建在 `embedding` 上 |
| 写入路径 | `POST /api/chats/{id}/messages` 落库后，异步对每条新消息切 chunk、算 embedding、写 `MessageChunk`。embedding 调用走上游 `/embeddings`（需确认 AI Builder Space 是否提供；否则本地 sentence-transformers，如 bge-m3） |
| `requirements.txt` | `pgvector`（SQLAlchemy 适配）；若本地 embedding 则加 `sentence-transformers` |
| `context_strategies.py` | 新增 `rag` 组装：检索 + recency 衰减 + 去重 + 拼层 |
| `main.py` | `strategy` 白名单加 `"rag"` |
| `context_report.py` | `retrieved` 层带明细（§3 的 eval_plan 已预留结构）；`candidates` 字段填检索候选全集（含落选项及落选原因），供 debugger 展示 |
| `static/debugger.html` | `retrieved` 层颜色已预留；CONTEXT 区块展开检索候选列表（含 ✗ 落选项 + 分数 + 落选原因），对应 `eval_plan.md` §8.1 的 mockup |

### embedding 一致性纪律
query 和 document 用模型规定的指令前缀（BGE 系列要求）；模型一旦选定锁版本——换 embedding 模型 = 全量重建 `MessageChunk`，把这个成本写进决策记录。

## 4. context_report 变化

```jsonc
{
  "strategy": "rag",
  "layers": [
    { "layer": "summary",  "tokens": 300, "items": 1 },
    { "layer": "retrieved","tokens": 1240, "items": [
        { "msg_id": "...", "position": 12, "chunk_index": 1, "score": 0.83,
          "tokens": 310, "preview": "...384 token、重叠 64..." }
    ]},
    { "layer": "history",  "tokens": 900, "items": 4, "msg_ids": [...] },
    { "layer": "current",  "tokens": 40, "items": 1 }
  ],
  "candidates": [   // 检索候选全集，含落选——debugger 用它回答"模型为什么不知道 X"
    { "position": 12, "score": 0.83, "selected": true },
    { "position": 3,  "score": 0.41, "selected": false, "reason": "below_threshold" }
  ],
  "estimated_prompt_tokens": 2480,
  "full_history_tokens": 8750,
  "retrieval_cost_tokens": 30   // query embedding 开销
}
```

落选候选必须展示——"针没进 context"有两种病因（检索没召回 vs 召回了被预算挤掉），修法不同。

## 5. 评估计划

**跑哪个数据集**：**long tier 必跑**。long tier 的设计点正对 RAG 的软肋：
- **改写问法探针**（与针零关键词重叠，如 ml_001 的"改特征逻辑从哪个函数入手"对 needle `build_feature_matrix_v3`）：专测纯向量检索能不能跨词面召回——这是 RAG 相对 BM25 的价值证明区。
- **assistant 长回复中段的针**（chunking 测试）：针埋在 800+ 字回复中段，考验切 chunk 后该片段还能不能被召回，不被整条的"平均语义"淹没。
- **近似干扰针**（256/512 vs 384、尿酸 460 vs 455）：测检索精度而非只测召回——会不会把相似但错误的片段排在前面。

```bash
python eval/run.py --strategy concat         --tier long --model grok-4-fast  # 基线
python eval/run.py --strategy window_summary --tier long --model grok-4-fast  # P1 对照
python eval/run.py --strategy rag            --tier long --model grok-4-fast  # P2
# debugger Eval tab 三个 result 并排；维度下钻看 carrier=assistant、style=paraphrase 两行
```

**新增 needle/probe**：现有 long tier 够用，重点是用**维度下钻表**读结果——若 `carrier=assistant`（chunking）或 `style=paraphrase`（语义检索）这两行命中率明显低于 concat，直接指向 chunking 或 embedding 选型问题。建议补：几个纯关键词重叠低的改写探针，加密 paraphrase 维度的样本量（现 long tier 只有 3 个 paraphrase）。

**成功判据**（相对 long tier concat 基线）：
- 效率：平均 input token 比 concat 降 ≥ 50%，且**比 Phase 1 的 window_summary 不显著更高**（检索的开销别把摘要省下的吃回去）。
- 准确度：ans 命中率 ≥ Phase 1 的水平，且在 paraphrase 维度上 **高于** Phase 1（检索的精确召回应当在"指回旧精确值"类问题上胜过有损摘要——这是 Phase 2 存在的理由，要用数据证明）。
- ctx 命中率：此处恢复确定性判定（检索把原始 position 的片段取回，`ctx_hit` 按原 position 判定即可，不需要 Phase 1 那种摘要文本匹配）。

## 6. 风险
- embedding endpoint 可用性是外部依赖——先确认 AI Builder Space 有没有 `/embeddings`，没有就锁定本地模型，别让这个卡住排期。
- 纯向量对专有名词/代码标识符弱——hybrid（叠加关键词分）不是可选项是必需项，long tier 的 `code_identifier` 针会直接惩罚只做向量的实现。
- recency 衰减的系数是旋钮，用 long tier 的"早期针 vs 晚期针"命中率差来标定。
