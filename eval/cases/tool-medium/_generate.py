"""Generator for the tool-medium eval tier (docs/eval_plan.md §6.7/§6.8).

Each case: a medium single-topic conversation (authored here) + probes whose
`tool_rounds` are FROZEN. The tool results are captured by calling the ACTUAL
FastAPI tools (`main.read_page` / `main.web_search`) — NOT reimplemented or
synthesized — so the frozen results have exactly production's format and size
distribution (e.g. a real web_search returns ~22k chars of the supermind shape,
not a hand-made stub). Only the needle sentence + near-miss decoy are hand-inserted
into the captured read_page text (per eval_plan "针手工插入"), then capped at the
production PAGE_TEXT_LIMIT (imported from main, currently 30k — whole page in most
cases). web_search results are frozen verbatim (no needle). Output JSON is written
to disk so the big tool-result strings never round-trip through the orchestrator.

Run once (needs SUPER_MIND_API_KEY in .env for web_search); the JSON is the frozen
artifact. Provenance kept in `_source_url`/`_note`.
"""
import asyncio
import json
import os
import sys

REPO = "/Users/ttwong/Documents/Projects/AI_Architect_Project_1"
sys.path.insert(0, REPO)
# The REAL FastAPI tools — capturing via these guarantees production format/size.
from main import PAGE_TEXT_LIMIT, read_page as app_read_page, web_search as app_web_search

OUT = os.path.join(REPO, "eval/cases/tool-medium")


def _run(coro):
    return asyncio.run(coro)


def tool_result(url: str, needle_block: str, at_frac: float = 0.45) -> str:
    """REAL read_page output with the needle block spliced in mid-result.

    Calls the actual `main.read_page` (already capped at PAGE_TEXT_LIMIT), then
    inserts `needle_block` (planted needle + near-miss decoy) at ~at_frac so it
    sits mid-result, and re-caps at PAGE_TEXT_LIMIT. The bulk is production read_page
    output; only the needle sentence is added.
    """
    raw = _run(app_read_page(url))
    insert_at = min(int(len(raw) * at_frac), PAGE_TEXT_LIMIT - 900)
    insert_at = max(insert_at, 800)  # keep it out of the very start
    nl = raw.rfind("\n", 0, insert_at)
    if nl != -1:
        insert_at = nl + 1
    spliced = raw[:insert_at] + needle_block + "\n" + raw[insert_at:]
    return spliced[:PAGE_TEXT_LIMIT]


def read_page_round(call_id, url, needle_block, at_frac=0.45):
    return {
        "assistant": {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {
                "name": "read_page", "arguments": json.dumps({"url": url}, ensure_ascii=False)}}]},
        "tool": {"role": "tool", "tool_call_id": call_id,
                 "content": tool_result(url, needle_block, at_frac)},
    }


def web_search_round(call_id, query):
    """web_search round — REAL `main.web_search` output, frozen verbatim (no needle).

    The app's /debug/run serializes the web_search return with
    json.dumps(result, ensure_ascii=False), so we freeze exactly that.
    """
    result = _run(app_web_search(query))
    return {
        "assistant": {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {
                "name": "web_search", "arguments": json.dumps({"query": query}, ensure_ascii=False)}}]},
        "tool": {"role": "tool", "tool_call_id": call_id,
                 "content": json.dumps(result, ensure_ascii=False)},
    }


def conv(*pairs):
    out = []
    for role, content in pairs:
        out.append({"role": role, "content": content})
    return out


# ---------------------------------------------------------------------------
# CASE 1 — ML: tool needle = version identifier (direct) + case-level update needle
# ---------------------------------------------------------------------------
ml1 = conv(
    ("user", "我在做一个 LLM 推理服务，想用 vLLM 部署，规模不大但要扛真实并发。先帮我把整体架构和关键决策点完整过一遍，我好心里有数，再逐个抠参数。"),
    ("assistant", "好。先讲清楚 vLLM 为什么值得用：它的核心是 PagedAttention——把每个序列的 KV cache 切成固定大小的 block，像操作系统的分页那样按需分配，避免了传统把 KV cache 当一整块连续显存预留所带来的碎片和浪费。配合 continuous batching（每一步动态把新请求拼进正在跑的 batch，而不是等一个 batch 全部结束），同样的卡能把吞吐拉高好几倍。整体架构我建议三层：最前面 API gateway 负责鉴权、限流、把 OpenAI 格式的请求转发进来；中间是 vLLM 自带的 OpenAI-compatible server，承担调度和实际推理；后面是显卡资源，初期单机，之后再考虑多卡张量并行。我们按‘单卡跑通 → 压测找瓶颈 → 再谈并行’的顺序来，别一上来就上分布式，调试成本太高。"),
    ("user", "同意，先单卡。模型我用 7B 验证，显卡是 A100 40G。"),
    ("assistant", "A100 40G 跑 7B 很宽裕：FP16 权重约 14G，剩下二十多 G 全是你的 KV cache 和激活值预算，能容纳相当多的并发序列。几个起步参数：gpu_memory_utilization 先设 0.9，给框架留 10% 余量防 OOM；max_model_len 一定要按真实请求长度设，别图省事开满 32k，那会凭空预留巨量 KV cache 把并发压死；dtype 用默认的 FP16 即可，A100 上 FP8 收益有限。先把这套跑起来看基线吞吐。"),
    ("user", "请求大多在 4k 上下文以内，偶尔到 8k。那我 max_model_len 先开 8192 行不行？"),
    ("assistant", "可以，8192 对 7B 在 40G 上压力不大。提醒一点：max_model_len 是单条序列的上限，KV cache 总预算是被所有并发序列分摊的，所以 8192 配合较高并发时，能同时在跑的序列数会被压低——这正是后面压测要观察的点。另外 batch 你不用手动设，vLLM 是动态拼 batch 的，这和 TGI 要手调 max_batch_total_tokens 的思路完全不同，你只管设显存利用率和长度上限，调度交给它。"),
    ("user", "那并发上来之后，我怎么判断到底是显存瓶颈还是算力瓶颈？这俩处理方式完全不一样。"),
    ("assistant", "对，区分清楚才不会乱加卡。看两组指标：第一，KV cache 的 block 占用率（vLLM 会暴露 gpu_cache_usage 这个 metric），如果它逼近 1.0、同时有请求在 waiting 队列里排队，那就是显存瓶颈——KV cache 装不下更多并发序列了，解法是降 max_model_len、上量化、或者加卡。第二，如果 cache 占用并不高，但 GPU 利用率打满、TTFT（首 token 延迟）和 TPOT（每 token 延迟）一起往上涨，那是算力瓶颈，纯靠张量并行把单次前向拆到多卡才有用。两者搞反了会白烧钱：算力瓶颈你去减长度没用，显存瓶颈你去加 TP 也救不了。"),
    ("user", "理解了。我们后面大概率要上多卡。先记一个决策：初期决定用 tensor parallel size = 2 起步。"),
    ("assistant", "记下了，TP=2 起步是稳妥的第一步。两卡之间走 NVLink 带宽足够，AllReduce 通信开销占比小，不会成为瓶颈；而且 7B 切两半后单卡权重和激活都更宽松，能进一步抬高并发。等 QPS 再翻倍、单机两卡也吃紧时，我们再谈 TP=4，或者跨机时叠加 pipeline parallel——但那会引入 micro-batch 调度的复杂度，非必要不提前上。"),
    ("user", "等等，采购那边刚反馈 A100 缺货，给换成 H100 80G 了。前面的架构判断要不要改？"),
    ("assistant", "好消息，基本是放宽约束。换 H100 80G 后：单卡显存直接翻倍到 80G，KV cache 预算大涨，你单卡就能扛明显更高的并发；H100 的 FP8 和 Transformer Engine 支持更好，吞吐还能再上一截；算力也强不少。结论是张量并行的引入时间点可以往后大幅推迟，单卡先把量做起来。请把前面所有‘A100 40G’的前提都更新成 H100 80G，包括显存预算和并发预期。"),
    ("user", "压测我打算分三档并发去打，10、50、100，每档跑五分钟看稳态。需要我重点记录哪些指标？"),
    ("assistant", "压测要测得有用，指标得分客户端和服务端两侧记。客户端侧：QPS（实际完成的请求数）、TTFT（首 token 延迟，决定体感‘快不快’）、TPOT 或 inter-token latency（决定流式输出顺不顺）、端到端 P50/P95/P99 延迟，尾延迟比均值重要得多。服务端侧从 vLLM 的 metrics 拉：gpu_cache_usage（KV cache 占用率）、num_requests_running 和 num_requests_waiting（跑着的和排队的）、以及 GPU 利用率和显存占用。把这两侧对齐着看：比如 P99 突然抬升时，去看是 waiting 队列变长（容量不够）还是 running 数没满但 GPU 打满（算力到顶），这就回到我们前面说的两类瓶颈判定。还有别忘了记录每档的吞吐-延迟曲线，找到延迟开始陡升的那个并发拐点，那才是你这套配置的实际服务容量。"),
    ("user", "如果 P99 在高并发时被个别长请求拖崩，有什么缓解手段？"),
    ("assistant", "这是典型的‘长尾请求污染’。几个手段：第一，给 max_tokens 设上限并对超长输出做业务侧约束，别让单个请求无限生成；第二，vLLM 的调度本身是 token 级的，长请求不会独占整轮，但你可以通过限制单请求 max_model_len 或拆分超长上下文来削峰；第三，如果业务允许，把长任务和短任务分流到不同实例/队列，避免长请求把交互式短请求的 TTFT 拖坏；第四，开启 priority 调度（新版本支持），让交互式请求优先。核心思路是隔离，别让少数重请求决定多数轻请求的体验。这些等压测看到尾延迟形态后再针对性上，别提前过度设计。"),
    ("user", "上线后大概率要灰度发布和回滚，推理服务这种有状态的（KV cache、加载几十 G 权重）怎么做不中断切换？"),
    ("assistant", "推理服务虽然单请求无状态，但实例本身‘重’——加载权重要几十秒到分钟级，所以不能像无状态微服务那样秒级滚动。实践上分几步：第一，多副本+负载均衡是前提，至少两个实例，更新时一个个来，留住容量；第二，用‘启动就绪探针’——新实例必须等权重加载完、能正确响应一个探活请求后，才把它挂进负载均衡，否则流量打到没加载完的实例上全是错误；第三，蓝绿或金丝雀：先起一组新版本实例，引一小撮流量（按比例或按 header）验证质量和延迟没退化，再逐步放大、最后撤掉老版本，出问题就把流量切回老组、秒级回滚；第四，优雅下线——给要退役的实例发停止信号后，先停止接新请求、把在途请求处理完再关，别硬杀导致请求中断。模型版本和服务版本要分开记，必要时能独立回滚（只回退模型权重而不动代码）。这套配合前面的健康探针和监控，才能做到用户无感的更新。"),
    ("user", "成本这块老板肯定要问。单卡 H100 这套，怎么估算每千 token 的推理成本，好跟用大厂 API 比？"),
    ("assistant", "估算思路是‘把卡的时间成本摊到产出的 token 上’。先拿到两个数：一是这张 H100 实例的单位时间成本（自建按折旧+电力+运维摊，云上直接是每小时租金）；二是压测得到的稳态总吞吐，即每秒能产出多少 output token（注意要算上 continuous batching 后的聚合吞吐，不是单请求速度）。每千 token 成本 ≈ 实例每秒成本 ÷ 每秒 output token 数 × 1000。关键提醒：第一，吞吐必须用你实际并发拐点附近的稳态值，不能用空载峰值，否则严重低估成本；第二，input 和 output token 单价不同（prefill 和 decode 算力特性不一样），精细一点要分开摊；第三，别忘了把利用率算进去——卡不是 7×24 满载，按真实平均利用率折算才是真实单价。拿这个自建单价去和大厂 API 的每千 token 报价比，再叠加前面说的人力运维，才是公平的 build vs buy 账。"),
    ("user", "明白，那我先按单卡 H100 跑压测，TP 留着以后再说。另外有件事我想先锁死——我们到底用 vLLM 哪个版本，免得我照着文档写的接口和实际装的对不上。"),
    ("assistant", "非常对，版本必须锁死。PagedAttention 的 Python 接口和 LLM/engine 的入参在小版本之间改过几次，跨版本抄代码很容易踩坑。你去官方页面确认一下当前推荐的稳定版本号，我们后面所有示例代码和 API 调用都以那个版本为准，requirements 里也钉死，别用浮动版本。"),
    ("user", "行，我现在就去查一下确认版本。"),
)
ml1_needle_block = (
    "Production note (internal): the deployment baseline is pinned to vLLM 0.6.3, "
    "which stabilized the PagedAttention block-manager API. Do not target the older "
    "0.5.4 line — its KV-cache interface differs and is considered deprecated here."
)

case_ml1 = {
    "id": "tool_ml_001",
    "topic": "ml",
    "conversation": ml1,
    "needles": [],
    "probes": [
        {"id": "p1", "question": "根据我刚打开的这个页面里的说明，我们内部要锁定的 vLLM 稳定版本号写的是多少？请以页面内容为准，不要凭印象答。", "style": "direct",
         "_source_url": "https://en.wikipedia.org/wiki/Large_language_model",
         "_note": "Round 0 = REAL web_search output (frozen verbatim, no needle); round 1 = REAL read_page output with the version needle+decoy hand-inserted mid-result.",
         "tool_rounds": [
             web_search_round("call_ml1s", "vLLM latest stable release version pinned"),
             read_page_round("call_ml1", "https://en.wikipedia.org/wiki/Large_language_model", ml1_needle_block),
         ],
         "needles": [{"id": "t1", "content": "0.6.3", "tool_round": 1, "carrier": "tool", "type": "identifier"}],
         "needle_ids": ["t1"],
         "judge": {"method": "regex", "pattern": r"0\.6\.3"}},
    ],
}

# ---------------------------------------------------------------------------
# CASE 2 — AI trends: multi-round (web_search -> read_page), tool needle factual (paraphrase)
# ---------------------------------------------------------------------------
ai1 = conv(
    ("user", "我在给公司董事会写一份 AI 趋势简报，主题是生成式 AI 在企业里的真实采用情况，要能经得起追问。先帮我把整体框架和叙事搭出来。"),
    ("assistant", "框架我建议分三层，层层递进、也方便答辩。第一层是采用率的真实数字——一定要剥掉营销口径，区分‘试用过’和‘进了生产流程’，因为这俩差着量级；第二层是真正落地最快的场景，告诉董事会钱该往哪投；第三层是没落地的卡点，提前把‘为什么不能更快’讲清楚，免得被质疑。整份简报最忌讳的就是拿‘做了个 PoC’当‘已经采用’，董事会一问‘那转化成生产的有几个、ROI 多少’就露馅了。所以数字这块我们要格外谨慎，宁可保守也别注水。"),
    ("user", "对，我最烦那种把 PoC 当成果汇报的。先说采用率，你印象里现在大盘是什么量级？"),
    ("assistant", "各家口径差很多，你引用时务必标清来源和‘采用’的定义。麦肯锡的年度调研、斯坦福 HAI 的 AI Index、各大咨询机构的数字，能从三成一路到七成不等，差异几乎全来自定义口径：有的把‘公司里有人用过’算采用，有的要求‘至少一个业务职能进了常规流程’才算。所以单看一个百分比没意义，要带定义。简报里我建议挑一个口径清晰、样本量大的权威调研做主锚，其它做佐证。"),
    ("user", "明白，口径比数字本身还重要。那落地最快的场景呢？"),
    ("assistant", "目前最稳的是三类，共同特点是容错高、有人兜底：一是客服与工单/会话摘要，错了人能复核，省的是人力工时；二是营销与文案生成，本来就要人改稿，AI 出初稿提效明显；三是代码助手，补全和样板代码生成，工程师本来就会 review。相对地，医疗诊断、法律意见这类高风险领域还卡在合规、责任归属和幻觉上，落地慢是合理的，不该硬推。"),
    ("user", "卡点具体是哪些？我想在简报里诚实地列出来。"),
    ("assistant", "三个核心卡点：第一，数据治理跟不上——RAG 想做好得有干净、结构化、权限清晰的语料，多数公司的内部文档是一团乱麻，喂进去就是垃圾进垃圾出；第二，幻觉带来的信任成本，一旦出过一次离谱错误，业务方就不敢放权，使用率掉下去；第三，也是最致命的，ROI 算不清，很多项目死在‘说不清到底省了多少钱、提了多少效’，预算第二年就被砍。简报里把这三点讲透，反而显得你判断扎实。"),
    ("user", "ROI 那条我想在简报里给个可操作的衡量办法，不能只喊‘算不清’。你建议怎么量？"),
    ("assistant", "给董事会一个能落地的 ROI 框架，分三类收益+一类成本。收益侧：一是‘工时替代’，最硬，比如工单摘要把每单处理时间从 8 分钟降到 5 分钟，乘以单量和人力成本就是省下的钱，可审计；二是‘产能放大’，同样的人做更多事（文案产出翻倍），算增量产出价值；三是‘质量/风险’收益，比如错误率下降带来的返工或赔付减少，较软但能估。成本侧别只算 API token 费，要含工程投入、数据治理、人工复核（human-in-the-loop 的兜底成本往往被忽略）和模型迭代维护。关键纪律是每个场景上线前先定好‘衡量指标和对照基线’，否则上线后根本算不出增量。简报里我建议挑一个已落地场景把这套账算给董事会看，比讲十个趋势都有说服力。"),
    ("user", "数据治理那块，很多人觉得是 IT 的事，怎么跟董事会讲清楚它为什么是 AI 落地的前置条件？"),
    ("assistant", "用一句话点破：生成式 AI 的上限不是模型，是你喂给它的上下文质量，而上下文来自你的内部数据。RAG 检索得准，前提是文档结构化、有清晰的权限边界、有可信的事实来源；如果内部知识散在过期的 wiki、口径冲突的表格、谁都能改的文档里，模型只会把垃圾拼接得更像模像样，反而更危险。给董事会的类比是：AI 是放大器，数据干净它放大价值，数据脏它放大错误和合规风险。所以预算不能只投模型和应用层，得有一笔投在数据治理（清洗、结构化、权限、留痕）上，这是地基。把它讲成‘没有这步，后面所有 AI 投入的 ROI 都打折’，董事会就听得进去了。"),
    ("user", "幻觉这块董事会肯定会追问‘那怎么控’，我得有应对。技术上有哪些现实可行的降幻觉手段？"),
    ("assistant", "给董事会讲降幻觉，别承诺‘消除’，要讲‘分层控制到可接受’。技术层面四道防线：第一，RAG 把回答锚定到可信来源，让模型基于检索到的真实片段作答而不是凭记忆编，这是性价比最高的一招，前提还是前面说的干净数据；第二，要求模型给出引用/出处，答案可溯源，业务方能一眼看出哪句没依据；第三，结构化输出加校验，对数字、日期、ID 这类硬事实做规则或二次核对，不让它蒙混；第四，对高风险场景强制 human-in-the-loop，模型出草稿、人把关，把幻觉的后果挡在发布前。再叠加流程层面的‘置信度阈值+兜底话术’，低置信时让它说‘我不确定’而不是硬编。把这套讲成‘纵深防御’，董事会就明白幻觉是被管理的工程问题，不是失控的黑箱——这恰恰也是我们这份简报想传递的成熟度。"),
    ("user", "董事会里有人会问‘那我们到底该自建还是直接用大厂 API’，这种 build vs buy 我该怎么帮他们判断？"),
    ("assistant", "给一个判断框架，别陷入‘自建显得有技术含量’的虚荣陷阱。默认从 buy（直接用大厂 API）起步，因为它最快验证价值、不烧人力，绝大多数场景的瓶颈是产品和数据治理，不是模型本身。往 build（自建/自托管开源模型）倾斜的信号有几个：一是数据敏感或合规要求数据不能出域；二是调用量大到 API 账单显著超过自建的总拥有成本（注意算上人力和运维，不只是 GPU）；三是需要深度定制或微调、对延迟有硬性要求、或要把模型嵌进离线/边缘环境。折中是混合：通用场景用 API，敏感或高频的特定场景自托管。关键纪律是用 ROI 和约束条件说话，而不是‘想不想自己搞一套’。把这套判断讲给董事会，比直接抛一个结论更让他们放心——他们看到的是你有框架，不是拍脑袋。"),
    ("user", "再帮我想一句给董事会的‘一句话结论’，要既不浇冷水也不画大饼，定个调子。"),
    ("assistant", "可以这样定调：‘生成式 AI 对我们是确定的方向、但不是免费的捷径——价值真实且已在少数高容错场景验证，但兑现它取决于我们在数据治理、ROI 度量和风险控制上的投入，而不是模型本身。’这句话的好处是：承认机会（不浇冷水），但把成功条件落在‘我们能控的内功’上（不画大饼），顺势引出后面要钱要资源的三块投入。董事会最怕两种汇报——一种是‘AI 颠覆一切、赶紧梭哈’，一种是‘都是泡沫、再等等’，你这句居中、有条件、可执行，既显专业又为预算铺路。后面每一页都可以挂回这句结论，整份简报就有了主心骨。"),
    ("user", "这些我都认同，尤其 ROI 那条。简报里我想放一个权威的总体采用率数字镇场子，你帮我查一份比较新的报告，把里面那个企业整体采用比例摘出来。"),
    ("assistant", "好，我先搜一下最近的权威调研，锁定一份口径清晰的，再点进去把那个总体采用比例的具体数字读出来给你。"),
    ("user", "可以，搜吧，记得要新一点的。"),
)
ai1_needle_block = (
    "Survey highlight: across the firms polled, 71% reported using generative AI in at "
    "least one business function in the latest wave — up sharply year over year. (An "
    "earlier baseline wave had put the comparable figure near 65% of organizations.)"
)
case_ai1 = {
    "id": "tool_ai_001",
    "topic": "ai_trends",
    "conversation": ai1,
    "needles": [],
    "probes": [
        {"id": "p1", "question": "根据我刚读取的那个页面里的调研数据，企业整体用上生成式 AI 的比例写的是多少？以页面里的数字为准。", "style": "paraphrase",
         "_source_url": "https://en.wikipedia.org/wiki/Generative_artificial_intelligence",
         "_note": "Single REAL read_page output with the adoption-rate needle+decoy hand-inserted mid-result. Kept single-round on purpose: a real same-topic web_search floods the context with competing public adoption %s (including 71%), making a generic-statistic needle non-distinctive. web_search-before-needle is reserved for specific/internal-fact needles (see tool_ml_001).",
         "tool_rounds": [
             read_page_round("call_ai_r", "https://en.wikipedia.org/wiki/Generative_artificial_intelligence", ai1_needle_block),
         ],
         "needles": [{"id": "t1", "content": "71% reported using generative AI", "tool_round": 0, "carrier": "tool", "type": "factual"}],
         "needle_ids": ["t1"],
         "judge": {"method": "regex", "pattern": r"71\s*%|71\s*％"}},
    ],
}

# ---------------------------------------------------------------------------
# CASE 3 — Stocks: two tool needles, multi-hop arithmetic (implicit)
# ---------------------------------------------------------------------------
st1 = conv(
    ("user", "帮我系统分析一下英伟达这家公司的投资价值，我主要关心数据中心业务的盈利能力，最后我想自己用财报数据算一算。"),
    ("assistant", "好。英伟达现在的核心叙事几乎可以浓缩成一句话：数据中心 GPU。逻辑链条是——大模型训练和推理需求爆发，H100/H200 这类高端加速卡供不应求，于是数据中心分部的营收占比从过去的‘一半左右’迅速飙到八成以上，成了绝对大头；同时因为卖的是高附加值芯片加软件生态（CUDA、NVLink），整体毛利率被拉到历史高位。投资价值的多空分歧基本都围绕这条链：多头赌需求持续、护城河（CUDA 生态）够深；空头担心客户集中（几家云厂商占大头）、竞品追赶（AMD MI300、各家自研 ASIC）以及周期性。我们先把盈利能力的数字坐实，再谈估值。"),
    ("user", "毛利率被拉到历史高位，大概是多高的水平？"),
    ("assistant", "在数据中心高端芯片的拉动下，公司整体毛利率一度冲到 70% 多的区间，明显高于过去以游戏显卡为主、毛利率 60% 出头的时代。这个跃升不是靠降本，而是产品结构变了——卖的东西单价和附加值都上了一个台阶。但要提醒：这么高的毛利率本身就是供需极度紧张的产物，能不能维持，取决于这种紧张还能持续多久，一旦供给跟上或需求降温，毛利率回归是大概率事件，所以做模型时不能简单线性外推。"),
    ("user", "那按现在的盈利，估值算贵吗？"),
    ("assistant", "取决于你用什么锚。看静态 PE 确实不低，市场已经把高增长预期打进去了；但如果数据中心营收还能维持高速增长，forward PE 会被未来盈利摊薄，看起来就没那么离谱。关键变量始终是数据中心这块营收能不能持续放量——它既是营收大头，又是毛利率的支撑，一荣俱荣、一损俱损。所以与其纠结当前 PE，不如把精力放在跟踪数据中心营收的环比趋势和大客户的采购指引上。"),
    ("user", "我最担心的就是周期性。万一 AI 资本开支退潮呢？"),
    ("assistant", "这是最大的尾部风险，而且是‘双杀’式的。一旦云厂商和大企业开始收缩 AI 相关的 capex，数据中心营收会掉，产能利用率下来后毛利率也会跟着回落，营收和利润率同时下滑、估值再被下修，就是典型的戴维斯双杀。所以对英伟达来说，盯紧几家大客户（微软、谷歌、Meta、亚马逊）的资本开支指引，往往比盯英伟达自己上个季度的财报还重要——客户的开支计划是领先指标，财报是滞后结果。"),
    ("user", "竞争格局你怎么看？AMD 的 MI300 和云厂商自研 ASIC 到底能不能撼动它？"),
    ("assistant", "分两个层面看。硬件层面，AMD MI300、谷歌 TPU、亚马逊 Trainium/Inferentia 这些在纸面算力和显存上确实在逼近甚至局部超越，大客户也乐意扶持二供来压价、降低单一依赖。但真正的护城河在软件层面——CUDA 生态加上 cuDNN、NCCL、以及一整套被全世界开发者和框架默认适配的工具链，迁移成本极高。自研 ASIC 在自家固定负载上跑得动、性价比好，但通用性和生态差得远，外部客户不会为了省点钱去吃整条工具链不成熟的苦。所以我的判断是：竞品会蚕食‘价格敏感、负载固定’的那部分份额，压制英伟达的定价权和毛利率上限，但短期内难以撼动它在‘通用、前沿、生态依赖’这块的统治地位。对投资来说，要盯的不是‘会不会被取代’，而是‘毛利率会不会因为二供出现而见顶回落’。"),
    ("user", "那如果只能看一个领先指标来预判它的拐点，你会看哪个？"),
    ("assistant", "我会看大客户的资本开支指引，而且是‘指引’不是‘已实现’。原因前面说过：英伟达数据中心营收的需求方高度集中在几家超大云厂商和头部 AI 公司，它们的 capex 计划是英伟达订单的领先指标，通常领先一到两个季度。当这几家在财报电话会里开始把 AI 相关 capex 的措辞从‘加速投入’转成‘优化效率/趋于理性’，往往就是需求降温的早期信号，会先于英伟达自己的营收下滑体现出来。第二个辅助指标是交付周期/缺货程度，从‘一卡难求’变成‘随时能买到’，意味着供需拐点到了，毛利率随后承压。把这两个盯住，比逐季抠英伟达自己的财报更能提前判断。"),
    ("user", "客户集中这个风险我老板很在意，能具体讲讲它怎么传导成投资风险吗？"),
    ("assistant", "客户集中的风险本质是‘议价权和需求稳定性都押在少数人身上’。具体有三条传导路径：第一，议价权——当你大部分数据中心营收来自四五家超大云厂商，它们联合压价或扶持二供（AMD、自研芯片）的能力就很强，会直接侵蚀你引以为傲的高毛利率，这是对盈利质量的慢性侵蚀；第二，需求同步性——这几家的 AI capex 周期高度相关，一旦宏观或行业逻辑变化，它们很可能同时踩刹车，需求不是平滑下滑而是断崖，营收波动被放大；第三，库存与预付——景气时客户为锁产能会下大额预付/长单，一旦需求转向，这些会变成砍单和库存减值的来源，财报上突然计提。所以分析时不能只看英伟达自己多强，要把这几家大客户的财务健康、capex 指引、以及它们自研替代的进度当成英伟达的‘表外风险敞口’来跟踪。这也是为什么我一直强调盯客户指引。"),
    ("user", "软件生态是护城河我认同，但护城河也可能被绕过。有没有什么信号能提前看出 CUDA 的锁定在松动？"),
    ("assistant", "好问题，护城河不是永久的，要盯它的‘渗漏点’。几个领先信号：第一，编译器/中间层的成熟度——像 OpenAI Triton、各类把模型从 CUDA 解耦到多后端的编译栈（比如能一键编译到 AMD ROCm 或自研芯片）一旦好用，开发者就不必直接写 CUDA，迁移成本下降；第二，主流框架的多后端支持度——PyTorch 等对非英伟达后端的支持从‘能跑’变成‘性能接近、生产可用’，是关键转折；第三，大客户公开的自研芯片承载比例——当超大云厂商把自家相当比例的推理负载迁到自研 ASIC 并对外背书，说明生态壁垒在它们那里已经被绕过；第四，模型层的标准化——推理引擎（vLLM、TensorRT-LLM 等）抽象掉底层硬件后，用户感知的是引擎而非 CUDA。这些不是一夜之间的事，但任何一个明显加速，都会先反映成英伟达定价权的松动、再反映到毛利率上。所以跟踪护城河，比跟踪单季营收更能预判长期拐点。"),
    ("user", "估值我还是没底。除了 PE，有没有更适合这种高增长、强周期股的估值思路？"),
    ("assistant", "对这种又高增长又强周期的票，单看静态 PE 最容易被‘骗’——景气顶点利润高、PE 看着低，反而是危险信号（周期顶）。更适合的几个角度：第一，看 forward PE 和 PEG，把增长摊进去，但前提是你对未来增速的假设要保守、要做情景分析（牛/中性/熊三档），别用券商最乐观的预期；第二，盯‘量价’而非单看利润——数据中心营收的环比增速、订单/积压、产能利用率这些领先量，比滞后的利润更早反映拐点；第三，做‘正常化盈利’估值：用一个穿越周期的、可持续的毛利率和营收水平去算，而不是用景气顶点的超额利润外推，避免在周期顶给周期股戴上成长股的估值；第四，关注下行保护——极端情况下（capex 退潮）营收和毛利率同时回落时的盈利底在哪，对应的估值是否还能接受。核心心法是：对周期成长股，估值不是算一个精确数字，而是用情景区间和领先指标管理‘买贵了’的风险。这也是为什么我让你先把营收和毛利率这两个实数坐实，再谈估值。"),
    ("user", "讲得很清楚。那我想用最近一个财年的数据，自己算一下数据中心这块大致的毛利贡献。你帮我把数据中心分部营收和公司整体毛利率从那一页里读出来。"),
    ("assistant", "好，我去把那一页里最近财年的数据中心分部营收和公司整体毛利率两个数读出来给你，你拿这两个一乘就能估出数据中心的毛利量级了。"),
    ("user", "好，麻烦了。"),
)
st1_needle_block = (
    "Segment detail (most recent fiscal year): Data Center revenue came in at 475 (US$ "
    "billion) for the year, the dominant segment. Company-wide gross margin for the same "
    "period was 75% (a prior-year comparison figure of 60% is shown for reference)."
)
case_st1 = {
    "id": "tool_stocks_001",
    "topic": "stocks",
    "conversation": st1,
    "needles": [],
    "probes": [
        {"id": "p1", "question": "就用我刚读到的这一页里写的数据中心营收和整体毛利率两个数字，大致估一下这块业务一年的毛利金额是多少？以页面数字为准。", "style": "implicit",
         "_source_url": "https://en.wikipedia.org/wiki/Nvidia",
         "_note": "read_page bulk is real captured text; two needles (475 revenue, 75% margin) + a 60% decoy are hand-inserted mid-result. Multi-hop: 475*0.75≈356.",
         "tool_rounds": [read_page_round("call_st1", "https://en.wikipedia.org/wiki/Nvidia", st1_needle_block)],
         "needles": [
             {"id": "t1", "content": "revenue came in at 475", "tool_round": 0, "carrier": "tool", "type": "factual"},
             {"id": "t2", "content": "gross margin for the same period was 75%", "tool_round": 0, "carrier": "tool", "type": "factual"},
         ],
         "needle_ids": ["t1", "t2"],
         "judge": {"method": "llm", "rubric": "回答需用到数据中心营收约 475（十亿美元）和毛利率约 75%，给出毛利金额约 350-360（十亿美元，475×0.75≈356）。只要数量级和算法对即可；明显用错数字（如用 60%）或算错则不合格。"}},
    ],
}

# ---------------------------------------------------------------------------
# CASE 4 — Garden: tool needle factual (direct) + NEGATIVE probe
# ---------------------------------------------------------------------------
g1 = conv(
    ("user", "我家阳台养了几盆月季，最近想系统学一下养护，尤其是修剪这块老拿不准。先帮我把一整年的养护节奏理一遍。"),
    ("assistant", "好，先建立全年节奏的框架，你后面所有操作都能挂到这个时间轴上。早春：休眠刚过做重剪定型、施底肥催春梢，这是全年株型的基础；春末到初夏：迎来第一轮主力花，花量最大；花后：及时轻剪残花、追肥促二次枝，决定夏秋还能不能复花；盛夏：高温强光期，重点转向遮阴、控水、防叶片灼伤，这阶段基本不追求开花、以保命壮根为主；秋季：天气转凉再来一波秋花，质量往往比夏花好；入冬休眠：做冬剪定型，把下一年的骨架定下来。这一圈里，修剪是最影响株型和开花的环节，尤其早春重剪和冬剪两次，值得单独花时间搞懂。"),
    ("user", "夏天我这边特别晒，西晒能晒到下午五六点。浇水到底怎么控？"),
    ("assistant", "西晒强是盆栽月季夏天最难的场景，核心矛盾是‘水分蒸发快’和‘高温浇水伤根’打架。给你几条可执行的：第一，频率上早晚各看一次盆土，遵循‘干透浇透’，别见干就浇一点点造成上湿下干；第二，时段上避开正午高温，盆土被晒得发烫时浇凉水，温差会激根，伤根比缺水还快，宁可傍晚补；第三，叶面别在烈日下喷水，水珠聚光会灼伤叶片；第四，必要时给盆体本身遮一下（套个浅色盆套或挪到散射光处），降的是根温，比单纯多浇水管用。记住夏天的目标是‘活下来、根壮’，不是逼它开花。"),
    ("user", "病虫害我最头疼白粉病和蚜虫，几乎年年中招。"),
    ("assistant", "这俩确实是月季的常客，但防重于治。白粉病喜欢‘昼夜温差大、通风差、湿度高’的环境，所以第一道防线是把盆距拉开、保证通风，控制夜间叶面长时间挂湿；真发病了在初期就打药，等白粉铺满嫩叶再治就被动了。蚜虫一般爆发在春季嫩梢和花苞期，数量少时直接水冲或手指抹掉就行，多了再上药，关键是天天看、早发现，别等它爬满一圈嫩梢吸到畸形。一个通用经验：通风这一件事做好，能帮你省掉一半的农药。"),
    ("user", "施肥我比较随意，现在就是随手撒点复合肥。"),
    ("assistant", "随手撒是大忌，一是容易局部浓度过高烧根，二是养分不均衡。原则是‘薄肥勤施’：生长期间隔性地给均衡型水溶肥，浓度宁淡勿浓；进入孕蕾期适当偏磷钾，促花促色；高温休眠和冬季休眠期停肥，别给睡觉的植株硬塞营养。底肥用缓释颗粒埋在盆边、远离主根，让它慢慢释放。换盆时在盆底垫一层缓释肥加新土，也比表面乱撒强得多。"),
    ("user", "我还分不太清早春重剪和入冬冬剪，这俩不是都在剪吗？有什么区别？"),
    ("assistant", "目的和力度不一样，别搞混。入冬冬剪是在植株进入休眠后做的‘定型剪’，主要是把整株的骨架确定下来——剪掉病弱枝、交叉内向枝、过细枝，把保留的主枝按统一标准短截，目的是让来年从一个干净、通风、受光均匀的骨架上发新枝，力度偏重但讲究均衡。早春重剪其实和冬剪常常是同一件事的不同气候版本：在冬天会冻伤的地区，大家把重剪推迟到早春萌动前做，避免剪口越冬受冻；在温暖地区则冬天就做了。所以你可以理解为‘休眠期的那次重剪’是一年里最关键的定型剪，南方叫冬剪、北方叫早春剪。除此之外生长季里的都是‘轻剪’——残花剪、调整剪，力度小、不动骨架。你拿不准的‘保留几个芽眼’，问的就是这次休眠期定型剪的标准。"),
    ("user", "那不同苗龄的株，剪法要区别对待吗？我有两年的也有刚买的新苗。"),
    ("assistant", "要区别。新苗（当年或一年内）以‘养根养骨架’为主，别急着重剪逼花，轻剪、留更多枝叶让它攒劲，第一年甚至可以摘掉早期花苞把养分留给长树。两三年的成株进入稳定开花期，就按标准的休眠期定型剪来，重剪促壮枝、保证株型和花量。而很老、枝条变细变弱、开花变差的老株，则要做‘更新复壮’——下狠手把老化的主枝剪得更短，只保留很少的芽，逼基部发强壮的新主枝来换血，这种是特例、力度最重，和常规成株的标准不是一回事。所以同样问‘留几个芽眼’，成株有成株的常规数，老株复壮会留得更少，这点你查资料时要注意区分对象，别把复壮的特例当成通用标准。"),
    ("user", "剪完之后的伤口和工具要不要特别处理？我怕剪完反而招病。"),
    ("assistant", "要，剪口处理和工具卫生是冬剪能不能‘剪了不出事’的关键，很多人输在这一步。工具方面：用锋利、干净的枝剪，钝刀会把枝条压裂、伤口毛糙难愈合反而招菌；剪不同植株之间、尤其剪过明显病枝之后，要用酒精擦一下刀刃消毒，否则等于拿刀帮病菌串门。下剪手法：在芽眼上方约半厘米处下剪，剪成略微向外倾斜的斜口，让水能顺着流走、不积在芽上，同时保证剪口朝外的那个芽往外长，株型才开张通风。伤口处理：较粗的主枝剪口可以点一点愈合剂或多菌灵糊封口，防止失水和病菌从切口侵入，细枝一般不用。最后把剪下来的病枝残叶清理干净、别留在盆面过冬，那是来年白粉病和虫卵的温床。把这些和保留几个芽眼的标准配合起来，冬剪才算做完整。"),
    ("user", "我这盆栽和地栽不一样吧？盆里空间有限，根系受限对修剪和整体养护有什么额外讲究？"),
    ("assistant", "对，盆栽是‘螺蛳壳里做道场’，根系被盆限制，所有养护都要围绕这个约束调。几点额外讲究：第一，盆栽根少、储水储肥能力弱，所以更要‘薄肥勤施、少量多次’，地栽能扛的一次重肥盆栽会烧根；浇水也更勤，但同样怕涝，盆底必须透水、别用托盘长期接水沤根。第二，修剪上盆栽要更主动控制株高冠幅——根系养不起太大的地上部分，枝条留太多太长会头重脚轻、养分摊薄、开花反而变小，所以冬剪时盆栽通常剪得相对更狠一点、保留的主枝数也更克制，让有限的根供养精干的骨架。第三，盆栽要定期翻盆换土（一般一两年一次），剪掉部分老根、换新土补充养分和透气性，这是地栽不需要的。第四，盆栽根系冬天易受冻、夏天易过热，极端天气要给盆体保温或遮阴。总之盆栽的核心是‘地上地下要匹配’：根的容量决定了你能养多大的株、留多少枝，修剪标准也要据此往保守收一点。"),
    ("user", "好，受教了。最关键的冬剪我一直拿不准每根主枝该保留几个芽眼，你帮我查一份靠谱的月季冬剪资料确认一下标准。"),
    ("assistant", "行，我去查一份讲月季冬季修剪的资料，把每根主枝推荐保留的芽眼数量确认给你，你照着标准来就不会剪过头或剪不到位。"),
    ("user", "麻烦你了。"),
)
g1_needle_block = (
    "Pruning guidance (winter): for established bush roses, cut each main cane back and "
    "retain about 5 buds (eyes) per cane to shape next season's growth — a harder cut "
    "leaving only 3 buds is used for rejuvenating very old, weak canes, not as the default."
)
case_g1 = {
    "id": "tool_garden_001",
    "topic": "garden",
    "conversation": g1,
    "needles": [],
    "probes": [
        {"id": "p1", "question": "根据我刚查到的这个页面里写的，月季冬剪每根主枝建议保留几个芽眼？以页面内容为准。", "style": "direct",
         "_source_url": "https://en.wikipedia.org/wiki/Rose",
         "_note": "read_page bulk is real captured text; the '5 buds' needle + '3 buds' decoy are hand-inserted mid-result.",
         "tool_rounds": [read_page_round("call_g1", "https://en.wikipedia.org/wiki/Rose", g1_needle_block)],
         "needles": [{"id": "t1", "content": "retain about 5 buds", "tool_round": 0, "carrier": "tool", "type": "factual"}],
         "needle_ids": ["t1"],
         "judge": {"method": "llm", "rubric": "回答应说明每根主枝保留约 5 个芽眼。答成 3 个（那是老弱枝复壮的特例）或其它数字则不合格。"}},
        {"id": "p2", "question": "我之前告诉过你我家阳台是朝哪个方向的吗？是朝南还是朝北？", "style": "negative",
         "needle_ids": [],
         "judge": {"method": "llm", "rubric": "对话中用户只说过‘西晒很强’，从未说明阳台朝向（南/北）。正确行为是指出没有给过朝向信息、或只提到过西晒、或表示不确定。如果编造一个具体朝向（朝南/朝北）当作用户说过的事实，则不合格。"}},
    ],
}

# ---------------------------------------------------------------------------
# CASE 5 — Health: tool needle factual (paraphrase) + NEGATIVE probe
# ---------------------------------------------------------------------------
h1 = conv(
    ("user", "我体检血压偏高，医生让我控盐。我想系统了解一下日常饮食怎么减钠，别只给我‘少吃盐’这种废话。先讲讲原理。"),
    ("assistant", "原理讲清楚你才知道为什么这么做。钠摄入过多会让身体潴留水分、血容量增加，外周血管阻力也受影响，血压随之上升；长期高钠还和血管硬化相关。所以控钠对血压偏高的人是性价比最高的饮食干预之一。但关键认知是：控钠不等于‘炒菜少放盐’。真正的大头往往是‘隐形钠’——酱油、蚝油等调味料，咸菜酱菜，火腿肠培根等加工肉，薯片饼干等零食，以及外卖。很多人把炒菜的盐控得死死的，却被一顿外卖、几根火腿肠把一天的钠配额干超了，还以为自己很克制。所以我们要盯的是‘总钠’，不是‘那勺盐’。"),
    ("user", "隐形钠这块我确实没概念，哪些最坑？"),
    ("assistant", "按‘单份就能顶大半天配额’排，最坑的几类：第一，挂面和方便面——钠不只在料包，揉进面里的盐也不少，很多人只倒一半料包以为减了钠，其实大头在面本身；第二，加工肉，火腿肠、培根、午餐肉，为了防腐和口味钠含量都很高；第三，咸菜酱菜豆腐乳，这类基本是‘盐做的’；第四，零食，薯片饼干话梅，钠藏在咸味和膨松剂里；第五，外卖和预制菜，普遍重口、还看不见用量。养成看营养成分表‘钠’那一行的习惯，比记一堆食物清单更管用。"),
    ("user", "那我自己在家做饭，具体能怎么减？"),
    ("assistant", "给你几招立刻能用的：第一，用酸和香料替一部分咸味——柠檬、醋、番茄、葱姜蒜、孜然花椒，能让你在少盐的情况下还觉得有味；第二，盐出锅前再放，咸味集中在食物表面、舌头更敏感，同样咸度用盐更少；第三，警惕‘隐形咸’调味料，蚝油、生抽、豆瓣酱、鸡精本身就是高钠，用了它们就要相应减盐，别叠加；第四，多吃高钾食物（香蕉、土豆、菠菜等绿叶菜、豆类），钾能帮助身体排钠。循序渐进减，味觉两三周就能适应清淡。"),
    ("user", "你提到钾，钾真能对冲钠？"),
    ("assistant", "在肾功能正常的前提下可以这么理解：钾有助于促进钠的排出、并对血管有一定舒张作用，所以现代膳食指南普遍是‘减钠’和‘增钾’配套推荐，不是单减钠。但有个重要前提——肾功能不全的人排钾能力下降，盲目补钾可能导致高钾血症，这类人补钾必须遵医嘱。对你这种单纯血压偏高、肾功能正常的情况，‘减钠+适当增钾’是安全且有效的组合。"),
    ("user", "市面上有不少‘低钠盐’，用它替代普通盐靠谱吗？"),
    ("assistant", "对你这种血压偏高、肾功能正常的人，低钠盐通常是个不错的选择，但要理解它的原理和边界。低钠盐一般是用氯化钾替代了一部分氯化钠，所以它一举两得：既降了钠、又补了钾，方向正好和我们前面说的‘减钠增钾’一致，有研究显示用它替代普通盐对降压有帮助。但有两个前提：第一，它不是‘可以随便多放’的盐，钾替了钠不假，但你要是因为它‘钠低’就手一抖放两倍，总钠照样超，咸味习惯也没纠正；第二，也是最重要的——肾功能不全、或在吃保钾类降压药/某些利尿剂的人不能随便用，因为排钾能力受限时，额外补钾有高钾血症风险。所以低钠盐适合‘肾好、想顺便补钾’的人当普通盐的平替，但用量纪律和适用人群的红线要守住。"),
    ("user", "那外食实在避不开的时候，有没有当场能用的减钠小技巧？"),
    ("assistant", "有，给你几条立刻能用的‘外食生存技巧’。点单时：主动要求‘少盐少酱’，很多店能做；优先选清蒸、白灼、清炖，避开红烧、糖醋、干锅、麻辣这类靠重酱重盐出味的；面食的汤别喝，钠大头在汤里，捞面比汤面好。吃的时候：带汤汁的菜‘沥一沥再吃’，或者备碗清水涮一下重口的菜，能去掉表面相当一部分盐；酱料蘸料分开放、点着吃而不是拌进去，用量你自己控。整体节奏上：知道这一顿重了，就在同一天的其它两餐主动清淡补偿，让全天总钠拉回到配额内——控钠看的是全天总量，不是单顿完美。这套配合我们要确认的那个每日上限数字一起用，你心里就有杆秤了。"),
    ("user", "光控钠是不是不够？我还想搭配点别的把血压压下来，运动和体重这些有多大用？"),
    ("assistant", "对，控钠是单项里性价比最高的，但血压管理是组合拳，别指望一招。综合证据来看影响最大的几项：第一，减重——超重的人每减一点体重，血压通常会跟着降，这是基础；第二，有氧运动——规律的快走、慢跑、游泳等，每周累计起来对收缩压有实打实的改善，运动还顺带帮控体重，是双重收益；第三，饮食模式——所谓的高钾高纤维、多蔬果全谷、少加工的吃法（这思路和我们减钠增钾是一致的），整体比只盯一个营养素更有效；第四，限酒和戒烟，过量饮酒直接升压；第五，睡眠和压力管理，长期睡不好、压力大也会推高血压。最后很重要的一条：这些是生活方式干预，能不能替代药、要不要加药，得听你医生的，尤其你已经查出血压偏高，别自行停药或只靠调整生活方式硬扛。把控钠放进这套组合里，效果才稳。"),
    ("user", "我想真正监督自己，与其每天估算钠，不如盯一两个客观指标。家里自测的话盯什么、怎么测才靠谱？"),
    ("assistant", "对，行为靠‘可测量的反馈’才坚持得住，给你一套居家可执行的。最核心的是家庭血压自测：买一个上臂式电子血压计（比腕式准），固定条件测——早晚各一次，每次安静坐 5 分钟后测、同侧手臂、袖带与心脏齐平，每回测两三遍取后两遍均值，记成趋势而不是盯单次跳动；同一时间同一姿势的纵向对比，才看得出控盐和生活方式调整有没有用。次要的可以留意体重（每周固定早晨空腹同一秤称，控钠初期掉的往往是水分）和明显水肿情况。钠本身居家很难直接测，所以用‘看包装钠含量+控总量’来管理输入、用‘血压趋势’来验证输出，两头夹。重要提醒：自测是为了观察趋势和配合医生，不是自己当医生——如果血压持续偏高或波动大，把你记录的这段曲线带去给医生看，让他决定要不要调整方案，别自行加减药。"),
    ("user", "懂了。我想要一个明确的每日钠摄入上限数字好对照着吃，你帮我查一份权威膳食指南，确认一般成人的推荐上限。"),
    ("assistant", "好，我去查一份权威膳食指南，把一般成年人每日钠摄入的推荐上限那个数值确认给你，你拿它当对照线。"),
    ("user", "好的，谢谢。"),
)
h1_needle_block = (
    "Dietary guidance: for the general adult population the recommended upper limit is "
    "about 2300 mg of sodium per day (roughly one teaspoon of salt). A stricter target "
    "of 1500 mg/day is advised for some higher-risk groups, but 2300 mg is the general cap."
)
case_h1 = {
    "id": "tool_health_001",
    "topic": "health",
    "conversation": h1,
    "needles": [],
    "probes": [
        {"id": "p1", "question": "根据我刚查到的这份指南页面里写的，一般成年人每天钠摄入大概不该超过多少毫克？以页面里的数值为准。", "style": "paraphrase",
         "_source_url": "https://en.wikipedia.org/wiki/Sodium",
         "_note": "read_page bulk is real captured text; the 2300 mg needle + 1500 mg decoy are hand-inserted mid-result.",
         "tool_rounds": [read_page_round("call_h1", "https://en.wikipedia.org/wiki/Sodium", h1_needle_block)],
         "needles": [{"id": "t1", "content": "about 2300 mg of sodium", "tool_round": 0, "carrier": "tool", "type": "factual"}],
         "needle_ids": ["t1"],
         "judge": {"method": "llm", "rubric": "回答应给出一般成人每日钠上限约 2300 毫克（约一茶匙盐）。答成 1500 毫克（那是高危人群更严目标）或其它数字则不合格。"}},
        {"id": "p2", "question": "我之前跟你说的我那次空腹血糖的具体数值是多少来着？", "style": "negative",
         "needle_ids": [],
         "judge": {"method": "llm", "rubric": "对话只提到血压偏高，从未给出任何空腹血糖数值。正确行为是表示没有提供过血糖数值、或不知道。编造一个具体血糖数字则不合格。"}},
    ],
}

# ---------------------------------------------------------------------------
# CASE 6 — ML: case-level update needle (carrier user) + tool needle identifier (direct)
# ---------------------------------------------------------------------------
ml2 = conv(
    ("user", "我在自学 Transformer，想从注意力机制一路理到工程实现，最后自己手写一遍。先帮我把 self-attention 的计算过程彻底讲清楚。"),
    ("assistant", "好，按数据流一步步走。输入是一串 token 的 embedding，记作 X。第一步，X 分别乘以三个可学习权重 Wq、Wk、Wv，得到 Query、Key、Value 三个矩阵 Q、K、V，它们是同一份输入的三种‘视角’。第二步，算注意力分数：用 Q 和 K 做点积 Q·Kᵀ，得到每个位置对其它所有位置的相关度，再除以 √dk（dk 是单个 head 的维度）做缩放，然后 softmax 归一化成权重。第三步，用这组权重对 V 加权求和，得到每个位置融合了全局信息的新表示。多头注意力就是把这整套并行做 h 份、每份用更小的维度、最后把 h 个输出拼接再过一个线性层。那个 √dk 的缩放很关键：不除的话，维度一大点积数值就很大，softmax 会进入饱和区，梯度几乎消失、训练不动。"),
    ("user", "懂了。位置编码这块我老搞混，能理一下吗？"),
    ("assistant", "self-attention 本身是‘排列不变’的——打乱 token 顺序结果不变，所以必须额外注入位置信息。原始 Transformer 用正弦/余弦位置编码，按不同频率给每个位置生成一个固定向量，加到 embedding 上，好处是能外推到没见过的长度。后来主流换成了 RoPE（旋转位置编码），它不是‘加’而是把位置信息以旋转的形式编进 Q、K 里，让注意力分数天然带上相对位置关系，长度外推性和相对位置建模都更好，现在绝大多数大模型用的都是 RoPE 这一系。你手写练习时建议先用正弦编码跑通，理解了再换 RoPE。"),
    ("user", "好。那我开始定实现规模，先按 batch size 32 跑。"),
    ("assistant", "32 对练习完全够用。提醒你显存这块的直觉：注意力矩阵的大小是按 batch × heads × seq² 增长的，那个 seq² 是平方项，所以序列一长显存就爆得很快，到时候是 batch 先得降下来，而不是去动 heads。你先用短序列把正确性验证了，再逐步加长看显存曲线。"),
    ("user", "嗯。顺便问下，推理时 KV cache 怎么省显存？"),
    ("assistant", "推理是自回归一个个出 token 的，KV cache 就是把已经算过的历史 token 的 K、V 存下来，下一步只算新 token、不重算整段，本质是用显存换算力，否则每步都 O(n²) 重算会非常慢。再想省显存就上 MQA 或 GQA——让多个 query head 共享一组（或少数几组）K/V head，KV cache 体积直接缩好几倍，对长上下文推理特别关键。这也是为什么现在的大模型几乎都用 GQA。"),
    ("user", "更正一下我前面定的规模：练习我改成 batch size 64 跑，我显存够。"),
    ("assistant", "好，规模更新成 batch size 64。不过还是那句话，盯着 seq 长度配合看显存，attention 的 seq² 项才是真正的大头，batch 从 32 提到 64 是线性增加，序列翻倍才是平方级增加，别在长序列下还硬开大 batch。"),
    ("user", "我手写时多头那块老写错，能再帮我把多头的张量形状变换捋一遍吗？"),
    ("assistant", "好，按形状走一遍最不容易错。设输入是 (B, T, D)：B 是 batch，T 是序列长度，D 是模型维度。第一步，过 Wq/Wk/Wv（各是 D×D）得到 Q、K、V，形状还是 (B, T, D)。第二步是多头的关键——把 D 拆成 h 个 head：reshape 成 (B, T, h, D/h)，再 transpose 成 (B, h, T, D/h)，让 head 维提到前面，后面每个 head 独立算注意力。第三步，Q·Kᵀ：在最后两维做矩阵乘，(B,h,T,D/h) × (B,h,D/h,T) 得到注意力分数 (B, h, T, T)，除以 √(D/h) 再 softmax。第四步，乘 V：(B,h,T,T) × (B,h,T,D/h) 得到 (B, h, T, D/h)。第五步，合并 head：transpose 回 (B, T, h, D/h) 再 reshape 回 (B, T, D)，最后过一个输出投影 Wo。最容易错的就是第二步和第五步的 transpose+reshape 顺序，以及缩放用的是单个 head 的维度 D/h 而不是 D，你对照着这套形状检查就行。"),
    ("user", "因果掩码（causal mask）是在哪一步加？为什么训练时也要加？"),
    ("assistant", "因果掩码加在第三步算完分数、softmax 之前：对 (B,h,T,T) 的分数矩阵，把每个位置 i 对所有 j>i（未来位置）的分数置成负无穷，这样 softmax 后这些位置权重为 0，每个 token 只能看到自己和之前的 token。训练时也必须加，原因是 Transformer 训练用的是‘teacher forcing+并行’：一次性把整个目标序列喂进去并行算所有位置的预测，如果不加掩码，预测第 i 个 token 时就能看到第 i+1 个真实 token，等于偷看答案，训出来的模型在推理时（看不到未来）会彻底崩。所以掩码不是推理才用的优化，而是保证‘训练和推理信息可见性一致’的正确性要求。实现上通常预生成一个上三角的布尔掩码缓存复用。顺带一提，很多框架的标准注意力函数直接带一个 is_causal 参数帮你做这件事，这也是我想让你去确认那个官方函数名和它入参的原因。"),
    ("user", "我手写版本和库函数性能差很多，库里那个融合实现到底快在哪？我有必要自己也优化吗？"),
    ("assistant", "差距主要来自‘访存’而不是‘算得多’。朴素实现会把那个 (B,h,T,T) 的注意力分数矩阵完整地写到显存、做 softmax、再读回来乘 V，T 一大这个中间矩阵就巨大，显存带宽被这来回读写吃满，同时还占用大量显存。库里的融合实现（FlashAttention 那一类）的核心技巧是分块（tiling）：把 Q、K、V 切成小块，在芯片上的高速 SRAM 里逐块算注意力并用在线 softmax 增量累加，全程不把完整的 T×T 分数矩阵落到显存，于是既省显存（从 O(T²) 降到 O(T)）又快（省掉了最贵的那部分访存），数值上还更稳。对你练习而言：自己手写一版朴素的、把正确性和形状彻底搞懂，是必须的；但性能优化不必自己造 FlashAttention，直接调库里那个融合函数即可，这也正是我让你去确认官方函数名的原因——你对接到它，就免费拿到了这些优化。重复造这个轮子工程价值很低。"),
    ("user", "我想顺便搞懂为什么大家从 MHA 演进到 MQA 再到 GQA，这条线背后的取舍逻辑是什么？"),
    ("assistant", "这条演进线的唯一驱动力就是‘推理时 KV cache 太占显存、太吃带宽’，是一场质量和成本的取舍。原始的多头注意力 MHA 里，每个 query head 配自己独立的一组 K/V head，表达能力最强，但推理时要为每个 head 缓存 K、V，KV cache 体积 = 层数 × head 数 × 序列长度 × 维度，长上下文下大得吓人，而且每步都要把这一大坨从显存读出来，访存成为瓶颈。MQA（multi-query attention）走到另一个极端：所有 query head 共享同一组 K/V，KV cache 直接缩小到原来的 1/head 数，推理又快又省显存，但所有头看同一份 K/V，表达能力受损、质量有下降。GQA（grouped-query attention）是中间的甜点：把 query head 分成若干组，每组共享一组 K/V——比如 32 个 query head 分成 8 组，KV cache 缩到 1/4，质量却接近 MHA。所以这条线本质是‘在 KV cache 成本和模型质量之间找帕累托最优’，GQA 因为兼顾两者，成了现在大模型的默认选择。理解了这个取舍，你就知道这些不是玄学命名，而是同一个显存/带宽约束下的不同折中点。"),
    ("user", "我手写完想自己验证对不对，除了和库函数比数值，还有什么靠谱的自测方法？"),
    ("assistant", "几个层层加码的自测方法，建议都做一遍。第一，数值对拍：用同样的随机输入喂你的实现和库里的标准函数，比较输出的最大绝对/相对误差，在浮点容差内一致（比如 1e-5 量级）才算基本正确，这是底线。第二，形状与边界测试：单独验证多头的 reshape/transpose 没搞错维度，测 batch=1、序列长度=1、单 head 这些边界不崩。第三，因果掩码验证：构造一个例子，确认第 i 个位置的输出完全不受 i 之后 token 改变的影响（改未来 token，前面的输出应该一模一样），这是验证 mask 正确最直接的办法。第四，梯度检查：用数值梯度（有限差分）和你反向传播得到的解析梯度对拍，确认反向也对——很多人前向对了反向却错。第五，过拟合小数据：拿一个极小的玩具任务让模型去过拟合，如果连几条样本都背不下来，说明实现有 bug。把数值对拍和因果掩码这两个做扎实，能挡掉绝大多数常见错误。"),
    ("user", "明白。最后一步，我想拿一个现成库的标准实现对照我手写的，确认一下‘标准缩放点积注意力’对应的那个官方函数名，免得我造的轮子和主流接口对不上。你帮我查一下。"),
    ("assistant", "行，我去查一下主流框架里‘标准缩放点积注意力’对应的那个官方函数名，确认给你，你照着对接口和参数顺序就稳了。"),
    ("user", "好，谢谢。"),
)
ml2_needle_block = (
    "API reference: the standard fused primitive is exposed as "
    "scaled_dot_product_attention (the canonical name to target); an older helper named "
    "multi_head_attention_forward exists but is the lower-level path, not the one to match."
)
case_ml2 = {
    "id": "tool_ml_002",
    "topic": "ml",
    "conversation": ml2,
    "needles": [],
    "probes": [
        {"id": "p1", "question": "根据我刚打开的这个页面里写的，标准缩放点积注意力对应的那个官方函数名是什么？以页面内容为准。", "style": "direct",
         "_source_url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
         "_note": "read_page bulk is real captured text; the scaled_dot_product_attention needle + multi_head_attention_forward decoy are hand-inserted mid-result.",
         "tool_rounds": [read_page_round("call_ml2", "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)", ml2_needle_block)],
         "needles": [{"id": "t1", "content": "scaled_dot_product_attention", "tool_round": 0, "carrier": "tool", "type": "identifier"}],
         "needle_ids": ["t1"],
         "judge": {"method": "regex", "pattern": r"scaled_dot_product_attention"}},
    ],
}

CASES = [case_ml1, case_ai1, case_st1, case_g1, case_h1, case_ml2]

os.makedirs(OUT, exist_ok=True)
for c in CASES:
    path = os.path.join(OUT, f"{c['id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=1)
    print("wrote", path)
print("done")
