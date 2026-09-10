# ML Fundamentals — Week 3 Notes

## One thing that surprised me

A "neuron" is literally just a number — an activation value between 0 and 1. The model's
intelligence lives entirely in the connection weights between neurons, not in the neurons
themselves. "Training" is not magic: gradient descent repeatedly takes tiny downhill steps
over a massive multidimensional loss landscape, and backpropagation computes which weights
contributed most to each mistake. The word "learning" is a metaphor for numerical
optimization.

The Hugging Face `pipeline()` abstraction made this feel production-real: three lines of
code load the model, tokenizer, preprocessing, inference, and postprocessing automatically.
Transformers feel less like research tools and more like reusable infrastructure components.

## Connection to my APM / SRE background

The entire training loop maps cleanly onto distributed systems observability:

- **Cost function ~ SLI / error budget**: measures how wrong the model is, exactly like
  an SLO measures how far a system is from acceptable performance.
- **Backpropagation ~ distributed tracing / RCA**: error propagates backward through layers,
  attributing contribution to each weight — the same way span tracing attributes latency
  back through microservice dependencies.
- **Gradient descent ~ feedback controller / auto-remediation**: the system continuously
  nudges itself toward lower error, like an auto-remediation loop driving toward SLO
  compliance.
- **Learning rate ~ autoscaling sensitivity**: too aggressive causes oscillation (overshoots
  the loss minimum); too small means slow convergence — the exact same tradeoff as
  autoscaling step size.

Also: transformers handling all NLP tasks (classification, NER, summarization, generation)
under one architecture feels structurally like cloud-native platforms unifying infrastructure
concerns under one orchestration layer. Same architecture, different configuration.

## One thing I still don't understand

How local mathematical weight updates on billions of parameters produce emergent
capabilities — reasoning, planning, coherent long-form generation — rather than just
sophisticated pattern matching. The mechanics of learning are clear; where *understanding*
comes from is not.

This turns out to be an active research question, not just a personal knowledge gap. The
attention mechanism (Day 3) will explain how transformers work mechanically, but won't
fully resolve it — because the field hasn't either.

---

# Day 2 — The Illustrated Transformer + HF Ch1 §4–7

## What is attention?

Attention is a dynamic key-value lookup where each token computes a weighted vector sum
over all other tokens in the sequence based on context relevance. Every token asks "who
is relevant to me?" (query), every token broadcasts "here's what I am" (key), and the
answer is a weighted blend of what those relevant tokens contain (value). The weights are
learned, not fixed — so the model figures out what to attend to during training.

Note: "parallelizable" compared to RNNs (which processed tokens sequentially), but
attention itself is O(n²) in sequence length — every token attends to every other token.
Doubling context length roughly quadruples compute.

## Context window vs model weights

Model weights are a fixed, one-time infrastructure cost — loaded into GPU memory at
startup, identical across all requests. When someone says "Llama 70B," they mean the
size of these weights.

Context window is variable, per-request operational cost that scales with input length
due to dynamic compute and memory allocation. The cost scaling is nonlinear: attention
is O(n²), so doubling context length quadruples compute. A 32K-token request is not
8× a 4K request — it's closer to 64×. This makes context length the primary FinOps
lever for LLM serving.

## Performance / observability implication: KV cache

The KV cache turns a stateless transformer model into a stateful, memory-bound bottleneck.
Every generated token forces the server to retain growing context tensors in VRAM, causing
linear memory growth per active user session. Under high concurrency this creates a
noisy-neighbor eviction problem: a user with a long context competes for VRAM with every
other concurrent session.

This is why P99 latency for LLMs must be bucketed by output token count, not just request
count — a 10-token response and a 2000-token response on the same model are not comparable
latency events. Standard APM dashboards will mislead you if you treat them the same way.

KV cache pressure is what vLLM was built to solve (Week 17).

---

# Day 3 — Illustrated GPT-2 + Prefill/Decode

## 1. Prefill vs decode — and why the distinction matters for monitoring

**The Prefill Phase (The Sprint)**
The model processes the input tokens all at once, computing the KV cache for the entire prompt in a single batched forward pass. Highly compute-bound — parallelizes beautifully across GPU cores (matrix multiplication heavy). Metric to watch: **TTFT (Time to First Token)**. If TTFT is spiking but generation is fast, users are likely dumping massive contexts into the prompt, or KV cache allocation is thrashing.

**The Decode Phase (The Marathon)**
The model generates the response one token at a time. Each new token requires loading the model weights and the entire KV cache from HBM to SRAM — cannot be parallelized across time. Highly memory-bandwidth bound. Metric to watch: **ITL (Inter-Token Latency) / TPOT**. If TPOT is crawling, GPUs are likely starved for memory bandwidth or the continuous batching mechanism is overloaded.

A single p99 latency alert conflates two unrelated problems. TTFT spiking means prompt processing is slow — likely a compute or batching issue. ITL spiking means generation is slow — likely a memory bandwidth or KV cache pressure issue. The fix for one doesn't help the other. An on-call engineer needs both metrics, broken out.

## 2. Why "tokens per second" is a misleading single metric

"Tokens per second" blends the massive, parallel throughput of the prefill phase with the slow, sequential pace of the decode phase into a single meaningless average. Because processing an input prompt is orders of magnitude faster in terms of tokens/sec than generating a response, a high overall metric can easily mask a painfully sluggish user experience if the generation phase is bottlenecked.

The same model on the same GPU can show 10× different tokens/sec numbers depending on batch size, output length distribution, and concurrent request count. A benchmark that says "X tokens/sec" without specifying those three things is not a benchmark — it's a marketing number.

## 3. Open question going into Day 4 (Chip Huyen)

How do you actually measure prefill and decode tokens/sec separately in a production system, and what are the concrete levers for improving ITL — speculative decoding, batching strategies, hardware choices?

---

# Day 5 (deferred from June) — LLM Evals

Source: Eugene Yan, "Patterns for Building LLM-Based Systems & Products," Pattern 1: Evals.

- **Evals are the test pyramid I already have, with the assertion loosened.** My respx-mocked failure paths in `tests/test_joke.py` pin a fixed input to an exact assertion (`route.call_count == MAX_ATTEMPTS`), which is exactly the heuristic/rule-based layer; what changes for LLMs is not the harness but that the output is non-deterministic, so the exact match degrades into a fuzzy one (BLEU/ROUGE) and then into a model-graded or human judgment when even fuzzy fails.

- **The operational gap is that almost nobody has a regression signal — "no evals beyond vibes."** Yan's argument is that evals are the *starting point*, not the QA step at the end, and the SRE translation is exact: shipping prompt changes without an eval set is shipping without an SLI, so every change is unfalsifiable and every regression is discovered by users.

- **Question going into Project A (Weeks 13-14):** if self-enhancement bias is real — GPT-4 favors itself ~10%, Claude-v1 ~25% — then judging a Claude-based RAG system with Claude is measuring the wrong thing, so what's the actual escape? And for RAG specifically, is the reference set pinned to *retrieved passages* or to *final answers*, since those two fail independently.

---

# Day 6 — First real LLM call

- **First real Anthropic API call: 28 tokens in, 138 tokens out, 0.359 cents** (`claude-opus-5`, effort `low`, single non-streaming `messages.create`). Output tokens are priced 5× input ($25 vs $5 per MTok) and there were ~5× more of them, so ~96% of the cost was generation — the reverse of what pdf-summarizer will look like, where a large PDF makes input dominate and prompt caching becomes the lever.

---

# Day 7 — LLM latency variance

- **Same prompt, 5 runs, `claude-haiku-4-5`: latency 1067–1717 ms (mean 1542, 42% of mean).**
  Input tokens identical at 25 every run; output tokens varied 40–50. Cost $0.000225–$0.000275
  per call, $0.001305 for all five.

- **The spread is mostly output length, not API noise.** The fastest run was also the shortest
  output (40 tokens / 1067 ms); normalizing to ms-per-output-token tightens the range from 61%
  to 34% (26.7–35.8 ms/tok). This is decode-phase dominance measured directly — each output
  token is a sequential forward pass, so latency tracks how much the model says.

- **Cost is deterministic on token counts; latency is not.** 25 × $1/M + 48 × $5/M = $0.000265,
  exact to the last digit. So an SLO on LLM serving has to bucket by output length before it
  means anything — the Day 3 claim ("P99 must be bucketed by output token count") is now
  something I've measured rather than read. Averages across mixed output lengths compare
  nothing to nothing.

- Also: requesting `claude-haiku-4-5` returned `claude-haiku-4-5-20251001` — the alias resolves
  to a dated build, so a latency baseline meant to hold across weeks should pin the snapshot.

---

# Day 8 — TTFT vs ITL vs total latency

Streaming API, same prompt, 5 runs, `claude-haiku-4-5` (25 input tokens, 39–49 output):

- **TTFT: 664–881 ms** (mean 765, spread 28% of mean)
- **ITL: 8.5–12.9 ms/token** (mean 11.6, spread 38%)
- **Total: 1206–1434 ms** (mean 1294, spread 18%)

- **TTFT is ~59% of total latency, not decode.** At 48 output tokens the prefill/queue phase
  costs more than the entire decode phase. The Day 3 framing (prefill = sprint, decode =
  marathon) holds at long outputs and inverts at short ones — where the crossover sits is the
  operationally useful question, and it isn't in those notes.

- **Total latency was the *most* stable of the three metrics, not the least.** Predicted the
  opposite (that output-token variation would dominate total variance). Components that vary
  somewhat independently partially cancel when summed, so a stable aggregate can hide two
  unstable parts — an argument for component SLIs that is stronger than the one I wrote on
  Day 3, and for the opposite reason.

- **Caveat that limits all of the above: `stream_chunks` was only 3–4 for ~46 tokens.**
  Something on the path is coalescing SSE deltas, so "first chunk" != "first token" and ITL
  is arithmetic on an average rather than an observed inter-token gap. Buffering moves time
  out of decode and into TTFT, which is the exact shape of the result — so the 59/41 split
  is not yet trustworthy.

- **Open question for Day 9:** is the coalescing client-side (SDK/TLS record buffering) or
  network-side? Test: log per-chunk arrival time and character length. Many small chunks with
  even spacing means real per-token streaming and the split stands; few large chunks means
  TTFT is inflated and ITL understated.

- Streaming total (mean 1294 ms) came in ~250 ms under Day 7's non-streaming total
  (mean 1542 ms), consistent with the non-streaming call buffering the full response before
  sending — but n=5 can't establish that.

---

# Day 9 — Long output: decode linearity and the buffering artifact

Same script + per-chunk arrival logging, prompt forced to ~500 words, 5 runs,
`claude-haiku-4-5` (27 input tokens, 1388–1607 output).

| run | out tok | chunks | median chunk | median gap | TTFT ms | total ms | ITL ms/tok |
|-----|---------|--------|--------------|------------|---------|----------|------------|
| 1   | 1388    | 53     | 113 ch       | 316.2 ms   | 1369    | 17741    | 11.8       |
| 2   | 1478    | 50     | 123 ch       | 316.5 ms   | 710     | 16259    | 10.5       |
| 3   | 1607    | 56     | 107 ch       | 315.3 ms   | 662     | 18034    | 10.8       |
| 4   | 1454    | 55     | 119 ch       | 317.5 ms   | 667     | 17756    | 11.8       |
| 5   | 1520    | 648    | 9 ch         | 26.1 ms    | 692     | 17922    | 11.3       |

- **The Day 8 buffering caveat is resolved, and the metrics survive it.** Runs 1–4 were
  coalesced (~113-char chunks, median gap 316 ms — that regularity across four runs is a fixed
  flush timer on the network path, not model behaviour); run 5 escaped it (648 chunks, 9 chars,
  26 ms). Identical script, machine, and minute, so the coalescing is intermittent and
  transport-side. ITL landed at 10.5–11.8 ms/token in *all five* regardless, because it is
  derived from total decode time, not from observed gaps. TTFT was likewise unaffected
  (662–710 coalesced vs 692 fine-grained) because `chunk_chars_min` is 3–10 — the first chunk
  stays small even when later ones batch. Lesson: log delivery granularity as a *diagnostic*,
  but don't derive latency SLIs from inter-chunk gaps.

- **Decode is linear in output length.** ITL was 11.6 ms/token at ~48 output tokens (Day 8) and
  11.2 ms/token at ~1490 output tokens here. 30× the output, ~3% change in per-token cost.
  This is the assumption underneath every LLM latency SLO, and it holds.

- **TTFT is independent of output length, and tracks input.** ~765 ms on Day 8 (25 input
  tokens), ~683 ms here (27 input tokens), while output grew 30×. Prefill is a function of the
  prompt; decode is a function of the response. They are genuinely separable metrics.

- **The prefill/decode crossover is ≈ 61 output tokens.** With TTFT ≈ 680 ms and ITL ≈ 11.2
  ms/token, decode overtakes prefill at 680/11.2 tokens. Day 8's 48-token runs sat just under
  it (TTFT = 59% of total); today's ~1490-token runs are far past it (TTFT = 3.9%). The Day 3
  "prefill = sprint, decode = marathon" framing is correct only above ~60 output tokens — for
  short responses the sprint is the whole race. **This is the number to know per model/route:
  it decides whether latency work should target the prompt or the response.**

- Run 1's TTFT of 1369 ms (2× the others) is the cold start predicted on Day 7 and not seen
  then — first call of a batch, paying DNS/TLS/connection setup.

- Cost: $0.0070–$0.0081 per call, $0.0374 for all five.

## Cross-model comparison (haiku-4-5 vs sonnet-4-5)

Same script, 5 short + 5 long runs on each model. Prices verified against
platform.claude.com pricing page 2026-09-09: haiku-4-5 $1/$5 per MTok,
sonnet-4-5 $3/$15. (Not $0.80/$4 — those are Haiku *3.5*'s retired rates.)

| model | TTFT ms (long, excl cold) | ITL ms/tok | crossover (tok) | $/output token | $/call short | $/call long |
|-------|---------------------------|------------|-----------------|----------------|--------------|-------------|
| haiku-4-5  | 683  | 11.2 | ~61 | $5.02e-6  | $0.000255 | $0.00747 |
| sonnet-4-5 | 1316 | 23.1 | ~57 | $1.508e-5 | $0.000744 | $0.01601 |
| ratio      | 1.93x | 2.06x | ~1.0x | 3.00x | 2.92x | 2.14x |

- **The crossover is model-invariant at ~60 output tokens, and that is the finding.**
  Predicted (both by me and the plan) that a bigger model would lower the crossover
  because decode would get disproportionately more expensive. Wrong: TTFT scaled 1.93x
  and ITL scaled 2.06x, so their *ratio* barely moved. Model choice changes absolute
  latency without moving the point where decode overtakes prefill. Operationally this
  means the output-length bucketing threshold is portable across models — derive it
  once, not per model.

- **Cost per output token is exactly the price ratio (3.004x); cost per call is not
  (2.14x).** Sonnet wrote ~1062 tokens where haiku wrote ~1489 for the identical prompt,
  so the per-call number understates the true multiple by a third. A model that looks
  cheaper per request may simply be less verbose — normalize by tokens before comparing,
  or a model-swap decision gets made on the wrong number.

- **Coalescing reproduces on a second model, so it is path-side, not model-side.** Exactly
  one of five long runs escaped it on each model (haiku: 648 chunks vs ~53; sonnet: 428 vs
  ~35). It still doesn't perturb the derived metrics: sonnet's fine-grained run gave ITL
  23.7 against 22.3–23.5 for the four coalesced ones. Day 8's caveat is now closed on two
  models.

- Cold start appears again as run 1 of each batch (sonnet short run 1: TTFT 1787 vs
  1256–1530 for runs 2–5). Consistent across three days now — worth discarding the first
  run of any latency batch by default.

- Open question: at 25–27 input tokens, prefill compute is negligible, so why does TTFT
  double between models? The model-dependent part of TTFT (scheduling, weight/KV setup)
  must scale with model size even when there is almost nothing to prefill. Network RTT is
  model-independent and cannot explain it.

- Cost: $0.0838 for all ten calls ($0.00372 short, $0.08007 long).

## Cost is deterministic on tokens, latency is not — and that splits the SLO surface

Every call across Days 6–9 reproduced its cost exactly from input and output token counts;
not one needed a percentile. Latency never reproduced twice. That asymmetry is not a
curiosity, it decides how the two get monitored.

Cost belongs in budget alerts, not percentile dashboards. It is strictly additive and
perfectly predictable given token counts, so the only thing worth alerting on is a change
in those counts: a prompt that grew, output that stopped being bounded, a model swap that
went out unnoticed. A cost SLO is really a token-count SLO wearing a dollar sign, and it
fires on regressions in the *system*, not on variance in the service.

Latency needs both percentiles and workload bucketing, and bucketing is the part people
skip. A p99 over mixed traffic — chat, summarization, extraction, code gen — is dominated
by whichever bucket happens to generate the most tokens, so the number moves when the
traffic mix moves and tells you nothing about the service. Bucket first, then measure TTFT
and ITL separately inside each bucket. That is the real operational tax of putting several
workloads behind one endpoint.

The crossover decides which SLI leads. Below ~60 output tokens — classification,
extraction, routing, short chat turns — TTFT is nearly the whole request, and work to
reduce latency belongs on the prompt and the queue. Above it, ITL x output_tokens
dominates and the lever is output length and decode throughput. The ~60 figure held across
a 3x price range, so it can be treated as a property of the serving architecture rather
than of the model.
