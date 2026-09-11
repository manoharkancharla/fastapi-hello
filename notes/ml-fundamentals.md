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

---

# Day 11 — TTFT vs input length: is the model gap prefill or overhead?

Question from Day 10: at ~25 input tokens sonnet-4-5's TTFT was ~2x haiku-4-5's, but prefill
compute on 25 tokens is negligible — so what is the gap made of? Method: sweep input length
over 4 orders of magnitude on both models, `max_tokens=16` (TTFT is measured before a second
token exists, so output length is pure cost and pure noise), 3 runs per point,
`cache_read_input_tokens=0` verified on every call.

| actual input tok | haiku TTFT mean | sonnet TTFT mean |
|------------------|-----------------|------------------|
| 15               | 620 ms          | 862 ms           |
| 2,014            | 614 ms          | 1,182 ms         |
| 23,082           | 763 ms          | 1,430 ms         |
| 116,776          | 1,637 ms        | 2,314 ms         |

Least squares, TTFT = intercept + slope x input_tokens:

| model | intercept | slope | R2 |
|-------|-----------|-------|-----|
| haiku-4-5  | 594 ms  | 8.88 us/tok  | 0.77 |
| sonnet-4-5 | 1058 ms | 10.97 us/tok | 0.91 |
| ratio      | **1.78x** | **1.24x**  |      |

- **The model gap is in the intercept, not the slope — question answered.** Over the largest
  segment (23K -> 117K) the slopes are indistinguishable: 9.33 us/tok haiku vs 9.43 us/tok
  sonnet. Per-token input cost is essentially model-independent; fixed per-request overhead
  is what differs by ~1.8x. So the Day 10 doubling at 25 input tokens was setup and
  scheduling, not prefill compute — consistent with prefill on 25 tokens being ~0.2 ms.

- **Caveat, and it may be the more important finding: that shared slope is probably not
  prefill at all — it's probably upload.** 116,776 tokens at 4.15 chars/token is ~485 KB of
  request body. Haiku's slope predicts ~1036 ms above intercept at that size; 485 KB in
  1.036 s is ~3.7 Mbps, an ordinary home upload speed. A slope that is *identical* across
  two different-sized models is exactly what a network-bound measurement looks like — a
  bigger model should cost more FLOPs per token, and this one doesn't. **Test (Day 12):**
  send the same 100K payload twice with `cache_control` set. The second call uploads the
  same bytes but skips prefill compute. TTFT unchanged => upload dominates. TTFT drops =>
  prefill was real. This also means my client-side TTFT is not the server's TTFT, and
  anything I publish about TTFT-vs-input needs that stated.

- **Below ~2,000 input tokens, input length is invisible.** haiku went 620 -> 614 ms from 15
  to 2,014 tokens. The original plan's 25/500/2000 sweep would have produced a flat line and
  no conclusion; the signal only clears the noise floor above ~20K.

- **Two anomalies, recorded not smoothed.** (1) Sonnet's 15 -> 2,014 step is +321 ms =
  160 us/tok, 17x steeper than any later segment, with tight variance at both ends, so not
  noise — unexplained, possibly a small-request fast path. (2) Haiku's three 100K runs were
  2046 / 1843 / 1022 ms; that third value is half the others with cache_read=0, and it is
  what drags haiku's R2 to 0.77.

- **Prefill == fixed overhead at ~67K input tokens (haiku) / ~96K (sonnet)** — the input-side
  analogue of the ~60-output-token crossover from Day 10. Subject to the upload caveat above:
  if the slope is network, these are properties of my connection, not the service.

- Cost: $1.70 for 24 calls ($0.43 haiku, $1.28 sonnet). Estimate said $1.47 — actual ran 16%
  over because the chars/token calibration (done on the first 8 KB) undershot at large sizes,
  so "20,000" landed at 23,082 and "100,000" at 116,776. Calibrate on the whole payload, or
  quote estimates as lower bounds.

---

# Day 12 — one-shot document summarization baseline

First real document through the API: "Attention Is All You Need" (arXiv 1706.03762),
15 pages, 39,510 chars after `pypdf` text extraction. One non-streaming `messages.create`
on haiku-4-5, "summarize in 200 words", `max_tokens=1024`, 3 runs, no `cache_control`
(`cache_read_input_tokens=0` verified on every run). Script: `scratch/pdf_summarize_baseline.py`
(untracked — `scratch/` is gitignored, same as Days 6–11).

| run | input tok | output tok | latency | cost |
|-----|-----------|------------|---------|------|
| 1   | 11,323    | 348        | 5,037 ms | $0.01306 |
| 2   | 11,323    | 375        | 4,964 ms | $0.01320 |
| 3   | 11,323    | 340        | 4,884 ms | $0.01302 |

- **A realistic 15-page document is 11.3K input tokens — half of the smallest point
  (23K) the Day 11 slope was fit on, and inside the band Day 11 called barely visible.**
  The slope predicts 9 us x 11.3K ≈ 100 ms of input-attributable latency out of ~5,000 —
  about 2%. Decode (~350 tok x 11.2 ms ≈ 3.9 s) is ~80%. So the TTFT-vs-input work was
  diagnostic; it does not describe single-document summarization. It starts to matter
  at 60+ pages, or when chunks are concatenated back into one prompt.

- **The lever asymmetry: latency is output-driven, cost is input-driven.** Input is 86.7%
  of cost ($0.0113 of $0.0131) but ~2% of latency. Output is ~13% of cost but ~80% of
  latency. To make this cheaper, cache the document (prompt caching on the 11K prefix).
  To make it faster, shorten the output — the document length is nearly irrelevant.
  Reverse of Day 6 (28 in / 138 out, 96% of cost was output).

- **"200 words" produced 340–375 tokens, not the ~250–300 I predicted, and that is most
  of why latency came in at ~5 s instead of 3–4 s.** Haiku answered in markdown — headers,
  bullets, bold — ~230 words of prose plus formatting. Output-token count, not word count,
  is what you pay for in both time and money; a plain-prose instruction would trim it.
  The remaining ~400 ms of the miss is unattributable without streaming (no TTFT to
  separate from decode) — instrument limit, not a finding.

- **PDF-extracted text is 3.49 chars/token, vs 4.15 on Day 11's synthetic payload.**
  Citations, equations, and hyphen-broken line endings tokenize denser than prose. The
  Day 11 calibration would have over-estimated this document by ~19%. Calibrate on the
  actual corpus, not on generated filler.

- Cold start: run 1 slowest but only by 73 ms over run 2, which emitted 27 more tokens.
  Per output token 14.5 / 13.2 / 14.4 ms — no clean signal at this granularity. Weaker
  evidence than Days 7–9; not enough to retract the discard-run-1 rule, not enough to
  confirm it either.

- Cost: $0.039 for 3 runs. Prediction ($0.012–0.015/run) held because input tokens
  landed in range (predicted 10–13K).

- **Deferred, not resolved:** the upload-vs-prefill test Day 11 scheduled for today
  (`scratch/bytes_vs_tokens.py`, ~$0.60) is still unrun. The open question stands.

---

# Day 13 — prompt caching on the same document: $0.0131 -> $0.0028

Same 15-page paper, same haiku-4-5, same 200-word instruction. The document block now
carries `cache_control: {type: ephemeral}` (5-min TTL); the instruction is a separate
uncached text block. Three calls in one process: call 1 writes the cache, calls 2–3 read it.
Script: `scratch/pdf_summarize_baseline.py --cache` (untracked).

| call | input | cache_write | cache_read | output | latency | cost |
|------|-------|-------------|------------|--------|---------|------|
| 1 (write) | 14 | 11,311 | 0      | 344 | 5,285 ms | $0.01587 |
| 2 (read)  | 14 | 0      | 11,311 | 320 | 4,632 ms | $0.00275 |
| 3 (read)  | 14 | 0      | 11,311 | 332 | 4,583 ms | $0.00281 |

Day 12 uncached, same doc: 11,323 input, ~354 output, ~4,962 ms, $0.0131 per call.

- **Cache hit cuts the call from $0.0131 to $0.0028 — 78.8% off.** Not 90%: reads are
  0.1x on the document ($0.00113), but the ~330 output tokens ($0.00165) don't cache, so
  output is now ~60% of the bill. Input's share of cost flipped from 87% (Day 12) to ~41%.
  The Day 2 FinOps claim is now a measurement on a real document, not an assertion.

- **The first call costs more, not less: $0.0159 vs $0.0131 (+21%)** — the 1.25x write
  premium with nothing to read yet. Break-even is exactly two calls ($0.0186 cached vs
  $0.0262 uncached); three calls are $0.0214 vs $0.0393. Caching is a bet that the same
  prefix comes back within 5 minutes; a document summarized once is 21% worse off.

- **`usage.input_tokens` dropped from 11,323 to 14.** Once caching is on, `input_tokens`
  is only the uncached remainder; the document lives in `cache_creation_input_tokens`
  or `cache_read_input_tokens`. A cost formula that only reads `input_tokens` would
  report the cached calls at ~$0.0017 and the write call at ~$0.0017 too — off by 9x on
  call 1. Four terms, always.

- **Latency: cache reads are not measurably faster at this size.** Reads averaged 4,608 ms
  vs 4,962 ms uncached (Day 12), but emitted 28 fewer output tokens on average — at
  11.2 ms/tok that is ~310 ms of the 354 ms gap. Residual ~40 ms, consistent with the
  Day 11 slope predicting ~100 ms of prefill on 11K tokens, under decode noise. Prediction
  was framed as "call 2 vs call 1" (-653 ms) and that framing was wrong: call 1 is not
  the uncached baseline, it is uncached + write (+ possible cold start; it is the slowest
  of all seven calls on this doc). Compare reads to Day 12, matched on output length.
  Non-streaming, so none of this is decomposable further.

- **Haiku 4.5 minimum cacheable prefix is 4,096 tokens** (Opus 5 / Sonnet 5: 512). This
  doc clears it at 11.3K; a ~5-page PDF would silently not cache — no error,
  `cache_creation_input_tokens: 0`. pdf-summarizer needs to check that field, not assume.

- Cost: $0.0214 for 3 runs. Predictions held on every cost number (call 1 $0.016 -> $0.0159;
  reads $0.0029–0.0031 -> $0.0028, low edge because output ran ~330 not ~350).

- Still deferred: `scratch/bytes_vs_tokens.py` (upload vs prefill). Today's latency result
  is a too-small-to-discriminate version of it — 11K tokens is ~100 ms of slope either way.
