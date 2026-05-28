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
