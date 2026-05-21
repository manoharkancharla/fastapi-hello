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
