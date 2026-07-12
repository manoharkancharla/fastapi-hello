# Anthropic CCA Retrospective
_Written Week 3 Day 4, after a 3-week break._

---

## Agentic Architecture & Orchestration (27%)

The heaviest domain and the most scenario-driven (~60% production design, not syntax recall).

The raw agentic loop — `stop_reason == "tool_use"` triggers tool execution, result gets appended to conversation history, loop continues. The trap is parallel tool calls in a single turn: the orchestrator must map results back concurrently and in order. Getting this wrong is the most common failure mode on the exam.

Hub-and-spoke (coordinator-subagent) pattern — coordinator handles decomposition and routing; subagents are task-specific with isolated contexts. The `Task` tool is the formal mechanism for spawning subagents, must be declared in `allowedTools`. Single agents with 30 tools underperform. Over-decomposing into micro-agents kills latency and causes context loss at boundaries — the exam explicitly punishes this.

Deterministic vs. prompt-based compliance — for anything with a hard policy gate, prompt instructions are an anti-pattern (non-zero failure rate). The correct answer is SDK hooks (`PreToolUse`, `PostToolUse`) that physically block downstream calls until prerequisites are met. This distinction is the most important architectural judgment the cert tests.

What felt thin: Agent SDK hooks in practice, and session forking/state recovery.

**Portfolio connection:** Weeks 26-27 (triage agent). Coordinator-subagent pattern and hook-based compliance gates apply directly.

---

## Claude Code Configuration & Workflows (20%)

This domain tests whether you treat Claude Code as a chatbot or a headless automation agent.

`.claude/rules/` vs `CLAUDE.md` — rules are path-scoped and file-type-scoped (e.g., `src/**/*.tsx`), loaded only when Claude Code touches a matching file. The exam trap: a bloated `CLAUDE.md` hitting token limits or creating conflicting rules — the fix is moving specialized constraints into `.claude/rules/` or custom skills.

Planning mode decouples exploration from execution. Claude discovers the codebase, drafts implementation, then runs a deterministic verification check (linter, build, tests). It loops inside planning mode until the check passes. This is the correct pattern for complex, multi-step code tasks.

CI/CD integration: the `-p` flag puts Claude Code in headless, non-interactive mode. Combined with `--output-format json` and `--json-schema`, it produces structured findings that a GitHub Actions pipeline can parse to post inline PR reviews. This is the differentiator most traditional API courses miss entirely.

**Portfolio connection:** Mostly new vs. my plan. Relevant for any Phase 2 automation work and the Week 26 triage agent's CI integration.

---

## Prompt Engineering & Structured Output (20%)

Maps directly to the Pydantic validation boundary pattern I've already built in `fastapi-hello`.

Validation-retry loop: catch structural failures programmatically. If Claude returns malformed JSON, capture the exact `ValidationError` or parser trace, append it to conversation history as a user message, and ask the model to fix only that specific discrepancy. Re-prompting with "fix this" is wrong.

Few-shot examples must be valid JSON objects that adhere strictly to the target schema — loose prose examples in system prompts are penalized. Optional fields must be typed as `null` explicitly, not omitted — omitting breaks downstream deserializers.

**Portfolio connection:** Week 4 (first LLM call) and Weeks 13-14 (RAG evals). The retry loop pattern is directly applicable.

---

## Tool Design & MCP Integration (18%)

MCP is an open-standard, bilateral contract that abstracts how an LLM connects to data and tools — a universal protocol layer so any model can interact with any data source via JSON-RPC.

MCP Server: exposes resources and tools through a standardized interface, contains the business logic.
MCP Client: the application hosting the LLM — discovers available tools, exposes them to Claude's context, handles permission gates, passes intent to the server.

Use MCP for reusable, shared connectors (a database explorer multiple teams use across different Claude tools). Use bespoke tool calling when the action is tightly coupled to a single application's internal state.

Operational concerns the exam surfaced: servers must run with least privilege, sandboxed from mutating root directories. Logging at the transport layer is required — log the exact JSON-RPC payloads to distinguish a tool failure from a model reasoning error.

This domain felt the most forward-looking. MCP as a design pattern is going to be the standard interface layer for LLM-to-infrastructure connections.

**Portfolio connection:** Week 26 (triage agent's tools). Highest-leverage domain for 2026 differentiation.

---

## Context Management & Reliability (15%)

Direct reinforcement of Day 2 KV cache notes, with tactical application added.

Prompt caching requires structural consistency — dynamically modifying the middle of a system prompt (e.g., inserting timestamps) busts the cache and destroys performance. Static prefix, dynamic suffix is the correct layout.

Conversation compaction: when a session fills up, Claude Code purges older tool outputs first, then historical summaries. The manual equivalent at the API level is forking the session (`--fork-session`), writing a state summary message, and starting a fresh context window — preserving critical state metadata while clearing intermediate debugging noise.

**Portfolio connection:** Confirms and extends Day 2 notes. Directly applicable to Week 4+ when I'm building real LLM call loops.

---

## Which weeks does this compress, and which remain full-effort?

The cert compresses Weeks 4 and 26-27 meaningfully. The structured output / validation-retry pattern (Week 4) I can move through faster because I've already implemented the shape of it in `fastapi-hello`. The triage agent weeks (26-27) will go faster because coordinator-subagent architecture and hook-based compliance are no longer abstract. Context management (Week 17+ KV cache work) is reinforced, not compressed — understanding the theory is different from profiling and tuning a real serving stack.

Claude Code CI/CD integration (the `-p` flag, headless mode, structured output pipelines) remains full-effort — the cert covered the concepts but I haven't built any of it. MCP integration also remains full-effort: I can explain the protocol, but I haven't written a server or wired one into an orchestrator. Those are still hands-on weeks. The cert accelerated the mental model; it didn't replace the reps.
