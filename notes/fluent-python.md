# Fluent Python — Chapters 1–3 Notes

## Two patterns from Chapter 2 I didn't know

**1. Inline if/else inside a comprehension (transforms vs filters)**

Two different uses of `if` inside a comprehension:

```python
# Filter — skips items that don't match
slow = [log for log in logs if log["ms"] > 100]

# Transform — every item gets mapped, just differently
labels = ["ok" if code == 200 else "error" for code in codes]
```

The filter `if` goes at the end. The transform `if/else` goes at the front, wrapping the output expression. Easy to mix up, different behavior.

**2. Nested loops in one comprehension**

```python
flat = [n for row in [[1,2],[3,4],[5,6]] for n in row]
# → [1, 2, 3, 4, 5, 6]
```

Read left to right: "give me n, for each row, for each n in that row." Order matches how you'd write nested for loops — outer loop first.

---

## One Chapter 1 idea that connects to FastAPI/Pydantic

`FrenchDeck` gets iteration, slicing, `in`, and `random.choice` for free just by implementing `__len__` and `__getitem__`. No interface declared, nothing inherited.

Pydantic's `BaseModel` works the same way — it implements `__init__`, `__repr__`, and the descriptor protocol under the hood. When you write:

```python
class JokeResponse(BaseModel):
    setup: str
    punchline: str
```

You get validation, JSON serialization, and OpenAPI schema generation for free. You didn't ask for those — Pydantic's dunder methods just make them available. Same mental model as FrenchDeck.

---

## One Chapter 3 pattern I'll use this week

`defaultdict(list)` for grouping, then a dict comprehension for aggregation:

```python
by_endpoint = defaultdict(list)
for log in logs:
    by_endpoint[log["endpoint"]].append(log["ms"])

avg_latency = {
    endpoint: sum(times) / len(times)
    for endpoint, times in by_endpoint.items()
}
```

This is the shape of pandas `groupby` before you know pandas — and it's what I'll reach for when analyzing LLM call latencies, token counts, or retry rates in Phase 2.

---

## Week 2 Day 2: pandas as the pattern you already know

### groupby().agg() is just defaultdict(list) + dict comprehension

The manual version from Chapter 3 takes six lines:

```python
by_endpoint = defaultdict(list)
for log in logs:
    by_endpoint[log["endpoint"]].append(log["ms"])

avg_latency = {
    endpoint: sum(times) / len(times)
    for endpoint, times in by_endpoint.items()
}
```

The pandas version is one line:

```python
by_endpoint = df.groupby("endpoint").agg(mean_ms=("latency_ms", "mean"))
```

Same operation: collect values by key, then reduce each group. `groupby("endpoint")` does the `defaultdict(list)` accumulation. `agg(mean_ms=("latency_ms", "mean"))` does the comprehension. The named-aggregate syntax `mean_ms=("col", func)` is just a cleaner way to say "apply this reducer to this column and name the result."

The difference is that pandas does all of it in C, vectorized, without a Python loop touching each row.

### df[df["latency_ms"] > 200] — boolean indexing via __getitem__

This works in two steps you can read left to right:

```python
mask = df["latency_ms"] > 200   # Step 1: a Series of True/False, one per row
slow = df[mask]                  # Step 2: df.__getitem__(mask) returns only True rows
```

`df["latency_ms"] > 200` produces a boolean Series because `Series.__gt__` broadcasts the comparison across every element and returns a new Series. Then `df[some_series_of_bools]` hits `DataFrame.__getitem__`, which checks the dtype — if it's boolean, it filters rows instead of selecting columns.

This is exactly the Chapter 1 dunder point: `df[...]` isn't magic syntax, it's `__getitem__`. Pass it a string → column lookup. Pass it a boolean Series → row filter. Same method, different behavior based on what you hand it.

### One thing that surprised me

Aggregations chain directly onto the grouped result, and you can mix built-in strings (`"mean"`, `"count"`) with arbitrary lambdas in the same `agg()` call:

```python
df.groupby("endpoint").agg(
    mean_ms=("latency_ms", "mean"),
    p95_ms=("latency_ms", lambda s: s.quantile(0.95)),
    error_rate=("status_code", lambda s: (s >= 500).mean()),
)
```

`"mean"` dispatches to an optimized C path. The lambda gets the whole group as a Series and can return anything scalar. Both live in the same call. I expected to need separate passes or method chaining to mix those two things.
