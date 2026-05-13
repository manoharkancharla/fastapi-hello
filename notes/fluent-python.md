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
