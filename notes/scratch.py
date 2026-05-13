import collections

# --- Chapter 2: List comprehensions ---

logs = [
    {"endpoint": "/health", "status": 200, "ms": 12},
    {"endpoint": "/joke",   "status": 200, "ms": 340},
    {"endpoint": "/joke",   "status": 502, "ms": 5001},
    {"endpoint": "/health", "status": 200, "ms": 8},
    {"endpoint": "/joke",   "status": 200, "ms": 150},
]

# Exercise 1: (latency_ms, endpoint) for requests slower than 100ms
slow = [(log["ms"], log["endpoint"]) for log in logs if log["ms"] > 100]
print("Slow requests:", slow)

# Exercise 2: flatten [[1,2],[3,4],[5,6]] in one comprehension
nested = [[1, 2], [3, 4], [5, 6]]
flat = [n for row in nested for n in row]
print("Flattened:", flat)

# Exercise 3: map status codes to "ok" or "error"
codes = [200, 404, 500, 200, 503]
labels = ["ok" if code == 200 else "error" for code in codes]
print("Labels:", labels)

# --- Chapter 3: Counter, defaultdict, dict comprehension ---

counter = collections.Counter(log["endpoint"] for log in logs)
print("\nEndpoint counts:", counter)

by_endpoint: dict[str, list[int]] = collections.defaultdict(list)
for log in logs:
    by_endpoint[log["endpoint"]].append(log["ms"])
print("Latencies by endpoint:", dict(by_endpoint))

avg_latency = {
    endpoint: sum(times) / len(times)
    for endpoint, times in by_endpoint.items()
}
print("Avg latency:", avg_latency)
