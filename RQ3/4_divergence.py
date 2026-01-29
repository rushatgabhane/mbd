from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import heapq
from collections import deque
import time

spark = (SparkSession.builder
    .appName("route_divergence_analysis")
    .config("spark.local.dir", "/home/s3549976/spark_tmp")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print("Loading route data...")
edges_df = spark.read.parquet("/user/s3549976/direct_routes")
edges_df = edges_df.filter(col("flight_count") >= 10)
edges_df.cache()

total_routes = edges_df.count()
print(f"Routes: {total_routes:,}")

print("Building in-memory graph...")
edges_local = edges_df.collect()

graph = {}
edge_data = {}
all_airports = set()
INVALID_AIRPORTS = {"USA"}

for row in edges_local:
    src, dst = row["src"], row["dst"]
    if src in INVALID_AIRPORTS or dst in INVALID_AIRPORTS:
        continue
    all_airports.add(src)
    all_airports.add(dst)
    graph.setdefault(src, []).append(dst)
    edge_data[(src, dst)] = {
        "flight_time": row["avg_flight_time"] or 999999,
        "reliability": row["reliability_score"] or 999999,
        "flight_count": row["flight_count"],
        "carriers": row["carriers"]
    }

all_airports = list(all_airports)
num_airports = len(all_airports)
print(f"{num_airports} airports, {len(edge_data):,} edges")


def dijkstra_all(graph, edge_data, source, weight_key):
    INF = float('inf')
    dist = {source: 0}
    pred = {source: None}
    heap = [(0, source)]
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v in graph.get(u, []):
            if v in visited:
                continue
            w = edge_data.get((u, v), {}).get(weight_key, INF)
            if w is None:
                w = INF
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, pred


def bfs_all(graph, source):
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in graph.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def reconstruct_path(pred, source, target):
    if target not in pred:
        return None
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = pred.get(node)
    path.reverse()
    if path[0] != source:
        return None
    return path


print("\n--- Network diameter (BFS from all nodes) ---")
start_time = time.time()

all_pairs_hops = {}
max_hops = 0
max_hops_pairs = []
unreachable_pairs = []

for i, src in enumerate(all_airports):
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{num_airports} done")

    hop_dist = bfs_all(graph, src)
    for dst in all_airports:
        if dst == src:
            continue
        if dst in hop_dist:
            hops = hop_dist[dst]
            all_pairs_hops[(src, dst)] = hops
            if hops > max_hops:
                max_hops = hops
                max_hops_pairs = [(src, dst)]
            elif hops == max_hops:
                max_hops_pairs.append((src, dst))
        else:
            unreachable_pairs.append((src, dst))

elapsed = time.time() - start_time
print(f"  Done in {elapsed:.1f}s")

total_pairs = num_airports * (num_airports - 1)
reachable_pairs = len(all_pairs_hops)

print(f"  Total pairs: {total_pairs:,}")
print(f"  Reachable: {reachable_pairs:,} ({100*reachable_pairs/total_pairs:.1f}%)")
print(f"  Unreachable: {len(unreachable_pairs):,}")
print(f"  Diameter: {max_hops} hops ({len(max_hops_pairs)} pairs)")

print(f"\n  Sample pairs at max distance ({max_hops} hops):")
for src, dst in max_hops_pairs[:10]:
    print(f"    {src} -> {dst}")

hop_counts = {}
for hops in all_pairs_hops.values():
    hop_counts[hops] = hop_counts.get(hops, 0) + 1

print(f"\n  Hop distribution:")
print(f"  {'Hops':<6} {'Pairs':<12} {'%':<10}")
for h in sorted(hop_counts.keys()):
    pct = 100 * hop_counts[h] / reachable_pairs
    print(f"  {h:<6} {hop_counts[h]:<12,} {pct:.1f}%")

high_hop_pairs = [(k, v) for k, v in all_pairs_hops.items() if v >= 4]
print(f"\n  Pairs needing 4+ hops: {len(high_hop_pairs):,}")
if high_hop_pairs:
    for (src, dst), hops in sorted(high_hop_pairs, key=lambda x: -x[1])[:20]:
        print(f"    {src} -> {dst}: {hops} hops")


print("\n--- Fastest vs reliable divergence ---")
start_time = time.time()

fastest_from = {}
reliable_from = {}

for i, src in enumerate(all_airports):
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{num_airports} done")

    dist_fast, pred_fast = dijkstra_all(graph, edge_data, src, "flight_time")
    fastest_from[src] = (dist_fast, pred_fast)

    dist_rel, pred_rel = dijkstra_all(graph, edge_data, src, "reliability")
    reliable_from[src] = (dist_rel, pred_rel)

elapsed = time.time() - start_time
print(f"  Done in {elapsed:.1f}s")

print("  Comparing paths...")

divergent_pairs = []

for src in all_airports:
    dist_fast, pred_fast = fastest_from[src]
    dist_rel, pred_rel = reliable_from[src]

    for dst in all_airports:
        if dst == src:
            continue
        if dst not in dist_fast or dst not in dist_rel:
            continue

        path_fast = reconstruct_path(pred_fast, src, dst)
        path_rel = reconstruct_path(pred_rel, src, dst)

        if not path_fast or not path_rel:
            continue

        if path_fast != path_rel:
            fast_time = dist_fast[dst]

            fast_rel_score = 0
            for j in range(len(path_fast) - 1):
                edge = edge_data.get((path_fast[j], path_fast[j+1]), {})
                fast_rel_score += edge.get("reliability", 0)

            rel_score = dist_rel[dst]

            rel_time = 0
            for j in range(len(path_rel) - 1):
                edge = edge_data.get((path_rel[j], path_rel[j+1]), {})
                rel_time += edge.get("flight_time", 0)

            time_penalty = rel_time - fast_time
            reliability_gain = fast_rel_score - rel_score

            divergent_pairs.append({
                "src": src, "dst": dst,
                "path_fast": path_fast, "path_rel": path_rel,
                "fast_time": fast_time, "rel_time": rel_time,
                "fast_rel_score": fast_rel_score, "rel_score": rel_score,
                "time_penalty": time_penalty,
                "reliability_gain": reliability_gain,
                "hops_fast": len(path_fast) - 1,
                "hops_rel": len(path_rel) - 1
            })

print(f"  Same path: {reachable_pairs - len(divergent_pairs):,}")
print(f"  Different paths: {len(divergent_pairs):,} ({100*len(divergent_pairs)/reachable_pairs:.1f}%)")

divergent_pairs.sort(key=lambda x: -x["reliability_gain"])

print(f"\n  Top 20 pairs with biggest reliability gain:")
print(f"  {'Route':<15} {'Fast Path':<30} {'Reliable Path':<30} {'Time+':>8} {'Rel Gain':>10}")
for pair in divergent_pairs[:20]:
    route = f"{pair['src']}->{pair['dst']}"
    fast_p = "->".join(pair['path_fast'])
    rel_p = "->".join(pair['path_rel'])
    if len(fast_p) > 28:
        fast_p = fast_p[:25] + "..."
    if len(rel_p) > 28:
        rel_p = rel_p[:25] + "..."
    print(f"  {route:<15} {fast_p:<30} {rel_p:<30} {pair['time_penalty']:>+7.0f}m {pair['reliability_gain']:>10.0f}")

print(f"\n  Detailed examples:")
for pair in divergent_pairs[:5]:
    print(f"\n  {pair['src']} -> {pair['dst']}")
    print(f"    Fastest: {' -> '.join(pair['path_fast'])} ({pair['hops_fast']} hops, {pair['fast_time']:.0f} min, rel={pair['fast_rel_score']:.0f})")
    print(f"    Reliable: {' -> '.join(pair['path_rel'])} ({pair['hops_rel']} hops, {pair['rel_time']:.0f} min, rel={pair['rel_score']:.0f})")
    print(f"    Tradeoff: +{pair['time_penalty']:.0f} min for {pair['reliability_gain']:.0f} reliability improvement")

if divergent_pairs:
    avg_time_penalty = sum(p["time_penalty"] for p in divergent_pairs) / len(divergent_pairs)
    avg_rel_gain = sum(p["reliability_gain"] for p in divergent_pairs) / len(divergent_pairs)
    max_time_penalty = max(p["time_penalty"] for p in divergent_pairs)
    max_rel_gain = max(p["reliability_gain"] for p in divergent_pairs)

    print(f"\n  Stats:")
    print(f"    Avg time penalty: {avg_time_penalty:+.1f} min")
    print(f"    Avg reliability gain: {avg_rel_gain:.1f}")
    print(f"    Max time penalty: {max_time_penalty:.0f} min")
    print(f"    Max reliability gain: {max_rel_gain:.0f}")

    print(f"\nSummary: diameter={max_hops} hops, "
          f"{len(divergent_pairs):,} divergent pairs ({100*len(divergent_pairs)/reachable_pairs:.1f}%), "
          f"avg tradeoff: {abs(avg_time_penalty):.0f} min for {avg_rel_gain:.0f} reliability")

edges_df.unpersist()
spark.stop()
