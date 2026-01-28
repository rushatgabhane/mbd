"""
Step 7: Large-Scale Route Divergence & Network Diameter Analysis

1. FASTEST vs RELIABLE DIVERGENCE
   - For all reachable airport pairs, compare fastest vs most reliable paths
   - Find pairs with biggest tradeoffs

2. NETWORK DIAMETER
   - Find maximum hops needed between any two airports
   - Identify hardest-to-reach pairs
"""

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

print(f"\n{'#'*70}")
print(f" LARGE-SCALE ROUTE DIVERGENCE & NETWORK DIAMETER")
print(f"{'#'*70}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1] Loading route data...")

edges_df = spark.read.parquet("/user/s3549976/direct_routes")
edges_df = edges_df.filter(col("flight_count") >= 10)
edges_df.cache()

total_routes = edges_df.count()
print(f"    Routes: {total_routes:,}")

# Collect to driver for graph algorithms
print("\n[2] Building in-memory graph...")
edges_local = edges_df.collect()

# Build adjacency structures
graph = {}  # node -> [neighbors]
edge_data = {}  # (src, dst) -> {weight data}
all_airports = set()

# Invalid airport codes to exclude
INVALID_AIRPORTS = {"USA"}

for row in edges_local:
    src, dst = row["src"], row["dst"]
    # Skip invalid airports
    if src in INVALID_AIRPORTS or dst in INVALID_AIRPORTS:
        continue
    all_airports.add(src)
    all_airports.add(dst)
    if src not in graph:
        graph[src] = []
    graph[src].append(dst)
    edge_data[(src, dst)] = {
        "flight_time": row["avg_flight_time"] or 999999,
        "reliability": row["reliability_score"] or 999999,
        "flight_count": row["flight_count"],
        "carriers": row["carriers"]
    }

all_airports = list(all_airports)
num_airports = len(all_airports)
print(f"    Airports: {num_airports}")
print(f"    Edges: {len(edge_data):,}")


# =============================================================================
# DIJKSTRA IMPLEMENTATION
# =============================================================================
def dijkstra_all(graph, edge_data, source, weight_key):
    """Dijkstra from source to ALL reachable nodes."""
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
            weight = edge_data.get((u, v), {}).get(weight_key, INF)
            if weight is None:
                weight = INF
            new_dist = d + weight
            if new_dist < dist.get(v, INF):
                dist[v] = new_dist
                pred[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist, pred


def bfs_all(graph, source):
    """BFS from source - returns hop distances to all reachable nodes."""
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
    """Reconstruct path from predecessor dict."""
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


# =============================================================================
# ANALYSIS 1: NETWORK DIAMETER (BFS from all nodes)
# =============================================================================
print("\n" + "="*70)
print(" ANALYSIS 1: NETWORK DIAMETER")
print(" Finding maximum hops between any airport pair")
print("="*70)

start_time = time.time()

# BFS from every airport
all_pairs_hops = {}  # (src, dst) -> hops
max_hops = 0
max_hops_pairs = []
unreachable_pairs = []

print(f"\n[*] Running BFS from all {num_airports} airports...")

for i, src in enumerate(all_airports):
    if (i + 1) % 50 == 0:
        print(f"    Progress: {i+1}/{num_airports} airports processed")

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
print(f"\n    Completed in {elapsed:.1f} seconds")

# Results
total_pairs = num_airports * (num_airports - 1)
reachable_pairs = len(all_pairs_hops)

print(f"\n  NETWORK DIAMETER RESULTS")
print(f"  {'─'*40}")
print(f"  Total airport pairs: {total_pairs:,}")
print(f"  Reachable pairs: {reachable_pairs:,} ({100*reachable_pairs/total_pairs:.1f}%)")
print(f"  Unreachable pairs: {len(unreachable_pairs):,}")
print(f"\n  NETWORK DIAMETER: {max_hops} hops")
print(f"  Pairs requiring {max_hops} hops: {len(max_hops_pairs)}")

# Show some max-hop pairs
print(f"\n  Sample pairs requiring {max_hops} hops:")
for src, dst in max_hops_pairs[:10]:
    print(f"    {src} -> {dst}")

# Hop distribution
hop_counts = {}
for hops in all_pairs_hops.values():
    hop_counts[hops] = hop_counts.get(hops, 0) + 1

print(f"\n  HOP DISTRIBUTION:")
print(f"  {'Hops':<6} {'Pairs':<12} {'Percentage':<10}")
print(f"  {'─'*6} {'─'*12} {'─'*10}")
for h in sorted(hop_counts.keys()):
    pct = 100 * hop_counts[h] / reachable_pairs
    print(f"  {h:<6} {hop_counts[h]:<12,} {pct:.1f}%")

# Find pairs needing 4+ hops
high_hop_pairs = [(k, v) for k, v in all_pairs_hops.items() if v >= 4]
print(f"\n  Pairs requiring 4+ hops: {len(high_hop_pairs):,}")
if high_hop_pairs:
    print(f"\n  Sample 4+ hop pairs:")
    for (src, dst), hops in sorted(high_hop_pairs, key=lambda x: -x[1])[:20]:
        print(f"    {src} -> {dst}: {hops} hops")


# =============================================================================
# ANALYSIS 2: FASTEST vs RELIABLE DIVERGENCE
# =============================================================================
print("\n" + "="*70)
print(" ANALYSIS 2: FASTEST vs RELIABLE DIVERGENCE")
print(" Comparing optimal paths by different criteria")
print("="*70)

start_time = time.time()

# Run Dijkstra from all airports for both criteria
print(f"\n[*] Computing shortest paths for all airports...")
print(f"    (This computes fastest AND most reliable from each airport)")

fastest_from = {}  # src -> {dst: (dist, pred)}
reliable_from = {}  # src -> {dst: (dist, pred)}

for i, src in enumerate(all_airports):
    if (i + 1) % 50 == 0:
        print(f"    Progress: {i+1}/{num_airports} airports processed")

    # Fastest (by flight time)
    dist_fast, pred_fast = dijkstra_all(graph, edge_data, src, "flight_time")
    fastest_from[src] = (dist_fast, pred_fast)

    # Most reliable
    dist_rel, pred_rel = dijkstra_all(graph, edge_data, src, "reliability")
    reliable_from[src] = (dist_rel, pred_rel)

elapsed = time.time() - start_time
print(f"\n    Completed in {elapsed:.1f} seconds")

# Compare paths for all pairs
print(f"\n[*] Analyzing divergence between fastest and most reliable paths...")

divergent_pairs = []  # [(src, dst, fast_path, rel_path, fast_time, rel_time, fast_score, rel_score)]

for src in all_airports:
    dist_fast, pred_fast = fastest_from[src]
    dist_rel, pred_rel = reliable_from[src]

    for dst in all_airports:
        if dst == src:
            continue
        if dst not in dist_fast or dst not in dist_rel:
            continue

        # Get paths
        path_fast = reconstruct_path(pred_fast, src, dst)
        path_rel = reconstruct_path(pred_rel, src, dst)

        if not path_fast or not path_rel:
            continue

        # Check if paths differ
        if path_fast != path_rel:
            # Calculate metrics for both paths
            fast_time = dist_fast[dst]

            # Calculate reliability score for fastest path
            fast_rel_score = 0
            for j in range(len(path_fast) - 1):
                edge = edge_data.get((path_fast[j], path_fast[j+1]), {})
                fast_rel_score += edge.get("reliability", 0)

            rel_score = dist_rel[dst]

            # Calculate flight time for reliable path
            rel_time = 0
            for j in range(len(path_rel) - 1):
                edge = edge_data.get((path_rel[j], path_rel[j+1]), {})
                rel_time += edge.get("flight_time", 0)

            # Time penalty for choosing reliable over fast
            time_penalty = rel_time - fast_time
            # Reliability gain for choosing reliable over fast
            reliability_gain = fast_rel_score - rel_score

            divergent_pairs.append({
                "src": src,
                "dst": dst,
                "path_fast": path_fast,
                "path_rel": path_rel,
                "fast_time": fast_time,
                "rel_time": rel_time,
                "fast_rel_score": fast_rel_score,
                "rel_score": rel_score,
                "time_penalty": time_penalty,
                "reliability_gain": reliability_gain,
                "hops_fast": len(path_fast) - 1,
                "hops_rel": len(path_rel) - 1
            })

print(f"\n  DIVERGENCE RESULTS")
print(f"  {'─'*40}")
print(f"  Total reachable pairs: {reachable_pairs:,}")
print(f"  Pairs with SAME path: {reachable_pairs - len(divergent_pairs):,}")
print(f"  Pairs with DIFFERENT paths: {len(divergent_pairs):,} ({100*len(divergent_pairs)/reachable_pairs:.1f}%)")

# Sort by reliability gain (biggest tradeoff)
divergent_pairs.sort(key=lambda x: -x["reliability_gain"])

print(f"\n  TOP 20 PAIRS WITH BIGGEST RELIABILITY GAIN")
print(f"  (Where choosing 'reliable' over 'fast' helps most)")
print(f"\n  {'Route':<15} {'Fast Path':<30} {'Reliable Path':<30} {'Time+':>8} {'Rel Gain':>10}")
print(f"  {'─'*15} {'─'*30} {'─'*30} {'─'*8} {'─'*10}")

for pair in divergent_pairs[:20]:
    route = f"{pair['src']}->{pair['dst']}"
    fast_p = "->".join(pair['path_fast'])
    rel_p = "->".join(pair['path_rel'])
    if len(fast_p) > 28:
        fast_p = fast_p[:25] + "..."
    if len(rel_p) > 28:
        rel_p = rel_p[:25] + "..."
    print(f"  {route:<15} {fast_p:<30} {rel_p:<30} {pair['time_penalty']:>+7.0f}m {pair['reliability_gain']:>10.0f}")

# Show some interesting detailed examples
print(f"\n  DETAILED EXAMPLES OF DIVERGENT ROUTES")
print(f"  {'─'*60}")

for pair in divergent_pairs[:5]:
    print(f"\n  {pair['src']} -> {pair['dst']}")
    print(f"  ")
    print(f"    FASTEST PATH ({pair['hops_fast']} hops, {pair['fast_time']:.0f} min):")
    print(f"      Route: {' -> '.join(pair['path_fast'])}")
    print(f"      Reliability score: {pair['fast_rel_score']:.0f}")
    print(f"  ")
    print(f"    MOST RELIABLE PATH ({pair['hops_rel']} hops, {pair['rel_time']:.0f} min):")
    print(f"      Route: {' -> '.join(pair['path_rel'])}")
    print(f"      Reliability score: {pair['rel_score']:.0f}")
    print(f"  ")
    print(f"    TRADEOFF: +{pair['time_penalty']:.0f} min for {pair['reliability_gain']:.0f} reliability improvement")

# Statistics on divergence
if divergent_pairs:
    avg_time_penalty = sum(p["time_penalty"] for p in divergent_pairs) / len(divergent_pairs)
    avg_rel_gain = sum(p["reliability_gain"] for p in divergent_pairs) / len(divergent_pairs)
    max_time_penalty = max(p["time_penalty"] for p in divergent_pairs)
    max_rel_gain = max(p["reliability_gain"] for p in divergent_pairs)

    print(f"\n  DIVERGENCE STATISTICS")
    print(f"  {'─'*40}")
    print(f"  Avg time penalty for reliable path: {avg_time_penalty:+.1f} min")
    print(f"  Avg reliability gain: {avg_rel_gain:.1f} points")
    print(f"  Max time penalty: {max_time_penalty:.0f} min")
    print(f"  Max reliability gain: {max_rel_gain:.0f} points")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "#"*70)
print(" SUMMARY")
print("#"*70)

print(f"""
  NETWORK DIAMETER
  ────────────────
  • Maximum hops between any pair: {max_hops}
  • Pairs at max distance: {len(max_hops_pairs)}
  • Pairs requiring 4+ hops: {len(high_hop_pairs):,}

  ROUTE DIVERGENCE
  ────────────────
  • Pairs where fastest ≠ most reliable: {len(divergent_pairs):,} ({100*len(divergent_pairs)/reachable_pairs:.1f}%)
  • Avg time cost for reliability: {avg_time_penalty:+.1f} min
  • Avg reliability improvement: {avg_rel_gain:.1f} points

  KEY INSIGHT
  ───────────
  For {100*len(divergent_pairs)/reachable_pairs:.1f}% of routes, choosing the "most reliable"
  path over the "fastest" path involves a tradeoff.
  On average, travelers gain {avg_rel_gain:.0f} reliability points
  at the cost of {abs(avg_time_penalty):.0f} extra minutes.
""")

edges_df.unpersist()
spark.stop()
