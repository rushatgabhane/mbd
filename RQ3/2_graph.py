"""
Step 5: Graph-Based Path Finding using Proven Connections

Uses Spark to load data, then efficient algorithms:
- Distributed BFS for fewest hops
- In-memory Dijkstra for weighted paths (graph is small: 431 nodes, 12k edges)

Three optimization criteria:
1. FEWEST HOPS - minimize number of flights (BFS)
2. FASTEST - minimize avg_flight_time (Dijkstra)
3. MOST RELIABLE - minimize μ + σ = avg_flight_time + avg_delay + std_delay (Dijkstra)

Edges are PROVEN connections from historical data.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import heapq
import sys

spark = (SparkSession.builder
    .appName("flight_routing_graph")
    .config("spark.local.dir", "/home/s3549976/spark_tmp")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# Args
origin = sys.argv[1] if len(sys.argv) > 1 else "ACV"
destination = sys.argv[2] if len(sys.argv) > 2 else "OTH"

print(f"\n{'#'*70}")
print(f" GRAPH-BASED FLIGHT ROUTING")
print(f" {origin} -> {destination}")
print(f" Algorithm: BFS (hops) + Dijkstra (weighted)")
print(f" Data: PROVEN historical connections")
print(f"{'#'*70}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1] Loading proven connections from HDFS...")

edges_df = spark.read.parquet("/user/s3549976/direct_routes")

# Filter out rare routes (< 10 flights) - likely anomalies
# Also filter out invalid airport codes
edges_df = edges_df.filter(col("flight_count") >= 10)
edges_df = edges_df.filter(~col("src").isin(["USA"]) & ~col("dst").isin(["USA"]))
edges_df.cache()
edge_count = edges_df.count()
print(f"    Direct routes (edges): {edge_count:,} (filtered: flight_count >= 10)")

airport_count = edges_df.select("src").union(edges_df.select(col("dst").alias("src"))).distinct().count()
print(f"    Airports (vertices): {airport_count}")

# Collect edges to driver for efficient graph algorithms
print("\n[2] Building in-memory graph...")
edges_local = edges_df.collect()
print(f"    Collected {len(edges_local)} edges to driver")

# Build adjacency list
graph = {}  # node -> [(neighbor, weight_dict), ...]
edge_data = {}  # (src, dst) -> full edge data

for row in edges_local:
    src, dst = row["src"], row["dst"]
    if src not in graph:
        graph[src] = []
    graph[src].append(dst)
    edge_data[(src, dst)] = row.asDict()

print(f"    Graph has {len(graph)} nodes with outgoing edges")


# =============================================================================
# BFS FOR FEWEST HOPS (DISTRIBUTED SPARK)
# =============================================================================
def bfs_spark(edges_df, source, target, max_hops=6):
    """BFS using Spark - distributed for demonstration."""

    schema = StructType([
        StructField("node", StringType(), False),
        StructField("dist", IntegerType(), False),
        StructField("pred", StringType(), True)
    ])
    visited = spark.createDataFrame([(source, 0, None)], schema)
    frontier = spark.createDataFrame([(source,)], ["node"])

    for hop in range(1, max_hops + 1):
        new_nodes = edges_df.alias("e").join(
            frontier.alias("f"),
            col("e.src") == col("f.node")
        ).join(
            visited.alias("v"),
            col("e.dst") == col("v.node"),
            "left_anti"
        ).select(
            col("e.dst").alias("node"),
            lit(hop).alias("dist"),
            col("e.src").alias("pred")
        ).dropDuplicates(["node"])

        new_count = new_nodes.count()
        if new_count == 0:
            break

        visited = visited.union(new_nodes)

        target_row = new_nodes.filter(col("node") == target).first()
        if target_row:
            print(f"    Found target at hop {hop}")
            break

        frontier = new_nodes.select("node")
        print(f"    Hop {hop}: expanded to {new_count} new nodes")

    return visited


# =============================================================================
# DIJKSTRA FOR WEIGHTED PATHS (IN-MEMORY - EFFICIENT)
# =============================================================================
def dijkstra(graph, edge_data, source, target, weight_key):
    """Standard Dijkstra using heapq - O(E log V)."""
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

        if u == target:
            break

        if u not in graph:
            continue

        for v in graph[u]:
            if v in visited:
                continue
            edge = edge_data.get((u, v), {})
            weight = edge.get(weight_key, INF)
            if weight is None:
                weight = INF
            new_dist = d + weight

            if new_dist < dist.get(v, INF):
                dist[v] = new_dist
                pred[v] = u
                heapq.heappush(heap, (new_dist, v))

    if target not in dist or dist[target] == INF:
        return None, None

    # Reconstruct path
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = pred.get(node)
    path.reverse()

    return path, dist[target]


def reconstruct_bfs_path(visited_df, source, target):
    """Reconstruct path from BFS visited DataFrame."""
    dist_data = {row["node"]: (row["dist"], row["pred"]) for row in visited_df.collect()}

    if target not in dist_data:
        return None, None

    path = [target]
    current = target
    while current != source and dist_data[current][1]:
        current = dist_data[current][1]
        path.insert(0, current)

    if path[0] != source:
        return None, None

    return path, dist_data[target][0]


def print_path(path, edge_data, show_reliability=False):
    """Print path details."""
    if not path or len(path) < 2:
        return

    route_str = path[0]
    total_time = 0

    print(f"\n[*] Path:")
    for i in range(len(path) - 1):
        edge = edge_data.get((path[i], path[i+1]), {})
        ft = edge.get("avg_flight_time") or 0
        total_time += ft
        route_str += f" -> {path[i+1]}"

        print(f"    Leg {i+1}: {path[i]} -> {path[i+1]}")
        print(f"           Carriers: {edge.get('carriers', 'N/A')}")
        print(f"           Flights in history: {edge.get('flight_count', 0):,}")
        print(f"           Days available: {edge.get('days_with_service', 0):,} ({edge.get('availability_pct', 0)}%)")
        if show_reliability:
            print(f"           Avg flight time: {ft:.0f} min | p90 delay: {edge.get('p90_delay', 0):.0f} min | std: {edge.get('std_delay', 0):.1f}")
        else:
            print(f"           Avg flight time: {ft:.0f} min")

    print(f"\n    Route: {route_str}")
    print(f"    Total flight time: {total_time:.0f} min ({total_time/60:.1f} hrs)")


# =============================================================================
# CRITERION 1: FEWEST HOPS (Distributed BFS)
# =============================================================================
print("\n" + "="*70)
print(" CRITERION 1: FEWEST HOPS (Distributed BFS via Spark)")
print("="*70)

print("\n[*] Running distributed BFS...")
visited_hops = bfs_spark(edges_df, origin, destination, max_hops=6)
path_hops, hops_result = reconstruct_bfs_path(visited_hops, origin, destination)

if hops_result is not None:
    hops_result = int(hops_result)
    print(f"\n[*] Result: {origin} -> {destination} = {hops_result} hops")
    print_path(path_hops, edge_data)
else:
    print(f"\n[*] No path found from {origin} to {destination}")


# =============================================================================
# CRITERION 2: FASTEST (Dijkstra on collected graph)
# =============================================================================
print("\n" + "="*70)
print(" CRITERION 2: FASTEST (Dijkstra)")
print(" Weight = avg_flight_time")
print("="*70)

print("\n[*] Running Dijkstra...")
path_time, time_result = dijkstra(graph, edge_data, origin, destination, "avg_flight_time")

if time_result is not None:
    print(f"\n[*] Result: {origin} -> {destination} = {time_result:.0f} min ({time_result/60:.1f} hrs)")
    print_path(path_time, edge_data)
else:
    time_result = None
    print(f"\n[*] No path found")


# =============================================================================
# CRITERION 3: MOST RELIABLE (Dijkstra on collected graph)
# =============================================================================
print("\n" + "="*70)
print(" CRITERION 3: MOST RELIABLE (Dijkstra)")
print(" Weight = avg_flight_time + avg_delay + std_delay (μ + σ)")
print("="*70)

print("\n[*] Running Dijkstra...")
path_rel, rel_result = dijkstra(graph, edge_data, origin, destination, "reliability_score")

if rel_result is not None:
    print(f"\n[*] Result: {origin} -> {destination} = {rel_result:.0f} reliability score")
    print_path(path_rel, edge_data, show_reliability=True)
else:
    rel_result = None
    print(f"\n[*] No path found")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "#"*70)
print(f" SUMMARY: {origin} -> {destination}")
print("#"*70)

print(f"\n  FEWEST HOPS:   {f'{hops_result} hops' if hops_result else 'No path'}")
print(f"  FASTEST:       {f'{time_result:.0f} min ({time_result/60:.1f} hrs)' if time_result else 'No path'}")
print(f"  MOST RELIABLE: {f'{rel_result:.0f} score' if rel_result else 'No path'}")

edges_df.unpersist()
spark.stop()
