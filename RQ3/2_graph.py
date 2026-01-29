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

origin = sys.argv[1] if len(sys.argv) > 1 else "ACV"
destination = sys.argv[2] if len(sys.argv) > 2 else "OTH"

print(f"\nRouting {origin} -> {destination}")

print("Loading routes from HDFS...")
edges_df = spark.read.parquet("/user/s3549976/direct_routes")
edges_df = edges_df.filter(col("flight_count") >= 10)
edges_df = edges_df.filter(~col("src").isin(["USA"]) & ~col("dst").isin(["USA"]))
edges_df.cache()

edge_count = edges_df.count()
airport_count = edges_df.select("src").union(edges_df.select(col("dst").alias("src"))).distinct().count()
print(f"{airport_count} airports, {edge_count:,} routes")

print("Building in-memory graph...")
edges_local = edges_df.collect()

graph = {}
edge_data = {}

for row in edges_local:
    src, dst = row["src"], row["dst"]
    graph.setdefault(src, []).append(dst)
    edge_data[(src, dst)] = row.asDict()


def bfs_spark(edges_df, source, target, max_hops=6):
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
            print(f"  Found target at hop {hop}")
            break

        frontier = new_nodes.select("node")
        print(f"  Hop {hop}: {new_count} new nodes")

    return visited


def dijkstra(graph, edge_data, source, target, weight_key):
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
            w = edge.get(weight_key, INF)
            if w is None:
                w = INF
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))

    if target not in dist or dist[target] == INF:
        return None, None

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = pred.get(node)
    path.reverse()
    return path, dist[target]


def reconstruct_bfs_path(visited_df, source, target):
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
    if not path or len(path) < 2:
        return

    total_time = 0
    for i in range(len(path) - 1):
        edge = edge_data.get((path[i], path[i+1]), {})
        ft = edge.get("avg_flight_time") or 0
        total_time += ft

        print(f"  Leg {i+1}: {path[i]} -> {path[i+1]}")
        print(f"    Carriers: {edge.get('carriers', 'N/A')}")
        print(f"    Flights in history: {edge.get('flight_count', 0):,}")
        print(f"    Days available: {edge.get('days_with_service', 0):,} ({edge.get('availability_pct', 0)}%)")
        if show_reliability:
            print(f"    Avg flight time: {ft:.0f} min | p90 delay: {edge.get('p90_delay', 0):.0f} min | std: {edge.get('std_delay', 0):.1f}")
        else:
            print(f"    Avg flight time: {ft:.0f} min")

    print(f"\n  Route: {' -> '.join(path)}")
    print(f"  Total flight time: {total_time:.0f} min ({total_time/60:.1f} hrs)")


print("\n--- Criterion 1: Fewest hops (BFS via Spark) ---")
visited_hops = bfs_spark(edges_df, origin, destination, max_hops=6)
path_hops, hops_result = reconstruct_bfs_path(visited_hops, origin, destination)

if hops_result is not None:
    hops_result = int(hops_result)
    print(f"Result: {hops_result} hops")
    print_path(path_hops, edge_data)
else:
    print(f"No path found from {origin} to {destination}")


print(f"\n--- Criterion 2: Fastest (Dijkstra, weight=avg_flight_time) ---")
path_time, time_result = dijkstra(graph, edge_data, origin, destination, "avg_flight_time")

if time_result is not None:
    print(f"Result: {time_result:.0f} min ({time_result/60:.1f} hrs)")
    print_path(path_time, edge_data)
else:
    time_result = None
    print("No path found")


print(f"\n--- Criterion 3: Most reliable (Dijkstra, weight=reliability_score) ---")
path_rel, rel_result = dijkstra(graph, edge_data, origin, destination, "reliability_score")

if rel_result is not None:
    print(f"Result: reliability score = {rel_result:.0f}")
    print_path(path_rel, edge_data, show_reliability=True)
else:
    rel_result = None
    print("No path found")


print(f"\n--- Summary: {origin} -> {destination} ---")
print(f"  Fewest hops:   {f'{hops_result} hops' if hops_result else 'No path'}")
print(f"  Fastest:       {f'{time_result:.0f} min ({time_result/60:.1f} hrs)' if time_result else 'No path'}")
print(f"  Most reliable: {f'{rel_result:.0f} score' if rel_result else 'No path'}")

edges_df.unpersist()
spark.stop()
