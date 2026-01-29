from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.window import Window
import sys

spark = (SparkSession.builder
    .appName("flight_routing_graph")
    .config("spark.local.dir", "/home/s3549976/spark_tmp")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

origin = sys.argv[1] if len(sys.argv) > 1 else "ACV"
destination = sys.argv[2] if len(sys.argv) > 2 else "OTH"

print(f"\nRouting {origin} -> {destination}")

print("Loading routes...")
edges_df = spark.read.parquet("/user/s3549976/direct_routes")
edges_df = edges_df.filter(col("flight_count") >= 10)
edges_df = edges_df.filter(~col("src").isin(["USA"]) & ~col("dst").isin(["USA"]))
edges_df.cache()

edge_count = edges_df.count()
airport_count = edges_df.select("src").union(edges_df.select(col("dst").alias("src"))).distinct().count()
print(f"{airport_count} airports, {edge_count:,} routes")


def bfs_spark(source, target, max_hops=6):
    schema = StructType([
        StructField("node", StringType(), False),
        StructField("dist", IntegerType(), False),
        StructField("pred", StringType(), True)
    ])
    visited = spark.createDataFrame([(source, 0, None)], schema)
    frontier = spark.createDataFrame([(source,)], ["node"])

    for hop in range(1, max_hops + 1):
        new_nodes = edges_df.alias("e").join(
            frontier.alias("f"), col("e.src") == col("f.node")
        ).join(
            visited.alias("v"), col("e.dst") == col("v.node"), "left_anti"
        ).select(
            col("e.dst").alias("node"),
            lit(hop).alias("dist"),
            col("e.src").alias("pred")
        ).dropDuplicates(["node"])

        cnt = new_nodes.count()
        if cnt == 0:
            break

        visited = visited.union(new_nodes)

        if new_nodes.filter(col("node") == target).count() > 0:
            print(f"  Found target at hop {hop}")
            break

        frontier = new_nodes.select("node")
        print(f"  Hop {hop}: {cnt} new nodes")

    return visited


def shortest_path_spark(source, weight_col, max_iter=8):
    schema = StructType([
        StructField("node", StringType(), False),
        StructField("dist", DoubleType(), False),
        StructField("pred", StringType(), True)
    ])
    distances = spark.createDataFrame([(source, 0.0, None)], schema)

    weight_edges = edges_df.select(
        "src", "dst",
        F.coalesce(col(weight_col), lit(999999.0)).alias("weight")
    )

    for _ in range(max_iter):
        relaxed = distances.alias("d").join(
            weight_edges.alias("e"), col("d.node") == col("e.src")
        ).select(
            col("e.dst").alias("node"),
            (col("d.dist") + col("e.weight")).alias("dist"),
            col("e.src").alias("pred")
        )

        combined = distances.union(relaxed)
        w = Window.partitionBy("node").orderBy("dist")
        new_distances = combined.withColumn("rn", F.row_number().over(w)) \
                                .filter(col("rn") == 1).drop("rn")
        new_distances.cache()
        new_distances.count()
        distances.unpersist()
        distances = new_distances

    return distances


def reconstruct_path(dist_df, source, target):
    pred_rows = dist_df.collect()
    pred_map = {r["node"]: r["pred"] for r in pred_rows}

    if target not in pred_map:
        return None, None

    target_dist = None
    for r in pred_rows:
        if r["node"] == target:
            target_dist = r["dist"]
            break

    if target_dist is None:
        return None, None

    path = [target]
    cur = target
    while cur != source:
        p = pred_map.get(cur)
        if not p:
            return None, None
        path.append(p)
        cur = p
    path.reverse()
    return path, target_dist


def print_path(path, show_reliability=False):
    if not path or len(path) < 2:
        return

    leg_cond = None
    for i in range(len(path) - 1):
        c = (col("src") == path[i]) & (col("dst") == path[i+1])
        leg_cond = c if leg_cond is None else (leg_cond | c)

    legs = edges_df.filter(leg_cond).collect()
    leg_map = {(r["src"], r["dst"]): r for r in legs}

    total_time = 0
    for i in range(len(path) - 1):
        edge = leg_map.get((path[i], path[i+1]))
        if not edge:
            continue
        ft = edge["avg_flight_time"] or 0
        total_time += ft

        print(f"  Leg {i+1}: {path[i]} -> {path[i+1]}")
        print(f"    Carriers: {edge['carriers'] or 'N/A'}")
        print(f"    Flights in history: {edge['flight_count']:,}")
        print(f"    Days available: {edge['days_with_service']:,} ({edge['availability_pct']}%)")
        if show_reliability:
            print(f"    Avg flight time: {ft:.0f} min | p90 delay: {edge['p90_delay']:.0f} min | std: {edge['std_delay']:.1f}")
        else:
            print(f"    Avg flight time: {ft:.0f} min")

    print(f"\n  Route: {' -> '.join(path)}")
    print(f"  Total flight time: {total_time:.0f} min ({total_time/60:.1f} hrs)")


print("\n--- Criterion 1: Fewest hops (BFS) ---")
visited = bfs_spark(origin, destination)
path_hops, hops_result = reconstruct_path(visited, origin, destination)

if hops_result is not None:
    hops_result = int(hops_result)
    print(f"Result: {hops_result} hops")
    print_path(path_hops)
else:
    print(f"No path found from {origin} to {destination}")


print(f"\n--- Criterion 2: Fastest (weight=avg_flight_time) ---")
dist_time = shortest_path_spark(origin, "avg_flight_time")
path_time, time_result = reconstruct_path(dist_time, origin, destination)

if time_result is not None:
    print(f"Result: {time_result:.0f} min ({time_result/60:.1f} hrs)")
    print_path(path_time)
else:
    time_result = None
    print("No path found")


print(f"\n--- Criterion 3: Most reliable (weight=reliability_score) ---")
dist_rel = shortest_path_spark(origin, "reliability_score")
path_rel, rel_result = reconstruct_path(dist_rel, origin, destination)

if rel_result is not None:
    print(f"Result: reliability score = {rel_result:.0f}")
    print_path(path_rel, show_reliability=True)
else:
    rel_result = None
    print("No path found")


print(f"\n--- Summary: {origin} -> {destination} ---")
print(f"  Fewest hops:   {f'{hops_result} hops' if hops_result else 'No path'}")
print(f"  Fastest:       {f'{time_result:.0f} min ({time_result/60:.1f} hrs)' if time_result else 'No path'}")
print(f"  Most reliable: {f'{rel_result:.0f} score' if rel_result else 'No path'}")

edges_df.unpersist()
spark.stop()
