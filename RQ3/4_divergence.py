from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
import time

spark = (SparkSession.builder
    .appName("route_divergence_analysis")
    .config("spark.local.dir", "/home/s3549976/spark_tmp")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print("Loading route data...")
edges_df = spark.read.parquet("/user/s3549976/direct_routes")
edges_df = edges_df.filter(col("flight_count") >= 10)
edges_df = edges_df.filter(~col("src").isin(["USA"]) & ~col("dst").isin(["USA"]))
edges_df.cache()

total_routes = edges_df.count()
print(f"Routes: {total_routes:,}")

all_nodes = edges_df.select(col("src").alias("node")).union(
    edges_df.select(col("dst").alias("node"))
).distinct()
num_airports = all_nodes.count()
print(f"{num_airports} airports")

edges = edges_df.select(
    "src", "dst",
    F.coalesce(col("avg_flight_time"), lit(999999.0)).alias("avg_flight_time"),
    F.coalesce(col("reliability_score"), lit(999999.0)).alias("reliability_score")
)


print("\n--- Network diameter (all-pairs BFS) ---")
start = time.time()

bfs_dists = all_nodes.select(
    col("node").alias("source"), col("node"), lit(0).alias("hops")
)
frontier = bfs_dists.select("source", "node")

for hop in range(1, 10):
    new_pairs = frontier.alias("f").join(
        edges.alias("e"), col("f.node") == col("e.src")
    ).select(
        col("f.source"), col("e.dst").alias("node")
    ).distinct().alias("n").join(
        bfs_dists.alias("d"),
        (col("n.source") == col("d.source")) & (col("n.node") == col("d.node")),
        "left_anti"
    ).withColumn("hops", lit(hop))

    cnt = new_pairs.count()
    print(f"  Hop {hop}: {cnt:,} new pairs")
    if cnt == 0:
        break
    bfs_dists = bfs_dists.union(new_pairs)
    frontier = new_pairs.select("source", "node")
    bfs_dists.cache()
    bfs_dists.count()

print(f"  Done in {time.time() - start:.1f}s")

pair_dists = bfs_dists.filter(col("source") != col("node"))
total_pairs = num_airports * (num_airports - 1)
reachable_count = pair_dists.count()
diameter = pair_dists.agg(F.max("hops")).first()[0]

print(f"  Total pairs: {total_pairs:,}")
print(f"  Reachable: {reachable_count:,} ({100*reachable_count/total_pairs:.1f}%)")
print(f"  Unreachable: {total_pairs - reachable_count:,}")
print(f"  Diameter: {diameter} hops")

max_dist_pairs = pair_dists.filter(col("hops") == diameter)
print(f"  Pairs at diameter: {max_dist_pairs.count()}")

print(f"\n  Sample pairs at max distance ({diameter} hops):")
for r in max_dist_pairs.limit(10).collect():
    print(f"    {r['source']} -> {r['node']}")

print(f"\n  Hop distribution:")
hop_dist = pair_dists.groupBy("hops").count().orderBy("hops").collect()
print(f"  {'Hops':<6} {'Pairs':<12} {'%':<10}")
for r in hop_dist:
    pct = 100 * r["count"] / reachable_count
    print(f"  {r['hops']:<6} {r['count']:<12,} {pct:.1f}%")

high_hop = pair_dists.filter(col("hops") >= 4)
high_hop_count = high_hop.count()
print(f"\n  Pairs needing 4+ hops: {high_hop_count:,}")
if high_hop_count > 0:
    for r in high_hop.orderBy(col("hops").desc()).limit(20).collect():
        print(f"    {r['source']} -> {r['node']}: {r['hops']} hops")


print("\n--- Fastest vs reliable divergence (all-pairs shortest paths) ---")
start = time.time()


def all_pairs_shortest(weight_col, cross_col, max_iter=8):
    dists = all_nodes.select(
        col("node").alias("source"), col("node"),
        lit(0.0).alias("dist"),
        lit(0.0).alias("cross_dist"),
        lit(None).cast(StringType()).alias("pred")
    )

    for i in range(max_iter):
        relaxed = dists.alias("d").join(
            edges.alias("e"), col("d.node") == col("e.src")
        ).select(
            col("d.source"),
            col("e.dst").alias("node"),
            (col("d.dist") + col(f"e.{weight_col}")).alias("dist"),
            (col("d.cross_dist") + col(f"e.{cross_col}")).alias("cross_dist"),
            col("e.src").alias("pred")
        )

        combined = dists.union(relaxed)
        w = Window.partitionBy("source", "node").orderBy("dist")
        new_dists = combined.withColumn("rn", F.row_number().over(w)) \
                            .filter(col("rn") == 1).drop("rn")
        new_dists.cache()
        new_dists.count()
        dists.unpersist()
        dists = new_dists
        print(f"    Iter {i+1} done")

    return dists


print("  Computing fastest paths (all-pairs)...")
dist_fast = all_pairs_shortest("avg_flight_time", "reliability_score")

print("  Computing reliable paths (all-pairs)...")
dist_rel = all_pairs_shortest("reliability_score", "avg_flight_time")

print(f"  Done in {time.time() - start:.1f}s")

fast_pairs = dist_fast.filter(col("source") != col("node")).alias("f")
rel_pairs = dist_rel.filter(col("source") != col("node")).alias("r")

comparison = fast_pairs.join(
    rel_pairs,
    (col("f.source") == col("r.source")) & (col("f.node") == col("r.node"))
).select(
    col("f.source").alias("src"),
    col("f.node").alias("dst"),
    col("f.dist").alias("fast_time"),
    col("f.cross_dist").alias("fast_rel_score"),
    col("r.dist").alias("rel_score"),
    col("r.cross_dist").alias("rel_time"),
).withColumn(
    "time_penalty", F.round(col("rel_time") - col("fast_time"), 1)
).withColumn(
    "reliability_gain", F.round(col("fast_rel_score") - col("rel_score"), 1)
)
comparison.cache()

divergent = comparison.filter(F.abs(col("reliability_gain")) > 0.1)
div_count = divergent.count()
total_compared = comparison.count()

print(f"  Same path: {total_compared - div_count:,}")
print(f"  Different paths: {div_count:,} ({100*div_count/total_compared:.1f}%)")

print(f"\n  Top 20 pairs with biggest reliability gain:")
divergent.orderBy(col("reliability_gain").desc()).select(
    "src", "dst",
    F.round("fast_time", 0).alias("fast_time"),
    F.round("rel_time", 0).alias("rel_time"),
    "time_penalty", "reliability_gain"
).show(20, truncate=False)

div_stats = divergent.agg(
    F.avg("time_penalty").alias("avg_tp"),
    F.avg("reliability_gain").alias("avg_rg"),
    F.max("time_penalty").alias("max_tp"),
    F.max("reliability_gain").alias("max_rg")
).first()

print(f"  Stats:")
print(f"    Avg time penalty: {div_stats['avg_tp']:+.1f} min")
print(f"    Avg reliability gain: {div_stats['avg_rg']:.1f}")
print(f"    Max time penalty: {div_stats['max_tp']:.0f} min")
print(f"    Max reliability gain: {div_stats['max_rg']:.0f}")

top5 = divergent.orderBy(col("reliability_gain").desc()).limit(5).collect()

print(f"\n  Detailed examples:")
for pair in top5:
    src, dst = pair["src"], pair["dst"]

    fast_preds = dist_fast.filter(col("source") == src) \
        .select("node", "pred").collect()
    fast_map = {r["node"]: r["pred"] for r in fast_preds}

    rel_preds = dist_rel.filter(col("source") == src) \
        .select("node", "pred").collect()
    rel_map = {r["node"]: r["pred"] for r in rel_preds}

    def get_path(pred_map, source, target):
        path = [target]
        cur = target
        while cur != source:
            p = pred_map.get(cur)
            if not p:
                return None
            path.append(p)
            cur = p
        path.reverse()
        return path

    pf = get_path(fast_map, src, dst)
    pr = get_path(rel_map, src, dst)

    print(f"\n  {src} -> {dst}")
    if pf:
        print(f"    Fastest: {' -> '.join(pf)} ({pair['fast_time']:.0f} min, rel={pair['fast_rel_score']:.0f})")
    if pr:
        print(f"    Reliable: {' -> '.join(pr)} ({pair['rel_time']:.0f} min, rel={pair['rel_score']:.0f})")
    print(f"    Tradeoff: +{pair['time_penalty']:.0f} min for {pair['reliability_gain']:.0f} reliability improvement")

print(f"\nSummary: diameter={diameter} hops, "
      f"{div_count:,} divergent pairs ({100*div_count/total_compared:.1f}%), "
      f"avg tradeoff: {abs(div_stats['avg_tp']):.0f} min for {div_stats['avg_rg']:.0f} reliability")

comparison.unpersist()
edges_df.unpersist()
spark.stop()
