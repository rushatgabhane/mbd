from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, desc, asc, explode, split

spark = (SparkSession.builder
    .appName("flight_route_analysis")
    .config("spark.local.dir", "/home/s3549976/spark_tmp")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

routes = spark.read.parquet("/user/s3549976/direct_routes")
routes = routes.filter(col("flight_count") >= 10)
routes.cache()

total_routes = routes.count()
print(f"Loaded {total_routes:,} routes")

airports_out = routes.groupBy("src").agg(
    F.count("*").alias("out_degree"),
    F.sum("flight_count").alias("total_departures")
)
airports_in = routes.groupBy("dst").agg(
    F.count("*").alias("in_degree"),
    F.sum("flight_count").alias("total_arrivals")
)
airports = airports_out.alias("o").join(
    airports_in.alias("i"),
    col("o.src") == col("i.dst"),
    "outer"
).select(
    F.coalesce(col("o.src"), col("i.dst")).alias("airport"),
    F.coalesce(col("out_degree"), lit(0)).alias("out_degree"),
    F.coalesce(col("in_degree"), lit(0)).alias("in_degree"),
    F.coalesce(col("total_departures"), lit(0)).alias("total_departures"),
    F.coalesce(col("total_arrivals"), lit(0)).alias("total_arrivals")
).withColumn(
    "total_degree", col("out_degree") + col("in_degree")
).withColumn(
    "total_flights", col("total_departures") + col("total_arrivals")
)

num_airports = airports.count()
avg_degree = airports.agg(F.avg("total_degree")).first()[0]
print(f"{num_airports} airports, avg {avg_degree:.1f} connections each")

print("\n--- Top 15 hubs ---")
airports.orderBy(desc("total_degree")).select(
    "airport", "out_degree", "in_degree", "total_degree", "total_flights"
).show(15, truncate=False)

print("--- Top 20 busiest routes ---")
routes.orderBy(desc("flight_count")).select(
    "src", "dst", "flight_count", "days_with_service", "carriers", "avg_flight_time"
).show(20, truncate=False)

busy = routes.filter(col("flight_count") >= 1000)

print("--- Most reliable routes (1000+ flights) ---")
reliable = busy.orderBy(asc("reliability_score")).select(
    "src", "dst", "flight_count", "avg_flight_time",
    F.round("avg_delay", 1).alias("avg_delay"),
    F.round("p90_delay", 1).alias("p90_delay"),
    F.round("std_delay", 1).alias("std_delay"),
    F.round("reliability_score", 1).alias("rel_score")
)
reliable.show(20, truncate=False)

print("--- Least reliable routes (1000+ flights) ---")
unreliable = busy.orderBy(desc("reliability_score")).select(
    "src", "dst", "flight_count", "avg_flight_time",
    F.round("avg_delay", 1).alias("avg_delay"),
    F.round("p90_delay", 1).alias("p90_delay"),
    F.round("std_delay", 1).alias("std_delay"),
    F.round("reliability_score", 1).alias("rel_score")
)
unreliable.show(20, truncate=False)

print("--- Carrier coverage ---")
carrier_routes = routes.select(
    explode(split(col("carriers"), ",")).alias("carrier"),
    "flight_count"
)
carrier_stats = carrier_routes.groupBy("carrier").agg(
    F.count("*").alias("num_routes"),
    F.sum("flight_count").alias("total_flights")
).orderBy(desc("total_flights"))
carrier_stats.show(20, truncate=False)

print("--- Flight time distribution ---")
time_stats = routes.agg(
    F.min("avg_flight_time").alias("min"),
    F.avg("avg_flight_time").alias("avg"),
    F.max("avg_flight_time").alias("max"),
    F.percentile_approx("avg_flight_time", 0.5).alias("median"),
    F.percentile_approx("avg_flight_time", 0.9).alias("p90")
).first()
print(f"  min={time_stats['min']:.0f}, avg={time_stats['avg']:.0f}, "
      f"median={time_stats['median']:.0f}, p90={time_stats['p90']:.0f}, "
      f"max={time_stats['max']:.0f} min")

print("\n--- Delay distribution ---")
delay_stats = routes.agg(
    F.avg("avg_delay").alias("network_avg_delay"),
    F.avg("p90_delay").alias("network_avg_p90"),
    F.avg("std_delay").alias("network_avg_std"),
    F.percentile_approx("avg_delay", 0.9).alias("worst_10pct_avg_delay")
).first()
print(f"  avg delay: {delay_stats['network_avg_delay']:.1f} min")
print(f"  avg p90 delay: {delay_stats['network_avg_p90']:.1f} min")
print(f"  avg delay std: {delay_stats['network_avg_std']:.1f} min")
print(f"  worst 10% routes avg delay: {delay_stats['worst_10pct_avg_delay']:.1f} min")

print("\n--- Availability ---")
avail_stats = routes.agg(
    F.avg("availability_pct").alias("avg_avail"),
    F.percentile_approx("availability_pct", 0.5).alias("median_avail"),
    F.sum(F.when(col("availability_pct") >= 90, 1).otherwise(0)).alias("above_90"),
    F.sum(F.when(col("availability_pct") >= 50, 1).otherwise(0)).alias("above_50"),
    F.sum(F.when(col("availability_pct") < 10, 1).otherwise(0)).alias("below_10")
).first()
print(f"  avg={avail_stats['avg_avail']:.1f}%, median={avail_stats['median_avail']:.1f}%")
print(f"  90%+ avail: {avail_stats['above_90']:,} routes")
print(f"  50%+ avail: {avail_stats['above_50']:,} routes")
print(f"  <10% avail: {avail_stats['below_10']:,} routes")

print("\n--- Multi-hop reachability from major hubs ---")
edges_local = routes.select("src", "dst").collect()
graph = {}
for row in edges_local:
    graph.setdefault(row["src"], []).append(row["dst"])

def bfs_reachability(graph, source, max_hops=4):
    visited = {source}
    frontier = {source}
    reach_by_hop = {}
    for hop in range(1, max_hops + 1):
        next_frontier = set()
        for node in frontier:
            for nb in graph.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        reach_by_hop[hop] = len(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return reach_by_hop, len(visited) - 1

print(f"  {'Hub':<6} {'1-hop':<8} {'2-hop':<8} {'3-hop':<8} {'Total':<8}")
for hub in ["ATL", "ORD", "DFW", "DEN", "LAX"]:
    if hub in graph:
        reach, total = bfs_reachability(graph, hub)
        print(f"  {hub:<6} {reach.get(1,0):<8} {reach.get(2,0):<8} {reach.get(3,0):<8} {total:<8}")

total_flights = routes.agg(F.sum("flight_count")).first()[0]
print(f"\nSummary: {num_airports} airports, {total_routes:,} routes, {total_flights:,} total flights")
print(f"  avg delay {delay_stats['network_avg_delay']:.1f} min, "
      f"avg p90 delay {delay_stats['network_avg_p90']:.1f} min, "
      f"{avail_stats['above_90']:,} routes with 90%+ availability")

routes.unpersist()
spark.stop()
