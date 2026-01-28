"""
Step 6: Large-Scale Route Analysis

Analyzes patterns across ALL routes in the network.
Computes network-wide statistics and insights.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, desc, asc
import heapq

spark = (SparkSession.builder
    .appName("flight_route_analysis")
    .config("spark.local.dir", "/home/s3549976/spark_tmp")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print(f"\n{'#'*70}")
print(f" LARGE-SCALE ROUTE ANALYSIS")
print(f" Analyzing patterns across ALL proven connections")
print(f"{'#'*70}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1] Loading route data...")

routes = spark.read.parquet("/user/s3549976/direct_routes")
routes = routes.filter(col("flight_count") >= 10)  # Filter rare routes
routes.cache()

total_routes = routes.count()
print(f"    Total routes: {total_routes:,}")

# =============================================================================
# NETWORK STATISTICS
# =============================================================================
print("\n" + "="*70)
print(" NETWORK STATISTICS")
print("="*70)

# Airport connectivity
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

print(f"\n  Airports: {num_airports}")
print(f"  Routes: {total_routes:,}")
print(f"  Avg connections per airport: {avg_degree:.1f}")

# =============================================================================
# TOP HUBS (MOST CONNECTED)
# =============================================================================
print("\n" + "="*70)
print(" TOP 15 HUB AIRPORTS (by connections)")
print("="*70)

airports.orderBy(desc("total_degree")).select(
    "airport", "out_degree", "in_degree", "total_degree", "total_flights"
).show(15, truncate=False)

# =============================================================================
# BUSIEST ROUTES
# =============================================================================
print("\n" + "="*70)
print(" TOP 20 BUSIEST ROUTES (by flight count)")
print("="*70)

routes.orderBy(desc("flight_count")).select(
    "src", "dst", "flight_count", "days_with_service", "carriers", "avg_flight_time"
).show(20, truncate=False)

# =============================================================================
# MOST RELIABLE ROUTES (LOW DELAY VARIANCE)
# =============================================================================
print("\n" + "="*70)
print(" TOP 20 MOST RELIABLE ROUTES")
print(" (high flight count + low p90 delay + low std)")
print("="*70)

reliable = routes.filter(col("flight_count") >= 1000).orderBy(
    asc("reliability_score")
).select(
    "src", "dst", "flight_count", "avg_flight_time",
    F.round("avg_delay", 1).alias("avg_delay"),
    F.round("p90_delay", 1).alias("p90_delay"),
    F.round("std_delay", 1).alias("std_delay"),
    F.round("reliability_score", 1).alias("rel_score")
)
reliable.show(20, truncate=False)

# =============================================================================
# LEAST RELIABLE ROUTES (HIGH DELAYS)
# =============================================================================
print("\n" + "="*70)
print(" TOP 20 LEAST RELIABLE ROUTES")
print(" (high flight count + high delays)")
print("="*70)

unreliable = routes.filter(col("flight_count") >= 1000).orderBy(
    desc("reliability_score")
).select(
    "src", "dst", "flight_count", "avg_flight_time",
    F.round("avg_delay", 1).alias("avg_delay"),
    F.round("p90_delay", 1).alias("p90_delay"),
    F.round("std_delay", 1).alias("std_delay"),
    F.round("reliability_score", 1).alias("rel_score")
)
unreliable.show(20, truncate=False)

# =============================================================================
# CARRIER ANALYSIS
# =============================================================================
print("\n" + "="*70)
print(" CARRIER COVERAGE")
print("="*70)

# Extract individual carriers and count routes
from pyspark.sql.functions import explode, split

carrier_routes = routes.select(
    explode(split(col("carriers"), ",")).alias("carrier"),
    "flight_count"
)
carrier_stats = carrier_routes.groupBy("carrier").agg(
    F.count("*").alias("num_routes"),
    F.sum("flight_count").alias("total_flights")
).orderBy(desc("total_flights"))

print("\n  Top carriers by total flights:")
carrier_stats.show(20, truncate=False)

# =============================================================================
# ROUTE LENGTH DISTRIBUTION
# =============================================================================
print("\n" + "="*70)
print(" FLIGHT TIME DISTRIBUTION")
print("="*70)

time_stats = routes.agg(
    F.min("avg_flight_time").alias("min"),
    F.avg("avg_flight_time").alias("avg"),
    F.max("avg_flight_time").alias("max"),
    F.percentile_approx("avg_flight_time", 0.5).alias("median"),
    F.percentile_approx("avg_flight_time", 0.9).alias("p90")
).first()

print(f"\n  Min flight time: {time_stats['min']:.0f} min")
print(f"  Avg flight time: {time_stats['avg']:.0f} min")
print(f"  Median flight time: {time_stats['median']:.0f} min")
print(f"  90th percentile: {time_stats['p90']:.0f} min")
print(f"  Max flight time: {time_stats['max']:.0f} min")

# =============================================================================
# DELAY ANALYSIS
# =============================================================================
print("\n" + "="*70)
print(" DELAY DISTRIBUTION ACROSS ROUTES")
print("="*70)

delay_stats = routes.agg(
    F.avg("avg_delay").alias("network_avg_delay"),
    F.avg("p90_delay").alias("network_avg_p90"),
    F.avg("std_delay").alias("network_avg_std"),
    F.percentile_approx("avg_delay", 0.9).alias("worst_10pct_avg_delay")
).first()

print(f"\n  Network average delay: {delay_stats['network_avg_delay']:.1f} min")
print(f"  Network average p90 delay: {delay_stats['network_avg_p90']:.1f} min")
print(f"  Network average delay std: {delay_stats['network_avg_std']:.1f} min")
print(f"  Worst 10% routes avg delay: {delay_stats['worst_10pct_avg_delay']:.1f} min")

# =============================================================================
# AVAILABILITY ANALYSIS
# =============================================================================
print("\n" + "="*70)
print(" ROUTE AVAILABILITY DISTRIBUTION")
print("="*70)

avail_stats = routes.agg(
    F.avg("availability_pct").alias("avg_availability"),
    F.percentile_approx("availability_pct", 0.5).alias("median_availability"),
    F.sum(F.when(col("availability_pct") >= 90, 1).otherwise(0)).alias("routes_90pct_plus"),
    F.sum(F.when(col("availability_pct") >= 50, 1).otherwise(0)).alias("routes_50pct_plus"),
    F.sum(F.when(col("availability_pct") < 10, 1).otherwise(0)).alias("routes_under_10pct")
).first()

print(f"\n  Average availability: {avail_stats['avg_availability']:.1f}%")
print(f"  Median availability: {avail_stats['median_availability']:.1f}%")
print(f"  Routes with 90%+ availability: {avail_stats['routes_90pct_plus']:,}")
print(f"  Routes with 50%+ availability: {avail_stats['routes_50pct_plus']:,}")
print(f"  Routes with <10% availability: {avail_stats['routes_under_10pct']:,}")

# =============================================================================
# SAMPLE MULTI-HOP ANALYSIS
# =============================================================================
print("\n" + "="*70)
print(" MULTI-HOP REACHABILITY SAMPLE")
print(" (from major hubs)")
print("="*70)

# Collect graph for BFS analysis
edges_local = routes.select("src", "dst").collect()
graph = {}
for row in edges_local:
    if row["src"] not in graph:
        graph[row["src"]] = []
    graph[row["src"]].append(row["dst"])

def bfs_reachability(graph, source, max_hops=4):
    """Count how many airports reachable at each hop level."""
    visited = {source}
    frontier = {source}
    reach_by_hop = {0: 1}

    for hop in range(1, max_hops + 1):
        new_frontier = set()
        for node in frontier:
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_frontier.add(neighbor)
        reach_by_hop[hop] = len(new_frontier)
        frontier = new_frontier
        if not frontier:
            break

    return reach_by_hop, len(visited) - 1  # -1 to exclude source

# Analyze reachability from top hubs
top_hubs = ["ATL", "ORD", "DFW", "DEN", "LAX"]
print(f"\n  Reachability from major hubs (airports reachable at each hop):\n")
print(f"  {'Hub':<6} {'1-hop':<8} {'2-hop':<8} {'3-hop':<8} {'Total':<8}")
print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for hub in top_hubs:
    if hub in graph:
        reach, total = bfs_reachability(graph, hub)
        h1 = reach.get(1, 0)
        h2 = reach.get(2, 0)
        h3 = reach.get(3, 0)
        print(f"  {hub:<6} {h1:<8} {h2:<8} {h3:<8} {total:<8}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "#"*70)
print(" ANALYSIS SUMMARY")
print("#"*70)

total_flights = routes.agg(F.sum("flight_count")).first()[0]

print(f"""
  NETWORK SCALE
  ─────────────
  • Airports: {num_airports}
  • Direct routes: {total_routes:,}
  • Total flights analyzed: {total_flights:,}
  • Avg connections/airport: {avg_degree:.1f}

  RELIABILITY INSIGHTS
  ────────────────────
  • Network avg delay: {delay_stats['network_avg_delay']:.1f} min
  • Network avg p90 delay: {delay_stats['network_avg_p90']:.1f} min
  • Routes with 90%+ availability: {avail_stats['routes_90pct_plus']:,}

  DATA SOURCE
  ───────────
  • Historical US flight data (1988-2025)
  • Filtered to routes with 10+ flights
""")

routes.unpersist()
spark.stop()
