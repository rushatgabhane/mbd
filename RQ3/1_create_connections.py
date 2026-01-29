from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, when, floor

spark = (SparkSession.builder
    .appName("flight_connections_builder")
    .config("spark.local.dir", "/home/s3549976/spark_tmp")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print("Loading flight data...")
df = spark.read.csv("/user/s3549976/flight_data/", header=True, inferSchema=True)

total_raw = df.count()
print(f"Raw flights: {total_raw:,}")

flights = df.filter(
    (col("CANCELLED") == 0) &
    (col("DIVERTED") == 0) &
    col("CRS_DEP_TIME").isNotNull() &
    col("CRS_ARR_TIME").isNotNull() &
    col("ARR_DELAY").isNotNull()
)

flights = flights.withColumn(
    "dep_minutes",
    (floor(col("CRS_DEP_TIME") / 100) * 60 + col("CRS_DEP_TIME") % 100).cast("int")
).withColumn(
    "arr_minutes",
    (floor(col("CRS_ARR_TIME") / 100) * 60 + col("CRS_ARR_TIME") % 100).cast("int")
)

flights = flights.withColumn(
    "arr_minutes",
    when(col("arr_minutes") < col("dep_minutes"), col("arr_minutes") + 24 * 60)
    .otherwise(col("arr_minutes"))
)

flights = flights.select(
    col("FL_DATE").alias("date"),
    col("ORIGIN").alias("src"),
    col("DEST").alias("dst"),
    col("OP_CARRIER").alias("carrier"),
    col("CRS_DEP_TIME").alias("dep_time"),
    col("CRS_ARR_TIME").alias("arr_time"),
    col("dep_minutes"),
    col("arr_minutes"),
    col("CRS_ELAPSED_TIME").alias("flight_time"),
    col("ARR_DELAY").alias("arr_delay")
)

flights.cache()
total_flights = flights.count()
print(f"Completed flights: {total_flights:,}")

date_stats = flights.agg(
    F.min("date").alias("min_date"),
    F.max("date").alias("max_date"),
    F.countDistinct("date").alias("total_days")
).first()
print(f"Date range: {date_stats['min_date']} to {date_stats['max_date']} ({date_stats['total_days']} days)")

print("Computing direct route statistics...")

direct_routes = flights.groupBy("src", "dst").agg(
    F.count("*").alias("flight_count"),
    F.countDistinct("date").alias("days_with_service"),
    F.countDistinct("carrier").alias("num_carriers"),
    F.concat_ws(",", F.collect_set("carrier")).alias("carriers"),
    F.avg("flight_time").alias("avg_flight_time"),
    F.avg("arr_delay").alias("avg_delay"),
    F.stddev("arr_delay").alias("std_delay"),
    F.percentile_approx("arr_delay", 0.9).alias("p90_delay")
).withColumn(
    "std_delay", F.coalesce(col("std_delay"), lit(0.0))
).withColumn(
    "reliability_score",
    col("avg_flight_time") + col("avg_delay") + col("std_delay")
).withColumn(
    "availability_pct",
    F.round(col("days_with_service") / lit(date_stats['total_days']) * 100, 1)
)

direct_count = direct_routes.count()
print(f"Unique direct routes: {direct_count:,}")

print("Saving direct routes...")
direct_routes.write.mode("overwrite").parquet("/user/s3549976/direct_routes")

print(f"\nDone: {total_flights:,} flights, {date_stats['total_days']} days, {direct_count:,} routes")
print("Saved to /user/s3549976/direct_routes")

print("\n--- Top routes by frequency ---")
direct_routes.orderBy(col("flight_count").desc()).select(
    "src", "dst", "flight_count", "days_with_service",
    "avg_flight_time", "reliability_score", "carriers"
).show(20, truncate=False)

flights.unpersist()
spark.stop()
