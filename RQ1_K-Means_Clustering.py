#project
#pyspark --conf spark.dynamicAllocation.maxExecutors=5
#spark-submit --conf spark.dynamicAllocation.maxExecutors=5 kmeans_final.py
from pyspark.sql.functions import avg, stddev, col, expr

from pyspark.sql.functions import col
from pyspark.sql.functions import col, when
from pyspark.sql import SparkSession

from pyspark.sql import functions

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.appName('').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


hdfs_path = "/user/s3549976/flight_data/20[0-2][3-5]-*.csv"

df = spark.read.option("header", True).option("inferSchema", True).csv(hdfs_path)

print("Total rows:", df.count())
df.printSchema()

base_cols = ["YEAR", "ORIGIN", "OP_UNIQUE_CARRIER", "DEP_DELAY", "ARR_DELAY", "TAXI_OUT", "TAXI_IN", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "LATE_AIRCRAFT_DELAY", "SECURITY_DELAY", "CANCELLED", "DIVERTED"]

df_small = df.select(base_cols)

df_small = df_small.filter((col("CANCELLED") == 0) &(col("DIVERTED") == 0) &(col("DEP_DELAY") > 10))

df_small = df_small.filter(col("ARR_DELAY") >= -15)

df_small = df_small.filter((col("DEP_DELAY") <= 360) &(col("ARR_DELAY") <= 360))


required_cols = ["DEP_DELAY", "ARR_DELAY"]

df_small = df_small.dropna(subset=required_cols)

print("Rows after filtering:", df_small.count())

df_small = df_small.withColumn("DELAY_CHANGE",col("ARR_DELAY") - col("DEP_DELAY"))

assembler = VectorAssembler(inputCols=["DEP_DELAY", "ARR_DELAY", "DELAY_CHANGE"], outputCol="FEATURES")

vectors = assembler.transform(df_small)

vectors.select("FEATURES").show(5, truncate=False)


scaler = StandardScaler(inputCol="FEATURES",outputCol="FEATURES_SCALED",withMean=True,withStd=True)

scaler_model = scaler.fit(vectors)
df_scaled = scaler_model.transform(vectors)

kmeans = KMeans(k=3, seed=42, featuresCol="FEATURES_SCALED", predictionCol="cluster")

kmeans_model = kmeans.fit(df_scaled)
df_clustered = kmeans_model.transform(df_scaled)

cluster_summary = (
    df_clustered
    .groupBy("cluster")
    .agg(
        functions.mean("DEP_DELAY").alias("DEP_DELAY"),
        functions.mean("ARR_DELAY").alias("ARR_DELAY"),
        functions.mean("DELAY_CHANGE").alias("DELAY_CHANGE")
    )
)

cluster_summary_threshold = cluster_summary.withColumn(
    "threshold",
    col("DEP_DELAY") * 0.10   # 10% of mean departure delay 
)

cluster_summary_labeled = cluster_summary_threshold.withColumn(
    "recovery_status",
    when(col("DELAY_CHANGE") <= -col("threshold"), "Recovery Achieved")
    .when(col("DELAY_CHANGE") >=  col("threshold"), "Delay Worsened")
    .otherwise("Not Achieved")
)

cluster_summary_labeled.select(
    "cluster",
    "DEP_DELAY",
    "DELAY_CHANGE",
    "threshold",
    "recovery_status"
).orderBy("cluster").show(truncate=False)


df_labeled_auto = (
    df_clustered
    .join(
        cluster_summary_labeled.select("cluster", "recovery_status"),
        on="cluster",
        how="left"
    )
)

#export csv for the delay recovery across all years in all airports for all flights(literally all flights the absolute number)

(
    df_labeled_auto
    .groupBy("YEAR", "recovery_status")
    .count()
    .orderBy("YEAR", "recovery_status")
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("exports/recovery_regimes")
)

df_all_delayed = df_labeled_auto.filter(col("ARR_DELAY") >= 15)

delay_causes = [
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "SECURITY_DELAY",
    "TAXI_IN",
    "TAXI_OUT"
]

df_all_filled = df_all_delayed.fillna({c: 0 for c in delay_causes})


agg_all = (
    df_all_filled
    .groupBy("recovery_status")
    .agg(*[functions.mean(c).alias(c) for c in delay_causes])
    .orderBy("recovery_status")
)

agg_all.show(truncate=False)

total_delay = sum([col(c) for c in delay_causes])

agg_all_pct = agg_all.withColumn("total_delay", total_delay)

for c in delay_causes:
    agg_all_pct = agg_all_pct.withColumn(
        f"{c}_pct", col(c) / col("total_delay") * 100
    )

agg_all_pct.select(
    "recovery_status",
    *[f"{c}_pct" for c in delay_causes]
).show(truncate=False)



#export CSV




# agg reasons how much percentage do they contribute 

(
    agg_all_pct
    .select(
        "recovery_status",
        *[f"{c}_pct" for c in delay_causes]
    )
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("exports/agg_all_percent")
)



#1. Comparing 2003 vs 2013 vs 2015 vs 2015 -
# mean increases but median doesnt increase by a lot so middle values arent influencing the mean, a few late flights are pulling the average up
# mean median delay times wrt all years but years are now in the graph to show trajectory

from pyspark.sql.functions import percentile_approx

yearly_summary = (
    df_labeled_auto
    .filter(col("YEAR").isin([2003, 2013, 2015, 2025]))
    .groupBy("YEAR")
    .agg(
        avg("DEP_DELAY").alias("mean_dep"),
        avg("ARR_DELAY").alias("mean_arr"),
        percentile_approx("ARR_DELAY", 0.5).alias("median_arr"),
        expr("avg(case when ARR_DELAY > 60 then 1 else 0 end)").alias("pct_very_late")
    )
)

(
    yearly_summary
    .orderBy("YEAR")
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("exports/yearly_summary")
)

'''
Mean increases = longer delays

Median increases = more frequent delays

% > 60 increases = tail worsening

'''


#2. Cluster interpretation 
# To make clusters interpretable, we summarize average delay per cluster and add labels.
# 10 % of mean departure delay is used to classify clusters into recovery achieved, delay worsened, not achieved 


flight_delay_clusters_labeled = 
(

    df_labeled_auto
    .select(
        "YEAR",
        "ORIGIN",
        "OP_UNIQUE_CARRIER",
        "DEP_DELAY",
        "ARR_DELAY",
        "DELAY_CHANGE",
        "cluster",
        "recovery_status"
    )
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("exports/flight_delay_clusters_labeled")
)



#3. Are delays longer, more frequent, or more uneven
#this measures how much of total yearly delay is caused by the worst 10% of delayed flights in that same year

from pyspark.sql import Window
from pyspark.sql.functions import (
    col, expr,
    sum as spark_sum,
    count
)
w = Window.partitionBy("YEAR").orderBy(col("ARR_DELAY").desc())

# Window for ranking delays within each year (descending)
w_rank = Window.partitionBy("YEAR").orderBy(col("ARR_DELAY").desc())

# Window for total delay per year
w_year = Window.partitionBy("YEAR")

unevenness_top10 = (
    df_labeled_auto
    .filter(col("ARR_DELAY") > 0)  # focus on delay mass only
    .withColumn("rank", expr("row_number() over (partition by YEAR order by ARR_DELAY desc)"))
    .withColumn("total_flights", count("*").over(w_year))
    .withColumn("cum_delay", spark_sum("ARR_DELAY").over(w_rank))
    .withColumn("total_delay", spark_sum("ARR_DELAY").over(w_year))
    .filter(col("rank") <= col("total_flights") * 0.1)
    .groupBy("YEAR")
    .agg(
        expr("max(cum_delay / total_delay)").alias("top10_delay_share")
    )
    .orderBy("YEAR")
)

# Windows for Gini computation
w_gini_ordered = Window.partitionBy("YEAR").orderBy("ARR_DELAY")
w_gini_year = Window.partitionBy("YEAR")

gini = (
    df_labeled_auto
    .filter(col("ARR_DELAY") > 0)
    .withColumn("rank", expr("row_number() over (partition by YEAR order by ARR_DELAY)"))
    .withColumn("n", count("*").over(w_gini_year))
    .withColumn("sum_delay", spark_sum("ARR_DELAY").over(w_gini_year))
    .withColumn("ranked_delay", col("rank") * col("ARR_DELAY"))
    .withColumn(
        "gini_term",
        (2 * spark_sum("ranked_delay").over(w_gini_year)) /
        (col("n") * col("sum_delay")) - (col("n") + 1) / col("n")
    )
    .groupBy("YEAR")
    .agg(expr("max(gini_term)").alias("gini_delay"))
    .orderBy("YEAR")
)

# join both

unevenness = (
    unevenness_top10
    .join(gini, on="YEAR", how="inner")
    .orderBy("YEAR")
)


(
    unevenness
    .orderBy("YEAR")
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("exports/unevenness")
)

#4. Concentration by airport or airline
# this is how the top 10 airports which caused the total delay across all years

from pyspark.sql.functions import sum as spark_sum, col, expr
from pyspark.sql.window import Window

airport_delay = (
    df_labeled_auto
    .groupBy("YEAR", "ORIGIN")
    .agg(spark_sum("ARR_DELAY").alias("total_delay"))
)

w = Window.partitionBy("YEAR").orderBy(col("total_delay").desc())

airport_ranked = airport_delay.withColumn("rank", expr("row_number() over (partition by YEAR order by total_delay desc)"))

airport_concentration = (
    airport_ranked
    .groupBy("YEAR")
    .agg(
        expr("sum(case when rank <= 10 then total_delay else 0 end) / sum(total_delay)")
        .alias("top10_airport_delay_share")
    )
)

(
    airport_concentration
    .orderBy("YEAR")
    .select(
        "YEAR",
        "top10_airport_delay_share"
    )
    .show(50, truncate=False)
)

(
    airport_concentration
    .orderBy("YEAR")
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("exports/airport_concentration")
)


#4.1 exactly which airports caused

#How much delay does each airport contribute to the total delay in a given year?”

from pyspark.sql.functions import col, sum as spark_sum

years_of_interest = [2003, 2004, 2005, 2013, 2014, 2015, 2023, 2024, 2025]

airport_delay = (
    df_labeled_auto
    .filter(col("ARR_DELAY") > 0)
    .groupBy("YEAR", "ORIGIN")
    .agg(
        spark_sum("ARR_DELAY").alias("airport_delay")
    )
)

yearly_total_delay = (
    df_labeled_auto
    .filter(col("ARR_DELAY") > 0)
    .groupBy("YEAR")
    .agg(
        spark_sum("ARR_DELAY").alias("total_delay")
    )
)

airport_contribution = (
    airport_delay
    .join(yearly_total_delay, on="YEAR", how="inner")
    .withColumn(
        "delay_share",
        col("airport_delay") / col("total_delay")
    )
    .orderBy("YEAR", col("delay_share").desc())
)

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

w_airport = Window.partitionBy("YEAR").orderBy(col("delay_share").desc())

top_airports = (
    airport_contribution
    .withColumn("rank", row_number().over(w_airport))
    .filter(col("rank") <= 10)
)

(
    top_airports
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("exports/airport_contribution")
)

airport_contribution_selected = (
    top_airports
    .filter(col("YEAR").isin(years_of_interest))
    .orderBy("YEAR", "rank")
)

(
    airport_contribution_selected
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("exports/airport_contribution_selected_years")
)

#5. Are delays becoming more unpredictable or extreme?
#CV is coefficient of variability

from pyspark.sql.functions import avg, stddev, col, expr

volatility = (
    df_labeled_auto
    .groupBy("YEAR")
    .agg(
        avg("ARR_DELAY").alias("mean"),
        stddev("ARR_DELAY").alias("std"),
        expr("avg(case when ARR_DELAY > 120 then 1 else 0 end)").alias("pct_2hr"),
        expr("avg(case when ARR_DELAY > 240 then 1 else 0 end)").alias("pct_4hr")
    )
    .withColumn("cv", col("std") / col("mean"))
)


(
    volatility
    .orderBy("YEAR")
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("exports/volatility")
)

