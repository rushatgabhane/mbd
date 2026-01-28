from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, lit, when, lag, to_date
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import split, trim
from pyspark.sql.functions import regexp_extract, to_date
spark = SparkSession.builder.appName("prj_").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")



#<USERNAME> replace it by your student number, i removed mine
#take all years from 2003 - 2024
path = "/user/s3549976/flight_data/"
filenames = [f"{path}{year}-{month:02d}.csv"
             for year in range(2003, 2025)
             for month in range(1, 13)]



df = spark.read.option("header", True).option("inferSchema", True).csv(filenames)
df.printSchema()

#these are just delay columns
delay_cols = ['CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY',
              'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']
feature_cols = ['DEP_DELAY', 'ARR_DELAY', 'DISTANCE', 'AIR_TIME'] + delay_cols



df_cleaned = df.select(
    *[
        coalesce(
            when(col(c) == '', None).otherwise(col(c)).cast("double"),
            lit(0.0)
        ).alias(c)
        for c in feature_cols
    ],
    coalesce(
        when(col('ARR_TIME') == '', None).otherwise(col('ARR_TIME')).cast("double"),
        lit(None)
    ).alias('ARR_TIME'),
    coalesce(
        when(col('DEP_TIME') == '', None).otherwise(col('DEP_TIME')).cast("double"),
        lit(None)
    ).alias('DEP_TIME'),
    col('CANCELLED'),
    col('DIVERTED'),
    col('TAIL_NUM'),
    col('FL_DATE'),
    col("ORIGIN")
).filter(
    (col('ARR_TIME').isNotNull()) &
    (col('DEP_TIME').isNotNull()) &
    (col('CANCELLED') == 0) &
    (col('DIVERTED') == 0)
)



df_cleaned = df_cleaned.withColumn(
    "FL_DATE_DT",
    to_date(
        regexp_extract(col("FL_DATE"), r"^(\d{1,2}/\d{1,2}/\d{4})", 1),
        "M/d/yyyy"
    )
)
#groups flight by tail_num the order them by date, departure_time
#say last flight arrived 5 minutes late which means the next one takes 30 min to fly 30 -5 25 could have been caused by soemthing else 
tail_window = Window.partitionBy("TAIL_NUM") \
    .orderBy(col("FL_DATE_DT"), col("DEP_TIME"))


df_cleaned = df_cleaned.withColumn(
    "PREV_ARR_DELAY", lag("ARR_DELAY").over(tail_window) #look at the previous delay of the same aircraft
).withColumn(
    "PREV_FL_DATE", lag("FL_DATE_DT").over(tail_window)  #gets the flight date as well
).withColumn(
    "EXTRA_DELAY",
    when(
        col("FL_DATE_DT") == col("PREV_FL_DATE"), #consider propogation only when on the same day
        F.greatest(
            col("DEP_DELAY") - coalesce(col("PREV_ARR_DELAY"), lit(0)), #subtract departure delay from previous arrival delay. Ensures that if there’s no previous flight, it treats it as 0.
            lit(0)
        )
    ).otherwise(
        F.greatest(col("DEP_DELAY"), lit(0))
    )
)


df_cleaned = df_cleaned.withColumn(
    "DELAY_RECOVERY",
    col("DEP_DELAY") - col("ARR_DELAY")
)


cascade_stats = df_cleaned.select(
    F.mean("EXTRA_DELAY").alias("avg_extra_delay"),
    F.stddev("EXTRA_DELAY").alias("std_extra_delay"),
    F.max("EXTRA_DELAY").alias("max_extra_delay"),
    F.min("EXTRA_DELAY").alias("min_extra_delay")
)

cascade_stats.show(truncate=False)

cascade_stats.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv("/user/<USERNAME>/flight_anomaly_results/cascade_stats")


assembler = VectorAssembler(
    inputCols=[
    "DEP_DELAY",
    "ARR_DELAY",
    "EXTRA_DELAY",
    "DISTANCE"
],
    outputCol="features"
)

df_features = assembler.transform(df_cleaned)

k = 6
kmeans = KMeans(
    featuresCol="features",
    predictionCol="cluster",
    k=k,
    seed=42
)

kmeans_model = kmeans.fit(df_features)
df_clustered = kmeans_model.transform(df_features)

impact_stats = df_clustered.groupBy("cluster").agg(
    F.count("*").alias("num_flights"),
    F.sum(when(col("EXTRA_DELAY") > 10, 1).otherwise(0)).alias("num_propagated"),
    F.mean("EXTRA_DELAY").alias("avg_extra_delay"),
    F.sum(col("DEP_DELAY") + col("ARR_DELAY")).alias("total_delay")
)

impact_stats.show(truncate=False)

impact_stats.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv("/user/<USERNAME>/flight_anomaly_results/cluster_stats")




clusters = df_clustered.select("cluster").distinct().rdd.flatMap(lambda x: x).collect()
analysis_features = delay_cols  

for c in clusters:
    print(f"\nCluster {c}")
    cluster_df = df_clustered.filter(col("cluster") == c)
    for f in analysis_features:
        corr_val = cluster_df.stat.corr(f, "EXTRA_DELAY")
        print(f"{f} correlation with EXTRA_DELAY: {corr_val:.3f}")



total_problem_flights = df_cleaned.filter(col("EXTRA_DELAY") > 10).count()
freq_airports = df_cleaned.filter(col("EXTRA_DELAY") > 10) \
    .groupBy("ORIGIN") \
    .agg(
        F.count("*").alias("num_problem_flights")
    ) \
    .withColumn(
        "fraction_problem_flights",
        F.col("num_problem_flights") / F.lit(total_problem_flights)
    ) \
    .orderBy(F.desc("num_problem_flights"))
freq_airports.show(20, truncate=False)


df_yearly = df_cleaned.withColumn("Year", F.year("FL_DATE_DT"))
airport_yearly_delay = df_yearly.groupBy("ORIGIN", "Year") \
    .agg(F.sum("EXTRA_DELAY").alias("total_extra_delay"))

year_window = Window.partitionBy("Year")
airport_yearly_delay = airport_yearly_delay.withColumn(
    "year_total_delay", F.sum("total_extra_delay").over(year_window)
).withColumn(
    "share", F.col("total_extra_delay") / F.col("year_total_delay")
)

rank_window = Window.partitionBy("Year").orderBy(F.desc("total_extra_delay"))
airport_yearly_delay = airport_yearly_delay.withColumn(
    "rank", F.rank().over(rank_window)
)

delay_cols = ['CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']

airport_yearly_causes = df_yearly.groupBy("ORIGIN", "Year") \
    .agg(*[F.sum(c).alias(c) for c in delay_cols], F.sum("EXTRA_DELAY").alias("total_extra_delay"))

for c in delay_cols:
    airport_yearly_causes = airport_yearly_causes.withColumn(
        f"{c}_share", F.col(c) / F.col("total_extra_delay")
    )

airport_analysis = airport_yearly_causes.join(
    airport_yearly_delay.select("ORIGIN", "Year", "share", "rank"),
    ["ORIGIN", "Year"]
).orderBy("Year", "rank")

airport_analysis_filtered = airport_analysis.filter(F.col("Year").isin([2005, 2010, 2015, 2020, 2025]))

airport_analysis_filtered.show(50, truncate=False)

airport_analysis_filtered.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv("/user/<USERNAME>/flight_anomaly_results/airport_cascade_causes_all")
