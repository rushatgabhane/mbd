from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, lit, when, avg, stddev
from pyspark.ml.feature import VectorAssembler, StandardScaler
from synapse.ml.isolationforest import IsolationForest
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("prj_").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")



path = "/user/s3549976/flight_data/"
df = spark.read.option("header", True).option("inferSchema", True).csv(f"{path}*.csv")


#all the delay columns
delay_cols   = ['CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']
#the rest of the columns
feature_cols = ['DEP_DELAY', 'ARR_DELAY', 'DISTANCE', 'AIR_TIME'] + delay_cols


#some cleaning if empty => None else keep the value
df_cleaned = df.select(
    *[coalesce(when(col(c) == '', None).otherwise(col(c)).cast("double"), lit(0.0)).alias(c) for c in feature_cols],
    coalesce(when(col('ARR_TIME') == '', None).otherwise(col('ARR_TIME')).cast("double"), lit(None)).alias('ARR_TIME'),
    coalesce(when(col('DEP_TIME') == '', None).otherwise(col('DEP_TIME')).cast("double"), lit(None)).alias('DEP_TIME'),
    col('CANCELLED'),
    col('DIVERTED')
)


#just keeping completed flights for now and arrival and departive are not null
df_cleaned = df_cleaned.filter(
    (col('ARR_TIME').isNotNull()) &
    (col('DEP_TIME').isNotNull()) &
    (col('CANCELLED') == 0) &
    (col('DIVERTED') == 0)
)


#added a new column known as delay recovery, dep_delay - arr_delay
df_cleaned = df_cleaned.withColumn('DELAY_RECOVERY', col('DEP_DELAY') - col('ARR_DELAY'))
#added it to the columns
feature_cols += ['DELAY_RECOVERY']


#convert it into vector column so it can be used by isolation forest
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_features = assembler.transform(df_cleaned)


#scaled them 
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=False, withStd=True)
scaler_model = scaler.fit(df_features)
df_scaled = scaler_model.transform(df_features)

train_sample, predict_data = df_scaled.randomSplit([0.3, 0.7], seed=42)


iforest = IsolationForest(
    featuresCol="scaled_features",
    predictionCol="final_prediction",
    scoreCol="anomaly_score",
    contamination=0.05,
    numEstimators=50
)
model = iforest.fit(train_sample)
df_anomalies = model.transform(predict_data)


normal_flights = df_anomalies.filter(col("final_prediction") == 0)
anomalies      = df_anomalies.filter(col("final_prediction") == 1)


def stats_calc(df, label):
    stats = df.select(
        *([F.mean(i).alias(f"{i}_mean") for i in feature_cols] +
          [F.stddev(i).alias(f"{i}_std") for i in feature_cols])
    )
    print(f"Stat for {label}:")
    stats.show(truncate=False)

stats_calc(normal_flights, "Normal Flights")
stats_calc(anomalies, "Anomalous Flights")
