# mbd

1. `pip install -r requirements.txt`
2. Create a venv - `python3 -m venv venv`
3. Activate a venv - `source venv/bin/activate`
3. Start notebook `jupyter lab`
4. Or, run a script using python.


### How to access the flight data on HDFS?

The flight data is stored in `/user/s3549976/flight_data/`. 
```
1987-10.csv
1987-11.csv
...

2025-09.csv
```

So you should be able to access it using `df = spark.read.csv("/user/s3549976/flight_data/", header=True)`
