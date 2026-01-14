## If you want to execute from scratch
1. Execute the .py file in your head node 

2. Get the exports to your NFS by 

'''

hdfs dfs -get \exports/agg_all_percent/part-*.csv agg_all_percent.csv

hdfs dfs -get \exports/yearly_summary/part-*.csv yearly_summary.csv

hdfs dfs -get \exports/unevenness/part-*.csv unevenness.csv

hdfs dfs -get \exports/airport_concentration/part-*.csv airport_concentration.csv

hdfs dfs -get \exports/airport_contribution/part-*.csv airport_contribution.csv

hdfs dfs -get \exports/airport_contribution_selected_years/part-*.csv airport_contribution_selected_years.csv

hdfs dfs -get \exports/volatility/part-*.csv volatility.csv

hdfs dfs -get \exports/recovery_regimes/part-*.csv recovery_regimes.csv

'''

3. Move the .csv files to your local and execute the .ipynb file to get the graphs.

