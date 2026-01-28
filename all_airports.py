import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file_path = "<PROJECT_ROOT>/flight_anomaly_results/airport_cascade_causes_all/y.csv"
df = pd.read_csv(file_path)

def plot_top_airports_share(df, year, delay_types=None, top_n=10):
    if delay_types is None:
        delay_types = ['CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']
    
    share_cols = [f"{c}_share" for c in delay_types]
    
  
    df_year = df[df['Year'] == year].copy()
    

    airport_sum = df_year.groupby('ORIGIN')[share_cols + ['total_extra_delay']].mean().reset_index()
    
  
    top_airports = airport_sum.nlargest(top_n, 'total_extra_delay').sort_values('total_extra_delay', ascending=False)
    
    # Normalize shares so sum = 1 per airport
    top_airports[share_cols] = top_airports[share_cols].div(top_airports[share_cols].sum(axis=1), axis=0)
    
    # Plot
    plt.figure(figsize=(12, 7))
    bottom = np.zeros(len(top_airports))
    
    for col in share_cols:
        plt.bar(top_airports['ORIGIN'], top_airports[col], bottom=bottom, label=col.replace('_share',''))
        bottom += top_airports[col]
    
    plt.xlabel('Airport')
    plt.ylabel('Fraction of Total Extra Delay')
    plt.title(f'Top {top_n} Airports by Extra Delay Share ({year})')
    plt.xticks(rotation=30, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage:
years = [2005, 2010, 2015, 2020]
for year in years:
    plot_top_airports_share(df, year, top_n=10)
