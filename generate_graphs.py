import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os

# Load & filter
df = pd.read_csv('data/intermittent-renewables-production-france.csv')
df = df[df['Source'] == 'Solar'].copy()

df['datetime'] = pd.to_datetime(df['Date and Hour'], utc=True)
df['hour']     = df['datetime'].dt.hour
df['month']    = df['datetime'].dt.month
df = df.sort_values('datetime').reset_index(drop=True)
df['lag1']        = df['Production'].shift(1)
df['rolling_mean']= df['Production'].rolling(6).mean()
df['rolling_std'] = df['Production'].rolling(6).std()
df = df.dropna().reset_index(drop=True)

# Cluster
cluster_features = ['Production', 'rolling_mean', 'rolling_std']
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[cluster_features])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster)
cluster_means = df.groupby('cluster')['Production'].mean().sort_values()
label_map = {
    cluster_means.index[0]: 'Overcast',
    cluster_means.index[1]: 'Intermittent',
    cluster_means.index[2]: 'Sunny'
}
df['weather_type'] = df['cluster'].map(label_map)

os.makedirs('static', exist_ok=True)

# 1. Hourly average production per cluster
hourly = df.groupby(['weather_type', 'hour'])['Production'].mean().reset_index()
hourly_data = {}
for wtype in ['Sunny', 'Intermittent', 'Overcast']:
    subset = hourly[hourly['weather_type'] == wtype]
    hourly_data[wtype] = {
        'hours': subset['hour'].tolist(),
        'production': [round(x, 2) for x in subset['Production'].tolist()]
    }
with open('static/hourly_data.json', 'w') as f:
    json.dump(hourly_data, f)

# 2. Monthly average production
monthly = df.groupby('month')['Production'].mean().reset_index()
monthly_data = {
    'months': monthly['month'].tolist(),
    'production': [round(x, 2) for x in monthly['Production'].tolist()]
}
with open('static/monthly_data.json', 'w') as f:
    json.dump(monthly_data, f)

# 3. Cluster distribution
dist = df['weather_type'].value_counts()
dist_data = {
    'labels': dist.index.tolist(),
    'counts': dist.values.tolist()
}
with open('static/cluster_dist.json', 'w') as f:
    json.dump(dist_data, f)

print("✅ Graph data generated!")
print("Hourly data:", hourly_data.keys())
print("Monthly months:", monthly_data['months'])
print("Cluster distribution:", dist_data)