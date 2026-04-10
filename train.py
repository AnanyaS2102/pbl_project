import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── 1. Load & filter solar data ──────────────────────────────────────────────
df = pd.read_csv('data/intermittent-renewables-production-france.csv')
df = df[df['Source'] == 'Solar'].copy()

# ── 2. Parse datetime & extract time features ────────────────────────────────
df['datetime'] = pd.to_datetime(df['Date and Hour'], utc=True)
df['hour']     = df['datetime'].dt.hour
df['month']    = df['datetime'].dt.month
df['dayOfWeek']= df['datetime'].dt.dayofweek

# ── 3. Sort & create lag / rolling features ──────────────────────────────────
df = df.sort_values('datetime').reset_index(drop=True)
df['lag1']        = df['Production'].shift(1)
df['lag24']       = df['Production'].shift(24)
df['rolling_mean']= df['Production'].rolling(6).mean()
df['rolling_std'] = df['Production'].rolling(6).std()
df = df.dropna().reset_index(drop=True)

# ── 4. K-Means clustering (Sunny / Intermittent / Overcast) ──────────────────
cluster_features = ['Production', 'rolling_mean', 'rolling_std']
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster)

# Label clusters by mean production
cluster_means = df.groupby('cluster')['Production'].mean().sort_values()
label_map = {
    cluster_means.index[0]: 'Overcast',
    cluster_means.index[1]: 'Intermittent',
    cluster_means.index[2]: 'Sunny'
}
df['weather_type'] = df['cluster'].map(label_map)
print("Cluster distribution:\n", df['weather_type'].value_counts())

# ── 5. Train cluster-specific models ─────────────────────────────────────────
FEATURES = ['hour', 'month', 'dayOfWeek', 'dayOfYear',
            'lag1', 'lag24', 'rolling_mean', 'rolling_std']

os.makedirs('models', exist_ok=True)
results = {}

for cluster_id, weather in label_map.items():
    subset = df[df['cluster'] == cluster_id]
    X = subset[FEATURES]
    y = subset['Production']

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    results[weather] = {'MAE': round(mae, 2), 'R2': round(r2, 3)}
    print(f"{weather}: MAE={mae:.2f}, R²={r2:.3f}")

    joblib.dump(model, f'models/model_{weather}.pkl')

# ── 6. Save supporting objects ────────────────────────────────────────────────
joblib.dump(kmeans,  'models/kmeans.pkl')
joblib.dump(scaler,  'models/scaler.pkl')

label_map_named = {int(k): v for k, v in label_map.items()}
joblib.dump(label_map_named, 'models/label_map.pkl')

print("\n✅ All models saved in /models folder!")
print("Results:", results)