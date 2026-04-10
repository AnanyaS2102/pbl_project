import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import json
import os

# Load & filter
df = pd.read_csv('data/intermittent-renewables-production-france.csv')
df = df[df['Source'] == 'Solar'].copy()

df['datetime'] = pd.to_datetime(df['Date and Hour'], utc=True)
df['hour']     = df['datetime'].dt.hour
df['month']    = df['datetime'].dt.month
df['dayOfWeek']= df['datetime'].dt.dayofweek
df = df.sort_values('datetime').reset_index(drop=True)
df['lag1']        = df['Production'].shift(1)
df['lag24']       = df['Production'].shift(24)
df['rolling_mean']= df['Production'].rolling(6).mean()
df['rolling_std'] = df['Production'].rolling(6).std()
df = df.dropna().reset_index(drop=True)

FEATURES = ['hour', 'month', 'dayOfWeek', 'dayOfYear',
            'lag1', 'lag24', 'rolling_mean', 'rolling_std']

X = df[FEATURES]
y = df['Production']
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Scale for SVR
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    'Linear Regression': (LinearRegression(), False),
    'Random Forest':     (RandomForestRegressor(n_estimators=100, random_state=42), False),
    'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
    'SVR':               (SVR(kernel='rbf', C=100, epsilon=0.1), True),
}

results = []
for name, (model, use_scaled) in models.items():
    print(f"Training {name}...")
    if use_scaled:
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    results.append({'Model': name, 'MAE': round(mae,2), 'RMSE': round(rmse,2), 'R2': round(r2,4)})
    print(f"✅ {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")

os.makedirs('static', exist_ok=True)
with open('static/comparison.json', 'w') as f:
    json.dump(results, f)

best = max(results, key=lambda x: x['R2'])
print(f"\n🏆 Best Model: {best['Model']} with R²={best['R2']}")
print("\n✅ Results saved to static/comparison.json!")