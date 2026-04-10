from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

kmeans = joblib.load('models/kmeans.pkl')
scaler = joblib.load('models/scaler.pkl')
label_map = joblib.load('models/label_map.pkl')

models = {
    'Overcast':     joblib.load('models/model_Overcast.pkl'),
    'Intermittent': joblib.load('models/model_Intermittent.pkl'),
    'Sunny':        joblib.load('models/model_Sunny.pkl'),
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    hour         = int(data['hour'])
    month        = int(data['month'])
    dayOfWeek    = int(data['dayOfWeek'])
    dayOfYear    = int(data['dayOfYear'])
    lag1         = float(data['lag1'])
    lag24        = float(data['lag24'])
    rolling_mean = float(data['rolling_mean'])
    rolling_std  = float(data['rolling_std'])

    production_proxy = (lag1 + lag24) / 2
    cluster_input = scaler.transform([[production_proxy, rolling_mean, rolling_std]])
    cluster_id = int(kmeans.predict(cluster_input)[0])
    weather_type = label_map[cluster_id]

    features = np.array([[hour, month, dayOfWeek, dayOfYear, lag1, lag24, rolling_mean, rolling_std]])
    prediction = models[weather_type].predict(features)[0]

    return jsonify({
        'prediction': round(float(prediction), 2),
        'weather_type': weather_type,
        'unit': 'MWh'
    })

if __name__ == '__main__':
    app.run(debug=True)