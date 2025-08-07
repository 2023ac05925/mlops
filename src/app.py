from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from pydantic import BaseModel, Field, field_validator
from typing import List, Annotated
import numpy as np
import logging
import sqlite3
from datetime import datetime
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter

# Initialize Flask app
app = Flask(__name__)

# Load model
model = joblib.load(os.getenv('MODEL_PATH', 'models/iris_rf_model.joblib'))

metrics = PrometheusMetrics(app)

# Prometheus custom counters
prediction_counter = Counter('iris_predictions_total', 'Total prediction requests')
# error_counter = Counter('iris_prediction_errors_total', 'Total prediction errors')

# Pydantic models for validation
class IrisFeatures(BaseModel):
    sepal_length: Annotated[float, Field(gt=0, le=10)]  # cm
    sepal_width: Annotated[float, Field(gt=0, le=10)]   # cm
    petal_length: Annotated[float, Field(gt=0, le=10)]  # cm
    petal_width: Annotated[float, Field(gt=0, le=10)]   # cm

    @field_validator('sepal_width')
    @classmethod
    def sepal_width_lt_length(cls, v, info):
        if 'sepal_length' in info.data and v >= info.data['sepal_length']:
            raise ValueError('Sepal width must be less than sepal length')
        return v

    @field_validator('petal_width')
    @classmethod
    def petal_width_lt_length(cls, v, info):
        if 'petal_length' in info.data and v >= info.data['petal_length']:
            raise ValueError('Petal width must be less than petal length')
        return v

class PredictionRequest(BaseModel):
    data: List[IrisFeatures] = Field(..., min_length=1)


# Set up logging to file
logging.basicConfig(filename='prediction.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Helper: Log to SQLite
def log_to_db(input_data, prediction):
    conn = sqlite3.connect('logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (timestamp TEXT, input TEXT, prediction TEXT)''')
    c.execute("INSERT INTO logs VALUES (?, ?, ?)", (datetime.now().isoformat(), str(input_data), str(prediction)))
    conn.commit()
    conn.close()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        prediction_counter.inc()  # Increment Prometheus prediction counter

        # Validate input
        request_data = request.get_json()
        if not request_data:
            raise ValueError("No input data provided")
        
        validated_data = PredictionRequest.model_validate(request_data)
        
        # Convert to DataFrame with correct feature names
        input_data = pd.DataFrame([
              {
                'sepal_length': item.sepal_length,
                'sepal_width': item.sepal_width,
                'petal_length': item.petal_length,
                'petal_width': item.petal_width
            }
            for item in validated_data.data
        ])
        
        
        # Reorder columns to match training
        input_data = input_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

        # Make prediction
        predictions = model.predict(input_data).tolist()
        class_names = ['setosa', 'versicolor', 'virginica']
        predicted_classes = [class_names[p] for p in predictions]

        # Log to file
        logging.info(f"Input: {input_data.to_dict(orient='records')} | Predictions: {predictions} | Class names: {predicted_classes}")

        # Log to SQLite
        log_to_db(input_data.to_dict(orient='records'), {'predictions': predictions, 'class_names': predicted_classes})

        return jsonify({
            'predictions': predictions,
            'class_names': predicted_classes,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

#  /metrics endpoint
@app.route('/metrics', methods=['GET'])
def metrics():
    conn = sqlite3.connect('logs.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM logs")
    count = c.fetchone()[0]
    conn.close()
    return jsonify({'total_predictions': count})

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Assume new data is sent as a JSON payload
        new_data = request.get_json()
        if not new_data or 'data' not in new_data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        # Convert to DataFrame
        df = pd.DataFrame(new_data['data'])
        if 'target' not in df.columns:
            return jsonify({'status': 'error', 'message': 'Target column missing'}), 400

        X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = df['target']

        # Retrain model (RandomForest example)
        from sklearn.ensemble import RandomForestClassifier
        new_model = RandomForestClassifier()
        new_model.fit(X, y)

        # Save new model
        joblib.dump(new_model, 'models/iris_rf_model.joblib')
        
        # Reload the model in memory
        global model
        model = joblib.load('models/iris_rf_model.joblib')

        return jsonify({'status': 'success', 'message': 'Model retrained and saved.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)