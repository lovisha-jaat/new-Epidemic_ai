from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import time

app = Flask(__name__)
from flask_cors import CORS # Must be at the top

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # This allows ALL frontends (like Lovable) to talk to it

# Load global data once
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df = pd.read_csv(url)

@app.route('/predict', methods=['GET'])
def predict():
    region = request.args.get('region', default='India')
    days_ahead = int(request.args.get('day', default=10))

    # 1. Filter for country
    country_df = df[df['Country/Region'] == region]
    if country_df.empty:
        return jsonify({"error": "Country not found"}), 404
    
    # 2. Get the last 100 days of data
    # We sum in case a country has multiple provinces/states listed
    full_series = country_df.iloc[:, 4:].sum()
    y = full_series.values[-100:] 
    
    # 3. Create X as 0, 1, 2 ... 99
    X = np.arange(len(y)).reshape(-1, 1)

    # 4. Train the model on this specific window
    model = LinearRegression()
    model.fit(X, y)
    
    # 5. Predict for (100 + days_ahead)
    future_day_index = len(y) + days_ahead
    prediction_raw = model.predict([[future_day_index]])[0]
    
    # Ensure prediction isn't negative
    prediction_final = int(max(0, prediction_raw))

    # 6. Return JSON with a timestamp to prove it's new
    return jsonify({
        "status": "Success - New Logic Active",
        "timestamp": time.time(),
        "country": region,
        "current_cases": int(y[-1]),
        "prediction": prediction_final,
        "increase": prediction_final - int(y[-1])
    })

if __name__ == '__main__':
    # Force it to run on port 5001 to avoid the "stuck" port 5000
    app.run(port=5001, debug=True)