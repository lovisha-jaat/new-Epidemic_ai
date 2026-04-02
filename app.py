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
    # 1. Clean the input: remove spaces and make it lowercase for comparison
    raw_region = request.args.get('region', default='India').strip()
    days_ahead = int(request.args.get('day', default=10))

    # 2. Dataset "Translation" Map
    # This fixes common user inputs to match the JHU dataset names
    mapping = {
        "usa": "US",
        "united states": "US",
        "uk": "United Kingdom",
        "south korea": "Korea, South",
        "uae": "United Arab Emirates",
        "russia": "Russian Federation"
    }
    
    # Check if the input needs a translation, otherwise use capitalized raw input
    search_term = mapping.get(raw_region.lower(), raw_region)

    # 3. Flexible Filter (Finds 'india' or 'India' or 'INDIA')
    # We use a case-insensitive regex to match the Country/Region column
    mask = df['Country/Region'].str.fullmatch(search_term, case=False)
    country_df = df[mask]

    if country_df.empty:
        return jsonify({
            "error": "Country not found",
            "received": raw_region,
            "tried_searching_for": search_term,
            "tip": "Try 'US' or 'United Kingdom' or 'Brazil'"
        }), 404
    
    # 4. Logic stays the same (Sum all regions for that country)
    full_data = country_df.iloc[:, 4:].sum()
    y = full_data.values[-100:] 
    
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    
    future_day_index = len(y) + days_ahead
    prediction = model.predict([[future_day_index]])[0]

    return jsonify({
        "status": "Success - Logic Fixed",
        "country_found": search_term,
        "current_cases": int(y[-1]),
        "prediction": int(max(0, prediction)),
        "increase": int(max(0, prediction) - y[-1])
    })

if __name__ == '__main__':
    # Force it to run on port 5001 to avoid the "stuck" port 5000
    app.run(port=5001, debug=True)