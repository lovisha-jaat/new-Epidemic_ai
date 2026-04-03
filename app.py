from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# Load dataset once
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df = pd.read_csv(url)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # 1. Input
        raw_region = request.args.get('region', default='India').strip()
        days_ahead = int(request.args.get('day', default=10))

        # 2. Mapping (fix user input)
        mapping = {
            "usa": "US",
            "united states": "US",
            "uk": "United Kingdom",
            "south korea": "Korea, South",
            "uae": "United Arab Emirates",
            "russia": "Russian Federation"
        }

        search_term = mapping.get(raw_region.lower(), raw_region)

        # 3. Filter country
        mask = df['Country/Region'].str.fullmatch(search_term, case=False)
        country_df = df[mask]

        if country_df.empty:
            return jsonify({
                "error": "Country not found",
                "received": raw_region,
                "tip": "Try 'US', 'India', 'Brazil'"
            }), 404

        # 4. Prepare data
        full_data = country_df.iloc[:, 4:].sum()
        y = full_data.values[-100:]

        X = np.arange(len(y)).reshape(-1, 1)

        # 5. Train model
        model = LinearRegression()
        model.fit(X, y)

        # 6. Predict
        future_day_index = len(y) + days_ahead
        prediction = model.predict([[future_day_index]])[0]

        prediction = max(0, prediction)  # avoid negative

        # 7. 🔥 NEW IMPROVED LOGIC
        increase = prediction - y[-1]
        growth_percent = (increase / y[-1]) * 100

        # 8. Smart Risk Logic
        if growth_percent > 5:
            risk = "High"
        elif growth_percent > 1:
            risk = "Medium"
        else:
            risk = "Low"

        # 9. Response
        return jsonify({
            "status": "Success",
            "country": search_term,
            "current_cases": int(y[-1]),
            "prediction": int(prediction),
            "increase": int(increase),
            "growth_percent": round(growth_percent, 2),
            "risk": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)