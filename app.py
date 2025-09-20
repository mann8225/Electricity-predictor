from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from dateutil.relativedelta import relativedelta

app = Flask(__name__)

# Load dataset and train model once
df = pd.read_csv("bhopal_climate_population_electricity.csv")

X = df[["Year", "Month", "Temperature", "Humidity", "Population"]]
y = df["Electricity_Consumption_MWh"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

month_temp = {
    1: 24.9, 2: 26.5, 3: 28.0, 4: 30.0, 5: 30.5, 6: 32.0,
    7: 30.9, 8: 29.5, 9: 29.0, 10: 27.5, 11: 26.0, 12: 24.95
}

month_humidity = {
    1: 50, 2: 52, 3: 55, 4: 58, 5: 26, 6: 40,
    7: 60, 8: 87, 9: 75, 10: 65, 11: 55, 12: 60
}

def interpolate_population(year):
    # Basic linear interpolation for population; adjust logic as needed
    if year <= 2001:
        return 3326228
    elif year <= 2011:
        return int(3326228 + (year - 2001) * (2371061 - 3326228) / (2011 - 2001))
    elif year <= 2025:
        return int(2371061 + (year - 2011) * (2300000 - 2371061) / (2025 - 2011))
    else:
        return 2300000

def get_prev_month(year, month, offset):
    dt = datetime(year, month, 1) - relativedelta(months=offset)
    return dt.year, dt.month

def get_next_month(year, month, offset):
    dt = datetime(year, month, 1) + relativedelta(months=offset)
    return dt.year, dt.month

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        year = int(data.get('year'))
        month = int(data.get('month'))
        if month < 1 or month > 12:
            return jsonify({"error": "Month must be between 1 and 12"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input data"}), 400

    months_to_predict = []
    # 5 months before
    for i in range(5, 0, -1):
        y, m = get_prev_month(year, month, i)
        months_to_predict.append((y, m))
    # current month
    months_to_predict.append((year, month))
    # 6 months after
    for i in range(1, 7):
        y, m = get_next_month(year, month, i)
        months_to_predict.append((y, m))

    predictions = []
    for y, m in months_to_predict:
        temperature = month_temp.get(m, 25.0)
        humidity = month_humidity.get(m, 58)
        population = interpolate_population(y)

        input_df = pd.DataFrame({
            "Year": [y],
            "Month": [m],
            "Temperature": [temperature],
            "Humidity": [humidity],
            "Population": [population]
        })

        pred = model.predict(input_df)[0]
        predictions.append({
            "month": f"{y}-{str(m).zfill(2)}",
            "value": round(float(pred), 2)
        })

    return jsonify({
        "success": True,
        "data": predictions
    })

if __name__ == '__main__':
    app.run(debug=True)
