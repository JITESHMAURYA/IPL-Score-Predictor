from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and columns
model = joblib.load('model.pkl')
model_columns = joblib.load('columns.pkl')

# Teams used for selection in dropdowns
teams = [
    'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        try:
            overs = float(request.form['overs'])
            wickets = int(request.form['wickets'])
            runs = int(request.form['runs'])
            bat_team = request.form['bat_team']
            bowl_team = request.form['bowl_team']

            # Create a zero-filled dict for all expected model columns
            input_dict = dict.fromkeys(model_columns, 0)
            input_dict['overs'] = overs
            input_dict['wickets'] = wickets
            input_dict['total'] = runs  # <- New feature added here

            # One-hot encode teams
            input_dict[f'bat_team_{bat_team}'] = 1
            input_dict[f'bowl_team_{bowl_team}'] = 1

            # Convert to DataFrame with correct column order
            input_df = pd.DataFrame([input_dict])[model_columns]

            # Predict final score
            pred = model.predict(input_df)[0]
            prediction = int(round(pred))

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, teams=teams)


if __name__ == '__main__':
    app.run(debug=True)
