import pickle
import numpy as np
from flask import Flask, request, render_template

# Flask app initialize
application = Flask(__name__)
app = application

# Load trained ML model + scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Create array
        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Scale input
        scaled_data = scaler.transform(input_data)

        # Predict
        result = ridge_model.predict(scaled_data)[0]

        return render_template('home.html', result=result)

    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080, debug=True)
