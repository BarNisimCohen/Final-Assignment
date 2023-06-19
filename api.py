import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = {
        'City': request.form.get('City'),
        'type': request.form.get('type'),
        'condition': request.form.get('condition'),
        'entrance_date': request.form.get('entrance_date'),
        'hasElevator': request.form.get('hasElevator'),
        'hasParking': request.form.get('hasParking'),
        'hasStorage': request.form.get('hasStorage'),
        'hasBalcony': request.form.get('hasBalcony'),
        'hasMamad': request.form.get('hasMamad'),
        'furniture': request.form.get('furniture'),
        'room_number': request.form.get('room_number'),
        'Area': request.form.get('Area'),
        'floor': request.form.get('floor'),
        'total_floors': request.form.get('total_floors')
    }
    
    # Create a DataFrame from the features dictionary
    final_features = pd.DataFrame(features, index=[0])
    # Make the prediction
    prediction = model.predict(final_features)[0]
    output_text = f"Predicted Property Value: {prediction:.2f}"
    
    return render_template('index.html', prediction_text=output_text)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port, debug=True)
