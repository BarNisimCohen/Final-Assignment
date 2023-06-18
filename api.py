import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

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
    features = request.form.getlist('feature')
    
    final_features = [features]
    final_features = np.array(final_features)
    # Reshape the array to match the expected shape of the model input
    final_features = final_features.reshape(1, -1)
    
    prediction = model.predict(final_features)[0]
    output_text = f"Predicted Property Value: {prediction:.2f}"
    
    return render_template('index.html', prediction_text=output_text)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port, debug=True)


