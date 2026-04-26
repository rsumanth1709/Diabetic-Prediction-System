from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect all 8 input values
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    
    prediction = model.predict(final_features)
    output = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return render_template('index.html', prediction_text=f'Patient is {output}')

@app.route('/about')
def about():
    return "<h2>This web app predicts diabetes using ML model trained on PIMA dataset.</h2>"

if __name__ == "__main__":
    app.run(debug=True)
