from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# -------------------------------
# Train model (or load existing)
# -------------------------------
data = pd.read_csv('diabetes.csv')

X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model for reuse (optional)
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input values from the form
        features = [float(request.form[key]) for key in request.form.keys()]
        prediction = model.predict([features])[0]
        
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == "_main_":
    app.run(debug=True)