from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained logistic regression model
model = joblib.load('/Users/hamdilibanahmed/Downloads/ML Model/diabetes_model.pkl')

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        bmi = float(request.form['bmi'])
        glucose = float(request.form['glucose'])
        hba1c = float(request.form['hba1c'])
        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])
        family_history = int(request.form['family_history'])
        physical_activity = int(request.form['physical_activity'])
        region = int(request.form['region'])
        smoking_status = int(request.form['smoking_status'])
        alcohol_consumption = int(request.form['alcohol_consumption'])
        fasting_insulin = float(request.form['fasting_insulin'])

        # Create dataframe from form data
        data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'BMI': [bmi],
            'Blood Glucose (mg/dL)': [glucose],
            'HbA1c (%)': [hba1c],
            'Systolic BP (mmHg)': [systolic_bp],
            'Diastolic BP (mmHg)': [diastolic_bp],
            'Family History of Diabetes': [family_history],
            'Physical Activity Level': [physical_activity],
            'Region': [region],
            'Smoking Status': [smoking_status],
            'Alcohol Consumption': [alcohol_consumption],
            'Fasting Insulin (µU/mL)': [fasting_insulin]
        })

        # Make prediction
        prediction = model.predict(data)

        # Render result page with prediction
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
