from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model and encoders
with open('xgb_classifier.pkl', 'rb') as file:
    model = pickle.load(file)
with open('le_gender.pkl', 'rb') as file:
    le_gender = pickle.load(file)
with open('le_geography.pkl', 'rb') as file:
    le_geography = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            form_data = request.form

            # Convert form data to a DataFrame
            input_data = pd.DataFrame({
                'CreditScore': [float(form_data['CreditScore'])],
                'Geography': [form_data['Geography']],
                'Gender': [form_data['Gender']],
                'Age': [int(form_data['Age'])],
                'Tenure': [int(form_data['Tenure'])],
                'Balance': [float(form_data['Balance'])],
                'NumOfProducts': [int(form_data['NumOfProducts'])],
                'IsActiveMember': [int(form_data['IsActiveMember'])],
                'EstimatedSalary': [float(form_data['EstimatedSalary'])]
            })

            # Encode the categorical data (Geography, Gender)
            input_data['Geography'] = le_geography.transform(input_data['Geography'])
            input_data['Gender'] = le_gender.transform(input_data['Gender'])

            # Make prediction
            prediction = model.predict(input_data)[0]

        except Exception as e:
            prediction = f"An error occurred: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
