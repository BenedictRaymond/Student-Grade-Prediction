from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scalers
model = joblib.load('model/student_grades_model.pkl')
scaler_x = joblib.load('model/scaler_x.pkl')
scaler_y = joblib.load('model/scaler_y.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form data
        sex = 1 if request.form['sex'] == 'male' else 0
        age = int(request.form['age'])
        internet = 1 if request.form['internet'] == 'yes' else 0
        romantic = 1 if request.form['romantic'] == 'yes' else 0
        absences = int(request.form['absences'])
        G1 = int(request.form['G1'])
        G2 = int(request.form['G2'])
        traveltime = request.form['traveltime']
        studytime = request.form['studytime']
        freetime = request.form['freetime']

        # Convert categorical variables to numerical
        traveltime_mapping = {
            '1 hour to 2 hour': [0, 0, 0, 1],
            '15 to 30 min.': [0, 1, 0, 0],
            '30 min. to 1 hour': [0, 0, 1, 0],
            '<15 min.': [1, 0, 0, 0]
        }
        studytime_mapping = {
            '1 hour to 2 hour': [0, 0, 0, 1],
            '15 to 30 min.': [0, 1, 0, 0],
            '30 min. to 1 hour': [0, 0, 1, 0],
            '<15 min.': [1, 0, 0, 0]
        }
        freetime_mapping = {
            '1 hour to 2 hour': [0, 0, 0, 1, 0],
            '15 to 30 min.': [0, 1, 0, 0, 0],
            '30 min. to 1 hour': [0, 0, 1, 0, 0],
            '>2 hour': [0, 0, 0, 0, 1],
            '<15 min.': [1, 0, 0, 0, 0]
        }

        # Prepare the input array for the model
        input_array = [
            sex,
            age,
            internet,
            romantic,
            absences,
            G1,
            G2
        ]
        input_array.extend(traveltime_mapping[traveltime])
        input_array.extend(studytime_mapping[studytime])
        input_array.extend(freetime_mapping[freetime])
        input_array = [input_array]
        print(f"Before Scaling: {input_array}")
        # Scale the input features that need scaling
        to_scale = [input_array[0][1], input_array[0][4], input_array[0][5], input_array[0][6]]

        scaled_features = scaler_x.transform([to_scale])[0]

        input_array[0][1] = scaled_features[0]
        input_array[0][5] = scaled_features[1]
        input_array[0][6] = scaled_features[2]

        print(f"After Scaling: {input_array}")
        # Make a prediction
        prediction = model.predict(input_array)

        # Inverse transform the prediction to the original scale
        prediction = scaler_y.inverse_transform(prediction.reshape(-1, 1))[0, 0]
        prediction = f'{round(prediction * 5)} %'
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
