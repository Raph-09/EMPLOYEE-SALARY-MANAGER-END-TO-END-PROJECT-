import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import pickle
from preproces import prepro





app = Flask(__name__)
model = pickle.load(open('salary_model_2.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        experience = int(request.form['experience'])

        test_score = int(request.form['test_score'])

        Interview_Score = int(request.form['interview_score'])

        scaled_x = prepro([[experience, test_score, Interview_Score]])
        prediction = model.predict(scaled_x)
        output = round(prediction[0], 0)
        if output < 0:
            return render_template('predit.html', prediction_texts="You cannot have a negative salary")
        else:
            return render_template('index.html', prediction_text="The employee's salary should be:$ {}".format(output))
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)