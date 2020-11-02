import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("model.h5")
scalar = pickle.load(open('scalar.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ["POST"])
def predict():
    ...
    #For rendering result on HTML GUI
    ...
#    int_features = [x for x in request.form.values()]

    
    CreditScore = int(request.form['creditScore'])
    age = int(request.form['age'])
    Tenure = int(request.form['Tenure'])
    Balance = int(request.form['Balance'])
    numOfProducts = int(request.form['numOfProducts'])
    hasCrCard = int(request.form['hasCrCard'])
    estimatedSalary = int(request.form['estimatedSalary'])
    geography = str(request.form['geography'])
    gender = str(request.form['gender'])
    isActiveMember = int(request.form['isActiveMember'])


    from flask import render_template

    geography = geography.lower()
    gender = gender.lower()

    Geography1 = (1 if geography == "germany" else 0)
    Geography2 = (1 if geography == "spain" else 0)
    Gender = (1 if gender == "male" else 0)

    int_features = [CreditScore,age,Tenure,Balance,numOfProducts,hasCrCard,estimatedSalary,Geography1,Geography2,Gender,isActiveMember]
    int_features = scalar.transform(int_features)
    
    final_features = (np.array(int_features))
    final_features.resize(1,11)

    model = keras.models.load_model('model.h5')

    prediction =  model.predict(final_features)
    prediction = (1 if prediction > 0.5 else 0)

    if prediction == 1:
        return render_template('index.html',prediction_text = "The Customer closed the account")
    else:
        return render_template('index.html',prediction_text = "The Customer did not closed the account")


if __name__ == "__main__":
    app.run(debug=True)

