import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ["POST"])
def predict():
    ...
    #For rendering result on HTML GUI
    ...
    int_features = [x for x in request.form.values()]
    from flask import render_template
    var1 = str(int_features[8])
    var2 = str(int_features[9])

    var1 = var1.lower()
    var2 = var2.lower()

    Geography1 = (1 if var1 == "germany" else 0)
    Geography2 = (1 if var1 == "spain" else 0)
    Gender = (1 if var2 == "male" else 0)

    int_features[8] = Geography1
    int_features[9] = Geography2
    int_features.append(Gender)

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

