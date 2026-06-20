import joblib
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "artifacts/models/model.pkl"
PREPROCESSOR_PATH = "artifacts/processed/preprocessor.pkl"
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

FEATURES = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'Year', 'Month', 'Day']

LABELS = {0:"No", 1:"Yes"}

@app.route("/",methods = ['GET','POST'])
def index():
    predication = None

    if request.method == 'POST':
        try:
            cat_cols = [
                "Location",
                "WindGustDir",
                "WindDir9am",
                "WindDir3pm",
                "RainToday"
            ]
            input_dict = {}
            for feature in FEATURES:
                value = request.form.get(feature)

                if feature in cat_cols:
                    input_dict[feature] = value
                else:
                    input_dict[feature] = float(value)

            input_df = pd.DataFrame([input_dict])
            
            input_processed = preprocessor.transform(input_df)

            pred = model.predict(input_processed)[0]
            predication = LABELS.get(pred,"Unkown")

        except Exception as e:
            print(str(e))

    return render_template(template_name_or_list="index.html",predication=predication, features=FEATURES)
    
if __name__=="__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")


