from flask import Flask, make_response, request, render_template
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)

    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = stream.read()

    df = pd.read_csv(StringIO(result))
    

    # load the model from disk
    loaded_model = pickle.load(open("model.pkl", 'rb'))
    prediction = loaded_model.predict(df)

    return render_template("index.html",prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

