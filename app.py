from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

model_in = open("ml_model.pkl", "rb")
mlmodel = pickle.load(model_in)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/Application', methods=["POST", "GET"])
def application():
    if request.method == "GET":
        return render_template("getdata.html")
    else:
        return redirect(url_for("prediction"))

@app.route('/prediction', methods=["POST"])
def prediction():
    raw_features = [float(x) for x in request.form.values()]
    final_features = [np.array(raw_features)]
    predictions = mlmodel.predict(final_features)
    
    if predictions == [0]:
        predictions = "Iris Setosa"
    elif predictions == [1]:
        predictions = "Iris Versicolour"
    else:
        predictions = "Iris Virginica"
    
    return render_template("prediction.html", predictions=predictions)

if __name__ == "__main__":
    app.run()

