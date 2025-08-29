from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("artifacts/models/model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == "POST":
        sepal_length = float(request.form.get("sepal_length"))
        sepal_width = float(request.form.get("sepal_width"))
        petal_length = float(request.form.get("petal_length"))
        petal_width = float(request.form.get("petal_width"))

        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
