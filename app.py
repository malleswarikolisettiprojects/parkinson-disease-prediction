import os
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load secret key from environment variable (set a default only for local dev)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Load the trained model
model = pickle.load(open('model1.pkl', 'rb'))


@app.errorhandler(404)
def pageNotFound(e):
    return render_template('404.html'), 404


@app.errorhandler(405)
def methodNotFound(e1):
    return render_template('405.html'), 405


@app.errorhandler(500)
def incorrectinput(error):
    return render_template('500.html'), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [[float(x) for x in request.form.values()]]
    final = np.array(int_features)

    prediction = model.predict(final)
    output = prediction[0]

    if output == 0:
        s = 'Negative'
    elif output == 1:
        s = 'Positive'

    proba = model.predict_proba(final)
    prob1 = proba[0][1] * 100

    if prob1 > 70:
        a = "High"
    elif 30 < prob1 <= 70:
        a = "Medium"
    else:
        a = "Low"

    return render_template(
        'result.html',
        pred='Test Result : {}'.format(s),
        pred1='Percentage of risk  : {:.2f}%'.format(prob1),
        pred2='Risk Level : {}'.format(a)
    )


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    final = np.array([list(data.values())])
    prediction = model.predict_proba(final)
    output = prediction[0]
    return jsonify(output.tolist())


if __name__ == "__main__":
    # Read DEBUG and PORT from environment variables for safe deployment
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)