# Backend packages
from flask import Flask, url_for, request, render_template
# ML packages
import numpy as np
from sklearn.externals import joblib

# to extract name featrues
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0],  # First letter
        'first2-letters': name[0:2],  # First 2 letters
        'first3-letters': name[0:3],  # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:]
    }
features = np.vectorize(features)

app = Flask(__name__)

# Load the Vectorizer & Logistic Regression Pipeline
model = joblib.load('models/vectorizer_and_lr.pkl')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		name = request.form['name']
		name = features([name])
		prediction = model.predict(name)[0]

	return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
	app.run(debug=True)