import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
import category_encoders as ce

import pandas as pd

# Load your dataset
df= pd.read_csv('data/cleaned_data_withextrainfo.csv')
df['locality'] = df['locality'].fillna('Unknown').astype(str)
localities = sorted(df['locality'].unique())
# Load your trained model and scaler

data= pickle.load(open('pickle/price_model.pkl', 'rb'))
model=data['model'] 
scaler=data['scaler'] 
pca=data['pca']
poly=data['poly']
target_Encoder=data['targetEncoder']
features=data['features']


app = Flask(__name__, template_folder='templates')

# Route for homepage (user input form)
@app.route('/')
def index():
    return render_template('index.html',localities=localities)

# Route for handling form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    locality = request.form['locality'].lower()
    livingArea = float(request.form['livingArea'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    facades = int(request.form['facades'])
    fireplace = int(request.form['fireplace'])
    toilets = int(request.form['toilets'])
    pool = int(request.form['pool'])
    buildingState = request.form['buildingState']
    constructionYear = int(request.form['constructionYear'])
    cadastralIncome = float(request.form['cadastralIncome'])
    mobib_score = float(request.form['mobib_score'])

    # Convert data into DataFrame for model prediction
    input_data = pd.DataFrame([[locality, mobib_score, bedrooms, bathrooms, cadastralIncome, livingArea, buildingState, constructionYear, facades, fireplace, toilets, pool]],
                               columns=['locality','mobib_score','bedrooms', 'bathrooms','cadastralIncome','livingArea','buildingState','constructionYear','facades','fireplace','toilets','pool'])

    # Apply any necessary preprocessing (e.g., encoding, scaling)
    input_data['locality'] = target_Encoder.transform(input_data['locality'])
    input_data = scaler.transform(input_data)
    input_data = pca.transform(input_data)
    input_data = poly.transform(input_data)

    # Make the prediction using the model
    predicted_price = model.predict(input_data)[0]

    # Display the result
    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)

