# immoEliza_propertyPrice_predictor


## Table of Contents
- [Project Overview](#project_overview)
- [Prerequisites](#Prerequisites)
- [Usage](#Usage)
- [Structure](#Structure)
- [Contributors](#Contributors)

# Project Overview
- Create a prediction model whose target is the price of a property located in any locality in Belgium. 
- Get the most accurate Linear Regression model possible in other terms the lowest mean absolute error possible on the prediciton of a given property.
 

# Prerequisites

## Technologies
- Pandas
- Scikit-learn
- Seaborn
- LinearRegression(ElasticNet)
- Python 3.11.3 

Make sure you have the following:
- requirements.txt --- install using the command pip install -r utils/requirements.txt


# Usage

This script will:
1. Clean the dataset to handle NaN values and outliers.
2. Split the dataset into Training and testing set.
3. Perform Data preprocessing.
4. Build Linear Regression model(ElasticNet).
5. Evaluate the model.


--- To execute the project: 
         python property_price_predictor.py

# Structure
The project has the following core components:

1. assets: is a directory contains visual files
    - PredvsActual.png
    - ResdidualvsPred.png
    - ResidualDensity.png

2. data: is a directory contains data files
    - cleaned_data_withextrainfo.csv

3. utils: is a directory with ipynb files
    - linearreg_Elasticnet.ipynb
    - requirements.txt  #contains list of dependencies for the project.

4. property_price_predictor.py 
5. model_evaluation.md # describes the model and its efficiency.
5. .gitignore



## Contributors
![Karthika](https://github.com/karthika-elimireddy)
