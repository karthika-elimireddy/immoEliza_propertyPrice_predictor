# immoEliza_propertyPrice_predictor

## Dataset description
- The training and test dataset comes from data scrapped as a part of our previous project from the site [Immoweb](https://www.immoweb.be/)

### Shape

- The dataset is initially composed of 16,408 entries and 35 features each.

#### Selected Features list

1. Locality : The locality where the property is located.
2. mobib_score : A float number.
3. livingArea : A float in square meter.
4. bedrooms : An integer number representing the number of bedrooms in the property..
5. bathrooms : An integer number representing the number of bathrooms in the property.
6. toilets : An integer number representing the number of toilets in the property.
7. fireplace : *Does the property have an open fire ?* A boolean either true or false.
8. Number of facades : A float number representing the number of facades of the property.
9. Swimming pool : A boolean. Either true or false.
10. State of the building : Either 'As_new', 'Just_renovated', 'Good','To_be_done_up','To_renovate','To_restore' regarding the state of the property.

#### Target
1. Price : A float number in euros.


## WorkFlow
The following steps will be explored in more details below

1. Data analysis
2. Data preprocessing
3. Possible Improvements done.
4. Model selection: Linear Regression(ElasticNet)
5. Model training
6. Model evaluation
7. Results interpretation


## Data analysis

Some insights on the dataset. 

Price correlation with the other features : 

- locality            0.554733
- bathrooms           0.476861
- livingArea          0.401326
- bedrooms            0.315892
- toilets             0.274314
- buildingState       0.227274
- cadastralIncome     0.191139
- facades             0.148701
- constructionYear    0.147859
- fireplace           0.100461
- mobib_score        -0.017061
- pool                     NaN 

Heatmap :  

![heatmap](assets/images/Correlation_heatmap.png)

## Data preprocessing

Some features needed to be transformed to be able to train and test our model with it. 
Here are the said transformation and the respective library used:

- Handle the missing data and outliers in the dataset 
- Custom encoding of the categorical column 'buildingState'
- Separating the data between the train and the test set
- Target encoding of the categorical column 'Locality 
- Scaling the numerical columns and the encoded categorical columns
- PCA on the selected features/components.
- applied polynomialfeatures.

## Possible improvements Done 

As the data cleaning and hyperparameter tuning has already been done in depth, my main question to ask is on "the model".

In general , property price in belgium depends on many factors like structual , socio-economical, people entiments, market trends and many more. 
Firstly, taking only structural characteristics of the properties to predict the price would not be sufficient. 
secondly,the non-linear behaviour of those characteristics would make Linear Regression not a best fit in this case.

To stick to the project specifications of using Linear Regression, I thought of thinking of another type of linear regression model and considered ElasticNet with hyperparameters like max_iters , alpha, l1_ratio. 

To go even deeper into the information available for each property, one could retrieve information on the neighbourhood in which the property is located. Here are some questions that could be interesting and determining in the price of a house or a flat.

*How far away is the nearest public transport?*  

*How far is it to the schools?*  

*How far is it to the nearest supermarket?*  

Used OpenStreetMap (OSM) API, to calculate the proximity score or mobib score.

## Model selection

- Linear Regression( as defined by the project specifications)
 - ElasticNet

## Model training

The full dataset was split with a proportion of 0.8 for the training dataset.

## Hyperparameter Tuning

I used GridSearch to find the best hyperparameters.

## Model Evaluation
Here are the results of the model against the test data : 

- Mean Absolute Error on test data: 99972.10023833191
- Mean Squared Error: 27614867874.05416
- rmse: 166177.21827631537
- R² Score: 0.6924323813473299

Here are the results of the model against the training data : 

- Mean Absolute Error on training data: 100486.63313682767
- Mean Squared Error: 28607567050.683548
- rmse: 169137.7162275864
- R² Score: 0.672584329986172

In addition performed Cross fold validation with k=5  and the results are :

- Cross-Validation MAE scores: [ 89574.89383659 101418.19472758  87084.80726414 108309.01820205 128477.16748372]
- Average MAE: 102972.8163028167
- Standard Deviation of MAE: 14921.775938102104

- Cross-Validation R2 scores: [0.67493706 0.64356374 0.68223519 0.66052817 0.61103973]
- Average R2: 0.6544607798627968
- Standard Deviation of R2: 0.025422991087866366


## Model coefficient matrix

Feature           Coefficient    Importance 
mobib_score       90526.756067   90526.756067
toilets           76244.994865   76244.994865
cadastralIncome   69345.434871   69345.434871
locality          69297.585222   69297.585222
constructionYear  51528.605724   51528.605724
buildingState    -36031.848536   36031.848536

## Results interpretation

When estimating the price, the main indicator to take into consideration is the MAE (Mean Absolute Error).  

On average there is a 99 000 euros difference between the price estimated by the model and the actual price. 
In reflexion to the prices available on the belgian market its still quite a high difference. 

## Visual interpretation:

Predicted vs Actual price scatter plot :  

![Predicted vs Actual price](assets/images/PredvsActual.png)

Residual vs Predicted price scatter plot :  

![Residual vs Predicted price](assets/images/ResdidualvsPred.png)

Residual Density plot :  

![Residual Density](assets/images/ResidualDensity.png)

## Model Predictions:
- new test property data:
    new_property={
            'locality':'LEUVEN',
            'mobib_score':8,
            'bedrooms':2, 
            'bathrooms':1,
            'cadastralIncome':650,
            'livingArea':100,
            'buildingState':4,
            'constructionYear':1957,
            'facades':2,
            'fireplace':0,
            'toilets':2,
            'pool':0
        }
- Predicted Price for the new property: 302001.95524874394
- Inference Time: 0.001393 seconds