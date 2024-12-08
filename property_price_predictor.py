

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
import category_encoders as ce
import matplotlib.pyplot as plt
import pickle

class PricePredictionModel:
    def __init__(self,num_of_features : int):
        """Initialize the model and other required attributes."""
        self.model =ElasticNet(max_iter=10000, alpha=1.0, l1_ratio=1.0)
        self.feature_names = None
        self.scaler = StandardScaler()
        self.poly=PolynomialFeatures(degree=2, include_bias=False)
        self.num_of_features=num_of_features
        self.pca=PCA(n_components=self.num_of_features)
        self.targetEncoder = ce.TargetEncoder(cols=['locality'], handle_unknown='value', handle_missing='value',smoothing=5)
        

    def preprocess_data(self, df, target_column):
        """
        Split the data into training and test sets, and standardize features.
        Args:
        - df: DataFrame containing the data.
        - target_column: The column name of the target variable.

        Returns:
        - X_train, X_test, y_train, y_test: Processed data splits.
        """
        
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        
        
        train_data['locality'] = self.targetEncoder.fit_transform(train_data['locality'], train_data['price'])
        test_data['locality'] = self.targetEncoder.transform(test_data['locality'])
        

        X_train = train_data.drop(columns=['price'])  # Drop the target column
        y_train = train_data['price']                # Target variable

        X_test = test_data.drop(columns=['price'])   # Drop the target column
        y_test = test_data['price']     
        
        original_feature_names=X_train.columns

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train = self.pca.fit_transform(X_train)
        X_test = self.pca.transform(X_test)

        X_train = self.poly.fit_transform(X_train)
        X_test = self.poly.transform(X_test)

        self.feature_names = self.poly.get_feature_names_out(original_feature_names)
       
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Train the linear regression model on the training data.
        """
        self.model.fit(X_train, y_train)
        print("Model trained successfully!")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        Args:
        - X_test: Features of the test set.
        - y_test: Target values of the test set.

        Returns:
        - Metrics: Dictionary containing MAE, MSE, RMSE, and R2.
        """
        y_pred = self.model.predict(X_test)
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': mean_squared_error(y_test, y_pred, squared=False),
            'R2 Score': r2_score(y_test, y_pred)
        }

        
        return metrics

    def save_model(self, filename):
        """
        Save the trained model and scaler to a file using pickle.
        Args:
        - filename: The name of the file to save the model.
        """
        with open(filename, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'pca': self.pca, 'poly': self.poly, 'targetEncoder':self.targetEncoder,'features': self.feature_names}, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """
        Load a trained model and scaler from a file.
        Args:
        - filename: The name of the file to load the model from.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.targetEncoder = data['targetEncoder']
            self.scaler = data['scaler']
            self.pca = data['pca']
            self.poly = data['poly']
            self.feature_names = data['features']
        print(f"Model loaded from {filename}")

    def predict(self, new_data):
        """
        Predict the price for a new property.
        Args:
        - new_data: A dictionary of feature values.

        Returns:
        - Predicted price.
        """
        # Convert new data to a DataFrame
        new_data_df = pd.DataFrame([new_data])
        new_data_df['locality'] = new_data_df['locality'].str.lower()
        new_data_df['locality'] = self.targetEncoder.transform(new_data_df['locality'])
        new_data_scaled = self.scaler.transform(new_data_df)
        new_data_pca = self.pca.transform(new_data_scaled)
        new_data_poly = self.poly.transform(new_data_pca)
        
        # Predict
        return self.model.predict(new_data_poly)[0]
    
    def clean_data(self, df):
        df['locality'] = df['locality'].str.lower()
        df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mode())
        df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].mode())
        df['toilets'] = df['toilets'].fillna(df['toilets'].mode())
        df['livingArea'] = df['livingArea'].fillna(df['livingArea'].mean())
        df['cadastralIncome'] = df['cadastralIncome'].fillna(df['cadastralIncome'].mean())
        df['facades'] = df['facades'].fillna(2)
        df['pool'] = df['pool'].fillna(0)
        df['fireplace'] = df['fireplace'].fillna(1)
        
        building_state_mapping = {
            'AS_NEW': 6,
            'JUST_RENOVATED': 5,
            'GOOD': 4,
            'TO_BE_DONE_UP': 3,
            'TO_RENOVATE':2,
            'TO_RESTORE':1
        }
        df['buildingState'] = df['buildingState'].map(building_state_mapping)
        #Handling Outliers and NaN
        df= df.dropna()
        df= df.dropna(subset=['price'])
        #handling the outliers
        df=self.remove_outliers(df, 'price')
        df=self.remove_outliers(df, 'livingArea')
        df=self.remove_outliers(df, 'bathrooms')
        df=self.remove_outliers(df, 'bedrooms')
        df=self.remove_outliers(df, 'facades')
        df=self.remove_outliers(df, 'toilets')
        df=self.remove_outliers(df, 'pool')
        df=self.remove_outliers(df,'cadastralIncome')

        return df
    
    def remove_outliers(self,df, column):
            Q1 = df[column].quantile(0.05)
            Q3 = df[column].quantile(0.95)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_without_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            #print(f'{len(df)-len(df_without_outliers)} rows have been removed from the {column} column')
            return df_without_outliers
    

if __name__ == "__main__":
    # Sample dataset
    df= pd.read_csv('data/cleaned_data_withextrainfo.csv')
    #Select Features list
    # df_featured=df[['locality','bedrooms', 'bathrooms','livingArea','cadastralIncome','buildingState','toilets','facades','pool','price']]
    df_featured=df[['locality','mobib_score','bedrooms', 'bathrooms','cadastralIncome','livingArea','buildingState','constructionYear','facades','fireplace','toilets','pool','price']]
    
    
    num_of_features=df_featured.shape[1]-1
    # Initialize and preprocess
    model = PricePredictionModel(num_of_features)
    df_featured=model.clean_data(df_featured)
    X_train, X_test, y_train, y_test = model.preprocess_data(df_featured, target_column='price')
    # Train the model
    model.train(X_train, y_train)

    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print("Evaluation Metrics:", metrics)

    # Save the model
    model.save_model('pickle/price_model.pkl')

    # Predict a new property
    #'HOUSE'--property_type=1
    #'APARTMENT'--property_type=0
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

    predicted_price = model.predict(new_property)
    print("Predicted Price for the new property:", predicted_price)
