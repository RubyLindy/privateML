import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from model_info import model_selector
from parameters import model_parameters

## SET UP
#Preparing features (X) and target (y)
def prep_data(df):
    X = df[['top_2_pxcount','top2_fruitcount','top2_maxfruit','top2_minfruit']].values
    y = df['label'].values
    return train_test_split(X, y, test_size=0.3, random_state=None)

##MODEL TRAINING
def train_evaluate_model(model_type, df):

    X_train, X_test, y_train, y_test= prep_data(df)

    if model_type in model_selector:
        params = model_parameters.get(model_type, {})
        model, mse, mae = model_selector[model_type](X_train, X_test, y_train, y_test, params)
    else:
        print("Invalid model name. Please choose from the available options.")
        return

    print("\nMean Squared Error: ", mse)
    print("Mean Absolute Error: ", mae)

    return model

##MODEL EVALUATION

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using mean squared error (MSE), mean absolute error (MAE), and R² score.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

### MODELS

## LINEAR REGRESSION
def linear_model(X_train, X_test, y_train, y_test, params):

    print("\nRunning the linear regression model.")

    model = LinearRegression(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse, mae, r2 = evaluate_model(y_test, y_pred)

    return model, mse, mae, r2

## DECISION TREE
def decision_tree_model(X_train, X_test, y_train, y_test, params):

    print("\nRunning the decision tree model.")

    model = DecisionTreeRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse, mae, r2 = evaluate_model(y_test, y_pred)

    return model, mse, mae, r2

## RANDOM FOREST REGRESSOR
def random_forest_model(X_train, X_test, y_train, y_test, params):
    print("\nRunning the random forest model.")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse, mae, r2 = evaluate_model(y_test, y_pred)

    return model, mse, mae, r2