import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

## SET UP
#Preparing features (X) and target (y)
def prep_data(df):
    X = df[['top_2_pxcount','top2_fruitcount','top2_maxfruit','top2_minfruit']].values
    y = df['label'].values
    return train_test_split(X, y, test_size=0.3, random_state=None)

##MODEL TRAINING
def train_evaluate_model(model_type, df, **kwargs):
    X_train, X_test, y_train, y_test= prep_data(df)

    if model_type == 1:
        model ,mse, mae = linear_model(X_train, X_test, y_train, y_test)
        print("Quality of predictions of linear regression:")
    elif model_type == 2:
        model, mse, mae = decision_tree_model(X_train, X_test, y_train, y_test)
        print("Quality of predictions of the decision tree:")
        #Visualize the tree (not really necessary)
        plt.figure(figsize=(10, 5))
        tree.plot_tree(model, feature_names=['top_2_pxcount', 'top2_fruitcount', 'top2_maxfruit', 'top2_minfruit'], 
               impurity=True, filled=True, fontsize=4)
        plt.show()
    else:
        raise ValueError("Invalid input. Please provide a model number.")

    print("Mean Squared Error: ", mse)
    print("Mean Absolute Error: ", mae)

    return model

### MODELS

## LINEAR REGRESSION
def linear_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #Evaluation results (Probably should make a function out of this)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mse, mae

## DECISION TREE
def decision_tree_model(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_depth=6, random_state=None) #Should I mess with the parameter manually if the set is so small?
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #Evaluation results
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mse, mae