import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import tree

def main():
    #load in DataFrame
    df = pd.read_csv("aardbei.csv", delimiter=';')

    #Ask for model of choice
    print("Currently there are 2 model choices:")
    print("1: Linear Regression.")
    print("2: Decision Tree.")
    model_type = input("Enter model number:").strip()
    if model_type == '1':
        max_depth = int(input("Enter the maximum depth for the decision tree: ").strip())
        train_evaluate_model(model_type, df, max_depth=max_depth)
    if model_type == '2':
        train_evaluate_model(model_type, df)
    else:
        raise ValueError("Invalid input. Please provide a model number.")


## SET UP
#Preparing features (X) and target (y)
def prep_data(df):
    X = df[['top_2_pxcount','top2_fruitcount','top2_maxfruit','top2_minfruit']].values
    y = df['label'].values
    return train_test_split(X, y, test_size=0.2, random_state=None)

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

def train_evaluate_model(model_type, df, **kwargs):
    X_train, X_test, y_train, y_test= prep_data(df)

    if model_type == '1':
        model ,mse, mae = linear_model(X_train, X_test, y_train, y_test)
        print("Quality prediction of linear regression:")
    elif model_type == '2':
        model, mse, mae = decision_tree_model(X_train, X_test, y_train, y_test)
        print("Quality prediction of the decision tree:")
        #Visualize the tree (not really necessary)
        plt.figure(figsize=(10, 5))
        tree.plot_tree(model, feature_names=['top_2_pxcount', 'top2_fruitcount', 'top2_maxfruit', 'top2_minfruit'], 
               impurity=True, filled=True, fontsize=4)
        plt.show()
    else:
        raise ValueError("Invalid input. Please provide a model number.")

    print("Mean Squared Error: {mse:.2f}")
    print("Mean Absolute Error: {mae:.2f}")

    return model

if __name__ == "__main__":
    main()