import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

##MODEL EVALUATION

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

def plot_predictions_vs_actual(y_test, y_pred, mae, mse):
    plt.figure(figsize=(8, 6))

    # Scatter plot of predictions vs actual values
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Predicted vs Actual")

    # Reference line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             color='red', linestyle='--', label="Perfect Prediction")

    # Plot aesthetics
    plt.title("Predicted vs Actual Values\nMAE:" + str(mae) + " | MSE: " + str(mse))
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

### MODELS

## LINEAR REGRESSION
def linear_model(X_train, X_test, y_train, y_test, params):

    print("\nRunning the linear regression model.")

    model = LinearRegression(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse, mae, r2 = evaluate_model(y_test, y_pred)

    plot_predictions_vs_actual(y_test, y_pred, mae, mse)

    return model, mse, mae, r2

## DECISION TREE
def decision_tree_model(X_train, X_test, y_train, y_test, params):

    print("\nRunning the decision tree model.")

    model = DecisionTreeRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse, mae, r2 = evaluate_model(y_test, y_pred)

    plot_predictions_vs_actual(y_test, y_pred, mae, mse)

    return model, mse, mae, r2

## RANDOM FOREST REGRESSOR
def random_forest_model(X_train, X_test, y_train, y_test, params):
    print("\nRunning the random forest model.")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse, mae, r2 = evaluate_model(y_test, y_pred)

    plot_predictions_vs_actual(y_test, y_pred, mae, mse)

    return model, mse, mae, r2

##SUPPORT VECTOR REGRESSION
def svr_model(X_train, X_test, y_train, y_test, params):
    print("\nRunning the support vector regression model.")
    model = SVR(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse, mae, r2 = evaluate_model(y_test, y_pred)

    plot_predictions_vs_actual(y_test, y_pred, mae, mse)

    return model, mse, mae, r2

##SIMPLE DIVISION MODEL
def simple_division_model(X_train, X_test, y_train, y_test, params):
    print("\nRunning the simple division model.")

    divisor = params.get("divisor", 2000000)
    weight = params.get("weight", 0.001)

    y_pred = X_test[:, 0] / (divisor + weight * X_test[:, 0])

    mse, mae, r2 = evaluate_model(y_test, y_pred)

    plot_predictions_vs_actual(y_test, y_pred, mae, mse)

    return None, mse, mae, r2