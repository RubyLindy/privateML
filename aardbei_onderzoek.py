import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import tree

df = pd.read_csv("aardbei.csv", delimiter=';')

## SET UP
#Preparing features (X) and target (y)
X = df[['top_2_pxcount','top2_fruitcount','top2_maxfruit','top2_minfruit']].values
y = df['label'].values

#Splitting data for training vs testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

## LINEAR REGRESSION

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Evaluation results (Probably should make a function out of this)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Prediction quality of linear regression:")

print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)

## DECISION TREE

regressor = DecisionTreeRegressor(max_depth=5, random_state=None) #Should I mess with the parameter manually if the set is so small?
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Evaluation results
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Prediction quality of the decision tree:")

print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)

#Visualize tree (not really necessary)
plt.figure(figsize=(10, 5))
tree.plot_tree(regressor, feature_names=['top_2_pxcount', 'top2_fruitcount', 'top2_maxfruit', 'top2_minfruit'], 
               filled=True)
plt.show()