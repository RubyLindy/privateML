# privateML
## General information
The goal of this software is find/create a machine learning model to correctly estimate the amount of strawberries in a punnet based on the pixel count of visible strawberries in the dataset.
### Contents of files
#### requirements.txt
requirements.txt contains the names and versions of the packages I have installed on this PC, most of them I do use or need, some are irrelevant.

#### aardbei_models.py
aardbei_models.py is where all the functions that call upon the skilearn models are located, also most evaluation functions are here.

#### main.py
main.py is where the dataframe is loaded in, set up, and where we call upon the functions in aardbei_models.py. What models are called is decided through user input in the menu functions.

The actual main function itself is found here and only loads in the dataframe and calls the menu function.

#### model_info.py
model_info.py is where we associate a string, the user input of the model type, to the actual function in aardbei_models.py

#### parameters.py
parameters.py is where all the models' hyperparameters are saved so the models in aardbei_models.py can access them.

These parameters are also able to be changed through user input in the parameter_menu function in main.py.

## Data analysis
### Plots
Plots can be found in the plots folder, sorted by model type, each having their own folder, and labeled by version.

### Evaluation of errors
MSE, MAE, and R2 are all calculated and presented in the terminal every time a model is run and are added to the generated plots via the title.

### Observations
17-12-2024 - All observations done with standard hyperparameters as input

MSE, MAE, and R2 are across all models quite similair in results. MSE is usually around 9, MAE is usually around 2.3, and R2 is usually around 0.87, all with variations due to the random state.

Looking at the plots, all of the models, beside linear regression, have a pattern of worse performance the more strawberries are actually in the punnet, least so for the Support Vector Regression (SVR) model. Linear regression has a consistent pattern accross the graph.

23-12-2024

For the simple_division model, version 1 was a very simple equation of just the top_2_pxcount divided by 2000000. The subsequent version were all weighted to some extent, with only marginally improved results.

Simple_division has a generally worse performance than all other models, but exhibits similiar behaviour, with adequate performance at the lower actual values but predicting significantly too low with higher actual values. This problem has not been observed to be able to be solved with weighted dividors.

24-12-2024

KNN was added and exhibits the same behaviour as the other models, except simple_division. I conclude that the only way to improve results is to add more instances or more features.

7-1-2024

The other perspectives were added to the data set and new plots were made for all beside simple division (all v3 beside knn, there it is v2). All noticably perform better than how they did with the previous dataset. But the new dataset does have some strange outliers, some instances have all values beside the label set as 0, causing some of the more susceptible models to also present 0's as predictions.

After this, the Gradient Boosting Regressor model was added and noticibly is less sensitive to outliers and performs similiarly well than the other models.

13-1-2024

All instances of all 0's deleted (7 instances). All models are run again