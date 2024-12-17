# privateML
## General information
The goal of this software is find/create a machine learning model to correctly estimate the amount of strawberries in a punnet based on the pixel count of visible strawberries in the dataset.
### Contents of files
#### requirements.txt
requirements.txt is merely what packages I have installed on this PC, most of them I do use or need.

#### aardbei_models.py
aardbei_models.py is where all the functions that all upon the models are located, also most evaluation functions are here.

#### main.py
main.py is where the dataframe is loaded in, set up, and where we call upon the functions in aardbei_models.py. What models are called is decided through user input in the menu functions.

The actual main function itself is found here and only loads in the dataframe and calls the menu function.

#### model_info.py
model_info.py is where we associate a string, the user input of the model type, to the actual function in aardbei_models.py

#### parameters.py
parameters.py is where all the models' hyperparameters are saved so the models in aardbei_models.py can easily access them.

These parameters are also able to be changed through user input in the parameter_menu function in main.py.

## Data analysis
### Plots
Plots can be found in the plots folder, sorted by model type and labeled by version.

### Evaluation of errors
MSE, MAE, and R2 are all calculated and presented in the terminal everytime a model is run but not saved.

### Observations
17-12-2024 - All observations done with standard hyperparameters as input

MSE, MAE, and R2 are across all models quite similair in results. MSE is usually around 9, MAE is usually around 2.3, and R2 is usually around 0.87, all with variations due to the random state.

Looking at the plots, all of the models, beside linear regression, have a pattern of worse performance the more strawberries are actually in the punnet, least so for the Support Vector Regression (SVR) model. Linear regression has a consistent pattern accross the graph.