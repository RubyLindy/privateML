# parameter.py

# Default model parameters
model_parameters = {
    "linear_regression": {
        "fit_intercept": True,
        "n_jobs": None
    },
    "decision_tree": {
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": None,
        "max_leaf_nodes": None,
    },
    "random_forest": { ##Unfinished
        "n_estimators": 100,
        "max_depth": None,
        "random_state": None
    }
}
