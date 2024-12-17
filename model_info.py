#model_selector.py

from aardbei_models import linear_model, decision_tree_model, random_forest_model, svr_model

model_selector = {
    "linear_regression": linear_model,
    "decision_tree": decision_tree_model,
    "random_forest": random_forest_model,
    "svr": svr_model
}