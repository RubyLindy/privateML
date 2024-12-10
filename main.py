#main.py
#Importing models and packages
from aardbei_models import train_evaluate_model
from parameters import model_parameters

import pandas as pd


def main():
    #load in DataFrame
    df = pd.read_csv("aardbei.csv", delimiter=';')

    menu(df)

## MENUS
#Starting menu
def menu(df):
    choice = 0
    while (choice != 3):
        print("Enter what you want to do:")
        print("1: Run a model.")
        print("2: Edit parameters")
        print("3: Exit")
        
        try:
            choice = int(input("Input: ").strip())

            if choice == 1:
                model_menu(df)
            elif choice == 2:
                parameter_menu()
            elif choice == 3:
                print("Exiting the program.")
            else: 
                print("Invalid input. Provide a choice between 1 and 3.")
        except ValueError as e:
            print(f"Error: {e}")
    return

def model_menu(df):
    choice = 0
    while(choice != 3):
        print("These are the available models:")
        print("1: Linear Regression")
        print("2: Decision Tree")
        print("3: Exit")
        choice = int(input("Input:").strip)

        match choice:
            case 1:
                train_evaluate_model(choice, df)
            case 2:
                train_evaluate_model(choice, df)
            case 3:
                print("Exiting.")
            case _:
                print("Provide a valid value.")
    return

def parameter_menu(): ##Continue on automating this
    choice = 0
    while(choice != 3):
        model_names = list(model_parameters.keys())
        print("Available models:")
        for model in model_names:
            print(f"  - {model}")
        model_name = int(input("Input:").strip)

        if model_name not in model_names:
            print("Model not found")
        return

        

def edit_parameters(model_name, model_parameters):
    #Exit if the model does not exist
    if model_name not in model_parameters:
        print("Model not found")
        return

    parameters = model_parameters[model_name]

    print(f"Current parameters for {model_name}:")
    for param, value in parameters.items():
        print(f"{param}: {value}")
    
    param_to_edit = input("\nEnter the name of the parameter you want to change:").strip()
    if param_to_edit not in parameters:
        print(f"Parameter '{param_to_edit}' does not exist for {model_name}.")
        return
    
    new_value = input(f"Enter the new value for {param_to_edit}:").strip()

    current_value = parameters[param_to_edit]
    if isinstance(current_value, bool):
        new_value = new_value.lower() in ['true', '1', 'yes']
    elif isinstance(current_value, int):
        new_value = int(new_value)
    elif isinstance(current_value, float):
        new_value = float(new_value)
    elif current_value is None:
        # If the current value is None, accept string, number, or 'None'
        if new_value.lower() == 'none':
            new_value = None
        elif new_value.isdigit():
            new_value = int(new_value)
        else:
            try:
                new_value = float(new_value)
            except ValueError:
                pass  # Leave as string if conversion fails

    parameters[param_to_edit] = new_value
    print(f"\nParameter '{param_to_edit}' updated to {new_value}.")

if __name__ == "__main__":
    main()

"""
    fit_intercept = float(input(f"Enter alpha (current: {model_parameters['linear_regression']['alpha']}): "))
    n_jobs = int(input(f"Enter max_iter (current: {model_parameters['linear_regression']['max_iter']}): "))

    model_parameters['linear_regression']['fit_intercept'] = fit_intercept
    model_parameters['linear_regression']['n_jobs'] = n_jobs
"""
