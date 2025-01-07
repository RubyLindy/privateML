#main.py
#Importing models and packages
from parameters import model_parameters
from model_info import model_selector
from sklearn.model_selection import train_test_split

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
        print("\nEnter the number of what you want to do:")
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

    model_names = list(model_parameters.keys())
    print("\nEnter what model you'd like to run:")
    print("Available models:")
    for model in model_names:
        print(f"  - {model}")
    print("Enter 'exit' to return.")

    choice = input("Input:").strip().lower()

    if choice != "exit":
        train_evaluate_model(choice, df)

    return

def parameter_menu():
    model_name = ""
    while model_name != "exit":
        model_names = list(model_parameters.keys())
        print("\nEnter what model you'd like to edit: ")
        print("Available models:")
        for model in model_names:
            print(f"  - {model}")
        print("Enter 'exit' to return.")
        model_name = input("Input:").strip()

        if model_name != "exit":
            edit_parameters(model_name)
    return

        

def edit_parameters(model_name):
    #Exit if the model does not exist
    if model_name not in model_parameters:
        print("\nModel not found")
        return

    parameters = model_parameters[model_name]

    print(f"\nCurrent parameters for {model_name}:")
    for param, value in parameters.items():
        print(f"   {param}: {value}")
    
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

## SET UP
#Preparing features (X) and target (y)
def prep_data(df):
    X = df[['top_2_pxcount','top2_fruitcount','top2_maxfruit','top2_minfruit','left2_pxcount','left2_fruitcount','left2_maxfruit','left2_minfruit','right2_pxcount','right2_fruitcount','right2_maxfruit','right2_minfruit']].values
    y = df['label'].values
    return train_test_split(X, y, test_size=0.3, random_state=None)

## MODEL TRAINING
def train_evaluate_model(model_type, df):

    X_train, X_test, y_train, y_test= prep_data(df)

    if model_type in model_selector:
        params = model_parameters.get(model_type, {})
        model, mse, mae, r2 = model_selector[model_type](X_train, X_test, y_train, y_test, params)
    else:
        print("Invalid model name. Please choose from the available options.")
        return

    print("\nMean Squared Error: ", mse)
    print("Mean Absolute Error: ", mae)
    print("R2:", r2)

if __name__ == "__main__":
    main()

