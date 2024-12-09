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
            choice = int(input("Input: ").strip)
            if choice == 1:
                model_menu(df)
            elif choice == 2:
                parameter_menu()
            else: ValueError("Invalid input. Provide an existing choice.")
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

def parameter_menu():
    choice = 0
    while(choice != 3):
        print("\nParameter Menu:")
        print("1. Edit Linear Regression Parameters")
        print("2. Edit Decision Tree Parameters")
        print("3. Go Back to Model Menu")
        choice = int(input("Input:").strip)

        match choice:
            case 1: 
                edit_linear_regression()
            case 2:
                edit_decision_tree()
            case 3:
                print("Exiting.")
            case _:
                print("Provide a valid value.")

def edit_linear_regression():
    print("\nLinear Regression Parameters:")
    alpha = float(input(f"Enter alpha (current: {model_parameters['linear_regression']['alpha']}): "))
    max_iter = int(input(f"Enter max_iter (current: {model_parameters['linear_regression']['max_iter']}): "))

    model_parameters['linear_regression']['alpha'] = alpha
    model_parameters['linear_regression']['max_iter'] = max_iter

    print("Linear Regression parameters have been updated")
    parameter_menu()

def edit_decision_tree():
    print("\nDecision Tree Parameters:")
    max_depth = int(input(f"Enter max_depth (current: {model_parameters['decision_tree']['max_depth']}): "))
    min_samples_split = int(input(f"Enter min_samples_split (current: {model_parameters['decision_tree']['min_samples_split']}): "))
    
    model_parameters['decision_tree']['max_depth'] = max_depth
    model_parameters['decision_tree']['min_samples_split'] = min_samples_split
    
    print("Decision Tree parameters updated!")
    parameter_menu()

if __name__ == "__main__":
    main()