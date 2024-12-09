#main.py
#Importing models and packages
from aardbei_models import train_evaluate_model

import pandas as pd


def main():
    #load in DataFrame
    df = pd.read_csv("aardbei.csv", delimiter=';')

## MENUS
#Starting menu
def menu(df, max_depth):
    choice = 0
    while (choice != 3):
        print("Enter what you want to do:")
        print("1: Run a model.")
        print("2: Edit parameters")
        print("3: Exit")
        
        try:
            choice = int(input("Input: ").strip)
            if choice == 1:
                model_menu(df, max_depth)
            elif choice == 2:
                parameter_menu()
            else: ValueError("Invalid input. Provide an existing choice.")
        except ValueError as e:
            print(f"Error: {e}")
    return

def model_menu(df, max_depth):
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







if __name__ == "__main__":
    main()