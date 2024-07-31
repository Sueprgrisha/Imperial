from src.config import *
from src.data_raw_prep import *
from src.models.models_scripts.basic_model import *

def process_data():

    print("Processing data...")
    preproc_db(RAW_DATA_PATH, PROCESSED_DATA_DIR)
    preproc_for_model(PROCESSED_DATA_DIR,DATA_FOR_MODEL_PATH)

def run_training(model_name):
    print("Running training...")
    fit_and_predict(DATA_FOR_MODEL_PATH, model_name)

# def predict():
#     print("Predicting...")


def main():
    options = {
        '0': ("End", ""),
        '1': ("Process data", process_data),
        '2': ("Run training and predict", run_training),
        # '3': ("Predict", predict)
    }
    models = ['model_tree','model_rf']
    print(options)
    while True:
        print("Choose the option:")
        for key, (description, _) in options.items():
            print(f"[{key}] {description}")

        choice = input("Enter the number of your choice: ")
        if choice == '0':
            break


        elif choice in options:
            print("\n")
            if choice == '2':
                print("choose one of the models:")
                print(models)
                choice_model = input("Enter the model: ")
                options[choice][1](choice_model)
            else:
                options[choice][1]()
            print("Done!")
        else:
            print("\n\nInvalid choice. Please enter 0, 1 or 2")


if __name__ == "__main__":
    main()
