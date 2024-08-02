import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def model_tree(data):

    X_train, X_test, y_train, y_test = data
    clf = tree.DecisionTreeClassifier()



    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f'5-Fold Cross-Validation Scores: {cv_scores}')
    print(f'Mean Cross-Validation Accuracy: {np.mean(cv_scores)}')

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Print classification report which includes precision, recall, and F1-score
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Optionally, print predictions and ground truth for a few test samples
    print('Sample Predictions vs Ground Truth:')
    for i in range(min(10, len(y_test))):  # Displaying at most 10 samples
        print(f'Predicted: {y_pred[i]}, Ground Truth: {y_test[i]}')


def model_rf(data,n_estimators=20):

    X_train, X_test, y_train, y_test = data
    clf = RandomForestClassifier(random_state=42, n_estimators = n_estimators)
    X = pd.concat([X_train,X_test])
    y = pd.concat([y_train y_test])
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f'5-Fold Cross-Validation Scores: {cv_scores}')
    print(f'Mean Cross-Validation Accuracy: {np.mean(cv_scores)}')

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Print classification report which includes precision, recall, and F1-score
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Optionally, print predictions and ground truth for a few test samples
    print('Sample Predictions vs Ground Truth:')

    # Convert the model to ONNX format
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)

    # Save the ONNX model
    with open("src/models/models_packed/random_forest.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


    for i in range(min(10, len(y_test))):  # Displaying at most 10 samples
        print(f'Predicted: {y_pred[i]}, Ground Truth: {y_test[i]}')






def fit_and_predict(data_path,model_name):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    try:
        # Get the function by name
        func = globals()[model_name]
        # Call the function
        func(data)
    except KeyError:
        print(f"No such function: {model_name}")












