import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


def train_model(X, y, categorical_cols):
    # create one hot encoding of categorical variables
    X = pd.get_dummies(X, columns=categorical_cols)
    # enocde target variable
    y = np.where(y == "Yes", 1, 0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )
    # Initialize the XGBoost classifier
    model = xgb.XGBClassifier()
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    # Print out the accuracy score
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # other metrics
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("F1: ", f1_score(y_test, y_pred))
    # calculate feature importance
    idx = np.argsort(model.feature_importances_)
    plt.barh(X.columns[idx], model.feature_importances_[idx])
    plt.xlabel("Xgboost Feature Importance")
    plt.show()
    importance = model.feature_importances_
    # print name of most important features
    print(X.columns[np.argsort(importance)[-5:][::-1]])
    return model


if __name__ == "__main__":
    data = pd.read_csv("../data/default_clean_v1.csv")
    data = data.drop(columns=["Unnamed: 0", "default payment next month"], axis=1)
    # check if 9 is present in PAY_0 or PAY_2
    print(data[(data["PAY_0"] == 9) | (data["PAY_2"] == 9)])
    X_protected = data.drop(["default", "ID"], axis=1)
    y = data["default"]
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    model_protected = train_model(X_protected, y, categorical_cols)
    X = data.drop(["default", "ID", "SEX"], axis=1)
    categorical_cols = ["EDUCATION", "MARRIAGE"]
    model = train_model(X, y, categorical_cols)
