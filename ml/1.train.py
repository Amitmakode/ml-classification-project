# This file trains ML models using preprocessed data

import pandas as pd  # data manipulation
from sklearn.model_selection import train_test_split  # split data into train and test sets
from sklearn.ensemble import RandomForestClassifier  # random forest model
from xgboost import XGBClassifier  # xgboost model
from sklearn.metrics import accuracy_score, classification_report  # evaluate model performance
import pickle  # save model to disk
from dotenv import load_dotenv  # load env variables
import os  # access env variables
from preprocess import load_transformed_data, preprocess  # import preprocess functions

load_dotenv()  # load .env file

def train():
    df = load_transformed_data()  # load transformed data from mysql
    X, y = preprocess(df)  # apply scaling, feature selection and SMOTE

    # split into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train random forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # fit on training data
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))  # evaluate accuracy
    print(f"Random Forest Accuracy: {rf_acc:.4f}")  # log accuracy
    print(classification_report(y_test, rf_model.predict(X_test)))  # detailed report

    # train xgboost model
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)  # fit on training data
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))  # evaluate accuracy
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")  # log accuracy
    print(classification_report(y_test, xgb_model.predict(X_test)))  # detailed report

    # select best model based on accuracy
    if xgb_acc >= rf_acc:
        best_model = xgb_model  # xgboost better or equal
        best_name = "XGBoost"
    else:
        best_model = rf_model  # random forest better
        best_name = "RandomForest"

    print(f"Best Model: {best_name}")  # log best model name

    # save best model to disk
    os.makedirs('./ml', exist_ok=True)  # create ml dir if not exists
    with open(os.getenv('MODEL_PATH'), 'wb') as f:
        pickle.dump(best_model, f)  # serialize model
    print(f"Model saved to {os.getenv('MODEL_PATH')}")  # log save path

    return best_model, X_test, y_test, best_name  # return for evaluate use

if __name__ == "__main__":
    train()  # run training