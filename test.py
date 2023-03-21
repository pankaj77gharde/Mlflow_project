import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import warnings
import sys


def splitData(X, y):
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, random_state=True, test_size=0.2
    )
    return train_X, test_X, train_y, test_y


def evalMetrics(test_y, pred_y):
    accuracy = accuracy_score(test_y, pred_y)
    precision = precision_score(test_y, pred_y, average="weighted")
    recall = recall_score(test_y, pred_y, average="weighted")
    return accuracy, precision, recall


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_depth = int(sys.argv[2]) if len(sys.argv) > 1 else 5
    min_samples_leaf = int(sys.argv[3]) if len(sys.argv) > 1 else 3

    try:
        df_iris = load_iris()
        X = df_iris.data
        y = df_iris.target
    except Exception as e:
        print("Unable to get Dataset, please check....", e)

    train_X, test_X, train_y, test_y = splitData(X, y)

    best_param = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
    }
    with mlflow.start_run(run_name="Testing_RF") as run:
        final_modelRF = RandomForestClassifier(
            n_estimators=best_param["n_estimators"],
            max_depth=best_param["max_depth"],
            min_samples_leaf=best_param["min_samples_leaf"],
        )
        mlflow.sklearn.autolog()  # single code will logg all the data from model
        final_modelRF.fit(train_X, train_y)
        pred_y = final_modelRF.predict(test_X)
        accuracy, precision, recall = evalMetrics(test_y, pred_y)

        print(mlflow.get_artifact_uri())
        print(mlflow.get_registry_uri())
        print(mlflow.get_tracking_uri())
        print("run_id : ", run.info.run_id)

    print("accuracy : ", accuracy)
    print("precision : ", precision)
    print("recall : ", recall)
    print("best_param : ", best_param)
