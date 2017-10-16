import json
import pandas as pd
import numpy as np
import datetime as dt
from flask import Blueprint, request
import services.apierrors as apierrors
from sklearn import svm
from services.storage import read_file, write_file

svmBp = Blueprint("svmBp", __name__)


@svmBp.route("/svm")
def root():
    return json.dumps({
        "name": "One-class Support Vector Machine",
        "type": "clustering"
    })


@svmBp.route("/svm/fit", methods=['POST'])
def fit():
    req = request.get_json()
    if "params" in req:
        columns = req["params"]["columns"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["project_id"]
    else:
        return apierrors.NoData()

    fullPath = user_id + "/"+project_id+"/" + filename
    dataset = read_file(fullPath)
    if(dataset==None): return apierrors.ErrorMessage("dataset not found")
    X = dataset
    X_train = X[0:int(len(X) * 0.66)]

    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    s = pickle.dumps(clf)
    write_file(user_id, project_id, "pickle.pkl", s)
    return json.dumps({
        "train": X_train,
        "train_labels": y_pred_train.tolist(),
        "dataset": X,
        "labels": y_pred_test.tolist()
    })

@svmBp.route("/svm/predict", methods=['POST'])
def predict():
    return ""