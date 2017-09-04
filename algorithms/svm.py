import json
import pandas as pd
import numpy as np
import datetime as dt
from flask import Blueprint, request
import services.apierrors as apierrors
from sklearn import svm

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
    if not "dataset" in req:
        return apierrors.NoData()

    X = req["dataset"]
    X_train = X[0:int(len(X) * 0.66)]

    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    return json.dumps({
        "train": X_train,
        "train_labels": y_pred_train.tolist(),
        "dataset": X,
        "labels": y_pred_test.tolist()
    })
