import json
import pandas as pd
import numpy as np
import datetime as dt
from flask import Blueprint, request
import services.apierrors as apierrors
from sklearn import svm
from io import StringIO
from services.storage import read_file, write_file, get_pickle
import pickle

svmBp = Blueprint("svmBp", __name__)


def preProcess(dataset):
    from sklearn.preprocessing  import StandardScaler
    from sklearn                import preprocessing
    le = preprocessing.LabelEncoder()
    X = dataset.apply(le.fit_transform)
    X = StandardScaler().fit_transform(X)            
    return X


@svmBp.route("/svm")
def root():
    return json.dumps({
        "name": "One-class Support Vector Machine",
        "type": "clustering"
    })


@svmBp.route("/svm/fit", methods=['POST'])
def fit():
    nu=0.1
    kernel="rbf"
    gamma=0.1    
    req = request.get_json()
    if "params" in req:
        if("nu" in req["params"]): nu = req["params"]["nu"]
        if("kernel" in req["params"]): kernel = req["params"]["kernel"]
        if("gamma" in req["params"]): gamma = req["params"]["gamma"]        
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
    else:
        return apierrors.NoData()

    print("start")
    fullPath = user_id + "/"+project_id+"/" + filename
    print(fullPath)
    dataset = read_file(fullPath)
    print(dataset)
    if(dataset==None): return apierrors.ErrorMessage("dataset not found")
    X = pd.read_csv(StringIO(dataset.decode('utf-8')))
    X = preProcess(dataset=X)
    X_train = X[0:int(len(X) * 0.66)]
    print("start")
    # fit the model
    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    s = pickle.dumps(clf)
    write_file(user_id, project_id, "pickle.pkl", s)

    return json.dumps({  
        "dataset": json.loads(pd.DataFrame(X).to_json()),
        "train_labels": y_pred_train.tolist(),        
        "labels": y_pred_test.tolist()
    })

@svmBp.route("/svm/predict", methods=['POST'])
def predict():
    req = request.get_json()
    if "params" in req:        
        data = req["params"]["data"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
    else:
        return apierrors.NoData()
    print("start")
    pkl_path = user_id + "/"+project_id+"/pickle.pkl"
    print(pkl_path)    
    model = get_pickle(pkl_path)
    y_pred = model.predict([data])    
    
    return json.dumps(y_pred)