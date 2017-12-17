import services.apierrors    as apierrors
import numpy                 as np
import pandas as pd
from io import StringIO
from services.storage import read_file, write_file
from sklearn.neighbors       import LocalOutlierFactor
from flask                   import Blueprint, request
import json
import pickle

lofBP = Blueprint("lofBP", __name__)

def preProcess(dataset):
    from sklearn.preprocessing  import StandardScaler
    from sklearn                import preprocessing
    le = preprocessing.LabelEncoder()
    X = dataset.apply(le.fit_transform)
    X = StandardScaler().fit_transform(X)            
    return X


@lofBP.route("/lof")
def root():
    return json.dumps({
        "name": "lof",
        "type": "cluster",
        "description": "algorithm is optional or a choice between",
        "params": ["neighbours", "algorithm"]
    })

@lofBP.route("/lof/fit", methods=['POST'])
def fit():
    req=request.get_json()
    neighburs=2
    if("params" in req):
        neighbours = req["params"]["neighbours"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
        if(user_id == None or project_id == None or filename == None):  return apierrors.NoData()
    else:
        return apierrors.NoData();

    fullPath = user_id + "/"+project_id+"/" + filename
    dataset = read_file(fullPath)
    clf = LocalOutlierFactor(n_neighbors=int(neighbours))
    rawX = pd.read_csv(StringIO(dataset.decode('utf-8')))
    X = preProcess(dataset=rawX)
    y_pred = clf.fit_predict(X)
    y_pred_outliers = y_pred

    s = pickle.dumps(clf)
    write_file(user_id, project_id, "pickle.pkl", s)
    resultObj = {
        "dataset": json.loads(rawX.to_json()),
        "labels": json.loads(pd.DataFrame(y_pred_outliers).to_json()),
        
    }

    return json.dumps(resultObj)

@lofBP.route("/lof/predict", methods=['POST'])
def predict():
    return "ok"