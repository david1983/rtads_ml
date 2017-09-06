import pickle
import json
from io import StringIO
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd
import numpy as np
import datetime as dt
from sklearn import metrics
from flask import Blueprint, request
import services.apierrors as apierrors
from services.storage import read_file, write_file


dbscanBP = Blueprint("dbscanBP", __name__)


@dbscanBP.route("/dbscan")
def root():
    return json.dumps({"name": "dbscan", "type": "clustering"})


@dbscanBP.route("/dbscan/fit", methods=['POST'])
def fit():
    req = request.get_json()
    eps=0.7
    min_samples=4
    if("params" in req):
        eps = req["params"]["eps"]
        min_samples = req["params"]["min"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["project_id"]
        if(user_id == None or project_id == None or filename == None):  return apierrors.NoData()        
        if "max" in req:
            max_samples = req["params"]["max"]
    else:
        return apierrors.NoData();

    
    fullPath = user_id + "/"+project_id+"/" + filename

    dataset = read_file(fullPath)
    if(dataset==None): return apierrors.ErrorMessage("dataset not found")
    le = preprocessing.LabelEncoder()
    X = pd.read_csv(StringIO(dataset.decode('utf-8')))
    X = X.fillna(0)
    for i in X:
        print(i)
        le.fit(X[i]) 
        print(le.classes_)
        X[i] = le.transform(X[i])    
        print(X[i])       

    # # X = req["dataset"]
    X = StandardScaler().fit_transform(X)
    print(X)
    db = DBSCAN(eps, min_samples).fit(X)
    s = pickle.dumps(db)
    print(s)
    write_file(user_id, project_id, "pickle.pkl", s)    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    resultObj = {
        "clusters": n_clusters_,
        "dataset": X.tolist(),
        "labels": labels.tolist()
    }            
    # resultObj = {}
    return json.dumps(resultObj)


@dbscanBP.route("/dbscan/predict", methods=['POST'])
def predict():
    return ""