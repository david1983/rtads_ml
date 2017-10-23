import services.apierrors   as apierrors
from services.storage import read_file, write_file
from sklearn.neighbors import NearestNeighbors
from flask import Blueprint, request
from io                     import StringIO
import pandas as pd
import json
import pickle
import numpy as np

knnBP = Blueprint("knnBP", __name__)


@knnBP.route("/knn")
def root():
    return json.dumps({
        "name": "knn",
        "type": "cluster",
        "description": " algorithm is optional or a choice between",
        "params": ["neighbours", "algorithm"]
    })

def preProcess(dataset):
    from sklearn.preprocessing  import StandardScaler
    from sklearn                import preprocessing
    le = preprocessing.LabelEncoder()
    X = dataset.apply(le.fit_transform)
    X = StandardScaler().fit_transform(X)            
    return X

@knnBP.route("/knn/fit", methods=['POST'])
def fit():
    req = request.get_json()
    neighburs = 2
    algorithm = "ball_tree"
    metric = "euclidean"
    if ("params" in req):
        if("neighbours" in req["params"]): neighburs = req["params"]["neighbours"]
        if("algorithm" in req["params"]): algorithm = req["params"]["algorithm"]
        if("metric" in req["params"]): metric = req["params"]["metric"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
        if (user_id == None or project_id == None or filename == None):  return apierrors.NoData()
    else:
        return apierrors.NoData();

    fullPath = user_id + "/" + project_id + "/" + filename
    dataset = read_file(fullPath)
    rawX = pd.read_csv(StringIO(dataset.decode('utf-8')))
    X = preProcess(dataset=rawX)

    print(X)
    nbrs = NearestNeighbors(n_neighbors=neighburs, algorithm=algorithm, metric=metric).fit(X)
    s = pickle.dumps(nbrs)
    write_file(user_id, project_id, "pickle.pkl", s)
    distances, indices = nbrs.kneighbors(X)
    print(pd.DataFrame(X).describe())
    print(pd.DataFrame(indices).describe())
    data = rawX.to_json()
    indexes = pd.DataFrame(indices).to_json()

    return '{ "data": ' + data + ', "indexes": '+indexes+'}'